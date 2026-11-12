"""
The imagecalc module contains functions for performing calculations on raster images.
"""

from contextlib import ExitStack
import math
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pymu_eval
import rasterio
from rasterio.windows import Window

import rsgispy
from rsgispy.imageprocess import (
    BandDefn,
    cast_result_block,
    compute_spatial_intersection,
    get_default_block_size,
    iter_image_blocks,
    iter_single_image_band_blocks,
    normalise_band_defs,
    open_input_datasets,
    open_multi_image_datasets,
    open_output_dataset,
    process_blocks,
)
from rsgispy.tools.stats import (
    compute_nanpercentiles,
    validate_percentiles_0_100,
)

__all__ = [
    "BandDefn",
    "band_maths",
    "band_math",
    "image_math",
    "image_band_math",
    "calc_basic_img_band_stats",
    "calc_band_percentile",
    "calc_band_percentile_msk",
    "calc_imgs_pxl_percentiles",
    "calc_img_band_pxl_percentiles",
    "calc_prop_true_exp",
    "count_pxls_of_val",
    "get_unique_values",
]


def _create_parser(exp: str, band_names: Sequence[str]) -> pymu_eval.Parser:
    """
    Create and validate a pymu_eval.Parser instance for the given expression and bands.

    :param exp: Mathematical expression string (muParser syntax).
    :param band_names: Variable names corresponding to the input bands.
    :return: Configured pymu_eval.Parser instance.
    """
    if not exp or not isinstance(exp, str) or not exp.strip():
        raise rsgispy.RSGISPyException("Expression 'exp' must be a non-empty string.")

    parser = pymu_eval.Parser()
    band_name_set = set(band_names)
    if "pi" not in band_name_set:
        parser.define_const("pi", pymu_eval.MathImpl.CONST_PI)
    if "e" not in band_name_set:
        parser.define_const("e", pymu_eval.MathImpl.CONST_E)

    dummy = np.zeros((1, 1), dtype=np.float64)
    for name in band_names:
        try:
            parser.define_var(name, dummy)
        except pymu_eval.ParserError as err:
            raise rsgispy.RSGISPyException(
                f"Invalid band variable name '{name}': {err}"
            ) from err

    try:
        parser.set_expr(exp)
        with np.errstate(all="ignore"):
            parser.eval()
    except pymu_eval.ParserError as err:
        raise rsgispy.RSGISPyException(
            f"Error parsing or evaluating expression '{exp}': {err}"
        ) from err

    return parser


def band_maths(
    output_img: str,
    exp: str,
    gdalformat: str,
    datatype: int,
    band_defs: Sequence[BandDefn],
    exp_band_name: bool = False,
    output_exists: bool = False,
    use_pbar: bool = True,
    creation_opts: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Performs band math calculation using pymu-eval and rasterio.

    The syntax for the expression follows the muParser library syntax
    (supported by ``pymu_eval``).

    :param output_img: Path to the output image file.
    :param exp: String containing the muParser expression to evaluate over the bands.
    :param gdalformat: GDAL/rasterio driver format for the output file (e.g., 'GTIFF', 'KEA').
    :param datatype: Output data type constant from ``rsgispy.TYPE_*`` (or ``rsgislib.TYPE_*``).
    :param band_defs: Sequence of :class:`rsgispy.imagecalc.BandDefn` objects defining the inputs.
    :param exp_band_name: Optional bool specifying whether the output band description
                          should be set to the expression string (Default = False).
    :param output_exists: Optional bool specifying whether the output image already exists
                          and should be updated in place rather than overwritten (Default = False).
    :param use_pbar: Optional bool specifying whether to display a tqdm progress bar
                     during processing (Default = True).
    :param creation_opts: Optional mapping of GDAL creation options (e.g.,
                          ``{'COMPRESS': 'DEFLATE', 'TILED': 'YES'}``) overriding or
                          supplementing any options set via :mod:`rsgispy.imageutils`.

    .. code:: python

        import rsgispy
        import rsgispy.imagecalc

        band_defns = list()
        band_defns.append(
            rsgispy.imagecalc.BandDefn(band_name='b1', input_img='img.kea', img_band=1)
        )
        band_defns.append(
            rsgispy.imagecalc.BandDefn(band_name='b2', input_img='img.kea', img_band=2)
        )
        rsgispy.imagecalc.band_maths(
            output_img='out.tif',
            exp='b1*b2',
            gdalformat='GTIFF',
            datatype=rsgispy.TYPE_32FLOAT,
            band_defs=band_defns,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    out_rio_dtype = rsgispy.get_rasterio_datatype(datatype)
    out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(datatype))
    norm_defs = normalise_band_defs(band_defs)
    parser = _create_parser(exp, [b.band_name for b in norm_defs if b.band_name])

    with ExitStack() as stack:
        datasets = open_input_datasets(stack, norm_defs)
        grid = compute_spatial_intersection(datasets)
        dst = open_output_dataset(
            stack=stack,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype=out_rio_dtype,
            grid=grid,
            count=1,
            output_exists=output_exists,
            creation_opts=creation_opts,
        )

        if exp_band_name:
            dst.set_band_description(1, exp)

        process_blocks(
            dst=dst,
            datasets=datasets,
            norm_defs=norm_defs,
            grid=grid,
            block_func=parser.eval,
            out_np_dtype=out_np_dtype,
            use_pbar=use_pbar,
        )


# Alias for backwards compatibility with rsgislib.imagecalc.band_math
band_math = band_maths


def calc_basic_img_band_stats(
    input_img: str,
    no_data_val: Optional[float] = None,
    use_no_data: bool = True,
    use_pbar: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the minimum, maximum, mean, and standard deviation values
    for each band in the input image using block-wise iteration.

    :param input_img: Path to the input image file.
    :param no_data_val: Optional no-data value. If ``None`` and ``use_no_data`` is True,
                        reads the no-data value from the image header.
    :param use_no_data: Whether to exclude the no-data value from statistics (Default = True).
    :param use_pbar: Whether to display a tqdm progress bar (Default = True).
    :return: Tuple of ``(min_vals, max_vals, mean_vals, stddev_vals)`` NumPy arrays.
    """
    # pylint: disable=too-many-locals,too-many-branches
    if not os.path.exists(input_img):
        raise rsgispy.RSGISPyException(
            f"Input image file does not exist: '{input_img}'."
        )

    try:
        with rasterio.open(input_img, "r") as src:
            hdr_nodata = src.nodata
            n_bands = src.count
            rsgis_dt = rsgispy.get_rsgis_datatype(src.dtypes[0])
    except Exception as err:
        raise rsgispy.RSGISPyException(
            f"Could not open input image '{input_img}': {err}"
        ) from err

    if use_no_data and no_data_val is None:
        no_data_val = hdr_nodata
        if no_data_val is None:
            raise rsgispy.RSGISPyException(
                "A no data value needs to be specified either passed to the "
                "function or read from the input image header."
            )

    np_dt = rsgispy.get_numpy_datatype(rsgis_dt)
    min_vals = np.zeros(n_bands, dtype=np_dt)
    max_vals = np.zeros(n_bands, dtype=np_dt)
    sum_vals = np.zeros(n_bands, dtype=np.float64)
    n_vals = np.zeros(n_bands, dtype=np.int64)
    first_arr = np.ones(n_bands, dtype=bool)

    for _, block_arrays in iter_single_image_band_blocks(
        input_img=input_img, use_pbar=use_pbar
    ):
        for idx, band_arr in enumerate(block_arrays):
            data = band_arr.ravel()
            if use_no_data and no_data_val is not None:
                data = data[data != no_data_val]
            data = data[np.isfinite(data)]
            if data.size > 0:
                d_min = data.min()
                d_max = data.max()
                if first_arr[idx]:
                    min_vals[idx] = d_min
                    max_vals[idx] = d_max
                    first_arr[idx] = False
                else:
                    if d_min < min_vals[idx]:
                        min_vals[idx] = d_min
                    if d_max > max_vals[idx]:
                        max_vals[idx] = d_max
                sum_vals[idx] += float(np.sum(data))
                n_vals[idx] += data.size

    mean_vals = np.where(n_vals > 0, sum_vals / np.maximum(n_vals, 1), 0.0)
    sq_diff_vals = np.zeros(n_bands, dtype=np.float64)

    for _, block_arrays in iter_single_image_band_blocks(
        input_img=input_img, use_pbar=use_pbar
    ):
        for idx, band_arr in enumerate(block_arrays):
            data = band_arr.ravel()
            if use_no_data and no_data_val is not None:
                data = data[data != no_data_val]
            data = data[np.isfinite(data)]
            if data.size > 0:
                sq_diff_vals[idx] += float(
                    np.sum(np.power(data - mean_vals[idx], 2))
                )

    stddev_vals = np.where(
        n_vals > 0, np.sqrt(sq_diff_vals / np.maximum(n_vals, 1)), 0.0
    )

    return min_vals, max_vals, mean_vals, stddev_vals


def image_math(
    input_img: str,
    output_img: str,
    exp: str,
    gdalformat: str,
    datatype: int,
    exp_band_name: bool = False,
    output_exists: bool = False,
    use_pbar: bool = True,
    creation_opts: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Perform image math calculations using ``pymu_eval`` and ``rasterio``.

    Produces an output image with the same number of bands as ``input_img``,
    applying the same muParser expression (using variable ``b1``) independently
    to each band.

    :param input_img: Path to the input image file.
    :param output_img: Path to the output image file.
    :param exp: muParser mathematical expression referencing ``b1``.
    :param gdalformat: GDAL/rasterio driver format for the output file (e.g., ``'GTIFF'``).
    :param datatype: Output data type constant from ``rsgispy.TYPE_*``.
    :param exp_band_name: If True, set each output band description to ``exp`` (Default = False).
    :param output_exists: If True and ``output_img`` exists, update it in place (Default = False).
    :param use_pbar: If True, display a tqdm progress bar during processing (Default = True).
    :param creation_opts: Optional mapping of GDAL creation options.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    if not os.path.exists(input_img):
        raise rsgispy.RSGISPyException(
            f"Input image file does not exist: '{input_img}'."
        )

    out_rio_dtype = rsgispy.get_rasterio_datatype(datatype)
    out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(datatype))
    parser = _create_parser(exp, ["b1"])

    with ExitStack() as stack:
        try:
            src = stack.enter_context(rasterio.open(input_img, "r"))
        except Exception as err:
            raise rsgispy.RSGISPyException(
                f"Could not open input image '{input_img}': {err}"
            ) from err

        n_bands = src.count
        grid = compute_spatial_intersection({input_img: src})
        dst = open_output_dataset(
            stack=stack,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype=out_rio_dtype,
            grid=grid,
            count=n_bands,
            output_exists=output_exists,
            creation_opts=creation_opts,
        )

        if exp_band_name:
            for band_idx in range(1, n_bands + 1):
                dst.set_band_description(band_idx, exp)

        block_size = get_default_block_size()
        tqdm_cls = rsgispy.get_tqdm()
        total_blocks = math.ceil(grid.height / block_size) * math.ceil(
            grid.width / block_size
        )

        with tqdm_cls(total=total_blocks, disable=not use_pbar) as pbar:
            for row_off in range(0, grid.height, block_size):
                win_h = min(block_size, grid.height - row_off)
                for col_off in range(0, grid.width, block_size):
                    win_w = min(block_size, grid.width - col_off)
                    out_win = Window(col_off, row_off, win_w, win_h)
                    in_block = src.read(window=out_win, out_dtype=np.float64)
                    out_bands = [
                        cast_result_block(
                            parser.eval({"b1": in_block[band_idx]}),
                            win_h,
                            win_w,
                            out_np_dtype,
                        )
                        for band_idx in range(n_bands)
                    ]
                    dst.write(np.stack(out_bands, axis=0), window=out_win)
                    pbar.update(1)


def image_band_math(
    input_img: str,
    output_img: str,
    exp: str,
    gdalformat: str,
    datatype: int,
    exp_band_name: bool = False,
    output_exists: bool = False,
    use_pbar: bool = True,
    creation_opts: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Perform image band math calculations using ``pymu_eval`` and ``rasterio``.

    Produces a single-band output image where input bands are bound to variables
    ``b1, b2, ..., bn`` (1-based).

    :param input_img: Path to the input image file.
    :param output_img: Path to the output image file.
    :param exp: muParser expression referencing input bands as ``b1``, ``b2``, etc.
    :param gdalformat: GDAL/rasterio driver format for the output file (e.g., ``'GTIFF'``).
    :param datatype: Output data type constant from ``rsgispy.TYPE_*``.
    :param exp_band_name: If True, set the output band description to ``exp`` (Default = False).
    :param output_exists: If True and ``output_img`` exists, update it in place (Default = False).
    :param use_pbar: If True, display a tqdm progress bar during processing (Default = True).
    :param creation_opts: Optional mapping of GDAL creation options.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    if not os.path.exists(input_img):
        raise rsgispy.RSGISPyException(
            f"Input image file does not exist: '{input_img}'."
        )

    try:
        with rasterio.open(input_img, "r") as src:
            n_bands = src.count
    except Exception as err:
        raise rsgispy.RSGISPyException(
            f"Could not open input image '{input_img}': {err}"
        ) from err

    band_defs = [
        BandDefn(band_name=f"b{idx + 1}", input_img=input_img, img_band=idx + 1)
        for idx in range(n_bands)
    ]
    band_maths(
        output_img=output_img,
        exp=exp,
        gdalformat=gdalformat,
        datatype=datatype,
        band_defs=band_defs,
        exp_band_name=exp_band_name,
        output_exists=output_exists,
        use_pbar=use_pbar,
        creation_opts=creation_opts,
    )


def calc_band_percentile(
    input_img: str,
    percentile: float,
    no_data_val: Optional[float] = None,
) -> List[float]:
    """
    Calculate a percentile value for each band of an input image.

    :param input_img: Path to the input image file.
    :param percentile: Percentile fraction between ``0.0`` and ``1.0`` inclusive.
    :param no_data_val: Optional no-data value to exclude (or ``None``).
    :return: List of float percentile values, one per band.
    """
    if percentile < 0.0 or percentile > 1.0:
        raise rsgispy.RSGISPyException("Percentile value must be between 0 - 1.")
    if not os.path.exists(input_img):
        raise rsgispy.RSGISPyException(
            f"Input image file does not exist: '{input_img}'."
        )

    pct_val = float(percentile) * 100.0
    out_vals: List[float] = []
    with rasterio.open(input_img, "r") as src:
        for band_idx in range(1, src.count + 1):
            arr = src.read(band_idx, out_dtype=np.float64).ravel()
            if no_data_val is not None:
                if isinstance(no_data_val, float) and math.isnan(no_data_val):
                    arr = arr[~np.isnan(arr)]
                else:
                    arr = arr[arr != no_data_val]
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                out_vals.append(float("nan"))
            else:
                out_vals.append(float(np.percentile(arr, pct_val)))

    return out_vals


def calc_band_percentile_msk(
    input_img: str,
    in_msk_img: str,
    msk_val: int,
    percentiles: Sequence[float],
    no_data_val: Optional[float] = None,
    use_pbar: bool = True,
) -> np.ndarray:
    """
    Calculate percentiles for all bands of ``input_img`` over pixels matching
    ``msk_val`` in ``in_msk_img``.

    :param input_img: Input image path on which percentiles are calculated.
    :param in_msk_img: Mask image path (band 1 is compared against ``msk_val``).
    :param msk_val: Pixel value within ``in_msk_img`` identifying valid pixels.
    :param percentiles: Sequence of percentiles in ``[0, 100]``.
    :param no_data_val: Optional no-data value to exclude across all bands.
    :param use_pbar: If True, display a tqdm progress bar (Default = True).
    :return: NumPy array of shape ``(len(percentiles), n_bands)`` (or ``(n_bands,)``
             if ``percentiles`` is a scalar-like single float passed to ``np.percentile``).
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    validate_percentiles_0_100(percentiles)
    if not os.path.exists(input_img):
        raise rsgispy.RSGISPyException(
            f"Input image file does not exist: '{input_img}'."
        )
    if not os.path.exists(in_msk_img):
        raise rsgispy.RSGISPyException(
            f"Mask image file does not exist: '{in_msk_img}'."
        )

    with rasterio.open(input_img, "r") as img_src:
        n_bands = img_src.count

    norm_defs = [
        BandDefn(band_name=f"b{idx + 1}", input_img=input_img, img_band=idx + 1)
        for idx in range(n_bands)
    ]

    collected: List[np.ndarray] = []
    with ExitStack() as stack:
        datasets = open_input_datasets(stack, norm_defs)
        if in_msk_img not in datasets:
            datasets[in_msk_img] = stack.enter_context(rasterio.open(in_msk_img, "r"))
        grid = compute_spatial_intersection(datasets)
        msk_ds = datasets[in_msk_img]
        base_col, base_row = grid.offsets[in_msk_img]

        for out_win, block_vars in iter_image_blocks(
            datasets=datasets,
            norm_defs=norm_defs,
            grid=grid,
            read_dtype=None,
            use_pbar=use_pbar,
        ):
            msk_win = Window(
                base_col + int(out_win.col_off),
                base_row + int(out_win.row_off),
                int(out_win.width),
                int(out_win.height),
            )
            msk_flat = msk_ds.read(1, window=msk_win).ravel()
            sel = msk_flat == msk_val
            if not np.any(sel):
                continue

            band_cols = np.column_stack(
                [block_vars[f"b{idx + 1}"].ravel()[sel] for idx in range(n_bands)]
            )
            if no_data_val is not None:
                band_cols = band_cols[(band_cols != no_data_val).all(axis=1)]
            if band_cols.shape[0] > 0:
                collected.append(band_cols)

    if not collected:
        raise rsgispy.RSGISPyException(
            "No valid pixels found within the specified mask."
        )

    pxl_vals = np.concatenate(collected, axis=0)
    return np.percentile(pxl_vals, percentiles, axis=0)


def calc_imgs_pxl_percentiles(
    input_imgs: Sequence[str],
    percentiles: Sequence[float],
    output_img: str,
    gdalformat: str,
    no_data_val: float = 0.0,
    use_pbar: bool = True,
    creation_opts: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Calculate percentiles on a per-pixel basis across all bands of ``input_imgs``.

    The output image has ``len(percentiles)`` bands and matches the spatial
    intersection and data type of the first input image.

    :param input_imgs: Sequence of input image paths (all bands are stacked).
    :param percentiles: Sequence of percentiles in ``[0, 100]``.
    :param output_img: Path to the output image file.
    :param gdalformat: GDAL/rasterio driver format for the output file.
    :param no_data_val: No-data value to ignore during percentile calculation (Default = 0).
    :param use_pbar: If True, display a tqdm progress bar (Default = True).
    :param creation_opts: Optional mapping of GDAL creation options.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    validate_percentiles_0_100(percentiles)

    with ExitStack() as stack:
        datasets, img_paths = open_multi_image_datasets(stack, input_imgs)
        grid = compute_spatial_intersection(datasets)
        rsgis_dt = rsgispy.get_rsgis_datatype(datasets[img_paths[0]].dtypes[0])
        out_rio_dtype = rsgispy.get_rasterio_datatype(rsgis_dt)
        out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(rsgis_dt))

        dst = open_output_dataset(
            stack=stack,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype=out_rio_dtype,
            grid=grid,
            count=len(percentiles),
            creation_opts=creation_opts,
        )

        block_size = get_default_block_size()
        tqdm_cls = rsgispy.get_tqdm()
        total_blocks = math.ceil(grid.height / block_size) * math.ceil(
            grid.width / block_size
        )

        with tqdm_cls(total=total_blocks, disable=not use_pbar) as pbar:
            for row_off in range(0, grid.height, block_size):
                win_h = min(block_size, grid.height - row_off)
                for col_off in range(0, grid.width, block_size):
                    win_w = min(block_size, grid.width - col_off)
                    out_win = Window(col_off, row_off, win_w, win_h)
                    img_blocks = [
                        datasets[path].read(
                            window=Window(
                                grid.offsets[path][0] + col_off,
                                grid.offsets[path][1] + row_off,
                                win_w,
                                win_h,
                            ),
                            out_dtype=np.float32,
                        )
                        for path in img_paths
                    ]
                    pct_arr = compute_nanpercentiles(
                        np.concatenate(img_blocks, axis=0), percentiles, no_data_val
                    )
                    dst.write(pct_arr.astype(out_np_dtype, copy=False), window=out_win)
                    pbar.update(1)


def calc_img_band_pxl_percentiles(
    input_imgs: Sequence[str],
    percentiles: Sequence[float],
    output_img: str,
    gdalformat: str,
    no_data_val: float = 0.0,
    use_pbar: bool = True,
    creation_opts: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Calculate per-pixel percentiles across ``input_imgs`` on a per-band basis.

    All input images must have the same band count ``n_bands``. The output
    image contains ``n_bands * len(percentiles)`` bands ordered as band 1
    percentiles, band 2 percentiles, ..., band ``n`` percentiles.

    :param input_imgs: Sequence of input image paths with identical band counts.
    :param percentiles: Sequence of percentiles in ``[0, 100]``.
    :param output_img: Path to the output image file.
    :param gdalformat: GDAL/rasterio driver format for the output file.
    :param no_data_val: No-data value to ignore during percentile calculation (Default = 0).
    :param use_pbar: If True, display a tqdm progress bar (Default = True).
    :param creation_opts: Optional mapping of GDAL creation options.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    validate_percentiles_0_100(percentiles)

    with ExitStack() as stack:
        datasets, img_paths = open_multi_image_datasets(stack, input_imgs)
        n_bands = datasets[img_paths[0]].count
        for path in img_paths[1:]:
            if datasets[path].count != n_bands:
                raise rsgispy.RSGISPyException(
                    "Input images have a different number of bands."
                )

        grid = compute_spatial_intersection(datasets)
        rsgis_dt = rsgispy.get_rsgis_datatype(datasets[img_paths[0]].dtypes[0])
        out_rio_dtype = rsgispy.get_rasterio_datatype(rsgis_dt)
        out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(rsgis_dt))

        dst = open_output_dataset(
            stack=stack,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype=out_rio_dtype,
            grid=grid,
            count=n_bands * len(percentiles),
            creation_opts=creation_opts,
        )

        block_size = get_default_block_size()
        tqdm_cls = rsgispy.get_tqdm()
        total_blocks = math.ceil(grid.height / block_size) * math.ceil(
            grid.width / block_size
        )

        with tqdm_cls(total=total_blocks, disable=not use_pbar) as pbar:
            for row_off in range(0, grid.height, block_size):
                win_h = min(block_size, grid.height - row_off)
                for col_off in range(0, grid.width, block_size):
                    win_w = min(block_size, grid.width - col_off)
                    out_win = Window(col_off, row_off, win_w, win_h)
                    img_blocks = [
                        datasets[path].read(
                            window=Window(
                                grid.offsets[path][0] + col_off,
                                grid.offsets[path][1] + row_off,
                                win_w,
                                win_h,
                            ),
                            out_dtype=np.float32,
                        )
                        for path in img_paths
                    ]
                    out_percent_arrs = [
                        compute_nanpercentiles(
                            np.stack([blk[b_idx] for blk in img_blocks], axis=0),
                            percentiles,
                            no_data_val,
                        )
                        for b_idx in range(n_bands)
                    ]
                    out_arr = np.concatenate(out_percent_arrs, axis=0).astype(
                        out_np_dtype, copy=False
                    )
                    dst.write(out_arr, window=out_win)
                    pbar.update(1)


def calc_prop_true_exp(
    exp: str,
    band_defs: Sequence[BandDefn],
    in_vld_img: Optional[str] = None,
    use_pbar: bool = True,
) -> float:
    """
    Calculate the proportion of pixels where ``exp`` evaluates to ``1``.

    :param exp: muParser expression over ``band_defs`` (``floor(result) == 1`` is true).
    :param band_defs: Sequence of :class:`rsgispy.imagecalc.BandDefn` objects.
    :param in_vld_img: Optional valid-area mask image path (pixels == 1 in band 1).
    :param use_pbar: If True, display a tqdm progress bar (Default = True).
    :return: Proportion of true pixels in ``[0.0, 1.0]``.
    """
    # pylint: disable=too-many-locals
    norm_defs = normalise_band_defs(band_defs)
    parser = _create_parser(exp, [b.band_name for b in norm_defs if b.band_name])

    if in_vld_img is not None and not os.path.exists(in_vld_img):
        raise rsgispy.RSGISPyException(
            f"Valid image file does not exist: '{in_vld_img}'."
        )

    true_count = 0
    total_count = 0
    with ExitStack() as stack:
        datasets = open_input_datasets(stack, norm_defs)
        if in_vld_img is not None and in_vld_img not in datasets:
            datasets[in_vld_img] = stack.enter_context(rasterio.open(in_vld_img, "r"))
        grid = compute_spatial_intersection(datasets)

        for out_win, block_vars in iter_image_blocks(
            datasets=datasets,
            norm_defs=norm_defs,
            grid=grid,
            use_pbar=use_pbar,
        ):
            res = parser.eval(block_vars)
            win_h = int(out_win.height)
            win_w = int(out_win.width)
            if np.ndim(res) == 0:
                with np.errstate(invalid="ignore"):
                    is_one = math.floor(float(res)) == 1 if np.isfinite(res) else False
                true_mask = np.full((win_h, win_w), is_one, dtype=bool)
            else:
                with np.errstate(invalid="ignore"):
                    true_mask = np.floor(np.asarray(res)) == 1

            if in_vld_img is not None:
                base_col, base_row = grid.offsets[in_vld_img]
                vld_win = Window(
                    base_col + int(out_win.col_off),
                    base_row + int(out_win.row_off),
                    win_w,
                    win_h,
                )
                vld_mask = datasets[in_vld_img].read(1, window=vld_win) == 1
                true_count += int(np.count_nonzero(true_mask & vld_mask))
                total_count += int(np.count_nonzero(vld_mask))
            else:
                true_count += int(np.count_nonzero(true_mask))
                total_count += int(true_mask.size)

    return float(true_count / total_count) if total_count > 0 else 0.0


def count_pxls_of_val(
    input_img: str,
    vals: Sequence[int],
    img_band: Optional[int] = None,
    use_pbar: bool = True,
) -> np.ndarray:
    """
    Count the number of pixels equal to each value in ``vals``.

    :param input_img: Path to the input image file.
    :param vals: Sequence of pixel values to count.
    :param img_band: 1-based band index, or ``None`` to count across all bands.
    :param use_pbar: If True, display a tqdm progress bar (Default = True).
    :return: 1D ``int64`` NumPy array of counts matching the order of ``vals``.
    """
    if vals is None:
        raise rsgispy.RSGISPyException("Input vals list must not be None.")
    if len(vals) == 0:
        raise rsgispy.RSGISPyException(
            "At least 1 value should be provided within the vals input variable."
        )
    if not os.path.exists(input_img):
        raise rsgispy.RSGISPyException(
            f"Input image file does not exist: '{input_img}'."
        )

    with rasterio.open(input_img, "r") as src:
        n_bands = src.count
    if img_band is not None and (img_band < 1 or img_band > n_bands):
        raise rsgispy.RSGISPyException(
            "The specified input image band is not within the input image."
        )

    out_vals = np.zeros(len(vals), dtype=np.int64)
    for _, block_arrays in iter_single_image_band_blocks(
        input_img=input_img,
        read_dtype=None,
        use_pbar=use_pbar,
    ):
        target_arrays = (
            [block_arrays[img_band - 1]] if img_band is not None else block_arrays
        )
        for band_arr in target_arrays:
            for idx, val in enumerate(vals):
                out_vals[idx] += int(np.count_nonzero(band_arr == val))

    return out_vals


def get_unique_values(
    input_img: str,
    img_band: int = 1,
) -> np.ndarray:
    """
    Find the unique pixel values within a single image band.

    Note: the entire image band is read into memory.

    :param input_img: Path to the input image file.
    :param img_band: 1-based band index to process (Default = 1).
    :return: 1D sorted NumPy array of unique values.
    """
    if not os.path.exists(input_img):
        raise rsgispy.RSGISPyException(
            f"Input image file does not exist: '{input_img}'."
        )

    try:
        with rasterio.open(input_img, "r") as src:
            if img_band < 1 or img_band > src.count:
                raise rsgispy.RSGISPyException(
                    f"Could not open image band ({img_band})."
                )
            vals_arr = src.read(img_band)
    except rsgispy.RSGISPyException:
        raise
    except Exception as err:
        raise rsgispy.RSGISPyException(
            f"Could not open input image '{input_img}': {err}"
        ) from err

    return np.unique(vals_arr)
