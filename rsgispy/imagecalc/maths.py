"""
The imagecalc.maths module provides muParser-based band and image math
calculation functions using ``pymu_eval`` and ``rasterio``.
"""

from contextlib import ExitStack
import math
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pymu_eval
import rasterio
from rasterio.windows import Window

import rsgispy
from rsgispy.imageprocess import (
    BandDefn,
    cast_result_block,
    compute_spatial_intersection,
    ensure_file_exists,
    get_default_block_size,
    iter_dataset_windows,
    iter_image_blocks,
    normalise_band_defs,
    open_input_datasets,
    open_output_dataset,
    process_blocks,
)

__all__ = [
    "BandDefn",
    "band_maths",
    "band_math",
    "image_math",
    "image_band_math",
    "calc_prop_true_exp",
    "calc_area_under_curve",
]


def _create_parser(exp: str, band_names: Sequence[str]) -> pymu_eval.Parser:
    """
    Create and validate a pymu_eval.Parser instance for the given expression and bands.

    :param exp: Mathematical expression string (muParser syntax).
    :param band_names: Variable names corresponding to the input bands.
    :return: Configured pymu_eval.Parser instance.
    """
    if not exp or not isinstance(exp, str) or not exp.strip():
        raise rsgispy.RSGISPyException(
            "Expression 'exp' must be a non-empty string."
        )

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
    :param band_defs: Sequence of :class:`rsgispy.imagecalc.maths.BandDefn`
                      objects defining the inputs.
    :param exp_band_name: Optional bool specifying whether the output band description
                          should be set to the expression string (Default = False).
    :param output_exists: Optional bool specifying whether the output image already exists
                          and should be updated in place rather than overwritten (Default = False).
    :param use_pbar: Optional bool specifying whether to display a tqdm progress bar
                     during processing (Default = True).
    :param creation_opts: Optional mapping of GDAL creation options (e.g.,
                          ``{'COMPRESS': 'DEFLATE', 'TILED': 'YES'}``) overriding or
                          supplementing any options set via :mod:`rsgispy.imageutils.envopts`.

    .. code:: python

        import rsgispy
        import rsgispy.imagecalc.maths

        band_defns = [
            rsgispy.imagecalc.maths.BandDefn(band_name='b1', input_img='img.kea', img_band=1),
            rsgispy.imagecalc.maths.BandDefn(band_name='b2', input_img='img.kea', img_band=2),
        ]
        rsgispy.imagecalc.maths.band_maths(
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
    ensure_file_exists(input_img)

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
    ensure_file_exists(input_img)

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


def calc_prop_true_exp(
    exp: str,
    band_defs: Sequence[BandDefn],
    in_vld_img: Optional[str] = None,
    use_pbar: bool = True,
) -> float:
    """
    Calculate the proportion of pixels where ``exp`` evaluates to ``1``.

    :param exp: muParser expression over ``band_defs`` (``floor(result) == 1`` is true).
    :param band_defs: Sequence of :class:`rsgispy.imagecalc.maths.BandDefn` objects.
    :param in_vld_img: Optional valid-area mask image path (pixels == 1 in band 1).
    :param use_pbar: If True, display a tqdm progress bar (Default = True).
    :return: Proportion of true pixels in ``[0.0, 1.0]``.
    """
    # pylint: disable=too-many-locals
    norm_defs = normalise_band_defs(band_defs)
    parser = _create_parser(exp, [b.band_name for b in norm_defs if b.band_name])

    if in_vld_img is not None:
        ensure_file_exists(in_vld_img, label="Valid image")

    true_count = 0
    total_count = 0
    with ExitStack() as stack:
        datasets = open_input_datasets(stack, norm_defs)
        if in_vld_img is not None and in_vld_img not in datasets:
            datasets[in_vld_img] = stack.enter_context(
                rasterio.open(in_vld_img, "r")
            )
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
                    is_one = (
                        math.floor(float(res)) == 1
                        if np.isfinite(res)
                        else False
                    )
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


def calc_area_under_curve(
    input_img: str,
    band_widths: Sequence[float],
    output_img: str,
    gdalformat: str = "GTIFF",
    datatype: int = rsgispy.TYPE_32FLOAT,
    use_pbar: bool = True,
    creation_opts: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Calculate the area under the curve defined by the input image bands and
    their corresponding ``band_widths`` (e.g., spectral or temporal widths).

    For each pixel, the output equals ``sum(band_i * band_widths[i])``.

    :param input_img: Path to the input image file.
    :param band_widths: Sequence of float widths matching the number of bands
                        in ``input_img``.
    :param output_img: Path to the output single-band image file.
    :param gdalformat: GDAL/rasterio driver format for the output file
                       (Default = ``'GTIFF'``).
    :param datatype: Output data type constant from ``rsgispy.TYPE_*``
                     (Default = ``rsgispy.TYPE_32FLOAT``).
    :param use_pbar: If True, display a tqdm progress bar (Default = True).
    :param creation_opts: Optional mapping of GDAL creation options.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    if band_widths is None:
        raise rsgispy.RSGISPyException("Input band_widths list must not be None.")
    ensure_file_exists(input_img)

    out_rio_dtype = rsgispy.get_rasterio_datatype(datatype)
    out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(datatype))
    widths_arr = np.asarray(band_widths, dtype=np.float64)
    if widths_arr.ndim != 1:
        raise rsgispy.RSGISPyException("Input band_widths must be a 1D sequence.")

    with ExitStack() as stack:
        try:
            src = stack.enter_context(rasterio.open(input_img, "r"))
        except Exception as err:
            raise rsgispy.RSGISPyException(
                f"Could not open input image '{input_img}': {err}"
            ) from err

        if src.count != widths_arr.size:
            raise rsgispy.RSGISPyException(
                f"Length of band_widths ({widths_arr.size}) must be equal to "
                f"the number of bands in the input image (N Bands: {src.count})."
            )

        grid = compute_spatial_intersection({input_img: src})
        dst = open_output_dataset(
            stack=stack,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype=out_rio_dtype,
            grid=grid,
            count=1,
            creation_opts=creation_opts,
        )

        for out_win in iter_dataset_windows(
            grid.height, grid.width, use_pbar=use_pbar
        ):
            in_block = src.read(window=out_win, out_dtype=np.float64)
            area = np.tensordot(widths_arr, in_block, axes=([0], [0]))
            out_block = cast_result_block(
                area, int(out_win.height), int(out_win.width), out_np_dtype
            )
            dst.write(out_block, 1, window=out_win)
