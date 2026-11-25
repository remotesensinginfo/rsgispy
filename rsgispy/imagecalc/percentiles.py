"""
The imagecalc.percentiles module provides band-level and pixel-level raster
percentile calculation functions using ``rasterio``.
"""

from contextlib import ExitStack
import math
from typing import Any, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import rasterio
from rasterio.windows import Window

import rsgispy
from rsgispy.imageprocess import (
    BandDefn,
    IntersectionGrid,
    compute_spatial_intersection,
    ensure_file_exists,
    fits_dtype_range,
    iter_dataset_windows,
    iter_image_blocks,
    open_input_datasets,
    open_multi_image_datasets,
    open_output_dataset,
    pop_stats_on_close,
)
from rsgispy.tools.stats import (
    compute_nanpercentiles,
    validate_percentiles_0_100,
)

__all__ = [
    "calc_band_percentile",
    "calc_band_percentile_msk",
    "calc_imgs_pxl_percentiles",
    "calc_img_band_pxl_percentiles",
]


def _read_grid_blocks(
    datasets: Mapping[str, Any],
    img_paths: Sequence[str],
    grid: IntersectionGrid,
    out_win: Window,
    read_dtype: Any = np.float32,
) -> List[np.ndarray]:
    """Read ``read_dtype`` blocks aligned to ``out_win`` for each image in ``img_paths``."""
    col_off = int(out_win.col_off)
    row_off = int(out_win.row_off)
    win_w = int(out_win.width)
    win_h = int(out_win.height)
    return [
        datasets[path].read(
            window=Window(
                grid.offsets[path][0] + col_off,
                grid.offsets[path][1] + row_off,
                win_w,
                win_h,
            ),
            out_dtype=read_dtype,
        )
        for path in img_paths
    ]


def _pct_label(percentile: float) -> str:
    """Return the band name suffix for ``percentile`` (e.g. ``'p10'``, ``'p2.5'``)."""
    return f"p{float(percentile):g}"


def _resolve_pct_output(
    datasets: Mapping[str, Any],
    img_paths: Sequence[str],
    no_data_val: Optional[float],
) -> Tuple[str, np.dtype, Any]:
    """
    Return the output rasterio dtype, NumPy dtype and the dtype used to read
    the inputs, and check that ``no_data_val`` can be stored in the output.
    """
    rsgis_dt = rsgispy.get_rsgis_datatype(datasets[img_paths[0]].dtypes[0])
    out_rio_dtype = rsgispy.get_rasterio_datatype(rsgis_dt)
    out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(rsgis_dt))
    if no_data_val is not None:
        nd_is_nan = math.isnan(float(no_data_val))
        if (nd_is_nan and not np.issubdtype(out_np_dtype, np.floating)) or (
            not nd_is_nan and not fits_dtype_range(float(no_data_val), out_np_dtype)
        ):
            raise rsgispy.RSGISPyException(
                f"no_data_val ({no_data_val}) cannot be stored in the output data "
                f"type '{out_np_dtype.name}' (the data type of the first input "
                "image)."
            )
    # Read in float32 unless an input needs more precision (e.g. float64, int32).
    read_dtype = np.result_type(
        np.float32,
        *[np.dtype(dt) for path in img_paths for dt in datasets[path].dtypes],
    )
    return out_rio_dtype, out_np_dtype, read_dtype


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

    .. code:: python

        import rsgispy.imagecalc.percentiles

        pct_vals = rsgispy.imagecalc.percentiles.calc_band_percentile(
            input_img="img.tif",
            percentile=0.95,
            no_data_val=0.0,
        )
    """
    if percentile < 0.0 or percentile > 1.0:
        raise rsgispy.RSGISPyException("Percentile value must be between 0 - 1.")
    ensure_file_exists(input_img)

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

    .. code:: python

        import rsgispy.imagecalc.percentiles

        pcts = rsgispy.imagecalc.percentiles.calc_band_percentile_msk(
            input_img="img.tif",
            in_msk_img="mask.tif",
            msk_val=1,
            percentiles=[5.0, 50.0, 95.0],
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    validate_percentiles_0_100(percentiles)
    ensure_file_exists(input_img)
    ensure_file_exists(in_msk_img, label="Mask image")

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
            datasets[in_msk_img] = stack.enter_context(
                rasterio.open(in_msk_img, "r")
            )
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
            valid = np.isfinite(band_cols).all(axis=1)
            if no_data_val is not None and not math.isnan(float(no_data_val)):
                valid &= (band_cols != no_data_val).all(axis=1)
            band_cols = band_cols[valid]
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
    output_img: str,
    percentiles: Sequence[float],
    *,
    gdalformat: str = "GTIFF",
    no_data_val: Optional[float] = 0.0,
    calc_stats: bool = True,
    use_pbar: bool = True,
    creation_opts: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Calculate percentiles on a per-pixel basis across all bands of ``input_imgs``.

    The output image has ``len(percentiles)`` bands, named ``p<percentile>``
    (e.g. ``p10``, ``p50``, ``p2.5``), and matches the spatial intersection
    and data type of the first input image. The images are read and
    processed block by block (:func:`rsgispy.imageprocess.get_default_block_size`),
    so memory use does not depend on the image size. The percentiles match
    ``numpy.nanpercentile`` (linear interpolation) and are cast to the output
    data type (truncated for integer types).

    :param input_imgs: Sequence of input image paths (all bands are stacked).
    :param output_img: Path to the output image file.
    :param percentiles: Sequence of percentiles in ``[0, 100]``.
    :param gdalformat: GDAL/rasterio driver format for the output file
                       (Default = ``'GTIFF'``).
    :param no_data_val: No-data value of the inputs, ignored when calculating
                        the percentiles. It is written to the pixels with no
                        valid values and set as the no-data value of the
                        output, so it must be representable in the output data
                        type (Default = 0). ``None`` ignores only ``NaN``.
    :param calc_stats: If ``True``, calculate image statistics and pyramids
                       for the output, ignoring ``no_data_val``
                       (Default = ``True``).
    :param use_pbar: If True, display a tqdm progress bar (Default = True).
    :param creation_opts: Optional mapping of GDAL creation options.

    .. code:: python

        import rsgispy.imagecalc.percentiles

        rsgispy.imagecalc.percentiles.calc_imgs_pxl_percentiles(
            input_imgs=["img1.tif", "img2.tif", "img3.tif"],
            output_img="pxl_pcts.tif",
            percentiles=[10.0, 50.0, 90.0],
            gdalformat="GTIFF",
            no_data_val=0.0,
        )
    """
    # pylint: disable=too-many-arguments,too-many-locals
    validate_percentiles_0_100(percentiles)
    pcts = [float(pct) for pct in np.atleast_1d(np.asarray(percentiles, dtype=np.float64))]

    with ExitStack() as stack:
        datasets, img_paths = open_multi_image_datasets(stack, input_imgs)
        grid = compute_spatial_intersection(datasets)
        out_rio_dtype, out_np_dtype, read_dtype = _resolve_pct_output(
            datasets, img_paths, no_data_val
        )

        dst = open_output_dataset(
            stack=stack,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype=out_rio_dtype,
            grid=grid,
            count=len(pcts),
            creation_opts=creation_opts,
        )
        if no_data_val is not None:
            dst.nodata = no_data_val
        pop_stats_on_close(
            dst, calc_stats, no_data_val=no_data_val, use_pbar=use_pbar
        )
        for band_idx, pct in enumerate(pcts, start=1):
            dst.set_band_description(band_idx, _pct_label(pct))

        for out_win in iter_dataset_windows(
            grid.height, grid.width, use_pbar=use_pbar
        ):
            img_blocks = _read_grid_blocks(
                datasets, img_paths, grid, out_win, read_dtype=read_dtype
            )
            pct_arr = compute_nanpercentiles(
                np.concatenate(img_blocks, axis=0),
                pcts,
                no_data_val,
            )
            dst.write(
                pct_arr.astype(out_np_dtype, copy=False), window=out_win
            )


def calc_img_band_pxl_percentiles(
    input_imgs: Sequence[str],
    output_img: str,
    percentiles: Sequence[float],
    *,
    gdalformat: str = "GTIFF",
    no_data_val: Optional[float] = 0.0,
    calc_stats: bool = True,
    use_pbar: bool = True,
    creation_opts: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Calculate per-pixel percentiles across ``input_imgs`` on a per-band basis.

    All input images must have the same band count ``n_bands``. The output
    image contains ``n_bands * len(percentiles)`` bands ordered as band 1
    percentiles, band 2 percentiles, ..., band ``n`` percentiles. The output
    bands are named ``<band name>_p<percentile>`` (e.g. ``Blue_p50``), using
    the band names of the first input image, or ``b<band>_p<percentile>``
    (e.g. ``b1_p50``) for bands without a name. The output matches the
    spatial intersection and data type of the first input image. The images
    are read and processed block by block
    (:func:`rsgispy.imageprocess.get_default_block_size`), so memory use does
    not depend on the image size. The percentiles match
    ``numpy.nanpercentile`` (linear interpolation) and are cast to the output
    data type (truncated for integer types).

    :param input_imgs: Sequence of input image paths with identical band counts.
    :param output_img: Path to the output image file.
    :param percentiles: Sequence of percentiles in ``[0, 100]``.
    :param gdalformat: GDAL/rasterio driver format for the output file
                       (Default = ``'GTIFF'``).
    :param no_data_val: No-data value of the inputs, ignored when calculating
                        the percentiles. It is written to the pixels with no
                        valid values and set as the no-data value of the
                        output, so it must be representable in the output data
                        type (Default = 0). ``None`` ignores only ``NaN``.
    :param calc_stats: If ``True``, calculate image statistics and pyramids
                       for the output, ignoring ``no_data_val``
                       (Default = ``True``).
    :param use_pbar: If True, display a tqdm progress bar (Default = True).
    :param creation_opts: Optional mapping of GDAL creation options.

    .. code:: python

        import rsgispy.imagecalc.percentiles

        rsgispy.imagecalc.percentiles.calc_img_band_pxl_percentiles(
            input_imgs=["img1.tif", "img2.tif", "img3.tif"],
            output_img="band_pxl_pcts.tif",
            percentiles=[10.0, 50.0, 90.0],
            gdalformat="GTIFF",
            no_data_val=0.0,
        )
    """
    # pylint: disable=too-many-arguments,too-many-locals
    validate_percentiles_0_100(percentiles)
    pcts = [float(pct) for pct in np.atleast_1d(np.asarray(percentiles, dtype=np.float64))]

    with ExitStack() as stack:
        datasets, img_paths = open_multi_image_datasets(stack, input_imgs)
        n_bands = datasets[img_paths[0]].count
        for path in img_paths[1:]:
            if datasets[path].count != n_bands:
                raise rsgispy.RSGISPyException(
                    "Input images have a different number of bands."
                )

        grid = compute_spatial_intersection(datasets)
        out_rio_dtype, out_np_dtype, read_dtype = _resolve_pct_output(
            datasets, img_paths, no_data_val
        )
        in_names = datasets[img_paths[0]].descriptions

        dst = open_output_dataset(
            stack=stack,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype=out_rio_dtype,
            grid=grid,
            count=n_bands * len(pcts),
            creation_opts=creation_opts,
        )
        if no_data_val is not None:
            dst.nodata = no_data_val
        pop_stats_on_close(
            dst, calc_stats, no_data_val=no_data_val, use_pbar=use_pbar
        )
        out_band = 1
        for b_idx in range(n_bands):
            base_name = in_names[b_idx] if in_names[b_idx] else f"b{b_idx + 1}"
            for pct in pcts:
                dst.set_band_description(out_band, f"{base_name}_{_pct_label(pct)}")
                out_band += 1

        for out_win in iter_dataset_windows(
            grid.height, grid.width, use_pbar=use_pbar
        ):
            img_blocks = _read_grid_blocks(
                datasets, img_paths, grid, out_win, read_dtype=read_dtype
            )
            out_percent_arrs = [
                compute_nanpercentiles(
                    np.stack([blk[b_idx] for blk in img_blocks], axis=0),
                    pcts,
                    no_data_val,
                )
                for b_idx in range(n_bands)
            ]
            out_arr = np.concatenate(out_percent_arrs, axis=0).astype(
                out_np_dtype, copy=False
            )
            dst.write(out_arr, window=out_win)
