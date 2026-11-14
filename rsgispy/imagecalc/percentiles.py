"""
The imagecalc.percentiles module provides band-level and pixel-level raster
percentile calculation functions using ``rasterio``.
"""

from contextlib import ExitStack
import math
from typing import Any, List, Mapping, Optional, Sequence

import numpy as np
import rasterio
from rasterio.windows import Window

import rsgispy
from rsgispy.imageprocess import (
    BandDefn,
    IntersectionGrid,
    compute_spatial_intersection,
    ensure_file_exists,
    iter_dataset_windows,
    iter_image_blocks,
    open_input_datasets,
    open_multi_image_datasets,
    open_output_dataset,
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
) -> List[np.ndarray]:
    """Read float32 blocks aligned to ``out_win`` for each image in ``img_paths``."""
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
            out_dtype=np.float32,
        )
        for path in img_paths
    ]


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

    .. code:: python

        import rsgispy.imagecalc.percentiles

        rsgispy.imagecalc.percentiles.calc_imgs_pxl_percentiles(
            input_imgs=["img1.tif", "img2.tif", "img3.tif"],
            percentiles=[10.0, 50.0, 90.0],
            output_img="pxl_pcts.tif",
            gdalformat="GTIFF",
            no_data_val=0.0,
        )
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

        for out_win in iter_dataset_windows(
            grid.height, grid.width, use_pbar=use_pbar
        ):
            img_blocks = _read_grid_blocks(datasets, img_paths, grid, out_win)
            pct_arr = compute_nanpercentiles(
                np.concatenate(img_blocks, axis=0),
                percentiles,
                no_data_val,
            )
            dst.write(
                pct_arr.astype(out_np_dtype, copy=False), window=out_win
            )


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

    .. code:: python

        import rsgispy.imagecalc.percentiles

        rsgispy.imagecalc.percentiles.calc_img_band_pxl_percentiles(
            input_imgs=["img1.tif", "img2.tif", "img3.tif"],
            percentiles=[10.0, 50.0, 90.0],
            output_img="band_pxl_pcts.tif",
            gdalformat="GTIFF",
            no_data_val=0.0,
        )
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

        for out_win in iter_dataset_windows(
            grid.height, grid.width, use_pbar=use_pbar
        ):
            img_blocks = _read_grid_blocks(datasets, img_paths, grid, out_win)
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
