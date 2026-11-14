"""
Internal shared DEM windowing, Horn 3x3 derivative, and block-processing
helpers for :mod:`rsgispy.elevation`.
"""

from contextlib import ExitStack
import math
from typing import Any, Callable, Optional, Tuple

import numpy as np
import rasterio

import rsgispy
from rsgispy.imageprocess import (
    create_output_from_dataset,
    ensure_file_exists,
    iter_dataset_windows,
    read_window_with_halo,
)


def validate_sun_angles(azimuth: float, zenith: float) -> None:
    """Validate solar/view azimuth in ``[0, 360]`` and zenith in ``[0, 90]``."""
    if zenith < 0.0 or zenith > 90.0:
        raise rsgispy.RSGISPyException(
            "The zenith angle should be between 0 and 90 degrees."
        )
    if azimuth < 0.0 or azimuth > 360.0:
        raise rsgispy.RSGISPyException(
            "The azimuth angle should be between 0 and 360 degrees."
        )


def validate_slope_unit(unit: str) -> str:
    """Validate and normalise slope output unit ('degrees' or 'radians')."""
    unit_norm = unit.strip().lower()
    if unit_norm not in ("degrees", "radians"):
        raise rsgispy.RSGISPyException(
            "The unit option needs to be specified as either 'degrees' or 'radians'."
        )
    return unit_norm


def valid_mask(arr: np.ndarray, no_data_val: Optional[float]) -> np.ndarray:
    """Return boolean mask of finite values excluding ``no_data_val``."""
    mask = np.isfinite(arr)
    if no_data_val is not None and not (
        isinstance(no_data_val, (float, np.floating))
        and math.isnan(float(no_data_val))
    ):
        mask &= arr != no_data_val
    return mask


def prepare_3x3_horn_diffs(
    dem_halo: np.ndarray,
    no_data_val: Optional[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build cleaned 3x3 windows (replacing isolated nodata values with window
    means when ``n_vals > 1``) and return ``(dx_w_minus_e, dy_s_minus_n, n_vals)``.
    """
    win = np.lib.stride_tricks.sliding_window_view(dem_halo, (3, 3))
    valid = valid_mask(win, no_data_val)
    n_vals = valid.sum(axis=(2, 3))

    has_nodata = n_vals < 9
    if np.any(has_nodata):
        sum_vals = np.where(valid, win, 0.0).sum(axis=(2, 3))
        mean_vals = np.divide(
            sum_vals,
            n_vals,
            out=np.zeros_like(sum_vals, dtype=np.float64),
            where=n_vals > 0,
        )
        fill_mask = (has_nodata & (n_vals > 1))[..., None, None]
        w_clean = np.where(valid | ~fill_mask, win, mean_vals[..., None, None])
    else:
        w_clean = win

    dx_w_minus_e = (
        w_clean[..., 0, 0] + 2.0 * w_clean[..., 1, 0] + w_clean[..., 2, 0]
    ) - (w_clean[..., 0, 2] + 2.0 * w_clean[..., 1, 2] + w_clean[..., 2, 2])
    dy_s_minus_n = (
        w_clean[..., 2, 0] + 2.0 * w_clean[..., 2, 1] + w_clean[..., 2, 2]
    ) - (w_clean[..., 0, 0] + 2.0 * w_clean[..., 0, 1] + w_clean[..., 0, 2])
    return dx_w_minus_e, dy_s_minus_n, n_vals


def calc_slope_aspect_from_diffs(
    dx_w_minus_e: np.ndarray,
    dy_s_minus_n: np.ndarray,
    n_vals: np.ndarray,
    ew_res: Any,
    ns_res: Any,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ``(slope_rad, aspect_deg, flat_mask)`` from 3x3 Horn differences.

    Flat areas and cells with ``n_vals <= 1`` have ``slope_rad = 0`` and
    ``aspect_deg = NaN``.
    """
    dx_s = dx_w_minus_e / ew_res
    dy_s = dy_s_minus_n / ns_res
    slope_rad = np.where(
        n_vals > 1, np.arctan(np.sqrt((dx_s * dx_s) + (dy_s * dy_s)) / 8.0), 0.0
    )

    flat_mask = (dx_s == 0.0) & (dy_s == 0.0)
    valid_asp = (n_vals > 1) & ~flat_mask
    aspect_deg = np.full(dx_s.shape, np.nan, dtype=np.float64)
    if np.any(valid_asp):
        raw_deg = np.degrees(np.arctan2(dx_s[valid_asp], dy_s[valid_asp]))
        wrapped = np.mod(raw_deg, 360.0)
        wrapped[wrapped == 360.0] = 0.0
        aspect_deg[valid_asp] = wrapped
    return slope_rad, aspect_deg, flat_mask


def process_single_dem_3x3(
    input_img: str,
    output_img: str,
    gdalformat: str,
    out_rio_dtype: str,
    block_fn: Callable[[np.ndarray, np.ndarray, np.ndarray, float, float], np.ndarray],
    use_pbar: bool = True,
) -> None:
    """Run a 3x3 windowed calculation block-by-block over a single DEM raster."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    ensure_file_exists(input_img)
    with ExitStack() as stack:
        src = stack.enter_context(rasterio.open(input_img, "r"))
        ew_res = abs(float(src.transform.a))
        ns_res = abs(float(src.transform.e))
        no_data_val = src.nodata

        dst = create_output_from_dataset(
            stack=stack,
            src=src,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype=out_rio_dtype,
            count=1,
        )

        for out_win in iter_dataset_windows(
            src.height, src.width, use_pbar=use_pbar
        ):
            dem_halo = read_window_with_halo(src, 1, out_win, pad=1)
            dx, dy, n_vals = prepare_3x3_horn_diffs(dem_halo, no_data_val)
            out_arr = block_fn(dx, dy, n_vals, ew_res, ns_res)
            dst.write(out_arr, 1, window=out_win)


def process_pxl_res_dem_3x3(
    in_dem_img: str,
    in_pxl_res_img: str,
    output_img: str,
    gdalformat: str,
    out_rio_dtype: str,
    block_fn: Callable[
        [np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray
    ],
    use_pbar: bool = True,
) -> None:
    """Run a 3x3 windowed calculation using per-pixel E-W/N-S resolution bands."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    ensure_file_exists(in_dem_img, "Input DEM image")
    ensure_file_exists(in_pxl_res_img, "Input pixel resolution image")

    with ExitStack() as stack:
        dem_src = stack.enter_context(rasterio.open(in_dem_img, "r"))
        res_src = stack.enter_context(rasterio.open(in_pxl_res_img, "r"))
        if res_src.count < 2:
            raise rsgispy.RSGISPyException(
                "in_pxl_res_img must contain at least 2 bands (EW and NS resolution)."
            )
        if dem_src.width != res_src.width or dem_src.height != res_src.height:
            raise rsgispy.RSGISPyException(
                "Input DEM and pixel resolution images must have matching dimensions."
            )

        no_data_val = dem_src.nodata
        dst = create_output_from_dataset(
            stack=stack,
            src=dem_src,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype=out_rio_dtype,
            count=1,
        )

        for out_win in iter_dataset_windows(
            dem_src.height, dem_src.width, use_pbar=use_pbar
        ):
            dem_halo = read_window_with_halo(dem_src, 1, out_win, pad=1)
            dx, dy, n_vals = prepare_3x3_horn_diffs(dem_halo, no_data_val)
            ew_res = res_src.read(1, window=out_win, out_dtype=np.float64)
            ns_res = np.abs(
                res_src.read(2, window=out_win, out_dtype=np.float64)
            )
            out_arr = block_fn(dx, dy, n_vals, ew_res, ns_res)
            dst.write(out_arr, 1, window=out_win)
