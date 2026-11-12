"""
The elevation module contains functions for performing terrain and elevation analysis
on digital elevation models (DEMs) using ``rasterio`` and ``numpy``.
"""

# pylint: disable=too-many-lines

from contextlib import ExitStack
import heapq
import math
import os
import tempfile
from typing import Any, Callable, Optional, Tuple

import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.windows import Window

import rsgispy
from rsgispy.imageprocess import (
    IntersectionGrid,
    get_default_block_size,
    open_output_dataset,
)
import rsgispy.imageutils
import rsgispy.tools.filetools
import rsgispy.tools.utils

__all__ = [
    "slope",
    "slope_pxl_res_img",
    "aspect",
    "aspect_pxl_res_img",
    "catagorise_aspect",
    "hillshade",
    "hillshade_pxl_res_img",
    "shadow_mask",
    "local_incidence_angle",
    "local_existance_angle",
    "dtm_aspect_median_filter",
    "fill_dem_soille_gratin_1994",
    "plane_fit_detreat_dem",
    "fft_dem_fusion",
    "resampling_detread_dem",
]


def _validate_input_file(img_path: str, label: str = "Input image") -> None:
    """Validate that ``img_path`` exists on disk."""
    if not os.path.exists(img_path):
        raise rsgispy.RSGISPyException(
            f"{label} file does not exist: '{img_path}'."
        )


def _validate_sun_angles(azimuth: float, zenith: float) -> None:
    """Validate solar/view azimuth in ``[0, 360]`` and zenith in ``[0, 90]``."""
    if zenith < 0.0 or zenith > 90.0:
        raise rsgispy.RSGISPyException(
            "The zenith angle should be between 0 and 90 degrees."
        )
    if azimuth < 0.0 or azimuth > 360.0:
        raise rsgispy.RSGISPyException(
            "The azimuth angle should be between 0 and 360 degrees."
        )


def _validate_slope_unit(unit: str) -> str:
    """Validate and normalise slope output unit ('degrees' or 'radians')."""
    unit_norm = unit.strip().lower()
    if unit_norm not in ("degrees", "radians"):
        raise rsgispy.RSGISPyException(
            "The unit option needs to be specified as either 'degrees' or 'radians'."
        )
    return unit_norm


def _grid_from_dataset(src: rasterio.DatasetReader) -> IntersectionGrid:
    """Construct an :class:`IntersectionGrid` matching an opened dataset."""
    return IntersectionGrid(
        crs=src.crs,
        transform=src.transform,
        width=src.width,
        height=src.height,
        offsets={},
    )


def _read_window_with_halo(
    src: rasterio.DatasetReader,
    band: int,
    out_win: Window,
    pad: int,
    out_dtype: Any = np.float64,
) -> np.ndarray:
    """Read a window expanded by ``pad`` pixels, edge-padding at image borders."""
    # pylint: disable=too-many-locals
    col_off = int(out_win.col_off)
    row_off = int(out_win.row_off)
    win_w = int(out_win.width)
    win_h = int(out_win.height)

    r0 = row_off - pad
    r1 = row_off + win_h + pad
    c0 = col_off - pad
    c1 = col_off + win_w + pad

    cr0 = max(0, r0)
    cr1 = min(src.height, r1)
    cc0 = max(0, c0)
    cc1 = min(src.width, c1)

    arr = src.read(
        band,
        window=Window(cc0, cr0, cc1 - cc0, cr1 - cr0),
        out_dtype=out_dtype,
    )
    pads = (
        (cr0 - r0, r1 - cr1),
        (cc0 - c0, c1 - cc1),
    )
    if any(p > 0 for pair in pads for p in pair):
        arr = np.pad(arr, pads, mode="edge")
    return arr


def _valid_mask(arr: np.ndarray, no_data_val: Optional[float]) -> np.ndarray:
    """Return boolean mask of finite values excluding ``no_data_val``."""
    mask = np.isfinite(arr)
    if no_data_val is not None and not (
        isinstance(no_data_val, (float, np.floating))
        and math.isnan(float(no_data_val))
    ):
        mask &= arr != no_data_val
    return mask


def _prepare_3x3_horn_diffs(
    dem_halo: np.ndarray,
    no_data_val: Optional[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build cleaned 3x3 windows (replacing isolated nodata values with window
    means when ``n_vals > 1``) and return ``(dx_w_minus_e, dy_s_minus_n, n_vals)``.
    """
    win = np.lib.stride_tricks.sliding_window_view(dem_halo, (3, 3))
    valid = _valid_mask(win, no_data_val)
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


def _calc_slope_aspect_from_diffs(
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


def _calc_hillshade_block(
    dx_w_minus_e: np.ndarray,
    dy_s_minus_n: np.ndarray,
    n_vals: np.ndarray,
    ew_res: Any,
    ns_res: Any,
    azimuth: float,
    zenith: float,
) -> np.ndarray:
    """Compute uint8 hillshade values (range 1..255) matching RSGISLib."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    sun_elev_rad = math.radians(90.0 - zenith)
    sun_az_trans = 360.0 - azimuth + 90.0
    if sun_az_trans > 360.0:
        sun_az_trans -= 360.0
    sun_az_rad = math.radians(sun_az_trans)

    dx = -dx_w_minus_e / (ew_res * 8.0)
    dy = -dy_s_minus_n / (ns_res * 8.0)
    xx_plus_yy = (dx * dx) + (dy * dy)
    aspect_math = np.arctan2(dy, dx)

    cang = (
        math.sin(sun_elev_rad)
        - math.cos(sun_elev_rad)
        * np.sqrt(xx_plus_yy)
        * np.sin(aspect_math - (sun_az_rad - (math.pi / 2.0)))
    ) / np.sqrt(1.0 + xx_plus_yy)

    out = np.where((n_vals > 1) & (cang > 0.0), 1.0 + (254.0 * cang), 1.0)
    return np.clip(out, 0, 255).astype(np.uint8)


def _calc_ray_angle_block(
    dx_w_minus_e: np.ndarray,
    dy_s_minus_n: np.ndarray,
    n_vals: np.ndarray,
    ew_res: float,
    ns_res: float,
    azimuth: float,
    zenith: float,
    flat_aspect_val: float,
    fallback_val: float,
) -> np.ndarray:
    """Shared ray incidence/exitance angle calculation for a 3x3 DEM block."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    slope_rad, aspect_deg, flat_mask = _calc_slope_aspect_from_diffs(
        dx_w_minus_e, dy_s_minus_n, n_vals, ew_res, ns_res
    )
    if not math.isnan(flat_aspect_val):
        aspect_deg = np.where((n_vals > 1) & flat_mask, flat_aspect_val, aspect_deg)

    aspect_rad = np.radians(aspect_deg)
    pa = np.sin(slope_rad) * np.cos(aspect_rad)
    pb = np.sin(slope_rad) * np.sin(aspect_rad)
    pc = np.cos(slope_rad)

    zen_rad = math.radians(zenith)
    az_rad = math.radians(azimuth)
    ra = math.sin(zen_rad) * math.cos(az_rad)
    rb = math.sin(zen_rad) * math.sin(az_rad)
    rc = math.cos(zen_rad)

    dot = np.clip((pa * ra) + (pb * rb) + (pc * rc), -1.0, 1.0)
    ang_deg = np.degrees(np.arccos(dot))
    invalid = (n_vals <= 1) | np.isnan(ang_deg)
    return np.where(invalid, fallback_val, ang_deg).astype(np.float32)


def _process_single_dem_3x3(
    input_img: str,
    output_img: str,
    gdalformat: str,
    out_rio_dtype: str,
    block_fn: Callable[[np.ndarray, np.ndarray, np.ndarray, float, float], np.ndarray],
) -> None:
    """Run a 3x3 windowed calculation block-by-block over a single DEM raster."""
    # pylint: disable=too-many-locals
    _validate_input_file(input_img)
    with ExitStack() as stack:
        src = stack.enter_context(rasterio.open(input_img, "r"))
        grid = _grid_from_dataset(src)
        ew_res = abs(float(src.transform.a))
        ns_res = abs(float(src.transform.e))
        no_data_val = src.nodata

        dst = open_output_dataset(
            stack=stack,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype=out_rio_dtype,
            grid=grid,
            count=1,
        )

        block_size = get_default_block_size()
        tqdm_cls = rsgispy.get_tqdm()
        total_blocks = math.ceil(src.height / block_size) * math.ceil(
            src.width / block_size
        )

        with tqdm_cls(total=total_blocks) as pbar:
            for row_off in range(0, src.height, block_size):
                win_h = min(block_size, src.height - row_off)
                for col_off in range(0, src.width, block_size):
                    win_w = min(block_size, src.width - col_off)
                    out_win = Window(col_off, row_off, win_w, win_h)
                    dem_halo = _read_window_with_halo(src, 1, out_win, pad=1)
                    dx, dy, n_vals = _prepare_3x3_horn_diffs(dem_halo, no_data_val)
                    out_arr = block_fn(dx, dy, n_vals, ew_res, ns_res)
                    dst.write(out_arr, 1, window=out_win)
                    pbar.update(1)


def _process_pxl_res_dem_3x3(
    in_dem_img: str,
    in_pxl_res_img: str,
    output_img: str,
    gdalformat: str,
    out_rio_dtype: str,
    block_fn: Callable[
        [np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray
    ],
) -> None:
    """Run a 3x3 windowed calculation using per-pixel E-W/N-S resolution bands."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    _validate_input_file(in_dem_img, "Input DEM image")
    _validate_input_file(in_pxl_res_img, "Input pixel resolution image")

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

        grid = _grid_from_dataset(dem_src)
        no_data_val = dem_src.nodata
        dst = open_output_dataset(
            stack=stack,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype=out_rio_dtype,
            grid=grid,
            count=1,
        )

        block_size = get_default_block_size()
        tqdm_cls = rsgispy.get_tqdm()
        total_blocks = math.ceil(dem_src.height / block_size) * math.ceil(
            dem_src.width / block_size
        )

        with tqdm_cls(total=total_blocks) as pbar:
            for row_off in range(0, dem_src.height, block_size):
                win_h = min(block_size, dem_src.height - row_off)
                for col_off in range(0, dem_src.width, block_size):
                    win_w = min(block_size, dem_src.width - col_off)
                    out_win = Window(col_off, row_off, win_w, win_h)
                    dem_halo = _read_window_with_halo(dem_src, 1, out_win, pad=1)
                    dx, dy, n_vals = _prepare_3x3_horn_diffs(dem_halo, no_data_val)
                    ew_res = res_src.read(1, window=out_win, out_dtype=np.float64)
                    ns_res = np.abs(
                        res_src.read(2, window=out_win, out_dtype=np.float64)
                    )
                    out_arr = block_fn(dx, dy, n_vals, ew_res, ns_res)
                    dst.write(out_arr, 1, window=out_win)
                    pbar.update(1)


def slope(
    input_img: str,
    output_img: str,
    gdalformat: str,
    unit: str = "degrees",
) -> None:
    """
    Calculate a slope layer from an input elevation model.

    :param input_img: Path to the input DEM image file.
    :param output_img: Path to the output slope image file.
    :param gdalformat: GDAL/rasterio driver format for the output file.
    :param unit: Output angular unit, either ``'degrees'`` or ``'radians'``.
    """
    unit_norm = _validate_slope_unit(unit)

    def _block(
        dx: np.ndarray,
        dy: np.ndarray,
        n_vals: np.ndarray,
        ew_res: float,
        ns_res: float,
    ) -> np.ndarray:
        slope_rad, _, _ = _calc_slope_aspect_from_diffs(dx, dy, n_vals, ew_res, ns_res)
        if unit_norm == "degrees":
            return np.degrees(slope_rad).astype(np.float32)
        return slope_rad.astype(np.float32)

    _process_single_dem_3x3(input_img, output_img, gdalformat, "float32", _block)


def slope_pxl_res_img(
    in_dem_img: str,
    in_pxl_res_img: str,
    output_img: str,
    gdalformat: str,
    unit: str = "degrees",
) -> None:
    """
    Calculate a slope layer from a DEM and an external 2-band pixel resolution
    image (Band 1: East-West resolution, Band 2: North-South resolution).

    Useful for geographic projections (e.g., EPSG:4326) whose horizontal units
    differ from the elevation units.

    :param in_dem_img: Path to the input DEM image file.
    :param in_pxl_res_img: Path to the 2-band pixel resolution image file.
    :param output_img: Path to the output slope image file.
    :param gdalformat: GDAL/rasterio driver format for the output file.
    :param unit: Output angular unit, either ``'degrees'`` or ``'radians'``.
    """
    unit_norm = _validate_slope_unit(unit)

    def _block(
        dx: np.ndarray,
        dy: np.ndarray,
        n_vals: np.ndarray,
        ew_res: np.ndarray,
        ns_res: np.ndarray,
    ) -> np.ndarray:
        slope_rad, _, _ = _calc_slope_aspect_from_diffs(dx, dy, n_vals, ew_res, ns_res)
        if unit_norm == "degrees":
            return np.degrees(slope_rad).astype(np.float32)
        return slope_rad.astype(np.float32)

    _process_pxl_res_dem_3x3(
        in_dem_img, in_pxl_res_img, output_img, gdalformat, "float32", _block
    )


def aspect(
    input_img: str,
    output_img: str,
    gdalformat: str,
) -> None:
    """
    Calculate an aspect layer (in degrees clockwise from North, ``[0, 360)``,
    ``NaN`` for flat areas) from an input elevation model.

    :param input_img: Path to the input DEM image file.
    :param output_img: Path to the output aspect image file.
    :param gdalformat: GDAL/rasterio driver format for the output file.
    """

    def _block(
        dx: np.ndarray,
        dy: np.ndarray,
        n_vals: np.ndarray,
        ew_res: float,
        ns_res: float,
    ) -> np.ndarray:
        _, aspect_deg, _ = _calc_slope_aspect_from_diffs(dx, dy, n_vals, ew_res, ns_res)
        return aspect_deg.astype(np.float32)

    _process_single_dem_3x3(input_img, output_img, gdalformat, "float32", _block)


def aspect_pxl_res_img(
    in_dem_img: str,
    in_pxl_res_img: str,
    output_img: str,
    gdalformat: str,
) -> None:
    """
    Calculate an aspect layer from a DEM and an external 2-band pixel resolution
    image (Band 1: East-West resolution, Band 2: North-South resolution).

    :param in_dem_img: Path to the input DEM image file.
    :param in_pxl_res_img: Path to the 2-band pixel resolution image file.
    :param output_img: Path to the output aspect image file.
    :param gdalformat: GDAL/rasterio driver format for the output file.
    """

    def _block(
        dx: np.ndarray,
        dy: np.ndarray,
        n_vals: np.ndarray,
        ew_res: np.ndarray,
        ns_res: np.ndarray,
    ) -> np.ndarray:
        _, aspect_deg, _ = _calc_slope_aspect_from_diffs(dx, dy, n_vals, ew_res, ns_res)
        return aspect_deg.astype(np.float32)

    _process_pxl_res_dem_3x3(
        in_dem_img, in_pxl_res_img, output_img, gdalformat, "float32", _block
    )


def catagorise_aspect(
    input_img: str,
    output_img: str,
    gdalformat: str,
) -> None:
    """
    Categorise an aspect image (in degrees) into 8 classes (1-8 for 45-degree
    bins: 0-45, 45-90, ..., 315-360) and ``0`` for ``NaN``/flat pixels.

    :param input_img: Path to the input aspect image file (in degrees).
    :param output_img: Path to the output categorised uint8 image file.
    :param gdalformat: GDAL/rasterio driver format for the output file.
    """
    # pylint: disable=too-many-locals
    _validate_input_file(input_img)

    with ExitStack() as stack:
        src = stack.enter_context(rasterio.open(input_img, "r"))
        grid = _grid_from_dataset(src)
        dst = open_output_dataset(
            stack=stack,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype="uint8",
            grid=grid,
            count=1,
        )

        block_size = get_default_block_size()
        tqdm_cls = rsgispy.get_tqdm()
        total_blocks = math.ceil(src.height / block_size) * math.ceil(
            src.width / block_size
        )

        with tqdm_cls(total=total_blocks) as pbar:
            for row_off in range(0, src.height, block_size):
                win_h = min(block_size, src.height - row_off)
                for col_off in range(0, src.width, block_size):
                    win_w = min(block_size, src.width - col_off)
                    out_win = Window(col_off, row_off, win_w, win_h)
                    arr = src.read(1, window=out_win, out_dtype=np.float64)
                    nan_mask = np.isnan(arr)
                    finite_vals = arr[~nan_mask]
                    if np.any(finite_vals < 0.0):
                        raise rsgispy.RSGISPyException(
                            "The input image pixel values must be between 0 and 360 degrees."
                        )

                    norm = np.where(arr > 360.0, np.mod(arr, 360.0), arr)
                    out_cls = np.zeros(arr.shape, dtype=np.uint8)
                    valid = ~nan_mask
                    bins = np.clip((norm[valid] // 45.0).astype(np.int64) + 1, 1, 8)
                    out_cls[valid] = bins.astype(np.uint8)
                    dst.write(out_cls, 1, window=out_win)
                    pbar.update(1)


def hillshade(
    input_img: str,
    output_img: str,
    azimuth: float,
    zenith: float,
    gdalformat: str,
) -> None:
    """
    Calculate a uint8 hillshade layer from an input elevation model.

    :param input_img: Path to the input DEM image file.
    :param output_img: Path to the output hillshade image file.
    :param azimuth: Solar azimuth angle in degrees ``[0, 360]`` (e.g., 315).
    :param zenith: Solar zenith angle in degrees ``[0, 90]`` (e.g., 45).
    :param gdalformat: GDAL/rasterio driver format for the output file.
    """
    _validate_sun_angles(azimuth, zenith)

    def _block(
        dx: np.ndarray,
        dy: np.ndarray,
        n_vals: np.ndarray,
        ew_res: float,
        ns_res: float,
    ) -> np.ndarray:
        return _calc_hillshade_block(dx, dy, n_vals, ew_res, ns_res, azimuth, zenith)

    _process_single_dem_3x3(input_img, output_img, gdalformat, "uint8", _block)


def hillshade_pxl_res_img(
    in_dem_img: str,
    in_pxl_res_img: str,
    output_img: str,
    azimuth: float,
    zenith: float,
    gdalformat: str,
) -> None:
    """
    Calculate a uint8 hillshade layer from a DEM and an external 2-band pixel
    resolution image (Band 1: East-West resolution, Band 2: North-South resolution).

    :param in_dem_img: Path to the input DEM image file.
    :param in_pxl_res_img: Path to the 2-band pixel resolution image file.
    :param output_img: Path to the output hillshade image file.
    :param azimuth: Solar azimuth angle in degrees ``[0, 360]`` (e.g., 315).
    :param zenith: Solar zenith angle in degrees ``[0, 90]`` (e.g., 45).
    :param gdalformat: GDAL/rasterio driver format for the output file.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    _validate_sun_angles(azimuth, zenith)

    def _block(
        dx: np.ndarray,
        dy: np.ndarray,
        n_vals: np.ndarray,
        ew_res: np.ndarray,
        ns_res: np.ndarray,
    ) -> np.ndarray:
        return _calc_hillshade_block(dx, dy, n_vals, ew_res, ns_res, azimuth, zenith)

    _process_pxl_res_dem_3x3(
        in_dem_img, in_pxl_res_img, output_img, gdalformat, "uint8", _block
    )


def shadow_mask(
    input_img: str,
    output_img: str,
    azimuth: float,
    zenith: float,
    max_height: float,
    gdalformat: str,
) -> None:
    """
    Calculate a binary shadow mask (``1`` = shadow, ``0`` = illuminated/nodata)
    given an input elevation model using self-shadowing and solar ray tracing.

    :param input_img: Path to the input DEM image file.
    :param output_img: Path to the output binary mask image file.
    :param azimuth: Solar azimuth angle in degrees ``[0, 360]``.
    :param zenith: Solar zenith angle in degrees ``[0, 90]``.
    :param max_height: Maximum elevation threshold terminating upward ray tracing.
    :param gdalformat: GDAL/rasterio driver format for the output file.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals,too-many-statements
    _validate_sun_angles(azimuth, zenith)
    _validate_input_file(input_img)

    with ExitStack() as stack:
        src = stack.enter_context(rasterio.open(input_img, "r"))
        grid = _grid_from_dataset(src)
        ew_res = abs(float(src.transform.a))
        ns_res = abs(float(src.transform.e))
        no_data_val = src.nodata
        dem = src.read(1, out_dtype=np.float64)

        dem_valid = _valid_mask(dem, no_data_val)
        dem_halo = np.pad(dem, 1, mode="edge")
        dx, dy, n_vals = _prepare_3x3_horn_diffs(dem_halo, no_data_val)
        slope_rad, aspect_deg, flat_mask = _calc_slope_aspect_from_diffs(
            dx, dy, n_vals, ew_res, ns_res
        )

        sun_zen_rad = math.radians(zenith)
        sun_az_rad = math.radians(azimuth)

        shadow = np.zeros(dem.shape, dtype=np.uint8)
        has_slope = dem_valid & (n_vals > 1) & ~flat_mask
        ic = (
            math.cos(sun_zen_rad) * np.cos(slope_rad)
            + math.sin(sun_zen_rad)
            * np.sin(slope_rad)
            * np.cos(sun_az_rad - np.radians(aspect_deg))
        )
        shadow[has_slope & (ic < 0.0)] = 1

        # Ray tracing for remaining valid pixels not already self-shadowed
        sun_az_trans = (360.0 - azimuth + 90.0) % 360.0
        sun_az_trans_rad = math.radians(sun_az_trans)
        step_range = min(ew_res, ns_res)
        x_step = (step_range * math.sin(sun_zen_rad) * math.cos(sun_az_trans_rad)) / 3.0
        y_step = (step_range * math.sin(sun_zen_rad) * math.sin(sun_az_trans_rad)) / 3.0
        z_step = (step_range * math.cos(sun_zen_rad)) / 3.0

        col_step = x_step / ew_res
        row_step = -y_step / ns_res

        if abs(col_step) > 1e-12 or abs(row_step) > 1e-12:
            cand_r, cand_c = np.where(dem_valid & (shadow == 0))
            if cand_r.size > 0:
                orig_r = cand_r.copy()
                orig_c = cand_c.copy()
                col_f = orig_c.astype(np.float64) + 0.5
                row_f = orig_r.astype(np.float64) + 0.5
                z_f = dem[orig_r, orig_c].copy()

                max_steps = int(
                    math.ceil(
                        math.hypot(src.width, src.height) * 3.0
                        * max(ew_res, ns_res)
                        / step_range
                    )
                ) + 2

                for _ in range(max_steps):
                    col_f += col_step
                    row_f += row_step
                    z_f += z_step

                    in_bounds = (
                        (col_f >= 0.0)
                        & (col_f < src.width)
                        & (row_f >= 0.0)
                        & (row_f < src.height)
                        & (z_f <= max_height)
                    )
                    if not np.any(in_bounds):
                        break

                    orig_r = orig_r[in_bounds]
                    orig_c = orig_c[in_bounds]
                    col_f = col_f[in_bounds]
                    row_f = row_f[in_bounds]
                    z_f = z_f[in_bounds]

                    rr = row_f.astype(np.int64)
                    cc = col_f.astype(np.int64)
                    moved = (rr != orig_r) | (cc != orig_c)
                    hit = moved & dem_valid[rr, cc] & (z_f < dem[rr, cc])
                    if np.any(hit):
                        shadow[orig_r[hit], orig_c[hit]] = 1
                        keep = ~hit
                        if not np.any(keep):
                            break
                        orig_r = orig_r[keep]
                        orig_c = orig_c[keep]
                        col_f = col_f[keep]
                        row_f = row_f[keep]
                        z_f = z_f[keep]

        dst = open_output_dataset(
            stack=stack,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype="uint8",
            grid=grid,
            count=1,
        )
        dst.write(shadow, 1)


def local_incidence_angle(
    input_img: str,
    output_img: str,
    azimuth: float,
    zenith: float,
    gdalformat: str,
) -> None:
    """
    Calculate the local solar incidence angle (degrees) for each pixel of a DEM.

    :param input_img: Path to the input DEM image file.
    :param output_img: Path to the output incidence angle image file.
    :param azimuth: Solar azimuth angle in degrees.
    :param zenith: Solar zenith angle in degrees.
    :param gdalformat: GDAL/rasterio driver format for the output file.
    """

    def _block(
        dx: np.ndarray,
        dy: np.ndarray,
        n_vals: np.ndarray,
        ew_res: float,
        ns_res: float,
    ) -> np.ndarray:
        return _calc_ray_angle_block(
            dx,
            dy,
            n_vals,
            ew_res,
            ns_res,
            azimuth,
            zenith,
            flat_aspect_val=float("nan"),
            fallback_val=float(zenith),
        )

    _process_single_dem_3x3(input_img, output_img, gdalformat, "float32", _block)


def local_existance_angle(
    input_img: str,
    output_img: str,
    azimuth: float,
    zenith: float,
    gdalformat: str,
) -> None:
    """
    Calculate the local sensor exitance (viewing) angle (degrees) for each pixel
    of a DEM.

    :param input_img: Path to the input DEM image file.
    :param output_img: Path to the output exitance angle image file.
    :param azimuth: Sensor/view azimuth angle in degrees.
    :param zenith: Sensor/view zenith angle in degrees.
    :param gdalformat: GDAL/rasterio driver format for the output file.
    """

    def _block(
        dx: np.ndarray,
        dy: np.ndarray,
        n_vals: np.ndarray,
        ew_res: float,
        ns_res: float,
    ) -> np.ndarray:
        return _calc_ray_angle_block(
            dx,
            dy,
            n_vals,
            ew_res,
            ns_res,
            azimuth,
            zenith,
            flat_aspect_val=0.0,
            fallback_val=0.0,
        )

    _process_single_dem_3x3(input_img, output_img, gdalformat, "float32", _block)


def dtm_aspect_median_filter(
    in_dem_img: str,
    in_aspect_img: str,
    output_img: str,
    aspect_range: float,
    win_h_size: int,
    gdalformat: str,
) -> None:
    """
    Filter a DTM for noise using a median filter restricted to neighbouring pixels
    whose aspect lies within ``aspect_range`` degrees of the centre pixel.

    Falls back to the full window median of valid DTM pixels when no neighbours
    match the aspect criterion, or ``NaN`` if no valid pixels exist in the window.

    :param in_dem_img: Path to the input DTM image file.
    :param in_aspect_img: Path to the input aspect image file (in degrees).
    :param output_img: Path to the output filtered DTM image file.
    :param aspect_range: Angular tolerance around the centre pixel aspect (degrees).
    :param win_h_size: Half-window size in pixels (full window is ``2 * win_h_size + 1``).
    :param gdalformat: GDAL/rasterio driver format for the output file.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    if win_h_size < 1:
        raise rsgispy.RSGISPyException("win_h_size must be at least 1.")
    if aspect_range < 0.0:
        raise rsgispy.RSGISPyException("aspect_range must be non-negative.")

    _validate_input_file(in_dem_img, "Input DEM image")
    _validate_input_file(in_aspect_img, "Input aspect image")

    win_size = (win_h_size * 2) + 1
    with ExitStack() as stack:
        dem_src = stack.enter_context(rasterio.open(in_dem_img, "r"))
        asp_src = stack.enter_context(rasterio.open(in_aspect_img, "r"))
        if dem_src.width != asp_src.width or dem_src.height != asp_src.height:
            raise rsgispy.RSGISPyException(
                "Input DEM and aspect images must have matching dimensions."
            )

        grid = _grid_from_dataset(dem_src)
        no_data_val = dem_src.nodata
        dst = open_output_dataset(
            stack=stack,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype="float32",
            grid=grid,
            count=1,
        )

        block_size = get_default_block_size()
        tqdm_cls = rsgispy.get_tqdm()
        total_blocks = math.ceil(dem_src.height / block_size) * math.ceil(
            dem_src.width / block_size
        )

        with tqdm_cls(total=total_blocks) as pbar:
            for row_off in range(0, dem_src.height, block_size):
                win_h = min(block_size, dem_src.height - row_off)
                for col_off in range(0, dem_src.width, block_size):
                    win_w = min(block_size, dem_src.width - col_off)
                    out_win = Window(col_off, row_off, win_w, win_h)

                    dem_halo = _read_window_with_halo(dem_src, 1, out_win, win_h_size)
                    asp_halo = _read_window_with_halo(asp_src, 1, out_win, win_h_size)

                    dem_wins = np.lib.stride_tricks.sliding_window_view(
                        dem_halo, (win_size, win_size)
                    ).reshape(win_h, win_w, -1)
                    asp_wins = np.lib.stride_tricks.sliding_window_view(
                        asp_halo, (win_size, win_size)
                    ).reshape(win_h, win_w, -1)

                    center_asp = asp_halo[
                        win_h_size : win_h_size + win_h,
                        win_h_size : win_h_size + win_w,
                    ][..., None]
                    ang_diff = np.abs(
                        np.mod((asp_wins - center_asp) + 180.0, 360.0) - 180.0
                    )
                    asp_match = np.isfinite(ang_diff) & (ang_diff <= aspect_range)
                    dem_valid = _valid_mask(dem_wins, no_data_val)

                    sel_match = dem_valid & asp_match
                    has_match = np.any(sel_match, axis=2)
                    has_any = np.any(dem_valid, axis=2)

                    out_blk = np.full((win_h, win_w), np.nan, dtype=np.float32)
                    if np.any(has_match):
                        matched_vals = np.where(
                            sel_match[has_match], dem_wins[has_match], np.nan
                        )
                        out_blk[has_match] = np.nanmedian(matched_vals, axis=1).astype(
                            np.float32
                        )
                    fallback_mask = (~has_match) & has_any
                    if np.any(fallback_mask):
                        fb_vals = np.where(
                            dem_valid[fallback_mask], dem_wins[fallback_mask], np.nan
                        )
                        out_blk[fallback_mask] = np.nanmedian(fb_vals, axis=1).astype(
                            np.float32
                        )

                    dst.write(out_blk, 1, window=out_win)
                    pbar.update(1)


def fill_dem_soille_gratin_1994(
    in_dem_img: str,
    in_vld_img: str,
    output_img: str,
    gdalformat: str,
) -> None:
    """
    Fill local minima/depressions in an integer DEM using the Soille & Gratin
    (1994) hierarchical reconstruction algorithm.

    Reference: Soille, P., and Gratin, C. (1994). An efficient algorithm for
    drainage network extraction on DEMs. *Journal of Visual Communication and
    Image Representation*, 5(2), 181-189.

    :param in_dem_img: Path to the input integer-dtype single-band DEM image.
    :param in_vld_img: Path to a binary mask image specifying valid pixels (``1`` = valid).
    :param output_img: Path to the output filled DEM image file.
    :param gdalformat: GDAL/rasterio driver format for the output file.
    """
    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    _validate_input_file(in_dem_img, "Input DEM image")
    _validate_input_file(in_vld_img, "Input valid mask image")

    with ExitStack() as stack:
        dem_src = stack.enter_context(rasterio.open(in_dem_img, "r"))
        vld_src = stack.enter_context(rasterio.open(in_vld_img, "r"))

        if dem_src.count != 1:
            raise rsgispy.RSGISPyException(
                "The image to be filled should only have 1 image band."
            )
        rsgis_dt = rsgispy.get_rsgis_datatype(dem_src.dtypes[0])
        out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(rsgis_dt))
        if not np.issubdtype(out_np_dtype, np.integer):
            raise rsgispy.RSGISPyException(
                "Input image must be of an integer data type."
            )
        if dem_src.width != vld_src.width or dem_src.height != vld_src.height:
            raise rsgispy.RSGISPyException(
                "Input DEM and valid area mask images must have matching dimensions."
            )

        dem = dem_src.read(1)
        vld = vld_src.read(1) == 1
        dem_nodata = dem_src.nodata

        stat_mask = vld.copy()
        if dem_nodata is not None:
            stat_mask &= dem != dem_nodata
        valid_vals = dem[stat_mask]
        if valid_vals.size == 0:
            raise rsgispy.RSGISPyException(
                "No valid DEM pixels found within the valid area mask."
            )

        min_val = int(valid_vals.min())
        max_val = int(valid_vals.max())
        mean_val = float(valid_vals.mean())
        std_val = float(valid_vals.std())
        if (mean_val - std_val) > min_val:
            border_val = int(math.floor((mean_val - std_val) + 0.5))
        else:
            border_val = int(math.floor(mean_val + 0.5))

        fill_nodata = int(dem_nodata) if dem_nodata is not None else 0
        out = np.full(dem.shape, fill_nodata, dtype=np.int64)
        out[vld] = max_val

        # Identify invalid pixels adjacent (8-connected) to valid region
        vld_pad = np.pad(vld, 1, mode="constant", constant_values=False)
        vld_nbr_wins = np.lib.stride_tricks.sliding_window_view(vld_pad, (3, 3))
        boundary = (~vld) & np.any(vld_nbr_wins, axis=(2, 3))

        heap: list = []
        bnd_r, bnd_c = np.where(boundary)
        if bnd_r.size > 0:
            out[bnd_r, bnd_c] = border_val
            for r_idx, c_idx in zip(bnd_r.tolist(), bnd_c.tolist()):
                heap.append((border_val, r_idx, c_idx))
        else:
            height, width = dem.shape
            edge_coords = (
                [(0, c) for c in range(width)]
                + [(height - 1, c) for c in range(width)]
                + [(r, 0) for r in range(height)]
                + [(r, width - 1) for r in range(height)]
            )
            for r_idx, c_idx in set(edge_coords):
                out[r_idx, c_idx] = border_val
                heap.append((border_val, r_idx, c_idx))

        heapq.heapify(heap)
        height, width = dem.shape
        while heap:
            h_cur, r_cur, c_cur = heapq.heappop(heap)
            r_min = max(0, r_cur - 1)
            r_max = min(height, r_cur + 2)
            c_min = max(0, c_cur - 1)
            c_max = min(width, c_cur + 2)
            for nr in range(r_min, r_max):
                for nc in range(c_min, c_max):
                    if (nr != r_cur or nc != c_cur) and vld[nr, nc] and out[nr, nc] == max_val:
                        new_val = max(h_cur, int(dem[nr, nc]))
                        out[nr, nc] = new_val
                        if new_val < max_val:
                            heapq.heappush(heap, (new_val, nr, nc))

        dst = open_output_dataset(
            stack=stack,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype=rsgispy.get_rasterio_datatype(rsgis_dt),
            grid=_grid_from_dataset(dem_src),
            count=1,
        )
        if dem_nodata is not None:
            dst.nodata = dem_nodata
        dst.write(out.astype(out_np_dtype), 1)


def plane_fit_detreat_dem(
    input_img: str,
    output_img: str,
    gdalformat: str,
    win_size: int,
) -> None:
    """
    Detrend a DEM by fitting a local least-squares plane ``z = a*x + b*y + c``
    over each ``win_size x win_size`` window and outputting the fitted centre
    elevation ``c``.

    :param input_img: Path to the input DEM image file.
    :param output_img: Path to the output detrended DEM image file.
    :param gdalformat: GDAL/rasterio driver format for the output file.
    :param win_size: Odd window size in pixels (``>= 3``) for plane fitting.
    """
    # pylint: disable=too-many-locals
    if win_size < 3 or win_size % 2 == 0:
        raise rsgispy.RSGISPyException(
            "win_size must be an odd integer greater than or equal to 3."
        )
    _validate_input_file(input_img)

    half = win_size // 2
    offs = np.arange(-half, half + 1, dtype=np.float64)
    gx, gy = np.meshgrid(offs, offs)
    flat_x = gx.ravel()
    flat_y = gy.ravel()
    flat_xx = flat_x * flat_x
    flat_yy = flat_y * flat_y
    flat_xy = flat_x * flat_y

    with ExitStack() as stack:
        src = stack.enter_context(rasterio.open(input_img, "r"))
        grid = _grid_from_dataset(src)
        no_data_val = src.nodata
        dst = open_output_dataset(
            stack=stack,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype="float32",
            grid=grid,
            count=1,
        )

        block_size = get_default_block_size()
        tqdm_cls = rsgispy.get_tqdm()
        total_blocks = math.ceil(src.height / block_size) * math.ceil(
            src.width / block_size
        )

        with tqdm_cls(total=total_blocks) as pbar:
            for row_off in range(0, src.height, block_size):
                win_h = min(block_size, src.height - row_off)
                for col_off in range(0, src.width, block_size):
                    win_w = min(block_size, src.width - col_off)
                    out_win = Window(col_off, row_off, win_w, win_h)
                    dem_halo = _read_window_with_halo(src, 1, out_win, pad=half)
                    wins = np.lib.stride_tricks.sliding_window_view(
                        dem_halo, (win_size, win_size)
                    ).reshape(win_h, win_w, -1)

                    valid = _valid_mask(wins, no_data_val).astype(np.float64)
                    z_clean = np.where(valid > 0.0, wins, 0.0)

                    s_n = valid.sum(axis=2)
                    s_x = valid @ flat_x
                    s_y = valid @ flat_y
                    s_xx = valid @ flat_xx
                    s_yy = valid @ flat_yy
                    s_xy = valid @ flat_xy
                    s_xz = (z_clean * flat_x).sum(axis=2)
                    s_yz = (z_clean * flat_y).sum(axis=2)
                    s_z = z_clean.sum(axis=2)

                    # Solve 3x3 normal equations for c = (c0*s_xz + c1*s_yz + c2*s_z) / det
                    det = (
                        s_xx * ((s_yy * s_n) - (s_y * s_y))
                        - s_xy * ((s_xy * s_n) - (s_x * s_y))
                        + s_x * ((s_xy * s_y) - (s_yy * s_x))
                    )
                    num_c = (
                        ((s_xy * s_y) - (s_x * s_yy)) * s_xz
                        + ((s_xy * s_x) - (s_xx * s_y)) * s_yz
                        + ((s_xx * s_yy) - (s_xy * s_xy)) * s_z
                    )
                    out_c = np.divide(
                        num_c,
                        det,
                        out=np.zeros_like(det, dtype=np.float64),
                        where=(s_n > 0) & (np.abs(det) > 1e-12),
                    )
                    dst.write(out_c.astype(np.float32), 1, window=out_win)
                    pbar.update(1)


def fft_dem_fusion(
    high_res_dem_img: str,
    low_res_dem_img: str,
    fused_dem_img: str,
    c_size: int = 20,
    gdalformat: str = "GTIFF",
) -> None:
    """
    Merge two DEMs using a 2D Fourier transform where frequency components inside
    a central square of half-width ``c_size`` are taken from ``high_res_dem_img``
    and remaining components are taken from ``low_res_dem_img``.

    Reference: Karkee, M., Steward, B. L., & Aziz, S. A. (2008). Improving
    quality of public domain digital elevation models through data fusion.
    *Biosystems Engineering*, 101(3), 293-305.

    :param high_res_dem_img: Input higher-resolution DEM file (band 1).
    :param low_res_dem_img: Input lower-resolution DEM file (band 1, matching grid).
    :param fused_dem_img: Output fused DEM file path.
    :param c_size: Half-width (in frequency pixels) of the central spectral patch
                   (Default = 20).
    :param gdalformat: GDAL/rasterio driver format for the output file
                       (Default = ``'GTIFF'``).
    """
    # pylint: disable=too-many-locals
    if c_size < 1:
        raise rsgispy.RSGISPyException("c_size must be at least 1.")
    _validate_input_file(high_res_dem_img, "High-res DEM image")
    _validate_input_file(low_res_dem_img, "Low-res DEM image")

    with rasterio.open(high_res_dem_img, "r") as high_src:
        high_arr = high_src.read(1, out_dtype=np.float64)
        grid = _grid_from_dataset(high_src)

    with rasterio.open(low_res_dem_img, "r") as low_src:
        low_arr = low_src.read(1, out_dtype=np.float64)

    if high_arr.shape != low_arr.shape:
        raise rsgispy.RSGISPyException(
            f"Input DEMs must have identical dimensions: "
            f"{high_arr.shape} != {low_arr.shape}."
        )

    fshift_high = np.fft.fftshift(np.fft.fft2(high_arr))
    fshift_low = np.fft.fftshift(np.fft.fft2(low_arr))

    rows, cols = low_arr.shape
    crow, ccol = rows // 2, cols // 2
    r0 = max(0, crow - c_size)
    r1 = min(rows, crow + c_size)
    c0 = max(0, ccol - c_size)
    c1 = min(cols, ccol + c_size)
    fshift_low[r0:r1, c0:c1] = fshift_high[r0:r1, c0:c1]

    img_back = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift_low))).astype(np.float32)

    with ExitStack() as stack:
        dst = open_output_dataset(
            stack=stack,
            output_img=fused_dem_img,
            gdalformat=gdalformat,
            out_rio_dtype="float32",
            grid=grid,
            count=1,
        )
        dst.nodata = -9999.0
        dst.write(img_back, 1)


def resampling_detread_dem(
    input_img: str,
    output_img: str,
    resmp_res: float,
    gdalformat: str,
    no_data_val: Optional[float] = None,
    datatype: Optional[int] = None,
    tmp_dir: Optional[str] = None,
    low_resmp_mthd: int = rsgispy.INTERP_AVERAGE,
    up_smpl_mthd: int = rsgispy.INTERP_CUBICSPLINE,
) -> None:
    """
    Detrend a DEM by down-sampling to ``resmp_res`` and then up-sampling back to
    the original grid resolution.

    :param input_img: Input DEM image path (band 1 is used).
    :param output_img: Output detrended DEM image path.
    :param resmp_res: Intermediate coarse pixel resolution (positive spatial units).
    :param gdalformat: GDAL/rasterio driver format for the output file.
    :param no_data_val: DEM no-data value (read from ``input_img`` header if ``None``).
    :param datatype: Output ``rsgispy.TYPE_*`` constant (inferred from ``input_img`` if ``None``).
    :param tmp_dir: Optional directory for intermediate rasters (temporary directory
                    created and cleaned up automatically if ``None``).
    :param low_resmp_mthd: Interpolation method for down-sampling
                           (Default = :data:`rsgispy.INTERP_AVERAGE`).
    :param up_smpl_mthd: Interpolation method for up-sampling
                         (Default = :data:`rsgispy.INTERP_CUBICSPLINE`).
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    if resmp_res <= 0.0:
        raise rsgispy.RSGISPyException("resmp_res must be greater than 0.")
    _validate_input_file(input_img)

    with rasterio.open(input_img, "r") as src:
        if datatype is None:
            datatype = rsgispy.get_rsgis_datatype(src.dtypes[0])
        if no_data_val is None:
            no_data_val = src.nodata
        if no_data_val is None:
            raise rsgispy.RSGISPyException(
                "A no data value has not been defined. Either set in image header "
                "or pass to function."
            )
        left, bottom, right, top = src.bounds
        src_crs = src.crs

    coord_width = right - left
    coord_height = top - bottom
    img_width = int(math.ceil(coord_width / resmp_res) + 1)
    img_height = int(math.ceil(coord_height / resmp_res) + 1)

    int_base_name = rsgispy.tools.filetools.get_file_basename(input_img)
    uid_str = rsgispy.tools.utils.uid_generator()
    out_rio_dtype = rsgispy.get_rasterio_datatype(datatype)
    out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(datatype))

    with ExitStack() as stack:
        if tmp_dir is None:
            work_dir = stack.enter_context(tempfile.TemporaryDirectory())
        else:
            os.makedirs(tmp_dir, exist_ok=True)
            work_dir = tmp_dir

        base_img = os.path.join(work_dir, f"{int_base_name}_base_{uid_str}.tif")
        low_res_img = os.path.join(work_dir, f"{int_base_name}_low_res_{uid_str}.tif")

        coarse_transform = Affine(
            float(resmp_res), 0.0, float(left), 0.0, -float(resmp_res), float(top)
        )
        coarse_grid = IntersectionGrid(
            crs=src_crs,
            transform=coarse_transform,
            width=img_width,
            height=img_height,
            offsets={},
        )
        with ExitStack() as base_stack:
            base_dst = open_output_dataset(
                stack=base_stack,
                output_img=base_img,
                gdalformat="GTIFF",
                out_rio_dtype=out_rio_dtype,
                grid=coarse_grid,
                count=1,
            )
            base_dst.nodata = no_data_val
            base_dst.write(
                np.full((img_height, img_width), 0, dtype=out_np_dtype), 1
            )

        rsgispy.imageutils.resample_img_to_match(
            base_img,
            input_img,
            low_res_img,
            gdalformat="GTIFF",
            interp_method=low_resmp_mthd,
            datatype=datatype,
            no_data_val=no_data_val,
            multicore=False,
        )

        rsgispy.imageutils.resample_img_to_match(
            input_img,
            low_res_img,
            output_img,
            gdalformat=gdalformat,
            interp_method=up_smpl_mthd,
            datatype=datatype,
            no_data_val=no_data_val,
            multicore=False,
        )
