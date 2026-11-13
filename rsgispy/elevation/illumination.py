"""
The elevation.illumination module provides solar hillshading, ray-traced shadow
masking, and local solar/sensor incidence and exitance angle functions using
``rasterio`` and ``numpy``.
"""

from contextlib import ExitStack
import math
from typing import Any

import numpy as np
import rasterio

from rsgispy.elevation._helpers import (
    calc_slope_aspect_from_diffs as _calc_slope_aspect_from_diffs,
    prepare_3x3_horn_diffs as _prepare_3x3_horn_diffs,
    process_pxl_res_dem_3x3 as _process_pxl_res_dem_3x3,
    process_single_dem_3x3 as _process_single_dem_3x3,
    valid_mask as _valid_mask,
    validate_sun_angles as _validate_sun_angles,
)
from rsgispy.imageprocess import (
    IntersectionGrid,
    ensure_file_exists,
    open_output_dataset,
)

__all__ = [
    "hillshade",
    "hillshade_pxl_res_img",
    "shadow_mask",
    "local_incidence_angle",
    "local_existance_angle",
]


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
    ra = math.sin(zen_rad) * np.cos(az_rad)
    rb = math.sin(zen_rad) * np.sin(az_rad)
    rc = math.cos(zen_rad)

    dot = np.clip((pa * ra) + (pb * rb) + (pc * rc), -1.0, 1.0)
    ang_deg = np.degrees(np.arccos(dot))
    invalid = (n_vals <= 1) | np.isnan(ang_deg)
    return np.where(invalid, fallback_val, ang_deg).astype(np.float32)


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
    ensure_file_exists(input_img)

    with ExitStack() as stack:
        src = stack.enter_context(rasterio.open(input_img, "r"))
        grid = IntersectionGrid.from_dataset(src)
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
                        math.hypot(src.width, src.height)
                        * 3.0
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
