"""
The elevation.terrain module provides slope and aspect terrain derivative
functions for digital elevation models (DEMs) using ``rasterio`` and ``numpy``.
"""

from contextlib import ExitStack

import numpy as np
import rasterio

import rsgispy
from rsgispy.elevation._helpers import (
    calc_slope_aspect_from_diffs as _calc_slope_aspect_from_diffs,
    process_pxl_res_dem_3x3 as _process_pxl_res_dem_3x3,
    process_single_dem_3x3 as _process_single_dem_3x3,
    validate_slope_unit as _validate_slope_unit,
)
from rsgispy.imageprocess import (
    create_output_from_dataset,
    ensure_file_exists,
    iter_dataset_windows,
)

__all__ = [
    "slope",
    "slope_pxl_res_img",
    "aspect",
    "aspect_pxl_res_img",
    "catagorise_aspect",
]


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
    ensure_file_exists(input_img)

    with ExitStack() as stack:
        src = stack.enter_context(rasterio.open(input_img, "r"))
        dst = create_output_from_dataset(
            stack=stack,
            src=src,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype="uint8",
            count=1,
        )

        for out_win in iter_dataset_windows(src.height, src.width):
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
