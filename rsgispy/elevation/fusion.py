"""
The elevation.fusion module provides digital elevation model (DEM) fusion
functions using ``rasterio`` and ``numpy``.
"""

from contextlib import ExitStack

import numpy as np
import rasterio

import rsgispy
from rsgispy.imageprocess import (
    IntersectionGrid,
    ensure_file_exists,
    open_output_dataset,
)

__all__ = [
    "fft_dem_fusion",
]


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
    ensure_file_exists(high_res_dem_img, "High-res DEM image")
    ensure_file_exists(low_res_dem_img, "Low-res DEM image")

    with rasterio.open(high_res_dem_img, "r") as high_src:
        high_arr = high_src.read(1, out_dtype=np.float64)
        grid = IntersectionGrid.from_dataset(high_src)

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
