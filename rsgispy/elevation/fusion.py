"""
The elevation.fusion module provides digital elevation model (DEM) fusion
functions using ``rasterio`` and ``numpy``.
"""

from contextlib import ExitStack
from typing import Optional

import numpy as np
import rasterio

try:
    from scipy.ndimage import distance_transform_edt as _distance_transform_edt
except ImportError:
    _distance_transform_edt = None

import rsgispy
from rsgispy.imageprocess import (
    IntersectionGrid,
    build_valid_mask,
    cast_result_block,
    ensure_file_exists,
    open_output_dataset,
)
from rsgispy.imageutils.stats import pop_img_stats

__all__ = [
    "fft_dem_fusion",
]


def _fill_invalid_for_fft(arr: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Replace invalid/non-finite cells via nearest valid pixel before FFT."""
    if np.all(valid):
        return arr
    if not np.any(valid):
        return np.zeros_like(arr, dtype=np.float64)
    if _distance_transform_edt is not None:
        indices = _distance_transform_edt(
            ~valid, return_distances=False, return_indices=True
        )
        return arr[tuple(indices)].astype(np.float64, copy=False)
    mean_val = float(np.mean(arr[valid]))
    return np.where(valid, arr, mean_val)


def fft_dem_fusion(
    high_res_dem_img: str,
    low_res_dem_img: str,
    fused_dem_img: str,
    c_size: int = 20,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
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
    :param datatype: Optional ``rsgispy.TYPE_*`` output data type constant
                     (Default = ``rsgispy.TYPE_32FLOAT`` when ``None``).
    :param no_data_val: Output no-data value (Default = ``-9999.0`` when ``None``).
    :param calc_stats: If ``True``, populate band statistics and overview
                       pyramids on ``fused_dem_img`` (Default = ``True``).
    :param use_pbar: Show a progress bar when computing statistics
                     (Default = ``True``).

    .. code:: python

        import rsgispy.elevation.fusion

        rsgispy.elevation.fusion.fft_dem_fusion(
            high_res_dem_img="dem_high.tif",
            low_res_dem_img="dem_low.tif",
            fused_dem_img="dem_fused.tif",
            c_size=20,
            gdalformat="GTIFF",
        )
    """
    # pylint: disable=too-many-arguments,too-many-locals
    if c_size < 1:
        raise rsgispy.RSGISPyException("c_size must be at least 1.")
    ensure_file_exists(high_res_dem_img, "High-res DEM image")
    ensure_file_exists(low_res_dem_img, "Low-res DEM image")

    eff_dtype = rsgispy.TYPE_32FLOAT if datatype is None else int(datatype)
    out_rio_dtype = rsgispy.get_rasterio_datatype(eff_dtype)
    out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(eff_dtype))
    eff_nd = -9999.0 if no_data_val is None else float(no_data_val)

    with rasterio.open(high_res_dem_img, "r") as high_src:
        high_arr = high_src.read(1, out_dtype=np.float64)
        high_nd = high_src.nodata if high_src.nodata is not None else no_data_val
        grid = IntersectionGrid.from_dataset(high_src)

    with rasterio.open(low_res_dem_img, "r") as low_src:
        low_arr = low_src.read(1, out_dtype=np.float64)
        low_nd = low_src.nodata if low_src.nodata is not None else no_data_val

    if high_arr.shape != low_arr.shape:
        raise rsgispy.RSGISPyException(
            f"Input DEMs must have identical dimensions: "
            f"{high_arr.shape} != {low_arr.shape}."
        )

    high_valid = build_valid_mask(high_arr, high_nd)
    low_valid = build_valid_mask(low_arr, low_nd)
    combined_valid = high_valid & low_valid

    high_clean = _fill_invalid_for_fft(high_arr, high_valid)
    low_clean = _fill_invalid_for_fft(low_arr, low_valid)

    fshift_high = np.fft.fftshift(np.fft.fft2(high_clean))
    fshift_low = np.fft.fftshift(np.fft.fft2(low_clean))

    rows, cols = low_arr.shape
    crow, ccol = rows // 2, cols // 2
    r0 = max(0, crow - c_size)
    r1 = min(rows, crow + c_size + 1)
    c0 = max(0, ccol - c_size)
    c1 = min(cols, ccol + c_size + 1)
    fshift_low[r0:r1, c0:c1] = fshift_high[r0:r1, c0:c1]

    fused_float = np.real(np.fft.ifft2(np.fft.ifftshift(fshift_low)))
    if not np.all(combined_valid):
        fused_float = np.where(combined_valid, fused_float, eff_nd)

    img_back = cast_result_block(fused_float, rows, cols, out_np_dtype)

    with ExitStack() as stack:
        dst = open_output_dataset(
            stack=stack,
            output_img=fused_dem_img,
            gdalformat=gdalformat,
            out_rio_dtype=out_rio_dtype,
            grid=grid,
            count=1,
        )
        dst.nodata = eff_nd
        dst.write(img_back, 1)

    if calc_stats:
        pop_img_stats(
            input_img=fused_dem_img,
            no_data_val=eff_nd,
            calc_pyramids=True,
            ignore_cog=gdalformat.upper().strip() == "COG",
            use_pbar=use_pbar,
        )
