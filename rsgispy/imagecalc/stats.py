"""
The imagecalc.stats module provides basic raster band summary statistics,
pixel value counting, and unique value extraction using ``rasterio``.
"""

from typing import Optional, Sequence, Tuple

import numpy as np
import rasterio

import rsgispy
from rsgispy.imageprocess import (
    ensure_file_exists,
    iter_single_image_band_blocks,
)

__all__ = [
    "calc_basic_img_band_stats",
    "count_pxls_of_val",
    "get_unique_values",
]


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
    ensure_file_exists(input_img)

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
    ensure_file_exists(input_img)

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
            [block_arrays[img_band - 1]]
            if img_band is not None
            else block_arrays
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
    ensure_file_exists(input_img)

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
