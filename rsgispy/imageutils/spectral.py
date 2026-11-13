"""
The imageutils.spectral module provides spectral smoothing, outlier masking,
polynomial NaN gap-filling, and ZCA noise-whitening functions using
``rasterio``, ``numpy``, and ``scipy``.
"""

from contextlib import ExitStack
from typing import Callable, Optional, Sequence

import numpy as np
import rasterio
from rasterio.windows import Window
import scipy.signal

import rsgispy
from rsgispy.imageprocess import (
    cast_result_block,
    compute_spatial_intersection,
    ensure_file_exists,
    iter_dataset_windows,
    open_multi_image_datasets,
    open_output_dataset,
)
from rsgispy.imageutils.blank import create_copy_img
from rsgispy.imageutils.metadata import get_img_no_data_value
from rsgispy.imageutils.stats import pop_img_stats

__all__ = [
    "spectral_smoothing",
    "mask_outliners_data_values",
    "polyfill_nan_data_values",
    "whiten_image",
]


def _apply_spectral_pixel_func(
    input_img: str,
    valid_msk_img: str,
    valid_msk_val: int,
    output_img: str,
    gdalformat: str,
    datatype: int,
    calc_stats: bool,
    pixel_fn: Callable[[np.ndarray], np.ndarray],
) -> None:
    """Apply a per-pixel spectral transform over valid mask blocks."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    ensure_file_exists(input_img)
    ensure_file_exists(valid_msk_img, label="Valid mask image")

    in_no_data = get_img_no_data_value(input_img)
    fill_val = 0.0 if in_no_data is None else float(in_no_data)
    out_rio_dtype = rsgispy.get_rasterio_datatype(datatype)
    out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(datatype))

    with ExitStack() as stack:
        datasets, paths = open_multi_image_datasets(
            stack, [input_img, valid_msk_img]
        )
        img_ds = datasets[paths[0]]
        msk_ds = datasets[paths[1]]
        grid = compute_spatial_intersection(datasets)
        img_off_x, img_off_y = grid.offsets[paths[0]]
        msk_off_x, msk_off_y = grid.offsets[paths[1]]
        n_bands = int(img_ds.count)

        dst = open_output_dataset(
            stack=stack,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype=out_rio_dtype,
            grid=grid,
            count=n_bands,
            output_exists=False,
        )
        if in_no_data is not None:
            dst.nodata = in_no_data
        for bidx in range(1, n_bands + 1):
            desc = img_ds.descriptions[bidx - 1]
            if desc:
                dst.set_band_description(bidx, desc)

        for dst_win in iter_dataset_windows(grid.height, grid.width):
            col_off, row_off = int(dst_win.col_off), int(dst_win.row_off)
            win_w, win_h = int(dst_win.width), int(dst_win.height)
            img_win = Window(
                img_off_x + col_off, img_off_y + row_off, win_w, win_h
            )
            msk_win = Window(
                msk_off_x + col_off, msk_off_y + row_off, win_w, win_h
            )

            msk_flat = msk_ds.read(1, window=msk_win).ravel()
            valid_idx = msk_flat == valid_msk_val
            if np.any(valid_idx):
                img_blk = img_ds.read(window=img_win, out_dtype=np.float64)
                img_flat = np.moveaxis(img_blk, 0, 2).reshape(-1, n_bands)
                out_flat = np.full(
                    (img_flat.shape[0], n_bands),
                    fill_val,
                    dtype=np.float64,
                )
                out_flat[valid_idx] = pixel_fn(img_flat[valid_idx])
                out_blk = np.moveaxis(
                    out_flat.reshape((win_h, win_w, n_bands)), 2, 0
                )
            else:
                out_blk = np.full(
                    (n_bands, win_h, win_w),
                    fill_val,
                    dtype=np.float64,
                )

            dst.write(
                cast_result_block(out_blk, win_h, win_w, out_np_dtype),
                window=dst_win,
            )

    if calc_stats:
        pop_img_stats(
            output_img,
            no_data_val=in_no_data,
            calc_pyramids=True,
        )


def spectral_smoothing(
    input_img: str,
    valid_msk_img: str,
    valid_msk_val: int,
    output_img: str,
    win_len: int = 5,
    polyorder: int = 3,
    gdalformat: str = "GTIFF",
    datatype: int = rsgispy.TYPE_32FLOAT,
    calc_stats: bool = True,
) -> None:
    """
    Smooth spectra along the band axis using a Savitzky-Golay filter for pixels
    matching ``valid_msk_val`` in ``valid_msk_img``.

    :param input_img: Path to the input multi-band image file.
    :param valid_msk_img: Path to the valid-region mask image file.
    :param valid_msk_val: Mask value identifying valid pixels.
    :param output_img: Path to the output smoothed image file.
    :param win_len: Odd Savitzky-Golay window length (Default = ``5``).
    :param polyorder: Polynomial order strictly less than ``win_len`` (Default = ``3``).
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param datatype: ``rsgispy.TYPE_*`` constant for the output image
                     (Default = :data:`rsgispy.TYPE_32FLOAT`).
    :param calc_stats: If ``True``, populate image statistics and pyramids
                       (Default = ``True``).
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    if win_len < 3 or win_len % 2 == 0:
        raise rsgispy.RSGISPyException(
            "win_len must be an odd integer greater than or equal to 3."
        )
    if polyorder < 0 or polyorder >= win_len:
        raise rsgispy.RSGISPyException(
            "polyorder must be non-negative and strictly less than win_len."
        )

    def _smooth_fn(pxls: np.ndarray) -> np.ndarray:
        return scipy.signal.savgol_filter(
            pxls, int(win_len), int(polyorder), axis=1
        )

    _apply_spectral_pixel_func(
        input_img=input_img,
        valid_msk_img=valid_msk_img,
        valid_msk_val=valid_msk_val,
        output_img=output_img,
        gdalformat=gdalformat,
        datatype=datatype,
        calc_stats=calc_stats,
        pixel_fn=_smooth_fn,
    )


def mask_outliners_data_values(
    input_img: str,
    valid_msk_img: str,
    valid_msk_val: int,
    output_img: str,
    lower: float = 5,
    upper: float = 95,
    gdalformat: str = "GTIFF",
    calc_stats: bool = True,
) -> None:
    """
    Mask per-pixel spectral outlier values outside the ``[lower, upper]``
    percentile range of each pixel's spectrum by setting them to ``NaN`` in a
    float32 output image.

    :param input_img: Path to the input multi-band image file.
    :param valid_msk_img: Path to the valid-region mask image file.
    :param valid_msk_val: Mask value identifying valid pixels.
    :param output_img: Path to the output float32 image file.
    :param lower: Lower percentile threshold in ``[0, 100]`` (Default = ``5``).
    :param upper: Upper percentile threshold in ``[0, 100]`` (Default = ``95``).
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param calc_stats: If ``True``, populate image statistics and pyramids
                       (Default = ``True``).
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    if not 0.0 <= float(lower) <= float(upper) <= 100.0:
        raise rsgispy.RSGISPyException(
            "lower and upper percentiles must satisfy 0 <= lower <= upper <= 100."
        )

    def _outlier_fn(pxls: np.ndarray) -> np.ndarray:
        out = pxls.copy()
        pcts = np.nanpercentile(out, [lower, upper], axis=1)
        p_low = pcts[0][:, None]
        p_high = pcts[1][:, None]
        out[(out < p_low) | (out > p_high)] = np.nan
        return out

    _apply_spectral_pixel_func(
        input_img=input_img,
        valid_msk_img=valid_msk_img,
        valid_msk_val=valid_msk_val,
        output_img=output_img,
        gdalformat=gdalformat,
        datatype=rsgispy.TYPE_32FLOAT,
        calc_stats=calc_stats,
        pixel_fn=_outlier_fn,
    )


def polyfill_nan_data_values(
    input_img: str,
    band_vals: Sequence[float],
    valid_msk_img: str,
    valid_msk_val: int,
    output_img: str,
    polyorder: int = 3,
    mean_abs_diff: Optional[float] = None,
    gdalformat: str = "GTIFF",
    calc_stats: bool = True,
) -> None:
    """
    Fill missing (``NaN``) band values along each pixel's spectral profile using
    a polynomial fit against ``band_vals``.

    :param input_img: Path to the input multi-band image file.
    :param band_vals: Sequence of coordinate values (e.g., wavelengths) for each band.
    :param valid_msk_img: Path to the valid-region mask image file.
    :param valid_msk_val: Mask value identifying valid pixels.
    :param output_img: Path to the output float32 image file.
    :param polyorder: Polynomial degree used for fitting (Default = ``3``).
    :param mean_abs_diff: Optional maximum deviation from the pixel's finite mean;
                          predicted values exceeding this threshold are replaced
                          by the pixel mean.
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param calc_stats: If ``True``, populate image statistics and pyramids
                       (Default = ``True``).
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    band_arr = np.asarray(band_vals, dtype=np.float64)
    if band_arr.ndim != 1 or band_arr.size == 0:
        raise rsgispy.RSGISPyException(
            "band_vals must be a non-empty 1D sequence of numeric values."
        )

    def _polyfill_fn(pxls: np.ndarray) -> np.ndarray:
        if pxls.shape[1] != band_arr.shape[0]:
            raise rsgispy.RSGISPyException(
                f"band_vals length ({band_arr.shape[0]}) does not match "
                f"image band count ({pxls.shape[1]})."
            )
        out = pxls.copy()
        for idx in range(out.shape[0]):
            row = out[idx]
            if not np.isnan(row).any():
                continue
            finite_msk = np.isfinite(row)
            if int(finite_msk.sum()) > polyorder:
                coefs = np.polyfit(
                    band_arr[finite_msk], row[finite_msk], deg=polyorder
                )
                pred = np.polyval(coefs, band_arr)
                if mean_abs_diff is not None:
                    pxl_mean = float(np.nanmean(row))
                    pred[np.abs(pred - pxl_mean) > mean_abs_diff] = pxl_mean
                row[~finite_msk] = pred[~finite_msk]
                out[idx] = row
            else:
                out[idx] = np.nan
        return out

    _apply_spectral_pixel_func(
        input_img=input_img,
        valid_msk_img=valid_msk_img,
        valid_msk_val=valid_msk_val,
        output_img=output_img,
        gdalformat=gdalformat,
        datatype=rsgispy.TYPE_32FLOAT,
        calc_stats=calc_stats,
        pixel_fn=_polyfill_fn,
    )


def whiten_image(
    input_img: str,
    valid_msk_img: str,
    valid_msk_val: int,
    output_img: str,
    gdalformat: str,
) -> None:
    """
    Apply ZCA noise whitening to ``input_img`` over pixels matching
    ``valid_msk_val`` in ``valid_msk_img`` so valid spectra have unit variance
    and zero band-to-band correlation.

    :param input_img: Path to the input multi-band image file.
    :param valid_msk_img: Path to the single-band valid-region mask image file.
    :param valid_msk_val: Mask value identifying valid pixels.
    :param output_img: Path to the output float32 whitened image file.
    :param gdalformat: Output GDAL format name (e.g., ``'KEA'``, ``'GTIFF'``).
    """
    # pylint: disable=too-many-locals
    ensure_file_exists(input_img)
    ensure_file_exists(valid_msk_img, label="Valid mask image")

    with rasterio.open(valid_msk_img, "r") as msk_src:
        if msk_src.count != 1:
            raise rsgispy.RSGISPyException(
                "Valid mask only expected to have a single band."
            )
        x_msk_size, y_msk_size = msk_src.width, msk_src.height
        vld_flat = msk_src.read(1).ravel()

    with rasterio.open(input_img, "r") as img_src:
        n_bands = img_src.count
        x_size, y_size = img_src.width, img_src.height
        if x_msk_size != x_size or y_msk_size != y_size:
            raise rsgispy.RSGISPyException(
                "Mask and input image dimensions do not match."
            )
        img_arr = img_src.read(out_dtype=np.float32)

    valid_mask = vld_flat == valid_msk_val
    n_valid = int(valid_mask.sum())
    if n_valid < 2:
        raise rsgispy.RSGISPyException(
            "At least 2 valid pixels are required to compute the covariance matrix."
        )

    img_flat = np.moveaxis(img_arr, 0, 2).reshape(-1, n_bands)
    valid_data = img_flat[valid_mask].astype(np.float64)
    centered = valid_data - valid_data.mean(axis=0, keepdims=True)
    cov = (centered.T @ centered) / (n_valid - 1)
    _, s_vals, vh_mat = np.linalg.svd(cov)
    safe_s = np.where(s_vals > 1e-12, s_vals, 1.0)
    inv_sqrt = np.where(s_vals > 1e-12, safe_s**-0.5, 0.0)
    whiten_mat = vh_mat.T @ np.diag(inv_sqrt) @ vh_mat
    whitened = (centered @ whiten_mat).astype(np.float32)

    create_copy_img(
        input_img, output_img, n_bands, 0, gdalformat, rsgispy.TYPE_32FLOAT
    )
    out_flat = np.zeros((vld_flat.shape[0], n_bands), dtype=np.float32)
    out_flat[valid_mask] = whitened
    out_cube = np.moveaxis(
        out_flat.reshape((y_size, x_size, n_bands)), 2, 0
    )
    with rasterio.open(output_img, "r+") as dst:
        dst.write(out_cube)
