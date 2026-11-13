"""
The imageutils.bands module provides raster band selection, band stacking, and
Shepherd & Dymond local linear band sharpening using ``rasterio`` and ``numpy``.
"""

from contextlib import ExitStack
import math
from typing import Dict, List, Optional, Sequence

import numpy as np
import rasterio
from rasterio.windows import Window

import rsgispy
from rsgispy.imageprocess import (
    cast_result_block,
    compute_spatial_intersection,
    create_output_from_dataset,
    ensure_file_exists,
    iter_dataset_windows,
    open_multi_image_datasets,
    open_output_dataset,
    validate_band_index,
    validate_crs_and_res,
)

__all__ = [
    "select_img_bands",
    "stack_img_bands",
    "SharpBandInfo",
    "sharpen_low_res_bands",
]


class SharpBandInfo:  # pylint: disable=too-few-public-methods
    """
    Configuration record describing an input image band for
    :func:`sharpen_low_res_bands`.

    :param band: 1-based band index within the input image.
    :param status: Band role constant: :data:`rsgispy.SHARP_RES_IGNORE`,
                   :data:`rsgispy.SHARP_RES_LOW`, or
                   :data:`rsgispy.SHARP_RES_HIGH`.
    :param name: Optional output band name.
    """

    def __init__(
        self,
        band: Optional[int] = None,
        status: Optional[int] = None,
        name: Optional[str] = None,
    ):
        self.band = band
        self.status = status
        self.name = name

    def __repr__(self) -> str:
        return (
            f"SharpBandInfo(band={self.band!r}, status={self.status!r}, "
            f"name={self.name!r})"
        )


def select_img_bands(
    input_img: str,
    output_img: str,
    gdalformat: str,
    datatype: int,
    bands: Sequence[int],
) -> None:
    """
    Copy selected 1-based ``bands`` from ``input_img`` into a new raster
    ``output_img``.

    :param input_img: Path to the input image file.
    :param output_img: Path to the output image file.
    :param gdalformat: Output GDAL format name (e.g., ``'KEA'``, ``'GTIFF'``).
    :param datatype: ``rsgispy.TYPE_*`` constant for the output image.
    :param bands: Sequence of 1-based band indices to extract in order.

    .. code:: python

        import rsgispy
        import rsgispy.imageutils.bands

        rsgispy.imageutils.bands.select_img_bands(
            "in_img.tif", "out_img.tif", "GTIFF", rsgispy.TYPE_16UINT, [1, 3, 5]
        )
    """
    # pylint: disable=too-many-locals
    ensure_file_exists(input_img)
    if not bands:
        raise rsgispy.RSGISPyException(
            "bands must contain at least one 1-based band index."
        )
    band_list = [int(b) for b in bands]

    out_rio_dtype = rsgispy.get_rasterio_datatype(datatype)
    out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(datatype))

    with ExitStack() as stack:
        src = stack.enter_context(rasterio.open(input_img, "r"))
        for bidx in band_list:
            validate_band_index(bidx, src.count, input_img)

        dst = create_output_from_dataset(
            stack=stack,
            src=src,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype=out_rio_dtype,
            count=len(band_list),
        )
        if src.nodata is not None:
            dst.nodata = src.nodata
        for out_idx, src_idx in enumerate(band_list, start=1):
            desc = src.descriptions[src_idx - 1]
            if desc:
                dst.set_band_description(out_idx, desc)

        for win in iter_dataset_windows(src.height, src.width):
            blk = np.stack(
                [src.read(bidx, window=win) for bidx in band_list],
                axis=0,
            )
            dst.write(
                cast_result_block(
                    blk, int(win.height), int(win.width), out_np_dtype
                ),
                window=win,
            )


def stack_img_bands(
    input_imgs: Sequence[str],
    band_names: Optional[Sequence[str]],
    output_img: str,
    skip_value: Optional[float],
    no_data_val: Optional[float],
    gdalformat: str,
    datatype: int,
) -> None:
    """
    Stack all bands from ``input_imgs`` over their spatial intersection into a
    single multi-band output image.

    If ``skip_value`` is provided, any pixel where at least one input band
    equals ``skip_value`` has all output bands at that pixel set to
    ``no_data_val``.

    :param input_imgs: Sequence of input image file paths.
    :param band_names: Optional sequence of output band names (or ``None``).
    :param output_img: Path to the output stacked image file.
    :param skip_value: Optional pixel value treated as invalid across inputs.
    :param no_data_val: No-data value written where pixels are skipped.
    :param gdalformat: Output GDAL format name (e.g., ``'KEA'``, ``'GTIFF'``).
    :param datatype: ``rsgispy.TYPE_*`` constant for the output image.

    .. code:: python

        import rsgispy
        import rsgispy.imageutils.bands

        rsgispy.imageutils.bands.stack_img_bands(
            ["b1.tif", "b2.tif", "b3.tif"],
            ["Blue", "Green", "Red"],
            "stacked.tif",
            None,
            0.0,
            "GTIFF",
            rsgispy.TYPE_16UINT,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals,too-many-branches
    out_rio_dtype = rsgispy.get_rasterio_datatype(datatype)
    out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(datatype))
    skip_is_nan = skip_value is not None and math.isnan(float(skip_value))

    with ExitStack() as stack:
        datasets, paths = open_multi_image_datasets(stack, input_imgs)
        first_ds = datasets[paths[0]]
        validate_crs_and_res(datasets, first_ds)
        grid = compute_spatial_intersection(datasets)

        total_bands = sum(int(datasets[p].count) for p in paths)
        if band_names is not None and len(band_names) != total_bands:
            raise rsgispy.RSGISPyException(
                f"Length of band_names ({len(band_names)}) must match total "
                f"stacked band count ({total_bands})."
            )

        dst = open_output_dataset(
            stack=stack,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype=out_rio_dtype,
            grid=grid,
            count=total_bands,
            output_exists=False,
        )
        if band_names is not None:
            for idx, bname in enumerate(band_names, start=1):
                dst.set_band_description(idx, str(bname))

        for dst_win in iter_dataset_windows(grid.height, grid.width):
            col_off, row_off = int(dst_win.col_off), int(dst_win.row_off)
            win_w, win_h = int(dst_win.width), int(dst_win.height)
            band_arrays = []
            for path in paths:
                ds = datasets[path]
                off_x, off_y = grid.offsets[path]
                src_win = Window(
                    off_x + col_off, off_y + row_off, win_w, win_h
                )
                band_arrays.append(ds.read(window=src_win))

            stacked = np.concatenate(band_arrays, axis=0)
            if skip_value is not None:
                if skip_is_nan:
                    skip_mask = np.isnan(stacked).any(axis=0)
                else:
                    skip_mask = (stacked == skip_value).any(axis=0)
                if np.any(skip_mask):
                    fill_val = 0.0 if no_data_val is None else no_data_val
                    stacked = stacked.copy()
                    stacked[:, skip_mask] = fill_val

            dst.write(
                cast_result_block(stacked, win_h, win_w, out_np_dtype),
                window=dst_win,
            )


def _read_padded_block(
    src: rasterio.DatasetReader,
    band_idx: int,
    win: Window,
    pad: int,
) -> np.ndarray:
    """Read a band block with ``pad`` halo pixels, edge-padded at boundaries."""
    # pylint: disable=too-many-locals
    row_off, col_off = int(win.row_off), int(win.col_off)
    win_h, win_w = int(win.height), int(win.width)
    r0, r1 = row_off - pad, row_off + win_h + pad
    c0, c1 = col_off - pad, col_off + win_w + pad
    cr0, cr1 = max(0, r0), min(src.height, r1)
    cc0, cc1 = max(0, c0), min(src.width, c1)
    arr = src.read(
        band_idx,
        window=Window(cc0, cr0, cc1 - cc0, cr1 - cr0),
        out_dtype=np.float64,
    )
    pads = ((cr0 - r0, r1 - cr1), (cc0 - c0, c1 - cc1))
    if any(p > 0 for pair in pads for p in pair):
        arr = np.pad(arr, pads, mode="edge")
    return arr


def _sharpen_single_high_band(
    high_halo: np.ndarray,
    low_halo: np.ndarray,
    valid_halo: np.ndarray,
    win_size: int,
    no_data_val: float,
    band_min: float,
    band_max: float,
) -> np.ndarray:
    """Vectorized Shepherd & Dymond linear sharpening for one high-res band."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    half = win_size // 2
    win_h = high_halo.shape[0] - (2 * half)
    win_w = high_halo.shape[1] - (2 * half)

    h_center = high_halo[half : half + win_h, half : half + win_w]
    l_center = low_halo[half : half + win_h, half : half + win_w]
    v_center = valid_halo[half : half + win_h, half : half + win_w]

    h_wins = np.lib.stride_tricks.sliding_window_view(
        high_halo, (win_size, win_size)
    ).reshape(win_h, win_w, -1)
    l_wins = np.lib.stride_tricks.sliding_window_view(
        low_halo, (win_size, win_size)
    ).reshape(win_h, win_w, -1)
    v_wins = np.lib.stride_tricks.sliding_window_view(
        valid_halo, (win_size, win_size)
    ).reshape(win_h, win_w, -1)

    v_f = v_wins.astype(np.float64)
    n_valid = v_f.sum(axis=2)
    h_clean = np.where(v_wins, h_wins, 0.0)
    l_clean = np.where(v_wins, l_wins, 0.0)

    sum_h = h_clean.sum(axis=2)
    sum_l = l_clean.sum(axis=2)
    mean_h = np.divide(sum_h, n_valid, out=np.zeros_like(sum_h), where=n_valid > 0)
    mean_l = np.divide(sum_l, n_valid, out=np.zeros_like(sum_l), where=n_valid > 0)

    dh = np.where(v_wins, h_clean - mean_h[..., None], 0.0)
    dl = np.where(v_wins, l_clean - mean_l[..., None], 0.0)
    s_hh = (dh * dh).sum(axis=2)
    s_hl = (dh * dl).sum(axis=2)

    can_fit = v_center & (n_valid >= 3) & (s_hh > 1e-12)
    slope = np.divide(
        s_hl, s_hh, out=np.zeros_like(s_hh), where=can_fit
    )
    pred = mean_l + slope * (h_center - mean_h)
    pred = np.clip(pred, band_min, band_max)

    out = np.where(v_center, l_center, float(no_data_val))
    out = np.where(can_fit, pred, out)
    return out


def _sharpen_multi_high_bands(
    high_halos: Sequence[np.ndarray],
    low_halo: np.ndarray,
    valid_halo: np.ndarray,
    win_size: int,
    no_data_val: float,
    band_min: float,
    band_max: float,
) -> np.ndarray:
    """Local least-squares sharpening when multiple high-res bands are given."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    half = win_size // 2
    win_h = low_halo.shape[0] - (2 * half)
    win_w = low_halo.shape[1] - (2 * half)
    n_high = len(high_halos)

    h_centers = np.stack(
        [arr[half : half + win_h, half : half + win_w] for arr in high_halos],
        axis=2,
    )
    l_center = low_halo[half : half + win_h, half : half + win_w]
    v_center = valid_halo[half : half + win_h, half : half + win_w]

    h_wins = np.stack(
        [
            np.lib.stride_tricks.sliding_window_view(
                arr, (win_size, win_size)
            ).reshape(win_h, win_w, -1)
            for arr in high_halos
        ],
        axis=3,
    )
    l_wins = np.lib.stride_tricks.sliding_window_view(
        low_halo, (win_size, win_size)
    ).reshape(win_h, win_w, -1)
    v_wins = np.lib.stride_tricks.sliding_window_view(
        valid_halo, (win_size, win_size)
    ).reshape(win_h, win_w, -1)

    out = np.where(v_center, l_center, float(no_data_val))
    rows, cols = np.where(v_center)
    for r_idx, c_idx in zip(rows.tolist(), cols.tolist()):
        vmask = v_wins[r_idx, c_idx]
        if int(vmask.sum()) < n_high + 2:
            continue
        x_mat = h_wins[r_idx, c_idx][vmask]
        x_design = np.column_stack(
            [np.ones(x_mat.shape[0], dtype=np.float64), x_mat]
        )
        y_vec = l_wins[r_idx, c_idx][vmask]
        coefs, _, _, _ = np.linalg.lstsq(x_design, y_vec, rcond=None)
        val = float(coefs[0] + np.dot(coefs[1:], h_centers[r_idx, c_idx]))
        out[r_idx, c_idx] = min(max(val, band_min), band_max)
    return out


def _calc_low_band_bounds(
    src: rasterio.DatasetReader,
    low_bands: Sequence[int],
    no_data_val: float,
    nd_is_nan: bool,
) -> Dict[int, List[float]]:
    """Compute valid min/max bounds for each low-resolution band."""
    low_bounds: Dict[int, List[float]] = {}
    for l_band in low_bands:
        arr = src.read(l_band, out_dtype=np.float64)
        vmask = np.isfinite(arr)
        if not nd_is_nan:
            vmask &= arr != no_data_val
        if np.any(vmask):
            low_bounds[l_band] = [
                float(arr[vmask].min()),
                float(arr[vmask].max()),
            ]
        else:
            low_bounds[l_band] = [0.0, 0.0]
    return low_bounds


def sharpen_low_res_bands(
    input_img: str,
    output_img: str,
    band_info: Sequence[SharpBandInfo],
    win_size: int = 7,
    no_data_val: float = 0,
    gdalformat: str = "GTIFF",
    datatype: int = rsgispy.TYPE_32FLOAT,
) -> None:
    """
    Sharpen lower-resolution image bands using local linear fitting against
    higher-resolution bands (Shepherd & Dymond method).

    :param input_img: Path to the input multi-band image (with low-res bands
                      already resampled to the high-res pixel grid).
    :param output_img: Path to the output sharpened image file.
    :param band_info: Sequence of :class:`SharpBandInfo` describing each band's
                      role (``SHARP_RES_LOW``, ``SHARP_RES_HIGH``, or
                      ``SHARP_RES_IGNORE``).
    :param win_size: Odd local regression window size in pixels (Default = ``7``).
    :param no_data_val: No-data value ignored during regression fitting
                        (Default = ``0``).
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param datatype: ``rsgispy.TYPE_*`` constant for the output image
                     (Default = :data:`rsgispy.TYPE_32FLOAT`).

    .. code:: python

        import rsgispy
        import rsgispy.imageutils.bands

        band_info = [
            rsgispy.imageutils.bands.SharpBandInfo(1, rsgispy.SHARP_RES_HIGH, "B2"),
            rsgispy.imageutils.bands.SharpBandInfo(2, rsgispy.SHARP_RES_LOW, "B5"),
        ]
        rsgispy.imageutils.bands.sharpen_low_res_bands(
            "in_img.tif", "sharp_img.tif", band_info
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals,too-many-branches
    ensure_file_exists(input_img)
    if win_size < 3 or win_size % 2 == 0:
        raise rsgispy.RSGISPyException(
            "win_size must be an odd integer greater than or equal to 3."
        )
    if not band_info:
        raise rsgispy.RSGISPyException("band_info must not be empty.")

    high_bands: List[int] = []
    low_bands: List[int] = []
    valid_statuses = (
        rsgispy.SHARP_RES_IGNORE,
        rsgispy.SHARP_RES_LOW,
        rsgispy.SHARP_RES_HIGH,
    )
    half = win_size // 2
    nd_is_nan = math.isnan(float(no_data_val))

    with ExitStack() as stack:
        src = stack.enter_context(rasterio.open(input_img, "r"))
        for binfo in band_info:
            if binfo.band is None or binfo.status not in valid_statuses:
                raise rsgispy.RSGISPyException(
                    f"Invalid SharpBandInfo entry: {binfo!r}."
                )
            validate_band_index(int(binfo.band), src.count, input_img)
            if binfo.status == rsgispy.SHARP_RES_HIGH:
                high_bands.append(int(binfo.band))
            elif binfo.status == rsgispy.SHARP_RES_LOW:
                low_bands.append(int(binfo.band))

        if not high_bands or not low_bands:
            raise rsgispy.RSGISPyException(
                "band_info must contain at least one SHARP_RES_HIGH band and "
                "one SHARP_RES_LOW band."
            )

        low_bounds = _calc_low_band_bounds(
            src, low_bands, no_data_val, nd_is_nan
        )
        dst = create_output_from_dataset(
            stack=stack,
            src=src,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype=rsgispy.get_rasterio_datatype(datatype),
            count=len(band_info),
        )
        for out_idx, binfo in enumerate(band_info, start=1):
            if binfo.name:
                dst.set_band_description(out_idx, str(binfo.name))

        out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(datatype))
        for dst_win in iter_dataset_windows(src.height, src.width):
            high_halos = [
                _read_padded_block(src, hb, dst_win, half) for hb in high_bands
            ]
            high_valid = np.ones_like(high_halos[0], dtype=bool)
            for h_arr in high_halos:
                high_valid &= np.isfinite(h_arr)
                if not nd_is_nan:
                    high_valid &= h_arr != no_data_val

            out_bands = []
            for binfo in band_info:
                src_b = int(binfo.band)
                if binfo.status == rsgispy.SHARP_RES_LOW:
                    l_halo = _read_padded_block(src, src_b, dst_win, half)
                    l_valid = high_valid & np.isfinite(l_halo)
                    if not nd_is_nan:
                        l_valid &= l_halo != no_data_val
                    bmin, bmax = low_bounds[src_b]
                    if len(high_halos) == 1:
                        sharpened = _sharpen_single_high_band(
                            high_halos[0],
                            l_halo,
                            l_valid,
                            win_size,
                            no_data_val,
                            bmin,
                            bmax,
                        )
                    else:
                        sharpened = _sharpen_multi_high_bands(
                            high_halos,
                            l_halo,
                            l_valid,
                            win_size,
                            no_data_val,
                            bmin,
                            bmax,
                        )
                    out_bands.append(sharpened)
                else:
                    out_bands.append(
                        src.read(src_b, window=dst_win, out_dtype=np.float64)
                    )

            dst.write(
                cast_result_block(
                    np.stack(out_bands, axis=0),
                    int(dst_win.height),
                    int(dst_win.width),
                    out_np_dtype,
                ),
                window=dst_win,
            )
