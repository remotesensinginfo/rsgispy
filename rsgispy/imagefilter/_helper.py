"""
Internal windowed spatial filtering and convolution engine for
:mod:`rsgispy.imagefilter`.
"""

import math
from typing import Callable, Optional, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

import rsgispy
from rsgispy.imageprocess import (
    build_valid_mask,
    ensure_file_exists,
    fits_dtype_range as _fits_dtype,
    resolve_out_nodata as _ip_resolve_out_nodata,
    run_window_halo_blocks,
    validate_odd_window_size,
)
from rsgispy.imageutils.metadata import (
    get_band_names,
    get_img_no_data_value,
    set_band_names,
)
from rsgispy.imageutils.stats import pop_img_stats

StatFunc = Callable[
    [np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    np.ndarray,
]


def validate_filter_size(filter_size: int) -> int:
    """Validate that ``filter_size`` is an odd integer >= 3."""
    return validate_odd_window_size(filter_size, min_size=3, name="filter_size")


def validate_positive(value: float, name: str) -> float:
    """Validate that ``value`` is a finite strictly positive float."""
    try:
        val = float(value)
    except (TypeError, ValueError) as err:
        raise rsgispy.RSGISPyException(
            f"{name} must be a positive number, got {value!r}."
        ) from err
    if not math.isfinite(val) or val <= 0.0:
        raise rsgispy.RSGISPyException(
            f"{name} must be finite and > 0, got {value!r}."
        )
    return val


def validate_kernel(kernel: np.ndarray) -> np.ndarray:
    """Validate a custom 2D odd-dimension finite convolution kernel."""
    arr = np.asarray(kernel, dtype=np.float64)
    if arr.ndim != 2:
        raise rsgispy.RSGISPyException(
            f"kernel must be a 2D array, got shape {arr.shape}."
        )
    kh, kw = arr.shape
    validate_odd_window_size(kh, min_size=3, name="kernel height")
    validate_odd_window_size(kw, min_size=3, name="kernel width")
    if not np.all(np.isfinite(arr)):
        raise rsgispy.RSGISPyException("kernel entries must all be finite.")
    return arr


def _resolve_out_nodata(
    eff_in_nd: Optional[float],
    out_np_dtype: np.dtype,
) -> Optional[float]:
    """Resolve output nodata value preserving input nodata when representable."""
    if eff_in_nd is None or not math.isfinite(float(eff_in_nd)):
        return None
    return _ip_resolve_out_nodata(
        out_np_dtype,
        fallback_nodata=float(eff_in_nd),
        default_float=-9999.0,
        default_int=0.0,
    )


def stat_mean(
    wins: np.ndarray,
    centre: np.ndarray,
    valid: np.ndarray,
    centre_valid: np.ndarray,
) -> np.ndarray:
    """Compute local window mean over valid pixels."""
    del centre
    flat = wins.reshape(wins.shape[0], wins.shape[1], -1)
    vflat = valid.reshape(valid.shape[0], valid.shape[1], -1)
    n_valid = np.sum(vflat, axis=-1)
    sums = np.sum(np.where(vflat, flat, 0.0), axis=-1)
    out = np.full(centre_valid.shape, np.nan, dtype=np.float64)
    ok = centre_valid & (n_valid > 0)
    out[ok] = sums[ok] / n_valid[ok]
    return out


def stat_total(
    wins: np.ndarray,
    centre: np.ndarray,
    valid: np.ndarray,
    centre_valid: np.ndarray,
) -> np.ndarray:
    """Compute local window sum over valid pixels."""
    del centre
    flat = wins.reshape(wins.shape[0], wins.shape[1], -1)
    vflat = valid.reshape(valid.shape[0], valid.shape[1], -1)
    n_valid = np.sum(vflat, axis=-1)
    sums = np.sum(np.where(vflat, flat, 0.0), axis=-1)
    out = np.full(centre_valid.shape, np.nan, dtype=np.float64)
    ok = centre_valid & (n_valid > 0)
    out[ok] = sums[ok]
    return out


def stat_median(
    wins: np.ndarray,
    centre: np.ndarray,
    valid: np.ndarray,
    centre_valid: np.ndarray,
) -> np.ndarray:
    """Compute local window median over valid pixels."""
    del centre
    flat = wins.reshape(wins.shape[0], wins.shape[1], -1)
    vflat = valid.reshape(valid.shape[0], valid.shape[1], -1)
    out = np.full(centre_valid.shape, np.nan, dtype=np.float64)
    if np.all(vflat):
        med = np.median(flat, axis=-1)
        out[centre_valid] = med[centre_valid]
        return out

    has_valid = centre_valid & np.any(vflat, axis=-1)
    if np.any(has_valid):
        sel_vals = np.where(vflat[has_valid], flat[has_valid], np.nan)
        out[has_valid] = np.nanmedian(sel_vals, axis=-1)
    return out


def stat_min(
    wins: np.ndarray,
    centre: np.ndarray,
    valid: np.ndarray,
    centre_valid: np.ndarray,
) -> np.ndarray:
    """Compute local window minimum over valid pixels."""
    del centre
    flat = wins.reshape(wins.shape[0], wins.shape[1], -1)
    vflat = valid.reshape(valid.shape[0], valid.shape[1], -1)
    out = np.full(centre_valid.shape, np.nan, dtype=np.float64)
    has_valid = centre_valid & np.any(vflat, axis=-1)
    if np.any(has_valid):
        vals = np.where(vflat[has_valid], flat[has_valid], np.inf)
        out[has_valid] = np.min(vals, axis=-1)
    return out


def stat_max(
    wins: np.ndarray,
    centre: np.ndarray,
    valid: np.ndarray,
    centre_valid: np.ndarray,
) -> np.ndarray:
    """Compute local window maximum over valid pixels."""
    del centre
    flat = wins.reshape(wins.shape[0], wins.shape[1], -1)
    vflat = valid.reshape(valid.shape[0], valid.shape[1], -1)
    out = np.full(centre_valid.shape, np.nan, dtype=np.float64)
    has_valid = centre_valid & np.any(vflat, axis=-1)
    if np.any(has_valid):
        vals = np.where(vflat[has_valid], flat[has_valid], -np.inf)
        out[has_valid] = np.max(vals, axis=-1)
    return out


def stat_range(
    wins: np.ndarray,
    centre: np.ndarray,
    valid: np.ndarray,
    centre_valid: np.ndarray,
) -> np.ndarray:
    """Compute local window range (max - min) over valid pixels."""
    del centre
    flat = wins.reshape(wins.shape[0], wins.shape[1], -1)
    vflat = valid.reshape(valid.shape[0], valid.shape[1], -1)
    out = np.full(centre_valid.shape, np.nan, dtype=np.float64)
    has_valid = centre_valid & np.any(vflat, axis=-1)
    if np.any(has_valid):
        vmax = np.max(np.where(vflat[has_valid], flat[has_valid], -np.inf), axis=-1)
        vmin = np.min(np.where(vflat[has_valid], flat[has_valid], np.inf), axis=-1)
        out[has_valid] = vmax - vmin
    return out


def stat_mode(
    wins: np.ndarray,
    centre: np.ndarray,
    valid: np.ndarray,
    centre_valid: np.ndarray,
) -> np.ndarray:
    """Compute local window mode over valid pixels with deterministic ties."""
    # pylint: disable=too-many-locals
    del centre
    h, w = centre_valid.shape
    k_len = wins.shape[2] * wins.shape[3]
    flat = wins.reshape(-1, k_len)
    vflat = valid.reshape(-1, k_len)
    cv_flat = centre_valid.ravel()

    out_flat = np.full(h * w, np.nan, dtype=np.float64)
    valid_px = np.flatnonzero(cv_flat & np.any(vflat, axis=1))
    if valid_px.size == 0:
        return out_flat.reshape(h, w)

    chunk_sz = max(1024, min(65536, 4_000_000 // max(1, k_len)))
    for start in range(0, valid_px.size, chunk_sz):
        stop = start + chunk_sz
        idx = valid_px[start:stop]
        v_c = vflat[idx]
        a_c = np.where(v_c, flat[idx], np.inf)
        a_sorted = np.sort(a_c, axis=1)
        v_sorted = np.isfinite(a_sorted)
        n_rows = idx.size
        counts = np.zeros((n_rows, k_len), dtype=np.int32)
        run_start = np.zeros(n_rows, dtype=np.int32)
        run_len = v_sorted[:, 0].astype(np.int32)
        counts[np.arange(n_rows), 0] = run_len
        rows_idx = np.arange(n_rows)
        for col in range(1, k_len):
            same = v_sorted[:, col] & (a_sorted[:, col] == a_sorted[:, col - 1])
            new_valid = v_sorted[:, col] & ~same
            run_start = np.where(new_valid, col, run_start)
            run_len = np.where(same, run_len + 1, np.where(new_valid, 1, 0))
            upd = v_sorted[:, col]
            if np.any(upd):
                counts[rows_idx[upd], run_start[upd]] = run_len[upd]
        best = np.argmax(counts, axis=1)
        out_flat[idx] = a_sorted[rows_idx, best]

    return out_flat.reshape(h, w)


def _window_mean_var(
    wins: np.ndarray, valid: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute exact vectorised window ``(mean, var, n_valid)``."""
    flat = wins.reshape(wins.shape[0], wins.shape[1], -1)
    vflat = valid.reshape(valid.shape[0], valid.shape[1], -1)
    n_valid = np.sum(vflat, axis=-1)
    zeros_masked = np.where(vflat, flat, 0.0)
    sums = np.sum(zeros_masked, axis=-1)
    mean = np.zeros(n_valid.shape, dtype=np.float64)
    ok = n_valid > 0
    mean[ok] = sums[ok] / n_valid[ok]
    diff = np.where(vflat, flat - mean[..., None], 0.0)
    var = np.zeros(n_valid.shape, dtype=np.float64)
    var[ok] = np.sum(diff * diff, axis=-1)[ok] / n_valid[ok]
    return mean, var, n_valid


def stat_stddev(
    wins: np.ndarray,
    centre: np.ndarray,
    valid: np.ndarray,
    centre_valid: np.ndarray,
) -> np.ndarray:
    """Compute local window standard deviation over valid pixels."""
    del centre
    _, var, n_valid = _window_mean_var(wins, valid)
    out = np.full(centre_valid.shape, np.nan, dtype=np.float64)
    ok = centre_valid & (n_valid > 0)
    out[ok] = np.sqrt(np.maximum(var[ok], 0.0))
    return out


def stat_texture_var(
    wins: np.ndarray,
    centre: np.ndarray,
    valid: np.ndarray,
    centre_valid: np.ndarray,
) -> np.ndarray:
    """Compute local window variance (texture variance) over valid pixels."""
    del centre
    _, var, n_valid = _window_mean_var(wins, valid)
    out = np.full(centre_valid.shape, np.nan, dtype=np.float64)
    ok = centre_valid & (n_valid > 0)
    out[ok] = np.maximum(var[ok], 0.0)
    return out


def stat_coeff_of_var(
    wins: np.ndarray,
    centre: np.ndarray,
    valid: np.ndarray,
    centre_valid: np.ndarray,
) -> np.ndarray:
    """Compute local coefficient of variation (std / mean)."""
    del centre
    mean, var, n_valid = _window_mean_var(wins, valid)
    out = np.full(centre_valid.shape, np.nan, dtype=np.float64)
    ok = centre_valid & (n_valid > 0)
    out[ok] = 0.0
    nz = ok & (mean != 0.0)
    out[nz] = np.sqrt(np.maximum(var[nz], 0.0)) / mean[nz]
    return out


def stat_norm_var(
    wins: np.ndarray,
    centre: np.ndarray,
    valid: np.ndarray,
    centre_valid: np.ndarray,
) -> np.ndarray:
    """Compute local normalised variance (var / mean^2)."""
    del centre
    mean, var, n_valid = _window_mean_var(wins, valid)
    out = np.full(centre_valid.shape, np.nan, dtype=np.float64)
    ok = centre_valid & (n_valid > 0)
    out[ok] = 0.0
    nz = ok & (mean != 0.0)
    out[nz] = np.maximum(var[nz], 0.0) / (mean[nz] * mean[nz])
    return out


def stat_norm_var_sqrt(
    wins: np.ndarray,
    centre: np.ndarray,
    valid: np.ndarray,
    centre_valid: np.ndarray,
) -> np.ndarray:
    """Compute square root of normalised variance (std / |mean|)."""
    del centre
    mean, var, n_valid = _window_mean_var(wins, valid)
    out = np.full(centre_valid.shape, np.nan, dtype=np.float64)
    ok = centre_valid & (n_valid > 0)
    out[ok] = 0.0
    nz = ok & (mean != 0.0)
    out[nz] = np.sqrt(np.maximum(var[nz], 0.0)) / np.abs(mean[nz])
    return out


def stat_norm_var_ln(
    wins: np.ndarray,
    centre: np.ndarray,
    valid: np.ndarray,
    centre_valid: np.ndarray,
) -> np.ndarray:
    """Compute natural logarithm ``ln(1 + var / mean^2)``."""
    nv = stat_norm_var(wins, centre, valid, centre_valid)
    ok = np.isfinite(nv)
    out = np.full(nv.shape, np.nan, dtype=np.float64)
    out[ok] = np.log1p(np.maximum(nv[ok], 0.0))
    return out


def stat_mean_diff(
    wins: np.ndarray,
    centre: np.ndarray,
    valid: np.ndarray,
    centre_valid: np.ndarray,
) -> np.ndarray:
    """Compute mean signed difference ``mean(window - centre)``."""
    flat = wins.reshape(wins.shape[0], wins.shape[1], -1)
    vflat = valid.reshape(valid.shape[0], valid.shape[1], -1)
    n_valid = np.sum(vflat, axis=-1)
    diff = np.where(vflat, flat - centre[..., None], 0.0)
    out = np.full(centre_valid.shape, np.nan, dtype=np.float64)
    ok = centre_valid & (n_valid > 0)
    out[ok] = np.sum(diff, axis=-1)[ok] / n_valid[ok]
    return out


def stat_mean_diff_abs(
    wins: np.ndarray,
    centre: np.ndarray,
    valid: np.ndarray,
    centre_valid: np.ndarray,
) -> np.ndarray:
    """Compute mean absolute difference ``mean(|window - centre|)``."""
    flat = wins.reshape(wins.shape[0], wins.shape[1], -1)
    vflat = valid.reshape(valid.shape[0], valid.shape[1], -1)
    n_valid = np.sum(vflat, axis=-1)
    diff = np.where(vflat, np.abs(flat - centre[..., None]), 0.0)
    out = np.full(centre_valid.shape, np.nan, dtype=np.float64)
    ok = centre_valid & (n_valid > 0)
    out[ok] = np.sum(diff, axis=-1)[ok] / n_valid[ok]
    return out


def stat_total_diff(
    wins: np.ndarray,
    centre: np.ndarray,
    valid: np.ndarray,
    centre_valid: np.ndarray,
) -> np.ndarray:
    """Compute sum of signed differences ``sum(window - centre)``."""
    flat = wins.reshape(wins.shape[0], wins.shape[1], -1)
    vflat = valid.reshape(valid.shape[0], valid.shape[1], -1)
    n_valid = np.sum(vflat, axis=-1)
    diff = np.where(vflat, flat - centre[..., None], 0.0)
    out = np.full(centre_valid.shape, np.nan, dtype=np.float64)
    ok = centre_valid & (n_valid > 0)
    out[ok] = np.sum(diff, axis=-1)[ok]
    return out


def stat_total_diff_abs(
    wins: np.ndarray,
    centre: np.ndarray,
    valid: np.ndarray,
    centre_valid: np.ndarray,
) -> np.ndarray:
    """Compute sum of absolute differences ``sum(|window - centre|)``."""
    flat = wins.reshape(wins.shape[0], wins.shape[1], -1)
    vflat = valid.reshape(valid.shape[0], valid.shape[1], -1)
    n_valid = np.sum(vflat, axis=-1)
    diff = np.where(vflat, np.abs(flat - centre[..., None]), 0.0)
    out = np.full(centre_valid.shape, np.nan, dtype=np.float64)
    ok = centre_valid & (n_valid > 0)
    out[ok] = np.sum(diff, axis=-1)[ok]
    return out


def stat_kuwahara(
    wins: np.ndarray,
    centre: np.ndarray,
    valid: np.ndarray,
    centre_valid: np.ndarray,
) -> np.ndarray:
    """Compute 4-quadrant edge-preserving Kuwahara filter."""
    # pylint: disable=too-many-locals
    del centre
    size = wins.shape[2]
    half = size // 2
    q_sz = half + 1
    slices = [
        (slice(0, q_sz), slice(0, q_sz)),
        (slice(0, q_sz), slice(half, size)),
        (slice(half, size), slice(0, q_sz)),
        (slice(half, size), slice(half, size)),
    ]
    q_means = []
    q_vars = []
    for rs, cs in slices:
        qm, qv, qn = _window_mean_var(wins[:, :, rs, cs], valid[:, :, rs, cs])
        qv_masked = np.where(qn > 0, qv, np.inf)
        q_means.append(qm)
        q_vars.append(qv_masked)

    means_stack = np.stack(q_means, axis=0)
    vars_stack = np.stack(q_vars, axis=0)
    best_q = np.argmin(vars_stack, axis=0)
    rr, cc = np.indices(centre_valid.shape)
    best_mean = means_stack[best_q, rr, cc]
    has_q = np.any(np.isfinite(vars_stack), axis=0)

    out = np.full(centre_valid.shape, np.nan, dtype=np.float64)
    ok = centre_valid & has_q
    out[ok] = best_mean[ok]
    return out


def make_lee_stat_fn(n_looks: float) -> StatFunc:
    """Create a Lee (1980) SAR adaptive speckle filter reducer for ``n_looks``."""
    cu2 = 1.0 / float(n_looks)

    def _stat_lee(
        wins: np.ndarray,
        centre: np.ndarray,
        valid: np.ndarray,
        centre_valid: np.ndarray,
    ) -> np.ndarray:
        mean, var, n_valid = _window_mean_var(wins, valid)
        out = np.full(centre_valid.shape, np.nan, dtype=np.float64)
        ok = centre_valid & (n_valid > 0)
        out[ok] = mean[ok]
        nz = ok & (mean != 0.0) & (var > 0.0)
        if np.any(nz):
            ci2 = var[nz] / (mean[nz] * mean[nz])
            weights = np.clip((ci2 - cu2) / np.maximum(ci2, 1e-12), 0.0, 1.0)
            out[nz] = mean[nz] + weights * (centre[nz] - mean[nz])
        return out

    return _stat_lee


def sobel_kernels() -> Tuple[np.ndarray, np.ndarray]:
    """Return horizontal and vertical 3x3 Sobel convolution kernels."""
    kx = np.array(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=np.float64
    )
    ky = np.array(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=np.float64
    )
    return kx, ky


def prewitt_kernels() -> Tuple[np.ndarray, np.ndarray]:
    """Return horizontal and vertical 3x3 Prewitt convolution kernels."""
    kx = np.array(
        [[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]], dtype=np.float64
    )
    ky = np.array(
        [[-1.0, -1.0, -1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float64
    )
    return kx, ky


def gaussian_kernel(
    filter_size: int,
    stddev_x: float,
    stddev_y: float,
    filter_angle: float,
    deriv_order: int = 0,
) -> np.ndarray:
    """
    Build a rotated 2D anisotropic Gaussian smoothing or derivative kernel
    (``deriv_order`` in ``{0, 1, 2}``), with ``filter_angle`` in degrees.
    """
    # pylint: disable=too-many-locals
    size = validate_filter_size(filter_size)
    sx = validate_positive(stddev_x, "stddev_x")
    sy = validate_positive(stddev_y, "stddev_y")
    theta = math.radians(float(filter_angle))

    half = size // 2
    coords = np.arange(-half, half + 1, dtype=np.float64)
    dx, dy = np.meshgrid(coords, coords)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    u = dx * cos_t + dy * sin_t
    v = -dx * sin_t + dy * cos_t

    gauss = np.exp(-0.5 * ((u * u) / (sx * sx) + (v * v) / (sy * sy)))
    gauss /= np.sum(gauss)
    if deriv_order == 0:
        return gauss
    if deriv_order == 1:
        k1 = (u / (sx * sx)) * gauss
        k1 -= np.mean(k1)
        return k1
    if deriv_order == 2:
        k2 = (((u * u) - (sx * sx)) / (sx**4)) * gauss
        k2 -= np.mean(k2)
        return k2
    raise rsgispy.RSGISPyException(
        f"Unsupported Gaussian derivative order: {deriv_order}."
    )


def laplacian_kernel(filter_size: int, stddev: float) -> np.ndarray:
    """Build a zero-sum 2D Laplacian-of-Gaussian (LoG) kernel."""
    size = validate_filter_size(filter_size)
    sig = validate_positive(stddev, "stddev")
    half = size // 2
    coords = np.arange(-half, half + 1, dtype=np.float64)
    dx, dy = np.meshgrid(coords, coords)
    r2 = dx * dx + dy * dy
    s2 = sig * sig
    log_k = ((r2 - 2.0 * s2) / (s2 * s2)) * np.exp(-r2 / (2.0 * s2))
    log_k /= 2.0 * math.pi * s2
    log_k -= np.mean(log_k)
    return log_k


def _convolve_tile_with_mask(
    wins: np.ndarray,
    centre: np.ndarray,
    valid: np.ndarray,
    centre_valid: np.ndarray,
    kernel: np.ndarray,
    zero_dc: bool,
) -> np.ndarray:
    """Convolve a ``(h, w, kh, kw)`` window array with a 2D kernel."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    out = np.full(centre_valid.shape, np.nan, dtype=np.float64)
    if np.all(valid):
        conv = np.einsum("ijkl,kl->ij", wins, kernel)
        out[centre_valid] = conv[centre_valid]
        return out

    if not zero_dc and np.all(kernel >= 0.0):
        k_sum = float(np.sum(kernel))
        denom = np.einsum("ijkl,kl->ij", valid.astype(np.float64), kernel)
        numer = np.einsum("ijkl,kl->ij", np.where(valid, wins, 0.0), kernel)
        ok = centre_valid & (denom > 0.0)
        out[ok] = (numer[ok] / denom[ok]) * k_sum
        return out

    wins_filled = np.where(valid, wins, centre[:, :, None, None])
    conv = np.einsum("ijkl,kl->ij", wins_filled, kernel)
    out[centre_valid] = conv[centre_valid]
    return out


def _apply_tile_nd(
    out_blk: np.ndarray, out_nd: Optional[float], out_np_dtype: np.dtype
) -> np.ndarray:
    """Replace NaN values in a tile result with the resolved nodata/fill value."""
    if out_nd is not None and math.isfinite(float(out_nd)):
        return np.where(np.isfinite(out_blk), out_blk, float(out_nd))
    if np.issubdtype(out_np_dtype, np.floating):
        return out_blk
    return np.where(np.isfinite(out_blk), out_blk, 0.0)


def _run_halo_filter(
    input_img: str,
    output_img: str,
    win_shape: Tuple[int, int],
    tile_fn: Callable[
        [np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        np.ndarray,
    ],
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """Shared windowed halo filter runner processing all bands of ``input_img``."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    ensure_file_exists(input_img, "Input image")
    band_names = get_band_names(input_img)
    eff_dtype = rsgispy.TYPE_32FLOAT if datatype is None else int(datatype)
    out_rio_dtype = rsgispy.get_rasterio_datatype(eff_dtype)
    out_np_dtype = rsgispy.get_numpy_datatype(eff_dtype)

    eff_in_nd = (
        float(no_data_val)
        if no_data_val is not None
        else get_img_no_data_value(input_img)
    )
    out_nd = _resolve_out_nodata(eff_in_nd, out_np_dtype)

    kh, kw = win_shape
    pad_y, pad_x = kh // 2, kw // 2

    def _tile_cb(halo: np.ndarray) -> np.ndarray:
        win_h = halo.shape[0] - 2 * pad_y
        win_w = halo.shape[1] - 2 * pad_x
        y_end = pad_y + win_h
        x_end = pad_x + win_w
        valid_halo = build_valid_mask(halo, eff_in_nd)
        centre = halo[pad_y:y_end, pad_x:x_end]
        centre_valid = valid_halo[pad_y:y_end, pad_x:x_end]
        wins = sliding_window_view(halo, (kh, kw))
        v_wins = sliding_window_view(valid_halo, (kh, kw))
        res = tile_fn(wins, centre, v_wins, centre_valid)
        return _apply_tile_nd(res, out_nd, out_np_dtype)

    run_window_halo_blocks(
        input_img=input_img,
        output_img=output_img,
        win_shape=win_shape,
        block_fn=_tile_cb,
        gdalformat=gdalformat,
        out_rio_dtype=out_rio_dtype,
        out_np_dtype=out_np_dtype,
        per_band=True,
        read_dtype=np.float64,
        pad_mode="edge",
        out_no_data_val=out_nd,
        clamp_int=True,
        use_pbar=use_pbar,
    )

    set_band_names(output_img, band_names, feedback=False)
    if calc_stats:
        pop_img_stats(
            output_img,
            no_data_val=out_nd,
            calc_pyramids=True,
            use_pbar=use_pbar,
        )


def run_window_stat_filter(
    input_img: str,
    output_img: str,
    filter_size: int,
    stat_fn: StatFunc,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """Apply a windowed neighbourhood statistical filter to all bands of a raster."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    size = validate_filter_size(filter_size)
    _run_halo_filter(
        input_img=input_img,
        output_img=output_img,
        win_shape=(size, size),
        tile_fn=stat_fn,
        gdalformat=gdalformat,
        datatype=datatype,
        no_data_val=no_data_val,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )


def run_kernel_filter(
    input_img: str,
    output_img: str,
    kernel: np.ndarray,
    zero_dc: bool = False,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """Apply a 2D convolution kernel filter window-by-window to all bands."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    k_arr = validate_kernel(kernel)

    def _tile_conv(
        wins: np.ndarray,
        centre: np.ndarray,
        valid: np.ndarray,
        centre_valid: np.ndarray,
    ) -> np.ndarray:
        return _convolve_tile_with_mask(
            wins, centre, valid, centre_valid, k_arr, zero_dc=zero_dc
        )

    _run_halo_filter(
        input_img=input_img,
        output_img=output_img,
        win_shape=(int(k_arr.shape[0]), int(k_arr.shape[1])),
        tile_fn=_tile_conv,
        gdalformat=gdalformat,
        datatype=datatype,
        no_data_val=no_data_val,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )


def run_gradient_magnitude_filter(
    input_img: str,
    output_img: str,
    kernel_x: np.ndarray,
    kernel_y: np.ndarray,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """Compute single-pass gradient magnitude ``sqrt(Gx^2 + Gy^2)``."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    kx = validate_kernel(kernel_x)
    ky = validate_kernel(kernel_y)
    if kx.shape != ky.shape:
        raise rsgispy.RSGISPyException(
            f"kernel_x shape {kx.shape} must match kernel_y shape {ky.shape}."
        )

    def _tile_grad(
        wins: np.ndarray,
        centre: np.ndarray,
        valid: np.ndarray,
        centre_valid: np.ndarray,
    ) -> np.ndarray:
        gx = _convolve_tile_with_mask(
            wins, centre, valid, centre_valid, kx, zero_dc=True
        )
        gy = _convolve_tile_with_mask(
            wins, centre, valid, centre_valid, ky, zero_dc=True
        )
        return np.hypot(gx, gy)

    _run_halo_filter(
        input_img=input_img,
        output_img=output_img,
        win_shape=(int(kx.shape[0]), int(kx.shape[1])),
        tile_fn=_tile_grad,
        gdalformat=gdalformat,
        datatype=datatype,
        no_data_val=no_data_val,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )


def _make_tile_stat_op(
    source: "rsgispy._io.TileNode",
    win_shape: Tuple[int, int],
    tile_fn: StatFunc,
    *,
    no_data_val: Optional[float] = None,
    border: Optional["rsgispy._io.HaloBorder"] = None,
    name: Optional[str] = None,
) -> "rsgispy._io.WindowOp":
    """Build a :class:`rsgispy._io.WindowOp` running ``tile_fn`` band-by-band."""
    from rsgispy._io import HaloBorder, WindowOp  # pylint: disable=import-outside-toplevel

    kh, kw = win_shape
    pad_y, pad_x = kh // 2, kw // 2

    def _op_fn(padded_3d: np.ndarray) -> np.ndarray:
        n_bands, h_in, w_in = padded_3d.shape
        out_h = h_in - 2 * pad_y
        out_w = w_in - 2 * pad_x
        y_end = pad_y + out_h
        x_end = pad_x + out_w
        out = np.empty((n_bands, out_h, out_w), dtype=np.float64)
        for b_idx in range(n_bands):
            band_halo = padded_3d[b_idx]
            valid_halo = build_valid_mask(band_halo, no_data_val)
            centre = band_halo[pad_y:y_end, pad_x:x_end]
            centre_valid = valid_halo[pad_y:y_end, pad_x:x_end]
            wins = sliding_window_view(band_halo, (kh, kw))
            v_wins = sliding_window_view(valid_halo, (kh, kw))
            res = tile_fn(wins, centre, v_wins, centre_valid)
            if no_data_val is not None:
                res = np.where(np.isfinite(res), res, float(no_data_val))
            out[b_idx] = res
        return out

    return WindowOp(
        source=source,
        radius=(pad_y, pad_x),
        func=_op_fn,
        border=border if border is not None else HaloBorder("edge"),
        name=name,
    )


def window_stat_node(
    source: "rsgispy._io.TileNode",
    filter_size: int,
    stat_fn: StatFunc,
    *,
    no_data_val: Optional[float] = None,
    border: Optional["rsgispy._io.HaloBorder"] = None,
    name: Optional[str] = None,
) -> "rsgispy._io.WindowOp":
    """Build a composable :class:`~rsgispy._io.WindowOp` for a moving-window statistic."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    size = validate_filter_size(filter_size)
    return _make_tile_stat_op(
        source=source,
        win_shape=(size, size),
        tile_fn=stat_fn,
        no_data_val=no_data_val,
        border=border,
        name=name,
    )


def kernel_conv_node(
    source: "rsgispy._io.TileNode",
    kernel: np.ndarray,
    zero_dc: bool = False,
    *,
    no_data_val: Optional[float] = None,
    border: Optional["rsgispy._io.HaloBorder"] = None,
    name: Optional[str] = None,
) -> "rsgispy._io.WindowOp":
    """Build a composable :class:`~rsgispy._io.WindowOp` for a 2D convolution kernel."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    k_arr = validate_kernel(kernel)

    def _tile_conv(
        wins: np.ndarray,
        centre: np.ndarray,
        valid: np.ndarray,
        centre_valid: np.ndarray,
    ) -> np.ndarray:
        return _convolve_tile_with_mask(
            wins, centre, valid, centre_valid, k_arr, zero_dc=zero_dc
        )

    return _make_tile_stat_op(
        source=source,
        win_shape=(int(k_arr.shape[0]), int(k_arr.shape[1])),
        tile_fn=_tile_conv,
        no_data_val=no_data_val,
        border=border,
        name=name,
    )


def gradient_magnitude_node(
    source: "rsgispy._io.TileNode",
    kernel_x: np.ndarray,
    kernel_y: np.ndarray,
    *,
    no_data_val: Optional[float] = None,
    border: Optional["rsgispy._io.HaloBorder"] = None,
    name: Optional[str] = None,
) -> "rsgispy._io.WindowOp":
    """Build a composable :class:`~rsgispy._io.WindowOp` computing ``hypot(Gx, Gy)``."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    kx = validate_kernel(kernel_x)
    ky = validate_kernel(kernel_y)
    if kx.shape != ky.shape:
        raise rsgispy.RSGISPyException(
            f"kernel_x shape {kx.shape} must match kernel_y shape {ky.shape}."
        )

    def _tile_grad(
        wins: np.ndarray,
        centre: np.ndarray,
        valid: np.ndarray,
        centre_valid: np.ndarray,
    ) -> np.ndarray:
        gx = _convolve_tile_with_mask(
            wins, centre, valid, centre_valid, kx, zero_dc=True
        )
        gy = _convolve_tile_with_mask(
            wins, centre, valid, centre_valid, ky, zero_dc=True
        )
        return np.hypot(gx, gy)

    return _make_tile_stat_op(
        source=source,
        win_shape=(int(kx.shape[0]), int(kx.shape[1])),
        tile_fn=_tile_grad,
        no_data_val=no_data_val,
        border=border,
        name=name,
    )
