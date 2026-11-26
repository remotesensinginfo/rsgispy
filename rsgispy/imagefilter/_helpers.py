# pylint: disable=too-many-lines
"""
Internal windowed spatial filtering and convolution engine for
:mod:`rsgispy.imagefilter`.
"""

import math
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

import rsgispy
from rsgispy._numba import optional_njit, prange, use_compiled
from rsgispy._io import (
    DiskSink,
    HaloBorder,
    NodeMeta,
    RasterSource,
    TileGraph,
    TileNode,
    WindowOp,
    same_bands_meta,
)
from rsgispy.imageprocess import (
    build_valid_mask,
    ensure_file_exists,
    resolve_out_nodata as _ip_resolve_out_nodata,
    validate_odd_window_size,
)
from rsgispy.imageutils.metadata import get_band_names, get_img_no_data_value

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


@optional_njit(parallel=True, cache=True, warn=False)
def _stat_mode_kernel(
    wins: np.ndarray,
    valid: np.ndarray,
    centre_valid: np.ndarray,
) -> np.ndarray:
    """
    Compiled sliding-window mode (parallel over rows). For each centre-valid
    pixel, valid window values are sorted and the smallest finite value with
    the longest run is returned, matching :func:`_stat_mode_numpy` exactly.
    """
    # pylint: disable=too-many-locals,too-many-nested-blocks
    h, w, k_h, k_w = wins.shape
    k_len = k_h * k_w
    out = np.full((h, w), np.nan, dtype=np.float64)
    for r in prange(h):
        buf = np.empty(k_len, dtype=np.float64)
        for c in range(w):
            if not centre_valid[r, c]:
                continue
            n_valid = 0
            for i in range(k_h):
                for j in range(k_w):
                    if valid[r, c, i, j]:
                        buf[i * k_w + j] = wins[r, c, i, j]
                        n_valid += 1
                    else:
                        buf[i * k_w + j] = np.inf
            if n_valid == 0:
                continue
            srt = np.sort(buf)
            best_cnt = 0
            best_idx = 0
            k = 0
            while k < k_len:
                val = srt[k]
                if not np.isfinite(val):
                    k += 1
                    continue
                j = k + 1
                while j < k_len and srt[j] == val:
                    j += 1
                if j - k > best_cnt:
                    best_cnt = j - k
                    best_idx = k
                k = j
            out[r, c] = srt[best_idx]
    return out


def stat_mode(
    wins: np.ndarray,
    centre: np.ndarray,
    valid: np.ndarray,
    centre_valid: np.ndarray,
) -> np.ndarray:
    """
    Compute local window mode over valid pixels with deterministic ties
    (smallest of the most frequent finite values). Uses a ``numba``-compiled
    kernel when available, otherwise :func:`_stat_mode_numpy`.
    """
    del centre
    if wins.dtype.kind in "biuf" and use_compiled(
        _stat_mode_kernel, "stat_mode", fallback_desc="the vectorised NumPy fallback"
    ):
        return _stat_mode_kernel(wins, valid.astype(np.bool_, copy=False), centre_valid)
    return _stat_mode_numpy(wins, valid, centre_valid)


def _stat_mode_numpy(
    wins: np.ndarray,
    valid: np.ndarray,
    centre_valid: np.ndarray,
) -> np.ndarray:
    """Vectorised NumPy sliding-window mode (fallback for :func:`stat_mode`)."""
    # pylint: disable=too-many-locals
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


def run_filter_node(
    source: RasterSource,
    node: TileNode,
    output_img: str,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
    band_names: Optional[Sequence[str]] = None,
) -> None:
    """
    Run ``source -> node -> DiskSink`` for a file-based filter function.

    ``node`` is a :class:`FilterWindowOp` or
    :class:`~rsgispy.imagefilter._glcm_helpers.GlcmWindowOp` built on
    ``source``. The input no-data value (``no_data_val``, or the image's) is
    set on ``source`` and ``node``; the output no-data value is the input one
    when the output type can store it (else ``-9999`` for float and ``0`` for
    integer outputs), and filter pixels without a value get that value (NaN
    for float outputs and ``0`` for integer outputs when there is none). The
    output band names are ``band_names`` or those of the input image, and the
    graph runs with the package default block size.
    """
    # pylint: disable=too-many-arguments,too-many-locals
    input_img = source.path
    ensure_file_exists(input_img, "Input image")
    eff_dtype = rsgispy.TYPE_32FLOAT if datatype is None else int(datatype)
    out_np_dtype = rsgispy.get_numpy_datatype(eff_dtype)
    eff_in_nd = (
        float(no_data_val)
        if no_data_val is not None
        else get_img_no_data_value(input_img)
    )
    out_nd = _resolve_out_nodata(eff_in_nd, out_np_dtype)
    source.nodata = eff_in_nd
    node.set_nodata(eff_in_nd, out_nd)  # type: ignore[attr-defined]
    names = list(band_names) if band_names is not None else get_band_names(input_img)
    sink = DiskSink(
        node,
        output_img,
        gdalformat=gdalformat,
        datatype=eff_dtype,
        nodata=out_nd,
        clamp_int=True,
        band_names=names,
        calc_stats=calc_stats,
        inherit_meta=False,
    )
    TileGraph(sink, block_size=None).run(use_pbar=use_pbar)


class FilterWindowOp(WindowOp):
    """
    :class:`~rsgispy.imageprocess.WindowOp` applying a per-band moving-window
    filter ``tile_fn(wins, centre, valid, centre_valid)`` to every band of its
    input (the node behind all the ``*_filter_node`` builders of
    :mod:`rsgispy.imagefilter`).

    Pixels equal to ``no_data_val`` (and non-finite pixels) are ignored, and
    output pixels without a finite value are set to ``fill_value`` (which is
    ``no_data_val`` unless changed with :meth:`set_nodata`). The output blocks
    are ``float64`` with the bands and band names of the input.

    :param source: Upstream :class:`~rsgispy.imageprocess.TileNode`.
    :param win_shape: ``(height, width)`` of the odd moving window.
    :param tile_fn: Window function (see :data:`StatFunc`).
    :param no_data_val: Optional input no-data value.
    :param border: Padding outside the image (Default = ``HaloBorder('edge')``).
    :param name: Optional node name.
    """

    # pylint: disable=too-many-arguments

    def __init__(
        self,
        source: TileNode,
        win_shape: Tuple[int, int],
        tile_fn: StatFunc,
        *,
        no_data_val: Optional[float] = None,
        border: Optional[HaloBorder] = None,
        name: Optional[str] = None,
    ) -> None:
        kh, kw = int(win_shape[0]), int(win_shape[1])
        self.win_shape: Tuple[int, int] = (kh, kw)
        self.tile_fn: StatFunc = tile_fn
        self.no_data_val: Optional[float] = no_data_val
        self.fill_value: Optional[float] = no_data_val
        super().__init__(
            source=source,
            radius=(kh // 2, kw // 2),
            func=self._filter_block,
            border=border if border is not None else HaloBorder("edge"),
            name=name,
            meta_fn=self._filter_meta,
        )

    def set_nodata(
        self, no_data_val: Optional[float], fill_value: Optional[float]
    ) -> None:
        """
        Set the input no-data value and the value written where the filter
        has no result (``None`` leaves NaN). Call before the graph is built.
        """
        self.no_data_val = no_data_val
        self.fill_value = fill_value

    def _filter_meta(self, inputs: Sequence[NodeMeta]) -> NodeMeta:
        return same_bands_meta(np.float64, self.fill_value)(inputs)

    def _filter_block(self, padded_3d: np.ndarray) -> np.ndarray:
        # pylint: disable=too-many-locals
        kh, kw = self.win_shape
        pad_y, pad_x = kh // 2, kw // 2
        n_bands, h_in, w_in = padded_3d.shape
        out_h = h_in - 2 * pad_y
        out_w = w_in - 2 * pad_x
        y_end = pad_y + out_h
        x_end = pad_x + out_w
        no_data_val = self.no_data_val
        fill = self.fill_value
        out = np.empty((n_bands, out_h, out_w), dtype=np.float64)
        for b_idx in range(n_bands):
            band_halo = padded_3d[b_idx]
            valid_halo = build_valid_mask(band_halo, no_data_val)
            centre = band_halo[pad_y:y_end, pad_x:x_end]
            centre_valid = valid_halo[pad_y:y_end, pad_x:x_end]
            wins = sliding_window_view(band_halo, (kh, kw))
            v_wins = sliding_window_view(valid_halo, (kh, kw))
            res = self.tile_fn(wins, centre, v_wins, centre_valid)
            if fill is not None:
                res = np.where(np.isfinite(res), res, float(fill))
            out[b_idx] = res
        return out


def _make_tile_stat_op(
    source: TileNode,
    win_shape: Tuple[int, int],
    tile_fn: StatFunc,
    *,
    no_data_val: Optional[float] = None,
    border: Optional[HaloBorder] = None,
    name: Optional[str] = None,
) -> FilterWindowOp:
    """Build a :class:`FilterWindowOp` running ``tile_fn`` band-by-band."""
    # pylint: disable=too-many-arguments
    return FilterWindowOp(
        source,
        win_shape,
        tile_fn,
        no_data_val=no_data_val,
        border=border,
        name=name,
    )


def window_stat_node(
    source: TileNode,
    filter_size: int,
    stat_fn: StatFunc,
    *,
    no_data_val: Optional[float] = None,
    border: Optional[HaloBorder] = None,
    name: Optional[str] = None,
) -> FilterWindowOp:
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
    source: TileNode,
    kernel: np.ndarray,
    zero_dc: bool = False,
    *,
    no_data_val: Optional[float] = None,
    border: Optional[HaloBorder] = None,
    name: Optional[str] = None,
) -> FilterWindowOp:
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
    source: TileNode,
    kernel_x: np.ndarray,
    kernel_y: np.ndarray,
    *,
    no_data_val: Optional[float] = None,
    border: Optional[HaloBorder] = None,
    name: Optional[str] = None,
) -> FilterWindowOp:
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
