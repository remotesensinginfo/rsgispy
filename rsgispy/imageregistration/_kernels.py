"""
Optional ``numba``-accelerated similarity-surface kernels for
:mod:`rsgispy.imageregistration` tie-point and shift searches.

The kernels evaluate :func:`rsgispy.tools.stats.calc_similarity_metric`
equivalents (correlation, RMSE, MSE, MAE, mutual information and normalised
mutual information) over every integer offset of a search window, gathering
masked pixels without allocating temporary NumPy arrays per offset. Mutual
information reproduces :func:`numpy.histogram2d` bin edges and bin assignment
exactly; floating-point sums are accumulated sequentially, so results agree
with the NumPy implementation to within rounding (``~1e-12`` relative).

Callers must only use these kernels when :func:`rsgispy._numba.use_compiled`
returns ``True`` and every masked value is finite (see
:func:`masked_values_finite`); otherwise they fall back to the NumPy code.
"""

import math
from typing import Tuple

import numpy as np

from rsgispy._numba import njit, optional_njit, prange

__all__ = [
    "SUPPORTED_KERNEL_METRICS",
    "masked_values_finite",
    "shift_surface_kernel",
    "template_surface_kernel",
]

# rsgispy.METRIC_EUCLIDEAN .. rsgispy.METRIC_NORM_MUTUAL_INFO
SUPPORTED_KERNEL_METRICS = frozenset((1, 2, 3, 4, 5, 6))


def masked_values_finite(arr: np.ndarray, mask: np.ndarray) -> bool:
    """Return ``True`` if all ``arr`` values selected by ``mask`` are finite."""
    if not np.issubdtype(arr.dtype, np.floating):
        return True
    if arr.ndim == mask.ndim:
        return bool(np.all(np.isfinite(arr[mask])))
    return bool(np.all(np.isfinite(arr[:, mask])))


@njit(cache=True)
def _hist_edges(vmin: float, vmax: float, n_bins: int, edges: np.ndarray) -> None:
    """Replicate ``np.linspace(vmin, vmax, n_bins + 1)`` into ``edges``."""
    delta = vmax - vmin
    step = delta / n_bins
    if step == 0.0:
        for i in range(n_bins + 1):
            edges[i] = (i / n_bins) * delta + vmin
    else:
        for i in range(n_bins + 1):
            edges[i] = i * step + vmin
    edges[n_bins] = vmax


@njit(cache=True)
def _hist_bin(val: float, edges: np.ndarray, n_bins: int) -> int:
    """Bin index matching ``np.histogramdd`` (``searchsorted(..., 'right')``)."""
    idx = np.searchsorted(edges, val, side="right")
    if val == edges[n_bins]:
        idx -= 1
    return idx - 1


@njit(cache=True)
def _mutual_information_1d(
    a_vals: np.ndarray,
    b_vals: np.ndarray,
    n: int,
    hist_bins: int,
    normalised: bool,
) -> float:
    """Compiled equivalent of :func:`rsgispy.tools.stats._mutual_information`."""
    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    if n < 2:
        return 0.0
    a_min = a_vals[0]
    a_max = a_vals[0]
    b_min = b_vals[0]
    b_max = b_vals[0]
    for i in range(1, n):
        a_min = min(a_min, a_vals[i])
        a_max = max(a_max, a_vals[i])
        b_min = min(b_min, b_vals[i])
        b_max = max(b_max, b_vals[i])
    if a_max - a_min == 0.0 or b_max - b_min == 0.0:
        return 0.0

    a_edges = np.empty(hist_bins + 1, dtype=np.float64)
    b_edges = np.empty(hist_bins + 1, dtype=np.float64)
    _hist_edges(a_min, a_max, hist_bins, a_edges)
    _hist_edges(b_min, b_max, hist_bins, b_edges)

    counts = np.zeros((hist_bins, hist_bins), dtype=np.float64)
    total = 0.0
    for i in range(n):
        ia = _hist_bin(a_vals[i], a_edges, hist_bins)
        ib = _hist_bin(b_vals[i], b_edges, hist_bins)
        if 0 <= ia < hist_bins and 0 <= ib < hist_bins:
            counts[ia, ib] += 1.0
            total += 1.0
    if total <= 0.0:
        return 0.0

    p_x = np.zeros(hist_bins, dtype=np.float64)
    p_y = np.zeros(hist_bins, dtype=np.float64)
    h_xy = 0.0
    for i in range(hist_bins):
        for j in range(hist_bins):
            p = counts[i, j] / total
            p_x[i] += p
            p_y[j] += p
            if p > 0.0:
                h_xy -= p * math.log(p)
    h_x = 0.0
    h_y = 0.0
    for i in range(hist_bins):
        if p_x[i] > 0.0:
            h_x -= p_x[i] * math.log(p_x[i])
        if p_y[i] > 0.0:
            h_y -= p_y[i] * math.log(p_y[i])
    mi = max(0.0, h_x + h_y - h_xy)
    if not normalised:
        return mi
    denom = h_x + h_y
    if denom <= 1e-12:
        return 0.0
    return 2.0 * mi / denom


@njit(cache=True)
def _similarity_1d(
    r_vals: np.ndarray,
    f_vals: np.ndarray,
    n: int,
    metric_id: int,
    hist_bins: int,
) -> float:
    """Compiled equivalent of ``calc_similarity_metric`` on 1D buffers."""
    # pylint: disable=too-many-return-statements,too-many-locals
    if n < 2:
        return np.nan
    if metric_id == 4:  # correlation
        r_sum = 0.0
        f_sum = 0.0
        for i in range(n):
            r_sum += r_vals[i]
            f_sum += f_vals[i]
        r_mean = r_sum / n
        f_mean = f_sum / n
        rr = 0.0
        ff = 0.0
        rf = 0.0
        for i in range(n):
            r0 = r_vals[i] - r_mean
            f0 = f_vals[i] - f_mean
            rr += r0 * r0
            ff += f0 * f0
            rf += r0 * f0
        denom = math.sqrt(rr * ff)
        if denom <= 1e-12:
            return 0.0
        return rf / denom
    if metric_id in (1, 2):  # euclidean (RMSE) / sqdiff (MSE)
        acc = 0.0
        for i in range(n):
            d = r_vals[i] - f_vals[i]
            acc += d * d
        mse = acc / n
        return math.sqrt(mse) if metric_id == 1 else mse
    if metric_id == 3:  # manhattan (MAE)
        acc = 0.0
        for i in range(n):
            acc += abs(r_vals[i] - f_vals[i])
        return acc / n
    if metric_id == 5:
        return _mutual_information_1d(r_vals, f_vals, n, hist_bins, False)
    if metric_id == 6:
        return _mutual_information_1d(r_vals, f_vals, n, hist_bins, True)
    return np.nan


@optional_njit(cache=True, warn=False)
def template_surface_kernel(
    ref_patch: np.ndarray,
    flt_search: np.ndarray,
    ref_mask: np.ndarray,
    flt_mask: np.ndarray,
    search_area: int,
    metric_id: int,
    min_pxls: int,
    hist_bins: int,
) -> np.ndarray:
    """
    Evaluate the masked template-matching surface ``(2*sa+1, 2*sa+1)`` where
    index ``[yi, xi]`` corresponds to shift ``(sy, sx) = (yi - sa, xi - sa)``
    and compares ``ref_patch[m]`` with
    ``flt_search[sa - sy : sa - sy + win_h, sa - sx : sa - sx + win_w][m]``.
    Offsets with fewer than ``min_pxls`` valid pixels are ``NaN``.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    # pylint: disable=too-many-nested-blocks
    win_h, win_w = ref_patch.shape
    sa = search_area
    n_grid = 2 * sa + 1
    surface = np.full((n_grid, n_grid), np.nan, dtype=np.float64)
    r_buf = np.empty(win_h * win_w, dtype=np.float64)
    f_buf = np.empty(win_h * win_w, dtype=np.float64)
    for yi in range(n_grid):
        r0 = sa - (yi - sa)
        for xi in range(n_grid):
            c0 = sa - (xi - sa)
            n = 0
            for i in range(win_h):
                for j in range(win_w):
                    if ref_mask[i, j] and flt_mask[r0 + i, c0 + j]:
                        r_buf[n] = ref_patch[i, j]
                        f_buf[n] = flt_search[r0 + i, c0 + j]
                        n += 1
            if n < min_pxls:
                continue
            surface[yi, xi] = _similarity_1d(r_buf, f_buf, n, metric_id, hist_bins)
    return surface


@optional_njit(parallel=True, cache=True, warn=False)
def shift_surface_kernel(
    ref_bands_arr: np.ndarray,
    flt_bands_arr: np.ndarray,
    ref_valid: np.ndarray,
    flt_valid: np.ndarray,
    y_search: int,
    x_search: int,
    metric_id: int,
    hist_bins: int,
    min_valid_pxls: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compiled equivalent of
    :func:`rsgispy.imageregistration._helpers.search_shift_surface`: evaluate
    the band-averaged metric for every integer shift ``(sy, sx)`` where
    ``ref[r, c]`` is matched with ``flt[r - sy, c - sx]``. Parallel over
    ``sy`` rows (each writes distinct output rows, so results are
    deterministic).
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    # pylint: disable=too-many-branches,too-many-nested-blocks
    n_bands, h, w = ref_bands_arr.shape
    n_y = 2 * y_search + 1
    n_x = 2 * x_search + 1
    surface = np.full((n_y, n_x), np.nan, dtype=np.float64)
    counts = np.zeros((n_y, n_x), dtype=np.int64)
    for yi in prange(n_y):
        sy = yi - y_search
        r_ref_0 = max(0, sy)
        r_ref_1 = min(h, h + sy)
        if r_ref_1 <= r_ref_0:
            continue
        r_buf = np.empty(h * w, dtype=np.float64)
        f_buf = np.empty(h * w, dtype=np.float64)
        for xi in range(n_x):
            sx = xi - x_search
            c_ref_0 = max(0, sx)
            c_ref_1 = min(w, w + sx)
            if c_ref_1 <= c_ref_0:
                continue
            n_val = 0
            for r in range(r_ref_0, r_ref_1):
                for c in range(c_ref_0, c_ref_1):
                    if ref_valid[r, c] and flt_valid[r - sy, c - sx]:
                        n_val += 1
            if n_val < min_valid_pxls:
                continue
            score_sum = 0.0
            n_scores = 0
            for b in range(n_bands):
                n = 0
                for r in range(r_ref_0, r_ref_1):
                    for c in range(c_ref_0, c_ref_1):
                        if ref_valid[r, c] and flt_valid[r - sy, c - sx]:
                            r_buf[n] = ref_bands_arr[b, r, c]
                            f_buf[n] = flt_bands_arr[b, r - sy, c - sx]
                            n += 1
                val = _similarity_1d(r_buf, f_buf, n, metric_id, hist_bins)
                if np.isfinite(val):
                    score_sum += val
                    n_scores += 1
            if n_scores > 0:
                surface[yi, xi] = score_sum / n_scores
                counts[yi, xi] = n_val
    return surface, counts
