"""
Internal SAR speckle filter reducers for :mod:`rsgispy.imagefilter.sar`.
"""

import math
from typing import Tuple

import numpy as np

from rsgispy.imagefilter._helper import StatFunc, _window_mean_var


def _distance_grid(size: int) -> np.ndarray:
    """Return Euclidean pixel distance from the centre of a ``size x size`` grid."""
    half = size // 2
    coords = np.arange(-half, half + 1, dtype=np.float64)
    dy, dx = np.meshgrid(coords, coords, indexing="ij")
    return np.hypot(dy, dx)


def _refined_lee_templates(size: int) -> np.ndarray:
    """
    Build 8 directional half-window boolean templates ``(8, size, size)``
    paired as opposite sides ``(0,1)=N/S, (2,3)=W/E, (4,5)=NW/SE, (6,7)=NE/SW``.
    """
    half = size // 2
    coords = np.arange(-half, half + 1, dtype=np.intp)
    dr, dc = np.meshgrid(coords, coords, indexing="ij")
    return np.stack(
        [
            dr <= 0,           # 0: N (incl centre row)
            dr >= 0,           # 1: S (incl centre row)
            dc <= 0,           # 2: W (incl centre col)
            dc >= 0,           # 3: E (incl centre col)
            (dr + dc) <= 0,    # 4: NW (anti-diagonal upper-left)
            (dr + dc) >= 0,    # 5: SE (anti-diagonal lower-right)
            (dr - dc) <= 0,    # 6: NE (main-diagonal upper-right)
            (dr - dc) >= 0,    # 7: SW (main-diagonal lower-left)
        ],
        axis=0,
    )


def make_enhanced_lee_stat_fn(n_looks: float, damping: float = 1.0) -> StatFunc:
    """Create a Lopes et al. (1990) Enhanced Lee SAR speckle filter reducer."""
    looks = float(n_looks)
    damp = float(damping)
    cu = 1.0 / math.sqrt(looks)
    cmax = math.sqrt(1.0 + 2.0 / looks)

    def _stat_enhanced_lee(
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
        if not np.any(nz):
            return out

        ci = np.zeros(centre_valid.shape, dtype=np.float64)
        ci[nz] = np.sqrt(np.maximum(var[nz], 0.0)) / np.abs(mean[nz])

        point = nz & (ci >= cmax)
        out[point] = centre[point]

        mid = nz & (ci > cu) & (ci < cmax)
        if np.any(mid):
            ratio = (ci[mid] - cu) / np.maximum(cmax - ci[mid], 1e-12)
            w = np.exp(-damp * ratio)
            out[mid] = mean[mid] * w + centre[mid] * (1.0 - w)
        return out

    return _stat_enhanced_lee


def make_frost_stat_fn(damping: float = 1.0) -> StatFunc:
    """Create a Frost et al. (1982) exponentially damped SAR speckle filter reducer."""
    damp = float(damping)

    def _stat_frost(
        wins: np.ndarray,
        centre: np.ndarray,
        valid: np.ndarray,
        centre_valid: np.ndarray,
    ) -> np.ndarray:
        del centre
        mean, var, n_valid = _window_mean_var(wins, valid)
        out = np.full(centre_valid.shape, np.nan, dtype=np.float64)
        ok = centre_valid & (n_valid > 0)
        out[ok] = mean[ok]

        nz = ok & (mean != 0.0) & (var > 0.0)
        if not np.any(nz):
            return out

        dist = _distance_grid(wins.shape[2])
        k_factor = np.zeros(centre_valid.shape, dtype=np.float64)
        k_factor[nz] = damp * (np.maximum(var[nz], 0.0) / (mean[nz] * mean[nz]))

        w_sub = np.exp(
            -k_factor[nz][:, None, None] * dist[None, :, :]
        )
        v_sub = valid[nz]
        w_masked = np.where(v_sub, w_sub, 0.0)
        w_sum = np.sum(w_masked, axis=(1, 2))
        val_sum = np.sum(w_masked * np.where(v_sub, wins[nz], 0.0), axis=(1, 2))
        good = w_sum > 0.0
        res = mean[nz].copy()
        res[good] = val_sum[good] / w_sum[good]
        out[nz] = res
        return out

    return _stat_frost


def make_enhanced_frost_stat_fn(n_looks: float, damping: float = 1.0) -> StatFunc:
    """Create a Lopes et al. (1990) Enhanced Frost SAR speckle filter reducer."""
    looks = float(n_looks)
    damp = float(damping)
    cu = 1.0 / math.sqrt(looks)
    cmax = math.sqrt(1.0 + 2.0 / looks)

    def _stat_enhanced_frost(
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
        if not np.any(nz):
            return out

        ci = np.zeros(centre_valid.shape, dtype=np.float64)
        ci[nz] = np.sqrt(np.maximum(var[nz], 0.0)) / np.abs(mean[nz])

        point = nz & (ci >= cmax)
        out[point] = centre[point]

        mid = nz & (ci > cu) & (ci < cmax)
        if np.any(mid):
            dist = _distance_grid(wins.shape[2])
            k_factor = damp * (ci[mid] - cu) / np.maximum(cmax - ci[mid], 1e-12)
            w_sub = np.exp(-k_factor[:, None, None] * dist[None, :, :])
            v_sub = valid[mid]
            w_masked = np.where(v_sub, w_sub, 0.0)
            w_sum = np.sum(w_masked, axis=(1, 2))
            val_sum = np.sum(w_masked * np.where(v_sub, wins[mid], 0.0), axis=(1, 2))
            good = w_sum > 0.0
            res = mean[mid].copy()
            res[good] = val_sum[good] / w_sum[good]
            out[mid] = res
        return out

    return _stat_enhanced_frost


def make_kuan_stat_fn(n_looks: float) -> StatFunc:
    """Create a Kuan et al. (1985) adaptive MMSE SAR speckle filter reducer."""
    cu2 = 1.0 / float(n_looks)

    def _stat_kuan(
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
            ci2 = np.maximum(var[nz], 0.0) / (mean[nz] * mean[nz])
            w = np.where(
                ci2 > cu2,
                np.clip((1.0 - cu2 / np.maximum(ci2, 1e-12)) / (1.0 + cu2), 0.0, 1.0),
                0.0,
            )
            out[nz] = mean[nz] + w * (centre[nz] - mean[nz])
        return out

    return _stat_kuan


def make_gamma_map_stat_fn(n_looks: float) -> StatFunc:
    """Create a Lopes et al. (1993) Gamma MAP SAR speckle filter reducer."""
    looks = float(n_looks)
    cu2 = 1.0 / looks
    cu = math.sqrt(cu2)
    cmax = math.sqrt(2.0) * cu

    def _stat_gamma_map(
        wins: np.ndarray,
        centre: np.ndarray,
        valid: np.ndarray,
        centre_valid: np.ndarray,
    ) -> np.ndarray:
        mean, var, n_valid = _window_mean_var(wins, valid)
        out = np.full(centre_valid.shape, np.nan, dtype=np.float64)
        ok = centre_valid & (n_valid > 0)
        out[ok] = mean[ok]

        nz = ok & (mean > 0.0) & (centre >= 0.0) & (var > 0.0)
        if not np.any(nz):
            return out

        ci = np.zeros(centre_valid.shape, dtype=np.float64)
        ci[nz] = np.sqrt(np.maximum(var[nz], 0.0)) / mean[nz]

        point = nz & (ci >= cmax)
        out[point] = centre[point]

        mid = nz & (ci > cu) & (ci < cmax)
        if np.any(mid):
            ci2 = ci[mid] * ci[mid]
            alpha = (1.0 + cu2) / np.maximum(ci2 - cu2, 1e-12)
            m_mid = mean[mid]
            c_mid = centre[mid]
            b_term = (alpha - looks - 1.0) * m_mid
            disc = np.maximum(
                b_term * b_term + 4.0 * alpha * looks * m_mid * c_mid,
                0.0,
            )
            out[mid] = (b_term + np.sqrt(disc)) / (2.0 * alpha)
        return out

    return _stat_gamma_map


def _select_directional_stats(
    wins_ok: np.ndarray,
    valid_ok: np.ndarray,
    centre_ok: np.ndarray,
    templates: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate 8 Lee directional sub-windows on ``(N, k, k)`` masked pixels and
    return ``(best_mean, best_var)`` for each pixel.
    """
    n_px = wins_ok.shape[0]
    m_all = valid_ok[:, None, :, :] & templates[None, :, :, :]  # (N, 8, k, k)
    counts = np.sum(m_all, axis=(2, 3))  # (N, 8)
    vals_masked = np.where(m_all, wins_ok[:, None, :, :], 0.0)
    means = np.divide(
        np.sum(vals_masked, axis=(2, 3)),
        np.maximum(counts, 1),
    )

    diff_sq = np.where(m_all, (wins_ok[:, None, :, :] - means[:, :, None, None]) ** 2, 0.0)
    vars_ = np.divide(
        np.sum(diff_sq, axis=(2, 3)),
        np.maximum(counts, 1),
    )

    # Opposite pairs: (0,1), (2,3), (4,5), (6,7)
    pair_diff = np.empty((n_px, 4), dtype=np.float64)
    for p_idx in range(4):
        a_idx, b_idx = 2 * p_idx, 2 * p_idx + 1
        both_ok = (counts[:, a_idx] > 0) & (counts[:, b_idx] > 0)
        pair_diff[:, p_idx] = np.where(
            both_ok, np.abs(means[:, a_idx] - means[:, b_idx]), -1.0
        )

    best_pair = np.argmax(pair_diff, axis=1)
    rows = np.arange(n_px)
    idx_a = 2 * best_pair
    idx_b = idx_a + 1
    ca = counts[rows, idx_a]
    cb = counts[rows, idx_b]
    ma = means[rows, idx_a]
    mb = means[rows, idx_b]

    choose_a = np.where(
        ca == 0,
        False,
        np.where(
            cb == 0,
            True,
            np.abs(ma - centre_ok) <= np.abs(mb - centre_ok),
        ),
    )
    best_tpl = np.where(choose_a, idx_a, idx_b)
    return means[rows, best_tpl], vars_[rows, best_tpl]


def make_refined_lee_stat_fn(n_looks: float) -> StatFunc:
    """Create a Lee et al. (1999) edge-directed Refined Lee SAR speckle filter reducer."""
    cu2 = 1.0 / float(n_looks)

    def _stat_refined_lee(
        wins: np.ndarray,
        centre: np.ndarray,
        valid: np.ndarray,
        centre_valid: np.ndarray,
    ) -> np.ndarray:
        out = np.full(centre_valid.shape, np.nan, dtype=np.float64)
        ok = centre_valid & np.any(valid, axis=(2, 3))
        if not np.any(ok):
            return out

        k_size = wins.shape[2]
        templates = _refined_lee_templates(k_size)
        wins_ok = wins[ok]
        valid_ok = valid[ok]
        centre_ok = centre[ok]

        sub_mean, sub_var = _select_directional_stats(
            wins_ok, valid_ok, centre_ok, templates
        )
        res = sub_mean.copy()
        nz = (sub_mean != 0.0) & (sub_var > 0.0)
        if np.any(nz):
            ci2 = sub_var[nz] / (sub_mean[nz] * sub_mean[nz])
            w = np.maximum(0.0, (ci2 - cu2) / (ci2 * (1.0 + cu2)))
            res[nz] = sub_mean[nz] + w * (centre_ok[nz] - sub_mean[nz])
        out[ok] = res
        return out

    return _stat_refined_lee
