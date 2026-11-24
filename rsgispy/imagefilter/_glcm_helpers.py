"""
Internal Grey-Level Co-occurrence Matrix (GLCM) texture computation engine for
:mod:`rsgispy.imagefilter.texture`.
"""

from contextlib import ExitStack
import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import rasterio

import rsgispy
from rsgispy._numba import optional_njit, prange, use_compiled
from rsgispy.imagefilter._helpers import (
    _apply_tile_nd,
    _resolve_out_nodata,
    validate_filter_size,
)
from rsgispy.imageprocess import (
    build_valid_mask,
    cast_result_block,
    create_output_from_dataset,
    ensure_file_exists,
    iter_dataset_windows,
    read_window_with_halo,
)
from rsgispy.imageutils.metadata import (
    get_band_names,
    get_img_no_data_value,
    set_band_names,
)
from rsgispy.imageutils.stats import pop_img_stats

GLCM_CONTRAST = "contrast"
GLCM_DISSIMILARITY = "dissimilarity"
GLCM_HOMOGENEITY = "homogeneity"
GLCM_ASM = "asm"
GLCM_ENERGY = "energy"
GLCM_CORRELATION = "correlation"
GLCM_ENTROPY = "entropy"
GLCM_MEAN = "mean"
GLCM_VARIANCE = "variance"
GLCM_MAX_PROB = "max_prob"

GLCM_ALL_PROPERTIES: Tuple[str, ...] = (
    GLCM_CONTRAST,
    GLCM_DISSIMILARITY,
    GLCM_HOMOGENEITY,
    GLCM_ASM,
    GLCM_ENERGY,
    GLCM_CORRELATION,
    GLCM_ENTROPY,
    GLCM_MEAN,
    GLCM_VARIANCE,
    GLCM_MAX_PROB,
)

_MOMENT_PROPS = {
    GLCM_CONTRAST,
    GLCM_DISSIMILARITY,
    GLCM_HOMOGENEITY,
    GLCM_CORRELATION,
    GLCM_MEAN,
    GLCM_VARIANCE,
}
_FREQ_PROPS = {GLCM_ASM, GLCM_ENERGY, GLCM_ENTROPY, GLCM_MAX_PROB}


def validate_glcm_properties(
    properties: Union[str, Sequence[str]],
) -> List[str]:
    """Validate and normalise requested GLCM texture property names."""
    raw = [properties] if isinstance(properties, str) else list(properties)
    if not raw:
        raise rsgispy.RSGISPyException("properties must not be empty.")
    out: List[str] = []
    for prop in raw:
        if not isinstance(prop, str):
            raise rsgispy.RSGISPyException(
                f"GLCM property names must be strings, got {prop!r}."
            )
        key = prop.strip().lower()
        if key not in GLCM_ALL_PROPERTIES:
            raise rsgispy.RSGISPyException(
                f"Unsupported GLCM property {prop!r}. "
                f"Supported properties: {', '.join(GLCM_ALL_PROPERTIES)}."
            )
        if key not in out:
            out.append(key)
    return out


def validate_glcm_distances(
    distances: Union[int, Sequence[int]],
    filter_size: int,
) -> List[int]:
    """Validate integer GLCM pixel offsets relative to ``filter_size``."""
    raw = (
        [distances]
        if isinstance(distances, (int, np.integer)) and not isinstance(distances, bool)
        else list(distances)
    )
    if not raw:
        raise rsgispy.RSGISPyException("distances must not be empty.")
    out: List[int] = []
    for dist in raw:
        if isinstance(dist, bool) or not isinstance(dist, (int, np.integer)):
            raise rsgispy.RSGISPyException(
                f"GLCM distances must be positive integers, got {dist!r}."
            )
        d_int = int(dist)
        if d_int < 1 or d_int >= filter_size:
            raise rsgispy.RSGISPyException(
                f"GLCM distance {d_int} must be >= 1 and < filter_size ({filter_size})."
            )
        if d_int not in out:
            out.append(d_int)
    return out


def validate_glcm_angles(angles: Union[float, Sequence[float]]) -> List[float]:
    """Validate orientation angles in degrees."""
    raw = (
        [angles]
        if isinstance(angles, (int, float, np.number)) and not isinstance(angles, bool)
        else list(angles)
    )
    if not raw:
        raise rsgispy.RSGISPyException("angles must not be empty.")
    out: List[float] = []
    for ang in raw:
        try:
            a_flt = float(ang)
        except (TypeError, ValueError) as err:
            raise rsgispy.RSGISPyException(
                f"GLCM angles must be finite numbers, got {ang!r}."
            ) from err
        if not math.isfinite(a_flt):
            raise rsgispy.RSGISPyException(f"GLCM angle must be finite, got {ang!r}.")
        out.append(a_flt)
    return out


def _compute_band_minmax(
    src: rasterio.io.DatasetReader,
    band_idx: int,
    eff_in_nd: Optional[float],
    use_pbar: bool,
) -> Tuple[float, float]:
    """Stream a single raster band to compute finite non-nodata ``(min, max)``."""
    b_min: Optional[float] = None
    b_max: Optional[float] = None
    for win in iter_dataset_windows(src.height, src.width, use_pbar=use_pbar):
        arr = src.read(band_idx, window=win, out_dtype=np.float64)
        valid = build_valid_mask(arr, eff_in_nd)
        if np.any(valid):
            v = arr[valid]
            lo = float(np.min(v))
            hi = float(np.max(v))
            b_min = lo if b_min is None else min(b_min, lo)
            b_max = hi if b_max is None else max(b_max, hi)
    if b_min is None or b_max is None:
        return 0.0, 1.0
    return b_min, b_max


def _quantise_halo(
    halo: np.ndarray,
    valid_halo: np.ndarray,
    lo: float,
    hi: float,
    n_levels: int,
) -> np.ndarray:
    """Quantise continuous halo values into integer bins ``0 .. n_levels - 1``."""
    q = np.zeros(halo.shape, dtype=np.int32)
    span = float(hi) - float(lo)
    if span <= 0.0 or not math.isfinite(span):
        return q
    scaled = np.floor(
        np.where(valid_halo, (halo - float(lo)) / span, 0.0) * float(n_levels)
    )
    return np.clip(scaled, 0, n_levels - 1).astype(np.int32)


def _extract_offset_pairs(
    q_wins: np.ndarray,
    v_wins: np.ndarray,
    dr: int,
    dc: int,
    symmetric: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract directed (or symmetric) co-occurring quantised grey-level pairs
    ``(u, v, pair_valid)`` of shape ``(H, W, P)`` inside each window.
    """
    kh, kw = q_wins.shape[2], q_wins.shape[3]
    src_r = slice(max(0, -dr), kh - max(0, dr))
    src_c = slice(max(0, -dc), kw - max(0, dc))
    dst_r = slice(max(0, dr), kh - max(0, -dr))
    dst_c = slice(max(0, dc), kw - max(0, -dc))

    u = q_wins[:, :, src_r, src_c]
    v = q_wins[:, :, dst_r, dst_c]
    pv = v_wins[:, :, src_r, src_c] & v_wins[:, :, dst_r, dst_c]

    h, w = q_wins.shape[0], q_wins.shape[1]
    u_flat = u.reshape(h, w, -1)
    v_flat = v.reshape(h, w, -1)
    pv_flat = pv.reshape(h, w, -1)
    if not symmetric:
        return u_flat, v_flat, pv_flat

    u_sym = np.concatenate([u_flat, v_flat], axis=2)
    v_sym = np.concatenate([v_flat, u_flat], axis=2)
    pv_sym = np.concatenate([pv_flat, pv_flat], axis=2)
    return u_sym, v_sym, pv_sym


def _calc_freq_props_1d(
    u_ok: np.ndarray,
    v_ok: np.ndarray,
    pv_ok: np.ndarray,
    n_levels: int,
    props: Sequence[str],
) -> Dict[str, np.ndarray]:
    """
    Compute sparse run-length GLCM frequency properties (``asm``, ``energy``,
    ``entropy``, ``max_prob``) for ``N`` valid pixels without ``L x L`` arrays.
    """
    # pylint: disable=too-many-locals
    n_px, n_pairs = u_ok.shape
    stride = n_levels * n_levels + 1
    sentinel = n_levels * n_levels
    out: Dict[str, np.ndarray] = {
        p: np.zeros(n_px, dtype=np.float64) for p in props if p in _FREQ_PROPS
    }
    if not out or n_px == 0 or n_pairs == 0:
        return out

    pair_counts = np.sum(pv_ok, axis=1)
    chunk_sz = 4096
    for start in range(0, n_px, chunk_sz):
        stop = min(start + chunk_sz, n_px)
        c_n = stop - start
        codes = np.where(
            pv_ok[start:stop],
            u_ok[start:stop] * n_levels + v_ok[start:stop],
            sentinel,
        )
        sorted_c = np.sort(codes, axis=1)
        is_valid = sorted_c < sentinel
        rows = np.arange(c_n, dtype=np.int64)[:, None]
        keys = (rows * stride + sorted_c)[is_valid]
        if keys.size == 0:
            continue

        is_start = np.empty(keys.size, dtype=bool)
        is_start[0] = True
        is_start[1:] = keys[1:] != keys[:-1]
        start_idx = np.flatnonzero(is_start)
        run_counts = np.diff(np.append(start_idx, keys.size)).astype(np.float64)
        run_rows = keys[start_idx] // stride
        probs = run_counts / pair_counts[start + run_rows]

        if GLCM_ASM in out or GLCM_ENERGY in out:
            asm_c = np.bincount(run_rows, weights=probs * probs, minlength=c_n)
            if GLCM_ASM in out:
                out[GLCM_ASM][start:stop] = asm_c
            if GLCM_ENERGY in out:
                out[GLCM_ENERGY][start:stop] = np.sqrt(np.maximum(asm_c, 0.0))
        if GLCM_ENTROPY in out:
            ent_w = -probs * np.log(np.maximum(probs, 1e-300))
            out[GLCM_ENTROPY][start:stop] = np.bincount(
                run_rows, weights=ent_w, minlength=c_n
            )
        if GLCM_MAX_PROB in out:
            max_c = np.zeros(c_n, dtype=np.float64)
            np.maximum.at(max_c, run_rows, probs)
            out[GLCM_MAX_PROB][start:stop] = max_c

    return out


@optional_njit(parallel=True, cache=True, warn=False)
def _glcm_offset_kernel(
    q_wins: np.ndarray,
    v_wins: np.ndarray,
    centre_valid: np.ndarray,
    dr: int,
    dc: int,
    n_levels: int,
    symmetric: bool,
    want: np.ndarray,
) -> np.ndarray:
    """
    Compiled per-pixel GLCM properties for a single ``(dr, dc)`` offset
    (parallel over rows). Returns ``(10, H, W)`` ordered as
    :data:`GLCM_ALL_PROPERTIES`; only properties flagged in ``want`` are
    computed and pixels without valid pairs stay ``NaN``.

    Pairs are visited in the same order as :func:`_extract_offset_pairs` and
    frequency properties are accumulated over sorted pair codes exactly as
    :func:`_calc_freq_props_1d`; moment sums are sequential, so results agree
    with the NumPy path to within floating-point rounding.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    # pylint: disable=too-many-nested-blocks
    h, w, kh, kw = q_wins.shape
    out = np.full((10, h, w), np.nan, dtype=np.float64)
    r0 = max(0, -dr)
    r1 = kh - max(0, dr)
    c0 = max(0, -dc)
    c1 = kw - max(0, dc)
    n_dir = max(0, r1 - r0) * max(0, c1 - c0)
    if n_dir == 0:
        return out
    n_pairs = 2 * n_dir if symmetric else n_dir
    need_mom = want[0] or want[1] or want[2] or want[5] or want[7] or want[8]
    need_freq = want[3] or want[4] or want[6] or want[9]
    sentinel = n_levels * n_levels
    for r in prange(h):
        u_buf = np.empty(n_pairs, dtype=np.float64)
        v_buf = np.empty(n_pairs, dtype=np.float64)
        codes = np.empty(n_pairs, dtype=np.int64)
        for c in range(w):
            if not centre_valid[r, c]:
                continue
            m = 0
            k = 0
            for wr in range(r0, r1):
                for wc in range(c0, c1):
                    if v_wins[r, c, wr, wc] and v_wins[r, c, wr + dr, wc + dc]:
                        u_val = q_wins[r, c, wr, wc]
                        v_val = q_wins[r, c, wr + dr, wc + dc]
                        u_buf[m] = u_val
                        v_buf[m] = v_val
                        codes[k] = u_val * n_levels + v_val
                        m += 1
                    else:
                        codes[k] = sentinel
                    k += 1
            n_fwd = m
            if symmetric:
                for i in range(n_fwd):
                    u_buf[n_fwd + i] = v_buf[i]
                    v_buf[n_fwd + i] = u_buf[i]
                for i in range(n_dir):
                    if codes[i] < sentinel:
                        u_q = codes[i] // n_levels
                        v_q = codes[i] - u_q * n_levels
                        codes[n_dir + i] = v_q * n_levels + u_q
                    else:
                        codes[n_dir + i] = sentinel
                m = 2 * n_fwd
            if m == 0:
                continue
            m_f = np.float64(m)

            if need_mom:
                s_con = 0.0
                s_dis = 0.0
                s_hom = 0.0
                s_u = 0.0
                s_v = 0.0
                for i in range(m):
                    d = u_buf[i] - v_buf[i]
                    s_con += d * d
                    s_dis += abs(d)
                    s_hom += 1.0 / (1.0 + d * d)
                    s_u += u_buf[i]
                    s_v += v_buf[i]
                if want[0]:
                    out[0, r, c] = s_con / m_f
                if want[1]:
                    out[1, r, c] = s_dis / m_f
                if want[2]:
                    out[2, r, c] = s_hom / m_f
                mu_i = s_u / m_f
                if want[7]:
                    out[7, r, c] = mu_i
                if want[8] or want[5]:
                    s_du = 0.0
                    for i in range(m):
                        du = u_buf[i] - mu_i
                        s_du += du * du
                    var_i = s_du / m_f
                    if want[8]:
                        out[8, r, c] = var_i
                    if want[5]:
                        mu_j = s_v / m_f
                        s_dv = 0.0
                        s_cov = 0.0
                        for i in range(m):
                            du = u_buf[i] - mu_i
                            dv = v_buf[i] - mu_j
                            s_dv += dv * dv
                            s_cov += du * dv
                        var_j = s_dv / m_f
                        cov = s_cov / m_f
                        denom = np.sqrt(max(var_i, 0.0) * max(var_j, 0.0))
                        out[5, r, c] = cov / denom if denom > 0.0 else 1.0

            if need_freq:
                srt = np.sort(codes)
                asm = 0.0
                ent = 0.0
                max_p = 0.0
                i = 0
                while i < n_pairs and srt[i] < sentinel:
                    j = i + 1
                    while j < n_pairs and srt[j] == srt[i]:
                        j += 1
                    prob = (j - i) / m_f
                    asm += prob * prob
                    ent += -prob * np.log(max(prob, 1e-300))
                    max_p = max(max_p, prob)
                    i = j
                if want[3]:
                    out[3, r, c] = asm
                if want[4]:
                    out[4, r, c] = np.sqrt(max(asm, 0.0))
                if want[6]:
                    out[6, r, c] = ent
                if want[9]:
                    out[9, r, c] = max_p
    return out


def compute_glcm_offset_tile(
    q_wins: np.ndarray,
    v_wins: np.ndarray,
    centre_valid: np.ndarray,
    dr: int,
    dc: int,
    n_levels: int,
    symmetric: bool,
    props: Sequence[str],
) -> Dict[str, np.ndarray]:
    """
    Compute requested GLCM properties for a single ``(dr, dc)`` offset on a
    tile, using a ``numba``-compiled per-pixel kernel when available and the
    vectorised NumPy implementation otherwise.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    if use_compiled(
        _glcm_offset_kernel,
        "compute_glcm_offset_tile",
        fallback_desc="the vectorised NumPy fallback",
    ):
        want = np.array([p in props for p in GLCM_ALL_PROPERTIES], dtype=np.bool_)
        all_res = _glcm_offset_kernel(
            q_wins,
            v_wins,
            centre_valid,
            int(dr),
            int(dc),
            int(n_levels),
            bool(symmetric),
            want,
        )
        return {
            p: all_res[GLCM_ALL_PROPERTIES.index(p)] for p in props
        }
    return _compute_glcm_offset_tile_numpy(
        q_wins, v_wins, centre_valid, dr, dc, n_levels, symmetric, props
    )


def _compute_glcm_offset_tile_numpy(
    q_wins: np.ndarray,
    v_wins: np.ndarray,
    centre_valid: np.ndarray,
    dr: int,
    dc: int,
    n_levels: int,
    symmetric: bool,
    props: Sequence[str],
) -> Dict[str, np.ndarray]:
    """Vectorised NumPy GLCM properties for one offset (compiled-path fallback)."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    h, w = centre_valid.shape
    results: Dict[str, np.ndarray] = {
        p: np.full((h, w), np.nan, dtype=np.float64) for p in props
    }
    u, v, pv = _extract_offset_pairs(q_wins, v_wins, dr, dc, symmetric=symmetric)
    if u.shape[2] == 0:
        return results

    counts = np.sum(pv, axis=2)
    ok = centre_valid & (counts > 0)
    if not np.any(ok):
        return results

    u_ok = u[ok]
    v_ok = v[ok]
    pv_ok = pv[ok]
    m_ok = counts[ok].astype(np.float64)
    u_flt = u_ok.astype(np.float64)
    v_flt = v_ok.astype(np.float64)
    need = set(props)

    if need & _MOMENT_PROPS:
        diff = u_flt - v_flt
        if GLCM_CONTRAST in need:
            results[GLCM_CONTRAST][ok] = np.sum(
                np.where(pv_ok, diff * diff, 0.0), axis=1
            ) / m_ok
        if GLCM_DISSIMILARITY in need:
            results[GLCM_DISSIMILARITY][ok] = np.sum(
                np.where(pv_ok, np.abs(diff), 0.0), axis=1
            ) / m_ok
        if GLCM_HOMOGENEITY in need:
            results[GLCM_HOMOGENEITY][ok] = np.sum(
                np.where(pv_ok, 1.0 / (1.0 + diff * diff), 0.0), axis=1
            ) / m_ok

        if need & {GLCM_MEAN, GLCM_VARIANCE, GLCM_CORRELATION}:
            mu_i = np.sum(np.where(pv_ok, u_flt, 0.0), axis=1) / m_ok
            if GLCM_MEAN in need:
                results[GLCM_MEAN][ok] = mu_i
            if need & {GLCM_VARIANCE, GLCM_CORRELATION}:
                du = np.where(pv_ok, u_flt - mu_i[:, None], 0.0)
                var_i = np.sum(du * du, axis=1) / m_ok
                if GLCM_VARIANCE in need:
                    results[GLCM_VARIANCE][ok] = var_i
                if GLCM_CORRELATION in need:
                    mu_j = np.sum(np.where(pv_ok, v_flt, 0.0), axis=1) / m_ok
                    dv = np.where(pv_ok, v_flt - mu_j[:, None], 0.0)
                    var_j = np.sum(dv * dv, axis=1) / m_ok
                    cov = np.sum(du * dv, axis=1) / m_ok
                    denom = np.sqrt(np.maximum(var_i, 0.0) * np.maximum(var_j, 0.0))
                    corr = np.ones_like(denom)
                    nz = denom > 0.0
                    corr[nz] = cov[nz] / denom[nz]
                    results[GLCM_CORRELATION][ok] = corr

    if need & _FREQ_PROPS:
        freq_res = _calc_freq_props_1d(u_ok, v_ok, pv_ok, n_levels, props)
        for p_name, vals_1d in freq_res.items():
            results[p_name][ok] = vals_1d

    return results


def run_glcm_filter(
    input_img: str,
    output_img: str,
    properties: Union[str, Sequence[str]] = GLCM_ALL_PROPERTIES,
    filter_size: int = 5,
    distances: Union[int, Sequence[int]] = (1,),
    angles: Union[float, Sequence[float]] = (0.0, 45.0, 90.0, 135.0),
    n_levels: int = 32,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    symmetric: bool = True,
    average_angles: bool = True,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """Execute windowed GLCM texture feature extraction across all bands."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-branches
    ensure_file_exists(input_img, "Input image")
    size = validate_filter_size(filter_size)
    prop_list = validate_glcm_properties(properties)
    dist_list = validate_glcm_distances(distances, size)
    angle_list = validate_glcm_angles(angles)
    if (
        isinstance(n_levels, bool)
        or not isinstance(n_levels, (int, np.integer))
        or int(n_levels) < 2
    ):
        raise rsgispy.RSGISPyException(
            f"n_levels must be an integer >= 2, got {n_levels!r}."
        )
    levels = int(n_levels)
    if (min_val is None) != (max_val is None):
        raise rsgispy.RSGISPyException(
            "min_val and max_val must either both be specified or both be None."
        )
    if min_val is not None and max_val is not None and float(max_val) < float(min_val):
        raise rsgispy.RSGISPyException("max_val must be >= min_val.")

    offsets: List[Tuple[int, float, int, int]] = []
    for dist in dist_list:
        for ang in angle_list:
            rad = math.radians(ang)
            dr = int(round(math.sin(rad) * dist))
            dc = int(round(math.cos(rad) * dist))
            if dr == 0 and dc == 0:
                continue
            offsets.append((dist, ang, dr, dc))
    if not offsets:
        raise rsgispy.RSGISPyException("No non-zero GLCM offsets derived.")

    eff_dtype = rsgispy.TYPE_32FLOAT if datatype is None else int(datatype)
    out_rio_dtype = rsgispy.get_rasterio_datatype(eff_dtype)
    out_np_dtype = rsgispy.get_numpy_datatype(eff_dtype)
    eff_in_nd = (
        float(no_data_val)
        if no_data_val is not None
        else get_img_no_data_value(input_img)
    )
    out_nd = _resolve_out_nodata(eff_in_nd, out_np_dtype)
    in_band_names = get_band_names(input_img)

    n_per_band = len(prop_list) if average_angles else len(prop_list) * len(offsets)
    pad = (size // 2, size // 2)

    out_names: List[str] = []
    with ExitStack() as stack:
        src = stack.enter_context(rasterio.open(input_img, "r"))
        out_count = src.count * n_per_band
        band_bounds: List[Tuple[float, float]] = []
        for b_idx in range(1, src.count + 1):
            if min_val is not None and max_val is not None:
                band_bounds.append((float(min_val), float(max_val)))
            else:
                band_bounds.append(
                    _compute_band_minmax(src, b_idx, eff_in_nd, use_pbar=use_pbar)
                )
            b_label = (
                in_band_names[b_idx - 1]
                if b_idx - 1 < len(in_band_names) and in_band_names[b_idx - 1]
                else f"b{b_idx}"
            )
            if average_angles:
                for p_name in prop_list:
                    out_names.append(f"{b_label}_glcm_{p_name}")
            else:
                for p_name in prop_list:
                    for dist, ang, _, _ in offsets:
                        out_names.append(
                            f"{b_label}_glcm_{p_name}_d{dist}_a{int(round(ang))}"
                        )

        dst = create_output_from_dataset(
            stack=stack,
            src=src,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype=out_rio_dtype,
            count=out_count,
        )
        if out_nd is not None:
            dst.nodata = out_nd

        for out_win in iter_dataset_windows(
            src.height, src.width, use_pbar=use_pbar
        ):
            win_h = int(out_win.height)
            win_w = int(out_win.width)
            y_end = pad[0] + win_h
            x_end = pad[1] + win_w
            out_b = 1
            for b_idx in range(1, src.count + 1):
                lo, hi = band_bounds[b_idx - 1]
                halo = read_window_with_halo(
                    src,
                    b_idx,
                    out_win,
                    pad=pad,
                    out_dtype=np.float64,
                    pad_mode="edge",
                )
                valid_halo = build_valid_mask(halo, eff_in_nd)
                centre_valid = valid_halo[pad[0]:y_end, pad[1]:x_end]
                q_halo = _quantise_halo(halo, valid_halo, lo, hi, levels)
                q_wins = sliding_window_view(q_halo, (size, size))
                v_wins = sliding_window_view(valid_halo, (size, size))

                per_offset = [
                    compute_glcm_offset_tile(
                        q_wins=q_wins,
                        v_wins=v_wins,
                        centre_valid=centre_valid,
                        dr=dr,
                        dc=dc,
                        n_levels=levels,
                        symmetric=symmetric,
                        props=prop_list,
                    )
                    for _, _, dr, dc in offsets
                ]

                if average_angles:
                    for p_name in prop_list:
                        stacked = np.stack([d[p_name] for d in per_offset], axis=0)
                        valid_any = np.any(np.isfinite(stacked), axis=0)
                        avg = np.full((win_h, win_w), np.nan, dtype=np.float64)
                        if np.any(valid_any):
                            cnt = np.sum(np.isfinite(stacked), axis=0)
                            s_val = np.sum(
                                np.where(np.isfinite(stacked), stacked, 0.0), axis=0
                            )
                            avg[valid_any] = s_val[valid_any] / cnt[valid_any]
                        tile_res = _apply_tile_nd(avg, out_nd, out_np_dtype)
                        dst.write(
                            cast_result_block(
                                tile_res, win_h, win_w, out_np_dtype, clamp_int=True
                            ),
                            out_b,
                            window=out_win,
                        )
                        out_b += 1
                else:
                    for p_name in prop_list:
                        for off_dict in per_offset:
                            tile_res = _apply_tile_nd(
                                off_dict[p_name], out_nd, out_np_dtype
                            )
                            dst.write(
                                cast_result_block(
                                    tile_res, win_h, win_w, out_np_dtype, clamp_int=True
                                ),
                                out_b,
                                window=out_win,
                            )
                            out_b += 1

    set_band_names(output_img, out_names, feedback=False)
    if calc_stats:
        pop_img_stats(
            output_img,
            no_data_val=out_nd,
            calc_pyramids=True,
            use_pbar=use_pbar,
        )
