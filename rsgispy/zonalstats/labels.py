"""
The zonalstats.labels module provides fast zonal statistics computed directly
from a label raster (e.g., a segmentation or clump image), where each zone is
the set of pixels sharing a label value, using ``rasterio`` and ``numpy``.

Unlike the polygon zonal statistics in :mod:`rsgispy.zonalstats.polygon`,
which rasterise each polygon and re-read the image window for every polygon
and band, :func:`calc_label_zonal_stats` makes a single streamed, block-wise
pass over the label raster and the value images, accumulating the statistics
for all labels at once with vectorised grouped reductions. Memory use is
bounded by the number of labels (not the image size), and the run time is
roughly proportional to the number of pixels rather than the number of zones,
making it suitable for segmentations with millions of segments.

The statistics are identical to polygon zonal statistics computed on the
polygonised labels with pixel-centre-in-polygon semantics
(:data:`rsgispy.zonalstats.METHOD_POLYCONTAINSPIXELCENTRE`).

The module also provides :func:`join_label_stats_to_vec` to join a label
statistics table onto a vector layer (e.g., the output of
:func:`rsgispy.segmentation.polygonise_segments`) and
:func:`rasterise_label_values` to map per-label values back onto the label
raster's pixel grid.
"""

# pylint: disable=too-many-lines

from contextlib import ExitStack
from dataclasses import dataclass, field
import math
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.io import DatasetReader

import rsgispy
from rsgispy.imageprocess import (
    create_output_from_dataset,
    ensure_file_exists,
    iter_dataset_windows,
    maybe_pop_stats,
    normalise_creation_options,
    resolve_out_dtype,
    rsgis_dtype_from_np_dtype,
    validate_band_index,
)
from rsgispy.tools.tables import write_dataframe_table
from rsgispy.vectorprocess import VectorInput, finish_vec_output, read_vec_gdf
from rsgispy.zonalstats._helpers import read_bands_native

__all__ = [
    "LABEL_ZONAL_STATS",
    "calc_label_zonal_stats",
    "join_label_stats_to_vec",
    "rasterise_label_values",
]

LABEL_ZONAL_STATS: Tuple[str, ...] = (
    "count",
    "sum",
    "mean",
    "stddev",
    "min",
    "max",
    "median",
    "mode",
)
"""Statistic names supported by :func:`calc_label_zonal_stats`."""

_DEFAULT_STATS: Tuple[str, ...] = ("mean", "stddev", "min", "max", "count")
_DEFAULT_DENSE_MAX_LABELS = 50_000_000
_PAIR_CONSOLIDATE_LEN = 5_000_000


@dataclass
class _FeatDef:
    """Resolved definition of one value band processed by the label statistics."""

    # pylint: disable=too-many-instance-attributes
    path: str
    band: int
    feat: str
    img: str
    no_data: Optional[float]
    min_vld: Optional[float]
    max_vld: Optional[float]
    categorical: bool
    is_int: bool
    stats: Tuple[str, ...] = ()
    percentiles: Tuple[float, ...] = ()
    needs_pairs: bool = False
    pair_mode: str = "exact"
    categories: Optional[List[float]] = None
    hist_min: float = 0.0
    hist_width: float = 1.0
    hist_nbins: int = 1
    hist_const: bool = False
    acc: Dict[str, np.ndarray] = field(default_factory=dict)
    pairs: Optional["_PairCounter"] = None


class _PairCounter:
    """
    Accumulate counts of ``(label_index, key)`` pairs over many blocks with
    periodic consolidation so memory is bounded by the number of distinct pairs.
    """

    def __init__(self, consolidate_len: int = _PAIR_CONSOLIDATE_LEN) -> None:
        self._g: List[np.ndarray] = []
        self._k: List[np.ndarray] = []
        self._c: List[np.ndarray] = []
        self._pending = 0
        self._consolidated = 0
        self._consolidate_len = int(consolidate_len)

    @staticmethod
    def _reduce(
        g_arr: np.ndarray, k_arr: np.ndarray, c_arr: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sort by ``(g, k)`` and sum the counts of duplicate pairs."""
        if g_arr.size == 0:
            return g_arr, k_arr, c_arr
        order = np.lexsort((k_arr, g_arr))
        g_s = g_arr[order]
        k_s = k_arr[order]
        c_s = c_arr[order]
        new_grp = np.empty(g_s.size, dtype=bool)
        new_grp[0] = True
        new_grp[1:] = (g_s[1:] != g_s[:-1]) | (k_s[1:] != k_s[:-1])
        starts = np.flatnonzero(new_grp)
        return g_s[starts], k_s[starts], np.add.reduceat(c_s, starts)

    def add(self, g_arr: np.ndarray, k_arr: np.ndarray) -> None:
        """Add one observation per ``(g_arr[i], k_arr[i])`` pair."""
        if g_arr.size == 0:
            return
        g_r, k_r, c_r = self._reduce(
            g_arr.astype(np.int64, copy=False),
            k_arr.astype(np.float64, copy=False),
            np.ones(g_arr.size, dtype=np.int64),
        )
        self._g.append(g_r)
        self._k.append(k_r)
        self._c.append(c_r)
        self._pending += g_r.size
        if self._pending > max(self._consolidate_len, 2 * self._consolidated):
            self._consolidate()

    def _consolidate(self) -> None:
        if len(self._g) <= 1:
            self._pending = sum(arr.size for arr in self._g)
            self._consolidated = self._pending
            return
        g_r, k_r, c_r = self._reduce(
            np.concatenate(self._g), np.concatenate(self._k), np.concatenate(self._c)
        )
        self._g, self._k, self._c = [g_r], [k_r], [c_r]
        self._pending = g_r.size
        self._consolidated = g_r.size

    def result(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the sorted unique ``(g, k, count)`` arrays."""
        if not self._g:
            empty_i = np.empty(0, dtype=np.int64)
            return empty_i, np.empty(0, dtype=np.float64), empty_i
        self._consolidate()
        return self._g[0], self._k[0], self._c[0]


def _group_starts(sorted_ids: np.ndarray) -> np.ndarray:
    """Return the start index of each run of equal values in ``sorted_ids``."""
    if sorted_ids.size == 0:
        return np.empty(0, dtype=np.int64)
    new_grp = np.empty(sorted_ids.size, dtype=bool)
    new_grp[0] = True
    new_grp[1:] = sorted_ids[1:] != sorted_ids[:-1]
    return np.flatnonzero(new_grp)


def _pair_quantiles(
    g_arr: np.ndarray,
    v_arr: np.ndarray,
    c_arr: np.ndarray,
    n_ids: int,
    qs: Sequence[float],
) -> Dict[float, np.ndarray]:
    """
    Compute linear-interpolation percentiles (as :func:`numpy.percentile`) per
    label from sorted ``(label, value, count)`` pairs.
    """
    # pylint: disable=too-many-locals
    out = {q: np.full(n_ids, np.nan, dtype=np.float64) for q in qs}
    if g_arr.size == 0:
        return out
    starts = _group_starts(g_arr)
    labels = g_arr[starts]
    cum = np.cumsum(c_arr)
    grp_end = np.append(starts[1:], g_arr.size) - 1
    totals = cum[grp_end] - (cum[starts] - c_arr[starts])
    base = cum[starts] - c_arr[starts]
    for q_val in qs:
        pos = (float(q_val) / 100.0) * (totals - 1)
        lo_rank = np.floor(pos)
        hi_rank = np.ceil(pos)
        frac = pos - lo_rank
        lo_idx = np.searchsorted(cum, base + lo_rank, side="right")
        hi_idx = np.searchsorted(cum, base + hi_rank, side="right")
        v_lo = v_arr[lo_idx]
        v_hi = v_arr[hi_idx]
        out[q_val][labels] = v_lo + (v_hi - v_lo) * frac
    return out


def _pair_mode(
    g_arr: np.ndarray, v_arr: np.ndarray, c_arr: np.ndarray, n_ids: int
) -> np.ndarray:
    """Return the most frequent value per label (ties resolve to the smallest value)."""
    out = np.full(n_ids, np.nan, dtype=np.float64)
    if g_arr.size == 0:
        return out
    starts = _group_starts(g_arr)
    sizes = np.diff(np.append(starts, g_arr.size))
    max_cnt = np.maximum.reduceat(c_arr, starts)
    is_max = c_arr == np.repeat(max_cnt, sizes)
    idx_arr = np.where(is_max, np.arange(g_arr.size), g_arr.size)
    first_max = np.minimum.reduceat(idx_arr, starts)
    out[g_arr[starts]] = v_arr[first_max]
    return out


def _resolve_label_nodata(
    lab_src: DatasetReader, label_band: int, label_no_data_val: Optional[int]
) -> Optional[int]:
    """Resolve the label no-data value (explicit, else header, else ``None``)."""
    if label_no_data_val is not None:
        return int(label_no_data_val)
    hdr_nd = lab_src.nodatavals[label_band - 1]
    if hdr_nd is None or (isinstance(hdr_nd, float) and math.isnan(hdr_nd)):
        return None
    return int(hdr_nd)


def _check_same_grid(
    lab_src: DatasetReader, src: DatasetReader, label_img: str, img_path: str
) -> None:
    """Raise if ``src`` is not on exactly the same pixel grid as ``lab_src``."""
    same_size = (lab_src.width == src.width) and (lab_src.height == src.height)
    same_trans = bool(
        np.allclose(
            np.asarray(lab_src.transform)[:6],
            np.asarray(src.transform)[:6],
            rtol=0.0,
            atol=1e-6 * max(abs(lab_src.transform.a), 1e-12),
        )
    )
    same_crs = (lab_src.crs is None and src.crs is None) or (
        lab_src.crs is not None and src.crs is not None and lab_src.crs == src.crs
    )
    if not (same_size and same_trans and same_crs):
        raise rsgispy.RSGISPyException(
            f"Image '{img_path}' is not on the same pixel grid (size, transform, "
            f"CRS) as the label image '{label_img}'. Resample it onto the label "
            "image grid first, e.g. with "
            "rsgispy.imageutils.warp.resample_img_to_match."
        )


def _label_is_grid_compatible(label_img: str, img_path: str) -> bool:
    """Return ``True`` if ``img_path`` shares the pixel grid of ``label_img``."""
    with rasterio.open(label_img, "r") as lab_src, rasterio.open(img_path, "r") as src:
        try:
            _check_same_grid(lab_src, src, label_img, img_path)
        except rsgispy.RSGISPyException:
            return False
    return True


def _resolve_feat_defs(
    band_specs: Union[Any, Sequence[Any]],
) -> List[_FeatDef]:
    """Expand band specifications into one :class:`_FeatDef` per value band."""
    # pylint: disable=too-many-locals
    # pylint: disable=import-outside-toplevel
    from rsgispy.ml.stack import ImageFeatureStack

    if band_specs is None:
        return []
    if isinstance(band_specs, (str, os.PathLike)) or not isinstance(
        band_specs, Sequence
    ):
        band_specs = [band_specs]
    if len(band_specs) == 0:
        return []
    stack = ImageFeatureStack(list(band_specs))
    resolved = stack.resolve_band_specs()
    feat_defs: List[_FeatDef] = []
    res_idx = 0
    for img_idx, spec in enumerate(stack.specs):
        img_name = (
            spec.name.strip()
            if spec.name and spec.name.strip()
            else os.path.splitext(os.path.basename(spec.file_name))[0]
            or f"img{img_idx + 1}"
        )
        with rasterio.open(spec.file_name, "r") as src:
            n_bands = len(spec.bands) if spec.bands is not None else int(src.count)
            for _ in range(n_bands):
                path, band, feat = resolved[res_idx]
                res_idx += 1
                hdr_nd = src.nodatavals[band - 1]
                nd_val = (
                    float(spec.no_data_val)
                    if spec.no_data_val is not None
                    else (float(hdr_nd) if hdr_nd is not None else None)
                )
                feat_defs.append(
                    _FeatDef(
                        path=path,
                        band=int(band),
                        feat=feat,
                        img=img_name,
                        no_data=nd_val,
                        min_vld=spec.min_vld_val,
                        max_vld=spec.max_vld_val,
                        categorical=int(spec.data_type)
                        == int(rsgispy.VAR_TYPE_CATEGORICAL),
                        is_int=bool(
                            np.issubdtype(np.dtype(src.dtypes[band - 1]), np.integer)
                        ),
                    )
                )
    return feat_defs


def _scan_label_ids(
    lab_src: DatasetReader,
    label_band: int,
    lab_nd: Optional[int],
    block_size: Optional[int],
) -> np.ndarray:
    """Return the sorted unique valid label IDs in the label raster."""
    pending: List[np.ndarray] = []
    n_pending = 0
    for win in iter_dataset_windows(
        lab_src.height, lab_src.width, block_size=block_size, use_pbar=False
    ):
        lab = lab_src.read(label_band, window=win).ravel()
        if lab_nd is not None:
            lab = lab[lab != lab_nd]
        if lab.size == 0:
            continue
        uniq = np.unique(lab)
        pending.append(uniq)
        n_pending += uniq.size
        if n_pending > _PAIR_CONSOLIDATE_LEN and len(pending) > 1:
            merged = np.unique(np.concatenate(pending))
            pending = [merged]
            n_pending = merged.size
    if not pending:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.concatenate(pending))


class _LabelIndexer:
    """Map label values to compact ``0..N-1`` indices (dense LUT or sorted search)."""

    def __init__(self, ids: np.ndarray, dense_max_labels: int) -> None:
        self.ids = ids
        self.n_ids = int(ids.size)
        self.dense_lut: Optional[np.ndarray] = None
        if self.n_ids > 0:
            max_id = ids[-1]
            min_id = ids[0]
            if (
                int(min_id) >= 0
                and int(max_id) < int(dense_max_labels)
                and int(max_id) + 1 <= max(4 * self.n_ids, 1_000_000)
            ):
                lut = np.full(int(max_id) + 1, -1, dtype=np.int64)
                lut[ids.astype(np.int64)] = np.arange(self.n_ids, dtype=np.int64)
                self.dense_lut = lut

    @property
    def is_dense(self) -> bool:
        """``True`` when a dense lookup table is used."""
        return self.dense_lut is not None

    def map(self, labels: np.ndarray) -> np.ndarray:
        """Map label values (all present in ``ids``) to compact indices."""
        if self.dense_lut is not None:
            return self.dense_lut[labels.astype(np.int64, copy=False)]
        return np.searchsorted(self.ids, labels.astype(self.ids.dtype, copy=False))

    def lookup(self, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Map arbitrary label values, returning ``(indices, found_mask)``."""
        if self.n_ids == 0:
            return (
                np.zeros(labels.shape, dtype=np.int64),
                np.zeros(labels.shape, dtype=bool),
            )
        if self.dense_lut is not None:
            lab_i = labels.astype(np.int64, copy=False)
            in_rng = (lab_i >= 0) & (lab_i < self.dense_lut.size)
            idx = np.zeros(labels.shape, dtype=np.int64)
            idx[in_rng] = self.dense_lut[lab_i[in_rng]]
            found = in_rng & (idx >= 0)
            idx[~found] = 0
            return idx, found
        try:
            lab_c = labels.astype(self.ids.dtype, copy=False)
        except (TypeError, ValueError):
            lab_c = labels
        pos = np.searchsorted(self.ids, lab_c)
        pos_c = np.minimum(pos, self.n_ids - 1)
        found = (pos < self.n_ids) & (self.ids[pos_c] == lab_c)
        return np.where(found, pos_c, 0), found


def _normalise_stats(
    stats: Optional[Sequence[str]], percentiles: Optional[Sequence[float]]
) -> Tuple[Tuple[str, ...], Tuple[float, ...]]:
    """Validate the requested statistics and percentiles."""
    stat_list = tuple(_DEFAULT_STATS if stats is None else [str(s).lower() for s in stats])
    aliases = {"std": "stddev", "cnt": "count", "avg": "mean"}
    stat_list = tuple(aliases.get(s, s) for s in stat_list)
    bad = [s for s in stat_list if s not in LABEL_ZONAL_STATS]
    if bad:
        raise rsgispy.RSGISPyException(
            f"Unsupported label zonal statistic(s) {bad}; supported: "
            f"{list(LABEL_ZONAL_STATS)} (plus percentiles)."
        )
    pct_list: Tuple[float, ...] = ()
    if percentiles is not None:
        pct_list = tuple(float(p) for p in percentiles)
        for p_val in pct_list:
            if not 0.0 <= p_val <= 100.0:
                raise rsgispy.RSGISPyException(
                    f"Percentiles must be in [0, 100], got {p_val}."
                )
    if not stat_list and not pct_list:
        raise rsgispy.RSGISPyException(
            "At least one statistic or percentile must be requested."
        )
    return stat_list, pct_list


def _pct_stat_name(p_val: float) -> str:
    """Return the statistic name for a percentile (e.g. ``p10`` or ``p2.5``)."""
    return f"p{int(p_val)}" if float(p_val).is_integer() else f"p{p_val:g}"


def _fmt_category(val: float) -> str:
    """Format a category value for a column name."""
    return str(int(val)) if float(val).is_integer() else f"{val:g}"


def _init_feat_acc(fdef: _FeatDef, n_ids: int) -> None:
    """Allocate the per-label accumulators for one value band."""
    fdef.acc = {
        "n": np.zeros(n_ids, dtype=np.int64),
        "sum": np.zeros(n_ids, dtype=np.float64),
        "mean": np.zeros(n_ids, dtype=np.float64),
        "m2": np.zeros(n_ids, dtype=np.float64),
        "min": np.full(n_ids, np.inf, dtype=np.float64),
        "max": np.full(n_ids, -np.inf, dtype=np.float64),
    }


def _feat_valid_mask(fdef: _FeatDef, vals: np.ndarray) -> np.ndarray:
    """Return the per-pixel validity mask for one value band."""
    valid = np.isfinite(vals) if not fdef.is_int else np.ones(vals.shape, dtype=bool)
    if fdef.no_data is not None and not math.isnan(fdef.no_data):
        valid &= vals != fdef.no_data
    if fdef.min_vld is not None:
        valid &= vals >= fdef.min_vld
    if fdef.max_vld is not None:
        valid &= vals <= fdef.max_vld
    return valid


def _accumulate_moments(fdef: _FeatDef, g_v: np.ndarray, v_v: np.ndarray) -> None:
    """Merge one block's grouped moments into the global accumulators (Chan et al.)."""
    starts = _group_starts(g_v)
    ug = g_v[starts]
    n_b = np.diff(np.append(starts, g_v.size)).astype(np.int64)
    s_b = np.add.reduceat(v_v, starts)
    mean_b = s_b / n_b
    dev = v_v - np.repeat(mean_b, n_b)
    m2_b = np.add.reduceat(dev * dev, starts)
    acc = fdef.acc
    n_a = acc["n"][ug].astype(np.float64)
    n_ab = n_a + n_b
    delta = mean_b - acc["mean"][ug]
    acc["mean"][ug] = acc["mean"][ug] + delta * (n_b / n_ab)
    acc["m2"][ug] = acc["m2"][ug] + m2_b + (delta * delta) * (n_a * n_b / n_ab)
    acc["n"][ug] += n_b
    acc["sum"][ug] += s_b
    acc["min"][ug] = np.minimum(acc["min"][ug], np.minimum.reduceat(v_v, starts))
    acc["max"][ug] = np.maximum(acc["max"][ug], np.maximum.reduceat(v_v, starts))


def _pair_keys(fdef: _FeatDef, v_v: np.ndarray) -> np.ndarray:
    """Convert valid values into histogram keys for quantile/mode/category counts."""
    if fdef.pair_mode == "hist":
        bins = np.floor((v_v - fdef.hist_min) / fdef.hist_width)
        return np.clip(bins, 0, fdef.hist_nbins - 1)
    return v_v


def _block_reads(
    srcs: Mapping[str, DatasetReader],
    file_bands: Mapping[str, List[int]],
    win: Any,
) -> Dict[Tuple[str, int], np.ndarray]:
    """
    Read all the needed bands for a window, one read per file, each band in
    its own data type (a file such as a VRT stack may mix data types).
    """
    out: Dict[Tuple[str, int], np.ndarray] = {}
    for path, bands in file_bands.items():
        arrs = read_bands_native(srcs[path], bands, window=win)
        for band, arr in zip(bands, arrs):
            out[(path, band)] = arr.ravel()
    return out


def _run_label_pass(
    lab_src: DatasetReader,
    label_band: int,
    lab_nd: Optional[int],
    indexer: _LabelIndexer,
    srcs: Mapping[str, DatasetReader],
    feat_defs: Sequence[_FeatDef],
    block_size: Optional[int],
    use_pbar: bool,
    geom_acc: Optional[Dict[str, np.ndarray]],
    moments: bool,
) -> None:
    """One streamed block-wise pass accumulating label/value statistics."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    file_bands: Dict[str, List[int]] = {}
    for fdef in feat_defs:
        file_bands.setdefault(fdef.path, [])
        if fdef.band not in file_bands[fdef.path]:
            file_bands[fdef.path].append(fdef.band)

    for win in iter_dataset_windows(
        lab_src.height, lab_src.width, block_size=block_size, use_pbar=use_pbar
    ):
        lab = lab_src.read(label_band, window=win).ravel()
        if lab_nd is not None:
            pos = np.flatnonzero(lab != lab_nd)
        else:
            pos = np.arange(lab.size)
        if pos.size == 0:
            continue
        gidx = indexer.map(lab[pos])
        order = np.argsort(gidx, kind="stable")
        pos_s = pos[order]
        g_s = gidx[order]

        if geom_acc is not None:
            starts = _group_starts(g_s)
            ug = g_s[starts]
            cnts = np.diff(np.append(starts, g_s.size))
            geom_acc["count"][ug] += cnts
            win_w = int(win.width)
            rows = (pos_s // win_w + int(win.row_off)).astype(np.int64)
            cols = (pos_s % win_w + int(win.col_off)).astype(np.int64)
            geom_acc["min_row"][ug] = np.minimum(
                geom_acc["min_row"][ug], np.minimum.reduceat(rows, starts)
            )
            geom_acc["max_row"][ug] = np.maximum(
                geom_acc["max_row"][ug], np.maximum.reduceat(rows, starts)
            )
            geom_acc["min_col"][ug] = np.minimum(
                geom_acc["min_col"][ug], np.minimum.reduceat(cols, starts)
            )
            geom_acc["max_col"][ug] = np.maximum(
                geom_acc["max_col"][ug], np.maximum.reduceat(cols, starts)
            )
            geom_acc["sum_row"][ug] += np.add.reduceat(rows.astype(np.float64), starts)
            geom_acc["sum_col"][ug] += np.add.reduceat(cols.astype(np.float64), starts)

        if not feat_defs:
            continue
        blk_vals = _block_reads(srcs, file_bands, win)
        for fdef in feat_defs:
            vals = blk_vals[(fdef.path, fdef.band)][pos_s]
            fvalid = _feat_valid_mask(fdef, vals)
            if not np.any(fvalid):
                continue
            g_v = g_s[fvalid]
            v_v = vals[fvalid].astype(np.float64, copy=False)
            if moments:
                _accumulate_moments(fdef, g_v, v_v)
            if fdef.needs_pairs and fdef.pairs is not None and (
                not moments or fdef.pair_mode != "hist"
            ):
                fdef.pairs.add(g_v, _pair_keys(fdef, v_v))


def _feat_columns(
    fdef: _FeatDef,
    n_ids: int,
    out_no_data_val: float,
    col_pattern: str,
    cat_props: bool,
) -> Dict[str, np.ndarray]:
    """Derive the output columns for one value band from its accumulators."""
    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    acc = fdef.acc
    n_arr = acc["n"]
    has = n_arr > 0
    cols: Dict[str, np.ndarray] = {}

    def _name(stat: str) -> str:
        return col_pattern.format(
            feat=fdef.feat, stat=stat, img=fdef.img, band=fdef.band
        )

    def _fill(arr: np.ndarray) -> np.ndarray:
        return np.where(has, arr, float(out_no_data_val))

    pair_vals: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
    if fdef.needs_pairs and fdef.pairs is not None:
        g_p, k_p, c_p = fdef.pairs.result()
        if fdef.pair_mode == "hist":
            if fdef.hist_const:
                v_p = np.full(k_p.shape, fdef.hist_min)
            else:
                v_p = fdef.hist_min + (k_p + 0.5) * fdef.hist_width
        else:
            v_p = k_p
        pair_vals = (g_p, v_p, c_p)

    if fdef.categorical:
        if "count" in fdef.stats:
            cols[_name("count")] = n_arr.copy()
        assert pair_vals is not None
        g_p, v_p, c_p = pair_vals
        cols[_name("mode")] = _fill(_pair_mode(g_p, v_p, c_p, n_ids))
        if cat_props and fdef.categories is not None:
            denom = np.where(has, n_arr, 1).astype(np.float64)
            for cat in fdef.categories:
                sel = v_p == cat
                cnt = np.bincount(
                    g_p[sel], weights=c_p[sel].astype(np.float64), minlength=n_ids
                )
                cols[_name(f"prop_{_fmt_category(cat)}")] = _fill(cnt / denom)
        return cols

    quants: Dict[float, np.ndarray] = {}
    q_needed: List[float] = list(fdef.percentiles)
    if "median" in fdef.stats:
        q_needed.append(50.0)
    if q_needed and pair_vals is not None:
        quants = _pair_quantiles(*pair_vals, n_ids=n_ids, qs=sorted(set(q_needed)))

    for stat in fdef.stats:
        if stat == "count":
            cols[_name("count")] = n_arr.copy()
        elif stat == "sum":
            cols[_name("sum")] = _fill(acc["sum"])
        elif stat == "mean":
            cols[_name("mean")] = _fill(acc["mean"])
        elif stat == "stddev":
            safe_n = np.where(has, n_arr, 1).astype(np.float64)
            std = np.sqrt(np.maximum(acc["m2"] / safe_n, 0.0))
            cols[_name("stddev")] = _fill(std)
        elif stat == "min":
            cols[_name("min")] = _fill(acc["min"])
        elif stat == "max":
            cols[_name("max")] = _fill(acc["max"])
        elif stat == "median":
            cols[_name("median")] = _fill(quants[50.0])
        elif stat == "mode":
            assert pair_vals is not None
            cols[_name("mode")] = _fill(_pair_mode(*pair_vals, n_ids=n_ids))
    for p_val in fdef.percentiles:
        cols[_name(_pct_stat_name(p_val))] = _fill(quants[p_val])
    return cols


def calc_label_zonal_stats(
    label_img: Union[str, os.PathLike],
    band_specs: Union[Any, Sequence[Any]],
    stats: Optional[Sequence[str]] = None,
    percentiles: Optional[Sequence[float]] = None,
    label_band: int = 1,
    label_no_data_val: Optional[int] = 0,
    out_no_data_val: float = -9999.0,
    id_col: str = "seg_id",
    col_pattern: str = "{feat}_{stat}",
    quantile_method: str = "auto",
    hist_bins: int = 1000,
    categories: Optional[Mapping[str, Sequence[float]]] = None,
    cat_props: bool = True,
    pxl_count_col: Optional[str] = "pxl_count",
    inc_bbox: bool = False,
    inc_centroid: bool = False,
    dense_max_labels: int = _DEFAULT_DENSE_MAX_LABELS,
    *,
    block_size: Optional[int] = None,
    out_file: Optional[Union[str, os.PathLike]] = None,
    out_format: Optional[str] = None,
    out_compress: Optional[Union[str, bool]] = "snappy",
    use_pbar: bool = True,
) -> pd.DataFrame:
    """
    Calculate per-label zonal statistics for one or more image bands directly
    from a label raster (e.g., a segmentation or clump image), in a single
    streamed block-wise pass.

    Every pixel with the same label value (excluding ``label_no_data_val``)
    forms one zone. For each value band, pixels that are no data, non-finite,
    or outside ``[min_vld_val, max_vld_val]`` are excluded for that band only.
    Labels with no valid pixels in a band receive ``out_no_data_val`` for the
    band's statistics and a count of ``0``.

    Statistics (``stats``) may include ``'count'``, ``'sum'``, ``'mean'``,
    ``'stddev'`` (population standard deviation), ``'min'``, ``'max'``,
    ``'median'`` and ``'mode'``, plus any ``percentiles`` (``0..100``, linear
    interpolation as :func:`numpy.percentile`). The count, sum, mean, standard
    deviation, minimum and maximum are exact and accumulated with a numerically
    stable grouped (Chan et al.) merge. Median, percentiles and mode are
    computed from per-label value histograms:

    * integer bands (and ``quantile_method='exact'``): exact, using the pixel
      values themselves as histogram keys (memory grows with the number of
      distinct ``(label, value)`` pairs, which for float data can approach the
      number of valid pixels);
    * float bands with ``quantile_method='hist'`` (the ``'auto'`` default for
      float bands): approximated from a ``hist_bins``-bin histogram spanning
      the band's valid range (error at most half a bin width), requiring a
      second pass over the data.

    Bands defined as categorical (``BandSpec.data_type ==
    rsgispy.VAR_TYPE_CATEGORICAL``) do not produce continuous statistics;
    instead the ``'mode'``, ``'count'`` (if requested) and, when
    ``cat_props=True``, the proportion of each label's valid pixels in each
    category (``<feat>_prop_<category>``) are output. The category list for
    each band is taken from ``categories`` (keyed by feature name) or
    discovered from the data, and is returned in
    ``df.attrs['categories']`` so it can be reused for another scene
    (unseen categories are ignored; missing categories give ``0``).

    The value images must be on the same pixel grid (size, transform and CRS)
    as the label image; resample them first if necessary (e.g., with
    :func:`rsgispy.imageutils.warp.resample_img_to_match`).

    Label IDs may be any non-negative or signed integers including ``uint64``.
    A dense lookup table is used when the largest ID is small enough
    (``< dense_max_labels`` and not much larger than the number of labels);
    otherwise a compact sorted mapping is used, so sparse or very large IDs do
    not allocate memory proportional to the largest ID.

    Compared with :func:`rsgispy.zonalstats.polygon.calc_zonal_band_stats`
    on the polygonised labels (``METHOD_POLYCONTAINSPIXELCENTRE`` semantics,
    which this function reproduces exactly), it reads each pixel once rather
    than once per polygon and band. In a benchmark with a 2000 x 2000 pixel
    image, 40,000 labels and 4 bands (mean, stddev, min, max, count), it took
    0.24 s compared with 22 s for the polygon method (about 90 times faster);
    the speed-up grows with the number of labels and bands.

    :param label_img: Path to the label (segment/clump) raster.
    :param band_specs: Value bands: a :class:`~rsgispy.ml.stack.BandSpec`,
                       :class:`~rsgispy.imageutils.masking.ImageBandInfo`,
                       image path, dictionary, or a sequence of these
                       (``BandSpec`` provides per-band ``no_data_val``,
                       ``min_vld_val``, ``max_vld_val`` and ``data_type``).
                       ``None`` or ``[]`` computes only the per-label pixel
                       count, bounding box and centroid.
    :param stats: Statistics to compute for continuous bands
                  (Default = ``('mean', 'stddev', 'min', 'max', 'count')``).
    :param percentiles: Optional percentiles (``0..100``) to compute.
    :param label_band: 1-based band of ``label_img`` holding the labels
                       (Default = ``1``).
    :param label_no_data_val: Label value excluded from the zones (Default =
                              ``0``). If ``None``, the label image header
                              no-data value is used (if any).
    :param out_no_data_val: Output value for statistics of labels with no
                            valid pixels (Default = ``-9999.0``).
    :param id_col: Output label ID column name (Default = ``'seg_id'``).
    :param col_pattern: Output column name pattern using ``{feat}`` (feature
                        name as used by the pixel ML workflow: the band name
                        if given, else ``<image name>_b<band>``), ``{stat}``,
                        ``{img}`` and ``{band}`` (Default = ``'{feat}_{stat}'``).
    :param quantile_method: ``'auto'``, ``'exact'`` or ``'hist'`` (see above;
                            Default = ``'auto'``).
    :param hist_bins: Number of histogram bins for ``'hist'`` quantiles
                      (Default = ``1000``).
    :param categories: Optional mapping of categorical feature name to the
                       list of category values for proportion columns.
    :param cat_props: If ``True``, output category proportion columns for
                      categorical bands (Default = ``True``).
    :param pxl_count_col: Column for the total number of pixels per label
                          (independent of value validity), or ``None`` to omit
                          (Default = ``'pxl_count'``).
    :param inc_bbox: If ``True``, add the pixel bounding box of each label
                     (``min_col``, ``max_col``, ``min_row``, ``max_row``).
    :param inc_centroid: If ``True``, add the pixel-centre centroid of each
                         label in map coordinates (``centroid_x``,
                         ``centroid_y``).
    :param dense_max_labels: Largest label ID for which a dense lookup table
                             is allowed (Default = ``50_000_000``).
    :param block_size: Optional processing block size in pixels.
    :param out_file: Optional output table path (Parquet by default; any
                     format supported by
                     :func:`rsgispy.tools.tables.write_dataframe_table`).
    :param out_format: Optional output table format overriding extension
                       detection.
    :param out_compress: Parquet compression (Default = ``'snappy'``).
    :param use_pbar: If ``True``, display progress bars (Default = ``True``).
    :return: :class:`pandas.DataFrame` with one row per label (sorted by
             ``id_col``) and one column per statistic.

    .. code:: python

        from rsgispy.ml.stack import BandSpec
        import rsgispy.zonalstats.labels as zs_lbl

        df = zs_lbl.calc_label_zonal_stats(
            "segments.kea",
            [
                BandSpec("s2_sref.tif", name="s2", bands=(3, 4, 8),
                         band_names=("Red", "NIR", "SWIR1"), no_data_val=0),
                BandSpec("landcover.tif", name="lc", data_type=2),
            ],
            stats=["mean", "stddev", "median"],
            out_file="segment_stats.parquet",
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    lab_path = os.fspath(label_img)
    ensure_file_exists(lab_path, label="Label image")
    stat_list, pct_list = _normalise_stats(stats, percentiles)
    q_method = str(quantile_method).strip().lower()
    if q_method not in ("auto", "exact", "hist"):
        raise rsgispy.RSGISPyException(
            f"quantile_method must be 'auto', 'exact' or 'hist', got {quantile_method!r}."
        )
    if int(hist_bins) < 1:
        raise rsgispy.RSGISPyException(f"hist_bins must be >= 1, got {hist_bins}.")
    if not id_col:
        raise rsgispy.RSGISPyException("id_col must be a non-empty string.")

    feat_defs = _resolve_feat_defs(band_specs)
    cat_lut = dict(categories) if categories is not None else {}
    for fdef in feat_defs:
        fdef.stats = stat_list
        fdef.percentiles = () if fdef.categorical else pct_list
        if fdef.categorical:
            fdef.needs_pairs = True
            fdef.pair_mode = "exact"
            if fdef.feat in cat_lut:
                fdef.categories = sorted(float(c) for c in cat_lut[fdef.feat])
        else:
            fdef.needs_pairs = bool(
                pct_list or ("median" in stat_list) or ("mode" in stat_list)
            )
            if q_method == "hist" or (q_method == "auto" and not fdef.is_int):
                fdef.pair_mode = "hist"
            else:
                fdef.pair_mode = "exact"
        if fdef.needs_pairs:
            fdef.pairs = _PairCounter()

    with ExitStack() as stack:
        lab_src = stack.enter_context(rasterio.open(lab_path, "r"))
        validate_band_index(label_band, lab_src.count, lab_path)
        lab_dtype = np.dtype(lab_src.dtypes[label_band - 1])
        if not np.issubdtype(lab_dtype, np.integer):
            raise rsgispy.RSGISPyException(
                f"Label image '{lab_path}' band {label_band} must be an integer "
                f"type, got '{lab_dtype}'."
            )
        lab_nd = _resolve_label_nodata(lab_src, label_band, label_no_data_val)
        srcs: Dict[str, DatasetReader] = {}
        for fdef in feat_defs:
            if fdef.path not in srcs:
                srcs[fdef.path] = stack.enter_context(rasterio.open(fdef.path, "r"))
                _check_same_grid(lab_src, srcs[fdef.path], lab_path, fdef.path)

        ids = _scan_label_ids(lab_src, label_band, lab_nd, block_size)
        indexer = _LabelIndexer(ids, dense_max_labels)
        n_ids = indexer.n_ids
        for fdef in feat_defs:
            _init_feat_acc(fdef, n_ids)

        geom_acc: Dict[str, np.ndarray] = {
            "count": np.zeros(n_ids, dtype=np.int64),
            "min_row": np.full(n_ids, np.iinfo(np.int64).max, dtype=np.int64),
            "max_row": np.full(n_ids, -1, dtype=np.int64),
            "min_col": np.full(n_ids, np.iinfo(np.int64).max, dtype=np.int64),
            "max_col": np.full(n_ids, -1, dtype=np.int64),
            "sum_row": np.zeros(n_ids, dtype=np.float64),
            "sum_col": np.zeros(n_ids, dtype=np.float64),
        }
        _run_label_pass(
            lab_src,
            label_band,
            lab_nd,
            indexer,
            srcs,
            feat_defs,
            block_size,
            use_pbar,
            geom_acc,
            moments=True,
        )

        hist_feats = [f for f in feat_defs if f.needs_pairs and f.pair_mode == "hist"]
        if hist_feats:
            for fdef in hist_feats:
                has = fdef.acc["n"] > 0
                if np.any(has):
                    v_min = float(fdef.acc["min"][has].min())
                    v_max = float(fdef.acc["max"][has].max())
                else:
                    v_min, v_max = 0.0, 0.0
                fdef.hist_min = v_min
                fdef.hist_const = v_max <= v_min
                fdef.hist_nbins = 1 if fdef.hist_const else int(hist_bins)
                fdef.hist_width = (
                    1.0 if fdef.hist_const else (v_max - v_min) / float(hist_bins)
                )
            _run_label_pass(
                lab_src,
                label_band,
                lab_nd,
                indexer,
                srcs,
                hist_feats,
                block_size,
                use_pbar,
                None,
                moments=False,
            )
        transform = lab_src.transform

    out_cols: Dict[str, Any] = {id_col: ids}
    if pxl_count_col:
        out_cols[pxl_count_col] = geom_acc["count"]
    if inc_bbox:
        for key in ("min_col", "max_col", "min_row", "max_row"):
            out_cols[key] = geom_acc[key]
    if inc_centroid:
        cnt_f = np.maximum(geom_acc["count"], 1).astype(np.float64)
        mean_col = geom_acc["sum_col"] / cnt_f + 0.5
        mean_row = geom_acc["sum_row"] / cnt_f + 0.5
        out_cols["centroid_x"] = transform.a * mean_col + transform.b * mean_row + transform.c
        out_cols["centroid_y"] = transform.d * mean_col + transform.e * mean_row + transform.f

    found_cats: Dict[str, List[float]] = {}
    for fdef in feat_defs:
        if fdef.categorical and fdef.categories is None and fdef.pairs is not None:
            _, k_p, _ = fdef.pairs.result()
            fdef.categories = [float(c) for c in np.unique(k_p)]
        if fdef.categorical:
            found_cats[fdef.feat] = list(fdef.categories or [])
        for col_name, col_vals in _feat_columns(
            fdef, n_ids, out_no_data_val, col_pattern, cat_props
        ).items():
            if col_name in out_cols:
                raise rsgispy.RSGISPyException(
                    f"Duplicate output column '{col_name}'; check col_pattern and "
                    "feature names."
                )
            out_cols[col_name] = col_vals

    out_df = pd.DataFrame(out_cols)
    out_df.attrs["categories"] = found_cats
    if out_file is not None:
        write_dataframe_table(
            out_df, out_file, out_format=out_format, out_compress=out_compress
        )
    return out_df


def join_label_stats_to_vec(
    vec_file: VectorInput,
    stats_df: Union[pd.DataFrame, str, os.PathLike],
    vec_lyr: Optional[str] = None,
    id_col: str = "seg_id",
    vec_id_col: Optional[str] = None,
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    *,
    out_format: Optional[str] = None,
    out_compress: Optional[Union[str, bool]] = None,
    overwrite: bool = False,
) -> Optional[gpd.GeoDataFrame]:
    """
    Join a label statistics table (e.g., from :func:`calc_label_zonal_stats`)
    onto a vector layer by the label ID column, keeping the vector feature
    order (a left join: features without statistics receive missing values).

    :param vec_file: Input vector file path or
                     :class:`geopandas.GeoDataFrame` (e.g., the output of
                     :func:`rsgispy.segmentation.polygonise_segments`).
    :param stats_df: Statistics :class:`pandas.DataFrame` or table file path
                     (Parquet, CSV, etc.).
    :param vec_lyr: Optional input vector layer name.
    :param id_col: Label ID column in ``stats_df`` (Default = ``'seg_id'``).
    :param vec_id_col: Label ID column in the vector layer (Default = ``id_col``).
    :param out_vec_file: Optional output vector path; if ``None``, the joined
                         :class:`geopandas.GeoDataFrame` is returned.
    :param out_vec_lyr: Optional output layer name.
    :param out_format: Optional output vector driver.
    :param out_compress: Optional GeoParquet compression setting.
    :param overwrite: If ``True``, overwrite an existing output file.
    :return: Joined :class:`geopandas.GeoDataFrame` when ``out_vec_file`` is
             ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.zonalstats.labels as zs_lbl

        zs_lbl.join_label_stats_to_vec(
            "segments.parquet", "segment_stats.parquet",
            out_vec_file="segments_stats.parquet",
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    if not isinstance(stats_df, pd.DataFrame):
        from rsgispy.tools.tables import (  # pylint: disable=import-outside-toplevel
            read_dataframe_table,
        )

        stats_df = read_dataframe_table(stats_df)
    v_id = vec_id_col if vec_id_col is not None else id_col
    if id_col not in stats_df.columns:
        raise rsgispy.RSGISPyException(
            f"Column '{id_col}' is not in the statistics table."
        )
    gdf = read_vec_gdf(vec_file, vec_lyr)
    if v_id not in gdf.columns:
        raise rsgispy.RSGISPyException(
            f"Column '{v_id}' is not in the vector layer."
        )
    if stats_df[id_col].duplicated().any():
        raise rsgispy.RSGISPyException(
            f"Column '{id_col}' in the statistics table contains duplicate IDs."
        )
    stats_cols = [c for c in stats_df.columns if c != id_col]
    clash = [c for c in stats_cols if c in gdf.columns]
    if clash:
        raise rsgispy.RSGISPyException(
            f"Statistics columns {clash} already exist in the vector layer."
        )
    right = stats_df.rename(columns={id_col: v_id}) if v_id != id_col else stats_df
    right = right.astype({v_id: gdf[v_id].dtype})
    joined = gdf.merge(right, on=v_id, how="left")
    joined = gpd.GeoDataFrame(joined, geometry=gdf.geometry.name, crs=gdf.crs)
    return finish_vec_output(
        joined,
        out_vec_file,
        out_vec_lyr=out_vec_lyr,
        out_format=out_format,
        overwrite=overwrite,
        out_compress=out_compress,
    )


def rasterise_label_values(
    label_img: Union[str, os.PathLike],
    ids: Sequence[Any],
    values: Any,
    output_img: Union[str, os.PathLike],
    label_band: int = 1,
    label_no_data_val: Optional[int] = 0,
    out_no_data_val: Optional[float] = 0,
    band_names: Optional[Sequence[str]] = None,
    thematic: bool = False,
    clr_tab: Optional[Mapping[int, Sequence[int]]] = None,
    dense_max_labels: int = _DEFAULT_DENSE_MAX_LABELS,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    calc_stats: bool = True,
    block_size: Optional[int] = None,
    use_pbar: bool = True,
    creation_opts: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
) -> None:
    """
    Write per-label values onto the pixel grid of a label raster by streaming
    the label raster block by block through a ``label -> value`` lookup (no
    geometry burning). Pixels whose label is the label no data value, or is
    not in ``ids``, are set to ``out_no_data_val``.

    :param label_img: Path to the label (segment/clump) raster.
    :param ids: Sequence of label IDs (unique).
    :param values: Values per ID: 1D (one output band) or 2D
                   ``(n_ids, n_bands)`` array-like.
    :param output_img: Output raster path.
    :param label_band: 1-based band of ``label_img`` holding the labels
                       (Default = ``1``).
    :param label_no_data_val: Label no data value (Default = ``0``); if
                              ``None``, the header no-data value is used.
    :param out_no_data_val: Output no data value (Default = ``0``).
    :param band_names: Optional output band names.
    :param thematic: If ``True`` (integer outputs), mark the output as
                     thematic and compute thematic statistics.
    :param clr_tab: Optional colour table ``{value: (R, G, B[, A])}`` written
                    to band 1 (``uint8``/``uint16`` outputs only).
    :param dense_max_labels: Largest label ID for which a dense lookup table
                             is allowed (Default = ``50_000_000``).
    :param gdalformat: Output GDAL driver format (Default = ``'GTIFF'``).
    :param datatype: Output ``rsgispy.TYPE_*`` constant; if ``None``, derived
                     from the dtype of ``values``.
    :param calc_stats: If ``True``, populate image statistics and pyramids
                       (Default = ``True``).
    :param block_size: Optional processing block size in pixels.
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).
    :param creation_opts: Optional GDAL creation options.

    .. code:: python

        import numpy as np
        import rsgispy.zonalstats.labels as zs_lbl

        zs_lbl.rasterise_label_values(
            "segments.kea", [1, 2, 3], np.array([10, 20, 10], dtype=np.uint8),
            "seg_classes.tif", thematic=True,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    lab_path = os.fspath(label_img)
    out_path = os.fspath(output_img)
    ensure_file_exists(lab_path, label="Label image")
    ids_arr = np.asarray(ids)
    vals_arr = np.asarray(values)
    if ids_arr.ndim != 1:
        raise rsgispy.RSGISPyException("ids must be a 1D sequence.")
    if not np.issubdtype(ids_arr.dtype, np.integer):
        if ids_arr.size and not np.all(np.mod(ids_arr.astype(np.float64), 1.0) == 0):
            raise rsgispy.RSGISPyException("ids must be integer label values.")
        ids_arr = ids_arr.astype(np.int64)
    if vals_arr.ndim == 1:
        vals_arr = vals_arr[:, np.newaxis]
    if vals_arr.ndim != 2 or vals_arr.shape[0] != ids_arr.size:
        raise rsgispy.RSGISPyException(
            "values must be 1D or 2D with one row per ID "
            f"(got shape {vals_arr.shape} for {ids_arr.size} IDs)."
        )
    n_out = int(vals_arr.shape[1])
    if band_names is not None and len(band_names) != n_out:
        raise rsgispy.RSGISPyException(
            f"band_names has {len(band_names)} entries for {n_out} output bands."
        )
    order = np.argsort(ids_arr, kind="stable")
    ids_s = ids_arr[order]
    vals_s = vals_arr[order]
    if ids_s.size > 1 and np.any(ids_s[1:] == ids_s[:-1]):
        raise rsgispy.RSGISPyException("ids contains duplicate label values.")

    if datatype is None:
        datatype = rsgis_dtype_from_np_dtype(
            _infer_out_np_dtype(vals_s, out_no_data_val)
        )
    _, out_rio_dtype, out_np_dtype = resolve_out_dtype(datatype)
    norm_opts = normalise_creation_options(creation_opts)
    nd_fill = 0 if out_no_data_val is None else out_no_data_val
    is_int_out = np.issubdtype(np.dtype(out_np_dtype), np.integer)
    if is_int_out:
        info = np.iinfo(np.dtype(out_np_dtype))
        if not info.min <= float(nd_fill) <= info.max:
            raise rsgispy.RSGISPyException(
                f"out_no_data_val {out_no_data_val} does not fit output type "
                f"'{out_rio_dtype}'."
            )
    vals_out = vals_s.astype(out_np_dtype)
    indexer = _LabelIndexer(ids_s, dense_max_labels)

    with ExitStack() as stack:
        lab_src = stack.enter_context(rasterio.open(lab_path, "r"))
        validate_band_index(label_band, lab_src.count, lab_path)
        lab_nd = _resolve_label_nodata(lab_src, label_band, label_no_data_val)
        dst = create_output_from_dataset(
            stack=stack,
            src=lab_src,
            output_img=out_path,
            gdalformat=gdalformat,
            out_rio_dtype=out_rio_dtype,
            count=n_out,
            creation_opts=norm_opts,
        )
        if out_no_data_val is not None:
            dst.nodata = out_no_data_val
        if band_names is not None:
            for b_idx, b_name in enumerate(band_names, start=1):
                dst.set_band_description(b_idx, str(b_name))
        for win in iter_dataset_windows(
            lab_src.height, lab_src.width, block_size=block_size, use_pbar=use_pbar
        ):
            lab = lab_src.read(label_band, window=win)
            idx, found = indexer.lookup(lab.ravel())
            if lab_nd is not None:
                found &= lab.ravel() != lab_nd
            out_blk = np.empty((n_out, lab.size), dtype=out_np_dtype)
            out_blk[:] = np.asarray(nd_fill).astype(out_np_dtype)
            if np.any(found):
                out_blk[:, found] = vals_out[idx[found]].T
            dst.write(out_blk.reshape((n_out, lab.shape[0], lab.shape[1])), window=win)

    _finalise_label_raster(
        out_path,
        out_no_data_val,
        thematic and is_int_out,
        clr_tab,
        out_np_dtype,
        calc_stats,
        use_pbar,
        gdalformat,
    )


def _infer_out_np_dtype(vals: np.ndarray, out_no_data_val: Optional[float]) -> np.dtype:
    """Select a GDAL-compatible output dtype fitting ``vals`` and the no data value."""
    if vals.dtype == np.bool_:
        return np.dtype(np.uint8)
    if not np.issubdtype(vals.dtype, np.integer):
        return np.dtype(np.float32)
    cands = [vals.astype(np.float64)] if vals.size else []
    if out_no_data_val is not None:
        cands.append(np.array([float(out_no_data_val)]))
    if not cands:
        return np.dtype(np.uint8)
    all_v = np.concatenate([c.ravel() for c in cands])
    v_min, v_max = float(all_v.min()), float(all_v.max())
    for cand in (np.uint8, np.uint16, np.int16, np.uint32, np.int32):
        info = np.iinfo(cand)
        if info.min <= v_min and v_max <= info.max:
            return np.dtype(cand)
    return np.dtype(np.float64)


def _finalise_label_raster(
    out_path: str,
    out_no_data_val: Optional[float],
    thematic: bool,
    clr_tab: Optional[Mapping[int, Sequence[int]]],
    out_np_dtype: Any,
    calc_stats: bool,
    use_pbar: bool,
    gdalformat: str,
) -> None:
    """Set thematic metadata, statistics, pyramids and colour table."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=import-outside-toplevel
    from rsgispy.imageutils.colourtable import (
        define_colour_table,
        set_img_not_thematic,
        set_img_thematic,
    )
    from rsgispy.imageutils.stats import pop_thmt_img_stats

    is_cog = gdalformat.upper().strip() == "COG"
    if thematic:
        if not is_cog:
            set_img_thematic(out_path)
        if calc_stats:
            pop_thmt_img_stats(
                out_path,
                add_clr_tab=True,
                calc_pyramids=True,
                ignore_zero=(out_no_data_val == 0),
                use_pbar=use_pbar,
                ignore_cog=is_cog,
            )
            if out_no_data_val is not None and out_no_data_val != 0 and not is_cog:
                with rasterio.open(out_path, "r+") as dst:
                    dst.nodata = out_no_data_val
    else:
        if not is_cog:
            set_img_not_thematic(out_path)
        maybe_pop_stats(
            output_img=out_path,
            calc_stats=calc_stats,
            no_data_val=out_no_data_val,
            thematic=False,
            use_pbar=use_pbar,
            gdalformat=gdalformat,
        )
    if clr_tab is not None and np.dtype(out_np_dtype) in (
        np.dtype(np.uint8),
        np.dtype(np.uint16),
    ):
        define_colour_table(
            out_path, {int(k): list(v) for k, v in clr_tab.items()}, img_band=1
        )
