# pylint: disable=too-many-lines
"""
The imageutils.composite module creates per-pixel composites of a stack of
observations (e.g. the cloud-masked scenes of a season) on the same pixel
grid:

* :func:`create_geomedian_composite` — the multi-band geometric median
  (Roberts et al., 2017), with its median absolute deviations;
* :func:`create_quality_mosaic` — each pixel takes the whole observation
  with the largest (or smallest, or a percentile of the) quality value, like
  Google Earth Engine's ``qualityMosaic``; :func:`create_max_ndvi_composite`
  is the maximum NDVI preset;
* :func:`create_land_water_composite` — a quality mosaic using one quality
  over land and another (or a medoid or geomedian) over water;
  :func:`create_max_ndvi_ndwi_composite` is the maximum NDVI / NDWI preset;
* :func:`create_medoid_composite` — the observation nearest all the others;
* :func:`create_percentile_composite` — per-band percentiles (e.g. the
  median) over the observations.

Every composite has a tile-graph node builder (:func:`geomedian_node`,
:func:`quality_mosaic_node`, :func:`land_water_composite_node`,
:func:`medoid_node` and :func:`percentile_composite_node`) taking one
:class:`~rsgispy.imageprocess.TileNode` per observation; the file-based
functions are wrappers around them and give the same result. A builder
returns a :class:`CompositeNode` (the composite bands), whose other outputs
(reference image, quality, count, ...) are nodes available as attributes,
so they can be written by other sinks of the same graph without computing
the composite twice.

Common rules:

* **Valid observations**: an observation is valid at a pixel only when every
  composited band (``bands``) is valid there (finite and not the no-data
  value), and, for a quality given as a band number or an expression, every
  band the quality uses.
* **Empty pixels**: a pixel with no valid observation is no data in every
  output.
* **Reference image**: a thematic ``uint16`` image of the 1-based index into
  ``input_imgs`` of the observation used at each pixel (``0`` = none), with a
  colour table and statistics. For the geomedian, which is not an
  observation, it is the observation nearest the geomedian.
* **Grids**: the images must be on the same pixel grid (size, transform and
  CRS); otherwise an :class:`rsgispy.RSGISPyException` is raised. Resample
  them first, e.g. with :func:`~rsgispy.imageutils.warp.resample_img_to_match`.
* **Processing**: block by block, so memory does not grow with the image
  size, and the results are identical for any ``block_size`` and number of
  threads (``n_jobs``).
"""

import colorsys
import math
import json
import os
import re
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import rasterio

import rsgispy
from rsgispy._numba import optional_njit
from rsgispy._tilegraph._wrap import WrapperDiskSink
from rsgispy.imageprocess import (
    Cast,
    NodeMeta,
    RasterSource,
    SelectBands,
    SetNoData,
    TileContext,
    TileGraph,
    TileNode,
    TileTensor,
    cast_result_block,
    ensure_file_exists,
    fits_dtype_range,
    infer_node_meta,
    normalise_creation_options,
)
from rsgispy.imageutils.colourtable import CLASS_NAMES_TAG, define_colour_table, set_img_thematic
from rsgispy.imageutils.stats import pop_thmt_img_stats
from rsgispy.tools.stats import compute_nanpercentiles, validate_percentiles_0_100

__all__ = [
    "CompositeNode",
    "RSGISLibLandWaterRule",
    "create_geomedian_composite",
    "geomedian_node",
    "create_quality_mosaic",
    "quality_mosaic_node",
    "create_max_ndvi_composite",
    "create_land_water_composite",
    "land_water_composite_node",
    "create_max_ndvi_ndwi_composite",
    "create_medoid_composite",
    "medoid_node",
    "create_percentile_composite",
    "percentile_composite_node",
]

#: Band name of the reference (observation index) output.
REF_BAND_NAME = "Reference"
#: Band name of the quality output.
QUALITY_BAND_NAME = "Quality"
#: Band name of the count output.
COUNT_BAND_NAME = "Count"
#: Band names of the geomedian median absolute deviations.
MAD_BAND_NAMES = ("EMAD", "SMAD", "BCMAD")
#: Band name of the land / water mask output.
LW_BAND_NAME = "LandWater"
#: Land class of the land / water mask.
LAND_CLASS = 1
#: Water class of the land / water mask.
WATER_CLASS = 2
#: Colours of the land / water mask (land green, water blue).
LW_COLOURS = {LAND_CLASS: "#4CAF50", WATER_CLASS: "#1E88E5"}
LW_CLASS_NAMES = {LAND_CLASS: "Land", WATER_CLASS: "Water"}

QualitySpec = Union[int, str, Callable[[np.ndarray], np.ndarray]]
SourcesArg = Union[TileNode, Sequence[TileNode]]


# ---------------------------------------------------------------------------
# Observations, validity and qualities
# ---------------------------------------------------------------------------


def _source_list(sources: SourcesArg) -> Tuple[TileNode, ...]:
    """Return the observation nodes as a tuple, checking them."""
    src = (sources,) if isinstance(sources, TileNode) else tuple(sources)
    if not src:
        raise rsgispy.RSGISPyException("At least one observation is required.")
    for node in src:
        if not isinstance(node, TileNode):
            raise rsgispy.RSGISPyException(
                f"Observations must be TileNode objects, got {type(node).__name__}."
            )
    return src


def _band_index_list(bands: Optional[Union[int, Sequence[int]]], n_bands: int) -> List[int]:
    """0-based indices of the composited ``bands`` (1-based; all when ``None``)."""
    if bands is None:
        return list(range(n_bands))
    band_list = [int(bands)] if isinstance(bands, (int, np.integer)) else [int(b) for b in bands]
    if not band_list:
        raise rsgispy.RSGISPyException("bands must contain at least one band.")
    if len(set(band_list)) != len(band_list):
        raise rsgispy.RSGISPyException(f"bands contains duplicate bands: {band_list}.")
    for band in band_list:
        if band < 1 or band > n_bands:
            raise rsgispy.RSGISPyException(
                f"Band {band} requested but the observations have {n_bands} "
                "band(s) (bands are 1-based)."
            )
    return [b - 1 for b in band_list]


class _ObsInfo:
    """Band layout and no-data value of the observations of a composite."""

    # pylint: disable=too-few-public-methods

    def __init__(
        self,
        n_obs: int,
        n_bands: int,
        band_names: Optional[Tuple[str, ...]],
        out_idx: List[int],
        nodata: Optional[float],
    ) -> None:
        self.n_obs = n_obs
        self.n_bands = n_bands
        self.band_names = band_names
        self.out_idx = out_idx
        self.n_out = len(out_idx)
        self.nodata = None if nodata is None else float(nodata)

    def out_names(self) -> Optional[List[str]]:
        """Names of the composited bands (``None`` when unknown)."""
        if self.band_names is None:
            return None
        return [self.band_names[i] for i in self.out_idx]

    def base_names(self) -> List[str]:
        """Names of the composited bands, ``b<band>`` when unknown."""
        names = self.out_names()
        if names is not None:
            return names
        return [f"b{i + 1}" for i in self.out_idx]


def _resolve_obs(
    sources: Sequence[TileNode],
    bands: Optional[Union[int, Sequence[int]]],
    no_data_val: Optional[float],
) -> _ObsInfo:
    """Read the observations' metadata and check that they have the same bands."""
    metas = [infer_node_meta(src) for src in sources]
    n_bands = metas[0].n_bands
    if n_bands is None:
        raise rsgispy.RSGISPyException(
            "The number of bands of the observation nodes must be known (give "
            "the node a metadata function or use SelectBands)."
        )
    for idx, meta in enumerate(metas[1:], start=2):
        if meta.n_bands is not None and meta.n_bands != n_bands:
            raise rsgispy.RSGISPyException(
                "All observations must have the same bands: observation 1 has "
                f"{n_bands} band(s) but observation {idx} has {meta.n_bands}."
            )
    out_idx = _band_index_list(bands, int(n_bands))
    nodata = no_data_val if no_data_val is not None else metas[0].nodata
    return _ObsInfo(len(sources), int(n_bands), metas[0].band_names, out_idx, nodata)


def _band_valid(blk: np.ndarray, nodata: Optional[float]) -> np.ndarray:
    """Per-band validity of an observation block (finite and not no data)."""
    valid = np.isfinite(blk)
    if nodata is not None and not math.isnan(nodata):
        valid &= blk != nodata
    return valid


def _prep_obs(
    blocks: Sequence[np.ndarray], nodata: Optional[float]
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Return the observation blocks (float64) with invalid values set to NaN,
    and the per-band validity of each.
    """
    nan_blks: List[np.ndarray] = []
    valid_bands: List[np.ndarray] = []
    for blk in blocks:
        arr = np.asarray(blk, dtype=np.float64)
        valid = _band_valid(arr, nodata)
        nan_blks.append(np.where(valid, arr, np.nan))
        valid_bands.append(valid)
    return nan_blks, valid_bands


def _obs_valid(valid_bands: Sequence[np.ndarray], check_idx: Sequence[int]) -> np.ndarray:
    """``(n_obs, rows, cols)`` validity: every band in ``check_idx`` is valid."""
    idx = list(check_idx)
    return np.stack([vb[idx].all(axis=0) for vb in valid_bands], axis=0)


def _check_bands(out_idx: Sequence[int], used: Optional[Sequence[int]]) -> List[int]:
    """Bands whose validity is checked: the composited bands and those used."""
    if used is None:
        return list(out_idx)
    return sorted(set(out_idx) | set(used))


class _Quality:
    """
    A quality: a 1-based band number, a band-maths expression over ``b1`` …
    ``bn`` (the bands of each observation, in ``pymu-eval`` / muParser
    syntax, as :func:`rsgispy.imagecalc.maths.image_band_math`), or a
    callable taking a ``(n_bands, rows, cols)`` block (invalid values are
    NaN) and returning a ``(rows, cols)`` array.
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(self, spec: QualitySpec, n_bands: int, label: str = "quality") -> None:
        self.label = label
        self.n_bands = int(n_bands)
        self.band = -1
        self.expr = ""
        self.func: Optional[Callable[[np.ndarray], np.ndarray]] = None
        self.var_names = [f"b{i + 1}" for i in range(self.n_bands)]
        if isinstance(spec, (bool, np.bool_)):
            raise rsgispy.RSGISPyException(
                f"{label} must be a band number, an expression or a callable."
            )
        if isinstance(spec, (int, np.integer)):
            band = int(spec)
            if band < 1 or band > self.n_bands:
                raise rsgispy.RSGISPyException(
                    f"{label} band {band} is not in 1..{self.n_bands} (bands are 1-based)."
                )
            self.kind = "band"
            self.band = band - 1
            self.used: Optional[List[int]] = [band - 1]
        elif isinstance(spec, str):
            expr = spec.strip()
            if not expr:
                raise rsgispy.RSGISPyException(f"{label} expression must not be empty.")
            used = sorted({int(num) - 1 for num in re.findall(r"\bb(\d+)\b", expr)})
            for b_idx in used:
                if b_idx < 0 or b_idx >= self.n_bands:
                    raise rsgispy.RSGISPyException(
                        f"{label} expression '{expr}' uses b{b_idx + 1} but the "
                        f"observations have {self.n_bands} band(s)."
                    )
            self.kind = "expr"
            self.expr = expr
            self.used = used
            self.new_state()  # check the expression now
        elif callable(spec):
            self.kind = "callable"
            self.func = spec
            self.used = None
        else:
            raise rsgispy.RSGISPyException(
                f"{label} must be a band number, an expression or a callable, "
                f"got {type(spec).__name__}."
            )

    def new_state(self) -> Any:
        """A new expression parser (one per run and thread), else ``None``."""
        if self.kind != "expr":
            return None
        # pylint: disable-next=import-outside-toplevel
        from rsgispy.imagecalc.maths import _create_parser

        return _create_parser(self.expr, self.var_names)

    def evaluate(self, nan_blk: np.ndarray, state: Any) -> np.ndarray:
        """Quality ``(rows, cols)`` of one observation block (NaN = invalid)."""
        shape = nan_blk.shape[1:]
        if self.kind == "band":
            return nan_blk[self.band]
        if self.kind == "callable":
            res = np.asarray(self.func(nan_blk), dtype=np.float64)  # type: ignore[misc]
            if res.shape != shape:
                if res.ndim == 0:
                    return np.full(shape, float(res))
                raise rsgispy.RSGISPyException(
                    f"The {self.label} callable returned an array of shape "
                    f"{res.shape}; expected {shape}."
                )
            return res
        arrs = [np.ascontiguousarray(nan_blk[i]) for i in range(self.n_bands)]
        for name, arr in zip(self.var_names, arrs):
            state.define_var(name, arr)
        with np.errstate(all="ignore"):
            res = np.array(state.eval(), dtype=np.float64)
        if res.ndim == 0:
            return np.full(shape, float(res))
        return res.reshape(shape)

    def describe(self) -> str:
        """Short description for ``TileGraph.describe``."""
        if self.kind == "band":
            return f"band {self.band + 1}"
        if self.kind == "expr":
            return f"'{self.expr}'"
        return getattr(self.func, "__name__", type(self.func).__name__)


def _check_mode(
    mode: str, percentile: Optional[float], mode_arg: str = "mode",
    pct_arg: str = "percentile",
) -> Tuple[str, Optional[float]]:
    """Validate a selection mode (``max``, ``min`` or ``percentile``)."""
    mode_l = str(mode).lower()
    if mode_l not in ("max", "min", "percentile"):
        raise rsgispy.RSGISPyException(
            f"{mode_arg} must be 'max', 'min' or 'percentile', got {mode!r}."
        )
    if mode_l == "percentile":
        if percentile is None:
            raise rsgispy.RSGISPyException(
                f"{pct_arg} must be given when {mode_arg}='percentile'."
            )
        validate_percentiles_0_100(percentile)
        return mode_l, float(percentile)
    if percentile is not None:
        raise rsgispy.RSGISPyException(
            f"{pct_arg} is only used with {mode_arg}='percentile'."
        )
    return mode_l, None


# ---------------------------------------------------------------------------
# Selection kernels
# ---------------------------------------------------------------------------


def _select_by_quality(
    qual: np.ndarray, cand: np.ndarray, mode: str, percentile: Optional[float]
) -> np.ndarray:
    """
    0-based index (``-1``: none) of the candidate observation with the
    largest (``max``) or smallest (``min``) quality, or the quality nearest
    the percentile ``percentile`` of the candidates' qualities (ties: the
    higher quality, then the earliest observation). Ties in ``max`` / ``min``
    go to the earliest observation.
    """
    any_cand = cand.any(axis=0)
    if mode == "max":
        idx = np.argmax(np.where(cand, qual, -np.inf), axis=0)
    elif mode == "min":
        idx = np.argmin(np.where(cand, qual, np.inf), axis=0)
    else:
        target = compute_nanpercentiles(np.where(cand, qual, np.nan), [percentile])[0]
        idx = np.full(any_cand.shape, -1, dtype=np.int64)
        best_dist = np.full(any_cand.shape, np.inf)
        best_q = np.full(any_cand.shape, -np.inf)
        for t_idx in range(qual.shape[0]):
            with np.errstate(invalid="ignore"):
                dist = np.abs(qual[t_idx] - target)
                better = cand[t_idx] & (
                    (dist < best_dist) | ((dist == best_dist) & (qual[t_idx] > best_q))
                )
            idx[better] = t_idx
            best_dist[better] = dist[better]
            best_q[better] = qual[t_idx][better]
    idx = idx.astype(np.int64)
    idx[~any_cand] = -1
    return idx


def _medoid_index(data: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """
    0-based index (``-1``: none) of the valid observation with the smallest
    sum of Euclidean distances to the other valid observations (ties: the
    earliest). ``data`` is ``(n_obs, n_bands, ...)``, ``valid`` ``(n_obs, ...)``.
    """
    n_obs = data.shape[0]
    sums = np.zeros(valid.shape, dtype=np.float64)
    for i_idx in range(n_obs):
        for j_idx in range(i_idx + 1, n_obs):
            d2 = np.zeros(valid.shape[1:], dtype=np.float64)
            for b_idx in range(data.shape[1]):
                diff = data[i_idx, b_idx] - data[j_idx, b_idx]
                d2 = d2 + diff * diff
            dist = np.sqrt(d2)
            both = valid[i_idx] & valid[j_idx]
            sums[i_idx] = np.where(both, sums[i_idx] + dist, sums[i_idx])
            sums[j_idx] = np.where(both, sums[j_idx] + dist, sums[j_idx])
    sums[~valid] = np.inf
    idx = np.argmin(sums, axis=0).astype(np.int64)
    idx[~valid.any(axis=0)] = -1
    return idx


def _geomedian_loop(
    data: np.ndarray, valid: np.ndarray, max_iter: int, tol: float
) -> np.ndarray:  # pragma: no cover - compiled by numba
    """
    Weiszfeld iteration with the Vardi-Zhang modification for every pixel
    (numba kernel). ``data`` is ``(n_obs, n_bands, n_pix)`` float64,
    ``valid`` ``(n_obs, n_pix)``; returns ``(n_bands, n_pix)`` (NaN where no
    observation is valid).
    """
    # pylint: disable=too-many-locals,too-many-branches,too-many-nested-blocks
    # pylint: disable=too-many-statements
    n_obs, n_bands, n_pix = data.shape
    out = np.full((n_bands, n_pix), np.nan)
    y_cur = np.empty(n_bands)
    y_new = np.empty(n_bands)
    num = np.empty(n_bands)
    rvec = np.empty(n_bands)
    for pix in range(n_pix):
        n_valid = 0
        for t_idx in range(n_obs):
            if valid[t_idx, pix]:
                n_valid += 1
        if n_valid == 0:
            continue
        for b_idx in range(n_bands):
            y_cur[b_idx] = 0.0
        for t_idx in range(n_obs):
            if valid[t_idx, pix]:
                for b_idx in range(n_bands):
                    y_cur[b_idx] += data[t_idx, b_idx, pix]
        for b_idx in range(n_bands):
            y_cur[b_idx] = y_cur[b_idx] / n_valid
        if n_valid > 2:
            for _ in range(max_iter):
                for b_idx in range(n_bands):
                    num[b_idx] = 0.0
                    rvec[b_idx] = 0.0
                den = 0.0
                eta = 0
                for t_idx in range(n_obs):
                    if not valid[t_idx, pix]:
                        continue
                    d2 = 0.0
                    for b_idx in range(n_bands):
                        diff = data[t_idx, b_idx, pix] - y_cur[b_idx]
                        d2 += diff * diff
                    dist = np.sqrt(d2)
                    if dist == 0.0:
                        eta += 1
                        continue
                    wgt = 1.0 / dist
                    den += wgt
                    for b_idx in range(n_bands):
                        num[b_idx] += data[t_idx, b_idx, pix] * wgt
                        rvec[b_idx] += (data[t_idx, b_idx, pix] - y_cur[b_idx]) * wgt
                if den == 0.0:
                    break
                r2 = 0.0
                for b_idx in range(n_bands):
                    r2 += rvec[b_idx] * rvec[b_idx]
                rnorm = np.sqrt(r2)
                if eta == 0:
                    for b_idx in range(n_bands):
                        y_new[b_idx] = num[b_idx] / den
                elif rnorm == 0.0:
                    break
                else:
                    gam = min(eta / rnorm, 1.0)
                    for b_idx in range(n_bands):
                        y_new[b_idx] = (1.0 - gam) * (num[b_idx] / den) + gam * y_cur[b_idx]
                s2 = 0.0
                n2 = 0.0
                for b_idx in range(n_bands):
                    diff = y_new[b_idx] - y_cur[b_idx]
                    s2 += diff * diff
                    n2 += y_cur[b_idx] * y_cur[b_idx]
                for b_idx in range(n_bands):
                    y_cur[b_idx] = y_new[b_idx]
                if np.sqrt(s2) <= tol * max(1.0, np.sqrt(n2)):
                    break
        for b_idx in range(n_bands):
            out[b_idx, pix] = y_cur[b_idx]
    return out


def _geomedian_numpy(
    data: np.ndarray, valid: np.ndarray, max_iter: int, tol: float
) -> np.ndarray:
    """
    NumPy fallback of :func:`_geomedian_loop`, vectorised over the pixels
    with the same arithmetic (and order of operations), so the two agree.
    """
    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    n_obs, n_bands, n_pix = data.shape
    out = np.full((n_bands, n_pix), np.nan)
    n_valid = np.zeros(n_pix, dtype=np.int64)
    for t_idx in range(n_obs):
        n_valid += valid[t_idx]
    y_all = np.zeros((n_bands, n_pix))
    for t_idx in range(n_obs):
        for b_idx in range(n_bands):
            y_all[b_idx] = np.where(valid[t_idx], y_all[b_idx] + data[t_idx, b_idx], y_all[b_idx])
    has = n_valid > 0
    y_all[:, has] = y_all[:, has] / n_valid[has]
    out[:, has] = y_all[:, has]

    act = np.nonzero(n_valid > 2)[0]
    y_cur = y_all[:, act]
    x_act = data[:, :, act]
    v_act = valid[:, act]
    with np.errstate(divide="ignore", invalid="ignore"):
        for _ in range(int(max_iter)):
            if act.size == 0:
                break
            k = act.size
            num = np.zeros((n_bands, k))
            rvec = np.zeros((n_bands, k))
            den = np.zeros(k)
            eta = np.zeros(k, dtype=np.int64)
            for t_idx in range(n_obs):
                v_t = v_act[t_idx]
                d2 = np.zeros(k)
                for b_idx in range(n_bands):
                    diff = x_act[t_idx, b_idx] - y_cur[b_idx]
                    d2 = d2 + diff * diff
                dist = np.sqrt(d2)
                eta += v_t & (dist == 0.0)
                use = v_t & (dist > 0.0)
                wgt = np.where(use, 1.0 / dist, 0.0)
                den = np.where(use, den + wgt, den)
                for b_idx in range(n_bands):
                    x_tb = x_act[t_idx, b_idx]
                    num[b_idx] = np.where(use, num[b_idx] + x_tb * wgt, num[b_idx])
                    rvec[b_idx] = np.where(
                        use, rvec[b_idx] + (x_tb - y_cur[b_idx]) * wgt, rvec[b_idx]
                    )
            r2 = np.zeros(k)
            for b_idx in range(n_bands):
                r2 = r2 + rvec[b_idx] * rvec[b_idx]
            rnorm = np.sqrt(r2)
            stop = (den == 0.0) | ((eta > 0) & (rnorm == 0.0))
            gam = np.minimum(eta / rnorm, 1.0)
            y_new = np.empty_like(y_cur)
            for b_idx in range(n_bands):
                t_b = num[b_idx] / den
                y_new[b_idx] = np.where(
                    eta == 0, t_b, (1.0 - gam) * t_b + gam * y_cur[b_idx]
                )
            y_new[:, stop] = y_cur[:, stop]
            s2 = np.zeros(k)
            n2 = np.zeros(k)
            for b_idx in range(n_bands):
                diff = y_new[b_idx] - y_cur[b_idx]
                s2 = s2 + diff * diff
                n2 = n2 + y_cur[b_idx] * y_cur[b_idx]
            conv = np.sqrt(s2) <= tol * np.maximum(1.0, np.sqrt(n2))
            y_cur = y_new
            done = stop | conv
            if np.any(done):
                out[:, act[done]] = y_cur[:, done]
                keep = ~done
                act = act[keep]
                y_cur = y_cur[:, keep]
                x_act = x_act[:, :, keep]
                v_act = v_act[:, keep]
    if act.size:
        out[:, act] = y_cur
    return out


_geomedian_kernel = optional_njit(
    _geomedian_loop,
    nogil=True,
    fallback=_geomedian_numpy,
    fallback_desc="the vectorised NumPy fallback",
)


def _geomedian_pixels(
    data: np.ndarray, valid: np.ndarray, max_iter: int, tol: float
) -> np.ndarray:
    """
    Geometric median of the valid observations of each pixel: ``data`` is
    ``(n_obs, n_bands, n_pix)``, ``valid`` ``(n_obs, n_pix)``; returns
    ``(n_bands, n_pix)``. Uses the numba kernel when available, else the
    NumPy fallback (the two agree to ``rtol=1e-9``).
    """
    return _geomedian_kernel(
        np.ascontiguousarray(data, dtype=np.float64),
        np.ascontiguousarray(valid, dtype=np.bool_),
        int(max_iter),
        float(tol),
    )


def _nearest_obs(data: np.ndarray, valid: np.ndarray, centre: np.ndarray) -> np.ndarray:
    """0-based index (``-1``: none) of the valid observation nearest ``centre``."""
    d2 = np.zeros(valid.shape, dtype=np.float64)
    for b_idx in range(data.shape[1]):
        diff = data[:, b_idx] - centre[b_idx]
        d2 = d2 + diff * diff
    d2[~valid] = np.inf
    idx = np.argmin(d2, axis=0).astype(np.int64)
    idx[~valid.any(axis=0)] = -1
    return idx


def _geomedian_mads(data: np.ndarray, valid: np.ndarray, centre: np.ndarray) -> np.ndarray:
    """
    ``(3, ...)`` medians over the valid observations of the Euclidean,
    cosine (1 - cosine similarity) and Bray-Curtis distances to ``centre``.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        diff = data - centre[np.newaxis]
        euc = np.sqrt(np.sum(diff * diff, axis=1))
        dot = np.sum(data * centre[np.newaxis], axis=1)
        nrm = np.sqrt(np.sum(data * data, axis=1)) * np.sqrt(np.sum(centre * centre, axis=0))
        cos = np.where(nrm > 0.0, 1.0 - dot / nrm, np.nan)
        bc_den = np.sum(np.abs(data + centre[np.newaxis]), axis=1)
        bray = np.where(bc_den > 0.0, np.sum(np.abs(diff), axis=1) / bc_den, np.nan)
    out = []
    for dist in (euc, cos, bray):
        dist = np.where(valid, dist, np.nan)
        out.append(compute_nanpercentiles(dist, [50.0])[0])
    return np.stack(out, axis=0)


# ---------------------------------------------------------------------------
# Tile-graph nodes
# ---------------------------------------------------------------------------


class _PixelNode(TileNode):
    """
    Pixel-wise node over several inputs (the first ``n_obs`` are observations
    with the same bands) evaluating ``func(blocks, state)``; ``state_fn``
    creates the per-run (per-thread) state, e.g. expression parsers.
    """

    # pylint: disable=too-many-arguments

    def __init__(
        self,
        inputs: Sequence[TileNode],
        func: Callable[[List[np.ndarray], Any], np.ndarray],
        meta: NodeMeta,
        *,
        n_obs: int = 0,
        state_fn: Optional[Callable[[], Any]] = None,
        name: Optional[str] = None,
        params: str = "",
    ) -> None:
        super().__init__(inputs=tuple(inputs), name=name)
        self.func = func
        self.meta = meta
        self.n_obs = int(n_obs)
        self.state_fn = state_fn
        self.params = params

    def infer_meta(self, inputs: Sequence[NodeMeta]) -> NodeMeta:
        counts = [m.n_bands for m in inputs[: self.n_obs]]
        if len({c for c in counts if c is not None}) > 1:
            raise rsgispy.RSGISPyException(
                f"{self.label}: all observations must have the same number of "
                f"bands, got {counts}."
            )
        return self.meta

    def start_run(self, resources: Any, halo_y: int, halo_x: int) -> Any:
        del resources, halo_y, halo_x
        return self.state_fn() if self.state_fn is not None else None

    def evaluate(
        self,
        inputs: Sequence[TileTensor],
        halo_y: int,
        halo_x: int,
        ctx: TileContext,
        state: Any = None,
    ) -> TileTensor:
        aligned = [t.crop(halo_y, halo_x) for t in inputs]
        with np.errstate(all="ignore"):
            out = self.func([t.data for t in aligned], state)
        return aligned[0].with_data(out)

    def describe_params(self) -> str:
        return self.params


class CompositeNode(SelectBands):
    """
    The composite bands of a composite node builder (e.g.
    :func:`quality_mosaic_node`). The other outputs of the composite are
    nodes computed from the same tiles, available as attributes (``None``
    when the composite does not have that output):

    * :attr:`ref` — the 1-based index of the observation used (``uint16``,
      ``0`` = none);
    * :attr:`quality` — the quality of the selected observation (``float32``);
    * :attr:`count` — the number of valid observations (``uint16``);
    * :attr:`mad` — the geomedian's median absolute deviations (``float32``,
      bands ``EMAD``, ``SMAD`` and ``BCMAD``);
    * :attr:`lw_msk` — the land (1) / water (2) mask used (``uint8``).

    :param core: Node with the composite bands first, then the other outputs.
    :param n_out: Number of composite bands.
    :param outputs: Mapping of output name to node.
    :param dtype: NumPy dtype of the composite blocks (Default = ``None``:
                  ``float64``).
    :param band_names: Optional names of the composite bands.
    :param name: Optional node identifier.

    .. code:: python

        import glob
        from rsgispy.imageprocess import DiskSink, RasterSource, TileGraph
        from rsgispy.imageutils.composite import quality_mosaic_node

        srcs = [RasterSource(img) for img in sorted(glob.glob("ls_*.tif"))]
        comp = quality_mosaic_node(srcs, "(b5 - b4) / (b5 + b4)")
        TileGraph([DiskSink(comp, "max_ndvi.tif"), DiskSink(comp.ref, "ref.tif")]).run()
    """

    # pylint: disable=too-many-arguments

    def __init__(
        self,
        core: TileNode,
        n_out: int,
        outputs: Mapping[str, TileNode],
        *,
        dtype: Any = None,
        band_names: Optional[Sequence[str]] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(core, list(range(1, int(n_out) + 1)), name=name, band_names=band_names)
        self.out_dtype = np.dtype(dtype) if dtype is not None else None
        self.outputs: Dict[str, TileNode] = dict(outputs)

    def infer_meta(self, inputs: Sequence[NodeMeta]) -> NodeMeta:
        """The composite bands' metadata (with ``dtype`` when given)."""
        meta = super().infer_meta(inputs)
        if self.out_dtype is not None:
            meta = meta.replace(dtype=self.out_dtype)
        return meta

    def evaluate(
        self,
        inputs: Sequence[TileTensor],
        halo_y: int,
        halo_x: int,
        ctx: TileContext,
        state: Any = None,
    ) -> TileTensor:
        """The composite bands of the tile (cast to ``dtype`` when given)."""
        out = super().evaluate(inputs, halo_y, halo_x, ctx, state)
        if self.out_dtype is None or self.out_dtype == np.float64:
            return out
        arr = out.data
        return out.with_data(
            cast_result_block(
                arr, int(arr.shape[1]), int(arr.shape[2]), self.out_dtype, clamp_int=True
            )
        )

    @property
    def ref(self) -> Optional[TileNode]:
        """Reference image node (index of the observation used, ``uint16``)."""
        return self.outputs.get("ref")

    @property
    def quality(self) -> Optional[TileNode]:
        """Quality of the selected observation (``float32``)."""
        return self.outputs.get("quality")

    @property
    def count(self) -> Optional[TileNode]:
        """Number of valid observations (``uint16``)."""
        return self.outputs.get("count")

    @property
    def mad(self) -> Optional[TileNode]:
        """Geomedian median absolute deviations (``float32``, 3 bands)."""
        return self.outputs.get("mad")

    @property
    def lw_msk(self) -> Optional[TileNode]:
        """Land (1) / water (2) mask used (``uint8``)."""
        return self.outputs.get("lw_msk")


def _resolve_nodata(in_nd: Optional[float], np_dtype: Optional[np.dtype]) -> float:
    """
    Output no-data value: ``in_nd`` when it can be stored in ``np_dtype``
    (``float64`` when ``None``), otherwise NaN for float types and 0 for
    integer types.
    """
    dt = np.dtype(np.float64 if np_dtype is None else np_dtype)
    is_int = np.issubdtype(dt, np.integer)
    if in_nd is not None:
        val = float(in_nd)
        if math.isnan(val):
            if not is_int:
                return val
        elif fits_dtype_range(val, dt) and (not is_int or val.is_integer()):
            return val
    return 0.0 if is_int else float("nan")


def _is_int_dtype(np_dtype: Optional[np.dtype]) -> bool:
    return np_dtype is not None and np.issubdtype(np.dtype(np_dtype), np.integer)


def _aux_node(
    core: TileNode,
    bands: Union[int, Sequence[int]],
    names: Sequence[str],
    dtype: Any,
    nodata: float,
    name: str,
) -> TileNode:
    """Output of a composite core (selected bands) with its dtype and no data."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    sel = SelectBands(core, bands, name=f"{name}_select", band_names=list(names))
    return Cast(SetNoData(sel, nodata, name=f"{name}_nodata"), np.dtype(dtype), name=name)


def _sel_meta(n_obs: int, n_extra: int = 0) -> NodeMeta:
    """Metadata of a selection tensor (see :func:`_assemble_selection`)."""
    return NodeMeta(n_bands=2 + 2 * n_obs + n_extra, dtype=np.float64, nodata=None)


def _assemble_selection(
    idx: np.ndarray,
    cls: np.ndarray,
    cand: np.ndarray,
    qual: np.ndarray,
    override: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Selection tensor: band 0 the 1-based reference (0 = none), band 1 the
    class (land/water, or 1 where the pixel has a candidate), then one
    candidate flag and one quality per observation, then optional override
    values (e.g. geomedian bands; NaN where not used).
    """
    parts = [
        (idx + 1).astype(np.float64)[np.newaxis],
        np.asarray(cls, dtype=np.float64)[np.newaxis],
        cand.astype(np.float64),
        np.asarray(qual, dtype=np.float64),
    ]
    if override is not None:
        parts.append(override)
    return np.concatenate(parts, axis=0)


def _smooth_selection(
    sel: TileNode,
    n_obs: int,
    size: int,
    classes: Sequence[int],
    name: str,
) -> TileNode:
    """
    Majority (mode) filter of the reference of a selection tensor within
    each class in ``classes``: a pixel takes the most frequent observation
    of its ``size x size`` window among the pixels of its own class, unless
    that observation is not a candidate at the pixel.
    """
    # pylint: disable=import-outside-toplevel
    from rsgispy.classification.classpostprocess import majority_filter_node
    from rsgispy.imagefilter._helpers import validate_filter_size

    size = validate_filter_size(size)
    majs: List[TileNode] = []
    for cls_val in classes:

        def _mask(blocks: List[np.ndarray], state: Any, cls_val: int = cls_val) -> np.ndarray:
            del state
            ref = blocks[0][0]
            keep = (blocks[0][1] == cls_val) & (ref > 0)
            return np.where(keep, ref, 0.0)[np.newaxis]

        masked = _PixelNode(
            [sel], _mask, NodeMeta(n_bands=1, dtype=np.float64, nodata=0.0),
            name=f"{name}_ref_class{cls_val}",
        )
        majs.append(
            majority_filter_node(
                masked, size, keep_centre_on_tie=True, no_data_val=0.0,
                name=f"{name}_majority_class{cls_val}",
            )
        )
    cls_list = [int(c) for c in classes]

    def _apply(blocks: List[np.ndarray], state: Any) -> np.ndarray:
        del state
        out = np.array(blocks[0], dtype=np.float64, copy=True)
        ref = out[0]
        cand = out[2: 2 + n_obs]
        new_ref = ref.copy()
        for cls_val, maj in zip(cls_list, blocks[1:]):
            m_ref = maj[0]
            chg = (out[1] == cls_val) & (ref > 0) & np.isfinite(m_ref)
            chg &= (m_ref > 0) & (m_ref != ref)
            if np.any(chg):
                m_idx = np.clip(np.where(chg, m_ref, 1.0).astype(np.int64) - 1, 0, n_obs - 1)
                ok = np.take_along_axis(cand, m_idx[np.newaxis], axis=0)[0] > 0
                upd = chg & ok
                new_ref[upd] = m_ref[upd]
        out[0] = new_ref
        return out

    return _PixelNode(
        [sel, *majs], _apply, infer_node_meta(sel),
        name=f"{name}_smoothed", params=f"size={size} classes={cls_list}",
    )


def _gather_composite(
    sources: Sequence[TileNode],
    sel: TileNode,
    info: _ObsInfo,
    dtype: Any,
    name: str,
    outputs: Sequence[str],
    has_override: bool = False,
) -> CompositeNode:
    """
    Copy the bands of the observation selected at each pixel (or the
    override values) and return the :class:`CompositeNode` with the
    ``outputs`` (``ref``, ``quality``, ``lw_msk``).
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    n_obs = info.n_obs
    n_out = info.n_out
    out_idx = list(info.out_idx)
    np_dtype = np.dtype(dtype) if dtype is not None else None
    out_nd = _resolve_nodata(info.nodata, np_dtype)
    round_int = _is_int_dtype(np_dtype)

    def _gather(blocks: List[np.ndarray], state: Any) -> np.ndarray:
        del state
        sel_blk = blocks[n_obs]
        ref = sel_blk[0]
        comp = np.full((n_out,) + ref.shape, out_nd, dtype=np.float64)
        for t_idx in range(n_obs):
            msk = ref == t_idx + 1
            if np.any(msk):
                comp[:, msk] = np.asarray(blocks[t_idx], dtype=np.float64)[out_idx][:, msk]
        if has_override:
            ovr = sel_blk[2 + 2 * n_obs: 2 + 2 * n_obs + n_out]
            msk = (ref > 0) & np.isfinite(ovr[0])
            vals = ovr[:, msk]
            comp[:, msk] = np.rint(vals) if round_int else vals
        qual = sel_blk[2 + n_obs: 2 + 2 * n_obs]
        r_idx = np.clip(ref.astype(np.int64) - 1, 0, n_obs - 1)
        q_sel = np.take_along_axis(qual, r_idx[np.newaxis], axis=0)[0]
        q_sel[ref <= 0] = np.nan
        return np.concatenate(
            [comp, ref[np.newaxis], q_sel[np.newaxis], sel_blk[1][np.newaxis]], axis=0
        )

    names = info.out_names()
    core_names = None if names is None else names + [REF_BAND_NAME, QUALITY_BAND_NAME, LW_BAND_NAME]
    core = _PixelNode(
        [*sources, sel], _gather,
        NodeMeta(n_bands=n_out + 3, band_names=core_names, dtype=np.float64, nodata=out_nd),
        n_obs=n_obs, name=f"{name}_gather",
    )
    aux: Dict[str, TileNode] = {}
    if "ref" in outputs:
        aux["ref"] = _aux_node(core, n_out + 1, [REF_BAND_NAME], np.uint16, 0.0, f"{name}_ref")
    if "quality" in outputs:
        aux["quality"] = _aux_node(
            core, n_out + 2, [QUALITY_BAND_NAME], np.float32, float("nan"), f"{name}_quality"
        )
    if "lw_msk" in outputs:
        aux["lw_msk"] = _aux_node(core, n_out + 3, [LW_BAND_NAME], np.uint8, 0.0, f"{name}_lw_msk")
    return CompositeNode(core, n_out, aux, dtype=np_dtype, name=name)


# ---------------------------------------------------------------------------
# Node builders
# ---------------------------------------------------------------------------


def geomedian_node(
    sources: SourcesArg,
    max_iter: int = 1000,
    tol: float = 1e-7,
    *,
    bands: Optional[Union[int, Sequence[int]]] = None,
    no_data_val: Optional[float] = None,
    dtype: Any = None,
    name: Optional[str] = None,
) -> CompositeNode:
    """
    Tile-graph node of :func:`create_geomedian_composite`: the per-pixel
    multi-band geometric median of the valid observations (one
    :class:`~rsgispy.imageprocess.TileNode` per date, each with the same
    bands). The composite has the band names of the observations; its
    :attr:`~CompositeNode.ref` (observation nearest the geomedian),
    :attr:`~CompositeNode.count` and :attr:`~CompositeNode.mad` outputs are
    attributes of the returned node.

    :param sources: One node per observation.
    :param max_iter: Maximum number of Weiszfeld iterations (Default = ``1000``).
    :param tol: Relative step tolerance (Default = ``1e-7``).
    :param bands: Optional 1-based bands of the observations to composite
                  (Default = ``None``: all).
    :param no_data_val: No-data value of the observations (Default = ``None``:
                        that of the first observation's metadata).
    :param dtype: Optional NumPy dtype of the composite blocks; values are
                  rounded for integer types (Default = ``None``: ``float64``).
    :param name: Optional node name (Default = ``'geomedian'``).
    :return: :class:`CompositeNode`.

    .. code:: python

        import glob
        from rsgispy.imageprocess import DiskSink, RasterSource, TileGraph
        from rsgispy.imageutils.composite import geomedian_node

        srcs = [RasterSource(img) for img in sorted(glob.glob("s2_*.tif"))]
        gm = geomedian_node(srcs)
        TileGraph([DiskSink(gm, "geomedian.tif"), DiskSink(gm.mad, "mads.tif")]).run()
    """
    # pylint: disable=too-many-arguments,too-many-locals
    srcs = _source_list(sources)
    info = _resolve_obs(srcs, bands, no_data_val)
    if int(max_iter) < 1:
        raise rsgispy.RSGISPyException(f"max_iter must be >= 1, got {max_iter}.")
    if not math.isfinite(float(tol)) or float(tol) <= 0.0:
        raise rsgispy.RSGISPyException(f"tol must be > 0, got {tol}.")
    name = name or "geomedian"
    np_dtype = np.dtype(dtype) if dtype is not None else None
    out_nd = _resolve_nodata(info.nodata, np_dtype)
    round_int = _is_int_dtype(np_dtype)
    n_out = info.n_out
    out_idx = list(info.out_idx)
    m_iter = int(max_iter)
    f_tol = float(tol)

    def _func(blocks: List[np.ndarray], state: Any) -> np.ndarray:
        del state
        nan_blks, vbs = _prep_obs(blocks, info.nodata)
        valid = _obs_valid(vbs, out_idx)
        shape = valid.shape[1:]
        n_pix = int(np.prod(shape))
        data = np.stack([blk[out_idx] for blk in nan_blks], axis=0).reshape(
            info.n_obs, n_out, n_pix
        )
        vflat = valid.reshape(info.n_obs, n_pix)
        geo = _geomedian_pixels(data, vflat, m_iter, f_tol)
        idx = _nearest_obs(data, vflat, geo)
        mads = _geomedian_mads(data, vflat, geo)
        count = vflat.sum(axis=0).astype(np.float64)
        empty = count == 0
        if round_int:
            geo = np.rint(geo)
        geo[:, empty] = out_nd
        out = np.concatenate(
            [geo, (idx + 1).astype(np.float64)[np.newaxis], count[np.newaxis], mads], axis=0
        )
        return out.reshape((n_out + 5,) + shape)

    names = info.out_names()
    core_names = (
        None if names is None else names + [REF_BAND_NAME, COUNT_BAND_NAME, *MAD_BAND_NAMES]
    )
    core = _PixelNode(
        srcs, _func,
        NodeMeta(n_bands=n_out + 5, band_names=core_names, dtype=np.float64, nodata=out_nd),
        n_obs=info.n_obs, name=f"{name}_core", params=f"max_iter={m_iter} tol={f_tol:g}",
    )
    aux = {
        "ref": _aux_node(core, n_out + 1, [REF_BAND_NAME], np.uint16, 0.0, f"{name}_ref"),
        "count": _aux_node(core, n_out + 2, [COUNT_BAND_NAME], np.uint16, 0.0, f"{name}_count"),
        "mad": _aux_node(
            core, [n_out + 3, n_out + 4, n_out + 5], MAD_BAND_NAMES, np.float32,
            float("nan"), f"{name}_mad",
        ),
    }
    return CompositeNode(core, n_out, aux, dtype=np_dtype, name=name)


def quality_mosaic_node(
    sources: SourcesArg,
    quality: QualitySpec,
    mode: str = "max",
    percentile: Optional[float] = None,
    *,
    smooth_ref_size: Optional[int] = None,
    bands: Optional[Union[int, Sequence[int]]] = None,
    no_data_val: Optional[float] = None,
    dtype: Any = None,
    name: Optional[str] = None,
) -> CompositeNode:
    """
    Tile-graph node of :func:`create_quality_mosaic`: each pixel takes the
    whole valid observation (one :class:`~rsgispy.imageprocess.TileNode`
    per date, each with the same bands) whose quality is largest
    (``mode='max'``), smallest (``'min'``) or nearest a percentile
    (``'percentile'``) of the valid qualities. The
    :attr:`~CompositeNode.ref` and :attr:`~CompositeNode.quality` outputs
    are attributes of the returned node. See :func:`create_quality_mosaic`
    for the parameters.

    :param sources: One node per observation.
    :param quality: Band number, band-maths expression over ``b1`` … ``bn``
                    or callable (see :func:`create_quality_mosaic`).
    :param mode: ``'max'`` (Default), ``'min'`` or ``'percentile'``.
    :param percentile: Percentile (0–100) for ``mode='percentile'``.
    :param smooth_ref_size: Optional odd window size of the majority filter
                            applied to the observation choice.
    :param bands: Optional 1-based bands to composite (Default = all).
    :param no_data_val: No-data value of the observations (Default = ``None``:
                        that of the first observation's metadata).
    :param dtype: Optional NumPy dtype of the composite blocks (Default =
                  ``None``: ``float64``).
    :param name: Optional node name (Default = ``'quality_mosaic'``).
    :return: :class:`CompositeNode`.

    .. code:: python

        import glob
        from rsgispy.imageprocess import DiskSink, RasterSource, TileGraph
        from rsgispy.imageutils.composite import quality_mosaic_node

        srcs = [RasterSource(img) for img in sorted(glob.glob("ls_*.tif"))]
        comp = quality_mosaic_node(srcs, "(b5 - b4) / (b5 + b4)", "percentile", 90)
        TileGraph([DiskSink(comp, "ndvi_p90.tif"), DiskSink(comp.ref, "ref.tif")]).run()
    """
    # pylint: disable=too-many-arguments,too-many-locals
    srcs = _source_list(sources)
    info = _resolve_obs(srcs, bands, no_data_val)
    mode_l, pct = _check_mode(mode, percentile)
    qual = _Quality(quality, info.n_bands, "quality")
    name = name or "quality_mosaic"
    n_obs = info.n_obs
    check_idx = _check_bands(info.out_idx, qual.used)

    def _select(blocks: List[np.ndarray], state: Any) -> np.ndarray:
        nan_blks, vbs = _prep_obs(blocks, info.nodata)
        valid = _obs_valid(vbs, check_idx)
        q_vals = np.stack([qual.evaluate(blk, state) for blk in nan_blks], axis=0)
        cand = valid & np.isfinite(q_vals)
        idx = _select_by_quality(q_vals, cand, mode_l, pct)
        return _assemble_selection(idx, cand.any(axis=0), cand, q_vals)

    sel: TileNode = _PixelNode(
        srcs, _select, _sel_meta(n_obs), n_obs=n_obs, state_fn=qual.new_state,
        name=f"{name}_select", params=f"quality={qual.describe()} mode={mode_l}"
        + (f" percentile={pct:g}" if pct is not None else ""),
    )
    if smooth_ref_size is not None:
        sel = _smooth_selection(sel, n_obs, smooth_ref_size, (1,), name)
    return _gather_composite(srcs, sel, info, dtype, name, ("ref", "quality"))


def medoid_node(
    sources: SourcesArg,
    *,
    bands: Optional[Union[int, Sequence[int]]] = None,
    no_data_val: Optional[float] = None,
    dtype: Any = None,
    name: Optional[str] = None,
) -> CompositeNode:
    """
    Tile-graph node of :func:`create_medoid_composite`: each pixel takes the
    valid observation with the smallest sum of Euclidean distances (over
    ``bands``) to the other valid observations (ties: the earliest). The
    :attr:`~CompositeNode.ref` output is an attribute of the returned node.

    :param sources: One node per observation, each with the same bands.
    :param bands: Optional 1-based bands to composite and measure distances
                  over (Default = all).
    :param no_data_val: No-data value of the observations (Default = ``None``:
                        that of the first observation's metadata).
    :param dtype: Optional NumPy dtype of the composite blocks (Default =
                  ``None``: ``float64``).
    :param name: Optional node name (Default = ``'medoid'``).
    :return: :class:`CompositeNode`.

    .. code:: python

        import glob
        from rsgispy.imageprocess import DiskSink, RasterSource, TileGraph
        from rsgispy.imageutils.composite import medoid_node

        srcs = [RasterSource(img) for img in sorted(glob.glob("ls_*.tif"))]
        TileGraph(DiskSink(medoid_node(srcs), "medoid.tif")).run()
    """
    srcs = _source_list(sources)
    info = _resolve_obs(srcs, bands, no_data_val)
    name = name or "medoid"
    n_obs = info.n_obs
    out_idx = list(info.out_idx)

    def _select(blocks: List[np.ndarray], state: Any) -> np.ndarray:
        del state
        nan_blks, vbs = _prep_obs(blocks, info.nodata)
        valid = _obs_valid(vbs, out_idx)
        data = np.stack([blk[out_idx] for blk in nan_blks], axis=0)
        idx = _medoid_index(data, valid)
        return _assemble_selection(
            idx, valid.any(axis=0), valid, np.full(valid.shape, np.nan)
        )

    sel = _PixelNode(srcs, _select, _sel_meta(n_obs), n_obs=n_obs, name=f"{name}_select")
    return _gather_composite(srcs, sel, info, dtype, name, ("ref",))


def land_water_composite_node(
    sources: SourcesArg,
    land_quality: QualitySpec,
    water_quality: Optional[QualitySpec],
    *,
    land_mode: str = "max",
    water_mode: str = "max",
    land_percentile: Optional[float] = None,
    water_percentile: Optional[float] = None,
    smooth_lw_msk_size: Optional[int] = None,
    smooth_ref_size: Optional[int] = None,
    lw_msk: Optional[TileNode] = None,
    water_rule: Optional[QualitySpec] = None,
    water_rule_mode: str = "max",
    min_water_frac: float = 0.5,
    water_method: str = "quality",
    max_iter: int = 1000,
    tol: float = 1e-7,
    bands: Optional[Union[int, Sequence[int]]] = None,
    no_data_val: Optional[float] = None,
    dtype: Any = None,
    name: Optional[str] = None,
) -> CompositeNode:
    """
    Tile-graph node of :func:`create_land_water_composite`: a quality mosaic
    using ``land_quality`` over land and ``water_quality`` (or the medoid or
    geomedian, ``water_method``) over water. The land / water decision comes
    from ``lw_msk`` (a node with land = 1, water = 2) or from the
    per-observation vote of ``water_rule``. The :attr:`~CompositeNode.ref`,
    :attr:`~CompositeNode.quality` and :attr:`~CompositeNode.lw_msk` outputs
    are attributes of the returned node. See
    :func:`create_land_water_composite` for the parameters (``lw_msk`` is
    the node of its ``lw_msk_img``).

    :param sources: One node per observation, each with the same bands.
    :param land_quality: Quality used over land.
    :param water_quality: Quality used over water (may be ``None`` unless
                          ``water_method='quality'``).
    :return: :class:`CompositeNode`.

    .. code:: python

        import glob
        from rsgispy.imageprocess import DiskSink, RasterSource, TileGraph
        from rsgispy.imageutils.composite import land_water_composite_node

        srcs = [RasterSource(img) for img in sorted(glob.glob("ls_*.tif"))]
        comp = land_water_composite_node(
            srcs, "(b5 - b4) / (b5 + b4)", "(b3 - b5) / (b3 + b5)",
            water_rule="b5 < 500 ? 1 : 0",
        )
        TileGraph([DiskSink(comp, "comp.tif"), DiskSink(comp.lw_msk, "lw.tif")]).run()
    """
    # pylint: disable=too-many-arguments,too-many-locals,too-many-statements
    srcs = _source_list(sources)
    info = _resolve_obs(srcs, bands, no_data_val)
    name = name or "land_water_composite"
    n_obs = info.n_obs
    n_out = info.n_out
    out_idx = list(info.out_idx)
    land_mode_l, land_pct = _check_mode(land_mode, land_percentile, "land_mode", "land_percentile")
    method = str(water_method).lower()
    if method not in ("quality", "medoid", "geomedian"):
        raise rsgispy.RSGISPyException(
            f"water_method must be 'quality', 'medoid' or 'geomedian', got {water_method!r}."
        )
    if method == "quality":
        water_mode_l, water_pct = _check_mode(
            water_mode, water_percentile, "water_mode", "water_percentile"
        )
        if water_quality is None:
            raise rsgispy.RSGISPyException(
                "water_quality must be given when water_method='quality'."
            )
    else:
        water_mode_l, water_pct = "max", None
    land_q = _Quality(land_quality, info.n_bands, "land_quality")
    water_q = (
        _Quality(water_quality, info.n_bands, "water_quality")
        if water_quality is not None else None
    )
    land_check = _check_bands(out_idx, land_q.used)
    water_check = (
        _check_bands(out_idx, water_q.used)
        if (water_q is not None and method == "quality") else list(out_idx)
    )
    m_iter = int(max_iter)
    f_tol = float(tol)

    # --- The land / water decision.
    lw_node: TileNode = _lw_decision_node(
        srcs, info, lw_msk, water_rule, water_rule_mode, min_water_frac, name
    )
    if smooth_lw_msk_size is not None:
        # pylint: disable-next=import-outside-toplevel
        from rsgispy.classification.classpostprocess import majority_filter_node

        lw_node = majority_filter_node(
            lw_node, smooth_lw_msk_size, keep_centre_on_tie=True, no_data_val=0.0,
            name=f"{name}_lw_majority",
        )

    # --- The selection.
    def _state() -> Tuple[Any, Any]:
        return (land_q.new_state(), water_q.new_state() if water_q is not None else None)

    def _select(blocks: List[np.ndarray], state: Any) -> np.ndarray:
        # pylint: disable=too-many-locals
        nan_blks, vbs = _prep_obs(blocks[:n_obs], info.nodata)
        lw_cls = np.asarray(blocks[n_obs][0], dtype=np.float64)
        is_land = lw_cls == LAND_CLASS
        is_water = lw_cls == WATER_CLASS
        land_valid = _obs_valid(vbs, land_check)
        q_land = np.stack([land_q.evaluate(blk, state[0]) for blk in nan_blks], axis=0)
        cand_land = land_valid & np.isfinite(q_land)
        idx_land = _select_by_quality(q_land, cand_land, land_mode_l, land_pct)

        water_valid = _obs_valid(vbs, water_check)
        if water_q is not None:
            q_water = np.stack([water_q.evaluate(blk, state[1]) for blk in nan_blks], axis=0)
        else:
            q_water = np.full(water_valid.shape, np.nan)
        override = None
        if method == "quality":
            cand_water = water_valid & np.isfinite(q_water)
            idx_water = _select_by_quality(q_water, cand_water, water_mode_l, water_pct)
        else:
            cand_water = water_valid
            idx_water = np.full(lw_cls.shape, -1, dtype=np.int64)
            if method == "geomedian":
                override = np.full((n_out,) + lw_cls.shape, np.nan)
            if np.any(is_water):
                data_w = np.stack([blk[out_idx][:, is_water] for blk in nan_blks], axis=0)
                valid_w = water_valid[:, is_water]
                if method == "medoid":
                    idx_water[is_water] = _medoid_index(data_w, valid_w)
                else:
                    geo = _geomedian_pixels(data_w, valid_w, m_iter, f_tol)
                    idx_water[is_water] = _nearest_obs(data_w, valid_w, geo)
                    override[:, is_water] = geo  # type: ignore[index]
        idx = np.where(is_land, idx_land, np.where(is_water, idx_water, -1))
        cand = np.where(is_land[np.newaxis], cand_land, cand_water & is_water[np.newaxis])
        qual = np.where(is_land[np.newaxis], q_land, q_water)
        return _assemble_selection(idx, lw_cls, cand, qual, override)

    n_extra = n_out if method == "geomedian" else 0
    sel: TileNode = _PixelNode(
        [*srcs, lw_node], _select, _sel_meta(n_obs, n_extra), n_obs=n_obs,
        state_fn=_state, name=f"{name}_select",
        params=f"land={land_q.describe()} water_method={method}",
    )
    if smooth_ref_size is not None:
        classes = (LAND_CLASS,) if method == "geomedian" else (LAND_CLASS, WATER_CLASS)
        sel = _smooth_selection(sel, n_obs, smooth_ref_size, classes, name)
    return _gather_composite(
        srcs, sel, info, dtype, name, ("ref", "quality", "lw_msk"),
        has_override=method == "geomedian",
    )


def _lw_decision_node(
    srcs: Sequence[TileNode],
    info: _ObsInfo,
    lw_msk: Optional[TileNode],
    water_rule: Optional[QualitySpec],
    water_rule_mode: str,
    min_water_frac: float,
    name: str,
) -> TileNode:
    """Node of the land (1) / water (2) decision (0 where no observation is valid)."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    n_obs = info.n_obs
    out_idx = list(info.out_idx)
    if lw_msk is None and water_rule is None:
        raise rsgispy.RSGISPyException(
            "A land / water mask (lw_msk_img) or a water_rule must be given."
        )
    rule = None
    rule_mode = str(water_rule_mode).lower()
    frac = float(min_water_frac)
    rule_check: List[int] = []
    if lw_msk is None:
        if rule_mode not in ("max", "min"):
            raise rsgispy.RSGISPyException(
                f"water_rule_mode must be 'max' or 'min', got {water_rule_mode!r}."
            )
        if not 0.0 < frac <= 1.0:
            raise rsgispy.RSGISPyException(
                f"min_water_frac must be in (0, 1], got {min_water_frac!r}."
            )
        rule = _Quality(water_rule, info.n_bands, "water_rule")  # type: ignore[arg-type]
        rule_check = _check_bands(out_idx, rule.used)

    def _func(blocks: List[np.ndarray], state: Any) -> np.ndarray:
        # pylint: disable=too-many-locals
        nan_blks, vbs = _prep_obs(blocks[:n_obs], info.nodata)
        any_valid = _obs_valid(vbs, out_idx).any(axis=0)
        if lw_msk is not None:
            msk = np.asarray(blocks[n_obs][0], dtype=np.float64)
            lw_cls = np.where(
                msk == LAND_CLASS, float(LAND_CLASS),
                np.where(msk == WATER_CLASS, float(WATER_CLASS), 0.0),
            )
        else:
            vote_valid = _obs_valid(vbs, rule_check)
            n_vote = np.zeros(any_valid.shape)
            n_water = np.zeros(any_valid.shape)
            for t_idx, blk in enumerate(nan_blks):
                r_val = rule.evaluate(blk, state)  # type: ignore[union-attr]
                vote = vote_valid[t_idx] & np.isfinite(r_val)
                water = vote & ((r_val > 0.0) if rule_mode == "max" else (r_val < 0.0))
                n_vote += vote
                n_water += water
            lw_cls = np.where(any_valid, float(LAND_CLASS), 0.0)
            with np.errstate(divide="ignore", invalid="ignore"):
                is_water = (n_vote > 0) & (n_water / n_vote >= frac)
            lw_cls[is_water] = float(WATER_CLASS)
        lw_cls[~any_valid] = 0.0
        return lw_cls[np.newaxis]

    inputs = [*srcs] + ([lw_msk] if lw_msk is not None else [])
    params = (
        "lw_msk" if lw_msk is not None
        else f"water_rule={rule.describe()} mode={rule_mode} min_water_frac={frac:g}"  # type: ignore[union-attr]
    )
    return _PixelNode(
        inputs, _func,
        NodeMeta(n_bands=1, band_names=(LW_BAND_NAME,), dtype=np.float64, nodata=0.0),
        n_obs=n_obs, state_fn=rule.new_state if rule is not None else None,
        name=f"{name}_lw_decision", params=params,
    )


def percentile_composite_node(
    sources: SourcesArg,
    percentiles: Union[float, Sequence[float]] = (50,),
    *,
    bands: Optional[Union[int, Sequence[int]]] = None,
    no_data_val: Optional[float] = None,
    valid_rule: str = "all_bands",
    dtype: Any = None,
    name: Optional[str] = None,
) -> CompositeNode:
    """
    Tile-graph node of :func:`create_percentile_composite`: per-band
    percentiles over the observations (one
    :class:`~rsgispy.imageprocess.TileNode` per date, each with the same
    bands), grouped by band and named ``{band name}_p{percentile}``. The
    :attr:`~CompositeNode.count` output (valid observations; one band per
    composited band with ``valid_rule='per_band'``) is an attribute of the
    returned node.

    Unlike :func:`~rsgispy.imagecalc.percentiles.calc_imgs_pxl_percentiles_node`,
    which pools the bands of all the inputs into one set of values, each
    band is summarised separately.

    :param sources: One node per observation.
    :param percentiles: Percentile or percentiles in ``[0, 100]``
                        (Default = ``(50,)``, the median).
    :param bands: Optional 1-based bands (Default = all).
    :param no_data_val: No-data value of the observations (Default = ``None``:
                        that of the first observation's metadata).
    :param valid_rule: ``'all_bands'`` (Default) or ``'per_band'``.
    :param dtype: Optional NumPy dtype of the blocks; values are rounded for
                  integer types (Default = ``None``: ``float64``).
    :param name: Optional node name (Default = ``'percentile_composite'``).
    :return: :class:`CompositeNode`.

    .. code:: python

        import glob
        from rsgispy.imageprocess import DiskSink, RasterSource, TileGraph
        from rsgispy.imageutils.composite import percentile_composite_node

        srcs = [RasterSource(img) for img in sorted(glob.glob("s2_*.tif"))]
        pcts = percentile_composite_node(srcs, [10, 50, 90])
        TileGraph(DiskSink(pcts, "s2_pcts.tif")).run()
    """
    # pylint: disable=too-many-arguments,too-many-locals
    srcs = _source_list(sources)
    info = _resolve_obs(srcs, bands, no_data_val)
    validate_percentiles_0_100(percentiles)
    pcts = [float(p) for p in np.atleast_1d(np.asarray(percentiles, dtype=np.float64))]
    rule = str(valid_rule).lower()
    if rule not in ("all_bands", "per_band"):
        raise rsgispy.RSGISPyException(
            f"valid_rule must be 'all_bands' or 'per_band', got {valid_rule!r}."
        )
    per_band = rule == "per_band"
    name = name or "percentile_composite"
    np_dtype = np.dtype(dtype) if dtype is not None else None
    out_nd = _resolve_nodata(info.nodata, np_dtype)
    round_int = _is_int_dtype(np_dtype)
    out_idx = list(info.out_idx)
    n_p = len(pcts)
    n_out = info.n_out * n_p
    n_count = info.n_out if per_band else 1

    def _func(blocks: List[np.ndarray], state: Any) -> np.ndarray:
        del state
        nan_blks, vbs = _prep_obs(blocks, info.nodata)
        valid = _obs_valid(vbs, out_idx)
        shape = valid.shape[1:]
        res = np.empty((n_out,) + shape)
        counts = []
        for o_idx, b_idx in enumerate(out_idx):
            stack = np.stack([blk[b_idx] for blk in nan_blks], axis=0)
            if not per_band:
                stack = np.where(valid, stack, np.nan)
            vals = compute_nanpercentiles(stack, pcts)
            n_vld = np.count_nonzero(np.isfinite(stack), axis=0)
            if round_int:
                vals = np.rint(vals)
            vals[:, n_vld == 0] = out_nd
            res[o_idx * n_p: (o_idx + 1) * n_p] = vals
            if per_band:
                counts.append(n_vld.astype(np.float64))
        if not per_band:
            counts.append(valid.sum(axis=0).astype(np.float64))
        return np.concatenate([res, np.stack(counts, axis=0)], axis=0)

    from rsgispy.imagecalc.percentiles import _pct_label  # pylint: disable=import-outside-toplevel

    base = info.base_names()
    pct_names = [f"{nm}_{_pct_label(p)}" for nm in base for p in pcts]
    count_names = [f"{nm}_count" for nm in base] if per_band else [COUNT_BAND_NAME]
    core = _PixelNode(
        srcs, _func,
        NodeMeta(
            n_bands=n_out + n_count, band_names=pct_names + count_names,
            dtype=np.float64, nodata=out_nd,
        ),
        n_obs=info.n_obs, name=f"{name}_core",
        params=f"percentiles={pcts} valid_rule={rule}",
    )
    aux = {
        "count": _aux_node(
            core, list(range(n_out + 1, n_out + n_count + 1)), count_names, np.uint16, 0.0,
            f"{name}_count",
        )
    }
    return CompositeNode(core, n_out, aux, dtype=np_dtype, name=name)


# ---------------------------------------------------------------------------
# File-based functions
# ---------------------------------------------------------------------------


class _Inputs:
    """The input images of a file-based composite."""

    # pylint: disable=too-few-public-methods

    def __init__(self, input_imgs: Sequence[Union[str, os.PathLike]]) -> None:
        if isinstance(input_imgs, (str, os.PathLike)):
            raise rsgispy.RSGISPyException(
                "input_imgs must be a sequence of image paths, not a single path."
            )
        self.paths = [os.fspath(p) for p in input_imgs]
        if not self.paths:
            raise rsgispy.RSGISPyException("input_imgs must contain at least one image.")
        counts = []
        for path in self.paths:
            ensure_file_exists(path)
            with rasterio.open(path) as dset:
                counts.append(int(dset.count))
                if len(counts) == 1:
                    self.dtype = str(dset.dtypes[0])
                    self.nodata = dset.nodata
        for path, count in zip(self.paths[1:], counts[1:]):
            if count != counts[0]:
                raise rsgispy.RSGISPyException(
                    "All input images must have the same bands: "
                    f"'{self.paths[0]}' has {counts[0]} band(s) but '{path}' has {count}."
                )
        self.sources = [
            RasterSource(path, dtype=np.float64, name=f"obs{idx + 1}")
            for idx, path in enumerate(self.paths)
        ]

    def datatype(self, datatype: Optional[int]) -> Tuple[int, np.dtype]:
        """``(rsgispy.TYPE_*, numpy dtype)`` of the output (default: the input's)."""
        dt = rsgispy.get_rsgis_datatype(self.dtype) if datatype is None else int(datatype)
        return dt, np.dtype(rsgispy.get_numpy_datatype(dt))


def _ref_colour_table(n_imgs: int) -> Dict[int, Tuple[int, int, int, int]]:
    """Deterministic, well-separated colours for observations 1..``n_imgs``."""
    clrs: Dict[int, Tuple[int, int, int, int]] = {0: (0, 0, 0, 0)}
    golden = 0.618033988749895
    for idx in range(1, n_imgs + 1):
        hue = (idx * golden) % 1.0
        val = 0.95 if idx % 2 else 0.75
        red, green, blue = colorsys.hsv_to_rgb(hue, 0.65, val)
        clrs[idx] = (int(round(red * 255)), int(round(green * 255)), int(round(blue * 255)), 255)
    return clrs


def _continuous_sink(
    node: TileNode,
    output_img: str,
    gdalformat: str,
    datatype: int,
    nodata: float,
    band_names: Optional[Sequence[str]],
    calc_stats: bool,
    creation_opts: Optional[Mapping[str, Any]],
) -> WrapperDiskSink:
    """Sink of a continuous output with its no-data value and band names."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    return WrapperDiskSink(
        node,
        os.fspath(output_img),
        gdalformat=gdalformat,
        datatype=datatype,
        nodata=nodata,
        band_names=None if band_names is None else list(band_names),
        clamp_int=True,
        calc_stats=calc_stats,
        creation_opts=normalise_creation_options(creation_opts),
    )


def _ref_class_names(paths: Sequence[str]) -> Dict[int, str]:
    """Class names of the reference image: the input images' file names."""
    return {idx + 1: os.path.basename(os.fspath(path)) for idx, path in enumerate(paths)}


def _thematic_sink(
    node: TileNode,
    output_img: str,
    gdalformat: str,
    datatype: int,
    band_name: str,
    clr_lut: Mapping[int, Any],
    calc_stats: bool,
    creation_opts: Optional[Mapping[str, Any]],
    class_names: Optional[Mapping[int, str]] = None,
) -> WrapperDiskSink:
    """
    Sink of a thematic output (no data 0): marked thematic, with thematic
    statistics (when ``calc_stats``), the colour table ``clr_lut`` and the
    class names ``class_names`` (read back with
    :func:`rsgispy.imageutils.colourtable.get_thematic_classes`).
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    is_cog = str(gdalformat).upper() == "COG"

    def _finish(path: str) -> None:
        set_img_thematic(path)
        if calc_stats:
            pop_thmt_img_stats(
                path, add_clr_tab=True, calc_pyramids=not is_cog, ignore_zero=True,
                use_pbar=False,
            )
        define_colour_table(path, clr_lut)
        if class_names:
            with rasterio.open(path, "r+") as dst:
                dst.update_tags(1, **{CLASS_NAMES_TAG: json.dumps(
                    {str(int(k)): str(v) for k, v in class_names.items()})})

    return WrapperDiskSink(
        node,
        os.fspath(output_img),
        gdalformat=gdalformat,
        datatype=datatype,
        nodata=0.0,
        band_names=[band_name],
        clamp_int=True,
        calc_stats=False,
        thematic=True,
        close_actions=[_finish],
        creation_opts=normalise_creation_options(creation_opts),
    )


def _run(
    sinks: Sequence[WrapperDiskSink],
    block_size: Optional[int],
    use_pbar: bool,
    n_jobs: Optional[int],
) -> None:
    TileGraph(list(sinks), block_size=block_size).run(use_pbar=use_pbar, n_jobs=n_jobs)


def create_geomedian_composite(
    input_imgs: Sequence[Union[str, os.PathLike]],
    output_img: Union[str, os.PathLike],
    *,
    bands: Optional[Union[int, Sequence[int]]] = None,
    no_data_val: Optional[float] = None,
    max_iter: int = 1000,
    tol: float = 1e-7,
    out_mad_img: Optional[Union[str, os.PathLike]] = None,
    out_count_img: Optional[Union[str, os.PathLike]] = None,
    out_ref_img: Optional[Union[str, os.PathLike]] = None,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    block_size: Optional[int] = None,
    use_pbar: bool = True,
    calc_stats: bool = True,
    creation_opts: Optional[Mapping[str, Any]] = None,
    n_jobs: Optional[int] = 1,
) -> None:
    """
    Create a geometric median (geomedian) composite: for each pixel, the
    multi-band point that minimises the sum of Euclidean distances to the
    valid observations (Roberts, Mueller and McIntyre, 2017). Unlike a
    per-band median, the bands stay consistent with each other, and it is
    robust to outliers such as residual cloud and shadow.

    Roberts, D., Mueller, N. and McIntyre, A. (2017). High-dimensional pixel
    composites from Earth observation time series. IEEE Transactions on
    Geoscience and Remote Sensing, 55(11), 6254–6264.

    * **Algorithm**: Weiszfeld iteration with the Vardi–Zhang modification
      (so an iterate on an observation does not divide by zero), started
      from the per-band mean of the valid observations and stopped when the
      step is ``<= tol * max(1, ||x||)`` or after ``max_iter`` iterations. It
      runs in float64 (numba kernel, or a NumPy fallback giving the same
      result) and the result is written in ``datatype`` (rounded for integer
      types).
    * **Small cases**: one valid observation gives that observation; two
      give their mean (every point on the segment between them is a
      geomedian, and the mean is the one chosen).
    * **Valid observations**: an observation is valid at a pixel only when
      every band in ``bands`` is valid there (finite and not
      ``no_data_val``). Pixels with no valid observation are no data in
      every output.
    * **Band scales**: the distances, and so the result and the deviations,
      depend on the scale of each band, so the bands should share one scale
      (e.g. surface reflectance x 10,000).
    * **Grids**: the images must be on the same pixel grid; otherwise an
      :class:`rsgispy.RSGISPyException` is raised.

    :param input_imgs: Sequence of image paths, all with the same bands.
    :param output_img: Output composite image (the bands in ``bands``, with
                       the input band names).
    :param bands: Optional 1-based band or bands to composite (Default =
                  ``None``: all).
    :param no_data_val: No-data value of the inputs, also written to the
                        outputs (Default = ``None``: that of the first image;
                        NaN for float outputs, or 0 for integer outputs, when
                        there is none or it cannot be stored).
    :param max_iter: Maximum number of iterations (Default = ``1000``).
    :param tol: Relative step tolerance (Default = ``1e-7``).
    :param out_mad_img: Optional output image of three ``float32`` bands,
                        ``EMAD``, ``SMAD`` and ``BCMAD``: the median absolute
                        deviations of Roberts et al. (2018), i.e. the median
                        over the valid observations of the Euclidean, cosine
                        (1 - cosine similarity) and Bray–Curtis
                        (``sum|x - m| / sum|x + m|``) distances between each
                        observation and the geomedian (no data NaN).
    :param out_count_img: Optional output image of the number of valid
                          observations (``uint16``, no data 0).
    :param out_ref_img: Optional thematic ``uint16`` image of the 1-based
                        index into ``input_imgs`` of the observation nearest
                        the geomedian (0 = none), with a colour table and
                        statistics.
    :param gdalformat: Output GDAL format (Default = ``'GTIFF'``).
    :param datatype: Output ``rsgispy.TYPE_*`` of ``output_img`` (Default =
                     ``None``: that of the first input image).
    :param block_size: Processing tile size in pixels (Default = ``None``:
                       the package default); the result does not depend on it.
    :param use_pbar: Show a progress bar (Default = ``True``).
    :param calc_stats: Calculate statistics and pyramids for the outputs
                       (Default = ``True``).
    :param creation_opts: Optional GDAL creation options.
    :param n_jobs: Number of threads evaluating tiles (Default = ``1``; see
                   :meth:`rsgispy.imageprocess.TileGraph.run`).

    Roberts, D., Dunn, B. and Mueller, N. (2018). Open Data Cube products
    using high-dimensional statistics of time series. IGARSS 2018, 8647–8650.

    .. code:: python

        import glob
        import rsgispy.imageutils.composite

        rsgispy.imageutils.composite.create_geomedian_composite(
            sorted(glob.glob("ls_imgs/*.tif")),
            "geomedian.tif",
            out_mad_img="geomedian_mads.tif",
            out_count_img="geomedian_count.tif",
            out_ref_img="geomedian_ref.tif",
        )
    """
    # pylint: disable=too-many-arguments,too-many-locals
    inputs = _Inputs(input_imgs)
    dt_out, np_out = inputs.datatype(datatype)
    in_nd = no_data_val if no_data_val is not None else inputs.nodata
    node = geomedian_node(
        inputs.sources, max_iter, tol, bands=bands, no_data_val=in_nd, dtype=np_out
    )
    out_nd = _resolve_nodata(in_nd, np_out)
    names = infer_node_meta(node).band_names
    sinks = [
        _continuous_sink(
            node, output_img, gdalformat, dt_out, out_nd, names, calc_stats, creation_opts
        )
    ]
    if out_mad_img is not None:
        sinks.append(_continuous_sink(
            node.mad, out_mad_img, gdalformat, rsgispy.TYPE_32FLOAT, float("nan"),
            MAD_BAND_NAMES, calc_stats, creation_opts,
        ))
    if out_count_img is not None:
        sinks.append(_continuous_sink(
            node.count, out_count_img, gdalformat, rsgispy.TYPE_16UINT, 0.0,
            [COUNT_BAND_NAME], calc_stats, creation_opts,
        ))
    if out_ref_img is not None:
        sinks.append(_thematic_sink(
            node.ref, out_ref_img, gdalformat, rsgispy.TYPE_16UINT, REF_BAND_NAME,
            _ref_colour_table(len(inputs.paths)), calc_stats, creation_opts,
            _ref_class_names(inputs.paths),
        ))
    _run(sinks, block_size, use_pbar, n_jobs)


def _selection_sinks(
    node: CompositeNode,
    inputs: _Inputs,
    output_img: Union[str, os.PathLike],
    out_imgs: Mapping[str, Optional[Union[str, os.PathLike]]],
    gdalformat: str,
    dt_out: int,
    out_nd: float,
    calc_stats: bool,
    creation_opts: Optional[Mapping[str, Any]],
) -> List[WrapperDiskSink]:
    """Sinks of a selection composite and its optional outputs."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    names = infer_node_meta(node).band_names
    sinks = [
        _continuous_sink(
            node, output_img, gdalformat, dt_out, out_nd, names, calc_stats, creation_opts
        )
    ]
    if out_imgs.get("ref") is not None:
        sinks.append(_thematic_sink(
            node.ref, out_imgs["ref"], gdalformat, rsgispy.TYPE_16UINT,  # type: ignore[arg-type]
            REF_BAND_NAME, _ref_colour_table(len(inputs.paths)), calc_stats, creation_opts,
            _ref_class_names(inputs.paths),
        ))
    if out_imgs.get("quality") is not None:
        sinks.append(_continuous_sink(
            node.quality, out_imgs["quality"], gdalformat,  # type: ignore[arg-type]
            rsgispy.TYPE_32FLOAT, float("nan"), [QUALITY_BAND_NAME], calc_stats, creation_opts,
        ))
    if out_imgs.get("lw_msk") is not None:
        sinks.append(_thematic_sink(
            node.lw_msk, out_imgs["lw_msk"], gdalformat, rsgispy.TYPE_8UINT,  # type: ignore[arg-type]
            LW_BAND_NAME, LW_COLOURS, calc_stats, creation_opts, LW_CLASS_NAMES,
        ))
    return sinks


def create_quality_mosaic(
    input_imgs: Sequence[Union[str, os.PathLike]],
    output_img: Union[str, os.PathLike],
    quality: QualitySpec,
    *,
    mode: str = "max",
    percentile: Optional[float] = None,
    smooth_ref_size: Optional[int] = None,
    bands: Optional[Union[int, Sequence[int]]] = None,
    no_data_val: Optional[float] = None,
    out_ref_img: Optional[Union[str, os.PathLike]] = None,
    out_quality_img: Optional[Union[str, os.PathLike]] = None,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    block_size: Optional[int] = None,
    use_pbar: bool = True,
    calc_stats: bool = True,
    creation_opts: Optional[Mapping[str, Any]] = None,
    n_jobs: Optional[int] = 1,
) -> None:
    """
    Create a quality mosaic: each pixel takes the whole observation (every
    band) whose quality value is the largest (``mode='max'``) or smallest
    (``mode='min'``), like Google Earth Engine's ``qualityMosaic``. The
    output pixels are real observations, so the bands stay consistent.

    * **Quality**: one of

      - a 1-based band number, e.g. the NIR band for a maximum NIR
        composite, or the blue band with ``mode='min'`` as a simple haze and
        cloud filter;
      - a band-maths expression over the bands of each observation, named
        ``b1`` … ``bn`` (the muParser syntax of
        :func:`rsgispy.imagecalc.maths.image_band_math`), e.g.
        ``'(b4 - b3) / (b4 + b3)'``;
      - a callable taking a ``(n_bands, rows, cols)`` float64 block of one
        observation (all its bands; invalid values are NaN) and returning a
        ``(rows, cols)`` quality array.

      Band numbers and ``b1`` … ``bn`` refer to the bands of the input
      images, whichever ``bands`` are composited.
    * **Valid observations**: an observation is a candidate at a pixel only
      when every band in ``bands`` (and every band a band-number or
      expression quality uses) is valid there, and its quality is finite, so
      a division by zero in an index gives no candidate rather than an
      error. Pixels without a candidate are no data in every output.
    * **Ties** go to the earliest image in ``input_imgs``.
    * **Percentile selection** (``mode='percentile'``, ``percentile=p``):
      each pixel takes the observation whose quality is nearest the p-th
      percentile of its candidates' qualities (``numpy.nanpercentile``,
      linear interpolation), so a whole observation is still chosen; ties go
      to the higher quality, then the earliest image. A pure maximum is drawn
      to outliers (unmasked cloud edges, odd index values), so e.g.
      ``percentile=90`` for NDVI is more robust than the maximum; use it
      when the inputs are not perfectly masked. ``percentile=100`` equals
      ``mode='max'`` and ``percentile=0`` equals ``mode='min'``.
    * **Smoothing the choice** (``smooth_ref_size=n``, an odd window size):
      a majority (mode) filter (as
      :func:`~rsgispy.classification.classpostprocess.apply_majority_filter`)
      is applied to the choice of observation before the bands are copied,
      reducing speckled seams where neighbouring pixels come from different
      dates. A pixel keeps its own choice when the majority observation is
      not a candidate there. The result does not depend on the block size.
      ``out_ref_img`` and ``out_quality_img`` describe the smoothed choice.
    * **Grids**: the images must be on the same pixel grid; otherwise an
      :class:`rsgispy.RSGISPyException` is raised.

    :param input_imgs: Sequence of image paths, all with the same bands.
    :param output_img: Output composite image (the bands in ``bands``).
    :param quality: Band number, expression or callable (see above).
    :param mode: ``'max'`` (Default), ``'min'`` or ``'percentile'``.
    :param percentile: Percentile in ``[0, 100]`` for ``mode='percentile'``.
    :param smooth_ref_size: Optional odd window size (>= 3) of the majority
                            filter applied to the choice of observation
                            (Default = ``None``: no smoothing).
    :param bands: Optional 1-based band or bands to composite (Default =
                  ``None``: all).
    :param no_data_val: No-data value of the inputs, also written to the
                        output (Default = ``None``: that of the first image;
                        NaN or 0 when there is none or it cannot be stored).
    :param out_ref_img: Optional thematic ``uint16`` image of the 1-based
                        index into ``input_imgs`` of the observation used
                        (0 = none), with a colour table and statistics.
    :param out_quality_img: Optional ``float32`` image of the quality of the
                            selected observation (no data NaN).
    :param gdalformat: Output GDAL format (Default = ``'GTIFF'``).
    :param datatype: Output ``rsgispy.TYPE_*`` (Default = ``None``: that of
                     the first input image).
    :param block_size: Processing tile size in pixels (Default = ``None``:
                       the package default).
    :param use_pbar: Show a progress bar (Default = ``True``).
    :param calc_stats: Calculate statistics and pyramids (Default = ``True``).
    :param creation_opts: Optional GDAL creation options.
    :param n_jobs: Number of threads evaluating tiles (Default = ``1``).

    .. code:: python

        import glob
        import rsgispy.imageutils.composite

        imgs = sorted(glob.glob("ls_imgs/*.tif"))
        # Landsat 8/9: b4 = red, b5 = NIR; the observation at the 90th
        # percentile of NDVI, with a 3x3 smoothing of the choice.
        rsgispy.imageutils.composite.create_quality_mosaic(
            imgs,
            "ndvi_p90_comp.tif",
            "(b5 - b4) / (b5 + b4)",
            mode="percentile",
            percentile=90,
            smooth_ref_size=3,
            out_ref_img="ndvi_p90_ref.tif",
            out_quality_img="ndvi_p90_ndvi.tif",
        )
    """
    # pylint: disable=too-many-arguments,too-many-locals
    inputs = _Inputs(input_imgs)
    dt_out, np_out = inputs.datatype(datatype)
    in_nd = no_data_val if no_data_val is not None else inputs.nodata
    node = quality_mosaic_node(
        inputs.sources, quality, mode, percentile, smooth_ref_size=smooth_ref_size,
        bands=bands, no_data_val=in_nd, dtype=np_out,
    )
    sinks = _selection_sinks(
        node, inputs, output_img, {"ref": out_ref_img, "quality": out_quality_img},
        gdalformat, dt_out, _resolve_nodata(in_nd, np_out), calc_stats, creation_opts,
    )
    _run(sinks, block_size, use_pbar, n_jobs)


def _norm_diff_expr(band_a: int, band_b: int) -> str:
    """Band-maths expression of the normalised difference ``(a - b) / (a + b)``."""
    return f"(b{int(band_a)} - b{int(band_b)}) / (b{int(band_a)} + b{int(band_b)})"


def _ndvi_expr(red_band: int, nir_band: int) -> str:
    """NDVI band-maths expression."""
    return _norm_diff_expr(nir_band, red_band)


def create_max_ndvi_composite(
    input_imgs: Sequence[Union[str, os.PathLike]],
    output_img: Union[str, os.PathLike],
    red_band: int,
    nir_band: int,
    *,
    out_ref_img: Optional[Union[str, os.PathLike]] = None,
    **kwargs: Any,
) -> None:
    """
    Create a maximum NDVI composite: each pixel takes the whole observation
    with the largest NDVI, ``(NIR - red) / (NIR + red)`` (the greenest, and
    usually the least cloudy, observation). This is
    :func:`create_quality_mosaic` with that expression and ``mode='max'``;
    observations where NDVI is undefined (``NIR + red = 0``) are not
    candidates.

    :param input_imgs: Sequence of image paths, all with the same bands.
    :param output_img: Output composite image.
    :param red_band: 1-based red band.
    :param nir_band: 1-based near-infrared band.
    :param out_ref_img: Optional thematic ``uint16`` image of the 1-based
                        index of the observation used (0 = none).
    :param kwargs: Other keyword arguments of :func:`create_quality_mosaic`
                   (e.g. ``out_quality_img`` for the NDVI of the selected
                   observation, ``smooth_ref_size``, ``bands``,
                   ``no_data_val``, ``gdalformat``, ``datatype``).

    .. code:: python

        import glob
        import rsgispy.imageutils.composite

        rsgispy.imageutils.composite.create_max_ndvi_composite(
            sorted(glob.glob("ls_imgs/*.tif")),
            "max_ndvi_comp.tif",
            red_band=4,
            nir_band=5,
            out_ref_img="max_ndvi_ref.tif",
        )
    """
    create_quality_mosaic(
        input_imgs, output_img, _ndvi_expr(red_band, nir_band), mode="max",
        out_ref_img=out_ref_img, **kwargs,
    )


def create_land_water_composite(
    input_imgs: Sequence[Union[str, os.PathLike]],
    output_img: Union[str, os.PathLike],
    land_quality: QualitySpec,
    water_quality: Optional[QualitySpec],
    *,
    land_mode: str = "max",
    water_mode: str = "max",
    land_percentile: Optional[float] = None,
    water_percentile: Optional[float] = None,
    smooth_lw_msk_size: Optional[int] = None,
    smooth_ref_size: Optional[int] = None,
    lw_msk_img: Optional[Union[str, os.PathLike]] = None,
    water_rule: Optional[QualitySpec] = None,
    water_rule_mode: str = "max",
    min_water_frac: float = 0.5,
    water_method: str = "quality",
    max_iter: int = 1000,
    tol: float = 1e-7,
    bands: Optional[Union[int, Sequence[int]]] = None,
    no_data_val: Optional[float] = None,
    out_ref_img: Optional[Union[str, os.PathLike]] = None,
    out_lw_msk_img: Optional[Union[str, os.PathLike]] = None,
    out_quality_img: Optional[Union[str, os.PathLike]] = None,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    block_size: Optional[int] = None,
    use_pbar: bool = True,
    calc_stats: bool = True,
    creation_opts: Optional[Mapping[str, Any]] = None,
    n_jobs: Optional[int] = 1,
) -> None:
    """
    Create a land / water composite: a quality mosaic (see
    :func:`create_quality_mosaic`) in which land and water pixels use
    different rules to choose the observation. Maximum NDVI picks the
    greenest observation, which suits land but not water; over water the
    observation with the largest NDWI (the "wettest", least affected by
    glint, cloud or haze), or a robust medoid or geomedian, is usually
    better.

    **Land / water decision**, in order of precedence:

    1. ``lw_msk_img``: a land (1) / water (2) image on the same grid (e.g.
       from a land cover map). Pixels that are neither are no data.
    2. ``water_rule``: a quality in the form of :func:`create_quality_mosaic`
       evaluated for each valid observation. With ``water_rule_mode='max'``
       (Default) an observation votes water where the rule is ``> 0`` (e.g.
       ``'b5 < 500 ? 1 : 0'``, 1 for water and 0 for land, or an NDWI, which
       is positive over water) and land otherwise; with ``'min'`` it votes
       water where the rule is ``< 0`` (e.g. ``'b5 - 500'``). A pixel is
       water when at least ``min_water_frac`` of its votes (valid
       observations with a finite rule) are water, so tidal and seasonal
       water can be handled with the fraction; a pixel with valid
       observations but no vote is land.

    One of the two must be given (:func:`create_max_ndvi_ndwi_composite`
    has a default rule).

    **Selection**: land pixels are a quality mosaic of ``land_quality``
    with ``land_mode`` (and ``land_percentile``). Water pixels use
    ``water_method``:

    * ``'quality'`` (Default): a quality mosaic of ``water_quality`` with
      ``water_mode`` (and ``water_percentile``);
    * ``'medoid'``: the medoid of the valid observations (see
      :func:`create_medoid_composite`);
    * ``'geomedian'``: the geomedian (see :func:`create_geomedian_composite`,
      with ``max_iter`` and ``tol``); the reference image then holds the
      observation nearest the geomedian.

    **Smoothing**:

    * ``smooth_lw_msk_size=n`` applies a majority filter to the land /
      water decision before the selection, removing isolated land pixels in
      water (and the reverse), which matter most at coastlines, where the
      decision sets which rule is used. Pixels with no valid observation
      stay no data.
    * ``smooth_ref_size=n`` smooths the choice of observation afterwards, as
      in :func:`create_quality_mosaic`, within each class only: a land pixel
      never takes a water pixel's observation. Water pixels of a geomedian
      are not observations and are not smoothed.

    Valid observations, empty pixels, ties and grids follow
    :func:`create_quality_mosaic`.

    :param input_imgs: Sequence of image paths, all with the same bands.
    :param output_img: Output composite image (the bands in ``bands``).
    :param land_quality: Quality over land (band, expression or callable).
    :param water_quality: Quality over water; only needed with
                          ``water_method='quality'`` (otherwise, when given,
                          its value for the selected observation is written
                          to ``out_quality_img``).
    :param land_mode: ``'max'`` (Default), ``'min'`` or ``'percentile'``.
    :param water_mode: ``'max'`` (Default), ``'min'`` or ``'percentile'``.
    :param land_percentile: Percentile for ``land_mode='percentile'``.
    :param water_percentile: Percentile for ``water_mode='percentile'``.
    :param smooth_lw_msk_size: Optional odd window size of the majority
                               filter of the land / water decision.
    :param smooth_ref_size: Optional odd window size of the majority filter
                            of the choice of observation (within each class).
    :param lw_msk_img: Optional land (1) / water (2) image on the same grid.
    :param water_rule: Optional per-observation water rule (see above).
    :param water_rule_mode: ``'max'`` (Default; water where the rule is
                            ``> 0``) or ``'min'`` (water where ``< 0``).
    :param min_water_frac: Minimum fraction of water votes for a water pixel,
                           in ``(0, 1]`` (Default = ``0.5``).
    :param water_method: ``'quality'`` (Default), ``'medoid'`` or
                         ``'geomedian'``.
    :param max_iter: Geomedian iterations (Default = ``1000``).
    :param tol: Geomedian tolerance (Default = ``1e-7``).
    :param bands: Optional 1-based band or bands to composite (Default = all).
    :param no_data_val: No-data value of the inputs, also written to the
                        output (Default = ``None``: that of the first image).
    :param out_ref_img: Optional thematic ``uint16`` image of the 1-based
                        index of the observation used (0 = none).
    :param out_lw_msk_img: Optional thematic ``uint8`` land (1) / water (2)
                           mask used (after smoothing; 0 = no data), with a
                           colour table (land green, water blue).
    :param out_quality_img: Optional ``float32`` image of the quality of the
                            selected observation (``land_quality`` over land,
                            ``water_quality`` over water; no data NaN).
    :param gdalformat: Output GDAL format (Default = ``'GTIFF'``).
    :param datatype: Output ``rsgispy.TYPE_*`` (Default = ``None``: that of
                     the first input image).
    :param block_size: Processing tile size in pixels (Default = ``None``).
    :param use_pbar: Show a progress bar (Default = ``True``).
    :param calc_stats: Calculate statistics and pyramids (Default = ``True``).
    :param creation_opts: Optional GDAL creation options.
    :param n_jobs: Number of threads evaluating tiles (Default = ``1``).

    .. code:: python

        import glob
        import rsgispy.imageutils.composite

        # Landsat 8/9: b3 = green, b4 = red, b5 = NIR. Max NDVI over land,
        # the medoid over water; water where NIR < 5 % in most observations.
        rsgispy.imageutils.composite.create_land_water_composite(
            sorted(glob.glob("ls_imgs/*.tif")),
            "lw_comp.tif",
            "(b5 - b4) / (b5 + b4)",
            None,
            water_rule="b5 < 500 ? 1 : 0",
            water_method="medoid",
            smooth_lw_msk_size=3,
            out_ref_img="lw_comp_ref.tif",
            out_lw_msk_img="lw_comp_msk.tif",
        )
    """
    # pylint: disable=too-many-arguments,too-many-locals
    inputs = _Inputs(input_imgs)
    dt_out, np_out = inputs.datatype(datatype)
    in_nd = no_data_val if no_data_val is not None else inputs.nodata
    msk_node = (
        RasterSource(os.fspath(lw_msk_img), bands=1, dtype=np.float64, name="lw_msk")
        if lw_msk_img is not None else None
    )
    node = land_water_composite_node(
        inputs.sources, land_quality, water_quality,
        land_mode=land_mode, water_mode=water_mode,
        land_percentile=land_percentile, water_percentile=water_percentile,
        smooth_lw_msk_size=smooth_lw_msk_size, smooth_ref_size=smooth_ref_size,
        lw_msk=msk_node, water_rule=water_rule, water_rule_mode=water_rule_mode,
        min_water_frac=min_water_frac, water_method=water_method,
        max_iter=max_iter, tol=tol, bands=bands, no_data_val=in_nd, dtype=np_out,
    )
    sinks = _selection_sinks(
        node, inputs, output_img,
        {"ref": out_ref_img, "quality": out_quality_img, "lw_msk": out_lw_msk_img},
        gdalformat, dt_out, _resolve_nodata(in_nd, np_out), calc_stats, creation_opts,
    )
    _run(sinks, block_size, use_pbar, n_jobs)


class RSGISLibLandWaterRule:
    """
    The per-observation land / water rule of RSGISLib's
    ``imageutils.imagecomp.create_max_ndvi_ndwi_composite``, as a quality
    callable for ``water_rule``: land where NDVI > ``ndvi_thres``, otherwise
    water where NDWI > ``ndwi_thres``, otherwise land; no vote where NDVI is
    undefined. NDWI is ``(NIR - SWIR) / (NIR + SWIR)`` (Gao, as RSGISLib)
    when ``swir_band`` is given, otherwise ``(green - NIR) / (green + NIR)``
    (McFeeters).

    :param red_band: 1-based red band.
    :param nir_band: 1-based NIR band.
    :param green_band: Optional 1-based green band.
    :param swir_band: Optional 1-based SWIR band.
    :param ndvi_thres: NDVI above which an observation is land
                       (Default = ``0.3``, as RSGISLib).
    :param ndwi_thres: NDWI above which an observation is water
                       (Default = ``0.01``, as RSGISLib).

    .. code:: python

        from rsgispy.imageutils.composite import RSGISLibLandWaterRule

        rule = RSGISLibLandWaterRule(red_band=4, nir_band=5, swir_band=6)
    """

    # pylint: disable=too-few-public-methods,too-many-arguments
    # pylint: disable=too-many-positional-arguments

    def __init__(
        self,
        red_band: int,
        nir_band: int,
        green_band: Optional[int] = None,
        swir_band: Optional[int] = None,
        ndvi_thres: float = 0.3,
        ndwi_thres: float = 0.01,
    ) -> None:
        if green_band is None and swir_band is None:
            raise rsgispy.RSGISPyException(
                "A green_band or a swir_band is needed for the NDWI."
            )
        self.red = int(red_band) - 1
        self.nir = int(nir_band) - 1
        self.green = None if green_band is None else int(green_band) - 1
        self.swir = None if swir_band is None else int(swir_band) - 1
        self.ndvi_thres = float(ndvi_thres)
        self.ndwi_thres = float(ndwi_thres)
        self.__name__ = "rsgislib_land_water_rule"

    def __call__(self, blk: np.ndarray) -> np.ndarray:
        red, nir = blk[self.red], blk[self.nir]
        with np.errstate(divide="ignore", invalid="ignore"):
            ndvi = (nir - red) / (nir + red)
            if self.swir is not None:
                ndwi = (nir - blk[self.swir]) / (nir + blk[self.swir])
            else:
                ndwi = (blk[self.green] - nir) / (blk[self.green] + nir)
        water = np.where(ndvi > self.ndvi_thres, 0.0, np.where(ndwi > self.ndwi_thres, 1.0, 0.0))
        water[~np.isfinite(ndvi)] = np.nan
        return water


def create_max_ndvi_ndwi_composite(
    input_imgs: Sequence[Union[str, os.PathLike]],
    output_img: Union[str, os.PathLike],
    red_band: int,
    nir_band: int,
    green_band: Optional[int] = None,
    swir_band: Optional[int] = None,
    **kwargs: Any,
) -> None:
    """
    Create a maximum NDVI / NDWI composite: :func:`create_land_water_composite`
    with the maximum NDVI, ``(NIR - red) / (NIR + red)``, over land and the
    maximum NDWI over water. The NDWI is

    * ``(green - NIR) / (green + NIR)`` (McFeeters) with ``green_band``;
    * ``(green - SWIR) / (green + SWIR)`` (MNDWI) with ``green_band`` and
      ``swir_band``;
    * ``(NIR - SWIR) / (NIR + SWIR)`` (Gao, as RSGISLib) with ``swir_band``
      only, i.e. RSGISLib's band arguments (``r_band``, ``n_band``,
      ``s_band``).

    Unless ``lw_msk_img`` or ``water_rule`` is given, the land / water
    decision is RSGISLib's (:class:`RSGISLibLandWaterRule`): each
    observation is land where NDVI > 0.3, otherwise water where NDWI > 0.01
    (the Gao NDWI when ``swir_band`` is given, otherwise McFeeters'),
    otherwise land, and the votes are combined with ``min_water_frac``
    (Default = 0.5, i.e. the majority, as RSGISLib's mode). Unlike RSGISLib,
    the images must already be on the same grid (RSGISLib subset or
    resampled them to a reference image).

    :param input_imgs: Sequence of image paths, all with the same bands.
    :param output_img: Output composite image.
    :param red_band: 1-based red band.
    :param nir_band: 1-based NIR band.
    :param green_band: Optional 1-based green band.
    :param swir_band: Optional 1-based SWIR band (one of ``green_band`` and
                      ``swir_band`` must be given).
    :param kwargs: Other keyword arguments of
                   :func:`create_land_water_composite` (e.g. ``out_ref_img``,
                   ``out_lw_msk_img``, ``water_method``, ``lw_msk_img``,
                   ``water_rule``, ``smooth_lw_msk_size``); ``land_quality``
                   and ``water_quality`` can override the NDVI and NDWI.

    .. code:: python

        import glob
        import rsgispy.imageutils.composite

        # Landsat 8/9: b3 = green, b4 = red, b5 = NIR, b6 = SWIR 1.
        rsgispy.imageutils.composite.create_max_ndvi_ndwi_composite(
            sorted(glob.glob("ls_imgs/*.tif")),
            "max_ndvi_ndwi_comp.tif",
            red_band=4,
            nir_band=5,
            swir_band=6,
            out_ref_img="max_ndvi_ndwi_ref.tif",
            out_lw_msk_img="max_ndvi_ndwi_lw_msk.tif",
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    if green_band is None and swir_band is None:
        raise rsgispy.RSGISPyException("green_band or swir_band must be given.")
    if green_band is not None and swir_band is not None:
        ndwi = _norm_diff_expr(green_band, swir_band)  # MNDWI
    elif green_band is not None:
        ndwi = _norm_diff_expr(green_band, nir_band)  # McFeeters NDWI
    else:
        ndwi = _norm_diff_expr(nir_band, swir_band)  # Gao NDWI, as RSGISLib
    land_quality = kwargs.pop("land_quality", _ndvi_expr(red_band, nir_band))
    water_quality = kwargs.pop("water_quality", ndwi)
    if kwargs.get("lw_msk_img") is None and kwargs.get("water_rule") is None:
        kwargs["water_rule"] = RSGISLibLandWaterRule(
            red_band, nir_band, green_band=green_band, swir_band=swir_band
        )
    create_land_water_composite(input_imgs, output_img, land_quality, water_quality, **kwargs)


def create_medoid_composite(
    input_imgs: Sequence[Union[str, os.PathLike]],
    output_img: Union[str, os.PathLike],
    *,
    bands: Optional[Union[int, Sequence[int]]] = None,
    no_data_val: Optional[float] = None,
    out_ref_img: Optional[Union[str, os.PathLike]] = None,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    block_size: Optional[int] = None,
    use_pbar: bool = True,
    calc_stats: bool = True,
    creation_opts: Optional[Mapping[str, Any]] = None,
    n_jobs: Optional[int] = 1,
) -> None:
    """
    Create a medoid composite (Flood, 2013): each pixel takes the valid
    observation with the smallest sum of Euclidean distances (over
    ``bands``) to the other valid observations; ties go to the earliest
    image. Like a median it is robust to outliers, but the output pixel is a
    real observation. For a point that need not be an observation, see
    :func:`create_geomedian_composite`.

    An observation is valid at a pixel only when every band in ``bands`` is
    valid there; pixels with no valid observation are no data in every
    output. The distances depend on the band scales, so the bands should
    share one scale. The images must be on the same pixel grid.

    Flood, N. (2013). Seasonal composite Landsat TM/ETM+ images using the
    medoid (a multi-dimensional median). Remote Sensing, 5(12), 6481–6500.

    :param input_imgs: Sequence of image paths, all with the same bands.
    :param output_img: Output composite image (the bands in ``bands``).
    :param bands: Optional 1-based band or bands to composite (Default = all).
    :param no_data_val: No-data value of the inputs, also written to the
                        output (Default = ``None``: that of the first image).
    :param out_ref_img: Optional thematic ``uint16`` image of the 1-based
                        index of the observation used (0 = none), with a
                        colour table and statistics.
    :param gdalformat: Output GDAL format (Default = ``'GTIFF'``).
    :param datatype: Output ``rsgispy.TYPE_*`` (Default = ``None``: that of
                     the first input image).
    :param block_size: Processing tile size in pixels (Default = ``None``).
    :param use_pbar: Show a progress bar (Default = ``True``).
    :param calc_stats: Calculate statistics and pyramids (Default = ``True``).
    :param creation_opts: Optional GDAL creation options.
    :param n_jobs: Number of threads evaluating tiles (Default = ``1``).

    .. code:: python

        import glob
        import rsgispy.imageutils.composite

        rsgispy.imageutils.composite.create_medoid_composite(
            sorted(glob.glob("ls_imgs/*.tif")),
            "medoid_comp.tif",
            out_ref_img="medoid_ref.tif",
        )
    """
    # pylint: disable=too-many-arguments,too-many-locals
    inputs = _Inputs(input_imgs)
    dt_out, np_out = inputs.datatype(datatype)
    in_nd = no_data_val if no_data_val is not None else inputs.nodata
    node = medoid_node(inputs.sources, bands=bands, no_data_val=in_nd, dtype=np_out)
    sinks = _selection_sinks(
        node, inputs, output_img, {"ref": out_ref_img}, gdalformat, dt_out,
        _resolve_nodata(in_nd, np_out), calc_stats, creation_opts,
    )
    _run(sinks, block_size, use_pbar, n_jobs)


def create_percentile_composite(
    input_imgs: Sequence[Union[str, os.PathLike]],
    output_img: Union[str, os.PathLike],
    percentiles: Union[float, Sequence[float]] = (50,),
    *,
    bands: Optional[Union[int, Sequence[int]]] = None,
    no_data_val: Optional[float] = None,
    valid_rule: str = "all_bands",
    out_count_img: Optional[Union[str, os.PathLike]] = None,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = rsgispy.TYPE_32FLOAT,
    block_size: Optional[int] = None,
    use_pbar: bool = True,
    calc_stats: bool = True,
    creation_opts: Optional[Mapping[str, Any]] = None,
    n_jobs: Optional[int] = 1,
) -> None:
    """
    Create a per-band percentile composite: the median, or any percentiles
    (e.g. 10, 25, 75, 90), of each band separately over the observations.
    Percentile composites summarise a season, so they are widely used as
    classification features.

    Unlike :func:`rsgispy.imagecalc.percentiles.calc_imgs_pxl_percentiles`,
    which pools all the bands of all the images into one set of values (so
    it cannot make a multi-band composite), each band is summarised
    separately. :func:`rsgispy.imagecalc.percentiles.calc_img_band_pxl_percentiles`
    also works per band, but over each band's own valid values, in the input
    data type (truncated) and without a band selection or count output.

    * **Output bands**: one band per input band and percentile, grouped by
      band (all the percentiles of band 1, then band 2, …), named
      ``{band name}_p{percentile}`` (e.g. ``Red_p50``, ``NIR_p90``, or
      ``b1_p2.5`` for an unnamed band and a fractional percentile).
    * **Values**: ``numpy.nanpercentile`` (linear interpolation) of the
      valid observations, written as ``float32`` by default; with an integer
      ``datatype`` the values are rounded, not truncated.
    * **Validity**: with ``valid_rule='all_bands'`` (Default) an observation
      counts only where every band in ``bands`` is valid (the rule of the
      other composites); with ``'per_band'`` each band uses its own valid
      observations, so more data survive where the bands have different
      gaps. Pixels (or, with ``'per_band'``, bands) without a valid
      observation are no data.
    * **Grids**: the images must be on the same pixel grid.

    :param input_imgs: Sequence of image paths, all with the same bands.
    :param output_img: Output image.
    :param percentiles: Percentile or percentiles in ``[0, 100]``
                        (Default = ``(50,)``, the median).
    :param bands: Optional 1-based band or bands (Default = all).
    :param no_data_val: No-data value of the inputs, also written to the
                        output (Default = ``None``: that of the first image;
                        NaN when there is none).
    :param valid_rule: ``'all_bands'`` (Default) or ``'per_band'``.
    :param out_count_img: Optional ``uint16`` image of the number of valid
                          observations (no data 0): one band ``Count``, or
                          with ``valid_rule='per_band'`` one band per
                          composited band (``{band name}_count``).
    :param gdalformat: Output GDAL format (Default = ``'GTIFF'``).
    :param datatype: Output ``rsgispy.TYPE_*`` (Default =
                     ``rsgispy.TYPE_32FLOAT``; ``None``: that of the first
                     input image).
    :param block_size: Processing tile size in pixels (Default = ``None``).
    :param use_pbar: Show a progress bar (Default = ``True``).
    :param calc_stats: Calculate statistics and pyramids (Default = ``True``).
    :param creation_opts: Optional GDAL creation options.
    :param n_jobs: Number of threads evaluating tiles (Default = ``1``).

    .. code:: python

        import glob
        import rsgispy.imageutils.composite

        rsgispy.imageutils.composite.create_percentile_composite(
            sorted(glob.glob("s2_imgs/*.tif")),
            "s2_pcts.tif",
            percentiles=[10, 50, 90],
            out_count_img="s2_count.tif",
        )
    """
    # pylint: disable=too-many-arguments,too-many-locals
    inputs = _Inputs(input_imgs)
    dt_out, np_out = inputs.datatype(datatype)
    in_nd = no_data_val if no_data_val is not None else inputs.nodata
    node = percentile_composite_node(
        inputs.sources, percentiles, bands=bands, no_data_val=in_nd,
        valid_rule=valid_rule, dtype=np_out,
    )
    meta = infer_node_meta(node)
    sinks = [
        _continuous_sink(
            node, output_img, gdalformat, dt_out, _resolve_nodata(in_nd, np_out),
            meta.band_names, calc_stats, creation_opts,
        )
    ]
    if out_count_img is not None:
        sinks.append(_continuous_sink(
            node.count, out_count_img, gdalformat, rsgispy.TYPE_16UINT, 0.0,  # type: ignore[arg-type]
            infer_node_meta(node.count).band_names, calc_stats, creation_opts,  # type: ignore[arg-type]
        ))
    _run(sinks, block_size, use_pbar, n_jobs)
