"""
Statistics sinks for :mod:`rsgispy._tilegraph`: :class:`StatsSink` (streaming
per-band statistics and histograms) and :class:`ZonalStatsSink` (per-label
statistics of a value node within the zones of a label node).

Both accumulate the core block of every tile as it is written, so the
statistics of a whole processing chain are computed in the same single pass
as any other output, without writing (or re-reading) an image.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from rasterio.windows import Window

import rsgispy
from rsgispy._tilegraph._nodes import StackBands, TileNode, TileSink
from rsgispy._tilegraph._tensor import NodeMeta, TileContext, TileTensor

_STAT_NAMES = ("count", "min", "max", "mean", "std")


def _valid_mask(arr: np.ndarray, nodata: Optional[float]) -> np.ndarray:
    """Finite pixels that differ from ``nodata`` (NaN aware)."""
    # pylint: disable-next=import-outside-toplevel
    from rsgispy.imageprocess import build_valid_mask

    return build_valid_mask(arr, None if nodata is None else float(nodata))


def _band_labels(meta: NodeMeta, n_bands: int) -> List[str]:
    """Band names of ``meta`` or ``'b1'``, ``'b2'``, ..."""
    if meta.band_names is not None and len(meta.band_names) == n_bands:
        return list(meta.band_names)
    return [f"b{i + 1}" for i in range(n_bands)]


@dataclass
class RasterStats:
    """
    Per-band statistics returned by :class:`StatsSink` (one entry per band
    in every array). Bands without valid pixels have ``count == 0`` and NaN
    statistics.

    :param band_names: Band names (those of the input node, else ``'b1'``, ...).
    :param count: Number of valid pixels (``int64``).
    :param min: Minimum.
    :param max: Maximum.
    :param mean: Mean.
    :param std: Standard deviation (with the sink's ``ddof``).
    :param nodata: No-data value used to exclude pixels (``None`` if none).
    :param hist: Optional histogram counts, shape ``(bands, n_bins)``
                 (``int64``), as :func:`numpy.histogram` of each band's valid
                 pixels with ``bin_edges``.
    :param bin_edges: Optional histogram bin edges (``n_bins + 1`` values).
    :param ddof: Delta degrees of freedom of ``std`` (Default = ``0``).
    """

    # pylint: disable=too-many-instance-attributes

    band_names: List[str]
    count: np.ndarray
    min: np.ndarray
    max: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    nodata: Optional[float] = None
    hist: Optional[np.ndarray] = None
    bin_edges: Optional[np.ndarray] = None
    ddof: int = 0

    def to_dataframe(self) -> Any:
        """
        Return the statistics as a :class:`pandas.DataFrame` with one row per
        band (indexed by band name) and the columns ``count``, ``min``,
        ``max``, ``mean`` and ``std``.
        """
        import pandas as pd  # pylint: disable=import-outside-toplevel

        return pd.DataFrame(
            {name: getattr(self, name) for name in _STAT_NAMES},
            index=pd.Index(self.band_names, name="band"),
        )


class StatsSink(TileSink):
    """
    Sink computing streaming per-band statistics of its input: the count of
    valid pixels, minimum, maximum, mean and standard deviation (with the
    numerically stable updates of
    :class:`~rsgispy.imageprocess.OnlineStatsAccumulator`), and optionally a
    histogram. Pixels that are not finite or equal the no-data value are
    excluded. The :class:`RasterStats` result is available on :attr:`result`
    and in the dictionary returned by
    :meth:`~rsgispy.imageprocess.TileGraph.run` (under the sink's ``name``,
    or ``'stats_<i>'``).

    The statistics are those of the whole image (computed in one pass, tile
    by tile): the counts, minima, maxima and histograms are exact and the
    mean and standard deviation equal those of :func:`numpy.mean` /
    :func:`numpy.std` on all valid pixels to within floating-point rounding.

    :param source: Upstream :class:`~rsgispy.imageprocess.TileNode`.
    :param no_data_val: No-data value of the input (Default = ``None``: the
                        no-data value of ``source``'s metadata; wrap
                        ``source`` in :class:`~rsgispy.imageprocess.SetNoData`
                        with ``None`` to count every finite pixel).
    :param bins: Optional histogram: the number of bins (then ``hist_range``
                 is required) or a sequence of monotonically increasing bin
                 edges, as for :func:`numpy.histogram`.
    :param hist_range: ``(min, max)`` of the histogram when ``bins`` is a
                       number (values outside are not counted; the last bin
                       includes ``max``).
    :param ddof: Delta degrees of freedom of the standard deviation
                 (Default = ``0``, the population standard deviation, as
                 :func:`numpy.std`).
    :param name: Optional sink name (the key of the result).

    .. code:: python

        from rsgispy.imageprocess import RasterSource, StatsSink, TileGraph
        from rsgispy.imagecalc.calcindices import calc_indices_node

        refl = RasterSource("s2_refl.tif")
        ndvi = calc_indices_node(refl, {4: "red", 8: "nir"}, ["NDVI"])
        stats = TileGraph(
            StatsSink(ndvi, bins=20, hist_range=(-1, 1), name="ndvi")
        ).run()["ndvi"]
        print(stats.to_dataframe())
        print(stats.hist[0], stats.bin_edges)
    """

    # pylint: disable=too-many-instance-attributes

    returns_result = True
    result_prefix = "stats"

    def __init__(
        self,
        source: TileNode,
        *,
        no_data_val: Optional[float] = None,
        bins: Optional[Union[int, Sequence[float]]] = None,
        hist_range: Optional[Tuple[float, float]] = None,
        ddof: int = 0,
        name: Optional[str] = None,
    ) -> None:
        # pylint: disable=too-many-arguments
        super().__init__(source, name=name)
        self.no_data_val = no_data_val
        self.ddof = int(ddof)
        if self.ddof < 0:
            raise rsgispy.RSGISPyException(f"ddof must be >= 0, got {ddof}.")
        self.bin_edges: Optional[np.ndarray] = _resolve_bin_edges(bins, hist_range)
        self.result: Optional[RasterStats] = None
        self._nodata: Optional[float] = None
        self._meta = NodeMeta()
        self._acc: Any = None
        self._hist: Optional[np.ndarray] = None

    def open(
        self,
        grid: Any,
        meta: NodeMeta,
        *,
        stack: ExitStack,
        use_pbar: bool = True,
    ) -> None:
        del grid, stack, use_pbar
        self._meta = meta
        self._nodata = self.no_data_val if self.no_data_val is not None else meta.nodata
        self._acc = None
        self._hist = None
        self.result = None

    def write(self, block: np.ndarray, window: Window) -> None:
        # pylint: disable-next=import-outside-toplevel
        from rsgispy.imageprocess import OnlineStatsAccumulator

        n_bands = int(block.shape[0])
        if self._acc is None:
            self._acc = OnlineStatsAccumulator(n_bands)
            if self.bin_edges is not None:
                self._hist = np.zeros((n_bands, self.bin_edges.size - 1), dtype=np.int64)
        elif self._acc.n_bands != n_bands:
            raise rsgispy.RSGISPyException(
                f"{self.label}: a block has {n_bands} band(s), expected "
                f"{self._acc.n_bands}."
            )
        for b_idx in range(n_bands):
            band = block[b_idx]
            vals = band[_valid_mask(band, self._nodata)]
            if vals.size == 0:
                continue
            self._acc.update_band(b_idx, vals)
            if self._hist is not None:
                counts, _ = np.histogram(vals, bins=self.bin_edges)
                self._hist[b_idx] += counts

    def close(self) -> None:
        acc = self._acc
        if acc is None:
            n_bands = int(self._meta.n_bands or 0)
            counts = np.zeros(n_bands, dtype=np.int64)
            mins = maxs = means = stds = np.full(n_bands, np.nan)
            hist = (
                np.zeros((n_bands, self.bin_edges.size - 1), dtype=np.int64)
                if self.bin_edges is not None
                else None
            )
        else:
            n_bands = acc.n_bands
            counts = acc.counts.copy()
            has = counts > 0
            mins = np.where(has, acc.mins, np.nan)
            maxs = np.where(has, acc.maxs, np.nan)
            means = np.where(has, acc.means, np.nan)
            stds = np.where(counts > self.ddof, acc.stds(self.ddof), np.nan)
            hist = self._hist
        self.result = RasterStats(
            band_names=_band_labels(self._meta, n_bands),
            count=counts,
            min=mins,
            max=maxs,
            mean=means,
            std=stds,
            nodata=self._nodata,
            hist=hist,
            bin_edges=None if self.bin_edges is None else self.bin_edges.copy(),
            ddof=self.ddof,
        )
        self._acc = None
        self._hist = None

    def describe_params(self) -> str:
        nd = "from input" if self.no_data_val is None else repr(self.no_data_val)
        hist = "" if self.bin_edges is None else f" bins={self.bin_edges.size - 1}"
        return f"no_data_val={nd}{hist}"


def _resolve_bin_edges(
    bins: Optional[Union[int, Sequence[float]]],
    hist_range: Optional[Tuple[float, float]],
) -> Optional[np.ndarray]:
    """Histogram bin edges from ``bins`` / ``hist_range`` (``None``: no histogram)."""
    if bins is None:
        if hist_range is not None:
            raise rsgispy.RSGISPyException("hist_range was given without bins.")
        return None
    if isinstance(bins, (int, np.integer)) and not isinstance(bins, bool):
        if int(bins) < 1:
            raise rsgispy.RSGISPyException(f"bins must be >= 1, got {bins}.")
        if hist_range is None:
            raise rsgispy.RSGISPyException(
                "hist_range=(min, max) is required when bins is a number of bins "
                "(the statistics are computed in one pass)."
            )
        lo, hi = float(hist_range[0]), float(hist_range[1])
        if not (math.isfinite(lo) and math.isfinite(hi)) or lo > hi:
            raise rsgispy.RSGISPyException(
                f"hist_range must be finite with min <= max, got {hist_range}."
            )
        _, edges = np.histogram(np.empty(0), bins=int(bins), range=(lo, hi))
        return edges
    edges = np.asarray(bins, dtype=np.float64)
    if edges.ndim != 1 or edges.size < 2 or np.any(np.diff(edges) < 0):
        raise rsgispy.RSGISPyException(
            "bins must be a number of bins or a 1D sequence of at least two "
            "increasing bin edges."
        )
    if hist_range is not None:
        raise rsgispy.RSGISPyException(
            "hist_range cannot be given with explicit bin edges."
        )
    return edges


class _ZonalStack(StackBands):
    """:class:`StackBands` of ``[labels, values]`` keeping the inputs' metadata."""

    def __init__(self, labels: TileNode, values: TileNode, name: Optional[str]) -> None:
        super().__init__([labels, values], name=name)
        self.input_metas: Tuple[NodeMeta, ...] = (NodeMeta(), NodeMeta())

    def infer_meta(self, inputs: Sequence[NodeMeta]) -> NodeMeta:
        label_meta = inputs[0]
        if label_meta.n_bands is not None and label_meta.n_bands != 1:
            raise rsgispy.RSGISPyException(
                f"ZonalStatsSink: the label node must have 1 band, got "
                f"{label_meta.n_bands} (select one with SelectBands)."
            )
        self.input_metas = (inputs[0], inputs[1])
        return super().infer_meta(inputs).replace(dtype=np.dtype(np.float64))

    def evaluate(
        self,
        inputs: Sequence[TileTensor],
        halo_y: int,
        halo_x: int,
        ctx: TileContext,
        state: Any = None,
    ) -> TileTensor:
        # float64, so integer labels are kept exactly whatever the value dtype.
        aligned = [t.crop(halo_y, halo_x) for t in inputs]
        return aligned[0].with_data(
            np.concatenate(
                [t.data.astype(np.float64, copy=False) for t in aligned], axis=0
            )
        )


class _ZoneTable:
    """Per-label running count, mean, M2, min and max of each value band."""

    # pylint: disable=too-few-public-methods

    def __init__(self, n_bands: int) -> None:
        self.n_bands = int(n_bands)
        self.labels = np.empty(0, dtype=np.float64)
        self.count = np.zeros((0, n_bands), dtype=np.int64)
        self.mean = np.zeros((0, n_bands), dtype=np.float64)
        self.m2 = np.zeros((0, n_bands), dtype=np.float64)
        self.min = np.zeros((0, n_bands), dtype=np.float64)
        self.max = np.zeros((0, n_bands), dtype=np.float64)

    def _add_labels(self, new_labels: np.ndarray) -> None:
        missing = np.setdiff1d(new_labels, self.labels, assume_unique=True)
        if missing.size == 0:
            return
        labels = np.concatenate([self.labels, missing])
        order = np.argsort(labels, kind="stable")
        n_new = missing.size
        shape = (n_new, self.n_bands)

        def _grow(arr: np.ndarray, fill: float) -> np.ndarray:
            return np.concatenate([arr, np.full(shape, fill, dtype=arr.dtype)])[order]

        self.count = _grow(self.count, 0)
        self.mean = _grow(self.mean, 0.0)
        self.m2 = _grow(self.m2, 0.0)
        self.min = _grow(self.min, np.inf)
        self.max = _grow(self.max, -np.inf)
        self.labels = labels[order]

    def update(self, labels: np.ndarray, values: np.ndarray, valid: np.ndarray) -> None:
        """
        Add one tile: ``labels`` (2D), ``values`` (bands, 2D) and ``valid``
        (bands, 2D) where both the label and the band value are valid.
        """
        # pylint: disable=too-many-locals
        tile_labels = np.unique(labels[np.any(valid, axis=0)])
        if tile_labels.size == 0:
            return
        self._add_labels(tile_labels)
        for b_idx in range(self.n_bands):
            sel = valid[b_idx]
            lab = labels[sel]
            if lab.size == 0:
                continue
            vals = values[b_idx][sel].astype(np.float64, copy=False)
            uniq, inv = np.unique(lab, return_inverse=True)
            inv = inv.ravel()
            cnt = np.bincount(inv).astype(np.int64)
            mean_b = np.bincount(inv, weights=vals) / cnt
            m2_b = np.bincount(inv, weights=(vals - mean_b[inv]) ** 2)
            order = np.argsort(inv, kind="stable")
            starts = np.concatenate([[0], np.cumsum(cnt)[:-1]])
            sorted_vals = vals[order]
            min_b = np.minimum.reduceat(sorted_vals, starts)
            max_b = np.maximum.reduceat(sorted_vals, starts)

            rows = np.searchsorted(self.labels, uniq)
            n_a = self.count[rows, b_idx]
            mean_a = self.mean[rows, b_idx]
            n_tot = n_a + cnt
            delta = mean_b - mean_a
            frac = cnt / n_tot
            self.mean[rows, b_idx] = mean_a + delta * frac
            self.m2[rows, b_idx] = (
                self.m2[rows, b_idx] + m2_b + delta * delta * n_a * frac
            )
            self.count[rows, b_idx] = n_tot
            self.min[rows, b_idx] = np.minimum(self.min[rows, b_idx], min_b)
            self.max[rows, b_idx] = np.maximum(self.max[rows, b_idx], max_b)


class ZonalStatsSink(TileSink):
    """
    Sink computing, for every label (zone) of a single-band label node, the
    count of valid pixels and the mean, minimum, maximum and standard
    deviation of each band of a value node, in one streaming pass. The
    result is a :class:`pandas.DataFrame` indexed by label (``label``) with
    the columns ``count``, ``mean``, ``min``, ``max`` and ``std`` (with a
    single value band) or ``<band>_count``, ``<band>_mean``, ... (one set per
    value band, named after the band names, else ``b1``, ``b2``, ...).

    A pixel counts for a label when its label is finite and not the label
    no-data value, and for a band when that band's value is finite and not
    the value no-data value. Labels that never have a valid value in a band
    have a count of ``0`` and NaN statistics for it.

    :param labels: Single-band label (zone) node, e.g. a classification or a
                   segmentation (integer label values).
    :param values: Value node (one or more bands), on the same grid.
    :param no_data_val: No-data value of ``values`` (Default = ``None``: the
                        no-data value of ``values``' metadata).
    :param label_no_data_val: No-data value of ``labels`` (Default = ``None``:
                              the no-data value of ``labels``' metadata), e.g.
                              ``0`` to skip the background.
    :param ddof: Delta degrees of freedom of the standard deviation
                 (Default = ``0``).
    :param name: Optional sink name (the key of the result).

    .. code:: python

        from rsgispy.imageprocess import RasterSource, TileGraph, ZonalStatsSink
        from rsgispy.imagefilter.stats import median_filter_node

        zones = RasterSource("segments.kea")
        smooth = median_filter_node(RasterSource("s2.tif", bands=[4, 8]), 3)
        df = TileGraph(
            ZonalStatsSink(zones, smooth, label_no_data_val=0, name="zonal")
        ).run()["zonal"]
    """

    # pylint: disable=too-many-instance-attributes

    returns_result = True
    result_prefix = "zonal"

    def __init__(
        self,
        labels: TileNode,
        values: TileNode,
        *,
        no_data_val: Optional[float] = None,
        label_no_data_val: Optional[float] = None,
        ddof: int = 0,
        name: Optional[str] = None,
    ) -> None:
        # pylint: disable=too-many-arguments
        for node in (labels, values):
            if not isinstance(node, TileNode):
                raise rsgispy.RSGISPyException(
                    "ZonalStatsSink labels and values must be TileNode objects, "
                    f"got {type(node).__name__}."
                )
        self._stack = _ZonalStack(labels, values, name=f"{name or 'zonal'}_stack")
        super().__init__(self._stack, name=name)
        self.no_data_val = no_data_val
        self.label_no_data_val = label_no_data_val
        self.ddof = int(ddof)
        if self.ddof < 0:
            raise rsgispy.RSGISPyException(f"ddof must be >= 0, got {ddof}.")
        self.result: Any = None
        self._table: Optional[_ZoneTable] = None
        self._nd: Optional[float] = None
        self._label_nd: Optional[float] = None
        self._value_meta = NodeMeta()

    def open(
        self,
        grid: Any,
        meta: NodeMeta,
        *,
        stack: ExitStack,
        use_pbar: bool = True,
    ) -> None:
        del grid, meta, stack, use_pbar
        label_meta, value_meta = self._stack.input_metas
        self._value_meta = value_meta
        self._nd = self.no_data_val if self.no_data_val is not None else value_meta.nodata
        self._label_nd = (
            self.label_no_data_val
            if self.label_no_data_val is not None
            else label_meta.nodata
        )
        self._table = None
        self.result = None

    def write(self, block: np.ndarray, window: Window) -> None:
        labels = block[0]
        values = block[1:]
        if self._table is None:
            self._table = _ZoneTable(int(values.shape[0]))
        label_valid = _valid_mask(labels, self._label_nd)
        valid = _valid_mask(values, self._nd) & label_valid[np.newaxis]
        self._table.update(labels, values, valid)

    def close(self) -> None:
        import pandas as pd  # pylint: disable=import-outside-toplevel

        table = self._table
        n_bands = int(self._value_meta.n_bands or 1) if table is None else table.n_bands
        if table is None:
            table = _ZoneTable(n_bands)
        has = table.count > 0
        denom = np.maximum(table.count - self.ddof, 1)
        cols: Dict[str, np.ndarray] = {}
        names = _band_labels(self._value_meta, n_bands)
        for b_idx in range(n_bands):
            prefix = "" if n_bands == 1 else f"{names[b_idx]}_"
            h_b = has[:, b_idx]
            cols[f"{prefix}count"] = table.count[:, b_idx]
            cols[f"{prefix}mean"] = np.where(h_b, table.mean[:, b_idx], np.nan)
            cols[f"{prefix}min"] = np.where(h_b, table.min[:, b_idx], np.nan)
            cols[f"{prefix}max"] = np.where(h_b, table.max[:, b_idx], np.nan)
            cols[f"{prefix}std"] = np.where(
                table.count[:, b_idx] > self.ddof,
                np.sqrt(np.maximum(table.m2[:, b_idx] / denom[:, b_idx], 0.0)),
                np.nan,
            )
        labels: np.ndarray = table.labels
        if labels.size and np.all(labels == np.round(labels)):
            labels = labels.astype(np.int64)
        self.result = pd.DataFrame(cols, index=pd.Index(labels, name="label"))
        self._table = None

    def describe_params(self) -> str:
        nd = "from input" if self.no_data_val is None else repr(self.no_data_val)
        lnd = (
            "from input"
            if self.label_no_data_val is None
            else repr(self.label_no_data_val)
        )
        return f"no_data_val={nd} label_no_data_val={lnd}"
