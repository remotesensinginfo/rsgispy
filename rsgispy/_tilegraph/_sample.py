"""
:class:`SampleSink`: training samples (feature values of labelled pixels)
collected tile by tile while a :class:`~rsgispy.imageprocess.TileGraph` runs.
"""

from __future__ import annotations

from contextlib import ExitStack
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from rasterio.windows import Window

import rsgispy
from rsgispy._tilegraph._nodes import TileNode, TileSink
from rsgispy._tilegraph._stats import _ZonalStack, _band_labels, _valid_mask
from rsgispy._tilegraph._tensor import NodeMeta

#: Columns of the result before the feature columns, as in the sample tables
#: of :func:`rsgispy.classification.classdata.extract_class_training_samples`.
CLASS_NAME_COL = "class_name"
OUT_ID_COL = "out_cls_id"
INTERNAL_ID_COL = "internal_cls_id"
COORD_COLS = ("x", "y")

_MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)


def _splitmix64(vals: np.ndarray) -> np.ndarray:
    """SplitMix64 finaliser of ``uint64`` values (wrapping arithmetic)."""
    with np.errstate(over="ignore"):
        z = vals + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return (z ^ (z >> np.uint64(31))) & _MASK64


def pixel_keys(pxl_idx: np.ndarray, rnd_seed: int) -> np.ndarray:
    """
    Pseudo-random ``uint64`` sampling key of each pixel, a function only of
    its index in the image (``row * width + col``) and ``rnd_seed``, so a
    selection by key does not depend on the block size or ``n_jobs``.
    """
    seed_mix = _splitmix64(np.array([int(rnd_seed) & 0xFFFFFFFFFFFFFFFF], dtype=np.uint64))
    return _splitmix64(pxl_idx.astype(np.uint64) ^ seed_mix[0])


class _ClassSamples:
    """Kept samples of one class: keys, pixel indices, coordinates, features."""

    # pylint: disable=too-few-public-methods

    def __init__(self, n_feats: int, limit: Optional[int]) -> None:
        self.limit = limit
        self.parts: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self.n_feats = n_feats

    def add(self, keys: np.ndarray, idx: np.ndarray, feats: np.ndarray) -> None:
        """Add candidate pixels; with a limit, keep only the smallest keys."""
        self.parts.append((keys, idx, feats))
        if self.limit is not None:
            keys_all, idx_all, feats_all = self.merged()
            if keys_all.size > self.limit:
                keep = np.argpartition(keys_all, self.limit - 1)[: self.limit]
                keys_all, idx_all, feats_all = keys_all[keep], idx_all[keep], feats_all[keep]
            self.parts = [(keys_all, idx_all, feats_all)]

    def merged(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """All kept samples as ``(keys, pixel indices, features)``."""
        if not self.parts:
            return (
                np.empty(0, dtype=np.uint64),
                np.empty(0, dtype=np.int64),
                np.empty((0, self.n_feats), dtype=np.float64),
            )
        if len(self.parts) == 1:
            return self.parts[0]
        return (
            np.concatenate([p[0] for p in self.parts]),
            np.concatenate([p[1] for p in self.parts]),
            np.concatenate([p[2] for p in self.parts]),
        )


class SampleSink(TileSink):
    """
    Sink collecting the feature values of labelled pixels while the graph
    runs, e.g. to train a classifier on features computed in the same graph
    without writing them. The result is a pandas ``DataFrame`` in the
    layout of the classification sample tables
    (:func:`~rsgispy.classification.classdata.extract_class_training_samples`),
    so it works with :func:`~rsgispy.classification.classdata.write_cls_samples`,
    :func:`~rsgispy.classification.classdata.split_class_samples` and
    scikit-learn. Its columns are:

    * ``class_name`` — the name from ``class_names``, else the label as text;
    * ``out_cls_id`` — the label (class ID);
    * ``internal_cls_id`` — consecutive 0-based class index (in label order);
    * ``x`` and ``y`` — the pixel-centre coordinates (with ``include_xy``);
    * one column per feature band, named after the band names (else ``b1``,
      ``b2``, ...). Give the feature node unique band names, e.g. with
      ``StackBands(..., band_names=...)`` or
      :class:`~rsgispy.imageprocess.SetBandNames`; repeated names raise an
      error.

    A pixel is sampled when its label is finite and not the label no-data
    value, and every feature band is finite and not the feature no-data
    value (the rule of :class:`ZonalStatsSink` and
    :func:`~rsgispy.ml.predict.predict_raster_node`). The rows are ordered by
    class, then by position in the image.

    With ``max_per_class``, at most that many pixels are kept per class: the
    pixels with the smallest pseudo-random keys, where each key depends only
    on the pixel's position and ``rnd_seed``. The sample is therefore the
    same for any block size and any ``n_jobs``, and memory use is bounded by
    the number of samples kept. Without ``max_per_class``, every labelled
    pixel is kept, all in memory.

    :param features: Feature node (one or more bands).
    :param labels: Single-band label node on the same grid: integer class
                   IDs (e.g. rasterised training polygons, with ``0`` or the
                   no-data value for unlabelled pixels) or a 0/1 mask (pass
                   ``label_no_data_val=0`` to sample the 1s).
    :param max_per_class: Maximum number of samples per class, as one number
                          for every class or ``{label: n}`` (classes missing
                          from the mapping are not limited) (Default = ``None``:
                          all pixels).
    :param class_names: Optional ``{label: name}`` for the ``class_name``
                        column.
    :param no_data_val: No-data value of ``features`` (Default = ``None``:
                        that of ``features``' metadata).
    :param label_no_data_val: No-data value of ``labels`` (Default = ``None``:
                              that of ``labels``' metadata).
    :param rnd_seed: Seed of the sampling keys (Default = ``42``).
    :param include_xy: Add the ``x`` and ``y`` columns (Default = ``True``).
    :param name: Optional sink name (the key of the result).

    .. code:: python

        from sklearn.ensemble import RandomForestClassifier
        from rsgispy.imageprocess import RasterSource, SampleSink, TileGraph
        from rsgispy.imagecalc.calcindices import calc_indices_node

        s2 = RasterSource("s2.tif")
        feats = calc_indices_node(s2, {3: "Red", 4: "NIR", 2: "Green"},
                                  ["NDVI", "NDWI"])
        samples = TileGraph(
            SampleSink(feats, RasterSource("train_labels.tif"),
                       max_per_class=5000, name="samples")
        ).run()["samples"]
        clf = RandomForestClassifier().fit(
            samples[["NDVI", "NDWI"]], samples["out_cls_id"]
        )
    """

    # pylint: disable=too-many-instance-attributes

    returns_result = True
    result_prefix = "samples"

    def __init__(
        self,
        features: TileNode,
        labels: TileNode,
        *,
        max_per_class: Optional[Union[int, Mapping[Any, int]]] = None,
        class_names: Optional[Mapping[Any, str]] = None,
        no_data_val: Optional[float] = None,
        label_no_data_val: Optional[float] = None,
        rnd_seed: int = 42,
        include_xy: bool = True,
        name: Optional[str] = None,
    ) -> None:
        # pylint: disable=too-many-arguments
        for node in (features, labels):
            if not isinstance(node, TileNode):
                raise rsgispy.RSGISPyException(
                    "SampleSink features and labels must be TileNode objects, "
                    f"got {type(node).__name__}."
                )
        self._stack = _ZonalStack(
            labels, features, name=f"{name or 'samples'}_stack", owner="SampleSink"
        )
        super().__init__(self._stack, name=name)
        self.max_per_class = self._check_limits(max_per_class)
        self.class_names = dict(class_names) if class_names is not None else None
        self.no_data_val = no_data_val
        self.label_no_data_val = label_no_data_val
        self.rnd_seed = int(rnd_seed)
        self.include_xy = bool(include_xy)
        self.result: Any = None
        self._run: Optional[Dict[str, Any]] = None

    @staticmethod
    def _check_limits(
        max_per_class: Optional[Union[int, Mapping[Any, int]]]
    ) -> Optional[Union[int, Dict[float, int]]]:
        if max_per_class is None:
            return None
        if isinstance(max_per_class, Mapping):
            limits = {float(k): int(v) for k, v in max_per_class.items()}
            vals = list(limits.values())
        else:
            limits = int(max_per_class)
            vals = [limits]
        if any(v < 1 for v in vals):
            raise rsgispy.RSGISPyException(
                f"max_per_class must be >= 1, got {max_per_class}."
            )
        return limits

    def _limit(self, label: float) -> Optional[int]:
        if self.max_per_class is None or isinstance(self.max_per_class, int):
            return self.max_per_class
        return self.max_per_class.get(float(label))

    def _feature_names(self, meta: NodeMeta, n_feats: int) -> List[str]:
        names = _band_labels(meta, n_feats)
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise rsgispy.RSGISPyException(
                f"SampleSink: the feature bands have repeated names {dupes}; "
                "give them unique names, e.g. StackBands(..., band_names=[...]) "
                "or SetBandNames."
            )
        clash = set(names) & {CLASS_NAME_COL, OUT_ID_COL, INTERNAL_ID_COL, *COORD_COLS}
        if clash:
            raise rsgispy.RSGISPyException(
                f"SampleSink: feature band names {sorted(clash)} clash with the "
                "sample table columns; rename the bands (SetBandNames)."
            )
        return names

    def open(
        self,
        grid: Any,
        meta: NodeMeta,
        *,
        stack: ExitStack,
        use_pbar: bool = True,
    ) -> None:
        del meta, stack, use_pbar
        label_meta, feat_meta = self._stack.input_metas
        self._run = {
            "width": int(grid.width),
            "transform": grid.transform,
            "nd": self.no_data_val if self.no_data_val is not None else feat_meta.nodata,
            "label_nd": (
                self.label_no_data_val
                if self.label_no_data_val is not None
                else label_meta.nodata
            ),
            "feat_meta": feat_meta,
            "classes": {},
            "names": None,
        }
        self.result = None

    def write(self, block: np.ndarray, window: Window) -> None:
        # pylint: disable=too-many-locals
        run = self._run
        if run is None:
            raise rsgispy.RSGISPyException("SampleSink.write() called before open().")
        labels = block[0]
        feats = block[1:]
        n_feats = int(feats.shape[0])
        if run["names"] is None:
            run["names"] = self._feature_names(run["feat_meta"], n_feats)
        valid = _valid_mask(labels, run["label_nd"])
        valid &= np.all(_valid_mask(feats, run["nd"]), axis=0)
        if not np.any(valid):
            return
        rows, cols = np.nonzero(valid)
        pxl_idx = (rows + int(window.row_off)).astype(np.int64) * run["width"] + (
            cols + int(window.col_off)
        )
        lbl_vals = labels[rows, cols]
        feat_vals = feats[:, rows, cols].T.astype(np.float64, copy=False)
        keys = pixel_keys(pxl_idx, self.rnd_seed)
        for label in np.unique(lbl_vals):
            sel = lbl_vals == label
            store = run["classes"].get(float(label))
            if store is None:
                store = _ClassSamples(n_feats, self._limit(label))
                run["classes"][float(label)] = store
            store.add(keys[sel], pxl_idx[sel], feat_vals[sel])

    def close(self) -> None:
        # pylint: disable=too-many-locals
        import pandas as pd  # pylint: disable=import-outside-toplevel

        run = self._run
        if run is None:
            return
        names = run["names"] or _band_labels(
            run["feat_meta"], int(run["feat_meta"].n_bands or 0)
        )
        labels = sorted(run["classes"])
        int_labels = all(float(lbl).is_integer() for lbl in labels)
        parts: List[Dict[str, Any]] = []
        for internal_id, label in enumerate(labels):
            _, idx, feats = run["classes"][label].merged()
            order = np.argsort(idx, kind="stable")
            parts.append({"label": label, "internal": internal_id,
                          "idx": idx[order], "feats": feats[order]})
        n_rows = sum(int(p["idx"].size) for p in parts)
        idx_all = (
            np.concatenate([p["idx"] for p in parts]) if parts else np.empty(0, np.int64)
        )
        out_ids = np.concatenate(
            [np.full(p["idx"].size, p["label"]) for p in parts]
        ) if parts else np.empty(0)
        cols: Dict[str, Any] = {}
        cols[CLASS_NAME_COL] = np.concatenate(
            [np.full(p["idx"].size, self._class_name(p["label"], int_labels), dtype=object)
             for p in parts]
        ) if parts else np.empty(0, dtype=object)
        cols[OUT_ID_COL] = out_ids.astype(np.int64) if int_labels else out_ids
        cols[INTERNAL_ID_COL] = np.concatenate(
            [np.full(p["idx"].size, p["internal"], dtype=np.int64) for p in parts]
        ) if parts else np.empty(0, dtype=np.int64)
        if self.include_xy:
            x_vals, y_vals = self._pixel_xy(idx_all, run["width"], run["transform"])
            cols[COORD_COLS[0]] = x_vals
            cols[COORD_COLS[1]] = y_vals
        feat_all = (
            np.concatenate([p["feats"] for p in parts])
            if parts else np.empty((0, len(names)))
        )
        for f_idx, f_name in enumerate(names):
            cols[f_name] = feat_all[:, f_idx] if n_rows else np.empty(0)
        self.result = pd.DataFrame(cols)
        self._run = None

    def _class_name(self, label: float, int_labels: bool) -> str:
        key: Any = int(label) if int_labels else label
        if self.class_names is not None:
            for cand in (key, label, str(key)):
                if cand in self.class_names:
                    return str(self.class_names[cand])
        return str(key)

    @staticmethod
    def _pixel_xy(
        pxl_idx: np.ndarray, width: int, transform: Any
    ) -> Tuple[np.ndarray, np.ndarray]:
        rows = (pxl_idx // width).astype(np.float64) + 0.5
        cols = (pxl_idx % width).astype(np.float64) + 0.5
        if transform is None:
            return cols, rows
        x_vals = transform.a * cols + transform.b * rows + transform.c
        y_vals = transform.d * cols + transform.e * rows + transform.f
        return x_vals, y_vals

    def describe_params(self) -> str:
        nd = "from input" if self.no_data_val is None else repr(self.no_data_val)
        lnd = (
            "from input"
            if self.label_no_data_val is None
            else repr(self.label_no_data_val)
        )
        return (
            f"max_per_class={self.max_per_class!r} no_data_val={nd} "
            f"label_no_data_val={lnd} rnd_seed={self.rnd_seed}"
        )


def sample_columns(feature_names: Sequence[str], include_xy: bool = True) -> List[str]:
    """Column names of a :class:`SampleSink` result for ``feature_names``."""
    meta = [CLASS_NAME_COL, OUT_ID_COL, INTERNAL_ID_COL]
    return meta + (list(COORD_COLS) if include_xy else []) + list(feature_names)
