"""
Spatial cross-validation splitters and nested spatial cross-validation
(:class:`SpatialBlockKFold`, :class:`SpatialBufferCV`,
:class:`PolygonGroupKFold`, :func:`nested_spatial_cv`) designed to prevent
spatial autocorrelation leakage between training and validation folds.
"""

from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd

import rsgispy
from rsgispy.ml.scaling import FeatureScaler

__all__ = [
    "SpatialBlockKFold",
    "SpatialBufferCV",
    "PolygonGroupKFold",
    "NestedCVResult",
    "nested_spatial_cv",
]


def _validate_coords(
    coords: Any,
    n_samples: int,
) -> np.ndarray:
    """Validate and coerce ``coords`` into a finite ``(n_samples, 2)`` float64 array."""
    if coords is None:
        raise rsgispy.RSGISPyException(
            "Spatial cross-validation requires 2D sample coordinates 'coords' "
            "of shape (n_samples, 2)."
        )
    arr = np.asarray(coords, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] != n_samples:
        raise rsgispy.RSGISPyException(
            f"coords must have shape ({n_samples}, 2), got {arr.shape}."
        )
    if not np.all(np.isfinite(arr)):
        raise rsgispy.RSGISPyException("coords must contain only finite values.")
    return arr


def _simple_kmeans_2d(
    pts: np.ndarray,
    k: int,
    rnd_seed: Optional[int] = 42,
    max_iter: int = 50,
) -> np.ndarray:
    """Lightweight deterministic k-means++ clustering in 2D without extra deps."""
    n_pts = pts.shape[0]
    if k >= n_pts:
        return np.arange(n_pts, dtype=np.int64) % k
    rng = np.random.default_rng(rnd_seed)

    # K-means++ seeding for well-separated spatial clusters
    centres = np.empty((k, 2), dtype=np.float64)
    centres[0] = pts[int(rng.integers(0, n_pts))]
    min_d2 = np.sum((pts - centres[0]) ** 2, axis=1)
    for c_idx in range(1, k):
        total = float(np.sum(min_d2))
        if total <= 0.0:
            pick = int(rng.integers(0, n_pts))
        else:
            probs = min_d2 / total
            pick = int(rng.choice(n_pts, p=probs))
        centres[c_idx] = pts[pick]
        d2 = np.sum((pts - centres[c_idx]) ** 2, axis=1)
        min_d2 = np.minimum(min_d2, d2)

    labels = np.zeros(n_pts, dtype=np.int64)
    for _ in range(max_iter):
        dists = np.sum((pts[:, None, :] - centres[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(dists, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c_idx in range(k):
            members = pts[labels == c_idx]
            if members.shape[0] > 0:
                centres[c_idx] = members.mean(axis=0)
    return labels


def _assign_blocks_to_folds(
    block_ids: np.ndarray,
    n_splits: int,
    y: Optional[np.ndarray],
    stratify: bool,
    shuffle: bool,
    rnd_seed: Optional[int],
) -> np.ndarray:
    """Assign each spatial block ID to one of ``n_splits`` balanced folds."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    unique_blocks = np.unique(block_ids)
    n_blocks = unique_blocks.size
    if n_blocks < n_splits:
        raise rsgispy.RSGISPyException(
            f"Number of occupied spatial blocks ({n_blocks}) is smaller than "
            f"n_splits ({n_splits}). Reduce block_size or n_splits."
        )

    rng = np.random.default_rng(rnd_seed)
    order = np.arange(n_blocks)
    if shuffle:
        rng.shuffle(order)
    ordered_blocks = unique_blocks[order]

    # Sort blocks descending by size so greedy assignment balances fold counts
    sizes = np.array([int(np.sum(block_ids == b)) for b in ordered_blocks])
    size_order = np.argsort(-sizes, kind="mergesort")
    ordered_blocks = ordered_blocks[size_order]
    sizes = sizes[size_order]

    block_to_fold: Dict[int, int] = {}
    if stratify and y is not None:
        y_arr = np.asarray(y)
        classes = np.unique(y_arr)
        n_cls = classes.size
        cls_to_i = {c: i for i, c in enumerate(classes)}
        y_idx = np.array([cls_to_i[val] for val in y_arr], dtype=np.int64)
        blk_hists = np.zeros((n_blocks, n_cls), dtype=np.int64)
        for b_idx, blk_id in enumerate(ordered_blocks):
            blk_hists[b_idx] = np.bincount(
                y_idx[block_ids == blk_id], minlength=n_cls
            )
        cls_totals = blk_hists.sum(axis=0).astype(np.float64)
        # Classes from rarest to most common; blocks holding the rarest class
        # are assigned first (most rare-class samples first, then by size), so
        # minority-class blocks are spread over the folds before the majority
        # class fills them.
        rare_order = np.argsort(cls_totals, kind="mergesort")
        sort_keys = [sizes] + [blk_hists[:, c] for c in rare_order[::-1]]
        blk_order = np.lexsort([-k for k in sort_keys])
        fold_cls_counts = np.zeros((n_splits, n_cls), dtype=np.float64)
        fold_totals = np.zeros(n_splits, dtype=np.int64)
        n_total = float(max(1, int(sizes.sum())))
        for b_idx in blk_order:
            blk_id = ordered_blocks[b_idx]
            blk_hist = blk_hists[b_idx]
            # A fold with none of a class the block contains is preferred
            # (rarest class first), so every fold gets each class whenever
            # at least n_splits blocks contain it.
            cand = np.arange(n_splits)
            for c_idx in rare_order:
                if blk_hist[c_idx] == 0:
                    continue
                empty = cand[fold_cls_counts[cand, c_idx] == 0]
                if empty.size > 0:
                    cand = empty
            # Among candidates, minimise the variance across folds of each
            # class's share of its total (so the majority class does not
            # dominate), tie-breaking on the smallest fold.
            best_fold = int(cand[0])
            best_cost = float("inf")
            for f_idx in cand:
                trial = fold_cls_counts.copy()
                trial[f_idx] += blk_hist
                cost = float(np.var(trial / cls_totals, axis=0).sum()) + 1e-9 * (
                    float(fold_totals[f_idx]) / n_total
                )
                if cost < best_cost:
                    best_cost = cost
                    best_fold = int(f_idx)
            block_to_fold[int(blk_id)] = best_fold
            fold_cls_counts[best_fold] += blk_hist
            fold_totals[best_fold] += int(sizes[b_idx])
    else:
        fold_totals = np.zeros(n_splits, dtype=np.int64)
        for blk_id, blk_sz in zip(ordered_blocks, sizes):
            f_idx = int(np.argmin(fold_totals))
            block_to_fold[int(blk_id)] = f_idx
            fold_totals[f_idx] += int(blk_sz)

    return np.array([block_to_fold[int(b)] for b in block_ids], dtype=np.int64)


class SpatialBlockKFold:
    """
    Spatial block k-fold cross-validator that partitions samples into spatially
    disjoint blocks so nearby observations never appear in both train and
    validation folds.

    :param n_splits: Number of cross-validation folds (Default = ``5``).
    :param block_size: Side length of square spatial blocks in coordinate units
                       (for ``method='grid'`` or ``'checkerboard'``). When
                       ``None`` (Default), chosen automatically from sample
                       extent and ``n_splits``.
    :param method: Spatial partitioning strategy: ``'grid'`` (Default),
                   ``'checkerboard'``, or ``'kmeans'``.
    :param stratify: If ``True``, assign spatial blocks to folds to balance
                     class frequencies in ``y`` (Default = ``False``). Each
                     class is balanced as a share of its own total, so a rare
                     class (e.g. presences among many background points) is
                     spread too: blocks containing the rarest class are
                     assigned first and every fold gets samples of a class
                     whenever at least ``n_splits`` blocks contain it.
    :param shuffle: Randomly permute blocks before fold assignment
                    (Default = ``True``).
    :param rnd_seed: Optional random seed (Default = ``42``).

    .. code:: python

        import numpy as np
        from rsgispy.ml.cv import SpatialBlockKFold

        coords = np.random.default_rng(0).uniform(0, 1000, size=(200, 2))
        cv = SpatialBlockKFold(n_splits=5, block_size=200.0, method="grid")
        for train_idx, val_idx in cv.split(coords, coords=coords):
            pass
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments

    def __init__(
        self,
        n_splits: int = 5,
        block_size: Optional[float] = None,
        method: str = "grid",
        stratify: bool = False,
        shuffle: bool = True,
        rnd_seed: Optional[int] = 42,
    ) -> None:
        if int(n_splits) < 2:
            raise rsgispy.RSGISPyException(
                f"n_splits must be >= 2, got {n_splits}."
            )
        if block_size is not None and float(block_size) <= 0.0:
            raise rsgispy.RSGISPyException(
                f"block_size must be > 0, got {block_size}."
            )
        mthd = str(method).strip().lower()
        if mthd not in ("grid", "checkerboard", "kmeans"):
            raise rsgispy.RSGISPyException(
                f"Unsupported SpatialBlockKFold method '{method}'. "
                "Expected 'grid', 'checkerboard', or 'kmeans'."
            )
        self.n_splits = int(n_splits)
        self.block_size = float(block_size) if block_size is not None else None
        self.method = mthd
        self.stratify = bool(stratify)
        self.shuffle = bool(shuffle)
        self.rnd_seed = rnd_seed

    def get_n_splits(
        self,
        X: Any = None,
        y: Any = None,
        groups: Any = None,
    ) -> int:
        """Return the number of splitting iterations."""
        _ = (X, y, groups)
        return self.n_splits

    def _compute_fold_ids(
        self,
        coords: np.ndarray,
        y: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if self.method == "kmeans":
            return _simple_kmeans_2d(
                coords, k=self.n_splits, rnd_seed=self.rnd_seed
            )

        min_xy = coords.min(axis=0)
        span_xy = np.maximum(coords.max(axis=0) - min_xy, 1e-9)
        if self.block_size is None:
            # Target ~3*n_splits blocks across the bounding box
            target_side = max(int(np.ceil(np.sqrt(self.n_splits * 3.0))), 2)
            blk_sz = float(max(span_xy[0], span_xy[1])) / float(target_side)
        else:
            blk_sz = self.block_size

        col_idx = np.floor((coords[:, 0] - min_xy[0]) / blk_sz).astype(np.int64)
        row_idx = np.floor((coords[:, 1] - min_xy[1]) / blk_sz).astype(np.int64)

        if self.method == "checkerboard":
            fold_ids = (row_idx + col_idx) % self.n_splits
            if np.unique(fold_ids).size < self.n_splits:
                raise rsgispy.RSGISPyException(
                    "Checkerboard partitioning produced fewer folds than "
                    f"n_splits={self.n_splits}. Reduce block_size or n_splits."
                )
            return fold_ids

        n_cols = int(col_idx.max()) + 1
        block_ids = row_idx * n_cols + col_idx
        return _assign_blocks_to_folds(
            block_ids=block_ids,
            n_splits=self.n_splits,
            y=y,
            stratify=self.stratify,
            shuffle=self.shuffle,
            rnd_seed=self.rnd_seed,
        )

    def split(
        self,
        X: Any,
        y: Optional[Any] = None,
        groups: Optional[Any] = None,
        coords: Optional[Any] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate ``(train_indices, val_indices)`` arrays for each spatial fold.

        Coordinates may be supplied via ``coords`` or via ``groups`` (when
        ``coords`` is ``None`` and ``groups`` is a 2D ``(n_samples, 2)`` array
        for scikit-learn compatibility).
        """
        n_samples = len(X)
        eff_coords = coords if coords is not None else groups
        c_arr = _validate_coords(eff_coords, n_samples)
        y_arr = np.asarray(y) if y is not None else None
        fold_ids = self._compute_fold_ids(c_arr, y=y_arr)
        all_idx = np.arange(n_samples, dtype=np.intp)
        for fold in range(self.n_splits):
            val_mask = fold_ids == fold
            train_idx = all_idx[~val_mask]
            val_idx = all_idx[val_mask]
            if train_idx.size == 0 or val_idx.size == 0:
                continue
            yield train_idx, val_idx


class SpatialBufferCV:
    """
    Buffered spatial cross-validator that creates ``n_splits`` spatial test
    folds and excludes every training sample within Euclidean distance
    ``<= buffer_dist`` of any test sample in that fold.

    :param n_splits: Number of folds (Default = ``5``).
    :param buffer_dist: Spatial exclusion radius in coordinate units
                        (must be ``>= 0``; Default = ``100.0``).
    :param method: Base spatial fold partitioning: ``'kmeans'`` (Default) or
                   ``'grid'``.
    :param block_size: Optional grid block size when ``method='grid'``.
    :param rnd_seed: Optional random seed (Default = ``42``).
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments

    def __init__(
        self,
        n_splits: int = 5,
        buffer_dist: float = 100.0,
        method: str = "kmeans",
        block_size: Optional[float] = None,
        rnd_seed: Optional[int] = 42,
    ) -> None:
        if int(n_splits) < 2:
            raise rsgispy.RSGISPyException(
                f"n_splits must be >= 2, got {n_splits}."
            )
        if float(buffer_dist) < 0.0:
            raise rsgispy.RSGISPyException(
                f"buffer_dist must be >= 0, got {buffer_dist}."
            )
        self.n_splits = int(n_splits)
        self.buffer_dist = float(buffer_dist)
        self.method = str(method).strip().lower()
        self.block_size = block_size
        self.rnd_seed = rnd_seed

    def get_n_splits(
        self,
        X: Any = None,
        y: Any = None,
        groups: Any = None,
    ) -> int:
        """Return the number of folds."""
        _ = (X, y, groups)
        return self.n_splits

    def split(
        self,
        X: Any,
        y: Optional[Any] = None,
        groups: Optional[Any] = None,
        coords: Optional[Any] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Yield ``(train_idx, val_idx)`` with buffered exclusion around ``val_idx``."""
        n_samples = len(X)
        eff_coords = coords if coords is not None else groups
        c_arr = _validate_coords(eff_coords, n_samples)

        base_cv = SpatialBlockKFold(
            n_splits=self.n_splits,
            block_size=self.block_size,
            method=self.method,
            rnd_seed=self.rnd_seed,
        )
        buf_sq = self.buffer_dist * self.buffer_dist
        for raw_train_idx, val_idx in base_cv.split(X, y=y, coords=c_arr):
            if self.buffer_dist == 0.0:
                yield raw_train_idx, val_idx
                continue
            val_pts = c_arr[val_idx]
            tr_pts = c_arr[raw_train_idx]
            # Chunked squared-distance check to bound peak memory
            keep = np.ones(raw_train_idx.size, dtype=bool)
            chunk = 1024
            for start in range(0, raw_train_idx.size, chunk):
                end = min(start + chunk, raw_train_idx.size)
                diff = tr_pts[start:end, None, :] - val_pts[None, :, :]
                min_d2 = np.min(np.sum(diff * diff, axis=2), axis=1)
                keep[start:end] = min_d2 > buf_sq
            filt_train = raw_train_idx[keep]
            if filt_train.size == 0:
                raise rsgispy.RSGISPyException(
                    f"SpatialBufferCV(buffer_dist={self.buffer_dist}) excluded "
                    "all training samples in a fold. Reduce buffer_dist."
                )
            yield filt_train, val_idx


class PolygonGroupKFold:
    """
    Group k-fold splitter ensuring all pixels extracted from the same training
    polygon (``groups``) stay strictly together in either train or validation.

    :param n_splits: Number of folds (Default = ``5``).
    :param stratify: Balance class distribution across polygon groups
                     (Default = ``False``).
    :param shuffle: Shuffle polygon groups prior to fold assignment
                    (Default = ``True``).
    :param rnd_seed: Optional random seed (Default = ``42``).
    """

    def __init__(
        self,
        n_splits: int = 5,
        stratify: bool = False,
        shuffle: bool = True,
        rnd_seed: Optional[int] = 42,
    ) -> None:
        if int(n_splits) < 2:
            raise rsgispy.RSGISPyException(
                f"n_splits must be >= 2, got {n_splits}."
            )
        self.n_splits = int(n_splits)
        self.stratify = bool(stratify)
        self.shuffle = bool(shuffle)
        self.rnd_seed = rnd_seed

    def get_n_splits(
        self,
        X: Any = None,
        y: Any = None,
        groups: Any = None,
    ) -> int:
        """Return the number of folds."""
        _ = (X, y, groups)
        return self.n_splits

    def split(
        self,
        X: Any,
        y: Optional[Any] = None,
        groups: Optional[Any] = None,
        coords: Optional[Any] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Yield ``(train_idx, val_idx)`` keeping every group ID intact."""
        _ = coords
        n_samples = len(X)
        if groups is None:
            raise rsgispy.RSGISPyException(
                "PolygonGroupKFold.split requires 'groups' (e.g. polygon IDs)."
            )
        grp_arr = np.asarray(groups)
        if grp_arr.shape[0] != n_samples:
            raise rsgispy.RSGISPyException(
                f"groups length ({grp_arr.shape[0]}) must match X ({n_samples})."
            )
        _, inv_ids = np.unique(grp_arr, return_inverse=True)
        y_arr = np.asarray(y) if y is not None else None
        fold_ids = _assign_blocks_to_folds(
            block_ids=inv_ids,
            n_splits=self.n_splits,
            y=y_arr,
            stratify=self.stratify,
            shuffle=self.shuffle,
            rnd_seed=self.rnd_seed,
        )
        all_idx = np.arange(n_samples, dtype=np.intp)
        for fold in range(self.n_splits):
            val_mask = fold_ids == fold
            yield all_idx[~val_mask], all_idx[val_mask]


@dataclass
class NestedCVResult:
    """Structured results returned by :func:`nested_spatial_cv`."""

    fold_scores: List[float]
    mean_score: float
    std_score: float
    oof_predictions: np.ndarray
    oof_probabilities: Optional[np.ndarray] = None
    best_params_per_fold: List[Dict[str, Any]] = field(default_factory=list)
    fold_metrics: List[Dict[str, float]] = field(default_factory=list)
    overall_metrics: Dict[str, float] = field(default_factory=dict)


def _index_rows(data: Any, indices: np.ndarray) -> Any:
    if isinstance(data, pd.DataFrame):
        return data.iloc[indices]
    if isinstance(data, pd.Series):
        return data.iloc[indices]
    return np.asarray(data)[indices]


def _compute_task_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task: str,
) -> Dict[str, float]:
    """Compute core classification or regression summary metrics."""
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    if task == "regression":
        err = yp.astype(np.float64) - yt.astype(np.float64)
        rmse = float(np.sqrt(np.mean(err * err)))
        mae = float(np.mean(np.abs(err)))
        ss_res = float(np.sum(err * err))
        ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0.0 else 0.0
        return {"r2": r2, "rmse": rmse, "mae": mae}

    acc = float(np.mean(yt == yp))
    classes = np.unique(yt)
    f1s: List[float] = []
    for cls_val in classes:
        tp = float(np.sum((yt == cls_val) & (yp == cls_val)))
        fp = float(np.sum((yt != cls_val) & (yp == cls_val)))
        fn = float(np.sum((yt == cls_val) & (yp != cls_val)))
        denom = 2.0 * tp + fp + fn
        f1s.append((2.0 * tp / denom) if denom > 0.0 else 0.0)
    return {
        "accuracy": acc,
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
    }


def _evaluate_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task: str,
    scoring: Optional[Union[str, Callable[[np.ndarray, np.ndarray], float]]],
) -> float:
    if callable(scoring):
        return float(scoring(y_true, y_pred))
    metrics = _compute_task_metrics(y_true, y_pred, task)
    if scoring is None:
        return metrics["r2"] if task == "regression" else metrics["accuracy"]
    key = str(scoring).strip().lower()
    if key in ("r2", "r2_score"):
        return metrics["r2"]
    if key in ("neg_rmse", "neg_root_mean_squared_error"):
        return -metrics["rmse"]
    if key in ("rmse",):
        return metrics["rmse"]
    if key in ("neg_mae", "neg_mean_absolute_error"):
        return -metrics["mae"]
    if key in ("mae",):
        return metrics["mae"]
    if key in ("accuracy", "acc"):
        return metrics["accuracy"]
    if key in ("f1", "f1_macro", "macro_f1"):
        return metrics["macro_f1"]
    raise rsgispy.RSGISPyException(f"Unsupported scoring metric: {scoring!r}.")


_MINIMISE_METRICS = frozenset({"rmse", "mae"})
_MAXIMISE_METRICS = frozenset(
    {
        "r2",
        "r2_score",
        "neg_rmse",
        "neg_root_mean_squared_error",
        "neg_mae",
        "neg_mean_absolute_error",
        "accuracy",
        "acc",
        "f1",
        "f1_macro",
        "macro_f1",
    }
)


def _resolve_direction(
    direction: Optional[str],
    scoring: Optional[Union[str, Callable[[np.ndarray, np.ndarray], float]]],
) -> str:
    """
    Resolve the Optuna direction for ``scoring``.

    Error metrics (``'rmse'``, ``'mae'``) are minimised and every other named
    metric (including the default and the ``'neg_*'`` forms) is maximised. When
    ``direction`` is ``None`` it is derived from the metric (``'maximize'`` for
    a callable). An explicit ``direction`` that contradicts a named metric
    raises, because Optuna would then select the worst model.
    """
    dir_norm = None if direction is None else str(direction).strip().lower()
    if dir_norm is not None and dir_norm not in ("maximize", "minimize"):
        raise rsgispy.RSGISPyException(
            f"direction must be 'maximize' or 'minimize', got {direction!r}."
        )
    if callable(scoring):
        return dir_norm if dir_norm is not None else "maximize"
    key = None if scoring is None else str(scoring).strip().lower()
    if key in _MINIMISE_METRICS:
        expected = "minimize"
    elif key is None or key in _MAXIMISE_METRICS:
        expected = "maximize"
    else:
        raise rsgispy.RSGISPyException(f"Unsupported scoring metric: {scoring!r}.")
    if dir_norm is not None and dir_norm != expected:
        raise rsgispy.RSGISPyException(
            f"direction={direction!r} contradicts scoring={scoring!r}, which "
            f"must be optimised with direction='{expected}'. Leave direction "
            "as None to derive it from the metric."
        )
    return expected


def _int_kfold_splits(
    n_splits: int,
    y_arr: np.ndarray,
    task: str,
    rnd_seed: Optional[int],
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Shuffled K-fold splits for an integer ``cv`` without coordinates or groups.

    Rows are shuffled with ``rnd_seed`` so the folds do not depend on the row
    order (sample tables are usually stored class by class). Classification
    folds are stratified: the shuffled rows of each class are dealt to the
    folds in turn, so every class is represented in every fold as evenly as
    its sample count allows.
    """
    n_s = int(y_arr.shape[0])
    if n_splits < 2:
        raise rsgispy.RSGISPyException(f"cv must be at least 2 folds, got {n_splits}.")
    if n_s < n_splits:
        raise rsgispy.RSGISPyException(
            f"cv={n_splits} exceeds the number of samples ({n_s})."
        )
    rng = np.random.default_rng(rnd_seed)
    fold_of = np.empty(n_s, dtype=np.intp)
    if task == "classification":
        offset = 0
        for cls_val in np.unique(y_arr):
            cls_idx = rng.permutation(np.flatnonzero(y_arr == cls_val))
            fold_of[cls_idx] = (offset + np.arange(cls_idx.size)) % n_splits
            offset += cls_idx.size
    else:
        perm = rng.permutation(n_s)
        fold_of[perm] = np.arange(n_s) % n_splits
    out: List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(n_splits):
        out.append((np.flatnonzero(fold_of != i), np.flatnonzero(fold_of == i)))
    return out


def _iter_cv_splits(
    cv: Any,
    x_data: Any,
    y_arr: np.ndarray,
    coords: Optional[np.ndarray],
    groups: Optional[np.ndarray],
    *,
    task: str = "classification",
    rnd_seed: Optional[int] = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Invoke ``cv.split`` supporting both spatial splitters and sklearn splitters.

    An integer ``cv`` gives :class:`SpatialBlockKFold` folds when ``coords``
    are given, :class:`PolygonGroupKFold` folds when ``groups`` are given, and
    otherwise shuffled (stratified for classification) K-fold splits seeded
    with ``rnd_seed``.
    """
    # pylint: disable=too-many-arguments
    if isinstance(cv, (SpatialBlockKFold, SpatialBufferCV, PolygonGroupKFold)):
        return list(cv.split(x_data, y=y_arr, groups=groups, coords=coords))
    if isinstance(cv, (int, np.integer)) and not isinstance(cv, bool):
        n_splits = int(cv)
        if coords is not None:
            return list(
                SpatialBlockKFold(n_splits=n_splits, rnd_seed=rnd_seed).split(
                    x_data, y=y_arr, coords=coords
                )
            )
        if groups is not None:
            return list(
                PolygonGroupKFold(n_splits=n_splits, rnd_seed=rnd_seed).split(
                    x_data, y=y_arr, groups=groups
                )
            )
        return _int_kfold_splits(n_splits, y_arr, task, rnd_seed)
    return list(cv.split(x_data, y_arr, groups))


def nested_spatial_cv(
    estimator_factory: Callable[[Mapping[str, Any]], Any],
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    *,
    task: str = "classification",
    coords: Optional[Any] = None,
    groups: Optional[Any] = None,
    outer_cv: Any = None,
    inner_cv: Any = None,
    param_space: Optional[Any] = None,
    scaler: Optional[FeatureScaler] = None,
    scoring: Optional[Union[str, Callable[[np.ndarray, np.ndarray], float]]] = None,
    n_trials: int = 20,
    timeout: Optional[float] = None,
    direction: Optional[str] = None,
    rnd_seed: Optional[int] = 42,
    use_pbar: bool = True,
) -> NestedCVResult:
    """
    Run nested spatial cross-validation with fold-local feature scaling and
    optional inner-loop Optuna hyperparameter tuning.

    Every outer fold fits ``scaler`` strictly on outer-train (and inside each
    inner fold during hyperparameter tuning) to prevent both spatial and
    scaling data leakage.

    :param estimator_factory: Callable ``(params_dict) -> unfitted_estimator``.
    :param X: Feature matrix or DataFrame of shape ``(n_samples, n_features)``.
    :param y: Target array or Series of length ``n_samples``.
    :param task: ``'classification'`` (Default) or ``'regression'``.
    :param coords: Optional ``(n_samples, 2)`` spatial coordinates array.
    :param groups: Optional 1D group/polygon IDs array.
    :param outer_cv: Outer CV splitter (defaults to :class:`SpatialBlockKFold`
                     when ``coords`` is given, :class:`PolygonGroupKFold` when
                     ``groups`` is given, or shuffled 5-fold CV, stratified
                     for classification and seeded with ``rnd_seed``,
                     otherwise). An integer gives the same choice of splitter
                     with that number of folds.
    :param inner_cv: Inner CV splitter for Optuna tuning (defaults to 3-fold
                     spatial/group CV).
    :param param_space: Optional Optuna hyperparameter space dict or callable
                        (when ``None``, skips inner tuning and calls
                        ``estimator_factory({})``).
    :param scaler: Optional :class:`~rsgispy.ml.scaling.FeatureScaler` cloned
                   and fitted strictly inside each training fold.
    :param scoring: Scoring name (``'accuracy'``, ``'macro_f1'``, ``'r2'``,
                    ``'rmse'``, ``'mae'``, ``'neg_rmse'``, ``'neg_mae'``) or
                    callable ``(y_true, y_pred) -> float``.
    :param n_trials: Optuna trials per outer fold when ``param_space`` is given
                     (Default = ``20``).
    :param timeout: Optional Optuna timeout in seconds per outer fold.
    :param direction: Optuna direction ``'maximize'`` or ``'minimize'``. The
                      default (``None``) derives it from ``scoring``: error
                      metrics (``'rmse'``, ``'mae'``) are minimised and all
                      others maximised. A direction that contradicts a named
                      metric raises :class:`rsgispy.RSGISPyException`.
    :param rnd_seed: Random seed (Default = ``42``).
    :param use_pbar: Display Optuna/tqdm progress bars (Default = ``True``).
    :return: :class:`NestedCVResult` with out-of-fold predictions and metrics.
    """
    # pylint: disable=too-many-arguments,too-many-locals,too-many-branches
    # pylint: disable=too-many-statements
    task_norm = str(task).strip().lower()
    if task_norm not in ("classification", "regression"):
        raise rsgispy.RSGISPyException(
            f"task must be 'classification' or 'regression', got {task!r}."
        )
    eff_direction = _resolve_direction(direction, scoring)
    y_arr = np.asarray(y)
    n_samples = y_arr.shape[0]
    c_arr = _validate_coords(coords, n_samples) if coords is not None else None
    g_arr = np.asarray(groups) if groups is not None else None

    if outer_cv is None:
        if c_arr is not None:
            eff_outer: Any = SpatialBlockKFold(
                n_splits=5, rnd_seed=rnd_seed
            )
        elif g_arr is not None:
            eff_outer = PolygonGroupKFold(n_splits=5, rnd_seed=rnd_seed)
        else:
            eff_outer = 5
    else:
        eff_outer = outer_cv

    outer_splits = _iter_cv_splits(
        eff_outer, X, y_arr, c_arr, g_arr, task=task_norm, rnd_seed=rnd_seed
    )
    if not outer_splits:
        raise rsgispy.RSGISPyException("outer_cv produced zero valid splits.")

    oof_preds = np.full(
        n_samples,
        np.nan if task_norm == "regression" else 0,
        dtype=np.float64 if task_norm == "regression" else y_arr.dtype,
    )
    oof_probs: Optional[np.ndarray] = None
    global_classes: Optional[np.ndarray] = None
    cls_to_col: Dict[Any, int] = {}
    if task_norm == "classification":
        global_classes = np.unique(y_arr)
        cls_to_col = {
            cls_val: idx for idx, cls_val in enumerate(global_classes)
        }
    covered = np.zeros(n_samples, dtype=bool)
    fold_scores: List[float] = []
    fold_metrics: List[Dict[str, float]] = []
    best_params_list: List[Dict[str, Any]] = []

    for fold_idx, (tr_idx, val_idx) in enumerate(outer_splits):
        x_tr = _index_rows(X, tr_idx)
        y_tr = y_arr[tr_idx]
        x_val = _index_rows(X, val_idx)
        y_val = y_arr[val_idx]
        c_tr = c_arr[tr_idx] if c_arr is not None else None
        g_tr = g_arr[tr_idx] if g_arr is not None else None

        if param_space is not None:
            # pylint: disable=import-outside-toplevel
            from rsgispy.ml.tune import tune_estimator

            eff_inner = inner_cv
            if eff_inner is None:
                if c_tr is not None:
                    eff_inner = SpatialBlockKFold(
                        n_splits=3,
                        method="kmeans",
                        rnd_seed=(
                            None if rnd_seed is None else int(rnd_seed) + fold_idx
                        ),
                    )
                elif g_tr is not None:
                    eff_inner = PolygonGroupKFold(
                        n_splits=3,
                        rnd_seed=(
                            None if rnd_seed is None else int(rnd_seed) + fold_idx
                        ),
                    )
                else:
                    eff_inner = 3
            tune_res = tune_estimator(
                estimator_factory=estimator_factory,
                X=x_tr,
                y=y_tr,
                param_space=param_space,
                task=task_norm,
                cv=eff_inner,
                coords=c_tr,
                groups=g_tr,
                scaler=scaler,
                scoring=scoring,
                n_trials=n_trials,
                timeout=timeout,
                direction=eff_direction,
                refit=False,
                rnd_seed=(
                    None if rnd_seed is None else int(rnd_seed) + fold_idx
                ),
                use_pbar=use_pbar,
            )
            best_params = dict(tune_res.best_params)
        else:
            best_params = {}
        best_params_list.append(best_params)

        if scaler is not None:
            fold_scaler = scaler.clone_unfitted()
            x_tr_fit = fold_scaler.fit_transform(x_tr)
            x_val_fit = fold_scaler.transform(x_val)
        else:
            x_tr_fit = x_tr
            x_val_fit = x_val

        est = estimator_factory(best_params)
        est.fit(x_tr_fit, y_tr)
        preds = np.asarray(est.predict(x_val_fit)).ravel()
        oof_preds[val_idx] = preds
        covered[val_idx] = True

        if task_norm == "classification" and hasattr(est, "predict_proba"):
            probs = np.asarray(est.predict_proba(x_val_fit), dtype=np.float64)
            if probs.ndim == 1:
                probs = probs[:, None]
            if oof_probs is None and global_classes is not None:
                oof_probs = np.full(
                    (n_samples, global_classes.size), np.nan, dtype=np.float64
                )
            if oof_probs is not None:
                est_classes = getattr(est, "classes_", None)
                if est_classes is not None:
                    oof_probs[val_idx] = 0.0
                    for local_col, cls_val in enumerate(est_classes):
                        if (
                            cls_val in cls_to_col
                            and local_col < probs.shape[1]
                        ):
                            oof_probs[val_idx, cls_to_col[cls_val]] = probs[
                                :, local_col
                            ]
                elif probs.shape[1] == oof_probs.shape[1]:
                    oof_probs[val_idx] = probs

        fold_scores.append(_evaluate_score(y_val, preds, task_norm, scoring))
        fold_metrics.append(_compute_task_metrics(y_val, preds, task_norm))

    overall = _compute_task_metrics(y_arr[covered], oof_preds[covered], task_norm)
    return NestedCVResult(
        fold_scores=fold_scores,
        mean_score=float(np.mean(fold_scores)),
        std_score=float(np.std(fold_scores)),
        oof_predictions=oof_preds,
        oof_probabilities=oof_probs,
        best_params_per_fold=best_params_list,
        fold_metrics=fold_metrics,
        overall_metrics=overall,
    )
