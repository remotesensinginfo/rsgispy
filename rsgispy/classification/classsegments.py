"""
The classification.classsegments module provides object-based (segment)
classification: the image segments from :mod:`rsgispy.segmentation` (e.g.,
:func:`~rsgispy.segmentation.shepherd_segmentation`) are the units of
analysis instead of pixels.

The workflow mirrors the pixel-based classification tools and reuses their
training and optimisation functions unchanged:

1. :func:`create_cls_segment_features` — create the segment feature layer
   (per-band statistics from the label raster, polygon shape metrics) and
   record the segmentation settings in the config's ``seg_info``.
2. :func:`extract_class_segment_samples` / :func:`split_class_segment_samples`
   / :func:`extract_and_split_class_segment_samples` — assign the class
   training points/polygons to segments (one sample per segment, conflicts
   handled) and split them into train/test/validation sets (keeping segments
   from the same training polygon together).
3. Train or optimise with the existing functions, e.g.
   :func:`~rsgispy.classification.classsklearn.train_sklearn_classifier` or
   :func:`~rsgispy.classification.classxgboost.train_xgboost_multiclass_classifier`
   (feature columns default to the segment model features in ``seg_info``).
4. Apply the classifier to the segment layer in chunks
   (:func:`apply_sklearn_classifier_segments`, and the XGBoost, LightGBM and
   CatBoost equivalents), writing a vector layer with the class ID, name,
   colour and optional probabilities.
5. :func:`rasterise_segment_classification` — map the segment classes back
   onto the segment raster grid with a colour table.
6. :func:`pop_segment_cls_accuracy_pts` — populate accuracy assessment points
   with the segment classes, for use with
   :mod:`~rsgispy.classification.classaccuracymetrics`.

:func:`cluster_segments` provides unsupervised classification (clustering)
of segments.
"""

# pylint: disable=too-many-lines

import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import warnings

import numpy as np
import pandas as pd

import rsgispy
from rsgispy.tools._gbdt_common import get_gbdt_backend
from rsgispy.classification._cls_common import (
    apply_classifier_vector,
    resolve_ordered_classes,
)
from rsgispy.classification._gbdt import apply_gbdt_classifier_segments
from rsgispy.classification._helpers import (
    ClsConfigInput,
    check_sklearn_available,
    coerce_cls_config,
    finalise_thematic_output_img,
    resolve_out_nodata,
    select_out_int_dtype,
)
from rsgispy.classification.classdata import (
    ClassificationConfig,
    write_cls_config,
    write_cls_samples,
)
from rsgispy.ml.segments import (
    SegmentFeatureConfig,
    assign_vectors_to_segments,
    attach_segment_features,
    create_segment_features,
    read_segment_features,
    resolve_seg_info,
)
from rsgispy.ml.vector import iter_vec_chunks, map_vector_chunks, prepare_feature_matrix

__all__ = [
    "create_cls_segment_features",
    "extract_class_segment_samples",
    "split_class_segment_samples",
    "extract_and_split_class_segment_samples",
    "apply_sklearn_classifier_segments",
    "apply_xgboost_binary_classifier_segments",
    "apply_xgboost_multiclass_classifier_segments",
    "apply_lightgbm_binary_classifier_segments",
    "apply_lightgbm_multiclass_classifier_segments",
    "apply_catboost_binary_classifier_segments",
    "apply_catboost_multiclass_classifier_segments",
    "rasterise_segment_classification",
    "cluster_segments",
    "pop_segment_cls_accuracy_pts",
]


def _load_cfg(cls_cfg: ClsConfigInput) -> ClassificationConfig:
    """Load a :class:`ClassificationConfig` (object or JSON path)."""
    cfg, _, _, _, _ = coerce_cls_config(cls_cfg)
    if cfg is None:
        raise rsgispy.RSGISPyException(
            "Segment classification requires a ClassificationConfig or JSON "
            "config file path (with seg_info), not a plain class mapping."
        )
    return cfg


def create_cls_segment_features(
    cls_cfg: ClsConfigInput,
    label_img: Union[str, os.PathLike],
    out_vec_file: Union[str, os.PathLike],
    out_cfg_file: Optional[Union[str, os.PathLike]] = None,
    **kwargs: Any,
) -> ClassificationConfig:
    """
    Create the segment feature layer for a classification from the config's
    image bands (``img_band_info``) and a segment label raster with
    :func:`rsgispy.ml.segments.create_segment_features`, and return a copy of
    the config with the segmentation settings stored in ``seg_info``
    (optionally written to ``out_cfg_file``).

    :param cls_cfg: :class:`~rsgispy.classification.classdata.ClassificationConfig`
                    or JSON config path.
    :param label_img: Segment label raster (e.g., from
                      :func:`rsgispy.segmentation.shepherd_segmentation`).
    :param out_vec_file: Output segment feature layer (GeoParquet with
                         compression for ``.parquet``, or any vector format).
    :param out_cfg_file: Optional output JSON config path.
    :param kwargs: Further arguments for
                   :func:`~rsgispy.ml.segments.create_segment_features`
                   (e.g., ``stats``, ``shape_metrics``, ``band_opts``,
                   ``out_table_file``, ``min_seg_pxls``).
    :return: Updated :class:`~rsgispy.classification.classdata.ClassificationConfig`.

    .. code:: python

        import rsgispy.classification.classsegments as cseg

        cfg = cseg.create_cls_segment_features(
            "cls_config.json", "s2_segs.tif", "s2_segs.parquet",
            out_cfg_file="cls_seg_config.json",
            band_opts={"s2": {"no_data_val": 0}},
        )
    """
    cfg = _load_cfg(cls_cfg)
    seg_info = create_segment_features(label_img, cfg, out_vec_file, **kwargs)
    new_cfg = ClassificationConfig(
        img_band_info=cfg.img_band_info,
        class_info=cfg.class_info,
        out_id_col=cfg.out_id_col,
        internal_id_col=cfg.internal_id_col,
        class_name_col=cfg.class_name_col,
        seg_info=seg_info,
    )
    if out_cfg_file is not None:
        write_cls_config(out_cfg_file, new_cfg)
    return new_cfg


def extract_class_segment_samples(
    cls_cfg: ClsConfigInput,
    out_file: Optional[Union[str, os.PathLike]] = None,
    poly_assign: Optional[str] = None,
    min_overlap: Optional[float] = None,
    conflict: Optional[str] = None,
    min_seg_pxls: Optional[int] = None,
    reproj_vec: bool = False,
    vec_def_epsg: Optional[int] = None,
    *,
    seg_info: Optional[SegmentFeatureConfig] = None,
    out_format: Optional[str] = None,
    out_compress: Optional[Union[str, bool]] = "snappy",
    group_col: str = "train_grp",
) -> pd.DataFrame:
    """
    Extract one training sample per segment from the class training vectors
    (``ClassInfoObj.vec_file``, points and/or polygons) defined in the
    classification config.

    * Points are assigned to the segment under them; polygons to the segments
      they cover by at least ``min_overlap`` of the segment's pixels
      (``poly_assign='overlap'``) or that contain the segment's centroid
      (``poly_assign='centroid'``).
    * Each segment gives one sample however many training points or polygons
      fall in it (avoiding duplicated samples and train/test leakage).
    * Segments hit by training data of more than one class are dropped
      (``conflict='drop'``) or take the class with most hits
      (``conflict='majority'``; ties are dropped).
    * Segments that are not valid, smaller than ``min_seg_pxls`` pixels, or
      (with the ``'drop'`` missing policy) have missing features are removed.

    The sample table has the same schema as the pixel workflow (output class
    ID, internal class ID and class name columns plus the feature columns) and
    the segment ID, a ``group_col`` (source training polygon, or segment for
    points) for group-aware splitting, ``n_hits`` and the pixel count, so it
    can be used directly with the existing training functions. A summary of
    the assignment (counts per class, conflicts and removed segments) is in
    ``df.attrs['report']``.

    :param cls_cfg: Classification config (object or JSON path) with
                    ``seg_info``.
    :param out_file: Optional output table (Parquet, snappy by default).
    :param poly_assign: Polygon rule (Default = ``seg_info.poly_assign``).
    :param min_overlap: Minimum covered fraction (Default = ``seg_info.min_overlap``).
    :param conflict: ``'drop'`` or ``'majority'`` (Default = ``seg_info.conflict``).
    :param min_seg_pxls: Minimum segment size (Default = ``seg_info.min_seg_pxls``).
    :param reproj_vec: Reproject training vectors to the segment CRS if needed.
    :param vec_def_epsg: Optional EPSG code for training vectors without a CRS.
    :param seg_info: Optional :class:`~rsgispy.ml.segments.SegmentFeatureConfig`
                     overriding ``cls_cfg.seg_info``.
    :param out_format: Optional output table format.
    :param out_compress: Parquet compression (Default = ``'snappy'``).
    :param group_col: Group column name (Default = ``'train_grp'``).
    :return: Sample :class:`pandas.DataFrame`.

    .. code:: python

        import rsgispy.classification.classsegments as cseg

        samples = cseg.extract_class_segment_samples(
            "cls_seg_config.json", "seg_samples.parquet"
        )
        print(samples.attrs["report"])
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    cfg = _load_cfg(cls_cfg)
    seg = resolve_seg_info(cfg, seg_info)
    eff_conflict = seg.conflict if conflict is None else conflict
    if eff_conflict not in ("drop", "majority"):
        raise rsgispy.RSGISPyException(
            f"conflict must be 'drop' or 'majority', got {conflict!r}."
        )
    eff_min_pxls = seg.min_seg_pxls if min_seg_pxls is None else int(min_seg_pxls)
    id_col = seg.id_col
    cls_name_col = cfg.class_name_col

    hits_list: List[pd.DataFrame] = []
    for cls_name, obj in cfg.class_info.items():
        if obj.vec_file is None:
            raise rsgispy.RSGISPyException(
                f"Class '{cls_name}' has no training vector (vec_file)."
            )
        hits = assign_vectors_to_segments(
            seg,
            obj.vec_file,
            obj.vec_lyr,
            poly_assign=poly_assign,
            min_overlap=min_overlap,
            reproj_vec=reproj_vec,
            vec_def_epsg=vec_def_epsg,
            group_prefix=f"{cls_name}:",
        )
        hits[cls_name_col] = cls_name
        hits_list.append(hits)
    all_hits = pd.concat(hits_list, ignore_index=True)
    report: Dict[str, Any] = {
        "n_hits": int(len(all_hits)),
        "n_hits_per_class": all_hits[cls_name_col].value_counts().to_dict(),
    }
    if all_hits.empty:
        raise rsgispy.RSGISPyException(
            "No training points or polygons fell on any segment; check the "
            "training vectors and their CRS."
        )

    votes = (
        all_hits.groupby([id_col, cls_name_col]).size().rename("n_hits").reset_index()
    )
    n_cls_per_seg = votes.groupby(id_col)[cls_name_col].transform("nunique")
    conflict_ids = votes.loc[n_cls_per_seg > 1, id_col].unique()
    report["n_conflict_segments"] = int(conflict_ids.size)
    if eff_conflict == "drop":
        kept = votes[n_cls_per_seg == 1]
    else:
        max_votes = votes.groupby(id_col)["n_hits"].transform("max")
        top = votes[votes["n_hits"] == max_votes]
        n_top = top.groupby(id_col)[cls_name_col].transform("nunique")
        kept = top[n_top == 1]
    n_conflict_dropped = int(votes[id_col].nunique() - kept[id_col].nunique())
    report["n_conflict_segments_dropped"] = n_conflict_dropped
    if conflict_ids.size:
        warnings.warn(
            f"{conflict_ids.size} segment(s) were hit by training data of more "
            f"than one class ({n_conflict_dropped} dropped, conflict="
            f"'{eff_conflict}').",
            UserWarning,
            stacklevel=2,
        )

    # Group per segment: the source polygon if any, else the segment itself.
    hit_kept = all_hits.merge(kept[[id_col, cls_name_col]], on=[id_col, cls_name_col])
    hit_kept = hit_kept.sort_values(
        by=["src_type", group_col], ascending=[False, True], kind="stable"
    )
    grp = hit_kept.drop_duplicates(subset=[id_col], keep="first")[[id_col, group_col]]
    samples = kept.merge(grp, on=id_col)

    samples = attach_segment_features(seg, samples, eff_min_pxls, report)

    ordered = resolve_ordered_classes(cfg.class_info, use_internal_id=True)
    out_lut = {name: int(obj.out_id) for name, obj, _ in ordered}
    int_lut = {name: int(t_id) for name, _, t_id in ordered}
    samples[cfg.out_id_col] = samples[cls_name_col].map(out_lut).astype(np.int64)
    samples[cfg.internal_id_col] = samples[cls_name_col].map(int_lut).astype(np.int64)
    col_order = [
        id_col,
        cfg.out_id_col,
        cfg.internal_id_col,
        cls_name_col,
        group_col,
        "n_hits",
        seg.pxl_count_col,
    ] + seg.model_features
    samples = samples[col_order].sort_values(id_col).reset_index(drop=True)
    report["n_samples"] = int(len(samples))
    report["n_samples_per_class"] = samples[cls_name_col].value_counts().to_dict()
    samples.attrs["report"] = report
    if out_file is not None:
        write_cls_samples(samples, out_file, out_format=out_format, out_compress=out_compress)
    return samples


def split_class_segment_samples(
    samples_df: pd.DataFrame,
    train_prop: Optional[float] = None,
    test_prop: float = 0.2,
    valid_prop: float = 0.2,
    out_train_file: Optional[Union[str, os.PathLike]] = None,
    out_test_file: Optional[Union[str, os.PathLike]] = None,
    out_valid_file: Optional[Union[str, os.PathLike]] = None,
    stratify_col: str = "internal_cls_id",
    group_col: Optional[str] = "train_grp",
    rnd_seed: Optional[int] = None,
    *,
    min_cls_smpls_warn: int = 10,
    out_format: Optional[str] = None,
    out_compress: Optional[Union[str, bool]] = "snappy",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split segment samples per class into training, testing and validation
    sets, keeping all samples of a group (``group_col``, e.g. the segments
    from one training polygon) in the same set to avoid spatial leakage.

    Groups are assigned (in random order) to the set furthest below its target
    number of samples. A class with fewer groups than the number of sets with
    a non-zero proportion raises an exception (reduce ``test_prop`` /
    ``valid_prop`` or add training data); a warning is issued for classes with
    fewer than ``min_cls_smpls_warn`` samples. Segment workflows usually give
    far fewer samples than pixel workflows, so consider fewer cross-validation
    folds, ``valid_prop=0`` (with cross-validation instead of a fixed
    validation set) and class weights for imbalanced classes.

    :param samples_df: Sample table (e.g., from
                       :func:`extract_class_segment_samples`).
    :param train_prop: Training proportion (Default = ``1 - test_prop - valid_prop``).
    :param test_prop: Test proportion (Default = ``0.2``).
    :param valid_prop: Validation proportion (Default = ``0.2``).
    :param out_train_file: Optional training table output path.
    :param out_test_file: Optional test table output path.
    :param out_valid_file: Optional validation table output path.
    :param stratify_col: Class column to stratify by (Default = ``'internal_cls_id'``).
    :param group_col: Group column (``None`` = every sample its own group).
    :param rnd_seed: Optional random seed.
    :param min_cls_smpls_warn: Warn for classes with fewer samples (Default = ``10``).
    :param out_format: Optional output table format.
    :param out_compress: Parquet compression (Default = ``'snappy'``).
    :return: ``(train_df, test_df, valid_df)``.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    props = [
        1.0 - float(test_prop) - float(valid_prop) if train_prop is None else float(train_prop),
        float(test_prop),
        float(valid_prop),
    ]
    if any(p < 0 for p in props) or not np.isclose(sum(props), 1.0):
        raise rsgispy.RSGISPyException(
            f"Split proportions must be non-negative and sum to 1, got {props}."
        )
    if stratify_col not in samples_df.columns:
        raise rsgispy.RSGISPyException(f"Column '{stratify_col}' not in samples_df.")
    rng = np.random.default_rng(rnd_seed)
    grp_series = (
        samples_df[group_col].astype(str)
        if group_col is not None and group_col in samples_df.columns
        else pd.Series(np.arange(len(samples_df)).astype(str), index=samples_df.index)
    )
    split_idx = np.full(len(samples_df), -1, dtype=np.int64)
    n_active = sum(1 for p in props if p > 0)
    pos_of = {idx: pos for pos, idx in enumerate(samples_df.index)}
    for cls_val, cls_df in samples_df.groupby(stratify_col):
        groups = grp_series.loc[cls_df.index]
        grp_sizes = groups.value_counts()
        if len(cls_df) < int(min_cls_smpls_warn):
            warnings.warn(
                f"Class {cls_val} has only {len(cls_df)} sample(s).",
                UserWarning,
                stacklevel=2,
            )
        if len(grp_sizes) < n_active:
            raise rsgispy.RSGISPyException(
                f"Class {cls_val} has {len(grp_sizes)} sample group(s), fewer than "
                f"the {n_active} sets requested; add training data or reduce "
                "test_prop / valid_prop."
            )
        targets = np.array(props) * len(cls_df)
        counts = np.zeros(3)
        order = rng.permutation(grp_sizes.index.to_numpy())
        for g_i, grp_name in enumerate(order):
            remaining = len(order) - g_i
            empty_needed = [s for s in range(3) if props[s] > 0 and counts[s] == 0]
            if len(empty_needed) >= remaining:
                tgt = empty_needed[0]
            else:
                deficit = np.where(np.array(props) > 0, targets - counts, -np.inf)
                tgt = int(np.argmax(deficit))
            counts[tgt] += grp_sizes[grp_name]
            for row_idx in groups.index[groups == grp_name]:
                split_idx[pos_of[row_idx]] = tgt
    outs = []
    for s_idx, out_path in enumerate((out_train_file, out_test_file, out_valid_file)):
        part = samples_df[split_idx == s_idx].reset_index(drop=True)
        if out_path is not None:
            write_cls_samples(part, out_path, out_format=out_format, out_compress=out_compress)
        outs.append(part)
    return outs[0], outs[1], outs[2]


def extract_and_split_class_segment_samples(
    cls_cfg: ClsConfigInput,
    out_train_file: Optional[Union[str, os.PathLike]] = None,
    out_test_file: Optional[Union[str, os.PathLike]] = None,
    out_valid_file: Optional[Union[str, os.PathLike]] = None,
    out_samples_file: Optional[Union[str, os.PathLike]] = None,
    test_prop: float = 0.2,
    valid_prop: float = 0.2,
    rnd_seed: Optional[int] = None,
    **kwargs: Any,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Extract segment samples with :func:`extract_class_segment_samples` and
    split them with :func:`split_class_segment_samples` (group-aware,
    stratified by class).

    :param cls_cfg: Classification config with ``seg_info``.
    :param out_train_file: Optional training table output path.
    :param out_test_file: Optional test table output path.
    :param out_valid_file: Optional validation table output path.
    :param out_samples_file: Optional output path for all samples.
    :param test_prop: Test proportion (Default = ``0.2``).
    :param valid_prop: Validation proportion (Default = ``0.2``).
    :param rnd_seed: Optional random seed.
    :param kwargs: Further arguments for :func:`extract_class_segment_samples`.
    :return: ``(train_df, test_df, valid_df)``.

    .. code:: python

        import rsgispy.classification.classsegments as cseg

        train_df, test_df, valid_df = cseg.extract_and_split_class_segment_samples(
            "cls_seg_config.json", "train.parquet", "test.parquet",
            "valid.parquet", rnd_seed=42,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    samples = extract_class_segment_samples(cls_cfg, out_file=out_samples_file, **kwargs)
    cfg = _load_cfg(cls_cfg)
    return split_class_segment_samples(
        samples,
        test_prop=test_prop,
        valid_prop=valid_prop,
        out_train_file=out_train_file,
        out_test_file=out_test_file,
        out_valid_file=out_valid_file,
        stratify_col=cfg.internal_id_col,
        group_col=kwargs.get("group_col", "train_grp"),
        rnd_seed=rnd_seed,
    )


def apply_sklearn_classifier_segments(
    cls_cfg: ClsConfigInput,
    sk_classifier: Union[Any, str, os.PathLike],
    out_vec_file: Union[str, os.PathLike],
    out_probs: Optional[str] = None,
    **kwargs: Any,
) -> None:
    """
    Apply a trained ``scikit-learn`` (or compatible) classifier to the segment
    feature layer in chunks (reading only the segment ID, valid flag and model
    feature columns, never the geometry) and write a vector layer with
    ``cls_id`` (user output class ID), ``cls_name``, ``cls_colour``
    (``#rrggbb``), ``cls_red``/``cls_green``/``cls_blue`` and optional
    probability columns. Segments that are not valid, or have missing features
    under the ``'drop'`` policy, get the no-data class ID.

    :param cls_cfg: Classification config (object or JSON path) with ``seg_info``.
    :param sk_classifier: Fitted classifier or path to a saved ``.joblib`` model.
    :param out_vec_file: Output vector layer (GeoParquet with compression for
                         ``.parquet``, or any vector format, e.g. ``.gpkg``).
    :param out_probs: Optional probability outputs: ``'all'`` (one
                      ``<prefix>prob_<class>`` column per class), ``'max'``
                      (``<prefix>prob_max``) or, for binary classifiers,
                      ``'pos'`` (positive class).
    :param seg_info: Optional :class:`~rsgispy.ml.segments.SegmentFeatureConfig`
                     overriding ``cls_cfg.seg_info``.
    :param in_vec_file: Optional feature layer/table (Default = from ``seg_info``).
    :param copy_cols: Input columns copied to the output (Default = all;
                      ``[]`` = segment ID only).
    :param out_prefix: Output column prefix (Default = ``'cls_'``).
    :param missing: Missing feature policy override (Default = ``seg_info.missing``).
    :param scaler: Optional fitted :class:`~rsgispy.ml.scaling.FeatureScaler`.
    :param chunk_size: Segments per chunk (Default = ``100_000``).
    :param out_no_data_val: Class ID for segments not classified (Default =
                            ``0`` unless used by a class).
    :param use_internal_id: Whether the model predicts internal class IDs
                            (Default = ``True``).
    :param out_format: Optional output format.
    :param out_compress: Parquet compression (Default = ``'snappy'``).
    :param overwrite: If ``True``, replace an existing output.
    :param use_pbar: If ``True``, display a progress bar.

    .. code:: python

        import rsgispy.classification.classsegments as cseg

        cseg.apply_sklearn_classifier_segments(
            "cls_seg_config.json", "rf_model.joblib", "segs_classified.parquet",
            out_probs="max",
        )
    """
    # pylint: disable=import-outside-toplevel
    from rsgispy.classification.classsklearn import load_sklearn_classifier

    check_sklearn_available("apply_sklearn_classifier_segments")
    is_path = isinstance(sk_classifier, (str, os.PathLike))
    estimator = load_sklearn_classifier(sk_classifier) if is_path else sk_classifier
    if is_path:
        kwargs.setdefault("model_file", sk_classifier)
    apply_classifier_vector(cls_cfg, estimator, out_vec_file, out_probs=out_probs, **kwargs)


def _gbdt_backend(name: str) -> Any:
    """Return the GBDT classification backend spec for ``name``."""
    return get_gbdt_backend("classification", name)


def _make_gbdt_apply(backend_name: str, binary: bool, label: str) -> Any:
    """Build a public GBDT segment apply function."""
    func_name = f"apply_{backend_name}_{'binary' if binary else 'multiclass'}_classifier_segments"

    def _apply(
        cls_cfg: ClsConfigInput,
        classifier: Union[Any, str, os.PathLike],
        out_vec_file: Union[str, os.PathLike],
        out_probs: Optional[str] = None,
        pos_class: Optional[Union[str, int]] = None,
        pred_threshold: Optional[float] = None,
        *,
        use_gpu: bool = False,
        gpu_device_id: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        # pylint: disable=too-many-arguments,too-many-positional-arguments
        if binary:
            kwargs["pos_class"] = pos_class
            kwargs["pred_threshold"] = 0.5 if pred_threshold is None else pred_threshold
        elif pos_class is not None or pred_threshold is not None:
            raise rsgispy.RSGISPyException(
                "pos_class and pred_threshold are only used by binary classifiers."
            )
        apply_gbdt_classifier_segments(
            _gbdt_backend(backend_name),
            func_name,
            cls_cfg,
            classifier,
            out_vec_file,
            binary,
            out_probs=out_probs,
            use_gpu=use_gpu,
            gpu_device_id=gpu_device_id,
            **kwargs,
        )

    kind = "binary" if binary else "multi-class"
    extra = (
        """
    :param pos_class: Positive class name or ID (Default = higher class ID).
    :param pred_threshold: Positive class probability threshold (Default = ``0.5``)."""
        if binary
        else """
    :param pos_class: Not used (binary classifiers only).
    :param pred_threshold: Not used (binary classifiers only)."""
    )
    _apply.__name__ = func_name
    _apply.__qualname__ = func_name
    _apply.__doc__ = f"""
    Apply a trained {kind} {label} classifier to the segment feature layer in
    chunks and write a vector layer with the class outputs (see
    :func:`apply_sklearn_classifier_segments` for the output columns). The
    ``'nan'`` missing policy can be used as {label} handles missing values.

    :param cls_cfg: Classification config (object or JSON path) with ``seg_info``.
    :param classifier: Fitted {label} classifier or saved model path.
    :param out_vec_file: Output vector layer.
    :param out_probs: Optional probability outputs (``'all'``, ``'max'``,
                      ``'pos'`` for binary).{extra}
    :param use_gpu: Predict on the GPU if available (Default = ``False``).
    :param gpu_device_id: Optional GPU device ID.
    :param kwargs: Further options (``seg_info``, ``copy_cols``, ``out_prefix``,
                   ``missing``, ``scaler``, ``chunk_size``, ``out_no_data_val``,
                   ``out_format``, ``out_compress``, ``overwrite``, ``use_pbar``).

    .. code:: python

        import rsgispy.classification.classsegments as cseg

        cseg.{func_name}(
            "cls_seg_config.json", "model.json", "segs_classified.parquet"
        )
    """
    return _apply


apply_xgboost_binary_classifier_segments = _make_gbdt_apply("xgboost", True, "XGBoost")
apply_xgboost_multiclass_classifier_segments = _make_gbdt_apply(
    "xgboost", False, "XGBoost"
)
apply_lightgbm_binary_classifier_segments = _make_gbdt_apply("lightgbm", True, "LightGBM")
apply_lightgbm_multiclass_classifier_segments = _make_gbdt_apply(
    "lightgbm", False, "LightGBM"
)
apply_catboost_binary_classifier_segments = _make_gbdt_apply("catboost", True, "CatBoost")
apply_catboost_multiclass_classifier_segments = _make_gbdt_apply(
    "catboost", False, "CatBoost"
)


def _read_id_values(
    vec_file: Union[str, os.PathLike],
    vec_lyr: Optional[str],
    id_col: str,
    cols: Sequence[str],
    chunk_size: int,
) -> pd.DataFrame:
    """Read the ID and value columns of a layer/table in chunks (no geometry)."""
    parts = list(
        iter_vec_chunks(vec_file, [id_col] + list(cols), vec_lyr=vec_lyr, chunk_size=chunk_size)
    )
    if not parts:
        return pd.DataFrame(columns=[id_col] + list(cols))
    return pd.concat(parts, ignore_index=True)


def rasterise_segment_classification(
    cls_cfg: Optional[ClsConfigInput],
    cls_vec_file: Union[str, os.PathLike],
    output_img: Union[str, os.PathLike],
    cls_col: str = "cls_id",
    out_score_img: Optional[Union[str, os.PathLike]] = None,
    score_cols: Optional[Sequence[str]] = None,
    label_img: Optional[Union[str, os.PathLike]] = None,
    clr_lut: Optional[Mapping[int, Sequence[int]]] = None,
    *,
    cls_vec_lyr: Optional[str] = None,
    id_col: Optional[str] = None,
    out_no_data_val: Optional[int] = None,
    class_clr_names: bool = True,
    score_no_data_val: float = -1.0,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    calc_stats: bool = True,
    chunk_size: int = 100_000,
    use_pbar: bool = True,
    creation_opts: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Rasterise a classified segment layer onto the segment label raster's
    pixel grid by mapping each segment ID to its class ID (no geometry is
    read or burnt: only the ID and class columns are read, in chunks, and the
    label raster is streamed block by block through the lookup).

    With a classification config, pixel values are the user ``out_id``\\ s,
    the output datatype is the smallest unsigned integer type fitting the class
    IDs and the no-data value (chosen as in the pixel workflow: ``0`` unless
    used by a class), and the class colour table is written. Without a config
    (e.g., for :func:`cluster_segments` output) the values of ``cls_col`` are
    written with ``clr_lut`` (or random colours).

    :param cls_cfg: Classification config with ``seg_info``, or ``None``.
    :param cls_vec_file: Classified segment layer/table.
    :param output_img: Output thematic raster.
    :param cls_col: Class ID column (Default = ``'cls_id'``).
    :param out_score_img: Optional ``float32`` probability raster.
    :param score_cols: Probability columns for ``out_score_img`` (Default =
                       all ``cls_prob_*`` columns).
    :param label_img: Segment label raster (Default = ``seg_info.label_img``).
    :param clr_lut: Optional colour table ``{value: (R, G, B[, A])}``.
    :param cls_vec_lyr: Optional layer name.
    :param id_col: Segment ID column (Default = ``seg_info.id_col`` or ``'seg_id'``).
    :param out_no_data_val: Optional output no-data value.
    :param class_clr_names: Write the class colour table (Default = ``True``).
    :param score_no_data_val: Probability raster no-data (Default = ``-1``).
    :param gdalformat: Output GDAL format (Default = ``'GTIFF'``).
    :param datatype: Optional ``rsgispy.TYPE_*`` output datatype.
    :param calc_stats: Calculate statistics and pyramids (Default = ``True``).
    :param chunk_size: Rows per chunk when reading the layer.
    :param use_pbar: Display progress bars.
    :param creation_opts: Optional GDAL creation options.

    .. code:: python

        import rsgispy.classification.classsegments as cseg

        cseg.rasterise_segment_classification(
            "cls_seg_config.json", "segs_classified.parquet", "classified.tif",
            out_score_img="class_probs.tif",
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,import-outside-toplevel
    from rsgispy.ml.vector import read_vec_layer_info
    from rsgispy.zonalstats.labels import rasterise_label_values

    cfg = _load_cfg(cls_cfg) if cls_cfg is not None else None
    seg = getattr(cfg, "seg_info", None) if cfg is not None else None
    eff_id = id_col or (seg.id_col if seg is not None else "seg_id")
    eff_label = label_img or (seg.label_img if seg is not None else None)
    if eff_label is None:
        raise rsgispy.RSGISPyException("label_img must be given (no seg_info).")
    lab_nd = seg.label_no_data_val if seg is not None else 0

    info = read_vec_layer_info(cls_vec_file, cls_vec_lyr)
    if out_score_img is not None and score_cols is None:
        score_cols = [c for c in info.columns if c.startswith("cls_prob_")]
    tbl = _read_id_values(
        cls_vec_file, cls_vec_lyr, eff_id, [cls_col] + list(score_cols or []), chunk_size
    )
    cls_vals = tbl[cls_col].to_numpy()
    clr: Optional[Dict[int, List[int]]] = None
    if cfg is not None:
        ordered = resolve_ordered_classes(cfg.class_info, use_internal_id=True)
        out_ids = [int(obj.out_id) for _, obj, _ in ordered]
        eff_nodata = resolve_out_nodata(out_no_data_val, out_ids)
        cls_vals = np.where(np.isin(cls_vals, out_ids), cls_vals, eff_nodata)
        if class_clr_names:
            clr = {int(obj.out_id): [obj.red, obj.green, obj.blue, 255] for _, obj, _ in ordered}
    else:
        out_ids = sorted({int(v) for v in np.unique(cls_vals)})
        eff_nodata = 0 if out_no_data_val is None else int(out_no_data_val)
        out_ids = [v for v in out_ids if v != eff_nodata] or [eff_nodata]
    if clr_lut is not None:
        clr = {int(k): list(v) for k, v in clr_lut.items()}
    rsgis_dt, np_dt = select_out_int_dtype(out_ids + [eff_nodata])
    if datatype is not None:
        rsgis_dt = datatype
        np_dt = np.dtype(rsgispy.get_numpy_datatype(datatype))
    rasterise_label_values(
        eff_label,
        tbl[eff_id].to_numpy(),
        cls_vals.astype(np_dt),
        output_img,
        label_no_data_val=lab_nd,
        out_no_data_val=eff_nodata,
        thematic=True,
        gdalformat=gdalformat,
        datatype=rsgis_dt,
        calc_stats=False,
        use_pbar=use_pbar,
        creation_opts=creation_opts,
    )
    finalise_thematic_output_img(
        out_cls_path=os.fspath(output_img),
        eff_nodata=eff_nodata,
        out_rio_dtype=rsgispy.get_rasterio_datatype(rsgis_dt),
        out_np_dtype=np.dtype(np_dt),
        calc_stats=calc_stats,
        add_clr_tab=clr is None,
        clr_lut=clr,
        use_pbar=use_pbar,
    )
    if out_score_img is not None:
        if not score_cols:
            raise rsgispy.RSGISPyException(
                "No probability columns found for out_score_img."
            )
        rasterise_label_values(
            eff_label,
            tbl[eff_id].to_numpy(),
            np.nan_to_num(
                tbl[list(score_cols)].to_numpy(dtype=np.float32), nan=score_no_data_val
            ),
            out_score_img,
            label_no_data_val=lab_nd,
            out_no_data_val=score_no_data_val,
            band_names=[c.replace("cls_prob_", "prob_") for c in score_cols],
            gdalformat=gdalformat,
            datatype=rsgispy.TYPE_32FLOAT,
            calc_stats=calc_stats,
            use_pbar=use_pbar,
            creation_opts=creation_opts,
        )


def cluster_segments(
    cls_cfg_or_seg: Union[ClsConfigInput, SegmentFeatureConfig],
    clusterer: Any,
    out_vec_file: Union[str, os.PathLike],
    n_sample: Optional[int] = None,
    scaler: Optional[Any] = None,
    out_col: str = "cluster_id",
    rnd_seed: Optional[int] = None,
    *,
    feature_cols: Optional[Sequence[str]] = None,
    copy_cols: Optional[Sequence[str]] = None,
    chunk_size: int = 100_000,
    out_format: Optional[str] = None,
    out_compress: Optional[Union[str, bool]] = "snappy",
    overwrite: bool = False,
    use_pbar: bool = True,
) -> Any:
    """
    Unsupervised classification (clustering) of segments: fit a
    ``scikit-learn``-style clusterer (with ``fit`` and ``predict``, e.g.
    :class:`sklearn.cluster.KMeans` or
    :class:`sklearn.mixture.GaussianMixture`) to all segments, or a random
    sample of ``n_sample`` segments, then apply it to the segment layer in
    chunks, writing ``out_col`` (cluster IDs ``1..k``; ``0`` = not clustered,
    e.g. invalid or missing features). The output can be rasterised with
    :func:`rasterise_segment_classification` (``cls_cfg=None``,
    ``cls_col=out_col``).

    :param cls_cfg_or_seg: Classification config with ``seg_info`` (object or
                           JSON path) or a
                           :class:`~rsgispy.ml.segments.SegmentFeatureConfig`.
    :param clusterer: Unfitted clusterer.
    :param out_vec_file: Output vector layer.
    :param n_sample: Optional number of segments sampled to fit the clusterer.
    :param scaler: Optional :class:`~rsgispy.ml.scaling.FeatureScaler` (fitted
                   on the sample if not already fitted).
    :param out_col: Output cluster column (Default = ``'cluster_id'``).
    :param rnd_seed: Optional random seed for the sample.
    :param feature_cols: Optional features (Default = ``seg_info`` model features).
    :param copy_cols: Input columns copied to the output (Default = all).
    :param chunk_size: Segments per chunk.
    :param out_format: Optional output format.
    :param out_compress: Parquet compression.
    :param overwrite: Replace an existing output.
    :param use_pbar: Display progress bars.
    :return: The fitted clusterer.

    .. code:: python

        from sklearn.cluster import KMeans
        from rsgispy.ml.scaling import FeatureScaler
        import rsgispy.classification.classsegments as cseg

        cseg.cluster_segments(
            "cls_seg_config.json", KMeans(n_clusters=8, random_state=1),
            "segs_clusters.parquet", scaler=FeatureScaler(method="zscore"),
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    seg = (
        cls_cfg_or_seg
        if isinstance(cls_cfg_or_seg, SegmentFeatureConfig)
        else resolve_seg_info(_load_cfg(cls_cfg_or_seg))
    )
    feats = list(feature_cols) if feature_cols is not None else seg.model_features
    if not hasattr(clusterer, "predict"):
        raise rsgispy.RSGISPyException("The clusterer must provide predict().")
    missing = "drop" if seg.missing == "nan" else seg.missing
    feat_df = read_segment_features(seg, None, [seg.valid_col] + feats)
    x_df, ok = prepare_feature_matrix(
        feat_df, feats, seg.valid_col, missing, seg.fill_value, seg.feat_no_data_val
    )
    if x_df.empty:
        raise rsgispy.RSGISPyException("No valid segments to cluster.")
    if n_sample is not None and int(n_sample) < len(x_df):
        rng = np.random.default_rng(rnd_seed)
        x_df = x_df.iloc[np.sort(rng.choice(len(x_df), int(n_sample), replace=False))]
    del ok
    x_fit = x_df.to_numpy()
    if scaler is not None:
        if not scaler.is_fitted:
            scaler.fit(x_fit)
        x_fit = np.asarray(scaler.transform(x_fit))
    clusterer.fit(x_fit)

    def _chunk(df: pd.DataFrame) -> pd.DataFrame:
        x_c, row_ok = prepare_feature_matrix(
            df, feats, seg.valid_col, missing, seg.fill_value, seg.feat_no_data_val, scaler
        )
        out = np.zeros(len(df), dtype=np.int32)
        if np.any(row_ok):
            out[row_ok] = np.asarray(clusterer.predict(x_c.to_numpy())).astype(np.int32) + 1
        return pd.DataFrame({out_col: out})

    src_path, src_lyr = seg.feature_source
    geom_file = seg.seg_vec_file if seg.feat_table_file is not None else None
    map_vector_chunks(
        src_path,
        out_vec_file,
        [seg.valid_col] + feats,
        _chunk,
        in_vec_lyr=src_lyr,
        geom_vec_file=geom_file,
        geom_vec_lyr=seg.seg_vec_lyr if geom_file else None,
        id_col=seg.id_col,
        copy_cols=copy_cols,
        chunk_size=chunk_size,
        out_format=out_format,
        out_compress=out_compress,
        overwrite=overwrite,
        use_pbar=use_pbar,
    )
    return clusterer


def pop_segment_cls_accuracy_pts(
    cls_cfg: ClsConfigInput,
    cls_vec_file: Union[str, os.PathLike],
    acc_pts_vec_file: Any,
    acc_pts_vec_lyr: Optional[str] = None,
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    cls_col: str = "cls_id",
    out_col: str = "seg_cls",
    *,
    cls_vec_lyr: Optional[str] = None,
    out_name_col: Optional[str] = "seg_cls_name",
    reproj_vec: bool = False,
    vec_def_epsg: Optional[int] = None,
    out_format: Optional[str] = None,
    out_compress: Optional[Union[str, bool]] = None,
    chunk_size: int = 100_000,
) -> Optional[Any]:
    """
    Populate accuracy assessment points with the class of the segment they
    fall in (sampling the segment label raster for the segment ID and looking
    the class up in the classified segment layer, without reading geometry).
    The result can be used with
    :func:`~rsgispy.classification.classaccuracymetrics.calc_acc_stats_ptonly`
    (point-based accuracy) or, after
    :func:`rasterise_segment_classification`, with
    :func:`~rsgispy.classification.classaccuracymetrics.calc_acc_stats_img`
    (area-weighted accuracy).

    :param cls_cfg: Classification config with ``seg_info``.
    :param cls_vec_file: Classified segment layer/table.
    :param acc_pts_vec_file: Accuracy points (file or GeoDataFrame).
    :param acc_pts_vec_lyr: Optional points layer name.
    :param out_vec_file: Optional output vector file (else returns the
                         GeoDataFrame).
    :param out_vec_lyr: Optional output layer name.
    :param cls_col: Class ID column in ``cls_vec_file`` (Default = ``'cls_id'``).
    :param out_col: Output class ID column (Default = ``'seg_cls'``).
    :param cls_vec_lyr: Optional classified layer name.
    :param out_name_col: Optional output class name column.
    :param reproj_vec: Reproject the points to the segment CRS if needed.
    :param vec_def_epsg: Optional EPSG code for points with no CRS.
    :param out_format: Optional output format.
    :param out_compress: Optional GeoParquet compression.
    :param chunk_size: Rows per chunk when reading the classified layer.
    :return: GeoDataFrame when ``out_vec_file`` is ``None``, else ``None``.

    .. code:: python

        import rsgispy.classification.classsegments as cseg
        import rsgispy.classification.classaccuracymetrics as cam

        pts = cseg.pop_segment_cls_accuracy_pts(
            "cls_seg_config.json", "segs_classified.parquet", "acc_pts.gpkg"
        )
        stats = cam.calc_acc_stats_ptonly(pts, None, "ref_cls", "seg_cls")
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,import-outside-toplevel
    from rsgispy.ml.segments import _load_training_vec
    from rsgispy.vectorprocess import finish_vec_output
    import rasterio

    cfg = _load_cfg(cls_cfg)
    seg = resolve_seg_info(cfg)
    with rasterio.open(seg.label_img) as lab_src:
        pts = _load_training_vec(
            acc_pts_vec_file, acc_pts_vec_lyr, lab_src.crs, reproj_vec, vec_def_epsg
        )
        bnd = lab_src.bounds
        coords = [(g.x, g.y) for g in pts.geometry]
        inside = np.array(
            [bnd.left <= x < bnd.right and bnd.bottom < y <= bnd.top for x, y in coords],
            dtype=bool,
        )
        seg_ids = np.full(len(coords), -1, dtype=np.int64)
        if np.any(inside):
            seg_ids[inside] = [
                int(v[0])
                for v in lab_src.sample(
                    [c for c, ok in zip(coords, inside) if ok], indexes=seg.label_band
                )
            ]
    tbl = _read_id_values(cls_vec_file, cls_vec_lyr, seg.id_col, [cls_col], chunk_size)
    lut = pd.Series(tbl[cls_col].to_numpy(), index=tbl[seg.id_col].to_numpy())
    ordered = resolve_ordered_classes(cfg.class_info, use_internal_id=True)
    out_ids = [int(obj.out_id) for _, obj, _ in ordered]
    nodata = resolve_out_nodata(None, out_ids)
    cls_ids = lut.reindex(seg_ids).fillna(nodata).to_numpy().astype(np.int64)
    pts[out_col] = cls_ids
    if out_name_col:
        names = {int(obj.out_id): name for name, obj, _ in ordered}
        pts[out_name_col] = [names.get(int(v)) for v in cls_ids]
    return finish_vec_output(
        pts, out_vec_file, out_vec_lyr=out_vec_lyr, out_format=out_format,
        out_compress=out_compress,
    )
