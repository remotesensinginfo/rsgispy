"""
The regression.regsegments module provides object-based (segment)
regression: the image segments from :mod:`rsgispy.segmentation` are the units
of analysis instead of pixels. It mirrors
:mod:`rsgispy.classification.classsegments`:

1. :func:`create_reg_segment_features` — create the segment feature layer
   and store the segmentation settings in the config's ``seg_info``.
2. :func:`extract_reg_segment_samples` / :func:`split_reg_segment_samples` /
   :func:`extract_and_split_reg_segment_samples` — assign the reference
   points/polygons (with target values) to segments, aggregating several
   values in one segment (mean or median), and split into train/test/
   validation sets keeping segments from the same reference polygon together.
3. Train or optimise with the existing functions (e.g.
   :func:`~rsgispy.regression.regsklearn.train_sklearn_regressor`); feature
   columns default to the segment model features in ``seg_info``.
4. Apply the regressor to the segment layer in chunks
   (:func:`apply_sklearn_regressor_segments` and the XGBoost, LightGBM and
   CatBoost equivalents), writing one ``pred_<target>`` column per target.
5. :func:`rasterise_segment_regression` — map the predictions back onto the
   segment raster grid (one band per target).
"""

import os
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

import rsgispy
from rsgispy.tools._gbdt_common import get_gbdt_backend
from rsgispy.ml.segments import (
    SegmentFeatureConfig,
    assign_vectors_to_segments,
    attach_segment_features,
    create_segment_features,
    resolve_seg_info,
)
from rsgispy.ml.vector import iter_vec_chunks
from rsgispy.regression._reg_common import (
    RegConfigInput,
    apply_gbdt_regressor_segments,
    apply_regressor_vector,
    check_sklearn_available,
    coerce_reg_config,
)
from rsgispy.regression.regdata import (
    RegressionConfig,
    write_reg_config,
    write_reg_samples,
)

__all__ = [
    "create_reg_segment_features",
    "extract_reg_segment_samples",
    "split_reg_segment_samples",
    "extract_and_split_reg_segment_samples",
    "apply_sklearn_regressor_segments",
    "apply_xgboost_regressor_segments",
    "apply_lightgbm_regressor_segments",
    "apply_catboost_regressor_segments",
    "rasterise_segment_regression",
]


def _load_cfg(reg_cfg: RegConfigInput) -> RegressionConfig:
    """Load a :class:`RegressionConfig` (object or JSON path)."""
    cfg, _ = coerce_reg_config(reg_cfg)
    if cfg is None:
        raise rsgispy.RSGISPyException(
            "Segment regression requires a RegressionConfig or JSON config path."
        )
    return cfg


def create_reg_segment_features(
    reg_cfg: RegConfigInput,
    label_img: Union[str, os.PathLike],
    out_vec_file: Union[str, os.PathLike],
    out_cfg_file: Optional[Union[str, os.PathLike]] = None,
    **kwargs: Any,
) -> RegressionConfig:
    """
    Create the segment feature layer for a regression from the config's image
    bands and a segment label raster with
    :func:`rsgispy.ml.segments.create_segment_features`, and return a copy of
    the config with ``seg_info`` set (optionally written to ``out_cfg_file``).

    :param reg_cfg: :class:`~rsgispy.regression.regdata.RegressionConfig` or
                    JSON config path.
    :param label_img: Segment label raster.
    :param out_vec_file: Output segment feature layer.
    :param out_cfg_file: Optional output JSON config path.
    :param kwargs: Further arguments for
                   :func:`~rsgispy.ml.segments.create_segment_features`.
    :return: Updated :class:`~rsgispy.regression.regdata.RegressionConfig`.

    .. code:: python

        import rsgispy.regression.regsegments as rseg

        cfg = rseg.create_reg_segment_features(
            "reg_config.json", "s2_segs.tif", "s2_segs.parquet",
            out_cfg_file="reg_seg_config.json",
        )
    """
    cfg = _load_cfg(reg_cfg)
    seg_info = create_segment_features(label_img, cfg, out_vec_file, **kwargs)
    new_cfg = RegressionConfig(
        img_band_info=cfg.img_band_info,
        target_cols=cfg.target_cols,
        vec_file=cfg.vec_file,
        vec_lyr=cfg.vec_lyr,
        seg_info=seg_info,
    )
    if out_cfg_file is not None:
        write_reg_config(out_cfg_file, new_cfg)
    return new_cfg


def extract_reg_segment_samples(
    reg_cfg: RegConfigInput,
    out_file: Optional[Union[str, os.PathLike]] = None,
    vec_file: Optional[Any] = None,
    vec_lyr: Optional[str] = None,
    poly_assign: Optional[str] = None,
    min_overlap: Optional[float] = None,
    reg_agg: Optional[str] = None,
    min_seg_pxls: Optional[int] = None,
    reproj_vec: bool = False,
    vec_def_epsg: Optional[int] = None,
    *,
    seg_info: Optional[SegmentFeatureConfig] = None,
    out_format: Optional[str] = None,
    out_compress: Optional[Union[str, bool]] = "snappy",
    group_col: str = "train_grp",
    n_ref_col: str = "n_ref",
) -> pd.DataFrame:
    """
    Extract one regression sample per segment from the reference vector
    (points and/or polygons with the target attribute(s), default the
    config's ``vec_file``).

    Points are assigned to the segment under them and polygons to the
    segments they cover by at least ``min_overlap`` (``poly_assign='overlap'``)
    or that contain the segment centroid (``'centroid'``). When several
    reference values fall in one segment they are aggregated with the mean
    (``reg_agg='mean'``) or median, and their number is recorded in
    ``n_ref_col``. Invalid, small (``min_seg_pxls``) and (with the ``'drop'``
    missing policy) incomplete segments are removed. The table contains the
    segment ID, target(s), ``group_col`` (for group-aware splitting),
    ``n_ref_col``, the pixel count and the model features. A summary is in
    ``df.attrs['report']``.

    :param reg_cfg: Regression config (object or JSON path) with ``seg_info``.
    :param out_file: Optional output table (Parquet, snappy by default).
    :param vec_file: Optional reference vector (Default = ``reg_cfg.vec_file``).
    :param vec_lyr: Optional reference layer (Default = ``reg_cfg.vec_lyr``).
    :param poly_assign: Polygon rule (Default = ``seg_info.poly_assign``).
    :param min_overlap: Minimum covered fraction (Default = ``seg_info.min_overlap``).
    :param reg_agg: ``'mean'`` or ``'median'`` (Default = ``seg_info.reg_agg``).
    :param min_seg_pxls: Minimum segment size (Default = ``seg_info.min_seg_pxls``).
    :param reproj_vec: Reproject the reference vector to the segment CRS if needed.
    :param vec_def_epsg: Optional EPSG code for a vector with no CRS.
    :param seg_info: Optional ``SegmentFeatureConfig`` overriding the config's.
    :param out_format: Optional output table format.
    :param out_compress: Parquet compression (Default = ``'snappy'``).
    :param group_col: Group column name (Default = ``'train_grp'``).
    :param n_ref_col: Reference count column (Default = ``'n_ref'``).
    :return: Sample :class:`pandas.DataFrame`.

    .. code:: python

        import rsgispy.regression.regsegments as rseg

        samples = rseg.extract_reg_segment_samples(
            "reg_seg_config.json", "reg_seg_samples.parquet"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    cfg = _load_cfg(reg_cfg)
    seg = resolve_seg_info(cfg, seg_info)
    targets = list(cfg.target_cols)
    agg = seg.reg_agg if reg_agg is None else reg_agg
    if agg not in ("mean", "median"):
        raise rsgispy.RSGISPyException(f"reg_agg must be 'mean' or 'median', got {reg_agg!r}.")
    eff_vec = vec_file if vec_file is not None else cfg.vec_file
    eff_lyr = vec_lyr if vec_lyr is not None else cfg.vec_lyr
    if eff_vec is None:
        raise rsgispy.RSGISPyException("No reference vector (vec_file) was given.")
    eff_min_pxls = seg.min_seg_pxls if min_seg_pxls is None else int(min_seg_pxls)
    id_col = seg.id_col

    hits = assign_vectors_to_segments(
        seg, eff_vec, eff_lyr, value_cols=targets, poly_assign=poly_assign,
        min_overlap=min_overlap, reproj_vec=reproj_vec, vec_def_epsg=vec_def_epsg,
    )
    hits = hits.dropna(subset=targets)
    if hits.empty:
        raise rsgispy.RSGISPyException(
            "No reference points or polygons with target values fell on any segment."
        )
    grouped = hits.groupby(id_col)
    samples = grouped[targets].agg(agg).reset_index()
    samples[n_ref_col] = grouped.size().to_numpy()
    srt = hits.sort_values(["src_type", group_col], ascending=[False, True], kind="stable")
    grp = srt.drop_duplicates(subset=[id_col])[[id_col, group_col]]
    samples = samples.merge(grp, on=id_col)
    report = {"n_hits": int(len(hits)), "n_segments_hit": int(len(samples))}

    samples = attach_segment_features(seg, samples, eff_min_pxls, report)
    cols = [id_col] + targets + [group_col, n_ref_col, seg.pxl_count_col] + seg.model_features
    samples = samples[cols].sort_values(id_col).reset_index(drop=True)
    report["n_samples"] = int(len(samples))
    samples.attrs["report"] = report
    if out_file is not None:
        write_reg_samples(samples, out_file, out_format=out_format, out_compress=out_compress)
    return samples


def split_reg_segment_samples(
    samples_df: pd.DataFrame,
    train_prop: Optional[float] = None,
    test_prop: float = 0.2,
    valid_prop: float = 0.2,
    out_train_file: Optional[Union[str, os.PathLike]] = None,
    out_test_file: Optional[Union[str, os.PathLike]] = None,
    out_valid_file: Optional[Union[str, os.PathLike]] = None,
    group_col: Optional[str] = "train_grp",
    rnd_seed: Optional[int] = None,
    *,
    out_format: Optional[str] = None,
    out_compress: Optional[Union[str, bool]] = "snappy",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Randomly split segment regression samples into training, testing and
    validation sets, keeping all samples of a group (``group_col``, e.g. the
    segments from one reference polygon) in the same set. Raises an exception
    if there are fewer groups than sets with a non-zero proportion.

    :param samples_df: Sample table (e.g., from :func:`extract_reg_segment_samples`).
    :param train_prop: Training proportion (Default = ``1 - test_prop - valid_prop``).
    :param test_prop: Test proportion (Default = ``0.2``).
    :param valid_prop: Validation proportion (Default = ``0.2``).
    :param out_train_file: Optional training table output path.
    :param out_test_file: Optional test table output path.
    :param out_valid_file: Optional validation table output path.
    :param group_col: Group column (``None`` = each sample its own group).
    :param rnd_seed: Optional random seed.
    :param out_format: Optional output table format.
    :param out_compress: Parquet compression (Default = ``'snappy'``).
    :return: ``(train_df, test_df, valid_df)``.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    # pylint: disable=import-outside-toplevel
    from rsgispy.classification.classsegments import split_class_segment_samples

    props = [
        1.0 - test_prop - valid_prop if train_prop is None else train_prop,
        test_prop,
        valid_prop,
    ]
    n_groups = (
        samples_df[group_col].nunique()
        if group_col is not None and group_col in samples_df.columns
        else len(samples_df)
    )
    n_sets = sum(1 for p in props if p > 0)
    if n_groups < n_sets:
        raise rsgispy.RSGISPyException(
            f"The samples have {n_groups} group(s), fewer than the {n_sets} sets "
            "requested; add reference data or reduce test_prop / valid_prop."
        )
    tmp = samples_df.copy()
    tmp["_strat"] = 0
    parts = split_class_segment_samples(
        tmp,
        train_prop=train_prop,
        test_prop=test_prop,
        valid_prop=valid_prop,
        stratify_col="_strat",
        group_col=group_col,
        rnd_seed=rnd_seed,
        min_cls_smpls_warn=0,
    )
    outs: List[pd.DataFrame] = []
    for part, out_path in zip(parts, (out_train_file, out_test_file, out_valid_file)):
        part = part.drop(columns="_strat")
        if out_path is not None:
            write_reg_samples(part, out_path, out_format=out_format, out_compress=out_compress)
        outs.append(part)
    return outs[0], outs[1], outs[2]


def extract_and_split_reg_segment_samples(
    reg_cfg: RegConfigInput,
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
    Extract segment regression samples with :func:`extract_reg_segment_samples`
    and split them with :func:`split_reg_segment_samples`.

    :param reg_cfg: Regression config with ``seg_info``.
    :param out_train_file: Optional training table output path.
    :param out_test_file: Optional test table output path.
    :param out_valid_file: Optional validation table output path.
    :param out_samples_file: Optional output path for all samples.
    :param test_prop: Test proportion (Default = ``0.2``).
    :param valid_prop: Validation proportion (Default = ``0.2``).
    :param rnd_seed: Optional random seed.
    :param kwargs: Further arguments for :func:`extract_reg_segment_samples`.
    :return: ``(train_df, test_df, valid_df)``.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    samples = extract_reg_segment_samples(reg_cfg, out_file=out_samples_file, **kwargs)
    return split_reg_segment_samples(
        samples,
        test_prop=test_prop,
        valid_prop=valid_prop,
        out_train_file=out_train_file,
        out_test_file=out_test_file,
        out_valid_file=out_valid_file,
        group_col=kwargs.get("group_col", "train_grp"),
        rnd_seed=rnd_seed,
    )


def apply_sklearn_regressor_segments(
    reg_cfg: RegConfigInput,
    sk_regressor: Union[Any, str, os.PathLike],
    out_vec_file: Union[str, os.PathLike],
    out_quantiles: Optional[Sequence[float]] = None,
    **kwargs: Any,
) -> None:
    """
    Apply a trained ``scikit-learn`` (or compatible) regressor to the segment
    feature layer in chunks (reading only the segment ID, valid flag and model
    feature columns, never the geometry) and write a vector layer with one
    ``pred_<target>`` column per target (multi-target supported). Segments
    that are not valid, or have missing features under the ``'drop'`` policy,
    get ``out_no_data_val`` (NaN by default).

    :param reg_cfg: Regression config (object or JSON path) with ``seg_info``.
    :param sk_regressor: Fitted regressor or saved ``.joblib`` model path.
    :param out_vec_file: Output vector layer (GeoParquet for ``.parquet``, or
                         any vector format).
    :param out_quantiles: Optional quantiles (in ``(0, 1)``) from the per-tree
                          predictions of a univariate tree ensemble (e.g.
                          ``RandomForestRegressor``), written as
                          ``pred_<target>_q<quantile>`` columns.
    :param kwargs: Further options: ``out_uncert_method`` (``'std'``,
                   ``'iqr'`` or ``'cv'``: add ``pred_<target>_<method>``
                   uncertainty columns from the spread of the ensemble
                   members, as ``out_uncert_img`` of
                   :func:`~rsgispy.regression.regsklearn.apply_sklearn_regressor`;
                   Default = ``None``), ``seg_info``, ``in_vec_file``,
                   ``copy_cols`` (Default = all input columns; ``[]`` = ID
                   only), ``out_prefix`` (Default = ``'pred_'``),
                   ``out_no_data_val``, ``missing``, ``fill_value``,
                   ``scaler``, ``chunk_size``, ``out_format``,
                   ``out_compress``, ``overwrite``, ``use_pbar``.

    .. code:: python

        import rsgispy.regression.regsegments as rseg

        rseg.apply_sklearn_regressor_segments(
            "reg_seg_config.json", "rf_reg.joblib", "segs_canopy.parquet",
            out_quantiles=[0.05, 0.95],
        )
    """
    # pylint: disable=import-outside-toplevel
    from rsgispy.regression.regsklearn import load_sklearn_regressor

    check_sklearn_available("apply_sklearn_regressor_segments")
    is_path = isinstance(sk_regressor, (str, os.PathLike))
    estimator = load_sklearn_regressor(sk_regressor) if is_path else sk_regressor
    if is_path:
        kwargs.setdefault("model_file", sk_regressor)
    apply_regressor_vector(
        reg_cfg, estimator, out_vec_file, out_quantiles=out_quantiles, **kwargs
    )


def _make_gbdt_apply(backend_name: str, label: str) -> Any:
    """Build a public GBDT segment regression apply function."""
    func_name = f"apply_{backend_name}_regressor_segments"

    def _apply(
        reg_cfg: RegConfigInput,
        regressor: Union[Any, str, os.PathLike],
        out_vec_file: Union[str, os.PathLike],
        *,
        use_gpu: bool = False,
        gpu_device_id: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        backend = get_gbdt_backend("regression", backend_name)
        apply_gbdt_regressor_segments(
            backend, func_name, reg_cfg, regressor, out_vec_file,
            use_gpu=use_gpu, gpu_device_id=gpu_device_id, **kwargs,
        )

    _apply.__name__ = func_name
    _apply.__qualname__ = func_name
    _apply.__doc__ = f"""
    Apply a trained {label} regressor (univariate or multi-target) to the
    segment feature layer in chunks and write a vector layer with one
    ``pred_<target>`` column per target (see
    :func:`apply_sklearn_regressor_segments`). The ``'nan'`` missing policy
    can be used as {label} handles missing values.

    :param reg_cfg: Regression config (object or JSON path) with ``seg_info``.
    :param regressor: Fitted {label} regressor or saved model path.
    :param out_vec_file: Output vector layer.
    :param use_gpu: Predict on the GPU if available (Default = ``False``).
    :param gpu_device_id: Optional GPU device ID.
    :param kwargs: Further options (``seg_info``, ``copy_cols``, ``out_prefix``,
                   ``out_no_data_val``, ``missing``, ``scaler``, ``chunk_size``,
                   ``out_format``, ``out_compress``, ``overwrite``, ``use_pbar``).

    .. code:: python

        import rsgispy.regression.regsegments as rseg

        rseg.{func_name}(
            "reg_seg_config.json", "model.json", "segs_canopy.parquet"
        )
    """
    return _apply


apply_xgboost_regressor_segments = _make_gbdt_apply("xgboost", "XGBoost")
apply_lightgbm_regressor_segments = _make_gbdt_apply("lightgbm", "LightGBM")
apply_catboost_regressor_segments = _make_gbdt_apply("catboost", "CatBoost")


def rasterise_segment_regression(
    reg_cfg: Optional[RegConfigInput],
    reg_vec_file: Union[str, os.PathLike],
    output_img: Union[str, os.PathLike],
    pred_cols: Optional[Sequence[str]] = None,
    label_img: Optional[Union[str, os.PathLike]] = None,
    out_no_data_val: float = -9999.0,
    *,
    reg_vec_lyr: Optional[str] = None,
    id_col: Optional[str] = None,
    gdalformat: str = "GTIFF",
    datatype: int = rsgispy.TYPE_32FLOAT,
    calc_stats: bool = True,
    chunk_size: int = 100_000,
    use_pbar: bool = True,
    creation_opts: Optional[Any] = None,
) -> None:
    """
    Rasterise segment regression predictions onto the segment label raster's
    pixel grid (one band per prediction column, named after the column) by
    mapping segment IDs to values (no geometry is read or burnt).

    :param reg_cfg: Regression config with ``seg_info`` (or ``None`` with
                    ``label_img`` and ``pred_cols`` given).
    :param reg_vec_file: Segment prediction layer/table.
    :param output_img: Output raster.
    :param pred_cols: Prediction columns (Default = ``pred_<target>`` for each
                      config target).
    :param label_img: Segment label raster (Default = ``seg_info.label_img``).
    :param out_no_data_val: Output no-data value (Default = ``-9999``); NaN
                            predictions are written as no data.
    :param reg_vec_lyr: Optional layer name.
    :param id_col: Segment ID column (Default = ``seg_info.id_col``).
    :param gdalformat: Output GDAL format (Default = ``'GTIFF'``).
    :param datatype: Output datatype (Default = :data:`rsgispy.TYPE_32FLOAT`).
    :param calc_stats: Calculate statistics and pyramids (Default = ``True``).
    :param chunk_size: Rows per chunk when reading the layer.
    :param use_pbar: Display progress bars.
    :param creation_opts: Optional GDAL creation options.

    .. code:: python

        import rsgispy.regression.regsegments as rseg

        rseg.rasterise_segment_regression(
            "reg_seg_config.json", "segs_canopy.parquet", "canopy_cover.tif"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    # pylint: disable=import-outside-toplevel
    from rsgispy.zonalstats.labels import rasterise_label_values

    cfg = _load_cfg(reg_cfg) if reg_cfg is not None else None
    seg = getattr(cfg, "seg_info", None)
    cols = (
        list(pred_cols)
        if pred_cols is not None
        else ([f"pred_{t}" for t in cfg.target_cols] if cfg is not None else None)
    )
    if not cols:
        raise rsgispy.RSGISPyException("pred_cols must be given (no config).")
    eff_id = id_col or (seg.id_col if seg is not None else "seg_id")
    eff_label = label_img or (seg.label_img if seg is not None else None)
    if eff_label is None:
        raise rsgispy.RSGISPyException("label_img must be given (no seg_info).")
    parts = list(
        iter_vec_chunks(reg_vec_file, [eff_id] + cols, vec_lyr=reg_vec_lyr, chunk_size=chunk_size)
    )
    tbl = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=[eff_id] + cols)
    vals = tbl[cols].to_numpy(dtype=np.float64, copy=True)
    vals[~np.isfinite(vals)] = out_no_data_val
    rasterise_label_values(
        eff_label,
        tbl[eff_id].to_numpy(),
        vals,
        output_img,
        label_no_data_val=seg.label_no_data_val if seg is not None else 0,
        out_no_data_val=out_no_data_val,
        band_names=cols,
        gdalformat=gdalformat,
        datatype=datatype,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
        creation_opts=creation_opts,
    )
