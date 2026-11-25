"""
Shared Optuna parameter sampling, dataset preparation, binary/multi-class
evaluation metrics, and block-wise raster classification utilities reused
across :mod:`rsgispy.classification`.
"""

from contextlib import ExitStack
import os
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)
import warnings

import numpy as np
import pandas as pd

import rsgispy
from rsgispy.classification._helpers import (
    seg_allows_nan,
    ClsConfigInput,
    SamplesInput,
    calc_split_metrics,
    coerce_cls_config,
    extract_xy,
    finalise_thematic_output_img,
    load_samples_df,
    resolve_apply_band_specs,
    resolve_feature_cols,
    resolve_out_nodata,
    select_out_int_dtype,
)
from rsgispy.classification.classdata import (
    ClassificationConfig,
    ClassInfoObj,
    ImageBandInfo,
    recode_internal_to_out_ids,
)
from rsgispy.imageprocess import (
    ensure_file_exists,
    iter_dataset_windows,
    open_output_dataset,
)
from rsgispy.imageprocess import normalise_creation_options
from rsgispy.imageutils.colourtable import set_img_not_thematic
from rsgispy.imageutils.stats import pop_img_stats
from rsgispy.tools._ml_data import (
    open_apply_datasets as _open_apply_datasets,
    read_valid_roi_block as _read_valid_roi_block,
    resolve_estimator_feature_cols as _resolve_estimator_feature_cols,
)
from rsgispy.tools._sklearn import check_sklearn_estimator_fitted, sklearn
from rsgispy.tools.files import ensure_parent_dir


def resolve_ordered_classes(
    class_info: Mapping[str, ClassInfoObj],
    use_internal_id: bool,
) -> List[Tuple[str, ClassInfoObj, int]]:
    """Return class definitions ordered consistently by training target ID."""
    items: List[Tuple[str, ClassInfoObj, int]] = []
    for cls_name, obj in class_info.items():
        train_id = (
            int(obj.internal_id)  # type: ignore[arg-type]
            if use_internal_id
            else int(obj.out_id)
        )
        items.append((cls_name, obj, train_id))
    items.sort(key=lambda entry: entry[2])
    return items


def align_proba_columns(
    probs: np.ndarray,
    estimator_classes: Optional[Sequence[Any]],
    ordered_train_ids: Sequence[int],
) -> np.ndarray:
    """Align ``predict_proba`` columns onto ``ordered_train_ids`` order."""
    n_rows = probs.shape[0]
    n_classes = len(ordered_train_ids)
    if probs.ndim == 1:
        probs = probs[:, None]
    if estimator_classes is None:
        if probs.shape[1] != n_classes:
            raise rsgispy.RSGISPyException(
                f"predict_proba returned {probs.shape[1]} columns, expected "
                f"{n_classes}."
            )
        return probs.astype(np.float32, copy=False)

    aligned = np.zeros((n_rows, n_classes), dtype=np.float32)
    cls_to_col = {int(c): idx for idx, c in enumerate(ordered_train_ids)}
    for src_idx, cls_val in enumerate(estimator_classes):
        int_cls = int(cls_val)
        if int_cls in cls_to_col and src_idx < probs.shape[1]:
            aligned[:, cls_to_col[int_cls]] = probs[:, src_idx].astype(
                np.float32, copy=False
            )
    return aligned


def resolve_binary_classes(
    class_info: Mapping[str, ClassInfoObj],
    use_internal_id: bool,
    pos_class: Optional[Union[str, int]] = None,
) -> Tuple[Tuple[str, ClassInfoObj, int], Tuple[str, ClassInfoObj, int]]:
    """
    Validate a binary classification configuration and return
    ``(neg_entry, pos_entry)`` tuples ``(cls_name, ClassInfoObj, train_id)``.
    """
    ordered = resolve_ordered_classes(class_info, use_internal_id)
    if len(ordered) != 2:
        raise rsgispy.RSGISPyException(
            f"Binary classification requires exactly 2 classes in cls_cfg, "
            f"found {len(ordered)}."
        )
    if pos_class is None:
        return ordered[0], ordered[1]

    if isinstance(pos_class, str):
        for idx, entry in enumerate(ordered):
            if entry[0] == pos_class:
                return ordered[1 - idx], ordered[idx]
        raise rsgispy.RSGISPyException(
            f"pos_class '{pos_class}' not found in classes "
            f"{[e[0] for e in ordered]}."
        )

    if isinstance(pos_class, (int, np.integer)) and not isinstance(
        pos_class, bool
    ):
        pos_int = int(pos_class)
        for idx, entry in enumerate(ordered):
            if entry[2] == pos_int or int(entry[1].out_id) == pos_int:
                return ordered[1 - idx], ordered[idx]
        raise rsgispy.RSGISPyException(
            f"pos_class ID {pos_int} not found in binary class definitions."
        )

    raise rsgispy.RSGISPyException(
        f"pos_class must be a class name string or integer ID, got {pos_class!r}."
    )


def prepare_cls_splits(
    cls_cfg: ClsConfigInput,
    train_data: SamplesInput,
    test_data: Optional[SamplesInput] = None,
    valid_data: Optional[SamplesInput] = None,
    feature_cols: Optional[Sequence[str]] = None,
    use_internal_id: bool = True,
    out_id_col: Optional[str] = None,
    internal_id_col: Optional[str] = None,
    binary: Optional[bool] = None,
) -> Tuple[
    Optional[ClassificationConfig],
    Dict[str, ClassInfoObj],
    List[str],
    pd.DataFrame,
    np.ndarray,
    Optional[Tuple[pd.DataFrame, np.ndarray]],
    Optional[Tuple[pd.DataFrame, np.ndarray]],
]:
    """Load config, validate class count, and extract train/test/valid splits."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    cfg, class_info, eff_out_col, eff_int_col, cls_name_col = coerce_cls_config(
        cls_cfg, out_id_col=out_id_col, internal_id_col=internal_id_col
    )
    n_classes = len(class_info)
    if binary is True and n_classes != 2:
        raise rsgispy.RSGISPyException(
            f"Binary classifier requires exactly 2 classes in cls_cfg, found {n_classes}."
        )
    if binary is False and n_classes < 2:
        raise rsgispy.RSGISPyException(
            f"Multi-class classifier requires at least 2 classes in cls_cfg, "
            f"found {n_classes}."
        )

    target_col = eff_int_col if use_internal_id else eff_out_col
    valid_ids = [
        int(obj.internal_id) if use_internal_id else int(obj.out_id)  # type: ignore[arg-type]
        for obj in class_info.values()
    ]

    train_df = load_samples_df(train_data, "train_data")
    feat_cols = resolve_feature_cols(
        cfg,
        train_df,
        feature_cols,
        excluded_cols=[eff_out_col, eff_int_col, cls_name_col],
    )
    train_x, train_y = extract_xy(
        train_df, feat_cols, target_col, "train_data", valid_ids=valid_ids,
        allow_nan=seg_allows_nan(cfg),
    )

    test_xy: Optional[Tuple[pd.DataFrame, np.ndarray]] = None
    if test_data is not None:
        test_df = load_samples_df(test_data, "test_data")
        test_xy = extract_xy(
            test_df, feat_cols, target_col, "test_data", valid_ids=valid_ids,
            allow_nan=seg_allows_nan(cfg),
        )

    valid_xy: Optional[Tuple[pd.DataFrame, np.ndarray]] = None
    if valid_data is not None:
        valid_df = load_samples_df(valid_data, "valid_data")
        valid_xy = extract_xy(
            valid_df, feat_cols, target_col, "valid_data", valid_ids=valid_ids,
            allow_nan=seg_allows_nan(cfg),
        )

    return cfg, class_info, feat_cols, train_x, train_y, test_xy, valid_xy


def calc_cls_split_metrics(
    fitted_est: Any,
    x_df: pd.DataFrame,
    y_true: np.ndarray,
    prefix: str,
    binary: bool = False,
    pos_train_id: int = 1,
    predict_kwargs: Optional[Mapping[str, Any]] = None,
) -> Dict[str, float]:
    """
    Compute standard classification metrics and, when ``binary=True``,
    positive-class precision, recall, F1, ROC-AUC, and log-loss.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    p_kwargs = dict(predict_kwargs) if predict_kwargs else {}
    metrics = calc_split_metrics(
        fitted_est, x_df, y_true, prefix, predict_kwargs=p_kwargs
    )
    if not binary:
        return metrics

    preds = np.asarray(
        fitted_est.predict(x_df, **p_kwargs) if p_kwargs else fitted_est.predict(x_df)
    ).ravel()
    y_bin = (y_true == pos_train_id).astype(np.int64)
    p_bin = (preds == pos_train_id).astype(np.int64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prec = float(
            sklearn.metrics.precision_score(y_bin, p_bin, zero_division=0)
        )
        rec = float(sklearn.metrics.recall_score(y_bin, p_bin, zero_division=0))
        f1_bin = float(
            sklearn.metrics.f1_score(y_bin, p_bin, zero_division=0)
        )
    metrics[f"{prefix}_precision"] = prec
    metrics[f"{prefix}_recall"] = rec
    metrics[f"{prefix}_f1_binary"] = f1_bin

    if hasattr(fitted_est, "predict_proba"):
        raw_probs = np.asarray(
            fitted_est.predict_proba(x_df, **p_kwargs)
            if p_kwargs
            else fitted_est.predict_proba(x_df),
            dtype=np.float64,
        )
        est_classes = list(getattr(fitted_est, "classes_", []))
        if raw_probs.ndim == 2 and raw_probs.shape[1] >= 2:
            if est_classes and pos_train_id in [int(c) for c in est_classes]:
                pos_col = [int(c) for c in est_classes].index(pos_train_id)
            else:
                pos_col = 1
            pos_probs = raw_probs[:, pos_col]
        else:
            pos_probs = raw_probs.ravel()

        if np.unique(y_bin).size == 2:
            metrics[f"{prefix}_roc_auc"] = float(
                sklearn.metrics.roc_auc_score(y_bin, pos_probs)
            )
            metrics[f"{prefix}_logloss"] = float(
                sklearn.metrics.log_loss(y_bin, np.clip(pos_probs, 1e-7, 1.0 - 1e-7))
            )
    return metrics


def apply_classifier_raster(
    cls_cfg: ClsConfigInput,
    estimator: Any,
    in_msk_img: Union[str, os.PathLike],
    img_msk_val: int,
    output_img: Optional[Union[str, os.PathLike]] = None,
    out_score_img: Optional[Union[str, os.PathLike]] = None,
    img_band_info: Optional[Sequence[ImageBandInfo]] = None,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    out_no_data_val: Optional[int] = None,
    class_clr_names: bool = True,
    use_internal_id: bool = True,
    recode_to_out_ids: bool = True,
    score_max_only: bool = False,
    score_no_data_val: float = -1.0,
    calc_stats: bool = True,
    block_size: Optional[int] = None,
    creation_opts: Optional[Mapping[str, Any]] = None,
    use_pbar: bool = True,
    binary: bool = False,
    pos_class: Optional[Union[str, int]] = None,
    pred_threshold: Optional[float] = None,
    score_all_classes: bool = False,
    predict_kwargs: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Shared block-wise image classification engine for binary and multi-class
    classifiers supporting ``predict()`` and ``predict_proba()``.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    if output_img is None and out_score_img is None:
        raise rsgispy.RSGISPyException(
            "At least one of output_img or out_score_img must be specified."
        )
    p_kwargs = dict(predict_kwargs) if predict_kwargs else {}
    check_sklearn_estimator_fitted(estimator, label="classifier")

    if pred_threshold is not None:
        thresh_val = float(pred_threshold)
        if thresh_val <= 0.0 or thresh_val >= 1.0:
            raise rsgispy.RSGISPyException(
                f"pred_threshold must be strictly between 0 and 1, got {pred_threshold}."
            )
        if not hasattr(estimator, "predict_proba"):
            raise rsgispy.RSGISPyException(
                "Classifier does not support predict_proba(), which is required "
                "when pred_threshold is set."
            )

    if out_score_img is not None and not hasattr(estimator, "predict_proba"):
        raise rsgispy.RSGISPyException(
            "Classifier does not support predict_proba(), which is required "
            "to output out_score_img."
        )

    cfg, class_info, _, _, _ = coerce_cls_config(cls_cfg)
    band_specs = resolve_apply_band_specs(cfg, img_band_info)
    ordered_classes = resolve_ordered_classes(class_info, use_internal_id)
    if binary:
        neg_entry, pos_entry = resolve_binary_classes(
            class_info, use_internal_id, pos_class=pos_class
        )
        pos_train_id = pos_entry[2]
        neg_train_id = neg_entry[2]
        pos_col_idx = [e[2] for e in ordered_classes].index(pos_train_id)
    else:
        if len(ordered_classes) < 2:
            raise rsgispy.RSGISPyException(
                "Multi-class classification requires at least 2 classes in cls_cfg."
            )
        pos_train_id = ordered_classes[-1][2]
        neg_train_id = ordered_classes[0][2]
        pos_col_idx = 0

    ordered_names = [entry[0] for entry in ordered_classes]
    ordered_train_ids = [entry[2] for entry in ordered_classes]
    should_recode = bool(use_internal_id and recode_to_out_ids)
    output_cls_ids = (
        [int(entry[1].out_id) for entry in ordered_classes]
        if should_recode
        else list(ordered_train_ids)
    )
    eff_nodata = resolve_out_nodata(out_no_data_val, output_cls_ids)

    if datatype is None:
        out_rsgis_dtype, out_np_dtype = select_out_int_dtype(
            output_cls_ids + [eff_nodata]
        )
        out_rio_dtype = rsgispy.get_rasterio_datatype(out_rsgis_dtype)
    else:
        out_rio_dtype = rsgispy.get_rasterio_datatype(datatype)
        out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(datatype))

    msk_path = os.fspath(in_msk_img)
    ensure_file_exists(msk_path, label="Input mask image")
    c_opts = normalise_creation_options(creation_opts)
    use_df_input, df_cols, band_specs = _resolve_estimator_feature_cols(
        estimator, band_specs
    )

    out_cls_path = (
        ensure_parent_dir(output_img) if output_img is not None else None
    )
    out_scr_path = (
        ensure_parent_dir(out_score_img) if out_score_img is not None else None
    )

    with ExitStack() as stack:
        msk_ds, datasets, grid, all_compatible = _open_apply_datasets(
            stack, msk_path, band_specs
        )

        cls_dst = None
        if out_cls_path is not None:
            cls_dst = open_output_dataset(
                stack=stack,
                output_img=out_cls_path,
                gdalformat=gdalformat,
                out_rio_dtype=out_rio_dtype,
                grid=grid,
                count=1,
                creation_opts=c_opts,
            )
            cls_dst.nodata = eff_nodata

        scr_dst = None
        if out_scr_path is not None:
            if binary and not score_all_classes and not score_max_only:
                n_score_bands = 1
            elif score_max_only:
                n_score_bands = 1
            else:
                n_score_bands = len(ordered_classes)

            scr_dst = open_output_dataset(
                stack=stack,
                output_img=out_scr_path,
                gdalformat=gdalformat,
                out_rio_dtype="float32",
                grid=grid,
                count=n_score_bands,
                creation_opts=c_opts,
            )
            scr_dst.nodata = float(score_no_data_val)
            if score_max_only:
                scr_dst.set_band_description(1, "max_class_probability")
            elif binary and not score_all_classes:
                scr_dst.set_band_description(1, f"prob_{ordered_names[pos_col_idx]}")
            else:
                for b_i, cls_name in enumerate(ordered_names, start=1):
                    scr_dst.set_band_description(b_i, cls_name)

        base_msk_col, base_msk_row = grid.offsets[msk_path]
        est_classes = getattr(estimator, "classes_", None)
        if est_classes is not None and len(est_classes) > 0:
            est_id_set: set = set()
            for cls_val in est_classes:
                try:
                    est_id_set.add(int(cls_val))
                except (TypeError, ValueError):
                    pass
            expected_id_set = {int(c) for c in ordered_train_ids}
            if est_id_set and not est_id_set.intersection(expected_id_set):
                alt_col = "out_id" if use_internal_id else "internal_id"
                alt_ids = {
                    int(
                        entry[1].out_id
                        if use_internal_id
                        else entry[1].internal_id  # type: ignore[arg-type]
                    )
                    for entry in ordered_classes
                }
                hint = (
                    f" Hint: estimator classes match cls_cfg {alt_col} values "
                    f"{sorted(alt_ids)}; set use_internal_id="
                    f"{not use_internal_id} to match how the classifier was "
                    "trained."
                    if est_id_set.intersection(alt_ids)
                    else ""
                )
                raise rsgispy.RSGISPyException(
                    f"Classifier trained class IDs {sorted(est_id_set)} do "
                    f"not match any expected target class IDs "
                    f"{sorted(expected_id_set)} (use_internal_id="
                    f"{use_internal_id}).{hint}"
                )

        total_valid_pxls = 0
        for out_win in iter_dataset_windows(
            grid.height, grid.width, block_size=block_size, use_pbar=use_pbar
        ):
            win_h = int(out_win.height)
            win_w = int(out_win.width)
            valid_idx, x_valid = _read_valid_roi_block(
                msk_ds=msk_ds,
                msk_path=msk_path,
                img_msk_val=img_msk_val,
                band_specs=band_specs,
                datasets=datasets,
                grid=grid,
                out_win=out_win,
                all_compatible=all_compatible,
                use_df_input=use_df_input,
                df_cols=df_cols,
                base_msk_col=base_msk_col,
                base_msk_row=base_msk_row,
            )
            total_valid_pxls += int(valid_idx.size)

            aligned_probs: Optional[np.ndarray] = None
            need_probs = (
                scr_dst is not None
                or (cls_dst is not None and binary and pred_threshold is not None)
            )
            if valid_idx.size > 0 and x_valid is not None and need_probs:
                raw_probs = np.asarray(
                    estimator.predict_proba(x_valid, **p_kwargs)
                    if p_kwargs
                    else estimator.predict_proba(x_valid),
                    dtype=np.float32,
                )
                aligned_probs = align_proba_columns(
                    raw_probs, est_classes, ordered_train_ids
                )

            if cls_dst is not None:
                cls_flat = np.full(
                    win_h * win_w, eff_nodata, dtype=out_np_dtype
                )
                if valid_idx.size > 0 and x_valid is not None:
                    if binary and pred_threshold is not None and aligned_probs is not None:
                        pos_probs = aligned_probs[:, pos_col_idx]
                        preds = np.where(
                            pos_probs >= float(pred_threshold),
                            pos_train_id,
                            neg_train_id,
                        ).astype(np.int64)
                    else:
                        preds = np.asarray(
                            estimator.predict(x_valid, **p_kwargs)
                            if p_kwargs
                            else estimator.predict(x_valid),
                            dtype=np.int64,
                        ).ravel()
                    if should_recode:
                        preds = recode_internal_to_out_ids(
                            preds,
                            class_info,
                            out_dtype=out_np_dtype,
                            keep_unmapped=False,
                        )
                    else:
                        preds = preds.astype(out_np_dtype, copy=False)
                    cls_flat[valid_idx] = preds
                cls_dst.write(cls_flat.reshape((win_h, win_w)), 1, window=out_win)

            if scr_dst is not None:
                scr_flat = np.full(
                    (scr_dst.count, win_h * win_w),
                    float(score_no_data_val),
                    dtype=np.float32,
                )
                if valid_idx.size > 0 and aligned_probs is not None:
                    if score_max_only:
                        scr_flat[0, valid_idx] = np.max(aligned_probs, axis=1)
                    elif binary and not score_all_classes:
                        scr_flat[0, valid_idx] = aligned_probs[:, pos_col_idx]
                    else:
                        scr_flat[:, valid_idx] = aligned_probs.T
                scr_dst.write(
                    scr_flat.reshape((scr_dst.count, win_h, win_w)),
                    window=out_win,
                )

        if total_valid_pxls == 0:
            warnings.warn(
                f"No valid pixels were classified (in_msk_img='{msk_path}', "
                f"img_msk_val={img_msk_val}). Output raster contains only "
                f"no-data values ({eff_nodata}); check that img_msk_val "
                "is present in in_msk_img and that source bands are not "
                "masked out by band nodata metadata.",
                UserWarning,
                stacklevel=2,
            )

    if out_cls_path is not None:
        clr_lut: Optional[Dict[int, List[int]]] = None
        if class_clr_names and out_np_dtype in (
            np.dtype(np.uint8),
            np.dtype(np.uint16),
        ):
            clr_lut = {
                int(
                    obj.out_id if should_recode else obj.internal_id  # type: ignore[arg-type]
                ): [
                    obj.red,
                    obj.green,
                    obj.blue,
                    255,
                ]
                for obj in class_info.values()
            }
        finalise_thematic_output_img(
            out_cls_path=out_cls_path,
            eff_nodata=eff_nodata,
            out_rio_dtype=out_rio_dtype,
            out_np_dtype=out_np_dtype,
            calc_stats=calc_stats,
            add_clr_tab=False,
            clr_lut=clr_lut,
            use_pbar=use_pbar,
        )

    if out_scr_path is not None:
        set_img_not_thematic(out_scr_path)
        if calc_stats:
            pop_img_stats(
                out_scr_path,
                no_data_val=float(score_no_data_val),
                calc_pyramids=True,
                use_pbar=use_pbar,
            )


def _rgb_hex(obj: ClassInfoObj) -> str:
    """Return a ``#rrggbb`` colour string for a class."""
    return f"#{int(obj.red):02x}{int(obj.green):02x}{int(obj.blue):02x}"


def apply_classifier_vector(
    cls_cfg: ClsConfigInput,
    estimator: Any,
    out_vec_file: Union[str, os.PathLike],
    *,
    seg_info: Optional[Any] = None,
    in_vec_file: Optional[Union[str, os.PathLike]] = None,
    in_vec_lyr: Optional[str] = None,
    geom_vec_file: Optional[Union[str, os.PathLike]] = None,
    geom_vec_lyr: Optional[str] = None,
    out_vec_lyr: Optional[str] = None,
    out_format: Optional[str] = None,
    out_compress: Optional[Union[str, bool]] = "snappy",
    copy_cols: Optional[Sequence[str]] = None,
    out_prefix: str = "cls_",
    out_probs: Optional[str] = None,
    use_internal_id: bool = True,
    recode_to_out_ids: bool = True,
    out_no_data_val: Optional[int] = None,
    missing: Optional[str] = None,
    fill_value: Optional[float] = None,
    scaler: Optional[Any] = None,
    chunk_size: int = 100_000,
    binary: bool = False,
    pos_class: Optional[Union[str, int]] = None,
    pred_threshold: Optional[float] = None,
    predict_kwargs: Optional[Mapping[str, Any]] = None,
    model_file: Optional[Union[str, os.PathLike]] = None,
    overwrite: bool = False,
    use_pbar: bool = True,
) -> None:
    """
    Shared chunked engine applying a fitted classifier to a segment feature
    layer (or attribute table) and writing a vector layer with the class
    output columns (``<prefix>id``, ``<prefix>name``, ``<prefix>colour``,
    ``<prefix>red``/``green``/``blue`` and optional probability columns).
    Only the ID, valid flag and model feature columns are read for
    prediction; geometry is passed through without being decoded.
    """
    # pylint: disable=too-many-arguments,too-many-locals,too-many-branches
    # pylint: disable=too-many-statements,import-outside-toplevel
    from rsgispy.ml.segments import resolve_seg_info
    from rsgispy.ml.vector import (
        build_provenance,
        map_vector_chunks,
        model_input,
        prepare_feature_matrix,
    )

    p_kwargs = dict(predict_kwargs) if predict_kwargs else {}
    check_sklearn_estimator_fitted(estimator, label="classifier")
    probs_opt = None if out_probs is None else str(out_probs).lower()
    if probs_opt not in (None, "max", "all", "pos"):
        raise rsgispy.RSGISPyException(
            f"out_probs must be None, 'max', 'all' or 'pos', got {out_probs!r}."
        )
    if probs_opt == "pos" and not binary:
        raise rsgispy.RSGISPyException("out_probs='pos' requires a binary classifier.")
    if pred_threshold is not None and not 0.0 < float(pred_threshold) < 1.0:
        raise rsgispy.RSGISPyException(
            f"pred_threshold must be strictly between 0 and 1, got {pred_threshold}."
        )
    need_probs = probs_opt is not None or (binary and pred_threshold is not None)
    if need_probs and not hasattr(estimator, "predict_proba"):
        raise rsgispy.RSGISPyException(
            "Classifier does not support predict_proba(), which is required for "
            "probability outputs or pred_threshold."
        )

    cfg, class_info, _, _, _ = coerce_cls_config(cls_cfg)
    seg = resolve_seg_info(cfg, seg_info)
    ordered_classes = resolve_ordered_classes(class_info, use_internal_id)
    pos_train_id = neg_train_id = -1
    if binary:
        neg_entry, pos_entry = resolve_binary_classes(
            class_info, use_internal_id, pos_class=pos_class
        )
        pos_train_id, neg_train_id = pos_entry[2], neg_entry[2]
    elif len(ordered_classes) < 2:
        raise rsgispy.RSGISPyException(
            "Multi-class classification requires at least 2 classes in cls_cfg."
        )
    ordered_names = [entry[0] for entry in ordered_classes]
    ordered_train_ids = [entry[2] for entry in ordered_classes]
    pos_col_idx = ordered_train_ids.index(pos_train_id) if binary else 0
    should_recode = bool(use_internal_id and recode_to_out_ids)
    lut = {
        entry[2]: (
            int(entry[1].out_id) if should_recode else int(entry[2]),
            entry[0],
            entry[1],
        )
        for entry in ordered_classes
    }
    output_ids = [val[0] for val in lut.values()]
    eff_nodata = resolve_out_nodata(out_no_data_val, output_ids)

    est_classes = getattr(estimator, "classes_", None)
    if est_classes is not None:
        unknown = {int(c) for c in est_classes} - set(ordered_train_ids)
        if unknown:
            raise rsgispy.RSGISPyException(
                f"Classifier trained class IDs {sorted(unknown)} are not defined in "
                f"cls_cfg (use_internal_id={use_internal_id})."
            )

    est_names = getattr(estimator, "feature_names_in_", None)
    feat_cols = (
        [str(c) for c in est_names] if est_names is not None else seg.model_features
    )
    if not feat_cols:
        raise rsgispy.RSGISPyException("No model feature columns are defined.")
    eff_missing = seg.missing if missing is None else missing
    eff_fill = seg.fill_value if fill_value is None else float(fill_value)

    if in_vec_file is None:
        in_path, in_lyr = seg.feature_source
        if seg.feat_table_file is not None and geom_vec_file is None:
            geom_vec_file, geom_vec_lyr = seg.seg_vec_file, seg.seg_vec_lyr
    else:
        in_path, in_lyr = os.fspath(in_vec_file), in_vec_lyr

    read_cols = list(dict.fromkeys([seg.valid_col] + feat_cols))
    prob_names: List[str] = []
    if probs_opt == "max":
        prob_names = [f"{out_prefix}prob_max"]
    elif probs_opt == "pos":
        prob_names = [f"{out_prefix}prob_{ordered_names[pos_col_idx]}"]
    elif probs_opt == "all":
        prob_names = [f"{out_prefix}prob_{name}" for name in ordered_names]
    id_dtype = select_out_int_dtype(output_ids + [eff_nodata])[1]

    def _chunk(df: pd.DataFrame) -> pd.DataFrame:
        n_rows = len(df)
        x_df, row_ok = prepare_feature_matrix(
            df,
            feat_cols,
            seg.valid_col,
            eff_missing,
            eff_fill,
            seg.feat_no_data_val,
            scaler,
        )
        train_pred = np.full(n_rows, -1, dtype=np.int64)
        probs: Optional[np.ndarray] = None
        if np.any(row_ok):
            x_in = model_input(estimator, x_df)
            if need_probs:
                raw = np.asarray(
                    estimator.predict_proba(x_in, **p_kwargs), dtype=np.float32
                )
                probs = align_proba_columns(raw, est_classes, ordered_train_ids)
            if binary and pred_threshold is not None and probs is not None:
                preds = np.where(
                    probs[:, pos_col_idx] >= float(pred_threshold),
                    pos_train_id,
                    neg_train_id,
                )
            else:
                preds = np.asarray(
                    estimator.predict(x_in, **p_kwargs), dtype=np.int64
                ).ravel()
            train_pred[row_ok] = preds
        out_ids = np.full(n_rows, eff_nodata, dtype=np.int64)
        names: List[Optional[str]] = [None] * n_rows
        colours: List[Optional[str]] = [None] * n_rows
        rgb = np.zeros((n_rows, 3), dtype=np.uint8)
        for t_id, (o_id, c_name, c_obj) in lut.items():
            sel = train_pred == t_id
            if not np.any(sel):
                continue
            out_ids[sel] = o_id
            rgb[sel] = (int(c_obj.red), int(c_obj.green), int(c_obj.blue))
            hex_clr = _rgb_hex(c_obj)
            for idx in np.flatnonzero(sel):
                names[idx] = c_name
                colours[idx] = hex_clr
        out = pd.DataFrame(
            {
                f"{out_prefix}id": out_ids.astype(id_dtype),
                f"{out_prefix}name": pd.Series(names, dtype=object),
                f"{out_prefix}colour": pd.Series(colours, dtype=object),
                f"{out_prefix}red": rgb[:, 0],
                f"{out_prefix}green": rgb[:, 1],
                f"{out_prefix}blue": rgb[:, 2],
            }
        )
        if prob_names:
            pvals = np.full((n_rows, len(prob_names)), np.nan, dtype=np.float32)
            if probs is not None:
                if probs_opt == "max":
                    pvals[row_ok, 0] = probs.max(axis=1)
                elif probs_opt == "pos":
                    pvals[row_ok, 0] = probs[:, pos_col_idx]
                else:
                    pvals[row_ok] = probs
            for p_idx, p_name in enumerate(prob_names):
                out[p_name] = pvals[:, p_idx]
        return out

    meta = build_provenance(
        model=estimator,
        model_file=model_file,
        config_file=cls_cfg if isinstance(cls_cfg, (str, os.PathLike)) else None,
        feature_cols=feat_cols,
        extra={
            "task": "classification",
            "classes": {name: int(lut[tid][0]) for name, tid in zip(
                ordered_names, ordered_train_ids
            )},
            "cls_no_data_val": int(eff_nodata),
        },
    )
    map_vector_chunks(
        in_path,
        out_vec_file,
        read_cols,
        _chunk,
        in_vec_lyr=in_lyr,
        geom_vec_file=geom_vec_file,
        geom_vec_lyr=geom_vec_lyr,
        id_col=seg.id_col,
        copy_cols=copy_cols,
        chunk_size=chunk_size,
        out_vec_lyr=out_vec_lyr,
        out_format=out_format,
        out_compress=out_compress,
        metadata=meta,
        overwrite=overwrite,
        use_pbar=use_pbar,
    )
