# pylint: disable=too-many-lines
"""
Feature exploration, multicollinearity diagnostics, feature selection, and
feature importance / contribution analysis for classification workflows.

Provides sample-table exploratory plots and diagnostics, multiple supervised and
unsupervised feature selection methods (CQV, correlation clustering, iterative
VIF pruning, univariate F/MI/chi2 tests, RFE/RFECV, and model/LassoLars
selection), and model interpretability tools (permutation & native feature
importance, jackknife analysis, multiclass marginal response curves, SHAP
contributions, and SALib Sobol global sensitivity analysis).
"""

import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import rsgispy
from rsgispy.classification._helpers import (
    ClsConfigInput,
    SamplesInput,
    calc_split_metrics,
    check_sklearn_available,
    coerce_cls_config,
    extract_xy,
    load_samples_df,
    resolve_cv_splitter,
    resolve_feature_cols,
)
from rsgispy.classification.classdata import ClassInfoObj
from rsgispy.tools._ml_explain import (
    extract_native_feature_importances,
    finalise_plot,
    plot_class_feature_distributions,
    plot_correlation_heatmap,
    plot_feature_importance_bars,
    plot_jackknife_bars,
    run_boruta_shap_selection,
    run_salib_sobol_sensitivity,
    run_shap_explainer,
    run_variable_response_curves,
)
from rsgispy.tools._sklearn import (
    check_sklearn_estimator_fitted,
    clone_sklearn_estimator,
    fit_clean_sklearn_estimator,
    sklearn,
)
from rsgispy.tools.featureselection import (
    calc_pandas_vif,
    corr_feature_selection,
    cqv_threshold,
    lassolars_feature_selection,
)
from rsgispy.tools.files import ensure_parent_dir

__all__ = [
    "comparison_box_plots",
    "correlation_matrix",
    "calc_vif_multicollinearity",
    "select_features_cqv",
    "select_features_corr",
    "select_features_vif",
    "select_features_univariate",
    "select_features_rfe",
    "select_features_from_model",
    "select_features_boruta_shap",
    "calc_feature_importance",
    "calc_jackknife_feature_importance",
    "calc_feature_response_curves",
    "calc_shap_feature_contributions",
    "calc_salib_feature_sensitivity",
]

_VALID_CORR_METHODS = {"pearson", "spearman", "kendall"}


def _prepare_classification_data(
    samples_data: SamplesInput,
    cls_cfg: Optional[ClsConfigInput],
    feature_cols: Optional[Sequence[str]],
    use_internal_id: bool,
    out_id_col: Optional[str],
    internal_id_col: Optional[str],
    class_name_col: Optional[str] = None,
    label: str = "samples_data",
    require_target: bool = True,
) -> Tuple[
    pd.DataFrame,
    List[str],
    Optional[pd.DataFrame],
    Optional[np.ndarray],
    Dict[str, ClassInfoObj],
    str,
    str,
    str,
]:
    """Load sample DataFrame, resolve feature columns, and extract optional XY."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    cfg, class_info, eff_out_col, eff_int_col, eff_name_col = coerce_cls_config(
        cls_cfg,
        out_id_col=out_id_col,
        internal_id_col=internal_id_col,
        class_name_col=class_name_col,
        allow_none=True,
    )
    df = load_samples_df(samples_data, label)
    if len(df) == 0:
        raise rsgispy.RSGISPyException(f"{label} sample table is empty.")

    feat_cols = resolve_feature_cols(
        cfg,
        df,
        feature_cols,
        excluded_cols=[eff_out_col, eff_int_col, eff_name_col],
    )

    if not require_target:
        x_df = df[feat_cols].copy()
        x_vals = x_df.to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(x_vals)):
            raise rsgispy.RSGISPyException(
                f"{label} contains NaN or infinite values in feature columns."
            )
        return (
            df,
            feat_cols,
            x_df,
            None,
            class_info,
            eff_out_col,
            eff_int_col,
            eff_name_col,
        )

    if use_internal_id and eff_int_col in df.columns:
        target_col = eff_int_col
        valid_ids = (
            [int(obj.internal_id) for obj in class_info.values() if obj.internal_id is not None]
            if class_info
            else None
        )
    elif eff_out_col in df.columns:
        target_col = eff_out_col
        valid_ids = (
            [int(obj.out_id) for obj in class_info.values()]
            if class_info
            else None
        )
    else:
        target_col = eff_int_col if use_internal_id else eff_out_col
        valid_ids = None

    x_df, y_arr = extract_xy(
        df, feat_cols, target_col, label, valid_ids=valid_ids
    )
    return (
        df,
        feat_cols,
        x_df,
        y_arr,
        class_info,
        eff_out_col,
        eff_int_col,
        eff_name_col,
    )


def _resolve_estimator_class_labels(
    fitted_est: Any,
    class_info: Mapping[str, ClassInfoObj],
    use_internal_id: bool,
) -> Tuple[List[Any], List[str]]:
    """Return ordered estimator class IDs and human-readable display labels."""
    est_classes = list(getattr(fitted_est, "classes_", []))
    if not est_classes:
        if class_info:
            sorted_items = sorted(
                class_info.items(),
                key=lambda item: int(
                    item[1].internal_id
                    if use_internal_id and item[1].internal_id is not None
                    else item[1].out_id
                ),
            )
            ids = [
                int(
                    obj.internal_id
                    if use_internal_id and obj.internal_id is not None
                    else obj.out_id
                )
                for _, obj in sorted_items
            ]
            names = [name for name, _ in sorted_items]
            return ids, names
        return [], []

    id_to_name: Dict[int, str] = {}
    for cls_name, obj in class_info.items():
        key_id = (
            int(obj.internal_id)
            if use_internal_id and obj.internal_id is not None
            else int(obj.out_id)
        )
        id_to_name[key_id] = cls_name

    labels = [
        id_to_name.get(int(cid), str(cid)) if isinstance(cid, (int, np.integer)) else str(cid)
        for cid in est_classes
    ]
    return est_classes, labels


def _resolve_target_class_index(
    fitted_est: Any,
    class_info: Mapping[str, ClassInfoObj],
    target_class: Optional[Union[str, int]],
    use_internal_id: bool,
    default_binary_pos: bool = True,
) -> int:
    """Resolve ``target_class`` (name or class ID) into 0-based ``classes_`` index."""
    est_classes, est_labels = _resolve_estimator_class_labels(
        fitted_est, class_info, use_internal_id
    )
    if not est_classes:
        return 0

    if target_class is None:
        if default_binary_pos and len(est_classes) == 2:
            return 1
        return 0

    if isinstance(target_class, str):
        if target_class in est_labels:
            return est_labels.index(target_class)
        raise rsgispy.RSGISPyException(
            f"target_class '{target_class}' not found in classes {est_labels}."
        )

    if isinstance(target_class, (int, np.integer)) and not isinstance(target_class, bool):
        target_int = int(target_class)
        int_classes = [int(c) for c in est_classes]
        if target_int in int_classes:
            return int_classes.index(target_int)
        if use_internal_id and class_info:
            for obj in class_info.values():
                if int(obj.out_id) == target_int and obj.internal_id is not None:
                    mapped_int = int(obj.internal_id)
                    if mapped_int in int_classes:
                        return int_classes.index(mapped_int)
        raise rsgispy.RSGISPyException(
            f"target_class ID {target_int} not found in estimator classes "
            f"{int_classes}."
        )

    raise rsgispy.RSGISPyException(
        "target_class must be None, a class name string, or an integer class ID."
    )


def _predict_class_probs(
    fitted_est: Any,
    x_df: pd.DataFrame,
    class_ids: Sequence[Any],
) -> np.ndarray:
    """Return class probability array of shape ``(n_rows, n_classes)``."""
    if hasattr(fitted_est, "predict_proba"):
        probs = np.asarray(fitted_est.predict_proba(x_df), dtype=np.float64)
        if probs.ndim == 2:
            return probs
    if hasattr(fitted_est, "decision_function"):
        scores = np.asarray(
            fitted_est.decision_function(x_df), dtype=np.float64
        )
        if scores.ndim == 1:
            p1 = 1.0 / (1.0 + np.exp(-np.clip(scores, -50.0, 50.0)))
            return np.column_stack([1.0 - p1, p1])
        shifted = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    preds = np.asarray(fitted_est.predict(x_df))
    out = np.zeros((len(preds), max(1, len(class_ids))), dtype=np.float64)
    for c_idx, cid in enumerate(class_ids):
        out[preds == cid, c_idx] = 1.0
    return out


# ---------------------------------------------------------------------------
# Exploratory Feature Diagnostics & Collinearity
# ---------------------------------------------------------------------------


# pylint: disable=too-many-arguments,too-many-positional-arguments
def comparison_box_plots(
    samples_data: SamplesInput,
    cls_cfg: Optional[ClsConfigInput] = None,
    feature_cols: Optional[Sequence[str]] = None,
    out_dir: Union[str, os.PathLike] = "boxplots",
    use_internal_id: bool = False,
    use_class_names: bool = True,
    out_id_col: Optional[str] = None,
    internal_id_col: Optional[str] = None,
    class_name_col: Optional[str] = None,
    box_plots: bool = False,
    use_pbar: bool = True,
) -> None:
    """
    Create per-feature distribution comparison plots (violin or box plots)
    grouped by land-cover class across extracted classification samples.

    :param samples_data: Sample :class:`pandas.DataFrame` or table path
                         (Parquet, CSV, etc.).
    :param cls_cfg: Optional :class:`~rsgispy.classification.classdata.ClassificationConfig`,
                    JSON config path, or mapping of ``{class_name: ClassInfoObj}``.
    :param feature_cols: Optional subset/order of feature column names.
    :param out_dir: Output directory where ``<feature>_comp_plot.png`` plots
                    are saved (Default = ``'boxplots'``).
    :param use_internal_id: Group by ``internal_cls_id`` when ``use_class_names=False``
                            or class names are absent (Default = ``False``).
    :param use_class_names: If ``True`` (default), group plots by class names
                            from ``class_name_col`` or ``cls_cfg`` when available.
    :param out_id_col: Optional override for the output class ID column name.
    :param internal_id_col: Optional override for the internal class ID column.
    :param class_name_col: Optional override for the class name column.
    :param box_plots: If ``True``, draw box plots; if ``False`` (default),
                      draw violin plots.
    :param use_pbar: Display a tqdm progress bar (Default = ``True``).

    .. code:: python

        import rsgispy.classification.classfeatures as cf

        cf.comparison_box_plots(
            "train.parquet", cls_cfg="cls_config.json", out_dir="plots", box_plots=True
        )
    """
    # pylint: disable=too-many-locals
    df, feat_cols, _, _, class_info, eff_out_col, eff_int_col, eff_name_col = (
        _prepare_classification_data(
            samples_data=samples_data,
            cls_cfg=cls_cfg,
            feature_cols=feature_cols,
            use_internal_id=use_internal_id,
            out_id_col=out_id_col,
            internal_id_col=internal_id_col,
            class_name_col=class_name_col,
            label="samples_data",
            require_target=False,
        )
    )

    plot_df = df.copy()
    group_col: Optional[str] = None
    if use_class_names and eff_name_col in plot_df.columns:
        group_col = eff_name_col
    elif use_class_names and class_info:
        if use_internal_id and eff_int_col in plot_df.columns:
            id_map = {
                int(obj.internal_id): name
                for name, obj in class_info.items()
                if obj.internal_id is not None
            }
            src_col = eff_int_col
        elif eff_out_col in plot_df.columns:
            id_map = {int(obj.out_id): name for name, obj in class_info.items()}
            src_col = eff_out_col
        else:
            id_map = {}
            src_col = ""
        if src_col:
            group_col = "__rsgis_cls_label__"
            plot_df[group_col] = plot_df[src_col].map(
                lambda v: id_map.get(int(v), str(v))
            )

    if group_col is None:
        if use_internal_id and eff_int_col in plot_df.columns:
            group_col = eff_int_col
        elif eff_out_col in plot_df.columns:
            group_col = eff_out_col
        elif eff_int_col in plot_df.columns:
            group_col = eff_int_col
        elif eff_name_col in plot_df.columns:
            group_col = eff_name_col
        else:
            raise rsgispy.RSGISPyException(
                "Could not find a class label or class ID column in samples_data."
            )

    plot_class_feature_distributions(
        data_df=plot_df,
        feature_cols=feat_cols,
        cls_col=group_col,
        out_dir=out_dir,
        box_plots=box_plots,
        use_pbar=use_pbar,
    )


# pylint: disable=too-many-arguments,too-many-positional-arguments
def correlation_matrix(
    samples_data: SamplesInput,
    cls_cfg: Optional[ClsConfigInput] = None,
    feature_cols: Optional[Sequence[str]] = None,
    method: str = "pearson",
    out_corr_file: Optional[Union[str, os.PathLike]] = "correlation_matrix.csv",
    out_plt_file: Optional[Union[str, os.PathLike]] = "correlation_matrix.png",
    fig_width: int = 15,
    fig_height: int = 14,
) -> pd.DataFrame:
    """
    Calculate pairwise feature correlations across classification sample
    predictors, optionally saving the matrix CSV and lower-triangular heatmap.

    :param samples_data: Sample :class:`pandas.DataFrame` or table path.
    :param cls_cfg: Optional classification configuration or class mapping.
    :param feature_cols: Optional ordered feature column names.
    :param method: Correlation method: ``'pearson'``, ``'spearman'``, or
                   ``'kendall'`` (Default = ``'pearson'``).
    :param out_corr_file: Output correlation CSV path (or ``None`` to skip).
    :param out_plt_file: Output heatmap PNG path (or ``None`` to skip).
    :param fig_width: Heatmap figure width in inches (Default = ``15``).
    :param fig_height: Heatmap figure height in inches (Default = ``14``).
    :return: Correlation matrix :class:`pandas.DataFrame`.

    .. code:: python

        import rsgispy.classification.classfeatures as cf

        corr_df = cf.correlation_matrix(
            "train.parquet", cls_cfg="cls_config.json",
            out_corr_file="corr.csv", out_plt_file="corr.png",
        )
    """
    corr_method = method.strip().lower()
    if corr_method not in _VALID_CORR_METHODS:
        raise rsgispy.RSGISPyException(
            f"Unsupported correlation method '{method}'. Expected one of "
            f"{sorted(_VALID_CORR_METHODS)}."
        )

    df, feat_cols, _, _, _, _, _, _ = _prepare_classification_data(
        samples_data=samples_data,
        cls_cfg=cls_cfg,
        feature_cols=feature_cols,
        use_internal_id=True,
        out_id_col=None,
        internal_id_col=None,
        label="samples_data",
        require_target=False,
    )

    corr_df = df[feat_cols].corr(method=corr_method)  # type: ignore[arg-type]
    if out_corr_file is not None:
        corr_df.to_csv(ensure_parent_dir(out_corr_file), index=True)

    if out_plt_file is not None:
        plot_correlation_heatmap(
            corr_df=corr_df,
            var_names=feat_cols,
            out_plt_file=out_plt_file,
            fig_width=fig_width,
            fig_height=fig_height,
        )
    return corr_df


def calc_vif_multicollinearity(
    samples_data: SamplesInput,
    cls_cfg: Optional[ClsConfigInput] = None,
    feature_cols: Optional[Sequence[str]] = None,
    out_vif_file: Optional[Union[str, os.PathLike]] = "vif_multicollinearity.csv",
) -> pd.Series:
    """
    Calculate Variance Inflation Factors (VIF) across classification feature
    columns using :func:`rsgispy.tools.featureselection.calc_pandas_vif`.

    :param samples_data: Sample :class:`pandas.DataFrame` or table path.
    :param cls_cfg: Optional classification configuration or class mapping.
    :param feature_cols: Optional ordered feature column names (at least 2).
    :param out_vif_file: Output CSV path for VIF scores (or ``None``).
    :return: :class:`pandas.Series` indexed by feature name containing VIF values.

    .. code:: python

        import rsgispy.classification.classfeatures as cf

        vif = cf.calc_vif_multicollinearity("train.parquet", cls_cfg="cls_config.json")
    """
    _, feat_cols, x_df, _, _, _, _, _ = _prepare_classification_data(
        samples_data=samples_data,
        cls_cfg=cls_cfg,
        feature_cols=feature_cols,
        use_internal_id=True,
        out_id_col=None,
        internal_id_col=None,
        label="samples_data",
        require_target=False,
    )
    assert x_df is not None
    vif_scores = calc_pandas_vif(x_df, cols=feat_cols)
    if out_vif_file is not None:
        vif_scores.to_csv(ensure_parent_dir(out_vif_file), index=True)
    return vif_scores


# ---------------------------------------------------------------------------
# Feature Selection Methods
# ---------------------------------------------------------------------------


def select_features_cqv(
    samples_data: SamplesInput,
    cls_cfg: Optional[ClsConfigInput] = None,
    feature_cols: Optional[Sequence[str]] = None,
    low_threshold: float = 0.25,
    high_threshold: float = 0.75,
) -> List[str]:
    """
    Select classification features using the Coefficient of Quartile Variation
    (CQV) filter via :func:`rsgispy.tools.featureselection.cqv_threshold`.

    :param samples_data: Sample :class:`pandas.DataFrame` or table path.
    :param cls_cfg: Optional classification configuration or class mapping.
    :param feature_cols: Optional ordered feature column names (at least 2).
    :param low_threshold: Minimum inclusive CQV threshold (Default = ``0.25``).
    :param high_threshold: Maximum inclusive CQV threshold (Default = ``0.75``).
    :return: List of selected feature column names.

    .. code:: python

        import rsgispy.classification.classfeatures as cf

        selected = cf.select_features_cqv(
            "train.parquet", cls_cfg="cls_config.json", low_threshold=0.1, high_threshold=0.9
        )
    """
    if low_threshold > high_threshold:
        raise rsgispy.RSGISPyException(
            "low_threshold must be less than or equal to high_threshold."
        )
    _, feat_cols, x_df, _, _, _, _, _ = _prepare_classification_data(
        samples_data=samples_data,
        cls_cfg=cls_cfg,
        feature_cols=feature_cols,
        use_internal_id=True,
        out_id_col=None,
        internal_id_col=None,
        label="samples_data",
        require_target=False,
    )
    assert x_df is not None
    return cqv_threshold(
        x_df,
        cols=feat_cols,
        lowthreshold=low_threshold,
        highthreshold=high_threshold,
    )


# pylint: disable=too-many-arguments,too-many-positional-arguments
def select_features_corr(
    samples_data: SamplesInput,
    cls_cfg: Optional[ClsConfigInput] = None,
    feature_cols: Optional[Sequence[str]] = None,
    use_internal_id: bool = True,
    out_id_col: Optional[str] = None,
    internal_id_col: Optional[str] = None,
    n_min_clusters: int = 3,
    n_max_clusters: int = 25,
    use_pbar: bool = True,
) -> List[str]:
    """
    Select representative predictors via correlation hierarchical clustering and
    Silhouette optimisation using
    :func:`rsgispy.tools.featureselection.corr_feature_selection`.

    :param samples_data: Sample :class:`pandas.DataFrame` or table path.
    :param cls_cfg: Optional classification configuration or class mapping.
    :param feature_cols: Optional ordered feature column names (at least 3).
    :param use_internal_id: Use ``internal_cls_id`` as target when ``True``
                            (Default = ``True``).
    :param out_id_col: Optional override for the output class ID column name.
    :param internal_id_col: Optional override for the internal class ID column.
    :param n_min_clusters: Minimum cluster count evaluated (Default = ``3``).
    :param n_max_clusters: Exclusive upper bound on cluster count
                           (clamped to ``len(feature_cols)`` when smaller).
    :param use_pbar: Display a tqdm progress bar (Default = ``True``).
    :return: List of selected feature column names.

    .. code:: python

        import rsgispy.classification.classfeatures as cf

        selected = cf.select_features_corr(
            "train.parquet", cls_cfg="cls_config.json", n_min_clusters=2, n_max_clusters=6
        )
    """
    # pylint: disable=too-many-locals
    df, feat_cols, _, y_arr, _, eff_out_col, eff_int_col, _ = (
        _prepare_classification_data(
            samples_data=samples_data,
            cls_cfg=cls_cfg,
            feature_cols=feature_cols,
            use_internal_id=use_internal_id,
            out_id_col=out_id_col,
            internal_id_col=internal_id_col,
            label="samples_data",
            require_target=True,
        )
    )
    assert y_arr is not None
    if len(feat_cols) < 3:
        raise rsgispy.RSGISPyException(
            "select_features_corr requires at least 3 features."
        )

    eff_max = min(int(n_max_clusters), len(feat_cols))
    eff_min = max(2, min(int(n_min_clusters), eff_max - 1))
    if eff_min >= eff_max:
        raise rsgispy.RSGISPyException(
            f"Invalid cluster range [{eff_min}, {eff_max}) for "
            f"{len(feat_cols)} features."
        )

    target_col = eff_int_col if (use_internal_id and eff_int_col in df.columns) else eff_out_col
    work_df = df[feat_cols].copy()
    work_df["__rsgis_target__"] = y_arr.astype(np.float64)
    if np.std(y_arr) == 0.0:
        raise rsgispy.RSGISPyException(
            f"Target column '{target_col}' has zero variance; cannot compute "
            "correlation-based feature selection."
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return corr_feature_selection(
            work_df,
            dep_vars=["__rsgis_target__"],
            ind_vars=feat_cols,
            n_min_clusters=eff_min,
            n_max_clusters=eff_max,
            use_pbar=use_pbar,
        )


def select_features_vif(
    samples_data: SamplesInput,
    cls_cfg: Optional[ClsConfigInput] = None,
    feature_cols: Optional[Sequence[str]] = None,
    vif_threshold: float = 10.0,
    min_features: int = 2,
    out_vif_file: Optional[Union[str, os.PathLike]] = None,
    return_vif: bool = False,
) -> Union[List[str], Tuple[List[str], pd.Series]]:
    """
    Iteratively prune multicollinear features by dropping the highest-VIF
    predictor until all remaining features have ``VIF <= vif_threshold`` or
    ``min_features`` remain.

    :param samples_data: Sample :class:`pandas.DataFrame` or table path.
    :param cls_cfg: Optional classification configuration or class mapping.
    :param feature_cols: Optional ordered feature column names.
    :param vif_threshold: Maximum acceptable VIF (Default = ``10.0``).
    :param min_features: Minimum number of features to retain (at least ``2``,
                         Default = ``2``).
    :param out_vif_file: Optional CSV path to save final VIF scores of retained
                         features.
    :param return_vif: If ``True``, return ``(selected_features, final_vif)``
                       (Default = ``False``).
    :return: List of retained feature names (or ``(selected_features, vif_series)``).

    .. code:: python

        import rsgispy.classification.classfeatures as cf

        kept = cf.select_features_vif(
            "train.parquet", cls_cfg="cls_config.json", vif_threshold=5.0
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    if vif_threshold <= 0.0:
        raise rsgispy.RSGISPyException("vif_threshold must be > 0.")
    if min_features < 2:
        raise rsgispy.RSGISPyException("min_features must be at least 2.")

    _, feat_cols, x_df, _, _, _, _, _ = _prepare_classification_data(
        samples_data=samples_data,
        cls_cfg=cls_cfg,
        feature_cols=feature_cols,
        use_internal_id=True,
        out_id_col=None,
        internal_id_col=None,
        label="samples_data",
        require_target=False,
    )
    assert x_df is not None
    if len(feat_cols) < 2:
        raise rsgispy.RSGISPyException(
            "select_features_vif requires at least 2 features."
        )

    remaining = list(feat_cols)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        vif_series = calc_pandas_vif(x_df[remaining], cols=remaining)
        while len(remaining) > min_features:
            eff_scores: Dict[str, float] = {}
            for col in remaining:
                val = float(vif_series[col])
                if not np.isfinite(val):
                    eff_scores[col] = np.inf
                elif val == 0.0 and float(x_df[col].std()) > 0.0:
                    eff_scores[col] = np.inf
                else:
                    eff_scores[col] = val
            worst_col = max(eff_scores, key=lambda k: eff_scores[k])
            if eff_scores[worst_col] <= vif_threshold:
                break
            remaining.remove(worst_col)
            vif_series = calc_pandas_vif(x_df[remaining], cols=remaining)

    if out_vif_file is not None:
        vif_series.to_csv(ensure_parent_dir(out_vif_file), index=True)
    if return_vif:
        return remaining, vif_series
    return remaining


# pylint: disable=too-many-arguments,too-many-positional-arguments
def select_features_univariate(
    cls_cfg: Optional[ClsConfigInput],
    train_data: SamplesInput,
    feature_cols: Optional[Sequence[str]] = None,
    score_func: str = "f_classif",
    k: Optional[Union[int, str]] = 10,
    percentile: Optional[int] = None,
    use_internal_id: bool = True,
    out_id_col: Optional[str] = None,
    internal_id_col: Optional[str] = None,
    out_scores_file: Optional[Union[str, os.PathLike]] = None,
    rnd_seed: Optional[int] = None,
    return_scores: bool = False,
) -> Union[List[str], Tuple[List[str], pd.DataFrame]]:
    """
    Select features for classification using univariate statistical tests via
    ``scikit-learn`` ``SelectKBest`` or ``SelectPercentile``.

    :param cls_cfg: Optional :class:`~rsgispy.classification.classdata.ClassificationConfig`,
                    JSON config path, or mapping of ``{class_name: ClassInfoObj}``.
    :param train_data: Training sample :class:`pandas.DataFrame` or file path.
    :param feature_cols: Optional ordered feature column names.
    :param score_func: Scoring test: ``'f_classif'`` (ANOVA F-value, default),
                       ``'mutual_info_classif'``, or ``'chi2'``.
    :param k: Number of top features for ``SelectKBest`` (ignored when
              ``percentile`` is set; clamped to feature count if larger when
              positive int).
    :param percentile: Optional percentage ``(1..100]`` of top features to
                       keep using ``SelectPercentile``.
    :param use_internal_id: Train/score against ``internal_cls_id`` when ``True``.
    :param out_id_col: Optional override for output class ID column name.
    :param internal_id_col: Optional override for internal class ID column name.
    :param out_scores_file: Optional CSV path saving per-feature scores/ranks.
    :param rnd_seed: Optional random seed passed to ``mutual_info_classif``.
    :param return_scores: If ``True``, return ``(selected_features, scores_df)``.
    :return: List of selected feature names (or ``(selected_features, scores_df)``).

    .. code:: python

        import rsgispy.classification.classfeatures as cf

        selected, scores_df = cf.select_features_univariate(
            "cls_config.json", "train.parquet", score_func="f_classif", k=5, return_scores=True
        )
    """
    # pylint: disable=too-many-locals
    check_sklearn_available("select_features_univariate")
    _, feat_cols, x_df, y_arr, _, _, _, _ = _prepare_classification_data(
        samples_data=train_data,
        cls_cfg=cls_cfg,
        feature_cols=feature_cols,
        use_internal_id=use_internal_id,
        out_id_col=out_id_col,
        internal_id_col=internal_id_col,
        label="train_data",
        require_target=True,
    )
    assert x_df is not None and y_arr is not None

    method_key = score_func.strip().lower()
    if method_key in {"f_classif", "anova", "f"}:
        fn: Any = sklearn.feature_selection.f_classif
    elif method_key in {"mutual_info_classif", "mutual_info", "mi"}:
        def _mi_fn(x_vals: Any, y_vals: Any) -> np.ndarray:
            return sklearn.feature_selection.mutual_info_classif(
                x_vals, y_vals, random_state=rnd_seed
            )

        fn = _mi_fn
    elif method_key == "chi2":
        if np.any(x_df.to_numpy(dtype=np.float64) < 0.0):
            raise rsgispy.RSGISPyException(
                "score_func='chi2' requires non-negative feature values."
            )
        fn = sklearn.feature_selection.chi2
    else:
        raise rsgispy.RSGISPyException(
            f"Unsupported score_func '{score_func}'. Expected 'f_classif', "
            "'mutual_info_classif', or 'chi2'."
        )

    if percentile is not None:
        if percentile <= 0 or percentile > 100:
            raise rsgispy.RSGISPyException(
                "percentile must be in the range (0, 100]."
            )
        selector: Any = sklearn.feature_selection.SelectPercentile(
            score_func=fn, percentile=int(percentile)
        )
    else:
        if isinstance(k, (int, np.integer)) and not isinstance(k, bool):
            if int(k) < 1:
                raise rsgispy.RSGISPyException("k must be at least 1.")
            eff_k: Union[int, str] = min(int(k), len(feat_cols))
        elif k == "all":
            eff_k = "all"
        else:
            raise rsgispy.RSGISPyException(
                "k must be a positive integer or 'all'."
            )
        selector = sklearn.feature_selection.SelectKBest(
            score_func=fn, k=eff_k
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        selector.fit(x_df, y_arr)

    support = np.asarray(selector.get_support(), dtype=bool)
    raw_scores = np.asarray(selector.scores_, dtype=np.float64)
    raw_pvals = (
        np.asarray(selector.pvalues_, dtype=np.float64)
        if getattr(selector, "pvalues_", None) is not None
        else np.full(len(feat_cols), np.nan, dtype=np.float64)
    )

    selected = [col for col, keep in zip(feat_cols, support) if keep]
    scores_df = pd.DataFrame(
        {
            "feature": feat_cols,
            "score": raw_scores,
            "p_value": raw_pvals,
            "selected": support,
        }
    )
    scores_df = scores_df.sort_values(
        "score", ascending=False, na_position="last"
    ).reset_index(drop=True)
    scores_df["rank"] = np.arange(1, len(scores_df) + 1, dtype=int)

    if out_scores_file is not None:
        scores_df.to_csv(ensure_parent_dir(out_scores_file), index=False)
    if return_scores:
        return selected, scores_df
    return selected


# pylint: disable=too-many-arguments,too-many-positional-arguments
def select_features_rfe(
    cls_cfg: Optional[ClsConfigInput],
    sk_classifier: Any,
    train_data: SamplesInput,
    feature_cols: Optional[Sequence[str]] = None,
    n_features_to_select: Optional[Union[int, float]] = None,
    step: Union[int, float] = 1,
    use_cv: bool = False,
    cv: Union[int, Any] = 5,
    scoring: Optional[Union[str, Any]] = "accuracy",
    min_features_to_select: int = 1,
    use_internal_id: bool = True,
    rnd_seed: Optional[int] = None,
    out_id_col: Optional[str] = None,
    internal_id_col: Optional[str] = None,
    out_scores_file: Optional[Union[str, os.PathLike]] = None,
    out_cv_plot: Optional[Union[str, os.PathLike]] = None,
    n_jobs: int = 1,
    return_scores: bool = False,
) -> Union[List[str], Tuple[List[str], pd.DataFrame]]:
    """
    Select classification features using Recursive Feature Elimination
    (``sklearn.feature_selection.RFE`` or stratified ``RFECV``) with a clean
    clone of ``sk_classifier``.

    :param cls_cfg: Optional classification configuration or class mapping.
    :param sk_classifier: ``scikit-learn`` classifier exposing ``feature_importances_``
                          or ``coef_`` after fitting.
    :param train_data: Training sample :class:`pandas.DataFrame` or file path.
    :param feature_cols: Optional ordered feature column names.
    :param n_features_to_select: Number (int) or fraction (float in ``(0, 1]``)
                                 of features to retain for ``RFE`` (ignored when
                                 ``use_cv=True``).
    :param step: Number/fraction of features removed per iteration (Default = ``1``).
    :param use_cv: If ``True``, use ``RFECV`` with stratified CV to determine
                   the optimal feature count automatically (Default = ``False``).
    :param cv: Cross-validation folds or splitter for ``RFECV`` (Default = ``5``).
    :param scoring: Scoring metric for ``RFECV`` (Default = ``'accuracy'``).
    :param min_features_to_select: Minimum features considered by ``RFECV``
                                   (Default = ``1``).
    :param use_internal_id: Train against ``internal_cls_id`` when ``True``.
    :param rnd_seed: Optional random seed passed to cloned estimator and CV splitter.
    :param out_id_col: Optional override for output class ID column name.
    :param internal_id_col: Optional override for internal class ID column name.
    :param out_scores_file: Optional CSV path saving per-feature rankings.
    :param out_cv_plot: Optional PNG plot path showing CV score vs number of
                        features (only generated when ``use_cv=True``).
    :param n_jobs: Parallel CV jobs for ``RFECV`` (Default = ``1``).
    :param return_scores: If ``True``, return ``(selected_features, ranking_df)``.
    :return: List of selected feature names (or ``(selected_features, ranking_df)``).

    .. code:: python

        from sklearn.ensemble import RandomForestClassifier
        import rsgispy.classification.classfeatures as cf

        selected = cf.select_features_rfe(
            "cls_config.json",
            RandomForestClassifier(n_estimators=50, random_state=42),
            "train.parquet",
            n_features_to_select=4,
        )
    """
    # pylint: disable=too-many-locals
    check_sklearn_available("select_features_rfe")
    _, feat_cols, x_df, y_arr, _, _, _, _ = _prepare_classification_data(
        samples_data=train_data,
        cls_cfg=cls_cfg,
        feature_cols=feature_cols,
        use_internal_id=use_internal_id,
        out_id_col=out_id_col,
        internal_id_col=internal_id_col,
        label="train_data",
        require_target=True,
    )
    assert x_df is not None and y_arr is not None

    base_est = clone_sklearn_estimator(
        sk_classifier, random_state=rnd_seed, extra="classification"
    )
    if use_cv:
        cv_splitter = resolve_cv_splitter(cv, y_arr, rnd_seed=rnd_seed)
        selector: Any = sklearn.feature_selection.RFECV(
            estimator=base_est,
            step=step,
            min_features_to_select=min(int(min_features_to_select), len(feat_cols)),
            cv=cv_splitter,
            scoring=scoring,
            n_jobs=n_jobs,
        )
    else:
        eff_n = n_features_to_select
        if isinstance(eff_n, (int, np.integer)) and not isinstance(eff_n, bool):
            eff_n = min(max(1, int(eff_n)), len(feat_cols))
        selector = sklearn.feature_selection.RFE(
            estimator=base_est,
            n_features_to_select=eff_n,
            step=step,
        )

    selector.fit(x_df, y_arr)
    support = np.asarray(selector.get_support(), dtype=bool)
    ranking = np.asarray(selector.ranking_, dtype=int)
    selected = [col for col, keep in zip(feat_cols, support) if keep]

    rank_df = pd.DataFrame(
        {"feature": feat_cols, "ranking": ranking, "selected": support}
    )
    rank_df = rank_df.sort_values(
        ["ranking", "feature"], ascending=[True, True]
    ).reset_index(drop=True)

    if out_cv_plot is not None and use_cv and hasattr(selector, "cv_results_"):
        mean_scores = np.asarray(
            selector.cv_results_["mean_test_score"], dtype=float
        )
        n_feats_axis = np.asarray(
            selector.cv_results_.get(
                "n_features",
                np.arange(
                    len(feat_cols) - len(mean_scores) + 1, len(feat_cols) + 1
                ),
            ),
            dtype=int,
        )
        fig, ax = plt.subplots(figsize=(7, 4))
        try:
            ax.plot(n_feats_axis, mean_scores, marker="o", color="#1E88E5")
            ax.axvline(
                x=int(selector.n_features_),
                color="#D81B60",
                linestyle="--",
                label=f"Optimal = {int(selector.n_features_)}",
            )
            ax.set_xlabel("Number of Selected Features")
            ax.set_ylabel("CV Score")
            ax.set_title("RFECV Feature Selection")
            ax.legend()
            fig.tight_layout()
        finally:
            finalise_plot(fig, out_file=out_cv_plot)

    if out_scores_file is not None:
        rank_df.to_csv(ensure_parent_dir(out_scores_file), index=False)
    if return_scores:
        return selected, rank_df
    return selected


# pylint: disable=too-many-arguments,too-many-positional-arguments
def select_features_from_model(
    cls_cfg: Optional[ClsConfigInput],
    sk_classifier: Optional[Any],
    train_data: SamplesInput,
    feature_cols: Optional[Sequence[str]] = None,
    threshold: Optional[Union[str, float]] = "mean",
    max_features: Optional[int] = None,
    prefit: bool = False,
    use_lassolars: bool = False,
    alpha_val: Optional[float] = None,
    use_internal_id: bool = True,
    rnd_seed: Optional[int] = None,
    out_id_col: Optional[str] = None,
    internal_id_col: Optional[str] = None,
    out_scores_file: Optional[Union[str, os.PathLike]] = None,
    return_scores: bool = False,
) -> Union[List[str], Tuple[List[str], pd.DataFrame]]:
    """
    Select classification features based on estimator importance weights via
    ``sklearn.feature_selection.SelectFromModel``, or L1 regularisation via
    :func:`rsgispy.tools.featureselection.lassolars_feature_selection`.

    :param cls_cfg: Optional classification configuration or class mapping.
    :param sk_classifier: ``scikit-learn`` classifier exposing ``feature_importances_``
                          or ``coef_`` (may be ``None`` when ``use_lassolars=True``).
    :param train_data: Training sample :class:`pandas.DataFrame` or file path.
    :param feature_cols: Optional ordered feature column names.
    :param threshold: Threshold passed to ``SelectFromModel`` (Default = ``'mean'``).
    :param max_features: Optional cap on retained features in ``SelectFromModel``.
    :param prefit: Set ``True`` if ``sk_classifier`` is already fitted (Default = ``False``).
    :param use_lassolars: If ``True``, use
                          :func:`rsgispy.tools.featureselection.lassolars_feature_selection`
                          instead of ``SelectFromModel`` (Default = ``False``).
    :param alpha_val: Optional ``alpha`` for ``LassoLars`` when ``use_lassolars=True``.
    :param use_internal_id: Train against ``internal_cls_id`` when ``True``.
    :param rnd_seed: Optional random seed passed to cloned estimator.
    :param out_id_col: Optional override for output class ID column name.
    :param internal_id_col: Optional override for internal class ID column name.
    :param out_scores_file: Optional CSV path saving per-feature selection weights.
    :param return_scores: If ``True``, return ``(selected_features, scores_df)``.
    :return: List of selected feature names (or ``(selected_features, scores_df)``).

    .. code:: python

        from sklearn.ensemble import RandomForestClassifier
        import rsgispy.classification.classfeatures as cf

        selected = cf.select_features_from_model(
            "cls_config.json",
            RandomForestClassifier(n_estimators=50, random_state=42),
            "train.parquet",
        )
    """
    # pylint: disable=too-many-locals
    check_sklearn_available("select_features_from_model")
    df, feat_cols, x_df, y_arr, _, _, _, _ = _prepare_classification_data(
        samples_data=train_data,
        cls_cfg=cls_cfg,
        feature_cols=feature_cols,
        use_internal_id=use_internal_id,
        out_id_col=out_id_col,
        internal_id_col=internal_id_col,
        label="train_data",
        require_target=True,
    )
    assert x_df is not None and y_arr is not None

    if use_lassolars:
        work_df = df[feat_cols].copy()
        work_df["__rsgis_target__"] = y_arr.astype(np.float64)
        selected = lassolars_feature_selection(
            work_df,
            dep_vars=["__rsgis_target__"],
            ind_vars=feat_cols,
            alpha_val=alpha_val,
        )
        selected_set = set(selected)
        scores_df = pd.DataFrame(
            {
                "feature": feat_cols,
                "selected": [col in selected_set for col in feat_cols],
            }
        )
        if out_scores_file is not None:
            scores_df.to_csv(ensure_parent_dir(out_scores_file), index=False)
        if return_scores:
            return selected, scores_df
        return selected

    if sk_classifier is None:
        raise rsgispy.RSGISPyException(
            "sk_classifier must be provided when use_lassolars=False."
        )

    if prefit:
        check_sklearn_estimator_fitted(sk_classifier, label="sk_classifier")
        est = sk_classifier
    else:
        est = fit_clean_sklearn_estimator(
            sk_classifier,
            x_df,
            y_arr,
            sync_input=False,
            random_state=rnd_seed,
            extra="classification",
        )

    selector = sklearn.feature_selection.SelectFromModel(
        est, threshold=threshold, max_features=max_features, prefit=True
    )
    support = np.asarray(selector.get_support(), dtype=bool)
    selected = [col for col, keep in zip(feat_cols, support) if keep]

    importances = extract_native_feature_importances(est, len(feat_cols))
    scores_df = pd.DataFrame(
        {
            "feature": feat_cols,
            "importance": importances if importances is not None else np.nan,
            "selected": support,
        }
    )
    if importances is not None:
        scores_df = scores_df.sort_values(
            "importance", ascending=False
        ).reset_index(drop=True)

    if out_scores_file is not None:
        scores_df.to_csv(ensure_parent_dir(out_scores_file), index=False)
    if return_scores:
        return selected, scores_df
    return selected


# pylint: disable=too-many-arguments,too-many-positional-arguments
def select_features_boruta_shap(
    cls_cfg: Optional[ClsConfigInput],
    train_data: SamplesInput,
    sk_classifier: Optional[Any] = None,
    feature_cols: Optional[Sequence[str]] = None,
    n_trials: int = 50,
    pvalue: float = 0.05,
    percentile: float = 100.0,
    bonferroni: bool = True,
    include_tentative: bool = False,
    tentative_rough_fix: bool = False,
    target_class: Optional[Union[str, int]] = None,
    subsample_n_smpls: Optional[int] = None,
    use_tree_explainer: Optional[bool] = None,
    use_linear_explainer: bool = False,
    use_internal_id: bool = True,
    rnd_seed: Optional[int] = None,
    out_id_col: Optional[str] = None,
    internal_id_col: Optional[str] = None,
    out_scores_file: Optional[Union[str, os.PathLike]] = None,
    out_plot_file: Optional[Union[str, os.PathLike]] = None,
    use_pbar: bool = True,
    return_scores: bool = False,
) -> Union[List[str], Tuple[List[str], pd.DataFrame]]:
    """
    Select classification features using **BorutaShap**, combining the Boruta
    all-relevant shadow-feature hypothesis test with Shapley (SHAP) feature
    attributions.

    **Why BorutaShap is a strong feature selection option:**

    * **All-relevant rather than minimal-optimal selection**: Greedy or
      stepwise selectors (e.g., RFE, L1/Lasso, or univariate filters) aim for
      the smallest subset that preserves accuracy and frequently drop
      informative features whenever predictors are correlated or interact.
      Boruta tests *every* predictor individually against randomised shadow
      noise copies of the dataset to identify the complete set of relevant
      variables—critical in remote sensing where correlated spectral bands,
      indices, and terrain derivatives often carry complementary signal.
    * **Eliminates impurity/split bias using SHAP values**: Classical Boruta
      relies on Gini / mean-decrease-in-impurity (MDI) importance, which is
      biased towards high-cardinality continuous features, scales inconsistently
      under collinearity, and restricts selection to tree models. Replacing MDI
      with game-theoretic **SHAP** attributions yields consistent, locally
      accurate, model-agnostic importance estimates across tree ensembles,
      linear models, and general classifiers.
    * **Captures non-linear and multiclass interactions with error control**:
      Because SHAP values are evaluated on a trained classifier (defaulting to
      ``RandomForestClassifier``) across repeated trials and compared against
      the shadow threshold (e.g., ``ShadowMax``) via exact binomial testing
      (``scipy.stats.binomtest``), non-linear multi-band relationships are
      detected while controlling false discoveries.
    * **Self-contained modern SciPy/SHAP implementation**: Implements the
      Boruta-SHAP algorithm directly using ``shap`` and
      ``scipy.stats.binomtest``, avoiding compatibility issues present in older
      unmaintained third-party ``BorutaShap`` packages.

    Requires optional dependencies ``shap`` and ``scipy``.

    :param cls_cfg: Optional :class:`~rsgispy.classification.classdata.ClassificationConfig`,
                    JSON config path, or mapping of ``{class_name: ClassInfoObj}``.
    :param train_data: Training sample :class:`pandas.DataFrame` or file path.
    :param sk_classifier: Optional ``scikit-learn`` classifier template
                          (defaults to ``RandomForestClassifier(n_estimators=100)``).
    :param feature_cols: Optional ordered feature column names (at least 2).
    :param n_trials: Maximum number of Boruta shadow-feature trials
                     (Default = ``50``).
    :param pvalue: Significance level for exact binomial acceptance/rejection
                   testing against ``p = 0.5`` (Default = ``0.05``).
    :param percentile: Shadow importance percentile threshold ``(0, 100]``
                       against which real features compete (``100.0`` =
                       ``ShadowMax``, Default = ``100.0``).
    :param bonferroni: Apply Bonferroni multiple-testing correction
                       (``pvalue / n_features``) when ``True`` (Default = ``True``).
    :param include_tentative: Include ``'Tentative'`` features alongside
                              ``'Confirmed'`` features in the returned list
                              (Default = ``False``).
    :param tentative_rough_fix: Promote any remaining ``'Tentative'`` features
                                whose median SHAP importance exceeds the median
                                shadow threshold to ``'Confirmed'`` (and reject
                                the rest) after ``n_trials`` (Default = ``False``).
    :param target_class: Optional class name or class ID to target SHAP
                         attributions (when ``None``, averages ``|SHAP|``
                         across all classes for multiclass classifiers).
    :param subsample_n_smpls: Optional sample cap for SHAP evaluation inside each
                              trial (speeds up large datasets while fitting on
                              all rows).
    :param use_tree_explainer: Explicitly force ``shap.TreeExplainer``
                               (``True``), disable it (``False``), or
                               auto-detect based on the estimator (``None``,
                               default).
    :param use_linear_explainer: Use ``shap.LinearExplainer`` when ``True``
                                 (Default = ``False``).
    :param use_internal_id: Train against ``internal_cls_id`` when ``True``.
    :param rnd_seed: Optional random seed for reproducible shadow shuffling,
                     estimator training, and subsampling.
    :param out_id_col: Optional override for output class ID column name.
    :param internal_id_col: Optional override for internal class ID column name.
    :param out_scores_file: Optional CSV path saving per-feature BorutaShap
                            decisions, hit counts, median importances, and
                            binomial p-values.
    :param out_plot_file: Optional PNG path saving colour-coded horizontal
                          BorutaShap importance boxplots alongside
                          ``ShadowMax`` and ``ShadowMean``.
    :param use_pbar: Display a tqdm progress bar across trials (Default = ``True``).
    :param return_scores: If ``True``, return ``(selected_features, scores_df)``.
    :return: List of selected feature names (or ``(selected_features, scores_df)``).

    .. code:: python

        from sklearn.ensemble import RandomForestClassifier
        import rsgispy.classification.classfeatures as cf

        selected, summary_df = cf.select_features_boruta_shap(
            "cls_config.json",
            "train.parquet",
            sk_classifier=RandomForestClassifier(n_estimators=100),
            n_trials=40,
            rnd_seed=42,
            out_scores_file="boruta_shap.csv",
            out_plot_file="boruta_shap.png",
            return_scores=True,
        )
    """
    # pylint: disable=too-many-locals
    check_sklearn_available("select_features_boruta_shap")
    _, feat_cols, x_df, y_arr, class_info, _, _, _ = _prepare_classification_data(
        samples_data=train_data,
        cls_cfg=cls_cfg,
        feature_cols=feature_cols,
        use_internal_id=use_internal_id,
        out_id_col=out_id_col,
        internal_id_col=internal_id_col,
        label="train_data",
        require_target=True,
    )
    assert x_df is not None and y_arr is not None

    target_idx: Optional[int] = None
    if target_class is not None:
        unique_ids = sorted(np.unique(y_arr).tolist())
        id_to_name = {
            int(
                obj.internal_id
                if use_internal_id and obj.internal_id is not None
                else obj.out_id
            ): name
            for name, obj in class_info.items()
        }
        if isinstance(target_class, str):
            matched_ids = [
                cid for cid, cname in id_to_name.items() if cname == target_class
            ]
            if not matched_ids or matched_ids[0] not in unique_ids:
                raise rsgispy.RSGISPyException(
                    f"target_class '{target_class}' not found in training labels."
                )
            target_idx = unique_ids.index(matched_ids[0])
        elif isinstance(target_class, (int, np.integer)) and not isinstance(
            target_class, bool
        ):
            t_int = int(target_class)
            if t_int in unique_ids:
                target_idx = unique_ids.index(t_int)
            elif use_internal_id and class_info:
                mapped = [
                    int(obj.internal_id)
                    for obj in class_info.values()
                    if int(obj.out_id) == t_int and obj.internal_id is not None
                ]
                if mapped and mapped[0] in unique_ids:
                    target_idx = unique_ids.index(mapped[0])
                else:
                    raise rsgispy.RSGISPyException(
                        f"target_class ID {t_int} not found in training labels {unique_ids}."
                    )
            else:
                raise rsgispy.RSGISPyException(
                    f"target_class ID {t_int} not found in training labels {unique_ids}."
                )
        else:
            raise rsgispy.RSGISPyException(
                "target_class must be None, a class name string, or an integer class ID."
            )

    selected, scores_df = run_boruta_shap_selection(
        train_x=x_df,
        train_y=y_arr,
        feature_names=feat_cols,
        feature_name="select_features_boruta_shap",
        extra="classification",
        sk_classifier=sk_classifier,
        n_trials=n_trials,
        pvalue=pvalue,
        percentile=percentile,
        bonferroni=bonferroni,
        include_tentative=include_tentative,
        tentative_rough_fix=tentative_rough_fix,
        target_class_idx=target_idx,
        subsample_n_smpls=subsample_n_smpls,
        use_tree_explainer=use_tree_explainer,
        use_linear_explainer=use_linear_explainer,
        rnd_seed=rnd_seed,
        out_plot_file=out_plot_file,
        use_pbar=use_pbar,
    )

    if out_scores_file is not None:
        scores_df.to_csv(ensure_parent_dir(out_scores_file), index=False)
    if return_scores:
        return selected, scores_df
    return selected


# ---------------------------------------------------------------------------
# Feature Importance, Contribution, Response Curves & Sensitivity Analysis
# ---------------------------------------------------------------------------


# pylint: disable=too-many-arguments,too-many-positional-arguments
def calc_feature_importance(
    cls_cfg: Optional[ClsConfigInput],
    sk_classifier: Any,
    eval_data: SamplesInput,
    feature_cols: Optional[Sequence[str]] = None,
    method: str = "permutation",
    scoring: Optional[Union[str, Any]] = "accuracy",
    n_repeats: int = 10,
    use_internal_id: bool = True,
    rnd_seed: Optional[int] = None,
    out_id_col: Optional[str] = None,
    internal_id_col: Optional[str] = None,
    n_jobs: int = 1,
    out_importance_file: Optional[Union[str, os.PathLike]] = None,
    out_plot_file: Optional[Union[str, os.PathLike]] = None,
) -> pd.DataFrame:
    """
    Compute feature importances for a fitted ``scikit-learn`` classifier using
    permutation importance (``sklearn.inspection.permutation_importance``),
    native estimator weights (``feature_importances_`` / ``coef_``), or both.

    :param cls_cfg: Optional classification configuration or class mapping.
    :param sk_classifier: Fitted ``scikit-learn`` classifier.
    :param eval_data: Evaluation sample :class:`pandas.DataFrame` or file path
                      (typically test or validation split for permutation).
    :param feature_cols: Optional ordered feature column names.
    :param method: Importance calculation method: ``'permutation'`` (default),
                   ``'model'`` (native ``feature_importances_`` / ``coef_``), or
                   ``'both'``.
    :param scoring: Scoring metric passed to ``permutation_importance``
                    (Default = ``'accuracy'``).
    :param n_repeats: Number of permutation repeats per feature (Default = ``10``).
    :param use_internal_id: Evaluate against ``internal_cls_id`` when ``True``.
    :param rnd_seed: Optional random seed for ``permutation_importance``.
    :param out_id_col: Optional override for output class ID column name.
    :param internal_id_col: Optional override for internal class ID column name.
    :param n_jobs: Parallel jobs for ``permutation_importance`` (Default = ``1``).
    :param out_importance_file: Optional CSV output path for the importance table.
    :param out_plot_file: Optional PNG horizontal bar chart output path.
    :return: Sorted :class:`pandas.DataFrame` of feature importances and ranks.

    .. code:: python

        import rsgispy.classification.classfeatures as cf

        imp_df = cf.calc_feature_importance(
            "cls_config.json", clf, "test.parquet", method="both",
            out_importance_file="feat_imp.csv", out_plot_file="feat_imp.png",
        )
    """
    # pylint: disable=too-many-locals
    check_sklearn_available("calc_feature_importance")
    check_sklearn_estimator_fitted(sk_classifier, label="sk_classifier")

    method_key = method.strip().lower()
    if method_key in {"model", "native", "gini", "impurity"}:
        eff_method = "model"
    elif method_key in {"permutation", "perm"}:
        eff_method = "permutation"
    elif method_key == "both":
        eff_method = "both"
    else:
        raise rsgispy.RSGISPyException(
            f"Unsupported importance method '{method}'. Expected 'permutation', "
            "'model', or 'both'."
        )

    _, feat_cols, x_df, y_arr, _, _, _, _ = _prepare_classification_data(
        samples_data=eval_data,
        cls_cfg=cls_cfg,
        feature_cols=feature_cols,
        use_internal_id=use_internal_id,
        out_id_col=out_id_col,
        internal_id_col=internal_id_col,
        label="eval_data",
        require_target=True,
    )
    assert x_df is not None and y_arr is not None

    data: Dict[str, Any] = {"feature": feat_cols}
    sort_col = "importance_mean"
    err_vals: Optional[np.ndarray] = None

    if eff_method in {"model", "both"}:
        native_imp = extract_native_feature_importances(
            sk_classifier, len(feat_cols)
        )
        if native_imp is None:
            if eff_method == "model":
                raise rsgispy.RSGISPyException(
                    "Classifier does not expose 'feature_importances_' or 'coef_'."
                )
        else:
            data["model_importance"] = native_imp
            if eff_method == "model":
                sort_col = "model_importance"

    if eff_method in {"permutation", "both"}:
        perm_res = sklearn.inspection.permutation_importance(
            sk_classifier,
            x_df,
            y_arr,
            scoring=scoring,
            n_repeats=int(n_repeats),
            random_state=rnd_seed,
            n_jobs=n_jobs,
        )
        data["importance_mean"] = np.asarray(
            perm_res.importances_mean, dtype=np.float64
        )
        data["importance_std"] = np.asarray(
            perm_res.importances_std, dtype=np.float64
        )
        err_vals = data["importance_std"]

    imp_df = pd.DataFrame(data)
    imp_df = imp_df.sort_values(sort_col, ascending=False).reset_index(drop=True)
    imp_df["rank"] = np.arange(1, len(imp_df) + 1, dtype=int)

    if out_importance_file is not None:
        imp_df.to_csv(ensure_parent_dir(out_importance_file), index=False)

    if out_plot_file is not None:
        plot_errs = imp_df["importance_std"].to_numpy() if err_vals is not None else None
        xlabel = (
            "Permutation Importance (Mean Score Decrease)"
            if eff_method in {"permutation", "both"}
            else "Model Feature Importance"
        )
        plot_feature_importance_bars(
            features=imp_df["feature"].tolist(),
            importances=imp_df[sort_col].to_numpy(),
            out_plot=out_plot_file,
            errors=plot_errs,
            title="Classification Feature Importance",
            xlabel=xlabel,
        )

    return imp_df


# pylint: disable=too-many-arguments,too-many-positional-arguments
def calc_jackknife_feature_importance(
    cls_cfg: Optional[ClsConfigInput],
    sk_classifier: Any,
    train_data: SamplesInput,
    test_data: SamplesInput,
    feature_cols: Optional[Sequence[str]] = None,
    use_internal_id: bool = True,
    rnd_seed: Optional[int] = None,
    out_id_col: Optional[str] = None,
    internal_id_col: Optional[str] = None,
    out_var_importance_file: Optional[Union[str, os.PathLike]] = None,
    out_importance_acc_plot: Optional[Union[str, os.PathLike]] = None,
    out_importance_f1_plot: Optional[Union[str, os.PathLike]] = None,
    use_pbar: bool = True,
) -> pd.DataFrame:
    """
    Evaluate classification feature importance via jackknife leave-one-out and
    single-feature training runs using clean cloned estimators in each iteration.

    Computes training accuracy and testing accuracy, balanced accuracy, macro F1,
    and Cohen's kappa across every feature drop/solo subset plus the full-feature
    baseline.

    :param cls_cfg: Optional classification configuration or class mapping.
    :param sk_classifier: Base ``scikit-learn`` classifier template.
    :param train_data: Training sample :class:`pandas.DataFrame` or file path.
    :param test_data: Testing sample :class:`pandas.DataFrame` or file path.
    :param feature_cols: Optional ordered feature column names (at least 2).
    :param use_internal_id: Train/evaluate using ``internal_cls_id`` when ``True``.
    :param rnd_seed: Optional random seed passed to cloned estimators.
    :param out_id_col: Optional override for output class ID column name.
    :param internal_id_col: Optional override for internal class ID column name.
    :param out_var_importance_file: Optional CSV output path for jackknife metrics.
    :param out_importance_acc_plot: Optional horizontal bar plot path for test accuracy.
    :param out_importance_f1_plot: Optional horizontal bar plot path for macro F1.
    :param use_pbar: Display a tqdm progress bar (Default = ``True``).
    :return: :class:`pandas.DataFrame` indexed by ``feature_cols``.

    .. code:: python

        from sklearn.ensemble import RandomForestClassifier
        import rsgispy.classification.classfeatures as cf

        jack_df = cf.calc_jackknife_feature_importance(
            "cls_config.json",
            RandomForestClassifier(n_estimators=50, random_state=42),
            "train.parquet",
            "test.parquet",
            out_var_importance_file="jackknife.csv",
            out_importance_acc_plot="jack_acc.png",
        )
    """
    # pylint: disable=too-many-locals
    check_sklearn_available("calc_jackknife_feature_importance")
    _, feat_cols, train_x_all, train_y, _, _, _, _ = _prepare_classification_data(
        samples_data=train_data,
        cls_cfg=cls_cfg,
        feature_cols=feature_cols,
        use_internal_id=use_internal_id,
        out_id_col=out_id_col,
        internal_id_col=internal_id_col,
        label="train_data",
        require_target=True,
    )
    if len(feat_cols) < 2:
        raise rsgispy.RSGISPyException(
            "Jackknife feature importance requires at least 2 features."
        )
    _, _, test_x_all, test_y, _, _, _, _ = _prepare_classification_data(
        samples_data=test_data,
        cls_cfg=cls_cfg,
        feature_cols=feat_cols,
        use_internal_id=use_internal_id,
        out_id_col=out_id_col,
        internal_id_col=internal_id_col,
        label="test_data",
        require_target=True,
    )
    assert train_x_all is not None and train_y is not None
    assert test_x_all is not None and test_y is not None

    def _eval_subset(sub_vars: List[str]) -> Dict[str, float]:
        tr_sub = train_x_all[sub_vars]
        te_sub = test_x_all[sub_vars]
        est_sub = fit_clean_sklearn_estimator(
            sk_classifier,
            tr_sub,
            train_y,
            sync_input=False,
            random_state=rnd_seed,
            extra="classification",
        )
        res = calc_split_metrics(est_sub, tr_sub, train_y, "train")
        res.update(calc_split_metrics(est_sub, te_sub, test_y, "test"))
        return res

    all_metrics = _eval_subset(feat_cols)

    out_data: Dict[str, List[float]] = {
        "train_acc_leave_var_out": [],
        "test_acc_leave_var_out": [],
        "test_bal_acc_leave_var_out": [],
        "test_f1_macro_leave_var_out": [],
        "test_kappa_leave_var_out": [],
        "train_acc_only_var": [],
        "test_acc_only_var": [],
        "test_bal_acc_only_var": [],
        "test_f1_macro_only_var": [],
        "test_kappa_only_var": [],
        "all_vars_test_accuracy": [],
        "all_vars_test_f1_macro": [],
    }

    pbar = rsgispy.get_tqdm()
    for feat in pbar(feat_cols, disable=not use_pbar):
        m_out = _eval_subset([f for f in feat_cols if f != feat])
        out_data["train_acc_leave_var_out"].append(m_out["train_accuracy"])
        out_data["test_acc_leave_var_out"].append(m_out["test_accuracy"])
        out_data["test_bal_acc_leave_var_out"].append(m_out["test_bal_accuracy"])
        out_data["test_f1_macro_leave_var_out"].append(m_out["test_f1_macro"])
        out_data["test_kappa_leave_var_out"].append(m_out["test_cohen_kappa"])

    for feat in pbar(feat_cols, disable=not use_pbar):
        m_only = _eval_subset([feat])
        out_data["train_acc_only_var"].append(m_only["train_accuracy"])
        out_data["test_acc_only_var"].append(m_only["test_accuracy"])
        out_data["test_bal_acc_only_var"].append(m_only["test_bal_accuracy"])
        out_data["test_f1_macro_only_var"].append(m_only["test_f1_macro"])
        out_data["test_kappa_only_var"].append(m_only["test_cohen_kappa"])
        out_data["all_vars_test_accuracy"].append(all_metrics["test_accuracy"])
        out_data["all_vars_test_f1_macro"].append(all_metrics["test_f1_macro"])

    jack_df = pd.DataFrame(data=out_data, index=feat_cols)
    if out_var_importance_file is not None:
        jack_df.to_csv(ensure_parent_dir(out_var_importance_file), index=True)

    if out_importance_acc_plot is not None:
        plot_jackknife_bars(
            feat_cols,
            out_data["test_acc_leave_var_out"],
            out_data["test_acc_only_var"],
            all_metrics["test_accuracy"],
            "Jackknife Accuracy Gain",
            "Testing Accuracy",
            out_importance_acc_plot,
        )

    if out_importance_f1_plot is not None:
        plot_jackknife_bars(
            feat_cols,
            out_data["test_f1_macro_leave_var_out"],
            out_data["test_f1_macro_only_var"],
            all_metrics["test_f1_macro"],
            "Jackknife Macro F1 Gain",
            "Testing Macro F1",
            out_importance_f1_plot,
        )

    return jack_df


# pylint: disable=too-many-arguments,too-many-positional-arguments
def calc_feature_response_curves(
    cls_cfg: Optional[ClsConfigInput],
    sk_classifier: Any,
    train_data: SamplesInput,
    feature_cols: Optional[Sequence[str]] = None,
    out_file: Optional[Union[str, os.PathLike]] = None,
    response_plots_dir: Optional[Union[str, os.PathLike]] = None,
    n_samples: int = 100,
    target_class: Optional[Union[str, int]] = None,
    use_internal_id: bool = True,
    out_id_col: Optional[str] = None,
    internal_id_col: Optional[str] = None,
    summary_stat: str = "mean",
    use_pbar: bool = True,
) -> pd.DataFrame:
    """
    Generate marginal response curves showing how predicted class probabilities
    vary across each feature's range while holding other predictors constant
    at their sample mean or median.

    :param cls_cfg: Optional classification configuration or class mapping.
    :param sk_classifier: Fitted ``scikit-learn`` classifier.
    :param train_data: Reference sample :class:`pandas.DataFrame` or file path
                       used to compute feature ranges and baseline constants.
    :param feature_cols: Optional ordered feature column names.
    :param out_file: Optional output CSV path storing grid values and class
                     probabilities.
    :param response_plots_dir: Optional directory saving ``<feature>_response_curve.png``
                               figures.
    :param n_samples: Number of evaluation steps across each feature range
                      (Default = ``100``).
    :param target_class: Optional single class name or ID to isolate (when
                         ``None``, evaluates and plots curves for all classes).
    :param use_internal_id: Map class IDs via ``internal_cls_id`` when ``True``.
    :param out_id_col: Optional override for output class ID column name.
    :param internal_id_col: Optional override for internal class ID column name.
    :param summary_stat: Baseline value used for held-constant features:
                         ``'mean'`` (default) or ``'median'``.
    :param use_pbar: Display a tqdm progress bar (Default = ``True``).
    :return: :class:`pandas.DataFrame` containing ``<feat>_vals`` and
             ``<feat>_<class>_prob`` columns.

    .. code:: python

        import rsgispy.classification.classfeatures as cf

        resp_df = cf.calc_feature_response_curves(
            "cls_config.json", clf, "train.parquet",
            out_file="resp.csv", response_plots_dir="resp_plots",
        )
    """
    # pylint: disable=too-many-locals
    check_sklearn_available("calc_feature_response_curves")
    check_sklearn_estimator_fitted(sk_classifier, label="sk_classifier")
    if n_samples < 2:
        raise rsgispy.RSGISPyException("n_samples must be at least 2.")

    stat_key = summary_stat.strip().lower()
    if stat_key not in {"mean", "median"}:
        raise rsgispy.RSGISPyException(
            f"Unsupported summary_stat '{summary_stat}'. Expected 'mean' or 'median'."
        )

    _, feat_cols, x_df, _, class_info, _, _, _ = _prepare_classification_data(
        samples_data=train_data,
        cls_cfg=cls_cfg,
        feature_cols=feature_cols,
        use_internal_id=use_internal_id,
        out_id_col=out_id_col,
        internal_id_col=internal_id_col,
        label="train_data",
        require_target=False,
    )
    assert x_df is not None

    est_classes, est_labels = _resolve_estimator_class_labels(
        sk_classifier, class_info, use_internal_id
    )
    if target_class is not None:
        t_idx = _resolve_target_class_index(
            sk_classifier,
            class_info,
            target_class,
            use_internal_id,
            default_binary_pos=False,
        )
        plot_indices = [t_idx]
    else:
        plot_indices = list(range(max(1, len(est_classes))))

    n_outputs = max(1, len(est_classes), (max(plot_indices) + 1 if plot_indices else 1))
    resolved_labels = [
        est_labels[idx] if idx < len(est_labels) else f"cls_{idx}"
        for idx in range(n_outputs)
    ]

    return run_variable_response_curves(
        x_df=x_df,
        feature_cols=feat_cols,
        predict_fn=lambda df: _predict_class_probs(sk_classifier, df, est_classes),
        output_labels=resolved_labels,
        out_indices=plot_indices,
        col_name_fn=lambda feat, lbl, _i: f"{feat}_{lbl}_prob",
        n_samples=n_samples,
        summary_stat=summary_stat,
        method="marginal",
        out_file=out_file,
        response_plots_dir=response_plots_dir,
        ylabel="Class Probability",
        ylim=(0.0, 1.0),
        multi_panel=False,
        use_pbar=use_pbar,
    )


# pylint: disable=too-many-arguments,too-many-positional-arguments
def calc_shap_feature_contributions(
    cls_cfg: Optional[ClsConfigInput],
    sk_classifier: Any,
    samples_data: SamplesInput,
    feature_cols: Optional[Sequence[str]] = None,
    target_class: Optional[Union[str, int]] = None,
    use_internal_id: bool = True,
    out_id_col: Optional[str] = None,
    internal_id_col: Optional[str] = None,
    shap_summary_plot: Optional[Union[str, os.PathLike]] = None,
    shap_heatmap_plot: Optional[Union[str, os.PathLike]] = None,
    shap_scatter_plots_dir: Optional[Union[str, os.PathLike]] = None,
    shap_depend_plots_dir: Optional[Union[str, os.PathLike]] = None,
    out_shap_file: Optional[Union[str, os.PathLike]] = None,
    subsample_n_smpls: Optional[int] = None,
    use_tree_explainer: bool = False,
    use_linear_explainer: bool = False,
    rnd_seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Compute SHAP feature contributions for a fitted ``scikit-learn`` classifier
    and generate summary, heatmap, scatter, and partial dependence plots.

    Requires optional dependency ``shap``.

    :param cls_cfg: Optional classification configuration or class mapping.
    :param sk_classifier: Fitted ``scikit-learn`` classifier.
    :param samples_data: Sample :class:`pandas.DataFrame` or file path.
    :param feature_cols: Optional ordered feature column names.
    :param target_class: Optional class name or class ID used for single-class
                         SHAP heatmap, scatter, and dependence plots (and focused
                         summary plot; when ``None``, summary plot shows all
                         classes for multiclass models).
    :param use_internal_id: Map class IDs via ``internal_cls_id`` when ``True``.
    :param out_id_col: Optional override for output class ID column name.
    :param internal_id_col: Optional override for internal class ID column name.
    :param shap_summary_plot: Optional output path for SHAP summary plot.
    :param shap_heatmap_plot: Optional output path for SHAP heatmap plot.
    :param shap_scatter_plots_dir: Optional directory for per-feature SHAP scatter plots.
    :param shap_depend_plots_dir: Optional directory for per-feature partial dependence plots.
    :param out_shap_file: Optional output CSV path saving mean ``|SHAP|`` values.
    :param subsample_n_smpls: Optional subsample count for SHAP evaluation.
    :param use_tree_explainer: Use ``shap.TreeExplainer`` when ``True``.
    :param use_linear_explainer: Use ``shap.LinearExplainer`` when ``True``.
    :param rnd_seed: Optional random seed for subsampling.
    :return: :class:`pandas.DataFrame` of mean absolute SHAP contributions per
             feature (and per class when multiclass).

    .. code:: python

        import rsgispy.classification.classfeatures as cf

        shap_df = cf.calc_shap_feature_contributions(
            "cls_config.json", clf, "train.parquet",
            shap_summary_plot="shap_summary.png",
            use_tree_explainer=True,
        )
    """
    # pylint: disable=too-many-locals
    check_sklearn_available("calc_shap_feature_contributions")
    check_sklearn_estimator_fitted(sk_classifier, label="sk_classifier")

    _, feat_cols, x_df, _, class_info, _, _, _ = _prepare_classification_data(
        samples_data=samples_data,
        cls_cfg=cls_cfg,
        feature_cols=feature_cols,
        use_internal_id=use_internal_id,
        out_id_col=out_id_col,
        internal_id_col=internal_id_col,
        label="samples_data",
        require_target=False,
    )
    assert x_df is not None

    _, class_labels = _resolve_estimator_class_labels(
        sk_classifier, class_info, use_internal_id
    )
    target_idx: Optional[int]
    if target_class is not None:
        target_idx = _resolve_target_class_index(
            sk_classifier,
            class_info,
            target_class,
            use_internal_id,
            default_binary_pos=True,
        )
    else:
        target_idx = None

    summary_df = run_shap_explainer(
        est_cls_obj=sk_classifier,
        train_x=x_df,
        analysis_vars=feat_cols,
        feature_name="calc_shap_feature_contributions",
        extra="classification",
        shap_summary_plot=shap_summary_plot,
        shap_heatmap_plot=shap_heatmap_plot,
        shap_scatter_plots_dir=shap_scatter_plots_dir,
        shap_depend_plots_dir=shap_depend_plots_dir,
        subsample_n_smpls=subsample_n_smpls,
        use_tree_explainer=use_tree_explainer,
        use_linear_explainer=use_linear_explainer,
        target_class_idx=target_idx,
        class_labels=class_labels if class_labels else None,
        rnd_seed=rnd_seed,
    )
    if out_shap_file is not None:
        summary_df.to_csv(ensure_parent_dir(out_shap_file), index=False)
    return summary_df


# pylint: disable=too-many-arguments,too-many-positional-arguments
def calc_salib_feature_sensitivity(
    cls_cfg: Optional[ClsConfigInput],
    sk_classifier: Any,
    samples_data: SamplesInput,
    feature_cols: Optional[Sequence[str]] = None,
    target_class: Optional[Union[str, int]] = None,
    use_internal_id: bool = True,
    out_id_col: Optional[str] = None,
    internal_id_col: Optional[str] = None,
    n_samples: int = 256,
    sobol_plot_file: Optional[Union[str, os.PathLike]] = None,
    sobol_overall_file: Optional[Union[str, os.PathLike]] = None,
    sobol_first_file: Optional[Union[str, os.PathLike]] = None,
    sobol_second_file: Optional[Union[str, os.PathLike]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform SALib Sobol sensitivity analysis quantifying first-order,
    second-order, and total-order feature sensitivity for predicted class
    probabilities of ``sk_classifier``.

    Requires optional dependency ``SALib``.

    :param cls_cfg: Optional classification configuration or class mapping.
    :param sk_classifier: Fitted ``scikit-learn`` classifier.
    :param samples_data: Sample :class:`pandas.DataFrame` or file path defining
                         empirical feature bounds.
    :param feature_cols: Optional ordered feature column names.
    :param target_class: Target class name or ID whose predicted probability is
                         analysed (defaults to positive class for binary models,
                         or the first class for multiclass models).
    :param use_internal_id: Map class IDs via ``internal_cls_id`` when ``True``.
    :param out_id_col: Optional override for output class ID column name.
    :param internal_id_col: Optional override for internal class ID column name.
    :param n_samples: Base Saltelli sample count (Default = ``256``).
    :param sobol_plot_file: Optional output Sobol sensitivity bar plot PNG path.
    :param sobol_overall_file: Optional total-order Sobol CSV output path.
    :param sobol_first_file: Optional first-order Sobol CSV output path.
    :param sobol_second_file: Optional second-order Sobol CSV output path.
    :return: Tuple ``(total_Si, first_Si, second_Si)`` DataFrames.

    .. code:: python

        import rsgispy.classification.classfeatures as cf

        total_si, first_si, second_si = cf.calc_salib_feature_sensitivity(
            "cls_config.json", clf, "train.parquet", n_samples=128
        )
    """
    # pylint: disable=too-many-locals
    check_sklearn_available("calc_salib_feature_sensitivity")
    check_sklearn_estimator_fitted(sk_classifier, label="sk_classifier")

    _, feat_cols, x_df, _, class_info, _, _, _ = _prepare_classification_data(
        samples_data=samples_data,
        cls_cfg=cls_cfg,
        feature_cols=feature_cols,
        use_internal_id=use_internal_id,
        out_id_col=out_id_col,
        internal_id_col=internal_id_col,
        label="samples_data",
        require_target=False,
    )
    assert x_df is not None

    est_classes, _ = _resolve_estimator_class_labels(
        sk_classifier, class_info, use_internal_id
    )
    target_idx = _resolve_target_class_index(
        sk_classifier,
        class_info,
        target_class,
        use_internal_id,
        default_binary_pos=True,
    )

    return run_salib_sobol_sensitivity(
        train_df=x_df,
        analysis_vars=feat_cols,
        predict_1d_fn=lambda grid_df: _predict_class_probs(
            sk_classifier, grid_df, est_classes
        )[:, target_idx],
        feature_name="calc_salib_feature_sensitivity",
        extra="classification",
        n_samples=n_samples,
        sobol_plot_file=sobol_plot_file,
        sobol_overall_file=sobol_overall_file,
        sobol_first_file=sobol_first_file,
        sobol_second_file=sobol_second_file,
    )
