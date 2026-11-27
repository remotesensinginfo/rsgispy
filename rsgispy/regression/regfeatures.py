# pylint: disable=too-many-lines
"""
Feature exploration, multicollinearity diagnostics, feature selection, response
function curves, and interpretability analysis for regression workflows.

Provides predictor-vs-target scatter diagnostics, collinearity analysis
(correlation matrices and VIF), supervised and unsupervised feature selection
(CQV, correlation clustering, iterative VIF pruning, univariate F/MI tests,
RFE/RFECV, SelectFromModel/LassoLars, and BorutaShap), and model
interpretability tools (permutation & native feature importance, jackknife
analysis, univariate/multivariate response curves, SHAP contributions, and
SALib Sobol global sensitivity analysis).
"""

import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

import rsgispy
from rsgispy.regression._reg_common import (
    RegConfigInput,
    SamplesInput,
    TargetColsInput,
    calc_reg_split_metrics,
    check_sklearn_available,
    prepare_reg_data,
)
from rsgispy.regression.regdata import normalise_target_cols
from rsgispy.tools._ml_explain import (
    finalise_plot,
    plot_jackknife_bars,
    run_boruta_shap_selection,
    run_salib_sobol_sensitivity,
    run_shap_explainer,
    run_variable_response_curves,
)
from rsgispy.tools._ml_features import (
    run_corr_cluster_selection,
    run_correlation_matrix,
    run_feature_importance,
    run_from_model_selection,
    run_iterative_vif_selection,
    run_rfe_selection,
    run_univariate_selection,
)
from rsgispy.tools._sklearn import (
    check_sklearn_estimator_fitted,
    fit_clean_sklearn_estimator,
    sklearn,
)
from rsgispy.tools.featureselection import (
    calc_pandas_vif,
    cqv_threshold,
)
from rsgispy.tools.files import ensure_parent_dir
from rsgispy.tools.plotting import CATEGORICAL_COLOURS, get_fig_ax, use_plot_style

__all__ = [
    "target_scatter_plots",
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


def _resolve_single_target(
    eff_targets: Sequence[str],
    y_arr: Optional[np.ndarray],
    target_var: Optional[str],
) -> Tuple[Optional[int], List[str], Optional[np.ndarray]]:
    """Slice ``y_arr`` and target list when ``target_var`` is specified."""
    target_list = list(eff_targets)
    if target_var is None:
        return None, target_list, y_arr
    if target_var not in target_list:
        raise rsgispy.RSGISPyException(
            f"target_var '{target_var}' not found in target columns "
            f"{target_list}."
        )
    idx = target_list.index(target_var)
    if y_arr is None:
        return idx, [target_var], None
    y_1d = y_arr[:, idx] if y_arr.ndim == 2 else y_arr
    return idx, [target_var], y_1d


def _resolve_model_feature_cols(
    sk_regressor: Any,
    df: pd.DataFrame,
    feat_cols: Sequence[str],
) -> List[str]:
    """Return ordered feature columns expected by fitted ``sk_regressor``."""
    raw_names = getattr(sk_regressor, "feature_names_in_", None)
    if raw_names is None and hasattr(sk_regressor, "estimators_"):
        ests = getattr(sk_regressor, "estimators_", None)
        if ests:
            raw_names = getattr(ests[0], "feature_names_in_", None)
    if raw_names is None:
        raw_names = getattr(sk_regressor, "feature_name_", None)
    if raw_names is None:
        raw_names = getattr(sk_regressor, "feature_names_", None)
    if raw_names is not None:
        model_cols = [str(c) for c in raw_names]
        missing = [c for c in model_cols if c not in df.columns]
        if missing:
            raise rsgispy.RSGISPyException(
                f"Model expects feature column(s) {missing} which were not "
                f"found in input data columns {list(df.columns)}."
            )
        return model_cols
    return list(feat_cols)


def _resolve_estimator_target_labels(
    sk_regressor: Any,
    x_df: pd.DataFrame,
    eff_targets: Sequence[str],
) -> Tuple[int, List[str]]:
    """Infer fitted regressor output count and resolve output label names."""
    probe = np.asarray(sk_regressor.predict(x_df.iloc[:1]), dtype=np.float64)
    n_outputs = 1 if probe.ndim == 1 else int(probe.shape[1])
    t_list = list(eff_targets)
    if len(t_list) == n_outputs:
        return n_outputs, t_list
    if n_outputs == 1 and t_list:
        return 1, [t_list[0]]
    if n_outputs == 1:
        return 1, ["prediction"]
    return n_outputs, [
        t_list[i] if i < len(t_list) else f"target_{i}"
        for i in range(n_outputs)
    ]


def _predict_1d_target(
    sk_regressor: Any,
    grid_df: pd.DataFrame,
    target_idx: int,
) -> np.ndarray:
    """Return 1D float predictions for ``target_idx``."""
    preds = np.asarray(sk_regressor.predict(grid_df), dtype=np.float64)
    if preds.ndim == 1:
        return preds
    return preds[:, target_idx]


# ---------------------------------------------------------------------------
# Exploratory Feature Diagnostics & Collinearity
# ---------------------------------------------------------------------------


@use_plot_style
def _plot_target_scatter(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    f_col: str,
    t_col: str,
    add_trendline: bool = True,
    ax: Optional[Any] = None,
) -> Tuple[Any, Any]:
    """Scatter one feature against one target, with an optional OLS trendline."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    fig, ax = get_fig_ax(ax, figsize=(6, 4.5))
    ax.scatter(
        x_vals,
        y_vals,
        color=CATEGORICAL_COLOURS[0],
        alpha=0.6,
        s=18,
    )
    r_val = 0.0
    if np.std(x_vals) > 0.0 and np.std(y_vals) > 0.0:
        r_val = float(np.corrcoef(x_vals, y_vals)[0, 1])
        if np.isnan(r_val):
            r_val = 0.0
    if add_trendline and np.std(x_vals) > 0.0 and x_vals.size >= 2:
        slope, intercept = np.polyfit(x_vals, y_vals, 1)
        x_line = np.linspace(float(np.min(x_vals)), float(np.max(x_vals)), 100)
        ax.plot(
            x_line,
            slope * x_line + intercept,
            color=CATEGORICAL_COLOURS[1],
            linestyle="--",
            label=f"Trend (r = {r_val:.3f})",
        )
        ax.legend(loc="best")
    ax.set_xlabel(f_col)
    ax.set_ylabel(t_col)
    ax.set_title(f"{t_col} against {f_col} (r = {r_val:.3f})")
    return fig, ax


# pylint: disable=too-many-arguments,too-many-positional-arguments
def target_scatter_plots(
    samples_data: SamplesInput,
    reg_cfg: Optional[RegConfigInput] = None,
    target_cols: TargetColsInput = None,
    feature_cols: Optional[Sequence[str]] = None,
    out_dir: Union[str, os.PathLike] = "scatter_plots",
    add_trendline: bool = True,
    use_pbar: bool = True,
) -> List[str]:
    """
    Create per-feature scatter plots against each continuous target variable
    with optional linear regression trendlines and Pearson correlation
    annotations. One file is written per feature and target, and each figure
    is closed after it is saved (see :ref:`plotting_conventions`).

    :param samples_data: Sample :class:`pandas.DataFrame` or table path
                         (Parquet, CSV, etc.).
    :param reg_cfg: Optional :class:`~rsgispy.regression.regdata.RegressionConfig`,
                    JSON config path, or sequence of ``ImageBandInfo`` objects.
    :param target_cols: Optional target column name(s) overriding ``reg_cfg``.
    :param feature_cols: Optional subset/order of feature column names.
    :param out_dir: Output directory where
                    ``<feature>_<target>_scatter.png`` plots are saved
                    (Default = ``'scatter_plots'``).
    :param add_trendline: If ``True`` (default), overlay a fitted ordinary
                          least-squares trendline and Pearson ``r`` label.
    :param use_pbar: Display a tqdm progress bar (Default = ``True``).
    :return: List of the plot files written.

    .. code:: python

        import rsgispy.regression.regfeatures as rf

        plot_files = rf.target_scatter_plots(
            "train.parquet",
            reg_cfg="reg_config.json",
            out_dir="scatter_plots",
        )
    """
    # pylint: disable=too-many-locals
    df, feat_cols, eff_targets, x_df, _ = prepare_reg_data(
        samples_data=samples_data,
        reg_cfg=reg_cfg,
        feature_cols=feature_cols,
        target_cols=target_cols,
        label="samples_data",
        require_target=True,
    )
    out_dir_str = os.fspath(out_dir)
    os.makedirs(out_dir_str, exist_ok=True)

    out_files: List[str] = []
    pairs = [(t_col, f_col) for t_col in eff_targets for f_col in feat_cols]
    pbar = rsgispy.get_tqdm()
    for t_col, f_col in pbar(pairs, disable=not use_pbar):
        fig, _ = _plot_target_scatter(
            x_df[f_col].to_numpy(dtype=np.float64),
            df[t_col].to_numpy(dtype=np.float64),
            f_col,
            t_col,
            add_trendline=add_trendline,
        )
        out_plot = os.path.join(out_dir_str, f"{f_col}_{t_col}_scatter.png")
        finalise_plot(fig, out_file=out_plot)
        out_files.append(out_plot)
    return out_files


# pylint: disable=too-many-arguments,too-many-positional-arguments
def correlation_matrix(
    samples_data: SamplesInput,
    reg_cfg: Optional[RegConfigInput] = None,
    target_cols: TargetColsInput = None,
    feature_cols: Optional[Sequence[str]] = None,
    include_targets: bool = False,
    method: str = "pearson",
    out_corr_file: Optional[Union[str, os.PathLike]] = "correlation_matrix.csv",
    out_plt_file: Optional[Union[str, os.PathLike]] = "correlation_matrix.png",
    fig_width: int = 15,
    fig_height: int = 14,
) -> pd.DataFrame:
    """
    Calculate pairwise correlations across regression predictors (and optionally
    target variables), saving the matrix CSV and lower-triangular heatmap.

    :param samples_data: Sample :class:`pandas.DataFrame` or table path.
    :param reg_cfg: Optional regression configuration or image band metadata.
    :param target_cols: Optional target column name(s) overriding ``reg_cfg``.
    :param feature_cols: Optional ordered feature column names.
    :param include_targets: If ``True``, include target column(s) in the
                            correlation matrix alongside predictors
                            (Default = ``False``).
    :param method: Correlation method: ``'pearson'``, ``'spearman'``, or
                   ``'kendall'`` (Default = ``'pearson'``).
    :param out_corr_file: Output correlation CSV path (or ``None`` to skip).
    :param out_plt_file: Output heatmap PNG path (or ``None`` to skip).
    :param fig_width: Heatmap figure width in inches (Default = ``15``).
    :param fig_height: Heatmap figure height in inches (Default = ``14``).
    :return: Correlation matrix :class:`pandas.DataFrame`.

    .. code:: python

        import rsgispy.regression.regfeatures as rf

        corr_df = rf.correlation_matrix(
            "train.parquet", reg_cfg="reg_config.json",
            include_targets=True,
            out_corr_file="corr.csv", out_plt_file="corr.png",
        )
    """
    df, feat_cols, eff_targets, _, _ = prepare_reg_data(
        samples_data=samples_data,
        reg_cfg=reg_cfg,
        feature_cols=feature_cols,
        target_cols=target_cols,
        label="samples_data",
        require_target=bool(include_targets),
    )
    cols = list(feat_cols) + (list(eff_targets) if include_targets else [])
    return run_correlation_matrix(
        df=df,
        cols=cols,
        method=method,
        out_corr_file=out_corr_file,
        out_plt_file=out_plt_file,
        fig_width=fig_width,
        fig_height=fig_height,
    )


def calc_vif_multicollinearity(
    samples_data: SamplesInput,
    reg_cfg: Optional[RegConfigInput] = None,
    target_cols: TargetColsInput = None,
    feature_cols: Optional[Sequence[str]] = None,
    out_vif_file: Optional[Union[str, os.PathLike]] = "vif_multicollinearity.csv",
) -> pd.Series:
    """
    Calculate Variance Inflation Factors (VIF) across regression feature
    columns using :func:`rsgispy.tools.featureselection.calc_pandas_vif`.

    :param samples_data: Sample :class:`pandas.DataFrame` or table path.
    :param reg_cfg: Optional regression configuration or image band metadata.
    :param target_cols: Optional target column name(s) excluded from auto-detected
                        features when ``feature_cols`` is ``None``.
    :param feature_cols: Optional ordered feature column names (at least 2).
    :param out_vif_file: Output CSV path for VIF scores (or ``None``).
    :return: :class:`pandas.Series` indexed by feature name containing VIF values.

    .. code:: python

        import rsgispy.regression.regfeatures as rf

        vif = rf.calc_vif_multicollinearity("train.parquet", reg_cfg="reg_config.json")
    """
    _, feat_cols, _, x_df, _ = prepare_reg_data(
        samples_data=samples_data,
        reg_cfg=reg_cfg,
        feature_cols=feature_cols,
        target_cols=target_cols,
        label="samples_data",
        require_target=False,
    )
    vif_scores = calc_pandas_vif(x_df, cols=feat_cols)
    if out_vif_file is not None:
        vif_scores.to_csv(ensure_parent_dir(out_vif_file), index=True)
    return vif_scores


# ---------------------------------------------------------------------------
# Feature Selection Methods
# ---------------------------------------------------------------------------


def select_features_cqv(
    samples_data: SamplesInput,
    reg_cfg: Optional[RegConfigInput] = None,
    target_cols: TargetColsInput = None,
    feature_cols: Optional[Sequence[str]] = None,
    low_threshold: float = 0.25,
    high_threshold: float = 0.75,
) -> List[str]:
    """
    Select regression features using the Coefficient of Quartile Variation
    (CQV) filter via :func:`rsgispy.tools.featureselection.cqv_threshold`.

    :param samples_data: Sample :class:`pandas.DataFrame` or table path.
    :param reg_cfg: Optional regression configuration or image band metadata.
    :param target_cols: Optional target column name(s) excluded from auto-detected
                        features.
    :param feature_cols: Optional ordered feature column names (at least 2).
    :param low_threshold: Minimum inclusive CQV threshold (Default = ``0.25``).
    :param high_threshold: Maximum inclusive CQV threshold (Default = ``0.75``).
    :return: List of selected feature column names.

    .. code:: python

        import rsgispy.regression.regfeatures as rf

        selected = rf.select_features_cqv(
            "train.parquet", reg_cfg="reg_config.json",
            low_threshold=0.1, high_threshold=0.9,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    if low_threshold > high_threshold:
        raise rsgispy.RSGISPyException(
            "low_threshold must be less than or equal to high_threshold."
        )
    _, feat_cols, _, x_df, _ = prepare_reg_data(
        samples_data=samples_data,
        reg_cfg=reg_cfg,
        feature_cols=feature_cols,
        target_cols=target_cols,
        label="samples_data",
        require_target=False,
    )
    return cqv_threshold(
        x_df,
        cols=feat_cols,
        lowthreshold=low_threshold,
        highthreshold=high_threshold,
    )


# pylint: disable=too-many-arguments,too-many-positional-arguments
def select_features_corr(
    samples_data: SamplesInput,
    reg_cfg: Optional[RegConfigInput] = None,
    target_cols: TargetColsInput = None,
    feature_cols: Optional[Sequence[str]] = None,
    n_min_clusters: int = 3,
    n_max_clusters: int = 25,
    use_pbar: bool = True,
) -> List[str]:
    """
    Select representative predictors via correlation hierarchical clustering and
    Silhouette optimisation using
    :func:`rsgispy.tools.featureselection.corr_feature_selection`. Supports
    both univariate and multivariate targets.

    :param samples_data: Sample :class:`pandas.DataFrame` or table path.
    :param reg_cfg: Optional regression configuration or image band metadata.
    :param target_cols: Optional target column name(s) overriding ``reg_cfg``.
    :param feature_cols: Optional ordered feature column names (at least 3).
    :param n_min_clusters: Minimum cluster count evaluated (Default = ``3``).
    :param n_max_clusters: Exclusive upper bound on cluster count
                           (clamped to ``len(feature_cols)`` when smaller).
    :param use_pbar: Display a tqdm progress bar (Default = ``True``).
    :return: List of selected feature column names.

    .. code:: python

        import rsgispy.regression.regfeatures as rf

        selected = rf.select_features_corr(
            "train.parquet", reg_cfg="reg_config.json",
            n_min_clusters=2, n_max_clusters=6,
        )
    """
    _, feat_cols, eff_targets, x_df, y_arr = prepare_reg_data(
        samples_data=samples_data,
        reg_cfg=reg_cfg,
        feature_cols=feature_cols,
        target_cols=target_cols,
        label="samples_data",
        require_target=True,
    )
    if y_arr is None:
        raise rsgispy.RSGISPyException(
            "Prepared target array is unexpectedly empty."
        )
    return run_corr_cluster_selection(
        x_df=x_df,
        y_arr=y_arr,
        feat_cols=feat_cols,
        target_names=eff_targets,
        n_min_clusters=n_min_clusters,
        n_max_clusters=n_max_clusters,
        use_pbar=use_pbar,
    )


def select_features_vif(
    samples_data: SamplesInput,
    reg_cfg: Optional[RegConfigInput] = None,
    target_cols: TargetColsInput = None,
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
    :param reg_cfg: Optional regression configuration or image band metadata.
    :param target_cols: Optional target column name(s) excluded from auto-detected
                        features.
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

        import rsgispy.regression.regfeatures as rf

        kept = rf.select_features_vif(
            "train.parquet", reg_cfg="reg_config.json", vif_threshold=5.0
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    _, feat_cols, _, x_df, _ = prepare_reg_data(
        samples_data=samples_data,
        reg_cfg=reg_cfg,
        feature_cols=feature_cols,
        target_cols=target_cols,
        label="samples_data",
        require_target=False,
    )
    return run_iterative_vif_selection(
        x_df=x_df,
        feat_cols=feat_cols,
        vif_threshold=vif_threshold,
        min_features=min_features,
        out_vif_file=out_vif_file,
        return_vif=return_vif,
    )


# pylint: disable=too-many-arguments,too-many-positional-arguments
def select_features_univariate(
    reg_cfg: Optional[RegConfigInput],
    train_data: SamplesInput,
    target_cols: TargetColsInput = None,
    feature_cols: Optional[Sequence[str]] = None,
    target_var: Optional[str] = None,
    score_func: str = "f_regression",
    k: Optional[Union[int, str]] = 10,
    percentile: Optional[int] = None,
    out_scores_file: Optional[Union[str, os.PathLike]] = None,
    rnd_seed: Optional[int] = None,
    return_scores: bool = False,
) -> Union[List[str], Tuple[List[str], pd.DataFrame]]:
    """
    Select features for univariate or multivariate regression using univariate
    statistical tests via ``scikit-learn`` ``SelectKBest`` or
    ``SelectPercentile``.

    For multivariate targets without ``target_var``, scores are computed per
    target column and averaged across targets.

    :param reg_cfg: Optional :class:`~rsgispy.regression.regdata.RegressionConfig`,
                    JSON config path, or sequence of ``ImageBandInfo`` objects.
    :param train_data: Training sample :class:`pandas.DataFrame` or file path.
    :param target_cols: Optional target column name(s) overriding ``reg_cfg``.
    :param feature_cols: Optional ordered feature column names.
    :param target_var: Optional single target variable name when ``train_data``
                       contains multiple targets.
    :param score_func: Scoring test: ``'f_regression'`` (default) or
                       ``'mutual_info_regression'``.
    :param k: Number of top features for ``SelectKBest`` (ignored when
              ``percentile`` is set; clamped to feature count if larger when
              positive int).
    :param percentile: Optional percentage ``(1..100]`` of top features to
                       keep using ``SelectPercentile``.
    :param out_scores_file: Optional CSV path saving per-feature scores/ranks.
    :param rnd_seed: Optional random seed passed to ``mutual_info_regression``.
    :param return_scores: If ``True``, return ``(selected_features, scores_df)``.
    :return: List of selected feature names (or ``(selected_features, scores_df)``).

    .. code:: python

        import rsgispy.regression.regfeatures as rf

        selected, scores_df = rf.select_features_univariate(
            "reg_config.json", "train.parquet",
            score_func="f_regression", k=5, return_scores=True,
        )
    """
    # pylint: disable=too-many-locals
    check_sklearn_available("select_features_univariate")
    _, feat_cols, eff_targets, x_df, y_arr = prepare_reg_data(
        samples_data=train_data,
        reg_cfg=reg_cfg,
        feature_cols=feature_cols,
        target_cols=target_cols,
        label="train_data",
        require_target=True,
    )
    if y_arr is None:
        raise rsgispy.RSGISPyException(
            "Prepared target array is unexpectedly empty."
        )
    _, _, eff_y = _resolve_single_target(eff_targets, y_arr, target_var)
    if eff_y is None:
        raise rsgispy.RSGISPyException(
            "Resolved target array is unexpectedly empty."
        )

    method_key = score_func.strip().lower()
    if method_key in {"f_regression", "f", "f_reg"}:
        base_fn: Callable[[Any, Any], Any] = sklearn.feature_selection.f_regression
        has_pvals = True
    elif method_key in {"mutual_info_regression", "mutual_info", "mi"}:
        def _mi_fn(x_vals: Any, y_vals: Any) -> np.ndarray:
            return sklearn.feature_selection.mutual_info_regression(
                x_vals, y_vals, random_state=rnd_seed
            )

        base_fn = _mi_fn
        has_pvals = False
    else:
        raise rsgispy.RSGISPyException(
            f"Unsupported score_func '{score_func}'. Expected 'f_regression' "
            "or 'mutual_info_regression'."
        )

    if eff_y.ndim == 2:
        y_mat = eff_y

        def _multi_target_score(x_vals: Any, _y_col0: Any) -> Any:
            score_list: List[np.ndarray] = []
            pval_list: List[np.ndarray] = []
            for j in range(y_mat.shape[1]):
                res = base_fn(x_vals, y_mat[:, j])
                if has_pvals:
                    sc, pv = res
                    score_list.append(np.asarray(sc, dtype=np.float64))
                    pval_list.append(np.asarray(pv, dtype=np.float64))
                else:
                    score_list.append(np.asarray(res, dtype=np.float64))
            mean_sc = np.mean(np.vstack(score_list), axis=0)
            if has_pvals:
                min_pv = np.min(np.vstack(pval_list), axis=0)
                return mean_sc, min_pv
            return mean_sc

        score_callable: Callable[[Any, Any], Any] = _multi_target_score
        y_1d_pass = y_mat[:, 0]
    else:
        score_callable = base_fn
        y_1d_pass = eff_y

    return run_univariate_selection(
        x_df=x_df,
        y_arr=y_1d_pass,
        feat_cols=feat_cols,
        score_callable=score_callable,
        k=k,
        percentile=percentile,
        out_scores_file=out_scores_file,
        return_scores=return_scores,
    )


# pylint: disable=too-many-arguments,too-many-positional-arguments
def select_features_rfe(
    reg_cfg: Optional[RegConfigInput],
    sk_regressor: Any,
    train_data: SamplesInput,
    target_cols: TargetColsInput = None,
    feature_cols: Optional[Sequence[str]] = None,
    target_var: Optional[str] = None,
    n_features_to_select: Optional[Union[int, float]] = None,
    step: Union[int, float] = 1,
    use_cv: bool = False,
    cv: Union[int, Any] = 5,
    scoring: Optional[Union[str, Any]] = "r2",
    min_features_to_select: int = 1,
    rnd_seed: Optional[int] = None,
    out_scores_file: Optional[Union[str, os.PathLike]] = None,
    out_cv_plot: Optional[Union[str, os.PathLike]] = None,
    n_jobs: int = 1,
    return_scores: bool = False,
) -> Union[List[str], Tuple[List[str], pd.DataFrame]]:
    """
    Select regression features using Recursive Feature Elimination
    (``sklearn.feature_selection.RFE`` or ``RFECV``) with a clean clone of
    ``sk_regressor``.

    :param reg_cfg: Optional regression configuration or image band metadata.
    :param sk_regressor: ``scikit-learn`` regressor exposing
                         ``feature_importances_`` or ``coef_`` after fitting.
    :param train_data: Training sample :class:`pandas.DataFrame` or file path.
    :param target_cols: Optional target column name(s) overriding ``reg_cfg``.
    :param feature_cols: Optional ordered feature column names.
    :param target_var: Optional single target column to isolate for estimators
                       that only support 1D targets.
    :param n_features_to_select: Number (int) or fraction (float in ``(0, 1]``)
                                 of features to retain for ``RFE`` (ignored when
                                 ``use_cv=True``).
    :param step: Number/fraction of features removed per iteration (Default = ``1``).
    :param use_cv: If ``True``, use ``RFECV`` with K-fold CV to determine the
                   optimal feature count automatically (Default = ``False``).
    :param cv: Cross-validation folds or splitter for ``RFECV`` (Default = ``5``).
    :param scoring: Scoring metric for ``RFECV`` (Default = ``'r2'``).
    :param min_features_to_select: Minimum features considered by ``RFECV``
                                   (Default = ``1``).
    :param rnd_seed: Optional random seed passed to cloned estimator and CV splitter.
    :param out_scores_file: Optional CSV path saving per-feature rankings.
    :param out_cv_plot: Optional PNG plot path showing CV score vs number of
                        features (only generated when ``use_cv=True``).
    :param n_jobs: Parallel CV jobs for ``RFECV`` (Default = ``1``).
    :param return_scores: If ``True``, return ``(selected_features, ranking_df)``.
    :return: List of selected feature names (or ``(selected_features, ranking_df)``).

    .. code:: python

        from sklearn.ensemble import RandomForestRegressor
        import rsgispy.regression.regfeatures as rf

        selected = rf.select_features_rfe(
            "reg_config.json",
            RandomForestRegressor(n_estimators=50, random_state=42),
            "train.parquet",
            n_features_to_select=4,
        )
    """
    check_sklearn_available("select_features_rfe")
    _, feat_cols, eff_targets, x_df, y_arr = prepare_reg_data(
        samples_data=train_data,
        reg_cfg=reg_cfg,
        feature_cols=feature_cols,
        target_cols=target_cols,
        label="train_data",
        require_target=True,
    )
    if y_arr is None:
        raise rsgispy.RSGISPyException(
            "Prepared target array is unexpectedly empty."
        )
    _, _, eff_y = _resolve_single_target(eff_targets, y_arr, target_var)
    if eff_y is None:
        raise rsgispy.RSGISPyException(
            "Resolved target array is unexpectedly empty."
        )
    return run_rfe_selection(
        sk_estimator=sk_regressor,
        x_df=x_df,
        y_arr=eff_y,
        feat_cols=feat_cols,
        n_features_to_select=n_features_to_select,
        step=step,
        use_cv=use_cv,
        cv=cv,
        scoring=scoring,
        min_features_to_select=min_features_to_select,
        stratified=False,
        rnd_seed=rnd_seed,
        out_scores_file=out_scores_file,
        out_cv_plot=out_cv_plot,
        n_jobs=n_jobs,
        extra="regression",
        return_scores=return_scores,
    )


# pylint: disable=too-many-arguments,too-many-positional-arguments
def select_features_from_model(
    reg_cfg: Optional[RegConfigInput],
    sk_regressor: Optional[Any],
    train_data: SamplesInput,
    target_cols: TargetColsInput = None,
    feature_cols: Optional[Sequence[str]] = None,
    target_var: Optional[str] = None,
    threshold: Optional[Union[str, float]] = "mean",
    max_features: Optional[int] = None,
    prefit: bool = False,
    use_lassolars: bool = False,
    alpha_val: Optional[float] = None,
    rnd_seed: Optional[int] = None,
    out_scores_file: Optional[Union[str, os.PathLike]] = None,
    return_scores: bool = False,
) -> Union[List[str], Tuple[List[str], pd.DataFrame]]:
    """
    Select regression features based on estimator importance weights via
    ``sklearn.feature_selection.SelectFromModel``, or L1 regularisation via
    :func:`rsgispy.tools.featureselection.lassolars_feature_selection`.

    :param reg_cfg: Optional regression configuration or image band metadata.
    :param sk_regressor: ``scikit-learn`` regressor exposing
                         ``feature_importances_`` or ``coef_`` (may be
                         ``None`` when ``use_lassolars=True``).
    :param train_data: Training sample :class:`pandas.DataFrame` or file path.
    :param target_cols: Optional target column name(s) overriding ``reg_cfg``.
    :param feature_cols: Optional ordered feature column names.
    :param target_var: Optional single target variable name to isolate when
                       training a univariate selector.
    :param threshold: Threshold passed to ``SelectFromModel`` (Default = ``'mean'``).
    :param max_features: Optional cap on retained features in ``SelectFromModel``.
    :param prefit: Set ``True`` if ``sk_regressor`` is already fitted (Default = ``False``).
    :param use_lassolars: If ``True``, use
                          :func:`rsgispy.tools.featureselection.lassolars_feature_selection`
                          instead of ``SelectFromModel`` (Default = ``False``).
    :param alpha_val: Optional ``alpha`` for ``LassoLars`` when ``use_lassolars=True``.
    :param rnd_seed: Optional random seed passed to cloned estimator.
    :param out_scores_file: Optional CSV path saving per-feature selection weights.
    :param return_scores: If ``True``, return ``(selected_features, scores_df)``.
    :return: List of selected feature names (or ``(selected_features, scores_df)``).

    .. code:: python

        from sklearn.ensemble import RandomForestRegressor
        import rsgispy.regression.regfeatures as rf

        selected = rf.select_features_from_model(
            "reg_config.json",
            RandomForestRegressor(n_estimators=50, random_state=42),
            "train.parquet",
        )
    """
    check_sklearn_available("select_features_from_model")
    _, feat_cols, eff_targets, x_df, y_arr = prepare_reg_data(
        samples_data=train_data,
        reg_cfg=reg_cfg,
        feature_cols=feature_cols,
        target_cols=target_cols,
        label="train_data",
        require_target=True,
    )
    if y_arr is None:
        raise rsgispy.RSGISPyException(
            "Prepared target array is unexpectedly empty."
        )
    _, _, eff_y = _resolve_single_target(eff_targets, y_arr, target_var)
    if eff_y is None:
        raise rsgispy.RSGISPyException(
            "Resolved target array is unexpectedly empty."
        )
    return run_from_model_selection(
        sk_estimator=sk_regressor,
        x_df=x_df,
        y_arr=eff_y,
        feat_cols=feat_cols,
        threshold=threshold,
        max_features=max_features,
        prefit=prefit,
        use_lassolars=use_lassolars,
        alpha_val=alpha_val,
        rnd_seed=rnd_seed,
        out_scores_file=out_scores_file,
        return_scores=return_scores,
        extra="regression",
        role_label="sk_regressor",
    )


# pylint: disable=too-many-arguments,too-many-positional-arguments
def select_features_boruta_shap(
    reg_cfg: Optional[RegConfigInput],
    train_data: SamplesInput,
    target_cols: TargetColsInput = None,
    sk_regressor: Optional[Any] = None,
    feature_cols: Optional[Sequence[str]] = None,
    target_var: Optional[str] = None,
    n_trials: int = 50,
    pvalue: float = 0.05,
    percentile: float = 100.0,
    bonferroni: bool = True,
    include_tentative: bool = False,
    tentative_rough_fix: bool = False,
    subsample_n_smpls: Optional[int] = None,
    use_tree_explainer: Optional[bool] = None,
    use_linear_explainer: bool = False,
    rnd_seed: Optional[int] = None,
    out_scores_file: Optional[Union[str, os.PathLike]] = None,
    out_plot_file: Optional[Union[str, os.PathLike]] = None,
    use_pbar: bool = True,
    return_scores: bool = False,
    *,
    linear_perturbation: str = "interventional",
) -> Union[List[str], Tuple[List[str], pd.DataFrame]]:
    """
    Select regression features using **BorutaShap**, combining the Boruta
    all-relevant shadow-feature hypothesis test with Shapley (SHAP) feature
    attributions.

    Supports both univariate and multivariate targets (defaults to
    ``RandomForestRegressor(n_estimators=100)``). Requires optional
    dependencies ``shap`` and ``scipy``.

    :param reg_cfg: Optional regression configuration or image band metadata.
    :param train_data: Training sample :class:`pandas.DataFrame` or file path.
    :param target_cols: Optional target column name(s) overriding ``reg_cfg``.
    :param sk_regressor: Optional ``scikit-learn`` regressor template
                         (defaults to ``RandomForestRegressor(n_estimators=100)``).
    :param feature_cols: Optional ordered feature column names (at least 2).
    :param target_var: Optional single target column name when isolating one
                       variable from a multivariate target table.
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
                                shadow threshold to ``'Confirmed'`` after
                                ``n_trials`` (Default = ``False``).
    :param subsample_n_smpls: Optional sample cap for SHAP evaluation inside each
                              trial.
    :param use_tree_explainer: Explicitly force ``shap.TreeExplainer``
                               (``True``), disable it (``False``), or
                               auto-detect based on the estimator (``None``,
                               default).
    :param use_linear_explainer: Use ``shap.LinearExplainer`` when ``True``
                                 (Default = ``False``).
    :param linear_perturbation: How ``shap.LinearExplainer`` handles
                                correlated variables (keyword only):
                                ``'interventional'`` (Default; the ``shap``
                                default, each variable's SHAP value is
                                ``coef * (x - mean(x))`` so a variable with a
                                coefficient of ~0 gets ~0 SHAP values) or
                                ``'correlation_dependent'`` (effects are shared
                                between correlated variables, so a variable with
                                a coefficient of ~0 can get large SHAP values).
                                Only used when a linear explainer is chosen.
    :param rnd_seed: Optional random seed for reproducible shadow shuffling,
                     estimator training, and subsampling.
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

        from sklearn.ensemble import RandomForestRegressor
        import rsgispy.regression.regfeatures as rf

        selected, summary_df = rf.select_features_boruta_shap(
            "reg_config.json",
            "train.parquet",
            sk_regressor=RandomForestRegressor(n_estimators=100),
            n_trials=40,
            rnd_seed=42,
            out_scores_file="boruta_shap.csv",
            out_plot_file="boruta_shap.png",
            return_scores=True,
        )
    """
    check_sklearn_available("select_features_boruta_shap")
    _, feat_cols, eff_targets, x_df, y_arr = prepare_reg_data(
        samples_data=train_data,
        reg_cfg=reg_cfg,
        feature_cols=feature_cols,
        target_cols=target_cols,
        label="train_data",
        require_target=True,
    )
    if y_arr is None:
        raise rsgispy.RSGISPyException(
            "Prepared target array is unexpectedly empty."
        )
    _, _, eff_y = _resolve_single_target(eff_targets, y_arr, target_var)
    if eff_y is None:
        raise rsgispy.RSGISPyException(
            "Resolved target array is unexpectedly empty."
        )

    selected, scores_df = run_boruta_shap_selection(
        train_x=x_df,
        train_y=eff_y,
        feature_names=feat_cols,
        feature_name="select_features_boruta_shap",
        extra="regression",
        sk_classifier=sk_regressor,
        n_trials=n_trials,
        pvalue=pvalue,
        percentile=percentile,
        bonferroni=bonferroni,
        include_tentative=include_tentative,
        tentative_rough_fix=tentative_rough_fix,
        target_class_idx=None,
        subsample_n_smpls=subsample_n_smpls,
        use_tree_explainer=use_tree_explainer,
        use_linear_explainer=use_linear_explainer,
        linear_perturbation=linear_perturbation,
        rnd_seed=rnd_seed,
        out_plot_file=out_plot_file,
        use_pbar=use_pbar,
        task="regression",
    )

    if out_scores_file is not None:
        scores_df.to_csv(ensure_parent_dir(out_scores_file), index=False)
    if return_scores:
        return selected, scores_df
    return selected


# ---------------------------------------------------------------------------
# Feature Importance, Response Curves, SHAP & Sensitivity Analysis
# ---------------------------------------------------------------------------


# pylint: disable=too-many-arguments,too-many-positional-arguments
def calc_feature_importance(
    reg_cfg: Optional[RegConfigInput],
    sk_regressor: Any,
    eval_data: SamplesInput,
    target_cols: TargetColsInput = None,
    feature_cols: Optional[Sequence[str]] = None,
    target_var: Optional[str] = None,
    method: str = "permutation",
    scoring: Optional[Union[str, Any]] = "r2",
    n_repeats: int = 10,
    rnd_seed: Optional[int] = None,
    n_jobs: int = 1,
    out_importance_file: Optional[Union[str, os.PathLike]] = None,
    out_plot_file: Optional[Union[str, os.PathLike]] = None,
) -> pd.DataFrame:
    """
    Compute feature importances for a fitted univariate or multivariate
    regressor using permutation importance
    (``sklearn.inspection.permutation_importance``), native estimator weights
    (``feature_importances_`` / ``coef_``), or both.

    :param reg_cfg: Optional regression configuration or image band metadata.
    :param sk_regressor: Fitted ``scikit-learn`` or GBDT regressor.
    :param eval_data: Evaluation sample :class:`pandas.DataFrame` or file path
                      (typically test or validation split for permutation).
    :param target_cols: Optional target column name(s) overriding ``reg_cfg``.
    :param feature_cols: Optional ordered feature column names.
    :param target_var: Optional single target variable name when isolating
                       permutation importance scoring on one multivariate output.
    :param method: Importance calculation method: ``'permutation'`` (default),
                   ``'model'`` (native ``feature_importances_`` / ``coef_``), or
                   ``'both'``.
    :param scoring: Scoring metric passed to ``permutation_importance``
                    (Default = ``'r2'``).
    :param n_repeats: Number of permutation repeats per feature (Default = ``10``).
    :param rnd_seed: Optional random seed for ``permutation_importance``.
    :param n_jobs: Parallel jobs for ``permutation_importance`` (Default = ``1``).
    :param out_importance_file: Optional CSV output path for the importance table.
    :param out_plot_file: Optional PNG horizontal bar chart output path.
    :return: Sorted :class:`pandas.DataFrame` of feature importances and ranks.

    .. code:: python

        import rsgispy.regression.regfeatures as rf

        imp_df = rf.calc_feature_importance(
            "reg_config.json", reg, "test.parquet", method="both",
            out_importance_file="feat_imp.csv", out_plot_file="feat_imp.png",
        )
    """
    # pylint: disable=too-many-locals
    check_sklearn_available("calc_feature_importance")
    check_sklearn_estimator_fitted(sk_regressor, label="sk_regressor")
    _, feat_cols, eff_targets, x_df, y_arr = prepare_reg_data(
        samples_data=eval_data,
        reg_cfg=reg_cfg,
        feature_cols=feature_cols,
        target_cols=target_cols,
        label="eval_data",
        require_target=True,
    )
    if y_arr is None:
        raise rsgispy.RSGISPyException(
            "Prepared target array is unexpectedly empty."
        )

    eff_scoring = scoring
    eff_y = y_arr
    if target_var is not None:
        if target_var not in eff_targets:
            raise rsgispy.RSGISPyException(
                f"target_var '{target_var}' not found in targets {eff_targets}."
            )
        t_idx = eff_targets.index(target_var)
        if y_arr.ndim == 2:
            base_scorer = (
                sklearn.metrics.get_scorer(scoring)
                if isinstance(scoring, str)
                else scoring
            )

            class _TargetColSliceEst(
                sklearn.base.BaseEstimator, sklearn.base.RegressorMixin
            ):
                """Adapter exposing 1D predictions for a single target column."""

                def __init__(self, inner: Any = None, col_i: int = 0) -> None:
                    self.inner = inner
                    self.col_i = col_i

                def fit(
                    self, x_in: Any = None, y_in: Any = None
                ) -> "_TargetColSliceEst":
                    """No-op fit for sklearn estimator compatibility."""
                    del x_in, y_in
                    return self

                def __sklearn_is_fitted__(self) -> bool:
                    return True

                def predict(self, x_in: Any) -> np.ndarray:
                    """Return 1D predictions for target column ``col_i``."""
                    preds = np.asarray(
                        self.inner.predict(x_in), dtype=np.float64
                    )
                    return preds[:, self.col_i] if preds.ndim == 2 else preds

            def _target_scorer(est_obj: Any, x_in: Any, y_1d: Any) -> float:
                return float(
                    base_scorer(_TargetColSliceEst(est_obj, t_idx), x_in, y_1d)
                )

            eff_y = y_arr[:, t_idx]
            eff_scoring = _target_scorer

    return run_feature_importance(
        sk_estimator=sk_regressor,
        x_df=x_df,
        y_arr=eff_y,
        feat_cols=feat_cols,
        method=method,
        scoring=eff_scoring,
        n_repeats=n_repeats,
        rnd_seed=rnd_seed,
        n_jobs=n_jobs,
        out_importance_file=out_importance_file,
        out_plot_file=out_plot_file,
        plot_title="Regression Feature Importance",
        missing_native_msg="Regressor does not expose 'feature_importances_' or 'coef_'.",
    )


# pylint: disable=too-many-arguments,too-many-positional-arguments
def calc_jackknife_feature_importance(
    reg_cfg: Optional[RegConfigInput],
    sk_regressor: Any,
    train_data: SamplesInput,
    test_data: SamplesInput,
    target_cols: TargetColsInput = None,
    feature_cols: Optional[Sequence[str]] = None,
    target_var: Optional[str] = None,
    rnd_seed: Optional[int] = None,
    out_var_importance_file: Optional[Union[str, os.PathLike]] = None,
    out_importance_r2_plot: Optional[Union[str, os.PathLike]] = None,
    out_importance_rmse_plot: Optional[Union[str, os.PathLike]] = None,
    use_pbar: bool = True,
) -> pd.DataFrame:
    """
    Evaluate regression feature importance via jackknife leave-one-out and
    single-feature training runs using clean cloned estimators in each iteration.

    Computes training and testing ``R^2``, ``RMSE``, and ``MAE`` across every
    leave-one-out and solo-feature subset alongside the full-feature baseline.

    :param reg_cfg: Optional regression configuration or image band metadata.
    :param sk_regressor: Base ``scikit-learn`` regressor template.
    :param train_data: Training sample :class:`pandas.DataFrame` or file path.
    :param test_data: Testing sample :class:`pandas.DataFrame` or file path.
    :param target_cols: Optional target column name(s) overriding ``reg_cfg``.
    :param feature_cols: Optional ordered feature column names (at least 2).
    :param target_var: Optional single target variable name to isolate from
                       multivariate targets.
    :param rnd_seed: Optional random seed passed to cloned estimators.
    :param out_var_importance_file: Optional CSV output path for jackknife metrics.
    :param out_importance_r2_plot: Optional horizontal bar plot path for test ``R^2``.
    :param out_importance_rmse_plot: Optional horizontal bar plot path for test ``RMSE``.
    :param use_pbar: Display a tqdm progress bar (Default = ``True``).
    :return: :class:`pandas.DataFrame` indexed by ``feature_cols``.

    .. code:: python

        from sklearn.ensemble import RandomForestRegressor
        import rsgispy.regression.regfeatures as rf

        jack_df = rf.calc_jackknife_feature_importance(
            "reg_config.json",
            RandomForestRegressor(n_estimators=50, random_state=42),
            "train.parquet",
            "test.parquet",
            out_var_importance_file="jackknife.csv",
            out_importance_r2_plot="jack_r2.png",
        )
    """
    # pylint: disable=too-many-locals
    check_sklearn_available("calc_jackknife_feature_importance")
    _, feat_cols, eff_targets, train_x_all, train_y = prepare_reg_data(
        samples_data=train_data,
        reg_cfg=reg_cfg,
        feature_cols=feature_cols,
        target_cols=target_cols,
        label="train_data",
        require_target=True,
    )
    if len(feat_cols) < 2:
        raise rsgispy.RSGISPyException(
            "Jackknife feature importance requires at least 2 features."
        )
    _, _, _, test_x_all, test_y = prepare_reg_data(
        samples_data=test_data,
        reg_cfg=reg_cfg,
        feature_cols=feat_cols,
        target_cols=eff_targets,
        label="test_data",
        require_target=True,
    )
    if train_y is None or test_y is None:
        raise rsgispy.RSGISPyException(
            "Prepared training or testing target arrays are unexpectedly empty."
        )
    _, run_targets, eff_train_y = _resolve_single_target(
        eff_targets, train_y, target_var
    )
    _, _, eff_test_y = _resolve_single_target(eff_targets, test_y, target_var)
    if eff_train_y is None or eff_test_y is None:
        raise rsgispy.RSGISPyException(
            "Resolved training or testing target arrays are unexpectedly empty."
        )

    def _eval_subset(sub_vars: List[str]) -> Dict[str, float]:
        tr_sub = train_x_all[sub_vars]
        te_sub = test_x_all[sub_vars]
        est_sub = fit_clean_sklearn_estimator(
            sk_regressor,
            tr_sub,
            eff_train_y,
            sync_input=False,
            rnd_seed=rnd_seed,
            extra="regression",
        )
        res = calc_reg_split_metrics(
            est_sub, tr_sub, eff_train_y, run_targets, "train"
        )
        res.update(
            calc_reg_split_metrics(
                est_sub, te_sub, eff_test_y, run_targets, "test"
            )
        )
        return res

    all_metrics = _eval_subset(feat_cols)

    out_data: Dict[str, List[float]] = {
        "train_r2_leave_var_out": [],
        "test_r2_leave_var_out": [],
        "test_rmse_leave_var_out": [],
        "test_mae_leave_var_out": [],
        "train_r2_only_var": [],
        "test_r2_only_var": [],
        "test_rmse_only_var": [],
        "test_mae_only_var": [],
        "all_vars_test_r2": [],
        "all_vars_test_rmse": [],
    }

    pbar = rsgispy.get_tqdm()
    for feat in pbar(feat_cols, disable=not use_pbar):
        m_out = _eval_subset([f for f in feat_cols if f != feat])
        out_data["train_r2_leave_var_out"].append(m_out["train_r2"])
        out_data["test_r2_leave_var_out"].append(m_out["test_r2"])
        out_data["test_rmse_leave_var_out"].append(m_out["test_rmse"])
        out_data["test_mae_leave_var_out"].append(m_out["test_mae"])

    for feat in pbar(feat_cols, disable=not use_pbar):
        m_only = _eval_subset([feat])
        out_data["train_r2_only_var"].append(m_only["train_r2"])
        out_data["test_r2_only_var"].append(m_only["test_r2"])
        out_data["test_rmse_only_var"].append(m_only["test_rmse"])
        out_data["test_mae_only_var"].append(m_only["test_mae"])
        out_data["all_vars_test_r2"].append(all_metrics["test_r2"])
        out_data["all_vars_test_rmse"].append(all_metrics["test_rmse"])

    jack_df = pd.DataFrame(data=out_data, index=feat_cols)
    if out_var_importance_file is not None:
        jack_df.to_csv(ensure_parent_dir(out_var_importance_file), index=True)

    if out_importance_r2_plot is not None:
        fig, _ = plot_jackknife_bars(
            feat_cols,
            out_data["test_r2_leave_var_out"],
            out_data["test_r2_only_var"],
            all_metrics["test_r2"],
            "Jackknife R^2 Comparison",
            "Testing R^2",
        )
        finalise_plot(fig, out_file=out_importance_r2_plot)

    if out_importance_rmse_plot is not None:
        fig, _ = plot_jackknife_bars(
            feat_cols,
            out_data["test_rmse_leave_var_out"],
            out_data["test_rmse_only_var"],
            all_metrics["test_rmse"],
            "Jackknife RMSE Comparison",
            "Testing RMSE",
        )
        finalise_plot(fig, out_file=out_importance_rmse_plot)

    return jack_df


# pylint: disable=too-many-arguments,too-many-positional-arguments
def calc_feature_response_curves(
    reg_cfg: Optional[RegConfigInput],
    sk_regressor: Any,
    train_data: SamplesInput,
    target_cols: TargetColsInput = None,
    feature_cols: Optional[Sequence[str]] = None,
    target_vars: TargetColsInput = None,
    out_file: Optional[Union[str, os.PathLike]] = None,
    response_plots_dir: Optional[Union[str, os.PathLike]] = None,
    out_grid_plot: Optional[Union[str, os.PathLike]] = None,
    n_samples: int = 100,
    summary_stat: str = "mean",
    method: str = "marginal",
    pdp_background_samples: int = 200,
    rnd_seed: Optional[int] = None,
    ylim: Optional[Tuple[float, float]] = None,
    multi_panel: bool = False,
    plot_samples: bool = False,
    use_pbar: bool = True,
) -> pd.DataFrame:
    """
    Generate marginal or partial dependence (PDP) response function curves
    showing how predicted regression target variable(s) vary across each
    input feature's empirical range.

    Supports both **univariate** and **multivariate** regression models,
    optional observed sample scatter overlays (``plot_samples=True``), stacked
    per-target subplots (``multi_panel=True``), per-feature PNG figures
    (``response_plots_dir``), and a combined summary figure matrix
    (``out_grid_plot``).

    :param reg_cfg: Optional regression configuration or image band metadata.
    :param sk_regressor: Fitted ``scikit-learn`` or GBDT regressor.
    :param train_data: Reference sample :class:`pandas.DataFrame` or file path
                       defining feature ranges and baseline constants.
    :param target_cols: Optional target column name(s) overriding ``reg_cfg``.
    :param feature_cols: Optional ordered feature column names.
    :param target_vars: Optional subset of target variable name(s) to evaluate
                        and plot (when ``None``, evaluates all model outputs).
    :param out_file: Optional output CSV path storing grid values and
                     predicted response curves.
    :param response_plots_dir: Optional directory saving
                               ``<feature>_response_curve.png`` figures.
    :param out_grid_plot: Optional output PNG path saving a multi-feature grid
                          summary plot.
    :param n_samples: Number of evaluation steps across each feature range
                      (Default = ``100``).
    :param summary_stat: Baseline value used for held-constant features when
                         ``method='marginal'``: ``'mean'`` (default) or
                         ``'median'``.
    :param method: Response computation method: ``'marginal'`` (hold other
                   features at sample mean/median, default) or ``'pdp'``
                   (Partial Dependence averaged over background samples).
    :param pdp_background_samples: Maximum number of background rows sampled
                                   when ``method='pdp'`` (Default = ``200``).
    :param rnd_seed: Optional random seed for background subsampling when
                     ``method='pdp'``.
    :param ylim: Optional ``(ymin, ymax)`` y-axis limits for single-panel plots.
    :param multi_panel: If ``True`` and multiple targets are plotted, draw
                        stacked per-target subplots rather than overlaid lines
                        (Default = ``False``).
    :param plot_samples: If ``True``, overlay observed sample scatter points
                         behind response curves (Default = ``False``).
    :param use_pbar: Display a tqdm progress bar (Default = ``True``).
    :return: :class:`pandas.DataFrame` containing ``<feat>_vals`` and
             ``<feat>_<target>_pred`` columns.

    .. code:: python

        import rsgispy.regression.regfeatures as rf

        resp_df = rf.calc_feature_response_curves(
            "reg_config.json",
            reg_model,
            "train.parquet",
            method="pdp",
            plot_samples=True,
            response_plots_dir="response_plots",
            out_grid_plot="response_matrix.png",
        )
    """
    # pylint: disable=too-many-locals
    check_sklearn_available("calc_feature_response_curves")
    check_sklearn_estimator_fitted(sk_regressor, label="sk_regressor")

    df, feat_cols, eff_targets, x_df, _ = prepare_reg_data(
        samples_data=train_data,
        reg_cfg=reg_cfg,
        feature_cols=feature_cols,
        target_cols=target_cols,
        label="train_data",
        require_target=False,
    )
    stat_key = summary_stat.strip().lower()
    if stat_key not in {"mean", "median"}:
        raise rsgispy.RSGISPyException(
            f"Unsupported summary_stat '{summary_stat}'. Expected 'mean' or 'median'."
        )
    model_cols = _resolve_model_feature_cols(sk_regressor, df, feat_cols)
    extra_cols = [c for c in model_cols if c not in feat_cols]
    extra_base = {
        c: float(df[c].mean() if stat_key == "mean" else df[c].median())
        for c in extra_cols
    }
    n_outputs, output_labels = _resolve_estimator_target_labels(
        sk_regressor, df[model_cols], eff_targets
    )

    if target_vars is not None:
        req_targets = normalise_target_cols(target_vars)
        out_indices: List[int] = []
        for t_name in req_targets:
            if t_name not in output_labels:
                raise rsgispy.RSGISPyException(
                    f"target_vars entry '{t_name}' not found in model target "
                    f"labels {output_labels}."
                )
            out_indices.append(output_labels.index(t_name))
    else:
        out_indices = list(range(n_outputs))

    obs_y_df: Optional[pd.DataFrame] = None
    if plot_samples:
        present_obs = [lbl for lbl in output_labels if lbl in df.columns]
        if present_obs:
            obs_y_df = df[present_obs]

    def _predict_2d(in_df: pd.DataFrame) -> np.ndarray:
        if model_cols == list(in_df.columns):
            eval_df = in_df
        else:
            eval_data: Dict[str, Any] = {}
            n_rows = len(in_df)
            for col in model_cols:
                if col in in_df.columns:
                    eval_data[col] = in_df[col].to_numpy()
                else:
                    eval_data[col] = np.full(
                        n_rows, extra_base[col], dtype=float
                    )
            eval_df = pd.DataFrame(eval_data)
        preds = np.asarray(sk_regressor.predict(eval_df), dtype=np.float64)
        return preds[:, None] if preds.ndim == 1 else preds

    single_ylabel = (
        output_labels[out_indices[0]]
        if len(out_indices) == 1
        else "Predicted Value"
    )
    return run_variable_response_curves(
        x_df=x_df,
        feature_cols=feat_cols,
        predict_fn=_predict_2d,
        output_labels=output_labels,
        out_indices=out_indices,
        col_name_fn=lambda feat, lbl, _i: f"{feat}_{lbl}_pred",
        n_samples=n_samples,
        summary_stat=summary_stat,
        method=method,
        pdp_background_samples=pdp_background_samples,
        rnd_seed=rnd_seed,
        out_file=out_file,
        response_plots_dir=response_plots_dir,
        out_grid_plot=out_grid_plot,
        ylabel=single_ylabel,
        ylim=ylim,
        multi_panel=multi_panel,
        observed_y_df=obs_y_df,
        plot_samples=plot_samples,
        use_pbar=use_pbar,
    )


# pylint: disable=too-many-arguments,too-many-positional-arguments
def calc_shap_feature_contributions(
    reg_cfg: Optional[RegConfigInput],
    sk_regressor: Any,
    samples_data: SamplesInput,
    target_cols: TargetColsInput = None,
    feature_cols: Optional[Sequence[str]] = None,
    target_var: Optional[str] = None,
    shap_summary_plot: Optional[Union[str, os.PathLike]] = None,
    shap_heatmap_plot: Optional[Union[str, os.PathLike]] = None,
    shap_scatter_plots_dir: Optional[Union[str, os.PathLike]] = None,
    shap_depend_plots_dir: Optional[Union[str, os.PathLike]] = None,
    out_shap_file: Optional[Union[str, os.PathLike]] = None,
    subsample_n_smpls: Optional[int] = None,
    use_tree_explainer: Optional[bool] = None,
    use_linear_explainer: bool = False,
    rnd_seed: Optional[int] = None,
    *,
    linear_perturbation: str = "interventional",
) -> pd.DataFrame:
    """
    Compute SHAP feature contributions for a fitted univariate or multivariate
    regressor and generate summary, heatmap, scatter, and partial dependence
    plots.

    Requires optional dependency ``shap``.

    :param reg_cfg: Optional regression configuration or image band metadata.
    :param sk_regressor: Fitted ``scikit-learn`` or GBDT regressor.
    :param samples_data: Sample :class:`pandas.DataFrame` or file path.
    :param target_cols: Optional target column name(s) overriding ``reg_cfg``.
    :param feature_cols: Optional ordered feature column names.
    :param target_var: Optional target variable name used to isolate one
                       output in multivariate SHAP plots (when ``None``,
                       multivariate summary plots show all outputs).
    :param shap_summary_plot: Optional output path for SHAP summary plot.
    :param shap_heatmap_plot: Optional output path for SHAP heatmap plot.
    :param shap_scatter_plots_dir: Optional directory for per-feature SHAP
                                   scatter plots.
    :param shap_depend_plots_dir: Optional directory for per-feature partial
                                  dependence plots.
    :param out_shap_file: Optional output CSV path saving mean ``|SHAP|`` values.
    :param subsample_n_smpls: Optional subsample count for SHAP evaluation.
    :param use_tree_explainer: Explicitly force ``shap.TreeExplainer``
                               (``True``), disable it (``False``), or
                               auto-detect (``None``, default).
    :param use_linear_explainer: Use ``shap.LinearExplainer`` when ``True``
                                 (Default = ``False``).
    :param linear_perturbation: How ``shap.LinearExplainer`` handles
                                correlated variables (keyword only):
                                ``'interventional'`` (Default; the ``shap``
                                default, each variable's SHAP value is
                                ``coef * (x - mean(x))`` so a variable with a
                                coefficient of ~0 gets ~0 SHAP values) or
                                ``'correlation_dependent'`` (effects are shared
                                between correlated variables, so a variable with
                                a coefficient of ~0 can get large SHAP values).
                                Only used when a linear explainer is chosen.
    :param rnd_seed: Optional random seed for subsampling.
    :return: :class:`pandas.DataFrame` of mean absolute SHAP contributions per
             feature (and per target when multivariate).

    .. code:: python

        import rsgispy.regression.regfeatures as rf

        shap_df = rf.calc_shap_feature_contributions(
            "reg_config.json", reg_model, "train.parquet",
            shap_summary_plot="shap_summary.png",
        )
    """
    # pylint: disable=too-many-locals
    check_sklearn_available("calc_shap_feature_contributions")
    check_sklearn_estimator_fitted(sk_regressor, label="sk_regressor")

    _, feat_cols, eff_targets, x_df, _ = prepare_reg_data(
        samples_data=samples_data,
        reg_cfg=reg_cfg,
        feature_cols=feature_cols,
        target_cols=target_cols,
        label="samples_data",
        require_target=False,
    )
    _, output_labels = _resolve_estimator_target_labels(
        sk_regressor, x_df, eff_targets
    )
    target_idx: Optional[int] = None
    if target_var is not None:
        if target_var not in output_labels:
            raise rsgispy.RSGISPyException(
                f"target_var '{target_var}' not found in targets "
                f"{output_labels}."
            )
        target_idx = output_labels.index(target_var)

    summary_df = run_shap_explainer(
        est_cls_obj=sk_regressor,
        train_x=x_df,
        analysis_vars=feat_cols,
        feature_name="calc_shap_feature_contributions",
        extra="regression",
        shap_summary_plot=shap_summary_plot,
        shap_heatmap_plot=shap_heatmap_plot,
        shap_scatter_plots_dir=shap_scatter_plots_dir,
        shap_depend_plots_dir=shap_depend_plots_dir,
        subsample_n_smpls=subsample_n_smpls,
        use_tree_explainer=use_tree_explainer,
        use_linear_explainer=use_linear_explainer,
        linear_perturbation=linear_perturbation,
        target_class_idx=target_idx,
        class_labels=output_labels if len(output_labels) > 1 else None,
        rnd_seed=rnd_seed,
        task="regression",
    )
    if out_shap_file is not None:
        summary_df.to_csv(ensure_parent_dir(out_shap_file), index=False)
    return summary_df


# pylint: disable=too-many-arguments,too-many-positional-arguments
def calc_salib_feature_sensitivity(
    reg_cfg: Optional[RegConfigInput],
    sk_regressor: Any,
    samples_data: SamplesInput,
    target_cols: TargetColsInput = None,
    feature_cols: Optional[Sequence[str]] = None,
    target_var: Optional[str] = None,
    n_samples: int = 256,
    sobol_plot_file: Optional[Union[str, os.PathLike]] = None,
    sobol_overall_file: Optional[Union[str, os.PathLike]] = None,
    sobol_first_file: Optional[Union[str, os.PathLike]] = None,
    sobol_second_file: Optional[Union[str, os.PathLike]] = None,
    rnd_seed: Optional[int] = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform SALib Sobol sensitivity analysis quantifying first-order,
    second-order, and total-order feature sensitivity for continuous predictions
    of ``sk_regressor``.

    Requires optional dependency ``SALib``.

    :param reg_cfg: Optional regression configuration or image band metadata.
    :param sk_regressor: Fitted ``scikit-learn`` or GBDT regressor.
    :param samples_data: Sample :class:`pandas.DataFrame` or file path defining
                         empirical feature bounds.
    :param target_cols: Optional target column name(s) overriding ``reg_cfg``.
    :param feature_cols: Optional ordered feature column names.
    :param target_var: Optional target variable name analysed for multivariate
                       regressors (defaults to the first output target).
    :param n_samples: Base Saltelli sample count (Default = ``256``).
    :param sobol_plot_file: Optional output Sobol sensitivity bar plot PNG path.
    :param sobol_overall_file: Optional total-order Sobol CSV output path.
    :param sobol_first_file: Optional first-order Sobol CSV output path.
    :param sobol_second_file: Optional second-order Sobol CSV output path.
    :param rnd_seed: Seed for the Sobol sample and the bootstrap confidence
                     intervals, so the indices are the same on every run
                     (Default = ``42``; ``None`` for a different sample each run).
    :return: Tuple ``(total_Si, first_Si, second_Si)`` DataFrames.

    .. code:: python

        import rsgispy.regression.regfeatures as rf

        total_si, first_si, second_si = rf.calc_salib_feature_sensitivity(
            "reg_config.json", reg_model, "train.parquet", n_samples=128
        )
    """
    check_sklearn_available("calc_salib_feature_sensitivity")
    check_sklearn_estimator_fitted(sk_regressor, label="sk_regressor")

    _, feat_cols, eff_targets, x_df, _ = prepare_reg_data(
        samples_data=samples_data,
        reg_cfg=reg_cfg,
        feature_cols=feature_cols,
        target_cols=target_cols,
        label="samples_data",
        require_target=False,
    )
    _, output_labels = _resolve_estimator_target_labels(
        sk_regressor, x_df, eff_targets
    )
    target_idx = 0
    if target_var is not None:
        if target_var not in output_labels:
            raise rsgispy.RSGISPyException(
                f"target_var '{target_var}' not found in targets "
                f"{output_labels}."
            )
        target_idx = output_labels.index(target_var)

    return run_salib_sobol_sensitivity(
        train_df=x_df,
        analysis_vars=feat_cols,
        predict_1d_fn=lambda grid_df: _predict_1d_target(
            sk_regressor, grid_df, target_idx
        ),
        feature_name="calc_salib_feature_sensitivity",
        extra="regression",
        n_samples=n_samples,
        sobol_plot_file=sobol_plot_file,
        sobol_overall_file=sobol_overall_file,
        sobol_first_file=sobol_first_file,
        sobol_second_file=sobol_second_file,
        rnd_seed=rnd_seed,
    )
