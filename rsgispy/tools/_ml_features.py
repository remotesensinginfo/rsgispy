"""
Shared internal feature diagnostics, selection, and importance helpers reused
across :mod:`rsgispy.classification` and :mod:`rsgispy.regression`.
"""

import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import warnings

import numpy as np
import pandas as pd

import rsgispy
from rsgispy.tools._ml_explain import (
    extract_native_feature_importances,
    finalise_plot,
    plot_correlation_heatmap,
    plot_feature_importance_bars,
)
from rsgispy.tools._ml_tune import resolve_cv_splitter
from rsgispy.tools._sklearn import (
    check_sklearn_estimator_fitted,
    clone_sklearn_estimator,
    fit_clean_sklearn_estimator,
    sklearn,
)
from rsgispy.tools.featureselection import (
    calc_pandas_vif,
    corr_feature_selection,
    lassolars_feature_selection,
)
from rsgispy.tools.files import ensure_parent_dir
from rsgispy.tools.plotting import CATEGORICAL_COLOURS, get_fig_ax, use_plot_style

_VALID_CORR_METHODS = {"pearson", "spearman", "kendall"}


@use_plot_style
def _plot_rfecv_scores(
    n_feats: np.ndarray,
    mean_scores: np.ndarray,
    n_optimal: int,
    ax: Optional[Any] = None,
) -> Tuple[Any, Any]:
    """Plot the RFECV cross-validation score against the number of features."""
    fig, ax = get_fig_ax(ax, figsize=(7, 4))
    ax.plot(n_feats, mean_scores, marker="o", color=CATEGORICAL_COLOURS[0])
    ax.axvline(
        x=n_optimal,
        color=CATEGORICAL_COLOURS[1],
        linestyle="--",
        label=f"Optimal = {n_optimal}",
    )
    ax.set_xlabel("Number of selected features")
    ax.set_ylabel("CV Score")
    ax.set_title("RFECV feature selection")
    ax.legend()
    return fig, ax


def run_correlation_matrix(
    df: pd.DataFrame,
    cols: Sequence[str],
    method: str = "pearson",
    out_corr_file: Optional[Union[str, os.PathLike]] = None,
    out_plt_file: Optional[Union[str, os.PathLike]] = None,
    fig_width: int = 15,
    fig_height: int = 14,
) -> pd.DataFrame:
    """Compute pairwise correlations and optionally save CSV and heatmap PNG."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    corr_method = method.strip().lower()
    if corr_method not in _VALID_CORR_METHODS:
        raise rsgispy.RSGISPyException(
            f"Unsupported correlation method '{method}'. Expected one of "
            f"{sorted(_VALID_CORR_METHODS)}."
        )
    col_list = list(cols)
    corr_df = df[col_list].corr(method=corr_method)  # type: ignore[arg-type]
    if out_corr_file is not None:
        corr_df.to_csv(ensure_parent_dir(out_corr_file), index=True)
    if out_plt_file is not None:
        fig, _ = plot_correlation_heatmap(
            corr_df=corr_df,
            var_names=col_list,
            figsize=(fig_width, fig_height),
        )
        finalise_plot(fig, out_file=out_plt_file)
    return corr_df


def run_iterative_vif_selection(
    x_df: pd.DataFrame,
    feat_cols: Sequence[str],
    vif_threshold: float = 10.0,
    min_features: int = 2,
    out_vif_file: Optional[Union[str, os.PathLike]] = None,
    return_vif: bool = False,
) -> Union[List[str], Tuple[List[str], pd.Series]]:
    """Iteratively prune highest-VIF feature until threshold or floor reached."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    if vif_threshold <= 0.0:
        raise rsgispy.RSGISPyException("vif_threshold must be > 0.")
    if min_features < 2:
        raise rsgispy.RSGISPyException("min_features must be at least 2.")
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


def run_corr_cluster_selection(
    x_df: pd.DataFrame,
    y_arr: np.ndarray,
    feat_cols: Sequence[str],
    target_names: Sequence[str],
    n_min_clusters: int = 3,
    n_max_clusters: int = 25,
    use_pbar: bool = True,
) -> List[str]:
    """Select representative predictors via correlation hierarchical clustering."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    feat_list = list(feat_cols)
    if len(feat_list) < 3:
        raise rsgispy.RSGISPyException(
            "select_features_corr requires at least 3 features."
        )

    eff_max = min(int(n_max_clusters), len(feat_list))
    eff_min = max(2, min(int(n_min_clusters), eff_max - 1))
    if eff_min >= eff_max:
        raise rsgispy.RSGISPyException(
            f"Invalid cluster range [{eff_min}, {eff_max}) for "
            f"{len(feat_list)} features."
        )

    work_df = x_df[feat_list].copy()
    y_2d = np.asarray(y_arr, dtype=np.float64)
    if y_2d.ndim == 1:
        y_2d = y_2d[:, None]
    dep_cols: List[str] = []
    for idx in range(y_2d.shape[1]):
        if np.std(y_2d[:, idx]) == 0.0:
            t_lbl = target_names[idx] if idx < len(target_names) else f"target_{idx}"
            raise rsgispy.RSGISPyException(
                f"Target column '{t_lbl}' has zero variance; cannot compute "
                "correlation-based feature selection."
            )
        col_key = f"__rsgis_target_{idx}__"
        work_df[col_key] = y_2d[:, idx]
        dep_cols.append(col_key)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return corr_feature_selection(
            work_df,
            dep_vars=dep_cols,
            ind_vars=feat_list,
            n_min_clusters=eff_min,
            n_max_clusters=eff_max,
            use_pbar=use_pbar,
        )


def run_univariate_selection(
    x_df: pd.DataFrame,
    y_arr: np.ndarray,
    feat_cols: Sequence[str],
    score_callable: Callable[[Any, Any], Any],
    k: Optional[Union[int, str]] = 10,
    percentile: Optional[int] = None,
    out_scores_file: Optional[Union[str, os.PathLike]] = None,
    return_scores: bool = False,
) -> Union[List[str], Tuple[List[str], pd.DataFrame]]:
    """Run ``SelectKBest`` or ``SelectPercentile`` and build sorted score table."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    feat_list = list(feat_cols)
    if percentile is not None:
        if percentile <= 0 or percentile > 100:
            raise rsgispy.RSGISPyException(
                "percentile must be in the range (0, 100]."
            )
        selector: Any = sklearn.feature_selection.SelectPercentile(
            score_func=score_callable, percentile=int(percentile)
        )
    else:
        if isinstance(k, (int, np.integer)) and not isinstance(k, bool):
            if int(k) < 1:
                raise rsgispy.RSGISPyException("k must be at least 1.")
            eff_k: Union[int, str] = min(int(k), len(feat_list))
        elif k == "all":
            eff_k = "all"
        else:
            raise rsgispy.RSGISPyException(
                "k must be a positive integer or 'all'."
            )
        selector = sklearn.feature_selection.SelectKBest(
            score_func=score_callable, k=eff_k
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        selector.fit(x_df, y_arr)

    support = np.asarray(selector.get_support(), dtype=bool)
    raw_scores = np.asarray(selector.scores_, dtype=np.float64)
    raw_pvals = (
        np.asarray(selector.pvalues_, dtype=np.float64)
        if getattr(selector, "pvalues_", None) is not None
        else np.full(len(feat_list), np.nan, dtype=np.float64)
    )

    selected = [col for col, keep in zip(feat_list, support) if keep]
    scores_df = pd.DataFrame(
        {
            "feature": feat_list,
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


def run_rfe_selection(
    sk_estimator: Any,
    x_df: pd.DataFrame,
    y_arr: np.ndarray,
    feat_cols: Sequence[str],
    n_features_to_select: Optional[Union[int, float]] = None,
    step: Union[int, float] = 1,
    use_cv: bool = False,
    cv: Union[int, Any] = 5,
    scoring: Optional[Union[str, Any]] = None,
    min_features_to_select: int = 1,
    stratified: bool = True,
    rnd_seed: Optional[int] = None,
    out_scores_file: Optional[Union[str, os.PathLike]] = None,
    out_cv_plot: Optional[Union[str, os.PathLike]] = None,
    n_jobs: int = 1,
    extra: str = "classification",
    return_scores: bool = False,
) -> Union[List[str], Tuple[List[str], pd.DataFrame]]:
    """Execute RFE or RFECV with optional CV curve plotting."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    feat_list = list(feat_cols)
    base_est = clone_sklearn_estimator(
        sk_estimator, rnd_seed=rnd_seed, extra=extra
    )
    if use_cv:
        cv_splitter = resolve_cv_splitter(
            cv, y_arr, rnd_seed=rnd_seed, stratified=stratified
        )
        selector: Any = sklearn.feature_selection.RFECV(
            estimator=base_est,
            step=step,
            min_features_to_select=min(int(min_features_to_select), len(feat_list)),
            cv=cv_splitter,
            scoring=scoring,
            n_jobs=n_jobs,
        )
    else:
        eff_n = n_features_to_select
        if isinstance(eff_n, (int, np.integer)) and not isinstance(eff_n, bool):
            eff_n = min(max(1, int(eff_n)), len(feat_list))
        selector = sklearn.feature_selection.RFE(
            estimator=base_est,
            n_features_to_select=eff_n,
            step=step,
        )

    selector.fit(x_df, y_arr)
    support = np.asarray(selector.get_support(), dtype=bool)
    ranking = np.asarray(selector.ranking_, dtype=int)
    selected = [col for col, keep in zip(feat_list, support) if keep]

    rank_df = pd.DataFrame(
        {"feature": feat_list, "ranking": ranking, "selected": support}
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
                    len(feat_list) - len(mean_scores) + 1, len(feat_list) + 1
                ),
            ),
            dtype=int,
        )
        fig, _ = _plot_rfecv_scores(
            n_feats_axis, mean_scores, int(selector.n_features_)
        )
        finalise_plot(fig, out_file=out_cv_plot)

    if out_scores_file is not None:
        rank_df.to_csv(ensure_parent_dir(out_scores_file), index=False)
    if return_scores:
        return selected, rank_df
    return selected


def run_from_model_selection(
    sk_estimator: Optional[Any],
    x_df: pd.DataFrame,
    y_arr: np.ndarray,
    feat_cols: Sequence[str],
    threshold: Optional[Union[str, float]] = "mean",
    max_features: Optional[int] = None,
    prefit: bool = False,
    use_lassolars: bool = False,
    alpha_val: Optional[float] = None,
    rnd_seed: Optional[int] = None,
    out_scores_file: Optional[Union[str, os.PathLike]] = None,
    return_scores: bool = False,
    extra: str = "classification",
    role_label: str = "sk_classifier",
) -> Union[List[str], Tuple[List[str], pd.DataFrame]]:
    """Run ``SelectFromModel`` or ``lassolars_feature_selection``."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    feat_list = list(feat_cols)
    if use_lassolars:
        work_df = x_df[feat_list].copy()
        y_2d = np.asarray(y_arr, dtype=np.float64)
        if y_2d.ndim == 1:
            y_2d = y_2d[:, None]
        dep_cols: List[str] = []
        for idx in range(y_2d.shape[1]):
            col_key = f"__rsgis_target_{idx}__"
            work_df[col_key] = y_2d[:, idx]
            dep_cols.append(col_key)
        selected = lassolars_feature_selection(
            work_df,
            dep_vars=dep_cols,
            ind_vars=feat_list,
            alpha_val=alpha_val,
        )
        selected_set = set(selected)
        scores_df = pd.DataFrame(
            {
                "feature": feat_list,
                "selected": [col in selected_set for col in feat_list],
            }
        )
        if out_scores_file is not None:
            scores_df.to_csv(ensure_parent_dir(out_scores_file), index=False)
        if return_scores:
            return selected, scores_df
        return selected

    if sk_estimator is None:
        raise rsgispy.RSGISPyException(
            f"{role_label} must be provided when use_lassolars=False."
        )

    if prefit:
        check_sklearn_estimator_fitted(sk_estimator, label=role_label)
        est = sk_estimator
    else:
        est = fit_clean_sklearn_estimator(
            sk_estimator,
            x_df,
            y_arr,
            sync_input=False,
            rnd_seed=rnd_seed,
            extra=extra,
        )

    selector = sklearn.feature_selection.SelectFromModel(
        est, threshold=threshold, max_features=max_features, prefit=True
    )
    support = np.asarray(selector.get_support(), dtype=bool)
    selected = [col for col, keep in zip(feat_list, support) if keep]

    importances = extract_native_feature_importances(est, len(feat_list))
    scores_df = pd.DataFrame(
        {
            "feature": feat_list,
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


def run_feature_importance(
    sk_estimator: Any,
    x_df: pd.DataFrame,
    y_arr: np.ndarray,
    feat_cols: Sequence[str],
    method: str = "permutation",
    scoring: Optional[Union[str, Any]] = None,
    n_repeats: int = 10,
    rnd_seed: Optional[int] = None,
    n_jobs: int = 1,
    out_importance_file: Optional[Union[str, os.PathLike]] = None,
    out_plot_file: Optional[Union[str, os.PathLike]] = None,
    plot_title: str = "Feature Importance",
    missing_native_msg: str = "Estimator does not expose 'feature_importances_' or 'coef_'.",
) -> pd.DataFrame:
    """Compute permutation, native model, or combined feature importances."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    feat_list = list(feat_cols)
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

    data: Dict[str, Any] = {"feature": feat_list}
    sort_col = "importance_mean"
    err_vals: Optional[np.ndarray] = None

    if eff_method in {"model", "both"}:
        native_imp = extract_native_feature_importances(
            sk_estimator, len(feat_list)
        )
        if native_imp is None:
            if eff_method == "model":
                raise rsgispy.RSGISPyException(missing_native_msg)
        else:
            data["model_importance"] = native_imp
            if eff_method == "model":
                sort_col = "model_importance"

    if eff_method in {"permutation", "both"}:
        perm_res = sklearn.inspection.permutation_importance(
            sk_estimator,
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
        fig, _ = plot_feature_importance_bars(
            features=imp_df["feature"].tolist(),
            importances=imp_df[sort_col].to_numpy(),
            errors=plot_errs,
            title=plot_title,
            xlabel=xlabel,
        )
        finalise_plot(fig, out_file=out_plot_file)

    return imp_df
