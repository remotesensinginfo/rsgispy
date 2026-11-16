"""
Shared internal plotting, SHAP explanation, and SALib Sobol sensitivity helpers
reused across :mod:`rsgispy.classification` and :mod:`rsgispy.modelling`.
"""

import inspect
import os
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import rsgispy
from rsgispy.tools._sklearn import (
    check_sklearn_available,
    clone_sklearn_estimator,
    sklearn,
)
from rsgispy.tools.files import ensure_parent_dir

try:
    import scipy.stats  # pylint: disable=import-error

    HAVE_SCIPY = True
except ImportError:
    scipy = None  # type: ignore[assignment]  # pylint: disable=invalid-name
    HAVE_SCIPY = False

try:
    import seaborn as sns  # pylint: disable=import-error

    HAVE_SEABORN = True
except ImportError:
    sns = None  # type: ignore[assignment]
    HAVE_SEABORN = False

try:
    import shap  # pylint: disable=import-error

    HAVE_SHAP = True
except ImportError:
    shap = None  # type: ignore[assignment]
    HAVE_SHAP = False

try:
    import SALib.analyze.sobol  # pylint: disable=import-error
    import SALib.sample.sobol  # pylint: disable=import-error

    HAVE_SALIB = True
except ImportError:
    SALib = None  # type: ignore[assignment]  # pylint: disable=invalid-name
    HAVE_SALIB = False

__all__ = [
    "HAVE_SALIB",
    "HAVE_SCIPY",
    "HAVE_SEABORN",
    "HAVE_SHAP",
    "finalise_plot",
    "plot_class_feature_distributions",
    "plot_correlation_heatmap",
    "plot_feature_importance_bars",
    "plot_jackknife_bars",
    "run_boruta_shap_selection",
    "run_salib_sobol_sensitivity",
    "run_shap_explainer",
]


def finalise_plot(
    fig: Any,
    out_file: Optional[Union[str, os.PathLike]] = None,
    dpi: int = 150,
) -> None:
    """
    Save or display ``fig`` and always close it to prevent Matplotlib state and
    memory leaks when iterating over multiple models or covariates.
    """
    try:
        if out_file is not None:
            out_path = ensure_parent_dir(out_file)
            fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        elif rsgispy.is_notebook():
            plt.show()
    finally:
        plt.close(fig)


def plot_class_feature_distributions(
    data_df: pd.DataFrame,
    feature_cols: Sequence[str],
    cls_col: str,
    out_dir: Union[str, os.PathLike] = "boxplots",
    box_plots: bool = False,
    use_pbar: bool = True,
) -> None:
    """
    Create distribution comparison plots (violin or box plots) grouped by
    ``cls_col`` for every column in ``feature_cols``.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    out_dir_str = os.fspath(out_dir)
    os.makedirs(out_dir_str, exist_ok=True)
    pbar = rsgispy.get_tqdm()
    for col_name in pbar(list(feature_cols), disable=not use_pbar):
        out_path = os.path.join(out_dir_str, f"{col_name}_comp_plot.png")
        fig, ax = plt.subplots(figsize=(6, 4))
        try:
            if HAVE_SEABORN:
                if box_plots:
                    sns.boxplot(data=data_df, x=cls_col, y=col_name, ax=ax)
                else:
                    sns.violinplot(data=data_df, x=cls_col, y=col_name, ax=ax)
            else:
                groups = sorted(
                    data_df[cls_col].dropna().unique().tolist(),
                    key=str,
                )
                series_list = [
                    data_df.loc[data_df[cls_col] == g, col_name]
                    .dropna()
                    .to_numpy()
                    for g in groups
                ]
                if box_plots:
                    ax.boxplot(series_list, tick_labels=[str(g) for g in groups])
                else:
                    ax.violinplot(series_list)
                    ax.set_xticks(range(1, len(groups) + 1))
                    ax.set_xticklabels([str(g) for g in groups])
                ax.set_xlabel(cls_col)
                ax.set_ylabel(col_name)
        finally:
            finalise_plot(fig, out_file=out_path)


def plot_correlation_heatmap(
    corr_df: pd.DataFrame,
    var_names: Sequence[str],
    out_plt_file: Union[str, os.PathLike],
    fig_width: int = 15,
    fig_height: int = 14,
) -> None:
    """
    Plot a lower-triangular correlation heatmap and save to ``out_plt_file``.
    """
    mask = np.triu(np.ones_like(corr_df, dtype=bool))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    try:
        if HAVE_SEABORN:
            cmap = sns.diverging_palette(230, 20, as_cmap=True)
            sns.heatmap(
                corr_df,
                mask=mask,
                cmap=cmap,
                vmax=0.3,
                center=0,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.5},
                ax=ax,
            )
        else:
            masked_vals = np.ma.array(
                corr_df.to_numpy(dtype=float), mask=mask
            )
            im = ax.imshow(masked_vals, cmap="coolwarm", vmin=-1.0, vmax=1.0)
            fig.colorbar(im, ax=ax, shrink=0.5)
            ticks = range(len(var_names))
            ax.set_xticks(ticks)
            ax.set_xticklabels(var_names, rotation=45, ha="right")
            ax.set_yticks(ticks)
            ax.set_yticklabels(var_names)
    finally:
        finalise_plot(fig, out_file=out_plt_file)


def plot_jackknife_bars(
    var_names: Sequence[str],
    leave_vals: Sequence[float],
    only_vals: Sequence[float],
    all_val: float,
    title: str,
    xlabel: str,
    out_plot: Union[str, os.PathLike],
) -> None:
    """
    Plot a horizontal jackknife comparison bar chart showing leave-one-out,
    single-variable-only, and all-variable baseline performance.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    n_vars = len(var_names)
    var_idx = list(range(n_vars))
    plt_height = (0.5 * n_vars) + 1.5
    var_idx_all = var_idx + [n_vars]
    all_var_names = list(var_names) + ["All"]

    fig, ax = plt.subplots(figsize=(8, plt_height))
    try:
        ax.barh(
            var_idx,
            leave_vals,
            height=0.8,
            color="#1E88E5",
            label="Without Var",
        )
        ax.barh(
            var_idx,
            only_vals,
            height=0.8,
            color="#FFC107",
            label="Only Var",
        )
        ax.barh(
            [n_vars],
            [all_val],
            height=0.8,
            color="#D81B60",
            label="All Vars",
        )
        ax.set_yticks(var_idx_all, labels=all_var_names)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.11),
            fancybox=True,
            shadow=False,
            ncol=3,
        )
        fig.tight_layout()
    finally:
        finalise_plot(fig, out_file=out_plot)


def plot_feature_importance_bars(
    features: Sequence[str],
    importances: Sequence[float],
    out_plot: Union[str, os.PathLike],
    errors: Optional[Sequence[float]] = None,
    title: str = "Feature Importance",
    xlabel: str = "Importance",
) -> None:
    """
    Plot a horizontal bar chart of sorted feature importances with optional
    error bars.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    feat_list = list(features)
    val_arr = np.asarray(importances, dtype=float)
    err_arr = (
        np.asarray(errors, dtype=float) if errors is not None else None
    )
    order = np.argsort(val_arr)
    sorted_feats = [feat_list[i] for i in order]
    sorted_vals = val_arr[order]
    sorted_errs = err_arr[order] if err_arr is not None else None

    plt_height = max(3.0, (0.45 * len(sorted_feats)) + 1.5)
    fig, ax = plt.subplots(figsize=(8, plt_height))
    try:
        y_pos = np.arange(len(sorted_feats))
        ax.barh(
            y_pos,
            sorted_vals,
            xerr=sorted_errs,
            color="#1E88E5",
            ecolor="#0D47A1",
            capsize=3,
        )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_feats)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        fig.tight_layout()
    finally:
        finalise_plot(fig, out_file=out_plot)


def _compute_shap_mean_abs_df(
    raw_shap: Any,
    feature_names: Sequence[str],
    class_labels: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Summarise mean absolute SHAP contributions per feature (and class)."""
    vals = np.asarray(getattr(raw_shap, "values", raw_shap), dtype=float)
    feat_list = list(feature_names)
    if vals.ndim == 2:
        mean_abs = np.mean(np.abs(vals), axis=0)
        out_df = pd.DataFrame(
            {"feature": feat_list, "mean_abs_shap": mean_abs}
        )
    elif vals.ndim == 3:
        mean_per_cls = np.mean(np.abs(vals), axis=0)  # (n_features, n_classes)
        n_classes = mean_per_cls.shape[1]
        if class_labels is not None and len(class_labels) == n_classes:
            cls_cols = [f"mean_abs_shap_{lbl}" for lbl in class_labels]
        else:
            cls_cols = [f"mean_abs_shap_cls_{i}" for i in range(n_classes)]
        data: Dict[str, Any] = {
            "feature": feat_list,
            "mean_abs_shap": np.mean(mean_per_cls, axis=1),
        }
        for c_idx, col_name in enumerate(cls_cols):
            data[col_name] = mean_per_cls[:, c_idx]
        out_df = pd.DataFrame(data)
    else:
        raise rsgispy.RSGISPyException(
            f"Unexpected SHAP values shape {vals.shape}."
        )
    out_df = out_df.sort_values(
        "mean_abs_shap", ascending=False
    ).reset_index(drop=True)
    return out_df


def _subsample_shap_df(
    train_x: pd.DataFrame,
    subsample_n_smpls: Optional[int],
    rnd_seed: Optional[int],
) -> pd.DataFrame:
    """Subsample ``train_x`` for SHAP evaluation when ``subsample_n_smpls`` is set."""
    if subsample_n_smpls is not None and subsample_n_smpls > 10:
        sample_k = min(int(subsample_n_smpls), len(train_x))
        if rnd_seed is not None:
            return shap.sample(train_x, sample_k, random_state=int(rnd_seed))
        return shap.sample(train_x, sample_k)
    return train_x


def _build_shap_explainer(
    est_cls_obj: Any,
    eval_x: pd.DataFrame,
    feature_names: Sequence[str],
    use_tree_explainer: Optional[bool] = False,
    use_linear_explainer: bool = False,
) -> Any:
    """Create a SHAP explainer (Tree, Linear, Kernel, or auto-detected)."""
    if use_tree_explainer is True and use_linear_explainer:
        raise rsgispy.RSGISPyException(
            "You can only use one explainer: use_tree_explainer and "
            "use_linear_explainer cannot both be True."
        )

    var_list = list(feature_names)
    if use_tree_explainer is True:
        return shap.TreeExplainer(est_cls_obj, feature_names=var_list)
    if use_linear_explainer:
        return shap.LinearExplainer(
            est_cls_obj,
            eval_x,
            feature_perturbation="correlation_dependent",
        )
    if use_tree_explainer is None:
        is_tree = (
            hasattr(est_cls_obj, "estimators_")
            or hasattr(est_cls_obj, "tree_")
            or hasattr(est_cls_obj, "get_booster")
            or hasattr(est_cls_obj, "booster_")
        )
        if is_tree:
            try:
                return shap.TreeExplainer(est_cls_obj, feature_names=var_list)
            except Exception:  # pylint: disable=broad-exception-caught
                pass
        if hasattr(est_cls_obj, "coef_"):
            try:
                return shap.LinearExplainer(
                    est_cls_obj,
                    eval_x,
                    feature_perturbation="correlation_dependent",
                )
            except Exception:  # pylint: disable=broad-exception-caught
                pass
    return shap.KernelExplainer(est_cls_obj.predict_proba, eval_x)


def run_shap_explainer(
    est_cls_obj: Any,
    train_x: pd.DataFrame,
    analysis_vars: Sequence[str],
    feature_name: str,
    extra: str,
    shap_summary_plot: Optional[Union[str, os.PathLike]] = None,
    shap_heatmap_plot: Optional[Union[str, os.PathLike]] = None,
    shap_scatter_plots_dir: Optional[Union[str, os.PathLike]] = None,
    shap_depend_plots_dir: Optional[Union[str, os.PathLike]] = None,
    subsample_n_smpls: Optional[int] = None,
    use_tree_explainer: bool = False,
    use_linear_explainer: bool = False,
    target_class_idx: Optional[int] = -1,
    class_labels: Optional[Sequence[str]] = None,
    rnd_seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Compute SHAP feature explanations for a fitted ``scikit-learn`` classifier
    and generate summary, heatmap, scatter, and partial dependence plots.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-branches
    rsgispy.check_optional_dep(HAVE_SHAP, feature_name, "shap", extra)
    if use_tree_explainer and use_linear_explainer:
        raise rsgispy.RSGISPyException(
            "You can only use one explainer: use_tree_explainer and "
            "use_linear_explainer cannot both be True."
        )

    if rsgispy.is_notebook():
        shap.initjs()

    var_list = list(analysis_vars)
    train_x_smpl = _subsample_shap_df(train_x, subsample_n_smpls, rnd_seed)
    explainer = _build_shap_explainer(
        est_cls_obj=est_cls_obj,
        eval_x=train_x_smpl,
        feature_names=var_list,
        use_tree_explainer=use_tree_explainer,
        use_linear_explainer=use_linear_explainer,
    )

    raw_shap_values = explainer(train_x_smpl)
    summary_df = _compute_shap_mean_abs_df(
        raw_shap_values, var_list, class_labels=class_labels
    )

    raw_vals = getattr(raw_shap_values, "values", None)
    shap_vals_arr: Optional[np.ndarray] = (
        np.asarray(raw_vals, dtype=float) if raw_vals is not None else None
    )
    if shap_vals_arr is not None and shap_vals_arr.ndim == 3:
        eff_idx = -1 if target_class_idx is None else int(target_class_idx)
        shap_2d: Any = raw_shap_values[:, :, eff_idx]
    else:
        shap_2d = raw_shap_values

    if shap_summary_plot is not None:
        if (
            shap_vals_arr is not None
            and shap_vals_arr.ndim == 3
            and target_class_idx is None
        ):
            summary_target: Any = [
                shap_vals_arr[:, :, i] for i in range(shap_vals_arr.shape[2])
            ]
            shap.summary_plot(
                summary_target,
                train_x_smpl,
                class_names=list(class_labels) if class_labels is not None else None,
                show=False,
            )
        else:
            shap.summary_plot(shap_2d, train_x_smpl, show=False)
        finalise_plot(plt.gcf(), out_file=shap_summary_plot)

    if shap_heatmap_plot is not None:
        shap.plots.heatmap(shap_2d, show=False)
        finalise_plot(plt.gcf(), out_file=shap_heatmap_plot)

    if shap_scatter_plots_dir is not None:
        scatter_dir = os.fspath(shap_scatter_plots_dir)
        os.makedirs(scatter_dir, exist_ok=True)
        for name in var_list:
            out_plot_file = os.path.join(scatter_dir, f"shap_scatter_{name}.png")
            shap.plots.scatter(shap_2d[:, name], show=False)
            finalise_plot(plt.gcf(), out_file=out_plot_file)

    if shap_depend_plots_dir is not None:
        depend_dir = os.fspath(shap_depend_plots_dir)
        os.makedirs(depend_dir, exist_ok=True)
        sample_ind = min(100, max(0, len(train_x_smpl) - 1))
        if shap_vals_arr is not None and shap_vals_arr.ndim == 3:
            eff_idx = -1 if target_class_idx is None else int(target_class_idx)
            norm_idx = (
                eff_idx if eff_idx >= 0 else shap_vals_arr.shape[2] + eff_idx
            )

            def _dep_pred(x_arr: Any, _idx: int = norm_idx) -> np.ndarray:
                return np.asarray(est_cls_obj.predict_proba(x_arr))[:, _idx]

            model_fn: Callable[[Any], np.ndarray] = _dep_pred
        else:
            model_fn = est_cls_obj.predict_proba

        for name in var_list:
            out_plot_file = os.path.join(depend_dir, f"shap_depends_{name}.png")
            # pylint: disable=unpacking-non-sequence
            fig, _ = shap.partial_dependence_plot(
                name,
                model_fn,
                train_x_smpl,
                model_expected_value=True,
                feature_expected_value=True,
                show=False,
                ice=False,
                shap_values=shap_2d[sample_ind:sample_ind + 1, :],
            )
            finalise_plot(fig, out_file=out_plot_file)  # pylint: disable=unreachable

    return summary_df


def run_salib_sobol_sensitivity(
    train_df: pd.DataFrame,
    analysis_vars: Sequence[str],
    predict_1d_fn: Callable[[pd.DataFrame], np.ndarray],
    feature_name: str,
    extra: str,
    n_samples: int = 10000,
    sobol_plot_file: Optional[Union[str, os.PathLike]] = None,
    sobol_overall_file: Optional[Union[str, os.PathLike]] = None,
    sobol_first_file: Optional[Union[str, os.PathLike]] = None,
    sobol_second_file: Optional[Union[str, os.PathLike]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Execute SALib Sobol global sensitivity analysis over ``analysis_vars`` using
    ``predict_1d_fn`` and save optional CSV tables and plots.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    rsgispy.check_optional_dep(HAVE_SALIB, feature_name, "SALib", extra)
    var_list = list(analysis_vars)
    var_bounds: List[List[float]] = []
    for var in var_list:
        min_val = float(train_df[var].min())
        max_val = float(train_df[var].max())
        if max_val <= min_val:
            max_val = min_val + 1e-6
        var_bounds.append([min_val, max_val])
        print(f"{var} Range:\t({min_val} - {max_val})")

    problem = {
        "num_vars": len(var_list),
        "names": var_list,
        "bounds": var_bounds,
    }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        param_values = SALib.sample.sobol.sample(problem, N=int(n_samples))
    param_df = pd.DataFrame(data=param_values, columns=var_list)

    pred_y = np.asarray(predict_1d_fn(param_df), dtype=np.float64).ravel()
    si = SALib.analyze.sobol.analyze(
        problem, pred_y, calc_second_order=True, print_to_console=False
    )
    total_si, first_si, second_si = si.to_df()

    if sobol_overall_file is not None:
        total_si.to_csv(ensure_parent_dir(sobol_overall_file), index=False)
    if sobol_first_file is not None:
        first_si.to_csv(ensure_parent_dir(sobol_first_file), index=False)
    if sobol_second_file is not None:
        second_si.to_csv(ensure_parent_dir(sobol_second_file), index=False)

    if sobol_plot_file is not None:
        plt_width = ((len(var_list) * 0.4) * 3) + 2
        si.plot()
        fig = plt.gcf()
        fig.set_size_inches(plt_width, 10)
        fig.tight_layout()
        finalise_plot(fig, out_file=sobol_plot_file)

    return total_si, first_si, second_si


def _binom_pvalue(hits: int, n_trials: int, alternative: str) -> float:
    """Compute exact two-category binomial test p-value against ``p = 0.5``."""
    if hasattr(scipy.stats, "binomtest"):
        return float(
            scipy.stats.binomtest(
                hits, n_trials, p=0.5, alternative=alternative
            ).pvalue
        )
    binom_test_fn = getattr(scipy.stats, "binom_test")  # pragma: no cover
    return float(
        binom_test_fn(  # pragma: no cover
            hits, n_trials, p=0.5, alternative=alternative
        )
    )


def _shap_col_importances(
    raw_shap: Any,
    target_class_idx: Optional[int],
) -> np.ndarray:
    """Reduce SHAP values to 1D mean absolute importance per column."""
    vals = np.asarray(getattr(raw_shap, "values", raw_shap), dtype=float)
    if vals.ndim == 2:
        return np.mean(np.abs(vals), axis=0)
    if vals.ndim == 3:
        if target_class_idx is not None:
            return np.mean(np.abs(vals[:, :, target_class_idx]), axis=0)
        return np.mean(np.abs(vals), axis=(0, 2))
    raise rsgispy.RSGISPyException(
        f"Unexpected SHAP values shape {vals.shape} during BorutaShap selection."
    )


def _plot_boruta_shap_boxplots(
    history: Mapping[str, Sequence[float]],
    status: Mapping[str, str],
    shadow_max_hist: Sequence[float],
    shadow_mean_hist: Sequence[float],
    out_plot_file: Union[str, os.PathLike],
) -> None:
    """Plot BorutaShap importance distributions colour-coded by decision."""
    entries: List[Tuple[str, List[float], str]] = []
    for col, vals in history.items():
        if vals:
            entries.append((col, list(vals), status[col]))
    entries.sort(key=lambda item: float(np.median(item[1])))

    if shadow_mean_hist:
        entries.append(("ShadowMean", list(shadow_mean_hist), "ShadowMean"))
    if shadow_max_hist:
        entries.append(("ShadowMax", list(shadow_max_hist), "ShadowMax"))

    colour_map = {
        "Confirmed": "#2E7D32",
        "Tentative": "#F9A825",
        "Rejected": "#C62828",
        "ShadowMean": "#64B5F6",
        "ShadowMax": "#1565C0",
    }

    fig_height = max(3.5, 0.45 * len(entries) + 1.5)
    fig, ax = plt.subplots(figsize=(9, fig_height))
    try:
        bp_params = inspect.signature(ax.boxplot).parameters
        bp_kwargs: Dict[str, Any] = {"patch_artist": True}
        if "orientation" in bp_params:
            bp_kwargs["orientation"] = "horizontal"
        else:
            bp_kwargs["vert"] = False
        if "tick_labels" in bp_params:
            bp_kwargs["tick_labels"] = [item[0] for item in entries]
        else:
            bp_kwargs["labels"] = [item[0] for item in entries]

        bp = ax.boxplot([item[1] for item in entries], **bp_kwargs)
        for patch, (_, _, dec) in zip(bp["boxes"], entries):
            patch.set_facecolor(colour_map.get(dec, "#90A4AE"))
            patch.set_alpha(0.85)
        for median in bp["medians"]:
            median.set_color("#111111")
            median.set_linewidth(1.4)

        ax.set_xlabel("Mean |SHAP| Importance")
        ax.set_title("BorutaShap Feature Importance Distributions")
        fig.tight_layout()
    finally:
        finalise_plot(fig, out_file=out_plot_file)


def run_boruta_shap_selection(
    train_x: pd.DataFrame,
    train_y: np.ndarray,
    feature_names: Sequence[str],
    feature_name: str,
    extra: str,
    sk_classifier: Optional[Any] = None,
    n_trials: int = 50,
    pvalue: float = 0.05,
    percentile: float = 100.0,
    bonferroni: bool = True,
    include_tentative: bool = False,
    tentative_rough_fix: bool = False,
    target_class_idx: Optional[int] = None,
    subsample_n_smpls: Optional[int] = None,
    use_tree_explainer: Optional[bool] = None,
    use_linear_explainer: bool = False,
    rnd_seed: Optional[int] = None,
    out_plot_file: Optional[Union[str, os.PathLike]] = None,
    use_pbar: bool = True,
) -> Tuple[List[str], pd.DataFrame]:
    """
    Execute the Boruta-SHAP all-relevant feature selection algorithm.

    In each trial, randomised shadow copies of all non-rejected predictors are
    appended, a clean clone of ``sk_classifier`` is trained on the combined
    table, and mean absolute SHAP attributions are compared against the
    specified percentile of shadow attributions via exact binomial testing
    (``scipy.stats.binomtest``).
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    check_sklearn_available(feature_name, extra=extra)
    rsgispy.check_optional_dep(HAVE_SHAP, feature_name, "shap", extra)
    rsgispy.check_optional_dep(HAVE_SCIPY, feature_name, "scipy", extra)

    feat_list = list(feature_names)
    if len(feat_list) < 2:
        raise rsgispy.RSGISPyException(
            f"{feature_name} requires at least 2 features."
        )
    if n_trials < 1:
        raise rsgispy.RSGISPyException("n_trials must be at least 1.")
    if pvalue <= 0.0 or pvalue >= 1.0:
        raise rsgispy.RSGISPyException("pvalue must be in the open interval (0, 1).")
    if percentile <= 0.0 or percentile > 100.0:
        raise rsgispy.RSGISPyException(
            "percentile must be in the range (0, 100]."
        )
    if use_tree_explainer is True and use_linear_explainer:
        raise rsgispy.RSGISPyException(
            "You can only use one explainer: use_tree_explainer and "
            "use_linear_explainer cannot both be True."
        )

    base_clf = (
        sk_classifier
        if sk_classifier is not None
        else sklearn.ensemble.RandomForestClassifier(n_estimators=100)
    )
    rng = (
        np.random.default_rng(abs(int(rnd_seed)))
        if rnd_seed is not None
        else np.random.default_rng()
    )
    eff_alpha = float(pvalue) / float(len(feat_list)) if bonferroni else float(pvalue)

    status: Dict[str, str] = {col: "Tentative" for col in feat_list}
    hits: Dict[str, int] = {col: 0 for col in feat_list}
    n_eval: Dict[str, int] = {col: 0 for col in feat_list}
    history: Dict[str, List[float]] = {col: [] for col in feat_list}
    shadow_max_hist: List[float] = []
    shadow_mean_hist: List[float] = []

    pbar = rsgispy.get_tqdm()
    for trial_idx in pbar(range(1, int(n_trials) + 1), disable=not use_pbar):
        tentative_cols = [col for col in feat_list if status[col] == "Tentative"]
        if not tentative_cols:
            break
        active_cols = [col for col in feat_list if status[col] != "Rejected"]
        if not active_cols:
            break

        trial_seed = int(rng.integers(0, 2**31 - 1))
        n_shadows = max(5, len(active_cols))
        shadow_data: Dict[str, np.ndarray] = {}
        for s_idx in range(n_shadows):
            src_col = active_cols[s_idx % len(active_cols)]
            vals = train_x[src_col].to_numpy(copy=True)
            rng.shuffle(vals)
            shadow_data[f"__shadow_{s_idx}_{src_col}__"] = vals
        shadow_df = pd.DataFrame(shadow_data, index=train_x.index)
        comb_x = pd.concat([train_x[active_cols], shadow_df], axis=1)

        trial_est = clone_sklearn_estimator(
            base_clf, random_state=trial_seed, extra=extra
        )
        trial_est.fit(comb_x, train_y)

        eval_x = _subsample_shap_df(comb_x, subsample_n_smpls, trial_seed)
        explainer = _build_shap_explainer(
            est_cls_obj=trial_est,
            eval_x=eval_x,
            feature_names=list(comb_x.columns),
            use_tree_explainer=use_tree_explainer,
            use_linear_explainer=use_linear_explainer,
        )
        raw_shap = explainer(eval_x)
        col_imps = _shap_col_importances(raw_shap, target_class_idx)

        n_act = len(active_cols)
        feat_imps = col_imps[:n_act]
        shadow_imps = col_imps[n_act:]
        shadow_thresh = float(np.percentile(shadow_imps, float(percentile)))
        shadow_max_hist.append(shadow_thresh)
        shadow_mean_hist.append(float(np.mean(shadow_imps)))

        for col_idx, col_name in enumerate(active_cols):
            imp_val = float(feat_imps[col_idx])
            history[col_name].append(imp_val)
            n_eval[col_name] = trial_idx
            if imp_val > shadow_thresh:
                hits[col_name] += 1

            if status[col_name] == "Tentative":
                p_acc = _binom_pvalue(hits[col_name], trial_idx, "greater")
                p_rej = _binom_pvalue(hits[col_name], trial_idx, "less")
                if p_acc <= eff_alpha:
                    status[col_name] = "Confirmed"
                elif p_rej <= eff_alpha:
                    status[col_name] = "Rejected"

    if tentative_rough_fix and shadow_max_hist:
        med_shadow = float(np.median(shadow_max_hist))
        for col_name in feat_list:
            if status[col_name] == "Tentative" and history[col_name]:
                if float(np.median(history[col_name])) > med_shadow:
                    status[col_name] = "Confirmed"
                else:
                    status[col_name] = "Rejected"

    keep_decisions = (
        {"Confirmed", "Tentative"} if include_tentative else {"Confirmed"}
    )
    selected = [col for col in feat_list if status[col] in keep_decisions]

    decision_order = {"Confirmed": 0, "Tentative": 1, "Rejected": 2}
    rows: List[Dict[str, Any]] = []
    for col_name in feat_list:
        vals = history[col_name]
        trials_ran = max(1, n_eval[col_name])
        p_val_acc = (
            _binom_pvalue(hits[col_name], trials_ran, "greater")
            if n_eval[col_name] > 0
            else 1.0
        )
        rows.append(
            {
                "feature": col_name,
                "decision": status[col_name],
                "selected": col_name in selected,
                "hits": int(hits[col_name]),
                "n_trials_evaluated": int(n_eval[col_name]),
                "mean_importance": float(np.mean(vals)) if vals else 0.0,
                "median_importance": float(np.median(vals)) if vals else 0.0,
                "p_value_accept": float(p_val_acc),
                "_dec_rank": decision_order[status[col_name]],
            }
        )

    scores_df = pd.DataFrame(rows)
    scores_df = (
        scores_df.sort_values(
            ["_dec_rank", "median_importance", "feature"],
            ascending=[True, False, True],
        )
        .drop(columns=["_dec_rank"])
        .reset_index(drop=True)
    )
    scores_df["rank"] = np.arange(1, len(scores_df) + 1, dtype=int)

    if out_plot_file is not None:
        _plot_boruta_shap_boxplots(
            history=history,
            status=status,
            shadow_max_hist=shadow_max_hist,
            shadow_mean_hist=shadow_mean_hist,
            out_plot_file=out_plot_file,
        )

    return selected, scores_df
