"""
Model training, hyperparameter search, k-fold ensemble fitting, explainability,
and variable importance routines for :mod:`rsgispy.modelling.species_distribution`.
"""

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

import rsgispy
from rsgispy.modelling._helpers import (
    EnvVarInfo,
    calc_binary_roc_auc,
    calc_single_class_predictions,
    check_sklearn_available,
    extract_features_and_target,
    fit_clean_estimator,
    get_positive_class_probs,
    lookup_env_var,
    plot_roc_curve,
    read_model_vector,
    sklearn,
)
from rsgispy.tools._ml_explain import (
    finalise_plot,
    plot_jackknife_bars,
    run_salib_sobol_sensitivity,
    run_shap_explainer,
    run_variable_response_curves,
)
import rsgispy.tools.files
from rsgispy.vectorprocess import VectorInput, validate_columns_exist

__all__ = [
    "search_mdl_params",
    "fit_sklearn_mdl",
    "fit_sklearn_sgl_cls_mdl",
    "fit_kfold_sklearn_mdls",
    "fit_kfold_sklearn_sgl_cls_mdls",
    "shap_sklearn_mdl_explainer",
    "salib_sklearn_mdl_sensitivity",
    "sklearn_mdl_variable_response_curves",
    "sklearn_jackknife_var_importance",
]


def search_mdl_params(
    search_obj: Any,
    train_vec_file: VectorInput,
    train_vec_lyr: Optional[str],
    analysis_vars: Sequence[str],
    cls_col: str = "clsid",
) -> Tuple[Any, Dict[str, Any]]:
    """
    Execute a cloned ``scikit-learn`` hyperparameter search object (e.g.,
    ``GridSearchCV`` or ``RandomizedSearchCV``) without mutating stale state.

    :param search_obj: Unfitted ``scikit-learn`` SearchCV estimator.
    :param train_vec_file: Training sample vector path or GeoDataFrame.
    :param train_vec_lyr: Training vector layer name.
    :param analysis_vars: Ordered predictor column names.
    :param cls_col: Binary class column name (Default = ``'clsid'``).
    :return: Tuple ``(best_estimator, best_params)``.

    .. code:: python

        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import GridSearchCV
        import rsgispy.modelling.species_distribution as sdm

        search = GridSearchCV(RandomForestClassifier(), {"n_estimators": [20, 50]})
        best_est, best_params = sdm.search_mdl_params(
            search, train_gdf, None, ["v1", "v2"]
        )
    """
    check_sklearn_available("search_mdl_params")
    train_gdf = read_model_vector(train_vec_file, vec_lyr=train_vec_lyr)
    train_x, train_y = extract_features_and_target(
        train_gdf, analysis_vars, cls_col=cls_col
    )

    clean_search = fit_clean_estimator(
        search_obj, train_x, train_y, sync_input=False
    )
    if not getattr(clean_search, "refit", False):
        raise rsgispy.RSGISPyException(
            "Search did not find a refitted model and failed."
        )

    print(
        f"Best score was {clean_search.best_score_} and has parameters "
        f"{clean_search.best_params_}."
    )
    return clean_search.best_estimator_, clean_search.best_params_


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
def fit_sklearn_mdl(
    est_cls_obj: Any,
    train_vec_file: VectorInput,
    train_vec_lyr: Optional[str],
    test_vec_file: VectorInput,
    test_vec_lyr: Optional[str],
    analysis_vars: Sequence[str],
    cls_col: str = "clsid",
    roc_curve_plot: Optional[str] = None,
    return_model: bool = False,
) -> Union[Tuple[float, float, float], Tuple[Any, float, float, float]]:
    """
    Fit a ``scikit-learn`` binary classifier on a clean cloned instance so no
    state leaks across repeated model-comparison loops, returning training
    accuracy, testing accuracy, and ROC AUC (and optionally the fitted model).

    :param est_cls_obj: ``scikit-learn`` classifier estimator template.
    :param train_vec_file: Training vector path or :class:`geopandas.GeoDataFrame`.
    :param train_vec_lyr: Training vector layer name.
    :param test_vec_file: Testing vector path or :class:`geopandas.GeoDataFrame`.
    :param test_vec_lyr: Testing vector layer name.
    :param analysis_vars: Ordered predictor column names.
    :param cls_col: Binary class column name (Default = ``'clsid'``).
    :param roc_curve_plot: Optional output path for the ROC curve PNG.
    :param return_model: When ``True``, prepend the cleanly cloned fitted
                         estimator to the returned tuple (without mutating
                         ``est_cls_obj``). When ``False`` (default),
                         ``est_cls_obj`` is updated cleanly with the fitted
                         clone's state and ``(train_acc, test_acc, roc_auc)``
                         is returned.
    :return: ``(train_acc, test_acc, roc_auc)`` or
             ``(fitted_model, train_acc, test_acc, roc_auc)``.

    .. code:: python

        from sklearn.ensemble import RandomForestClassifier
        import rsgispy.modelling.species_distribution as sdm

        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        train_acc, test_acc, auc_val = sdm.fit_sklearn_mdl(
            clf, train_gdf, None, test_gdf, None, ["v1", "v2"]
        )
    """
    check_sklearn_available("fit_sklearn_mdl")
    train_gdf = read_model_vector(train_vec_file, vec_lyr=train_vec_lyr)
    test_gdf = read_model_vector(test_vec_file, vec_lyr=test_vec_lyr)

    train_x, train_y = extract_features_and_target(
        train_gdf, analysis_vars, cls_col=cls_col
    )
    test_x, test_y = extract_features_and_target(
        test_gdf, analysis_vars, cls_col=cls_col
    )
    if train_y is None or test_y is None:
        raise rsgispy.RSGISPyException(
            f"Target column '{cls_col}' could not be extracted from data."
        )

    fitted_mdl = fit_clean_estimator(
        est_cls_obj, train_x, train_y, sync_input=not return_model
    )

    train_acc_val = float(fitted_mdl.score(train_x, train_y))
    print(f"Classifier Train Score = {round(train_acc_val * 100, 2)}%")

    test_acc_val = float(fitted_mdl.score(test_x, test_y))
    print(f"Classifier Test Score = {round(test_acc_val * 100, 2)}%")

    test_probs = get_positive_class_probs(fitted_mdl, test_x, pos_label=1)
    fpr, tpr, roc_auc = calc_binary_roc_auc(test_y, test_probs)

    if roc_curve_plot is not None:
        fig, _ = plot_roc_curve(fpr, tpr, roc_auc)
        finalise_plot(fig, out_file=roc_curve_plot)

    if return_model:
        return fitted_mdl, train_acc_val, test_acc_val, roc_auc
    return train_acc_val, test_acc_val, roc_auc


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
def fit_sklearn_sgl_cls_mdl(
    est_cls_obj: Any,
    train_vec_file: VectorInput,
    train_vec_lyr: Optional[str],
    test_vec_file: VectorInput,
    test_vec_lyr: Optional[str],
    analysis_vars: Sequence[str],
    cls_col: str = "clsid",
    roc_curve_plot: Optional[str] = None,
    return_model: bool = False,
) -> Union[Tuple[float, float], Tuple[Any, float, float]]:
    """
    Fit a single-class / outlier ``scikit-learn`` estimator on a clean clone,
    returning testing accuracy and ROC AUC (and optionally the fitted model).

    :param est_cls_obj: Single-class ``scikit-learn`` estimator template.
    :param train_vec_file: Presence-only training vector path or GeoDataFrame.
    :param train_vec_lyr: Training vector layer name.
    :param test_vec_file: Testing vector path or GeoDataFrame with ``cls_col``.
    :param test_vec_lyr: Testing vector layer name.
    :param analysis_vars: Ordered predictor column names.
    :param cls_col: Binary class column name in ``test_vec_file`` (Default = ``'clsid'``).
    :param roc_curve_plot: Optional output path for ROC curve PNG.
    :param return_model: When ``True``, return ``(fitted_model, test_acc, roc_auc)``
                         without mutating ``est_cls_obj``.
    :return: ``(test_acc, roc_auc)`` or ``(fitted_model, test_acc, roc_auc)``.

    .. code:: python

        from sklearn.ensemble import IsolationForest
        import rsgispy.modelling.species_distribution as sdm

        mdl = IsolationForest(random_state=42)
        test_acc, auc_val = sdm.fit_sklearn_sgl_cls_mdl(
            mdl, pres_train_gdf, None, test_gdf, None, ["v1", "v2"]
        )
    """
    check_sklearn_available("fit_sklearn_sgl_cls_mdl")
    train_gdf = read_model_vector(train_vec_file, vec_lyr=train_vec_lyr)
    test_gdf = read_model_vector(test_vec_file, vec_lyr=test_vec_lyr)

    train_x, _ = extract_features_and_target(
        train_gdf, analysis_vars, cls_col=None
    )
    test_x, test_y = extract_features_and_target(
        test_gdf, analysis_vars, cls_col=cls_col
    )
    if test_y is None:
        raise rsgispy.RSGISPyException(
            f"Target column '{cls_col}' could not be extracted from test data."
        )

    fitted_mdl = fit_clean_estimator(
        est_cls_obj, train_x, train_y=None, sync_input=not return_model
    )

    test_pred = calc_single_class_predictions(fitted_mdl, test_x)
    test_acc_val = float(sklearn.metrics.accuracy_score(test_y, test_pred))
    print(f"Classifier Test Score = {round(test_acc_val * 100, 2)}%")

    test_scores = fitted_mdl.decision_function(test_x)
    fpr, tpr, roc_auc = calc_binary_roc_auc(test_y, test_scores)

    if roc_curve_plot is not None:
        fig, _ = plot_roc_curve(fpr, tpr, roc_auc)
        finalise_plot(fig, out_file=roc_curve_plot)

    if return_model:
        return fitted_mdl, test_acc_val, roc_auc
    return test_acc_val, roc_auc


def _ensure_all_classes_in_fold(
    train_idx: np.ndarray,
    train_y: np.ndarray,
    unique_classes: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Ensure every class in ``unique_classes`` appears at least once in
    ``train_idx`` by replacing surplus slots of already-present classes,
    preserving fold size and never introducing duplicate indices when
    ``train_idx`` was sampled without replacement.
    """
    present_classes = set(np.unique(train_y[train_idx]).tolist())
    if len(present_classes) >= unique_classes.size:
        return train_idx

    missing_classes = [
        cls_val for cls_val in unique_classes if cls_val not in present_classes
    ]
    seen_cls: set = set()
    replaceable_slots: List[int] = []
    for slot_pos, sample_idx in enumerate(train_idx):
        cls_val = train_y[sample_idx]
        if cls_val not in seen_cls:
            seen_cls.add(cls_val)
        else:
            replaceable_slots.append(slot_pos)

    updated_idx = train_idx.copy()
    for slot_pos, cls_val in zip(replaceable_slots, missing_classes):
        cls_candidates = np.flatnonzero(train_y == cls_val)
        updated_idx[slot_pos] = int(rng.choice(cls_candidates, size=1)[0])
    return updated_idx


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
def fit_kfold_sklearn_mdls(
    est_cls_obj: Any,
    train_vec_file: VectorInput,
    train_vec_lyr: Optional[str],
    test_vec_file: VectorInput,
    test_vec_lyr: Optional[str],
    analysis_vars: Sequence[str],
    n_kfolds: int = 10,
    fold_prop: float = 0.6,
    cls_col: str = "clsid",
    rnd_seed: Optional[int] = None,
    sel_replacement: bool = True,
    use_pbar: bool = True,
) -> List[Any]:
    """
    Fit an ensemble of cleanly cloned ``scikit-learn`` binary classifiers across
    ``n_kfolds`` random training subsets without mutating ``est_cls_obj`` or
    global NumPy random state.

    :param est_cls_obj: Base ``scikit-learn`` classifier estimator.
    :param train_vec_file: Training vector path or GeoDataFrame.
    :param train_vec_lyr: Training vector layer name.
    :param test_vec_file: Testing vector path or GeoDataFrame.
    :param test_vec_lyr: Testing vector layer name.
    :param analysis_vars: Ordered predictor column names.
    :param n_kfolds: Number of ensemble folds to fit (Default = ``10``).
    :param fold_prop: Proportion of training rows per fold in ``(0, 1]``.
    :param cls_col: Binary class column name (Default = ``'clsid'``).
    :param rnd_seed: Optional local random seed for reproducibility.
    :param sel_replacement: Sample training subsets with replacement (Default = ``True``).
    :param use_pbar: Display a tqdm progress bar (Default = ``True``).
    :return: List of cleanly cloned and fitted ``scikit-learn`` estimators.

    .. code:: python

        from sklearn.ensemble import RandomForestClassifier
        import rsgispy.modelling.species_distribution as sdm

        mdls = sdm.fit_kfold_sklearn_mdls(
            RandomForestClassifier(n_estimators=20),
            train_gdf, None, test_gdf, None, ["v1", "v2"],
            n_kfolds=5, rnd_seed=42,
        )
    """
    check_sklearn_available("fit_kfold_sklearn_mdls")
    if n_kfolds < 1:
        raise rsgispy.RSGISPyException("n_kfolds must be >= 1.")
    if not 0.0 < float(fold_prop) <= 1.0:
        raise rsgispy.RSGISPyException(
            f"fold_prop must be in (0, 1], got {fold_prop!r}."
        )

    rng = np.random.default_rng(rnd_seed)
    train_gdf = read_model_vector(train_vec_file, vec_lyr=train_vec_lyr)
    test_gdf = read_model_vector(test_vec_file, vec_lyr=test_vec_lyr)

    train_x, train_y = extract_features_and_target(
        train_gdf, analysis_vars, cls_col=cls_col
    )
    test_x, test_y = extract_features_and_target(
        test_gdf, analysis_vars, cls_col=cls_col
    )
    if train_y is None or test_y is None:
        raise rsgispy.RSGISPyException(
            f"Target column '{cls_col}' could not be extracted from data."
        )

    n_train = train_y.shape[0]
    if n_train == 0:
        raise rsgispy.RSGISPyException("Training dataset must not be empty.")
    n_fold_size = max(1, int(n_train * fold_prop))
    unique_classes = np.unique(train_y)
    if unique_classes.size > 1 and n_fold_size < unique_classes.size:
        raise rsgispy.RSGISPyException(
            f"Computed fold size ({n_fold_size}) is smaller than the number "
            f"of classes ({unique_classes.size}). Increase fold_prop or "
            "provide more training samples."
        )

    train_scores: List[float] = []
    test_scores: List[float] = []
    roc_scores: List[float] = []
    est_cls_objs: List[Any] = []

    pbar = rsgispy.get_tqdm()
    for _ in pbar(range(n_kfolds), disable=not use_pbar):
        train_idx = rng.choice(n_train, size=n_fold_size, replace=sel_replacement)
        if unique_classes.size > 1:
            train_idx = _ensure_all_classes_in_fold(
                train_idx, train_y, unique_classes, rng
            )

        train_y_set = train_y[train_idx]
        train_x_set = train_x.iloc[train_idx]

        fold_seed = int(rng.integers(0, 2**31 - 1)) if rnd_seed is not None else None
        est_cls_fold = fit_clean_estimator(
            est_cls_obj,
            train_x_set,
            train_y_set,
            sync_input=False,
            rnd_seed=fold_seed,
        )
        est_cls_objs.append(est_cls_fold)

        train_scores.append(float(est_cls_fold.score(train_x_set, train_y_set)))
        test_scores.append(float(est_cls_fold.score(test_x, test_y)))

        test_probs = get_positive_class_probs(est_cls_fold, test_x, pos_label=1)
        _, _, roc_auc = calc_binary_roc_auc(test_y, test_probs)
        roc_scores.append(roc_auc)

    print(
        f"ROC: {np.min(roc_scores)} - {np.median(roc_scores)} - {np.max(roc_scores)}"
    )
    print(
        f"Train: {np.min(train_scores)} - {np.median(train_scores)} - "
        f"{np.max(train_scores)}"
    )
    print(
        f"Test: {np.min(test_scores)} - {np.median(test_scores)} - "
        f"{np.max(test_scores)}"
    )

    return est_cls_objs


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
def fit_kfold_sklearn_sgl_cls_mdls(
    est_cls_obj: Any,
    train_vec_file: VectorInput,
    train_vec_lyr: Optional[str],
    test_vec_file: VectorInput,
    test_vec_lyr: Optional[str],
    analysis_vars: Sequence[str],
    n_kfolds: int = 10,
    fold_prop: float = 0.6,
    cls_col: str = "clsid",
    rnd_seed: Optional[int] = None,
    sel_replacement: bool = True,
    use_pbar: bool = True,
) -> List[Any]:
    """
    Fit an ensemble of cleanly cloned single-class / outlier ``scikit-learn``
    estimators across ``n_kfolds`` random presence subsets.

    :param est_cls_obj: Base single-class ``scikit-learn`` estimator.
    :param train_vec_file: Presence-only training vector path or GeoDataFrame.
    :param train_vec_lyr: Training vector layer name.
    :param test_vec_file: Testing vector path or GeoDataFrame with ``cls_col``.
    :param test_vec_lyr: Testing vector layer name.
    :param analysis_vars: Ordered predictor column names.
    :param n_kfolds: Number of ensemble folds (Default = ``10``).
    :param fold_prop: Proportion of training rows per fold in ``(0, 1]``.
    :param cls_col: Binary class column name in ``test_vec_file`` (Default = ``'clsid'``).
    :param rnd_seed: Optional local random seed for reproducibility.
    :param sel_replacement: Sample training subsets with replacement (Default = ``True``).
    :param use_pbar: Display a tqdm progress bar (Default = ``True``).
    :return: List of cleanly cloned and fitted ``scikit-learn`` estimators.

    .. code:: python

        from sklearn.ensemble import IsolationForest
        import rsgispy.modelling.species_distribution as sdm

        mdls = sdm.fit_kfold_sklearn_sgl_cls_mdls(
            IsolationForest(), pres_train_gdf, None, test_gdf, None,
            ["v1", "v2"], n_kfolds=5, rnd_seed=42,
        )
    """
    check_sklearn_available("fit_kfold_sklearn_sgl_cls_mdls")
    if n_kfolds < 1:
        raise rsgispy.RSGISPyException("n_kfolds must be >= 1.")
    if not 0.0 < float(fold_prop) <= 1.0:
        raise rsgispy.RSGISPyException(
            f"fold_prop must be in (0, 1], got {fold_prop!r}."
        )

    rng = np.random.default_rng(rnd_seed)
    train_gdf = read_model_vector(train_vec_file, vec_lyr=train_vec_lyr)
    test_gdf = read_model_vector(test_vec_file, vec_lyr=test_vec_lyr)

    train_x, _ = extract_features_and_target(
        train_gdf, analysis_vars, cls_col=None
    )
    test_x, test_y = extract_features_and_target(
        test_gdf, analysis_vars, cls_col=cls_col
    )
    if test_y is None:
        raise rsgispy.RSGISPyException(
            f"Target column '{cls_col}' could not be extracted from test data."
        )

    n_train = train_x.shape[0]
    n_fold_size = max(1, int(n_train * fold_prop))

    test_scores: List[float] = []
    roc_scores: List[float] = []
    est_cls_objs: List[Any] = []

    pbar = rsgispy.get_tqdm()
    for _ in pbar(range(n_kfolds), disable=not use_pbar):
        train_idx = rng.choice(n_train, size=n_fold_size, replace=sel_replacement)
        train_x_set = train_x.iloc[train_idx]

        fold_seed = int(rng.integers(0, 2**31 - 1)) if rnd_seed is not None else None
        est_cls_fold = fit_clean_estimator(
            est_cls_obj,
            train_x_set,
            train_y=None,
            sync_input=False,
            rnd_seed=fold_seed,
        )
        est_cls_objs.append(est_cls_fold)

        test_pred = calc_single_class_predictions(est_cls_fold, test_x)
        test_scores.append(float(sklearn.metrics.accuracy_score(test_y, test_pred)))

        test_probs = est_cls_fold.decision_function(test_x)
        _, _, roc_auc = calc_binary_roc_auc(test_y, test_probs)
        roc_scores.append(roc_auc)

    print(
        f"ROC: {np.min(roc_scores)} - {np.median(roc_scores)} - {np.max(roc_scores)}"
    )
    print(
        f"Test: {np.min(test_scores)} - {np.median(test_scores)} - "
        f"{np.max(test_scores)}"
    )

    return est_cls_objs


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals,too-many-branches
def shap_sklearn_mdl_explainer(
    est_cls_obj: Any,
    train_vec_file: VectorInput,
    train_vec_lyr: Optional[str],
    analysis_vars: Sequence[str],
    shap_summary_plot: Optional[str] = None,
    shap_heatmap_plot: Optional[str] = None,
    shap_scatter_plots_dir: Optional[str] = None,
    shap_depend_plots_dir: Optional[str] = None,
    subsample_n_smpls: Optional[int] = None,
    use_tree_explainer: Optional[bool] = None,
    use_linear_explainer: bool = False,
) -> None:
    """
    Compute SHAP feature explanations and generate summary, heatmap, scatter,
    and partial dependence plots for a fitted ``scikit-learn`` model.

    Requires optional dependency ``shap``.

    :param est_cls_obj: Fitted ``scikit-learn`` estimator.
    :param train_vec_file: Training vector path or :class:`geopandas.GeoDataFrame`.
    :param train_vec_lyr: Training vector layer name.
    :param analysis_vars: Ordered predictor column names.
    :param shap_summary_plot: Optional output path for SHAP summary plot.
    :param shap_heatmap_plot: Optional output path for SHAP heatmap plot.
    :param shap_scatter_plots_dir: Optional directory for per-variable SHAP scatter plots.
    :param shap_depend_plots_dir: Optional directory for per-variable partial dependence plots.
    :param subsample_n_smpls: Optional subsample size for SHAP evaluation.
    :param use_tree_explainer: Use ``shap.TreeExplainer`` when ``True``.
    :param use_linear_explainer: Use ``shap.LinearExplainer`` when ``True``.

    .. code:: python

        import rsgispy.modelling.species_distribution as sdm

        sdm.shap_sklearn_mdl_explainer(
            clf, train_gdf, None, ["v1", "v2"],
            shap_summary_plot="shap_summary.png", use_tree_explainer=True,
        )
    """
    train_gdf = read_model_vector(train_vec_file, vec_lyr=train_vec_lyr)
    train_x, _ = extract_features_and_target(
        train_gdf, analysis_vars, cls_col=None
    )
    run_shap_explainer(
        est_cls_obj=est_cls_obj,
        train_x=train_x,
        analysis_vars=analysis_vars,
        feature_name="shap_sklearn_mdl_explainer",
        extra="modelling",
        shap_summary_plot=shap_summary_plot,
        shap_heatmap_plot=shap_heatmap_plot,
        shap_scatter_plots_dir=shap_scatter_plots_dir,
        shap_depend_plots_dir=shap_depend_plots_dir,
        subsample_n_smpls=subsample_n_smpls,
        use_tree_explainer=use_tree_explainer,
        use_linear_explainer=use_linear_explainer,
        target_class_idx=-1,
    )


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
def salib_sklearn_mdl_sensitivity(
    est_cls_obj: Any,
    train_vec_file: VectorInput,
    train_vec_lyr: Optional[str],
    analysis_vars: Sequence[str],
    n_samples: int = 10000,
    sobol_plot_file: Optional[str] = None,
    sobol_overall_file: Optional[str] = None,
    sobol_first_file: Optional[str] = None,
    sobol_second_file: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform SALib Sobol sensitivity analysis to quantify first-order,
    second-order, and total-order feature sensitivities of ``est_cls_obj``.

    Requires optional dependency ``SALib``.

    :param est_cls_obj: Fitted ``scikit-learn`` classifier with ``predict_proba``.
    :param train_vec_file: Training vector path or :class:`geopandas.GeoDataFrame`.
    :param train_vec_lyr: Training vector layer name.
    :param analysis_vars: Ordered predictor column names.
    :param n_samples: Base Saltelli sample count (Default = ``10000``).
    :param sobol_plot_file: Optional output Sobol sensitivity plot path.
    :param sobol_overall_file: Optional total-order Sobol CSV output path.
    :param sobol_first_file: Optional first-order Sobol CSV output path.
    :param sobol_second_file: Optional second-order Sobol CSV output path.
    :return: Tuple ``(total_Si, first_Si, second_Si)`` DataFrames.

    .. code:: python

        import rsgispy.modelling.species_distribution as sdm

        total_si, first_si, second_si = sdm.salib_sklearn_mdl_sensitivity(
            clf, train_gdf, None, ["v1", "v2"], n_samples=256
        )
    """
    train_gdf = read_model_vector(train_vec_file, vec_lyr=train_vec_lyr)
    var_list = list(analysis_vars)
    validate_columns_exist(train_gdf, var_list)
    return run_salib_sobol_sensitivity(
        train_df=train_gdf,
        analysis_vars=var_list,
        predict_1d_fn=lambda df: get_positive_class_probs(
            est_cls_obj, df, pos_label=1
        ),
        feature_name="salib_sklearn_mdl_sensitivity",
        extra="modelling",
        n_samples=n_samples,
        sobol_plot_file=sobol_plot_file,
        sobol_overall_file=sobol_overall_file,
        sobol_first_file=sobol_first_file,
        sobol_second_file=sobol_second_file,
    )


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
def sklearn_mdl_variable_response_curves(
    est_cls_obj: Any,
    train_vec_file: VectorInput,
    train_vec_lyr: Optional[str],
    analysis_vars: Sequence[str],
    output_file: Optional[str],
    response_plots_dir: Optional[str],
    n_samples: int = 1000,
    cls_col: str = "clsid",
    normalised_data: bool = False,
    env_vars: Optional[Mapping[str, EnvVarInfo]] = None,
) -> pd.DataFrame:
    """
    Generate marginal response curves for each environmental variable while
    holding other covariates at their mean presence value.

    :param est_cls_obj: Fitted ``scikit-learn`` classifier with ``predict_proba``.
    :param train_vec_file: Training vector path or :class:`geopandas.GeoDataFrame`.
    :param train_vec_lyr: Training vector layer name.
    :param analysis_vars: Ordered predictor column names.
    :param output_file: Output CSV path for response curve data (or ``None``).
    :param response_plots_dir: Output directory for PNG plots (or ``None``).
    :param n_samples: Number of sample points along each variable range.
    :param cls_col: Binary class column name (Default = ``'clsid'``).
    :param normalised_data: Un-normalise X-axis values back to raw units using
                            ``env_vars`` when ``True``.
    :param env_vars: Mapping of :class:`EnvVarInfo` required if
                     ``normalised_data=True``.
    :return: :class:`pandas.DataFrame` containing ``<var>_vals`` and ``<var>_resp`` columns.

    .. code:: python

        import rsgispy.modelling.species_distribution as sdm

        df = sdm.sklearn_mdl_variable_response_curves(
            clf, train_gdf, None, ["v1", "v2"], "resp.csv", "resp_plots", n_samples=50
        )
    """
    if normalised_data and env_vars is None:
        raise rsgispy.RSGISPyException(
            "env_vars must be provided when normalised_data=True."
        )

    train_gdf = read_model_vector(train_vec_file, vec_lyr=train_vec_lyr)
    var_list = list(analysis_vars)
    validate_columns_exist(train_gdf, var_list + [cls_col])
    train_pres_gdf = train_gdf[train_gdf[cls_col] == 1]
    if train_pres_gdf.empty:
        train_pres_gdf = train_gdf

    for var in var_list:
        min_val = float(train_gdf[var].min())
        max_val = float(train_gdf[var].max())
        mean_val = float(train_pres_gdf[var].mean())
        print(f"{var} Range:\t({min_val} - {max_val}): Mean = {mean_val}")
        print(f"Calculating Response Curve for '{var}'")

    x_transform_map: Optional[Dict[str, Any]] = None
    if normalised_data and env_vars is not None:
        x_transform_map = {}
        for var in var_list:
            try:
                info = lookup_env_var(env_vars, var)
            except rsgispy.RSGISPyException:
                continue
            if info.norm_mean is not None and info.norm_std is not None:
                norm_std = float(info.norm_std)
                norm_mean = float(info.norm_mean)
                x_transform_map[var] = (
                    lambda vals, s=norm_std, m=norm_mean: (vals * s) + m
                )

    return run_variable_response_curves(
        x_df=train_gdf[var_list],
        feature_cols=var_list,
        predict_fn=lambda df: get_positive_class_probs(
            est_cls_obj, df, pos_label=1
        ),
        output_labels=["resp"],
        out_indices=[0],
        col_name_fn=lambda feat, _lbl, _i: f"{feat}_resp",
        n_samples=int(n_samples),
        summary_stat="mean",
        baseline_df=train_pres_gdf[var_list],
        method="marginal",
        x_transform_map=x_transform_map,
        out_file=output_file,
        response_plots_dir=response_plots_dir,
        ylabel="Class Probability",
        ylim=(0.0, 1.0),
        multi_panel=False,
        use_pbar=False,
    )


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
def sklearn_jackknife_var_importance(
    est_cls_obj: Any,
    train_vec_file: VectorInput,
    train_vec_lyr: Optional[str],
    test_vec_file: VectorInput,
    test_vec_lyr: Optional[str],
    analysis_vars: Sequence[str],
    out_var_importance_file: Optional[str],
    out_importance_auc_plot: Optional[str] = None,
    out_importance_test_plot: Optional[str] = None,
    cls_col: str = "clsid",
    use_pbar: bool = True,
) -> pd.DataFrame:
    """
    Evaluate variable importance via jackknife leave-one-out and single-variable
    training runs using clean estimator clones in every iteration.

    :param est_cls_obj: Base ``scikit-learn`` classifier estimator.
    :param train_vec_file: Training vector path or GeoDataFrame.
    :param train_vec_lyr: Training vector layer name.
    :param test_vec_file: Testing vector path or GeoDataFrame.
    :param test_vec_lyr: Testing vector layer name.
    :param analysis_vars: Ordered predictor column names (at least 2).
    :param out_var_importance_file: Output CSV path for jackknife metrics (or ``None``).
    :param out_importance_auc_plot: Optional horizontal bar plot path for ROC AUC.
    :param out_importance_test_plot: Optional horizontal bar plot path for test accuracy.
    :param cls_col: Binary class column name (Default = ``'clsid'``).
    :param use_pbar: Display a tqdm progress bar (Default = ``True``).
    :return: :class:`pandas.DataFrame` indexed by ``analysis_vars``.

    .. code:: python

        import rsgispy.modelling.species_distribution as sdm

        imp_df = sdm.sklearn_jackknife_var_importance(
            clf, train_gdf, None, test_gdf, None, ["v1", "v2", "v3"], "jackknife.csv"
        )
    """
    check_sklearn_available("sklearn_jackknife_var_importance")
    var_list = list(analysis_vars)
    if len(var_list) < 2:
        raise rsgispy.RSGISPyException(
            "Jackknife variable importance requires at least 2 variables."
        )

    train_gdf = read_model_vector(train_vec_file, vec_lyr=train_vec_lyr)
    test_gdf = read_model_vector(test_vec_file, vec_lyr=test_vec_lyr)

    train_x_all, train_y = extract_features_and_target(
        train_gdf, var_list, cls_col=cls_col
    )
    test_x_all, test_y = extract_features_and_target(
        test_gdf, var_list, cls_col=cls_col
    )
    if train_y is None or test_y is None:
        raise rsgispy.RSGISPyException(
            f"Target column '{cls_col}' could not be extracted from data."
        )

    def _eval_subset(sub_vars: List[str]) -> Tuple[float, float, float]:
        tr_sub = train_x_all[sub_vars]
        te_sub = test_x_all[sub_vars]
        est_sub = fit_clean_estimator(
            est_cls_obj, tr_sub, train_y, sync_input=False
        )
        tr_acc = float(est_sub.score(tr_sub, train_y))
        te_acc = float(est_sub.score(te_sub, test_y))
        te_probs = get_positive_class_probs(est_sub, te_sub, pos_label=1)
        _, _, roc_auc = calc_binary_roc_auc(test_y, te_probs)
        return tr_acc, te_acc, roc_auc

    _, all_test_acc_val, all_roc_auc = _eval_subset(var_list)

    out_data: Dict[str, List[float]] = {
        "train_acc_leave_var_out": [],
        "test_acc_leave_var_out": [],
        "roc_test_acc_leave_var_out": [],
        "train_acc_only_var": [],
        "test_acc_only_var": [],
        "roc_test_acc_only_var": [],
    }

    pbar = rsgispy.get_tqdm()
    for var in pbar(var_list, disable=not use_pbar):
        tr_acc, te_acc, roc_auc = _eval_subset([v for v in var_list if v != var])
        out_data["train_acc_leave_var_out"].append(tr_acc)
        out_data["test_acc_leave_var_out"].append(te_acc)
        out_data["roc_test_acc_leave_var_out"].append(roc_auc)

    for var in pbar(var_list, disable=not use_pbar):
        tr_acc, te_acc, roc_auc = _eval_subset([var])
        out_data["train_acc_only_var"].append(tr_acc)
        out_data["test_acc_only_var"].append(te_acc)
        out_data["roc_test_acc_only_var"].append(roc_auc)

    var_import_vals_df = pd.DataFrame(data=out_data, index=var_list)
    if out_var_importance_file is not None:
        imp_path = rsgispy.tools.files.ensure_parent_dir(out_var_importance_file)
        var_import_vals_df.to_csv(imp_path, index=True)

    if out_importance_auc_plot is not None:
        fig, _ = plot_jackknife_bars(
            var_list,
            out_data["roc_test_acc_leave_var_out"],
            out_data["roc_test_acc_only_var"],
            all_roc_auc,
            "Jackknife AUC Gain",
            "Testing AUC",
        )
        finalise_plot(fig, out_file=out_importance_auc_plot)

    if out_importance_test_plot is not None:
        fig, _ = plot_jackknife_bars(
            var_list,
            out_data["test_acc_leave_var_out"],
            out_data["test_acc_only_var"],
            all_test_acc_val,
            "Jackknife Test Gain",
            "Testing Accuracy",
        )
        finalise_plot(fig, out_file=out_importance_test_plot)

    return var_import_vals_df
