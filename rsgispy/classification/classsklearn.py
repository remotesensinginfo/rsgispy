"""
Scikit-learn classifier hyperparameter optimisation (Grid Search, Random Search,
and Bayesian optimisation via Optuna), classifier training, and block-wise
raster classification outputting class IDs and probability score maps.
"""

import os
from typing import (
    Any,
    Callable,
    Dict,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd

import rsgispy
from rsgispy.classification._cls_common import (
    apply_classifier_raster as _apply_classifier_raster,
    suggest_optuna_params as _suggest_optuna_params,
)
from rsgispy.classification._helpers import (
    ClsConfigInput,
    SamplesInput,
    calc_split_metrics as _calc_split_metrics,
    check_sklearn_available,
    coerce_cls_config as _coerce_cls_config,
    extract_xy as _extract_xy,
    load_samples_df as _load_samples_df,
    resolve_cv_splitter as _resolve_cv_splitter,
    resolve_feature_cols as _resolve_feature_cols,
)
from rsgispy.classification.classdata import ImageBandInfo
from rsgispy.imageprocess import ensure_file_exists
from rsgispy.tools._sklearn import (
    clone_sklearn_estimator,
    fit_clean_sklearn_estimator,
    is_sklearn_search_cv,
    sklearn,
)
from rsgispy.tools.files import ensure_parent_dir
from rsgispy.tools.jsonio import to_json_compatible, write_dict_to_json
from rsgispy.tools.tables import write_dataframe_table

try:
    import joblib  # pylint: disable=import-error

    HAVE_JOBLIB = True
except ImportError:
    joblib = None  # type: ignore[assignment]
    HAVE_JOBLIB = False

try:
    import optuna  # pylint: disable=import-error

    HAVE_OPTUNA = True
except ImportError:
    optuna = None  # type: ignore[assignment]
    HAVE_OPTUNA = False

__all__ = [
    "save_sklearn_classifier",
    "load_sklearn_classifier",
    "perform_sklearn_classifier_param_search",
    "train_sklearn_classifier",
    "apply_sklearn_classifier",
]

_GRID_ALIASES = {"grid", "gridsearch", "grid_search", "gridsearchcv"}
_RANDOM_ALIASES = {
    "random",
    "randomsearch",
    "random_search",
    "randomized",
    "randomizedsearch",
    "randomizedsearchcv",
}
_OPTUNA_ALIASES = {"optuna", "bayesian", "bayes", "tpe"}


def save_sklearn_classifier(
    sk_classifier: Any,
    out_file: Union[str, os.PathLike],
) -> None:
    """
    Save a trained ``scikit-learn`` (or compatible) classifier object to disk
    using ``joblib``.

    :param sk_classifier: Classifier object to persist.
    :param out_file: Output file path (e.g., ``'rf_classifier.joblib'``).

    .. code:: python

        from sklearn.ensemble import RandomForestClassifier
        import rsgispy.classification.classsklearn as csk

        clf = RandomForestClassifier(n_estimators=50)
        csk.save_sklearn_classifier(clf, "rf_classifier.joblib")
    """
    rsgispy.check_optional_dep(
        HAVE_JOBLIB, "save_sklearn_classifier", "joblib", "classification"
    )
    out_path = ensure_parent_dir(out_file)
    joblib.dump(sk_classifier, out_path)


def load_sklearn_classifier(
    in_file: Union[str, os.PathLike],
) -> Any:
    """
    Load a persisted ``scikit-learn`` (or compatible) classifier object from
    disk using ``joblib``.

    :param in_file: Path to the saved classifier file.
    :return: Loaded classifier object.

    .. code:: python

        import rsgispy.classification.classsklearn as csk

        clf = csk.load_sklearn_classifier("rf_classifier.joblib")
    """
    rsgispy.check_optional_dep(
        HAVE_JOBLIB, "load_sklearn_classifier", "joblib", "classification"
    )
    in_path = os.fspath(in_file)
    ensure_file_exists(in_path, label="Classifier model")
    return joblib.load(in_path)


def _normalise_search_method(search_method: str) -> str:
    """Normalise hyperparameter search method aliases."""
    key = search_method.strip().lower()
    if key in _GRID_ALIASES:
        return "grid"
    if key in _RANDOM_ALIASES:
        return "random"
    if key in _OPTUNA_ALIASES:
        return "optuna"
    raise rsgispy.RSGISPyException(
        f"Unsupported search_method '{search_method}'. Expected 'grid', "
        "'random', or 'optuna'."
    )


def _build_predefined_split_data(
    train_x: pd.DataFrame,
    train_y: np.ndarray,
    valid_x: pd.DataFrame,
    valid_y: np.ndarray,
) -> Tuple[pd.DataFrame, np.ndarray, Any]:
    """Combine train/valid sets with a 0-leakage ``PredefinedSplit``."""
    comb_x = pd.concat([train_x, valid_x], ignore_index=True)
    comb_y = np.concatenate([train_y, valid_y])
    test_fold = np.concatenate(
        [
            np.full(len(train_y), -1, dtype=int),
            np.zeros(len(valid_y), dtype=int),
        ]
    )
    split = sklearn.model_selection.PredefinedSplit(test_fold=test_fold)
    return comb_x, comb_y, split


def _run_sklearn_search_cv(
    method: str,
    sk_classifier: Any,
    param_space: Any,
    train_x: pd.DataFrame,
    train_y: np.ndarray,
    valid_x: Optional[pd.DataFrame],
    valid_y: Optional[np.ndarray],
    scoring: Union[str, Callable[..., float]],
    cv: Union[int, Any],
    n_iter: int,
    n_jobs: Optional[int],
    rnd_seed: Optional[int],
    refit: bool,
    verbose: int,
) -> Tuple[Any, Dict[str, Any], float, pd.DataFrame, Any]:
    """Execute GridSearchCV, RandomizedSearchCV, or a passed BaseSearchCV."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    if valid_x is not None and valid_y is not None:
        fit_x, fit_y, cv_obj = _build_predefined_split_data(
            train_x, train_y, valid_x, valid_y
        )
        # Never refit PredefinedSplit on combined train+valid; refit on train_x.
        cv_refit = False
    else:
        fit_x, fit_y = train_x, train_y
        cv_obj = _resolve_cv_splitter(cv, train_y, rnd_seed)
        cv_refit = refit

    if is_sklearn_search_cv(sk_classifier):
        search_obj = clone_sklearn_estimator(sk_classifier)
        if valid_x is not None:
            search_obj.set_params(cv=cv_obj, refit=False)
    else:
        if param_space is None or (
            isinstance(param_space, Mapping) and not param_space
        ):
            raise rsgispy.RSGISPyException(
                "param_space must be provided for grid or random search."
            )
        base_est = clone_sklearn_estimator(sk_classifier, random_state=rnd_seed)
        if method == "grid":
            search_obj = sklearn.model_selection.GridSearchCV(
                estimator=base_est,
                param_grid=param_space,
                scoring=scoring,
                cv=cv_obj,
                n_jobs=n_jobs,
                refit=cv_refit,
                verbose=verbose,
            )
        else:
            search_obj = sklearn.model_selection.RandomizedSearchCV(
                estimator=base_est,
                param_distributions=param_space,
                n_iter=int(n_iter),
                scoring=scoring,
                cv=cv_obj,
                n_jobs=n_jobs,
                random_state=rnd_seed,
                refit=cv_refit,
                verbose=verbose,
            )

    search_obj.fit(fit_x, fit_y)
    best_params = to_json_compatible(dict(search_obj.best_params_))
    best_score = float(search_obj.best_score_)
    results_df = pd.DataFrame(search_obj.cv_results_)

    if valid_x is not None or not cv_refit:
        base_for_best = (
            search_obj.estimator
            if hasattr(search_obj, "estimator")
            else sk_classifier
        )
        best_est = clone_sklearn_estimator(
            base_for_best, random_state=rnd_seed
        )
        best_est.set_params(**search_obj.best_params_)
        if refit:
            best_est.fit(train_x, train_y)
    else:
        best_est = search_obj.best_estimator_

    return best_est, best_params, best_score, results_df, search_obj


def _run_optuna_search(
    sk_classifier: Any,
    param_space: Union[Mapping[str, Any], Callable[[Any], Mapping[str, Any]]],
    train_x: pd.DataFrame,
    train_y: np.ndarray,
    valid_x: Optional[pd.DataFrame],
    valid_y: Optional[np.ndarray],
    scoring: Union[str, Callable[..., float]],
    cv: Union[int, Any],
    n_trials: int,
    timeout: Optional[float],
    direction: str,
    n_jobs: Optional[int],
    rnd_seed: Optional[int],
    refit: bool,
    verbose: int,
) -> Tuple[Any, Dict[str, Any], float, pd.DataFrame, Any]:
    """Execute Bayesian hyperparameter optimisation with Optuna."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    rsgispy.check_optional_dep(
        HAVE_OPTUNA,
        "perform_sklearn_classifier_param_search(..., search_method='optuna')",
        "optuna",
        "classification",
    )
    if param_space is None:
        raise rsgispy.RSGISPyException(
            "param_space must be provided when search_method='optuna'."
        )
    if int(n_trials) < 1:
        raise rsgispy.RSGISPyException("n_trials must be at least 1.")

    dir_norm = direction.strip().lower()
    if dir_norm not in ("maximize", "minimize"):
        raise rsgispy.RSGISPyException(
            f"direction must be 'maximize' or 'minimize', got {direction!r}."
        )

    scorer = (
        sklearn.metrics.get_scorer(scoring)
        if isinstance(scoring, str)
        else scoring
    )
    cv_obj = (
        None
        if valid_x is not None
        else _resolve_cv_splitter(cv, train_y, rnd_seed)
    )

    def _objective(trial: Any) -> float:
        params = _suggest_optuna_params(trial, param_space)
        est = clone_sklearn_estimator(sk_classifier, random_state=rnd_seed)
        est.set_params(**params)
        if valid_x is not None and valid_y is not None:
            est.fit(train_x, train_y)
            return float(scorer(est, valid_x, valid_y))
        scores = sklearn.model_selection.cross_val_score(
            est,
            train_x,
            train_y,
            cv=cv_obj,
            scoring=scoring,
            n_jobs=n_jobs,
        )
        return float(np.mean(scores))

    sampler = optuna.samplers.TPESampler(seed=rnd_seed)
    prev_verbosity = optuna.logging.get_verbosity()
    if verbose <= 0:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    try:
        study = optuna.create_study(direction=dir_norm, sampler=sampler)
        study.optimize(
            _objective,
            n_trials=int(n_trials),
            timeout=timeout,
            show_progress_bar=False,
        )
    finally:
        optuna.logging.set_verbosity(prev_verbosity)

    if callable(param_space):
        best_trial_params = dict(study.best_trial.params)
    else:
        best_trial_params = dict(study.best_trial.params)
        for p_name, spec in param_space.items():
            if not isinstance(spec, (Mapping, tuple, list)):
                best_trial_params[p_name] = spec

    best_params = to_json_compatible(best_trial_params)
    best_score = float(study.best_value)
    results_df = study.trials_dataframe()

    best_est = clone_sklearn_estimator(sk_classifier, random_state=rnd_seed)
    best_est.set_params(**best_trial_params)
    if refit:
        best_est.fit(train_x, train_y)

    return best_est, best_params, best_score, results_df, study


def perform_sklearn_classifier_param_search(
    cls_cfg: ClsConfigInput,
    sk_classifier: Any,
    train_data: SamplesInput,
    param_space: Optional[
        Union[
            Mapping[str, Any],
            Sequence[Mapping[str, Any]],
            Callable[[Any], Mapping[str, Any]],
        ]
    ] = None,
    search_method: str = "grid",
    valid_data: Optional[SamplesInput] = None,
    scoring: Union[str, Callable[..., float]] = "accuracy",
    cv: Union[int, Any] = 5,
    n_iter: int = 20,
    n_trials: int = 50,
    timeout: Optional[float] = None,
    direction: str = "maximize",
    n_jobs: Optional[int] = None,
    rnd_seed: Optional[int] = None,
    feature_cols: Optional[Sequence[str]] = None,
    use_internal_id: bool = True,
    refit: bool = True,
    out_id_col: Optional[str] = None,
    internal_id_col: Optional[str] = None,
    out_results_file: Optional[Union[str, os.PathLike]] = None,
    out_params_json: Optional[Union[str, os.PathLike]] = None,
    return_results: bool = False,
    verbose: int = 0,
) -> Union[Any, Tuple[Any, Dict[str, Any]]]:
    """
    Optimise hyperparameters for a ``scikit-learn`` (or compatible) classifier
    using Grid Search, Random Search, or Bayesian optimisation with Optuna.

    When ``valid_data`` is provided, candidate parameter configurations are fit
    on ``train_data`` and evaluated on ``valid_data`` (with the final optimal
    estimator refit only on ``train_data`` when ``refit=True``). When
    ``valid_data`` is ``None``, stratified ``cv``-fold cross-validation is
    performed on ``train_data``.

    For ``search_method='optuna'``, ``param_space`` may be either:

    * A callable ``fn(trial) -> dict`` using ``optuna.Trial`` suggestions; or
    * A dictionary mapping parameter names to lists (categorical), 2-tuples
      ``(low, high)`` (numeric range), or explicit spec dicts such as
      ``{'type': 'float', 'low': 1e-3, 'high': 1.0, 'log': True}``.

    :param cls_cfg: :class:`~rsgispy.classification.classdata.ClassificationConfig`,
                    JSON config path, or mapping of ``{class_name: ClassInfoObj}``.
    :param sk_classifier: Base ``scikit-learn`` classifier estimator (or a
                          pre-configured ``GridSearchCV`` / ``RandomizedSearchCV``).
    :param train_data: Training sample :class:`pandas.DataFrame` or file path
                       (Parquet, CSV, etc.).
    :param param_space: Parameter search space (dict, list of dicts, or Optuna
                        callable). Optional only when ``sk_classifier`` is
                        already a ``BaseSearchCV`` object.
    :param search_method: Optimisation strategy: ``'grid'`` (default),
                          ``'random'``, or ``'optuna'``.
    :param valid_data: Optional validation :class:`pandas.DataFrame` or file path
                       used to score candidates out-of-sample.
    :param scoring: Scikit-learn metric name or scorer callable
                    (Default = ``'accuracy'``).
    :param cv: Cross-validation folds or splitter used when ``valid_data`` is
               ``None`` (Default = ``5``).
    :param n_iter: Number of parameter samples for ``search_method='random'``
                   (Default = ``20``).
    :param n_trials: Number of Optuna trials for ``search_method='optuna'``
                     (Default = ``50``).
    :param timeout: Optional time limit in seconds for Optuna optimisation.
    :param direction: Optuna optimisation direction (``'maximize'`` or
                      ``'minimize'``, Default = ``'maximize'``).
    :param n_jobs: Number of parallel workers for scikit-learn CV evaluations.
    :param rnd_seed: Optional random seed for reproducible search and splits.
    :param feature_cols: Optional ordered feature column names (inferred from
                         ``cls_cfg`` when ``None``).
    :param use_internal_id: If ``True`` (default), train against consecutive
                            0-based ``internal_cls_id`` values; otherwise use
                            ``out_cls_id``.
    :param refit: If ``True`` (default), fit the returned optimal estimator on
                  ``train_data``; if ``False``, return an unfitted clone
                  initialised with the optimal parameters.
    :param out_id_col: Optional override for the output class ID column name.
    :param internal_id_col: Optional override for the internal class ID column
                            name.
    :param out_results_file: Optional path (e.g., ``'search_results.parquet'``
                             or ``'search_results.csv'``) to save the trials /
                             CV results table.
    :param out_params_json: Optional JSON path to save best parameters and score.
    :param return_results: If ``True``, return ``(best_estimator, results_dict)``;
                           if ``False`` (default), return ``best_estimator``.
    :param verbose: Verbosity level (Default = ``0``).
    :return: Optimised classifier estimator, or ``(best_estimator, results_dict)``
             when ``return_results=True``.

    .. code:: python

        from sklearn.ensemble import RandomForestClassifier
        import rsgispy.classification.classsklearn as csk

        best_clf = csk.perform_sklearn_classifier_param_search(
            cls_cfg="cls_config.json",
            sk_classifier=RandomForestClassifier(),
            train_data="train.parquet",
            param_space={"n_estimators": (20, 150), "max_depth": (2, 12)},
            search_method="optuna",
            n_trials=25,
            rnd_seed=42,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    check_sklearn_available("perform_sklearn_classifier_param_search")
    method = _normalise_search_method(search_method)
    cfg, class_info, eff_out_col, eff_int_col, cls_name_col = _coerce_cls_config(
        cls_cfg, out_id_col=out_id_col, internal_id_col=internal_id_col
    )
    target_col = eff_int_col if use_internal_id else eff_out_col
    valid_ids = [
        int(obj.internal_id) if use_internal_id else int(obj.out_id)  # type: ignore[arg-type]
        for obj in class_info.values()
    ]

    train_df = _load_samples_df(train_data, "train_data")
    feat_cols = _resolve_feature_cols(
        cfg,
        train_df,
        feature_cols,
        excluded_cols=[eff_out_col, eff_int_col, cls_name_col],
    )
    train_x, train_y = _extract_xy(
        train_df, feat_cols, target_col, "train_data", valid_ids=valid_ids
    )

    valid_x: Optional[pd.DataFrame] = None
    valid_y: Optional[np.ndarray] = None
    if valid_data is not None:
        valid_df = _load_samples_df(valid_data, "valid_data")
        valid_x, valid_y = _extract_xy(
            valid_df, feat_cols, target_col, "valid_data", valid_ids=valid_ids
        )

    if method == "optuna":
        best_est, best_params, best_score, results_df, search_obj = (
            _run_optuna_search(
                sk_classifier=sk_classifier,
                param_space=param_space,  # type: ignore[arg-type]
                train_x=train_x,
                train_y=train_y,
                valid_x=valid_x,
                valid_y=valid_y,
                scoring=scoring,
                cv=cv,
                n_trials=n_trials,
                timeout=timeout,
                direction=direction,
                n_jobs=n_jobs,
                rnd_seed=rnd_seed,
                refit=refit,
                verbose=verbose,
            )
        )
    else:
        best_est, best_params, best_score, results_df, search_obj = (
            _run_sklearn_search_cv(
                method=method,
                sk_classifier=sk_classifier,
                param_space=param_space,
                train_x=train_x,
                train_y=train_y,
                valid_x=valid_x,
                valid_y=valid_y,
                scoring=scoring,
                cv=cv,
                n_iter=n_iter,
                n_jobs=n_jobs,
                rnd_seed=rnd_seed,
                refit=refit,
                verbose=verbose,
            )
        )

    if out_results_file is not None:
        write_dataframe_table(results_df, out_results_file)

    if out_params_json is not None:
        out_json_path = ensure_parent_dir(out_params_json)
        scoring_name = (
            scoring if isinstance(scoring, str) else getattr(scoring, "__name__", "custom")
        )
        write_dict_to_json(
            {
                "search_method": method,
                "scoring": scoring_name,
                "best_score": float(best_score),
                "best_params": best_params,
            },
            out_json_path,
        )

    if return_results:
        return best_est, {
            "search_method": method,
            "best_params": best_params,
            "best_score": float(best_score),
            "results_df": results_df,
            "search_obj": search_obj,
        }
    return best_est


def train_sklearn_classifier(
    cls_cfg: ClsConfigInput,
    sk_classifier: Any,
    train_data: SamplesInput,
    test_data: Optional[SamplesInput] = None,
    valid_data: Optional[SamplesInput] = None,
    feature_cols: Optional[Sequence[str]] = None,
    use_internal_id: bool = True,
    rnd_seed: Optional[int] = None,
    out_id_col: Optional[str] = None,
    internal_id_col: Optional[str] = None,
    out_model_file: Optional[Union[str, os.PathLike]] = None,
    out_metrics_file: Optional[Union[str, os.PathLike]] = None,
    return_model: bool = False,
) -> Union[Dict[str, float], Tuple[Any, Dict[str, float]]]:
    """
    Train a ``scikit-learn`` (or compatible) classifier on extracted sample
    tables using the class and band configuration in ``cls_cfg``, optionally
    evaluating on testing and validation datasets.

    By default (``return_model=False``), fits a clean clone and updates
    ``sk_classifier`` in place, returning the evaluation metrics dictionary.
    When ``return_model=True``, ``sk_classifier`` is left untouched and
    ``(fitted_model, metrics)`` is returned.

    :param cls_cfg: :class:`~rsgispy.classification.classdata.ClassificationConfig`,
                    JSON config path, or mapping of ``{class_name: ClassInfoObj}``.
    :param sk_classifier: ``scikit-learn`` classifier estimator template.
    :param train_data: Training sample :class:`pandas.DataFrame` or file path.
    :param test_data: Optional testing sample :class:`pandas.DataFrame` or path.
    :param valid_data: Optional validation sample :class:`pandas.DataFrame` or
                       path.
    :param feature_cols: Optional ordered feature column names (inferred from
                         ``cls_cfg`` when ``None``).
    :param use_internal_id: If ``True`` (default), train using consecutive
                            0-based ``internal_cls_id`` labels (recommended for
                            classifiers such as XGBoost); otherwise use
                            ``out_cls_id``.
    :param rnd_seed: Optional random seed passed to the estimator when
                     supported.
    :param out_id_col: Optional override for the output class ID column name.
    :param internal_id_col: Optional override for the internal class ID column
                            name.
    :param out_model_file: Optional output ``.joblib`` path to save the fitted
                           classifier.
    :param out_metrics_file: Optional output JSON path to save training/test/
                             validation metrics.
    :param return_model: When ``True``, return ``(fitted_model, metrics)``
                         without mutating ``sk_classifier``.
    :return: Metrics dictionary (or ``(fitted_model, metrics)`` when
             ``return_model=True``).

    .. code:: python

        from sklearn.ensemble import RandomForestClassifier
        import rsgispy.classification.classsklearn as csk

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        metrics = csk.train_sklearn_classifier(
            cls_cfg="cls_config.json",
            sk_classifier=clf,
            train_data="train.parquet",
            test_data="test.parquet",
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    check_sklearn_available("train_sklearn_classifier")
    cfg, class_info, eff_out_col, eff_int_col, cls_name_col = _coerce_cls_config(
        cls_cfg, out_id_col=out_id_col, internal_id_col=internal_id_col
    )
    target_col = eff_int_col if use_internal_id else eff_out_col
    valid_ids = [
        int(obj.internal_id) if use_internal_id else int(obj.out_id)  # type: ignore[arg-type]
        for obj in class_info.values()
    ]

    train_df = _load_samples_df(train_data, "train_data")
    feat_cols = _resolve_feature_cols(
        cfg,
        train_df,
        feature_cols,
        excluded_cols=[eff_out_col, eff_int_col, cls_name_col],
    )
    train_x, train_y = _extract_xy(
        train_df, feat_cols, target_col, "train_data", valid_ids=valid_ids
    )

    fitted_est = fit_clean_sklearn_estimator(
        sk_classifier,
        train_x,
        train_y,
        sync_input=not return_model,
        random_state=rnd_seed,
    )

    metrics = _calc_split_metrics(fitted_est, train_x, train_y, "train")
    if test_data is not None:
        test_df = _load_samples_df(test_data, "test_data")
        test_x, test_y = _extract_xy(
            test_df, feat_cols, target_col, "test_data", valid_ids=valid_ids
        )
        metrics.update(_calc_split_metrics(fitted_est, test_x, test_y, "test"))
    if valid_data is not None:
        valid_df = _load_samples_df(valid_data, "valid_data")
        valid_x, valid_y = _extract_xy(
            valid_df, feat_cols, target_col, "valid_data", valid_ids=valid_ids
        )
        metrics.update(
            _calc_split_metrics(fitted_est, valid_x, valid_y, "valid")
        )

    if out_model_file is not None:
        save_sklearn_classifier(fitted_est, out_model_file)

    if out_metrics_file is not None:
        metrics_path = ensure_parent_dir(out_metrics_file)
        write_dict_to_json(metrics, metrics_path)

    if return_model:
        return fitted_est, metrics
    return metrics


def apply_sklearn_classifier(
    cls_cfg: ClsConfigInput,
    sk_classifier: Union[Any, str, os.PathLike],
    in_msk_img: Union[str, os.PathLike],
    img_msk_val: int,
    out_class_img: Optional[Union[str, os.PathLike]] = None,
    out_score_img: Optional[Union[str, os.PathLike]] = None,
    img_band_info: Optional[Sequence[ImageBandInfo]] = None,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    out_no_data_val: Optional[int] = None,
    class_clr_names: bool = True,
    use_internal_id: bool = True,
    recode_to_out_ids: bool = True,
    score_max_only: bool = False,
    score_no_data_val: float = -1.0,
    calc_img_stats: bool = True,
    block_size: Optional[int] = None,
    creation_opts: Optional[Mapping[str, Any]] = None,
    use_pbar: bool = True,
) -> None:
    """
    Apply a trained ``scikit-learn`` (or compatible) classifier block-by-block
    to image pixels selected by ``in_msk_img == img_msk_val``, outputting the
    classified class ID raster and/or classification probability score raster.

    When the classifier was trained on consecutive 0-based ``internal_cls_id``
    values (``use_internal_id=True``) and ``recode_to_out_ids=True`` (default),
    predicted class IDs are automatically recoded back to the user-defined
    unsigned ``out_id`` values from ``cls_cfg``.

    :param cls_cfg: :class:`~rsgispy.classification.classdata.ClassificationConfig`,
                    JSON config path, or mapping of ``{class_name: ClassInfoObj}``.
    :param sk_classifier: Fitted classifier instance or path to a saved
                          ``.joblib`` model file.
    :param in_msk_img: Path to the mask image controlling which pixels are
                       classified.
    :param img_msk_val: Pixel value within ``in_msk_img`` identifying pixels to
                        classify.
    :param out_class_img: Optional output classified ID raster path. At least
                          one of ``out_class_img`` or ``out_score_img`` must be
                          provided.
    :param out_score_img: Optional output ``float32`` probability score raster
                          path (requires classifier ``predict_proba``). By
                          default writes one band per class ordered by class ID;
                          set ``score_max_only=True`` for a single-band maximum
                          probability raster.
    :param img_band_info: Optional sequence of
                          :class:`~rsgispy.classification.classdata.ImageBandInfo`
                          overriding ``cls_cfg.img_band_info`` (required when
                          ``cls_cfg`` is a plain class-info mapping).
    :param gdalformat: Output GDAL/rasterio format (Default = ``'GTIFF'``).
    :param datatype: Optional ``rsgispy.TYPE_*`` integer datatype for
                     ``out_class_img``. When ``None`` (default), automatically
                     selects the smallest unsigned integer datatype fitting all
                     output class IDs and nodata.
    :param out_no_data_val: Optional unsigned integer nodata value for
                            ``out_class_img``. When ``None`` (default), selects
                            ``0`` if unused by any output class ID, otherwise the
                            next unused unsigned integer.
    :param class_clr_names: If ``True`` (default), apply class RGB colour table
                            to ``out_class_img`` when output dtype is ``uint8``
                            or ``uint16``.
    :param use_internal_id: Whether ``sk_classifier`` predicts 0-based
                            ``internal_id`` values (Default = ``True``).
    :param recode_to_out_ids: If ``True`` (default) and ``use_internal_id=True``,
                              recode predictions to ``out_id`` values before
                              writing ``out_class_img``.
    :param score_max_only: If ``False`` (default), write one probability band
                           per class to ``out_score_img``; if ``True``, write a
                           single band containing maximum class probability.
    :param score_no_data_val: Nodata fill value for ``out_score_img``
                              (Default = ``-1.0``).
    :param calc_img_stats: If ``True`` (default), compute output statistics and
                           overview pyramids.
    :param block_size: Optional processing block size in pixels.
    :param creation_opts: Optional GDAL creation options mapping.
    :param use_pbar: Display a tqdm progress bar during classification
                     (Default = ``True``).

    .. code:: python

        import rsgispy.classification.classsklearn as csk

        csk.apply_sklearn_classifier(
            cls_cfg="cls_config.json",
            sk_classifier=clf,
            in_msk_img="valid_mask.tif",
            img_msk_val=1,
            out_class_img="classified.tif",
            out_score_img="class_probs.tif",
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    check_sklearn_available("apply_sklearn_classifier")
    estimator = (
        load_sklearn_classifier(sk_classifier)
        if isinstance(sk_classifier, (str, os.PathLike))
        else sk_classifier
    )
    _apply_classifier_raster(
        cls_cfg=cls_cfg,
        estimator=estimator,
        in_msk_img=in_msk_img,
        img_msk_val=img_msk_val,
        out_class_img=out_class_img,
        out_score_img=out_score_img,
        img_band_info=img_band_info,
        gdalformat=gdalformat,
        datatype=datatype,
        out_no_data_val=out_no_data_val,
        class_clr_names=class_clr_names,
        use_internal_id=use_internal_id,
        recode_to_out_ids=recode_to_out_ids,
        score_max_only=score_max_only,
        score_no_data_val=score_no_data_val,
        calc_img_stats=calc_img_stats,
        block_size=block_size,
        creation_opts=creation_opts,
        use_pbar=use_pbar,
        binary=False,
    )
