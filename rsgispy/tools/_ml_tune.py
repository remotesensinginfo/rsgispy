"""
Shared hyperparameter optimisation utilities (Optuna Bayesian search, Grid
Search, Random Search, CV splitters, and PredefinedSplit builders) reused
across :mod:`rsgispy.classification` and :mod:`rsgispy.regression`.
"""

import os
from typing import (
    Any,
    Callable,
    Dict,
    Mapping,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd

import rsgispy
from rsgispy.tools._sklearn import (
    clone_sklearn_estimator,
    is_sklearn_search_cv,
    sklearn,
)
from rsgispy.tools._sklearn_spaces import build_sklearn_param_space
from rsgispy.tools.files import ensure_parent_dir
from rsgispy.tools.jsonio import to_json_compatible, write_dict_to_json
from rsgispy.tools.tables import write_dataframe_table

try:
    import optuna  # pylint: disable=import-error

    HAVE_OPTUNA = True
except ImportError:
    optuna = None  # type: ignore[assignment]
    HAVE_OPTUNA = False

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


def normalise_search_method(search_method: str) -> str:
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


def suggest_optuna_params(
    trial: Any,
    param_space: Union[Mapping[str, Any], Callable[[Any], Mapping[str, Any]]],
) -> Dict[str, Any]:
    """
    Sample hyperparameters for an Optuna trial from either a user-supplied
    ``trial -> dict`` callable or a declarative dictionary specification.
    """
    if callable(param_space):
        suggested = param_space(trial)
        if not isinstance(suggested, Mapping):
            raise rsgispy.RSGISPyException(
                "Callable param_space for Optuna must return a dict mapping "
                "parameter names to values."
            )
        return dict(suggested)

    if not isinstance(param_space, Mapping) or not param_space:
        raise rsgispy.RSGISPyException(
            "param_space must be a non-empty dictionary or callable for Optuna."
        )

    params: Dict[str, Any] = {}
    for p_name, spec in param_space.items():
        if isinstance(spec, Mapping):
            p_type = str(spec.get("type", "")).strip().lower()
            if p_type == "int":
                params[p_name] = trial.suggest_int(
                    p_name,
                    int(spec["low"]),
                    int(spec["high"]),
                    step=int(spec.get("step", 1)),
                    log=bool(spec.get("log", False)),
                )
            elif p_type == "float":
                params[p_name] = trial.suggest_float(
                    p_name,
                    float(spec["low"]),
                    float(spec["high"]),
                    step=spec.get("step"),
                    log=bool(spec.get("log", False)),
                )
            elif p_type == "categorical":
                params[p_name] = trial.suggest_categorical(
                    p_name, list(spec["choices"])
                )
            else:
                raise rsgispy.RSGISPyException(
                    f"Unsupported Optuna param spec type '{p_type}' for "
                    f"'{p_name}'. Expected 'int', 'float', or 'categorical'."
                )
        elif isinstance(spec, tuple):
            if (
                len(spec) == 2
                and isinstance(spec[0], (int, float, np.integer, np.floating))
                and isinstance(spec[1], (int, float, np.integer, np.floating))
                and not isinstance(spec[0], bool)
                and not isinstance(spec[1], bool)
            ):
                if isinstance(spec[0], (int, np.integer)) and isinstance(
                    spec[1], (int, np.integer)
                ):
                    params[p_name] = trial.suggest_int(
                        p_name, int(spec[0]), int(spec[1])
                    )
                else:
                    params[p_name] = trial.suggest_float(
                        p_name, float(spec[0]), float(spec[1])
                    )
            else:
                params[p_name] = trial.suggest_categorical(p_name, list(spec))
        elif isinstance(spec, list):
            params[p_name] = trial.suggest_categorical(p_name, list(spec))
        else:
            params[p_name] = spec
    return params


def resolve_cv_splitter(
    cv: Union[int, Any],
    train_y: np.ndarray,
    rnd_seed: Optional[int],
    stratified: bool = True,
) -> Any:
    """Resolve an integer or sklearn CV splitter for classification or regression."""
    if isinstance(cv, (int, np.integer)) and not isinstance(cv, bool):
        n_splits = int(cv)
        if n_splits < 2:
            raise rsgispy.RSGISPyException(
                f"cv must be at least 2 folds, got {n_splits}."
            )
        if stratified:
            _, counts = np.unique(train_y, return_counts=True)
            min_count = int(counts.min()) if counts.size > 0 else 0
            if min_count < n_splits:
                raise rsgispy.RSGISPyException(
                    f"cv={n_splits} exceeds minimum class sample count "
                    f"({min_count}) in train_data."
                )
            return sklearn.model_selection.StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=rnd_seed,
            )
        n_samples = int(train_y.shape[0])
        if n_samples < n_splits:
            raise rsgispy.RSGISPyException(
                f"cv={n_splits} exceeds sample count ({n_samples}) in train_data."
            )
        return sklearn.model_selection.KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=rnd_seed,
        )
    return cv


def build_predefined_split_data(
    train_x: pd.DataFrame,
    train_y: np.ndarray,
    valid_x: pd.DataFrame,
    valid_y: np.ndarray,
) -> Tuple[pd.DataFrame, np.ndarray, Any]:
    """Combine train/valid sets (1D or 2D targets) with ``PredefinedSplit``."""
    comb_x = pd.concat([train_x, valid_x], ignore_index=True)
    comb_y = np.concatenate([train_y, valid_y], axis=0)
    test_fold = np.concatenate(
        [
            np.full(len(train_y), -1, dtype=int),
            np.zeros(len(valid_y), dtype=int),
        ]
    )
    split = sklearn.model_selection.PredefinedSplit(test_fold=test_fold)
    return comb_x, comb_y, split


def resolve_sklearn_param_space(
    sk_estimator: Any,
    method: str,
    param_space: Optional[Any],
    extended_param_space: bool,
    task: str = "classification",
) -> Tuple[Any, str]:
    """Resolve effective parameter search space and preset mode label."""
    if isinstance(param_space, str):
        preset = param_space.strip().lower()
        if preset in ("default", "key", "standard"):
            if extended_param_space:
                raise rsgispy.RSGISPyException(
                    "Cannot combine param_space='default' with "
                    "extended_param_space=True."
                )
            return (
                build_sklearn_param_space(
                    sk_estimator, method, extended=False, task=task
                ),
                "default",
            )
        if preset in ("extended", "full", "large"):
            return (
                build_sklearn_param_space(
                    sk_estimator, method, extended=True, task=task
                ),
                "extended",
            )
        raise rsgispy.RSGISPyException(
            f"Unsupported param_space preset '{param_space}'. Expected "
            "'default' or 'extended', or pass a dict, list, or callable."
        )

    if param_space is not None:
        if extended_param_space:
            raise rsgispy.RSGISPyException(
                "Provide either a custom param_space or "
                "extended_param_space=True, not both."
            )
        return param_space, "custom"

    if method != "optuna" and is_sklearn_search_cv(sk_estimator):
        return None, "preconfigured"

    use_ext = bool(extended_param_space)
    eff_space = build_sklearn_param_space(
        sk_estimator, method, extended=use_ext, task=task
    )
    return eff_space, ("extended" if use_ext else "default")


def run_optuna_study(
    objective: Callable[[Any], float],
    n_trials: int,
    timeout: Optional[float],
    direction: str,
    rnd_seed: Optional[int],
    verbose: int,
    func_name: str,
    extra: str,
) -> Tuple[Any, str]:
    """Create and execute an Optuna TPE study cleanly."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    rsgispy.check_optional_dep(HAVE_OPTUNA, func_name, "optuna", extra)
    if int(n_trials) < 1:
        raise rsgispy.RSGISPyException("n_trials must be at least 1.")
    dir_norm = direction.strip().lower()
    if dir_norm not in ("maximize", "minimize"):
        raise rsgispy.RSGISPyException(
            f"direction must be 'maximize' or 'minimize', got {direction!r}."
        )

    sampler = optuna.samplers.TPESampler(seed=rnd_seed)
    prev_verbosity = optuna.logging.get_verbosity()
    if verbose <= 0:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    try:
        study = optuna.create_study(direction=dir_norm, sampler=sampler)
        study.optimize(
            objective,
            n_trials=int(n_trials),
            timeout=timeout,
            show_progress_bar=False,
        )
    finally:
        optuna.logging.set_verbosity(prev_verbosity)
    return study, dir_norm


def run_sklearn_search_cv(
    method: str,
    sk_estimator: Any,
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
    stratified: bool = True,
    extra: str = "classification",
) -> Tuple[Any, Dict[str, Any], float, pd.DataFrame, Any]:
    """Execute GridSearchCV, RandomizedSearchCV, or a passed BaseSearchCV."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    if valid_x is not None and valid_y is not None:
        fit_x, fit_y, cv_obj = build_predefined_split_data(
            train_x, train_y, valid_x, valid_y
        )
        cv_refit = False
    else:
        fit_x, fit_y = train_x, train_y
        cv_obj = resolve_cv_splitter(
            cv, train_y, rnd_seed, stratified=stratified
        )
        cv_refit = refit

    if is_sklearn_search_cv(sk_estimator):
        search_obj = clone_sklearn_estimator(sk_estimator, extra=extra)
        if valid_x is not None:
            search_obj.set_params(cv=cv_obj, refit=False)
    else:
        if param_space is None or (
            isinstance(param_space, Mapping) and not param_space
        ):
            raise rsgispy.RSGISPyException(
                "param_space must be provided for grid or random search."
            )
        base_est = clone_sklearn_estimator(
            sk_estimator, random_state=rnd_seed, extra=extra
        )
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
            else sk_estimator
        )
        best_est = clone_sklearn_estimator(
            base_for_best, random_state=rnd_seed, extra=extra
        )
        best_est.set_params(**search_obj.best_params_)
        if refit:
            best_est.fit(train_x, train_y)
    else:
        best_est = search_obj.best_estimator_

    return best_est, best_params, best_score, results_df, search_obj


def run_optuna_search(
    sk_estimator: Any,
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
    stratified: bool = True,
    func_name: str = "perform_sklearn_classifier_param_search",
    extra: str = "classification",
) -> Tuple[Any, Dict[str, Any], float, pd.DataFrame, Any]:
    """Execute Bayesian hyperparameter optimisation with Optuna."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    if param_space is None:
        raise rsgispy.RSGISPyException(
            "param_space must be provided when search_method='optuna'."
        )

    scorer = (
        sklearn.metrics.get_scorer(scoring)
        if isinstance(scoring, str)
        else scoring
    )
    cv_obj = (
        None
        if valid_x is not None
        else resolve_cv_splitter(
            cv, train_y, rnd_seed, stratified=stratified
        )
    )

    def _objective(trial: Any) -> float:
        params = suggest_optuna_params(trial, param_space)
        est = clone_sklearn_estimator(
            sk_estimator, random_state=rnd_seed, extra=extra
        )
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

    study, _ = run_optuna_study(
        objective=_objective,
        n_trials=n_trials,
        timeout=timeout,
        direction=direction,
        rnd_seed=rnd_seed,
        verbose=verbose,
        func_name=f"{func_name}(..., search_method='optuna')",
        extra=extra,
    )

    if callable(param_space):
        best_trial_params = dict(
            param_space(optuna.trial.FixedTrial(study.best_trial.params))
        )
    else:
        best_trial_params = dict(study.best_trial.params)
        for p_name, spec in param_space.items():
            if not isinstance(spec, (Mapping, tuple, list)):
                best_trial_params[p_name] = spec

    best_params = to_json_compatible(best_trial_params)
    best_score = float(study.best_value)
    results_df = study.trials_dataframe()

    best_est = clone_sklearn_estimator(
        sk_estimator, random_state=rnd_seed, extra=extra
    )
    best_est.set_params(**best_trial_params)
    if refit:
        best_est.fit(train_x, train_y)

    return best_est, best_params, best_score, results_df, study


def save_search_outputs(
    results_df: pd.DataFrame,
    best_params: Any,
    best_score: float,
    scoring: Union[str, Callable[..., float]],
    search_method: str,
    space_mode: str,
    out_results_file: Optional[Union[str, os.PathLike]] = None,
    out_params_json: Optional[Union[str, os.PathLike]] = None,
    extra_meta: Optional[Mapping[str, Any]] = None,
) -> None:
    """Persist CV/Optuna results table and best-parameters JSON if requested."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    if out_results_file is not None:
        write_dataframe_table(results_df, out_results_file)

    if out_params_json is not None:
        out_json_path = ensure_parent_dir(out_params_json)
        scoring_name = (
            scoring
            if isinstance(scoring, str)
            else getattr(scoring, "__name__", "custom")
        )
        payload: Dict[str, Any] = {
            "search_method": search_method,
            "param_space_mode": space_mode,
            "scoring": scoring_name,
            "best_score": float(best_score),
            "best_params": to_json_compatible(best_params),
        }
        if extra_meta:
            payload.update(extra_meta)
        write_dict_to_json(payload, out_json_path)
