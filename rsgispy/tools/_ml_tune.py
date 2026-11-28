"""
Shared hyperparameter optimisation utilities (Optuna Bayesian search, Grid
Search, Random Search, CV splitters, and PredefinedSplit builders) reused
across :mod:`rsgispy.classification` and :mod:`rsgispy.regression`.
"""

import inspect
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
    *,
    allow_empty: bool = False,
) -> Dict[str, Any]:
    """
    Sample hyperparameters for an Optuna trial from either a user-supplied
    ``trial -> dict`` callable or a declarative dictionary specification.

    Supported dictionary value specifications:

    * Tagged tuples: ``('int', low, high[, step])``,
      ``('float', low, high[, log])``, ``('logfloat', low, high)``, and
      ``('categorical', choices)`` (or ``('categorical', c1, c2, ...)``).
    * Explicit spec mappings: ``{'type': 'int'|'float'|'categorical', ...}``.
    * Numeric 2-tuples ``(low, high)`` for uniform integer or float ranges.
    * Lists or other tuples for categorical choices.
    * Fixed scalar constants passed through unchanged.
    """
    # pylint: disable=too-many-branches
    if callable(param_space):
        suggested = param_space(trial)
        if not isinstance(suggested, Mapping):
            raise rsgispy.RSGISPyException(
                "Callable param_space for Optuna must return a dict mapping "
                "parameter names to values."
            )
        return dict(suggested)

    if not isinstance(param_space, Mapping) or (
        not allow_empty and not param_space
    ):
        raise rsgispy.RSGISPyException(
            "param_space must be a non-empty dictionary or callable for Optuna."
        )

    params: Dict[str, Any] = {}
    for p_name, spec in param_space.items():
        if isinstance(spec, tuple) and len(spec) >= 2 and isinstance(spec[0], str):
            tag = spec[0].strip().lower()
            if tag == "int" and len(spec) >= 3:
                step = int(spec[3]) if len(spec) > 3 else 1
                params[p_name] = trial.suggest_int(
                    p_name, int(spec[1]), int(spec[2]), step=step
                )
                continue
            if tag in ("float", "logfloat") and len(spec) >= 3:
                use_log = (tag == "logfloat") or (
                    len(spec) > 3 and bool(spec[3])
                )
                params[p_name] = trial.suggest_float(
                    p_name, float(spec[1]), float(spec[2]), log=use_log
                )
                continue
            if tag == "categorical":
                choices = (
                    list(spec[1])
                    if len(spec) == 2 and isinstance(spec[1], (list, tuple))
                    else list(spec[1:])
                )
                params[p_name] = trial.suggest_categorical(p_name, choices)
                continue

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
                step_val = spec.get("step")
                params[p_name] = trial.suggest_float(
                    p_name,
                    float(spec["low"]),
                    float(spec["high"]),
                    step=float(step_val) if step_val is not None else None,
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


def extract_best_optuna_params(
    study: Any,
    param_space: Union[Mapping[str, Any], Callable[[Any], Mapping[str, Any]]],
) -> Dict[str, Any]:
    """
    Reconstruct the complete best hyperparameter dictionary (including fixed
    constants and conditional parameters) from a completed Optuna ``study``.
    """
    fixed_trial = optuna.trial.FixedTrial(study.best_trial.params)
    if callable(param_space):
        return dict(suggest_optuna_params(fixed_trial, param_space))
    if not param_space:
        return dict(study.best_trial.params)
    return dict(
        suggest_optuna_params(fixed_trial, param_space, allow_empty=True)
    )


def resolve_cv_splitter(
    cv: Union[int, Any],
    train_y: np.ndarray,
    rnd_seed: Optional[int],
    stratified: bool = True,
) -> Any:
    """
    Resolve an integer or sklearn CV splitter for classification or
    regression. An integer gives a :class:`rsgispy.ml.cv.ShuffledKFold`
    (stratified when ``stratified=True``), the K-fold implementation shared
    with :mod:`rsgispy.ml.cv`.
    """
    if isinstance(cv, (int, np.integer)) and not isinstance(cv, bool):
        # pylint: disable=import-outside-toplevel
        from rsgispy.ml.cv import ShuffledKFold  # rsgispy.ml imports this module

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
            return ShuffledKFold(n_splits, stratified=True, rnd_seed=rnd_seed)
        n_samples = int(train_y.shape[0])
        if n_samples < n_splits:
            raise rsgispy.RSGISPyException(
                f"cv={n_splits} exceeds sample count ({n_samples}) in train_data."
            )
        return ShuffledKFold(n_splits, stratified=False, rnd_seed=rnd_seed)
    return cv


def resolve_cv_groups_coords(
    samples_df: pd.DataFrame,
    groups: Optional[Any] = None,
    coords: Optional[Any] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Resolve the ``groups`` (a column name of ``samples_df`` or one label per
    row) and ``coords`` (two column names, e.g. ``('x', 'y')``, or an
    ``(n_samples, 2)`` array) arguments of the parameter-search functions.
    """
    n_rows = len(samples_df)
    grp_arr: Optional[np.ndarray] = None
    if groups is not None:
        if isinstance(groups, str):
            if groups not in samples_df.columns:
                raise rsgispy.RSGISPyException(
                    f"groups column '{groups}' was not found in train_data "
                    f"columns {list(samples_df.columns)}."
                )
            grp_arr = samples_df[groups].to_numpy()
        else:
            grp_arr = np.asarray(groups)
        if grp_arr.ndim != 1 or grp_arr.shape[0] != n_rows:
            raise rsgispy.RSGISPyException(
                f"groups must give one label for each of the {n_rows} "
                f"training samples, got shape {grp_arr.shape}."
            )
        if pd.isna(grp_arr).any():
            raise rsgispy.RSGISPyException("groups contains missing values.")

    xy_arr: Optional[np.ndarray] = None
    if coords is not None:
        if (
            isinstance(coords, (list, tuple))
            and coords
            and all(isinstance(col, str) for col in coords)
        ):
            if len(coords) != 2:
                raise rsgispy.RSGISPyException(
                    "coords must name two columns (x, y), got "
                    f"{list(coords)}."
                )
            missing = [col for col in coords if col not in samples_df.columns]
            if missing:
                raise rsgispy.RSGISPyException(
                    f"coords column(s) {missing} not found in train_data "
                    f"columns {list(samples_df.columns)}."
                )
            xy_arr = samples_df[list(coords)].to_numpy(dtype=np.float64)
        else:
            xy_arr = np.asarray(coords, dtype=np.float64)
        if xy_arr.ndim != 2 or xy_arr.shape != (n_rows, 2):
            raise rsgispy.RSGISPyException(
                f"coords must have shape ({n_rows}, 2), got {xy_arr.shape}."
            )
    return grp_arr, xy_arr


def _resolve_spatial_group_cv(
    cv: Union[int, Any],
    rnd_seed: Optional[int],
    stratified: bool,
    groups: Optional[np.ndarray],
    coords: Optional[np.ndarray],
) -> Any:
    """
    Check that an :mod:`rsgispy.ml.cv` splitter has the inputs it needs and,
    with ``groups`` or ``coords``, turn an integer ``cv`` into a
    ``SpatialBlockKFold`` / ``PolygonGroupKFold``.
    """
    # pylint: disable=import-outside-toplevel
    from rsgispy.ml.cv import PolygonGroupKFold, SpatialBlockKFold, SpatialBufferCV

    if isinstance(cv, (SpatialBlockKFold, SpatialBufferCV)) and coords is None:
        raise rsgispy.RSGISPyException(
            f"cv={type(cv).__name__} needs the sample coordinates: pass "
            "coords=('x', 'y') (the coordinate columns of train_data) or an "
            "(n_samples, 2) array."
        )
    if isinstance(cv, PolygonGroupKFold) and groups is None:
        raise rsgispy.RSGISPyException(
            "cv=PolygonGroupKFold needs the group (e.g. source polygon) of "
            "each sample: pass groups='train_grp' (a column of train_data) "
            "or an array of labels."
        )
    is_int = isinstance(cv, (int, np.integer)) and not isinstance(cv, bool)
    if not is_int or (groups is None and coords is None):
        return cv
    if int(cv) < 2:
        raise rsgispy.RSGISPyException(f"cv must be at least 2 folds, got {cv}.")
    if coords is not None:
        return SpatialBlockKFold(
            n_splits=int(cv), stratify=stratified, rnd_seed=rnd_seed
        )
    return PolygonGroupKFold(
        n_splits=int(cv), stratify=stratified, rnd_seed=rnd_seed
    )


def resolve_cv_splits(
    cv: Union[int, Any],
    train_x: Any,
    train_y: np.ndarray,
    rnd_seed: Optional[int],
    stratified: bool = True,
    groups: Optional[np.ndarray] = None,
    coords: Optional[np.ndarray] = None,
) -> Any:
    """
    Resolve ``cv`` for a parameter search. Without ``groups`` and ``coords``
    this is :func:`resolve_cv_splitter`. Otherwise the folds are made here
    (``cv.split(X, y, groups=groups, coords=coords)`` for the
    :mod:`rsgispy.ml.cv` splitters, ``cv.split(X, y, groups)`` for
    scikit-learn group splitters such as ``GroupKFold``) and returned as a
    list of ``(train_idx, test_idx)`` arrays, which every scikit-learn search
    accepts. An integer ``cv`` gives a
    :class:`~rsgispy.ml.cv.SpatialBlockKFold` (with ``coords``) or a
    :class:`~rsgispy.ml.cv.PolygonGroupKFold` (with ``groups``), stratified by
    class for classification.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    cv = _resolve_spatial_group_cv(cv, rnd_seed, stratified, groups, coords)
    if groups is None and coords is None:
        return resolve_cv_splitter(cv, train_y, rnd_seed, stratified=stratified)
    if not hasattr(cv, "split"):
        return cv  # already an iterable of (train_idx, test_idx) arrays

    try:
        split_params = inspect.signature(cv.split).parameters
    except (TypeError, ValueError):
        split_params = {}
    if "coords" in split_params:
        raw_splits = cv.split(train_x, train_y, groups=groups, coords=coords)
    elif groups is None:
        raise rsgispy.RSGISPyException(
            f"cv={type(cv).__name__} does not accept coords; use a "
            "rsgispy.ml.cv splitter (e.g. SpatialBlockKFold) or pass groups."
        )
    else:
        raw_splits = cv.split(train_x, train_y, groups)
    splits = [
        (np.asarray(tr_idx, dtype=np.intp), np.asarray(te_idx, dtype=np.intp))
        for tr_idx, te_idx in raw_splits
        if len(tr_idx) > 0 and len(te_idx) > 0
    ]
    if not splits:
        raise rsgispy.RSGISPyException(
            f"cv={type(cv).__name__} produced no folds with both training "
            "and validation samples."
        )
    return splits


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


def check_sklearn_scorer_direction(
    scoring: Union[str, Callable[..., float]],
    direction: str,
) -> None:
    """
    Raise when ``direction='minimize'`` is used with a named scikit-learn
    scorer. Named scorers are greater-is-better (error metrics are exposed as
    ``'neg_*'``), so minimising one makes Optuna select the worst model. A
    callable ``scoring`` may be optimised in either direction.
    """
    if isinstance(scoring, str) and str(direction).strip().lower() == "minimize":
        raise rsgispy.RSGISPyException(
            f"scoring={scoring!r} is a scikit-learn scorer, which is "
            "greater-is-better, so it must be used with direction='maximize'. "
            "For error metrics use the 'neg_*' scorer names (e.g. "
            "'neg_root_mean_squared_error') with direction='maximize', or "
            "pass a callable scorer to minimise."
        )


def run_optuna_study(
    objective: Callable[[Any], float],
    n_trials: int,
    timeout: Optional[float],
    direction: str,
    rnd_seed: Optional[int],
    verbose: int,
    func_name: str,
    extra: str,
    use_pbar: bool = True,
    sampler: Optional[Any] = None,
    pruner: Optional[Any] = None,
) -> Tuple[Any, str]:
    """Create and execute an Optuna study cleanly, restoring logging verbosity."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    rsgispy.check_optional_dep(HAVE_OPTUNA, func_name, "optuna", extra)
    if int(n_trials) < 1:
        raise rsgispy.RSGISPyException("n_trials must be at least 1.")
    dir_norm = direction.strip().lower()
    if dir_norm not in ("maximize", "minimize"):
        raise rsgispy.RSGISPyException(
            f"direction must be 'maximize' or 'minimize', got {direction!r}."
        )

    eff_sampler = (
        sampler
        if sampler is not None
        else optuna.samplers.TPESampler(seed=rnd_seed)
    )
    prev_verbosity = optuna.logging.get_verbosity()
    if verbose <= 0:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    try:
        study = optuna.create_study(
            direction=dir_norm,
            sampler=eff_sampler,
            pruner=pruner,
        )
        study.optimize(
            objective,
            n_trials=int(n_trials),
            timeout=timeout,
            show_progress_bar=bool(use_pbar),
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
    groups: Optional[np.ndarray] = None,
    coords: Optional[np.ndarray] = None,
) -> Tuple[Any, Dict[str, Any], float, pd.DataFrame, Any]:
    """
    Execute GridSearchCV, RandomizedSearchCV, or a passed BaseSearchCV. The
    optional ``groups`` / ``coords`` of the training samples are used to make
    the cross-validation folds (see :func:`resolve_cv_splits`).
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    preconfigured = is_sklearn_search_cv(sk_estimator)
    use_grp = groups is not None or coords is not None
    if valid_x is not None and valid_y is not None:
        fit_x, fit_y, cv_obj = build_predefined_split_data(
            train_x, train_y, valid_x, valid_y
        )
        cv_refit = False
    else:
        fit_x, fit_y = train_x, train_y
        cv_obj = (
            None
            if preconfigured and use_grp
            else resolve_cv_splits(
                cv, train_x, train_y, rnd_seed, stratified=stratified,
                groups=groups, coords=coords,
            )
        )
        cv_refit = refit

    if preconfigured:
        search_obj = clone_sklearn_estimator(sk_estimator, extra=extra)
        if valid_x is not None:
            search_obj.set_params(cv=cv_obj, refit=False)
        elif use_grp:
            search_obj.set_params(
                cv=resolve_cv_splits(
                    search_obj.cv, train_x, train_y, rnd_seed,
                    stratified=stratified, groups=groups, coords=coords,
                )
            )
    else:
        if param_space is None or (
            isinstance(param_space, Mapping) and not param_space
        ):
            raise rsgispy.RSGISPyException(
                "param_space must be provided for grid or random search."
            )
        base_est = clone_sklearn_estimator(
            sk_estimator, rnd_seed=rnd_seed, extra=extra
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
            base_for_best, rnd_seed=rnd_seed, extra=extra
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
    use_pbar: bool = True,
    groups: Optional[np.ndarray] = None,
    coords: Optional[np.ndarray] = None,
) -> Tuple[Any, Dict[str, Any], float, pd.DataFrame, Any]:
    """
    Execute Bayesian hyperparameter optimisation with Optuna. The optional
    ``groups`` / ``coords`` of the training samples are used to make the
    cross-validation folds (see :func:`resolve_cv_splits`).
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    if param_space is None:
        raise rsgispy.RSGISPyException(
            "param_space must be provided when search_method='optuna'."
        )
    check_sklearn_scorer_direction(scoring, direction)

    scorer = (
        sklearn.metrics.get_scorer(scoring)
        if isinstance(scoring, str)
        else scoring
    )
    cv_obj = (
        None
        if valid_x is not None
        else resolve_cv_splits(
            cv, train_x, train_y, rnd_seed, stratified=stratified,
            groups=groups, coords=coords,
        )
    )

    def _objective(trial: Any) -> float:
        params = suggest_optuna_params(trial, param_space)
        est = clone_sklearn_estimator(
            sk_estimator, rnd_seed=rnd_seed, extra=extra
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
        use_pbar=use_pbar,
    )

    best_trial_params = extract_best_optuna_params(study, param_space)

    best_params = to_json_compatible(best_trial_params)
    best_score = float(study.best_value)
    results_df = study.trials_dataframe()

    best_est = clone_sklearn_estimator(
        sk_estimator, rnd_seed=rnd_seed, extra=extra
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
