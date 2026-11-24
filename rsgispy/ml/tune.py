"""
Unified Optuna hyperparameter tuning engine (:func:`tune_estimator`,
:func:`get_default_param_space`, :func:`suggest_params_from_space`) supporting
spatial cross-validation, fold-local :class:`~rsgispy.ml.scaling.FeatureScaler`
scaling, and ``tqdm`` progress reporting across ``scikit-learn``, ``XGBoost``,
``LightGBM``, ``CatBoost``, and species-distribution / custom estimators.
"""

from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Union,
)

import numpy as np
import pandas as pd

import rsgispy
from rsgispy.ml.cv import (
    _evaluate_score,
    _index_rows,
    _iter_cv_splits,
    _validate_coords,
)
from rsgispy.ml.scaling import FeatureScaler
from rsgispy.tools._gbdt_common import (
    get_catboost_optuna_param_space,
    get_lightgbm_optuna_param_space,
    get_xgboost_optuna_param_space,
)

try:
    import optuna  # pylint: disable=import-error

    HAVE_OPTUNA = True
except ImportError:
    optuna = None  # type: ignore[assignment]
    HAVE_OPTUNA = False

__all__ = [
    "TuneResult",
    "get_default_param_space",
    "suggest_params_from_space",
    "tune_estimator",
]


@dataclass
class TuneResult:
    """Results returned by :func:`tune_estimator`."""

    best_params: Dict[str, Any]
    best_score: float
    best_estimator: Optional[Any]
    best_scaler: Optional[FeatureScaler]
    study: Any
    trials_df: pd.DataFrame


def get_default_param_space(
    model_type: str,
    task: str = "classification",
    extended: bool = False,
    binary: bool = True,
) -> Dict[str, Any]:
    """
    Return a sensible built-in Optuna hyperparameter search space for common
    estimators across ``classification``, ``regression``, and ``modelling``.

    :param model_type: One of ``'xgboost'``, ``'lightgbm'``, ``'catboost'``,
                       ``'random_forest'``, ``'extra_trees'``,
                       ``'hist_gb'``, ``'svm'``, ``'mlp'``, or ``'linear'``.
    :param task: ``'classification'`` (Default) or ``'regression'``.
    :param extended: Use broader search space for GBDTs (Default = ``False``).
    :param binary: Binary vs. multiclass GBDT hints (Default = ``True``).
    :return: Dictionary mapping parameter names to search-space definitions.
    """
    key = str(model_type).strip().lower()
    if key in ("xgb", "xgboost"):
        return get_xgboost_optuna_param_space(
            extended=extended, binary=binary, task=task
        )
    if key in ("lgb", "lgbm", "lightgbm"):
        return get_lightgbm_optuna_param_space(extended=extended, task=task)
    if key in ("cb", "catboost"):
        return get_catboost_optuna_param_space(extended=extended, task=task)
    if key in ("rf", "random_forest", "randomforest"):
        return {
            "n_estimators": {"type": "int", "low": 50, "high": 400, "step": 50},
            "max_depth": {"type": "int", "low": 3, "high": 25},
            "min_samples_split": {"type": "int", "low": 2, "high": 20},
            "min_samples_leaf": {"type": "int", "low": 1, "high": 10},
            "max_features": {"type": "categorical", "choices": ["sqrt", "log2", 0.5, 0.8]},
        }
    if key in ("et", "extra_trees", "extratrees"):
        return {
            "n_estimators": {"type": "int", "low": 50, "high": 400, "step": 50},
            "max_depth": {"type": "int", "low": 4, "high": 30},
            "min_samples_leaf": {"type": "int", "low": 1, "high": 10},
            "max_features": {"type": "categorical", "choices": ["sqrt", "log2", 0.5, 0.8]},
        }
    if key in ("hist_gb", "histgradientboosting"):
        return {
            "max_iter": {"type": "int", "low": 50, "high": 400, "step": 25},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
            "max_leaf_nodes": {"type": "int", "low": 15, "high": 127},
            "min_samples_leaf": {"type": "int", "low": 5, "high": 50},
            "l2_regularization": {"type": "float", "low": 1e-4, "high": 10.0, "log": True},
        }
    if key in ("svm", "svc", "svr"):
        return {
            "C": {"type": "float", "low": 1e-2, "high": 1e2, "log": True},
            "gamma": {"type": "categorical", "choices": ["scale", "auto"]},
        }
    if key in ("mlp", "neural_net"):
        return {
            "alpha": {"type": "float", "low": 1e-5, "high": 1e-1, "log": True},
            "learning_rate_init": {"type": "float", "low": 1e-4, "high": 1e-1, "log": True},
        }
    if key in ("linear", "ridge", "logistic"):
        if task == "regression":
            return {"alpha": {"type": "float", "low": 1e-4, "high": 1e3, "log": True}}
        return {"C": {"type": "float", "low": 1e-3, "high": 1e3, "log": True}}

    raise rsgispy.RSGISPyException(
        f"Unknown model_type '{model_type}' for get_default_param_space()."
    )


def suggest_params_from_space(
    trial: Any,
    param_space: Union[Mapping[str, Any], Callable[[Any], Mapping[str, Any]]],
) -> Dict[str, Any]:
    """
    Sample a hyperparameter dictionary from ``param_space`` using an Optuna
    ``trial``.

    Supports both a callable ``(trial) -> dict`` and a declarative dictionary
    whose values are either ``{'type': 'int'|'float'|'categorical', ...}``
    mappings or fixed scalar constants.
    """
    if callable(param_space):
        return dict(param_space(trial))
    if not isinstance(param_space, Mapping):
        raise rsgispy.RSGISPyException(
            "param_space must be a dictionary or callable(trial) -> dict."
        )

    params: Dict[str, Any] = {}
    for name, spec in param_space.items():
        if isinstance(spec, tuple) and len(spec) >= 2 and isinstance(spec[0], str):
            tag = spec[0].strip().lower()
            if tag == "int" and len(spec) >= 3:
                step = int(spec[3]) if len(spec) > 3 else 1
                params[name] = trial.suggest_int(
                    name, int(spec[1]), int(spec[2]), step=step
                )
                continue
            if tag in ("float", "logfloat") and len(spec) >= 3:
                use_log = (tag == "logfloat") or (
                    len(spec) > 3 and bool(spec[3])
                )
                params[name] = trial.suggest_float(
                    name, float(spec[1]), float(spec[2]), log=use_log
                )
                continue
            if tag == "categorical":
                choices = (
                    list(spec[1])
                    if len(spec) == 2 and isinstance(spec[1], (list, tuple))
                    else list(spec[1:])
                )
                params[name] = trial.suggest_categorical(name, choices)
                continue
        if isinstance(spec, Mapping) and "type" in spec:
            p_type = str(spec["type"]).strip().lower()
            if p_type == "int":
                params[name] = trial.suggest_int(
                    name,
                    int(spec["low"]),
                    int(spec["high"]),
                    step=int(spec.get("step", 1)),
                    log=bool(spec.get("log", False)),
                )
            elif p_type == "float":
                step = spec.get("step")
                params[name] = trial.suggest_float(
                    name,
                    float(spec["low"]),
                    float(spec["high"]),
                    step=float(step) if step is not None else None,
                    log=bool(spec.get("log", False)),
                )
            elif p_type == "categorical":
                params[name] = trial.suggest_categorical(
                    name, list(spec["choices"])
                )
            else:
                raise rsgispy.RSGISPyException(
                    f"Unsupported Optuna param type '{p_type}' for '{name}'."
                )
        else:
            params[name] = spec
    return params


def tune_estimator(
    estimator_factory: Callable[[Mapping[str, Any]], Any],
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    param_space: Union[Mapping[str, Any], Callable[[Any], Mapping[str, Any]]],
    *,
    task: str = "classification",
    cv: Any = 5,
    coords: Optional[Any] = None,
    groups: Optional[Any] = None,
    scaler: Optional[FeatureScaler] = None,
    scoring: Optional[Union[str, Callable[[np.ndarray, np.ndarray], float]]] = None,
    n_trials: int = 30,
    timeout: Optional[float] = None,
    direction: str = "maximize",
    pruner: Optional[Any] = None,
    sampler: Optional[Any] = None,
    refit: bool = True,
    rnd_seed: Optional[int] = 42,
    use_pbar: bool = True,
) -> TuneResult:
    """
    Tune hyperparameters for any estimator via Optuna using spatial, group, or
    standard cross-validation with fold-local feature scaling and a ``tqdm``
    progress bar.

    :param estimator_factory: Callable ``(params_dict) -> unfitted_estimator``.
    :param X: Feature matrix or DataFrame of shape ``(n_samples, n_features)``.
    :param y: Target array or Series of length ``n_samples``.
    :param param_space: Optuna search-space dict or callable ``(trial) -> dict``.
    :param task: ``'classification'`` (Default) or ``'regression'``.
    :param cv: Cross-validation splitter (integer, sklearn CV, or
               :class:`~rsgispy.ml.cv.SpatialBlockKFold` /
               :class:`~rsgispy.ml.cv.SpatialBufferCV` /
               :class:`~rsgispy.ml.cv.PolygonGroupKFold`).
    :param coords: Optional ``(n_samples, 2)`` spatial coordinates for spatial CV.
    :param groups: Optional 1D polygon/group IDs for group CV.
    :param scaler: Optional :class:`~rsgispy.ml.scaling.FeatureScaler` cloned
                   and fitted strictly inside each CV training fold.
    :param scoring: Scoring name (``'accuracy'``, ``'macro_f1'``, ``'r2'``,
                    ``'neg_rmse'``, ``'neg_mae'``) or callable ``(y_true, y_pred) -> float``.
    :param n_trials: Number of Optuna trials (Default = ``30``).
    :param timeout: Optional time limit in seconds.
    :param direction: ``'maximize'`` (Default) or ``'minimize'``.
    :param pruner: Optional Optuna pruner instance.
    :param sampler: Optional Optuna sampler instance (defaults to seeded TPESampler).
    :param refit: Refit best estimator (and ``best_scaler`` if ``scaler`` is
                  provided) on the full dataset ``(X, y)`` (Default = ``True``).
    :param rnd_seed: Optional seed for the default TPESampler (Default = ``42``).
    :param use_pbar: Display Optuna ``tqdm`` progress bar (Default = ``True``).
    :return: :class:`TuneResult`.
    """
    # pylint: disable=too-many-arguments,too-many-locals
    rsgispy.check_optional_dep(HAVE_OPTUNA, "tune_estimator", "optuna", "classification")
    task_norm = str(task).strip().lower()
    if task_norm not in ("classification", "regression"):
        raise rsgispy.RSGISPyException(
            f"task must be 'classification' or 'regression', got {task!r}."
        )
    if int(n_trials) < 1:
        raise rsgispy.RSGISPyException(f"n_trials must be >= 1, got {n_trials}.")

    y_arr = np.asarray(y)
    n_samples = y_arr.shape[0]
    c_arr = _validate_coords(coords, n_samples) if coords is not None else None
    g_arr = np.asarray(groups) if groups is not None else None
    splits = _iter_cv_splits(cv, X, y_arr, c_arr, g_arr)
    if not splits:
        raise rsgispy.RSGISPyException("cv produced zero valid splits.")

    prev_verbosity = optuna.logging.get_verbosity()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    try:
        eff_sampler = (
            sampler
            if sampler is not None
            else optuna.samplers.TPESampler(seed=rnd_seed)
        )
        study = optuna.create_study(
            direction=direction,
            sampler=eff_sampler,
            pruner=pruner,
        )

        def _objective(trial: Any) -> float:
            params = suggest_params_from_space(trial, param_space)
            scores: List[float] = []
            for step_idx, (tr_idx, val_idx) in enumerate(splits):
                x_tr = _index_rows(X, tr_idx)
                y_tr = y_arr[tr_idx]
                x_val = _index_rows(X, val_idx)
                y_val = y_arr[val_idx]

                if scaler is not None:
                    fold_scaler = scaler.clone_unfitted()
                    x_tr = fold_scaler.fit_transform(x_tr)
                    x_val = fold_scaler.transform(x_val)

                est = estimator_factory(params)
                est.fit(x_tr, y_tr)
                preds = np.asarray(est.predict(x_val)).ravel()
                sc = _evaluate_score(y_val, preds, task_norm, scoring)
                scores.append(sc)
                trial.report(float(np.mean(scores)), step=step_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            return float(np.mean(scores))

        study.optimize(
            _objective,
            n_trials=int(n_trials),
            timeout=timeout,
            show_progress_bar=bool(use_pbar),
        )
    finally:
        optuna.logging.set_verbosity(prev_verbosity)

    fixed_trial = optuna.trial.FixedTrial(study.best_trial.params)
    best_params = dict(suggest_params_from_space(fixed_trial, param_space))
    best_score = float(study.best_value)

    best_est: Optional[Any] = None
    best_scaler: Optional[FeatureScaler] = None
    if refit:
        x_fit: Any = X
        if scaler is not None:
            best_scaler = scaler.clone_unfitted()
            x_fit = best_scaler.fit_transform(X)
        best_est = estimator_factory(best_params)
        best_est.fit(x_fit, y_arr)

    records = [
        {
            "number": int(t.number),
            "value": float(t.value) if t.value is not None else np.nan,
            "state": str(t.state.name),
            **dict(t.params),
        }
        for t in study.trials
    ]
    trials_df = pd.DataFrame(records)

    return TuneResult(
        best_params=best_params,
        best_score=best_score,
        best_estimator=best_est,
        best_scaler=best_scaler,
        study=study,
        trials_df=trials_df,
    )
