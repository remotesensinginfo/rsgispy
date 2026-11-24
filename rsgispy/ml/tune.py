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
from rsgispy.tools._ml_tune import (
    HAVE_OPTUNA,
    extract_best_optuna_params,
    optuna,
    run_optuna_study,
    suggest_optuna_params,
)
from rsgispy.tools._sklearn_spaces import build_sklearn_param_space

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
) -> Union[Dict[str, Any], Callable[[Any], Mapping[str, Any]]]:
    """
    Return a sensible built-in Optuna hyperparameter search space for common
    estimators across ``classification``, ``regression``, and ``modelling``.

    Delegates to :mod:`rsgispy.tools._gbdt_common` for GBDT backends and
    :mod:`rsgispy.tools._sklearn_spaces` for ``scikit-learn`` estimators.

    :param model_type: Estimator name or alias such as ``'xgboost'``,
                       ``'lightgbm'``, ``'catboost'``, ``'rf'``, ``'et'``,
                       ``'hist_gb'``, ``'svm'``, ``'mlp'``, ``'neural_net'``,
                       or ``'linear'``.
    :param task: ``'classification'`` (Default) or ``'regression'``.
    :param extended: Use broader search space (Default = ``False``).
    :param binary: Binary vs. multiclass GBDT hints (Default = ``True``).
    :return: Dictionary or Optuna ``fn(trial) -> dict`` callable mapping
             parameter names to search-space definitions.
    """
    task_norm = str(task).strip().lower()
    if task_norm not in ("classification", "regression"):
        raise rsgispy.RSGISPyException(
            f"task must be 'classification' or 'regression', got {task!r}."
        )
    key = str(model_type).strip().lower().replace("_", "").replace("-", "").replace(" ", "")
    if key in ("xgb", "xgboost"):
        return get_xgboost_optuna_param_space(
            extended=extended, binary=binary, task=task_norm
        )
    if key in ("lgb", "lgbm", "lightgbm"):
        return get_lightgbm_optuna_param_space(
            extended=extended, task=task_norm
        )
    if key in ("cb", "catboost"):
        return get_catboost_optuna_param_space(
            extended=extended, task=task_norm
        )
    return build_sklearn_param_space(
        sk_estimator=model_type,
        method="optuna",
        extended=bool(extended),
        task=task_norm,
    )


def suggest_params_from_space(
    trial: Any,
    param_space: Union[Mapping[str, Any], Callable[[Any], Mapping[str, Any]]],
) -> Dict[str, Any]:
    """
    Sample a hyperparameter dictionary from ``param_space`` using an Optuna
    ``trial``.

    Supports a callable ``(trial) -> dict`` or a declarative dictionary whose
    values are tagged tuples (``('int', ...)``, ``('float', ...)``,
    ``('logfloat', ...)``, ``('categorical', ...)``), 2-tuple numeric ranges,
    categorical lists, ``{'type': 'int'|'float'|'categorical', ...}`` dicts,
    or fixed scalar constants.
    """
    return suggest_optuna_params(trial, param_space, allow_empty=True)


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

    study, _ = run_optuna_study(
        objective=_objective,
        n_trials=int(n_trials),
        timeout=timeout,
        direction=direction,
        rnd_seed=rnd_seed,
        verbose=0,
        func_name="tune_estimator",
        extra="classification",
        use_pbar=use_pbar,
        sampler=sampler,
        pruner=pruner,
    )

    best_params = extract_best_optuna_params(study, param_space)
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
