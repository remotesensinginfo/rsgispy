"""
Shared Gradient Boosted Decision Tree (GBDT) engine supporting Optuna
hyperparameter optimisation, binary and multi-class training with optional
validation early stopping, model persistence, and block-wise raster
classification for XGBoost, LightGBM, and CatBoost.
"""

from dataclasses import dataclass
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
    apply_classifier_raster,
    calc_cls_split_metrics,
    prepare_cls_splits,
    resolve_binary_classes,
    suggest_optuna_params,
)
from rsgispy.classification._helpers import (
    ClsConfigInput,
    SamplesInput,
    check_sklearn_available,
    resolve_cv_splitter,
)
from rsgispy.classification.classdata import ImageBandInfo
from rsgispy.tools._sklearn import sklearn, sync_sklearn_estimator_state
from rsgispy.tools.files import ensure_parent_dir
from rsgispy.tools.jsonio import to_json_compatible, write_dict_to_json
from rsgispy.tools.tables import write_dataframe_table

try:
    import optuna  # pylint: disable=import-error

    HAVE_OPTUNA = True
except ImportError:
    optuna = None  # type: ignore[assignment]
    HAVE_OPTUNA = False


_OptunaSpaceType = Union[Dict[str, Any], Callable[[Any], Mapping[str, Any]]]


@dataclass(frozen=True)
class GBDTBackendSpec:
    """Backend descriptor abstracting XGBoost, LightGBM, and CatBoost."""

    # pylint: disable=too-many-instance-attributes

    name: str
    package_name: str
    extra: str
    is_available: Callable[[], bool]
    seed_param: str
    jobs_param: str
    default_binary_params: Callable[[int], Dict[str, Any]]
    default_multiclass_params: Callable[[int, int], Dict[str, Any]]
    default_binary_space: Callable[[], _OptunaSpaceType]
    default_multiclass_space: Callable[[], _OptunaSpaceType]
    extended_binary_space: Callable[[], _OptunaSpaceType]
    extended_multiclass_space: Callable[[], _OptunaSpaceType]
    build_estimator: Callable[[Dict[str, Any], Optional[Any]], Any]
    fit_estimator: Callable[
        [
            Any,
            pd.DataFrame,
            np.ndarray,
            Optional[pd.DataFrame],
            Optional[np.ndarray],
            Optional[int],
        ],
        Any,
    ]
    load_model: Callable[[Union[str, os.PathLike]], Any]
    save_model: Callable[[Any, Union[str, os.PathLike]], None]
    require_consecutive_zero_labels: bool = False


def _validate_target_labels(
    backend: GBDTBackendSpec,
    train_y: np.ndarray,
    n_classes: int,
    binary: bool,
) -> None:
    """Validate label encoding requirements for the selected GBDT backend."""
    if not backend.require_consecutive_zero_labels:
        return
    expected = {0, 1} if binary else set(range(n_classes))
    observed = {int(v) for v in np.unique(train_y).tolist()}
    if not observed.issubset(expected):
        raise rsgispy.RSGISPyException(
            f"{backend.name} requires consecutive 0-based integer class labels "
            f"{sorted(expected)}, got {sorted(observed)}. Set "
            "use_internal_id=True to train on internal_cls_id values."
        )


def _merge_gbdt_params(
    backend: GBDTBackendSpec,
    n_classes: int,
    binary: bool,
    base_classifier: Optional[Any],
    explicit_params: Optional[Mapping[str, Any]],
    rnd_seed: Optional[int],
    n_jobs: Optional[int],
    early_stopping_rounds: Optional[int],
    has_valid: bool,
    verbose: int,
) -> Dict[str, Any]:
    """Build clean estimator parameter dictionary with task defaults."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    task_defaults = (
        backend.default_binary_params(verbose)
        if binary
        else backend.default_multiclass_params(n_classes, verbose)
    )
    merged: Dict[str, Any] = dict(task_defaults)

    if base_classifier is not None and hasattr(base_classifier, "get_params"):
        base_params = {
            k: v
            for k, v in base_classifier.get_params(deep=False).items()
            if v is not None
        }
        merged.update(base_params)
        for k, v in task_defaults.items():
            if k not in base_params or base_params[k] is None:
                merged[k] = v

    if explicit_params:
        merged.update(dict(explicit_params))

    if rnd_seed is not None:
        if backend.seed_param == "random_seed":
            merged.pop("random_state", None)
        merged[backend.seed_param] = int(rnd_seed)

    if n_jobs is not None:
        if backend.jobs_param == "thread_count":
            merged.pop("n_jobs", None)
        merged[backend.jobs_param] = int(n_jobs)

    if not has_valid:
        merged.pop("early_stopping_rounds", None)
    elif early_stopping_rounds is not None and backend.name == "XGBoost":
        merged["early_stopping_rounds"] = int(early_stopping_rounds)

    return merged


def _build_and_fit_gbdt(
    backend: GBDTBackendSpec,
    params: Dict[str, Any],
    base_classifier: Optional[Any],
    train_x: pd.DataFrame,
    train_y: np.ndarray,
    valid_x: Optional[pd.DataFrame],
    valid_y: Optional[np.ndarray],
    early_stopping_rounds: Optional[int],
) -> Any:
    """Instantiate a clean GBDT estimator and fit on training data."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    est = backend.build_estimator(params, base_classifier)
    eff_early = early_stopping_rounds if valid_x is not None else None
    return backend.fit_estimator(est, train_x, train_y, valid_x, valid_y, eff_early)


def _resolve_eff_optuna_space(
    backend: GBDTBackendSpec,
    binary: bool,
    param_space: Optional[
        Union[str, Mapping[str, Any], Callable[[Any], Mapping[str, Any]]]
    ],
    extended_param_space: bool,
    fixed_params: Optional[Mapping[str, Any]],
) -> Tuple[_OptunaSpaceType, str]:
    """Resolve effective Optuna trial search space and mode name."""
    use_extended = bool(extended_param_space)
    custom_space: Optional[Any] = None

    if isinstance(param_space, str):
        preset = param_space.strip().lower()
        if preset in ("default", "key", "standard"):
            if extended_param_space:
                raise rsgispy.RSGISPyException(
                    "Cannot combine param_space='default' with "
                    "extended_param_space=True."
                )
            use_extended = False
        elif preset in ("extended", "full", "large"):
            use_extended = True
        else:
            raise rsgispy.RSGISPyException(
                f"Unsupported param_space preset '{param_space}'. Expected "
                "'default' or 'extended', or pass a dict or callable."
            )
    elif param_space is not None:
        if extended_param_space:
            raise rsgispy.RSGISPyException(
                "Provide either a custom param_space or "
                "extended_param_space=True, not both."
            )
        custom_space = param_space

    if custom_space is not None:
        eff_space: _OptunaSpaceType = custom_space
        space_mode = "custom"
    elif use_extended:
        eff_space = (
            backend.extended_binary_space()
            if binary
            else backend.extended_multiclass_space()
        )
        space_mode = "extended"
    else:
        eff_space = (
            backend.default_binary_space()
            if binary
            else backend.default_multiclass_space()
        )
        space_mode = "default"

    if not callable(eff_space) and fixed_params:
        trial_space: _OptunaSpaceType = {
            k: v for k, v in eff_space.items() if k not in fixed_params
        }
    else:
        trial_space = eff_space

    return trial_space, space_mode


def optimise_gbdt_classifier(
    backend: GBDTBackendSpec,
    func_name: str,
    cls_cfg: ClsConfigInput,
    train_data: SamplesInput,
    binary: bool,
    valid_data: Optional[SamplesInput] = None,
    param_space: Optional[
        Union[str, Mapping[str, Any], Callable[[Any], Mapping[str, Any]]]
    ] = None,
    extended_param_space: bool = False,
    fixed_params: Optional[Mapping[str, Any]] = None,
    classifier: Optional[Any] = None,
    scoring: Union[str, Callable[..., float]] = "accuracy",
    cv: Union[int, Any] = 5,
    n_trials: int = 50,
    timeout: Optional[float] = None,
    direction: str = "maximize",
    early_stopping_rounds: Optional[int] = None,
    n_jobs: Optional[int] = None,
    rnd_seed: Optional[int] = None,
    feature_cols: Optional[Sequence[str]] = None,
    use_internal_id: bool = True,
    refit: bool = True,
    out_id_col: Optional[str] = None,
    internal_id_col: Optional[str] = None,
    out_results_file: Optional[Union[str, os.PathLike]] = None,
    out_params_json: Optional[Union[str, os.PathLike]] = None,
    out_model_file: Optional[Union[str, os.PathLike]] = None,
    return_results: bool = False,
    verbose: int = 0,
) -> Union[Any, Tuple[Any, Dict[str, Any]]]:
    """
    Optimise hyperparameters for a GBDT binary or multi-class classifier using
    Optuna Bayesian optimisation (TPE sampler).
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    rsgispy.check_optional_dep(
        backend.is_available(), func_name, backend.package_name, backend.extra
    )
    rsgispy.check_optional_dep(
        HAVE_OPTUNA, func_name, "optuna", backend.extra
    )
    check_sklearn_available(func_name)

    if int(n_trials) < 1:
        raise rsgispy.RSGISPyException("n_trials must be at least 1.")
    dir_norm = direction.strip().lower()
    if dir_norm not in ("maximize", "minimize"):
        raise rsgispy.RSGISPyException(
            f"direction must be 'maximize' or 'minimize', got {direction!r}."
        )
    if early_stopping_rounds is not None and int(early_stopping_rounds) < 1:
        raise rsgispy.RSGISPyException(
            "early_stopping_rounds must be a positive integer."
        )

    _, class_info, _, train_x, train_y, _, valid_xy = prepare_cls_splits(
        cls_cfg=cls_cfg,
        train_data=train_data,
        test_data=None,
        valid_data=valid_data,
        feature_cols=feature_cols,
        use_internal_id=use_internal_id,
        out_id_col=out_id_col,
        internal_id_col=internal_id_col,
        binary=binary,
    )
    n_classes = len(class_info)
    _validate_target_labels(backend, train_y, n_classes, binary)

    valid_x = valid_xy[0] if valid_xy is not None else None
    valid_y = valid_xy[1] if valid_xy is not None else None

    trial_space, space_mode = _resolve_eff_optuna_space(
        backend=backend,
        binary=binary,
        param_space=param_space,
        extended_param_space=extended_param_space,
        fixed_params=fixed_params,
    )
    scorer = (
        sklearn.metrics.get_scorer(scoring)
        if isinstance(scoring, str)
        else scoring
    )
    cv_obj = (
        None
        if valid_x is not None
        else resolve_cv_splitter(cv, train_y, rnd_seed)
    )

    def _objective(trial: Any) -> float:
        if callable(trial_space) or trial_space:
            trial_params = suggest_optuna_params(trial, trial_space)
        else:
            trial_params = {}
        combined = dict(trial_params)
        if fixed_params:
            combined.update(fixed_params)
        est_jobs = n_jobs if valid_x is not None else 1
        params = _merge_gbdt_params(
            backend=backend,
            n_classes=n_classes,
            binary=binary,
            base_classifier=classifier,
            explicit_params=combined,
            rnd_seed=rnd_seed,
            n_jobs=est_jobs,
            early_stopping_rounds=early_stopping_rounds,
            has_valid=valid_x is not None,
            verbose=verbose,
        )
        if valid_x is not None and valid_y is not None:
            fitted = _build_and_fit_gbdt(
                backend=backend,
                params=params,
                base_classifier=classifier,
                train_x=train_x,
                train_y=train_y,
                valid_x=valid_x,
                valid_y=valid_y,
                early_stopping_rounds=early_stopping_rounds,
            )
            return float(scorer(fitted, valid_x, valid_y))

        est = backend.build_estimator(params, classifier)
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

    best_trial_params = dict(study.best_trial.params)
    if not callable(trial_space):
        for p_name, spec in trial_space.items():
            if not isinstance(spec, (Mapping, tuple, list)):
                best_trial_params[p_name] = spec

    best_combined = dict(best_trial_params)
    if fixed_params:
        best_combined.update(fixed_params)
    final_params = _merge_gbdt_params(
        backend=backend,
        n_classes=n_classes,
        binary=binary,
        base_classifier=classifier,
        explicit_params=best_combined,
        rnd_seed=rnd_seed,
        n_jobs=n_jobs,
        early_stopping_rounds=early_stopping_rounds,
        has_valid=valid_x is not None,
        verbose=verbose,
    )

    if refit:
        best_est = _build_and_fit_gbdt(
            backend=backend,
            params=final_params,
            base_classifier=classifier,
            train_x=train_x,
            train_y=train_y,
            valid_x=valid_x,
            valid_y=valid_y,
            early_stopping_rounds=early_stopping_rounds,
        )
    else:
        best_est = backend.build_estimator(final_params, classifier)

    best_params_json = to_json_compatible(best_combined)
    best_score = float(study.best_value)
    results_df = study.trials_dataframe()

    if out_results_file is not None:
        write_dataframe_table(results_df, out_results_file)

    if out_params_json is not None:
        out_json_path = ensure_parent_dir(out_params_json)
        scoring_name = (
            scoring if isinstance(scoring, str) else getattr(scoring, "__name__", "custom")
        )
        write_dict_to_json(
            {
                "library": backend.name.lower(),
                "task": "binary" if binary else "multiclass",
                "search_method": "optuna",
                "param_space_mode": space_mode,
                "scoring": scoring_name,
                "best_score": best_score,
                "best_params": best_params_json,
            },
            out_json_path,
        )

    if out_model_file is not None:
        if not refit:
            raise rsgispy.RSGISPyException(
                "out_model_file requires refit=True so the saved model is fitted."
            )
        backend.save_model(best_est, out_model_file)

    if return_results:
        return best_est, {
            "search_method": "optuna",
            "param_space_mode": space_mode,
            "best_params": best_params_json,
            "best_score": best_score,
            "results_df": results_df,
            "search_obj": study,
        }
    return best_est


def train_gbdt_classifier(
    backend: GBDTBackendSpec,
    func_name: str,
    cls_cfg: ClsConfigInput,
    train_data: SamplesInput,
    binary: bool,
    test_data: Optional[SamplesInput] = None,
    valid_data: Optional[SamplesInput] = None,
    classifier: Optional[Any] = None,
    model_params: Optional[Mapping[str, Any]] = None,
    early_stopping_rounds: Optional[int] = None,
    feature_cols: Optional[Sequence[str]] = None,
    use_internal_id: bool = True,
    rnd_seed: Optional[int] = None,
    n_jobs: Optional[int] = None,
    out_id_col: Optional[str] = None,
    internal_id_col: Optional[str] = None,
    out_model_file: Optional[Union[str, os.PathLike]] = None,
    out_metrics_file: Optional[Union[str, os.PathLike]] = None,
    return_model: bool = True,
    verbose: int = 0,
) -> Union[Dict[str, float], Tuple[Any, Dict[str, float]]]:
    """
    Train a GBDT binary or multi-class classifier on extracted sample tables
    and evaluate on optional test and validation datasets.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    rsgispy.check_optional_dep(
        backend.is_available(), func_name, backend.package_name, backend.extra
    )
    check_sklearn_available(func_name)

    if early_stopping_rounds is not None and int(early_stopping_rounds) < 1:
        raise rsgispy.RSGISPyException(
            "early_stopping_rounds must be a positive integer."
        )

    _, class_info, _, train_x, train_y, test_xy, valid_xy = prepare_cls_splits(
        cls_cfg=cls_cfg,
        train_data=train_data,
        test_data=test_data,
        valid_data=valid_data,
        feature_cols=feature_cols,
        use_internal_id=use_internal_id,
        out_id_col=out_id_col,
        internal_id_col=internal_id_col,
        binary=binary,
    )
    n_classes = len(class_info)
    _validate_target_labels(backend, train_y, n_classes, binary)

    valid_x = valid_xy[0] if valid_xy is not None else None
    valid_y = valid_xy[1] if valid_xy is not None else None

    params = _merge_gbdt_params(
        backend=backend,
        n_classes=n_classes,
        binary=binary,
        base_classifier=classifier,
        explicit_params=model_params,
        rnd_seed=rnd_seed,
        n_jobs=n_jobs,
        early_stopping_rounds=early_stopping_rounds,
        has_valid=valid_x is not None,
        verbose=verbose,
    )
    fitted_est = _build_and_fit_gbdt(
        backend=backend,
        params=params,
        base_classifier=classifier,
        train_x=train_x,
        train_y=train_y,
        valid_x=valid_x,
        valid_y=valid_y,
        early_stopping_rounds=early_stopping_rounds,
    )
    if not return_model and classifier is not None:
        sync_sklearn_estimator_state(classifier, fitted_est)

    pos_train_id = 1
    if binary:
        _, pos_entry = resolve_binary_classes(class_info, use_internal_id)
        pos_train_id = pos_entry[2]

    metrics = calc_cls_split_metrics(
        fitted_est,
        train_x,
        train_y,
        "train",
        binary=binary,
        pos_train_id=pos_train_id,
    )
    if test_xy is not None:
        metrics.update(
            calc_cls_split_metrics(
                fitted_est,
                test_xy[0],
                test_xy[1],
                "test",
                binary=binary,
                pos_train_id=pos_train_id,
            )
        )
    if valid_xy is not None:
        metrics.update(
            calc_cls_split_metrics(
                fitted_est,
                valid_xy[0],
                valid_xy[1],
                "valid",
                binary=binary,
                pos_train_id=pos_train_id,
            )
        )

    if out_model_file is not None:
        backend.save_model(fitted_est, out_model_file)

    if out_metrics_file is not None:
        metrics_path = ensure_parent_dir(out_metrics_file)
        write_dict_to_json(metrics, metrics_path)

    if return_model:
        return fitted_est, metrics
    return metrics


def apply_gbdt_classifier(
    backend: GBDTBackendSpec,
    func_name: str,
    cls_cfg: ClsConfigInput,
    classifier: Union[Any, str, os.PathLike],
    in_msk_img: Union[str, os.PathLike],
    img_msk_val: int,
    binary: bool,
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
    pos_class: Optional[Union[str, int]] = None,
    pred_threshold: Optional[float] = None,
    score_all_classes: bool = False,
) -> None:
    """
    Apply a trained GBDT binary or multi-class classifier block-by-block to
    raster imagery.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    rsgispy.check_optional_dep(
        backend.is_available(), func_name, backend.package_name, backend.extra
    )
    check_sklearn_available(func_name)
    estimator = (
        backend.load_model(classifier)
        if isinstance(classifier, (str, os.PathLike))
        else classifier
    )
    apply_classifier_raster(
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
        binary=binary,
        pos_class=pos_class,
        pred_threshold=pred_threshold,
        score_all_classes=score_all_classes,
    )
