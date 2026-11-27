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
    apply_classifier_vector,
    calc_cls_split_metrics,
    prepare_cls_splits,
    resolve_binary_classes,
)
from rsgispy.classification._helpers import (
    ClsConfigInput,
    SamplesInput,
    check_sklearn_available,
)
from rsgispy.classification.classdata import ImageBandInfo
from rsgispy.tools._gbdt_common import (
    check_gbdt_backend_available as _check_backend_available,
    merge_gbdt_params as _tools_merge_gbdt_params,
    prepared_gbdt_predict_estimator as _prepared_gbdt_predict_estimator,
    resolve_gbdt_gpu as _tools_resolve_gbdt_gpu,
    resolve_gbdt_optuna_space as _tools_resolve_gbdt_optuna_space,
    run_gbdt_optuna_optimisation as _run_gbdt_optuna_optimisation,
)
from rsgispy.tools._ml_tune import HAVE_OPTUNA
from rsgispy.tools._sklearn import (  # noqa: F401
    sklearn,
    sync_sklearn_estimator_state,
)
from rsgispy.tools.files import ensure_parent_dir
from rsgispy.tools.jsonio import write_dict_to_json

_OptunaSpaceType = Union[Dict[str, Any], Callable[[Any], Mapping[str, Any]]]


@dataclass(frozen=True)
class GBDTBackendSpec:
    """Backend descriptor abstracting XGBoost, LightGBM, and CatBoost."""

    # pylint: disable=too-many-instance-attributes

    name: str
    package_name: str
    extra: str
    is_available: Callable[[], bool]
    gpu_available: Callable[[Optional[int]], bool]
    apply_gpu_params: Callable[[Dict[str, Any], Optional[int]], Dict[str, Any]]
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
    prepare_predict_estimator: Optional[
        Callable[
            [Any, bool, Optional[int]],
            Tuple[Any, Dict[str, Any], Optional[Callable[[], None]]],
        ]
    ] = None
    require_consecutive_zero_labels: bool = False
    get_error_detail: Optional[Callable[[], Optional[str]]] = None


def resolve_gbdt_gpu(
    backend: GBDTBackendSpec,
    use_gpu: bool,
    gpu_device_id: Optional[int],
) -> Tuple[bool, Optional[int]]:
    """
    Validate GPU device selection and resolve whether GPU execution is active,
    falling back to CPU with a ``UserWarning`` when unavailable.
    """
    return _tools_resolve_gbdt_gpu(
        backend_name=backend.name,
        gpu_check_fn=backend.gpu_available,
        use_gpu=use_gpu,
        gpu_device_id=gpu_device_id,
    )


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
    gpu_active: bool = False,
    gpu_device_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Build clean estimator parameter dictionary with task defaults."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    task_defaults = (
        backend.default_binary_params(verbose)
        if binary
        else backend.default_multiclass_params(n_classes, verbose)
    )
    return _tools_merge_gbdt_params(
        backend_name=backend.name,
        task_defaults=task_defaults,
        seed_param=backend.seed_param,
        jobs_param=backend.jobs_param,
        apply_gpu_params=backend.apply_gpu_params,
        base_estimator=base_classifier,
        explicit_params=explicit_params,
        rnd_seed=rnd_seed,
        n_jobs=n_jobs,
        early_stopping_rounds=early_stopping_rounds,
        has_valid=has_valid,
        gpu_active=gpu_active,
        gpu_device_id=gpu_device_id,
    )


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
    def_fn = (
        backend.default_binary_space
        if binary
        else backend.default_multiclass_space
    )
    ext_fn = (
        backend.extended_binary_space
        if binary
        else backend.extended_multiclass_space
    )
    return _tools_resolve_gbdt_optuna_space(
        default_space_fn=def_fn,
        extended_space_fn=ext_fn,
        param_space=param_space,
        extended_param_space=extended_param_space,
        fixed_params=fixed_params,
    )


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
    use_gpu: bool = False,
    gpu_device_id: Optional[int] = None,
    use_pbar: bool = True,
) -> Union[Any, Tuple[Any, Dict[str, Any]]]:
    """
    Optimise hyperparameters for a GBDT binary or multi-class classifier using
    Optuna Bayesian optimisation (TPE sampler).
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    _check_backend_available(backend, func_name)
    rsgispy.check_optional_dep(
        HAVE_OPTUNA, func_name, "optuna", backend.extra
    )
    check_sklearn_available(func_name)
    gpu_active, eff_gpu_id = resolve_gbdt_gpu(backend, use_gpu, gpu_device_id)

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

    def _merge(
        explicit: Mapping[str, Any],
        eff_jobs: Optional[int],
    ) -> Dict[str, Any]:
        return _merge_gbdt_params(
            backend=backend,
            n_classes=n_classes,
            binary=binary,
            base_classifier=classifier,
            explicit_params=explicit,
            rnd_seed=rnd_seed,
            n_jobs=eff_jobs,
            early_stopping_rounds=early_stopping_rounds,
            has_valid=valid_x is not None,
            verbose=verbose,
            gpu_active=gpu_active,
            gpu_device_id=eff_gpu_id,
        )

    def _build_fit(params: Dict[str, Any]) -> Any:
        return _build_and_fit_gbdt(
            backend=backend,
            params=params,
            base_classifier=classifier,
            train_x=train_x,
            train_y=train_y,
            valid_x=valid_x,
            valid_y=valid_y,
            early_stopping_rounds=early_stopping_rounds,
        )

    return _run_gbdt_optuna_optimisation(
        backend=backend,
        func_name=func_name,
        train_x=train_x,
        train_y=train_y,
        valid_x=valid_x,
        valid_y=valid_y,
        trial_space=trial_space,
        space_mode=space_mode,
        fixed_params=fixed_params,
        merge_params_fn=_merge,
        build_estimator_fn=lambda p: backend.build_estimator(p, classifier),
        build_and_fit_fn=_build_fit,
        scoring=scoring,
        cv=cv,
        stratified=True,
        n_trials=n_trials,
        timeout=timeout,
        direction=direction,
        n_jobs=n_jobs,
        rnd_seed=rnd_seed,
        refit=refit,
        verbose=verbose,
        use_gpu=use_gpu,
        gpu_active=gpu_active,
        use_pbar=use_pbar,
        out_results_file=out_results_file,
        out_params_json=out_params_json,
        out_model_file=out_model_file,
        return_results=return_results,
        extra_meta={
            "library": backend.name.lower(),
            "task": "binary" if binary else "multiclass",
            "use_gpu_requested": bool(use_gpu),
            "gpu_used": bool(gpu_active),
        },
    )


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
    use_gpu: bool = False,
    gpu_device_id: Optional[int] = None,
) -> Union[Dict[str, float], Tuple[Any, Dict[str, float]]]:
    """
    Train a GBDT binary or multi-class classifier on extracted sample tables
    and evaluate on optional test and validation datasets.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    _check_backend_available(backend, func_name)
    check_sklearn_available(func_name)
    gpu_active, eff_gpu_id = resolve_gbdt_gpu(backend, use_gpu, gpu_device_id)

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
        gpu_active=gpu_active,
        gpu_device_id=eff_gpu_id,
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

    with _prepared_gbdt_predict_estimator(
        backend, fitted_est, gpu_active, eff_gpu_id
    ) as (eval_est, predict_kwargs):
        metrics = calc_cls_split_metrics(
            eval_est,
            train_x,
            train_y,
            "train",
            binary=binary,
            pos_train_id=pos_train_id,
            predict_kwargs=predict_kwargs,
        )
        if test_xy is not None:
            metrics.update(
                calc_cls_split_metrics(
                    eval_est,
                    test_xy[0],
                    test_xy[1],
                    "test",
                    binary=binary,
                    pos_train_id=pos_train_id,
                    predict_kwargs=predict_kwargs,
                )
            )
        if valid_xy is not None:
            metrics.update(
                calc_cls_split_metrics(
                    eval_est,
                    valid_xy[0],
                    valid_xy[1],
                    "valid",
                    binary=binary,
                    pos_train_id=pos_train_id,
                    predict_kwargs=predict_kwargs,
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
    output_img: Optional[Union[str, os.PathLike]] = None,
    out_score_img: Optional[Union[str, os.PathLike]] = None,
    img_band_info: Optional[Sequence[ImageBandInfo]] = None,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    out_no_data_val: Optional[int] = None,
    class_clr_names: bool = True,
    use_internal_id: bool = True,
    recode_to_out_ids: bool = True,
    score_max_only: bool = False,
    score_no_data_val: float = -1.0,
    calc_stats: bool = True,
    block_size: Optional[int] = None,
    creation_opts: Optional[Mapping[str, Any]] = None,
    use_pbar: bool = True,
    pos_class: Optional[Union[str, int]] = None,
    pred_threshold: Optional[float] = None,
    score_all_classes: bool = False,
    use_gpu: bool = False,
    gpu_device_id: Optional[int] = None,
    out_conf_img: Optional[Union[str, os.PathLike]] = None,
    conf_method: str = "margin",
) -> None:
    """
    Apply a trained GBDT binary or multi-class classifier block-by-block to
    raster imagery.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    _check_backend_available(backend, func_name)
    check_sklearn_available(func_name)
    gpu_active, eff_gpu_id = resolve_gbdt_gpu(backend, use_gpu, gpu_device_id)
    estimator = (
        backend.load_model(classifier)
        if isinstance(classifier, (str, os.PathLike))
        else classifier
    )
    with _prepared_gbdt_predict_estimator(
        backend, estimator, gpu_active, eff_gpu_id
    ) as (eval_est, predict_kwargs):
        apply_classifier_raster(
            cls_cfg=cls_cfg,
            estimator=eval_est,
            in_msk_img=in_msk_img,
            img_msk_val=img_msk_val,
            output_img=output_img,
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
            calc_stats=calc_stats,
            block_size=block_size,
            creation_opts=creation_opts,
            use_pbar=use_pbar,
            binary=binary,
            pos_class=pos_class,
            pred_threshold=pred_threshold,
            score_all_classes=score_all_classes,
            out_conf_img=out_conf_img,
            conf_method=conf_method,
            predict_kwargs=predict_kwargs,
        )


def apply_gbdt_classifier_segments(
    backend: GBDTBackendSpec,
    func_name: str,
    cls_cfg: ClsConfigInput,
    classifier: Union[Any, str, os.PathLike],
    out_vec_file: Union[str, os.PathLike],
    binary: bool,
    *,
    use_gpu: bool = False,
    gpu_device_id: Optional[int] = None,
    **kwargs: Any,
) -> None:
    """
    Apply a trained GBDT binary or multi-class classifier to a segment feature
    layer in chunks (see
    :func:`rsgispy.classification._cls_common.apply_classifier_vector`).
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    _check_backend_available(backend, func_name)
    check_sklearn_available(func_name)
    gpu_active, eff_gpu_id = resolve_gbdt_gpu(backend, use_gpu, gpu_device_id)
    is_path = isinstance(classifier, (str, os.PathLike))
    estimator = backend.load_model(classifier) if is_path else classifier
    if is_path and kwargs.get("model_file") is None:
        kwargs["model_file"] = classifier
    with _prepared_gbdt_predict_estimator(
        backend, estimator, gpu_active, eff_gpu_id
    ) as (eval_est, predict_kwargs):
        apply_classifier_vector(
            cls_cfg,
            eval_est,
            out_vec_file,
            binary=binary,
            predict_kwargs=predict_kwargs,
            **kwargs,
        )
