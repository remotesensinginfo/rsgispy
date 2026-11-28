"""
LightGBM binary and multi-class classification workflows built on
:mod:`lightgbm`, Optuna Bayesian hyperparameter optimisation, model persistence,
and block-wise raster classification.
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

from rsgispy.classification._gbdt import (
    GBDTBackendSpec,
    apply_gbdt_classifier,
    optimise_gbdt_classifier,
    train_gbdt_classifier,
)
from rsgispy.classification._helpers import ClsConfigInput, SamplesInput
from rsgispy.classification.classdata import ImageBandInfo
from rsgispy.tools._gbdt_common import (
    apply_lgb_gpu_params as _tools_apply_lgb_gpu_params,
    detect_lightgbm_gpu_device as _tools_detect_lightgbm_gpu_device,
    fit_lgb_estimator as _fit_lgb_estimator,
    get_lightgbm_optuna_param_space as _tools_get_lightgbm_optuna_param_space,
    load_joblib_model,
    safe_import_optional,
    save_joblib_model,
    validate_gpu_device_id,
)

lgb, HAVE_LIGHTGBM, _LIGHTGBM_LOAD_ERROR = safe_import_optional("lightgbm")

__all__ = [
    "is_lightgbm_gpu_available",
    "save_lightgbm_classifier",
    "load_lightgbm_classifier",
    "get_lightgbm_optuna_param_space",
    "optimise_lightgbm_binary_classifier",
    "optimise_lightgbm_multiclass_classifier",
    "train_lightgbm_binary_classifier",
    "train_lightgbm_multiclass_classifier",
    "apply_lightgbm_binary_classifier",
    "apply_lightgbm_multiclass_classifier",
]


def _detect_lightgbm_gpu_device(
    gpu_device_id: Optional[int] = None,
) -> Optional[str]:
    """
    Detect an operational LightGBM GPU backend (``'cuda'`` or ``'gpu'``).

    Delegates to :func:`rsgispy.tools._gbdt_common.detect_lightgbm_gpu_device`.
    """
    dev_idx = validate_gpu_device_id(gpu_device_id)
    if not HAVE_LIGHTGBM or lgb is None:
        return None
    return _tools_detect_lightgbm_gpu_device(dev_idx)


def is_lightgbm_gpu_available(gpu_device_id: Optional[int] = None) -> bool:
    """
    Check whether CUDA or OpenCL GPU execution is supported and available for
    ``lightgbm``.

    :param gpu_device_id: Optional non-negative GPU device index
                          (Default = ``None``, using default device).
    :return: ``True`` if LightGBM GPU training is available, ``False`` otherwise.

    .. code:: python

        import rsgispy.classification.classlightgbm as clgb

        if clgb.is_lightgbm_gpu_available():
            print("LightGBM GPU ready")
    """
    return _detect_lightgbm_gpu_device(gpu_device_id) is not None


def get_lightgbm_optuna_param_space(extended: bool = False) -> Dict[str, Any]:
    """
    Return the built-in Optuna hyperparameter search space for
    ``lightgbm.LGBMClassifier``.

    When ``extended=False`` (default), returns the 10 core leaf-wise tree,
    split-gain, subsampling (with ``subsample_freq=1`` fixed so bagging is
    active), and L1/L2 regularisation parameters. When ``extended=True``,
    returns a 17-parameter space adding ``min_child_weight``,
    ``subsample_freq``, ``max_bin``, ``subsample_for_bin``, ``path_smooth``,
    ``extra_trees``, and ``class_weight``.

    :param extended: If ``True``, return the larger extended parameter space;
                     if ``False`` (default), return the core key parameters.
    :return: Dictionary of Optuna hyperparameter search specifications.

    .. code:: python

        import rsgispy.classification.classlightgbm as clgb

        key_space = clgb.get_lightgbm_optuna_param_space()
        full_space = clgb.get_lightgbm_optuna_param_space(extended=True)
    """
    return _tools_get_lightgbm_optuna_param_space(
        extended=extended, task="classification"
    )


def _default_lgb_binary_params(verbose: int) -> Dict[str, Any]:
    """Return default LGBMClassifier parameters for binary classification."""
    return {
        "objective": "binary",
        "n_estimators": 200,
        "verbose": -1 if verbose <= 0 else verbose,
    }


def _default_lgb_multiclass_params(n_classes: int, verbose: int) -> Dict[str, Any]:
    """Return default LGBMClassifier parameters for multi-class classification."""
    return {
        "objective": "multiclass",
        "num_class": int(n_classes),
        "n_estimators": 200,
        "verbose": -1 if verbose <= 0 else verbose,
    }


def _build_lgb_estimator(
    params: Dict[str, Any],
    base_classifier: Optional[Any],
) -> Any:
    """Create a fresh ``lightgbm.LGBMClassifier`` instance."""
    cls = type(base_classifier) if base_classifier is not None else lgb.LGBMClassifier
    return cls(**params)


def save_lightgbm_classifier(
    lgb_classifier: Any,
    out_file: Union[str, os.PathLike],
) -> None:
    """
    Save a fitted ``lightgbm.LGBMClassifier`` to disk using ``joblib``.

    :param lgb_classifier: Trained ``lightgbm.LGBMClassifier`` object.
    :param out_file: Output file path (e.g., ``'lgb_model.joblib'``).

    .. code:: python

        import rsgispy.classification.classlightgbm as clgb

        clgb.save_lightgbm_classifier(clf, "lgb_model.joblib")
    """
    save_joblib_model(
        model=lgb_classifier,
        out_file=out_file,
        have_pkg=HAVE_LIGHTGBM,
        pkg_name="lightgbm",
        extra="lightgbm",
        func_name="save_lightgbm_classifier",
        error_detail=_LIGHTGBM_LOAD_ERROR,
    )


def load_lightgbm_classifier(
    in_file: Union[str, os.PathLike],
) -> Any:
    """
    Load a saved ``lightgbm.LGBMClassifier`` from disk using ``joblib``.

    :param in_file: Input model file path.
    :return: Loaded ``lightgbm.LGBMClassifier`` instance.

    .. code:: python

        import rsgispy.classification.classlightgbm as clgb

        clf = clgb.load_lightgbm_classifier("lgb_model.joblib")
    """
    return load_joblib_model(
        in_file=in_file,
        have_pkg=HAVE_LIGHTGBM,
        pkg_name="lightgbm",
        extra="lightgbm",
        func_name="load_lightgbm_classifier",
        label="LightGBM model",
        error_detail=_LIGHTGBM_LOAD_ERROR,
    )


def _apply_lgb_gpu_params(
    params: Dict[str, Any],
    gpu_device_id: Optional[int],
) -> Dict[str, Any]:
    """Configure LightGBM parameters for CUDA or OpenCL GPU training."""
    return _tools_apply_lgb_gpu_params(
        params,
        gpu_device_id,
        detect_device_fn=_detect_lightgbm_gpu_device,
    )


_LGB_BACKEND = GBDTBackendSpec(
    name="LightGBM",
    package_name="lightgbm",
    extra="lightgbm",
    is_available=lambda: HAVE_LIGHTGBM,
    gpu_available=is_lightgbm_gpu_available,
    apply_gpu_params=_apply_lgb_gpu_params,
    seed_param="random_state",
    jobs_param="n_jobs",
    default_binary_params=_default_lgb_binary_params,
    default_multiclass_params=_default_lgb_multiclass_params,
    default_binary_space=lambda: get_lightgbm_optuna_param_space(extended=False),
    default_multiclass_space=lambda: get_lightgbm_optuna_param_space(extended=False),
    extended_binary_space=lambda: get_lightgbm_optuna_param_space(extended=True),
    extended_multiclass_space=lambda: get_lightgbm_optuna_param_space(extended=True),
    build_estimator=_build_lgb_estimator,
    fit_estimator=_fit_lgb_estimator,
    load_model=load_lightgbm_classifier,
    save_model=save_lightgbm_classifier,
    require_consecutive_zero_labels=False,
    get_error_detail=lambda: _LIGHTGBM_LOAD_ERROR,
)


def optimise_lightgbm_binary_classifier(
    cls_cfg: ClsConfigInput,
    train_data: SamplesInput,
    valid_data: Optional[SamplesInput] = None,
    param_space: Optional[
        Union[str, Mapping[str, Any], Callable[[Any], Mapping[str, Any]]]
    ] = None,
    extended_param_space: bool = False,
    fixed_params: Optional[Mapping[str, Any]] = None,
    lgb_classifier: Optional[Any] = None,
    scoring: Union[str, Callable[..., float]] = "roc_auc",
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
    Optimise hyperparameters for a binary ``lightgbm.LGBMClassifier``
    (``objective='binary'``) using Bayesian optimisation with Optuna.

    By default (``param_space=None``, ``extended_param_space=False``), optimises
    the 10 core LightGBM parameters (with ``subsample_freq=1`` fixed so bagging
    is active; see :func:`get_lightgbm_optuna_param_space`). Set
    ``extended_param_space=True`` (or ``param_space='extended'``) to optimise
    an expanded 17-parameter space including ``min_child_weight``,
    ``subsample_freq``, ``max_bin``, ``subsample_for_bin``, ``path_smooth``,
    ``extra_trees``, and ``class_weight``.

    :param cls_cfg: Binary :class:`~rsgispy.classification.classdata.ClassificationConfig`,
                    JSON config path, or 2-class mapping.
    :param train_data: Training sample :class:`pandas.DataFrame` or file path.
    :param valid_data: Optional validation sample table for out-of-sample
                       scoring and optional early stopping. When ``None``,
                       stratified ``cv``-fold cross-validation is used.
    :param param_space: Optional Optuna search space dictionary,
                        ``fn(trial) -> dict`` callable, or preset name
                        (``'default'`` / ``'extended'``). Uses built-in defaults
                        when ``None``.
    :param extended_param_space: If ``True`` (and ``param_space`` is ``None``),
                                 use the larger extended LightGBM search space
                                 (Default = ``False``).
    :param fixed_params: Optional fixed ``LGBMClassifier`` keyword parameters.
    :param lgb_classifier: Optional base ``LGBMClassifier`` template.
    :param scoring: Scikit-learn scoring name or callable (Default = ``'roc_auc'``).
    :param cv: Cross-validation folds or splitter (Default = ``5``).
    :param n_trials: Number of Optuna trials (Default = ``50``).
    :param timeout: Optional Optuna time budget in seconds.
    :param direction: Optuna direction (``'maximize'`` or ``'minimize'``).
                      Named scikit-learn scorers are greater-is-better, so
                      ``'minimize'`` is only valid with a callable ``scoring``.
    :param early_stopping_rounds: Optional early stopping rounds on ``valid_data``.
    :param n_jobs: Number of parallel threads/workers.
    :param rnd_seed: Optional random seed.
    :param feature_cols: Optional ordered feature column names.
    :param use_internal_id: Use consecutive 0/1 ``internal_cls_id`` labels
                            (Default = ``True``).
    :param refit: Refit and return a trained model (Default = ``True``).
    :param out_id_col: Optional override for the output class ID column name.
    :param internal_id_col: Optional override for the internal class ID column.
    :param out_results_file: Optional path to save Optuna trials table.
    :param out_params_json: Optional JSON path to save best parameters.
    :param out_model_file: Optional path to save refitted model.
    :param return_results: When ``True``, return ``(best_model, results_dict)``.
    :param verbose: Verbosity level (Default = ``0``).
    :param use_gpu: If ``True``, use CUDA or OpenCL GPU acceleration
                    (automatically clamping ``max_bin <= 255`` for OpenCL)
                    when available, falling back to CPU with a warning
                    otherwise (Default = ``False``).
    :param gpu_device_id: Optional non-negative GPU device index.
    :return: Optimised ``LGBMClassifier``, or ``(best_model, results_dict)``
             when ``return_results=True``.

    .. code:: python

        import rsgispy.classification.classlightgbm as clgb

        best_clf = clgb.optimise_lightgbm_binary_classifier(
            cls_cfg="cls_binary.json",
            train_data="train.parquet",
            valid_data="valid.parquet",
            extended_param_space=True,
            n_trials=50,
            rnd_seed=42,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    return optimise_gbdt_classifier(
        backend=_LGB_BACKEND,
        func_name="optimise_lightgbm_binary_classifier",
        cls_cfg=cls_cfg,
        train_data=train_data,
        binary=True,
        valid_data=valid_data,
        param_space=param_space,
        extended_param_space=extended_param_space,
        fixed_params=fixed_params,
        classifier=lgb_classifier,
        scoring=scoring,
        cv=cv,
        n_trials=n_trials,
        timeout=timeout,
        direction=direction,
        early_stopping_rounds=early_stopping_rounds,
        n_jobs=n_jobs,
        rnd_seed=rnd_seed,
        feature_cols=feature_cols,
        use_internal_id=use_internal_id,
        refit=refit,
        out_id_col=out_id_col,
        internal_id_col=internal_id_col,
        out_results_file=out_results_file,
        out_params_json=out_params_json,
        out_model_file=out_model_file,
        return_results=return_results,
        verbose=verbose,
        use_gpu=use_gpu,
        gpu_device_id=gpu_device_id,
        use_pbar=use_pbar,
    )


def optimise_lightgbm_multiclass_classifier(
    cls_cfg: ClsConfigInput,
    train_data: SamplesInput,
    valid_data: Optional[SamplesInput] = None,
    param_space: Optional[
        Union[str, Mapping[str, Any], Callable[[Any], Mapping[str, Any]]]
    ] = None,
    extended_param_space: bool = False,
    fixed_params: Optional[Mapping[str, Any]] = None,
    lgb_classifier: Optional[Any] = None,
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
    Optimise hyperparameters for a multi-class ``lightgbm.LGBMClassifier``
    (``objective='multiclass'``) using Bayesian optimisation with Optuna.

    By default (``param_space=None``, ``extended_param_space=False``), optimises
    the 10 core LightGBM parameters (with ``subsample_freq=1`` fixed so bagging
    is active; see :func:`get_lightgbm_optuna_param_space`). Set
    ``extended_param_space=True`` (or ``param_space='extended'``) to optimise
    the expanded 17-parameter space.

    :param cls_cfg: Multi-class :class:`~rsgispy.classification.classdata.ClassificationConfig`,
                    JSON config path, or class mapping.
    :param train_data: Training sample :class:`pandas.DataFrame` or file path.
    :param valid_data: Optional validation sample table.
    :param param_space: Optional Optuna search space dictionary, callable, or
                        preset name (``'default'`` / ``'extended'``).
    :param extended_param_space: If ``True`` (and ``param_space`` is ``None``),
                                 use the larger extended LightGBM search space
                                 (Default = ``False``).
    :param fixed_params: Optional fixed ``LGBMClassifier`` parameters.
    :param lgb_classifier: Optional base ``LGBMClassifier`` template.
    :param scoring: Scikit-learn scoring metric or callable (Default = ``'accuracy'``).
    :param cv: Cross-validation folds or splitter (Default = ``5``).
    :param n_trials: Number of Optuna trials (Default = ``50``).
    :param timeout: Optional Optuna time budget in seconds.
    :param direction: Optuna direction (``'maximize'`` or ``'minimize'``).
                      Named scikit-learn scorers are greater-is-better, so
                      ``'minimize'`` is only valid with a callable ``scoring``.
    :param early_stopping_rounds: Optional early stopping rounds on ``valid_data``.
    :param n_jobs: Number of parallel threads/workers.
    :param rnd_seed: Optional random seed.
    :param feature_cols: Optional ordered feature column names.
    :param use_internal_id: Use consecutive 0-based ``internal_cls_id`` labels.
    :param refit: Refit and return a trained model (Default = ``True``).
    :param out_id_col: Optional override for the output class ID column name.
    :param internal_id_col: Optional override for the internal class ID column.
    :param out_results_file: Optional path to save Optuna trials table.
    :param out_params_json: Optional JSON path to save best parameters.
    :param out_model_file: Optional path to save refitted model.
    :param return_results: When ``True``, return ``(best_model, results_dict)``.
    :param verbose: Verbosity level (Default = ``0``).
    :param use_gpu: If ``True``, use CUDA or OpenCL GPU acceleration
                    (automatically clamping ``max_bin <= 255`` for OpenCL)
                    when available, falling back to CPU with a warning
                    otherwise (Default = ``False``).
    :param gpu_device_id: Optional non-negative GPU device index.
    :return: Optimised ``LGBMClassifier``, or ``(best_model, results_dict)``
             when ``return_results=True``.

    .. code:: python

        import rsgispy.classification.classlightgbm as clgb

        best_clf = clgb.optimise_lightgbm_multiclass_classifier(
            cls_cfg="cls_config.json",
            train_data="train.parquet",
            extended_param_space=True,
            n_trials=50,
            rnd_seed=42,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    return optimise_gbdt_classifier(
        backend=_LGB_BACKEND,
        func_name="optimise_lightgbm_multiclass_classifier",
        cls_cfg=cls_cfg,
        train_data=train_data,
        binary=False,
        valid_data=valid_data,
        param_space=param_space,
        extended_param_space=extended_param_space,
        fixed_params=fixed_params,
        classifier=lgb_classifier,
        scoring=scoring,
        cv=cv,
        n_trials=n_trials,
        timeout=timeout,
        direction=direction,
        early_stopping_rounds=early_stopping_rounds,
        n_jobs=n_jobs,
        rnd_seed=rnd_seed,
        feature_cols=feature_cols,
        use_internal_id=use_internal_id,
        refit=refit,
        out_id_col=out_id_col,
        internal_id_col=internal_id_col,
        out_results_file=out_results_file,
        out_params_json=out_params_json,
        out_model_file=out_model_file,
        return_results=return_results,
        verbose=verbose,
        use_gpu=use_gpu,
        gpu_device_id=gpu_device_id,
        use_pbar=use_pbar,
    )


def train_lightgbm_binary_classifier(
    cls_cfg: ClsConfigInput,
    train_data: SamplesInput,
    test_data: Optional[SamplesInput] = None,
    valid_data: Optional[SamplesInput] = None,
    lgb_classifier: Optional[Any] = None,
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
    Train a binary ``lightgbm.LGBMClassifier`` on extracted sample tables and
    evaluate accuracy, precision, recall, F1, ROC-AUC, kappa, and log-loss on
    training, optional test, and optional validation splits.

    :param cls_cfg: Binary :class:`~rsgispy.classification.classdata.ClassificationConfig`,
                    JSON config path, or 2-class mapping.
    :param train_data: Training sample :class:`pandas.DataFrame` or file path.
    :param test_data: Optional testing sample table.
    :param valid_data: Optional validation sample table.
    :param lgb_classifier: Optional pre-configured ``LGBMClassifier`` template.
    :param model_params: Optional parameter dictionary.
    :param early_stopping_rounds: Optional early stopping rounds on ``valid_data``.
    :param feature_cols: Optional ordered feature column names.
    :param use_internal_id: Train on consecutive 0/1 ``internal_cls_id`` labels.
    :param rnd_seed: Optional random seed.
    :param n_jobs: Number of parallel threads.
    :param out_id_col: Optional override for the output class ID column name.
    :param internal_id_col: Optional override for the internal class ID column.
    :param out_model_file: Optional ``.joblib`` path to save model.
    :param out_metrics_file: Optional JSON path to save split metrics.
    :param return_model: When ``True`` (default), return ``(fitted_model, metrics)``.
    :param verbose: Verbosity level (Default = ``0``).
    :param use_gpu: If ``True``, use CUDA or OpenCL GPU acceleration when
                    available, falling back to CPU with a warning otherwise
                    (Default = ``False``).
    :param gpu_device_id: Optional non-negative GPU device index.
    :return: ``(fitted_model, metrics)`` or ``metrics``.

    .. code:: python

        import rsgispy.classification.classlightgbm as clgb

        model, metrics = clgb.train_lightgbm_binary_classifier(
            cls_cfg="cls_binary.json",
            train_data="train.parquet",
            test_data="test.parquet",
            rnd_seed=42,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    return train_gbdt_classifier(
        backend=_LGB_BACKEND,
        func_name="train_lightgbm_binary_classifier",
        cls_cfg=cls_cfg,
        train_data=train_data,
        binary=True,
        test_data=test_data,
        valid_data=valid_data,
        classifier=lgb_classifier,
        model_params=model_params,
        early_stopping_rounds=early_stopping_rounds,
        feature_cols=feature_cols,
        use_internal_id=use_internal_id,
        rnd_seed=rnd_seed,
        n_jobs=n_jobs,
        out_id_col=out_id_col,
        internal_id_col=internal_id_col,
        out_model_file=out_model_file,
        out_metrics_file=out_metrics_file,
        return_model=return_model,
        verbose=verbose,
        use_gpu=use_gpu,
        gpu_device_id=gpu_device_id,
    )


def train_lightgbm_multiclass_classifier(
    cls_cfg: ClsConfigInput,
    train_data: SamplesInput,
    test_data: Optional[SamplesInput] = None,
    valid_data: Optional[SamplesInput] = None,
    lgb_classifier: Optional[Any] = None,
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
    Train a multi-class ``lightgbm.LGBMClassifier`` (``objective='multiclass'``)
    on extracted sample tables and evaluate on optional test and validation sets.

    :param cls_cfg: Multi-class :class:`~rsgispy.classification.classdata.ClassificationConfig`,
                    JSON config path, or class mapping.
    :param train_data: Training sample :class:`pandas.DataFrame` or file path.
    :param test_data: Optional testing sample table.
    :param valid_data: Optional validation sample table.
    :param lgb_classifier: Optional pre-configured ``LGBMClassifier`` template.
    :param model_params: Optional parameter dictionary.
    :param early_stopping_rounds: Optional early stopping rounds on ``valid_data``.
    :param feature_cols: Optional ordered feature column names.
    :param use_internal_id: Train on consecutive 0-based ``internal_cls_id`` labels.
    :param rnd_seed: Optional random seed.
    :param n_jobs: Number of parallel threads.
    :param out_id_col: Optional override for the output class ID column name.
    :param internal_id_col: Optional override for the internal class ID column.
    :param out_model_file: Optional ``.joblib`` path to save model.
    :param out_metrics_file: Optional JSON path to save split metrics.
    :param return_model: When ``True`` (default), return ``(fitted_model, metrics)``.
    :param verbose: Verbosity level (Default = ``0``).
    :param use_gpu: If ``True``, use CUDA or OpenCL GPU acceleration when
                    available, falling back to CPU with a warning otherwise
                    (Default = ``False``).
    :param gpu_device_id: Optional non-negative GPU device index.
    :return: ``(fitted_model, metrics)`` or ``metrics``.

    .. code:: python

        import rsgispy.classification.classlightgbm as clgb

        model, metrics = clgb.train_lightgbm_multiclass_classifier(
            cls_cfg="cls_config.json",
            train_data="train.parquet",
            test_data="test.parquet",
            rnd_seed=42,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    return train_gbdt_classifier(
        backend=_LGB_BACKEND,
        func_name="train_lightgbm_multiclass_classifier",
        cls_cfg=cls_cfg,
        train_data=train_data,
        binary=False,
        test_data=test_data,
        valid_data=valid_data,
        classifier=lgb_classifier,
        model_params=model_params,
        early_stopping_rounds=early_stopping_rounds,
        feature_cols=feature_cols,
        use_internal_id=use_internal_id,
        rnd_seed=rnd_seed,
        n_jobs=n_jobs,
        out_id_col=out_id_col,
        internal_id_col=internal_id_col,
        out_model_file=out_model_file,
        out_metrics_file=out_metrics_file,
        return_model=return_model,
        verbose=verbose,
        use_gpu=use_gpu,
        gpu_device_id=gpu_device_id,
    )


def apply_lightgbm_binary_classifier(
    cls_cfg: ClsConfigInput,
    lgb_classifier: Union[Any, str, os.PathLike],
    in_msk_img: Union[str, os.PathLike],
    img_msk_val: int,
    output_img: Optional[Union[str, os.PathLike]] = None,
    out_score_img: Optional[Union[str, os.PathLike]] = None,
    img_band_info: Optional[Sequence[ImageBandInfo]] = None,
    pos_class: Optional[Union[str, int]] = None,
    pred_threshold: float = 0.5,
    score_all_classes: bool = False,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    out_no_data_val: Optional[int] = None,
    class_clr_names: bool = True,
    use_internal_id: bool = True,
    recode_to_out_ids: bool = True,
    score_no_data_val: float = -1.0,
    calc_stats: bool = True,
    block_size: Optional[int] = None,
    creation_opts: Optional[Mapping[str, Any]] = None,
    use_pbar: bool = True,
    use_gpu: bool = False,
    gpu_device_id: Optional[int] = None,
    out_conf_img: Optional[Union[str, os.PathLike]] = None,
    conf_method: str = "margin",
) -> None:
    """
    Apply a trained binary ``lightgbm.LGBMClassifier`` block-by-block to pixels
    where ``in_msk_img == img_msk_val``.

    Outputs a thematic class ID raster (thresholded at ``pred_threshold`` and
    automatically recoded to user ``out_id`` values) and/or a ``float32``
    positive-class probability score raster.

    :param cls_cfg: Binary :class:`~rsgispy.classification.classdata.ClassificationConfig`,
                    JSON config path, or 2-class mapping.
    :param lgb_classifier: Fitted ``LGBMClassifier`` or path to saved ``.joblib`` file.
    :param in_msk_img: Mask image path.
    :param img_msk_val: Mask pixel value defining pixels to classify.
    :param output_img: Optional output classified ID raster path.
    :param out_score_img: Optional output ``float32`` probability score raster.
    :param img_band_info: Optional sequence of
                          :class:`~rsgispy.classification.classdata.ImageBandInfo`.
    :param pos_class: Optional positive class name or ID.
    :param pred_threshold: Probability threshold in ``(0, 1)`` (Default = ``0.5``).
    :param score_all_classes: Write 1 positive-class probability band (``False``)
                              or 2 bands (``True``).
    :param gdalformat: Output raster format (Default = ``'GTIFF'``).
    :param datatype: Optional ``rsgispy.TYPE_*`` integer datatype for ``output_img``.
    :param out_no_data_val: Optional unsigned integer nodata value for ``output_img``.
    :param class_clr_names: Apply class RGB colour table to ``output_img``.
    :param use_internal_id: Whether model predicts 0-based ``internal_id``.
    :param recode_to_out_ids: Recode predicted IDs to ``out_id`` values.
    :param score_no_data_val: Nodata fill value for ``out_score_img``.
    :param calc_stats: Compute statistics and overviews (Default = ``True``).
    :param block_size: Optional processing block size in pixels.
    :param creation_opts: Optional GDAL creation options.
    :param use_pbar: Display a tqdm progress bar (Default = ``True``).
    :param use_gpu: If ``True``, validate GPU availability (emitting a warning
                    and falling back to CPU when unavailable; note that
                    LightGBM inference executes on CPU) (Default = ``False``).
    :param gpu_device_id: Optional non-negative GPU device index.
    :param out_conf_img: Optional output ``float32`` per-pixel confidence
                         raster in ``[0, 1]`` (``0`` = low confidence, ``1``
                         = confident, ``-9999`` outside the mask), computed
                         from the same ``predict_proba`` call (requires
                         classifier ``predict_proba``; Default = ``None``).
    :param conf_method: Confidence method for ``out_conf_img``: ``'margin'``
                        (best minus second-best class probability; binary
                        ``|2p - 1|``), ``'max_prob'`` (maximum probability
                        rescaled so ``1/K`` gives ``0``) or ``'entropy'``
                        (``1 -`` normalised Shannon entropy)
                        (Default = ``'margin'``). Tree-ensemble
                        probabilities are not calibrated, so confidence is
                        not the expected accuracy; see
                        :func:`~rsgispy.ml.predict.calc_proba_confidence`.

    .. code:: python

        import rsgispy.classification.classlightgbm as clgb

        clgb.apply_lightgbm_binary_classifier(
            cls_cfg="cls_binary.json",
            lgb_classifier=model,
            in_msk_img="mask.tif",
            img_msk_val=1,
            output_img="lgb_binary_cls.tif",
            out_score_img="lgb_binary_prob.tif",
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    apply_gbdt_classifier(
        backend=_LGB_BACKEND,
        func_name="apply_lightgbm_binary_classifier",
        cls_cfg=cls_cfg,
        classifier=lgb_classifier,
        in_msk_img=in_msk_img,
        img_msk_val=img_msk_val,
        binary=True,
        output_img=output_img,
        out_score_img=out_score_img,
        img_band_info=img_band_info,
        gdalformat=gdalformat,
        datatype=datatype,
        out_no_data_val=out_no_data_val,
        class_clr_names=class_clr_names,
        use_internal_id=use_internal_id,
        recode_to_out_ids=recode_to_out_ids,
        score_max_only=False,
        score_no_data_val=score_no_data_val,
        calc_stats=calc_stats,
        block_size=block_size,
        creation_opts=creation_opts,
        use_pbar=use_pbar,
        pos_class=pos_class,
        pred_threshold=pred_threshold,
        score_all_classes=score_all_classes,
        use_gpu=use_gpu,
        gpu_device_id=gpu_device_id,
        out_conf_img=out_conf_img,
        conf_method=conf_method,
    )


def apply_lightgbm_multiclass_classifier(
    cls_cfg: ClsConfigInput,
    lgb_classifier: Union[Any, str, os.PathLike],
    in_msk_img: Union[str, os.PathLike],
    img_msk_val: int,
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
    use_gpu: bool = False,
    gpu_device_id: Optional[int] = None,
    out_conf_img: Optional[Union[str, os.PathLike]] = None,
    conf_method: str = "margin",
) -> None:
    """
    Apply a trained multi-class ``lightgbm.LGBMClassifier`` block-by-block to
    pixels where ``in_msk_img == img_msk_val``, outputting classified class IDs
    and/or per-class (or maximum) probability score rasters.

    :param cls_cfg: Multi-class :class:`~rsgispy.classification.classdata.ClassificationConfig`,
                    JSON config path, or class mapping.
    :param lgb_classifier: Fitted ``LGBMClassifier`` or path to saved ``.joblib`` file.
    :param in_msk_img: Mask image path.
    :param img_msk_val: Mask pixel value defining pixels to classify.
    :param output_img: Optional output classified ID raster path.
    :param out_score_img: Optional output ``float32`` probability score raster.
    :param img_band_info: Optional sequence of
                          :class:`~rsgispy.classification.classdata.ImageBandInfo`.
    :param gdalformat: Output raster format (Default = ``'GTIFF'``).
    :param datatype: Optional ``rsgispy.TYPE_*`` integer datatype for ``output_img``.
    :param out_no_data_val: Optional unsigned integer nodata value for ``output_img``.
    :param class_clr_names: Apply class RGB colour table to ``output_img``.
    :param use_internal_id: Whether model predicts 0-based ``internal_id``.
    :param recode_to_out_ids: Recode predicted IDs to ``out_id`` values.
    :param score_max_only: Write single-band maximum probability when ``True``.
    :param score_no_data_val: Nodata fill value for ``out_score_img``.
    :param calc_stats: Compute statistics and overviews (Default = ``True``).
    :param block_size: Optional processing block size in pixels.
    :param creation_opts: Optional GDAL creation options.
    :param use_pbar: Display a tqdm progress bar (Default = ``True``).
    :param use_gpu: If ``True``, validate GPU availability (emitting a warning
                    and falling back to CPU when unavailable; note that
                    LightGBM inference executes on CPU) (Default = ``False``).
    :param gpu_device_id: Optional non-negative GPU device index.
    :param out_conf_img: Optional output ``float32`` per-pixel confidence
                         raster in ``[0, 1]`` (``0`` = low confidence, ``1``
                         = confident, ``-9999`` outside the mask), computed
                         from the same ``predict_proba`` call (requires
                         classifier ``predict_proba``; Default = ``None``).
    :param conf_method: Confidence method for ``out_conf_img``: ``'margin'``
                        (best minus second-best class probability; binary
                        ``|2p - 1|``), ``'max_prob'`` (maximum probability
                        rescaled so ``1/K`` gives ``0``) or ``'entropy'``
                        (``1 -`` normalised Shannon entropy)
                        (Default = ``'margin'``). Tree-ensemble
                        probabilities are not calibrated, so confidence is
                        not the expected accuracy; see
                        :func:`~rsgispy.ml.predict.calc_proba_confidence`.

    .. code:: python

        import rsgispy.classification.classlightgbm as clgb

        clgb.apply_lightgbm_multiclass_classifier(
            cls_cfg="cls_config.json",
            lgb_classifier=model,
            in_msk_img="mask.tif",
            img_msk_val=1,
            output_img="lgb_multiclass.tif",
            out_score_img="lgb_probs.tif",
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    apply_gbdt_classifier(
        backend=_LGB_BACKEND,
        func_name="apply_lightgbm_multiclass_classifier",
        cls_cfg=cls_cfg,
        classifier=lgb_classifier,
        in_msk_img=in_msk_img,
        img_msk_val=img_msk_val,
        binary=False,
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
        use_gpu=use_gpu,
        gpu_device_id=gpu_device_id,
        out_conf_img=out_conf_img,
        conf_method=conf_method,
    )
