"""
CatBoost binary and multi-class classification workflows built on
:mod:`catboost`, Optuna Bayesian hyperparameter optimisation, model persistence,
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
    apply_cb_gpu_params as _apply_cb_gpu_params,
    fit_cb_estimator as _fit_cb_estimator,
    get_catboost_optuna_param_space as _tools_get_catboost_optuna_param_space,
    is_catboost_gpu_available as _tools_is_catboost_gpu_available,
    load_xgb_or_cb_model,
    prepare_cb_predict_estimator as _prepare_cb_predict_estimator,
    safe_import_optional,
    save_xgb_or_cb_model,
    validate_gpu_device_id,
)

cb, HAVE_CATBOOST, _CATBOOST_LOAD_ERROR = safe_import_optional("catboost")
if HAVE_CATBOOST:
    _cb_utils, _have_cb_utils, _ = safe_import_optional("catboost.utils")
    _cb_get_gpu_device_count = (
        getattr(_cb_utils, "get_gpu_device_count", None)
        if _have_cb_utils and _cb_utils is not None
        else None
    )
else:
    _cb_get_gpu_device_count = None  # type: ignore[assignment]

__all__ = [
    "is_catboost_gpu_available",
    "save_catboost_classifier",
    "load_catboost_classifier",
    "get_catboost_optuna_param_space",
    "optimise_catboost_binary_classifier",
    "optimise_catboost_multiclass_classifier",
    "train_catboost_binary_classifier",
    "train_catboost_multiclass_classifier",
    "apply_catboost_binary_classifier",
    "apply_catboost_multiclass_classifier",
]


def is_catboost_gpu_available(gpu_device_id: Optional[int] = None) -> bool:
    """
    Check whether CUDA GPU execution is supported and available for
    ``catboost``.

    :param gpu_device_id: Optional non-negative GPU device index to verify
                          availability (Default = ``None``, checking that at
                          least one GPU is detected).
    :return: ``True`` if CatBoost detects a usable GPU, ``False`` otherwise.

    .. code:: python

        import rsgispy.classification.classcatboost as ccb

        if ccb.is_catboost_gpu_available():
            print("CatBoost GPU ready")
    """
    dev_idx = validate_gpu_device_id(gpu_device_id)
    if not HAVE_CATBOOST or _cb_get_gpu_device_count is None:
        return False
    return _tools_is_catboost_gpu_available(
        dev_idx, get_gpu_count_fn=_cb_get_gpu_device_count
    )


def get_catboost_optuna_param_space(
    extended: bool = False,
) -> Union[Dict[str, Any], Callable[[Any], Dict[str, Any]]]:
    """
    Return the built-in Optuna hyperparameter search space for
    ``catboost.CatBoostClassifier``.

    When ``extended=False`` (default), returns a dictionary specifying the 8
    core CatBoost parameters (``iterations``, ``learning_rate``, ``depth``,
    ``l2_leaf_reg``, ``random_strength``, ``bagging_temperature``, ``rsm``, and
    ``border_count``).

    When ``extended=True``, returns an Optuna ``fn(trial) -> dict`` callable
    covering a larger set of parameters while respecting CatBoost's conditional
    constraints: ``bagging_temperature`` is sampled for ``Bayesian``
    bootstrap, ``subsample`` for ``Bernoulli``/``MVS``, ``boosting_type`` for
    ``SymmetricTree``, and ``max_leaves`` for ``Lossguide``.

    :param extended: If ``True``, return the extended conditional search space
                     callable; if ``False`` (default), return the core key
                     parameter dictionary.
    :return: Optuna search space dict (when ``extended=False``) or callable
             ``fn(trial) -> dict`` (when ``extended=True``).

    .. code:: python

        import rsgispy.classification.classcatboost as ccb

        key_space = ccb.get_catboost_optuna_param_space()
        full_space = ccb.get_catboost_optuna_param_space(extended=True)
    """
    return _tools_get_catboost_optuna_param_space(
        extended=extended, task="classification"
    )


def _default_cb_binary_params(verbose: int) -> Dict[str, Any]:
    """Return default CatBoostClassifier parameters for binary classification."""
    return {
        "loss_function": "Logloss",
        "iterations": 200,
        "verbose": False if verbose <= 0 else verbose,
        "allow_writing_files": False,
    }


def _default_cb_multiclass_params(
    _n_classes: int,
    verbose: int,
) -> Dict[str, Any]:
    """Return default CatBoostClassifier parameters for multi-class classification."""
    return {
        "loss_function": "MultiClass",
        "iterations": 200,
        "verbose": False if verbose <= 0 else verbose,
        "allow_writing_files": False,
    }


def _build_cb_estimator(
    params: Dict[str, Any],
    base_classifier: Optional[Any],
) -> Any:
    """Create a fresh ``catboost.CatBoostClassifier`` instance."""
    cls = (
        type(base_classifier)
        if base_classifier is not None
        else cb.CatBoostClassifier
    )
    return cls(**params)


def save_catboost_classifier(
    cb_classifier: Any,
    out_file: Union[str, os.PathLike],
) -> None:
    """
    Save a fitted ``catboost.CatBoostClassifier`` to disk.

    Files ending in ``.joblib`` or ``.pkl`` are serialised with ``joblib``;
    all other extensions (such as ``.cbm`` or ``.json``) use CatBoost's native
    ``save_model()`` format.

    :param cb_classifier: Trained ``catboost.CatBoostClassifier`` object.
    :param out_file: Output model file path (e.g., ``'cb_model.cbm'`` or
                     ``'cb_model.joblib'``).

    .. code:: python

        import rsgispy.classification.classcatboost as ccb

        ccb.save_catboost_classifier(clf, "cb_model.cbm")
    """
    save_xgb_or_cb_model(
        model=cb_classifier,
        out_file=out_file,
        have_pkg=HAVE_CATBOOST,
        pkg_name="catboost",
        extra="catboost",
        func_name="save_catboost_classifier",
        error_detail=_CATBOOST_LOAD_ERROR,
    )


def load_catboost_classifier(
    in_file: Union[str, os.PathLike],
) -> Any:
    """
    Load a saved ``catboost.CatBoostClassifier`` from disk.

    Supports both ``joblib`` archives (``.joblib`` / ``.pkl``) and native
    CatBoost model files (``.cbm`` / ``.json`` / ``.bin``).

    :param in_file: Input model file path.
    :return: Loaded ``catboost.CatBoostClassifier`` instance.

    .. code:: python

        import rsgispy.classification.classcatboost as ccb

        clf = ccb.load_catboost_classifier("cb_model.cbm")
    """
    return load_xgb_or_cb_model(
        in_file=in_file,
        model_factory=lambda: cb.CatBoostClassifier(),
        have_pkg=HAVE_CATBOOST,
        pkg_name="catboost",
        extra="catboost",
        func_name="load_catboost_classifier",
        label="CatBoost model",
        error_detail=_CATBOOST_LOAD_ERROR,
    )


_CB_BACKEND = GBDTBackendSpec(
    name="CatBoost",
    package_name="catboost",
    extra="catboost",
    is_available=lambda: HAVE_CATBOOST,
    gpu_available=is_catboost_gpu_available,
    apply_gpu_params=_apply_cb_gpu_params,
    seed_param="random_seed",
    jobs_param="thread_count",
    default_binary_params=_default_cb_binary_params,
    default_multiclass_params=_default_cb_multiclass_params,
    default_binary_space=lambda: get_catboost_optuna_param_space(extended=False),
    default_multiclass_space=lambda: get_catboost_optuna_param_space(extended=False),
    extended_binary_space=lambda: get_catboost_optuna_param_space(extended=True),
    extended_multiclass_space=lambda: get_catboost_optuna_param_space(extended=True),
    build_estimator=_build_cb_estimator,
    fit_estimator=_fit_cb_estimator,
    load_model=load_catboost_classifier,
    save_model=save_catboost_classifier,
    prepare_predict_estimator=_prepare_cb_predict_estimator,
    require_consecutive_zero_labels=False,
    get_error_detail=lambda: _CATBOOST_LOAD_ERROR,
)


def optimise_catboost_binary_classifier(
    cls_cfg: ClsConfigInput,
    train_data: SamplesInput,
    valid_data: Optional[SamplesInput] = None,
    param_space: Optional[
        Union[str, Mapping[str, Any], Callable[[Any], Mapping[str, Any]]]
    ] = None,
    extended_param_space: bool = False,
    fixed_params: Optional[Mapping[str, Any]] = None,
    cb_classifier: Optional[Any] = None,
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
    Optimise hyperparameters for a binary ``catboost.CatBoostClassifier``
    (``loss_function='Logloss'``) using Bayesian optimisation with Optuna.

    By default (``param_space=None``, ``extended_param_space=False``), optimises
    the 8 core CatBoost parameters (see :func:`get_catboost_optuna_param_space`).
    Set ``extended_param_space=True`` (or ``param_space='extended'``) to
    optimise a larger conditional search space adding ``min_data_in_leaf``,
    ``leaf_estimation_iterations``, ``leaf_estimation_method``,
    ``model_size_reg``, ``auto_class_weights``, ``score_function``,
    ``bootstrap_type`` (with ``subsample`` or ``bagging_temperature``), and
    ``grow_policy`` (with ``boosting_type`` or ``max_leaves``).

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
                                 use the larger extended CatBoost search space
                                 (Default = ``False``).
    :param fixed_params: Optional fixed ``CatBoostClassifier`` keyword parameters.
    :param cb_classifier: Optional base ``CatBoostClassifier`` template.
    :param scoring: Scikit-learn scoring name or callable (Default = ``'roc_auc'``).
    :param cv: Cross-validation folds or splitter (Default = ``5``).
    :param n_trials: Number of Optuna trials (Default = ``50``).
    :param timeout: Optional Optuna time budget in seconds.
    :param direction: Optuna direction (``'maximize'`` or ``'minimize'``).
    :param early_stopping_rounds: Optional early stopping rounds on ``valid_data``.
    :param n_jobs: Number of parallel threads/workers (mapped to ``thread_count``).
    :param rnd_seed: Optional random seed.
    :param feature_cols: Optional ordered feature column names.
    :param use_internal_id: Use consecutive 0/1 ``internal_cls_id`` labels.
    :param refit: Refit and return a trained model (Default = ``True``).
    :param out_id_col: Optional override for the output class ID column name.
    :param internal_id_col: Optional override for the internal class ID column.
    :param out_results_file: Optional path to save Optuna trials table.
    :param out_params_json: Optional JSON path to save best parameters.
    :param out_model_file: Optional path to save refitted model.
    :param return_results: When ``True``, return ``(best_model, results_dict)``.
    :param verbose: Verbosity level (Default = ``0``).
    :param use_gpu: If ``True``, use CUDA GPU acceleration (``task_type='GPU'``,
                    automatically omitting CPU-only ``rsm`` and mapping
                    ``bootstrap_type='MVS'`` to ``'Bernoulli'``) when
                    available, falling back to CPU with a warning otherwise
                    (Default = ``False``).
    :param gpu_device_id: Optional non-negative CUDA GPU device index
                          (mapped to ``devices=str(gpu_device_id)``).
    :return: Optimised ``CatBoostClassifier``, or ``(best_model, results_dict)``
             when ``return_results=True``.

    .. code:: python

        import rsgispy.classification.classcatboost as ccb

        best_clf = ccb.optimise_catboost_binary_classifier(
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
        backend=_CB_BACKEND,
        func_name="optimise_catboost_binary_classifier",
        cls_cfg=cls_cfg,
        train_data=train_data,
        binary=True,
        valid_data=valid_data,
        param_space=param_space,
        extended_param_space=extended_param_space,
        fixed_params=fixed_params,
        classifier=cb_classifier,
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


def optimise_catboost_multiclass_classifier(
    cls_cfg: ClsConfigInput,
    train_data: SamplesInput,
    valid_data: Optional[SamplesInput] = None,
    param_space: Optional[
        Union[str, Mapping[str, Any], Callable[[Any], Mapping[str, Any]]]
    ] = None,
    extended_param_space: bool = False,
    fixed_params: Optional[Mapping[str, Any]] = None,
    cb_classifier: Optional[Any] = None,
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
    Optimise hyperparameters for a multi-class ``catboost.CatBoostClassifier``
    (``loss_function='MultiClass'``) using Bayesian optimisation with Optuna.

    By default (``param_space=None``, ``extended_param_space=False``), optimises
    the 8 core CatBoost parameters (see :func:`get_catboost_optuna_param_space`).
    Set ``extended_param_space=True`` (or ``param_space='extended'``) to
    optimise the extended conditional search space.

    :param cls_cfg: Multi-class :class:`~rsgispy.classification.classdata.ClassificationConfig`,
                    JSON config path, or class mapping.
    :param train_data: Training sample :class:`pandas.DataFrame` or file path.
    :param valid_data: Optional validation sample table.
    :param param_space: Optional Optuna search space dictionary, callable, or
                        preset name (``'default'`` / ``'extended'``).
    :param extended_param_space: If ``True`` (and ``param_space`` is ``None``),
                                 use the larger extended CatBoost search space
                                 (Default = ``False``).
    :param fixed_params: Optional fixed ``CatBoostClassifier`` parameters.
    :param cb_classifier: Optional base ``CatBoostClassifier`` template.
    :param scoring: Scikit-learn scoring metric or callable (Default = ``'accuracy'``).
    :param cv: Cross-validation folds or splitter (Default = ``5``).
    :param n_trials: Number of Optuna trials (Default = ``50``).
    :param timeout: Optional Optuna time budget in seconds.
    :param direction: Optuna direction (``'maximize'`` or ``'minimize'``).
    :param early_stopping_rounds: Optional early stopping rounds on ``valid_data``.
    :param n_jobs: Number of parallel threads/workers (mapped to ``thread_count``).
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
    :param use_gpu: If ``True``, use CUDA GPU acceleration (``task_type='GPU'``,
                    automatically omitting CPU-only ``rsm`` and mapping
                    ``bootstrap_type='MVS'`` to ``'Bernoulli'``) when
                    available, falling back to CPU with a warning otherwise
                    (Default = ``False``).
    :param gpu_device_id: Optional non-negative CUDA GPU device index.
    :return: Optimised ``CatBoostClassifier``, or ``(best_model, results_dict)``
             when ``return_results=True``.

    .. code:: python

        import rsgispy.classification.classcatboost as ccb

        best_clf = ccb.optimise_catboost_multiclass_classifier(
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
        backend=_CB_BACKEND,
        func_name="optimise_catboost_multiclass_classifier",
        cls_cfg=cls_cfg,
        train_data=train_data,
        binary=False,
        valid_data=valid_data,
        param_space=param_space,
        extended_param_space=extended_param_space,
        fixed_params=fixed_params,
        classifier=cb_classifier,
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


def train_catboost_binary_classifier(
    cls_cfg: ClsConfigInput,
    train_data: SamplesInput,
    test_data: Optional[SamplesInput] = None,
    valid_data: Optional[SamplesInput] = None,
    cb_classifier: Optional[Any] = None,
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
    Train a binary ``catboost.CatBoostClassifier`` on extracted sample tables
    and evaluate accuracy, precision, recall, F1, ROC-AUC, kappa, and log-loss
    on training, optional test, and optional validation splits.

    :param cls_cfg: Binary :class:`~rsgispy.classification.classdata.ClassificationConfig`,
                    JSON config path, or 2-class mapping.
    :param train_data: Training sample :class:`pandas.DataFrame` or file path.
    :param test_data: Optional testing sample table.
    :param valid_data: Optional validation sample table.
    :param cb_classifier: Optional pre-configured ``CatBoostClassifier`` template.
    :param model_params: Optional parameter dictionary.
    :param early_stopping_rounds: Optional early stopping rounds on ``valid_data``.
    :param feature_cols: Optional ordered feature column names.
    :param use_internal_id: Train on consecutive 0/1 ``internal_cls_id`` labels.
    :param rnd_seed: Optional random seed.
    :param n_jobs: Number of parallel threads (mapped to ``thread_count``).
    :param out_id_col: Optional override for the output class ID column name.
    :param internal_id_col: Optional override for the internal class ID column.
    :param out_model_file: Optional ``.cbm`` or ``.joblib`` path to save model.
    :param out_metrics_file: Optional JSON path to save split metrics.
    :param return_model: When ``True`` (default), return ``(fitted_model, metrics)``.
    :param verbose: Verbosity level (Default = ``0``).
    :param use_gpu: If ``True``, use CUDA GPU acceleration when available,
                    falling back to CPU with a warning otherwise
                    (Default = ``False``).
    :param gpu_device_id: Optional non-negative CUDA GPU device index.
    :return: ``(fitted_model, metrics)`` or ``metrics``.

    .. code:: python

        import rsgispy.classification.classcatboost as ccb

        model, metrics = ccb.train_catboost_binary_classifier(
            cls_cfg="cls_binary.json",
            train_data="train.parquet",
            test_data="test.parquet",
            rnd_seed=42,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    return train_gbdt_classifier(
        backend=_CB_BACKEND,
        func_name="train_catboost_binary_classifier",
        cls_cfg=cls_cfg,
        train_data=train_data,
        binary=True,
        test_data=test_data,
        valid_data=valid_data,
        classifier=cb_classifier,
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


def train_catboost_multiclass_classifier(
    cls_cfg: ClsConfigInput,
    train_data: SamplesInput,
    test_data: Optional[SamplesInput] = None,
    valid_data: Optional[SamplesInput] = None,
    cb_classifier: Optional[Any] = None,
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
    Train a multi-class ``catboost.CatBoostClassifier``
    (``loss_function='MultiClass'``) on extracted sample tables and evaluate on
    optional test and validation sets.

    :param cls_cfg: Multi-class :class:`~rsgispy.classification.classdata.ClassificationConfig`,
                    JSON config path, or class mapping.
    :param train_data: Training sample :class:`pandas.DataFrame` or file path.
    :param test_data: Optional testing sample table.
    :param valid_data: Optional validation sample table.
    :param cb_classifier: Optional pre-configured ``CatBoostClassifier`` template.
    :param model_params: Optional parameter dictionary.
    :param early_stopping_rounds: Optional early stopping rounds on ``valid_data``.
    :param feature_cols: Optional ordered feature column names.
    :param use_internal_id: Train on consecutive 0-based ``internal_cls_id`` labels.
    :param rnd_seed: Optional random seed.
    :param n_jobs: Number of parallel threads (mapped to ``thread_count``).
    :param out_id_col: Optional override for the output class ID column name.
    :param internal_id_col: Optional override for the internal class ID column.
    :param out_model_file: Optional ``.cbm`` or ``.joblib`` path to save model.
    :param out_metrics_file: Optional JSON path to save split metrics.
    :param return_model: When ``True`` (default), return ``(fitted_model, metrics)``.
    :param verbose: Verbosity level (Default = ``0``).
    :param use_gpu: If ``True``, use CUDA GPU acceleration when available,
                    falling back to CPU with a warning otherwise
                    (Default = ``False``).
    :param gpu_device_id: Optional non-negative CUDA GPU device index.
    :return: ``(fitted_model, metrics)`` or ``metrics``.

    .. code:: python

        import rsgispy.classification.classcatboost as ccb

        model, metrics = ccb.train_catboost_multiclass_classifier(
            cls_cfg="cls_config.json",
            train_data="train.parquet",
            test_data="test.parquet",
            rnd_seed=42,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    return train_gbdt_classifier(
        backend=_CB_BACKEND,
        func_name="train_catboost_multiclass_classifier",
        cls_cfg=cls_cfg,
        train_data=train_data,
        binary=False,
        test_data=test_data,
        valid_data=valid_data,
        classifier=cb_classifier,
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


def apply_catboost_binary_classifier(
    cls_cfg: ClsConfigInput,
    cb_classifier: Union[Any, str, os.PathLike],
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
    Apply a trained binary ``catboost.CatBoostClassifier`` block-by-block to
    pixels where ``in_msk_img == img_msk_val``.

    Outputs a thematic class ID raster (thresholded at ``pred_threshold`` and
    automatically recoded to user ``out_id`` values) and/or a ``float32``
    positive-class probability score raster.

    :param cls_cfg: Binary :class:`~rsgispy.classification.classdata.ClassificationConfig`,
                    JSON config path, or 2-class mapping.
    :param cb_classifier: Fitted ``CatBoostClassifier`` or path to saved model.
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
    :param use_gpu: If ``True``, run CUDA GPU prediction (``task_type='GPU'``)
                    when available, falling back to CPU with a warning otherwise
                    (Default = ``False``).
    :param gpu_device_id: Optional non-negative CUDA GPU device index.
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

        import rsgispy.classification.classcatboost as ccb

        ccb.apply_catboost_binary_classifier(
            cls_cfg="cls_binary.json",
            cb_classifier=model,
            in_msk_img="mask.tif",
            img_msk_val=1,
            output_img="cb_binary_cls.tif",
            out_score_img="cb_binary_prob.tif",
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    apply_gbdt_classifier(
        backend=_CB_BACKEND,
        func_name="apply_catboost_binary_classifier",
        cls_cfg=cls_cfg,
        classifier=cb_classifier,
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


def apply_catboost_multiclass_classifier(
    cls_cfg: ClsConfigInput,
    cb_classifier: Union[Any, str, os.PathLike],
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
    Apply a trained multi-class ``catboost.CatBoostClassifier`` block-by-block to
    pixels where ``in_msk_img == img_msk_val``, outputting classified class IDs
    and/or per-class (or maximum) probability score rasters.

    :param cls_cfg: Multi-class :class:`~rsgispy.classification.classdata.ClassificationConfig`,
                    JSON config path, or class mapping.
    :param cb_classifier: Fitted ``CatBoostClassifier`` or path to saved model.
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
    :param use_gpu: If ``True``, run CUDA GPU prediction (``task_type='GPU'``)
                    when available, falling back to CPU with a warning otherwise
                    (Default = ``False``).
    :param gpu_device_id: Optional non-negative CUDA GPU device index.
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

        import rsgispy.classification.classcatboost as ccb

        ccb.apply_catboost_multiclass_classifier(
            cls_cfg="cls_config.json",
            cb_classifier=model,
            in_msk_img="mask.tif",
            img_msk_val=1,
            output_img="cb_multiclass.tif",
            out_score_img="cb_probs.tif",
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    apply_gbdt_classifier(
        backend=_CB_BACKEND,
        func_name="apply_catboost_multiclass_classifier",
        cls_cfg=cls_cfg,
        classifier=cb_classifier,
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
