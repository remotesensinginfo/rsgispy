"""
XGBoost binary and multi-class classification workflows built on
:mod:`xgboost`, Optuna Bayesian hyperparameter optimisation, model persistence,
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
    register_gbdt_backend,
    forward_gbdt_call,
    apply_xgb_gpu_params as _apply_xgb_gpu_params,
    fit_xgb_estimator as _fit_xgb_estimator,
    get_xgboost_optuna_param_space as _tools_get_xgboost_optuna_param_space,
    is_xgboost_gpu_available as _tools_is_xgboost_gpu_available,
    load_xgb_or_cb_model,
    prepare_xgb_predict_estimator as _prepare_xgb_predict_estimator,
    safe_import_optional,
    save_xgb_or_cb_model,
    validate_gpu_device_id,
)

xgb, HAVE_XGBOOST, _XGBOOST_LOAD_ERROR = safe_import_optional("xgboost")

__all__ = [
    "is_xgboost_gpu_available",
    "save_xgboost_classifier",
    "load_xgboost_classifier",
    "get_xgboost_optuna_param_space",
    "optimise_xgboost_binary_classifier",
    "optimise_xgboost_multiclass_classifier",
    "train_xgboost_binary_classifier",
    "train_xgboost_multiclass_classifier",
    "apply_xgboost_binary_classifier",
    "apply_xgboost_multiclass_classifier",
]


def is_xgboost_gpu_available(gpu_device_id: Optional[int] = None) -> bool:
    """
    Check whether CUDA GPU execution is supported and available for
    ``xgboost``.

    Verifies that ``xgboost`` is installed, built with CUDA support
    (``build_info()['USE_CUDA']``), and able to fit a minimal histogram tree
    on the target CUDA device.

    :param gpu_device_id: Optional non-negative CUDA GPU device index
                          (Default = ``None``, using default CUDA device).
    :return: ``True`` if an XGBoost CUDA GPU is available, ``False`` otherwise.

    .. code:: python

        import rsgispy.classification.classxgboost as cxgb

        if cxgb.is_xgboost_gpu_available():
            print("XGBoost CUDA GPU ready")
    """
    dev_idx = validate_gpu_device_id(gpu_device_id)
    if not HAVE_XGBOOST or xgb is None:
        return False
    return _tools_is_xgboost_gpu_available(dev_idx)


def get_xgboost_optuna_param_space(
    extended: bool = False,
    binary: bool = True,
) -> Dict[str, Any]:
    """
    Return the built-in Optuna hyperparameter search space for
    ``xgboost.XGBClassifier``.

    When ``extended=False`` (default), returns the 9 core boosting, tree-depth,
    split-pruning, subsampling, and L1/L2 regularisation parameters. When
    ``extended=True``, returns an expanded search space adding level/node
    column subsampling, ``max_delta_step``, ``max_bin``, ``grow_policy``,
    ``max_leaves``, and (for ``binary=True``) ``scale_pos_weight``.

    :param extended: If ``True``, return the larger extended parameter space;
                     if ``False`` (default), return the core key parameters.
    :param binary: Whether target task is binary classification
                   (adds ``scale_pos_weight`` when ``extended=True``).
    :return: Dictionary of Optuna hyperparameter search specifications.

    .. code:: python

        import rsgispy.classification.classxgboost as cxgb

        key_space = cxgb.get_xgboost_optuna_param_space()
        full_space = cxgb.get_xgboost_optuna_param_space(extended=True)
    """
    return _tools_get_xgboost_optuna_param_space(
        extended=extended, binary=binary, task="classification"
    )


def _default_xgb_binary_params(verbose: int) -> Dict[str, Any]:
    """Return default XGBClassifier parameters for binary classification."""
    return {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "n_estimators": 200,
        "verbosity": 0 if verbose <= 0 else 1,
    }


def _default_xgb_multiclass_params(n_classes: int, verbose: int) -> Dict[str, Any]:
    """Return default XGBClassifier parameters for multi-class classification."""
    return {
        "objective": "multi:softprob",
        "num_class": int(n_classes),
        "eval_metric": "mlogloss",
        "n_estimators": 200,
        "verbosity": 0 if verbose <= 0 else 1,
    }


def _build_xgb_estimator(
    params: Dict[str, Any],
    base_classifier: Optional[Any],
) -> Any:
    """Create a fresh ``xgboost.XGBClassifier`` instance."""
    cls = type(base_classifier) if base_classifier is not None else xgb.XGBClassifier
    return cls(**params)


def save_xgboost_classifier(
    xgb_classifier: Any,
    out_file: Union[str, os.PathLike],
) -> None:
    """
    Save a fitted ``xgboost.XGBClassifier`` to disk.

    Files ending in ``.joblib`` or ``.pkl`` are serialised with ``joblib``;
    all other extensions (such as ``.json`` or ``.ubj``) use XGBoost's native
    ``save_model()`` format.

    :param xgb_classifier: Trained ``xgboost.XGBClassifier`` object.
    :param out_file: Output model file path (e.g., ``'xgb_model.json'`` or
                     ``'xgb_model.joblib'``).

    .. code:: python

        import rsgispy.classification.classxgboost as cxgb

        cxgb.save_xgboost_classifier(clf, "xgb_model.json")
    """
    save_xgb_or_cb_model(
        model=xgb_classifier,
        out_file=out_file,
        have_pkg=HAVE_XGBOOST,
        pkg_name="xgboost",
        extra="xgboost",
        func_name="save_xgboost_classifier",
        error_detail=_XGBOOST_LOAD_ERROR,
    )


def load_xgboost_classifier(
    in_file: Union[str, os.PathLike],
) -> Any:
    """
    Load a saved ``xgboost.XGBClassifier`` from disk.

    Supports both ``joblib`` archives (``.joblib`` / ``.pkl``) and native
    XGBoost model files (``.json`` / ``.ubj`` / ``.bin``).

    :param in_file: Input model file path.
    :return: Loaded ``xgboost.XGBClassifier`` instance.

    .. code:: python

        import rsgispy.classification.classxgboost as cxgb

        clf = cxgb.load_xgboost_classifier("xgb_model.json")
    """
    return load_xgb_or_cb_model(
        in_file=in_file,
        model_factory=lambda: xgb.XGBClassifier(),
        have_pkg=HAVE_XGBOOST,
        pkg_name="xgboost",
        extra="xgboost",
        func_name="load_xgboost_classifier",
        label="XGBoost model",
        error_detail=_XGBOOST_LOAD_ERROR,
    )


_XGB_BACKEND = register_gbdt_backend(
    "classification",
    "xgboost",
    GBDTBackendSpec(
        name="XGBoost",
        package_name="xgboost",
        extra="xgboost",
        is_available=lambda: HAVE_XGBOOST,
        gpu_available=is_xgboost_gpu_available,
        apply_gpu_params=_apply_xgb_gpu_params,
        seed_param="random_state",
        jobs_param="n_jobs",
        default_binary_params=_default_xgb_binary_params,
        default_multiclass_params=_default_xgb_multiclass_params,
        default_binary_space=lambda: get_xgboost_optuna_param_space(
            extended=False, binary=True
        ),
        default_multiclass_space=lambda: get_xgboost_optuna_param_space(
            extended=False, binary=False
        ),
        extended_binary_space=lambda: get_xgboost_optuna_param_space(
            extended=True, binary=True
        ),
        extended_multiclass_space=lambda: get_xgboost_optuna_param_space(
            extended=True, binary=False
        ),
        build_estimator=_build_xgb_estimator,
        fit_estimator=_fit_xgb_estimator,
        load_model=load_xgboost_classifier,
        save_model=save_xgboost_classifier,
        prepare_predict_estimator=_prepare_xgb_predict_estimator,
        require_consecutive_zero_labels=True,
        get_error_detail=lambda: _XGBOOST_LOAD_ERROR,
    ),
)


def optimise_xgboost_binary_classifier(
    cls_cfg: ClsConfigInput,
    train_data: SamplesInput,
    valid_data: Optional[SamplesInput] = None,
    param_space: Optional[
        Union[str, Mapping[str, Any], Callable[[Any], Mapping[str, Any]]]
    ] = None,
    extended_param_space: bool = False,
    fixed_params: Optional[Mapping[str, Any]] = None,
    xgb_classifier: Optional[Any] = None,
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
    Optimise hyperparameters for a binary ``xgboost.XGBClassifier``
    (``objective='binary:logistic'``) using Bayesian optimisation with Optuna.

    By default (``param_space=None``, ``extended_param_space=False``), optimises
    the 9 core XGBoost parameters (see :func:`get_xgboost_optuna_param_space`).
    Set ``extended_param_space=True`` (or ``param_space='extended'``) to
    optimise an expanded 16-parameter space including ``colsample_bylevel``,
    ``colsample_bynode``, ``max_delta_step``, ``max_bin``, ``grow_policy``,
    ``max_leaves``, and ``scale_pos_weight``.

    :param cls_cfg: :class:`~rsgispy.classification.classdata.ClassificationConfig`,
                    JSON config path, or mapping of ``{class_name: ClassInfoObj}``
                    defining exactly 2 classes.
    :param train_data: Training sample :class:`pandas.DataFrame` or file path.
    :param valid_data: Optional validation sample table for out-of-sample
                       scoring and optional early stopping. When ``None``,
                       stratified ``cv``-fold cross-validation is used.
    :param param_space: Optional Optuna search space dictionary,
                        ``fn(trial) -> dict`` callable, or preset name
                        (``'default'`` / ``'extended'``). Uses built-in defaults
                        when ``None``.
    :param extended_param_space: If ``True`` (and ``param_space`` is ``None``),
                                 use the larger extended XGBoost search space
                                 (Default = ``False``).
    :param fixed_params: Optional fixed ``XGBClassifier`` keyword parameters
                         merged into every trial (e.g., ``tree_method='hist'``).
    :param xgb_classifier: Optional base ``XGBClassifier`` template whose
                           configured attributes provide parameter defaults.
    :param scoring: Scikit-learn scoring name or callable (Default = ``'roc_auc'``).
    :param cv: Cross-validation folds or splitter when ``valid_data`` is
               ``None`` (Default = ``5``).
    :param n_trials: Number of Optuna trials (Default = ``50``).
    :param timeout: Optional Optuna time budget in seconds.
    :param direction: Optuna direction (``'maximize'`` or ``'minimize'``).
                      Named scikit-learn scorers are greater-is-better, so
                      ``'minimize'`` is only valid with a callable ``scoring``.
    :param early_stopping_rounds: Optional early stopping rounds used when
                                  ``valid_data`` is supplied.
    :param n_jobs: Number of parallel threads/workers.
    :param rnd_seed: Optional random seed for reproducibility.
    :param feature_cols: Optional ordered feature column names.
    :param use_internal_id: Use consecutive 0-based ``internal_cls_id`` labels
                            (Default = ``True``).
    :param refit: Refit and return a trained model (Default = ``True``).
    :param out_id_col: Optional override for the output class ID column name.
    :param internal_id_col: Optional override for the internal class ID column.
    :param out_results_file: Optional Parquet/CSV path to save Optuna trials.
    :param out_params_json: Optional JSON path to save best parameters and score.
    :param out_model_file: Optional output path to save the refitted model.
    :param return_results: When ``True``, return ``(best_model, results_dict)``.
    :param verbose: Verbosity level (Default = ``0``).
    :param use_gpu: If ``True``, use CUDA GPU acceleration (``tree_method='hist'``,
                    ``device='cuda'``) when available, falling back to CPU with
                    a warning otherwise (Default = ``False``).
    :param gpu_device_id: Optional non-negative CUDA GPU device index
                          (e.g., ``0`` -> ``device='cuda:0'``).
    :return: Optimised ``XGBClassifier``, or ``(best_model, results_dict)``
             when ``return_results=True``.

    .. code:: python

        import rsgispy.classification.classxgboost as cxgb

        best_clf = cxgb.optimise_xgboost_binary_classifier(
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
    return forward_gbdt_call(
        optimise_gbdt_classifier,
        _XGB_BACKEND,
        "optimise_xgboost_binary_classifier",
        locals(),
        rename={"xgb_classifier": "classifier"},
        binary=True,
    )


def optimise_xgboost_multiclass_classifier(
    cls_cfg: ClsConfigInput,
    train_data: SamplesInput,
    valid_data: Optional[SamplesInput] = None,
    param_space: Optional[
        Union[str, Mapping[str, Any], Callable[[Any], Mapping[str, Any]]]
    ] = None,
    extended_param_space: bool = False,
    fixed_params: Optional[Mapping[str, Any]] = None,
    xgb_classifier: Optional[Any] = None,
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
    Optimise hyperparameters for a multi-class ``xgboost.XGBClassifier``
    (``objective='multi:softprob'``) using Bayesian optimisation with Optuna.

    By default (``param_space=None``, ``extended_param_space=False``), optimises
    the 9 core XGBoost parameters (see :func:`get_xgboost_optuna_param_space`).
    Set ``extended_param_space=True`` (or ``param_space='extended'``) to
    optimise an expanded 15-parameter space including ``colsample_bylevel``,
    ``colsample_bynode``, ``max_delta_step``, ``max_bin``, ``grow_policy``,
    and ``max_leaves``.

    :param cls_cfg: :class:`~rsgispy.classification.classdata.ClassificationConfig`,
                    JSON config path, or mapping of ``{class_name: ClassInfoObj}``
                    defining 2 or more classes.
    :param train_data: Training sample :class:`pandas.DataFrame` or file path.
    :param valid_data: Optional validation sample table for out-of-sample
                       scoring and optional early stopping.
    :param param_space: Optional Optuna search space dictionary, callable, or
                        preset name (``'default'`` / ``'extended'``).
    :param extended_param_space: If ``True`` (and ``param_space`` is ``None``),
                                 use the larger extended XGBoost search space
                                 (Default = ``False``).
    :param fixed_params: Optional fixed ``XGBClassifier`` parameters.
    :param xgb_classifier: Optional base ``XGBClassifier`` template.
    :param scoring: Scikit-learn scoring metric or callable (Default = ``'accuracy'``).
    :param cv: Cross-validation folds or splitter (Default = ``5``).
    :param n_trials: Number of Optuna trials (Default = ``50``).
    :param timeout: Optional Optuna time budget in seconds.
    :param direction: Optuna direction (``'maximize'`` or ``'minimize'``).
                      Named scikit-learn scorers are greater-is-better, so
                      ``'minimize'`` is only valid with a callable ``scoring``.
    :param early_stopping_rounds: Optional early stopping rounds when
                                  ``valid_data`` is provided.
    :param n_jobs: Number of parallel threads/workers.
    :param rnd_seed: Optional random seed for reproducibility.
    :param feature_cols: Optional ordered feature column names.
    :param use_internal_id: Use consecutive 0-based ``internal_cls_id`` labels
                            (Default = ``True``).
    :param refit: Refit and return a trained model (Default = ``True``).
    :param out_id_col: Optional override for the output class ID column name.
    :param internal_id_col: Optional override for the internal class ID column.
    :param out_results_file: Optional path to save Optuna trials table.
    :param out_params_json: Optional JSON path to save best parameters.
    :param out_model_file: Optional path to save refitted model.
    :param return_results: When ``True``, return ``(best_model, results_dict)``.
    :param verbose: Verbosity level (Default = ``0``).
    :param use_gpu: If ``True``, use CUDA GPU acceleration when available,
                    falling back to CPU with a warning otherwise
                    (Default = ``False``).
    :param gpu_device_id: Optional non-negative CUDA GPU device index.
    :return: Optimised ``XGBClassifier``, or ``(best_model, results_dict)``
             when ``return_results=True``.

    .. code:: python

        import rsgispy.classification.classxgboost as cxgb

        best_clf = cxgb.optimise_xgboost_multiclass_classifier(
            cls_cfg="cls_config.json",
            train_data="train.parquet",
            extended_param_space=True,
            n_trials=50,
            rnd_seed=42,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    return forward_gbdt_call(
        optimise_gbdt_classifier,
        _XGB_BACKEND,
        "optimise_xgboost_multiclass_classifier",
        locals(),
        rename={"xgb_classifier": "classifier"},
        binary=False,
    )


def train_xgboost_binary_classifier(
    cls_cfg: ClsConfigInput,
    train_data: SamplesInput,
    test_data: Optional[SamplesInput] = None,
    valid_data: Optional[SamplesInput] = None,
    xgb_classifier: Optional[Any] = None,
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
    Train a binary ``xgboost.XGBClassifier`` on extracted sample tables and
    evaluate accuracy, precision, recall, F1, ROC-AUC, kappa, and log-loss on
    training, optional test, and optional validation splits.

    :param cls_cfg: Binary :class:`~rsgispy.classification.classdata.ClassificationConfig`,
                    JSON config path, or 2-class mapping.
    :param train_data: Training sample :class:`pandas.DataFrame` or file path.
    :param test_data: Optional testing sample table.
    :param valid_data: Optional validation sample table (also used for early
                       stopping when ``early_stopping_rounds`` is set).
    :param xgb_classifier: Optional pre-configured ``XGBClassifier`` template.
    :param model_params: Optional parameter dictionary (e.g., from Optuna).
    :param early_stopping_rounds: Optional early stopping rounds on ``valid_data``.
    :param feature_cols: Optional ordered feature column names.
    :param use_internal_id: Train on consecutive 0/1 ``internal_cls_id`` labels
                            (Default = ``True``).
    :param rnd_seed: Optional random seed.
    :param n_jobs: Number of parallel threads.
    :param out_id_col: Optional override for the output class ID column name.
    :param internal_id_col: Optional override for the internal class ID column.
    :param out_model_file: Optional path (``.json`` / ``.joblib``) to save model.
    :param out_metrics_file: Optional JSON path to save split metrics.
    :param return_model: When ``True`` (default), return ``(fitted_model, metrics)``;
                         when ``False``, update ``xgb_classifier`` in place if
                         supplied and return ``metrics``.
    :param verbose: Verbosity level (Default = ``0``).
    :param use_gpu: If ``True``, use CUDA GPU acceleration when available,
                    falling back to CPU with a warning otherwise
                    (Default = ``False``).
    :param gpu_device_id: Optional non-negative CUDA GPU device index.
    :return: ``(fitted_model, metrics)`` or ``metrics``.

    .. code:: python

        import rsgispy.classification.classxgboost as cxgb

        model, metrics = cxgb.train_xgboost_binary_classifier(
            cls_cfg="cls_binary.json",
            train_data="train.parquet",
            test_data="test.parquet",
            model_params={"n_estimators": 100, "max_depth": 4},
            rnd_seed=42,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    return forward_gbdt_call(
        train_gbdt_classifier,
        _XGB_BACKEND,
        "train_xgboost_binary_classifier",
        locals(),
        rename={"xgb_classifier": "classifier"},
        binary=True,
    )


def train_xgboost_multiclass_classifier(
    cls_cfg: ClsConfigInput,
    train_data: SamplesInput,
    test_data: Optional[SamplesInput] = None,
    valid_data: Optional[SamplesInput] = None,
    xgb_classifier: Optional[Any] = None,
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
    Train a multi-class ``xgboost.XGBClassifier`` (``objective='multi:softprob'``)
    on extracted sample tables and evaluate on optional test and validation sets.

    :param cls_cfg: Multi-class :class:`~rsgispy.classification.classdata.ClassificationConfig`,
                    JSON config path, or class mapping.
    :param train_data: Training sample :class:`pandas.DataFrame` or file path.
    :param test_data: Optional testing sample table.
    :param valid_data: Optional validation sample table.
    :param xgb_classifier: Optional pre-configured ``XGBClassifier`` template.
    :param model_params: Optional parameter dictionary.
    :param early_stopping_rounds: Optional early stopping rounds on ``valid_data``.
    :param feature_cols: Optional ordered feature column names.
    :param use_internal_id: Train on consecutive 0-based ``internal_cls_id`` labels
                            (Default = ``True``).
    :param rnd_seed: Optional random seed.
    :param n_jobs: Number of parallel threads.
    :param out_id_col: Optional override for the output class ID column name.
    :param internal_id_col: Optional override for the internal class ID column.
    :param out_model_file: Optional path (``.json`` / ``.joblib``) to save model.
    :param out_metrics_file: Optional JSON path to save split metrics.
    :param return_model: When ``True`` (default), return ``(fitted_model, metrics)``.
    :param verbose: Verbosity level (Default = ``0``).
    :param use_gpu: If ``True``, use CUDA GPU acceleration when available,
                    falling back to CPU with a warning otherwise
                    (Default = ``False``).
    :param gpu_device_id: Optional non-negative CUDA GPU device index.
    :return: ``(fitted_model, metrics)`` or ``metrics``.

    .. code:: python

        import rsgispy.classification.classxgboost as cxgb

        model, metrics = cxgb.train_xgboost_multiclass_classifier(
            cls_cfg="cls_config.json",
            train_data="train.parquet",
            test_data="test.parquet",
            rnd_seed=42,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    return forward_gbdt_call(
        train_gbdt_classifier,
        _XGB_BACKEND,
        "train_xgboost_multiclass_classifier",
        locals(),
        rename={"xgb_classifier": "classifier"},
        binary=False,
    )


def apply_xgboost_binary_classifier(
    cls_cfg: ClsConfigInput,
    xgb_classifier: Union[Any, str, os.PathLike],
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
    Apply a trained binary ``xgboost.XGBClassifier`` block-by-block to pixels
    where ``in_msk_img == img_msk_val``.

    Outputs a thematic class ID raster (thresholded at ``pred_threshold`` and
    automatically recoded to user ``out_id`` values) and/or a ``float32``
    positive-class probability score raster.

    :param cls_cfg: Binary :class:`~rsgispy.classification.classdata.ClassificationConfig`,
                    JSON config path, or 2-class mapping.
    :param xgb_classifier: Fitted ``XGBClassifier`` or path to saved model.
    :param in_msk_img: Mask image path.
    :param img_msk_val: Mask pixel value defining pixels to classify.
    :param output_img: Optional output classified ID raster path.
    :param out_score_img: Optional output ``float32`` probability score raster.
    :param img_band_info: Optional sequence of
                          :class:`~rsgispy.classification.classdata.ImageBandInfo`.
    :param pos_class: Optional positive class name or ID (Default = ``None``,
                      selecting the higher internal/output class ID).
    :param pred_threshold: Probability threshold in ``(0, 1)`` applied to the
                           positive class probability (Default = ``0.5``).
    :param score_all_classes: If ``False`` (default), write a single-band
                              positive-class probability raster; if ``True``,
                              write 2 probability bands (negative, positive).
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
    :param use_gpu: If ``True``, run CUDA GPU prediction when available,
                    falling back to CPU with a warning otherwise
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

        import rsgispy.classification.classxgboost as cxgb

        cxgb.apply_xgboost_binary_classifier(
            cls_cfg="cls_binary.json",
            xgb_classifier=model,
            in_msk_img="mask.tif",
            img_msk_val=1,
            output_img="xgb_binary_cls.tif",
            out_score_img="xgb_binary_prob.tif",
            pred_threshold=0.5,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    forward_gbdt_call(
        apply_gbdt_classifier,
        _XGB_BACKEND,
        "apply_xgboost_binary_classifier",
        locals(),
        rename={"xgb_classifier": "classifier"},
        binary=True,
        score_max_only=False,
    )


def apply_xgboost_multiclass_classifier(
    cls_cfg: ClsConfigInput,
    xgb_classifier: Union[Any, str, os.PathLike],
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
    Apply a trained multi-class ``xgboost.XGBClassifier`` block-by-block to
    pixels where ``in_msk_img == img_msk_val``, outputting classified class IDs
    and/or per-class (or maximum) probability score rasters.

    :param cls_cfg: Multi-class :class:`~rsgispy.classification.classdata.ClassificationConfig`,
                    JSON config path, or class mapping.
    :param xgb_classifier: Fitted ``XGBClassifier`` or path to saved model.
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
    :param use_gpu: If ``True``, run CUDA GPU prediction when available,
                    falling back to CPU with a warning otherwise
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

        import rsgispy.classification.classxgboost as cxgb

        cxgb.apply_xgboost_multiclass_classifier(
            cls_cfg="cls_config.json",
            xgb_classifier=model,
            in_msk_img="mask.tif",
            img_msk_val=1,
            output_img="xgb_multiclass.tif",
            out_score_img="xgb_probs.tif",
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    forward_gbdt_call(
        apply_gbdt_classifier,
        _XGB_BACKEND,
        "apply_xgboost_multiclass_classifier",
        locals(),
        rename={"xgb_classifier": "classifier"},
        binary=False,
    )
