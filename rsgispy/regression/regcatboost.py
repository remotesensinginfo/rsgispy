"""
CatBoost univariate and multi-target (``MultiRMSE``) regression workflows built
on :mod:`catboost`, Optuna Bayesian hyperparameter optimisation, model
persistence, and block-wise raster regression.
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

from rsgispy.imageutils.masking import ImageBandInfo
from rsgispy.regression._reg_common import (
    GBDTRegBackendSpec,
    RegConfigInput,
    SamplesInput,
    TargetColsInput,
    apply_gbdt_regressor,
    optimise_gbdt_regressor,
    train_gbdt_regressor,
)
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

__all__ = [
    "is_catboost_gpu_available",
    "save_catboost_regressor",
    "load_catboost_regressor",
    "get_catboost_optuna_param_space",
    "optimise_catboost_regressor",
    "train_catboost_regressor",
    "apply_catboost_regressor",
]


def is_catboost_gpu_available(gpu_device_id: Optional[int] = None) -> bool:
    """
    Check whether CUDA GPU execution is supported and available for
    ``catboost``.

    :param gpu_device_id: Optional non-negative GPU device index
                          (Default = ``None``).
    :return: ``True`` if CatBoost detects a usable GPU, ``False`` otherwise.

    .. code:: python

        import rsgispy.regression.regcatboost as rcb

        if rcb.is_catboost_gpu_available():
            print("CatBoost GPU ready")
    """
    validate_gpu_device_id(gpu_device_id)
    if not HAVE_CATBOOST:
        return False
    return _tools_is_catboost_gpu_available(gpu_device_id)


def get_catboost_optuna_param_space(
    extended: bool = False,
) -> Union[Dict[str, Any], Callable[[Any], Dict[str, Any]]]:
    """
    Return the built-in Optuna hyperparameter search space for
    ``catboost.CatBoostRegressor``.

    :param extended: If ``True``, return the extended conditional search space
                     callable; if ``False`` (default), return the 8 core
                     key parameters.
    :return: Optuna search space dictionary or conditional ``fn(trial)`` callable.

    .. code:: python

        import rsgispy.regression.regcatboost as rcb

        space = rcb.get_catboost_optuna_param_space()
    """
    return _tools_get_catboost_optuna_param_space(
        extended=extended, task="regression"
    )


def _default_cb_reg_params(n_targets: int, verbose: int) -> Dict[str, Any]:
    """Return default CatBoostRegressor parameters (MultiRMSE when multi-target)."""
    return {
        "loss_function": "MultiRMSE" if n_targets > 1 else "RMSE",
        "iterations": 200,
        "verbose": False if verbose <= 0 else verbose,
        "allow_writing_files": False,
    }


def _build_cb_reg_estimator(
    params: Dict[str, Any],
    base_regressor: Optional[Any],
    _n_targets: int,
) -> Any:
    """Create a fresh ``catboost.CatBoostRegressor`` instance."""
    cls = (
        type(base_regressor)
        if base_regressor is not None
        else cb.CatBoostRegressor
    )
    return cls(**params)


def save_catboost_regressor(
    cb_regressor: Any,
    out_file: Union[str, os.PathLike],
) -> None:
    """
    Save a fitted ``catboost.CatBoostRegressor`` to disk (``.joblib``/``.pkl``
    or native ``.cbm``/``.json``).

    :param cb_regressor: Trained ``catboost.CatBoostRegressor`` object.
    :param out_file: Output model file path.

    .. code:: python

        import rsgispy.regression.regcatboost as rcb

        rcb.save_catboost_regressor(reg, "cb_reg.cbm")
    """
    save_xgb_or_cb_model(
        model=cb_regressor,
        out_file=out_file,
        have_pkg=HAVE_CATBOOST,
        pkg_name="catboost",
        extra="catboost",
        func_name="save_catboost_regressor",
        error_detail=_CATBOOST_LOAD_ERROR,
    )


def load_catboost_regressor(
    in_file: Union[str, os.PathLike],
) -> Any:
    """
    Load a saved ``catboost.CatBoostRegressor`` from disk.

    :param in_file: Input model file path.
    :return: Loaded ``catboost.CatBoostRegressor`` instance.

    .. code:: python

        import rsgispy.regression.regcatboost as rcb

        reg = rcb.load_catboost_regressor("cb_reg.cbm")
    """
    return load_xgb_or_cb_model(
        in_file=in_file,
        model_factory=lambda: cb.CatBoostRegressor(),
        have_pkg=HAVE_CATBOOST,
        pkg_name="catboost",
        extra="catboost",
        func_name="load_catboost_regressor",
        label="CatBoost regressor model",
        error_detail=_CATBOOST_LOAD_ERROR,
    )


_CB_REG_BACKEND = GBDTRegBackendSpec(
    name="CatBoost",
    package_name="catboost",
    extra="catboost",
    is_available=lambda: HAVE_CATBOOST,
    gpu_available=is_catboost_gpu_available,
    apply_gpu_params=_apply_cb_gpu_params,
    seed_param="random_seed",
    jobs_param="thread_count",
    default_params=_default_cb_reg_params,
    default_space=lambda: get_catboost_optuna_param_space(extended=False),
    extended_space=lambda: get_catboost_optuna_param_space(extended=True),
    build_estimator=_build_cb_reg_estimator,
    fit_estimator=_fit_cb_estimator,
    load_model=load_catboost_regressor,
    save_model=save_catboost_regressor,
    prepare_predict_estimator=_prepare_cb_predict_estimator,
    get_error_detail=lambda: _CATBOOST_LOAD_ERROR,
)


def optimise_catboost_regressor(
    reg_cfg: Optional[RegConfigInput],
    train_data: SamplesInput,
    valid_data: Optional[SamplesInput] = None,
    target_cols: TargetColsInput = None,
    param_space: Optional[
        Union[str, Mapping[str, Any], Callable[[Any], Mapping[str, Any]]]
    ] = None,
    extended_param_space: bool = False,
    fixed_params: Optional[Mapping[str, Any]] = None,
    cb_regressor: Optional[Any] = None,
    scoring: Union[str, Callable[..., float]] = "r2",
    cv: Union[int, Any] = 5,
    n_trials: int = 50,
    timeout: Optional[float] = None,
    direction: str = "maximize",
    early_stopping_rounds: Optional[int] = None,
    n_jobs: Optional[int] = None,
    rnd_seed: Optional[int] = None,
    feature_cols: Optional[Sequence[str]] = None,
    refit: bool = True,
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
    Optimise hyperparameters for a univariate or multi-target
    ``catboost.CatBoostRegressor`` using Bayesian optimisation with Optuna.

    :param reg_cfg: :class:`~rsgispy.regression.regdata.RegressionConfig`, JSON
                    path, sequence of ``ImageBandInfo``, or ``None``.
    :param train_data: Training sample :class:`pandas.DataFrame` or file path.
    :param valid_data: Optional validation :class:`pandas.DataFrame` or path.
    :param target_cols: Optional target column name(s) overriding ``reg_cfg``.
    :param param_space: Optional Optuna space dict, callable, or preset
                        (``'default'`` / ``'extended'``).
    :param extended_param_space: Use expanded conditional search space when ``True``.
    :param fixed_params: Optional fixed CatBoost parameters merged into trials.
    :param cb_regressor: Optional base ``catboost.CatBoostRegressor`` template.
    :param scoring: Scikit-learn scoring metric (Default = ``'r2'``).
    :param cv: Cross-validation folds when ``valid_data`` is ``None``
               (Default = ``5``).
    :param n_trials: Number of Optuna trials (Default = ``50``).
    :param timeout: Optional Optuna study timeout in seconds.
    :param direction: Optimisation direction (Default = ``'maximize'``).
    :param early_stopping_rounds: Early stopping rounds when ``valid_data`` is
                                  provided.
    :param n_jobs: Number of parallel threads/jobs.
    :param rnd_seed: Optional random seed for reproducibility.
    :param feature_cols: Optional ordered feature column names.
    :param refit: Refit best model on ``train_data`` when ``True`` (default).
    :param out_results_file: Optional path saving Optuna trials table.
    :param out_params_json: Optional JSON path saving best parameters/score.
    :param out_model_file: Optional path saving fitted best model.
    :param return_results: Return ``(best_model, results_dict)`` when ``True``.
    :param verbose: Verbosity level (Default = ``0``).
    :param use_gpu: Attempt CUDA GPU acceleration (falls back to CPU with warning
                    if unavailable).
    :param gpu_device_id: Optional CUDA device index.
    :return: Optimised ``CatBoostRegressor`` (or ``(best_model, results_dict)``).

    .. code:: python

        import rsgispy.regression.regcatboost as rcb

        best_cb = rcb.optimise_catboost_regressor(
            "reg_config.json", "train.parquet", valid_data="valid.parquet",
            n_trials=30, early_stopping_rounds=20, rnd_seed=42,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    return optimise_gbdt_regressor(
        backend=_CB_REG_BACKEND,
        func_name="optimise_catboost_regressor",
        reg_cfg=reg_cfg,
        train_data=train_data,
        valid_data=valid_data,
        target_cols=target_cols,
        param_space=param_space,
        extended_param_space=extended_param_space,
        fixed_params=fixed_params,
        regressor=cb_regressor,
        scoring=scoring,
        cv=cv,
        n_trials=n_trials,
        timeout=timeout,
        direction=direction,
        early_stopping_rounds=early_stopping_rounds,
        n_jobs=n_jobs,
        rnd_seed=rnd_seed,
        feature_cols=feature_cols,
        refit=refit,
        out_results_file=out_results_file,
        out_params_json=out_params_json,
        out_model_file=out_model_file,
        return_results=return_results,
        verbose=verbose,
        use_gpu=use_gpu,
        gpu_device_id=gpu_device_id,
        use_pbar=use_pbar,
    )


def train_catboost_regressor(
    reg_cfg: Optional[RegConfigInput],
    train_data: SamplesInput,
    test_data: Optional[SamplesInput] = None,
    valid_data: Optional[SamplesInput] = None,
    target_cols: TargetColsInput = None,
    cb_regressor: Optional[Any] = None,
    model_params: Optional[Mapping[str, Any]] = None,
    early_stopping_rounds: Optional[int] = None,
    feature_cols: Optional[Sequence[str]] = None,
    rnd_seed: Optional[int] = None,
    n_jobs: Optional[int] = None,
    out_model_file: Optional[Union[str, os.PathLike]] = None,
    out_metrics_file: Optional[Union[str, os.PathLike]] = None,
    return_model: bool = True,
    verbose: int = 0,
    use_gpu: bool = False,
    gpu_device_id: Optional[int] = None,
) -> Union[Dict[str, float], Tuple[Any, Dict[str, float]]]:
    """
    Train a univariate or multi-target ``catboost.CatBoostRegressor`` and
    evaluate on optional testing and validation splits.

    :param reg_cfg: Regression configuration, JSON path, or image-band sequence.
    :param train_data: Training sample :class:`pandas.DataFrame` or file path.
    :param test_data: Optional testing sample DataFrame or file path.
    :param valid_data: Optional validation sample DataFrame or file path.
    :param target_cols: Optional target column name(s) overriding ``reg_cfg``.
    :param cb_regressor: Optional pre-configured ``CatBoostRegressor`` template.
    :param model_params: Optional hyperparameter overrides mapping.
    :param early_stopping_rounds: Optional early stopping rounds with ``valid_data``.
    :param feature_cols: Optional ordered feature column names.
    :param rnd_seed: Optional random seed.
    :param n_jobs: Optional CPU thread count.
    :param out_model_file: Optional model output path.
    :param out_metrics_file: Optional metrics JSON output path.
    :param return_model: Return ``(fitted_model, metrics)`` when ``True``.
    :param verbose: Verbosity level (Default = ``0``).
    :param use_gpu: Use CUDA GPU acceleration if available.
    :param gpu_device_id: Optional CUDA GPU device index.
    :return: ``(fitted_model, metrics)`` or ``metrics``.

    .. code:: python

        import rsgispy.regression.regcatboost as rcb

        model, metrics = rcb.train_catboost_regressor(
            "reg_config.json", "train.parquet", test_data="test.parquet", rnd_seed=42
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    return train_gbdt_regressor(
        backend=_CB_REG_BACKEND,
        func_name="train_catboost_regressor",
        reg_cfg=reg_cfg,
        train_data=train_data,
        test_data=test_data,
        valid_data=valid_data,
        target_cols=target_cols,
        regressor=cb_regressor,
        model_params=model_params,
        early_stopping_rounds=early_stopping_rounds,
        feature_cols=feature_cols,
        rnd_seed=rnd_seed,
        n_jobs=n_jobs,
        out_model_file=out_model_file,
        out_metrics_file=out_metrics_file,
        return_model=return_model,
        verbose=verbose,
        use_gpu=use_gpu,
        gpu_device_id=gpu_device_id,
    )


def apply_catboost_regressor(
    reg_cfg: Optional[RegConfigInput],
    cb_regressor: Union[Any, str, os.PathLike],
    in_msk_img: Union[str, os.PathLike],
    img_msk_val: int,
    out_img: Union[str, os.PathLike],
    img_band_info: Optional[Sequence[ImageBandInfo]] = None,
    target_cols: TargetColsInput = None,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    out_no_data_val: float = -9999.0,
    calc_img_stats: bool = True,
    block_size: Optional[int] = None,
    creation_opts: Optional[Mapping[str, Any]] = None,
    use_pbar: bool = True,
    use_gpu: bool = False,
    gpu_device_id: Optional[int] = None,
) -> None:
    """
    Apply a trained ``catboost.CatBoostRegressor`` block-by-block to raster
    imagery.

    :param reg_cfg: Regression configuration, JSON path, or image-band sequence.
    :param cb_regressor: Fitted ``CatBoostRegressor`` instance or saved model path.
    :param in_msk_img: Mask raster path.
    :param img_msk_val: Mask pixel value to predict.
    :param out_img: Output continuous raster path.
    :param img_band_info: Optional sequence of :class:`ImageBandInfo`.
    :param target_cols: Optional target variable name(s) for band descriptions.
    :param gdalformat: Output GDAL driver format (Default = ``'GTIFF'``).
    :param datatype: Optional ``rsgispy.TYPE_*`` datatype.
    :param out_no_data_val: Nodata fill value (Default = ``-9999.0``).
    :param calc_img_stats: Compute statistics and pyramids (Default = ``True``).
    :param block_size: Optional block size in pixels.
    :param creation_opts: Optional GDAL creation options mapping.
    :param use_pbar: Display a tqdm progress bar (Default = ``True``).
    :param use_gpu: Use CUDA GPU inference if available.
    :param gpu_device_id: Optional CUDA GPU device index.

    .. code:: python

        import rsgispy.regression.regcatboost as rcb

        rcb.apply_catboost_regressor(
            "reg_config.json", "cb_reg.cbm", "mask.tif", 1, "canopy_cb.tif"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    apply_gbdt_regressor(
        backend=_CB_REG_BACKEND,
        func_name="apply_catboost_regressor",
        reg_cfg=reg_cfg,
        regressor=cb_regressor,
        in_msk_img=in_msk_img,
        img_msk_val=img_msk_val,
        out_img=out_img,
        img_band_info=img_band_info,
        target_cols=target_cols,
        gdalformat=gdalformat,
        datatype=datatype,
        out_no_data_val=out_no_data_val,
        calc_img_stats=calc_img_stats,
        block_size=block_size,
        creation_opts=creation_opts,
        use_pbar=use_pbar,
        use_gpu=use_gpu,
        gpu_device_id=gpu_device_id,
    )
