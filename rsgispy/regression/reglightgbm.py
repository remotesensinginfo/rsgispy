"""
LightGBM univariate and multivariate regression workflows built on
:mod:`lightgbm`, Optuna Bayesian hyperparameter optimisation, model persistence,
and block-wise raster regression.
"""

import os
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd

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
    apply_lgb_gpu_params as _apply_lgb_gpu_params,
    fit_lgb_estimator as _fit_lgb_estimator,
    get_lightgbm_optuna_param_space as _tools_get_lightgbm_optuna_param_space,
    is_lightgbm_gpu_available as _tools_is_lightgbm_gpu_available,
    load_joblib_model,
    safe_import_optional,
    save_joblib_model,
    validate_gpu_device_id,
)
from rsgispy.tools._sklearn import clone_sklearn_estimator, sklearn

lgb, HAVE_LIGHTGBM, _LIGHTGBM_LOAD_ERROR = safe_import_optional("lightgbm")

__all__ = [
    "is_lightgbm_gpu_available",
    "save_lightgbm_regressor",
    "load_lightgbm_regressor",
    "get_lightgbm_optuna_param_space",
    "optimise_lightgbm_regressor",
    "train_lightgbm_regressor",
    "apply_lightgbm_regressor",
]


def is_lightgbm_gpu_available(gpu_device_id: Optional[int] = None) -> bool:
    """
    Check whether CUDA or OpenCL GPU execution is supported and available for
    ``lightgbm``.

    :param gpu_device_id: Optional non-negative GPU device index
                          (Default = ``None``).
    :return: ``True`` if LightGBM GPU training is available, ``False`` otherwise.

    .. code:: python

        import rsgispy.regression.reglightgbm as rlgb

        if rlgb.is_lightgbm_gpu_available():
            print("LightGBM GPU ready")
    """
    validate_gpu_device_id(gpu_device_id)
    if not HAVE_LIGHTGBM:
        return False
    return _tools_is_lightgbm_gpu_available(gpu_device_id)


def get_lightgbm_optuna_param_space(extended: bool = False) -> Dict[str, Any]:
    """
    Return the built-in Optuna hyperparameter search space for
    ``lightgbm.LGBMRegressor``.

    :param extended: If ``True``, return the larger extended parameter space;
                     if ``False`` (default), return the 10 core parameters.
    :return: Dictionary of Optuna hyperparameter search specifications.

    .. code:: python

        import rsgispy.regression.reglightgbm as rlgb

        space = rlgb.get_lightgbm_optuna_param_space()
    """
    return _tools_get_lightgbm_optuna_param_space(
        extended=extended, task="regression"
    )


def _default_lgb_reg_params(_n_targets: int, verbose: int) -> Dict[str, Any]:
    """Return default LGBMRegressor parameters."""
    return {
        "objective": "regression",
        "n_estimators": 200,
        "verbose": -1 if verbose <= 0 else verbose,
    }


def _build_lgb_reg_estimator(
    params: Dict[str, Any],
    base_regressor: Optional[Any],
    n_targets: int,
) -> Any:
    """
    Create a fresh ``lightgbm.LGBMRegressor`` (or ``MultiOutputRegressor``
    wrapping one when ``n_targets > 1``).
    """
    unwrapped = base_regressor
    if (
        unwrapped is not None
        and type(unwrapped).__name__ == "MultiOutputRegressor"
        and hasattr(unwrapped, "estimator")
    ):
        unwrapped = unwrapped.estimator
    cls = type(unwrapped) if unwrapped is not None else lgb.LGBMRegressor
    est = cls(**params)
    if n_targets > 1:
        return sklearn.multioutput.MultiOutputRegressor(est)
    return est


def _fit_lgb_reg_estimator(
    est: Any,
    train_x: pd.DataFrame,
    train_y: np.ndarray,
    valid_x: Optional[pd.DataFrame],
    valid_y: Optional[np.ndarray],
    early_stopping_rounds: Optional[int],
) -> Any:
    """
    Fit ``LGBMRegressor`` or ``MultiOutputRegressor(LGBMRegressor)`` with
    optional validation early stopping per output target.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    if type(est).__name__ == "MultiOutputRegressor":
        n_targets = int(train_y.shape[1])
        fitted_subs: List[Any] = []
        for t_idx in range(n_targets):
            sub_est = clone_sklearn_estimator(est.estimator, extra="lightgbm")
            sub_valid_y = valid_y[:, t_idx] if valid_y is not None else None
            _fit_lgb_estimator(
                sub_est,
                train_x,
                train_y[:, t_idx],
                valid_x,
                sub_valid_y,
                early_stopping_rounds,
            )
            fitted_subs.append(sub_est)
        est.estimators_ = fitted_subs
        est.n_features_in_ = train_x.shape[1]
        if isinstance(train_x, pd.DataFrame):
            est.feature_names_in_ = np.asarray(train_x.columns, dtype=object)
        return est

    return _fit_lgb_estimator(
        est, train_x, train_y, valid_x, valid_y, early_stopping_rounds
    )


def save_lightgbm_regressor(
    lgb_regressor: Any,
    out_file: Union[str, os.PathLike],
) -> None:
    """
    Save a fitted ``lightgbm.LGBMRegressor`` (or multi-output wrapper) to disk
    using ``joblib``.

    :param lgb_regressor: Trained LightGBM regressor object.
    :param out_file: Output file path (e.g., ``'lgb_reg.joblib'``).

    .. code:: python

        import rsgispy.regression.reglightgbm as rlgb

        rlgb.save_lightgbm_regressor(reg, "lgb_reg.joblib")
    """
    save_joblib_model(
        model=lgb_regressor,
        out_file=out_file,
        have_pkg=HAVE_LIGHTGBM,
        pkg_name="lightgbm",
        extra="lightgbm",
        func_name="save_lightgbm_regressor",
        error_detail=_LIGHTGBM_LOAD_ERROR,
    )


def load_lightgbm_regressor(
    in_file: Union[str, os.PathLike],
) -> Any:
    """
    Load a saved ``lightgbm.LGBMRegressor`` (or multi-output wrapper) from disk
    using ``joblib``.

    :param in_file: Input model file path.
    :return: Loaded regressor instance.

    .. code:: python

        import rsgispy.regression.reglightgbm as rlgb

        reg = rlgb.load_lightgbm_regressor("lgb_reg.joblib")
    """
    return load_joblib_model(
        in_file=in_file,
        have_pkg=HAVE_LIGHTGBM,
        pkg_name="lightgbm",
        extra="lightgbm",
        func_name="load_lightgbm_regressor",
        label="LightGBM regressor model",
        error_detail=_LIGHTGBM_LOAD_ERROR,
    )


_LGB_REG_BACKEND = GBDTRegBackendSpec(
    name="LightGBM",
    package_name="lightgbm",
    extra="lightgbm",
    is_available=lambda: HAVE_LIGHTGBM,
    gpu_available=is_lightgbm_gpu_available,
    apply_gpu_params=_apply_lgb_gpu_params,
    seed_param="random_state",
    jobs_param="n_jobs",
    default_params=_default_lgb_reg_params,
    default_space=lambda: get_lightgbm_optuna_param_space(extended=False),
    extended_space=lambda: get_lightgbm_optuna_param_space(extended=True),
    build_estimator=_build_lgb_reg_estimator,
    fit_estimator=_fit_lgb_reg_estimator,
    load_model=load_lightgbm_regressor,
    save_model=save_lightgbm_regressor,
    get_error_detail=lambda: _LIGHTGBM_LOAD_ERROR,
)


def optimise_lightgbm_regressor(
    reg_cfg: Optional[RegConfigInput],
    train_data: SamplesInput,
    valid_data: Optional[SamplesInput] = None,
    target_cols: TargetColsInput = None,
    param_space: Optional[
        Union[str, Mapping[str, Any], Callable[[Any], Mapping[str, Any]]]
    ] = None,
    extended_param_space: bool = False,
    fixed_params: Optional[Mapping[str, Any]] = None,
    lgb_regressor: Optional[Any] = None,
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
    ``lightgbm.LGBMRegressor`` using Bayesian optimisation with Optuna.

    :param reg_cfg: :class:`~rsgispy.regression.regdata.RegressionConfig`, JSON
                    path, sequence of ``ImageBandInfo``, or ``None``.
    :param train_data: Training sample :class:`pandas.DataFrame` or file path.
    :param valid_data: Optional validation :class:`pandas.DataFrame` or path.
    :param target_cols: Optional target column name(s) overriding ``reg_cfg``.
    :param param_space: Optional Optuna space dict, callable, or preset
                        (``'default'`` / ``'extended'``).
    :param extended_param_space: Use expanded search space when ``True``.
    :param fixed_params: Optional fixed LightGBM parameters merged into trials.
    :param lgb_regressor: Optional base ``lightgbm.LGBMRegressor`` template.
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
    :param use_gpu: Attempt CUDA/OpenCL GPU acceleration (falls back to CPU with
                    warning if unavailable).
    :param gpu_device_id: Optional GPU device index.
    :return: Optimised ``LGBMRegressor`` (or ``MultiOutputRegressor`` wrapper).

    .. code:: python

        import rsgispy.regression.reglightgbm as rlgb

        best_lgb = rlgb.optimise_lightgbm_regressor(
            "reg_config.json", "train.parquet", valid_data="valid.parquet",
            n_trials=30, early_stopping_rounds=20, rnd_seed=42,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    return optimise_gbdt_regressor(
        backend=_LGB_REG_BACKEND,
        func_name="optimise_lightgbm_regressor",
        reg_cfg=reg_cfg,
        train_data=train_data,
        valid_data=valid_data,
        target_cols=target_cols,
        param_space=param_space,
        extended_param_space=extended_param_space,
        fixed_params=fixed_params,
        regressor=lgb_regressor,
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


def train_lightgbm_regressor(
    reg_cfg: Optional[RegConfigInput],
    train_data: SamplesInput,
    test_data: Optional[SamplesInput] = None,
    valid_data: Optional[SamplesInput] = None,
    target_cols: TargetColsInput = None,
    lgb_regressor: Optional[Any] = None,
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
    Train a univariate or multi-target ``lightgbm.LGBMRegressor`` and evaluate
    on optional testing and validation splits.

    :param reg_cfg: Regression configuration, JSON path, or image-band sequence.
    :param train_data: Training sample :class:`pandas.DataFrame` or file path.
    :param test_data: Optional testing sample DataFrame or file path.
    :param valid_data: Optional validation sample DataFrame or file path.
    :param target_cols: Optional target column name(s) overriding ``reg_cfg``.
    :param lgb_regressor: Optional pre-configured ``LGBMRegressor`` template.
    :param model_params: Optional hyperparameter overrides mapping.
    :param early_stopping_rounds: Optional early stopping rounds with ``valid_data``.
    :param feature_cols: Optional ordered feature column names.
    :param rnd_seed: Optional random seed.
    :param n_jobs: Optional CPU thread count.
    :param out_model_file: Optional model output path.
    :param out_metrics_file: Optional metrics JSON output path.
    :param return_model: Return ``(fitted_model, metrics)`` when ``True``.
    :param verbose: Verbosity level (Default = ``0``).
    :param use_gpu: Use CUDA/OpenCL GPU acceleration if available.
    :param gpu_device_id: Optional GPU device index.
    :return: ``(fitted_model, metrics)`` or ``metrics``.

    .. code:: python

        import rsgispy.regression.reglightgbm as rlgb

        model, metrics = rlgb.train_lightgbm_regressor(
            "reg_config.json", "train.parquet", test_data="test.parquet", rnd_seed=42
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    return train_gbdt_regressor(
        backend=_LGB_REG_BACKEND,
        func_name="train_lightgbm_regressor",
        reg_cfg=reg_cfg,
        train_data=train_data,
        test_data=test_data,
        valid_data=valid_data,
        target_cols=target_cols,
        regressor=lgb_regressor,
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


def apply_lightgbm_regressor(
    reg_cfg: Optional[RegConfigInput],
    lgb_regressor: Union[Any, str, os.PathLike],
    in_msk_img: Union[str, os.PathLike],
    img_msk_val: int,
    output_img: Union[str, os.PathLike],
    img_band_info: Optional[Sequence[ImageBandInfo]] = None,
    target_cols: TargetColsInput = None,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    out_no_data_val: float = -9999.0,
    calc_stats: bool = True,
    block_size: Optional[int] = None,
    creation_opts: Optional[Mapping[str, Any]] = None,
    use_pbar: bool = True,
    use_gpu: bool = False,
    gpu_device_id: Optional[int] = None,
) -> None:
    """
    Apply a trained ``lightgbm.LGBMRegressor`` (or multi-output wrapper)
    block-by-block to raster imagery.

    :param reg_cfg: Regression configuration, JSON path, or image-band sequence.
    :param lgb_regressor: Fitted regressor instance or saved model path.
    :param in_msk_img: Mask raster path.
    :param img_msk_val: Mask pixel value to predict.
    :param output_img: Output continuous raster path.
    :param img_band_info: Optional sequence of :class:`ImageBandInfo`.
    :param target_cols: Optional target variable name(s) for band descriptions.
    :param gdalformat: Output GDAL driver format (Default = ``'GTIFF'``).
    :param datatype: Optional ``rsgispy.TYPE_*`` datatype.
    :param out_no_data_val: Nodata fill value (Default = ``-9999.0``).
    :param calc_stats: Compute statistics and pyramids (Default = ``True``).
    :param block_size: Optional block size in pixels.
    :param creation_opts: Optional GDAL creation options mapping.
    :param use_pbar: Display a tqdm progress bar (Default = ``True``).
    :param use_gpu: Reserved for GPU parity with other GBDT backends.
    :param gpu_device_id: Optional GPU device index.

    .. code:: python

        import rsgispy.regression.reglightgbm as rlgb

        rlgb.apply_lightgbm_regressor(
            "reg_config.json", "lgb_reg.joblib", "mask.tif", 1, "canopy_lgb.tif"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    apply_gbdt_regressor(
        backend=_LGB_REG_BACKEND,
        func_name="apply_lightgbm_regressor",
        reg_cfg=reg_cfg,
        regressor=lgb_regressor,
        in_msk_img=in_msk_img,
        img_msk_val=img_msk_val,
        output_img=output_img,
        img_band_info=img_band_info,
        target_cols=target_cols,
        gdalformat=gdalformat,
        datatype=datatype,
        out_no_data_val=out_no_data_val,
        calc_stats=calc_stats,
        block_size=block_size,
        creation_opts=creation_opts,
        use_pbar=use_pbar,
        use_gpu=use_gpu,
        gpu_device_id=gpu_device_id,
    )
