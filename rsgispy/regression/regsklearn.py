"""
Scikit-learn regressor hyperparameter optimisation (Grid Search, Random Search,
and Bayesian optimisation via Optuna with built-in standard parameter spaces),
univariate and multivariate regressor training, and block-wise image regression.
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

import rsgispy
from rsgispy.imageprocess import ensure_file_exists
from rsgispy.imageutils.masking import ImageBandInfo
from rsgispy.regression._reg_common import (
    RegConfigInput,
    SamplesInput,
    TargetColsInput,
    apply_regressor_raster,
    calc_reg_split_metrics,
    check_sklearn_available,
    prepare_reg_splits,
)
from rsgispy.tools._gbdt_common import load_joblib_model, save_joblib_model
from rsgispy.tools._ml_tune import (
    normalise_search_method as _normalise_search_method,
    resolve_sklearn_param_space as _tools_resolve_sklearn_param_space,
    run_optuna_search as _tools_run_optuna_search,
    run_sklearn_search_cv as _tools_run_sklearn_search_cv,
    save_search_outputs as _save_search_outputs,
)
from rsgispy.tools._sklearn import (
    fit_clean_sklearn_estimator,
    is_sklearn_search_cv,
    sklearn,
)
from rsgispy.tools._sklearn_spaces import (
    build_sklearn_param_space as _build_sklearn_param_space,
)
from rsgispy.tools.files import ensure_parent_dir
from rsgispy.tools.jsonio import write_dict_to_json

_NATIVE_MULTIOUTPUT_REGRESSORS = {
    "LinearRegression",
    "Ridge",
    "RidgeCV",
    "Lasso",
    "LassoCV",
    "ElasticNet",
    "ElasticNetCV",
    "MultiTaskLasso",
    "MultiTaskLassoCV",
    "MultiTaskElasticNet",
    "MultiTaskElasticNetCV",
    "DecisionTreeRegressor",
    "ExtraTreeRegressor",
    "RandomForestRegressor",
    "ExtraTreesRegressor",
    "KNeighborsRegressor",
    "RadiusNeighborsRegressor",
    "MLPRegressor",
    "PLSRegression",
    "KernelRidge",
    "DummyRegressor",
    "MultiOutputRegressor",
    "RegressorChain",
}

__all__ = [
    "get_sklearn_regressor_param_space",
    "save_sklearn_regressor",
    "load_sklearn_regressor",
    "perform_sklearn_regressor_param_search",
    "train_sklearn_regressor",
    "apply_sklearn_regressor",
]


def _is_native_multioutput_regressor(estimator: Any) -> bool:
    """Return True if ``estimator`` supports 2D multi-target regression directly."""
    target = estimator
    while target is not None:
        t_name = type(target).__name__
        if t_name in _NATIVE_MULTIOUTPUT_REGRESSORS:
            return True
        if (
            hasattr(target, "steps")
            and isinstance(target.steps, (list, tuple))
            and target.steps
        ):
            target = target.steps[-1][1]
        elif hasattr(target, "regressor") and target.regressor is not None:
            target = target.regressor
        else:
            break
    return False


def _wrap_multioutput_if_needed(
    sk_regressor: Any,
    n_targets: int,
) -> Tuple[Any, bool]:
    """Wrap single-output estimator in ``MultiOutputRegressor`` if needed."""
    if n_targets <= 1 or is_sklearn_search_cv(sk_regressor):
        return sk_regressor, False
    if _is_native_multioutput_regressor(sk_regressor):
        return sk_regressor, False
    return sklearn.multioutput.MultiOutputRegressor(sk_regressor), True


def save_sklearn_regressor(
    sk_regressor: Any,
    out_file: Union[str, os.PathLike],
) -> None:
    """
    Save a trained ``scikit-learn`` (or compatible) regressor object to disk
    using ``joblib``.

    :param sk_regressor: Regressor object to persist.
    :param out_file: Output file path (e.g., ``'rf_regressor.joblib'``).

    .. code:: python

        from sklearn.ensemble import RandomForestRegressor
        import rsgispy.regression.regsklearn as rsk

        reg = RandomForestRegressor(n_estimators=50)
        rsk.save_sklearn_regressor(reg, "rf_regressor.joblib")
    """
    save_joblib_model(
        model=sk_regressor,
        out_file=out_file,
        have_pkg=True,
        pkg_name="joblib",
        extra="regression",
        func_name="save_sklearn_regressor",
    )


def load_sklearn_regressor(
    in_file: Union[str, os.PathLike],
) -> Any:
    """
    Load a persisted ``scikit-learn`` (or compatible) regressor object from
    disk using ``joblib``.

    :param in_file: Path to the saved regressor file.
    :return: Loaded regressor object.

    .. code:: python

        import rsgispy.regression.regsklearn as rsk

        reg = rsk.load_sklearn_regressor("rf_regressor.joblib")
    """
    in_path = os.fspath(in_file)
    ensure_file_exists(in_path, label="Regressor model")
    return load_joblib_model(
        in_file=in_path,
        have_pkg=True,
        pkg_name="joblib",
        extra="regression",
        func_name="load_sklearn_regressor",
        label="Regressor model",
    )


def get_sklearn_regressor_param_space(
    sk_regressor: Any,
    search_method: str = "optuna",
    extended: bool = False,
) -> Union[Dict[str, Any], Callable[[Any], Dict[str, Any]]]:
    """
    Return a standard built-in hyperparameter search space for a commonly used
    ``scikit-learn`` regressor (or a ``Pipeline`` / ``MultiOutputRegressor``
    / ``TransformedTargetRegressor`` wrapping one).

    Supports ``RandomForestRegressor``, ``ExtraTreesRegressor``,
    ``GradientBoostingRegressor``, ``HistGradientBoostingRegressor``, ``SVR``,
    ``MLPRegressor``, ``KNeighborsRegressor``, ``Ridge``, ``Lasso``,
    ``ElasticNet``, ``DecisionTreeRegressor``, and ``AdaBoostRegressor``
    (passed as an instance, class, wrapper, or alias string such as ``'rf'``,
    ``'svr'``, ``'mlp'``, or ``'ridge'``).

    :param sk_regressor: Regressor instance, class, wrapper, or name/alias
                         string.
    :param search_method: Target search strategy: ``'optuna'`` (default),
                          ``'grid'``, or ``'random'``.
    :param extended: If ``True``, return a broader search space with additional
                     regularisation and structural parameters; if ``False``
                     (default), return the standard key parameter space.
    :return: Search space dictionary or Optuna ``fn(trial) -> dict`` callable.

    .. code:: python

        from sklearn.ensemble import RandomForestRegressor
        import rsgispy.regression.regsklearn as rsk

        optuna_space = rsk.get_sklearn_regressor_param_space(
            RandomForestRegressor(), search_method="optuna"
        )
        grid_space = rsk.get_sklearn_regressor_param_space(
            "svr", search_method="grid", extended=True
        )
    """
    method = _normalise_search_method(search_method)
    return _build_sklearn_param_space(
        sk_estimator=sk_regressor,
        method=method,
        extended=bool(extended),
        task="regression",
    )


def perform_sklearn_regressor_param_search(
    reg_cfg: Optional[RegConfigInput],
    sk_regressor: Any,
    train_data: SamplesInput,
    target_cols: TargetColsInput = None,
    param_space: Optional[
        Union[
            str,
            Mapping[str, Any],
            Sequence[Mapping[str, Any]],
            Callable[[Any], Mapping[str, Any]],
        ]
    ] = None,
    search_method: str = "grid",
    valid_data: Optional[SamplesInput] = None,
    scoring: Union[str, Callable[..., float]] = "r2",
    cv: Union[int, Any] = 5,
    n_iter: int = 20,
    n_trials: int = 50,
    timeout: Optional[float] = None,
    direction: str = "maximize",
    n_jobs: Optional[int] = None,
    rnd_seed: Optional[int] = None,
    feature_cols: Optional[Sequence[str]] = None,
    refit: bool = True,
    out_results_file: Optional[Union[str, os.PathLike]] = None,
    out_params_json: Optional[Union[str, os.PathLike]] = None,
    return_results: bool = False,
    verbose: int = 0,
    extended_param_space: bool = False,
    use_pbar: bool = True,
) -> Union[Any, Tuple[Any, Dict[str, Any]]]:
    """
    Optimise hyperparameters for a univariate or multivariate ``scikit-learn``
    regressor using Grid Search, Random Search, or Bayesian optimisation with
    Optuna.

    When ``param_space=None`` (or ``'default'`` / ``'extended'``), built-in
    standard search spaces are automatically selected for supported
    ``scikit-learn`` regressors (see :func:`get_sklearn_regressor_param_space`).
    If multiple targets are configured and ``sk_regressor`` does not natively
    support 2D targets (e.g., ``SVR``, ``GradientBoostingRegressor``), it is
    automatically wrapped in ``sklearn.multioutput.MultiOutputRegressor``.

    :param reg_cfg: :class:`~rsgispy.regression.regdata.RegressionConfig`, JSON
                    config path, sequence of ``ImageBandInfo``, or ``None``
                    when ``target_cols`` is supplied.
    :param sk_regressor: Base ``scikit-learn`` regressor estimator, wrapper, or
                         pre-configured ``GridSearchCV`` / ``RandomizedSearchCV``.
    :param train_data: Training sample :class:`pandas.DataFrame` or file path.
    :param target_cols: Optional target column name(s) overriding ``reg_cfg``.
    :param param_space: Parameter search space (dict, list of dicts, Optuna
                        callable, ``'default'`` / ``'extended'``, or ``None``
                        for the built-in standard space).
    :param search_method: Optimisation strategy: ``'grid'`` (default),
                          ``'random'``, or ``'optuna'``.
    :param valid_data: Optional validation :class:`pandas.DataFrame` or path.
    :param scoring: Scikit-learn regression metric name or scorer callable
                    (Default = ``'r2'``).
    :param cv: Cross-validation folds or splitter used when ``valid_data`` is
               ``None`` (Default = ``5``).
    :param n_iter: Candidate samples for ``search_method='random'``
                   (Default = ``20``).
    :param n_trials: Optuna trials for ``search_method='optuna'``
                     (Default = ``50``).
    :param timeout: Optional time limit in seconds for Optuna optimisation.
    :param direction: Optuna direction (``'maximize'`` or ``'minimize'``,
                      Default = ``'maximize'``).
    :param n_jobs: Number of parallel workers for CV evaluations.
    :param rnd_seed: Optional random seed for reproducible search and splits.
    :param feature_cols: Optional ordered feature column names.
    :param refit: If ``True`` (default), fit the returned optimal estimator on
                  ``train_data``; otherwise return an unfitted clone.
    :param out_results_file: Optional path saving CV/Optuna trials table.
    :param out_params_json: Optional JSON path saving best parameters and score.
    :param return_results: If ``True``, return ``(best_estimator, results_dict)``.
    :param verbose: Verbosity level (Default = ``0``).
    :param extended_param_space: Use broader extended built-in parameter space
                                 when ``param_space`` is ``None``.
    :param use_pbar: Display a ``tqdm`` progress bar during Optuna trials
                     (Default = ``True``).
    :return: Optimised regressor estimator, or ``(best_estimator, results_dict)``.

    .. code:: python

        from sklearn.ensemble import RandomForestRegressor
        import rsgispy.regression.regsklearn as rsk

        best_rf = rsk.perform_sklearn_regressor_param_search(
            reg_cfg="reg_config.json",
            sk_regressor=RandomForestRegressor(),
            train_data="train.parquet",
            search_method="optuna",
            n_trials=25,
            rnd_seed=42,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    check_sklearn_available("perform_sklearn_regressor_param_search")
    method = _normalise_search_method(search_method)
    _, eff_targets, _, train_x, train_y, _, valid_xy = prepare_reg_splits(
        reg_cfg=reg_cfg,
        train_data=train_data,
        test_data=None,
        valid_data=valid_data,
        feature_cols=feature_cols,
        target_cols=target_cols,
    )
    valid_x = valid_xy[0] if valid_xy is not None else None
    valid_y = valid_xy[1] if valid_xy is not None else None

    eff_regressor, was_wrapped = _wrap_multioutput_if_needed(
        sk_regressor, len(eff_targets)
    )
    eff_param_space, space_mode = _tools_resolve_sklearn_param_space(
        sk_estimator=eff_regressor,
        method=method,
        param_space=param_space,
        extended_param_space=extended_param_space,
        task="regression",
    )
    if (
        was_wrapped
        and space_mode == "custom"
        and isinstance(eff_param_space, Mapping)
        and not any(k.startswith("estimator__") for k in eff_param_space)
    ):
        eff_param_space = {
            f"estimator__{k}": v for k, v in eff_param_space.items()
        }

    if method == "optuna":
        best_est, best_params, best_score, results_df, search_obj = (
            _tools_run_optuna_search(
                sk_estimator=eff_regressor,
                param_space=eff_param_space,
                train_x=train_x,
                train_y=train_y,
                valid_x=valid_x,
                valid_y=valid_y,
                scoring=scoring,
                cv=cv,
                n_trials=n_trials,
                timeout=timeout,
                direction=direction,
                n_jobs=n_jobs,
                rnd_seed=rnd_seed,
                refit=refit,
                verbose=verbose,
                stratified=False,
                func_name="perform_sklearn_regressor_param_search",
                extra="regression",
                use_pbar=use_pbar,
            )
        )
    else:
        best_est, best_params, best_score, results_df, search_obj = (
            _tools_run_sklearn_search_cv(
                method=method,
                sk_estimator=eff_regressor,
                param_space=eff_param_space,
                train_x=train_x,
                train_y=train_y,
                valid_x=valid_x,
                valid_y=valid_y,
                scoring=scoring,
                cv=cv,
                n_iter=n_iter,
                n_jobs=n_jobs,
                rnd_seed=rnd_seed,
                refit=refit,
                verbose=verbose,
                stratified=False,
                extra="regression",
            )
        )

    _save_search_outputs(
        results_df=results_df,
        best_params=best_params,
        best_score=best_score,
        scoring=scoring,
        search_method=method,
        space_mode=space_mode,
        out_results_file=out_results_file,
        out_params_json=out_params_json,
        extra_meta={"target_cols": list(eff_targets)},
    )

    if return_results:
        return best_est, {
            "search_method": method,
            "param_space_mode": space_mode,
            "best_params": best_params,
            "best_score": float(best_score),
            "results_df": results_df,
            "search_obj": search_obj,
        }
    return best_est


def train_sklearn_regressor(
    reg_cfg: Optional[RegConfigInput],
    sk_regressor: Any,
    train_data: SamplesInput,
    test_data: Optional[SamplesInput] = None,
    valid_data: Optional[SamplesInput] = None,
    target_cols: TargetColsInput = None,
    feature_cols: Optional[Sequence[str]] = None,
    rnd_seed: Optional[int] = None,
    out_model_file: Optional[Union[str, os.PathLike]] = None,
    out_metrics_file: Optional[Union[str, os.PathLike]] = None,
    return_model: bool = False,
) -> Union[Dict[str, float], Tuple[Any, Dict[str, float]]]:
    """
    Train a ``scikit-learn`` (or compatible) regressor on univariate or
    multivariate regression sample tables and evaluate on optional test and
    validation datasets.

    When multiple targets are specified and ``sk_regressor`` does not natively
    support multi-output regression, set ``return_model=True`` (or pass a
    ``MultiOutputRegressor`` directly) so the wrapped fitted estimator is
    returned cleanly.

    :param reg_cfg: :class:`~rsgispy.regression.regdata.RegressionConfig`, JSON
                    config path, sequence of ``ImageBandInfo``, or ``None``.
    :param sk_regressor: ``scikit-learn`` regressor estimator template.
    :param train_data: Training sample :class:`pandas.DataFrame` or file path.
    :param test_data: Optional testing sample :class:`pandas.DataFrame` or path.
    :param valid_data: Optional validation sample :class:`pandas.DataFrame` or
                       path.
    :param target_cols: Optional target column name(s) overriding ``reg_cfg``.
    :param feature_cols: Optional ordered feature column names.
    :param rnd_seed: Optional random seed passed to the estimator when
                     supported.
    :param out_model_file: Optional output ``.joblib`` path to save the fitted
                           regressor.
    :param out_metrics_file: Optional output JSON path to save training/test/
                             validation metrics (``r2``, ``rmse``, ``mae``,
                             ``bias``, ``explained_variance``, ``pearson_r``).
    :param return_model: When ``True``, return ``(fitted_model, metrics)``
                         without mutating ``sk_regressor``.
    :return: Metrics dictionary (or ``(fitted_model, metrics)`` when
             ``return_model=True``).

    .. code:: python

        from sklearn.ensemble import RandomForestRegressor
        import rsgispy.regression.regsklearn as rsk

        fitted_rf, metrics = rsk.train_sklearn_regressor(
            reg_cfg="reg_config.json",
            sk_regressor=RandomForestRegressor(n_estimators=100, random_state=42),
            train_data="train.parquet",
            test_data="test.parquet",
            return_model=True,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    check_sklearn_available("train_sklearn_regressor")
    _, eff_targets, _, train_x, train_y, test_xy, valid_xy = prepare_reg_splits(
        reg_cfg=reg_cfg,
        train_data=train_data,
        test_data=test_data,
        valid_data=valid_data,
        feature_cols=feature_cols,
        target_cols=target_cols,
    )

    eff_regressor, was_wrapped = _wrap_multioutput_if_needed(
        sk_regressor, len(eff_targets)
    )
    if was_wrapped and not return_model:
        raise rsgispy.RSGISPyException(
            f"Estimator '{type(sk_regressor).__name__}' does not natively support "
            "multi-target regression and must be auto-wrapped in "
            "MultiOutputRegressor. Pass return_model=True or wrap with "
            "sklearn.multioutput.MultiOutputRegressor directly."
        )

    fitted_est = fit_clean_sklearn_estimator(
        eff_regressor,
        train_x,
        train_y,
        sync_input=not return_model and not was_wrapped,
        rnd_seed=rnd_seed,
        extra="regression",
    )

    metrics = calc_reg_split_metrics(
        fitted_est, train_x, train_y, eff_targets, "train"
    )
    if test_xy is not None:
        metrics.update(
            calc_reg_split_metrics(
                fitted_est, test_xy[0], test_xy[1], eff_targets, "test"
            )
        )
    if valid_xy is not None:
        metrics.update(
            calc_reg_split_metrics(
                fitted_est, valid_xy[0], valid_xy[1], eff_targets, "valid"
            )
        )

    if out_model_file is not None:
        save_sklearn_regressor(fitted_est, out_model_file)

    if out_metrics_file is not None:
        write_dict_to_json(metrics, ensure_parent_dir(out_metrics_file))

    if return_model:
        return fitted_est, metrics
    return metrics


def apply_sklearn_regressor(
    reg_cfg: Optional[RegConfigInput],
    sk_regressor: Union[Any, str, os.PathLike],
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
) -> None:
    """
    Apply a trained univariate or multivariate ``scikit-learn`` regressor
    block-by-block to raster pixels where ``in_msk_img == img_msk_val``,
    writing one continuous output band per target variable.

    :param reg_cfg: :class:`~rsgispy.regression.regdata.RegressionConfig`, JSON
                    config path, sequence of ``ImageBandInfo``, or ``None``
                    when ``img_band_info`` is passed explicitly.
    :param sk_regressor: Fitted regressor instance or path to a saved
                         ``.joblib`` model file.
    :param in_msk_img: Path to the mask image controlling which pixels are
                       predicted.
    :param img_msk_val: Pixel value within ``in_msk_img`` identifying valid
                        pixels to predict.
    :param output_img: Output continuous prediction raster path (single-band for
                       univariate regression, multi-band for multivariate).
    :param img_band_info: Optional sequence of :class:`ImageBandInfo` overriding
                          ``reg_cfg.img_band_info``.
    :param target_cols: Optional target variable name(s) used for band count
                        validation and output raster band descriptions.
    :param gdalformat: Output GDAL/rasterio format (Default = ``'GTIFF'``).
    :param datatype: Optional ``rsgispy.TYPE_*`` integer datatype (defaults to
                     ``rsgispy.TYPE_32FLOAT``).
    :param out_no_data_val: Output nodata fill value (Default = ``-9999.0``).
    :param calc_stats: Compute output band statistics and overview pyramids
                       (Default = ``True``).
    :param block_size: Optional processing block size in pixels.
    :param creation_opts: Optional GDAL creation options mapping.
    :param use_pbar: Display a tqdm progress bar during prediction
                     (Default = ``True``).

    .. code:: python

        import rsgispy.regression.regsklearn as rsk

        rsk.apply_sklearn_regressor(
            reg_cfg="reg_config.json",
            sk_regressor="landsat_rf_model.joblib",
            in_msk_img="landsat_valid_mask.tif",
            img_msk_val=1,
            output_img="canopy_cover_pred.tif",
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    check_sklearn_available("apply_sklearn_regressor")
    estimator = (
        load_sklearn_regressor(sk_regressor)
        if isinstance(sk_regressor, (str, os.PathLike))
        else sk_regressor
    )
    apply_regressor_raster(
        reg_cfg=reg_cfg,
        estimator=estimator,
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
    )
