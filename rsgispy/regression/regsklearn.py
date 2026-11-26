# pylint: disable=too-many-lines
"""
Scikit-learn regressor hyperparameter optimisation (Grid Search, Random Search,
and Bayesian optimisation via Optuna with built-in standard parameter spaces),
univariate and multivariate regressor training, block-wise image regression,
and repeated k-fold and jackknife (leave-one-out) assessment of any regressor.
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
    load_samples_df,
    prepare_reg_splits,
)
from rsgispy.tools._gbdt_common import load_joblib_model, save_joblib_model
from rsgispy.tools._ml_tune import (
    normalise_search_method as _normalise_search_method,
    resolve_cv_groups_coords as _resolve_cv_groups_coords,
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
from rsgispy.tools.strings import check_str

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
    "perform_kfold_fit",
    "summarise_kfold_metrics",
    "perform_jackknife_fit",
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
    *,
    groups: Optional[Union[str, Sequence[Any], np.ndarray]] = None,
    coords: Optional[Union[Sequence[str], np.ndarray]] = None,
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

    The spatial and group splitters of :mod:`rsgispy.ml.cv` can be passed
    directly as ``cv`` with the sample coordinates (``coords``, for
    :class:`~rsgispy.ml.cv.SpatialBlockKFold` and
    :class:`~rsgispy.ml.cv.SpatialBufferCV`) or groups (``groups``, e.g. the
    ``poly_id_col`` column of
    :func:`~rsgispy.regression.regdata.extract_reg_training_samples`, for
    :class:`~rsgispy.ml.cv.PolygonGroupKFold`); scikit-learn group splitters
    (e.g. ``GroupKFold``) receive ``groups``. With ``groups`` or ``coords`` an
    integer ``cv`` gives a ``SpatialBlockKFold`` (``coords``) or
    ``PolygonGroupKFold`` (``groups``) with ``cv`` folds. The folds are made
    once and used for every candidate (they are in
    ``results_dict['search_obj'].cv`` for grid and random searches).

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
    :param groups: Optional group of each training sample for the
                   cross-validation folds: a column name of ``train_data``
                   (e.g. ``'train_grp'``) or one label per row. Used only
                   when ``valid_data`` is ``None`` (Default = ``None``).
    :param coords: Optional coordinates of each training sample for spatial
                   cross-validation folds: two column names of
                   ``train_data`` (e.g. ``('x', 'y')``) or an
                   ``(n_samples, 2)`` array. Used only when ``valid_data`` is
                   ``None`` (Default = ``None``).
    :return: Optimised regressor estimator, or ``(best_estimator, results_dict)``.

    .. code:: python

        from sklearn.ensemble import RandomForestRegressor
        import rsgispy.regression.regsklearn as rsk
        from rsgispy.ml.cv import SpatialBlockKFold

        best_rf = rsk.perform_sklearn_regressor_param_search(
            reg_cfg="reg_config.json",
            sk_regressor=RandomForestRegressor(),
            train_data="train.parquet",
            search_method="optuna",
            n_trials=25,
            rnd_seed=42,
        )
        # Spatially blocked folds (train.parquet has x and y columns)
        best_rf = rsk.perform_sklearn_regressor_param_search(
            "reg_config.json",
            RandomForestRegressor(),
            "train.parquet",
            param_space={"min_samples_leaf": [1, 3, 10]},
            cv=SpatialBlockKFold(n_splits=4, block_size=2000.0),
            coords=("x", "y"),
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    check_sklearn_available("perform_sklearn_regressor_param_search")
    method = _normalise_search_method(search_method)
    train_df = load_samples_df(train_data, "train_data")
    _, eff_targets, _, train_x, train_y, _, valid_xy = prepare_reg_splits(
        reg_cfg=reg_cfg,
        train_data=train_df,
        test_data=None,
        valid_data=valid_data,
        feature_cols=feature_cols,
        target_cols=target_cols,
    )
    cv_groups, cv_coords = _resolve_cv_groups_coords(train_df, groups, coords)
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
                groups=cv_groups,
                coords=cv_coords,
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
                groups=cv_groups,
                coords=cv_coords,
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


# ---------------------------------------------------------------------------
# Repeated k-fold and jackknife (leave-one-out) assessment of a regressor
# ---------------------------------------------------------------------------

_FOLD_ID_COLS = ("repeat", "fold", "n_samples")


def _prep_assess_xy(
    x: Any,
    y: Any,
    target_names: Optional[Sequence[str]],
    groups: Optional[Any],
) -> Tuple[Any, np.ndarray, List[str], Optional[np.ndarray]]:
    """
    Check and normalise the ``x`` / ``y`` / ``groups`` inputs of the k-fold
    and jackknife functions. Returns ``x`` (a DataFrame is kept so feature
    names reach the estimator), ``y`` as a 2D float array, the target names
    and ``groups`` as a 1D array (or ``None``).
    """
    # pylint: disable=too-many-branches
    if isinstance(x, pd.Series):
        x = x.to_frame()
    if not isinstance(x, pd.DataFrame):
        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if x.ndim != 2:
            raise rsgispy.RSGISPyException(
                f"x must be 1D or 2D: got an array with {x.ndim} dimensions."
            )

    if isinstance(y, pd.Series):
        def_names = [str(y.name) if y.name is not None else "target"]
        y_arr = y.to_numpy(dtype=np.float64).reshape(-1, 1)
    elif isinstance(y, pd.DataFrame):
        def_names = [str(c) for c in y.columns]
        y_arr = y.to_numpy(dtype=np.float64)
    else:
        y_arr = np.asarray(y, dtype=np.float64)
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)
        if y_arr.ndim != 2:
            raise rsgispy.RSGISPyException(
                f"y must be 1D or 2D: got an array with {y_arr.ndim} "
                "dimensions."
            )
        if y_arr.shape[1] == 1:
            def_names = ["target"]
        else:
            def_names = [f"target_{i + 1}" for i in range(y_arr.shape[1])]

    if x.shape[0] != y_arr.shape[0]:
        raise rsgispy.RSGISPyException(
            f"x has {x.shape[0]} samples but y has {y_arr.shape[0]}."
        )
    if y_arr.shape[1] == 0:
        raise rsgispy.RSGISPyException("y has no target columns.")

    if target_names is None:
        names = def_names
    else:
        names = (
            [str(target_names)]
            if isinstance(target_names, str)
            else [str(t) for t in target_names]
        )
        if len(names) != y_arr.shape[1]:
            raise rsgispy.RSGISPyException(
                f"target_names has {len(names)} name(s) but y has "
                f"{y_arr.shape[1]} target column(s)."
            )
    if len(set(names)) != len(names):
        raise rsgispy.RSGISPyException(
            f"The target names must be unique: got {names}."
        )

    grp_arr = None
    if groups is not None:
        grp_arr = np.asarray(groups)
        if grp_arr.ndim != 1 or grp_arr.shape[0] != y_arr.shape[0]:
            raise rsgispy.RSGISPyException(
                "groups must be 1D with one value per sample "
                f"({y_arr.shape[0]}): got shape {grp_arr.shape}."
            )
    return x, y_arr, names, grp_arr


def _take_rows(x: Any, idx: np.ndarray) -> Any:
    """Select rows of a NumPy array or DataFrame by position."""
    if isinstance(x, pd.DataFrame):
        return x.iloc[idx]
    return x[idx]


def _fit_predict_rows(
    regrs_obj: Any,
    x: Any,
    y_arr: np.ndarray,
    train_idx: np.ndarray,
    pred_x: Any,
) -> Tuple[Any, np.ndarray]:
    """
    Fit a clone of ``regrs_obj`` to the ``train_idx`` rows and predict
    ``pred_x``. Returns the fitted clone and the predictions as a 2D array
    (samples, targets).
    """
    n_targets = y_arr.shape[1]
    mdl, _ = _wrap_multioutput_if_needed(
        sklearn.base.clone(regrs_obj), n_targets
    )
    y_fit = y_arr[train_idx]
    if n_targets == 1:
        y_fit = y_fit[:, 0]
    mdl.fit(_take_rows(x, train_idx), y_fit)
    n_pred = pred_x.shape[0]
    y_hat = np.asarray(mdl.predict(pred_x), dtype=np.float64)
    return mdl, y_hat.reshape(n_pred, -1)


def _target_file_stems(target_names: Sequence[str]) -> List[str]:
    """File-name safe, unique versions of the target names."""
    stems = [
        check_str(
            str(name),
            rm_non_ascii=True,
            rm_dashs=True,
            rm_spaces=True,
            rm_punc=True,
        ).lower()
        or "target"
        for name in target_names
    ]
    if len(set(stems)) != len(stems):
        raise rsgispy.RSGISPyException(
            f"The target names {list(target_names)} give the same file name "
            f"({stems}): give distinct target_names."
        )
    return stems


def _write_target_csvs(
    out_dir: Union[str, os.PathLike],
    out_prefix: str,
    tables: Mapping[str, Mapping[str, pd.DataFrame]],
    target_names: Sequence[str],
    index_tables: Sequence[str] = (),
) -> None:
    """
    Write ``tables[kind][target]`` to
    ``<out_dir>/<out_prefix>_<kind>_<target>.csv``. The tables named in
    ``index_tables`` are written with their index.
    """
    out_dir = os.fspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    stems = _target_file_stems(target_names)
    for kind, per_target in tables.items():
        for name, stem in zip(target_names, stems):
            out_file = os.path.join(out_dir, f"{out_prefix}_{kind}_{stem}.csv")
            per_target[name].to_csv(out_file, index=kind in index_tables)


def _pooled_metrics(y_t: np.ndarray, y_p: np.ndarray) -> Dict[str, Any]:
    """The calc_reg_metrics metrics of one target's observed / predicted values."""
    # Imported here so importing regsklearn does not import matplotlib.
    # pylint: disable-next=import-outside-toplevel
    from rsgispy.regression.regmetrics import calc_reg_metrics

    return calc_reg_metrics(
        pd.DataFrame({"obs": y_t, "pred": y_p}), ref_col="obs", pred_col="pred"
    )


def _rmse(y_t: np.ndarray, y_p: np.ndarray) -> float:
    """Root mean squared error of the finite pairs (``nan`` if there are none)."""
    valid = np.isfinite(y_t) & np.isfinite(y_p)
    if not np.any(valid):
        return float("nan")
    return float(np.sqrt(np.mean((y_p[valid] - y_t[valid]) ** 2)))


def _group_kfold_splits(
    groups: np.ndarray,
    n_splits: int,
    n_repeats: int,
    rnd_seed: Optional[int],
):
    """
    Yield ``(repeat, fold, train_idx, test_idx)`` for a k-fold over groups,
    repeated with a new random order of the groups each time. The shuffled
    groups are split into ``n_splits`` folds with (nearly) the same number of
    groups, so a group is never in both the train and test samples.
    """
    uniq_grps, grp_codes = np.unique(groups, return_inverse=True)
    grp_codes = grp_codes.reshape(-1)
    n_grps = uniq_grps.shape[0]
    if n_grps < n_splits:
        raise rsgispy.RSGISPyException(
            f"There are {n_grps} groups, fewer than n_splits ({n_splits})."
        )
    rng = np.random.default_rng(rnd_seed)
    for i_rep in range(n_repeats):
        grp_fold = np.empty(n_grps, dtype=np.int64)
        for i_fold, fold_grps in enumerate(
            np.array_split(rng.permutation(n_grps), n_splits)
        ):
            grp_fold[fold_grps] = i_fold
        smpl_fold = grp_fold[grp_codes]
        for i_fold in range(n_splits):
            yield (
                i_rep,
                i_fold,
                np.flatnonzero(smpl_fold != i_fold),
                np.flatnonzero(smpl_fold == i_fold),
            )


def perform_kfold_fit(
    regrs_obj: Any,
    x: Any,
    y: Any,
    target_names: Optional[Sequence[str]] = None,
    *,
    n_splits: int = 5,
    n_repeats: int = 20,
    groups: Optional[Any] = None,
    rnd_seed: Optional[int] = 42,
    out_dir: Optional[Union[str, os.PathLike]] = None,
    out_prefix: str = "kfold",
    use_pbar: bool = True,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    Assess a regressor with a repeated k-fold cross-validation. In each
    repeat the samples are split at random into ``n_splits`` folds, and each
    fold is predicted by a copy of the regressor fitted to the other folds.
    The accuracy of each fold is calculated with
    :func:`~rsgispy.regression.regmetrics.calc_reg_metrics` (including
    ``mse``, ``nrmse``, ``nrmse_range`` and ``nbias``), and the out-of-fold
    predictions of every repeat are returned for residual plots. Use
    :func:`summarise_kfold_metrics` to summarise the metrics.

    Without ``groups`` the splits are scikit-learn's ``RepeatedKFold``. With
    ``groups`` the unique groups are shuffled (with a new order in each
    repeat) and split into ``n_splits`` folds with (nearly) the same number
    of groups, as ``GroupKFold(shuffle=True)`` does, so the samples of a
    group (for example the pixels of one plot or the plots of one site) are
    never in both the train and test samples. The folds then contain the same
    number of groups, not the same number of samples.

    The regressor can be any unfitted estimator with ``fit`` / ``predict``,
    including a scikit-learn ``Pipeline`` (so a scaler is fitted to the
    training folds only) and the scikit-learn wrappers of XGBoost, LightGBM
    and CatBoost. It is cloned for each fit. For several targets, an
    estimator that cannot fit several targets is wrapped in a
    ``MultiOutputRegressor``. For one target the estimator is fitted to a 1D
    ``y``.

    :param regrs_obj: An unfitted regressor (it is not changed).
    :param x: The predictor variables: a NumPy array (samples, predictors)
              or a ``pandas.DataFrame`` (passed to the estimator as a
              DataFrame, so the column names are kept).
    :param y: The response variables: a 1D array or
              ``pandas.Series`` for one target, or a 2D array or
              ``pandas.DataFrame`` (samples, targets).
    :param target_names: The names of the targets (the columns of ``y``).
                         Defaults to the ``y`` column (or Series) names, or
                         to ``'target'`` (one target) or ``'target_1'``,
                         ``'target_2'``, ... for a NumPy ``y``.
    :param n_splits: The number of folds (Default = 5).
    :param n_repeats: The number of times the k-fold is repeated, each time
                      with a different random split (Default = 20).
    :param groups: Optional group label of each sample; a group is never
                   split across the train and test samples.
    :param rnd_seed: Seed for the random splits, so the results can be
                     repeated (Default = 42). Seed the estimator itself (for
                     example ``random_state``) for identical results.
    :param out_dir: Optional output directory. When given, the metrics and
                    residuals of each target are written to
                    ``<out_prefix>_metrics_<target>.csv`` and
                    ``<out_prefix>_residuals_<target>.csv``, with the target
                    name in lower case with spaces, dashes and punctuation
                    removed (for example ``'Vol / ha'`` becomes
                    ``vol_ha``).
    :param out_prefix: Prefix of the output file names (Default = ``'kfold'``).
    :param use_pbar: Show a progress bar over the fits (Default = ``True``).
    :return: ``(metrics, residuals)``, two dictionaries keyed by target
             name. ``metrics[target]`` is a ``pandas.DataFrame`` with
             one row per fold and repeat: the ``repeat`` and ``fold``
             numbers (from 0), the number of test samples (``n_samples``)
             and the :func:`~rsgispy.regression.regmetrics.calc_reg_metrics`
             metrics. ``residuals[target]`` is a ``pandas.DataFrame``
             of the out-of-fold predictions of every repeat, with the
             ``repeat``, ``fold``, ``sample_idx`` (the position of the
             sample in ``x`` / ``y``), ``y_true`` and ``y_pred`` columns.

    .. code:: python

        import pandas
        from sklearn.ensemble import ExtraTreesRegressor
        import rsgispy.regression.regsklearn as rsk

        df = pandas.read_csv("forest_plots.csv")
        metrics, residuals = rsk.perform_kfold_fit(
            ExtraTreesRegressor(random_state=0),
            df[["p95", "cover", "skew"]],
            df[["Vol / ha", "BA / ha"]],
            n_splits=5,
            n_repeats=20,
            out_dir="kfold_outputs",
            out_prefix="kfold_et",
        )
        print(rsk.summarise_kfold_metrics(metrics))
    """
    # pylint: disable=too-many-arguments,too-many-locals
    check_sklearn_available("perform_kfold_fit")
    if int(n_splits) < 2:
        raise rsgispy.RSGISPyException(f"n_splits must be >= 2: got {n_splits}.")
    if int(n_repeats) < 1:
        raise rsgispy.RSGISPyException(
            f"n_repeats must be >= 1: got {n_repeats}."
        )
    n_splits = int(n_splits)
    n_repeats = int(n_repeats)
    x, y_arr, names, grp_arr = _prep_assess_xy(x, y, target_names, groups)
    if out_dir is not None:
        _target_file_stems(names)

    if grp_arr is None:
        rkf = sklearn.model_selection.RepeatedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=rnd_seed
        )
        splits = (
            (i_split // n_splits, i_split % n_splits, train_idx, test_idx)
            for i_split, (train_idx, test_idx) in enumerate(rkf.split(y_arr))
        )
    else:
        splits = _group_kfold_splits(grp_arr, n_splits, n_repeats, rnd_seed)

    fold_rows: Dict[str, List[Dict[str, Any]]] = {n: [] for n in names}
    resid_parts: List[pd.DataFrame] = []
    tqdm_cls = rsgispy.get_tqdm()
    for i_rep, i_fold, train_idx, test_idx in tqdm_cls(
        splits, total=n_splits * n_repeats, disable=not use_pbar
    ):
        _, y_hat = _fit_predict_rows(
            regrs_obj, x, y_arr, train_idx, _take_rows(x, test_idx)
        )
        y_test = y_arr[test_idx]
        part = {
            "repeat": np.full(test_idx.shape[0], i_rep, dtype=np.int64),
            "fold": np.full(test_idx.shape[0], i_fold, dtype=np.int64),
            "sample_idx": test_idx.astype(np.int64),
        }
        for i_tgt, name in enumerate(names):
            f_metrics = _pooled_metrics(y_test[:, i_tgt], y_hat[:, i_tgt])
            row = {
                "repeat": i_rep,
                "fold": i_fold,
                "n_samples": f_metrics.pop("n_samples"),
            }
            row.update(f_metrics)
            fold_rows[name].append(row)
            part[f"y_true_{i_tgt}"] = y_test[:, i_tgt]
            part[f"y_pred_{i_tgt}"] = y_hat[:, i_tgt]
        resid_parts.append(pd.DataFrame(part))

    all_resids = pd.concat(resid_parts, ignore_index=True)
    metrics: Dict[str, pd.DataFrame] = {}
    residuals: Dict[str, pd.DataFrame] = {}
    for i_tgt, name in enumerate(names):
        metrics[name] = pd.DataFrame(fold_rows[name])
        residuals[name] = pd.DataFrame(
            {
                "repeat": all_resids["repeat"],
                "fold": all_resids["fold"],
                "sample_idx": all_resids["sample_idx"],
                "y_true": all_resids[f"y_true_{i_tgt}"],
                "y_pred": all_resids[f"y_pred_{i_tgt}"],
            }
        )

    if out_dir is not None:
        _write_target_csvs(
            out_dir,
            out_prefix,
            {"metrics": metrics, "residuals": residuals},
            names,
        )
    return metrics, residuals


def summarise_kfold_metrics(
    metrics: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Summarise the per-fold metrics returned by :func:`perform_kfold_fit`:
    the mean, median, standard deviation and the 5th and 95th percentiles of
    each metric of each target over all the folds and repeats. ``nan``
    values (for example ``nrmse`` of a fold with a zero mean) are ignored.

    :param metrics: The ``metrics`` dictionary returned by
                    :func:`perform_kfold_fit` (target name to a DataFrame
                    with a row per fold).
    :return: A ``pandas.DataFrame`` with a ``(target, metric)`` index
             and the ``mean``, ``median``, ``std`` (sample standard
             deviation), ``p05`` and ``p95`` columns. Select one target with
             ``summary.loc["Vol / ha"]``.

    .. code:: python

        import rsgispy.regression.regsklearn as rsk

        metrics, residuals = rsk.perform_kfold_fit(regrs_obj, x, y, dep_vars)
        summary = rsk.summarise_kfold_metrics(metrics)
        print(summary.loc["Vol / ha"].round(3))
    """
    parts = []
    for name, m_df in metrics.items():
        vals = m_df.drop(
            columns=[c for c in _FOLD_ID_COLS if c in m_df.columns]
        ).apply(pd.to_numeric, errors="coerce")
        summ = pd.DataFrame(
            {
                "mean": vals.mean(),
                "median": vals.median(),
                "std": vals.std(),
                "p05": vals.quantile(0.05),
                "p95": vals.quantile(0.95),
            }
        )
        summ.index = pd.MultiIndex.from_product(
            [[name], summ.index], names=["target", "metric"]
        )
        parts.append(summ)
    if not parts:
        raise rsgispy.RSGISPyException("metrics is empty.")
    return pd.concat(parts)


def _final_estimator(fitted: Any) -> Any:
    """
    The last fitted step of a pipeline (or the target regressor of a
    ``TransformedTargetRegressor``), or the estimator itself.
    """
    est = fitted
    while True:
        if hasattr(est, "steps") and est.steps:
            est = est.steps[-1][1]
        elif hasattr(est, "regressor_"):
            est = est.regressor_
        else:
            return est


def _linear_coefs(fitted: Any, n_targets: int) -> Optional[np.ndarray]:
    """
    The fitted intercept and coefficients as an array (targets, 1 + features)
    with the intercept first, or ``None`` if the estimator has no ``coef_``.
    """
    est = _final_estimator(fitted)
    if isinstance(est, sklearn.multioutput.MultiOutputRegressor):
        per_tgt = [_linear_coefs(sub, 1) for sub in est.estimators_]
        if any(c is None for c in per_tgt):
            return None
        return np.vstack(per_tgt)
    if not hasattr(est, "coef_"):
        return None
    coef = np.atleast_2d(np.asarray(est.coef_, dtype=np.float64))
    if coef.shape[0] != n_targets and coef.shape[1] == n_targets:
        coef = coef.T
    if coef.shape[0] != n_targets:
        return None
    intercept = np.broadcast_to(
        np.atleast_1d(
            np.asarray(getattr(est, "intercept_", 0.0), dtype=np.float64)
        ),
        (n_targets,),
    )
    return np.column_stack([intercept, coef])


def _coef_names(fitted: Any, x: Any, n_coefs: int) -> List[str]:
    """Names of the coefficients (intercept first)."""
    feat_names = None
    if isinstance(x, pd.DataFrame) and x.shape[1] == n_coefs:
        feat_names = [str(c) for c in x.columns]
    elif hasattr(fitted, "steps") and len(fitted.steps) > 1:
        try:
            out_names = list(fitted[:-1].get_feature_names_out())
            if len(out_names) == n_coefs:
                feat_names = [str(c) for c in out_names]
        except (AttributeError, ValueError, TypeError):
            feat_names = None
    if feat_names is None:
        feat_names = [f"x{i}" for i in range(n_coefs)]
    return ["intercept", *feat_names]


def _jackknife_coef_stats(
    full_coefs: np.ndarray,
    loo_coefs: np.ndarray,
    coef_names: Sequence[str],
    target_names: Sequence[str],
) -> Dict[str, pd.DataFrame]:
    """
    Jackknife bias, standard error and 95% interval of each coefficient
    (Efron and Tibshirani, 1993). ``full_coefs`` is (targets, coefs) and
    ``loo_coefs`` is (left-out units, targets, coefs).
    """
    import scipy.stats  # pylint: disable=import-outside-toplevel

    n_units = loo_coefs.shape[0]
    jack_mean = loo_coefs.mean(axis=0)
    bias = (n_units - 1) * (jack_mean - full_coefs)
    std_err = np.sqrt(
        (n_units - 1) / n_units * ((loo_coefs - jack_mean) ** 2).sum(axis=0)
    )
    jack_est = full_coefs - bias
    t_crit = float(scipy.stats.t.ppf(0.975, n_units - 1))
    out = {}
    for i_tgt, name in enumerate(target_names):
        out[name] = pd.DataFrame(
            {
                "estimate": full_coefs[i_tgt],
                "jack_mean": jack_mean[i_tgt],
                "bias": bias[i_tgt],
                "jack_estimate": jack_est[i_tgt],
                "std_err": std_err[i_tgt],
                "ci_lower": jack_est[i_tgt] - t_crit * std_err[i_tgt],
                "ci_upper": jack_est[i_tgt] + t_crit * std_err[i_tgt],
            },
            index=pd.Index(list(coef_names), name="coef"),
        )
    return out


def _jackknife_influence(
    y_t: np.ndarray,
    y_p: np.ndarray,
    unit_test_idx: Sequence[np.ndarray],
    grp_arr: Optional[np.ndarray],
    fit_rmse_delta: np.ndarray,
    pred_shift: np.ndarray,
) -> pd.DataFrame:
    """The influence table of one target (a row per left-out unit)."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    n_smpls = y_t.shape[0]
    cv_rmse = _rmse(y_t, y_p)
    delta_cv = []
    for test_idx in unit_test_idx:
        keep = np.ones(n_smpls, dtype=bool)
        keep[test_idx] = False
        delta_cv.append(_rmse(y_t[keep], y_p[keep]) - cv_rmse)
    first_idx = np.array([t[0] for t in unit_test_idx], dtype=np.int64)
    if grp_arr is None:
        infl_cols: Dict[str, Any] = {
            "sample_idx": first_idx,
            "y_true": y_t[first_idx],
            "y_pred": y_p[first_idx],
        }
    else:
        infl_cols = {
            "group": grp_arr[first_idx],
            "n_samples": [int(t.shape[0]) for t in unit_test_idx],
        }
    infl_cols["delta_cv_rmse"] = np.asarray(delta_cv, dtype=np.float64)
    infl_cols["delta_fit_rmse"] = fit_rmse_delta
    infl_cols["pred_shift_rmse"] = pred_shift
    return pd.DataFrame(infl_cols)


def perform_jackknife_fit(
    regrs_obj: Any,
    x: Any,
    y: Any,
    target_names: Optional[Sequence[str]] = None,
    *,
    groups: Optional[Any] = None,
    calc_coef_stats: bool = True,
    out_dir: Optional[Union[str, os.PathLike]] = None,
    out_prefix: str = "jackknife",
    use_pbar: bool = True,
) -> Dict[str, Any]:
    """
    Assess a regressor with a jackknife (leave-one-out) cross-validation:
    the regressor is fitted once for each sample with that sample left out,
    and predicts the left-out sample. With ``groups`` one group is left out
    at a time (scikit-learn's ``LeaveOneGroupOut``). The pooled
    leave-one-out predictions give one set of accuracy metrics
    (:func:`~rsgispy.regression.regmetrics.calc_reg_metrics`), the
    influence of each sample on the RMSE and, for a linear model, jackknife
    estimates of the bias and standard error of each coefficient.

    This leaves out samples; it is not the same as
    :func:`~rsgispy.regression.regfeatures.calc_jackknife_feature_importance`,
    which leaves out features.

    The regressor is fitted once per sample (or group), plus once to all the
    samples, so it suits small samples (for example a forest inventory of
    about a hundred plots). Use :func:`perform_kfold_fit` for large
    samples. The estimator rules are the same as for
    :func:`perform_kfold_fit`.

    **Influence.** For each left-out sample (or group) :math:`i`:

    * ``delta_cv_rmse``: the RMSE of the pooled leave-one-out predictions
      without sample :math:`i`, minus the RMSE with it. A large negative
      value is a sample that is predicted badly.
    * ``delta_fit_rmse``: the RMSE, on the other samples, of the model
      fitted without sample :math:`i`, minus the RMSE on the same samples of
      the model fitted to all the samples. For a least squares model it is
      never positive, and a large negative value is a sample that pulls the
      model away from the other samples.
    * ``pred_shift_rmse``: the root mean squared difference, over all the
      samples, between the predictions of the model fitted to all the
      samples and of the model fitted without sample :math:`i`. A large
      value is an influential sample, one that changes the model a lot. For
      ordinary least squares with :math:`n` samples and :math:`p`
      coefficients (with the intercept), Cook's distance is
      :math:`n \\, \\text{pred\\_shift\\_rmse}^2 / (p s^2)`, with :math:`s^2`
      the residual variance of the full model.

    **Coefficients.** With ``calc_coef_stats`` and an estimator with
    ``coef_`` / ``intercept_`` (or a pipeline whose last step has them, or a
    ``MultiOutputRegressor`` of them), the jackknife statistics of each
    coefficient :math:`\\theta` are calculated (Efron and Tibshirani, 1993,
    *An Introduction to the Bootstrap*, chapter 11). With :math:`n` the
    number of left-out samples (or groups), :math:`\\hat{\\theta}` the
    estimate from all the samples, :math:`\\hat{\\theta}_{(i)}` the
    estimate without sample :math:`i` and
    :math:`\\hat{\\theta}_{(\\cdot)} = \\frac{1}{n}\\sum_i \\hat{\\theta}_{(i)}`:

    * bias: :math:`(n - 1)(\\hat{\\theta}_{(\\cdot)} - \\hat{\\theta})`;
    * standard error:
      :math:`\\sqrt{\\frac{n - 1}{n} \\sum_i (\\hat{\\theta}_{(i)} - \\hat{\\theta}_{(\\cdot)})^2}`;
    * bias-corrected estimate: :math:`\\hat{\\theta} - \\text{bias}`;
    * 95% interval: the bias-corrected estimate
      :math:`\\pm\\, t_{0.975,\\, n-1}` times the standard error.

    For a pipeline the coefficients are those of the last step, so they are
    in the units of its input (for example the scaled predictors).

    :param regrs_obj: An unfitted regressor (it is not changed).
    :param x: The predictor variables: a NumPy array or a
              ``pandas.DataFrame`` (see :func:`perform_kfold_fit`).
    :param y: The response variables: 1D for one target or 2D (samples,
              targets), as an array, Series or DataFrame.
    :param target_names: The names of the targets; defaults as in
                         :func:`perform_kfold_fit`.
    :param groups: Optional group label of each sample; one group is left
                   out at a time.
    :param calc_coef_stats: Calculate the coefficient statistics when the
                            estimator is linear (Default = ``True``).
    :param out_dir: Optional output directory. When given, the metrics
                    (one row), residuals, influence and coefficient
                    statistics of each target are written to
                    ``<out_prefix>_metrics_<target>.csv``,
                    ``<out_prefix>_residuals_<target>.csv``,
                    ``<out_prefix>_influence_<target>.csv`` and
                    ``<out_prefix>_coef_stats_<target>.csv``, with the
                    target names changed as in :func:`perform_kfold_fit`.
    :param out_prefix: Prefix of the output file names
                       (Default = ``'jackknife'``).
    :param use_pbar: Show a progress bar over the fits (Default = ``True``).
    :return: A dictionary with four dictionaries keyed by target name:

             * ``metrics[target]``: the
               :func:`~rsgispy.regression.regmetrics.calc_reg_metrics`
               dictionary of the pooled leave-one-out predictions;
             * ``residuals[target]``: a DataFrame with the ``sample_idx``
               (position in ``x`` / ``y``), ``group`` (with ``groups``),
               ``y_true`` and ``y_pred`` columns;
             * ``influence[target]``: a DataFrame with a row per left-out
               sample (``sample_idx``, ``y_true``, ``y_pred``) or group
               (``group``, ``n_samples``) and the ``delta_cv_rmse``,
               ``delta_fit_rmse`` and ``pred_shift_rmse`` columns;
             * ``coef_stats[target]``: a DataFrame indexed by coefficient
               (``intercept`` then the predictors) with the ``estimate``,
               ``jack_mean``, ``bias``, ``jack_estimate``, ``std_err``,
               ``ci_lower`` and ``ci_upper`` columns. ``coef_stats`` is
               ``None`` when ``calc_coef_stats`` is ``False`` or the
               estimator has no coefficients.

    .. code:: python

        import pandas
        from sklearn.linear_model import LinearRegression
        import rsgispy.regression.regsklearn as rsk

        df = pandas.read_csv("forest_plots.csv")
        jk = rsk.perform_jackknife_fit(
            LinearRegression(), df[["p95"]], df["Vol / ha"], out_dir="jk_out"
        )
        print(jk["metrics"]["Vol / ha"]["nrmse"])
        print(jk["coef_stats"]["Vol / ha"])
        # The five samples that change the model the most.
        print(jk["influence"]["Vol / ha"].nlargest(5, "pred_shift_rmse"))
    """
    # pylint: disable=too-many-arguments,too-many-locals,too-many-branches
    # pylint: disable=too-many-statements
    check_sklearn_available("perform_jackknife_fit")
    x, y_arr, names, grp_arr = _prep_assess_xy(x, y, target_names, groups)
    if out_dir is not None:
        _target_file_stems(names)
    n_smpls, n_targets = y_arr.shape

    if grp_arr is None:
        splitter = sklearn.model_selection.LeaveOneOut()
        n_units = n_smpls
    else:
        splitter = sklearn.model_selection.LeaveOneGroupOut()
        n_units = int(np.unique(grp_arr).shape[0])
    if n_units < 3:
        raise rsgispy.RSGISPyException(
            f"The jackknife needs at least 3 samples (or groups): got {n_units}."
        )

    all_idx = np.arange(n_smpls)
    full_mdl, full_pred = _fit_predict_rows(regrs_obj, x, y_arr, all_idx, x)
    full_coefs = _linear_coefs(full_mdl, n_targets) if calc_coef_stats else None

    loo_pred = np.full((n_smpls, n_targets), np.nan, dtype=np.float64)
    unit_test_idx: List[np.ndarray] = []
    fit_rmse_delta = np.full((n_units, n_targets), np.nan, dtype=np.float64)
    pred_shift = np.full((n_units, n_targets), np.nan, dtype=np.float64)
    loo_coefs = []
    tqdm_cls = rsgispy.get_tqdm()
    for i_unit, (train_idx, test_idx) in enumerate(
        tqdm_cls(
            splitter.split(y_arr, groups=grp_arr),
            total=n_units,
            disable=not use_pbar,
        )
    ):
        mdl, pred_all = _fit_predict_rows(regrs_obj, x, y_arr, train_idx, x)
        loo_pred[test_idx] = pred_all[test_idx]
        unit_test_idx.append(test_idx)
        for i_tgt in range(n_targets):
            y_other = y_arr[train_idx, i_tgt]
            fit_rmse_delta[i_unit, i_tgt] = _rmse(
                y_other, pred_all[train_idx, i_tgt]
            ) - _rmse(y_other, full_pred[train_idx, i_tgt])
            pred_shift[i_unit, i_tgt] = _rmse(
                full_pred[:, i_tgt], pred_all[:, i_tgt]
            )
        if full_coefs is not None:
            unit_coefs = _linear_coefs(mdl, n_targets)
            if unit_coefs is None or unit_coefs.shape != full_coefs.shape:
                full_coefs = None
            else:
                loo_coefs.append(unit_coefs)

    metrics: Dict[str, Dict[str, Any]] = {}
    residuals: Dict[str, pd.DataFrame] = {}
    influence: Dict[str, pd.DataFrame] = {}
    for i_tgt, name in enumerate(names):
        metrics[name] = _pooled_metrics(y_arr[:, i_tgt], loo_pred[:, i_tgt])
        resid_cols: Dict[str, Any] = {"sample_idx": all_idx}
        if grp_arr is not None:
            resid_cols["group"] = grp_arr
        resid_cols["y_true"] = y_arr[:, i_tgt]
        resid_cols["y_pred"] = loo_pred[:, i_tgt]
        residuals[name] = pd.DataFrame(resid_cols)
        influence[name] = _jackknife_influence(
            y_arr[:, i_tgt],
            loo_pred[:, i_tgt],
            unit_test_idx,
            grp_arr,
            fit_rmse_delta[:, i_tgt],
            pred_shift[:, i_tgt],
        )

    coef_stats = None
    if full_coefs is not None and len(loo_coefs) == n_units:
        coef_stats = _jackknife_coef_stats(
            full_coefs,
            np.stack(loo_coefs),
            _coef_names(full_mdl, x, full_coefs.shape[1] - 1),
            names,
        )

    if out_dir is not None:
        tables: Dict[str, Dict[str, pd.DataFrame]] = {
            "metrics": {n: pd.DataFrame([metrics[n]]) for n in names},
            "residuals": residuals,
            "influence": influence,
        }
        if coef_stats is not None:
            tables["coef_stats"] = coef_stats
        _write_target_csvs(
            out_dir, out_prefix, tables, names, index_tables=("coef_stats",)
        )

    return {
        "metrics": metrics,
        "residuals": residuals,
        "influence": influence,
        "coef_stats": coef_stats,
    }
