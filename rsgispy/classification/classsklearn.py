"""
Scikit-learn classifier hyperparameter optimisation (Grid Search, Random Search,
and Bayesian optimisation via Optuna), classifier training, and block-wise
raster classification outputting class IDs and probability score maps.
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

import numpy as np
import pandas as pd

from rsgispy.classification._cls_common import (
    apply_classifier_raster as _apply_classifier_raster,
)
from rsgispy.classification._helpers import (
    seg_allows_nan,
    ClsConfigInput,
    SamplesInput,
    calc_split_metrics as _calc_split_metrics,
    check_sklearn_available,
    coerce_cls_config as _coerce_cls_config,
    extract_xy as _extract_xy,
    load_samples_df as _load_samples_df,
    resolve_feature_cols as _resolve_feature_cols,
)
from rsgispy.classification.classdata import ImageBandInfo
from rsgispy.imageprocess import ensure_file_exists
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
)
from rsgispy.tools._sklearn_spaces import (
    build_sklearn_param_space as _build_sklearn_param_space,
)
from rsgispy.tools.files import ensure_parent_dir
from rsgispy.tools.jsonio import write_dict_to_json

__all__ = [
    "get_sklearn_classifier_param_space",
    "save_sklearn_classifier",
    "load_sklearn_classifier",
    "perform_sklearn_classifier_param_search",
    "train_sklearn_classifier",
    "apply_sklearn_classifier",
]


def save_sklearn_classifier(
    sk_classifier: Any,
    out_file: Union[str, os.PathLike],
) -> None:
    """
    Save a trained ``scikit-learn`` (or compatible) classifier object to disk
    using ``joblib``.

    :param sk_classifier: Classifier object to persist.
    :param out_file: Output file path (e.g., ``'rf_classifier.joblib'``).

    .. code:: python

        from sklearn.ensemble import RandomForestClassifier
        import rsgispy.classification.classsklearn as csk

        clf = RandomForestClassifier(n_estimators=50)
        csk.save_sklearn_classifier(clf, "rf_classifier.joblib")
    """
    save_joblib_model(
        model=sk_classifier,
        out_file=out_file,
        have_pkg=True,
        pkg_name="joblib",
        extra="classification",
        func_name="save_sklearn_classifier",
    )


def load_sklearn_classifier(
    in_file: Union[str, os.PathLike],
) -> Any:
    """
    Load a persisted ``scikit-learn`` (or compatible) classifier object from
    disk using ``joblib``.

    :param in_file: Path to the saved classifier file.
    :return: Loaded classifier object.

    .. code:: python

        import rsgispy.classification.classsklearn as csk

        clf = csk.load_sklearn_classifier("rf_classifier.joblib")
    """
    in_path = os.fspath(in_file)
    ensure_file_exists(in_path, label="Classifier model")
    return load_joblib_model(
        in_file=in_path,
        have_pkg=True,
        pkg_name="joblib",
        extra="classification",
        func_name="load_sklearn_classifier",
        label="Classifier model",
    )


def get_sklearn_classifier_param_space(
    sk_classifier: Any,
    search_method: str = "optuna",
    extended: bool = False,
) -> Union[Dict[str, Any], Callable[[Any], Dict[str, Any]]]:
    """
    Return a standard built-in hyperparameter search space for a commonly used
    ``scikit-learn`` classifier (or a ``Pipeline`` ending with one).

    Supports ``RandomForestClassifier``, ``ExtraTreesClassifier``,
    ``GradientBoostingClassifier``, ``HistGradientBoostingClassifier``,
    ``SVC``, ``MLPClassifier``, ``KNeighborsClassifier``,
    ``LogisticRegression``, ``DecisionTreeClassifier``, and
    ``AdaBoostClassifier`` (passed as an instance, class, ``Pipeline``, or
    name/alias string such as ``'rf'``, ``'svc'``, or ``'mlp'``).

    When ``search_method='optuna'`` (default), returns Optuna parameter
    specifications (or a conditional callable ``fn(trial) -> dict`` when
    ``extended=True`` for ``SVC`` or ``MLPClassifier``). When
    ``search_method='grid'`` or ``'random'``, returns discrete candidate value
    lists suitable for ``GridSearchCV`` and ``RandomizedSearchCV``.

    :param sk_classifier: Classifier instance, class, ``Pipeline``, or name /
                          alias string.
    :param search_method: Target search strategy: ``'optuna'`` (default),
                          ``'grid'``, or ``'random'``.
    :param extended: If ``True``, return a broader search space with additional
                     regularisation and structural parameters; if ``False``
                     (default), return the standard key parameter space.
    :return: Search space dictionary or Optuna ``fn(trial) -> dict`` callable.

    .. code:: python

        from sklearn.ensemble import RandomForestClassifier
        import rsgispy.classification.classsklearn as csk

        optuna_space = csk.get_sklearn_classifier_param_space(
            RandomForestClassifier(), search_method="optuna"
        )
        grid_space = csk.get_sklearn_classifier_param_space(
            "svc", search_method="grid", extended=True
        )
    """
    method = _normalise_search_method(search_method)
    return _build_sklearn_param_space(
        sk_estimator=sk_classifier,
        method=method,
        extended=bool(extended),
        task="classification",
    )


def perform_sklearn_classifier_param_search(
    cls_cfg: ClsConfigInput,
    sk_classifier: Any,
    train_data: SamplesInput,
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
    scoring: Union[str, Callable[..., float]] = "accuracy",
    cv: Union[int, Any] = 5,
    n_iter: int = 20,
    n_trials: int = 50,
    timeout: Optional[float] = None,
    direction: str = "maximize",
    n_jobs: Optional[int] = None,
    rnd_seed: Optional[int] = None,
    feature_cols: Optional[Sequence[str]] = None,
    use_internal_id: bool = True,
    refit: bool = True,
    out_id_col: Optional[str] = None,
    internal_id_col: Optional[str] = None,
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
    Optimise hyperparameters for a ``scikit-learn`` (or compatible) classifier
    using Grid Search, Random Search, or Bayesian optimisation with Optuna.

    When ``param_space=None`` (or ``'default'`` / ``'extended'``) and
    ``extended_param_space`` controls preset breadth, built-in standard search
    spaces are automatically selected for common ``scikit-learn`` classifiers
    (including ``RandomForestClassifier``, ``ExtraTreesClassifier``,
    ``GradientBoostingClassifier``, ``HistGradientBoostingClassifier``,
    ``SVC``, ``MLPClassifier``, ``KNeighborsClassifier``,
    ``LogisticRegression``, ``DecisionTreeClassifier``, and
    ``AdaBoostClassifier``, as well as ``Pipeline`` objects ending in any of
    those). See :func:`get_sklearn_classifier_param_space`.

    When ``valid_data`` is provided, candidate parameter configurations are fit
    on ``train_data`` and evaluated on ``valid_data`` (with the final optimal
    estimator refit only on ``train_data`` when ``refit=True``). When
    ``valid_data`` is ``None``, stratified ``cv``-fold cross-validation is
    performed on ``train_data``.

    Random folds put neighbouring pixels (e.g. from the same training
    polygon) in both the training and validation folds, which favours
    parameters that memorise the training polygons. The spatial and group
    splitters of :mod:`rsgispy.ml.cv` can be passed directly as ``cv`` with
    the sample coordinates (``coords``, for
    :class:`~rsgispy.ml.cv.SpatialBlockKFold` and
    :class:`~rsgispy.ml.cv.SpatialBufferCV`) or groups (``groups``, e.g. the
    ``poly_id_col`` column of
    :func:`~rsgispy.classification.classdata.extract_class_training_samples`,
    for :class:`~rsgispy.ml.cv.PolygonGroupKFold`); scikit-learn group
    splitters (e.g. ``StratifiedGroupKFold``) receive ``groups``. With
    ``groups`` or ``coords`` an integer ``cv`` gives a class-stratified
    ``SpatialBlockKFold`` (``coords``) or ``PolygonGroupKFold`` (``groups``)
    with ``cv`` folds. The folds are made once and used for every candidate
    (they are in ``results_dict['search_obj'].cv`` for grid and random
    searches).

    For ``search_method='optuna'``, a custom ``param_space`` may be either:

    * A callable ``fn(trial) -> dict`` using ``optuna.Trial`` suggestions; or
    * A dictionary mapping parameter names to lists (categorical), 2-tuples
      ``(low, high)`` (numeric range), or explicit spec dicts such as
      ``{'type': 'float', 'low': 1e-3, 'high': 1.0, 'log': True}``.

    :param cls_cfg: :class:`~rsgispy.classification.classdata.ClassificationConfig`,
                    JSON config path, or mapping of ``{class_name: ClassInfoObj}``.
    :param sk_classifier: Base ``scikit-learn`` classifier estimator, ``Pipeline``,
                          or pre-configured ``GridSearchCV`` / ``RandomizedSearchCV``.
    :param train_data: Training sample :class:`pandas.DataFrame` or file path
                       (Parquet, CSV, etc.).
    :param param_space: Parameter search space (dict, list of dicts, Optuna
                        callable, preset string ``'default'`` / ``'extended'``,
                        or ``None`` to use the built-in standard space for
                        supported classifiers).
    :param search_method: Optimisation strategy: ``'grid'`` (default),
                          ``'random'``, or ``'optuna'``.
    :param valid_data: Optional validation :class:`pandas.DataFrame` or file path
                       used to score candidates out-of-sample.
    :param scoring: Scikit-learn metric name or scorer callable
                    (Default = ``'accuracy'``).
    :param cv: Cross-validation folds or splitter used when ``valid_data`` is
               ``None`` (Default = ``5``).
    :param n_iter: Number of parameter samples for ``search_method='random'``
                   (Default = ``20``).
    :param n_trials: Number of Optuna trials for ``search_method='optuna'``
                     (Default = ``50``).
    :param timeout: Optional time limit in seconds for Optuna optimisation.
    :param direction: Optuna optimisation direction (``'maximize'`` or
                      ``'minimize'``, Default = ``'maximize'``).
                      Named scikit-learn scorers are greater-is-better, so
                      ``'minimize'`` is only valid with a callable ``scoring``.
    :param n_jobs: Number of parallel workers for scikit-learn CV evaluations.
    :param rnd_seed: Optional random seed for reproducible search and splits.
    :param feature_cols: Optional ordered feature column names (inferred from
                         ``cls_cfg`` when ``None``).
    :param use_internal_id: If ``True`` (default), train against consecutive
                            0-based ``internal_cls_id`` values; otherwise use
                            ``out_cls_id``.
    :param refit: If ``True`` (default), fit the returned optimal estimator on
                  ``train_data``; if ``False``, return an unfitted clone
                  initialised with the optimal parameters.
    :param out_id_col: Optional override for the output class ID column name.
    :param internal_id_col: Optional override for the internal class ID column
                            name.
    :param out_results_file: Optional path (e.g., ``'search_results.parquet'``
                             or ``'search_results.csv'``) to save the trials /
                             CV results table.
    :param out_params_json: Optional JSON path to save best parameters and score.
    :param return_results: If ``True``, return ``(best_estimator, results_dict)``;
                           if ``False`` (default), return ``best_estimator``.
    :param verbose: Verbosity level (Default = ``0``).
    :param extended_param_space: If ``True`` (and ``param_space`` is ``None``),
                                 use the broader extended built-in parameter
                                 space for ``sk_classifier`` (Default = ``False``).
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
    :return: Optimised classifier estimator, or ``(best_estimator, results_dict)``
             when ``return_results=True``.

    .. code:: python

        from sklearn.ensemble import RandomForestClassifier
        import rsgispy.classification.classsklearn as csk
        from rsgispy.ml.cv import SpatialBlockKFold

        # Automatically uses built-in RandomForestClassifier Optuna space
        best_clf = csk.perform_sklearn_classifier_param_search(
            cls_cfg="cls_config.json",
            sk_classifier=RandomForestClassifier(),
            train_data="train.parquet",
            search_method="optuna",
            n_trials=25,
            rnd_seed=42,
        )
        # Spatially blocked folds (train.parquet has x and y columns)
        best_clf = csk.perform_sklearn_classifier_param_search(
            "cls_config.json",
            RandomForestClassifier(),
            "train.parquet",
            param_space={"min_samples_leaf": [1, 5, 20]},
            cv=SpatialBlockKFold(n_splits=4, block_size=2000.0, stratify=True),
            coords=("x", "y"),
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    check_sklearn_available("perform_sklearn_classifier_param_search")
    method = _normalise_search_method(search_method)
    eff_param_space, space_mode = _tools_resolve_sklearn_param_space(
        sk_estimator=sk_classifier,
        method=method,
        param_space=param_space,
        extended_param_space=extended_param_space,
        task="classification",
    )
    cfg, class_info, eff_out_col, eff_int_col, cls_name_col = _coerce_cls_config(
        cls_cfg, out_id_col=out_id_col, internal_id_col=internal_id_col
    )
    target_col = eff_int_col if use_internal_id else eff_out_col
    valid_ids = [
        int(obj.internal_id) if use_internal_id else int(obj.out_id)  # type: ignore[arg-type]
        for obj in class_info.values()
    ]

    train_df = _load_samples_df(train_data, "train_data")
    feat_cols = _resolve_feature_cols(
        cfg,
        train_df,
        feature_cols,
        excluded_cols=[eff_out_col, eff_int_col, cls_name_col],
    )
    train_x, train_y = _extract_xy(
        train_df, feat_cols, target_col, "train_data", valid_ids=valid_ids,
        allow_nan=seg_allows_nan(cfg),
    )
    cv_groups, cv_coords = _resolve_cv_groups_coords(train_df, groups, coords)

    valid_x: Optional[pd.DataFrame] = None
    valid_y: Optional[np.ndarray] = None
    if valid_data is not None:
        valid_df = _load_samples_df(valid_data, "valid_data")
        valid_x, valid_y = _extract_xy(
            valid_df, feat_cols, target_col, "valid_data", valid_ids=valid_ids,
            allow_nan=seg_allows_nan(cfg),
        )

    if method == "optuna":
        best_est, best_params, best_score, results_df, search_obj = (
            _tools_run_optuna_search(
                sk_estimator=sk_classifier,
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
                stratified=True,
                func_name="perform_sklearn_classifier_param_search",
                extra="classification",
                use_pbar=use_pbar,
                groups=cv_groups,
                coords=cv_coords,
            )
        )
    else:
        best_est, best_params, best_score, results_df, search_obj = (
            _tools_run_sklearn_search_cv(
                method=method,
                sk_estimator=sk_classifier,
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
                stratified=True,
                extra="classification",
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


def train_sklearn_classifier(
    cls_cfg: ClsConfigInput,
    sk_classifier: Any,
    train_data: SamplesInput,
    test_data: Optional[SamplesInput] = None,
    valid_data: Optional[SamplesInput] = None,
    feature_cols: Optional[Sequence[str]] = None,
    use_internal_id: bool = True,
    rnd_seed: Optional[int] = None,
    out_id_col: Optional[str] = None,
    internal_id_col: Optional[str] = None,
    out_model_file: Optional[Union[str, os.PathLike]] = None,
    out_metrics_file: Optional[Union[str, os.PathLike]] = None,
    return_model: bool = False,
) -> Union[Dict[str, float], Tuple[Any, Dict[str, float]]]:
    """
    Train a ``scikit-learn`` (or compatible) classifier on extracted sample
    tables using the class and band configuration in ``cls_cfg``, optionally
    evaluating on testing and validation datasets.

    By default (``return_model=False``), fits a clean clone and updates
    ``sk_classifier`` in place, returning the evaluation metrics dictionary.
    When ``return_model=True``, ``sk_classifier`` is left untouched and
    ``(fitted_model, metrics)`` is returned.

    :param cls_cfg: :class:`~rsgispy.classification.classdata.ClassificationConfig`,
                    JSON config path, or mapping of ``{class_name: ClassInfoObj}``.
    :param sk_classifier: ``scikit-learn`` classifier estimator template.
    :param train_data: Training sample :class:`pandas.DataFrame` or file path.
    :param test_data: Optional testing sample :class:`pandas.DataFrame` or path.
    :param valid_data: Optional validation sample :class:`pandas.DataFrame` or
                       path.
    :param feature_cols: Optional ordered feature column names (inferred from
                         ``cls_cfg`` when ``None``).
    :param use_internal_id: If ``True`` (default), train using consecutive
                            0-based ``internal_cls_id`` labels (recommended for
                            classifiers such as XGBoost); otherwise use
                            ``out_cls_id``.
    :param rnd_seed: Optional random seed passed to the estimator when
                     supported.
    :param out_id_col: Optional override for the output class ID column name.
    :param internal_id_col: Optional override for the internal class ID column
                            name.
    :param out_model_file: Optional output ``.joblib`` path to save the fitted
                           classifier.
    :param out_metrics_file: Optional output JSON path to save training/test/
                             validation metrics.
    :param return_model: When ``True``, return ``(fitted_model, metrics)``
                         without mutating ``sk_classifier``.
    :return: Metrics dictionary (or ``(fitted_model, metrics)`` when
             ``return_model=True``).

    .. code:: python

        from sklearn.ensemble import RandomForestClassifier
        import rsgispy.classification.classsklearn as csk

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        metrics = csk.train_sklearn_classifier(
            cls_cfg="cls_config.json",
            sk_classifier=clf,
            train_data="train.parquet",
            test_data="test.parquet",
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    check_sklearn_available("train_sklearn_classifier")
    cfg, class_info, eff_out_col, eff_int_col, cls_name_col = _coerce_cls_config(
        cls_cfg, out_id_col=out_id_col, internal_id_col=internal_id_col
    )
    target_col = eff_int_col if use_internal_id else eff_out_col
    valid_ids = [
        int(obj.internal_id) if use_internal_id else int(obj.out_id)  # type: ignore[arg-type]
        for obj in class_info.values()
    ]

    train_df = _load_samples_df(train_data, "train_data")
    feat_cols = _resolve_feature_cols(
        cfg,
        train_df,
        feature_cols,
        excluded_cols=[eff_out_col, eff_int_col, cls_name_col],
    )
    train_x, train_y = _extract_xy(
        train_df, feat_cols, target_col, "train_data", valid_ids=valid_ids,
        allow_nan=seg_allows_nan(cfg),
    )

    fitted_est = fit_clean_sklearn_estimator(
        sk_classifier,
        train_x,
        train_y,
        sync_input=not return_model,
        rnd_seed=rnd_seed,
    )

    metrics = _calc_split_metrics(fitted_est, train_x, train_y, "train")
    if test_data is not None:
        test_df = _load_samples_df(test_data, "test_data")
        test_x, test_y = _extract_xy(
            test_df, feat_cols, target_col, "test_data", valid_ids=valid_ids,
            allow_nan=seg_allows_nan(cfg),
        )
        metrics.update(_calc_split_metrics(fitted_est, test_x, test_y, "test"))
    if valid_data is not None:
        valid_df = _load_samples_df(valid_data, "valid_data")
        valid_x, valid_y = _extract_xy(
            valid_df, feat_cols, target_col, "valid_data", valid_ids=valid_ids,
            allow_nan=seg_allows_nan(cfg),
        )
        metrics.update(
            _calc_split_metrics(fitted_est, valid_x, valid_y, "valid")
        )

    if out_model_file is not None:
        save_sklearn_classifier(fitted_est, out_model_file)

    if out_metrics_file is not None:
        metrics_path = ensure_parent_dir(out_metrics_file)
        write_dict_to_json(metrics, metrics_path)

    if return_model:
        return fitted_est, metrics
    return metrics


def apply_sklearn_classifier(
    cls_cfg: ClsConfigInput,
    sk_classifier: Union[Any, str, os.PathLike],
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
    out_conf_img: Optional[Union[str, os.PathLike]] = None,
    conf_method: str = "margin",
) -> None:
    """
    Apply a trained ``scikit-learn`` (or compatible) classifier block-by-block
    to image pixels selected by ``in_msk_img == img_msk_val``, outputting the
    classified class ID raster and/or classification probability score raster.

    When the classifier was trained on consecutive 0-based ``internal_cls_id``
    values (``use_internal_id=True``) and ``recode_to_out_ids=True`` (default),
    predicted class IDs are automatically recoded back to the user-defined
    unsigned ``out_id`` values from ``cls_cfg``.

    :param cls_cfg: :class:`~rsgispy.classification.classdata.ClassificationConfig`,
                    JSON config path, or mapping of ``{class_name: ClassInfoObj}``.
    :param sk_classifier: Fitted classifier instance or path to a saved
                          ``.joblib`` model file.
    :param in_msk_img: Path to the mask image controlling which pixels are
                       classified.
    :param img_msk_val: Pixel value within ``in_msk_img`` identifying pixels to
                        classify.
    :param output_img: Optional output classified ID raster path. At least
                       one of ``output_img`` or ``out_score_img`` must be
                       provided.
    :param out_score_img: Optional output ``float32`` probability score raster
                          path (requires classifier ``predict_proba``). By
                          default writes one band per class ordered by class ID;
                          set ``score_max_only=True`` for a single-band maximum
                          probability raster.
    :param img_band_info: Optional sequence of
                          :class:`~rsgispy.classification.classdata.ImageBandInfo`
                          overriding ``cls_cfg.img_band_info`` (required when
                          ``cls_cfg`` is a plain class-info mapping).
    :param gdalformat: Output GDAL/rasterio format (Default = ``'GTIFF'``).
    :param datatype: Optional ``rsgispy.TYPE_*`` integer datatype for
                     ``output_img``. When ``None`` (default), automatically
                     selects the smallest unsigned integer datatype fitting all
                     output class IDs and nodata.
    :param out_no_data_val: Optional unsigned integer nodata value for
                            ``output_img``. When ``None`` (default), selects
                            ``0`` if unused by any output class ID, otherwise the
                            next unused unsigned integer.
    :param class_clr_names: If ``True`` (default), apply class RGB colour table
                            to ``output_img`` when output dtype is ``uint8``
                            or ``uint16``.
    :param use_internal_id: Whether ``sk_classifier`` predicts 0-based
                            ``internal_id`` values (Default = ``True``).
    :param recode_to_out_ids: If ``True`` (default) and ``use_internal_id=True``,
                              recode predictions to ``out_id`` values before
                              writing ``output_img``.
    :param score_max_only: If ``False`` (default), write one probability band
                           per class to ``out_score_img``; if ``True``, write a
                           single band containing maximum class probability.
    :param score_no_data_val: Nodata fill value for ``out_score_img``
                              (Default = ``-1.0``).
    :param calc_stats: If ``True`` (default), compute output statistics and
                       overview pyramids.
    :param block_size: Optional processing block size in pixels.
    :param creation_opts: Optional GDAL creation options mapping.
    :param use_pbar: Display a tqdm progress bar during classification
                     (Default = ``True``).
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

        import rsgispy.classification.classsklearn as csk

        csk.apply_sklearn_classifier(
            cls_cfg="cls_config.json",
            sk_classifier=clf,
            in_msk_img="valid_mask.tif",
            img_msk_val=1,
            output_img="classified.tif",
            out_score_img="class_probs.tif",
            out_conf_img="class_conf.tif",
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    check_sklearn_available("apply_sklearn_classifier")
    estimator = (
        load_sklearn_classifier(sk_classifier)
        if isinstance(sk_classifier, (str, os.PathLike))
        else sk_classifier
    )
    _apply_classifier_raster(
        cls_cfg=cls_cfg,
        estimator=estimator,
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
        binary=False,
        out_conf_img=out_conf_img,
        conf_method=conf_method,
    )
