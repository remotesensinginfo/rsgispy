"""
Scikit-learn classifier hyperparameter optimisation (Grid Search, Random Search,
and Bayesian optimisation via Optuna), classifier training, block-wise
raster classification outputting class IDs and probability score maps, and
repeated k-fold and jackknife (leave-one-out) assessment of any classifier.
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
from rsgispy.classification._cls_common import (
    apply_classifier_raster as _apply_classifier_raster,
)
from rsgispy.classification._helpers import (
    ClsConfigInput,
    SamplesInput,
    calc_label_metrics as _calc_label_metrics,
    calc_split_metrics as _calc_split_metrics,
    check_sklearn_available,
    coerce_cls_config as _coerce_cls_config,
    extract_xy as _extract_xy,
    resolve_feature_cols as _resolve_feature_cols,
)
from rsgispy.tools._ml_data import (
    seg_allows_nan,
    load_samples_df as _load_samples_df,
)
from rsgispy.classification.classdata import ImageBandInfo
from rsgispy.imageprocess import ensure_file_exists
from rsgispy.tools._gbdt_common import load_joblib_model, save_joblib_model
from rsgispy.tools._ml_assess import (
    coef_names,
    final_estimator,
    jackknife_coef_stats,
    jackknife_splits,
    prep_assess_groups,
    prep_assess_x,
    repeated_kfold_splits,
    summarise_fold_metrics,
    take_rows,
    write_named_csvs,
)
from rsgispy.tools._ml_tune import (
    normalise_search_method as _normalise_search_method,
    resolve_cv_groups_coords as _resolve_cv_groups_coords,
    resolve_sklearn_param_space as _tools_resolve_sklearn_param_space,
    run_param_search as _tools_run_param_search,
    save_search_outputs as _save_search_outputs,
)
from rsgispy.tools._sklearn import (
    fit_clean_sklearn_estimator,
    sklearn,
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
    "perform_kfold_fit",
    "summarise_kfold_metrics",
    "perform_jackknife_fit",
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

    best_est, best_params, best_score, results_df, search_obj = (
        _tools_run_param_search(
            method,
            sk_classifier,
            eff_param_space,
            train_x,
            train_y,
            valid_x,
            valid_y,
            scoring=scoring,
            cv=cv,
            n_trials=n_trials,
            timeout=timeout,
            direction=direction,
            n_iter=n_iter,
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


# ---------------------------------------------------------------------------
# Repeated k-fold and jackknife (leave-one-out) assessment of a classifier
# ---------------------------------------------------------------------------

def _prep_cls_assess_xy(
    x: Any, y: Any, groups: Optional[Any]
) -> Tuple[Any, np.ndarray, Optional[np.ndarray]]:
    """
    Check and normalise the ``x`` / ``y`` / ``groups`` inputs of the
    classifier k-fold and jackknife functions. ``y`` is returned as a 1D
    array of class labels (any dtype).
    """
    x = prep_assess_x(x)
    if isinstance(y, pd.DataFrame):
        if y.shape[1] != 1:
            raise rsgispy.RSGISPyException(
                f"y must hold one class label column: got {y.shape[1]} columns."
            )
        y = y.iloc[:, 0]
    y_arr = np.asarray(y.to_numpy() if isinstance(y, pd.Series) else y)
    if y_arr.ndim == 2 and y_arr.shape[1] == 1:
        y_arr = y_arr[:, 0]
    if y_arr.ndim != 1:
        raise rsgispy.RSGISPyException(
            f"y must be 1D (one class label per sample): got shape {y_arr.shape}."
        )
    if x.shape[0] != y_arr.shape[0]:
        raise rsgispy.RSGISPyException(
            f"x has {x.shape[0]} samples but y has {y_arr.shape[0]}."
        )
    return x, y_arr, prep_assess_groups(groups, y_arr.shape[0])


def _fit_predict_cls_rows(
    clf_obj: Any,
    x: Any,
    y_arr: np.ndarray,
    train_idx: np.ndarray,
    pred_x: Any,
) -> Tuple[Any, np.ndarray]:
    """Fit a clone of ``clf_obj`` to the ``train_idx`` rows and predict ``pred_x``."""
    mdl = sklearn.base.clone(clf_obj)
    mdl.fit(take_rows(x, train_idx), y_arr[train_idx])
    return mdl, np.asarray(mdl.predict(pred_x)).ravel()


def _label_metrics_row(y_t: np.ndarray, y_p: np.ndarray) -> Dict[str, Any]:
    """``n_samples`` plus the accuracy metrics of predicted class labels."""
    row: Dict[str, Any] = {"n_samples": int(y_t.shape[0])}
    row.update(_calc_label_metrics(y_t, y_p))
    return row


def _accuracy(y_t: np.ndarray, y_p: np.ndarray) -> float:
    """Overall accuracy (``nan`` when there are no samples)."""
    if y_t.shape[0] == 0:
        return float("nan")
    return float(np.mean(y_t == y_p))


def _cls_linear_coefs(fitted: Any) -> Optional[Tuple[np.ndarray, List[str]]]:
    """
    The fitted intercept and coefficients of a linear classifier as an array
    (rows, 1 + features) with the intercept first, and the class label of
    each row (a binary model has one row, for the second, positive, class).
    ``None`` if the estimator has no ``coef_``.
    """
    est = final_estimator(fitted)
    if not hasattr(est, "coef_"):
        return None
    classes = list(getattr(est, "classes_", getattr(fitted, "classes_", [])))
    coef = np.atleast_2d(np.asarray(est.coef_, dtype=np.float64))
    n_rows = coef.shape[0]
    if n_rows == 1 and len(classes) == 2:
        row_names = [str(classes[1])]
    elif n_rows == len(classes):
        row_names = [str(c) for c in classes]
    else:
        return None
    intercept = np.broadcast_to(
        np.atleast_1d(np.asarray(getattr(est, "intercept_", 0.0), dtype=np.float64)),
        (n_rows,),
    )
    return np.column_stack([intercept, coef]), row_names


def perform_kfold_fit(
    clf_obj: Any,
    x: Any,
    y: Any,
    *,
    n_splits: int = 5,
    n_repeats: int = 20,
    groups: Optional[Any] = None,
    stratify: bool = True,
    rnd_seed: Optional[int] = 42,
    out_dir: Optional[Union[str, os.PathLike]] = None,
    out_prefix: str = "kfold",
    use_pbar: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Assess a classifier with a repeated k-fold cross-validation, the
    classification equivalent of
    :func:`rsgispy.regression.regsklearn.perform_kfold_fit`. In each repeat
    the samples are split at random into ``n_splits`` folds, and each fold
    is predicted by a copy of the classifier fitted to the other folds. The
    overall accuracy, balanced accuracy, macro F1 and Cohen's kappa of each
    fold are returned with the out-of-fold predictions of every repeat. Use
    :func:`summarise_kfold_metrics` to summarise the metrics.

    Without ``groups`` the splits are scikit-learn's
    ``RepeatedStratifiedKFold`` (each fold has about the same class
    proportions as the whole sample), or ``RepeatedKFold`` with
    ``stratify=False``. With ``groups`` the unique groups are shuffled (with
    a new order in each repeat) and split into ``n_splits`` folds with
    (nearly) the same number of groups, so the samples of a group (for
    example the pixels of one training polygon) are never in both the train
    and test samples; the folds are then not stratified.

    The classifier can be any unfitted estimator with ``fit`` / ``predict``,
    including a scikit-learn ``Pipeline`` (so a scaler is fitted to the
    training folds only) and the scikit-learn wrappers of XGBoost, LightGBM
    and CatBoost. It is cloned for each fit.

    :param clf_obj: An unfitted classifier (it is not changed).
    :param x: The predictor variables: a NumPy array (samples, predictors)
              or a ``pandas.DataFrame`` (passed to the estimator as a
              DataFrame, so the column names are kept).
    :param y: The class label of each sample: a 1D array or
              ``pandas.Series`` (integer class IDs or class names).
    :param n_splits: The number of folds (Default = 5).
    :param n_repeats: The number of times the k-fold is repeated, each time
                      with a different random split (Default = 20).
    :param groups: Optional group label of each sample; a group is never
                   split across the train and test samples.
    :param stratify: Stratify the folds by class when ``groups`` is not
                     given (Default = ``True``).
    :param rnd_seed: Seed for the random splits, so the results can be
                     repeated (Default = 42). Seed the estimator itself (for
                     example ``random_state``) for identical results.
    :param out_dir: Optional output directory. When given, the metrics and
                    predictions are written to ``<out_prefix>_metrics.csv``
                    and ``<out_prefix>_predictions.csv``.
    :param out_prefix: Prefix of the output file names (Default = ``'kfold'``).
    :param use_pbar: Show a progress bar over the fits (Default = ``True``).
    :return: ``(metrics, predictions)``. ``metrics`` is a
             ``pandas.DataFrame`` with one row per fold and repeat: the
             ``repeat`` and ``fold`` numbers (from 0), the number of test
             samples (``n_samples``) and the ``accuracy``, ``bal_accuracy``,
             ``f1_macro`` and ``cohen_kappa`` metrics. ``predictions`` is a
             ``pandas.DataFrame`` of the out-of-fold predictions of every
             repeat, with the ``repeat``, ``fold``, ``sample_idx`` (the
             position of the sample in ``x`` / ``y``), ``y_true`` and
             ``y_pred`` columns.

    .. code:: python

        import pandas
        from sklearn.ensemble import RandomForestClassifier
        import rsgispy.classification.classsklearn as csk

        df = pandas.read_parquet("train.parquet")
        metrics, preds = csk.perform_kfold_fit(
            RandomForestClassifier(n_estimators=200, random_state=0),
            df[["blue", "green", "red", "nir"]],
            df["class_name"],
            n_splits=5,
            n_repeats=10,
            groups=df["poly_id"],
            out_dir="kfold_outputs",
        )
        print(csk.summarise_kfold_metrics(metrics))
    """
    # pylint: disable=too-many-arguments,too-many-locals
    check_sklearn_available("perform_kfold_fit")
    if int(n_splits) < 2:
        raise rsgispy.RSGISPyException(f"n_splits must be >= 2: got {n_splits}.")
    x, y_arr, grp_arr = _prep_cls_assess_xy(x, y, groups)
    splits, n_splits, n_repeats = repeated_kfold_splits(
        y_arr.shape[0],
        n_splits,
        n_repeats,
        groups=grp_arr,
        rnd_seed=rnd_seed,
        strat_y=y_arr if stratify else None,
    )

    fold_rows: List[Dict[str, Any]] = []
    pred_parts: List[pd.DataFrame] = []
    tqdm_cls = rsgispy.get_tqdm()
    for i_rep, i_fold, train_idx, test_idx in tqdm_cls(
        splits, total=n_splits * n_repeats, disable=not use_pbar
    ):
        _, y_hat = _fit_predict_cls_rows(
            clf_obj, x, y_arr, train_idx, take_rows(x, test_idx)
        )
        y_test = y_arr[test_idx]
        row: Dict[str, Any] = {"repeat": i_rep, "fold": i_fold}
        row.update(_label_metrics_row(y_test, y_hat))
        fold_rows.append(row)
        pred_parts.append(
            pd.DataFrame(
                {
                    "repeat": np.full(test_idx.shape[0], i_rep, dtype=np.int64),
                    "fold": np.full(test_idx.shape[0], i_fold, dtype=np.int64),
                    "sample_idx": test_idx.astype(np.int64),
                    "y_true": y_test,
                    "y_pred": y_hat,
                }
            )
        )

    metrics = pd.DataFrame(fold_rows)
    predictions = pd.concat(pred_parts, ignore_index=True)
    if out_dir is not None:
        out_dir = os.fspath(out_dir)
        os.makedirs(out_dir, exist_ok=True)
        metrics.to_csv(os.path.join(out_dir, f"{out_prefix}_metrics.csv"), index=False)
        predictions.to_csv(
            os.path.join(out_dir, f"{out_prefix}_predictions.csv"), index=False
        )
    return metrics, predictions


def summarise_kfold_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise the per-fold metrics returned by :func:`perform_kfold_fit`:
    the mean, median, standard deviation and the 5th and 95th percentiles of
    each metric over all the folds and repeats (``nan`` values ignored).

    :param metrics: The ``metrics`` DataFrame returned by
                    :func:`perform_kfold_fit` (a row per fold).
    :return: A ``pandas.DataFrame`` indexed by metric with the ``mean``,
             ``median``, ``std`` (sample standard deviation), ``p05`` and
             ``p95`` columns.

    .. code:: python

        import rsgispy.classification.classsklearn as csk

        metrics, preds = csk.perform_kfold_fit(clf_obj, x, y)
        summary = csk.summarise_kfold_metrics(metrics)
        print(summary.loc["accuracy"])
    """
    if len(metrics) == 0:
        raise rsgispy.RSGISPyException("metrics is empty.")
    summ = summarise_fold_metrics(metrics)
    summ.index.name = "metric"
    return summ


def perform_jackknife_fit(
    clf_obj: Any,
    x: Any,
    y: Any,
    *,
    groups: Optional[Any] = None,
    calc_coef_stats: bool = True,
    out_dir: Optional[Union[str, os.PathLike]] = None,
    out_prefix: str = "jackknife",
    use_pbar: bool = True,
) -> Dict[str, Any]:
    """
    Assess a classifier with a jackknife (leave-one-out) cross-validation,
    the classification equivalent of
    :func:`rsgispy.regression.regsklearn.perform_jackknife_fit`: the
    classifier is fitted once for each sample with that sample left out, and
    predicts the left-out sample. With ``groups`` one group is left out at a
    time (scikit-learn's ``LeaveOneGroupOut``). The pooled leave-one-out
    predictions give one set of accuracy metrics, the influence of each
    sample and, for a linear classifier (e.g. ``LogisticRegression``),
    jackknife estimates of the bias and standard error of each coefficient.

    This leaves out samples; it is not the same as
    :func:`~rsgispy.classification.classfeatures.calc_jackknife_feature_importance`,
    which leaves out features. The classifier is fitted once per sample (or
    group), plus once to all the samples, so it suits small samples; use
    :func:`perform_kfold_fit` for large samples.

    **Influence.** For each left-out sample (or group) :math:`i`:

    * ``delta_cv_accuracy``: the accuracy of the pooled leave-one-out
      predictions without sample :math:`i`, minus the accuracy with it. A
      positive value is a sample that is misclassified (possibly a labelling
      error).
    * ``delta_fit_accuracy``: the accuracy, on the other samples, of the
      model fitted without sample :math:`i`, minus the accuracy on the same
      samples of the model fitted to all the samples.
    * ``pred_change_frac``: the fraction of all the samples whose predicted
      class differs between the model fitted to all the samples and the
      model fitted without sample :math:`i`. A large value is an
      influential sample.

    **Coefficients.** With ``calc_coef_stats`` and an estimator with
    ``coef_`` / ``intercept_`` (or a pipeline whose last step has them), the
    jackknife bias, standard error, bias-corrected estimate and 95% interval
    of each coefficient are calculated as in
    :func:`rsgispy.regression.regsklearn.perform_jackknife_fit`, with one
    table per class (a binary model has a single table, for the second,
    positive, class). They are ``None`` if a leave-one-out fit sees a
    different set of classes (for example when a group holds all the samples
    of a class).

    :param clf_obj: An unfitted classifier (it is not changed).
    :param x: The predictor variables: a NumPy array or a
              ``pandas.DataFrame`` (see :func:`perform_kfold_fit`).
    :param y: The class label of each sample (1D array or Series).
    :param groups: Optional group label of each sample; one group is left
                   out at a time.
    :param calc_coef_stats: Calculate the coefficient statistics when the
                            estimator is linear (Default = ``True``).
    :param out_dir: Optional output directory. When given, the metrics (one
                    row), predictions and influence are written to
                    ``<out_prefix>_metrics.csv``,
                    ``<out_prefix>_predictions.csv`` and
                    ``<out_prefix>_influence.csv``, and the coefficient
                    statistics to ``<out_prefix>_coef_stats_<class>.csv``
                    (the class name in lower case with spaces, dashes and
                    punctuation removed).
    :param out_prefix: Prefix of the output file names
                       (Default = ``'jackknife'``).
    :param use_pbar: Show a progress bar over the fits (Default = ``True``).
    :return: A dictionary with:

             * ``metrics``: a dictionary of ``n_samples``, ``accuracy``,
               ``bal_accuracy``, ``f1_macro`` and ``cohen_kappa`` of the
               pooled leave-one-out predictions;
             * ``predictions``: a DataFrame with the ``sample_idx``,
               ``group`` (with ``groups``), ``y_true`` and ``y_pred``
               columns;
             * ``influence``: a DataFrame with a row per left-out sample
               (``sample_idx``, ``y_true``, ``y_pred``) or group
               (``group``, ``n_samples``) and the ``delta_cv_accuracy``,
               ``delta_fit_accuracy`` and ``pred_change_frac`` columns;
             * ``coef_stats``: a dictionary keyed by class label of
               DataFrames indexed by coefficient (``intercept`` then the
               predictors) with the ``estimate``, ``jack_mean``, ``bias``,
               ``jack_estimate``, ``std_err``, ``ci_lower`` and
               ``ci_upper`` columns, or ``None``.

    .. code:: python

        import pandas
        from sklearn.linear_model import LogisticRegression
        import rsgispy.classification.classsklearn as csk

        df = pandas.read_csv("plots.csv")
        jk = csk.perform_jackknife_fit(
            LogisticRegression(max_iter=1000), df[["ndvi", "ndwi"]],
            df["class_name"], out_dir="jk_out",
        )
        print(jk["metrics"]["accuracy"])
        print(jk["coef_stats"]["Water"])
        # The samples whose removal improves the accuracy the most.
        print(jk["influence"].nlargest(5, "delta_cv_accuracy"))
    """
    # pylint: disable=too-many-arguments,too-many-locals,too-many-branches
    # pylint: disable=too-many-statements
    check_sklearn_available("perform_jackknife_fit")
    x, y_arr, grp_arr = _prep_cls_assess_xy(x, y, groups)
    n_smpls = y_arr.shape[0]
    jk_splits, n_units = jackknife_splits(n_smpls, grp_arr)

    all_idx = np.arange(n_smpls)
    full_mdl, full_pred = _fit_predict_cls_rows(clf_obj, x, y_arr, all_idx, x)
    full_coefs = _cls_linear_coefs(full_mdl) if calc_coef_stats else None

    loo_pred = np.empty(n_smpls, dtype=full_pred.dtype)
    unit_test_idx: List[np.ndarray] = []
    fit_acc_delta = np.full(n_units, np.nan, dtype=np.float64)
    pred_change = np.full(n_units, np.nan, dtype=np.float64)
    loo_coefs = []
    tqdm_cls = rsgispy.get_tqdm()
    for i_unit, (train_idx, test_idx) in enumerate(
        tqdm_cls(jk_splits, total=n_units, disable=not use_pbar)
    ):
        mdl, pred_all = _fit_predict_cls_rows(clf_obj, x, y_arr, train_idx, x)
        if pred_all.dtype != loo_pred.dtype:
            loo_pred = loo_pred.astype(object)
        loo_pred[test_idx] = pred_all[test_idx]
        unit_test_idx.append(test_idx)
        y_other = y_arr[train_idx]
        fit_acc_delta[i_unit] = _accuracy(y_other, pred_all[train_idx]) - _accuracy(
            y_other, full_pred[train_idx]
        )
        pred_change[i_unit] = float(np.mean(pred_all != full_pred))
        if full_coefs is not None:
            unit_coefs = _cls_linear_coefs(mdl)
            if (
                unit_coefs is None
                or unit_coefs[1] != full_coefs[1]
                or unit_coefs[0].shape != full_coefs[0].shape
            ):
                full_coefs = None
            else:
                loo_coefs.append(unit_coefs[0])

    metrics = _label_metrics_row(y_arr, loo_pred)
    pred_cols: Dict[str, Any] = {"sample_idx": all_idx}
    if grp_arr is not None:
        pred_cols["group"] = grp_arr
    pred_cols["y_true"] = y_arr
    pred_cols["y_pred"] = loo_pred
    predictions = pd.DataFrame(pred_cols)

    cv_acc = _accuracy(y_arr, loo_pred)
    delta_cv = []
    for test_idx in unit_test_idx:
        keep = np.ones(n_smpls, dtype=bool)
        keep[test_idx] = False
        delta_cv.append(_accuracy(y_arr[keep], loo_pred[keep]) - cv_acc)
    first_idx = np.array([t[0] for t in unit_test_idx], dtype=np.int64)
    if grp_arr is None:
        infl_cols: Dict[str, Any] = {
            "sample_idx": first_idx,
            "y_true": y_arr[first_idx],
            "y_pred": loo_pred[first_idx],
        }
    else:
        infl_cols = {
            "group": grp_arr[first_idx],
            "n_samples": [int(t.shape[0]) for t in unit_test_idx],
        }
    infl_cols["delta_cv_accuracy"] = np.asarray(delta_cv, dtype=np.float64)
    infl_cols["delta_fit_accuracy"] = fit_acc_delta
    infl_cols["pred_change_frac"] = pred_change
    influence = pd.DataFrame(infl_cols)

    coef_stats = None
    if full_coefs is not None and len(loo_coefs) == n_units:
        coef_mat, row_names = full_coefs
        coef_stats = jackknife_coef_stats(
            coef_mat,
            np.stack(loo_coefs),
            coef_names(full_mdl, x, coef_mat.shape[1] - 1),
            row_names,
        )

    if out_dir is not None:
        out_dir = os.fspath(out_dir)
        os.makedirs(out_dir, exist_ok=True)
        for kind, tbl in (
            ("metrics", pd.DataFrame([metrics])),
            ("predictions", predictions),
            ("influence", influence),
        ):
            tbl.to_csv(os.path.join(out_dir, f"{out_prefix}_{kind}.csv"), index=False)
        if coef_stats is not None:
            write_named_csvs(
                out_dir,
                out_prefix,
                {"coef_stats": coef_stats},
                list(coef_stats),
                index_tables=("coef_stats",),
                default_stem="class",
            )

    return {
        "metrics": metrics,
        "predictions": predictions,
        "influence": influence,
        "coef_stats": coef_stats,
    }
