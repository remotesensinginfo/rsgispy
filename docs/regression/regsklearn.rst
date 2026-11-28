rsgispy.regression.regsklearn
=============================

Landsat Forest Canopy Cover Example
-----------------------------------

The following example predicts continuous forest canopy cover percentages
(``canopy_cover``, range ``0`` to ``100``) across a 6-band Landsat surface
reflectance scene using reference estimates stored in a vector layer
(``canopy_cover_samples.gpkg``). Samples are extracted into Parquet tables, a
``RandomForestRegressor`` is tuned with Bayesian optimisation (Optuna TPE) using
the built-in standard parameter space, and the trained model is applied
block-by-block to produce a continuous canopy cover map.

.. code-block:: python

    from sklearn.ensemble import RandomForestRegressor
    import rsgispy.regression.regdata as rd
    import rsgispy.regression.regsklearn as rsk

    # 1. Define Landsat bands and target vector layer with canopy cover estimates
    img_band_info = [
        rd.ImageBandInfo(
            file_name="landsat_sr.tif",
            name="lsat",
            bands=[1, 2, 3, 4, 5, 6],
            band_names=["blue", "green", "red", "nir", "swir1", "swir2"],
        )
    ]
    reg_cfg = rd.RegressionConfig(
        img_band_info=img_band_info,
        target_cols="canopy_cover",
        vec_file="canopy_cover_samples.gpkg",
        vec_lyr="canopy_cover_samples",
    )

    # 2. Extract and split samples into train / test / validation sets
    train_df, test_df, valid_df = rd.extract_and_split_reg_samples(
        reg_cfg,
        test_prop=0.2,
        valid_prop=0.2,
        target_min=0.0,
        target_max=100.0,
        out_train_file="train_samples.parquet",
        out_test_file="test_samples.parquet",
        out_valid_file="valid_samples.parquet",
        rnd_seed=42,
    )

    # 3. Optimise Random Forest parameters using Bayesian updating (Optuna)
    # (omitting param_space automatically uses the built-in RandomForest space)
    best_rf = rsk.perform_sklearn_regressor_param_search(
        reg_cfg=reg_cfg,
        sk_regressor=RandomForestRegressor(n_jobs=-1, random_state=42),
        train_data=train_df,
        valid_data=valid_df,
        search_method="optuna",
        scoring="r2",
        n_trials=30,
        refit=False,
        out_params_json="rf_best_params.json",
        rnd_seed=42,
    )

    # 4. Train and evaluate the optimised regressor
    fitted_rf, metrics = rsk.train_sklearn_regressor(
        reg_cfg=reg_cfg,
        sk_regressor=best_rf,
        train_data=train_df,
        test_data=test_df,
        valid_data=valid_df,
        out_model_file="landsat_canopy_rf_model.joblib",
        out_metrics_file="landsat_canopy_rf_metrics.json",
        return_model=True,
        rnd_seed=42,
    )

    # 5. Apply the model to predict continuous forest canopy cover (%) across Landsat
    rsk.apply_sklearn_regressor(
        reg_cfg=reg_cfg,
        sk_regressor=fitted_rf,
        in_msk_img="landsat_valid_mask.tif",
        img_msk_val=1,
        output_img="landsat_canopy_cover_rf.tif",
    )

Multivariate Regression Example
-------------------------------

Passing a list of target attributes (for example, predicting both
``canopy_cover`` and ``canopy_height`` simultaneously) trains a multivariate
model and writes a multi-band continuous output raster with matching band
descriptions:

.. code-block:: python

    from sklearn.ensemble import ExtraTreesRegressor
    import rsgispy.regression.regdata as rd
    import rsgispy.regression.regsklearn as rsk

    mv_cfg = rd.RegressionConfig(
        img_band_info=img_band_info,
        target_cols=["canopy_cover", "canopy_height"],
        vec_file="forest_structure_samples.gpkg",
    )
    train_df, test_df, _ = rd.extract_and_split_reg_samples(
        mv_cfg, test_prop=0.25, rnd_seed=42
    )
    mv_model, mv_metrics = rsk.train_sklearn_regressor(
        reg_cfg=mv_cfg,
        sk_regressor=ExtraTreesRegressor(n_estimators=200, random_state=42),
        train_data=train_df,
        test_data=test_df,
        return_model=True,
    )
    # Writes a 2-band GeoTIFF: Band 1 = 'canopy_cover', Band 2 = 'canopy_height'
    rsk.apply_sklearn_regressor(
        reg_cfg=mv_cfg,
        sk_regressor=mv_model,
        in_msk_img="landsat_valid_mask.tif",
        img_msk_val=1,
        output_img="landsat_forest_structure.tif",
    )

Per-pixel Uncertainty
---------------------

For an ensemble of independently fitted members (Random Forest, Extra Trees,
Bagging or Voting regressors, also inside a ``Pipeline`` or
``MultiOutputRegressor``), ``out_uncert_img`` writes a ``float32`` image with
one band per target (``<target>_<uncert_method>``, no data ``-9999``) giving
how much the members disagree at each pixel: ``uncert_method='std'`` (the
standard deviation of the member predictions, in the target's units;
default), ``'iqr'`` (inter-quartile range) or ``'cv'`` (``std / |mean|``). See
:func:`~rsgispy.ml.predict.calc_ensemble_uncertainty`. Larger values are less
certain; the values rank pixels but are not a calibrated interval (use
:func:`~rsgispy.ml.predict.predict_quantiles_raster` with conformal residuals
for that). Models without independent members (linear models, SVR, boosted
trees) raise an exception before any output is written.
:func:`~rsgispy.regression.regsegments.apply_sklearn_regressor_segments`
takes ``out_uncert_method`` to add the same values as
``pred_<target>_<method>`` columns.

.. code-block:: python

    rsk.apply_sklearn_regressor(
        reg_cfg=mv_cfg,
        sk_regressor=mv_model,
        in_msk_img="landsat_valid_mask.tif",
        img_msk_val=1,
        output_img="landsat_forest_structure.tif",
        out_uncert_img="landsat_forest_structure_std.tif",  # 2 bands
    )

Repeated k-fold and Jackknife Assessment
----------------------------------------

With a small sample table (for example a forest inventory of a hundred or
so plots), a single train / test split gives noisy accuracy figures.
:func:`~rsgispy.regression.regsklearn.perform_kfold_fit` fits the regressor
to every fold of a repeated k-fold and returns the
:func:`~rsgispy.regression.regmetrics.calc_reg_metrics` metrics of each fold
(including ``mse``, ``nrmse``, ``nrmse_range`` and ``nbias``) and the
out-of-fold predictions; :func:`~rsgispy.regression.regsklearn.summarise_kfold_metrics`
summarises them. :func:`~rsgispy.regression.regsklearn.perform_jackknife_fit`
leaves out one sample (or group) at a time, giving the pooled leave-one-out
metrics, the influence of each sample and, for linear models, jackknife
standard errors of the coefficients. Both take any unfitted estimator with
``fit`` / ``predict``, including pipelines and the XGBoost, LightGBM and
CatBoost scikit-learn wrappers.

.. code-block:: python

    import pandas
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.linear_model import LinearRegression
    import rsgispy.regression.regsklearn as rsk

    df = pandas.read_csv("forest_plots.csv", index_col=0)
    x = df[["p95", "cover", "skew"]]
    y = df[["Vol / ha", "BA / ha"]]

    # 5 folds repeated 20 times (100 fits); writes kfold_et_metrics_vol_ha.csv etc.
    metrics, residuals = rsk.perform_kfold_fit(
        ExtraTreesRegressor(random_state=0),
        x,
        y,
        n_splits=5,
        n_repeats=20,
        out_dir="kfold_outputs",
        out_prefix="kfold_et",
    )
    summary = rsk.summarise_kfold_metrics(metrics)
    print(summary.loc["Vol / ha"].loc[["r2", "rmse", "nrmse", "nbias"]])

    # Pixels of one plot (or plots of one site) are never split across folds.
    metrics, residuals = rsk.perform_kfold_fit(
        ExtraTreesRegressor(random_state=0), x, y, groups=df["site"]
    )

    # Leave-one-out: one fit per plot.
    jk = rsk.perform_jackknife_fit(LinearRegression(), x, y["Vol / ha"])
    print(jk["metrics"]["Vol / ha"]["rmse"])
    print(jk["coef_stats"]["Vol / ha"])  # bias, std_err, 95% interval
    print(jk["influence"]["Vol / ha"].nlargest(5, "pred_shift_rmse"))

Linear Model Equations
----------------------

For a fitted linear model (``LinearRegression``, ``Ridge``, ``Lasso``,
``LassoLars`` …, or a pipeline that is linear overall, such as a
``StandardScaler`` followed by a linear model),
:func:`~rsgispy.regression.regsklearn.get_linear_model_coefs` returns the
intercept and coefficients in the units of the input features,
:func:`~rsgispy.regression.regsklearn.get_linear_model_equation` writes the
model as an equation for a report, and
:func:`~rsgispy.regression.regsklearn.get_linear_model_band_maths_exp` writes
it as a :func:`~rsgispy.imagecalc.maths.band_maths` expression, to apply it
to an image of the features (an alternative to
:func:`~rsgispy.regression.regsklearn.apply_sklearn_regressor` that needs no
saved model):

.. code-block:: python

    from sklearn.linear_model import LinearRegression
    import rsgispy
    from rsgispy.imagecalc.maths import BandDefn, band_maths
    import rsgispy.regression.regsklearn as rsk

    vol_mdl = LinearRegression().fit(df[ind_vars], df["Vol / ha"])
    print(rsk.get_linear_model_equation(vol_mdl, target_name="vol", precision=4))
    # vol = -12.3456 + 0.8123 * h_p95 + 1.2004 * cover ...

    band_defns = [BandDefn("vld_msk", "valid_mask.tif", 1)] + [
        BandDefn(var, "als_metrics.tif", band) for band, var in enumerate(ind_vars, 1)
    ]
    exp = rsk.get_linear_model_band_maths_exp(vol_mdl, valid_msk_name="vld_msk")
    band_maths("volume.tif", exp, band_defns, datatype=rsgispy.TYPE_32FLOAT,
               no_data_val=0)

Standard Parameter Search Spaces
--------------------------------

When ``param_space=None`` (or ``'default'`` / ``'extended'``) is passed to
:func:`~rsgispy.regression.regsklearn.perform_sklearn_regressor_param_search`,
built-in standard search spaces are automatically selected for the regressors
below (including ``Pipeline``, ``MultiOutputRegressor``, and
``TransformedTargetRegressor`` wrappers around them). All spaces can also be
inspected or customised via
:func:`~rsgispy.regression.regsklearn.get_sklearn_regressor_param_space`.

.. list-table::
   :header-rows: 1
   :widths: 26 14 30 30

   * - Regressor
     - Aliases
     - Default Parameters (``extended=False``)
     - Extended Parameters (``extended=True``)
   * - ``RandomForestRegressor``
     - ``'rf'``, ``'rfr'``
     - ``n_estimators`` (50..500), ``max_depth`` (4..32), ``min_samples_split`` (2..20), ``min_samples_leaf`` (1..10), ``max_features`` (``'sqrt'``, ``'log2'``, ``None``)
     - Widens tree/depth ranges (``n_estimators`` 50..1000, ``max_depth`` 3..40) and adds ``criterion``, ``bootstrap``, ``ccp_alpha``, numeric ``max_features``
   * - ``ExtraTreesRegressor``
     - ``'et'``, ``'etr'``
     - ``n_estimators`` (50..500), ``max_depth`` (4..32), ``min_samples_split`` (2..20), ``min_samples_leaf`` (1..10), ``max_features``
     - Widens tree/depth ranges and adds ``criterion``, ``bootstrap``, ``ccp_alpha``, numeric ``max_features``
   * - ``GradientBoostingRegressor``
     - ``'gb'``, ``'gbr'``
     - ``n_estimators`` (50..400), ``learning_rate`` (0.01..0.3 log), ``max_depth`` (2..8), ``min_samples_split`` (2..20), ``min_samples_leaf`` (1..15), ``subsample`` (0.5..1.0)
     - Widens boosting ranges and adds ``loss`` (``'squared_error'``, ``'huber'``, ``'absolute_error'``), ``max_features``, ``ccp_alpha``
   * - ``HistGradientBoostingRegressor``
     - ``'hgb'``, ``'hgbr'``
     - ``learning_rate`` (0.01..0.3 log), ``max_iter`` (50..500), ``max_leaf_nodes`` (15..127), ``max_depth`` (3..15), ``min_samples_leaf`` (5..50), ``l2_regularization`` (1e-4..10.0 log)
     - Widens tree/regularisation ranges and adds ``max_bins`` (64..255), ``loss`` (``'squared_error'``, ``'absolute_error'``)
   * - ``SVR``
     - ``'svm'``, ``'svr'``
     - ``C`` (0.1..100.0 log), ``kernel`` (``'rbf'``, ``'linear'``), ``gamma`` (``'scale'``, ``'auto'``), ``epsilon`` (1e-3..1.0 log)
     - Conditional search over ``C`` (0.01..1000.0 log), ``kernel`` (``'rbf'``, ``'poly'``, ``'sigmoid'``, ``'linear'``), heuristic/numeric ``gamma``, ``degree``, ``coef0``, ``epsilon``
   * - ``MLPRegressor``
     - ``'mlp'``
     - ``hidden_layer_sizes`` (32..256), ``activation`` (``'relu'``, ``'tanh'``), ``alpha`` (1e-5..1e-1 log), ``learning_rate_init`` (1e-4..1e-2 log)
     - Conditional multi-layer networks (1..3 layers of 32..256 units), ``activation`` (adds ``'logistic'``), ``alpha``, ``learning_rate_init``, ``learning_rate``, ``early_stopping``
   * - ``KNeighborsRegressor``
     - ``'knn'``, ``'knr'``
     - ``n_neighbors`` (3..25 step 2), ``weights`` (``'uniform'``, ``'distance'``), ``p`` (1..2)
     - Widens ``n_neighbors`` (1..51 step 2), ``p`` (1..3), and adds ``leaf_size`` (15..60)
   * - ``Ridge``
     - ``'ridge'``
     - ``alpha`` (1e-4..100.0 log), ``fit_intercept`` (``True``, ``False``)
     - Widens ``alpha`` (1e-5..1000.0 log) and adds ``tol`` (1e-5..1e-2 log)
   * - ``Lasso``
     - ``'lasso'``
     - ``alpha`` (1e-4..100.0 log), ``fit_intercept`` (``True``, ``False``)
     - Widens ``alpha`` (1e-5..1000.0 log) and adds ``tol`` (1e-5..1e-2 log)
   * - ``ElasticNet``
     - ``'elasticnet'``, ``'enet'``
     - ``alpha`` (1e-4..100.0 log), ``l1_ratio`` (0.05..0.95), ``fit_intercept``
     - Widens ``alpha`` (1e-5..1000.0 log), ``l1_ratio`` (0.05..0.95), and adds ``tol``
   * - ``DecisionTreeRegressor``
     - ``'dt'``, ``'dtr'``
     - ``criterion`` (``'squared_error'``, ``'friedman_mse'``), ``max_depth`` (3..30), ``min_samples_split`` (2..20), ``min_samples_leaf`` (1..15)
     - Widens ranges and adds ``'absolute_error'``, ``splitter``, ``max_features``, ``ccp_alpha``
   * - ``AdaBoostRegressor``
     - ``'ada'``, ``'adar'``
     - ``n_estimators`` (50..400), ``learning_rate`` (0.01..2.0 log)
     - Widens ``n_estimators`` (50..800), ``learning_rate`` (0.005..2.0 log), and adds ``loss`` (``'linear'``, ``'square'``, ``'exponential'``)

API Reference
-------------

.. automodule:: rsgispy.regression.regsklearn
   :members:
   :undoc-members:
   :show-inheritance:
