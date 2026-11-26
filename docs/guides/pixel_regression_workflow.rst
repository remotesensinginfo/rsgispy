Pixel-based regression
======================

This guide shows how to map a continuous variable, here **canopy cover**
(%), from field plots and satellite imagery, pixel by pixel. It is the
regression counterpart of the pixel-based classification guide
(:doc:`pixel_classification_workflow`) and uses the same inputs (a 10 m
Sentinel-2 surface reflectance image and a DEM), so read that guide first for
the details of building features and configurations. The same approach
works for biomass, tree height, soil moisture or any other variable measured
at points or plots.

Regression differs from classification in a few important ways, which this
guide concentrates on:

* the target is a number, so feature exploration uses scatter plots and
  correlations with the target rather than class box plots;
* accuracy is described by errors in the target's units (RMSE, MAE, bias) and
  by R², and the residuals (predicted minus observed) should be checked for
  patterns;
* every prediction is uncertain, and a map of **prediction intervals** is
  often as important as the map of predictions.

.. contents:: Contents
   :local:
   :depth: 1

Installation
------------

The workflow needs the ``regression`` extra (``scikit-learn``, ``pyarrow``,
``optuna``, ``scipy``); the XGBoost model needs the ``xgboost`` (or ``gbdt``)
extra:

.. code-block:: bash

   pip install "rsgispy[regression,gbdt]"

SHAP and Sobol sensitivity analysis for regression models
(:func:`~rsgispy.regression.regfeatures.calc_shap_feature_contributions`,
:func:`~rsgispy.regression.regfeatures.calc_salib_feature_sensitivity`) also
need the ``modelling`` extra. The example script runs without XGBoost when
``use_gbdt=False``.

Overview
--------

.. code-block:: text

   Sentinel-2 SREF + DEM ─► 1. features (bands, indices, slope; one VRT stack)
            │
            ▼
   2. RegressionConfig (JSON) ◄── field plots with canopy_cover (%)
            │
            ▼
   3. Samples ─► train / test / calibration tables (+ x, y, plot ID)
            │
            ▼
   4. Explore (target scatter plots, correlation, VIF)
            │
            ▼
   5. Train and tune (Random Forest, XGBoost) with spatial CV folds
            │
            ├─► 6. Nested spatial CV (unbiased accuracy)
            ├─► 7. Interpret (importance, partial dependence)
            ▼
   8. Apply ─► canopy cover raster
            │
            ▼
   9. Prediction intervals (tree quantiles, conformal)
            │
            ▼
   10. Validate with independent plots (metrics, plots, residuals)
            │
            ▼
   11. Maps

The code comes from the example script
``docs/examples/pixel_regression_workflow.py``, which the test suite runs on
a small synthetic scene. The inputs are the Sentinel-2 stack and DEM
described in :doc:`pixel_classification_workflow`, a vector file of training
plots with the measured canopy cover in a ``canopy_cover`` column, and a
second, independent set of plots for validation:

.. code-block:: bash

   python docs/examples/pixel_regression_workflow.py --s2 s2_sref_10m.tif \
       --dem dem_30m.tif --plots canopy_plots.gpkg \
       --valid-plots canopy_plots_valid.gpkg --out-dir outputs

Step 1 — Features
-----------------

The features are built exactly as in the classification guide (step 1 there
explains the choices): spectral indices with
:func:`~rsgispy.imagecalc.calcindices.calc_indices`, slope with
:func:`~rsgispy.elevation.terrain.slope` resampled onto the image grid with
:func:`~rsgispy.imageutils.warp.resample_img_to_match`, and one virtual stack
of the Sentinel-2 bands (``uint16``) and the derived layers (``float32``)
with :func:`~rsgispy.imageutils.vrt.create_stack_images_vrt`, each band
keeping its own data type and no data value.
For canopy cover, vegetation indices that do not saturate quickly at high
cover (EVI) and moisture indices (NDMI, sensitive to canopy water) are useful
additions to NDVI.

.. literalinclude:: ../examples/pixel_regression_workflow.py
   :language: python
   :start-after: # [step-1-features]
   :end-before: # [end-step-1-features]
   :dedent: 4

Step 2 — The configuration
--------------------------

:class:`~rsgispy.regression.regdata.RegressionConfig` holds the image bands
(:class:`~rsgispy.classification.classdata.ImageBandInfo`, as for
classification), the name of the target column (or a list of columns, to
predict several variables with one model) and the plot vector file.
:func:`~rsgispy.regression.regdata.write_reg_config` saves it as JSON and
returns it; :func:`~rsgispy.regression.regdata.read_reg_config` reads it
back.

.. literalinclude:: ../examples/pixel_regression_workflow.py
   :language: python
   :start-after: # [step-2-config]
   :end-before: # [end-step-2-config]
   :dedent: 4

Step 3 — Samples
----------------

:func:`~rsgispy.regression.regdata.extract_and_split_reg_samples` reads the
feature values under each plot (a point gives the pixel it falls in; a
polygon gives every pixel inside it) and splits the samples into three
tables:

* **train** — fits the model;
* **test** — reports the model's performance after training;
* **valid** — here used as a **calibration** set for the conformal prediction
  intervals in step 9, so it must not be used for training.

Options specific to regression:

* ``target_min`` / ``target_max`` drop samples with impossible target values
  (e.g. data entry errors); ``drop_nonfinite_targets`` (default ``True``)
  drops missing values;
* ``stratify_col`` with ``n_bins`` splits within quantile bins of the target,
  so that every table covers the full range of values. Without it, a small
  test set can miss the rare high or low values;
* ``include_coords=True`` keeps the ``x`` and ``y`` columns for spatial
  cross-validation;
* ``group_col='train_grp'`` records the source plot of each sample in that
  column (``'pt<n>'`` for the ``n``-th point, ``'poly<n>'`` for a polygon)
  and puts all samples of a plot in the same table.

If a plot is larger than a pixel, all of its pixels get the plot's value;
they are not independent samples. With ``group_col`` they stay together in
one table (and ``groups='train_grp'`` with
:class:`~rsgispy.ml.cv.PolygonGroupKFold` keeps them in one fold when
tuning), so the test error is not flattered by pixels of the training plots;
still, prefer plot sizes similar to the pixel size, or average the pixels of
each plot before training. Here the plots are points, so each plot is its
own group.

.. literalinclude:: ../examples/pixel_regression_workflow.py
   :language: python
   :start-after: # [step-3-samples]
   :end-before: # [end-step-3-samples]
   :dedent: 4

**Check:** the summary of the target (count, mean, quartiles, range) is
plausible and similar in the three tables.

Step 4 — Exploring the features
-------------------------------

:mod:`rsgispy.regression.regfeatures` mirrors the classification feature
functions, with versions that use the target:

* :func:`~rsgispy.regression.regfeatures.target_scatter_plots` plots the
  target against each feature (one PNG per feature and target) with a trend
  line. Look for features with a clear relationship, and for its shape:
  saturation at high values (common for NDVI with dense canopies) or
  thresholds suggest a non-linear model;
* :func:`~rsgispy.regression.regfeatures.correlation_matrix` with
  ``include_targets=True`` adds the target to the matrix, so the column of the
  target lists each feature's correlation with it;
* :func:`~rsgispy.regression.regfeatures.select_features_vif` (and the
  other selection functions:
  :func:`~rsgispy.regression.regfeatures.select_features_corr`,
  :func:`~rsgispy.regression.regfeatures.select_features_rfe`,
  :func:`~rsgispy.regression.regfeatures.select_features_from_model`,
  :func:`~rsgispy.regression.regfeatures.select_features_boruta_shap`) work as
  described in the classification guide. A selection is applied in the same
  way, by writing a configuration with only the selected bands.

.. literalinclude:: ../examples/pixel_regression_workflow.py
   :language: python
   :start-after: # [step-4-explore]
   :end-before: # [end-step-4-explore]
   :dedent: 4

The correlations with canopy cover look like:

.. code-block:: text

   Blue    -0.71
   Red     -0.36
   ...
   NIR      0.84
   NDVI     0.84
   EVI      0.94

Step 5 — Training and optimisation
----------------------------------

:func:`~rsgispy.regression.regsklearn.perform_sklearn_regressor_param_search`
searches the hyperparameters of any scikit-learn regressor (here a Random
Forest) using spatially blocked folds: a
:class:`~rsgispy.ml.cv.SpatialBlockKFold` is passed as ``cv`` with the
coordinate columns of the training table as ``coords`` (see the
classification guide for why random folds are optimistic and how to choose
``block_size``; :class:`~rsgispy.ml.cv.PolygonGroupKFold` with
``groups='train_grp'`` keeps the pixels of each plot in one fold instead).
``scoring='neg_root_mean_squared_error'`` chooses the parameters with the
lowest RMSE; ``'r2'`` is the default.
:func:`~rsgispy.regression.regsklearn.train_sklearn_regressor` then fits the
chosen model, saves it and returns metrics for each table: ``r2``, ``rmse``,
``mae``, ``bias`` (mean of predicted minus observed), ``explained_variance``
and ``pearson_r``, prefixed with ``train_``, ``test_`` (and ``valid_``).

:func:`~rsgispy.regression.regxgboost.train_xgboost_regressor` trains an
XGBoost model; with a validation table and ``early_stopping_rounds`` it stops
adding trees when the validation error stops improving. The example uses
the calibration table for this, which is fine because the intervals in
step 9 are for the Random Forest; conformal intervals for the XGBoost model
would need a calibration set it has not seen.
:func:`~rsgispy.regression.regxgboost.optimise_xgboost_regressor` tunes its
hyperparameters with Optuna, as for classification.

.. literalinclude:: ../examples/pixel_regression_workflow.py
   :language: python
   :start-after: # [step-5-train]
   :end-before: # [end-step-5-train]
   :dedent: 4

A Random Forest averages training values, so it cannot predict outside the
range of the training targets, which is often desirable (cover stays within 0–100 %) but means that extreme values
are under-predicted. A test RMSE much larger than the training RMSE means
over-fitting: increase ``min_samples_leaf`` or reduce ``max_features``.

Step 6 — Spatial cross-validation
---------------------------------

As for classification, :func:`~rsgispy.ml.cv.nested_spatial_cv` with
``task='regression'`` gives an accuracy estimate for new areas: the model is
trained on some spatial blocks and evaluated on the others, for every fold
(with inner tuning when a ``param_space`` is given). ``overall_metrics``
contains ``r2``, ``rmse`` and ``mae`` of the out-of-fold predictions, and
``oof_predictions`` the predictions themselves, which can be mapped at the
sample locations to see where the model performs badly.

.. literalinclude:: ../examples/pixel_regression_workflow.py
   :language: python
   :start-after: # [step-6-spatial-cv]
   :end-before: # [end-step-6-spatial-cv]
   :dedent: 4

Step 7 — Interpreting the model
-------------------------------

* :func:`~rsgispy.regression.regfeatures.calc_feature_importance` gives model
  (impurity) and permutation importance, with R² as the permutation score.
* :func:`~rsgispy.regression.regfeatures.calc_feature_response_curves` shows
  how the predicted target changes with each feature. With
  ``method='marginal'`` (default) the other features are held at their mean;
  with ``method='pdp'`` (partial dependence) the curve is averaged over a
  sample of real training rows (``pdp_background_samples``) instead of a
  single artificial "average" pixel. Both methods combine values of one
  feature with values of the others that may never occur together when the
  features are correlated (e.g. NDVI and EVI), so trust a curve only where
  there are training samples. ``out_grid_plot`` draws all the
  curves in one figure, and ``plot_samples=True`` adds the training samples
  so you can see where the curve is supported by data. Unlike the
  classification curves (a probability between 0 and 1), these curves are in
  the target's units: a curve that rises with EVI and flattens at high values
  is what we expect for canopy cover.
* :func:`~rsgispy.regression.regfeatures.calc_jackknife_feature_importance`,
  :func:`~rsgispy.regression.regfeatures.calc_shap_feature_contributions` and
  :func:`~rsgispy.regression.regfeatures.calc_salib_feature_sensitivity` work
  as described in the classification guide, with R² and RMSE in place of
  accuracy and the predicted value in place of a class probability.

.. literalinclude:: ../examples/pixel_regression_workflow.py
   :language: python
   :start-after: # [step-7-interpret]
   :end-before: # [end-step-7-interpret]
   :dedent: 4

Step 8 — Applying the model
---------------------------

:func:`~rsgispy.regression.regsklearn.apply_sklearn_regressor` predicts every
pixel where the mask equals ``img_msk_val``, block by block, and writes a
``float32`` image with one band per target (named after it) and no data value
``-9999``. As for classification, the valid-data mask comes from the
configuration's feature stack.
:func:`~rsgispy.regression.regxgboost.apply_xgboost_regressor` applies an
XGBoost model.

.. literalinclude:: ../examples/pixel_regression_workflow.py
   :language: python
   :start-after: # [step-8-apply]
   :end-before: # [end-step-8-apply]
   :dedent: 4

Step 9 — Prediction intervals
-----------------------------

:func:`~rsgispy.ml.predict.predict_quantiles_raster` writes per-pixel
intervals to a multi-band image, in one of three ways:

* **Tree spread** — for a tree ensemble (Random Forest, Extra Trees), the
  ``quantiles`` (here 5 %, 50 % and 95 %) of the predictions of the
  individual trees (bands ``q05``, ``q50``, ``q95``). It is quick and shows
  where the model is uncertain (e.g. conditions not covered by the
  training plots), but it is not calibrated: the 5–95 % range does not
  necessarily contain 90 % of the true values.
* **Split-conformal intervals** — a fitted model plus the absolute residuals
  ``|observed - predicted|`` on a **calibration** set that was not used for
  training (here the third table from step 3). The interval is
  ``prediction ± q``, where ``q`` is the ``1 - alpha`` quantile of the
  calibration residuals (with a small-sample correction), so with
  ``alpha=0.1`` at least 90 % of new observations from the same population
  are expected to fall inside it (bands ``lower_90pct``, ``prediction``,
  ``upper_90pct``). The width is the same everywhere.
* **Quantile models** — a dictionary ``{quantile: fitted_model}`` of models
  trained for each quantile (e.g. GBDT models with a quantile loss).

Use the tree spread to see *where* the predictions are uncertain and the
conformal interval to state *how* uncertain they are.

.. literalinclude:: ../examples/pixel_regression_workflow.py
   :language: python
   :start-after: # [step-9-intervals]
   :end-before: # [end-step-9-intervals]
   :dedent: 4

Step 10 — Validation with independent plots
-------------------------------------------

The map should be validated with plots that were not used in any way for
training or tuning, ideally a probability sample of the area.

* :func:`~rsgispy.regression.regmetrics.pop_reg_img_pred_pts` samples the
  predicted image at each plot (``pred_col`` for one band, ``pred_cols`` for
  several, e.g. the interval bands). Plots in no data areas get ``NaN`` and
  are removed.
* :func:`~rsgispy.regression.regmetrics.calc_reg_metrics` returns ``r2``,
  ``rmse``, ``mae``, ``bias``, ``explained_variance``, ``pearson_r`` and
  ``n_samples`` (and can write them to JSON or CSV);
  :func:`~rsgispy.regression.regmetrics.calc_reg_stats_ptonly_bootstrap_conf`
  adds bootstrap confidence intervals.
* :func:`~rsgispy.regression.regmetrics.plot_pred_vs_obs` plots predicted
  against observed values with the 1:1 line. Points below the line at high
  values and above it at low values (a slope below 1) are typical of tree
  models, which shrink predictions towards the mean.
* :func:`~rsgispy.regression.regmetrics.plot_residuals` plots the residuals
  against the predictions. They should be centred on zero with no trend and
  a similar spread everywhere; a funnel shape means the error grows with
  the value.
* The **coverage** of the prediction interval (the fraction of plots inside
  it) should be close to the nominal level (0.9 here).
* Writing the residuals to a vector file gives a **residual map**: clusters
  of positive or negative residuals show areas where the model is biased
  (e.g. a forest type missing from the training plots), and suggest where
  to add plots or features.

.. literalinclude:: ../examples/pixel_regression_workflow.py
   :language: python
   :start-after: # [step-10-validate]
   :end-before: # [end-step-10-validate]
   :dedent: 4

Step 11 — Maps
--------------

:func:`~rsgispy.mapping.raster.create_raster_cmap_img_map` draws one band of
a continuous raster with a colour map; ``norm_vmin`` / ``norm_vmax`` fix the
colour range so that maps of different areas or dates are comparable. Here
the prediction is shown next to the width of the tree-spread interval,
calculated with :func:`~rsgispy.imagecalc.maths.band_maths` (no data is set
to 0 so it is not counted as a very large width).

.. literalinclude:: ../examples/pixel_regression_workflow.py
   :language: python
   :start-after: # [step-11-map]
   :end-before: # [end-step-11-map]
   :dedent: 4

Performance and large datasets
------------------------------

* **Plots are the limiting factor** — field data are expensive, so regression
  training sets are usually small (tens to hundreds of plots). Prefer simple
  models with few features, use cross-validation rather than a large
  validation set, and keep the calibration set big enough (at least ~50
  plots) for stable conformal intervals.
* **Applying the model** — the apply and
  :func:`~rsgispy.ml.predict.predict_quantiles_raster` functions process the
  image in blocks (``block_size``). Tree quantiles evaluate every tree for
  every pixel and keep all their predictions, which is slower than the mean
  prediction; use fewer trees or the conformal interval for very large areas.
* **Parallel processing and VRTs** — as in the classification guide:
  ``n_jobs=-1`` for the models, VRTs to avoid copies, and tiling for very
  large areas.

Common problems
---------------

* **Plots and pixels do not match** — GPS error, plot size and the date of
  the field work relative to the image all add noise. Use plots at least as
  large as a pixel, check the plot locations against the image, and use
  imagery from close to the field date.
* **Low R² but sensible maps** — R² depends on the range of the target in
  the sample; report RMSE and bias as well, in the target's units.
* **Predictions outside the valid range** — linear models and GBDT models can
  predict values below 0 or above 100 %. Clip them with
  :func:`~rsgispy.imagecalc.maths.band_maths`, or use a tree ensemble.
* **No data in the predictions** — pixels where any feature is no data are
  not predicted (``-9999``). Check the no data values of the input images
  (e.g. the indices are ``-9999``, the Sentinel-2 stack ``0``).
* **CRS mismatches** — the plots must be in the image's CRS, or use
  ``reproj_vec=True`` when extracting the samples and sampling the
  predictions.
* **Interval coverage far from nominal** — the calibration plots are not
  representative of the area (e.g. all from one region), or were used for
  training. Calibrate with independent plots spread over the area.
* **Optional dependencies** — XGBoost needs the ``xgboost``/``gbdt`` extra and
  SHAP/SALib the ``modelling`` extra.

Related guides
--------------

* :doc:`pixel_classification_workflow`
* :doc:`segment_ml_workflow`
* :doc:`species_distribution_modelling`
