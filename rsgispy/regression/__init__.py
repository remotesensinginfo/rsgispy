"""
The regression subpackage provides pure-Python continuous univariate and
multivariate image regression workflows using ``numpy``, ``rasterio``,
``geopandas``, and ``scikit-learn``, alongside optional Gradient Boosted
Decision Tree backends (XGBoost, LightGBM, CatBoost), feature analysis and
response curves, and point-based regression accuracy assessment.

Submodules:

* :mod:`rsgispy.regression.regdata` — Regression image/band and target
  configuration (:class:`~rsgispy.regression.regdata.ImageBandInfo`,
  :class:`~rsgispy.regression.regdata.RegressionConfig`), JSON configuration I/O
  (:func:`~rsgispy.regression.regdata.write_reg_config`,
  :func:`~rsgispy.regression.regdata.read_reg_config`), and sample table I/O,
  extraction, and train/test/validation splitting defaulting to Snappy-compressed
  Parquet alongside other pandas-supported formats
  (:func:`~rsgispy.regression.regdata.write_reg_samples`,
  :func:`~rsgispy.regression.regdata.read_reg_samples`,
  :func:`~rsgispy.regression.regdata.extract_reg_training_samples`,
  :func:`~rsgispy.regression.regdata.split_reg_samples`,
  :func:`~rsgispy.regression.regdata.extract_and_split_reg_samples`).
* :mod:`rsgispy.regression.regsklearn` — Scikit-learn regressor hyperparameter
  optimisation (Grid Search, Random Search, and Optuna Bayesian optimisation
  with built-in standard search spaces for common regressors), univariate and
  multivariate model training and persistence, block-wise image regression,
  and repeated k-fold and jackknife (leave-one-out) assessment of any regressor
  (:func:`~rsgispy.regression.regsklearn.get_sklearn_regressor_param_space`,
  :func:`~rsgispy.regression.regsklearn.perform_sklearn_regressor_param_search`,
  :func:`~rsgispy.regression.regsklearn.train_sklearn_regressor`,
  :func:`~rsgispy.regression.regsklearn.apply_sklearn_regressor`,
  :func:`~rsgispy.regression.regsklearn.save_sklearn_regressor`,
  :func:`~rsgispy.regression.regsklearn.load_sklearn_regressor`,
  :func:`~rsgispy.regression.regsklearn.perform_kfold_fit`,
  :func:`~rsgispy.regression.regsklearn.summarise_kfold_metrics`,
  :func:`~rsgispy.regression.regsklearn.perform_jackknife_fit`).
* :mod:`rsgispy.regression.regxgboost` — XGBoost univariate and multi-target
  regression with Optuna Bayesian hyperparameter optimisation, early stopping,
  optional CUDA GPU acceleration with CPU fallback, model persistence, and
  block-wise image regression
  (:func:`~rsgispy.regression.regxgboost.is_xgboost_gpu_available`,
  :func:`~rsgispy.regression.regxgboost.get_xgboost_optuna_param_space`,
  :func:`~rsgispy.regression.regxgboost.optimise_xgboost_regressor`,
  :func:`~rsgispy.regression.regxgboost.train_xgboost_regressor`,
  :func:`~rsgispy.regression.regxgboost.apply_xgboost_regressor`,
  :func:`~rsgispy.regression.regxgboost.save_xgboost_regressor`,
  :func:`~rsgispy.regression.regxgboost.load_xgboost_regressor`).
* :mod:`rsgispy.regression.reglightgbm` — LightGBM univariate and multi-target
  regression with Optuna Bayesian hyperparameter optimisation, early stopping,
  optional CUDA/OpenCL GPU acceleration with CPU fallback, model persistence, and
  block-wise image regression
  (:func:`~rsgispy.regression.reglightgbm.is_lightgbm_gpu_available`,
  :func:`~rsgispy.regression.reglightgbm.get_lightgbm_optuna_param_space`,
  :func:`~rsgispy.regression.reglightgbm.optimise_lightgbm_regressor`,
  :func:`~rsgispy.regression.reglightgbm.train_lightgbm_regressor`,
  :func:`~rsgispy.regression.reglightgbm.apply_lightgbm_regressor`,
  :func:`~rsgispy.regression.reglightgbm.save_lightgbm_regressor`,
  :func:`~rsgispy.regression.reglightgbm.load_lightgbm_regressor`).
* :mod:`rsgispy.regression.regcatboost` — CatBoost univariate and multi-target
  (``MultiRMSE``) regression with Optuna Bayesian hyperparameter optimisation,
  early stopping, optional CUDA GPU acceleration with CPU fallback, model
  persistence, and block-wise image regression
  (:func:`~rsgispy.regression.regcatboost.is_catboost_gpu_available`,
  :func:`~rsgispy.regression.regcatboost.get_catboost_optuna_param_space`,
  :func:`~rsgispy.regression.regcatboost.optimise_catboost_regressor`,
  :func:`~rsgispy.regression.regcatboost.train_catboost_regressor`,
  :func:`~rsgispy.regression.regcatboost.apply_catboost_regressor`,
  :func:`~rsgispy.regression.regcatboost.save_catboost_regressor`,
  :func:`~rsgispy.regression.regcatboost.load_catboost_regressor`).
* :mod:`rsgispy.regression.regfeatures` — Exploratory target vs feature scatter
  plots, correlation/VIF multicollinearity diagnostics, feature selection (CQV,
  correlation clustering, iterative VIF, univariate F/MI, RFE/RFECV,
  model/LassoLars, BorutaShap), and interpretability/sensitivity tools
  (permutation/model importance, jackknife analysis, marginal/PDP feature
  response curves, SHAP contributions, and SALib Sobol sensitivity)
  (:func:`~rsgispy.regression.regfeatures.target_scatter_plots`,
  :func:`~rsgispy.regression.regfeatures.correlation_matrix`,
  :func:`~rsgispy.regression.regfeatures.calc_vif_multicollinearity`,
  :func:`~rsgispy.regression.regfeatures.select_features_cqv`,
  :func:`~rsgispy.regression.regfeatures.select_features_corr`,
  :func:`~rsgispy.regression.regfeatures.select_features_vif`,
  :func:`~rsgispy.regression.regfeatures.select_features_univariate`,
  :func:`~rsgispy.regression.regfeatures.select_features_rfe`,
  :func:`~rsgispy.regression.regfeatures.select_features_from_model`,
  :func:`~rsgispy.regression.regfeatures.select_features_boruta_shap`,
  :func:`~rsgispy.regression.regfeatures.calc_feature_importance`,
  :func:`~rsgispy.regression.regfeatures.calc_jackknife_feature_importance`,
  :func:`~rsgispy.regression.regfeatures.calc_feature_response_curves`,
  :func:`~rsgispy.regression.regfeatures.calc_shap_feature_contributions`,
  :func:`~rsgispy.regression.regfeatures.calc_salib_feature_sensitivity`).
* :mod:`rsgispy.regression.regsegments` — Object-based (segment) regression:
  segment features
  (:func:`~rsgispy.regression.regsegments.create_reg_segment_features`),
  per-segment reference value aggregation and group-aware splitting
  (:func:`~rsgispy.regression.regsegments.extract_reg_segment_samples`,
  :func:`~rsgispy.regression.regsegments.split_reg_segment_samples`,
  :func:`~rsgispy.regression.regsegments.extract_and_split_reg_segment_samples`),
  chunked regressor application to segment vector layers
  (:func:`~rsgispy.regression.regsegments.apply_sklearn_regressor_segments`
  and the XGBoost, LightGBM and CatBoost equivalents) and rasterising
  (:func:`~rsgispy.regression.regsegments.rasterise_segment_regression`).
* :mod:`rsgispy.regression.regmetrics` — Regression accuracy metrics, predicted
  raster point sampling, bootstrap confidence intervals, and predicted vs
  observed / residual diagnostic plots
  (:func:`~rsgispy.regression.regmetrics.calc_reg_metrics`,
  :func:`~rsgispy.regression.regmetrics.pop_reg_img_pred_pts`,
  :func:`~rsgispy.regression.regmetrics.calc_reg_stats_ptonly_bootstrap_conf`,
  :func:`~rsgispy.regression.regmetrics.plot_pred_vs_obs`,
  :func:`~rsgispy.regression.regmetrics.plot_residuals`,
  :func:`~rsgispy.regression.regmetrics.plot_qq_residuals`).
"""
