"""
The classification subpackage provides pure-Python reference point sampling,
accuracy point population, and classification accuracy assessment metrics using
``numpy``, ``rasterio``, ``geopandas``, and ``scikit-learn``.

Submodules:

* :mod:`rsgispy.classification.classdata` — Classification image/band and
  class training data structures (:class:`~rsgispy.classification.classdata.ImageBandInfo`,
  :class:`~rsgispy.classification.classdata.ClassInfoObj`,
  :class:`~rsgispy.classification.classdata.ClassificationConfig`), JSON
  configuration I/O (:func:`~rsgispy.classification.classdata.write_cls_config`,
  :func:`~rsgispy.classification.classdata.read_cls_config`), sample table I/O,
  extraction, and train/test/validation splitting defaulting to Snappy-compressed
  Parquet alongside other pandas-supported formats
  (:func:`~rsgispy.classification.classdata.write_cls_samples`,
  :func:`~rsgispy.classification.classdata.read_cls_samples`,
  :func:`~rsgispy.classification.classdata.extract_class_training_samples`,
  :func:`~rsgispy.classification.classdata.split_class_samples`,
  :func:`~rsgispy.classification.classdata.extract_and_split_class_samples`),
  and internal-to-output unsigned class ID recoding
  (:func:`~rsgispy.classification.classdata.recode_internal_to_out_ids`,
  :func:`~rsgispy.classification.classdata.recode_cls_img_to_out_ids`).
* :mod:`rsgispy.classification.classsklearn` — Scikit-learn classifier
  hyperparameter optimisation (Grid Search, Random Search, and Optuna Bayesian
  optimisation), model training and persistence, and block-wise image
  classification outputting class IDs and probability scores
  (:func:`~rsgispy.classification.classsklearn.perform_sklearn_classifier_param_search`,
  :func:`~rsgispy.classification.classsklearn.train_sklearn_classifier`,
  :func:`~rsgispy.classification.classsklearn.apply_sklearn_classifier`,
  :func:`~rsgispy.classification.classsklearn.save_sklearn_classifier`,
  :func:`~rsgispy.classification.classsklearn.load_sklearn_classifier`).
* :mod:`rsgispy.classification.clustersklearn` — Scikit-learn unsupervised
  pixel classification (clustering) across either the full image dataset in
  memory or a sampled pixel subset applied block-by-block across the image
  (:func:`~rsgispy.classification.clustersklearn.img_pixel_cluster`,
  :func:`~rsgispy.classification.clustersklearn.img_pixel_sample_cluster`).
* :mod:`rsgispy.classification.classfeatures` — Exploratory feature plots and
  multicollinearity diagnostics
  (:func:`~rsgispy.classification.classfeatures.comparison_box_plots`,
  :func:`~rsgispy.classification.classfeatures.correlation_matrix`,
  :func:`~rsgispy.classification.classfeatures.calc_vif_multicollinearity`),
  feature selection
  (:func:`~rsgispy.classification.classfeatures.select_features_cqv`,
  :func:`~rsgispy.classification.classfeatures.select_features_corr`,
  :func:`~rsgispy.classification.classfeatures.select_features_vif`,
  :func:`~rsgispy.classification.classfeatures.select_features_univariate`,
  :func:`~rsgispy.classification.classfeatures.select_features_rfe`,
  :func:`~rsgispy.classification.classfeatures.select_features_from_model`,
  :func:`~rsgispy.classification.classfeatures.select_features_boruta_shap`),
  and feature importance, response curve, SHAP, and Sobol sensitivity analysis
  (:func:`~rsgispy.classification.classfeatures.calc_feature_importance`,
  :func:`~rsgispy.classification.classfeatures.calc_jackknife_feature_importance`,
  :func:`~rsgispy.classification.classfeatures.calc_feature_response_curves`,
  :func:`~rsgispy.classification.classfeatures.calc_shap_feature_contributions`,
  :func:`~rsgispy.classification.classfeatures.calc_salib_feature_sensitivity`).
* :mod:`rsgispy.classification.classrefpts` — Random and stratified random
  reference point generation and accuracy point population
  (:func:`~rsgispy.classification.classrefpts.create_random_ref_smpls_darts`,
  :func:`~rsgispy.classification.classrefpts.create_stratified_random_ref_smpls_darts`,
  :func:`~rsgispy.classification.classrefpts.create_stratified_random_ref_smpls_lut`,
  :func:`~rsgispy.classification.classrefpts.pop_class_info_accuracy_pts`).
* :mod:`rsgispy.classification.classaccuracymetrics` — Image area-weighted and
  point-only classification accuracy statistics and bootstrap confidence
  intervals
  (:func:`~rsgispy.classification.classaccuracymetrics.calc_acc_stats_img`,
  :func:`~rsgispy.classification.classaccuracymetrics.calc_acc_stats_ptonly`,
  :func:`~rsgispy.classification.classaccuracymetrics.calc_acc_stats_img_bootstrap_conf`,
  :func:`~rsgispy.classification.classaccuracymetrics.calc_acc_stats_ptonly_bootstrap_conf`).
"""
