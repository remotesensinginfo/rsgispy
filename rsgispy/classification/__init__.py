"""
The classification subpackage provides pure-Python reference point sampling,
accuracy point population, and classification accuracy assessment metrics using
``numpy``, ``rasterio``, ``geopandas``, and ``scikit-learn``.

Submodules:

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
