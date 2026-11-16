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
