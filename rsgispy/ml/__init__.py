"""
Unified machine learning core subpackage (:mod:`rsgispy.ml`) shared across
:mod:`rsgispy.classification`, :mod:`rsgispy.regression`, and
:mod:`rsgispy.modelling`.

Submodules:

* :mod:`rsgispy.ml.stack` — :class:`~rsgispy.ml.stack.ImageFeatureStack` and
  :class:`~rsgispy.ml.stack.BandSpec` unifying raster stacks, valid-mask
  generation, coordinate-preserving sample extraction, and tiled feature reading.
* :mod:`rsgispy.ml.scaling` — :class:`~rsgispy.ml.scaling.FeatureScaler`
  integrating all 23 scaling/normalisation methods from
  :mod:`rsgispy.imagecalc.rescale` as a fold-safe, JSON-serialisable transformer.
* :mod:`rsgispy.ml.predict` — Unified tiled raster prediction
  (:func:`~rsgispy.ml.predict.predict_raster`,
  :func:`~rsgispy.ml.predict.predict_proba_raster`,
  :func:`~rsgispy.ml.predict.predict_quantiles_raster`) with on-the-fly valid
  masking and feature scaling.
* :mod:`rsgispy.ml.cv` — Spatial cross-validation splitters
  (:class:`~rsgispy.ml.cv.SpatialBlockKFold`,
  :class:`~rsgispy.ml.cv.SpatialBufferCV`,
  :class:`~rsgispy.ml.cv.PolygonGroupKFold`) and
  :func:`~rsgispy.ml.cv.nested_spatial_cv`.
* :mod:`rsgispy.ml.tune` — Unified Optuna hyperparameter tuning
  (:func:`~rsgispy.ml.tune.tune_estimator`) with spatial CV, fold-local
  scaling, and ``tqdm`` progress reporting.
"""

from rsgispy.ml import cv, predict, scaling, stack, tune

__all__ = [
    "stack",
    "scaling",
    "predict",
    "cv",
    "tune",
]
