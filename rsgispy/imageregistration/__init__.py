"""
The imageregistration subpackage provides pure-Python image registration,
tie-point co-registration, outlier filtering, whole-image shift estimation,
and GCP/polynomial image warping using ``rasterio`` and ``numpy``.

GCP file output formats:

    * ``TYPE_ENVI_IMG2IMG = 1``
    * ``TYPE_ENVI_IMG2MAP = 2``
    * ``TYPE_RSGIS_IMG2MAP = 3``
    * ``TYPE_RSGIS_MAPOFFS = 4``

Tie-point generation methods:

    * ``TIE_METHOD_PHASE_CORR = "phase_correlation"``
    * ``TIE_METHOD_TEMPLATE_MATCH = "template_matching"``

Outlier filtering methods:

    * ``OUTLIER_METHOD_RANSAC = "ransac"``
    * ``OUTLIER_METHOD_RESIDUAL = "residual"``
    * ``OUTLIER_METHOD_MAD = "mad"``

Geometric transformation types:

    * ``TRANSFORM_SHIFT = "shift"``
    * ``TRANSFORM_SIMILARITY = "similarity"``
    * ``TRANSFORM_AFFINE = "affine"``
    * ``TRANSFORM_POLY1 = "poly1"``
    * ``TRANSFORM_POLY2 = "poly2"``
    * ``TRANSFORM_POLY3 = "poly3"``
    * ``TRANSFORM_TPS = "tps"``

Submodules:

* :mod:`rsgispy.imageregistration.shift` — whole-image shift estimation using
  correlation, mutual information, phase correlation, and distance metrics, plus
  linear geotransform offset application
* :mod:`rsgispy.imageregistration.tiepoints` — phase cross-correlation and
  template-matching tie-point generation, geometric model fitting, and RANSAC /
  residual-threshold outlier filtering
* :mod:`rsgispy.imageregistration.warping` — GCP embedding, vector-point GCP
  conversion, GCP/polynomial/shift warping, and end-to-end co-registration
"""

#: ENVI image-to-image pixel-coordinate GCP text format (``1``).
TYPE_ENVI_IMG2IMG = 1
#: ENVI image-to-map GCP text format (``2``).
TYPE_ENVI_IMG2MAP = 2
#: RSGISLib image-to-map GCP text format (``3``).
TYPE_RSGIS_IMG2MAP = 3
#: RSGISLib map-offset GCP text format (``4``).
TYPE_RSGIS_MAPOFFS = 4

#: FFT phase cross-correlation tie-point matching method.
TIE_METHOD_PHASE_CORR = "phase_correlation"
#: Sliding-window template matching tie-point method.
TIE_METHOD_TEMPLATE_MATCH = "template_matching"

#: RANSAC consensus outlier filtering method.
OUTLIER_METHOD_RANSAC = "ransac"
#: Iterative Euclidean residual thresholding outlier filtering method.
OUTLIER_METHOD_RESIDUAL = "residual"
#: Robust MAD-scaled residual thresholding outlier filtering method.
OUTLIER_METHOD_MAD = "mad"

#: Global X/Y translation transformation model.
TRANSFORM_SHIFT = "shift"
#: 2D similarity (translation, rotation, uniform scale) model.
TRANSFORM_SIMILARITY = "similarity"
#: First-order 2D affine transformation model.
TRANSFORM_AFFINE = "affine"
#: First-order 2D polynomial transformation model.
TRANSFORM_POLY1 = "poly1"
#: Second-order 2D polynomial transformation model.
TRANSFORM_POLY2 = "poly2"
#: Third-order 2D polynomial transformation model.
TRANSFORM_POLY3 = "poly3"
#: Thin-plate spline non-rigid transformation model.
TRANSFORM_TPS = "tps"
