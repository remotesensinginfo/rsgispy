rsgispy.imagespectral
=====================

The :mod:`rsgispy.imagespectral` subpackage provides spectral image processing,
classification, linear spectral unmixing, endmember table management, spectral
feature transformations, and pre-processing using ``rasterio``, ``numpy``, and
``pandas``.

Submodules
----------

* :doc:`imagespectral/classification` — Spectral Angle Mapper (SAM) and Spectral Correlation Mapper (SCM) similarity mapping and classification.
* :doc:`imagespectral/endmembers` — Endmember ``pandas.DataFrame`` I/O, legacy ``.mtxt`` conversion, polygon extraction, plotting, equality checks, and Pixel Purity Index (PPI).
* :doc:`imagespectral/indices` — Spectral curve and index summary utilities, including area under the curve.
* :doc:`imagespectral/preprocessing` — Savitzky-Golay spectral smoothing, outlier masking, and polynomial NaN gap-filling.
* :doc:`imagespectral/transforms` — ZCA noise whitening, windowed/incremental PCA, Minimum Noise Fraction (MNF), Tasseled Cap transformations, and streaming histogram matching.
* :doc:`imagespectral/unmixing` — Unconstrained, non-negative, fully constrained, and exhaustive linear spectral unmixing, reflectance prediction, RMSE/RMSPE evaluation, and multi-endmember model summarisation.

.. toctree::
   :maxdepth: 1

   imagespectral/classification
   imagespectral/endmembers
   imagespectral/indices
   imagespectral/preprocessing
   imagespectral/transforms
   imagespectral/unmixing
