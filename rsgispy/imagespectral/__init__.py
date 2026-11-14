"""
The imagespectral subpackage provides modules for spectral image processing,
endmember management, linear spectral unmixing, spectral classification,
spectral feature transformations, and spectral pre-processing using
``rasterio``, ``numpy``, and ``pandas``:

* :mod:`rsgispy.imagespectral.classification` — spectral similarity mapping and
  classification (:func:`~rsgispy.imagespectral.classification.spectral_angle_mapper`,
  :func:`~rsgispy.imagespectral.classification.spectral_correlation_mapper`)
* :mod:`rsgispy.imagespectral.endmembers` — endmember table I/O (DataFrame and
  legacy ``.mtxt`` conversion), polygon extraction, plotting, equality checks,
  and Pixel Purity Index (:func:`~rsgispy.imagespectral.endmembers.calc_ppi`)
* :mod:`rsgispy.imagespectral.indices` — spectral curve and index summary
  utilities (:func:`~rsgispy.imagespectral.indices.calc_area_under_curve`)
* :mod:`rsgispy.imagespectral.preprocessing` — Savitzky-Golay spectral
  smoothing, outlier masking, and polynomial NaN gap-filling
* :mod:`rsgispy.imagespectral.transforms` — ZCA noise whitening, windowed/incremental
  PCA, Minimum Noise Fraction (MNF), Tasseled Cap transformations, and
  streaming histogram matching
* :mod:`rsgispy.imagespectral.unmixing` — unconstrained, non-negative, fully
  constrained, and exhaustive linear spectral unmixing, reflectance prediction,
  RMSE/RMSPE evaluation, and multi-endmember model summarisation
"""
