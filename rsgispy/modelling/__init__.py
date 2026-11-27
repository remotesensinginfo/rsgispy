"""
The modelling subpackage provides pure-Python environmental and spatial
modelling workflows built on ``scikit-learn``, ``geopandas``, ``numpy``,
``pandas``, and ``rasterio`` via :mod:`rsgispy.imageprocess`.

Submodules:

* :mod:`rsgispy.modelling.species_distribution` — Species distribution
  modelling (pseudo-absence generation, sampling-bias grids and bias-weighted
  background, environmental covariate extraction, environmental-space
  thinning, normalisation, collinearity diagnostics, isolated scikit-learn
  model fitting (including an in-house MaxEnt) and k-fold or spatial
  cross-validation ensembles, ROC/AUC, TSS, continuous Boyce index and
  null-model evaluation, residual spatial autocorrelation and threshold
  selection, variable response/importance analysis, block-wise spatial raster
  prediction with ensemble uncertainty bands, projection to other scenarios
  with MESS / ExDet extrapolation maps, and range-change maps).
"""
