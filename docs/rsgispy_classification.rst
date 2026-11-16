rsgispy.classification
======================

.. automodule:: rsgispy.classification
   :no-index:

Submodules
----------

* :doc:`classification/classdata` — Image/band and class training definitions, JSON configuration I/O, Parquet/tabular sample extraction/splitting, and unsigned class ID recoding.
* :doc:`classification/classsklearn` — Scikit-learn parameter search (Grid, Random, Optuna), classifier training, and pixel classification with class ID and probability outputs.
* :doc:`classification/classfeatures` — Feature diagnostics (box/violin plots, correlation, VIF), feature selection (CQV, correlation, VIF, univariate, RFE/RFECV, model/LassoLars, BorutaShap), and importance/contribution analysis (permutation/model importance, jackknife, response curves, SHAP, SALib Sobol).
* :doc:`classification/classrefpts` — Random and stratified random reference point generation and accuracy point population.
* :doc:`classification/classaccuracymetrics` — Image area-weighted and point-only classification accuracy statistics and bootstrap confidence intervals.

.. toctree::
   :maxdepth: 1

   classification/classdata
   classification/classsklearn
   classification/classfeatures
   classification/classrefpts
   classification/classaccuracymetrics
