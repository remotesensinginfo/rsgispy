rsgispy.classification
======================

.. seealso::

   Guides: :doc:`guides/pixel_classification_workflow`, :doc:`guides/segment_ml_workflow`, :doc:`guides/change_detection`

.. automodule:: rsgispy.classification
   :no-index:

Submodules
----------

* :doc:`classification/classdata` — Image/band and class training definitions, JSON configuration I/O, Parquet/tabular sample extraction/splitting, and unsigned class ID recoding.
* :doc:`classification/classsklearn` — Scikit-learn parameter search (Grid, Random, Optuna), classifier training, and pixel classification with class ID and probability outputs.
* :doc:`classification/classxgboost` — XGBoost binary and multi-class Optuna hyperparameter optimisation, classifier training, and block-wise pixel classification.
* :doc:`classification/classlightgbm` — LightGBM binary and multi-class Optuna hyperparameter optimisation, classifier training, and block-wise pixel classification.
* :doc:`classification/classcatboost` — CatBoost binary and multi-class Optuna hyperparameter optimisation, classifier training, and block-wise pixel classification.
* :doc:`classification/clustersklearn` — Scikit-learn unsupervised pixel classification using either full-image clustering or pixel sample fitting with whole-image prediction.
* :doc:`classification/classfeatures` — Feature diagnostics (box/violin plots, correlation, VIF), feature selection (CQV, correlation, VIF, univariate, RFE/RFECV, model/LassoLars, BorutaShap), and importance/contribution analysis (permutation/model importance, jackknife, response curves, SHAP, SALib Sobol).
* :doc:`classification/classrefpts` — Random and stratified random reference point generation and accuracy point population.
* :doc:`classification/classaccuracymetrics` — Image area-weighted and point-only classification accuracy statistics and bootstrap confidence intervals.
* :doc:`classification/classsegments` — Object-based (segment) classification: segment feature creation, one-sample-per-segment training data extraction and group-aware splitting, chunked application of scikit-learn/XGBoost/LightGBM/CatBoost classifiers to segment vector layers, rasterising with class colours, segment clustering, and accuracy points. See the :doc:`guides/segment_ml_workflow` guide.
* :doc:`classification/classpostprocess` — Thematic classification post-processing (moving-window majority/modal filter and 4/8-connected sieve clump elimination).

.. toctree::
   :maxdepth: 1

   classification/classdata
   classification/classsklearn
   classification/classxgboost
   classification/classlightgbm
   classification/classcatboost
   classification/clustersklearn
   classification/classfeatures
   classification/classrefpts
   classification/classaccuracymetrics
   classification/classsegments
   classification/classpostprocess
