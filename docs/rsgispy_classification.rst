rsgispy.classification
======================

.. seealso::

   Guides: :doc:`guides/pixel_classification_workflow`, :doc:`guides/segment_ml_workflow`, :doc:`guides/change_detection`

.. automodule:: rsgispy.classification
   :no-index:

Chaining
--------

The majority filter is also a tile-graph node,
:func:`~rsgispy.classification.classpostprocess.majority_filter_node`, for
smoothing a classification in the same
:class:`~rsgispy.imageprocess.TileGraph` pass that predicts it (see
:doc:`classification/classpostprocess` and the "Chaining" section of
:doc:`rsgispy_ml`). The sieve filter works on connected components of the
whole image and has no node.

A clusterer fitted by
:func:`~rsgispy.classification.clustersklearn.img_pixel_sample_cluster`
(``return_model=True``) can be applied in a tile graph with
:func:`~rsgispy.ml.predict.predict_raster_node`: pass ``thematic=True`` and a
``lut`` adding the cluster ID offset (``lut={k: k + 1 for k in
range(n_clusters)}``) to get the same cluster IDs as
:func:`~rsgispy.classification.clustersklearn.apply_img_pixel_clusterer`.

Submodules
----------

* :doc:`classification/classdata` — Image/band and class training definitions, JSON configuration I/O, Parquet/tabular sample extraction/splitting, combining pixel and segment samples, and unsigned class ID recoding.
* :doc:`classification/classsklearn` — Scikit-learn parameter search (Grid, Random, Optuna), classifier training, and pixel classification with class ID and probability outputs.
* :doc:`classification/classxgboost` — XGBoost binary and multi-class Optuna hyperparameter optimisation, classifier training, and block-wise pixel classification.
* :doc:`classification/classlightgbm` — LightGBM binary and multi-class Optuna hyperparameter optimisation, classifier training, and block-wise pixel classification.
* :doc:`classification/classcatboost` — CatBoost binary and multi-class Optuna hyperparameter optimisation, classifier training, and block-wise pixel classification.
* :doc:`classification/clustersklearn` — Scikit-learn unsupervised pixel classification using either full-image clustering or pixel sample fitting with whole-image prediction, and applying a fitted clusterer to another image.
* :doc:`classification/classfeatures` — Feature diagnostics (box/violin plots, correlation, VIF), feature selection (CQV, correlation, VIF, univariate, RFE/RFECV, model/LassoLars, BorutaShap), and importance/contribution analysis (permutation/model importance, jackknife, response curves, SHAP, SALib Sobol).
* :doc:`classification/classrefpts` — Random and stratified random reference point generation and accuracy point population.
* :doc:`classification/classaccuracymetrics` — Image area-weighted and point-only classification accuracy statistics and bootstrap confidence intervals.
* :doc:`classification/classsegments` — Object-based (segment) classification: segment feature creation, one-sample-per-segment training data extraction and group-aware splitting, chunked application of scikit-learn/XGBoost/LightGBM/CatBoost classifiers to segment vector layers, rasterising with class colours, segment clustering, and accuracy points. See the :doc:`guides/segment_ml_workflow` guide.
* :doc:`classification/classpostprocess` — Thematic classification post-processing (moving-window majority/modal filter, also as a tile-graph node, and 4/8-connected sieve clump elimination).

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
