rsgispy.ml
==========

The :mod:`rsgispy.ml` subpackage provides a shared machine-learning core
unifying :mod:`rsgispy.classification`, :mod:`rsgispy.regression`, and
:mod:`rsgispy.modelling`:

.. seealso::

   Guides: :doc:`guides/pixel_classification_workflow`, :doc:`guides/pixel_regression_workflow`, :doc:`guides/segment_ml_workflow`

* :mod:`rsgispy.ml.stack` — :class:`~rsgispy.ml.stack.ImageFeatureStack` and
  :class:`~rsgispy.ml.stack.BandSpec` for managing multi-image feature stacks,
  valid-mask generation, tiled block iteration, and coordinate-preserving
  point/polygon extraction.
* :mod:`rsgispy.ml.scaling` — :class:`~rsgispy.ml.scaling.FeatureScaler`
  providing ``scikit-learn``-compatible ``fit``/``transform`` wrappers over
  every scaling and normalisation method in :mod:`rsgispy.imagecalc.rescale`,
  with JSON persistence and streaming raster stack fitting.
* :mod:`rsgispy.ml.predict` — Memory-bounded tiled raster inference via
  :func:`~rsgispy.ml.predict.predict_raster`,
  :func:`~rsgispy.ml.predict.predict_proba_raster`, and
  :func:`~rsgispy.ml.predict.predict_quantiles_raster` with on-the-fly
  valid-mask computation and optional feature scaling.
* :mod:`rsgispy.ml.cv` — Spatial cross-validation splitters
  (:class:`~rsgispy.ml.cv.SpatialBlockKFold`,
  :class:`~rsgispy.ml.cv.SpatialBufferCV`,
  :class:`~rsgispy.ml.cv.PolygonGroupKFold`) and
  :func:`~rsgispy.ml.cv.nested_spatial_cv` to guard against spatial
  autocorrelation leakage.
* :mod:`rsgispy.ml.tune` — Unified Optuna hyperparameter optimisation
  (:func:`~rsgispy.ml.tune.tune_estimator`) with ``tqdm`` progress reporting
  and fold-local scaling.

Examples
--------

Build a feature stack, fit a scaler, run nested spatial CV, and predict a
raster without writing intermediate scaled GeoTIFFs:

.. code-block:: python

   from sklearn.ensemble import RandomForestClassifier
   import rsgispy.ml.cv
   import rsgispy.ml.predict
   import rsgispy.ml.scaling
   import rsgispy.ml.stack

   stack = rsgispy.ml.stack.ImageFeatureStack(
       [rsgispy.ml.stack.BandSpec("s2_scene.tif", name="s2", bands=(1, 2, 3, 4))]
   )
   df = stack.extract_points("samples.gpkg", "samples", keep_cols=["class_id"])
   X = df[stack.feature_names].to_numpy()
   y = df["class_id"].to_numpy()
   coords = df[["x", "y"]].to_numpy()

   outer_cv = rsgispy.ml.cv.SpatialBlockKFold(n_splits=5, method="grid")
   scaler = rsgispy.ml.scaling.FeatureScaler(method="standard_scaler")

   cv_res = rsgispy.ml.cv.nested_spatial_cv(
       estimator_factory=lambda p: RandomForestClassifier(**p, random_state=42),
       X=X,
       y=y,
       coords=coords,
       outer_cv=outer_cv,
       param_space={"n_estimators": ("int", 50, 200), "max_depth": ("int", 3, 15)},
       scaler=scaler,
       n_trials=20,
       task="classification",
   )

   scaler.fit(X)
   model = RandomForestClassifier(random_state=42).fit(scaler.transform(X), y)
   rsgispy.ml.predict.predict_raster(
       estimator=model,
       feature_stack=stack,
       output_img="predicted_classes.tif",
       scaler=scaler,
   )

Chaining
--------

.. seealso::

   :doc:`guides/tile_graph` — a guide to chaining operations in memory with a
   tile graph, with a worked example.

The predictions are also available as tile-graph nodes,
:func:`~rsgispy.ml.predict.predict_raster_node`,
:func:`~rsgispy.ml.predict.predict_proba_raster_node` and
:func:`~rsgispy.ml.predict.predict_quantiles_raster_node`, taking feature
nodes, a fitted estimator, an optional
:class:`~rsgispy.ml.scaling.FeatureScaler` and an optional mask node. Build
the feature nodes of an :class:`~rsgispy.ml.stack.ImageFeatureStack` with
:func:`~rsgispy.ml.predict.feature_stack_nodes` (or
``stack.to_tile_nodes()``), or use any nodes (filters, indices, band maths)
as features, so a speckle filter, dB conversion, indices, the model and a
majority filter run in one :class:`~rsgispy.imageprocess.TileGraph` pass:

.. code-block:: python

   import numpy as np
   import rsgispy
   from rsgispy.classification.classpostprocess import majority_filter_node
   from rsgispy.imageprocess import DiskSink, RasterSource, TileGraph
   from rsgispy.ml.predict import feature_stack_nodes, predict_raster_node

   cls = predict_raster_node(
       feature_stack_nodes(stack),
       model,
       scaler=scaler,
       thematic=True,
       mask=RasterSource("valid_mask.tif", bands=1),
   )
   smooth = majority_filter_node(cls, 3, dtype=np.uint8, no_data_val=0)
   TileGraph(DiskSink(smooth, "classes_smooth.tif", datatype=rsgispy.TYPE_8UINT)).run()

A pixel is predicted when it is inside the mask and every feature is finite
and not its node's no-data value, as in the file-based functions, and the
nodes give the same values. The file-based functions
(:func:`~rsgispy.ml.predict.predict_raster` etc.) keep their own block loop:
they also accept stacks whose rasters are not on one pixel grid (predicting
over their intersection, reprojecting on the fly), while a tile graph needs
all its rasters on the same grid.

Submodules
----------

.. toctree::
   :maxdepth: 2

   ml/stack
   ml/scaling
   ml/predict
   ml/cv
   ml/tune
   ml/vector
   ml/segments
