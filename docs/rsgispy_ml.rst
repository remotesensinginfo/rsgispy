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
