Object-based (segment) classification and regression
=====================================================

This guide walks through an object-based image analysis (OBIA) workflow in
rsgispy: an image is segmented into objects (segments), each segment is
described by statistics of the image bands inside it and by its shape, and
machine learning models are trained and applied to the segments rather than
to individual pixels.

If you know the pixel-based workflow
(:mod:`rsgispy.classification` / :mod:`rsgispy.regression`), almost
everything carries over: the same configuration objects
(:class:`~rsgispy.classification.classdata.ClassificationConfig`,
:class:`~rsgispy.regression.regdata.RegressionConfig`), the same training and
optimisation functions, and the same accuracy assessment. Only creating the
training samples and applying the model are different, because the unit of
analysis is a segment in a vector layer instead of a pixel in a raster.

Why segments?

* Segment statistics (mean, standard deviation, …) are less noisy than
  single pixels, and segments add information pixels do not have, such as
  texture within the object and its shape (area, compactness, elongation).
* The output is a map of objects without the "salt and pepper" noise of pixel
  classifications.
* There are far fewer segments than pixels, so applying a model is fast.

.. contents:: Contents
   :local:
   :depth: 1

Installation
------------

The workflow needs the classification/regression extras (``scikit-learn``,
``pyarrow``, ``optuna``); the XGBoost, LightGBM and CatBoost models need the
``gbdt`` extra:

.. code-block:: bash

   pip install "rsgispy[classification,regression,gbdt]"

Overview
--------

.. code-block:: text

   Sentinel-2 SREF stack + slope image
            │
            ▼
   1. Segmentation ─────────────► segment label raster (s2_segs.tif)
            │
            ▼
   2. Segment features ─────────► segment layer (s2_segs.parquet):
            │                       seg_id, valid, band statistics,
            │                       shape metrics, geometry
            ▼
   3. Config (JSON, seg_info) ◄── training points/polygons per class
            │
            ▼
   4. Training samples ─────────► one row per segment, split into
            │                       train / test (grouped)
            ▼
   5–6. Feature selection, train / optimise (existing functions)
            │
            ▼
   7. Apply to segments (chunked) ► classified segment layer
            │
            ├─► 8. Rasterise with class colours ─► classified raster
            ├─► 9. Accuracy assessment
            ├─► 10. Apply to another scene
            ├─► 11. Regression (canopy cover)
            └─► 12. Clustering (unsupervised)

The code in this guide comes from the example script
``docs/examples/segment_ml_workflow.py`` (included below step by step), which
is run by the rsgispy test suite on synthetic data. The scenario is a
Sentinel-2 surface reflectance image (e.g., the 10 m stack from
:func:`~rsgispy.imagecalibration.sensorlvl2data.create_stacked_sen2_esa_l2a_img`,
with no data value 0) and a slope image derived from a DEM on the same grid.
The land cover classes (e.g., Water, Forest, Grassland, Urban) each have a
training vector file of points and/or polygons, and a set of field plots
gives canopy cover for the regression example.

Step 1 — Segmentation
---------------------

The Shepherd et al. (2019) algorithm clusters the pixels spectrally
(K-means with ``n_clusters`` clusters), groups connected pixels of the same
cluster into segments and merges segments smaller than ``min_pixels`` into
their most similar neighbour. More clusters and a smaller minimum size give
smaller, more homogeneous segments. Use the bands that best separate the
objects you are interested in (typically visible, NIR and SWIR). The other
segmentation algorithms in :mod:`rsgispy.segmentation` (SLIC, region merging,
graph-based, watershed) produce the same kind of label raster and can be used
instead.

.. literalinclude:: ../examples/segment_ml_workflow.py
   :language: python
   :start-after: # [step-1-segmentation]
   :end-before: # [end-step-1-segmentation]
   :dedent: 4

Step 2 — Segment features
-------------------------

:func:`~rsgispy.classification.classsegments.create_cls_segment_features`
builds the segment layer from the label raster and the config's image bands:

* **Band statistics** — for every image band, the chosen statistics over the
  pixels of each segment (here mean, standard deviation, minimum and
  maximum; median, percentiles and mode are also available). They are
  computed by :func:`~rsgispy.zonalstats.labels.calc_label_zonal_stats` in a
  single pass over the label raster, which is far faster than polygon zonal
  statistics when there are many segments. Pixels that are no data (here
  ``0`` in the Sentinel-2 stack, set with ``band_opts``) are ignored.
* **Categorical bands** — a band marked as categorical (e.g. a land cover or
  soil map; ``band_opts={'lc': {'data_type': rsgispy.VAR_TYPE_CATEGORICAL}}``)
  gives the most common category and the proportion of each category in the
  segment, instead of means.
* **Shape metrics** — computed with
  :func:`~rsgispy.vector.stats.calc_shape_metrics_geoseries` (area,
  perimeter, compactness, solidity, elongation, orientation, …; see its
  documentation for the formulas). Use a projected CRS, or set
  ``shape_proj_epsg``.
* **valid** and **pxl_count** — segments with no valid pixels in any band are
  flagged ``valid = 0`` and are never classified.

Feature columns are named ``<band name>_<statistic>`` (or
``<image name>_b<band>_<statistic>`` if band names are not given) and
``shp_<metric>``. The layer is written as GeoParquet with snappy compression
(any vector format, e.g. ``.gpkg``, can be used instead), with one row per
segment, sorted by ``seg_id``. Use ``out_table_file`` to also write the
features as an attribute-only Parquet table.

.. literalinclude:: ../examples/segment_ml_workflow.py
   :language: python
   :start-after: # [step-2-features]
   :end-before: # [end-step-2-features]
   :dedent: 4

The printed feature list looks like:

.. code-block:: text

   ['Blue_mean', 'Blue_stddev', 'Blue_min', 'Blue_max', ..., 'slope_max',
    'shp_area', 'shp_compactness', 'shp_solidity', 'shp_elongation']

Step 3 — The configuration
--------------------------

The returned config is the classification config with a ``seg_info`` block
(:class:`~rsgispy.ml.segments.SegmentFeatureConfig`) recording everything
needed to use the segments later: the label raster, the segment layer, the
ID and valid columns, the statistics and shape metrics, the ordered feature
columns used by the model, the missing value policy and the rules for
assigning training data to segments. It is saved in the JSON config file, so
every later step only needs the config file. Configs without ``seg_info``
(pixel workflows) are unchanged.

.. literalinclude:: ../examples/segment_ml_workflow.py
   :language: python
   :start-after: # [step-3-config]
   :end-before: # [end-step-3-config]
   :dedent: 4

Missing feature values (``seg_info.missing``):

* ``'drop'`` (default) — segments with a missing feature value are not used
  for training and are not classified. Works with every model.
* ``'nan'`` — missing values are passed to the model as NaN. Use with
  XGBoost, LightGBM and CatBoost, which handle missing values natively.
* ``'fill'`` — missing values are replaced by ``fill_value``; use with
  scikit-learn models when every segment must be classified.

Step 4 — Training samples
-------------------------

:func:`~rsgispy.classification.classsegments.extract_class_segment_samples`
turns the training points and polygons of each class into **one sample per
segment**:

* a point is assigned to the segment under it;
* a polygon is assigned to the segments it covers by at least
  ``seg_info.min_overlap`` (default 0.5) of their pixels, or, with
  ``poly_assign='centroid'``, to the segments whose centroid it contains;
* a segment hit by several points or polygons of the same class still gives
  one sample (so samples are not duplicated);
* a segment hit by training data of different classes is dropped (or, with
  ``conflict='majority'``, takes the class with most hits); a warning reports
  how many;
* invalid segments and segments smaller than ``min_seg_pxls`` are removed.

The sample table has the same columns as the pixel workflow
(``out_cls_id``, ``internal_cls_id``, ``class_name`` and the features) plus
``seg_id``, ``n_hits`` and ``train_grp``. ``train_grp`` identifies the source
polygon: :func:`~rsgispy.classification.classsegments.split_class_segment_samples`
keeps all segments from one training polygon in the same set (train or test),
because neighbouring segments of one polygon are very similar and splitting
them would make the test accuracy too optimistic. A summary of the assignment
is in ``samples.attrs['report']``.

.. literalinclude:: ../examples/segment_ml_workflow.py
   :language: python
   :start-after: # [step-4-samples]
   :end-before: # [end-step-4-samples]
   :dedent: 4

.. note::

   **Small sample sizes.** Hundreds of training points or polygons give
   millions of pixel samples, but only hundreds of segment samples. Use fewer
   cross-validation folds (e.g., 3), prefer cross-validation to a separate
   validation set (``valid_prop=0``), balance classes with class weights
   (e.g., ``class_weight='balanced'``), and for GBDT models reduce the minimum
   leaf size (e.g., LightGBM ``min_child_samples``) and be careful with early
   stopping on very small validation sets. The split functions raise an
   exception when a class has fewer groups than sets requested, and warn when
   a class has fewer than ten samples.

   Pixel samples can be added to the segment samples with
   :func:`~rsgispy.classification.classdata.combine_cls_samples`, which
   checks that the class names and IDs agree and writes one table in the
   sample layout (``source_col`` records where each row came from). This is
   only sound for features that mean the same for a pixel and a segment
   (a band value and a segment's mean of that band), so name the pixel
   features as the segment features
   (``ImageBandInfo(..., band_names=cfg.seg_info.feature_cols)``) and leave
   out the segment-only features (standard deviations, shape).

Step 5 — Exploring and selecting features (optional)
----------------------------------------------------

The functions in :mod:`~rsgispy.classification.classfeatures` work with the
segment samples unchanged. When a config has ``seg_info``, the feature columns
default to its model features. A selected subset can be stored in
``seg_info.selected_features``; the model is then trained and applied with
those features only.

.. literalinclude:: ../examples/segment_ml_workflow.py
   :language: python
   :start-after: # [step-5-features-explore]
   :end-before: # [end-step-5-features-explore]
   :dedent: 4

Step 6 — Training and optimising
--------------------------------

The existing training and optimisation functions are used unchanged with the
segment sample tables and the config, e.g. a random forest with
:func:`~rsgispy.classification.classsklearn.train_sklearn_classifier` (or
:func:`~rsgispy.classification.classsklearn.perform_sklearn_classifier_param_search`
to optimise its hyperparameters):

.. literalinclude:: ../examples/segment_ml_workflow.py
   :language: python
   :start-after: # [step-6-train]
   :end-before: # [end-step-6-train]
   :dedent: 4

or a gradient boosted model, e.g. XGBoost (with
:func:`~rsgispy.classification.classxgboost.optimise_xgboost_multiclass_classifier`
to optimise it):

.. literalinclude:: ../examples/segment_ml_workflow.py
   :language: python
   :start-after: # [step-6b-train-gbdt]
   :end-before: # [end-step-6b-train-gbdt]
   :dedent: 4

Step 7 — Applying the classifier to the segments
------------------------------------------------

:func:`~rsgispy.classification.classsegments.apply_sklearn_classifier_segments`
(and the XGBoost, LightGBM and CatBoost equivalents, e.g.
:func:`~rsgispy.classification.classsegments.apply_xgboost_multiclass_classifier_segments`)
applies the model to the segment layer **in chunks** of ``chunk_size``
segments. For the prediction only the segment ID, the valid flag and the
model's feature columns are read — never the geometry — so even layers with
millions of segments and many feature columns need little memory. The
geometry is copied to the output as stored (WKB, not decoded).

The output vector layer has:

* ``cls_id`` — the class ``out_id`` (``0``, or the next unused value, for
  segments that were not classified);
* ``cls_name``, ``cls_colour`` (``#rrggbb``) and ``cls_red``/``cls_green``/
  ``cls_blue`` from the config;
* optionally the class probabilities (``out_probs='all'``, ``'max'`` or, for
  binary models, ``'pos'``);
* by default all the input columns (``copy_cols=[]`` keeps only ``seg_id``).

Provenance (rsgispy version, model file and hash, model type, feature list,
creation time) is stored in the file metadata.

.. literalinclude:: ../examples/segment_ml_workflow.py
   :language: python
   :start-after: # [step-7-apply]
   :end-before: # [end-step-7-apply]
   :dedent: 4

Step 8 — Rasterising the classification
---------------------------------------

:func:`~rsgispy.classification.classsegments.rasterise_segment_classification`
maps each segment's class onto the label raster grid. It reads only the
segment ID and class columns and streams the label raster through a lookup,
so no geometry is burnt. Pixel values are the class ``out_id``\ s, the data
type is the smallest unsigned integer type that fits them, and the class
colours are written as a colour table. GeoTIFF has no attribute table for
the class names, so the example stores them (with the colours) as a band tag
with :func:`~rsgispy.imageutils.colourtable.set_thematic_classes`; read them
back with :func:`~rsgispy.imageutils.colourtable.get_thematic_classes`. With
``out_score_img`` the
probabilities are written to a float raster. The thematic post-processing in
:mod:`~rsgispy.classification.classpostprocess` can be applied to the
result if needed, but segment classifications rarely need it.

.. literalinclude:: ../examples/segment_ml_workflow.py
   :language: python
   :start-after: # [step-8-rasterise]
   :end-before: # [end-step-8-rasterise]
   :dedent: 4

Step 9 — Accuracy assessment
----------------------------

:func:`~rsgispy.classification.classsegments.pop_segment_cls_accuracy_pts`
adds the class of the segment under each accuracy point. The existing
functions in :mod:`~rsgispy.classification.classaccuracymetrics` then give
point-based accuracy, and area-weighted accuracy using the class areas of the
rasterised classification.

In the output of
:func:`~rsgispy.classification.classaccuracymetrics.calc_acc_stats_img`,
``accuracy``, the user's and producer's accuracies, F1 and kappa are still
computed from the point counts. The area-weighted results are
``norm_confusion_matrix``, ``est_prop_cls_area`` (the estimated proportion of
the map in each class) and ``quantity_metrics["Proportion Correct (C)"]``.
:doc:`pixel_classification_workflow` explains these outputs in more detail.

.. literalinclude:: ../examples/segment_ml_workflow.py
   :language: python
   :start-after: # [step-9-accuracy]
   :end-before: # [end-step-9-accuracy]
   :dedent: 4

Step 10 — Applying the model to another scene
---------------------------------------------

A trained model can be applied to another date or tile.
:func:`~rsgispy.ml.segments.substitute_scene` returns a copy of the config
with the image paths replaced (matched by image name) and the new label
raster. Creating the features with ``template=new_cfg.seg_info`` uses exactly
the same statistics, shape metrics and category lists, and raises an
exception if any feature the model needs cannot be produced (for example a
missing band). The new scene must be comparable to the training scene
(same sensor, processing and scaling).

.. literalinclude:: ../examples/segment_ml_workflow.py
   :language: python
   :start-after: # [step-10-new-scene]
   :end-before: # [end-step-10-new-scene]
   :dedent: 4

Step 11 — Segment regression
----------------------------

Regression follows the same steps with
:class:`~rsgispy.regression.regdata.RegressionConfig` and
:mod:`rsgispy.regression.regsegments`. Here the segment features created for
the classification are reused by passing the same ``seg_info``. Field plot
values are assigned to segments (several plots in one segment are averaged,
or the median taken with ``reg_agg='median'``, and their number recorded in
``n_ref``). The model is applied in chunks, writing ``pred_canopy_cover`` and,
for tree ensembles, prediction quantiles; the result is rasterised with
:func:`~rsgispy.regression.regsegments.rasterise_segment_regression`.
Several targets can be predicted at once by giving a list of target columns.

.. literalinclude:: ../examples/segment_ml_workflow.py
   :language: python
   :start-after: # [step-11-regression]
   :end-before: # [end-step-11-regression]
   :dedent: 4

Step 12 — Clustering segments
-----------------------------

:func:`~rsgispy.classification.classsegments.cluster_segments` fits a
clusterer (e.g. K-means) to all, or a sample of, the segments and labels every
segment with a cluster ID (``1..k``; ``0`` for invalid segments). Scaling the
features first (here z-scores) stops features with large values dominating.

.. literalinclude:: ../examples/segment_ml_workflow.py
   :language: python
   :start-after: # [step-12-clustering]
   :end-before: # [end-step-12-clustering]
   :dedent: 4

Performance and large datasets
------------------------------

* **Segment statistics** —
  :func:`~rsgispy.zonalstats.labels.calc_label_zonal_stats` reads every pixel
  once. In a benchmark with a 2000 x 2000 pixel image, 40,000 segments and 4
  bands it took 0.24 s, against 22 s for polygon zonal statistics (about 90
  times faster), and the gap grows with the number of segments. Memory grows
  with the number of segments, not the image size. Median, percentiles and
  mode of float bands use a histogram (``quantile_method='hist'``) unless
  ``quantile_method='exact'`` is set.
* **Segment polygons** — the segment layer is polygonised with
  :func:`~rsgispy.vector.create.polygonise_raster_tiled` (one
  (Multi)Polygon per segment, sorted by ID), in tiles of ``tile_size`` pixels
  and on ``n_jobs`` cores (options of
  :func:`~rsgispy.ml.segments.create_segment_features` and
  :func:`~rsgispy.classification.classsegments.create_cls_segment_features`).
  The polygons are streamed to disk and joined to the statistics chunk by
  chunk, so they are never all held in memory. See
  :ref:`polygonise-benchmark` for timings.
* **Large or sparse segment IDs** — label rasters with very large IDs (e.g.
  ``uint64`` IDs from tiled segmentation) are handled with a compact ID
  mapping, so memory is not proportional to the largest ID.
* **chunk_size** — the number of segments processed at a time when applying
  a model (default 100,000). Memory per chunk is roughly
  ``chunk_size x number of features x 8`` bytes plus the copied columns;
  reduce it if memory is short, increase it for slightly faster processing.
* **Formats** — GeoParquet is recommended: it is compressed, columnar (so
  reading a few feature columns is fast) and written in row groups. Other
  formats (e.g. GeoPackage) work but are slower to read column subsets.
* **Separate feature table** — with ``out_table_file`` the features are also
  written to an attribute-only table; the apply functions then read the
  features from the table and the geometry from the segment layer, joined on
  ``seg_id``.

Common problems
---------------

* **"not on the same pixel grid"** — every image band must have the same
  size, transform and CRS as the label raster. Resample the image first, e.g.
  with :func:`rsgispy.imageutils.warp.resample_img_to_match`.
* **CRS mismatch of training data** — training vectors in a different CRS
  from the segments raise an exception; pass ``reproj_vec=True`` (and
  ``vec_def_epsg`` for vectors without a CRS) to reproject them on the fly.
* **Missing feature columns when applying a model** — the model was trained
  with features that are not in the segment layer (e.g. a feature selection,
  or features from another scene). Create the features with
  ``template=cfg.seg_info`` so the same columns are produced.
* **Classes with too few segments** — the split functions raise an exception
  if a class has fewer groups (training polygons or segments) than requested
  sets; add training data or use ``valid_prop=0`` and cross-validation.
* **NaN features with scikit-learn models** — with ``missing='nan'`` most
  scikit-learn estimators fail on NaN values; use ``'drop'`` or ``'fill'``, or
  a GBDT model.
* **A model predicts a single class** — with few samples, GBDT models may not
  split at all because of their default minimum leaf sizes; reduce them (e.g.
  LightGBM ``min_child_samples``) or add training data.

Related guides
--------------

* :doc:`pixel_classification_workflow`
* :doc:`pixel_regression_workflow`
* :doc:`large_rasters`
