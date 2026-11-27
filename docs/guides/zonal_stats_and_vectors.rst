Zonal statistics and vector analysis
====================================

This guide is a set of short recipes for summarising raster data over vector
features and for preparing vector layers for analysis: statistics for
polygons, values at points, statistics from a label raster, hexagon grids as
reporting units, polygon shape metrics, spatial selection and splitting
samples into training and testing sets.

Why a separate guide? rsgispy has three ways to get raster values into a
vector layer, and they give the same numbers but differ a lot in speed and in
what they assume about the data. Choosing the right one is the main decision
in most zonal statistics tasks:

.. list-table::
   :header-rows: 1
   :widths: 22 39 39

   * - Method
     - Use it when
     - Notes
   * - Polygon statistics
       (:mod:`rsgispy.zonalstats.polygon`)
     - A few hundred to a few thousand polygons; polygons may overlap; one
       band at a time is enough.
     - Reads the pixels under each polygon separately, so the time grows with
       the number of polygons. A pixel is used when its centre is inside the
       polygon.
   * - Point values
       (:mod:`rsgispy.zonalstats.point`)
     - Field plots, sample points, accuracy points.
     - One pixel per point and per band; fast even for many thousands of
       points.
   * - Label raster statistics
       (:mod:`rsgispy.zonalstats.labels`)
     - Many polygons (thousands to millions), several bands, or medians and
       percentiles; segments from :mod:`rsgispy.segmentation`.
     - Reads each pixel once whatever the number of zones. Zones cannot
       overlap, because each pixel holds one label.

All the recipes use the same files and the same conventions: a Sentinel-2
surface reflectance image, field polygons, survey points and a region of
interest, all in one projected CRS (metres).

.. seealso::

   The rsgispy tutorials cover zonal statistics in a notebook you can run:
   :tutorial:`Zonal statistics <03_vector_data_and_zonal_stats/01_zonal_statistics.ipynb>`
   (section 3), with point values, buffered points and a time series of
   Sentinel-1 backscatter for rice fields in the Mekong Delta. See
   :doc:`../tutorials` for how to get them and their data.

.. contents:: Contents
   :local:
   :depth: 1

Installation
------------

The polygon, point and shape metric recipes only need the core rsgispy
dependencies. The label raster statistics write a Parquet table (``pyarrow``)
and the hexagon grid uses ``h3``; both are in the ``vector`` extra:

.. code-block:: bash

   pip install "rsgispy[vector]"

Overview
--------

.. code-block:: text

   Sentinel-2 SREF stack            field polygons      survey points    ROI
            │                             │                   │           │
            ▼                             │                   │           │
   1. NDVI ───────────────► s2_ndvi.tif   │                   │           │
            │                             ▼                   │           │
            ├──────────► 2. Polygon statistics ► fields_ndvi.gpkg         │
            │                                                 ▼           │
            ├──────────► 3. Point values ──────────────► plots_s2.gpkg    │
            │                                                 │           │
            ├──────────► 4. Label raster statistics ► fields_stats.gpkg   │
            │                                   (+ fields_stats.parquet)  │
            └──────────► 5. Hexagon grid (H3) ◄───────────────┴───────────┘
                                     └─► hex_ndvi.gpkg
   6. Shape metrics (fields) ─► fields_shape.gpkg
   7. Spatial selection (fields within the ROI)
   8. Train / test split (survey points) ─► plots_train.gpkg, plots_test.gpkg

The code in this guide comes from the example script
``docs/examples/zonal_stats_and_vectors.py`` (included below step by step),
which is run by the rsgispy test suite on synthetic data. The scenario is a
Sentinel-2 surface reflectance stack (reflectance x 10,000, no data value 0;
for example from
:func:`~rsgispy.imagecalibration.sensorlvl2data.create_stacked_sen2_esa_l2a_img`),
a layer of non-overlapping field polygons with a unique integer ``field_id``,
survey points with a ``crop`` attribute, and a region of interest polygon.
Run it on your own data with:

.. code-block:: bash

   python docs/examples/zonal_stats_and_vectors.py --s2 s2_sref_10m.tif \
       --fields fields.gpkg --plots survey_plots.gpkg --roi district.gpkg \
       --out-dir outputs

Most vector functions in rsgispy accept a file path and layer name **or** a
:class:`geopandas.GeoDataFrame` (with the layer name ``None``), and return a
GeoDataFrame when no output file is given. The recipes use both forms: files
when the result is worth keeping, GeoDataFrames for intermediate layers.

Step 1 — An index image to summarise
------------------------------------

The recipes summarise NDVI, calculated with
:func:`~rsgispy.imagecalc.calcindices.calc_indices`. ``band_names`` maps the
1-based band indices of the stack to the names the index needs (here ``red``
and ``nir``), ``in_gain`` converts the stored values to reflectance, and the
input no data value (0) becomes ``out_no_data_val`` (-999) in the output, a
value outside the valid NDVI range so it is easy to exclude later. Any other
single or multi-band image can be summarised in the same way.

.. literalinclude:: ../examples/zonal_stats_and_vectors.py
   :language: python
   :start-after: # [step-1-ndvi]
   :end-before: # [end-step-1-ndvi]
   :dedent: 4

Step 2 — Polygon statistics
---------------------------

:func:`~rsgispy.zonalstats.polygon.calc_zonal_band_stats_file` adds one
column per statistic you name (``min_field``, ``max_field``, ``mean_field``,
``stddev_field``, ``sum_field``, ``count_field``, ``mode_field`` and
``median_field``; statistics without a column name are not calculated). Only
pixels whose centre is inside the polygon **and** whose value is between
``min_thres`` and ``max_thres`` (inclusive) are used, so the thresholds are
also how no data is excluded: here the -999 no data pixels are outside
``[-1, 1]``. Polygons with no valid pixels get ``out_no_data_val``.

Without ``out_vec_file`` the input file is **updated in place**, as in
RSGISLib; give an output file to keep the input unchanged. For a
GeoDataFrame use :func:`~rsgispy.zonalstats.polygon.calc_zonal_band_stats`,
which returns the updated GeoDataFrame (see Step 5).

.. literalinclude:: ../examples/zonal_stats_and_vectors.py
   :language: python
   :start-after: # [step-2-polygon-stats]
   :end-before: # [end-step-2-polygon-stats]
   :dedent: 4

The output has the input columns plus ``ndvi_mean``, ``ndvi_std``,
``ndvi_min``, ``ndvi_max`` (NDVI, -1 to 1) and ``ndvi_count`` (number of
valid pixels). Check the counts: a count much lower than the polygon area
divided by the pixel area means the polygon is partly over no data (clouds,
the image edge) and its statistics are less reliable.

**Small polygons.** If polygons are small compared with the pixels (for
example field plots on a 30 m image), some contain no pixel centre and get
``out_no_data_val``. Use
:func:`~rsgispy.zonalstats.polygon.calc_zonal_band_stats_test_poly_pts_file`,
which falls back to the pixel under the polygon centroid, or
:func:`~rsgispy.zonalstats.polygon.calc_zonal_poly_pts_band_stats_file`,
which uses the centroid value only. For per-class pixel counts in each
polygon (e.g. the land cover composition of a district), use
:func:`~rsgispy.zonalstats.polygon.calc_zonal_band_pxl_counts_file`.

Step 3 — Values at points
-------------------------

:func:`~rsgispy.zonalstats.point.ext_point_band_values_file` reads the pixel
under each point, for one band, a list of bands, or all bands
(``img_band=None``). ``out_field`` gives one column name per band (or a
prefix, giving ``<prefix>_<band>``). The column names are converted to lower
case, so the columns here are ``blue``, ``green``, ``red`` and ``nir``.
Points outside the image, or on a pixel outside ``[min_thres, max_thres]``,
get ``out_no_data_val``.

.. literalinclude:: ../examples/zonal_stats_and_vectors.py
   :language: python
   :start-after: # [step-3-point-values]
   :end-before: # [end-step-3-point-values]
   :dedent: 4

Check for points with ``out_no_data_val`` (-1 here) before using the values,
for example as training data: they are usually outside the image or under
cloud. If the points are in a different CRS from the image, the function
raises an exception unless ``reproj_vec=True``.

Step 4 — Statistics from a label raster
---------------------------------------

For many zones, or many bands, rasterise the zones once and use
:func:`~rsgispy.zonalstats.labels.calc_label_zonal_stats`. It reads every
pixel once, accumulating the statistics of every zone at the same time,
instead of reading the pixels under each polygon separately.

1. :func:`~rsgispy.imageutils.rasterise.rasterise_vec_lyr` burns the
   ``field_id`` values into a label raster on the grid of a reference image.
   Use an integer type large enough for the IDs (``rsgispy.TYPE_32UINT``
   here) and keep 0 for "no zone". A pixel is burned when its centre is
   inside the polygon, the same rule as the polygon statistics.
2. :func:`~rsgispy.zonalstats.labels.calc_label_zonal_stats` takes the label
   raster and a list of :class:`~rsgispy.ml.stack.BandSpec` (image, bands,
   band names and a no data value per image; ``min_vld_val`` and
   ``max_vld_val`` give a valid range like the thresholds in Step 2). The
   statistics can be ``count``, ``sum``, ``mean``, ``stddev``, ``min``,
   ``max``, ``median`` and ``mode``, plus any ``percentiles``. The result is
   a :class:`pandas.DataFrame` with one row per zone, sorted by ``id_col``,
   with ``pxl_count`` (all pixels in the zone) and one
   ``<band name>_<statistic>`` column per band and statistic (e.g.
   ``NIR_mean``, ``ndvi_median``); ``out_file`` also writes it as Parquet.
3. :func:`~rsgispy.zonalstats.labels.join_label_stats_to_vec` joins the table
   back onto the polygons by ID (a left join, keeping the polygon order).

.. literalinclude:: ../examples/zonal_stats_and_vectors.py
   :language: python
   :start-after: # [step-4-label-stats]
   :end-before: # [end-step-4-label-stats]
   :dedent: 4

The last lines are a check you can use on your own data: on the same grid
and with the same valid range, the polygon and label methods give the same
means, so the printed difference should be 0 (up to rounding). A large
difference usually means the zones overlap (a pixel can hold only one ID, so
the later polygon wins) or the IDs in the vector layer are not unique.

Medians and percentiles of float images are calculated from a histogram
(``quantile_method='hist'``, error at most half a bin of ``hist_bins``
bins); integer images, or ``quantile_method='exact'``, give exact values.
Categorical images (``data_type=rsgispy.VAR_TYPE_CATEGORICAL`` in the
``BandSpec``) give the most common class and the proportion of each class in
the zone instead.

Step 5 — Hexagon grids as reporting units
-----------------------------------------

Administrative units differ in size and shape, which makes them hard to
compare. A grid of equal-area hexagons is a common alternative for
reporting. :func:`~rsgispy.vector.create.create_hex_grid_polys` creates
`H3 <https://h3geo.org>`_ hexagons over the polygons of a layer, in that
layer's CRS, with the H3 cell index in a ``hex_id`` column (use :func:`~rsgispy.vector.create.create_hex_grid_bbox` for a
bounding box instead). The H3 resolution sets the size: on average about
5.2 km² at resolution 7, 0.74 km² at 8, 0.11 km² at 9 and 0.015 km² at 10.
Choose cells that contain enough pixels (and points) for stable statistics:
a resolution 8 hexagon holds about 7,000 Sentinel-2 10 m pixels.

The hexagons are defined on the sphere and reprojected, so in a projected
CRS their areas vary slightly. A cell is included when its centre is inside
an input polygon, so along the boundary some cells extend outside the
polygon and some slivers of the polygon are not covered. Here they get an
integer ID (``hex_fid``, next to the H3 ``hex_id``) with
:func:`~rsgispy.vector.attrs.add_fid_col`, the mean NDVI with
:func:`~rsgispy.zonalstats.polygon.calc_zonal_band_stats` (the GeoDataFrame
version of Step 2) and the number of survey points in each cell with
:func:`~rsgispy.vector.stats.count_pt_intersects`. For a large region with
many hexagons, use Step 4 instead of polygon statistics.

.. literalinclude:: ../examples/zonal_stats_and_vectors.py
   :language: python
   :start-after: # [step-5-hex-grid]
   :end-before: # [end-step-5-hex-grid]
   :dedent: 4

Step 6 — Shape metrics
----------------------

:func:`~rsgispy.vector.stats.calc_poly_shape_metrics` adds polygon shape
metrics as columns (prefixed with ``col_prefix``, ``shp_`` here): size (``area``, ``perimeter``), and unitless measures of
form such as ``compactness`` (Polsby–Popper, 1 for a circle, π/4 ≈ 0.785 for
a square), ``elongation`` (length / width of the minimum rotated rectangle,
1 for a square), ``solidity`` (area / convex hull area, 1 for a convex
shape), ``rectangularity``, ``roundness``, ``fractal_dim`` and others; see
:func:`~rsgispy.vector.stats.calc_shape_metrics_geoseries` for the full list
and formulas. Areas and lengths are in the units of the CRS (m² and m here);
for a layer in latitude/longitude, set ``proj_epsg`` (or ``'utm'``) to
calculate them in a projected CRS.

.. literalinclude:: ../examples/zonal_stats_and_vectors.py
   :language: python
   :start-after: # [step-6-shape-metrics]
   :end-before: # [end-step-6-shape-metrics]
   :dedent: 4

Shape metrics are useful features for classifying objects (fields are
compact, roads and rivers are elongated) and for finding digitising or
segmentation errors (very low solidity, many vertices).

Step 7 — Spatial selection
--------------------------

:func:`~rsgispy.vector.selection.spatial_select` keeps the features of one
layer that match the features of another with a GeoPandas spatial
predicate: ``'intersects'`` (the default; any overlap), ``'within'``
(completely inside), ``'contains'``, ``'touches'``, etc. Here, the fields
completely inside the region of interest are kept; with ``'intersects'``,
fields crossing the boundary would be kept too. If the two layers are in
different CRSs, the region of interest is reprojected to the CRS of the
input layer (set ``vec_epsg`` / ``vec_roi_epsg`` for layers without one).

.. literalinclude:: ../examples/zonal_stats_and_vectors.py
   :language: python
   :start-after: # [step-7-spatial-select]
   :end-before: # [end-step-7-spatial-select]
   :dedent: 4

For selection by attribute use
:func:`~rsgispy.vector.selection.subset_by_attribute`; to clip geometries to
the region (rather than select whole features) use
:func:`~rsgispy.vector.geoms.clip_vec_lyr`.

Step 8 — Training and testing samples
-------------------------------------

:func:`~rsgispy.vector.selection.create_train_test_smpls` splits a layer at
random into two disjoint layers, with ``prop_test`` of the features in the
testing layer, and numbers the input features (``fid``, from 1) so each
sample can be traced back to the input. In a GeoPackage this becomes the
layer's feature ID, which :func:`geopandas.read_file` returns only with
``fid_as_index=True``. Set ``rnd_seed`` so the split can be repeated. To take a fixed
number of samples instead of a proportion, use
:func:`~rsgispy.vector.selection.split_vec_lyr_random_subset`.

.. literalinclude:: ../examples/zonal_stats_and_vectors.py
   :language: python
   :start-after: # [step-8-train-test]
   :end-before: # [end-step-8-train-test]
   :dedent: 4

The split is purely random: it is not stratified by class, and nearby points
can end up in both sets. Neighbouring samples are usually similar (spatial
autocorrelation), so a random split tends to over-estimate accuracy. Check
that every class is in both sets, and consider a spatial split, for example
:class:`~rsgispy.ml.cv.SpatialBlockKFold` or
:class:`~rsgispy.ml.cv.SpatialBufferCV`, or the grouped splits used in the
:doc:`object-based guide <segment_ml_workflow>`.

Performance and large datasets
------------------------------

* **Polygon or label statistics** — polygon statistics read the pixels of
  each polygon and band separately, so the time grows with the number of
  polygons times the number of bands; label statistics read each pixel once.
  In the benchmark reported in
  :func:`~rsgispy.zonalstats.labels.calc_label_zonal_stats` (a 2000 x 2000
  pixel image, 40,000 zones, 4 bands) the label method took 0.24 s against
  22 s for polygon statistics, and the gap grows with the number of zones.
  For a handful of large polygons (e.g. a few districts) the difference is
  small and polygon statistics are simpler; the label raster also costs a
  rasterisation step and some disk space.
* **Memory** — label statistics need memory per zone, not per pixel;
  ``block_size`` sets the size of the blocks read from the images. Very large
  or sparse IDs are handled without a lookup table as large as the largest
  ID.
* **Points** — point extraction reads one pixel per point and band, so it is
  fast even for many points. Sort or group very large point sets by area if
  they cover many separate images.
* **Formats** — GeoPackage is used here because it is widely supported;
  GeoParquet (``.parquet``) is faster to read and write for large layers and
  can be used with the functions in this guide.
* **Large rasters** — the images do not need to be loaded into memory. Use a
  VRT (:func:`~rsgispy.imageutils.vrt.create_mosaic_images_vrt`) to treat
  many tiles as one image.

Common problems
---------------

* **CRS mismatch** — polygon and point statistics raise an exception if the
  vector layer and the image have different CRSs. Reproject the vectors with
  :func:`~rsgispy.vector.io.reproj_vec_lyr` (or use ``reproj_vec=True`` for
  points). Layers without a CRS can be given one with ``vec_def_epsg``.
* **All statistics equal to out_no_data_val** — the thresholds exclude every
  pixel (for example ``min_thres=0, max_thres=1`` on reflectance stored as
  integers x 10,000), or the polygons are smaller than a pixel (see Step 2).
* **No data counted as data** — the thresholds, not the image no data value,
  decide which pixels are used in polygon and point statistics; make sure the
  no data value is outside ``[min_thres, max_thres]``.
* **Input file changed** — without ``out_vec_file``, the ``*_file`` zonal
  functions add the columns to the input file.
* **Label statistics differ from polygon statistics** — the zones overlap or
  the IDs are not unique (see Step 4), or the label raster is not on the
  image's grid; the images must have the same size, transform and CRS as the
  label raster (resample with
  :func:`~rsgispy.imageutils.warp.resample_img_to_match`).
* **ImportError for h3 or pyarrow** — install the ``vector`` extra (see
  Installation).

Related guides
--------------

* :doc:`hydrology_workflow`
* :doc:`multi_criteria_analysis`
* :doc:`segment_ml_workflow`
