rsgispy.vector
==============

The :mod:`rsgispy.vector` subpackage consolidates vector I/O, geometry creation,
geometric transformations, attribute table management, spatial statistics, and
selection into six focused submodules.  Every function accepts either a vector
file path (``str`` / :class:`os.PathLike`) or an in-memory
:class:`geopandas.GeoDataFrame`.

Examples
--------

Inspect metadata, reproject, and select features using :class:`~rsgispy.BBox`:

.. code-block:: python

   import rsgispy.vector.io
   import rsgispy.vector.selection

   bbox = rsgispy.vector.io.get_vec_lyr_bbox("polys.gpkg", "polys")
   rsgispy.vector.io.reproj_vec_lyr(
       "polys.gpkg", "polys", "polys_wgs84.gpkg", "polys", out_epsg=4326
   )
   sub_gdf = rsgispy.vector.selection.spatial_select_bbox(
       "polys_wgs84.gpkg", "polys", bbox.reproject(4326)
   )

Create footprints, enrich attributes, and compute spatial metrics in memory:

.. code-block:: python

   import rsgispy.vector.attrs
   import rsgispy.vector.create
   import rsgispy.vector.stats

   fp_gdf = rsgispy.vector.create.extract_img_footprint("scene.tif")
   fp_gdf = rsgispy.vector.attrs.add_fid_col(fp_gdf, None, out_col="fid")
   fp_gdf = rsgispy.vector.stats.calc_vec_area(fp_gdf, None, "area_m2")

Calculate polygon shape metrics (compactness, solidity, elongation,
orientation, etc.) in a projected CRS, e.g. for segment-based classification:

.. code-block:: python

   import rsgispy.vector.stats

   seg_gdf = rsgispy.vector.stats.calc_poly_shape_metrics(
       "segments.parquet",
       None,
       metrics=["area", "compactness", "solidity", "elongation", "orientation"],
       col_prefix="shp_",
       proj_epsg="utm",
   )

Calculate multiple new attribute columns in one pass using comma-separated
``pymu-eval`` expressions with :func:`~rsgispy.vector.attrs.vector_maths`
(each top-level comma-separated sub-expression maps positionally to the
corresponding entry in ``out_col``, while commas inside function calls such as
``min(blue, green, red)`` remain function arguments, and new intermediate
variables (e.g., ``ndvi = ...``) can be defined and reused by later outputs):

.. code-block:: python

   import numpy as np
   import rsgispy.vector.attrs

   # Multi-output expression defining and reusing intermediate `ndvi`
   exp = """
       ndvi = (nir + red) != 0 ? (nir - red) / (nir + red) : -9999,
       avg(blue, green, red),
       ndvi >= 0.5 ? 1 : 0
   """
   out_gdf = rsgispy.vector.attrs.vector_maths(
       "samples.gpkg",
       "samples",
       out_col=["ndvi", "vis_mean", "is_veg"],
       exp=exp,
       out_dtype=[np.float32, np.float32, int],
   )

   # Alias columns with spaces/hyphens and reuse scaled values across outputs
   out_gdf = rsgispy.vector.attrs.vector_maths(
       "samples.gpkg",
       "samples",
       out_col=["nir_refl", "red_refl", "ndvi"],
       exp="nir = nir * 0.0001, red = red * 0.0001, (nir - red) / (nir + red)",
       vec_cols={"nir": "Band 4", "red": "Band 3"},
   )

Polygonising large rasters
--------------------------

:func:`~rsgispy.vector.create.polygonise_raster_to_vec_lyr` holds the whole
band, every polygon and the output GeoDataFrame in memory and uses one core,
which is slow and can run out of memory for rasters producing millions of
polygons, such as segmentations or fine-scale classifications. For these use
:func:`~rsgispy.vector.create.polygonise_raster_tiled`:

* the raster is processed in tiles (``tile_size``), so memory use does not
  depend on the image size;
* tiles are processed in parallel by ``n_jobs`` worker processes;
* the ``'numba'`` backend (default when ``numba`` is installed) labels the
  connected regions and traces their boundaries with a compiled kernel and
  writes WKB directly, without creating a Python object per polygon; the
  ``'rasterio'`` backend runs GDAL's polygonisation on each tile;
* the output is streamed to disk in chunks (GeoParquet with compression by
  default, or any OGR vector format);
* the polygons are the same as those from
  :func:`~rsgispy.vector.create.polygonise_raster_to_vec_lyr` (same coverage,
  values, holes and connectivity) whatever the tile size and number of cores,
  and the output is identical for any ``n_jobs``.

Polygons that do not touch an internal tile edge are written as soon as their
tile is finished. Pieces of polygons cut by tile edges are merged at the end
(pieces of the same value sharing an edge, or touching at a corner with
``use_8_conn=True``, are unioned in exact pixel coordinates), so only these
seam pieces are held in memory: their number grows with the total length of
the tile edges, not with the image area. ``sort_by_value`` and
``dissolve_by_value`` use an out-of-core bucket sort, so they also work for
outputs larger than memory.

.. code-block:: python

   import rsgispy.vector.create

   # One (Multi)Polygon per segment ID, sorted by ID, with pixel counts,
   # using 8 worker processes.
   rsgispy.vector.create.polygonise_raster_tiled(
       "segments.kea",
       "segments.parquet",
       pxl_val_fieldname="seg_id",
       no_data_val=0,
       dissolve_by_value=True,
       sort_by_value=True,
       inc_pxl_count=True,
       n_jobs=8,
   )

   # Every connected region of a classification, in memory.
   gdf = rsgispy.vector.create.polygonise_raster_tiled(
       "classes.tif", None, use_8_conn=True, tile_size=1024
   )

Output geometries are always valid. With ``use_8_conn=True``, regions whose
parts touch only diagonally are written as MultiPolygons with the parts
touching at the pixel corner (GDAL writes them as a single self-touching ring,
which is invalid).

.. _polygonise-benchmark:

Performance
~~~~~~~~~~~

Measured with ``benchmarks/polygonise_benchmark.py`` on segmentation-like
label rasters (jittered Voronoi segments, about ten polygons per segment),
writing GeoParquet, on an Apple M2 Max laptop (8 performance + 4 efficiency
cores, 96 GB RAM). Memory is the peak resident memory of the main process and,
with ``n_jobs > 1``, of each worker process.

.. list-table::
   :header-rows: 1
   :widths: 30 12 12 12 34

   * - Function / settings
     - Cores
     - Time
     - Speed-up
     - Peak memory
   * - **3000 x 3000 pixels, 1.45 million polygons**
     -
     -
     -
     -
   * - ``polygonise_raster_to_vec_lyr`` (+ write)
     - 1
     - 26.7 s
     - 1x
     - 3.9 GB
   * - tiled, ``'numba'``, ``tile_size=1024``
     - 1
     - 2.0 s
     - 13x
     - 0.9 GB
   * - tiled, ``'rasterio'``, ``tile_size=1024``
     - 1 / 8
     - 10.0 / 3.9 s
     - 2.7x / 6.8x
     - 1.0 GB / 0.7 GB + 0.7 GB per worker
   * - **10,000 x 10,000 pixels, 16.2 million polygons**
     -
     -
     -
     -
   * - ``polygonise_raster_to_vec_lyr`` (+ write)
     - 1
     - 311 s
     - 1x
     - 38 GB
   * - tiled, ``'numba'``, ``tile_size=1024``
     - 1 / 4
     - 20.9 / 9.6 s
     - 15x / 32x
     - 1.3 GB / 1.2 GB + 0.8 GB per worker
   * - tiled, ``'numba'``, ``tile_size=2048``
     - 1 / 4
     - 18.5 / 8.1 s
     - 17x / 38x
     - 2.9 GB / 3.4 GB + 1.9 GB per worker
   * - tiled, ``'rasterio'``, ``tile_size=1024``
     - 1 / 8
     - 107 / 17.0 s
     - 2.9x / 18x
     - 1.3 GB / 1.0 GB + 0.9 GB per worker
   * - tiled, ``'numba'``, ``dissolve_by_value`` + ``sort_by_value``
       (1.67 million segments)
     - 8
     - 20.4 s
     -
     - 4.9 GB + 0.7 GB per worker

Most of the time of the current function is spent converting GDAL's output
into shapely geometries one polygon at a time (296 of 311 s for the larger
raster). The ``'numba'`` backend avoids this, and its single-core time is
dominated by the boundary tracing; with 4 or more workers the transfer of
results between processes, the writing of the output and the seam merging
in the main process limit further gains (8 workers were no faster than 4 on
this machine). The ``'rasterio'`` backend gains more from extra cores.

Recommendations: ``tile_size`` of 1024 (less memory per worker) to 2048, and
``n_jobs`` up to the number of performance cores for rasters of tens of
millions of pixels or more; worker processes take 1-2 s to start, so for
small rasters ``n_jobs=1`` is fastest.

Submodules
----------

.. toctree::
   :maxdepth: 2

   vector/io
   vector/create
   vector/geoms
   vector/attrs
   vector/stats
   vector/selection
