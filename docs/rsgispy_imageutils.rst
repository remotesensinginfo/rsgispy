rsgispy.imageutils
==================

The :mod:`rsgispy.imageutils` subpackage provides thematic modules for raster
image utilities using ``rasterio``, ``numpy``, and ``pyproj``.

Examples
--------

Configuring Output Creation Options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import rsgispy
   import rsgispy.imagecalc.maths
   import rsgispy.imageutils.envopts

   # GTIFF outputs are tiled and DEFLATE-compressed by default; use ZSTD
   # instead (reset_img_creation_opts() restores the default).
   rsgispy.imageutils.envopts.set_env_vars_zstd_gtiff_outs(bigtiff=False)

   band_defns = [
       rsgispy.imagecalc.maths.BandDefn(band_name="b1", input_img="img.tif", img_band=1),
       rsgispy.imagecalc.maths.BandDefn(band_name="b2", input_img="img.tif", img_band=2),
   ]
   rsgispy.imagecalc.maths.band_maths(
       output_img="out.tif",
       exp="b1*b2",
       band_defs=band_defns,
       gdalformat="GTIFF",
       datatype=rsgispy.TYPE_32FLOAT,
   )

Populating Image Statistics and Pyramids
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import rsgispy.imageutils.stats

   # Populate statistics and overview pyramids on a continuous raster
   rsgispy.imageutils.stats.pop_img_stats("continuous_img.tif", no_data_val=-999.0)

   # Populate thematic statistics, colour table, and overview pyramids on a uint8 raster
   rsgispy.imageutils.stats.pop_thmt_img_stats("land_cover_cls_img.tif", ignore_zero=True)

   # Generate tiled overview pyramids for a large raster across multiple cores
   rsgispy.imageutils.stats.build_img_pyramids_multi_core("large_img.tif", tile_size=1024, n_cores=4)

Chaining
~~~~~~~~

The masking functions of :mod:`rsgispy.imageutils.masking` have tile-graph
node builders (``mask_img_node``, ``gen_finite_mask_node``,
``gen_valid_mask_node``, ``gen_img_edge_mask_node``,
``mask_all_band_zero_vals_node`` and ``create_valid_mask_node``) for use in
an in-memory :class:`~rsgispy.imageprocess.TileGraph`; the file-based
functions are wrappers around them and give the same result. The mask of
``mask_img_node`` can be any node, e.g. one computed in the same graph:

.. code-block:: python

   import rsgispy
   from rsgispy.imageprocess import BandMathOp, DiskSink, RasterSource, TileGraph
   from rsgispy.imageutils.masking import mask_img_node

   img = RasterSource("s2.tif")
   swir = RasterSource("s2.tif", bands=5)
   water = BandMathOp({"swir": swir}, "swir < 500 ? 1 : 0")
   masked = mask_img_node(img, water, 0, 1, no_data_val=0)
   TileGraph(DiskSink(masked, "s2_land.tif", datatype=rsgispy.TYPE_16UINT)).run()

See :doc:`rsgispy_imageprocess` for the list of chainable operations.

Configuration, Metadata & Validation
------------------------------------

* :doc:`imageutils/envopts` — GDAL/rasterio creation options and COG/GTIFF environment presets.
* :doc:`imageutils/metadata` — Bounding boxes, resolution, CRS/EPSG/UTM, GCPs, and metadata tags.
* :doc:`imageutils/checks` — Image validity, resolution matching, projection, and overlap checks.
* :doc:`imageutils/colourtable` — Thematic flags, colour interpretation, and colour tables.
* :doc:`imageutils/stats` — Raster statistics and overview pyramid generation.

.. toctree::
   :maxdepth: 1

   imageutils/envopts
   imageutils/metadata
   imageutils/checks
   imageutils/colourtable
   imageutils/stats

Raster Creation, Copying & VRT
------------------------------

* :doc:`imageutils/blank` — Blank raster creation and copying from reference images/bboxes.
* :doc:`imageutils/rasterise` — Vector rasterisation, look-up score rasters, and distance zone classification.
* :doc:`imageutils/vrt` — GDAL VRT band-stack, mosaic, and band-subset generation.
* :doc:`imageutils/imagelut` — Spatial vector LUT creation and querying across raster files.

.. toctree::
   :maxdepth: 1

   imageutils/blank
   imageutils/rasterise
   imageutils/vrt
   imageutils/imagelut

Band, Sampling & Array Operations
---------------------------------

* :doc:`imageutils/arrays` — NumPy array/raster conversion and pixel/column extraction.
* :doc:`imageutils/bands` — Band selection, stacking, and pan-sharpening.
* :doc:`imageutils/sampling` — Random and mask-based pixel sampling.

.. toctree::
   :maxdepth: 1

   imageutils/arrays
   imageutils/bands
   imageutils/sampling

Spatial Manipulation, Warping & Mosaicking
------------------------------------------

* :doc:`imageutils/subsetting` — Subsetting rasters by bounding box or reference image.
* :doc:`imageutils/tiling` — Splitting rasters into regular spatial tiles.
* :doc:`imageutils/mosaic` — Image mosaicking, inclusion, band combining, and binary masks.
* :doc:`imageutils/masking` — Masking, finite/valid/edge mask generation.
* :doc:`imageutils/warp` — Resampling, reprojection, ``gdal_warp``, and ``gdal_translate``.
* :doc:`imageutils/spatial` — WGS84 pixel area/size/locations and scattered-point gridding.

.. toctree::
   :maxdepth: 1

   imageutils/subsetting
   imageutils/tiling
   imageutils/mosaic
   imageutils/masking
   imageutils/warp
   imageutils/spatial
