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

   # Configure all GTIFF outputs to use Deflate compression and tiling
   rsgispy.imageutils.envopts.set_env_vars_deflate_gtiff_outs(bigtiff=False)

   band_defns = [
       rsgispy.imagecalc.maths.BandDefn(band_name="b1", input_img="img.kea", img_band=1),
       rsgispy.imagecalc.maths.BandDefn(band_name="b2", input_img="img.kea", img_band=2),
   ]
   rsgispy.imagecalc.maths.band_maths(
       output_img="out.tif",
       exp="b1*b2",
       gdalformat="GTIFF",
       datatype=rsgispy.TYPE_32FLOAT,
       band_defs=band_defns,
   )

Populating Image Statistics and Pyramids
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import rsgispy.imageutils.stats

   # Populate statistics and overview pyramids on a continuous raster
   rsgispy.imageutils.stats.pop_img_stats("continuous_img.tif", no_data_val=-999.0)

   # Populate thematic statistics, colour table, and overview pyramids on a uint8 raster
   rsgispy.imageutils.stats.pop_thmt_img_stats("land_cover_cls_img.tif", ignore_zero=True)

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

Band, Spectral & Array Operations
---------------------------------

* :doc:`imageutils/arrays` — NumPy array/raster conversion and pixel/column extraction.
* :doc:`imageutils/bands` — Band selection, stacking, and pan-sharpening.
* :doc:`imageutils/spectral` — Spectral smoothing, outlier masking, polyfill, and whitening.
* :doc:`imageutils/sampling` — Random and mask-based pixel sampling.

.. toctree::
   :maxdepth: 1

   imageutils/arrays
   imageutils/bands
   imageutils/spectral
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
