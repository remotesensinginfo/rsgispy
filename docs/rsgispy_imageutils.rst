rsgispy.imageutils
==================

The :mod:`rsgispy.imageutils` subpackage provides thematic modules for raster image utilities, including configuring GDAL/rasterio creation options (`envopts`), inspecting/modifying metadata (`metadata`), colour tables (`colourtable`), statistics and pyramids (`stats`), creating blank/copy images (`blank`), resampling/reprojection (`warp`), mosaicking (`mosaic`), tiling (`tiling`), masking (`masking`), spatial subsetting (`subsetting`), GDAL VRT generation (`vrt`), and spatial image LUTs (`imagelut`).

Configuring Output Creation Options
-----------------------------------

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
----------------------------------------

.. code-block:: python

   import rsgispy.imageutils.stats

   # Populate statistics and overview pyramids on a continuous raster
   rsgispy.imageutils.stats.pop_img_stats("continuous_img.tif", no_data_val=-999.0)

   # Populate thematic statistics, colour table, and overview pyramids on a uint8 raster
   rsgispy.imageutils.stats.pop_thmt_img_stats("land_cover_cls_img.tif", ignore_zero=True)

API Reference
-------------

.. automodule:: rsgispy.imageutils.blank
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: rsgispy.imageutils.colourtable
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: rsgispy.imageutils.envopts
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: rsgispy.imageutils.imagelut
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: rsgispy.imageutils.masking
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: rsgispy.imageutils.metadata
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: rsgispy.imageutils.mosaic
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: rsgispy.imageutils.stats
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: rsgispy.imageutils.subsetting
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: rsgispy.imageutils.tiling
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: rsgispy.imageutils.vrt
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: rsgispy.imageutils.warp
   :members:
   :undoc-members:
   :show-inheritance:
