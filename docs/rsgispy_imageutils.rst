rsgispy.imageutils
==================

The :mod:`rsgispy.imageutils` module provides general image utility functions, including helpers for configuring GDAL/rasterio image creation options (such as compression, tiling, interleaving, and BigTIFF settings) via environment variables.

Configuring Output Creation Options
-----------------------------------

.. code-block:: python

   import rsgispy
   import rsgispy.imagecalc
   import rsgispy.imageutils

   # Configure all GTIFF outputs to use Deflate compression and tiling
   rsgispy.imageutils.set_env_vars_deflate_gtiff_outs(bigtiff=False)

   band_defns = [
       rsgispy.imagecalc.BandDefn(band_name="b1", input_img="img.kea", img_band=1),
       rsgispy.imagecalc.BandDefn(band_name="b2", input_img="img.kea", img_band=2),
   ]
   rsgispy.imagecalc.band_maths(
       output_img="out.tif",
       exp="b1*b2",
       gdalformat="GTIFF",
       datatype=rsgispy.TYPE_32FLOAT,
       band_defs=band_defns,
   )

Populating Image Statistics and Pyramids
----------------------------------------

.. code-block:: python

   import rsgispy.imageutils

   # Populate statistics and overview pyramids on a continuous raster
   rsgispy.imageutils.pop_img_stats("continuous_img.tif", no_data_val=-999.0)

   # Populate thematic statistics, colour table, and overview pyramids on a uint8 raster
   rsgispy.imageutils.pop_thmt_img_stats("land_cover_cls_img.tif", ignore_zero=True)

API Reference
-------------

.. automodule:: rsgispy.imageutils
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: rsgispy.imageutils.imagelut
   :members:
   :undoc-members:
   :show-inheritance:
