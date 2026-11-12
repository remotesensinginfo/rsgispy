rsgispy.imagecalc
=================

The :mod:`rsgispy.imagecalc` module provides functions for performing mathematical calculations on raster image bands using `pymu-eval <https://github.com/remotesensinginfo/pymu-eval>`_ (muParser syntax) and `rasterio <https://rasterio.readthedocs.io/>`_.

Examples
--------

Calculating a Band Product
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import rsgispy
   import rsgispy.imagecalc

   band_defns = list()
   band_defns.append(
       rsgispy.imagecalc.BandDefn(band_name="b1", input_img="img.kea", img_band=1)
   )
   band_defns.append(
       rsgispy.imagecalc.BandDefn(band_name="b2", input_img="img.kea", img_band=2)
   )
   rsgispy.imagecalc.band_maths(
       output_img="out.tif",
       exp="b1*b2",
       gdalformat="GTIFF",
       datatype=rsgispy.TYPE_32FLOAT,
       band_defs=band_defns,
   )

Conditional (Ternary) Expressions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import rsgispy
   import rsgispy.imagecalc

   band_defns = [
       rsgispy.imagecalc.BandDefn(band_name="b1", input_img="img1.tif", img_band=1),
       rsgispy.imagecalc.BandDefn(band_name="b2", input_img="img2.tif", img_band=1),
       rsgispy.imagecalc.BandDefn(band_name="b3", input_img="img3.tif", img_band=2),
   ]
   rsgispy.imagecalc.band_maths(
       output_img="out_mask.tif",
       exp="(b1==1) || (b2==1) || (b3==1) ? 1 : 0",
       gdalformat="GTIFF",
       datatype=rsgispy.TYPE_8UINT,
       band_defs=band_defns,
   )

API Reference
-------------

.. automodule:: rsgispy.imagecalc
   :members:
   :undoc-members:
   :show-inheritance:
