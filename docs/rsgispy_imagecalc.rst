rsgispy.imagecalc
=================

The :mod:`rsgispy.imagecalc` subpackage provides modules for performing mathematical calculations (`maths`), summary statistics (`stats`), percentiles (`percentiles`), and windowed sampling (`calc_pt_win_smpls`) on raster image bands using `pymu-eval <https://github.com/remotesensinginfo/pymu-eval>`_ (muParser syntax) and `rasterio <https://rasterio.readthedocs.io/>`_.

Examples
--------

Calculating a Band Product
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import rsgispy
   import rsgispy.imagecalc.maths

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

Conditional (Ternary) Expressions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import rsgispy
   import rsgispy.imagecalc.maths

   band_defns = [
       rsgispy.imagecalc.maths.BandDefn(band_name="b1", input_img="img1.tif", img_band=1),
       rsgispy.imagecalc.maths.BandDefn(band_name="b2", input_img="img2.tif", img_band=1),
       rsgispy.imagecalc.maths.BandDefn(band_name="b3", input_img="img3.tif", img_band=2),
   ]
   rsgispy.imagecalc.maths.band_maths(
       output_img="out_mask.tif",
       exp="(b1==1) || (b2==1) || (b3==1) ? 1 : 0",
       gdalformat="GTIFF",
       datatype=rsgispy.TYPE_8UINT,
       band_defs=band_defns,
   )

API Reference
-------------

.. automodule:: rsgispy.imagecalc.maths
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: rsgispy.imagecalc.stats
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: rsgispy.imagecalc.percentiles
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: rsgispy.imagecalc.comparison
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: rsgispy.imagecalc.encoding
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: rsgispy.imagecalc.rescale
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: rsgispy.imagecalc.distance
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: rsgispy.imagecalc.fill
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: rsgispy.imagecalc.calc_pt_win_smpls
   :members:
   :undoc-members:
   :show-inheritance:
