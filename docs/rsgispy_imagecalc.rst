rsgispy.imagecalc
=================

The :mod:`rsgispy.imagecalc` subpackage provides modules for performing
mathematical calculations, summary statistics, percentiles, image comparison,
encoding, rescaling, distance/buffering, KNN gap filling, and windowed sampling
on raster image bands using `pymu-eval <https://github.com/remotesensinginfo/pymu-eval>`_
and `rasterio <https://rasterio.readthedocs.io/>`_.

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

Band Mathematics & Value Rescaling
----------------------------------

* :doc:`imagecalc/maths` — muParser band/image expressions, proportions, and area under curve.
* :doc:`imagecalc/rescale` — Linear rescaling and gain/offset/nodata transformations.

.. toctree::
   :maxdepth: 1

   imagecalc/maths
   imagecalc/rescale

Statistics, Percentiles & Comparison
------------------------------------

* :doc:`imagecalc/stats` — Basic band statistics, unique values, multi-image summaries, and mask probabilities.
* :doc:`imagecalc/percentiles` — Band percentiles and pixel-wise cross-band/cross-image percentiles.
* :doc:`imagecalc/comparison` — Image difference, equality tests, correlation, mutual info, and EMD.

.. toctree::
   :maxdepth: 1

   imagecalc/stats
   imagecalc/percentiles
   imagecalc/comparison

Encoding, Distance, Filling & Sampling
--------------------------------------

* :doc:`imagecalc/encoding` — Integer recoding and one-hot category encoding.
* :doc:`imagecalc/distance` — Distance-to-pixel-values (full and tiled) and morphological buffering.
* :doc:`imagecalc/fill` — K-nearest-neighbour region filling.
* :doc:`imagecalc/calc_pt_win_smpls` — Windowed summary statistics around point samples.

.. toctree::
   :maxdepth: 1

   imagecalc/encoding
   imagecalc/distance
   imagecalc/fill
   imagecalc/calc_pt_win_smpls
