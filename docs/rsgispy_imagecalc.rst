rsgispy.imagecalc
=================

The :mod:`rsgispy.imagecalc` subpackage provides modules for performing
mathematical calculations, summary statistics, percentiles, image comparison,
encoding, rescaling, distance/buffering, region filling, and windowed sampling
on raster image bands using `pymu-eval <https://github.com/remotesensinginfo/pymu-eval>`_
and `rasterio <https://rasterio.readthedocs.io/>`_.

.. seealso::

   Guides: :doc:`guides/multi_criteria_analysis`, :doc:`guides/large_rasters`

Examples
--------

Calculating a Band Product
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import rsgispy
   import rsgispy.imagecalc.maths

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
       band_defs=band_defns,
       gdalformat="GTIFF",
       datatype=rsgispy.TYPE_8UINT,
   )

Multi-Output Expressions (Multiple Output Bands)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``pymu-eval`` supports comma-separated sub-expressions (``expr1, expr2, ...``)
that evaluate to multiple outputs in a single pass. In
:func:`~rsgispy.imagecalc.maths.band_maths` and
:func:`~rsgispy.imagecalc.maths.image_band_math`, each top-level comma-separated
sub-expression is written to its corresponding 1-based output band. Commas
inside function calls (such as ``min(b1, b2, b3)``) stay function argument
separators, and expressions may span multiple lines, include ternary
conditionals, re-assign an input variable, or define arbitrary new
intermediate variables (e.g., ``ndvi = (nir - red) / (nir + red + 1e-6)``) for
reuse in later sub-expressions:

.. code-block:: python

   import rsgispy
   import rsgispy.imagecalc.maths

   band_defns = [
       rsgispy.imagecalc.maths.BandDefn(band_name="red", input_img="s2.tif", img_band=3),
       rsgispy.imagecalc.maths.BandDefn(band_name="nir", input_img="s2.tif", img_band=4),
       rsgispy.imagecalc.maths.BandDefn(band_name="swir1", input_img="s2.tif", img_band=5),
   ]

   # Define a new intermediate variable (`ndvi`) and reuse it in a later band
   indices_exp = """
       ndvi = (nir - red) / (nir + red + 1e-6),
       (nir - swir1) / (nir + swir1 + 1e-6),
       ndvi >= 0.5 ? 1 : 0
   """
   rsgispy.imagecalc.maths.band_maths(
       output_img="indices.tif",
       exp=indices_exp,
       band_defs=band_defns,
       band_names=["NDVI", "NDMI", "VegMask"],
       gdalformat="GTIFF",
       datatype=rsgispy.TYPE_32FLOAT,
   )

   # Per-pixel min, max, and mean across bands (inner commas are function args;
   # top-level commas separate the three output bands)
   rsgispy.imagecalc.maths.band_maths(
       output_img="band_summary.tif",
       exp="min(red, nir, swir1), max(red, nir, swir1), avg(red, nir, swir1)",
       band_defs=band_defns,
       band_names=["min_refl", "max_refl", "mean_refl"],
   )

   # Rescale inputs in-place and reuse the scaled values in a third output band
   rsgispy.imagecalc.maths.band_maths(
       output_img="scaled_and_ndvi.tif",
       exp="nir = nir * 0.0001, red = red * 0.0001, (nir - red) / (nir + red)",
       band_defs=band_defns[:2],
       band_names=["nir_refl", "red_refl", "NDVI"],
   )

Output No-Data Values and Statistics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The input bands are read without masking, so the expression must test for
input no-data values and write the output no-data value itself. Passing the
same value as ``no_data_val`` records it as the no-data value of every output
band, and ``calc_stats=True`` calculates statistics and pyramids that exclude
it (both are also accepted by
:func:`~rsgispy.imagecalc.maths.image_math` and
:func:`~rsgispy.imagecalc.maths.image_band_math`):

.. code-block:: python

   import rsgispy
   import rsgispy.imagecalc.maths

   band_defns = [
       rsgispy.imagecalc.maths.BandDefn(band_name="red", input_img="s2.tif", img_band=3),
       rsgispy.imagecalc.maths.BandDefn(band_name="nir", input_img="s2.tif", img_band=4),
   ]
   rsgispy.imagecalc.maths.band_maths(
       output_img="ndvi.tif",
       exp="(red == 0 || nir == 0) ? -9999 : (nir - red) / (nir + red)",
       band_defs=band_defns,
       band_names=["NDVI"],
       no_data_val=-9999,
       calc_stats=True,
   )

Image Histograms (Continuous & Categorical Integer)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`~rsgispy.imagecalc.stats.get_img_histogram` computes streaming band
histograms supporting fixed bin counts (``n_bins``), fixed bin widths
(``bin_size``), automatic Scott's-rule binning, or single-integer bins
(``integer_bins=True``) for categorical/thematic rasters:

.. code-block:: python

   import rsgispy.imagecalc.stats as imgstats

   # Continuous raster with fixed bin width over a custom value range
   ndvi_hist = imgstats.get_img_histogram(
       input_img="ndvi.tif",
       img_band=1,
       bin_size=0.1,
       val_range=(-1.0, 1.0),
   )
   print(ndvi_hist["bin_edges"], ndvi_hist["counts"], ndvi_hist["proportions"])

   # Categorical integer raster with one bin per integer class ID
   cls_hist = imgstats.get_img_histogram(
       input_img="landcover.tif",
       img_band=1,
       integer_bins=True,
   )
   for bin_info in cls_hist["bins"]:
       print(bin_info["value"], bin_info["count"], bin_info["proportion"])

Chaining
--------

The pixel-wise functions of this subpackage can also be used as nodes of an
in-memory :class:`~rsgispy.imageprocess.TileGraph`, chained with filters,
masks and other operations in one pass without intermediate files. Each has a
node builder next to its file-based function, named after it with ``_node``
(the file-based functions are wrappers around them and give the same
result):

* :mod:`~rsgispy.imagecalc.calcindices`: ``calc_indices_node``,
  ``calc_sar_indices_node`` (one band per index; ``Brightness_Scaled``
  bounds and the ``DPSVI`` ``vv_max`` must be given in ``index_params``);
* :mod:`~rsgispy.imagecalc.rescale`: ``rescale_img_pxl_vals_node``,
  ``calc_img_rescale_node``, ``img_norm_params_node`` (applies parameters
  from :func:`~rsgispy.imagecalc.rescale.calc_img_norm_params`);
* :mod:`~rsgispy.imagecalc.encoding`: ``recode_int_raster_node``,
  ``one_hot_category_encoding_node`` (categories given);
* :mod:`~rsgispy.imagecalc.percentiles`: ``calc_imgs_pxl_percentiles_node``,
  ``calc_img_band_pxl_percentiles_node`` (N inputs, e.g. dates).

.. code-block:: python

   from rsgispy.imageprocess import DiskSink, RasterSource, TileGraph
   from rsgispy.imagefilter.sar import lee_filter_node
   from rsgispy.imagecalc.calcindices import calc_sar_indices_node
   from rsgispy.imagecalc.percentiles import calc_imgs_pxl_percentiles_node

   # Speckle filter -> SAR index for each date -> per-pixel median, one pass.
   rvis = [
       calc_sar_indices_node(
           lee_filter_node(RasterSource(img), 7),
           {1: "VV", 2: "VH"},
           ["RVI_VV_VH"],
           "power",
       )
       for img in ["s1_t1.tif", "s1_t2.tif", "s1_t3.tif"]
   ]
   med = calc_imgs_pxl_percentiles_node(rvis, [50], no_data_val=-999)
   TileGraph(DiskSink(med, "rvi_median.tif")).run()

See :ref:`tilegraph-chainable` for the list of chainable operations.

Band Mathematics, Spectral Indices & Value Rescaling
----------------------------------------------------

* :doc:`imagecalc/maths` — muParser band/image expressions and true-pixel proportions.
* :doc:`imagecalc/calcindices` — Unified multi-band spectral index stack calculator with gain/offset scaling.
* :doc:`imagecalc/rescale` — Linear rescaling and gain/offset/nodata transformations.

.. toctree::
   :maxdepth: 1

   imagecalc/maths
   imagecalc/calcindices
   imagecalc/rescale

Statistics, Percentiles & Comparison
------------------------------------

* :doc:`imagecalc/stats` — Basic band statistics, streaming histograms (:func:`~rsgispy.imagecalc.stats.get_img_histogram`), unique values, multi-image summaries, and mask probabilities.
* :doc:`imagecalc/percentiles` — Band percentiles and pixel-wise cross-band/cross-image percentiles.
* :doc:`imagecalc/comparison` — Image difference, equality tests, correlation, mutual info, EMD, and similarity metrics.

.. toctree::
   :maxdepth: 1

   imagecalc/stats
   imagecalc/percentiles
   imagecalc/comparison

Encoding, Distance, Least-Cost Path, Filling & Sampling
--------------------------------------------------------

* :doc:`imagecalc/encoding` — Integer recoding and one-hot category encoding.
* :doc:`imagecalc/distance` — Distance-to-pixel-values (full and tiled) and morphological buffering.
* :doc:`imagecalc/leastcostpath` — Least-cost path calculation across raster cost surfaces.
* :doc:`imagecalc/fill` — K-nearest-neighbour and iterative region-growing filling.
* :doc:`imagecalc/calc_pt_win_smpls` — Windowed summary statistics around point samples.

.. toctree::
   :maxdepth: 1

   imagecalc/encoding
   imagecalc/distance
   imagecalc/leastcostpath
   imagecalc/fill
   imagecalc/calc_pt_win_smpls
