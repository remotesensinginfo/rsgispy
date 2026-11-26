rsgispy.imagefilter
===================

The :mod:`rsgispy.imagefilter` subpackage provides spatial moving-window and
convolution filters for raster images using ``rasterio`` and ``numpy``,
organised into thematic submodules:

.. seealso::

   Guides: :doc:`guides/sar_processing`, :doc:`guides/large_rasters`

* :doc:`imagefilter/stats` — Statistical, dispersion, and local-difference moving-window filters.
* :doc:`imagefilter/smooth` — Edge-preserving Kuwahara, Gaussian smoothing, and custom 2D convolution kernel filters.
* :doc:`imagefilter/edge` — Sobel, Prewitt, Gaussian derivative, and Laplacian edge-detection filters.
* :doc:`imagefilter/sar` — Lee, Enhanced Lee, Frost, Enhanced Frost, Kuan, Gamma MAP, and Refined Lee SAR speckle filters.
* :doc:`imagefilter/texture` — Gray-Level Co-occurrence Matrix (GLCM) spatial texture filters.
* :doc:`imagefilter/chained` — Halo-fused in-memory filter chains (:func:`~rsgispy.imagefilter.chained.apply_filter_chain`) eliminating intermediate GeoTIFF writes.

Example
-------

Applying a median smoothing filter, a Refined Lee SAR speckle filter, and GLCM
texture extraction:

.. code-block:: python

   import rsgispy
   import rsgispy.imagefilter.stats
   import rsgispy.imagefilter.sar
   import rsgispy.imagefilter.texture as tex

   rsgispy.imagefilter.stats.apply_median_filter(
       input_img="input.tif",
       output_img="median_3x3.tif",
       filter_size=3,
   )
   rsgispy.imagefilter.sar.apply_refined_lee_filter(
       input_img="sar.tif",
       output_img="sar_refined_lee.tif",
       filter_size=7,
       n_looks=4.0,
   )
   tex.apply_glcm_filter(
       input_img="input.tif",
       output_img="glcm.tif",
       properties=[tex.GLCM_CONTRAST, tex.GLCM_HOMOGENEITY, tex.GLCM_ENTROPY],
       filter_size=5,
   )

Chaining
--------

Most filters can also be used as nodes of an in-memory
:class:`~rsgispy.imageprocess.TileGraph`, chained with other filters, band
maths and indices in one pass without intermediate files. Each has a node
builder next to its file-based function, named without ``apply_`` and with
``_node``, which gives the same result:

* :mod:`~rsgispy.imagefilter.stats`: ``median_filter_node``,
  ``mean_filter_node``, ``min_filter_node``, ``max_filter_node``,
  ``mode_filter_node``, ``stddev_filter_node``, ``range_filter_node``,
  ``total_filter_node``, ``mean_diff_filter_node``,
  ``mean_diff_abs_filter_node``, ``total_diff_filter_node``,
  ``total_diff_abs_filter_node``, ``coeff_of_var_filter_node``,
  ``norm_var_filter_node``, ``norm_var_sqrt_filter_node``,
  ``norm_var_ln_filter_node``, ``texture_var_filter_node``;
* :mod:`~rsgispy.imagefilter.smooth`: ``kuwahara_filter_node``,
  ``gaussian_smooth_filter_node``, ``kernel_filter_node``;
* :mod:`~rsgispy.imagefilter.edge`: ``sobel_filter_node``,
  ``prewitt_filter_node``, ``laplacian_filter_node``;
* :mod:`~rsgispy.imagefilter.sar`: ``lee_filter_node``.

.. code-block:: python

   from rsgispy.imageprocess import DiskSink, RasterSource, TileGraph
   from rsgispy.imagefilter.edge import sobel_filter_node
   from rsgispy.imagefilter.stats import median_filter_node

   src = RasterSource("input.tif")
   edges = sobel_filter_node(median_filter_node(src, 3, no_data_val=0), no_data_val=0)
   TileGraph(DiskSink(edges, "median_sobel.tif")).run()

See :doc:`rsgispy_imageprocess` for the list of chainable operations and
:func:`~rsgispy.imagefilter.chained.apply_filter_chain` for filter-only
chains.

Submodules
----------

.. toctree::
   :maxdepth: 1

   imagefilter/stats
   imagefilter/smooth
   imagefilter/edge
   imagefilter/sar
   imagefilter/texture
   imagefilter/chained
