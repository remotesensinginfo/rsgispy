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

Every filter in this package can also be used as a node of an in-memory
:class:`~rsgispy.imageprocess.TileGraph`, chained with other filters, band
maths and indices in one pass without intermediate files. Each has a node
builder next to its file-based function, named without ``apply_`` and with
``_node``, taking ``(source, <the function's processing parameters>, *,
no_data_val=None, name=None)``; the file options (``gdalformat``,
``datatype``, ``calc_stats``) belong to the
:class:`~rsgispy.imageprocess.DiskSink`. ``no_data_val`` is not read from
the source, so pass the input's no-data value. Each builder also has a
:class:`~rsgispy.imagefilter.chained.FilterStep` kind for
:func:`~rsgispy.imagefilter.chained.apply_filter_chain`.

The file-based functions are themselves thin wrappers that run
``RasterSource -> <builder> -> DiskSink`` with the package default block
size, so a builder gives exactly the same result as its function, and a
chain of builders the same result as running the functions one after
another (with ``datatype=rsgispy.TYPE_64FLOAT`` intermediate files, as the
nodes pass ``float64`` blocks on).

.. list-table::
   :header-rows: 1
   :widths: 38 38 24

   * - File-based function
     - Node builder
     - ``FilterStep`` kind
   * - :func:`~rsgispy.imagefilter.stats.apply_median_filter`
     - :func:`~rsgispy.imagefilter.stats.median_filter_node`
     - ``'median'``
   * - :func:`~rsgispy.imagefilter.stats.apply_mean_filter`
     - :func:`~rsgispy.imagefilter.stats.mean_filter_node`
     - ``'mean'``
   * - :func:`~rsgispy.imagefilter.stats.apply_min_filter`
     - :func:`~rsgispy.imagefilter.stats.min_filter_node`
     - ``'min'``
   * - :func:`~rsgispy.imagefilter.stats.apply_max_filter`
     - :func:`~rsgispy.imagefilter.stats.max_filter_node`
     - ``'max'``
   * - :func:`~rsgispy.imagefilter.stats.apply_mode_filter`
     - :func:`~rsgispy.imagefilter.stats.mode_filter_node`
     - ``'mode'``
   * - :func:`~rsgispy.imagefilter.stats.apply_stddev_filter`
     - :func:`~rsgispy.imagefilter.stats.stddev_filter_node`
     - ``'stddev'``
   * - :func:`~rsgispy.imagefilter.stats.apply_range_filter`
     - :func:`~rsgispy.imagefilter.stats.range_filter_node`
     - ``'range'``
   * - :func:`~rsgispy.imagefilter.stats.apply_total_filter`
     - :func:`~rsgispy.imagefilter.stats.total_filter_node`
     - ``'total'``
   * - :func:`~rsgispy.imagefilter.stats.apply_mean_diff_filter`
     - :func:`~rsgispy.imagefilter.stats.mean_diff_filter_node`
     - ``'mean_diff'``
   * - :func:`~rsgispy.imagefilter.stats.apply_mean_diff_abs_filter`
     - :func:`~rsgispy.imagefilter.stats.mean_diff_abs_filter_node`
     - ``'mean_diff_abs'``
   * - :func:`~rsgispy.imagefilter.stats.apply_total_diff_filter`
     - :func:`~rsgispy.imagefilter.stats.total_diff_filter_node`
     - ``'total_diff'``
   * - :func:`~rsgispy.imagefilter.stats.apply_total_diff_abs_filter`
     - :func:`~rsgispy.imagefilter.stats.total_diff_abs_filter_node`
     - ``'total_diff_abs'``
   * - :func:`~rsgispy.imagefilter.stats.apply_coeff_of_var_filter`
     - :func:`~rsgispy.imagefilter.stats.coeff_of_var_filter_node`
     - ``'coeff_of_var'``
   * - :func:`~rsgispy.imagefilter.stats.apply_norm_var_filter`
     - :func:`~rsgispy.imagefilter.stats.norm_var_filter_node`
     - ``'norm_var'``
   * - :func:`~rsgispy.imagefilter.stats.apply_norm_var_sqrt_filter`
     - :func:`~rsgispy.imagefilter.stats.norm_var_sqrt_filter_node`
     - ``'norm_var_sqrt'``
   * - :func:`~rsgispy.imagefilter.stats.apply_norm_var_ln_filter`
     - :func:`~rsgispy.imagefilter.stats.norm_var_ln_filter_node`
     - ``'norm_var_ln'``
   * - :func:`~rsgispy.imagefilter.stats.apply_texture_var_filter`
     - :func:`~rsgispy.imagefilter.stats.texture_var_filter_node`
     - ``'variance'``
   * - :func:`~rsgispy.imagefilter.smooth.apply_kuwahara_filter`
     - :func:`~rsgispy.imagefilter.smooth.kuwahara_filter_node`
     - ``'kuwahara'``
   * - :func:`~rsgispy.imagefilter.smooth.apply_gaussian_smooth_filter`
     - :func:`~rsgispy.imagefilter.smooth.gaussian_smooth_filter_node`
     - ``'gaussian'``
   * - :func:`~rsgispy.imagefilter.smooth.apply_kernel_filter`
     - :func:`~rsgispy.imagefilter.smooth.kernel_filter_node`
     - ``'kernel'``
   * - :func:`~rsgispy.imagefilter.edge.apply_sobel_filter`
     - :func:`~rsgispy.imagefilter.edge.sobel_filter_node`
     - ``'sobel'``
   * - :func:`~rsgispy.imagefilter.edge.apply_sobel_x_filter`
     - :func:`~rsgispy.imagefilter.edge.sobel_x_filter_node`
     - ``'sobel_x'``
   * - :func:`~rsgispy.imagefilter.edge.apply_sobel_y_filter`
     - :func:`~rsgispy.imagefilter.edge.sobel_y_filter_node`
     - ``'sobel_y'``
   * - :func:`~rsgispy.imagefilter.edge.apply_prewitt_filter`
     - :func:`~rsgispy.imagefilter.edge.prewitt_filter_node`
     - ``'prewitt'``
   * - :func:`~rsgispy.imagefilter.edge.apply_prewitt_x_filter`
     - :func:`~rsgispy.imagefilter.edge.prewitt_x_filter_node`
     - ``'prewitt_x'``
   * - :func:`~rsgispy.imagefilter.edge.apply_prewitt_y_filter`
     - :func:`~rsgispy.imagefilter.edge.prewitt_y_filter_node`
     - ``'prewitt_y'``
   * - :func:`~rsgispy.imagefilter.edge.apply_gaussian_1st_deriv_filter`
     - :func:`~rsgispy.imagefilter.edge.gaussian_1st_deriv_filter_node`
     - ``'gaussian_1st_deriv'``
   * - :func:`~rsgispy.imagefilter.edge.apply_gaussian_2nd_deriv_filter`
     - :func:`~rsgispy.imagefilter.edge.gaussian_2nd_deriv_filter_node`
     - ``'gaussian_2nd_deriv'``
   * - :func:`~rsgispy.imagefilter.edge.apply_laplacian_filter`
     - :func:`~rsgispy.imagefilter.edge.laplacian_filter_node`
     - ``'laplacian'``
   * - :func:`~rsgispy.imagefilter.sar.apply_lee_filter`
     - :func:`~rsgispy.imagefilter.sar.lee_filter_node`
     - ``'lee'``
   * - :func:`~rsgispy.imagefilter.sar.apply_enhanced_lee_filter`
     - :func:`~rsgispy.imagefilter.sar.enhanced_lee_filter_node`
     - ``'enhanced_lee'``
   * - :func:`~rsgispy.imagefilter.sar.apply_frost_filter`
     - :func:`~rsgispy.imagefilter.sar.frost_filter_node`
     - ``'frost'``
   * - :func:`~rsgispy.imagefilter.sar.apply_enhanced_frost_filter`
     - :func:`~rsgispy.imagefilter.sar.enhanced_frost_filter_node`
     - ``'enhanced_frost'``
   * - :func:`~rsgispy.imagefilter.sar.apply_kuan_filter`
     - :func:`~rsgispy.imagefilter.sar.kuan_filter_node`
     - ``'kuan'``
   * - :func:`~rsgispy.imagefilter.sar.apply_gamma_map_filter`
     - :func:`~rsgispy.imagefilter.sar.gamma_map_filter_node`
     - ``'gamma_map'``
   * - :func:`~rsgispy.imagefilter.sar.apply_refined_lee_filter`
     - :func:`~rsgispy.imagefilter.sar.refined_lee_filter_node`
     - ``'refined_lee'``
   * - :func:`~rsgispy.imagefilter.texture.apply_glcm_filter`
     - :func:`~rsgispy.imagefilter.texture.glcm_filter_node`
     - ``'glcm'``

Node outputs are ``float64`` blocks with the bands and band names of the
input, and the halo radius is half the window or kernel size. The GLCM
node gives ``len(properties)`` bands per input band (times the number of
offsets without ``average_angles``), named ``<band>_glcm_<property>``.

GLCM quantises the grey levels over a range that must be known before the
graph runs: give ``min_val`` and ``max_val``, or per-band ``band_bounds``.
:func:`~rsgispy.imagefilter.texture.apply_glcm_filter` without
``min_val``/``max_val`` uses the minimum and maximum of each band, which
:func:`~rsgispy.imagefilter.texture.calc_glcm_band_bounds` calculates in a
first pass.

.. code-block:: python

   from rsgispy.imageprocess import BandMathOp, DiskSink, RasterSource, TileGraph
   import rsgispy.imagefilter.sar as sar
   import rsgispy.imagefilter.texture as tex

   # Refined Lee speckle filter, then dB, without intermediate files.
   s1 = RasterSource("s1_vv_linear.tif", nodata=0.0)
   rlee = sar.refined_lee_filter_node(s1, 7, n_looks=4, no_data_val=0.0)
   db = BandMathOp({"x": rlee}, "10 * log10(x)", nodata=0.0)
   TileGraph(DiskSink(db, "s1_vv_reflee_db.tif")).run()

   # GLCM texture of an image with the ranges apply_glcm_filter would use.
   bounds = tex.calc_glcm_band_bounds("s2_nir.tif")
   glcm = tex.glcm_filter_node(
       RasterSource("s2_nir.tif"),
       properties=[tex.GLCM_CONTRAST, tex.GLCM_ENTROPY],
       band_bounds=bounds,
   )
   TileGraph(DiskSink(glcm, "s2_nir_glcm.tif")).run()

The same chain with ``FilterStep`` kinds (the parameters go in ``params``):

.. code-block:: python

   import rsgispy.imagefilter.chained as ifc

   ifc.apply_filter_chain(
       "s1_vv_linear.tif",
       "s1_vv_reflee_sobel_x.tif",
       steps=[
           ifc.FilterStep("refined_lee", 7, params={"n_looks": 4}),
           ifc.FilterStep("sobel_x"),
       ],
   )

See :doc:`rsgispy_imageprocess` for the other chainable operations and the
operations that cannot be chained.

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
