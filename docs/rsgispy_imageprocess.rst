rsgispy.imageprocess
====================

The :mod:`rsgispy.imageprocess` module provides reusable utilities for validating raster metadata, computing spatial intersections across multiple input rasters, opening input/output datasets safely, and applying custom block-processing functions over raster windows.

.. seealso::

   Guide: :doc:`guides/large_rasters`

Custom Block Processing Example
-------------------------------

You can pass any Python callable ``block_func(block_vars)`` to :func:`rsgispy.imageprocess.process_blocks` to run custom windowed computations over one or more input rasters:

.. code-block:: python

   from contextlib import ExitStack
   import numpy as np
   import rsgispy
   import rsgispy.imageprocess

   band_defs = rsgispy.imageprocess.normalise_band_defs([
       rsgispy.imageprocess.BandDefn(band_name="nir", input_img="s2.tif", img_band=8),
       rsgispy.imageprocess.BandDefn(band_name="red", input_img="s2.tif", img_band=4),
   ])

   def calc_ndvi(block_vars):
       nir = block_vars["nir"]
       red = block_vars["red"]
       return (nir - red) / (nir + red)

   with ExitStack() as stack:
       datasets = rsgispy.imageprocess.open_input_datasets(stack, band_defs)
       grid = rsgispy.imageprocess.compute_spatial_intersection(datasets)
       dst = rsgispy.imageprocess.open_output_dataset(
           stack=stack,
           output_img="ndvi.tif",
           gdalformat="GTIFF",
           out_rio_dtype=rsgispy.get_rasterio_datatype(rsgispy.TYPE_32FLOAT),
           grid=grid,
       )
       rsgispy.imageprocess.process_blocks(
           dst=dst,
           datasets=datasets,
           norm_defs=band_defs,
           grid=grid,
           block_func=calc_ndvi,
           out_np_dtype=np.dtype(np.float32),
       )

Composable In-Memory Tile-Graph Pipelines
-----------------------------------------

:class:`rsgispy.imageprocess.TileGraph` chains multiple windowed
(:class:`~rsgispy.imageprocess.WindowOp`) and point-wise
(:class:`~rsgispy.imageprocess.PointwiseOp`, :class:`~rsgispy.imageprocess.BandMathOp`)
operations in memory per tile with automatic cumulative halo propagation,
avoiding intermediate disk writes:

.. code-block:: python

   import rsgispy
   from rsgispy.imageprocess import (
       BandMathOp,
       DiskSink,
       RasterSource,
       TileGraph,
   )
   from rsgispy.imagefilter.stats import mean_filter_node, median_filter_node

   src = RasterSource("input.tif")
   smooth = mean_filter_node(median_filter_node(src, 3), 5)
   diff = BandMathOp({"smooth": smooth, "orig": src}, "smooth - orig")
   sink = DiskSink(diff, "highpass.tif", datatype=rsgispy.TYPE_32FLOAT)
   graph = TileGraph(sink)
   print(graph.describe())
   graph.run()

The building blocks are:

* **sources**: :class:`~rsgispy.imageprocess.RasterSource` (bands of a file)
  and :class:`~rsgispy.imageprocess.ArraySource` (a NumPy array).
  ``RasterSource`` nodes on the same file share one open dataset and read
  their bands together for each tile;
* **operations**: :class:`~rsgispy.imageprocess.WindowOp`,
  :class:`~rsgispy.imageprocess.PointwiseOp`,
  :class:`~rsgispy.imageprocess.BandMathOp`, the band nodes
  :class:`~rsgispy.imageprocess.SelectBands` (1-based band selection),
  :class:`~rsgispy.imageprocess.StackBands` (stack the bands of several
  nodes, keeping band names), :class:`~rsgispy.imageprocess.Cast` and
  :class:`~rsgispy.imageprocess.SetNoData` (set or clear the no-data value
  in the metadata), and the node builders of the chainable operations below;
* **sinks**: :class:`~rsgispy.imageprocess.DiskSink` (write a raster),
  :class:`~rsgispy.imageprocess.MemorySink` (return an array),
  :class:`~rsgispy.imageprocess.StatsSink` (per-band statistics and
  histograms) and :class:`~rsgispy.imageprocess.ZonalStatsSink` (statistics
  per label).

When a graph is built it is checked before any pixel is read:

* **Grid checks**: all sources must have the same size, transform and CRS
  (to within a thousandth of a pixel for the origin); otherwise an error
  names the rasters that differ. Resample first with
  :func:`~rsgispy.imageutils.warp.resample_img_to_match`.
* **Metadata**: every node reports its number of bands, band names, data
  type and no-data value (:class:`~rsgispy.imageprocess.NodeMeta`, in
  ``graph.node_meta``), derived from its inputs. Incompatible
  ``BandMathOp`` band counts, bands out of range and too many sink band
  names raise :class:`rsgispy.RSGISPyException`. A ``DiskSink`` without
  ``band_names`` or ``nodata`` uses those of its input (pass
  ``inherit_meta=False`` to turn this off).
* :meth:`~rsgispy.imageprocess.TileGraph.describe` returns a text listing of
  the nodes with their inputs, parameters, halos and metadata.

``block_size`` defaults to the package default block size
(:func:`~rsgispy.imageprocess.get_default_block_size`, 256 pixels); up to
rsgispy 0.3 the tile graph always defaulted to 256 and the package default
was 512. ``TileGraph.run(n_jobs=...)`` evaluates tiles on several threads
(see :ref:`tilegraph-parallel`).

.. _tilegraph-chainable:

Chainable operations
~~~~~~~~~~~~~~~~~~~~

An operation can be a node when it is **tile-local**: every output pixel
depends only on the input pixels within a fixed radius (the halo) and on
parameters known before the graph runs. Each such operation has a public
node builder in the module of its file-based function, named after the
function without any ``apply_`` prefix and with ``_node``. Builders take
``(source_or_sources, <the function's processing parameters>, *,
no_data_val=None, name=None)``; the file options (``gdalformat``,
``datatype``, ``calc_stats``, ``creation_opts``) belong to the
:class:`~rsgispy.imageprocess.DiskSink`. They validate their parameters as
the file-based functions do, and the file-based functions are thin wrappers
(``RasterSource`` → node → ``DiskSink``, run with the package default block
size), so a node gives exactly the output of its function. The filters also
have a :class:`~rsgispy.imagefilter.chained.FilterStep` kind for
:func:`~rsgispy.imagefilter.chained.apply_filter_chain` and
:func:`~rsgispy.imagefilter.chained.build_filter_chain_node`, which calls the
same builder.

.. list-table:: Filters (:mod:`rsgispy.imagefilter`)
   :header-rows: 1
   :widths: 38 38 24

   * - File-based function
     - Node builder
     - ``FilterStep`` kind
   * - ``stats.apply_<stat>_filter`` (``median``, ``mean``, ``min``,
       ``max``, ``mode``, ``stddev``, ``range``, ``total``, ``mean_diff``,
       ``mean_diff_abs``, ``total_diff``, ``total_diff_abs``,
       ``coeff_of_var``, ``norm_var``, ``norm_var_sqrt``, ``norm_var_ln``)
     - ``stats.<stat>_filter_node``
     - ``'<stat>'``
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
   * - ``edge.apply_sobel_filter``, ``apply_sobel_x_filter``,
       ``apply_sobel_y_filter``, ``apply_prewitt_filter``,
       ``apply_prewitt_x_filter``, ``apply_prewitt_y_filter``
     - ``edge.sobel_filter_node``, ``sobel_x_filter_node``, ...
       (:func:`~rsgispy.imagefilter.edge.sobel_filter_node`)
     - ``'sobel'``, ``'sobel_x'``, ``'sobel_y'``, ``'prewitt'``,
       ``'prewitt_x'``, ``'prewitt_y'``
   * - :func:`~rsgispy.imagefilter.edge.apply_gaussian_1st_deriv_filter`,
       :func:`~rsgispy.imagefilter.edge.apply_gaussian_2nd_deriv_filter`
     - :func:`~rsgispy.imagefilter.edge.gaussian_1st_deriv_filter_node`,
       :func:`~rsgispy.imagefilter.edge.gaussian_2nd_deriv_filter_node`
     - ``'gaussian_1st_deriv'``, ``'gaussian_2nd_deriv'``
   * - :func:`~rsgispy.imagefilter.edge.apply_laplacian_filter`
     - :func:`~rsgispy.imagefilter.edge.laplacian_filter_node`
     - ``'laplacian'``
   * - ``sar.apply_<filter>_filter`` (``lee``, ``enhanced_lee``, ``frost``,
       ``enhanced_frost``, ``kuan``, ``gamma_map``, ``refined_lee``)
     - ``sar.<filter>_filter_node``
       (:func:`~rsgispy.imagefilter.sar.refined_lee_filter_node`, ...)
     - ``'<filter>'``
   * - :func:`~rsgispy.imagefilter.texture.apply_glcm_filter`
     - :func:`~rsgispy.imagefilter.texture.glcm_filter_node` (grey-level
       range given, e.g. from
       :func:`~rsgispy.imagefilter.texture.calc_glcm_band_bounds`)
     - ``'glcm'``
   * - band maths, any function
     - :class:`~rsgispy.imageprocess.BandMathOp`,
       :class:`~rsgispy.imageprocess.PointwiseOp`
     - ``'band_math'``, ``'pointwise'``

.. list-table:: Pixel-wise operations (indices, value transforms, masks, change detection, composites, spectral)
   :header-rows: 1
   :widths: 50 50

   * - File-based function
     - Node builder
   * - :func:`~rsgispy.imagecalc.calcindices.calc_indices`,
       :func:`~rsgispy.imagecalc.calcindices.calc_sar_indices`
     - :func:`~rsgispy.imagecalc.calcindices.calc_indices_node`,
       :func:`~rsgispy.imagecalc.calcindices.calc_sar_indices_node`
   * - :func:`~rsgispy.imagecalc.rescale.rescale_img_pxl_vals`,
       :func:`~rsgispy.imagecalc.rescale.calc_img_rescale`,
       :func:`~rsgispy.imagecalc.rescale.apply_img_norm_params`
     - :func:`~rsgispy.imagecalc.rescale.rescale_img_pxl_vals_node`,
       :func:`~rsgispy.imagecalc.rescale.calc_img_rescale_node`,
       :func:`~rsgispy.imagecalc.rescale.img_norm_params_node`
   * - :func:`~rsgispy.imagecalc.encoding.recode_int_raster`,
       :func:`~rsgispy.imagecalc.encoding.one_hot_category_encoding`
     - :func:`~rsgispy.imagecalc.encoding.recode_int_raster_node`,
       :func:`~rsgispy.imagecalc.encoding.one_hot_category_encoding_node`
   * - :func:`~rsgispy.imageutils.masking.mask_img`,
       :func:`~rsgispy.imageutils.masking.gen_finite_mask`,
       :func:`~rsgispy.imageutils.masking.gen_valid_mask`,
       :func:`~rsgispy.imageutils.masking.gen_img_edge_mask` (window),
       :func:`~rsgispy.imageutils.masking.mask_all_band_zero_vals`,
       :func:`~rsgispy.imageutils.masking.create_valid_mask`
     - ``mask_img_node``, ``gen_finite_mask_node``, ``gen_valid_mask_node``,
       ``gen_img_edge_mask_node``, ``mask_all_band_zero_vals_node``,
       ``create_valid_mask_node`` (:mod:`rsgispy.imageutils.masking`)
   * - :func:`~rsgispy.changedetect.imgchngmthds.image_difference`,
       :func:`~rsgispy.changedetect.imgchngmthds.image_ratio`
     - :func:`~rsgispy.changedetect.imgchngmthds.image_difference_node`,
       :func:`~rsgispy.changedetect.imgchngmthds.image_ratio_node` (two
       inputs; standardisation statistics given as ``std_params``)
   * - :func:`~rsgispy.imagecalc.percentiles.calc_imgs_pxl_percentiles`,
       :func:`~rsgispy.imagecalc.percentiles.calc_img_band_pxl_percentiles`
     - :func:`~rsgispy.imagecalc.percentiles.calc_imgs_pxl_percentiles_node`,
       :func:`~rsgispy.imagecalc.percentiles.calc_img_band_pxl_percentiles_node`
       (N inputs)
   * - :func:`~rsgispy.imagespectral.transforms.calc_tasseled_cap`;
       applying a fitted :func:`~rsgispy.imagespectral.transforms.calc_pca`
       or :func:`~rsgispy.imagespectral.transforms.calc_mnf`
     - :func:`~rsgispy.imagespectral.transforms.calc_tasseled_cap_node`,
       :func:`~rsgispy.imagespectral.transforms.pca_projection_node`,
       :func:`~rsgispy.imagespectral.transforms.mnf_projection_node`
   * - :func:`~rsgispy.imagespectral.classification.spectral_angle_mapper`,
       :func:`~rsgispy.imagespectral.classification.spectral_correlation_mapper`
     - :func:`~rsgispy.imagespectral.classification.spectral_angle_mapper_node`,
       :func:`~rsgispy.imagespectral.classification.spectral_correlation_mapper_node`
   * - :func:`~rsgispy.imagespectral.unmixing.spec_unmix_ucls`,
       :func:`~rsgispy.imagespectral.unmixing.spec_unmix_nnls`,
       :func:`~rsgispy.imagespectral.unmixing.spec_unmix_fcls`,
       :func:`~rsgispy.imagespectral.unmixing.predict_refl_linear_unmixing`
     - ``spec_unmix_ucls_node``, ``spec_unmix_nnls_node``,
       ``spec_unmix_fcls_node``, ``predict_refl_linear_unmixing_node``
       (:mod:`rsgispy.imagespectral.unmixing`)

.. list-table:: Terrain, morphology, classification post-processing and machine learning
   :header-rows: 1
   :widths: 50 50

   * - File-based function
     - Node builder
   * - :func:`~rsgispy.elevation.terrain.slope`,
       :func:`~rsgispy.elevation.terrain.aspect`,
       :func:`~rsgispy.elevation.terrain.categorise_aspect` and the
       ``slope_pxl_res_img`` / ``aspect_pxl_res_img`` variants
     - :func:`~rsgispy.elevation.terrain.slope_node`,
       :func:`~rsgispy.elevation.terrain.aspect_node`,
       :func:`~rsgispy.elevation.terrain.categorise_aspect_node`,
       ``slope_pxl_res_img_node``, ``aspect_pxl_res_img_node`` (the
       pixel-resolution image is a second input)
   * - :func:`~rsgispy.elevation.illumination.hillshade` (and
       ``hillshade_pxl_res_img``),
       :func:`~rsgispy.elevation.illumination.local_incidence_angle`,
       :func:`~rsgispy.elevation.illumination.local_exitance_angle`
     - :func:`~rsgispy.elevation.illumination.hillshade_node`,
       ``hillshade_pxl_res_img_node``,
       :func:`~rsgispy.elevation.illumination.local_incidence_angle_node`,
       :func:`~rsgispy.elevation.illumination.local_exitance_angle_node`
   * - ``rsgispy.imagemorphology.image_dilate``, ``image_erode``,
       ``image_gradient``, ``image_local_minima`` (binary output),
       ``image_opening``, ``image_closing``, ``image_white_top_hat``,
       ``image_black_top_hat`` (``*_combined_out``: ``combined=True``)
     - ``image_dilate_node``, ``image_erode_node``, ... (:mod:`rsgispy.imagemorphology`)
   * - :func:`~rsgispy.classification.classpostprocess.apply_majority_filter`
     - :func:`~rsgispy.classification.classpostprocess.majority_filter_node`
   * - :func:`~rsgispy.ml.predict.predict_raster`,
       :func:`~rsgispy.ml.predict.predict_proba_raster`,
       :func:`~rsgispy.ml.predict.predict_quantiles_raster`
     - :func:`~rsgispy.ml.predict.predict_raster_node`,
       :func:`~rsgispy.ml.predict.predict_proba_raster_node`,
       :func:`~rsgispy.ml.predict.predict_quantiles_raster_node` (feature
       nodes, e.g. from :func:`~rsgispy.ml.predict.feature_stack_nodes`, an
       estimator and an optional ``FeatureScaler``)

Some of these functions need whole-image parameters, which their builders
take as arguments; the file-based functions still calculate them in a first
pass: ``Brightness_Scaled`` bounds and the ``DPSVI`` scene maximum
(``index_params``), the ``apply_std`` statistics of change detection
(``std_params``), the categories of ``one_hot_category_encoding``
(``recode_dict``), the normalisation parameters of
:func:`~rsgispy.imagecalc.rescale.calc_img_norm_params` and the GLCM
grey-level range. A :class:`~rsgispy.imageprocess.StatsSink` computes such
statistics for any node in one pass (see below).

Operations that are **not** nodes, because an output pixel does not depend
only on the input pixels within a fixed radius:

* **whole-image operations**: segmentation (:mod:`rsgispy.segmentation`),
  hydrological flow routing and accumulation, distance transforms
  (``calc_dist_to_img_vals``), ``apply_sieve_filter`` (connected
  components), region-growing fills, ``shadow_mask`` (rays of unbounded
  length) and sequentially numbered local minima
  (``image_local_minima(sequencial_out=True)``: the numbering depends on all
  earlier pixels);
* **operations that change the pixel grid**: warping, resampling and
  reprojection (resample first, e.g. with
  :func:`~rsgispy.imageutils.warp.resample_img_to_match`);
* **operations that need whole-image parameters**, unless they are given:
  ``calc_pca``, ``calc_mnf``, ``whiten_image``, ``histogram_matching`` and
  normalisation without parameters. These can be done in **two passes**:
  compute the parameters first (e.g. with a ``StatsSink``, or the fitting
  function), then apply them pixel by pixel (``pca_projection_node``,
  ``img_norm_params_node``, ...);
* **optional backends**: unmixing with ``pysptools`` or ``pyMCR`` keeps its
  file-based functions.

.. _tilegraph-nodata:

No-data convention
~~~~~~~~~~~~~~~~~~

Every node carries a no-data value in its metadata (a ``RasterSource`` has
the file's, or its ``nodata`` override; an operation node the value it
writes where it has no result). All node builders follow one rule for their
input no-data value:

* ``no_data_val=None`` (the default) — use the no-data value of the input
  node's metadata (``None`` when it has none: then only non-finite values
  are invalid);
* ``no_data_val=<value>`` — use ``value`` instead.

To turn the no-data handling off, or to declare a value for an input
without one, set it in the metadata with
:class:`~rsgispy.imageprocess.SetNoData`, which changes no pixels:
``median_filter_node(SetNoData(src, None), 5)`` treats every finite pixel as
valid even if the file declares a no-data value;
``SetNoData(src, 0)`` (or ``RasterSource(path, nodata=0)``) makes ``0`` the
no-data value of everything downstream. The few builders whose parameter is
not the input's no-data value keep the meaning of their file-based function
(e.g. ``gen_valid_mask_node`` and the percentile nodes default to ``0`` and
check only finiteness with ``None``; the prediction nodes' ``no_data_val`` is
the output value). :func:`~rsgispy.imageprocess.resolve_input_nodata` and
:func:`~rsgispy.imageprocess.infer_node_meta` apply the rule in custom
builders.

Before rsgispy 0.4 the filter, morphology and majority-filter builders (and
:func:`~rsgispy.imagefilter.chained.build_filter_chain_node`) ignored the
input's no-data value when ``no_data_val`` was ``None``; wrap the input in
``SetNoData(src, None)`` for that behaviour. The file-based functions and
:func:`~rsgispy.imagefilter.chained.apply_filter_chain` are unchanged.

Statistics sinks
~~~~~~~~~~~~~~~~

:class:`~rsgispy.imageprocess.StatsSink` computes per-band statistics of any
node in the same single pass: the count of valid pixels, minimum, maximum,
mean and standard deviation (streaming, numerically stable updates of
:class:`~rsgispy.imageprocess.OnlineStatsAccumulator`) and, with ``bins``
and ``hist_range`` (or bin edges), a histogram. Non-finite and no-data pixels
are excluded (the input's no-data value by default). The
:class:`~rsgispy.imageprocess.RasterStats` result is returned by
:meth:`~rsgispy.imageprocess.TileGraph.run` under the sink's ``name``.
:class:`~rsgispy.imageprocess.ZonalStatsSink` takes a single-band label node
and a value node and returns a :class:`pandas.DataFrame` of the count, mean,
minimum, maximum and standard deviation of each value band per label.

.. code-block:: python

   from rsgispy.imageprocess import DiskSink, RasterSource, StatsSink, TileGraph
   from rsgispy.imagecalc.calcindices import calc_indices_node
   from rsgispy.imagefilter.stats import median_filter_node

   refl = RasterSource("s2_refl.tif")
   ndvi = median_filter_node(
       calc_indices_node(refl, {4: "red", 8: "nir"}, ["NDVI"]), 3
   )
   res = TileGraph(
       [
           DiskSink(ndvi, "ndvi_med3.tif"),
           StatsSink(ndvi, bins=20, hist_range=(-1, 1), name="ndvi_stats"),
       ]
   ).run()
   print(res["ndvi_stats"].to_dataframe())

.. _tilegraph-parallel:

Parallel tiles
~~~~~~~~~~~~~~

``TileGraph.run(n_jobs=4)`` (and ``run_tile_graph(..., n_jobs=4)``) evaluates
the tiles on a pool of threads; NumPy, GDAL reads and the numba kernels
release the GIL. Every thread has its own open datasets and per-run node
state, and the blocks are written to the sinks in tile order from the calling
thread, so the outputs (and statistics) are identical to ``n_jobs=1``, the
default; ``n_jobs=None`` or ``-1`` uses all CPUs. Custom nodes must keep
per-run state in the object returned by
:meth:`~rsgispy.imageprocess.TileNode.start_run`, not on the node. On a
4096 x 4096 image (block size 256, Apple M2 Max), ``n_jobs=4`` ran 3.4-4x and
``n_jobs=8`` 4.7-7x faster than one thread for a median → mean chain, a
Refined Lee → dB → SAR indices chain, slope + hillshade and a random-forest
prediction → majority filter. The file-based functions run with one thread.

Custom nodes
~~~~~~~~~~~~

Nodes implement a small protocol (see :class:`~rsgispy.imageprocess.TileNode`):
``halo_radius``, ``infer_meta``, ``start_run`` and ``evaluate`` (which
receives and returns :class:`~rsgispy.imageprocess.TileTensor` blocks).
Sinks derive from :class:`~rsgispy.imageprocess.TileSink` and implement
``open``, ``write`` and ``close``. A new node type therefore works in a
``TileGraph`` without changing the graph.

API Reference
-------------

.. automodule:: rsgispy.imageprocess
   :members:
   :undoc-members:
   :show-inheritance:
