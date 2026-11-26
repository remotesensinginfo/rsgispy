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

.. seealso::

   :doc:`guides/tile_graph` — a guide to chaining operations in memory with a
   tile graph, with a worked example.

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
  nodes, keeping band names; both take ``band_names=`` to rename the
  result), :class:`~rsgispy.imageprocess.Cast`,
  :class:`~rsgispy.imageprocess.SetNoData` (set or clear the no-data value
  in the metadata) and :class:`~rsgispy.imageprocess.SetBandNames` (rename
  the bands of any node, in the metadata), and the node builders of the
  chainable operations below;
* **sinks**: :class:`~rsgispy.imageprocess.DiskSink` (write a raster; with
  ``thematic=True``, ``colour_table=`` or ``classes=`` a class map with its
  colour table, class names and thematic statistics),
  :class:`~rsgispy.imageprocess.MemorySink` (return an array),
  :class:`~rsgispy.imageprocess.StatsSink` (per-band statistics and
  histograms), :class:`~rsgispy.imageprocess.ZonalStatsSink` (statistics
  per label) and :class:`~rsgispy.imageprocess.SampleSink` (training
  samples: the feature values of labelled pixels).

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
* **Output data type**: a ``DiskSink`` without ``datatype`` writes the data
  type of its input node (:func:`~rsgispy.imageprocess.datatype_from_meta`):
  integer and ``float32`` nodes keep their type, ``bool`` is written as
  ``uint8``, and ``float64`` nodes (the blocks of most filters and indices)
  or nodes of unknown type as ``float32``; ``describe()`` shows the type
  chosen. Pass ``datatype`` to choose another (e.g.
  ``rsgispy.TYPE_64FLOAT``). **Changed in rsgispy 0.4**: the default was
  always ``TYPE_32FLOAT``. Graphs that write these nodes without
  ``datatype`` now get integer files: the masks of
  ``gen_valid_mask_node``, ``gen_finite_mask_node``,
  ``gen_img_edge_mask_node`` and ``create_valid_mask_node``,
  ``one_hot_category_encoding_node``, ``recode_int_raster_node``,
  ``categorise_aspect_node``, ``hillshade_node``, the class output of
  ``spectral_angle_mapper_node`` and ``spectral_correlation_mapper_node``,
  ``majority_filter_node`` with an integer ``dtype``, a thematic or
  thresholded ``predict_raster_node`` (``uint8`` when every class fits),
  builders given an integer ``out_dtype``, a
  :class:`~rsgispy.imageprocess.Cast` to an integer type, and integer
  ``RasterSource`` / ``ArraySource`` nodes. The file-based functions are
  unchanged.
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

.. seealso::

   :doc:`guides/tile_graph` shows these builders in a worked chain.

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
   * - :func:`~rsgispy.imageutils.composite.create_geomedian_composite`,
       :func:`~rsgispy.imageutils.composite.create_quality_mosaic`
       (:func:`~rsgispy.imageutils.composite.create_max_ndvi_composite`),
       :func:`~rsgispy.imageutils.composite.create_land_water_composite`
       (:func:`~rsgispy.imageutils.composite.create_max_ndvi_ndwi_composite`),
       :func:`~rsgispy.imageutils.composite.create_medoid_composite`,
       :func:`~rsgispy.imageutils.composite.create_percentile_composite`
     - :func:`~rsgispy.imageutils.composite.geomedian_node`,
       :func:`~rsgispy.imageutils.composite.quality_mosaic_node`,
       :func:`~rsgispy.imageutils.composite.land_water_composite_node`,
       :func:`~rsgispy.imageutils.composite.medoid_node`,
       :func:`~rsgispy.imageutils.composite.percentile_composite_node`
       (one input per observation; they return a
       :class:`~rsgispy.imageutils.composite.CompositeNode` whose ``ref``,
       ``quality``, ``count``, ``mad`` and ``lw_msk`` outputs are nodes; the
       smoothing options use a halo)
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

**Stacking inputs with different no-data values.** A node has one no-data
value for all its bands, and the sinks treat a pixel that is invalid in any
band as invalid. :class:`~rsgispy.imageprocess.StackBands` keeps that rule
working when its inputs disagree (e.g. Sentinel-2 bands with no data ``0``
and an index with ``-999``):

* ``no_data_val=None`` (the default): the bands are stacked unchanged; the
  stack has the inputs' no-data value when they share one, else none
  (``None``), with a :class:`UserWarning`, because a
  :class:`~rsgispy.imageprocess.SampleSink` or
  :func:`~rsgispy.ml.predict.predict_raster_node` on it would then not skip
  the inputs' no-data pixels;
* ``no_data_val=<value>``: every band of a pixel where any input is invalid
  (its own no-data value, or not finite) is set to ``value``, the stack's
  no-data value;
* ``no_data_val='auto'``: the same with the first input's no-data value, or
  NaN for a floating-point stack whose first input has none (an integer
  stack then needs a value).

``describe()`` shows the stack's no-data value and whether it masks across
its inputs. The internal ``[labels, values]`` stacks that
:class:`~rsgispy.imageprocess.SampleSink` and
:class:`~rsgispy.imageprocess.ZonalStatsSink` add are shown as
``SampleSink input stack '...' (internal)``; they apply each input's own
no-data value and do not mask.

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

:func:`~rsgispy.imagecalc.rescale.norm_params_from_stats` turns a
``RasterStats`` into the normalisation parameters of
:func:`~rsgispy.imagecalc.rescale.calc_img_norm_params`, for
:func:`~rsgispy.imagecalc.rescale.img_norm_params_node` in a second pass,
without writing the node to a file.

Training samples
~~~~~~~~~~~~~~~~

:class:`~rsgispy.imageprocess.SampleSink` takes a feature node and a
single-band label node (class IDs, with the label no-data value, e.g. ``0``,
for unlabelled pixels) and returns a pandas ``DataFrame`` of the
labelled pixels in the layout of the classification sample tables
(``class_name``, ``out_cls_id``, ``internal_cls_id``, ``x``, ``y`` and one
column per feature band, named after the band names). A pixel is sampled
when its label and every feature are valid. With ``max_per_class`` at most
that many pixels are kept per class, chosen by a pseudo-random key of each
pixel's position and ``rnd_seed``, so the sample is the same for any block
size and ``n_jobs`` and memory use stays bounded.

.. code-block:: python

   from sklearn.ensemble import RandomForestClassifier
   from rsgispy.imageprocess import RasterSource, SampleSink, StackBands, TileGraph
   from rsgispy.imagecalc.calcindices import calc_indices_node
   from rsgispy.imagefilter.stats import stddev_filter_node

   idx = calc_indices_node(RasterSource("s2.tif"),
                           {2: "Green", 3: "Red", 4: "NIR"}, ["NDVI", "NDWI"])
   feats = StackBands([idx, stddev_filter_node(idx, 5)],
                      band_names=["NDVI", "NDWI", "NDVI_sd", "NDWI_sd"])
   samples = TileGraph(
       SampleSink(feats, RasterSource("train_labels.tif"),
                  max_per_class=5000, name="samples")
   ).run()["samples"]
   cols = ["NDVI", "NDWI", "NDVI_sd", "NDWI_sd"]
   clf = RandomForestClassifier().fit(samples[cols].to_numpy(),
                                      samples["out_cls_id"])

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

The numba kernels that run in parallel themselves (the mode and majority
filters, GLCM texture, the shadow rays of the illumination functions and
image co-registration) need a thread-safe numba threading layer (TBB or
OpenMP) to be launched from several threads at once. With the ``workqueue``
layer, used when neither is installed, rsgispy runs them one at a time, so
the results are the same but those steps do not speed up with ``n_jobs``;
``numba.threading_layer()`` reports the layer once a parallel kernel has run.

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
