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
  nodes, keeping band names) and :class:`~rsgispy.imageprocess.Cast`, and
  the filter node builders below;
* **sinks**: :class:`~rsgispy.imageprocess.DiskSink` and
  :class:`~rsgispy.imageprocess.MemorySink`.

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
was 512.

Chainable operations
~~~~~~~~~~~~~~~~~~~~

Each chainable operation has a public node builder in the module of its
file-based function, named after the function without ``apply_`` and with
``_node``. Builders take ``(source, <the function's processing parameters>,
*, no_data_val=None, name=None)`` and give results identical to the
file-based function. ``no_data_val`` is not read from the source: pass the
input's no-data value. The :class:`~rsgispy.imagefilter.chained.FilterStep`
kinds call the same builders.

.. list-table::
   :header-rows: 1
   :widths: 40 40 20

   * - File-based function
     - Node builder
     - ``FilterStep`` kind
   * - ``rsgispy.imagefilter.stats.apply_<stat>_filter`` (``median``,
       ``mean``, ``min``, ``max``, ``mode``, ``stddev``, ``range``,
       ``total``, ``mean_diff``, ``mean_diff_abs``, ``total_diff``,
       ``total_diff_abs``, ``coeff_of_var``, ``norm_var``,
       ``norm_var_sqrt``, ``norm_var_ln``)
     - ``rsgispy.imagefilter.stats.<stat>_filter_node``
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
   * - :func:`~rsgispy.imagefilter.edge.apply_sobel_filter`
     - :func:`~rsgispy.imagefilter.edge.sobel_filter_node`
     - ``'sobel'``
   * - :func:`~rsgispy.imagefilter.edge.apply_prewitt_filter`
     - :func:`~rsgispy.imagefilter.edge.prewitt_filter_node`
     - ``'prewitt'``
   * - :func:`~rsgispy.imagefilter.edge.apply_laplacian_filter`
     - :func:`~rsgispy.imagefilter.edge.laplacian_filter_node`
     - ``'laplacian'``
   * - :func:`~rsgispy.imagefilter.sar.apply_lee_filter`
     - :func:`~rsgispy.imagefilter.sar.lee_filter_node`
     - ``'lee'``

Operations that cannot be nodes, because an output pixel does not depend
only on the input pixels within a fixed radius:

* **whole-image operations**: segmentation (:mod:`rsgispy.segmentation`),
  hydrological flow routing and accumulation, distance transforms
  (``calc_dist_to_img_vals``), ``apply_sieve_filter`` (connected
  components), region-growing fills and ``shadow_mask`` (rays of unbounded
  length);
* **operations that change the pixel grid**: warping, resampling and
  reprojection;
* **operations that need whole-image parameters**: ``calc_pca``,
  ``calc_mnf``, ``whiten_image``, ``histogram_matching`` and normalisation
  without given parameters. Compute the parameters first, then apply them
  pixel by pixel;
* **optional backends**: unmixing with ``pysptools`` or ``pyMCR``.

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
