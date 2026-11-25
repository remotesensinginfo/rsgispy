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
   from rsgispy.imagefilter.chained import FilterStep, build_filter_chain_node

   src = RasterSource("input.tif")
   smooth = build_filter_chain_node(
       src, [FilterStep("median", filter_size=3), FilterStep("mean", filter_size=5)]
   )
   diff = BandMathOp({"smooth": smooth, "orig": src}, "smooth - orig")
   sink = DiskSink(diff, "highpass.tif", datatype=rsgispy.TYPE_32FLOAT)
   TileGraph(sink).run()

API Reference
-------------

.. automodule:: rsgispy.imageprocess
   :members:
   :undoc-members:
   :show-inheritance:
