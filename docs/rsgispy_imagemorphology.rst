rsgispy.imagemorphology
=======================

The :mod:`rsgispy.imagemorphology` module provides mathematical morphology
operations for raster images using ``rasterio`` and ``numpy``.

Example
-------

Creating a circular structuring element and dilating a raster image:

.. code-block:: python

   import rsgispy
   import rsgispy.imagemorphology

   rsgispy.imagemorphology.create_circular_op("circle_5x5.gmtxt", op_size=5)
   rsgispy.imagemorphology.image_dilate(
       input_img="input_mask.tif",
       output_img="dilated_mask.tif",
       morph_op_file="circle_5x5.gmtxt",
       use_op_file=True,
       op_size=5,
       gdalformat="GTIFF",
       datatype=rsgispy.TYPE_8UINT,
   )

Image borders
-------------

The keyword-only ``border_mode`` option of the dilation, erosion, gradient,
opening, closing and top-hat functions (and of their node builders) sets how
the structuring element is filled beyond the image edge: ``'edge'`` (the
default, repeat the edge pixels), ``'reflect'``, ``'symmetric'`` or
``'constant'`` with ``border_fill`` (e.g. ``border_fill=0`` to treat the
area beyond the edge as background, so that an erosion or opening shrinks
objects that touch the edge). See
:ref:`filter-border-mode`. The local minima always leave out-of-image
neighbours out and have no ``border_mode``.

Chaining
--------

Every operation also has a node builder, named after the function with
``_node`` (``image_dilate_node``, ``image_erode_node``,
``image_gradient_node``, ``image_local_minima_node``,
``image_opening_node``, ``image_closing_node``,
``image_white_top_hat_node``, ``image_black_top_hat_node``; the
``*_combined_out`` variants are ``combined=True``), for chaining morphology
with filters, band maths or a majority filter in one
:class:`~rsgispy.imageprocess.TileGraph` pass without intermediate files.
The builders take the structuring element as the file functions do
(``morph_op_file``, ``use_op_file``, ``op_size``). ``no_data_val`` is the
input no-data value (by default that of the source node; see
:ref:`tilegraph-nodata`) and ``dtype``
the NumPy type the result will be written as: openings, closings and top-hats
round each step to an integer ``dtype``, as the file functions do. The
file-based functions run the same nodes (``RasterSource -> node ->
DiskSink``), so the results are identical.

Sequentially numbered local minima (``sequencial_out=True``) are numbered in
raster scan order, which depends on the whole image, so they are only
available from :func:`~rsgispy.imagemorphology.image_local_minima` and
:func:`~rsgispy.imagemorphology.image_local_minima_combined_out`.

.. code-block:: python

   import numpy as np
   import rsgispy
   from rsgispy.imageprocess import DiskSink, RasterSource, TileGraph
   from rsgispy.imagemorphology import image_closing_node, image_opening_node

   mask = RasterSource("cloud_mask.tif")
   # Remove small specks, then fill small holes, in one pass.
   clean = image_closing_node(
       image_opening_node(mask, op_size=3, dtype=np.uint8),
       op_size=5,
       dtype=np.uint8,
   )
   TileGraph(
       DiskSink(clean, "cloud_mask_clean.tif", datatype=rsgispy.TYPE_8UINT)
   ).run()

API Reference
-------------

.. automodule:: rsgispy.imagemorphology
   :members:
   :undoc-members:
   :show-inheritance:
