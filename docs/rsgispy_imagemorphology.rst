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

API Reference
-------------

.. automodule:: rsgispy.imagemorphology
   :members:
   :undoc-members:
   :show-inheritance:
