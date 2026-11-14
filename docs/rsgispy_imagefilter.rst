rsgispy.imagefilter
===================

The :mod:`rsgispy.imagefilter` module provides spatial moving-window statistical,
edge-detection, Gaussian derivative, Kuwahara, Lee SAR speckle, and custom 2D
convolution kernel filters for raster images using ``rasterio`` and ``numpy``.

Example
-------

Applying a median smoothing filter and a 3x3 Sobel edge-magnitude filter:

.. code-block:: python

   import rsgispy
   import rsgispy.imagefilter

   rsgispy.imagefilter.apply_median_filter(
       input_img="input.tif",
       output_img="median_3x3.tif",
       filter_size=3,
       gdalformat="GTIFF",
       datatype=rsgispy.TYPE_32FLOAT,
   )
   rsgispy.imagefilter.apply_sobel_filter(
       input_img="input.tif",
       output_img="sobel.tif",
   )

API Reference
-------------

.. automodule:: rsgispy.imagefilter
   :members:
   :undoc-members:
   :show-inheritance:
