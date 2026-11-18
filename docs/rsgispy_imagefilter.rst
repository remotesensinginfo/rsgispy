rsgispy.imagefilter
===================

The :mod:`rsgispy.imagefilter` subpackage provides spatial moving-window and
convolution filters for raster images using ``rasterio`` and ``numpy``,
organised into thematic submodules:

* :doc:`imagefilter/stats` — Statistical, dispersion, and local-difference moving-window filters.
* :doc:`imagefilter/smooth` — Edge-preserving Kuwahara, Gaussian smoothing, and custom 2D convolution kernel filters.
* :doc:`imagefilter/edge` — Sobel, Prewitt, Gaussian derivative, and Laplacian edge-detection filters.
* :doc:`imagefilter/sar` — Lee, Enhanced Lee, Frost, Enhanced Frost, Kuan, Gamma MAP, and Refined Lee SAR speckle filters.
* :doc:`imagefilter/texture` — Gray-Level Co-occurrence Matrix (GLCM) spatial texture filters.

For convenience and backwards compatibility, all filter functions are also
re-exported directly under :mod:`rsgispy.imagefilter`.

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

Submodules
----------

.. toctree::
   :maxdepth: 1

   imagefilter/stats
   imagefilter/smooth
   imagefilter/edge
   imagefilter/sar
   imagefilter/texture
