rsgispy.changedetect
====================

The :mod:`rsgispy.changedetect` subpackage provides multi-temporal change
detection methods and per-class pixel outlier change detection using
``rasterio`` and ``numpy``.

.. seealso::

   Guide: :doc:`guides/change_detection`

   Tutorials: :tutorial-dir:`section 6, change detection <06_change_detection>` (:doc:`tutorials`)

Chaining
--------

:func:`~rsgispy.changedetect.imgchngmthds.image_difference_node` and
:func:`~rsgispy.changedetect.imgchngmthds.image_ratio_node` take two input
nodes (and an optional valid-mask node), so the dates can be processed (e.g.
filtered or converted to indices) in the same in-memory
:class:`~rsgispy.imageprocess.TileGraph` pass. Standardisation needs
whole-image statistics: give them as ``std_params``
(``image_difference`` / ``image_ratio`` with ``apply_std=True`` calculate
them in a first pass).

.. code-block:: python

   from rsgispy.imageprocess import DiskSink, RasterSource, TileGraph
   from rsgispy.imagecalc.calcindices import calc_indices_node
   from rsgispy.changedetect.imgchngmthds import image_difference_node

   bands = {3: "red", 4: "nir"}
   ndvi1 = calc_indices_node(RasterSource("t1.tif"), bands, ["NDVI"])
   ndvi2 = calc_indices_node(RasterSource("t2.tif"), bands, ["NDVI"])
   dndvi = image_difference_node(
       ndvi1, ndvi2, base_img_no_data=-999, chng_img_no_data=-999, no_data_val=-999
   )
   TileGraph(DiskSink(dndvi, "dndvi.tif")).run()

CVA, ISFA and IR-MAD need whole-image statistics and stay file-based. See
:ref:`tilegraph-chainable` for the list of chainable operations.

Submodules
----------

* :doc:`changedetect/imgchngmthds` — Image differencing, ratioing, Change Vector Analysis (CVA), Slow Feature Analysis (ISFA), and Multivariate Alteration Detection (IR-MAD).
* :doc:`changedetect/pxloutlierchng` — Per-class outlier change detection using PyOD estimators, Otsu thresholding, Li's cross-entropy thresholding, and kurtosis/skewness optimisation.

.. toctree::
   :maxdepth: 1

   changedetect/imgchngmthds
   changedetect/pxloutlierchng
