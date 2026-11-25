rsgispy.changedetect
====================

The :mod:`rsgispy.changedetect` subpackage provides multi-temporal change
detection methods and per-class pixel outlier change detection using
``rasterio`` and ``numpy``.

.. seealso::

   Guide: :doc:`guides/change_detection`

Submodules
----------

* :doc:`changedetect/imgchngmthds` — Image differencing, ratioing, Change Vector Analysis (CVA), Slow Feature Analysis (ISFA), and Multivariate Alteration Detection (IR-MAD).
* :doc:`changedetect/pxloutlierchng` — Per-class outlier change detection using PyOD estimators, Otsu thresholding, Li's cross-entropy thresholding, and kurtosis/skewness optimisation.

.. toctree::
   :maxdepth: 1

   changedetect/imgchngmthds
   changedetect/pxloutlierchng
