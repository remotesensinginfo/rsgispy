rsgispy.imagespectral
=====================

The :mod:`rsgispy.imagespectral` subpackage provides spectral image processing,
classification, linear spectral unmixing, endmember table management, spectral
feature transformations, and pre-processing using ``rasterio``, ``numpy``, and
``pandas``.

.. seealso::

   Guide: :doc:`guides/spectral_unmixing`

Chaining
--------

The pixel-wise spectral functions can be used as nodes of an in-memory
:class:`~rsgispy.imageprocess.TileGraph` (the file-based functions are
wrappers around these builders and give the same result):

* :mod:`~rsgispy.imagespectral.transforms`: ``calc_tasseled_cap_node``, and
  ``pca_projection_node`` / ``mnf_projection_node`` to apply a transformation
  fitted by :func:`~rsgispy.imagespectral.transforms.calc_pca` or
  :func:`~rsgispy.imagespectral.transforms.calc_mnf` (fitting needs
  whole-image statistics; applying it is pixel-wise);
* :mod:`~rsgispy.imagespectral.classification`:
  ``spectral_angle_mapper_node``, ``spectral_correlation_mapper_node``
  (``output='scores'``, ``'best'``, ``'class'`` or ``'all'``);
* :mod:`~rsgispy.imagespectral.unmixing`: ``spec_unmix_ucls_node``,
  ``spec_unmix_nnls_node``, ``spec_unmix_fcls_node`` (given endmembers) and
  ``predict_refl_linear_unmixing_node``. The ``pysptools`` and ``pyMCR``
  unmixing functions stay file-based.

.. code-block:: python

   from rsgispy.imageprocess import DiskSink, RasterSource, TileGraph
   from rsgispy.imagespectral.unmixing import (
       predict_refl_linear_unmixing_node,
       spec_unmix_fcls_node,
   )

   fracs = spec_unmix_fcls_node(RasterSource("refl.tif"), "endmembers.csv")
   pred = predict_refl_linear_unmixing_node(fracs, "endmembers.csv")
   TileGraph([DiskSink(fracs, "fracs.tif"), DiskSink(pred, "pred_refl.tif")]).run()

See :ref:`tilegraph-chainable` for the list of chainable operations.

Submodules
----------

* :doc:`imagespectral/classification` — Spectral Angle Mapper (SAM) and Spectral Correlation Mapper (SCM) similarity mapping and classification.
* :doc:`imagespectral/endmembers` — Endmember ``pandas.DataFrame`` I/O, legacy ``.mtxt`` conversion, polygon extraction, plotting, equality checks, and Pixel Purity Index (PPI).
* :doc:`imagespectral/indices` — Spectral curve and index summary utilities, including area under the curve.
* :doc:`imagespectral/preprocessing` — Savitzky-Golay spectral smoothing, outlier masking, and polynomial NaN gap-filling.
* :doc:`imagespectral/transforms` — ZCA noise whitening, windowed/incremental PCA, Minimum Noise Fraction (MNF), Tasseled Cap transformations, and streaming histogram matching.
* :doc:`imagespectral/unmixing` — Unconstrained, non-negative, fully constrained, and exhaustive linear spectral unmixing, reflectance prediction, RMSE/RMSPE evaluation, and multi-endmember model summarisation.

.. toctree::
   :maxdepth: 1

   imagespectral/classification
   imagespectral/endmembers
   imagespectral/indices
   imagespectral/preprocessing
   imagespectral/transforms
   imagespectral/unmixing
