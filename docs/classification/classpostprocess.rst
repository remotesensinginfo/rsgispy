rsgispy.classification.classpostprocess
=======================================

Chaining
--------

:func:`~rsgispy.classification.classpostprocess.majority_filter_node` is the
tile-graph node of the majority filter, for smoothing a classification as
the last step of a :class:`~rsgispy.imageprocess.TileGraph`, e.g. directly
after :func:`~rsgispy.ml.predict.predict_raster_node`, without writing the
raw classification. It filters every band of its source (select a band with
:class:`~rsgispy.imageprocess.SelectBands`); pass the input no-data value
and the output ``dtype``. :func:`~rsgispy.classification.classpostprocess.apply_majority_filter`
and the majority passes of
:func:`~rsgispy.classification.classpostprocess.apply_postprocess_pipeline`
run the same node. The sieve filter labels connected components across the
whole image, so it is not tile-local and has no node.

.. code-block:: python

   import numpy as np
   import rsgispy
   from rsgispy.classification.classpostprocess import majority_filter_node
   from rsgispy.imageprocess import DiskSink, RasterSource, TileGraph

   cls = RasterSource("classified.tif", bands=1)
   smooth = majority_filter_node(
       majority_filter_node(cls, 3, dtype=np.uint8, no_data_val=0),
       5,
       dtype=np.uint8,
       no_data_val=0,
   )
   TileGraph(DiskSink(smooth, "classified_smooth.tif", datatype=rsgispy.TYPE_8UINT)).run()

API Reference
-------------

.. automodule:: rsgispy.classification.classpostprocess
   :members:
   :undoc-members:
   :show-inheritance:
