rsgispy.imagespectral.endmembers
================================

.. note::

   **Endmember representation:** Endmembers are stored and passed as
   :class:`pandas.DataFrame` tables with **spectral bands as rows** and
   **endmember class names as columns**. All unmixing, classification,
   prediction, plotting, and evaluation functions accept either an in-memory
   ``pandas.DataFrame`` or a file path in the user's chosen format (CSV, TSV,
   Parquet, Feather, Excel, JSON, Pickle, or HDF5 via
   :func:`~rsgispy.imagespectral.endmembers.read_endmembers_df` and
   :func:`~rsgispy.imagespectral.endmembers.write_endmembers_df`). Legacy
   RSGISLib ``.mtxt`` files can be migrated using
   :func:`~rsgispy.imagespectral.endmembers.convert_mtxt_to_df`.

.. automodule:: rsgispy.imagespectral.endmembers
   :members:
   :undoc-members:
   :show-inheritance:
