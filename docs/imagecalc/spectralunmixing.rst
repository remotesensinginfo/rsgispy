rsgispy.imagecalc.spectralunmixing
==================================

.. tip::

   **Recommended spectral unmixing functions:** Use the pure
   ``numpy`` / ``scipy`` implementations
   :func:`~rsgispy.imagecalc.spectralunmixing.spec_unmix_ucls` (NumPy only),
   :func:`~rsgispy.imagecalc.spectralunmixing.spec_unmix_nnls` (SciPy), and
   :func:`~rsgispy.imagecalc.spectralunmixing.spec_unmix_fcls` (SciPy).
   They avoid heavy or compiled optional dependencies (``pysptools``,
   ``cvxopt``, ``pyMCR``), solve exact least-squares / NNLS / FCLS objectives
   directly, and provide significantly faster performance. The optional
   ``spec_unmix_spts_*`` and ``spec_unmix_pymcr_*`` functions remain available
   for workflows that explicitly target those backends.

.. note::

   **Endmember representation:** Endmembers are stored and passed as
   :class:`pandas.DataFrame` tables with **spectral bands as rows** and
   **endmember class names as columns**. All unmixing, prediction, plotting,
   and evaluation functions accept either an in-memory ``pandas.DataFrame``
   or a file path in the user's chosen format (CSV, TSV, Parquet, Feather,
   Excel, JSON, Pickle, or HDF5 via
   :func:`~rsgispy.imagecalc.spectralunmixing.read_endmembers_df` and
   :func:`~rsgispy.imagecalc.spectralunmixing.write_endmembers_df`). Legacy
   RSGISLib ``.mtxt`` files can be migrated using
   :func:`~rsgispy.imagecalc.spectralunmixing.convert_mtxt_to_df`.

.. automodule:: rsgispy.imagecalc.spectralunmixing
   :members:
   :undoc-members:
   :show-inheritance:
