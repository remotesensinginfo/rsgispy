rsgispy.imagespectral.unmixing
==============================

.. tip::

   **Recommended spectral unmixing functions:** Use the pure
   ``numpy`` / ``scipy`` implementations
   :func:`~rsgispy.imagespectral.unmixing.spec_unmix_ucls` (NumPy only),
   :func:`~rsgispy.imagespectral.unmixing.spec_unmix_nnls` (SciPy), and
   :func:`~rsgispy.imagespectral.unmixing.spec_unmix_fcls` (SciPy).
   They avoid heavy or compiled optional dependencies (``pysptools``,
   ``cvxopt``, ``pyMCR``), solve exact least-squares / NNLS / FCLS objectives
   directly, and provide significantly faster performance. The optional
   ``spec_unmix_spts_*`` and ``spec_unmix_pymcr_*`` functions remain available
   for workflows that explicitly target those backends.

.. automodule:: rsgispy.imagespectral.unmixing
   :members:
   :undoc-members:
   :show-inheritance:
