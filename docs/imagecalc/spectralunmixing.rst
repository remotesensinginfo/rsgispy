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

.. automodule:: rsgispy.imagecalc.spectralunmixing
   :members:
   :undoc-members:
   :show-inheritance:
