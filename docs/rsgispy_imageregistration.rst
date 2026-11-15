rsgispy.imageregistration
=========================

.. automodule:: rsgispy.imageregistration
   :no-index:

Submodules
----------

* :doc:`imageregistration/shift` — Whole-image offset estimation via :func:`~rsgispy.imageregistration.shift.find_image_offset` (correlation, mutual information, phase correlation, distance metrics) and linear offset application.
* :doc:`imageregistration/tiepoints` — Phase-correlation and template-matching tie-point generation, geometric model fitting, and RANSAC / residual outlier filtering.
* :doc:`imageregistration/warping` — GCP embedding, vector-point GCP conversion, :func:`~rsgispy.imageregistration.warping.warp_with_gcps`, polynomial/shift warping, and end-to-end co-registration.

.. toctree::
   :maxdepth: 1

   imageregistration/shift
   imageregistration/tiepoints
   imageregistration/warping
