rsgispy.elevation
=================

The :mod:`rsgispy.elevation` subpackage provides terrain and digital elevation
model (DEM) analysis utilities using ``rasterio`` and ``numpy``.

Terrain Derivatives & Illumination
----------------------------------

* :doc:`elevation/terrain` — Slope, aspect, and aspect categorisation.
* :doc:`elevation/illumination` — Hillshade, shadow masks, local incidence and exitance angles.
* :doc:`elevation/roughness` — Surface roughness profile and Smith roughness metrics.

.. toctree::
   :maxdepth: 1

   elevation/terrain
   elevation/illumination
   elevation/roughness

DEM Filtering & Fusion
----------------------

* :doc:`elevation/filtering` — Aspect median filtering, depression filling, and detrending.
* :doc:`elevation/fusion` — FFT-based DEM fusion.

.. toctree::
   :maxdepth: 1

   elevation/filtering
   elevation/fusion
