rsgispy.elevation
=================

The :mod:`rsgispy.elevation` subpackage provides terrain and digital elevation
model (DEM) analysis utilities using ``rasterio`` and ``numpy``.

.. seealso::

   Guide: :doc:`guides/hydrology_workflow`

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

Hydrological Analysis
---------------------

* :doc:`elevation/hydrology` — Sink identification, depression filling and breaching,
  stream burning, flat resolution, D8/D-Infinity/MFD flow routing, flow accumulation,
  stream ordering and vectorisation, watershed delineation, HAND flood inundation,
  TWI, SPI, and RUSLE LS factor.

.. toctree::
   :maxdepth: 1

   elevation/hydrology

