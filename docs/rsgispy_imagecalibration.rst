rsgispy.imagecalibration
========================

The :mod:`rsgispy.imagecalibration` subpackage provides pure-Python radiometric
calibration, solar-angle modelling, Dark Object Subtraction (DOS) atmospheric
correction, and Level-1/Level-2 satellite product ingestion for Landsat
Collection 2 and ESA Sentinel-2 L2A SAFE archives using ``rasterio`` and
``numpy``.

Radiometry & DOS
----------------

* :doc:`imagecalibration/radiometry` — Julian day, Earth–Sun distance,
  ESUN calculation, visibility-to-AOT conversion, TOA reflectance, Landsat
  thermal brightness temperature, and Dark Object Subtraction (DOS).

.. toctree::
   :maxdepth: 1

   imagecalibration/radiometry

Solar Geometry
--------------

* :doc:`imagecalibration/solarangles` — Solar azimuth convention helpers and
  per-pixel solar azimuth/zenith raster calculation.

.. toctree::
   :maxdepth: 1

   imagecalibration/solarangles

Sensor Level-1 & Level-2 Data
-----------------------------

* :doc:`imagecalibration/sensorlvl1data` — Landsat Collection 2 Level-1 (OLI)
  TOA reflectance stacking and masking.
* :doc:`imagecalibration/sensorlvl2data` — Landsat Collection 2 ``QA_PIXEL``
  parsing, Landsat Collection 2 Level-2 (OLI and TM/ETM+) surface reflectance
  stacking, and ESA Sentinel-2 Level-2A SAFE stacking and sharpening.

.. toctree::
   :maxdepth: 1

   imagecalibration/sensorlvl1data
   imagecalibration/sensorlvl2data
