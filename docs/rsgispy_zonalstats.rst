rsgispy.zonalstats
==================

The :mod:`rsgispy.zonalstats` subpackage provides zonal statistics extraction
modules for polygon and point vector datasets and for zones defined by a
label raster (e.g., segments) using ``rasterio`` and ``geopandas``.

Modules
-------

* :doc:`zonalstats/polygon` — Polygon zonal statistics and pixel counts.
* :doc:`zonalstats/point` — Point band value extraction.
* :doc:`zonalstats/labels` — Fast per-label statistics from a label
  (segment/clump) raster in one streamed pass, joining label statistics to a
  vector layer, and rasterising per-label values.

.. toctree::
   :maxdepth: 1

   zonalstats/polygon
   zonalstats/point
   zonalstats/labels
