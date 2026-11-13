rsgispy.dataaccess
==================

The :mod:`rsgispy.dataaccess` subpackage provides clients for searching and
downloading geospatial and Earth observation datasets from Copernicus Data Space
Ecosystem (OData), NASA CMR, OpenStreetMap, Planet Data/Orders API, and USGS M2M.

Satellite & Earth Observation Catalogues
----------------------------------------

* :doc:`dataaccess/copernicus_odata` — Copernicus Data Space Ecosystem OData search and download.
* :doc:`dataaccess/nasa_cmr` — NASA Common Metadata Repository (CMR) granule search and download.
* :doc:`dataaccess/planet_data_api` — Planet Data Search and Orders API v2 client utilities.
* :doc:`dataaccess/usgs_m2m` — USGS Machine-to-Machine (M2M) Landsat/EarthExplorer client.

.. toctree::
   :maxdepth: 1

   dataaccess/copernicus_odata
   dataaccess/nasa_cmr
   dataaccess/planet_data_api
   dataaccess/usgs_m2m

Vector & Open Data
------------------

* :doc:`dataaccess/openstreetmap` — OpenStreetMap Overpass API vector extraction utilities.

.. toctree::
   :maxdepth: 1

   dataaccess/openstreetmap
