rsgispy.elevation.hydrology
===========================

The :mod:`rsgispy.elevation.hydrology` module provides a pure-Python hydrological
analysis suite covering DEM conditioning (sink identification, depression
filling and breaching, stream burning, flat resolution), flow routing (D8,
D-Infinity and MFD) and accumulation, flow lengths, distances and paths,
stream network extraction, ordering and vectorisation, watershed, basin and
subcatchment delineation, Height Above Nearest Drainage (HAND) and flood
inundation, and terrain wetness/erosivity indices (TWI, SPI and the RUSLE LS
factor).

.. important::

   All functions in :mod:`rsgispy.elevation.hydrology` require input rasters in
   a **projected coordinate reference system with linear units in metres** (for
   example, UTM). Geographic (latitude/longitude) rasters raise
   :class:`~rsgispy.RSGISPyException` and should be reprojected beforehand with
   :func:`rsgispy.imageutils.warp.reproject_image`.

.. seealso::

   The guide :doc:`../guides/hydrology_workflow` walks through a complete
   catchment analysis with these functions, from a raw DEM to conditioned
   elevation, streams, gauged catchments with zonal statistics, HAND flood
   maps, terrain indices and a map, and explains how to choose the routing
   method, stream threshold, snapping distance and breaching limits.

API Reference
-------------

.. automodule:: rsgispy.elevation.hydrology
   :members:
   :undoc-members:
   :show-inheritance:
