rsgispy.vectorprocess
=====================

The :mod:`rsgispy.vectorprocess` module provides shared low-level vector processing utilities for validating, reading, writing, casting, and inspecting vector datasets using ``geopandas``, ``pyogrio``, ``shapely``, and ``pyproj``.

Reading and Inspecting Vector Data Example
------------------------------------------

.. code-block:: python

   import rsgispy.vectorprocess

   lyr_name = rsgispy.vectorprocess.resolve_vec_layer("polys.gpkg")
   gdf = rsgispy.vectorprocess.read_vec_gdf("polys.gpkg", vec_lyr=lyr_name)
   bbox = rsgispy.vectorprocess.get_gdf_bbox(gdf)
   geom_type = rsgispy.vectorprocess.get_gdf_geom_type(gdf)

API Reference
-------------

.. automodule:: rsgispy.vectorprocess
   :members:
   :undoc-members:
   :show-inheritance:
