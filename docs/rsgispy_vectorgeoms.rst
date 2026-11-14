rsgispy.vectorgeoms
===================

The :mod:`rsgispy.vectorgeoms` subpackage provides geometric operations on
vector datasets using ``geopandas``, ``shapely``, ``pyogrio``, and ``rtree``
without ``osgeo`` dependencies.

Examples
--------

Clipping and overlaying vector layers:

.. code-block:: python

   import rsgispy.vectorgeoms.overlay

   rsgispy.vectorgeoms.overlay.clip_vec_lyr(
       "roads.gpkg", "roads", "roi.gpkg", "roi", "roads_clip.gpkg", "roads"
   )
   rsgispy.vectorgeoms.overlay.vec_lyr_intersection_gp(
       "a.gpkg", "a", "b.gpkg", "b", "inter.gpkg", "inter"
   )

Generating orthogonal transects and a regular spatial grid:

.. code-block:: python

   import rsgispy.vectorgeoms.create
   import rsgispy.vectorgeoms.lines

   rsgispy.vectorgeoms.lines.create_orthg_lines(
       "coast.gpkg", "coast", "transects.gpkg", "transects",
       pt_step=50.0, line_len=100.0
   )
   rsgispy.vectorgeoms.create.define_grid(
       (500000.0, 501000.0, 5600000.0, 5601000.0),
       100.0, 100.0, 32630, "grid.gpkg", "grid"
   )

Geometry & Grid Creation
------------------------

* :doc:`vectorgeoms/create` — Polygonisation, pixel-to-point vectorisation, image footprints, bounding-box/WGS84/H3 grids, and random point layers.

.. toctree::
   :maxdepth: 1

   vectorgeoms/create

Line Geometries & Transects
---------------------------

* :doc:`vectorgeoms/lines` — Line creation, image transects, orthogonal transects, regular point splitting, boundary intersections, and coastline prediction.

.. toctree::
   :maxdepth: 1

   vectorgeoms/lines

Overlays & Clipping
-------------------

* :doc:`vectorgeoms/overlay` — Vector intersection, difference, symmetric difference, identity, union, and ROI clipping.

.. toctree::
   :maxdepth: 1

   vectorgeoms/overlay

Polygons & Transformations
--------------------------

* :doc:`vectorgeoms/polygons` — Polygon boundaries, centroids, exploding, simplification, hole removal, buffering, dissolving, and thinning.

.. toctree::
   :maxdepth: 1

   vectorgeoms/polygons

Spatial Queries, Indexing & Alpha Shapes
----------------------------------------

* :doc:`vectorgeoms/spatial` — Vertex extraction, concave hulls, bounding-box queries, R-tree indexing, and spatial predicates.

.. toctree::
   :maxdepth: 1

   vectorgeoms/spatial
