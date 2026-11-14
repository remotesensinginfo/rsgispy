rsgispy.vectorattrs
===================

The :mod:`rsgispy.vectorattrs` subpackage provides vector attribute table and
spatial attribute utilities implemented using ``geopandas`` without ``osgeo``
dependencies.

Examples
--------

Adding a sequential FID column and geometry bounding-box columns:

.. code-block:: python

   import rsgispy.vectorattrs.columns
   import rsgispy.vectorattrs.spatial

   rsgispy.vectorattrs.columns.add_fid_col(
       "polys.gpkg", "polys", "polys_fid.gpkg", "polys", out_col="feat_id"
   )
   rsgispy.vectorattrs.spatial.add_geom_bbox_cols(
       "polys_fid.gpkg", "polys", "polys_bbox.gpkg", "polys"
   )

Reading and writing attribute columns:

.. code-block:: python

   import rsgispy.vectorattrs.io

   vals = rsgispy.vectorattrs.io.read_vec_column("polys.gpkg", "polys", "class_id")
   rsgispy.vectorattrs.io.write_vec_column(
       "polys.gpkg", "polys", "scaled_id", int, [v * 10 for v in vals]
   )

Attribute I/O & Tabular Management
----------------------------------

* :doc:`vectorattrs/io` — Read, write, drop, rename, sort, and export attribute columns.

.. toctree::
   :maxdepth: 1

   vectorattrs/io

Derived Columns, Lookups & Text Cleaning
----------------------------------------

* :doc:`vectorattrs/columns` — Add constant, LUT, random, name, date, string, and angle columns.

.. toctree::
   :maxdepth: 1

   vectorattrs/columns

Spatial Bounding Boxes & Joins
------------------------------

* :doc:`vectorattrs/spatial` — Geometry bounding-box columns, spatial selection annotation, and spatial attribute joins.

.. toctree::
   :maxdepth: 1

   vectorattrs/spatial
