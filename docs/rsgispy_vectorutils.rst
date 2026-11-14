rsgispy.vectorutils
===================

The :mod:`rsgispy.vectorutils` subpackage provides vector inspection,
reprojection, merging, splitting, and spatial/attribute selection utilities
using ``geopandas``, ``pyogrio``, ``shapely``, ``rasterio``, and ``pyproj``
without ``osgeo`` dependencies.

Examples
--------

Inspecting metadata, reprojecting, and selecting features spatially:

.. code-block:: python

   import rsgispy.vectorutils.io
   import rsgispy.vectorutils.metadata
   import rsgispy.vectorutils.selection

   bbox = rsgispy.vectorutils.metadata.get_vec_lyr_bbox("polys.gpkg", "polys")
   rsgispy.vectorutils.io.reproj_vec_lyr(
       "polys.gpkg", "polys_utm.gpkg", out_vec_lyr="polys", epsg_code=32630
   )
   rsgispy.vectorutils.selection.spatial_select(
       "polys_utm.gpkg", "polys", "roi.gpkg", "roi", "polys_roi.gpkg", "polys_roi"
   )

Merging and splitting vector layers by attribute:

.. code-block:: python

   import rsgispy.vectorutils.io

   rsgispy.vectorutils.io.merge_vec_files(
       ["tile1.gpkg", "tile2.gpkg"], "merged.gpkg", out_vec_lyr="merged"
   )
   rsgispy.vectorutils.io.split_by_attribute(
       "merged.gpkg", "merged", "region_code", multi_layers=True,
       out_vec_file="regions.gpkg",
   )

Metadata & Inspection
---------------------

* :doc:`vectorutils/metadata` — Projections, standardised bounding boxes (``get_vec_lyr_bbox``), feature counts, columns, layer lists, and file deletion.

.. toctree::
   :maxdepth: 1

   vectorutils/metadata

Translation, Reprojection, Merging & Splitting
----------------------------------------------

* :doc:`vectorutils/io` — Format translation, copy, reprojection, UTM/WGS84 wrapping, layer merging, and attribute/chunk splitting.

.. toctree::
   :maxdepth: 1

   vectorutils/io

Selection, Matching & Sampling
------------------------------

* :doc:`vectorutils/selection` — Spatial/attribute selection, deduplication, nearest-point matching, and train/test/validation splitting.

.. toctree::
   :maxdepth: 1

   vectorutils/selection
