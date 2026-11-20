rsgispy.vector
==============

The :mod:`rsgispy.vector` subpackage consolidates vector I/O, geometry creation,
geometric transformations, attribute table management, spatial statistics, and
selection into six focused submodules.  Every function accepts either a vector
file path (``str`` / :class:`os.PathLike`) or an in-memory
:class:`geopandas.GeoDataFrame`.

Examples
--------

Inspect metadata, reproject, and select features using :class:`~rsgispy.BBox`:

.. code-block:: python

   import rsgispy.vector.io
   import rsgispy.vector.selection

   bbox = rsgispy.vector.io.get_vec_lyr_bbox("polys.gpkg", "polys")
   rsgispy.vector.io.reproj_vec_lyr(
       "polys.gpkg", "polys", "polys_wgs84.gpkg", "polys", out_epsg=4326
   )
   sub_gdf = rsgispy.vector.selection.spatial_select_bbox(
       "polys_wgs84.gpkg", "polys", bbox.reproject(4326)
   )

Create footprints, enrich attributes, and compute spatial metrics in memory:

.. code-block:: python

   import rsgispy.vector.attrs
   import rsgispy.vector.create
   import rsgispy.vector.stats

   fp_gdf = rsgispy.vector.create.extract_img_footprint("scene.tif")
   fp_gdf = rsgispy.vector.attrs.add_fid_col(fp_gdf, None, out_col="fid")
   fp_gdf = rsgispy.vector.stats.calc_vec_area(fp_gdf, None, "area_m2")

Submodules
----------

.. toctree::
   :maxdepth: 2

   vector/io
   vector/create
   vector/geoms
   vector/attrs
   vector/stats
   vector/selection
