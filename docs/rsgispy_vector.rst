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

Calculate multiple new attribute columns in one pass using comma-separated
``pymu-eval`` expressions with :func:`~rsgispy.vector.attrs.vector_maths`
(each top-level comma-separated sub-expression maps positionally to the
corresponding entry in ``out_col``, while commas inside function calls such as
``min(blue, green, red)`` remain function arguments, and new intermediate
variables (e.g., ``ndvi = ...``) can be defined and reused by later outputs):

.. code-block:: python

   import numpy as np
   import rsgispy.vector.attrs

   # Multi-output expression defining and reusing intermediate `ndvi`
   exp = """
       ndvi = (nir + red) != 0 ? (nir - red) / (nir + red) : -9999,
       avg(blue, green, red),
       ndvi >= 0.5 ? 1 : 0
   """
   out_gdf = rsgispy.vector.attrs.vector_maths(
       "samples.gpkg",
       "samples",
       out_col=["ndvi", "vis_mean", "is_veg"],
       exp=exp,
       out_dtype=[np.float32, np.float32, int],
   )

   # Alias columns with spaces/hyphens and reuse scaled values across outputs
   out_gdf = rsgispy.vector.attrs.vector_maths(
       "samples.gpkg",
       "samples",
       out_col=["nir_refl", "red_refl", "ndvi"],
       exp="nir = nir * 0.0001, red = red * 0.0001, (nir - red) / (nir + red)",
       vec_cols={"nir": "Band 4", "red": "Band 3"},
   )

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
