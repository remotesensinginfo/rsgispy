rsgispy.imageutils.composite
============================

Per-pixel composites of a stack of observations (e.g. the cloud-masked
scenes of a season) on the same pixel grid:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Function
     - Each output pixel is
   * - :func:`~rsgispy.imageutils.composite.create_geomedian_composite`
     - the multi-band geometric median of the valid observations (Roberts et
       al., 2017), with optional median absolute deviations, count and
       nearest-observation reference images.
   * - :func:`~rsgispy.imageutils.composite.create_quality_mosaic`
     - the whole observation with the largest, smallest or a percentile of
       a quality value (a band, a band-maths expression or a callable), like
       Google Earth Engine's ``qualityMosaic``.
       :func:`~rsgispy.imageutils.composite.create_max_ndvi_composite` is the
       maximum NDVI preset.
   * - :func:`~rsgispy.imageutils.composite.create_land_water_composite`
     - a quality mosaic with one quality over land and another (or the
       medoid or geomedian) over water, from a land / water mask or a
       per-observation vote.
       :func:`~rsgispy.imageutils.composite.create_max_ndvi_ndwi_composite`
       is the maximum NDVI / NDWI preset (RSGISLib's
       ``imageutils.imagecomp.create_max_ndvi_ndwi_composite``).
   * - :func:`~rsgispy.imageutils.composite.create_medoid_composite`
     - the valid observation nearest all the others (Flood, 2013).
   * - :func:`~rsgispy.imageutils.composite.create_percentile_composite`
     - per-band percentiles (e.g. the median) over the observations.

The common rules:

* **Valid observations**: an observation is valid at a pixel only when every
  composited band (``bands``) is valid there, and, for a quality given as a
  band number or an expression, every band the quality uses.
* **Empty pixels**: a pixel with no valid observation is no data in every
  output.
* **Reference image** (``out_ref_img``): a thematic ``uint16`` image of the
  1-based index into ``input_imgs`` of the observation used at each pixel
  (``0`` = none), with a colour table and statistics.
* **Grids**: the images must be on the same pixel grid, otherwise an
  :class:`rsgispy.RSGISPyException` is raised; resample them first with
  :func:`~rsgispy.imageutils.warp.resample_img_to_match`.
* **Processing** is block by block, and the results are identical for any
  ``block_size`` and number of threads (``n_jobs``).

Example
-------

.. code-block:: python

   import glob
   import rsgispy.imageutils.composite as composite

   imgs = sorted(glob.glob("ls_imgs/*.tif"))  # Landsat 8/9, no data -9999

   # Geomedian, with its deviations and the observation nearest it.
   composite.create_geomedian_composite(
       imgs, "geomedian.tif", out_mad_img="geomedian_mads.tif",
       out_ref_img="geomedian_ref.tif",
   )

   # Max. NDVI over land, the medoid over water, where NIR < 5 % in most
   # observations; isolated land / water pixels removed with a 3x3 filter.
   composite.create_max_ndvi_ndwi_composite(
       imgs, "lw_comp.tif", red_band=4, nir_band=5, green_band=3,
       water_rule="b5 < 500 ? 1 : 0", water_method="medoid",
       smooth_lw_msk_size=3, out_ref_img="lw_comp_ref.tif",
       out_lw_msk_img="lw_comp_msk.tif",
   )

   # Seasonal percentiles of every band, as classification features.
   composite.create_percentile_composite(imgs, "pcts.tif", [10, 50, 90])

Chaining
--------

Every composite has a tile-graph node builder taking one
:class:`~rsgispy.imageprocess.TileNode` per observation
(:func:`~rsgispy.imageutils.composite.geomedian_node`,
:func:`~rsgispy.imageutils.composite.quality_mosaic_node`,
:func:`~rsgispy.imageutils.composite.land_water_composite_node`,
:func:`~rsgispy.imageutils.composite.medoid_node` and
:func:`~rsgispy.imageutils.composite.percentile_composite_node`). A builder
returns a :class:`~rsgispy.imageutils.composite.CompositeNode` whose other
outputs (``ref``, ``quality``, ``count``, ``mad``, ``lw_msk``) are nodes
computed from the same tiles:

.. code-block:: python

   import glob
   from rsgispy.imageprocess import DiskSink, RasterSource, TileGraph
   from rsgispy.imageutils.composite import quality_mosaic_node

   srcs = [RasterSource(img) for img in sorted(glob.glob("ls_imgs/*.tif"))]
   comp = quality_mosaic_node(
       srcs, "(b5 - b4) / (b5 + b4)", "percentile", 90, smooth_ref_size=3
   )
   TileGraph(
       [DiskSink(comp, "ndvi_p90.tif"), DiskSink(comp.ref, "ndvi_p90_ref.tif")]
   ).run()

API Reference
-------------

.. automodule:: rsgispy.imageutils.composite
   :members:
   :undoc-members:
   :show-inheritance:
