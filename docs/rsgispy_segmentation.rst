rsgispy.segmentation
====================

.. automodule:: rsgispy.segmentation
   :no-index:

Overview
--------

.. seealso::

   The :doc:`guides/segment_ml_workflow` guide shows how to use the segments
   for object-based classification and regression.

The :mod:`rsgispy.segmentation` package provides pure-Python image segmentation
algorithms designed for multi-band remote sensing imagery (e.g., Sentinel-2,
Landsat, and multi-spectral orthomosaics) with:

* **Bounded-memory tiled execution**: Configurable ``tile_size`` and halo
  ``overlap`` padding with cross-tile seam stitching (``stitch_seams=True``).
* **Multi-core parallelism**: Parallel per-tile execution via ``n_jobs``
  with deterministic row-major output ordering.
* **Optional ``numba`` JIT acceleration**: Compiled hot loops via
  ``use_numba=True`` with automatic pure-Python fallback when ``numba`` is not
  installed.
* **Downstream GEOBIA integration**: Segment clean-up
  (:func:`~rsgispy.segmentation.utils.eliminate_small_segments`,
  :func:`~rsgispy.segmentation.utils.relabel_segments`), polygonisation
  (:func:`~rsgispy.segmentation.utils.polygonise_segments`, which uses the
  tiled, multi-core
  :func:`~rsgispy.vector.create.polygonise_raster_tiled`; see
  :ref:`polygonise-benchmark`), and zonal band statistics
  (:func:`~rsgispy.segmentation.utils.extract_segment_stats`).

GEOBIA Workflow Example
-----------------------

.. code-block:: python

   import rsgispy
   from rsgispy.segmentation import (
       shepherd_segmentation,
       polygonise_segments,
       extract_segment_stats,
   )

   # 1. Tiled, parallel Shepherd segmentation with halo overlap & seam stitching
   shepherd_segmentation(
       "sentinel2_4band.tif",
       "segments.tif",
       n_clusters=60,
       min_pixels=100,
       tile_size=1024,
       overlap=32,
       n_jobs=4,
       gdalformat="GTIFF",
       datatype=rsgispy.TYPE_32UINT,
   )

   # 2. Polygonise the segments (tiled, 4 worker processes), one feature per
   #    connected region sorted by segment ID
   polygonise_segments("segments.tif", "segments.gpkg", n_jobs=4)

   # 3. Polygonise segments and summarise mean/stddev/min/max per band
   extract_segment_stats(
       segments_img="segments.tif",
       feature_img="sentinel2_4band.tif",
       out_vec_file="segments_stats.gpkg",
       out_vec_lyr="segments",
   )

Submodules
----------

* :doc:`segmentation/clump` — Connected-component clumping (:func:`~rsgispy.segmentation.clump.clump`) and union of clumps (:func:`~rsgispy.segmentation.clump.union_of_clumps`).
* :doc:`segmentation/shepherd` — Shepherd et al. (2019) iterative K-means and small-segment elimination (:func:`~rsgispy.segmentation.shepherd.shepherd_segmentation`).
* :doc:`segmentation/region_merge` — Baatz & Schäpe multiresolution region merging / FNEA (:func:`~rsgispy.segmentation.region_merge.region_merge_segmentation`).
* :doc:`segmentation/slic` — Global grid-aligned SLIC superpixels (:func:`~rsgispy.segmentation.slic.slic_segmentation`).
* :doc:`segmentation/graph` — Felzenszwalb–Huttenlocher graph-based segmentation (:func:`~rsgispy.segmentation.graph.graph_segmentation`).
* :doc:`segmentation/watershed` — Seeded and auto-seeded compact watershed segmentation (:func:`~rsgispy.segmentation.watershed.watershed_segmentation`).
* :doc:`segmentation/tiling` — Tiled halo planning, ID offsetting, and cross-tile seam stitching (:class:`~rsgispy.segmentation.tiling.TileWindow`, :func:`~rsgispy.segmentation.tiling.plan_tiles`).
* :doc:`segmentation/utils` — Post-processing, optional ``numba`` checks, polygonisation, and zonal object statistics.

.. toctree::
   :maxdepth: 1

   segmentation/clump
   segmentation/shepherd
   segmentation/region_merge
   segmentation/slic
   segmentation/graph
   segmentation/watershed
   segmentation/tiling
   segmentation/utils
