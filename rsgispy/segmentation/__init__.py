"""
The segmentation subpackage provides pure-Python raster segmentation,
superpixel generation, region merging, connected-component clumping,
and segment post-processing using ``numpy``, ``rasterio``, ``scipy``,
optional ``numba`` JIT acceleration, and :mod:`rsgispy.imageprocess`.

Submodules:

* :mod:`rsgispy.segmentation.clump` — Connected-component raster clumping
  (:func:`~rsgispy.segmentation.clump.clump`) and union of clumps
  (:func:`~rsgispy.segmentation.clump.union_of_clumps`).
* :mod:`rsgispy.segmentation.shepherd` — Shepherd et al. (2019) iterative
  K-means and small-segment elimination segmentation.
* :mod:`rsgispy.segmentation.region_merge` — Baatz & Schäpe multiresolution
  region-merging (FNEA) segmentation.
* :mod:`rsgispy.segmentation.slic` — Global grid-aligned SLIC superpixel
  segmentation.
* :mod:`rsgispy.segmentation.graph` — Felzenszwalb–Huttenlocher graph-based
  segmentation.
* :mod:`rsgispy.segmentation.watershed` — Seeded and auto-seeded compact
  gradient watershed segmentation.
* :mod:`rsgispy.segmentation.tiling` — Tiled halo planning, per-tile ID
  offsetting, seam reconciliation, and bounded-memory streaming pipelines.
* :mod:`rsgispy.segmentation.utils` — Optional ``numba`` acceleration checks,
  relabelling, small-segment elimination, polygonisation, and zonal statistics.
"""

from rsgispy.segmentation.graph import (
    graph_segmentation,
    graph_segmentation_array,
)
from rsgispy.segmentation.region_merge import (
    region_merge_segmentation,
    region_merge_segmentation_array,
)
from rsgispy.segmentation.shepherd import (
    shepherd_segmentation,
    shepherd_segmentation_array,
)
from rsgispy.segmentation.slic import (
    slic_segmentation,
    slic_segmentation_array,
)
from rsgispy.segmentation.tiling import (
    TileWindow,
    plan_tiles,
)
from rsgispy.segmentation.utils import (
    eliminate_small_segments,
    eliminate_small_segments_array,
    extract_segment_stats,
    is_numba_available,
    polygonise_segments,
    relabel_segments,
    relabel_segments_array,
)
from rsgispy.segmentation.watershed import (
    watershed_segmentation,
    watershed_segmentation_array,
)

__all__ = [
    "TileWindow",
    "plan_tiles",
    "is_numba_available",
    "shepherd_segmentation",
    "shepherd_segmentation_array",
    "region_merge_segmentation",
    "region_merge_segmentation_array",
    "slic_segmentation",
    "slic_segmentation_array",
    "graph_segmentation",
    "graph_segmentation_array",
    "watershed_segmentation",
    "watershed_segmentation_array",
    "eliminate_small_segments",
    "eliminate_small_segments_array",
    "relabel_segments",
    "relabel_segments_array",
    "polygonise_segments",
    "extract_segment_stats",
]
