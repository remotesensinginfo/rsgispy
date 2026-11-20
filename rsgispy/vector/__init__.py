"""
The vector subpackage consolidates vector I/O, geometry creation, geometric
operations, attribute table management, spatial statistics, and spatial/attribute
selection using ``geopandas``, ``pyogrio``, ``shapely``, ``rasterio``,
``pyproj``, ``rtree``, and ``h3``.

Every vector function accepts either a vector file path (``str`` or
:class:`os.PathLike`) or an in-memory :class:`geopandas.GeoDataFrame`.

Submodules
----------

* :mod:`rsgispy.vector.io` — metadata inspection, typed bounding boxes, format
  translation, copying, reprojection, UTM/WGS84 wrapping, merging, and splitting
* :mod:`rsgispy.vector.create` — polygonisation, pixel-to-point vectorisation,
  image footprints, bounding-box/WGS84/H3 grids, transects, and random points
* :mod:`rsgispy.vector.geoms` — line geometry operations, vector overlays,
  polygon transformations, vertex extraction, concave hulls, R-tree indexing,
  and spatial predicates
* :mod:`rsgispy.vector.attrs` — attribute table I/O, column manipulation,
  tabular export, and spatial attribute joins
* :mod:`rsgispy.vector.stats` — area, length, hole area, counts, radius
  density, nearest-neighbour metrics, and empirical semi-variograms
* :mod:`rsgispy.vector.selection` — spatial and attribute selection,
  deduplication, nearest-point matching, and train/test/validation splitting
"""

from rsgispy.vector import attrs, create, geoms, io, selection, stats

__all__ = [
    "io",
    "create",
    "geoms",
    "attrs",
    "stats",
    "selection",
]
