"""
The vectorgeoms subpackage performs geometry creation, vectorisation, grid
generation, geometric transformations, overlays, and spatial predicates using
``geopandas``, ``shapely``, ``pyogrio``, ``rasterio``, ``rtree``, and ``h3``:

* :mod:`rsgispy.vectorgeoms.create` — polygonisation, pixel-to-point
  vectorisation, image footprints, bounding-box/WGS84/H3 grids, and random
  point layers
* :mod:`rsgispy.vectorgeoms.lines` — line creation, image transects, orthogonal
  transects, regular point splitting, boundary intersections, and coastline
  prediction
* :mod:`rsgispy.vectorgeoms.overlay` — vector intersection, difference,
  symmetric difference, identity, union, and ROI clipping
* :mod:`rsgispy.vectorgeoms.polygons` — polygon boundaries, centroids,
  exploding, simplification, hole removal, buffering, dissolving, and thinning
* :mod:`rsgispy.vectorgeoms.spatial` — vertex extraction, concave hulls,
  bounding-box queries, R-tree indexing, and spatial predicates
"""
