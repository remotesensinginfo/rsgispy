"""
The vectorutils subpackage provides utilities for inspecting, transforming,
reprojecting, merging, splitting, and selecting vector datasets using
``geopandas``, ``pyogrio``, ``shapely``, ``rasterio``, and ``pyproj`` without
``osgeo`` dependencies:

* :mod:`rsgispy.vectorutils.metadata` — projection inspection, standardised
  bounding boxes, feature counts, columns, layer lists, and file deletion
* :mod:`rsgispy.vectorutils.io` — format translation, copying, reprojection,
  UTM/WGS84 wrapping, layer merging, and attribute/chunk splitting
* :mod:`rsgispy.vectorutils.selection` — spatial and attribute selection,
  deduplication, nearest-point matching, and train/test/validation splitting
"""
