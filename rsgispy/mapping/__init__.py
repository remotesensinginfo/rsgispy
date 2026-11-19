"""
The mapping subpackage provides cartographic map composition and visualisation
utilities using ``matplotlib``, ``rasterio``, ``geopandas``, and ``pyproj``,
organised into submodules by functionality:

* :mod:`rsgispy.mapping.layout` — figure sizing, axis extent, tick spacing, and inset overview maps.
* :mod:`rsgispy.mapping.elements` — scale bars, north arrows, bounding boxes, and axis labels.
* :mod:`rsgispy.mapping.raster` — continuous, colour-mapped, and thematic raster maps.
* :mod:`rsgispy.mapping.vector` — vector layer, choropleth, and point-density maps.
* :mod:`rsgispy.mapping.basemaps` — contextily tile basemaps and WMTS layer mapping.
* :mod:`rsgispy.mapping.legend` — categorical and classification map legends.
"""

from typing import List

from rsgispy.mapping import basemaps, elements, layout, legend, raster, vector

__all__: List[str] = [
    "layout",
    "elements",
    "raster",
    "vector",
    "basemaps",
    "legend",
]
