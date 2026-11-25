"""
The zonalstats subpackage provides modules for pixel-in-polygon
(:mod:`rsgispy.zonalstats.polygon`), point-in-pixel
(:mod:`rsgispy.zonalstats.point`) and label raster
(:mod:`rsgispy.zonalstats.labels`) zonal statistics using ``rasterio`` and
``geopandas``.

For zones defined by a label raster (e.g., segments from
:mod:`rsgispy.segmentation`), use
:func:`~rsgispy.zonalstats.labels.calc_label_zonal_stats`, which computes the
statistics for all labels in one streamed pass and is far faster than
polygon zonal statistics for large numbers of zones.

Pixel-in-polygon selection method constants:

    * ``METHOD_POLYCONTAINSPIXEL = 0``
    * ``METHOD_POLYCONTAINSPIXELCENTRE = 1``
    * ``METHOD_POLYOVERLAPSPIXEL = 2``
    * ``METHOD_POLYOVERLAPSORCONTAINSPIXEL = 3``
    * ``METHOD_PIXELCONTAINSPOLY = 4``
    * ``METHOD_PIXELCONTAINSPOLYCENTRE = 5``
    * ``METHOD_ADAPTIVE = 6``
    * ``METHOD_ENVELOPE = 7``
    * ``METHOD_PIXELAREAINPOLY = 8``
    * ``METHOD_POLYAREAINPIXEL = 9``
"""

from rsgispy.zonalstats._helpers import (
    open_validated_zonal_inputs,
    open_validated_zonal_multiband_inputs,
    sample_point_pixel,
)
from rsgispy.zonalstats.labels import (
    calc_label_zonal_stats,
    join_label_stats_to_vec,
    rasterise_label_values,
)

#: Pixel is completely contained within polygon (``0``).
METHOD_POLYCONTAINSPIXEL = 0
#: Pixel centre falls inside polygon (``1``).
METHOD_POLYCONTAINSPIXELCENTRE = 1
#: Compatibility alias for :data:`METHOD_POLYCONTAINSPIXELCENTRE`.
METHOD_POLYCONTAINSPIXELCENTER = METHOD_POLYCONTAINSPIXELCENTRE
#: Polygon overlaps pixel (``2``).
METHOD_POLYOVERLAPSPIXEL = 2
#: Polygon overlaps or contains pixel (``3``).
METHOD_POLYOVERLAPSORCONTAINSPIXEL = 3
#: Pixel contains polygon (``4``).
METHOD_PIXELCONTAINSPOLY = 4
#: Pixel contains polygon centroid (``5``).
METHOD_PIXELCONTAINSPOLYCENTRE = 5
#: Compatibility alias for :data:`METHOD_PIXELCONTAINSPOLYCENTRE`.
METHOD_PIXELCONTAINSPOLYCENTER = METHOD_PIXELCONTAINSPOLYCENTRE
#: Adaptive pixel-in-polygon method (``6``).
METHOD_ADAPTIVE = 6
#: Bounding envelope overlap method (``7``).
METHOD_ENVELOPE = 7
#: Pixel area within polygon method (``8``).
METHOD_PIXELAREAINPOLY = 8
#: Polygon area within pixel method (``9``).
METHOD_POLYAREAINPIXEL = 9

__all__ = [
    "METHOD_POLYCONTAINSPIXEL",
    "METHOD_POLYCONTAINSPIXELCENTRE",
    "METHOD_POLYOVERLAPSPIXEL",
    "METHOD_POLYOVERLAPSORCONTAINSPIXEL",
    "METHOD_PIXELCONTAINSPOLY",
    "METHOD_PIXELCONTAINSPOLYCENTRE",
    "METHOD_ADAPTIVE",
    "METHOD_ENVELOPE",
    "METHOD_PIXELAREAINPOLY",
    "METHOD_POLYAREAINPIXEL",
    "open_validated_zonal_inputs",
    "open_validated_zonal_multiband_inputs",
    "sample_point_pixel",
    "calc_label_zonal_stats",
    "join_label_stats_to_vec",
    "rasterise_label_values",
]
