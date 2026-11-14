"""
The zonalstats subpackage provides modules for pixel-in-polygon
(:mod:`rsgispy.zonalstats.polygon`) and point-in-pixel
(:mod:`rsgispy.zonalstats.point`) zonal statistics using ``rasterio`` and
``geopandas``.

Pixel-in-polygon selection method constants:

    * ``METHOD_POLYCONTAINSPIXEL = 0``
    * ``METHOD_POLYCONTAINSPIXELCENTER = 1``
    * ``METHOD_POLYOVERLAPSPIXEL = 2``
    * ``METHOD_POLYOVERLAPSORCONTAINSPIXEL = 3``
    * ``METHOD_PIXELCONTAINSPOLY = 4``
    * ``METHOD_PIXELCONTAINSPOLYCENTER = 5``
    * ``METHOD_ADAPTIVE = 6``
    * ``METHOD_ENVELOPE = 7``
    * ``METHOD_PIXELAREAINPOLY = 8``
    * ``METHOD_POLYAREAINPIXEL = 9``
"""

#: Pixel is completely contained within polygon (``0``).
METHOD_POLYCONTAINSPIXEL = 0
#: Pixel centre falls inside polygon (``1``).
METHOD_POLYCONTAINSPIXELCENTER = 1
#: Polygon overlaps pixel (``2``).
METHOD_POLYOVERLAPSPIXEL = 2
#: Polygon overlaps or contains pixel (``3``).
METHOD_POLYOVERLAPSORCONTAINSPIXEL = 3
#: Pixel contains polygon (``4``).
METHOD_PIXELCONTAINSPOLY = 4
#: Pixel contains polygon centroid (``5``).
METHOD_PIXELCONTAINSPOLYCENTER = 5
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
    "METHOD_POLYCONTAINSPIXELCENTER",
    "METHOD_POLYOVERLAPSPIXEL",
    "METHOD_POLYOVERLAPSORCONTAINSPIXEL",
    "METHOD_PIXELCONTAINSPOLY",
    "METHOD_PIXELCONTAINSPOLYCENTER",
    "METHOD_ADAPTIVE",
    "METHOD_ENVELOPE",
    "METHOD_PIXELAREAINPOLY",
    "METHOD_POLYAREAINPIXEL",
]
