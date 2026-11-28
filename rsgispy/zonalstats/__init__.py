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

.. versionchanged:: 0.5.0
   The pixel-in-polygon ``METHOD_*`` constants are no longer exported: no
   zonal statistics function takes them. The names still resolve, with a
   :class:`DeprecationWarning`, and will be removed in a later release.
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

import warnings as _warnings

#: Deprecated pixel-in-polygon method codes (RSGISLib numbering), kept only
#: for :func:`__getattr__`. :func:`rsgispy.imagespectral.endmembers.
#: extract_avg_endmembers` takes the integer codes directly.
_DEPRECATED_METHODS = {
    "METHOD_POLYCONTAINSPIXEL": 0,
    "METHOD_POLYCONTAINSPIXELCENTRE": 1,
    "METHOD_POLYCONTAINSPIXELCENTER": 1,
    "METHOD_POLYOVERLAPSPIXEL": 2,
    "METHOD_POLYOVERLAPSORCONTAINSPIXEL": 3,
    "METHOD_PIXELCONTAINSPOLY": 4,
    "METHOD_PIXELCONTAINSPOLYCENTRE": 5,
    "METHOD_PIXELCONTAINSPOLYCENTER": 5,
    "METHOD_ADAPTIVE": 6,
    "METHOD_ENVELOPE": 7,
    "METHOD_PIXELAREAINPOLY": 8,
    "METHOD_POLYAREAINPIXEL": 9,
}


def __getattr__(name: str) -> int:
    """Resolve the deprecated ``METHOD_*`` names with a warning."""
    if name in _DEPRECATED_METHODS:
        _warnings.warn(
            f"rsgispy.zonalstats.{name} is deprecated and will be removed: no "
            "zonal statistics function takes it. Pass the integer code "
            f"({_DEPRECATED_METHODS[name]}) where one is needed.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _DEPRECATED_METHODS[name]
    raise AttributeError(f"module 'rsgispy.zonalstats' has no attribute '{name}'")


__all__ = [
    "open_validated_zonal_inputs",
    "open_validated_zonal_multiband_inputs",
    "sample_point_pixel",
    "calc_label_zonal_stats",
    "join_label_stats_to_vec",
    "rasterise_label_values",
]
