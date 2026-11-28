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
   The pixel-in-polygon ``METHOD_*`` constants were removed: no zonal
   statistics function takes them.
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

__all__ = [
    "open_validated_zonal_inputs",
    "open_validated_zonal_multiband_inputs",
    "sample_point_pixel",
    "calc_label_zonal_stats",
    "join_label_stats_to_vec",
    "rasterise_label_values",
]
