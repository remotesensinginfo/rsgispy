"""
The elevation.hydrology module provides pure-Python hydrological conditioning,
flow routing, stream network extraction, watershed delineation, and terrain
index calculation functions using ``numpy``, ``rasterio``, ``geopandas``,
``shapely``, ``pyproj``, and optional ``scipy`` / ``numba`` acceleration.
"""

from rsgispy.elevation._hydro_common import (  # noqa: F401
    HAS_NUMBA,
    HAVE_SCIPY,
    distance_transform_edt,
    nd_label,
)
from rsgispy.elevation._hydro_conditioning import (
    breach_depressions,
    burn_streams,
    condition_dem,
    fill_depressions,
    identify_sinks,
    resolve_flats,
)
from rsgispy.elevation._hydro_flow import (
    flow_accumulation,
    flow_direction,
    flow_distance,
    flow_length,
    trace_flow_path,
)
from rsgispy.elevation._hydro_indices import (
    calc_hand,
    calc_ls_factor,
    calc_spi,
    calc_twi,
    flood_inundation_from_hand,
)
from rsgispy.elevation._hydro_streams import (
    extract_streams,
    stream_link,
    stream_order,
    vectorise_streams,
)
from rsgispy.elevation._hydro_watersheds import (
    delineate_basins,
    delineate_subcatchments,
    delineate_watersheds,
    polygonise_watersheds,
    snap_pour_points,
)


__all__ = [
    "HAS_NUMBA",
    "identify_sinks",
    "fill_depressions",
    "breach_depressions",
    "burn_streams",
    "resolve_flats",
    "condition_dem",
    "flow_direction",
    "flow_accumulation",
    "flow_length",
    "flow_distance",
    "trace_flow_path",
    "extract_streams",
    "stream_link",
    "stream_order",
    "vectorise_streams",
    "snap_pour_points",
    "delineate_watersheds",
    "delineate_basins",
    "delineate_subcatchments",
    "polygonise_watersheds",
    "calc_hand",
    "flood_inundation_from_hand",
    "calc_twi",
    "calc_spi",
    "calc_ls_factor",
]
