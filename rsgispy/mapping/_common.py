"""
Internal shared placement, validation and vector plotting helpers for
:mod:`rsgispy.mapping`.
"""

from typing import Any, Dict, Optional, Union

import geopandas as gpd
import matplotlib.axes
from matplotlib.offsetbox import AnchoredOffsetbox

import rsgispy

_POLY_GEOM_TYPES = ("Polygon", "MultiPolygon")
_LINE_GEOM_TYPES = ("LineString", "MultiLineString", "LinearRing")
_POINT_GEOM_TYPES = ("Point", "MultiPoint")

_LOC_ALIASES = {
    "top right": "upper right",
    "top left": "upper left",
    "bottom left": "lower left",
    "bottom right": "lower right",
    "top center": "upper center",
    "top centre": "upper center",
    "upper centre": "upper center",
    "bottom center": "lower center",
    "bottom centre": "lower center",
    "lower centre": "lower center",
    "center": "center",
    "centre": "center",
    "left": "center left",
    "centre left": "center left",
    "centre right": "center right",
}

_CODE_TO_LOC = {code: name for name, code in AnchoredOffsetbox.codes.items()}


def resolve_mpl_location(loc: Union[str, int]) -> str:
    """Validate and normalise a Matplotlib legend-style placement string or int code."""
    if isinstance(loc, int) and not isinstance(loc, bool):
        if loc in _CODE_TO_LOC:
            return _CODE_TO_LOC[loc]
        valid_codes = ", ".join(str(c) for c in sorted(_CODE_TO_LOC))
        raise rsgispy.RSGISPyException(
            f"Invalid placement location code '{loc}'. Valid codes are: {valid_codes}."
        )
    if not isinstance(loc, str):
        raise rsgispy.RSGISPyException(
            f"Location must be a string or integer code, got {type(loc).__name__}."
        )
    norm_loc = " ".join(loc.strip().lower().split())
    norm_loc = _LOC_ALIASES.get(norm_loc, norm_loc)
    if norm_loc not in AnchoredOffsetbox.codes:
        valid = ", ".join(sorted(AnchoredOffsetbox.codes))
        raise rsgispy.RSGISPyException(
            f"Invalid placement location '{loc}'. Valid options are: {valid}."
        )
    return norm_loc


def add_map_colour_bar(
    ax: matplotlib.axes.Axes,
    mappable: Any,
    cbar_label: Optional[str] = None,
) -> Any:
    """
    Add a colour bar for ``mappable`` beside the map axis ``ax`` (taking its
    space from ``ax``) and return it. The colour bar is also available as
    ``mappable.colorbar``.
    """
    fig = ax.get_figure()
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.85, pad=0.02)
    if cbar_label is not None:
        cbar.set_label(cbar_label)
    return cbar


def plot_vec_lyr_by_geom_type(
    ax: matplotlib.axes.Axes,
    gp_vec: Union[gpd.GeoDataFrame, gpd.GeoSeries],
    fill_clr: Any,
    line_clr: Any,
    line_width: float,
    *,
    alpha: float = 1.0,
    markersize: Optional[Union[str, float]] = None,
    zorder: Optional[float] = None,
) -> None:
    """
    Plot a GeoPandas layer on ``ax``, split by geometry type so each type uses
    the right colours: polygons and points are filled with ``fill_clr`` and
    outlined with ``line_clr``; lines are drawn in ``line_clr`` (GeoPandas
    draws lines in its ``color`` argument, which would otherwise be the fill
    colour). Geometry collections are exploded into their parts first.
    """
    # pylint: disable=too-many-arguments
    if gp_vec.empty:
        return
    geom_types = gp_vec.geom_type
    if (geom_types == "GeometryCollection").any():
        gp_vec = gp_vec.explode(index_parts=False)
        geom_types = gp_vec.geom_type

    common: Dict[str, Any] = {"ax": ax, "linewidth": line_width, "alpha": alpha}
    if zorder is not None:
        common["zorder"] = zorder

    polys = gp_vec[geom_types.isin(_POLY_GEOM_TYPES)]
    if not polys.empty:
        polys.plot(color=fill_clr, edgecolor=line_clr, **common)
    lines = gp_vec[geom_types.isin(_LINE_GEOM_TYPES)]
    if not lines.empty:
        lines.plot(color=line_clr, **common)
    points = gp_vec[geom_types.isin(_POINT_GEOM_TYPES)]
    if not points.empty:
        points.plot(
            color=fill_clr, edgecolor=line_clr, markersize=markersize, **common
        )
