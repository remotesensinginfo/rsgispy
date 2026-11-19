"""
The mapping.layout module provides figure sizing, axis extent, tick spacing,
and inset overview map utilities for cartographic plots.
"""

from typing import Optional, Tuple, Union

import geopandas as gpd
import matplotlib.axes
from matplotlib import ticker
from rasterio.warp import transform_bounds

import rsgispy
from rsgispy.mapping._common import resolve_mpl_location

__all__ = [
    "calc_y_fig_size",
    "define_axis_extent",
    "define_map_tick_spacing",
    "get_overview_info",
    "add_overview_maps",
]


def calc_y_fig_size(
    bbox: Tuple[float, float, float, float],
    fig_x_size: Union[int, float],
) -> float:
    """
    Calculate the figure y-axis size given a bounding box and figure x-axis size
    so the mapped area fills the figure cleanly.

    :param bbox: Bounding box ``(MinX, MaxX, MinY, MaxY)`` for the region of interest.
    :param fig_x_size: Size for the figure in the x-axis.
    :return: Figure size in the y-axis.

    .. code:: python

        import rsgispy.mapping.layout

        fig_y_size = rsgispy.mapping.layout.calc_y_fig_size(
            bbox=(250000.0, 270000.0, 280000.0, 290000.0), fig_x_size=10.0
        )
    """
    x_bbox_size = bbox[1] - bbox[0]
    if x_bbox_size == 0:
        raise rsgispy.RSGISPyException(
            "Bounding box width (MaxX - MinX) must be non-zero."
        )
    y_bbox_size = bbox[3] - bbox[2]
    bbox_xy_ratio = y_bbox_size / x_bbox_size
    return float(fig_x_size * bbox_xy_ratio)


def define_axis_extent(
    ax: matplotlib.axes.Axes,
    bbox: Tuple[float, float, float, float],
) -> None:
    """
    Set equal aspect ratio, x/y limits, and plain tick labels on ``ax`` using
    ``bbox``.

    :param ax: Matplotlib axis for which limits are set.
    :param bbox: Bounding box ``(MinX, MaxX, MinY, MaxY)``.

    .. code:: python

        import matplotlib.pyplot as plt
        import rsgispy.mapping.layout

        fig, ax = plt.subplots()
        rsgispy.mapping.layout.define_axis_extent(
            ax, bbox=(250000.0, 270000.0, 280000.0, 290000.0)
        )
    """
    ax.set_aspect("equal", "box")
    ax.set_xlim([bbox[0], bbox[1]])
    ax.set_ylim([bbox[2], bbox[3]])
    ax.ticklabel_format(useOffset=False, style="plain")


def define_map_tick_spacing(
    ax: matplotlib.axes.Axes,
    tick_spacing: float,
) -> None:
    """
    Define the major tick spacing on both axes of the map.

    :param ax: Matplotlib axis for which tick spacing is defined.
    :param tick_spacing: Spacing between major ticks in map units.

    .. code:: python

        import matplotlib.pyplot as plt
        import rsgispy.mapping.layout

        fig, ax = plt.subplots()
        rsgispy.mapping.layout.define_map_tick_spacing(ax, tick_spacing=5000.0)
    """
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.ticklabel_format(useOffset=False, style="plain")


def get_overview_info(
    roi_bbox: Tuple[float, float, float, float],
    roi_epsg: int = 4326,
    overview_buf: float = 30.0,
    out_epsg: int = 4326,
) -> Tuple[Tuple[float, float, float, float], Tuple[float, float]]:
    """
    Compute an overview map bounding box and centre point from a region of
    interest bounding box, optionally reprojecting between EPSG codes.

    :param roi_bbox: Region of interest bbox ``(MinX, MaxX, MinY, MaxY)``.
    :param roi_epsg: EPSG code of ``roi_bbox`` (Default = 4326).
    :param overview_buf: Buffer added around the ROI centre point in ``out_epsg``
                         units (Default = 30.0).
    :param out_epsg: Output EPSG code for the overview bbox and point (Default = 4326).
    :return: Tuple ``(overview_bbox, centre_point)`` where ``overview_bbox`` is
             ``(MinX, MaxX, MinY, MaxY)`` and ``centre_point`` is ``(X, Y)``.

    .. code:: python

        import rsgispy.mapping.layout

        over_bbox, centre_pt = rsgispy.mapping.layout.get_overview_info(
            roi_bbox=(-4.2, -4.0, 52.3, 52.5),
            roi_epsg=4326,
            overview_buf=5.0,
        )
    """
    if roi_epsg != out_epsg:
        min_x, min_y, max_x, max_y = transform_bounds(
            f"EPSG:{roi_epsg}",
            f"EPSG:{out_epsg}",
            roi_bbox[0],
            roi_bbox[2],
            roi_bbox[1],
            roi_bbox[3],
        )
        roi_proj_bbox = (min_x, max_x, min_y, max_y)
    else:
        roi_proj_bbox = (
            float(roi_bbox[0]),
            float(roi_bbox[1]),
            float(roi_bbox[2]),
            float(roi_bbox[3]),
        )

    c_pt_x = roi_proj_bbox[0] + ((roi_proj_bbox[1] - roi_proj_bbox[0]) / 2.0)
    c_pt_y = roi_proj_bbox[2] + ((roi_proj_bbox[3] - roi_proj_bbox[2]) / 2.0)
    over_bbox: Tuple[float, float, float, float] = (
        c_pt_x - overview_buf,
        c_pt_x + overview_buf,
        c_pt_y - overview_buf,
        c_pt_y + overview_buf,
    )
    return over_bbox, (c_pt_x, c_pt_y)


_LOC_TO_ANCHOR = {
    "upper left": "NW",
    "upper center": "N",
    "upper right": "NE",
    "center left": "W",
    "center": "C",
    "center right": "E",
    "right": "E",
    "lower left": "SW",
    "lower center": "S",
    "lower right": "SE",
}


def _calc_inset_bounds(
    loc: Union[str, int],
    over_size_x: float,
    over_size_y: float,
    over_x_off: float,
    over_y_off: float,
    bbox_to_anchor: Optional[Tuple[float, ...]],
) -> Tuple[str, Tuple[float, float, float, float]]:
    """Compute normalised location and inset bounds ``(x0, y0, width, height)``."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    norm_loc = resolve_mpl_location(loc)

    if bbox_to_anchor is None:
        bx, by, bw, bh = 0.0, 0.0, 1.0, 1.0
    elif len(bbox_to_anchor) == 2:
        bx, by = float(bbox_to_anchor[0]), float(bbox_to_anchor[1])
        bw, bh = 0.0, 0.0
    elif len(bbox_to_anchor) == 4:
        bx, by, bw, bh = (
            float(bbox_to_anchor[0]),
            float(bbox_to_anchor[1]),
            float(bbox_to_anchor[2]),
            float(bbox_to_anchor[3]),
        )
    else:
        raise rsgispy.RSGISPyException(
            "bbox_to_anchor must be a 2-tuple (x, y) or 4-tuple (x, y, width, height)."
        )

    if norm_loc in ("upper left", "center left", "lower left"):
        x0 = bx
    elif norm_loc in ("upper right", "center right", "right", "lower right"):
        x0 = bx + bw - over_size_x
    else:
        x0 = bx + (bw - over_size_x) / 2.0

    if norm_loc in ("lower left", "lower center", "lower right"):
        y0 = by
    elif norm_loc in ("upper left", "upper center", "upper right"):
        y0 = by + bh - over_size_y
    else:
        y0 = by + (bh - over_size_y) / 2.0

    bounds = (
        x0 + float(over_x_off),
        y0 + float(over_y_off),
        float(over_size_x),
        float(over_size_y),
    )
    return norm_loc, bounds


def add_overview_maps(
    ax: matplotlib.axes.Axes,
    overview_lyr: gpd.GeoDataFrame,
    overview_bbox: Tuple[float, float, float, float],
    overview_pt: Optional[Tuple[float, float]] = None,
    over_size_x: float = 0.1,
    over_size_y: float = 0.1,
    over_x_off: float = 0.0,
    over_y_off: float = 0.0,
    pt_clr: str = "red",
    pt_size: float = 10.0,
    fill_clr: str = "white",
    line_clr: str = "black",
    line_width: float = 0.25,
    loc: Union[str, int] = "upper left",
    bbox_to_anchor: Optional[Tuple[float, ...]] = None,
) -> matplotlib.axes.Axes:
    """
    Add an inset overview map to ``ax`` using Matplotlib legend-style placement.

    :param ax: Parent matplotlib axis.
    :param overview_lyr: GeoDataFrame used as the overview map backdrop.
    :param overview_bbox: Bounding box ``(MinX, MaxX, MinY, MaxY)`` shown in
                          the inset axis.
    :param overview_pt: Optional ``(X, Y)`` marker coordinate on the overview map.
    :param over_size_x: Width of the inset axis in axes fraction coordinates.
    :param over_size_y: Height of the inset axis in axes fraction coordinates.
    :param over_x_off: X-axis offset added to the resolved position in axes
                       fraction coordinates (Default = 0.0).
    :param over_y_off: Y-axis offset added to the resolved position in axes
                       fraction coordinates (Default = 0.0).
    :param pt_clr: Colour of ``overview_pt`` marker (Default = ``'red'``).
    :param pt_size: Marker size for ``overview_pt`` (Default = 10.0).
    :param fill_clr: Polygon fill colour for ``overview_lyr`` (Default = ``'white'``).
    :param line_clr: Edge line colour for ``overview_lyr`` (Default = ``'black'``).
    :param line_width: Line width for ``overview_lyr`` (Default = 0.25).
    :param loc: Matplotlib legend placement string or integer code (Default =
                ``'upper left'``). Supports ``'upper left'``, ``'upper right'``,
                ``'lower left'``, ``'lower right'``, ``'upper center'``,
                ``'lower center'``, ``'center left'``, ``'center right'``,
                ``'right'``, ``'center'``, common aliases (e.g., ``'top right'``,
                ``'bottom left'``, ``'centre'``, ``'left'``), and codes ``1``–``10``.
    :param bbox_to_anchor: Optional Matplotlib ``bbox_to_anchor`` 2-tuple
                          ``(x, y)`` or 4-tuple ``(x, y, width, height)`` in
                          axes fraction coordinates (Default = ``None``).
    :return: Created inset matplotlib axis.

    .. code:: python

        import geopandas as gpd
        import matplotlib.pyplot as plt
        import rsgispy.mapping.layout

        fig, ax = plt.subplots()
        world_gdf = gpd.read_file("world.gpkg")
        ax_inset = rsgispy.mapping.layout.add_overview_maps(
            ax=ax,
            overview_lyr=world_gdf,
            overview_bbox=(-10.0, 5.0, 48.0, 60.0),
            overview_pt=(-4.08, 52.41),
            loc="lower right",
            over_x_off=-0.02,
            over_y_off=0.02,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    norm_loc, inset_bounds = _calc_inset_bounds(
        loc=loc,
        over_size_x=over_size_x,
        over_size_y=over_size_y,
        over_x_off=over_x_off,
        over_y_off=over_y_off,
        bbox_to_anchor=bbox_to_anchor,
    )
    ax_over = ax.inset_axes(list(inset_bounds))
    ax_over.set_anchor(_LOC_TO_ANCHOR[norm_loc])
    ax_over.set_xticks([])
    ax_over.set_yticks([])
    overview_lyr.plot(
        ax=ax_over,
        color=fill_clr,
        edgecolor=line_clr,
        linewidth=line_width,
        alpha=1.0,
    )
    if overview_pt is not None:
        ax_over.scatter(overview_pt[0], overview_pt[1], color=pt_clr, s=pt_size)
    ax_over.set_xlim([overview_bbox[0], overview_bbox[1]])
    ax_over.set_ylim([overview_bbox[2], overview_bbox[3]])
    return ax_over
