"""
The mapping.layout module provides figure sizing, axis extent, tick spacing,
inset overview map utilities and multi-panel map figures for cartographic
plots.
"""

import dataclasses
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import geopandas as gpd
import matplotlib.axes
import matplotlib.figure
import numpy as np
import rasterio
from matplotlib import ticker
from rasterio.warp import transform_bounds

import rsgispy
from rsgispy.tools.plotting import (
    PLOT_DPI,
    get_fig_axs,
    save_plot,
    use_plot_style,
)
from rsgispy.mapping._common import (
    MAP_BOX_ALPHA,
    MAP_BOX_EDGE_COLOUR,
    add_map_colour_bar,
    plot_vec_lyr_by_geom_type,
    resolve_mpl_location,
)

__all__ = [
    "calc_y_fig_size",
    "define_axis_extent",
    "define_map_tick_spacing",
    "get_overview_info",
    "add_overview_maps",
    "MapPanel",
    "create_map_panels",
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
    over_bbox = rsgispy.BBox(
        c_pt_x - overview_buf,
        c_pt_x + overview_buf,
        c_pt_y - overview_buf,
        c_pt_y + overview_buf,
        epsg=int(out_epsg),
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


@use_plot_style(plot_type="map")
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
    The inset has no ticks or axis labels.

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
    :param line_clr: Colour of polygon outlines and line features in
                     ``overview_lyr`` (Default = ``'black'``).
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
    plot_vec_lyr_by_geom_type(
        ax_over,
        overview_lyr,
        fill_clr=fill_clr,
        line_clr=line_clr,
        line_width=line_width,
    )
    # GeoPandas labels the axes from the layer's CRS; not wanted on an inset.
    ax_over.set_xlabel("")
    ax_over.set_ylabel("")
    if overview_pt is not None:
        ax_over.scatter(overview_pt[0], overview_pt[1], color=pt_clr, s=pt_size)
    ax_over.set_xlim([overview_bbox[0], overview_bbox[1]])
    ax_over.set_ylim([overview_bbox[2], overview_bbox[3]])
    return ax_over


#: Panel types supported by :class:`MapPanel`.
MAP_PANEL_TYPES = ("rgb", "cmap", "thematic", "vector")

_DEFAULT_PANEL_WIDTH = 5.0


@dataclasses.dataclass
class MapPanel:
    """
    One panel of a :func:`create_map_panels` figure. A plain ``dict`` with
    the same keys can be used instead.

    :param panel_type: ``'rgb'`` (a 1- or 3-band stretched raster, drawn with
                       :func:`~rsgispy.mapping.raster.create_raster_img_map`),
                       ``'cmap'`` (one band drawn with a colour map,
                       :func:`~rsgispy.mapping.raster.create_raster_cmap_img_map`),
                       ``'thematic'`` (a raster with a colour table,
                       :func:`~rsgispy.mapping.raster.create_thematic_raster_map`)
                       or ``'vector'`` (GeoDataFrame layers,
                       :func:`~rsgispy.mapping.vector.create_vec_lyr_map`).
    :param input_img: The raster image (for ``'rgb'``, ``'cmap'`` and
                      ``'thematic'``).
    :param bands: The image band(s): a list of 1 or 3 bands for ``'rgb'``
                  (Default = ``[1, 2, 3]``) or one band for ``'cmap'``
                  (Default = 1).
    :param img_stch: Stretch for ``'rgb'`` (``rsgispy.IMG_STRETCH_*``;
                     Default = ``rsgispy.IMG_STRETCH_CUMULATIVE``).
    :param vec: GeoDataFrame, or list of GeoDataFrames, for ``'vector'``.
    :param title: Optional panel title.
    :param cmap_name: Colour map for ``'cmap'`` (Default = ``'viridis'``).
    :param vmin: Optional minimum of the ``'cmap'`` colour scale
                 (Default = the minimum within the extent).
    :param vmax: Optional maximum of the ``'cmap'`` colour scale
                 (Default = the maximum within the extent).
    :param show_colour_bar: Draw a colour bar beside a ``'cmap'`` panel
                            (Default = False).
    :param cbar_label: Optional colour bar label.
    :param classes_info: Optional class names and colours of a
                         ``'thematic'`` panel for the shared legend (see
                         :func:`~rsgispy.mapping.legend.add_categorical_legend`).
                         Default: the names stored by
                         :func:`~rsgispy.imageutils.colourtable.set_thematic_classes`,
                         or the colour table entries.
    :param in_legend: Add the classes of a ``'thematic'`` panel to the shared
                      legend (Default = True).
    :param base_img: Optional image drawn below the panel layer (e.g. the
                     image a change map is drawn over). Pixels of value 0 of
                     a thematic layer are transparent.
    :param base_bands: Bands of ``base_img`` (Default = ``(1, 2, 3)``).
    :param base_stch: Stretch of ``base_img`` (Default =
                      ``rsgispy.IMG_STRETCH_CUMULATIVE``).
    :param map_kw: Optional extra keyword arguments for the function that
                   draws the panel layer (e.g. ``{"no_data_val": -9999}`` for
                   ``'rgb'`` or ``{"vec_fill_clrs": "none"}`` for
                   ``'vector'``).
    """

    # pylint: disable=too-many-instance-attributes
    panel_type: str
    input_img: Optional[str] = None
    bands: Optional[Union[int, Sequence[int]]] = None
    img_stch: int = rsgispy.IMG_STRETCH_CUMULATIVE
    vec: Optional[Any] = None
    title: Optional[str] = None
    cmap_name: str = "viridis"
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    show_colour_bar: bool = False
    cbar_label: Optional[str] = None
    classes_info: Optional[Any] = None
    in_legend: bool = True
    base_img: Optional[str] = None
    base_bands: Sequence[int] = (1, 2, 3)
    base_stch: int = rsgispy.IMG_STRETCH_CUMULATIVE
    map_kw: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        self.panel_type = str(self.panel_type).strip().lower()
        if self.panel_type not in MAP_PANEL_TYPES:
            raise rsgispy.RSGISPyException(
                f"panel_type must be one of {MAP_PANEL_TYPES}, not "
                f"'{self.panel_type}'."
            )
        if self.panel_type == "vector":
            if self.vec is None:
                raise rsgispy.RSGISPyException("A 'vector' panel needs vec.")
        elif self.input_img is None:
            raise rsgispy.RSGISPyException(
                f"A '{self.panel_type}' panel needs input_img."
            )


def _to_map_panel(panel: Union[MapPanel, Mapping[str, Any]]) -> MapPanel:
    """Convert a panel specification (dict or :class:`MapPanel`) to a MapPanel."""
    if isinstance(panel, MapPanel):
        return panel
    if isinstance(panel, Mapping):
        field_names = {fld.name for fld in dataclasses.fields(MapPanel)}
        unknown = set(panel) - field_names
        if unknown:
            raise rsgispy.RSGISPyException(
                f"Unknown map panel keys: {sorted(unknown)}. "
                f"Valid keys: {sorted(field_names)}."
            )
        return MapPanel(**dict(panel))
    raise rsgispy.RSGISPyException(
        f"A map panel must be a MapPanel or a dict, not {type(panel).__name__}."
    )


def _panel_bbox(panel: MapPanel) -> Tuple[float, float, float, float]:
    """The extent ``(MinX, MaxX, MinY, MaxY)`` of a panel's data."""
    if panel.input_img is not None:
        with rasterio.open(panel.input_img) as src:
            bnds = src.bounds
        return (bnds.left, bnds.right, bnds.bottom, bnds.top)
    vecs = panel.vec if isinstance(panel.vec, list) else [panel.vec]
    tot_bnds = np.array([v.total_bounds for v in vecs], dtype=float)
    return (
        float(tot_bnds[:, 0].min()),
        float(tot_bnds[:, 2].max()),
        float(tot_bnds[:, 1].min()),
        float(tot_bnds[:, 3].max()),
    )


def _panel_epsg(panel: MapPanel) -> Optional[int]:
    """The EPSG code of a panel's data, if known."""
    img = panel.input_img if panel.input_img is not None else panel.base_img
    if img is not None:
        with rasterio.open(img) as src:
            return src.crs.to_epsg() if src.crs is not None else None
    vec = panel.vec[0] if isinstance(panel.vec, list) else panel.vec
    return vec.crs.to_epsg() if vec.crs is not None else None


def _draw_map_panel(
    ax: matplotlib.axes.Axes,
    panel: MapPanel,
    bbox: Tuple[float, float, float, float],
    show_map_axis: bool,
) -> Any:
    """Draw one panel (optional base image, then the layer) and return its artist."""
    # pylint: disable=import-outside-toplevel
    import rsgispy.mapping.raster as map_raster
    import rsgispy.mapping.vector as map_vector

    common: Dict[str, Any] = {
        "bbox": bbox,
        "title_str": panel.title,
        "show_scale_bar": False,
        "show_map_axis": show_map_axis,
    }
    layer_z = 1.0
    if panel.base_img is not None:
        map_raster.create_raster_img_map(
            ax,
            panel.base_img,
            img_bands=list(panel.base_bands),
            img_stch=panel.base_stch,
            plot_zorder=1.0,
            **common,
        )
        layer_z = 2.0
    extra = dict(panel.map_kw or {})

    if panel.panel_type == "rgb":
        bands = panel.bands if panel.bands is not None else [1, 2, 3]
        bands = [bands] if isinstance(bands, int) else list(bands)
        return map_raster.create_raster_img_map(
            ax,
            panel.input_img,
            img_bands=bands,
            img_stch=panel.img_stch,
            plot_zorder=layer_z,
            **common,
            **extra,
        )
    if panel.panel_type == "cmap":
        band = panel.bands if panel.bands is not None else 1
        if not isinstance(band, int):
            band = int(list(band)[0])
        img_artist = map_raster.create_raster_cmap_img_map(
            ax,
            panel.input_img,
            img_band=band,
            cmap_name=panel.cmap_name,
            norm_img_vals=True,
            norm_vmin=panel.vmin,
            norm_vmax=panel.vmax,
            plot_zorder=layer_z,
            **common,
            **extra,
        )
        if panel.show_colour_bar:
            add_map_colour_bar(ax, img_artist, cbar_label=panel.cbar_label)
        return img_artist
    if panel.panel_type == "thematic":
        return map_raster.create_thematic_raster_map(
            ax,
            panel.input_img,
            plot_zorders=layer_z,
            **common,
            **extra,
        )
    return map_vector.create_vec_lyr_map(
        ax,
        panel.vec,
        plot_zorders=layer_z,
        **common,
        **extra,
    )


def _shared_legend_classes(
    panels: Sequence[MapPanel],
    bbox: Tuple[float, float, float, float],
    classes_info: Optional[Any],
) -> Dict[int, Tuple[str, Any]]:
    """
    The classes of the thematic panels within ``bbox`` as
    ``{pixel value: (name, colour)}``; a value in more than one panel takes
    the name and colour of the first.
    """
    # pylint: disable=import-outside-toplevel
    from rsgispy.imageutils.colourtable import get_thematic_classes
    from rsgispy.mapping._legend import resolve_legend_entries

    merged: Dict[int, Tuple[str, Any]] = {}
    for panel in panels:
        if panel.panel_type != "thematic" or not panel.in_legend:
            continue
        cls_info = classes_info if classes_info is not None else panel.classes_info
        if cls_info is None:
            cls_info = get_thematic_classes(panel.input_img) or None
        try:
            entries = resolve_legend_entries(
                classes_info=cls_info,
                input_img=panel.input_img,
                bbox=bbox,
                only_present_in_img=True,
            )
        except rsgispy.RSGISPyException:
            # None of the panel's classes is within bbox.
            continue
        for entry in entries:
            if entry.pxl_val not in merged:
                merged[entry.pxl_val] = (entry.label, entry.colour)
    return dict(sorted(merged.items()))


def _add_shared_legend(
    fig: matplotlib.figure.Figure,
    legend_classes: Mapping[int, Tuple[str, Any]],
    legend_title: Optional[str],
    legend_position: str,
) -> Any:
    """Add one figure legend for ``legend_classes`` beside or below the panels."""
    # pylint: disable=import-outside-toplevel
    from rsgispy.mapping._legend import LegendEntry, build_legend_handles

    handles = build_legend_handles(
        [
            LegendEntry(pxl_val=val, label=name, colour=clr)
            for val, (name, clr) in legend_classes.items()
        ]
    )
    if legend_position == "right":
        loc, anchor, n_cols = "center left", (1.0, 0.5), 1
    else:
        loc, anchor, n_cols = "upper center", (0.5, 0.0), min(len(handles), 6)
    lgd = fig.legend(
        handles=handles,
        loc=loc,
        bbox_to_anchor=anchor,
        ncol=n_cols,
        title=legend_title,
        frameon=True,
        facecolor="white",
        edgecolor=MAP_BOX_EDGE_COLOUR,
        framealpha=MAP_BOX_ALPHA,
        fancybox=False,
    )
    if legend_title is not None:
        lgd.get_title().set_fontweight("bold")
    return lgd


@use_plot_style(plot_type="map")
def create_map_panels(
    panels: Sequence[Union[MapPanel, Mapping[str, Any]]],
    ncols: Optional[int] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    *,
    suptitle: Optional[str] = None,
    show_map_axis: bool = False,
    show_scale_bar: bool = True,
    scale_bar_loc: str = "upper right",
    show_north_arrow: bool = True,
    north_arrow_loc: str = "upper left",
    show_legend: bool = True,
    classes_info: Optional[Any] = None,
    legend_title: Optional[str] = None,
    legend_position: str = "right",
    axs: Optional[Any] = None,
    out_plot_file: Optional[str] = None,
    out_dpi: int = PLOT_DPI,
) -> Tuple[matplotlib.figure.Figure, np.ndarray]:
    """
    Draw maps side by side that share one extent: e.g. an image and one or
    more classifications, or a change distance and the change drawn over the
    image. Each panel is an RGB (or 1-band stretched) raster, a colour-mapped
    raster, a thematic raster or vector layers, optionally over a base image,
    with a title (see :class:`MapPanel`). The thematic panels share one
    legend, colour-mapped panels can have colour bars, and one scale bar and
    north arrow are drawn on the first panel.

    :param panels: List of panel specifications, each a :class:`MapPanel` or
                   a ``dict`` with the same keys.
    :param ncols: Number of panels per row (Default = all in one row, up to
                  4 per row).
    :param bbox: Extent ``(MinX, MaxX, MinY, MaxY)`` of every panel (Default =
                 the extent of the first panel's data).
    :param figsize: Figure size in inches (Default = 5 inches wide per panel,
                    with the height from the extent).
    :param suptitle: Optional title of the figure.
    :param show_map_axis: Show the coordinate axes of each panel (Default =
                          False; the scale bar gives the scale).
    :param show_scale_bar: Draw a scale bar on the first panel (Default =
                           True).
    :param scale_bar_loc: Scale bar placement (Default = ``'upper right'``).
    :param show_north_arrow: Draw a north arrow on the first panel (Default =
                             True).
    :param north_arrow_loc: North arrow placement (Default = ``'upper left'``).
    :param show_legend: Draw one legend of the classes of the thematic
                        panels found within ``bbox`` (Default = True).
    :param classes_info: Optional class names and colours for the legend,
                         used for every thematic panel (Default = each
                         panel's ``classes_info``, or the names stored in its
                         image).
    :param legend_title: Optional legend title.
    :param legend_position: ``'right'`` (Default) or ``'bottom'`` of the
                            panels.
    :param axs: Optional axes to draw on (at least one per panel).
    :param out_plot_file: Optional output image file; the figure is saved and
                          left open (Default = None).
    :param out_dpi: Resolution of the saved image (Default = 300).
    :return: Tuple ``(fig, axs)``, with ``axs`` a NumPy array of the axis of
             each panel, in the order of ``panels``. The shared legend is
             ``fig.legends[-1]``; the colour bar of a ``'cmap'`` panel is
             ``axs[i].images[-1].colorbar``.

    .. code:: python

        import rsgispy.mapping.layout as map_layout

        fig, axs = map_layout.create_map_panels(
            [
                {"panel_type": "rgb", "input_img": "sen2.tif",
                 "bands": [8, 9, 3], "title": "Sentinel-2"},
                {"panel_type": "thematic", "input_img": "cls_raw.tif",
                 "title": "Raw classification"},
                {"panel_type": "thematic", "input_img": "cls_maj.tif",
                 "title": "Majority filter"},
            ],
            bbox=(262000, 268000, 276000, 282000),
            legend_title="Land cover",
        )

        # A change distance with a colour bar, and the change over the image.
        fig, axs = map_layout.create_map_panels(
            [
                map_layout.MapPanel(
                    "cmap", "cva_dist.tif", cmap_name="RdBu_r", vmin=-5,
                    vmax=5, show_colour_bar=True, cbar_label="Change distance",
                    title="CVA distance",
                ),
                map_layout.MapPanel(
                    "thematic", "cva_chng.tif", base_img="sen2_2018.tif",
                    base_bands=[5, 6, 4], title="Change over 2018",
                    in_legend=False,
                ),
            ],
            out_plot_file="cva_change.png",
        )
    """
    # pylint: disable=too-many-arguments,too-many-locals,too-many-branches
    # pylint: disable=import-outside-toplevel
    from rsgispy.mapping.elements import add_north_arrow, add_scale_bar

    panel_list = [_to_map_panel(p) for p in panels]
    if not panel_list:
        raise rsgispy.RSGISPyException("At least one map panel must be given.")
    if legend_position not in ("right", "bottom"):
        raise rsgispy.RSGISPyException(
            f"legend_position must be 'right' or 'bottom', not '{legend_position}'."
        )
    n_panels = len(panel_list)
    if bbox is None:
        bbox = _panel_bbox(panel_list[0])
    bbox = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))

    if axs is not None:
        ax_arr = np.asarray(axs, dtype=object).ravel()
        if ax_arr.size < n_panels:
            raise rsgispy.RSGISPyException(
                f"{n_panels} axes are needed but {ax_arr.size} were given."
            )
        fig = ax_arr[0].get_figure()
    else:
        n_cols = min(n_panels, 4) if ncols is None else max(1, int(ncols))
        n_cols = min(n_cols, n_panels)
        n_rows = int(np.ceil(n_panels / n_cols))
        if figsize is None:
            panel_h = min(max(calc_y_fig_size(bbox, _DEFAULT_PANEL_WIDTH), 2.5), 8.0)
            cbar_w = 1.0 if any(p.show_colour_bar for p in panel_list) else 0.0
            figsize = (
                n_cols * _DEFAULT_PANEL_WIDTH + cbar_w,
                n_rows * (panel_h + 0.4) + (0.4 if suptitle else 0.0),
            )
        fig, grid_axs = get_fig_axs(None, nrows=n_rows, ncols=n_cols, figsize=figsize)
        ax_arr = grid_axs.ravel()
        for unused_ax in ax_arr[n_panels:]:
            unused_ax.set_axis_off()
    panel_axs = np.asarray(list(ax_arr[:n_panels]), dtype=object)

    for ax, panel in zip(panel_axs, panel_list):
        _draw_map_panel(ax, panel, bbox, show_map_axis)

    if show_scale_bar:
        add_scale_bar(
            panel_axs[0],
            epsg_code=_panel_epsg(panel_list[0]),
            bbox=bbox,
            loc=scale_bar_loc,
        )
    if show_north_arrow:
        add_north_arrow(panel_axs[0], loc=north_arrow_loc)

    if show_legend:
        legend_classes = _shared_legend_classes(panel_list, bbox, classes_info)
        if legend_classes:
            _add_shared_legend(fig, legend_classes, legend_title, legend_position)

    if suptitle is not None:
        fig.suptitle(suptitle)
    if out_plot_file is not None:
        save_plot(fig, out_plot_file, out_dpi=out_dpi)
    return fig, panel_axs
