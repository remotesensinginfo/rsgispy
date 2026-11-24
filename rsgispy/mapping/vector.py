"""
The mapping.vector module provides functions for rendering vector layers,
choropleth maps, and point-density plots onto Matplotlib axes.
"""

from typing import Any, List, Optional, Sequence, Tuple, Union

import geopandas as gpd
import matplotlib
import matplotlib.axes
import matplotlib.colors

import rsgispy
from rsgispy.mapping.elements import (
    NORTH_ARROW_STYLE_TRIANGLE,
    SCALEBAR_STYLE_SOLID,
    _finalise_map_axis,
)

__all__ = [
    "create_vec_lyr_map",
    "create_choropleth_vec_lyr_map",
    "create_vec_pt_density_map",
]


def _get_vec_epsg(
    gp_vec: Union[gpd.GeoDataFrame, gpd.GeoSeries],
) -> Optional[int]:
    """Return EPSG integer code for a GeoDataFrame/GeoSeries, or ``None``."""
    if gp_vec.crs is None:
        return None
    return gp_vec.crs.to_epsg()


def _subset_vec_bbox(
    gp_vec: Union[gpd.GeoDataFrame, gpd.GeoSeries],
    bbox: Tuple[float, float, float, float],
) -> Union[gpd.GeoDataFrame, gpd.GeoSeries]:
    """Spatially subset a GeoDataFrame/GeoSeries to ``bbox``."""
    return gp_vec.cx[bbox[0] : bbox[1], bbox[2] : bbox[3]]


def _resolve_layer_param(
    param: Union[Any, Sequence[Any]],
    n_lyrs: int,
    idx: int,
    err_msg: str,
) -> Any:
    """Resolve a scalar or per-layer list parameter for multi-layer plotting."""
    if isinstance(param, list):
        if len(param) != n_lyrs:
            raise rsgispy.RSGISPyException(err_msg)
        return param[idx]
    return param


def create_vec_lyr_map(
    ax: matplotlib.axes.Axes,
    gp_vecs: Union[
        gpd.GeoDataFrame,
        gpd.GeoSeries,
        List[Union[gpd.GeoDataFrame, gpd.GeoSeries]],
    ],
    bbox: Tuple[float, float, float, float],
    title_str: Optional[str] = None,
    vec_fill_clrs: Union[str, List[str]] = "grey",
    vec_line_clrs: Union[str, List[str]] = "black",
    vec_line_widths: Union[float, List[float]] = 0.25,
    vec_fill_alphas: Union[float, List[float]] = 1.0,
    vec_markersize: Optional[Union[str, float]] = None,
    show_scale_bar: bool = True,
    use_grid: bool = False,
    show_map_axis: bool = True,
    sub_in_vec: bool = False,
    scale_bar_loc: str = "upper right",
    plot_zorders: Union[float, List[float]] = 1.0,
    scale_bar_style: str = SCALEBAR_STYLE_SOLID,
    scale_bar_units: Optional[str] = None,
    show_north_arrow: bool = False,
    north_arrow_style: str = NORTH_ARROW_STYLE_TRIANGLE,
    north_arrow_loc: str = "upper left",
) -> None:
    """
    Add one or more GeoPandas vector layers to ``ax``.

    :param ax: Matplotlib axis to which the vector layer(s) are added.
    :param gp_vecs: GeoDataFrame/GeoSeries or list of layers sharing the same CRS.
    :param bbox: Bounding box ``(MinX, MaxX, MinY, MaxY)`` for the display extent.
    :param title_str: Optional title for the map (Default = None).
    :param vec_fill_clrs: Fill colour string or list of colours (Default = ``'grey'``).
    :param vec_line_clrs: Edge colour string or list of colours (Default = ``'black'``).
    :param vec_line_widths: Line width float or list of widths (Default = 0.25).
    :param vec_fill_alphas: Alpha value or list of alpha values (Default = 1.0).
    :param vec_markersize: Constant marker size or column name for point layers.
    :param show_scale_bar: Whether to add a scale bar (Default = True).
    :param use_grid: Whether to enable axis gridlines (Default = False).
    :param show_map_axis: Whether to display axis ticks and frame (Default = True).
    :param sub_in_vec: Spatially subset vector layers to ``bbox`` before plotting.
    :param scale_bar_loc: Scale bar placement string (Default = ``'upper right'``).
    :param plot_zorders: Drawing z-order float or list of z-orders (Default = 1.0).
    :param scale_bar_style: Scale bar style constant
                            (Default = :data:`~rsgispy.mapping.elements.SCALEBAR_STYLE_SOLID`).
    :param scale_bar_units: Scale bar units (``'m'``, ``'km'``, or ``None`` for auto).
    :param show_north_arrow: Whether to add a north arrow (Default = False).
    :param north_arrow_style: North arrow style constant (Default =
                              :data:`~rsgispy.mapping.elements.NORTH_ARROW_STYLE_TRIANGLE`).
    :param north_arrow_loc: North arrow placement string (Default = ``'upper left'``).

    .. code:: python

        import geopandas as gpd
        import matplotlib.pyplot as plt
        import rsgispy.mapping.vector as map_vec

        fig, ax = plt.subplots()
        gdf = gpd.read_file("polys.gpkg")
        map_vec.create_vec_lyr_map(
            ax=ax,
            gp_vecs=gdf,
            bbox=(250000.0, 270000.0, 280000.0, 290000.0),
            title_str="Study Area",
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    vec_list = gp_vecs if isinstance(gp_vecs, list) else [gp_vecs]
    n_vec_lyrs = len(vec_list)

    for idx, gp_vec in enumerate(vec_list):
        gp_vec_sub = _subset_vec_bbox(gp_vec, bbox) if sub_in_vec else gp_vec
        vec_fill_clr = _resolve_layer_param(
            vec_fill_clrs,
            n_vec_lyrs,
            idx,
            "Number of fill colours not same as number of vector layers",
        )
        vec_line_clr = _resolve_layer_param(
            vec_line_clrs,
            n_vec_lyrs,
            idx,
            "Number of line colours not same as number of vector layers",
        )
        vec_line_width = _resolve_layer_param(
            vec_line_widths,
            n_vec_lyrs,
            idx,
            "Number of line widths not same as number of vector layers",
        )
        vec_fill_alpha = _resolve_layer_param(
            vec_fill_alphas,
            n_vec_lyrs,
            idx,
            "Number of fill alphas not same as number of vector layers",
        )
        plot_zorder = _resolve_layer_param(
            plot_zorders,
            n_vec_lyrs,
            idx,
            "Number of zorders not same as number of vector layers",
        )

        if not gp_vec_sub.empty:
            gp_vec_sub.plot(
                ax=ax,
                color=vec_fill_clr,
                edgecolor=vec_line_clr,
                linewidth=vec_line_width,
                alpha=vec_fill_alpha,
                markersize=vec_markersize,
                zorder=plot_zorder,
            )

    ax.set_aspect("equal")
    ax.set_xlim([bbox[0], bbox[1]])
    ax.set_ylim([bbox[2], bbox[3]])
    ax.ticklabel_format(useOffset=False, style="plain")

    vec_epsg = _get_vec_epsg(vec_list[0]) if vec_list else None
    _finalise_map_axis(
        ax=ax,
        epsg_code=vec_epsg,
        bbox=bbox,
        show_scale_bar=show_scale_bar,
        use_grid=use_grid,
        show_map_axis=show_map_axis,
        scale_bar_loc=scale_bar_loc,
        title_str=title_str,
        scale_bar_style=scale_bar_style,
        scale_bar_units=scale_bar_units,
        show_north_arrow=show_north_arrow,
        north_arrow_style=north_arrow_style,
        north_arrow_loc=north_arrow_loc,
    )


def create_choropleth_vec_lyr_map(
    ax: matplotlib.axes.Axes,
    gp_vec: gpd.GeoDataFrame,
    vec_col: str,
    bbox: Tuple[float, float, float, float],
    title_str: Optional[str] = None,
    vec_fill_cmap: Optional[Union[str, matplotlib.colors.Colormap]] = None,
    vec_var_norm: Optional[matplotlib.colors.Normalize] = None,
    vec_line_clr: str = "black",
    vec_line_width: float = 0.25,
    vec_fill_alpha: float = 1.0,
    vec_markersize: Optional[Union[str, float]] = None,
    show_scale_bar: bool = True,
    use_grid: bool = False,
    show_map_axis: bool = True,
    sub_in_vec: bool = False,
    scale_bar_loc: str = "upper right",
    plot_zorder: float = 1.0,
    scale_bar_style: str = SCALEBAR_STYLE_SOLID,
    scale_bar_units: Optional[str] = None,
    show_north_arrow: bool = False,
    north_arrow_style: str = NORTH_ARROW_STYLE_TRIANGLE,
    north_arrow_loc: str = "upper left",
) -> None:
    """
    Display a choropleth vector map coloured by ``vec_col`` on ``ax``.

    :param ax: Matplotlib axis to which the vector layer is added.
    :param gp_vec: GeoDataFrame containing features and attribute column ``vec_col``.
    :param vec_col: Column name used to colour features.
    :param bbox: Bounding box ``(MinX, MaxX, MinY, MaxY)`` for the display extent.
    :param title_str: Optional map title (Default = None).
    :param vec_fill_cmap: Matplotlib colour map name or object (Default = ``'viridis'``).
    :param vec_var_norm: Optional Matplotlib Normalize instance.
    :param vec_line_clr: Edge line colour (Default = ``'black'``).
    :param vec_line_width: Edge line width (Default = 0.25).
    :param vec_fill_alpha: Fill alpha transparency (Default = 1.0).
    :param vec_markersize: Constant marker size or column name for point layers.
    :param show_scale_bar: Whether to add a scale bar (Default = True).
    :param use_grid: Whether to enable axis gridlines (Default = False).
    :param show_map_axis: Whether to display axis ticks and frame (Default = True).
    :param sub_in_vec: Spatially subset vector layer to ``bbox`` before plotting.
    :param scale_bar_loc: Scale bar placement string (Default = ``'upper right'``).
    :param plot_zorder: Drawing z-order (Default = 1.0).
    :param scale_bar_style: Scale bar style constant
                            (Default = :data:`~rsgispy.mapping.elements.SCALEBAR_STYLE_SOLID`).
    :param scale_bar_units: Scale bar units (``'m'``, ``'km'``, or ``None`` for auto).
    :param show_north_arrow: Whether to add a north arrow (Default = False).
    :param north_arrow_style: North arrow style constant (Default =
                              :data:`~rsgispy.mapping.elements.NORTH_ARROW_STYLE_TRIANGLE`).
    :param north_arrow_loc: North arrow placement string (Default = ``'upper left'``).

    .. code:: python

        import geopandas as gpd
        import matplotlib.pyplot as plt
        import rsgispy.mapping.vector as map_vec

        fig, ax = plt.subplots()
        gdf = gpd.read_file("regions.gpkg")
        map_vec.create_choropleth_vec_lyr_map(
            ax=ax,
            gp_vec=gdf,
            vec_col="area_ha",
            bbox=(250000.0, 270000.0, 280000.0, 290000.0),
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    if vec_fill_cmap is None:
        vec_fill_cmap = matplotlib.colormaps["viridis"]

    gp_vec_sub = _subset_vec_bbox(gp_vec, bbox) if sub_in_vec else gp_vec

    if not gp_vec_sub.empty:
        gp_vec_sub.plot(
            column=vec_col,
            ax=ax,
            cmap=vec_fill_cmap,
            norm=vec_var_norm,
            edgecolor=vec_line_clr,
            linewidth=vec_line_width,
            alpha=vec_fill_alpha,
            markersize=vec_markersize,
            zorder=plot_zorder,
        )

    ax.set_aspect("equal")
    ax.set_xlim([bbox[0], bbox[1]])
    ax.set_ylim([bbox[2], bbox[3]])
    ax.ticklabel_format(useOffset=False, style="plain")

    _finalise_map_axis(
        ax=ax,
        epsg_code=_get_vec_epsg(gp_vec),
        bbox=bbox,
        show_scale_bar=show_scale_bar,
        use_grid=use_grid,
        show_map_axis=show_map_axis,
        scale_bar_loc=scale_bar_loc,
        title_str=title_str,
        scale_bar_style=scale_bar_style,
        scale_bar_units=scale_bar_units,
        show_north_arrow=show_north_arrow,
        north_arrow_style=north_arrow_style,
        north_arrow_loc=north_arrow_loc,
    )


def create_vec_pt_density_map(
    ax: matplotlib.axes.Axes,
    gp_vec: gpd.GeoDataFrame,
    bbox: Tuple[float, float, float, float],
    title_str: Optional[str] = None,
    cmap_name: str = "viridis",
    use_log_norm: bool = False,
    density_norm_vmin: float = 1.0,
    density_norm_vmax: Optional[float] = None,
    density_dpi: int = 72,
    show_scale_bar: bool = True,
    use_grid: bool = False,
    show_map_axis: bool = True,
    sub_in_vec: bool = False,
    scale_bar_loc: str = "upper right",
    plot_zorder: float = 1.0,
    scale_bar_style: str = SCALEBAR_STYLE_SOLID,
    scale_bar_units: Optional[str] = None,
    show_north_arrow: bool = False,
    north_arrow_style: str = NORTH_ARROW_STYLE_TRIANGLE,
    north_arrow_loc: str = "upper left",
) -> Tuple[matplotlib.colors.Colormap, matplotlib.colors.Normalize]:
    """
    Add a point-density plot from a GeoPandas points layer onto ``ax``.

    Note: ``ax`` must be created with ``projection='scatter_density'``, which
    requires the optional ``mpl_scatter_density`` package.

    :param ax: Matplotlib axis created with ``projection='scatter_density'``.
    :param gp_vec: GeoDataFrame containing point geometries.
    :param bbox: Bounding box ``(MinX, MaxX, MinY, MaxY)`` for the display extent.
    :param title_str: Optional map title (Default = None).
    :param cmap_name: Matplotlib colour map name (Default = ``'viridis'``).
    :param use_log_norm: Use logarithmic density normalisation (Default = False).
    :param density_norm_vmin: Minimum density normalisation value (Default = 1.0).
    :param density_norm_vmax: Maximum density normalisation value (Default = None).
    :param density_dpi: DPI used to bin point density (Default = 72).
    :param show_scale_bar: Whether to add a scale bar (Default = True).
    :param use_grid: Whether to enable axis gridlines (Default = False).
    :param show_map_axis: Whether to display axis ticks and frame (Default = True).
    :param sub_in_vec: Spatially subset vector layer to ``bbox`` before plotting.
    :param scale_bar_loc: Scale bar placement string (Default = ``'upper right'``).
    :param plot_zorder: Drawing z-order (Default = 1.0).
    :param scale_bar_style: Scale bar style constant
                            (Default = :data:`~rsgispy.mapping.elements.SCALEBAR_STYLE_SOLID`).
    :param scale_bar_units: Scale bar units (``'m'``, ``'km'``, or ``None`` for auto).
    :param show_north_arrow: Whether to add a north arrow (Default = False).
    :param north_arrow_style: North arrow style constant (Default =
                              :data:`~rsgispy.mapping.elements.NORTH_ARROW_STYLE_TRIANGLE`).
    :param north_arrow_loc: North arrow placement string (Default = ``'upper left'``).
    :return: Tuple ``(c_cmap, c_norm)`` for creating a Matplotlib colourbar.

    .. code:: python

        import geopandas as gpd
        import matplotlib.pyplot as plt
        import mpl_scatter_density  # noqa: F401
        import rsgispy.mapping.vector as map_vec

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection="scatter_density")
        pts_gdf = gpd.read_file("points.gpkg")
        cmap, norm = map_vec.create_vec_pt_density_map(
            ax=ax,
            gp_vec=pts_gdf,
            bbox=(250000.0, 270000.0, 280000.0, 290000.0),
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    # pylint: disable=import-outside-toplevel,import-error,unused-import
    try:
        import mpl_scatter_density  # noqa: F401

        has_msd = True
        msd_err: Optional[str] = None
    except ImportError as exc:
        has_msd = False
        msd_err = str(exc)
    rsgispy.check_optional_dep(
        has_msd,
        "create_vec_pt_density_map",
        "mpl-scatter-density",
        "mapping",
        error_detail=msd_err,
    )

    gp_vec_sub = _subset_vec_bbox(gp_vec, bbox) if sub_in_vec else gp_vec

    c_cmap = (
        matplotlib.colormaps[cmap_name].copy().with_extremes(under=[1, 1, 1, 0])
    )

    if use_log_norm:
        c_norm: matplotlib.colors.Normalize = matplotlib.colors.LogNorm(
            vmin=density_norm_vmin, vmax=density_norm_vmax
        )
    else:
        c_norm = matplotlib.colors.Normalize(
            vmin=density_norm_vmin, vmax=density_norm_vmax
        )

    ax.scatter_density(
        gp_vec_sub.geometry.x,
        gp_vec_sub.geometry.y,
        dpi=density_dpi,
        norm=c_norm,
        cmap=c_cmap,
        zorder=plot_zorder,
    )

    ax.set_aspect("equal")
    ax.set_xlim([bbox[0], bbox[1]])
    ax.set_ylim([bbox[2], bbox[3]])
    ax.ticklabel_format(useOffset=False, style="plain")

    _finalise_map_axis(
        ax=ax,
        epsg_code=_get_vec_epsg(gp_vec),
        bbox=bbox,
        show_scale_bar=show_scale_bar,
        use_grid=use_grid,
        show_map_axis=show_map_axis,
        scale_bar_loc=scale_bar_loc,
        title_str=title_str,
        scale_bar_style=scale_bar_style,
        scale_bar_units=scale_bar_units,
        show_north_arrow=show_north_arrow,
        north_arrow_style=north_arrow_style,
        north_arrow_loc=north_arrow_loc,
    )

    return c_cmap, c_norm
