"""
The tools.mapping module contains functions for creating maps with geospatial
raster and vector data using rasterio, geopandas, and matplotlib.
"""

# pylint: disable=too-many-lines
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import uuid
import xml.etree.ElementTree as ET

import geopandas
import matplotlib
import matplotlib.axes
import matplotlib.colors
from matplotlib import ticker
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
import numpy as np
import rasterio
import rasterio.shutil
from rasterio.warp import transform_bounds

import rsgispy
import rsgispy.imageutils.warp
import rsgispy.tools.filetools
import rsgispy.tools.plotting

__all__ = [
    "calc_y_fig_size",
    "define_axis_extent",
    "define_map_tick_spacing",
    "get_overview_info",
    "add_overview_maps",
    "add_contextily_basemap",
    "create_wmts_img_map",
    "draw_bboxes_to_axis",
    "add_axis_label",
    "create_vec_lyr_map",
    "create_raster_img_map",
    "create_thematic_raster_map",
    "create_choropleth_vec_lyr_map",
    "create_raster_cmap_img_map",
    "create_vec_pt_density_map",
]


def _great_circle_distance(
    wgs84_p1: Sequence[float],
    wgs84_p2: Sequence[float],
    earth_radius: float = 6378137.0,
) -> float:
    """
    Calculate the great-circle distance between two WGS84 ``(lon, lat)`` points.
    """
    if (wgs84_p1[1] < -90.0) or (wgs84_p1[1] > 90.0):
        raise rsgispy.RSGISPyException(
            "Point 1 latitude is not within valid range (-90 - 90)"
        )
    if (wgs84_p2[1] < -90.0) or (wgs84_p2[1] > 90.0):
        raise rsgispy.RSGISPyException(
            "Point 2 latitude is not within valid range (-90 - 90)"
        )
    if (wgs84_p1[0] < -180.0) or (wgs84_p1[0] > 180.0):
        raise rsgispy.RSGISPyException(
            "Point 1 longitude is not within valid range (-180 - 180)"
        )
    if (wgs84_p2[0] < -180.0) or (wgs84_p2[0] > 180.0):
        raise rsgispy.RSGISPyException(
            "Point 2 longitude is not within valid range (-180 - 180)"
        )

    lon1 = math.radians(wgs84_p1[0])
    lat1 = math.radians(wgs84_p1[1])
    lon2 = math.radians(wgs84_p2[0])
    lat2 = math.radians(wgs84_p2[1])

    lon_abs_diff = abs(lon1 - lon2)
    cos_angle = min(
        1.0,
        max(
            -1.0,
            math.sin(lat1) * math.sin(lat2)
            + math.cos(lat1) * math.cos(lat2) * math.cos(lon_abs_diff),
        ),
    )
    return earth_radius * math.acos(cos_angle)


def _get_img_bbox(input_img: str) -> Tuple[float, float, float, float]:
    """Return ``(MinX, MaxX, MinY, MaxY)`` for a raster using rasterio."""
    with rasterio.open(input_img, "r") as src:
        left, bottom, right, top = src.bounds
    return (float(left), float(right), float(bottom), float(top))


def _get_img_epsg(input_img: str) -> Optional[int]:
    """Return EPSG integer code for a raster using rasterio, or ``None``."""
    try:
        with rasterio.open(input_img, "r") as src:
            if src.crs is None:
                return None
            return src.crs.to_epsg()
    except Exception:  # pylint: disable=broad-exception-caught
        return None


def _get_vec_epsg(
    gp_vec: Union[geopandas.GeoDataFrame, geopandas.GeoSeries],
) -> Optional[int]:
    """Return EPSG integer code for a GeoDataFrame/GeoSeries, or ``None``."""
    if gp_vec.crs is None:
        return None
    return gp_vec.crs.to_epsg()


def _subset_vec_bbox(
    gp_vec: Union[geopandas.GeoDataFrame, geopandas.GeoSeries],
    bbox: Tuple[float, float, float, float],
) -> Union[geopandas.GeoDataFrame, geopandas.GeoSeries]:
    """Spatially subset a GeoDataFrame/GeoSeries using a buffered bounding box."""
    min_x_sub = math.floor(bbox[0] - 4)
    max_x_sub = math.ceil(bbox[1] + 4)
    min_y_sub = math.floor(bbox[2] - 4)
    max_y_sub = math.ceil(bbox[3] + 4)
    return gp_vec.cx[min_x_sub:max_x_sub, min_y_sub:max_y_sub]


def _add_scale_bar(
    ax: matplotlib.axes.Axes,
    epsg_code: Optional[int],
    bbox: Tuple[float, float, float, float],
    scale_bar_loc: str,
) -> None:
    """Add a ScaleBar artist to ``ax``, converting degrees to metres for EPSG:4326."""
    distance_meters = 1.0
    if epsg_code == 4326:
        distance_meters = _great_circle_distance(
            wgs84_p1=(bbox[0], bbox[3]),
            wgs84_p2=(bbox[0] + 1.0, bbox[3]),
        )
    ax.add_artist(ScaleBar(distance_meters, location=scale_bar_loc))


def _finalize_map_axis(
    ax: matplotlib.axes.Axes,
    epsg_code: Optional[int],
    bbox: Tuple[float, float, float, float],
    show_scale_bar: bool,
    use_grid: bool,
    show_map_axis: bool,
    scale_bar_loc: str,
    title_str: Optional[str],
) -> None:
    """Apply common map formatting options (grid, axis visibility, scale bar, title)."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    if use_grid:
        ax.grid()
    if not show_map_axis:
        ax.set_axis_off()
    if show_scale_bar:
        _add_scale_bar(ax, epsg_code, bbox, scale_bar_loc)
    if title_str is not None:
        ax.title.set_text(title_str)


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


def add_overview_maps(
    ax: matplotlib.axes.Axes,
    overview_lyr: geopandas.GeoDataFrame,
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
) -> matplotlib.axes.Axes:
    """
    Add an inset overview map to ``ax`` (defaulting to the top-left corner).

    :param ax: Parent matplotlib axis.
    :param overview_lyr: GeoDataFrame used as the overview map backdrop.
    :param overview_bbox: Bounding box ``(MinX, MaxX, MinY, MaxY)`` shown in
                          the inset axis.
    :param overview_pt: Optional ``(X, Y)`` marker coordinate on the overview map.
    :param over_size_x: Width of the inset axis in axes fraction coordinates.
    :param over_size_y: Height of the inset axis in axes fraction coordinates.
    :param over_x_off: X-axis offset of the inset axis.
    :param over_y_off: Y-axis offset of the inset axis.
    :param pt_clr: Colour of ``overview_pt`` marker (Default = ``'red'``).
    :param pt_size: Marker size for ``overview_pt`` (Default = 10.0).
    :param fill_clr: Polygon fill colour for ``overview_lyr`` (Default = ``'white'``).
    :param line_clr: Edge line colour for ``overview_lyr`` (Default = ``'black'``).
    :param line_width: Line width for ``overview_lyr`` (Default = 0.25).
    :return: Created inset matplotlib axis.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    ax_over = ax.inset_axes(
        [over_x_off, (1.0 - over_size_y) + over_y_off, over_size_x, over_size_y]
    )
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


def add_contextily_basemap(
    ax: matplotlib.axes.Axes,
    epsg_crs: int,
    cx_src: Any,
    cx_zoom_lvl: Union[int, str] = "auto",
    cx_attribution: Optional[Union[str, bool]] = None,
    cx_att_size: int = 8,
) -> None:
    """
    Add a contextily tile basemap beneath existing layers on ``ax``.

    :param ax: Matplotlib axis to which the basemap is added.
    :param epsg_crs: EPSG code of the coordinate reference system used on ``ax``.
    :param cx_src: Contextily tile provider (e.g., ``contextily.providers.OpenTopoMap``).
    :param cx_zoom_lvl: Tile zoom level or ``'auto'`` (Default = ``'auto'``).
    :param cx_attribution: Attribution string, ``False`` to suppress, or ``None``
                           for default attribution.
    :param cx_att_size: Attribution font size (Default = 8).
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,import-outside-toplevel
    import contextily

    if cx_zoom_lvl is None:
        cx_zoom_lvl = "auto"

    contextily.add_basemap(
        ax,
        zoom=cx_zoom_lvl,
        crs=f"epsg:{epsg_crs}",
        source=cx_src,
        attribution=cx_attribution,
        attribution_size=cx_att_size,
    )


def _update_wmts_xml(
    in_xml_path: str,
    out_xml_path: str,
    wmts_bbox: Tuple[float, float, float, float],
    wmts_zoom_level: Optional[int],
) -> None:
    """Update DataWindow bounds and optional ZoomLevel in a GDAL WMTS XML file."""
    tree = ET.parse(in_xml_path)
    gdal_wmts_tag = tree.getroot()
    if gdal_wmts_tag.tag != "GDAL_WMTS":
        raise rsgispy.RSGISPyException(
            "Something has gone wrong: Root not GDAL_WMTS."
        )
    data_win_tag = gdal_wmts_tag.find("DataWindow")
    if data_win_tag is None:
        raise rsgispy.RSGISPyException(
            "Something has gone wrong: No DataWindow tag."
        )
    for tag_name, val in (
        ("UpperLeftX", wmts_bbox[0]),
        ("UpperLeftY", wmts_bbox[3]),
        ("LowerRightX", wmts_bbox[1]),
        ("LowerRightY", wmts_bbox[2]),
    ):
        tag = data_win_tag.find(tag_name)
        if tag is None:
            raise rsgispy.RSGISPyException(
                f"Something has gone wrong: No {tag_name} tag."
            )
        tag.text = f"{val}"

    if wmts_zoom_level is not None:
        zoom_lvl_tag = ET.Element("ZoomLevel")
        zoom_lvl_tag.text = f"{wmts_zoom_level}"
        gdal_wmts_tag.append(zoom_lvl_tag)

    ET.indent(tree, space="    ", level=0)
    tree.write(out_xml_path)


def create_wmts_img_map(
    ax: matplotlib.axes.Axes,
    wmts_url: str,
    wmts_lyr: str,
    bbox: Tuple[float, float, float, float],
    bbox_epsg: int,
    wmts_zoom_level: Optional[int] = None,
    title_str: Optional[str] = None,
    show_scale_bar: bool = True,
    use_grid: bool = False,
    show_map_axis: bool = True,
    tmp_dir: Optional[str] = None,
    wmts_epsg: Optional[int] = None,
    scale_bar_loc: str = "upper right",
) -> None:
    """
    Download an image subset from a WMTS service via ``rasterio`` and display
    it on ``ax``.

    :param ax: Matplotlib axis to which the WMTS image is added.
    :param wmts_url: URL of the WMTS capabilities/service endpoint.
    :param wmts_lyr: Layer name within the WMTS service.
    :param bbox: Bounding box ``(MinX, MaxX, MinY, MaxY)`` specifying the
                 spatial region to display.
    :param bbox_epsg: EPSG code of ``bbox`` and the output map.
    :param wmts_zoom_level: Optional WMTS zoom level.
    :param title_str: Optional title for the map (Default = None).
    :param show_scale_bar: Whether to add a scale bar (Default = True).
    :param use_grid: Whether to enable axis gridlines (Default = False).
    :param show_map_axis: Whether to display axis ticks and frame (Default = True).
    :param tmp_dir: Optional directory for intermediate WMTS files.
    :param wmts_epsg: Optional EPSG code for the WMTS layer if auto-detection fails.
    :param scale_bar_loc: Scale bar placement string (Default = ``'upper right'``).
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    uid_str = uuid.uuid4().hex[:8]
    with rsgispy.tools.filetools.work_dir_context(tmp_dir) as work_dir:

        gdal_lyr_url = f"WMTS:{wmts_url},layer={wmts_lyr}"
        out_tmp_xml = os.path.join(work_dir, f"wmts_tmp_xml_{uid_str}.xml")
        rasterio.shutil.copy(  # pylint: disable=c-extension-no-member
            gdal_lyr_url, out_tmp_xml, driver="WMTS"
        )

        resolved_wmts_epsg = (
            wmts_epsg if wmts_epsg is not None else _get_img_epsg(out_tmp_xml)
        )
        if resolved_wmts_epsg is not None and resolved_wmts_epsg != bbox_epsg:
            w_min_x, w_min_y, w_max_x, w_max_y = transform_bounds(
                f"EPSG:{bbox_epsg}",
                f"EPSG:{resolved_wmts_epsg}",
                bbox[0],
                bbox[2],
                bbox[1],
                bbox[3],
            )
            wmts_bbox: Tuple[float, float, float, float] = (
                w_min_x,
                w_max_x,
                w_min_y,
                w_max_y,
            )
        else:
            wmts_bbox = bbox

        out_sub_xml = os.path.join(work_dir, f"wmts_tmp_sub_xml_{uid_str}.xml")
        _update_wmts_xml(out_tmp_xml, out_sub_xml, wmts_bbox, wmts_zoom_level)

        wmts_tmp_img = os.path.join(work_dir, f"wmts_tmp_img_{uid_str}.tif")
        if resolved_wmts_epsg is not None and resolved_wmts_epsg != bbox_epsg:
            wmts_raw_img = os.path.join(work_dir, f"wmts_raw_img_{uid_str}.tif")
            rasterio.shutil.copy(  # pylint: disable=c-extension-no-member
                out_sub_xml, wmts_raw_img, driver="GTiff"
            )
            rsgispy.imageutils.warp.reproject_image(
                input_img=wmts_raw_img,
                output_img=wmts_tmp_img,
                out_epsg=bbox_epsg,
                interp_method=rsgispy.INTERP_CUBIC,
                gdalformat="GTIFF",
            )
            if os.path.exists(wmts_raw_img):
                os.remove(wmts_raw_img)
        else:
            rasterio.shutil.copy(  # pylint: disable=c-extension-no-member
                out_sub_xml, wmts_tmp_img, driver="GTiff"
            )

        img_data, img_coords = rsgispy.tools.plotting.get_raster_mpl_imshow(
            wmts_tmp_img, bands=[1, 2, 3]
        )

        for tmp_path in (out_tmp_xml, out_sub_xml, wmts_tmp_img):
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    ax.imshow(img_data, extent=img_coords)
    ax.set_xlim([bbox[0], bbox[1]])
    ax.set_ylim([bbox[2], bbox[3]])
    _finalize_map_axis(
        ax=ax,
        epsg_code=bbox_epsg,
        bbox=bbox,
        show_scale_bar=show_scale_bar,
        use_grid=use_grid,
        show_map_axis=show_map_axis,
        scale_bar_loc=scale_bar_loc,
        title_str=title_str,
    )


def draw_bboxes_to_axis(
    ax: matplotlib.axes.Axes,
    bboxes: Sequence[Tuple[float, float, float, float]],
    bbox_labels: Optional[Sequence[str]] = None,
    rect_clr: str = "black",
    line_width: float = 1.0,
    fill_rect: bool = False,
    clr_alpha: float = 1.0,
    lbl_font_size: int = 12,
    lbl_font_weight: str = "normal",
    lbl_font_clr: str = "black",
    lbl_pos: Optional[str] = "centre",
    lbl_pos_buf: float = 0.0,
    lbl_fill_clr: Optional[str] = None,
    lbl_padding: float = 3.0,
) -> None:
    """
    Draw one or more bounding box rectangles and optional labels onto ``ax``.

    :param ax: Matplotlib axis on which rectangles are drawn.
    :param bboxes: Sequence of bboxes ``(MinX, MaxX, MinY, MaxY)``.
    :param bbox_labels: Optional sequence of text labels matching ``bboxes``.
    :param rect_clr: Edge/fill colour for the rectangles (Default = ``'black'``).
    :param line_width: Rectangle outline width (Default = 1.0).
    :param fill_rect: Whether to fill the rectangle interior (Default = False).
    :param clr_alpha: Alpha transparency for rectangles (Default = 1.0).
    :param lbl_font_size: Label font size (Default = 12).
    :param lbl_font_weight: Label font weight (Default = ``'normal'``).
    :param lbl_font_clr: Label text colour (Default = ``'black'``).
    :param lbl_pos: Label placement relative to each bbox: ``'centre'``, ``'above'``,
                    ``'below'``, ``'left'``, ``'right'``, or ``None``.
    :param lbl_pos_buf: Offset distance between rectangle edge and label.
    :param lbl_fill_clr: Optional background fill colour behind label text.
    :param lbl_padding: Padding around label text when ``lbl_fill_clr`` is set.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    if bbox_labels is not None and len(bboxes) != len(bbox_labels):
        raise rsgispy.RSGISPyException(
            "If labels are provided then the list length must be the same as bboxes."
        )

    for idx, bbox in enumerate(bboxes):
        width = bbox[1] - bbox[0]
        height = bbox[3] - bbox[2]
        bl_x = bbox[0]
        bl_y = bbox[2]
        rect = Rectangle(
            (bl_x, bl_y),
            width,
            height,
            fill=fill_rect,
            color=rect_clr,
            linewidth=line_width,
            alpha=clr_alpha,
        )
        ax.add_patch(rect)

        if bbox_labels is not None:
            cx = bl_x + width / 2.0
            cy = bl_y + height / 2.0
            ha_val = "center"
            va_val = "center"

            if lbl_pos is not None:
                pos_lower = lbl_pos.lower()
                if pos_lower == "above":
                    cy = bl_y + height + lbl_pos_buf
                    va_val = "bottom"
                elif pos_lower == "below":
                    cy = bl_y - lbl_pos_buf
                    va_val = "top"
                elif pos_lower == "left":
                    cx = bl_x - lbl_pos_buf
                    ha_val = "right"
                elif pos_lower == "right":
                    cx = bl_x + width + lbl_pos_buf
                    ha_val = "left"

            bbox_kw = (
                {"facecolor": lbl_fill_clr, "edgecolor": "none", "pad": lbl_padding}
                if lbl_fill_clr is not None
                else None
            )
            ax.annotate(
                bbox_labels[idx],
                (cx, cy),
                color=lbl_font_clr,
                weight=lbl_font_weight,
                fontsize=lbl_font_size,
                ha=ha_val,
                va=va_val,
                bbox=bbox_kw,
            )


def add_axis_label(
    ax: matplotlib.axes.Axes,
    lbl_text: str,
    lbl_font_clr: str = "black",
    lbl_font_weight: str = "normal",
    lbl_font_size: int = 12,
    fill_clr: str = "white",
    padding: float = 3.0,
) -> None:
    """
    Add a label in the top-left corner of ``ax`` (e.g. panel labels ``a``, ``b``).

    :param ax: Matplotlib axis to which the label is added.
    :param lbl_text: Label string to display.
    :param lbl_font_clr: Text colour (Default = ``'black'``).
    :param lbl_font_weight: Font weight (Default = ``'normal'``).
    :param lbl_font_size: Font size (Default = 12).
    :param fill_clr: Background box fill colour (Default = ``'white'``).
    :param padding: Padding around label text (Default = 3.0).
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    xmin, _ = ax.get_xlim()
    _, ymax = ax.get_ylim()
    ax.text(
        xmin,
        ymax,
        lbl_text,
        color=lbl_font_clr,
        weight=lbl_font_weight,
        fontsize=lbl_font_size,
        ha="left",
        va="top",
        bbox={"facecolor": fill_clr, "edgecolor": "none", "pad": padding},
    )


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
        geopandas.GeoDataFrame,
        geopandas.GeoSeries,
        List[Union[geopandas.GeoDataFrame, geopandas.GeoSeries]],
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
    _finalize_map_axis(
        ax=ax,
        epsg_code=vec_epsg,
        bbox=bbox,
        show_scale_bar=show_scale_bar,
        use_grid=use_grid,
        show_map_axis=show_map_axis,
        scale_bar_loc=scale_bar_loc,
        title_str=title_str,
    )


def create_raster_img_map(
    ax: matplotlib.axes.Axes,
    input_img: str,
    img_bands: List[int],
    img_stch: int,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    title_str: Optional[str] = None,
    show_scale_bar: bool = True,
    use_grid: bool = False,
    show_map_axis: bool = True,
    img_no_data_val: Optional[float] = None,
    no_data_out_clr: Optional[Union[float, Tuple[float, float, float]]] = None,
    stch_min_max_vals: Optional[Union[Dict[str, float], List[Dict[str, float]]]] = None,
    stch_n_stdevs: float = 2.0,
    img_alpha: float = 1.0,
    scale_bar_loc: str = "upper right",
    plot_zorder: float = 1.0,
) -> None:
    """
    Display a continuous 1-band or 3-band raster layer onto ``ax`` with stretching.

    :param ax: Matplotlib axis to which the image is added.
    :param input_img: Path to the input raster image file.
    :param img_bands: List of 1 or 3 1-based band indices.
    :param img_stch: Stretch constant (``rsgispy.IMG_STRETCH_*``).
    :param bbox: Optional bounding box ``(MinX, MaxX, MinY, MaxY)``. If ``None``,
                 the full image bounding box is used.
    :param title_str: Optional map title (Default = None).
    :param show_scale_bar: Whether to add a scale bar (Default = True).
    :param use_grid: Whether to enable axis gridlines (Default = False).
    :param show_map_axis: Whether to display axis ticks and frame (Default = True).
    :param img_no_data_val: Optional no-data value; read from image header if ``None``.
    :param no_data_out_clr: Optional colour value(s) assigned to no-data pixels.
    :param stch_min_max_vals: Stretch min/max dict(s) required when
                              ``img_stch == rsgispy.IMG_STRETCH_USER``.
    :param stch_n_stdevs: Number of standard deviations when
                          ``img_stch == rsgispy.IMG_STRETCH_STDEV`` (Default = 2.0).
    :param img_alpha: Display alpha transparency (Default = 1.0).
    :param scale_bar_loc: Scale bar placement string (Default = ``'upper right'``).
    :param plot_zorder: Drawing z-order (Default = 1.0).
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    if bbox is None:
        bbox = _get_img_bbox(input_img)

    if img_no_data_val is None:
        with rasterio.open(input_img, "r") as src:
            img_no_data_val = src.nodata

    img_data, img_coords = rsgispy.tools.plotting.get_raster_mpl_imshow(
        input_img, bands=img_bands, bbox=bbox
    )

    if img_stch == rsgispy.IMG_STRETCH_USER:
        if stch_min_max_vals is None:
            raise rsgispy.RSGISPyException(
                "Manual stretch requires user parameters to be passed"
            )
        img_data_strch = rsgispy.tools.plotting.manual_stretch_np_arr(
            img_data,
            min_max_vals=stch_min_max_vals,
            no_data_val=img_no_data_val,
            no_data_clr=no_data_out_clr,
        )
    elif img_stch == rsgispy.IMG_STRETCH_LINEAR:
        img_data_strch = rsgispy.tools.plotting.linear_stretch_np_arr(
            img_data,
            no_data_val=img_no_data_val,
            no_data_clr=no_data_out_clr,
        )
    elif img_stch == rsgispy.IMG_STRETCH_STDEV:
        img_data_strch = rsgispy.tools.plotting.stdev_stretch_np_arr(
            img_data,
            n_stdevs=stch_n_stdevs,
            no_data_val=img_no_data_val,
            no_data_clr=no_data_out_clr,
        )
    elif img_stch == rsgispy.IMG_STRETCH_CUMULATIVE:
        img_data_strch = rsgispy.tools.plotting.cumulative_stretch_np_arr(
            img_data,
            no_data_val=img_no_data_val,
            no_data_clr=no_data_out_clr,
        )
    else:
        img_data_strch = img_data

    clr_map = "gist_gray" if len(img_bands) == 1 else None
    ax.imshow(
        img_data_strch,
        cmap=clr_map,
        extent=img_coords,
        alpha=img_alpha,
        zorder=plot_zorder,
    )
    ax.set_xlim([img_coords[0], img_coords[1]])
    ax.set_ylim([img_coords[2], img_coords[3]])
    ax.ticklabel_format(useOffset=False, style="plain")

    _finalize_map_axis(
        ax=ax,
        epsg_code=_get_img_epsg(input_img),
        bbox=bbox,
        show_scale_bar=show_scale_bar,
        use_grid=use_grid,
        show_map_axis=show_map_axis,
        scale_bar_loc=scale_bar_loc,
        title_str=title_str,
    )


def create_thematic_raster_map(
    ax: matplotlib.axes.Axes,
    input_img: Union[str, List[str]],
    bbox: Optional[Tuple[float, float, float, float]] = None,
    title_str: Optional[str] = None,
    alpha_backgd: bool = True,
    img_alpha: float = 1.0,
    ctab_alpha: bool = True,
    show_scale_bar: bool = True,
    use_grid: bool = False,
    show_map_axis: bool = True,
    scale_bar_loc: str = "upper right",
    plot_zorders: Union[float, List[float]] = 1.0,
) -> None:
    """
    Display one or more thematic raster layers onto ``ax`` using the colour
    table stored in each image.

    :param ax: Matplotlib axis to which the thematic image(s) are added.
    :param input_img: Path or list of paths to thematic raster images.
    :param bbox: Optional bounding box ``(MinX, MaxX, MinY, MaxY)``. If ``None``,
                 the full extent of the first image is used.
    :param title_str: Optional map title (Default = None).
    :param alpha_backgd: Make pixel value 0 transparent (Default = True).
    :param img_alpha: Display alpha transparency (Default = 1.0).
    :param ctab_alpha: Use alpha channel values from the colour table (Default = True).
    :param show_scale_bar: Whether to add a scale bar (Default = True).
    :param use_grid: Whether to enable axis gridlines (Default = False).
    :param show_map_axis: Whether to display axis ticks and frame (Default = True).
    :param scale_bar_loc: Scale bar placement string (Default = ``'upper right'``).
    :param plot_zorders: Drawing z-order float or list of z-orders (Default = 1.0).
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    input_imgs = [input_img] if isinstance(input_img, str) else list(input_img)
    if not input_imgs:
        raise rsgispy.RSGISPyException("At least one input image must be provided.")

    if isinstance(plot_zorders, list):
        if len(plot_zorders) != len(input_imgs):
            raise rsgispy.RSGISPyException(
                "The number of zorders provided does not match the number of input images."
            )
        zorder_list = plot_zorders
    else:
        zorder_list = [plot_zorders] * len(input_imgs)

    if bbox is None:
        bbox = _get_img_bbox(input_imgs[0])

    img_data_arr_scns = []
    img_coords_scns = []
    for img_path in input_imgs:
        img_data_arr, img_coords, _ = (
            rsgispy.tools.plotting.get_thematic_raster_mpl_imshow(
                img_path,
                band=1,
                bbox=bbox,
                alpha_lyr=alpha_backgd,
                ctab_alpha=ctab_alpha,
            )
        )
        img_data_arr_scns.append(img_data_arr)
        img_coords_scns.append(img_coords)

    ax.set_xlim([img_coords_scns[0][0], img_coords_scns[0][1]])
    ax.set_ylim([img_coords_scns[0][2], img_coords_scns[0][3]])
    ax.ticklabel_format(useOffset=False, style="plain")

    for img_data_arr, img_coords, plot_zorder in zip(
        img_data_arr_scns, img_coords_scns, zorder_list
    ):
        ax.imshow(img_data_arr, extent=img_coords, alpha=img_alpha, zorder=plot_zorder)

    _finalize_map_axis(
        ax=ax,
        epsg_code=_get_img_epsg(input_imgs[0]),
        bbox=bbox,
        show_scale_bar=show_scale_bar,
        use_grid=use_grid,
        show_map_axis=show_map_axis,
        scale_bar_loc=scale_bar_loc,
        title_str=title_str,
    )


def create_choropleth_vec_lyr_map(
    ax: matplotlib.axes.Axes,
    gp_vec: geopandas.GeoDataFrame,
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
) -> None:
    """
    Display a choropleth vector map coloured by ``vec_col`` on ``ax``.

    :param ax: Matplotlib axis to which the vector layer is added.
    :param gp_vec: GeoDataFrame containing features and attribute column ``vec_col``.
    :param vec_col: Column name used to colour features.
    :param bbox: Bounding box ``(MinX, MaxX, MinY, MaxY)`` for the display extent.
    :param title_str: Optional map title (Default = None).
    :param vec_fill_cmap: Matplotlib colormap name or object (Default = ``'viridis'``).
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

    _finalize_map_axis(
        ax=ax,
        epsg_code=_get_vec_epsg(gp_vec),
        bbox=bbox,
        show_scale_bar=show_scale_bar,
        use_grid=use_grid,
        show_map_axis=show_map_axis,
        scale_bar_loc=scale_bar_loc,
        title_str=title_str,
    )


def create_raster_cmap_img_map(
    ax: matplotlib.axes.Axes,
    input_img: str,
    img_band: int,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    title_str: Optional[str] = None,
    show_scale_bar: bool = True,
    use_grid: bool = False,
    show_map_axis: bool = True,
    cmap_name: str = "viridis",
    norm_img_vals: bool = False,
    use_log_norm: bool = False,
    norm_vmin: Optional[float] = None,
    norm_vmax: Optional[float] = None,
    vals_under_white: bool = False,
    print_norm_vals: bool = False,
    scale_bar_loc: str = "upper right",
    plot_zorder: float = 1.0,
) -> Tuple[matplotlib.colors.Colormap, Optional[matplotlib.colors.Normalize]]:
    """
    Display a single-band raster layer on ``ax`` using a Matplotlib colormap and
    optional value normalisation.

    :param ax: Matplotlib axis to which the image is added.
    :param input_img: Path to the input raster image file.
    :param img_band: 1-based image band index.
    :param bbox: Optional bounding box ``(MinX, MaxX, MinY, MaxY)``. If ``None``,
                 the full image bounding box is used.
    :param title_str: Optional map title (Default = None).
    :param show_scale_bar: Whether to add a scale bar (Default = True).
    :param use_grid: Whether to enable axis gridlines (Default = False).
    :param show_map_axis: Whether to display axis ticks and frame (Default = True).
    :param cmap_name: Matplotlib colormap name (Default = ``'viridis'``).
    :param norm_img_vals: Whether to normalise pixel values (Default = False).
    :param use_log_norm: Whether to use logarithmic normalisation (Default = False).
    :param norm_vmin: Minimum normalisation value (auto-computed if ``None``).
    :param norm_vmax: Maximum normalisation value (auto-computed if ``None``).
    :param vals_under_white: Render values below ``norm_vmin`` as white.
    :param print_norm_vals: Print computed normalisation min/max values.
    :param scale_bar_loc: Scale bar placement string (Default = ``'upper right'``).
    :param plot_zorder: Drawing z-order (Default = 1.0).
    :return: Tuple ``(c_cmap, c_norm)`` for creating a Matplotlib colorbar.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    if bbox is None:
        bbox = _get_img_bbox(input_img)

    img_data, img_coords = rsgispy.tools.plotting.get_raster_mpl_imshow(
        input_img, bands=[img_band], bbox=bbox
    )

    c_cmap = plt.get_cmap(cmap_name).copy()
    c_norm: Optional[matplotlib.colors.Normalize] = None
    if norm_img_vals:
        if norm_vmin is None:
            norm_vmin = float(np.nanmin(img_data))
        if norm_vmax is None:
            norm_vmax = float(np.nanmax(img_data))

        if vals_under_white:
            c_cmap = c_cmap.with_extremes(under="white")

        if use_log_norm:
            c_norm = matplotlib.colors.LogNorm(vmin=norm_vmin, vmax=norm_vmax)
        else:
            c_norm = matplotlib.colors.Normalize(vmin=norm_vmin, vmax=norm_vmax)

        if print_norm_vals:
            print(f"{norm_vmin} - {norm_vmax}")

    ax.imshow(img_data, extent=img_coords, cmap=c_cmap, norm=c_norm, zorder=plot_zorder)
    ax.set_xlim([img_coords[0], img_coords[1]])
    ax.set_ylim([img_coords[2], img_coords[3]])
    ax.ticklabel_format(useOffset=False, style="plain")

    _finalize_map_axis(
        ax=ax,
        epsg_code=_get_img_epsg(input_img),
        bbox=bbox,
        show_scale_bar=show_scale_bar,
        use_grid=use_grid,
        show_map_axis=show_map_axis,
        scale_bar_loc=scale_bar_loc,
        title_str=title_str,
    )

    return c_cmap, c_norm


def create_vec_pt_density_map(
    ax: matplotlib.axes.Axes,
    gp_vec: geopandas.GeoDataFrame,
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
) -> Tuple[matplotlib.colors.Colormap, matplotlib.colors.Normalize]:
    """
    Add a point-density plot from a GeoPandas points layer onto ``ax``.

    Note: ``ax`` must be created with ``projection='scatter_density'``, which
    requires the optional ``mpl_scatter_density`` package.

    :param ax: Matplotlib axis created with ``projection='scatter_density'``.
    :param gp_vec: GeoDataFrame containing point geometries.
    :param bbox: Bounding box ``(MinX, MaxX, MinY, MaxY)`` for the display extent.
    :param title_str: Optional map title (Default = None).
    :param cmap_name: Matplotlib colormap name (Default = ``'viridis'``).
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
    :return: Tuple ``(c_cmap, c_norm)`` for creating a Matplotlib colorbar.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    # pylint: disable=import-outside-toplevel,import-error,unused-import
    import mpl_scatter_density  # noqa: F401

    gp_vec_sub = _subset_vec_bbox(gp_vec, bbox) if sub_in_vec else gp_vec

    c_cmap = plt.get_cmap(cmap_name).copy().with_extremes(under=[1, 1, 1, 0])

    if use_log_norm:
        c_norm: matplotlib.colors.Normalize = matplotlib.colors.LogNorm(
            vmin=density_norm_vmin, vmax=density_norm_vmax
        )
    else:
        c_norm = matplotlib.colors.Normalize(
            vmin=density_norm_vmin, vmax=density_norm_vmax
        )

    ax.scatter_density(
        gp_vec_sub["geometry"].x,
        gp_vec_sub["geometry"].y,
        dpi=density_dpi,
        norm=c_norm,
        cmap=c_cmap,
        zorder=plot_zorder,
    )

    ax.set_aspect("equal")
    ax.set_xlim([bbox[0], bbox[1]])
    ax.set_ylim([bbox[2], bbox[3]])
    ax.ticklabel_format(useOffset=False, style="plain")

    _finalize_map_axis(
        ax=ax,
        epsg_code=_get_vec_epsg(gp_vec),
        bbox=bbox,
        show_scale_bar=show_scale_bar,
        use_grid=use_grid,
        show_map_axis=show_map_axis,
        scale_bar_loc=scale_bar_loc,
        title_str=title_str,
    )

    return c_cmap, c_norm
