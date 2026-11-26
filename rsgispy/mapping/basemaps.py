"""
The mapping.basemaps module provides functions for adding contextily web-tile
basemaps and WMTS imagery layers to Matplotlib map axes.
"""

import os
from typing import Any, Optional, Tuple
import uuid
import xml.etree.ElementTree as ET

import matplotlib.axes
import rasterio
import rasterio.shutil
from rasterio.warp import transform_bounds

import rsgispy
import rsgispy.imageutils.metadata
import rsgispy.imageutils.warp
from rsgispy.mapping.elements import (
    NORTH_ARROW_STYLE_TRIANGLE,
    SCALEBAR_STYLE_SOLID,
    _finalise_map_axis,
)
import rsgispy.tools.files
import rsgispy.tools.plotting
from rsgispy.tools.plotting import use_plot_style

__all__ = [
    "add_contextily_basemap",
    "create_wmts_img_map",
]


@use_plot_style(plot_type="map")
def add_contextily_basemap(
    ax: matplotlib.axes.Axes,
    epsg_crs: int,
    cx_src: Any,
    cx_zoom_lvl: Any = "auto",
    cx_attribution: Optional[Any] = None,
    cx_att_size: int = 8,
) -> Any:
    """
    Add a contextily tile basemap beneath existing layers on ``ax``.

    :param ax: Matplotlib axis to which the basemap is added.
    :param epsg_crs: EPSG code of the coordinate reference system used on ``ax``.
    :param cx_src: Contextily tile provider (e.g., ``contextily.providers.OpenTopoMap``).
    :param cx_zoom_lvl: Tile zoom level or ``'auto'`` (Default = ``'auto'``).
    :param cx_attribution: Attribution string, ``False`` to suppress, or ``None``
                           for default attribution.
    :param cx_att_size: Attribution font size (Default = 8).
    :return: The basemap image artist (``matplotlib.image.AxesImage``).

    .. code:: python

        import contextily
        import matplotlib.pyplot as plt
        import rsgispy.mapping.basemaps
        import rsgispy.mapping.layout

        fig, ax = plt.subplots()
        rsgispy.mapping.layout.define_axis_extent(
            ax, bbox=(250000.0, 270000.0, 280000.0, 290000.0)
        )
        basemap_img = rsgispy.mapping.basemaps.add_contextily_basemap(
            ax=ax, epsg_crs=27700, cx_src=contextily.providers.OpenStreetMap.Mapnik
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,import-outside-toplevel
    try:
        import contextily

        have_contextily = True
    except ImportError:
        contextily = None
        have_contextily = False
    rsgispy.check_optional_dep(
        have_contextily, "add_contextily_basemap", "contextily", "mapping"
    )

    if cx_zoom_lvl is None:
        cx_zoom_lvl = "auto"

    n_imgs = len(ax.images)
    contextily.add_basemap(
        ax,
        zoom=cx_zoom_lvl,
        crs=f"epsg:{epsg_crs}",
        source=cx_src,
        attribution=cx_attribution,
        attribution_size=cx_att_size,
    )
    return ax.images[n_imgs] if len(ax.images) > n_imgs else None


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
        zoom_lvl_tag = gdal_wmts_tag.find("ZoomLevel")
        if zoom_lvl_tag is None:
            zoom_lvl_tag = ET.SubElement(gdal_wmts_tag, "ZoomLevel")
        zoom_lvl_tag.text = f"{wmts_zoom_level}"

    ET.indent(tree, space="    ", level=0)
    tree.write(out_xml_path)


@use_plot_style(plot_type="map")
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
    scale_bar_style: str = SCALEBAR_STYLE_SOLID,
    scale_bar_units: Optional[str] = None,
    show_north_arrow: bool = False,
    north_arrow_style: str = NORTH_ARROW_STYLE_TRIANGLE,
    north_arrow_loc: str = "upper left",
) -> Any:
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
    :param scale_bar_style: Scale bar style constant
                            (Default = :data:`~rsgispy.mapping.elements.SCALEBAR_STYLE_SOLID`).
    :param scale_bar_units: Scale bar units (``'m'``, ``'km'``, or ``None`` for auto).
    :param show_north_arrow: Whether to add a north arrow (Default = False).
    :param north_arrow_style: North arrow style constant (Default =
                              :data:`~rsgispy.mapping.elements.NORTH_ARROW_STYLE_TRIANGLE`).
    :param north_arrow_loc: North arrow placement string (Default = ``'upper left'``).
    :return: The image artist (``matplotlib.image.AxesImage``).

    .. code:: python

        import matplotlib.pyplot as plt
        import rsgispy.mapping.basemaps

        fig, ax = plt.subplots()
        img = rsgispy.mapping.basemaps.create_wmts_img_map(
            ax=ax,
            wmts_url="https://example.com/wmts/1.0.0/WMTSCapabilities.xml",
            wmts_lyr="ortho_layer",
            bbox=(250000.0, 260000.0, 280000.0, 290000.0),
            bbox_epsg=27700,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    uid_str = uuid.uuid4().hex[:8]
    with rsgispy.tools.files.work_dir_context(tmp_dir) as work_dir:

        gdal_lyr_url = f"WMTS:{wmts_url},layer={wmts_lyr}"
        out_tmp_xml = os.path.join(work_dir, f"wmts_tmp_xml_{uid_str}.xml")
        rasterio.shutil.copy(  # pylint: disable=c-extension-no-member
            gdal_lyr_url, out_tmp_xml, driver="WMTS"
        )

        resolved_wmts_epsg = (
            wmts_epsg
            if wmts_epsg is not None
            else rsgispy.imageutils.metadata.get_epsg_proj_from_img(out_tmp_xml)
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

        with rasterio.open(wmts_tmp_img, "r") as wmts_src:
            wmts_band_count = wmts_src.count
        if wmts_band_count >= 3:
            wmts_bands = [1, 2, 3]
            imshow_cmap: Optional[str] = None
        elif wmts_band_count >= 1:
            wmts_bands = [1]
            imshow_cmap = "gray"
        else:
            raise rsgispy.RSGISPyException(
                f"WMTS image '{wmts_tmp_img}' contains no raster bands."
            )

        img_data, img_coords = rsgispy.tools.plotting.get_raster_mpl_imshow(
            wmts_tmp_img, bands=wmts_bands
        )

        for tmp_path in (out_tmp_xml, out_sub_xml, wmts_tmp_img):
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    img_artist = ax.imshow(img_data, extent=img_coords, cmap=imshow_cmap)
    ax.set_xlim([bbox[0], bbox[1]])
    ax.set_ylim([bbox[2], bbox[3]])
    _finalise_map_axis(
        ax=ax,
        epsg_code=bbox_epsg,
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

    return img_artist
