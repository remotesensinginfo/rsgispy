"""
Tools for downloading OpenStreetMap (OSM) data.
"""

from importlib.util import find_spec
from typing import Any, Dict, List, Tuple, Union

import rsgispy
import rsgispy.tools.geometrytools

# pylint: disable=invalid-name
OSMNX_AVAIL = True
try:
    import osmnx
except ImportError:
    OSMNX_AVAIL = False

GEOPANDAS_AVAIL = find_spec("geopandas") is not None

PANDAS_AVAIL = True
try:
    import pandas
except ImportError:
    PANDAS_AVAIL = False
# pylint: enable=invalid-name

_OSM_GEOM_TYPE_MAP: Dict[int, str] = {
    rsgispy.GEOM_POLY: "Polygon",
    rsgispy.GEOM_PT: "Point",
    rsgispy.GEOM_LINE: "LineString",
}

__all__ = [
    "OSMNX_AVAIL",
    "GEOPANDAS_AVAIL",
    "PANDAS_AVAIL",
    "get_osm_geom_type_name",
    "get_osm_gdf",
    "get_osm_to_file",
    "get_osm_multi_to_file",
]


def get_osm_geom_type_name(geom_type: int) -> str:
    """
    Return the Shapely/OSM geometry type name for an ``rsgispy.GEOM_*`` constant.

    :param geom_type: Numerical geometry type (e.g., ``rsgispy.GEOM_POLY``).
    :return: Name of the geometry type (``"Polygon"``, ``"Point"``, or ``"LineString"``).
    """
    if geom_type not in _OSM_GEOM_TYPE_MAP:
        raise rsgispy.RSGISPyException(
            f"Do not recognise Geometry Type: '{geom_type}'"
        )
    return _OSM_GEOM_TYPE_MAP[geom_type]


def get_osm_gdf(
    bbox: Union[Tuple[float, float, float, float], List[float]],
    osm_tags: Dict[str, Union[bool, str]],
    rsgis_geom_type: int = rsgispy.GEOM_POLY,
    bbox_epsg: int = 4326,
    clip_to_roi: bool = False,
) -> Any:
    """
    Download OpenStreetMap data into a ``geopandas.GeoDataFrame``.

    :param bbox: Bounding box ``(x_min, x_max, y_min, y_max)``. Reprojected to
                 EPSG:4326 when ``bbox_epsg != 4326``.
    :param osm_tags: OSM tags to query (e.g., ``{"building": True}``).
    :param rsgis_geom_type: Target geometry type (``rsgispy.GEOM_PT``,
                            ``rsgispy.GEOM_LINE``, or ``rsgispy.GEOM_POLY``).
    :param bbox_epsg: EPSG code of ``bbox`` (default: 4326).
    :param clip_to_roi: If ``True``, clip returned geometries to ``bbox``.
    :return: ``geopandas.GeoDataFrame`` containing matching OSM features.
    """
    if not OSMNX_AVAIL:
        raise rsgispy.RSGISPyException(
            "osmnx module is not available, "
            "please install: https://osmnx.readthedocs.io"
        )
    if not GEOPANDAS_AVAIL:
        raise rsgispy.RSGISPyException(
            "geopandas module is not available, please install"
        )

    try:
        geom_type_str = get_osm_geom_type_name(rsgis_geom_type)

        if bbox_epsg != 4326:
            wgs84_bbox = rsgispy.tools.geometrytools.reproj_bbox_epsg(
                bbox, in_epsg=bbox_epsg, out_epsg=4326
            )
        else:
            wgs84_bbox = bbox

        # rsgispy bbox order: (West, East, South, North)
        # osmnx bbox order: (left, bottom, right, top)
        osm_bbox = (wgs84_bbox[0], wgs84_bbox[2], wgs84_bbox[1], wgs84_bbox[3])
        data_gdf = osmnx.features.features_from_bbox(bbox=osm_bbox, tags=osm_tags)

        data_gdf = data_gdf.loc[:, data_gdf.columns.str.contains("addr:|geometry")]
        data_gdf = data_gdf.loc[data_gdf.geometry.type == geom_type_str]

        if clip_to_roi:
            data_gdf = data_gdf.clip(osm_bbox)

        return data_gdf
    except Exception:
        print(f"Error: Unable to download OSM {osm_tags}")
        raise


def _write_osm_gdf(
    data_gdf: Any,
    out_vec_file: str,
    out_vec_lyr: str,
    out_format: str,
    out_vec_epsg: int,
) -> None:
    """Reproject and write a non-empty OSM GeoDataFrame to disk."""
    if out_vec_epsg != 4326:
        data_gdf = data_gdf.to_crs(epsg=out_vec_epsg)

    if len(data_gdf) > 0:
        if out_format == "GPKG":
            data_gdf.to_file(out_vec_file, layer=out_vec_lyr, driver=out_format)
        else:
            data_gdf.to_file(out_vec_file, driver=out_format)
    else:
        print("The geodataframe was empty - check the geometry type, tags and ROI.")


# pylint: disable=too-many-arguments,too-many-positional-arguments
def get_osm_to_file(
    out_vec_file: str,
    out_vec_lyr: str,
    bbox: Union[Tuple[float, float, float, float], List[float]],
    osm_tags: Dict[str, Union[bool, str]],
    rsgis_geom_type: int = rsgispy.GEOM_POLY,
    bbox_epsg: int = 4326,
    out_format: str = "GPKG",
    out_vec_epsg: int = 4326,
    clip_to_roi: bool = False,
) -> None:
    """
    Download OpenStreetMap data and write it to a vector file.

    :param out_vec_file: Output vector file path.
    :param out_vec_lyr: Output vector layer name (used for GPKG).
    :param bbox: Bounding box ``(x_min, x_max, y_min, y_max)``.
    :param osm_tags: OSM tags to query (e.g., ``{"building": True}``).
    :param rsgis_geom_type: Target geometry type (``rsgispy.GEOM_*``).
    :param bbox_epsg: EPSG code of ``bbox`` (default: 4326).
    :param out_format: Output vector driver (default: ``"GPKG"``).
    :param out_vec_epsg: Target EPSG code for output vector (default: 4326).
    :param clip_to_roi: If ``True``, clip returned geometries to ``bbox``.
    """
    data_gdf = get_osm_gdf(
        bbox=bbox,
        osm_tags=osm_tags,
        rsgis_geom_type=rsgis_geom_type,
        bbox_epsg=bbox_epsg,
        clip_to_roi=clip_to_roi,
    )
    _write_osm_gdf(data_gdf, out_vec_file, out_vec_lyr, out_format, out_vec_epsg)


# pylint: disable=too-many-arguments,too-many-positional-arguments
def get_osm_multi_to_file(
    out_vec_file: str,
    out_vec_lyr: str,
    bbox: Union[Tuple[float, float, float, float], List[float]],
    osm_tags_lst: List[Dict[str, Union[bool, str]]],
    rsgis_geom_type: int = rsgispy.GEOM_POLY,
    bbox_epsg: int = 4326,
    out_format: str = "GPKG",
    out_vec_epsg: int = 4326,
    clip_to_roi: bool = False,
) -> None:
    """
    Download OpenStreetMap data for multiple tag queries and merge into one file.

    :param out_vec_file: Output vector file path.
    :param out_vec_lyr: Output vector layer name (used for GPKG).
    :param bbox: Bounding box ``(x_min, x_max, y_min, y_max)``.
    :param osm_tags_lst: List of OSM tag dictionaries to query and merge.
    :param rsgis_geom_type: Target geometry type (``rsgispy.GEOM_*``).
    :param bbox_epsg: EPSG code of ``bbox`` (default: 4326).
    :param out_format: Output vector driver (default: ``"GPKG"``).
    :param out_vec_epsg: Target EPSG code for output vector (default: 4326).
    :param clip_to_roi: If ``True``, clip returned geometries to ``bbox``.
    """
    if not PANDAS_AVAIL:
        raise rsgispy.RSGISPyException(
            "pandas module is not available, please install"
        )

    data_lst = []
    for osm_tags in osm_tags_lst:
        try:
            data_gdf = get_osm_gdf(
                bbox=bbox,
                osm_tags=osm_tags,
                rsgis_geom_type=rsgis_geom_type,
                bbox_epsg=bbox_epsg,
                clip_to_roi=clip_to_roi,
            )
            if len(data_gdf) > 0:
                data_lst.append(data_gdf)
        except Exception:  # pylint: disable=broad-exception-caught
            print(
                f"Unable to download OSM data for tags: {osm_tags} "
                "- there might not be any data within the ROI."
            )

    if data_lst:
        data_gdf = pandas.concat(data_lst)
        data_gdf = data_gdf.set_crs(epsg=4326, allow_override=True)
        _write_osm_gdf(data_gdf, out_vec_file, out_vec_lyr, out_format, out_vec_epsg)
    else:
        print("The geodataframes were empty - check the geometry type, tags and ROI.")
