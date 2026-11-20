"""
Vector metadata, projection, bounding-box, layer inspection, and file management
utilities for :mod:`rsgispy.vector` using ``geopandas``, ``pyogrio``, and
``pyproj``.
"""

from typing import Dict, List, Optional

import pyogrio
import pyproj

import rsgispy
from rsgispy.tools.geometry import reproj_bbox
from rsgispy.vectorprocess import (
    VectorInput,
    bbox_to_rsgis,
    check_format_name,
    delete_vec_file,
    get_file_vec_extension,
    get_gdf_geom_type,
    is_geoparquet_file,
    list_vec_layers,
    read_vec_gdf,
    read_vec_info,
    resolve_vec_layer,
)

__all__ = [
    "VecLayersInfoObj",
    "check_format_name",
    "get_file_vec_extension",
    "delete_vec_file",
    "get_wkt_proj_from_vec",
    "get_epsg_proj_from_vec",
    "get_vec_lyr_bbox",
    "get_vec_lyr_bbox_in_proj",
    "get_vec_feat_count",
    "get_geom_type_name",
    "get_vec_lyr_geom_type",
    "get_vec_lyrs_lst",
    "get_vec_lyr_cols",
]

_GEOM_TYPE_NAMES: Dict[int, str] = {
    rsgispy.GEOM_POLY: "Polygon",
    rsgispy.GEOM_PT: "Point",
    rsgispy.GEOM_LINE: "Line",
    rsgispy.GEOM_MPOLY: "Multi-Polygon",
    rsgispy.GEOM_MPT: "Multi-Point",
    rsgispy.GEOM_MLINE: "Multi-Line",
}


class VecLayersInfoObj:
    """
    Store input file/layer and output layer metadata for
    :func:`rsgispy.vector.io.merge_to_multi_lyr_vec`.

    :param vec_file: Input vector file path or :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Input vector layer name.
    :param vec_out_lyr: Output vector layer name.

    .. code:: python

        import rsgispy.vector.io

        lyr_info = rsgispy.vector.io.VecLayersInfoObj(
            vec_file="in_data.gpkg",
            vec_lyr="polygons",
            vec_out_lyr="merged_polygons",
        )
    """

    # pylint: disable=too-few-public-methods
    def __init__(
        self,
        vec_file: Optional[VectorInput] = None,
        vec_lyr: Optional[str] = None,
        vec_out_lyr: Optional[str] = None,
    ) -> None:
        """Initialise a :class:`VecLayersInfoObj` descriptor."""
        self.vec_file = vec_file
        self.vec_lyr = vec_lyr
        self.vec_out_lyr = vec_out_lyr


def get_wkt_proj_from_vec(
    vec_file: VectorInput, vec_lyr: Optional[str] = None
) -> str:
    """
    Return the WKT projection string of ``vec_file`` (empty string if unset).

    :param vec_file: Input vector file path or :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Input vector layer name. If ``None``, reads the first layer.
    :return: WKT projection string.

    .. code:: python

        import rsgispy.vector.io

        wkt_str = rsgispy.vector.io.get_wkt_proj_from_vec("polys.gpkg", "polys")
    """
    info = read_vec_info(vec_file, vec_lyr)
    crs_val = info.get("crs")
    if not crs_val:
        return ""
    return pyproj.CRS.from_user_input(crs_val).to_wkt()


def get_epsg_proj_from_vec(
    vec_file: VectorInput, vec_lyr: Optional[str] = None
) -> Optional[int]:
    """
    Return the EPSG integer code for the CRS of ``vec_file``, or ``None`` if
    unavailable.

    :param vec_file: Input vector file path or :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Input vector layer name. If ``None``, reads the first layer.
    :return: Integer EPSG code or ``None``.

    .. code:: python

        import rsgispy.vector.io

        epsg_code = rsgispy.vector.io.get_epsg_proj_from_vec("polys.gpkg", "polys")
    """
    info = read_vec_info(vec_file, vec_lyr)
    crs_val = info.get("crs")
    if not crs_val:
        return None
    try:
        epsg_code = pyproj.CRS.from_user_input(crs_val).to_epsg()
        return int(epsg_code) if epsg_code is not None else None
    except Exception:  # pylint: disable=broad-exception-caught
        return None


def _crs_to_epsg(crs_val: object) -> Optional[int]:
    """Extract an integer EPSG code from a CRS-like object, or ``None``."""
    if not crs_val:
        return None
    try:
        epsg_code = pyproj.CRS.from_user_input(crs_val).to_epsg()
        return int(epsg_code) if epsg_code is not None else None
    except Exception:  # pylint: disable=broad-exception-caught
        return None


def get_vec_lyr_bbox(
    vec_file: VectorInput,
    vec_lyr: Optional[str] = None,
    compute_if_exp: bool = True,
) -> rsgispy.BBox:
    """
    Retrieve the bounding box of a vector layer in its native CRS.

    :param vec_file: Input vector file path or :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Input vector layer name. If ``None``, reads the first layer.
    :param compute_if_exp: If ``True`` (default), compute the exact bounds from
                           geometries if header bounds are unavailable.
    :return: :class:`rsgispy.BBox` ``(west, east, south, north)`` with ``epsg``
             populated when known.

    .. code:: python

        import rsgispy.vector.io

        bbox = rsgispy.vector.io.get_vec_lyr_bbox("polys.gpkg", "polys")
    """
    info = read_vec_info(vec_file, vec_lyr)
    epsg_code = _crs_to_epsg(info.get("crs"))
    bounds = info.get("total_bounds")
    if bounds is not None and not compute_if_exp:
        return bbox_to_rsgis(bounds, epsg=epsg_code)

    capabilities = info.get("capabilities") or {}
    if bounds is not None and capabilities.get("fast_total_bounds", False):
        return bbox_to_rsgis(bounds, epsg=epsg_code)

    gdf = read_vec_gdf(vec_file, vec_lyr=info["layer_name"])
    if len(gdf) == 0:
        raise rsgispy.RSGISPyException(
            f"Vector layer '{info['layer_name']}' is empty."
        )
    if epsg_code is None:
        epsg_code = _crs_to_epsg(gdf.crs)
    return bbox_to_rsgis(gdf.total_bounds, epsg=epsg_code)


def get_vec_lyr_bbox_in_proj(
    vec_file: VectorInput,
    out_epsg: int,
    vec_lyr: Optional[str] = None,
) -> rsgispy.BBox:
    """
    Retrieve the bounding box of a vector layer reprojected into ``out_epsg``.

    :param vec_file: Input vector file path or :class:`geopandas.GeoDataFrame`.
    :param out_epsg: Target EPSG integer code.
    :param vec_lyr: Input vector layer name. If ``None``, reads the first layer.
    :return: Reprojected :class:`rsgispy.BBox` ``(west, east, south, north)``
             with ``epsg=out_epsg``.

    .. code:: python

        import rsgispy.vector.io

        bbox = rsgispy.vector.io.get_vec_lyr_bbox_in_proj(
            "polys.gpkg", out_epsg=4326, vec_lyr="polys"
        )
    """
    in_wkt = get_wkt_proj_from_vec(vec_file, vec_lyr=vec_lyr)
    if not in_wkt:
        raise rsgispy.RSGISPyException(
            "Input vector layer does not have a CRS defined."
        )
    vec_bbox = get_vec_lyr_bbox(vec_file, vec_lyr=vec_lyr)
    return reproj_bbox(
        vec_bbox,
        pyproj.CRS.from_wkt(in_wkt),
        pyproj.CRS.from_epsg(int(out_epsg)),
    )


def get_vec_feat_count(
    vec_file: VectorInput,
    vec_lyr: Optional[str] = None,
    compute_count: bool = True,
) -> int:
    """
    Return the number of features in a vector layer.

    :param vec_file: Input vector file path or :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Input vector layer name. If ``None``, reads the first layer.
    :param compute_count: If ``True`` (default), compute the exact count when
                          header metadata does not store it.
    :return: Feature count integer.

    .. code:: python

        import rsgispy.vector.io

        n_feats = rsgispy.vector.io.get_vec_feat_count("polys.gpkg", "polys")
    """
    info = read_vec_info(vec_file, vec_lyr)
    feat_count = int(info.get("features", -1))
    if feat_count >= 0:
        return feat_count
    if not compute_count:
        return feat_count
    if is_geoparquet_file(vec_file):  # type: ignore[arg-type]
        return int(len(read_vec_gdf(vec_file, vec_lyr=info["layer_name"])))
    df = pyogrio.read_dataframe(
        vec_file,
        layer=info["layer_name"],
        read_geometry=False,
        columns=[],
    )
    return int(len(df))


def get_geom_type_name(geom_type: int) -> str:
    """
    Return a human-readable geometry name for an ``rsgispy.GEOM_*`` constant.

    :param geom_type: Geometry constant (e.g., ``rsgispy.GEOM_POLY``).
    :return: Geometry type name (e.g., ``'Polygon'``).

    .. code:: python

        import rsgispy
        import rsgispy.vector.io

        name = rsgispy.vector.io.get_geom_type_name(rsgispy.GEOM_POLY)
    """
    if geom_type not in _GEOM_TYPE_NAMES:
        raise rsgispy.RSGISPyException(
            f"Do not recognise Geometry Type: '{geom_type}'"
        )
    return _GEOM_TYPE_NAMES[geom_type]


def get_vec_lyr_geom_type(
    vec_file: VectorInput, vec_lyr: Optional[str] = None
) -> int:
    """
    Return the ``rsgispy.GEOM_*`` constant describing the geometry type of a
    vector layer.

    :param vec_file: Input vector file path or :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Input vector layer name. If ``None``, reads the first layer.
    :return: Integer geometry type constant from ``rsgispy.GEOM_*``.

    .. code:: python

        import rsgispy.vector.io

        geom_type = rsgispy.vector.io.get_vec_lyr_geom_type("polys.gpkg", "polys")
    """
    lyr_name = resolve_vec_layer(vec_file, vec_lyr)
    gdf = read_vec_gdf(vec_file, vec_lyr=lyr_name)
    return get_gdf_geom_type(gdf)


def get_vec_lyrs_lst(vec_file: VectorInput) -> List[str]:
    """
    Return the list of layer names stored in ``vec_file``.

    :param vec_file: Input vector file path or :class:`geopandas.GeoDataFrame`.
    :return: List of layer name strings.

    .. code:: python

        import rsgispy.vector.io

        layers = rsgispy.vector.io.get_vec_lyrs_lst("dataset.gpkg")
    """
    layers = list_vec_layers(vec_file)
    out_lyrs: List[str] = []
    for name in layers:
        if name not in out_lyrs:
            out_lyrs.append(name)
    return out_lyrs


def get_vec_lyr_cols(
    vec_file: VectorInput, vec_lyr: Optional[str] = None
) -> List[str]:
    """
    Return the list of attribute column names in a vector layer.

    :param vec_file: Input vector file path or :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Input vector layer name. If ``None``, reads the first layer.
    :return: List of attribute column names.

    .. code:: python

        import rsgispy.vector.io

        cols = rsgispy.vector.io.get_vec_lyr_cols("polys.gpkg", "polys")
    """
    info = read_vec_info(vec_file, vec_lyr)
    fields = info.get("fields")
    if fields is None:
        return []
    return [str(col) for col in fields]
