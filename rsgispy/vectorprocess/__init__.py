"""
Shared low-level vector processing helpers for :mod:`rsgispy` using
``geopandas``, ``pyogrio``, ``shapely``, and ``pyproj``.
"""

import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import geopandas
import numpy
import pyogrio
import pyproj
import shapely.geometry
from shapely.geometry.base import BaseGeometry

import rsgispy
from rsgispy.imageprocess import ensure_file_exists
import rsgispy.tools.files

CRSLike = Union[pyproj.CRS, int, str]

_LEGACY_OGR_INT_TYPES = {0, 12}
_LEGACY_OGR_FLOAT_TYPES = {2}
_LEGACY_OGR_STR_TYPES = {4}

_VEC_FORMAT_TO_EXT: Dict[str, str] = {
    "gpkg": "gpkg",
    "geojson": "geojson",
    "esri shapefile": "shp",
    "kml": "kml",
    "gml": "gml",
    "flatgeobuf": "fgb",
    "parquet": "parquet",
}


def check_format_name(vec_file_format: str) -> str:
    """
    Normalise vector driver names for ``geopandas`` / ``pyogrio``
    (e.g. ``'GEOJSON'`` -> ``'GeoJSON'``).

    :param vec_file_format: Input vector driver name.
    :return: Normalised driver name string.

    .. code:: python

        import rsgispy.vectorprocess

        fmt = rsgispy.vectorprocess.check_format_name("GEOJSON")
    """
    if vec_file_format.lower() == "geojson":
        return "GeoJSON"
    return vec_file_format


def get_file_vec_extension(vec_format: str) -> str:
    """
    Return the standard file extension (without leading dot) for ``vec_format``.

    Supports ``GPKG``, ``GeoJSON``, ``ESRI Shapefile``, ``KML``, ``GML``,
    ``FlatGeobuf``, and ``Parquet``.

    :param vec_format: Vector driver name string (e.g., ``'GPKG'`` or
                       ``'GeoJSON'``).
    :return: Lowercase file extension string without leading dot.

    .. code:: python

        import rsgispy.vectorprocess

        ext = rsgispy.vectorprocess.get_file_vec_extension("GPKG")
    """
    fmt_key = vec_format.lower().strip()
    if fmt_key not in _VEC_FORMAT_TO_EXT:
        raise rsgispy.RSGISPyException(
            f"The extension for the format ({vec_format}) specified is unknown."
        )
    return _VEC_FORMAT_TO_EXT[fmt_key]


def delete_vec_file(vec_file: str, feedback: bool = False) -> None:
    """
    Delete an existing vector file and any associated shapefile sidecars.

    :param vec_file: Vector file path to delete.
    :param feedback: If ``True``, print each deleted file path
                     (Default = ``False``).

    .. code:: python

        import rsgispy.vectorprocess

        rsgispy.vectorprocess.delete_vec_file("tmp_layer.gpkg", feedback=False)
    """
    if not os.path.exists(vec_file):
        return

    if vec_file.lower().endswith(".shp"):
        rsgispy.tools.files.delete_file_with_basename(
            vec_file, print_rms=feedback
        )
    else:
        if feedback:
            print(f"Deleting: {vec_file}")
        rsgispy.tools.files.delete_file_silent(vec_file)


def prepare_out_vec_file(
    out_vec_file: str,
    del_exist_vec: bool = False,
) -> None:
    """
    Validate output path availability and delete an existing file (including
    shapefile sidecars) when ``del_exist_vec=True``.

    :param out_vec_file: Target output vector path.
    :param del_exist_vec: If ``True``, delete any existing file at
                          ``out_vec_file`` (Default = ``False``).

    .. code:: python

        import rsgispy.vectorprocess

        rsgispy.vectorprocess.prepare_out_vec_file("out.gpkg", del_exist_vec=True)
    """
    if os.path.exists(out_vec_file):
        if del_exist_vec:
            delete_vec_file(out_vec_file, feedback=False)
        else:
            raise rsgispy.RSGISPyException(
                f"The output vector file ({out_vec_file}) already exists, "
                "remove it and re-run."
            )


def resolve_vec_layer(vec_file: str, vec_lyr: Optional[str] = None) -> str:
    """
    Validate ``vec_file`` exists and resolve ``vec_lyr`` (returning the first
    layer when ``vec_lyr`` is ``None``).

    :param vec_file: Input vector file path.
    :param vec_lyr: Optional vector layer name.
    :return: Resolved layer name string.

    .. code:: python

        import rsgispy.vectorprocess

        lyr = rsgispy.vectorprocess.resolve_vec_layer("polys.gpkg")
    """
    ensure_file_exists(vec_file, label="Vector file")
    try:
        layers = pyogrio.list_layers(vec_file)
    except Exception as err:
        raise rsgispy.RSGISPyException(
            f"Could not open vector file '{vec_file}': {err}"
        ) from err

    if len(layers) == 0:
        raise rsgispy.RSGISPyException(
            f"No vector layers found in file '{vec_file}'."
        )

    lyr_names = [str(row[0]) for row in layers]
    if vec_lyr is None:
        return lyr_names[0]
    if vec_lyr not in lyr_names:
        raise rsgispy.RSGISPyException(
            f"Could not find vector layer '{vec_lyr}' in '{vec_file}'."
        )
    return vec_lyr


def resolve_out_lyr_name(
    out_vec_file: str,
    out_vec_lyr: Optional[str] = None,
) -> str:
    """
    Return ``out_vec_lyr`` if given, otherwise derive a valid layer name from
    ``out_vec_file``.

    :param out_vec_file: Output vector file path.
    :param out_vec_lyr: Optional output layer name.
    :return: Layer name string.

    .. code:: python

        import rsgispy.vectorprocess

        lyr = rsgispy.vectorprocess.resolve_out_lyr_name("polys.gpkg", None)
    """
    if out_vec_lyr is not None:
        return out_vec_lyr
    return rsgispy.tools.files.get_file_basename(out_vec_file, check_valid=True)


def read_vec_info(vec_file: str, vec_lyr: Optional[str] = None) -> Dict[str, Any]:
    """
    Return ``pyogrio.read_info`` metadata dictionary for ``vec_file`` /
    ``vec_lyr``.

    :param vec_file: Input vector file path.
    :param vec_lyr: Optional layer name (first layer if ``None``).
    :return: Metadata dictionary from ``pyogrio.read_info``.

    .. code:: python

        import rsgispy.vectorprocess

        info = rsgispy.vectorprocess.read_vec_info("polys.gpkg", "polys")
    """
    lyr_name = resolve_vec_layer(vec_file, vec_lyr)
    try:
        return pyogrio.read_info(vec_file, layer=lyr_name)
    except Exception as err:
        raise rsgispy.RSGISPyException(
            f"Could not inspect layer '{lyr_name}' in '{vec_file}': {err}"
        ) from err


def read_vec_gdf(
    vec_file: str,
    vec_lyr: Optional[str] = None,
    epsg: Optional[int] = None,
) -> geopandas.GeoDataFrame:
    """
    Read a vector file/layer into a :class:`geopandas.GeoDataFrame`, raising
    :class:`rsgispy.RSGISPyException` on failure.

    :param vec_file: Input vector file path.
    :param vec_lyr: Optional layer name.
    :param epsg: Optional EPSG code to assign (override without reprojecting).
    :return: Loaded :class:`geopandas.GeoDataFrame`.

    .. code:: python

        import rsgispy.vectorprocess

        gdf = rsgispy.vectorprocess.read_vec_gdf("polys.gpkg", "polys")
    """
    ensure_file_exists(vec_file, label="Vector file")
    read_kw = {"layer": vec_lyr} if vec_lyr is not None else {}
    try:
        gdf = geopandas.read_file(vec_file, **read_kw)
    except Exception as err:
        raise rsgispy.RSGISPyException(
            f"Could not open '{vec_file}' (layer='{vec_lyr}'): {err}"
        ) from err

    if epsg is not None:
        gdf = gdf.set_crs(epsg=int(epsg), allow_override=True)
    return gdf


def write_vec_gdf(
    gdf: geopandas.GeoDataFrame,
    out_vec_file: str,
    out_vec_lyr: Optional[str] = None,
    out_format: Optional[str] = None,
    overwrite: bool = False,
) -> None:
    """
    Write a :class:`geopandas.GeoDataFrame` to disk, handling GPKG layer naming,
    driver normalisation, and optional file deletion when ``overwrite=True``.

    :param gdf: GeoDataFrame to write.
    :param out_vec_file: Destination vector file path.
    :param out_vec_lyr: Optional destination layer name (used for GPKG).
    :param out_format: Optional vector driver name.
    :param overwrite: If ``True``, delete an existing ``out_vec_file`` before
                      writing (Default = ``False``).

    .. code:: python

        import rsgispy.vectorprocess

        rsgispy.vectorprocess.write_vec_gdf(
            gdf, "out.gpkg", out_vec_lyr="polys", out_format="GPKG", overwrite=True
        )
    """
    if overwrite and os.path.exists(out_vec_file):
        delete_vec_file(out_vec_file, feedback=False)

    fmt = check_format_name(out_format) if out_format is not None else None
    is_gpkg = (
        fmt == "GPKG"
        if fmt is not None
        else out_vec_file.lower().endswith(".gpkg")
    )

    write_kw: Dict[str, Any] = {}
    if fmt is not None:
        write_kw["driver"] = fmt
    if is_gpkg:
        write_kw["layer"] = resolve_out_lyr_name(out_vec_file, out_vec_lyr)

    try:
        gdf.to_file(out_vec_file, **write_kw)
    except Exception as err:
        raise rsgispy.RSGISPyException(
            f"Could not write vector file '{out_vec_file}': {err}"
        ) from err


def validate_columns_exist(
    gdf: geopandas.GeoDataFrame,
    columns: Sequence[str],
) -> None:
    """
    Raise :class:`rsgispy.RSGISPyException` if any requested column is absent.

    :param gdf: Input GeoDataFrame.
    :param columns: Sequence of required column names.

    .. code:: python

        import rsgispy.vectorprocess

        rsgispy.vectorprocess.validate_columns_exist(gdf, ["class_id", "area"])
    """
    for col in columns:
        if col not in gdf.columns:
            raise rsgispy.RSGISPyException(
                f"Could not find the attribute ('{col}') specified "
                "within the vector layer."
            )


def bbox_to_rsgis(
    bounds: Sequence[float],
) -> Tuple[float, float, float, float]:
    """
    Convert a Shapely/GeoPandas ``(minx, miny, maxx, maxy)`` bounds tuple to an
    ``rsgispy`` ``(MinX, MaxX, MinY, MaxY)`` bounding box.

    :param bounds: Four-element bounds ``(minx, miny, maxx, maxy)``.
    :return: Bounding box ``(MinX, MaxX, MinY, MaxY)``.

    .. code:: python

        import rsgispy.vectorprocess

        bbox = rsgispy.vectorprocess.bbox_to_rsgis([0.0, 10.0, 50.0, 60.0])
    """
    return (
        float(bounds[0]),
        float(bounds[2]),
        float(bounds[1]),
        float(bounds[3]),
    )


def get_gdf_bbox(
    vec_gdf: geopandas.GeoDataFrame,
) -> Tuple[float, float, float, float]:
    """
    Return ``(MinX, MaxX, MinY, MaxY)`` bounding box for a non-empty
    :class:`geopandas.GeoDataFrame`.

    :param vec_gdf: Input GeoDataFrame.
    :return: Bounding box ``(MinX, MaxX, MinY, MaxY)``.

    .. code:: python

        import rsgispy.vectorprocess

        bbox = rsgispy.vectorprocess.get_gdf_bbox(gdf)
    """
    if vec_gdf.empty:
        raise rsgispy.RSGISPyException(
            "The inputted vector layer contains no features."
        )
    min_x, min_y, max_x, max_y = vec_gdf.total_bounds
    if not all(math.isfinite(val) for val in (min_x, min_y, max_x, max_y)):
        raise rsgispy.RSGISPyException(
            "Could not compute valid bounding box for the vector layer."
        )
    return bbox_to_rsgis((min_x, min_y, max_x, max_y))


def parse_crs_input(crs_val: CRSLike) -> pyproj.CRS:
    """
    Parse an EPSG integer, WKT/PROJ string, or ``pyproj.CRS`` into a
    ``pyproj.CRS``.

    :param crs_val: CRS descriptor.
    :return: Parsed ``pyproj.CRS`` object.

    .. code:: python

        import rsgispy.vectorprocess

        crs = rsgispy.vectorprocess.parse_crs_input(4326)
    """
    try:
        return pyproj.CRS.from_user_input(crs_val)
    except Exception as err:
        raise rsgispy.RSGISPyException(
            f"Could not parse CRS '{crs_val}': {err}"
        ) from err


def resolve_vector_crs(
    vec_gdf: geopandas.GeoDataFrame,
    vec_def_epsg: Optional[int] = None,
) -> pyproj.CRS:
    """
    Resolve the CRS from ``vec_def_epsg`` or ``vec_gdf.crs``.

    :param vec_gdf: Input GeoDataFrame.
    :param vec_def_epsg: Optional fallback/override EPSG code.
    :return: Resolved ``pyproj.CRS`` object.

    .. code:: python

        import rsgispy.vectorprocess

        crs = rsgispy.vectorprocess.resolve_vector_crs(gdf)
    """
    if vec_def_epsg is not None:
        return pyproj.CRS.from_epsg(int(vec_def_epsg))
    if vec_gdf.crs is None:
        raise rsgispy.RSGISPyException(
            "Could not retrieve a projection object from the vector layer - "
            "projection might not be defined."
        )
    return pyproj.CRS.from_user_input(vec_gdf.crs)


def get_gdf_geom_type(gdf: geopandas.GeoDataFrame) -> int:
    """
    Map :class:`geopandas.GeoDataFrame` geometry types to an ``rsgispy.GEOM_*``
    constant.

    :param gdf: Input GeoDataFrame.
    :return: Integer geometry constant from ``rsgispy.GEOM_*``.

    .. code:: python

        import rsgispy.vectorprocess

        geom_type = rsgispy.vectorprocess.get_gdf_geom_type(gdf)
    """
    geom_types = set(gdf.geometry.geom_type.dropna().unique())
    if not geom_types:
        raise rsgispy.RSGISPyException("Vector layer contains no valid geometries.")
    if geom_types == {"Point"}:
        return rsgispy.GEOM_PT
    if geom_types == {"LineString"}:
        return rsgispy.GEOM_LINE
    if geom_types == {"Polygon"}:
        return rsgispy.GEOM_POLY
    if geom_types == {"MultiPoint"}:
        return rsgispy.GEOM_MPT
    if geom_types == {"MultiLineString"}:
        return rsgispy.GEOM_MLINE
    if geom_types == {"MultiPolygon"}:
        return rsgispy.GEOM_MPOLY
    if geom_types.issubset({"Polygon", "MultiPolygon"}):
        return rsgispy.GEOM_MPOLY
    if geom_types.issubset({"LineString", "MultiLineString"}):
        return rsgispy.GEOM_MLINE
    if geom_types.issubset({"Point", "MultiPoint"}):
        return rsgispy.GEOM_MPT
    raise rsgispy.RSGISPyException(
        f"Unrecognised geometry types: {sorted(geom_types)}"
    )


def cast_col_data(
    att_col_data: Sequence[Any],
    att_col_datatype: Any,
) -> List[Any]:
    """
    Cast sequence values according to ``att_col_datatype`` (Python type, string
    name such as ``'int'``/``'float'``/``'str'``, NumPy dtype, legacy integer
    type code, or ``None``).

    :param att_col_data: Sequence of attribute values.
    :param att_col_datatype: Target type specifier or ``None``.
    :return: List of cast values.

    .. code:: python

        import rsgispy.vectorprocess

        vals = rsgispy.vectorprocess.cast_col_data(["1", "2"], int)
    """
    if att_col_datatype is None:
        return list(att_col_data)

    if att_col_datatype in (int, numpy.integer) or (
        isinstance(att_col_datatype, int)
        and not isinstance(att_col_datatype, bool)
        and att_col_datatype in _LEGACY_OGR_INT_TYPES
    ):
        return [None if v is None else int(v) for v in att_col_data]

    if att_col_datatype in (float, numpy.floating) or (
        isinstance(att_col_datatype, int)
        and not isinstance(att_col_datatype, bool)
        and att_col_datatype in _LEGACY_OGR_FLOAT_TYPES
    ):
        return [None if v is None else float(v) for v in att_col_data]

    if att_col_datatype is str or (
        isinstance(att_col_datatype, int)
        and not isinstance(att_col_datatype, bool)
        and att_col_datatype in _LEGACY_OGR_STR_TYPES
    ):
        return ["" if v is None else str(v) for v in att_col_data]

    if att_col_datatype is bool:
        return [bool(v) for v in att_col_data]

    if isinstance(att_col_datatype, str):
        key = att_col_datatype.strip().lower()
        if key in ("int", "int32", "int64", "integer", "oftinteger", "oftinteger64"):
            return [None if v is None else int(v) for v in att_col_data]
        if key in ("float", "float32", "float64", "real", "double", "oftreal"):
            return [None if v is None else float(v) for v in att_col_data]
        if key in ("str", "string", "text", "oftstring"):
            return ["" if v is None else str(v) for v in att_col_data]
        if key in ("bool", "boolean"):
            return [bool(v) for v in att_col_data]

    if isinstance(att_col_datatype, numpy.dtype):
        if numpy.issubdtype(att_col_datatype, numpy.integer):
            return [None if v is None else int(v) for v in att_col_data]
        if numpy.issubdtype(att_col_datatype, numpy.floating):
            return [None if v is None else float(v) for v in att_col_data]
        if numpy.issubdtype(att_col_datatype, numpy.bool_):
            return [bool(v) for v in att_col_data]
        if numpy.issubdtype(att_col_datatype, numpy.str_):
            return ["" if v is None else str(v) for v in att_col_data]

    return list(att_col_data)


def to_python_scalar(val: Any) -> Any:
    """
    Convert NumPy/pandas scalars to native Python scalars for JSON/dict output.

    :param val: Input scalar value.
    :return: Native Python scalar when ``val`` provides ``.item()``.

    .. code:: python

        import numpy as np
        import rsgispy.vectorprocess

        py_val = rsgispy.vectorprocess.to_python_scalar(np.int64(5))
    """
    if hasattr(val, "item"):
        return val.item()
    return val


def build_atts_dataframe(
    n_items: int,
    atts: Optional[Dict[str, Sequence[Any]]],
    att_types: Optional[Dict[str, Sequence[Any]]],
    length_label: str = "geometries",
) -> Dict[str, List[Any]]:
    """
    Validate and cast ``atts`` / ``att_types`` dictionaries into attribute
    columns matching ``n_items``.

    :param n_items: Expected number of features.
    :param atts: Mapping of column names to sequences of values.
    :param att_types: Dictionary with ``'names'`` and ``'types'`` sequences.
    :param length_label: Human-readable entity name used in error messages.
    :return: Dictionary mapping column name to cast list of values.

    .. code:: python

        import rsgispy.vectorprocess

        cols = rsgispy.vectorprocess.build_atts_dataframe(
            2, {"id": [1, 2]}, {"names": ["id"], "types": [int]}
        )
    """
    if (atts is None) and (att_types is None):
        return {}
    if (atts is None) or (att_types is None):
        raise rsgispy.RSGISPyException(
            "If atts or att_types is provided then both must be supplied."
        )
    if "names" not in att_types:
        raise rsgispy.RSGISPyException(
            'att_types must include a list for "names".'
        )
    if "types" not in att_types:
        raise rsgispy.RSGISPyException(
            'att_types must include a list for "types".'
        )

    names = att_types["names"]
    types = att_types["types"]
    if len(names) != len(types):
        raise rsgispy.RSGISPyException(
            'att_types "names" and "types" lists must be the same length.'
        )

    out_cols: Dict[str, List[Any]] = {}
    for col_name, col_type in zip(names, types):
        if col_name not in atts:
            raise rsgispy.RSGISPyException(
                f'"{col_name}" is not within atts.'
            )
        col_vals = atts[col_name]
        if len(col_vals) != n_items:
            raise rsgispy.RSGISPyException(
                f'"{col_name}" in atts does not have the same length as '
                f"{length_label}."
            )
        out_cols[col_name] = cast_col_data(col_vals, col_type)

    return out_cols


def pt_xy(
    pt: Union[shapely.geometry.Point, Sequence[float]],
) -> Tuple[float, float]:
    """
    Extract ``(x, y)`` float coordinates from a Shapely ``Point`` or sequence.

    :param pt: Shapely Point or 2-element sequence.
    :return: Coordinate tuple ``(x, y)``.

    .. code:: python

        from shapely.geometry import Point
        import rsgispy.vectorprocess

        xy = rsgispy.vectorprocess.pt_xy(Point(1.0, 2.0))
    """
    if hasattr(pt, "x") and hasattr(pt, "y"):
        return float(pt.x), float(pt.y)
    return float(pt[0]), float(pt[1])


def extract_geom_pts(
    geom: Optional[BaseGeometry],
    pts_lst: Optional[List[Tuple[float, float]]] = None,
) -> List[Tuple[float, float]]:
    """
    Recursively extract ``(x, y)`` vertex coordinates from a Shapely geometry.

    :param geom: Input Shapely geometry.
    :param pts_lst: Optional destination list to append coordinates into.
    :return: List of ``(x, y)`` tuples.

    .. code:: python

        from shapely.geometry import box
        import rsgispy.vectorprocess

        pts = rsgispy.vectorprocess.extract_geom_pts(box(0, 0, 10, 10))
    """
    if pts_lst is None:
        pts_lst = []

    if geom is None or geom.is_empty:
        return pts_lst

    geom_type = geom.geom_type
    if geom_type == "Point":
        pts_lst.append((float(geom.x), float(geom.y)))
    elif geom_type in ("LineString", "LinearRing"):
        for coord in geom.coords:
            pts_lst.append((float(coord[0]), float(coord[1])))
    elif geom_type == "Polygon":
        for coord in geom.exterior.coords:
            pts_lst.append((float(coord[0]), float(coord[1])))
        for interior in geom.interiors:
            for coord in interior.coords:
                pts_lst.append((float(coord[0]), float(coord[1])))
    elif hasattr(geom, "geoms"):
        for sub_geom in geom.geoms:
            extract_geom_pts(sub_geom, pts_lst)
    else:
        raise rsgispy.RSGISPyException(
            f"Unsupported geometry type '{geom_type}' when extracting points."
        )

    return pts_lst
