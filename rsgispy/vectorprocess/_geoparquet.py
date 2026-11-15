"""
Internal GeoParquet metadata parsing helpers for :mod:`rsgispy.vectorprocess`.
"""

import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyproj

try:
    import pyarrow.parquet as pq

    HAVE_PYARROW = True
except ImportError:
    pq = None
    HAVE_PYARROW = False

import rsgispy
from rsgispy.imageprocess import ensure_file_exists
import rsgispy.tools.files


def geoparquet_geom_type_name(geom_types: Sequence[str]) -> str:
    """
    Normalise a GeoParquet ``geometry_types`` list into an OGR-compatible
    geometry type name.
    """
    if not geom_types:
        return "Unknown"
    if len(geom_types) == 1:
        return str(geom_types[0])
    gset = set(geom_types)
    if gset.issubset({"Point", "MultiPoint"}):
        return "MultiPoint"
    if gset.issubset({"LineString", "MultiLineString"}):
        return "MultiLineString"
    if gset.issubset({"Polygon", "MultiPolygon"}):
        return "MultiPolygon"
    return "GeometryCollection"


def _extract_geoparquet_crs(col_meta: Dict[str, Any]) -> Optional[str]:
    """Extract an OGR-compatible CRS string from GeoParquet column metadata."""
    if "crs" not in col_meta:
        return "OGC:CRS84"
    raw_crs = col_meta["crs"]
    if raw_crs is None:
        return None
    crs_obj = pyproj.CRS.from_user_input(raw_crs)
    epsg = crs_obj.to_epsg()
    return f"EPSG:{epsg}" if epsg is not None else crs_obj.to_wkt()


def _extract_geoparquet_bounds(
    col_meta: Dict[str, Any],
) -> Optional[Tuple[float, float, float, float]]:
    """Extract 2D ``(xmin, ymin, xmax, ymax)`` bounds from GeoParquet metadata."""
    raw_bbox = col_meta.get("bbox")
    if not isinstance(raw_bbox, (list, tuple)):
        return None
    if len(raw_bbox) == 4:
        return (
            float(raw_bbox[0]),
            float(raw_bbox[1]),
            float(raw_bbox[2]),
            float(raw_bbox[3]),
        )
    if len(raw_bbox) >= 6:
        return (
            float(raw_bbox[0]),
            float(raw_bbox[1]),
            float(raw_bbox[3]),
            float(raw_bbox[4]),
        )
    return None


def _extract_pandas_index_cols(meta: Dict[bytes, bytes]) -> set:
    """Return index column names stored in pandas Parquet metadata."""
    index_cols = set()
    if b"pandas" in meta:
        try:
            pandas_meta = json.loads(meta[b"pandas"].decode("utf-8"))
            for item in pandas_meta.get("index_columns", []):
                if isinstance(item, str):
                    index_cols.add(item)
        except (ValueError, KeyError, TypeError):
            pass
    return index_cols


def read_geoparquet_info(
    vec_file: str,
    vec_lyr: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Read metadata from a GeoParquet file using ``pyarrow.parquet`` and return a
    dictionary matching ``pyogrio.read_info``.
    """
    ensure_file_exists(vec_file, label="Vector file")
    lyr_name = rsgispy.tools.files.get_file_basename(vec_file, rm_n_exts=0)
    if vec_lyr is not None and vec_lyr != lyr_name:
        raise rsgispy.RSGISPyException(
            f"Could not find vector layer '{vec_lyr}' in '{vec_file}'."
        )

    if not HAVE_PYARROW or pq is None:
        raise rsgispy.RSGISPyException(
            "Reading GeoParquet metadata requires 'pyarrow'."
        )

    try:
        schema = pq.read_schema(vec_file)
        pq_meta = pq.read_metadata(vec_file)
    except Exception as err:
        raise rsgispy.RSGISPyException(
            f"Could not open GeoParquet file '{vec_file}': {err}"
        ) from err

    meta = schema.metadata or {}
    if b"geo" not in meta:
        raise rsgispy.RSGISPyException(
            f"Parquet file '{vec_file}' does not contain GeoParquet ('geo') metadata."
        )

    try:
        geo_meta = json.loads(meta[b"geo"].decode("utf-8"))
    except Exception as err:
        raise rsgispy.RSGISPyException(
            f"Could not parse GeoParquet metadata in '{vec_file}': {err}"
        ) from err

    geom_name = str(geo_meta.get("primary_column", "geometry"))
    col_meta = geo_meta.get("columns", {}).get(geom_name, {})
    index_cols = _extract_pandas_index_cols(meta)
    skip_cols = {geom_name} | index_cols
    covering_bbox = col_meta.get("covering", {}).get("bbox", {})
    if isinstance(covering_bbox, dict):
        for cov_path in covering_bbox.values():
            if isinstance(cov_path, (list, tuple)) and cov_path:
                skip_cols.add(str(cov_path[0]))

    fields: List[str] = []
    dtypes: List[str] = []
    for field in schema:
        if field.name in skip_cols:
            continue
        fields.append(field.name)
        dtypes.append(str(field.type))

    geom_type = geoparquet_geom_type_name(col_meta.get("geometry_types", []))
    total_bounds = _extract_geoparquet_bounds(col_meta)

    return {
        "layer_name": lyr_name,
        "crs": _extract_geoparquet_crs(col_meta),
        "fields": np.array(fields, dtype=object),
        "dtypes": np.array(dtypes, dtype=object),
        "geometry_name": geom_name,
        "geometry_type": geom_type,
        "features": int(pq_meta.num_rows),
        "total_bounds": total_bounds,
        "driver": "Parquet",
        "capabilities": {
            "random_read": False,
            "fast_set_next_by_index": False,
            "fast_spatial_filter": True,
            "fast_feature_count": True,
            "fast_total_bounds": total_bounds is not None,
        },
    }
