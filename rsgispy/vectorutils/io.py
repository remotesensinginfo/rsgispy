"""
Vector copying, translation, reprojection, merging, and splitting utilities for
:mod:`rsgispy.vectorutils` using ``geopandas``, ``pyogrio``, and ``shapely``.
"""

import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import xml.etree.ElementTree as ET

import geopandas
import pandas
from shapely.geometry import (
    LinearRing,
    LineString,
    MultiLineString,
    Point,
    Polygon,
)

import rsgispy
from rsgispy.imageprocess import ensure_file_exists
import rsgispy.tools.files
from rsgispy.tools.geometry import calc_bbox_area
import rsgispy.tools.strings
import rsgispy.tools.utm
import rsgispy.vectorgeoms.spatial
from rsgispy.vectorprocess import (
    CRSLike,
    VectorInput,
    bbox_to_rsgis,
    check_format_name,
    extract_geom_pts,
    finish_vec_output,
    is_geoparquet_file,
    is_geoparquet_format,
    parse_crs_input,
    prepare_optional_out_vec,
    prepare_out_vec_file,
    read_vec_gdf,
    resolve_out_lyr_name,
    resolve_vec_layer,
    validate_columns_exist,
    write_vec_gdf,
)
from rsgispy.vectorutils.metadata import (
    VecLayersInfoObj,
    get_epsg_proj_from_vec,
    get_vec_lyrs_lst,
)

__all__ = [
    "create_copy_vec_lyr",
    "vector_translate",
    "reproj_vec_lyr",
    "reproj_wgs84_vec_to_utm",
    "redefine_vec_lyr_proj",
    "merge_vec_files",
    "merge_vec_lyrs",
    "merge_vec_file_lyrs",
    "merge_to_multi_lyr_vec",
    "check_polys_wgs84_bounds_geometry",
    "merge_utm_vecs_wgs84",
    "split_vec_lyr",
    "split_feats_to_mlyrs",
    "split_by_attribute",
    "convert_gpx_to_vec",
]


def _concat_gdfs(gdfs: Sequence[geopandas.GeoDataFrame]) -> geopandas.GeoDataFrame:
    """Concatenate non-empty GeoDataFrames while preserving CRS."""
    if len(gdfs) == 1:
        return gdfs[0].copy()
    crs = gdfs[0].crs
    merged = pandas.concat(list(gdfs), ignore_index=True)
    if not isinstance(merged, geopandas.GeoDataFrame):
        merged = geopandas.GeoDataFrame(merged, geometry="geometry", crs=crs)
    elif merged.crs is None and crs is not None:
        merged = merged.set_crs(crs, allow_override=True)
    return merged


def create_copy_vec_lyr(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    out_format: str = "GPKG",
    replace: bool = False,
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[geopandas.GeoDataFrame]:
    """
    Create a copy of an input vector layer.

    :param vec_file: Input vector file path or :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Input vector layer name. If ``None``, reads the first layer.
    :param out_vec_file: Output vector file path, or ``None`` to return an
                         in-memory copy as a :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format (Default = ``'GPKG'``).
    :param replace: If ``True``, delete ``out_vec_file`` before writing
                    (Default = ``False``).
    :param out_compress: Optional GeoParquet compression setting (``None`` to
                         infer from extension or write uncompressed, ``True``
                         for ``'snappy'``, ``False``/``'none'`` for uncompressed,
                         or codec name ``'snappy'``, ``'gzip'``, ``'zstd'``,
                         ``'brotli'``, ``'lz4'``).
    :return: Copied :class:`geopandas.GeoDataFrame` when ``out_vec_file`` is
             ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.vectorutils.io

        rsgispy.vectorutils.io.create_copy_vec_lyr(
            "in.gpkg", "polys", "copy.gpkg", "polys_copy", replace=True
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    lyr_name = resolve_vec_layer(vec_file, vec_lyr)
    gdf = read_vec_gdf(vec_file, vec_lyr=lyr_name, copy=True)
    out_lyr = resolve_out_lyr_name(out_vec_file, out_vec_lyr or lyr_name)
    return finish_vec_output(
        gdf,
        out_vec_file=out_vec_file,
        out_vec_lyr=out_lyr,
        out_format=out_format,
        overwrite=replace,
        out_compress=out_compress,
    )


def vector_translate(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    out_format: str = "GPKG",
    access_mode: Optional[str] = None,
    src_srs: Optional[CRSLike] = None,
    dst_srs: Optional[CRSLike] = None,
    del_exist_vec: bool = False,
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[geopandas.GeoDataFrame]:
    """
    Translate a vector layer between formats and optionally assign or reproject
    its coordinate reference system.

    :param vec_file: Input vector file path or :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Input vector layer name. If ``None``, reads the first layer.
    :param out_vec_file: Output vector file path, or ``None`` to return an
                         in-memory :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name (defaults to input layer name).
    :param out_format: Output vector driver name (Default = ``'GPKG'``).
    :param access_mode: Optional write mode: ``None`` (create), ``'update'``,
                        ``'append'``, or ``'overwrite'``.
    :param src_srs: Optional source CRS override (EPSG int, WKT/PROJ string,
                    or ``pyproj.CRS``) applied without reprojecting.
    :param dst_srs: Optional target CRS (EPSG int, WKT/PROJ string, or
                    ``pyproj.CRS``) to reproject the layer to.
    :param del_exist_vec: If ``True``, delete ``out_vec_file`` if it exists
                          (Default = ``False``).
    :param out_compress: Optional GeoParquet compression setting.
    :return: Translated :class:`geopandas.GeoDataFrame` when ``out_vec_file``
             is ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.vectorutils.io

        rsgispy.vectorutils.io.vector_translate(
            "in.gpkg", "pts", "out.geojson", out_format="GeoJSON", dst_srs=4326
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    if access_mode is not None and access_mode not in (
        "update",
        "append",
        "overwrite",
    ):
        raise rsgispy.RSGISPyException(
            "access_mode must be one of: [None, 'update', 'append', 'overwrite']"
        )

    if out_vec_file is None and access_mode == "append":
        raise rsgispy.RSGISPyException(
            "out_vec_file must be provided when access_mode='append'."
        )

    lyr_name = resolve_vec_layer(vec_file, vec_lyr)
    gdf = read_vec_gdf(vec_file, vec_lyr=lyr_name)

    if src_srs is not None:
        gdf = gdf.set_crs(parse_crs_input(src_srs), allow_override=True)
    if dst_srs is not None:
        gdf = gdf.to_crs(parse_crs_input(dst_srs))

    if out_vec_file is None:
        return gdf

    out_path = os.fspath(out_vec_file)
    if del_exist_vec or access_mode == "overwrite":
        prepare_out_vec_file(out_path, del_exist_vec=True)

    fmt = check_format_name(out_format)
    out_lyr = resolve_out_lyr_name(out_path, out_vec_lyr or lyr_name)
    is_pq = is_geoparquet_format(fmt) or is_geoparquet_file(out_path)
    is_gpkg = fmt == "GPKG" or out_path.lower().endswith(".gpkg")

    if access_mode == "append" and os.path.exists(out_path):
        if is_pq:
            existing_gdf = read_vec_gdf(out_path)
            merged_gdf = _concat_gdfs([existing_gdf, gdf])
            write_vec_gdf(
                merged_gdf,
                out_vec_file=out_path,
                out_vec_lyr=out_lyr,
                out_format=fmt,
                overwrite=True,
                out_compress=out_compress,
            )
            return None
        if is_gpkg:
            existing_lyrs = get_vec_lyrs_lst(out_path)
            if out_lyr in existing_lyrs:
                gdf.to_file(
                    out_path, layer=out_lyr, driver=fmt, mode="a"
                )
                return None
        else:
            gdf.to_file(out_path, driver=fmt, mode="a")
            return None

    overwrite_file = del_exist_vec or (access_mode == "overwrite")
    write_vec_gdf(
        gdf,
        out_vec_file=out_path,
        out_vec_lyr=out_lyr,
        out_format=fmt,
        overwrite=overwrite_file,
        out_compress=out_compress,
    )
    return None


def reproj_vec_lyr(
    vec_file: VectorInput,
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    epsg_code: Optional[int] = None,
    wkt_str: Optional[str] = None,
    out_format: str = "GPKG",
    vec_lyr: Optional[str] = None,
    in_epsg_code: Optional[int] = None,
    in_wkt_str: Optional[str] = None,
    del_exist_vec: bool = False,
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[geopandas.GeoDataFrame]:
    """
    Reproject a vector layer to a target EPSG code or WKT CRS string using
    ``geopandas``.

    :param vec_file: Input vector file path or :class:`geopandas.GeoDataFrame`.
    :param out_vec_file: Output vector file path, or ``None`` to return an
                         in-memory :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name.
    :param epsg_code: Target EPSG integer code (mutually optional with ``wkt_str``).
    :param wkt_str: Target projection WKT string (used if ``epsg_code`` is ``None``).
    :param out_format: Output vector format (Default = ``'GPKG'``).
    :param vec_lyr: Input vector layer name. If ``None``, reads the first layer.
    :param in_epsg_code: Optional input EPSG override applied before reprojecting.
    :param in_wkt_str: Optional input WKT CRS override applied before reprojecting.
    :param del_exist_vec: If ``True``, delete existing output file first
                          (Default = ``False``).
    :param out_compress: Optional GeoParquet compression setting.
    :return: Reprojected :class:`geopandas.GeoDataFrame` when ``out_vec_file``
             is ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.vectorutils.io

        rsgispy.vectorutils.io.reproj_vec_lyr(
            "polys_wgs84.gpkg",
            "polys_bng.gpkg",
            out_vec_lyr="polys",
            epsg_code=27700,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    if epsg_code is None and wkt_str is None:
        raise rsgispy.RSGISPyException(
            "An EPSG code or WKT string must be provided for reprojection."
        )

    out_crs = parse_crs_input(
        int(epsg_code) if epsg_code is not None else str(wkt_str)
    )
    lyr_name = resolve_vec_layer(vec_file, vec_lyr)
    gdf = read_vec_gdf(vec_file, vec_lyr=lyr_name)

    if in_epsg_code is not None:
        gdf = gdf.set_crs(epsg=int(in_epsg_code), allow_override=True)
    elif in_wkt_str is not None:
        gdf = gdf.set_crs(parse_crs_input(in_wkt_str), allow_override=True)
    elif gdf.crs is None:
        raise rsgispy.RSGISPyException(
            f"Input layer '{lyr_name}' has no CRS defined; "
            "provide in_epsg_code or in_wkt_str."
        )

    out_path = prepare_optional_out_vec(
        out_vec_file, del_exist_vec=del_exist_vec
    )
    reproj_gdf = gdf.to_crs(out_crs)
    return finish_vec_output(
        reproj_gdf,
        out_vec_file=out_path,
        out_vec_lyr=resolve_out_lyr_name(out_path, out_vec_lyr or lyr_name),
        out_format=out_format,
        out_compress=out_compress,
    )


def reproj_wgs84_vec_to_utm(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    use_hemi: bool = False,
    out_format: str = "GPKG",
    del_exist_vec: bool = False,
    out_compress: Optional[Union[str, bool]] = None,
) -> Union[int, geopandas.GeoDataFrame]:
    """
    Reproject a WGS84 (``EPSG:4326``) vector layer to the UTM zone containing its
    mean feature vertex coordinate.

    :param vec_file: Input WGS84 vector file path or
                     :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Input vector layer name. If ``None``, reads the first layer.
    :param out_vec_file: Output vector file path, or ``None`` to return an
                         in-memory :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name.
    :param use_hemi: If ``True``, select Southern hemisphere EPSG when mean
                     latitude < 0; otherwise always use Northern hemisphere
                     (Default = ``False``).
    :param out_format: Output vector format (Default = ``'GPKG'``).
    :param del_exist_vec: If ``True``, delete existing output file first
                          (Default = ``False``).
    :param out_compress: Optional GeoParquet compression setting.
    :return: Reprojected :class:`geopandas.GeoDataFrame` when ``out_vec_file``
             is ``None``; otherwise the target UTM EPSG integer code.

    .. code:: python

        import rsgispy.vectorutils.io

        utm_epsg = rsgispy.vectorutils.io.reproj_wgs84_vec_to_utm(
            "sites_wgs84.gpkg", "sites", "sites_utm.gpkg", "sites", use_hemi=True
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    lyr_name = resolve_vec_layer(vec_file, vec_lyr)
    gdf = read_vec_gdf(vec_file, vec_lyr=lyr_name, epsg=4326, copy=False)

    pts: List[tuple] = []
    for geom in gdf.geometry:
        extract_geom_pts(geom, pts)
    if not pts:
        raise rsgispy.RSGISPyException(
            f"Vector layer '{lyr_name}' contains no valid geometries."
        )

    lon = sum(pt[0] for pt in pts) / len(pts)
    lat = sum(pt[1] for pt in pts) / len(pts)

    utm_zone = rsgispy.tools.utm.latlon_to_zone_number(lat, lon)
    hemi = "S" if (use_hemi and lat < 0) else "N"
    out_epsg = int(rsgispy.tools.utm.epsg_for_utm(utm_zone, hemi))

    reproj_res = reproj_vec_lyr(
        vec_file=gdf,
        out_vec_file=out_vec_file,
        out_vec_lyr=out_vec_lyr or lyr_name,
        epsg_code=out_epsg,
        out_format=out_format,
        vec_lyr=lyr_name,
        in_epsg_code=4326,
        del_exist_vec=del_exist_vec,
        out_compress=out_compress,
    )
    if out_vec_file is None:
        assert reproj_res is not None
        return reproj_res
    return out_epsg


def redefine_vec_lyr_proj(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    epsg_code: int,
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    out_format: str = "GPKG",
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[geopandas.GeoDataFrame]:
    """
    Redefine the CRS metadata of a vector layer in place of reprojecting
    coordinates.

    :param vec_file: Input vector file path or :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Input vector layer name.
    :param epsg_code: EPSG integer code to assign without altering coordinates.
    :param out_vec_file: Output vector file path, or ``None`` to return an
                         in-memory :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format (Default = ``'GPKG'``).
    :param out_compress: Optional GeoParquet compression setting.
    :return: Resulting :class:`geopandas.GeoDataFrame` when ``out_vec_file`` is
             ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.vectorutils.io

        rsgispy.vectorutils.io.redefine_vec_lyr_proj(
            "raw.gpkg", "raw", 27700, "defined.gpkg", "defined"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    gdf = read_vec_gdf(vec_file, vec_lyr=vec_lyr, epsg=int(epsg_code))
    return finish_vec_output(
        gdf,
        out_vec_file=out_vec_file,
        out_vec_lyr=resolve_out_lyr_name(out_vec_file, out_vec_lyr),
        out_format=out_format,
        out_compress=out_compress,
    )


def merge_vec_files(
    vec_files: Sequence[VectorInput],
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    out_format: str = "GPKG",
    out_epsg: Optional[int] = None,
    remove_cols: Optional[Sequence[str]] = None,
    use_pbar: bool = True,
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[geopandas.GeoDataFrame]:
    """
    Merge all layers from a sequence of vector files or
    :class:`geopandas.GeoDataFrame` objects into a single output vector layer
    using ``geopandas``.

    :param vec_files: Sequence of input vector file paths or
                      :class:`geopandas.GeoDataFrame` objects.
    :param out_vec_file: Output vector file path, or ``None`` to return an
                         in-memory :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format (Default = ``'GPKG'``).
    :param out_epsg: Optional target EPSG code to unify differing projections.
    :param remove_cols: Optional column names to drop during merging.
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).
    :param out_compress: Optional GeoParquet compression setting.
    :return: Merged :class:`geopandas.GeoDataFrame` when ``out_vec_file`` is
             ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.vectorutils.io

        rsgispy.vectorutils.io.merge_vec_files(
            ["tile1.gpkg", "tile2.gpkg"], "merged.gpkg", out_vec_lyr="merged"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    if not vec_files:
        raise rsgispy.RSGISPyException(
            "At least 1 vector file needs to be provided."
        )

    pbar_cls = rsgispy.get_tqdm()
    gp_lyrs: List[geopandas.GeoDataFrame] = []
    rm_set = set(remove_cols) if remove_cols is not None else set()

    for vec_file in pbar_cls(vec_files, disable=not use_pbar):
        lyrs: Sequence[Optional[str]] = (
            [None]
            if isinstance(vec_file, geopandas.GeoDataFrame)
            else get_vec_lyrs_lst(vec_file)
        )
        for lyr in lyrs:
            data_gdf = read_vec_gdf(vec_file, vec_lyr=lyr)
            if out_epsg is not None:
                data_gdf = data_gdf.to_crs(epsg=int(out_epsg))
            if rm_set:
                drop_cols = [c for c in data_gdf.columns if c in rm_set]
                if drop_cols:
                    data_gdf = data_gdf.drop(columns=drop_cols)
            if len(data_gdf) > 0:
                gp_lyrs.append(data_gdf)

    if not gp_lyrs:
        raise rsgispy.RSGISPyException("No layers with data were provided.")

    merged_gdf = _concat_gdfs(gp_lyrs)
    return finish_vec_output(
        merged_gdf,
        out_vec_file=out_vec_file,
        out_vec_lyr=resolve_out_lyr_name(out_vec_file, out_vec_lyr),
        out_format=out_format,
        out_compress=out_compress,
    )


def merge_vec_lyrs(
    vecs_dict: Sequence[Dict[str, Any]],
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    out_format: str = "GPKG",
    out_epsg: Optional[int] = None,
    remove_cols: Optional[Sequence[str]] = None,
    use_pbar: bool = True,
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[geopandas.GeoDataFrame]:
    """
    Merge specified ``{'file': ..., 'layer': ...}`` entries (where ``'file'``
    may be a path or :class:`geopandas.GeoDataFrame`) into a single output
    vector layer using ``geopandas``.

    :param vecs_dict: Sequence of dicts with key ``'file'`` and (for file paths)
                      ``'layer'``.
    :param out_vec_file: Output vector file path, or ``None`` to return an
                         in-memory :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format (Default = ``'GPKG'``).
    :param out_epsg: Optional target EPSG code to unify differing projections.
    :param remove_cols: Optional column names to drop during merging.
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).
    :param out_compress: Optional GeoParquet compression setting.
    :return: Merged :class:`geopandas.GeoDataFrame` when ``out_vec_file`` is
             ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.vectorutils.io

        rsgispy.vectorutils.io.merge_vec_lyrs(
            [{"file": "a.gpkg", "layer": "l1"}, {"file": "a.gpkg", "layer": "l2"}],
            "merged.gpkg",
            out_vec_lyr="all_feats",
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    if not vecs_dict:
        raise rsgispy.RSGISPyException(
            "At least 1 vector layer needs to be provided."
        )

    pbar_cls = rsgispy.get_tqdm()
    gp_lyrs: List[geopandas.GeoDataFrame] = []
    rm_set = set(remove_cols) if remove_cols is not None else set()

    for vec_info in pbar_cls(vecs_dict, disable=not use_pbar):
        if "file" not in vec_info:
            raise rsgispy.RSGISPyException(
                "Each entry must be a dict containing key 'file'."
            )
        src_file = vec_info["file"]
        if (
            not isinstance(src_file, geopandas.GeoDataFrame)
            and "layer" not in vec_info
        ):
            raise rsgispy.RSGISPyException(
                "Each entry must be a dict with keys 'file' and 'layer'."
            )
        data_gdf = read_vec_gdf(src_file, vec_lyr=vec_info.get("layer"))
        if out_epsg is not None:
            data_gdf = data_gdf.to_crs(epsg=int(out_epsg))
        if rm_set:
            drop_cols = [c for c in data_gdf.columns if c in rm_set]
            if drop_cols:
                data_gdf = data_gdf.drop(columns=drop_cols)
        if len(data_gdf) > 0:
            gp_lyrs.append(data_gdf)

    if not gp_lyrs:
        raise rsgispy.RSGISPyException("No layers with data were provided.")

    merged_gdf = _concat_gdfs(gp_lyrs)
    return finish_vec_output(
        merged_gdf,
        out_vec_file=out_vec_file,
        out_vec_lyr=resolve_out_lyr_name(out_vec_file, out_vec_lyr),
        out_format=out_format,
        out_compress=out_compress,
    )


def merge_vec_file_lyrs(
    vec_file: VectorInput,
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    out_format: str = "GPKG",
    out_epsg: Optional[int] = None,
    remove_cols: Optional[Sequence[str]] = None,
    use_pbar: bool = True,
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[geopandas.GeoDataFrame]:
    """
    Merge all layers inside a single multi-layer vector file into a single
    output layer.

    :param vec_file: Input multi-layer vector file path or
                     :class:`geopandas.GeoDataFrame`.
    :param out_vec_file: Output vector file path, or ``None`` to return an
                         in-memory :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format (Default = ``'GPKG'``).
    :param out_epsg: Optional target EPSG code.
    :param remove_cols: Optional column names to drop during merging.
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).
    :param out_compress: Optional GeoParquet compression setting.
    :return: Merged :class:`geopandas.GeoDataFrame` when ``out_vec_file`` is
             ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.vectorutils.io

        rsgispy.vectorutils.io.merge_vec_file_lyrs(
            "multi.gpkg", "single.gpkg", out_vec_lyr="combined"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    return merge_vec_files(
        vec_files=[vec_file],
        out_vec_file=out_vec_file,
        out_vec_lyr=out_vec_lyr,
        out_format=out_format,
        out_epsg=out_epsg,
        remove_cols=remove_cols,
        use_pbar=use_pbar,
        out_compress=out_compress,
    )


def merge_to_multi_lyr_vec(
    input_file_lyrs: Sequence[Union[VecLayersInfoObj, VectorInput]],
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_format: str = "GPKG",
    overwrite: bool = True,
    rename_dup_lyrs: bool = False,
    use_pbar: bool = True,
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[Dict[str, geopandas.GeoDataFrame]]:
    """
    Combine multiple input vector files, :class:`geopandas.GeoDataFrame`
    objects, or :class:`VecLayersInfoObj` descriptors as separate layers in a
    multi-layer vector file (e.g., ``GPKG``), or return a dictionary of
    ``{layer_name: GeoDataFrame}`` when ``out_vec_file`` is ``None``.

    :param input_file_lyrs: Sequence of :class:`VecLayersInfoObj` items, input
                            file path strings, or
                            :class:`geopandas.GeoDataFrame` objects.
    :param out_vec_file: Output multi-layer vector file path, or ``None`` to
                         return a mapping ``{layer_name: GeoDataFrame}``.
    :param out_format: Output vector format (Default = ``'GPKG'``).
    :param overwrite: If ``True``, delete ``out_vec_file`` if it exists before
                      writing (Default = ``True``).
    :param rename_dup_lyrs: If ``True``, append a unique suffix when two inputs
                            share the same output layer name; if ``False``,
                            raise :class:`rsgispy.RSGISPyException`
                            (Default = ``False``).
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).
    :param out_compress: Optional GeoParquet compression setting (when writing a
                         single-layer GeoParquet file).
    :return: Dictionary mapping output layer names to
             :class:`geopandas.GeoDataFrame` objects when ``out_vec_file`` is
             ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.vectorutils.io
        import rsgispy.vectorutils.metadata

        items = [
            rsgispy.vectorutils.metadata.VecLayersInfoObj("a.gpkg", "lyr1", "layer_a"),
            rsgispy.vectorutils.metadata.VecLayersInfoObj("b.gpkg", "lyr1", "layer_b"),
        ]
        rsgispy.vectorutils.io.merge_to_multi_lyr_vec(items, "multi.gpkg")
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    seen_out_lyrs = set()
    out_path: Optional[str] = None
    if out_vec_file is not None:
        out_path = os.fspath(out_vec_file)
        if not rsgispy.tools.files.does_path_exists_or_creatable(out_path):
            raise rsgispy.RSGISPyException(
                f"Output file path is not creatable: {out_path}"
            )
        if overwrite and os.path.exists(out_path):
            prepare_out_vec_file(out_path, del_exist_vec=True)
        if os.path.exists(out_path) and not overwrite:
            seen_out_lyrs.update(get_vec_lyrs_lst(out_path))

    entries: List[VecLayersInfoObj] = []
    for item in input_file_lyrs:
        if isinstance(item, VecLayersInfoObj):
            if item.vec_file is None:
                raise rsgispy.RSGISPyException(
                    "VecLayersInfoObj.vec_file must not be None."
                )
            src_lyr = resolve_vec_layer(item.vec_file, item.vec_lyr)
            entries.append(
                VecLayersInfoObj(
                    vec_file=item.vec_file,
                    vec_lyr=src_lyr,
                    vec_out_lyr=item.vec_out_lyr or src_lyr,
                )
            )
        elif isinstance(item, geopandas.GeoDataFrame):
            entries.append(
                VecLayersInfoObj(
                    vec_file=item,
                    vec_lyr="layer",
                    vec_out_lyr="layer",
                )
            )
        else:
            item_path = os.fspath(item)
            for lyr in get_vec_lyrs_lst(item_path):
                entries.append(
                    VecLayersInfoObj(
                        vec_file=item_path,
                        vec_lyr=lyr,
                        vec_out_lyr=lyr,
                    )
                )

    if out_path is not None:
        is_pq = is_geoparquet_format(out_format) or is_geoparquet_file(
            out_path
        )
        if is_pq and len(entries) > 1:
            raise rsgispy.RSGISPyException(
                "GeoParquet does not support multiple layers in a single file; "
                "use merge_vec_files or GPKG format."
            )

    out_layers: Dict[str, geopandas.GeoDataFrame] = {}
    pbar_cls = rsgispy.get_tqdm()
    for entry in pbar_cls(entries, disable=not use_pbar):
        assert entry.vec_file is not None
        out_lyr = entry.vec_out_lyr or entry.vec_lyr or "layer"
        if out_lyr in seen_out_lyrs:
            if rename_dup_lyrs:
                out_lyr = f"{out_lyr}_{rsgispy.tools.strings.uid_generator()}"
            else:
                raise rsgispy.RSGISPyException(
                    f"Duplicate output layer name '{out_lyr}' detected."
                )

        gdf = read_vec_gdf(entry.vec_file, vec_lyr=entry.vec_lyr)
        if len(gdf) > 0:
            if out_path is None:
                out_layers[out_lyr] = gdf
            else:
                write_vec_gdf(
                    gdf,
                    out_vec_file=out_path,
                    out_vec_lyr=out_lyr,
                    out_format=out_format,
                    out_compress=out_compress,
                )
            seen_out_lyrs.add(out_lyr)

    if out_path is None:
        return out_layers
    return None


def _unwrap_polygon_coords(
    poly: Polygon, east_focus: bool
) -> Polygon:
    """Shift coordinates across the +/-180 meridian on one Polygon."""
    out_coords = []
    for coord in poly.exterior.coords:
        x_val = float(coord[0])
        if east_focus and x_val < 0:
            x_val = 180.0 + (x_val + 180.0)
        elif (not east_focus) and x_val > 0:
            x_val = -180.0 - (180.0 - x_val)
        out_coords.append((x_val, float(coord[1])))

    out_holes = []
    for hole in poly.interiors:
        hole_coords = []
        for coord in hole.coords:
            x_val = float(coord[0])
            if east_focus and x_val < 0:
                x_val = 180.0 + (x_val + 180.0)
            elif (not east_focus) and x_val > 0:
                x_val = -180.0 - (180.0 - x_val)
            hole_coords.append((x_val, float(coord[1])))
        out_holes.append(LinearRing(hole_coords))

    return Polygon(out_coords, holes=out_holes)


def check_polys_wgs84_bounds_geometry(
    data_gdf: geopandas.GeoDataFrame,
    width_thres: float = 350.0,
    use_pbar: bool = True,
) -> geopandas.GeoDataFrame:
    """
    Unwrap WGS84 polygons whose longitude span exceeds ``width_thres`` degrees
    (indicating crossing of the 180/-180 antimeridian).

    :param data_gdf: Input WGS84 GeoDataFrame of Polygon/MultiPolygon geometries.
    :param width_thres: Longitude width threshold in degrees above which polygons
                        are unwrapped (Default = ``350.0``).
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).
    :return: GeoDataFrame with unwrapped polygons.

    .. code:: python

        import geopandas
        import rsgispy.vectorutils.io

        gdf = geopandas.read_file("world_polys.gpkg")
        fixed = rsgispy.vectorutils.io.check_polys_wgs84_bounds_geometry(gdf)
    """
    out_gdf = data_gdf.copy(deep=True)
    pbar_cls = rsgispy.get_tqdm()
    for idx in pbar_cls(out_gdf.index, disable=not use_pbar):
        geom = out_gdf.at[idx, "geometry"]
        if geom is None or geom.is_empty:
            continue
        bounds = geom.bounds
        if (bounds[2] - bounds[0]) <= width_thres:
            continue

        pts = extract_geom_pts(geom)
        if not pts:
            continue
        pos_count = sum(1 for pt in pts if pt[0] > 0)
        east_focus = pos_count >= (len(pts) / 2.0)

        if geom.geom_type == "Polygon":
            out_gdf.at[idx, "geometry"] = _unwrap_polygon_coords(
                geom, east_focus
            )
        elif geom.geom_type == "MultiPolygon":
            fixed_parts = [
                _unwrap_polygon_coords(part, east_focus) for part in geom.geoms
            ]
            if len(fixed_parts) == 1:
                out_gdf.at[idx, "geometry"] = fixed_parts[0]
            else:
                from shapely.geometry import MultiPolygon

                out_gdf.at[idx, "geometry"] = MultiPolygon(fixed_parts)

    return out_gdf


def merge_utm_vecs_wgs84(
    in_vec_files: Sequence[VectorInput],
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    out_format: str = "GPKG",
    n_hemi_utm_file: Optional[VectorInput] = None,
    s_hemi_utm_file: Optional[VectorInput] = None,
    width_thres: float = 350.0,
    use_pbar: bool = True,
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[geopandas.GeoDataFrame]:
    """
    Merge input vector files or :class:`geopandas.GeoDataFrame` objects
    projected in UTM zones into a single WGS84 (``EPSG:4326``) vector layer,
    splitting features crossing UTM zone boundaries and unwrapping geometries
    spanning the 180/-180 antimeridian.

    :param in_vec_files: Sequence of input UTM vector file paths or
                         :class:`geopandas.GeoDataFrame` objects.
    :param out_vec_file: Output WGS84 vector file path, or ``None`` to return
                         an in-memory :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format (Default = ``'GPKG'``).
    :param n_hemi_utm_file: Path (or GeoDataFrame) for Northern hemisphere UTM
                            zone boundaries (layers ``'01'`` to ``'60'``).
    :param s_hemi_utm_file: Path (or GeoDataFrame) for Southern hemisphere UTM
                            zone boundaries (layers ``'01'`` to ``'60'``).
    :param width_thres: Longitude width threshold in degrees for unwrapping
                        antimeridian polygons (Default = ``350.0``).
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).
    :param out_compress: Optional GeoParquet compression setting.
    :return: Merged WGS84 :class:`geopandas.GeoDataFrame` when ``out_vec_file``
             is ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.vectorutils.io

        rsgispy.vectorutils.io.merge_utm_vecs_wgs84(
            ["z30n.gpkg", "z31n.gpkg"],
            "merged_wgs84.gpkg",
            out_vec_lyr="merged",
            n_hemi_utm_file="utm_north.gpkg",
            s_hemi_utm_file="utm_south.gpkg",
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    if n_hemi_utm_file is None or s_hemi_utm_file is None:
        raise rsgispy.RSGISPyException(
            "Both n_hemi_utm_file and s_hemi_utm_file must be provided."
        )
    if not isinstance(n_hemi_utm_file, geopandas.GeoDataFrame):
        ensure_file_exists(
            os.fspath(n_hemi_utm_file), label="Northern UTM zones file"
        )
    if not isinstance(s_hemi_utm_file, geopandas.GeoDataFrame):
        ensure_file_exists(
            os.fspath(s_hemi_utm_file), label="Southern UTM zones file"
        )

    pbar_cls = rsgispy.get_tqdm()
    out_gdfs: List[geopandas.GeoDataFrame] = []

    for vec_file in pbar_cls(in_vec_files, disable=not use_pbar):
        lyrs: Sequence[Optional[str]] = (
            [None]
            if isinstance(vec_file, geopandas.GeoDataFrame)
            else get_vec_lyrs_lst(vec_file)
        )
        for lyr in lyrs:
            data_gdf = read_vec_gdf(vec_file, vec_lyr=lyr)
            if len(data_gdf) == 0:
                continue
            bbox = bbox_to_rsgis(data_gdf.total_bounds)
            if calc_bbox_area(bbox) <= 0:
                continue

            vec_epsg = get_epsg_proj_from_vec(
                data_gdf if isinstance(vec_file, geopandas.GeoDataFrame) else vec_file,
                vec_lyr=lyr,
            )
            if vec_epsg is None:
                raise rsgispy.RSGISPyException(
                    "Could not determine EPSG code for input UTM layer."
                )
            zone, hemi = rsgispy.tools.utm.utm_from_epsg(int(vec_epsg))
            zone_str = rsgispy.tools.strings.zero_pad_num_str(
                zone, str_len=2, integerise=True
            )
            utm_zones_file = (
                s_hemi_utm_file if hemi.upper() == "S" else n_hemi_utm_file
            )

            contained = rsgispy.vectorgeoms.spatial.vec_within_vec(
                utm_zones_file, zone_str, data_gdf, None, use_pbar=False
            )
            if not contained:
                utm_gdf = read_vec_gdf(
                    utm_zones_file, vec_lyr=zone_str, copy=False
                )
                data_inter = geopandas.overlay(
                    data_gdf, utm_gdf, how="intersection"
                )
                data_diff = geopandas.overlay(
                    data_gdf, utm_gdf, how="difference"
                )
                parts = [df for df in (data_inter, data_diff) if len(df) > 0]
                if parts:
                    data_gdf = _concat_gdfs(parts).to_crs("EPSG:4326")
                else:
                    data_gdf = geopandas.GeoDataFrame(
                        columns=data_gdf.columns, crs="EPSG:4326"
                    )
            else:
                data_gdf = data_gdf.to_crs("EPSG:4326")

            if len(data_gdf) > 0:
                bounds_df = data_gdf.bounds
                widths = bounds_df["maxx"] - bounds_df["minx"]
                if float(widths.max()) > width_thres:
                    data_gdf = check_polys_wgs84_bounds_geometry(
                        data_gdf, width_thres=width_thres, use_pbar=False
                    )
                out_gdfs.append(data_gdf)

    if out_gdfs:
        merged = _concat_gdfs(out_gdfs)
        return finish_vec_output(
            merged,
            out_vec_file=out_vec_file,
            out_vec_lyr=resolve_out_lyr_name(out_vec_file, out_vec_lyr),
            out_format=out_format,
            out_compress=out_compress,
        )
    if out_vec_file is None:
        return geopandas.GeoDataFrame(geometry=[], crs="EPSG:4326")
    return None


def split_vec_lyr(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    n_feats: int,
    out_format: str = "GPKG",
    out_dir: Optional[Union[str, os.PathLike]] = None,
    out_vec_base: Optional[str] = None,
    out_vec_ext: Optional[str] = None,
    use_pbar: bool = True,
    out_compress: Optional[Union[str, bool]] = None,
) -> Union[List[str], List[geopandas.GeoDataFrame]]:
    """
    Split an input vector layer into sequential chunks each containing at most
    ``n_feats`` features, writing output files or returning a list of
    :class:`geopandas.GeoDataFrame` objects when ``out_dir`` is ``None``.

    :param vec_file: Input vector file path or :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Input vector layer name.
    :param n_feats: Maximum number of features per output chunk.
    :param out_format: Output vector driver name (Default = ``'GPKG'``).
    :param out_dir: Output directory path (created if absent), or ``None`` to
                    return a list of :class:`geopandas.GeoDataFrame` objects.
    :param out_vec_base: Output filename and layer name prefix (required when
                         ``out_dir`` is specified).
    :param out_vec_ext: Output file extension without leading dot (e.g.,
                        ``'gpkg'``; required when ``out_dir`` is specified).
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).
    :param out_compress: Optional GeoParquet compression setting.
    :return: List of created output vector file paths, or list of chunk
             :class:`geopandas.GeoDataFrame` objects when ``out_dir`` is
             ``None``.

    .. code:: python

        import rsgispy.vectorutils.io

        out_files = rsgispy.vectorutils.io.split_vec_lyr(
            "large.gpkg", "large", 1000, "GPKG", "splits", "chunk_", "gpkg"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    if n_feats < 1:
        raise rsgispy.RSGISPyException("n_feats must be >= 1.")
    if out_dir is not None:
        if out_vec_base is None or out_vec_ext is None:
            raise rsgispy.RSGISPyException(
                "out_vec_base and out_vec_ext must be provided when out_dir is specified."
            )
        os.makedirs(out_dir, exist_ok=True)

    gdf = read_vec_gdf(vec_file, vec_lyr=vec_lyr, copy=False)
    gdf = gdf[gdf.geometry.notna()]
    total_feats = len(gdf)
    n_chunks = math.ceil(total_feats / n_feats)

    pbar_cls = rsgispy.get_tqdm()
    out_files: List[str] = []
    out_gdfs: List[geopandas.GeoDataFrame] = []
    for chunk_idx in pbar_cls(range(n_chunks), disable=not use_pbar):
        start = chunk_idx * n_feats
        end = min((chunk_idx + 1) * n_feats, total_feats)
        chunk_gdf = gdf.iloc[start:end].copy()
        if out_dir is None:
            out_gdfs.append(chunk_gdf)
        else:
            out_lyr = f"{out_vec_base}{chunk_idx + 1}"
            out_file = os.path.join(
                os.fspath(out_dir), f"{out_lyr}.{out_vec_ext}"
            )
            write_vec_gdf(
                chunk_gdf,
                out_vec_file=out_file,
                out_vec_lyr=out_lyr,
                out_format=out_format,
                out_compress=out_compress,
            )
            out_files.append(out_file)
    if out_dir is None:
        return out_gdfs
    return out_files


def split_feats_to_mlyrs(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_format: str = "GPKG",
    use_pbar: bool = True,
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[Dict[str, geopandas.GeoDataFrame]]:
    """
    Split every feature in an input vector layer into its own named layer
    (``veclyr_0``, ``veclyr_1``, ...) inside ``out_vec_file`` (or return a
    ``{layer_name: GeoDataFrame}`` mapping when ``out_vec_file`` is ``None``).

    :param vec_file: Input vector file path or :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Input vector layer name.
    :param out_vec_file: Output multi-layer vector file path, or ``None`` to
                         return a dict of single-feature GeoDataFrames.
    :param out_format: Output vector format supporting multiple layers
                       (Default = ``'GPKG'``).
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).
    :param out_compress: Optional GeoParquet compression setting (when input
                         contains at most one feature).
    :return: Dictionary ``{layer_name: GeoDataFrame}`` when ``out_vec_file`` is
             ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.vectorutils.io

        rsgispy.vectorutils.io.split_feats_to_mlyrs(
            "polys.gpkg", "polys", "polys_multi.gpkg"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    gdf = read_vec_gdf(vec_file, vec_lyr=vec_lyr, copy=False)
    if out_vec_file is not None:
        is_pq = is_geoparquet_format(out_format) or is_geoparquet_file(
            out_vec_file
        )
        if is_pq and len(gdf) > 1:
            raise rsgispy.RSGISPyException(
                "GeoParquet does not support multiple layers in a single file; "
                "use split_vec_lyr instead."
            )
    pbar_cls = rsgispy.get_tqdm()
    out_layers: Dict[str, geopandas.GeoDataFrame] = {}
    for i in pbar_cls(range(len(gdf)), disable=not use_pbar):
        lyr_name = f"veclyr_{i}"
        row_gdf = gdf.iloc[[i]].copy()
        if out_vec_file is None:
            out_layers[lyr_name] = row_gdf
        else:
            write_vec_gdf(
                row_gdf,
                out_vec_file=out_vec_file,
                out_vec_lyr=lyr_name,
                out_format=out_format,
                out_compress=out_compress,
            )
    if out_vec_file is None:
        return out_layers
    return None


def split_by_attribute(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    split_col_name: str,
    multi_layers: bool = True,
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_file_path: Optional[Union[str, os.PathLike]] = None,
    out_file_ext: Optional[str] = None,
    out_format: str = "GPKG",
    dissolve: bool = False,
    chk_lyr_names: bool = True,
    use_pbar: bool = True,
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[Dict[str, geopandas.GeoDataFrame]]:
    """
    Split a vector layer by unique values in ``split_col_name`` into either
    multiple layers in a single GeoPackage, separate output files, or an
    in-memory ``{value_name: GeoDataFrame}`` dictionary when both
    ``out_vec_file`` and ``out_file_path`` are ``None``.

    :param vec_file: Input vector file path or :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Input vector layer name.
    :param split_col_name: Attribute column whose unique values define splits.
    :param multi_layers: If ``True`` (default), write all splits into layers in
                         ``out_vec_file`` (when writing to disk); if ``False``,
                         write separate files to ``out_file_path``.
    :param out_vec_file: Output multi-layer file path (or ``None`` for in-memory
                         return when ``out_file_path`` is also ``None``).
    :param out_file_path: Output directory path (required when
                          ``multi_layers=False`` for disk output).
    :param out_file_ext: Output file extension without dot (required when
                         ``multi_layers=False`` for disk output).
    :param out_format: Output vector format (Default = ``'GPKG'``).
    :param dissolve: If ``True``, dissolve features per group before returning
                     or writing (Default = ``False``).
    :param chk_lyr_names: If ``True``, strip non-ASCII characters and
                          punctuation from split layer/file/key names
                          (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).
    :param out_compress: Optional GeoParquet compression setting.
    :return: Dictionary mapping split names to :class:`geopandas.GeoDataFrame`
             objects when both ``out_vec_file`` and ``out_file_path`` are
             ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.vectorutils.io

        rsgispy.vectorutils.io.split_by_attribute(
            "regions.gpkg",
            "regions",
            "region_code",
            multi_layers=True,
            out_vec_file="regions_split.gpkg",
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    in_memory = (out_vec_file is None) and (out_file_path is None)
    fmt = check_format_name(out_format)
    if not in_memory:
        if multi_layers and out_vec_file is None:
            raise rsgispy.RSGISPyException(
                "out_vec_file must be specified when multi_layers=True."
            )
        if multi_layers and (
            is_geoparquet_format(fmt)
            or (out_vec_file is not None and is_geoparquet_file(out_vec_file))
        ):
            raise rsgispy.RSGISPyException(
                "GeoParquet does not support multiple layers in a single file; "
                "set multi_layers=False to write separate files."
            )
        if (not multi_layers) and (
            (out_file_path is None) or (out_file_ext is None)
        ):
            raise rsgispy.RSGISPyException(
                "out_file_path and out_file_ext must be specified when "
                "multi_layers=False."
            )

    base_gdf = read_vec_gdf(vec_file, vec_lyr=vec_lyr, copy=False)
    validate_columns_exist(base_gdf, [split_col_name])
    unq_vals = base_gdf[split_col_name].unique()

    pbar_cls = rsgispy.get_tqdm()
    out_splits: Dict[str, geopandas.GeoDataFrame] = {}
    for val in pbar_cls(unq_vals, disable=not use_pbar):
        sub_gdf = base_gdf.loc[base_gdf[split_col_name] == val]
        sub_gdf = sub_gdf[
            sub_gdf.geometry.notna() & (~sub_gdf.geometry.is_empty)
        ].copy()
        if len(sub_gdf) == 0:
            continue

        if dissolve:
            try:
                sub_gdf = sub_gdf.dissolve(by=split_col_name, as_index=False)
            except Exception:  # pylint: disable=broad-exception-caught
                sub_gdf = sub_gdf.copy()
                sub_gdf["geometry"] = sub_gdf.geometry.buffer(0)
                sub_gdf = sub_gdf.dissolve(by=split_col_name, as_index=False)

        val_str = f"{val}"
        if chk_lyr_names:
            val_str = rsgispy.tools.strings.check_str(
                val_str,
                rm_non_ascii=True,
                rm_dashs=True,
                rm_spaces=False,
                rm_punc=True,
            )

        if in_memory:
            out_splits[val_str] = sub_gdf
        elif multi_layers and fmt == "GPKG":
            assert out_vec_file is not None
            write_vec_gdf(
                sub_gdf,
                out_vec_file=out_vec_file,
                out_vec_lyr=val_str,
                out_format="GPKG",
                out_compress=out_compress,
            )
        else:
            assert out_file_path is not None
            assert out_file_ext is not None
            c_out_lyr = f"vec_{val_str}"
            c_out_file = os.path.join(
                os.fspath(out_file_path), f"{c_out_lyr}.{out_file_ext}"
            )
            write_vec_gdf(
                sub_gdf,
                out_vec_file=c_out_file,
                out_vec_lyr=c_out_lyr,
                out_format=fmt,
                out_compress=out_compress,
            )

    if in_memory:
        return out_splits
    return None


_GPX_FLOAT_TAGS = frozenset(
    {
        "ele",
        "magvar",
        "geoidheight",
        "hdop",
        "vdop",
        "pdop",
        "ageofdgpsdata",
        "speed",
        "course",
        "atemp",
        "wtemp",
        "depth",
    }
)
_GPX_INT_TAGS = frozenset({"sat", "dgpsid", "number", "hr", "cad", "power"})
_GPX_CORE_PT_COLS = frozenset(
    {
        "track_fid",
        "track_seg_id",
        "track_seg_point_id",
        "track_name",
        "geometry",
    }
)


def _xml_local_name(tag: str) -> str:
    """Return XML element local name without namespace URI."""
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _coerce_gpx_scalar(key: str, raw_val: str) -> Any:
    """Convert known numeric GPX tag text to int or float where valid."""
    if key in _GPX_INT_TAGS:
        try:
            return int(raw_val)
        except ValueError:
            try:
                return float(raw_val)
            except ValueError:
                return raw_val
    if key in _GPX_FLOAT_TAGS:
        try:
            return float(raw_val)
        except ValueError:
            return raw_val
    return raw_val


def _extract_gpx_extensions(ext_elem: ET.Element, row: Dict[str, Any]) -> None:
    """Populate leaf scalar values from a GPX ``<extensions>`` element."""
    for child in ext_elem.iter():
        if child is ext_elem or len(child) > 0:
            continue
        if child.text is None:
            continue
        text_val = child.text.strip()
        if not text_val:
            continue
        col_name = _xml_local_name(child.tag)
        if col_name not in _GPX_CORE_PT_COLS and col_name not in row:
            row[col_name] = _coerce_gpx_scalar(col_name, text_val)


def _parse_gpx_trkpt(
    pt_elem: ET.Element,
    track_fid: int,
    seg_id: int,
    pt_id: int,
    track_name: Optional[str],
    include_z: bool,
) -> Tuple[Dict[str, Any], Tuple[float, ...], Optional[str]]:
    """Parse one ``<trkpt>`` element into a point row, coordinate tuple, and time."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    try:
        lat = float(pt_elem.attrib["lat"])
        lon = float(pt_elem.attrib["lon"])
    except (KeyError, TypeError, ValueError) as err:
        raise rsgispy.RSGISPyException(
            f"Invalid or missing lat/lon on GPX track point {pt_id} in "
            f"track {track_fid}, segment {seg_id}."
        ) from err

    row: Dict[str, Any] = {
        "track_fid": track_fid,
        "track_seg_id": seg_id,
        "track_seg_point_id": pt_id,
        "track_name": track_name,
        "ele": math.nan,
        "time": None,
    }
    for child in pt_elem:
        cname = _xml_local_name(child.tag)
        if cname == "extensions":
            _extract_gpx_extensions(child, row)
        elif len(child) == 0 and child.text is not None:
            text_val = child.text.strip()
            if text_val != "":
                row[cname] = _coerce_gpx_scalar(cname, text_val)

    ele_val = row.get("ele", math.nan)
    has_z = (
        include_z
        and isinstance(ele_val, (int, float))
        and not math.isnan(float(ele_val))
    )
    coord: Tuple[float, ...] = (
        (lon, lat, float(ele_val)) if has_z else (lon, lat)
    )
    row["geometry"] = Point(coord)
    time_val = row.get("time")
    return row, coord, (str(time_val) if time_val is not None else None)


def _parse_gpx_track_meta(trk_elem: ET.Element) -> Dict[str, Any]:
    """Extract standard track-level metadata attributes from a ``<trk>``."""
    meta: Dict[str, Any] = {
        "name": None,
        "cmt": None,
        "desc": None,
        "src": None,
        "type": None,
        "number": None,
    }
    for child in trk_elem:
        cname = _xml_local_name(child.tag)
        if cname in meta and child.text is not None:
            text_val = child.text.strip()
            if text_val != "":
                meta[cname] = _coerce_gpx_scalar(cname, text_val)
    return meta


def _parse_gpx_tracks(
    gpx_path: str,
    include_z: bool = False,
    split_segments: bool = False,
) -> Tuple[geopandas.GeoDataFrame, geopandas.GeoDataFrame]:
    """Parse GPX tracks into point and line GeoDataFrames in ``EPSG:4326``."""
    # pylint: disable=too-many-locals
    try:
        tree = ET.parse(gpx_path)
    except (ET.ParseError, OSError) as err:
        raise rsgispy.RSGISPyException(
            f"Could not parse GPX file '{gpx_path}': {err}"
        ) from err

    root = tree.getroot()
    if _xml_local_name(root.tag).lower() != "gpx":
        raise rsgispy.RSGISPyException(
            f"File '{gpx_path}' is not a valid GPX document "
            f"(root element '{root.tag}')."
        )

    pt_rows: List[Dict[str, Any]] = []
    line_rows: List[Dict[str, Any]] = []

    trk_elems = [el for el in root if _xml_local_name(el.tag) == "trk"]
    for track_fid, trk_elem in enumerate(trk_elems):
        meta = _parse_gpx_track_meta(trk_elem)
        seg_elems = [
            el for el in trk_elem if _xml_local_name(el.tag) == "trkseg"
        ]
        trk_lines: List[LineString] = []
        trk_n_pts = 0
        trk_times: List[str] = []

        for seg_id, seg_elem in enumerate(seg_elems):
            seg_coords: List[Tuple[float, ...]] = []
            seg_times: List[str] = []
            pt_elems = [
                el for el in seg_elem if _xml_local_name(el.tag) == "trkpt"
            ]
            for pt_id, pt_elem in enumerate(pt_elems):
                pt_row, coord, time_str = _parse_gpx_trkpt(
                    pt_elem,
                    track_fid=track_fid,
                    seg_id=seg_id,
                    pt_id=pt_id,
                    track_name=meta["name"],
                    include_z=include_z,
                )
                pt_rows.append(pt_row)
                seg_coords.append(coord)
                if time_str is not None:
                    seg_times.append(time_str)

            trk_n_pts += len(seg_coords)
            trk_times.extend(seg_times)
            if len(seg_coords) >= 2:
                seg_line = LineString(seg_coords)
                trk_lines.append(seg_line)
                if split_segments:
                    line_rows.append(
                        {
                            "track_fid": track_fid,
                            "track_seg_id": seg_id,
                            **meta,
                            "n_pts": len(seg_coords),
                            "start_time": seg_times[0] if seg_times else None,
                            "end_time": seg_times[-1] if seg_times else None,
                            "geometry": seg_line,
                        }
                    )

        if (not split_segments) and trk_lines:
            trk_geom: Union[LineString, MultiLineString] = (
                trk_lines[0]
                if len(trk_lines) == 1
                else MultiLineString(trk_lines)
            )
            line_rows.append(
                {
                    "track_fid": track_fid,
                    **meta,
                    "n_pts": trk_n_pts,
                    "start_time": trk_times[0] if trk_times else None,
                    "end_time": trk_times[-1] if trk_times else None,
                    "geometry": trk_geom,
                }
            )

    if pt_rows:
        pts_gdf = geopandas.GeoDataFrame(
            pt_rows, geometry="geometry", crs="EPSG:4326"
        )
    else:
        pts_gdf = geopandas.GeoDataFrame(
            {
                "track_fid": pandas.Series(dtype="int64"),
                "track_seg_id": pandas.Series(dtype="int64"),
                "track_seg_point_id": pandas.Series(dtype="int64"),
                "track_name": pandas.Series(dtype="object"),
                "ele": pandas.Series(dtype="float64"),
                "time": pandas.Series(dtype="object"),
            },
            geometry=geopandas.GeoSeries([], crs="EPSG:4326"),
            crs="EPSG:4326",
        )

    if line_rows:
        lines_gdf = geopandas.GeoDataFrame(
            line_rows, geometry="geometry", crs="EPSG:4326"
        )
    else:
        empty_line_cols: Dict[str, Any] = {
            "track_fid": pandas.Series(dtype="int64"),
        }
        if split_segments:
            empty_line_cols["track_seg_id"] = pandas.Series(dtype="int64")
        empty_line_cols.update(
            {
                "name": pandas.Series(dtype="object"),
                "cmt": pandas.Series(dtype="object"),
                "desc": pandas.Series(dtype="object"),
                "src": pandas.Series(dtype="object"),
                "type": pandas.Series(dtype="object"),
                "number": pandas.Series(dtype="object"),
                "n_pts": pandas.Series(dtype="int64"),
                "start_time": pandas.Series(dtype="object"),
                "end_time": pandas.Series(dtype="object"),
            }
        )
        lines_gdf = geopandas.GeoDataFrame(
            empty_line_cols,
            geometry=geopandas.GeoSeries([], crs="EPSG:4326"),
            crs="EPSG:4326",
        )

    return pts_gdf, lines_gdf


def convert_gpx_to_vec(
    gpx_file: Union[str, os.PathLike],
    pts_vec_file: Optional[Union[str, os.PathLike]] = None,
    pts_vec_lyr: Optional[str] = None,
    line_vec_file: Optional[Union[str, os.PathLike]] = None,
    line_vec_lyr: Optional[str] = None,
    out_format: str = "GPKG",
    out_epsg: int = 4326,
    split_segments: bool = False,
    include_z: bool = False,
    del_exist_vec: bool = False,
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[
    Union[
        geopandas.GeoDataFrame,
        Tuple[geopandas.GeoDataFrame, geopandas.GeoDataFrame],
    ]
]:
    """
    Convert a GPX file into track-point and track-line vector layers,
    returning in-memory :class:`geopandas.GeoDataFrame` objects and/or saving
    to vector files.

    When both ``pts_vec_file`` and ``line_vec_file`` are ``None`` (the default),
    a ``(pts_gdf, lines_gdf)`` tuple is returned. If only one output path is
    provided, the remaining layer's ``GeoDataFrame`` is returned. For
    multi-layer formats such as ``GPKG``, the same path may be passed to both
    ``pts_vec_file`` and ``line_vec_file`` with distinct layer names (defaulting
    to ``'track_points'`` and ``'tracks'``).

    :param gpx_file: Path to the input GPX file.
    :param pts_vec_file: Optional output vector file path for track points, or
                         ``None`` (default) to return the points
                         :class:`geopandas.GeoDataFrame`.
    :param pts_vec_lyr: Output layer name for track points (defaults to
                        ``'track_points'`` for GPKG).
    :param line_vec_file: Optional output vector file path for track lines, or
                          ``None`` (default) to return the lines
                          :class:`geopandas.GeoDataFrame`.
    :param line_vec_lyr: Output layer name for track lines (defaults to
                         ``'tracks'`` for GPKG).
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param out_epsg: Target EPSG integer code for both output layers
                     (Default = ``4326``, WGS84).
    :param split_segments: If ``True``, emit one ``LineString`` feature per
                           ``<trkseg>`` instead of one ``LineString`` /
                           ``MultiLineString`` per ``<trk>`` (Default = ``False``).
    :param include_z: If ``True``, include elevation (``<ele>``) as a Z
                      coordinate on geometries when present (Default = ``False``).
    :param del_exist_vec: If ``True``, delete existing output file(s) before
                          writing (Default = ``False``).
    :param out_compress: Optional GeoParquet compression setting (``None`` to
                         infer from extension or write uncompressed, ``True``
                         for ``'snappy'``, ``False``/``'none'`` for uncompressed,
                         or codec name ``'snappy'``, ``'gzip'``, ``'zstd'``,
                         ``'brotli'``, ``'lz4'``).
    :return: ``(pts_gdf, lines_gdf)`` when both ``pts_vec_file`` and
             ``line_vec_file`` are ``None``; a single
             :class:`geopandas.GeoDataFrame` when only one path is ``None``;
             or ``None`` when both layers are written to disk.

    .. code:: python

        import rsgispy.vectorutils.io

        pts_gdf, lines_gdf = rsgispy.vectorutils.io.convert_gpx_to_vec(
            "route.gpx"
        )
        rsgispy.vectorutils.io.convert_gpx_to_vec(
            "route.gpx",
            pts_vec_file="route.gpkg",
            pts_vec_lyr="track_points",
            line_vec_file="route.gpkg",
            line_vec_lyr="tracks",
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    gpx_path = os.fspath(gpx_file)
    ensure_file_exists(gpx_path)

    same_file = (
        pts_vec_file is not None
        and line_vec_file is not None
        and os.path.abspath(os.fspath(pts_vec_file))
        == os.path.abspath(os.fspath(line_vec_file))
    )
    resolved_pts_lyr = pts_vec_lyr or "track_points"
    resolved_line_lyr = line_vec_lyr or "tracks"
    if same_file:
        fmt = check_format_name(out_format)
        if is_geoparquet_format(fmt) or is_geoparquet_file(pts_vec_file):  # type: ignore[arg-type]
            raise rsgispy.RSGISPyException(
                "GeoParquet only supports a single layer per file; specify "
                "separate paths for pts_vec_file and line_vec_file."
            )
        if resolved_pts_lyr == resolved_line_lyr:
            raise rsgispy.RSGISPyException(
                "pts_vec_lyr and line_vec_lyr must differ when writing both "
                "layers to the same file."
            )

    prepare_optional_out_vec(pts_vec_file, del_exist_vec=del_exist_vec)
    if not same_file:
        prepare_optional_out_vec(line_vec_file, del_exist_vec=del_exist_vec)

    pts_gdf, lines_gdf = _parse_gpx_tracks(
        gpx_path,
        include_z=include_z,
        split_segments=split_segments,
    )
    if int(out_epsg) != 4326:
        target_crs = parse_crs_input(int(out_epsg))
        pts_gdf = pts_gdf.to_crs(target_crs)
        lines_gdf = lines_gdf.to_crs(target_crs)

    if pts_vec_file is not None:
        write_vec_gdf(
            pts_gdf,
            out_vec_file=pts_vec_file,
            out_vec_lyr=resolve_out_lyr_name(pts_vec_file, resolved_pts_lyr),
            out_format=out_format,
            overwrite=False,
            out_compress=out_compress,
        )

    if line_vec_file is not None:
        write_vec_gdf(
            lines_gdf,
            out_vec_file=line_vec_file,
            out_vec_lyr=resolve_out_lyr_name(line_vec_file, resolved_line_lyr),
            out_format=out_format,
            overwrite=False,
            out_compress=out_compress,
        )

    if pts_vec_file is None and line_vec_file is None:
        return pts_gdf, lines_gdf
    if pts_vec_file is None:
        return pts_gdf
    if line_vec_file is None:
        return lines_gdf
    return None
