"""
Vector copying, translation, reprojection, merging, and splitting utilities for
:mod:`rsgispy.vectorutils` using ``geopandas``, ``pyogrio``, and ``shapely``.
"""

import math
import os
from typing import Dict, List, Optional, Sequence, Union

import geopandas
import pandas
from shapely.geometry import LinearRing, Polygon

import rsgispy
from rsgispy.imageprocess import ensure_file_exists
import rsgispy.tools.files
from rsgispy.tools.geometry import calc_bbox_area
import rsgispy.tools.strings
import rsgispy.tools.utm
import rsgispy.vectorgeoms.spatial
from rsgispy.vectorprocess import (
    CRSLike,
    bbox_to_rsgis,
    check_format_name,
    extract_geom_pts,
    parse_crs_input,
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
    vec_file: str,
    vec_lyr: Optional[str],
    out_vec_file: str,
    out_vec_lyr: Optional[str] = None,
    out_format: str = "GPKG",
    replace: bool = False,
) -> None:
    """
    Create a copy of an input vector layer.

    :param vec_file: Input vector file path.
    :param vec_lyr: Input vector layer name. If ``None``, reads the first layer.
    :param out_vec_file: Output vector file path.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format (Default = ``'GPKG'``).
    :param replace: If ``True``, delete ``out_vec_file`` before writing
                    (Default = ``False``).

    .. code:: python

        import rsgispy.vectorutils.io

        rsgispy.vectorutils.io.create_copy_vec_lyr(
            "in.gpkg", "polys", "copy.gpkg", "polys_copy", replace=True
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    lyr_name = resolve_vec_layer(vec_file, vec_lyr)
    gdf = read_vec_gdf(vec_file, vec_lyr=lyr_name)
    out_lyr = resolve_out_lyr_name(out_vec_file, out_vec_lyr or lyr_name)
    write_vec_gdf(
        gdf,
        out_vec_file=out_vec_file,
        out_vec_lyr=out_lyr,
        out_format=out_format,
        overwrite=replace,
    )


def vector_translate(
    vec_file: str,
    vec_lyr: Optional[str],
    out_vec_file: str,
    out_vec_lyr: Optional[str] = None,
    out_format: str = "GPKG",
    access_mode: Optional[str] = None,
    src_srs: Optional[CRSLike] = None,
    dst_srs: Optional[CRSLike] = None,
    del_exist_vec: bool = False,
) -> None:
    """
    Translate a vector layer between formats and optionally assign or reproject
    its coordinate reference system.

    :param vec_file: Input vector file path.
    :param vec_lyr: Input vector layer name. If ``None``, reads the first layer.
    :param out_vec_file: Output vector file path.
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

    lyr_name = resolve_vec_layer(vec_file, vec_lyr)
    gdf = read_vec_gdf(vec_file, vec_lyr=lyr_name)

    if src_srs is not None:
        gdf = gdf.set_crs(parse_crs_input(src_srs), allow_override=True)
    if dst_srs is not None:
        gdf = gdf.to_crs(parse_crs_input(dst_srs))

    if del_exist_vec or access_mode == "overwrite":
        prepare_out_vec_file(out_vec_file, del_exist_vec=True)

    fmt = check_format_name(out_format)
    out_lyr = resolve_out_lyr_name(out_vec_file, out_vec_lyr or lyr_name)
    is_gpkg = fmt == "GPKG" or out_vec_file.lower().endswith(".gpkg")

    if access_mode == "append" and os.path.exists(out_vec_file):
        if is_gpkg:
            existing_lyrs = get_vec_lyrs_lst(out_vec_file)
            if out_lyr in existing_lyrs:
                gdf.to_file(
                    out_vec_file, layer=out_lyr, driver=fmt, mode="a"
                )
                return
        else:
            gdf.to_file(out_vec_file, driver=fmt, mode="a")
            return

    overwrite_file = del_exist_vec or (access_mode == "overwrite")
    write_vec_gdf(
        gdf,
        out_vec_file=out_vec_file,
        out_vec_lyr=out_lyr,
        out_format=fmt,
        overwrite=overwrite_file,
    )


def reproj_vec_lyr(
    vec_file: str,
    out_vec_file: str,
    out_vec_lyr: Optional[str] = None,
    epsg_code: Optional[int] = None,
    wkt_str: Optional[str] = None,
    out_format: str = "GPKG",
    vec_lyr: Optional[str] = None,
    in_epsg_code: Optional[int] = None,
    in_wkt_str: Optional[str] = None,
    del_exist_vec: bool = False,
) -> None:
    """
    Reproject a vector layer to a target EPSG code or WKT CRS string using
    ``geopandas``.

    :param vec_file: Input vector file path.
    :param out_vec_file: Output vector file path.
    :param out_vec_lyr: Output vector layer name.
    :param epsg_code: Target EPSG integer code (mutually optional with ``wkt_str``).
    :param wkt_str: Target projection WKT string (used if ``epsg_code`` is ``None``).
    :param out_format: Output vector format (Default = ``'GPKG'``).
    :param vec_lyr: Input vector layer name. If ``None``, reads the first layer.
    :param in_epsg_code: Optional input EPSG override applied before reprojecting.
    :param in_wkt_str: Optional input WKT CRS override applied before reprojecting.
    :param del_exist_vec: If ``True``, delete existing output file first
                          (Default = ``False``).

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
            f"Input layer '{lyr_name}' in '{vec_file}' has no CRS defined; "
            "provide in_epsg_code or in_wkt_str."
        )

    prepare_out_vec_file(out_vec_file, del_exist_vec=del_exist_vec)
    reproj_gdf = gdf.to_crs(out_crs)
    write_vec_gdf(
        reproj_gdf,
        out_vec_file=out_vec_file,
        out_vec_lyr=resolve_out_lyr_name(out_vec_file, out_vec_lyr or lyr_name),
        out_format=out_format,
    )


def reproj_wgs84_vec_to_utm(
    vec_file: str,
    vec_lyr: Optional[str],
    out_vec_file: str,
    out_vec_lyr: Optional[str] = None,
    use_hemi: bool = False,
    out_format: str = "GPKG",
    del_exist_vec: bool = False,
) -> int:
    """
    Reproject a WGS84 (``EPSG:4326``) vector layer to the UTM zone containing its
    mean feature vertex coordinate.

    :param vec_file: Input WGS84 vector file path.
    :param vec_lyr: Input vector layer name. If ``None``, reads the first layer.
    :param out_vec_file: Output vector file path.
    :param out_vec_lyr: Output vector layer name.
    :param use_hemi: If ``True``, select Southern hemisphere EPSG when mean
                     latitude < 0; otherwise always use Northern hemisphere
                     (Default = ``False``).
    :param out_format: Output vector format (Default = ``'GPKG'``).
    :param del_exist_vec: If ``True``, delete existing output file first
                          (Default = ``False``).
    :return: Target UTM EPSG code used for reprojection.

    .. code:: python

        import rsgispy.vectorutils.io

        utm_epsg = rsgispy.vectorutils.io.reproj_wgs84_vec_to_utm(
            "sites_wgs84.gpkg", "sites", "sites_utm.gpkg", "sites", use_hemi=True
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    lyr_name = resolve_vec_layer(vec_file, vec_lyr)
    gdf = read_vec_gdf(vec_file, vec_lyr=lyr_name, epsg=4326)

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

    reproj_vec_lyr(
        vec_file=vec_file,
        out_vec_file=out_vec_file,
        out_vec_lyr=out_vec_lyr or lyr_name,
        epsg_code=out_epsg,
        out_format=out_format,
        vec_lyr=lyr_name,
        in_epsg_code=4326,
        del_exist_vec=del_exist_vec,
    )
    return out_epsg


def redefine_vec_lyr_proj(
    vec_file: str,
    vec_lyr: str,
    epsg_code: int,
    out_vec_file: str,
    out_vec_lyr: str,
    out_format: str = "GPKG",
) -> None:
    """
    Redefine the CRS metadata of a vector layer in place of reprojecting
    coordinates.

    :param vec_file: Input vector file path.
    :param vec_lyr: Input vector layer name.
    :param epsg_code: EPSG integer code to assign without altering coordinates.
    :param out_vec_file: Output vector file path.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format (Default = ``'GPKG'``).

    .. code:: python

        import rsgispy.vectorutils.io

        rsgispy.vectorutils.io.redefine_vec_lyr_proj(
            "raw.gpkg", "raw", 27700, "defined.gpkg", "defined"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    gdf = read_vec_gdf(vec_file, vec_lyr=vec_lyr, epsg=int(epsg_code))
    write_vec_gdf(
        gdf,
        out_vec_file=out_vec_file,
        out_vec_lyr=out_vec_lyr,
        out_format=out_format,
    )


def merge_vec_files(
    vec_files: Sequence[str],
    out_vec_file: str,
    out_vec_lyr: Optional[str] = None,
    out_format: str = "GPKG",
    out_epsg: Optional[int] = None,
    remove_cols: Optional[Sequence[str]] = None,
    use_pbar: bool = True,
) -> None:
    """
    Merge all layers from a sequence of vector files into a single output
    vector layer using ``geopandas``.

    :param vec_files: Sequence of input vector file paths.
    :param out_vec_file: Output vector file path.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format (Default = ``'GPKG'``).
    :param out_epsg: Optional target EPSG code to unify differing projections.
    :param remove_cols: Optional column names to drop during merging.
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).

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
        lyrs = get_vec_lyrs_lst(vec_file)
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
    write_vec_gdf(
        merged_gdf,
        out_vec_file=out_vec_file,
        out_vec_lyr=resolve_out_lyr_name(out_vec_file, out_vec_lyr),
        out_format=out_format,
    )


def merge_vec_lyrs(
    vecs_dict: Sequence[Dict[str, str]],
    out_vec_file: str,
    out_vec_lyr: Optional[str] = None,
    out_format: str = "GPKG",
    out_epsg: Optional[int] = None,
    remove_cols: Optional[Sequence[str]] = None,
    use_pbar: bool = True,
) -> None:
    """
    Merge specified ``{'file': ..., 'layer': ...}`` entries into a single
    output vector layer using ``geopandas``.

    :param vecs_dict: Sequence of dicts with keys ``'file'`` and ``'layer'``.
    :param out_vec_file: Output vector file path.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format (Default = ``'GPKG'``).
    :param out_epsg: Optional target EPSG code to unify differing projections.
    :param remove_cols: Optional column names to drop during merging.
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).

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
        if "file" not in vec_info or "layer" not in vec_info:
            raise rsgispy.RSGISPyException(
                "Each entry must be a dict with keys 'file' and 'layer'."
            )
        data_gdf = read_vec_gdf(vec_info["file"], vec_lyr=vec_info["layer"])
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
    write_vec_gdf(
        merged_gdf,
        out_vec_file=out_vec_file,
        out_vec_lyr=resolve_out_lyr_name(out_vec_file, out_vec_lyr),
        out_format=out_format,
    )


def merge_vec_file_lyrs(
    vec_file: str,
    out_vec_file: str,
    out_vec_lyr: Optional[str] = None,
    out_format: str = "GPKG",
    out_epsg: Optional[int] = None,
    remove_cols: Optional[Sequence[str]] = None,
    use_pbar: bool = True,
) -> None:
    """
    Merge all layers inside a single multi-layer vector file into a single
    output layer.

    :param vec_file: Input multi-layer vector file path.
    :param out_vec_file: Output vector file path.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format (Default = ``'GPKG'``).
    :param out_epsg: Optional target EPSG code.
    :param remove_cols: Optional column names to drop during merging.
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).

    .. code:: python

        import rsgispy.vectorutils.io

        rsgispy.vectorutils.io.merge_vec_file_lyrs(
            "multi.gpkg", "single.gpkg", out_vec_lyr="combined"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    merge_vec_files(
        vec_files=[vec_file],
        out_vec_file=out_vec_file,
        out_vec_lyr=out_vec_lyr,
        out_format=out_format,
        out_epsg=out_epsg,
        remove_cols=remove_cols,
        use_pbar=use_pbar,
    )


def merge_to_multi_lyr_vec(
    input_file_lyrs: Sequence[Union[VecLayersInfoObj, str]],
    out_vec_file: str,
    out_format: str = "GPKG",
    overwrite: bool = True,
    rename_dup_lyrs: bool = False,
    use_pbar: bool = True,
) -> None:
    """
    Combine multiple input vector files or :class:`VecLayersInfoObj` descriptors
    as separate layers in a multi-layer vector file (e.g., ``GPKG``).

    :param input_file_lyrs: Sequence of :class:`VecLayersInfoObj` items or input
                            file path strings (all layers from each file are
                            copied when strings are supplied).
    :param out_vec_file: Output multi-layer vector file path.
    :param out_format: Output vector format (Default = ``'GPKG'``).
    :param overwrite: If ``True``, delete ``out_vec_file`` if it exists before
                      writing (Default = ``True``).
    :param rename_dup_lyrs: If ``True``, append a unique suffix when two inputs
                            share the same output layer name; if ``False``,
                            raise :class:`rsgispy.RSGISPyException`
                            (Default = ``False``).
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).

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
    if not rsgispy.tools.files.does_path_exists_or_creatable(out_vec_file):
        raise rsgispy.RSGISPyException(
            f"Output file path is not creatable: {out_vec_file}"
        )

    if overwrite and os.path.exists(out_vec_file):
        prepare_out_vec_file(out_vec_file, del_exist_vec=True)

    seen_out_lyrs = set()
    if os.path.exists(out_vec_file) and not overwrite:
        seen_out_lyrs.update(get_vec_lyrs_lst(out_vec_file))

    entries: List[VecLayersInfoObj] = []
    for item in input_file_lyrs:
        if isinstance(item, VecLayersInfoObj):
            if not item.vec_file:
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
        else:
            for lyr in get_vec_lyrs_lst(str(item)):
                entries.append(
                    VecLayersInfoObj(
                        vec_file=str(item),
                        vec_lyr=lyr,
                        vec_out_lyr=lyr,
                    )
                )

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
            write_vec_gdf(
                gdf,
                out_vec_file=out_vec_file,
                out_vec_lyr=out_lyr,
                out_format=out_format,
            )
            seen_out_lyrs.add(out_lyr)


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
    in_vec_files: Sequence[str],
    out_vec_file: str,
    out_vec_lyr: Optional[str] = None,
    out_format: str = "GPKG",
    n_hemi_utm_file: Optional[str] = None,
    s_hemi_utm_file: Optional[str] = None,
    width_thres: float = 350.0,
    use_pbar: bool = True,
) -> None:
    """
    Merge input vector files projected in UTM zones into a single WGS84
    (``EPSG:4326``) vector layer, splitting features crossing UTM zone boundaries
    and unwrapping geometries spanning the 180/-180 antimeridian.

    :param in_vec_files: Sequence of input UTM vector file paths.
    :param out_vec_file: Output WGS84 vector file path.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format (Default = ``'GPKG'``).
    :param n_hemi_utm_file: Path to Northern hemisphere UTM zone boundary
                            multi-layer vector file (layers ``'01'`` to ``'60'``).
    :param s_hemi_utm_file: Path to Southern hemisphere UTM zone boundary
                            multi-layer vector file (layers ``'01'`` to ``'60'``).
    :param width_thres: Longitude width threshold in degrees for unwrapping
                        antimeridian polygons (Default = ``350.0``).
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).

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
    ensure_file_exists(n_hemi_utm_file, label="Northern UTM zones file")
    ensure_file_exists(s_hemi_utm_file, label="Southern UTM zones file")

    pbar_cls = rsgispy.get_tqdm()
    out_gdfs: List[geopandas.GeoDataFrame] = []

    for vec_file in pbar_cls(in_vec_files, disable=not use_pbar):
        for lyr in get_vec_lyrs_lst(vec_file):
            data_gdf = read_vec_gdf(vec_file, vec_lyr=lyr)
            if len(data_gdf) == 0:
                continue
            bbox = bbox_to_rsgis(data_gdf.total_bounds)
            if calc_bbox_area(bbox) <= 0:
                continue

            vec_epsg = get_epsg_proj_from_vec(vec_file, vec_lyr=lyr)
            if vec_epsg is None:
                raise rsgispy.RSGISPyException(
                    f"Could not determine EPSG code for '{vec_file}' ({lyr})."
                )
            zone, hemi = rsgispy.tools.utm.utm_from_epsg(int(vec_epsg))
            zone_str = rsgispy.tools.strings.zero_pad_num_str(
                zone, str_len=2, integerise=True
            )
            utm_zones_file = (
                s_hemi_utm_file if hemi.upper() == "S" else n_hemi_utm_file
            )

            contained = rsgispy.vectorgeoms.spatial.vec_within_vec(
                utm_zones_file, zone_str, vec_file, lyr, use_pbar=False
            )
            if not contained:
                utm_gdf = read_vec_gdf(utm_zones_file, vec_lyr=zone_str)
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
        write_vec_gdf(
            merged,
            out_vec_file=out_vec_file,
            out_vec_lyr=resolve_out_lyr_name(out_vec_file, out_vec_lyr),
            out_format=out_format,
        )


def split_vec_lyr(
    vec_file: str,
    vec_lyr: str,
    n_feats: int,
    out_format: str,
    out_dir: str,
    out_vec_base: str,
    out_vec_ext: str,
    use_pbar: bool = True,
) -> List[str]:
    """
    Split an input vector layer into sequential output files each containing at
    most ``n_feats`` features.

    :param vec_file: Input vector file path.
    :param vec_lyr: Input vector layer name.
    :param n_feats: Maximum number of features per output file.
    :param out_format: Output vector driver name.
    :param out_dir: Output directory path (created if absent).
    :param out_vec_base: Output filename and layer name prefix.
    :param out_vec_ext: Output file extension without leading dot (e.g., ``'gpkg'``).
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).
    :return: List of created output vector file paths.

    .. code:: python

        import rsgispy.vectorutils.io

        out_files = rsgispy.vectorutils.io.split_vec_lyr(
            "large.gpkg", "large", 1000, "GPKG", "splits", "chunk_", "gpkg"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    if n_feats < 1:
        raise rsgispy.RSGISPyException("n_feats must be >= 1.")
    os.makedirs(out_dir, exist_ok=True)

    gdf = read_vec_gdf(vec_file, vec_lyr=vec_lyr)
    gdf = gdf[gdf.geometry.notna()]
    total_feats = len(gdf)
    n_chunks = math.ceil(total_feats / n_feats)

    pbar_cls = rsgispy.get_tqdm()
    out_files: List[str] = []
    for chunk_idx in pbar_cls(range(n_chunks), disable=not use_pbar):
        start = chunk_idx * n_feats
        end = min((chunk_idx + 1) * n_feats, total_feats)
        chunk_gdf = gdf.iloc[start:end]
        out_lyr = f"{out_vec_base}{chunk_idx + 1}"
        out_file = os.path.join(out_dir, f"{out_lyr}.{out_vec_ext}")
        write_vec_gdf(
            chunk_gdf,
            out_vec_file=out_file,
            out_vec_lyr=out_lyr,
            out_format=out_format,
        )
        out_files.append(out_file)
    return out_files


def split_feats_to_mlyrs(
    vec_file: str,
    vec_lyr: str,
    out_vec_file: str,
    out_format: str = "GPKG",
    use_pbar: bool = True,
) -> None:
    """
    Split every feature in an input vector layer into its own named layer
    (``veclyr_0``, ``veclyr_1``, ...) inside ``out_vec_file``.

    :param vec_file: Input vector file path.
    :param vec_lyr: Input vector layer name.
    :param out_vec_file: Output multi-layer vector file path.
    :param out_format: Output vector format supporting multiple layers
                       (Default = ``'GPKG'``).
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).

    .. code:: python

        import rsgispy.vectorutils.io

        rsgispy.vectorutils.io.split_feats_to_mlyrs(
            "polys.gpkg", "polys", "polys_multi.gpkg"
        )
    """
    gdf = read_vec_gdf(vec_file, vec_lyr=vec_lyr)
    pbar_cls = rsgispy.get_tqdm()
    for i in pbar_cls(range(len(gdf)), disable=not use_pbar):
        row_gdf = gdf.iloc[[i]]
        write_vec_gdf(
            row_gdf,
            out_vec_file=out_vec_file,
            out_vec_lyr=f"veclyr_{i}",
            out_format=out_format,
        )


def split_by_attribute(
    vec_file: str,
    vec_lyr: str,
    split_col_name: str,
    multi_layers: bool = True,
    out_vec_file: Optional[str] = None,
    out_file_path: Optional[str] = None,
    out_file_ext: Optional[str] = None,
    out_format: str = "GPKG",
    dissolve: bool = False,
    chk_lyr_names: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Split a vector layer by unique values in ``split_col_name`` into either
    multiple layers in a single GeoPackage or separate output files.

    :param vec_file: Input vector file path.
    :param vec_lyr: Input vector layer name.
    :param split_col_name: Attribute column whose unique values define splits.
    :param multi_layers: If ``True`` (default), write all splits into layers in
                         ``out_vec_file``; if ``False``, write separate files to
                         ``out_file_path``.
    :param out_vec_file: Output multi-layer file path (required when
                         ``multi_layers=True``).
    :param out_file_path: Output directory path (required when
                          ``multi_layers=False``).
    :param out_file_ext: Output file extension without dot (required when
                         ``multi_layers=False``).
    :param out_format: Output vector format (Default = ``'GPKG'``).
    :param dissolve: If ``True``, dissolve features per group before writing
                     (Default = ``False``).
    :param chk_lyr_names: If ``True``, strip non-ASCII characters and
                          punctuation from split layer/file names
                          (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).

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
    if multi_layers and out_vec_file is None:
        raise rsgispy.RSGISPyException(
            "out_vec_file must be specified when multi_layers=True."
        )
    if (not multi_layers) and ((out_file_path is None) or (out_file_ext is None)):
        raise rsgispy.RSGISPyException(
            "out_file_path and out_file_ext must be specified when multi_layers=False."
        )

    base_gdf = read_vec_gdf(vec_file, vec_lyr=vec_lyr)
    validate_columns_exist(base_gdf, [split_col_name])
    unq_vals = base_gdf[split_col_name].unique()
    fmt = check_format_name(out_format)

    pbar_cls = rsgispy.get_tqdm()
    for val in pbar_cls(unq_vals, disable=not use_pbar):
        sub_gdf = base_gdf.loc[base_gdf[split_col_name] == val]
        sub_gdf = sub_gdf[sub_gdf.geometry.notna() & (~sub_gdf.geometry.is_empty)]
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

        if multi_layers and fmt == "GPKG":
            assert out_vec_file is not None
            write_vec_gdf(
                sub_gdf,
                out_vec_file=out_vec_file,
                out_vec_lyr=val_str,
                out_format="GPKG",
            )
        else:
            assert out_file_path is not None
            assert out_file_ext is not None
            c_out_lyr = f"vec_{val_str}"
            c_out_file = os.path.join(
                out_file_path, f"{c_out_lyr}.{out_file_ext}"
            )
            write_vec_gdf(
                sub_gdf,
                out_vec_file=c_out_file,
                out_vec_lyr=c_out_lyr,
                out_format=fmt,
            )
