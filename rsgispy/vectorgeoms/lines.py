"""
The vectorgeoms.lines module provides geometric functions for line vectors,
including orthogonal transects, regular point sampling, line-boundary
intersections, point-to-line construction, and coastline estimation using
``geopandas`` and ``shapely``.
"""

import datetime
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import geopandas as gpd
import numpy as np
import rasterio
import shapely.geometry

import rsgispy
from rsgispy.imageutils.metadata import get_img_bbox
from rsgispy.vectorprocess import (
    CRSLike,
    VectorInput,
    extract_geom_pts,
    parse_crs_input,
    prepare_optional_out_vec,
    pt_xy,
    read_vec_gdf,
    resolve_out_lyr_name,
    validate_columns_exist,
    write_vec_gdf,
)

__all__ = [
    "get_pt_on_line",
    "find_pt_to_side",
    "create_orthg_lines",
    "split_lines_to_reg_pts",
    "closest_line_intersection",
    "line_intersection_range",
    "scnd_line_intersection_range",
    "pts_to_line_geoms",
    "create_angle_lines_from_points",
    "estimate_future_coastline",
    "create_lines_vec",
    "create_img_transects",
]


def get_pt_on_line(
    pt1: Union[shapely.geometry.Point, Sequence[float]],
    pt2: Union[shapely.geometry.Point, Sequence[float]],
    dist: float,
) -> Tuple[float, float]:
    """
    Calculate a point along the ray from ``pt1`` towards ``pt2`` at distance
    ``dist``.

    :param pt1: Start point (``shapely.geometry.Point`` or ``(x, y)``).
    :param pt2: Direction/end point (``shapely.geometry.Point`` or ``(x, y)``).
    :param dist: Distance from ``pt1`` at which to create the point.
    :return: Created point coordinates ``(x, y)``.

    .. code:: python

        from shapely.geometry import Point
        import rsgispy.vectorgeoms.lines

        x, y = rsgispy.vectorgeoms.lines.get_pt_on_line(
            Point(0.0, 0.0), Point(10.0, 0.0), dist=4.0
        )
    """
    x1, y1 = pt_xy(pt1)
    x2, y2 = pt_xy(pt2)
    if dist == 0:
        return x1, y1

    dx = x2 - x1
    dy = y2 - y1
    seg_len = math.hypot(dx, dy)
    if seg_len == 0:
        return x1, y1

    scale = dist / seg_len
    return x1 + (dx * scale), y1 + (dy * scale)


def find_pt_to_side(
    pt_start: Union[shapely.geometry.Point, Sequence[float]],
    pt: Union[shapely.geometry.Point, Sequence[float]],
    pt_end: Union[shapely.geometry.Point, Sequence[float]],
    line_len: float,
    left_hand: bool = False,
) -> Tuple[float, float]:
    """
    Calculate a point location perpendicular to the vector defined by
    ``pt_start`` and ``pt_end`` at the base point ``pt``.

    :param pt_start: Segment start point (``shapely.geometry.Point`` or ``(x, y)``).
    :param pt: Base point on the segment from which to project
               (``shapely.geometry.Point`` or ``(x, y)``).
    :param pt_end: Segment end point (``shapely.geometry.Point`` or ``(x, y)``).
    :param line_len: Perpendicular distance from ``pt`` to the new point.
    :param left_hand: If ``False`` (default), project to the right-hand side
                      of ``pt_start -> pt_end``; if ``True``, project to the
                      left-hand side.
    :return: Projected point coordinates ``(x, y)``.

    .. code:: python

        from shapely.geometry import Point
        import rsgispy.vectorgeoms.lines

        x, y = rsgispy.vectorgeoms.lines.find_pt_to_side(
            Point(0.0, 0.0), Point(5.0, 0.0), Point(10.0, 0.0), line_len=3.0
        )
    """
    x_start, y_start = pt_xy(pt_start)
    x_base, y_base = pt_xy(pt)
    x_end, y_end = pt_xy(pt_end)

    dx = x_end - x_start
    dy = y_end - y_start
    seg_len = math.hypot(dx, dy)
    if seg_len == 0:
        raise rsgispy.RSGISPyException(
            "Cannot determine perpendicular direction for a zero-length segment."
        )

    if left_hand:
        out_x = x_base - (line_len * dy / seg_len)
        out_y = y_base + (line_len * dx / seg_len)
    else:
        out_x = x_base + (line_len * dy / seg_len)
        out_y = y_base - (line_len * dx / seg_len)
    return out_x, out_y


def _iter_line_parts(geom) -> List[shapely.geometry.LineString]:
    """Return single LineString parts from LineString or MultiLineString."""
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "LineString":
        return [geom]
    if geom.geom_type == "MultiLineString":
        return [g for g in geom.geoms if not g.is_empty]
    raise rsgispy.RSGISPyException(
        f"Expected LineString or MultiLineString geometry, got '{geom.geom_type}'."
    )


def create_orthg_lines(
    vec_in_file: VectorInput,
    vec_in_lyr: Optional[str],
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    pt_step: float = 1000,
    line_len: float = 10000,
    left_hand: bool = False,
    out_format: str = "GPKG",
    del_exist_vec: bool = False,
    use_pbar: bool = True,
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Create a set of lines orthogonal to the lines in an input vector layer.

    :param vec_in_file: Input polyline vector file path or
                        :class:`geopandas.GeoDataFrame`.
    :param vec_in_lyr: Input vector layer name (or ``None``).
    :param out_vec_file: Output polyline vector file path, or ``None`` to return
                         an in-memory :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name (derived from filename if ``None``).
    :param pt_step: Step interval along lines (in CRS units) at which
                    orthogonal lines are generated (Default = ``1000``).
    :param line_len: Length of each generated orthogonal line (Default = ``10000``).
    :param left_hand: Project to the left-hand side when ``True``, or
                      right-hand side when ``False`` (Default = ``False``).
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param del_exist_vec: If ``True``, overwrite existing ``out_vec_file``
                          (Default = ``False``).
    :param use_pbar: Show progress bar (Default = ``True``).
    :param out_compress: Optional GeoParquet compression setting (``None`` to
                         infer from extension or write uncompressed, ``True``
                         for ``'snappy'``, ``False``/``'none'`` for uncompressed,
                         or codec name ``'snappy'``, ``'gzip'``, ``'zstd'``,
                         ``'brotli'``, ``'lz4'``).
    :return: Orthogonal lines :class:`geopandas.GeoDataFrame` when
             ``out_vec_file`` is ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.vectorgeoms.lines

        rsgispy.vectorgeoms.lines.create_orthg_lines(
            "coast.gpkg", "coast", "transects.gpkg", "transects",
            pt_step=50.0, line_len=100.0
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    out_path = prepare_optional_out_vec(out_vec_file, del_exist_vec=del_exist_vec)
    out_lyr = resolve_out_lyr_name(out_path, out_vec_lyr)

    in_gdf = read_vec_gdf(vec_in_file, vec_in_lyr, copy=False)
    records: List[Dict[str, Any]] = []
    line_uid = 1
    pbar_cls = rsgispy.get_tqdm()

    for geom in pbar_cls(
        in_gdf.geometry, total=len(in_gdf), disable=not use_pbar
    ):
        for line_part in _iter_line_parts(geom):
            pts = list(line_part.coords)
            if len(pts) < 2:
                continue
            p_pt = pts[0]
            c_pt = pts[0]
            c_dist = 0.0
            for pt in pts[1:]:
                p_pt = c_pt
                c_pt = pt
                seg_len = math.hypot(c_pt[0] - p_pt[0], c_pt[1] - p_pt[1])
                n_step = 0
                step_c_dist = c_dist
                while ((seg_len + step_c_dist) - (pt_step * n_step)) > pt_step:
                    pt_at_dist = ((pt_step * n_step) + pt_step) - step_c_dist
                    ptx, pty = get_pt_on_line(p_pt, c_pt, pt_at_dist)
                    ptx_end, pty_end = find_pt_to_side(
                        p_pt, (ptx, pty), c_pt, line_len, left_hand
                    )
                    records.append(
                        {
                            "uid": line_uid,
                            "start_x": ptx,
                            "start_y": pty,
                            "end_x": ptx_end,
                            "end_y": pty_end,
                            "geometry": shapely.geometry.LineString(
                                [(ptx, pty), (ptx_end, pty_end)]
                            ),
                        }
                    )
                    line_uid += 1
                    n_step += 1
                if n_step == 0:
                    c_dist = c_dist + seg_len
                else:
                    c_dist = (seg_len + step_c_dist) - (pt_step * n_step)

    if records:
        out_gdf = gpd.GeoDataFrame(
            records, geometry="geometry", crs=in_gdf.crs
        )
    else:
        out_gdf = gpd.GeoDataFrame(
            {
                "uid": pandas_int_empty(),
                "start_x": pandas_float_empty(),
                "start_y": pandas_float_empty(),
                "end_x": pandas_float_empty(),
                "end_y": pandas_float_empty(),
            },
            geometry=gpd.GeoSeries([], crs=in_gdf.crs),
            crs=in_gdf.crs,
        )
    if out_path is None:
        return out_gdf
    write_vec_gdf(
        out_gdf,
        out_path,
        out_lyr,
        out_format=out_format,
        out_compress=out_compress,
    )
    return None


def pandas_int_empty() -> np.ndarray:
    """Return an empty integer array for empty typed GeoDataFrames."""
    return np.array([], dtype=int)


def pandas_float_empty() -> np.ndarray:
    """Return an empty float array for empty typed GeoDataFrames."""
    return np.array([], dtype=float)


def split_lines_to_reg_pts(
    vec_in_file: VectorInput,
    vec_in_lyr: Optional[str],
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    pt_step: float = 1000,
    out_format: str = "GPKG",
    del_exist_vec: bool = False,
    use_pbar: bool = True,
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Split a line vector layer into points placed at regular intervals
    (``pt_step``) along each line.

    Output points contain ``uid``, ``line_id``, ``pt_x``, ``pt_y``, and
    ``dist`` attributes.

    :param vec_in_file: Input polyline vector file path or
                        :class:`geopandas.GeoDataFrame`.
    :param vec_in_lyr: Input vector layer name (or ``None``).
    :param out_vec_file: Output points vector file path, or ``None`` to return
                         an in-memory :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name (derived from filename if ``None``).
    :param pt_step: Step interval along lines in CRS units (Default = ``1000``).
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param del_exist_vec: If ``True``, overwrite existing ``out_vec_file``
                          (Default = ``False``).
    :param use_pbar: Show progress bar (Default = ``True``).
    :param out_compress: Optional GeoParquet compression setting (``None`` to
                         infer from extension or write uncompressed, ``True``
                         for ``'snappy'``, ``False``/``'none'`` for uncompressed,
                         or codec name ``'snappy'``, ``'gzip'``, ``'zstd'``,
                         ``'brotli'``, ``'lz4'``).
    :return: Points :class:`geopandas.GeoDataFrame` when ``out_vec_file`` is
             ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.vectorgeoms.lines

        rsgispy.vectorgeoms.lines.split_lines_to_reg_pts(
            "lines.gpkg", "lines", "pts.gpkg", "pts", pt_step=25.0
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    out_path = prepare_optional_out_vec(out_vec_file, del_exist_vec=del_exist_vec)
    out_lyr = resolve_out_lyr_name(out_path, out_vec_lyr)

    in_gdf = read_vec_gdf(vec_in_file, vec_in_lyr, copy=False)
    records: List[Dict[str, Any]] = []
    pt_uid = 1
    line_id = 1
    pbar_cls = rsgispy.get_tqdm()

    for geom in pbar_cls(
        in_gdf.geometry, total=len(in_gdf), disable=not use_pbar
    ):
        for line_part in _iter_line_parts(geom):
            pts = list(line_part.coords)
            if len(pts) < 2:
                continue
            p_pt = pts[0]
            c_pt = pts[0]
            c_dist = 0.0
            line_dist = 0.0
            for pt in pts[1:]:
                p_pt = c_pt
                c_pt = pt
                seg_len = math.hypot(c_pt[0] - p_pt[0], c_pt[1] - p_pt[1])
                n_step = 0
                step_c_dist = c_dist
                while ((seg_len + step_c_dist) - (pt_step * n_step)) > pt_step:
                    pt_at_dist = ((pt_step * n_step) + pt_step) - step_c_dist
                    ptx, pty = get_pt_on_line(p_pt, c_pt, pt_at_dist)
                    records.append(
                        {
                            "uid": pt_uid,
                            "line_id": line_id,
                            "pt_x": ptx,
                            "pt_y": pty,
                            "dist": line_dist,
                            "geometry": shapely.geometry.Point(ptx, pty),
                        }
                    )
                    pt_uid += 1
                    n_step += 1
                    line_dist += pt_step
                if n_step == 0:
                    c_dist = c_dist + seg_len
                else:
                    c_dist = (seg_len + step_c_dist) - (pt_step * n_step)
        line_id += 1

    if records:
        out_gdf = gpd.GeoDataFrame(
            records, geometry="geometry", crs=in_gdf.crs
        )
    else:
        out_gdf = gpd.GeoDataFrame(
            {
                "uid": pandas_int_empty(),
                "line_id": pandas_int_empty(),
                "pt_x": pandas_float_empty(),
                "pt_y": pandas_float_empty(),
                "dist": pandas_float_empty(),
            },
            geometry=gpd.GeoSeries([], crs=in_gdf.crs),
            crs=in_gdf.crs,
        )
    if out_path is None:
        return out_gdf
    write_vec_gdf(
        out_gdf,
        out_path,
        out_lyr,
        out_format=out_format,
        out_compress=out_compress,
    )
    return None


def _calc_line_boundary_intersections(
    vec_line_file: VectorInput,
    vec_line_lyr: Optional[str],
    vec_objs_file: VectorInput,
    vec_objs_lyr: Optional[str],
    out_vec_file: Optional[Union[str, os.PathLike]],
    out_vec_lyr: Optional[str],
    start_x_field: str,
    start_y_field: str,
    uid_field: str,
    out_format: str,
    del_exist_vec: bool,
    mode: str,
    use_pbar: bool,
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Shared implementation for line-to-object-boundary intersection modes:
    ``'closest'``, ``'range'``, and ``'second'``.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    out_path = prepare_optional_out_vec(out_vec_file, del_exist_vec=del_exist_vec)
    out_lyr = resolve_out_lyr_name(out_path, out_vec_lyr)

    line_gdf = read_vec_gdf(vec_line_file, vec_line_lyr, copy=False)
    validate_columns_exist(line_gdf, [start_x_field, start_y_field, uid_field])
    objs_gdf = read_vec_gdf(vec_objs_file, vec_objs_lyr, copy=False)

    if not line_gdf.empty and not objs_gdf.empty:
        xmin, ymin, xmax, ymax = line_gdf.total_bounds
        objs_sub_gdf = objs_gdf.cx[xmin:xmax, ymin:ymax]
    else:
        objs_sub_gdf = objs_gdf

    bnd_geoms = objs_sub_gdf.geometry[objs_sub_gdf.geometry.notna()].boundary
    bnd_geoms = bnd_geoms[~bnd_geoms.is_empty]

    records: List[Dict[str, Any]] = []
    if len(bnd_geoms) > 0 and not line_gdf.empty:
        bnd_sindex = bnd_geoms.sindex
        bnd_arr = bnd_geoms.to_numpy()
        pbar_cls = rsgispy.get_tqdm()
        rows_iter = pbar_cls(
            line_gdf.itertuples(index=False),
            total=len(line_gdf),
            disable=not use_pbar,
        )

        x_idx = line_gdf.columns.get_loc(start_x_field)
        y_idx = line_gdf.columns.get_loc(start_y_field)
        uid_idx = line_gdf.columns.get_loc(uid_field)
        geom_idx = line_gdf.columns.get_loc("geometry")

        for row in rows_iter:
            line_geom = row[geom_idx]
            if line_geom is None or line_geom.is_empty:
                continue

            cand_idxs = bnd_sindex.query(line_geom, predicate="intersects")
            if len(cand_idxs) == 0:
                continue

            inter_pts: List[Tuple[float, float]] = []
            for idx in cand_idxs:
                inter_geom = bnd_arr[idx].intersection(line_geom)
                if not inter_geom.is_empty:
                    extract_geom_pts(inter_geom, inter_pts)

            if not inter_pts:
                continue

            start_x = float(row[x_idx])
            start_y = float(row[y_idx])
            uid_val = row[uid_idx]

            pts_with_dist = sorted(
                (math.hypot(px - start_x, py - start_y), px, py)
                for px, py in inter_pts
            )

            if mode == "closest":
                min_dist, end_x, end_y = pts_with_dist[0]
                records.append(
                    {
                        "uid": uid_val,
                        "len": min_dist,
                        "geometry": shapely.geometry.LineString(
                            [(start_x, start_y), (end_x, end_y)]
                        ),
                    }
                )
            elif mode == "range":
                _, min_x, min_y = pts_with_dist[0]
                _, max_x, max_y = pts_with_dist[-1]
                dist = math.hypot(max_x - min_x, max_y - min_y)
                records.append(
                    {
                        "uid": uid_val,
                        "len": dist,
                        "geometry": shapely.geometry.LineString(
                            [(min_x, min_y), (max_x, max_y)]
                        ),
                    }
                )
            elif mode == "second":
                if len(pts_with_dist) < 2:
                    continue
                sec_dist, end_x, end_y = pts_with_dist[1]
                records.append(
                    {
                        "uid": uid_val,
                        "start_x": start_x,
                        "start_y": start_y,
                        "end_x": end_x,
                        "end_y": end_y,
                        "len": sec_dist,
                        "geometry": shapely.geometry.LineString(
                            [(start_x, start_y), (end_x, end_y)]
                        ),
                    }
                )

    if records:
        out_gdf = gpd.GeoDataFrame(
            records, geometry="geometry", crs=objs_gdf.crs
        )
    else:
        empty_cols: Dict[str, Any] = {
            "uid": pandas_int_empty(),
            "len": pandas_float_empty(),
        }
        if mode == "second":
            empty_cols = {
                "uid": pandas_int_empty(),
                "start_x": pandas_float_empty(),
                "start_y": pandas_float_empty(),
                "end_x": pandas_float_empty(),
                "end_y": pandas_float_empty(),
                "len": pandas_float_empty(),
            }
        out_gdf = gpd.GeoDataFrame(
            empty_cols,
            geometry=gpd.GeoSeries([], crs=objs_gdf.crs),
            crs=objs_gdf.crs,
        )

    if out_path is None:
        return out_gdf
    write_vec_gdf(
        out_gdf,
        out_path,
        out_lyr,
        out_format=out_format,
        out_compress=out_compress,
    )
    return None


def closest_line_intersection(
    vec_line_file: VectorInput,
    vec_line_lyr: Optional[str],
    vec_objs_file: VectorInput,
    vec_objs_lyr: Optional[str],
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    start_x_field: str = "start_x",
    start_y_field: str = "start_y",
    uid_field: str = "uid",
    out_format: str = "GPKG",
    del_exist_vec: bool = False,
    use_pbar: bool = True,
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Intersect each line with the boundaries of objects in ``vec_objs_file``,
    creating a new line between each input line's start point
    (``start_x_field``, ``start_y_field``) and its closest intersection point.

    :param vec_line_file: Input lines vector file path or
                          :class:`geopandas.GeoDataFrame`.
    :param vec_line_lyr: Input lines vector layer name (or ``None``).
    :param vec_objs_file: Object polygons vector file path or
                          :class:`geopandas.GeoDataFrame` to intersect against.
    :param vec_objs_lyr: Object polygons vector layer name (or ``None``).
    :param out_vec_file: Output vector file path, or ``None`` to return an
                         in-memory :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name.
    :param start_x_field: Attribute column for line start X (Default = ``'start_x'``).
    :param start_y_field: Attribute column for line start Y (Default = ``'start_y'``).
    :param uid_field: Attribute column for line UID (Default = ``'uid'``).
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param del_exist_vec: If ``True``, overwrite existing ``out_vec_file``
                          (Default = ``False``).
    :param use_pbar: Show progress bar (Default = ``True``).
    :param out_compress: Optional GeoParquet compression setting (``None`` to
                         infer from extension or write uncompressed, ``True``
                         for ``'snappy'``, ``False``/``'none'`` for uncompressed,
                         or codec name ``'snappy'``, ``'gzip'``, ``'zstd'``,
                         ``'brotli'``, ``'lz4'``).
    :return: Intersected lines :class:`geopandas.GeoDataFrame` when
             ``out_vec_file`` is ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.vectorgeoms.lines

        rsgispy.vectorgeoms.lines.closest_line_intersection(
            "transects.gpkg", "transects", "polys.gpkg", "polys",
            "closest.gpkg", "closest"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    return _calc_line_boundary_intersections(
        vec_line_file=vec_line_file,
        vec_line_lyr=vec_line_lyr,
        vec_objs_file=vec_objs_file,
        vec_objs_lyr=vec_objs_lyr,
        out_vec_file=out_vec_file,
        out_vec_lyr=out_vec_lyr,
        start_x_field=start_x_field,
        start_y_field=start_y_field,
        uid_field=uid_field,
        out_format=out_format,
        del_exist_vec=del_exist_vec,
        mode="closest",
        use_pbar=use_pbar,
        out_compress=out_compress,
    )


def line_intersection_range(
    vec_line_file: VectorInput,
    vec_line_lyr: Optional[str],
    vec_objs_file: VectorInput,
    vec_objs_lyr: Optional[str],
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    start_x_field: str = "start_x",
    start_y_field: str = "start_y",
    uid_field: str = "uid",
    out_format: str = "GPKG",
    del_exist_vec: bool = False,
    use_pbar: bool = True,
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Intersect each line with the boundaries of objects in ``vec_objs_file``,
    creating a new line spanning from the closest to the furthest intersection
    point relative to ``(start_x_field, start_y_field)``.

    :param vec_line_file: Input lines vector file path or GeoDataFrame.
    :param vec_line_lyr: Input lines vector layer name (or ``None``).
    :param vec_objs_file: Object polygons vector file path or GeoDataFrame to
                          intersect against.
    :param vec_objs_lyr: Object polygons vector layer name (or ``None``).
    :param out_vec_file: Output vector file path, or ``None`` to return an
                         in-memory :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name.
    :param start_x_field: Attribute column for line start X (Default = ``'start_x'``).
    :param start_y_field: Attribute column for line start Y (Default = ``'start_y'``).
    :param uid_field: Attribute column for line UID (Default = ``'uid'``).
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param del_exist_vec: If ``True``, overwrite existing ``out_vec_file``
                          (Default = ``False``).
    :param use_pbar: Show progress bar (Default = ``True``).
    :param out_compress: Optional GeoParquet compression setting (``None`` to
                         infer from extension or write uncompressed, ``True``
                         for ``'snappy'``, ``False``/``'none'`` for uncompressed,
                         or codec name ``'snappy'``, ``'gzip'``, ``'zstd'``,
                         ``'brotli'``, ``'lz4'``).
    :return: Resulting :class:`geopandas.GeoDataFrame` when ``out_vec_file`` is
             ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.vectorgeoms.lines

        rsgispy.vectorgeoms.lines.line_intersection_range(
            "transects.gpkg", "transects", "polys.gpkg", "polys",
            "range.gpkg", "range"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    return _calc_line_boundary_intersections(
        vec_line_file=vec_line_file,
        vec_line_lyr=vec_line_lyr,
        vec_objs_file=vec_objs_file,
        vec_objs_lyr=vec_objs_lyr,
        out_vec_file=out_vec_file,
        out_vec_lyr=out_vec_lyr,
        start_x_field=start_x_field,
        start_y_field=start_y_field,
        uid_field=uid_field,
        out_format=out_format,
        del_exist_vec=del_exist_vec,
        mode="range",
        use_pbar=use_pbar,
        out_compress=out_compress,
    )


def scnd_line_intersection_range(
    vec_line_file: VectorInput,
    vec_line_lyr: Optional[str],
    vec_objs_file: VectorInput,
    vec_objs_lyr: Optional[str],
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    start_x_field: str = "start_x",
    start_y_field: str = "start_y",
    uid_field: str = "uid",
    out_format: str = "GPKG",
    del_exist_vec: bool = False,
    use_pbar: bool = True,
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Intersect each line with the boundaries of objects in ``vec_objs_file``,
    cutting each line from its start point to its second closest intersection
    point.

    :param vec_line_file: Input lines vector file path or GeoDataFrame.
    :param vec_line_lyr: Input lines vector layer name (or ``None``).
    :param vec_objs_file: Object polygons vector file path or GeoDataFrame to
                          intersect against.
    :param vec_objs_lyr: Object polygons vector layer name (or ``None``).
    :param out_vec_file: Output vector file path, or ``None`` to return an
                         in-memory :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name.
    :param start_x_field: Attribute column for line start X (Default = ``'start_x'``).
    :param start_y_field: Attribute column for line start Y (Default = ``'start_y'``).
    :param uid_field: Attribute column for line UID (Default = ``'uid'``).
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param del_exist_vec: If ``True``, overwrite existing ``out_vec_file``
                          (Default = ``False``).
    :param use_pbar: Show progress bar (Default = ``True``).
    :param out_compress: Optional GeoParquet compression setting (``None`` to
                         infer from extension or write uncompressed, ``True``
                         for ``'snappy'``, ``False``/``'none'`` for uncompressed,
                         or codec name ``'snappy'``, ``'gzip'``, ``'zstd'``,
                         ``'brotli'``, ``'lz4'``).
    :return: Resulting :class:`geopandas.GeoDataFrame` when ``out_vec_file`` is
             ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.vectorgeoms.lines

        rsgispy.vectorgeoms.lines.scnd_line_intersection_range(
            "transects.gpkg", "transects", "polys.gpkg", "polys",
            "second.gpkg", "second"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    return _calc_line_boundary_intersections(
        vec_line_file=vec_line_file,
        vec_line_lyr=vec_line_lyr,
        vec_objs_file=vec_objs_file,
        vec_objs_lyr=vec_objs_lyr,
        out_vec_file=out_vec_file,
        out_vec_lyr=out_vec_lyr,
        start_x_field=start_x_field,
        start_y_field=start_y_field,
        uid_field=uid_field,
        out_format=out_format,
        del_exist_vec=del_exist_vec,
        mode="second",
        use_pbar=use_pbar,
        out_compress=out_compress,
    )


def pts_to_line_geoms(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    out_format: str = "GPKG",
    sort_col: Optional[str] = None,
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Convert a vector layer of points to a single line geometry. If the input
    contains ``MultiPoint`` geometries, they are exploded before constructing
    the line.

    :param vec_file: Input point vector file path or GeoDataFrame.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param out_vec_file: Output line vector file path, or ``None`` to return an
                         in-memory :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name (derived from filename if ``None``).
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param sort_col: Optional attribute column name used to sort points prior
                     to creating the line.
    :param out_compress: Optional GeoParquet compression setting (``None`` to
                         infer from extension or write uncompressed, ``True``
                         for ``'snappy'``, ``False``/``'none'`` for uncompressed,
                         or codec name ``'snappy'``, ``'gzip'``, ``'zstd'``,
                         ``'brotli'``, ``'lz4'``).
    :return: Resulting line :class:`geopandas.GeoDataFrame` when
             ``out_vec_file`` is ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.vectorgeoms.lines

        rsgispy.vectorgeoms.lines.pts_to_line_geoms(
            "track_pts.gpkg", "pts", "track_line.gpkg", "line", sort_col="seq"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    data_gdf = read_vec_gdf(vec_file, vec_lyr)
    geom_types = set(data_gdf.geometry.geom_type.dropna().unique())

    if not geom_types.issubset({"Point", "MultiPoint"}):
        raise rsgispy.RSGISPyException("Input vector must be Point geometry.")
    if "MultiPoint" in geom_types:
        data_gdf = data_gdf.explode(index_parts=False)

    if sort_col is not None:
        validate_columns_exist(data_gdf, [sort_col])
        data_gdf = data_gdf.sort_values(by=sort_col, ascending=True)

    pts = data_gdf.geometry.values
    if len(pts) < 2:
        raise rsgispy.RSGISPyException(
            "At least two points are required to construct a line geometry."
        )
    line_geom = shapely.geometry.LineString(pts)
    line_gdf = gpd.GeoDataFrame(
        {"id": [1], "geometry": [line_geom]}, crs=data_gdf.crs
    )
    if out_vec_file is None:
        return line_gdf
    out_lyr = resolve_out_lyr_name(out_vec_file, out_vec_lyr)
    write_vec_gdf(
        line_gdf,
        out_vec_file,
        out_lyr,
        out_format=out_format,
        out_compress=out_compress,
    )
    return None


def create_angle_lines_from_points(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    angle: float,
    line_len: float,
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    out_format: str = "GPKG",
    use_pbar: bool = True,
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Generate radial line segments of length ``line_len`` at angular increments
    ``angle`` (in degrees) around each point feature in ``vec_file``.

    :param vec_file: Input point vector file path or GeoDataFrame.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param angle: Angular spacing in degrees between generated lines.
    :param line_len: Length of each generated line in CRS units.
    :param out_vec_file: Output vector file path, or ``None`` to return an
                         in-memory :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param use_pbar: Show progress bar (Default = ``True``).
    :param out_compress: Optional GeoParquet compression setting (``None`` to
                         infer from extension or write uncompressed, ``True``
                         for ``'snappy'``, ``False``/``'none'`` for uncompressed,
                         or codec name ``'snappy'``, ``'gzip'``, ``'zstd'``,
                         ``'brotli'``, ``'lz4'``).
    :return: Resulting radial lines :class:`geopandas.GeoDataFrame` when
             ``out_vec_file`` is ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.vectorgeoms.lines

        rsgispy.vectorgeoms.lines.create_angle_lines_from_points(
            "pts.gpkg", "pts", angle=45.0, line_len=10.0,
            out_vec_file="rays.gpkg", out_vec_lyr="rays"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    line_angles = np.arange(0, 360, angle)
    pts_gdf = read_vec_gdf(vec_file, vec_lyr, copy=False)
    pbar_cls = rsgispy.get_tqdm()

    pt_ids: List[Any] = []
    geoms: List[shapely.geometry.LineString] = []

    for idx, pt in pbar_cls(
        zip(pts_gdf.index, pts_gdf.geometry),
        total=len(pts_gdf),
        disable=not use_pbar,
    ):
        for line_angle in line_angles:
            line_angle_rad = math.radians(float(line_angle))
            end_x = pt.x + (line_len * math.cos(line_angle_rad))
            end_y = pt.y + (line_len * math.sin(line_angle_rad))
            shp_line = shapely.geometry.LineString(
                [pt, shapely.geometry.Point(end_x, end_y)]
            )
            pt_ids.append(idx)
            geoms.append(shp_line)

    out_gdf = gpd.GeoDataFrame(
        {"pt_id": pt_ids, "geometry": geoms}, geometry="geometry", crs=pts_gdf.crs
    )
    if out_vec_file is None:
        return out_gdf
    write_vec_gdf(
        out_gdf,
        out_vec_file,
        out_vec_lyr,
        out_format=out_format,
        out_compress=out_compress,
    )
    return None


def estimate_future_coastline(
    curr_vec_file: VectorInput,
    curr_vec_lyr: Optional[str],
    curr_vec_date: datetime.date,
    trans_vec_file: VectorInput,
    trans_vec_lyr: Optional[str],
    out_vec_file: Optional[Union[str, os.PathLike]],
    out_vec_lyr: Optional[str],
    out_pred_date: datetime.date,
    trans_rate_col: str = "LRR",
    out_format: str = "GeoJSON",
    trans_pt_order_land_first: bool = False,
    use_pbar: bool = True,
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Estimate a future coastline position using baseline coastline geometry and
    DSAS-style transects with annual rate-of-change values.

    :param curr_vec_file: Current/baseline coastline vector file path or
                          GeoDataFrame.
    :param curr_vec_lyr: Baseline coastline vector layer name (or ``None``).
    :param curr_vec_date: Date of the current coastline (``datetime.date``).
    :param trans_vec_file: Transect vector file path or GeoDataFrame.
    :param trans_vec_lyr: Transect vector layer name (or ``None``).
    :param out_vec_file: Output predicted coastline vector file path, or ``None``
                         to return an in-memory :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name.
    :param out_pred_date: Future prediction date (``datetime.date``).
    :param trans_rate_col: Transect rate-of-change column name (Default = ``'LRR'``).
    :param out_format: Output vector format driver (Default = ``'GeoJSON'``).
    :param trans_pt_order_land_first: ``True`` if transect coordinates are ordered
                                      land-to-sea; ``False`` if sea-to-land
                                      (Default = ``False``).
    :param use_pbar: Show progress bar (Default = ``True``).
    :param out_compress: Optional GeoParquet compression setting (``None`` to
                         infer from extension or write uncompressed, ``True``
                         for ``'snappy'``, ``False``/``'none'`` for uncompressed,
                         or codec name ``'snappy'``, ``'gzip'``, ``'zstd'``,
                         ``'brotli'``, ``'lz4'``).
    :return: Predicted coastline :class:`geopandas.GeoDataFrame` when
             ``out_vec_file`` is ``None``, otherwise ``None``.

    .. code:: python

        import datetime
        import rsgispy.vectorgeoms.lines

        rsgispy.vectorgeoms.lines.estimate_future_coastline(
            curr_vec_file="coast_2020.geojson",
            curr_vec_lyr=None,
            curr_vec_date=datetime.date(2020, 1, 1),
            trans_vec_file="transects.geojson",
            trans_vec_lyr=None,
            out_vec_file="coast_2050.geojson",
            out_vec_lyr=None,
            out_pred_date=datetime.date(2050, 1, 1),
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    if curr_vec_date > out_pred_date:
        raise rsgispy.RSGISPyException(
            "The current date needs to be before the prediction date."
        )

    pred_time_period = out_pred_date - curr_vec_date
    pred_period_years = pred_time_period.days / 365
    if pred_period_years < 1:
        raise rsgispy.RSGISPyException(
            "Difference between the current and predicted "
            "date needs to be more than 1 year."
        )

    curr_data_gdf = read_vec_gdf(curr_vec_file, curr_vec_lyr, copy=False)
    trans_data_gdf = read_vec_gdf(trans_vec_file, trans_vec_lyr, copy=False)
    validate_columns_exist(trans_data_gdf, [trans_rate_col])

    pbar_cls = rsgispy.get_tqdm()
    pt_geoms: List[shapely.geometry.Point] = []

    for trans_idx, trans_row in pbar_cls(
        trans_data_gdf.iterrows(),
        total=len(trans_data_gdf),
        disable=not use_pbar,
    ):
        trans_geom = trans_row["geometry"]
        trans_geom_coords = list(trans_geom.coords)
        if len(trans_geom_coords) < 2:
            continue

        if trans_pt_order_land_first:
            pt1 = trans_geom_coords[-1]
            pt2 = trans_geom_coords[0]
        else:
            pt1 = trans_geom_coords[0]
            pt2 = trans_geom_coords[-1]

        dy = pt2[1] - pt1[1]
        dx = pt2[0] - pt1[0]
        theta = math.atan2(dy, dx)

        rate_of_chng = float(trans_row[trans_rate_col])
        new_coast_dist = pred_period_years * rate_of_chng
        x_add = math.cos(theta) * new_coast_dist
        y_add = math.sin(theta) * new_coast_dist

        for _, curr_row in curr_data_gdf.iterrows():
            intersect = trans_geom.intersection(curr_row["geometry"])
            if intersect.is_empty or intersect.geom_type != "Point":
                continue
            st_pt = (intersect.x, intersect.y)
            pt_geoms.append(
                shapely.geometry.Point(st_pt[0] + x_add, st_pt[1] + y_add)
            )
            _ = trans_idx

    if len(pt_geoms) < 2:
        raise rsgispy.RSGISPyException(
            "Insufficient transect-coastline intersections to construct a "
            "predicted coastline."
        )

    new_coastline_geom = shapely.geometry.LineString(pt_geoms)
    out_gdf = gpd.GeoDataFrame(
        {"coastline_id": [1], "geometry": [new_coastline_geom]},
        geometry="geometry",
        crs=trans_data_gdf.crs,
    )
    if out_vec_file is None:
        return out_gdf
    write_vec_gdf(
        out_gdf,
        out_vec_file,
        out_vec_lyr,
        out_format=out_format,
        out_compress=out_compress,
    )
    return None


def create_lines_vec(
    vec_file: Optional[Union[str, os.PathLike]],
    vec_lyr: Optional[str],
    out_format: str,
    epsg_code: CRSLike,
    lines: Sequence[Sequence[Tuple[float, float]]],
    overwrite: bool = True,
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Create a vector layer of polylines (each given as ``[(x, y), ...]``
    vertices) with sequential ``line_id`` attributes.

    :param vec_file: Output vector file path, or ``None`` to return an
                     in-memory :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Output vector layer name (or ``None``).
    :param out_format: Output vector format driver.
    :param epsg_code: EPSG integer code or CRS defining output projection.
    :param lines: Sequence of vertex sequences defining each ``LineString``.
    :param overwrite: If ``True``, delete an existing ``vec_file`` first;
                      set ``False`` when appending layers to a GeoPackage
                      (Default = ``True``).
    :param out_compress: Optional GeoParquet compression setting (``None`` to
                         infer from extension or write uncompressed, ``True``
                         for ``'snappy'``, ``False``/``'none'`` for uncompressed,
                         or codec name ``'snappy'``, ``'gzip'``, ``'zstd'``,
                         ``'brotli'``, ``'lz4'``).
    :return: :class:`geopandas.GeoDataFrame` if ``vec_file`` is ``None``,
             otherwise ``None``.

    .. code:: python

        import rsgispy.vectorgeoms.lines

        rsgispy.vectorgeoms.lines.create_lines_vec(
            "lines.gpkg",
            "lines",
            "GPKG",
            32630,
            [[(0.0, 0.0), (10.0, 10.0)], [(0.0, 10.0), (10.0, 0.0)]],
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    geoms = [
        shapely.geometry.LineString([(float(pt[0]), float(pt[1])) for pt in line])
        for line in lines
    ]
    gdf = gpd.GeoDataFrame(
        {"line_id": list(range(1, len(lines) + 1))},
        geometry=geoms,
        crs=parse_crs_input(epsg_code),
    )
    if vec_file is None:
        return gdf
    write_vec_gdf(
        gdf,
        out_vec_file=vec_file,
        out_vec_lyr=vec_lyr,
        out_format=out_format,
        overwrite=overwrite,
        out_compress=out_compress,
    )
    return None


def create_img_transects(
    input_img: str,
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    x_intervals: Optional[Sequence[float]] = None,
    y_intervals: Optional[Sequence[float]] = None,
    out_format: str = "GPKG",
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Create axis-aligned transect lines across ``input_img`` at fractional
    positions ``0..1`` along X and/or Y.

    :param input_img: Input raster image path.
    :param out_vec_file: Optional output vector file path. If ``None``
                         (default), returns an in-memory
                         :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name (or ``None``).
    :param x_intervals: Sequence of fractional positions ``[0..1]`` along X for
                        vertical (North-South) transects.
    :param y_intervals: Sequence of fractional positions ``[0..1]`` along Y for
                        horizontal (East-West) transects.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param out_compress: Optional GeoParquet compression setting (``None`` to
                         infer from extension or write uncompressed, ``True``
                         for ``'snappy'``, ``False``/``'none'`` for uncompressed,
                         or codec name ``'snappy'``, ``'gzip'``, ``'zstd'``,
                         ``'brotli'``, ``'lz4'``).
    :return: :class:`geopandas.GeoDataFrame` if ``out_vec_file`` is ``None``,
             otherwise ``None``.

    .. code:: python

        import rsgispy.vectorgeoms.lines

        rsgispy.vectorgeoms.lines.create_img_transects(
            "scene.tif", "transects.gpkg", "transects", x_intervals=[0.25, 0.5, 0.75]
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    if (x_intervals is None) and (y_intervals is None):
        raise rsgispy.RSGISPyException(
            "Provide at least one of x_intervals or y_intervals."
        )

    for label, vals in (("x_intervals", x_intervals), ("y_intervals", y_intervals)):
        if vals is not None:
            for val in vals:
                if (val < 0.0) or (val > 1.0):
                    raise rsgispy.RSGISPyException(
                        f"{label} values must be between 0 and 1."
                    )

    img_bbox = get_img_bbox(input_img)
    with rasterio.open(input_img, "r") as src:
        img_crs = src.crs
    if img_crs is None:
        raise rsgispy.RSGISPyException(
            f"Input image '{input_img}' does not have a CRS defined."
        )

    img_width = img_bbox[1] - img_bbox[0]
    img_height = img_bbox[3] - img_bbox[2]

    out_lines: List[List[Tuple[float, float]]] = []
    if x_intervals is not None:
        for frac in x_intervals:
            x_pos = img_bbox[0] + (img_width * float(frac))
            out_lines.append([(x_pos, img_bbox[2]), (x_pos, img_bbox[3])])
    if y_intervals is not None:
        for frac in y_intervals:
            y_pos = img_bbox[2] + (img_height * float(frac))
            out_lines.append([(img_bbox[0], y_pos), (img_bbox[1], y_pos)])

    return create_lines_vec(
        vec_file=out_vec_file,
        vec_lyr=out_vec_lyr,
        out_format=out_format,
        epsg_code=img_crs,
        lines=out_lines,
        overwrite=True,
        out_compress=out_compress,
    )
