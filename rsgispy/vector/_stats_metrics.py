"""
Geometric measurements, spatial counts, and attribute summary statistics for
:mod:`rsgispy.vector` using ``geopandas``, ``shapely``, ``numpy``, and
``scipy``.
"""

import os
from typing import Any, Dict, List, Optional, Union

import geopandas as gpd
import numpy as np
import shapely.geometry

try:
    import scipy.spatial

    HAVE_SCIPY = True
except ImportError:
    scipy = None  # type: ignore[assignment]
    HAVE_SCIPY = False

import rsgispy
from rsgispy.vectorprocess import (
    VectorInput,
    finish_vec_output,
    get_gdf_geom_type,
    read_vec_gdf,
    validate_columns_exist,
)

__all__ = [
    "calc_vec_area",
    "calc_vec_length",
    "get_poly_hole_area",
    "calc_vec_pt_dist_angle",
    "calc_near_dist_to_feats",
    "count_pt_intersects",
    "calc_npts_in_radius",
    "count_feats_per_att_val",
]


def calc_vec_area(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    out_col: str,
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    out_format: str = "GPKG",
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Calculate the area of each polygon feature and store it in ``out_col``.

    :param vec_file: Input vector file path or :class:`geopandas.GeoDataFrame`
                     (polygon geometry expected).
    :param vec_lyr: Input vector layer name (or ``None``).
    :param out_col: Output area column name.
    :param out_vec_file: Output vector file path, or ``None`` to return an
                         in-memory :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param out_compress: Optional GeoParquet compression setting (``None`` to
                         infer from extension or write uncompressed, ``True``
                         for ``'snappy'``, ``False``/``'none'`` for uncompressed,
                         or codec name ``'snappy'``, ``'gzip'``, ``'zstd'``,
                         ``'brotli'``, ``'lz4'``).
    :return: Resulting :class:`geopandas.GeoDataFrame` when ``out_vec_file`` is
             ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.vector.stats

        rsgispy.vector.stats.calc_vec_area(
            "polys.gpkg", "polys", "area_m2", "polys_area.gpkg", "polys"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    data_gdf = read_vec_gdf(vec_file, vec_lyr)
    data_gdf[out_col] = data_gdf.geometry.area
    return finish_vec_output(
        data_gdf,
        out_vec_file,
        out_vec_lyr=out_vec_lyr,
        out_format=out_format,
        out_compress=out_compress,
    )


def calc_vec_length(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    out_col: str,
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    out_format: str = "GPKG",
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Calculate the length (or perimeter) of each line/polygon feature and store
    it in ``out_col``.

    :param vec_file: Input vector file path or :class:`geopandas.GeoDataFrame`
                     (line or polygon geometry expected).
    :param vec_lyr: Input vector layer name (or ``None``).
    :param out_col: Output length column name.
    :param out_vec_file: Output vector file path, or ``None`` to return an
                         in-memory :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param out_compress: Optional GeoParquet compression setting (``None`` to
                         infer from extension or write uncompressed, ``True``
                         for ``'snappy'``, ``False``/``'none'`` for uncompressed,
                         or codec name ``'snappy'``, ``'gzip'``, ``'zstd'``,
                         ``'brotli'``, ``'lz4'``).
    :return: Resulting :class:`geopandas.GeoDataFrame` when ``out_vec_file`` is
             ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.vector.stats

        rsgispy.vector.stats.calc_vec_length(
            "lines.gpkg", "lines", "len_m", "lines_len.gpkg", "lines"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    data_gdf = read_vec_gdf(vec_file, vec_lyr)
    data_gdf[out_col] = data_gdf.geometry.length
    return finish_vec_output(
        data_gdf,
        out_vec_file,
        out_vec_lyr=out_vec_lyr,
        out_format=out_format,
        out_compress=out_compress,
    )


def get_poly_hole_area(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    use_pbar: bool = True,
) -> List[float]:
    """
    Calculate the area of every interior hole across all polygons in a vector
    layer.

    :param vec_file: Input polygon vector file path or
                     :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param use_pbar: Show progress bar (Default = ``True``).
    :return: List of hole areas in CRS units squared.

    .. code:: python

        import rsgispy.vector.stats

        hole_areas = rsgispy.vector.stats.get_poly_hole_area(
            "polys_holes.gpkg", "polys"
        )
    """
    in_gdf = read_vec_gdf(vec_file, vec_lyr, copy=False)
    pbar_cls = rsgispy.get_tqdm()
    hole_areas: List[float] = []

    for geom in pbar_cls(
        in_gdf.geometry, total=len(in_gdf), disable=not use_pbar
    ):
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "Polygon":
            polys = [geom]
        elif geom.geom_type == "MultiPolygon":
            polys = list(geom.geoms)
        else:
            raise rsgispy.RSGISPyException(
                "Can only compute hole areas for polygon geometry."
            )
        for poly in polys:
            for ring in poly.interiors:
                hole_areas.append(float(shapely.geometry.Polygon(ring).area))

    return hole_areas


def calc_vec_pt_dist_angle(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    out_format: str = "GPKG",
    x_centre: Optional[float] = None,
    y_centre: Optional[float] = None,
    angle_col: str = "angle",
    dist_col: str = "dist",
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Calculate Euclidean distance and clockwise-from-north azimuth angle
    (0-360 degrees) for each point relative to ``(x_centre, y_centre)``.

    If ``x_centre`` or ``y_centre`` is ``None``, the mean of the point
    coordinates is used.

    :param vec_file: Input point vector file path or
                     :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param out_vec_file: Output vector file path, or ``None`` to return an
                         in-memory :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param x_centre: Optional reference X coordinate.
    :param y_centre: Optional reference Y coordinate.
    :param angle_col: Output azimuth column name (Default = ``'angle'``).
    :param dist_col: Output distance column name (Default = ``'dist'``).
    :param out_compress: Optional GeoParquet compression setting (``None`` to
                         infer from extension or write uncompressed, ``True``
                         for ``'snappy'``, ``False``/``'none'`` for uncompressed,
                         or codec name ``'snappy'``, ``'gzip'``, ``'zstd'``,
                         ``'brotli'``, ``'lz4'``).
    :return: Resulting :class:`geopandas.GeoDataFrame` when ``out_vec_file`` is
             ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.vector.stats

        rsgispy.vector.stats.calc_vec_pt_dist_angle(
            "pts.gpkg", "pts", "pts_ang.gpkg", "pts"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    data_gdf = read_vec_gdf(vec_file, vec_lyr)
    if get_gdf_geom_type(data_gdf) != rsgispy.GEOM_PT:
        raise rsgispy.RSGISPyException("Vector geometry type must be points.")

    valid_mask = (
        data_gdf.geometry.notna() & ~data_gdf.geometry.is_empty
    ).to_numpy()
    dists = np.full(len(data_gdf), np.nan, dtype=float)
    angles_out = np.full(len(data_gdf), np.nan, dtype=float)

    if np.any(valid_mask):
        valid_geoms = data_gdf.geometry.iloc[np.flatnonzero(valid_mask)]
        x_coords = valid_geoms.x.to_numpy(dtype=float)
        y_coords = valid_geoms.y.to_numpy(dtype=float)
        if x_centre is None:
            x_centre = float(x_coords.mean())
        if y_centre is None:
            y_centre = float(y_coords.mean())

        dists[valid_mask] = np.sqrt(
            (x_coords - x_centre) ** 2 + (y_coords - y_centre) ** 2
        )

        angles = np.rad2deg(
            np.arctan2(y_coords - y_centre, x_coords - x_centre)
        )
        angles_secs = np.zeros_like(angles)
        angles_secs[(angles >= 0) & (angles <= 90)] = 1
        angles_secs[angles > 90] = 2
        angles_secs[angles < 0] = 3

        angles[angles_secs == 1] = -(angles[angles_secs == 1] - 90.0)
        angles[angles_secs == 2] = -(angles[angles_secs == 2] - 180.0) + 270.0
        angles[angles_secs == 3] = -angles[angles_secs == 3] + 90.0
        angles_out[valid_mask] = angles

    data_gdf[dist_col] = dists
    data_gdf[angle_col] = angles_out
    return finish_vec_output(
        data_gdf,
        out_vec_file,
        out_vec_lyr=out_vec_lyr,
        out_format=out_format,
        out_compress=out_compress,
    )


def calc_near_dist_to_feats(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    vec_dist_file: VectorInput,
    vec_dist_lyr: Optional[str],
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    out_col: str = "dist",
    out_format: str = "GPKG",
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Compute nearest-feature distance from each feature in ``vec_file`` to
    features in ``vec_dist_file`` using :func:`geopandas.sjoin_nearest`.

    :param vec_file: Input vector file path or :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param vec_dist_file: Target vector file path or
                          :class:`geopandas.GeoDataFrame` to measure distance
                          to.
    :param vec_dist_lyr: Target vector layer name (or ``None``).
    :param out_vec_file: Output vector file path, or ``None`` to return an
                         in-memory :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name.
    :param out_col: Output distance column name (Default = ``'dist'``).
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param out_compress: Optional GeoParquet compression setting (``None`` to
                         infer from extension or write uncompressed, ``True``
                         for ``'snappy'``, ``False``/``'none'`` for uncompressed,
                         or codec name ``'snappy'``, ``'gzip'``, ``'zstd'``,
                         ``'brotli'``, ``'lz4'``).
    :return: Resulting :class:`geopandas.GeoDataFrame` when ``out_vec_file`` is
             ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.vector.stats

        rsgispy.vector.stats.calc_near_dist_to_feats(
            "pts.gpkg", "pts", "roads.gpkg", "roads", "pts_dist.gpkg", "pts"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    data_gdf = read_vec_gdf(vec_file, vec_lyr)
    dist_gdf = read_vec_gdf(vec_dist_file, vec_dist_lyr, copy=False)

    orig_index = data_gdf.index
    work_gdf = data_gdf.reset_index(drop=True)
    dist_geom_gdf = dist_gdf[[dist_gdf.geometry.name]]
    data_dist_gdf = gpd.sjoin_nearest(
        work_gdf,
        dist_geom_gdf,
        distance_col=out_col,
        lsuffix="left",
        rsuffix="right",
        exclusive=True,
    )
    if "index_right" in data_dist_gdf.columns:
        data_dist_gdf = data_dist_gdf.drop(columns=["index_right"])
    if data_dist_gdf.index.duplicated().any():
        data_dist_gdf = data_dist_gdf[
            ~data_dist_gdf.index.duplicated(keep="first")
        ]
    data_dist_gdf.index = orig_index[data_dist_gdf.index.to_numpy()]
    return finish_vec_output(
        data_dist_gdf,
        out_vec_file,
        out_vec_lyr=out_vec_lyr,
        out_format=out_format,
        out_compress=out_compress,
    )


def count_pt_intersects(
    vec_in_file: VectorInput,
    vec_in_lyr: Optional[str],
    vec_pts_file: VectorInput,
    vec_pts_lyr: Optional[str],
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    out_format: str = "GPKG",
    out_count_col: str = "n_points",
    tmp_col_name: str = "tmp_join_fid",
    vec_in_epsg: Optional[int] = None,
    vec_pts_epsg: Optional[int] = None,
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Count the number of points from ``vec_pts_file`` intersecting each polygon
    in ``vec_in_file``.

    Specifying EPSG codes overrides CRS definitions without reprojecting.

    :param vec_in_file: Input polygon vector file path or
                        :class:`geopandas.GeoDataFrame`.
    :param vec_in_lyr: Input polygon vector layer name (or ``None``).
    :param vec_pts_file: Input point vector file path or
                         :class:`geopandas.GeoDataFrame`.
    :param vec_pts_lyr: Input point vector layer name (or ``None``).
    :param out_vec_file: Output vector file path, or ``None`` to return an
                         in-memory :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param out_count_col: Output count column name (Default = ``'n_points'``).
    :param tmp_col_name: Temporary join ID column name (Default = ``'tmp_join_fid'``).
    :param vec_in_epsg: Optional EPSG code override for ``vec_in_file``.
    :param vec_pts_epsg: Optional EPSG code override for ``vec_pts_file``.
    :param out_compress: Optional GeoParquet compression setting (``None`` to
                         infer from extension or write uncompressed, ``True``
                         for ``'snappy'``, ``False``/``'none'`` for uncompressed,
                         or codec name ``'snappy'``, ``'gzip'``, ``'zstd'``,
                         ``'brotli'``, ``'lz4'``).
    :return: Resulting :class:`geopandas.GeoDataFrame` when ``out_vec_file`` is
             ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.vector.stats

        rsgispy.vector.stats.count_pt_intersects(
            "polys.gpkg", "polys", "pts.gpkg", "pts", "counts.gpkg", "polys"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    in_gdf = read_vec_gdf(vec_in_file, vec_in_lyr, epsg=vec_in_epsg)
    pts_gdf = read_vec_gdf(
        vec_pts_file, vec_pts_lyr, epsg=vec_pts_epsg, copy=False
    )

    drop_cols = [c for c in ("index_left", "index_right") if c in in_gdf.columns]
    if drop_cols:
        in_gdf = in_gdf.drop(columns=drop_cols)

    join_ids = np.arange(1, len(in_gdf) + 1, 1, dtype=int)
    in_gdf[tmp_col_name] = join_ids
    join_left = gpd.GeoDataFrame(
        {tmp_col_name: join_ids},
        geometry=in_gdf.geometry.to_numpy(),
        crs=in_gdf.crs,
    )
    join_right = gpd.GeoDataFrame(
        geometry=pts_gdf.geometry.to_numpy(),
        crs=pts_gdf.crs,
    )
    counts = (
        gpd.sjoin(join_left, join_right, how="inner", predicate="intersects")
        .groupby(tmp_col_name)
        .size()
        .rename(out_count_col)
        .reset_index()
    )
    in_count_gdf = in_gdf.merge(counts, on=tmp_col_name, how="left")
    in_count_gdf[out_count_col] = (
        in_count_gdf[out_count_col].fillna(0).astype(int)
    )
    drop_after = [
        c
        for c in ("index_left", "index_right", tmp_col_name)
        if c in in_count_gdf.columns
    ]
    in_count_gdf = in_count_gdf.drop(columns=drop_after)

    return finish_vec_output(
        in_count_gdf,
        out_vec_file,
        out_vec_lyr=out_vec_lyr,
        out_format=out_format,
        out_compress=out_compress,
    )


def calc_npts_in_radius(
    vec_in_file: VectorInput,
    vec_in_lyr: Optional[str],
    radius: float,
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    out_format: str = "GPKG",
    out_col_name: str = "n_pts_r",
    n_cores: int = 1,
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Count how many other points lie within ``radius`` of each point feature.

    :param vec_in_file: Input point vector file path or
                        :class:`geopandas.GeoDataFrame`.
    :param vec_in_lyr: Input point vector layer name (or ``None``).
    :param radius: Search radius in layer CRS units.
    :param out_vec_file: Output vector file path, or ``None`` to return an
                         in-memory :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param out_col_name: Output neighbour count column (Default = ``'n_pts_r'``).
    :param n_cores: Number of worker threads for KD-tree queries (``-1`` uses
                    all cores; Default = ``1``).
    :param out_compress: Optional GeoParquet compression setting (``None`` to
                         infer from extension or write uncompressed, ``True``
                         for ``'snappy'``, ``False``/``'none'`` for uncompressed,
                         or codec name ``'snappy'``, ``'gzip'``, ``'zstd'``,
                         ``'brotli'``, ``'lz4'``).
    :return: Resulting :class:`geopandas.GeoDataFrame` when ``out_vec_file`` is
             ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.vector.stats

        rsgispy.vector.stats.calc_npts_in_radius(
            "pts.gpkg", "pts", 50.0, "pts_density.gpkg", "pts"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    rsgispy.check_optional_dep(HAVE_SCIPY, "calc_npts_in_radius", "scipy", "stats")
    in_gdf = read_vec_gdf(vec_in_file, vec_lyr=vec_in_lyr)
    if get_gdf_geom_type(in_gdf) != rsgispy.GEOM_PT:
        raise rsgispy.RSGISPyException("Vector geometry type must be points.")

    valid_mask = (
        in_gdf.geometry.notna() & ~in_gdf.geometry.is_empty
    ).to_numpy()
    counts = np.zeros(len(in_gdf), dtype=int)
    if np.any(valid_mask):
        valid_geoms = in_gdf.geometry.iloc[np.flatnonzero(valid_mask)]
        coords = np.column_stack(
            [
                valid_geoms.x.to_numpy(dtype=float),
                valid_geoms.y.to_numpy(dtype=float),
            ]
        )
        tree = scipy.spatial.KDTree(coords)
        n_pts = tree.query_ball_point(
            coords,
            r=radius,
            p=2.0,
            eps=0,
            workers=n_cores,
            return_sorted=None,
            return_length=True,
        )
        counts[valid_mask] = np.asarray(n_pts, dtype=int) - 1
    in_gdf[out_col_name] = counts
    return finish_vec_output(
        in_gdf,
        out_vec_file,
        out_vec_lyr=out_vec_lyr,
        out_format=out_format,
        out_compress=out_compress,
    )


def count_feats_per_att_val(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    col_name: str,
    out_df_dict: bool = False,
) -> Union[Dict[Any, int], Dict[str, List[Any]]]:
    """
    Count the number of features for each unique value in ``col_name``.

    :param vec_file: Input vector file path or :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param col_name: Column name whose unique values are counted.
    :param out_df_dict: If ``True``, return ``{'vals': [...], 'count': [...]}``
                        suitable for ``pandas.DataFrame``; otherwise return a
                        mapping ``{value: count}`` (Default = ``False``).
    :return: Value-count dictionary.

    .. code:: python

        import rsgispy.vector.stats

        counts = rsgispy.vector.stats.count_feats_per_att_val(
            "classes.gpkg", "classes", "class_name"
        )
    """
    gdf = read_vec_gdf(vec_file, vec_lyr=vec_lyr, copy=False)
    validate_columns_exist(gdf, [col_name])
    counts_series = gdf[col_name].value_counts(dropna=False, sort=False)

    out_dict: Dict[Any, int] = {
        val: int(cnt) for val, cnt in counts_series.items()
    }
    if out_df_dict:
        return {
            "vals": list(out_dict.keys()),
            "count": list(out_dict.values()),
        }
    return out_dict
