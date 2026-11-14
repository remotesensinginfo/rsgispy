"""
Geometric measurements, spatial counts, and attribute summary statistics for
:mod:`rsgispy.vectorstats` using ``geopandas``, ``shapely``, ``numpy``, and
``scipy``.
"""

from typing import Any, Dict, List, Optional, Union

import geopandas
import numpy
import scipy.spatial
import shapely.geometry

import rsgispy
from rsgispy.vectorprocess import (
    get_gdf_geom_type,
    read_vec_gdf,
    validate_columns_exist,
    write_vec_gdf,
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
    vec_file: str,
    vec_lyr: Optional[str],
    out_col: str,
    out_vec_file: str,
    out_vec_lyr: Optional[str],
    out_format: str = "GPKG",
) -> None:
    """
    Calculate the area of each polygon feature and store it in ``out_col``.

    :param vec_file: Input vector file path (polygon geometry expected).
    :param vec_lyr: Input vector layer name (or ``None``).
    :param out_col: Output area column name.
    :param out_vec_file: Output vector file path.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).

    .. code:: python

        import rsgispy.vectorstats.metrics

        rsgispy.vectorstats.metrics.calc_vec_area(
            "polys.gpkg", "polys", "area_m2", "polys_area.gpkg", "polys"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    data_gdf = read_vec_gdf(vec_file, vec_lyr)
    data_gdf[out_col] = data_gdf.geometry.area
    write_vec_gdf(data_gdf, out_vec_file, out_vec_lyr, out_format=out_format)


def calc_vec_length(
    vec_file: str,
    vec_lyr: Optional[str],
    out_col: str,
    out_vec_file: str,
    out_vec_lyr: Optional[str],
    out_format: str = "GPKG",
) -> None:
    """
    Calculate the length (or perimeter) of each line/polygon feature and store
    it in ``out_col``.

    :param vec_file: Input vector file path (line or polygon geometry expected).
    :param vec_lyr: Input vector layer name (or ``None``).
    :param out_col: Output length column name.
    :param out_vec_file: Output vector file path.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).

    .. code:: python

        import rsgispy.vectorstats.metrics

        rsgispy.vectorstats.metrics.calc_vec_length(
            "lines.gpkg", "lines", "len_m", "lines_len.gpkg", "lines"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    data_gdf = read_vec_gdf(vec_file, vec_lyr)
    data_gdf[out_col] = data_gdf.geometry.length
    write_vec_gdf(data_gdf, out_vec_file, out_vec_lyr, out_format=out_format)


def get_poly_hole_area(
    vec_file: str,
    vec_lyr: Optional[str],
    use_pbar: bool = True,
) -> List[float]:
    """
    Calculate the area of every interior hole across all polygons in a vector
    layer.

    :param vec_file: Input polygon vector file path.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param use_pbar: Show progress bar (Default = ``True``).
    :return: List of hole areas in CRS units squared.

    .. code:: python

        import rsgispy.vectorstats.metrics

        hole_areas = rsgispy.vectorstats.metrics.get_poly_hole_area(
            "polys_holes.gpkg", "polys"
        )
    """
    in_gdf = read_vec_gdf(vec_file, vec_lyr)
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
    vec_file: str,
    vec_lyr: Optional[str],
    out_vec_file: str,
    out_vec_lyr: Optional[str],
    out_format: str = "GPKG",
    x_centre: Optional[float] = None,
    y_centre: Optional[float] = None,
    angle_col: str = "angle",
    dist_col: str = "dist",
) -> None:
    """
    Calculate Euclidean distance and clockwise-from-north azimuth angle
    (0-360 degrees) for each point relative to ``(x_centre, y_centre)``.

    If ``x_centre`` or ``y_centre`` is ``None``, the mean of the point
    coordinates is used.

    :param vec_file: Input point vector file path.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param out_vec_file: Output vector file path.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param x_centre: Optional reference X coordinate.
    :param y_centre: Optional reference Y coordinate.
    :param angle_col: Output azimuth column name (Default = ``'angle'``).
    :param dist_col: Output distance column name (Default = ``'dist'``).

    .. code:: python

        import rsgispy.vectorstats.metrics

        rsgispy.vectorstats.metrics.calc_vec_pt_dist_angle(
            "pts.gpkg", "pts", "pts_ang.gpkg", "pts"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    data_gdf = read_vec_gdf(vec_file, vec_lyr)
    if get_gdf_geom_type(data_gdf) != rsgispy.GEOM_PT:
        raise rsgispy.RSGISPyException("Vector geometry type must be points.")

    geom_pts = data_gdf.geometry.get_coordinates()
    if not {"x", "y"}.issubset(geom_pts.columns):
        raise rsgispy.RSGISPyException(
            "The geometry is not as expected - need x and y coordinates."
        )

    x_coords = geom_pts["x"].to_numpy()
    y_coords = geom_pts["y"].to_numpy()
    if x_centre is None:
        x_centre = float(x_coords.mean())
    if y_centre is None:
        y_centre = float(y_coords.mean())

    data_gdf[dist_col] = numpy.sqrt(
        (x_coords - x_centre) ** 2 + (y_coords - y_centre) ** 2
    )

    angles = numpy.rad2deg(
        numpy.arctan2(y_coords - y_centre, x_coords - x_centre)
    )
    angles_secs = numpy.zeros_like(angles)
    angles_secs[(angles >= 0) & (angles <= 90)] = 1
    angles_secs[angles > 90] = 2
    angles_secs[angles < 0] = 3

    angles[angles_secs == 1] = -(angles[angles_secs == 1] - 90.0)
    angles[angles_secs == 2] = -(angles[angles_secs == 2] - 180.0) + 270.0
    angles[angles_secs == 3] = -angles[angles_secs == 3] + 90.0

    data_gdf[angle_col] = angles
    write_vec_gdf(data_gdf, out_vec_file, out_vec_lyr, out_format=out_format)


def calc_near_dist_to_feats(
    vec_file: str,
    vec_lyr: Optional[str],
    vec_dist_file: str,
    vec_dist_lyr: Optional[str],
    out_vec_file: str,
    out_vec_lyr: Optional[str],
    out_col: str = "dist",
    out_format: str = "GPKG",
) -> None:
    """
    Compute nearest-feature distance from each feature in ``vec_file`` to
    features in ``vec_dist_file`` using :func:`geopandas.sjoin_nearest`.

    :param vec_file: Input vector file path.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param vec_dist_file: Target vector file path to measure distance to.
    :param vec_dist_lyr: Target vector layer name (or ``None``).
    :param out_vec_file: Output vector file path.
    :param out_vec_lyr: Output vector layer name.
    :param out_col: Output distance column name (Default = ``'dist'``).
    :param out_format: Output vector format driver (Default = ``'GPKG'``).

    .. code:: python

        import rsgispy.vectorstats.metrics

        rsgispy.vectorstats.metrics.calc_near_dist_to_feats(
            "pts.gpkg", "pts", "roads.gpkg", "roads", "pts_dist.gpkg", "pts"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    data_gdf = read_vec_gdf(vec_file, vec_lyr)
    dist_gdf = read_vec_gdf(vec_dist_file, vec_dist_lyr)

    data_dist_gdf = geopandas.sjoin_nearest(
        data_gdf,
        dist_gdf,
        distance_col=out_col,
        lsuffix="left",
        rsuffix="right",
        exclusive=True,
    )
    write_vec_gdf(data_dist_gdf, out_vec_file, out_vec_lyr, out_format=out_format)


def count_pt_intersects(
    vec_in_file: str,
    vec_in_lyr: Optional[str],
    vec_pts_file: str,
    vec_pts_lyr: Optional[str],
    out_vec_file: str,
    out_vec_lyr: Optional[str],
    out_format: str = "GPKG",
    out_count_col: str = "n_points",
    tmp_col_name: str = "tmp_join_fid",
    vec_in_epsg: Optional[int] = None,
    vec_pts_epsg: Optional[int] = None,
) -> None:
    """
    Count the number of points from ``vec_pts_file`` intersecting each polygon
    in ``vec_in_file``.

    Specifying EPSG codes overrides CRS definitions without reprojecting.

    :param vec_in_file: Input polygon vector file path.
    :param vec_in_lyr: Input polygon vector layer name (or ``None``).
    :param vec_pts_file: Input point vector file path.
    :param vec_pts_lyr: Input point vector layer name (or ``None``).
    :param out_vec_file: Output vector file path.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param out_count_col: Output count column name (Default = ``'n_points'``).
    :param tmp_col_name: Temporary join ID column name (Default = ``'tmp_join_fid'``).
    :param vec_in_epsg: Optional EPSG code override for ``vec_in_file``.
    :param vec_pts_epsg: Optional EPSG code override for ``vec_pts_file``.

    .. code:: python

        import rsgispy.vectorstats.metrics

        rsgispy.vectorstats.metrics.count_pt_intersects(
            "polys.gpkg", "polys", "pts.gpkg", "pts", "counts.gpkg", "polys"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    in_gdf = read_vec_gdf(vec_in_file, vec_in_lyr, epsg=vec_in_epsg)
    pts_gdf = read_vec_gdf(vec_pts_file, vec_pts_lyr, epsg=vec_pts_epsg)

    drop_cols = [c for c in ("index_left", "index_right") if c in in_gdf.columns]
    if drop_cols:
        in_gdf = in_gdf.drop(columns=drop_cols)

    in_gdf[tmp_col_name] = numpy.arange(1, len(in_gdf) + 1, 1, dtype=int)
    counts = (
        in_gdf.sjoin(pts_gdf)
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

    write_vec_gdf(in_count_gdf, out_vec_file, out_vec_lyr, out_format=out_format)


def calc_npts_in_radius(
    vec_in_file: str,
    vec_in_lyr: Optional[str],
    radius: float,
    out_vec_file: str,
    out_vec_lyr: Optional[str],
    out_format: str = "GPKG",
    out_col_name: str = "n_pts_r",
    n_cores: int = 1,
) -> None:
    """
    Count how many other points lie within ``radius`` of each point feature.

    :param vec_in_file: Input point vector file path.
    :param vec_in_lyr: Input point vector layer name (or ``None``).
    :param radius: Search radius in layer CRS units.
    :param out_vec_file: Output vector file path.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param out_col_name: Output neighbour count column (Default = ``'n_pts_r'``).
    :param n_cores: Number of worker threads for KD-tree queries (``-1`` uses
                    all cores; Default = ``1``).

    .. code:: python

        import rsgispy.vectorstats.metrics

        rsgispy.vectorstats.metrics.calc_npts_in_radius(
            "pts.gpkg", "pts", 50.0, "pts_density.gpkg", "pts"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    in_gdf = read_vec_gdf(vec_in_file, vec_in_lyr)
    if get_gdf_geom_type(in_gdf) != rsgispy.GEOM_PT:
        raise rsgispy.RSGISPyException("Vector geometry type must be points.")

    coords = list(zip(in_gdf.geometry.x, in_gdf.geometry.y))
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
    in_gdf[out_col_name] = n_pts - 1
    write_vec_gdf(in_gdf, out_vec_file, out_vec_lyr, out_format=out_format)


def count_feats_per_att_val(
    vec_file: str,
    vec_lyr: Optional[str],
    col_name: str,
    out_df_dict: bool = False,
) -> Union[Dict[Any, int], Dict[str, List[Any]]]:
    """
    Count the number of features for each unique value in ``col_name``.

    :param vec_file: Input vector file path.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param col_name: Column name whose unique values are counted.
    :param out_df_dict: If ``True``, return ``{'vals': [...], 'count': [...]}``
                        suitable for ``pandas.DataFrame``; otherwise return a
                        mapping ``{value: count}`` (Default = ``False``).
    :return: Value-count dictionary.

    .. code:: python

        import rsgispy.vectorstats.metrics

        counts = rsgispy.vectorstats.metrics.count_feats_per_att_val(
            "classes.gpkg", "classes", "class_name"
        )
    """
    gdf = read_vec_gdf(vec_file, vec_lyr=vec_lyr)
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
