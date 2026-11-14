"""
The vectorattrs.spatial module provides spatial attribute table enrichment
(bounding-box columns, spatial selection annotation, and spatial attribute
joins) using ``geopandas``.
"""

from typing import Optional

import geopandas
import numpy

import rsgispy
from rsgispy.vectorprocess import read_vec_gdf, write_vec_gdf

__all__ = [
    "pop_bbox_cols",
    "add_geom_bbox_cols",
    "annotate_vec_selection",
    "perform_spatial_join",
]


def _populate_bbox_columns(
    gdf: geopandas.GeoDataFrame,
    x_min_col: str,
    x_max_col: str,
    y_min_col: str,
    y_max_col: str,
    fill_empty_zero: bool = False,
) -> geopandas.GeoDataFrame:
    """Populate bounding-box columns from feature geometries."""
    bounds = gdf.geometry.bounds
    if fill_empty_zero:
        bounds = bounds.fillna(0.0)
    gdf[x_min_col] = bounds["minx"].astype(float)
    gdf[x_max_col] = bounds["maxx"].astype(float)
    gdf[y_min_col] = bounds["miny"].astype(float)
    gdf[y_max_col] = bounds["maxy"].astype(float)
    return gdf


def pop_bbox_cols(
    vec_file: str,
    vec_lyr: Optional[str],
    x_min_col: str = "xmin",
    x_max_col: str = "xmax",
    y_min_col: str = "ymin",
    y_max_col: str = "ymax",
    out_format: Optional[str] = None,
) -> None:
    """
    Populate geometry bounding-box columns in-place on an existing vector file.

    :param vec_file: Path to the vector file to update.
    :param vec_lyr: Layer name within ``vec_file`` (or ``None``).
    :param x_min_col: Output MinX column name (Default = ``'xmin'``).
    :param x_max_col: Output MaxX column name (Default = ``'xmax'``).
    :param y_min_col: Output MinY column name (Default = ``'ymin'``).
    :param y_max_col: Output MaxY column name (Default = ``'ymax'``).
    :param out_format: Optional output driver name (inferred from extension
                       when ``None``).

    .. code:: python

        import rsgispy.vectorattrs.spatial

        rsgispy.vectorattrs.spatial.pop_bbox_cols("polys.gpkg", "polys")
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    gdf = read_vec_gdf(vec_file, vec_lyr)
    _populate_bbox_columns(
        gdf, x_min_col, x_max_col, y_min_col, y_max_col, fill_empty_zero=True
    )
    write_vec_gdf(gdf, vec_file, vec_lyr, out_format=out_format)


def add_geom_bbox_cols(
    vec_file: str,
    vec_lyr: Optional[str],
    out_vec_file: str,
    out_vec_lyr: Optional[str],
    out_format: str = "GPKG",
    min_x_col: str = "MinX",
    max_x_col: str = "MaxX",
    min_y_col: str = "MinY",
    max_y_col: str = "MaxY",
) -> None:
    """
    Add geometry bounding-box columns and write to a new vector file.

    :param vec_file: Input vector file path.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param out_vec_file: Output vector file path.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param min_x_col: Output MinX column name (Default = ``'MinX'``).
    :param max_x_col: Output MaxX column name (Default = ``'MaxX'``).
    :param min_y_col: Output MinY column name (Default = ``'MinY'``).
    :param max_y_col: Output MaxY column name (Default = ``'MaxY'``).

    .. code:: python

        import rsgispy.vectorattrs.spatial

        rsgispy.vectorattrs.spatial.add_geom_bbox_cols(
            "in.gpkg", "polys", "out_bbox.gpkg", "polys"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    base_gdf = read_vec_gdf(vec_file, vec_lyr)
    _populate_bbox_columns(
        base_gdf, min_x_col, max_x_col, min_y_col, max_y_col, fill_empty_zero=False
    )
    write_vec_gdf(base_gdf, out_vec_file, out_vec_lyr, out_format=out_format)


def annotate_vec_selection(
    vec_in_file: str,
    vec_in_lyr: Optional[str],
    vec_sel_file: str,
    vec_sel_lyr: Optional[str],
    out_vec_file: str,
    out_vec_lyr: Optional[str],
    out_col_name: str = "sel_feats",
    out_format: str = "GPKG",
    tmp_col_name: str = "tmp_sel_join_fid",
    vec_in_epsg: Optional[int] = None,
    vec_sel_epsg: Optional[int] = None,
) -> None:
    """
    Annotate features in ``vec_in_file`` with a boolean column indicating
    whether each feature intersects any feature in ``vec_sel_file``.

    :param vec_in_file: Input vector file path.
    :param vec_in_lyr: Input vector layer name (or ``None``).
    :param vec_sel_file: Selection vector file path.
    :param vec_sel_lyr: Selection vector layer name (or ``None``).
    :param out_vec_file: Output vector file path.
    :param out_vec_lyr: Output vector layer name.
    :param out_col_name: Output boolean column name (Default = ``'sel_feats'``).
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param tmp_col_name: Temporary ID column name
                         (Default = ``'tmp_sel_join_fid'``).
    :param vec_in_epsg: Optional EPSG code override for ``vec_in_file``.
    :param vec_sel_epsg: Optional EPSG code override for ``vec_sel_file``.

    .. code:: python

        import rsgispy.vectorattrs.spatial

        rsgispy.vectorattrs.spatial.annotate_vec_selection(
            "in.gpkg", "polys", "roi.gpkg", "roi", "out.gpkg", "polys"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    in_gdf = read_vec_gdf(vec_in_file, vec_in_lyr, epsg=vec_in_epsg)
    sel_gdf = read_vec_gdf(vec_sel_file, vec_sel_lyr, epsg=vec_sel_epsg)

    in_gdf[tmp_col_name] = numpy.arange(1, len(in_gdf) + 1, 1, dtype=int)
    in_sel_gdf = geopandas.sjoin(
        in_gdf, sel_gdf, how="inner", predicate="intersects"
    ).drop_duplicates(subset=[tmp_col_name])

    in_gdf[out_col_name] = in_gdf[tmp_col_name].isin(in_sel_gdf[tmp_col_name])
    in_gdf = in_gdf.drop(columns=[tmp_col_name])
    write_vec_gdf(in_gdf, out_vec_file, out_vec_lyr, out_format=out_format)


def perform_spatial_join(
    vec_base_file: str,
    vec_base_lyr: Optional[str],
    vec_join_file: str,
    vec_join_lyr: Optional[str],
    out_vec_file: str,
    out_vec_lyr: Optional[str],
    out_format: str = "GPKG",
    join_how: str = "inner",
    join_op: str = "within",
    vec_base_epsg: Optional[int] = None,
    vec_join_epsg: Optional[int] = None,
) -> None:
    """
    Perform a spatial attribute join between two vector layers using
    :func:`geopandas.sjoin`.

    :param vec_base_file: Base vector file providing output geometries.
    :param vec_base_lyr: Base vector layer name (or ``None``).
    :param vec_join_file: Vector file providing attributes to join.
    :param vec_join_lyr: Join vector layer name (or ``None``).
    :param out_vec_file: Output vector file path.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param join_how: Join type (``'left'``, ``'right'``, or ``'inner'``;
                     Default = ``'inner'``).
    :param join_op: Spatial predicate (``'intersects'``, ``'within'``, or
                    ``'contains'``; Default = ``'within'``).
    :param vec_base_epsg: Optional EPSG code override for ``vec_base_file``.
    :param vec_join_epsg: Optional EPSG code override for ``vec_join_file``.

    .. code:: python

        import rsgispy.vectorattrs.spatial

        rsgispy.vectorattrs.spatial.perform_spatial_join(
            "pts.gpkg", "pts", "polys.gpkg", "polys", "joined.gpkg", "pts"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    if join_how not in ("left", "right", "inner"):
        raise rsgispy.RSGISPyException(
            f"The join_how ('{join_how}') specified is not valid."
        )
    if join_op not in ("intersects", "within", "contains"):
        raise rsgispy.RSGISPyException(
            f"The join_op ('{join_op}') specified is not valid."
        )

    base_gdf = read_vec_gdf(vec_base_file, vec_base_lyr, epsg=vec_base_epsg)
    join_gdf = read_vec_gdf(vec_join_file, vec_join_lyr, epsg=vec_join_epsg)

    joined_gdf = geopandas.sjoin(
        base_gdf, join_gdf, how=join_how, predicate=join_op
    )
    if len(joined_gdf) > 0:
        write_vec_gdf(joined_gdf, out_vec_file, out_vec_lyr, out_format=out_format)
