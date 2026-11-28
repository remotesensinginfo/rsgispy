"""
The vector.attrs module provides spatial attribute table enrichment
(bounding-box columns, spatial selection annotation, and spatial attribute
joins) using ``geopandas``.
"""

import os
from typing import Optional, Union

import geopandas as gpd
import numpy as np

import rsgispy
from rsgispy.vectorprocess import (
    VectorInput,
    ensure_matching_crs,
    read_vec_gdf,
    write_vec_gdf,
)

__all__ = [
    "pop_bbox_cols",
    "add_geom_bbox_cols",
    "annotate_vec_selection",
    "perform_spatial_join",
]


def _populate_bbox_columns(
    gdf: gpd.GeoDataFrame,
    x_min_col: str,
    x_max_col: str,
    y_min_col: str,
    y_max_col: str,
    fill_empty_zero: bool = False,
) -> gpd.GeoDataFrame:
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
    vec_file: VectorInput,
    vec_lyr: Optional[str] = None,
    x_min_col: str = "xmin",
    x_max_col: str = "xmax",
    y_min_col: str = "ymin",
    y_max_col: str = "ymax",
    *,
    out_format: Optional[str] = None,
    out_compress: Optional[Union[str, bool]] = None,
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Populate geometry bounding-box columns on a vector file or an in-memory
    :class:`geopandas.GeoDataFrame`.

    :param vec_file: Path to the vector file to update (in-place unless
                     ``out_vec_file`` is given) or
                     :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Layer name within ``vec_file`` (or ``None``).
    :param x_min_col: Output MinX column name (Default = ``'xmin'``).
    :param x_max_col: Output MaxX column name (Default = ``'xmax'``).
    :param y_min_col: Output MinY column name (Default = ``'ymin'``).
    :param y_max_col: Output MaxY column name (Default = ``'ymax'``).
    :param out_format: Optional output driver name (inferred from extension
                       when ``None``).
    :param out_compress: Optional GeoParquet compression setting (``None`` to
                         infer from extension or write uncompressed, ``True``
                         for ``'snappy'``, ``False``/``'none'`` for uncompressed,
                         or codec name ``'snappy'``, ``'gzip'``, ``'zstd'``,
                         ``'brotli'``, ``'lz4'``).
    :param out_vec_file: Optional separate output vector file path. When
                         ``vec_file`` is a :class:`geopandas.GeoDataFrame` and
                         ``out_vec_file`` is ``None``, the updated GeoDataFrame
                         is returned in memory.
    :param out_vec_lyr: Optional output vector layer name (defaults to
                        ``vec_lyr``).
    :return: Updated :class:`geopandas.GeoDataFrame` when ``vec_file`` is a
             ``GeoDataFrame`` and ``out_vec_file`` is ``None``, otherwise
             ``None``.

    .. code:: python

        import rsgispy.vector.attrs

        rsgispy.vector.attrs.pop_bbox_cols("polys.gpkg", "polys")
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    is_gdf_input = isinstance(vec_file, gpd.GeoDataFrame)
    gdf = read_vec_gdf(vec_file, vec_lyr)
    _populate_bbox_columns(
        gdf, x_min_col, x_max_col, y_min_col, y_max_col, fill_empty_zero=True
    )
    if out_vec_file is not None:
        write_vec_gdf(
            gdf,
            out_vec_file,
            out_vec_lyr if out_vec_lyr is not None else vec_lyr,
            out_format=out_format,
            out_compress=out_compress,
        )
        return None
    if is_gdf_input:
        return gdf
    write_vec_gdf(
        gdf,
        vec_file,  # type: ignore[arg-type]
        out_vec_lyr if out_vec_lyr is not None else vec_lyr,
        out_format=out_format,
        replace_layer=True,
        out_compress=out_compress,
    )
    return None


def add_geom_bbox_cols(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    *,
    out_format: str = "GPKG",
    min_x_col: str = "MinX",
    max_x_col: str = "MaxX",
    min_y_col: str = "MinY",
    max_y_col: str = "MaxY",
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Add geometry bounding-box columns and write to a new vector file (or return
    an updated :class:`geopandas.GeoDataFrame` when ``out_vec_file`` is
    ``None``).

    :param vec_file: Input vector file path or :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param out_vec_file: Output vector file path, or ``None`` to return an
                         in-memory :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param min_x_col: Output MinX column name (Default = ``'MinX'``).
    :param max_x_col: Output MaxX column name (Default = ``'MaxX'``).
    :param min_y_col: Output MinY column name (Default = ``'MinY'``).
    :param max_y_col: Output MaxY column name (Default = ``'MaxY'``).
    :param out_compress: Optional GeoParquet compression setting (``None`` to
                         infer from extension or write uncompressed, ``True``
                         for ``'snappy'``, ``False``/``'none'`` for uncompressed,
                         or codec name ``'snappy'``, ``'gzip'``, ``'zstd'``,
                         ``'brotli'``, ``'lz4'``).
    :return: Updated :class:`geopandas.GeoDataFrame` when ``out_vec_file`` is
             ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.vector.attrs

        rsgispy.vector.attrs.add_geom_bbox_cols(
            "in.gpkg", "polys", "out_bbox.gpkg", "polys"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    base_gdf = read_vec_gdf(vec_file, vec_lyr)
    _populate_bbox_columns(
        base_gdf, min_x_col, max_x_col, min_y_col, max_y_col, fill_empty_zero=False
    )
    if out_vec_file is None:
        return base_gdf
    write_vec_gdf(
        base_gdf,
        out_vec_file,
        out_vec_lyr,
        out_format=out_format,
        out_compress=out_compress,
    )
    return None


def annotate_vec_selection(
    vec_in_file: VectorInput,
    vec_in_lyr: Optional[str],
    vec_sel_file: VectorInput,
    vec_sel_lyr: Optional[str],
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    out_col_name: str = "sel_feats",
    *,
    out_format: str = "GPKG",
    tmp_col_name: str = "tmp_sel_join_fid",
    vec_in_epsg: Optional[int] = None,
    vec_sel_epsg: Optional[int] = None,
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Annotate features in ``vec_in_file`` with a boolean column indicating
    whether each feature intersects any feature in ``vec_sel_file``. The
    selection layer is reprojected onto the input layer's CRS if they differ.

    :param vec_in_file: Input vector file path or :class:`geopandas.GeoDataFrame`.
    :param vec_in_lyr: Input vector layer name (or ``None``).
    :param vec_sel_file: Selection vector file path or
                         :class:`geopandas.GeoDataFrame`.
    :param vec_sel_lyr: Selection vector layer name (or ``None``).
    :param out_vec_file: Output vector file path, or ``None`` to return an
                         in-memory :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name.
    :param out_col_name: Output boolean column name (Default = ``'sel_feats'``).
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param tmp_col_name: Temporary ID column name
                         (Default = ``'tmp_sel_join_fid'``).
    :param vec_in_epsg: Optional EPSG code override for ``vec_in_file``.
    :param vec_sel_epsg: Optional EPSG code override for ``vec_sel_file``.
    :param out_compress: Optional GeoParquet compression setting (``None`` to
                         infer from extension or write uncompressed, ``True``
                         for ``'snappy'``, ``False``/``'none'`` for uncompressed,
                         or codec name ``'snappy'``, ``'gzip'``, ``'zstd'``,
                         ``'brotli'``, ``'lz4'``).
    :return: Annotated :class:`geopandas.GeoDataFrame` when ``out_vec_file``
             is ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.vector.attrs

        rsgispy.vector.attrs.annotate_vec_selection(
            "in.gpkg", "polys", "roi.gpkg", "roi", "out.gpkg", "polys"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    in_gdf = read_vec_gdf(vec_in_file, vec_in_lyr, epsg=vec_in_epsg)
    sel_gdf = ensure_matching_crs(
        in_gdf, read_vec_gdf(vec_sel_file, vec_sel_lyr, epsg=vec_sel_epsg)
    )

    in_gdf[tmp_col_name] = np.arange(1, len(in_gdf) + 1, 1, dtype=int)
    in_sel_gdf = gpd.sjoin(
        in_gdf, sel_gdf, how="inner", predicate="intersects"
    ).drop_duplicates(subset=[tmp_col_name])

    in_gdf[out_col_name] = in_gdf[tmp_col_name].isin(in_sel_gdf[tmp_col_name])
    in_gdf = in_gdf.drop(columns=[tmp_col_name])
    if out_vec_file is None:
        return in_gdf
    write_vec_gdf(
        in_gdf,
        out_vec_file,
        out_vec_lyr,
        out_format=out_format,
        out_compress=out_compress,
    )
    return None


def perform_spatial_join(
    vec_base_file: VectorInput,
    vec_base_lyr: Optional[str],
    vec_join_file: VectorInput,
    vec_join_lyr: Optional[str],
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    *,
    out_format: str = "GPKG",
    join_how: str = "inner",
    join_op: str = "within",
    vec_base_epsg: Optional[int] = None,
    vec_join_epsg: Optional[int] = None,
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Perform a spatial attribute join between two vector layers using
    :func:`geopandas.sjoin`. The join layer is reprojected onto the base
    layer's CRS if they differ.

    :param vec_base_file: Base vector file or :class:`geopandas.GeoDataFrame`
                          providing output geometries.
    :param vec_base_lyr: Base vector layer name (or ``None``).
    :param vec_join_file: Vector file or :class:`geopandas.GeoDataFrame`
                          providing attributes to join.
    :param vec_join_lyr: Join vector layer name (or ``None``).
    :param out_vec_file: Output vector file path, or ``None`` to return an
                         in-memory :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param join_how: Join type (``'left'``, ``'right'``, or ``'inner'``;
                     Default = ``'inner'``).
    :param join_op: Spatial predicate (``'intersects'``, ``'within'``, or
                    ``'contains'``; Default = ``'within'``).
    :param vec_base_epsg: Optional EPSG code override for ``vec_base_file``.
    :param vec_join_epsg: Optional EPSG code override for ``vec_join_file``.
    :param out_compress: Optional GeoParquet compression setting (``None`` to
                         infer from extension or write uncompressed, ``True``
                         for ``'snappy'``, ``False``/``'none'`` for uncompressed,
                         or codec name ``'snappy'``, ``'gzip'``, ``'zstd'``,
                         ``'brotli'``, ``'lz4'``).
    :return: Joined :class:`geopandas.GeoDataFrame` when ``out_vec_file`` is
             ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.vector.attrs

        rsgispy.vector.attrs.perform_spatial_join(
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
    join_gdf = ensure_matching_crs(
        base_gdf, read_vec_gdf(vec_join_file, vec_join_lyr, epsg=vec_join_epsg)
    )

    joined_gdf = gpd.sjoin(
        base_gdf, join_gdf, how=join_how, predicate=join_op
    )
    if out_vec_file is None:
        return joined_gdf
    if len(joined_gdf) > 0:
        write_vec_gdf(
            joined_gdf,
            out_vec_file,
            out_vec_lyr,
            out_format=out_format,
            out_compress=out_compress,
        )
    return None
