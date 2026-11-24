"""
The vector.geoms module provides geometric transformations and polygon
utilities (boundary extraction, centroids, exploding multi-geometries,
simplification, hole removal/inspection, buffering, grid splitting, area
filtering, dissolving, thinning, and coordinate shifting) using ``geopandas``,
``pyogrio``, and ``shapely``.
"""

import os
from typing import List, Optional, Sequence, Union

import geopandas as gpd
import numpy as np
import shapely.geometry

import rsgispy
import rsgispy.tools.files
from rsgispy.vectorprocess import (
    VectorInput,
    is_geoparquet_file,
    is_geoparquet_format,
    list_vec_layers,
    prepare_optional_out_vec,
    read_vec_gdf,
    resolve_out_lyr_name,
    validate_columns_exist,
    write_vec_gdf,
)

__all__ = [
    "convert_polys_to_lines_gp",
    "calc_poly_centroids",
    "explode_vec_lyr",
    "explode_vec_files",
    "simplify_geometries",
    "delete_polygon_holes",
    "buffer_vec_layer_gp",
    "split_vec_by_grid",
    "rm_polys_area",
    "vec_lyr_dissolve",
    "thin_vector_geoms",
    "shiftxy_vec_lyr",
]


def convert_polys_to_lines_gp(
    vec_poly_file: VectorInput,
    vec_poly_lyr: Optional[str],
    vec_line_file: Optional[Union[str, os.PathLike]] = None,
    vec_line_lyr: Optional[str] = None,
    *,
    out_format: str = "GPKG",
    del_exist_vec: bool = False,
    exterior_lines: bool = False,
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Convert a polygon vector layer to a line vector layer using ``geopandas``.

    :param vec_poly_file: Input polygon vector file path or
                          :class:`geopandas.GeoDataFrame`.
    :param vec_poly_lyr: Input vector layer name (or ``None``).
    :param vec_line_file: Output line vector file path, or ``None`` to return an
                          in-memory :class:`geopandas.GeoDataFrame`.
    :param vec_line_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param del_exist_vec: If ``True``, overwrite existing ``vec_line_file``
                          (Default = ``False``).
    :param exterior_lines: If ``True``, extract only outer exterior rings as
                           lines; otherwise include interior hole boundaries
                           (Default = ``False``).
    :param out_compress: Optional GeoParquet compression setting (``None`` to
                         infer from extension or write uncompressed, ``True``
                         for ``'snappy'``, ``False``/``'none'`` for uncompressed,
                         or codec name ``'snappy'``, ``'gzip'``, ``'zstd'``,
                         ``'brotli'``, ``'lz4'``).
    :return: Line :class:`geopandas.GeoDataFrame` when ``vec_line_file`` is
             ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.vector.geoms

        rsgispy.vector.geoms.convert_polys_to_lines_gp(
            "polys.gpkg", "polys", "lines.gpkg", "lines", exterior_lines=True
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    out_path = prepare_optional_out_vec(vec_line_file, del_exist_vec=del_exist_vec)
    out_lyr = resolve_out_lyr_name(out_path, vec_line_lyr)

    data_gdf = read_vec_gdf(vec_poly_file, vec_poly_lyr, copy=False)
    out_gdf = data_gdf.copy()
    if exterior_lines:
        ext_geoms = []
        for geom in data_gdf.geometry:
            if geom is None or geom.is_empty:
                ext_geoms.append(geom)
            elif geom.geom_type == "Polygon":
                ext_geoms.append(shapely.geometry.LineString(geom.exterior.coords))
            elif geom.geom_type == "MultiPolygon":
                ext_geoms.append(
                    shapely.geometry.MultiLineString(
                        [list(poly.exterior.coords) for poly in geom.geoms]
                    )
                )
            else:
                raise rsgispy.RSGISPyException(
                    f"Expected Polygon or MultiPolygon geometry, got '{geom.geom_type}'."
                )
        out_gdf = out_gdf.set_geometry(ext_geoms)
    else:
        out_gdf = out_gdf.set_geometry(data_gdf.geometry.boundary)

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


def calc_poly_centroids(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    out_format: str = "GPKG",
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    *,
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Create a point vector layer containing the centroid of each polygon feature.

    :param vec_file: Input vector file path or :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param out_vec_file: Output centroid vector file path, or ``None`` to return
                         an in-memory :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name.
    :param out_compress: Optional GeoParquet compression setting (``None`` to
                         infer from extension or write uncompressed, ``True``
                         for ``'snappy'``, ``False``/``'none'`` for uncompressed,
                         or codec name ``'snappy'``, ``'gzip'``, ``'zstd'``,
                         ``'brotli'``, ``'lz4'``).
    :return: Centroid :class:`geopandas.GeoDataFrame` when ``out_vec_file`` is
             ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.vector.geoms

        rsgispy.vector.geoms.calc_poly_centroids(
            "polys.gpkg", "polys", "GPKG", "centroids.gpkg", "centroids"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    in_gdf = read_vec_gdf(vec_file, vec_lyr, copy=False)
    out_gdf = in_gdf.copy()
    out_gdf = out_gdf.set_geometry(in_gdf.geometry.centroid)
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


def explode_vec_lyr(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    *,
    out_format: str = "GPKG",
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Explode multi-part geometries (e.g., ``MultiPolygon``,
    ``MultiLineString``, ``MultiPoint``) into single-part features using
    ``geopandas``.

    :param vec_file: Input vector file path or :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param out_vec_file: Output vector file path, or ``None`` to return an
                         in-memory :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name (can be ``None`` for single-layer
                        formats).
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param out_compress: Optional GeoParquet compression setting (``None`` to
                         infer from extension or write uncompressed, ``True``
                         for ``'snappy'``, ``False``/``'none'`` for uncompressed,
                         or codec name ``'snappy'``, ``'gzip'``, ``'zstd'``,
                         ``'brotli'``, ``'lz4'``).
    :return: Exploded :class:`geopandas.GeoDataFrame` when ``out_vec_file`` is
             ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.vector.geoms

        rsgispy.vector.geoms.explode_vec_lyr(
            "multi.gpkg", "polys", "single.gpkg", "polys"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    data_gdf = read_vec_gdf(vec_file, vec_lyr, copy=False)
    data_explode_gdf = data_gdf.explode(index_parts=False)
    if out_vec_file is None:
        return data_explode_gdf
    if len(data_explode_gdf) > 0:
        write_vec_gdf(
            data_explode_gdf,
            out_vec_file,
            out_vec_lyr,
            out_format=out_format,
            out_compress=out_compress,
        )
    return None


def explode_vec_files(
    in_vec_files: Sequence[VectorInput],
    output_dir: Optional[Union[str, os.PathLike]] = None,
    *,
    out_format: str = "GPKG",
    out_vec_ext: str = "gpkg",
    use_pbar: bool = True,
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[List[gpd.GeoDataFrame]]:
    """
    Explode multi-part geometries across every layer in each input file or
    :class:`geopandas.GeoDataFrame` in ``in_vec_files``.

    When ``output_dir`` is provided, exploded layers are written to files in
    ``output_dir``; when ``output_dir`` is ``None``, a list of exploded
    :class:`geopandas.GeoDataFrame` objects is returned in memory.

    :param in_vec_files: Sequence of input vector file paths or
                         :class:`geopandas.GeoDataFrame` objects.
    :param output_dir: Optional output directory (must differ from input
                       directory to avoid overwriting inputs). If ``None``,
                       returns a list of exploded
                       :class:`geopandas.GeoDataFrame` objects.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param out_vec_ext: Output file extension without leading dot
                        (Default = ``'gpkg'``).
    :param use_pbar: Show progress bar (Default = ``True``).
    :param out_compress: Optional GeoParquet compression setting (``None`` to
                         infer from extension or write uncompressed, ``True``
                         for ``'snappy'``, ``False``/``'none'`` for uncompressed,
                         or codec name ``'snappy'``, ``'gzip'``, ``'zstd'``,
                         ``'brotli'``, ``'lz4'``).
    :return: List of exploded :class:`geopandas.GeoDataFrame` objects when
             ``output_dir`` is ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.vector.geoms

        rsgispy.vector.geoms.explode_vec_files(
            ["a.gpkg", "b.gpkg"], output_dir="exploded"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    out_dir: Optional[str] = None
    if output_dir is not None:
        out_dir = os.fspath(output_dir)
        os.makedirs(out_dir, exist_ok=True)
    out_gdfs: List[gpd.GeoDataFrame] = []
    pbar_cls = rsgispy.get_tqdm()
    for idx, vec_item in enumerate(
        pbar_cls(in_vec_files, total=len(in_vec_files), disable=not use_pbar)
    ):
        if isinstance(vec_item, gpd.GeoDataFrame):
            if out_dir is None:
                exploded = explode_vec_lyr(vec_item, None, None, None)
                if exploded is not None:
                    out_gdfs.append(exploded)
            else:
                out_vec_file = os.path.join(
                    out_dir, f"layer_{idx}.{out_vec_ext}"
                )
                explode_vec_lyr(
                    vec_item,
                    None,
                    out_vec_file,
                    f"layer_{idx}",
                    out_format=out_format,
                    out_compress=out_compress,
                )
            continue

        vec_path = os.fspath(vec_item)
        lyrs = list_vec_layers(vec_path)
        if out_dir is None:
            for lyr in lyrs:
                exploded = explode_vec_lyr(vec_path, lyr, None, None)
                if exploded is not None:
                    out_gdfs.append(exploded)
            continue

        basename = rsgispy.tools.files.get_file_basename(vec_path)
        out_vec_file = os.path.join(out_dir, f"{basename}.{out_vec_ext}")
        if (
            is_geoparquet_format(out_format) or is_geoparquet_file(out_vec_file)
        ) and (len(lyrs) > 1):
            raise rsgispy.RSGISPyException(
                "GeoParquet does not support multiple layers in a single file."
            )
        for lyr in lyrs:
            explode_vec_lyr(
                vec_path,
                lyr,
                out_vec_file,
                lyr,
                out_format=out_format,
                out_compress=out_compress,
            )
    if out_dir is None:
        return out_gdfs
    return None


def simplify_geometries(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    tolerance: float,
    out_format: str = "GPKG",
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    *,
    del_exist_vec: bool = False,
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Simplify vector geometries using the Douglas-Peucker algorithm with
    topology preservation.

    :param vec_file: Input vector file path or :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param tolerance: Simplification tolerance in CRS units.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param out_vec_file: Output vector file path, or ``None`` to return an
                         in-memory :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name.
    :param del_exist_vec: If ``True``, overwrite existing ``out_vec_file``
                          (Default = ``False``).
    :param out_compress: Optional GeoParquet compression setting (``None`` to
                         infer from extension or write uncompressed, ``True``
                         for ``'snappy'``, ``False``/``'none'`` for uncompressed,
                         or codec name ``'snappy'``, ``'gzip'``, ``'zstd'``,
                         ``'brotli'``, ``'lz4'``).
    :return: Simplified :class:`geopandas.GeoDataFrame` when ``out_vec_file`` is
             ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.vector.geoms

        rsgispy.vector.geoms.simplify_geometries(
            "polys.gpkg", "polys", 2.0, "GPKG", "simple.gpkg", "polys"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    out_path = prepare_optional_out_vec(out_vec_file, del_exist_vec=del_exist_vec)
    in_gdf = read_vec_gdf(vec_file, vec_lyr, copy=False)
    out_gdf = in_gdf.copy()
    out_gdf = out_gdf.set_geometry(
        in_gdf.geometry.simplify(tolerance, preserve_topology=True)
    )
    if out_path is None:
        return out_gdf
    write_vec_gdf(
        out_gdf,
        out_path,
        out_vec_lyr,
        out_format=out_format,
        out_compress=out_compress,
    )
    return None


def _remove_holes_from_polygon(
    polygon: shapely.geometry.Polygon,
    area_thres: Optional[float] = None,
) -> shapely.geometry.Polygon:
    """Remove all holes (or holes <= ``area_thres``) from a Shapely Polygon."""
    if not polygon.interiors:
        return polygon
    if area_thres is None:
        return shapely.geometry.Polygon(polygon.exterior)
    kept_interiors = [
        ring
        for ring in polygon.interiors
        if shapely.geometry.Polygon(ring).area > area_thres
    ]
    return shapely.geometry.Polygon(polygon.exterior, holes=kept_interiors)


def delete_polygon_holes(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    out_format: str = "GPKG",
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    *,
    area_thres: Optional[float] = None,
    del_exist_vec: bool = False,
    use_pbar: bool = True,
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Delete interior holes from input polygons whose area is below or equal to
    ``area_thres`` (or all holes if ``area_thres`` is ``None``).

    :param vec_file: Input polygon vector file path or
                     :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param out_vec_file: Output vector file path, or ``None`` to return an
                         in-memory :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name.
    :param area_thres: Area threshold in CRS units squared; if ``None``, all
                       holes are removed.
    :param del_exist_vec: If ``True``, overwrite existing ``out_vec_file``
                          (Default = ``False``).
    :param use_pbar: Show progress bar (Default = ``True``).
    :param out_compress: Optional GeoParquet compression setting (``None`` to
                         infer from extension or write uncompressed, ``True``
                         for ``'snappy'``, ``False``/``'none'`` for uncompressed,
                         or codec name ``'snappy'``, ``'gzip'``, ``'zstd'``,
                         ``'brotli'``, ``'lz4'``).
    :return: Cleaned :class:`geopandas.GeoDataFrame` when ``out_vec_file`` is
             ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.vector.geoms

        rsgispy.vector.geoms.delete_polygon_holes(
            "polys_holes.gpkg", "polys", "GPKG", "polys_clean.gpkg", "polys",
            area_thres=100.0
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    out_path = prepare_optional_out_vec(out_vec_file, del_exist_vec=del_exist_vec)
    in_gdf = read_vec_gdf(vec_file, vec_lyr, copy=False)
    pbar_cls = rsgispy.get_tqdm()
    out_geoms = []

    for geom in pbar_cls(
        in_gdf.geometry, total=len(in_gdf), disable=not use_pbar
    ):
        if geom is None or geom.is_empty:
            out_geoms.append(geom)
        elif geom.geom_type == "Polygon":
            out_geoms.append(_remove_holes_from_polygon(geom, area_thres))
        elif geom.geom_type == "MultiPolygon":
            out_geoms.append(
                shapely.geometry.MultiPolygon(
                    [_remove_holes_from_polygon(g, area_thres) for g in geom.geoms]
                )
            )
        else:
            raise rsgispy.RSGISPyException(
                "Can only remove holes from polygon geometry."
            )

    out_gdf = in_gdf.copy()
    out_gdf = out_gdf.set_geometry(out_geoms)
    if out_path is None:
        return out_gdf
    write_vec_gdf(
        out_gdf,
        out_path,
        out_vec_lyr,
        out_format=out_format,
        out_compress=out_compress,
    )
    return None


def buffer_vec_layer_gp(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    buf_dist: float,
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    *,
    out_format: str = "GPKG",
    del_exist_vec: bool = False,
    buf_res: int = 16,
    buf_sgl_sided: bool = False,
    cap_style: int = 1,
    join_style: int = 1,
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Buffer the geometries in a vector layer using ``geopandas``.

    :param vec_file: Input vector file path or :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param buf_dist: Buffer distance in CRS units (negative values shrink
                     polygons or indicate the right-hand side when
                     ``buf_sgl_sided=True``).
    :param out_vec_file: Output vector file path, or ``None`` to return an
                         in-memory :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param del_exist_vec: If ``True``, overwrite existing ``out_vec_file``
                          (Default = ``False``).
    :param buf_res: Number of segments used to approximate a quarter circle
                    (Default = ``16``).
    :param buf_sgl_sided: Buffer only a single side when ``True``
                          (Default = ``False``).
    :param cap_style: End cap style: ``1`` (round), ``2`` (flat), ``3`` (square)
                      (Default = ``1``).
    :param join_style: Join style: ``1`` (round), ``2`` (mitre), ``3`` (bevel)
                       (Default = ``1``).
    :param out_compress: Optional GeoParquet compression setting (``None`` to
                         infer from extension or write uncompressed, ``True``
                         for ``'snappy'``, ``False``/``'none'`` for uncompressed,
                         or codec name ``'snappy'``, ``'gzip'``, ``'zstd'``,
                         ``'brotli'``, ``'lz4'``).
    :return: Buffered :class:`geopandas.GeoDataFrame` when ``out_vec_file`` is
             ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.vector.geoms

        rsgispy.vector.geoms.buffer_vec_layer_gp(
            "pts.gpkg", "pts", 10.0, "pts_buf.gpkg", "pts_buf"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    if cap_style not in (1, 2, 3):
        raise rsgispy.RSGISPyException(
            "cap_style must be one of round = 1, flat = 2 or square = 3"
        )
    if join_style not in (1, 2, 3):
        raise rsgispy.RSGISPyException(
            "join_style must be one of round = 1, mitre = 2 or bevel = 3"
        )

    out_path = prepare_optional_out_vec(out_vec_file, del_exist_vec=del_exist_vec)
    out_lyr = resolve_out_lyr_name(out_path, out_vec_lyr)

    data_gdf = read_vec_gdf(vec_file, vec_lyr)
    data_gdf = data_gdf.set_geometry(
        data_gdf.geometry.buffer(
            distance=buf_dist,
            resolution=buf_res,
            single_sided=buf_sgl_sided,
            cap_style=cap_style,
            join_style=join_style,
        )
    )
    if out_path is None:
        return data_gdf
    write_vec_gdf(
        data_gdf,
        out_path,
        out_lyr,
        out_format=out_format,
        out_compress=out_compress,
    )
    return None


def split_vec_by_grid(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    x_grid_size: float,
    y_grid_size: float,
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    *,
    out_format: str = "GPKG",
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Split an input vector layer along a regular grid of spacing
    ``(x_grid_size, y_grid_size)`` covering its bounding extent.

    :param vec_file: Input vector file path or :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param x_grid_size: Grid cell width in CRS units.
    :param y_grid_size: Grid cell height in CRS units.
    :param out_vec_file: Output vector file path, or ``None`` to return an
                         in-memory :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param out_compress: Optional GeoParquet compression setting (``None`` to
                         infer from extension or write uncompressed, ``True``
                         for ``'snappy'``, ``False``/``'none'`` for uncompressed,
                         or codec name ``'snappy'``, ``'gzip'``, ``'zstd'``,
                         ``'brotli'``, ``'lz4'``).
    :return: Split :class:`geopandas.GeoDataFrame` when ``out_vec_file`` is
             ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.vector.geoms

        rsgispy.vector.geoms.split_vec_by_grid(
            "polys.gpkg", "polys", 100.0, 100.0, "split.gpkg", "split"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    in_vec_gdf = read_vec_gdf(vec_file, vec_lyr, copy=False)
    xmin, ymin, xmax, ymax = in_vec_gdf.total_bounds
    tile_bboxes = rsgispy.tools.geometry.get_bbox_grid(
        (float(xmin), float(xmax), float(ymin), float(ymax)),
        x_grid_size,
        y_grid_size,
    )
    polygons = [
        shapely.geometry.box(b[0], b[2], b[1], b[3]) for b in tile_bboxes
    ]

    grid_gdf = gpd.GeoDataFrame(
        {"geometry": polygons}, crs=in_vec_gdf.crs
    )
    vec_split_gdf = in_vec_gdf.overlay(grid_gdf, how="intersection")
    if out_vec_file is None:
        return vec_split_gdf
    write_vec_gdf(
        vec_split_gdf,
        out_vec_file,
        out_vec_lyr,
        out_format=out_format,
        out_compress=out_compress,
    )
    return None


def rm_polys_area(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    area_thres: float,
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    *,
    out_format: str = "GPKG",
    less_than: bool = True,
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Filter polygons by keeping features whose area is either strictly less than
    or strictly greater than ``area_thres``.

    :param vec_file: Input vector file path or :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param area_thres: Area threshold in CRS units squared.
    :param out_vec_file: Output vector file path, or ``None`` to return an
                         in-memory :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param less_than: When ``True`` (default), keep features with
                      ``area < area_thres``; when ``False``, keep features with
                      ``area > area_thres``.
    :param out_compress: Optional GeoParquet compression setting (``None`` to
                         infer from extension or write uncompressed, ``True``
                         for ``'snappy'``, ``False``/``'none'`` for uncompressed,
                         or codec name ``'snappy'``, ``'gzip'``, ``'zstd'``,
                         ``'brotli'``, ``'lz4'``).
    :return: Filtered :class:`geopandas.GeoDataFrame` when ``out_vec_file`` is
             ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.vector.geoms

        rsgispy.vector.geoms.rm_polys_area(
            "polys.gpkg", "polys", 50.0, "large.gpkg", "polys", less_than=False
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    in_vec_gdf = read_vec_gdf(vec_file, vec_lyr, copy=False)
    if less_than:
        area_msk = in_vec_gdf.area < area_thres
    else:
        area_msk = in_vec_gdf.area > area_thres
    vec_sel_gdf = in_vec_gdf.loc[area_msk].copy()

    if out_vec_file is None:
        return vec_sel_gdf
    if len(vec_sel_gdf) > 0:
        write_vec_gdf(
            vec_sel_gdf,
            out_vec_file,
            out_vec_lyr,
            out_format=out_format,
            out_compress=out_compress,
        )
    return None


def vec_lyr_dissolve(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    vec_col: str,
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    *,
    out_format: str = "GPKG",
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Dissolve vector features by grouping on categorical attribute ``vec_col``.

    :param vec_file: Input vector file path or :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param vec_col: Categorical column name to dissolve on.
    :param out_vec_file: Output vector file path, or ``None`` to return an
                         in-memory :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param out_compress: Optional GeoParquet compression setting (``None`` to
                         infer from extension or write uncompressed, ``True``
                         for ``'snappy'``, ``False``/``'none'`` for uncompressed,
                         or codec name ``'snappy'``, ``'gzip'``, ``'zstd'``,
                         ``'brotli'``, ``'lz4'``).
    :return: Dissolved :class:`geopandas.GeoDataFrame` when ``out_vec_file`` is
             ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.vector.geoms

        rsgispy.vector.geoms.vec_lyr_dissolve(
            "polys.gpkg", "polys", "cls", "dissolved.gpkg", "dissolved"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    data_gdf = read_vec_gdf(vec_file, vec_lyr, copy=False)
    validate_columns_exist(data_gdf, [vec_col])
    dis_data_gdf = data_gdf.dissolve(vec_col, as_index=False)
    if out_vec_file is None:
        return dis_data_gdf
    write_vec_gdf(
        dis_data_gdf,
        out_vec_file,
        out_vec_lyr,
        out_format=out_format,
        out_compress=out_compress,
    )
    return None


def thin_vector_geoms(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    grid_vec_file: VectorInput,
    grid_vec_lyr: Optional[str],
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    *,
    out_format: str = "GPKG",
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Thin vector geometries (typically points) using a polygon grid so that at
    most one feature intersecting each grid cell is retained.

    :param vec_file: Input vector file path or :class:`geopandas.GeoDataFrame`
                     to thin.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param grid_vec_file: Grid polygon vector file path or
                          :class:`geopandas.GeoDataFrame`.
    :param grid_vec_lyr: Grid polygon vector layer name (or ``None``).
    :param out_vec_file: Output vector file path, or ``None`` to return an
                         in-memory :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param out_compress: Optional GeoParquet compression setting (``None`` to
                         infer from extension or write uncompressed, ``True``
                         for ``'snappy'``, ``False``/``'none'`` for uncompressed,
                         or codec name ``'snappy'``, ``'gzip'``, ``'zstd'``,
                         ``'brotli'``, ``'lz4'``).
    :return: Thinned :class:`geopandas.GeoDataFrame` when ``out_vec_file`` is
             ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.vector.geoms

        rsgispy.vector.geoms.thin_vector_geoms(
            "pts.gpkg", "pts", "grid.gpkg", "grid", "pts_thin.gpkg", "pts"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    pts_gdf = read_vec_gdf(vec_file, vec_lyr, copy=False)
    grid_gdf = read_vec_gdf(grid_vec_file, grid_vec_lyr, copy=False)
    if (
        pts_gdf.crs is not None
        and grid_gdf.crs is not None
        and pts_gdf.crs != grid_gdf.crs
    ):
        grid_gdf = grid_gdf.to_crs(pts_gdf.crs)

    pts_only = gpd.GeoDataFrame(
        geometry=pts_gdf.geometry.values,
        index=np.arange(len(pts_gdf)),
        crs=pts_gdf.crs,
    )
    grid_only = gpd.GeoDataFrame(
        geometry=grid_gdf.geometry.values,
        index=np.arange(len(grid_gdf)),
        crs=grid_gdf.crs,
    )
    joined = gpd.sjoin(pts_only, grid_only, how="inner", predicate="intersects")
    joined = joined.drop_duplicates(subset="index_right", keep="first")
    kept_pos = joined.index.drop_duplicates(keep="first")
    pts_thin_gdf = pts_gdf.iloc[kept_pos].copy()

    if out_vec_file is None:
        return pts_thin_gdf
    if len(pts_thin_gdf) > 0:
        write_vec_gdf(
            pts_thin_gdf,
            out_vec_file,
            out_vec_lyr,
            out_format=out_format,
            out_compress=out_compress,
        )
    return None


def shiftxy_vec_lyr(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    x_shift: float,
    y_shift: float,
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    *,
    out_format: str = "GPKG",
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Translate (shift) all geometries in a vector layer by ``(x_shift, y_shift)``
    in the layer's coordinate reference system units.

    :param vec_file: Input vector file path or :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param x_shift: Shift along the X axis in CRS units.
    :param y_shift: Shift along the Y axis in CRS units.
    :param out_vec_file: Output vector file path, or ``None`` to return an
                         in-memory :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param out_compress: Optional GeoParquet compression setting (``None`` to
                         infer from extension or write uncompressed, ``True``
                         for ``'snappy'``, ``False``/``'none'`` for uncompressed,
                         or codec name ``'snappy'``, ``'gzip'``, ``'zstd'``,
                         ``'brotli'``, ``'lz4'``).
    :return: Shifted :class:`geopandas.GeoDataFrame` when ``out_vec_file`` is
             ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.vector.geoms

        rsgispy.vector.geoms.shiftxy_vec_lyr(
            "polys.gpkg", "polys", 10.0, -5.0, "shifted.gpkg", "polys"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    base_gdf = read_vec_gdf(vec_file, vec_lyr, copy=False)
    shifted_gdf = base_gdf.copy()
    shifted_gdf = shifted_gdf.set_geometry(
        base_gdf.geometry.translate(xoff=x_shift, yoff=y_shift)
    )
    if out_vec_file is None:
        return shifted_gdf
    write_vec_gdf(
        shifted_gdf,
        out_vec_file,
        out_vec_lyr,
        out_format=out_format,
        out_compress=out_compress,
    )
    return None
