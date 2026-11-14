"""
The vectorgeoms.polygons module provides geometric transformations and polygon
utilities (boundary extraction, centroids, exploding multi-geometries,
simplification, hole removal/inspection, buffering, grid splitting, area
filtering, dissolving, thinning, and coordinate shifting) using ``geopandas``,
``pyogrio``, and ``shapely``.
"""

import os
from typing import Optional, Sequence

import geopandas
import numpy
import pyogrio
import shapely.geometry

import rsgispy
import rsgispy.tools.files
import rsgispy.tools.geometry
from rsgispy.vectorprocess import (
    prepare_out_vec_file,
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
    "create_bbox_vec_lyr",
    "vec_lyr_dissolve",
    "thin_vector_geoms",
    "shiftxy_vec_lyr",
]


def convert_polys_to_lines_gp(
    vec_poly_file: str,
    vec_poly_lyr: Optional[str],
    vec_line_file: str,
    vec_line_lyr: Optional[str] = None,
    out_format: str = "GPKG",
    del_exist_vec: bool = False,
    exterior_lines: bool = False,
) -> None:
    """
    Convert a polygon vector layer to a line vector layer using ``geopandas``.

    :param vec_poly_file: Input polygon vector file path.
    :param vec_poly_lyr: Input vector layer name (or ``None``).
    :param vec_line_file: Output line vector file path.
    :param vec_line_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param del_exist_vec: If ``True``, overwrite existing ``vec_line_file``
                          (Default = ``False``).
    :param exterior_lines: If ``True``, extract only outer exterior rings as
                           lines; otherwise include interior hole boundaries
                           (Default = ``False``).

    .. code:: python

        import rsgispy.vectorgeoms.polygons

        rsgispy.vectorgeoms.polygons.convert_polys_to_lines_gp(
            "polys.gpkg", "polys", "lines.gpkg", "lines", exterior_lines=True
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    prepare_out_vec_file(vec_line_file, del_exist_vec=del_exist_vec)
    out_lyr = resolve_out_lyr_name(vec_line_file, vec_line_lyr)

    data_gdf = read_vec_gdf(vec_poly_file, vec_poly_lyr)
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
        out_gdf["geometry"] = ext_geoms
    else:
        out_gdf["geometry"] = data_gdf.geometry.boundary

    write_vec_gdf(out_gdf, vec_line_file, out_lyr, out_format=out_format)


def calc_poly_centroids(
    vec_file: str,
    vec_lyr: Optional[str],
    out_format: str,
    out_vec_file: str,
    out_vec_lyr: Optional[str],
) -> None:
    """
    Create a point vector layer containing the centroid of each polygon feature.

    :param vec_file: Input vector file path.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param out_format: Output vector format driver (e.g., ``'GPKG'``).
    :param out_vec_file: Output centroid vector file path.
    :param out_vec_lyr: Output vector layer name.

    .. code:: python

        import rsgispy.vectorgeoms.polygons

        rsgispy.vectorgeoms.polygons.calc_poly_centroids(
            "polys.gpkg", "polys", "GPKG", "centroids.gpkg", "centroids"
        )
    """
    in_gdf = read_vec_gdf(vec_file, vec_lyr)
    out_gdf = in_gdf.copy()
    out_gdf["geometry"] = in_gdf.geometry.centroid
    write_vec_gdf(out_gdf, out_vec_file, out_vec_lyr, out_format=out_format)


def explode_vec_lyr(
    vec_file: str,
    vec_lyr: Optional[str],
    out_vec_file: str,
    out_vec_lyr: Optional[str],
    out_format: str = "GPKG",
) -> None:
    """
    Explode multi-part geometries (e.g., ``MultiPolygon``,
    ``MultiLineString``, ``MultiPoint``) into single-part features using
    ``geopandas``.

    :param vec_file: Input vector file path.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param out_vec_file: Output vector file path.
    :param out_vec_lyr: Output vector layer name (can be ``None`` for single-layer
                        formats).
    :param out_format: Output vector format driver (Default = ``'GPKG'``).

    .. code:: python

        import rsgispy.vectorgeoms.polygons

        rsgispy.vectorgeoms.polygons.explode_vec_lyr(
            "multi.gpkg", "polys", "single.gpkg", "polys"
        )
    """
    data_gdf = read_vec_gdf(vec_file, vec_lyr)
    data_explode_gdf = data_gdf.explode(index_parts=False)
    if len(data_explode_gdf) > 0:
        write_vec_gdf(
            data_explode_gdf, out_vec_file, out_vec_lyr, out_format=out_format
        )


def explode_vec_files(
    in_vec_files: Sequence[str],
    output_dir: str,
    out_format: str = "GPKG",
    out_vec_ext: str = "gpkg",
    use_pbar: bool = True,
) -> None:
    """
    Explode multi-part geometries across every layer in each file of
    ``in_vec_files``, writing outputs to ``output_dir``.

    :param in_vec_files: Sequence of input vector file paths.
    :param output_dir: Output directory (must differ from input directory to avoid
                       overwriting inputs).
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param out_vec_ext: Output file extension without leading dot
                        (Default = ``'gpkg'``).
    :param use_pbar: Show progress bar (Default = ``True``).

    .. code:: python

        import rsgispy.vectorgeoms.polygons

        rsgispy.vectorgeoms.polygons.explode_vec_files(
            ["a.gpkg", "b.gpkg"], output_dir="exploded"
        )
    """
    os.makedirs(output_dir, exist_ok=True)
    pbar_cls = rsgispy.get_tqdm()
    for vec_file in pbar_cls(
        in_vec_files, total=len(in_vec_files), disable=not use_pbar
    ):
        lyrs = [str(row[0]) for row in pyogrio.list_layers(vec_file)]
        basename = rsgispy.tools.files.get_file_basename(vec_file)
        out_vec_file = os.path.join(output_dir, f"{basename}.{out_vec_ext}")
        for lyr in lyrs:
            explode_vec_lyr(vec_file, lyr, out_vec_file, lyr, out_format=out_format)


def simplify_geometries(
    vec_file: str,
    vec_lyr: Optional[str],
    tolerance: float,
    out_format: str,
    out_vec_file: str,
    out_vec_lyr: Optional[str],
    del_exist_vec: bool = False,
) -> None:
    """
    Simplify vector geometries using the Douglas-Peucker algorithm with
    topology preservation.

    :param vec_file: Input vector file path.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param tolerance: Simplification tolerance in CRS units.
    :param out_format: Output vector format driver (e.g., ``'GPKG'``).
    :param out_vec_file: Output vector file path.
    :param out_vec_lyr: Output vector layer name.
    :param del_exist_vec: If ``True``, overwrite existing ``out_vec_file``
                          (Default = ``False``).

    .. code:: python

        import rsgispy.vectorgeoms.polygons

        rsgispy.vectorgeoms.polygons.simplify_geometries(
            "polys.gpkg", "polys", 2.0, "GPKG", "simple.gpkg", "polys"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    prepare_out_vec_file(out_vec_file, del_exist_vec=del_exist_vec)
    in_gdf = read_vec_gdf(vec_file, vec_lyr)
    out_gdf = in_gdf.copy()
    out_gdf["geometry"] = in_gdf.geometry.simplify(tolerance, preserve_topology=True)
    write_vec_gdf(out_gdf, out_vec_file, out_vec_lyr, out_format=out_format)


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
    vec_file: str,
    vec_lyr: Optional[str],
    out_format: str,
    out_vec_file: str,
    out_vec_lyr: Optional[str],
    area_thres: Optional[float] = None,
    del_exist_vec: bool = False,
    use_pbar: bool = True,
) -> None:
    """
    Delete interior holes from input polygons whose area is below or equal to
    ``area_thres`` (or all holes if ``area_thres`` is ``None``).

    :param vec_file: Input polygon vector file path.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param out_format: Output vector format driver (e.g., ``'GPKG'``).
    :param out_vec_file: Output vector file path.
    :param out_vec_lyr: Output vector layer name.
    :param area_thres: Area threshold in CRS units squared; if ``None``, all
                       holes are removed.
    :param del_exist_vec: If ``True``, overwrite existing ``out_vec_file``
                          (Default = ``False``).
    :param use_pbar: Show progress bar (Default = ``True``).

    .. code:: python

        import rsgispy.vectorgeoms.polygons

        rsgispy.vectorgeoms.polygons.delete_polygon_holes(
            "polys_holes.gpkg", "polys", "GPKG", "polys_clean.gpkg", "polys",
            area_thres=100.0
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    prepare_out_vec_file(out_vec_file, del_exist_vec=del_exist_vec)
    in_gdf = read_vec_gdf(vec_file, vec_lyr)
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
    out_gdf["geometry"] = out_geoms
    write_vec_gdf(out_gdf, out_vec_file, out_vec_lyr, out_format=out_format)


def buffer_vec_layer_gp(
    vec_file: str,
    vec_lyr: Optional[str],
    buf_dist: float,
    out_vec_file: str,
    out_vec_lyr: Optional[str] = None,
    out_format: str = "GPKG",
    del_exist_vec: bool = False,
    buf_res: int = 16,
    buf_sgl_sided: bool = False,
    cap_style: int = 1,
    join_style: int = 1,
) -> None:
    """
    Buffer the geometries in a vector layer using ``geopandas``.

    :param vec_file: Input vector file path.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param buf_dist: Buffer distance in CRS units (negative values shrink
                     polygons or indicate the right-hand side when
                     ``buf_sgl_sided=True``).
    :param out_vec_file: Output vector file path.
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

    .. code:: python

        import rsgispy.vectorgeoms.polygons

        rsgispy.vectorgeoms.polygons.buffer_vec_layer_gp(
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

    prepare_out_vec_file(out_vec_file, del_exist_vec=del_exist_vec)
    out_lyr = resolve_out_lyr_name(out_vec_file, out_vec_lyr)

    data_gdf = read_vec_gdf(vec_file, vec_lyr)
    data_gdf["geometry"] = data_gdf["geometry"].buffer(
        distance=buf_dist,
        resolution=buf_res,
        single_sided=buf_sgl_sided,
        cap_style=cap_style,
        join_style=join_style,
    )
    write_vec_gdf(data_gdf, out_vec_file, out_lyr, out_format=out_format)


def split_vec_by_grid(
    vec_file: str,
    vec_lyr: Optional[str],
    x_grid_size: float,
    y_grid_size: float,
    out_vec_file: str,
    out_vec_lyr: Optional[str],
    out_format: str = "GPKG",
) -> None:
    """
    Split an input vector layer along a regular grid of spacing
    ``(x_grid_size, y_grid_size)`` covering its bounding extent.

    :param vec_file: Input vector file path.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param x_grid_size: Grid cell width in CRS units.
    :param y_grid_size: Grid cell height in CRS units.
    :param out_vec_file: Output vector file path.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).

    .. code:: python

        import rsgispy.vectorgeoms.polygons

        rsgispy.vectorgeoms.polygons.split_vec_by_grid(
            "polys.gpkg", "polys", 100.0, 100.0, "split.gpkg", "split"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    in_vec_gdf = read_vec_gdf(vec_file, vec_lyr)
    xmin, ymin, xmax, ymax = in_vec_gdf.total_bounds
    tile_bboxes = rsgispy.tools.geometry.get_bbox_grid(
        (float(xmin), float(xmax), float(ymin), float(ymax)),
        x_grid_size,
        y_grid_size,
    )
    polygons = [
        shapely.geometry.box(b[0], b[2], b[1], b[3]) for b in tile_bboxes
    ]

    grid_gdf = geopandas.GeoDataFrame(
        {"geometry": polygons}, crs=in_vec_gdf.crs
    )
    vec_split_gdf = in_vec_gdf.overlay(grid_gdf, how="intersection")
    write_vec_gdf(vec_split_gdf, out_vec_file, out_vec_lyr, out_format=out_format)


def rm_polys_area(
    vec_file: str,
    vec_lyr: Optional[str],
    area_thres: float,
    out_vec_file: str,
    out_vec_lyr: Optional[str],
    out_format: str = "GPKG",
    less_than: bool = True,
) -> None:
    """
    Filter polygons by keeping features whose area is either strictly less than
    or strictly greater than ``area_thres``.

    :param vec_file: Input vector file path.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param area_thres: Area threshold in CRS units squared.
    :param out_vec_file: Output vector file path.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param less_than: When ``True`` (default), keep features with
                      ``area < area_thres``; when ``False``, keep features with
                      ``area > area_thres``.

    .. code:: python

        import rsgispy.vectorgeoms.polygons

        rsgispy.vectorgeoms.polygons.rm_polys_area(
            "polys.gpkg", "polys", 50.0, "large.gpkg", "polys", less_than=False
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    in_vec_gdf = read_vec_gdf(vec_file, vec_lyr)
    if less_than:
        area_msk = in_vec_gdf.area < area_thres
    else:
        area_msk = in_vec_gdf.area > area_thres
    vec_sel_gdf = in_vec_gdf.loc[area_msk]

    if len(vec_sel_gdf) > 0:
        write_vec_gdf(vec_sel_gdf, out_vec_file, out_vec_lyr, out_format=out_format)


def create_bbox_vec_lyr(
    vec_file: str,
    vec_lyr: Optional[str],
    out_vec_file: str,
    out_vec_lyr: Optional[str],
    out_format: str = "GPKG",
    whole_num_grid: bool = False,
    whole_num_base_grid: float = 10.0,
    whole_num_round: Optional[int] = None,
) -> None:
    """
    Create a single-polygon vector layer representing the bounding box of
    ``vec_file``, optionally snapped to a whole-number grid.

    :param vec_file: Input vector file path.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param out_vec_file: Output vector file path.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param whole_num_grid: Snap bounding box onto a whole-number grid via
                           :func:`rsgispy.tools.geometry.find_extent_on_whole_num_grid`
                           (Default = ``False``).
    :param whole_num_base_grid: Grid cell size in CRS units (Default = ``10.0``).
    :param whole_num_round: Optional decimal places to round coordinates.

    .. code:: python

        import rsgispy.vectorgeoms.polygons

        rsgispy.vectorgeoms.polygons.create_bbox_vec_lyr(
            "polys.gpkg", "polys", "bbox.gpkg", "bbox", whole_num_grid=True
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    in_vec_gdf = read_vec_gdf(vec_file, vec_lyr)
    x_min, y_min, x_max, y_max = in_vec_gdf.total_bounds

    if whole_num_grid:
        x_min, x_max, y_min, y_max = rsgispy.tools.geometry.find_extent_on_whole_num_grid(
            base_extent=(float(x_min), float(x_max), float(y_min), float(y_max)),
            base_grid=whole_num_base_grid,
            full_contain=True,
            round_vals=whole_num_round,
        )

    bbox_gdf = geopandas.GeoDataFrame(
        {"geometry": [shapely.geometry.box(x_min, y_min, x_max, y_max)]},
        crs=in_vec_gdf.crs,
    )
    write_vec_gdf(bbox_gdf, out_vec_file, out_vec_lyr, out_format=out_format)


def vec_lyr_dissolve(
    vec_file: str,
    vec_lyr: Optional[str],
    vec_col: str,
    out_vec_file: str,
    out_vec_lyr: Optional[str],
    out_format: str = "GPKG",
) -> None:
    """
    Dissolve vector features by grouping on categorical attribute ``vec_col``.

    :param vec_file: Input vector file path.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param vec_col: Categorical column name to dissolve on.
    :param out_vec_file: Output vector file path.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).

    .. code:: python

        import rsgispy.vectorgeoms.polygons

        rsgispy.vectorgeoms.polygons.vec_lyr_dissolve(
            "polys.gpkg", "polys", "cls", "dissolved.gpkg", "dissolved"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    data_gdf = read_vec_gdf(vec_file, vec_lyr)
    validate_columns_exist(data_gdf, [vec_col])
    dis_data_gdf = data_gdf.dissolve(vec_col, as_index=False)
    write_vec_gdf(dis_data_gdf, out_vec_file, out_vec_lyr, out_format=out_format)


def thin_vector_geoms(
    vec_file: str,
    vec_lyr: Optional[str],
    grid_vec_file: str,
    grid_vec_lyr: Optional[str],
    out_vec_file: str,
    out_vec_lyr: Optional[str],
    out_format: str = "GPKG",
) -> None:
    """
    Thin vector geometries (typically points) using a polygon grid so that at
    most one feature intersecting each grid cell is retained.

    :param vec_file: Input vector file path to thin.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param grid_vec_file: Grid polygon vector file path.
    :param grid_vec_lyr: Grid polygon vector layer name (or ``None``).
    :param out_vec_file: Output vector file path.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).

    .. code:: python

        import rsgispy.vectorgeoms.polygons

        rsgispy.vectorgeoms.polygons.thin_vector_geoms(
            "pts.gpkg", "pts", "grid.gpkg", "grid", "pts_thin.gpkg", "pts"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    pts_gdf = read_vec_gdf(vec_file, vec_lyr)
    grid_gdf = read_vec_gdf(grid_vec_file, grid_vec_lyr)
    grid_gdf["unq_id"] = numpy.arange(1, len(grid_gdf) + 1, 1, dtype=int)

    pts_join_gdf = geopandas.sjoin(
        pts_gdf, grid_gdf, how="inner", predicate="intersects"
    )
    pts_join_gdf.drop_duplicates(
        subset="unq_id", keep="first", inplace=True, ignore_index=False
    )
    if len(pts_join_gdf) > 0:
        write_vec_gdf(pts_join_gdf, out_vec_file, out_vec_lyr, out_format=out_format)


def shiftxy_vec_lyr(
    vec_file: str,
    vec_lyr: Optional[str],
    x_shift: float,
    y_shift: float,
    out_vec_file: str,
    out_vec_lyr: Optional[str],
    out_format: str = "GPKG",
) -> None:
    """
    Translate (shift) all geometries in a vector layer by ``(x_shift, y_shift)``
    in the layer's coordinate reference system units.

    :param vec_file: Input vector file path.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param x_shift: Shift along the X axis in CRS units.
    :param y_shift: Shift along the Y axis in CRS units.
    :param out_vec_file: Output vector file path.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).

    .. code:: python

        import rsgispy.vectorgeoms.polygons

        rsgispy.vectorgeoms.polygons.shiftxy_vec_lyr(
            "polys.gpkg", "polys", 10.0, -5.0, "shifted.gpkg", "polys"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    base_gdf = read_vec_gdf(vec_file, vec_lyr)
    shifted_gdf = base_gdf.copy()
    shifted_gdf["geometry"] = base_gdf.geometry.translate(
        xoff=x_shift, yoff=y_shift
    )
    write_vec_gdf(shifted_gdf, out_vec_file, out_vec_lyr, out_format=out_format)
