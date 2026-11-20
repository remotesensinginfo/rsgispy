"""
Geometric operations, line and polygon manipulation, vector overlays, vertex
extraction, concave hulls, R-tree indexing, and spatial predicates for
:mod:`rsgispy.vector.geoms`.
"""

from rsgispy.vector._lines import (
    closest_line_intersection,
    create_angle_lines_from_points,
    create_orthg_lines,
    estimate_future_coastline,
    find_pt_to_side,
    get_pt_on_line,
    line_intersection_range,
    pts_to_line_geoms,
    scnd_line_intersection_range,
    split_lines_to_reg_pts,
)
from rsgispy.vector._overlay import (
    clip_and_merge_with_roi,
    clip_vec_into_sets,
    clip_vec_lyr,
    vec_lyr_difference_gp,
    vec_lyr_identity_gp,
    vec_lyr_intersection_gp,
    vec_lyr_sym_difference_gp,
    vec_lyr_union_gp,
)
from rsgispy.vector._polygons import (
    buffer_vec_layer_gp,
    calc_poly_centroids,
    convert_polys_to_lines_gp,
    delete_polygon_holes,
    explode_vec_files,
    explode_vec_lyr,
    rm_polys_area,
    shiftxy_vec_lyr,
    simplify_geometries,
    split_vec_by_grid,
    thin_vector_geoms,
    vec_lyr_dissolve,
)
from rsgispy.vector._spatial import (
    bbox_intersects_index,
    bbox_intersects_vec_lyr,
    create_alpha_shape,
    create_rtree_index,
    get_geom_pts,
    get_geoms_as_bboxs,
    get_vec_lyr_as_pts,
    vec_contains_vec,
    vec_crosses_vec,
    vec_intersects_vec,
    vec_overlaps_vec,
    vec_touches_vec,
    vec_within_vec,
)

__all__ = [
    # Line geometry operations
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
    # Overlay operations
    "vec_lyr_intersection_gp",
    "vec_lyr_difference_gp",
    "vec_lyr_sym_difference_gp",
    "vec_lyr_identity_gp",
    "vec_lyr_union_gp",
    "clip_vec_lyr",
    "clip_and_merge_with_roi",
    "clip_vec_into_sets",
    # Polygon operations
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
    # Spatial indexing & predicates
    "get_geom_pts",
    "get_vec_lyr_as_pts",
    "create_alpha_shape",
    "get_geoms_as_bboxs",
    "bbox_intersects_vec_lyr",
    "create_rtree_index",
    "bbox_intersects_index",
    "vec_intersects_vec",
    "vec_overlaps_vec",
    "vec_within_vec",
    "vec_contains_vec",
    "vec_touches_vec",
    "vec_crosses_vec",
]
