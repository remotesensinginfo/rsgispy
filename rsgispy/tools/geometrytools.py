"""
The tools.geometrytools module provides bounding box, grid-snapping, and
coordinate reprojection utilities using pyproj.
"""

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pyproj

import rsgispy

__all__ = [
    "reproj_bbox",
    "reproj_bbox_epsg",
    "do_bboxes_intersect",
    "does_bbox_contain",
    "calc_bbox_area",
    "bbox_equal",
    "bbox_intersection",
    "bboxes_intersection",
    "buffer_bbox",
    "find_bbox_union",
    "unwrap_wgs84_bbox",
    "find_common_extent_on_grid",
    "find_extent_on_grid",
    "find_extent_on_whole_num_grid",
    "get_bbox_grid",
    "get_bbox_geojson_poly",
    "get_bbox_centre_pt",
    "reproj_point",
    "reproj_point_to_wgs84",
    "calc_pt_distance",
    "pt_in_bbox",
    "find_point_on_whole_num_grid",
    "get_bbox_wkt_poly",
    "degrees_to_metres",
]

CRSLike = Union[pyproj.CRS, int, str]


def _to_pyproj_crs(crs_val: CRSLike) -> pyproj.CRS:
    """Convert a CRS-like input (pyproj.CRS, EPSG int, string) into pyproj.CRS."""
    try:
        return pyproj.CRS.from_user_input(crs_val)
    except Exception as err:
        raise rsgispy.RSGISPyException(
            f"Could not parse CRS '{crs_val}': {err}"
        ) from err


def reproj_bbox(
    bbox: Tuple[float, float, float, float],
    in_crs: CRSLike,
    out_crs: CRSLike,
) -> Tuple[float, float, float, float]:
    """
    Reproject a bounding box from ``in_crs`` to ``out_crs`` using ``pyproj``.

    :param bbox: Input bounding box ``(MinX, MaxX, MinY, MaxY)``.
    :param in_crs: Input CRS (``pyproj.CRS``, EPSG int, or CRS string/WKT).
    :param out_crs: Output CRS (``pyproj.CRS``, EPSG int, or CRS string/WKT).
    :return: Reprojected bounding box ``(MinX, MaxX, MinY, MaxY)``.
    """
    in_crs_obj = _to_pyproj_crs(in_crs)
    out_crs_obj = _to_pyproj_crs(out_crs)
    transformer = pyproj.Transformer.from_crs(
        in_crs_obj, out_crs_obj, always_xy=True
    )
    min_x, min_y, max_x, max_y = transformer.transform_bounds(
        bbox[0], bbox[2], bbox[1], bbox[3]
    )
    return (float(min_x), float(max_x), float(min_y), float(max_y))


def reproj_bbox_epsg(
    bbox: Tuple[float, float, float, float],
    in_epsg: int,
    out_epsg: int,
) -> Tuple[float, float, float, float]:
    """
    Reproject a bounding box between EPSG codes using ``pyproj``.

    :param bbox: Input bounding box ``(MinX, MaxX, MinY, MaxY)``.
    :param in_epsg: Input projection EPSG integer code.
    :param out_epsg: Output projection EPSG integer code.
    :return: Reprojected bounding box ``(MinX, MaxX, MinY, MaxY)``.
    """
    return reproj_bbox(
        bbox,
        pyproj.CRS.from_epsg(int(in_epsg)),
        pyproj.CRS.from_epsg(int(out_epsg)),
    )


def do_bboxes_intersect(
    bbox1: Tuple[float, float, float, float],
    bbox2: Tuple[float, float, float, float],
) -> bool:
    """
    Test whether two bounding boxes ``(MinX, MaxX, MinY, MaxY)`` intersect.

    :param bbox1: First bounding box ``(MinX, MaxX, MinY, MaxY)``.
    :param bbox2: Second bounding box ``(MinX, MaxX, MinY, MaxY)``.
    :return: True if the bounding boxes overlap, False otherwise.
    """
    return (
        (bbox1[1] > bbox2[0])
        and (bbox2[1] > bbox1[0])
        and (bbox1[3] > bbox2[2])
        and (bbox2[3] > bbox1[2])
    )


def does_bbox_contain(
    bbox1: Tuple[float, float, float, float],
    bbox2: Tuple[float, float, float, float],
) -> bool:
    """
    Test whether ``bbox1`` completely contains ``bbox2``.

    :param bbox1: Outer bounding box ``(MinX, MaxX, MinY, MaxY)``.
    :param bbox2: Inner bounding box ``(MinX, MaxX, MinY, MaxY)``.
    :return: True if ``bbox1`` contains ``bbox2``, False otherwise.
    """
    return (
        (bbox1[0] <= bbox2[0])
        and (bbox1[1] >= bbox2[1])
        and (bbox1[2] <= bbox2[2])
        and (bbox1[3] >= bbox2[3])
    )


def calc_bbox_area(
    bbox: Tuple[float, float, float, float],
) -> float:
    """
    Calculate the area of a bounding box in its native CRS units.

    :param bbox: Bounding box ``(MinX, MaxX, MinY, MaxY)``.
    :return: Bounding box area.
    """
    return float((bbox[1] - bbox[0]) * (bbox[3] - bbox[2]))


def bbox_equal(
    bbox1: Tuple[float, float, float, float],
    bbox2: Tuple[float, float, float, float],
) -> bool:
    """
    Test whether two bounding boxes ``(MinX, MaxX, MinY, MaxY)`` are equal.

    :param bbox1: First bounding box ``(MinX, MaxX, MinY, MaxY)``.
    :param bbox2: Second bounding box ``(MinX, MaxX, MinY, MaxY)``.
    :return: True if all four coordinates are equal, False otherwise.
    """
    return (
        (bbox1[0] == bbox2[0])
        and (bbox1[1] == bbox2[1])
        and (bbox1[2] == bbox2[2])
        and (bbox1[3] == bbox2[3])
    )


def bbox_intersection(
    bbox1: Tuple[float, float, float, float],
    bbox2: Tuple[float, float, float, float],
) -> Tuple[float, float, float, float]:
    """
    Calculate the intersection of two bounding boxes ``(MinX, MaxX, MinY, MaxY)``.

    :param bbox1: First bounding box ``(MinX, MaxX, MinY, MaxY)``.
    :param bbox2: Second bounding box ``(MinX, MaxX, MinY, MaxY)``.
    :return: Intersecting bounding box ``(MinX, MaxX, MinY, MaxY)``.
    """
    if not do_bboxes_intersect(bbox1, bbox2):
        raise rsgispy.RSGISPyException("Bounding Boxes do not intersect.")

    return (
        max(bbox1[0], bbox2[0]),
        min(bbox1[1], bbox2[1]),
        max(bbox1[2], bbox2[2]),
        min(bbox1[3], bbox2[3]),
    )


def bboxes_intersection(
    bboxes: Sequence[Tuple[float, float, float, float]],
) -> Tuple[float, float, float, float]:
    """
    Find the common intersection across a sequence of bounding boxes.

    :param bboxes: Non-empty sequence of bounding boxes ``(MinX, MaxX, MinY, MaxY)``.
    :return: Intersecting bounding box ``(MinX, MaxX, MinY, MaxY)``.
    """
    if not bboxes:
        raise rsgispy.RSGISPyException("bboxes must contain at least one bbox.")

    inter_bbox = tuple(float(v) for v in bboxes[0])
    inter: Tuple[float, float, float, float] = (
        inter_bbox[0],
        inter_bbox[1],
        inter_bbox[2],
        inter_bbox[3],
    )
    for bbox in bboxes[1:]:
        inter = bbox_intersection(inter, bbox)
    return inter


def buffer_bbox(
    bbox: Tuple[float, float, float, float], buf: float
) -> Tuple[float, float, float, float]:
    """
    Buffer a bounding box by ``buf`` in all four directions.

    :param bbox: Bounding box ``(MinX, MaxX, MinY, MaxY)``.
    :param buf: Buffer distance added outward on each side.
    :return: Buffered bounding box ``(MinX, MaxX, MinY, MaxY)``.
    """
    return (
        bbox[0] - buf,
        bbox[1] + buf,
        bbox[2] - buf,
        bbox[3] + buf,
    )


def find_bbox_union(
    bboxes: Sequence[Tuple[float, float, float, float]],
) -> Optional[Tuple[float, float, float, float]]:
    """
    Find the union bounding box enclosing all bounding boxes in ``bboxes``.

    :param bboxes: Sequence of bounding boxes ``(MinX, MaxX, MinY, MaxY)``.
    :return: Union bounding box ``(MinX, MaxX, MinY, MaxY)``, or ``None`` if empty.
    """
    if not bboxes:
        return None
    min_x = min(b[0] for b in bboxes)
    max_x = max(b[1] for b in bboxes)
    min_y = min(b[2] for b in bboxes)
    max_y = max(b[3] for b in bboxes)
    return (min_x, max_x, min_y, max_y)


def unwrap_wgs84_bbox(
    bbox: Tuple[float, float, float, float],
) -> List[Tuple[float, float, float, float]]:
    """
    Split a WGS84 bounding box that crosses the 180/-180 antimeridian
    (where ``MaxX < MinX``) into two non-wrapping bounding boxes.

    :param bbox: WGS84 bounding box ``(MinX, MaxX, MinY, MaxY)``.
    :return: List of one or two bounding boxes ``(MinX, MaxX, MinY, MaxY)``.
    """
    if bbox[1] < bbox[0]:
        return [
            (-180.0, bbox[1], bbox[2], bbox[3]),
            (bbox[0], 180.0, bbox[2], bbox[3]),
        ]
    return [bbox]


def find_common_extent_on_grid(
    base_extent: Tuple[float, float, float, float],
    base_grid: float,
    other_extent: Tuple[float, float, float, float],
    full_contain: bool = True,
) -> Tuple[float, float, float, float]:
    """
    Compute the common intersection extent between ``base_extent`` and
    ``other_extent`` snapped onto a grid of spacing ``base_grid`` anchored at
    ``base_extent``.

    :param base_extent: Reference bounding box ``(MinX, MaxX, MinY, MaxY)``.
    :param base_grid: Square grid spacing.
    :param other_extent: Second bounding box ``(MinX, MaxX, MinY, MaxY)``.
    :param full_contain: If True, snap outward to contain the intersection;
                         if False, snap inward within the intersection.
    :return: Grid-aligned bounding box ``(MinX, MaxX, MinY, MaxY)``.
    """
    x_min_overlap, x_max_overlap, y_min_overlap, y_max_overlap = base_extent
    snap_fn = math.floor if full_contain else math.ceil

    if other_extent[0] > x_min_overlap:
        diff = snap_fn((other_extent[0] - x_min_overlap) / base_grid) * base_grid
        x_min_overlap += diff

    if other_extent[1] < x_max_overlap:
        diff = snap_fn((x_max_overlap - other_extent[1]) / base_grid) * base_grid
        x_max_overlap -= diff

    if other_extent[2] > y_min_overlap:
        diff = (
            snap_fn(abs(other_extent[2] - y_min_overlap) / base_grid) * base_grid
        )
        y_min_overlap += diff

    if other_extent[3] < y_max_overlap:
        diff = (
            snap_fn(abs(y_max_overlap - other_extent[3]) / base_grid) * base_grid
        )
        y_max_overlap -= diff

    return (x_min_overlap, x_max_overlap, y_min_overlap, y_max_overlap)


def find_extent_on_grid(
    base_extent: Tuple[float, float, float, float],
    base_grid: float,
    full_contain: bool = True,
) -> Tuple[float, float, float, float]:
    """
    Snap an extent onto a grid of spacing ``base_grid``.

    :param base_extent: Input bounding box ``(MinX, MaxX, MinY, MaxY)``.
    :param base_grid: Square grid spacing.
    :param full_contain: If True, round cell counts up to contain ``base_extent``;
                         if False, round down.
    :return: Grid-aligned bounding box ``(MinX, MaxX, MinY, MaxY)``.
    """
    x_min, x_max, y_min, y_max = base_extent
    x_min_out = (
        math.floor(x_min)
        + math.floor((x_min - math.floor(x_min)) / base_grid) * base_grid
    )
    y_max_out = (
        math.ceil(y_max)
        - math.floor((math.ceil(y_max) - y_max) / base_grid) * base_grid
    )

    count_fn = math.ceil if full_contain else math.floor
    x_max_out = x_min_out + (count_fn((x_max - x_min_out) / base_grid) * base_grid)
    y_min_out = y_max_out - (count_fn(abs(y_max_out - y_min) / base_grid) * base_grid)

    return (x_min_out, x_max_out, y_min_out, y_max_out)


def find_extent_on_whole_num_grid(
    base_extent: Tuple[float, float, float, float],
    base_grid: float,
    full_contain: bool = True,
    round_vals: Optional[int] = None,
) -> Tuple[float, float, float, float]:
    """
    Snap an extent onto a grid anchored at integer multiples of ``base_grid``.

    :param base_extent: Input bounding box ``(MinX, MaxX, MinY, MaxY)``.
    :param base_grid: Square grid spacing.
    :param full_contain: If True, round cell counts up to contain ``base_extent``;
                         if False, round down.
    :param round_vals: Optional decimal places to round the output coordinates.
    :return: Grid-aligned bounding box ``(MinX, MaxX, MinY, MaxY)``.
    """
    x_min, x_max, y_min, y_max = base_extent
    x_min_out = math.floor(x_min / base_grid) * base_grid
    y_min_out = math.floor(y_min / base_grid) * base_grid

    count_fn = math.ceil if full_contain else math.floor
    x_max_out = x_min_out + (count_fn((x_max - x_min_out) / base_grid) * base_grid)
    y_max_out = y_min_out + (count_fn(abs(y_max - y_min_out) / base_grid) * base_grid)

    if round_vals is None:
        return (x_min_out, x_max_out, y_min_out, y_max_out)
    return (
        round(x_min_out, round_vals),
        round(x_max_out, round_vals),
        round(y_min_out, round_vals),
        round(y_max_out, round_vals),
    )


def get_bbox_grid(
    bbox: Tuple[float, float, float, float],
    x_size: float,
    y_size: float,
) -> List[Tuple[float, float, float, float]]:
    """
    Subdivide ``bbox`` into a grid of tiles of width ``x_size`` and height
    ``y_size`` (plus remainder edge tiles where needed).

    :param bbox: Bounding box ``(MinX, MaxX, MinY, MaxY)`` to tile.
    :param x_size: Tile width in bbox units.
    :param y_size: Tile height in bbox units.
    :return: List of tile bounding boxes ``(MinX, MaxX, MinY, MaxY)``.
    """
    # pylint: disable=too-many-locals
    width = bbox[1] - bbox[0]
    height = bbox[3] - bbox[2]

    n_tiles_x = math.floor(width / x_size)
    n_tiles_y = math.floor(height / y_size)

    if (n_tiles_x > 10000) or (n_tiles_y > 10000):
        print(
            f"WARNING: did you mean to produce so many tiles "
            f"(X: {n_tiles_x}, Y: {n_tiles_y}) might want to check your units"
        )

    x_remain = width - (n_tiles_x * x_size)
    if x_remain < 1e-6:
        x_remain = 0.0
    y_remain = height - (n_tiles_y * y_size)
    if y_remain < 1e-6:
        y_remain = 0.0

    def _row_tiles(row_min_y: float, row_max_y: float) -> List[Tuple[float, float, float, float]]:
        row_list: List[Tuple[float, float, float, float]] = []
        c_min_x = bbox[0]
        for _ in range(n_tiles_x):
            c_max_x = c_min_x + x_size
            row_list.append((c_min_x, c_max_x, row_min_y, row_max_y))
            c_min_x = c_max_x
        if x_remain > 0:
            row_list.append((c_min_x, c_min_x + x_remain, row_min_y, row_max_y))
        return row_list

    bboxes: List[Tuple[float, float, float, float]] = []
    c_min_y = bbox[2]
    for _ in range(n_tiles_y):
        c_max_y = c_min_y + y_size
        bboxes.extend(_row_tiles(c_min_y, c_max_y))
        c_min_y = c_max_y
    if y_remain > 0:
        bboxes.extend(_row_tiles(c_min_y, c_min_y + y_remain))

    return bboxes


def get_bbox_geojson_poly(
    bbox: Tuple[float, float, float, float],
) -> Dict[str, Any]:
    """
    Represent ``bbox`` ``(MinX, MaxX, MinY, MaxY)`` as a GeoJSON Polygon dict.

    :param bbox: Bounding box ``(MinX, MaxX, MinY, MaxY)``.
    :return: GeoJSON Polygon dictionary.
    """
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [bbox[0], bbox[3]],
                [bbox[1], bbox[3]],
                [bbox[1], bbox[2]],
                [bbox[0], bbox[2]],
                [bbox[0], bbox[3]],
            ]
        ],
    }


def get_bbox_centre_pt(
    bbox: Tuple[float, float, float, float],
) -> Tuple[float, float]:
    """
    Return the centre point ``(x, y)`` of ``bbox`` ``(MinX, MaxX, MinY, MaxY)``.

    :param bbox: Bounding box ``(MinX, MaxX, MinY, MaxY)``.
    :return: Centre point ``(x, y)``.
    """
    x_pt = bbox[0] + (bbox[1] - bbox[0]) / 2.0
    y_pt = bbox[2] + (bbox[3] - bbox[2]) / 2.0
    return (x_pt, y_pt)


def reproj_point(
    in_crs: CRSLike,
    out_crs: CRSLike,
    x: float,
    y: float,
) -> Tuple[float, float]:
    """
    Reproject a point ``(x, y)`` from ``in_crs`` to ``out_crs`` using ``pyproj``.

    :param in_crs: Input CRS (``pyproj.CRS``, EPSG int, or CRS string/WKT).
    :param out_crs: Output CRS (``pyproj.CRS``, EPSG int, or CRS string/WKT).
    :param x: Input X (or longitude) coordinate.
    :param y: Input Y (or latitude) coordinate.
    :return: Reprojected ``(x, y)`` tuple.
    """
    in_crs_obj = _to_pyproj_crs(in_crs)
    out_crs_obj = _to_pyproj_crs(out_crs)
    transformer = pyproj.Transformer.from_crs(
        in_crs_obj, out_crs_obj, always_xy=True
    )
    out_x, out_y = transformer.transform(x, y)
    return (float(out_x), float(out_y))


def reproj_point_to_wgs84(
    in_crs: CRSLike,
    x: float,
    y: float,
) -> Tuple[float, float]:
    """
    Reproject a point ``(x, y)`` from ``in_crs`` to WGS84 (``EPSG:4326``)
    returning ``(lon, lat)``.

    :param in_crs: Input CRS (``pyproj.CRS``, EPSG int, or CRS string/WKT).
    :param x: Input X coordinate.
    :param y: Input Y coordinate.
    :return: ``(lon, lat)`` in EPSG:4326.
    """
    return reproj_point(in_crs, pyproj.CRS.from_epsg(4326), x, y)


def calc_pt_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """
    Calculate the Euclidean distance between two points ``(x1, y1)`` and
    ``(x2, y2)``.

    :param x1: First point X coordinate.
    :param y1: First point Y coordinate.
    :param x2: Second point X coordinate.
    :param y2: Second point Y coordinate.
    :return: Euclidean distance.
    """
    return math.hypot(x1 - x2, y1 - y2)


def pt_in_bbox(
    pt: Tuple[float, float],
    bbox: Tuple[float, float, float, float],
) -> bool:
    """
    Test whether a point ``(x, y)`` lies within ``bbox`` ``(MinX, MaxX, MinY, MaxY)``.

    :param pt: Point ``(x, y)``.
    :param bbox: Bounding box ``(MinX, MaxX, MinY, MaxY)``.
    :return: True if inside or on the boundary of ``bbox``, False otherwise.
    """
    return (
        (pt[0] >= bbox[0])
        and (pt[0] <= bbox[1])
        and (pt[1] >= bbox[2])
        and (pt[1] <= bbox[3])
    )


def find_point_on_whole_num_grid(
    pt: Tuple[float, float],
    bbox: Tuple[float, float, float, float],
    x_grid_res: float,
    y_grid_res: float,
) -> Tuple[float, float]:
    """
    Snap a point ``(x, y)`` inside ``bbox`` onto the grid defined by ``bbox``'s
    top-left corner and pixel spacings ``(x_grid_res, y_grid_res)``.

    :param pt: Point ``(x, y)`` within ``bbox``.
    :param bbox: Grid extent bounding box ``(MinX, MaxX, MinY, MaxY)``.
    :param x_grid_res: Grid resolution along the X axis.
    :param y_grid_res: Grid resolution along the Y axis.
    :return: Grid-snapped point ``(x, y)``.
    """
    if not pt_in_bbox(pt, bbox):
        raise rsgispy.RSGISPyException("Point is not within the bbox!")

    diff_x = pt[0] - bbox[0]
    diff_y = bbox[3] - pt[1]

    n_pxl_x = math.ceil(diff_x / x_grid_res)
    n_pxl_y = math.ceil(diff_y / y_grid_res)

    return (
        bbox[0] + (n_pxl_x * x_grid_res),
        bbox[3] - (n_pxl_y * y_grid_res),
    )


def get_bbox_wkt_poly(
    bbox: Tuple[float, float, float, float],
) -> str:
    """
    Return the WKT Polygon representation of ``bbox`` ``(MinX, MaxX, MinY, MaxY)``.

    :param bbox: Bounding box ``(MinX, MaxX, MinY, MaxY)``.
    :return: WKT Polygon string.
    """
    return (
        f"POLYGON (({bbox[0]} {bbox[2]}, "
        f"{bbox[1]} {bbox[2]}, "
        f"{bbox[1]} {bbox[3]}, "
        f"{bbox[0]} {bbox[3]}, "
        f"{bbox[0]} {bbox[2]}))"
    )


def degrees_to_metres(
    latitude: Union[float, np.ndarray],
    lon_size: Union[float, np.ndarray],
    lat_size: Union[float, np.ndarray],
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Convert pixel sizes or distances in degrees (WGS84 ellipsoid) to metres.

    :param latitude: Latitude in degrees (scalar or array).
    :param lon_size: Longitude delta/pixel size in degrees (scalar or array).
    :param lat_size: Latitude delta/pixel size in degrees (scalar or array).
    :return: Tuple ``(x_size_metres, y_size_metres)``.
    """
    semi_major = 6378137.0
    semi_minor = 6356752.314245
    rad_lat = np.deg2rad(latitude)
    r_sq = (semi_major * np.cos(rad_lat)) ** 2 + (
        semi_minor * np.sin(rad_lat)
    ) ** 2
    m_lat = ((semi_major * semi_minor) ** 2) / (r_sq**1.5)
    n_lon = (semi_major**2) / np.sqrt(r_sq)
    deg_to_rad = math.pi / 180.0
    x_size = deg_to_rad * np.cos(rad_lat) * n_lon * lon_size
    y_size = deg_to_rad * m_lat * lat_size
    return x_size, y_size
