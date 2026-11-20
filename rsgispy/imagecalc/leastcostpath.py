"""
The imagecalc.leastcostpath module provides least-cost path calculation across
a raster cost surface using ``rasterio`` and optionally ``scikit-image``.
"""

from contextlib import ExitStack
import heapq
import math
from typing import List, Sequence, Tuple

import numpy as np
import rasterio

import rsgispy
from rsgispy.imageprocess import (
    create_output_from_dataset,
    ensure_file_exists,
    validate_band_index,
)

try:
    import skimage.graph  # pylint: disable=import-error

    HAVE_SKIMAGE = True
except ImportError:
    HAVE_SKIMAGE = False

__all__ = ["perform_least_cost_path_calc"]

_NEIGHBOURS_8: Tuple[Tuple[int, int, float], ...] = (
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (-1, -1, math.sqrt(2.0)),
    (-1, 1, math.sqrt(2.0)),
    (1, -1, math.sqrt(2.0)),
    (1, 1, math.sqrt(2.0)),
)


def _dijkstra_geometric_path(
    cost_arr: np.ndarray,
    start_rc: Tuple[int, int],
    stop_rc: Tuple[int, int],
) -> List[Tuple[int, int]]:
    """
    Pure-Python 8-connected geometric Dijkstra fallback matching
    ``skimage.graph.route_through_array(..., geometric=True, fully_connected=True)``.
    """
    # pylint: disable=too-many-locals
    height, width = cost_arr.shape
    start_r, start_c = start_rc
    stop_r, stop_c = stop_rc

    if start_rc == stop_rc:
        return [start_rc]

    dist = np.full((height, width), np.inf, dtype=np.float64)
    prev_r = np.full((height, width), -1, dtype=np.int32)
    prev_c = np.full((height, width), -1, dtype=np.int32)
    visited = np.zeros((height, width), dtype=bool)

    dist[start_r, start_c] = 0.0
    heap: List[Tuple[float, int, int]] = [(0.0, start_r, start_c)]

    while heap:
        cur_d, r, c = heapq.heappop(heap)
        if visited[r, c]:
            continue
        visited[r, c] = True
        if r == stop_r and c == stop_c:
            break

        c_u = float(cost_arr[r, c])
        for dr, dc, step_len in _NEIGHBOURS_8:
            nr = r + dr
            nc = c + dc
            if nr < 0 or nr >= height or nc < 0 or nc >= width or visited[nr, nc]:
                continue
            c_v = float(cost_arr[nr, nc])
            if not (math.isfinite(c_u) and math.isfinite(c_v)):
                continue
            alt = cur_d + 0.5 * (c_u + c_v) * step_len
            if alt < dist[nr, nc]:
                dist[nr, nc] = alt
                prev_r[nr, nc] = r
                prev_c[nr, nc] = c
                heapq.heappush(heap, (alt, nr, nc))

    if not math.isfinite(dist[stop_r, stop_c]):
        raise rsgispy.RSGISPyException(
            "No valid least-cost path found between start_coord and stop_coord."
        )

    path: List[Tuple[int, int]] = []
    cr, cc = stop_r, stop_c
    while cr != -1 and cc != -1:
        path.append((cr, cc))
        if cr == start_r and cc == start_c:
            break
        pr = int(prev_r[cr, cc])
        pc = int(prev_c[cr, cc])
        cr, cc = pr, pc
    path.reverse()
    return path


def _coord_to_pixel(
    src: rasterio.DatasetReader,
    coord: Sequence[float],
    label: str,
) -> Tuple[int, int]:
    """Convert spatial ``(x, y)`` coordinate to validated ``(row, col)`` indices."""
    if len(coord) < 2:
        raise rsgispy.RSGISPyException(
            f"{label} must contain at least two values (x, y)."
        )
    row, col = src.index(float(coord[0]), float(coord[1]))
    if row < 0 or row >= src.height or col < 0 or col >= src.width:
        raise rsgispy.RSGISPyException(
            f"{label} {tuple(coord[:2])} maps to pixel ({row}, {col}), "
            f"which is outside image dimensions ({src.height}x{src.width})."
        )
    return int(row), int(col)


def perform_least_cost_path_calc(
    cost_surface_img: str,
    output_img: str,
    start_coord: Sequence[float],
    stop_coord: Sequence[float],
    cost_img_band: int = 1,
    *,
    gdalformat: str = "GTIFF",
) -> None:
    """
    Calculate the least-cost path across a raster cost surface from
    ``start_coord`` to ``stop_coord`` and write a binary ``uint8`` path raster.

    Uses ``skimage.graph.route_through_array`` when ``scikit-image`` is
    installed, falling back to a pure-Python 8-connected Dijkstra geometric
    solver otherwise.

    :param cost_surface_img: Path to the input raster cost surface image.
    :param output_img: Path to the output binary raster image (``1`` on path,
                       ``0`` elsewhere).
    :param start_coord: Start coordinate ``(x, y)`` in the image CRS.
    :param stop_coord: Stop coordinate ``(x, y)`` in the image CRS.
    :param cost_img_band: 1-based band index within ``cost_surface_img`` to use
                          as the cost surface (Default = ``1``).
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).

    .. code:: python

        import rsgispy.imagecalc.leastcostpath

        rsgispy.imagecalc.leastcostpath.perform_least_cost_path_calc(
            cost_surface_img="cost_surface.tif",
            output_img="least_cost_path.tif",
            start_coord=(263155.9, 291809.1),
            stop_coord=(263000.1, 292263.7),
            gdalformat="GTIFF",
        )
    """
    # pylint: disable=too-many-arguments,too-many-locals
    ensure_file_exists(cost_surface_img, "Cost surface image")

    with ExitStack() as stack:
        src = stack.enter_context(rasterio.open(cost_surface_img, "r"))
        validate_band_index(cost_img_band, src.count, cost_surface_img)

        start_rc = _coord_to_pixel(src, start_coord, "start_coord")
        stop_rc = _coord_to_pixel(src, stop_coord, "stop_coord")
        cost_arr = src.read(cost_img_band).astype(np.float64, copy=False)

        if np.any(cost_arr < 0):
            raise rsgispy.RSGISPyException(
                "Cost surface contains negative values, which are unsupported."
            )

        if HAVE_SKIMAGE:
            indices, _ = skimage.graph.route_through_array(
                cost_arr,
                start_rc,
                stop_rc,
                geometric=True,
                fully_connected=True,
            )
            idx_arr = np.asarray(indices, dtype=np.int64)
            path_arr = np.zeros(cost_arr.shape, dtype=np.uint8)
            path_arr[idx_arr[:, 0], idx_arr[:, 1]] = 1
        else:
            path_coords = _dijkstra_geometric_path(cost_arr, start_rc, stop_rc)
            path_arr = np.zeros(cost_arr.shape, dtype=np.uint8)
            for pr, pc in path_coords:
                path_arr[pr, pc] = 1

        dst = create_output_from_dataset(
            stack=stack,
            src=src,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype="uint8",
            count=1,
        )
        dst.write(path_arr, 1)
