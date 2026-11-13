"""
The imageutils.tiling module provides functions for splitting raster images into
tiles using ``rasterio``.
"""

import math
import multiprocessing
import os
from typing import List, Tuple

import rasterio

import rsgispy
from rsgispy.imageutils._helpers import copy_raster_window
from rsgispy.imageutils.blank import create_blank_img_from_bbox
from rsgispy.tools.files import does_path_exists_or_creatable

__all__ = [
    "create_tiles",
    "create_tiles_multi_core",
    "create_raster_tiles_bbox",
]


def _extract_tile_worker(
    task: Tuple[str, str, int, int, int, int, str, int]
) -> None:
    """Top-level picklable worker for :func:`create_tiles_multi_core`."""
    copy_raster_window(*task)


def _compute_axis_slices(
    img_len: int, tile_len: int, overlap: int, offset: int
) -> List[Tuple[int, int]]:
    """Compute clipped ``(start_pxl, end_pxl)`` ranges covering ``[0, img_len)``."""
    ranges: List[Tuple[int, int]] = []
    if offset > 0:
        first_end = min(img_len, offset + overlap)
        if first_end > 0:
            ranges.append((0, first_end))
        pos = offset
    else:
        pos = 0

    while pos < img_len:
        start = max(0, pos - overlap)
        end = min(img_len, pos + tile_len + overlap)
        if end > start:
            ranges.append((start, end))
        pos += tile_len

    return ranges


def create_tiles(
    input_img: str,
    out_img_base: str,
    tile_width: int,
    tile_height: int,
    tile_overlap: int,
    offset_tiles: bool,
    gdalformat: str,
    datatype: int,
    out_img_ext: str,
) -> List[str]:
    """
    Split ``input_img`` into a grid of overlapping/non-overlapping tile files.

    :param input_img: Path to the input image file.
    :param out_img_base: Base path and filename prefix for output tiles.
    :param tile_width: Core tile width in pixels.
    :param tile_height: Core tile height in pixels.
    :param tile_overlap: Overlap in pixels around each tile edge.
    :param offset_tiles: If ``True``, offset the tile grid by half a tile size.
    :param gdalformat: Output GDAL format name (e.g., ``'KEA'``, ``'GTIFF'``).
    :param datatype: ``rsgispy.TYPE_*`` constant for the output tiles.
    :param out_img_ext: Output file extension (e.g., ``'tif'`` or ``'kea'``).
    :return: List of created tile file paths.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    if not os.path.exists(input_img):
        raise rsgispy.RSGISPyException(
            f"Input image file does not exist: '{input_img}'."
        )
    if not does_path_exists_or_creatable(out_img_base):
        raise rsgispy.RSGISPyException(
            f"Output base directory is not writable or does not exist: '{out_img_base}'."
        )
    if int(tile_width) <= 0 or int(tile_height) <= 0:
        raise rsgispy.RSGISPyException(
            "tile_width and tile_height must be positive integers."
        )
    if int(tile_overlap) < 0:
        raise rsgispy.RSGISPyException(
            "tile_overlap must be a non-negative integer."
        )

    ext = out_img_ext.lstrip(".")
    t_w = int(tile_width)
    t_h = int(tile_height)
    overlap = int(tile_overlap)
    x_off = (t_w // 2) if offset_tiles else 0
    y_off = (t_h // 2) if offset_tiles else 0

    with rasterio.open(input_img, "r") as src:
        img_w = int(src.width)
        img_h = int(src.height)

    x_ranges = _compute_axis_slices(img_w, t_w, overlap, x_off)
    y_ranges = _compute_axis_slices(img_h, t_h, overlap, y_off)

    out_files: List[str] = []
    tile_idx = 0
    for r0, r1 in y_ranges:
        for c0, c1 in x_ranges:
            out_file = f"{out_img_base}_tile{tile_idx}.{ext}"
            copy_raster_window(
                input_img=input_img,
                output_img=out_file,
                col_off=c0,
                row_off=r0,
                width=c1 - c0,
                height=r1 - r0,
                gdalformat=gdalformat,
                datatype=datatype,
            )
            out_files.append(out_file)
            tile_idx += 1

    return out_files


def create_tiles_multi_core(
    input_img: str,
    out_img_base: str,
    width: int,
    height: int,
    gdalformat: str,
    datatype: int,
    out_img_ext: str,
    n_cores: int = 1,
) -> None:
    """
    Create non-overlapping raster tiles using single or multiple CPU cores, named
    ``{out_img_base}_x{ix}y{iy}.{out_img_ext}`` (1-based indices).

    :param input_img: Path to the input image file.
    :param out_img_base: Base path and filename prefix for output tiles.
    :param width: Tile width in pixels.
    :param height: Tile height in pixels.
    :param gdalformat: Output GDAL format name (e.g., ``'KEA'``, ``'GTIFF'``).
    :param datatype: ``rsgispy.TYPE_*`` constant for the output tiles.
    :param out_img_ext: Output file extension (e.g., ``'tif'`` or ``'kea'``).
    :param n_cores: Number of worker processes (Default = ``1``).
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    if not os.path.exists(input_img):
        raise rsgispy.RSGISPyException(
            f"Input image file does not exist: '{input_img}'."
        )
    if not does_path_exists_or_creatable(out_img_base):
        raise rsgispy.RSGISPyException(
            f"Output base path is not creatable: '{out_img_base}'."
        )
    if int(width) <= 0 or int(height) <= 0:
        raise rsgispy.RSGISPyException(
            "width and height must be positive integers."
        )

    tile_w = int(width)
    tile_h = int(height)
    ext = out_img_ext.lstrip(".")

    with rasterio.open(input_img, "r") as src:
        img_w = int(src.width)
        img_h = int(src.height)

    n_x = math.ceil(img_w / tile_w)
    n_y = math.ceil(img_h / tile_h)

    tasks: List[Tuple[str, str, int, int, int, int, str, int]] = []
    for iy in range(n_y):
        row_off = iy * tile_h
        cur_h = min(tile_h, img_h - row_off)
        for ix in range(n_x):
            col_off = ix * tile_w
            cur_w = min(tile_w, img_w - col_off)
            out_file = f"{out_img_base}_x{ix + 1}y{iy + 1}.{ext}"
            tasks.append(
                (
                    input_img,
                    out_file,
                    col_off,
                    row_off,
                    cur_w,
                    cur_h,
                    gdalformat,
                    datatype,
                )
            )

    if int(n_cores) > 1 and len(tasks) > 1:
        with multiprocessing.Pool(processes=int(n_cores)) as pool:
            pool.map(_extract_tile_worker, tasks)
    else:
        for task in tasks:
            _extract_tile_worker(task)


def create_raster_tiles_bbox(
    bbox: Tuple[float, float, float, float],
    tile_x_size: float,
    tile_y_size: float,
    out_img_res: float,
    out_epsg: int,
    out_tile_dir: str,
    out_tile_name: str = "base_tile_{}",
    out_img_ext: str = "kea",
    gdalformat: str = "GTIFF",
    datatype: int = rsgispy.TYPE_8UINT,
    out_img_pxl_val: float = 0.0,
    out_img_n_bands: int = 1,
) -> None:
    """
    Create a regular grid of blank raster tiles covering ``bbox``.

    :param bbox: Bounding box ``(xMin, xMax, yMin, yMax)``.
    :param tile_x_size: Tile width in spatial units.
    :param tile_y_size: Tile height in spatial units.
    :param out_img_res: Output pixel resolution.
    :param out_epsg: EPSG code for the output tiles.
    :param out_tile_dir: Existing directory where tiles will be written.
    :param out_tile_name: Format template containing ``{}`` for tile ``x_y``
                          index (Default = ``'base_tile_{}'``).
    :param out_img_ext: Output file extension (Default = ``'kea'``).
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param datatype: ``rsgispy.TYPE_*`` constant (Default = ``rsgispy.TYPE_8UINT``).
    :param out_img_pxl_val: Initial pixel value (Default = ``0.0``).
    :param out_img_n_bands: Number of bands in each tile (Default = ``1``).
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    x_min, x_max, y_min, y_max = (float(v) for v in bbox[:4])
    bbox_width = x_max - x_min
    bbox_height = y_max - y_min
    if bbox_width < 0 or bbox_height < 0:
        raise rsgispy.RSGISPyException(
            "Invalid bbox: width and height must be non-negative."
        )
    if tile_x_size <= 0 or tile_y_size <= 0 or out_img_res <= 0:
        raise rsgispy.RSGISPyException(
            "tile_x_size, tile_y_size, and out_img_res must be positive."
        )
    if not os.path.isdir(out_tile_dir):
        raise rsgispy.RSGISPyException(
            f"Output tile directory does not exist: '{out_tile_dir}'."
        )

    ext = out_img_ext.lstrip(".")
    n_x_tiles = max(1, math.ceil(bbox_width / float(tile_x_size)))
    n_y_tiles = max(1, math.ceil(bbox_height / float(tile_y_size)))

    for y_idx in range(n_y_tiles):
        c_max_y = y_max - (y_idx * float(tile_y_size))
        c_min_y = c_max_y - float(tile_y_size)
        for x_idx in range(n_x_tiles):
            c_min_x = x_min + (x_idx * float(tile_x_size))
            c_max_x = c_min_x + float(tile_x_size)
            tile_bbox = (c_min_x, c_max_x, c_min_y, c_max_y)
            tile_base = out_tile_name.format(f"{x_idx}_{y_idx}")
            out_img = os.path.join(out_tile_dir, f"{tile_base}.{ext}")
            create_blank_img_from_bbox(
                bbox=tile_bbox,
                epsg_code=out_epsg,
                output_img=out_img,
                out_img_res=out_img_res,
                out_img_pxl_val=out_img_pxl_val,
                out_img_n_bands=out_img_n_bands,
                gdalformat=gdalformat,
                datatype=datatype,
                snap_to_grid=False,
            )
