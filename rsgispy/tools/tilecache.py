"""
The tools.tilecache module contains utilities for working with XYZ and TMS
Web Mercator tile caches.
"""

from collections.abc import Sequence
import math
import os
from typing import Dict, List, Tuple, Union

import rsgispy
import rsgispy.tools.maths

__all__ = [
    "TILE_SIZE",
    "EARTH_RADIUS",
    "ORIGIN_SHIFT",
    "INITIAL_RESOLUTION",
    "resolution",
    "truncate_lng_lat",
    "get_tile_for_point",
    "get_tile_tl_lonlat",
    "get_tile_bbox",
    "flip_xyz_tms_tiles",
    "get_web_mercator_xy",
    "get_xy_mercator_lonlat",
    "xy_mercator_bounds",
    "get_quadkey_for_tile_path",
    "get_tile_path_from_quadkey",
    "get_tiles",
    "convert_between_tms_xyz",
]

TILE_SIZE = 256
EARTH_RADIUS = 6378137.0
ORIGIN_SHIFT = 2.0 * math.pi * EARTH_RADIUS / 2.0
INITIAL_RESOLUTION = 2.0 * math.pi * EARTH_RADIUS / float(TILE_SIZE)


def resolution(zoom: int) -> float:
    """
    Calculate the pixel resolution (in Web Mercator metres) for a zoom level.

    :param zoom: Zoom level.
    :return: Pixel resolution in metres.

    .. code:: python

        import rsgispy.tools.tilecache
        res_m = rsgispy.tools.tilecache.resolution(zoom=12)
    """
    return INITIAL_RESOLUTION / (2**zoom)


def truncate_lng_lat(lon_deg: float, lat_deg: float) -> Tuple[float, float]:
    """
    Clamp longitude to ``[-180, 180]`` and latitude to ``[-90, 90]``.

    :param lon_deg: Longitude in degrees.
    :param lat_deg: Latitude in degrees.
    :return: Clamped ``(lon_deg, lat_deg)``.

    .. code:: python

        import rsgispy.tools.tilecache
        lon, lat = rsgispy.tools.tilecache.truncate_lng_lat(185.0, -95.0)
    """
    lon_deg = max(-180.0, min(180.0, lon_deg))
    lat_deg = max(-90.0, min(90.0, lat_deg))
    return lon_deg, lat_deg


def get_tile_for_point(
    lon_deg: float,
    lat_deg: float,
    zoom: int,
    tms: bool = True,
    tile_size: int = TILE_SIZE,
) -> Tuple[int, int, int]:
    """
    Return the tile ``(tile_x, tile_y, zoom)`` containing ``(lon_deg, lat_deg)``.

    :param lon_deg: Longitude in degrees.
    :param lat_deg: Latitude in degrees.
    :param zoom: Zoom level.
    :param tms: If ``True``, use TMS origin (bottom-left); if ``False``, XYZ (top-left).
    :param tile_size: Tile size in pixels (Default = ``256``).
    :return: ``(tile_x, tile_y, zoom)``.

    .. code:: python

        import rsgispy.tools.tilecache
        tx, ty, z = rsgispy.tools.tilecache.get_tile_for_point(-4.08, 52.41, zoom=12)
    """
    metre_x = lon_deg * ORIGIN_SHIFT / 180.0
    metre_y = math.log(math.tan((90.0 + lat_deg) * math.pi / 360.0)) / (
        math.pi / 180.0
    )
    metre_y = metre_y * ORIGIN_SHIFT / 180.0

    pixel_x = abs(round((metre_x + ORIGIN_SHIFT) / resolution(zoom)))
    pixel_y = abs(round((metre_y - ORIGIN_SHIFT) / resolution(zoom)))

    tile_x = int(math.ceil(pixel_x / float(tile_size)) - 1)
    tile_y = int(math.ceil(pixel_y / float(tile_size)) - 1)
    if tms:
        tile_y = (2**zoom - 1) - tile_y

    return tile_x, tile_y, zoom


def get_tile_tl_lonlat(
    tile_x: int,
    tile_y: int,
    zoom: int,
    tms: bool = True,
) -> Tuple[float, float]:
    """
    Return the top-left ``(longitude, latitude)`` coordinate for a tile.

    :param tile_x: Tile X index.
    :param tile_y: Tile Y index.
    :param zoom: Zoom level.
    :param tms: If ``True``, interpret ``tile_y`` as TMS; otherwise XYZ.
    :return: ``(lon_deg, lat_deg)``.

    .. code:: python

        import rsgispy.tools.tilecache
        lon, lat = rsgispy.tools.tilecache.get_tile_tl_lonlat(2001, 2745, zoom=12)
    """
    if tms:
        tile_y = (2**zoom - 1) - tile_y
    n_tiles = 2.0**zoom
    lon_deg = tile_x / n_tiles * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * tile_y / n_tiles)))
    lat_deg = math.degrees(lat_rad)
    return (lon_deg, lat_deg)


def get_tile_bbox(
    tile_x: int,
    tile_y: int,
    zoom: int,
    tms: bool = True,
) -> List[float]:
    """
    Return the WGS84 bounding box ``[MinLon, MaxLon, MinLat, MaxLat]`` of a tile.

    :param tile_x: Tile X index.
    :param tile_y: Tile Y index.
    :param zoom: Zoom level.
    :param tms: If ``True``, interpret ``tile_y`` as TMS; otherwise XYZ.
    :return: ``[MinLon, MaxLon, MinLat, MaxLat]``.

    .. code:: python

        import rsgispy.tools.tilecache
        bbox = rsgispy.tools.tilecache.get_tile_bbox(2001, 2745, zoom=12)
    """
    tl_pt = get_tile_tl_lonlat(tile_x, tile_y, zoom, tms)
    br_pt = get_tile_tl_lonlat(tile_x + 1, tile_y + 1, zoom, tms)
    return [tl_pt[0], br_pt[0], br_pt[1], tl_pt[1]]


def flip_xyz_tms_tiles(
    tile_x: int,
    tile_y: int,
    zoom: int,
) -> Tuple[int, int, int]:
    """
    Convert a tile Y index between XYZ and TMS conventions.

    :param tile_x: Tile X index.
    :param tile_y: Tile Y index.
    :param zoom: Zoom level.
    :return: ``(tile_x, flipped_tile_y, zoom)``.

    .. code:: python

        import rsgispy.tools.tilecache
        tx, ty, z = rsgispy.tools.tilecache.flip_xyz_tms_tiles(2001, 2745, zoom=12)
    """
    return tile_x, (2**zoom - 1) - tile_y, zoom


def get_web_mercator_xy(
    lon_deg: float,
    lat_deg: float,
    truncate: bool = False,
) -> Tuple[float, float]:
    """
    Convert WGS84 ``(lon_deg, lat_deg)`` to Web Mercator ``(metre_x, metre_y)``.

    :param lon_deg: Longitude in degrees.
    :param lat_deg: Latitude in degrees.
    :param truncate: Clamp coordinates to valid lon/lat limits first.
    :return: ``(metre_x, metre_y)``.

    .. code:: python

        import rsgispy.tools.tilecache
        mx, my = rsgispy.tools.tilecache.get_web_mercator_xy(-4.08, 52.41)
    """
    if truncate:
        lon_deg, lat_deg = truncate_lng_lat(lon_deg, lat_deg)

    metre_x = lon_deg * ORIGIN_SHIFT / 180.0
    metre_y = math.log(math.tan((90.0 + lat_deg) * math.pi / 360.0)) / (
        math.pi / 180.0
    )
    metre_y = metre_y * ORIGIN_SHIFT / 180.0
    return metre_x, metre_y


def get_xy_mercator_lonlat(
    metre_x: float,
    metre_y: float,
    truncate: bool = False,
) -> Tuple[float, float]:
    """
    Convert Web Mercator ``(metre_x, metre_y)`` to WGS84 ``(lon_deg, lat_deg)``.

    :param metre_x: X coordinate in Web Mercator metres.
    :param metre_y: Y coordinate in Web Mercator metres.
    :param truncate: Clamp output coordinates to valid lon/lat limits.
    :return: ``(lon_deg, lat_deg)``.

    .. code:: python

        import rsgispy.tools.tilecache
        lon, lat = rsgispy.tools.tilecache.get_xy_mercator_lonlat(-454183.0, 6874676.0)
    """
    lon_deg = (metre_x / ORIGIN_SHIFT) * 180.0
    lat_deg = (metre_y / ORIGIN_SHIFT) * 180.0
    lat_deg = (
        180.0
        / math.pi
        * (2 * math.atan(math.exp(lat_deg * math.pi / 180.0)) - math.pi / 2.0)
    )
    if truncate:
        lon_deg, lat_deg = truncate_lng_lat(lon_deg, lat_deg)
    return lon_deg, lat_deg


def xy_mercator_bounds(
    tile_x: int,
    tile_y: int,
    zoom: int,
    tms: bool = True,
) -> Tuple[float, float, float, float]:
    """
    Return the bounding box of a tile in Web Mercator metres ``(MinX, MaxX, MinY, MaxY)``.

    :param tile_x: Tile X index.
    :param tile_y: Tile Y index.
    :param zoom: Zoom level.
    :param tms: If ``True``, interpret ``tile_y`` as TMS; otherwise XYZ.
    :return: ``(x_min, x_max, y_min, y_max)``.

    .. code:: python

        import rsgispy.tools.tilecache
        bbox_m = rsgispy.tools.tilecache.xy_mercator_bounds(2001, 2745, zoom=12)
    """
    bbox = get_tile_bbox(tile_x, tile_y, zoom, tms)
    x_min, y_max = get_web_mercator_xy(bbox[0], bbox[3])
    x_max, y_min = get_web_mercator_xy(bbox[1], bbox[2])
    return (x_min, x_max, y_min, y_max)


def get_quadkey_for_tile_path(
    tile_x: int,
    tile_y: int,
    zoom: int,
    tms: bool = True,
) -> str:
    """
    Return the Bing Maps quadkey string for a tile.

    :param tile_x: Tile X index.
    :param tile_y: Tile Y index.
    :param zoom: Zoom level.
    :param tms: If ``True``, interpret ``tile_y`` as TMS; otherwise XYZ.
    :return: Quadkey string.

    .. code:: python

        import rsgispy.tools.tilecache
        qk = rsgispy.tools.tilecache.get_quadkey_for_tile_path(2001, 2745, zoom=12)
    """
    if tms:
        tile_y = (2**zoom - 1) - tile_y
    qk: List[str] = []
    for z_idx in range(zoom, 0, -1):
        digit = 0
        mask = 1 << (z_idx - 1)
        if tile_x & mask:
            digit += 1
        if tile_y & mask:
            digit += 2
        qk.append(str(digit))
    return "".join(qk)


def get_tile_path_from_quadkey(
    quadkey: str,
    tms: bool = True,
) -> Tuple[int, int, int]:
    """
    Decode a quadkey string into ``(tile_x, tile_y, zoom)``.

    :param quadkey: Input quadkey string.
    :param tms: If ``True``, return ``tile_y`` in TMS; otherwise XYZ.
    :return: ``(tile_x, tile_y, zoom)``.

    .. code:: python

        import rsgispy.tools.tilecache
        tx, ty, z = rsgispy.tools.tilecache.get_tile_path_from_quadkey("031311310201")
    """
    if len(quadkey) == 0:
        return 0, 0, 0
    tile_x, tile_y = 0, 0
    for idx, digit in enumerate(reversed(quadkey)):
        mask = 1 << idx
        if digit == "1":
            tile_x |= mask
        elif digit == "2":
            tile_y |= mask
        elif digit == "3":
            tile_x |= mask
            tile_y |= mask
        elif digit != "0":
            raise rsgispy.RSGISPyException(
                f"Unexpected quadkey digit: {digit}"
            )
    zoom = len(quadkey)
    if tms:
        tile_y = (2**zoom - 1) - tile_y
    return tile_x, tile_y, zoom


def get_tiles(
    bbox: Sequence[float],
    zooms: Union[int, Sequence[int]],
    tms: bool = True,
    truncate: bool = False,
) -> Dict[int, List[Tuple[int, int, int]]]:
    """
    Return tile indices ``(tile_x, tile_y, zoom)`` grouped by zoom level
    intersecting ``bbox`` ``(MinX, MaxX, MinY, MaxY)`` in WGS84 degrees.

    :param bbox: Bounding box ``(MinX, MaxX, MinY, MaxY)`` in WGS84.
    :param zooms: Zoom level or sequence of zoom levels.
    :param tms: If ``True``, return TMS tile indices; otherwise XYZ.
    :param truncate: Clamp coordinates to valid lon/lat bounds.
    :return: Dictionary mapping ``zoom -> [(tile_x, tile_y, zoom), ...]``.

    .. code:: python

        import rsgispy.tools.tilecache

        tiles = rsgispy.tools.tilecache.get_tiles(
            bbox=(-4.2, -4.0, 52.3, 52.5), zooms=[10, 11]
        )
    """
    # pylint: disable=too-many-locals,too-many-branches
    west, east, south, north = bbox[0], bbox[1], bbox[2], bbox[3]
    if truncate:
        west, south = truncate_lng_lat(west, south)
        east, north = truncate_lng_lat(east, north)
    if west > east:
        bboxes = [(-180.0, south, east, north), (west, south, 180.0, north)]
    else:
        bboxes = [(west, south, east, north)]

    zoom_seq = [zooms] if not isinstance(zooms, Sequence) else zooms
    tiles: Dict[int, List[Tuple[int, int, int]]] = {}
    epsilon = 1.0e-9

    for west_b, south_b, east_b, north_b in bboxes:
        west_b = max(-180.0, west_b)
        south_b = max(-85.051129, south_b)
        east_b = min(180.0, east_b)
        north_b = min(85.051129, north_b)

        for z_lvl in zoom_seq:
            if z_lvl not in tiles:
                tiles[z_lvl] = []
            llx, lly, _ = get_tile_for_point(west_b, south_b, z_lvl, tms)
            if lly % 1 < epsilon / 10:
                lly = lly - epsilon

            urx, ury, _ = get_tile_for_point(east_b, north_b, z_lvl, tms)
            if urx % 1 < epsilon / 10:
                urx = urx - epsilon

            llx = max(0, llx)
            ury = max(0, ury)

            llx_i, urx_i, lly_i, ury_i = (
                int(math.floor(val)) for val in (llx, urx, lly, ury)
            )

            min_x_tile = min(llx_i, urx_i)
            if llx_i == urx_i:
                n_x_tiles = 1
            elif llx_i < urx_i:
                n_x_tiles = max(1, urx_i - min(llx_i + 1, 2**z_lvl))
            else:
                n_x_tiles = max(1, llx_i - min(urx_i + 1, 2**z_lvl))

            min_y_tile = min(ury_i, lly_i)
            if ury_i == lly_i:
                n_y_tiles = 1
            elif ury_i < lly_i:
                n_y_tiles = max(1, lly_i - min(ury_i + 1, 2**z_lvl))
            else:
                n_y_tiles = max(1, ury_i - min(lly_i + 1, 2**z_lvl))

            for i_x in range(min_x_tile, min_x_tile + n_x_tiles):
                for j_y in range(min_y_tile, min_y_tile + n_y_tiles):
                    tiles[z_lvl].append((i_x, j_y, z_lvl))
    return tiles


def convert_between_tms_xyz(tiles_dir: str, use_pbar: bool = True) -> None:
    """
    Convert a tile cache directory tree between TMS and XYZ indexing in-situ
    by renaming tile files along the Y axis.

    :param tiles_dir: Root directory of the tile cache (``zoom/x/y.ext``).
    :param use_pbar: If ``True``, display tqdm progress bars during conversion
                     (Default = ``True``).

    .. code:: python

        import rsgispy.tools.tilecache
        rsgispy.tools.tilecache.convert_between_tms_xyz("tiles_cache")
    """
    tqdm_cls = rsgispy.get_tqdm()
    in_zoom_dirs = [
        os.path.abspath(os.path.join(tiles_dir, entry))
        for entry in os.listdir(tiles_dir)
        if os.path.isdir(os.path.join(tiles_dir, entry))
        and rsgispy.tools.maths.is_number(entry)
    ]

    for zoom_dir in tqdm_cls(in_zoom_dirs, desc="overall", disable=not use_pbar):
        zoom = int(os.path.basename(zoom_dir))
        for x_entry in tqdm_cls(
            os.listdir(zoom_dir), desc=f"zoom {zoom}", leave=False, disable=not use_pbar
        ):
            x_dir = os.path.join(zoom_dir, x_entry)
            if os.path.isdir(x_dir) and rsgispy.tools.maths.is_number(x_entry):
                x_val = int(x_entry)
                for y_file in tqdm_cls(
                    os.listdir(x_dir), desc="y", leave=False, disable=not use_pbar
                ):
                    y_cur_file = os.path.join(x_dir, y_file)
                    y_in_str, img_ext = os.path.splitext(y_file)
                    if rsgispy.tools.maths.is_number(y_in_str):
                        _, n_tile_y, _ = flip_xyz_tms_tiles(
                            x_val, int(y_in_str), zoom
                        )
                        out_img_file = os.path.join(x_dir, f"{n_tile_y}{img_ext}")
                        if y_cur_file != out_img_file:
                            os.rename(y_cur_file, out_img_file)
