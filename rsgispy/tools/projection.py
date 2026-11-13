"""
The tools.projection module provides coordinate reference system conversion and
geodesic/ellipsoid distance utilities using ``pyproj`` and ``numpy``.
"""

import math
from typing import Optional, Sequence, Tuple, Union

import numpy
import pyproj

import rsgispy
from rsgispy.tools.geometrytools import degrees_to_metres
import rsgispy.tools.utils

__all__ = [
    "get_epsg_code_from_wkt",
    "get_wkt_from_epsg_code",
    "get_crs_obj",
    "degrees_to_metres",
    "metres_to_degrees",
    "get_deg_coord_as_str",
    "great_circle_distance",
]


def get_epsg_code_from_wkt(wkt_str: str) -> Optional[int]:
    """
    Return the integer EPSG code for a WKT projection string using ``pyproj``.

    :param wkt_str: Projection WKT string.
    :return: Integer EPSG code, or ``None`` if it cannot be identified.
    """
    try:
        crs = pyproj.CRS.from_wkt(wkt_str)
        epsg_code = crs.to_epsg()
        return int(epsg_code) if epsg_code is not None else None
    except Exception:  # pylint: disable=broad-exception-caught
        return None


def get_wkt_from_epsg_code(epsg_code: int) -> Optional[str]:
    """
    Return the WKT projection string for an integer EPSG code using ``pyproj``.

    :param epsg_code: Integer EPSG code.
    :return: WKT string representation, or ``None`` if lookup fails.
    """
    try:
        crs = pyproj.CRS.from_epsg(int(epsg_code))
        return crs.to_wkt()
    except Exception:  # pylint: disable=broad-exception-caught
        return None


def get_crs_obj(epsg_code: int) -> pyproj.CRS:
    """
    Return a :class:`pyproj.CRS` object for a given integer EPSG code.

    :param epsg_code: Integer EPSG code for the projection.
    :return: :class:`pyproj.CRS` object.
    """
    try:
        return pyproj.CRS.from_epsg(int(epsg_code))
    except Exception as err:
        raise rsgispy.RSGISPyException(
            f"Could not create CRS for EPSG:{epsg_code}: {err}"
        ) from err


def metres_to_degrees(
    latitude: Union[float, numpy.ndarray],
    x_size: Union[float, numpy.ndarray],
    y_size: Union[float, numpy.ndarray],
) -> Tuple[Union[float, numpy.ndarray], Union[float, numpy.ndarray]]:
    """
    Convert pixel sizes or distances in metres (WGS84 ellipsoid) to degrees.

    :param latitude: Latitude in degrees (scalar or array).
    :param x_size: X sizes/distances in metres (scalar or array).
    :param y_size: Y sizes/distances in metres (scalar or array).
    :return: Tuple ``(lon_size_degrees, lat_size_degrees)``.
    """
    semi_major = 6378137.0
    semi_minor = 6356752.314245
    rad_lat = numpy.deg2rad(latitude)
    r_sq = (semi_major * numpy.cos(rad_lat)) ** 2 + (
        semi_minor * numpy.sin(rad_lat)
    ) ** 2
    m_lat = ((semi_major * semi_minor) ** 2) / (r_sq**1.5)
    n_lon = (semi_major**2) / numpy.sqrt(r_sq)
    deg_to_rad = math.pi / 180.0
    lon_size = x_size / (deg_to_rad * numpy.cos(rad_lat) * n_lon)
    lat_size = y_size / (deg_to_rad * m_lat)
    return lon_size, lat_size


def get_deg_coord_as_str(lat: float, lon: float, n_chars: int = 4) -> str:
    """
    Produce a zero-padded hemisphere string representation of a latitude and
    longitude coordinate in degrees (e.g., ``'n524w0041'``).

    :param lat: Latitude in degrees.
    :param lon: Longitude in degrees.
    :param n_chars: Zero-padded character width for each coordinate component.
    :return: Formatted coordinate string.
    """
    east_west = "w" if lon < 0 else "e"
    north_south = "s" if lat < 0 else "n"

    lat_str = rsgispy.tools.utils.zero_pad_num_str(
        lat,
        str_len=n_chars,
        round_num=True,
        round_n_digts=1,
        integerise=True,
        absolute=True,
        gain=10,
    )
    lon_str = rsgispy.tools.utils.zero_pad_num_str(
        lon,
        str_len=n_chars,
        round_num=True,
        round_n_digts=1,
        integerise=True,
        absolute=True,
        gain=10,
    )

    return f"{north_south}{lat_str}{east_west}{lon_str}"


def great_circle_distance(
    wgs84_p1: Sequence[float],
    wgs84_p2: Sequence[float],
    earth_radius: float = 6378137.0,
) -> float:
    """
    Calculate the great-circle distance between two WGS84 points given in degrees.

    :param wgs84_p1: First point ``[lon (x), lat (y)]``.
    :param wgs84_p2: Second point ``[lon (x), lat (y)]``.
    :param earth_radius: Earth radius defining the unit of the returned distance
                         (Default = ``6378137.0`` metres).
    :return: Great-circle distance between the two points.
    """
    if (wgs84_p1[1] < -90.0) or (wgs84_p1[1] > 90.0):
        raise rsgispy.RSGISPyException(
            "Point 1 latitude is not within valid range (-90 - 90)"
        )
    if (wgs84_p2[1] < -90.0) or (wgs84_p2[1] > 90.0):
        raise rsgispy.RSGISPyException(
            "Point 2 latitude is not within valid range (-90 - 90)"
        )
    if (wgs84_p1[0] < -180.0) or (wgs84_p1[0] > 180.0):
        raise rsgispy.RSGISPyException(
            "Point 1 longitude is not within valid range (-180 - 180)"
        )
    if (wgs84_p2[0] < -180.0) or (wgs84_p2[0] > 180.0):
        raise rsgispy.RSGISPyException(
            "Point 2 longitude is not within valid range (-180 - 180)"
        )

    lon1_rad, lat1_rad = math.radians(wgs84_p1[0]), math.radians(wgs84_p1[1])
    lon2_rad, lat2_rad = math.radians(wgs84_p2[0]), math.radians(wgs84_p2[1])
    lon_abs_diff = abs(lon1_rad - lon2_rad)

    cos_angle = (
        math.sin(lat1_rad) * math.sin(lat2_rad)
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.cos(lon_abs_diff)
    )
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return earth_radius * math.acos(cos_angle)
