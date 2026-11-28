"""
The tools.utm module provides UTM zone lookups, WGS84/UTM coordinate
conversions, and bounding box splitting utilities.

Adapted from https://github.com/Turbo87/utm/blob/master/utm/conversion.py.
"""

import re
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import scipy.stats

    HAVE_SCIPY = True
except ImportError:
    scipy = None  # type: ignore[assignment]
    HAVE_SCIPY = False

import rsgispy
import rsgispy.tools.maths

__all__ = [
    "K0",
    "E",
    "E2",
    "E3",
    "E_P2",
    "SQRT_E",
    "M1",
    "M2",
    "M3",
    "M4",
    "P2",
    "P3",
    "P4",
    "P5",
    "R",
    "ZONE_LETTERS",
    "min_max_eastings_for_lat",
    "parse_utm_zone_str",
    "epsg_for_utm",
    "utm_from_epsg",
    "split_wgs84_bbox_utm_zones",
    "check_valid_zone",
    "to_latlon",
    "from_latlon",
    "latitude_to_zone_letter",
    "latlon_to_zone_number",
    "latlon_arr_to_utm_zone_number",
    "latlon_to_mode_utm_zone_number",
    "zone_number_to_central_longitude",
]

K0 = 0.9996

E = 0.00669438
E2 = E * E
E3 = E2 * E
E_P2 = E / (1.0 - E)

SQRT_E = float(np.sqrt(1 - E))
_E = (1 - SQRT_E) / (1 + SQRT_E)
_E2 = _E * _E
_E3 = _E2 * _E
_E4 = _E3 * _E
_E5 = _E4 * _E

M1 = 1 - E / 4 - 3 * E2 / 64 - 5 * E3 / 256
M2 = 3 * E / 8 + 3 * E2 / 32 + 45 * E3 / 1024
M3 = 15 * E2 / 256 + 45 * E3 / 1024
M4 = 35 * E3 / 3072

P2 = 3.0 / 2 * _E - 27.0 / 32 * _E3 + 269.0 / 512 * _E5
P3 = 21.0 / 16 * _E2 - 55.0 / 32 * _E4
P4 = 151.0 / 96 * _E3 - 417.0 / 128 * _E5
P5 = 1097.0 / 512 * _E4

R = 6378137

ZONE_LETTERS = "CDEFGHJKLMNPQRSTUVWXX"
_UTM_ZONE_RE = re.compile(r"^(\d{1,2})([NSns])$")


def min_max_eastings_for_lat(latitude: float, zone: int = 1) -> Tuple[float, float]:
    """
    Find the minimum and maximum Eastings for a specified latitude within a UTM zone.

    :param latitude: Latitude in decimal degrees (scalar).
    :param zone: UTM zone number (Default = ``1``).
    :return: Tuple ``(min_easting, max_easting)``.

    .. code:: python

        import rsgispy.tools.utm
        min_e, max_e = rsgispy.tools.utm.min_max_eastings_for_lat(52.4, zone=30)
    """
    if isinstance(latitude, np.ndarray):
        raise rsgispy.RSGISPyException("Don't input an array.")

    central_long = zone_number_to_central_longitude(zone)
    min_long = central_long - 3.0
    max_long = central_long + 3.0

    min_utm = from_latlon(latitude, min_long, force_zone_number=zone)
    max_utm = from_latlon(latitude, max_long, force_zone_number=zone)

    return float(min_utm[0]), float(max_utm[0])


def parse_utm_zone_str(utm_str: str) -> Tuple[int, str]:
    """
    Parse a WGS84 UTM zone specification string such as ``'30N'`` or ``'01S'``
    into ``(zone, hemisphere)``.

    :param utm_str: UTM zone string matching ``<zone><N|S>`` (zone ``1..60``).
    :return: Tuple ``(zone_number, hemisphere)`` where ``hemisphere`` is
             ``'N'`` or ``'S'``.

    .. code:: python

        import rsgispy.tools.utm
        zone, hemi = rsgispy.tools.utm.parse_utm_zone_str("30N")
    """
    match = _UTM_ZONE_RE.match(utm_str.strip())
    if not match:
        raise rsgispy.RSGISPyException(
            f"Invalid UTM specification '{utm_str}': expected <zone><N|S> "
            "(e.g., 36N or 01S)."
        )
    zone = int(match.group(1))
    hemi = match.group(2).upper()
    if not 1 <= zone <= 60:
        raise rsgispy.RSGISPyException(
            f"UTM zone must be between 1 and 60 (got {zone})."
        )
    return zone, hemi


def epsg_for_utm(zone: int, hemisphere: str) -> int:
    """
    Return the WGS84 EPSG code for a given UTM ``zone`` and ``hemisphere``.

    :param zone: UTM zone number (0..60).
    :param hemisphere: Hemisphere indicator (``'N'`` or ``'S'``).
    :return: Corresponding EPSG integer code.

    .. code:: python

        import rsgispy.tools.utm
        epsg = rsgispy.tools.utm.epsg_for_utm(zone=30, hemisphere="N")
    """
    if hemisphere not in ["N", "S"]:
        raise rsgispy.RSGISPyException('Invalid hemisphere ("N" or "S").')

    if zone < 0 or zone > 60:
        raise rsgispy.RSGISPyException("UTM zone outside valid range.")

    ns_offset = 600 if hemisphere == "N" else 700
    if zone == 0:
        zone = 61

    return int(32000 + ns_offset + zone)


def utm_from_epsg(epsg_code: int) -> Tuple[int, str]:
    """
    Return ``(zone, hemisphere)`` from a WGS84 UTM EPSG code.

    :param epsg_code: EPSG integer code for the UTM projection.
    :return: Tuple ``(zone, hemisphere)``.
    :raises rsgispy.RSGISPyException: if ``epsg_code`` is not 32601-32660 or
                                      32701-32760.

    .. code:: python

        import rsgispy.tools.utm
        zone, hemi = rsgispy.tools.utm.utm_from_epsg(32630)
    """
    code = int(epsg_code)
    if 32601 <= code <= 32660:
        return code - 32600, "N"
    if 32701 <= code <= 32760:
        return code - 32700, "S"
    raise rsgispy.RSGISPyException(
        f"EPSG:{code} is not a WGS84 UTM zone (32601-32660 for the north, "
        "32701-32760 for the south)."
    )


#: Latitude bands where UTM zones depart from the regular 6 degree grid:
#: ``(lat_min, lat_max, [(zone, lon_min, lon_max), ...])``. Longitudes
#: outside the listed ranges use the regular zones.
_UTM_ZONE_EXCEPTIONS = (
    (56.0, 64.0, [(31, 0.0, 3.0), (32, 3.0, 12.0)]),  # Norway
    (72.0, 84.0, [(31, 0.0, 9.0), (33, 9.0, 21.0), (35, 21.0, 33.0),
                  (37, 33.0, 42.0)]),  # Svalbard
)


def _utm_zone_intervals(lat_min: float, lat_max: float) -> List[Tuple[int, float, float]]:
    """``(zone, lon_min, lon_max)`` of every zone in a latitude band that
    lies entirely inside, or entirely outside, the exception bands."""
    intervals = [
        (zone, ((zone * 6.0) - 6.0) - 180.0, (zone * 6.0) - 180.0)
        for zone in range(1, 61)
    ]
    for ex_min, ex_max, ex_zones in _UTM_ZONE_EXCEPTIONS:
        if lat_min >= ex_min and lat_max <= ex_max:
            ex_lo = min(z[1] for z in ex_zones)
            ex_hi = max(z[2] for z in ex_zones)
            intervals = [iv for iv in intervals if iv[2] <= ex_lo or iv[1] >= ex_hi]
            intervals += list(ex_zones)
    return sorted(intervals, key=lambda iv: iv[1])


def split_wgs84_bbox_utm_zones(
    wgs84_bbox: Sequence[float],
) -> List[List]:
    """
    Split a WGS84 bounding box ``(MinX, MaxX, MinY, MaxY)`` at UTM zone
    boundaries, including the Norway (zone 32 widened to 3-12 E between 56
    and 64 N) and Svalbard (zones 31, 33, 35 and 37 between 72 and 84 N)
    exceptions, so a bbox is also split at those latitudes when it crosses
    them.

    A bbox with ``MinX > MaxX`` crosses the antimeridian: it is split into
    ``[MinX, 180]`` and ``[-180, MaxX]`` (zones 60 and 1 meet at 180).

    :param wgs84_bbox: Bounding box in WGS84 lon/lat ``(MinX, MaxX, MinY, MaxY)``.
    :return: List of ``[utm_zone, [x_min, x_max, y_min, y_max]]`` entries,
             ordered by latitude band, then west to east.

    .. code:: python

        import rsgispy.tools.utm
        zone_bboxes = rsgispy.tools.utm.split_wgs84_bbox_utm_zones((-8.0, 2.0, 50.0, 55.0))
    """
    x_min, x_max, y_min, y_max = (float(v) for v in wgs84_bbox[:4])
    if x_min > x_max:
        return split_wgs84_bbox_utm_zones(
            (x_min, 180.0, y_min, y_max)
        ) + split_wgs84_bbox_utm_zones((-180.0, x_max, y_min, y_max))

    breaks = sorted(
        {y_min, y_max}
        | {
            lat
            for ex_min, ex_max, _ in _UTM_ZONE_EXCEPTIONS
            for lat in (ex_min, ex_max)
            if y_min < lat < y_max
        }
    )
    out_bboxes: List[List] = []
    for b_min, b_max in zip(breaks[:-1], breaks[1:]):
        for zone, z_min, z_max in _utm_zone_intervals(b_min, b_max):
            lo = max(x_min, z_min)
            hi = min(x_max, z_max)
            if hi > lo or (x_min == x_max and z_min <= x_min < z_max):
                out_bboxes.append([zone, [lo, hi, b_min, b_max]])
    return out_bboxes


def check_valid_zone(
    zone_number: int,
    zone_letter: Optional[str],
    use_exp: bool = True,
) -> bool:
    """
    Check whether a UTM zone number (1..60) and optional zone letter (C..X, excluding I/O)
    are valid.

    :param zone_number: UTM zone number.
    :param zone_letter: Optional UTM zone designator letter.
    :param use_exp: If ``True``, raise :class:`rsgispy.RSGISPyException` on invalid values.
    :return: ``True`` if valid, ``False`` otherwise (when ``use_exp=False``).

    .. code:: python

        import rsgispy.tools.utm
        is_valid = rsgispy.tools.utm.check_valid_zone(30, "U", use_exp=False)
    """
    valid_zone = True
    if not 1 <= zone_number <= 60:
        if use_exp:
            raise rsgispy.RSGISPyException(
                "zone number out of range (must be between 1 and 60)"
            )
        valid_zone = False

    if zone_letter:
        zone_letter = zone_letter.upper()
        if not "C" <= zone_letter <= "X" or zone_letter in ["I", "O"]:
            if use_exp:
                raise rsgispy.RSGISPyException(
                    "zone letter out of range (must be between C and X)"
                )
            valid_zone = False

    return valid_zone


def to_latlon(
    easting: Union[float, np.ndarray],
    northing: Union[float, np.ndarray],
    zone_number: int,
    zone_letter: Optional[str] = None,
    northern: Optional[bool] = None,
    strict: bool = True,
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Convert UTM coordinates to WGS84 latitude and longitude.

    :param easting: Easting coordinate (scalar or array).
    :param northing: Northing coordinate (scalar or array).
    :param zone_number: UTM zone number (1..60).
    :param zone_letter: Optional UTM zone letter (mutually exclusive with ``northern``).
    :param northern: Hemisphere flag (``True`` for Northern, ``False`` for Southern).
    :param strict: Validate easting/northing bounds (Default = ``True``).
    :return: Tuple ``(latitude, longitude)``.

    .. code:: python

        import rsgispy.tools.utm
        lat, lon = rsgispy.tools.utm.to_latlon(426500.0, 5807000.0, 30, zone_letter="U")
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    if not zone_letter and northern is None:
        raise rsgispy.RSGISPyException(
            "either zone_letter or northern needs to be set"
        )
    if zone_letter and northern is not None:
        raise rsgispy.RSGISPyException(
            "set either zone_letter or northern, but not both"
        )

    if strict:
        if not rsgispy.tools.maths.in_bounds(
            easting, 100000, 1000000, upper_strict=True
        ):
            raise rsgispy.RSGISPyException(
                "easting out of range (must be between 100.000 m and 999.999 m)"
            )
        if not rsgispy.tools.maths.in_bounds(northing, 0, 10000000):
            raise rsgispy.RSGISPyException(
                "northing out of range (must be between 0 m and 10.000.000 m)"
            )

    check_valid_zone(zone_number, zone_letter)

    if zone_letter:
        zone_letter = zone_letter.upper()
        northern = zone_letter >= "N"

    x_coord = easting - 500000
    y_coord = northing if northern else northing - 10000000

    merid_arc = y_coord / K0
    mu_val = merid_arc / (R * M1)

    p_rad = (
        mu_val
        + P2 * np.sin(2 * mu_val)
        + P3 * np.sin(4 * mu_val)
        + P4 * np.sin(6 * mu_val)
        + P5 * np.sin(8 * mu_val)
    )

    p_sin = np.sin(p_rad)
    p_sin2 = p_sin * p_sin
    p_cos = np.cos(p_rad)

    p_tan = p_sin / p_cos
    p_tan2 = p_tan * p_tan
    p_tan4 = p_tan2 * p_tan2

    ep_sin = 1 - E * p_sin2
    ep_sin_sqrt = np.sqrt(ep_sin)

    n_rad = R / ep_sin_sqrt
    r_rad = (1 - E) / ep_sin

    c_val = _E * p_cos**2
    c2_val = c_val * c_val

    d_val = x_coord / (n_rad * K0)
    d2 = d_val * d_val
    d3 = d2 * d_val
    d4 = d3 * d_val
    d5 = d4 * d_val
    d6 = d5 * d_val

    latitude = (
        p_rad
        - (p_tan / r_rad)
        * (d2 / 2 - d4 / 24 * (5 + 3 * p_tan2 + 10 * c_val - 4 * c2_val - 9 * E_P2))
        + d6
        / 720
        * (61 + 90 * p_tan2 + 298 * c_val + 45 * p_tan4 - 252 * E_P2 - 3 * c2_val)
    )

    longitude = (
        d_val
        - d3 / 6 * (1 + 2 * p_tan2 + c_val)
        + d5
        / 120
        * (5 - 2 * c_val + 28 * p_tan2 - 3 * c2_val + 8 * E_P2 + 24 * p_tan4)
    ) / p_cos

    return (
        np.degrees(latitude),
        np.degrees(longitude) + zone_number_to_central_longitude(zone_number),
    )


def from_latlon(
    latitude: Union[float, np.ndarray],
    longitude: Union[float, np.ndarray],
    force_zone_number: Optional[int] = None,
    force_zone_letter: Optional[str] = None,
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray], int, Optional[str]]:
    """
    Convert WGS84 latitude and longitude coordinates to UTM coordinates.

    :param latitude: Latitude in decimal degrees (-80.0 to 84.0).
    :param longitude: Longitude in decimal degrees (-180.0 to 180.0).
    :param force_zone_number: Optional forced UTM zone number.
    :param force_zone_letter: Optional forced UTM zone letter.
    :return: Tuple ``(eastings, northings, zone_number, zone_letter)``.

    .. code:: python

        import rsgispy.tools.utm
        east, north, zone, letter = rsgispy.tools.utm.from_latlon(52.41, -4.08)
    """
    # pylint: disable=too-many-locals
    if not rsgispy.tools.maths.in_bounds(latitude, -80.0, 84.0):
        raise rsgispy.RSGISPyException(
            "latitude out of range (must be between 80 deg S and 84 deg N)"
        )
    if not rsgispy.tools.maths.in_bounds(longitude, -180.0, 180.0):
        raise rsgispy.RSGISPyException(
            "longitude out of range (must be between 180 deg W and 180 deg E)"
        )
    if force_zone_number is not None:
        check_valid_zone(force_zone_number, force_zone_letter)

    lat_rad = np.radians(latitude)
    lat_sin = np.sin(lat_rad)
    lat_cos = np.cos(lat_rad)

    lat_tan = lat_sin / lat_cos
    lat_tan2 = lat_tan * lat_tan
    lat_tan4 = lat_tan2 * lat_tan2

    zone_number = (
        latlon_to_zone_number(latitude, longitude)
        if force_zone_number is None
        else force_zone_number
    )
    zone_letter = (
        latitude_to_zone_letter(latitude)
        if force_zone_letter is None
        else force_zone_letter
    )

    lon_rad = np.radians(longitude)
    central_lon = zone_number_to_central_longitude(zone_number)
    central_lon_rad = np.radians(central_lon)

    n_rad = R / np.sqrt(1 - E * lat_sin**2)
    c_val = E_P2 * lat_cos**2

    a_val = lat_cos * (lon_rad - central_lon_rad)
    a2 = a_val * a_val
    a3 = a2 * a_val
    a4 = a3 * a_val
    a5 = a4 * a_val
    a6 = a5 * a_val

    m_arc = R * (
        M1 * lat_rad
        - M2 * np.sin(2 * lat_rad)
        + M3 * np.sin(4 * lat_rad)
        - M4 * np.sin(6 * lat_rad)
    )

    easting = (
        K0
        * n_rad
        * (
            a_val
            + a3 / 6 * (1 - lat_tan2 + c_val)
            + a5 / 120 * (5 - 18 * lat_tan2 + lat_tan4 + 72 * c_val - 58 * E_P2)
        )
        + 500000
    )

    northing = K0 * (
        m_arc
        + n_rad
        * lat_tan
        * (
            a2 / 2
            + a4 / 24 * (5 - lat_tan2 + 9 * c_val + 4 * c_val**2)
            + a6
            / 720
            * (61 - 58 * lat_tan2 + lat_tan4 + 600 * c_val - 330 * E_P2)
        )
    )

    if rsgispy.tools.maths.mixed_signs(latitude):
        raise rsgispy.RSGISPyException("latitudes must all have the same sign")
    if rsgispy.tools.maths.negative(latitude):
        northing += 10000000

    return easting, northing, zone_number, zone_letter


def latitude_to_zone_letter(
    latitude: Union[float, np.ndarray],
) -> Optional[str]:
    """
    Find the UTM zone letter for ``latitude``.

    :param latitude: Latitude in decimal degrees (if an array, uses first element).
    :return: Zone letter string, or ``None`` if outside [-80, 84].

    .. code:: python

        import rsgispy.tools.utm
        letter = rsgispy.tools.utm.latitude_to_zone_letter(52.41)
    """
    if isinstance(latitude, np.ndarray):
        latitude = float(latitude.flat[0])

    if -80 <= latitude <= 84:
        return ZONE_LETTERS[int(latitude + 80) >> 3]
    return None


def latlon_to_zone_number(
    latitude: Union[float, np.ndarray],
    longitude: Union[float, np.ndarray],
) -> int:
    """
    Find the UTM zone number for a given latitude and longitude.

    :param latitude: Latitude in decimal degrees.
    :param longitude: Longitude in decimal degrees.
    :return: Integer UTM zone number.

    .. code:: python

        import rsgispy.tools.utm
        zone = rsgispy.tools.utm.latlon_to_zone_number(52.41, -4.08)
    """
    if isinstance(latitude, np.ndarray):
        latitude = float(latitude.flat[0])
    if isinstance(longitude, np.ndarray):
        longitude = float(longitude.flat[0])
    # Longitude 180 is the antimeridian, i.e. -180 (zone 1).
    longitude = ((float(longitude) + 180.0) % 360.0) - 180.0

    if 56 <= latitude < 64 and 3 <= longitude < 12:
        return 32

    if 72 <= latitude <= 84 and longitude >= 0:
        if longitude < 9:
            return 31
        if longitude < 21:
            return 33
        if longitude < 33:
            return 35
        if longitude < 42:
            return 37

    return int((longitude + 180) / 6) + 1


def latlon_arr_to_utm_zone_number(
    latitude: np.ndarray,
    longitude: np.ndarray,
) -> np.ndarray:
    """
    Vectorised UTM zone number calculation for NumPy arrays of latitudes and longitudes.

    :param latitude: NumPy array of latitudes.
    :param longitude: NumPy array of longitudes.
    :return: NumPy array of integer UTM zone numbers.

    .. code:: python

        import numpy as np
        import rsgispy.tools.utm

        lats = np.array([52.41, 51.48])
        lons = np.array([-4.08, -3.18])
        zones = rsgispy.tools.utm.latlon_arr_to_utm_zone_number(lats, lons)
    """
    latitude = np.asarray(latitude)
    # Longitude 180 is the antimeridian, i.e. -180 (zone 1).
    longitude = np.mod(np.asarray(longitude, dtype=np.float64) + 180.0, 360.0) - 180.0
    utm_zones = np.floor((longitude + 180) / 6).astype(int) + 1

    utm_zones[
        (56 <= latitude) & (latitude < 64) & (3 <= longitude) & (longitude < 12)
    ] = 32
    utm_zones[
        (72 <= latitude) & (latitude <= 84) & (longitude >= 0) & (longitude < 9)
    ] = 31
    utm_zones[
        (72 <= latitude) & (latitude <= 84) & (longitude >= 9) & (longitude < 21)
    ] = 33
    utm_zones[
        (72 <= latitude) & (latitude <= 84) & (longitude >= 21) & (longitude < 33)
    ] = 35
    utm_zones[
        (72 <= latitude) & (latitude <= 84) & (longitude >= 33) & (longitude < 42)
    ] = 37

    return utm_zones


def latlon_to_mode_utm_zone_number(
    latitude: np.ndarray,
    longitude: np.ndarray,
) -> int:
    """
    Find the modal UTM zone number for arrays of latitude and longitude values.

    :param latitude: NumPy array of latitudes.
    :param longitude: NumPy array of longitudes.
    :return: Modal UTM zone integer.

    .. code:: python

        import numpy as np
        import rsgispy.tools.utm

        lats = np.array([52.41, 51.48, 53.00])
        lons = np.array([-4.08, -3.18, -3.50])
        mode_zone = rsgispy.tools.utm.latlon_to_mode_utm_zone_number(lats, lons)
    """
    utm_zones = latlon_arr_to_utm_zone_number(latitude, longitude)
    if HAVE_SCIPY:
        mode_res = scipy.stats.mode(utm_zones)
        return int(np.asarray(mode_res.mode).flat[0])
    vals, counts = np.unique(utm_zones, return_counts=True)
    return int(vals[np.argmax(counts)])


def zone_number_to_central_longitude(zone_number: int) -> float:
    """
    Return the central meridian longitude for a UTM zone number.

    :param zone_number: UTM zone number (1..60).
    :return: Central longitude in degrees.

    .. code:: python

        import rsgispy.tools.utm
        central_lon = rsgispy.tools.utm.zone_number_to_central_longitude(30)
    """
    return (zone_number - 1) * 6 - 180 + 3
