"""
The imagecalibration.solarangles module provides vectorised per-pixel solar
azimuth and zenith image calculation and solar azimuth convention helpers.
"""

from contextlib import ExitStack
import datetime
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pyproj
import rasterio

import rsgispy
from rsgispy.imageprocess import (
    cast_result_block,
    create_output_from_dataset,
    ensure_file_exists,
    iter_dataset_windows,
    maybe_pop_stats,
    normalise_creation_options,
    resolve_out_dtype,
)

__all__ = [
    "get_solar_irr_convention_solar_azimuth_from_usgs",
    "get_solar_irr_convention_solar_azimuth_from_trad",
    "calc_solar_azimuth_zenith",
]

_J2000_UTC = datetime.datetime(2000, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)


def get_solar_irr_convention_solar_azimuth_from_usgs(solar_az: float) -> float:
    """
    Convert USGS solar azimuth (``[-180, 180]``, North=0, East=90, West=-90)
    to solar irradiance convention (``[0, 360)``, North=0, East=90, South=180,
    West=270).

    :param solar_az: Solar azimuth angle in degrees under USGS convention.
    :return: Solar azimuth angle in ``[0, 360)`` degrees.
    """
    return float(float(solar_az) % 360.0)


def get_solar_irr_convention_solar_azimuth_from_trad(solar_az: float) -> float:
    """
    Convert traditional solar azimuth (South=0, East=+90, West=-90, North=+-180)
    to solar irradiance convention (``[0, 360)``, North=0, East=90, South=180,
    West=270).

    :param solar_az: Solar azimuth angle in degrees under traditional convention.
    :return: Solar azimuth angle in ``[0, 360)`` degrees.
    """
    return float((180.0 - float(solar_az)) % 360.0)


def _calc_noaa_solar_angles(
    lats_deg: np.ndarray,
    lons_deg: np.ndarray,
    dt_utc: datetime.datetime,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorised NOAA Solar Position equations returning ``(azimuth_deg, zenith_deg)``.

    Azimuth is clockwise from North in ``[0, 360)``; zenith is angle from
    vertical in ``[0, 180]``.
    """
    # pylint: disable=too-many-locals
    days_since_j2000 = (dt_utc - _J2000_UTC).total_seconds() / 86400.0
    jc = days_since_j2000 / 36525.0

    geom_mean_long = (280.46646 + jc * (36000.76983 + (jc * 0.0003032))) % 360.0
    geom_mean_anom = 357.52911 + jc * (35999.05029 - (0.0001537 * jc))
    eccent = 0.016708634 - jc * (0.000042037 + (0.0000001267 * jc))

    anom_rad = np.radians(geom_mean_anom)
    sun_eq_ctr = (
        np.sin(anom_rad) * (1.914602 - jc * (0.004817 + (0.000014 * jc)))
        + np.sin(2.0 * anom_rad) * (0.019993 - (0.000101 * jc))
        + np.sin(3.0 * anom_rad) * 0.000289
    )
    sun_true_long = geom_mean_long + sun_eq_ctr
    omega_rad = np.radians(125.04 - (1934.136 * jc))
    sun_app_long = sun_true_long - 0.00569 - (0.00478 * np.sin(omega_rad))

    mean_obliq = (
        23.0
        + (26.0 + ((21.448 - jc * (46.815 + jc * (0.00059 - (jc * 0.001813)))) / 60.0))
        / 60.0
    )
    obliq_rad = np.radians(mean_obliq + (0.00256 * np.cos(omega_rad)))
    app_long_rad = np.radians(sun_app_long)
    decl_rad = np.arcsin(np.sin(obliq_rad) * np.sin(app_long_rad))

    y_var = np.tan(obliq_rad / 2.0) ** 2
    ml_rad = np.radians(geom_mean_long)
    eq_of_time = 4.0 * np.degrees(
        (y_var * np.sin(2.0 * ml_rad))
        - (2.0 * eccent * np.sin(anom_rad))
        + (4.0 * eccent * y_var * np.sin(anom_rad) * np.cos(2.0 * ml_rad))
        - (0.5 * y_var * y_var * np.sin(4.0 * ml_rad))
        - (1.25 * eccent * eccent * np.sin(2.0 * anom_rad))
    )

    utc_minutes = (
        (dt_utc.hour * 60.0)
        + dt_utc.minute
        + ((dt_utc.second + (dt_utc.microsecond / 1e6)) / 60.0)
    )
    true_solar_time = (utc_minutes + eq_of_time + (4.0 * lons_deg)) % 1440.0
    quarter_tst = true_solar_time / 4.0
    hour_angle_deg = np.where(quarter_tst < 0.0, quarter_tst + 180.0, quarter_tst - 180.0)

    ha_rad = np.radians(hour_angle_deg)
    lat_rad = np.radians(lats_deg)

    cos_zenith = np.clip(
        (np.sin(lat_rad) * np.sin(decl_rad))
        + (np.cos(lat_rad) * np.cos(decl_rad) * np.cos(ha_rad)),
        -1.0,
        1.0,
    )
    zenith_deg = np.degrees(np.arccos(cos_zenith))
    zen_rad = np.radians(zenith_deg)

    denom = np.cos(lat_rad) * np.sin(zen_rad)
    with np.errstate(divide="ignore", invalid="ignore"):
        cos_az = ((np.sin(lat_rad) * np.cos(zen_rad)) - np.sin(decl_rad)) / denom
        cos_az = np.clip(cos_az, -1.0, 1.0)
        az_base = np.degrees(np.arccos(cos_az))

    azimuth_deg = np.where(
        np.abs(denom) < 1e-12,
        0.0,
        np.where(
            hour_angle_deg > 0.0,
            (az_base + 180.0) % 360.0,
            (540.0 - az_base) % 360.0,
        ),
    )
    return azimuth_deg, zenith_deg


def calc_solar_azimuth_zenith(
    input_img: str,
    img_datetime: datetime.datetime,
    output_img: str,
    gdalformat: str = "GTIFF",
    *,
    datatype: int = rsgispy.TYPE_32FLOAT,
    calc_stats: bool = True,
    use_pbar: bool = True,
    creation_opts: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
) -> None:
    """
    Calculate a two-band solar azimuth (band 1, degrees clockwise from North)
    and solar zenith (band 2, degrees from vertical) raster matching the grid of
    ``input_img`` at acquisition timestamp ``img_datetime``.

    Naive ``datetime.datetime`` instances are interpreted as UTC.

    :param input_img: Path to any georeferenced input image defining the grid
                      and CRS.
    :param img_datetime: Acquisition ``datetime.datetime`` (naive treated as UTC).
    :param output_img: Path to the output 2-band solar angles raster.
    :param gdalformat: Output GDAL driver format (Default = ``'GTIFF'``).
    :param datatype: Output ``rsgispy.TYPE_*`` constant
                     (Default = :data:`rsgispy.TYPE_32FLOAT`).
    :param calc_stats: If ``True``, populate image statistics and pyramids
                       (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).
    :param creation_opts: Optional GDAL creation options (dict or sequence of
                          ``'KEY=VALUE'``).

    .. code:: python

        import datetime
        import rsgispy.imagecalibration.solarangles

        acq_dt = datetime.datetime(2020, 6, 21, 11, 30, 0, tzinfo=datetime.timezone.utc)
        rsgispy.imagecalibration.solarangles.calc_solar_azimuth_zenith(
            "scene.tif", acq_dt, "solar_angles.tif"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    ensure_file_exists(input_img)
    if not isinstance(img_datetime, datetime.datetime):
        raise rsgispy.RSGISPyException(
            f"img_datetime must be a datetime.datetime instance, got {type(img_datetime).__name__}."
        )
    dt_utc = (
        img_datetime.replace(tzinfo=datetime.timezone.utc)
        if img_datetime.tzinfo is None
        else img_datetime.astimezone(datetime.timezone.utc)
    )
    norm_opts = normalise_creation_options(creation_opts)
    _, out_rio_dtype, out_np_dtype = resolve_out_dtype(
        datatype, default_dtype=rsgispy.TYPE_32FLOAT
    )

    with ExitStack() as stack:
        src = stack.enter_context(rasterio.open(input_img, "r"))
        if src.crs is None:
            raise rsgispy.RSGISPyException(
                f"Input image '{input_img}' does not have a CRS defined."
            )
        transformer = pyproj.Transformer.from_crs(
            pyproj.CRS.from_user_input(src.crs),
            pyproj.CRS.from_epsg(4326),
            always_xy=True,
        )
        t_a, t_b, t_c, t_d, t_e, t_f = (
            src.transform.a,
            src.transform.b,
            src.transform.c,
            src.transform.d,
            src.transform.e,
            src.transform.f,
        )

        dst = create_output_from_dataset(
            stack=stack,
            src=src,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype=out_rio_dtype,
            count=2,
            creation_opts=norm_opts,
        )
        dst.set_band_description(1, "SolarAzimuth")
        dst.set_band_description(2, "SolarZenith")

        for win in iter_dataset_windows(src.height, src.width, use_pbar=use_pbar):
            rows = (
                np.arange(win.row_off, win.row_off + win.height, dtype=np.float64)
                + 0.5
            )
            cols = (
                np.arange(win.col_off, win.col_off + win.width, dtype=np.float64)
                + 0.5
            )
            cc, rr = np.meshgrid(cols, rows)
            xs = (t_a * cc) + (t_b * rr) + t_c
            ys = (t_d * cc) + (t_e * rr) + t_f
            lons, lats = transformer.transform(xs, ys)
            az_arr, zen_arr = _calc_noaa_solar_angles(
                np.asarray(lats, dtype=np.float64),
                np.asarray(lons, dtype=np.float64),
                dt_utc,
            )
            out_blk = np.stack((az_arr, zen_arr), axis=0)
            dst.write(
                cast_result_block(
                    out_blk,
                    int(win.height),
                    int(win.width),
                    out_np_dtype,
                    clamp_int=True,
                ),
                window=win,
            )

    maybe_pop_stats(
        output_img=output_img,
        calc_stats=calc_stats,
        no_data_val=None,
        use_pbar=use_pbar,
        gdalformat=gdalformat,
    )
