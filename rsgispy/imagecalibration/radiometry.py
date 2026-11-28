"""
The imagecalibration.radiometry module provides radiometric calibration,
solar-geometry utilities, brightness temperature conversion, and Dark Object
Subtraction (DOS) atmospheric correction using ``rasterio`` and ``numpy``.
"""

from contextlib import ExitStack
from dataclasses import dataclass
import datetime
import math
import warnings
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import rasterio

import rsgispy
from rsgispy.imagecalc.percentiles import calc_band_percentile
from rsgispy.imageprocess import (
    build_valid_mask,
    cast_result_block,
    create_output_from_dataset,
    ensure_file_exists,
    iter_dataset_windows,
    normalise_creation_options,
    pop_stats_on_close,
    resolve_nodata_and_fill,
    resolve_out_dtype,
    validate_band_index,
)

__all__ = [
    "LandsatThermalCoeffs",
    "get_julian_day",
    "calc_solar_distance",
    "get_esun_value",
    "visibility_to_aot",
    "radiance_to_toa_refl",
    "landsat_thermal_rad_to_brightness",
    "perform_dos_calc",
]

_J2000_DATE = datetime.date(2000, 1, 1)
_J2000_JD = 2451545


@dataclass
class LandsatThermalCoeffs:
    """
    Calibration coefficient record for a single Landsat thermal band passed to
    :func:`landsat_thermal_rad_to_brightness`.

    :param band_name: Output band description name (e.g., ``'ThermalB10'``).
    :param img_band: 1-based band index within the input radiance image.
    :param k1: Thermal conversion constant ``K1`` from the Landsat MTL header.
    :param k2: Thermal conversion constant ``K2`` from the Landsat MTL header.
    """

    band_name: str
    img_band: int
    k1: float
    k2: float


def get_julian_day(year: int, month: int, day: int) -> int:
    """
    Calculate the astronomical Julian Day Number (at 12:00 UT) for a Gregorian
    calendar date.

    :param year: Acquisition year.
    :param month: Acquisition month (1..12).
    :param day: Acquisition day of month (1..31).
    :return: Integer Julian Day Number.

    .. code:: python

        import rsgispy.imagecalibration

        jd = rsgispy.imagecalibration.get_julian_day(2020, 6, 21)
    """
    try:
        acq_date = datetime.date(int(year), int(month), int(day))
    except (TypeError, ValueError) as err:
        raise rsgispy.RSGISPyException(
            f"Invalid calendar date ({year}, {month}, {day}): {err}"
        ) from err
    return _J2000_JD + (acq_date - _J2000_DATE).days


def calc_solar_distance(julian_day: Union[int, float]) -> float:
    """
    Calculate the Earth–Sun distance in astronomical units (AU) for a given
    Julian Day Number.

    :param julian_day: Julian Day Number (e.g., from :func:`get_julian_day`).
    :return: Earth–Sun distance in AU.

    .. code:: python

        import rsgispy.imagecalibration

        jd = rsgispy.imagecalibration.get_julian_day(2020, 6, 21)
        dist_au = rsgispy.imagecalibration.calc_solar_distance(jd)
    """
    jd_val = float(julian_day)
    if not math.isfinite(jd_val):
        raise rsgispy.RSGISPyException(
            f"julian_day must be finite, got {julian_day!r}."
        )
    days_j2000 = jd_val - float(_J2000_JD)
    mean_anom_rad = math.radians(357.529 + (0.98560028 * days_j2000))
    return float(
        1.00014
        - (0.01671 * math.cos(mean_anom_rad))
        - (0.00014 * math.cos(2.0 * mean_anom_rad))
    )


def get_esun_value(
    radiance: float,
    toa_refl: float,
    day: int,
    month: int,
    year: int,
    solar_zenith: float,
) -> float:
    """
    Calculate the mean exoatmospheric solar irradiance (ESUN) when both
    at-sensor radiance and Top-of-Atmosphere (TOA) reflectance are known.

    :param radiance: At-sensor radiance value.
    :param toa_refl: Known TOA reflectance value (> 0).
    :param day: Acquisition day of month (1..31).
    :param month: Acquisition month (1..12).
    :param year: Acquisition year.
    :param solar_zenith: Solar zenith angle in degrees (in ``[0, 90)``).
    :return: Estimated ESUN solar irradiance value.

    .. code:: python

        import rsgispy.imagecalibration

        esun = rsgispy.imagecalibration.get_esun_value(120.5, 0.25, 21, 6, 2020, 30.0)
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    rad_val = float(radiance)
    refl_val = float(toa_refl)
    zen_val = float(solar_zenith)
    if not math.isfinite(rad_val):
        raise rsgispy.RSGISPyException(
            f"radiance must be finite, got {radiance!r}."
        )
    if not math.isfinite(refl_val) or refl_val <= 0.0:
        raise rsgispy.RSGISPyException(
            f"toa_refl must be a finite positive value, got {toa_refl!r}."
        )
    if not math.isfinite(zen_val) or zen_val < 0.0 or zen_val >= 90.0:
        raise rsgispy.RSGISPyException(
            f"solar_zenith must be in [0, 90) degrees, got {solar_zenith!r}."
        )

    julian_day = get_julian_day(year, month, day)
    solar_dist = calc_solar_distance(julian_day)
    cos_zen = math.cos(math.radians(zen_val))
    return float(
        (math.pi * rad_val * (solar_dist * solar_dist)) / (refl_val * cos_zen)
    )


def visibility_to_aot(vis: float) -> float:
    """
    Convert an atmospheric visibility estimate (km) to Aerosol Optical
    Thickness (AOT at 550 nm).

    :param vis: Visibility estimate (> 0).
    :return: Estimated AOT value.

    .. code:: python

        import rsgispy.imagecalibration

        aot = rsgispy.imagecalibration.visibility_to_aot(25.0)
    """
    vis_val = float(vis)
    if not math.isfinite(vis_val) or vis_val <= 0.0:
        raise rsgispy.RSGISPyException(
            f"Visibility must be a finite positive value, got {vis!r}."
        )
    return float((3.9449 / vis_val) + 0.08498)


def _extract_irradiance_values(
    solar_irradiance: Sequence[Any], n_bands: int
) -> Sequence[float]:
    """Extract and validate per-band ESUN irradiance floats."""
    if solar_irradiance is None or len(solar_irradiance) != n_bands:
        raise rsgispy.RSGISPyException(
            f"solar_irradiance must contain {n_bands} entry/entries matching "
            f"the input image band count."
        )
    vals = []
    for idx, entry in enumerate(solar_irradiance, start=1):
        if isinstance(entry, Mapping):
            raw = entry.get("irradiance")
        elif hasattr(entry, "irradiance"):
            raw = entry.irradiance
        else:
            raw = entry
        try:
            fval = float(raw)
        except (TypeError, ValueError) as err:
            raise rsgispy.RSGISPyException(
                f"Invalid irradiance entry at index {idx}: {entry!r}."
            ) from err
        if not math.isfinite(fval) or fval <= 0.0:
            raise rsgispy.RSGISPyException(
                f"Irradiance for band {idx} must be finite and > 0, got {fval!r}."
            )
        vals.append(fval)
    return vals


def _extract_thermal_coeffs(
    band_defs: Sequence[Any], n_bands: int, input_img: str
) -> Sequence[Tuple[str, int, float, float]]:
    """Normalise and validate thermal band coefficient records."""
    if not band_defs:
        raise rsgispy.RSGISPyException(
            "band_defs must contain at least one thermal band definition."
        )
    parsed = []
    for idx, entry in enumerate(band_defs, start=1):
        if isinstance(entry, Mapping):
            bname = entry.get("band_name")
            img_band = entry.get("img_band")
            k1 = entry.get("k1")
            k2 = entry.get("k2")
        else:
            bname = getattr(entry, "band_name", None)
            img_band = getattr(entry, "img_band", None)
            k1 = getattr(entry, "k1", None)
            k2 = getattr(entry, "k2", None)

        if img_band is None or k1 is None or k2 is None:
            raise rsgispy.RSGISPyException(
                f"Thermal band definition {idx} must provide 'img_band', "
                f"'k1', and 'k2': {entry!r}."
            )
        bidx = int(img_band)
        validate_band_index(bidx, n_bands, input_img)
        k1_val = float(k1)
        k2_val = float(k2)
        if not math.isfinite(k1_val) or k1_val <= 0.0:
            raise rsgispy.RSGISPyException(
                f"Thermal K1 must be finite and > 0 for band {bidx}, got {k1!r}."
            )
        if not math.isfinite(k2_val) or k2_val <= 0.0:
            raise rsgispy.RSGISPyException(
                f"Thermal K2 must be finite and > 0 for band {bidx}, got {k2!r}."
            )
        out_name = str(bname) if bname else f"Band_{bidx}"
        parsed.append((out_name, bidx, k1_val, k2_val))
    return parsed


def radiance_to_toa_refl(
    input_img: str,
    output_img: str,
    scale_factor: float,
    year: int,
    month: int,
    day: int,
    solar_zenith: float,
    solar_irradiance: Sequence[Any],
    *,
    gdalformat: str = "GTIFF",
    datatype: int = rsgispy.TYPE_32FLOAT,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
    creation_opts: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
) -> None:
    """
    Convert at-sensor radiance pixel values to Top-of-Atmosphere (TOA)
    reflectance using ``rasterio`` and ``numpy``.

    :param input_img: Path to the input at-sensor radiance image.
    :param output_img: Path to the output TOA reflectance image.
    :param scale_factor: Multiplicative scale factor applied to TOA reflectance
                         (set to ``1.0`` for unscaled ``[0, 1]`` values, or e.g.
                         ``1000.0`` / ``10000.0`` for scaled integer outputs).
    :param year: Acquisition year.
    :param month: Acquisition month (1..12).
    :param day: Acquisition day of month (1..31).
    :param solar_zenith: Solar zenith angle in degrees (``90 - solar_elevation``).
    :param solar_irradiance: Sequence of per-band ESUN irradiance values (either
                             numeric floats or objects/mappings with an
                             ``irradiance`` field).
    :param gdalformat: Output GDAL driver format (Default = ``'GTIFF'``).
    :param datatype: Output ``rsgispy.TYPE_*`` constant
                     (Default = :data:`rsgispy.TYPE_32FLOAT`).
    :param no_data_val: Optional input/output no-data value (defaults to the
                        input image's header nodata when ``None``).
    :param calc_stats: If ``True``, populate image statistics and pyramids
                       (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).
    :param creation_opts: Optional GDAL creation options (dict or sequence of
                          ``'KEY=VALUE'``).

    .. code:: python

        import rsgispy
        import rsgispy.imagecalibration

        rsgispy.imagecalibration.radiance_to_toa_refl(
            "rad.tif",
            "toa.tif",
            scale_factor=1000.0,
            year=2020,
            month=6,
            day=21,
            solar_zenith=30.0,
            solar_irradiance=[1850.0, 1550.0],
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    ensure_file_exists(input_img)
    sf_val = float(scale_factor)
    if not math.isfinite(sf_val) or sf_val <= 0.0:
        raise rsgispy.RSGISPyException(
            f"scale_factor must be finite and > 0, got {scale_factor!r}."
        )
    zen_val = float(solar_zenith)
    if not math.isfinite(zen_val) or zen_val < 0.0 or zen_val >= 90.0:
        raise rsgispy.RSGISPyException(
            f"solar_zenith must be in [0, 90) degrees, got {solar_zenith!r}."
        )

    julian_day = get_julian_day(year, month, day)
    solar_dist = calc_solar_distance(julian_day)
    cos_zen = math.cos(math.radians(zen_val))
    norm_opts = normalise_creation_options(creation_opts)

    _, out_rio_dtype, out_np_dtype = resolve_out_dtype(
        datatype, default_dtype=rsgispy.TYPE_32FLOAT
    )

    with ExitStack() as stack:
        src = stack.enter_context(rasterio.open(input_img, "r"))
        esun_vals = _extract_irradiance_values(solar_irradiance, src.count)
        eff_in_nd = no_data_val if no_data_val is not None else src.nodata
        dst_nodata, fill_val = resolve_nodata_and_fill(eff_in_nd, out_np_dtype)

        band_factors = [
            (math.pi * (solar_dist * solar_dist) * sf_val) / (esun * cos_zen)
            for esun in esun_vals
        ]

        dst = create_output_from_dataset(
            stack=stack,
            src=src,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype=out_rio_dtype,
            count=src.count,
            creation_opts=norm_opts,
        )
        if dst_nodata is not None:
            dst.nodata = dst_nodata
        pop_stats_on_close(
            dst, calc_stats, no_data_val=dst_nodata, use_pbar=use_pbar
        )
        for bidx, desc in enumerate(src.descriptions, start=1):
            if desc:
                dst.set_band_description(bidx, desc)

        for win in iter_dataset_windows(src.height, src.width, use_pbar=use_pbar):
            blk = src.read(window=win, out_dtype=np.float64)
            out_blk = np.empty_like(blk, dtype=np.float64)
            valid_blk = np.empty(blk.shape, dtype=bool)
            for b_i, factor in enumerate(band_factors):
                band_arr = blk[b_i]
                valid = build_valid_mask(band_arr, no_data_val=eff_in_nd)
                valid_blk[b_i] = valid
                out_blk[b_i] = np.where(valid, band_arr * factor, fill_val)
            dst.write(
                cast_result_block(
                    out_blk,
                    int(win.height),
                    int(win.width),
                    out_np_dtype,
                    nodata=dst_nodata,
                    valid_mask=None if dst_nodata is None else valid_blk,
                ),
                window=win,
            )


def landsat_thermal_rad_to_brightness(
    input_img: str,
    output_img: str,
    band_defs: Sequence[Any],
    scale_factor: float = 1.0,
    *,
    gdalformat: str = "GTIFF",
    datatype: int = rsgispy.TYPE_32FLOAT,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
    creation_opts: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
) -> None:
    """
    Convert Landsat thermal at-sensor radiance to brightness temperature in
    degrees Celsius using the Planck inverse approximation.

    :param input_img: Path to the input thermal radiance image.
    :param output_img: Path to the output brightness temperature image.
    :param band_defs: Sequence of :class:`LandsatThermalCoeffs` (or matching
                      objects/dicts with ``band_name``, ``img_band``, ``k1``,
                      ``k2`` attributes/keys).
    :param scale_factor: Multiplicative scale factor applied to degrees Celsius
                         (Default = ``1.0``).
    :param gdalformat: Output GDAL driver format (Default = ``'GTIFF'``).
    :param datatype: Output ``rsgispy.TYPE_*`` constant
                     (Default = :data:`rsgispy.TYPE_32FLOAT`).
    :param no_data_val: Optional input/output no-data value (defaults to the
                        input image's header nodata when ``None``).
    :param calc_stats: If ``True``, populate image statistics and pyramids
                       (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).
    :param creation_opts: Optional GDAL creation options (dict or sequence of
                          ``'KEY=VALUE'``).

    .. code:: python

        import rsgispy.imagecalibration

        coeffs = [
            rsgispy.imagecalibration.LandsatThermalCoeffs(
                band_name="B10", img_band=1, k1=774.8853, k2=1321.0789
            )
        ]
        rsgispy.imagecalibration.landsat_thermal_rad_to_brightness(
            "thermal_rad.tif", "brightness_c.tif", coeffs, scale_factor=100.0
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    ensure_file_exists(input_img)
    sf_val = float(scale_factor)
    if not math.isfinite(sf_val) or sf_val <= 0.0:
        raise rsgispy.RSGISPyException(
            f"scale_factor must be finite and > 0, got {scale_factor!r}."
        )
    norm_opts = normalise_creation_options(creation_opts)
    _, out_rio_dtype, out_np_dtype = resolve_out_dtype(
        datatype, default_dtype=rsgispy.TYPE_32FLOAT
    )

    with ExitStack() as stack:
        src = stack.enter_context(rasterio.open(input_img, "r"))
        coeffs = _extract_thermal_coeffs(band_defs, src.count, input_img)
        eff_in_nd = no_data_val if no_data_val is not None else src.nodata
        dst_nodata, fill_val = resolve_nodata_and_fill(eff_in_nd, out_np_dtype)
        invalid_fill = dst_nodata if dst_nodata is not None else 0.0

        dst = create_output_from_dataset(
            stack=stack,
            src=src,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype=out_rio_dtype,
            count=len(coeffs),
            creation_opts=norm_opts,
        )
        if dst_nodata is not None:
            dst.nodata = dst_nodata
        pop_stats_on_close(
            dst, calc_stats, no_data_val=dst_nodata, use_pbar=use_pbar
        )
        for out_idx, (bname, _, _, _) in enumerate(coeffs, start=1):
            dst.set_band_description(out_idx, bname)

        for win in iter_dataset_windows(src.height, src.width, use_pbar=use_pbar):
            out_bands = []
            for _, src_band, k1_val, k2_val in coeffs:
                arr = src.read(src_band, window=win, out_dtype=np.float64)
                valid = build_valid_mask(arr, no_data_val=eff_in_nd) & (arr > 0.0)
                out_arr = np.full(arr.shape, invalid_fill, dtype=np.float64)
                if np.any(valid):
                    safe_rad = arr[valid]
                    out_arr[valid] = (
                        (k2_val / np.log((k1_val / safe_rad) + 1.0)) - 273.15
                    ) * sf_val
                if eff_in_nd is not None and math.isfinite(float(fill_val)):
                    nd_mask = ~build_valid_mask(arr, no_data_val=eff_in_nd)
                    out_arr[nd_mask] = fill_val
                out_bands.append(out_arr)

            dst.write(
                cast_result_block(
                    np.stack(out_bands, axis=0),
                    int(win.height),
                    int(win.width),
                    out_np_dtype,
                ),
                window=win,
            )


def perform_dos_calc(
    input_img: str,
    output_img: str,
    non_negative: bool = True,
    no_data_val: Optional[float] = 0.0,
    dark_obj_refl_val: float = 0.0,
    dark_obj_percentile: float = 1.0,
    copy_band_names: bool = True,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
    creation_opts: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
) -> None:
    """
    Perform Dark Object Subtraction (DOS) atmospheric correction on an input
    image (typically Top-of-Atmosphere reflectance).

    Computes the per-band ``dark_obj_percentile`` value ignoring ``no_data_val``,
    subtracts that dark-object offset from valid pixels, and adds
    ``dark_obj_refl_val``.

    :param input_img: Path to the input image (e.g., TOA reflectance).
    :param output_img: Path to the output DOS-corrected image.
    :param non_negative: If ``True``, clamp negative reflectance residuals so
                         valid output pixels are at least
                         ``max(0.0, dark_obj_refl_val)`` (Default = ``True``).
    :param no_data_val: No-data value ignored during percentile estimation and
                        preserved in output (Default = ``0.0``; if ``None``,
                        uses input dataset nodata if set).
    :param dark_obj_refl_val: Minimum dark-object reflectance offset added back
                              to subtracted values (Default = ``0.0``).
    :param dark_obj_percentile: Image percentile in ``[0, 100]`` defining the
                                dark-object threshold (Default = ``1.0``, the
                                1st percentile).
    :param copy_band_names: If ``True``, copy band descriptions from
                            ``input_img`` to ``output_img`` (Default = ``True``).
    :param gdalformat: Output GDAL driver format (Default = ``'GTIFF'``).
    :param datatype: Optional ``rsgispy.TYPE_*`` constant for the output image
                     (defaults to matching ``input_img``).
    :param calc_stats: If ``True``, populate image statistics and pyramids
                       (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).
    :param creation_opts: Optional GDAL creation options (dict or sequence of
                          ``'KEY=VALUE'``).

    .. code:: python

        import rsgispy.imagecalibration

        rsgispy.imagecalibration.perform_dos_calc(
            "toa_refl.tif",
            "dos_sref.tif",
            no_data_val=0.0,
            dark_obj_percentile=1.0,
        )

    .. versionchanged:: 0.5.0
       ``dark_obj_percentile`` is on the 0-100 scale, as for
       :func:`rsgispy.imagecalc.percentiles.calc_band_percentile` (it was a
       0-1 fraction); a value between 0 and 1 issues a warning.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    ensure_file_exists(input_img)
    dark_refl = float(dark_obj_refl_val)
    if not math.isfinite(dark_refl):
        raise rsgispy.RSGISPyException(
            f"dark_obj_refl_val must be finite, got {dark_obj_refl_val!r}."
        )
    dark_pct = float(dark_obj_percentile)
    if not 0.0 <= dark_pct <= 100.0:
        raise rsgispy.RSGISPyException(
            f"dark_obj_percentile must be in [0, 100], got {dark_obj_percentile!r}."
        )
    if 0.0 < dark_pct < 1.0:
        warnings.warn(
            f"perform_dos_calc dark_obj_percentile={dark_pct} is on the 0-100 "
            "scale since rsgispy 0.5.0 (it was a 0-1 fraction); pass e.g. 1.0 "
            "for the 1st percentile.",
            UserWarning,
            stacklevel=2,
        )
    norm_opts = normalise_creation_options(creation_opts)

    with ExitStack() as stack:
        src = stack.enter_context(rasterio.open(input_img, "r"))
        eff_in_nd = no_data_val if no_data_val is not None else src.nodata
        _, out_rio_dtype, out_np_dtype = resolve_out_dtype(
            datatype, ref_np_dtype=src.dtypes[0]
        )
        dst_nodata, fill_val = resolve_nodata_and_fill(eff_in_nd, out_np_dtype)
        n_bands = src.count

    with warnings.catch_warnings():
        # The 0-1 scale warning has been given above, for this function.
        warnings.filterwarnings("ignore", message="calc_band_percentile percentile=")
        percent_vals = calc_band_percentile(
            input_img=input_img,
            percentile=dark_pct,
            no_data_val=eff_in_nd,
        )
    if len(percent_vals) != n_bands:
        raise rsgispy.RSGISPyException(
            f"Expected {n_bands} band percentiles, got {len(percent_vals)}."
        )
    for b_idx, pval in enumerate(percent_vals, start=1):
        if not math.isfinite(pval):
            raise rsgispy.RSGISPyException(
                f"Band {b_idx} in '{input_img}' contains no valid pixels for "
                f"DOS percentile calculation."
            )

    min_floor = max(0.0, dark_refl)
    with ExitStack() as stack:
        src = stack.enter_context(rasterio.open(input_img, "r"))
        dst = create_output_from_dataset(
            stack=stack,
            src=src,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype=out_rio_dtype,
            count=src.count,
            creation_opts=norm_opts,
        )
        if dst_nodata is not None:
            dst.nodata = dst_nodata
        pop_stats_on_close(
            dst, calc_stats, no_data_val=dst_nodata, use_pbar=use_pbar
        )
        if copy_band_names:
            for bidx, desc in enumerate(src.descriptions, start=1):
                if desc:
                    dst.set_band_description(bidx, desc)

        for win in iter_dataset_windows(src.height, src.width, use_pbar=use_pbar):
            blk = src.read(window=win, out_dtype=np.float64)
            out_blk = np.empty_like(blk, dtype=np.float64)
            valid_blk = np.empty(blk.shape, dtype=bool)
            for b_i, offset_val in enumerate(percent_vals):
                arr = blk[b_i]
                valid = build_valid_mask(arr, no_data_val=eff_in_nd)
                valid_blk[b_i] = valid
                corrected = (arr - float(offset_val)) + dark_refl
                if non_negative:
                    corrected = np.maximum(corrected, min_floor)
                out_blk[b_i] = np.where(valid, corrected, fill_val)
            dst.write(
                cast_result_block(
                    out_blk,
                    int(win.height),
                    int(win.width),
                    out_np_dtype,
                    nodata=dst_nodata,
                    valid_mask=None if dst_nodata is None else valid_blk,
                ),
                window=win,
            )
