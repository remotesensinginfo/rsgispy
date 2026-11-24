"""
The imagecalibration subpackage provides pure-Python radiometric calibration,
solar-angle modelling, Dark Object Subtraction (DOS) atmospheric correction,
and sensor product ingestion for Landsat Collection 2 and Sentinel-2 L2A data.
"""

from rsgispy.imagecalibration.radiometry import (
    LandsatThermalCoeffs,
    calc_solar_distance,
    get_esun_value,
    get_julian_day,
    landsat_thermal_rad_to_brightness,
    perform_dos_calc,
    radiance_to_toa_refl,
    visibility_to_aot,
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
