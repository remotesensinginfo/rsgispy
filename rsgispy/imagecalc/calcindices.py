# pylint: disable=too-many-lines
"""
The imagecalc.calcindices module provides unified spectral index stack
calculation using ``rasterio`` and ``numpy``.
"""

from contextlib import ExitStack
from dataclasses import dataclass
import math
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import rasterio

import rsgispy
from rsgispy.imageprocess import (
    compute_spatial_intersection,
    create_output_from_dataset,
    ensure_file_exists,
    iter_dataset_windows,
    validate_band_index,
)
from rsgispy.imageutils.stats import pop_img_stats

__all__ = [
    "calc_indices",
    "get_available_indices",
    "get_index_req_bands",
    "get_index_equation",
]

_BAND_ALIASES: Dict[str, str] = {
    "blue": "blue",
    "green": "green",
    "red": "red",
    "nir": "nir",
    "swir": "swir1",
    "swir1": "swir1",
    "swir_1": "swir1",
    "swir-1": "swir1",
    "swir2": "swir2",
    "swir_2": "swir2",
    "swir-2": "swir2",
    "re": "red_edge",
    "rededge": "red_edge",
    "red_edge": "red_edge",
    "red-edge": "red_edge",
    "red edge": "red_edge",
}


@dataclass(frozen=True)
class _IndexSpec:
    """Metadata and vectorised calculation function for a spectral index."""

    canonical_name: str
    req_bands: Tuple[str, ...]
    equation: str
    calc_fn: Callable[
        [Dict[str, np.ndarray], Mapping[str, float]],
        Tuple[np.ndarray, np.ndarray],
    ]
    requires_brightness_stats: bool = False


def _calc_ndvi_block(
    bands: Dict[str, np.ndarray], _params: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Normalised Difference Vegetation Index (NDVI)."""
    nir = bands["nir"]
    red = bands["red"]
    denom = nir + red
    valid = denom != 0.0
    out = np.zeros_like(denom, dtype=np.float64)
    np.divide(nir - red, denom, out=out, where=valid)
    return out, valid


def _calc_wbi_bluenir_block(
    bands: Dict[str, np.ndarray], _params: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate multispectral Blue/NIR Water Band Ratio (``blue / nir``)."""
    blue = bands["blue"]
    nir = bands["nir"]
    valid = nir != 0.0
    out = np.zeros_like(nir, dtype=np.float64)
    np.divide(blue, nir, out=out, where=valid)
    return out, valid


def _calc_ndwi_gao_block(
    bands: Dict[str, np.ndarray], _params: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Gao (1996) NDWI / NDMI (``(nir - swir1) / (nir + swir1)``)."""
    nir = bands["nir"]
    swir1 = bands["swir1"]
    denom = nir + swir1
    valid = denom != 0.0
    out = np.zeros_like(denom, dtype=np.float64)
    np.divide(nir - swir1, denom, out=out, where=valid)
    return out, valid


def _calc_ndwi_mcfeeters_block(
    bands: Dict[str, np.ndarray], _params: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate McFeeters (1996) NDWI (``(green - nir) / (green + nir)``)."""
    green = bands["green"]
    nir = bands["nir"]
    denom = green + nir
    valid = denom != 0.0
    out = np.zeros_like(denom, dtype=np.float64)
    np.divide(green - nir, denom, out=out, where=valid)
    return out, valid


def _calc_mndwi_block(
    bands: Dict[str, np.ndarray], _params: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Xu (2006) MNDWI / NDSI (``(green - swir1) / (green + swir1)``)."""
    green = bands["green"]
    swir1 = bands["swir1"]
    denom = green + swir1
    valid = denom != 0.0
    out = np.zeros_like(denom, dtype=np.float64)
    np.divide(green - swir1, denom, out=out, where=valid)
    return out, valid


def _calc_awei_nsh_block(
    bands: Dict[str, np.ndarray], _params: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Automated Water Extraction Index (non-shadow)."""
    out = (
        4.0 * (bands["green"] - bands["swir1"])
        - (0.25 * bands["nir"] + 2.75 * bands["swir2"])
    )
    return out, np.ones(out.shape, dtype=bool)


def _calc_awei_sh_block(
    bands: Dict[str, np.ndarray], _params: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Automated Water Extraction Index (shadow)."""
    out = (
        bands["blue"]
        + 2.5 * bands["green"]
        - 1.5 * (bands["nir"] + bands["swir1"])
        - 0.25 * bands["swir2"]
    )
    return out, np.ones(out.shape, dtype=bool)


def _calc_wi2015_block(
    bands: Dict[str, np.ndarray], _params: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Fisher et al. (2016) Water Index 2015."""
    out = (
        1.7204
        + 171.0 * bands["green"]
        + 3.0 * bands["red"]
        - 70.0 * bands["nir"]
        - 45.0 * bands["swir1"]
        - 71.0 * bands["swir2"]
    )
    return out, np.ones(out.shape, dtype=bool)


def _calc_whiteness_block(
    bands: Dict[str, np.ndarray], _params: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate cloud Whiteness index from visible bands."""
    blue = bands["blue"]
    green = bands["green"]
    red = bands["red"]
    mean_vis = (blue + green + red) / 3.0
    valid = mean_vis != 0.0
    numer = (
        np.abs(blue - mean_vis)
        + np.abs(green - mean_vis)
        + np.abs(red - mean_vis)
    )
    out = np.zeros_like(mean_vis, dtype=np.float64)
    np.divide(numer, mean_vis, out=out, where=valid)
    return out, valid


def _calc_brightness_block(
    bands: Dict[str, np.ndarray], _params: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate visible Brightness index (``(blue + green + red) / 3``)."""
    brightness = (bands["blue"] + bands["green"] + bands["red"]) / 3.0
    valid = np.ones(brightness.shape, dtype=bool)
    return brightness, valid


def _calc_brightness_scaled_block(
    bands: Dict[str, np.ndarray], params: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate standard-deviation stretched visible Brightness scaled to ``[0, 1]``."""
    brightness = (bands["blue"] + bands["green"] + bands["red"]) / 3.0
    in_min = float(params["in_min"])
    in_max = float(params["in_max"])
    if in_max > in_min:
        scaled = np.clip((brightness - in_min) / (in_max - in_min), 0.0, 1.0)
    else:
        scaled = np.zeros_like(brightness, dtype=np.float64)
    valid = np.ones(brightness.shape, dtype=bool)
    return scaled, valid


def _calc_ctvi_block(
    bands: Dict[str, np.ndarray], _params: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Corrected Transformed Vegetation Index (CTVI)."""
    nir = bands["nir"]
    red = bands["red"]
    denom = nir + red
    valid = denom != 0.0
    ndvi = np.zeros_like(denom, dtype=np.float64)
    np.divide(nir - red, denom, out=ndvi, where=valid)
    shifted = ndvi + 0.5
    valid = valid & (shifted != 0.0)
    out = np.zeros_like(denom, dtype=np.float64)
    np.divide(shifted, np.sqrt(np.abs(shifted)), out=out, where=valid)
    return out, valid


def _calc_tvi_block(
    bands: Dict[str, np.ndarray], _params: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Transformed Vegetation Index (TVI)."""
    ndvi, valid = _calc_ndvi_block(bands, _params)
    shifted = ndvi + 0.5
    valid = valid & (shifted >= 0.0)
    out = np.zeros_like(shifted, dtype=np.float64)
    np.sqrt(np.where(valid, shifted, 0.0), out=out, where=valid)
    return out, valid


def _calc_ttvi_block(
    bands: Dict[str, np.ndarray], _params: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Thiam's Transformed Vegetation Index (TTVI)."""
    ndvi, valid = _calc_ndvi_block(bands, _params)
    out = np.zeros_like(ndvi, dtype=np.float64)
    np.sqrt(np.abs(ndvi + 0.5), out=out, where=valid)
    return out, valid


def _calc_nbr_block(
    bands: Dict[str, np.ndarray], _params: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Normalised Burn Ratio (NBR = ``(nir - swir2) / (nir + swir2)``)."""
    nir = bands["nir"]
    swir2 = bands["swir2"]
    denom = nir + swir2
    valid = denom != 0.0
    out = np.zeros_like(denom, dtype=np.float64)
    np.divide(nir - swir2, denom, out=out, where=valid)
    return out, valid


def _calc_nbr2_block(
    bands: Dict[str, np.ndarray], _params: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Normalised Burn Ratio 2 (NBR2 = ``(swir1 - swir2) / (swir1 + swir2)``)."""
    swir1 = bands["swir1"]
    swir2 = bands["swir2"]
    denom = swir1 + swir2
    valid = denom != 0.0
    out = np.zeros_like(denom, dtype=np.float64)
    np.divide(swir1 - swir2, denom, out=out, where=valid)
    return out, valid


def _calc_bai_block(
    bands: Dict[str, np.ndarray], _params: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Burn Area Index (BAI)."""
    red = bands["red"]
    nir = bands["nir"]
    denom = ((0.1 - red) ** 2) + ((0.06 - nir) ** 2)
    valid = denom != 0.0
    out = np.zeros_like(denom, dtype=np.float64)
    np.divide(1.0, denom, out=out, where=valid)
    return out, valid


def _calc_mvi_block(
    bands: Dict[str, np.ndarray], _params: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Mangrove Vegetation Index (MVI = ``(nir - green) / (swir1 - green)``)."""
    green = bands["green"]
    nir = bands["nir"]
    swir1 = bands["swir1"]
    denom = swir1 - green
    valid = denom != 0.0
    out = np.zeros_like(denom, dtype=np.float64)
    np.divide(nir - green, denom, out=out, where=valid)
    return out, valid


def _calc_evi_block(
    bands: Dict[str, np.ndarray], params: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Enhanced Vegetation Index (EVI)."""
    blue = bands["blue"]
    red = bands["red"]
    nir = bands["nir"]
    gain_g = float(params.get("g", 2.5))
    coef_c1 = float(params.get("c1", 6.0))
    coef_c2 = float(params.get("c2", 7.5))
    coef_l = float(params.get("l", 1.0))

    denom = nir + (coef_c1 * red) - (coef_c2 * blue) + coef_l
    valid = denom != 0.0
    out = np.zeros_like(denom, dtype=np.float64)
    np.divide(gain_g * (nir - red), denom, out=out, where=valid)
    return out, valid


def _calc_evi2_block(
    bands: Dict[str, np.ndarray], params: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate two-band Enhanced Vegetation Index (EVI2)."""
    red = bands["red"]
    nir = bands["nir"]
    gain_g = float(params.get("g", 2.5))
    coef_c = float(params.get("c", 2.4))
    coef_l = float(params.get("l", 1.0))

    denom = nir + (coef_c * red) + coef_l
    valid = denom != 0.0
    out = np.zeros_like(denom, dtype=np.float64)
    np.divide(gain_g * (nir - red), denom, out=out, where=valid)
    return out, valid


def _calc_savi_block(
    bands: Dict[str, np.ndarray], params: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Soil Adjusted Vegetation Index (SAVI)."""
    red = bands["red"]
    nir = bands["nir"]
    coef_l = float(params.get("l", 0.5))
    denom = nir + red + coef_l
    valid = denom != 0.0
    out = np.zeros_like(denom, dtype=np.float64)
    np.divide((nir - red) * (1.0 + coef_l), denom, out=out, where=valid)
    return out, valid


def _calc_osavi_block(
    bands: Dict[str, np.ndarray], params: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Optimised Soil Adjusted Vegetation Index (OSAVI)."""
    red = bands["red"]
    nir = bands["nir"]
    coef_y = float(params.get("y", params.get("l", 0.16)))
    denom = nir + red + coef_y
    valid = denom != 0.0
    out = np.zeros_like(denom, dtype=np.float64)
    np.divide((1.0 + coef_y) * (nir - red), denom, out=out, where=valid)
    return out, valid


def _calc_msavi2_block(
    bands: Dict[str, np.ndarray], _params: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Modified Soil Adjusted Vegetation Index 2 (MSAVI / MSAVI2)."""
    red = bands["red"]
    nir = bands["nir"]
    two_nir_plus_one = (2.0 * nir) + 1.0
    disc = (two_nir_plus_one ** 2) - (8.0 * (nir - red))
    valid = disc >= 0.0
    out = np.zeros_like(disc, dtype=np.float64)
    root = np.sqrt(np.where(valid, disc, 0.0))
    out = np.where(valid, 0.5 * (two_nir_plus_one - root), 0.0)
    return out, valid


def _calc_satvi_block(
    bands: Dict[str, np.ndarray], params: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Soil Adjusted Total Vegetation Index (SATVI)."""
    red = bands["red"]
    swir1 = bands["swir1"]
    swir2 = bands["swir2"]
    coef_l = float(params.get("l", 0.5))
    denom = swir1 + red + coef_l
    valid = denom != 0.0
    term = np.zeros_like(denom, dtype=np.float64)
    np.divide((swir1 - red) * (1.0 + coef_l), denom, out=term, where=valid)
    out = np.where(valid, term - (0.5 * swir2), 0.0)
    return out, valid


def _calc_gndvi_block(
    bands: Dict[str, np.ndarray], _params: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Green Normalised Difference Vegetation Index (GNDVI)."""
    green = bands["green"]
    nir = bands["nir"]
    denom = nir + green
    valid = denom != 0.0
    out = np.zeros_like(denom, dtype=np.float64)
    np.divide(nir - green, denom, out=out, where=valid)
    return out, valid


def _calc_arvi_block(
    bands: Dict[str, np.ndarray], params: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Atmospherically Resistant Vegetation Index (ARVI)."""
    blue = bands["blue"]
    red = bands["red"]
    nir = bands["nir"]
    gamma = float(params.get("gamma", 1.0))
    rb_term = red - gamma * (blue - red)
    denom = nir + rb_term
    valid = denom != 0.0
    out = np.zeros_like(denom, dtype=np.float64)
    np.divide(nir - rb_term, denom, out=out, where=valid)
    return out, valid


def _calc_gemi_block(
    bands: Dict[str, np.ndarray], _params: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Global Environmental Monitoring Index (GEMI)."""
    red = bands["red"]
    nir = bands["nir"]
    eta_denom = nir + red + 0.5
    one_minus_red = 1.0 - red
    valid = (eta_denom != 0.0) & (one_minus_red != 0.0)
    eta_num = (2.0 * ((nir ** 2) - (red ** 2))) + (1.5 * nir) + (0.5 * red)
    eta = np.zeros_like(eta_denom, dtype=np.float64)
    np.divide(eta_num, eta_denom, out=eta, where=valid)
    tail = np.zeros_like(one_minus_red, dtype=np.float64)
    np.divide(red - 0.125, one_minus_red, out=tail, where=valid)
    out = np.where(valid, (eta * (1.0 - (0.25 * eta))) - tail, 0.0)
    return out, valid


def _calc_sr_block(
    bands: Dict[str, np.ndarray], _params: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Simple Ratio / Ratio Vegetation Index (``nir / red``)."""
    red = bands["red"]
    nir = bands["nir"]
    valid = red != 0.0
    out = np.zeros_like(red, dtype=np.float64)
    np.divide(nir, red, out=out, where=valid)
    return out, valid


def _calc_rvi_red_nir_block(
    bands: Dict[str, np.ndarray], _params: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate inverse Ratio Vegetation Index (``red / nir``)."""
    red = bands["red"]
    nir = bands["nir"]
    valid = nir != 0.0
    out = np.zeros_like(nir, dtype=np.float64)
    np.divide(red, nir, out=out, where=valid)
    return out, valid


def _calc_dvi_block(
    bands: Dict[str, np.ndarray], _params: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Difference Vegetation Index (``nir - red``)."""
    out = bands["nir"] - bands["red"]
    return out, np.ones(out.shape, dtype=bool)


def _calc_wdvi_block(
    bands: Dict[str, np.ndarray], params: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Weighted Difference Vegetation Index (``nir - s * red``)."""
    slope_s = float(params.get("s", 1.0))
    out = bands["nir"] - (slope_s * bands["red"])
    return out, np.ones(out.shape, dtype=bool)


def _calc_nrvi_block(
    bands: Dict[str, np.ndarray], _params: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Normalised Ratio Vegetation Index (Baret & Guyot 1991)."""
    red = bands["red"]
    nir = bands["nir"]
    valid = nir != 0.0
    rvi = np.zeros_like(nir, dtype=np.float64)
    np.divide(red, nir, out=rvi, where=valid)
    denom = rvi + 1.0
    valid = valid & (denom != 0.0)
    out = np.zeros_like(denom, dtype=np.float64)
    np.divide(rvi - 1.0, denom, out=out, where=valid)
    return out, valid


def _calc_slavi_block(
    bands: Dict[str, np.ndarray], _params: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Specific Leaf Area Vegetation Index (``nir / (red + swir2)``)."""
    denom = bands["red"] + bands["swir2"]
    valid = denom != 0.0
    out = np.zeros_like(denom, dtype=np.float64)
    np.divide(bands["nir"], denom, out=out, where=valid)
    return out, valid


def _calc_kndvi_block(
    bands: Dict[str, np.ndarray], _params: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Kernel NDVI (Camps-Valls et al. 2021)."""
    ndvi, valid = _calc_ndvi_block(bands, _params)
    out = np.where(valid, np.tanh(ndvi ** 2), 0.0)
    return out, valid


def _calc_nirv_block(
    bands: Dict[str, np.ndarray], _params: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Near-Infrared Reflectance of Vegetation (``NDVI * nir``)."""
    ndvi, valid = _calc_ndvi_block(bands, _params)
    out = np.where(valid, ndvi * bands["nir"], 0.0)
    return out, valid


def _calc_ndre_block(
    bands: Dict[str, np.ndarray], _params: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Normalised Difference Red-Edge Index."""
    nir = bands["nir"]
    red_edge = bands["red_edge"]
    denom = nir + red_edge
    valid = denom != 0.0
    out = np.zeros_like(denom, dtype=np.float64)
    np.divide(nir - red_edge, denom, out=out, where=valid)
    return out, valid


def _calc_cire_block(
    bands: Dict[str, np.ndarray], _params: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Chlorophyll Index Red-Edge (``(nir / red_edge) - 1``)."""
    red_edge = bands["red_edge"]
    valid = red_edge != 0.0
    out = np.zeros_like(red_edge, dtype=np.float64)
    np.divide(bands["nir"], red_edge, out=out, where=valid)
    out = np.where(valid, out - 1.0, 0.0)
    return out, valid


def _calc_cigreen_block(
    bands: Dict[str, np.ndarray], _params: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Chlorophyll Index Green (``(nir / green) - 1``)."""
    green = bands["green"]
    valid = green != 0.0
    out = np.zeros_like(green, dtype=np.float64)
    np.divide(bands["nir"], green, out=out, where=valid)
    out = np.where(valid, out - 1.0, 0.0)
    return out, valid


def _calc_vari_block(
    bands: Dict[str, np.ndarray], _params: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Visible Atmospherically Resistant Index (VARI)."""
    green = bands["green"]
    red = bands["red"]
    blue = bands["blue"]
    denom = green + red - blue
    valid = denom != 0.0
    out = np.zeros_like(denom, dtype=np.float64)
    np.divide(green - red, denom, out=out, where=valid)
    return out, valid


def _calc_gli_block(
    bands: Dict[str, np.ndarray], _params: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Green Leaf Index (GLI)."""
    green = bands["green"]
    red = bands["red"]
    blue = bands["blue"]
    denom = (2.0 * green) + red + blue
    valid = denom != 0.0
    out = np.zeros_like(denom, dtype=np.float64)
    np.divide((2.0 * green) - red - blue, denom, out=out, where=valid)
    return out, valid


def _calc_exg_block(
    bands: Dict[str, np.ndarray], _params: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Excess Green Index (``2 * green - red - blue``)."""
    out = (2.0 * bands["green"]) - bands["red"] - bands["blue"]
    return out, np.ones(out.shape, dtype=bool)


def _calc_remi_block(
    bands: Dict[str, np.ndarray], _params: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Red-Edge Mangrove Index (``(red_edge - red) / (swir1 + green)``)."""
    green = bands["green"]
    red = bands["red"]
    red_edge = bands["red_edge"]
    swir1 = bands["swir1"]
    denom = swir1 + green
    valid = denom != 0.0
    out = np.zeros_like(denom, dtype=np.float64)
    np.divide(red_edge - red, denom, out=out, where=valid)
    return out, valid


def _calc_bsi_block(
    bands: Dict[str, np.ndarray], _params: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate Bare Soil Index
    (``((swir1 + red) - (nir + blue)) / ((swir1 + red) + (nir + blue))``).
    """
    blue = bands["blue"]
    red = bands["red"]
    nir = bands["nir"]
    swir1 = bands["swir1"]
    term1 = swir1 + red
    term2 = nir + blue
    denom = term1 + term2
    valid = denom != 0.0
    out = np.zeros_like(denom, dtype=np.float64)
    np.divide(term1 - term2, denom, out=out, where=valid)
    return out, valid


def _calc_ndbi_block(
    bands: Dict[str, np.ndarray], _params: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Normalised Difference Built-up Index (``(swir1 - nir) / (swir1 + nir)``)."""
    swir1 = bands["swir1"]
    nir = bands["nir"]
    denom = swir1 + nir
    valid = denom != 0.0
    out = np.zeros_like(denom, dtype=np.float64)
    np.divide(swir1 - nir, denom, out=out, where=valid)
    return out, valid


def _calc_si_block(
    bands: Dict[str, np.ndarray], _params: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Shadow Index (``((1 - blue) * (1 - green) * (1 - red)) ** (1/3)``)."""
    prod = (1.0 - bands["blue"]) * (1.0 - bands["green"]) * (1.0 - bands["red"])
    valid = prod >= 0.0
    out = np.zeros_like(prod, dtype=np.float64)
    np.power(np.where(valid, prod, 0.0), 1.0 / 3.0, out=out, where=valid)
    return out, valid


def _calc_ndgi_block(
    bands: Dict[str, np.ndarray], _params: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Normalised Difference Glacier Index (``(green - red) / (green + red)``)."""
    green = bands["green"]
    red = bands["red"]
    denom = green + red
    valid = denom != 0.0
    out = np.zeros_like(denom, dtype=np.float64)
    np.divide(green - red, denom, out=out, where=valid)
    return out, valid


def _calc_npcri_block(
    bands: Dict[str, np.ndarray], _params: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate Normalised Pigment Chlorophyll Ratio Index
    (``(red - blue) / (red + blue)``).
    """
    red = bands["red"]
    blue = bands["blue"]
    denom = red + blue
    valid = denom != 0.0
    out = np.zeros_like(denom, dtype=np.float64)
    np.divide(red - blue, denom, out=out, where=valid)
    return out, valid


def _calc_ndbsi_block(
    bands: Dict[str, np.ndarray], _params: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Normalised Difference Bare Soil Index (NDBSI)."""
    blue = bands["blue"]
    green = bands["green"]
    red = bands["red"]
    nir = bands["nir"]
    swir1 = bands["swir1"]

    v_term = red - green
    r_denom = 3.0 * np.abs(nir - red)
    out_denom = swir1 + blue
    valid = (r_denom != 0.0) & (out_denom != 0.0)

    r_term = np.zeros_like(r_denom, dtype=np.float64)
    np.divide(swir1 - nir, r_denom, out=r_term, where=valid)
    r_term = 1.0 - r_term
    k_term = v_term * r_term

    base = np.zeros_like(out_denom, dtype=np.float64)
    np.divide(swir1 - blue, out_denom, out=base, where=valid)
    neg_k = np.zeros_like(out_denom, dtype=np.float64)
    np.divide(-np.abs(swir1 - blue), out_denom, out=neg_k, where=valid)

    out = np.where(k_term < 0.0, neg_k, base)
    return out, valid


def _calc_cmr_block(
    bands: Dict[str, np.ndarray], _params: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Clay Minerals Ratio (``swir1 / swir2``)."""
    swir1 = bands["swir1"]
    swir2 = bands["swir2"]
    valid = swir2 != 0.0
    out = np.zeros_like(swir2, dtype=np.float64)
    np.divide(swir1, swir2, out=out, where=valid)
    return out, valid


def _calc_fmr_block(
    bands: Dict[str, np.ndarray], _params: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Ferrous Minerals Ratio (``swir1 / nir``)."""
    nir = bands["nir"]
    swir1 = bands["swir1"]
    valid = nir != 0.0
    out = np.zeros_like(nir, dtype=np.float64)
    np.divide(swir1, nir, out=out, where=valid)
    return out, valid


def _calc_ior_block(
    bands: Dict[str, np.ndarray], _params: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Iron Oxide Ratio (``red / blue``)."""
    blue = bands["blue"]
    red = bands["red"]
    valid = blue != 0.0
    out = np.zeros_like(blue, dtype=np.float64)
    np.divide(red, blue, out=out, where=valid)
    return out, valid


_INDEX_REGISTRY: Dict[str, _IndexSpec] = {
    # Vegetation & Canopy
    "ndvi": _IndexSpec(
        "NDVI", ("red", "nir"), "(nir - red) / (nir + red)", _calc_ndvi_block
    ),
    "evi": _IndexSpec(
        "EVI",
        ("blue", "red", "nir"),
        "G * ((nir - red) / (nir + C1 * red - C2 * blue + L))",
        _calc_evi_block,
    ),
    "evi2": _IndexSpec(
        "EVI2",
        ("red", "nir"),
        "G * ((nir - red) / (nir + C * red + L))",
        _calc_evi2_block,
    ),
    "gndvi": _IndexSpec(
        "GNDVI",
        ("green", "nir"),
        "(nir - green) / (nir + green)",
        _calc_gndvi_block,
    ),
    "savi": _IndexSpec(
        "SAVI",
        ("red", "nir"),
        "((nir - red) / (nir + red + L)) * (1 + L)",
        _calc_savi_block,
    ),
    "osavi": _IndexSpec(
        "OSAVI",
        ("red", "nir"),
        "(1 + Y) * ((nir - red) / (nir + red + Y))",
        _calc_osavi_block,
    ),
    "msavi2": _IndexSpec(
        "MSAVI2",
        ("red", "nir"),
        "(2 * nir + 1 - sqrt((2 * nir + 1)^2 - 8 * (nir - red))) / 2",
        _calc_msavi2_block,
    ),
    "msavi": _IndexSpec(
        "MSAVI",
        ("red", "nir"),
        "(2 * nir + 1 - sqrt((2 * nir + 1)^2 - 8 * (nir - red))) / 2",
        _calc_msavi2_block,
    ),
    "satvi": _IndexSpec(
        "SATVI",
        ("red", "swir1", "swir2"),
        "((swir1 - red) / (swir1 + red + L)) * (1 + L) - (swir2 / 2)",
        _calc_satvi_block,
    ),
    "sr": _IndexSpec("SR", ("red", "nir"), "nir / red", _calc_sr_block),
    "rvi": _IndexSpec("RVI", ("red", "nir"), "nir / red", _calc_sr_block),
    "rvi_nir_red": _IndexSpec(
        "RVI_NIR_Red", ("red", "nir"), "nir / red", _calc_sr_block
    ),
    "rvi_red_nir": _IndexSpec(
        "RVI_Red_NIR", ("red", "nir"), "red / nir", _calc_rvi_red_nir_block
    ),
    "dvi": _IndexSpec("DVI", ("red", "nir"), "nir - red", _calc_dvi_block),
    "wdvi": _IndexSpec(
        "WDVI", ("red", "nir"), "nir - s * red", _calc_wdvi_block
    ),
    "nrvi": _IndexSpec(
        "NRVI",
        ("red", "nir"),
        "((red / nir) - 1) / ((red / nir) + 1)",
        _calc_nrvi_block,
    ),
    "ctvi": _IndexSpec(
        "CTVI",
        ("red", "nir"),
        "(NDVI + 0.5) / sqrt(abs(NDVI + 0.5))",
        _calc_ctvi_block,
    ),
    "tvi": _IndexSpec(
        "TVI", ("red", "nir"), "sqrt(NDVI + 0.5)", _calc_tvi_block
    ),
    "ttvi": _IndexSpec(
        "TTVI", ("red", "nir"), "sqrt(abs(NDVI + 0.5))", _calc_ttvi_block
    ),
    "gemi": _IndexSpec(
        "GEMI",
        ("red", "nir"),
        "eta * (1 - 0.25 * eta) - ((red - 0.125) / (1 - red)), "
        "eta = (2 * (nir^2 - red^2) + 1.5 * nir + 0.5 * red) / (nir + red + 0.5)",
        _calc_gemi_block,
    ),
    "arvi": _IndexSpec(
        "ARVI",
        ("blue", "red", "nir"),
        "(nir - (red - gamma * (blue - red))) / "
        "(nir + (red - gamma * (blue - red)))",
        _calc_arvi_block,
    ),
    "slavi": _IndexSpec(
        "SLAVI",
        ("red", "nir", "swir2"),
        "nir / (red + swir2)",
        _calc_slavi_block,
    ),
    "kndvi": _IndexSpec(
        "kNDVI",
        ("red", "nir"),
        "tanh(((nir - red) / (nir + red))^2)",
        _calc_kndvi_block,
    ),
    "nirv": _IndexSpec(
        "NIRv",
        ("red", "nir"),
        "((nir - red) / (nir + red)) * nir",
        _calc_nirv_block,
    ),
    "ndre": _IndexSpec(
        "NDRE",
        ("red_edge", "nir"),
        "(nir - red_edge) / (nir + red_edge)",
        _calc_ndre_block,
    ),
    "cire": _IndexSpec(
        "CIre", ("red_edge", "nir"), "(nir / red_edge) - 1", _calc_cire_block
    ),
    "cigreen": _IndexSpec(
        "CIgreen", ("green", "nir"), "(nir / green) - 1", _calc_cigreen_block
    ),
    "npcri": _IndexSpec(
        "NPCRI",
        ("red", "blue"),
        "(red - blue) / (red + blue)",
        _calc_npcri_block,
    ),
    "remi": _IndexSpec(
        "REMI",
        ("green", "red", "red_edge", "swir1"),
        "(red_edge - red) / (swir1 + green)",
        _calc_remi_block,
    ),
    "mvi": _IndexSpec(
        "MVI",
        ("green", "nir", "swir1"),
        "(nir - green) / (swir1 - green)",
        _calc_mvi_block,
    ),
    "vari": _IndexSpec(
        "VARI",
        ("blue", "green", "red"),
        "(green - red) / (green + red - blue)",
        _calc_vari_block,
    ),
    "gli": _IndexSpec(
        "GLI",
        ("blue", "green", "red"),
        "(2 * green - red - blue) / (2 * green + red + blue)",
        _calc_gli_block,
    ),
    "exg": _IndexSpec(
        "ExG",
        ("blue", "green", "red"),
        "2 * green - red - blue",
        _calc_exg_block,
    ),
    # Water, Snow & Moisture
    "ndwi_mcfeeters": _IndexSpec(
        "NDWI_McFeeters",
        ("green", "nir"),
        "(green - nir) / (green + nir)",
        _calc_ndwi_mcfeeters_block,
    ),
    "ndwi": _IndexSpec(
        "NDWI",
        ("green", "nir"),
        "(green - nir) / (green + nir)",
        _calc_ndwi_mcfeeters_block,
    ),
    "gndwi": _IndexSpec(
        "GNDWI",
        ("green", "nir"),
        "(green - nir) / (green + nir)",
        _calc_ndwi_mcfeeters_block,
    ),
    "ndwi_gao": _IndexSpec(
        "NDWI_Gao",
        ("nir", "swir1"),
        "(nir - swir1) / (nir + swir1)",
        _calc_ndwi_gao_block,
    ),
    "ndmi": _IndexSpec(
        "NDMI",
        ("nir", "swir1"),
        "(nir - swir1) / (nir + swir1)",
        _calc_ndwi_gao_block,
    ),
    "mndwi": _IndexSpec(
        "MNDWI",
        ("green", "swir1"),
        "(green - swir1) / (green + swir1)",
        _calc_mndwi_block,
    ),
    "gmndwi": _IndexSpec(
        "GMNDWI",
        ("green", "swir1"),
        "(green - swir1) / (green + swir1)",
        _calc_mndwi_block,
    ),
    "ndsi": _IndexSpec(
        "NDSI",
        ("green", "swir1"),
        "(green - swir1) / (green + swir1)",
        _calc_mndwi_block,
    ),
    "ndgi": _IndexSpec(
        "NDGI",
        ("green", "red"),
        "(green - red) / (green + red)",
        _calc_ndgi_block,
    ),
    "awei_nsh": _IndexSpec(
        "AWEI_nsh",
        ("green", "nir", "swir1", "swir2"),
        "4 * (green - swir1) - (0.25 * nir + 2.75 * swir2)",
        _calc_awei_nsh_block,
    ),
    "awei_sh": _IndexSpec(
        "AWEI_sh",
        ("blue", "green", "nir", "swir1", "swir2"),
        "blue + 2.5 * green - 1.5 * (nir + swir1) - 0.25 * swir2",
        _calc_awei_sh_block,
    ),
    "wi2015": _IndexSpec(
        "WI2015",
        ("green", "red", "nir", "swir1", "swir2"),
        "1.7204 + 171 * green + 3 * red - 70 * nir - 45 * swir1 - 71 * swir2",
        _calc_wi2015_block,
    ),
    "wbi_bluenir": _IndexSpec(
        "WBI_BlueNIR",
        ("blue", "nir"),
        "blue / nir",
        _calc_wbi_bluenir_block,
    ),
    "wbi": _IndexSpec(
        "WBI", ("blue", "nir"), "blue / nir", _calc_wbi_bluenir_block
    ),
    # Burn, Soil, Built-up, Cloud & Minerals
    "nbr": _IndexSpec(
        "NBR", ("nir", "swir2"), "(nir - swir2) / (nir + swir2)", _calc_nbr_block
    ),
    "nbri": _IndexSpec(
        "NBRI",
        ("nir", "swir2"),
        "(nir - swir2) / (nir + swir2)",
        _calc_nbr_block,
    ),
    "nbr2": _IndexSpec(
        "NBR2",
        ("swir1", "swir2"),
        "(swir1 - swir2) / (swir1 + swir2)",
        _calc_nbr2_block,
    ),
    "bai": _IndexSpec(
        "BAI",
        ("red", "nir"),
        "1 / ((0.1 - red)^2 + (0.06 - nir)^2)",
        _calc_bai_block,
    ),
    "bsi": _IndexSpec(
        "BSI",
        ("blue", "red", "nir", "swir1"),
        "((swir1 + red) - (nir + blue)) / ((swir1 + red) + (nir + blue))",
        _calc_bsi_block,
    ),
    "ndbsi": _IndexSpec(
        "NDBSI",
        ("blue", "green", "red", "nir", "swir1"),
        "Liu et al. (2022): v = red - green, "
        "r = 1 - (swir1 - nir) / (3 * |nir - red|), k = v * r; "
        "-|swir1 - blue| / (swir1 + blue) if k < 0 else "
        "(swir1 - blue) / (swir1 + blue)",
        _calc_ndbsi_block,
    ),
    "ndbi": _IndexSpec(
        "NDBI",
        ("nir", "swir1"),
        "(swir1 - nir) / (swir1 + nir)",
        _calc_ndbi_block,
    ),
    "brightness": _IndexSpec(
        "Brightness",
        ("blue", "green", "red"),
        "(blue + green + red) / 3",
        _calc_brightness_block,
    ),
    "brightness_scaled": _IndexSpec(
        "Brightness_Scaled",
        ("blue", "green", "red"),
        "clip((Brightness - (mean - k * std)) / (2 * k * std), 0, 1)",
        _calc_brightness_scaled_block,
        requires_brightness_stats=True,
    ),
    "whiteness": _IndexSpec(
        "Whiteness",
        ("blue", "green", "red"),
        "(|blue - mean| + |green - mean| + |red - mean|) / mean, "
        "mean = (blue + green + red) / 3",
        _calc_whiteness_block,
    ),
    "si": _IndexSpec(
        "SI",
        ("blue", "green", "red"),
        "((1 - blue) * (1 - green) * (1 - red))^(1/3)",
        _calc_si_block,
    ),
    "cmr": _IndexSpec(
        "CMR", ("swir1", "swir2"), "swir1 / swir2", _calc_cmr_block
    ),
    "fmr": _IndexSpec(
        "FMR", ("nir", "swir1"), "swir1 / nir", _calc_fmr_block
    ),
    "ior": _IndexSpec(
        "IOR", ("blue", "red"), "red / blue", _calc_ior_block
    ),
}


def get_available_indices() -> List[str]:
    """
    Return the sorted list of spectral index names supported by
    :func:`calc_indices`.

    :return: Sorted list of supported spectral index names.

    .. code:: python

        import rsgispy.imagecalc.calcindices

        names = rsgispy.imagecalc.calcindices.get_available_indices()
    """
    return sorted(spec.canonical_name for spec in _INDEX_REGISTRY.values())


def _lookup_index_spec(index_name: str) -> _IndexSpec:
    """Validate and resolve ``index_name`` in ``_INDEX_REGISTRY``."""
    if not isinstance(index_name, str) or not index_name.strip():
        raise rsgispy.RSGISPyException(
            "index_name must be a non-empty string."
        )
    key = index_name.strip().lower()
    if key not in _INDEX_REGISTRY:
        raise rsgispy.RSGISPyException(
            f"Unknown spectral index '{index_name}'. "
            f"Supported indices: {', '.join(get_available_indices())}."
        )
    return _INDEX_REGISTRY[key]


def get_index_req_bands(index_name: str) -> Tuple[str, ...]:
    """
    Return the tuple of canonical band names required to compute ``index_name``.

    :param index_name: Name of the spectral index (case-insensitive).
    :return: Tuple of canonical band names (e.g., ``('red', 'nir')``).

    .. code:: python

        import rsgispy.imagecalc.calcindices

        req_bands = rsgispy.imagecalc.calcindices.get_index_req_bands("NDVI")
    """
    return _lookup_index_spec(index_name).req_bands


def get_index_equation(index_name: str) -> str:
    """
    Return the mathematical equation string for ``index_name``.

    :param index_name: Name of the spectral index (case-insensitive).
    :return: Equation string using canonical band names.

    .. code:: python

        import rsgispy.imagecalc.calcindices

        eq = rsgispy.imagecalc.calcindices.get_index_equation("NDVI")
    """
    return _lookup_index_spec(index_name).equation


def _normalise_band_names(
    band_names: Mapping[int, str], n_bands: int, input_img: str
) -> Dict[str, int]:
    """
    Validate and normalise ``{band_idx: band_name}`` into
    ``{canonical_band_name: band_idx}``.
    """
    if not isinstance(band_names, Mapping) or not band_names:
        raise rsgispy.RSGISPyException(
            "band_names must be a non-empty mapping of 1-based band index "
            "to band name (e.g., {1: 'blue', 2: 'green', 5: 'swir1'})."
        )

    canonical_to_idx: Dict[str, int] = {}
    for raw_idx, raw_name in band_names.items():
        if isinstance(raw_idx, bool) or not isinstance(raw_idx, (int, np.integer)):
            raise rsgispy.RSGISPyException(
                f"Band index keys in band_names must be integers, got {raw_idx!r}."
            )
        band_idx = int(raw_idx)
        validate_band_index(band_idx, n_bands, input_img)

        if not isinstance(raw_name, str) or not raw_name.strip():
            raise rsgispy.RSGISPyException(
                f"Band name for band {band_idx} must be a non-empty string."
            )
        norm_name = raw_name.strip().lower()
        canonical_name = _BAND_ALIASES.get(norm_name, norm_name)
        if canonical_name in canonical_to_idx:
            raise rsgispy.RSGISPyException(
                f"Duplicate band name '{canonical_name}' mapped to bands "
                f"{canonical_to_idx[canonical_name]} and {band_idx}."
            )
        canonical_to_idx[canonical_name] = band_idx

    return canonical_to_idx


def _resolve_requested_indices(
    indices: Sequence[str], canonical_to_idx: Mapping[str, int]
) -> List[Tuple[str, _IndexSpec]]:
    """Validate requested index names and verify required bands exist."""
    if (
        isinstance(indices, (str, bytes))
        or not isinstance(indices, Sequence)
        or not indices
    ):
        raise rsgispy.RSGISPyException(
            "indices must be a non-empty sequence of spectral index names."
        )

    resolved: List[Tuple[str, _IndexSpec]] = []
    for raw_idx_name in indices:
        if not isinstance(raw_idx_name, str) or not raw_idx_name.strip():
            raise rsgispy.RSGISPyException(
                "Each entry in indices must be a non-empty string."
            )
        out_band_name = raw_idx_name.strip()
        spec = _lookup_index_spec(out_band_name)
        missing = [b for b in spec.req_bands if b not in canonical_to_idx]
        if missing:
            raise rsgispy.RSGISPyException(
                f"Index '{spec.canonical_name}' requires band(s) "
                f"{', '.join(missing)}, which were not provided in band_names."
            )
        resolved.append((out_band_name, spec))

    return resolved


def _resolve_in_scaling(
    val: Optional[Union[float, Sequence[float], Mapping[Union[int, str], float]]],
    default: float,
    req_bands: Sequence[str],
    canonical_to_idx: Mapping[str, int],
    n_bands: int,
    label: str,
    check_nonzero: bool = False,
) -> Dict[str, float]:
    """Resolve scalar, per-image-band sequence, or mapping into per-band values."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-branches
    if val is None:
        return {b: float(default) for b in req_bands}

    if isinstance(val, (int, float, np.number)) and not isinstance(val, bool):
        f_val = float(val)
        if check_nonzero and f_val == 0.0:
            raise rsgispy.RSGISPyException(f"{label} must not be zero.")
        return {b: f_val for b in req_bands}

    if isinstance(val, Mapping):
        lower_map: Dict[Union[int, str], float] = {}
        for k, v in val.items():
            if isinstance(k, str):
                k_norm = k.strip().lower()
                lower_map[_BAND_ALIASES.get(k_norm, k_norm)] = float(v)
            elif isinstance(k, (int, np.integer)) and not isinstance(k, bool):
                lower_map[int(k)] = float(v)
            else:
                raise rsgispy.RSGISPyException(
                    f"Invalid key {k!r} in {label} mapping."
                )

        out: Dict[str, float] = {}
        for b_name in req_bands:
            b_idx = canonical_to_idx[b_name]
            if b_idx in lower_map:
                f_val = lower_map[b_idx]
            elif b_name in lower_map:
                f_val = lower_map[b_name]
            else:
                raise rsgispy.RSGISPyException(
                    f"Missing {label} value for band '{b_name}' (band {b_idx})."
                )
            if check_nonzero and f_val == 0.0:
                raise rsgispy.RSGISPyException(
                    f"{label} for band '{b_name}' must not be zero."
                )
            out[b_name] = f_val
        return out

    if isinstance(val, Sequence) and not isinstance(val, (str, bytes)):
        if len(val) != n_bands:
            raise rsgispy.RSGISPyException(
                f"{label} sequence length ({len(val)}) must match input "
                f"image band count ({n_bands})."
            )
        out = {}
        for b_name in req_bands:
            b_idx = canonical_to_idx[b_name]
            f_val = float(val[b_idx - 1])
            if check_nonzero and f_val == 0.0:
                raise rsgispy.RSGISPyException(
                    f"{label} for band '{b_name}' (band {b_idx}) must not be zero."
                )
            out[b_name] = f_val
        return out

    raise rsgispy.RSGISPyException(
        f"{label} must be a scalar, sequence, or mapping."
    )


def _resolve_out_scaling(
    val: Optional[Union[float, Sequence[float], Mapping[str, float]]],
    default: float,
    resolved_indices: Sequence[Tuple[str, _IndexSpec]],
    label: str,
    check_nonzero: bool = False,
) -> List[float]:
    """Resolve scalar, per-output-band sequence, or mapping into a list per index."""
    # pylint: disable=too-many-branches
    if val is None:
        return [float(default) for _ in resolved_indices]

    if isinstance(val, (int, float, np.number)) and not isinstance(val, bool):
        f_val = float(val)
        if check_nonzero and f_val == 0.0:
            raise rsgispy.RSGISPyException(f"{label} must not be zero.")
        return [f_val for _ in resolved_indices]

    if isinstance(val, Mapping):
        lower_map = {
            str(k).strip().lower(): float(v) for k, v in val.items()
        }
        out: List[float] = []
        for out_name, spec in resolved_indices:
            key_req = out_name.lower()
            key_can = spec.canonical_name.lower()
            if key_req in lower_map:
                f_val = lower_map[key_req]
            elif key_can in lower_map:
                f_val = lower_map[key_can]
            else:
                raise rsgispy.RSGISPyException(
                    f"Missing {label} value for index '{out_name}'."
                )
            if check_nonzero and f_val == 0.0:
                raise rsgispy.RSGISPyException(
                    f"{label} for index '{out_name}' must not be zero."
                )
            out.append(f_val)
        return out

    if isinstance(val, Sequence) and not isinstance(val, (str, bytes)):
        if len(val) != len(resolved_indices):
            raise rsgispy.RSGISPyException(
                f"{label} sequence length ({len(val)}) must match number of "
                f"requested indices ({len(resolved_indices)})."
            )
        out = [float(v) for v in val]
        if check_nonzero:
            for (out_name, _), f_val in zip(resolved_indices, out):
                if f_val == 0.0:
                    raise rsgispy.RSGISPyException(
                        f"{label} for index '{out_name}' must not be zero."
                    )
        return out

    raise rsgispy.RSGISPyException(
        f"{label} must be a scalar, sequence, or mapping."
    )


def _resolve_out_nodata(
    out_no_data_val: Optional[float], out_np_dtype: np.dtype
) -> float:
    """Resolve and validate the output no-data value for the target dtype."""
    is_int = np.issubdtype(out_np_dtype, np.integer)
    if out_no_data_val is None:
        if not is_int:
            return -999.0
        info = np.iinfo(out_np_dtype)
        if info.min <= -999 <= info.max:
            return -999.0
        if info.min == 0:
            return 0.0
        return float(info.min)

    f_val = float(out_no_data_val)
    if is_int:
        if not math.isfinite(f_val):
            raise rsgispy.RSGISPyException(
                "out_no_data_val must be finite for integer output datatypes."
            )
        info = np.iinfo(out_np_dtype)
        if f_val < info.min or f_val > info.max:
            raise rsgispy.RSGISPyException(
                f"out_no_data_val ({out_no_data_val}) is outside the valid range "
                f"[{info.min}, {info.max}] for dtype '{out_np_dtype.name}'."
            )
    return f_val


def _is_nodata_mask(arr: np.ndarray, nodata_val: Optional[float]) -> np.ndarray:
    """Return boolean mask of pixels matching ``nodata_val`` or non-finite."""
    mask = ~np.isfinite(arr)
    if nodata_val is not None:
        f_nd = float(nodata_val)
        if not math.isnan(f_nd):
            mask = mask | (arr == f_nd)
    return mask


def _resolve_index_params(
    index_params: Optional[Mapping[str, Mapping[str, float]]],
) -> Dict[str, Dict[str, float]]:
    """Normalise per-index parameter dictionaries keyed by lower-case index name."""
    if index_params is None:
        return {}
    if not isinstance(index_params, Mapping):
        raise rsgispy.RSGISPyException(
            "index_params must be a mapping of index name to parameter dict."
        )
    out: Dict[str, Dict[str, float]] = {}
    for k, v in index_params.items():
        if not isinstance(v, Mapping):
            raise rsgispy.RSGISPyException(
                f"Parameters for index '{k}' in index_params must be a mapping."
            )
        out[str(k).strip().lower()] = {
            str(pk).strip().lower(): float(pv) for pk, pv in v.items()
        }
    return out


def _read_scaled_bands(
    src: rasterio.io.DatasetReader,
    win: rasterio.windows.Window,
    band_list: Sequence[str],
    canonical_to_idx: Mapping[str, int],
    in_nodata_map: Mapping[str, Optional[float]],
    in_gain_map: Mapping[str, float],
    in_offset_map: Mapping[str, float],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Read, mask, and unscale a window of required bands."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    band_indices = [canonical_to_idx[b] for b in band_list]
    raw_block = src.read(band_indices, window=win, out_dtype=np.float64)
    scaled_bands: Dict[str, np.ndarray] = {}
    invalid_masks: Dict[str, np.ndarray] = {}

    for pos, b_name in enumerate(band_list):
        raw_arr = raw_block[pos]
        invalid = _is_nodata_mask(raw_arr, in_nodata_map[b_name])
        phys = (raw_arr - in_offset_map[b_name]) / in_gain_map[b_name]
        invalid_masks[b_name] = invalid
        scaled_bands[b_name] = phys

    return scaled_bands, invalid_masks


def _calc_brightness_stretch_bounds(
    src: rasterio.io.DatasetReader,
    canonical_to_idx: Mapping[str, int],
    in_nodata_map: Mapping[str, Optional[float]],
    in_gain_map: Mapping[str, float],
    in_offset_map: Mapping[str, float],
    stretch_std_devs: float,
    use_pbar: bool,
) -> Tuple[float, float]:
    """Compute ``(mean - k*std, mean + k*std)`` for ``Brightness_Scaled``."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    vis_bands = ("blue", "green", "red")
    total_count = 0
    shift_val: Optional[float] = None
    sum_shifted = 0.0
    sum_sq_shifted = 0.0

    for win in iter_dataset_windows(src.height, src.width, use_pbar=use_pbar):
        scaled_bands, invalid_masks = _read_scaled_bands(
            src,
            win,
            vis_bands,
            canonical_to_idx,
            in_nodata_map,
            in_gain_map,
            in_offset_map,
        )
        valid_mask = ~(
            invalid_masks["blue"] | invalid_masks["green"] | invalid_masks["red"]
        )
        if not np.any(valid_mask):
            continue
        brightness = (
            scaled_bands["blue"] + scaled_bands["green"] + scaled_bands["red"]
        ) / 3.0
        vals = brightness[valid_mask]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        if shift_val is None:
            shift_val = float(vals[0])
        diff = vals - shift_val
        total_count += int(vals.size)
        sum_shifted += float(np.sum(diff))
        sum_sq_shifted += float(np.sum(diff * diff))

    if total_count == 0 or shift_val is None:
        return 0.0, 0.0

    mean_shifted = sum_shifted / total_count
    mean_val = shift_val + mean_shifted
    variance = max(0.0, (sum_sq_shifted / total_count) - (mean_shifted ** 2))
    stddev_val = math.sqrt(variance)
    half_width = stretch_std_devs * stddev_val
    return mean_val - half_width, mean_val + half_width


def _format_output_band(
    index_arr: np.ndarray,
    valid_mask: np.ndarray,
    out_g: float,
    out_o: float,
    out_nd: float,
    out_np_dtype: np.dtype,
) -> np.ndarray:
    """Apply output gain/offset scaling, integerisation, and no-data masking."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    scaled = (index_arr * out_g) + out_o
    good = valid_mask & np.isfinite(scaled)

    if np.issubdtype(out_np_dtype, np.integer):
        info = np.iinfo(out_np_dtype)
        rounded = np.rint(np.where(good, scaled, out_nd))
        clamped = np.clip(rounded, info.min, info.max)
        result = np.where(good, clamped, out_nd)
        return result.astype(out_np_dtype)

    result = np.where(good, scaled, out_nd)
    return result.astype(out_np_dtype)


def calc_indices(
    input_img: str,
    band_names: Mapping[int, str],
    indices: Sequence[str],
    output_img: str,
    gdalformat: str = "GTIFF",
    datatype: int = rsgispy.TYPE_32FLOAT,
    in_no_data_val: Optional[float] = None,
    out_no_data_val: Optional[float] = None,
    in_gain: Optional[
        Union[float, Sequence[float], Mapping[Union[int, str], float]]
    ] = 1.0,
    in_offset: Optional[
        Union[float, Sequence[float], Mapping[Union[int, str], float]]
    ] = 0.0,
    out_gain: Optional[
        Union[float, Sequence[float], Mapping[str, float]]
    ] = 1.0,
    out_offset: Optional[
        Union[float, Sequence[float], Mapping[str, float]]
    ] = 0.0,
    calc_stats: bool = True,
    use_pbar: bool = True,
    creation_opts: Optional[Mapping[str, Any]] = None,
    index_params: Optional[Mapping[str, Mapping[str, float]]] = None,
) -> None:
    """
    Calculate multiple spectral indices in a single pass and write them to a
    multi-band output image with each band named after its index.

    Input pixel values are converted to physical reflectance/units via
    ``physical = (dn - in_offset) / in_gain`` prior to index calculation, and
    computed index values are converted to output pixel values via
    ``output = (index_val * out_gain) + out_offset``. When an integer
    ``datatype`` is used (e.g., ``rsgispy.TYPE_16INT``), valid values are
    rounded to the nearest integer and clamped to the target data type range.

    Water-index disambiguation:

    * ``NDWI_McFeeters`` (also ``NDWI`` and ``GNDWI``):
      ``(green - nir) / (green + nir)`` (McFeeters 1996).
    * ``NDWI_Gao`` (also ``NDMI``):
      ``(nir - swir1) / (nir + swir1)`` (Gao 1996; note ``rsgislib.calc_ndwi``
      used this formula).
    * ``MNDWI`` (also ``GMNDWI`` and ``NDSI``):
      ``(green - swir1) / (green + swir1)`` (Xu 2006 / Salomonson & Appel 2004).
    * ``WBI_BlueNIR`` (also ``WBI``):
      ``blue / nir`` (multispectral RSGISLib ratio, distinct from narrowband
      ``R900 / R970``).
    * ``SR`` / ``RVI`` / ``RVI_NIR_Red`` compute ``nir / red``, whereas
      ``RVI_Red_NIR`` computes ``red / nir``.

    Canonical band names accepted in ``band_names``: ``'blue'``, ``'green'``,
    ``'red'``, ``'red_edge'`` (alias ``'re'``), ``'nir'``, ``'swir1'``
    (alias ``'swir'``), and ``'swir2'``.

    Full equations for all supported indices are documented in
    :doc:`imagecalc/calcindices <../imagecalc/calcindices>` and inspectable via
    :func:`get_index_equation`.

    :param input_img: Path to the input multi-band raster image file.
    :param band_names: Mapping from 1-based band index to band name
                       (e.g., ``{1: "blue", 2: "green", 3: "red", 4: "nir", 5: "swir1"}``).
    :param indices: Sequence of spectral index names to calculate and stack
                    into ``output_img`` in the specified order.
    :param output_img: Path to the output multi-band raster image file.
    :param gdalformat: GDAL/rasterio driver format for the output file
                       (Default = ``'GTIFF'``).
    :param datatype: Output data type constant from ``rsgispy.TYPE_*``
                     (Default = ``rsgispy.TYPE_32FLOAT``).
    :param in_no_data_val: Optional input no-data value. If ``None``, reads the
                           no-data value(s) from the input image header.
    :param out_no_data_val: Optional output no-data value. If ``None``, uses
                            ``-999.0`` for signed/floating-point types (or
                            ``0`` / type minimum for narrow integer types).
    :param in_gain: Scalar, per-band sequence (length matching input band count),
                    or mapping (keyed by band index or band name) used to unscale
                    input values via ``(dn - in_offset) / in_gain``
                    (Default = ``1.0``).
    :param in_offset: Scalar, per-band sequence, or mapping subtracted from
                      input values prior to dividing by ``in_gain``
                      (Default = ``0.0``).
    :param out_gain: Optional scalar, per-index sequence, or mapping applied to
                     output index values via ``(index_val * out_gain) + out_offset``
                     to support integerisation (Default = ``1.0``).
    :param out_offset: Optional scalar, per-index sequence, or mapping added
                       after multiplying by ``out_gain`` (Default = ``0.0``).
    :param calc_stats: If ``True``, populate band statistics, histograms, and
                       overview pyramids on ``output_img`` (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).
    :param creation_opts: Optional mapping of GDAL creation options.
    :param index_params: Optional mapping of per-index parameter overrides
                         (e.g., ``{"SAVI": {"l": 0.5}, "WDVI": {"s": 1.2}}``).

    .. code:: python

        import rsgispy
        import rsgispy.imagecalc.calcindices

        band_names = {1: "blue", 2: "green", 3: "red", 4: "nir", 5: "swir1"}
        rsgispy.imagecalc.calcindices.calc_indices(
            input_img="s2_refl.tif",
            band_names=band_names,
            indices=["NDVI", "NDWI_McFeeters", "MNDWI", "EVI"],
            output_img="indices_int16.tif",
            gdalformat="GTIFF",
            datatype=rsgispy.TYPE_16INT,
            in_gain=10000.0,
            in_offset=0.0,
            out_gain=1000.0,
            out_offset=0.0,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-branches
    ensure_file_exists(input_img)
    out_rio_dtype = rsgispy.get_rasterio_datatype(datatype)
    out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(datatype))
    out_nd = _resolve_out_nodata(out_no_data_val, out_np_dtype)
    parsed_idx_params = _resolve_index_params(index_params)

    with ExitStack() as stack:
        try:
            src = stack.enter_context(rasterio.open(input_img, "r"))
        except Exception as err:
            raise rsgispy.RSGISPyException(
                f"Could not open input image '{input_img}': {err}"
            ) from err

        canonical_to_idx = _normalise_band_names(band_names, src.count, input_img)
        resolved_indices = _resolve_requested_indices(indices, canonical_to_idx)

        req_band_set: List[str] = []
        seen_bands = set()
        for _, spec in resolved_indices:
            for b_name in spec.req_bands:
                if b_name not in seen_bands:
                    seen_bands.add(b_name)
                    req_band_set.append(b_name)

        in_gain_map = _resolve_in_scaling(
            in_gain,
            1.0,
            req_band_set,
            canonical_to_idx,
            src.count,
            "in_gain",
            check_nonzero=True,
        )
        in_offset_map = _resolve_in_scaling(
            in_offset,
            0.0,
            req_band_set,
            canonical_to_idx,
            src.count,
            "in_offset",
            check_nonzero=False,
        )
        out_gain_list = _resolve_out_scaling(
            out_gain, 1.0, resolved_indices, "out_gain", check_nonzero=True
        )
        out_offset_list = _resolve_out_scaling(
            out_offset, 0.0, resolved_indices, "out_offset", check_nonzero=False
        )

        in_nodata_map: Dict[str, Optional[float]] = {}
        for b_name in req_band_set:
            if in_no_data_val is not None:
                in_nodata_map[b_name] = float(in_no_data_val)
            else:
                b_idx = canonical_to_idx[b_name]
                in_nodata_map[b_name] = src.nodatavals[b_idx - 1]

        needs_b_stats = any(
            spec.requires_brightness_stats for _, spec in resolved_indices
        )
        if needs_b_stats:
            b_params = parsed_idx_params.get("brightness_scaled", {})
            stretch_k = float(b_params.get("stretch_std_devs", 2.0))
            b_min, b_max = _calc_brightness_stretch_bounds(
                src=src,
                canonical_to_idx=canonical_to_idx,
                in_nodata_map=in_nodata_map,
                in_gain_map=in_gain_map,
                in_offset_map=in_offset_map,
                stretch_std_devs=stretch_k,
                use_pbar=use_pbar,
            )
            parsed_idx_params["brightness_scaled"] = {
                **b_params,
                "in_min": b_min,
                "in_max": b_max,
            }

        grid = compute_spatial_intersection({input_img: src})
        dst = create_output_from_dataset(
            stack=stack,
            src=src,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype=out_rio_dtype,
            count=len(resolved_indices),
            creation_opts=creation_opts,
        )
        dst.nodata = out_nd
        for band_pos, (out_name, _) in enumerate(resolved_indices, start=1):
            dst.set_band_description(band_pos, out_name)

        for out_win in iter_dataset_windows(
            grid.height, grid.width, use_pbar=use_pbar
        ):
            scaled_bands, invalid_masks = _read_scaled_bands(
                src,
                out_win,
                req_band_set,
                canonical_to_idx,
                in_nodata_map,
                in_gain_map,
                in_offset_map,
            )

            out_bands: List[np.ndarray] = []
            for idx_pos, (out_name, spec) in enumerate(resolved_indices):
                req_invalid = np.zeros(
                    (int(out_win.height), int(out_win.width)), dtype=bool
                )
                for b_name in spec.req_bands:
                    req_invalid = req_invalid | invalid_masks[b_name]

                per_idx_params = parsed_idx_params.get(
                    out_name.lower(),
                    parsed_idx_params.get(spec.canonical_name.lower(), {}),
                )
                with np.errstate(divide="ignore", invalid="ignore"):
                    raw_idx, expr_valid = spec.calc_fn(scaled_bands, per_idx_params)

                full_valid = (~req_invalid) & expr_valid & np.isfinite(raw_idx)
                out_bands.append(
                    _format_output_band(
                        index_arr=raw_idx,
                        valid_mask=full_valid,
                        out_g=out_gain_list[idx_pos],
                        out_o=out_offset_list[idx_pos],
                        out_nd=out_nd,
                        out_np_dtype=out_np_dtype,
                    )
                )

            dst.write(np.stack(out_bands, axis=0), window=out_win)

    if calc_stats:
        pop_img_stats(
            input_img=output_img,
            no_data_val=out_nd,
            calc_pyramids=True,
            ignore_cog=gdalformat.upper().strip() == "COG",
            use_pbar=use_pbar,
        )
