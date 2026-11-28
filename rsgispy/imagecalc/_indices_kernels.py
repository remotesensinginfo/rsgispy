"""
Vectorised block kernels for optical spectral indices used by
:mod:`rsgispy.imagecalc.calcindices`.
"""

from typing import Dict, Mapping, Tuple

import numpy as np


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
    """Calculate Red-Edge Mangrove Index (``(red_edge - red) / (swir1 - green)``)."""
    green = bands["green"]
    red = bands["red"]
    red_edge = bands["red_edge"]
    swir1 = bands["swir1"]
    denom = swir1 - green
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
