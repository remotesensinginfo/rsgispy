"""
Extended optical spectral index block functions (mangrove, wetland/forest,
red-edge, vegetation, and water indices) for
:mod:`rsgispy.imagecalc.calcindices`.
"""

import math
from typing import Dict, Mapping, Tuple

import numpy as np

import rsgispy


def calc_cmri_block(
    bands: Dict[str, np.ndarray], _params: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Combined Mangrove Recognition Index (Gupta et al. 2018)."""
    green = bands["green"]
    red = bands["red"]
    nir = bands["nir"]
    ndvi_denom = nir + red
    ndwi_denom = green + nir
    valid = (ndvi_denom != 0.0) & (ndwi_denom != 0.0)
    ndvi = np.zeros_like(nir, dtype=np.float64)
    ndwi = np.zeros_like(nir, dtype=np.float64)
    np.divide(nir - red, ndvi_denom, out=ndvi, where=valid)
    np.divide(green - nir, ndwi_denom, out=ndwi, where=valid)
    out = np.where(valid, ndvi - ndwi, 0.0)
    return out, valid


def calc_emi_block(
    bands: Dict[str, np.ndarray], _params: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Enhanced Mangrove Index (Prayudha et al. 2024)."""
    green = bands["green"]
    nir = bands["nir"]
    swir1 = bands["swir1"]
    denom = green + nir
    valid = denom != 0.0
    out = np.zeros_like(denom, dtype=np.float64)
    np.divide(nir - swir1, denom, out=out, where=valid)
    return out, valid


def calc_ndmai_block(
    bands: Dict[str, np.ndarray], _params: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Normalised Difference Mangrove Index (Shi et al. 2016)."""
    green = bands["green"]
    swir2 = bands["swir2"]
    denom = swir2 + green
    valid = denom != 0.0
    out = np.zeros_like(denom, dtype=np.float64)
    np.divide(swir2 - green, denom, out=out, where=valid)
    return out, valid


def calc_mvi_abs_block(
    bands: Dict[str, np.ndarray], _params: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Mangrove Vegetation Index absolute-value form (Baloloy et al. 2020)."""
    green = bands["green"]
    nir = bands["nir"]
    swir1 = bands["swir1"]
    denom = np.abs(swir1 - green)
    valid = denom != 0.0
    out = np.zeros_like(denom, dtype=np.float64)
    np.divide(np.abs(nir - green), denom, out=out, where=valid)
    return out, valid


def calc_mi_block(
    bands: Dict[str, np.ndarray], _params: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Mangrove Index (Winarso & Purwanto 2014)."""
    nir = bands["nir"]
    swir1 = bands["swir1"]
    denom = nir * swir1
    valid = denom != 0.0
    out = np.zeros_like(denom, dtype=np.float64)
    np.divide((nir - swir1) * 10000.0, denom, out=out, where=valid)
    return out, valid


def calc_ammi_block(
    bands: Dict[str, np.ndarray], _params: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Automatic Mangrove Map and Index (Suyarso 2022)."""
    red = bands["red"]
    nir = bands["nir"]
    swir1 = bands["swir1"]
    denom1 = red + swir1
    denom2 = swir1 - (0.65 * red)
    valid = (denom1 != 0.0) & (denom2 != 0.0)
    term1 = np.zeros_like(denom1, dtype=np.float64)
    term2 = np.zeros_like(denom2, dtype=np.float64)
    np.divide(nir - red, denom1, out=term1, where=valid)
    np.divide(nir - swir1, denom2, out=term2, where=valid)
    out = np.where(valid, term1 * term2, 0.0)
    return out, valid


def calc_imfi_block(
    bands: Dict[str, np.ndarray], _params: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Inundated Mangrove Forest Index (Jia 2014; Jia et al. 2018)."""
    blue = bands["blue"]
    green = bands["green"]
    nir = bands["nir"]
    two_nir = 2.0 * nir
    vis_sum = blue + green
    denom = vis_sum + two_nir
    valid = denom != 0.0
    out = np.zeros_like(denom, dtype=np.float64)
    np.divide(vis_sum - two_nir, denom, out=out, where=valid)
    return out, valid


def calc_mdi1_block(
    bands: Dict[str, np.ndarray], _params: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Mangrove Discrimination Index 1 (Wang et al. 2018)."""
    nir = bands["nir"]
    swir1 = bands["swir1"]
    valid = swir1 != 0.0
    out = np.zeros_like(swir1, dtype=np.float64)
    np.divide(nir - swir1, swir1, out=out, where=valid)
    return out, valid


def calc_mdi2_block(
    bands: Dict[str, np.ndarray], _params: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Mangrove Discrimination Index 2 (Wang et al. 2018)."""
    nir = bands["nir"]
    swir2 = bands["swir2"]
    valid = swir2 != 0.0
    out = np.zeros_like(swir2, dtype=np.float64)
    np.divide(nir - swir2, swir2, out=out, where=valid)
    return out, valid


def calc_wfi_block(
    bands: Dict[str, np.ndarray], _params: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Wetland Forest Index (Wang et al. 2018)."""
    red = bands["red"]
    nir = bands["nir"]
    swir2 = bands["swir2"]
    valid = swir2 != 0.0
    out = np.zeros_like(swir2, dtype=np.float64)
    np.divide(nir - red, swir2, out=out, where=valid)
    return out, valid


def calc_fdi_block(
    bands: Dict[str, np.ndarray], _params: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Forest Discrimination Index (Bunting & Lucas 2006; Kamal et al. 2015)."""
    out = bands["nir"] - (bands["red"] + bands["green"])
    return out, np.ones(out.shape, dtype=bool)


def calc_dnvi_block(
    bands: Dict[str, np.ndarray], _params: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Discriminant Normalised Vegetation Index (Manna & Raychaudhuri 2018)."""
    swir1 = bands["swir1"]
    swir2 = bands["swir2"]
    denom = swir1 + swir2
    valid = denom != 0.0
    out = np.zeros_like(denom, dtype=np.float64)
    np.divide((swir1 - swir2) ** 2, denom, out=out, where=valid)
    return out, valid


def calc_mmri_block(
    bands: Dict[str, np.ndarray], _params: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Modular Mangrove Recognition Index (Diniz et al. 2019)."""
    green = bands["green"]
    red = bands["red"]
    nir = bands["nir"]
    swir1 = bands["swir1"]
    mndwi_denom = green + swir1
    ndvi_denom = nir + red
    base_valid = (mndwi_denom != 0.0) & (ndvi_denom != 0.0)
    mndwi = np.zeros_like(nir, dtype=np.float64)
    ndvi = np.zeros_like(nir, dtype=np.float64)
    np.divide(green - swir1, mndwi_denom, out=mndwi, where=base_valid)
    np.divide(nir - red, ndvi_denom, out=ndvi, where=base_valid)
    abs_mndwi = np.abs(mndwi)
    abs_ndvi = np.abs(ndvi)
    denom = abs_mndwi + abs_ndvi
    valid = base_valid & (denom != 0.0)
    out = np.zeros_like(nir, dtype=np.float64)
    np.divide(abs_mndwi - abs_ndvi, denom, out=out, where=valid)
    return out, valid


def calc_mfi_block(
    bands: Dict[str, np.ndarray], params: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Mangrove Forest Index (Jia et al. 2019)."""
    wl_red = float(params.get("lambda_red", 665.0))
    wl_swir2 = float(params.get("lambda_swir2", 2190.0))
    span = wl_swir2 - wl_red
    if span == 0.0:
        raise rsgispy.RSGISPyException(
            "MFI requires lambda_swir2 != lambda_red."
        )
    wls = (
        ("re1", float(params.get("lambda_re1", 705.0))),
        ("re2", float(params.get("lambda_re2", 740.0))),
        ("re3", float(params.get("lambda_re3", 783.0))),
        ("nir_narrow", float(params.get("lambda_nir_narrow", 865.0))),
    )
    red = bands["red"]
    swir2 = bands["swir2"]
    diff_base = red - swir2
    acc = np.zeros_like(red, dtype=np.float64)
    for band_key, wl_i in wls:
        baseline_i = swir2 + diff_base * ((wl_swir2 - wl_i) / span)
        acc = acc + (bands[band_key] - baseline_i)
    out = 0.25 * acc
    return out, np.ones(out.shape, dtype=bool)


def calc_nimi_block(
    bands: Dict[str, np.ndarray], _params: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Normalised Intertidal Mangrove Index (Xu et al. 2020)."""
    three_red = 3.0 * bands["red"]
    re_nir_sum = bands["re2"] + bands["re3"] + bands["nir"]
    denom = three_red + re_nir_sum
    valid = denom != 0.0
    out = np.zeros_like(denom, dtype=np.float64)
    np.divide(three_red - re_nir_sum, denom, out=out, where=valid)
    return out, valid


def calc_emvi_block(
    bands: Dict[str, np.ndarray], _params: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Enhanced Mangrove Vegetation Index (Yang et al. 2022)."""
    green = bands["green"]
    swir1 = bands["swir1"]
    swir2 = bands["swir2"]
    denom = swir1 - green
    valid = denom != 0.0
    out = np.zeros_like(denom, dtype=np.float64)
    np.divide(green - swir2, denom, out=out, where=valid)
    return out, valid


def calc_gcmi_block(
    bands: Dict[str, np.ndarray], params: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Generalised Composite Mangrove Index (Xue & Qian 2022)."""
    # pylint: disable=too-many-locals
    blue = bands["blue"]
    green = bands["green"]
    red = bands["red"]
    nir = bands["nir"]
    swir1 = bands["swir1"]

    coef_a = float(params.get("a", 3.4))
    coef_b = float(params.get("b", 2.9))
    coef_c = float(params.get("c", 4.3))
    coef_d = float(params.get("d", 1.1))

    evi_denom = nir + (6.0 * red) - (7.5 * blue) + 1.0
    gndvi_denom = nir + green
    lswi_denom = nir + swir1
    rndwi_denom = red + swir1
    base_valid = (
        (evi_denom != 0.0)
        & (gndvi_denom != 0.0)
        & (lswi_denom != 0.0)
        & (rndwi_denom != 0.0)
    )

    evi = np.zeros_like(nir, dtype=np.float64)
    gndvi = np.zeros_like(nir, dtype=np.float64)
    lswi = np.zeros_like(nir, dtype=np.float64)
    rndwi = np.zeros_like(nir, dtype=np.float64)
    np.divide(2.5 * (nir - red), evi_denom, out=evi, where=base_valid)
    np.divide(nir - green, gndvi_denom, out=gndvi, where=base_valid)
    np.divide(nir - swir1, lswi_denom, out=lswi, where=base_valid)
    np.divide(red - swir1, rndwi_denom, out=rndwi, where=base_valid)

    numer = (coef_a * evi) + (coef_b * gndvi)
    denom = (coef_c * lswi) + (coef_d * rndwi)
    valid = base_valid & (denom != 0.0)
    out = np.zeros_like(nir, dtype=np.float64)
    np.divide(numer, denom, out=out, where=valid)
    return out, valid


def calc_imi_block(
    bands: Dict[str, np.ndarray], _params: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Integrated Mangrove Index (Umar & Abbas 2025)."""
    green = bands["green"]
    red = bands["red"]
    nir = bands["nir"]
    swir1 = bands["swir1"]
    denom = swir1 + green
    valid = denom != 0.0
    out = np.zeros_like(denom, dtype=np.float64)
    np.divide(nir - red, denom, out=out, where=valid)
    return out, valid


def calc_remch_block(
    bands: Dict[str, np.ndarray], _params: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Red-Edge Mangrove Canopy Height Index (Jamaluddin et al. 2025)."""
    re1 = bands["re1"]
    nir = bands["nir"]
    swir1 = bands["swir1"]
    swir2 = bands["swir2"]
    valid = (swir1 != 0.0) & (swir2 != 0.0) & (re1 != 0.0) & (nir != 0.0)
    t1 = np.zeros_like(nir, dtype=np.float64)
    t2_num = np.zeros_like(nir, dtype=np.float64)
    t3 = np.zeros_like(nir, dtype=np.float64)
    t2 = np.zeros_like(nir, dtype=np.float64)
    np.divide(nir, swir1, out=t1, where=valid)
    np.divide(nir, swir2, out=t2_num, where=valid)
    np.divide(nir, re1, out=t3, where=valid)
    np.divide(t2_num, t3, out=t2, where=valid)
    out = np.where(valid, t1 - t2 + t3, 0.0)
    return out, valid


def calc_pvi_block(
    bands: Dict[str, np.ndarray], params: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Perpendicular Vegetation Index (Richardson & Wiegand 1977)."""
    slope_a = float(params.get("a", 1.0))
    intercept_b = float(params.get("b", 0.0))
    denom = math.sqrt(1.0 + (slope_a ** 2))
    out = (bands["nir"] - (slope_a * bands["red"]) - intercept_b) / denom
    return out, np.ones(out.shape, dtype=bool)


def calc_wdrvi_block(
    bands: Dict[str, np.ndarray], params: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Wide Dynamic Range Vegetation Index (Gitelson 2004)."""
    coef_a = float(params.get("a", 0.2))
    a_nir = coef_a * bands["nir"]
    red = bands["red"]
    denom = a_nir + red
    valid = denom != 0.0
    out = np.zeros_like(denom, dtype=np.float64)
    np.divide(a_nir - red, denom, out=out, where=valid)
    return out, valid


def calc_ndre_re_block(
    bands: Dict[str, np.ndarray], _params: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Red-Edge-Only NDRE ``(re2 - re1) / (re2 + re1)``."""
    re1 = bands["re1"]
    re2 = bands["re2"]
    denom = re2 + re1
    valid = denom != 0.0
    out = np.zeros_like(denom, dtype=np.float64)
    np.divide(re2 - re1, denom, out=out, where=valid)
    return out, valid


def calc_mtci_block(
    bands: Dict[str, np.ndarray], _params: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate MERIS Terrestrial Chlorophyll Index (Dash & Curran 2004)."""
    red = bands["red"]
    re1 = bands["re1"]
    re2 = bands["re2"]
    denom = re1 - red
    valid = denom != 0.0
    out = np.zeros_like(denom, dtype=np.float64)
    np.divide(re2 - re1, denom, out=out, where=valid)
    return out, valid


def calc_mcari_block(
    bands: Dict[str, np.ndarray], _params: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Modified Chlorophyll Absorption Ratio Index (Daughtry et al. 2000)."""
    green = bands["green"]
    red = bands["red"]
    re1 = bands["re1"]
    valid = red != 0.0
    ratio = np.zeros_like(red, dtype=np.float64)
    np.divide(re1, red, out=ratio, where=valid)
    out = np.where(valid, ((re1 - red) - 0.2 * (re1 - green)) * ratio, 0.0)
    return out, valid


def calc_ireci_block(
    bands: Dict[str, np.ndarray], _params: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Inverted Red-Edge Chlorophyll Index (Frampton et al. 2013)."""
    red = bands["red"]
    re1 = bands["re1"]
    re2 = bands["re2"]
    re3 = bands["re3"]
    valid = (re1 != 0.0) & (re2 != 0.0)
    re_ratio = np.zeros_like(re1, dtype=np.float64)
    np.divide(re1, re2, out=re_ratio, where=valid)
    out = np.zeros_like(re1, dtype=np.float64)
    np.divide(re3 - red, re_ratio, out=out, where=valid)
    return out, valid


def calc_s2rep_block(
    bands: Dict[str, np.ndarray], _params: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Sentinel-2 Red-Edge Position (Frampton et al. 2013)."""
    red = bands["red"]
    re1 = bands["re1"]
    re2 = bands["re2"]
    re3 = bands["re3"]
    denom = re2 - re1
    valid = denom != 0.0
    ratio = np.zeros_like(denom, dtype=np.float64)
    np.divide((0.5 * (re3 + red)) - re1, denom, out=ratio, where=valid)
    out = np.where(valid, 705.0 + (35.0 * ratio), 0.0)
    return out, valid


def calc_psri_block(
    bands: Dict[str, np.ndarray], _params: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Plant Senescence Reflectance Index (Merzlyak et al. 1999)."""
    blue = bands["blue"]
    red = bands["red"]
    re2 = bands["re2"]
    valid = re2 != 0.0
    out = np.zeros_like(re2, dtype=np.float64)
    np.divide(red - blue, re2, out=out, where=valid)
    return out, valid


def calc_psri_green_block(
    bands: Dict[str, np.ndarray], _params: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Plant Senescence Reflectance Index green/red-edge variant."""
    green = bands["green"]
    red = bands["red"]
    red_edge = bands["red_edge"]
    valid = red_edge != 0.0
    out = np.zeros_like(red_edge, dtype=np.float64)
    np.divide(red - green, red_edge, out=out, where=valid)
    return out, valid


def calc_wri_block(
    bands: Dict[str, np.ndarray], _params: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Water Ratio Index (Shen & Li 2010)."""
    denom = bands["nir"] + bands["swir1"]
    valid = denom != 0.0
    out = np.zeros_like(denom, dtype=np.float64)
    np.divide(bands["green"] + bands["red"], denom, out=out, where=valid)
    return out, valid


def calc_rndwi_block(
    bands: Dict[str, np.ndarray], _params: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Revised Normalised Difference Water Index (Cao et al. 2008)."""
    red = bands["red"]
    swir1 = bands["swir1"]
    denom = red + swir1
    valid = denom != 0.0
    out = np.zeros_like(denom, dtype=np.float64)
    np.divide(red - swir1, denom, out=out, where=valid)
    return out, valid
