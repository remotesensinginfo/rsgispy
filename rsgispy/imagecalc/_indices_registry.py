"""
Optical spectral index registry, band aliases, and metadata query helpers for
:mod:`rsgispy.imagecalc.calcindices`.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Tuple

import numpy as np

import rsgispy
from rsgispy.imagecalc import _optical_indices_ext as _ext_opt
from rsgispy.imagecalc._indices_kernels import (
    _calc_arvi_block,
    _calc_awei_nsh_block,
    _calc_awei_sh_block,
    _calc_bai_block,
    _calc_brightness_block,
    _calc_brightness_scaled_block,
    _calc_bsi_block,
    _calc_cigreen_block,
    _calc_cire_block,
    _calc_cmr_block,
    _calc_ctvi_block,
    _calc_dvi_block,
    _calc_evi2_block,
    _calc_evi_block,
    _calc_exg_block,
    _calc_fmr_block,
    _calc_gemi_block,
    _calc_gli_block,
    _calc_gndvi_block,
    _calc_ior_block,
    _calc_kndvi_block,
    _calc_mndwi_block,
    _calc_msavi2_block,
    _calc_mvi_block,
    _calc_nbr2_block,
    _calc_nbr_block,
    _calc_ndbi_block,
    _calc_ndbsi_block,
    _calc_ndgi_block,
    _calc_ndre_block,
    _calc_ndvi_block,
    _calc_ndwi_gao_block,
    _calc_ndwi_mcfeeters_block,
    _calc_nirv_block,
    _calc_npcri_block,
    _calc_nrvi_block,
    _calc_osavi_block,
    _calc_remi_block,
    _calc_rvi_red_nir_block,
    _calc_satvi_block,
    _calc_savi_block,
    _calc_si_block,
    _calc_slavi_block,
    _calc_sr_block,
    _calc_ttvi_block,
    _calc_tvi_block,
    _calc_vari_block,
    _calc_wbi_bluenir_block,
    _calc_wdvi_block,
    _calc_whiteness_block,
    _calc_wi2015_block,
)


_BAND_ALIASES: Dict[str, str] = {
    "blue": "blue",
    "green": "green",
    "red": "red",
    "nir": "nir",
    "nir_narrow": "nir_narrow",
    "nir-narrow": "nir_narrow",
    "narrow_nir": "nir_narrow",
    "narrow-nir": "nir_narrow",
    "nir8a": "nir_narrow",
    "re4": "nir_narrow",
    "re_4": "nir_narrow",
    "re-4": "nir_narrow",
    "red_edge_4": "nir_narrow",
    "red-edge-4": "nir_narrow",
    "rededge4": "nir_narrow",
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
    "re1": "re1",
    "re_1": "re1",
    "re-1": "re1",
    "red_edge_1": "re1",
    "red-edge-1": "re1",
    "rededge1": "re1",
    "red edge 1": "re1",
    "re2": "re2",
    "re_2": "re2",
    "re-2": "re2",
    "red_edge_2": "re2",
    "red-edge-2": "re2",
    "rededge2": "re2",
    "red edge 2": "re2",
    "re3": "re3",
    "re_3": "re3",
    "re-3": "re3",
    "red_edge_3": "re3",
    "red-edge-3": "re3",
    "rededge3": "re3",
    "red edge 3": "re3",
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
    "pvi": _IndexSpec(
        "PVI",
        ("red", "nir"),
        "(nir - a * red - b) / sqrt(1 + a^2)",
        _ext_opt.calc_pvi_block,
    ),
    "wdrvi": _IndexSpec(
        "WDRVI",
        ("red", "nir"),
        "(a * nir - red) / (a * nir + red)",
        _ext_opt.calc_wdrvi_block,
    ),
    "cigreen": _IndexSpec(
        "CIgreen", ("green", "nir"), "(nir / green) - 1", _calc_cigreen_block
    ),
    "gcvi": _IndexSpec(
        "GCVI", ("green", "nir"), "(nir / green) - 1", _calc_cigreen_block
    ),
    "npcri": _IndexSpec(
        "NPCRI",
        ("red", "blue"),
        "(red - blue) / (red + blue)",
        _calc_npcri_block,
    ),
    # Red-Edge Vegetation & Stress
    "ndre": _IndexSpec(
        "NDRE",
        ("red_edge", "nir"),
        "(nir - red_edge) / (nir + red_edge)",
        _calc_ndre_block,
    ),
    "ndre_re": _IndexSpec(
        "NDRE_RE",
        ("re1", "re2"),
        "(re2 - re1) / (re2 + re1)",
        _ext_opt.calc_ndre_re_block,
    ),
    "cire": _IndexSpec(
        "CIre", ("red_edge", "nir"), "(nir / red_edge) - 1", _calc_cire_block
    ),
    "ci_red_edge": _IndexSpec(
        "CI_red_edge",
        ("red_edge", "nir"),
        "(nir / red_edge) - 1",
        _calc_cire_block,
    ),
    "mtci": _IndexSpec(
        "MTCI",
        ("red", "re1", "re2"),
        "(re2 - re1) / (re1 - red)",
        _ext_opt.calc_mtci_block,
    ),
    "mcari": _IndexSpec(
        "MCARI",
        ("green", "red", "re1"),
        "((re1 - red) - 0.2 * (re1 - green)) * (re1 / red)",
        _ext_opt.calc_mcari_block,
    ),
    "ireci": _IndexSpec(
        "IRECI",
        ("red", "re1", "re2", "re3"),
        "(re3 - red) / (re1 / re2)",
        _ext_opt.calc_ireci_block,
    ),
    "s2rep": _IndexSpec(
        "S2REP",
        ("red", "re1", "re2", "re3"),
        "705 + 35 * (((re3 + red) / 2) - re1) / (re2 - re1)",
        _ext_opt.calc_s2rep_block,
    ),
    "psri": _IndexSpec(
        "PSRI",
        ("blue", "red", "re2"),
        "(red - blue) / re2",
        _ext_opt.calc_psri_block,
    ),
    "psri_green": _IndexSpec(
        "PSRI_Green",
        ("green", "red", "red_edge"),
        "(red - green) / red_edge",
        _ext_opt.calc_psri_green_block,
    ),
    # Visible-Band / UAV Vegetation
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
    "ngrdi": _IndexSpec(
        "NGRDI",
        ("green", "red"),
        "(green - red) / (green + red)",
        _calc_ndgi_block,
    ),
    # Mangrove & Wetland Forest
    "cmri": _IndexSpec(
        "CMRI",
        ("green", "red", "nir"),
        "((nir - red) / (nir + red)) - ((green - nir) / (green + nir))",
        _ext_opt.calc_cmri_block,
    ),
    "emi": _IndexSpec(
        "EMI",
        ("green", "nir", "swir1"),
        "(nir - swir1) / (green + nir)",
        _ext_opt.calc_emi_block,
    ),
    "ndmai": _IndexSpec(
        "NDMaI",
        ("green", "swir2"),
        "(swir2 - green) / (swir2 + green)",
        _ext_opt.calc_ndmai_block,
    ),
    "ndwvi": _IndexSpec(
        "NDWVI",
        ("green", "swir2"),
        "(swir2 - green) / (swir2 + green)",
        _ext_opt.calc_ndmai_block,
    ),
    "mvi": _IndexSpec(
        "MVI",
        ("green", "nir", "swir1"),
        "(nir - green) / (swir1 - green)",
        _calc_mvi_block,
    ),
    "mvi_abs": _IndexSpec(
        "MVI_Abs",
        ("green", "nir", "swir1"),
        "|nir - green| / |swir1 - green|",
        _ext_opt.calc_mvi_abs_block,
    ),
    "mi": _IndexSpec(
        "MI",
        ("nir", "swir1"),
        "((nir - swir1) / (nir * swir1)) * 10000",
        _ext_opt.calc_mi_block,
    ),
    "ammi": _IndexSpec(
        "AMMI",
        ("red", "nir", "swir1"),
        "((nir - red) / (red + swir1)) * ((nir - swir1) / (swir1 - 0.65 * red))",
        _ext_opt.calc_ammi_block,
    ),
    "imfi": _IndexSpec(
        "IMFI",
        ("blue", "green", "nir"),
        "(blue + green - 2 * nir) / (blue + green + 2 * nir)",
        _ext_opt.calc_imfi_block,
    ),
    "mdi1": _IndexSpec(
        "MDI1",
        ("nir", "swir1"),
        "(nir - swir1) / swir1",
        _ext_opt.calc_mdi1_block,
    ),
    "mdi2": _IndexSpec(
        "MDI2",
        ("nir", "swir2"),
        "(nir - swir2) / swir2",
        _ext_opt.calc_mdi2_block,
    ),
    "wfi": _IndexSpec(
        "WFI",
        ("red", "nir", "swir2"),
        "(nir - red) / swir2",
        _ext_opt.calc_wfi_block,
    ),
    "fdi": _IndexSpec(
        "FDI",
        ("green", "red", "nir"),
        "nir - (red + green)",
        _ext_opt.calc_fdi_block,
    ),
    "dnvi": _IndexSpec(
        "DNVI",
        ("swir1", "swir2"),
        "(swir1 - swir2)^2 / (swir1 + swir2)",
        _ext_opt.calc_dnvi_block,
    ),
    "mmri": _IndexSpec(
        "MMRI",
        ("green", "red", "nir", "swir1"),
        "(|MNDWI| - |NDVI|) / (|MNDWI| + |NDVI|)",
        _ext_opt.calc_mmri_block,
    ),
    "mfi": _IndexSpec(
        "MFI",
        ("red", "re1", "re2", "re3", "nir_narrow", "swir2"),
        "0.25 * sum_i(rho_i - (swir2 + (red - swir2) * (2190 - lambda_i) / 1525))",
        _ext_opt.calc_mfi_block,
    ),
    "nimi": _IndexSpec(
        "NIMI",
        ("red", "re2", "re3", "nir"),
        "(3 * red - (re2 + re3 + nir)) / (3 * red + re2 + re3 + nir)",
        _ext_opt.calc_nimi_block,
    ),
    "emvi": _IndexSpec(
        "EMVI",
        ("green", "swir1", "swir2"),
        "(green - swir2) / (swir1 - green)",
        _ext_opt.calc_emvi_block,
    ),
    "gcmi": _IndexSpec(
        "GCMI",
        ("blue", "green", "red", "nir", "swir1"),
        "(A * EVI + B * GNDVI) / (C * LSWI + D * RNDWI)",
        _ext_opt.calc_gcmi_block,
    ),
    "remi": _IndexSpec(
        "REMI",
        ("green", "red", "red_edge", "swir1"),
        "(red_edge - red) / (swir1 - green)",
        _calc_remi_block,
    ),
    "imi": _IndexSpec(
        "IMI",
        ("green", "red", "nir", "swir1"),
        "(nir - red) / (swir1 + green)",
        _ext_opt.calc_imi_block,
    ),
    "remch": _IndexSpec(
        "REMCH",
        ("re1", "nir", "swir1", "swir2"),
        "(nir / swir1) - ((nir / swir2) / (nir / re1)) + (nir / re1)",
        _ext_opt.calc_remch_block,
    ),
    # Water, Intertidal, Snow & Moisture
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
    "ndii": _IndexSpec(
        "NDII",
        ("nir", "swir1"),
        "(nir - swir1) / (nir + swir1)",
        _calc_ndwi_gao_block,
    ),
    "lswi": _IndexSpec(
        "LSWI",
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
    "wri": _IndexSpec(
        "WRI",
        ("green", "red", "nir", "swir1"),
        "(green + red) / (nir + swir1)",
        _ext_opt.calc_wri_block,
    ),
    "rndwi": _IndexSpec(
        "RNDWI",
        ("red", "swir1"),
        "(red - swir1) / (red + swir1)",
        _ext_opt.calc_rndwi_block,
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
