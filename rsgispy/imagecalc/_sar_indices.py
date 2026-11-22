"""
SAR polarimetric backscatter index definitions and vectorised block functions
for :mod:`rsgispy.imagecalc.calcindices`.
"""

from dataclasses import dataclass
import math
from typing import Callable, Dict, List, Mapping, Tuple

import numpy as np

import rsgispy

_SQRT2 = math.sqrt(2.0)

_SAR_BAND_ALIASES: Dict[str, str] = {
    "vv": "vv",
    "vh": "vh",
    "hh": "hh",
    "hv": "hv",
    "sigma0_vv": "vv",
    "sigma0_vh": "vh",
    "sigma0_hh": "hh",
    "sigma0_hv": "hv",
    "gamma0_vv": "vv",
    "gamma0_vh": "vh",
    "gamma0_hh": "hh",
    "gamma0_hv": "hv",
    "beta0_vv": "vv",
    "beta0_vh": "vh",
    "beta0_hh": "hh",
    "beta0_hv": "hv",
}

_SAR_UNIT_ALIASES: Dict[str, str] = {
    "db": "dB",
    "decibel": "dB",
    "decibels": "dB",
    "power": "power",
    "linear": "power",
    "lin": "power",
}


@dataclass(frozen=True)
class _SARIndexSpec:
    """Metadata and vectorised calculation function for a SAR index."""

    canonical_name: str
    req_bands: Tuple[str, ...]
    equation: str
    calc_fn: Callable[
        [Dict[str, np.ndarray], Mapping[str, float]],
        Tuple[np.ndarray, np.ndarray],
    ]
    units: str  # "power" or "dB" — domain of arrays passed to calc_fn
    requires_vv_max: bool = False


def normalise_sar_units(in_units: str) -> str:
    """Validate and normalise ``in_units`` to ``'dB'`` or ``'power'``."""
    if not isinstance(in_units, str) or not in_units.strip():
        raise rsgispy.RSGISPyException(
            "in_units must be specified as 'dB' or 'power'."
        )
    key = in_units.strip().lower()
    if key not in _SAR_UNIT_ALIASES:
        raise rsgispy.RSGISPyException(
            f"Unsupported in_units '{in_units}'. Expected 'dB' or 'power'."
        )
    return _SAR_UNIT_ALIASES[key]


def _make_dual_rvi(
    co_pol: str, cross_pol: str
) -> Callable[
    [Dict[str, np.ndarray], Mapping[str, float]], Tuple[np.ndarray, np.ndarray]
]:
    """Build dual-pol Radar Vegetation Index ``4 * cross / (co + cross)``."""

    def _calc(
        bands: Dict[str, np.ndarray], _params: Mapping[str, float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        co = bands[co_pol]
        cr = bands[cross_pol]
        denom = co + cr
        valid = (co >= 0.0) & (cr >= 0.0) & (denom > 0.0)
        out = np.zeros_like(denom, dtype=np.float64)
        np.divide(4.0 * cr, denom, out=out, where=valid)
        return out, valid

    return _calc


def _calc_rvi_quad_block(
    bands: Dict[str, np.ndarray], _params: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate quad-pol Radar Vegetation Index (Kim & van Zyl 2000, 2009)."""
    hh = bands["hh"]
    vv = bands["vv"]
    hv = bands["hv"]
    denom = hh + vv + 2.0 * hv
    valid = (hh >= 0.0) & (vv >= 0.0) & (hv >= 0.0) & (denom > 0.0)
    out = np.zeros_like(denom, dtype=np.float64)
    np.divide(8.0 * hv, denom, out=out, where=valid)
    return out, valid


def _make_dprvic(
    co_pol: str, cross_pol: str
) -> Callable[
    [Dict[str, np.ndarray], Mapping[str, float]], Tuple[np.ndarray, np.ndarray]
]:
    """Build GRD Dual-Pol RVI ``q * (q + 3) / (q + 1)^2`` (Bhogapurapu et al. 2022)."""

    def _calc(
        bands: Dict[str, np.ndarray], _params: Mapping[str, float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        co = bands[co_pol]
        cr = bands[cross_pol]
        valid = (co > 0.0) & (cr >= 0.0)
        q = np.zeros_like(co, dtype=np.float64)
        np.divide(cr, co, out=q, where=valid)
        denom = (q + 1.0) ** 2
        valid = valid & (denom > 0.0)
        out = np.zeros_like(co, dtype=np.float64)
        np.divide(q * (q + 3.0), denom, out=out, where=valid)
        return out, valid

    return _calc


def _calc_dpsvi_block(
    bands: Dict[str, np.ndarray], params: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Dual-Polarisation SAR Vegetation Index (Periasamy 2018)."""
    vv = bands["vv"]
    vh = bands["vh"]
    vv_max = float(params["vv_max"])
    valid = (vv > 0.0) & (vh >= 0.0) & np.isfinite(vv_max)
    vddpi = np.zeros_like(vv, dtype=np.float64)
    np.divide(vv + vh, vv, out=vddpi, where=valid)
    out = (((vv_max - vv) + vh) / _SQRT2) * vddpi * vh
    return out, valid


def _calc_dpsvim_block(
    bands: Dict[str, np.ndarray], _params: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Modified Dual-Pol SAR Vegetation Index (Dos Santos et al. 2021)."""
    vv = bands["vv"]
    vh = bands["vh"]
    valid = (vv >= 0.0) & (vh >= 0.0)
    out = (vv * vv + vv * vh) / _SQRT2
    return out, valid


def _make_ddpi(
    co_pol: str, cross_pol: str
) -> Callable[
    [Dict[str, np.ndarray], Mapping[str, float]], Tuple[np.ndarray, np.ndarray]
]:
    """Build Dual Depolarisation Index ``(co + cross) / co`` (Periasamy 2018)."""

    def _calc(
        bands: Dict[str, np.ndarray], _params: Mapping[str, float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        co = bands[co_pol]
        cr = bands[cross_pol]
        valid = (co > 0.0) & (cr >= 0.0)
        out = np.zeros_like(co, dtype=np.float64)
        np.divide(co + cr, co, out=out, where=valid)
        return out, valid

    return _calc


def _make_norm_diff_power(
    num_a: str, num_b: str
) -> Callable[
    [Dict[str, np.ndarray], Mapping[str, float]], Tuple[np.ndarray, np.ndarray]
]:
    """Build linear-power normalised difference ``(a - b) / (a + b)``."""

    def _calc(
        bands: Dict[str, np.ndarray], _params: Mapping[str, float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        a = bands[num_a]
        b = bands[num_b]
        denom = a + b
        valid = (a >= 0.0) & (b >= 0.0) & (denom > 0.0)
        out = np.zeros_like(denom, dtype=np.float64)
        np.divide(a - b, denom, out=out, where=valid)
        return out, valid

    return _calc


def _calc_bmi_block(
    bands: Dict[str, np.ndarray], _params: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Biomass Index ``(hh + vv) / 2`` (Pope et al. 1994)."""
    hh = bands["hh"]
    vv = bands["vv"]
    valid = (hh >= 0.0) & (vv >= 0.0)
    out = 0.5 * (hh + vv)
    return out, valid


def _calc_csi_block(
    bands: Dict[str, np.ndarray], _params: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Canopy Structure Index ``vv / (vv + hh)`` (Pope et al. 1994)."""
    hh = bands["hh"]
    vv = bands["vv"]
    denom = vv + hh
    valid = (hh >= 0.0) & (vv >= 0.0) & (denom > 0.0)
    out = np.zeros_like(denom, dtype=np.float64)
    np.divide(vv, denom, out=out, where=valid)
    return out, valid


def _calc_vsi_block(
    bands: Dict[str, np.ndarray], _params: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Volume Scattering Index ``hv / (hv + (hh + vv) / 2)`` (Pope et al. 1994)."""
    hh = bands["hh"]
    vv = bands["vv"]
    hv = bands["hv"]
    denom = hv + 0.5 * (hh + vv)
    valid = (hh >= 0.0) & (vv >= 0.0) & (hv >= 0.0) & (denom > 0.0)
    out = np.zeros_like(denom, dtype=np.float64)
    np.divide(hv, denom, out=out, where=valid)
    return out, valid


def _make_sdwi(
    co_pol: str, cross_pol: str
) -> Callable[
    [Dict[str, np.ndarray], Mapping[str, float]], Tuple[np.ndarray, np.ndarray]
]:
    """Build SAR Dual-Polarised Water Index ``ln(10 * co * cross) - 8`` (Jia et al. 2019)."""

    def _calc(
        bands: Dict[str, np.ndarray], _params: Mapping[str, float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        co = bands[co_pol]
        cr = bands[cross_pol]
        prod = 10.0 * co * cr
        valid = (co > 0.0) & (cr > 0.0) & (prod > 0.0)
        out = np.zeros_like(prod, dtype=np.float64)
        np.log(np.where(valid, prod, 1.0), out=out, where=valid)
        out = np.where(valid, out - 8.0, 0.0)
        return out, valid

    return _calc


def _make_power_ratio(
    num_pol: str, den_pol: str
) -> Callable[
    [Dict[str, np.ndarray], Mapping[str, float]], Tuple[np.ndarray, np.ndarray]
]:
    """Build linear-power ratio ``num / den``."""

    def _calc(
        bands: Dict[str, np.ndarray], _params: Mapping[str, float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        num = bands[num_pol]
        den = bands[den_pol]
        valid = (num >= 0.0) & (den > 0.0)
        out = np.zeros_like(den, dtype=np.float64)
        np.divide(num, den, out=out, where=valid)
        return out, valid

    return _calc


def _make_power_diff(
    a_pol: str, b_pol: str
) -> Callable[
    [Dict[str, np.ndarray], Mapping[str, float]], Tuple[np.ndarray, np.ndarray]
]:
    """Build linear-power difference ``a - b``."""

    def _calc(
        bands: Dict[str, np.ndarray], _params: Mapping[str, float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        a = bands[a_pol]
        b = bands[b_pol]
        valid = (a >= 0.0) & (b >= 0.0)
        return a - b, valid

    return _calc


def _make_db_diff(
    a_pol: str, b_pol: str
) -> Callable[
    [Dict[str, np.ndarray], Mapping[str, float]], Tuple[np.ndarray, np.ndarray]
]:
    """Build dB difference ``a_dB - b_dB`` (equivalent to ``10 * log10(a / b)``)."""

    def _calc(
        bands: Dict[str, np.ndarray], _params: Mapping[str, float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        out = bands[a_pol] - bands[b_pol]
        return out, np.ones(out.shape, dtype=bool)

    return _calc


def _make_db_mean(
    a_pol: str, b_pol: str
) -> Callable[
    [Dict[str, np.ndarray], Mapping[str, float]], Tuple[np.ndarray, np.ndarray]
]:
    """Build dB mean ``(a_dB + b_dB) / 2``."""

    def _calc(
        bands: Dict[str, np.ndarray], _params: Mapping[str, float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        out = 0.5 * (bands[a_pol] + bands[b_pol])
        return out, np.ones(out.shape, dtype=bool)

    return _calc


def _make_dual_span(
    co_pol: str, cross_pol: str, to_db: bool = False
) -> Callable[
    [Dict[str, np.ndarray], Mapping[str, float]], Tuple[np.ndarray, np.ndarray]
]:
    """Build dual-pol total backscatter power Span (in linear power or dB)."""

    def _calc(
        bands: Dict[str, np.ndarray], _params: Mapping[str, float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        co = bands[co_pol]
        cr = bands[cross_pol]
        span = co + cr
        if not to_db:
            valid = (co >= 0.0) & (cr >= 0.0)
            return span, valid
        valid = (co >= 0.0) & (cr >= 0.0) & (span > 0.0)
        out = np.zeros_like(span, dtype=np.float64)
        np.log10(np.where(valid, span, 1.0), out=out, where=valid)
        out = np.where(valid, 10.0 * out, 0.0)
        return out, valid

    return _calc


def _make_quad_span(
    to_db: bool = False,
) -> Callable[
    [Dict[str, np.ndarray], Mapping[str, float]], Tuple[np.ndarray, np.ndarray]
]:
    """Build quad-pol total backscatter power Span (in linear power or dB)."""

    def _calc(
        bands: Dict[str, np.ndarray], _params: Mapping[str, float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        hh = bands["hh"]
        vv = bands["vv"]
        hv = bands["hv"]
        span = hh + vv + 2.0 * hv
        if not to_db:
            valid = (hh >= 0.0) & (vv >= 0.0) & (hv >= 0.0)
            return span, valid
        valid = (hh >= 0.0) & (vv >= 0.0) & (hv >= 0.0) & (span > 0.0)
        out = np.zeros_like(span, dtype=np.float64)
        np.log10(np.where(valid, span, 1.0), out=out, where=valid)
        out = np.where(valid, 10.0 * out, 0.0)
        return out, valid

    return _calc


_calc_rvi_vv_vh = _make_dual_rvi("vv", "vh")
_calc_rvi_hh_hv = _make_dual_rvi("hh", "hv")
_calc_dprvic_vv_vh = _make_dprvic("vv", "vh")
_calc_dprvic_hh_hv = _make_dprvic("hh", "hv")
_calc_vddpi = _make_ddpi("vv", "vh")
_calc_hddpi = _make_ddpi("hh", "hv")
_calc_rfdi_hh_hv = _make_norm_diff_power("hh", "hv")
_calc_rfdi_vv_vh = _make_norm_diff_power("vv", "vh")
_calc_ndi_vh_vv = _make_norm_diff_power("vh", "vv")
_calc_ndi_hv_hh = _make_norm_diff_power("hv", "hh")
_calc_sdwi_vv_vh = _make_sdwi("vv", "vh")
_calc_sdwi_hh_hv = _make_sdwi("hh", "hv")
_calc_cr_vh_vv = _make_power_ratio("vh", "vv")
_calc_cr_vv_vh = _make_power_ratio("vv", "vh")
_calc_cr_hv_hh = _make_power_ratio("hv", "hh")
_calc_cr_hh_hv = _make_power_ratio("hh", "hv")
_calc_pr_hh_vv = _make_power_ratio("hh", "vv")
_calc_pr_vv_hh = _make_power_ratio("vv", "hh")
_calc_pol_diff_vv_vh = _make_power_diff("vv", "vh")
_calc_pol_diff_hh_hv = _make_power_diff("hh", "hv")
_calc_cr_vh_vv_db = _make_db_diff("vh", "vv")
_calc_cr_vv_vh_db = _make_db_diff("vv", "vh")
_calc_cr_hv_hh_db = _make_db_diff("hv", "hh")
_calc_cr_hh_hv_db = _make_db_diff("hh", "hv")
_calc_pr_hh_vv_db = _make_db_diff("hh", "vv")
_calc_pr_vv_hh_db = _make_db_diff("vv", "hh")
_calc_mean_vv_vh_db = _make_db_mean("vv", "vh")
_calc_mean_hh_hv_db = _make_db_mean("hh", "hv")
_calc_span_vv_vh = _make_dual_span("vv", "vh", to_db=False)
_calc_span_vv_vh_db = _make_dual_span("vv", "vh", to_db=True)
_calc_span_hh_hv = _make_dual_span("hh", "hv", to_db=False)
_calc_span_hh_hv_db = _make_dual_span("hh", "hv", to_db=True)
_calc_span_quad = _make_quad_span(to_db=False)
_calc_span_quad_db = _make_quad_span(to_db=True)


_SAR_INDEX_REGISTRY: Dict[str, _SARIndexSpec] = {
    # Vegetation, Canopy & Forest Biomass (linear power)
    "rvi": _SARIndexSpec(
        "RVI",
        ("hh", "vv", "hv"),
        "(8 * hv) / (hh + vv + 2 * hv)",
        _calc_rvi_quad_block,
        units="power",
    ),
    "rvi_quad": _SARIndexSpec(
        "RVI_Quad",
        ("hh", "vv", "hv"),
        "(8 * hv) / (hh + vv + 2 * hv)",
        _calc_rvi_quad_block,
        units="power",
    ),
    "rvi_vv_vh": _SARIndexSpec(
        "RVI_VV_VH",
        ("vv", "vh"),
        "(4 * vh) / (vv + vh)",
        _calc_rvi_vv_vh,
        units="power",
    ),
    "dprvi_vv": _SARIndexSpec(
        "DpRVI_VV",
        ("vv", "vh"),
        "(4 * vh) / (vv + vh)",
        _calc_rvi_vv_vh,
        units="power",
    ),
    "rvi4s1": _SARIndexSpec(
        "RVI4S1",
        ("vv", "vh"),
        "(4 * vh) / (vv + vh)",
        _calc_rvi_vv_vh,
        units="power",
    ),
    "rvi_hh_hv": _SARIndexSpec(
        "RVI_HH_HV",
        ("hh", "hv"),
        "(4 * hv) / (hh + hv)",
        _calc_rvi_hh_hv,
        units="power",
    ),
    "dprvi_hh": _SARIndexSpec(
        "DpRVI_HH",
        ("hh", "hv"),
        "(4 * hv) / (hh + hv)",
        _calc_rvi_hh_hv,
        units="power",
    ),
    "dprvic": _SARIndexSpec(
        "DpRVIc",
        ("vv", "vh"),
        "(q * (q + 3)) / (q + 1)^2, q = vh / vv",
        _calc_dprvic_vv_vh,
        units="power",
    ),
    "dprvic_vv_vh": _SARIndexSpec(
        "DpRVIc_VV_VH",
        ("vv", "vh"),
        "(q * (q + 3)) / (q + 1)^2, q = vh / vv",
        _calc_dprvic_vv_vh,
        units="power",
    ),
    "dprvic_hh_hv": _SARIndexSpec(
        "DpRVIc_HH_HV",
        ("hh", "hv"),
        "(q * (q + 3)) / (q + 1)^2, q = hv / hh",
        _calc_dprvic_hh_hv,
        units="power",
    ),
    "dpsvi": _SARIndexSpec(
        "DPSVI",
        ("vv", "vh"),
        "(((vv_max - vv) + vh) / sqrt(2)) * ((vv + vh) / vv) * vh",
        _calc_dpsvi_block,
        units="power",
        requires_vv_max=True,
    ),
    "dpsvim": _SARIndexSpec(
        "DPSVIm",
        ("vv", "vh"),
        "(vv^2 + vv * vh) / sqrt(2)",
        _calc_dpsvim_block,
        units="power",
    ),
    "vddpi": _SARIndexSpec(
        "VDDPI",
        ("vv", "vh"),
        "(vv + vh) / vv",
        _calc_vddpi,
        units="power",
    ),
    "hddpi": _SARIndexSpec(
        "HDDPI",
        ("hh", "hv"),
        "(hh + hv) / hh",
        _calc_hddpi,
        units="power",
    ),
    "rfdi": _SARIndexSpec(
        "RFDI",
        ("hh", "hv"),
        "(hh - hv) / (hh + hv)",
        _calc_rfdi_hh_hv,
        units="power",
    ),
    "rfdi_hh_hv": _SARIndexSpec(
        "RFDI_HH_HV",
        ("hh", "hv"),
        "(hh - hv) / (hh + hv)",
        _calc_rfdi_hh_hv,
        units="power",
    ),
    "ndpi_hh_hv": _SARIndexSpec(
        "NDPI_HH_HV",
        ("hh", "hv"),
        "(hh - hv) / (hh + hv)",
        _calc_rfdi_hh_hv,
        units="power",
    ),
    "ndpdi_hh_hv": _SARIndexSpec(
        "NDPDI_HH_HV",
        ("hh", "hv"),
        "(hh - hv) / (hh + hv)",
        _calc_rfdi_hh_hv,
        units="power",
    ),
    "ndi_hh_hv": _SARIndexSpec(
        "NDI_HH_HV",
        ("hh", "hv"),
        "(hh - hv) / (hh + hv)",
        _calc_rfdi_hh_hv,
        units="power",
    ),
    "rfdi_vv_vh": _SARIndexSpec(
        "RFDI_VV_VH",
        ("vv", "vh"),
        "(vv - vh) / (vv + vh)",
        _calc_rfdi_vv_vh,
        units="power",
    ),
    "ndpi_vv_vh": _SARIndexSpec(
        "NDPI_VV_VH",
        ("vv", "vh"),
        "(vv - vh) / (vv + vh)",
        _calc_rfdi_vv_vh,
        units="power",
    ),
    "ndpdi": _SARIndexSpec(
        "NDPDI",
        ("vv", "vh"),
        "(vv - vh) / (vv + vh)",
        _calc_rfdi_vv_vh,
        units="power",
    ),
    "ndpdi_vv_vh": _SARIndexSpec(
        "NDPDI_VV_VH",
        ("vv", "vh"),
        "(vv - vh) / (vv + vh)",
        _calc_rfdi_vv_vh,
        units="power",
    ),
    "ndi_vv_vh": _SARIndexSpec(
        "NDI_VV_VH",
        ("vv", "vh"),
        "(vv - vh) / (vv + vh)",
        _calc_rfdi_vv_vh,
        units="power",
    ),
    "ndi_vh_vv": _SARIndexSpec(
        "NDI_VH_VV",
        ("vv", "vh"),
        "(vh - vv) / (vh + vv)",
        _calc_ndi_vh_vv,
        units="power",
    ),
    "nrpb": _SARIndexSpec(
        "NRPB",
        ("vv", "vh"),
        "(vh - vv) / (vh + vv)",
        _calc_ndi_vh_vv,
        units="power",
    ),
    "ndi_hv_hh": _SARIndexSpec(
        "NDI_HV_HH",
        ("hh", "hv"),
        "(hv - hh) / (hv + hh)",
        _calc_ndi_hv_hh,
        units="power",
    ),
    "bmi": _SARIndexSpec(
        "BMI",
        ("hh", "vv"),
        "(hh + vv) / 2",
        _calc_bmi_block,
        units="power",
    ),
    "csi": _SARIndexSpec(
        "CSI",
        ("hh", "vv"),
        "vv / (vv + hh)",
        _calc_csi_block,
        units="power",
    ),
    "vsi": _SARIndexSpec(
        "VSI",
        ("hh", "vv", "hv"),
        "hv / (hv + (hh + vv) / 2)",
        _calc_vsi_block,
        units="power",
    ),
    # Water Indices (linear power)
    "sdwi": _SARIndexSpec(
        "SDWI",
        ("vv", "vh"),
        "ln(10 * vv * vh) - 8",
        _calc_sdwi_vv_vh,
        units="power",
    ),
    "sdwi_vv_vh": _SARIndexSpec(
        "SDWI_VV_VH",
        ("vv", "vh"),
        "ln(10 * vv * vh) - 8",
        _calc_sdwi_vv_vh,
        units="power",
    ),
    "sdwi_hh_hv": _SARIndexSpec(
        "SDWI_HH_HV",
        ("hh", "hv"),
        "ln(10 * hh * hv) - 8",
        _calc_sdwi_hh_hv,
        units="power",
    ),
    # Polarisation Ratios & Differences (linear power)
    "cr_vh_vv": _SARIndexSpec(
        "CR_VH_VV",
        ("vv", "vh"),
        "vh / vv",
        _calc_cr_vh_vv,
        units="power",
    ),
    "pr_vh_vv": _SARIndexSpec(
        "PR_VH_VV",
        ("vv", "vh"),
        "vh / vv",
        _calc_cr_vh_vv,
        units="power",
    ),
    "cr_vv_vh": _SARIndexSpec(
        "CR_VV_VH",
        ("vv", "vh"),
        "vv / vh",
        _calc_cr_vv_vh,
        units="power",
    ),
    "pr_vv_vh": _SARIndexSpec(
        "PR_VV_VH",
        ("vv", "vh"),
        "vv / vh",
        _calc_cr_vv_vh,
        units="power",
    ),
    "cr_hv_hh": _SARIndexSpec(
        "CR_HV_HH",
        ("hh", "hv"),
        "hv / hh",
        _calc_cr_hv_hh,
        units="power",
    ),
    "pr_hv_hh": _SARIndexSpec(
        "PR_HV_HH",
        ("hh", "hv"),
        "hv / hh",
        _calc_cr_hv_hh,
        units="power",
    ),
    "cr_hh_hv": _SARIndexSpec(
        "CR_HH_HV",
        ("hh", "hv"),
        "hh / hv",
        _calc_cr_hh_hv,
        units="power",
    ),
    "pr_hh_hv": _SARIndexSpec(
        "PR_HH_HV",
        ("hh", "hv"),
        "hh / hv",
        _calc_cr_hh_hv,
        units="power",
    ),
    "pr_hh_vv": _SARIndexSpec(
        "PR_HH_VV",
        ("hh", "vv"),
        "hh / vv",
        _calc_pr_hh_vv,
        units="power",
    ),
    "pr_vv_hh": _SARIndexSpec(
        "PR_VV_HH",
        ("hh", "vv"),
        "vv / hh",
        _calc_pr_vv_hh,
        units="power",
    ),
    "pol_diff_vv_vh": _SARIndexSpec(
        "Pol_Diff_VV_VH",
        ("vv", "vh"),
        "vv - vh",
        _calc_pol_diff_vv_vh,
        units="power",
    ),
    "pol_diff_hh_hv": _SARIndexSpec(
        "Pol_Diff_HH_HV",
        ("hh", "hv"),
        "hh - hv",
        _calc_pol_diff_hh_hv,
        units="power",
    ),
    # Polarisation Ratios & Summaries (dB)
    "cr_vh_vv_db": _SARIndexSpec(
        "CR_VH_VV_dB",
        ("vv", "vh"),
        "vh_dB - vv_dB",
        _calc_cr_vh_vv_db,
        units="dB",
    ),
    "pr_vh_vv_db": _SARIndexSpec(
        "PR_VH_VV_dB",
        ("vv", "vh"),
        "vh_dB - vv_dB",
        _calc_cr_vh_vv_db,
        units="dB",
    ),
    "cr_vv_vh_db": _SARIndexSpec(
        "CR_VV_VH_dB",
        ("vv", "vh"),
        "vv_dB - vh_dB",
        _calc_cr_vv_vh_db,
        units="dB",
    ),
    "pr_vv_vh_db": _SARIndexSpec(
        "PR_VV_VH_dB",
        ("vv", "vh"),
        "vv_dB - vh_dB",
        _calc_cr_vv_vh_db,
        units="dB",
    ),
    "cr_hv_hh_db": _SARIndexSpec(
        "CR_HV_HH_dB",
        ("hh", "hv"),
        "hv_dB - hh_dB",
        _calc_cr_hv_hh_db,
        units="dB",
    ),
    "pr_hv_hh_db": _SARIndexSpec(
        "PR_HV_HH_dB",
        ("hh", "hv"),
        "hv_dB - hh_dB",
        _calc_cr_hv_hh_db,
        units="dB",
    ),
    "cr_hh_hv_db": _SARIndexSpec(
        "CR_HH_HV_dB",
        ("hh", "hv"),
        "hh_dB - hv_dB",
        _calc_cr_hh_hv_db,
        units="dB",
    ),
    "pr_hh_hv_db": _SARIndexSpec(
        "PR_HH_HV_dB",
        ("hh", "hv"),
        "hh_dB - hv_dB",
        _calc_cr_hh_hv_db,
        units="dB",
    ),
    "pr_hh_vv_db": _SARIndexSpec(
        "PR_HH_VV_dB",
        ("hh", "vv"),
        "hh_dB - vv_dB",
        _calc_pr_hh_vv_db,
        units="dB",
    ),
    "pr_vv_hh_db": _SARIndexSpec(
        "PR_VV_HH_dB",
        ("hh", "vv"),
        "vv_dB - hh_dB",
        _calc_pr_vv_hh_db,
        units="dB",
    ),
    "mean_vv_vh_db": _SARIndexSpec(
        "Mean_VV_VH_dB",
        ("vv", "vh"),
        "(vv_dB + vh_dB) / 2",
        _calc_mean_vv_vh_db,
        units="dB",
    ),
    "mean_hh_hv_db": _SARIndexSpec(
        "Mean_HH_HV_dB",
        ("hh", "hv"),
        "(hh_dB + hv_dB) / 2",
        _calc_mean_hh_hv_db,
        units="dB",
    ),
    # Total Backscatter Power (Span)
    "span_vv_vh": _SARIndexSpec(
        "Span_VV_VH",
        ("vv", "vh"),
        "vv + vh",
        _calc_span_vv_vh,
        units="power",
    ),
    "span_vv_vh_db": _SARIndexSpec(
        "Span_VV_VH_dB",
        ("vv", "vh"),
        "10 * log10(vv + vh)",
        _calc_span_vv_vh_db,
        units="power",
    ),
    "span_hh_hv": _SARIndexSpec(
        "Span_HH_HV",
        ("hh", "hv"),
        "hh + hv",
        _calc_span_hh_hv,
        units="power",
    ),
    "span_hh_hv_db": _SARIndexSpec(
        "Span_HH_HV_dB",
        ("hh", "hv"),
        "10 * log10(hh + hv)",
        _calc_span_hh_hv_db,
        units="power",
    ),
    "span_quad": _SARIndexSpec(
        "Span_Quad",
        ("hh", "vv", "hv"),
        "hh + vv + 2 * hv",
        _calc_span_quad,
        units="power",
    ),
    "span_quad_db": _SARIndexSpec(
        "Span_Quad_dB",
        ("hh", "vv", "hv"),
        "10 * log10(hh + vv + 2 * hv)",
        _calc_span_quad_db,
        units="power",
    ),
}


def get_available_sar_indices() -> List[str]:
    """
    Return the sorted list of SAR index names supported by
    :func:`rsgispy.imagecalc.calcindices.calc_sar_indices`.

    :return: Sorted list of supported SAR index names.

    .. code:: python

        import rsgispy.imagecalc.calcindices

        names = rsgispy.imagecalc.calcindices.get_available_sar_indices()
    """
    return sorted(spec.canonical_name for spec in _SAR_INDEX_REGISTRY.values())


def lookup_sar_index_spec(index_name: str) -> _SARIndexSpec:
    """Validate and resolve ``index_name`` in ``_SAR_INDEX_REGISTRY``."""
    if not isinstance(index_name, str) or not index_name.strip():
        raise rsgispy.RSGISPyException(
            "index_name must be a non-empty string."
        )
    key = index_name.strip().lower()
    if key not in _SAR_INDEX_REGISTRY:
        raise rsgispy.RSGISPyException(
            f"Unknown SAR index '{index_name}'. "
            f"Supported SAR indices: {', '.join(get_available_sar_indices())}."
        )
    return _SAR_INDEX_REGISTRY[key]


def get_sar_index_req_bands(index_name: str) -> Tuple[str, ...]:
    """
    Return the tuple of canonical polarisation band names required to compute
    ``index_name``.

    :param index_name: Name of the SAR index (case-insensitive).
    :return: Tuple of canonical polarisation names (e.g., ``('vv', 'vh')``).

    .. code:: python

        import rsgispy.imagecalc.calcindices

        req_bands = rsgispy.imagecalc.calcindices.get_sar_index_req_bands("RVI_VV_VH")
    """
    return lookup_sar_index_spec(index_name).req_bands


def get_sar_index_equation(index_name: str) -> str:
    """
    Return the mathematical equation string for SAR index ``index_name``.

    :param index_name: Name of the SAR index (case-insensitive).
    :return: Equation string using canonical polarisation names.

    .. code:: python

        import rsgispy.imagecalc.calcindices

        eq = rsgispy.imagecalc.calcindices.get_sar_index_equation("RVI_VV_VH")
    """
    return lookup_sar_index_spec(index_name).equation


def get_sar_index_units(index_name: str) -> str:
    """
    Return the working backscatter domain (``'power'`` or ``'dB'``) in which
    SAR index ``index_name`` is evaluated.

    :param index_name: Name of the SAR index (case-insensitive).
    :return: ``'power'`` (linear intensity) or ``'dB'`` (decibels).

    .. code:: python

        import rsgispy.imagecalc.calcindices

        units = rsgispy.imagecalc.calcindices.get_sar_index_units("RVI_VV_VH")
    """
    return lookup_sar_index_spec(index_name).units


def _power_to_db(
    arr: np.ndarray, invalid: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert linear power array to dB, marking non-positive pixels invalid."""
    pos_ok = (~invalid) & np.isfinite(arr) & (arr > 0.0)
    db_arr = np.zeros_like(arr, dtype=np.float64)
    np.log10(np.where(pos_ok, arr, 1.0), out=db_arr, where=pos_ok)
    return np.where(pos_ok, 10.0 * db_arr, np.nan), ~pos_ok


def _db_to_power(
    arr: np.ndarray, invalid: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert dB array to linear power, marking non-finite pixels invalid."""
    base_ok = (~invalid) & np.isfinite(arr)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        pwr = np.power(10.0, np.where(base_ok, arr / 10.0, 0.0))
    pwr_ok = base_ok & np.isfinite(pwr) & (pwr >= 0.0)
    return np.where(pwr_ok, pwr, np.nan), ~pwr_ok


def convert_sar_block_units(
    phys_bands: Mapping[str, np.ndarray],
    invalid_masks: Mapping[str, np.ndarray],
    in_units: str,
    need_power: bool,
    need_db: bool,
) -> Tuple[
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
]:
    """
    Convert unscaled physical SAR bands into linear power and/or dB domains as
    needed, tracking per-band invalid masks for non-finite or non-positive
    values.
    """
    power_bands: Dict[str, np.ndarray] = {}
    power_invalid: Dict[str, np.ndarray] = {}
    db_bands: Dict[str, np.ndarray] = {}
    db_invalid: Dict[str, np.ndarray] = {}

    if in_units == "power":
        if need_power:
            for b_name, arr in phys_bands.items():
                power_bands[b_name] = arr
                power_invalid[b_name] = (
                    invalid_masks[b_name] | (~np.isfinite(arr)) | (arr < 0.0)
                )
        if need_db:
            for b_name, arr in phys_bands.items():
                db_bands[b_name], db_invalid[b_name] = _power_to_db(
                    arr, invalid_masks[b_name]
                )
    else:
        if need_db:
            for b_name, arr in phys_bands.items():
                db_bands[b_name] = arr
                db_invalid[b_name] = invalid_masks[b_name] | (~np.isfinite(arr))
        if need_power:
            for b_name, arr in phys_bands.items():
                power_bands[b_name], power_invalid[b_name] = _db_to_power(
                    arr, invalid_masks[b_name]
                )

    return power_bands, power_invalid, db_bands, db_invalid
