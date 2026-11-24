"""
Internal parameter-estimation and pixel-transformation helpers for
:mod:`rsgispy.imagecalc.rescale`.
"""

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

import rsgispy
from rsgispy.imageprocess import (
    build_valid_mask,
    resolve_out_nodata as _ip_resolve_out_nodata,
)

try:
    import sklearn.preprocessing as sk_prep

    HAVE_SKLEARN = True
except ImportError:
    sk_prep = None
    HAVE_SKLEARN = False

STRETCH_METHODS = (
    "none",
    "linear",
    "stdev",
    "cumulative",
    "user",
)

STAT_NORM_METHODS = (
    "zscore",
    "robust_zscore",
    "decimal_scaling",
    "log",
    "log1p",
    "sigmoid",
    "tanh",
    "softmax",
    "l1",
    "l2",
    "max_norm",
)

SKLEARN_NORM_METHODS = (
    "standard_scaler",
    "minmax_scaler",
    "maxabs_scaler",
    "robust_scaler",
    "power_transformer",
    "quantile_transformer",
    "normalizer",
)

ALL_NORM_METHODS = STRETCH_METHODS + STAT_NORM_METHODS + SKLEARN_NORM_METHODS

_METHOD_ALIASES: Dict[Any, str] = {
    rsgispy.IMG_STRETCH_NONE: "none",
    rsgispy.IMG_STRETCH_USER: "user",
    rsgispy.IMG_STRETCH_LINEAR: "linear",
    rsgispy.IMG_STRETCH_STDEV: "stdev",
    rsgispy.IMG_STRETCH_CUMULATIVE: "cumulative",
    "none": "none",
    "identity": "none",
    "user": "user",
    "manual": "user",
    "linear": "linear",
    "minmax_stretch": "linear",
    "stdev": "stdev",
    "stddev": "stdev",
    "cumulative": "cumulative",
    "percentile": "cumulative",
    "zscore": "zscore",
    "z_score": "zscore",
    "z-score": "zscore",
    "standardise": "zscore",
    "standardize": "zscore",
    "robust_zscore": "robust_zscore",
    "robust_z_score": "robust_zscore",
    "mad": "robust_zscore",
    "decimal_scaling": "decimal_scaling",
    "decimal": "decimal_scaling",
    "log": "log",
    "log1p": "log1p",
    "sigmoid": "sigmoid",
    "logistic": "sigmoid",
    "tanh": "tanh",
    "softmax": "softmax",
    "unit_sum": "softmax",
    "l1": "l1",
    "l2": "l2",
    "max_norm": "max_norm",
    "maxnorm": "max_norm",
    "standardscaler": "standard_scaler",
    "standard_scaler": "standard_scaler",
    "minmaxscaler": "minmax_scaler",
    "minmax_scaler": "minmax_scaler",
    "minmax": "minmax_scaler",
    "maxabsscaler": "maxabs_scaler",
    "maxabs_scaler": "maxabs_scaler",
    "maxabs": "maxabs_scaler",
    "robustscaler": "robust_scaler",
    "robust_scaler": "robust_scaler",
    "robust": "robust_scaler",
    "powertransformer": "power_transformer",
    "power_transformer": "power_transformer",
    "power": "power_transformer",
    "quantiletransformer": "quantile_transformer",
    "quantile_transformer": "quantile_transformer",
    "quantile": "quantile_transformer",
    "normalizer": "normalizer",
    "normaliser": "normalizer",
}

DEFAULT_RANGE_METHODS = {"linear", "stdev", "cumulative", "user", "minmax_scaler"}
DEFAULT_TRIM_METHODS = {"stdev", "cumulative", "user"}
PIXEL_VECTOR_METHODS = {"l1", "l2", "max_norm", "normalizer", "softmax"}


def resolve_norm_method(method: Union[str, int]) -> str:
    """Normalise an integer ``IMG_STRETCH_*`` constant or method name string."""
    if isinstance(method, bool):
        raise rsgispy.RSGISPyException(
            f"Unsupported normalisation method: {method!r}."
        )
    if isinstance(method, (int, np.integer)):
        key: Any = int(method)
    elif isinstance(method, str):
        key = method.strip().lower().replace(" ", "_")
    else:
        raise rsgispy.RSGISPyException(
            f"Normalisation method must be a string or IMG_STRETCH_* constant, "
            f"got {method!r}."
        )
    if key not in _METHOD_ALIASES:
        raise rsgispy.RSGISPyException(
            f"Unsupported normalisation method {method!r}. "
            f"Supported methods: {', '.join(ALL_NORM_METHODS)}."
        )
    canonical = _METHOD_ALIASES[key]
    if canonical in SKLEARN_NORM_METHODS:
        rsgispy.check_optional_dep(
            HAVE_SKLEARN,
            f"normalise_img_pxl_vals(method={method!r})",
            "scikit-learn",
            "classification",
        )
    return canonical


def resolve_out_bounds(
    method: str,
    out_min: Optional[float],
    out_max: Optional[float],
) -> Tuple[Optional[float], Optional[float]]:
    """Validate and resolve optional output ``(out_min, out_max)`` bounds."""
    if (out_min is None) != (out_max is None):
        raise rsgispy.RSGISPyException(
            "out_min and out_max must either both be specified or both be None."
        )
    if out_min is None and out_max is None:
        if method in DEFAULT_RANGE_METHODS:
            return 0.0, 1.0
        return None, None
    lo = float(out_min)  # type: ignore[arg-type]
    hi = float(out_max)  # type: ignore[arg-type]
    if not (math.isfinite(lo) and math.isfinite(hi)):
        raise rsgispy.RSGISPyException("out_min and out_max must be finite.")
    if lo == hi:
        raise rsgispy.RSGISPyException("out_min and out_max must differ.")
    return lo, hi


def _linear_map_array(
    arr: np.ndarray,
    in_min: float,
    in_max: float,
    out_min: float,
    out_max: float,
) -> np.ndarray:
    """Linearly map values from ``[in_min, in_max]`` to ``[out_min, out_max]``."""
    span = float(in_max) - float(in_min)
    if span == 0.0 or not math.isfinite(span):
        return np.full(arr.shape, float(out_min), dtype=np.float64)
    return (
        ((arr - float(in_min)) / span) * (float(out_max) - float(out_min))
    ) + float(out_min)


def _extract_user_bounds(
    in_min_max: Optional[Union[Mapping[str, float], Sequence[Mapping[str, float]]]],
    sel_idx: int,
    n_selected: int,
) -> Tuple[float, float]:
    """Extract manual ``('min', 'max')`` bounds for one selected band."""
    if in_min_max is None:
        raise rsgispy.RSGISPyException(
            "in_min_max must be provided when method is 'user' / 'manual'."
        )
    if isinstance(in_min_max, Mapping):
        spec = in_min_max
    else:
        if len(in_min_max) != n_selected:
            raise rsgispy.RSGISPyException(
                f"Length of in_min_max ({len(in_min_max)}) must match number of "
                f"selected bands ({n_selected})."
            )
        spec = in_min_max[sel_idx]
    if not isinstance(spec, Mapping) or "min" not in spec or "max" not in spec:
        raise rsgispy.RSGISPyException(
            "Each in_min_max entry must be a mapping containing 'min' and 'max'."
        )
    lo = float(spec["min"])
    hi = float(spec["max"])
    if not (math.isfinite(lo) and math.isfinite(hi)):
        raise rsgispy.RSGISPyException("in_min_max 'min' and 'max' must be finite.")
    return lo, hi


def _fit_sklearn_band_params(
    method: str,
    vals: np.ndarray,
    eff_out_min: Optional[float],
    eff_out_max: Optional[float],
    kwargs: Dict[str, Any],
    rnd_seed: Optional[int],
) -> Tuple[Dict[str, Any], np.ndarray]:
    """Fit a scikit-learn scaler/transformer on 1D valid pixel values."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    col = vals.reshape(-1, 1)
    if method == "standard_scaler":
        scaler = sk_prep.StandardScaler(**kwargs).fit(col)
        transformed = scaler.transform(col).ravel()
        return {
            "with_mean": bool(scaler.with_mean),
            "with_std": bool(scaler.with_std),
            "mean": float(scaler.mean_[0]) if scaler.mean_ is not None else 0.0,
            "scale": float(scaler.scale_[0]) if scaler.scale_ is not None else 1.0,
        }, transformed

    if method == "minmax_scaler":
        feat_range = kwargs.pop(
            "feature_range",
            (
                0.0 if eff_out_min is None else eff_out_min,
                1.0 if eff_out_max is None else eff_out_max,
            ),
        )
        scaler = sk_prep.MinMaxScaler(feature_range=feat_range, **kwargs).fit(col)
        transformed = scaler.transform(col).ravel()
        return {
            "feature_range": [float(feat_range[0]), float(feat_range[1])],
            "data_min": float(scaler.data_min_[0]),
            "data_max": float(scaler.data_max_[0]),
            "scale": float(scaler.scale_[0]),
            "min_offset": float(scaler.min_[0]),
        }, transformed

    if method == "maxabs_scaler":
        scaler = sk_prep.MaxAbsScaler(**kwargs).fit(col)
        transformed = scaler.transform(col).ravel()
        return {
            "max_abs": float(scaler.max_abs_[0]),
            "scale": float(scaler.scale_[0]),
        }, transformed

    if method == "robust_scaler":
        scaler = sk_prep.RobustScaler(**kwargs).fit(col)
        transformed = scaler.transform(col).ravel()
        return {
            "with_centering": bool(scaler.with_centering),
            "with_scaling": bool(scaler.with_scaling),
            "quantile_range": [
                float(scaler.quantile_range[0]),
                float(scaler.quantile_range[1]),
            ],
            "center": float(scaler.center_[0]) if scaler.center_ is not None else 0.0,
            "scale": float(scaler.scale_[0]) if scaler.scale_ is not None else 1.0,
        }, transformed

    if method == "power_transformer":
        pt_method = kwargs.get("method", "yeo-johnson")
        if pt_method == "box-cox" and float(np.min(vals)) <= 0.0:
            raise rsgispy.RSGISPyException(
                "PowerTransformer(method='box-cox') requires strictly positive "
                "pixel values."
            )
        scaler = sk_prep.PowerTransformer(**kwargs).fit(col)
        transformed = scaler.transform(col).ravel()
        sc = getattr(scaler, "_scaler", None)
        return {
            "pt_method": str(scaler.method),
            "standardize": bool(scaler.standardize),
            "lambdas": [float(v) for v in scaler.lambdas_],
            "scaler_mean": (
                [float(v) for v in sc.mean_] if sc is not None else None
            ),
            "scaler_scale": (
                [float(v) for v in sc.scale_] if sc is not None else None
            ),
            "scaler_var": (
                [float(v) for v in sc.var_] if sc is not None else None
            ),
        }, transformed

    # quantile_transformer
    qt_kwargs = dict(kwargs)
    if "random_state" not in qt_kwargs and rnd_seed is not None:
        qt_kwargs["random_state"] = rnd_seed
    qt_kwargs["n_quantiles"] = min(
        int(qt_kwargs.get("n_quantiles", 1000)), max(1, int(vals.size))
    )
    scaler = sk_prep.QuantileTransformer(**qt_kwargs).fit(col)
    transformed = scaler.transform(col).ravel()
    return {
        "n_quantiles": int(scaler.n_quantiles_),
        "output_distribution": str(scaler.output_distribution),
        "quantiles": scaler.quantiles_.tolist(),
        "references": scaler.references_.tolist(),
    }, transformed


def fit_band_norm_params(
    method: str,
    vals: np.ndarray,
    band_idx: int,
    eff_nd: Optional[float],
    eff_out_min: Optional[float],
    eff_out_max: Optional[float],
    lower: float,
    upper: float,
    n_stdevs: float,
    user_bounds: Optional[Tuple[float, float]],
    method_kwargs: Optional[Mapping[str, Any]],
    rnd_seed: Optional[int],
) -> Dict[str, Any]:
    """Fit JSON-serialisable normalisation parameters for a single raster band."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    if vals.size == 0:
        raise rsgispy.RSGISPyException(
            f"Band {band_idx} contains no valid finite non-nodata pixels."
        )
    kwargs = dict(method_kwargs or {})
    data_min = float(np.min(vals))
    data_max = float(np.max(vals))
    mean_val = float(np.mean(vals))
    std_val = float(np.std(vals))
    extra_params: Dict[str, Any] = {}

    if method == "none":
        transformed = vals.copy()
    elif method == "linear":
        extra_params = {"stretch_min": data_min, "stretch_max": data_max}
        transformed = _linear_map_array(
            vals,
            data_min,
            data_max,
            0.0 if eff_out_min is None else eff_out_min,
            1.0 if eff_out_max is None else eff_out_max,
        )
    elif method == "stdev":
        s_low = max(data_min, mean_val - float(n_stdevs) * std_val)
        s_high = min(data_max, mean_val + float(n_stdevs) * std_val)
        extra_params = {
            "stretch_min": s_low,
            "stretch_max": s_high,
            "n_stdevs": float(n_stdevs),
        }
        transformed = _linear_map_array(
            vals,
            s_low,
            s_high,
            0.0 if eff_out_min is None else eff_out_min,
            1.0 if eff_out_max is None else eff_out_max,
        )
    elif method == "cumulative":
        p_low, p_high = np.percentile(vals, [float(lower), float(upper)])
        extra_params = {
            "stretch_min": float(p_low),
            "stretch_max": float(p_high),
            "lower": float(lower),
            "upper": float(upper),
        }
        transformed = _linear_map_array(
            vals,
            float(p_low),
            float(p_high),
            0.0 if eff_out_min is None else eff_out_min,
            1.0 if eff_out_max is None else eff_out_max,
        )
    elif method == "user":
        if user_bounds is None:
            raise rsgispy.RSGISPyException(
                "in_min_max bounds are required for method='user'."
            )
        u_low, u_high = user_bounds
        extra_params = {"stretch_min": float(u_low), "stretch_max": float(u_high)}
        transformed = _linear_map_array(
            vals,
            u_low,
            u_high,
            0.0 if eff_out_min is None else eff_out_min,
            1.0 if eff_out_max is None else eff_out_max,
        )
    elif method == "zscore":
        scale = std_val if std_val > 0.0 else 1.0
        extra_params = {"mean": mean_val, "scale": scale}
        transformed = (vals - mean_val) / scale
    elif method == "robust_zscore":
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med))) * 1.4826
        scale = mad if mad > 0.0 else 1.0
        extra_params = {"median": med, "scale": scale}
        transformed = (vals - med) / scale
    elif method == "decimal_scaling":
        max_abs = float(np.max(np.abs(vals)))
        exp_j = math.ceil(math.log10(max_abs)) if max_abs > 0.0 else 0
        denom = float(10.0**exp_j)
        extra_params = {"exponent": int(exp_j), "denom": denom}
        transformed = vals / denom
    elif method in ("log", "log1p"):
        min_required = 1e-6 if method == "log" else 0.0
        shift = (
            max(0.0, min_required - data_min)
            if "shift" not in kwargs
            else float(kwargs["shift"])
        )
        if np.any(vals + shift <= 0.0) and method == "log":
            raise rsgispy.RSGISPyException(
                f"Band {band_idx} has values + shift <= 0 for method='log'."
            )
        if np.any(vals + shift <= -1.0) and method == "log1p":
            raise rsgispy.RSGISPyException(
                f"Band {band_idx} has values + shift <= -1 for method='log1p'."
            )
        extra_params = {"shift": float(shift)}
        transformed = (
            np.log(vals + shift) if method == "log" else np.log1p(vals + shift)
        )
    elif method in ("sigmoid", "tanh"):
        scale = std_val if std_val > 0.0 else 1.0
        extra_params = {"mean": mean_val, "scale": scale}
        z = (vals - mean_val) / scale
        if method == "sigmoid":
            transformed = 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))
        else:
            transformed = np.tanh(z)
    elif method in SKLEARN_NORM_METHODS:
        extra_params, transformed = _fit_sklearn_band_params(
            method=method,
            vals=vals,
            eff_out_min=eff_out_min,
            eff_out_max=eff_out_max,
            kwargs=kwargs,
            rnd_seed=rnd_seed,
        )
    else:
        raise rsgispy.RSGISPyException(f"Unhandled normalisation method: {method!r}.")

    band_rec: Dict[str, Any] = {
        "band": int(band_idx),
        "no_data_val": eff_nd,
        "in_min": data_min,
        "in_max": data_max,
        "mean": mean_val,
        "std": std_val,
        "trans_min": float(np.min(transformed)),
        "trans_max": float(np.max(transformed)),
        "out_min": eff_out_min,
        "out_max": eff_out_max,
    }
    band_rec.update(extra_params)
    return band_rec


def apply_band_norm_1d(
    method: str,
    vals: np.ndarray,
    band_params: Mapping[str, Any],
    eff_out_min: Optional[float],
    eff_out_max: Optional[float],
    trim_to_limits: bool,
) -> np.ndarray:
    """Apply fitted single-band normalisation parameters to a 1D float64 array."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-branches
    if vals.size == 0:
        return vals.copy()

    mapped_native_range = False
    if method == "none":
        out = vals.copy()
    elif method in ("linear", "stdev", "cumulative", "user"):
        stored_min = band_params.get("out_min")
        stored_max = band_params.get("out_max")
        target_min = (
            float(eff_out_min)
            if eff_out_min is not None
            else (float(stored_min) if stored_min is not None else 0.0)
        )
        target_max = (
            float(eff_out_max)
            if eff_out_max is not None
            else (float(stored_max) if stored_max is not None else 1.0)
        )
        out = _linear_map_array(
            vals,
            float(band_params["stretch_min"]),
            float(band_params["stretch_max"]),
            target_min,
            target_max,
        )
        mapped_native_range = True
    elif method == "zscore":
        out = (vals - float(band_params["mean"])) / float(band_params["scale"])
    elif method == "robust_zscore":
        out = (vals - float(band_params["median"])) / float(band_params["scale"])
    elif method == "decimal_scaling":
        out = vals / float(band_params["denom"])
    elif method == "log":
        shifted = np.maximum(vals + float(band_params["shift"]), 1e-12)
        out = np.log(shifted)
    elif method == "log1p":
        shifted = np.maximum(vals + float(band_params["shift"]), -1.0 + 1e-12)
        out = np.log1p(shifted)
    elif method == "sigmoid":
        z = (vals - float(band_params["mean"])) / float(band_params["scale"])
        out = 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))
    elif method == "tanh":
        z = (vals - float(band_params["mean"])) / float(band_params["scale"])
        out = np.tanh(z)
    elif method == "standard_scaler":
        out = vals.copy()
        if band_params.get("with_mean", True):
            out = out - float(band_params["mean"])
        if band_params.get("with_std", True):
            out = out / float(band_params["scale"])
    elif method == "minmax_scaler":
        out = vals * float(band_params["scale"]) + float(band_params["min_offset"])
        if eff_out_min is not None and eff_out_max is not None:
            saved_range = band_params.get("feature_range", [0.0, 1.0])
            if (
                float(saved_range[0]) != float(eff_out_min)
                or float(saved_range[1]) != float(eff_out_max)
            ):
                out = _linear_map_array(
                    out,
                    float(band_params["trans_min"]),
                    float(band_params["trans_max"]),
                    eff_out_min,
                    eff_out_max,
                )
        mapped_native_range = True
    elif method == "maxabs_scaler":
        out = vals / float(band_params["scale"])
    elif method == "robust_scaler":
        out = vals.copy()
        if band_params.get("with_centering", True):
            out = out - float(band_params["center"])
        if band_params.get("with_scaling", True):
            out = out / float(band_params["scale"])
    elif method == "power_transformer":
        rsgispy.check_optional_dep(
            HAVE_SKLEARN,
            "apply_img_norm_params(method='power_transformer')",
            "scikit-learn",
            "classification",
        )
        pt = sk_prep.PowerTransformer(
            method=str(band_params["pt_method"]),
            standardize=bool(band_params["standardize"]),
        )
        pt.lambdas_ = np.asarray(band_params["lambdas"], dtype=np.float64)
        pt.n_features_in_ = 1
        if pt.standardize:
            sc = sk_prep.StandardScaler()
            sc.mean_ = np.asarray(band_params["scaler_mean"], dtype=np.float64)
            sc.scale_ = np.asarray(band_params["scaler_scale"], dtype=np.float64)
            sc.var_ = np.asarray(band_params["scaler_var"], dtype=np.float64)
            sc.n_features_in_ = 1
            pt._scaler = sc  # pylint: disable=protected-access
        out = pt.transform(vals.reshape(-1, 1)).ravel()
    elif method == "quantile_transformer":
        rsgispy.check_optional_dep(
            HAVE_SKLEARN,
            "apply_img_norm_params(method='quantile_transformer')",
            "scikit-learn",
            "classification",
        )
        qt = sk_prep.QuantileTransformer(
            n_quantiles=int(band_params["n_quantiles"]),
            output_distribution=str(band_params["output_distribution"]),
        )
        qt.n_quantiles_ = int(band_params["n_quantiles"])
        qt.quantiles_ = np.asarray(band_params["quantiles"], dtype=np.float64)
        qt.references_ = np.asarray(band_params["references"], dtype=np.float64)
        qt.n_features_in_ = 1
        out = qt.transform(vals.reshape(-1, 1)).ravel()
    else:
        raise rsgispy.RSGISPyException(f"Unsupported normalisation method: {method!r}.")

    if (
        not mapped_native_range
        and eff_out_min is not None
        and eff_out_max is not None
    ):
        out = _linear_map_array(
            out,
            float(band_params["trans_min"]),
            float(band_params["trans_max"]),
            eff_out_min,
            eff_out_max,
        )

    if trim_to_limits and eff_out_min is not None and eff_out_max is not None:
        lo = min(float(eff_out_min), float(eff_out_max))
        hi = max(float(eff_out_min), float(eff_out_max))
        out = np.clip(out, lo, hi)
    return out


def compute_pixel_vector_norm_matrix(
    method: str,
    vecs: np.ndarray,
    norm_type: str,
) -> np.ndarray:
    """Apply cross-band per-pixel vector normalisation to ``(n_pxls, n_bands)``."""
    if method == "softmax":
        shifted = vecs - np.max(vecs, axis=1, keepdims=True)
        exp_v = np.exp(np.clip(shifted, -60.0, 60.0))
        return exp_v / np.sum(exp_v, axis=1, keepdims=True)
    if method == "normalizer":
        rsgispy.check_optional_dep(
            HAVE_SKLEARN,
            "apply_img_norm_params(method='Normalizer')",
            "scikit-learn",
            "classification",
        )
        return sk_prep.Normalizer(norm=norm_type).transform(vecs)
    if norm_type == "l1":
        denom = np.sum(np.abs(vecs), axis=1, keepdims=True)
        return np.where(denom > 0.0, vecs / np.where(denom > 0.0, denom, 1.0), 0.0)
    if norm_type == "max":
        denom = np.max(np.abs(vecs), axis=1, keepdims=True)
        return np.where(denom > 0.0, vecs / np.where(denom > 0.0, denom, 1.0), 0.0)
    # l2
    denom = np.linalg.norm(vecs, ord=2, axis=1, keepdims=True)
    return np.where(denom > 0.0, vecs / np.where(denom > 0.0, denom, 1.0), 0.0)


def apply_pixel_vector_norm(
    method: str,
    block: np.ndarray,
    valid_masks: List[np.ndarray],
    norm_type: str,
    eff_out_min: Optional[float],
    eff_out_max: Optional[float],
    trim_to_limits: bool,
    vec_min: Optional[float] = None,
    vec_max: Optional[float] = None,
) -> np.ndarray:
    """Apply cross-band per-pixel vector normalisation (L1, L2, Max, Normalizer, Softmax)."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    all_valid = valid_masks[0].copy()
    for m in valid_masks[1:]:
        all_valid &= m

    out_block = np.full(block.shape, np.nan, dtype=np.float64)
    if not np.any(all_valid):
        return out_block

    vecs = block[:, all_valid].T  # (n_pxls, n_bands)
    normed = compute_pixel_vector_norm_matrix(method, vecs, norm_type)

    if eff_out_min is not None and eff_out_max is not None:
        n_min = float(np.min(normed)) if vec_min is None else float(vec_min)
        n_max = float(np.max(normed)) if vec_max is None else float(vec_max)
        normed = _linear_map_array(normed, n_min, n_max, eff_out_min, eff_out_max)
        if trim_to_limits:
            lo = min(float(eff_out_min), float(eff_out_max))
            hi = max(float(eff_out_min), float(eff_out_max))
            normed = np.clip(normed, lo, hi)

    out_block[:, all_valid] = normed.T
    return out_block


def resolve_band_nodata(
    ds_nodatavals: Sequence[Optional[float]],
    band_idx: int,
    explicit_no_data: Optional[float],
) -> Optional[float]:
    """Resolve effective input no-data value for 1-based ``band_idx``."""
    if explicit_no_data is not None:
        return float(explicit_no_data)
    nd = ds_nodatavals[band_idx - 1]
    return float(nd) if nd is not None else None


def resolve_out_nodata(
    band_specs: Sequence[Mapping[str, Any]],
    out_no_data_val: Optional[float],
    out_np_dtype: np.dtype,
) -> Optional[float]:
    """Resolve output nodata value representable in ``out_np_dtype``."""
    first_nd = band_specs[0].get("no_data_val") if band_specs else None
    if out_no_data_val is None and first_nd is None:
        return None
    return _ip_resolve_out_nodata(
        out_np_dtype,
        out_no_data_val=out_no_data_val,
        fallback_nodata=first_nd,
        default_float=-9999.0,
        default_int=0.0,
        allow_nonfinite_float=True,
    )


def subsample_band_values(
    src: Any,
    band_idx: int,
    eff_nd: Optional[float],
    max_samples: Optional[int],
    rnd_seed: Optional[int],
    use_pbar: bool = False,
) -> np.ndarray:
    """Stream valid finite pixel values for 1-based ``band_idx`` with O(max_samples) RAM."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    # pylint: disable=import-outside-toplevel
    from rsgispy.imageprocess import iter_dataset_windows

    if max_samples is None:
        chunks: List[np.ndarray] = []
        for win in iter_dataset_windows(src.height, src.width, use_pbar=use_pbar):
            arr = src.read(band_idx, window=win, out_dtype=np.float64)
            valid = build_valid_mask(arr, eff_nd)
            if np.any(valid):
                chunks.append(arr[valid])
        if not chunks:
            return np.empty(0, dtype=np.float64)
        return chunks[0] if len(chunks) == 1 else np.concatenate(chunks)

    k = int(max_samples)
    if k < 1:
        raise rsgispy.RSGISPyException("max_samples must be >= 1.")

    reservoir = np.empty(k, dtype=np.float64)
    seen = 0
    rng = np.random.default_rng(rnd_seed)

    for win in iter_dataset_windows(src.height, src.width, use_pbar=use_pbar):
        arr = src.read(band_idx, window=win, out_dtype=np.float64)
        valid = build_valid_mask(arr, eff_nd)
        if not np.any(valid):
            continue
        vals = arr[valid]
        n_vals = int(vals.size)

        if seen < k:
            take = min(k - seen, n_vals)
            reservoir[seen : seen + take] = vals[:take]
            seen += take
            vals = vals[take:]
            n_vals = int(vals.size)

        if n_vals > 0:
            counts_arr = np.arange(seen + 1, seen + n_vals + 1, dtype=np.float64)
            js = np.floor(rng.random(n_vals) * counts_arr).astype(np.int64)
            hit = js < k
            if np.any(hit):
                for slot, val in zip(js[hit].tolist(), vals[hit].tolist()):
                    reservoir[slot] = val
            seen += n_vals

    if seen == 0:
        return np.empty(0, dtype=np.float64)
    if seen < k:
        return reservoir[:seen].copy()
    return reservoir
