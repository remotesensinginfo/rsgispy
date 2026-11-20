"""
Unified feature scaling and normalisation transformer
(:class:`FeatureScaler`) wrapping all 23 scaling/normalisation methods from
:mod:`rsgispy.imagecalc.rescale` for fold-local cross-validation and on-the-fly
tiled raster prediction.
"""

import json
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import rasterio

import rsgispy
from rsgispy.imagecalc._norm_helpers import (
    ALL_NORM_METHODS,
    DEFAULT_TRIM_METHODS,
    PIXEL_VECTOR_METHODS,
    _extract_user_bounds,
    apply_band_norm_1d,
    compute_pixel_vector_norm_matrix,
    fit_band_norm_params,
    resolve_norm_method,
    resolve_out_bounds,
    subsample_band_values,
)
from rsgispy.tools.files import ensure_parent_dir

__all__ = [
    "ALL_NORM_METHODS",
    "FeatureScaler",
]


def _normalise_array_input(
    x_data: Union[np.ndarray, pd.DataFrame],
) -> Tuple[np.ndarray, Optional[List[str]], Optional[pd.Index]]:
    """Coerce 2D DataFrame or array into ``(float64_2d, col_names, index)``."""
    if isinstance(x_data, pd.DataFrame):
        arr = x_data.to_numpy(dtype=np.float64, copy=False)
        return arr, [str(c) for c in x_data.columns], x_data.index
    arr = np.asarray(x_data, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    if arr.ndim != 2:
        raise rsgispy.RSGISPyException(
            f"FeatureScaler expects a 2D array or DataFrame, got shape {arr.shape}."
        )
    return arr, None, None


class FeatureScaler:
    """
    Scikit-learn compatible, JSON-serialisable feature scaler wrapping all 23
    scaling and normalisation algorithms in :mod:`rsgispy.imagecalc.rescale`.

    Supports both tabular ``fit(X)`` / ``transform(X)`` (safe for fold-local
    cross-validation and ``sklearn.pipeline.Pipeline``) and direct raster-stack
    fitting via :meth:`fit_from_stack`, as well as on-the-fly per-tile
    transformation during :mod:`rsgispy.ml.predict`.

    :param method: Scaling/normalisation algorithm name or ``IMG_STRETCH_*``
                   constant (Default = ``'zscore'``). Supported methods include
                   ``'zscore'``, ``'robust_zscore'``, ``'linear'``,
                   ``'stdev'``, ``'cumulative'``, ``'standard_scaler'``,
                   ``'minmax_scaler'``, ``'maxabs_scaler'``, ``'robust_scaler'``,
                   ``'power_transformer'``, ``'quantile_transformer'``,
                   ``'log'``, ``'log1p'``, ``'sigmoid'``, ``'tanh'``,
                   ``'decimal_scaling'``, ``'l1'``, ``'l2'``, ``'max_norm'``,
                   ``'softmax'``, ``'normalizer'``, ``'user'``, and ``'none'``.
    :param out_min: Optional target lower bound for range-mapped methods.
    :param out_max: Optional target upper bound for range-mapped methods.
    :param trim_to_limits: Clip transformed values to ``[out_min, out_max]``
                           when ``True`` (``None`` uses method default).
    :param n_stdevs: Number of standard deviations for ``'stdev'`` stretch
                     (Default = ``2.0``).
    :param lower_pct: Lower percentile for ``'cumulative'`` stretch
                      (Default = ``2.0``).
    :param upper_pct: Upper percentile for ``'cumulative'`` stretch
                      (Default = ``98.0``).
    :param in_min_max: Manual ``{'min': ..., 'max': ...}`` mapping or sequence
                       for ``method='user'``.
    :param method_kwargs: Optional extra keyword arguments passed to underlying
                          scikit-learn scalers/transformers.
    :param rnd_seed: Optional random seed for subsampling / quantile fitting
                     (Default = ``42``).

    .. code:: python

        import numpy as np
        from rsgispy.ml.scaling import FeatureScaler

        scaler = FeatureScaler(method="robust_zscore")
        X_scaled = scaler.fit_transform(np.random.default_rng(0).normal(size=(100, 4)))
        scaler.save("scaler_params.json")
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        method: Union[str, int] = "zscore",
        out_min: Optional[float] = None,
        out_max: Optional[float] = None,
        trim_to_limits: Optional[bool] = None,
        n_stdevs: float = 2.0,
        lower_pct: float = 2.0,
        upper_pct: float = 98.0,
        in_min_max: Optional[
            Union[Mapping[str, float], Sequence[Mapping[str, float]]]
        ] = None,
        method_kwargs: Optional[Mapping[str, Any]] = None,
        rnd_seed: Optional[int] = 42,
    ) -> None:
        self.method = method
        self.out_min = out_min
        self.out_max = out_max
        self.trim_to_limits = trim_to_limits
        self.n_stdevs = float(n_stdevs)
        self.lower_pct = float(lower_pct)
        self.upper_pct = float(upper_pct)
        self.in_min_max = in_min_max
        self.method_kwargs = dict(method_kwargs) if method_kwargs else None
        self.rnd_seed = rnd_seed

        # Fitted state
        self.canonical_method_: Optional[str] = None
        self.eff_out_min_: Optional[float] = None
        self.eff_out_max_: Optional[float] = None
        self.eff_trim_: bool = False
        self.n_features_in_: Optional[int] = None
        self.feature_names_in_: Optional[List[str]] = None
        self.band_params_: Optional[List[Dict[str, Any]]] = None
        self.vec_min_: Optional[float] = None
        self.vec_max_: Optional[float] = None
        self.norm_type_: str = "l2"

    @property
    def is_fitted(self) -> bool:
        """Return ``True`` if :meth:`fit` or :meth:`fit_from_stack` has been called."""
        return self.canonical_method_ is not None and self.n_features_in_ is not None

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Return estimator hyperparameters for scikit-learn compatibility."""
        _ = deep
        return {
            "method": self.method,
            "out_min": self.out_min,
            "out_max": self.out_max,
            "trim_to_limits": self.trim_to_limits,
            "n_stdevs": self.n_stdevs,
            "lower_pct": self.lower_pct,
            "upper_pct": self.upper_pct,
            "in_min_max": self.in_min_max,
            "method_kwargs": (
                dict(self.method_kwargs) if self.method_kwargs is not None else None
            ),
            "rnd_seed": self.rnd_seed,
        }

    def set_params(self, **params: Any) -> "FeatureScaler":
        """Set estimator hyperparameters for scikit-learn compatibility."""
        valid = set(self.get_params(deep=False).keys())
        for key, val in params.items():
            if key not in valid:
                raise ValueError(
                    f"Invalid parameter '{key}' for FeatureScaler."
                )
            setattr(self, key, val)
        return self

    def clone_unfitted(self) -> "FeatureScaler":
        """Return a fresh, unfitted copy with identical configuration."""
        return FeatureScaler(**self.get_params(deep=True))

    def _resolve_config(self) -> Tuple[str, Optional[float], Optional[float], bool]:
        canonical = resolve_norm_method(self.method)
        eff_min, eff_max = resolve_out_bounds(canonical, self.out_min, self.out_max)
        eff_trim = (
            canonical in DEFAULT_TRIM_METHODS
            if self.trim_to_limits is None
            else bool(self.trim_to_limits)
        )
        return canonical, eff_min, eff_max, eff_trim

    def _resolve_norm_type(self, canonical: str) -> str:
        if canonical == "normalizer":
            kw = self.method_kwargs or {}
            return str(kw.get("norm", "l2")).lower()
        if canonical == "max_norm":
            return "max"
        if canonical == "l1":
            return "l1"
        return "l2"

    def fit(
        self,
        x_data: Union[np.ndarray, pd.DataFrame],
        y: Optional[Any] = None,
    ) -> "FeatureScaler":
        """
        Fit scaling parameters on a 2D feature matrix or DataFrame ``x_data``.

        :param x_data: 2D array or DataFrame of shape ``(n_samples, n_features)``.
        :param y: Unused target vector (accepted for scikit-learn compatibility).
        :return: ``self``.
        """
        _ = y
        arr, col_names, _ = _normalise_array_input(x_data)
        if arr.shape[0] == 0 or arr.shape[1] == 0:
            raise rsgispy.RSGISPyException(
                "FeatureScaler.fit requires a non-empty 2D feature matrix."
            )

        canonical, eff_min, eff_max, eff_trim = self._resolve_config()
        n_feats = int(arr.shape[1])
        self.canonical_method_ = canonical
        self.eff_out_min_ = eff_min
        self.eff_out_max_ = eff_max
        self.eff_trim_ = eff_trim
        self.n_features_in_ = n_feats
        self.feature_names_in_ = col_names
        self.norm_type_ = self._resolve_norm_type(canonical)

        if canonical in PIXEL_VECTOR_METHODS:
            self.band_params_ = None
            finite_rows = np.all(np.isfinite(arr), axis=1)
            if not np.any(finite_rows):
                raise rsgispy.RSGISPyException(
                    "No finite feature rows found when fitting FeatureScaler."
                )
            normed = compute_pixel_vector_norm_matrix(
                canonical, arr[finite_rows], self.norm_type_
            )
            self.vec_min_ = float(np.min(normed))
            self.vec_max_ = float(np.max(normed))
            return self

        params_list: List[Dict[str, Any]] = []
        for col_idx in range(n_feats):
            col_vals = arr[:, col_idx]
            valid_vals = col_vals[np.isfinite(col_vals)]
            if valid_vals.size == 0:
                raise rsgispy.RSGISPyException(
                    f"Feature column {col_idx} contains no finite values."
                )
            user_bounds = (
                _extract_user_bounds(self.in_min_max, col_idx, n_feats)
                if canonical == "user"
                else None
            )
            b_params = fit_band_norm_params(
                method=canonical,
                vals=valid_vals,
                band_idx=col_idx + 1,
                eff_nd=None,
                eff_out_min=eff_min,
                eff_out_max=eff_max,
                lower=self.lower_pct,
                upper=self.upper_pct,
                n_stdevs=self.n_stdevs,
                user_bounds=user_bounds,
                method_kwargs=self.method_kwargs,
                rnd_seed=self.rnd_seed,
            )
            params_list.append(b_params)
        self.band_params_ = params_list
        self.vec_min_ = None
        self.vec_max_ = None
        return self

    def fit_from_stack(
        self,
        feature_stack: Any,
        max_samples: Optional[int] = 500_000,
        use_pbar: bool = True,
    ) -> "FeatureScaler":
        """
        Fit scaling parameters directly by streaming valid pixels from an
        :class:`~rsgispy.ml.stack.ImageFeatureStack`.

        :param feature_stack: :class:`~rsgispy.ml.stack.ImageFeatureStack`
                              defining the raster feature bands.
        :param max_samples: Optional maximum number of valid pixels sampled per
                            band (Default = ``500_000``).
        :param use_pbar: Display a progress bar while reading raster blocks
                         (Default = ``True``).
        :return: ``self``.
        """
        # pylint: disable=too-many-locals
        canonical, eff_min, eff_max, eff_trim = self._resolve_config()
        band_specs = feature_stack.resolve_band_specs()
        n_feats = len(band_specs)
        feat_names = [name for _, _, name in band_specs]

        self.canonical_method_ = canonical
        self.eff_out_min_ = eff_min
        self.eff_out_max_ = eff_max
        self.eff_trim_ = eff_trim
        self.n_features_in_ = n_feats
        self.feature_names_in_ = feat_names
        self.norm_type_ = self._resolve_norm_type(canonical)

        if canonical in PIXEL_VECTOR_METHODS:
            chunks: List[np.ndarray] = []
            for _, _, _, x_blk in feature_stack.iter_blocks(
                apply_scaler=False, as_dataframe=False, use_pbar=use_pbar
            ):
                if x_blk.shape[0] > 0:
                    chunks.append(x_blk)
            if not chunks:
                raise rsgispy.RSGISPyException(
                    "No valid pixels found in feature_stack when fitting FeatureScaler."
                )
            full_x = chunks[0] if len(chunks) == 1 else np.vstack(chunks)
            if max_samples is not None and full_x.shape[0] > int(max_samples):
                rng = np.random.default_rng(self.rnd_seed)
                sel = rng.choice(
                    full_x.shape[0], size=int(max_samples), replace=False
                )
                full_x = full_x[sel]
            normed = compute_pixel_vector_norm_matrix(
                canonical, full_x, self.norm_type_
            )
            self.band_params_ = None
            self.vec_min_ = float(np.min(normed))
            self.vec_max_ = float(np.max(normed))
            return self

        band_nd_map = feature_stack.resolve_band_nodata_list()
        params_list: List[Dict[str, Any]] = []
        for idx, ((img_path, b_idx, _), eff_nd) in enumerate(
            zip(band_specs, band_nd_map)
        ):
            with rasterio.open(img_path, "r") as src:
                vals = subsample_band_values(
                    src=src,
                    band_idx=b_idx,
                    eff_nd=eff_nd,
                    max_samples=max_samples,
                    rnd_seed=self.rnd_seed,
                    use_pbar=use_pbar,
                )
            if vals.size == 0:
                raise rsgispy.RSGISPyException(
                    f"No valid pixels found in '{img_path}' band {b_idx}."
                )
            user_bounds = (
                _extract_user_bounds(self.in_min_max, idx, n_feats)
                if canonical == "user"
                else None
            )
            b_params = fit_band_norm_params(
                method=canonical,
                vals=vals,
                band_idx=idx + 1,
                eff_nd=eff_nd,
                eff_out_min=eff_min,
                eff_out_max=eff_max,
                lower=self.lower_pct,
                upper=self.upper_pct,
                n_stdevs=self.n_stdevs,
                user_bounds=user_bounds,
                method_kwargs=self.method_kwargs,
                rnd_seed=self.rnd_seed,
            )
            params_list.append(b_params)
        self.band_params_ = params_list
        self.vec_min_ = None
        self.vec_max_ = None
        return self

    def transform(
        self,
        x_data: Union[np.ndarray, pd.DataFrame],
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Transform a 2D feature array or DataFrame using fitted parameters.

        :param x_data: 2D array or DataFrame of shape ``(n_samples, n_features)``.
        :return: Scaled array (or DataFrame if ``x_data`` was a DataFrame).
        """
        if not self.is_fitted:
            raise rsgispy.RSGISPyException(
                "FeatureScaler must be fitted before calling transform()."
            )
        arr, col_names, df_index = _normalise_array_input(x_data)
        assert self.n_features_in_ is not None
        assert self.canonical_method_ is not None
        if arr.shape[1] != self.n_features_in_:
            raise rsgispy.RSGISPyException(
                f"FeatureScaler expected {self.n_features_in_} features, "
                f"got {arr.shape[1]}."
            )

        out = np.full_like(arr, np.nan, dtype=np.float64)
        if arr.shape[0] == 0:
            if col_names is not None:
                return pd.DataFrame(out, columns=col_names, index=df_index)
            return out

        if self.canonical_method_ in PIXEL_VECTOR_METHODS:
            finite_rows = np.all(np.isfinite(arr), axis=1)
            if np.any(finite_rows):
                normed = compute_pixel_vector_norm_matrix(
                    self.canonical_method_, arr[finite_rows], self.norm_type_
                )
                if (
                    self.eff_out_min_ is not None
                    and self.eff_out_max_ is not None
                    and self.vec_min_ is not None
                    and self.vec_max_ is not None
                ):
                    span = self.vec_max_ - self.vec_min_
                    if span == 0.0:
                        normed = np.full_like(normed, float(self.eff_out_min_))
                    else:
                        normed = float(self.eff_out_min_) + (
                            (normed - self.vec_min_) / span
                        ) * (float(self.eff_out_max_) - float(self.eff_out_min_))
                    if self.eff_trim_:
                        lo = min(float(self.eff_out_min_), float(self.eff_out_max_))
                        hi = max(float(self.eff_out_min_), float(self.eff_out_max_))
                        normed = np.clip(normed, lo, hi)
                out[finite_rows] = normed
        else:
            assert self.band_params_ is not None
            for col_idx in range(self.n_features_in_):
                col = arr[:, col_idx]
                fin = np.isfinite(col)
                if np.any(fin):
                    out[fin, col_idx] = apply_band_norm_1d(
                        method=self.canonical_method_,
                        vals=col[fin],
                        band_params=self.band_params_[col_idx],
                        eff_out_min=self.eff_out_min_,
                        eff_out_max=self.eff_out_max_,
                        trim_to_limits=self.eff_trim_,
                    )

        if col_names is not None:
            return pd.DataFrame(out, columns=col_names, index=df_index)
        return out

    def fit_transform(
        self,
        x_data: Union[np.ndarray, pd.DataFrame],
        y: Optional[Any] = None,
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Fit on ``x_data`` and return the transformed array or DataFrame."""
        return self.fit(x_data, y=y).transform(x_data)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise configuration and fitted parameters to a JSON-safe dict."""
        return {
            "version": 1,
            "config": self.get_params(deep=True),
            "fitted": self.is_fitted,
            "canonical_method": self.canonical_method_,
            "eff_out_min": self.eff_out_min_,
            "eff_out_max": self.eff_out_max_,
            "eff_trim": self.eff_trim_,
            "n_features_in": self.n_features_in_,
            "feature_names_in": (
                list(self.feature_names_in_)
                if self.feature_names_in_ is not None
                else None
            ),
            "band_params": self.band_params_,
            "vec_min": self.vec_min_,
            "vec_max": self.vec_max_,
            "norm_type": self.norm_type_,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FeatureScaler":
        """Reconstruct a :class:`FeatureScaler` from a serialised dictionary."""
        if not isinstance(data, Mapping) or "config" not in data:
            raise rsgispy.RSGISPyException(
                "Invalid FeatureScaler dictionary: missing 'config'."
            )
        obj = cls(**dict(data["config"]))
        if data.get("fitted", False):
            obj.canonical_method_ = data.get("canonical_method")
            obj.eff_out_min_ = data.get("eff_out_min")
            obj.eff_out_max_ = data.get("eff_out_max")
            obj.eff_trim_ = bool(data.get("eff_trim", False))
            obj.n_features_in_ = (
                int(data["n_features_in"])
                if data.get("n_features_in") is not None
                else None
            )
            fnames = data.get("feature_names_in")
            obj.feature_names_in_ = [str(c) for c in fnames] if fnames else None
            obj.band_params_ = data.get("band_params")
            obj.vec_min_ = data.get("vec_min")
            obj.vec_max_ = data.get("vec_max")
            obj.norm_type_ = str(data.get("norm_type", "l2"))
        return obj

    def save(self, json_file: Union[str, os.PathLike]) -> None:
        """Save scaler configuration and fitted parameters to a JSON file."""
        out_path = os.fspath(json_file)
        ensure_parent_dir(out_path)
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)

    @classmethod
    def load(cls, json_file: Union[str, os.PathLike]) -> "FeatureScaler":
        """Load a :class:`FeatureScaler` from a JSON file created by :meth:`save`."""
        in_path = os.fspath(json_file)
        if not os.path.exists(in_path):
            raise rsgispy.RSGISPyException(
                f"FeatureScaler JSON file does not exist: '{in_path}'."
            )
        with open(in_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls.from_dict(data)
