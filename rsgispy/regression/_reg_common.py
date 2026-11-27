# pylint: disable=too-many-lines
"""
Shared internal configuration coercion, univariate/multivariate sample
preparation, regression metric calculation, block-wise raster regression, and
GBDT engine helpers reused across :mod:`rsgispy.regression`.
"""

from contextlib import ExitStack
from dataclasses import dataclass
import math
import os
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
import warnings

import numpy as np
import pandas as pd

import rsgispy
from rsgispy.imageprocess import (
    ensure_file_exists,
    iter_dataset_windows,
    open_output_dataset,
)
from rsgispy.imageprocess import normalise_creation_options
from rsgispy.imageutils.colourtable import set_img_not_thematic
from rsgispy.imageutils.masking import ImageBandInfo
from rsgispy.imageutils.stats import pop_img_stats
from rsgispy.regression.regdata import (
    RegressionConfig,
    normalise_target_cols,
    read_reg_config,
)
from rsgispy.tools._gbdt_common import (
    check_gbdt_backend_available as _check_reg_backend_available,
    merge_gbdt_params as _tools_merge_gbdt_params,
    prepared_gbdt_predict_estimator as _prepared_gbdt_predict_estimator,
    resolve_gbdt_gpu as _tools_resolve_gbdt_gpu,
    resolve_gbdt_optuna_space as _tools_resolve_gbdt_optuna_space,
    run_gbdt_optuna_optimisation as _run_gbdt_optuna_optimisation,
)
from rsgispy.tools._ml_data import (
    load_samples_df,
    open_apply_datasets as _open_apply_datasets,
    read_valid_roi_block as _read_valid_roi_block,
    resolve_apply_band_specs_from_seq,
    resolve_estimator_feature_cols as _resolve_estimator_feature_cols,
    resolve_feature_cols as _tools_resolve_feature_cols,
    seg_allows_nan,
    seg_model_features,
)
from rsgispy.tools._ml_tune import HAVE_OPTUNA
from rsgispy.tools._sklearn import (
    check_sklearn_available as _check_sklearn_available,
    check_sklearn_estimator_fitted,
    sklearn,
    sync_sklearn_estimator_state,
)
from rsgispy.tools.files import ensure_parent_dir
from rsgispy.tools.jsonio import write_dict_to_json

RegConfigInput = Union[
    RegressionConfig,
    str,
    os.PathLike,
    Sequence[ImageBandInfo],
]
TargetColsInput = Optional[Union[str, Sequence[str]]]
SamplesInput = Union[pd.DataFrame, str, os.PathLike]
_OptunaSpaceType = Union[Dict[str, Any], Callable[[Any], Mapping[str, Any]]]


def check_sklearn_available(func_name: str) -> None:
    """Verify ``scikit-learn`` is installed for regression workflows."""
    _check_sklearn_available(func_name, extra="regression")


def coerce_reg_config(
    reg_cfg: Optional[RegConfigInput],
    target_cols: TargetColsInput = None,
    allow_none: bool = False,
) -> Tuple[Optional[RegressionConfig], Optional[List[str]]]:
    """
    Normalise a :class:`RegressionConfig`, JSON path, or sequence of
    :class:`ImageBandInfo` into ``(cfg_obj_or_None, target_cols_or_None)``.
    """
    explicit_targets = (
        normalise_target_cols(target_cols)
        if target_cols is not None
        else None
    )
    if reg_cfg is None:
        if not allow_none and explicit_targets is None:
            raise rsgispy.RSGISPyException(
                "reg_cfg or target_cols must be provided."
            )
        return None, explicit_targets

    if isinstance(reg_cfg, RegressionConfig):
        eff_targets = (
            explicit_targets
            if explicit_targets is not None
            else list(reg_cfg.target_cols)
        )
        return reg_cfg, eff_targets

    if isinstance(reg_cfg, (str, os.PathLike)):
        cfg = read_reg_config(reg_cfg)
        eff_targets = (
            explicit_targets
            if explicit_targets is not None
            else list(cfg.target_cols)
        )
        return cfg, eff_targets

    if isinstance(reg_cfg, Sequence) and not isinstance(reg_cfg, bytes):
        if explicit_targets is None:
            if not allow_none:
                raise rsgispy.RSGISPyException(
                    "target_cols must be provided when reg_cfg is a sequence "
                    "of ImageBandInfo objects."
                )
            cfg = RegressionConfig(
                img_band_info=reg_cfg,
                target_cols=["target"],
            )
            return cfg, None
        cfg = RegressionConfig(
            img_band_info=reg_cfg,
            target_cols=explicit_targets,
        )
        return cfg, explicit_targets

    raise rsgispy.RSGISPyException(
        "reg_cfg must be a RegressionConfig instance, JSON config file path, "
        "or sequence of ImageBandInfo objects."
    )


def resolve_target_cols(
    cfg: Optional[RegressionConfig],
    df: pd.DataFrame,
    target_cols: TargetColsInput = None,
    require_target: bool = True,
) -> List[str]:
    """Resolve and validate target column name(s) present in ``df``."""
    if target_cols is not None:
        cols = normalise_target_cols(target_cols)
    elif cfg is not None:
        cols = list(cfg.target_cols)
    else:
        if not require_target:
            return []
        raise rsgispy.RSGISPyException(
            "target_cols must be specified when reg_cfg is omitted."
        )

    missing = [col for col in cols if col not in df.columns]
    if missing:
        if not require_target:
            return []
        raise rsgispy.RSGISPyException(
            f"Target column(s) {missing} missing from sample DataFrame."
        )
    return cols


def resolve_reg_feature_cols(
    cfg: Optional[RegressionConfig],
    df: pd.DataFrame,
    feature_cols: Optional[Sequence[str]],
    excluded_cols: Sequence[str],
) -> List[str]:
    """
    Determine and validate ordered feature columns for regression tables
    (for segment-based configs, the ``seg_info`` model features by default).
    """
    if feature_cols is None:
        feature_cols = seg_model_features(cfg)
    return _tools_resolve_feature_cols(
        cfg_img_band_info=cfg.img_band_info if cfg is not None else None,
        df=df,
        feature_cols=feature_cols,
        excluded_cols=excluded_cols,
    )


def extract_reg_xy(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_cols: Sequence[str],
    label: str,
    allow_nan: bool = False,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Extract feature DataFrame ``X`` and finite float target array ``y``
    (1D for univariate regression, 2D ``(n_samples, n_targets)`` for
    multivariate regression). NaN feature values are only accepted when
    ``allow_nan=True``.
    """
    if len(df) == 0:
        raise rsgispy.RSGISPyException(f"{label} sample table is empty.")

    missing_t = [col for col in target_cols if col not in df.columns]
    if missing_t:
        raise rsgispy.RSGISPyException(
            f"{label} is missing target column(s) {missing_t}."
        )
    missing_f = [col for col in feature_cols if col not in df.columns]
    if missing_f:
        raise rsgispy.RSGISPyException(
            f"{label} is missing feature column(s) {missing_f}."
        )

    x_df = df[list(feature_cols)].copy()
    x_vals = x_df.to_numpy(dtype=np.float64)
    bad_vals = np.isinf(x_vals) if allow_nan else ~np.isfinite(x_vals)
    if np.any(bad_vals):
        raise rsgispy.RSGISPyException(
            f"{label} contains NaN or infinite values in feature columns."
        )

    y_mat = df[list(target_cols)].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(y_mat)):
        raise rsgispy.RSGISPyException(
            f"{label} contains NaN or infinite values in target column(s) "
            f"{list(target_cols)}."
        )

    y_arr = y_mat[:, 0] if len(target_cols) == 1 else y_mat
    return x_df, y_arr


def prepare_reg_data(
    samples_data: SamplesInput,
    reg_cfg: Optional[RegConfigInput] = None,
    feature_cols: Optional[Sequence[str]] = None,
    target_cols: TargetColsInput = None,
    label: str = "samples_data",
    require_target: bool = True,
) -> Tuple[
    pd.DataFrame,
    List[str],
    List[str],
    pd.DataFrame,
    Optional[np.ndarray],
]:
    """Load sample DataFrame, resolve features and targets, and extract XY."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    cfg, cfg_targets = coerce_reg_config(
        reg_cfg, target_cols=target_cols, allow_none=not require_target
    )
    df = load_samples_df(samples_data, label)
    if len(df) == 0:
        raise rsgispy.RSGISPyException(f"{label} sample table is empty.")

    eff_targets = resolve_target_cols(
        cfg, df, target_cols=cfg_targets, require_target=require_target
    )
    feat_cols = resolve_reg_feature_cols(
        cfg,
        df,
        feature_cols,
        excluded_cols=eff_targets,
    )

    if not require_target:
        x_df = df[feat_cols].copy()
        x_vals = x_df.to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(x_vals)):
            raise rsgispy.RSGISPyException(
                f"{label} contains NaN or infinite values in feature columns."
            )
        return df, feat_cols, eff_targets, x_df, None

    x_df, y_arr = extract_reg_xy(
        df, feat_cols, eff_targets, label, allow_nan=seg_allows_nan(cfg)
    )
    return df, feat_cols, eff_targets, x_df, y_arr


def prepare_reg_splits(
    reg_cfg: Optional[RegConfigInput],
    train_data: SamplesInput,
    test_data: Optional[SamplesInput] = None,
    valid_data: Optional[SamplesInput] = None,
    feature_cols: Optional[Sequence[str]] = None,
    target_cols: TargetColsInput = None,
) -> Tuple[
    Optional[RegressionConfig],
    List[str],
    List[str],
    pd.DataFrame,
    np.ndarray,
    Optional[Tuple[pd.DataFrame, np.ndarray]],
    Optional[Tuple[pd.DataFrame, np.ndarray]],
]:
    """Load config, resolve feature/target columns, and extract splits."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    cfg, cfg_targets = coerce_reg_config(
        reg_cfg, target_cols=target_cols, allow_none=False
    )
    train_df = load_samples_df(train_data, "train_data")
    eff_targets = resolve_target_cols(
        cfg, train_df, target_cols=cfg_targets, require_target=True
    )
    feat_cols = resolve_reg_feature_cols(
        cfg, train_df, feature_cols, excluded_cols=eff_targets
    )
    train_x, train_y = extract_reg_xy(
        train_df, feat_cols, eff_targets, "train_data", allow_nan=seg_allows_nan(cfg)
    )

    test_xy: Optional[Tuple[pd.DataFrame, np.ndarray]] = None
    if test_data is not None:
        test_df = load_samples_df(test_data, "test_data")
        test_xy = extract_reg_xy(
            test_df, feat_cols, eff_targets, "test_data", allow_nan=seg_allows_nan(cfg)
        )

    valid_xy: Optional[Tuple[pd.DataFrame, np.ndarray]] = None
    if valid_data is not None:
        valid_df = load_samples_df(valid_data, "valid_data")
        valid_xy = extract_reg_xy(
            valid_df, feat_cols, eff_targets, "valid_data",
            allow_nan=seg_allows_nan(cfg),
        )

    return cfg, eff_targets, feat_cols, train_x, train_y, test_xy, valid_xy


def _safe_pearson_r(y_t: np.ndarray, y_p: np.ndarray) -> float:
    """Compute Pearson correlation coefficient handling constant arrays."""
    if y_t.size < 2:
        return 0.0
    std_t = float(np.std(y_t))
    std_p = float(np.std(y_p))
    if std_t == 0.0 or std_p == 0.0:
        return 0.0
    r_val = float(np.corrcoef(y_t, y_p)[0, 1])
    return 0.0 if math.isnan(r_val) else r_val


REG_BASE_METRIC_KEYS = (
    "r2",
    "rmse",
    "mae",
    "bias",
    "explained_variance",
    "pearson_r",
)
REG_EXTRA_METRIC_KEYS = ("mse", "nrmse", "nrmse_range", "nbias")


def _pct_of(value: float, denom: float) -> float:
    """Return ``100 * value / denom``, or ``nan`` when ``denom`` is zero."""
    if denom == 0.0 or not math.isfinite(denom):
        return float("nan")
    return float(100.0 * value / denom)


def _single_target_metrics(
    y_t: np.ndarray,
    y_p: np.ndarray,
    extra_metrics: bool = False,
) -> Dict[str, float]:
    """
    Compute scalar continuous regression metrics for one 1D target. With
    ``extra_metrics`` also return ``mse``, ``nrmse`` (RMSE as % of the mean
    observed value), ``nrmse_range`` (RMSE as % of the observed range) and
    ``nbias`` (bias as % of the mean observed value), each ``nan`` when its
    denominator is zero.
    """
    diff = y_p - y_t
    mse = float(sklearn.metrics.mean_squared_error(y_t, y_p))
    rmse = float(math.sqrt(mse))
    mae = float(sklearn.metrics.mean_absolute_error(y_t, y_p))
    bias = float(np.mean(diff))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r2 = (
            float(sklearn.metrics.r2_score(y_t, y_p))
            if y_t.size >= 2
            else 0.0
        )
        ev = (
            float(sklearn.metrics.explained_variance_score(y_t, y_p))
            if y_t.size >= 2
            else 0.0
        )
    if math.isnan(r2):
        r2 = 0.0
    if math.isnan(ev):
        ev = 0.0
    out = {
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "explained_variance": ev,
        "pearson_r": _safe_pearson_r(y_t, y_p),
    }
    if extra_metrics:
        mean_obs = float(np.mean(y_t))
        range_obs = float(np.max(y_t) - np.min(y_t))
        out["mse"] = mse
        out["nrmse"] = _pct_of(rmse, mean_obs)
        out["nrmse_range"] = _pct_of(rmse, range_obs)
        out["nbias"] = _pct_of(bias, mean_obs)
    return out


def compute_regression_metrics_dict(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_cols: Sequence[str],
    prefix: Optional[str] = None,
    extra_metrics: bool = False,
) -> Dict[str, Any]:
    """
    Compute univariate or multivariate continuous regression accuracy metrics
    (``r2``, ``rmse``, ``mae``, ``bias``, ``explained_variance``, and
    ``pearson_r``, plus per-target entries when multivariate). With
    ``extra_metrics`` the ``mse``, ``nrmse``, ``nrmse_range`` and ``nbias``
    keys are added in the same way (the headline values are the mean over
    the targets).
    """
    # pylint: disable=too-many-locals
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    if y_t.ndim == 1:
        y_t_2d = y_t[:, None]
    else:
        y_t_2d = y_t
    if y_p.ndim == 1:
        y_p_2d = y_p[:, None]
    else:
        y_p_2d = y_p

    if y_t_2d.shape != y_p_2d.shape:
        raise rsgispy.RSGISPyException(
            f"Prediction shape {y_p_2d.shape} does not match target shape "
            f"{y_t_2d.shape}."
        )

    t_list = list(target_cols)
    if len(t_list) != y_t_2d.shape[1]:
        raise rsgispy.RSGISPyException(
            f"Target array has {y_t_2d.shape[1]} column(s), expected "
            f"{len(t_list)} ({t_list})."
        )

    per_target = [
        _single_target_metrics(
            y_t_2d[:, idx], y_p_2d[:, idx], extra_metrics=extra_metrics
        )
        for idx in range(len(t_list))
    ]
    metric_keys = REG_BASE_METRIC_KEYS
    if extra_metrics:
        metric_keys = REG_BASE_METRIC_KEYS + REG_EXTRA_METRIC_KEYS

    out: Dict[str, Any] = {}
    key_prefix = f"{prefix}_" if prefix else ""
    for m_key in metric_keys:
        out[f"{key_prefix}{m_key}"] = float(
            np.mean([entry[m_key] for entry in per_target])
        )

    if len(t_list) > 1:
        for t_name, entry in zip(t_list, per_target):
            for m_key in metric_keys:
                out[f"{key_prefix}{t_name}_{m_key}"] = float(entry[m_key])
    return out


def calc_reg_split_metrics(
    fitted_est: Any,
    x_df: pd.DataFrame,
    y_true: np.ndarray,
    target_cols: Sequence[str],
    prefix: str,
    predict_kwargs: Optional[Mapping[str, Any]] = None,
) -> Dict[str, float]:
    """Predict on ``x_df`` and return prefixed regression split metrics."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    p_kwargs = dict(predict_kwargs) if predict_kwargs else {}
    preds = np.asarray(
        fitted_est.predict(x_df, **p_kwargs)
        if p_kwargs
        else fitted_est.predict(x_df),
        dtype=np.float64,
    )
    return compute_regression_metrics_dict(
        y_true=y_true,
        y_pred=preds,
        target_cols=target_cols,
        prefix=prefix,
    )


def apply_regressor_raster(
    reg_cfg: Optional[RegConfigInput],
    estimator: Any,
    in_msk_img: Union[str, os.PathLike],
    img_msk_val: int,
    output_img: Union[str, os.PathLike],
    img_band_info: Optional[Sequence[ImageBandInfo]] = None,
    target_cols: TargetColsInput = None,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    out_no_data_val: float = -9999.0,
    calc_stats: bool = True,
    block_size: Optional[int] = None,
    creation_opts: Optional[Mapping[str, Any]] = None,
    use_pbar: bool = True,
    predict_kwargs: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Apply a fitted univariate or multivariate regressor block-by-block over
    image pixels where ``in_msk_img == img_msk_val`` and write continuous
    predicted raster band(s).
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    p_kwargs = dict(predict_kwargs) if predict_kwargs else {}
    check_sklearn_estimator_fitted(estimator, label="regressor")

    cfg, cfg_targets = coerce_reg_config(
        reg_cfg, target_cols=target_cols, allow_none=True
    )
    band_specs = resolve_apply_band_specs_from_seq(
        img_band_info=img_band_info,
        cfg_img_band_info=cfg.img_band_info if cfg is not None else None,
        cfg_label="reg_cfg",
    )
    eff_targets = cfg_targets if cfg_targets else ["prediction"]
    n_targets = len(eff_targets)

    eff_rsgis_dtype = (
        rsgispy.TYPE_32FLOAT if datatype is None else int(datatype)
    )
    out_rio_dtype = rsgispy.get_rasterio_datatype(eff_rsgis_dtype)
    out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(eff_rsgis_dtype))

    msk_path = os.fspath(in_msk_img)
    ensure_file_exists(msk_path, label="Input mask image")
    c_opts = normalise_creation_options(creation_opts)
    use_df_input, df_cols, band_specs = _resolve_estimator_feature_cols(
        estimator, band_specs
    )
    out_path = ensure_parent_dir(output_img)

    with ExitStack() as stack:
        msk_ds, datasets, grid, all_compatible = _open_apply_datasets(
            stack, msk_path, band_specs
        )
        dst = open_output_dataset(
            stack=stack,
            output_img=out_path,
            gdalformat=gdalformat,
            out_rio_dtype=out_rio_dtype,
            grid=grid,
            count=n_targets,
            creation_opts=c_opts,
        )
        dst.nodata = float(out_no_data_val)
        for b_idx, t_name in enumerate(eff_targets, start=1):
            dst.set_band_description(b_idx, str(t_name))

        base_msk_col, base_msk_row = grid.offsets[msk_path]
        total_valid_pxls = 0
        for out_win in iter_dataset_windows(
            grid.height, grid.width, block_size=block_size, use_pbar=use_pbar
        ):
            win_h = int(out_win.height)
            win_w = int(out_win.width)
            valid_idx, x_valid = _read_valid_roi_block(
                msk_ds=msk_ds,
                msk_path=msk_path,
                img_msk_val=img_msk_val,
                band_specs=band_specs,
                datasets=datasets,
                grid=grid,
                out_win=out_win,
                all_compatible=all_compatible,
                use_df_input=use_df_input,
                df_cols=df_cols,
                base_msk_col=base_msk_col,
                base_msk_row=base_msk_row,
            )
            total_valid_pxls += int(valid_idx.size)

            out_flat = np.full(
                (n_targets, win_h * win_w),
                float(out_no_data_val),
                dtype=out_np_dtype,
            )
            if valid_idx.size > 0 and x_valid is not None:
                preds = np.asarray(
                    estimator.predict(x_valid, **p_kwargs)
                    if p_kwargs
                    else estimator.predict(x_valid),
                    dtype=out_np_dtype,
                )
                if preds.ndim == 1:
                    preds_2d = preds[:, None]
                else:
                    preds_2d = preds
                if preds_2d.shape[1] != n_targets:
                    raise rsgispy.RSGISPyException(
                        f"Regressor returned {preds_2d.shape[1]} target "
                        f"column(s), expected {n_targets} ({eff_targets})."
                    )
                out_flat[:, valid_idx] = preds_2d.T

            dst.write(
                out_flat.reshape((n_targets, win_h, win_w)),
                window=out_win,
            )

        if total_valid_pxls == 0:
            warnings.warn(
                f"No valid pixels were predicted (in_msk_img='{msk_path}', "
                f"img_msk_val={img_msk_val}). Output raster contains only "
                f"no-data values ({out_no_data_val}); check that img_msk_val "
                "is present in in_msk_img and that source bands are not "
                "masked out by band nodata metadata.",
                UserWarning,
                stacklevel=2,
            )

    set_img_not_thematic(out_path)
    if calc_stats:
        pop_img_stats(
            out_path,
            no_data_val=float(out_no_data_val),
            calc_pyramids=True,
            use_pbar=use_pbar,
        )


# ---------------------------------------------------------------------------
# Shared GBDT Regression Engine
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GBDTRegBackendSpec:
    """Backend descriptor abstracting GBDT regressors (XGBoost/LGBM/CatBoost)."""

    # pylint: disable=too-many-instance-attributes

    name: str
    package_name: str
    extra: str
    is_available: Callable[[], bool]
    gpu_available: Callable[[Optional[int]], bool]
    apply_gpu_params: Callable[[Dict[str, Any], Optional[int]], Dict[str, Any]]
    seed_param: str
    jobs_param: str
    default_params: Callable[[int, int], Dict[str, Any]]
    default_space: Callable[[], _OptunaSpaceType]
    extended_space: Callable[[], _OptunaSpaceType]
    build_estimator: Callable[[Dict[str, Any], Optional[Any], int], Any]
    fit_estimator: Callable[
        [
            Any,
            pd.DataFrame,
            np.ndarray,
            Optional[pd.DataFrame],
            Optional[np.ndarray],
            Optional[int],
        ],
        Any,
    ]
    load_model: Callable[[Union[str, os.PathLike]], Any]
    save_model: Callable[[Any, Union[str, os.PathLike]], None]
    prepare_predict_estimator: Optional[
        Callable[
            [Any, bool, Optional[int]],
            Tuple[Any, Dict[str, Any], Optional[Callable[[], None]]],
        ]
    ] = None
    get_error_detail: Optional[Callable[[], Optional[str]]] = None


def _merge_gbdt_reg_params(
    backend: GBDTRegBackendSpec,
    n_targets: int,
    base_regressor: Optional[Any],
    explicit_params: Optional[Mapping[str, Any]],
    rnd_seed: Optional[int],
    n_jobs: Optional[int],
    early_stopping_rounds: Optional[int],
    has_valid: bool,
    verbose: int,
    gpu_active: bool = False,
    gpu_device_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Merge default, base-estimator, trial, and GPU parameters."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    task_defaults = backend.default_params(n_targets, verbose)
    return _tools_merge_gbdt_params(
        backend_name=backend.name,
        task_defaults=task_defaults,
        seed_param=backend.seed_param,
        jobs_param=backend.jobs_param,
        apply_gpu_params=backend.apply_gpu_params,
        base_estimator=base_regressor,
        explicit_params=explicit_params,
        rnd_seed=rnd_seed,
        n_jobs=n_jobs,
        early_stopping_rounds=early_stopping_rounds,
        has_valid=has_valid,
        gpu_active=gpu_active,
        gpu_device_id=gpu_device_id,
    )


def _build_and_fit_gbdt_reg(
    backend: GBDTRegBackendSpec,
    params: Dict[str, Any],
    base_regressor: Optional[Any],
    n_targets: int,
    train_x: pd.DataFrame,
    train_y: np.ndarray,
    valid_x: Optional[pd.DataFrame],
    valid_y: Optional[np.ndarray],
    early_stopping_rounds: Optional[int],
) -> Any:
    """Build a fresh GBDT regressor and fit it on training data."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    est = backend.build_estimator(params, base_regressor, n_targets)
    eff_early = early_stopping_rounds if valid_x is not None else None
    return backend.fit_estimator(
        est, train_x, train_y, valid_x, valid_y, eff_early
    )


def optimise_gbdt_regressor(
    backend: GBDTRegBackendSpec,
    func_name: str,
    reg_cfg: Optional[RegConfigInput],
    train_data: SamplesInput,
    valid_data: Optional[SamplesInput] = None,
    target_cols: TargetColsInput = None,
    param_space: Optional[
        Union[str, Mapping[str, Any], Callable[[Any], Mapping[str, Any]]]
    ] = None,
    extended_param_space: bool = False,
    fixed_params: Optional[Mapping[str, Any]] = None,
    regressor: Optional[Any] = None,
    scoring: Union[str, Callable[..., float]] = "r2",
    cv: Union[int, Any] = 5,
    n_trials: int = 50,
    timeout: Optional[float] = None,
    direction: str = "maximize",
    early_stopping_rounds: Optional[int] = None,
    n_jobs: Optional[int] = None,
    rnd_seed: Optional[int] = None,
    feature_cols: Optional[Sequence[str]] = None,
    refit: bool = True,
    out_results_file: Optional[Union[str, os.PathLike]] = None,
    out_params_json: Optional[Union[str, os.PathLike]] = None,
    out_model_file: Optional[Union[str, os.PathLike]] = None,
    return_results: bool = False,
    verbose: int = 0,
    use_gpu: bool = False,
    gpu_device_id: Optional[int] = None,
    use_pbar: bool = True,
) -> Union[Any, Tuple[Any, Dict[str, Any]]]:
    """
    Optimise hyperparameters for a univariate or multivariate GBDT regressor
    using Optuna Bayesian optimisation (TPE sampler).
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    _check_reg_backend_available(backend, func_name)
    rsgispy.check_optional_dep(
        HAVE_OPTUNA, func_name, "optuna", backend.extra
    )
    check_sklearn_available(func_name)
    gpu_active, eff_gpu_id = _tools_resolve_gbdt_gpu(
        backend.name, backend.gpu_available, use_gpu, gpu_device_id
    )

    if int(n_trials) < 1:
        raise rsgispy.RSGISPyException("n_trials must be at least 1.")
    if early_stopping_rounds is not None and int(early_stopping_rounds) < 1:
        raise rsgispy.RSGISPyException(
            "early_stopping_rounds must be a positive integer."
        )

    _, eff_targets, _, train_x, train_y, _, valid_xy = prepare_reg_splits(
        reg_cfg=reg_cfg,
        train_data=train_data,
        test_data=None,
        valid_data=valid_data,
        feature_cols=feature_cols,
        target_cols=target_cols,
    )
    n_targets = len(eff_targets)
    valid_x = valid_xy[0] if valid_xy is not None else None
    valid_y = valid_xy[1] if valid_xy is not None else None

    trial_space, space_mode = _tools_resolve_gbdt_optuna_space(
        default_space_fn=backend.default_space,
        extended_space_fn=backend.extended_space,
        param_space=param_space,
        extended_param_space=extended_param_space,
        fixed_params=fixed_params,
    )

    def _merge(
        explicit: Mapping[str, Any],
        eff_jobs: Optional[int],
    ) -> Dict[str, Any]:
        return _merge_gbdt_reg_params(
            backend=backend,
            n_targets=n_targets,
            base_regressor=regressor,
            explicit_params=explicit,
            rnd_seed=rnd_seed,
            n_jobs=eff_jobs,
            early_stopping_rounds=early_stopping_rounds,
            has_valid=valid_x is not None,
            verbose=verbose,
            gpu_active=gpu_active,
            gpu_device_id=eff_gpu_id,
        )

    def _build_fit(params: Dict[str, Any]) -> Any:
        return _build_and_fit_gbdt_reg(
            backend=backend,
            params=params,
            base_regressor=regressor,
            n_targets=n_targets,
            train_x=train_x,
            train_y=train_y,
            valid_x=valid_x,
            valid_y=valid_y,
            early_stopping_rounds=early_stopping_rounds,
        )

    return _run_gbdt_optuna_optimisation(
        backend=backend,
        func_name=func_name,
        train_x=train_x,
        train_y=train_y,
        valid_x=valid_x,
        valid_y=valid_y,
        trial_space=trial_space,
        space_mode=space_mode,
        fixed_params=fixed_params,
        merge_params_fn=_merge,
        build_estimator_fn=lambda p: backend.build_estimator(
            p, regressor, n_targets
        ),
        build_and_fit_fn=_build_fit,
        scoring=scoring,
        cv=cv,
        stratified=False,
        n_trials=n_trials,
        timeout=timeout,
        direction=direction,
        n_jobs=n_jobs,
        rnd_seed=rnd_seed,
        refit=refit,
        verbose=verbose,
        use_gpu=use_gpu,
        gpu_active=gpu_active,
        use_pbar=use_pbar,
        out_results_file=out_results_file,
        out_params_json=out_params_json,
        out_model_file=out_model_file,
        return_results=return_results,
        extra_meta={
            "library": backend.name.lower(),
            "task": "multivariate" if n_targets > 1 else "univariate",
            "n_targets": n_targets,
            "use_gpu_requested": bool(use_gpu),
            "gpu_used": bool(gpu_active),
        },
    )


def train_gbdt_regressor(
    backend: GBDTRegBackendSpec,
    func_name: str,
    reg_cfg: Optional[RegConfigInput],
    train_data: SamplesInput,
    test_data: Optional[SamplesInput] = None,
    valid_data: Optional[SamplesInput] = None,
    target_cols: TargetColsInput = None,
    regressor: Optional[Any] = None,
    model_params: Optional[Mapping[str, Any]] = None,
    early_stopping_rounds: Optional[int] = None,
    feature_cols: Optional[Sequence[str]] = None,
    rnd_seed: Optional[int] = None,
    n_jobs: Optional[int] = None,
    out_model_file: Optional[Union[str, os.PathLike]] = None,
    out_metrics_file: Optional[Union[str, os.PathLike]] = None,
    return_model: bool = True,
    verbose: int = 0,
    use_gpu: bool = False,
    gpu_device_id: Optional[int] = None,
) -> Union[Dict[str, float], Tuple[Any, Dict[str, float]]]:
    """
    Train a univariate or multivariate GBDT regressor on sample tables and
    evaluate on optional test and validation datasets.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    _check_reg_backend_available(backend, func_name)
    check_sklearn_available(func_name)
    gpu_active, eff_gpu_id = _tools_resolve_gbdt_gpu(
        backend.name, backend.gpu_available, use_gpu, gpu_device_id
    )

    if early_stopping_rounds is not None and int(early_stopping_rounds) < 1:
        raise rsgispy.RSGISPyException(
            "early_stopping_rounds must be a positive integer."
        )

    _, eff_targets, _, train_x, train_y, test_xy, valid_xy = prepare_reg_splits(
        reg_cfg=reg_cfg,
        train_data=train_data,
        test_data=test_data,
        valid_data=valid_data,
        feature_cols=feature_cols,
        target_cols=target_cols,
    )
    n_targets = len(eff_targets)
    valid_x = valid_xy[0] if valid_xy is not None else None
    valid_y = valid_xy[1] if valid_xy is not None else None

    params = _merge_gbdt_reg_params(
        backend=backend,
        n_targets=n_targets,
        base_regressor=regressor,
        explicit_params=model_params,
        rnd_seed=rnd_seed,
        n_jobs=n_jobs,
        early_stopping_rounds=early_stopping_rounds,
        has_valid=valid_x is not None,
        verbose=verbose,
        gpu_active=gpu_active,
        gpu_device_id=eff_gpu_id,
    )
    fitted_est = _build_and_fit_gbdt_reg(
        backend=backend,
        params=params,
        base_regressor=regressor,
        n_targets=n_targets,
        train_x=train_x,
        train_y=train_y,
        valid_x=valid_x,
        valid_y=valid_y,
        early_stopping_rounds=early_stopping_rounds,
    )
    if not return_model and regressor is not None:
        sync_sklearn_estimator_state(regressor, fitted_est)

    with _prepared_gbdt_predict_estimator(
        backend, fitted_est, gpu_active, eff_gpu_id
    ) as (eval_est, predict_kwargs):
        metrics = calc_reg_split_metrics(
            eval_est,
            train_x,
            train_y,
            eff_targets,
            "train",
            predict_kwargs=predict_kwargs,
        )
        if test_xy is not None:
            metrics.update(
                calc_reg_split_metrics(
                    eval_est,
                    test_xy[0],
                    test_xy[1],
                    eff_targets,
                    "test",
                    predict_kwargs=predict_kwargs,
                )
            )
        if valid_xy is not None:
            metrics.update(
                calc_reg_split_metrics(
                    eval_est,
                    valid_xy[0],
                    valid_xy[1],
                    eff_targets,
                    "valid",
                    predict_kwargs=predict_kwargs,
                )
            )

    if out_model_file is not None:
        backend.save_model(fitted_est, out_model_file)

    if out_metrics_file is not None:
        write_dict_to_json(metrics, ensure_parent_dir(out_metrics_file))

    if return_model:
        return fitted_est, metrics
    return metrics


def apply_gbdt_regressor(
    backend: GBDTRegBackendSpec,
    func_name: str,
    reg_cfg: Optional[RegConfigInput],
    regressor: Union[Any, str, os.PathLike],
    in_msk_img: Union[str, os.PathLike],
    img_msk_val: int,
    output_img: Union[str, os.PathLike],
    img_band_info: Optional[Sequence[ImageBandInfo]] = None,
    target_cols: TargetColsInput = None,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    out_no_data_val: float = -9999.0,
    calc_stats: bool = True,
    block_size: Optional[int] = None,
    creation_opts: Optional[Mapping[str, Any]] = None,
    use_pbar: bool = True,
    use_gpu: bool = False,
    gpu_device_id: Optional[int] = None,
) -> None:
    """
    Apply a trained univariate or multivariate GBDT regressor block-by-block
    to raster imagery.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    _check_reg_backend_available(backend, func_name)
    check_sklearn_available(func_name)
    gpu_active, eff_gpu_id = _tools_resolve_gbdt_gpu(
        backend.name, backend.gpu_available, use_gpu, gpu_device_id
    )
    estimator = (
        backend.load_model(regressor)
        if isinstance(regressor, (str, os.PathLike))
        else regressor
    )
    with _prepared_gbdt_predict_estimator(
        backend, estimator, gpu_active, eff_gpu_id
    ) as (eval_est, predict_kwargs):
        apply_regressor_raster(
            reg_cfg=reg_cfg,
            estimator=eval_est,
            in_msk_img=in_msk_img,
            img_msk_val=img_msk_val,
            output_img=output_img,
            img_band_info=img_band_info,
            target_cols=target_cols,
            gdalformat=gdalformat,
            datatype=datatype,
            out_no_data_val=out_no_data_val,
            calc_stats=calc_stats,
            block_size=block_size,
            creation_opts=creation_opts,
            use_pbar=use_pbar,
            predict_kwargs=predict_kwargs,
        )


def apply_regressor_vector(
    reg_cfg: RegConfigInput,
    estimator: Any,
    out_vec_file: Union[str, os.PathLike],
    *,
    target_cols: TargetColsInput = None,
    seg_info: Optional[Any] = None,
    in_vec_file: Optional[Union[str, os.PathLike]] = None,
    in_vec_lyr: Optional[str] = None,
    geom_vec_file: Optional[Union[str, os.PathLike]] = None,
    geom_vec_lyr: Optional[str] = None,
    out_vec_lyr: Optional[str] = None,
    out_format: Optional[str] = None,
    out_compress: Optional[Union[str, bool]] = "snappy",
    copy_cols: Optional[Sequence[str]] = None,
    out_prefix: str = "pred_",
    out_quantiles: Optional[Sequence[float]] = None,
    out_no_data_val: float = np.nan,
    missing: Optional[str] = None,
    fill_value: Optional[float] = None,
    scaler: Optional[Any] = None,
    chunk_size: int = 100_000,
    predict_kwargs: Optional[Mapping[str, Any]] = None,
    model_file: Optional[Union[str, os.PathLike]] = None,
    overwrite: bool = False,
    use_pbar: bool = True,
) -> None:
    """
    Shared chunked engine applying a fitted (univariate or multi-target)
    regressor to a segment feature layer (or attribute table) and writing a
    vector layer with one ``<prefix><target>`` column per target, plus
    optional ``<prefix><target>_q<quantile>`` columns from the per-tree
    predictions of tree ensembles (e.g. random forests). Only the ID, valid
    flag and model feature columns are read for prediction.
    """
    # pylint: disable=too-many-arguments,too-many-locals,too-many-statements
    # pylint: disable=import-outside-toplevel
    from rsgispy.ml.segments import resolve_seg_info
    from rsgispy.ml.vector import (
        build_provenance,
        map_vector_chunks,
        model_input,
        prepare_feature_matrix,
    )

    p_kwargs = dict(predict_kwargs) if predict_kwargs else {}
    check_sklearn_estimator_fitted(estimator, label="regressor")
    cfg, cfg_targets = coerce_reg_config(reg_cfg, target_cols=target_cols)
    targets = list(cfg_targets or [])
    if not targets:
        raise rsgispy.RSGISPyException("No target columns are defined.")
    seg = resolve_seg_info(cfg, seg_info)
    q_list = [float(q) for q in (out_quantiles or [])]
    if q_list:
        if any(not 0.0 < q < 1.0 for q in q_list):
            raise rsgispy.RSGISPyException("out_quantiles must be in (0, 1).")
        if not hasattr(estimator, "estimators_") or len(targets) != 1:
            raise rsgispy.RSGISPyException(
                "out_quantiles requires a univariate tree ensemble exposing "
                "estimators_ (e.g. RandomForestRegressor, ExtraTreesRegressor)."
            )
    est_names = getattr(estimator, "feature_names_in_", None)
    feat_cols = (
        [str(c) for c in est_names] if est_names is not None else seg.model_features
    )
    eff_missing = seg.missing if missing is None else missing
    eff_fill = seg.fill_value if fill_value is None else float(fill_value)
    if in_vec_file is None:
        in_path, in_lyr = seg.feature_source
        if seg.feat_table_file is not None and geom_vec_file is None:
            geom_vec_file, geom_vec_lyr = seg.seg_vec_file, seg.seg_vec_lyr
    else:
        in_path, in_lyr = os.fspath(in_vec_file), in_vec_lyr
    out_names = [f"{out_prefix}{t}" for t in targets]
    q_names = [f"{out_prefix}{targets[0]}_q{q:g}" for q in q_list]

    def _chunk(df: pd.DataFrame) -> pd.DataFrame:
        x_df, row_ok = prepare_feature_matrix(
            df, feat_cols, seg.valid_col, eff_missing, eff_fill,
            seg.feat_no_data_val, scaler,
        )
        out = np.full((len(df), len(out_names)), out_no_data_val, dtype=np.float64)
        q_out = np.full((len(df), len(q_names)), out_no_data_val, dtype=np.float64)
        if np.any(row_ok):
            x_in = model_input(estimator, x_df)
            pred = np.asarray(estimator.predict(x_in, **p_kwargs), dtype=np.float64)
            pred = pred.reshape(pred.shape[0], -1)
            if pred.shape[1] != len(out_names):
                raise rsgispy.RSGISPyException(
                    f"The regressor returned {pred.shape[1]} outputs for "
                    f"{len(out_names)} target(s)."
                )
            out[row_ok] = pred
            if q_list:
                x_arr = np.asarray(x_df.to_numpy())
                tree_preds = np.stack(
                    [np.asarray(t.predict(x_arr)).ravel() for t in estimator.estimators_]
                )
                q_out[row_ok] = np.quantile(tree_preds, q_list, axis=0).T
        res = pd.DataFrame(out, columns=out_names)
        for q_idx, q_name in enumerate(q_names):
            res[q_name] = q_out[:, q_idx]
        return res

    meta = build_provenance(
        model=estimator,
        model_file=model_file,
        config_file=reg_cfg if isinstance(reg_cfg, (str, os.PathLike)) else None,
        feature_cols=feat_cols,
        extra={"task": "regression", "targets": targets},
    )
    map_vector_chunks(
        in_path,
        out_vec_file,
        list(dict.fromkeys([seg.valid_col] + feat_cols)),
        _chunk,
        in_vec_lyr=in_lyr,
        geom_vec_file=geom_vec_file,
        geom_vec_lyr=geom_vec_lyr,
        id_col=seg.id_col,
        copy_cols=copy_cols,
        chunk_size=chunk_size,
        out_vec_lyr=out_vec_lyr,
        out_format=out_format,
        out_compress=out_compress,
        metadata=meta,
        overwrite=overwrite,
        use_pbar=use_pbar,
    )


def apply_gbdt_regressor_segments(
    backend: GBDTRegBackendSpec,
    func_name: str,
    reg_cfg: RegConfigInput,
    regressor: Union[Any, str, os.PathLike],
    out_vec_file: Union[str, os.PathLike],
    *,
    use_gpu: bool = False,
    gpu_device_id: Optional[int] = None,
    **kwargs: Any,
) -> None:
    """
    Apply a trained GBDT regressor to a segment feature layer in chunks (see
    :func:`apply_regressor_vector`).
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    _check_reg_backend_available(backend, func_name)
    check_sklearn_available(func_name)
    gpu_active, eff_gpu_id = _tools_resolve_gbdt_gpu(
        backend.name, backend.gpu_available, use_gpu, gpu_device_id
    )
    is_path = isinstance(regressor, (str, os.PathLike))
    estimator = backend.load_model(regressor) if is_path else regressor
    if is_path and kwargs.get("model_file") is None:
        kwargs["model_file"] = regressor
    with _prepared_gbdt_predict_estimator(
        backend, estimator, gpu_active, eff_gpu_id
    ) as (eval_est, predict_kwargs):
        apply_regressor_vector(
            reg_cfg, eval_est, out_vec_file, predict_kwargs=predict_kwargs, **kwargs
        )
