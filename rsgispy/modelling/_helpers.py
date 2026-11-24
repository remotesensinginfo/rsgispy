"""
Shared internal and reusable modelling helpers for :mod:`rsgispy.modelling`.

Provides environmental covariate descriptors (:class:`EnvVarInfo`), vector and
table preparation helpers, isolated ``scikit-learn`` estimator cloning/fitting
utilities, safe matplotlib figure lifecycle management, and a streaming
:mod:`rsgispy.imageprocess` block-prediction engine shared across single-model
and ensemble raster inference functions.
"""

from contextlib import ExitStack
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

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import rsgispy
from rsgispy.imageprocess import (
    BandDefn,
    cast_result_block,
    compute_spatial_intersection,
    extract_roi_feature_matrix,
    iter_image_blocks,
    open_input_datasets,
    open_output_dataset,
)
from rsgispy.imageutils.stats import pop_img_stats, pop_thmt_img_stats
from rsgispy.tools._ml_explain import finalise_plot
from rsgispy.tools._sklearn import (
    HAVE_SKLEARN,
    check_sklearn_available as _check_sklearn_available,
    clone_sklearn_estimator as clone_estimator,
    fit_clean_sklearn_estimator as fit_clean_estimator,
    sklearn,
    sync_sklearn_estimator_state as sync_estimator_state,
)
from rsgispy.tools.files import ensure_parent_dir
from rsgispy.vectorprocess import (
    VectorInput,
    finish_vec_output,
    read_vec_gdf,
    validate_columns_exist,
)

__all__ = [
    "HAVE_SKLEARN",
    "EnvVarInfo",
    "build_env_block_matrix",
    "calc_binary_roc_auc",
    "calc_single_class_predictions",
    "check_sklearn_available",
    "clone_estimator",
    "env_vars_to_band_defs",
    "extract_features_and_target",
    "finalise_plot",
    "fit_clean_estimator",
    "get_positive_class_probs",
    "lookup_env_var",
    "normalise_array_with_info",
    "plot_roc_curve",
    "predict_raster_from_env_vars",
    "read_model_vector",
    "resolve_analysis_vars",
    "run_env_block_pipeline",
    "sync_estimator_state",
    "write_model_vector",
]

_MASK_BAND_KEY = "__rsgispy_modelling_mask__"


class EnvVarInfo:
    """
    Descriptor storing metadata and normalisation coefficients for an
    environmental covariate band.

    :param name: Variable/column name for the covariate.
    :param file: Path to the raster file representing the covariate.
    :param band: 1-based band index within ``file`` (Default = ``1``).
    :param data_type: Variable type constant (``rsgispy.VAR_TYPE_CONTINUOUS``,
                      ``rsgispy.VAR_TYPE_CATEGORICAL``, or
                      ``rsgispy.VAR_TYPE_UNDEFINED``).
    :param min_vld_val: Optional inclusive lower valid threshold.
    :param max_vld_val: Optional inclusive upper valid threshold.
    :param norm_mean: Optional mean value for z-score normalisation.
    :param norm_std: Optional standard deviation for z-score normalisation.
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        name: Optional[str] = None,
        file: Optional[str] = None,
        band: int = 1,
        data_type: int = rsgispy.VAR_TYPE_UNDEFINED,
        min_vld_val: Optional[float] = None,
        max_vld_val: Optional[float] = None,
        norm_mean: Optional[float] = None,
        norm_std: Optional[float] = None,
    ):
        self.name = name
        self.file = file
        self.band = int(band)
        self.data_type = int(data_type)
        self.min_vld_val = min_vld_val
        self.max_vld_val = max_vld_val
        self.norm_mean = norm_mean
        self.norm_std = norm_std

    def set_norm_mean(self, norm_mean: float) -> None:
        """Set the normalisation mean value."""
        self.norm_mean = float(norm_mean)

    def set_norm_std(self, norm_std: float) -> None:
        """Set the normalisation standard deviation value."""
        self.norm_std = float(norm_std)

    def set_norm_mean_std(self, norm_mean: float, norm_std: float) -> None:
        """Set both normalisation mean and standard deviation values."""
        self.norm_mean = float(norm_mean)
        self.norm_std = float(norm_std)

    def to_band_spec(self) -> Any:
        """Convert this environmental variable into a :class:`~rsgispy.ml.stack.BandSpec`."""
        # pylint: disable=import-outside-toplevel
        from rsgispy.ml.stack import BandSpec

        return BandSpec.from_env_var_info(self)

    def _format_desc(self) -> str:
        type_str = "Unknown"
        if self.data_type == rsgispy.VAR_TYPE_CONTINUOUS:
            type_str = "Continuous"
        elif self.data_type == rsgispy.VAR_TYPE_CATEGORICAL:
            type_str = "Categorical"
        return (
            f"{self.name}: {self.file} (Band: {self.band}) (Type: {type_str}) "
            f"(Range: {self.min_vld_val}, {self.max_vld_val}) "
            f"(Normalisation: (x - {self.norm_mean}) / {self.norm_std})"
        )

    def __str__(self) -> str:
        return self._format_desc()

    def __repr__(self) -> str:
        return self._format_desc()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EnvVarInfo):
            return NotImplemented
        return (
            self.name == other.name
            and self.file == other.file
            and self.band == other.band
            and self.data_type == other.data_type
            and self.min_vld_val == other.min_vld_val
            and self.max_vld_val == other.max_vld_val
            and self.norm_mean == other.norm_mean
            and self.norm_std == other.norm_std
        )


def check_sklearn_available(feature_name: str) -> None:
    """Verify ``scikit-learn`` is installed before running model routines."""
    _check_sklearn_available(feature_name, extra="modelling")


def lookup_env_var(
    env_vars: Mapping[str, EnvVarInfo],
    var_name: str,
) -> EnvVarInfo:
    """
    Look up an :class:`EnvVarInfo` in ``env_vars`` by dictionary key or by
    :attr:`EnvVarInfo.name`.
    """
    if var_name in env_vars:
        return env_vars[var_name]
    matches = [info for info in env_vars.values() if info.name == var_name]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise rsgispy.RSGISPyException(
            f"Variable '{var_name}' matches multiple entries in env_vars."
        )
    raise rsgispy.RSGISPyException(
        f"Variable '{var_name}' was not found in env_vars."
    )


def resolve_analysis_vars(
    env_vars: Union[Sequence[str], Mapping[str, EnvVarInfo]],
    analysis_vars: Optional[Sequence[str]] = None,
) -> List[str]:
    """
    Resolve and validate an ordered list of analysis variable column names
    from ``env_vars`` and optional ``analysis_vars``.
    """
    if analysis_vars is not None:
        raw_vars = [str(v) for v in analysis_vars]
    elif isinstance(env_vars, Mapping):
        raw_vars = [
            info.name if info.name is not None else str(key)
            for key, info in env_vars.items()
        ]
    else:
        raw_vars = [str(v) for v in env_vars]

    if not raw_vars:
        raise rsgispy.RSGISPyException(
            "analysis_vars must contain at least one variable name."
        )
    if isinstance(env_vars, Mapping):
        resolved: List[str] = []
        for var_name in raw_vars:
            info = lookup_env_var(env_vars, var_name)
            resolved.append(info.name if info.name is not None else var_name)
        return resolved
    return raw_vars


def read_model_vector(
    vec_file: VectorInput,
    vec_lyr: Optional[str] = None,
) -> gpd.GeoDataFrame:
    """Load vector data from file path or :class:`geopandas.GeoDataFrame`."""
    return read_vec_gdf(vec_file, vec_lyr=vec_lyr, copy=True)


def write_model_vector(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    gdf: gpd.GeoDataFrame,
    out_vec_file: Optional[Union[str, os.PathLike]],
    out_vec_lyr: Optional[str] = None,
    out_format: Optional[str] = "GPKG",
    overwrite: bool = True,
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """Write a modelling :class:`geopandas.GeoDataFrame` or return in memory."""
    return finish_vec_output(
        gdf,
        out_vec_file=out_vec_file,
        out_vec_lyr=out_vec_lyr,
        out_format=out_format,
        overwrite=overwrite,
        out_compress=out_compress,
    )


def extract_features_and_target(
    gdf: gpd.GeoDataFrame,
    analysis_vars: Sequence[str],
    cls_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
    """
    Validate columns and return a clean ``(X_df, y_arr)`` pair from ``gdf``.
    """
    var_list = list(analysis_vars)
    required = list(var_list)
    if cls_col is not None:
        required.append(cls_col)
    validate_columns_exist(gdf, required)

    x_df = gdf[var_list].copy()
    y_arr = gdf[cls_col].to_numpy() if cls_col is not None else None
    return x_df, y_arr


def normalise_array_with_info(
    values: Union[np.ndarray, pd.Series],
    info: EnvVarInfo,
) -> Union[np.ndarray, pd.Series]:
    """
    Apply z-score normalisation ``(x - norm_mean) / norm_std`` when ``info`` is
    continuous and normalisation parameters are defined, guarding zero std.
    """
    if (
        info.data_type == rsgispy.VAR_TYPE_CONTINUOUS
        and info.norm_mean is not None
        and info.norm_std is not None
    ):
        std_val = float(info.norm_std)
        if std_val == 0.0 or not np.isfinite(std_val):
            return values - float(info.norm_mean)
        return (values - float(info.norm_mean)) / std_val
    return values


def get_positive_class_probs(
    est_cls_obj: Any,
    x_data: pd.DataFrame,
    pos_label: int = 1,
) -> np.ndarray:
    """
    Extract positive-class probability scores robustly using
    ``est_cls_obj.classes_`` so class ordering never causes column-index mismatches.
    """
    probs = np.asarray(est_cls_obj.predict_proba(x_data), dtype=np.float64)
    if probs.ndim == 1:
        return probs
    classes = getattr(est_cls_obj, "classes_", None)
    if classes is not None:
        class_list = list(classes)
        if pos_label in class_list:
            pos_idx = class_list.index(pos_label)
            return probs[:, pos_idx]
        if probs.shape[1] == 1:
            return np.zeros(probs.shape[0], dtype=np.float64)
    return probs[:, -1]


def calc_single_class_predictions(
    est_cls_obj: Any,
    x_data: pd.DataFrame,
) -> np.ndarray:
    """
    Predict binary presence (``1``) vs absence/outlier (``0``) from a
    ``scikit-learn`` single-class/outlier estimator returning ``+1`` / ``-1``.
    """
    pred = np.asarray(est_cls_obj.predict(x_data), dtype=np.int16)
    pred = np.where(pred < 0, 0, pred)
    return pred.astype(np.uint8)


def calc_binary_roc_auc(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute ROC curve ``(fpr, tpr, roc_auc)`` for binary ground truth."""
    check_sklearn_available("calc_binary_roc_auc")
    fpr, tpr, _ = sklearn.metrics.roc_curve(y_true, scores)
    roc_auc = float(sklearn.metrics.auc(fpr, tpr))
    return fpr, tpr, roc_auc


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    roc_auc: float,
    out_file: Optional[Union[str, os.PathLike]],
    label: str = "ROC Curve",
) -> None:
    """
    Plot a binary ROC curve cleanly using Matplotlib so behaviour is identical
    across all ``scikit-learn`` versions and figures are always closed.
    """
    if out_file is None:
        return
    fig, ax = plt.subplots()
    try:
        ax.plot(fpr, tpr, label=f"{label} (AUC = {roc_auc:.3f})")
        ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="grey")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Receiver Operating Characteristic")
        ax.legend(loc="lower right")
    finally:
        finalise_plot(fig, out_file=out_file)


def env_vars_to_band_defs(
    env_vars: Mapping[str, EnvVarInfo],
    analysis_vars: Sequence[str],
    mask_img: Optional[str] = None,
    mask_band: int = 1,
) -> Tuple[List[BandDefn], Dict[str, str]]:
    """
    Build validated :class:`rsgispy.imageprocess.BandDefn` objects for
    ``analysis_vars`` and optional ``mask_img``, returning ``(band_defs, var_keys)``
    where ``var_keys`` maps each variable name to its internal block key.
    """
    band_defs: List[BandDefn] = []
    if mask_img is not None:
        band_defs.append(
            BandDefn(
                band_name=_MASK_BAND_KEY,
                input_img=os.fspath(mask_img),
                img_band=int(mask_band),
            )
        )

    var_keys: Dict[str, str] = {}
    for idx, var_name in enumerate(analysis_vars):
        info = lookup_env_var(env_vars, var_name)
        if info.file is None:
            raise rsgispy.RSGISPyException(
                f"Environmental variable '{var_name}' has file=None."
            )
        b_key = f"__env_{idx}__"
        var_keys[var_name] = b_key
        band_defs.append(
            BandDefn(
                band_name=b_key,
                input_img=os.fspath(info.file),
                img_band=int(info.band),
            )
        )

    return band_defs, var_keys


def build_env_block_matrix(
    block_vars: Mapping[str, np.ndarray],
    var_keys: Mapping[str, str],
    analysis_vars: Sequence[str],
    env_vars: Mapping[str, EnvVarInfo],
    datasets: Mapping[str, Any],
    img_msk_val: int,
    normalise_data: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assemble valid, finite environmental covariate rows for pixels matching
    ``img_msk_val`` inside a raster processing block, enforcing optional
    :attr:`EnvVarInfo.min_vld_val` and :attr:`EnvVarInfo.max_vld_val` bounds.

    Returns ``(valid_df_matrix, flat_valid_indices)`` where
    ``valid_df_matrix`` has shape ``(n_valid_pxls, n_vars)``.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    msk_flat = block_vars[_MASK_BAND_KEY].ravel()
    roi_idx = np.flatnonzero(msk_flat == img_msk_val)
    if roi_idx.size == 0:
        return np.empty((0, len(analysis_vars)), dtype=np.float64), roi_idx

    band_arrays: List[Tuple[np.ndarray, Optional[float]]] = []
    infos: List[EnvVarInfo] = []
    for var_name in analysis_vars:
        info = lookup_env_var(env_vars, var_name)
        if info.file is None:
            raise rsgispy.RSGISPyException(
                f"Environmental variable '{var_name}' has file=None."
            )
        ds = datasets[os.fspath(info.file)]
        nd_val = ds.nodatavals[info.band - 1]
        band_arrays.append((block_vars[var_keys[var_name]], nd_val))
        infos.append(info)

    env_mat, valid_idx = extract_roi_feature_matrix(band_arrays, roi_idx)
    if env_mat.shape[0] > 0:
        in_range = np.ones(env_mat.shape[0], dtype=bool)
        for col_idx, info in enumerate(infos):
            col_vals = env_mat[:, col_idx]
            if info.min_vld_val is not None:
                in_range &= col_vals >= float(info.min_vld_val)
            if info.max_vld_val is not None:
                in_range &= col_vals <= float(info.max_vld_val)
        if not np.all(in_range):
            env_mat = env_mat[in_range]
            valid_idx = valid_idx[in_range]

    if normalise_data and env_mat.shape[0] > 0:
        for col_idx, info in enumerate(infos):
            env_mat[:, col_idx] = np.asarray(
                normalise_array_with_info(env_mat[:, col_idx], info),
                dtype=np.float64,
            )
    return env_mat, valid_idx


def run_env_block_pipeline(
    in_msk_img: str,
    env_vars: Mapping[str, EnvVarInfo],
    analysis_vars: Sequence[str],
    output_img: str,
    block_eval_fn: Callable[
        [
            List[str],
            Dict[str, str],
            Dict[str, np.ndarray],
            Mapping[str, Any],
            int,
            int,
        ],
        np.ndarray,
    ],
    gdalformat: str = "GTIFF",
    out_rio_dtype: str = "float32",
    calc_stats: bool = True,
    thematic: bool = False,
    no_data_val: Optional[float] = -1.0,
    block_size: Optional[int] = None,
    use_pbar: bool = True,
) -> None:
    """
    Execute a shared streaming :mod:`rsgispy.imageprocess` block pipeline
    across a mask raster and environmental covariate layers, writing output
    blocks and optionally populating image statistics and overviews.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    resolved_vars = resolve_analysis_vars(env_vars, analysis_vars)
    band_defs, var_keys = env_vars_to_band_defs(
        env_vars=env_vars,
        analysis_vars=resolved_vars,
        mask_img=in_msk_img,
        mask_band=1,
    )
    out_np_dtype = np.dtype(out_rio_dtype)
    out_path = ensure_parent_dir(output_img)

    with ExitStack() as stack:
        datasets = open_input_datasets(stack, band_defs)
        grid = compute_spatial_intersection(datasets)
        dst = open_output_dataset(
            stack=stack,
            output_img=out_path,
            gdalformat=gdalformat,
            out_rio_dtype=out_rio_dtype,
            grid=grid,
            count=1,
        )
        if no_data_val is not None:
            dst.nodata = (
                int(no_data_val)
                if np.issubdtype(out_np_dtype, np.integer)
                else float(no_data_val)
            )

        for out_win, block_vars in iter_image_blocks(
            datasets=datasets,
            norm_defs=band_defs,
            grid=grid,
            block_size=block_size,
            read_dtype=np.float64,
            use_pbar=use_pbar,
        ):
            win_h = int(out_win.height)
            win_w = int(out_win.width)
            raw_block = block_eval_fn(
                resolved_vars, var_keys, block_vars, datasets, win_h, win_w
            )
            out_arr = cast_result_block(raw_block, win_h, win_w, out_np_dtype)
            dst.write(out_arr, 1, window=out_win)

    if calc_stats:
        if thematic:
            ignore_zero = no_data_val is None or float(no_data_val) == 0.0
            pop_thmt_img_stats(
                os.fspath(output_img),
                add_clr_tab=True,
                calc_pyramids=True,
                ignore_zero=ignore_zero,
                use_pbar=use_pbar,
            )
        else:
            pop_img_stats(
                os.fspath(output_img),
                no_data_val=float(no_data_val) if no_data_val is not None else None,
                calc_pyramids=True,
                use_pbar=use_pbar,
            )


def predict_raster_from_env_vars(
    in_msk_img: str,
    img_msk_val: int,
    env_vars: Mapping[str, EnvVarInfo],
    analysis_vars: Sequence[str],
    output_img: str,
    predict_fn: Callable[[pd.DataFrame], np.ndarray],
    gdalformat: str = "GTIFF",
    out_rio_dtype: str = "float32",
    fill_val: float = -1.0,
    normalise_data: bool = False,
    calc_stats: bool = True,
    thematic: bool = False,
    no_data_val: Optional[float] = -1.0,
    block_size: Optional[int] = None,
    use_pbar: bool = True,
) -> None:
    """
    Execute block-wise spatial prediction across environmental raster layers
    using :mod:`rsgispy.imageprocess` and write the output raster.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments

    def _eval_block(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        resolved_vars: List[str],
        var_keys: Dict[str, str],
        block_vars: Dict[str, np.ndarray],
        datasets: Mapping[str, Any],
        win_h: int,
        win_w: int,
    ) -> np.ndarray:
        out_flat = np.full(win_h * win_w, fill_val, dtype=np.float64)
        env_mat, valid_idx = build_env_block_matrix(
            block_vars=block_vars,
            var_keys=var_keys,
            analysis_vars=resolved_vars,
            env_vars=env_vars,
            datasets=datasets,
            img_msk_val=img_msk_val,
            normalise_data=normalise_data,
        )
        if valid_idx.size > 0:
            env_pxl_df = pd.DataFrame(data=env_mat, columns=resolved_vars)
            preds = np.asarray(predict_fn(env_pxl_df), dtype=np.float64).ravel()
            out_flat[valid_idx] = preds
        return out_flat.reshape((win_h, win_w))

    run_env_block_pipeline(
        in_msk_img=in_msk_img,
        env_vars=env_vars,
        analysis_vars=analysis_vars,
        output_img=output_img,
        block_eval_fn=_eval_block,
        gdalformat=gdalformat,
        out_rio_dtype=out_rio_dtype,
        calc_stats=calc_stats,
        thematic=thematic,
        no_data_val=no_data_val,
        block_size=block_size,
        use_pbar=use_pbar,
    )
