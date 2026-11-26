# pylint: disable=too-many-lines
"""
Unified tiled raster prediction engine (:func:`predict_raster`,
:func:`predict_proba_raster`, :func:`predict_quantiles_raster`) shared across
:mod:`rsgispy.classification`, :mod:`rsgispy.regression`, and
:mod:`rsgispy.modelling`.

Supports any fitted estimator exposing ``predict`` / ``predict_proba`` (e.g.,
``scikit-learn``, ``XGBoost``, ``LightGBM``, ``CatBoost``, or custom pipelines),
on-the-fly valid-pixel masking (``in_msk_img=None``), and on-the-fly
:class:`~rsgispy.ml.scaling.FeatureScaler` feature scaling.

The same predictions are available as tile-graph nodes
(:func:`predict_raster_node`, :func:`predict_proba_raster_node`,
:func:`predict_quantiles_raster_node`, fed by :func:`feature_stack_nodes`)
for chaining with other operations (filters, indices, a majority filter, ...)
in one :class:`~rsgispy.imageprocess.TileGraph` pass. The file-based
functions keep their own block loop (sharing the per-pixel prediction code
with the nodes) because they also accept stacks whose rasters are not on one
pixel grid (they predict over the intersection and reproject on the fly),
create the output when the first prediction gives the number of bands, and
skip reading tiles outside the mask.
"""

from contextlib import ExitStack
import os
from typing import (
    Any,
    Callable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
import rasterio

import rsgispy
from rsgispy._io import (
    NodeMeta,
    PointwiseOp,
    RasterSource,
    TileContext,
    TileNode,
    TileTensor,
)
from rsgispy.imageprocess import (
    _INT_DTYPES,
    build_valid_mask,
    cast_result_block,
    open_output_dataset,
)
from rsgispy.imageprocess import normalise_creation_options
from rsgispy.imageutils.colourtable import set_img_not_thematic, set_img_thematic
from rsgispy.imageutils.metadata import set_band_names, set_img_no_data_value
from rsgispy.imageutils.stats import pop_img_stats, pop_thmt_img_stats
from rsgispy.ml.scaling import FeatureScaler
from rsgispy.ml.stack import ImageFeatureStack
from rsgispy.tools.files import ensure_parent_dir

__all__ = [
    "predict_raster",
    "predict_proba_raster",
    "predict_quantiles_raster",
    "feature_stack_nodes",
    "predict_raster_node",
    "predict_proba_raster_node",
    "predict_quantiles_raster_node",
]


def _coerce_feature_stack(
    feature_stack: Union[ImageFeatureStack, Sequence[Any]],
) -> ImageFeatureStack:
    if isinstance(feature_stack, ImageFeatureStack):
        return feature_stack
    return ImageFeatureStack(feature_stack)


def _needs_dataframe_input(estimator: Any) -> bool:
    """Return ``True`` when ``estimator`` was fitted with named DataFrame columns."""
    return hasattr(estimator, "feature_names_in_") and (
        estimator.feature_names_in_ is not None
    )


def _resolve_estimator_df_cols(
    estimator: Any,
    stack_names: Sequence[str],
) -> List[str]:
    if hasattr(estimator, "feature_names_in_") and estimator.feature_names_in_ is not None:
        est_cols = [str(c) for c in estimator.feature_names_in_]
        if len(est_cols) == len(stack_names):
            return est_cols
    return list(stack_names)


def _apply_lut_1d(vals: np.ndarray, lut: Mapping[Any, Any]) -> np.ndarray:
    """Map predicted integer/label values via ``lut``."""
    if lut:
        target_arr = np.asarray(list(lut.values()))
        out_dtype = np.result_type(vals.dtype, target_arr.dtype)
    else:
        out_dtype = vals.dtype
    out = np.empty(vals.shape, dtype=out_dtype)
    for idx, val in enumerate(vals):
        key = int(val) if isinstance(val, (int, np.integer)) else val
        out[idx] = lut.get(key, val)
    return out


def _model_input(x_valid: np.ndarray, use_df: bool, df_cols: Sequence[str]) -> Any:
    """Feature matrix as passed to the estimator (a DataFrame when it was fitted on one)."""
    return pd.DataFrame(x_valid, columns=list(df_cols)) if use_df else x_valid


def _predict_labels(
    estimator: Any,
    x_in: Any,
    threshold: Optional[float],
    lut: Optional[Mapping[Any, Any]],
) -> np.ndarray:
    """
    Predictions of ``estimator`` for the valid pixels ``x_in`` (thresholded
    positive-class probabilities when ``threshold`` is given), mapped
    through ``lut``.
    """
    if threshold is not None:
        probs = np.asarray(estimator.predict_proba(x_in), dtype=np.float64)
        pos_prob = probs[:, -1] if probs.ndim == 2 else probs
        preds = (pos_prob >= float(threshold)).astype(np.int64)
    else:
        preds = np.asarray(estimator.predict(x_in))
    if lut is not None:
        preds = _apply_lut_1d(preds.ravel(), lut).reshape(preds.shape)
    return preds


def _predict_proba_2d(estimator: Any, x_in: Any) -> np.ndarray:
    """Class probabilities ``(n_valid, n_classes)`` (1D outputs taken as P(positive))."""
    probs = np.asarray(estimator.predict_proba(x_in), dtype=np.float64)
    if probs.ndim == 1:
        probs = np.column_stack([1.0 - probs, probs])
    return probs


def _proba_class_labels(
    probs: np.ndarray, est_classes: Any, lut: Optional[Mapping[Any, Any]]
) -> np.ndarray:
    """Most probable class label of each pixel (``int64``), mapped through ``lut``."""
    argm = np.argmax(probs, axis=1)
    labels = np.asarray(est_classes)[argm] if est_classes is not None else argm
    if lut is not None:
        labels = _apply_lut_1d(labels, lut)
    return labels.astype(np.int64, copy=False)


def _proba_entropy(probs: np.ndarray, n_classes: int) -> np.ndarray:
    """Shannon entropy of each pixel's probabilities, normalised to ``[0, 1]``."""
    p_clip = np.clip(probs, 1e-12, 1.0)
    denom = np.log(float(max(n_classes, 2)))
    return -np.sum(p_clip * np.log(p_clip), axis=1) / denom


class _QuantilePredictor:
    """
    Quantile / prediction-interval model of :func:`predict_quantiles_raster`:
    a ``{quantile: model}`` mapping, a point regressor with split-conformal
    residuals, or a bagging/forest ensemble (empirical quantiles of its
    trees). Validates the arguments and gives the output band names.
    """

    # pylint: disable=too-few-public-methods,too-many-instance-attributes

    def __init__(
        self,
        estimators_or_model: Any,
        quantiles: Sequence[float],
        conformal_residuals: Optional[Sequence[float]],
        alpha: float,
    ) -> None:
        self.model = estimators_or_model
        self.is_mapping = isinstance(estimators_or_model, Mapping)
        self.is_conformal = conformal_residuals is not None
        self.q_hat = 0.0
        self.q_models: List[Any] = []
        self.q_list: List[float] = []
        if self.is_mapping:
            if not estimators_or_model:
                raise rsgispy.RSGISPyException(
                    "estimators_or_model mapping must not be empty."
                )
            q_keys = sorted(float(k) for k in estimators_or_model.keys())
            self.q_models = [
                estimators_or_model[k] for k in sorted(estimators_or_model.keys())
            ]
            self.band_names = [f"q{int(round(q * 100)):02d}" for q in q_keys]
            self.ref_est = self.q_models[0]
        elif self.is_conformal:
            self._init_conformal(conformal_residuals, alpha)
        elif hasattr(estimators_or_model, "estimators_"):
            self._init_ensemble(quantiles)
        else:
            raise rsgispy.RSGISPyException(
                "estimators_or_model must be a {quantile: model} mapping, a tree "
                "ensemble with .estimators_, or a regressor with conformal_residuals."
            )

    def _init_conformal(self, conformal_residuals: Any, alpha: float) -> None:
        res_arr = np.abs(np.asarray(conformal_residuals, dtype=np.float64)).ravel()
        res_arr = res_arr[np.isfinite(res_arr)]
        if res_arr.size == 0:
            raise rsgispy.RSGISPyException(
                "conformal_residuals must contain at least one finite value."
            )
        if not 0.0 < float(alpha) < 1.0:
            raise rsgispy.RSGISPyException(
                f"alpha must be strictly between 0 and 1, got {alpha}."
            )
        n_cal = res_arr.size
        q_level = min(1.0, np.ceil((n_cal + 1) * (1.0 - float(alpha))) / n_cal)
        self.q_hat = float(np.quantile(res_arr, q_level, method="higher"))
        cov_pct = int(round((1.0 - float(alpha)) * 100))
        self.band_names = [
            f"lower_{cov_pct}pct",
            "prediction",
            f"upper_{cov_pct}pct",
        ]
        self.ref_est = self.model

    def _init_ensemble(self, quantiles: Sequence[float]) -> None:
        sub_ests = self.model.estimators_
        is_boosted = (
            isinstance(sub_ests, np.ndarray)
            or hasattr(self.model, "init_")
            or hasattr(self.model, "learning_rate")
            or hasattr(self.model, "estimator_weights_")
        )
        if is_boosted or not sub_ests or not all(hasattr(e, "predict") for e in sub_ests):
            raise rsgispy.RSGISPyException(
                "Empirical quantile prediction via .estimators_ is only supported "
                "for independent bagging/forest ensembles (e.g., RandomForestRegressor, "
                "ExtraTreesRegressor), not boosted ensembles such as "
                f"{type(self.model).__name__}. Pass a {{quantile: model}} "
                "mapping or conformal_residuals instead."
            )
        self.q_list = [float(q) for q in quantiles]
        if not self.q_list or any(q <= 0.0 or q >= 1.0 for q in self.q_list):
            raise rsgispy.RSGISPyException(
                "quantiles must be a non-empty sequence of values in (0, 1)."
            )
        self.band_names = [f"q{int(round(q * 100)):02d}" for q in self.q_list]
        self.ref_est = self.model

    def predict(self, x_in: Any, x_valid: np.ndarray) -> np.ndarray:
        """
        Output values ``(n_bands, n_valid)`` for the valid pixels (``x_in``
        as passed to the models; ``x_valid`` the plain array, used for the
        trees of an ensemble).
        """
        out = np.empty((len(self.band_names), int(x_valid.shape[0])), dtype=np.float64)
        if self.is_mapping:
            for b_idx, mdl in enumerate(self.q_models):
                out[b_idx] = np.asarray(mdl.predict(x_in), dtype=np.float64).ravel()
        elif self.is_conformal:
            y_hat = np.asarray(self.model.predict(x_in), dtype=np.float64).ravel()
            out[0] = y_hat - self.q_hat
            out[1] = y_hat
            out[2] = y_hat + self.q_hat
        else:
            tree_preds = np.vstack(
                [
                    np.asarray(est.predict(x_valid), dtype=np.float64).ravel()
                    for est in self.model.estimators_
                ]
            )
            out[:] = np.quantile(tree_preds, self.q_list, axis=0)
        return out


def _finalise_raster_output(
    out_path: str,
    out_rio_dt: str,
    nodata: Optional[float],
    band_names: Optional[Sequence[str]],
    thematic: bool,
    calc_stats: bool,
    use_pbar: bool,
) -> None:
    """Apply band names, nodata, thematic flag, and statistics/pyramids."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    if band_names:
        set_band_names(out_path, list(band_names), feedback=False)
    if thematic and out_rio_dt in _INT_DTYPES:
        set_img_thematic(out_path)
    else:
        set_img_not_thematic(out_path)
    if nodata is not None:
        set_img_no_data_value(out_path, nodata)
    if calc_stats:
        if thematic and out_rio_dt in _INT_DTYPES:
            pop_thmt_img_stats(
                out_path,
                add_clr_tab=False,
                calc_pyramids=True,
                ignore_zero=(nodata == 0),
                use_pbar=use_pbar,
            )
        else:
            pop_img_stats(
                out_path,
                no_data_val=nodata,
                calc_pyramids=True,
                use_pbar=use_pbar,
            )


def predict_raster(
    estimator: Any,
    feature_stack: Union[ImageFeatureStack, Sequence[Any]],
    output_img: Union[str, os.PathLike],
    in_msk_img: Optional[Union[str, os.PathLike]] = None,
    img_msk_val: int = 1,
    scaler: Optional[FeatureScaler] = None,
    lut: Optional[Mapping[Any, Any]] = None,
    threshold: Optional[float] = None,
    thematic: bool = False,
    band_names: Optional[Sequence[str]] = None,
    block_size: Optional[int] = None,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
    creation_opts: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Apply a fitted estimator (classifier, regressor, or SDM) tile-by-tile across
    an :class:`~rsgispy.ml.stack.ImageFeatureStack` and write predictions to
    ``output_img``.

    :param estimator: Fitted estimator exposing ``.predict(X)`` (or
                      ``.predict_proba(X)`` when ``threshold`` is specified).
    :param feature_stack: :class:`~rsgispy.ml.stack.ImageFeatureStack` or
                          sequence of :class:`~rsgispy.ml.stack.BandSpec` /
                          :class:`~rsgispy.imageutils.masking.ImageBandInfo` /
                          file paths.
    :param output_img: Path to the output prediction raster.
    :param in_msk_img: Optional binary ROI mask raster; when ``None`` (Default),
                       valid pixels are masked on the fly from input nodata,
                       finiteness, and valid bounds.
    :param img_msk_val: Mask pixel value indicating ROI pixels (Default = ``1``).
    :param scaler: Optional :class:`~rsgispy.ml.scaling.FeatureScaler` override
                   applied per tile (falls back to ``feature_stack.scaler``).
    :param lut: Optional dictionary mapping raw predicted labels to output IDs.
    :param threshold: Optional probability threshold in ``(0, 1)`` for binary
                      classification using ``predict_proba[:, -1] >= threshold``.
    :param thematic: Mark integer output as thematic (Default = ``False``).
    :param band_names: Optional output band name(s).
    :param block_size: Optional processing tile size in pixels.
    :param gdalformat: GDAL/rasterio driver format (Default = ``'GTIFF'``).
    :param datatype: Optional ``rsgispy.TYPE_*`` constant. Defaults to
                     :data:`rsgispy.TYPE_8UINT` when ``thematic=True`` or
                     ``threshold is not None``, else :data:`rsgispy.TYPE_32FLOAT`.
    :param no_data_val: Optional output nodata fill value (defaults to ``0`` for
                        integer outputs or ``-9999.0`` for floating outputs).
    :param calc_stats: Populate raster statistics and pyramids (Default = ``True``).
    :param use_pbar: Display a tqdm progress bar (Default = ``True``).
    :param creation_opts: Optional GDAL/rasterio creation options mapping.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    stack_obj = _coerce_feature_stack(feature_stack)
    out_path = os.fspath(output_img)
    ensure_parent_dir(out_path)
    norm_opts = normalise_creation_options(creation_opts)

    is_thematic = bool(thematic or threshold is not None)
    if datatype is None:
        out_dt = rsgispy.TYPE_8UINT if is_thematic else rsgispy.TYPE_32FLOAT
    else:
        out_dt = int(datatype)
    out_rio_dt = rsgispy.get_rasterio_datatype(out_dt)
    out_np_dt = np.dtype(rsgispy.get_numpy_datatype(out_dt))
    eff_nd = (
        no_data_val
        if no_data_val is not None
        else (0 if np.issubdtype(out_np_dt, np.integer) else -9999.0)
    )

    use_df = _needs_dataframe_input(estimator)
    df_cols = _resolve_estimator_df_cols(estimator, stack_obj.feature_names)

    with ExitStack() as stack:
        dst = None
        out_count = 1
        empty_windows: List[Any] = []
        last_grid: Optional[Any] = None

        def _write_nodata_windows(target_dst: Any, n_bands: int) -> None:
            for e_win in empty_windows:
                ew_h = int(e_win.height)
                ew_w = int(e_win.width)
                if n_bands == 1:
                    eblk = cast_result_block(
                        np.full((ew_h, ew_w), eff_nd, dtype=np.float64),
                        ew_h,
                        ew_w,
                        out_np_dt,
                    )
                    target_dst.write(eblk, 1, window=e_win)
                else:
                    eblk_m = cast_result_block(
                        np.full(
                            (n_bands, ew_h, ew_w), eff_nd, dtype=np.float64
                        ),
                        ew_h,
                        ew_w,
                        out_np_dt,
                    )
                    target_dst.write(eblk_m, window=e_win)
            empty_windows.clear()

        for out_win, grid, valid_idx, x_valid in stack_obj.iter_blocks(
            in_msk_img=in_msk_img,
            img_msk_val=img_msk_val,
            block_size=block_size,
            scaler=scaler,
            apply_scaler=True,
            as_dataframe=False,
            use_pbar=use_pbar,
        ):
            last_grid = grid
            win_h = int(out_win.height)
            win_w = int(out_win.width)
            if valid_idx.size == 0 and dst is None:
                empty_windows.append(out_win)
                continue

            preds: Optional[np.ndarray] = None
            if valid_idx.size > 0:
                preds = _predict_labels(
                    estimator, _model_input(x_valid, use_df, df_cols), threshold, lut
                )

            if dst is None:
                out_count = (
                    int(preds.shape[1])
                    if (preds is not None and preds.ndim == 2)
                    else 1
                )
                dst = open_output_dataset(
                    stack=stack,
                    output_img=out_path,
                    gdalformat=gdalformat,
                    out_rio_dtype=out_rio_dt,
                    grid=grid,
                    count=out_count,
                    creation_opts=norm_opts,
                )
                dst.nodata = eff_nd
                _write_nodata_windows(dst, out_count)

            if out_count == 1:
                flat = np.full(win_h * win_w, eff_nd, dtype=np.float64)
                if preds is not None:
                    flat[valid_idx] = preds.ravel().astype(
                        np.float64, copy=False
                    )
                blk = cast_result_block(
                    flat.reshape(win_h, win_w), win_h, win_w, out_np_dt
                )
                dst.write(blk, 1, window=out_win)
            else:
                flat_m = np.full(
                    (out_count, win_h * win_w), eff_nd, dtype=np.float64
                )
                if preds is not None:
                    flat_m[:, valid_idx] = preds.T.astype(
                        np.float64, copy=False
                    )
                blk_m = cast_result_block(
                    flat_m.reshape(out_count, win_h, win_w),
                    win_h,
                    win_w,
                    out_np_dt,
                )
                dst.write(blk_m, window=out_win)

        if dst is None and last_grid is not None:
            n_out_attr = getattr(estimator, "n_outputs_", None)
            if band_names:
                out_count = len(band_names)
            elif isinstance(n_out_attr, (int, np.integer)) and int(n_out_attr) > 0:
                out_count = int(n_out_attr)
            else:
                out_count = 1
            dst = open_output_dataset(
                stack=stack,
                output_img=out_path,
                gdalformat=gdalformat,
                out_rio_dtype=out_rio_dt,
                grid=last_grid,
                count=out_count,
                creation_opts=norm_opts,
            )
            dst.nodata = eff_nd
            _write_nodata_windows(dst, out_count)

    _finalise_raster_output(
        out_path=out_path,
        out_rio_dt=out_rio_dt,
        nodata=eff_nd,
        band_names=band_names,
        thematic=is_thematic,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )


def predict_proba_raster(
    estimator: Any,
    feature_stack: Union[ImageFeatureStack, Sequence[Any]],
    output_img: Union[str, os.PathLike],
    in_msk_img: Optional[Union[str, os.PathLike]] = None,
    img_msk_val: int = 1,
    scaler: Optional[FeatureScaler] = None,
    class_names: Optional[Sequence[str]] = None,
    out_class_img: Optional[Union[str, os.PathLike]] = None,
    out_entropy_img: Optional[Union[str, os.PathLike]] = None,
    lut: Optional[Mapping[Any, Any]] = None,
    scale_to_pct: bool = False,
    block_size: Optional[int] = None,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
    creation_opts: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Apply a fitted probabilistic classifier tile-by-tile across
    ``feature_stack``, writing per-class probabilities to ``output_img`` and
    optionally writing hard argmax classes (``out_class_img``) and normalised
    Shannon entropy (``out_entropy_img``) in the same single pass.

    :param estimator: Fitted classifier implementing ``.predict_proba(X)``.
    :param feature_stack: :class:`~rsgispy.ml.stack.ImageFeatureStack` or
                          band-spec sequence.
    :param output_img: Path to output multi-band probability raster.
    :param in_msk_img: Optional ROI mask raster (on-the-fly valid mask if ``None``).
    :param img_msk_val: Mask value indicating ROI pixels (Default = ``1``).
    :param scaler: Optional :class:`~rsgispy.ml.scaling.FeatureScaler` override.
    :param class_names: Optional band names for each class probability band.
    :param out_class_img: Optional path to also save argmax class raster.
    :param out_entropy_img: Optional path to also save Shannon entropy raster.
    :param lut: Optional mapping from estimator class labels to output IDs for
                ``out_class_img``.
    :param scale_to_pct: Scale probabilities from ``[0, 1]`` to ``[0, 100]``
                         (Default = ``False``).
    :param block_size: Optional processing tile size in pixels.
    :param gdalformat: GDAL/rasterio driver format (Default = ``'GTIFF'``).
    :param datatype: Optional ``rsgispy.TYPE_*`` constant for ``output_img``
                     (Default = :data:`rsgispy.TYPE_32FLOAT`).
    :param no_data_val: Optional nodata value for ``output_img``
                        (Default = ``-9999.0`` for float or ``255`` for uint8).
    :param calc_stats: Populate raster statistics and pyramids (Default = ``True``).
    :param use_pbar: Display a tqdm progress bar (Default = ``True``).
    :param creation_opts: Optional GDAL/rasterio creation options mapping.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    if not hasattr(estimator, "predict_proba"):
        raise rsgispy.RSGISPyException(
            "Estimator must implement predict_proba() for predict_proba_raster."
        )
    stack_obj = _coerce_feature_stack(feature_stack)
    out_path = os.fspath(output_img)
    ensure_parent_dir(out_path)
    cls_path = os.fspath(out_class_img) if out_class_img is not None else None
    ent_path = os.fspath(out_entropy_img) if out_entropy_img is not None else None
    if cls_path:
        ensure_parent_dir(cls_path)
    if ent_path:
        ensure_parent_dir(ent_path)

    norm_opts = normalise_creation_options(creation_opts)
    out_dt = rsgispy.TYPE_32FLOAT if datatype is None else int(datatype)
    out_rio_dt = rsgispy.get_rasterio_datatype(out_dt)
    out_np_dt = np.dtype(rsgispy.get_numpy_datatype(out_dt))
    eff_nd = (
        no_data_val
        if no_data_val is not None
        else (255 if out_np_dt == np.dtype(np.uint8) else -9999.0)
    )
    scale_mult = 100.0 if scale_to_pct else 1.0

    use_df = _needs_dataframe_input(estimator)
    df_cols = _resolve_estimator_df_cols(estimator, stack_obj.feature_names)
    est_classes = getattr(estimator, "classes_", None)

    with ExitStack() as stack:
        prob_dst = None
        cls_dst = None
        ent_dst = None
        n_classes: Optional[int] = (
            len(est_classes) if est_classes is not None else None
        )
        cls_out_dt = np.dtype(np.int32)
        cls_rio_dt = "int32"

        for out_win, grid, valid_idx, x_valid in stack_obj.iter_blocks(
            in_msk_img=in_msk_img,
            img_msk_val=img_msk_val,
            block_size=block_size,
            scaler=scaler,
            apply_scaler=True,
            as_dataframe=False,
            use_pbar=use_pbar,
        ):
            win_h = int(out_win.height)
            win_w = int(out_win.width)
            probs: Optional[np.ndarray] = None
            if valid_idx.size > 0:
                probs = _predict_proba_2d(
                    estimator, _model_input(x_valid, use_df, df_cols)
                )
                if n_classes is None:
                    n_classes = int(probs.shape[1])

            if prob_dst is None:
                eff_n_cls = n_classes if n_classes is not None else 2
                prob_dst = open_output_dataset(
                    stack=stack,
                    output_img=out_path,
                    gdalformat=gdalformat,
                    out_rio_dtype=out_rio_dt,
                    grid=grid,
                    count=eff_n_cls,
                    creation_opts=norm_opts,
                )
                prob_dst.nodata = eff_nd
                if cls_path is not None:
                    cls_dst = open_output_dataset(
                        stack=stack,
                        output_img=cls_path,
                        gdalformat=gdalformat,
                        out_rio_dtype=cls_rio_dt,
                        grid=grid,
                        count=1,
                        creation_opts=norm_opts,
                    )
                    cls_dst.nodata = 0
                if ent_path is not None:
                    ent_dst = open_output_dataset(
                        stack=stack,
                        output_img=ent_path,
                        gdalformat=gdalformat,
                        out_rio_dtype="float32",
                        grid=grid,
                        count=1,
                        creation_opts=norm_opts,
                    )
                    ent_dst.nodata = -9999.0

            eff_n_cls = int(prob_dst.count)
            prob_flat = np.full(
                (eff_n_cls, win_h * win_w), eff_nd, dtype=np.float64
            )
            if probs is not None:
                prob_flat[:, valid_idx] = (probs * scale_mult).T
            prob_blk = cast_result_block(
                prob_flat.reshape(eff_n_cls, win_h, win_w),
                win_h,
                win_w,
                out_np_dt,
            )
            prob_dst.write(prob_blk, window=out_win)

            if cls_dst is not None:
                cls_flat = np.zeros(win_h * win_w, dtype=np.int64)
                if probs is not None:
                    cls_flat[valid_idx] = _proba_class_labels(probs, est_classes, lut)
                cls_dst.write(
                    cast_result_block(
                        cls_flat.reshape(win_h, win_w),
                        win_h,
                        win_w,
                        cls_out_dt,
                    ),
                    1,
                    window=out_win,
                )

            if ent_dst is not None:
                ent_flat = np.full(win_h * win_w, -9999.0, dtype=np.float64)
                if probs is not None:
                    ent_flat[valid_idx] = _proba_entropy(probs, eff_n_cls)
                ent_dst.write(
                    ent_flat.reshape(win_h, win_w).astype(np.float32),
                    1,
                    window=out_win,
                )

    eff_names: Optional[List[str]] = None
    if class_names is not None:
        eff_names = [f"prob_{c}" for c in class_names]
    elif est_classes is not None:
        eff_names = [f"prob_{c}" for c in est_classes]

    _finalise_raster_output(
        out_path=out_path,
        out_rio_dt=out_rio_dt,
        nodata=eff_nd,
        band_names=eff_names,
        thematic=False,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )
    if cls_path is not None:
        _finalise_raster_output(
            out_path=cls_path,
            out_rio_dt="int32",
            nodata=0,
            band_names=["predicted_class"],
            thematic=True,
            calc_stats=calc_stats,
            use_pbar=use_pbar,
        )
    if ent_path is not None:
        _finalise_raster_output(
            out_path=ent_path,
            out_rio_dt="float32",
            nodata=-9999.0,
            band_names=["shannon_entropy"],
            thematic=False,
            calc_stats=calc_stats,
            use_pbar=use_pbar,
        )


def predict_quantiles_raster(
    estimators_or_model: Any,
    feature_stack: Union[ImageFeatureStack, Sequence[Any]],
    output_img: Union[str, os.PathLike],
    quantiles: Sequence[float] = (0.05, 0.5, 0.95),
    in_msk_img: Optional[Union[str, os.PathLike]] = None,
    img_msk_val: int = 1,
    scaler: Optional[FeatureScaler] = None,
    conformal_residuals: Optional[Sequence[float]] = None,
    alpha: float = 0.1,
    block_size: Optional[int] = None,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
    creation_opts: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Predict per-pixel regression quantiles or conformal prediction intervals
    across ``feature_stack``, writing one band per quantile to ``output_img``.

    Supports three modes:

    1. **Quantile model mapping** ``{q: fitted_model}`` (e.g., quantile GBDTs
       for ``{0.05: m_lo, 0.5: m_med, 0.95: m_hi}``).
    2. **Split-conformal prediction intervals**: pass a single fitted point
       regressor plus calibration residuals ``conformal_residuals`` (``|y - y_hat|``)
       and miscoverage level ``alpha`` (Default = ``0.1`` -> 90% interval
       ``[y_hat - q_hat, y_hat, y_hat + q_hat]``).
    3. **Ensemble empirical quantiles**: pass a tree ensemble exposing
       ``.estimators_`` (e.g., ``RandomForestRegressor`` or
       ``ExtraTreesRegressor``) to evaluate per-pixel empirical ``quantiles``
       across all trees.

    :param estimators_or_model: Mapping ``{quantile: model}``, tree ensemble,
                                or point regressor (with ``conformal_residuals``).
    :param feature_stack: :class:`~rsgispy.ml.stack.ImageFeatureStack` or
                          band-spec sequence.
    :param output_img: Path to output multi-band quantile/interval raster.
    :param quantiles: Sequence of quantiles in ``(0, 1)`` (Default = ``(0.05, 0.5, 0.95)``).
    :param in_msk_img: Optional ROI mask raster (on-the-fly valid mask if ``None``).
    :param img_msk_val: Mask pixel value indicating ROI pixels (Default = ``1``).
    :param scaler: Optional :class:`~rsgispy.ml.scaling.FeatureScaler` override.
    :param conformal_residuals: Optional 1D array of calibration absolute
                                residuals ``|y_cal - y_hat_cal|`` for
                                split-conformal prediction intervals.
    :param alpha: Miscoverage level in ``(0, 1)`` for conformal intervals
                  (Default = ``0.1`` -> 90% prediction interval).
    :param block_size: Optional processing tile size in pixels.
    :param gdalformat: GDAL/rasterio driver format (Default = ``'GTIFF'``).
    :param datatype: Optional ``rsgispy.TYPE_*`` constant
                     (Default = :data:`rsgispy.TYPE_32FLOAT`).
    :param no_data_val: Optional nodata value (Default = ``-9999.0``).
    :param calc_stats: Populate raster statistics and pyramids (Default = ``True``).
    :param use_pbar: Display a tqdm progress bar (Default = ``True``).
    :param creation_opts: Optional GDAL/rasterio creation options mapping.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-branches
    stack_obj = _coerce_feature_stack(feature_stack)
    out_path = os.fspath(output_img)
    ensure_parent_dir(out_path)
    norm_opts = normalise_creation_options(creation_opts)

    out_dt = rsgispy.TYPE_32FLOAT if datatype is None else int(datatype)
    out_rio_dt = rsgispy.get_rasterio_datatype(out_dt)
    out_np_dt = np.dtype(rsgispy.get_numpy_datatype(out_dt))
    eff_nd = -9999.0 if no_data_val is None else float(no_data_val)

    q_pred = _QuantilePredictor(
        estimators_or_model, quantiles, conformal_residuals, alpha
    )
    band_names = q_pred.band_names
    ref_est = q_pred.ref_est

    use_df = _needs_dataframe_input(ref_est)
    df_cols = _resolve_estimator_df_cols(ref_est, stack_obj.feature_names)
    n_out = len(band_names)

    with ExitStack() as stack:
        dst = None
        for out_win, grid, valid_idx, x_valid in stack_obj.iter_blocks(
            in_msk_img=in_msk_img,
            img_msk_val=img_msk_val,
            block_size=block_size,
            scaler=scaler,
            apply_scaler=True,
            as_dataframe=False,
            use_pbar=use_pbar,
        ):
            win_h = int(out_win.height)
            win_w = int(out_win.width)
            if dst is None:
                dst = open_output_dataset(
                    stack=stack,
                    output_img=out_path,
                    gdalformat=gdalformat,
                    out_rio_dtype=out_rio_dt,
                    grid=grid,
                    count=n_out,
                    creation_opts=norm_opts,
                )
                dst.nodata = eff_nd

            flat_out = np.full(
                (n_out, win_h * win_w), eff_nd, dtype=np.float64
            )
            if valid_idx.size > 0:
                flat_out[:, valid_idx] = q_pred.predict(
                    _model_input(x_valid, use_df, df_cols), x_valid
                )

            dst.write(
                cast_result_block(
                    flat_out.reshape(n_out, win_h, win_w),
                    win_h,
                    win_w,
                    out_np_dt,
                ),
                window=out_win,
            )

    _finalise_raster_output(
        out_path=out_path,
        out_rio_dt=out_rio_dt,
        nodata=eff_nd,
        band_names=band_names,
        thematic=False,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )


# ---------------------------------------------------------------------------
# Tile-graph nodes
# ---------------------------------------------------------------------------

FeatureNodes = Union[TileNode, Sequence[TileNode]]


def feature_stack_nodes(
    feature_stack: Union[ImageFeatureStack, Sequence[Any]],
) -> List[TileNode]:
    """
    Build the tile-graph source nodes of an
    :class:`~rsgispy.ml.stack.ImageFeatureStack`, to feed
    :func:`predict_raster_node`, :func:`predict_proba_raster_node` or
    :func:`predict_quantiles_raster_node` (or any other node).

    Each :class:`~rsgispy.ml.stack.BandSpec` gives one node with its bands
    read as ``float64``, the stack's feature names as band names and its
    no-data value (``BandSpec.no_data_val`` or the raster's) as the node's
    no-data value; values outside ``min_vld_val`` / ``max_vld_val`` are set
    to NaN, so the prediction nodes treat them as invalid, as the file-based
    prediction functions do. Specs on the same file share one reader.

    Unlike the file-based prediction functions, which use the intersection
    of the rasters (and reproject rasters on other grids on the fly), a
    :class:`~rsgispy.imageprocess.TileGraph` requires all rasters (and the
    mask) to be on the same pixel grid; it raises an exception naming the
    rasters otherwise. The stack's ``scaler`` is not applied by these nodes:
    pass it to the prediction node (``scaler=stack.scaler``).

    :param feature_stack: :class:`~rsgispy.ml.stack.ImageFeatureStack` or
                          sequence of :class:`~rsgispy.ml.stack.BandSpec` /
                          file paths.
    :return: One :class:`~rsgispy.imageprocess.TileNode` per band spec.

    .. code:: python

        from rsgispy.imageprocess import DiskSink, StackBands, TileGraph
        from rsgispy.ml.predict import feature_stack_nodes, predict_raster_node
        from rsgispy.ml.stack import BandSpec, ImageFeatureStack

        stack = ImageFeatureStack([BandSpec("s2.tif", name="s2"), BandSpec("dem.tif")])
        feats = feature_stack_nodes(stack)
        print(TileGraph(DiskSink(StackBands(feats), "features.tif")).describe())
        pred = predict_raster_node(feats, rf_model, scaler=stack.scaler)
    """
    stack_obj = _coerce_feature_stack(feature_stack)
    resolved = stack_obj.resolve_band_specs()
    nodes: List[TileNode] = []
    pos = 0
    for idx, spec in enumerate(stack_obj.specs):
        if spec.bands is None:
            with rasterio.open(spec.file_name, "r") as src_ds:
                n_bands = int(src_ds.count)
        else:
            n_bands = len(spec.bands)
        spec_items = resolved[pos:pos + n_bands]
        pos += n_bands
        names = tuple(name for _, _, name in spec_items)
        source = RasterSource(
            spec.file_name,
            bands=[band for _, band, _ in spec_items],
            dtype=np.float64,
            nodata=spec.no_data_val,
            name=f"{spec.name or 'features'}_{idx + 1}_src",
        )
        nodes.append(
            PointwiseOp(
                source,
                func=_valid_range_fn(spec.min_vld_val, spec.max_vld_val),
                name=spec.name or f"features_{idx + 1}",
                meta_fn=_rename_meta(names),
            )
        )
    return nodes


def _valid_range_fn(
    lo_val: Optional[float], hi_val: Optional[float]
) -> Callable[[np.ndarray], np.ndarray]:
    """Function setting values outside ``[lo_val, hi_val]`` to NaN."""
    if lo_val is None and hi_val is None:
        return lambda arr: arr

    def _apply(arr: np.ndarray) -> np.ndarray:
        bad = np.zeros(arr.shape, dtype=bool)
        with np.errstate(invalid="ignore"):
            if lo_val is not None:
                bad |= ~(arr >= float(lo_val)) & np.isfinite(arr)
            if hi_val is not None:
                bad |= ~(arr <= float(hi_val)) & np.isfinite(arr)
        return np.where(bad, np.nan, arr)

    return _apply


def _rename_meta(names: Tuple[str, ...]) -> Callable[[Sequence[NodeMeta]], NodeMeta]:
    def _meta(inputs: Sequence[NodeMeta]) -> NodeMeta:
        return inputs[0].replace(band_names=names)

    return _meta


class PredictionOp(TileNode):
    """
    Pixel-wise model prediction node: gathers the valid pixels of the
    feature inputs (and of an optional mask), applies an optional
    :class:`~rsgispy.ml.scaling.FeatureScaler` and evaluates ``compute`` on
    them. Built by :func:`predict_raster_node`,
    :func:`predict_proba_raster_node` and
    :func:`predict_quantiles_raster_node`.

    A pixel is valid when it is inside the mask (``mask == img_msk_val``)
    and every feature band is finite and different from its input's no-data
    value (from the inputs' metadata). Invalid pixels get ``fill_values``.

    :param features: Feature node or sequence of feature nodes (their bands,
                     in order, are the model's features).
    :param compute: ``compute(x_in, x_valid) -> (n_out, n_valid)`` array;
                    ``x_in`` is passed to the estimator (a DataFrame when
                    the estimator was fitted on one), ``x_valid`` is the
                    plain ``float64`` matrix.
    :param n_out: Number of output bands.
    :param fill_values: Output value per band at invalid pixels.
    :param estimator: Estimator (to check the number of features and to
                      name DataFrame columns).
    :param mask: Optional mask node (band 1 used).
    :param img_msk_val: Mask value of the pixels to predict.
    :param scaler: Optional fitted :class:`~rsgispy.ml.scaling.FeatureScaler`.
    :param band_names: Optional output band names.
    :param nodata: Output no-data value (metadata).
    :param name: Optional node name.
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        features: FeatureNodes,
        compute: Callable[[Any, np.ndarray], np.ndarray],
        n_out: int,
        fill_values: Sequence[float],
        *,
        estimator: Any = None,
        mask: Optional[TileNode] = None,
        img_msk_val: float = 1,
        scaler: Optional[FeatureScaler] = None,
        band_names: Optional[Sequence[str]] = None,
        nodata: Optional[float] = None,
        name: Optional[str] = None,
    ) -> None:
        # pylint: disable=too-many-arguments
        feat_list = [features] if isinstance(features, TileNode) else list(features)
        if not feat_list:
            raise rsgispy.RSGISPyException(
                "Prediction nodes require at least one feature node."
            )
        for node in feat_list + ([mask] if mask is not None else []):
            if not isinstance(node, TileNode):
                raise rsgispy.RSGISPyException(
                    "Prediction node features and mask must be TileNode objects, "
                    f"got {type(node).__name__}."
                )
        if len(fill_values) != int(n_out):
            raise rsgispy.RSGISPyException(
                f"{len(fill_values)} fill values given for {n_out} output band(s)."
            )
        inputs = feat_list + ([mask] if mask is not None else [])
        super().__init__(inputs=inputs, name=name)
        self.n_features_inputs = len(feat_list)
        self.has_mask = mask is not None
        self.compute = compute
        self.n_out = int(n_out)
        self.fill_values = np.asarray(fill_values, dtype=np.float64)
        self.estimator = estimator
        self.img_msk_val = img_msk_val
        self.scaler = scaler
        self.band_names = tuple(band_names) if band_names is not None else None
        self.nodata = nodata
        self.use_df = _needs_dataframe_input(estimator) if estimator is not None else False
        self._in_nodata: List[Optional[float]] = [None] * len(feat_list)
        self._df_cols: Optional[List[str]] = None

    def infer_meta(self, inputs: Sequence[NodeMeta]) -> NodeMeta:
        feat_metas = list(inputs[: self.n_features_inputs])
        self._in_nodata = [m.nodata for m in feat_metas]
        counts = [m.n_bands for m in feat_metas]
        if all(c is not None for c in counts):
            n_feat = sum(int(c) for c in counts if c is not None)
            n_expect = getattr(self.estimator, "n_features_in_", None)
            if isinstance(n_expect, (int, np.integer)) and int(n_expect) != n_feat:
                raise rsgispy.RSGISPyException(
                    f"{self.label}: the feature nodes have {n_feat} band(s) but "
                    f"the estimator expects {int(n_expect)} features."
                )
            names: List[str] = []
            for idx, meta in enumerate(feat_metas):
                if meta.band_names is not None:
                    names.extend(meta.band_names)
                else:
                    names.extend(
                        f"input{idx + 1}_b{b + 1}" for b in range(int(meta.n_bands or 0))
                    )
            if self.use_df:
                self._df_cols = _resolve_estimator_df_cols(self.estimator, names)
        return NodeMeta(
            n_bands=self.n_out,
            band_names=self.band_names,
            dtype=np.float64,
            nodata=self.nodata,
        )

    def _df_columns(self, n_feat: int) -> List[str]:
        if self._df_cols is None or len(self._df_cols) != n_feat:
            self._df_cols = _resolve_estimator_df_cols(
                self.estimator, [f"f{i + 1}" for i in range(n_feat)]
            )
        return self._df_cols

    def evaluate(
        self,
        inputs: Sequence[TileTensor],
        halo_y: int,
        halo_x: int,
        ctx: TileContext,
        state: Any = None,
    ) -> TileTensor:
        # pylint: disable=too-many-locals
        aligned = [t.crop(halo_y, halo_x) for t in inputs]
        height, width = aligned[0].data.shape[1:]
        if self.has_mask:
            valid = aligned[-1].data[0] == self.img_msk_val
        else:
            valid = np.ones((height, width), dtype=bool)
        bands: List[np.ndarray] = []
        for tensor, in_nd in zip(aligned[: self.n_features_inputs], self._in_nodata):
            for b_idx in range(tensor.data.shape[0]):
                band = tensor.data[b_idx]
                if np.any(valid):
                    valid &= build_valid_mask(band, in_nd)
                bands.append(band)
        out = np.empty((self.n_out, height * width), dtype=np.float64)
        out[:] = self.fill_values[:, np.newaxis]
        valid_idx = np.flatnonzero(valid.ravel())
        if valid_idx.size > 0:
            x_valid = np.empty((valid_idx.size, len(bands)), dtype=np.float64)
            for col, band in enumerate(bands):
                x_valid[:, col] = band.ravel()[valid_idx]
            if self.scaler is not None:
                x_valid = np.asarray(self.scaler.transform(x_valid), dtype=np.float64)
            x_in = (
                _model_input(x_valid, True, self._df_columns(len(bands)))
                if self.use_df
                else x_valid
            )
            vals = np.asarray(self.compute(x_in, x_valid), dtype=np.float64)
            if vals.ndim != 2 or vals.shape != (self.n_out, valid_idx.size):
                raise rsgispy.RSGISPyException(
                    f"{self.label}: the model returned {vals.shape[0]} output(s) "
                    f"per pixel but the node was built for {self.n_out}; pass "
                    "n_outputs to the node builder."
                )
            out[:, valid_idx] = vals
        # pylint: disable-next=too-many-function-args
        return aligned[0].with_data(out.reshape(self.n_out, height, width))

    def describe_params(self) -> str:
        return (
            f"estimator={type(self.estimator).__name__} outputs={self.n_out} "
            f"mask={'yes' if self.has_mask else 'no'}"
        )


def predict_raster_node(
    features: FeatureNodes,
    estimator: Any,
    scaler: Optional[FeatureScaler] = None,
    lut: Optional[Mapping[Any, Any]] = None,
    threshold: Optional[float] = None,
    thematic: bool = False,
    band_names: Optional[Sequence[str]] = None,
    *,
    mask: Optional[TileNode] = None,
    img_msk_val: int = 1,
    n_outputs: Optional[int] = None,
    no_data_val: Optional[float] = None,
    name: Optional[str] = None,
) -> TileNode:
    """
    Tile-graph node of :func:`predict_raster`: applies a fitted estimator
    (classifier, regressor or SDM) to every valid pixel of the feature nodes,
    for chaining with other operations (e.g. filters before, a majority
    filter after) in one :class:`~rsgispy.imageprocess.TileGraph` pass.

    The features are the bands of ``features``, in order (build them from an
    :class:`~rsgispy.ml.stack.ImageFeatureStack` with
    :func:`feature_stack_nodes`, or from any nodes). A pixel is predicted
    when it is inside ``mask`` (``mask == img_msk_val``, if given) and every
    feature is finite and not its input's no-data value; other pixels get
    ``no_data_val``. The node outputs ``float64`` blocks, one band per model
    output (``n_outputs``, else the estimator's ``n_outputs_``, else
    ``len(band_names)``, else 1).

    :param features: Feature node or sequence of feature nodes.
    :param estimator: Fitted estimator exposing ``predict`` (``predict_proba``
                      when ``threshold`` is given).
    :param scaler: Optional fitted :class:`~rsgispy.ml.scaling.FeatureScaler`
                   applied to the features (e.g. ``stack.scaler``).
    :param lut: Optional mapping of predicted labels to output values.
    :param threshold: Optional probability threshold in ``(0, 1)``: output
                      ``predict_proba[:, -1] >= threshold`` (binary).
    :param thematic: The output is a class map: the default ``no_data_val``
                     is ``0`` (Default = ``False``).
    :param band_names: Optional output band names.
    :param mask: Optional mask :class:`~rsgispy.imageprocess.TileNode`
                 (band 1 used).
    :param img_msk_val: Mask value of the pixels to predict (Default = ``1``).
    :param n_outputs: Number of model outputs (Default = ``None``, see above).
    :param no_data_val: Output value of pixels not predicted (Default =
                        ``0`` when ``thematic`` or ``threshold`` is set, else
                        ``-9999.0``, as :func:`predict_raster` with its
                        default data types).
    :param name: Optional node name (Default = ``'predict_raster'``).
    :return: Prediction :class:`~rsgispy.imageprocess.TileNode`.

    .. code:: python

        import numpy as np
        import rsgispy
        from rsgispy.imageprocess import DiskSink, RasterSource, TileGraph
        from rsgispy.classification.classpostprocess import majority_filter_node
        from rsgispy.ml.predict import feature_stack_nodes, predict_raster_node
        from rsgispy.ml.stack import ImageFeatureStack

        stack = ImageFeatureStack.load("stack.json")
        cls = predict_raster_node(
            feature_stack_nodes(stack),
            clf,
            scaler=stack.scaler,
            thematic=True,
            mask=RasterSource("valid_mask.tif", bands=1),
        )
        smooth = majority_filter_node(cls, 3, dtype=np.uint8, no_data_val=0)
        TileGraph(
            DiskSink(smooth, "landcover.tif", datatype=rsgispy.TYPE_8UINT)
        ).run()
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    if threshold is not None and not hasattr(estimator, "predict_proba"):
        raise rsgispy.RSGISPyException(
            "Estimator must implement predict_proba() when threshold is given."
        )
    is_thematic = bool(thematic or threshold is not None)
    eff_nd = (
        float(no_data_val)
        if no_data_val is not None
        else (0.0 if is_thematic else -9999.0)
    )
    n_out_attr = getattr(estimator, "n_outputs_", None)
    if n_outputs is not None:
        n_out = int(n_outputs)
    elif isinstance(n_out_attr, (int, np.integer)) and int(n_out_attr) > 0:
        n_out = int(n_out_attr)
    elif band_names:
        n_out = len(band_names)
    else:
        n_out = 1

    def _compute(x_in: Any, x_valid: np.ndarray) -> np.ndarray:
        del x_valid
        preds = _predict_labels(estimator, x_in, threshold, lut)
        if preds.ndim == 2 and preds.shape[1] != 1:
            return preds.T.astype(np.float64, copy=False)
        return preds.ravel().astype(np.float64, copy=False)[np.newaxis, :]

    return PredictionOp(
        features,
        _compute,
        n_out,
        [eff_nd] * n_out,
        estimator=estimator,
        mask=mask,
        img_msk_val=img_msk_val,
        scaler=scaler,
        band_names=band_names,
        nodata=eff_nd,
        name=name or "predict_raster",
    )


def predict_proba_raster_node(
    features: FeatureNodes,
    estimator: Any,
    scaler: Optional[FeatureScaler] = None,
    class_names: Optional[Sequence[str]] = None,
    lut: Optional[Mapping[Any, Any]] = None,
    scale_to_pct: bool = False,
    *,
    with_class: bool = False,
    with_entropy: bool = False,
    mask: Optional[TileNode] = None,
    img_msk_val: int = 1,
    n_classes: Optional[int] = None,
    no_data_val: Optional[float] = None,
    name: Optional[str] = None,
) -> TileNode:
    """
    Tile-graph node of :func:`predict_proba_raster`: per-class probabilities
    of a fitted classifier for every valid pixel of the feature nodes (see
    :func:`predict_raster_node` for the valid pixels and the features).

    The node has one band per class (``len(estimator.classes_)``, else
    ``n_classes``, else 2), named ``prob_<class>``; with ``with_class`` a
    band with the most probable class (mapped through ``lut``; ``0`` where
    not predicted) is appended, and with ``with_entropy`` a band with the
    normalised Shannon entropy (``-9999`` where not predicted), as the
    ``out_class_img`` / ``out_entropy_img`` outputs of
    :func:`predict_proba_raster`. Route the bands to separate sinks with
    :class:`~rsgispy.imageprocess.SelectBands`; the model runs once per tile.

    :param features: Feature node or sequence of feature nodes.
    :param estimator: Fitted classifier implementing ``predict_proba``.
    :param scaler: Optional fitted :class:`~rsgispy.ml.scaling.FeatureScaler`.
    :param class_names: Optional class names for the probability band names.
    :param lut: Optional mapping of class labels to output IDs (class band).
    :param scale_to_pct: Scale probabilities to ``[0, 100]``
                         (Default = ``False``).
    :param with_class: Append the most probable class band
                       (Default = ``False``).
    :param with_entropy: Append the entropy band (Default = ``False``).
    :param mask: Optional mask :class:`~rsgispy.imageprocess.TileNode`
                 (band 1 used).
    :param img_msk_val: Mask value of the pixels to predict (Default = ``1``).
    :param n_classes: Number of classes when the estimator has no
                      ``classes_`` (Default = ``None``).
    :param no_data_val: Probability value of pixels not predicted
                        (Default = ``-9999.0``; :func:`predict_proba_raster`
                        uses ``255`` for ``uint8`` outputs).
    :param name: Optional node name (Default = ``'predict_proba_raster'``).
    :return: Probability :class:`~rsgispy.imageprocess.TileNode`.

    .. code:: python

        import rsgispy
        from rsgispy.imageprocess import DiskSink, SelectBands, TileGraph
        from rsgispy.ml.predict import feature_stack_nodes, predict_proba_raster_node

        probs = predict_proba_raster_node(
            feature_stack_nodes(stack), clf, with_class=True, with_entropy=True
        )
        n_cls = len(clf.classes_)
        TileGraph(
            [
                DiskSink(SelectBands(probs, list(range(1, n_cls + 1))), "probs.tif"),
                DiskSink(SelectBands(probs, n_cls + 1), "class.tif",
                         datatype=rsgispy.TYPE_32INT, nodata=0),
                DiskSink(SelectBands(probs, n_cls + 2), "entropy.tif", nodata=-9999),
            ]
        ).run()
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    if not hasattr(estimator, "predict_proba"):
        raise rsgispy.RSGISPyException(
            "Estimator must implement predict_proba() for predict_proba_raster_node."
        )
    est_classes = getattr(estimator, "classes_", None)
    if est_classes is not None:
        n_cls = len(est_classes)
    else:
        n_cls = int(n_classes) if n_classes is not None else 2
    eff_nd = -9999.0 if no_data_val is None else float(no_data_val)
    scale_mult = 100.0 if scale_to_pct else 1.0

    names: Optional[List[str]] = None
    if class_names is not None:
        names = [f"prob_{c}" for c in class_names]
    elif est_classes is not None:
        names = [f"prob_{c}" for c in est_classes]
    fills = [eff_nd] * n_cls
    if with_class:
        fills.append(0.0)
    if with_entropy:
        fills.append(-9999.0)
    if names is not None:
        if len(names) != n_cls:
            raise rsgispy.RSGISPyException(
                f"{len(names)} class names given for {n_cls} classes."
            )
        names += (["predicted_class"] if with_class else []) + (
            ["shannon_entropy"] if with_entropy else []
        )

    def _compute(x_in: Any, x_valid: np.ndarray) -> np.ndarray:
        del x_valid
        probs = _predict_proba_2d(estimator, x_in)
        if probs.shape[1] != n_cls:
            raise rsgispy.RSGISPyException(
                f"The classifier returned {probs.shape[1]} class probabilities, "
                f"expected {n_cls} (pass n_classes)."
            )
        rows = [(probs * scale_mult).T]
        if with_class:
            rows.append(
                _proba_class_labels(probs, est_classes, lut).astype(np.float64)[np.newaxis]
            )
        if with_entropy:
            rows.append(_proba_entropy(probs, n_cls)[np.newaxis])
        return np.concatenate(rows, axis=0)

    return PredictionOp(
        features,
        _compute,
        len(fills),
        fills,
        estimator=estimator,
        mask=mask,
        img_msk_val=img_msk_val,
        scaler=scaler,
        band_names=names,
        nodata=eff_nd,
        name=name or "predict_proba_raster",
    )


def predict_quantiles_raster_node(
    features: FeatureNodes,
    estimators_or_model: Any,
    quantiles: Sequence[float] = (0.05, 0.5, 0.95),
    scaler: Optional[FeatureScaler] = None,
    conformal_residuals: Optional[Sequence[float]] = None,
    alpha: float = 0.1,
    *,
    mask: Optional[TileNode] = None,
    img_msk_val: int = 1,
    no_data_val: Optional[float] = None,
    name: Optional[str] = None,
) -> TileNode:
    """
    Tile-graph node of :func:`predict_quantiles_raster`: per-pixel
    regression quantiles or conformal prediction intervals (one band per
    quantile, named as by :func:`predict_quantiles_raster`) for every valid
    pixel of the feature nodes (see :func:`predict_raster_node` for the
    valid pixels and the features). The three modes of
    :func:`predict_quantiles_raster` are supported: a ``{quantile: model}``
    mapping, a point regressor with ``conformal_residuals``, or a
    bagging/forest ensemble (empirical quantiles across its trees).

    :param features: Feature node or sequence of feature nodes.
    :param estimators_or_model: ``{quantile: model}`` mapping, tree ensemble,
                                or point regressor (with
                                ``conformal_residuals``).
    :param quantiles: Quantiles in ``(0, 1)`` for an ensemble
                      (Default = ``(0.05, 0.5, 0.95)``).
    :param scaler: Optional fitted :class:`~rsgispy.ml.scaling.FeatureScaler`.
    :param conformal_residuals: Optional calibration absolute residuals for
                                split-conformal intervals.
    :param alpha: Miscoverage level in ``(0, 1)`` for conformal intervals
                  (Default = ``0.1``).
    :param mask: Optional mask :class:`~rsgispy.imageprocess.TileNode`
                 (band 1 used).
    :param img_msk_val: Mask value of the pixels to predict (Default = ``1``).
    :param no_data_val: Output value of pixels not predicted
                        (Default = ``-9999.0``).
    :param name: Optional node name (Default = ``'predict_quantiles_raster'``).
    :return: Quantile :class:`~rsgispy.imageprocess.TileNode`.

    .. code:: python

        from rsgispy.imageprocess import BandMathOp, DiskSink, SelectBands, TileGraph
        from rsgispy.ml.predict import feature_stack_nodes, predict_quantiles_raster_node

        q = predict_quantiles_raster_node(
            feature_stack_nodes(stack), rf_reg, quantiles=(0.1, 0.9)
        )
        # Width of the 80% interval, without writing the quantiles.
        width = BandMathOp(
            {"lo": SelectBands(q, 1), "hi": SelectBands(q, 2)}, "hi - lo", nodata=-9999
        )
        TileGraph(DiskSink(width, "interval_width.tif")).run()
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    q_pred = _QuantilePredictor(
        estimators_or_model, quantiles, conformal_residuals, alpha
    )
    eff_nd = -9999.0 if no_data_val is None else float(no_data_val)
    n_out = len(q_pred.band_names)
    return PredictionOp(
        features,
        q_pred.predict,
        n_out,
        [eff_nd] * n_out,
        estimator=q_pred.ref_est,
        mask=mask,
        img_msk_val=img_msk_val,
        scaler=scaler,
        band_names=q_pred.band_names,
        nodata=eff_nd,
        name=name or "predict_quantiles_raster",
    )
