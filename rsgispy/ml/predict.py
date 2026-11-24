"""
Unified tiled raster prediction engine (:func:`predict_raster`,
:func:`predict_proba_raster`, :func:`predict_quantiles_raster`) shared across
:mod:`rsgispy.classification`, :mod:`rsgispy.regression`, and
:mod:`rsgispy.modelling`.

Supports any fitted estimator exposing ``predict`` / ``predict_proba`` (e.g.,
``scikit-learn``, ``XGBoost``, ``LightGBM``, ``CatBoost``, or custom pipelines),
on-the-fly valid-pixel masking (``in_msk_img=None``), and on-the-fly
:class:`~rsgispy.ml.scaling.FeatureScaler` feature scaling.
"""

from contextlib import ExitStack
import os
from typing import (
    Any,
    List,
    Mapping,
    Optional,
    Sequence,
    Union,
)

import numpy as np
import pandas as pd

import rsgispy
from rsgispy.imageprocess import (
    _INT_DTYPES,
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
    # pylint: disable=too-many-locals,too-many-branches
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
                x_in: Any = (
                    pd.DataFrame(x_valid, columns=df_cols) if use_df else x_valid
                )
                if threshold is not None:
                    probs = np.asarray(
                        estimator.predict_proba(x_in), dtype=np.float64
                    )
                    pos_prob = probs[:, -1] if probs.ndim == 2 else probs
                    preds = (pos_prob >= float(threshold)).astype(np.int64)
                else:
                    preds = np.asarray(estimator.predict(x_in))
                if lut is not None:
                    preds = _apply_lut_1d(preds.ravel(), lut).reshape(
                        preds.shape
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
                x_in: Any = (
                    pd.DataFrame(x_valid, columns=df_cols) if use_df else x_valid
                )
                probs = np.asarray(estimator.predict_proba(x_in), dtype=np.float64)
                if probs.ndim == 1:
                    probs = np.column_stack([1.0 - probs, probs])
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
                    argm = np.argmax(probs, axis=1)
                    labels = (
                        np.asarray(est_classes)[argm]
                        if est_classes is not None
                        else argm
                    )
                    if lut is not None:
                        labels = _apply_lut_1d(labels, lut)
                    cls_flat[valid_idx] = labels.astype(np.int64, copy=False)
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
                    p_clip = np.clip(probs, 1e-12, 1.0)
                    denom = np.log(float(max(eff_n_cls, 2)))
                    ent = -np.sum(p_clip * np.log(p_clip), axis=1) / denom
                    ent_flat[valid_idx] = ent
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

    is_mapping = isinstance(estimators_or_model, Mapping)
    is_conformal = conformal_residuals is not None
    q_hat = 0.0
    if is_mapping:
        if not estimators_or_model:
            raise rsgispy.RSGISPyException(
                "estimators_or_model mapping must not be empty."
            )
        q_keys = sorted(float(k) for k in estimators_or_model.keys())
        q_models = [estimators_or_model[k] for k in sorted(estimators_or_model.keys())]
        band_names = [f"q{int(round(q * 100)):02d}" for q in q_keys]
        ref_est = q_models[0]
    elif is_conformal:
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
        q_hat = float(np.quantile(res_arr, q_level, method="higher"))
        cov_pct = int(round((1.0 - float(alpha)) * 100))
        band_names = [
            f"lower_{cov_pct}pct",
            "prediction",
            f"upper_{cov_pct}pct",
        ]
        ref_est = estimators_or_model
    elif hasattr(estimators_or_model, "estimators_"):
        sub_ests = estimators_or_model.estimators_
        is_boosted = (
            isinstance(sub_ests, np.ndarray)
            or hasattr(estimators_or_model, "init_")
            or hasattr(estimators_or_model, "learning_rate")
            or hasattr(estimators_or_model, "estimator_weights_")
        )
        if is_boosted or not sub_ests or not all(hasattr(e, "predict") for e in sub_ests):
            raise rsgispy.RSGISPyException(
                "Empirical quantile prediction via .estimators_ is only supported "
                "for independent bagging/forest ensembles (e.g., RandomForestRegressor, "
                "ExtraTreesRegressor), not boosted ensembles such as "
                f"{type(estimators_or_model).__name__}. Pass a {{quantile: model}} "
                "mapping or conformal_residuals instead."
            )
        q_list = [float(q) for q in quantiles]
        if not q_list or any(q <= 0.0 or q >= 1.0 for q in q_list):
            raise rsgispy.RSGISPyException(
                "quantiles must be a non-empty sequence of values in (0, 1)."
            )
        band_names = [f"q{int(round(q * 100)):02d}" for q in q_list]
        ref_est = estimators_or_model
    else:
        raise rsgispy.RSGISPyException(
            "estimators_or_model must be a {quantile: model} mapping, a tree "
            "ensemble with .estimators_, or a regressor with conformal_residuals."
        )

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
                x_in: Any = (
                    pd.DataFrame(x_valid, columns=df_cols) if use_df else x_valid
                )
                if is_mapping:
                    for b_idx, mdl in enumerate(q_models):
                        flat_out[b_idx, valid_idx] = np.asarray(
                            mdl.predict(x_in), dtype=np.float64
                        ).ravel()
                elif is_conformal:
                    y_hat = np.asarray(
                        estimators_or_model.predict(x_in), dtype=np.float64
                    ).ravel()
                    flat_out[0, valid_idx] = y_hat - q_hat
                    flat_out[1, valid_idx] = y_hat
                    flat_out[2, valid_idx] = y_hat + q_hat
                else:
                    tree_preds = np.vstack(
                        [
                            np.asarray(est.predict(x_valid), dtype=np.float64).ravel()
                            for est in estimators_or_model.estimators_
                        ]
                    )
                    q_vals = np.quantile(tree_preds, q_list, axis=0)
                    flat_out[:, valid_idx] = q_vals

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
