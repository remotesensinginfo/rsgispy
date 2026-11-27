"""
Single-model and ensemble block-wise spatial raster prediction routines for
:mod:`rsgispy.modelling.species_distribution`.
"""

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

import rsgispy
from rsgispy.modelling._helpers import (
    EnvVarInfo,
    calc_single_class_predictions,
    get_positive_class_probs,
    predict_raster_from_env_vars,
)

__all__ = [
    "pred_sklearn_mdl_prob",
    "pred_sklearn_mdl_cls",
    "pred_sklearn_sgl_cls_mdl_prob",
    "pred_sklearn_sgl_cls_mdl_cls",
    "pred_ensemble_sklearn_mdls_prob",
    "pred_ensemble_sklearn_mdls_cls",
    "pred_ensemble_sklearn_sgl_cls_mdls_prob",
    "pred_ensemble_sklearn_sgl_cls_mdls_cls",
]


def _score_single_class_samples(
    est_cls_obj: Any, env_df: pd.DataFrame
) -> np.ndarray:
    """Return continuous single-class score values (``score_samples`` or fallback)."""
    if hasattr(est_cls_obj, "score_samples"):
        return np.asarray(est_cls_obj.score_samples(env_df), dtype=np.float64)
    return np.asarray(est_cls_obj.decision_function(env_df), dtype=np.float64)


def _validate_uint8_nodata(out_no_data_val: int) -> float:
    """Validate a ``uint8`` thematic no-data fill value in ``[0, 255]``."""
    if isinstance(out_no_data_val, bool) or not 0 <= int(out_no_data_val) <= 255:
        raise rsgispy.RSGISPyException(
            f"out_no_data_val must be an integer in [0, 255], got {out_no_data_val!r}."
        )
    return float(int(out_no_data_val))


def _ens_cv(stack: np.ndarray) -> np.ndarray:
    mean = np.mean(stack, axis=0)
    std = np.std(stack, axis=0)
    return np.divide(std, np.abs(mean), out=np.zeros_like(std), where=mean != 0)


_UNCERTAINTY_FNS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "std": lambda stack: np.std(stack, axis=0),
    "iqr": lambda stack: np.subtract(*np.percentile(stack, [75.0, 25.0], axis=0)),
    "min": lambda stack: np.min(stack, axis=0),
    "max": lambda stack: np.max(stack, axis=0),
    "cv": _ens_cv,
}

def _validate_uncertainty(uncertainty: Optional[Sequence[str]]) -> List[str]:
    if uncertainty is None:
        return []
    if isinstance(uncertainty, str):
        uncertainty = [uncertainty]
    stats = [str(stat).strip().lower() for stat in uncertainty]
    unknown = [stat for stat in stats if stat not in _UNCERTAINTY_FNS]
    if unknown:
        raise rsgispy.RSGISPyException(
            f"Unsupported uncertainty statistic(s) {unknown}; use any of "
            f"{list(_UNCERTAINTY_FNS)}."
        )
    if len(set(stats)) != len(stats):
        raise rsgispy.RSGISPyException("uncertainty statistics must be unique.")
    return stats


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
def _pred_ensemble_sklearn_mdls(
    est_cls_objs: Sequence[Any],
    in_msk_img: str,
    img_msk_val: int,
    env_vars: Mapping[str, EnvVarInfo],
    analysis_vars: Sequence[str],
    output_img: str,
    *,
    model_predict_fn: Callable[[Any, pd.DataFrame], np.ndarray],
    agg_fn: Callable[[np.ndarray], np.ndarray],
    gdalformat: str = "GTIFF",
    tmp_dir: Optional[str] = None,
    normalise_data: bool = False,
    calc_stats: bool = True,
    out_no_data_val: float = -1.0,
    block_size: Optional[int] = None,
    use_pbar: bool = True,
    fixed_var_vals: Optional[Mapping[str, float]] = None,
    uncertainty: Optional[Sequence[str]] = None,
    main_band_name: str = "median",
) -> None:
    """Shared streaming block-prediction engine for ensemble SDM functions.
    With ``uncertainty``, extra bands of across-member statistics follow the
    aggregated band, all computed from the same per-block prediction stack."""
    _ = tmp_dir  # Retained for legacy signature compatibility; streaming needs no tmp.
    if not est_cls_objs:
        raise rsgispy.RSGISPyException("est_cls_objs must not be empty.")
    unc_stats = _validate_uncertainty(uncertainty)

    nd_val = float(out_no_data_val)

    def _ensemble_predict(env_df: pd.DataFrame) -> np.ndarray:
        stack = np.vstack([model_predict_fn(mdl, env_df) for mdl in est_cls_objs])
        if not unc_stats:
            return agg_fn(stack)
        return np.vstack(
            [agg_fn(stack)] + [_UNCERTAINTY_FNS[stat](stack) for stat in unc_stats]
        )

    predict_raster_from_env_vars(
        in_msk_img=in_msk_img,
        img_msk_val=img_msk_val,
        env_vars=env_vars,
        analysis_vars=analysis_vars,
        output_img=output_img,
        predict_fn=_ensemble_predict,
        gdalformat=gdalformat,
        out_rio_dtype="float32",
        fill_val=nd_val,
        normalise_data=normalise_data,
        calc_stats=calc_stats,
        thematic=False,
        no_data_val=nd_val,
        block_size=block_size,
        use_pbar=use_pbar,
        fixed_var_vals=fixed_var_vals,
        out_band_names=[main_band_name] + unc_stats if unc_stats else None,
    )


# pylint: disable=too-many-arguments,too-many-positional-arguments
def pred_sklearn_mdl_prob(
    est_cls_obj: Any,
    in_msk_img: str,
    img_msk_val: int,
    env_vars: Mapping[str, EnvVarInfo],
    analysis_vars: Sequence[str],
    output_img: str,
    *,
    gdalformat: str = "GTIFF",
    normalise_data: bool = False,
    calc_stats: bool = True,
    block_size: Optional[int] = None,
    use_pbar: bool = True,
    fixed_var_vals: Optional[Mapping[str, float]] = None,
) -> None:
    """
    Predict spatial probability of species presence (``float32``, nodata ``-1.0``)
    from a fitted ``scikit-learn`` classifier using block-wise streaming via
    :mod:`rsgispy.imageprocess`.

    :param est_cls_obj: Fitted ``scikit-learn`` classifier with ``predict_proba``.
    :param in_msk_img: Path to ROI mask image.
    :param img_msk_val: Pixel value inside ``in_msk_img`` defining the ROI.
    :param env_vars: Mapping of environmental variable descriptors.
    :param analysis_vars: Ordered covariate names matching training columns.
    :param output_img: Output single-band probability image path.
    :param gdalformat: Output GDAL/rasterio format (Default = ``'GTIFF'``).
    :param normalise_data: Apply z-score normalisation from ``env_vars`` when ``True``.
    :param calc_stats: Compute output image statistics and pyramids (Default = ``True``).
    :param block_size: Optional block size in pixels for :mod:`rsgispy.imageprocess`.
    :param use_pbar: Display a tqdm progress bar (Default = ``True``).
    :param fixed_var_vals: Optional mapping of variable name to a constant used
                           for every pixel instead of reading that variable's
                           raster, e.g. to hold a sampling-bias covariate
                           constant (see :func:`predict_raster_from_env_vars`).

    .. code:: python

        import rsgispy.modelling.species_distribution as sdm

        sdm.pred_sklearn_mdl_prob(
            clf, "valid_mask.tif", 1, env_vars, ["v1", "v2"], "prob.tif"
        )
    """
    predict_raster_from_env_vars(
        in_msk_img=in_msk_img,
        img_msk_val=img_msk_val,
        env_vars=env_vars,
        analysis_vars=analysis_vars,
        output_img=output_img,
        predict_fn=lambda df: get_positive_class_probs(est_cls_obj, df, pos_label=1),
        gdalformat=gdalformat,
        out_rio_dtype="float32",
        fill_val=-1.0,
        normalise_data=normalise_data,
        calc_stats=calc_stats,
        thematic=False,
        no_data_val=-1.0,
        block_size=block_size,
        use_pbar=use_pbar,
        fixed_var_vals=fixed_var_vals,
    )


# pylint: disable=too-many-arguments,too-many-positional-arguments
def pred_sklearn_mdl_cls(
    est_cls_obj: Any,
    in_msk_img: str,
    img_msk_val: int,
    env_vars: Mapping[str, EnvVarInfo],
    analysis_vars: Sequence[str],
    output_img: str,
    *,
    gdalformat: str = "GTIFF",
    normalise_data: bool = False,
    calc_stats: bool = True,
    block_size: Optional[int] = None,
    use_pbar: bool = True,
    out_no_data_val: int = 0,
    fixed_var_vals: Optional[Mapping[str, float]] = None,
) -> None:
    """
    Predict binary presence/absence thematic classification (``uint8``) from a
    fitted ``scikit-learn`` classifier using :mod:`rsgispy.imageprocess`.

    :param est_cls_obj: Fitted ``scikit-learn`` classifier with ``predict``.
    :param in_msk_img: Path to ROI mask image.
    :param img_msk_val: Pixel value inside ``in_msk_img`` defining the ROI.
    :param env_vars: Mapping of environmental variable descriptors.
    :param analysis_vars: Ordered covariate names matching training columns.
    :param output_img: Output thematic classification image path.
    :param gdalformat: Output GDAL/rasterio format (Default = ``'GTIFF'``).
    :param normalise_data: Apply z-score normalisation from ``env_vars`` when ``True``.
    :param calc_stats: Populate thematic statistics and colour table (Default = ``True``).
    :param block_size: Optional block size in pixels for :mod:`rsgispy.imageprocess`.
    :param use_pbar: Display a tqdm progress bar (Default = ``True``).
    :param fixed_var_vals: Optional mapping of variable name to a constant used
                           for every pixel instead of reading that variable's
                           raster, e.g. to hold a sampling-bias covariate
                           constant (see :func:`predict_raster_from_env_vars`).
    :param out_no_data_val: Output ``uint8`` no-data fill value outside the
                            valid mask (Default = ``0`` for backwards
                            compatibility; set to ``255`` to distinguish
                            no-data from absence class ``0``).

    .. code:: python

        import rsgispy.modelling.species_distribution as sdm

        sdm.pred_sklearn_mdl_cls(
            clf, "valid_mask.tif", 1, env_vars, ["v1", "v2"], "cls.tif",
            out_no_data_val=255,
        )
    """
    nd_val = _validate_uint8_nodata(out_no_data_val)
    predict_raster_from_env_vars(
        in_msk_img=in_msk_img,
        img_msk_val=img_msk_val,
        env_vars=env_vars,
        analysis_vars=analysis_vars,
        output_img=output_img,
        predict_fn=est_cls_obj.predict,
        gdalformat=gdalformat,
        out_rio_dtype="uint8",
        fill_val=nd_val,
        normalise_data=normalise_data,
        calc_stats=calc_stats,
        thematic=True,
        no_data_val=nd_val,
        block_size=block_size,
        use_pbar=use_pbar,
        fixed_var_vals=fixed_var_vals,
    )


# pylint: disable=too-many-arguments,too-many-positional-arguments
def pred_sklearn_sgl_cls_mdl_prob(
    est_cls_obj: Any,
    in_msk_img: str,
    img_msk_val: int,
    env_vars: Mapping[str, EnvVarInfo],
    analysis_vars: Sequence[str],
    output_img: str,
    *,
    gdalformat: str = "GTIFF",
    normalise_data: bool = False,
    calc_stats: bool = True,
    out_no_data_val: float = -9999.0,
    block_size: Optional[int] = None,
    use_pbar: bool = True,
    fixed_var_vals: Optional[Mapping[str, float]] = None,
) -> None:
    """
    Predict continuous habitat suitability/density scores from a fitted
    single-class ``scikit-learn`` estimator using :mod:`rsgispy.imageprocess`.

    :param est_cls_obj: Fitted single-class estimator (with ``score_samples`` or
                        ``decision_function``).
    :param in_msk_img: Path to ROI mask image.
    :param img_msk_val: Pixel value inside ``in_msk_img`` defining the ROI.
    :param env_vars: Mapping of environmental variable descriptors.
    :param analysis_vars: Ordered covariate names matching training columns.
    :param output_img: Output ``float32`` score image path.
    :param gdalformat: Output GDAL/rasterio format (Default = ``'GTIFF'``).
    :param normalise_data: Apply z-score normalisation from ``env_vars`` when ``True``.
    :param calc_stats: Compute output image statistics and pyramids (Default = ``True``).
    :param out_no_data_val: No-data fill value outside mask (Default = ``-9999.0``).
    :param block_size: Optional block size in pixels for :mod:`rsgispy.imageprocess`.
    :param use_pbar: Display a tqdm progress bar (Default = ``True``).
    :param fixed_var_vals: Optional mapping of variable name to a constant used
                           for every pixel instead of reading that variable's
                           raster, e.g. to hold a sampling-bias covariate
                           constant (see :func:`predict_raster_from_env_vars`).

    .. code:: python

        import rsgispy.modelling.species_distribution as sdm

        sdm.pred_sklearn_sgl_cls_mdl_prob(
            oc_mdl, "valid_mask.tif", 1, env_vars, ["v1", "v2"], "oc_score.tif"
        )
    """
    predict_raster_from_env_vars(
        in_msk_img=in_msk_img,
        img_msk_val=img_msk_val,
        env_vars=env_vars,
        analysis_vars=analysis_vars,
        output_img=output_img,
        predict_fn=lambda df: _score_single_class_samples(est_cls_obj, df),
        gdalformat=gdalformat,
        out_rio_dtype="float32",
        fill_val=float(out_no_data_val),
        normalise_data=normalise_data,
        calc_stats=calc_stats,
        thematic=False,
        no_data_val=float(out_no_data_val),
        block_size=block_size,
        use_pbar=use_pbar,
        fixed_var_vals=fixed_var_vals,
    )


# pylint: disable=too-many-arguments,too-many-positional-arguments
def pred_sklearn_sgl_cls_mdl_cls(
    est_cls_obj: Any,
    in_msk_img: str,
    img_msk_val: int,
    env_vars: Mapping[str, EnvVarInfo],
    analysis_vars: Sequence[str],
    output_img: str,
    *,
    gdalformat: str = "GTIFF",
    normalise_data: bool = False,
    calc_stats: bool = True,
    block_size: Optional[int] = None,
    use_pbar: bool = True,
    out_no_data_val: int = 0,
    fixed_var_vals: Optional[Mapping[str, float]] = None,
) -> None:
    """
    Predict binary presence (``1``) vs absence (``0``) thematic map from a
    fitted single-class ``scikit-learn`` estimator using :mod:`rsgispy.imageprocess`.

    :param est_cls_obj: Fitted single-class estimator with ``predict``.
    :param in_msk_img: Path to ROI mask image.
    :param img_msk_val: Pixel value inside ``in_msk_img`` defining the ROI.
    :param env_vars: Mapping of environmental variable descriptors.
    :param analysis_vars: Ordered covariate names matching training columns.
    :param output_img: Output thematic classification image path.
    :param gdalformat: Output GDAL/rasterio format (Default = ``'GTIFF'``).
    :param normalise_data: Apply z-score normalisation from ``env_vars`` when ``True``.
    :param calc_stats: Populate thematic statistics and colour table (Default = ``True``).
    :param block_size: Optional block size in pixels for :mod:`rsgispy.imageprocess`.
    :param use_pbar: Display a tqdm progress bar (Default = ``True``).
    :param fixed_var_vals: Optional mapping of variable name to a constant used
                           for every pixel instead of reading that variable's
                           raster, e.g. to hold a sampling-bias covariate
                           constant (see :func:`predict_raster_from_env_vars`).
    :param out_no_data_val: Output ``uint8`` no-data fill value outside the
                            valid mask (Default = ``0`` for backwards
                            compatibility; set to ``255`` to distinguish
                            no-data from absence class ``0``).

    .. code:: python

        import rsgispy.modelling.species_distribution as sdm

        sdm.pred_sklearn_sgl_cls_mdl_cls(
            oc_mdl, "valid_mask.tif", 1, env_vars, ["v1", "v2"], "oc_cls.tif",
            out_no_data_val=255,
        )
    """
    nd_val = _validate_uint8_nodata(out_no_data_val)
    predict_raster_from_env_vars(
        in_msk_img=in_msk_img,
        img_msk_val=img_msk_val,
        env_vars=env_vars,
        analysis_vars=analysis_vars,
        output_img=output_img,
        predict_fn=lambda df: calc_single_class_predictions(est_cls_obj, df),
        gdalformat=gdalformat,
        out_rio_dtype="uint8",
        fill_val=nd_val,
        normalise_data=normalise_data,
        calc_stats=calc_stats,
        thematic=True,
        no_data_val=nd_val,
        block_size=block_size,
        use_pbar=use_pbar,
        fixed_var_vals=fixed_var_vals,
    )


# pylint: disable=too-many-arguments,too-many-positional-arguments
def pred_ensemble_sklearn_mdls_prob(
    est_cls_objs: Sequence[Any],
    in_msk_img: str,
    img_msk_val: int,
    env_vars: Mapping[str, EnvVarInfo],
    analysis_vars: Sequence[str],
    output_img: str,
    *,
    gdalformat: str = "GTIFF",
    tmp_dir: Optional[str] = None,
    normalise_data: bool = False,
    calc_stats: bool = True,
    block_size: Optional[int] = None,
    use_pbar: bool = True,
    fixed_var_vals: Optional[Mapping[str, float]] = None,
    uncertainty: Optional[Sequence[str]] = None,
) -> None:
    """
    Predict ensemble species presence probability across ``est_cls_objs`` by
    computing the median probability per pixel in a single streaming pass via
    :mod:`rsgispy.imageprocess` (writing no intermediate raster files). With
    ``uncertainty``, bands describing the spread of the members' probabilities
    (e.g. standard deviation, interquartile range) follow the median band, so
    the map's uncertainty can be shown alongside it.

    :param est_cls_objs: Sequence of fitted ``scikit-learn`` classifiers.
    :param in_msk_img: Path to ROI mask image.
    :param img_msk_val: Pixel value inside ``in_msk_img`` defining the ROI.
    :param env_vars: Mapping of environmental variable descriptors.
    :param analysis_vars: Ordered covariate names matching training columns.
    :param output_img: Output ``float32`` ensemble probability image path.
    :param gdalformat: Output GDAL/rasterio format (Default = ``'GTIFF'``).
    :param tmp_dir: Unused legacy parameter retained for signature compatibility.
    :param normalise_data: Apply z-score normalisation from ``env_vars`` when ``True``.
    :param calc_stats: Compute output image statistics and pyramids (Default = ``True``).
    :param block_size: Optional block size in pixels for :mod:`rsgispy.imageprocess`.
    :param use_pbar: Display a tqdm progress bar (Default = ``True``).
    :param fixed_var_vals: Optional mapping of variable name to a constant used
                           for every pixel instead of reading that variable's
                           raster, e.g. to hold a sampling-bias covariate
                           constant (see :func:`predict_raster_from_env_vars`).
    :param uncertainty: Optional across-member statistics written as extra
                        bands after the median, in the order given: any of
                        ``"std"`` (standard deviation), ``"iqr"``
                        (interquartile range), ``"min"``, ``"max"`` and
                        ``"cv"`` (coefficient of variation, ``std / |mean|``;
                        ``0`` where the mean is ``0``). Band descriptions name
                        each band. Default ``None`` writes the median only.

    .. code:: python

        import rsgispy.modelling.species_distribution as sdm

        sdm.pred_ensemble_sklearn_mdls_prob(
            mdls, "valid_mask.tif", 1, env_vars, ["v1", "v2"], "ens_prob.tif"
        )
    """
    _pred_ensemble_sklearn_mdls(
        est_cls_objs=est_cls_objs,
        in_msk_img=in_msk_img,
        img_msk_val=img_msk_val,
        env_vars=env_vars,
        analysis_vars=analysis_vars,
        output_img=output_img,
        model_predict_fn=lambda mdl, df: get_positive_class_probs(
            mdl, df, pos_label=1
        ),
        agg_fn=lambda stack: np.median(stack, axis=0),
        gdalformat=gdalformat,
        tmp_dir=tmp_dir,
        normalise_data=normalise_data,
        calc_stats=calc_stats,
        out_no_data_val=-1.0,
        block_size=block_size,
        use_pbar=use_pbar,
        fixed_var_vals=fixed_var_vals,
        uncertainty=uncertainty,
        main_band_name="median",
    )


# pylint: disable=too-many-arguments,too-many-positional-arguments
def pred_ensemble_sklearn_mdls_cls(
    est_cls_objs: Sequence[Any],
    in_msk_img: str,
    img_msk_val: int,
    env_vars: Mapping[str, EnvVarInfo],
    analysis_vars: Sequence[str],
    output_img: str,
    *,
    gdalformat: str = "GTIFF",
    tmp_dir: Optional[str] = None,
    normalise_data: bool = False,
    calc_stats: bool = True,
    block_size: Optional[int] = None,
    use_pbar: bool = True,
    fixed_var_vals: Optional[Mapping[str, float]] = None,
) -> None:
    """
    Predict the proportion of ensemble classifiers voting presence (in ``[0, 1]``,
    nodata ``-1.0`` outside mask) in a single streaming pass via
    :mod:`rsgispy.imageprocess`.

    :param est_cls_objs: Sequence of fitted ``scikit-learn`` classifiers.
    :param in_msk_img: Path to ROI mask image.
    :param img_msk_val: Pixel value inside ``in_msk_img`` defining the ROI.
    :param env_vars: Mapping of environmental variable descriptors.
    :param analysis_vars: Ordered covariate names matching training columns.
    :param output_img: Output ``float32`` ensemble vote proportion image path.
    :param gdalformat: Output GDAL/rasterio format (Default = ``'GTIFF'``).
    :param tmp_dir: Unused legacy parameter retained for signature compatibility.
    :param normalise_data: Apply z-score normalisation from ``env_vars`` when ``True``.
    :param calc_stats: Compute output image statistics and pyramids (Default = ``True``).
    :param block_size: Optional block size in pixels for :mod:`rsgispy.imageprocess`.
    :param use_pbar: Display a tqdm progress bar (Default = ``True``).
    :param fixed_var_vals: Optional mapping of variable name to a constant used
                           for every pixel instead of reading that variable's
                           raster, e.g. to hold a sampling-bias covariate
                           constant (see :func:`predict_raster_from_env_vars`).

    .. code:: python

        import rsgispy.modelling.species_distribution as sdm

        sdm.pred_ensemble_sklearn_mdls_cls(
            mdls, "valid_mask.tif", 1, env_vars, ["v1", "v2"], "ens_cls.tif"
        )
    """
    _pred_ensemble_sklearn_mdls(
        est_cls_objs=est_cls_objs,
        in_msk_img=in_msk_img,
        img_msk_val=img_msk_val,
        env_vars=env_vars,
        analysis_vars=analysis_vars,
        output_img=output_img,
        model_predict_fn=lambda mdl, df: (
            np.asarray(mdl.predict(df)) > 0
        ).astype(np.float64),
        agg_fn=lambda stack: np.mean(stack, axis=0),
        gdalformat=gdalformat,
        tmp_dir=tmp_dir,
        normalise_data=normalise_data,
        calc_stats=calc_stats,
        out_no_data_val=-1.0,
        block_size=block_size,
        use_pbar=use_pbar,
        fixed_var_vals=fixed_var_vals,
    )


# pylint: disable=too-many-arguments,too-many-positional-arguments
def pred_ensemble_sklearn_sgl_cls_mdls_prob(
    est_cls_objs: Sequence[Any],
    in_msk_img: str,
    img_msk_val: int,
    env_vars: Mapping[str, EnvVarInfo],
    analysis_vars: Sequence[str],
    output_img: str,
    *,
    gdalformat: str = "GTIFF",
    tmp_dir: Optional[str] = None,
    normalise_data: bool = False,
    calc_stats: bool = True,
    out_no_data_val: float = -9999.0,
    block_size: Optional[int] = None,
    use_pbar: bool = True,
    fixed_var_vals: Optional[Mapping[str, float]] = None,
    uncertainty: Optional[Sequence[str]] = None,
) -> None:
    """
    Predict median suitability/density score across an ensemble of fitted
    single-class estimators in a single streaming pass via :mod:`rsgispy.imageprocess`.

    :param est_cls_objs: Sequence of fitted single-class ``scikit-learn`` estimators.
    :param in_msk_img: Path to ROI mask image.
    :param img_msk_val: Pixel value inside ``in_msk_img`` defining the ROI.
    :param env_vars: Mapping of environmental variable descriptors.
    :param analysis_vars: Ordered covariate names matching training columns.
    :param output_img: Output ``float32`` ensemble score image path.
    :param gdalformat: Output GDAL/rasterio format (Default = ``'GTIFF'``).
    :param tmp_dir: Unused legacy parameter retained for signature compatibility.
    :param normalise_data: Apply z-score normalisation from ``env_vars`` when ``True``.
    :param calc_stats: Compute output image statistics and pyramids (Default = ``True``).
    :param out_no_data_val: No-data fill value outside mask (Default = ``-9999.0``).
    :param block_size: Optional block size in pixels for :mod:`rsgispy.imageprocess`.
    :param use_pbar: Display a tqdm progress bar (Default = ``True``).
    :param fixed_var_vals: Optional mapping of variable name to a constant used
                           for every pixel instead of reading that variable's
                           raster, e.g. to hold a sampling-bias covariate
                           constant (see :func:`predict_raster_from_env_vars`).
    :param uncertainty: Optional across-member statistics written as extra
                        bands after the median, in the order given: any of
                        ``"std"`` (standard deviation), ``"iqr"``
                        (interquartile range), ``"min"``, ``"max"`` and
                        ``"cv"`` (coefficient of variation, ``std / |mean|``;
                        ``0`` where the mean is ``0``). Band descriptions name
                        each band. Default ``None`` writes the median only.

    .. code:: python

        import rsgispy.modelling.species_distribution as sdm

        sdm.pred_ensemble_sklearn_sgl_cls_mdls_prob(
            oc_mdls, "valid_mask.tif", 1, env_vars, ["v1", "v2"], "oc_ens_score.tif"
        )
    """
    _pred_ensemble_sklearn_mdls(
        est_cls_objs=est_cls_objs,
        in_msk_img=in_msk_img,
        img_msk_val=img_msk_val,
        env_vars=env_vars,
        analysis_vars=analysis_vars,
        output_img=output_img,
        model_predict_fn=_score_single_class_samples,
        agg_fn=lambda stack: np.median(stack, axis=0),
        gdalformat=gdalformat,
        tmp_dir=tmp_dir,
        normalise_data=normalise_data,
        calc_stats=calc_stats,
        out_no_data_val=out_no_data_val,
        block_size=block_size,
        use_pbar=use_pbar,
        fixed_var_vals=fixed_var_vals,
        uncertainty=uncertainty,
        main_band_name="median_score",
    )


# pylint: disable=too-many-arguments,too-many-positional-arguments
def pred_ensemble_sklearn_sgl_cls_mdls_cls(
    est_cls_objs: Sequence[Any],
    in_msk_img: str,
    img_msk_val: int,
    env_vars: Mapping[str, EnvVarInfo],
    analysis_vars: Sequence[str],
    output_img: str,
    *,
    gdalformat: str = "GTIFF",
    tmp_dir: Optional[str] = None,
    normalise_data: bool = False,
    calc_stats: bool = True,
    block_size: Optional[int] = None,
    use_pbar: bool = True,
    fixed_var_vals: Optional[Mapping[str, float]] = None,
) -> None:
    """
    Predict the proportion of single-class ensemble models voting inlier
    (presence) per pixel (in ``[0, 1]``, nodata ``-1.0`` outside mask) in a
    single streaming pass via :mod:`rsgispy.imageprocess`.

    :param est_cls_objs: Sequence of fitted single-class ``scikit-learn`` estimators.
    :param in_msk_img: Path to ROI mask image.
    :param img_msk_val: Pixel value inside ``in_msk_img`` defining the ROI.
    :param env_vars: Mapping of environmental variable descriptors.
    :param analysis_vars: Ordered covariate names matching training columns.
    :param output_img: Output ``float32`` ensemble vote proportion image path.
    :param gdalformat: Output GDAL/rasterio format (Default = ``'GTIFF'``).
    :param tmp_dir: Unused legacy parameter retained for signature compatibility.
    :param normalise_data: Apply z-score normalisation from ``env_vars`` when ``True``.
    :param calc_stats: Compute output image statistics and pyramids (Default = ``True``).
    :param block_size: Optional block size in pixels for :mod:`rsgispy.imageprocess`.
    :param use_pbar: Display a tqdm progress bar (Default = ``True``).
    :param fixed_var_vals: Optional mapping of variable name to a constant used
                           for every pixel instead of reading that variable's
                           raster, e.g. to hold a sampling-bias covariate
                           constant (see :func:`predict_raster_from_env_vars`).

    .. code:: python

        import rsgispy.modelling.species_distribution as sdm

        sdm.pred_ensemble_sklearn_sgl_cls_mdls_cls(
            oc_mdls, "valid_mask.tif", 1, env_vars, ["v1", "v2"], "oc_ens_cls.tif"
        )
    """
    _pred_ensemble_sklearn_mdls(
        est_cls_objs=est_cls_objs,
        in_msk_img=in_msk_img,
        img_msk_val=img_msk_val,
        env_vars=env_vars,
        analysis_vars=analysis_vars,
        output_img=output_img,
        model_predict_fn=lambda mdl, df: calc_single_class_predictions(
            mdl, df
        ).astype(np.float64),
        agg_fn=lambda stack: np.mean(stack, axis=0),
        gdalformat=gdalformat,
        tmp_dir=tmp_dir,
        normalise_data=normalise_data,
        calc_stats=calc_stats,
        out_no_data_val=-1.0,
        block_size=block_size,
        use_pbar=use_pbar,
        fixed_var_vals=fixed_var_vals,
    )
