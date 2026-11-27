"""
Regression accuracy assessment, point raster prediction sampling, bootstrap
confidence intervals, and diagnostic residual/scatter plotting utilities.
"""

import csv
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd

import rsgispy
from rsgispy.imageprocess import ensure_file_exists, validate_band_index
from rsgispy.regression._reg_common import (
    REG_BASE_METRIC_KEYS,
    REG_EXTRA_METRIC_KEYS,
    check_sklearn_available,
    compute_regression_metrics_dict,
)
from rsgispy.tools._ml_data import (
    load_samples_df,
    resolve_bootstrap_sample_size,
    resolve_rng,
    summarise_bootstrap_interval,
)
from rsgispy.tools.plotting import (
    CATEGORICAL_COLOURS,
    PLOT_DPI,
    REF_LINE_COLOUR,
    add_stats_text_box,
    draw_qq,
    get_fig_axs,
    save_plot,
    use_plot_style,
)
from rsgispy.tools.files import ensure_parent_dir
import rsgispy.tools.jsonio
from rsgispy.vector.attrs import write_vec_column_to_layer
from rsgispy.vectorprocess import VectorInput, finish_vec_output, read_vec_gdf
from rsgispy.zonalstats import (
    open_validated_zonal_inputs,
    sample_point_pixel,
)

__all__ = [
    "calc_reg_metrics",
    "pop_reg_img_pred_pts",
    "calc_reg_stats_ptonly_bootstrap_conf",
    "plot_pred_vs_obs",
    "plot_residuals",
    "plot_qq_residuals",
]

_TABULAR_EXTS = {".csv", ".tsv", ".txt", ".feather", ".parquet", ".pq"}

# The metric keys returned by calc_reg_metrics for each target, in the order
# they are written to the CSV file.
_REG_METRIC_KEYS = REG_BASE_METRIC_KEYS + REG_EXTRA_METRIC_KEYS


def _load_eval_table(
    in_data: Union[VectorInput, pd.DataFrame, str, os.PathLike],
    vec_lyr: Optional[str],
) -> pd.DataFrame:
    """Load evaluation attribute table from DataFrame, table file, or vector."""
    if isinstance(in_data, pd.DataFrame):
        return in_data
    in_path = os.fspath(in_data)
    ext = os.path.splitext(in_path)[1].lower()
    if vec_lyr is None and ext in _TABULAR_EXTS:
        return load_samples_df(in_path, "in_data")
    return read_vec_gdf(in_path, vec_lyr)


def _resolve_ref_pred_cols(
    ref_col: Optional[str],
    pred_col: Optional[str],
    ref_cols: Optional[Sequence[str]],
    pred_cols: Optional[Sequence[str]],
    target_names: Optional[Sequence[str]],
) -> Tuple[List[str], List[str], List[str]]:
    """Normalise single or multivariate reference/prediction column pairs."""
    if ref_cols is not None or pred_cols is not None:
        if ref_cols is None or pred_cols is None:
            raise rsgispy.RSGISPyException(
                "Both ref_cols and pred_cols must be provided together."
            )
        r_list = [str(c) for c in ref_cols]
        p_list = [str(c) for c in pred_cols]
    elif ref_col is not None or pred_col is not None:
        if ref_col is None or pred_col is None:
            raise rsgispy.RSGISPyException(
                "Both ref_col and pred_col must be provided together."
            )
        r_list = [str(ref_col)]
        p_list = [str(pred_col)]
    else:
        raise rsgispy.RSGISPyException(
            "Provide either (ref_col, pred_col) or (ref_cols, pred_cols)."
        )

    if not r_list or len(r_list) != len(p_list):
        raise rsgispy.RSGISPyException(
            "Reference and predicted column lists must be non-empty and the "
            "same length."
        )

    if target_names is not None:
        names = [str(t) for t in target_names]
        if len(names) != len(r_list):
            raise rsgispy.RSGISPyException(
                f"target_names length ({len(names)}) must match target count "
                f"({len(r_list)})."
            )
    else:
        names = list(r_list)

    return r_list, p_list, names


def _load_reg_eval_arrays(
    in_data: Union[VectorInput, pd.DataFrame, str, os.PathLike],
    vec_lyr: Optional[str],
    ref_col: Optional[str],
    pred_col: Optional[str],
    ref_cols: Optional[Sequence[str]],
    pred_cols: Optional[Sequence[str]],
    target_names: Optional[Sequence[str]],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Read paired finite reference and predicted float arrays."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    r_list, p_list, names = _resolve_ref_pred_cols(
        ref_col, pred_col, ref_cols, pred_cols, target_names
    )
    df = _load_eval_table(in_data, vec_lyr)
    if len(df) == 0:
        raise rsgispy.RSGISPyException("Input evaluation table is empty.")

    missing = [c for c in r_list + p_list if c not in df.columns]
    if missing:
        raise rsgispy.RSGISPyException(
            f"Missing required evaluation column(s): {missing}."
        )

    y_ref_df = df[r_list].apply(pd.to_numeric, errors="coerce")
    y_pred_df = df[p_list].apply(pd.to_numeric, errors="coerce")
    ref_arr = y_ref_df.to_numpy(dtype=np.float64)
    pred_arr = y_pred_df.to_numpy(dtype=np.float64)

    finite_mask = np.all(np.isfinite(ref_arr), axis=1) & np.all(
        np.isfinite(pred_arr), axis=1
    )
    ref_clean = ref_arr[finite_mask]
    pred_clean = pred_arr[finite_mask]
    if ref_clean.shape[0] == 0:
        raise rsgispy.RSGISPyException(
            "No valid finite reference/predicted sample pairs remain."
        )

    if len(names) == 1:
        return ref_clean[:, 0], pred_clean[:, 0], names
    return ref_clean, pred_clean, names


def _write_reg_metrics_csv(
    out_csv_file: Union[str, os.PathLike],
    metrics: Dict[str, Any],
    target_names: Sequence[str],
) -> None:
    """Write univariate or multivariate regression accuracy metrics to CSV."""
    out_path = ensure_parent_dir(out_csv_file)
    m_keys = _REG_METRIC_KEYS
    with open(out_path, mode="w", newline="", encoding="utf-8") as csv_fp:
        writer = csv.writer(
            csv_fp, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        writer.writerow(["target", "n_samples", *m_keys])
        n_smpls = int(metrics["n_samples"])
        if len(target_names) == 1:
            writer.writerow(
                [target_names[0], n_smpls, *[metrics[k] for k in m_keys]]
            )
        else:
            for t_name in target_names:
                writer.writerow(
                    [
                        t_name,
                        n_smpls,
                        *[metrics[f"{t_name}_{k}"] for k in m_keys],
                    ]
                )
            writer.writerow(
                ["overall_mean", n_smpls, *[metrics[k] for k in m_keys]]
            )


def calc_reg_metrics(
    in_data: Union[VectorInput, pd.DataFrame, str, os.PathLike],
    vec_lyr: Optional[str] = None,
    ref_col: Optional[str] = None,
    pred_col: Optional[str] = None,
    ref_cols: Optional[Sequence[str]] = None,
    pred_cols: Optional[Sequence[str]] = None,
    target_names: Optional[Sequence[str]] = None,
    out_json_file: Optional[Union[str, os.PathLike]] = None,
    out_csv_file: Optional[Union[str, os.PathLike]] = None,
) -> Dict[str, Any]:
    """
    Calculate continuous regression accuracy metrics from reference and
    predicted columns in a vector layer, DataFrame, CSV, or Parquet table.

    Supports both univariate regression (``ref_col``, ``pred_col``) and
    multivariate regression (``ref_cols``, ``pred_cols``). Rows where a
    reference or predicted value is not finite are ignored.

    The metrics for each target, with :math:`y` the observed (reference)
    values, :math:`\\hat{y}` the predicted values and :math:`n` the number
    of samples:

    .. list-table::
       :header-rows: 1
       :widths: 22 78

       * - Key
         - Metric
       * - ``r2``
         - Coefficient of determination (``sklearn.metrics.r2_score``).
       * - ``rmse``
         - Root mean squared error,
           :math:`\\sqrt{\\sum (\\hat{y} - y)^2 / n}`.
       * - ``mae``
         - Mean absolute error, :math:`\\sum |\\hat{y} - y| / n`.
       * - ``bias``
         - Mean error (predicted - observed), :math:`\\sum (\\hat{y} - y) / n`.
       * - ``explained_variance``
         - Explained variance score
           (``sklearn.metrics.explained_variance_score``).
       * - ``pearson_r``
         - Pearson correlation coefficient of observed and predicted values.
       * - ``mse``
         - Mean squared error, :math:`\\sum (\\hat{y} - y)^2 / n`.
       * - ``nrmse``
         - RMSE as a percentage of the mean observed value,
           :math:`100 \\times \\text{RMSE} / \\bar{y}`.
       * - ``nrmse_range``
         - RMSE as a percentage of the observed range,
           :math:`100 \\times \\text{RMSE} / (y_{max} - y_{min})`.
       * - ``nbias``
         - Bias as a percentage of the mean observed value,
           :math:`100 \\times \\text{bias} / \\bar{y}`.

    ``nrmse``, ``nrmse_range`` and ``nbias`` are ``nan`` when their
    denominator is zero. The mean observed value is signed, so a negative
    mean gives a negative ``nrmse``.

    For a single target the metrics use the keys above. For multiple targets
    the keys above hold the mean of each metric over the targets, and each
    target's own metrics use the keys ``<target name>_<key>`` (for example
    ``height_nrmse``). ``n_samples`` is the number of samples used.

    :param in_data: Input vector file path, :class:`geopandas.GeoDataFrame`,
                    :class:`pandas.DataFrame`, CSV, or Parquet file path.
    :param vec_lyr: Input vector layer name when ``in_data`` is a multi-layer
                    vector file (or ``None``).
    :param ref_col: Reference (observed) column name for univariate assessment.
    :param pred_col: Predicted column name for univariate assessment.
    :param ref_cols: Sequence of reference column names for multivariate
                     assessment.
    :param pred_cols: Sequence of predicted column names matching ``ref_cols``.
    :param target_names: Optional display names for multivariate outputs
                         (defaults to ``ref_cols``).
    :param out_json_file: Optional output JSON file path (Default = ``None``).
                          A ``nan`` metric is written as ``NaN``.
    :param out_csv_file: Optional output CSV file path (Default = ``None``),
                         with one row per target (and an ``overall_mean`` row
                         for multiple targets) and a column per metric.
    :return: Dictionary containing regression accuracy statistics and
             ``n_samples``.

    .. code:: python

        import rsgispy.regression.regmetrics as rm

        stats = rm.calc_reg_metrics(
            "val_pts.gpkg",
            vec_lyr="val_pts",
            ref_col="canopy_cover",
            pred_col="img_pred",
            out_json_file="reg_metrics.json",
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    check_sklearn_available("calc_reg_metrics")
    y_true, y_pred, eff_names = _load_reg_eval_arrays(
        in_data=in_data,
        vec_lyr=vec_lyr,
        ref_col=ref_col,
        pred_col=pred_col,
        ref_cols=ref_cols,
        pred_cols=pred_cols,
        target_names=target_names,
    )
    metrics = compute_regression_metrics_dict(
        y_true=y_true,
        y_pred=y_pred,
        target_cols=eff_names,
        extra_metrics=True,
    )
    metrics["n_samples"] = int(y_true.shape[0])

    if out_json_file is not None:
        rsgispy.tools.jsonio.write_dict_to_json(
            metrics, ensure_parent_dir(out_json_file)
        )
    if out_csv_file is not None:
        _write_reg_metrics_csv(out_csv_file, metrics, eff_names)

    return metrics


def pop_reg_img_pred_pts(
    input_reg_img: Union[str, os.PathLike],
    pts_vec_file: VectorInput,
    pts_vec_lyr: Optional[str] = None,
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    pred_col: Optional[str] = "img_pred",
    pred_cols: Optional[Sequence[str]] = None,
    pred_img_bands: Optional[Sequence[int]] = None,
    out_format: Optional[str] = None,
    reproj_vec: bool = False,
    vec_def_epsg: Optional[int] = None,
    out_compress: Optional[Union[str, bool]] = None,
    use_pbar: bool = True,
) -> Optional[gpd.GeoDataFrame]:
    """
    Sample continuous predicted raster values from single-band or multi-band
    ``input_reg_img`` at each point in ``pts_vec_file`` and store the values
    as attribute column(s).

    :param input_reg_img: Path to predicted regression raster (single-band or
                          multi-band for multivariate outputs).
    :param pts_vec_file: Input point vector file path or
                         :class:`geopandas.GeoDataFrame`.
    :param pts_vec_lyr: Input vector layer name (or ``None``).
    :param out_vec_file: Output vector file path, or ``None`` to return an
                         in-memory :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name (or ``None``).
    :param pred_col: Output column name when sampling a single band
                     (Default = ``'img_pred'``), or base prefix for multi-band
                     rasters without descriptions.
    :param pred_cols: Explicit sequence of output column names matching
                      ``pred_img_bands`` (or image bands).
    :param pred_img_bands: Optional sequence of 1-based band indices to sample
                           (defaults to all bands in ``input_reg_img``, or
                           ``1..len(pred_cols)`` when ``pred_cols`` is given).
    :param out_format: Optional output vector driver.
    :param reproj_vec: Reproject point coordinates on the fly when ``True`` if
                       vector and raster CRSs differ (Default = ``False``).
    :param vec_def_epsg: Optional EPSG code overriding vector CRS if undefined.
    :param out_compress: Optional GeoParquet compression setting.
    :param use_pbar: Display a tqdm progress bar while sampling points
                     (Default = ``True``).
    :return: Updated :class:`geopandas.GeoDataFrame` when ``out_vec_file`` is
             ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.regression.regmetrics as rm

        rm.pop_reg_img_pred_pts(
            input_reg_img="canopy_cover_pred.tif",
            pts_vec_file="val_pts.gpkg",
            pts_vec_lyr="val_pts",
            out_vec_file="val_pts_eval.gpkg",
            out_vec_lyr="val_pts",
            pred_col="pred_canopy_cover",
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-branches
    img_path = os.fspath(input_reg_img)
    ensure_file_exists(img_path, label="Input regression image")

    pts_gdf = read_vec_gdf(pts_vec_file, pts_vec_lyr)
    geom_types = set(pts_gdf.geom_type.dropna().unique())
    if not geom_types or not geom_types.issubset({"Point"}):
        raise rsgispy.RSGISPyException(
            f"The layer geometry type must be Point: Got '{sorted(geom_types)}'."
        )

    tqdm_cls = rsgispy.get_tqdm()
    with open_validated_zonal_inputs(
        pts_gdf,
        img_path,
        img_band=1,
        vec_def_epsg=vec_def_epsg,
        reproj_vec=reproj_vec,
    ) as (src, _img_nodata, geom_series):
        if pred_img_bands is not None:
            bands = [int(b) for b in pred_img_bands]
            for band_idx in bands:
                validate_band_index(band_idx, src.count, img_path)
        elif pred_cols is not None:
            if len(pred_cols) > src.count:
                raise rsgispy.RSGISPyException(
                    f"pred_cols has {len(pred_cols)} entries but image only "
                    f"has {src.count} band(s)."
                )
            bands = list(range(1, len(pred_cols) + 1))
        else:
            bands = list(range(1, src.count + 1))

        if pred_cols is not None:
            if len(pred_cols) != len(bands):
                raise rsgispy.RSGISPyException(
                    f"pred_cols length ({len(pred_cols)}) must match band "
                    f"count ({len(bands)})."
                )
            eff_cols = [str(c) for c in pred_cols]
        elif len(bands) == 1:
            eff_cols = [str(pred_col or "img_pred")]
        else:
            descs = [
                (src.descriptions[b - 1] or "").strip()
                if src.descriptions is not None
                else ""
                for b in bands
            ]
            if all(descs) and len(set(descs)) == len(bands):
                eff_cols = [f"pred_{d}" for d in descs]
            else:
                base = str(pred_col or "img_pred")
                eff_cols = [f"{base}_b{b}" for b in bands]

        nodatavals = src.nodatavals
        geom_list = list(geom_series)
        sampled_cols: Dict[str, List[float]] = {c: [] for c in eff_cols}
        for band_idx, col_name in zip(bands, eff_cols):
            band_nodata = (
                nodatavals[band_idx - 1]
                if nodatavals is not None and band_idx - 1 < len(nodatavals)
                else None
            )
            vals: List[float] = []
            for geom in tqdm_cls(
                geom_list, total=len(geom_list), disable=not use_pbar
            ):
                if geom is None or geom.is_empty:
                    vals.append(float("nan"))
                    continue
                px_val = sample_point_pixel(
                    src, band_idx, geom.x, geom.y, clamp=False
                )
                if px_val is None or not math.isfinite(float(px_val)):
                    vals.append(float("nan"))
                else:
                    f_val = float(px_val)
                    if band_nodata is not None and (
                        (math.isnan(float(band_nodata)) and math.isnan(f_val))
                        or (
                            math.isfinite(float(band_nodata))
                            and math.isclose(
                                f_val,
                                float(band_nodata),
                                rel_tol=1e-6,
                                abs_tol=1e-9,
                            )
                        )
                    ):
                        vals.append(float("nan"))
                    else:
                        vals.append(f_val)
            sampled_cols[col_name] = vals

    for col_name, vals in sampled_cols.items():
        write_vec_column_to_layer(pts_gdf, col_name, float, vals)

    return finish_vec_output(
        pts_gdf,
        out_vec_file,
        out_vec_lyr=out_vec_lyr,
        out_format=out_format,
        overwrite=True,
        out_compress=out_compress,
    )


def calc_reg_stats_ptonly_bootstrap_conf(
    in_data: Union[VectorInput, pd.DataFrame, str, os.PathLike],
    vec_lyr: Optional[str] = None,
    ref_col: Optional[str] = None,
    pred_col: Optional[str] = None,
    ref_cols: Optional[Sequence[str]] = None,
    pred_cols: Optional[Sequence[str]] = None,
    target_names: Optional[Sequence[str]] = None,
    out_json_file: Optional[Union[str, os.PathLike]] = None,
    sample_frac: float = 0.2,
    sample_n_smps: Optional[int] = None,
    bootstrap_n: int = 1000,
    rnd_seed: Optional[int] = None,
    use_pbar: bool = True,
) -> Dict[str, Any]:
    """
    Estimate bootstrap confidence intervals (mean, median, and 90th/95th/99th
    percentile bounds) for univariate or multivariate continuous regression
    accuracy metrics.

    :param in_data: Input vector file path, :class:`geopandas.GeoDataFrame`,
                    :class:`pandas.DataFrame`, CSV, or Parquet table.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param ref_col: Reference column name for univariate regression.
    :param pred_col: Predicted column name for univariate regression.
    :param ref_cols: Sequence of reference column names for multivariate
                     regression.
    :param pred_cols: Sequence of predicted column names matching ``ref_cols``.
    :param target_names: Optional display names for multivariate outputs.
    :param out_json_file: Optional output JSON file path (Default = ``None``).
    :param sample_frac: Fraction of samples drawn with replacement in each
                        bootstrap iteration when ``sample_n_smps`` is ``None``
                        (Default = ``0.2``).
    :param sample_n_smps: Explicit number of samples drawn per bootstrap
                          iteration (overrides ``sample_frac`` when provided).
    :param bootstrap_n: Number of bootstrap iterations (Default = ``1000``).
    :param rnd_seed: Optional random seed for reproducibility (Default = ``None``).
    :param use_pbar: Display a tqdm progress bar during bootstrap iterations
                     (Default = ``True``).
    :return: Dictionary mapping each metric key to its bootstrap interval
             summary dictionary.

    .. code:: python

        import rsgispy.regression.regmetrics as rm

        ci_stats = rm.calc_reg_stats_ptonly_bootstrap_conf(
            "val_pts_eval.gpkg",
            vec_lyr="val_pts",
            ref_col="canopy_cover",
            pred_col="pred_canopy_cover",
            bootstrap_n=500,
            rnd_seed=42,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    check_sklearn_available("calc_reg_stats_ptonly_bootstrap_conf")
    y_true, y_pred, eff_names = _load_reg_eval_arrays(
        in_data=in_data,
        vec_lyr=vec_lyr,
        ref_col=ref_col,
        pred_col=pred_col,
        ref_cols=ref_cols,
        pred_cols=pred_cols,
        target_names=target_names,
    )
    n_total = y_true.shape[0]
    n_samples = resolve_bootstrap_sample_size(
        n_total, sample_frac, sample_n_smps, bootstrap_n
    )
    if n_samples < 2:
        raise rsgispy.RSGISPyException(
            "Bootstrap sample size must be at least 2 to compute regression "
            "metrics."
        )

    rng = resolve_rng(rnd_seed)
    tqdm_cls = rsgispy.get_tqdm()
    runs: List[Dict[str, Any]] = []
    for _ in tqdm_cls(range(int(bootstrap_n)), disable=not use_pbar):
        bs_idx = rng.integers(0, n_total, size=n_samples)
        runs.append(
            compute_regression_metrics_dict(
                y_true=y_true[bs_idx],
                y_pred=y_pred[bs_idx],
                target_cols=eff_names,
            )
        )

    metric_keys = list(runs[0].keys())
    out_stats: Dict[str, Any] = {
        m_key: summarise_bootstrap_interval([run[m_key] for run in runs])
        for m_key in metric_keys
    }
    out_stats["n_samples_total"] = int(n_total)
    out_stats["n_samples_bootstrap"] = int(n_samples)
    out_stats["bootstrap_n"] = int(bootstrap_n)

    if out_json_file is not None:
        rsgispy.tools.jsonio.write_dict_to_json(
            out_stats, ensure_parent_dir(out_json_file)
        )
    return out_stats


def _prepare_2d_eval_arrays(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Ensure ``y_true`` and ``y_pred`` are 2D ``(n_samples, n_targets)``."""
    y_t = y_true[:, None] if y_true.ndim == 1 else y_true
    y_p = y_pred[:, None] if y_pred.ndim == 1 else y_pred
    return y_t, y_p


def _target_prefix(t_name: str, n_targets: int) -> str:
    """Metric key prefix of a target in a ``calc_reg_metrics``-style dictionary."""
    return f"{t_name}_" if n_targets > 1 else ""


def _fmt_num(val: float) -> str:
    """Format a statistic for a plot text box."""
    return f"{val:.3g}" if abs(val) < 1000 else f"{val:,.0f}"


# pylint: disable=too-many-arguments,too-many-positional-arguments

def _target_axes(
    axs: Optional[Any],
    n_targets: int,
    n_rows: int,
    n_cols: int,
    figsize: Optional[Tuple[float, float]],
) -> Tuple[Any, np.ndarray, List[Any]]:
    """
    The figure, the 2D axes array to return and one axis per target. Axes
    passed in keep the caller's layout (e.g. a 2 x 2 grid for 4 targets);
    they need at least one axis per target, used in order.
    """
    if axs is None:
        fig, axes = get_fig_axs(None, nrows=n_rows, ncols=n_cols, figsize=figsize)
        return fig, axes, [axes[divmod(i, n_cols)] for i in range(n_targets)]
    ax_arr = np.asarray(axs, dtype=object)
    flat = list(ax_arr.ravel())
    if len(flat) < n_targets:
        raise rsgispy.RSGISPyException(
            f"{n_targets} axes are needed (one per target) but {len(flat)} were given."
        )
    axes = ax_arr if ax_arr.ndim == 2 else ax_arr.reshape(1, -1)
    return flat[0].get_figure(), axes, flat[:n_targets]


@use_plot_style
def plot_pred_vs_obs(
    in_data: Union[VectorInput, pd.DataFrame, str, os.PathLike],
    vec_lyr: Optional[str] = None,
    ref_col: Optional[str] = None,
    pred_col: Optional[str] = None,
    ref_cols: Optional[Sequence[str]] = None,
    pred_cols: Optional[Sequence[str]] = None,
    target_names: Optional[Sequence[str]] = None,
    *,
    add_trendline: bool = True,
    show_stats: bool = True,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    axs: Optional[Any] = None,
    out_plot_file: Optional[Union[str, os.PathLike]] = None,
    out_dpi: int = PLOT_DPI,
) -> Tuple[Any, np.ndarray]:
    """
    Plot predicted against observed values, one panel per target, with the
    1:1 line, an optional least-squares fitted line (its equation in the
    legend) and equal axis ranges, so departures from the 1:1 line are easy
    to see. With ``show_stats`` a text box gives the coefficient of
    determination (R², ``sklearn.metrics.r2_score``, not the squared
    correlation), RMSE, bias (predicted − observed) and ``n``, computed as by
    :func:`calc_reg_metrics`.

    :param in_data: Input vector file path, :class:`geopandas.GeoDataFrame`,
                    :class:`pandas.DataFrame`, CSV, or Parquet table.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param ref_col: Reference (observed) column name for one target.
    :param pred_col: Predicted column name for one target.
    :param ref_cols: Reference column names for several targets.
    :param pred_cols: Predicted column names matching ``ref_cols``.
    :param target_names: Optional display names per target (e.g. with units).
    :param add_trendline: Draw the least-squares line of predicted on observed
                          (Default = ``True``).
    :param show_stats: Show R², RMSE, bias and ``n`` (Default = ``True``).
    :param title: Optional figure title.
    :param figsize: Size of a new figure in inches (Default = 5 inches per
                    panel, up to three panels per row).
    :param axs: Optional Matplotlib axes, one per target (Default = a new
                figure).
    :param out_plot_file: Optional output image file; the figure is saved
                          and left open (Default = ``None``).
    :param out_dpi: Resolution of the saved image (Default = ``300``).
    :return: Tuple ``(fig, axs)``, with ``axs`` a 2D array of axes (up to
             three panels per row). The metrics are given by
             :func:`calc_reg_metrics`.

    .. code:: python

        import rsgispy.regression.regmetrics as rm

        fig, axs = rm.plot_pred_vs_obs(
            "val_pts_eval.gpkg",
            vec_lyr="val_pts",
            ref_col="canopy_cover",
            pred_col="pred_canopy_cover",
            target_names=["Canopy cover (%)"],
            out_plot_file="pred_vs_obs.png",
        )
    """
    # pylint: disable=too-many-locals
    check_sklearn_available("plot_pred_vs_obs")
    y_true, y_pred, eff_names = _load_reg_eval_arrays(
        in_data=in_data,
        vec_lyr=vec_lyr,
        ref_col=ref_col,
        pred_col=pred_col,
        ref_cols=ref_cols,
        pred_cols=pred_cols,
        target_names=target_names,
    )
    metrics = compute_regression_metrics_dict(y_true, y_pred, eff_names)

    y_t_2d, y_p_2d = _prepare_2d_eval_arrays(y_true, y_pred)
    n_targets = len(eff_names)
    n_cols = 1 if n_targets == 1 else min(3, n_targets)
    n_rows = int(math.ceil(n_targets / float(n_cols)))
    if figsize is None:
        figsize = (5.0 * n_cols, 5.0 * n_rows)
    fig, axes, panel_axes = _target_axes(axs, n_targets, n_rows, n_cols, figsize)
    for idx, t_name in enumerate(eff_names):
        ax = panel_axes[idx]
        yt = y_t_2d[:, idx]
        yp = y_p_2d[:, idx]
        ax.scatter(yt, yp, s=16, alpha=0.6, color=CATEGORICAL_COLOURS[0], zorder=3)
        lo = float(min(np.min(yt), np.min(yp)))
        hi = float(max(np.max(yt), np.max(yp)))
        if hi <= lo:
            hi = lo + 1.0
        pad = 0.04 * (hi - lo)
        lims = (lo - pad, hi + pad)
        ax.plot(lims, lims, color=REF_LINE_COLOUR, linestyle="--", linewidth=1.2, label="1:1 line")
        if add_trendline and np.std(yt) > 0.0 and yt.size >= 2:
            slope, intercept = np.polyfit(yt, yp, 1)
            sign = "+" if intercept >= 0 else "−"
            ax.plot(
                lims,
                slope * np.asarray(lims) + intercept,
                color=CATEGORICAL_COLOURS[1],
                label=f"Fit: y = {slope:.3g}x {sign} {abs(intercept):.3g}",
            )
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(f"Observed {t_name}")
        ax.set_ylabel(f"Predicted {t_name}")
        if n_targets > 1:
            ax.set_title(t_name)
        ax.legend(loc="lower right")
        if show_stats:
            prefix = _target_prefix(t_name, n_targets)
            add_stats_text_box(
                ax,
                [
                    f"R² = {metrics[f'{prefix}r2']:.3f}",
                    f"RMSE = {_fmt_num(metrics[f'{prefix}rmse'])}",
                    f"Bias = {_fmt_num(metrics[f'{prefix}bias'])}",
                    f"n = {yt.size}",
                ],
                loc="upper left",
            )

    if axs is None:
        for idx in range(n_targets, n_rows * n_cols):
            axes[divmod(idx, n_cols)].set_visible(False)

    if title:
        fig.suptitle(title)
    if out_plot_file is not None:
        save_plot(fig, out_plot_file, out_dpi=out_dpi)
    return fig, axes


# pylint: disable=too-many-arguments,too-many-positional-arguments
@use_plot_style
def plot_residuals(
    in_data: Union[VectorInput, pd.DataFrame, str, os.PathLike],
    vec_lyr: Optional[str] = None,
    ref_col: Optional[str] = None,
    pred_col: Optional[str] = None,
    ref_cols: Optional[Sequence[str]] = None,
    pred_cols: Optional[Sequence[str]] = None,
    target_names: Optional[Sequence[str]] = None,
    *,
    x_axis: str = "pred",
    show_hist: bool = True,
    hist_bins: Union[int, str, Sequence[float]] = "auto",
    show_normal: bool = True,
    show_stats: bool = True,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    axs: Optional[Any] = None,
    out_plot_file: Optional[Union[str, os.PathLike]] = None,
    out_dpi: int = PLOT_DPI,
) -> Tuple[Any, np.ndarray]:
    """
    Plot the residuals (predicted − observed, the sign used by the ``bias``
    of :func:`calc_reg_metrics`) against the predicted (or observed) values,
    with a zero line, one row per target. With ``show_hist`` a histogram of
    the residuals is drawn in a narrow panel beside the y-axis, sharing it,
    so the distribution lines up with the scatter; ``show_normal`` overlays
    the normal curve with the residuals' mean and standard deviation. With
    ``show_stats`` a text box gives the bias, RMSE and ``n``.

    Residuals should be centred on zero with no trend and a similar spread
    everywhere; a funnel shape means the error grows with the value. See
    :func:`plot_qq_residuals` for a Q-Q plot of the same residuals.

    :param in_data: Input vector file path, :class:`geopandas.GeoDataFrame`,
                    :class:`pandas.DataFrame`, CSV, or Parquet table.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param ref_col: Reference (observed) column name for one target.
    :param pred_col: Predicted column name for one target.
    :param ref_cols: Reference column names for several targets.
    :param pred_cols: Predicted column names matching ``ref_cols``.
    :param target_names: Optional display names per target (e.g. with units).
    :param x_axis: ``'pred'`` (Default) to plot the residuals against the
                   predicted values, or ``'obs'`` against the observed values.
    :param show_hist: Draw the residual histogram beside the y-axis
                      (Default = ``True``).
    :param hist_bins: Bins of the histogram, as for :func:`numpy.histogram`
                      (Default = ``'auto'``).
    :param show_normal: Overlay the normal curve on the histogram
                        (Default = ``True``).
    :param show_stats: Show the bias, RMSE and ``n`` (Default = ``True``).
    :param title: Optional figure title.
    :param figsize: Size of a new figure in inches (Default = ``(7.5, 4.2)``
                    per target row).
    :param axs: Optional Matplotlib axes: per target, a scatter axis and (with
                ``show_hist``) a histogram axis, i.e. ``n_targets x 2`` axes
                (or ``n_targets``) (Default = a new figure).
    :param out_plot_file: Optional output image file; the figure is saved
                          and left open (Default = ``None``).
    :param out_dpi: Resolution of the saved image (Default = ``300``).
    :return: Tuple ``(fig, axs)``, with ``axs`` an array of shape
             ``(n_targets, 2)`` (scatter, histogram) with ``show_hist``, or
             ``(n_targets, 1)`` without.

    .. code:: python

        import rsgispy.regression.regmetrics as rm

        fig, axs = rm.plot_residuals(
            "val_pts_eval.gpkg",
            vec_lyr="val_pts",
            ref_col="canopy_cover",
            pred_col="pred_canopy_cover",
            target_names=["Canopy cover (%)"],
            out_plot_file="residuals.png",
        )
    """
    # pylint: disable=too-many-locals,too-many-statements,import-outside-toplevel
    check_sklearn_available("plot_residuals")
    if x_axis not in ("pred", "obs"):
        raise rsgispy.RSGISPyException("x_axis must be 'pred' or 'obs'.")
    y_true, y_pred, eff_names = _load_reg_eval_arrays(
        in_data=in_data,
        vec_lyr=vec_lyr,
        ref_col=ref_col,
        pred_col=pred_col,
        ref_cols=ref_cols,
        pred_cols=pred_cols,
        target_names=target_names,
    )
    y_t_2d, y_p_2d = _prepare_2d_eval_arrays(y_true, y_pred)
    n_targets = len(eff_names)
    n_cols = 2 if show_hist else 1
    if figsize is None:
        figsize = (7.5 if show_hist else 6.0, 4.2 * n_targets)
    if axs is None:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(
            n_targets,
            n_cols,
            figsize=figsize,
            squeeze=False,
            sharey="row" if show_hist else False,
            gridspec_kw={"width_ratios": [4, 1]} if show_hist else None,
        )
    else:
        fig, axes = get_fig_axs(axs, nrows=n_targets, ncols=n_cols)
        if show_hist:
            for r_i in range(n_targets):
                try:
                    axes[r_i, 1].sharey(axes[r_i, 0])
                except ValueError:  # already shared (e.g. plt.subplots(sharey="row"))
                    pass

    for idx, t_name in enumerate(eff_names):
        ax = axes[idx, 0]
        yt = y_t_2d[:, idx]
        yp = y_p_2d[:, idx]
        resid = yp - yt
        x_vals = yp if x_axis == "pred" else yt
        ax.scatter(x_vals, resid, s=16, alpha=0.6, color=CATEGORICAL_COLOURS[0], zorder=3)
        ax.axhline(0.0, color=REF_LINE_COLOUR, linestyle="--", linewidth=1.2)
        ax.set_xlabel(f"{'Predicted' if x_axis == 'pred' else 'Observed'} {t_name}")
        ax.set_ylabel("Residual (predicted − observed)")
        if n_targets > 1:
            ax.set_title(t_name)
        if show_stats:
            bias_v = float(np.mean(resid))
            rmse_v = float(np.sqrt(np.mean(resid**2)))
            add_stats_text_box(
                ax,
                [f"Bias = {_fmt_num(bias_v)}", f"RMSE = {_fmt_num(rmse_v)}", f"n = {resid.size}"],
                loc="upper left",
            )
        # Symmetric y range about zero, so over- and under-prediction compare.
        y_max = float(np.max(np.abs(resid))) * 1.1 if resid.size else 1.0
        ax.set_ylim(-y_max if y_max > 0 else -1.0, y_max if y_max > 0 else 1.0)

        if show_hist:
            h_ax = axes[idx, 1]
            counts, edges, _ = h_ax.hist(
                resid,
                bins=hist_bins,
                orientation="horizontal",
                color="#9a9a9a",
                edgecolor="white",
                linewidth=0.4,
            )
            h_ax.axhline(0.0, color=REF_LINE_COLOUR, linestyle="--", linewidth=1.2)
            if show_normal and resid.size > 1 and np.std(resid) > 0.0:
                mean_v = float(np.mean(resid))
                std_v = float(np.std(resid, ddof=1))
                y_line = np.linspace(edges[0], edges[-1], 200)
                bin_w = float(np.mean(np.diff(edges)))
                pdf = np.exp(-0.5 * ((y_line - mean_v) / std_v) ** 2)
                pdf /= std_v * np.sqrt(2.0 * np.pi)
                h_ax.plot(
                    pdf * resid.size * bin_w, y_line, color=CATEGORICAL_COLOURS[1], label="Normal"
                )
                h_ax.legend(loc="lower right", fontsize=7, handlelength=1.2)
            h_ax.tick_params(labelleft=False)
            h_ax.set_xlabel("Count")
            h_ax.grid(axis="y", visible=False)
            h_ax.set_xlim(0, max(1.0, float(np.max(counts)) * 1.1))

    if title:
        fig.suptitle(title)
    if out_plot_file is not None:
        save_plot(fig, out_plot_file, out_dpi=out_dpi)
    return fig, axes


# pylint: disable=too-many-arguments,too-many-positional-arguments
@use_plot_style
def plot_qq_residuals(
    in_data: Union[VectorInput, pd.DataFrame, str, os.PathLike],
    vec_lyr: Optional[str] = None,
    ref_col: Optional[str] = None,
    pred_col: Optional[str] = None,
    ref_cols: Optional[Sequence[str]] = None,
    pred_cols: Optional[Sequence[str]] = None,
    target_names: Optional[Sequence[str]] = None,
    *,
    show_shapiro: bool = True,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    axs: Optional[Any] = None,
    out_plot_file: Optional[Union[str, os.PathLike]] = None,
    out_dpi: int = PLOT_DPI,
) -> Tuple[Any, np.ndarray]:
    """
    Plot a normal Q-Q plot of the residuals (predicted − observed) of each
    target, with the fitted line and, with ``show_shapiro``, the Shapiro-Wilk
    test (``p < 0.05``: the residuals are unlikely to be normal). Takes the
    same data arguments as :func:`plot_residuals`; see
    :func:`rsgispy.tools.plotting.plot_qq` for any other values.

    :param in_data: Input vector file path, ``geopandas.GeoDataFrame``,
                    ``pandas.DataFrame``, CSV, or Parquet table.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param ref_col: Reference (observed) column name for one target.
    :param pred_col: Predicted column name for one target.
    :param ref_cols: Reference column names for several targets.
    :param pred_cols: Predicted column names matching ``ref_cols``.
    :param target_names: Optional display names per target.
    :param show_shapiro: Show the Shapiro-Wilk test (Default = ``True``).
    :param title: Optional figure title.
    :param figsize: Size of a new figure in inches (Default = 5 inches per
                    panel, up to three panels per row).
    :param axs: Optional Matplotlib axes, one per target (Default = a new
                figure).
    :param out_plot_file: Optional output image file; the figure is saved
                          and left open (Default = ``None``).
    :param out_dpi: Resolution of the saved image (Default = ``300``).
    :return: Tuple ``(fig, axs)``, with ``axs`` a 2D array of axes.

    .. code:: python

        import rsgispy.regression.regmetrics as rm

        fig, axs = rm.plot_qq_residuals(
            "val_pts_eval.gpkg",
            vec_lyr="val_pts",
            ref_col="canopy_cover",
            pred_col="pred_canopy_cover",
            out_plot_file="residuals_qq.png",
        )
    """
    y_true, y_pred, eff_names = _load_reg_eval_arrays(
        in_data=in_data,
        vec_lyr=vec_lyr,
        ref_col=ref_col,
        pred_col=pred_col,
        ref_cols=ref_cols,
        pred_cols=pred_cols,
        target_names=target_names,
    )
    y_t_2d, y_p_2d = _prepare_2d_eval_arrays(y_true, y_pred)
    n_targets = len(eff_names)
    n_cols = 1 if n_targets == 1 else min(3, n_targets)
    n_rows = int(math.ceil(n_targets / float(n_cols)))
    if figsize is None:
        figsize = (5.0 * n_cols, 5.0 * n_rows)
    fig, axes, panel_axes = _target_axes(axs, n_targets, n_rows, n_cols, figsize)
    for idx, t_name in enumerate(eff_names):
        ax = panel_axes[idx]
        resid = y_p_2d[:, idx] - y_t_2d[:, idx]
        if resid.size < 2:
            raise rsgispy.RSGISPyException("At least two samples are needed for a Q-Q plot.")
        draw_qq(ax, resid, show_shapiro=show_shapiro)
        ax.set_xlabel("Theoretical quantiles (normal)")
        ax.set_ylabel("Residual (predicted − observed)")
        ax.set_title(t_name if n_targets > 1 else f"Q-Q plot of {t_name} residuals")
    if axs is None:
        for idx in range(n_targets, n_rows * n_cols):
            axes[divmod(idx, n_cols)].set_visible(False)
    if title:
        fig.suptitle(title)
    if out_plot_file is not None:
        save_plot(fig, out_plot_file, out_dpi=out_dpi)
    return fig, axes
