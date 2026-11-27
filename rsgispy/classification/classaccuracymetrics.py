"""
Classification accuracy assessment and bootstrap confidence interval functions
for reference point samples.
"""

import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import rasterio

import rsgispy
from rsgispy.classification._helpers import (
    aggregate_bootstrap_runs,
    calc_class_area_accuracy_metrics,
    calc_class_pt_accuracy_metrics,
    check_sklearn_available,
    compute_cls_areas_from_img,
    load_ref_cls_samples,
    resolve_bootstrap_sample_size,
    resolve_rng,
    write_acc_stats_img_csv,
    write_acc_stats_ptonly_csv,
)
import rsgispy.tools.jsonio
from rsgispy.tools.files import ensure_parent_dir
from rsgispy.vectorprocess import VectorInput, read_vec_gdf

__all__ = [
    "calc_acc_stats_img",
    "calc_acc_stats_ptonly",
    "calc_acc_stats_img_bootstrap_conf",
    "calc_acc_stats_ptonly_bootstrap_conf",
    "calc_confidence_reliability",
]


def calc_acc_stats_img(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    ref_col: str,
    cls_col: str,
    cls_img: Union[str, os.PathLike],
    class_names_lut: Dict[int, str],
    out_json_file: Optional[Union[str, os.PathLike]] = None,
    out_csv_file: Optional[Union[str, os.PathLike]] = None,
    cls_img_band: int = 1,
    use_pbar: bool = True,
) -> Dict[str, Any]:
    """
    Calculate classification accuracy metrics from reference sample points in
    ``vec_file`` normalised by class areas derived directly from ``cls_img``
    and ``class_names_lut`` (without requiring a raster attribute table).

    :param vec_file: Input vector file path or :class:`geopandas.GeoDataFrame`
                     containing reference and classified labels.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param ref_col: Column name containing reference class labels.
    :param cls_col: Column name containing classified/predicted class labels.
    :param cls_img: Path to the classification image used to count class pixel
                    areas.
    :param class_names_lut: Dictionary mapping integer pixel class values to
                            class name strings.
    :param out_json_file: Optional output JSON file path (Default = ``None``).
    :param out_csv_file: Optional output CSV file path (Default = ``None``).
    :param cls_img_band: 1-based band index within ``cls_img``
                         (Default = ``1``).
    :param use_pbar: If ``True``, display a tqdm progress bar while counting
                     pixels in ``cls_img`` (Default = ``True``).
    :return: Dictionary containing classification accuracy statistics, area-
             normalised confusion matrix, and quantity/allocation metrics.

    .. code:: python

        import rsgispy.classification.classaccuracymetrics as cam

        stats = cam.calc_acc_stats_img(
            vec_file="ref_pts.gpkg",
            vec_lyr="ref_pts",
            ref_col="cls_ref",
            cls_col="img_cls",
            cls_img="classified.tif",
            class_names_lut={1: "Forest", 2: "Water"},
            out_json_file="acc_stats.json",
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    check_sklearn_available("calc_acc_stats_img")
    ref_int_vals, cls_int_vals, unq_cls_names = load_ref_cls_samples(
        vec_file, vec_lyr, ref_col, cls_col
    )
    cls_area_dict, cls_id_areas = compute_cls_areas_from_img(
        cls_img=cls_img,
        cls_img_band=cls_img_band,
        class_names_lut=class_names_lut,
        unq_cls_names=unq_cls_names,
        use_pbar=use_pbar,
    )

    acc_metrics = calc_class_area_accuracy_metrics(
        ref_int_vals, cls_int_vals, cls_id_areas, unq_cls_names
    )
    acc_metrics["class_area"] = cls_area_dict

    if out_json_file is not None:
        rsgispy.tools.jsonio.write_dict_to_json(
            acc_metrics, os.fspath(out_json_file)
        )
    if out_csv_file is not None:
        write_acc_stats_img_csv(
            out_csv_file, acc_metrics, unq_cls_names, cls_area_dict
        )

    return acc_metrics


def calc_acc_stats_ptonly(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    ref_col: str,
    cls_col: str,
    out_json_file: Optional[Union[str, os.PathLike]] = None,
    out_csv_file: Optional[Union[str, os.PathLike]] = None,
) -> Dict[str, Any]:
    """
    Calculate point-only classification accuracy metrics from reference and
    classified sample columns in ``vec_file``.

    :param vec_file: Input vector file path or :class:`geopandas.GeoDataFrame`
                     containing reference and classified labels.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param ref_col: Column name containing reference class labels.
    :param cls_col: Column name containing classified/predicted class labels.
    :param out_json_file: Optional output JSON file path (Default = ``None``).
    :param out_csv_file: Optional output CSV file path (Default = ``None``).
    :return: Dictionary containing point-based classification accuracy statistics
             and confusion matrix.

    .. code:: python

        import rsgispy.classification.classaccuracymetrics as cam

        stats = cam.calc_acc_stats_ptonly(
            vec_file="ref_pts.gpkg",
            vec_lyr="ref_pts",
            ref_col="cls_ref",
            cls_col="img_cls",
            out_json_file="acc_ptonly.json",
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    check_sklearn_available("calc_acc_stats_ptonly")
    ref_int_vals, cls_int_vals, unq_cls_names = load_ref_cls_samples(
        vec_file, vec_lyr, ref_col, cls_col
    )
    acc_metrics = calc_class_pt_accuracy_metrics(
        ref_int_vals, cls_int_vals, unq_cls_names
    )

    if out_json_file is not None:
        rsgispy.tools.jsonio.write_dict_to_json(
            acc_metrics, os.fspath(out_json_file)
        )
    if out_csv_file is not None:
        write_acc_stats_ptonly_csv(out_csv_file, acc_metrics, unq_cls_names)

    return acc_metrics


def calc_acc_stats_img_bootstrap_conf(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    ref_col: str,
    cls_col: str,
    cls_img: Union[str, os.PathLike],
    class_names_lut: Dict[int, str],
    out_json_file: Optional[Union[str, os.PathLike]] = None,
    sample_frac: float = 0.2,
    sample_n_smps: Optional[int] = None,
    bootstrap_n: int = 1000,
    cls_img_band: int = 1,
    rnd_seed: Optional[int] = None,
    use_pbar: bool = True,
) -> Dict[str, Any]:
    """
    Estimate bootstrap confidence intervals (mean, median, and 90th/95th/99th
    percentile bounds) for image area-weighted classification accuracy
    statistics computed by :func:`calc_acc_stats_img`.

    :param vec_file: Input vector file path or :class:`geopandas.GeoDataFrame`
                     containing reference and classified labels.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param ref_col: Column name containing reference class labels.
    :param cls_col: Column name containing classified/predicted class labels.
    :param cls_img: Path to the classification image used to compute mapped
                    class areas.
    :param class_names_lut: Dictionary mapping integer pixel class values to
                            class name strings.
    :param out_json_file: Optional output JSON file path (Default = ``None``).
    :param sample_frac: Fraction of samples drawn with replacement in each
                        bootstrap iteration when ``sample_n_smps`` is ``None``
                        (Default = ``0.2``).
    :param sample_n_smps: Explicit number of samples drawn per bootstrap
                          iteration (overrides ``sample_frac`` when provided).
    :param bootstrap_n: Number of bootstrap iterations (Default = ``1000``).
    :param cls_img_band: 1-based band index within ``cls_img``
                         (Default = ``1``).
    :param rnd_seed: Optional random seed for reproducibility (Default = ``None``).
    :param use_pbar: If ``True``, display tqdm progress bars (Default = ``True``).
    :return: Dictionary containing bootstrap mean, median, and 90th/95th/99th
             confidence intervals for overall, area-weighted, quantity, and
             per-class accuracy metrics.

    .. code:: python

        import rsgispy.classification.classaccuracymetrics as cam

        ci_stats = cam.calc_acc_stats_img_bootstrap_conf(
            vec_file="ref_pts.gpkg",
            vec_lyr="ref_pts",
            ref_col="cls_ref",
            cls_col="img_cls",
            cls_img="classified.tif",
            class_names_lut={1: "Forest", 2: "Water"},
            bootstrap_n=500,
            rnd_seed=42,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    check_sklearn_available("calc_acc_stats_img_bootstrap_conf")
    ref_int_vals, cls_int_vals, unq_cls_names = load_ref_cls_samples(
        vec_file, vec_lyr, ref_col, cls_col
    )
    n_total = ref_int_vals.shape[0]
    n_samples = resolve_bootstrap_sample_size(
        n_total, sample_frac, sample_n_smps, bootstrap_n
    )
    cls_area_dict, cls_id_areas = compute_cls_areas_from_img(
        cls_img=cls_img,
        cls_img_band=cls_img_band,
        class_names_lut=class_names_lut,
        unq_cls_names=unq_cls_names,
        use_pbar=use_pbar,
    )

    rng = resolve_rng(rnd_seed)
    tqdm_cls = rsgispy.get_tqdm()
    acc_metrics_runs: List[Dict[str, Any]] = []
    for _ in tqdm_cls(range(bootstrap_n), disable=not use_pbar):
        bs_idx = rng.integers(0, n_total, size=n_samples)
        acc_metrics_runs.append(
            calc_class_area_accuracy_metrics(
                ref_int_vals[bs_idx],
                cls_int_vals[bs_idx],
                cls_id_areas,
                unq_cls_names,
            )
        )

    out_interv_stats = aggregate_bootstrap_runs(
        acc_metrics_runs, unq_cls_names, include_area_metrics=True
    )
    out_interv_stats["class_area"] = cls_area_dict

    if out_json_file is not None:
        rsgispy.tools.jsonio.write_dict_to_json(
            out_interv_stats, os.fspath(out_json_file)
        )
    return out_interv_stats


def calc_acc_stats_ptonly_bootstrap_conf(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    ref_col: str,
    cls_col: str,
    out_json_file: Optional[Union[str, os.PathLike]] = None,
    sample_frac: float = 0.2,
    sample_n_smps: Optional[int] = None,
    bootstrap_n: int = 1000,
    rnd_seed: Optional[int] = None,
    use_pbar: bool = True,
) -> Dict[str, Any]:
    """
    Estimate bootstrap confidence intervals (mean, median, and 90th/95th/99th
    percentile bounds) for point-only classification accuracy statistics
    computed by :func:`calc_acc_stats_ptonly`.

    :param vec_file: Input vector file path or :class:`geopandas.GeoDataFrame`
                     containing reference and classified labels.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param ref_col: Column name containing reference class labels.
    :param cls_col: Column name containing classified/predicted class labels.
    :param out_json_file: Optional output JSON file path (Default = ``None``).
    :param sample_frac: Fraction of samples drawn with replacement in each
                        bootstrap iteration when ``sample_n_smps`` is ``None``
                        (Default = ``0.2``).
    :param sample_n_smps: Explicit number of samples drawn per bootstrap
                          iteration (overrides ``sample_frac`` when provided).
    :param bootstrap_n: Number of bootstrap iterations (Default = ``1000``).
    :param rnd_seed: Optional random seed for reproducibility (Default = ``None``).
    :param use_pbar: If ``True``, display a tqdm progress bar during bootstrap
                     iterations (Default = ``True``).
    :return: Dictionary containing bootstrap mean, median, and 90th/95th/99th
             confidence intervals for overall and per-class accuracy metrics.

    .. code:: python

        import rsgispy.classification.classaccuracymetrics as cam

        ci_stats = cam.calc_acc_stats_ptonly_bootstrap_conf(
            vec_file="ref_pts.gpkg",
            vec_lyr="ref_pts",
            ref_col="cls_ref",
            cls_col="img_cls",
            bootstrap_n=500,
            rnd_seed=42,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    check_sklearn_available("calc_acc_stats_ptonly_bootstrap_conf")
    ref_int_vals, cls_int_vals, unq_cls_names = load_ref_cls_samples(
        vec_file, vec_lyr, ref_col, cls_col
    )
    n_total = ref_int_vals.shape[0]
    n_samples = resolve_bootstrap_sample_size(
        n_total, sample_frac, sample_n_smps, bootstrap_n
    )

    rng = resolve_rng(rnd_seed)
    tqdm_cls = rsgispy.get_tqdm()
    acc_metrics_runs: List[Dict[str, Any]] = []
    for _ in tqdm_cls(range(bootstrap_n), disable=not use_pbar):
        bs_idx = rng.integers(0, n_total, size=n_samples)
        acc_metrics_runs.append(
            calc_class_pt_accuracy_metrics(
                ref_int_vals[bs_idx],
                cls_int_vals[bs_idx],
                unq_cls_names,
            )
        )

    out_interv_stats = aggregate_bootstrap_runs(
        acc_metrics_runs, unq_cls_names, include_area_metrics=False
    )

    if out_json_file is not None:
        rsgispy.tools.jsonio.write_dict_to_json(
            out_interv_stats, os.fspath(out_json_file)
        )
    return out_interv_stats


def _sample_img_at_pts(
    img_path: str, x_coords: np.ndarray, y_coords: np.ndarray, img_band: int
) -> np.ndarray:
    """Pixel values of band ``img_band`` at the points (NaN outside or nodata)."""
    vals = np.full(x_coords.size, np.nan, dtype=np.float64)
    with rasterio.open(img_path) as src:
        rows, cols = rasterio.transform.rowcol(src.transform, x_coords, y_coords)
        rows = np.asarray(rows, dtype=np.int64)
        cols = np.asarray(cols, dtype=np.int64)
        inside = (rows >= 0) & (rows < src.height) & (cols >= 0) & (cols < src.width)
        if np.any(inside):
            arr = src.read(img_band)
            vals[inside] = arr[rows[inside], cols[inside]].astype(np.float64)
        if src.nodata is not None and not np.isnan(src.nodata):
            vals[vals == float(src.nodata)] = np.nan
    return vals


def calc_confidence_reliability(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    ref_col: str,
    conf_img: Union[str, os.PathLike],
    cls_img: Union[str, os.PathLike],
    n_bins: int = 10,
    out_csv_file: Optional[Union[str, os.PathLike]] = None,
    *,
    conf_img_band: int = 1,
    cls_img_band: int = 1,
) -> pd.DataFrame:
    """
    Reliability table of a per-pixel classification confidence raster (the
    ``out_conf_img`` output of the ``apply_*_classifier`` functions or of
    :func:`~rsgispy.ml.predict.predict_proba_raster`): the reference points
    are grouped into ``n_bins`` equal-width confidence bins over ``[0, 1]``
    and the proportion classified correctly is reported for each bin.

    A useful confidence map has accuracy rising with confidence. When the
    probabilities are calibrated and the confidence is ``'max_prob'``-like,
    the accuracy of a bin should be close to its confidence; tree-ensemble
    probabilities are usually not calibrated, so compare the ranking rather
    than the values unless the model was calibrated (e.g., with
    :class:`sklearn.calibration.CalibratedClassifierCV`). Use reference points
    that were not used for training.

    Points outside the images, or on no-data pixels of either image, are
    ignored. ``ref_col`` must hold the class values of ``cls_img`` (e.g.,
    the ``out_id`` values).

    :param vec_file: Reference points vector file path or
                     :class:`geopandas.GeoDataFrame` (reprojected to the CRS
                     of ``conf_img`` if needed).
    :param vec_lyr: Vector layer name (or ``None``).
    :param ref_col: Column with the reference class values.
    :param conf_img: Confidence raster path (values in ``[0, 1]``).
    :param cls_img: Classified raster path (class values comparable to
                    ``ref_col``), on the same grid as ``conf_img``.
    :param n_bins: Number of equal-width confidence bins (Default = ``10``).
    :param out_csv_file: Optional output CSV file path (Default = ``None``).
    :param conf_img_band: Band of ``conf_img`` (Default = ``1``).
    :param cls_img_band: Band of ``cls_img`` (Default = ``1``).
    :return: :class:`pandas.DataFrame` with one row per bin: ``bin_lower``,
             ``bin_upper``, ``n_pts``, ``n_correct``, ``mean_conf`` and
             ``accuracy`` (NaN for empty bins). ``df.attrs`` holds
             ``n_pts`` (points used), ``overall_accuracy`` and
             ``mean_abs_gap`` (point-weighted mean of
             ``|accuracy - mean_conf|`` over non-empty bins).

    .. code:: python

        import rsgispy.classification.classaccuracymetrics as cam

        rel_df = cam.calc_confidence_reliability(
            vec_file="ref_pts.gpkg",
            vec_lyr="ref_pts",
            ref_col="ref_cls_id",
            conf_img="class_conf.tif",
            cls_img="classified.tif",
            n_bins=10,
            out_csv_file="conf_reliability.csv",
        )
        print(rel_df[["bin_lower", "bin_upper", "n_pts", "accuracy"]])
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    n_bins = int(n_bins)
    if n_bins < 1:
        raise rsgispy.RSGISPyException(f"n_bins must be >= 1, got {n_bins}.")
    conf_path = os.fspath(conf_img)
    cls_path = os.fspath(cls_img)
    for img_path in (conf_path, cls_path):
        if not os.path.exists(img_path):
            raise rsgispy.RSGISPyException(f"Image does not exist: {img_path}")
    pts_gdf = read_vec_gdf(vec_file, vec_lyr)
    if ref_col not in pts_gdf.columns:
        raise rsgispy.RSGISPyException(
            f"Column '{ref_col}' is not in the reference points."
        )
    with rasterio.open(conf_path) as src:
        img_crs = src.crs
    if img_crs is not None and pts_gdf.crs is not None and pts_gdf.crs != img_crs:
        pts_gdf = pts_gdf.to_crs(img_crs)
    pts_gdf = pts_gdf[pts_gdf.geometry.notna() & ~pts_gdf.geometry.is_empty]
    if not bool((pts_gdf.geometry.geom_type == "Point").all()):
        raise rsgispy.RSGISPyException("The reference vector must contain points.")

    x_coords = pts_gdf.geometry.x.to_numpy(dtype=np.float64)
    y_coords = pts_gdf.geometry.y.to_numpy(dtype=np.float64)
    ref_vals = pd.to_numeric(pts_gdf[ref_col], errors="coerce").to_numpy(
        dtype=np.float64
    )
    conf_vals = _sample_img_at_pts(conf_path, x_coords, y_coords, conf_img_band)
    cls_vals = _sample_img_at_pts(cls_path, x_coords, y_coords, cls_img_band)
    use = np.isfinite(conf_vals) & np.isfinite(cls_vals) & np.isfinite(ref_vals)
    conf_vals = np.clip(conf_vals[use], 0.0, 1.0)
    correct = cls_vals[use] == ref_vals[use]

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(conf_vals, edges[1:-1], right=False), 0, n_bins - 1)
    rows: List[Dict[str, Any]] = []
    for b_i in range(n_bins):
        in_bin = bin_idx == b_i
        n_pts = int(np.count_nonzero(in_bin))
        n_corr = int(np.count_nonzero(correct[in_bin]))
        rows.append(
            {
                "bin_lower": float(edges[b_i]),
                "bin_upper": float(edges[b_i + 1]),
                "n_pts": n_pts,
                "n_correct": n_corr,
                "mean_conf": float(np.mean(conf_vals[in_bin])) if n_pts else np.nan,
                "accuracy": n_corr / n_pts if n_pts else np.nan,
            }
        )
    rel_df = pd.DataFrame(rows)
    n_used = int(conf_vals.size)
    filled = rel_df["n_pts"] > 0
    rel_df.attrs["n_pts"] = n_used
    rel_df.attrs["overall_accuracy"] = (
        float(np.mean(correct)) if n_used else float("nan")
    )
    rel_df.attrs["mean_abs_gap"] = (
        float(
            np.sum(
                rel_df.loc[filled, "n_pts"]
                * np.abs(rel_df.loc[filled, "accuracy"] - rel_df.loc[filled, "mean_conf"])
            )
            / n_used
        )
        if n_used
        else float("nan")
    )
    if out_csv_file is not None:
        out_csv = ensure_parent_dir(out_csv_file)
        rel_df.to_csv(out_csv, index=False)
    return rel_df
