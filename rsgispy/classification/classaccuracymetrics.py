"""
Classification accuracy assessment and bootstrap confidence interval functions
for reference point samples.
"""

import os
from typing import Any, Dict, List, Optional, Union

import rsgispy
from rsgispy.classification._helpers import (
    HAVE_SKLEARN,
    aggregate_bootstrap_runs,
    calc_class_area_accuracy_metrics,
    calc_class_pt_accuracy_metrics,
    compute_cls_areas_from_img,
    load_ref_cls_samples,
    resolve_bootstrap_sample_size,
    resolve_rng,
    write_acc_stats_img_csv,
    write_acc_stats_ptonly_csv,
)
import rsgispy.tools.jsonio
from rsgispy.vectorprocess import VectorInput

__all__ = [
    "calc_acc_stats_img",
    "calc_acc_stats_ptonly",
    "calc_acc_stats_img_bootstrap_conf",
    "calc_acc_stats_ptonly_bootstrap_conf",
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
    rsgispy.check_optional_dep(
        HAVE_SKLEARN, "calc_acc_stats_img", "scikit-learn", "stats"
    )
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
    rsgispy.check_optional_dep(
        HAVE_SKLEARN, "calc_acc_stats_ptonly", "scikit-learn", "stats"
    )
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
    rsgispy.check_optional_dep(
        HAVE_SKLEARN,
        "calc_acc_stats_img_bootstrap_conf",
        "scikit-learn",
        "stats",
    )
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
    rsgispy.check_optional_dep(
        HAVE_SKLEARN,
        "calc_acc_stats_ptonly_bootstrap_conf",
        "scikit-learn",
        "stats",
    )
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
