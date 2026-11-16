"""
Internal shared helpers for :mod:`rsgispy.classification`.
"""

import csv
import math
import os
import secrets
from typing import (
    TYPE_CHECKING,
    Any,
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
import rasterio
from rasterio.enums import Resampling
from rasterio.io import DatasetReader
from rasterio.warp import reproject as rio_reproject
from rasterio.windows import Window
from rasterio.windows import transform as window_transform

import rsgispy
import rsgispy.imagecalc.stats
from rsgispy.imageprocess import (
    _INT_DTYPES,
    IntersectionGrid,
    ensure_file_exists,
    grids_compatible,
)
from rsgispy.imageutils.colourtable import define_colour_table, set_img_thematic
import rsgispy.imageutils.metadata
from rsgispy.imageutils.metadata import set_img_no_data_value
from rsgispy.imageutils.stats import build_img_pyramids, pop_thmt_img_stats
from rsgispy.tools._sklearn import (
    check_sklearn_available as _check_sklearn_available,
    sklearn,
)
from rsgispy.tools.files import ensure_parent_dir
import rsgispy.tools.strings
from rsgispy.vectorattrs.io import read_vec_columns
from rsgispy.vectorprocess import VectorInput

if TYPE_CHECKING:
    from rsgispy.classification.classdata import ClassificationConfig, ClassInfoObj, ImageBandInfo

ClsConfigInput = Union[
    "ClassificationConfig",
    str,
    os.PathLike,
    Mapping[str, "ClassInfoObj"],
]
SamplesInput = Union[pd.DataFrame, str, os.PathLike]

_CONF_INT_CONSTS: Tuple[float, ...] = (1.64, 1.96, 2.33, 2.58)
_BOOTSTRAP_PERCENTILES: Tuple[float, ...] = (
    0.5,
    2.5,
    5.0,
    50.0,
    95.0,
    97.5,
    99.5,
)


def resolve_rng(rnd_seed: Optional[int] = None) -> np.random.Generator:
    """Create a NumPy Generator from an optional integer seed."""
    seed = secrets.randbits(128) if rnd_seed is None else abs(int(rnd_seed))
    return np.random.default_rng(seed)


def _safe_divide_scalar(num: float, denom: float) -> float:
    """Return ``num / denom`` or ``0.0`` when ``denom <= 0``."""
    return float(num / denom) if denom > 0 else 0.0


def _wald_conf_intervals(
    acc_prop: float,
    err_prop: float,
    n_samples: int,
) -> List[float]:
    """Compute Wald normal confidence intervals for 90%, 95%, 98%, and 99%."""
    if n_samples <= 0:
        return [0.0 for _ in _CONF_INT_CONSTS]
    se_val = math.sqrt(max(0.0, acc_prop * err_prop) / float(n_samples))
    return [float(const * se_val) for const in _CONF_INT_CONSTS]


def _format_report_dict(
    raw_report: Dict[str, Any],
    cls_names: List[str],
) -> Dict[str, Any]:
    """Normalise keys and scalar types from ``sklearn.metrics.classification_report``."""
    out: Dict[str, Any] = {}
    for cls_name in cls_names:
        cls_entry = raw_report[cls_name]
        out[cls_name] = {
            "precision": float(cls_entry["precision"]),
            "recall": float(cls_entry["recall"]),
            "f1-score": float(cls_entry["f1-score"]),
            "support": float(cls_entry["support"]),
        }
    out["accuracy"] = float(raw_report["accuracy"])
    for avg_key in ("macro avg", "weighted avg"):
        avg_entry = raw_report[avg_key]
        out[avg_key] = {
            "precision": float(avg_entry["precision"]),
            "recall": float(avg_entry["recall"]),
            "f1-score": float(avg_entry["f1-score"]),
            "support": float(avg_entry["support"]),
        }
    return out


def cls_quantity_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cls_area: np.ndarray,
    labels: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Calculate Pontius & Millones (2011) quantity and allocation disagreement.
    """
    # pylint: disable=too-many-locals
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    cls_area_arr = np.asarray(cls_area, dtype=float)

    for arr in (y_true_arr, y_pred_arr, cls_area_arr):
        if arr.ndim != 1:
            raise rsgispy.RSGISPyException(
                "All input arrays for quantity accuracy must be 1D."
            )

    if labels is None:
        labels_arr = np.arange(cls_area_arr.size, dtype=int)
        if np.unique(y_true_arr).size != cls_area_arr.size:
            raise rsgispy.RSGISPyException(
                "Number of classes does not match number of class areas."
            )
    else:
        labels_arr = np.asarray(labels, dtype=int)
        if labels_arr.size != cls_area_arr.size:
            raise rsgispy.RSGISPyException(
                "Number of class labels does not match number of class areas."
            )

    cm = sklearn.metrics.confusion_matrix(
        y_true_arr, y_pred_arr, labels=labels_arr
    ).T
    total_area = float(cls_area_arr.sum())
    if total_area <= 0.0:
        raise rsgispy.RSGISPyException("Total class area must be positive.")

    prop_area = (cls_area_arr / total_area).reshape(-1, 1)
    row_sums = cm.sum(axis=1).reshape(-1, 1).astype(float)
    row_props = np.divide(
        cm.astype(float),
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums > 0,
    )
    norm_cm = row_props * prop_area
    comp_total = norm_cm.sum(axis=1)
    ref_total = norm_cm.sum(axis=0)

    quantity_disagreement = float(np.sum(np.abs(ref_total - comp_total)) / 2.0)
    commission = np.array(
        [row.sum() - row[idx] for idx, row in enumerate(norm_cm)],
        dtype=float,
    )
    omission = ref_total - np.diag(norm_cm)
    allocation_disagreement = float(
        np.sum(2.0 * np.minimum(commission, omission)) / 2.0
    )
    norm_sum = float(np.sum(norm_cm))
    prop_correct = _safe_divide_scalar(float(np.sum(np.diag(norm_cm))), norm_sum)
    disagreement = quantity_disagreement + allocation_disagreement

    return {
        "Quantity Disagreement (Q)": quantity_disagreement,
        "Allocation Disagreement (A)": allocation_disagreement,
        "Proportion Correct (C)": prop_correct,
        "Total Disagreement (D)": disagreement,
    }


def _calc_base_pt_metrics(
    ref_samples: np.ndarray,
    pred_samples: np.ndarray,
    cls_names: List[str],
) -> Tuple[Dict[str, Any], np.ndarray]:
    """Compute base point classification metrics and transposed confusion matrix."""
    # pylint: disable=too-many-locals
    n_cls = len(cls_names)
    labels = np.arange(n_cls, dtype=int)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw_report = sklearn.metrics.classification_report(
            ref_samples,
            pred_samples,
            labels=labels,
            target_names=cls_names,
            output_dict=True,
            zero_division=0,
        )
        cohen_kappa = float(
            sklearn.metrics.cohen_kappa_score(
                ref_samples, pred_samples, labels=labels
            )
        )
        bal_acc = float(
            sklearn.metrics.balanced_accuracy_score(ref_samples, pred_samples)
        )
        mcc_val = float(
            sklearn.metrics.matthews_corrcoef(ref_samples, pred_samples)
        )
        ham_loss = float(
            sklearn.metrics.hamming_loss(ref_samples, pred_samples)
        )

    if math.isnan(cohen_kappa):
        cohen_kappa = 1.0 if np.array_equal(ref_samples, pred_samples) else 0.0
    if math.isnan(bal_acc):
        bal_acc = 0.0
    if math.isnan(mcc_val):
        mcc_val = 0.0

    acc_metrics = _format_report_dict(raw_report, cls_names)
    acc_metrics["cohen_kappa"] = cohen_kappa
    acc_metrics["bal_accuracy_score"] = bal_acc
    acc_metrics["matthews_corrcoef"] = mcc_val
    acc_metrics["hamming_loss"] = ham_loss

    cm = sklearn.metrics.confusion_matrix(
        ref_samples, pred_samples, labels=labels
    ).T
    user_accuracy = [
        _safe_divide_scalar(float(row[idx]), float(row.sum())) * 100.0
        for idx, row in enumerate(cm)
    ]
    producer_accuracy = [
        _safe_divide_scalar(float(col[idx]), float(col.sum())) * 100.0
        for idx, col in enumerate(cm.T)
    ]

    cls_conf_intervals: Dict[str, List[float]] = {}
    sum_all_smpls = int(cm.sum())
    for idx, col in enumerate(cm.T):
        cls_acc = producer_accuracy[idx] / 100.0
        cls_conf_intervals[cls_names[idx]] = _wald_conf_intervals(
            cls_acc, 1.0 - cls_acc, int(col.sum())
        )

    overall_acc = float(acc_metrics["accuracy"])
    acc_metrics["cls_confidence_intervals"] = cls_conf_intervals
    acc_metrics["accuracy_conf_interval"] = _wald_conf_intervals(
        overall_acc, 1.0 - overall_acc, sum_all_smpls
    )
    acc_metrics["confusion_matrix"] = cm.tolist()
    acc_metrics["user_accuracy"] = user_accuracy
    acc_metrics["producer_accuracy"] = producer_accuracy
    return acc_metrics, cm


def calc_class_pt_accuracy_metrics(
    ref_samples: np.ndarray,
    pred_samples: np.ndarray,
    cls_names: Sequence[str],
) -> Dict[str, Any]:
    """Compute point-only classification accuracy metrics dictionary."""
    cls_list = sorted(str(name) for name in cls_names)
    acc_metrics, _ = _calc_base_pt_metrics(ref_samples, pred_samples, cls_list)
    acc_metrics["omission"] = [
        100.0 - float(prod_val) for prod_val in acc_metrics["producer_accuracy"]
    ]
    acc_metrics["commission"] = [
        100.0 - float(user_val) for user_val in acc_metrics["user_accuracy"]
    ]
    return acc_metrics


def calc_class_area_accuracy_metrics(
    ref_samples: np.ndarray,
    pred_samples: np.ndarray,
    cls_area: np.ndarray,
    cls_names: Sequence[str],
) -> Dict[str, Any]:
    """Compute area-weighted classification accuracy metrics dictionary."""
    # pylint: disable=too-many-locals
    cls_list = sorted(str(name) for name in cls_names)
    acc_metrics, cm = _calc_base_pt_metrics(ref_samples, pred_samples, cls_list)

    cls_area_arr = np.asarray(cls_area, dtype=float)
    sum_area = float(cls_area_arr.sum())
    if sum_area <= 0.0:
        raise rsgispy.RSGISPyException("Total mapped class area must be > 0.")

    sum_f1 = 0.0
    sum_recall = 0.0
    sum_precision = 0.0
    for clsname, clsarea in zip(cls_list, cls_area_arr):
        area_val = float(clsarea)
        sum_f1 += acc_metrics[clsname]["f1-score"] * area_val
        sum_recall += acc_metrics[clsname]["recall"] * area_val
        sum_precision += acc_metrics[clsname]["precision"] * area_val

    acc_metrics["weighted area avg"] = {
        "precision": float(sum_precision / sum_area),
        "recall": float(sum_recall / sum_area),
        "f1-score": float(sum_f1 / sum_area),
        "support": sum_area,
    }

    prop_area = (cls_area_arr / sum_area).reshape(-1, 1)
    row_sums = cm.sum(axis=1).reshape(-1, 1).astype(float)
    row_props = np.divide(
        cm.astype(float),
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums > 0,
    )
    norm_cm = row_props * prop_area

    ref_total = norm_cm.sum(axis=0)
    commission = [float(row.sum() - row[idx]) for idx, row in enumerate(norm_cm)]
    omission = (ref_total - np.diag(norm_cm)).tolist()
    cls_area_prop = np.sum(norm_cm, axis=0)

    acc_metrics["norm_confusion_matrix"] = norm_cm.tolist()
    acc_metrics["commission"] = commission
    acc_metrics["omission"] = [float(val) for val in omission]
    acc_metrics["est_prop_cls_area"] = [float(val) for val in cls_area_prop]

    labels = np.arange(len(cls_list), dtype=int)
    quantity_metrics = cls_quantity_accuracy(
        ref_samples, pred_samples, cls_area_arr, labels=labels
    )
    sum_all_smpls = int(cm.sum())
    quantity_metrics["C Conf Interval"] = _wald_conf_intervals(  # type: ignore[assignment]
        quantity_metrics["Proportion Correct (C)"],
        quantity_metrics["Total Disagreement (D)"],
        sum_all_smpls,
    )
    acc_metrics["quantity_metrics"] = quantity_metrics
    return acc_metrics


def load_ref_cls_samples(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    ref_col: str,
    cls_col: str,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Read reference and predicted class columns from a vector input and encode
    them as sorted integer IDs alongside the sorted class name list.
    """
    records = read_vec_columns(vec_file, vec_lyr, [ref_col, cls_col])
    if not records:
        raise rsgispy.RSGISPyException(
            "Input vector layer contains no features."
        )

    ref_vals = np.array([str(row[ref_col]) for row in records], dtype=str)
    cls_vals = np.array([str(row[cls_col]) for row in records], dtype=str)

    unq_arr = np.unique(np.concatenate((np.unique(ref_vals), np.unique(cls_vals))))
    unq_cls_names = sorted(str(name) for name in unq_arr.tolist())
    if len(unq_cls_names) < 2:
        raise rsgispy.RSGISPyException("Must have at least two classes.")

    cls_name_lut = {name: idx for idx, name in enumerate(unq_cls_names)}
    ref_int_vals = np.empty(ref_vals.shape[0], dtype=int)
    cls_int_vals = np.empty(cls_vals.shape[0], dtype=int)
    for cls_name, cls_id in cls_name_lut.items():
        ref_int_vals[ref_vals == cls_name] = cls_id
        cls_int_vals[cls_vals == cls_name] = cls_id

    return ref_int_vals, cls_int_vals, unq_cls_names


def compute_cls_areas_from_img(
    cls_img: Union[str, os.PathLike],
    cls_img_band: int,
    class_names_lut: Dict[int, str],
    unq_cls_names: List[str],
    use_pbar: bool = True,
) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Count pixels per class in ``cls_img`` and return ``(cls_area_dict, cls_id_areas)``
    matching ``unq_cls_names``.
    """
    # pylint: disable=too-many-locals
    if not class_names_lut:
        raise rsgispy.RSGISPyException("class_names_lut must not be empty.")

    img_path = os.fspath(cls_img)
    ensure_file_exists(img_path, label="Classification image")

    pxl_vals = list(class_names_lut.keys())
    n_pxls = rsgispy.imagecalc.stats.count_pxls_of_val(
        img_path,
        pxl_vals,
        img_band=cls_img_band,
        use_pbar=use_pbar,
    )
    pxl_size_x, pxl_size_y = rsgispy.imageutils.metadata.get_img_res(
        img_path, abs_vals=True
    )
    pxl_area = float(pxl_size_x * pxl_size_y)

    cls_area_dict: Dict[str, float] = {}
    for cls_id, pxl_ct in zip(pxl_vals, n_pxls):
        cls_name_str = rsgispy.tools.strings.check_str(
            class_names_lut[cls_id], rm_non_ascii=True
        )
        cls_area_dict[cls_name_str] = float(pxl_ct) * pxl_area

    missing = [name for name in unq_cls_names if name not in cls_area_dict]
    if missing:
        raise rsgispy.RSGISPyException(
            f"Did not calculate the class areas for: {', '.join(missing)}"
        )

    cls_id_areas = np.array(
        [cls_area_dict[cls_name] for cls_name in unq_cls_names],
        dtype=float,
    )
    if float(cls_id_areas.sum()) <= 0.0:
        raise rsgispy.RSGISPyException(
            "Total mapped pixel area across classes in cls_img is zero."
        )
    return cls_area_dict, cls_id_areas


def resolve_bootstrap_sample_size(
    n_total: int,
    sample_frac: float,
    sample_n_smps: Optional[int],
    bootstrap_n: int,
) -> int:
    """Validate bootstrap parameters and return sample size per iteration."""
    if bootstrap_n < 1:
        raise rsgispy.RSGISPyException("bootstrap_n must be at least 1.")
    if sample_n_smps is None:
        if sample_frac <= 0.0 or sample_frac > 1.0:
            raise rsgispy.RSGISPyException(
                "sample_frac must be within (0.0, 1.0]."
            )
        n_samples = max(1, int(n_total * sample_frac))
    else:
        if sample_n_smps < 1:
            raise rsgispy.RSGISPyException("sample_n_smps must be at least 1.")
        n_samples = min(int(sample_n_smps), n_total)
    return n_samples


def summarise_bootstrap_interval(values: Sequence[float]) -> Dict[str, Any]:
    """Compute mean, median, and 90th/95th/99th percentile bounds."""
    arr = np.asarray(values, dtype=float)
    inters = np.percentile(arr, _BOOTSTRAP_PERCENTILES)
    return {
        "mean": float(np.mean(arr)),
        "median": float(inters[3]),
        "90th": [float(inters[2]), float(inters[4])],
        "95th": [float(inters[1]), float(inters[5])],
        "99th": [float(inters[0]), float(inters[6])],
    }


def aggregate_bootstrap_runs(
    acc_metrics_runs: List[Dict[str, Any]],
    unq_cls_names: List[str],
    include_area_metrics: bool = False,
) -> Dict[str, Any]:
    """Aggregate repeated bootstrap accuracy metric dictionaries into CIs."""
    # pylint: disable=too-many-locals
    out_stats: Dict[str, Any] = {}

    for key in (
        "accuracy",
        "cohen_kappa",
        "bal_accuracy_score",
        "matthews_corrcoef",
        "hamming_loss",
    ):
        out_stats[key] = summarise_bootstrap_interval(
            [run[key] for run in acc_metrics_runs]
        )

    avg_keys = ["macro avg", "weighted avg"]
    if include_area_metrics:
        avg_keys.append("weighted area avg")

    for avg_key in avg_keys:
        out_stats[avg_key] = {
            metric: summarise_bootstrap_interval(
                [run[avg_key][metric] for run in acc_metrics_runs]
            )
            for metric in ("f1-score", "precision", "recall")
        }

    if include_area_metrics:
        q_keys = (
            "Quantity Disagreement (Q)",
            "Allocation Disagreement (A)",
            "Proportion Correct (C)",
            "Total Disagreement (D)",
        )
        out_stats["quantity_metrics"] = {
            q_key: summarise_bootstrap_interval(
                [run["quantity_metrics"][q_key] for run in acc_metrics_runs]
            )
            for q_key in q_keys
        }

    for idx, cls_name in enumerate(unq_cls_names):
        cls_summary: Dict[str, Any] = {}
        for metric in ("f1-score", "precision", "recall", "support"):
            cls_summary[metric] = summarise_bootstrap_interval(
                [run[cls_name][metric] for run in acc_metrics_runs]
            )
        for metric in (
            "producer_accuracy",
            "user_accuracy",
            "commission",
            "omission",
        ):
            cls_summary[metric] = summarise_bootstrap_interval(
                [run[metric][idx] for run in acc_metrics_runs]
            )
        if include_area_metrics:
            cls_summary["est_prop_cls_area"] = summarise_bootstrap_interval(
                [run["est_prop_cls_area"][idx] for run in acc_metrics_runs]
            )
        out_stats[cls_name] = cls_summary

    return out_stats


def _write_csv_common_header(
    writer: Any,
    acc_metrics: Dict[str, Any],
) -> None:
    """Write common top-level overall metrics and confidence interval rows."""
    writer.writerow(["overall accuracy", acc_metrics["accuracy"]])
    writer.writerow(["cohen kappa", acc_metrics["cohen_kappa"]])
    writer.writerow(["balanced accuracy score", acc_metrics["bal_accuracy_score"]])
    writer.writerow(
        ["matthews correlation coefficient", acc_metrics["matthews_corrcoef"]]
    )
    writer.writerow(["hamming loss", acc_metrics["hamming_loss"]])
    writer.writerow([""])
    writer.writerow(["Confidence Interval", "90%", "95%", "98%", "99%"])
    writer.writerow(["Overall Accuracy", *acc_metrics["accuracy_conf_interval"]])
    writer.writerow([""])


def _write_csv_class_scores(
    writer: Any,
    acc_metrics: Dict[str, Any],
    unq_cls_names: List[str],
    weighted_label: str,
    include_area_avg: bool,
) -> None:
    """Write per-class F1/precision/recall/support rows and averages."""
    writer.writerow(["class", "f1-score", "precision", "recall", "support"])
    for cls_name in unq_cls_names:
        writer.writerow(
            [
                cls_name,
                acc_metrics[cls_name]["f1-score"],
                acc_metrics[cls_name]["precision"],
                acc_metrics[cls_name]["recall"],
                acc_metrics[cls_name]["support"],
            ]
        )
    writer.writerow([""])
    for key, label in (
        ("macro avg", "macro avg"),
        ("weighted avg", weighted_label),
    ):
        writer.writerow(
            [
                label,
                acc_metrics[key]["f1-score"],
                acc_metrics[key]["precision"],
                acc_metrics[key]["recall"],
                acc_metrics[key]["support"],
            ]
        )
    if include_area_avg:
        writer.writerow(
            [
                "weighted (area) avg",
                acc_metrics["weighted area avg"]["f1-score"],
                acc_metrics["weighted area avg"]["precision"],
                acc_metrics["weighted area avg"]["recall"],
                acc_metrics["weighted area avg"]["support"],
            ]
        )
    writer.writerow([""])

    writer.writerow(["Confidence Interval", "90%", "95%", "98%", "99%"])
    for cls_name in unq_cls_names:
        writer.writerow(
            [cls_name, *acc_metrics["cls_confidence_intervals"][cls_name]]
        )
    writer.writerow([""])


def write_acc_stats_img_csv(
    out_csv_file: Union[str, os.PathLike],
    acc_metrics: Dict[str, Any],
    unq_cls_names: List[str],
    cls_area_dict: Dict[str, float],
) -> None:
    """Write image area-weighted accuracy metrics and matrices to CSV."""
    # pylint: disable=too-many-locals
    out_path = ensure_parent_dir(out_csv_file)
    tot_area = float(sum(cls_area_dict[name] for name in unq_cls_names))
    q_metrics = acc_metrics["quantity_metrics"]

    with open(out_path, mode="w", newline="", encoding="utf-8") as csv_fp:
        writer = csv.writer(
            csv_fp, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        _write_csv_common_header(writer, acc_metrics)

        for key in (
            "Allocation Disagreement (A)",
            "Quantity Disagreement (Q)",
            "Proportion Correct (C)",
            "Total Disagreement (D)",
        ):
            writer.writerow([key, q_metrics[key]])
        writer.writerow([""])

        writer.writerow(["Confidence Interval", "90%", "95%", "98%", "99%"])
        writer.writerow(["Area Norm Accuracy", *q_metrics["C Conf Interval"]])
        writer.writerow([""])

        _write_csv_class_scores(
            writer,
            acc_metrics,
            unq_cls_names,
            weighted_label="weighted (sample) avg",
            include_area_avg=True,
        )

        writer.writerow(["Point Count Confusion Matrix"])
        writer.writerow(["", *unq_cls_names, "User Acc"])
        for cls_name, cm_row, user_acc in zip(
            unq_cls_names,
            acc_metrics["confusion_matrix"],
            acc_metrics["user_accuracy"],
        ):
            writer.writerow([cls_name, *cm_row, user_acc])
        writer.writerow(["Producer", *acc_metrics["producer_accuracy"]])
        writer.writerow([""])

        writer.writerow(["Normalised Confusion Matrix"])
        writer.writerow(["", *unq_cls_names])
        for cls_name, cm_row in zip(
            unq_cls_names, acc_metrics["norm_confusion_matrix"]
        ):
            writer.writerow([cls_name, *cm_row])

        writer.writerow([""])
        writer.writerow(["class", "commission", "omission"])
        for idx, cls_name in enumerate(unq_cls_names):
            writer.writerow(
                [
                    cls_name,
                    acc_metrics["commission"][idx],
                    acc_metrics["omission"][idx],
                ]
            )

        writer.writerow([""])
        writer.writerow(
            ["class", "pixel area", "Est. Prop. Area", "Est. Area"]
        )
        for idx, cls_name in enumerate(unq_cls_names):
            est_prop = acc_metrics["est_prop_cls_area"][idx]
            writer.writerow(
                [
                    cls_name,
                    cls_area_dict[cls_name],
                    est_prop,
                    tot_area * est_prop,
                ]
            )


def write_acc_stats_ptonly_csv(
    out_csv_file: Union[str, os.PathLike],
    acc_metrics: Dict[str, Any],
    unq_cls_names: List[str],
) -> None:
    """Write point-only accuracy metrics and confusion matrix to CSV."""
    out_path = ensure_parent_dir(out_csv_file)

    with open(out_path, mode="w", newline="", encoding="utf-8") as csv_fp:
        writer = csv.writer(
            csv_fp, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        _write_csv_common_header(writer, acc_metrics)
        _write_csv_class_scores(
            writer,
            acc_metrics,
            unq_cls_names,
            weighted_label="weighted (pixel) avg",
            include_area_avg=False,
        )

        writer.writerow(["Point Count Confusion Matrix"])
        writer.writerow(["", *unq_cls_names, "Users", "Commission"])
        for cls_name, cm_row, user_acc, com_val in zip(
            unq_cls_names,
            acc_metrics["confusion_matrix"],
            acc_metrics["user_accuracy"],
            acc_metrics["commission"],
        ):
            writer.writerow([cls_name, *cm_row, user_acc, com_val])
        writer.writerow(["Producers", *acc_metrics["producer_accuracy"]])
        writer.writerow(["Omission", *acc_metrics["omission"]])
        writer.writerow([""])


def select_out_int_dtype(values: Sequence[int]) -> Tuple[int, np.dtype]:
    """Select the smallest rsgispy integer datatype covering ``values``."""
    min_val = min(values)
    max_val = max(values)
    if min_val >= 0:
        if max_val <= 255:
            return rsgispy.TYPE_8UINT, np.dtype(np.uint8)
        if max_val <= 65535:
            return rsgispy.TYPE_16UINT, np.dtype(np.uint16)
        if max_val <= 4294967295:
            return rsgispy.TYPE_32UINT, np.dtype(np.uint32)
        return rsgispy.TYPE_64UINT, np.dtype(np.uint64)
    if min_val >= -32768 and max_val <= 32767:
        return rsgispy.TYPE_16INT, np.dtype(np.int16)
    return rsgispy.TYPE_32INT, np.dtype(np.int32)


def check_sklearn_available(feature_name: str) -> None:
    """Verify ``scikit-learn`` is installed before running classification routines."""
    _check_sklearn_available(feature_name, extra="classification")


def coerce_cls_config(
    cls_cfg: Optional[ClsConfigInput],
    out_id_col: Optional[str] = None,
    internal_id_col: Optional[str] = None,
    class_name_col: Optional[str] = None,
    allow_none: bool = False,
) -> Tuple[
    Optional["ClassificationConfig"],
    Dict[str, "ClassInfoObj"],
    str,
    str,
    str,
]:
    """Resolve configuration or class mapping into normalised components."""
    # pylint: disable=import-outside-toplevel
    from rsgispy.classification.classdata import (
        ClassificationConfig,
        normalise_class_info,
        read_cls_config,
    )

    eff_out = "out_cls_id" if out_id_col is None else out_id_col
    eff_int = "internal_cls_id" if internal_id_col is None else internal_id_col
    eff_name = "class_name" if class_name_col is None else class_name_col

    if cls_cfg is None:
        if not allow_none:
            raise rsgispy.RSGISPyException(
                "cls_cfg must be a ClassificationConfig, a JSON config file "
                "path, or a mapping of {class_name: ClassInfoObj}."
            )
        return None, {}, eff_out, eff_int, eff_name

    if isinstance(cls_cfg, (str, os.PathLike)):
        cfg = read_cls_config(cls_cfg)
    elif isinstance(cls_cfg, ClassificationConfig):
        cfg = cls_cfg
    elif isinstance(cls_cfg, Mapping):
        norm_classes = normalise_class_info(cls_cfg)
        return None, norm_classes, eff_out, eff_int, eff_name
    else:
        raise rsgispy.RSGISPyException(
            "cls_cfg must be a ClassificationConfig, a JSON config file path, "
            "or a mapping of {class_name: ClassInfoObj}."
        )

    return (
        cfg,
        cfg.class_info,
        cfg.out_id_col if out_id_col is None else out_id_col,
        cfg.internal_id_col if internal_id_col is None else internal_id_col,
        cfg.class_name_col if class_name_col is None else class_name_col,
    )


def load_samples_df(samples: SamplesInput, label: str) -> pd.DataFrame:
    """Load sample DataFrame from an in-memory DataFrame or table file path."""
    # pylint: disable=import-outside-toplevel
    from rsgispy.classification.classdata import read_cls_samples

    if isinstance(samples, pd.DataFrame):
        return samples
    if isinstance(samples, (str, os.PathLike)):
        return read_cls_samples(samples)
    raise rsgispy.RSGISPyException(
        f"{label} must be a pandas DataFrame or sample table file path."
    )


def _infer_band_names_from_img_info(
    img_band_info: Sequence["ImageBandInfo"],
) -> Optional[List[str]]:
    """
    Resolve feature column names from ``ImageBandInfo`` objects, avoiding
    opening raster files when band indices and names are already explicit.
    """
    # pylint: disable=import-outside-toplevel
    from rsgispy.classification.classdata import resolve_img_band_info

    all_have_files = all(
        info.file_name is not None and os.path.exists(info.file_name)
        for info in img_band_info
    )
    if all_have_files:
        resolved = resolve_img_band_info(img_band_info)
        return [col for _, _, cols in resolved for col in cols]

    names: List[str] = []
    for idx, info in enumerate(img_band_info):
        if info.band_names is not None:
            names.extend(info.band_names)
        elif info.bands is not None:
            if info.name and info.name.strip():
                prefix = info.name.strip()
            elif info.file_name:
                prefix = os.path.splitext(os.path.basename(info.file_name))[0]
            else:
                prefix = f"img{idx + 1}"
            names.extend(f"{prefix}_b{int(b)}" for b in info.bands)
        else:
            return None
    return names


def resolve_feature_cols(
    cfg: Optional["ClassificationConfig"],
    df: pd.DataFrame,
    feature_cols: Optional[Sequence[str]],
    excluded_cols: Sequence[str],
) -> List[str]:
    """Determine and validate ordered feature column names for sample tables."""
    if feature_cols is not None:
        cols = [str(col) for col in feature_cols]
    elif cfg is not None:
        inferred = _infer_band_names_from_img_info(cfg.img_band_info)
        if inferred is not None and all(col in df.columns for col in inferred):
            cols = inferred
        else:
            excluded = set(excluded_cols) | {"x", "y"}
            cols = [
                str(col)
                for col in df.columns
                if col not in excluded
                and pd.api.types.is_numeric_dtype(df[col])
            ]
    else:
        excluded = set(excluded_cols) | {"x", "y"}
        cols = [
            str(col)
            for col in df.columns
            if col not in excluded
            and pd.api.types.is_numeric_dtype(df[col])
        ]

    if not cols:
        raise rsgispy.RSGISPyException(
            "Could not resolve any feature columns from training data."
        )
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise rsgispy.RSGISPyException(
            f"Feature column(s) {missing} missing from sample DataFrame."
        )
    return cols


def extract_xy(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    label: str,
    valid_ids: Optional[Sequence[int]] = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Extract validated ``(X_df, y_arr)`` arrays from a sample DataFrame."""
    if len(df) == 0:
        raise rsgispy.RSGISPyException(f"{label} sample table is empty.")
    missing_feats = [col for col in feature_cols if col not in df.columns]
    if missing_feats:
        raise rsgispy.RSGISPyException(
            f"{label} is missing feature column(s): {missing_feats}."
        )
    if target_col not in df.columns:
        raise rsgispy.RSGISPyException(
            f"{label} is missing target class column '{target_col}'."
        )

    x_df = df[list(feature_cols)].copy()
    x_vals = x_df.to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(x_vals)):
        raise rsgispy.RSGISPyException(
            f"{label} contains NaN or infinite values in feature columns."
        )

    y_series = df[target_col]
    if y_series.isna().any():
        raise rsgispy.RSGISPyException(
            f"{label} contains missing values in target column '{target_col}'."
        )
    y_arr = y_series.to_numpy(dtype=np.int64)
    if np.any(y_arr < 0):
        raise rsgispy.RSGISPyException(
            f"{label} target column '{target_col}' must contain unsigned "
            "(non-negative) integer class IDs."
        )
    if valid_ids is not None:
        unknown = set(np.unique(y_arr).tolist()) - set(valid_ids)
        if unknown:
            raise rsgispy.RSGISPyException(
                f"{label} contains class ID(s) {sorted(unknown)} not defined "
                "in cls_cfg."
            )
    return x_df, y_arr


def resolve_cv_splitter(
    cv: Union[int, Any],
    train_y: np.ndarray,
    rnd_seed: Optional[int],
) -> Any:
    """Resolve an integer or sklearn CV splitter cleanly."""
    if isinstance(cv, (int, np.integer)) and not isinstance(cv, bool):
        n_splits = int(cv)
        if n_splits < 2:
            raise rsgispy.RSGISPyException(
                f"cv must be at least 2 folds, got {n_splits}."
            )
        _, counts = np.unique(train_y, return_counts=True)
        min_count = int(counts.min()) if counts.size > 0 else 0
        if min_count < n_splits:
            raise rsgispy.RSGISPyException(
                f"cv={n_splits} exceeds minimum class sample count "
                f"({min_count}) in train_data."
            )
        return sklearn.model_selection.StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=rnd_seed,
        )
    return cv


def calc_split_metrics(
    fitted_est: Any,
    x_df: pd.DataFrame,
    y_true: np.ndarray,
    prefix: str,
    predict_kwargs: Optional[Mapping[str, Any]] = None,
) -> Dict[str, float]:
    """Compute accuracy, balanced accuracy, macro F1, and kappa for a split."""
    p_kwargs = dict(predict_kwargs) if predict_kwargs else {}
    preds = np.asarray(
        fitted_est.predict(x_df, **p_kwargs) if p_kwargs else fitted_est.predict(x_df)
    ).ravel()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        acc = float(sklearn.metrics.accuracy_score(y_true, preds))
        bal_acc = float(sklearn.metrics.balanced_accuracy_score(y_true, preds))
        f1_mac = float(
            sklearn.metrics.f1_score(
                y_true, preds, average="macro", zero_division=0
            )
        )
        kappa = float(sklearn.metrics.cohen_kappa_score(y_true, preds))
    if np.isnan(kappa):
        kappa = 1.0 if np.array_equal(y_true, preds) else 0.0
    return {
        f"{prefix}_accuracy": acc,
        f"{prefix}_bal_accuracy": bal_acc,
        f"{prefix}_f1_macro": f1_mac,
        f"{prefix}_cohen_kappa": kappa,
    }


def resolve_apply_band_specs(
    cfg: Optional["ClassificationConfig"],
    img_band_info: Optional[Sequence["ImageBandInfo"]],
) -> List[Tuple[str, int, str]]:
    """
    Resolve ordered ``(img_path, band_1based, col_name)`` specs for image
    classification and clustering.
    """
    # pylint: disable=import-outside-toplevel
    from rsgispy.classification.classdata import resolve_img_band_info

    info_seq = (
        img_band_info
        if img_band_info is not None
        else (cfg.img_band_info if cfg is not None else None)
    )
    if not info_seq:
        raise rsgispy.RSGISPyException(
            "img_band_info must be provided either inside ClassificationConfig "
            "or via the img_band_info parameter."
        )
    resolved = resolve_img_band_info(info_seq)
    band_specs: List[Tuple[str, int, str]] = []
    for img_path, bands, col_names in resolved:
        for b_idx, col_name in zip(bands, col_names):
            band_specs.append((img_path, int(b_idx), col_name))
    return band_specs


def resolve_out_nodata(
    out_no_data_val: Optional[int],
    target_ids: Sequence[int],
) -> int:
    """Resolve an unsigned nodata sentinel value avoiding class ID collisions."""
    if out_no_data_val is not None:
        if isinstance(out_no_data_val, bool) or not isinstance(
            out_no_data_val, (int, np.integer)
        ):
            raise rsgispy.RSGISPyException(
                f"out_no_data_val must be an integer, got {out_no_data_val!r}."
            )
        nd_val = int(out_no_data_val)
        if nd_val < 0:
            raise rsgispy.RSGISPyException(
                "out_no_data_val must be an unsigned (non-negative) integer, "
                f"got {nd_val}."
            )
        return nd_val

    used = {int(v) for v in target_ids}
    if 0 not in used:
        return 0
    if 255 not in used and max(used, default=0) < 255:
        return 255
    return max(used) + 1 if used else 0


def read_feature_block(
    band_specs: Sequence[Tuple[str, int, str]],
    datasets: Mapping[str, DatasetReader],
    ref_ds: DatasetReader,
    grid: IntersectionGrid,
    out_win: Window,
    all_compatible: bool,
    ref_path: str,
    explicit_no_data: Optional[float] = None,
) -> List[Tuple[np.ndarray, Optional[float]]]:
    """Read aligned or on-the-fly reprojected feature band arrays for a block."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    col_off = int(out_win.col_off)
    row_off = int(out_win.row_off)
    win_w = int(out_win.width)
    win_h = int(out_win.height)
    out_arrays: List[Tuple[np.ndarray, Optional[float]]] = []

    exp_nd: Optional[float] = (
        float(explicit_no_data) if explicit_no_data is not None else None
    )
    win_xfm = None
    for img_path, band_idx, _ in band_specs:
        src_ds = datasets[img_path]
        if exp_nd is not None:
            nd_val: Optional[float] = exp_nd
        else:
            raw_nd = src_ds.nodatavals[band_idx - 1]
            nd_val = None
            if raw_nd is not None and not (
                isinstance(raw_nd, float) and np.isnan(raw_nd)
            ):
                nd_val = float(raw_nd)

        if all_compatible:
            base_col, base_row = grid.offsets[img_path]
            in_win = Window(
                base_col + col_off, base_row + row_off, win_w, win_h
            )
            arr = src_ds.read(band_idx, window=in_win, out_dtype=np.float64)
        elif grids_compatible(ref_ds, src_ds):
            base_col, base_row = grid.offsets.get(
                img_path,
                (
                    int(
                        round(
                            (
                                ~src_ds.transform
                                @ (grid.transform.c, grid.transform.f)
                            )[0]
                        )
                    ),
                    int(
                        round(
                            (
                                ~src_ds.transform
                                @ (grid.transform.c, grid.transform.f)
                            )[1]
                        )
                    ),
                ),
            )
            in_win = Window(
                base_col + col_off, base_row + row_off, win_w, win_h
            )
            arr = src_ds.read(band_idx, window=in_win, out_dtype=np.float64)
        else:
            if win_xfm is None:
                base_ref_col, base_ref_row = grid.offsets[ref_path]
                ref_win = Window(
                    base_ref_col + col_off,
                    base_ref_row + row_off,
                    win_w,
                    win_h,
                )
                win_xfm = window_transform(ref_win, ref_ds.transform)
            arr = np.full((win_h, win_w), np.nan, dtype=np.float64)
            reproj_nd = (
                None
                if (nd_val is not None and np.isnan(nd_val))
                else nd_val
            )
            rio_reproject(
                source=rasterio.band(src_ds, band_idx),
                destination=arr,
                src_transform=src_ds.transform,
                src_crs=src_ds.crs,
                src_nodata=reproj_nd,
                dst_transform=win_xfm,
                dst_crs=grid.crs,
                dst_nodata=np.nan,
                resampling=Resampling.nearest,
            )
            # Reprojected invalid pixels are already NaN; don't re-test src nd.
            nd_val = None

        out_arrays.append((arr, nd_val))
    return out_arrays


def finalise_thematic_output_img(
    out_cls_path: str,
    eff_nodata: int,
    out_rio_dtype: str,
    out_np_dtype: np.dtype,
    calc_img_stats: bool = True,
    add_clr_tab: bool = False,
    clr_lut: Optional[Dict[int, List[int]]] = None,
    use_pbar: bool = True,
) -> None:
    """Set thematic metadata, nodata, stats/pyramids, and optional colour table."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    set_img_thematic(out_cls_path)
    set_img_no_data_value(out_cls_path, eff_nodata, img_band=1)
    if calc_img_stats:
        if out_rio_dtype in _INT_DTYPES:
            pop_thmt_img_stats(
                out_cls_path,
                add_clr_tab=bool(add_clr_tab and clr_lut is None),
                calc_pyramids=True,
                ignore_zero=(eff_nodata == 0),
                use_pbar=use_pbar,
            )
        else:
            build_img_pyramids(out_cls_path)
        set_img_no_data_value(out_cls_path, eff_nodata, img_band=1)

    if clr_lut is not None and out_np_dtype in (
        np.dtype(np.uint8),
        np.dtype(np.uint16),
    ):
        define_colour_table(out_cls_path, clr_lut, img_band=1)
