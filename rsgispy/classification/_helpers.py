"""
Internal shared helpers for :mod:`rsgispy.classification`.
"""

import csv
import math
import os
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

import rsgispy
import rsgispy.imagecalc.stats
from rsgispy.imageprocess import (
    _INT_DTYPES,
    ensure_file_exists,
)
from rsgispy.imageutils.colourtable import define_colour_table, set_img_thematic
import rsgispy.imageutils.metadata
from rsgispy.imageutils.metadata import set_img_no_data_value
from rsgispy.imageutils.stats import build_img_pyramids, pop_thmt_img_stats
from rsgispy.tools._ml_data import (
    resolve_apply_band_specs_from_seq,
    resolve_feature_cols as _tools_resolve_feature_cols,
    seg_model_features,
)
from rsgispy.tools._ml_data import (  # noqa: F401  # pylint: disable=unused-import
    seg_allows_nan,
)
from rsgispy.tools._ml_data import (  # noqa: F401  # pylint: disable=unused-import
    load_samples_df,
)
from rsgispy.tools._ml_data import (  # noqa: F401  # pylint: disable=unused-import
    read_feature_block,
)
from rsgispy.tools._ml_data import (  # noqa: F401  # pylint: disable=unused-import
    resolve_rng,
)
from rsgispy.tools._ml_data import (  # noqa: F401  # pylint: disable=unused-import
    resolve_bootstrap_sample_size,
)
from rsgispy.tools._ml_data import (  # noqa: F401  # pylint: disable=unused-import
    summarise_bootstrap_interval,
)
from rsgispy.tools._ml_tune import resolve_cv_splitter as _tools_resolve_cv_splitter
from rsgispy.tools._sklearn import (
    check_sklearn_available as _check_sklearn_available,
    sklearn,
)
from rsgispy.tools.files import ensure_parent_dir
import rsgispy.tools.strings
from rsgispy.vector.attrs import read_vec_columns
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
    """
    Compute point-only classification accuracy metrics dictionary.

    ``commission`` and ``omission`` are percentage error rates of each class
    (100 minus the user's / producer's accuracy).
    """
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
    """
    Compute area-weighted classification accuracy metrics dictionary.

    ``commission`` and ``omission`` are percentage error rates of each class
    (100 minus the user's / producer's accuracy of the area-normalised
    confusion matrix, as in Olofsson et al. 2014), on the same scale as
    :func:`calc_class_pt_accuracy_metrics`. ``commission_area_prop`` and
    ``omission_area_prop`` give the same errors as proportions (0-1) of the
    total mapped area.
    """
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

    map_total = norm_cm.sum(axis=1)
    ref_total = norm_cm.sum(axis=0)
    diag = np.diag(norm_cm)
    commission_prop = map_total - diag
    omission_prop = ref_total - diag
    cls_area_prop = ref_total

    acc_metrics["norm_confusion_matrix"] = norm_cm.tolist()
    # Commission / omission are percentage error rates (100 - user's /
    # producer's accuracy of the area-normalised matrix), on the same scale
    # as calc_class_pt_accuracy_metrics. The error as a proportion of the
    # total mapped area is in the '*_area_prop' keys.
    acc_metrics["commission"] = [
        _safe_divide_scalar(float(err), float(tot)) * 100.0
        for err, tot in zip(commission_prop, map_total)
    ]
    acc_metrics["omission"] = [
        _safe_divide_scalar(float(err), float(tot)) * 100.0
        for err, tot in zip(omission_prop, ref_total)
    ]
    acc_metrics["commission_area_prop"] = [float(val) for val in commission_prop]
    acc_metrics["omission_area_prop"] = [float(val) for val in omission_prop]
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

    def _norm_name(name: Any) -> str:
        # LUT and sample names are compared after the same cleaning, so
        # names with non-ASCII characters (e.g. 'Wald-Fläche') still match.
        return rsgispy.tools.strings.check_str(str(name), rm_non_ascii=True)

    area_by_norm: Dict[str, float] = {}
    for cls_id, pxl_ct in zip(pxl_vals, n_pxls):
        norm_key = _norm_name(class_names_lut[cls_id])
        area_by_norm[norm_key] = area_by_norm.get(norm_key, 0.0) + (
            float(pxl_ct) * pxl_area
        )

    missing = [
        name for name in unq_cls_names if _norm_name(name) not in area_by_norm
    ]
    if missing:
        raise rsgispy.RSGISPyException(
            f"Did not calculate the class areas for: {', '.join(missing)}"
        )
    cls_area_dict: Dict[str, float] = {
        name: area_by_norm[_norm_name(name)] for name in unq_cls_names
    }

    cls_id_areas = np.array(
        [cls_area_dict[cls_name] for cls_name in unq_cls_names],
        dtype=float,
    )
    if float(cls_id_areas.sum()) <= 0.0:
        raise rsgispy.RSGISPyException(
            "Total mapped pixel area across classes in cls_img is zero."
        )
    return cls_area_dict, cls_id_areas


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
        writer.writerow(
            [
                "class",
                "commission (%)",
                "omission (%)",
                "commission (prop. area)",
                "omission (prop. area)",
            ]
        )
        for idx, cls_name in enumerate(unq_cls_names):
            writer.writerow(
                [
                    cls_name,
                    acc_metrics["commission"][idx],
                    acc_metrics["omission"][idx],
                    acc_metrics["commission_area_prop"][idx],
                    acc_metrics["omission_area_prop"][idx],
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


def resolve_feature_cols(
    cfg: Optional["ClassificationConfig"],
    df: pd.DataFrame,
    feature_cols: Optional[Sequence[str]],
    excluded_cols: Sequence[str],
) -> List[str]:
    """
    Determine and validate ordered feature column names for sample tables
    (for segment-based configs, the ``seg_info`` model features by default).
    """
    if feature_cols is None:
        feature_cols = seg_model_features(cfg)
    img_info = cfg.img_band_info if cfg is not None else None
    return _tools_resolve_feature_cols(
        cfg_img_band_info=img_info,
        df=df,
        feature_cols=feature_cols,
        excluded_cols=excluded_cols,
    )


def extract_xy(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    label: str,
    valid_ids: Optional[Sequence[int]] = None,
    allow_nan: bool = False,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Extract validated ``(X_df, y_arr)`` arrays from a sample DataFrame (NaN
    feature values are only accepted when ``allow_nan=True``).
    """
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
    bad_vals = np.isinf(x_vals) if allow_nan else ~np.isfinite(x_vals)
    if np.any(bad_vals):
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
    return _tools_resolve_cv_splitter(cv, train_y, rnd_seed, stratified=True)


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
    return resolve_apply_band_specs_from_seq(info_seq)


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


def finalise_thematic_output_img(
    out_cls_path: str,
    eff_nodata: int,
    out_rio_dtype: str,
    out_np_dtype: np.dtype,
    calc_stats: bool = True,
    add_clr_tab: bool = False,
    clr_lut: Optional[Dict[int, List[int]]] = None,
    use_pbar: bool = True,
) -> None:
    """Set thematic metadata, nodata, stats/pyramids, and optional colour table."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    set_img_thematic(out_cls_path)
    set_img_no_data_value(out_cls_path, eff_nodata, img_band=1)
    if calc_stats:
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
