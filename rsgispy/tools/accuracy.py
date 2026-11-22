"""
The tools.accuracy module provides classification and regression accuracy
assessment utilities.

Functions requiring ``scipy`` and/or ``scikit-learn`` raise
:class:`rsgispy.RSGISPyException` if those optional dependencies are not
installed.
"""

from typing import Any, List, Sequence, Tuple, Union
import warnings

import numpy as np

import rsgispy

try:
    import scipy.stats  # pylint: disable=import-error

    HAVE_SCIPY = True
except ImportError:
    scipy = None  # type: ignore[assignment]
    HAVE_SCIPY = False

try:
    import sklearn.metrics  # pylint: disable=import-error

    HAVE_SKLEARN = True
except ImportError:
    sklearn = None  # type: ignore[assignment]
    HAVE_SKLEARN = False

__all__ = [
    "bias_score",
    "decompose_bias_variance",
    "accuracy_scores_binary",
    "bin_accuracy_scores_prob",
]


def _validate_binary_labels(
    arr: np.ndarray,
    label_desc: str,
    require_both_classes: bool = False,
) -> np.ndarray:
    """Validate that a 1D array strictly contains ``{0, 1}`` values before uint8 cast."""
    if arr.size == 0:
        raise rsgispy.RSGISPyException(
            f"ERROR: the {label_desc} array is empty."
        )
    if np.issubdtype(arr.dtype, np.bool_):
        out = arr.astype("uint8")
    else:
        if not np.issubdtype(arr.dtype, np.number):
            raise rsgispy.RSGISPyException(
                f"ERROR: the {label_desc} are not binary (zero or one values)."
            )
        if not np.all(np.isfinite(arr)) or not np.all((arr == 0) | (arr == 1)):
            raise rsgispy.RSGISPyException(
                f"ERROR: the {label_desc} are not binary (zero or one values)."
            )
        out = arr.astype("uint8")

    if require_both_classes and (np.min(out) != 0 or np.max(out) != 1):
        raise rsgispy.RSGISPyException(
            f"ERROR: the {label_desc} are not binary (zero or one values)."
        )
    return out


def bin_accuracy_scores_prob(
    y_true: Union[np.ndarray, Sequence[int]],
    y_prob: Union[np.ndarray, Sequence[float]],
) -> List[Any]:
    """
    Calculate probabilistic binary classification accuracy metrics using
    scikit-learn and SciPy.

    :param y_true: 1D binary class labels (0 = absence, 1 = presence).
    :param y_prob: 1D predicted probability scores.
    :return: ``[metric_names, scores_array]``.

    .. code:: python

        import rsgispy.tools.accuracy
        names, scores = rsgispy.tools.accuracy.bin_accuracy_scores_prob(
            [0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8]
        )
    """
    # pylint: disable=too-many-locals
    rsgispy.check_optional_dep(
        HAVE_SKLEARN, "bin_accuracy_scores_prob", "scikit-learn", "stats"
    )
    rsgispy.check_optional_dep(
        HAVE_SCIPY, "bin_accuracy_scores_prob", "scipy", "stats"
    )
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob)
    if y_true_arr.ndim != 1:
        raise rsgispy.RSGISPyException(
            "ERROR: the true labels are not in a 1D array."
        )
    if y_prob_arr.ndim != 1:
        raise rsgispy.RSGISPyException(
            "ERROR: the probability of presence values are not in a 1D array."
        )
    if y_true_arr.size != y_prob_arr.size:
        raise rsgispy.RSGISPyException(
            "ERROR: unequal number of binary labels and probabilities."
        )

    y_true_u8 = _validate_binary_labels(
        y_true_arr, "true labels", require_both_classes=True
    )

    roc_auc = sklearn.metrics.roc_auc_score(y_true_u8, y_prob_arr)
    average_precision = sklearn.metrics.average_precision_score(
        y_true_u8, y_prob_arr
    )
    brier_score = sklearn.metrics.brier_score_loss(y_true_u8, y_prob_arr)
    log_loss_score = sklearn.metrics.log_loss(y_true_u8, y_prob_arr)
    r_val = scipy.stats.pointbiserialr(y_true_u8, y_prob_arr)[0]

    metrics = [
        "Test AUC",
        "Point-Biserial r",
        "Av. Precision",
        "Brier Score",
        "Log-Loss Score",
    ]
    scores = np.array(
        [roc_auc, r_val, average_precision, brier_score, log_loss_score]
    ).round(decimals=6)
    return [metrics, scores]


def accuracy_scores_binary(
    y_true: Union[np.ndarray, Sequence[int]],
    y_pred: Union[np.ndarray, Sequence[int]],
) -> List[np.ndarray]:
    """
    Calculate accuracy metrics for a binary classification.

    :param y_true: Observed 1D binary labels (0 = absence, 1 = presence).
    :param y_pred: Predicted 1D binary labels (0 = absence, 1 = presence).
    :return: ``[metrics_array, scores_array]``.

    .. code:: python

        import rsgispy.tools.accuracy
        metrics, scores = rsgispy.tools.accuracy.accuracy_scores_binary(
            [0, 1, 0, 1], [0, 1, 1, 1]
        )
    """
    # pylint: disable=too-many-locals
    rsgispy.check_optional_dep(
        HAVE_SKLEARN, "accuracy_scores_binary", "scikit-learn", "stats"
    )
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    if y_true_arr.ndim != 1:
        raise rsgispy.RSGISPyException(
            "ERROR: the true labels are not in a 1D array."
        )
    if y_pred_arr.ndim != 1:
        raise rsgispy.RSGISPyException(
            "ERROR: the predicted labels are not in a 1D array."
        )
    if y_true_arr.size != y_pred_arr.size:
        raise rsgispy.RSGISPyException(
            "ERROR: unequal number of binary labels."
        )

    y_true_u8 = _validate_binary_labels(
        y_true_arr, "true labels", require_both_classes=False
    )
    y_pred_u8 = _validate_binary_labels(
        y_pred_arr, "predicted labels", require_both_classes=False
    )

    metrics = np.array(
        [
            "Prevalence",
            "Overall Diagnostic Power",
            "Correct Classification Rate",
            "Misclassification Rate",
            "Presence Predictive Power",
            "Absence Predictive Power",
            "Accuracy",
            "Balanced Accuracy",
            "Sensitivity",
            "Specificity",
            "Precision",
            "F1 Score",
            "Matthews Correlation",
            "Cohen Kappa",
            "Normalised Mutual Information",
            "Hanssen-Kuiper skill",
        ]
    )

    n_total = y_true_u8.size
    n_presence = int(np.count_nonzero(y_true_u8 == 1))
    n_absence = int(np.count_nonzero(y_true_u8 == 0))

    tp_cnt = int(np.count_nonzero((y_true_u8 == 1) & (y_pred_u8 == 1)))
    ta_cnt = int(np.count_nonzero((y_true_u8 == 0) & (y_pred_u8 == 0)))
    fp_cnt = int(np.count_nonzero((y_true_u8 == 0) & (y_pred_u8 == 1)))
    fa_cnt = int(np.count_nonzero((y_true_u8 == 1) & (y_pred_u8 == 0)))

    prevalence = n_presence / n_total
    odp = 1.0 - prevalence
    ccr = (tp_cnt + ta_cnt) / n_total
    mr_val = (fp_cnt + fa_cnt) / n_total

    sensitivity = tp_cnt / n_presence if n_presence > 0 else 0.0
    ppp = tp_cnt / (tp_cnt + fp_cnt) if (tp_cnt + fp_cnt) > 0 else 0.0
    app = ta_cnt / (ta_cnt + fa_cnt) if (ta_cnt + fa_cnt) > 0 else 0.0
    specificity = ta_cnt / n_absence if n_absence > 0 else 0.0

    accuracy = (tp_cnt + ta_cnt) / n_total
    if n_presence > 0 and n_absence > 0:
        balanced_accuracy = (sensitivity + specificity) / 2.0
        tss = sensitivity + specificity - 1.0
    else:
        balanced_accuracy = sensitivity if n_presence > 0 else specificity
        tss = 0.0
    precision = ppp
    f1_denom = (2.0 * tp_cnt) + fp_cnt + fa_cnt
    f1_val = (2.0 * tp_cnt) / f1_denom if f1_denom > 0 else 0.0
    mcc_denom = (
        (tp_cnt + fp_cnt)
        * (tp_cnt + fa_cnt)
        * (ta_cnt + fp_cnt)
        * (ta_cnt + fa_cnt)
    ) ** 0.5
    mcc = (
        ((tp_cnt * ta_cnt) - (fp_cnt * fa_cnt)) / mcc_denom
        if mcc_denom > 0.0
        else 0.0
    )
    nmi_val = float(
        sklearn.metrics.normalized_mutual_info_score(y_true_u8, y_pred_u8)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kappa_val = float(
            sklearn.metrics.cohen_kappa_score(
                y_true_u8, y_pred_u8, labels=[0, 1]
            )
        )
    if not np.isfinite(kappa_val):
        kappa_val = 1.0 if np.array_equal(y_true_u8, y_pred_u8) else 0.0

    scores = np.array(
        [
            prevalence,
            odp,
            ccr,
            mr_val,
            ppp,
            app,
            accuracy,
            balanced_accuracy,
            sensitivity,
            specificity,
            precision,
            f1_val,
            mcc,
            kappa_val,
            nmi_val,
            tss,
        ],
        dtype=np.float64,
    ).round(decimals=6)

    if metrics.size != scores.size:
        raise rsgispy.RSGISPyException(
            "ERROR: unable to calculate accuracy metrics."
        )
    return [metrics, scores]


def bias_score(
    y_true: Union[np.ndarray, Sequence[float]],
    y_pred: Union[np.ndarray, Sequence[float]],
) -> Tuple[Any, Any]:
    """
    Calculate regression model bias and normalised (percentage) bias.

    :param y_true: Array of true target values.
    :param y_pred: Array of predicted target values.
    :return: ``(bias, norm_bias)`` tuple.

    .. code:: python

        import rsgispy.tools.accuracy
        bias, norm_bias = rsgispy.tools.accuracy.bias_score(
            [1.0, 2.0, 3.0], [1.1, 2.2, 2.9]
        )
    """
    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64)
    if y_true_arr.ndim != y_pred_arr.ndim:
        raise rsgispy.RSGISPyException("Error: y_true dimensions != y_pred.")
    if y_true_arr.shape != y_pred_arr.shape:
        raise rsgispy.RSGISPyException("Error: y_true size != y_pred.")

    bias = np.mean(y_pred_arr - y_true_arr, axis=0)
    mean_true = np.mean(y_true_arr, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        norm_bias = (bias / mean_true) * 100.0
    return bias, norm_bias


def decompose_bias_variance(
    y_true: Union[np.ndarray, Sequence[float]],
    y_pred: Union[np.ndarray, Sequence[float]],
) -> Tuple[Any, Any, Any, Any]:
    """
    Decompose empirical MSE of paired residuals ``e = y_pred - y_true`` into
    ``Bias^2 + Error Variance + Residual``.

    For a single set of paired predictions ``(y_true, y_pred)``, the identity
    ``MSE = E[e]^2 + Var(e)`` holds identically (up to floating-point rounding),
    where ``bias_squared = E[y_pred - y_true]^2`` is squared mean bias,
    ``variance = Var(y_pred - y_true)`` is the variance of the prediction errors,
    and ``noise`` is the non-negative numerical residual ``max(0, MSE - (Bias^2 + Var))``.

    :param y_true: Array of true target values.
    :param y_pred: Array of predicted target values.
    :return: ``(mse, bias_squared, variance, noise)`` tuple.

    .. code:: python

        import rsgispy.tools.accuracy
        mse, bias_sq, var, noise = rsgispy.tools.accuracy.decompose_bias_variance(
            [1.0, 2.0, 3.0], [1.1, 2.2, 2.9]
        )
    """
    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64)
    if y_true_arr.ndim != y_pred_arr.ndim:
        raise rsgispy.RSGISPyException("Error: y_true dimensions != y_pred.")
    if y_true_arr.shape != y_pred_arr.shape:
        raise rsgispy.RSGISPyException("Error: y_true size != y_pred.")

    errors = y_pred_arr - y_true_arr
    mse = np.mean(errors**2, axis=0)
    bias_squared = np.mean(errors, axis=0) ** 2
    variance = np.var(errors, axis=0)
    noise = mse - (bias_squared + variance)
    if isinstance(noise, np.ndarray):
        noise[noise < 0] = 0.0
    elif noise < 0:
        noise = 0.0
    return mse, bias_squared, variance, noise
