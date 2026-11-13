"""
The tools.accuracy module provides classification and regression accuracy
assessment utilities.

Functions requiring ``scipy`` and/or ``scikit-learn`` are only defined and
exported when those optional dependencies are installed.
"""

from typing import Any, List, Sequence, Tuple, Union

import numpy

import rsgispy

try:
    import scipy.stats  # pylint: disable=import-error

    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

try:
    import sklearn.metrics  # pylint: disable=import-error

    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False

__all__ = [
    "bias_score",
    "decompose_bias_variance",
]

if HAVE_SKLEARN:
    __all__.append("accuracy_scores_binary")

if HAVE_SCIPY and HAVE_SKLEARN:
    __all__.append("bin_accuracy_scores_prob")


if HAVE_SCIPY and HAVE_SKLEARN:

    def bin_accuracy_scores_prob(
        y_true: Union[numpy.ndarray, Sequence[int]],
        y_prob: Union[numpy.ndarray, Sequence[float]],
    ) -> List[Any]:
        """
        Calculate probabilistic binary classification accuracy metrics using
        scikit-learn and SciPy.

        :param y_true: 1D binary class labels (0 = absence, 1 = presence).
        :param y_prob: 1D predicted probability scores.
        :return: ``[metric_names, scores_array]``.
        """
        # pylint: disable=too-many-locals
        y_true_arr = numpy.asarray(y_true)
        y_prob_arr = numpy.asarray(y_prob)
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

        y_true_u8 = y_true_arr.astype("uint8")
        if numpy.min(y_true_u8) != 0 or numpy.max(y_true_u8) != 1:
            raise rsgispy.RSGISPyException(
                "ERROR: the true labels are not binary (zero or one values)."
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
        scores = numpy.array(
            [roc_auc, r_val, average_precision, brier_score, log_loss_score]
        ).round(decimals=6)
        return [metrics, scores]


if HAVE_SKLEARN:

    def accuracy_scores_binary(
        y_true: Union[numpy.ndarray, Sequence[int]],
        y_pred: Union[numpy.ndarray, Sequence[int]],
    ) -> List[numpy.ndarray]:
        """
        Calculate accuracy metrics for a binary classification.

        :param y_true: Observed 1D binary labels (0 = absence, 1 = presence).
        :param y_pred: Predicted 1D binary labels (0 = absence, 1 = presence).
        :return: ``[metrics_array, scores_array]``.
        """
        # pylint: disable=too-many-locals
        y_true_arr = numpy.asarray(y_true)
        y_pred_arr = numpy.asarray(y_pred)
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

        y_true_u8 = y_true_arr.astype("uint8")
        y_pred_u8 = y_pred_arr.astype("uint8")
        if numpy.min(y_true_u8) != 0 or numpy.max(y_true_u8) != 1:
            raise rsgispy.RSGISPyException(
                "ERROR: the true labels are not binary (zero or one values)."
            )
        if numpy.min(y_pred_u8) != 0 or numpy.max(y_pred_u8) != 1:
            raise rsgispy.RSGISPyException(
                "ERROR: the predicted labels are not binary (zero or one values)."
            )

        metrics = numpy.array(
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

        try:
            n_presence = numpy.where(y_true_u8 == 1)[0].size
            n_absence = numpy.where(y_true_u8 == 0)[0].size

            tp_cnt = numpy.where((y_true_u8 == 1) & (y_pred_u8 == 1))[0].size
            ta_cnt = numpy.where((y_true_u8 == 0) & (y_pred_u8 == 0))[0].size
            fp_cnt = numpy.where((y_true_u8 == 1) & (y_pred_u8 == 0))[0].size
            fa_cnt = numpy.where((y_true_u8 == 0) & (y_pred_u8 == 1))[0].size

            prevalence = (tp_cnt / fa_cnt) / y_true_u8.size
            odp = 1.0 - prevalence
            ccr = (tp_cnt + ta_cnt) / y_true_u8.size
            mr_val = (fp_cnt + fa_cnt) / y_true_u8.size

            sensitivity = tp_cnt / n_presence
            ppp = tp_cnt / (tp_cnt + fp_cnt)
            app = ta_cnt / (ta_cnt + fa_cnt)
            specificity = ta_cnt / n_absence

            accuracy = (tp_cnt + ta_cnt) / (n_presence + n_absence)
            balanced_accuracy = ((tp_cnt / n_presence) + (ta_cnt / n_absence)) / 2.0
            precision = tp_cnt / (tp_cnt + fp_cnt)
            f1_val = 2.0 * tp_cnt / ((2.0 * tp_cnt) + fp_cnt + fa_cnt)
            mcc = ((tp_cnt * ta_cnt) - (fp_cnt * fa_cnt)) / (
                (
                    (tp_cnt + fp_cnt)
                    * (tp_cnt + fa_cnt)
                    * (ta_cnt + fp_cnt)
                    * (ta_cnt + fa_cnt)
                )
                ** 0.5
            )
            tss = sensitivity + specificity - 1.0
            nmi_val = sklearn.metrics.normalized_mutual_info_score(
                y_true_u8, y_pred_u8
            )
            kappa_val = sklearn.metrics.cohen_kappa_score(y_true_u8, y_pred_u8)

            scores = numpy.array(
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
                ]
            ).round(decimals=6)
        except Exception:  # pylint: disable=broad-exception-caught
            scores = numpy.zeros(len(metrics))

        if metrics.size != scores.size:
            raise rsgispy.RSGISPyException(
                "ERROR: unable to calculate accuracy metrics."
            )
        return [metrics, scores]


def bias_score(
    y_true: Union[numpy.ndarray, Sequence[float]],
    y_pred: Union[numpy.ndarray, Sequence[float]],
) -> Tuple[Any, Any]:
    """
    Calculate regression model bias and normalised (percentage) bias.

    :param y_true: Array of true target values.
    :param y_pred: Array of predicted target values.
    :return: ``(bias, norm_bias)`` tuple.
    """
    y_true_arr = numpy.asarray(y_true)
    y_pred_arr = numpy.asarray(y_pred)
    if y_true_arr.ndim != y_pred_arr.ndim:
        raise rsgispy.RSGISPyException("Error: y_true dimensions != y_pred.")
    if y_true_arr.size != y_pred_arr.size:
        raise rsgispy.RSGISPyException("Error: y_true size != y_pred.")

    bias = numpy.mean(y_pred_arr - y_true_arr, axis=0)
    norm_bias = (bias / numpy.mean(y_true_arr, axis=0)) * 100.0
    return bias, norm_bias


def decompose_bias_variance(
    y_true: Union[numpy.ndarray, Sequence[float]],
    y_pred: Union[numpy.ndarray, Sequence[float]],
) -> Tuple[Any, Any, Any, Any]:
    """
    Decompose MSE into ``Bias^2 + Variance + Irreducible Error``.

    :param y_true: Array of true target values.
    :param y_pred: Array of predicted target values.
    :return: ``(mse, bias_squared, variance, noise)`` tuple.
    """
    y_true_arr = numpy.asarray(y_true)
    y_pred_arr = numpy.asarray(y_pred)
    if y_true_arr.ndim != y_pred_arr.ndim:
        raise rsgispy.RSGISPyException("Error: y_true dimensions != y_pred.")
    if y_true_arr.size != y_pred_arr.size:
        raise rsgispy.RSGISPyException("Error: y_true size != y_pred.")

    mse = numpy.mean((y_pred_arr - y_true_arr) ** 2, axis=0)
    bias_squared = numpy.mean(y_pred_arr - y_true_arr, axis=0) ** 2
    variance = numpy.var(y_pred_arr - y_true_arr, axis=0)
    noise = mse - (bias_squared + variance)
    if isinstance(noise, numpy.ndarray):
        noise[noise < 0] = 0
    elif noise < 0:
        noise = 0
    return mse, bias_squared, variance, noise
