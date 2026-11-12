"""
The tools.stats module provides statistical analysis, feature selection,
accuracy assessment, thresholding, and data normalisation utilities.

Functions requiring ``scipy`` and/or ``scikit-learn`` are only defined and
exported when those optional dependencies are installed.
"""

# pylint: disable=too-many-lines

from typing import Any, List, Optional, Sequence, Tuple, Union
import warnings

import numpy

import rsgispy

try:
    import scipy.optimize  # pylint: disable=import-error
    import scipy.stats  # pylint: disable=import-error

    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

try:
    import sklearn.cluster  # pylint: disable=import-error
    import sklearn.exceptions  # pylint: disable=import-error
    import sklearn.feature_selection  # pylint: disable=import-error
    import sklearn.linear_model  # pylint: disable=import-error
    import sklearn.metrics  # pylint: disable=import-error

    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False

__all__ = [
    "cqv_threshold",
    "get_nbins_histogram",
    "get_bin_centres",
    "calc_otsu_threshold",
    "calc_yen_threshold",
    "calc_isodata_threshold",
    "calc_hist_cross_entropy",
    "calc_li_threshold",
    "mask_data_to_valid",
    "bias_score",
    "decompose_bias_variance",
    "calc_semivariogram",
    "standarise_img_data",
    "normalise_img_data",
]

if HAVE_SKLEARN:
    __all__.extend(
        [
            "calc_pandas_vif",
            "lassolars_feature_selection",
            "accuracy_scores_binary",
        ]
    )

if HAVE_SCIPY:
    __all__.append("calc_kurt_skew_threshold")

if HAVE_SCIPY and HAVE_SKLEARN:
    __all__.extend(
        [
            "corr_feature_selection",
            "breusch_pagan_test",
            "bin_accuracy_scores_prob",
        ]
    )


if HAVE_SKLEARN:

    def calc_pandas_vif(
        df: Any,
        cols: Optional[List[str]] = None,
    ) -> Any:
        """
        Calculate Variance Inflation Factors (VIF) to investigate multicollinearity
        between predictor variables in a pandas DataFrame.

        Interpretation of VIF scores:
        1 = no multicollinearity; 1-5 = moderate; > 5 = high; > 10 = remove predictor.

        :param df: Pandas DataFrame whose columns are predictor variables.
        :param cols: Optional subset of column names to evaluate (at least 2).
        :return: ``pandas.Series`` indexed by column name containing VIF scores.
        """
        # pylint: disable=import-outside-toplevel
        import pandas

        if cols is None:
            cols = list(df.columns)

        if len(cols) < 2:
            raise rsgispy.RSGISPyException(
                "The list of columns must be have a length of at least 2."
            )

        print(f"Calculating VIF for {len(cols)} predictors variables...")

        linear_model = sklearn.linear_model.LinearRegression()
        vif_scores = {}

        for col in cols:
            tmp_cols = list(cols)
            tmp_cols.remove(col)
            y_vals = df[col].values
            x_vals = df[tmp_cols].values
            linear_model.fit(x_vals, y_vals)
            y_pred = linear_model.predict(x_vals)

            ss_tot = float(numpy.sum((y_vals - numpy.mean(y_vals)) ** 2))
            ss_res = float(numpy.sum((y_vals - y_pred) ** 2))
            r2 = 1.0 - (ss_res / ss_tot)

            vif_scores[col] = 0.0 if r2 == 1.0 else 1.0 / (1.0 - r2)

        return pandas.Series(vif_scores, name="VIF")


def cqv_threshold(
    df: Any,
    cols: Optional[List[str]] = None,
    lowthreshold: float = 0.25,
    highthreshold: float = 0.75,
) -> List[str]:
    """
    Filter features based on the Coefficient of Quartile Variation (CQV).

    Predictors with CQV in ``[lowthreshold, highthreshold]`` are retained.

    :param df: Pandas DataFrame containing predictor columns.
    :param cols: Optional list of column names to inspect (at least 2).
    :param lowthreshold: Minimum CQV threshold (Default = 0.25).
    :param highthreshold: Maximum CQV threshold (Default = 0.75).
    :return: List of selected column names.
    """
    if cols is None:
        cols = list(df.columns)

    if len(cols) < 2:
        raise rsgispy.RSGISPyException(
            "The list of columns must be have a length of at least 2."
        )

    x_vals = df[cols].values

    print(f"Calculating CQV for {len(cols)} predictor variables...")
    q1 = numpy.percentile(x_vals, 25, axis=0)
    q3 = numpy.percentile(x_vals, 75, axis=0)
    cqv = (q3 - q1) / (q3 + q1)

    print(f"Median CQV: {numpy.median(cqv)}")

    good_idx = numpy.where((cqv >= lowthreshold) & (cqv <= highthreshold))[0]
    good_cols = [cols[i] for i in good_idx]

    print(f"Selected {len(good_cols)} useful predictors...")
    return good_cols


if HAVE_SCIPY and HAVE_SKLEARN:

    def corr_feature_selection(
        df: Any,
        dep_vars: List[str],
        ind_vars: List[str],
        n_min_clusters: int = 3,
        n_max_clusters: int = 25,
    ) -> List[str]:
        """
        Perform correlation-based feature selection by clustering independent
        variables using Pearson correlation distance and Silhouette selection.

        :param df: Pandas DataFrame containing dependent and independent variables.
        :param dep_vars: List of dependent variable column names.
        :param ind_vars: List of independent (predictor) variable column names.
        :param n_min_clusters: Minimum cluster count in the search (Default = 3).
        :param n_max_clusters: Exclusive upper cluster count bound (Default = 25).
        :return: List of selected predictor column names.
        """
        # pylint: disable=too-many-locals
        corr_data = []
        for dep_var in dep_vars:
            corr_vals = [
                df[dep_var].corr(df[ind_var], method="pearson") for ind_var in ind_vars
            ]
            corr_data.append(corr_vals)

        pearson_corr = numpy.tanh(numpy.mean(numpy.arctanh(corr_data), axis=0))

        def _pearson_affinity(arr: numpy.ndarray) -> numpy.ndarray:
            return 1.0 - numpy.array(
                [
                    [abs(scipy.stats.pearsonr(col_a, col_b)[0]) for col_a in arr]
                    for col_b in arr
                ]
            )

        def _cluster_features(feat_arr: numpy.ndarray, n_clst: int) -> numpy.ndarray:
            if feat_arr.ndim != 2:
                raise rsgispy.RSGISPyException("Error: the input array must be 2D.")
            clf = sklearn.cluster.FeatureAgglomeration(
                n_clusters=int(n_clst),
                metric=_pearson_affinity,
                linkage="complete",
            )
            clf.fit(feat_arr)
            return clf.labels_ + 1

        x_vals = df[ind_vars].values
        n_clusters = numpy.arange(n_min_clusters, n_max_clusters, dtype="int64")
        tqdm_cls = rsgispy.get_tqdm()
        cluster_score = []
        for n_clst in tqdm_cls(n_clusters):
            x_lbls = _cluster_features(x_vals, int(n_clst))
            cluster_score.append(
                sklearn.metrics.silhouette_score(x_vals.T, x_lbls, metric="correlation")
            )

        best_idx = int(numpy.argmax(cluster_score))
        opt_n_clusters = int(n_clusters[best_idx])
        opt_n_clusters_scr = cluster_score[best_idx]
        print(f"Found optimal number of clusters: {opt_n_clusters}")
        print(f"Silhouette Coefficient: {opt_n_clusters_scr}")

        cluster_lbls = _cluster_features(x_vals, opt_n_clusters)
        best_predictors: List[str] = []
        for lbl in numpy.unique(cluster_lbls):
            lbl_idx = numpy.where(cluster_lbls == lbl)[0]
            best_corr = int(numpy.argmax(pearson_corr[lbl_idx]))
            best_predictors.append(ind_vars[lbl_idx[best_corr]])

        return best_predictors


if HAVE_SKLEARN:

    def lassolars_feature_selection(
        df: Any,
        dep_vars: List[str],
        ind_vars: List[str],
        alpha_val: Optional[float] = None,
    ) -> List[str]:
        """
        Perform L1 regularisation-based feature selection using ``LassoLars`` in
        scikit-learn.

        :param df: Pandas DataFrame containing dependent and independent variables.
        :param dep_vars: List of dependent variable column names.
        :param ind_vars: List of independent (predictor) variable column names.
        :param alpha_val: Regularisation parameter ``alpha``; estimated via BIC
                          with ``LassoLarsIC`` when ``None``.
        :return: List of selected predictor column names.
        """
        # pylint: disable=too-many-locals
        x_vals = df[ind_vars].values
        y_vals = df[dep_vars].values

        if alpha_val is None:
            ic_est = sklearn.linear_model.LassoLarsIC("bic")
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", category=sklearn.exceptions.ConvergenceWarning
                )
                ic_est.fit(x_vals, y_vals[:, -1])
            optimal_alpha = float(ic_est.alpha_)
        else:
            optimal_alpha = float(alpha_val)

        print(
            f"Using regularization parameter (alpha) for the Lasso estimator of: "
            f"{optimal_alpha}"
        )

        estimator = sklearn.linear_model.LassoLars(alpha=optimal_alpha)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=sklearn.exceptions.ConvergenceWarning
            )
            estimator.fit(x_vals, y_vals)

        selector = sklearn.feature_selection.SelectFromModel(estimator, prefit=True)
        ind_vars_arr = numpy.array(ind_vars)
        return [str(col) for col in ind_vars_arr[selector.get_support()]]


if HAVE_SCIPY and HAVE_SKLEARN:

    def breusch_pagan_test(
        x: numpy.ndarray,
        y: numpy.ndarray,
    ) -> List[Union[float, str]]:
        """
        Perform a Breusch-Pagan test for heteroskedasticity in a linear model.

        :param x: Predictor array of shape ``(n_samples, n_predictors)``.
        :param y: 1D response array of shape ``(n_samples,)``.
        :return: ``[lm_stat, p_value, result_message]``.
        """
        # pylint: disable=too-many-locals
        if x.shape[0] != y.shape[0]:
            raise rsgispy.RSGISPyException(
                "Error: the number of samples differs between x and y."
            )
        n_samples = x.shape[0]

        linear_model = sklearn.linear_model.LinearRegression()
        linear_model.fit(x, y)

        err = (y - linear_model.predict(x)) ** 2

        linear_model.fit(x, err)
        pred_err = linear_model.predict(x)

        ss_tot = float(numpy.sum((err - numpy.mean(err, axis=0)) ** 2))
        ss_res = float(numpy.sum((err - pred_err) ** 2))
        r2 = 1.0 - (ss_res / ss_tot)

        lm_stat = float(n_samples * r2)
        pval = float(scipy.stats.chi2.sf(lm_stat, x.shape[1]))

        if pval < 0.01:
            test_result = "Heteroskedasticity present at 99% CI."
        elif pval < 0.05:
            test_result = "Heteroskedasticity present at 95% CI."
        else:
            test_result = "No significant heteroskedasticity."
        return [lm_stat, pval, test_result]

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


def get_nbins_histogram(data: numpy.ndarray) -> Tuple[int, float]:
    """
    Calculate the Freedman-Diaconis number of histogram bins and bin width.

    :param data: 1D numeric NumPy array.
    :return: ``(n_bins, bin_width)`` tuple.
    """
    n_pts = data.shape[0]
    lower_q = numpy.percentile(data, 25)
    upper_q = numpy.percentile(data, 75)
    iqr = upper_q - lower_q
    bin_width = 2.0 * iqr * n_pts ** (-1.0 / 3.0)
    n_bins = int((numpy.max(data) - numpy.min(data)) / bin_width) + 2
    return n_bins, float(bin_width)


def get_bin_centres(
    bin_edges: numpy.ndarray,
    geometric: bool = False,
) -> numpy.ndarray:
    """
    Calculate histogram bin centres from bin edges.

    :param bin_edges: 1D NumPy array of histogram bin edges.
    :param geometric: Compute geometric mean ``sqrt(min * max)`` when True, or
                      arithmetic mean ``(min + max) / 2`` otherwise.
    :return: 1D NumPy array of bin centres.
    """
    if geometric:
        return numpy.sqrt(bin_edges[1:] * bin_edges[:-1])
    return (bin_edges[1:] + bin_edges[:-1]) / 2.0


def calc_otsu_threshold(data: numpy.ndarray) -> float:
    """
    Calculate Otsu's threshold for a 1D numeric NumPy array.

    :param data: 1D numeric NumPy array.
    :return: Computed Otsu threshold value.
    """
    n_bins, _ = get_nbins_histogram(data)
    hist, bin_edges = numpy.histogram(data, bins=n_bins)
    bin_centres = get_bin_centres(bin_edges)

    hist_prob = hist.astype(numpy.float32)
    hist_prob = hist_prob / numpy.sum(hist_prob)

    weight1 = numpy.cumsum(hist_prob)
    weight2 = numpy.cumsum(hist_prob[::-1])[::-1]
    mean1 = numpy.cumsum(hist_prob * bin_centres) / weight1
    mean2 = (numpy.cumsum((hist_prob * bin_centres)[::-1]) / weight2[::-1])[::-1]

    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    return float(bin_centres[int(numpy.argmax(variance12))])


def calc_yen_threshold(data: numpy.ndarray) -> float:
    """
    Calculate Yen's multilevel threshold for a 1D numeric NumPy array.

    :param data: 1D numeric NumPy array.
    :return: Computed Yen threshold value.
    """
    n_bins, _ = get_nbins_histogram(data)
    hist, bin_edges = numpy.histogram(data, bins=n_bins)
    bin_centres = get_bin_centres(bin_edges)

    pmf = hist.astype(numpy.float32) / numpy.sum(hist)
    p1 = numpy.cumsum(pmf)
    p1_sq = numpy.cumsum(pmf**2)
    p2_sq = numpy.cumsum(pmf[::-1] ** 2)[::-1]

    crit = numpy.log(
        ((p1_sq[:-1] * p2_sq[1:]) ** -1) * (p1[:-1] * (1.0 - p1[:-1])) ** 2
    )
    return float(bin_centres[int(crit.argmax())])


def calc_isodata_threshold(data: numpy.ndarray) -> float:
    """
    Calculate the ISODATA (Ridler-Calvard) inter-means threshold for a 1D
    numeric NumPy array.

    :param data: 1D numeric NumPy array.
    :return: Computed ISODATA threshold value.
    """
    n_bins, _ = get_nbins_histogram(data)
    hist, bin_edges = numpy.histogram(data, bins=n_bins)
    bin_centres = get_bin_centres(bin_edges)

    hist_prob = hist.astype(numpy.float32)
    hist_prob = hist_prob / numpy.sum(hist_prob)

    csuml = numpy.cumsum(hist_prob)
    csum_intensity = numpy.cumsum(hist_prob * bin_centres)
    lower = csum_intensity[:-1] / csuml[:-1]
    higher = (csum_intensity[-1] - csum_intensity[:-1]) / (csuml[-1] - csuml)[:-1]

    distances = ((lower + higher) / 2.0) - bin_centres[:-1]
    bin_width = bin_centres[1] - bin_centres[0]
    thresholds = bin_centres[:-1][(distances >= 0) & (distances < bin_width)]
    return float(thresholds[0])


def calc_hist_cross_entropy(data: numpy.ndarray, threshold: float) -> float:
    """
    Compute cross-entropy between histogram distributions above and below
    ``threshold``.

    :param data: 1D numeric NumPy array.
    :param threshold: Threshold splitting the histogram into two classes.
    :return: Cross-entropy objective value.
    """
    n_bins, _ = get_nbins_histogram(data)
    histogram, bin_edges = numpy.histogram(data, bins=n_bins, density=True)
    bin_centres = get_bin_centres(bin_edges)

    if numpy.min(bin_centres) < 1:
        offset = abs(numpy.min(bin_centres)) + 1.0
        bin_centres = bin_centres + offset
        threshold = threshold + offset

    split_idx = int(numpy.flatnonzero(bin_centres > threshold)[0])
    m0a = numpy.sum(histogram[:split_idx])
    m0b = numpy.sum(histogram[split_idx:])
    m1a = numpy.sum(histogram[:split_idx] * bin_centres[:split_idx])
    m1b = numpy.sum(histogram[split_idx:] * bin_centres[split_idx:])
    mua = m1a / m0a
    mub = m1b / m0b

    return float(-m1a * numpy.log(mua) - m1b * numpy.log(mub))


def calc_li_threshold(
    data: numpy.ndarray,
    tolerance: Optional[float] = None,
    initial_guess: Optional[float] = None,
) -> float:
    """
    Calculate threshold using Li's iterative Minimum Cross Entropy method.

    :param data: 1D numeric NumPy array.
    :param tolerance: Convergence tolerance (defaults to half the smallest step
                      between unique sorted data values).
    :param initial_guess: Optional initial threshold estimate.
    :return: Computed threshold value.
    """
    finite_data = data[numpy.isfinite(data)]

    offset = 0.0
    if numpy.min(finite_data) < 1:
        offset = float(abs(numpy.min(finite_data)) + 1.0)
        finite_data = finite_data + offset
        if initial_guess is not None:
            initial_guess = initial_guess + offset

    if tolerance is None:
        tolerance = float(numpy.min(numpy.diff(numpy.unique(finite_data))) / 2.0)

    t_next = float(numpy.mean(finite_data)) if initial_guess is None else float(initial_guess)
    t_curr = -2.0 * tolerance

    while abs(t_next - t_curr) > tolerance:
        t_curr = t_next
        foreground = finite_data > t_curr
        mean_fore = numpy.mean(finite_data[foreground])
        mean_back = numpy.mean(finite_data[~foreground])
        t_next = float(
            (mean_back - mean_fore) / (numpy.log(mean_back) - numpy.log(mean_fore))
        )

    return float(t_next - offset)


if HAVE_SCIPY:

    def calc_kurt_skew_threshold(
        data: numpy.ndarray,
        max_val: float,
        min_val: float,
        init_thres: float,
        low_thres: bool = True,
        contamination: float = 10.0,
        only_kurtosis: bool = False,
    ) -> float:
        """
        Optimise a distribution tail threshold by minimising kurtosis (and skewness)
        via dual annealing.

        :param data: 1D numeric NumPy array.
        :param max_val: Upper bound for the threshold search.
        :param min_val: Lower bound for the threshold search.
        :param init_thres: Initial threshold guess.
        :param low_thres: Search lower tail (``data > x``) when True, upper tail
                          (``data < x``) when False.
        :param contamination: Tail contamination percentage in ``[1, 100]``.
        :param only_kurtosis: Optimise kurtosis alone when True; otherwise
                              ``abs(kurtosis) + abs(skew)``.
        :return: Optimised threshold value.
        """
        # pylint: disable=too-many-arguments,too-many-positional-arguments
        if len(data.shape) > 1:
            raise rsgispy.RSGISPyException("Expecting a single variable.")

        if not 1 <= contamination <= 100:
            raise rsgispy.RSGISPyException(
                "contamination parameter should have a value between 1 and 100."
            )

        if low_thres:
            max_val = min(max_val, float(numpy.percentile(data, contamination)))
            if min_val >= max_val:
                min_val = float(numpy.min(data))
        else:
            min_val = max(min_val, float(numpy.percentile(data, 100 - contamination)))
            if max_val <= min_val:
                max_val = float(numpy.max(data))

        if min_val == max_val:
            print(f"Min: {min_val}")
            print(f"Max: {max_val}")
            raise rsgispy.RSGISPyException("Min and Max values are the same.")
        if min_val > max_val:
            print(f"Min: {min_val}")
            print(f"Max: {max_val}")
            raise rsgispy.RSGISPyException(
                "Min value is greater than max - note this can happened if the "
                "contamination parameter caused threshold to be changed."
            )

        if not min_val <= init_thres <= max_val:
            init_thres = min_val + ((max_val - min_val) / 2.0)

        def _opt_fun(x_arr: numpy.ndarray, *args: Any) -> float:
            vals = args[0]
            data_sub = vals[vals > x_arr] if low_thres else vals[vals < x_arr]
            kurtosis = float(scipy.stats.kurtosis(data_sub))
            if only_kurtosis:
                return kurtosis
            skew = float(scipy.stats.skew(data_sub))
            return abs(kurtosis) + abs(skew)

        opt_rslt = scipy.optimize.dual_annealing(
            _opt_fun, bounds=[(min_val, max_val)], args=[data], x0=[init_thres]
        )

        if not opt_rslt.success:
            raise rsgispy.RSGISPyException("Optimisation failed, no threshold found.")
        return float(opt_rslt.x[0])


def mask_data_to_valid(
    data: numpy.ndarray,
    lower_limit: Optional[float] = None,
    upper_limit: Optional[float] = None,
) -> numpy.ndarray:
    """
    Filter rows from an ``(n, m)`` array keeping only finite values within
    ``(lower_limit, upper_limit)``.

    :param data: 2D NumPy array of shape ``(n_rows, n_features)``.
    :param lower_limit: Optional exclusive lower limit.
    :param upper_limit: Optional exclusive upper limit.
    :return: Filtered 2D NumPy array.
    """
    filtered = data[numpy.all(numpy.isfinite(data), axis=1)]
    if lower_limit is not None:
        filtered = filtered[numpy.all(filtered > lower_limit, axis=1)]
    if upper_limit is not None:
        filtered = filtered[numpy.all(filtered < upper_limit, axis=1)]
    return filtered


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


def calc_semivariogram(
    pts_coords: numpy.ndarray,
    data_vals: numpy.ndarray,
    out_data_file: Optional[str] = None,
    out_plot_file: Optional[str] = None,
    max_lag: Union[float, str] = "median",
    n_lags: int = 25,
    normalize_vals: bool = False,
) -> Any:
    """
    Calculate empirical semi-variance for spatial point data using ``skgstat``.

    :param pts_coords: Point XY coordinates of shape ``(n_points, 2)``.
    :param data_vals: Point data values of length ``n_points``.
    :param out_data_file: Optional output CSV path for variogram table.
    :param out_plot_file: Optional output image path for variogram plot.
    :param max_lag: Maximum lag distance or ``'median'`` / ``'mean'``.
    :param n_lags: Number of lag bins (Default = 25).
    :param normalize_vals: Normalise values to ``[0, 1]`` when True.
    :return: ``pandas.DataFrame`` with ``distance``, ``semivariance``, and ``count``.
    """
    # pylint: disable=import-outside-toplevel,too-many-arguments,too-many-positional-arguments,too-many-locals
    import matplotlib.pyplot as plt
    import pandas
    import skgstat  # pylint: disable=import-error

    if pts_coords.shape[1] != 2:
        raise rsgispy.RSGISPyException(
            "Error: pts_coords has wrong shape. Must be [n, 2] "
            "where n is the number of points."
        )

    if pts_coords.shape[0] != data_vals.shape[0]:
        raise rsgispy.RSGISPyException(
            "Error: the length of pts_coords must be the same as data_vals."
        )

    vario_obj = skgstat.Variogram(
        pts_coords,
        data_vals,
        normalize=normalize_vals,
        maxlag=max_lag,
        n_lags=n_lags,
    )

    variogram_data = vario_obj.get_empirical()
    n_count = numpy.fromiter((g.size for g in vario_obj.lag_classes()), dtype=int)

    vario_out_df = pandas.DataFrame(
        {
            "distance": variogram_data[0],
            "semivariance": variogram_data[1],
            "count": n_count,
        }
    )

    if out_data_file is not None:
        vario_out_df.to_csv(out_data_file)

    if out_plot_file is not None:
        fig = plt.figure(figsize=(8, 5))
        ax1 = plt.subplot2grid((5, 1), (1, 0), rowspan=4)
        ax2 = plt.subplot2grid((5, 1), (0, 0), sharex=ax1)
        fig.subplots_adjust(hspace=0)

        ax1.scatter(x=vario_out_df["distance"], y=vario_out_df["semivariance"])
        ax1.set_xlabel("Distance")
        ax1.set_ylabel("Semi-Variance")

        if normalize_vals:
            ax1.set_xlim([0, 1.05])
            ax1.set_ylim([0, 1.05])

        ax1.grid(False)
        ax1.vlines(
            vario_out_df["distance"],
            *ax1.axes.get_ybound(),
            colors=(0.85, 0.85, 0.85),
            linestyles="dashed",
        )

        bar_width = (numpy.max(vario_out_df["distance"]) * 0.7) / len(
            vario_out_df["count"]
        )
        ax2.bar(
            vario_out_df["distance"],
            vario_out_df["count"],
            width=bar_width,
            align="center",
            color="red",
        )

        plt.setp(ax2.axes.get_xticklabels(), visible=False)
        ax2.axes.set_yticks(ax2.axes.get_yticks()[1:])
        ax2.grid(False)
        ax2.vlines(
            vario_out_df["distance"],
            *ax2.axes.get_ybound(),
            colors=(0.85, 0.85, 0.85),
            linestyles="dashed",
        )
        ax2.axes.set_ylabel("N")

        plt.savefig(out_plot_file)
        plt.close(fig)

    return vario_out_df


def standarise_img_data(img_data: numpy.ndarray) -> numpy.ndarray:
    """
    Standardise image data per band: ``(data - mean) / std``.

    :param img_data: Array of shape ``(bands, height, width)`` or
                     ``(bands, n_pxls)``.
    :return: Standardised array of matching shape.
    """
    if len(img_data.shape) == 3:
        img_bands, img_height, img_width = img_data.shape
        n_img_pxls = img_height * img_width
        img_data_reshp = numpy.reshape(img_data, (img_bands, n_img_pxls))
    elif len(img_data.shape) == 2:
        img_bands, n_img_pxls = img_data.shape
        img_data_reshp = img_data
    else:
        raise rsgispy.RSGISPyException(
            "Input array must have either 2 or 3 dimensions."
        )

    img_band_mean = numpy.mean(img_data_reshp, axis=1, keepdims=True)
    img_band_cen = img_data_reshp - img_band_mean
    img_band_var = (
        numpy.sum(numpy.power(img_band_cen, 2), axis=1, keepdims=True) / n_img_pxls
    )
    img_band_std = numpy.sqrt(img_band_var)
    std_img_data = img_band_cen / img_band_std
    if len(img_data.shape) == 3:
        std_img_data = numpy.reshape(std_img_data, (img_bands, img_height, img_width))

    return std_img_data


def normalise_img_data(img_data: numpy.ndarray) -> numpy.ndarray:
    """
    Normalise image data per band: ``(data - min) / (max - min)``.

    :param img_data: Array of shape ``(bands, height, width)`` or
                     ``(bands, n_pxls)``.
    :return: Normalised array of matching shape.
    """
    if len(img_data.shape) == 3:
        img_bands, img_height, img_width = img_data.shape
        img_data_reshp = numpy.reshape(img_data, (img_bands, img_height * img_width))
    elif len(img_data.shape) == 2:
        img_bands, _ = img_data.shape
        img_data_reshp = img_data
    else:
        raise rsgispy.RSGISPyException(
            "Input array must have either 2 or 3 dimensions."
        )

    max_value = numpy.max(img_data_reshp, axis=1, keepdims=True)
    min_value = numpy.min(img_data_reshp, axis=1, keepdims=True)
    diff_value = max_value - min_value
    nm_img = (img_data_reshp - min_value) / diff_value
    if len(img_data.shape) == 3:
        nm_img = numpy.reshape(nm_img, (img_bands, img_height, img_width))

    return nm_img
