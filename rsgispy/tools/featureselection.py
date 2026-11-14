"""
The tools.featureselection module provides predictor filtering, collinearity
diagnostics, and regression diagnostic utilities.

Functions requiring ``scipy`` and/or ``scikit-learn`` are only defined and
exported when those optional dependencies are installed.
"""

from typing import Any, List, Optional, Union
import warnings

import numpy

import rsgispy

try:
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
]

if HAVE_SKLEARN:
    __all__.extend(
        [
            "calc_pandas_vif",
            "lassolars_feature_selection",
        ]
    )

if HAVE_SCIPY and HAVE_SKLEARN:
    __all__.extend(
        [
            "corr_feature_selection",
            "breusch_pagan_test",
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

        .. code:: python

            import pandas as pd
            import rsgispy.tools.featureselection
            df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [2, 3, 5, 7], "c": [4, 5, 8, 9]})
            vif = rsgispy.tools.featureselection.calc_pandas_vif(df)
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

    .. code:: python

        import pandas as pd
        import rsgispy.tools.featureselection
        df = pd.DataFrame({"a": [1, 2, 3, 10], "b": [5, 5, 6, 6]})
        cols = rsgispy.tools.featureselection.cqv_threshold(df)
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

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def corr_feature_selection(
        df: Any,
        dep_vars: List[str],
        ind_vars: List[str],
        n_min_clusters: int = 3,
        n_max_clusters: int = 25,
        use_pbar: bool = True,
    ) -> List[str]:
        """
        Perform correlation-based feature selection by clustering independent
        variables using Pearson correlation distance and Silhouette selection.

        :param df: Pandas DataFrame containing dependent and independent variables.
        :param dep_vars: List of dependent variable column names.
        :param ind_vars: List of independent (predictor) variable column names.
        :param n_min_clusters: Minimum cluster count in the search (Default = 3).
        :param n_max_clusters: Exclusive upper cluster count bound (Default = 25).
        :param use_pbar: If ``True``, display a tqdm progress bar during processing
                         (Default = ``True``).
        :return: List of selected predictor column names.

        .. code:: python

            import pandas as pd
            import rsgispy.tools.featureselection
            df = pd.DataFrame({
                "y": [1, 2, 3, 4, 5],
                "x1": [1.0, 2.1, 2.9, 4.0, 5.1],
                "x2": [5.0, 4.1, 3.0, 2.1, 1.0],
                "x3": [2.0, 3.0, 4.0, 5.0, 6.0],
                "x4": [0.1, 0.4, 0.2, 0.8, 0.5],
            })
            selected = rsgispy.tools.featureselection.corr_feature_selection(
                df, ["y"], ["x1", "x2", "x3", "x4"], n_min_clusters=2, n_max_clusters=4
            )
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
        for n_clst in tqdm_cls(n_clusters, disable=not use_pbar):
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

        .. code:: python

            import pandas as pd
            import rsgispy.tools.featureselection
            df = pd.DataFrame({
                "y": [1, 2, 3, 4, 5],
                "x1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "x2": [0.1, 0.2, 0.1, 0.3, 0.2],
            })
            cols = rsgispy.tools.featureselection.lassolars_feature_selection(
                df, ["y"], ["x1", "x2"]
            )
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
            f"Using regularisation parameter (alpha) for the Lasso estimator of: "
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

        .. code:: python

            import numpy as np
            import rsgispy.tools.featureselection
            x = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 5.0], [4.0, 7.0]])
            y = np.array([2.1, 3.9, 6.2, 8.1])
            lm_stat, pval, msg = rsgispy.tools.featureselection.breusch_pagan_test(x, y)
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
