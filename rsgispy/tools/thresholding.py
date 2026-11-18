"""
The tools.thresholding module provides histogram- and distribution-based
automatic thresholding utilities.
"""

from typing import Any, Optional, Tuple

import numpy as np

import rsgispy

try:
    import scipy.optimize  # pylint: disable=import-error
    import scipy.stats  # pylint: disable=import-error

    HAVE_SCIPY = True
except ImportError:
    scipy = None  # type: ignore[assignment]
    HAVE_SCIPY = False

__all__ = [
    "get_nbins_histogram",
    "get_bin_centres",
    "calc_otsu_threshold",
    "calc_yen_threshold",
    "calc_isodata_threshold",
    "calc_hist_cross_entropy",
    "calc_li_threshold",
    "calc_kurt_skew_threshold",
]


def get_nbins_histogram(data: np.ndarray) -> Tuple[int, float]:
    """
    Calculate the Freedman-Diaconis number of histogram bins and bin width.

    :param data: 1D numeric NumPy array.
    :return: ``(n_bins, bin_width)`` tuple.

    .. code:: python

        import numpy as np
        import rsgispy.tools.thresholding
        data = np.random.normal(0.0, 1.0, 500)
        n_bins, bin_width = rsgispy.tools.thresholding.get_nbins_histogram(data)
    """
    n_pts = data.shape[0]
    lower_q = np.percentile(data, 25)
    upper_q = np.percentile(data, 75)
    iqr = float(upper_q - lower_q)
    data_range = float(np.max(data) - np.min(data))
    if data_range <= 0.0:
        return 2, 1.0
    if iqr <= 0.0:
        n_bins = max(2, min(256, int(np.ceil(np.sqrt(max(n_pts, 1))))))
        return n_bins, data_range / float(n_bins)
    bin_width = 2.0 * iqr * n_pts ** (-1.0 / 3.0)
    n_bins = max(2, int(data_range / bin_width) + 2)
    return n_bins, float(bin_width)


def get_bin_centres(
    bin_edges: np.ndarray,
    geometric: bool = False,
) -> np.ndarray:
    """
    Calculate histogram bin centres from bin edges.

    :param bin_edges: 1D NumPy array of histogram bin edges.
    :param geometric: Compute geometric mean ``sqrt(min * max)`` when True, or
                      arithmetic mean ``(min + max) / 2`` otherwise.
    :return: 1D NumPy array of bin centres.

    .. code:: python

        import numpy as np
        import rsgispy.tools.thresholding
        edges = np.array([0.0, 1.0, 2.0, 3.0])
        centres = rsgispy.tools.thresholding.get_bin_centres(edges)
    """
    if geometric:
        return np.sqrt(bin_edges[1:] * bin_edges[:-1])
    return (bin_edges[1:] + bin_edges[:-1]) / 2.0


def calc_otsu_threshold(data: np.ndarray) -> float:
    """
    Calculate Otsu's threshold for a 1D numeric NumPy array.

    :param data: 1D numeric NumPy array.
    :return: Computed Otsu threshold value.

    .. code:: python

        import numpy as np
        import rsgispy.tools.thresholding
        data = np.concatenate([np.random.normal(2, 0.5, 200), np.random.normal(6, 0.5, 200)])
        thres = rsgispy.tools.thresholding.calc_otsu_threshold(data)
    """
    n_bins, _ = get_nbins_histogram(data)
    hist, bin_edges = np.histogram(data, bins=n_bins)
    bin_centres = get_bin_centres(bin_edges)

    hist_prob = hist.astype(np.float64)
    hist_prob = hist_prob / np.maximum(np.sum(hist_prob), 1e-12)

    weight1 = np.cumsum(hist_prob)
    weight2 = np.cumsum(hist_prob[::-1])[::-1]
    mean1 = np.cumsum(hist_prob * bin_centres) / np.maximum(weight1, 1e-12)
    mean2 = (
        np.cumsum((hist_prob * bin_centres)[::-1])
        / np.maximum(weight2[::-1], 1e-12)
    )[::-1]

    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    return float(bin_centres[int(np.argmax(variance12))])


def calc_yen_threshold(data: np.ndarray) -> float:
    """
    Calculate Yen's multilevel threshold for a 1D numeric NumPy array.

    :param data: 1D numeric NumPy array.
    :return: Computed Yen threshold value.

    .. code:: python

        import numpy as np
        import rsgispy.tools.thresholding
        data = np.concatenate([np.random.normal(2, 0.5, 200), np.random.normal(6, 0.5, 200)])
        thres = rsgispy.tools.thresholding.calc_yen_threshold(data)
    """
    n_bins, _ = get_nbins_histogram(data)
    hist, bin_edges = np.histogram(data, bins=n_bins)
    bin_centres = get_bin_centres(bin_edges)

    pmf = hist.astype(np.float64) / np.maximum(np.sum(hist), 1e-12)
    p1 = np.cumsum(pmf)
    p1_sq = np.cumsum(pmf**2)
    p2_sq = np.cumsum(pmf[::-1] ** 2)[::-1]

    denom = np.maximum(p1_sq[:-1] * p2_sq[1:], 1e-12)
    numer = np.maximum((p1[:-1] * (1.0 - p1[:-1])) ** 2, 1e-12)
    crit = np.log(numer / denom)
    return float(bin_centres[int(crit.argmax())])


def calc_isodata_threshold(data: np.ndarray) -> float:
    """
    Calculate the ISODATA (Ridler-Calvard) inter-means threshold for a 1D
    numeric NumPy array.

    :param data: 1D numeric NumPy array.
    :return: Computed ISODATA threshold value.

    .. code:: python

        import numpy as np
        import rsgispy.tools.thresholding
        data = np.concatenate([np.random.normal(2, 0.5, 200), np.random.normal(6, 0.5, 200)])
        thres = rsgispy.tools.thresholding.calc_isodata_threshold(data)
    """
    n_bins, _ = get_nbins_histogram(data)
    hist, bin_edges = np.histogram(data, bins=n_bins)
    bin_centres = get_bin_centres(bin_edges)

    hist_prob = hist.astype(np.float64)
    hist_prob = hist_prob / np.maximum(np.sum(hist_prob), 1e-12)

    csuml = np.cumsum(hist_prob)
    csum_intensity = np.cumsum(hist_prob * bin_centres)
    lower = csum_intensity[:-1] / np.maximum(csuml[:-1], 1e-12)
    higher = (csum_intensity[-1] - csum_intensity[:-1]) / np.maximum(
        (csuml[-1] - csuml)[:-1], 1e-12
    )

    distances = ((lower + higher) / 2.0) - bin_centres[:-1]
    bin_width = bin_centres[1] - bin_centres[0]
    thresholds = bin_centres[:-1][(distances >= 0) & (distances < bin_width)]
    if thresholds.size == 0:
        return float(bin_centres[:-1][int(np.argmin(np.abs(distances)))])
    return float(thresholds[0])


def calc_hist_cross_entropy(data: np.ndarray, threshold: float) -> float:
    """
    Compute cross-entropy between histogram distributions above and below
    ``threshold``.

    :param data: 1D numeric NumPy array.
    :param threshold: Threshold splitting the histogram into two classes.
    :return: Cross-entropy objective value.

    .. code:: python

        import numpy as np
        import rsgispy.tools.thresholding
        data = np.concatenate([np.random.normal(2, 0.5, 200), np.random.normal(6, 0.5, 200)])
        ce = rsgispy.tools.thresholding.calc_hist_cross_entropy(data, threshold=4.0)
    """
    n_bins, _ = get_nbins_histogram(data)
    histogram, bin_edges = np.histogram(data, bins=n_bins, density=True)
    bin_centres = get_bin_centres(bin_edges)

    if np.min(bin_centres) < 1:
        offset = abs(np.min(bin_centres)) + 1.0
        bin_centres = bin_centres + offset
        threshold = threshold + offset

    split_idx = int(np.flatnonzero(bin_centres > threshold)[0])
    m0a = np.sum(histogram[:split_idx])
    m0b = np.sum(histogram[split_idx:])
    m1a = np.sum(histogram[:split_idx] * bin_centres[:split_idx])
    m1b = np.sum(histogram[split_idx:] * bin_centres[split_idx:])
    mua = m1a / m0a
    mub = m1b / m0b

    return float(-m1a * np.log(mua) - m1b * np.log(mub))


def calc_li_threshold(
    data: np.ndarray,
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

    .. code:: python

        import numpy as np
        import rsgispy.tools.thresholding
        data = np.concatenate([np.random.normal(2, 0.5, 200), np.random.normal(6, 0.5, 200)])
        thres = rsgispy.tools.thresholding.calc_li_threshold(data)
    """
    finite_data = data[np.isfinite(data)]

    offset = 0.0
    if np.min(finite_data) < 1:
        offset = float(abs(np.min(finite_data)) + 1.0)
        finite_data = finite_data + offset
        if initial_guess is not None:
            initial_guess = initial_guess + offset

    if tolerance is None:
        tolerance = float(np.min(np.diff(np.unique(finite_data))) / 2.0)

    t_next = (
        float(np.mean(finite_data))
        if initial_guess is None
        else float(initial_guess)
    )
    t_curr = -2.0 * tolerance

    while abs(t_next - t_curr) > tolerance:
        t_curr = t_next
        foreground = finite_data > t_curr
        mean_fore = np.mean(finite_data[foreground])
        mean_back = np.mean(finite_data[~foreground])
        t_next = float(
            (mean_back - mean_fore) / (np.log(mean_back) - np.log(mean_fore))
        )

    return float(t_next - offset)


def calc_kurt_skew_threshold(
    data: np.ndarray,
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

    .. code:: python

        import numpy as np
        import rsgispy.tools.thresholding
        data = np.random.normal(5.0, 1.0, 500)
        thres = rsgispy.tools.thresholding.calc_kurt_skew_threshold(
            data, max_val=4.5, min_val=2.0, init_thres=3.0
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    rsgispy.check_optional_dep(
        HAVE_SCIPY, "calc_kurt_skew_threshold", "scipy", "stats"
    )
    if len(data.shape) > 1:
        raise rsgispy.RSGISPyException("Expecting a single variable.")

    if not 1 <= contamination <= 100:
        raise rsgispy.RSGISPyException(
            "contamination parameter should have a value between 1 and 100."
        )

    if low_thres:
        max_val = min(max_val, float(np.percentile(data, contamination)))
        if min_val >= max_val:
            min_val = float(np.min(data))
    else:
        min_val = max(min_val, float(np.percentile(data, 100 - contamination)))
        if max_val <= min_val:
            max_val = float(np.max(data))

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

    def _opt_fun(x_arr: np.ndarray, *args: Any) -> float:
        vals = args[0]
        data_sub = vals[vals > x_arr] if low_thres else vals[vals < x_arr]
        kurtosis = float(scipy.stats.kurtosis(data_sub))
        if only_kurtosis:
            return abs(kurtosis)
        skew = float(scipy.stats.skew(data_sub))
        return abs(kurtosis) + abs(skew)

    opt_rslt = scipy.optimize.dual_annealing(
        _opt_fun, bounds=[(min_val, max_val)], args=[data], x0=[init_thres]
    )

    if not opt_rslt.success:
        raise rsgispy.RSGISPyException("Optimisation failed, no threshold found.")
    return float(opt_rslt.x[0])
