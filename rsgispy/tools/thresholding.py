"""
The tools.thresholding module provides histogram- and distribution-based
automatic thresholding utilities.

Functions requiring ``scipy`` are only defined and exported when that optional
dependency is installed.
"""

from typing import Any, Optional, Tuple

import numpy

import rsgispy

try:
    import scipy.optimize  # pylint: disable=import-error
    import scipy.stats  # pylint: disable=import-error

    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

__all__ = [
    "get_nbins_histogram",
    "get_bin_centres",
    "calc_otsu_threshold",
    "calc_yen_threshold",
    "calc_isodata_threshold",
    "calc_hist_cross_entropy",
    "calc_li_threshold",
]

if HAVE_SCIPY:
    __all__.append("calc_kurt_skew_threshold")


def get_nbins_histogram(data: numpy.ndarray) -> Tuple[int, float]:
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

    .. code:: python

        import numpy as np
        import rsgispy.tools.thresholding
        edges = np.array([0.0, 1.0, 2.0, 3.0])
        centres = rsgispy.tools.thresholding.get_bin_centres(edges)
    """
    if geometric:
        return numpy.sqrt(bin_edges[1:] * bin_edges[:-1])
    return (bin_edges[1:] + bin_edges[:-1]) / 2.0


def calc_otsu_threshold(data: numpy.ndarray) -> float:
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

    .. code:: python

        import numpy as np
        import rsgispy.tools.thresholding
        data = np.concatenate([np.random.normal(2, 0.5, 200), np.random.normal(6, 0.5, 200)])
        thres = rsgispy.tools.thresholding.calc_yen_threshold(data)
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

    .. code:: python

        import numpy as np
        import rsgispy.tools.thresholding
        data = np.concatenate([np.random.normal(2, 0.5, 200), np.random.normal(6, 0.5, 200)])
        thres = rsgispy.tools.thresholding.calc_isodata_threshold(data)
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

    .. code:: python

        import numpy as np
        import rsgispy.tools.thresholding
        data = np.concatenate([np.random.normal(2, 0.5, 200), np.random.normal(6, 0.5, 200)])
        ce = rsgispy.tools.thresholding.calc_hist_cross_entropy(data, threshold=4.0)
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

    .. code:: python

        import numpy as np
        import rsgispy.tools.thresholding
        data = np.concatenate([np.random.normal(2, 0.5, 200), np.random.normal(6, 0.5, 200)])
        thres = rsgispy.tools.thresholding.calc_li_threshold(data)
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

    t_next = (
        float(numpy.mean(finite_data))
        if initial_guess is None
        else float(initial_guess)
    )
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

        .. code:: python

            import numpy as np
            import rsgispy.tools.thresholding
            data = np.random.normal(5.0, 1.0, 500)
            thres = rsgispy.tools.thresholding.calc_kurt_skew_threshold(
                data, max_val=4.5, min_val=2.0, init_thres=3.0
            )
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
