"""
The tools.stats module provides array statistical filtering, spatial
variogram calculation, percentile computation, and data normalisation utilities.
"""

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

import rsgispy

__all__ = [
    "mask_data_to_valid",
    "calc_semivariogram",
    "standardise_img_data",
    "normalise_img_data",
    "validate_percentiles_0_100",
    "compute_nanpercentiles",
    "resolve_metric_type",
    "calc_similarity_metric",
    "phase_cross_correlation_2d",
]

_METRIC_INFO: Dict[int, Tuple[str, bool]] = {
    rsgispy.METRIC_EUCLIDEAN: ("euclidean", False),
    rsgispy.METRIC_SQDIFF: ("sqdiff", False),
    rsgispy.METRIC_MANHATTAN: ("manhattan", False),
    rsgispy.METRIC_CORRELATION: ("correlation", True),
    rsgispy.METRIC_MUTUAL_INFO: ("mutual_info", True),
    rsgispy.METRIC_NORM_MUTUAL_INFO: ("norm_mutual_info", True),
    rsgispy.METRIC_PHASE_CORR: ("phase_correlation", True),
}

_METRIC_ALIASES: Dict[str, int] = {
    "euclidean": rsgispy.METRIC_EUCLIDEAN,
    "l2": rsgispy.METRIC_EUCLIDEAN,
    "rmse": rsgispy.METRIC_EUCLIDEAN,
    "sqdiff": rsgispy.METRIC_SQDIFF,
    "mse": rsgispy.METRIC_SQDIFF,
    "manhattan": rsgispy.METRIC_MANHATTAN,
    "l1": rsgispy.METRIC_MANHATTAN,
    "correlation": rsgispy.METRIC_CORRELATION,
    "corr": rsgispy.METRIC_CORRELATION,
    "pearson": rsgispy.METRIC_CORRELATION,
    "ncc": rsgispy.METRIC_CORRELATION,
    "mutual_info": rsgispy.METRIC_MUTUAL_INFO,
    "mutual_information": rsgispy.METRIC_MUTUAL_INFO,
    "mi": rsgispy.METRIC_MUTUAL_INFO,
    "norm_mutual_info": rsgispy.METRIC_NORM_MUTUAL_INFO,
    "nmi": rsgispy.METRIC_NORM_MUTUAL_INFO,
    "phase_correlation": rsgispy.METRIC_PHASE_CORR,
    "phase_corr": rsgispy.METRIC_PHASE_CORR,
    "phase": rsgispy.METRIC_PHASE_CORR,
}


def mask_data_to_valid(
    data: np.ndarray,
    lower_limit: Optional[float] = None,
    upper_limit: Optional[float] = None,
) -> np.ndarray:
    """
    Filter rows from an ``(n, m)`` array keeping only finite values within
    ``(lower_limit, upper_limit)``.

    :param data: 2D NumPy array of shape ``(n_rows, n_features)``.
    :param lower_limit: Optional exclusive lower limit.
    :param upper_limit: Optional exclusive upper limit.
    :return: Filtered 2D NumPy array.

    .. code:: python

        import numpy as np
        import rsgispy.tools.stats
        arr = np.array([[1.0, 2.0], [np.nan, 3.0], [0.0, 5.0]])
        valid = rsgispy.tools.stats.mask_data_to_valid(arr, lower_limit=0.0)
    """
    filtered = data[np.all(np.isfinite(data), axis=1)]
    if lower_limit is not None:
        filtered = filtered[np.all(filtered > lower_limit, axis=1)]
    if upper_limit is not None:
        filtered = filtered[np.all(filtered < upper_limit, axis=1)]
    return filtered


def calc_semivariogram(
    pts_coords: np.ndarray,
    data_vals: np.ndarray,
    out_data_file: Optional[str] = None,
    out_plot_file: Optional[str] = None,
    max_lag: Union[float, str] = "median",
    n_lags: int = 25,
    normalise_vals: bool = False,
) -> Any:
    """
    Calculate empirical semi-variance for spatial point data using ``skgstat``.

    :param pts_coords: Point XY coordinates of shape ``(n_points, 2)``.
    :param data_vals: Point data values of length ``n_points``.
    :param out_data_file: Optional output CSV path for variogram table.
    :param out_plot_file: Optional output image path for variogram plot.
    :param max_lag: Maximum lag distance or ``'median'`` / ``'mean'``.
    :param n_lags: Number of lag bins (Default = 25).
    :param normalise_vals: Normalise values to ``[0, 1]`` when True.
    :return: ``pandas.DataFrame`` with ``distance``, ``semivariance``, and ``count``.

    .. code:: python

        import numpy as np
        import rsgispy.tools.stats
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 2.0]])
        vals = np.array([1.2, 1.5, 1.4, 2.8])
        df = rsgispy.tools.stats.calc_semivariogram(coords, vals, n_lags=4)
    """
    # pylint: disable=import-outside-toplevel,too-many-arguments
    # pylint: disable=too-many-positional-arguments,too-many-locals
    import matplotlib.pyplot as plt
    import pandas as pd

    try:
        import skgstat  # pylint: disable=import-error
    except ImportError as exc:
        rsgispy.check_optional_dep(
            False,
            "calc_semivariogram",
            "scikit-gstat",
            "stats",
            error_detail=str(exc),
        )
        raise

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
        normalize=normalise_vals,
        maxlag=max_lag,
        n_lags=n_lags,
    )

    variogram_data = vario_obj.get_empirical()
    n_count = np.fromiter((g.size for g in vario_obj.lag_classes()), dtype=int)

    vario_out_df = pd.DataFrame(
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

        if normalise_vals:
            ax1.set_xlim([0, 1.05])
            ax1.set_ylim([0, 1.05])

        ax1.grid(False)
        ax1.vlines(
            vario_out_df["distance"],
            *ax1.axes.get_ybound(),
            colors=(0.85, 0.85, 0.85),
            linestyles="dashed",
        )

        bar_width = (np.max(vario_out_df["distance"]) * 0.7) / len(
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


def standardise_img_data(img_data: np.ndarray) -> np.ndarray:
    """
    Standardise image data per band: ``(data - mean) / std``.

    :param img_data: Array of shape ``(bands, height, width)`` or
                     ``(bands, n_pxls)``.
    :return: Standardised array of matching shape.

    .. code:: python

        import numpy as np
        import rsgispy.tools.stats
        img = np.random.rand(3, 10, 10)
        std_img = rsgispy.tools.stats.standardise_img_data(img)
    """
    if len(img_data.shape) == 3:
        img_bands, img_height, img_width = img_data.shape
        n_img_pxls = img_height * img_width
        img_data_reshp = np.reshape(img_data, (img_bands, n_img_pxls))
    elif len(img_data.shape) == 2:
        img_bands, n_img_pxls = img_data.shape
        img_data_reshp = img_data
    else:
        raise rsgispy.RSGISPyException(
            "Input array must have either 2 or 3 dimensions."
        )

    img_band_mean = np.mean(img_data_reshp, axis=1, keepdims=True)
    img_band_cen = img_data_reshp - img_band_mean
    img_band_var = (
        np.sum(np.power(img_band_cen, 2), axis=1, keepdims=True) / n_img_pxls
    )
    img_band_std = np.sqrt(img_band_var)
    std_img_data = np.divide(
        img_band_cen,
        img_band_std,
        out=np.zeros_like(img_band_cen, dtype=np.float64),
        where=img_band_std > 0,
    )
    if len(img_data.shape) == 3:
        std_img_data = np.reshape(std_img_data, (img_bands, img_height, img_width))

    return std_img_data


def normalise_img_data(img_data: np.ndarray) -> np.ndarray:
    """
    Normalise image data per band: ``(data - min) / (max - min)``.

    :param img_data: Array of shape ``(bands, height, width)`` or
                     ``(bands, n_pxls)``.
    :return: Normalised array of matching shape.

    .. code:: python

        import numpy as np
        import rsgispy.tools.stats
        img = np.random.rand(3, 10, 10)
        norm_img = rsgispy.tools.stats.normalise_img_data(img)
    """
    if len(img_data.shape) == 3:
        img_bands, img_height, img_width = img_data.shape
        img_data_reshp = np.reshape(img_data, (img_bands, img_height * img_width))
    elif len(img_data.shape) == 2:
        img_bands, _ = img_data.shape
        img_data_reshp = img_data
    else:
        raise rsgispy.RSGISPyException(
            "Input array must have either 2 or 3 dimensions."
        )

    max_value = np.max(img_data_reshp, axis=1, keepdims=True)
    min_value = np.min(img_data_reshp, axis=1, keepdims=True)
    diff_value = max_value - min_value
    shifted = img_data_reshp - min_value
    nm_img = np.divide(
        shifted,
        diff_value,
        out=np.zeros_like(shifted, dtype=np.float64),
        where=diff_value > 0,
    )
    if len(img_data.shape) == 3:
        nm_img = np.reshape(nm_img, (img_bands, img_height, img_width))

    return nm_img


def validate_percentiles_0_100(
    percentiles: Union[float, int, Sequence[float]],
) -> None:
    """
    Validate that percentile value(s) are non-empty and lie in ``[0, 100]``.

    :param percentiles: A single percentile value or a non-empty sequence of
                        percentile values expected in ``[0, 100]``.

    .. code:: python

        import rsgispy.tools.stats
        rsgispy.tools.stats.validate_percentiles_0_100([25, 50, 75])
    """
    if percentiles is None or isinstance(percentiles, (bool, str, bytes)):
        raise rsgispy.RSGISPyException(
            "percentiles must be a number or a non-empty sequence of numbers."
        )
    if isinstance(percentiles, (int, float, np.number)):
        vals: List[float] = [float(percentiles)]
    elif isinstance(percentiles, np.ndarray) and percentiles.ndim == 0:
        vals = [float(percentiles.item())]
    else:
        vals = list(percentiles)
        if not vals:
            raise rsgispy.RSGISPyException(
                "percentiles must contain at least one value."
            )

    for percent in vals:
        if isinstance(percent, bool) or not 0.0 <= float(percent) <= 100.0:
            raise rsgispy.RSGISPyException(
                f"Percentile must be between 0 and 100 ({percent})."
            )


def compute_nanpercentiles(
    stack_arr: np.ndarray,
    percentiles: Union[float, Sequence[float]],
    no_data_val: Optional[float] = None,
) -> np.ndarray:
    """
    Compute per-pixel percentiles along axis 0 while ignoring ``NaN`` and
    ``no_data_val``.

    The result equals ``np.nanpercentile(stack_arr, percentiles, axis=0)``
    (NumPy's default ``method="linear"``) to floating-point precision (the
    interpolation is done in float64; some NumPy versions interpolate float32
    input in float32, so results can differ in the last float32 digit), but is
    vectorised: the stack is sorted once along axis 0 (``NaN`` values sort to
    the end), the number of valid values ``n`` is counted for each pixel, and
    each percentile ``p`` is interpolated linearly between the sorted values
    either side of position ``p / 100 * (n - 1)``. This is several hundred
    times faster than ``np.nanpercentile`` with ``axis=0``, which loops over
    the pixels in Python. Pixels with no valid values give ``NaN`` (or
    ``no_data_val``).

    :param stack_arr: Input array stacked along axis 0 (e.g. ``(n_layers, height, width)``).
                      The array is not modified.
    :param percentiles: Percentile value or sequence of percentiles in ``[0, 100]``.
    :param no_data_val: Optional no-data value to ignore when computing the
                        percentiles (compared in the precision of
                        ``stack_arr``); it is also written to the pixels where
                        all values along axis 0 are no data or ``NaN``
                        (Default = None, which ignores only ``NaN`` and leaves
                        ``NaN`` in those pixels).
    :return: ``float64`` NumPy array of percentiles with shape
             ``(len(percentiles),) + stack_arr.shape[1:]``. For a scalar
             ``percentiles`` the leading axis is dropped, except that 3D
             inputs keep it (output shape ``(1, height, width)``).

    .. code:: python

        import numpy as np
        import rsgispy.tools.stats
        stack = np.random.rand(5, 10, 10)
        pct = rsgispy.tools.stats.compute_nanpercentiles(stack, [25, 50, 75])
    """
    # pylint: disable=too-many-locals
    arr = np.asarray(stack_arr)
    if arr.ndim == 0:
        raise rsgispy.RSGISPyException(
            "stack_arr must have at least one dimension."
        )
    if np.issubdtype(arr.dtype, np.floating):
        work = arr.copy()
    else:
        work = arr.astype(np.float64)

    if no_data_val is not None and not math.isnan(float(no_data_val)):
        with np.errstate(over="ignore"):
            nd_val = work.dtype.type(no_data_val)
        work[work == nd_val] = np.nan

    q_arr = np.asarray(percentiles, dtype=np.float64)
    q_flat = q_arr.reshape(-1) / 100.0
    pct_arr = np.full((q_flat.size,) + work.shape[1:], np.nan, dtype=np.float64)

    if work.shape[0] > 0:
        # NaN values sort to the end, so the valid values of each pixel are
        # the first n_valid elements along axis 0.
        work.sort(axis=0)
        n_valid = work.shape[0] - np.count_nonzero(np.isnan(work), axis=0)
        n_f = np.asarray(n_valid, dtype=np.float64)[np.newaxis, ...]
        last_idx = np.maximum(n_f - 1.0, 0.0).astype(np.intp)
        with np.errstate(invalid="ignore"):
            for q_idx, q_val in enumerate(q_flat):
                # Same virtual index as NumPy's 'linear' method: (n - 1) * q.
                v_idx = n_f * q_val + (1.0 - q_val) - 1.0
                lo_pos = np.floor(v_idx)
                gamma = v_idx - lo_pos
                lo_idx = np.minimum(np.maximum(lo_pos, 0.0).astype(np.intp), last_idx)
                hi_idx = np.minimum(lo_idx + 1, last_idx)
                lo_val = np.take_along_axis(work, lo_idx, axis=0)
                hi_val = np.take_along_axis(work, hi_idx, axis=0)
                # Linear interpolation, evaluated from the nearer end as
                # NumPy does (pixels with n == 0 give NaN).
                diff = hi_val - lo_val
                res = lo_val + diff * gamma
                np.subtract(hi_val, diff * (1.0 - gamma), out=res, where=gamma >= 0.5)
                pct_arr[q_idx] = res[0]

    if q_arr.ndim == 0:
        pct_arr = pct_arr[0, ...]
        if work.ndim == 3:
            pct_arr = np.expand_dims(pct_arr, axis=0)

    if no_data_val is not None and not math.isnan(float(no_data_val)):
        pct_arr[np.isnan(pct_arr)] = no_data_val
    return pct_arr


def resolve_metric_type(metric_type: Union[int, str]) -> Tuple[int, str, bool]:
    """
    Normalise ``metric_type`` (``rsgispy.METRIC_*`` integer or string alias)
    into ``(metric_id, metric_name, maximise)``.

    :param metric_type: Similarity metric constant ``rsgispy.METRIC_*`` or
                        name string (e.g., ``'correlation'``, ``'euclidean'``,
                        ``'sqdiff'``, ``'manhattan'``, ``'mutual_info'``,
                        ``'norm_mutual_info'``, ``'phase_correlation'``).
    :return: Tuple ``(metric_id, canonical_name, maximise)`` where ``maximise``
             is ``True`` when larger values indicate better agreement.

    .. code:: python

        import rsgispy
        import rsgispy.tools.stats

        m_id, name, maximise = rsgispy.tools.stats.resolve_metric_type(
            rsgispy.METRIC_CORRELATION
        )
    """
    if isinstance(metric_type, bool):
        raise rsgispy.RSGISPyException(
            "metric_type must be a METRIC_* integer or string, not bool."
        )
    if isinstance(metric_type, (int, np.integer)):
        m_id = int(metric_type)
        if m_id not in _METRIC_INFO:
            raise rsgispy.RSGISPyException(
                f"Unsupported metric_type id: {metric_type!r}"
            )
        name, maximise = _METRIC_INFO[m_id]
        return m_id, name, maximise

    if isinstance(metric_type, str):
        key = metric_type.strip().lower().replace("-", "_").replace(" ", "_")
        if key not in _METRIC_ALIASES:
            raise rsgispy.RSGISPyException(
                f"Unsupported metric_type '{metric_type}'. Supported: "
                f"{sorted(set(v[0] for v in _METRIC_INFO.values()))}."
            )
        m_id = _METRIC_ALIASES[key]
        name, maximise = _METRIC_INFO[m_id]
        return m_id, name, maximise

    raise rsgispy.RSGISPyException(
        f"Unsupported metric_type: {metric_type!r}"
    )


def _mutual_information(
    a_vals: np.ndarray,
    b_vals: np.ndarray,
    hist_bins: int = 64,
    normalised: bool = False,
) -> float:
    """Compute pure-NumPy Mutual Information or Normalised Mutual Information."""
    if a_vals.size < 2:
        return 0.0
    if np.ptp(a_vals) == 0.0 or np.ptp(b_vals) == 0.0:
        return 0.0
    c_xy, _, _ = np.histogram2d(a_vals, b_vals, bins=hist_bins)
    total = float(c_xy.sum())
    if total <= 0.0:
        return 0.0
    p_xy = c_xy / total
    p_x = p_xy.sum(axis=1)
    p_y = p_xy.sum(axis=0)

    nz_x = p_x > 0
    nz_y = p_y > 0
    nz_xy = p_xy > 0
    h_x = -float(np.sum(p_x[nz_x] * np.log(p_x[nz_x])))
    h_y = -float(np.sum(p_y[nz_y] * np.log(p_y[nz_y])))
    h_xy = -float(np.sum(p_xy[nz_xy] * np.log(p_xy[nz_xy])))
    mi = max(0.0, h_x + h_y - h_xy)
    if not normalised:
        return mi
    denom = h_x + h_y
    if denom <= 1e-12:
        return 0.0
    return float(2.0 * mi / denom)


def calc_similarity_metric(
    ref_vals: np.ndarray,
    flt_vals: np.ndarray,
    metric_id: Union[int, str] = rsgispy.METRIC_CORRELATION,
    hist_bins: int = 64,
) -> float:
    """
    Compute a pairwise similarity or distance metric between two matched arrays.

    Supports ``rsgispy.METRIC_CORRELATION`` (Pearson correlation),
    ``rsgispy.METRIC_EUCLIDEAN`` (RMSE), ``rsgispy.METRIC_SQDIFF`` (MSE),
    ``rsgispy.METRIC_MANHATTAN`` (MAE), ``rsgispy.METRIC_MUTUAL_INFO``
    (mutual information), ``rsgispy.METRIC_NORM_MUTUAL_INFO`` (normalised
    mutual information), and ``rsgispy.METRIC_PHASE_CORR`` (2D arrays only).

    :param ref_vals: Reference value array.
    :param flt_vals: Comparison value array of identical shape to ``ref_vals``.
    :param metric_id: Metric constant ``rsgispy.METRIC_*`` or metric name
                      string (Default = :data:`rsgispy.METRIC_CORRELATION`).
    :param hist_bins: Histogram bin count for mutual information metrics
                      (Default = ``64``).
    :return: Computed metric value as a float.

    .. code:: python

        import numpy as np
        import rsgispy
        import rsgispy.tools.stats

        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([2.0, 4.0, 6.0, 8.0])
        corr = rsgispy.tools.stats.calc_similarity_metric(
            a, b, metric_id=rsgispy.METRIC_CORRELATION
        )
    """
    m_id, _, _ = resolve_metric_type(metric_id)
    r_arr = np.asarray(ref_vals, dtype=np.float64)
    f_arr = np.asarray(flt_vals, dtype=np.float64)

    if r_arr.shape != f_arr.shape:
        raise rsgispy.RSGISPyException(
            f"Input arrays must have identical shape: {r_arr.shape} vs {f_arr.shape}."
        )

    if m_id == rsgispy.METRIC_PHASE_CORR:
        if r_arr.ndim != 2:
            raise rsgispy.RSGISPyException(
                "METRIC_PHASE_CORR requires 2D spatial arrays."
            )
        _, _, peak = phase_cross_correlation_2d(r_arr, f_arr)
        return peak

    r_flat = r_arr.ravel() if r_arr.ndim > 1 else r_arr
    f_flat = f_arr.ravel() if f_arr.ndim > 1 else f_arr

    if r_flat.size < 2:
        return float("nan")
    if m_id == rsgispy.METRIC_CORRELATION:
        r0 = r_flat - np.mean(r_flat)
        f0 = f_flat - np.mean(f_flat)
        denom = float(np.sqrt(np.sum(r0 * r0) * np.sum(f0 * f0)))
        if denom <= 1e-12:
            return 0.0
        return float(np.sum(r0 * f0) / denom)
    if m_id == rsgispy.METRIC_EUCLIDEAN:
        return float(np.sqrt(np.mean((r_flat - f_flat) ** 2)))
    if m_id == rsgispy.METRIC_SQDIFF:
        return float(np.mean((r_flat - f_flat) ** 2))
    if m_id == rsgispy.METRIC_MANHATTAN:
        return float(np.mean(np.abs(r_flat - f_flat)))
    if m_id == rsgispy.METRIC_MUTUAL_INFO:
        return _mutual_information(
            r_flat, f_flat, hist_bins=hist_bins, normalised=False
        )
    if m_id == rsgispy.METRIC_NORM_MUTUAL_INFO:
        return _mutual_information(
            r_flat, f_flat, hist_bins=hist_bins, normalised=True
        )
    raise rsgispy.RSGISPyException(f"Unsupported similarity metric id: {metric_id}")


# Regularisation of the cross-power spectrum normalisation (fraction of its
# largest magnitude): frequencies with little energy are not whitened, which
# keeps noise and numerical round-off from dominating the correlation.
_PC_REG_FRAC = 0.01
# Maximum number of integer re-centring and sub-pixel refinement iterations.
_PC_MAX_INT_ITER = 3
_PC_MAX_SUB_ITER = 8
# Integer shift candidates: up to _PC_N_CANDS local peaks of the first
# correlation, those at least _PC_CAND_RATIO times the highest peak.
_PC_N_CANDS = 3
_PC_CAND_RATIO = 0.5
_PC_CAND_MARGIN = 0.1


def _hann_taper(n_pxls: int, offset: float = 0.0) -> np.ndarray:
    """
    Hann taper of length ``n_pxls`` (zero at both ends, as ``np.hanning``)
    evaluated at sample positions ``i + offset``; zero outside the window.
    """
    if n_pxls < 2:
        return np.ones(n_pxls, dtype=np.float64)
    pos = np.arange(n_pxls, dtype=np.float64) + offset
    taper = 0.5 - 0.5 * np.cos(2.0 * np.pi * pos / (n_pxls - 1))
    taper[(pos < 0.0) | (pos > n_pxls - 1)] = 0.0
    return taper


def _norm_cross_power(
    ref_arr: np.ndarray, flt_arr: np.ndarray
) -> Optional[Tuple[np.ndarray, float]]:
    """
    Regularised normalised cross-power spectrum of two equal-shape arrays and
    the normalisation that makes the correlation peak 1 for identical arrays
    (``None`` if both arrays are constant).
    """
    cross_power = np.fft.fft2(ref_arr) * np.conj(np.fft.fft2(flt_arr))
    mag = np.abs(cross_power)
    max_mag = float(np.max(mag))
    if max_mag <= 1e-12:
        return None
    reg = _PC_REG_FRAC * max_mag
    denom = max(float(np.sum(mag / (mag + reg))), 1e-12)
    return cross_power / (mag + reg), denom


def _corr_peaks(
    norm_cross: np.ndarray, denom: float, n_peaks: int = 1
) -> List[Tuple[int, int, float]]:
    """
    Integer shifts ``(dy, dx)`` and heights of the ``n_peaks`` highest local
    maxima (3 x 3 neighbourhood, circular) of the correlation surface, the
    highest first.
    """
    # pylint: disable=too-many-locals
    h_px, w_px = norm_cross.shape
    corr = np.real(np.fft.ifft2(norm_cross)) * float(h_px * w_px) / denom
    if n_peaks <= 1:
        flat_idxs = np.array([int(np.argmax(corr))])
    else:
        nbr_max = corr.copy()
        for r_shift in (-1, 0, 1):
            for c_shift in (-1, 0, 1):
                if r_shift or c_shift:
                    nbr_max = np.maximum(nbr_max, np.roll(corr, (r_shift, c_shift), axis=(0, 1)))
        local_idxs = np.flatnonzero(corr >= nbr_max)
        order = np.argsort(corr.ravel()[local_idxs])[::-1][:n_peaks]
        flat_idxs = local_idxs[order]
    peaks: List[Tuple[int, int, float]] = []
    for flat_idx in flat_idxs:
        i_y, i_x = divmod(int(flat_idx), w_px)
        d_y = i_y - h_px if i_y > h_px // 2 else i_y
        d_x = i_x - w_px if i_x > w_px // 2 else i_x
        peaks.append((d_y, d_x, float(corr[i_y, i_x])))
    return peaks


def _upsampled_corr_peak(
    norm_cross: np.ndarray,
    denom: float,
    centre_y: float,
    centre_x: float,
    upsample_factor: int,
) -> Tuple[float, float, float]:
    """
    Locate the correlation peak to ``1 / upsample_factor`` pixel within 0.75
    pixels of ``(centre_y, centre_x)`` with the matrix-multiply DFT of
    Guizar-Sicairos et al. (2008), evaluating only that neighbourhood.
    """
    # pylint: disable=too-many-locals
    h_px, w_px = norm_cross.shape
    n_pts = int(np.ceil(upsample_factor * 1.5))
    if n_pts % 2 == 0:
        n_pts += 1
    offs = (np.arange(n_pts) - (n_pts // 2)) / float(upsample_factor)
    c_y = centre_y + offs
    c_x = centre_x + offs
    k_y = np.exp(2j * np.pi * c_y[:, None] * np.fft.fftfreq(h_px)[None, :])
    k_x = np.exp(2j * np.pi * np.fft.fftfreq(w_px)[:, None] * c_x[None, :])
    corr_up = np.real(k_y @ norm_cross @ k_x) / denom
    u_y, u_x = divmod(int(np.argmax(corr_up)), n_pts)
    return float(c_y[u_y]), float(c_x[u_x]), float(corr_up[u_y, u_x])


def _overlap_cross_power(
    ref_arr: np.ndarray,
    flt_arr: np.ndarray,
    shift_y: float,
    shift_x: float,
    apply_window: bool,
) -> Optional[Tuple[np.ndarray, float]]:
    """
    Cross-power spectrum of the parts of the two windows that overlap when
    ``flt_arr`` is offset by the integer part of ``(shift_y, shift_x)``. The
    taper of the floating window is moved by the fractional part, so that
    for a pure translation both tapered windows contain the same tapered
    image content and the correlation peak is not pulled towards zero.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    h_px, w_px = ref_arr.shape
    i_y = int(round(shift_y))
    i_x = int(round(shift_x))
    ov_h = h_px - abs(i_y)
    ov_w = w_px - abs(i_x)
    if ov_h < 4 or ov_w < 4:
        return None
    r_off = max(0, i_y)
    c_off = max(0, i_x)
    ref_ov = ref_arr[r_off:r_off + ov_h, c_off:c_off + ov_w]
    flt_ov = flt_arr[r_off - i_y:r_off - i_y + ov_h, c_off - i_x:c_off - i_x + ov_w]
    ref_ov = ref_ov - float(np.mean(ref_ov))
    flt_ov = flt_ov - float(np.mean(flt_ov))
    if apply_window:
        ref_ov = ref_ov * (_hann_taper(ov_h)[:, None] * _hann_taper(ov_w)[None, :])
        flt_ov = flt_ov * (
            _hann_taper(ov_h, shift_y - i_y)[:, None]
            * _hann_taper(ov_w, shift_x - i_x)[None, :]
        )
    return _norm_cross_power(ref_ov, flt_ov)


def _recentre_int_shift(
    ref_arr: np.ndarray,
    flt_arr: np.ndarray,
    shift: Tuple[int, int],
    apply_window: bool,
    cache: Dict[Tuple[int, int], Tuple[int, int, float]],
) -> Optional[Tuple[int, int, float]]:
    """
    Correlate the overlap of the windows at the integer ``shift`` and move
    the shift until the overlap's correlation peak is at zero (at most
    ``_PC_MAX_INT_ITER`` moves). Returns ``(dy, dx, overlap_peak)`` or
    ``None`` if the overlap is too small. ``cache`` stores the overlap
    correlation of each shift already evaluated.
    """
    d_y, d_x = shift
    for n_iter in range(_PC_MAX_INT_ITER + 1):
        if (d_y, d_x) not in cache:
            ncp = _overlap_cross_power(ref_arr, flt_arr, d_y, d_x, apply_window)
            if ncp is None:
                return None
            cache[(d_y, d_x)] = _corr_peaks(*ncp)[0]
        s_y, s_x, peak = cache[(d_y, d_x)]
        if (s_y == 0 and s_x == 0) or n_iter == _PC_MAX_INT_ITER:
            break
        d_y += s_y
        d_x += s_x
    return d_y, d_x, peak


def _select_int_shift(
    ref_arr: np.ndarray,
    flt_arr: np.ndarray,
    cands: List[Tuple[int, int, float]],
    apply_window: bool,
) -> Tuple[int, int, float]:
    """
    Re-centre each candidate integer shift (highest correlation peak first)
    on the overlap of the windows and return the one whose overlap correlates
    best. Another candidate replaces the highest one only if its overlap
    peak is higher by more than ``_PC_CAND_MARGIN`` (so a repeating texture
    keeps the most likely shift), and only candidates at least
    ``_PC_CAND_RATIO`` times as high as the highest peak are tried.
    """
    cache: Dict[Tuple[int, int], Tuple[int, int, float]] = {}
    best = cands[0]
    best_found = False
    for c_y, c_x, c_height in cands:
        if c_height < _PC_CAND_RATIO * cands[0][2]:
            break
        res = _recentre_int_shift(ref_arr, flt_arr, (c_y, c_x), apply_window, cache)
        if res is not None and (not best_found or res[2] > best[2] + _PC_CAND_MARGIN):
            best = res
            best_found = True
    return best


def _refine_subpixel_shift(
    ref_arr: np.ndarray,
    flt_arr: np.ndarray,
    int_shift: Tuple[int, int, float],
    up_fac: int,
    apply_window: bool,
) -> Tuple[float, float, float]:
    """
    Refine the integer shift ``(dy, dx, peak)`` to ``1 / up_fac`` pixel by
    upsampled DFT of the overlap correlation, moving the floating window's
    taper by the current fractional estimate until the estimate is stable.
    """
    # pylint: disable=too-many-locals
    est_y = float(int_shift[0])
    est_x = float(int_shift[1])
    peak = float(int_shift[2])
    for _ in range(_PC_MAX_SUB_ITER):
        i_y = int(round(est_y))
        i_x = int(round(est_x))
        ncp = _overlap_cross_power(ref_arr, flt_arr, est_y, est_x, apply_window)
        if ncp is None:
            break
        s_y, s_x, peak = _upsampled_corr_peak(*ncp, est_y - i_y, est_x - i_x, up_fac)
        new_y = round((i_y + s_y) * up_fac) / up_fac
        new_x = round((i_x + s_x) * up_fac) / up_fac
        converged = abs(new_y - est_y) < 0.5 / up_fac and abs(new_x - est_x) < 0.5 / up_fac
        est_y, est_x = new_y, new_x
        if converged or not apply_window:
            break
    return float(est_y), float(est_x), float(peak)


def phase_cross_correlation_2d(
    ref_arr: np.ndarray,
    flt_arr: np.ndarray,
    upsample_factor: int = 4,
    apply_window: bool = True,
) -> Tuple[float, float, float]:
    """
    Estimate the sub-pixel 2D translation between ``ref_arr`` and ``flt_arr``
    by FFT phase cross-correlation.

    Returns ``(dy, dx, peak_response)`` where a feature at ``(r, c)`` in
    ``ref_arr`` appears at ``(r - dy, c - dx)`` in ``flt_arr``.

    The estimate is made in three stages:

    1. **Integer shift** — phase correlation of the two Hann-tapered
       windows. The cross-power spectrum is normalised with a small
       regularisation (1 % of its largest magnitude), so frequencies with
       little energy are not amplified. The highest correlation peak, and up
       to two more local peaks at least half as high, are candidates.
    2. **Window offset** — for each candidate, the windows are cropped to
       the part that overlaps at that integer shift and correlated again; if
       the peak is not at zero the shift is corrected and the overlap cropped
       again. The candidate whose overlap correlates best is kept.
    3. **Sub-pixel refinement** — the correlation peak of the overlap is
       located to ``1 / upsample_factor`` pixel by matrix-multiply DFT
       upsampling (Guizar-Sicairos et al., 2008). The taper of the floating
       window is moved by the current fractional estimate, so both tapered
       windows contain the same image content, and the estimate is iterated
       until it no longer changes.

    Tapering both windows with the same fixed taper (a single-pass phase
    correlation) multiplies the correlation by the taper's autocorrelation,
    which pulls the peak towards zero shift; with windows of 64 pixels or
    less this underestimated shifts by up to 25 %. Stages 2 and 3 remove this
    bias: on noise-free smoothed random textures, shifts up to a quarter of
    the window are recovered to 0.01 pixel (with ``upsample_factor=100``) for
    windows of 32, 64 and 128 pixels. The shift must be well below half the
    window size, and textures that repeat within the window (e.g. regular
    field patterns) can give a shift that is wrong by one period.

    ``peak_response`` is the height of the normalised correlation peak of
    the overlapping parts of the windows: 1 for identical content and lower
    for noisy or poorly matching windows. Unrelated smooth textures typically
    give 0.3-0.7 with 32-pixel windows, 0.2-0.4 with 64 and 0.1-0.2 with 128.

    :param ref_arr: 2D reference array.
    :param flt_arr: 2D floating array of identical shape.
    :param upsample_factor: Sub-pixel refinement factor: the shift is
                            estimated to ``1 / upsample_factor`` pixels
                            (Default = ``4``; ``1`` gives an integer shift).
    :param apply_window: Apply a 2D Hann taper before the FFTs
                         (Default = ``True``). With ``False`` no taper is
                         applied and the sub-pixel estimate is not iterated,
                         so edge effects can bias it.
    :return: Tuple ``(dy, dx, peak_response)``.

    .. code:: python

        import numpy as np
        import rsgispy.tools.stats

        a = np.random.default_rng(0).normal(size=(32, 32))
        dy, dx, peak = rsgispy.tools.stats.phase_cross_correlation_2d(a, a)
    """
    # pylint: disable=too-many-locals
    h_px, w_px = ref_arr.shape
    if h_px < 4 or w_px < 4:
        return 0.0, 0.0, 0.0
    ref_f = np.asarray(ref_arr, dtype=np.float64)
    flt_f = np.asarray(flt_arr, dtype=np.float64)

    # 1. Integer shift from the whole (tapered) windows.
    ref_0 = ref_f - float(np.mean(ref_f))
    flt_0 = flt_f - float(np.mean(flt_f))
    if apply_window:
        taper = _hann_taper(h_px)[:, None] * _hann_taper(w_px)[None, :]
        ref_0 = ref_0 * taper
        flt_0 = flt_0 * taper
    ncp = _norm_cross_power(ref_0, flt_0)
    if ncp is None:
        return 0.0, 0.0, 0.0
    cands = _corr_peaks(*ncp, n_peaks=_PC_N_CANDS)

    # 2. Re-centre the candidates on the overlap of the windows.
    d_y, d_x, peak = _select_int_shift(ref_f, flt_f, cands, apply_window)

    # 3. Iterative sub-pixel refinement with a matched floating taper.
    up_fac = max(1, int(upsample_factor))
    if up_fac == 1:
        return float(d_y), float(d_x), float(peak)
    return _refine_subpixel_shift(ref_f, flt_f, (d_y, d_x, peak), up_fac, apply_window)
