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
    (NumPy's default ``method="linear"``) to floating-point precision, but is
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


def phase_cross_correlation_2d(
    ref_arr: np.ndarray,
    flt_arr: np.ndarray,
    upsample_factor: int = 4,
    apply_window: bool = True,
) -> Tuple[float, float, float]:
    """
    Estimate sub-pixel 2D translation between ``ref_arr`` and ``flt_arr`` via
    FFT phase cross-correlation with upsampled DFT refinement.

    Returns ``(dy, dx, peak_response)`` where a feature at ``(r, c)`` in
    ``ref_arr`` appears at ``(r - dy, c - dx)`` in ``flt_arr``.

    :param ref_arr: 2D reference array.
    :param flt_arr: 2D floating array of identical shape.
    :param upsample_factor: Sub-pixel DFT refinement factor (Default = ``4``).
    :param apply_window: Apply a 2D Hanning taper before FFT (Default = ``True``).
    :return: Tuple ``(dy, dx, peak_response)``.

    .. code:: python

        import numpy as np
        import rsgispy.tools.stats

        a = np.random.default_rng(0).normal(size=(32, 32))
        dy, dx, peak = rsgispy.tools.stats.phase_cross_correlation_2d(a, a)
    """
    # pylint: disable=too-many-locals
    h, w = ref_arr.shape
    if h < 4 or w < 4:
        return 0.0, 0.0, 0.0

    r0 = ref_arr.astype(np.float64, copy=True) - float(np.mean(ref_arr))
    f0 = flt_arr.astype(np.float64, copy=True) - float(np.mean(flt_arr))
    if apply_window:
        win = np.hanning(h)[:, None] * np.hanning(w)[None, :]
        r0 *= win
        f0 *= win

    f_r = np.fft.fft2(r0)
    f_f = np.fft.fft2(f0)
    cross_power = f_r * np.conj(f_f)
    mag = np.abs(cross_power)
    max_mag = float(np.max(mag))
    if max_mag <= 1e-12:
        return 0.0, 0.0, 0.0
    reg = 0.01 * max_mag
    norm_cross = cross_power / (mag + reg)
    denom = max(float(np.sum(mag / (mag + reg))) / float(h * w), 1e-12)
    cc = np.real(np.fft.ifft2(norm_cross)) / denom
    iy, ix = np.unravel_index(np.argmax(cc), cc.shape)
    dy = float(iy - h if iy > h // 2 else iy)
    dx = float(ix - w if ix > w // 2 else ix)
    peak = float(cc[iy, ix])

    up = max(1, int(upsample_factor))
    if up > 1:
        n_pts = int(np.ceil(up * 1.5))
        if n_pts % 2 == 0:
            n_pts += 1
        offs = (np.arange(n_pts) - (n_pts // 2)) / float(up)
        cy = dy + offs
        cx = dx + offs
        fy = np.fft.fftfreq(h)
        fx = np.fft.fftfreq(w)
        ky = np.exp(2j * np.pi * cy[:, None] * fy[None, :])
        kx = np.exp(2j * np.pi * fx[:, None] * cx[None, :])
        cc_up = np.real(ky @ norm_cross @ kx) / (float(h * w) * denom)
        uy, ux = np.unravel_index(np.argmax(cc_up), cc_up.shape)
        dy = float(cy[uy])
        dx = float(cx[ux])
        peak = float(cc_up[uy, ux])

    return dy, dx, peak
