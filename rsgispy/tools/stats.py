"""
The tools.stats module provides array statistical filtering, spatial
variogram calculation, percentile computation, and data normalisation utilities.
"""

import math
from typing import Any, List, Optional, Sequence, Union
import warnings

import numpy

import rsgispy

__all__ = [
    "mask_data_to_valid",
    "calc_semivariogram",
    "standarise_img_data",
    "normalise_img_data",
    "validate_percentiles_0_100",
    "compute_nanpercentiles",
]


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

    .. code:: python

        import numpy as np
        import rsgispy.tools.stats
        arr = np.array([[1.0, 2.0], [np.nan, 3.0], [0.0, 5.0]])
        valid = rsgispy.tools.stats.mask_data_to_valid(arr, lower_limit=0.0)
    """
    filtered = data[numpy.all(numpy.isfinite(data), axis=1)]
    if lower_limit is not None:
        filtered = filtered[numpy.all(filtered > lower_limit, axis=1)]
    if upper_limit is not None:
        filtered = filtered[numpy.all(filtered < upper_limit, axis=1)]
    return filtered


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

    .. code:: python

        import numpy as np
        import rsgispy.tools.stats
        img = np.random.rand(3, 10, 10)
        std_img = rsgispy.tools.stats.standarise_img_data(img)
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

    .. code:: python

        import numpy as np
        import rsgispy.tools.stats
        img = np.random.rand(3, 10, 10)
        norm_img = rsgispy.tools.stats.normalise_img_data(img)
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
    if isinstance(percentiles, (int, float, numpy.number)):
        vals: List[float] = [float(percentiles)]
    elif isinstance(percentiles, numpy.ndarray) and percentiles.ndim == 0:
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
    stack_arr: numpy.ndarray,
    percentiles: Union[float, Sequence[float]],
    no_data_val: Optional[float] = None,
) -> numpy.ndarray:
    """
    Compute per-pixel percentiles along axis 0 while ignoring ``no_data_val``.

    :param stack_arr: Input array stacked along axis 0 (e.g. ``(n_layers, height, width)``).
                      Floating-point arrays may be modified in-place to replace
                      ``no_data_val`` with ``NaN``.
    :param percentiles: Percentile value or sequence of percentiles in ``[0, 100]``.
    :param no_data_val: Optional no-data value to mask out before computing
                        percentiles and restore where all values along axis 0
                        are no-data/NaN (Default = None).
    :return: NumPy array of computed percentiles along axis 0. For 3D inputs and
             scalar ``percentiles``, the output is expanded to ``(1, height, width)``.

    .. code:: python

        import numpy as np
        import rsgispy.tools.stats
        stack = np.random.rand(5, 10, 10)
        pct = rsgispy.tools.stats.compute_nanpercentiles(stack, [25, 50, 75])
    """
    if not numpy.issubdtype(stack_arr.dtype, numpy.floating):
        stack_arr = stack_arr.astype(numpy.float64)

    if no_data_val is not None:
        if not (
            isinstance(no_data_val, (float, numpy.floating))
            and math.isnan(float(no_data_val))
        ):
            stack_arr[stack_arr == no_data_val] = numpy.nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        pct_arr = numpy.nanpercentile(stack_arr, percentiles, axis=0)

    if pct_arr.ndim == 2:
        pct_arr = numpy.expand_dims(pct_arr, axis=0)

    if no_data_val is not None:
        return numpy.nan_to_num(pct_arr, copy=False, nan=no_data_val)
    return pct_arr
