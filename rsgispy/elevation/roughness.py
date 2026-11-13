"""
A module to calculate surface roughness metrics for elevation data.
"""

import math
import os
from typing import Dict, Optional, Tuple, Union

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import rasterio
from rasterio.enums import Resampling

import rsgispy
import rsgispy.imagecalc.calc_pt_win_smpls

__all__ = [
    "CalcProfileRoughMetrics",
    "CalcMunroRoughnessMetric",
    "CalcSmithRoughnessMetric",
    "calc_simple_roughness_profile_metrics",
    "calc_munro_roughness_profile_metrics",
    "calc_smith_roughness_metrics",
    "calc_all_roughness_profile_metrics",
]


def _calc_skew(arr: np.ndarray) -> float:
    """Compute sample skewness (Fisher-Pearson, biased, matching ``scipy.stats.skew``)."""
    diff = arr - np.mean(arr)
    m2 = float(np.mean(diff**2))
    if m2 <= 0.0:
        return float("nan")
    m3 = float(np.mean(diff**3))
    return float(m3 / (m2**1.5))


def _calc_kurtosis(arr: np.ndarray) -> float:
    """Compute excess kurtosis (Fisher definition, matching ``scipy.stats.kurtosis``)."""
    diff = arr - np.mean(arr)
    m2 = float(np.mean(diff**2))
    if m2 <= 0.0:
        return float("nan")
    m4 = float(np.mean(diff**4))
    return float((m4 / (m2**2)) - 3.0)


def _extract_xy_profiles(
    in_img_ds_obj: rasterio.DatasetReader,
    detrend: bool,
    detrend_poly_order: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Read band 1 of ``in_img_ds_obj`` and return central X/Y coordinate arrays,
    optionally detrended X/Y elevation profiles, and ``(x_res, y_res)``.
    """
    # pylint: disable=too-many-locals
    x_res, y_res = in_img_ds_obj.res
    x_res = abs(float(x_res))
    y_res = abs(float(y_res))

    img_arr = in_img_ds_obj.read(1).astype(np.float64)
    y_size, x_size = img_arr.shape

    mid_x = min(math.ceil(x_size / 2), x_size - 1)
    mid_y = min(math.ceil(y_size / 2), y_size - 1)

    x_orig_arr = img_arr[mid_y, :]
    y_orig_arr = img_arr[:, mid_x]

    x_loc_arr = np.arange(x_size, dtype=np.float64) * x_res
    y_loc_arr = np.arange(y_size, dtype=np.float64) * y_res

    if detrend:
        x_poly_coef = np.polyfit(x_loc_arr, x_orig_arr, detrend_poly_order)
        y_poly_coef = np.polyfit(y_loc_arr, y_orig_arr, detrend_poly_order)
        x_arr = x_orig_arr - np.polyval(x_poly_coef, x_loc_arr)
        y_arr = y_orig_arr - np.polyval(y_poly_coef, y_loc_arr)
    else:
        x_arr = x_orig_arr
        y_arr = y_orig_arr

    return x_loc_arr, y_loc_arr, x_arr, y_arr, x_res, y_res


def _calc_munro_axis(
    arr_0mean: np.ndarray, profile_len: float
) -> Tuple[int, float, float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Compute Munro (1989) peak crossings and roughness length for a 1D profile."""
    arr_base = arr_0mean[1:-1]
    diff_right = arr_0mean[:-2]
    diff_left = arr_0mean[2:]

    pos_cross = np.logical_and(diff_right < 0.0, arr_base > 0.0)
    neg_cross = np.logical_and(arr_base > 0.0, diff_left < 0.0)
    min_pt_locs = np.logical_and(diff_right > arr_base, diff_left > arr_base)

    min_pt_locs = np.concatenate(([True], min_pt_locs, [True]))
    pos_cross = np.concatenate(([False], pos_cross, [False]))
    neg_cross = np.concatenate(([False], neg_cross, [False]))

    n_peaks = int(max(int(np.sum(pos_cross)), int(np.sum(neg_cross))))
    h_star = 2.0 * float(np.std(arr_0mean))

    if n_peaks == 0:
        return 0, 0.0, 0.0, 0.0, min_pt_locs, pos_cross, neg_cross

    little_s = (h_star * profile_len) / (2.0 * n_peaks)
    big_s = (profile_len / n_peaks) ** 2
    munro_z0 = 0.5 * h_star * (little_s / big_s)
    return (
        n_peaks,
        float(little_s),
        float(big_s),
        float(munro_z0),
        min_pt_locs,
        pos_cross,
        neg_cross,
    )


class CalcProfileRoughMetrics(rsgispy.imagecalc.calc_pt_win_smpls.RSGISCalcSumVals):
    """
    Calculator for profile surface roughness metrics in the X and Y axes
    (RA, RR, RQ, MIF, RV, RP, RZ, RSK, and RKU).
    """

    def __init__(
        self,
        detrend: bool = True,
        detrend_poly_order: int = 1,
        create_plots: bool = False,
        out_plot_path: Optional[str] = None,
    ) -> None:
        """
        :param detrend: Whether to detrend the elevation profiles. Default: ``True``.
        :param detrend_poly_order: Polynomial order used for detrending.
        :param create_plots: Whether to save diagnostic profile plots.
        :param out_plot_path: Directory path for exported plots when ``create_plots`` is ``True``.
        """
        super().__init__()
        self.detrend = detrend
        self.detrend_poly_order = detrend_poly_order
        self.create_plots = create_plots
        self.out_plot_path = out_plot_path
        self.out_val_names = [
            "x_ra",
            "y_ra",
            "sum_ra",
            "avg_ra",
            "x_rr",
            "y_rr",
            "sum_rr",
            "avg_rr",
            "x_rq",
            "y_rq",
            "sum_rq",
            "avg_rq",
            "x_MIF",
            "y_MIF",
            "sum_MIF",
            "avg_MIF",
            "x_rv",
            "y_rv",
            "sum_rv",
            "avg_rv",
            "x_rp",
            "y_rp",
            "sum_rp",
            "avg_rp",
            "x_rz",
            "y_rz",
            "sum_rz",
            "avg_rz",
            "x_rsk",
            "y_rsk",
            "sum_rsk",
            "avg_rsk",
            "x_rku",
            "y_rku",
            "sum_rku",
            "avg_rku",
        ]
        self.n_out_vals = len(self.out_val_names)

    def calcVals(  # pylint: disable=invalid-name,too-many-locals,too-many-statements
        self, smpl_idx: int, in_img_ds_obj: rasterio.DatasetReader
    ) -> Dict[str, float]:
        """
        Calculate profile roughness metrics along the central X and Y transects
        of the sample window.

        See https://en.wikipedia.org/wiki/Surface_roughness for metric definitions.

        :param smpl_idx: Unique index for the sample being processed.
        :param in_img_ds_obj: Input ``rasterio.DatasetReader`` window.
        :return: Dictionary of metric names and values.
        """
        x_loc_arr, y_loc_arr, x_arr, y_arr, _, _ = _extract_xy_profiles(
            in_img_ds_obj, self.detrend, self.detrend_poly_order
        )
        x_size = len(x_arr)
        y_size = len(y_arr)
        x_mean = float(np.mean(x_arr))
        y_mean = float(np.mean(y_arr))
        x_dev = x_arr - x_mean
        y_dev = y_arr - y_mean

        out_dict: Dict[str, float] = {}

        # Average roughness (RA)
        out_dict["x_ra"] = float(np.sum(np.abs(x_dev)) / x_size)
        out_dict["y_ra"] = float(np.sum(np.abs(y_dev)) / y_size)
        out_dict["sum_ra"] = out_dict["x_ra"] + out_dict["y_ra"]
        out_dict["avg_ra"] = out_dict["sum_ra"] / 2.0

        # Random roughness (RR)
        out_dict["x_rr"] = float(np.std(x_dev))
        out_dict["y_rr"] = float(np.std(y_dev))
        out_dict["sum_rr"] = out_dict["x_rr"] + out_dict["y_rr"]
        out_dict["avg_rr"] = out_dict["sum_rr"] / 2.0

        # Sum of absolute elevation differences (RQ)
        out_dict["x_rq"] = float(np.sum(np.abs(np.diff(x_arr))))
        out_dict["y_rq"] = float(np.sum(np.abs(np.diff(y_arr))))
        out_dict["sum_rq"] = out_dict["x_rq"] + out_dict["y_rq"]
        out_dict["avg_rq"] = out_dict["sum_rq"] / 2.0

        # Microrelief peak counts (MIF)
        x_mif_pxls = np.logical_and(
            (x_arr[1:-1] - x_arr[:-2]) > 0.0,
            (x_arr[2:] - x_arr[:-2]) > 0.0,
        )
        y_mif_pxls = np.logical_and(
            (y_arr[1:-1] - y_arr[:-2]) > 0.0,
            (y_arr[2:] - y_arr[:-2]) > 0.0,
        )
        out_dict["x_MIF"] = float(np.sum(x_mif_pxls))
        out_dict["y_MIF"] = float(np.sum(y_mif_pxls))
        out_dict["sum_MIF"] = out_dict["x_MIF"] + out_dict["y_MIF"]
        out_dict["avg_MIF"] = out_dict["sum_MIF"] / 2.0

        # Maximum valley depth below mean line (RV)
        out_dict["x_rv"] = float(np.min(x_dev))
        out_dict["y_rv"] = float(np.min(y_dev))
        out_dict["sum_rv"] = out_dict["x_rv"] + out_dict["y_rv"]
        out_dict["avg_rv"] = out_dict["sum_rv"] / 2.0

        # Maximum peak height above mean line (RP)
        out_dict["x_rp"] = float(np.max(x_dev))
        out_dict["y_rp"] = float(np.max(y_dev))
        out_dict["sum_rp"] = out_dict["x_rp"] + out_dict["y_rp"]
        out_dict["avg_rp"] = out_dict["sum_rp"] / 2.0

        # Maximum peak-to-valley height (RZ)
        out_dict["x_rz"] = out_dict["x_rp"] - out_dict["x_rv"]
        out_dict["y_rz"] = out_dict["y_rp"] - out_dict["y_rv"]
        out_dict["sum_rz"] = out_dict["x_rz"] + out_dict["y_rz"]
        out_dict["avg_rz"] = out_dict["sum_rz"] / 2.0

        # Skewness (RSK)
        out_dict["x_rsk"] = _calc_skew(x_dev)
        out_dict["y_rsk"] = _calc_skew(y_dev)
        out_dict["sum_rsk"] = out_dict["x_rsk"] + out_dict["y_rsk"]
        out_dict["avg_rsk"] = out_dict["sum_rsk"] / 2.0

        # Kurtosis (RKU)
        out_dict["x_rku"] = _calc_kurtosis(x_dev)
        out_dict["y_rku"] = _calc_kurtosis(y_dev)
        out_dict["sum_rku"] = out_dict["x_rku"] + out_dict["y_rku"]
        out_dict["avg_rku"] = out_dict["sum_rku"] / 2.0

        if self.create_plots:
            # pylint: disable=import-outside-toplevel
            from matplotlib import gridspec
            import matplotlib.pyplot as plt

            if self.out_plot_path is None:
                raise rsgispy.RSGISPyException(
                    "out_plot_path must be specified when create_plots is True."
                )
            os.makedirs(self.out_plot_path, exist_ok=True)

            fig = plt.figure(figsize=(10, 5))
            gspec = gridspec.GridSpec(ncols=1, nrows=2, figure=fig)

            ax1 = fig.add_subplot(gspec[0, 0])
            ax1.plot(x_loc_arr, x_arr)
            ax1.set_ylabel("Elevation (m)")
            ax1.set_title("X Axis Surface")

            ax2 = fig.add_subplot(gspec[1, 0], sharex=ax1)
            ax2.plot(y_loc_arr, y_arr)
            ax2.set_xlabel("Distance (m)")
            ax2.set_ylabel("Elevation (m)")
            ax2.set_title("Y Axis Surface")

            plt.savefig(
                os.path.join(self.out_plot_path, f"profile_smp_{smpl_idx}.png")
            )
            plt.close(fig)

        return out_dict


class CalcMunroRoughnessMetric(rsgispy.imagecalc.calc_pt_win_smpls.RSGISCalcSumVals):
    """
    Calculator for Munro (1989) aerodynamic roughness length (:math:`z_0`) metrics
    along the X and Y profile axes.
    """

    def __init__(
        self,
        detrend: bool = True,
        detrend_poly_order: int = 1,
        create_plots: bool = False,
        out_plot_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.detrend = detrend
        self.detrend_poly_order = detrend_poly_order
        self.create_plots = create_plots
        self.out_plot_path = out_plot_path
        self.out_val_names = [
            "x_munro_peaks",
            "x_munro_area",
            "x_munro_density",
            "x_munro_z0",
            "y_munro_peaks",
            "y_munro_area",
            "y_munro_density",
            "y_munro_z0",
            "sum_munro_z0",
            "avg_munro_z0",
        ]
        self.n_out_vals = len(self.out_val_names)

    def calcVals(  # pylint: disable=invalid-name,too-many-locals
        self, smpl_idx: int, in_img_ds_obj: rasterio.DatasetReader
    ) -> Dict[str, float]:
        """
        Calculate Munro (1989) roughness length metrics along X and Y transects.

        :param smpl_idx: Unique index for the sample being processed.
        :param in_img_ds_obj: Input ``rasterio.DatasetReader`` window.
        :return: Dictionary of metric names and values.
        """
        x_loc_arr, y_loc_arr, x_arr, y_arr, x_res, y_res = _extract_xy_profiles(
            in_img_ds_obj, self.detrend, self.detrend_poly_order
        )
        x_arr_0mean = x_arr - np.mean(x_arr)
        y_arr_0mean = y_arr - np.mean(y_arr)

        (
            x_n_peaks,
            x_little_s,
            x_big_s,
            x_munro_z0,
            x_min_pts,
            x_pos_cross,
            x_neg_cross,
        ) = _calc_munro_axis(x_arr_0mean, x_res * len(x_arr))
        (
            y_n_peaks,
            y_little_s,
            y_big_s,
            y_munro_z0,
            y_min_pts,
            y_pos_cross,
            y_neg_cross,
        ) = _calc_munro_axis(y_arr_0mean, y_res * len(y_arr))

        if self.create_plots:
            # pylint: disable=import-outside-toplevel
            from matplotlib import gridspec
            import matplotlib.pyplot as plt

            if self.out_plot_path is None:
                raise rsgispy.RSGISPyException(
                    "out_plot_path must be specified when create_plots is True."
                )
            os.makedirs(self.out_plot_path, exist_ok=True)

            fig = plt.figure(figsize=(10, 5))
            gspec = gridspec.GridSpec(ncols=1, nrows=2, figure=fig)

            ax1 = fig.add_subplot(gspec[0, 0])
            ax1.plot([x_loc_arr[0], x_loc_arr[-1]], [0.0, 0.0], color="black")
            ax1.plot(x_loc_arr, x_arr_0mean)
            ax1.scatter(x_loc_arr[x_pos_cross], x_arr_0mean[x_pos_cross])
            ax1.scatter(x_loc_arr[x_neg_cross], x_arr_0mean[x_neg_cross])
            ax1.plot(x_loc_arr[x_min_pts], x_arr_0mean[x_min_pts])
            ax1.set_ylabel("Elevation (m)")
            ax1.set_title("X Axis Surface")

            ax2 = fig.add_subplot(gspec[1, 0], sharex=ax1)
            ax2.plot([y_loc_arr[0], y_loc_arr[-1]], [0.0, 0.0], color="black")
            ax2.plot(y_loc_arr, y_arr_0mean)
            ax2.scatter(y_loc_arr[y_pos_cross], y_arr_0mean[y_pos_cross])
            ax2.scatter(y_loc_arr[y_neg_cross], y_arr_0mean[y_neg_cross])
            ax2.plot(y_loc_arr[y_min_pts], y_arr_0mean[y_min_pts])
            ax2.set_xlabel("Distance (m)")
            ax2.set_ylabel("Elevation (m)")
            ax2.set_title("Y Axis Surface")

            plt.savefig(
                os.path.join(
                    self.out_plot_path, f"munro_rough_profile_smp_{smpl_idx}.png"
                )
            )
            plt.close(fig)

        sum_munro_z0 = x_munro_z0 + y_munro_z0
        return {
            "x_munro_peaks": float(x_n_peaks),
            "x_munro_area": x_little_s,
            "x_munro_density": x_big_s,
            "x_munro_z0": x_munro_z0,
            "y_munro_peaks": float(y_n_peaks),
            "y_munro_area": y_little_s,
            "y_munro_density": y_big_s,
            "y_munro_z0": y_munro_z0,
            "sum_munro_z0": sum_munro_z0,
            "avg_munro_z0": sum_munro_z0 / 2.0,
        }


class CalcSmithRoughnessMetric(rsgispy.imagecalc.calc_pt_win_smpls.RSGISCalcSumVals):
    """
    Calculator for Smith et al. (2016) directional 2D surface roughness length
    (:math:`z_0`) metrics across a 2D elevation window.
    """

    def __init__(
        self,
        detrend: bool = True,
        create_plots: bool = False,
        out_plot_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.detrend = detrend
        self.create_plots = create_plots
        self.out_plot_path = out_plot_path
        self.out_val_names = [
            "smith_hs",
            "smith_area",
            "xe_smith_frt",
            "xe_smith_z0",
            "xw_smith_frt",
            "xw_smith_z0",
            "x_avg_smith_z0",
            "yn_smith_frt",
            "yn_smith_z0",
            "ys_smith_frt",
            "ys_smith_z0",
            "y_avg_smith_z0",
            "avg_smith_z0",
        ]
        self.n_out_vals = len(self.out_val_names)

    @staticmethod
    def _plot_smith_surfaces(
        smpl_idx: int,
        out_plot_path: Optional[str],
        detrend: bool,
        x_loc_2d_arr: np.ndarray,
        y_loc_2d_arr: np.ndarray,
        img_orig_arr: np.ndarray,
        plane_surf_arr: Optional[np.ndarray],
        img_data_arr: np.ndarray,
    ) -> None:
        # pylint: disable=import-outside-toplevel,too-many-arguments,too-many-positional-arguments
        import matplotlib.pyplot as plt

        if out_plot_path is None:
            raise rsgispy.RSGISPyException(
                "out_plot_path must be specified when create_plots is True."
            )
        os.makedirs(out_plot_path, exist_ok=True)

        if detrend:
            fig = plt.figure(figsize=(15, 5))
            panels = [
                (1, img_orig_arr, "Original Surface"),
                (2, plane_surf_arr, "Plane Surface"),
                (3, img_data_arr, "Detrended Surface"),
            ]
            for idx, z_arr, title in panels:
                ax = fig.add_subplot(1, 3, idx, projection="3d")
                ax.plot_surface(
                    x_loc_2d_arr,
                    y_loc_2d_arr,
                    z_arr,
                    cmap="coolwarm",
                    linewidth=0,
                    antialiased=False,
                )
                ax.set_title(title)
        else:
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(1, 1, 1, projection="3d")
            ax.plot_surface(
                x_loc_2d_arr,
                y_loc_2d_arr,
                img_orig_arr,
                cmap="coolwarm",
                linewidth=0,
                antialiased=False,
            )
            ax.set_title("Surface")

        plt.savefig(
            os.path.join(
                out_plot_path, f"smith_rough_profile_smp_{smpl_idx}.png"
            )
        )
        plt.close(fig)

    def calcVals(  # pylint: disable=invalid-name,too-many-locals
        self, smpl_idx: int, in_img_ds_obj: rasterio.DatasetReader
    ) -> Dict[str, float]:
        """
        Calculate Smith et al. (2016) directional roughness metrics.

        :param smpl_idx: Unique index for the sample being processed.
        :param in_img_ds_obj: Input ``rasterio.DatasetReader`` window.
        :return: Dictionary of metric names and values.
        """
        x_res, y_res = in_img_ds_obj.res
        x_res = abs(float(x_res))
        y_res = abs(float(y_res))

        img_orig_arr = in_img_ds_obj.read(1).astype(np.float64)
        y_size, x_size = img_orig_arr.shape
        if y_size < 3 or x_size < 3:
            raise rsgispy.RSGISPyException(
                f"Sample window must be at least 3x3 pixels (got {x_size}x{y_size})."
            )

        x_loc_2d_arr = None
        y_loc_2d_arr = None
        plane_surf_arr = None

        if self.detrend or self.create_plots:
            x_loc_arr = np.arange(x_size, dtype=np.float64) * x_res
            y_loc_arr = np.arange(y_size, dtype=np.float64) * y_res
            x_loc_2d_arr = np.repeat([x_loc_arr], repeats=y_size, axis=0)
            y_loc_2d_arr = np.repeat(
                np.reshape(y_loc_arr, (-1, 1)), repeats=x_size, axis=1
            )

        if self.detrend:
            plane_fit_locs = np.c_[
                x_loc_2d_arr.ravel(),
                y_loc_2d_arr.ravel(),
                np.ones(x_size * y_size, dtype=np.float64),
            ]
            plane_coefs, _, _, _ = np.linalg.lstsq(
                plane_fit_locs, img_orig_arr.ravel(), rcond=None
            )
            plane_surf_arr = (
                plane_coefs[0] * x_loc_2d_arr
                + plane_coefs[1] * y_loc_2d_arr
                + plane_coefs[2]
            )
            img_data_arr = img_orig_arr - plane_surf_arr
        else:
            img_data_arr = img_orig_arr

        h_star = 2.0 * float(np.std(img_data_arr))
        grd_area = (x_res * x_size) * (y_res * y_size)

        win_arr = sliding_window_view(img_data_arr, (3, 3)).reshape(-1, 3, 3)

        x_east_arr = np.maximum(win_arr[:, 1, 1] - win_arr[:, 1, 2], 0.0)
        x_west_arr = np.maximum(win_arr[:, 1, 1] - win_arr[:, 1, 0], 0.0)
        y_north_arr = np.maximum(win_arr[:, 1, 1] - win_arr[:, 0, 1], 0.0)
        y_south_arr = np.maximum(win_arr[:, 1, 1] - win_arr[:, 2, 1], 0.0)

        x_east_frt = float(np.sum(x_east_arr) * x_res)
        x_west_frt = float(np.sum(x_west_arr) * x_res)
        y_north_frt = float(np.sum(y_north_arr) * y_res)
        y_south_frt = float(np.sum(y_south_arr) * y_res)

        xe_smith_z0 = float(0.5 * h_star * (x_east_frt / grd_area))
        xw_smith_z0 = float(0.5 * h_star * (x_west_frt / grd_area))
        yn_smith_z0 = float(0.5 * h_star * (y_north_frt / grd_area))
        ys_smith_z0 = float(0.5 * h_star * (y_south_frt / grd_area))

        if self.create_plots:
            self._plot_smith_surfaces(
                smpl_idx=smpl_idx,
                out_plot_path=self.out_plot_path,
                detrend=self.detrend,
                x_loc_2d_arr=x_loc_2d_arr,
                y_loc_2d_arr=y_loc_2d_arr,
                img_orig_arr=img_orig_arr,
                plane_surf_arr=plane_surf_arr,
                img_data_arr=img_data_arr,
            )

        return {
            "smith_hs": h_star,
            "smith_area": float(grd_area),
            "xe_smith_frt": x_east_frt,
            "xe_smith_z0": xe_smith_z0,
            "xw_smith_frt": x_west_frt,
            "xw_smith_z0": xw_smith_z0,
            "x_avg_smith_z0": (xe_smith_z0 + xw_smith_z0) / 2.0,
            "yn_smith_frt": y_north_frt,
            "yn_smith_z0": yn_smith_z0,
            "ys_smith_frt": y_south_frt,
            "ys_smith_z0": ys_smith_z0,
            "y_avg_smith_z0": (yn_smith_z0 + ys_smith_z0) / 2.0,
            "avg_smith_z0": (
                xe_smith_z0 + xw_smith_z0 + yn_smith_z0 + ys_smith_z0
            )
            / 4.0,
        }


def calc_simple_roughness_profile_metrics(
    input_img: str,
    vec_file: str,
    vec_lyr: Optional[str],
    out_vec_file: str,
    out_vec_lyr: Optional[str] = None,
    out_format: str = "GPKG",
    interp_method: Union[int, Resampling] = rsgispy.INTERP_CUBIC,
    angle_col: Optional[str] = None,
    x_box_col: str = "xbox",
    y_box_col: str = "ybox",
    no_data_val: Optional[float] = None,
    detrend: bool = True,
    detrend_poly_order: int = 1,
    export_plot_path: Optional[str] = None,
) -> None:
    """
    Calculate profile roughness metrics (RA, RR, RQ, MIF, RV, RP, RZ, RSK, RKU)
    along the X and Y axes of each point's sample window using
    :class:`CalcProfileRoughMetrics`.

    :param input_img: Path to input elevation image file.
    :param vec_file: Path to input point vector file.
    :param vec_lyr: Input vector layer name (or ``None`` for default layer).
    :param out_vec_file: Path to output vector file.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector file format (e.g., ``'GPKG'``, ``'GeoJSON'``).
    :param interp_method: Interpolation method used when reorientating windows.
                          Default: ``rsgispy.INTERP_CUBIC``.
    :param angle_col: Column specifying rotation angle (degrees relative to north).
    :param x_box_col: Column specifying half-window width in X pixels.
    :param y_box_col: Column specifying half-window height in Y pixels.
    :param no_data_val: Image no-data value; ``None`` reads from header.
    :param detrend: Whether to detrend the surface profiles. Default: ``True``.
    :param detrend_poly_order: Polynomial order used to detrend the profiles.
    :param export_plot_path: Optional directory to save diagnostic PNG plots.

    .. code:: python

        import rsgispy.elevation.roughness

        rsgispy.elevation.roughness.calc_simple_roughness_profile_metrics(
            input_img="dem.tif",
            vec_file="sample_pts.gpkg",
            vec_lyr="pts",
            out_vec_file="roughness_out.gpkg",
            out_vec_lyr="pts_rough",
            out_format="GPKG",
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    rough_profile_obj = CalcProfileRoughMetrics(
        detrend=detrend,
        detrend_poly_order=detrend_poly_order,
        create_plots=export_plot_path is not None,
        out_plot_path=export_plot_path,
    )
    rsgispy.imagecalc.calc_pt_win_smpls.calc_pt_smpl_img_vals(
        input_img,
        vec_file,
        vec_lyr,
        calc_objs=[rough_profile_obj],
        out_vec_file=out_vec_file,
        out_vec_lyr=out_vec_lyr,
        out_format=out_format,
        interp_method=interp_method,
        angle_col=angle_col,
        x_box_col=x_box_col,
        y_box_col=y_box_col,
        no_data_val=no_data_val,
    )


def calc_munro_roughness_profile_metrics(
    input_img: str,
    vec_file: str,
    vec_lyr: Optional[str],
    out_vec_file: str,
    out_vec_lyr: Optional[str] = None,
    out_format: str = "GPKG",
    interp_method: Union[int, Resampling] = rsgispy.INTERP_CUBIC,
    angle_col: Optional[str] = None,
    x_box_col: str = "xbox",
    y_box_col: str = "ybox",
    no_data_val: Optional[float] = None,
    detrend: bool = True,
    detrend_poly_order: int = 1,
    export_plot_path: Optional[str] = None,
) -> None:
    """
    Calculate Munro (1989) :math:`z_0` roughness metrics along the X and Y axes
    of each point's sample window using :class:`CalcMunroRoughnessMetric`.

    :param input_img: Path to input elevation image file.
    :param vec_file: Path to input point vector file.
    :param vec_lyr: Input vector layer name (or ``None`` for default layer).
    :param out_vec_file: Path to output vector file.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector file format (e.g., ``'GPKG'``, ``'GeoJSON'``).
    :param interp_method: Interpolation method used when reorientating windows.
                          Default: ``rsgispy.INTERP_CUBIC``.
    :param angle_col: Column specifying rotation angle (degrees relative to north).
    :param x_box_col: Column specifying half-window width in X pixels.
    :param y_box_col: Column specifying half-window height in Y pixels.
    :param no_data_val: Image no-data value; ``None`` reads from header.
    :param detrend: Whether to detrend the surface profiles. Default: ``True``.
    :param detrend_poly_order: Polynomial order used to detrend the profiles.
    :param export_plot_path: Optional directory to save diagnostic PNG plots.

    .. code:: python

        import rsgispy.elevation.roughness

        rsgispy.elevation.roughness.calc_munro_roughness_profile_metrics(
            input_img="dem.tif",
            vec_file="sample_pts.gpkg",
            vec_lyr="pts",
            out_vec_file="munro_rough_out.gpkg",
            out_vec_lyr="pts_munro",
            out_format="GPKG",
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    rough_profile_obj = CalcMunroRoughnessMetric(
        detrend=detrend,
        detrend_poly_order=detrend_poly_order,
        create_plots=export_plot_path is not None,
        out_plot_path=export_plot_path,
    )
    rsgispy.imagecalc.calc_pt_win_smpls.calc_pt_smpl_img_vals(
        input_img,
        vec_file,
        vec_lyr,
        calc_objs=[rough_profile_obj],
        out_vec_file=out_vec_file,
        out_vec_lyr=out_vec_lyr,
        out_format=out_format,
        interp_method=interp_method,
        angle_col=angle_col,
        x_box_col=x_box_col,
        y_box_col=y_box_col,
        no_data_val=no_data_val,
    )


def calc_smith_roughness_metrics(
    input_img: str,
    vec_file: str,
    vec_lyr: Optional[str],
    out_vec_file: str,
    out_vec_lyr: Optional[str] = None,
    out_format: str = "GPKG",
    interp_method: Union[int, Resampling] = rsgispy.INTERP_CUBIC,
    angle_col: Optional[str] = None,
    x_box_col: str = "xbox",
    y_box_col: str = "ybox",
    no_data_val: Optional[float] = None,
    detrend: bool = True,
    export_plot_path: Optional[str] = None,
) -> None:
    """
    Calculate Smith et al. (2016) directional :math:`z_0` roughness metrics
    across each point's sample window using :class:`CalcSmithRoughnessMetric`.

    :param input_img: Path to input elevation image file.
    :param vec_file: Path to input point vector file.
    :param vec_lyr: Input vector layer name (or ``None`` for default layer).
    :param out_vec_file: Path to output vector file.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector file format (e.g., ``'GPKG'``, ``'GeoJSON'``).
    :param interp_method: Interpolation method used when reorientating windows.
                          Default: ``rsgispy.INTERP_CUBIC``.
    :param angle_col: Column specifying rotation angle (degrees relative to north).
    :param x_box_col: Column specifying half-window width in X pixels.
    :param y_box_col: Column specifying half-window height in Y pixels.
    :param no_data_val: Image no-data value; ``None`` reads from header.
    :param detrend: Whether to plane-detrend the surface. Default: ``True``.
    :param export_plot_path: Optional directory to save diagnostic PNG plots.

    .. code:: python

        import rsgispy.elevation.roughness

        rsgispy.elevation.roughness.calc_smith_roughness_metrics(
            input_img="dem.tif",
            vec_file="sample_pts.gpkg",
            vec_lyr="pts",
            out_vec_file="smith_rough_out.gpkg",
            out_vec_lyr="pts_smith",
            out_format="GPKG",
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    rough_profile_obj = CalcSmithRoughnessMetric(
        detrend=detrend,
        create_plots=export_plot_path is not None,
        out_plot_path=export_plot_path,
    )
    rsgispy.imagecalc.calc_pt_win_smpls.calc_pt_smpl_img_vals(
        input_img,
        vec_file,
        vec_lyr,
        calc_objs=[rough_profile_obj],
        out_vec_file=out_vec_file,
        out_vec_lyr=out_vec_lyr,
        out_format=out_format,
        interp_method=interp_method,
        angle_col=angle_col,
        x_box_col=x_box_col,
        y_box_col=y_box_col,
        no_data_val=no_data_val,
    )


def calc_all_roughness_profile_metrics(
    input_img: str,
    vec_file: str,
    vec_lyr: Optional[str],
    out_vec_file: str,
    out_vec_lyr: Optional[str] = None,
    out_format: str = "GPKG",
    interp_method: Union[int, Resampling] = rsgispy.INTERP_CUBIC,
    angle_col: Optional[str] = None,
    x_box_col: str = "xbox",
    y_box_col: str = "ybox",
    no_data_val: Optional[float] = None,
    detrend: bool = True,
    detrend_poly_order: int = 1,
) -> None:
    """
    Calculate all roughness metrics (:class:`CalcProfileRoughMetrics`,
    :class:`CalcMunroRoughnessMetric`, and :class:`CalcSmithRoughnessMetric`)
    in a single pass over the point sample windows.

    :param input_img: Path to input elevation image file.
    :param vec_file: Path to input point vector file.
    :param vec_lyr: Input vector layer name (or ``None`` for default layer).
    :param out_vec_file: Path to output vector file.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector file format (e.g., ``'GPKG'``, ``'GeoJSON'``).
    :param interp_method: Interpolation method used when reorientating windows.
                          Default: ``rsgispy.INTERP_CUBIC``.
    :param angle_col: Column specifying rotation angle (degrees relative to north).
    :param x_box_col: Column specifying half-window width in X pixels.
    :param y_box_col: Column specifying half-window height in Y pixels.
    :param no_data_val: Image no-data value; ``None`` reads from header.
    :param detrend: Whether to detrend the surfaces/profiles. Default: ``True``.
    :param detrend_poly_order: Polynomial order used to detrend 1D profiles.

    .. code:: python

        import rsgispy.elevation.roughness

        rsgispy.elevation.roughness.calc_all_roughness_profile_metrics(
            input_img="dem.tif",
            vec_file="sample_pts.gpkg",
            vec_lyr="pts",
            out_vec_file="all_rough_out.gpkg",
            out_vec_lyr="pts_rough",
            out_format="GPKG",
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    rough_profile_obj = CalcProfileRoughMetrics(
        detrend=detrend, detrend_poly_order=detrend_poly_order
    )
    munro_profile_obj = CalcMunroRoughnessMetric(
        detrend=detrend, detrend_poly_order=detrend_poly_order
    )
    smith_profile_obj = CalcSmithRoughnessMetric(detrend=detrend)

    rsgispy.imagecalc.calc_pt_win_smpls.calc_pt_smpl_img_vals(
        input_img,
        vec_file,
        vec_lyr,
        calc_objs=[rough_profile_obj, munro_profile_obj, smith_profile_obj],
        out_vec_file=out_vec_file,
        out_vec_lyr=out_vec_lyr,
        out_format=out_format,
        interp_method=interp_method,
        angle_col=angle_col,
        x_box_col=x_box_col,
        y_box_col=y_box_col,
        no_data_val=no_data_val,
    )
