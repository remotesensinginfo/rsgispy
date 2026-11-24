"""
Tie-point generation (FFT phase cross-correlation and template matching),
geometric model fitting, and outlier filtering (RANSAC and iterative residual
thresholding) for :mod:`rsgispy.imageregistration`.
"""

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import rasterio
from numpy.lib.stride_tricks import sliding_window_view

import rsgispy
from rsgispy.imageprocess import ensure_file_exists, extract_overlap_arrays
from rsgispy.imageregistration import (
    OUTLIER_METHOD_MAD,
    OUTLIER_METHOD_RANSAC,
    OUTLIER_METHOD_RESIDUAL,
    TIE_METHOD_PHASE_CORR,
    TIE_METHOD_TEMPLATE_MATCH,
    TRANSFORM_AFFINE,
    TRANSFORM_POLY1,
    TRANSFORM_POLY2,
    TRANSFORM_POLY3,
    TRANSFORM_SHIFT,
    TRANSFORM_SIMILARITY,
    TYPE_RSGIS_IMG2MAP,
)
from rsgispy.imageregistration._helpers import (
    find_surface_extremum,
    load_tie_points_table,
    refine_grid_subpixel,
    write_tie_points_file,
)
from rsgispy.tools.stats import (
    calc_similarity_metric,
    phase_cross_correlation_2d,
    resolve_metric_type,
)

__all__ = [
    "generate_tie_points",
    "basic_registration",
    "filter_tie_points_outliers",
    "fit_geometric_model",
    "evaluate_geometric_model",
]

_TIE_POINT_COLUMNS = [
    "ref_x",
    "ref_y",
    "flt_x",
    "flt_y",
    "ref_pxl_x",
    "ref_pxl_y",
    "flt_pxl_x",
    "flt_pxl_y",
    "dx_map",
    "dy_map",
    "dx_pxl",
    "dy_pxl",
    "metric",
]

_MODEL_MIN_SAMPLES: Dict[str, int] = {
    TRANSFORM_SHIFT: 1,
    "translation": 1,
    TRANSFORM_SIMILARITY: 2,
    TRANSFORM_AFFINE: 3,
    TRANSFORM_POLY1: 3,
    TRANSFORM_POLY2: 6,
    TRANSFORM_POLY3: 10,
}


def _match_template_patch(
    ref_patch: np.ndarray,
    flt_search: np.ndarray,
    ref_mask: np.ndarray,
    flt_mask: np.ndarray,
    search_area: int,
    metric_id: int,
    maximise: bool,
    sub_pxl_res: int,
    hist_bins: int,
) -> Optional[Tuple[float, float, float]]:
    """Find optimal sub-pixel shift ``(dy, dx, score)`` for a single patch."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    win_h, win_w = ref_patch.shape
    sa = int(search_area)
    n_grid = 2 * sa + 1

    # Fast vectorised path when patch and search area have no nodata pixels
    if np.all(ref_mask) and np.all(flt_mask):
        # sliding_window_view shape: (2*sa+1, 2*sa+1, win_h, win_w)
        # Note: index [r_idx, c_idx] corresponds to top-left (r_idx, c_idx) = (sa - sy, sa - sx)
        windows = sliding_window_view(flt_search, (win_h, win_w))
        if metric_id == rsgispy.METRIC_CORRELATION:
            r0 = ref_patch - float(np.mean(ref_patch))
            r_ss = float(np.sum(r0 * r0))
            if r_ss <= 1e-12:
                return None
            w_means = np.mean(windows, axis=(2, 3), keepdims=True)
            w0 = windows - w_means
            w_ss = np.sum(w0 * w0, axis=(2, 3))
            denom = np.sqrt(r_ss * np.maximum(w_ss, 0.0))
            num = np.sum(w0 * r0[None, None, :, :], axis=(2, 3))
            surface = np.where(denom > 1e-12, num / np.where(denom > 1e-12, denom, 1.0), 0.0)
        elif metric_id == rsgispy.METRIC_SQDIFF:
            diff = windows - ref_patch[None, None, :, :]
            surface = np.mean(diff * diff, axis=(2, 3))
        elif metric_id == rsgispy.METRIC_EUCLIDEAN:
            diff = windows - ref_patch[None, None, :, :]
            surface = np.sqrt(np.mean(diff * diff, axis=(2, 3)))
        elif metric_id == rsgispy.METRIC_MANHATTAN:
            surface = np.mean(np.abs(windows - ref_patch[None, None, :, :]), axis=(2, 3))
        else:
            surface = np.full((n_grid, n_grid), np.nan, dtype=np.float64)
            r_flat = ref_patch.ravel()
            for r_idx in range(n_grid):
                for c_idx in range(n_grid):
                    surface[r_idx, c_idx] = calc_similarity_metric(
                        r_flat,
                        windows[r_idx, c_idx].ravel(),
                        metric_id=metric_id,
                        hist_bins=hist_bins,
                    )
        # Flip axes so index [yi, xi] corresponds to sy = yi - sa, sx = xi - sa
        surface = surface[::-1, ::-1]
    else:
        surface = np.full((n_grid, n_grid), np.nan, dtype=np.float64)
        min_pxls = max(8, int(0.5 * win_h * win_w))
        for yi, sy in enumerate(range(-sa, sa + 1)):
            r0 = sa - sy
            for xi, sx in enumerate(range(-sa, sa + 1)):
                c0 = sa - sx
                m = ref_mask & flt_mask[r0:r0 + win_h, c0:c0 + win_w]
                if int(np.count_nonzero(m)) < min_pxls:
                    continue
                r_sub = ref_patch[m]
                f_sub = flt_search[r0:r0 + win_h, c0:c0 + win_w][m]
                surface[yi, xi] = calc_similarity_metric(
                    r_sub, f_sub, metric_id=metric_id, hist_bins=hist_bins
                )

    if not np.any(np.isfinite(surface)):
        return None

    iy, ix = find_surface_extremum(surface, maximise=maximise)
    iy_f, ix_f, score = refine_grid_subpixel(
        surface, iy, ix, sub_pxl_res=sub_pxl_res, maximise=maximise
    )
    return iy_f - float(sa), ix_f - float(sa), float(score)


def generate_tie_points(
    in_ref_img: str,
    in_float_img: str,
    out_gcp_file: Optional[Union[str, os.PathLike]] = None,
    method: str = TIE_METHOD_PHASE_CORR,
    ref_band: int = 1,
    flt_band: int = 1,
    pixel_gap: int = 50,
    win_size: int = 64,
    search_area: int = 15,
    threshold: Optional[float] = None,
    sd_ref_thres: float = 1.0,
    sd_flt_thres: float = 1.0,
    sub_pxl_res: int = 4,
    metric_type: Union[int, str] = rsgispy.METRIC_CORRELATION,
    output_type: int = TYPE_RSGIS_IMG2MAP,
    hist_bins: int = 64,
    ref_no_data: Optional[float] = None,
    flt_no_data: Optional[float] = None,
    use_pbar: bool = True,
) -> Optional[pd.DataFrame]:
    """
    Generate tie points across the overlapping region of ``in_ref_img`` and
    ``in_float_img`` using either FFT phase cross-correlation or sliding-window
    template matching.

    Returns a :class:`pandas.DataFrame` when ``out_gcp_file`` is ``None``, or
    writes the tie points to disk (CSV, Parquet, vector layer, or RSGIS/ENVI
    GCP text file) and returns ``None``.

    :param in_ref_img: Path to the reference raster image.
    :param in_float_img: Path to the floating raster image to co-register.
    :param out_gcp_file: Optional output file path (``None`` returns DataFrame).
    :param method: Matching method: :data:`TIE_METHOD_PHASE_CORR` (default) or
                   :data:`TIE_METHOD_TEMPLATE_MATCH`.
    :param ref_band: 1-based band index in ``in_ref_img`` (Default = ``1``).
    :param flt_band: 1-based band index in ``in_float_img`` (Default = ``1``).
    :param pixel_gap: Grid spacing in pixels between candidate tie points
                      (Default = ``50``).
    :param win_size: Matching window width/height in pixels (Default = ``64``).
    :param search_area: Search radius in pixels around each candidate point
                        (Default = ``15``).
    :param threshold: Minimum (or maximum for distance metrics) match score
                      threshold. Defaults to ``0.15`` for
                      :data:`TIE_METHOD_PHASE_CORR` and ``0.4`` for
                      :data:`TIE_METHOD_TEMPLATE_MATCH`.
    :param sd_ref_thres: Minimum reference patch standard deviation (Default = ``1.0``).
    :param sd_flt_thres: Minimum floating patch standard deviation (Default = ``1.0``).
    :param sub_pxl_res: Sub-pixel precision divisor (e.g., ``4`` -> ``0.25``
                        pixels; Default = ``4``).
    :param metric_type: Similarity/distance metric for
                        :data:`TIE_METHOD_TEMPLATE_MATCH`
                        (Default = :data:`rsgispy.METRIC_CORRELATION`).
    :param output_type: Output text format constant ``TYPE_*`` when writing a
                        plain text GCP file (Default = :data:`TYPE_RSGIS_IMG2MAP`).
    :param hist_bins: Histogram bins for mutual information metrics (Default = ``64``).
    :param ref_no_data: Optional nodata override for ``in_ref_img``.
    :param flt_no_data: Optional nodata override for ``in_float_img``.
    :param use_pbar: Display tqdm progress bar (Default = ``True``).
    :return: Tie-point :class:`pandas.DataFrame` if ``out_gcp_file`` is
             ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.imageregistration as imreg
        import rsgispy.imageregistration.tiepoints as imreg_tp

        tie_df = imreg_tp.generate_tie_points(
            in_ref_img="ref.tif",
            in_float_img="float.tif",
            method=imreg.TIE_METHOD_PHASE_CORR,
            pixel_gap=64,
            win_size=64,
            search_area=12,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    ref_path = os.fspath(in_ref_img)
    flt_path = os.fspath(in_float_img)
    ensure_file_exists(ref_path, label="Reference image")
    ensure_file_exists(flt_path, label="Floating image")

    method_key = method.strip().lower().replace("-", "_").replace(" ", "_")
    if method_key not in (TIE_METHOD_PHASE_CORR, TIE_METHOD_TEMPLATE_MATCH):
        raise rsgispy.RSGISPyException(
            f"Unsupported tie-point method '{method}'. Choose "
            f"'{TIE_METHOD_PHASE_CORR}' or '{TIE_METHOD_TEMPLATE_MATCH}'."
        )
    if pixel_gap < 1 or win_size < 4 or search_area < 0 or sub_pxl_res < 1:
        raise rsgispy.RSGISPyException(
            "pixel_gap >= 1, win_size >= 4, search_area >= 0, and sub_pxl_res >= 1 required."
        )

    metric_id, _, maximise = resolve_metric_type(metric_type)
    if threshold is None:
        thres_val = 0.15 if method_key == TIE_METHOD_PHASE_CORR else 0.4
    else:
        thres_val = float(threshold)

    with rasterio.open(ref_path, "r") as ref_src, rasterio.open(
        flt_path, "r"
    ) as flt_src:
        if ref_band < 1 or ref_band > ref_src.count:
            raise rsgispy.RSGISPyException(
                f"Reference band {ref_band} out of bounds for '{ref_path}'."
            )
        if flt_band < 1 or flt_band > flt_src.count:
            raise rsgispy.RSGISPyException(
                f"Floating band {flt_band} out of bounds for '{flt_path}'."
            )

        ov = extract_overlap_arrays(
            ref_src,
            flt_src,
            [ref_band],
            [flt_band],
            ref_path,
            flt_path,
            ref_no_data=ref_no_data,
            flt_no_data=flt_no_data,
        )
        ref_arr = ov["ref_arr"][0]
        flt_arr = ov["flt_arr"][0]
        ref_valid = ov["ref_valid"]
        flt_valid = ov["flt_valid"]
        ov_transform = ov["ov_transform"]
        flt_win_transform = ov["flt_win_transform"]
        ov_h = ov["height"]
        ov_w = ov["width"]

        ref_transform = ref_src.transform
        flt_transform = flt_src.transform
        ref_crs = ref_src.crs
        flt_crs = flt_src.crs
        crs_transformer = None
        if (
            not ov["same_grid"]
            and ref_crs is not None
            and flt_crs is not None
            and pyproj.CRS.from_user_input(ref_crs) != pyproj.CRS.from_user_input(flt_crs)
        ):
            crs_transformer = pyproj.Transformer.from_crs(
                pyproj.CRS.from_user_input(ref_crs),
                pyproj.CRS.from_user_input(flt_crs),
                always_xy=True,
            )

    half_win = win_size // 2
    margin = half_win + (search_area if method_key == TIE_METHOD_TEMPLATE_MATCH else 0)
    r_centers = list(range(margin, ov_h - margin, pixel_gap))
    c_centers = list(range(margin, ov_w - margin, pixel_gap))

    records: List[Dict[str, float]] = []
    grid_pts = [(r_c, c_c) for r_c in r_centers for c_c in c_centers]
    tqdm_cls = rsgispy.get_tqdm()
    for r_c, c_c in tqdm_cls(grid_pts, total=len(grid_pts), disable=not use_pbar):
        r0 = r_c - half_win
        r1 = r0 + win_size
        c0 = c_c - half_win
        c1 = c0 + win_size

        ref_p = ref_arr[r0:r1, c0:c1]
        ref_m = ref_valid[r0:r1, c0:c1]
        if float(np.mean(ref_m)) < 0.8:
            continue
        if float(np.std(ref_p[ref_m])) < sd_ref_thres:
            continue

        if method_key == TIE_METHOD_PHASE_CORR:
            flt_p = flt_arr[r0:r1, c0:c1]
            flt_m = flt_valid[r0:r1, c0:c1]
            if float(np.mean(flt_m)) < 0.8:
                continue
            if float(np.std(flt_p[flt_m])) < sd_flt_thres:
                continue
            r_fill = ref_p.copy()
            f_fill = flt_p.copy()
            r_fill[~ref_m] = float(np.mean(r_fill[ref_m]))
            f_fill[~flt_m] = float(np.mean(f_fill[flt_m]))
            dy, dx, score = phase_cross_correlation_2d(
                r_fill, f_fill, upsample_factor=sub_pxl_res
            )
            if search_area > 0 and (abs(dy) > search_area or abs(dx) > search_area):
                continue
            if score < thres_val:
                continue
        else:
            sr0 = r0 - search_area
            sr1 = r1 + search_area
            sc0 = c0 - search_area
            sc1 = c1 + search_area
            flt_s = flt_arr[sr0:sr1, sc0:sc1]
            flt_sm = flt_valid[sr0:sr1, sc0:sc1]
            if float(np.mean(flt_sm)) < 0.8:
                continue
            if float(np.std(flt_s[flt_sm])) < sd_flt_thres:
                continue
            matched = _match_template_patch(
                ref_p,
                flt_s,
                ref_m,
                flt_sm,
                search_area=search_area,
                metric_id=metric_id,
                maximise=maximise,
                sub_pxl_res=sub_pxl_res,
                hist_bins=hist_bins,
            )
            if matched is None:
                continue
            dy, dx, score = matched
            if maximise and score < thres_val:
                continue
            if not maximise and score > thres_val:
                continue

        ref_x, ref_y = ov_transform @ (float(c_c) + 0.5, float(r_c) + 0.5)
        ref_pxl_x, ref_pxl_y = ~ref_transform @ (ref_x, ref_y)

        flt_c_ov = float(c_c) - dx
        flt_r_ov = float(r_c) - dy
        flt_x, flt_y = flt_win_transform @ (flt_c_ov + 0.5, flt_r_ov + 0.5)
        if crs_transformer is not None:
            fx_crs, fy_crs = crs_transformer.transform(flt_x, flt_y)
            flt_pxl_x, flt_pxl_y = ~flt_transform @ (fx_crs, fy_crs)
        else:
            flt_pxl_x, flt_pxl_y = ~flt_transform @ (flt_x, flt_y)

        dx_map = float(ref_x - flt_x)
        dy_map = float(ref_y - flt_y)
        records.append(
            {
                "ref_x": float(ref_x),
                "ref_y": float(ref_y),
                "flt_x": float(flt_x),
                "flt_y": float(flt_y),
                "ref_pxl_x": float(ref_pxl_x),
                "ref_pxl_y": float(ref_pxl_y),
                "flt_pxl_x": float(flt_pxl_x),
                "flt_pxl_y": float(flt_pxl_y),
                "dx_map": dx_map,
                "dy_map": dy_map,
                "dx_pxl": float(dx_map / abs(ov_transform.a)),
                "dy_pxl": float(dy_map / abs(ov_transform.e)),
                "metric": float(score),
            }
        )

    tie_df = (
        pd.DataFrame(records, columns=_TIE_POINT_COLUMNS)
        if records
        else pd.DataFrame({c: pd.Series(dtype="float64") for c in _TIE_POINT_COLUMNS})
    )
    if out_gcp_file is not None:
        write_tie_points_file(
            tie_df, out_gcp_file, output_type=output_type, crs=ref_crs
        )
        return None
    return tie_df


def basic_registration(
    in_ref_img: str,
    in_float_img: str,
    out_gcp_file: Optional[Union[str, os.PathLike]],
    pixel_gap: int,
    threshold: float,
    win_size: int,
    search_area: int,
    sd_ref_thres: float,
    sd_flt_thres: float,
    sub_pxl_res: int,
    metric_type: Union[int, str] = rsgispy.METRIC_CORRELATION,
    output_type: int = TYPE_RSGIS_IMG2MAP,
    ref_band: int = 1,
    flt_band: int = 1,
    use_pbar: bool = True,
) -> Optional[pd.DataFrame]:
    """
    Generate tie points between ``in_float_img`` and ``in_ref_img`` using
    sliding-window template matching (RSGISLib-compatible API).

    :param in_ref_img: Path to the reference raster image.
    :param in_float_img: Path to the floating raster image.
    :param out_gcp_file: Output tie-point file path (or ``None`` to return a
                         :class:`pandas.DataFrame`).
    :param pixel_gap: Grid spacing in pixels between candidate tie points.
    :param threshold: Match score acceptance threshold.
    :param win_size: Window size in pixels around each tie point.
    :param search_area: Search radius in pixels around each candidate point.
    :param sd_ref_thres: Minimum reference window standard deviation.
    :param sd_flt_thres: Minimum floating window standard deviation.
    :param sub_pxl_res: Sub-pixel refinement factor (e.g., ``4`` -> ``0.25`` px).
    :param metric_type: Similarity metric constant ``rsgispy.METRIC_*`` (Default =
                        :data:`rsgispy.METRIC_CORRELATION`).
    :param output_type: Output format constant ``TYPE_*`` (Default =
                        :data:`TYPE_RSGIS_IMG2MAP`).
    :param ref_band: 1-based band index in ``in_ref_img`` (Default = ``1``).
    :param flt_band: 1-based band index in ``in_float_img`` (Default = ``1``).
    :param use_pbar: Display tqdm progress bar (Default = ``True``).
    :return: Tie-point :class:`pandas.DataFrame` if ``out_gcp_file`` is
             ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy
        import rsgispy.imageregistration as imreg
        import rsgispy.imageregistration.tiepoints as imreg_tp

        imreg_tp.basic_registration(
            in_ref_img="ref.tif",
            in_float_img="float.tif",
            out_gcp_file="tie_points.txt",
            pixel_gap=50,
            threshold=0.4,
            win_size=64,
            search_area=10,
            sd_ref_thres=2.0,
            sd_flt_thres=2.0,
            sub_pxl_res=4,
            metric_type=rsgispy.METRIC_CORRELATION,
            output_type=imreg.TYPE_RSGIS_IMG2MAP,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    return generate_tie_points(
        in_ref_img=in_ref_img,
        in_float_img=in_float_img,
        out_gcp_file=out_gcp_file,
        method=TIE_METHOD_TEMPLATE_MATCH,
        ref_band=ref_band,
        flt_band=flt_band,
        pixel_gap=pixel_gap,
        win_size=win_size,
        search_area=search_area,
        threshold=threshold,
        sd_ref_thres=sd_ref_thres,
        sd_flt_thres=sd_flt_thres,
        sub_pxl_res=sub_pxl_res,
        metric_type=metric_type,
        output_type=output_type,
        use_pbar=use_pbar,
    )


def _poly_design_matrix(u: np.ndarray, v: np.ndarray, model: str) -> np.ndarray:
    """Build Vandermonde design matrix for normalised coordinates ``(u, v)``."""
    ones = np.ones_like(u)
    if model in (TRANSFORM_AFFINE, TRANSFORM_POLY1):
        return np.column_stack([ones, u, v])
    if model == TRANSFORM_POLY2:
        return np.column_stack([ones, u, v, u * u, u * v, v * v])
    if model == TRANSFORM_POLY3:
        return np.column_stack(
            [
                ones,
                u,
                v,
                u * u,
                u * v,
                v * v,
                u**3,
                (u * u) * v,
                u * (v * v),
                v**3,
            ]
        )
    raise rsgispy.RSGISPyException(f"Unsupported geometric model: '{model}'.")


def fit_geometric_model(
    src_x: np.ndarray,
    src_y: np.ndarray,
    dst_x: np.ndarray,
    dst_y: np.ndarray,
    model: str = TRANSFORM_AFFINE,
) -> Dict[str, Any]:
    """
    Fit a 2D geometric transformation mapping ``(src_x, src_y)`` to
    ``(dst_x, dst_y)``.

    Supported models: :data:`TRANSFORM_SHIFT`, :data:`TRANSFORM_SIMILARITY`,
    :data:`TRANSFORM_AFFINE` (:data:`TRANSFORM_POLY1`),
    :data:`TRANSFORM_POLY2`, and :data:`TRANSFORM_POLY3`.

    :param src_x: 1D array of source X coordinates.
    :param src_y: 1D array of source Y coordinates.
    :param dst_x: 1D array of target X coordinates.
    :param dst_y: 1D array of target Y coordinates.
    :param model: Transformation model constant (Default = :data:`TRANSFORM_AFFINE`).
    :return: Dictionary representing the fitted coordinate transformation.

    .. code:: python

        import numpy as np
        import rsgispy.imageregistration as imreg
        import rsgispy.imageregistration.tiepoints as imreg_tp

        model_dict = imreg_tp.fit_geometric_model(
            src_x=np.array([0.0, 10.0, 0.0, 10.0]),
            src_y=np.array([0.0, 0.0, 10.0, 10.0]),
            dst_x=np.array([2.0, 12.0, 2.0, 12.0]),
            dst_y=np.array([3.0, 3.0, 13.0, 13.0]),
            model=imreg.TRANSFORM_AFFINE,
        )
    """
    key = model.strip().lower()
    if key not in _MODEL_MIN_SAMPLES:
        raise rsgispy.RSGISPyException(
            f"Unsupported geometric model '{model}'. Supported: "
            f"{sorted(_MODEL_MIN_SAMPLES)}."
        )
    sx_arr = np.asarray(src_x, dtype=np.float64)
    sy_arr = np.asarray(src_y, dtype=np.float64)
    dx_arr = np.asarray(dst_x, dtype=np.float64)
    dy_arr = np.asarray(dst_y, dtype=np.float64)

    min_req = _MODEL_MIN_SAMPLES[key]
    if sx_arr.size < min_req:
        raise rsgispy.RSGISPyException(
            f"Model '{key}' requires at least {min_req} tie points (got {sx_arr.size})."
        )

    if key in (TRANSFORM_SHIFT, "translation"):
        return {
            "model": TRANSFORM_SHIFT,
            "tx": float(np.mean(dx_arr - sx_arr)),
            "ty": float(np.mean(dy_arr - sy_arr)),
        }

    cx = float(np.mean(sx_arr))
    cy = float(np.mean(sy_arr))
    scale = float(max(np.std(sx_arr), np.std(sy_arr), 1.0))
    u = (sx_arr - cx) / scale
    v = (sy_arr - cy) / scale

    if key == TRANSFORM_SIMILARITY:
        n = sx_arr.size
        a_mat = np.zeros((2 * n, 4), dtype=np.float64)
        a_mat[:n, 0] = 1.0
        a_mat[:n, 2] = u
        a_mat[:n, 3] = -v
        a_mat[n:, 1] = 1.0
        a_mat[n:, 2] = v
        a_mat[n:, 3] = u
        b_vec = np.concatenate([dx_arr, dy_arr])
        params, _, _, _ = np.linalg.lstsq(a_mat, b_vec, rcond=None)
        return {
            "model": TRANSFORM_SIMILARITY,
            "cx": cx,
            "cy": cy,
            "scale": scale,
            "params": params,
        }

    design = _poly_design_matrix(u, v, key)
    coef_x, _, _, _ = np.linalg.lstsq(design, dx_arr, rcond=None)
    coef_y, _, _, _ = np.linalg.lstsq(design, dy_arr, rcond=None)
    return {
        "model": key,
        "cx": cx,
        "cy": cy,
        "scale": scale,
        "coef_x": coef_x,
        "coef_y": coef_y,
    }


def evaluate_geometric_model(
    model_dict: Dict[str, Any],
    src_x: np.ndarray,
    src_y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate a fitted model from :func:`fit_geometric_model` on ``(src_x, src_y)``.

    :param model_dict: Model dictionary returned by :func:`fit_geometric_model`.
    :param src_x: 1D array of source X coordinates.
    :param src_y: 1D array of source Y coordinates.
    :return: Tuple of predicted ``(pred_x, pred_y)`` arrays.
    """
    sx_arr = np.asarray(src_x, dtype=np.float64)
    sy_arr = np.asarray(src_y, dtype=np.float64)
    key = model_dict["model"]

    if key == TRANSFORM_SHIFT:
        return sx_arr + model_dict["tx"], sy_arr + model_dict["ty"]

    u = (sx_arr - model_dict["cx"]) / model_dict["scale"]
    v = (sy_arr - model_dict["cy"]) / model_dict["scale"]

    if key == TRANSFORM_SIMILARITY:
        tx, ty, a_p, b_p = model_dict["params"]
        return tx + a_p * u - b_p * v, ty + b_p * u + a_p * v

    design = _poly_design_matrix(u, v, key)
    return design @ model_dict["coef_x"], design @ model_dict["coef_y"]


def _resolve_coord_columns(
    tie_df: pd.DataFrame,
    coord_space: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract source and target coordinate arrays for outlier filtering."""
    space_key = coord_space.strip().lower()
    if space_key == "map":
        if {"flt_x", "flt_y", "ref_x", "ref_y"}.issubset(tie_df.columns):
            return (
                tie_df["flt_x"].to_numpy(dtype=np.float64),
                tie_df["flt_y"].to_numpy(dtype=np.float64),
                tie_df["ref_x"].to_numpy(dtype=np.float64),
                tie_df["ref_y"].to_numpy(dtype=np.float64),
            )
        if {"flt_pxl_x", "flt_pxl_y", "ref_x", "ref_y"}.issubset(tie_df.columns):
            return (
                tie_df["flt_pxl_x"].to_numpy(dtype=np.float64),
                tie_df["flt_pxl_y"].to_numpy(dtype=np.float64),
                tie_df["ref_x"].to_numpy(dtype=np.float64),
                tie_df["ref_y"].to_numpy(dtype=np.float64),
            )
        raise rsgispy.RSGISPyException(
            "Map-space filtering requires 'flt_x', 'flt_y', 'ref_x', 'ref_y' "
            "(or 'flt_pxl_x', 'flt_pxl_y', 'ref_x', 'ref_y') columns."
        )
    if space_key == "pixel":
        if {"flt_pxl_x", "flt_pxl_y", "ref_pxl_x", "ref_pxl_y"}.issubset(
            tie_df.columns
        ):
            return (
                tie_df["flt_pxl_x"].to_numpy(dtype=np.float64),
                tie_df["flt_pxl_y"].to_numpy(dtype=np.float64),
                tie_df["ref_pxl_x"].to_numpy(dtype=np.float64),
                tie_df["ref_pxl_y"].to_numpy(dtype=np.float64),
            )
        if {"flt_pxl_x", "flt_pxl_y", "ref_x", "ref_y"}.issubset(tie_df.columns):
            return (
                tie_df["flt_pxl_x"].to_numpy(dtype=np.float64),
                tie_df["flt_pxl_y"].to_numpy(dtype=np.float64),
                tie_df["ref_x"].to_numpy(dtype=np.float64),
                tie_df["ref_y"].to_numpy(dtype=np.float64),
            )
        raise rsgispy.RSGISPyException(
            "Pixel-space filtering requires 'flt_pxl_x', 'flt_pxl_y', "
            "'ref_pxl_x', and 'ref_pxl_y' columns."
        )
    raise rsgispy.RSGISPyException(
        f"Unsupported coord_space '{coord_space}'. Use 'map' or 'pixel'."
    )


def filter_tie_points_outliers(
    tie_points: Union[pd.DataFrame, gpd.GeoDataFrame, str, os.PathLike],
    out_gcp_file: Optional[Union[str, os.PathLike]] = None,
    method: str = OUTLIER_METHOD_RANSAC,
    model: str = TRANSFORM_AFFINE,
    residual_thres: float = 2.0,
    coord_space: str = "map",
    max_trials: int = 1000,
    min_samples: Optional[int] = None,
    mad_scale: float = 3.0,
    output_type: int = TYPE_RSGIS_IMG2MAP,
    rnd_seed: Optional[int] = None,
    return_model: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Any]], None]:
    """
    Filter outlier tie points using RANSAC consensus or iterative residual
    thresholding.

    :param tie_points: Input tie-point :class:`pandas.DataFrame`,
                       :class:`geopandas.GeoDataFrame`, or file path.
    :param out_gcp_file: Optional output file path (``None`` returns DataFrame).
    :param method: Filtering method: :data:`OUTLIER_METHOD_RANSAC` (default),
                   :data:`OUTLIER_METHOD_RESIDUAL`, or
                   :data:`OUTLIER_METHOD_MAD` (robust MAD-based residual thresholding).
    :param model: Geometric model to fit: :data:`TRANSFORM_SHIFT`,
                  :data:`TRANSFORM_SIMILARITY`, :data:`TRANSFORM_AFFINE`
                  (:data:`TRANSFORM_POLY1`), :data:`TRANSFORM_POLY2`, or
                  :data:`TRANSFORM_POLY3`.
    :param residual_thres: Maximum Euclidean residual in ``coord_space`` units
                           to accept a tie point as an inlier (Default = ``2.0``).
    :param coord_space: Coordinate space for fitting and residuals: ``'map'``
                        (default) or ``'pixel'``.
    :param max_trials: Maximum RANSAC sampling iterations (Default = ``1000``).
    :param min_samples: Optional override for RANSAC sample size per trial.
    :param mad_scale: MAD multiplier when ``method=OUTLIER_METHOD_MAD``
                      (Default = ``3.0``).
    :param output_type: Output format constant ``TYPE_*`` when writing a plain
                        text GCP file (Default = :data:`TYPE_RSGIS_IMG2MAP`).
    :param rnd_seed: Optional RNG seed for reproducible RANSAC sampling.
    :param return_model: If ``True`` and ``out_gcp_file`` is ``None``, return
                         ``(filtered_df, diagnostics_dict)``.
    :return: Filtered :class:`pandas.DataFrame` (or tuple when
             ``return_model=True``), or ``None`` if ``out_gcp_file`` is set.

    .. code:: python

        import rsgispy.imageregistration as imreg
        import rsgispy.imageregistration.tiepoints as imreg_tp

        inliers_df = imreg_tp.filter_tie_points_outliers(
            tie_points=tie_df,
            method=imreg.OUTLIER_METHOD_RANSAC,
            model=imreg.TRANSFORM_AFFINE,
            residual_thres=1.5,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-branches
    src_crs = getattr(tie_points, "crs", None)
    tie_df = load_tie_points_table(tie_points)
    if src_crs is None:
        src_crs = getattr(tie_df, "crs", None)

    if len(tie_df) == 0:
        raise rsgispy.RSGISPyException("Input tie_points table is empty.")

    method_key = method.strip().lower()
    if method_key not in (
        OUTLIER_METHOD_RANSAC,
        OUTLIER_METHOD_RESIDUAL,
        OUTLIER_METHOD_MAD,
    ):
        raise rsgispy.RSGISPyException(
            f"Unsupported outlier filtering method '{method}'. Choose "
            f"'{OUTLIER_METHOD_RANSAC}', '{OUTLIER_METHOD_RESIDUAL}', "
            f"or '{OUTLIER_METHOD_MAD}'."
        )
    model_key = model.strip().lower()
    if model_key not in _MODEL_MIN_SAMPLES:
        raise rsgispy.RSGISPyException(
            f"Unsupported geometric model '{model}'."
        )
    if residual_thres <= 0:
        raise rsgispy.RSGISPyException("residual_thres must be > 0.")

    src_x, src_y, dst_x, dst_y = _resolve_coord_columns(tie_df, coord_space)
    n_pts = src_x.size
    req_samples = (
        int(min_samples) if min_samples is not None else _MODEL_MIN_SAMPLES[model_key]
    )
    if req_samples < _MODEL_MIN_SAMPLES[model_key]:
        raise rsgispy.RSGISPyException(
            f"min_samples ({req_samples}) must be >= {_MODEL_MIN_SAMPLES[model_key]} "
            f"for model '{model_key}'."
        )
    if n_pts < req_samples:
        raise rsgispy.RSGISPyException(
            f"Not enough tie points ({n_pts}) for model '{model_key}' "
            f"(requires >= {req_samples})."
        )

    inlier_mask = np.ones(n_pts, dtype=bool)
    if method_key == OUTLIER_METHOD_RANSAC:
        rng = np.random.default_rng(rnd_seed)
        best_count = -1
        best_rmse = np.inf
        best_mask: Optional[np.ndarray] = None

        if n_pts == req_samples:
            best_mask = np.ones(n_pts, dtype=bool)
        else:
            for _ in range(max(1, int(max_trials))):
                idx = rng.choice(n_pts, size=req_samples, replace=False)
                try:
                    cand_model = fit_geometric_model(
                        src_x[idx], src_y[idx], dst_x[idx], dst_y[idx], model=model_key
                    )
                except np.linalg.LinAlgError:
                    continue
                px, py = evaluate_geometric_model(cand_model, src_x, src_y)
                res = np.hypot(px - dst_x, py - dst_y)
                cand_mask = res <= residual_thres
                cnt = int(np.count_nonzero(cand_mask))
                if cnt < req_samples:
                    continue
                rmse = float(np.sqrt(np.mean(res[cand_mask] ** 2)))
                if cnt > best_count or (cnt == best_count and rmse < best_rmse):
                    best_count = cnt
                    best_rmse = rmse
                    best_mask = cand_mask

        if best_mask is None:
            raise rsgispy.RSGISPyException(
                "RANSAC failed to find a consensus set meeting residual_thres."
            )
        inlier_mask = best_mask
    else:
        while int(np.count_nonzero(inlier_mask)) > req_samples:
            cur_idx = np.nonzero(inlier_mask)[0]
            cur_model = fit_geometric_model(
                src_x[cur_idx],
                src_y[cur_idx],
                dst_x[cur_idx],
                dst_y[cur_idx],
                model=model_key,
            )
            px, py = evaluate_geometric_model(cur_model, src_x[cur_idx], src_y[cur_idx])
            res = np.hypot(px - dst_x[cur_idx], py - dst_y[cur_idx])
            if method_key == OUTLIER_METHOD_MAD:
                med = float(np.median(res))
                mad = float(np.median(np.abs(res - med))) * 1.4826
                cut = min(residual_thres, med + mad_scale * max(mad, 1e-6))
            else:
                cut = residual_thres
            worst_pos = int(np.argmax(res))
            if res[worst_pos] <= cut:
                break
            inlier_mask[cur_idx[worst_pos]] = False

    final_model = fit_geometric_model(
        src_x[inlier_mask],
        src_y[inlier_mask],
        dst_x[inlier_mask],
        dst_y[inlier_mask],
        model=model_key,
    )
    px_all, py_all = evaluate_geometric_model(
        final_model, src_x[inlier_mask], src_y[inlier_mask]
    )
    inlier_res = np.hypot(px_all - dst_x[inlier_mask], py_all - dst_y[inlier_mask])

    filtered_df = tie_df.loc[inlier_mask].copy().reset_index(drop=True)
    filtered_df["residual"] = inlier_res

    diagnostics = {
        "model": final_model,
        "n_input": int(n_pts),
        "n_inliers": int(np.count_nonzero(inlier_mask)),
        "n_outliers": int(n_pts - np.count_nonzero(inlier_mask)),
        "rmse": float(np.sqrt(np.mean(inlier_res**2))) if inlier_res.size else 0.0,
        "max_residual": float(np.max(inlier_res)) if inlier_res.size else 0.0,
    }

    if out_gcp_file is not None:
        write_tie_points_file(
            filtered_df, out_gcp_file, output_type=output_type, crs=src_crs
        )
        return None
    if return_model:
        return filtered_df, diagnostics
    return filtered_df
