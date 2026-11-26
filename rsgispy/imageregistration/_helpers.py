"""
Internal shared image-registration surface search, sub-pixel refinement,
and tie-point I/O helpers for :mod:`rsgispy.imageregistration`.
"""

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.control import GroundControlPoint
from rasterio.transform import Affine

import rsgispy
from rsgispy._numba import use_compiled
from rsgispy.imageprocess import (
    StagedOutput,
    ensure_file_exists,
    is_cog_layout,
    set_dataset_nodata,
    update_cog,
)
from rsgispy.imageregistration import (
    TYPE_ENVI_IMG2IMG,
    TYPE_ENVI_IMG2MAP,
    TYPE_RSGIS_IMG2MAP,
    TYPE_RSGIS_MAPOFFS,
)
from rsgispy.imageregistration._kernels import (
    SUPPORTED_KERNEL_METRICS,
    masked_values_finite,
    shift_surface_kernel,
)
from rsgispy.tools.stats import calc_similarity_metric
from rsgispy.tools.tables import _TABLE_EXTS, write_dataframe_table
from rsgispy.vectorprocess import _VECTOR_EXTS, is_geoparquet_file, write_vec_gdf


def find_surface_extremum(
    surface: np.ndarray,
    maximise: bool,
) -> Tuple[int, int]:
    """Return ``(iy, ix)`` of the finite maximum or minimum in ``surface``."""
    if maximise:
        filled = np.where(np.isfinite(surface), surface, -np.inf)
        iy, ix = np.unravel_index(np.argmax(filled), surface.shape)
    else:
        filled = np.where(np.isfinite(surface), surface, np.inf)
        iy, ix = np.unravel_index(np.argmin(filled), surface.shape)
    return int(iy), int(ix)


def _quadratic_subpixel_offset(
    v_minus: float,
    v_zero: float,
    v_plus: float,
) -> float:
    """1D quadratic extremum offset clamped to ``[-0.5, 0.5]``."""
    denom = v_minus - 2.0 * v_zero + v_plus
    if abs(denom) <= 1e-12:
        return 0.0
    delta = 0.5 * (v_minus - v_plus) / denom
    return float(max(-0.5, min(0.5, delta)))


def refine_grid_subpixel(
    surface: np.ndarray,
    iy: int,
    ix: int,
    sub_pxl_res: int,
    maximise: bool,
) -> Tuple[float, float, float]:
    """
    Refine integer peak/trough ``(iy, ix)`` on a 2D metric surface using
    2D parabolic interpolation quantised to ``1 / sub_pxl_res``.
    """
    ny, nx = surface.shape
    val = float(surface[iy, ix])
    if sub_pxl_res <= 1:
        return float(iy), float(ix), val

    sy_off = 0.0
    sx_off = 0.0
    if 0 < iy < ny - 1:
        vm, v0, vp = float(surface[iy - 1, ix]), val, float(surface[iy + 1, ix])
        if np.isfinite(vm) and np.isfinite(v0) and np.isfinite(vp):
            if (maximise and v0 >= vm and v0 >= vp) or (
                not maximise and v0 <= vm and v0 <= vp
            ):
                sy_off = _quadratic_subpixel_offset(vm, v0, vp)
    if 0 < ix < nx - 1:
        vm, v0, vp = float(surface[iy, ix - 1]), val, float(surface[iy, ix + 1])
        if np.isfinite(vm) and np.isfinite(v0) and np.isfinite(vp):
            if (maximise and v0 >= vm and v0 >= vp) or (
                not maximise and v0 <= vm and v0 <= vp
            ):
                sx_off = _quadratic_subpixel_offset(vm, v0, vp)

    step = float(max(1, int(sub_pxl_res)))
    sy_off = round(sy_off * step) / step
    sx_off = round(sx_off * step) / step
    return float(iy) + sy_off, float(ix) + sx_off, val


def search_shift_surface(
    ref_bands_arr: np.ndarray,
    flt_bands_arr: np.ndarray,
    ref_valid: np.ndarray,
    flt_valid: np.ndarray,
    metric_id: int,
    maximise: bool,
    x_search: int,
    y_search: int,
    hist_bins: int = 64,
    min_valid_pxls: int = 16,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate a similarity/distance surface over integer pixel shifts
    ``sy in [-y_search, y_search]``, ``sx in [-x_search, x_search]`` where
    ``flt`` shifted by ``(-sy, -sx)`` aligns with ``ref``.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    if (
        int(metric_id) in SUPPORTED_KERNEL_METRICS
        and use_compiled(
            shift_surface_kernel,
            "search_shift_surface",
            fallback_desc="the NumPy similarity search",
        )
        and masked_values_finite(ref_bands_arr, ref_valid)
        and masked_values_finite(flt_bands_arr, flt_valid)
    ):
        surface, counts = shift_surface_kernel(
            np.ascontiguousarray(ref_bands_arr, dtype=np.float64),
            np.ascontiguousarray(flt_bands_arr, dtype=np.float64),
            np.ascontiguousarray(ref_valid, dtype=np.bool_),
            np.ascontiguousarray(flt_valid, dtype=np.bool_),
            int(y_search),
            int(x_search),
            int(metric_id),
            int(hist_bins),
            int(min_valid_pxls),
        )
        if not np.any(np.isfinite(surface)):
            raise rsgispy.RSGISPyException(
                "Insufficient overlapping valid pixels to evaluate image shift."
            )
        return surface, counts

    sy_vals = np.arange(-int(y_search), int(y_search) + 1)
    sx_vals = np.arange(-int(x_search), int(x_search) + 1)
    h, w = ref_valid.shape
    n_bands = ref_bands_arr.shape[0]
    surface = np.full((sy_vals.size, sx_vals.size), np.nan, dtype=np.float64)
    counts = np.zeros((sy_vals.size, sx_vals.size), dtype=np.int64)

    for yi, sy in enumerate(sy_vals):
        # Match ref[r, c] with flt[r - sy, c - sx]
        r_ref_0 = max(0, sy)
        r_ref_1 = min(h, h + sy)
        r_flt_0 = r_ref_0 - sy
        r_flt_1 = r_ref_1 - sy
        if r_ref_1 <= r_ref_0:
            continue

        for xi, sx in enumerate(sx_vals):
            c_ref_0 = max(0, sx)
            c_ref_1 = min(w, w + sx)
            c_flt_0 = c_ref_0 - sx
            c_flt_1 = c_ref_1 - sx
            if c_ref_1 <= c_ref_0:
                continue

            mask = (
                ref_valid[r_ref_0:r_ref_1, c_ref_0:c_ref_1]
                & flt_valid[r_flt_0:r_flt_1, c_flt_0:c_flt_1]
            )
            n_val = int(np.count_nonzero(mask))
            if n_val < min_valid_pxls:
                continue

            band_scores: List[float] = []
            for b in range(n_bands):
                r_sub = ref_bands_arr[b, r_ref_0:r_ref_1, c_ref_0:c_ref_1][mask]
                f_sub = flt_bands_arr[b, r_flt_0:r_flt_1, c_flt_0:c_flt_1][mask]
                val = calc_similarity_metric(
                    r_sub, f_sub, metric_id=metric_id, hist_bins=hist_bins
                )
                if np.isfinite(val):
                    band_scores.append(val)
            if band_scores:
                surface[yi, xi] = float(np.mean(band_scores))
                counts[yi, xi] = n_val

    if not np.any(np.isfinite(surface)):
        raise rsgispy.RSGISPyException(
            "Insufficient overlapping valid pixels to evaluate image shift."
        )
    _ = maximise
    return surface, counts


def load_tie_points_table(
    tie_points: Union[pd.DataFrame, gpd.GeoDataFrame, str, os.PathLike],
) -> pd.DataFrame:
    """
    Normalise a tie-point DataFrame, GeoDataFrame, or file path into a
    non-empty :class:`pandas.DataFrame` copy.
    """
    if isinstance(tie_points, (str, os.PathLike)):
        tie_df = read_tie_points_file(tie_points)
    elif isinstance(tie_points, gpd.GeoDataFrame):
        tie_df = pd.DataFrame(tie_points.drop(columns=["geometry"], errors="ignore"))
    else:
        tie_df = tie_points.copy()

    if len(tie_df) == 0:
        raise rsgispy.RSGISPyException("Input tie_points table is empty.")
    return tie_df


def read_tie_points_file(in_gcp_file: Union[str, os.PathLike]) -> pd.DataFrame:
    """
    Read tie points from a CSV, Parquet, vector point file, or RSGIS/ENVI GCP
    text file into a standardised :class:`pandas.DataFrame`.
    """
    # pylint: disable=too-many-locals,too-many-branches
    path_str = os.fspath(in_gcp_file)
    ensure_file_exists(path_str, label="Tie-point file")
    lower_path = path_str.lower().strip()

    if is_geoparquet_file(path_str):
        return pd.read_parquet(path_str)
    if lower_path.endswith(".csv"):
        return pd.read_csv(path_str)
    if lower_path.endswith(".tsv"):
        return pd.read_csv(path_str, sep="\t")
    if lower_path.endswith(_VECTOR_EXTS):
        gdf = gpd.read_file(path_str)
        return pd.DataFrame(gdf.drop(columns=["geometry"], errors="ignore"))

    rows: List[Dict[str, float]] = []
    file_mode: Optional[str] = None
    with open(path_str, "r", encoding="utf-8") as fp:
        for raw_line in fp:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("#") or line.startswith(";"):
                low = line.lower()
                if "envi image to image" in low:
                    file_mode = "envi_img2img"
                elif "envi registration" in low:
                    file_mode = "envi_img2map"
                elif "rsgis_mapoffs" in low:
                    file_mode = "rsgis_mapoffs"
                elif "rsgis_img2map" in low:
                    file_mode = "rsgis_img2map"
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 4:
                continue
            vals = [float(p) for p in parts[:5]]
            if file_mode == "envi_img2img":
                rows.append(
                    {
                        "ref_pxl_x": vals[0],
                        "ref_pxl_y": vals[1],
                        "flt_pxl_x": vals[2],
                        "flt_pxl_y": vals[3],
                        "metric": vals[4] if len(vals) > 4 else 1.0,
                    }
                )
            elif file_mode == "rsgis_mapoffs":
                rows.append(
                    {
                        "ref_x": vals[0],
                        "ref_y": vals[1],
                        "dx_map": vals[2],
                        "dy_map": vals[3],
                        "flt_x": vals[0] - vals[2],
                        "flt_y": vals[1] - vals[3],
                        "metric": vals[4] if len(vals) > 4 else 1.0,
                    }
                )
            else:
                rows.append(
                    {
                        "ref_x": vals[0],
                        "ref_y": vals[1],
                        "flt_pxl_x": vals[2],
                        "flt_pxl_y": vals[3],
                        "metric": vals[4] if len(vals) > 4 else 1.0,
                    }
                )
    if not rows:
        raise rsgispy.RSGISPyException(
            f"No tie points found in '{path_str}'."
        )
    return pd.DataFrame(rows)


def write_tie_points_file(
    tie_df: pd.DataFrame,
    out_gcp_file: Union[str, os.PathLike],
    output_type: int = TYPE_RSGIS_IMG2MAP,
    crs: Any = None,
) -> None:
    """
    Write a tie-point :class:`pandas.DataFrame` to a tabular file (CSV, Parquet,
    etc.), vector point file, or RSGIS/ENVI GCP text file.
    """
    path_str = os.fspath(out_gcp_file)
    lower_path = path_str.lower().strip()

    if is_geoparquet_file(path_str) or lower_path.endswith(_TABLE_EXTS):
        write_dataframe_table(tie_df, path_str)
        return

    if lower_path.endswith(_VECTOR_EXTS):
        x_col = "flt_x" if "flt_x" in tie_df.columns else "ref_x"
        y_col = "flt_y" if "flt_y" in tie_df.columns else "ref_y"
        gdf = gpd.GeoDataFrame(
            tie_df.copy(),
            geometry=gpd.points_from_xy(tie_df[x_col], tie_df[y_col]),
            crs=crs,
        )
        write_vec_gdf(gdf, path_str)
        return

    out_dir = os.path.dirname(os.path.abspath(path_str))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(path_str, "w", encoding="utf-8") as fp:
        if output_type == TYPE_ENVI_IMG2IMG:
            fp.write("; ENVI Image to Image GCP File\n")
            fp.write("; Base X, Base Y, Warp X, Warp Y\n")
            for _, row in tie_df.iterrows():
                fp.write(
                    f"{row['ref_pxl_x']:.6f} {row['ref_pxl_y']:.6f} "
                    f"{row['flt_pxl_x']:.6f} {row['flt_pxl_y']:.6f}\n"
                )
        elif output_type == TYPE_ENVI_IMG2MAP:
            fp.write("; ENVI Registration GCP File\n")
            fp.write("; Map X, Map Y, Image X, Image Y\n")
            for _, row in tie_df.iterrows():
                fp.write(
                    f"{row['ref_x']:.6f} {row['ref_y']:.6f} "
                    f"{row['flt_pxl_x']:.6f} {row['flt_pxl_y']:.6f}\n"
                )
        elif output_type == TYPE_RSGIS_MAPOFFS:
            fp.write("#RSGIS_MAPOFFS\n")
            fp.write("#ref_x ref_y dx_map dy_map metric\n")
            for _, row in tie_df.iterrows():
                fp.write(
                    f"{row['ref_x']:.6f} {row['ref_y']:.6f} "
                    f"{row['dx_map']:.6f} {row['dy_map']:.6f} "
                    f"{row.get('metric', 1.0):.6f}\n"
                )
        else:
            fp.write("#RSGIS_IMG2MAP\n")
            fp.write("#ref_x ref_y flt_pxl_x flt_pxl_y metric\n")
            for _, row in tie_df.iterrows():
                fp.write(
                    f"{row['ref_x']:.6f} {row['ref_y']:.6f} "
                    f"{row['flt_pxl_x']:.6f} {row['flt_pxl_y']:.6f} "
                    f"{row.get('metric', 1.0):.6f}\n"
                )


def df_to_gcps(tie_df: pd.DataFrame) -> List[GroundControlPoint]:
    """Convert tie-point DataFrame rows into ``rasterio`` ``GroundControlPoint`` objects."""
    if not {"ref_x", "ref_y", "flt_pxl_x", "flt_pxl_y"}.issubset(tie_df.columns):
        raise rsgispy.RSGISPyException(
            "Tie-point table must contain 'ref_x', 'ref_y', 'flt_pxl_x', and 'flt_pxl_y'."
        )
    gcps: List[GroundControlPoint] = []
    has_z = "ref_z" in tie_df.columns
    for idx, row in tie_df.reset_index(drop=True).iterrows():
        gcps.append(
            GroundControlPoint(
                row=float(row["flt_pxl_y"]),
                col=float(row["flt_pxl_x"]),
                x=float(row["ref_x"]),
                y=float(row["ref_y"]),
                z=float(row["ref_z"]) if has_z else 0.0,
                id=str(idx + 1),
            )
        )
    return gcps


def copy_raster_with_updates(
    in_path: str,
    out_path: str,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    new_transform: Optional[Affine] = None,
    gcps: Optional[Tuple[Any, Any]] = None,
    no_data_val: Optional[float] = None,
) -> Optional[float]:
    """
    Stream-copy ``in_path`` to ``out_path`` with optional updated transform,
    output datatype, nodata value, or embedded GCPs. Returns the output
    nodata value. The output uses the creation options of
    :func:`rsgispy.imageprocess.get_gdal_img_creation_opts` for ``gdalformat``;
    a COG output is written through a temporary GeoTIFF
    (:class:`rsgispy.imageprocess.StagedOutput`), and an existing COG that
    only gets GCPs is rebuilt (:func:`rsgispy.imageprocess.update_cog`).
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    if (
        os.path.abspath(in_path) == os.path.abspath(out_path)
        and datatype is None
        and new_transform is None
        and gcps is not None
    ):

        def _set_gcps(img_path: str) -> None:
            with rasterio.open(img_path, "r+") as dst:
                dst.gcps = gcps
                if no_data_val is not None:
                    set_dataset_nodata(dst, no_data_val)

        if is_cog_layout(out_path):
            update_cog(out_path, _set_gcps)
        else:
            _set_gcps(out_path)
        with rasterio.open(out_path, "r") as out_src:
            return out_src.nodata

    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with StagedOutput(out_path, gdalformat) as stage, rasterio.open(
        in_path, "r"
    ) as src:
        out_dtype = (
            rsgispy.get_rasterio_datatype(datatype)
            if datatype is not None
            else src.dtypes[0]
        )
        np_dtype = np.dtype(out_dtype)
        # The output layout (tiling, compression, interleave) comes from the
        # resolved creation options for ``gdalformat``, not from the source.
        profile = {
            key: val
            for key, val in src.profile.items()
            if key
            not in ("tiled", "blockxsize", "blockysize", "compress", "interleave")
        }
        profile.update(
            driver=stage.write_format, dtype=out_dtype, **stage.write_opts
        )
        if new_transform is not None:
            profile["transform"] = new_transform
        if no_data_val is not None:
            profile["nodata"] = no_data_val

        with rasterio.open(stage.path, "w", **profile) as dst:
            for _, win in src.block_windows(1):
                arr = src.read(window=win)
                if arr.dtype != np_dtype:
                    arr = arr.astype(np_dtype)
                dst.write(arr, window=win)
            for b_idx in range(1, src.count + 1):
                desc = src.descriptions[b_idx - 1]
                if desc:
                    dst.set_band_description(b_idx, desc)
                tags = src.tags(b_idx)
                if tags:
                    dst.update_tags(b_idx, **tags)
                try:
                    cmap = src.colormap(b_idx)
                    if cmap:
                        dst.write_colormap(b_idx, cmap)
                except ValueError:
                    pass
            ds_tags = src.tags()
            if ds_tags:
                dst.update_tags(**ds_tags)
            if gcps is not None:
                dst.gcps = gcps
            return no_data_val if no_data_val is not None else src.nodata
