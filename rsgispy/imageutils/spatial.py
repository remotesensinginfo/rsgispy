"""
The imageutils.spatial module provides per-pixel spatial coordinate, WGS84
pixel size/area, and scattered-point gridding functions using ``rasterio``,
``pyproj``, ``geopandas``, ``numpy``, and ``scipy``.
"""

from contextlib import ExitStack
import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import geopandas as gpd
import numpy as np
import pyproj
import rasterio

try:
    from scipy.interpolate import LinearNDInterpolator
    from scipy.spatial import cKDTree

    HAVE_SCIPY = True
except ImportError:
    LinearNDInterpolator = None  # type: ignore[assignment,misc]
    cKDTree = None  # type: ignore[assignment,misc]
    HAVE_SCIPY = False

import rsgispy
from rsgispy.imageprocess import (
    cast_result_block,
    create_output_from_dataset,
    ensure_file_exists,
    iter_dataset_windows,
    pop_stats_on_close,
)
from rsgispy.imageutils._helpers import normalise_creation_options
from rsgispy.imageutils.metadata import get_img_res
from rsgispy.tools.geometry import degrees_to_metres
from rsgispy.vectorprocess import VectorInput, read_vec_gdf

__all__ = [
    "calc_wgs84_pixel_area",
    "calc_wgs84_pixel_size",
    "calc_pixel_locations",
    "calc_wgs84_pixel_locations",
    "grid_scattered_pts",
]


def _block_centre_coords(
    transform: rasterio.Affine,
    col_off: int,
    row_off: int,
    win_w: int,
    win_h: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute 2D pixel-centre ``(x_grid, y_grid)`` arrays for a block window."""
    cols = col_off + np.arange(win_w, dtype=np.float64) + 0.5
    rows = row_off + np.arange(win_h, dtype=np.float64) + 0.5
    col_grid, row_grid = np.meshgrid(cols, rows)
    x_grid = transform.c + col_grid * transform.a + row_grid * transform.b
    y_grid = transform.f + col_grid * transform.d + row_grid * transform.e
    return x_grid, y_grid


def calc_wgs84_pixel_area(
    input_img: str,
    output_img: str,
    scale: float = 10000,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
    creation_opts: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
) -> None:
    """
    Calculate the area (in square metres scaled by ``scale``) of each pixel
    assuming WGS84 geographic (degree) coordinates in ``input_img``.

    Common ``scale`` values: ``10000`` (hectares), ``1`` (sq m),
    ``1000000`` (sq km), ``4046.856`` (acres).

    :param input_img: Path to the input WGS84 raster image file.
    :param output_img: Path to the output single-band area image.
    :param scale: Divisor converting square metres to target units (Default = ``10000``).
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param datatype: Optional ``rsgispy.TYPE_*`` constant for the output image
                     (defaults to :data:`rsgispy.TYPE_64FLOAT`).
    :param no_data_val: Optional output no-data value.
    :param calc_stats: Populate image statistics and pyramids
                       (Default = ``True``).
    :param use_pbar: Show a progress bar during calculation (Default = ``True``).
    :param creation_opts: Optional GDAL creation options mapping or list.

    .. code:: python

        import rsgispy.imageutils.spatial
        rsgispy.imageutils.spatial.calc_wgs84_pixel_area(
            "wgs84_img.tif", "pixel_area_ha.tif", scale=10000
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    ensure_file_exists(input_img)
    if float(scale) <= 0.0:
        raise rsgispy.RSGISPyException("scale must be strictly positive.")

    x_res, y_res = get_img_res(input_img, abs_vals=True)
    scale_f = float(scale)
    eff_dtype = rsgispy.TYPE_64FLOAT if datatype is None else datatype
    out_rio_dt = rsgispy.get_rasterio_datatype(eff_dtype)
    out_np_dt = np.dtype(rsgispy.get_numpy_datatype(eff_dtype))
    norm_opts = normalise_creation_options(creation_opts)

    with ExitStack() as stack:
        src = stack.enter_context(rasterio.open(input_img, "r"))
        dst = create_output_from_dataset(
            stack=stack,
            src=src,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype=out_rio_dt,
            count=1,
            creation_opts=norm_opts,
        )
        if no_data_val is not None:
            dst.nodata = float(no_data_val)
        pop_stats_on_close(
            dst, calc_stats, no_data_val=no_data_val, use_pbar=use_pbar
        )
        for win in iter_dataset_windows(src.height, src.width, use_pbar=use_pbar):
            _, y_grid = _block_centre_coords(
                src.transform,
                int(win.col_off),
                int(win.row_off),
                int(win.width),
                int(win.height),
            )
            x_m, y_m = degrees_to_metres(y_grid, x_res, y_res)
            dst.write(((x_m * y_m) / scale_f).astype(out_np_dt, copy=False), 1, window=win)


def calc_wgs84_pixel_size(
    input_img: str,
    output_img: str,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
    creation_opts: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
) -> None:
    """
    Calculate the X (band 1) and Y (band 2) pixel dimensions in metres for a
    WGS84 geographic raster image.

    :param input_img: Path to the input WGS84 raster image file.
    :param output_img: Path to the output 2-band image.
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param datatype: Optional ``rsgispy.TYPE_*`` constant for the output image
                     (defaults to :data:`rsgispy.TYPE_64FLOAT`).
    :param no_data_val: Optional output no-data value.
    :param calc_stats: Populate image statistics and pyramids
                       (Default = ``True``).
    :param use_pbar: Show a progress bar during calculation (Default = ``True``).
    :param creation_opts: Optional GDAL creation options mapping or list.

    .. code:: python

        import rsgispy.imageutils.spatial
        rsgispy.imageutils.spatial.calc_wgs84_pixel_size(
            "wgs84_img.tif", "pixel_size_m.tif"
        )
    """
    # pylint: disable=too-many-arguments
    ensure_file_exists(input_img)
    x_res, y_res = get_img_res(input_img, abs_vals=True)
    eff_dtype = rsgispy.TYPE_64FLOAT if datatype is None else datatype
    out_rio_dt = rsgispy.get_rasterio_datatype(eff_dtype)
    out_np_dt = np.dtype(rsgispy.get_numpy_datatype(eff_dtype))
    norm_opts = normalise_creation_options(creation_opts)

    with ExitStack() as stack:
        src = stack.enter_context(rasterio.open(input_img, "r"))
        dst = create_output_from_dataset(
            stack=stack,
            src=src,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype=out_rio_dt,
            count=2,
            creation_opts=norm_opts,
        )
        if no_data_val is not None:
            dst.nodata = float(no_data_val)
        pop_stats_on_close(
            dst, calc_stats, no_data_val=no_data_val, use_pbar=use_pbar
        )
        for win in iter_dataset_windows(src.height, src.width, use_pbar=use_pbar):
            _, y_grid = _block_centre_coords(
                src.transform,
                int(win.col_off),
                int(win.row_off),
                int(win.width),
                int(win.height),
            )
            x_res_arr = np.full_like(y_grid, x_res, dtype=np.float64)
            y_res_arr = np.full_like(y_grid, y_res, dtype=np.float64)
            x_m, y_m = degrees_to_metres(y_grid, x_res_arr, y_res_arr)
            dst.write(
                np.stack((x_m, y_m), axis=0).astype(out_np_dt, copy=False),
                window=win,
            )


def calc_pixel_locations(
    input_img: str,
    output_img: str,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
    creation_opts: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
) -> None:
    """
    Create a 2-band raster storing pixel-centre X (band 1) and Y
    (band 2) coordinates in ``input_img``'s coordinate system.

    :param input_img: Path to the reference input image file.
    :param output_img: Path to the output 2-band coordinate image file.
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param datatype: Optional ``rsgispy.TYPE_*`` constant for the output image
                     (defaults to :data:`rsgispy.TYPE_64FLOAT`).
    :param no_data_val: Optional output no-data value.
    :param calc_stats: Populate image statistics and pyramids
                       (Default = ``True``).
    :param use_pbar: Show a progress bar during calculation (Default = ``True``).
    :param creation_opts: Optional GDAL creation options mapping or list.

    .. code:: python

        import rsgispy.imageutils.spatial
        rsgispy.imageutils.spatial.calc_pixel_locations(
            "input_img.tif", "pxl_coords.tif", gdalformat="GTIFF"
        )
    """
    # pylint: disable=too-many-arguments
    ensure_file_exists(input_img)
    eff_dtype = rsgispy.TYPE_64FLOAT if datatype is None else datatype
    out_rio_dt = rsgispy.get_rasterio_datatype(eff_dtype)
    out_np_dt = np.dtype(rsgispy.get_numpy_datatype(eff_dtype))
    norm_opts = normalise_creation_options(creation_opts)

    with ExitStack() as stack:
        src = stack.enter_context(rasterio.open(input_img, "r"))
        dst = create_output_from_dataset(
            stack=stack,
            src=src,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype=out_rio_dt,
            count=2,
            creation_opts=norm_opts,
        )
        if no_data_val is not None:
            dst.nodata = float(no_data_val)
        pop_stats_on_close(
            dst, calc_stats, no_data_val=no_data_val, use_pbar=use_pbar
        )
        for win in iter_dataset_windows(src.height, src.width, use_pbar=use_pbar):
            x_grid, y_grid = _block_centre_coords(
                src.transform,
                int(win.col_off),
                int(win.row_off),
                int(win.width),
                int(win.height),
            )
            dst.write(
                np.stack((x_grid, y_grid), axis=0).astype(out_np_dt, copy=False),
                window=win,
            )


def calc_wgs84_pixel_locations(
    input_img: str,
    output_img: str,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = 0.0,
    calc_stats: bool = True,
    use_pbar: bool = True,
    creation_opts: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
) -> None:
    """
    Create a 2-band raster storing WGS84 (EPSG:4326) longitude (band 1)
    and latitude (band 2) coordinates for every pixel centre in ``input_img``.

    :param input_img: Path to the reference input image file.
    :param output_img: Path to the output 2-band WGS84 coordinate image file.
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param datatype: Optional ``rsgispy.TYPE_*`` constant for the output image
                     (defaults to :data:`rsgispy.TYPE_32FLOAT`).
    :param no_data_val: Output no-data value (Default = ``0.0``).
    :param calc_stats: Populate image statistics and pyramids
                       (Default = ``True``).
    :param use_pbar: Show a progress bar during calculation (Default = ``True``).
    :param creation_opts: Optional GDAL creation options mapping or list.

    .. code:: python

        import rsgispy.imageutils.spatial
        rsgispy.imageutils.spatial.calc_wgs84_pixel_locations(
            "input_img.tif", "wgs84_coords.tif", gdalformat="GTIFF"
        )
    """
    # pylint: disable=too-many-arguments
    ensure_file_exists(input_img)
    eff_dtype = rsgispy.TYPE_32FLOAT if datatype is None else datatype
    out_rio_dt = rsgispy.get_rasterio_datatype(eff_dtype)
    out_np_dt = np.dtype(rsgispy.get_numpy_datatype(eff_dtype))
    norm_opts = normalise_creation_options(creation_opts)

    with ExitStack() as stack:
        src = stack.enter_context(rasterio.open(input_img, "r"))
        if src.crs is None:
            raise rsgispy.RSGISPyException(
                f"Image '{input_img}' does not have a CRS/projection defined."
            )
        in_crs = pyproj.CRS.from_user_input(src.crs)
        wgs84_crs = pyproj.CRS.from_epsg(4326)
        transformer = pyproj.Transformer.from_crs(
            in_crs, wgs84_crs, always_xy=True
        )

        dst = create_output_from_dataset(
            stack=stack,
            src=src,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype=out_rio_dt,
            count=2,
            creation_opts=norm_opts,
        )
        if no_data_val is not None:
            dst.nodata = float(no_data_val)
        pop_stats_on_close(
            dst, calc_stats, no_data_val=no_data_val, use_pbar=use_pbar
        )

        for win in iter_dataset_windows(src.height, src.width, use_pbar=use_pbar):
            x_grid, y_grid = _block_centre_coords(
                src.transform,
                int(win.col_off),
                int(win.row_off),
                int(win.width),
                int(win.height),
            )
            lon_grid, lat_grid = transformer.transform(x_grid, y_grid)
            dst.write(
                np.stack((lon_grid, lat_grid), axis=0).astype(
                    out_np_dt, copy=False
                ),
                window=win,
            )


def _parse_grid_algorithm(
    gdal_grid_alg: Optional[str], default_nodata: float
) -> Tuple[str, Dict[str, float]]:
    """Parse a GDAL-style grid algorithm specification string."""
    if not gdal_grid_alg:
        return "invdist", {
            "power": 2.0,
            "smoothing": 0.0,
            "radius1": 0.0,
            "radius2": 0.0,
            "angle": 0.0,
            "max_points": 0.0,
            "min_points": 0.0,
            "nodata": float(default_nodata),
        }

    parts = [p.strip() for p in gdal_grid_alg.split(":") if p.strip()]
    alg_name = parts[0].lower()
    params: Dict[str, float] = {"nodata": float(default_nodata)}
    for token in parts[1:]:
        if "=" in token:
            key, val = token.split("=", 1)
            params[key.strip().lower()] = float(val.strip())
    return alg_name, params


_SUPPORTED_GRID_ALGS = frozenset(
    {
        "invdist",
        "invdistnn",
        "nearest",
        "linear",
        "average",
        "minimum",
        "maximum",
        "range",
        "count",
        "average_distance",
        "average_distance_pts",
    }
)


def _eval_grid_block_vec(
    alg: str,
    params: Dict[str, float],
    qx: np.ndarray,
    qy: np.ndarray,
    pts_x: np.ndarray,
    pts_y: np.ndarray,
    pts_z: np.ndarray,
) -> np.ndarray:
    """Vectorised evaluation of point-neighbourhood metrics on a chunk of cells."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    nodata = params["nodata"]
    min_pts = int(params.get("min_points", 0))
    max_pts = int(params.get("max_points", 0))
    radius1 = params.get("radius1", 0.0)
    radius2 = params.get("radius2", 0.0)
    angle_deg = params.get("angle", 0.0)
    n_cells = qx.shape[0]
    n_pts = pts_z.shape[0]

    if n_pts == 0:
        return np.full(n_cells, nodata, dtype=np.float64)

    dx = pts_x[None, :] - qx[:, None]
    dy = pts_y[None, :] - qy[:, None]
    dist = np.hypot(dx, dy)

    if radius1 <= 0.0 and radius2 <= 0.0:
        in_mask = np.ones((n_cells, n_pts), dtype=bool)
    else:
        r1 = radius1 if radius1 > 0.0 else radius2
        r2 = radius2 if radius2 > 0.0 else radius1
        if angle_deg != 0.0:
            theta = math.radians(angle_deg)
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            rx = dx * cos_t + dy * sin_t
            ry = -dx * sin_t + dy * cos_t
        else:
            rx, ry = dx, dy
        in_mask = ((rx / r1) ** 2 + (ry / r2) ** 2) <= 1.0

    need_xy = alg == "average_distance_pts"
    if 0 < max_pts < n_pts:
        eff_dist = np.where(in_mask, dist, np.inf)
        top_idx = np.argpartition(eff_dist, max_pts, axis=1)[:, :max_pts]
        dist = np.take_along_axis(eff_dist, top_idx, axis=1)
        in_mask = np.isfinite(dist)
        z_mat = pts_z[top_idx]
        x_mat = pts_x[top_idx] if need_xy else None
        y_mat = pts_y[top_idx] if need_xy else None
    else:
        z_mat = np.broadcast_to(pts_z[None, :], (n_cells, n_pts))
        x_mat = np.broadcast_to(pts_x[None, :], (n_cells, n_pts)) if need_xy else None
        y_mat = np.broadcast_to(pts_y[None, :], (n_cells, n_pts)) if need_xy else None

    cnt = in_mask.sum(axis=1)
    ok = (cnt > 0) & (cnt >= min_pts)
    res = np.full(n_cells, nodata, dtype=np.float64)
    if not np.any(ok):
        return res

    if alg == "invdist":
        power = params.get("power", 2.0)
        smoothing = params.get("smoothing", 0.0)
        has_exact = np.zeros(n_cells, dtype=bool)
        if smoothing == 0.0:
            exact = in_mask & (dist == 0.0)
            has_exact = exact.any(axis=1) & ok
            if np.any(has_exact):
                ex_m = exact[has_exact]
                res[has_exact] = np.where(ex_m, z_mat[has_exact], 0.0).sum(
                    axis=1
                ) / ex_m.sum(axis=1)
        need_w = ok & (~has_exact)
        if np.any(need_w):
            vm = in_mask[need_w]
            d_sub = dist[need_w]
            eff_d = (
                np.hypot(d_sub, smoothing) if smoothing != 0.0 else d_sub
            )
            safe_d = np.where(vm, eff_d, 1.0)
            weights = np.where(vm, 1.0 / np.power(safe_d, power), 0.0)
            z_sub = np.where(vm, z_mat[need_w], 0.0)
            res[need_w] = (weights * z_sub).sum(axis=1) / weights.sum(axis=1)
        return res

    if alg == "nearest":
        eff_d = np.where(in_mask, dist, np.inf)
        min_idx = np.argmin(eff_d, axis=1)
        res[ok] = z_mat[ok, min_idx[ok]]
        return res

    if alg == "average":
        z_sum = np.where(in_mask, z_mat, 0.0).sum(axis=1)
        res[ok] = z_sum[ok] / cnt[ok]
        return res

    if alg == "minimum":
        res[ok] = np.where(in_mask[ok], z_mat[ok], np.inf).min(axis=1)
        return res

    if alg == "maximum":
        res[ok] = np.where(in_mask[ok], z_mat[ok], -np.inf).max(axis=1)
        return res

    if alg == "range":
        z_ok = z_mat[ok]
        m_ok = in_mask[ok]
        res[ok] = np.where(m_ok, z_ok, -np.inf).max(axis=1) - np.where(
            m_ok, z_ok, np.inf
        ).min(axis=1)
        return res

    if alg == "count":
        res[ok] = cnt[ok].astype(np.float64)
        return res

    if alg == "average_distance":
        d_sum = np.where(in_mask, dist, 0.0).sum(axis=1)
        res[ok] = d_sum[ok] / cnt[ok]
        return res

    if alg == "average_distance_pts":
        valid_pair = ok & (cnt >= 2)
        if np.any(valid_pair) and x_mat is not None and y_mat is not None:
            vp_idxs = np.nonzero(valid_pair)[0]
            k_cols = x_mat.shape[1]
            sub_chunk = max(1, 1_000_000 // max(1, k_cols * k_cols))
            for s_off in range(0, vp_idxs.size, sub_chunk):
                s_idx = vp_idxs[s_off : s_off + sub_chunk]
                xm = x_mat[s_idx]
                ym = y_mat[s_idx]
                mm = in_mask[s_idx]
                c_cnt = cnt[s_idx].astype(np.float64)
                p_mask = mm[:, :, None] & mm[:, None, :]
                p_d = np.where(
                    p_mask,
                    np.hypot(
                        xm[:, :, None] - xm[:, None, :],
                        ym[:, :, None] - ym[:, None, :],
                    ),
                    0.0,
                )
                res[s_idx] = p_d.sum(axis=(1, 2)) / (c_cnt * (c_cnt - 1.0))
        return res

    raise rsgispy.RSGISPyException(f"Unsupported gridding algorithm: '{alg}'.")


def _grid_general_block(
    alg: str,
    params: Dict[str, float],
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    pts_x: np.ndarray,
    pts_y: np.ndarray,
    pts_z: np.ndarray,
    kd_tree: Optional[Any] = None,
) -> np.ndarray:
    """Vectorised grid evaluation for a 2D window using cKDTree / NumPy chunks."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    nodata = params["nodata"]
    radius1 = params.get("radius1", 0.0)
    radius2 = params.get("radius2", 0.0)
    min_pts = int(params.get("min_points", 0))

    # Fast path: unrestricted or circular nearest-neighbour via cKDTree
    if (
        alg == "nearest"
        and kd_tree is not None
        and min_pts <= 1
        and (
            (radius1 <= 0.0 and radius2 <= 0.0)
            or (
                params.get("angle", 0.0) == 0.0
                and (radius1 <= 0.0 or radius2 <= 0.0 or radius1 == radius2)
            )
        )
    ):
        q_pts = np.column_stack((x_grid.ravel(), y_grid.ravel()))
        if radius1 <= 0.0 and radius2 <= 0.0:
            _, idxs = kd_tree.query(q_pts, k=1)
            return pts_z[idxs].reshape(x_grid.shape)
        eff_r = radius1 if radius1 > 0.0 else radius2
        dists, idxs = kd_tree.query(
            q_pts, k=1, distance_upper_bound=eff_r * (1.0 + 1e-12)
        )
        valid = np.isfinite(dists) & (idxs < pts_z.size) & (dists <= eff_r)
        safe_idx = np.where(valid, idxs, 0)
        return np.where(valid, pts_z[safe_idx], nodata).reshape(x_grid.shape)

    # Pre-filter candidate points to block bounding box expanded by max radius
    sub_x, sub_y, sub_z = pts_x, pts_y, pts_z
    if radius1 > 0.0 or radius2 > 0.0:
        r_max = max(radius1, radius2)
        in_box = (
            (pts_x >= float(x_grid.min()) - r_max)
            & (pts_x <= float(x_grid.max()) + r_max)
            & (pts_y >= float(y_grid.min()) - r_max)
            & (pts_y <= float(y_grid.max()) + r_max)
        )
        if not np.any(in_box):
            return np.full(x_grid.shape, nodata, dtype=np.float64)
        sub_x = pts_x[in_box]
        sub_y = pts_y[in_box]
        sub_z = pts_z[in_box]

    qx_flat = x_grid.ravel()
    qy_flat = y_grid.ravel()
    n_cells = qx_flat.size
    chunk_cells = max(1, min(n_cells, 2_000_000 // max(1, sub_z.size)))
    out_flat = np.empty(n_cells, dtype=np.float64)
    for start in range(0, n_cells, chunk_cells):
        end = min(start + chunk_cells, n_cells)
        out_flat[start:end] = _eval_grid_block_vec(
            alg=alg,
            params=params,
            qx=qx_flat[start:end],
            qy=qy_flat[start:end],
            pts_x=sub_x,
            pts_y=sub_y,
            pts_z=sub_z,
        )
    return out_flat.reshape(x_grid.shape)


def _grid_invdistnn(
    params: Dict[str, float],
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    pts_x: np.ndarray,
    pts_y: np.ndarray,
    pts_z: np.ndarray,
    kd_tree: Optional[Any] = None,
) -> np.ndarray:
    """Evaluate k-nearest inverse distance weighting across the target grid."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    rsgispy.check_optional_dep(HAVE_SCIPY, "grid_scattered_pts", "scipy", "stats")
    out_nodata = params["nodata"]
    power = params.get("power", 2.0)
    radius = params.get("radius", 1.0)
    max_pts = max(1, int(params.get("max_points", 12)))
    min_pts = int(params.get("min_points", 0))
    k_query = min(max_pts, pts_z.size)

    tree = (
        kd_tree
        if kd_tree is not None
        else cKDTree(np.column_stack((pts_x, pts_y)))
    )
    q_pts = np.column_stack((x_grid.ravel(), y_grid.ravel()))
    dists, idxs = tree.query(
        q_pts,
        k=k_query,
        distance_upper_bound=radius if radius > 0.0 else np.inf,
    )
    if k_query == 1:
        dists = dists[:, None]
        idxs = idxs[:, None]

    valid_nbr = np.isfinite(dists) & (idxs < pts_z.size)
    n_found = valid_nbr.sum(axis=1)
    res_flat = np.full(q_pts.shape[0], out_nodata, dtype=np.float64)

    nbr_vals = pts_z[np.where(valid_nbr, idxs, 0)]
    exact = valid_nbr & (dists == 0.0)
    has_exact = np.any(exact, axis=1) & (n_found >= min_pts)
    if np.any(has_exact):
        ex_cnt = exact[has_exact].sum(axis=1)
        ex_sum = np.where(exact[has_exact], nbr_vals[has_exact], 0.0).sum(
            axis=1
        )
        res_flat[has_exact] = ex_sum / ex_cnt

    need_idw = (~has_exact) & (n_found >= max(1, min_pts))
    if np.any(need_idw):
        d_sub = dists[need_idw]
        v_sub = valid_nbr[need_idw]
        w_sub = np.where(
            v_sub, 1.0 / np.power(np.where(v_sub, d_sub, 1.0), power), 0.0
        )
        z_sub = np.where(v_sub, nbr_vals[need_idw], 0.0)
        res_flat[need_idw] = np.sum(w_sub * z_sub, axis=1) / np.sum(
            w_sub, axis=1
        )
    return res_flat.reshape(x_grid.shape)


def _grid_linear(
    params: Dict[str, float],
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    pts_x: np.ndarray,
    pts_y: np.ndarray,
    pts_z: np.ndarray,
    interp: Optional[Any] = None,
    kd_tree: Optional[Any] = None,
) -> np.ndarray:
    """Evaluate Delaunay linear interpolation across the target grid."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    rsgispy.check_optional_dep(HAVE_SCIPY, "grid_scattered_pts", "scipy", "stats")
    out_nodata = params["nodata"]
    radius = params.get("radius", -1.0)
    lin_interp = (
        interp
        if interp is not None
        else LinearNDInterpolator(
            np.column_stack((pts_x, pts_y)), pts_z, fill_value=np.nan
        )
    )
    vals = lin_interp(x_grid, y_grid)
    if radius >= 0.0:
        tree = (
            kd_tree
            if kd_tree is not None
            else cKDTree(np.column_stack((pts_x, pts_y)))
        )
        dists, _ = tree.query(np.column_stack((x_grid.ravel(), y_grid.ravel())))
        vals[dists.reshape(x_grid.shape) > radius] = np.nan
    return np.where(np.isfinite(vals), vals, out_nodata)


def grid_scattered_pts(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    vec_col: str,
    input_img: str,
    output_img: str,
    gdal_grid_alg: Optional[str] = None,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = 0.0,
    calc_stats: bool = True,
    use_pbar: bool = True,
    creation_opts: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
) -> None:
    """
    Interpolate scattered vector points from ``vec_file`` (attribute column
    ``vec_col``) onto the raster grid defined by ``input_img``.

    Supported algorithms in ``gdal_grid_alg`` match GDAL's ``gdal_grid`` syntax:
    ``invdist``, ``invdistnn``, ``average``, ``nearest``, ``linear``,
    ``minimum``, ``maximum``, ``range``, ``count``, ``average_distance``, and
    ``average_distance_pts``.

    :param vec_file: Input vector file path or an in-memory
                     :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Input vector layer name (or ``None`` for default layer).
    :param vec_col: Numeric attribute column name with values to grid.
    :param input_img: Reference raster image defining output extent, CRS, and
                      pixel grid.
    :param output_img: Output raster file path.
    :param gdal_grid_alg: Optional GDAL grid algorithm specification string.
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param datatype: Optional ``rsgispy.TYPE_*`` constant for the output image
                     (defaults to :data:`rsgispy.TYPE_32FLOAT`).
    :param no_data_val: Output no-data value (Default = ``0.0``).
    :param calc_stats: Populate image statistics and pyramids
                       (Default = ``True``).
    :param use_pbar: Show a progress bar during gridding (Default = ``True``).
    :param creation_opts: Optional GDAL creation options mapping or list.

    .. code:: python

        import rsgispy.imageutils.spatial
        rsgispy.imageutils.spatial.grid_scattered_pts(
            "points.gpkg",
            "pts_layer",
            "elev",
            "ref_img.tif",
            "gridded_elev.tif",
            gdal_grid_alg="invdist:power=2.0",
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-branches
    ensure_file_exists(input_img)

    gdf = read_vec_gdf(vec_file, vec_lyr=vec_lyr)
    vec_desc = (
        "<GeoDataFrame>"
        if isinstance(vec_file, gpd.GeoDataFrame)
        else vec_file
    )
    if gdf.empty:
        raise rsgispy.RSGISPyException(
            f"Vector layer '{vec_lyr}' in '{vec_desc}' contains no features."
        )
    if vec_col not in gdf.columns:
        raise rsgispy.RSGISPyException(
            f"Column '{vec_col}' not found in vector layer '{vec_lyr}'."
        )

    eff_nd = 0.0 if no_data_val is None else float(no_data_val)
    eff_dtype = rsgispy.TYPE_32FLOAT if datatype is None else datatype
    alg, params = _parse_grid_algorithm(gdal_grid_alg, eff_nd)
    if alg not in _SUPPORTED_GRID_ALGS:
        raise rsgispy.RSGISPyException(
            f"Unsupported gridding algorithm: '{alg}'."
        )
    out_nodata = params["nodata"]
    norm_opts = normalise_creation_options(creation_opts)

    with ExitStack() as stack:
        src = stack.enter_context(rasterio.open(input_img, "r"))
        if src.crs is not None and gdf.crs is not None and gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)

        centroids = gdf.geometry.centroid
        pts_x = centroids.x.to_numpy(dtype=np.float64)
        pts_y = centroids.y.to_numpy(dtype=np.float64)
        pts_z = gdf[vec_col].to_numpy(dtype=np.float64)
        valid_pts = np.isfinite(pts_x) & np.isfinite(pts_y) & np.isfinite(pts_z)
        pts_x, pts_y, pts_z = pts_x[valid_pts], pts_y[valid_pts], pts_z[valid_pts]
        if pts_z.size == 0:
            raise rsgispy.RSGISPyException(
                f"No valid point values found in column '{vec_col}'."
            )

        kd_tree: Optional[Any] = None
        lin_interp: Optional[Any] = None
        if alg == "invdistnn":
            rsgispy.check_optional_dep(
                HAVE_SCIPY, "grid_scattered_pts", "scipy", "stats"
            )
            kd_tree = cKDTree(np.column_stack((pts_x, pts_y)))
        elif alg == "linear":
            rsgispy.check_optional_dep(
                HAVE_SCIPY, "grid_scattered_pts", "scipy", "stats"
            )
            pts_xy = np.column_stack((pts_x, pts_y))
            lin_interp = LinearNDInterpolator(pts_xy, pts_z, fill_value=np.nan)
            if params.get("radius", -1.0) >= 0.0:
                kd_tree = cKDTree(pts_xy)
        elif alg == "nearest" and HAVE_SCIPY:
            kd_tree = cKDTree(np.column_stack((pts_x, pts_y)))

        dst = create_output_from_dataset(
            stack=stack,
            src=src,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype=rsgispy.get_rasterio_datatype(eff_dtype),
            count=1,
            creation_opts=norm_opts,
        )
        dst.nodata = out_nodata
        pop_stats_on_close(
            dst, calc_stats, no_data_val=out_nodata, use_pbar=use_pbar
        )
        out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(eff_dtype))

        for win in iter_dataset_windows(
            src.height, src.width, use_pbar=use_pbar
        ):
            win_w = int(win.width)
            win_h = int(win.height)
            x_grid, y_grid = _block_centre_coords(
                src.transform,
                int(win.col_off),
                int(win.row_off),
                win_w,
                win_h,
            )
            if alg == "invdistnn":
                blk = _grid_invdistnn(
                    params, x_grid, y_grid, pts_x, pts_y, pts_z, kd_tree=kd_tree
                )
            elif alg == "linear":
                blk = _grid_linear(
                    params,
                    x_grid,
                    y_grid,
                    pts_x,
                    pts_y,
                    pts_z,
                    interp=lin_interp,
                    kd_tree=kd_tree,
                )
            else:
                blk = _grid_general_block(
                    alg,
                    params,
                    x_grid,
                    y_grid,
                    pts_x,
                    pts_y,
                    pts_z,
                    kd_tree=kd_tree,
                )
            dst.write(
                cast_result_block(blk, win_h, win_w, out_np_dtype),
                1,
                window=win,
            )
