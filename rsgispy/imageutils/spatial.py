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
)
from rsgispy.imageutils._helpers import (
    maybe_pop_stats,
    normalise_creation_options,
)
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
    calc_stats: bool = False,
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
                       (Default = ``False``).
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

    maybe_pop_stats(
        output_img=output_img,
        calc_stats=calc_stats,
        no_data_val=no_data_val,
        use_pbar=use_pbar,
    )


def calc_wgs84_pixel_size(
    input_img: str,
    output_img: str,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = False,
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
                       (Default = ``False``).
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

    maybe_pop_stats(
        output_img=output_img,
        calc_stats=calc_stats,
        no_data_val=no_data_val,
        use_pbar=use_pbar,
    )


def calc_pixel_locations(
    input_img: str,
    output_img: str,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = False,
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
                       (Default = ``False``).
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

    maybe_pop_stats(
        output_img=output_img,
        calc_stats=calc_stats,
        no_data_val=no_data_val,
        use_pbar=use_pbar,
    )


def calc_wgs84_pixel_locations(
    input_img: str,
    output_img: str,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = 0.0,
    calc_stats: bool = False,
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
                       (Default = ``False``).
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

    maybe_pop_stats(
        output_img=output_img,
        calc_stats=calc_stats,
        no_data_val=no_data_val,
        use_pbar=use_pbar,
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


def _ellipse_distances(
    px: float,
    py: float,
    pts_x: np.ndarray,
    pts_y: np.ndarray,
    radius1: float,
    radius2: float,
    angle_deg: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return Euclidean distances and boolean ellipse mask for one grid cell."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    dx = pts_x - px
    dy = pts_y - py
    dist = np.hypot(dx, dy)
    if radius1 <= 0.0 and radius2 <= 0.0:
        return dist, np.ones(pts_x.shape[0], dtype=bool)

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
    mask = ((rx / r1) ** 2 + (ry / r2) ** 2) <= 1.0
    return dist, mask


def _eval_grid_cell(
    alg: str,
    params: Dict[str, float],
    px: float,
    py: float,
    pts_x: np.ndarray,
    pts_y: np.ndarray,
    pts_z: np.ndarray,
) -> float:
    """Evaluate a point-neighbourhood grid metric at coordinate ``(px, py)``."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-return-statements,too-many-branches
    nodata = params["nodata"]
    min_pts = int(params.get("min_points", 0))
    max_pts = int(params.get("max_points", 0))

    dists, in_mask = _ellipse_distances(
        px,
        py,
        pts_x,
        pts_y,
        params.get("radius1", 0.0),
        params.get("radius2", 0.0),
        params.get("angle", 0.0),
    )
    sel_dists = dists[in_mask]
    sel_z = pts_z[in_mask]

    if sel_z.size == 0 or sel_z.size < min_pts:
        return nodata

    order: Optional[np.ndarray] = None
    if 0 < max_pts < sel_z.size:
        order = np.argpartition(sel_dists, max_pts)[:max_pts]
        sel_dists = sel_dists[order]
        sel_z = sel_z[order]

    if alg == "invdist":
        power = params.get("power", 2.0)
        smoothing = params.get("smoothing", 0.0)
        if smoothing == 0.0:
            exact = sel_dists == 0.0
            if np.any(exact):
                return float(sel_z[exact].mean())
        weights = 1.0 / np.power(np.hypot(sel_dists, smoothing), power)
        return float(np.sum(weights * sel_z) / np.sum(weights))
    if alg == "nearest":
        return float(sel_z[int(np.argmin(sel_dists))])
    if alg == "average":
        return float(sel_z.mean())
    if alg == "minimum":
        return float(sel_z.min())
    if alg == "maximum":
        return float(sel_z.max())
    if alg == "range":
        return float(sel_z.max() - sel_z.min())
    if alg == "count":
        return float(sel_z.size)
    if alg == "average_distance":
        return float(sel_dists.mean())
    if alg == "average_distance_pts":
        if sel_z.size < 2:
            return nodata
        sx = pts_x[in_mask]
        sy = pts_y[in_mask]
        if order is not None:
            sx = sx[order]
            sy = sy[order]
        pair_dists = np.hypot(
            sx[:, None] - sx[None, :], sy[:, None] - sy[None, :]
        )
        tri = pair_dists[np.triu_indices(sx.size, k=1)]
        return float(tri.mean()) if tri.size > 0 else nodata

    raise rsgispy.RSGISPyException(f"Unsupported gridding algorithm: '{alg}'.")


def _grid_invdistnn(
    params: Dict[str, float],
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    pts_x: np.ndarray,
    pts_y: np.ndarray,
    pts_z: np.ndarray,
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

    tree = cKDTree(np.column_stack((pts_x, pts_y)))
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
) -> np.ndarray:
    """Evaluate Delaunay linear interpolation across the target grid."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    rsgispy.check_optional_dep(HAVE_SCIPY, "grid_scattered_pts", "scipy", "stats")
    out_nodata = params["nodata"]
    radius = params.get("radius", -1.0)
    interp = LinearNDInterpolator(
        np.column_stack((pts_x, pts_y)), pts_z, fill_value=np.nan
    )
    vals = interp(x_grid, y_grid)
    if radius >= 0.0:
        tree = cKDTree(np.column_stack((pts_x, pts_y)))
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
    calc_stats: bool = False,
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
                       (Default = ``False``).
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
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
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

        x_grid, y_grid = _block_centre_coords(
            src.transform, 0, 0, src.width, src.height
        )
        if alg == "invdistnn":
            out_arr = _grid_invdistnn(
                params, x_grid, y_grid, pts_x, pts_y, pts_z
            )
        elif alg == "linear":
            out_arr = _grid_linear(params, x_grid, y_grid, pts_x, pts_y, pts_z)
        else:
            out_arr = np.full(
                (src.height, src.width), out_nodata, dtype=np.float64
            )
            tqdm_cls = rsgispy.get_tqdm()
            for r_idx in tqdm_cls(
                range(src.height),
                desc="Gridding",
                unit="row",
                disable=not use_pbar,
            ):
                for c_idx in range(src.width):
                    out_arr[r_idx, c_idx] = _eval_grid_cell(
                        alg,
                        params,
                        float(x_grid[r_idx, c_idx]),
                        float(y_grid[r_idx, c_idx]),
                        pts_x,
                        pts_y,
                        pts_z,
                    )

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
        out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(eff_dtype))
        dst.write(
            cast_result_block(out_arr, src.height, src.width, out_np_dtype),
            1,
        )

    maybe_pop_stats(
        output_img=output_img,
        calc_stats=calc_stats,
        no_data_val=out_nodata,
        use_pbar=use_pbar,
    )
