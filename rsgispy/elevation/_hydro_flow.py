"""
Stage 2: Flow direction, flow accumulation, flow length, flow distance, and
flow path tracing for :mod:`rsgispy.elevation.hydrology`.
"""

from contextlib import ExitStack
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pyproj
import rasterio
from shapely.geometry import LineString

import rsgispy
from rsgispy import RSGISPyException
from rsgispy.elevation._hydro_common import (
    _D8_CODE_TO_ANGLE,
    _D8_CODE_TO_IDX,
    _D8_DIR_TO_OFFSET,
    _call_topo_kernel,
    _d8_flat_resolve_in_memory,
    _detect_flow_dir_mode,
    _has_numba,
    _read_d8_grid,
    _resolve_out_dtype,
    _resolve_out_nodata,
    _run_progress_steps,
    _validate_matching_grid,
    _validate_projected_crs,
    _warn_numpy_fallback,
    _write_raster_array,
)
from rsgispy.elevation._hydro_kernels import (
    D8_DC,
    D8_DR,
    apply_flat_routing_d8_kernel,
    flow_acc_d8_kernel,
    flow_acc_dinf_kernel,
    flow_acc_mfd_kernel,
    flow_dir_d8_njit,
    flow_dir_d8_numpy,
    flow_dir_dinf_njit,
    flow_dir_dinf_numpy,
    flow_dir_mfd_njit,
    flow_dir_mfd_numpy,
    flow_distance_horiz_kernel,
    flow_distance_vert_kernel,
    flow_length_downstream_kernel,
    flow_length_upstream_kernel,
    resolve_flats_kernel,
)
from rsgispy.imageprocess import build_valid_mask, ensure_file_exists
from rsgispy.vectorprocess import read_vec_gdf, write_vec_gdf


def flow_direction(
    input_img: str,
    output_img: str,
    method: str = "d8",
    flat_routing: bool = True,
    mfd_exponent: float = 1.1,
    allow_pure_python: bool = False,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Calculate hydrological flow direction using D8, D-Infinity, or MFD.

    Parameters
    ----------
    input_img : str
        Path to the input (preferably conditioned) DEM raster.
    output_img : str
        Path to the output flow direction raster.
    method : {'d8', 'dinf', 'mfd'}, optional
        Flow routing algorithm (Default = ``'d8'``).

        - ``'d8'``: single-band ``uint8`` power-of-two bitmask starting East.
        - ``'dinf'``: single-band ``float32`` Tarboton (1997) continuous angle
          in radians ``[0, 2*pi)`` CCW from East (``-1.0`` on flats).
        - ``'mfd'``: 8-band ``float32`` Quinn et al. (1991) flow fractions.
    flat_routing : bool, optional
        If ``True``, resolve flat cells via Barnes et al. (2014) double-gradient
        routing (Default = ``True``).
    mfd_exponent : float, optional
        Slope exponent ``p`` for ``method='mfd'`` (Default = ``1.1``).
    allow_pure_python : bool, optional
        Passed through to flat-routing traversal if ``numba`` is unavailable
        (Default = ``False``).
    gdalformat : str, optional
        GDAL/rasterio output driver name (Default = ``'GTIFF'``).
    datatype : int or None, optional
        Optional ``rsgispy.TYPE_*`` constant.
    no_data_val : float or None, optional
        Optional input DEM nodata override (Default = ``None``).
    calc_stats : bool, optional
        If ``True``, compute output statistics and overview pyramids
        (Default = ``True``).
    use_pbar : bool, optional
        Display a progress bar (Default = ``True``).
    """
    # pylint: disable=too-many-arguments,too-many-locals,too-many-branches
    method_norm = method.strip().lower()
    if method_norm not in ("d8", "dinf", "mfd"):
        raise RSGISPyException(
            f"Invalid flow direction method '{method}'. Must be 'd8', 'dinf', or 'mfd'."
        )
    if mfd_exponent <= 0.0:
        raise RSGISPyException("mfd_exponent must be positive.")
    ensure_file_exists(input_img, "Input DEM image")

    with _run_progress_steps("Computing flow direction", 3, use_pbar) as pbar:
        with rasterio.open(input_img, "r") as src:
            dx, dy = _validate_projected_crs(src, input_img)
            eff_nd = no_data_val if no_data_val is not None else src.nodata
            dem = src.read(1, out_dtype=np.float64)
            valid = build_valid_mask(dem, eff_nd)
            pbar.update(1)

            if method_norm == "d8":
                _, out_rio_dt, out_np_dt = _resolve_out_dtype(
                    datatype, rsgispy.TYPE_8UINT
                )
                out_nd = _resolve_out_nodata(out_np_dt, None, 255.0)
                if _has_numba():
                    out_arr = flow_dir_d8_njit(dem, valid, dx, dy)
                    if flat_routing and np.any(valid & (out_arr == 0)):
                        g_grid = resolve_flats_kernel(dem, valid)
                        apply_flat_routing_d8_kernel(
                            dem, valid, out_arr, g_grid, dx, dy
                        )
                else:
                    out_arr = flow_dir_d8_numpy(dem, valid, dx, dy)
                    if flat_routing and np.any(valid & (out_arr == 0)):
                        g_grid = _call_topo_kernel(
                            resolve_flats_kernel,
                            allow_pure_python,
                            "flow_direction",
                            dem,
                            valid,
                            warn=True,
                        )
                        _call_topo_kernel(
                            apply_flat_routing_d8_kernel,
                            allow_pure_python,
                            "flow_direction",
                            dem,
                            valid,
                            out_arr,
                            g_grid,
                            dx,
                            dy,
                            warn=False,
                        )
                    else:
                        _warn_numpy_fallback("flow_direction")

                if out_nd is not None and int(round(out_nd)) != 255:
                    out_arr = np.where(valid, out_arr, int(round(out_nd))).astype(
                        out_np_dt
                    )

            elif method_norm == "dinf":
                _, out_rio_dt, out_np_dt = _resolve_out_dtype(
                    datatype, rsgispy.TYPE_32FLOAT
                )
                out_nd = _resolve_out_nodata(out_np_dt, None, -9999.0)
                fill_nd = -9999.0 if out_nd is None else float(out_nd)
                if _has_numba():
                    out_arr = flow_dir_dinf_njit(dem, valid, dx, dy, fill_nd)
                else:
                    out_arr = flow_dir_dinf_numpy(dem, valid, dx, dy, fill_nd)

                if flat_routing and np.any(valid & (out_arr == -1.0)):
                    fdir_d8 = _d8_flat_resolve_in_memory(
                        dem,
                        valid,
                        dx,
                        dy,
                        allow_pure_python,
                        "flow_direction",
                        warn=True,
                    )
                    flat_pts = valid & (out_arr == -1.0) & (fdir_d8 > 0)
                    if np.any(flat_pts):
                        rr, cc = np.where(flat_pts)
                        for r_i, c_i in zip(rr, cc):
                            out_arr[r_i, c_i] = np.float32(
                                _D8_CODE_TO_ANGLE[int(fdir_d8[r_i, c_i])]
                            )
                elif not _has_numba():
                    _warn_numpy_fallback("flow_direction")

            else:  # mfd
                _, out_rio_dt, out_np_dt = _resolve_out_dtype(
                    datatype, rsgispy.TYPE_32FLOAT
                )
                out_nd = _resolve_out_nodata(out_np_dt, None, -9999.0)
                fill_nd = -9999.0 if out_nd is None else float(out_nd)
                if _has_numba():
                    out_arr = flow_dir_mfd_njit(
                        dem, valid, dx, dy, float(mfd_exponent), fill_nd
                    )
                else:
                    out_arr = flow_dir_mfd_numpy(
                        dem, valid, dx, dy, float(mfd_exponent), fill_nd
                    )

                needs_flat = False
                if flat_routing:
                    band_sum = np.sum(out_arr, axis=0)
                    unresolved = valid & (band_sum == 0.0)
                    if np.any(unresolved):
                        needs_flat = True
                        fdir_d8 = _d8_flat_resolve_in_memory(
                            dem,
                            valid,
                            dx,
                            dy,
                            allow_pure_python,
                            "flow_direction",
                            warn=True,
                        )
                        flat_pts = unresolved & (fdir_d8 > 0)
                        if np.any(flat_pts):
                            rr, cc = np.where(flat_pts)
                            for r_i, c_i in zip(rr, cc):
                                k_idx = _D8_CODE_TO_IDX[int(fdir_d8[r_i, c_i])]
                                out_arr[k_idx, r_i, c_i] = np.float32(1.0)
                if not needs_flat and not _has_numba():
                    _warn_numpy_fallback("flow_direction")

            pbar.update(1)

            _write_raster_array(
                output_img=output_img,
                arr=out_arr,
                ref_src=src,
                gdalformat=gdalformat,
                out_rio_dtype=out_rio_dt,
                out_np_dtype=out_np_dt,
                out_no_data_val=out_nd,
                calc_stats=calc_stats,
                use_pbar=use_pbar,
            )
            pbar.update(1)


def flow_accumulation(
    input_img: str,
    output_img: str,
    weight_img: Optional[str] = None,
    out_units: str = "cells",
    allow_pure_python: bool = False,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Compute upslope contributing flow accumulation via $O(N)$ Kahn's topological
    sort, auto-detecting D8, D-Infinity, or 8-band MFD inputs.

    Parameters
    ----------
    input_img : str
        Path to the input flow direction raster (D8 ``uint8``, D-Inf
        ``float32``, or 8-band MFD ``float32``).
    output_img : str
        Path to the output flow accumulation raster (``float32``).
    weight_img : str or None, optional
        Optional spatially aligned per-cell runoff/weight raster
        (Default = ``None``, unit weight ``1.0`` per cell).
    out_units : {'cells', 'area'}, optional
        Output units: contributing cell count (``'cells'``) or drainage area in
        square metres (``'area'``) (Default = ``'cells'``).
    allow_pure_python : bool, optional
        If ``True``, allow falling back to pure Python with a
        ``RuntimeWarning`` when ``numba`` is unavailable (Default = ``False``).
    gdalformat : str, optional
        GDAL/rasterio output driver name (Default = ``'GTIFF'``).
    datatype : int or None, optional
        Optional ``rsgispy.TYPE_*`` constant (Default = ``rsgispy.TYPE_32FLOAT``).
    no_data_val : float or None, optional
        Optional input flow direction nodata override (Default = ``None``).
    calc_stats : bool, optional
        If ``True``, compute output statistics and overview pyramids
        (Default = ``True``).
    use_pbar : bool, optional
        Display a progress bar (Default = ``True``).
    """
    # pylint: disable=too-many-arguments,too-many-locals
    units_norm = out_units.strip().lower()
    if units_norm not in ("cells", "area"):
        raise RSGISPyException(
            f"Invalid out_units '{out_units}'. Must be 'cells' or 'area'."
        )
    ensure_file_exists(input_img, "Input flow direction image")
    if weight_img is not None:
        ensure_file_exists(weight_img, "Input weight image")

    _, out_rio_dt, out_np_dt = _resolve_out_dtype(datatype, rsgispy.TYPE_32FLOAT)

    with _run_progress_steps("Computing flow accumulation", 3, use_pbar) as pbar:
        with ExitStack() as stack:
            src = stack.enter_context(rasterio.open(input_img, "r"))
            dx, dy = _validate_projected_crs(src, input_img)
            mode = _detect_flow_dir_mode(src)

            eff_nd = no_data_val if no_data_val is not None else src.nodata
            if eff_nd is None:
                eff_nd = 255.0 if mode == "d8" else -9999.0

            if weight_img is not None:
                w_src = stack.enter_context(rasterio.open(weight_img, "r"))
                _validate_matching_grid(src, w_src, input_img, weight_img)
                w_arr = w_src.read(1, out_dtype=np.float64)
                w_valid = build_valid_mask(w_arr, w_src.nodata)
                weights = np.where(w_valid, w_arr, 0.0)
            else:
                weights = np.ones((src.height, src.width), dtype=np.float64)
                w_valid = np.ones((src.height, src.width), dtype=bool)
            pbar.update(1)

            if mode == "d8":
                fdir = src.read(1).astype(np.uint8, copy=False)
                valid = build_valid_mask(fdir, eff_nd) & w_valid
                acc = _call_topo_kernel(
                    flow_acc_d8_kernel,
                    allow_pure_python,
                    "flow_accumulation",
                    fdir,
                    valid,
                    weights,
                )
            elif mode == "dinf":
                fdir = src.read(1, out_dtype=np.float64)
                valid = build_valid_mask(fdir, eff_nd) & w_valid
                acc = _call_topo_kernel(
                    flow_acc_dinf_kernel,
                    allow_pure_python,
                    "flow_accumulation",
                    fdir,
                    valid,
                    weights,
                    dx,
                    dy,
                )
            else:
                fdir = src.read(out_dtype=np.float64)
                valid = build_valid_mask(fdir[0], eff_nd) & w_valid
                acc = _call_topo_kernel(
                    flow_acc_mfd_kernel,
                    allow_pure_python,
                    "flow_accumulation",
                    fdir,
                    valid,
                    weights,
                )
            pbar.update(1)

            if units_norm == "area":
                acc = acc * (dx * dy)

            out_nd = _resolve_out_nodata(out_np_dt, None, -9999.0)
            fill_nd = -9999.0 if out_nd is None else float(out_nd)
            out_arr = np.where(valid, acc, fill_nd)

            _write_raster_array(
                output_img=output_img,
                arr=out_arr,
                ref_src=src,
                gdalformat=gdalformat,
                out_rio_dtype=out_rio_dt,
                out_np_dtype=out_np_dt,
                out_no_data_val=out_nd,
                calc_stats=calc_stats,
                use_pbar=use_pbar,
            )
            pbar.update(1)


def flow_length(
    input_img: str,
    output_img: str,
    direction: str = "downstream",
    weight_img: Optional[str] = None,
    allow_pure_python: bool = False,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Calculate downstream or longest-upstream flow path length along a D8 flow
    direction network.

    Parameters
    ----------
    input_img : str
        Path to the input D8 flow direction raster (``uint8``).
    output_img : str
        Path to the output flow length raster (``float32``, metres or weighted units).
    direction : {'downstream', 'upstream'}, optional
        Traversal direction (Default = ``'downstream'``).
    weight_img : str or None, optional
        Optional cost/impedance weight raster (Default = ``None``, unit weight).
    allow_pure_python : bool, optional
        If ``True``, allow falling back to pure Python with a
        ``RuntimeWarning`` when ``numba`` is unavailable (Default = ``False``).
    gdalformat : str, optional
        GDAL/rasterio output driver name (Default = ``'GTIFF'``).
    datatype : int or None, optional
        Optional ``rsgispy.TYPE_*`` constant (Default = ``rsgispy.TYPE_32FLOAT``).
    no_data_val : float or None, optional
        Optional flow direction nodata override (Default = ``None``).
    calc_stats : bool, optional
        If ``True``, compute output statistics and overview pyramids
        (Default = ``True``).
    use_pbar : bool, optional
        Display a progress bar (Default = ``True``).
    """
    # pylint: disable=too-many-arguments,too-many-locals
    dir_norm = direction.strip().lower()
    if dir_norm not in ("downstream", "upstream"):
        raise RSGISPyException(
            f"Invalid direction '{direction}'. Must be 'downstream' or 'upstream'."
        )
    ensure_file_exists(input_img, "Input flow direction image")
    if weight_img is not None:
        ensure_file_exists(weight_img, "Input weight image")

    _, out_rio_dt, out_np_dt = _resolve_out_dtype(datatype, rsgispy.TYPE_32FLOAT)

    with _run_progress_steps("Computing flow length", 3, use_pbar) as pbar:
        with ExitStack() as stack:
            src = stack.enter_context(rasterio.open(input_img, "r"))
            dx, dy = _validate_projected_crs(src, input_img)
            fdir, valid = _read_d8_grid(src, no_data_val)

            if weight_img is not None:
                w_src = stack.enter_context(rasterio.open(weight_img, "r"))
                _validate_matching_grid(src, w_src, input_img, weight_img)
                w_arr = w_src.read(1, out_dtype=np.float64)
                w_valid = build_valid_mask(w_arr, w_src.nodata)
                valid &= w_valid
                weights = np.where(valid, w_arr, 1.0)
            else:
                weights = np.ones((src.height, src.width), dtype=np.float64)
            pbar.update(1)

            if dir_norm == "downstream":
                lengths = _call_topo_kernel(
                    flow_length_downstream_kernel,
                    allow_pure_python,
                    "flow_length",
                    fdir,
                    valid,
                    weights,
                    dx,
                    dy,
                )
            else:
                lengths = _call_topo_kernel(
                    flow_length_upstream_kernel,
                    allow_pure_python,
                    "flow_length",
                    fdir,
                    valid,
                    weights,
                    dx,
                    dy,
                )
            pbar.update(1)

            out_nd = _resolve_out_nodata(out_np_dt, None, -9999.0)
            fill_nd = -9999.0 if out_nd is None else float(out_nd)
            ok = valid & np.isfinite(lengths)
            out_arr = np.where(ok, lengths, fill_nd)

            _write_raster_array(
                output_img=output_img,
                arr=out_arr,
                ref_src=src,
                gdalformat=gdalformat,
                out_rio_dtype=out_rio_dt,
                out_np_dtype=out_np_dt,
                out_no_data_val=out_nd,
                calc_stats=calc_stats,
                use_pbar=use_pbar,
            )
            pbar.update(1)


def flow_distance(
    flow_dir_img: str,
    streams_img: str,
    output_img: str,
    dem_img: Optional[str] = None,
    distance_type: str = "horizontal",
    allow_pure_python: bool = False,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Calculate overland horizontal distance or vertical elevation drop to the
    nearest downstream stream pixel along D8 flow paths.

    Parameters
    ----------
    flow_dir_img : str
        Path to the input D8 flow direction raster (``uint8``).
    streams_img : str
        Path to the binary stream network mask raster (``1`` = stream).
    output_img : str
        Path to the output flow distance raster (``float32``).
    dem_img : str or None, optional
        Path to the input DEM raster (required when
        ``distance_type='vertical'``).
    distance_type : {'horizontal', 'vertical'}, optional
        Distance metric to compute (Default = ``'horizontal'``).
    allow_pure_python : bool, optional
        If ``True``, allow falling back to pure Python with a
        ``RuntimeWarning`` when ``numba`` is unavailable (Default = ``False``).
    gdalformat : str, optional
        GDAL/rasterio output driver name (Default = ``'GTIFF'``).
    datatype : int or None, optional
        Optional ``rsgispy.TYPE_*`` constant (Default = ``rsgispy.TYPE_32FLOAT``).
    no_data_val : float or None, optional
        Optional flow direction nodata override (Default = ``None``).
    calc_stats : bool, optional
        If ``True``, compute output statistics and overview pyramids
        (Default = ``True``).
    use_pbar : bool, optional
        Display a progress bar (Default = ``True``).
    """
    # pylint: disable=too-many-arguments,too-many-locals
    dist_norm = distance_type.strip().lower()
    if dist_norm not in ("horizontal", "vertical"):
        raise RSGISPyException(
            f"Invalid distance_type '{distance_type}'. Must be 'horizontal' or 'vertical'."
        )
    if dist_norm == "vertical" and dem_img is None:
        raise RSGISPyException(
            "dem_img must be provided when distance_type='vertical'."
        )
    ensure_file_exists(flow_dir_img, "Input flow direction image")
    ensure_file_exists(streams_img, "Input streams image")
    if dem_img is not None:
        ensure_file_exists(dem_img, "Input DEM image")

    _, out_rio_dt, out_np_dt = _resolve_out_dtype(datatype, rsgispy.TYPE_32FLOAT)

    with _run_progress_steps("Computing flow distance", 3, use_pbar) as pbar:
        with ExitStack() as stack:
            fdir_src = stack.enter_context(rasterio.open(flow_dir_img, "r"))
            str_src = stack.enter_context(rasterio.open(streams_img, "r"))
            dx, dy = _validate_projected_crs(fdir_src, flow_dir_img)
            _validate_matching_grid(fdir_src, str_src, flow_dir_img, streams_img)

            fdir, valid = _read_d8_grid(fdir_src, no_data_val)
            streams = str_src.read(1)
            valid &= build_valid_mask(streams, str_src.nodata)
            streams_bin = (streams == 1).astype(np.uint8)
            pbar.update(1)

            if dist_norm == "horizontal":
                dist = _call_topo_kernel(
                    flow_distance_horiz_kernel,
                    allow_pure_python,
                    "flow_distance",
                    fdir,
                    valid,
                    streams_bin,
                    dx,
                    dy,
                )
            else:
                assert dem_img is not None
                dem_src = stack.enter_context(rasterio.open(dem_img, "r"))
                _validate_matching_grid(fdir_src, dem_src, flow_dir_img, dem_img)
                dem = dem_src.read(1, out_dtype=np.float64)
                valid &= build_valid_mask(dem, dem_src.nodata)
                dist = _call_topo_kernel(
                    flow_distance_vert_kernel,
                    allow_pure_python,
                    "flow_distance",
                    fdir,
                    valid,
                    streams_bin,
                    dem,
                )
            pbar.update(1)

            out_nd = _resolve_out_nodata(out_np_dt, None, -9999.0)
            fill_nd = -9999.0 if out_nd is None else float(out_nd)
            ok = valid & np.isfinite(dist)
            out_arr = np.where(ok, dist, fill_nd)

            _write_raster_array(
                output_img=output_img,
                arr=out_arr,
                ref_src=fdir_src,
                gdalformat=gdalformat,
                out_rio_dtype=out_rio_dt,
                out_np_dtype=out_np_dt,
                out_no_data_val=out_nd,
                calc_stats=calc_stats,
                use_pbar=use_pbar,
            )
            pbar.update(1)


def trace_flow_path(
    flow_dir_img: str,
    pour_points_vec: str,
    out_vec_file: str,
    direction: str = "downstream",
    flow_acc_img: Optional[str] = None,
    *,
    out_format: Optional[str] = None,
    out_vec_lyr: Optional[str] = None,
    overwrite: bool = True,
    no_data_val: Optional[float] = None,
    use_pbar: bool = True,
) -> None:
    """
    Trace D8 flow paths downstream to outlets or upstream along maximum
    accumulation to headwaters from vector seed points.

    Parameters
    ----------
    flow_dir_img : str
        Path to the input D8 flow direction raster (``uint8``).
    pour_points_vec : str
        Path to the input seed points vector file.
    out_vec_file : str
        Path to the output vector LineString dataset.
    direction : {'downstream', 'upstream'}, optional
        Tracing direction (Default = ``'downstream'``).
    flow_acc_img : str or None, optional
        Path to flow accumulation raster (required when ``direction='upstream'``).
    out_format : str or None, optional
        Optional output vector driver name (inferred from extension if ``None``).
    out_vec_lyr : str or None, optional
        Optional layer name for multi-layer formats (e.g., GPKG).
    overwrite : bool, optional
        Overwrite existing output vector file if ``True`` (Default = ``True``).
    no_data_val : float or None, optional
        Optional flow direction nodata override (Default = ``None``).
    use_pbar : bool, optional
        Display a progress bar (Default = ``True``).
    """
    # pylint: disable=too-many-arguments,too-many-locals,too-many-branches
    dir_norm = direction.strip().lower()
    if dir_norm not in ("downstream", "upstream"):
        raise RSGISPyException(
            f"Invalid direction '{direction}'. Must be 'downstream' or 'upstream'."
        )
    if dir_norm == "upstream" and flow_acc_img is None:
        raise RSGISPyException(
            "flow_acc_img must be provided when direction='upstream'."
        )
    ensure_file_exists(flow_dir_img, "Input flow direction image")
    ensure_file_exists(pour_points_vec, "Input pour points vector")
    if flow_acc_img is not None:
        ensure_file_exists(flow_acc_img, "Input flow accumulation image")

    pts_gdf = read_vec_gdf(pour_points_vec)
    if pts_gdf.empty:
        raise RSGISPyException("Input pour_points_vec contains no features.")

    with ExitStack() as stack:
        fdir_src = stack.enter_context(rasterio.open(flow_dir_img, "r"))
        _validate_projected_crs(fdir_src, flow_dir_img)
        fdir, valid = _read_d8_grid(fdir_src, no_data_val)
        height, width = fdir.shape
        transform = fdir_src.transform

        acc_arr: Optional[np.ndarray] = None
        acc_valid: Optional[np.ndarray] = None
        if dir_norm == "upstream":
            assert flow_acc_img is not None
            acc_src = stack.enter_context(rasterio.open(flow_acc_img, "r"))
            _validate_matching_grid(fdir_src, acc_src, flow_dir_img, flow_acc_img)
            acc_arr = acc_src.read(1, out_dtype=np.float64)
            acc_valid = build_valid_mask(acc_arr, acc_src.nodata)

        if pts_gdf.crs is not None and fdir_src.crs is not None:
            if pyproj.CRS.from_user_input(pts_gdf.crs) != pyproj.CRS.from_user_input(
                fdir_src.crs
            ):
                pts_gdf = pts_gdf.to_crs(fdir_src.crs)

        rev_map = np.array([16, 32, 64, 128, 1, 2, 4, 8], dtype=np.uint8)
        records: List[Dict[str, Any]] = []
        tqdm_cls = rsgispy.get_tqdm()
        attr_cols = [c for c in pts_gdf.columns if c != pts_gdf.active_geometry_name]

        for _, row_s in tqdm_cls(
            pts_gdf.iterrows(),
            total=len(pts_gdf),
            desc="Tracing flow paths",
            unit="pt",
            disable=not use_pbar,
        ):
            geom = row_s.geometry
            if geom is None or geom.is_empty:
                continue
            pt = geom.geoms[0] if geom.geom_type == "MultiPoint" else geom
            r0, c0 = fdir_src.index(pt.x, pt.y)
            if not (0 <= r0 < height and 0 <= c0 < width and valid[r0, c0]):
                continue

            visited = set()
            coords: List[Tuple[float, float]] = []
            curr_r, curr_c = int(r0), int(c0)
            while True:
                if (curr_r, curr_c) in visited:
                    break
                visited.add((curr_r, curr_c))
                px_x, px_y = rasterio.transform.xy(transform, curr_r, curr_c)
                coords.append((float(px_x), float(px_y)))

                if dir_norm == "downstream":
                    code = int(fdir[curr_r, curr_c])
                    offset = _D8_DIR_TO_OFFSET.get(code)
                    if offset is None:
                        break
                    nr = curr_r + offset[0]
                    nc = curr_c + offset[1]
                    if not (0 <= nr < height and 0 <= nc < width and valid[nr, nc]):
                        break
                    curr_r, curr_c = nr, nc
                else:
                    assert acc_arr is not None and acc_valid is not None
                    best_acc = -1.0
                    best_r = -1
                    best_c = -1
                    for k in range(8):
                        nr = curr_r + int(D8_DR[k])
                        nc = curr_c + int(D8_DC[k])
                        if 0 <= nr < height and 0 <= nc < width:
                            if valid[nr, nc] and acc_valid[nr, nc]:
                                if (nr, nc) not in visited and fdir[nr, nc] == rev_map[k]:
                                    if acc_arr[nr, nc] > best_acc:
                                        best_acc = float(acc_arr[nr, nc])
                                        best_r = nr
                                        best_c = nc
                    if best_r < 0:
                        break
                    curr_r, curr_c = best_r, best_c

            if len(coords) == 1:
                coords.append(coords[0])
            line_geom = LineString(coords)
            rec = {col: row_s[col] for col in attr_cols}
            rec["path_length_m"] = float(line_geom.length)
            rec["geometry"] = line_geom
            records.append(rec)

        out_gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=fdir_src.crs)
        write_vec_gdf(
            out_gdf,
            out_vec_file,
            out_vec_lyr=out_vec_lyr,
            out_format=out_format,
            overwrite=overwrite,
        )
