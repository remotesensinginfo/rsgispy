# pylint: disable=too-many-lines
"""
The elevation.hydrology module provides pure-Python hydrological conditioning,
flow routing, stream network extraction, watershed delineation, and terrain
index calculation functions using ``numpy``, ``rasterio``, ``geopandas``,
``shapely``, ``pyproj``, and optional ``scipy`` / ``numba`` acceleration.
"""

from contextlib import ExitStack
import math
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import geopandas as gpd
import numpy as np
import pyproj
import rasterio
import rasterio.features
from rasterio.transform import Affine
from shapely.geometry import LineString, Point, shape

try:
    from scipy.ndimage import distance_transform_edt  # pylint: disable=import-error
    from scipy.ndimage import label as nd_label  # pylint: disable=import-error

    HAVE_SCIPY = True
except ImportError:  # pragma: no cover
    distance_transform_edt = None
    nd_label = None
    HAVE_SCIPY = False

import rsgispy
from rsgispy import RSGISPyException
from rsgispy.elevation._hydro_kernels import (
    D8_CODES,
    D8_DC,
    D8_DR,
    apply_flat_routing_d8_kernel,
    breach_depressions_kernel,
    calc_ls_njit,
    calc_ls_numpy,
    calc_spi_njit,
    calc_spi_numpy,
    calc_twi_njit,
    calc_twi_numpy,
    extract_streams_njit,
    extract_streams_numpy,
    find_terminal_outlets_kernel,
    flood_inundation_raster_njit,
    flood_inundation_raster_numpy,
    flood_inundation_scalar_njit,
    flood_inundation_scalar_numpy,
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
    priority_flood_kernel,
    resolve_flats_kernel,
    reverse_bfs_label_kernel,
    single_cell_pits_kernel,
    stream_link_kernel,
    stream_order_kernel,
)
from rsgispy.imageprocess import (
    IntersectionGrid,
    build_valid_mask,
    cast_result_block,
    ensure_file_exists,
    open_output_dataset,
)
from rsgispy.imageutils.stats import pop_img_stats
from rsgispy.vectorprocess import read_vec_gdf, write_vec_gdf

try:
    import numba  # noqa: F401  # pylint: disable=unused-import

    HAS_NUMBA = True
except ImportError:  # pragma: no cover
    HAS_NUMBA = False


__all__ = [
    "HAS_NUMBA",
    "identify_sinks",
    "fill_depressions",
    "breach_depressions",
    "burn_streams",
    "resolve_flats",
    "condition_dem",
    "flow_direction",
    "flow_accumulation",
    "flow_length",
    "flow_distance",
    "trace_flow_path",
    "extract_streams",
    "stream_link",
    "stream_order",
    "vectorise_streams",
    "snap_pour_points",
    "delineate_watersheds",
    "delineate_basins",
    "delineate_subcatchments",
    "polygonise_watersheds",
    "calc_hand",
    "flood_inundation_from_hand",
    "calc_twi",
    "calc_spi",
    "calc_ls_factor",
]

_D8_DIR_TO_OFFSET: Dict[int, Tuple[int, int]] = {
    1: (0, 1),
    2: (1, 1),
    4: (1, 0),
    8: (1, -1),
    16: (0, -1),
    32: (-1, -1),
    64: (-1, 0),
    128: (-1, 1),
}

_D8_CODE_TO_ANGLE: Dict[int, float] = {
    1: 0.0,
    128: 0.25 * math.pi,
    64: 0.5 * math.pi,
    32: 0.75 * math.pi,
    16: math.pi,
    8: 1.25 * math.pi,
    4: 1.5 * math.pi,
    2: 1.75 * math.pi,
}

_D8_CODE_TO_IDX: Dict[int, int] = {int(code): idx for idx, code in enumerate(D8_CODES)}


# ---------------------------------------------------------------------------
# Shared validation, Numba dispatch, and raster/vector I/O helpers
# ---------------------------------------------------------------------------


def _call_topo_kernel(
    kernel: Any,
    allow_pure_python: bool,
    func_name: str,
    *args: Any,
) -> Any:
    """Execute a graph/PQ/topological kernel respecting ``HAS_NUMBA`` policy."""
    if HAS_NUMBA:
        return kernel(*args)
    if not allow_pure_python:
        raise RSGISPyException(
            f"numba is required to run '{func_name}' efficiently. "
            "Install numba (`pip install numba`) or pass "
            "`allow_pure_python=True` to run the slow pure-Python fallback."
        )
    warnings.warn(
        f"numba is not installed; running '{func_name}' in pure Python, "
        "which will be significantly slower on large rasters.",
        RuntimeWarning,
        stacklevel=3,
    )
    py_fn = getattr(kernel, "py_func", kernel)
    return py_fn(*args)


def _warn_numpy_fallback(func_name: str) -> None:
    """Emit a ``RuntimeWarning`` when falling back to vectorized NumPy."""
    warnings.warn(
        f"numba is not installed; running '{func_name}' using the vectorized "
        "NumPy fallback.",
        RuntimeWarning,
        stacklevel=3,
    )


def _validate_projected_crs(
    src: rasterio.DatasetReader, img_path: str
) -> Tuple[float, float]:
    """
    Validate that ``src`` uses a projected CRS with metric units and valid
    positive pixel dimensions, returning ``(dx, dy)`` in metres.
    """
    if src.crs is None:
        raise RSGISPyException(
            f"Input raster '{img_path}' has no CRS defined. A projected CRS "
            "with linear units in metres is required."
        )
    crs = pyproj.CRS.from_user_input(src.crs)
    if crs.is_geographic or not crs.is_projected:
        raise RSGISPyException(
            f"Input raster '{img_path}' uses a non-projected/geographic CRS "
            f"({crs.name}); a projected CRS with units in metres is required."
        )
    if crs.axis_info:
        factor = crs.axis_info[0].unit_conversion_factor
        if factor is not None and not math.isclose(factor, 1.0, rel_tol=1e-4, abs_tol=1e-4):
            raise RSGISPyException(
                f"Input raster '{img_path}' CRS '{crs.name}' linear unit is "
                f"'{crs.axis_info[0].unit_name}'; units in metres are required."
            )

    dx = abs(float(src.transform.a))
    dy = abs(float(src.transform.e))
    if not (math.isfinite(dx) and dx > 0.0 and math.isfinite(dy) and dy > 0.0):
        raise RSGISPyException(
            f"Input raster '{img_path}' has invalid pixel resolution: ({dx}, {dy})."
        )
    return dx, dy


def _validate_matching_grid(
    ref_src: rasterio.DatasetReader,
    other_src: rasterio.DatasetReader,
    ref_path: str,
    other_path: str,
) -> None:
    """Verify two open datasets share identical shape, CRS, and transform."""
    if ref_src.width != other_src.width or ref_src.height != other_src.height:
        raise RSGISPyException(
            f"Raster dimensions of '{other_path}' "
            f"({other_src.height}x{other_src.width}) do not match '{ref_path}' "
            f"({ref_src.height}x{ref_src.width})."
        )
    if (ref_src.crs is None) != (other_src.crs is None):
        raise RSGISPyException(
            f"CRS mismatch between '{ref_path}' and '{other_path}'."
        )
    if ref_src.crs is not None and other_src.crs is not None:
        if pyproj.CRS.from_user_input(ref_src.crs) != pyproj.CRS.from_user_input(
            other_src.crs
        ):
            raise RSGISPyException(
                f"CRS mismatch between '{ref_path}' ({ref_src.crs}) and "
                f"'{other_path}' ({other_src.crs})."
            )
    t_ref = np.array(
        [
            ref_src.transform.a,
            ref_src.transform.b,
            ref_src.transform.c,
            ref_src.transform.d,
            ref_src.transform.e,
            ref_src.transform.f,
        ],
        dtype=np.float64,
    )
    t_oth = np.array(
        [
            other_src.transform.a,
            other_src.transform.b,
            other_src.transform.c,
            other_src.transform.d,
            other_src.transform.e,
            other_src.transform.f,
        ],
        dtype=np.float64,
    )
    if not np.allclose(t_ref, t_oth, rtol=1e-5, atol=1e-6):
        raise RSGISPyException(
            f"Geotransform mismatch between '{ref_path}' and '{other_path}'."
        )


def _resolve_out_dtype(
    datatype: Optional[int], default_rsgis_dtype: int
) -> Tuple[int, str, np.dtype]:
    """Resolve ``(rsgis_dtype, rasterio_dtype_str, numpy_dtype)``."""
    eff_dt = default_rsgis_dtype if datatype is None else int(datatype)
    return (
        eff_dt,
        rsgispy.get_rasterio_datatype(eff_dt),
        np.dtype(rsgispy.get_numpy_datatype(eff_dt)),
    )


def _resolve_out_nodata(
    out_np_dtype: np.dtype,
    no_data_val: Optional[float],
    default_nodata: Optional[float],
) -> Optional[float]:
    """Return output nodata scalar compatible with ``out_np_dtype``."""
    cand = no_data_val if no_data_val is not None else default_nodata
    if cand is None:
        return None
    if np.issubdtype(out_np_dtype, np.integer):
        info = np.iinfo(out_np_dtype)
        val = int(round(float(cand)))
        if info.min <= val <= info.max:
            return float(val)
        if np.issubdtype(out_np_dtype, np.unsignedinteger):
            return float(info.max)
        return float(info.min)
    return float(cand)


def _write_raster_array(
    output_img: str,
    arr: np.ndarray,
    ref_src: rasterio.DatasetReader,
    gdalformat: str,
    out_rio_dtype: str,
    out_np_dtype: np.dtype,
    out_nodata: Optional[float],
    calc_stats: bool,
    use_pbar: bool,
) -> None:
    """Write a 2D or 3D array to disk using ``rsgispy`` conventions and stats."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    grid = IntersectionGrid.from_dataset(ref_src)
    if arr.ndim == 2:
        count = 1
        out_block = cast_result_block(arr, ref_src.height, ref_src.width, out_np_dtype)
        write_data = out_block[np.newaxis, :, :]
    else:
        count = int(arr.shape[0])
        write_data = np.empty(
            (count, ref_src.height, ref_src.width), dtype=out_np_dtype
        )
        for b_idx in range(count):
            write_data[b_idx] = cast_result_block(
                arr[b_idx], ref_src.height, ref_src.width, out_np_dtype
            )

    with ExitStack() as stack:
        dst = open_output_dataset(
            stack=stack,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype=out_rio_dtype,
            grid=grid,
            count=count,
        )
        if out_nodata is not None:
            dst.nodata = out_nodata
        dst.write(write_data)

    if calc_stats:
        pop_img_stats(
            input_img=output_img,
            no_data_val=out_nodata,
            calc_pyramids=True,
            ignore_cog=gdalformat.upper().strip() == "COG",
            use_pbar=use_pbar,
        )


def _validate_slope_units(slope_units: str) -> int:
    """Validate and map ``slope_units`` to integer code ``(0=deg, 1=rad, 2=pct)``."""
    norm = slope_units.strip().lower()
    if norm == "degrees":
        return 0
    if norm == "radians":
        return 1
    if norm == "percent":
        return 2
    raise RSGISPyException(
        f"Invalid slope_units '{slope_units}'. Must be 'degrees', 'radians', or 'percent'."
    )


def _agree_burn_array(
    dem: np.ndarray,
    valid: np.ndarray,
    streams_vec: str,
    dem_crs: Any,
    transform: Affine,
    dx: float,
    dy: float,
    burn_depth: float,
    buffer_dist: Optional[float],
    smooth_drop: float,
) -> np.ndarray:
    """Apply AGREE stream reconditioning in memory to a float64 DEM array."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    gdf = read_vec_gdf(streams_vec)
    if gdf.empty:
        return dem.copy()

    if gdf.crs is not None and dem_crs is not None:
        if pyproj.CRS.from_user_input(gdf.crs) != pyproj.CRS.from_user_input(dem_crs):
            gdf = gdf.to_crs(dem_crs)

    geoms = [g for g in gdf.geometry if g is not None and not g.is_empty]
    if not geoms:
        return dem.copy()

    stream_mask = rasterio.features.rasterize(
        [(g, 1) for g in geoms],
        out_shape=dem.shape,
        transform=transform,
        fill=0,
        dtype=np.uint8,
        all_touched=True,
    ) == 1

    out = dem.astype(np.float64).copy()
    has_buf = buffer_dist is not None and float(buffer_dist) > 0.0
    if has_buf:
        rsgispy.check_optional_dep(
            HAVE_SCIPY and distance_transform_edt is not None,
            "burn_streams (buffer_dist > 0)",
            "scipy",
            "stats",
        )
        buf_d = float(buffer_dist)  # type: ignore[arg-type]
        dist_m = distance_transform_edt(~stream_mask, sampling=(dy, dx))
        in_buf = valid & (~stream_mask) & (dist_m > 0.0) & (dist_m <= buf_d)
        if np.any(in_buf):
            out[in_buf] -= smooth_drop * (1.0 - dist_m[in_buf] / buf_d) ** 2

    drop_center = burn_depth + (smooth_drop if has_buf else 0.0)
    center_cells = valid & stream_mask
    out[center_cells] -= drop_center
    return out


def _read_d8_grid(
    src: rasterio.DatasetReader, no_data_val: Optional[float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Read a D8 flow direction raster and validate it is single-band integer."""
    if src.count != 1:
        raise RSGISPyException(
            f"D8 flow direction raster must have 1 band, found {src.count}."
        )
    dtype = np.dtype(src.dtypes[0])
    if not np.issubdtype(dtype, np.integer):
        raise RSGISPyException(
            f"Expected integer D8 flow direction raster, got dtype '{dtype}'."
        )
    eff_nd = no_data_val if no_data_val is not None else src.nodata
    if eff_nd is None:
        eff_nd = 255.0
    arr = src.read(1)
    valid = build_valid_mask(arr, eff_nd)
    return arr.astype(np.uint8, copy=False), valid


def _detect_flow_dir_mode(src: rasterio.DatasetReader) -> str:
    """Auto-detect ``'d8'``, ``'dinf'``, or ``'mfd'`` from raster band count/dtype."""
    if src.count == 8:
        return "mfd"
    if src.count != 1:
        raise RSGISPyException(
            f"Unsupported flow direction band count ({src.count}). "
            "Expected 1 band (D8 uint8 or D-Inf float32) or 8 bands (MFD float32)."
        )
    dtype = np.dtype(src.dtypes[0])
    if np.issubdtype(dtype, np.integer):
        return "d8"
    if np.issubdtype(dtype, np.floating):
        return "dinf"
    raise RSGISPyException(f"Unsupported flow direction dtype '{dtype}'.")


def _d8_flat_resolve_in_memory(
    dem: np.ndarray,
    valid: np.ndarray,
    dx: float,
    dy: float,
    allow_pure_python: bool,
    func_name: str,
) -> np.ndarray:
    """Compute D8 directions and route any flats via Barnes double-gradient."""
    if HAS_NUMBA:
        fdir = flow_dir_d8_njit(dem, valid, dx, dy)
    else:
        fdir = flow_dir_d8_numpy(dem, valid, dx, dy)

    if np.any(valid & (fdir == 0)):
        g_grid = _call_topo_kernel(
            resolve_flats_kernel, allow_pure_python, func_name, dem, valid
        )
        _call_topo_kernel(
            apply_flat_routing_d8_kernel,
            allow_pure_python,
            func_name,
            dem,
            valid,
            fdir,
            g_grid,
            dx,
            dy,
        )
    return fdir


def _run_progress_steps(
    desc: str, n_steps: int, use_pbar: bool
) -> Any:
    """Create a ``tqdm`` progress bar instance for stage tracking."""
    tqdm_cls = rsgispy.get_tqdm()
    return tqdm_cls(total=n_steps, desc=desc, unit="step", disable=not use_pbar)


# ---------------------------------------------------------------------------
# Stage 1: DEM Hydrological Conditioning
# ---------------------------------------------------------------------------


def identify_sinks(
    input_img: str,
    output_img: str,
    output_depth_img: Optional[str] = None,
    allow_pure_python: bool = False,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Identify and label contiguous topographic depressions (sinks) in a DEM
    without modifying ``input_img``.

    Runs Barnes et al. (2014) Priority-Flood in memory to locate cells where the
    filled elevation exceeds the original surface as well as isolated interior
    single-cell pits, labelling 8-connected sink regions with sequential IDs
    (``1..K``, ``0`` outside sinks, ``-1`` on nodata).

    Parameters
    ----------
    input_img : str
        Path to the input DEM raster.
    output_img : str
        Path to the output labelled sink region raster (``int32`` by default).
    output_depth_img : str or None, optional
        Optional output path for the continuous sink depth raster
        ``Z_filled - Z_orig`` (``float32``, ``0.0`` outside sinks).
    allow_pure_python : bool, optional
        If ``True``, allow falling back to pure Python with a
        ``RuntimeWarning`` when ``numba`` is unavailable (Default = ``False``).
    gdalformat : str, optional
        GDAL/rasterio output driver name (Default = ``'GTIFF'``).
    datatype : int or None, optional
        Optional ``rsgispy.TYPE_*`` constant for ``output_img``
        (Default = ``rsgispy.TYPE_32INT``).
    no_data_val : float or None, optional
        Optional input DEM nodata override (Default = ``None``, read from header).
    calc_stats : bool, optional
        If ``True``, compute output statistics and overview pyramids
        (Default = ``True``).
    use_pbar : bool, optional
        Display a progress bar (Default = ``True``).
    """
    # pylint: disable=too-many-arguments,too-many-locals
    rsgispy.check_optional_dep(
        HAVE_SCIPY and nd_label is not None,
        "identify_sinks",
        "scipy",
        "stats",
    )
    ensure_file_exists(input_img, "Input DEM image")
    eff_dt, out_rio_dt, out_np_dt = _resolve_out_dtype(datatype, rsgispy.TYPE_32INT)
    del eff_dt

    with _run_progress_steps("Identifying sinks", 3, use_pbar) as pbar:
        with rasterio.open(input_img, "r") as src:
            _validate_projected_crs(src, input_img)
            eff_nd = no_data_val if no_data_val is not None else src.nodata
            dem = src.read(1, out_dtype=np.float64)
            valid = build_valid_mask(dem, eff_nd)
            pbar.update(1)

            z_filled = _call_topo_kernel(
                priority_flood_kernel, allow_pure_python, "identify_sinks", dem, valid
            )
            single_pits = _call_topo_kernel(
                single_cell_pits_kernel, allow_pure_python, "identify_sinks", dem, valid
            )
            sink_mask = valid & ((z_filled > dem) | single_pits)

            labelled, _ = nd_label(sink_mask, structure=np.ones((3, 3), dtype=np.int32))
            out_nd = _resolve_out_nodata(out_np_dt, None, -1.0)
            fill_nd = out_nd if out_nd is not None else -1.0
            sink_out = np.where(valid, labelled.astype(np.int32), int(fill_nd))
            pbar.update(1)

            _write_raster_array(
                output_img=output_img,
                arr=sink_out,
                ref_src=src,
                gdalformat=gdalformat,
                out_rio_dtype=out_rio_dt,
                out_np_dtype=out_np_dt,
                out_nodata=out_nd,
                calc_stats=calc_stats,
                use_pbar=use_pbar,
            )

            if output_depth_img is not None:
                depth_np_dt = np.dtype(rsgispy.get_numpy_datatype(rsgispy.TYPE_32FLOAT))
                depth_rio_dt = rsgispy.get_rasterio_datatype(rsgispy.TYPE_32FLOAT)
                depth_nd = -9999.0 if eff_nd is None else float(eff_nd)
                depth_arr = np.where(
                    valid,
                    np.maximum(0.0, z_filled - dem),
                    depth_nd,
                ).astype(np.float32)
                _write_raster_array(
                    output_img=output_depth_img,
                    arr=depth_arr,
                    ref_src=src,
                    gdalformat=gdalformat,
                    out_rio_dtype=depth_rio_dt,
                    out_np_dtype=depth_np_dt,
                    out_nodata=depth_nd,
                    calc_stats=calc_stats,
                    use_pbar=use_pbar,
                )
            pbar.update(1)


def fill_depressions(
    input_img: str,
    output_img: str,
    resolve_flats: bool = True,
    allow_pure_python: bool = False,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Fill all topographic depressions in a DEM using Barnes et al. (2014)
    Priority-Flood with a secondary pit FIFO queue.

    Executes internally in ``float64`` without adding floating-point epsilon
    increments so filled depressions remain exact flats (which are routed
    cleanly by :func:`resolve_flats` or :func:`flow_direction`).

    Parameters
    ----------
    input_img : str
        Path to the input DEM raster.
    output_img : str
        Path to the output hydrologically filled DEM raster.
    resolve_flats : bool, optional
        Retained for workflow parity; flat routing is delegated to
        :func:`resolve_flats` / :func:`flow_direction` without epsilon
        perturbations (Default = ``True``).
    allow_pure_python : bool, optional
        If ``True``, allow falling back to pure Python with a
        ``RuntimeWarning`` when ``numba`` is unavailable (Default = ``False``).
    gdalformat : str, optional
        GDAL/rasterio output driver name (Default = ``'GTIFF'``).
    datatype : int or None, optional
        Optional ``rsgispy.TYPE_*`` constant (preserves input DEM dtype when
        ``None``).
    no_data_val : float or None, optional
        Optional nodata override (Default = ``None``, read from header).
    calc_stats : bool, optional
        If ``True``, compute output statistics and overview pyramids
        (Default = ``True``).
    use_pbar : bool, optional
        Display a progress bar (Default = ``True``).
    """
    # pylint: disable=too-many-arguments,too-many-locals,redefined-outer-name
    del resolve_flats
    ensure_file_exists(input_img, "Input DEM image")

    with _run_progress_steps("Filling depressions", 3, use_pbar) as pbar:
        with rasterio.open(input_img, "r") as src:
            _validate_projected_crs(src, input_img)
            in_rsgis_dt = rsgispy.get_rsgis_datatype(src.dtypes[0])
            _, out_rio_dt, out_np_dt = _resolve_out_dtype(datatype, in_rsgis_dt)

            eff_nd = no_data_val if no_data_val is not None else src.nodata
            dem = src.read(1, out_dtype=np.float64)
            valid = build_valid_mask(dem, eff_nd)
            pbar.update(1)

            z_filled = _call_topo_kernel(
                priority_flood_kernel, allow_pure_python, "fill_depressions", dem, valid
            )
            pbar.update(1)

            out_nd = _resolve_out_nodata(out_np_dt, no_data_val, src.nodata)
            fill_nd = out_nd if out_nd is not None else (
                float("nan") if np.issubdtype(out_np_dt, np.floating) else 0.0
            )
            out_arr = np.where(valid, z_filled, fill_nd)

            _write_raster_array(
                output_img=output_img,
                arr=out_arr,
                ref_src=src,
                gdalformat=gdalformat,
                out_rio_dtype=out_rio_dt,
                out_np_dtype=out_np_dt,
                out_nodata=out_nd,
                calc_stats=calc_stats,
                use_pbar=use_pbar,
            )
            pbar.update(1)


def breach_depressions(
    input_img: str,
    output_img: str,
    max_depth: Optional[float] = None,
    max_length: Optional[int] = None,
    fill_remaining: bool = True,
    allow_pure_python: bool = False,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Breach topographic depressions using Lindsay (2016) constrained least-cost
    channel carving.

    Parameters
    ----------
    input_img : str
        Path to the input DEM raster.
    output_img : str
        Path to the output breached DEM raster.
    max_depth : float or None, optional
        Maximum allowed embankment breach cut depth in elevation units
        (Default = ``None``, unconstrained).
    max_length : int or None, optional
        Maximum allowed breach trench length in pixel steps
        (Default = ``None``, unconstrained).
    fill_remaining : bool, optional
        If ``True``, run Priority-Flood on the breached surface to fill any
        residual deep depressions that exceeded constraints (Default = ``True``).
    allow_pure_python : bool, optional
        If ``True``, allow falling back to pure Python with a
        ``RuntimeWarning`` when ``numba`` is unavailable (Default = ``False``).
    gdalformat : str, optional
        GDAL/rasterio output driver name (Default = ``'GTIFF'``).
    datatype : int or None, optional
        Optional ``rsgispy.TYPE_*`` constant (defaults to input DEM float dtype
        or ``TYPE_32FLOAT`` for integer inputs).
    no_data_val : float or None, optional
        Optional nodata override (Default = ``None``, read from header).
    calc_stats : bool, optional
        If ``True``, compute output statistics and overview pyramids
        (Default = ``True``).
    use_pbar : bool, optional
        Display a progress bar (Default = ``True``).
    """
    # pylint: disable=too-many-arguments,too-many-locals
    if max_depth is not None and max_depth < 0.0:
        raise RSGISPyException("max_depth must be non-negative.")
    if max_length is not None and max_length < 1:
        raise RSGISPyException("max_length must be at least 1 pixel step.")
    ensure_file_exists(input_img, "Input DEM image")

    with _run_progress_steps("Breaching depressions", 3, use_pbar) as pbar:
        with rasterio.open(input_img, "r") as src:
            dx, dy = _validate_projected_crs(src, input_img)
            in_rsgis_dt = rsgispy.get_rsgis_datatype(src.dtypes[0])
            default_dt = (
                in_rsgis_dt
                if np.issubdtype(np.dtype(src.dtypes[0]), np.floating)
                else rsgispy.TYPE_32FLOAT
            )
            _, out_rio_dt, out_np_dt = _resolve_out_dtype(datatype, default_dt)

            eff_nd = no_data_val if no_data_val is not None else src.nodata
            dem = src.read(1, out_dtype=np.float64)
            valid = build_valid_mask(dem, eff_nd)
            pbar.update(1)

            z_breached = _call_topo_kernel(
                breach_depressions_kernel,
                allow_pure_python,
                "breach_depressions",
                dem,
                valid,
                dx,
                dy,
                0.0 if max_depth is None else float(max_depth),
                max_depth is not None,
                0 if max_length is None else int(max_length),
                max_length is not None,
            )
            if fill_remaining:
                z_breached = _call_topo_kernel(
                    priority_flood_kernel,
                    allow_pure_python,
                    "breach_depressions",
                    z_breached,
                    valid,
                )
            pbar.update(1)

            out_nd = _resolve_out_nodata(
                out_np_dt, no_data_val, src.nodata if src.nodata is not None else -9999.0
            )
            fill_nd = out_nd if out_nd is not None else -9999.0
            out_arr = np.where(valid, z_breached, fill_nd)

            _write_raster_array(
                output_img=output_img,
                arr=out_arr,
                ref_src=src,
                gdalformat=gdalformat,
                out_rio_dtype=out_rio_dt,
                out_np_dtype=out_np_dt,
                out_nodata=out_nd,
                calc_stats=calc_stats,
                use_pbar=use_pbar,
            )
            pbar.update(1)


def burn_streams(
    dem_img: str,
    streams_vec: str,
    output_img: str,
    burn_depth: float = 5.0,
    buffer_dist: Optional[float] = None,
    smooth_drop: float = 2.0,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Recondition a DEM using the AGREE (Hellweger, 1997) stream burning method.

    Parameters
    ----------
    dem_img : str
        Path to the input DEM raster.
    streams_vec : str
        Path to the input stream network vector file.
    output_img : str
        Path to the output stream-burned DEM raster.
    burn_depth : float, optional
        Sharp vertical drop carved directly along stream centreline pixels
        (Default = ``5.0``).
    buffer_dist : float or None, optional
        Optional Euclidean buffer decay distance in metres around streams
        (Default = ``None``, sharp burn only).
    smooth_drop : float, optional
        Maximum smooth buffer drop applied across ``buffer_dist``
        (Default = ``2.0``).
    gdalformat : str, optional
        GDAL/rasterio output driver name (Default = ``'GTIFF'``).
    datatype : int or None, optional
        Optional ``rsgispy.TYPE_*`` constant (defaults to input DEM float dtype
        or ``TYPE_32FLOAT`` for integer inputs).
    no_data_val : float or None, optional
        Optional nodata override (Default = ``None``, read from header).
    calc_stats : bool, optional
        If ``True``, compute output statistics and overview pyramids
        (Default = ``True``).
    use_pbar : bool, optional
        Display a progress bar (Default = ``True``).
    """
    # pylint: disable=too-many-arguments,too-many-locals
    if buffer_dist is not None and buffer_dist < 0.0:
        raise RSGISPyException("buffer_dist must be non-negative.")
    ensure_file_exists(dem_img, "Input DEM image")
    ensure_file_exists(streams_vec, "Input streams vector")

    with _run_progress_steps("Burning streams", 3, use_pbar) as pbar:
        with rasterio.open(dem_img, "r") as src:
            dx, dy = _validate_projected_crs(src, dem_img)
            in_rsgis_dt = rsgispy.get_rsgis_datatype(src.dtypes[0])
            default_dt = (
                in_rsgis_dt
                if np.issubdtype(np.dtype(src.dtypes[0]), np.floating)
                else rsgispy.TYPE_32FLOAT
            )
            _, out_rio_dt, out_np_dt = _resolve_out_dtype(datatype, default_dt)

            eff_nd = no_data_val if no_data_val is not None else src.nodata
            dem = src.read(1, out_dtype=np.float64)
            valid = build_valid_mask(dem, eff_nd)
            pbar.update(1)

            burned = _agree_burn_array(
                dem=dem,
                valid=valid,
                streams_vec=streams_vec,
                dem_crs=src.crs,
                transform=src.transform,
                dx=dx,
                dy=dy,
                burn_depth=burn_depth,
                buffer_dist=buffer_dist,
                smooth_drop=smooth_drop,
            )
            pbar.update(1)

            out_nd = _resolve_out_nodata(
                out_np_dt, no_data_val, src.nodata if src.nodata is not None else -9999.0
            )
            fill_nd = out_nd if out_nd is not None else -9999.0
            out_arr = np.where(valid, burned, fill_nd)

            _write_raster_array(
                output_img=output_img,
                arr=out_arr,
                ref_src=src,
                gdalformat=gdalformat,
                out_rio_dtype=out_rio_dt,
                out_np_dtype=out_np_dt,
                out_nodata=out_nd,
                calc_stats=calc_stats,
                use_pbar=use_pbar,
            )
            pbar.update(1)


def resolve_flats(
    input_img: str,
    output_img: str,
    allow_pure_python: bool = False,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Compute the Barnes et al. (2014) integer double-gradient surface over flat
    areas in a DEM without modifying elevation values.

    Drainable flat cells receive positive integer values decreasing strictly
    toward their outlets; non-flat valid cells receive ``0``; nodata cells
    receive ``-1``.

    Parameters
    ----------
    input_img : str
        Path to the input DEM raster.
    output_img : str
        Path to the output flat-resolution gradient raster (``int32``).
    allow_pure_python : bool, optional
        If ``True``, allow falling back to pure Python with a
        ``RuntimeWarning`` when ``numba`` is unavailable (Default = ``False``).
    gdalformat : str, optional
        GDAL/rasterio output driver name (Default = ``'GTIFF'``).
    datatype : int or None, optional
        Optional ``rsgispy.TYPE_*`` constant (Default = ``rsgispy.TYPE_32INT``).
    no_data_val : float or None, optional
        Optional input DEM nodata override (Default = ``None``, read from header).
    calc_stats : bool, optional
        If ``True``, compute output statistics and overview pyramids
        (Default = ``True``).
    use_pbar : bool, optional
        Display a progress bar (Default = ``True``).
    """
    # pylint: disable=too-many-arguments,too-many-locals
    ensure_file_exists(input_img, "Input DEM image")
    _, out_rio_dt, out_np_dt = _resolve_out_dtype(datatype, rsgispy.TYPE_32INT)

    with _run_progress_steps("Resolving flats", 3, use_pbar) as pbar:
        with rasterio.open(input_img, "r") as src:
            _validate_projected_crs(src, input_img)
            eff_nd = no_data_val if no_data_val is not None else src.nodata
            dem = src.read(1, out_dtype=np.float64)
            valid = build_valid_mask(dem, eff_nd)
            pbar.update(1)

            g_grid = _call_topo_kernel(
                resolve_flats_kernel, allow_pure_python, "resolve_flats", dem, valid
            )
            pbar.update(1)

            out_nd = _resolve_out_nodata(out_np_dt, None, -1.0)
            if out_nd is not None and int(round(out_nd)) != -1:
                g_grid = np.where(valid, g_grid, int(round(out_nd)))

            _write_raster_array(
                output_img=output_img,
                arr=g_grid,
                ref_src=src,
                gdalformat=gdalformat,
                out_rio_dtype=out_rio_dt,
                out_np_dtype=out_np_dt,
                out_nodata=out_nd,
                calc_stats=calc_stats,
                use_pbar=use_pbar,
            )
            pbar.update(1)


def condition_dem(
    input_img: str,
    output_img: str,
    method: str = "breach_and_fill",
    max_breach_depth: Optional[float] = None,
    max_breach_length: Optional[int] = None,
    streams_vec: Optional[str] = None,
    burn_depth: float = 5.0,
    buffer_dist: Optional[float] = None,
    allow_pure_python: bool = False,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Run a complete in-memory DEM hydrological conditioning pipeline (optional
    AGREE stream burning followed by breaching and/or Priority-Flood filling).

    Parameters
    ----------
    input_img : str
        Path to the input DEM raster.
    output_img : str
        Path to the output hydrologically conditioned DEM raster.
    method : {'breach_and_fill', 'fill', 'breach'}, optional
        Conditioning strategy (Default = ``'breach_and_fill'``).
    max_breach_depth : float or None, optional
        Maximum breach depth passed to depression breaching (Default = ``None``).
    max_breach_length : int or None, optional
        Maximum breach length in pixel steps (Default = ``None``).
    streams_vec : str or None, optional
        Optional vector stream network for AGREE stream burning (Default = ``None``).
    burn_depth : float, optional
        Stream centreline burn depth when ``streams_vec`` is set (Default = ``5.0``).
    buffer_dist : float or None, optional
        Optional AGREE smooth buffer distance in metres (Default = ``None``).
    allow_pure_python : bool, optional
        If ``True``, allow falling back to pure Python with a
        ``RuntimeWarning`` when ``numba`` is unavailable (Default = ``False``).
    gdalformat : str, optional
        GDAL/rasterio output driver name (Default = ``'GTIFF'``).
    datatype : int or None, optional
        Optional ``rsgispy.TYPE_*`` constant (defaults to input DEM float dtype
        or ``TYPE_32FLOAT`` for integer inputs).
    no_data_val : float or None, optional
        Optional nodata override (Default = ``None``, read from header).
    calc_stats : bool, optional
        If ``True``, compute output statistics and overview pyramids
        (Default = ``True``).
    use_pbar : bool, optional
        Display a progress bar (Default = ``True``).
    """
    # pylint: disable=too-many-arguments,too-many-locals
    method_norm = method.strip().lower()
    if method_norm not in ("breach_and_fill", "fill", "breach"):
        raise RSGISPyException(
            f"Invalid conditioning method '{method}'. "
            "Must be 'breach_and_fill', 'fill', or 'breach'."
        )
    ensure_file_exists(input_img, "Input DEM image")
    if streams_vec is not None:
        ensure_file_exists(streams_vec, "Input streams vector")

    with _run_progress_steps("Conditioning DEM", 4, use_pbar) as pbar:
        with rasterio.open(input_img, "r") as src:
            dx, dy = _validate_projected_crs(src, input_img)
            in_rsgis_dt = rsgispy.get_rsgis_datatype(src.dtypes[0])
            default_dt = (
                in_rsgis_dt
                if np.issubdtype(np.dtype(src.dtypes[0]), np.floating)
                else rsgispy.TYPE_32FLOAT
            )
            _, out_rio_dt, out_np_dt = _resolve_out_dtype(datatype, default_dt)

            eff_nd = no_data_val if no_data_val is not None else src.nodata
            dem = src.read(1, out_dtype=np.float64)
            valid = build_valid_mask(dem, eff_nd)
            pbar.update(1)

            if streams_vec is not None:
                dem = _agree_burn_array(
                    dem=dem,
                    valid=valid,
                    streams_vec=streams_vec,
                    dem_crs=src.crs,
                    transform=src.transform,
                    dx=dx,
                    dy=dy,
                    burn_depth=burn_depth,
                    buffer_dist=buffer_dist,
                    smooth_drop=2.0,
                )
            pbar.update(1)

            if method_norm in ("breach_and_fill", "breach"):
                dem = _call_topo_kernel(
                    breach_depressions_kernel,
                    allow_pure_python,
                    "condition_dem",
                    dem,
                    valid,
                    dx,
                    dy,
                    0.0 if max_breach_depth is None else float(max_breach_depth),
                    max_breach_depth is not None,
                    0 if max_breach_length is None else int(max_breach_length),
                    max_breach_length is not None,
                )
            if method_norm in ("breach_and_fill", "fill"):
                dem = _call_topo_kernel(
                    priority_flood_kernel,
                    allow_pure_python,
                    "condition_dem",
                    dem,
                    valid,
                )
            pbar.update(1)

            out_nd = _resolve_out_nodata(
                out_np_dt, no_data_val, src.nodata if src.nodata is not None else -9999.0
            )
            fill_nd = out_nd if out_nd is not None else -9999.0
            out_arr = np.where(valid, dem, fill_nd)

            _write_raster_array(
                output_img=output_img,
                arr=out_arr,
                ref_src=src,
                gdalformat=gdalformat,
                out_rio_dtype=out_rio_dt,
                out_np_dtype=out_np_dt,
                out_nodata=out_nd,
                calc_stats=calc_stats,
                use_pbar=use_pbar,
            )
            pbar.update(1)


# ---------------------------------------------------------------------------
# Stage 2: Flow Routing & Accumulation
# ---------------------------------------------------------------------------


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
    # pylint: disable=too-many-arguments,too-many-locals,too-many-branches,unused-argument
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
                if HAS_NUMBA:
                    out_arr = flow_dir_d8_njit(dem, valid, dx, dy)
                else:
                    _warn_numpy_fallback("flow_direction")
                    out_arr = flow_dir_d8_numpy(dem, valid, dx, dy)

                if flat_routing and np.any(valid & (out_arr == 0)):
                    if HAS_NUMBA:
                        g_grid = resolve_flats_kernel(dem, valid)
                        apply_flat_routing_d8_kernel(dem, valid, out_arr, g_grid, dx, dy)
                    else:
                        g_fn = getattr(resolve_flats_kernel, "py_func", resolve_flats_kernel)
                        a_fn = getattr(
                            apply_flat_routing_d8_kernel,
                            "py_func",
                            apply_flat_routing_d8_kernel,
                        )
                        g_grid = g_fn(dem, valid)
                        a_fn(dem, valid, out_arr, g_grid, dx, dy)

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
                if HAS_NUMBA:
                    out_arr = flow_dir_dinf_njit(dem, valid, dx, dy, fill_nd)
                else:
                    _warn_numpy_fallback("flow_direction")
                    out_arr = flow_dir_dinf_numpy(dem, valid, dx, dy, fill_nd)

                if flat_routing and np.any(valid & (out_arr == -1.0)):
                    fdir_d8 = _d8_flat_resolve_in_memory(
                        dem, valid, dx, dy, True, "flow_direction"
                    )
                    flat_pts = valid & (out_arr == -1.0) & (fdir_d8 > 0)
                    if np.any(flat_pts):
                        rr, cc = np.where(flat_pts)
                        for r_i, c_i in zip(rr, cc):
                            out_arr[r_i, c_i] = np.float32(
                                _D8_CODE_TO_ANGLE[int(fdir_d8[r_i, c_i])]
                            )

            else:  # mfd
                _, out_rio_dt, out_np_dt = _resolve_out_dtype(
                    datatype, rsgispy.TYPE_32FLOAT
                )
                out_nd = _resolve_out_nodata(out_np_dt, None, -9999.0)
                fill_nd = -9999.0 if out_nd is None else float(out_nd)
                if HAS_NUMBA:
                    out_arr = flow_dir_mfd_njit(
                        dem, valid, dx, dy, float(mfd_exponent), fill_nd
                    )
                else:
                    _warn_numpy_fallback("flow_direction")
                    out_arr = flow_dir_mfd_numpy(
                        dem, valid, dx, dy, float(mfd_exponent), fill_nd
                    )

                if flat_routing:
                    band_sum = np.sum(out_arr, axis=0)
                    unresolved = valid & (band_sum == 0.0)
                    if np.any(unresolved):
                        fdir_d8 = _d8_flat_resolve_in_memory(
                            dem, valid, dx, dy, True, "flow_direction"
                        )
                        flat_pts = unresolved & (fdir_d8 > 0)
                        if np.any(flat_pts):
                            rr, cc = np.where(flat_pts)
                            for r_i, c_i in zip(rr, cc):
                                k_idx = _D8_CODE_TO_IDX[int(fdir_d8[r_i, c_i])]
                                out_arr[k_idx, r_i, c_i] = np.float32(1.0)

            pbar.update(1)

            _write_raster_array(
                output_img=output_img,
                arr=out_arr,
                ref_src=src,
                gdalformat=gdalformat,
                out_rio_dtype=out_rio_dt,
                out_np_dtype=out_np_dt,
                out_nodata=out_nd,
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
                out_nodata=out_nd,
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
                out_nodata=out_nd,
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
                out_nodata=out_nd,
                calc_stats=calc_stats,
                use_pbar=use_pbar,
            )
            pbar.update(1)


def trace_flow_path(
    flow_dir_img: str,
    pour_points_vec: str,
    output_vec: str,
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
    output_vec : str
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
            output_vec,
            out_vec_lyr=out_vec_lyr,
            out_format=out_format,
            overwrite=overwrite,
        )


# ---------------------------------------------------------------------------
# Stage 3: Stream Network Extraction & Topology
# ---------------------------------------------------------------------------


def extract_streams(
    input_img: str,
    output_img: str,
    threshold: float,
    threshold_units: str = "cells",
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Extract a binary stream network raster from a flow accumulation grid.

    Parameters
    ----------
    input_img : str
        Path to the input flow accumulation raster.
    output_img : str
        Path to the output binary stream mask raster (``uint8``, ``1`` = stream,
        ``0`` = hillslope, ``255`` = nodata).
    threshold : float
        Minimum contributing area threshold for channel initiation.
    threshold_units : {'cells', 'area'}, optional
        Units of ``threshold``: pixel count (``'cells'``) or square metres
        (``'area'``) (Default = ``'cells'``).
    gdalformat : str, optional
        GDAL/rasterio output driver name (Default = ``'GTIFF'``).
    datatype : int or None, optional
        Optional ``rsgispy.TYPE_*`` constant (Default = ``rsgispy.TYPE_8UINT``).
    no_data_val : float or None, optional
        Optional input flow accumulation nodata override (Default = ``None``).
    calc_stats : bool, optional
        If ``True``, compute output statistics and overview pyramids
        (Default = ``True``).
    use_pbar : bool, optional
        Display a progress bar (Default = ``True``).
    """
    # pylint: disable=too-many-arguments,too-many-locals
    units_norm = threshold_units.strip().lower()
    if units_norm not in ("cells", "area"):
        raise RSGISPyException(
            f"Invalid threshold_units '{threshold_units}'. Must be 'cells' or 'area'."
        )
    if threshold <= 0.0:
        raise RSGISPyException("threshold must be strictly positive.")
    ensure_file_exists(input_img, "Input flow accumulation image")

    _, out_rio_dt, out_np_dt = _resolve_out_dtype(datatype, rsgispy.TYPE_8UINT)
    out_nd = _resolve_out_nodata(out_np_dt, None, 255.0)
    fill_nd_int = int(round(out_nd)) if out_nd is not None else 255

    with _run_progress_steps("Extracting streams", 3, use_pbar) as pbar:
        with rasterio.open(input_img, "r") as src:
            dx, dy = _validate_projected_crs(src, input_img)
            cell_thresh = (
                float(threshold) / (dx * dy)
                if units_norm == "area"
                else float(threshold)
            )
            eff_nd = no_data_val if no_data_val is not None else src.nodata
            acc = src.read(1, out_dtype=np.float64)
            valid = build_valid_mask(acc, eff_nd)
            pbar.update(1)

            if HAS_NUMBA:
                streams = extract_streams_njit(acc, valid, cell_thresh, fill_nd_int)
            else:
                _warn_numpy_fallback("extract_streams")
                streams = extract_streams_numpy(acc, valid, cell_thresh, fill_nd_int)
            pbar.update(1)

            _write_raster_array(
                output_img=output_img,
                arr=streams,
                ref_src=src,
                gdalformat=gdalformat,
                out_rio_dtype=out_rio_dt,
                out_np_dtype=out_np_dt,
                out_nodata=out_nd,
                calc_stats=calc_stats,
                use_pbar=use_pbar,
            )
            pbar.update(1)


def stream_link(
    streams_img: str,
    flow_dir_img: str,
    output_img: str,
    allow_pure_python: bool = False,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Assign unique integer reach IDs (``1..K``) to every topologically distinct
    stream segment between sources, confluences, and outlets.

    Parameters
    ----------
    streams_img : str
        Path to the binary stream network mask raster (``1`` = stream).
    flow_dir_img : str
        Path to the D8 flow direction raster (``uint8``).
    output_img : str
        Path to the output stream link raster (``int32``).
    allow_pure_python : bool, optional
        If ``True``, allow falling back to pure Python with a
        ``RuntimeWarning`` when ``numba`` is unavailable (Default = ``False``).
    gdalformat : str, optional
        GDAL/rasterio output driver name (Default = ``'GTIFF'``).
    datatype : int or None, optional
        Optional ``rsgispy.TYPE_*`` constant (Default = ``rsgispy.TYPE_32INT``).
    no_data_val : float or None, optional
        Optional flow direction nodata override (Default = ``None``).
    calc_stats : bool, optional
        If ``True``, compute output statistics and overview pyramids
        (Default = ``True``).
    use_pbar : bool, optional
        Display a progress bar (Default = ``True``).
    """
    # pylint: disable=too-many-arguments,too-many-locals
    ensure_file_exists(streams_img, "Input streams image")
    ensure_file_exists(flow_dir_img, "Input flow direction image")
    _, out_rio_dt, out_np_dt = _resolve_out_dtype(datatype, rsgispy.TYPE_32INT)

    with _run_progress_steps("Linking stream reaches", 3, use_pbar) as pbar:
        with ExitStack() as stack:
            str_src = stack.enter_context(rasterio.open(streams_img, "r"))
            fdir_src = stack.enter_context(rasterio.open(flow_dir_img, "r"))
            _validate_projected_crs(fdir_src, flow_dir_img)
            _validate_matching_grid(fdir_src, str_src, flow_dir_img, streams_img)

            fdir, valid = _read_d8_grid(fdir_src, no_data_val)
            streams = str_src.read(1)
            valid &= build_valid_mask(streams, str_src.nodata)
            streams_bin = (streams == 1).astype(np.uint8)
            pbar.update(1)

            link_grid, _ = _call_topo_kernel(
                stream_link_kernel,
                allow_pure_python,
                "stream_link",
                fdir,
                streams_bin,
                valid,
            )
            pbar.update(1)

            out_nd = _resolve_out_nodata(out_np_dt, None, -1.0)
            if out_nd is not None and int(round(out_nd)) != -1:
                link_grid = np.where(valid, link_grid, int(round(out_nd)))

            _write_raster_array(
                output_img=output_img,
                arr=link_grid,
                ref_src=fdir_src,
                gdalformat=gdalformat,
                out_rio_dtype=out_rio_dt,
                out_np_dtype=out_np_dt,
                out_nodata=out_nd,
                calc_stats=calc_stats,
                use_pbar=use_pbar,
            )
            pbar.update(1)


def stream_order(
    streams_img: str,
    flow_dir_img: str,
    output_img: str,
    method: str = "strahler",
    allow_pure_python: bool = False,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Compute Strahler or Shreve topological stream order along a D8 stream
    network.

    Parameters
    ----------
    streams_img : str
        Path to the binary stream network mask raster (``1`` = stream).
    flow_dir_img : str
        Path to the D8 flow direction raster (``uint8``).
    output_img : str
        Path to the output stream order raster (``int32``).
    method : {'strahler', 'shreve'}, optional
        Stream ordering scheme (Default = ``'strahler'``).
    allow_pure_python : bool, optional
        If ``True``, allow falling back to pure Python with a
        ``RuntimeWarning`` when ``numba`` is unavailable (Default = ``False``).
    gdalformat : str, optional
        GDAL/rasterio output driver name (Default = ``'GTIFF'``).
    datatype : int or None, optional
        Optional ``rsgispy.TYPE_*`` constant (Default = ``rsgispy.TYPE_32INT``).
    no_data_val : float or None, optional
        Optional flow direction nodata override (Default = ``None``).
    calc_stats : bool, optional
        If ``True``, compute output statistics and overview pyramids
        (Default = ``True``).
    use_pbar : bool, optional
        Display a progress bar (Default = ``True``).
    """
    # pylint: disable=too-many-arguments,too-many-locals
    method_norm = method.strip().lower()
    if method_norm not in ("strahler", "shreve"):
        raise RSGISPyException(
            f"Invalid stream ordering method '{method}'. Must be 'strahler' or 'shreve'."
        )
    ensure_file_exists(streams_img, "Input streams image")
    ensure_file_exists(flow_dir_img, "Input flow direction image")
    _, out_rio_dt, out_np_dt = _resolve_out_dtype(datatype, rsgispy.TYPE_32INT)

    with _run_progress_steps("Computing stream order", 3, use_pbar) as pbar:
        with ExitStack() as stack:
            str_src = stack.enter_context(rasterio.open(streams_img, "r"))
            fdir_src = stack.enter_context(rasterio.open(flow_dir_img, "r"))
            _validate_projected_crs(fdir_src, flow_dir_img)
            _validate_matching_grid(fdir_src, str_src, flow_dir_img, streams_img)

            fdir, valid = _read_d8_grid(fdir_src, no_data_val)
            streams = str_src.read(1)
            valid &= build_valid_mask(streams, str_src.nodata)
            streams_bin = (streams == 1).astype(np.uint8)
            pbar.update(1)

            ord_grid = _call_topo_kernel(
                stream_order_kernel,
                allow_pure_python,
                "stream_order",
                fdir,
                streams_bin,
                valid,
                method_norm == "strahler",
            )
            pbar.update(1)

            out_nd = _resolve_out_nodata(out_np_dt, None, -1.0)
            if out_nd is not None and int(round(out_nd)) != -1:
                ord_grid = np.where(valid, ord_grid, int(round(out_nd)))

            _write_raster_array(
                output_img=output_img,
                arr=ord_grid,
                ref_src=fdir_src,
                gdalformat=gdalformat,
                out_rio_dtype=out_rio_dt,
                out_np_dtype=out_np_dt,
                out_nodata=out_nd,
                calc_stats=calc_stats,
                use_pbar=use_pbar,
            )
            pbar.update(1)


def vectorise_streams(
    streams_img: str,
    flow_dir_img: str,
    output_vec: str,
    dem_img: Optional[str] = None,
    flow_acc_img: Optional[str] = None,
    order_method: str = "strahler",
    allow_pure_python: bool = False,
    *,
    out_format: Optional[str] = None,
    out_vec_lyr: Optional[str] = None,
    overwrite: bool = True,
    no_data_val: Optional[float] = None,
    use_pbar: bool = True,
) -> None:
    """
    Vectorise a raster stream network into a topologically connected vector
    dataset with Strahler/Shreve orders, link connectivity, and optional
    elevation/accumulation attributes.

    Parameters
    ----------
    streams_img : str
        Path to the binary stream network mask raster (``1`` = stream).
    flow_dir_img : str
        Path to the D8 flow direction raster (``uint8``).
    output_vec : str
        Path to the output vector stream file (e.g., ``.gpkg``, ``.parquet``,
        ``.shp``, ``.geojson``).
    dem_img : str or None, optional
        Optional aligned DEM raster to populate ``elev_start``, ``elev_end``,
        ``drop_m``, and ``slope`` attributes.
    flow_acc_img : str or None, optional
        Optional aligned flow accumulation raster to populate ``max_acc``.
    order_method : {'strahler', 'shreve'}, optional
        Stream ordering method (Default = ``'strahler'``).
    allow_pure_python : bool, optional
        If ``True``, allow falling back to pure Python with a
        ``RuntimeWarning`` when ``numba`` is unavailable (Default = ``False``).
    out_format : str or None, optional
        Optional output vector driver name.
    out_vec_lyr : str or None, optional
        Optional layer name for multi-layer vector containers.
    overwrite : bool, optional
        Overwrite existing output file (Default = ``True``).
    no_data_val : float or None, optional
        Optional flow direction nodata override (Default = ``None``).
    use_pbar : bool, optional
        Display a progress bar (Default = ``True``).
    """
    # pylint: disable=too-many-arguments,too-many-locals,too-many-branches,too-many-statements
    ord_norm = order_method.strip().lower()
    if ord_norm not in ("strahler", "shreve"):
        raise RSGISPyException(
            f"Invalid order_method '{order_method}'. Must be 'strahler' or 'shreve'."
        )
    ensure_file_exists(streams_img, "Input streams image")
    ensure_file_exists(flow_dir_img, "Input flow direction image")
    if dem_img is not None:
        ensure_file_exists(dem_img, "Input DEM image")
    if flow_acc_img is not None:
        ensure_file_exists(flow_acc_img, "Input flow accumulation image")

    with ExitStack() as stack:
        str_src = stack.enter_context(rasterio.open(streams_img, "r"))
        fdir_src = stack.enter_context(rasterio.open(flow_dir_img, "r"))
        _validate_projected_crs(fdir_src, flow_dir_img)
        _validate_matching_grid(fdir_src, str_src, flow_dir_img, streams_img)

        fdir, valid = _read_d8_grid(fdir_src, no_data_val)
        streams = str_src.read(1)
        valid &= build_valid_mask(streams, str_src.nodata)
        streams_bin = (streams == 1).astype(np.uint8)
        height, width = fdir.shape
        transform = fdir_src.transform

        dem_arr: Optional[np.ndarray] = None
        if dem_img is not None:
            dem_src = stack.enter_context(rasterio.open(dem_img, "r"))
            _validate_matching_grid(fdir_src, dem_src, flow_dir_img, dem_img)
            dem_arr = dem_src.read(1, out_dtype=np.float64)

        acc_arr: Optional[np.ndarray] = None
        if flow_acc_img is not None:
            acc_src = stack.enter_context(rasterio.open(flow_acc_img, "r"))
            _validate_matching_grid(fdir_src, acc_src, flow_dir_img, flow_acc_img)
            acc_arr = acc_src.read(1, out_dtype=np.float64)

        link_grid, in_deg = _call_topo_kernel(
            stream_link_kernel,
            allow_pure_python,
            "vectorise_streams",
            fdir,
            streams_bin,
            valid,
        )
        ord_grid = _call_topo_kernel(
            stream_order_kernel,
            allow_pure_python,
            "vectorise_streams",
            fdir,
            streams_bin,
            valid,
            ord_norm == "strahler",
        )

        # Locate upstream start cells for every link_id in a single O(N) pass
        start_cells = np.argwhere(
            (link_grid > 0) & ((in_deg == 0) | (in_deg >= 2))
        )
        tqdm_cls = rsgispy.get_tqdm()
        records: List[Dict[str, Any]] = []

        for sr, sc in tqdm_cls(
            start_cells,
            desc="Vectorising streams",
            unit="reach",
            disable=not use_pbar,
        ):
            r = int(sr)
            c = int(sc)
            lid = int(link_grid[r, c])
            ord_val = int(ord_grid[r, c])

            coords: List[Tuple[float, float]] = []
            reach_cells: List[Tuple[int, int]] = []
            down_link_id = -1
            end_r, end_c = r, c

            while True:
                px_x, px_y = rasterio.transform.xy(transform, r, c)
                coords.append((float(px_x), float(px_y)))
                reach_cells.append((r, c))
                end_r, end_c = r, c

                offset = _D8_DIR_TO_OFFSET.get(int(fdir[r, c]))
                if offset is None:
                    break
                nr = r + offset[0]
                nc = c + offset[1]
                if not (0 <= nr < height and 0 <= nc < width and valid[nr, nc]):
                    break
                if streams_bin[nr, nc] != 1:
                    break
                if int(link_grid[nr, nc]) != lid:
                    # Append downstream confluence vertex for exact topological snapping
                    nx_x, nx_y = rasterio.transform.xy(transform, nr, nc)
                    coords.append((float(nx_x), float(nx_y)))
                    down_link_id = int(link_grid[nr, nc])
                    end_r, end_c = nr, nc
                    break
                r, c = nr, nc

            if len(coords) == 1:
                coords.append(coords[0])
            geom = LineString(coords)
            length_m = float(geom.length)

            rec: Dict[str, Any] = {
                "link_id": lid,
                "down_link_id": down_link_id,
                "order": ord_val,
                "length_m": length_m,
            }
            if dem_arr is not None:
                z_start = float(dem_arr[int(sr), int(sc)])
                z_end = float(dem_arr[end_r, end_c])
                drop_m = z_start - z_end
                rec["elev_start"] = z_start
                rec["elev_end"] = z_end
                rec["drop_m"] = drop_m
                rec["slope"] = float(drop_m / length_m) if length_m > 0.0 else 0.0
            if acc_arr is not None:
                rec["max_acc"] = float(max(acc_arr[rr, cc] for rr, cc in reach_cells))
            rec["geometry"] = geom
            records.append(rec)

        if records:
            out_gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=fdir_src.crs)
            out_gdf["link_id"] = out_gdf["link_id"].astype(np.int32)
            out_gdf["down_link_id"] = out_gdf["down_link_id"].astype(np.int32)
            out_gdf["order"] = out_gdf["order"].astype(np.int32)
        else:
            out_gdf = gpd.GeoDataFrame(
                columns=["link_id", "down_link_id", "order", "length_m", "geometry"],
                geometry="geometry",
                crs=fdir_src.crs,
            )

        write_vec_gdf(
            out_gdf,
            output_vec,
            out_vec_lyr=out_vec_lyr,
            out_format=out_format,
            overwrite=overwrite,
        )


# ---------------------------------------------------------------------------
# Stage 4: Watershed & Subcatchment Delineation
# ---------------------------------------------------------------------------


def snap_pour_points(
    pour_points_vec: str,
    flow_acc_img: str,
    output_vec: str,
    snap_dist: float,
    streams_img: Optional[str] = None,
    *,
    out_format: Optional[str] = None,
    out_vec_lyr: Optional[str] = None,
    overwrite: bool = True,
    no_data_val: Optional[float] = None,
    use_pbar: bool = True,
) -> None:
    """
    Snap pour points to the cell of highest flow accumulation within a metric
    search radius, optionally constrained to stream cells.

    Parameters
    ----------
    pour_points_vec : str
        Path to the input pour points vector file.
    flow_acc_img : str
        Path to the flow accumulation raster.
    output_vec : str
        Path to the output snapped pour points vector file.
    snap_dist : float
        Maximum snap search distance in CRS metres.
    streams_img : str or None, optional
        Optional aligned binary stream mask raster restricting candidate cells
        to ``streams == 1`` (Default = ``None``).
    out_format : str or None, optional
        Optional output vector driver name.
    out_vec_lyr : str or None, optional
        Optional layer name for multi-layer vector containers.
    overwrite : bool, optional
        Overwrite existing output file (Default = ``True``).
    no_data_val : float or None, optional
        Optional flow accumulation nodata override (Default = ``None``).
    use_pbar : bool, optional
        Display a progress bar (Default = ``True``).
    """
    # pylint: disable=too-many-arguments,too-many-locals,too-many-branches
    if snap_dist < 0.0:
        raise RSGISPyException("snap_dist must be non-negative.")
    ensure_file_exists(pour_points_vec, "Input pour points vector")
    ensure_file_exists(flow_acc_img, "Input flow accumulation image")
    if streams_img is not None:
        ensure_file_exists(streams_img, "Input streams image")

    pts_gdf = read_vec_gdf(pour_points_vec)
    if pts_gdf.empty:
        raise RSGISPyException("Input pour_points_vec contains no features.")

    with ExitStack() as stack:
        acc_src = stack.enter_context(rasterio.open(flow_acc_img, "r"))
        dx, dy = _validate_projected_crs(acc_src, flow_acc_img)
        eff_nd = no_data_val if no_data_val is not None else acc_src.nodata
        acc = acc_src.read(1, out_dtype=np.float64)
        valid = build_valid_mask(acc, eff_nd)
        height, width = acc.shape

        if streams_img is not None:
            str_src = stack.enter_context(rasterio.open(streams_img, "r"))
            _validate_matching_grid(acc_src, str_src, flow_acc_img, streams_img)
            streams = str_src.read(1)
            valid &= build_valid_mask(streams, str_src.nodata) & (streams == 1)

        if pts_gdf.crs is not None and acc_src.crs is not None:
            if pyproj.CRS.from_user_input(pts_gdf.crs) != pyproj.CRS.from_user_input(
                acc_src.crs
            ):
                pts_gdf = pts_gdf.to_crs(acc_src.crs)

        rad_x = int(math.ceil(snap_dist / dx))
        rad_y = int(math.ceil(snap_dist / dy))
        attr_cols = [c for c in pts_gdf.columns if c != pts_gdf.active_geometry_name]
        tqdm_cls = rsgispy.get_tqdm()
        records: List[Dict[str, Any]] = []

        for pt_idx, (_, row_s) in enumerate(
            tqdm_cls(
                pts_gdf.iterrows(),
                total=len(pts_gdf),
                desc="Snapping pour points",
                unit="pt",
                disable=not use_pbar,
            )
        ):
            geom = row_s.geometry
            if geom is None or geom.is_empty:
                continue
            pt = geom.geoms[0] if geom.geom_type == "MultiPoint" else geom
            r0, c0 = acc_src.index(pt.x, pt.y)

            r_min = max(0, int(r0) - rad_y)
            r_max = min(height, int(r0) + rad_y + 1)
            c_min = max(0, int(c0) - rad_x)
            c_max = min(width, int(c0) + rad_x + 1)
            if r_min >= r_max or c_min >= c_max:
                raise RSGISPyException(
                    f"Pour point {pt_idx} is outside the search extent of '{flow_acc_img}'."
                )

            sub_valid = valid[r_min:r_max, c_min:c_max]
            sub_acc = acc[r_min:r_max, c_min:c_max]
            rr_grid, cc_grid = np.mgrid[r_min:r_max, c_min:c_max]
            metric_dist = np.hypot((rr_grid - r0) * dy, (cc_grid - c0) * dx)
            cand_mask = sub_valid & (metric_dist <= snap_dist + 1e-9)

            if not np.any(cand_mask):
                raise RSGISPyException(
                    f"No valid candidate cell found within snap_dist={snap_dist} m "
                    f"for pour point index {pt_idx}."
                )

            masked_acc = np.where(cand_mask, sub_acc, -np.inf)
            best_flat = int(np.argmax(masked_acc))
            idx_tuple = np.unravel_index(best_flat, masked_acc.shape)
            br_rel = int(idx_tuple[0])
            bc_rel = int(idx_tuple[1])
            best_r = r_min + br_rel
            best_c = c_min + bc_rel

            x_star, y_star = rasterio.transform.xy(acc_src.transform, best_r, best_c)
            rec = {col: row_s[col] for col in attr_cols}
            rec["snap_dist_m"] = float(metric_dist[br_rel, bc_rel])
            rec["snapped_acc"] = float(acc[best_r, best_c])
            rec["geometry"] = Point(float(x_star), float(y_star))
            records.append(rec)

        out_gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=acc_src.crs)
        write_vec_gdf(
            out_gdf,
            output_vec,
            out_vec_lyr=out_vec_lyr,
            out_format=out_format,
            overwrite=overwrite,
        )


def delineate_watersheds(
    flow_dir_img: str,
    pour_points_vec: str,
    output_img: str,
    id_field: Optional[str] = None,
    allow_pure_python: bool = False,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Delineate upslope contributing watersheds (and incremental sub-basins for
    nested gauges) from vector pour points via reverse-flow BFS.

    Parameters
    ----------
    flow_dir_img : str
        Path to the input D8 flow direction raster (``uint8``).
    pour_points_vec : str
        Path to the input pour points vector dataset.
    output_img : str
        Path to the output integer watershed raster (``int32``).
    id_field : str or None, optional
        Optional integer attribute column in ``pour_points_vec`` used as
        watershed IDs (Default = ``None``, sequential IDs ``1..M``).
    allow_pure_python : bool, optional
        If ``True``, allow falling back to pure Python with a
        ``RuntimeWarning`` when ``numba`` is unavailable (Default = ``False``).
    gdalformat : str, optional
        GDAL/rasterio output driver name (Default = ``'GTIFF'``).
    datatype : int or None, optional
        Optional ``rsgispy.TYPE_*`` constant (Default = ``rsgispy.TYPE_32INT``).
    no_data_val : float or None, optional
        Optional flow direction nodata override (Default = ``None``).
    calc_stats : bool, optional
        If ``True``, compute output statistics and overview pyramids
        (Default = ``True``).
    use_pbar : bool, optional
        Display a progress bar (Default = ``True``).
    """
    # pylint: disable=too-many-arguments,too-many-locals,too-many-branches
    ensure_file_exists(flow_dir_img, "Input flow direction image")
    ensure_file_exists(pour_points_vec, "Input pour points vector")
    _, out_rio_dt, out_np_dt = _resolve_out_dtype(datatype, rsgispy.TYPE_32INT)

    pts_gdf = read_vec_gdf(pour_points_vec)
    if pts_gdf.empty:
        raise RSGISPyException("Input pour_points_vec contains no features.")
    if id_field is not None and id_field not in pts_gdf.columns:
        raise RSGISPyException(
            f"id_field '{id_field}' not found in '{pour_points_vec}'."
        )

    with _run_progress_steps("Delineating watersheds", 3, use_pbar) as pbar:
        with rasterio.open(flow_dir_img, "r") as src:
            _validate_projected_crs(src, flow_dir_img)
            fdir, valid = _read_d8_grid(src, no_data_val)
            height, width = fdir.shape

            if pts_gdf.crs is not None and src.crs is not None:
                if pyproj.CRS.from_user_input(pts_gdf.crs) != pyproj.CRS.from_user_input(
                    src.crs
                ):
                    pts_gdf = pts_gdf.to_crs(src.crs)

            seed_grid = np.where(valid, np.int32(0), np.int32(-1)).astype(np.int32)
            for idx_i, (_, row_s) in enumerate(pts_gdf.iterrows(), start=1):
                geom = row_s.geometry
                if geom is None or geom.is_empty:
                    continue
                pt = geom.geoms[0] if geom.geom_type == "MultiPoint" else geom
                r0, c0 = src.index(pt.x, pt.y)
                if not (0 <= r0 < height and 0 <= c0 < width and valid[r0, c0]):
                    raise RSGISPyException(
                        f"Pour point {idx_i} falls outside valid domain of '{flow_dir_img}'."
                    )
                ws_id = int(row_s[id_field]) if id_field is not None else idx_i
                if ws_id <= 0:
                    raise RSGISPyException(
                        f"Watershed ID must be a positive integer, got {ws_id}."
                    )
                seed_grid[int(r0), int(c0)] = np.int32(ws_id)
            pbar.update(1)

            ws_grid = _call_topo_kernel(
                reverse_bfs_label_kernel,
                allow_pure_python,
                "delineate_watersheds",
                fdir,
                valid,
                seed_grid,
            )
            pbar.update(1)

            out_nd = _resolve_out_nodata(out_np_dt, None, -1.0)
            if out_nd is not None and int(round(out_nd)) != -1:
                ws_grid = np.where(valid, ws_grid, int(round(out_nd)))

            _write_raster_array(
                output_img=output_img,
                arr=ws_grid,
                ref_src=src,
                gdalformat=gdalformat,
                out_rio_dtype=out_rio_dt,
                out_np_dtype=out_np_dt,
                out_nodata=out_nd,
                calc_stats=calc_stats,
                use_pbar=use_pbar,
            )
            pbar.update(1)


def delineate_basins(
    input_img: str,
    output_img: str,
    allow_pure_python: bool = False,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Partition all valid cells of a D8 flow direction raster into complete
    drainage basins draining to terminal outlets.

    Parameters
    ----------
    input_img : str
        Path to the input D8 flow direction raster (``uint8``).
    output_img : str
        Path to the output drainage basin raster (``int32``).
    allow_pure_python : bool, optional
        If ``True``, allow falling back to pure Python with a
        ``RuntimeWarning`` when ``numba`` is unavailable (Default = ``False``).
    gdalformat : str, optional
        GDAL/rasterio output driver name (Default = ``'GTIFF'``).
    datatype : int or None, optional
        Optional ``rsgispy.TYPE_*`` constant (Default = ``rsgispy.TYPE_32INT``).
    no_data_val : float or None, optional
        Optional flow direction nodata override (Default = ``None``).
    calc_stats : bool, optional
        If ``True``, compute output statistics and overview pyramids
        (Default = ``True``).
    use_pbar : bool, optional
        Display a progress bar (Default = ``True``).
    """
    # pylint: disable=too-many-arguments,too-many-locals
    ensure_file_exists(input_img, "Input flow direction image")
    _, out_rio_dt, out_np_dt = _resolve_out_dtype(datatype, rsgispy.TYPE_32INT)

    with _run_progress_steps("Delineating basins", 3, use_pbar) as pbar:
        with rasterio.open(input_img, "r") as src:
            _validate_projected_crs(src, input_img)
            fdir, valid = _read_d8_grid(src, no_data_val)
            pbar.update(1)

            seed_grid = _call_topo_kernel(
                find_terminal_outlets_kernel,
                allow_pure_python,
                "delineate_basins",
                fdir,
                valid,
            )
            basins = _call_topo_kernel(
                reverse_bfs_label_kernel,
                allow_pure_python,
                "delineate_basins",
                fdir,
                valid,
                seed_grid,
            )
            pbar.update(1)

            out_nd = _resolve_out_nodata(out_np_dt, None, -1.0)
            if out_nd is not None and int(round(out_nd)) != -1:
                basins = np.where(valid, basins, int(round(out_nd)))

            _write_raster_array(
                output_img=output_img,
                arr=basins,
                ref_src=src,
                gdalformat=gdalformat,
                out_rio_dtype=out_rio_dt,
                out_np_dtype=out_np_dt,
                out_nodata=out_nd,
                calc_stats=calc_stats,
                use_pbar=use_pbar,
            )
            pbar.update(1)


def delineate_subcatchments(
    flow_dir_img: str,
    stream_link_img: str,
    output_img: str,
    allow_pure_python: bool = False,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Delineate local lateral contributing subcatchments for every stream reach ID
    in ``stream_link_img``.

    Parameters
    ----------
    flow_dir_img : str
        Path to the input D8 flow direction raster (``uint8``).
    stream_link_img : str
        Path to the input stream link ID raster from :func:`stream_link`.
    output_img : str
        Path to the output subcatchment raster (``int32``).
    allow_pure_python : bool, optional
        If ``True``, allow falling back to pure Python with a
        ``RuntimeWarning`` when ``numba`` is unavailable (Default = ``False``).
    gdalformat : str, optional
        GDAL/rasterio output driver name (Default = ``'GTIFF'``).
    datatype : int or None, optional
        Optional ``rsgispy.TYPE_*`` constant (Default = ``rsgispy.TYPE_32INT``).
    no_data_val : float or None, optional
        Optional flow direction nodata override (Default = ``None``).
    calc_stats : bool, optional
        If ``True``, compute output statistics and overview pyramids
        (Default = ``True``).
    use_pbar : bool, optional
        Display a progress bar (Default = ``True``).
    """
    # pylint: disable=too-many-arguments,too-many-locals
    ensure_file_exists(flow_dir_img, "Input flow direction image")
    ensure_file_exists(stream_link_img, "Input stream link image")
    _, out_rio_dt, out_np_dt = _resolve_out_dtype(datatype, rsgispy.TYPE_32INT)

    with _run_progress_steps("Delineating subcatchments", 3, use_pbar) as pbar:
        with ExitStack() as stack:
            fdir_src = stack.enter_context(rasterio.open(flow_dir_img, "r"))
            lnk_src = stack.enter_context(rasterio.open(stream_link_img, "r"))
            _validate_projected_crs(fdir_src, flow_dir_img)
            _validate_matching_grid(fdir_src, lnk_src, flow_dir_img, stream_link_img)

            fdir, valid = _read_d8_grid(fdir_src, no_data_val)
            links = lnk_src.read(1)
            valid &= build_valid_mask(links, lnk_src.nodata)
            seed_grid = np.where(
                valid,
                np.where(links > 0, links, 0),
                -1,
            ).astype(np.int32)
            pbar.update(1)

            subcats = _call_topo_kernel(
                reverse_bfs_label_kernel,
                allow_pure_python,
                "delineate_subcatchments",
                fdir,
                valid,
                seed_grid,
            )
            pbar.update(1)

            out_nd = _resolve_out_nodata(out_np_dt, None, -1.0)
            if out_nd is not None and int(round(out_nd)) != -1:
                subcats = np.where(valid, subcats, int(round(out_nd)))

            _write_raster_array(
                output_img=output_img,
                arr=subcats,
                ref_src=fdir_src,
                gdalformat=gdalformat,
                out_rio_dtype=out_rio_dt,
                out_np_dtype=out_np_dt,
                out_nodata=out_nd,
                calc_stats=calc_stats,
                use_pbar=use_pbar,
            )
            pbar.update(1)


def polygonise_watersheds(
    watershed_img: str,
    output_vec: str,
    dem_img: Optional[str] = None,
    dissolve: bool = True,
    simplify_tol: Optional[float] = None,
    *,
    out_format: Optional[str] = None,
    out_vec_lyr: Optional[str] = None,
    overwrite: bool = True,
    no_data_val: Optional[float] = None,
    use_pbar: bool = True,
) -> None:
    """
    Polygonise a labelled watershed or basin raster into vector polygons with
    area, perimeter, and optional fast DEM morphometric statistics.

    Parameters
    ----------
    watershed_img : str
        Path to the input integer watershed/basin raster.
    output_vec : str
        Path to the output vector polygon dataset.
    dem_img : str or None, optional
        Optional aligned DEM raster used to compute ``elev_min``, ``elev_mean``,
        and ``elev_max`` per watershed via fast vectorised grouping.
    dissolve : bool, optional
        If ``True``, dissolve multipart regions sharing the same
        ``watershed_id`` (Default = ``True``).
    simplify_tol : float or None, optional
        Optional topology-preserving polygon simplification tolerance in map
        metres (Default = ``None``).
    out_format : str or None, optional
        Optional output vector driver name.
    out_vec_lyr : str or None, optional
        Optional layer name for multi-layer vector containers.
    overwrite : bool, optional
        Overwrite existing output file (Default = ``True``).
    no_data_val : float or None, optional
        Optional watershed raster nodata override (Default = ``None``).
    use_pbar : bool, optional
        Display a progress bar (Default = ``True``).
    """
    # pylint: disable=too-many-arguments,too-many-locals,too-many-branches
    ensure_file_exists(watershed_img, "Input watershed image")
    if dem_img is not None:
        ensure_file_exists(dem_img, "Input DEM image")

    with _run_progress_steps("Polygonising watersheds", 3, use_pbar) as pbar:
        with ExitStack() as stack:
            ws_src = stack.enter_context(rasterio.open(watershed_img, "r"))
            _validate_projected_crs(ws_src, watershed_img)
            eff_nd = no_data_val if no_data_val is not None else ws_src.nodata
            ws_arr = ws_src.read(1)
            valid_ws = build_valid_mask(ws_arr, eff_nd) & (ws_arr > 0)
            ws_int = np.where(valid_ws, ws_arr, 0).astype(np.int32)

            records = [
                {"watershed_id": int(val), "geometry": shape(geom_dict)}
                for geom_dict, val in rasterio.features.shapes(
                    ws_int, mask=valid_ws, transform=ws_src.transform
                )
            ]
            pbar.update(1)

            if records:
                gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=ws_src.crs)
                gdf["watershed_id"] = gdf["watershed_id"].astype(np.int32)
                if dissolve:
                    gdf = gdf.dissolve(by="watershed_id", as_index=False)
                if simplify_tol is not None and simplify_tol > 0.0:
                    gdf["geometry"] = gdf.geometry.simplify(
                        float(simplify_tol), preserve_topology=True
                    )
                gdf["area_m2"] = gdf.geometry.area.astype(np.float64)
                gdf["perimeter_m"] = gdf.geometry.length.astype(np.float64)
            else:
                gdf = gpd.GeoDataFrame(
                    columns=["watershed_id", "area_m2", "perimeter_m", "geometry"],
                    geometry="geometry",
                    crs=ws_src.crs,
                )

            if dem_img is not None and not gdf.empty:
                dem_src = stack.enter_context(rasterio.open(dem_img, "r"))
                _validate_matching_grid(ws_src, dem_src, watershed_img, dem_img)
                dem = dem_src.read(1, out_dtype=np.float64)
                valid_both = valid_ws & build_valid_mask(dem, dem_src.nodata)
                ws_vals = ws_int[valid_both]
                dem_vals = dem[valid_both]

                if ws_vals.size > 0:
                    u_ids, inv = np.unique(ws_vals, return_inverse=True)
                    n_u = u_ids.shape[0]
                    counts = np.bincount(inv, minlength=n_u)
                    sums = np.bincount(inv, weights=dem_vals, minlength=n_u)
                    mins = np.full(n_u, np.inf, dtype=np.float64)
                    maxs = np.full(n_u, -np.inf, dtype=np.float64)
                    np.minimum.at(mins, inv, dem_vals)
                    np.maximum.at(maxs, inv, dem_vals)
                    means = sums / np.maximum(counts, 1)

                    stats_map = {
                        int(uid): (float(mn), float(av), float(mx))
                        for uid, mn, av, mx in zip(u_ids, mins, means, maxs)
                    }
                    gdf["elev_min"] = gdf["watershed_id"].map(
                        lambda wid: stats_map.get(int(wid), (np.nan, np.nan, np.nan))[0]
                    )
                    gdf["elev_mean"] = gdf["watershed_id"].map(
                        lambda wid: stats_map.get(int(wid), (np.nan, np.nan, np.nan))[1]
                    )
                    gdf["elev_max"] = gdf["watershed_id"].map(
                        lambda wid: stats_map.get(int(wid), (np.nan, np.nan, np.nan))[2]
                    )
            pbar.update(1)

            write_vec_gdf(
                gdf,
                output_vec,
                out_vec_lyr=out_vec_lyr,
                out_format=out_format,
                overwrite=overwrite,
            )
            pbar.update(1)


# ---------------------------------------------------------------------------
# Stage 5: Hydrological Terrain Indices
# ---------------------------------------------------------------------------


def calc_hand(
    dem_img: str,
    flow_dir_img: str,
    streams_img: str,
    output_img: str,
    allow_pure_python: bool = False,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Calculate Height Above Nearest Drainage (HAND) using $O(N)$ reverse-flow
    BFS (Nobre et al., 2011).

    Parameters
    ----------
    dem_img : str
        Path to the input DEM raster.
    flow_dir_img : str
        Path to the input D8 flow direction raster (``uint8``).
    streams_img : str
        Path to the binary stream network mask raster (``1`` = stream).
    output_img : str
        Path to the output HAND raster (``float32``, metres).
    allow_pure_python : bool, optional
        If ``True``, allow falling back to pure Python with a
        ``RuntimeWarning`` when ``numba`` is unavailable (Default = ``False``).
    gdalformat : str, optional
        GDAL/rasterio output driver name (Default = ``'GTIFF'``).
    datatype : int or None, optional
        Optional ``rsgispy.TYPE_*`` constant (Default = ``rsgispy.TYPE_32FLOAT``).
    no_data_val : float or None, optional
        Optional DEM nodata override (Default = ``None``).
    calc_stats : bool, optional
        If ``True``, compute output statistics and overview pyramids
        (Default = ``True``).
    use_pbar : bool, optional
        Display a progress bar (Default = ``True``).
    """
    # pylint: disable=too-many-arguments,too-many-locals
    ensure_file_exists(dem_img, "Input DEM image")
    ensure_file_exists(flow_dir_img, "Input flow direction image")
    ensure_file_exists(streams_img, "Input streams image")
    _, out_rio_dt, out_np_dt = _resolve_out_dtype(datatype, rsgispy.TYPE_32FLOAT)

    with _run_progress_steps("Calculating HAND", 3, use_pbar) as pbar:
        with ExitStack() as stack:
            dem_src = stack.enter_context(rasterio.open(dem_img, "r"))
            fdir_src = stack.enter_context(rasterio.open(flow_dir_img, "r"))
            str_src = stack.enter_context(rasterio.open(streams_img, "r"))

            _validate_projected_crs(dem_src, dem_img)
            _validate_matching_grid(dem_src, fdir_src, dem_img, flow_dir_img)
            _validate_matching_grid(dem_src, str_src, dem_img, streams_img)

            eff_dem_nd = no_data_val if no_data_val is not None else dem_src.nodata
            dem = dem_src.read(1, out_dtype=np.float64)
            valid = build_valid_mask(dem, eff_dem_nd)

            fdir, valid_fdir = _read_d8_grid(fdir_src, None)
            streams = str_src.read(1)
            valid &= valid_fdir & build_valid_mask(streams, str_src.nodata)
            streams_bin = (streams == 1).astype(np.uint8)
            pbar.update(1)

            hand = _call_topo_kernel(
                flow_distance_vert_kernel,
                allow_pure_python,
                "calc_hand",
                fdir,
                valid,
                streams_bin,
                dem,
            )
            pbar.update(1)

            out_nd = _resolve_out_nodata(out_np_dt, None, -9999.0)
            fill_nd = -9999.0 if out_nd is None else float(out_nd)
            ok = valid & np.isfinite(hand)
            out_arr = np.where(ok, hand, fill_nd)

            _write_raster_array(
                output_img=output_img,
                arr=out_arr,
                ref_src=dem_src,
                gdalformat=gdalformat,
                out_rio_dtype=out_rio_dt,
                out_np_dtype=out_np_dt,
                out_nodata=out_nd,
                calc_stats=calc_stats,
                use_pbar=use_pbar,
            )
            pbar.update(1)


def flood_inundation_from_hand(
    input_img: str,
    output_img: str,
    water_depth: Union[float, str, os.PathLike],
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Estimate static flood inundation water depth ``max(0, water_depth - HAND)``
    from a HAND raster and either a scalar flood stage or a stage raster.

    Parameters
    ----------
    input_img : str
        Path to the input HAND raster (metres).
    output_img : str
        Path to the output inundated water depth raster (``float32``, metres).
    water_depth : float or str
        Constant flood stage height above channel bed (metres) or path to an
        aligned spatially varying stage raster.
    gdalformat : str, optional
        GDAL/rasterio output driver name (Default = ``'GTIFF'``).
    datatype : int or None, optional
        Optional ``rsgispy.TYPE_*`` constant (Default = ``rsgispy.TYPE_32FLOAT``).
    no_data_val : float or None, optional
        Optional input HAND nodata override (Default = ``None``).
    calc_stats : bool, optional
        If ``True``, compute output statistics and overview pyramids
        (Default = ``True``).
    use_pbar : bool, optional
        Display a progress bar (Default = ``True``).
    """
    # pylint: disable=too-many-arguments,too-many-locals
    ensure_file_exists(input_img, "Input HAND image")
    _, out_rio_dt, out_np_dt = _resolve_out_dtype(datatype, rsgispy.TYPE_32FLOAT)
    out_nd = _resolve_out_nodata(out_np_dt, None, -9999.0)
    fill_nd = -9999.0 if out_nd is None else float(out_nd)

    is_raster_stage = isinstance(water_depth, (str, os.PathLike))
    if is_raster_stage:
        stage_path = os.fspath(water_depth)  # type: ignore[arg-type]
        ensure_file_exists(stage_path, "Input water depth raster")
    else:
        stage_scalar = float(water_depth)  # type: ignore[arg-type]

    with _run_progress_steps("Computing flood inundation", 3, use_pbar) as pbar:
        with ExitStack() as stack:
            hand_src = stack.enter_context(rasterio.open(input_img, "r"))
            _validate_projected_crs(hand_src, input_img)
            eff_nd = no_data_val if no_data_val is not None else hand_src.nodata
            hand = hand_src.read(1, out_dtype=np.float64)
            valid = build_valid_mask(hand, eff_nd)
            pbar.update(1)

            if is_raster_stage:
                stage_src = stack.enter_context(rasterio.open(stage_path, "r"))
                _validate_matching_grid(hand_src, stage_src, input_img, stage_path)
                stage_arr = stage_src.read(1, out_dtype=np.float64)
                valid &= build_valid_mask(stage_arr, stage_src.nodata)
                if HAS_NUMBA:
                    out_arr = flood_inundation_raster_njit(
                        hand, stage_arr, valid, fill_nd
                    )
                else:
                    _warn_numpy_fallback("flood_inundation_from_hand")
                    out_arr = flood_inundation_raster_numpy(
                        hand, stage_arr, valid, fill_nd
                    )
            else:
                if HAS_NUMBA:
                    out_arr = flood_inundation_scalar_njit(
                        hand, valid, stage_scalar, fill_nd
                    )
                else:
                    _warn_numpy_fallback("flood_inundation_from_hand")
                    out_arr = flood_inundation_scalar_numpy(
                        hand, valid, stage_scalar, fill_nd
                    )
            pbar.update(1)

            _write_raster_array(
                output_img=output_img,
                arr=out_arr,
                ref_src=hand_src,
                gdalformat=gdalformat,
                out_rio_dtype=out_rio_dt,
                out_np_dtype=out_np_dt,
                out_nodata=out_nd,
                calc_stats=calc_stats,
                use_pbar=use_pbar,
            )
            pbar.update(1)


def calc_twi(
    specific_catchment_area_img: str,
    slope_img: str,
    output_img: str,
    slope_units: str = "degrees",
    min_slope: float = 0.001,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Calculate the Topographic Wetness Index ``TWI = ln(a / tan(beta))``.

    Parameters
    ----------
    specific_catchment_area_img : str
        Path to the specific catchment area raster (``m^2 / m``).
    slope_img : str
        Path to the aligned terrain slope raster.
    output_img : str
        Path to the output TWI raster (``float32``).
    slope_units : {'degrees', 'radians', 'percent'}, optional
        Units of ``slope_img`` (Default = ``'degrees'``).
    min_slope : float, optional
        Lower clamp applied to ``tan(beta)`` to prevent division by zero on
        flat terrain (Default = ``0.001``).
    gdalformat : str, optional
        GDAL/rasterio output driver name (Default = ``'GTIFF'``).
    datatype : int or None, optional
        Optional ``rsgispy.TYPE_*`` constant (Default = ``rsgispy.TYPE_32FLOAT``).
    no_data_val : float or None, optional
        Optional input nodata override for ``specific_catchment_area_img``.
    calc_stats : bool, optional
        If ``True``, compute output statistics and overview pyramids
        (Default = ``True``).
    use_pbar : bool, optional
        Display a progress bar (Default = ``True``).
    """
    # pylint: disable=too-many-arguments,too-many-locals
    unit_code = _validate_slope_units(slope_units)
    if min_slope <= 0.0:
        raise RSGISPyException("min_slope must be strictly positive.")
    ensure_file_exists(specific_catchment_area_img, "Input specific catchment area image")
    ensure_file_exists(slope_img, "Input slope image")

    _, out_rio_dt, out_np_dt = _resolve_out_dtype(datatype, rsgispy.TYPE_32FLOAT)
    out_nd = _resolve_out_nodata(out_np_dt, None, -9999.0)
    fill_nd = -9999.0 if out_nd is None else float(out_nd)

    with _run_progress_steps("Calculating TWI", 3, use_pbar) as pbar:
        with ExitStack() as stack:
            sca_src = stack.enter_context(rasterio.open(specific_catchment_area_img, "r"))
            slp_src = stack.enter_context(rasterio.open(slope_img, "r"))
            dx, _ = _validate_projected_crs(sca_src, specific_catchment_area_img)
            _validate_matching_grid(
                sca_src, slp_src, specific_catchment_area_img, slope_img
            )

            eff_nd = no_data_val if no_data_val is not None else sca_src.nodata
            sca = sca_src.read(1, out_dtype=np.float64)
            slope_arr = slp_src.read(1, out_dtype=np.float64)
            valid = build_valid_mask(sca, eff_nd) & build_valid_mask(
                slope_arr, slp_src.nodata
            )
            pbar.update(1)

            if HAS_NUMBA:
                out_arr = calc_twi_njit(
                    sca, slope_arr, valid, unit_code, float(min_slope), dx, fill_nd
                )
            else:
                _warn_numpy_fallback("calc_twi")
                out_arr = calc_twi_numpy(
                    sca, slope_arr, valid, unit_code, float(min_slope), dx, fill_nd
                )
            pbar.update(1)

            _write_raster_array(
                output_img=output_img,
                arr=out_arr,
                ref_src=sca_src,
                gdalformat=gdalformat,
                out_rio_dtype=out_rio_dt,
                out_np_dtype=out_np_dt,
                out_nodata=out_nd,
                calc_stats=calc_stats,
                use_pbar=use_pbar,
            )
            pbar.update(1)


def calc_spi(
    specific_catchment_area_img: str,
    slope_img: str,
    output_img: str,
    slope_units: str = "degrees",
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Calculate the Stream Power Index ``SPI = a * tan(beta)``.

    Parameters
    ----------
    specific_catchment_area_img : str
        Path to the specific catchment area raster (``m^2 / m``).
    slope_img : str
        Path to the aligned terrain slope raster.
    output_img : str
        Path to the output SPI raster (``float32``).
    slope_units : {'degrees', 'radians', 'percent'}, optional
        Units of ``slope_img`` (Default = ``'degrees'``).
    gdalformat : str, optional
        GDAL/rasterio output driver name (Default = ``'GTIFF'``).
    datatype : int or None, optional
        Optional ``rsgispy.TYPE_*`` constant (Default = ``rsgispy.TYPE_32FLOAT``).
    no_data_val : float or None, optional
        Optional input nodata override for ``specific_catchment_area_img``.
    calc_stats : bool, optional
        If ``True``, compute output statistics and overview pyramids
        (Default = ``True``).
    use_pbar : bool, optional
        Display a progress bar (Default = ``True``).
    """
    # pylint: disable=too-many-arguments,too-many-locals
    unit_code = _validate_slope_units(slope_units)
    ensure_file_exists(specific_catchment_area_img, "Input specific catchment area image")
    ensure_file_exists(slope_img, "Input slope image")

    _, out_rio_dt, out_np_dt = _resolve_out_dtype(datatype, rsgispy.TYPE_32FLOAT)
    out_nd = _resolve_out_nodata(out_np_dt, None, -9999.0)
    fill_nd = -9999.0 if out_nd is None else float(out_nd)

    with _run_progress_steps("Calculating SPI", 3, use_pbar) as pbar:
        with ExitStack() as stack:
            sca_src = stack.enter_context(rasterio.open(specific_catchment_area_img, "r"))
            slp_src = stack.enter_context(rasterio.open(slope_img, "r"))
            _validate_projected_crs(sca_src, specific_catchment_area_img)
            _validate_matching_grid(
                sca_src, slp_src, specific_catchment_area_img, slope_img
            )

            eff_nd = no_data_val if no_data_val is not None else sca_src.nodata
            sca = sca_src.read(1, out_dtype=np.float64)
            slope_arr = slp_src.read(1, out_dtype=np.float64)
            valid = build_valid_mask(sca, eff_nd) & build_valid_mask(
                slope_arr, slp_src.nodata
            )
            pbar.update(1)

            if HAS_NUMBA:
                out_arr = calc_spi_njit(sca, slope_arr, valid, unit_code, fill_nd)
            else:
                _warn_numpy_fallback("calc_spi")
                out_arr = calc_spi_numpy(sca, slope_arr, valid, unit_code, fill_nd)
            pbar.update(1)

            _write_raster_array(
                output_img=output_img,
                arr=out_arr,
                ref_src=sca_src,
                gdalformat=gdalformat,
                out_rio_dtype=out_rio_dt,
                out_np_dtype=out_np_dt,
                out_nodata=out_nd,
                calc_stats=calc_stats,
                use_pbar=use_pbar,
            )
            pbar.update(1)


def calc_ls_factor(
    specific_catchment_area_img: str,
    slope_img: str,
    output_img: str,
    slope_units: str = "degrees",
    m: float = 0.4,
    n: float = 1.3,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Calculate the Moore & Burch (1986) unit-stream-power RUSLE slope-length
    and steepness (LS) factor.

    Parameters
    ----------
    specific_catchment_area_img : str
        Path to the specific catchment area raster (``m^2 / m``).
    slope_img : str
        Path to the aligned terrain slope raster.
    output_img : str
        Path to the output LS factor raster (``float32``).
    slope_units : {'degrees', 'radians', 'percent'}, optional
        Units of ``slope_img`` (Default = ``'degrees'``).
    m : float, optional
        Contributing area exponent (Default = ``0.4``).
    n : float, optional
        Slope steepness exponent (Default = ``1.3``).
    gdalformat : str, optional
        GDAL/rasterio output driver name (Default = ``'GTIFF'``).
    datatype : int or None, optional
        Optional ``rsgispy.TYPE_*`` constant (Default = ``rsgispy.TYPE_32FLOAT``).
    no_data_val : float or None, optional
        Optional input nodata override for ``specific_catchment_area_img``.
    calc_stats : bool, optional
        If ``True``, compute output statistics and overview pyramids
        (Default = ``True``).
    use_pbar : bool, optional
        Display a progress bar (Default = ``True``).
    """
    # pylint: disable=too-many-arguments,too-many-locals
    unit_code = _validate_slope_units(slope_units)
    ensure_file_exists(specific_catchment_area_img, "Input specific catchment area image")
    ensure_file_exists(slope_img, "Input slope image")

    _, out_rio_dt, out_np_dt = _resolve_out_dtype(datatype, rsgispy.TYPE_32FLOAT)
    out_nd = _resolve_out_nodata(out_np_dt, None, -9999.0)
    fill_nd = -9999.0 if out_nd is None else float(out_nd)

    with _run_progress_steps("Calculating LS factor", 3, use_pbar) as pbar:
        with ExitStack() as stack:
            sca_src = stack.enter_context(rasterio.open(specific_catchment_area_img, "r"))
            slp_src = stack.enter_context(rasterio.open(slope_img, "r"))
            _validate_projected_crs(sca_src, specific_catchment_area_img)
            _validate_matching_grid(
                sca_src, slp_src, specific_catchment_area_img, slope_img
            )

            eff_nd = no_data_val if no_data_val is not None else sca_src.nodata
            sca = sca_src.read(1, out_dtype=np.float64)
            slope_arr = slp_src.read(1, out_dtype=np.float64)
            valid = build_valid_mask(sca, eff_nd) & build_valid_mask(
                slope_arr, slp_src.nodata
            )
            pbar.update(1)

            if HAS_NUMBA:
                out_arr = calc_ls_njit(
                    sca, slope_arr, valid, unit_code, float(m), float(n), fill_nd
                )
            else:
                _warn_numpy_fallback("calc_ls_factor")
                out_arr = calc_ls_numpy(
                    sca, slope_arr, valid, unit_code, float(m), float(n), fill_nd
                )
            pbar.update(1)

            _write_raster_array(
                output_img=output_img,
                arr=out_arr,
                ref_src=sca_src,
                gdalformat=gdalformat,
                out_rio_dtype=out_rio_dt,
                out_np_dtype=out_np_dt,
                out_nodata=out_nd,
                calc_stats=calc_stats,
                use_pbar=use_pbar,
            )
            pbar.update(1)
