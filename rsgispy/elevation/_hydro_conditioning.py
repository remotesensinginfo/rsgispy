"""
Stage 1: DEM hydrological conditioning functions for
:mod:`rsgispy.elevation.hydrology`.
"""

from typing import Any, Optional

import numpy as np
import pyproj
import rasterio
import rasterio.features
from rasterio.transform import Affine

import rsgispy
from rsgispy import RSGISPyException
from rsgispy.elevation._hydro_common import (
    _call_topo_kernel,
    _get_distance_transform_edt,
    _get_nd_label,
    _have_scipy,
    _resolve_out_dtype,
    _resolve_out_nodata,
    _run_progress_steps,
    _validate_projected_crs,
    _write_raster_array,
)
from rsgispy.elevation._hydro_kernels import (
    breach_depressions_kernel,
    priority_flood_kernel,
    resolve_flats_kernel,
    single_cell_pits_kernel,
)
from rsgispy.imageprocess import build_valid_mask, ensure_file_exists
from rsgispy.vectorprocess import read_vec_gdf


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
        edt_fn = _get_distance_transform_edt()
        rsgispy.check_optional_dep(
            _have_scipy() and edt_fn is not None,
            "burn_streams (buffer_dist > 0)",
            "scipy",
            "stats",
        )
        buf_d = float(buffer_dist)  # type: ignore[arg-type]
        dist_m = edt_fn(~stream_mask, sampling=(dy, dx))
        in_buf = valid & (~stream_mask) & (dist_m > 0.0) & (dist_m <= buf_d)
        if np.any(in_buf):
            out[in_buf] -= smooth_drop * (1.0 - dist_m[in_buf] / buf_d) ** 2

    drop_centre = burn_depth + (smooth_drop if has_buf else 0.0)
    centre_cells = valid & stream_mask
    out[centre_cells] -= drop_centre
    return out


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
    label_fn = _get_nd_label()
    rsgispy.check_optional_dep(
        _have_scipy() and label_fn is not None,
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
                single_cell_pits_kernel,
                allow_pure_python,
                "identify_sinks",
                dem,
                valid,
                warn=False,
            )
            sink_mask = valid & ((z_filled > dem) | single_pits)

            labelled, _ = label_fn(sink_mask, structure=np.ones((3, 3), dtype=np.int32))
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
                out_no_data_val=out_nd,
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
                    out_no_data_val=depth_nd,
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
                out_no_data_val=out_nd,
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

    From the bottom of each pit, the breach path is the one with the smallest
    total cut (the depth each pixel must be lowered to the pit, times the step
    length) to an outlet: a pixel lower than the pit, or a pixel on the DEM
    edge or next to no data at any height (the flow leaves the DEM there, so
    that pixel is cut to just below the pit). The path is carved as a
    descending gradient from the pit to the outlet.

    Without ``max_depth`` / ``max_length`` every pit is breached, however
    deep or long the cut: a pit behind a real barrier (a dam, a closed basin,
    a deep quarry) gets a trench through it. For real DEMs, set
    ``max_depth`` and ``max_length`` (the pits that can't be breached within
    them are filled when ``fill_remaining=True``), or use
    :func:`condition_dem` with ``method='breach_and_fill'``.

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
                    warn=False,
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
                out_no_data_val=out_nd,
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
                out_no_data_val=out_nd,
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
                out_no_data_val=out_nd,
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

            warned = False
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
                warned = True
            if method_norm in ("breach_and_fill", "fill"):
                dem = _call_topo_kernel(
                    priority_flood_kernel,
                    allow_pure_python,
                    "condition_dem",
                    dem,
                    valid,
                    warn=not warned,
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
                out_no_data_val=out_nd,
                calc_stats=calc_stats,
                use_pbar=use_pbar,
            )
            pbar.update(1)
