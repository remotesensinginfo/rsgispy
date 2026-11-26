"""
Stage 5: Hydrological terrain indices (HAND, flood inundation, TWI, SPI, and
RUSLE LS-factor) for :mod:`rsgispy.elevation.hydrology`.
"""

from contextlib import ExitStack
import os
from typing import Optional, Union

import numpy as np
import rasterio

import rsgispy
from rsgispy import RSGISPyException
from rsgispy.elevation._hydro_common import (
    _call_topo_kernel,
    _reject_flow_acc_input,
    _has_numba,
    _read_d8_grid,
    _resolve_out_dtype,
    _resolve_out_nodata,
    _run_progress_steps,
    _validate_matching_grid,
    _validate_projected_crs,
    _validate_slope_units,
    _warn_numpy_fallback,
    _write_raster_array,
)
from rsgispy.elevation._hydro_kernels import (
    calc_ls_njit,
    calc_ls_numpy,
    calc_spi_njit,
    calc_spi_numpy,
    calc_twi_njit,
    calc_twi_numpy,
    flood_inundation_raster_njit,
    flood_inundation_raster_numpy,
    flood_inundation_scalar_njit,
    flood_inundation_scalar_numpy,
    flow_distance_vert_kernel,
)
from rsgispy.imageprocess import build_valid_mask, ensure_file_exists


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
                out_no_data_val=out_nd,
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
                if _has_numba():
                    out_arr = flood_inundation_raster_njit(
                        hand, stage_arr, valid, fill_nd
                    )
                else:
                    _warn_numpy_fallback("flood_inundation_from_hand")
                    out_arr = flood_inundation_raster_numpy(
                        hand, stage_arr, valid, fill_nd
                    )
            else:
                if _has_numba():
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
                out_no_data_val=out_nd,
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
        Path to the specific catchment area raster (``m^2 / m``): a flow
        accumulation in m² (``flow_accumulation(out_units='area')``) divided
        by the contour width, usually the pixel size. A raster tagged as a
        flow accumulation (``RSGISPY_FLOW_ACC_UNITS``) raises
        :class:`rsgispy.RSGISPyException`.
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
            _reject_flow_acc_input(
                sca_src, specific_catchment_area_img, "calc_twi"
            )
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

            if _has_numba():
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
                out_no_data_val=out_nd,
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
        Path to the specific catchment area raster (``m^2 / m``): a flow
        accumulation in m² (``flow_accumulation(out_units='area')``) divided
        by the contour width, usually the pixel size. A raster tagged as a
        flow accumulation (``RSGISPY_FLOW_ACC_UNITS``) raises
        :class:`rsgispy.RSGISPyException`.
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
            _reject_flow_acc_input(
                sca_src, specific_catchment_area_img, "calc_spi"
            )
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

            if _has_numba():
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
                out_no_data_val=out_nd,
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
        Path to the specific catchment area raster (``m^2 / m``): a flow
        accumulation in m² (``flow_accumulation(out_units='area')``) divided
        by the contour width, usually the pixel size. A raster tagged as a
        flow accumulation (``RSGISPY_FLOW_ACC_UNITS``) raises
        :class:`rsgispy.RSGISPyException`.
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
            _reject_flow_acc_input(
                sca_src, specific_catchment_area_img, "calc_ls_factor"
            )
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

            if _has_numba():
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
                out_no_data_val=out_nd,
                calc_stats=calc_stats,
                use_pbar=use_pbar,
            )
            pbar.update(1)
