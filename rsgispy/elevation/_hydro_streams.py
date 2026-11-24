"""
Stage 3: Stream network extraction, link labelling, ordering, and vectorisation
for :mod:`rsgispy.elevation.hydrology`.
"""

from contextlib import ExitStack
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import rasterio
from shapely.geometry import LineString

import rsgispy
from rsgispy import RSGISPyException
from rsgispy.elevation._hydro_common import (
    _call_topo_kernel,
    _has_numba,
    _run_optional_kernel,
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
    D8_LUT_DC,
    D8_LUT_DR,
    D8_LUT_OK,
    extract_streams_njit,
    extract_streams_numpy,
    stream_link_kernel,
    stream_order_kernel,
    trace_stream_reach_kernel,
)
from rsgispy.imageprocess import build_valid_mask, ensure_file_exists
from rsgispy.vectorprocess import write_vec_gdf


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

            if _has_numba():
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
                out_no_data_val=out_nd,
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
                out_no_data_val=out_nd,
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
                out_no_data_val=out_nd,
                calc_stats=calc_stats,
                use_pbar=use_pbar,
            )
            pbar.update(1)


def vectorise_streams(
    streams_img: str,
    flow_dir_img: str,
    out_vec_file: str,
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
    out_vec_file : str
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
            warn=False,
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

            reach_r, reach_c, down_link_id, conf_r, conf_c = _run_optional_kernel(
                trace_stream_reach_kernel,
                fdir,
                valid,
                streams_bin,
                link_grid,
                r,
                c,
                D8_LUT_OK,
                D8_LUT_DR,
                D8_LUT_DC,
            )
            down_link_id = int(down_link_id)
            if conf_r >= 0:
                # Append downstream confluence vertex for exact topological snapping
                pts_r = np.append(reach_r, conf_r)
                pts_c = np.append(reach_c, conf_c)
                end_r, end_c = int(conf_r), int(conf_c)
            else:
                pts_r = reach_r
                pts_c = reach_c
                end_r, end_c = int(reach_r[-1]), int(reach_c[-1])
            # Pixel-centre -> map coordinates for the whole reach at once
            xs, ys = rasterio.transform.xy(transform, pts_r, pts_c)
            coords: List[Tuple[float, float]] = list(
                zip(
                    np.asarray(xs, dtype=np.float64).tolist(),
                    np.asarray(ys, dtype=np.float64).tolist(),
                )
            )

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
                rec["max_acc"] = float(np.max(acc_arr[reach_r, reach_c]))
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
            out_vec_file,
            out_vec_lyr=out_vec_lyr,
            out_format=out_format,
            overwrite=overwrite,
        )
