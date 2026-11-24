"""
Stage 4: Pour-point snapping, watershed and basin delineation, subcatchments,
and watershed polygonisation for :mod:`rsgispy.elevation.hydrology`.
"""

from contextlib import ExitStack
import math
from typing import Any, Dict, List, Optional

import geopandas as gpd
import numpy as np
import pyproj
import rasterio
import rasterio.features
from shapely.geometry import Point, shape

import rsgispy
from rsgispy import RSGISPyException
from rsgispy.elevation._hydro_common import (
    _call_topo_kernel,
    _read_d8_grid,
    _resolve_out_dtype,
    _resolve_out_nodata,
    _run_progress_steps,
    _validate_matching_grid,
    _validate_projected_crs,
    _write_raster_array,
)
from rsgispy.elevation._hydro_kernels import (
    find_terminal_outlets_kernel,
    reverse_bfs_label_kernel,
)
from rsgispy.imageprocess import build_valid_mask, ensure_file_exists
from rsgispy.vectorprocess import read_vec_gdf, write_vec_gdf


def snap_pour_points(
    pour_points_vec: str,
    flow_acc_img: str,
    out_vec_file: str,
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
    out_vec_file : str
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
            out_vec_file,
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
                out_no_data_val=out_nd,
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
                warn=False,
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
                out_no_data_val=out_nd,
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
                out_no_data_val=out_nd,
                calc_stats=calc_stats,
                use_pbar=use_pbar,
            )
            pbar.update(1)


def polygonise_watersheds(
    watershed_img: str,
    out_vec_file: str,
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
    out_vec_file : str
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
                out_vec_file,
                out_vec_lyr=out_vec_lyr,
                out_format=out_format,
                overwrite=overwrite,
            )
            pbar.update(1)
