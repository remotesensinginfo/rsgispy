"""
Comprehensive pytest suite for :mod:`rsgispy.elevation.hydrology`.
"""

import math
import os
import warnings

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from shapely.geometry import LineString, Point

import rsgispy
from rsgispy import RSGISPyException
from rsgispy._numba import disable_numba
from rsgispy.elevation import hydrology
from rsgispy.elevation._hydro_kernels import _dinf_targets
from rsgispy.imagecalc.maths import BandDefn, band_maths

from conftest import write_raster

# These tests use numba-compiled topological kernels for reference outputs
# (topological ops raise without ``allow_pure_python=True`` when numba is
# unavailable); the no-numba policy is covered by the subprocess test below.
requires_numba = pytest.mark.skipif(
    not hydrology.HAS_NUMBA, reason="numba not installed or disabled"
)


def _write_raster(
    path,
    arr,
    res=10.0,
    nodata=None,
    crs="EPSG:32630",
    origin_x=500000.0,
    origin_y=5200000.0,
):
    return write_raster(
        path, arr, nodata=nodata, crs=crs, origin=(origin_x, origin_y), pixel_size=res
    )


def _make_v_valley_with_dam(rows=25, cols=25, res=10.0, dam=True, nodata=-9999.0):
    """
    Create a V-shaped valley sloping South (row 0 -> row rows-1) with channel
    along central column, optional cross-valley dam creating a closed pit, and
    a small nodata corner.
    """
    r_idx, c_idx = np.mgrid[0:rows, 0:cols]
    mid_c = cols // 2
    dem = 100.0 - 1.5 * r_idx + 2.0 * np.abs(c_idx - mid_c)
    if dam:
        # Artificial dam across row 15..16 trapping rows < 15
        dem[15:17, :] += 12.0
    dem = dem.astype(np.float32)
    dem[0, 0] = nodata
    return dem


def _make_convergent_y_dem(rows=31, cols=31):
    """
    Create a symmetric Y-shaped convergent catchment DEM (no sinks, strictly
    draining to a single outlet at the bottom centre ``(rows-1, 15)``).
    Upper half (r <= 15) has two tributary valleys along cols 8 and 22 sloping
    inward toward col 15; lower half (r >= 15) drains along col 15 to south edge.
    """
    r_idx, c_idx = np.mgrid[0:rows, 0:cols]
    # Base southward slope ensuring monotonic descent along valley lines
    base = (rows - 1 - r_idx) * 2.0

    # Distance from two upper-basin tributaries converging linearly from (0, 8) and (0, 22)
    # to (15, 15), and single mainstem at c = 15 for r >= 15.
    trib_left = 8.0 + (7.0 / 15.0) * np.minimum(r_idx, 15)
    trib_right = 22.0 - (7.0 / 15.0) * np.minimum(r_idx, 15)

    dist_left = np.abs(c_idx - trib_left)
    dist_right = np.abs(c_idx - trib_right)
    dist_main = np.abs(c_idx - 15.0)

    side_slope = np.where(
        r_idx <= 15,
        np.minimum(dist_left, dist_right) * 3.0,
        dist_main * 3.0,
    )
    # Raise outer perimeter slightly on N/E/W edges to concentrate all flow to (rows-1, 15)
    dem = base + side_slope
    # Ensure bottom row drains inward toward outlet (rows-1, 15)
    dem[rows - 1, :] = np.abs(c_idx[rows - 1, :] - 15.0) * 3.0
    return dem.astype(np.float32)


@requires_numba
def test_dem_conditioning_valley_and_dam(tmp_path):
    """
    Verify sink identification, breaching, filling (exact float32 flats),
    stream burning, flat resolution, and combined condition_dem.
    """
    dem = _make_v_valley_with_dam(rows=25, cols=25, res=10.0, dam=True, nodata=-9999.0)
    dem_path = tmp_path / "dem_dam.tif"
    _write_raster(dem_path, dem, res=10.0, nodata=-9999.0)

    sinks_path = tmp_path / "sinks.tif"
    depth_path = tmp_path / "sink_depth.tif"
    hydrology.identify_sinks(
        str(dem_path), str(sinks_path), output_depth_img=str(depth_path), use_pbar=False
    )
    with rasterio.open(sinks_path) as s_ds, rasterio.open(depth_path) as d_ds:
        s_arr = s_ds.read(1)
        d_arr = d_ds.read(1)
        assert s_arr.dtype == np.int32
        assert s_arr[0, 0] == -1
        assert np.any(s_arr > 0)
        assert np.max(d_arr[s_arr > 0]) > 1.0
        assert d_arr[20, 12] == pytest.approx(0.0)

    # 1. Fill depressions: zero remaining sinks + exact float32 flat preservation
    filled_path = tmp_path / "dem_filled.tif"
    filled_sinks = tmp_path / "filled_sinks.tif"
    hydrology.fill_depressions(str(dem_path), str(filled_path), use_pbar=False)
    hydrology.identify_sinks(str(filled_path), str(filled_sinks), use_pbar=False)
    with rasterio.open(filled_path) as f_ds, rasterio.open(filled_sinks) as fs_ds:
        assert f_ds.dtypes[0] == "float32"
        f_arr = f_ds.read(1)
        fs_arr = fs_ds.read(1)
        # Zero sinks remain after Priority-Flood filling
        assert np.all(fs_arr[fs_arr != -1] == 0)
        # Filled flat cells retain identical float32 value (no epsilon creep)
        # Check channel cells along filled impoundment are truly >= dam spill or equal
        assert f_arr[14, 12] == f_arr[13, 12]

    # Resolve flats produces positive decreasing gradient toward outlet
    flats_path = tmp_path / "flats.tif"
    hydrology.resolve_flats(str(filled_path), str(flats_path), use_pbar=False)
    with rasterio.open(flats_path) as fl_ds:
        fl_arr = fl_ds.read(1)
        assert fl_arr.dtype == np.int32
        assert fl_arr[0, 0] == -1
        assert np.any(fl_arr > 0)

    # 2. Breach depressions: cuts through dam, zero remaining sinks
    breached_path = tmp_path / "dem_breached.tif"
    breached_sinks = tmp_path / "breached_sinks.tif"
    hydrology.breach_depressions(
        str(dem_path), str(breached_path), fill_remaining=True, use_pbar=False
    )
    hydrology.identify_sinks(str(breached_path), str(breached_sinks), use_pbar=False)
    with rasterio.open(breached_path) as b_ds, rasterio.open(breached_sinks) as bs_ds:
        b_arr = b_ds.read(1)
        bs_arr = bs_ds.read(1)
        assert np.all(bs_arr[bs_arr != -1] == 0)
        # Dam crest along channel col 12 should be lowered compared to original dam
        assert b_arr[15, 12] < dem[15, 12]

    # 3. AGREE burn_streams
    stream_line = LineString([(500125.0, 5199985.0), (500125.0, 5199765.0)])
    stream_vec = tmp_path / "stream.gpkg"
    gpd.GeoDataFrame({"id": [1], "geometry": [stream_line]}, crs="EPSG:32630").to_file(
        str(stream_vec), driver="GPKG"
    )
    burned_path = tmp_path / "dem_burned.tif"
    hydrology.burn_streams(
        str(dem_path),
        str(stream_vec),
        str(burned_path),
        burn_depth=5.0,
        buffer_dist=25.0,
        smooth_drop=2.0,
        use_pbar=False,
    )
    with rasterio.open(burned_path) as br_ds:
        br_arr = br_ds.read(1)
        assert br_arr[0, 0] == pytest.approx(-9999.0)
        # Centreline cell lowered by burn_depth + smooth_drop = 7.0
        assert dem[10, 12] - br_arr[10, 12] == pytest.approx(7.0, abs=1e-4)
        # Adjacent cell (10 m away < 25 m buffer) lowered by smooth buffer drop
        assert 0.0 < (dem[10, 13] - br_arr[10, 13]) < 2.0

    # 4. condition_dem end-to-end
    cond_path = tmp_path / "dem_cond.tif"
    cond_sinks = tmp_path / "cond_sinks.tif"
    hydrology.condition_dem(
        str(dem_path),
        str(cond_path),
        method="breach_and_fill",
        streams_vec=str(stream_vec),
        use_pbar=False,
    )
    hydrology.identify_sinks(str(cond_path), str(cond_sinks), use_pbar=False)
    with rasterio.open(cond_sinks) as cs_ds:
        cs_arr = cs_ds.read(1)
        assert np.all(cs_arr[cs_arr != -1] == 0)


@pytest.fixture
def y_dem_routing(tmp_path):
    """Conditioned convergent Y-shaped DEM with D8 flow directions, cell
    accumulation and streams (threshold 25 cells)."""
    dem = _make_convergent_y_dem(rows=31, cols=31)
    dem_path = tmp_path / "y_dem.tif"
    _write_raster(dem_path, dem, res=10.0, nodata=-9999.0)

    # Ensure strictly depressionless surface
    cond_path = tmp_path / "y_cond.tif"
    hydrology.fill_depressions(str(dem_path), str(cond_path), use_pbar=False)

    fdir_d8 = tmp_path / "fdir_d8.tif"
    hydrology.flow_direction(str(cond_path), str(fdir_d8), method="d8", use_pbar=False)
    acc_d8 = tmp_path / "acc_d8.tif"
    hydrology.flow_accumulation(str(fdir_d8), str(acc_d8), out_units="cells", use_pbar=False)

    streams_path = tmp_path / "streams.tif"
    hydrology.extract_streams(
        str(acc_d8), str(streams_path), threshold=25.0, threshold_units="cells", use_pbar=False
    )
    return {
        "cond": cond_path,
        "fdir_d8": fdir_d8,
        "acc_d8": acc_d8,
        "streams": streams_path,
    }


@requires_numba
def test_flow_accumulation_d8_dinf_mfd_conserve_drainage_area(tmp_path, y_dem_routing):
    """D8, D-Inf and MFD accumulation conserve the drainage area at the outlet."""
    cond_path = y_dem_routing["cond"]
    fdir_d8 = y_dem_routing["fdir_d8"]
    acc_d8 = y_dem_routing["acc_d8"]

    # D-Inf, MFD flow directions + accumulations
    fdir_dinf = tmp_path / "fdir_dinf.tif"
    fdir_mfd = tmp_path / "fdir_mfd.tif"
    hydrology.flow_direction(str(cond_path), str(fdir_dinf), method="dinf", use_pbar=False)
    hydrology.flow_direction(str(cond_path), str(fdir_mfd), method="mfd", use_pbar=False)

    acc_area = tmp_path / "acc_area.tif"
    acc_dinf = tmp_path / "acc_dinf.tif"
    acc_mfd = tmp_path / "acc_mfd.tif"
    hydrology.flow_accumulation(str(fdir_d8), str(acc_area), out_units="area", use_pbar=False)
    hydrology.flow_accumulation(str(fdir_dinf), str(acc_dinf), out_units="cells", use_pbar=False)
    hydrology.flow_accumulation(str(fdir_mfd), str(acc_mfd), out_units="cells", use_pbar=False)

    n_total = 31 * 31
    with (
        rasterio.open(acc_d8) as d8_ds,
        rasterio.open(acc_area) as ar_ds,
        rasterio.open(acc_dinf) as di_ds,
        rasterio.open(acc_mfd) as m_ds,
    ):
        a8 = d8_ds.read(1)
        aa = ar_ds.read(1)
        ai = di_ds.read(1)
        am = m_ds.read(1)
        # All 961 cells converge to the bottom outlet (30, 15)
        assert a8[30, 15] == pytest.approx(float(n_total), rel=1e-5)
        assert aa[30, 15] == pytest.approx(float(n_total) * 100.0, rel=1e-5)
        assert ai[30, 15] == pytest.approx(float(n_total), rel=1e-4)
        assert am[30, 15] == pytest.approx(float(n_total), rel=1e-4)


@requires_numba
def test_stream_link_order_and_vectorise_streams(tmp_path, y_dem_routing):
    """stream_link, Strahler/Shreve stream_order and vectorise_streams give
    connected links with the documented attributes."""
    cond_path = y_dem_routing["cond"]
    fdir_d8 = y_dem_routing["fdir_d8"]
    acc_d8 = y_dem_routing["acc_d8"]
    streams_path = y_dem_routing["streams"]

    links_path = tmp_path / "links.tif"
    strahler_path = tmp_path / "strahler.tif"
    shreve_path = tmp_path / "shreve.tif"
    hydrology.stream_link(str(streams_path), str(fdir_d8), str(links_path), use_pbar=False)
    hydrology.stream_order(
        str(streams_path), str(fdir_d8), str(strahler_path), method="strahler", use_pbar=False
    )
    hydrology.stream_order(
        str(streams_path), str(fdir_d8), str(shreve_path), method="shreve", use_pbar=False
    )

    with rasterio.open(strahler_path) as st_ds, rasterio.open(shreve_path) as sh_ds:
        st_arr = st_ds.read(1)
        sh_arr = sh_ds.read(1)
        # Two tributary valleys merge into mainstem -> Strahler order 2, Shreve >= 2 at outlet
        assert st_arr[30, 15] >= 2
        assert sh_arr[30, 15] >= 2

    streams_gpkg = tmp_path / "streams.gpkg"
    streams_pq = tmp_path / "streams.parquet"
    hydrology.vectorise_streams(
        str(streams_path),
        str(fdir_d8),
        str(streams_gpkg),
        dem_img=str(cond_path),
        flow_acc_img=str(acc_d8),
        order_method="strahler",
        use_pbar=False,
    )
    hydrology.vectorise_streams(
        str(streams_path),
        str(fdir_d8),
        out_vec_file=str(streams_pq),
        dem_img=str(cond_path),
        flow_acc_img=str(acc_d8),
        order_method="shreve",
        use_pbar=False,
    )
    gdf_gpkg = gpd.read_file(str(streams_gpkg))
    gdf_pq = gpd.read_parquet(str(streams_pq))
    assert len(gdf_gpkg) >= 3
    assert len(gdf_pq) == len(gdf_gpkg)
    expected_cols = {
        "link_id",
        "down_link_id",
        "order",
        "length_m",
        "elev_start",
        "elev_end",
        "drop_m",
        "slope",
        "max_acc",
    }
    assert expected_cols.issubset(gdf_gpkg.columns)
    # Check exact shared vertex snapping between upstream link end and downstream link start
    by_id = gdf_gpkg.set_index("link_id")
    for _, row in gdf_gpkg.iterrows():
        dlid = int(row["down_link_id"])
        if dlid > 0:
            end_pt = row.geometry.coords[-1]
            start_pt = by_id.loc[dlid].geometry.coords[0]
            assert end_pt == pytest.approx(start_pt, abs=1e-6)


@requires_numba
def test_flow_length_and_flow_distance_zero_at_outlet(tmp_path, y_dem_routing):
    """flow_length (down/upstream) and flow_distance (horizontal/vertical)."""
    cond_path = y_dem_routing["cond"]
    fdir_d8 = y_dem_routing["fdir_d8"]
    streams_path = y_dem_routing["streams"]

    flen_down = tmp_path / "flen_down.tif"
    flen_up = tmp_path / "flen_up.tif"
    fdist_h = tmp_path / "fdist_h.tif"
    fdist_v = tmp_path / "fdist_v.tif"
    hydrology.flow_length(str(fdir_d8), str(flen_down), direction="downstream", use_pbar=False)
    hydrology.flow_length(str(fdir_d8), str(flen_up), direction="upstream", use_pbar=False)
    hydrology.flow_distance(
        str(fdir_d8), str(streams_path), str(fdist_h), distance_type="horizontal", use_pbar=False
    )
    hydrology.flow_distance(
        str(fdir_d8),
        str(streams_path),
        str(fdist_v),
        dem_img=str(cond_path),
        distance_type="vertical",
        use_pbar=False,
    )
    with (
        rasterio.open(flen_down) as fld,
        rasterio.open(flen_up) as flu,
        rasterio.open(fdist_h) as fdh,
        rasterio.open(fdist_v) as fdv,
    ):
        assert fld.read(1)[30, 15] == pytest.approx(0.0)
        assert fld.read(1)[0, 0] > 0.0
        assert flu.read(1)[30, 15] > 0.0
        assert fdh.read(1)[30, 15] == pytest.approx(0.0)
        assert fdv.read(1)[30, 15] == pytest.approx(0.0)


@requires_numba
def test_trace_flow_path_downstream_and_upstream(tmp_path, y_dem_routing):
    """trace_flow_path follows the flow down to the outlet and back upstream."""
    fdir_d8 = y_dem_routing["fdir_d8"]
    acc_d8 = y_dem_routing["acc_d8"]

    seed_vec = tmp_path / "seed_pt.gpkg"
    with rasterio.open(fdir_d8) as src:
        sx, sy = rasterio.transform.xy(src.transform, 2, 2)
        ox, oy = rasterio.transform.xy(src.transform, 30, 15)
    gpd.GeoDataFrame({"id": [1], "geometry": [Point(sx, sy)]}, crs="EPSG:32630").to_file(
        str(seed_vec), driver="GPKG"
    )
    trace_down = tmp_path / "trace_down.gpkg"
    hydrology.trace_flow_path(
        str(fdir_d8), str(seed_vec), str(trace_down), direction="downstream", use_pbar=False
    )
    tr_gdf = gpd.read_file(str(trace_down))
    assert tr_gdf.iloc[0]["path_length_m"] > 100.0
    assert tr_gdf.iloc[0].geometry.coords[-1] == pytest.approx((ox, oy), abs=1e-6)

    out_pt_vec = tmp_path / "out_pt.gpkg"
    gpd.GeoDataFrame({"id": [1], "geometry": [Point(ox, oy)]}, crs="EPSG:32630").to_file(
        str(out_pt_vec), driver="GPKG"
    )
    trace_up = tmp_path / "trace_up.gpkg"
    hydrology.trace_flow_path(
        str(fdir_d8),
        str(out_pt_vec),
        out_vec_file=str(trace_up),
        direction="upstream",
        flow_acc_img=str(acc_d8),
        use_pbar=False,
    )
    up_gdf = gpd.read_file(str(trace_up))
    assert up_gdf.iloc[0]["path_length_m"] > 100.0


@requires_numba
def test_watershed_delineation_and_polygonisation(tmp_path):
    """
    Verify snap_pour_points, delineate_watersheds (single & nested gauges),
    delineate_basins, delineate_subcatchments, and polygonise_watersheds
    drainage area conservation.
    """
    dem = _make_convergent_y_dem(rows=31, cols=31)
    dem_path = tmp_path / "y_dem.tif"
    _write_raster(dem_path, dem, res=10.0, nodata=-9999.0)

    fdir_path = tmp_path / "fdir.tif"
    acc_path = tmp_path / "acc.tif"
    streams_path = tmp_path / "streams.tif"
    links_path = tmp_path / "links.tif"
    hydrology.flow_direction(str(dem_path), str(fdir_path), method="d8", use_pbar=False)
    hydrology.flow_accumulation(str(fdir_path), str(acc_path), out_units="cells", use_pbar=False)
    hydrology.extract_streams(str(acc_path), str(streams_path), threshold=25.0, use_pbar=False)
    hydrology.stream_link(str(streams_path), str(fdir_path), str(links_path), use_pbar=False)

    # Place pour point 2 pixels (20 m) off the outlet (30, 15) -> (30, 17)
    with rasterio.open(acc_path) as src:
        px, py = rasterio.transform.xy(src.transform, 30, 17)
        ox, oy = rasterio.transform.xy(src.transform, 30, 15)
        expected_acc = float(src.read(1)[30, 15])

    raw_pts = tmp_path / "raw_pts.gpkg"
    snapped_pts = tmp_path / "snapped_pts.gpkg"
    gpd.GeoDataFrame({"gauge_id": [10], "geometry": [Point(px, py)]}, crs="EPSG:32630").to_file(
        str(raw_pts), driver="GPKG"
    )
    hydrology.snap_pour_points(
        str(raw_pts),
        str(acc_path),
        out_vec_file=str(snapped_pts),
        snap_dist=25.0,
        streams_img=str(streams_path),
        use_pbar=False,
    )
    sn_gdf = gpd.read_file(str(snapped_pts))
    assert (sn_gdf.iloc[0].geometry.x, sn_gdf.iloc[0].geometry.y) == pytest.approx(
        (ox, oy), abs=1e-6
    )
    assert sn_gdf.iloc[0]["snapped_acc"] == pytest.approx(expected_acc)

    ws_path = tmp_path / "ws.tif"
    hydrology.delineate_watersheds(
        str(fdir_path), str(snapped_pts), str(ws_path), id_field="gauge_id", use_pbar=False
    )

    ws_vec = tmp_path / "ws.gpkg"
    hydrology.polygonise_watersheds(
        str(ws_path), out_vec_file=str(ws_vec), dem_img=str(dem_path), dissolve=True, use_pbar=False
    )
    ws_gdf = gpd.read_file(str(ws_vec))
    assert len(ws_gdf) == 1
    assert ws_gdf.iloc[0]["watershed_id"] == 10
    assert ws_gdf.iloc[0]["area_m2"] == pytest.approx(expected_acc * 100.0, rel=1e-5)
    assert ws_gdf.iloc[0]["elev_min"] <= ws_gdf.iloc[0]["elev_mean"] <= ws_gdf.iloc[0]["elev_max"]

    # Nested pour points: upstream gauge + downstream outlet
    with rasterio.open(acc_path) as src:
        ux, uy = rasterio.transform.xy(src.transform, 15, 15)
    nested_pts = tmp_path / "nested_pts.gpkg"
    gpd.GeoDataFrame(
        {"gid": [1, 2], "geometry": [Point(ox, oy), Point(ux, uy)]},
        crs="EPSG:32630",
    ).to_file(str(nested_pts), driver="GPKG")
    nested_ws = tmp_path / "nested_ws.tif"
    hydrology.delineate_watersheds(
        str(fdir_path), str(nested_pts), str(nested_ws), id_field="gid", use_pbar=False
    )
    with rasterio.open(nested_ws) as n_ds:
        n_arr = n_ds.read(1)
        assert np.any(n_arr == 1)
        assert np.any(n_arr == 2)
        assert np.sum((n_arr == 1) | (n_arr == 2)) == int(expected_acc)

    # delineate_basins and delineate_subcatchments
    basins_path = tmp_path / "basins.tif"
    subcats_path = tmp_path / "subcats.tif"
    hydrology.delineate_basins(str(fdir_path), str(basins_path), use_pbar=False)
    hydrology.delineate_subcatchments(
        str(fdir_path), str(links_path), str(subcats_path), use_pbar=False
    )
    with rasterio.open(basins_path) as b_ds, rasterio.open(subcats_path) as sc_ds:
        assert np.all(b_ds.read(1) > 0)
        assert np.all(sc_ds.read(1) > 0)


def _flow_acc_units_scene(tmp_path, res=30.0):
    """D8 directions on the Y DEM plus cells and m2 accumulations."""
    dem = _make_convergent_y_dem(rows=31, cols=31)
    dem_path = tmp_path / "y_dem.tif"
    _write_raster(dem_path, dem, res=res, nodata=-9999.0)
    fdir_path = tmp_path / "fdir_d8.tif"
    hydrology.flow_direction(str(dem_path), str(fdir_path), method="d8", use_pbar=False)
    acc_cells = tmp_path / "acc_cells.tif"
    acc_area = tmp_path / "acc_area.tif"
    hydrology.flow_accumulation(
        str(fdir_path), str(acc_cells), out_units="cells", use_pbar=False
    )
    hydrology.flow_accumulation(
        str(fdir_path), str(acc_area), out_units="area", use_pbar=False
    )
    return fdir_path, acc_cells, acc_area


def _read_band1(path):
    with rasterio.open(path) as src:
        return src.read(1)


@requires_numba
def test_flow_accumulation_records_units(tmp_path):
    """flow_accumulation writes its units as a dataset tag and band description."""
    fdir_path, acc_cells, acc_area = _flow_acc_units_scene(tmp_path)
    with rasterio.open(acc_cells) as src:
        assert src.tags()["RSGISPY_FLOW_ACC_UNITS"] == "cells"
        assert "cells" in src.descriptions[0]
    with rasterio.open(acc_area) as src:
        assert src.tags()["RSGISPY_FLOW_ACC_UNITS"] == "area"
        assert "m2" in src.descriptions[0]

    dem_path = tmp_path / "y_dem.tif"
    for method in ("dinf", "mfd"):
        fdir_m = tmp_path / f"fdir_{method}.tif"
        acc_m = tmp_path / f"acc_{method}.tif"
        hydrology.flow_direction(str(dem_path), str(fdir_m), method=method, use_pbar=False)
        hydrology.flow_accumulation(
            str(fdir_m), str(acc_m), out_units="area", calc_stats=False, use_pbar=False
        )
        with rasterio.open(acc_m) as src:
            assert src.tags()["RSGISPY_FLOW_ACC_UNITS"] == "area"
    assert fdir_path.exists()


@requires_numba
def test_extract_streams_units_combinations_identical(tmp_path):
    """Cells / m2 accumulations with cells / area thresholds give one network."""
    _, acc_cells, acc_area = _flow_acc_units_scene(tmp_path, res=30.0)
    pxl_area = 30.0 * 30.0
    thresh_cells = 25.0

    nets = {}
    for acc_name, acc_img in (("cells", acc_cells), ("area", acc_area)):
        for thr_units, thr in (("cells", thresh_cells), ("area", thresh_cells * pxl_area)):
            out_img = tmp_path / f"streams_{acc_name}_{thr_units}.tif"
            hydrology.extract_streams(
                str(acc_img), str(out_img), threshold=thr,
                threshold_units=thr_units, use_pbar=False,
            )
            nets[(acc_name, thr_units)] = _read_band1(out_img)

    ref = nets[("cells", "cells")]
    assert 0 < np.count_nonzero(ref == 1) < ref.size / 4
    for key, net in nets.items():
        np.testing.assert_array_equal(net, ref, err_msg=str(key))

    # Explicit units that agree with the tag are accepted.
    out_img = tmp_path / "streams_explicit.tif"
    hydrology.extract_streams(
        str(acc_area), str(out_img), threshold=thresh_cells * pxl_area,
        threshold_units="area", acc_units="area", use_pbar=False,
    )
    np.testing.assert_array_equal(_read_band1(out_img), ref)


@requires_numba
def test_extract_streams_acc_units_conflict_and_legacy(tmp_path):
    """acc_units contradicting the tag raises; untagged rasters use acc_units."""
    _, acc_cells, acc_area = _flow_acc_units_scene(tmp_path, res=30.0)
    pxl_area = 30.0 * 30.0
    out_img = tmp_path / "streams.tif"

    with pytest.raises(RSGISPyException, match="RSGISPY_FLOW_ACC_UNITS"):
        hydrology.extract_streams(
            str(acc_cells), str(out_img), threshold=25.0, acc_units="area",
            use_pbar=False,
        )
    with pytest.raises(RSGISPyException, match="RSGISPY_FLOW_ACC_UNITS"):
        hydrology.extract_streams(
            str(acc_area), str(out_img), threshold=25.0, acc_units="cells",
            use_pbar=False,
        )
    with pytest.raises(RSGISPyException, match="acc_units"):
        hydrology.extract_streams(
            str(acc_cells), str(out_img), threshold=25.0, acc_units="m2",
            use_pbar=False,
        )

    hydrology.extract_streams(
        str(acc_cells), str(out_img), threshold=25.0, use_pbar=False
    )
    ref = _read_band1(out_img)

    # Legacy rasters (written before the tag existed) have no units tag.
    with rasterio.open(acc_cells) as src:
        cells_arr, cells_nd = src.read(1), src.nodata
    with rasterio.open(acc_area) as src:
        area_arr, area_nd = src.read(1), src.nodata
    legacy_cells = tmp_path / "legacy_cells.tif"
    legacy_area = tmp_path / "legacy_area.tif"
    _write_raster(legacy_cells, cells_arr, res=30.0, nodata=cells_nd)
    _write_raster(legacy_area, area_arr, res=30.0, nodata=area_nd)
    with rasterio.open(legacy_cells) as src:
        assert "RSGISPY_FLOW_ACC_UNITS" not in src.tags()

    # No tag + acc_units="auto": assumed to be in cells (the old behaviour).
    hydrology.extract_streams(
        str(legacy_cells), str(out_img), threshold=25.0 * pxl_area,
        threshold_units="area", use_pbar=False,
    )
    np.testing.assert_array_equal(_read_band1(out_img), ref)
    # No tag + explicit acc_units="area".
    hydrology.extract_streams(
        str(legacy_area), str(out_img), threshold=25.0 * pxl_area,
        threshold_units="area", acc_units="area", use_pbar=False,
    )
    np.testing.assert_array_equal(_read_band1(out_img), ref)
    hydrology.extract_streams(
        str(legacy_area), str(out_img), threshold=25.0,
        threshold_units="cells", acc_units="area", use_pbar=False,
    )
    np.testing.assert_array_equal(_read_band1(out_img), ref)


@requires_numba
def test_terrain_indices_reject_flow_accumulation(tmp_path):
    """TWI/SPI/LS need a specific catchment area, not a tagged accumulation."""
    _, acc_cells, acc_area = _flow_acc_units_scene(tmp_path, res=30.0)
    slope_path = tmp_path / "slope.tif"
    _write_raster(
        slope_path, np.full((31, 31), 10.0, dtype=np.float32), res=30.0,
        nodata=-9999.0,
    )
    for func in (hydrology.calc_twi, hydrology.calc_spi, hydrology.calc_ls_factor):
        for acc_img in (acc_cells, acc_area):
            with pytest.raises(RSGISPyException, match="specific catchment area"):
                func(
                    str(acc_img), str(slope_path), str(tmp_path / "idx.tif"),
                    use_pbar=False,
                )


@requires_numba
def test_hand_flood_inundation_and_terrain_indices(tmp_path):
    """
    Verify calc_hand (HAND == 0 on streams and monotonic upslope),
    flood_inundation_from_hand (scalar & raster stage), calc_twi, calc_spi,
    and calc_ls_factor across slope units.
    """
    dem = _make_v_valley_with_dam(rows=21, cols=21, res=10.0, dam=False, nodata=-9999.0)
    dem_path = tmp_path / "dem_v.tif"
    _write_raster(dem_path, dem, res=10.0, nodata=-9999.0)

    fdir_path = tmp_path / "fdir.tif"
    acc_path = tmp_path / "acc.tif"
    streams_path = tmp_path / "streams.tif"
    hand_path = tmp_path / "hand.tif"
    hydrology.flow_direction(str(dem_path), str(fdir_path), method="d8", use_pbar=False)
    hydrology.flow_accumulation(str(fdir_path), str(acc_path), out_units="cells", use_pbar=False)
    hydrology.extract_streams(str(acc_path), str(streams_path), threshold=15.0, use_pbar=False)
    hydrology.calc_hand(
        str(dem_path), str(fdir_path), str(streams_path), str(hand_path), use_pbar=False
    )

    with rasterio.open(hand_path) as h_ds, rasterio.open(streams_path) as s_ds:
        h_arr = h_ds.read(1)
        s_arr = s_ds.read(1)
        assert h_arr[0, 0] == pytest.approx(-9999.0)
        assert np.allclose(h_arr[s_arr == 1], 0.0, atol=1e-5)
        # Upslope away from central channel (row 10, cols 10 -> 15) must increase monotonically
        profile = h_arr[10, 10:18]
        assert np.all(np.diff(profile) >= 0.0)
        assert profile[-1] > 0.0

    # Flood inundation (scalar & raster stage)
    flood_scalar = tmp_path / "flood_scalar.tif"
    hydrology.flood_inundation_from_hand(
        str(hand_path), str(flood_scalar), water_depth=5.0, use_pbar=False
    )
    stage_arr = np.full(dem.shape, 5.0, dtype=np.float32)
    stage_path = tmp_path / "stage.tif"
    _write_raster(stage_path, stage_arr, res=10.0)
    flood_raster = tmp_path / "flood_raster.tif"
    hydrology.flood_inundation_from_hand(
        str(hand_path), str(flood_raster), water_depth=str(stage_path), use_pbar=False
    )
    with rasterio.open(flood_scalar) as fs_ds, rasterio.open(flood_raster) as fr_ds:
        fs = fs_ds.read(1)
        fr = fr_ds.read(1)
        np.testing.assert_allclose(fs, fr, rtol=1e-6)
        assert fs[10, 10] == pytest.approx(5.0)
        assert fs[0, 0] == pytest.approx(-9999.0)

    # TWI, SPI, LS Factor exact value verification
    sca_arr = np.full((10, 10), 200.0, dtype=np.float32)
    slp_deg = np.full((10, 10), 15.0, dtype=np.float32)
    slp_rad = np.deg2rad(slp_deg).astype(np.float32)
    slp_pct = (np.tan(slp_rad) * 100.0).astype(np.float32)
    sca_arr[0, 0] = -9999.0

    sca_path = tmp_path / "sca.tif"
    deg_path = tmp_path / "slp_deg.tif"
    rad_path = tmp_path / "slp_rad.tif"
    pct_path = tmp_path / "slp_pct.tif"
    _write_raster(sca_path, sca_arr, res=10.0, nodata=-9999.0)
    _write_raster(deg_path, slp_deg, res=10.0, nodata=-9999.0)
    _write_raster(rad_path, slp_rad, res=10.0, nodata=-9999.0)
    _write_raster(pct_path, slp_pct, res=10.0, nodata=-9999.0)

    twi_deg = tmp_path / "twi_deg.tif"
    twi_pct = tmp_path / "twi_pct.tif"
    spi_path = tmp_path / "spi.tif"
    ls_path = tmp_path / "ls.tif"
    hydrology.calc_twi(
        str(sca_path), str(deg_path), str(twi_deg), slope_units="degrees", use_pbar=False
    )
    hydrology.calc_twi(
        str(sca_path), str(pct_path), str(twi_pct), slope_units="percent", use_pbar=False
    )
    hydrology.calc_spi(
        str(sca_path), str(rad_path), str(spi_path), slope_units="radians", use_pbar=False
    )
    hydrology.calc_ls_factor(
        str(sca_path), str(deg_path), str(ls_path), slope_units="degrees", use_pbar=False
    )

    tan15 = math.tan(math.radians(15.0))
    sin15 = math.sin(math.radians(15.0))
    exp_twi = math.log(200.0 / tan15)
    exp_spi = 200.0 * tan15
    exp_ls = 1.4 * ((200.0 / 22.13) ** 0.4) * ((sin15 / 0.0896) ** 1.3)

    with (
        rasterio.open(twi_deg) as td,
        rasterio.open(twi_pct) as tp,
        rasterio.open(spi_path) as sp,
        rasterio.open(ls_path) as ls,
    ):
        assert td.read(1)[5, 5] == pytest.approx(exp_twi, rel=1e-4)
        assert tp.read(1)[5, 5] == pytest.approx(exp_twi, rel=1e-4)
        assert sp.read(1)[5, 5] == pytest.approx(exp_spi, rel=1e-4)
        assert ls.read(1)[5, 5] == pytest.approx(exp_ls, rel=1e-4)
        assert td.read(1)[0, 0] == pytest.approx(-9999.0)


@requires_numba
def test_numba_fallback_and_validation(tmp_path, monkeypatch):
    """
    Verify HAS_NUMBA = False policy (RSGISPyException by default for topo ops,
    RuntimeWarning + exact parity with allow_pure_python=True, and vectorised
    NumPy parity for stencil ops) as well as CRS & parameter validation.
    """
    dem = _make_v_valley_with_dam(rows=15, cols=15, res=10.0, dam=True, nodata=-9999.0)
    dem_path = tmp_path / "dem.tif"
    _write_raster(dem_path, dem, res=10.0, nodata=-9999.0)

    # Reference outputs with Numba enabled
    fill_nb = tmp_path / "fill_nb.tif"
    fdir_nb = tmp_path / "fdir_nb.tif"
    dinf_nb = tmp_path / "dinf_nb.tif"
    mfd_nb = tmp_path / "mfd_nb.tif"
    acc_nb = tmp_path / "acc_nb.tif"
    str_nb = tmp_path / "str_nb.tif"
    hand_nb = tmp_path / "hand_nb.tif"

    hydrology.fill_depressions(str(dem_path), str(fill_nb), use_pbar=False)
    hydrology.flow_direction(str(fill_nb), str(fdir_nb), method="d8", use_pbar=False)
    hydrology.flow_direction(str(fill_nb), str(dinf_nb), method="dinf", use_pbar=False)
    hydrology.flow_direction(str(fill_nb), str(mfd_nb), method="mfd", use_pbar=False)
    hydrology.flow_accumulation(str(fdir_nb), str(acc_nb), use_pbar=False)
    hydrology.extract_streams(str(acc_nb), str(str_nb), threshold=10.0, use_pbar=False)
    hydrology.calc_hand(str(fill_nb), str(fdir_nb), str(str_nb), str(hand_nb), use_pbar=False)

    # Simulate missing numba
    monkeypatch.setattr(hydrology, "HAS_NUMBA", False)

    fill_py = tmp_path / "fill_py.tif"
    with pytest.raises(RSGISPyException, match="numba"):
        hydrology.fill_depressions(str(dem_path), str(fill_py), use_pbar=False)

    with pytest.warns(RuntimeWarning):
        hydrology.fill_depressions(
            str(dem_path), str(fill_py), allow_pure_python=True, use_pbar=False
        )

    fdir_np = tmp_path / "fdir_np.tif"
    dinf_np = tmp_path / "dinf_np.tif"
    mfd_np = tmp_path / "mfd_np.tif"
    # Filled dam DEM contains flats: flat_routing=True must enforce allow_pure_python
    for meth, out_p in [("d8", fdir_np), ("dinf", dinf_np), ("mfd", mfd_np)]:
        with pytest.raises(RSGISPyException, match="numba"):
            hydrology.flow_direction(
                str(fill_py), str(out_p), method=meth, use_pbar=False
            )
        with pytest.warns(RuntimeWarning) as rec:
            hydrology.flow_direction(
                str(fill_py),
                str(out_p),
                method=meth,
                allow_pure_python=True,
                use_pbar=False,
            )
        assert len(rec) == 1

    # With flat_routing=False, vectorised NumPy path runs without allow_pure_python
    with pytest.warns(RuntimeWarning) as rec_nf:
        hydrology.flow_direction(
            str(fill_py),
            str(tmp_path / "fdir_noflat.tif"),
            method="d8",
            flat_routing=False,
            use_pbar=False,
        )
    assert len(rec_nf) == 1

    acc_py = tmp_path / "acc_py.tif"
    with pytest.raises(RSGISPyException, match="numba"):
        hydrology.flow_accumulation(str(fdir_np), str(acc_py), use_pbar=False)
    with pytest.warns(RuntimeWarning):
        hydrology.flow_accumulation(
            str(fdir_np), str(acc_py), allow_pure_python=True, use_pbar=False
        )

    str_np = tmp_path / "str_np.tif"
    with pytest.warns(RuntimeWarning):
        hydrology.extract_streams(str(acc_py), str(str_np), threshold=10.0, use_pbar=False)

    hand_py = tmp_path / "hand_py.tif"
    with pytest.raises(RSGISPyException, match="numba"):
        hydrology.calc_hand(str(fill_py), str(fdir_np), str(str_np), str(hand_py), use_pbar=False)
    with pytest.warns(RuntimeWarning):
        hydrology.calc_hand(
            str(fill_py),
            str(fdir_np),
            str(str_np),
            str(hand_py),
            allow_pure_python=True,
            use_pbar=False,
        )

    for nb_p, py_p in [
        (fill_nb, fill_py),
        (fdir_nb, fdir_np),
        (dinf_nb, dinf_np),
        (mfd_nb, mfd_np),
        (acc_nb, acc_py),
        (str_nb, str_np),
        (hand_nb, hand_py),
    ]:
        with rasterio.open(nb_p) as a_ds, rasterio.open(py_p) as b_ds:
            np.testing.assert_allclose(a_ds.read(), b_ds.read(), rtol=1e-5, atol=1e-5)

    # Geographic CRS rejection
    geo_dem = tmp_path / "geo_dem.tif"
    _write_raster(geo_dem, dem, res=0.001, crs="EPSG:4326")
    with pytest.raises(RSGISPyException, match="projected"):
        hydrology.flow_direction(str(geo_dem), str(tmp_path / "geo_fdir.tif"), use_pbar=False)


def test_optional_scipy_handling(tmp_path, monkeypatch):
    """
    Verify rsgispy.elevation.hydrology handles missing scipy cleanly:
    identify_sinks and buffered burn_streams raise RSGISPyException, while
    unbuffered burn_streams succeeds without scipy.
    """
    from shapely.geometry import LineString

    dem = _make_v_valley_with_dam(rows=11, cols=11, res=10.0, dam=True, nodata=-9999.0)
    dem_path = tmp_path / "dem.tif"
    _write_raster(dem_path, dem, res=10.0, nodata=-9999.0)

    streams_vec = tmp_path / "streams.gpkg"
    with rasterio.open(dem_path) as src:
        x0, y0 = rasterio.transform.xy(src.transform, 1, 5)
        x1, y1 = rasterio.transform.xy(src.transform, 9, 5)
    gpd.GeoDataFrame(
        {"id": [1], "geometry": [LineString([(x0, y0), (x1, y1)])]},
        crs="EPSG:32630",
    ).to_file(str(streams_vec), driver="GPKG")

    monkeypatch.setattr(hydrology, "HAVE_SCIPY", False)
    monkeypatch.setattr(hydrology, "nd_label", None)
    monkeypatch.setattr(hydrology, "distance_transform_edt", None)

    with pytest.raises(RSGISPyException, match="scipy"):
        hydrology.identify_sinks(
            str(dem_path), str(tmp_path / "sinks.tif"), use_pbar=False
        )

    with pytest.raises(RSGISPyException, match="scipy"):
        hydrology.burn_streams(
            str(dem_path),
            str(streams_vec),
            str(tmp_path / "burn_buf.tif"),
            buffer_dist=20.0,
            use_pbar=False,
        )

    sharp_out = tmp_path / "burn_sharp.tif"
    hydrology.burn_streams(
        str(dem_path),
        str(streams_vec),
        str(sharp_out),
        buffer_dist=None,
        use_pbar=False,
    )
    assert sharp_out.exists()


def test_true_no_numba_subprocess(tmp_path):
    """Verify hydrology import and pure-Python fallbacks with numba blocked at import time."""
    import subprocess
    import sys

    dem = _make_v_valley_with_dam(rows=11, cols=11, res=10.0, dam=True, nodata=-9999.0)
    dem_path = tmp_path / "dem.tif"
    _write_raster(dem_path, dem, res=10.0, nodata=-9999.0)

    script = f"""
import sys, warnings
sys.modules["numba"] = None
import rsgispy
from rsgispy.elevation import hydrology
assert hydrology.HAS_NUMBA is False

dem = {str(dem_path)!r}
tmp = {str(tmp_path)!r}

try:
    hydrology.fill_depressions(dem, tmp + "/fill.tif", use_pbar=False)
    raise AssertionError("Expected RSGISPyException without allow_pure_python")
except rsgispy.RSGISPyException:
    pass

with warnings.catch_warnings(record=True) as rec:
    warnings.simplefilter("always")
    hydrology.fill_depressions(dem, tmp + "/fill.tif", allow_pure_python=True, use_pbar=False)
    rt = [w for w in rec if issubclass(w.category, RuntimeWarning)]
    assert len(rt) == 1, f"Expected 1 RuntimeWarning, got {{len(rt)}}"

for meth in ("d8", "dinf", "mfd"):
    out_p = tmp + f"/fdir_{{meth}}.tif"
    try:
        hydrology.flow_direction(tmp + "/fill.tif", out_p, method=meth, use_pbar=False)
        raise AssertionError(f"Expected RSGISPyException for {{meth}} flat routing")
    except rsgispy.RSGISPyException:
        pass
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        hydrology.flow_direction(
            tmp + "/fill.tif", out_p, method=meth, allow_pure_python=True, use_pbar=False
        )
        rt = [w for w in rec if issubclass(w.category, RuntimeWarning)]
        assert len(rt) == 1, f"Expected 1 RuntimeWarning for {{meth}}, got {{len(rt)}}"

with warnings.catch_warnings(record=True) as rec:
    warnings.simplefilter("always")
    hydrology.flow_direction(
        tmp + "/fill.tif", tmp + "/fdir_nf.tif", method="d8", flat_routing=False, use_pbar=False
    )
    rt = [w for w in rec if issubclass(w.category, RuntimeWarning)]
    assert len(rt) == 1

with warnings.catch_warnings(record=True) as rec:
    warnings.simplefilter("always")
    hydrology.flow_accumulation(
        tmp + "/fdir_d8.tif", tmp + "/acc.tif", allow_pure_python=True, use_pbar=False
    )
    hydrology.extract_streams(tmp + "/acc.tif", tmp + "/str.tif", threshold=5.0, use_pbar=False)
    hydrology.stream_link(
        tmp + "/fdir_d8.tif",
        tmp + "/str.tif",
        tmp + "/lnk.tif",
        allow_pure_python=True,
        use_pbar=False,
    )
"""
    res = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, f"Subprocess failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"


# ---------------------------------------------------------------------------
# D-Infinity flow accumulation and depression breaching
# ---------------------------------------------------------------------------
# D-Infinity flow accumulation must not lose flow through float32
# facet-boundary angles (uphill edges forming cycles in the flow graph). The
# tests on the tutorial DEM are slow and need the tutorial data, so they only
# run when the ``RSGISPY_TUTORIAL_DEM`` environment variable gives the path to
# ``dem_10m_osgb.tif`` (the sea stored as 0 m).
TUTORIAL_DEM = os.environ.get("RSGISPY_TUTORIAL_DEM", "")
requires_tutorial_dem = pytest.mark.skipif(
    not (TUTORIAL_DEM and os.path.isfile(TUTORIAL_DEM)),
    reason="set RSGISPY_TUTORIAL_DEM to the tutorial dem_10m_osgb.tif",
)


ND = -9999.0


def _read(path):
    with rasterio.open(str(path)) as src:
        return src.read(1)


def _synthetic_dems():
    n = 61
    r, c = np.indices((n, n), dtype=np.float64)
    rng = np.random.default_rng(42)
    return {
        "plane_45": 1000.0 - r - c,
        "cone": 100.0 - np.hypot(r - 30, c - 30),
        "pyramid": 100.0 - np.maximum(np.abs(r - 30), np.abs(c - 30)),
        "diamond": 100.0 - np.abs(r - 30) - np.abs(c - 30),
        "ridges": 100.0 - np.minimum(np.abs(r - c), np.abs(r + c - 60)) - 0.1 * r,
        "random_int": np.asarray(rng.integers(0, 10, (n, n)), dtype=np.float64),
    }


def _dinf_terminal_total(tmp_path, name, dem, res=10.0, pure_python=False):
    """Fill, D-Inf route and accumulate ``dem``; return (outlet total, n valid, acc)."""
    dem_p = tmp_path / f"{name}_dem.tif"
    fill_p = tmp_path / f"{name}_fill.tif"
    fdir_p = tmp_path / f"{name}_dinf.tif"
    acc_p = tmp_path / f"{name}_acc.tif"
    write_raster(dem_p, dem, pixel_size=res, dtype="float32", nodata=ND)
    kw = {"allow_pure_python": True} if pure_python else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        hydrology.fill_depressions(str(dem_p), str(fill_p), use_pbar=False, **kw)
        hydrology.flow_direction(
            str(fill_p), str(fdir_p), method="dinf", use_pbar=False, **kw
        )
    with warnings.catch_warnings():
        # A cycle warning is a failure here, but the NumPy fallback warns too.
        warnings.filterwarnings("error", message=".*cycle.*")
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        hydrology.flow_accumulation(str(fdir_p), str(acc_p), use_pbar=False, **kw)
    ang = _read(fdir_p)
    acc = _read(acc_p)
    # Every cell drains to a terminal cell (no downslope neighbour, -1).
    terminal = ang == -1.0
    return float(acc[terminal].astype(np.float64).sum()), int(np.sum(ang != ND)), acc


@pytest.fixture(scope="module")
def tutorial_land_dem(tmp_path_factory):
    """The tutorial DEM with the sea (stored as 0 m) set to no data."""
    out_dir = tmp_path_factory.mktemp("tutorial_dem")
    dem_img = str(out_dir / "dem_land.tif")
    band_maths(
        dem_img,
        "dem <= 0 ? -32768 : dem",
        [BandDefn("dem", TUTORIAL_DEM, 1)],
        datatype=rsgispy.TYPE_32FLOAT,
        no_data_val=-32768,
        use_pbar=False,
    )
    return out_dir, dem_img


def _breach(tmp_path, name, dem, **kw):
    dem_p = tmp_path / f"{name}_dem.tif"
    out_p = tmp_path / f"{name}_breached.tif"
    write_raster(dem_p, dem, dtype="float32", nodata=ND)
    hydrology.breach_depressions(str(dem_p), str(out_p), use_pbar=False, **kw)
    return _read(out_p).astype(np.float64), out_p


@pytest.mark.parametrize(
    "ang",
    [
        0.0,
        0.25 * math.pi,
        0.5 * math.pi,
        0.75 * math.pi,
        math.pi,
        1.25 * math.pi,
        1.5 * math.pi,
        1.75 * math.pi,
        2.0 * math.pi,
    ],
)
def test_dinf_targets_snaps_float32_facet_boundaries(ang):
    """Boundary angles stored as float32 send all the flow to one neighbour."""
    f32 = float(np.float32(ang))
    for a in (f32, float(np.nextafter(np.float32(ang), np.float32(10.0))),
              float(np.nextafter(np.float32(ang), np.float32(-10.0)))):
        if a < 0.0:
            continue
        dr1, dc1, _, _, w1, w2 = _dinf_targets(a, 10.0, 10.0)
        assert w1 == 1.0 and w2 == 0.0
        k = int(round((ang % (2.0 * math.pi)) / (0.25 * math.pi))) % 8
        exp = [(0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1)][k]
        assert (dr1, dc1) == exp


def test_dinf_targets_splits_interior_angles_and_non_square_diagonal():
    """Interior angles still split; non-square diagonals snap to the diagonal."""
    dr1, dc1, dr2, dc2, w1, w2 = _dinf_targets(0.2, 10.0, 10.0)
    assert (dr1, dc1, dr2, dc2) == (0, 1, -1, 1)
    assert w2 == pytest.approx(0.2 / (0.25 * math.pi))
    assert w1 + w2 == pytest.approx(1.0)
    phi = float(np.float32(math.atan2(20.0, 10.0)))
    dr1, dc1, _, _, w1, w2 = _dinf_targets(phi, 10.0, 20.0)
    assert (dr1, dc1, w1, w2) == (-1, 1, 1.0, 0.0)


@requires_numba
@pytest.mark.parametrize("name", list(_synthetic_dems()))
def test_dinf_accumulation_mass_balance(tmp_path, name):
    """The D-Inf outlet total equals the number of valid cells."""
    total, n_valid, _ = _dinf_terminal_total(tmp_path, name, _synthetic_dems()[name])
    assert total == pytest.approx(n_valid, rel=1e-6)


@requires_numba
def test_dinf_accumulation_mass_balance_non_square_pixels(tmp_path):
    """Flats routed with D8 angles keep their flow with non-square pixels."""
    dem = _synthetic_dems()["random_int"]
    dem_p = tmp_path / "ns_dem.tif"
    with rasterio.open(
        str(dem_p), "w", driver="GTiff", height=dem.shape[0], width=dem.shape[1],
        count=1, dtype="float32", crs="EPSG:32630", nodata=ND,
        transform=rasterio.transform.Affine(10.0, 0.0, 500000.0, 0.0, -20.0, 5200000.0),
    ) as dst:
        dst.write(dem.astype(np.float32), 1)
    fill_p = tmp_path / "ns_fill.tif"
    fdir_p = tmp_path / "ns_dinf.tif"
    acc_p = tmp_path / "ns_acc.tif"
    hydrology.fill_depressions(str(dem_p), str(fill_p), use_pbar=False)
    hydrology.flow_direction(str(fill_p), str(fdir_p), method="dinf", use_pbar=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        hydrology.flow_accumulation(str(fdir_p), str(acc_p), use_pbar=False)
    ang = _read(fdir_p)
    acc = _read(acc_p)
    assert float(acc[ang == -1.0].sum()) == pytest.approx(dem.size, rel=1e-6)


@requires_numba
def test_dinf_accumulation_numba_numpy_agree(tmp_path, monkeypatch):
    """The numba and pure-Python / NumPy D-Inf paths give the same result."""
    dem = _synthetic_dems()["random_int"]
    total_nb, n_valid, acc_nb = _dinf_terminal_total(tmp_path, "nb", dem)
    monkeypatch.setattr(hydrology, "HAS_NUMBA", False)
    with disable_numba():
        total_py, _, acc_py = _dinf_terminal_total(tmp_path, "py", dem, pure_python=True)
    assert total_py == pytest.approx(n_valid, rel=1e-6)
    assert total_nb == pytest.approx(total_py, rel=1e-6)
    np.testing.assert_allclose(acc_nb, acc_py, rtol=1e-5, atol=1e-5)


@requires_numba
@pytest.mark.parametrize("mode", ["d8", "dinf", "mfd"])
def test_flow_accumulation_warns_on_cycle(tmp_path, mode):
    """A cycle in the flow directions is reported with the number of cells."""
    # Row of 4 cells: 0 -> 1 <-> 2 (cycle), 3 drains into 2.
    if mode == "d8":
        fdir = np.array([[1, 1, 16, 16]], dtype=np.uint8)
        write_raster(tmp_path / "f.tif", fdir, nodata=255, dtype="uint8")
    elif mode == "dinf":
        fdir = np.array([[0.0, 0.0, math.pi, math.pi]], dtype=np.float32)
        write_raster(tmp_path / "f.tif", fdir, dtype="float32", nodata=ND)
    else:
        fdir = np.zeros((8, 1, 4), dtype=np.float32)
        fdir[0, 0, :2] = 1.0  # East
        fdir[4, 0, 2:] = 1.0  # West
        write_raster(tmp_path / "f.tif", fdir, dtype="float32", nodata=ND)
    with pytest.warns(RuntimeWarning, match=r"cycle, so 2 cells"):
        hydrology.flow_accumulation(
            str(tmp_path / "f.tif"), str(tmp_path / "a.tif"), use_pbar=False
        )


@requires_numba
@requires_tutorial_dem
def test_tutorial_dem_dinf_outlet_total_matches_d8_mfd(tutorial_land_dem):
    """On the tutorial DEM, D-Inf, D8 and MFD outlet totals agree to 0.1%."""
    out_dir, dem_img = tutorial_land_dem
    cond = str(out_dir / "cond.tif")
    hydrology.condition_dem(
        dem_img, cond, method="breach_and_fill", max_breach_depth=5.0,
        max_breach_length=30, use_pbar=False,
    )
    n_valid = int(np.sum(_read(dem_img) != -32768))
    totals = {}
    for mode in ("d8", "dinf", "mfd"):
        fdir = str(out_dir / f"fdir_{mode}.tif")
        acc = str(out_dir / f"acc_{mode}.tif")
        hydrology.flow_direction(cond, fdir, method=mode, use_pbar=False)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            hydrology.flow_accumulation(fdir, acc, use_pbar=False)
        f_arr = _read(fdir)
        a_arr = _read(acc).astype(np.float64)
        if mode == "d8":
            terminal = f_arr == 0
        elif mode == "dinf":
            terminal = f_arr == -1.0
        else:
            with rasterio.open(fdir) as src:
                terminal = (src.read().sum(axis=0) == 0.0) & (f_arr != ND)
        totals[mode] = a_arr[terminal].sum()
    for mode, total in totals.items():
        assert total == pytest.approx(n_valid, rel=1e-3), mode


@requires_numba
def test_breach_to_higher_adjacent_edge_cell(tmp_path):
    """
    A pit at 196.27 m next to an edge cell at 196.81 m is breached through
    that cell (a 0.54 m cut), not through high terrain to a lower edge cell.
    """
    r, c = np.indices((7, 12), dtype=np.float64)
    dem = 300.0 + 0.5 * (r + c)  # drains to the (0, 0) corner, no flats
    dem[1, 1] = 196.27
    dem[0, 1] = 196.81
    dem[6, 11] = 150.0  # a lower edge cell far away over the high terrain
    out, _ = _breach(tmp_path, "edge", dem)
    cut = np.asarray(dem, dtype=np.float32).astype(np.float64) - out
    assert np.count_nonzero(cut > 1e-6) == 1
    assert cut[0, 1] == pytest.approx(0.54, abs=1e-3)
    assert out[0, 1] < out[1, 1]
    assert cut.max() < 0.55


@requires_numba
def test_breach_cheapest_exit_through_edge(tmp_path):
    """
    A pit whose only lower cells are behind a high ridge is breached by the
    short, shallow route to the DEM edge instead.
    """
    r, c = np.indices((15, 15), dtype=np.float64)
    # A cone rising 2 m per pixel from the pit at (7, 3): the west edge is
    # 6 m above the pit, while the lowland in the east (below the pit) is
    # behind 16 m of cone.
    dem = 100.0 + 2.0 * np.hypot(r - 7, c - 3)
    dem[:, 12:] = 90.0 - 5.0 * (c[:, 12:] - 12)
    out, out_p = _breach(tmp_path, "west", dem)
    cut = np.asarray(dem, dtype=np.float32).astype(np.float64) - out
    changed = cut > 1e-6
    assert np.count_nonzero(changed) == 3
    assert np.all(changed[7, 0:3])
    assert cut.max() == pytest.approx(6.0, abs=1e-3)
    assert np.all(np.diff(out[7, 0:4]) > 0.0)  # descends from the pit to the edge
    fdir_p = tmp_path / "west_fdir.tif"
    hydrology.flow_direction(str(out_p), str(fdir_p), method="d8", use_pbar=False)
    assert int(_read(fdir_p)[7, 3]) == 16  # west


@requires_numba
def test_breach_numba_numpy_agree(tmp_path, monkeypatch):
    """The numba and pure-Python breaching give the same result."""
    rng = np.random.default_rng(7)
    dem = 100.0 + np.cumsum(rng.normal(0.0, 1.0, (40, 40)), axis=0)
    out_nb, _ = _breach(tmp_path, "nb", dem)
    monkeypatch.setattr(hydrology, "HAS_NUMBA", False)
    with disable_numba(), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out_py, _ = _breach(tmp_path, "py", dem, allow_pure_python=True)
    np.testing.assert_allclose(out_nb, out_py, rtol=0.0, atol=1e-5)


@requires_numba
@requires_tutorial_dem
def test_tutorial_dem_unconstrained_breach_is_shallow(tutorial_land_dem):
    """On the tutorial DEM the deepest unconstrained cut is a few metres."""
    out_dir, dem_img = tutorial_land_dem
    out = str(out_dir / "breached.tif")
    hydrology.breach_depressions(dem_img, out, use_pbar=False)
    raw = _read(dem_img).astype(np.float64)
    vld = raw != -32768
    cut = (raw - _read(out).astype(np.float64))[vld]
    assert cut.max() < 15.0
