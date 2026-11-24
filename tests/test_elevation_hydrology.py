"""
Comprehensive pytest suite for :mod:`rsgispy.elevation.hydrology`.
"""

import inspect
import math

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString, Point

import rsgispy
from rsgispy import RSGISPyException
from rsgispy.elevation import hydrology


def _write_raster(
    path,
    arr,
    res=10.0,
    nodata=None,
    crs="EPSG:32630",
    origin_x=500000.0,
    origin_y=5200000.0,
):
    data = np.asarray(arr)
    if data.ndim == 2:
        data = data[np.newaxis, :, :]
    count, height, width = data.shape
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": count,
        "dtype": str(data.dtype),
        "crs": crs,
        "transform": from_origin(origin_x, origin_y, res, res),
    }
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(str(path), "w", **profile) as dst:
        dst.write(data)


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


def test_hydrology_signatures_and_exports():
    """Verify module exports and rsgispy positional/keyword-only conventions."""
    assert hasattr(rsgispy.elevation, "hydrology")
    expected_kw_defaults = {
        "gdalformat": "GTIFF",
        "datatype": None,
        "no_data_val": None,
        "calc_stats": True,
        "use_pbar": True,
    }
    single_input_raster_funcs = [
        hydrology.identify_sinks,
        hydrology.fill_depressions,
        hydrology.breach_depressions,
        hydrology.resolve_flats,
        hydrology.condition_dem,
        hydrology.flow_direction,
        hydrology.flow_accumulation,
        hydrology.flow_length,
        hydrology.extract_streams,
        hydrology.delineate_basins,
        hydrology.flood_inundation_from_hand,
    ]
    for fn in single_input_raster_funcs:
        sig = inspect.signature(fn)
        req_pos = [
            p.name
            for p in sig.parameters.values()
            if p.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
            and p.default is inspect.Parameter.empty
        ]
        assert req_pos[:2] == ["input_img", "output_img"], fn.__name__
        for kw_name, kw_default in expected_kw_defaults.items():
            assert kw_name in sig.parameters, (fn.__name__, kw_name)
            p = sig.parameters[kw_name]
            assert p.kind == inspect.Parameter.KEYWORD_ONLY, (fn.__name__, kw_name)
            assert p.default == kw_default, (fn.__name__, kw_name)

    multi_input_raster_funcs = [
        hydrology.burn_streams,
        hydrology.flow_distance,
        hydrology.stream_link,
        hydrology.stream_order,
        hydrology.delineate_watersheds,
        hydrology.delineate_subcatchments,
        hydrology.calc_hand,
        hydrology.calc_twi,
        hydrology.calc_spi,
        hydrology.calc_ls_factor,
    ]
    for fn in multi_input_raster_funcs:
        sig = inspect.signature(fn)
        for kw_name, kw_default in expected_kw_defaults.items():
            assert kw_name in sig.parameters, (fn.__name__, kw_name)
            p = sig.parameters[kw_name]
            assert p.kind == inspect.Parameter.KEYWORD_ONLY, (fn.__name__, kw_name)
            assert p.default == kw_default, (fn.__name__, kw_name)


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


def test_flow_routing_accumulation_and_stream_topology(tmp_path):
    """
    Test D8, D-Inf, MFD drainage area conservation, flow_length, flow_distance,
    trace_flow_path, extract_streams, stream_link, stream_order, and vectorise_streams.
    """
    dem = _make_convergent_y_dem(rows=31, cols=31)
    dem_path = tmp_path / "y_dem.tif"
    _write_raster(dem_path, dem, res=10.0, nodata=-9999.0)

    # Ensure strictly depressionless surface
    cond_path = tmp_path / "y_cond.tif"
    hydrology.fill_depressions(str(dem_path), str(cond_path), use_pbar=False)

    # D8, D-Inf, MFD flow directions + accumulations
    fdir_d8 = tmp_path / "fdir_d8.tif"
    fdir_dinf = tmp_path / "fdir_dinf.tif"
    fdir_mfd = tmp_path / "fdir_mfd.tif"
    hydrology.flow_direction(str(cond_path), str(fdir_d8), method="d8", use_pbar=False)
    hydrology.flow_direction(str(cond_path), str(fdir_dinf), method="dinf", use_pbar=False)
    hydrology.flow_direction(str(cond_path), str(fdir_mfd), method="mfd", use_pbar=False)

    acc_d8 = tmp_path / "acc_d8.tif"
    acc_area = tmp_path / "acc_area.tif"
    acc_dinf = tmp_path / "acc_dinf.tif"
    acc_mfd = tmp_path / "acc_mfd.tif"
    hydrology.flow_accumulation(str(fdir_d8), str(acc_d8), out_units="cells", use_pbar=False)
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

    # Stream extraction + link + Strahler/Shreve orders + vectorisation
    streams_path = tmp_path / "streams.tif"
    hydrology.extract_streams(
        str(acc_d8), str(streams_path), threshold=25.0, threshold_units="cells", use_pbar=False
    )

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

    # Flow length & flow distance & trace_flow_path
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
    hydrology.calc_twi(str(sca_path), str(deg_path), str(twi_deg), slope_units="degrees", use_pbar=False)
    hydrology.calc_twi(str(sca_path), str(pct_path), str(twi_pct), slope_units="percent", use_pbar=False)
    hydrology.calc_spi(str(sca_path), str(rad_path), str(spi_path), slope_units="radians", use_pbar=False)
    hydrology.calc_ls_factor(str(sca_path), str(deg_path), str(ls_path), slope_units="degrees", use_pbar=False)

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
    with pytest.warns(RuntimeWarning):
        hydrology.flow_direction(str(fill_py), str(fdir_np), method="d8", use_pbar=False)
    with pytest.warns(RuntimeWarning):
        hydrology.flow_direction(str(fill_py), str(dinf_np), method="dinf", use_pbar=False)
    with pytest.warns(RuntimeWarning):
        hydrology.flow_direction(str(fill_py), str(mfd_np), method="mfd", use_pbar=False)

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
