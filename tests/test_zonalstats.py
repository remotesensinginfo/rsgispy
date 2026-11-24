"""
Tests for rsgispy.zonalstats using rasterio and geopandas.
"""

import importlib.util

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point, box

import rsgispy
import rsgispy.zonalstats.point as zs_pt
import rsgispy.zonalstats.polygon as zs_poly


def _write_test_raster(path: str, data: np.ndarray, epsg: int = 32630, nodata=None) -> None:
    """Write a 2D array as a 1-band GeoTIFF with 10m pixels rooted at (500000, 5200000)."""
    height, width = data.shape
    transform = from_origin(500000.0, 5200000.0, 10.0, 10.0)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=data.dtype,
        crs=f"EPSG:{epsg}",
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data, 1)


def _write_multiband_test_raster(
    path: str, data: np.ndarray, epsg: int = 32630, nodata=None
) -> None:
    """Write a 3D (bands, H, W) array as a GeoTIFF rooted at (500000, 5200000)."""
    count, height, width = data.shape
    transform = from_origin(500000.0, 5200000.0, 10.0, 10.0)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=data.dtype,
        crs=f"EPSG:{epsg}",
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data)


def test_calc_zonal_band_stats_and_file(tmp_path):
    """Test calc_zonal_band_stats and calc_zonal_band_stats_file with nodata & thresholds."""
    # 6x6 grid: x in [500000, 500060], y in [5199940, 5200000]
    arr = np.array(
        [
            [1.0, 2.0, 2.0, 4.0, 5.0, 6.0],
            [1.0, 2.0, 10.0, 999.0, 5.0, 6.0],
            [3.0, 3.0, 4.0, 5.0, 5.0, 6.0],
            [-9999.0, 8.0, 8.0, 8.0, 9.0, 9.0],
            [1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
            [4.0, 4.0, 5.0, 5.0, 6.0, 6.0],
        ],
        dtype=np.float32,
    )
    tif_path = str(tmp_path / "vals.tif")
    _write_test_raster(tif_path, arr, nodata=-9999.0)

    # Poly 1 covers top-left 4x4 (cols 0..3, rows 0..3) including row1 col3=999
    # (above max_thres=100) and row3 col0=-9999 (nodata)
    poly1 = box(500000.0, 5199960.0, 500040.0, 5200000.0)
    # Poly 2 completely outside valid thresholds (row 1 col 3 only: 999)
    poly2 = box(500031.0, 5199981.0, 500039.0, 5199989.0)

    gdf = gpd.GeoDataFrame(
        {"id": [1, 2]},
        geometry=[poly1, poly2],
        crs="EPSG:32630",
    )

    out_gdf = zs_poly.calc_zonal_band_stats(
        gdf,
        tif_path,
        img_band=1,
        min_thres=0.0,
        max_thres=100.0,
        out_no_data_val=-1.0,
        min_field="MinVal",
        max_field="MaxVal",
        mean_field="MeanVal",
        stddev_field="StdVal",
        sum_field="SumVal",
        count_field="CountVal",
        mode_field="ModeVal",
        median_field="MedVal",
    )

    # Valid pixels in poly1: [1,2,2,4, 1,2,10, 3,3,4,5, 8,8,8] -> 14 pixels
    expected_vals = np.array([1, 2, 2, 4, 1, 2, 10, 3, 3, 4, 5, 8, 8, 8], dtype=float)
    assert out_gdf.loc[0, "minval"] == pytest.approx(expected_vals.min())
    assert out_gdf.loc[0, "maxval"] == pytest.approx(expected_vals.max())
    assert out_gdf.loc[0, "meanval"] == pytest.approx(expected_vals.mean())
    assert out_gdf.loc[0, "stdval"] == pytest.approx(expected_vals.std())
    assert out_gdf.loc[0, "sumval"] == pytest.approx(expected_vals.sum())
    assert out_gdf.loc[0, "countval"] == pytest.approx(14.0)
    assert out_gdf.loc[0, "modeval"] == pytest.approx(2.0)
    assert out_gdf.loc[0, "medval"] == pytest.approx(np.median(expected_vals))

    # Poly 2 has only 999.0 which is > max_thres=100.0 -> out_no_data_val for stats, 0 for count
    assert out_gdf.loc[1, "meanval"] == pytest.approx(-1.0)
    assert out_gdf.loc[1, "countval"] == pytest.approx(0.0)

    # File-based GPKG test
    gpkg_path = str(tmp_path / "polys.gpkg")
    gpd.GeoDataFrame({"id": [1, 2]}, geometry=[poly1, poly2], crs="EPSG:32630").to_file(
        gpkg_path, layer="polys"
    )
    zs_poly.calc_zonal_band_stats_file(
        gpkg_path,
        "polys",
        tif_path,
        img_band=1,
        min_thres=0.0,
        max_thres=100.0,
        out_no_data_val=-1.0,
        mean_field="mean_f",
    )
    read_back = gpd.read_file(gpkg_path, layer="polys")
    assert read_back.loc[0, "mean_f"] == pytest.approx(expected_vals.mean())
    assert read_back.loc[1, "mean_f"] == pytest.approx(-1.0)

    # Validation errors
    with pytest.raises(rsgispy.RSGISPyException):
        zs_poly.calc_zonal_band_stats(gdf, tif_path, 1, 0.0, 100.0, -1.0)
    with pytest.raises(rsgispy.RSGISPyException):
        zs_poly.calc_zonal_band_stats(
            gdf.to_crs("EPSG:4326"), tif_path, 1, 0.0, 100.0, -1.0, mean_field="m"
        )
    with pytest.raises(rsgispy.RSGISPyException):
        zs_poly.calc_zonal_band_stats(
            gpd.GeoDataFrame({"id": [1]}, geometry=[box(0, 0, 10, 10)], crs="EPSG:32630"),
            tif_path,
            1,
            0.0,
            100.0,
            -1.0,
            mean_field="m",
        )


def test_calc_zonal_poly_pts_and_test_poly_pts(tmp_path):
    """Test centroid-based stats and hybrid polygon/centroid fallback with percentiles."""
    arr = np.arange(1, 26, dtype=np.float32).reshape(5, 5)
    tif_path = str(tmp_path / "grid.tif")
    _write_test_raster(tif_path, arr, nodata=0.0)

    # Poly covering 2x2 pixels (rows 0..1, cols 0..1 -> values 1, 2, 6, 7)
    big_poly = box(500000.0, 5199980.0, 500020.0, 5200000.0)
    # Tiny poly strictly inside pixel (row=2, col=3 -> value 14) not containing
    # pixel centre (500035, 5199975)
    tiny_poly = box(500031.0, 5199971.0, 500033.0, 5199973.0)

    gpkg_path = str(tmp_path / "poly_pts.gpkg")
    gpd.GeoDataFrame({"id": [1, 2]}, geometry=[big_poly, tiny_poly], crs="EPSG:32630").to_file(
        gpkg_path, layer="zones"
    )

    zs_poly.calc_zonal_poly_pts_band_stats_file(
        gpkg_path, "zones", tif_path, img_band=1, out_field="CentVal"
    )
    res_cent = gpd.read_file(gpkg_path, layer="zones")
    # Centroid of big_poly is (500010, 5199990) -> row 1, col 1 -> value 7.0
    assert res_cent.loc[0, "centval"] == pytest.approx(7.0)
    assert res_cent.loc[1, "centval"] == pytest.approx(14.0)

    zs_poly.calc_zonal_band_stats_test_poly_pts_file(
        gpkg_path,
        "zones",
        tif_path,
        img_band=1,
        min_thres=0.0,
        max_thres=100.0,
        out_no_data_val=-99.0,
        percentile=50.0,
        percentile_field="P50",
        mean_field="MeanV",
        stddev_field="StdV",
        count_field="CntV",
    )
    res_hybrid = gpd.read_file(gpkg_path, layer="zones")
    assert res_hybrid.loc[0, "meanv"] == pytest.approx(np.mean([1.0, 2.0, 6.0, 7.0]))
    assert res_hybrid.loc[0, "p50"] == pytest.approx(4.0)
    assert res_hybrid.loc[0, "cntv"] == pytest.approx(4.0)

    # Tiny poly falls back to centroid -> val=14.0, std=0.0, count=1.0
    assert res_hybrid.loc[1, "meanv"] == pytest.approx(14.0)
    assert res_hybrid.loc[1, "stdv"] == pytest.approx(0.0)
    assert res_hybrid.loc[1, "cntv"] == pytest.approx(1.0)
    assert res_hybrid.loc[1, "p50"] == pytest.approx(14.0)

    # Percentile validation errors
    with pytest.raises(rsgispy.RSGISPyException):
        zs_poly.calc_zonal_band_stats_test_poly_pts(
            res_hybrid, tif_path, 1, 0.0, 100.0, -99.0, percentile=None, percentile_field="bad_p"
        )
    with pytest.raises(rsgispy.RSGISPyException):
        zs_poly.calc_zonal_band_stats_test_poly_pts(
            res_hybrid, tif_path, 1, 0.0, 100.0, -99.0, percentile=105.0, percentile_field="bad_p"
        )


def test_calc_zonal_band_pxl_counts_and_ext_point_band_values(tmp_path):
    """Test categorical pixel counting and point extraction (with reprojection)."""
    cat_arr = np.array(
        [
            [1, 1, 2, 3],
            [1, 2, 2, 3],
            [3, 3, 1, 2],
            [1, 2, 3, 9],
        ],
        dtype=np.int16,
    )
    tif_path = str(tmp_path / "cat.tif")
    _write_test_raster(tif_path, cat_arr, nodata=9)

    poly = box(500000.0, 5199960.0, 500040.0, 5200000.0)
    poly_gpkg = str(tmp_path / "cats.gpkg")
    gpd.GeoDataFrame({"id": [1]}, geometry=[poly], crs="EPSG:32630").to_file(
        poly_gpkg, layer="cats"
    )
    zs_poly.calc_zonal_band_pxl_counts_file(
        poly_gpkg,
        "cats",
        tif_path,
        img_band=1,
        vals_lut={1: "C1", 2: "C2", 3: "C3"},
        total_field="TotalPxls",
    )
    cat_res = gpd.read_file(poly_gpkg, layer="cats")
    assert cat_res.loc[0, "totalpxls"] == pytest.approx(16.0)
    assert cat_res.loc[0, "c1"] == pytest.approx(5.0)
    assert cat_res.loc[0, "c2"] == pytest.approx(5.0)
    assert cat_res.loc[0, "c3"] == pytest.approx(5.0)

    # Point extraction in same CRS + out-of-bounds + nodata/threshold
    pts_gdf = gpd.GeoDataFrame(
        {"id": [1, 2, 3, 4]},
        geometry=[
            Point(500005.0, 5199995.0),  # row 0, col 0 -> 1
            Point(500035.0, 5199965.0),  # row 3, col 3 -> 9 (nodata)
            Point(500025.0, 5199975.0),  # row 2, col 2 -> 1
            Point(500100.0, 5199995.0),  # outside image bounds
        ],
        crs="EPSG:32630",
    )
    pts_out = zs_pt.ext_point_band_values(
        pts_gdf,
        tif_path,
        img_band=1,
        min_thres=1.0,
        max_thres=5.0,
        out_no_data_val=-1.0,
        out_field="SampleVal",
    )
    assert list(pts_out["sampleval"]) == pytest.approx([1.0, -1.0, 1.0, -1.0])

    # Point extraction from WGS84 file with reproj_vec=True
    pts_wgs_file = str(tmp_path / "pts_wgs84.geojson")
    pts_gdf.iloc[:2].to_crs("EPSG:4326").to_file(pts_wgs_file)
    zs_pt.ext_point_band_values_file(
        pts_wgs_file,
        None,
        tif_path,
        img_band=1,
        min_thres=1.0,
        max_thres=5.0,
        out_no_data_val=-1.0,
        out_field="ValReproj",
        reproj_vec=True,
    )
    pts_reproj_read = gpd.read_file(pts_wgs_file)
    assert str(pts_reproj_read.crs.to_epsg()) == "4326"
    assert list(pts_reproj_read["valreproj"]) == pytest.approx([1.0, -1.0])

    # Mismatched CRS without reproj_vec raises
    with pytest.raises(rsgispy.RSGISPyException):
        zs_pt.ext_point_band_values(
            pts_gdf.to_crs("EPSG:4326"),
            tif_path,
            img_band=1,
            min_thres=1.0,
            max_thres=5.0,
            out_no_data_val=-1.0,
            out_field="SampleVal",
            reproj_vec=False,
        )

    # Non-point geometry raises
    with pytest.raises(rsgispy.RSGISPyException):
        zs_pt.ext_point_band_values(
            gpd.GeoDataFrame({"id": [1]}, geometry=[poly], crs="EPSG:32630"),
            tif_path,
            img_band=1,
            min_thres=1.0,
            max_thres=5.0,
            out_no_data_val=-1.0,
            out_field="SampleVal",
        )


def test_zonalstats_vector_input_gdf(tmp_path):
    """Test VectorInput GeoDataFrame input and optional in-memory output in zonalstats."""
    arr = np.arange(1, 17, dtype=np.float32).reshape(4, 4)
    tif_path = str(tmp_path / "grid4.tif")
    _write_test_raster(tif_path, arr, nodata=0.0)

    poly_gdf = gpd.GeoDataFrame(
        {"id": [1]},
        geometry=[box(500000.0, 5199980.0, 500020.0, 5200000.0)],
        crs="EPSG:32630",
    )
    orig_poly_cols = list(poly_gdf.columns)

    poly_res = zs_poly.calc_zonal_band_stats_file(
        poly_gdf,
        None,
        tif_path,
        img_band=1,
        min_thres=0.0,
        max_thres=100.0,
        out_no_data_val=-1.0,
        mean_field="mean_v",
    )
    assert isinstance(poly_res, gpd.GeoDataFrame)
    assert poly_res.loc[0, "mean_v"] == pytest.approx(np.mean([1.0, 2.0, 5.0, 6.0]))
    assert list(poly_gdf.columns) == orig_poly_cols

    cent_res = zs_poly.calc_zonal_poly_pts_band_stats_file(
        poly_gdf, None, tif_path, img_band=1, out_field="cent_v"
    )
    assert isinstance(cent_res, gpd.GeoDataFrame)
    assert cent_res.loc[0, "cent_v"] == pytest.approx(6.0)
    assert list(poly_gdf.columns) == orig_poly_cols

    hybrid_res = zs_poly.calc_zonal_band_stats_test_poly_pts_file(
        poly_gdf,
        None,
        tif_path,
        img_band=1,
        min_thres=0.0,
        max_thres=100.0,
        out_no_data_val=-1.0,
        mean_field="hyb_mean",
    )
    assert isinstance(hybrid_res, gpd.GeoDataFrame)
    assert hybrid_res.loc[0, "hyb_mean"] == pytest.approx(np.mean([1.0, 2.0, 5.0, 6.0]))

    cnt_res = zs_poly.calc_zonal_band_pxl_counts_file(
        poly_gdf,
        None,
        tif_path,
        img_band=1,
        vals_lut={1: "c1", 6: "c6"},
        total_field="tot",
    )
    assert isinstance(cnt_res, gpd.GeoDataFrame)
    assert cnt_res.loc[0, "c1"] == pytest.approx(1.0)
    assert cnt_res.loc[0, "c6"] == pytest.approx(1.0)

    pts_gdf = gpd.GeoDataFrame(
        {"id": [1, 2]},
        geometry=[Point(500005.0, 5199995.0), Point(500015.0, 5199985.0)],
        crs="EPSG:32630",
    )
    orig_pts_cols = list(pts_gdf.columns)
    pts_res = zs_pt.ext_point_band_values_file(
        pts_gdf,
        None,
        tif_path,
        img_band=1,
        min_thres=0.0,
        max_thres=100.0,
        out_no_data_val=-1.0,
        out_field="pt_val",
    )
    assert isinstance(pts_res, gpd.GeoDataFrame)
    assert list(pts_res["pt_val"]) == pytest.approx([1.0, 6.0])
    assert list(pts_gdf.columns) == orig_pts_cols


def test_ext_point_band_values_multiband(tmp_path):
    """Test ext_point_band_values and ext_point_band_values_file with band lists."""
    b1 = np.array([[10.0, 20.0], [30.0, -9999.0]], dtype=np.float32)
    b2 = np.array([[100.0, 200.0], [300.0, 400.0]], dtype=np.float32)
    b3 = np.array([[1.5, 2.5], [999.0, 4.5]], dtype=np.float32)
    tif_path = str(tmp_path / "mb3.tif")
    _write_multiband_test_raster(tif_path, np.stack([b1, b2, b3]), nodata=-9999.0)

    pts_gdf = gpd.GeoDataFrame(
        {"id": [1, 2, 3, 4]},
        geometry=[
            Point(500005.0, 5199995.0),  # (0, 0): [10, 100, 1.5]
            Point(500015.0, 5199985.0),  # (1, 1): [-9999, 400, 4.5]
            Point(500005.0, 5199985.0),  # (1, 0): [30, 300, 999]
            Point(500100.0, 5199995.0),  # out of bounds
        ],
        crs="EPSG:32630",
    )

    # List of bands with explicit field names
    res_list = zs_pt.ext_point_band_values(
        pts_gdf.copy(),
        tif_path,
        img_band=[1, 3],
        min_thres=0.0,
        max_thres=500.0,
        out_no_data_val=-1.0,
        out_field=["B1", "B3"],
    )
    assert list(res_list["b1"]) == pytest.approx([10.0, -1.0, 30.0, -1.0])
    assert list(res_list["b3"]) == pytest.approx([1.5, 4.5, -1.0, -1.0])

    # List of bands with prefix string
    res_pref = zs_pt.ext_point_band_values(
        pts_gdf.copy(),
        tif_path,
        img_band=[1, 2, 3],
        min_thres=0.0,
        max_thres=500.0,
        out_no_data_val=-1.0,
        out_field="S2",
    )
    assert list(res_pref["s2_1"]) == pytest.approx([10.0, -1.0, 30.0, -1.0])
    assert list(res_pref["s2_2"]) == pytest.approx([100.0, 400.0, 300.0, -1.0])
    assert list(res_pref["s2_3"]) == pytest.approx([1.5, 4.5, -1.0, -1.0])

    # All bands (img_band=None) with default band_<n> names
    res_all = zs_pt.ext_point_band_values(
        pts_gdf.copy(),
        tif_path,
        img_band=None,
        min_thres=0.0,
        max_thres=500.0,
        out_no_data_val=-1.0,
    )
    assert {"band_1", "band_2", "band_3"}.issubset(res_all.columns)
    assert list(res_all["band_2"]) == pytest.approx([100.0, 400.0, 300.0, -1.0])

    # File-based GPKG multi-band export via ext_point_band_values_file
    pts_gpkg = str(tmp_path / "pts_mb.gpkg")
    pts_gdf.to_file(pts_gpkg, layer="pts")
    ret = zs_pt.ext_point_band_values_file(
        pts_gpkg,
        "pts",
        tif_path,
        img_band=[2, 3],
        min_thres=0.0,
        max_thres=500.0,
        out_no_data_val=-1.0,
        out_field=["green", "nir"],
    )
    assert ret is None
    read_pts = gpd.read_file(pts_gpkg, layer="pts")
    assert list(read_pts["green"]) == pytest.approx([100.0, 400.0, 300.0, -1.0])
    assert list(read_pts["nir"]) == pytest.approx([1.5, 4.5, -1.0, -1.0])

    # Validation errors
    with pytest.raises(rsgispy.RSGISPyException):
        zs_pt.ext_point_band_values(
            pts_gdf.copy(), tif_path, [1, 2], 0.0, 500.0, -1.0, out_field=["only_one"]
        )
    with pytest.raises(rsgispy.RSGISPyException):
        zs_pt.ext_point_band_values(
            pts_gdf.copy(), tif_path, [], 0.0, 500.0, -1.0, out_field=[]
        )
    with pytest.raises(rsgispy.RSGISPyException):
        zs_pt.ext_point_band_values(
            pts_gdf.copy(), tif_path, [1, 99], 0.0, 500.0, -1.0
        )
    with pytest.raises(rsgispy.RSGISPyException):
        zs_pt.ext_point_band_values(
            pts_gdf.copy(), tif_path, [1, 2], 0.0, 500.0, -1.0, out_field=["a", "A"]
        )


def test_export_poly_pxls_to_df(tmp_path):
    """Test export_poly_pxls_to_df for DataFrame return, Parquet compression, and CSV."""
    # 4x4 grid, 3 bands
    b1 = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, -9999.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 999.0],
        ],
        dtype=np.float32,
    )
    b2 = b1 * 10.0
    b2[1, 1] = -9999.0
    b3 = b1 + 100.0
    b3[1, 1] = -9999.0
    tif_path = str(tmp_path / "poly_mb.tif")
    _write_multiband_test_raster(tif_path, np.stack([b1, b2, b3]), nodata=-9999.0)

    # Poly 0: top-left 2x2 (4 pixels, 1 is nodata -> 3 valid pixels)
    # Poly 1: bottom-right 2x2 (4 pixels, row3 col3 has b1=999 > max_thres=500 -> 3 valid)
    polys_gdf = gpd.GeoDataFrame(
        {"cls": ["forest", "urban"]},
        geometry=[
            box(500000.0, 5199980.0, 500020.0, 5200000.0),
            box(500020.0, 5199960.0, 500040.0, 5199980.0),
        ],
        crs="EPSG:32630",
    )

    # 1. Return DataFrame in memory with selected bands, attr_cols, coords, and nodata masking
    df = zs_poly.export_poly_pxls_to_df(
        polys_gdf,
        tif_path,
        img_bands=[1, 3],
        band_names=["blue", "nir"],
        attr_cols=["cls"],
        include_coords=True,
        use_pbar=False,
    )
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["poly_fid", "cls", "x", "y", "blue", "nir"]
    assert len(df) == 7
    assert df[df["poly_fid"] == 0]["blue"].tolist() == pytest.approx([1.0, 2.0, 5.0])
    assert df[df["poly_fid"] == 0]["nir"].tolist() == pytest.approx([101.0, 102.0, 105.0])
    assert df[df["poly_fid"] == 0]["cls"].unique().tolist() == ["forest"]
    assert df[df["poly_fid"] == 1]["blue"].tolist() == pytest.approx(
        [11.0, 12.0, 15.0, 999.0]
    )
    assert df[df["poly_fid"] == 1]["cls"].unique().tolist() == ["urban"]
    assert df["x"].iloc[0] == pytest.approx(500005.0)
    assert df["y"].iloc[0] == pytest.approx(5199995.0)

    # 2. All bands (img_bands=None) and mask_nodata=False
    df_all = zs_poly.export_poly_pxls_to_df(
        polys_gdf, tif_path, img_bands=None, mask_nodata=False, use_pbar=False
    )
    assert list(df_all.columns) == ["poly_fid", "band_1", "band_2", "band_3"]
    assert len(df_all) == 8

    # 3. Single band as int & reprojected input vector
    df_reproj = zs_poly.export_poly_pxls_to_df(
        polys_gdf.to_crs("EPSG:4326"),
        tif_path,
        img_bands=2,
        reproj_vec=True,
        use_pbar=False,
    )
    assert list(df_reproj.columns) == ["poly_fid", "band_2"]
    assert len(df_reproj) == 7

    # 4. Empty pixel selection (polygon covering only nodata pixel)
    nodata_poly = gpd.GeoDataFrame(
        {"cls": ["nodata"]},
        geometry=[box(500010.0, 5199980.0, 500020.0, 5199990.0)],
        crs="EPSG:32630",
    )
    df_empty = zs_poly.export_poly_pxls_to_df(
        nodata_poly, tif_path, img_bands=[1, 2], use_pbar=False
    )
    assert isinstance(df_empty, pd.DataFrame)
    assert len(df_empty) == 0
    assert list(df_empty.columns) == ["poly_fid", "band_1", "band_2"]

    # 5. Export to CSV from GPKG path input
    poly_gpkg = str(tmp_path / "polys.gpkg")
    polys_gdf.to_file(poly_gpkg, layer="zones")
    csv_out = str(tmp_path / "pixels.csv")
    ret_csv = zs_poly.export_poly_pxls_to_df(
        poly_gpkg,
        tif_path,
        img_bands=[1, 3],
        band_names=["b1", "b3"],
        attr_cols=["cls"],
        out_file=csv_out,
        vec_lyr="zones",
        use_pbar=False,
    )
    assert ret_csv is None
    df_csv = pd.read_csv(csv_out)
    assert len(df_csv) == 7
    assert list(df_csv.columns) == ["poly_fid", "cls", "b1", "b3"]

    # 6. Export to Parquet with compression options (when pyarrow installed)
    if importlib.util.find_spec("pyarrow") is not None:
        import pyarrow.parquet as pq

        pq_zstd = str(tmp_path / "pixels_zstd.parquet")
        assert (
            zs_poly.export_poly_pxls_to_df(
                polys_gdf,
                tif_path,
                img_bands=[1, 2],
                out_file=pq_zstd,
                out_compress="zstd",
                use_pbar=False,
            )
            is None
        )
        meta_zstd = pq.ParquetFile(pq_zstd).metadata
        assert meta_zstd.row_group(0).column(0).compression == "ZSTD"
        assert len(pd.read_parquet(pq_zstd)) == 7

        pq_ext_gz = str(tmp_path / "pixels.parquet.gzip")
        zs_poly.export_poly_pxls_to_df(
            polys_gdf,
            tif_path,
            img_bands=[1, 2],
            out_file=pq_ext_gz,
            use_pbar=False,
        )
        meta_gz = pq.ParquetFile(pq_ext_gz).metadata
        assert meta_gz.row_group(0).column(0).compression == "GZIP"
        assert len(pd.read_parquet(pq_ext_gz)) == 7

        pq_uncomp = str(tmp_path / "pixels_uncomp.parquet")
        zs_poly.export_poly_pxls_to_df(
            polys_gdf,
            tif_path,
            img_bands=[1],
            out_file=pq_uncomp,
            out_compress=False,
            use_pbar=False,
        )
        meta_uncomp = pq.ParquetFile(pq_uncomp).metadata
        assert meta_uncomp.row_group(0).column(0).compression == "UNCOMPRESSED"

    # 7. Validation errors
    with pytest.raises(rsgispy.RSGISPyException):
        zs_poly.export_poly_pxls_to_df(
            polys_gdf, tif_path, img_bands=[1, 2], band_names=["only_one"]
        )
    with pytest.raises(rsgispy.RSGISPyException):
        zs_poly.export_poly_pxls_to_df(
            polys_gdf, tif_path, img_bands=[1], attr_cols=["missing_col"]
        )
    with pytest.raises(rsgispy.RSGISPyException):
        zs_poly.export_poly_pxls_to_df(
            polys_gdf,
            tif_path,
            out_file=str(tmp_path / "bad.parquet"),
            out_compress="invalid_codec",
        )
    with pytest.raises(rsgispy.RSGISPyException):
        zs_poly.export_poly_pxls_to_df(
            polys_gdf, tif_path, out_file=str(tmp_path / "bad.xml")
        )
    pts_only = gpd.GeoDataFrame(
        {"id": [1]}, geometry=[Point(500005.0, 5199995.0)], crs="EPSG:32630"
    )
    with pytest.raises(rsgispy.RSGISPyException):
        zs_poly.export_poly_pxls_to_df(pts_only, tif_path)


def test_zonalstats_constants_helpers_and_kwonly(tmp_path):
    """Test canonical *CENTRE constants, aliases, normalise_band_list, and kw-only args."""
    import rsgispy.zonalstats as zs
    from rsgispy.zonalstats._helpers import normalise_band_list

    assert zs.METHOD_POLYCONTAINSPIXELCENTRE == 1
    assert zs.METHOD_PIXELCONTAINSPOLYCENTRE == 5
    assert zs.METHOD_POLYCONTAINSPIXELCENTER == zs.METHOD_POLYCONTAINSPIXELCENTRE
    assert zs.METHOD_PIXELCONTAINSPOLYCENTER == zs.METHOD_PIXELCONTAINSPOLYCENTRE
    assert "METHOD_POLYCONTAINSPIXELCENTRE" in zs.__all__
    assert "METHOD_PIXELCONTAINSPOLYCENTRE" in zs.__all__

    assert normalise_band_list(None, 3, "img.tif") == [1, 2, 3]
    assert normalise_band_list(2, 3, "img.tif") == [2]
    assert normalise_band_list([1, 3], 3, "img.tif") == [1, 3]

    arr = np.ones((2, 2), dtype=np.float32)
    tif_path = str(tmp_path / "kw.tif")
    _write_test_raster(tif_path, arr, nodata=-9999.0)
    pts_gdf = gpd.GeoDataFrame(
        {"id": [1]}, geometry=[Point(500005.0, 5199995.0)], crs="EPSG:32630"
    )
    poly_gdf = gpd.GeoDataFrame(
        {"id": [1]},
        geometry=[box(500000.0, 5199980.0, 500020.0, 5200000.0)],
        crs="EPSG:32630",
    )

    with pytest.raises(TypeError):
        zs_pt.ext_point_band_values(
            pts_gdf.copy(), tif_path, 1, 0.0, 10.0, -1.0, "v", False, None, False
        )
    with pytest.raises(TypeError):
        zs_pt.ext_point_band_values_file(
            pts_gdf.copy(), None, tif_path, 1, 0.0, 10.0, -1.0, "v", False, None, False
        )
    with pytest.raises(TypeError):
        zs_poly.calc_zonal_band_stats(
            poly_gdf.copy(),
            tif_path,
            1,
            0.0,
            10.0,
            -1.0,
            None,
            None,
            "m",
            None,
            None,
            None,
            None,
            None,
            None,
            False,
        )
    with pytest.raises(TypeError):
        zs_poly.export_poly_pxls_to_df(
            poly_gdf.copy(),
            tif_path,
            [1],
            None,
            None,
            None,
            None,
            None,
            True,
            False,
            False,
            False,
            None,
            None,
        )
