"""
Tests for rsgispy.zonalstats using rasterio and geopandas.
"""

import geopandas as gpd
import numpy as np
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

    # Poly 1 covers top-left 3x3 (rows 0..2, cols 0..2): values [1, 2, 2, 1, 2, 10, 3, 3, 4] -> valid <= 100
    # Wait: top-left 4x4 cols 0..3 rows 0..3 includes row1 col3=999 (above max_thres=100) and row3 col0=-9999 (nodata)
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

    # Poly 2 has only 999.0 which is > max_thres=100.0 -> out_no_data_val
    assert out_gdf.loc[1, "meanval"] == pytest.approx(-1.0)
    assert out_gdf.loc[1, "countval"] == pytest.approx(-1.0)

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
    # Tiny poly strictly inside pixel (row=2, col=3 -> value 14) not containing pixel centre (500035, 5199975)
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
