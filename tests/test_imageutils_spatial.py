"""
Tests for :mod:`rsgispy.imageutils.spatial`: pixel geometry rasters and
gridding scattered points.
"""


import geopandas as gpd
import numpy as np
import pytest
import rasterio
import rasterio.crs
from rasterio.transform import from_origin
from shapely.geometry import Point

import rsgispy.imageutils.spatial as iu_spat

from conftest import write_raster


def test_spatial_pixel_geometry(tmp_path):
    """Test WGS84 pixel area/size and local/WGS84 pixel location rasters."""
    wgs_img = tmp_path / "wgs.tif"
    write_raster(
        wgs_img,
        np.zeros((4, 4), dtype=np.float32),
        origin=(0.0, 52.0),
        pixel_size=0.01,
        crs="EPSG:4326",
    )

    area_img = tmp_path / "area.tif"
    size_img = tmp_path / "size.tif"
    loc_img = tmp_path / "loc.tif"
    wgs_loc_img = tmp_path / "wgs_loc.tif"

    iu_spat.calc_wgs84_pixel_area(
        str(wgs_img), str(area_img), scale=10000, gdalformat="GTIFF"
    )
    iu_spat.calc_wgs84_pixel_size(
        str(wgs_img), str(size_img), gdalformat="GTIFF"
    )
    iu_spat.calc_pixel_locations(str(wgs_img), str(loc_img), gdalformat="GTIFF")

    with rasterio.open(size_img) as s_src, rasterio.open(area_img) as a_src:
        xm, ym = s_src.read(1), s_src.read(2)
        assert np.all(xm > 600.0) and np.all(ym > 1000.0)
        assert np.allclose(a_src.read(1), (xm * ym) / 10000.0)

    with rasterio.open(loc_img) as l_src:
        assert l_src.read(1)[0, 0] == pytest.approx(0.005)
        assert l_src.read(2)[0, 0] == pytest.approx(51.995)

    utm_img = tmp_path / "utm.tif"
    write_raster(utm_img, np.zeros((4, 4), dtype=np.float32))
    iu_spat.calc_wgs84_pixel_locations(
        str(utm_img), str(wgs_loc_img), gdalformat="GTIFF"
    )
    with rasterio.open(wgs_loc_img) as wl_src:
        lons, lats = wl_src.read(1), wl_src.read(2)
        assert np.all((lons > -4.0) & (lons < -2.0))
        assert np.all((lats > 46.0) & (lats < 48.0))


def test_grid_scattered_pts(tmp_path):
    """Test grid_scattered_pts with invdistnn, nearest, and linear."""
    utm_img = tmp_path / "utm_grid.tif"
    write_raster(utm_img, np.zeros((4, 4), dtype=np.float32))

    vec_path = tmp_path / "pts.gpkg"
    gdf = gpd.GeoDataFrame(
        {"val": [10.0, 20.0, 30.0, 40.0]},
        geometry=[
            Point(500005.0, 5199995.0),
            Point(500035.0, 5199995.0),
            Point(500005.0, 5199965.0),
            Point(500035.0, 5199965.0),
        ],
        crs="EPSG:32630",
    )
    gdf.to_file(str(vec_path), layer="pts", driver="GPKG")

    grid_out = tmp_path / "grid_idw.tif"
    iu_spat.grid_scattered_pts(
        str(vec_path),
        "pts",
        "val",
        str(utm_img),
        str(grid_out),
        gdal_grid_alg="invdistnn:power=2.0:radius=100.0:max_points=4",
        gdalformat="GTIFF",
    )
    with rasterio.open(grid_out) as g_src:
        g_arr = g_src.read(1)
        assert g_arr[0, 0] == pytest.approx(10.0)
        assert g_arr[3, 3] == pytest.approx(40.0)

    lin_out = tmp_path / "grid_lin.tif"
    iu_spat.grid_scattered_pts(
        str(vec_path),
        "pts",
        "val",
        str(utm_img),
        str(lin_out),
        gdal_grid_alg="linear",
        gdalformat="GTIFF",
    )
    with rasterio.open(lin_out) as l_src:
        assert l_src.read(1)[0, 3] == pytest.approx(20.0)

    near_out = tmp_path / "grid_near.tif"
    iu_spat.grid_scattered_pts(
        str(vec_path),
        "pts",
        "val",
        str(utm_img),
        str(near_out),
        gdal_grid_alg="nearest:radius1=20.0:radius2=20.0",
        gdalformat="GTIFF",
    )
    with rasterio.open(near_out) as n_src:
        assert n_src.read(1)[0, 0] == pytest.approx(10.0)


def test_grid_scattered_pts_vectorised_algorithms(tmp_path):
    ref_img = write_raster(
        tmp_path / "grid_ref.tif",
        np.zeros((10, 10), dtype=np.float32),
        transform=from_origin(0.0, 100.0, 10.0, 10.0),
    )
    pts = [
        Point(15.0, 85.0),
        Point(85.0, 85.0),
        Point(15.0, 15.0),
        Point(85.0, 15.0),
        Point(50.0, 50.0),
    ]
    gdf = gpd.GeoDataFrame(
        {"z": [10.0, 20.0, 30.0, 40.0, 25.0]}, geometry=pts, crs="EPSG:32630"
    )
    for alg in (
        "nearest",
        "invdist:power=2.0",
        "invdistnn:power=2.0:radius=60.0:max_points=4",
        "average:radius1=60.0:radius2=60.0",
        "count:radius1=60.0:radius2=60.0",
    ):
        out_img = str(tmp_path / f"grid_{alg.split(':')[0]}.tif")
        iu_spat.grid_scattered_pts(
            vec_file=gdf,
            vec_lyr=None,
            vec_col="z",
            input_img=ref_img,
            output_img=out_img,
            gdal_grid_alg=alg,
            no_data_val=-999.0,
            use_pbar=False,
        )
        with rasterio.open(out_img) as dst:
            arr = dst.read(1)
            assert arr.shape == (10, 10)
            if alg == "nearest":
                assert arr[1, 1] == pytest.approx(10.0)
                assert arr[8, 8] == pytest.approx(40.0)
            elif alg.startswith("count"):
                assert arr[5, 5] >= 1.0
