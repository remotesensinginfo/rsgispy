"""
Tests for rsgispy.elevation.
"""

import math

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

import rsgispy
import rsgispy.elevation.filtering as elev_filt
import rsgispy.elevation.fusion as elev_fus
import rsgispy.elevation.illumination as elev_illum
import rsgispy.elevation.terrain as elev_terr


def _write_raster(path, arr, res=10.0, nodata=None, crs="EPSG:32630"):
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
        "transform": from_origin(500000.0, 5200000.0, res, res),
    }
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(str(path), "w", **profile) as dst:
        dst.write(data)


def test_slope_aspect_catagorise_and_pxl_res(tmp_path):
    """Test slope, aspect, catagorise_aspect, and *_pxl_res_img variants."""
    # Tilted plane: elevation increases linearly to the North (row 0 highest -> faces South = 180 deg)
    # dy = 10 m per 10 m pixel -> 45 deg slope
    rows = np.arange(5, dtype=np.float32)[:, None]
    dem_south = np.repeat((4.0 - rows) * 10.0, 5, axis=1)
    dem_path = tmp_path / "dem_south.tif"
    _write_raster(dem_path, dem_south, res=10.0, nodata=-9999.0)

    slope_deg_path = tmp_path / "slope_deg.tif"
    slope_rad_path = tmp_path / "slope_rad.tif"
    elev_terr.slope(str(dem_path), str(slope_deg_path), "GTIFF", "degrees")
    elev_terr.slope(str(dem_path), str(slope_rad_path), "GTIFF", "radians")
    with rasterio.open(slope_deg_path) as s_deg, rasterio.open(slope_rad_path) as s_rad:
        assert s_deg.dtypes[0] == "float32"
        # Interior pixel (2, 2) has exact 45-degree slope
        assert s_deg.read(1)[2, 2] == pytest.approx(45.0, abs=1e-4)
        assert s_rad.read(1)[2, 2] == pytest.approx(math.radians(45.0), abs=1e-4)

    aspect_path = tmp_path / "aspect.tif"
    elev_terr.aspect(str(dem_path), str(aspect_path), "GTIFF")
    with rasterio.open(aspect_path) as asp_ds:
        assert asp_ds.read(1)[2, 2] == pytest.approx(180.0, abs=1e-4)

    # Multi-directional aspect grid + catagorise_aspect + nodata NaN
    asp_vals = np.array(
        [[10.0, 60.0, 100.0], [150.0, 200.0, 250.0], [300.0, 350.0, np.nan]],
        dtype=np.float32,
    )
    asp_grid = tmp_path / "asp_grid.tif"
    cat_path = tmp_path / "asp_cat.tif"
    _write_raster(asp_grid, asp_vals)
    elev_terr.catagorise_aspect(str(asp_grid), str(cat_path), "GTIFF")
    with rasterio.open(cat_path) as cat_ds:
        assert cat_ds.dtypes[0] == "uint8"
        expected = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(cat_ds.read(1), expected)

    # Pixel resolution image variants (20 m resolution -> 26.565 deg slope, same 180 deg aspect)
    pxl_res_arr = np.stack(
        [np.full((5, 5), 20.0, dtype=np.float32), np.full((5, 5), 20.0, dtype=np.float32)]
    )
    pxl_res_path = tmp_path / "pxl_res.tif"
    _write_raster(pxl_res_path, pxl_res_arr)

    slope_pr_path = tmp_path / "slope_pr.tif"
    asp_pr_path = tmp_path / "asp_pr.tif"
    elev_terr.slope_pxl_res_img(str(dem_path), str(pxl_res_path), str(slope_pr_path), "GTIFF", "degrees")
    elev_terr.aspect_pxl_res_img(str(dem_path), str(pxl_res_path), str(asp_pr_path), "GTIFF")
    with rasterio.open(slope_pr_path) as spr, rasterio.open(asp_pr_path) as apr:
        assert spr.read(1)[2, 2] == pytest.approx(math.degrees(math.atan(0.5)), abs=1e-4)
        assert apr.read(1)[2, 2] == pytest.approx(180.0, abs=1e-4)

    with pytest.raises(rsgispy.RSGISPyException):
        elev_terr.slope(str(dem_path), str(slope_deg_path), "GTIFF", "percent")
    with pytest.raises(rsgispy.RSGISPyException):
        elev_terr.slope_pxl_res_img(str(dem_path), str(dem_path), str(slope_pr_path), "GTIFF", "degrees")


def test_hillshade_shadow_and_ray_angles(tmp_path):
    """Test hillshade, hillshade_pxl_res_img, shadow_mask, local_incidence_angle, local_existance_angle."""
    # North-facing vs South-facing ridge: row 0=0, row 1=50, row 2=100, row 3=50, row 4=0
    profile_col = np.array([0.0, 50.0, 100.0, 50.0, 0.0], dtype=np.float32)[:, None]
    dem_ridge = np.repeat(profile_col, 5, axis=1)
    dem_path = tmp_path / "ridge.tif"
    _write_raster(dem_path, dem_ridge, res=10.0, nodata=-9999.0)

    # Sun from South (azimuth=180, zenith=45): row 3 faces South (brighter), row 1 faces North (darker)
    hs_path = tmp_path / "hs.tif"
    elev_illum.hillshade(str(dem_path), str(hs_path), azimuth=180.0, zenith=45.0, gdalformat="GTIFF")
    with rasterio.open(hs_path) as hs_ds:
        assert hs_ds.dtypes[0] == "uint8"
        hs = hs_ds.read(1)
        assert hs[3, 2] > hs[1, 2]

    pxl_res_path = tmp_path / "res2b.tif"
    _write_raster(
        pxl_res_path,
        np.stack([np.full((5, 5), 10.0, dtype=np.float32), np.full((5, 5), 10.0, dtype=np.float32)]),
    )
    hs_pr_path = tmp_path / "hs_pr.tif"
    elev_illum.hillshade_pxl_res_img(
        str(dem_path), str(pxl_res_path), str(hs_pr_path), azimuth=180.0, zenith=45.0, gdalformat="GTIFF"
    )
    with rasterio.open(hs_path) as h1, rasterio.open(hs_pr_path) as h2:
        np.testing.assert_array_equal(h1.read(1), h2.read(1))

    # Shadow mask: row 1 faces away from southern low sun (zenith=75) -> shadowed
    shd_path = tmp_path / "shadow.tif"
    elev_illum.shadow_mask(
        str(dem_path), str(shd_path), azimuth=180.0, zenith=75.0, max_height=200.0, gdalformat="GTIFF"
    )
    with rasterio.open(shd_path) as shd_ds:
        assert shd_ds.dtypes[0] == "uint8"
        shd = shd_ds.read(1)
        assert shd[1, 2] == 1

    # Cast shadow check: tall wall at row 1 (elev 100), flat valley behind at row 0 (elev 0), sun from South (180)
    cliff = np.zeros((5, 5), dtype=np.float32)
    cliff[2, :] = 100.0
    cliff_path = tmp_path / "cliff.tif"
    cliff_shd = tmp_path / "cliff_shd.tif"
    _write_raster(cliff_path, cliff, res=10.0, nodata=-9999.0)
    elev_illum.shadow_mask(
        str(cliff_path), str(cliff_shd), azimuth=180.0, zenith=75.0, max_height=200.0, gdalformat="GTIFF"
    )
    with rasterio.open(cliff_shd) as cshd_ds:
        assert cshd_ds.read(1)[0, 2] == 1

    inc_path = tmp_path / "inc.tif"
    ext_path = tmp_path / "ext.tif"
    elev_illum.local_incidence_angle(str(dem_path), str(inc_path), azimuth=180.0, zenith=45.0, gdalformat="GTIFF")
    elev_illum.local_existance_angle(str(dem_path), str(ext_path), azimuth=180.0, zenith=45.0, gdalformat="GTIFF")
    with rasterio.open(inc_path) as inc_ds, rasterio.open(ext_path) as ext_ds:
        assert inc_ds.dtypes[0] == "float32"
        assert ext_ds.dtypes[0] == "float32"
        assert inc_ds.read(1)[2, 2] == pytest.approx(45.0, abs=1e-4)
        assert ext_ds.read(1)[2, 2] == pytest.approx(45.0, abs=1e-4)

    with pytest.raises(rsgispy.RSGISPyException):
        elev_illum.hillshade(str(dem_path), str(hs_path), azimuth=400.0, zenith=45.0, gdalformat="GTIFF")
    with pytest.raises(rsgispy.RSGISPyException):
        elev_illum.shadow_mask(str(dem_path), str(shd_path), azimuth=180.0, zenith=100.0, max_height=200.0, gdalformat="GTIFF")


def test_filtering_fill_detrend_and_fusion(tmp_path):
    """Test dtm_aspect_median_filter, fill_dem_soille_gratin_1994, plane_fit_detreat_dem, fft_dem_fusion, resampling_detread_dem."""
    # 1. dtm_aspect_median_filter: spike at centre (2, 2) with differing aspect shouldn't affect neighbours with matching aspect
    dtm = np.full((5, 5), 10.0, dtype=np.float32)
    dtm[2, 2] = 999.0
    asp = np.full((5, 5), 90.0, dtype=np.float32)
    asp[2, 2] = 270.0  # outside +/- 30 deg of 90 deg
    dtm_path = tmp_path / "dtm.tif"
    asp_path = tmp_path / "asp.tif"
    filt_path = tmp_path / "dtm_filt.tif"
    _write_raster(dtm_path, dtm, nodata=-9999.0)
    _write_raster(asp_path, asp)
    elev_filt.dtm_aspect_median_filter(
        str(dtm_path), str(asp_path), str(filt_path), aspect_range=30.0, win_h_size=1, gdalformat="GTIFF"
    )
    with rasterio.open(filt_path) as fds:
        # Neighbour (2, 1) has aspect 90 -> excludes spike at (2, 2) whose aspect is 270
        assert fds.read(1)[2, 1] == pytest.approx(10.0)

    # 2. fill_dem_soille_gratin_1994: fill central pit (val=1) surrounded by rim (val=5)
    pit_dem = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 5, 5, 5, 0],
            [0, 5, 1, 5, 0],
            [0, 5, 4, 5, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.int16,
    )
    vld_msk = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    pit_path = tmp_path / "pit.tif"
    vld_path = tmp_path / "vld.tif"
    filled_path = tmp_path / "filled.tif"
    _write_raster(pit_path, pit_dem, nodata=-9999)
    _write_raster(vld_path, vld_msk)
    elev_filt.fill_dem_soille_gratin_1994(str(pit_path), str(vld_path), str(filled_path), "GTIFF")
    with rasterio.open(filled_path) as fld_ds:
        assert fld_ds.dtypes[0] == "int16"
        # Lowest rim cell is 4, so pit at (2, 2) must be filled to 4
        assert fld_ds.read(1)[2, 2] == 4

    # 3. plane_fit_detreat_dem on exact planar DEM preserves value at centre
    yy, xx = np.mgrid[0:6, 0:6]
    plane = (2.0 * xx + 3.0 * yy + 10.0).astype(np.float32)
    plane_path = tmp_path / "plane.tif"
    plane_out = tmp_path / "plane_fit.tif"
    _write_raster(plane_path, plane, nodata=-9999.0)
    elev_filt.plane_fit_detreat_dem(str(plane_path), str(plane_out), "GTIFF", win_size=3)
    with rasterio.open(plane_out) as pds:
        np.testing.assert_allclose(pds.read(1)[1:-1, 1:-1], plane[1:-1, 1:-1], atol=1e-4)

    with pytest.raises(rsgispy.RSGISPyException):
        elev_filt.plane_fit_detreat_dem(str(plane_path), str(plane_out), "GTIFF", win_size=4)

    # 4. fft_dem_fusion
    high_dem = np.full((16, 16), 100.0, dtype=np.float32)
    low_dem = np.full((16, 16), 50.0, dtype=np.float32)
    high_path = tmp_path / "high.tif"
    low_path = tmp_path / "low.tif"
    fused_path = tmp_path / "fused.tif"
    _write_raster(high_path, high_dem)
    _write_raster(low_path, low_dem)
    elev_fus.fft_dem_fusion(str(high_path), str(low_path), str(fused_path), c_size=4, gdalformat="GTIFF")
    with rasterio.open(fused_path) as fds:
        # DC component came from high_dem -> mean matches 100.0
        assert float(fds.read(1).mean()) == pytest.approx(100.0, abs=1e-3)

    # 5. resampling_detread_dem
    resamp_in = np.full((10, 10), 25.0, dtype=np.float32)
    resamp_path = tmp_path / "resamp_in.tif"
    resamp_out = tmp_path / "resamp_out.tif"
    _write_raster(resamp_path, resamp_in, res=10.0, nodata=-9999.0)
    elev_filt.resampling_detread_dem(
        str(resamp_path),
        str(resamp_out),
        resmp_res=30.0,
        gdalformat="GTIFF",
    )
    with rasterio.open(resamp_out) as rds:
        assert rds.width == 10 and rds.height == 10
        assert float(rds.read(1)[5, 5]) == pytest.approx(25.0, abs=1e-3)
