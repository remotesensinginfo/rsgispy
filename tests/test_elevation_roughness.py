"""
Tests for rsgispy.elevation.roughness and rsgispy.imagecalc.calc_pt_win_smpls.
"""

import glob

import geopandas
import matplotlib
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point

import rsgispy
import rsgispy.elevation.roughness as rrough
import rsgispy.imagecalc.calc_pt_win_smpls as rcptwin

matplotlib.use("Agg")


def _write_dem(path, arr, res=10.0, nodata=-9999.0, crs="EPSG:32630"):
    data = np.asarray(arr, dtype=np.float32)
    height, width = data.shape
    with rasterio.open(
        str(path),
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs=crs,
        transform=from_origin(500000.0, 5200000.0, res, res),
        nodata=nodata,
    ) as dst:
        dst.write(data, 1)


def _write_points(path, layer_name="pts", crs="EPSG:32630", driver="GeoJSON"):
    gdf = geopandas.GeoDataFrame(
        {
            "id": [1, 2],
            "xbox": [6, 6],
            "ybox": [6, 6],
            "angle": [0.0, 30.0],
        },
        geometry=[
            Point(500255.0, 5199745.0),
            Point(500355.0, 5199645.0),
        ],
        crs=crs,
    )
    if driver.upper() == "GPKG":
        gdf.to_file(str(path), layer=layer_name, driver=driver)
    else:
        gdf.to_file(str(path), driver=driver)
    return gdf


def _corrugated_dem(size=64, res=10.0):
    yy, xx = np.mgrid[0:size, 0:size]
    x_m = xx * res
    y_m = yy * res
    # Sinusoidal microtopography on a tilted plane (0.1 * x + 0.05 * y)
    dem = (
        100.0
        + 0.1 * x_m
        + 0.05 * y_m
        + 2.0 * np.sin(2.0 * np.pi * xx / 6.0)
        + 1.5 * np.cos(2.0 * np.pi * yy / 6.0)
    )
    return dem.astype(np.float32)


def test_calc_pt_smpl_img_vals_debug_export(tmp_path):
    """Test calc_pt_smpl_img_vals with rotation and RSGISDebugExportImg."""
    dem_path = tmp_path / "dem.tif"
    _write_dem(dem_path, _corrugated_dem())

    pts_path = tmp_path / "pts.geojson"
    _write_points(pts_path)

    export_dir = tmp_path / "debug_wins"
    export_dir.mkdir()
    debug_obj = rcptwin.RSGISDebugExportImg(str(export_dir), "win")
    out_vec = tmp_path / "debug_out.geojson"

    rcptwin.calc_pt_smpl_img_vals(
        str(dem_path),
        str(pts_path),
        None,
        calc_objs=[debug_obj],
        out_vec_file=str(out_vec),
        out_format="GeoJSON",
        interp_method=rsgispy.INTERP_CUBIC,
        angle_col="angle",
        x_box_col="xbox",
        y_box_col="ybox",
    )

    out_gdf = geopandas.read_file(out_vec)
    assert "debug" in out_gdf.columns
    assert len(out_gdf) == 2
    assert len(glob.glob(str(export_dir / "*_win.tif"))) == 2

    # Negative angle must raise RSGISPyException
    bad_pts = tmp_path / "bad_pts.geojson"
    bad_gdf = geopandas.GeoDataFrame(
        {"xbox": [4], "ybox": [4], "angle": [-15.0]},
        geometry=[Point(500255.0, 5199745.0)],
        crs="EPSG:32630",
    )
    bad_gdf.to_file(str(bad_pts), driver="GeoJSON")
    with pytest.raises(rsgispy.RSGISPyException):
        rcptwin.calc_pt_smpl_img_vals(
            str(dem_path),
            str(bad_pts),
            None,
            calc_objs=[debug_obj],
            out_vec_file=str(out_vec),
            out_format="GeoJSON",
            angle_col="angle",
        )


def test_simple_and_munro_roughness_metrics(tmp_path):
    """Test calc_simple_roughness_profile_metrics and calc_munro_roughness_profile_metrics."""
    dem_path = tmp_path / "dem_corr.tif"
    _write_dem(dem_path, _corrugated_dem())

    # Tilted flat plane -> detrended roughness should be ~0
    yy, xx = np.mgrid[0:64, 0:64]
    plane_dem = (50.0 + 0.2 * xx * 10.0 + 0.1 * yy * 10.0).astype(np.float32)
    plane_path = tmp_path / "dem_plane.tif"
    _write_dem(plane_path, plane_dem)

    pts_path = tmp_path / "pts.geojson"
    _write_points(pts_path)

    simple_out = tmp_path / "simple.geojson"
    simple_plots = tmp_path / "simple_plots"
    rrough.calc_simple_roughness_profile_metrics(
        str(dem_path),
        str(pts_path),
        None,
        str(simple_out),
        out_format="GeoJSON",
        angle_col="angle",
        detrend=True,
        detrend_poly_order=1,
        export_plot_path=str(simple_plots),
    )
    simple_gdf = geopandas.read_file(simple_out)
    expected_simple_cols = rrough.CalcProfileRoughMetrics().getOutValNames()
    for col in expected_simple_cols:
        assert col in simple_gdf.columns
        assert np.all(np.isfinite(simple_gdf[col].to_numpy()))
    assert np.all(simple_gdf["avg_ra"] > 0.0)
    assert np.all(simple_gdf["avg_rz"] > simple_gdf["avg_ra"])
    assert len(glob.glob(str(simple_plots / "profile_smp_*.png"))) == 2

    plane_simple_out = tmp_path / "plane_simple.geojson"
    rrough.calc_simple_roughness_profile_metrics(
        str(plane_path),
        str(pts_path),
        None,
        str(plane_simple_out),
        out_format="GeoJSON",
        detrend=True,
    )
    plane_simple_gdf = geopandas.read_file(plane_simple_out)
    assert np.allclose(plane_simple_gdf["avg_ra"], 0.0, atol=1e-4)
    assert np.allclose(plane_simple_gdf["avg_rr"], 0.0, atol=1e-4)

    # Munro metrics on corrugated DEM + plots + flat plane edge case (0 peaks)
    munro_out = tmp_path / "munro.geojson"
    munro_plots = tmp_path / "munro_plots"
    rrough.calc_munro_roughness_profile_metrics(
        str(dem_path),
        str(pts_path),
        None,
        str(munro_out),
        out_format="GeoJSON",
        detrend=True,
        export_plot_path=str(munro_plots),
    )
    munro_gdf = geopandas.read_file(munro_out)
    for col in rrough.CalcMunroRoughnessMetric().getOutValNames():
        assert col in munro_gdf.columns
        assert np.all(np.isfinite(munro_gdf[col].to_numpy()))
    assert np.all(munro_gdf["x_munro_peaks"] >= 1)
    assert np.all(munro_gdf["avg_munro_z0"] > 0.0)
    assert len(glob.glob(str(munro_plots / "munro_rough_profile_smp_*.png"))) == 2

    plane_munro_out = tmp_path / "plane_munro.geojson"
    rrough.calc_munro_roughness_profile_metrics(
        str(plane_path),
        str(pts_path),
        None,
        str(plane_munro_out),
        out_format="GeoJSON",
        detrend=True,
    )
    plane_munro_gdf = geopandas.read_file(plane_munro_out)
    assert np.allclose(plane_munro_gdf["avg_munro_z0"], 0.0, atol=1e-6)


def test_smith_and_all_roughness_metrics(tmp_path):
    """Test calc_smith_roughness_metrics and calc_all_roughness_profile_metrics."""
    dem_path = tmp_path / "dem_corr.tif"
    _write_dem(dem_path, _corrugated_dem())

    yy, xx = np.mgrid[0:64, 0:64]
    plane_dem = (50.0 + 0.2 * xx * 10.0 + 0.1 * yy * 10.0).astype(np.float32)
    plane_path = tmp_path / "dem_plane.tif"
    _write_dem(plane_path, plane_dem)

    pts_path = tmp_path / "pts.geojson"
    _write_points(pts_path)

    # Smith metrics with detrend=True and diagnostic 3D plots
    smith_out = tmp_path / "smith.geojson"
    smith_plots = tmp_path / "smith_plots"
    rrough.calc_smith_roughness_metrics(
        str(dem_path),
        str(pts_path),
        None,
        str(smith_out),
        out_format="GeoJSON",
        detrend=True,
        export_plot_path=str(smith_plots),
    )
    smith_gdf = geopandas.read_file(smith_out)
    for col in rrough.CalcSmithRoughnessMetric().getOutValNames():
        assert col in smith_gdf.columns
        assert np.all(np.isfinite(smith_gdf[col].to_numpy()))
    assert np.all(smith_gdf["smith_hs"] > 0.0)
    assert np.all(smith_gdf["avg_smith_z0"] > 0.0)
    assert len(glob.glob(str(smith_plots / "smith_rough_profile_smp_*.png"))) == 2

    # Smith metrics on tilted plane -> detrends to 0
    plane_smith_out = tmp_path / "plane_smith.geojson"
    rrough.calc_smith_roughness_metrics(
        str(plane_path),
        str(pts_path),
        None,
        str(plane_smith_out),
        out_format="GeoJSON",
        detrend=True,
    )
    plane_smith_gdf = geopandas.read_file(plane_smith_out)
    assert np.allclose(plane_smith_gdf["smith_hs"], 0.0, atol=1e-4)
    assert np.allclose(plane_smith_gdf["avg_smith_z0"], 0.0, atol=1e-4)

    # Smith metrics with detrend=False + plots
    smith_nodetrend_plots = tmp_path / "smith_nodetrend_plots"
    rrough.calc_smith_roughness_metrics(
        str(dem_path),
        str(pts_path),
        None,
        str(tmp_path / "smith_nodetrend.geojson"),
        out_format="GeoJSON",
        detrend=False,
        export_plot_path=str(smith_nodetrend_plots),
    )
    assert len(glob.glob(str(smith_nodetrend_plots / "smith_rough_profile_smp_*.png"))) == 2

    # Combined calc_all_roughness_profile_metrics writing GPKG layer
    all_gpkg = tmp_path / "all_roughness.gpkg"
    rrough.calc_all_roughness_profile_metrics(
        str(dem_path),
        str(pts_path),
        None,
        str(all_gpkg),
        out_vec_lyr="roughness_pts",
        out_format="GPKG",
        interp_method=rsgispy.INTERP_BILINEAR,
        angle_col="angle",
        detrend=True,
        detrend_poly_order=1,
    )
    all_gdf = geopandas.read_file(all_gpkg, layer="roughness_pts")
    assert len(all_gdf) == 2
    for col in (
        rrough.CalcProfileRoughMetrics().getOutValNames()
        + rrough.CalcMunroRoughnessMetric().getOutValNames()
        + rrough.CalcSmithRoughnessMetric().getOutValNames()
    ):
        assert col in all_gdf.columns
        assert np.all(np.isfinite(all_gdf[col].to_numpy()))
