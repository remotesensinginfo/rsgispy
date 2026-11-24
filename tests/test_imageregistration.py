"""
Unit tests for rsgispy.imageregistration.
"""

import importlib.util

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point

import rsgispy
import rsgispy.imageregistration as imreg
import rsgispy.imageregistration.shift as imreg_shift
import rsgispy.imageregistration.tiepoints as imreg_tp
import rsgispy.imageregistration.warping as imreg_warp
from rsgispy.imageregistration._helper import read_tie_points_file


def _make_textured_array(height: int = 128, width: int = 128, seed: int = 123) -> np.ndarray:
    """Create a smooth multi-scale textured array suitable for image matching."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    pattern = (
        40.0 * np.sin(xx / 6.5) * np.cos(yy / 7.5)
        + 25.0 * np.sin((xx + yy) / 4.2)
        + 15.0 * np.cos((xx - 1.5 * yy) / 3.1)
    )
    noise = rng.normal(0.0, 2.5, (height, width))
    return (120.0 + pattern + noise).astype(np.float32)


def _write_raster(
    path: str,
    arr: np.ndarray,
    origin_x: float = 500000.0,
    origin_y: float = 5200000.0,
    res: float = 10.0,
    crs: str = "EPSG:32630",
    nodata: float = -9999.0,
) -> None:
    """Write a 2D or 3D NumPy array as a GeoTIFF."""
    data = arr[None, :, :] if arr.ndim == 2 else arr
    count, height, width = data.shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=str(data.dtype),
        crs=crs,
        transform=from_origin(origin_x, origin_y, res, res),
        nodata=nodata,
    ) as dst:
        dst.write(data)


def test_constants_and_module_exports():
    """Verify METRIC_* in rsgispy, constants in imreg, and submodule isolation."""
    for name in (
        "METRIC_EUCLIDEAN",
        "METRIC_SQDIFF",
        "METRIC_MANHATTAN",
        "METRIC_CORRELATION",
        "METRIC_MUTUAL_INFO",
        "METRIC_NORM_MUTUAL_INFO",
        "METRIC_PHASE_CORR",
    ):
        assert hasattr(rsgispy, name)

    for removed in ("METRIC_CORELATION", "METRIC_MANHATTEN"):
        assert not hasattr(rsgispy, removed)
        assert not hasattr(imreg, removed)

    for name in (
        "TYPE_ENVI_IMG2IMG",
        "TYPE_ENVI_IMG2MAP",
        "TYPE_RSGIS_IMG2MAP",
        "TYPE_RSGIS_MAPOFFS",
        "TIE_METHOD_PHASE_CORR",
        "TIE_METHOD_TEMPLATE_MATCH",
        "OUTLIER_METHOD_RANSAC",
        "OUTLIER_METHOD_RESIDUAL",
        "OUTLIER_METHOD_MAD",
        "TRANSFORM_SHIFT",
        "TRANSFORM_SIMILARITY",
        "TRANSFORM_AFFINE",
        "TRANSFORM_POLY1",
        "TRANSFORM_POLY2",
        "TRANSFORM_POLY3",
        "TRANSFORM_TPS",
    ):
        assert hasattr(imreg, name)

    assert "__all__" not in vars(imreg)
    for fn_name in (
        "estimate_image_shift",
        "warp_with_gcps_with_gdal",
        "find_image_offset",
        "apply_offset_to_image",
        "generate_tie_points",
        "basic_registration",
        "filter_tie_points_outliers",
        "fit_geometric_model",
        "evaluate_geometric_model",
        "gcp_to_gdal",
        "add_vec_pts_as_gcps_to_img",
        "warp_with_gcps",
        "warp_with_tie_points",
        "coregister_images",
    ):
        assert fn_name not in vars(imreg)


def test_find_image_offset_and_apply_offset(tmp_path):
    """Test whole-image offset estimation across metrics and apply_offset_to_image."""
    arr = _make_textured_array(128, 128, seed=7)
    ref_img = str(tmp_path / "ref.tif")
    flt_img = str(tmp_path / "flt.tif")
    _write_raster(ref_img, arr, origin_x=500000.0, origin_y=5200000.0, res=10.0)
    # Shift floating image origin +20m east, -30m south -> needs (-20m, +30m) = (-2px, +3px)
    _write_raster(flt_img, arr, origin_x=500020.0, origin_y=5199970.0, res=10.0)

    for metric in (
        rsgispy.METRIC_CORRELATION,
        rsgispy.METRIC_EUCLIDEAN,
        rsgispy.METRIC_MANHATTAN,
        rsgispy.METRIC_SQDIFF,
        rsgispy.METRIC_MUTUAL_INFO,
        rsgispy.METRIC_NORM_MUTUAL_INFO,
        rsgispy.METRIC_PHASE_CORR,
    ):
        res = imreg_shift.find_image_offset(
            ref_img,
            flt_img,
            ref_img_bands=[1],
            flt_img_bands=[1],
            metric_type=metric,
            x_search=5,
            y_search=5,
            sub_pxl_res=4,
            return_details=True,
        )
        assert res["x_offset_pxl"] == pytest.approx(-2.0, abs=0.25)
        assert res["y_offset_pxl"] == pytest.approx(3.0, abs=0.25)
        assert res["x_offset_map"] == pytest.approx(-20.0, abs=2.5)
        assert res["y_offset_map"] == pytest.approx(30.0, abs=2.5)

    # Default tuple return in pixel units
    x_px, y_px = imreg_shift.find_image_offset(
        ref_img,
        flt_img,
        ref_img_bands=[1],
        flt_img_bands=[1],
        metric_type=rsgispy.METRIC_CORRELATION,
        x_search=5,
        y_search=5,
        sub_pxl_res=4,
    )
    assert x_px == pytest.approx(-2.0, abs=0.25)
    assert y_px == pytest.approx(3.0, abs=0.25)

    # Non-linear intensity relationship: Mutual Information still finds true shift
    flt_nl = str(tmp_path / "flt_nonlinear.tif")
    nl_arr = (0.02 * (arr.astype(np.float64) - 120.0) ** 2 + 50.0).astype(np.float32)
    _write_raster(flt_nl, nl_arr, origin_x=500020.0, origin_y=5199970.0, res=10.0)
    x_mi, y_mi = imreg_shift.find_image_offset(
        ref_img,
        flt_nl,
        ref_img_bands=[1],
        flt_img_bands=[1],
        metric_type="mutual_info",
        x_search=5,
        y_search=5,
        sub_pxl_res=4,
    )
    assert x_mi == pytest.approx(-2.0, abs=0.25)
    assert y_mi == pytest.approx(3.0, abs=0.25)

    # Map-unit offset via find_image_offset & apply_offset_to_image
    dx_m, dy_m = imreg_shift.find_image_offset(
        ref_img,
        flt_img,
        metric_type="correlation",
        x_search=5,
        y_search=5,
        out_map_units=True,
    )
    fixed_img = str(tmp_path / "fixed.tif")
    imreg_shift.apply_offset_to_image(
        flt_img,
        fixed_img,
        gdalformat="GTIFF",
        datatype=rsgispy.TYPE_32FLOAT,
        x_offset=dx_m,
        y_offset=dy_m,
        calc_stats=False,
    )
    with rasterio.open(ref_img) as r_src, rasterio.open(fixed_img) as f_src:
        assert f_src.transform.c == pytest.approx(r_src.transform.c, abs=1.0)
        assert f_src.transform.f == pytest.approx(r_src.transform.f, abs=1.0)
        assert np.allclose(f_src.read(1), r_src.read(1))

    # Pixel-unit offset in apply_offset_to_image
    fixed_px = str(tmp_path / "fixed_px.tif")
    imreg_shift.apply_offset_to_image(
        flt_img,
        fixed_px,
        x_offset=-2.0,
        y_offset=3.0,
        in_pxl_units=True,
        calc_stats=False,
    )
    with rasterio.open(ref_img) as r_src, rasterio.open(fixed_px) as f_src:
        assert f_src.transform == r_src.transform


def test_subpixel_phase_correlation_shift(tmp_path):
    """Test sub-pixel shift estimation using FFT phase cross-correlation."""
    yy, xx = np.mgrid[0:128, 0:128].astype(np.float64)

    # Analytic multi-frequency signal evaluated on sub-pixel shifted grid
    def _eval(gx, gy):
        return (
            100.0
            + 30.0 * np.sin(gx / 5.5) * np.cos(gy / 6.5)
            + 20.0 * np.cos((gx + 0.7 * gy) / 4.3)
        ).astype(np.float32)

    ref_arr = _eval(xx, yy)
    # Evaluate floating image shifted by (+1.5 px in col, -2.25 px in row)
    flt_arr = _eval(xx - 1.5, yy + 2.25)

    ref_img = str(tmp_path / "ref_sub.tif")
    flt_img = str(tmp_path / "flt_sub.tif")
    _write_raster(ref_img, ref_arr, origin_x=500000.0, origin_y=5200000.0, res=10.0)
    _write_raster(flt_img, flt_arr, origin_x=500000.0, origin_y=5200000.0, res=10.0)

    res = imreg_shift.find_image_offset(
        ref_img,
        flt_img,
        metric_type=rsgispy.METRIC_PHASE_CORR,
        sub_pxl_res=8,
        return_details=True,
    )
    # Feature in flt moved +1.5 cols, -2.25 rows -> header shift is -1.5 px X, -2.25 px Y
    assert res["x_offset_pxl"] == pytest.approx(-1.5, abs=0.25)
    assert res["y_offset_pxl"] == pytest.approx(-2.25, abs=0.25)


def test_generate_tie_points_and_formats(tmp_path):
    """Test phase-correlation & template-matching tie points and I/O formats."""
    arr = _make_textured_array(160, 160, seed=11)
    ref_img = str(tmp_path / "ref.tif")
    flt_img = str(tmp_path / "flt.tif")
    _write_raster(ref_img, arr, origin_x=500000.0, origin_y=5200000.0, res=10.0)
    _write_raster(flt_img, arr, origin_x=500030.0, origin_y=5199980.0, res=10.0)

    # Phase correlation in memory
    tp_pcc = imreg_tp.generate_tie_points(
        ref_img,
        flt_img,
        method=imreg.TIE_METHOD_PHASE_CORR,
        pixel_gap=32,
        win_size=48,
        search_area=8,
        sub_pxl_res=4,
        use_pbar=False,
    )
    assert isinstance(tp_pcc, pd.DataFrame)
    assert len(tp_pcc) >= 4
    assert tp_pcc["dx_map"].median() == pytest.approx(-30.0, abs=2.5)
    assert tp_pcc["dy_map"].median() == pytest.approx(20.0, abs=2.5)

    # Template matching via basic_registration -> RSGIS_IMG2MAP file
    rsgis_txt = str(tmp_path / "tp_rsgis.txt")
    ret = imreg_tp.basic_registration(
        ref_img,
        flt_img,
        rsgis_txt,
        pixel_gap=32,
        threshold=0.5,
        win_size=32,
        search_area=6,
        sd_ref_thres=1.0,
        sd_flt_thres=1.0,
        sub_pxl_res=4,
        metric_type=rsgispy.METRIC_CORRELATION,
        output_type=imreg.TYPE_RSGIS_IMG2MAP,
        use_pbar=False,
    )
    assert ret is None
    df_read = read_tie_points_file(rsgis_txt)
    assert len(df_read) >= 4
    assert {"ref_x", "ref_y", "flt_pxl_x", "flt_pxl_y"}.issubset(df_read.columns)

    # Warp using RSGIS_IMG2MAP text file (tests derivation of flt_x/flt_y)
    shift_from_txt = str(tmp_path / "shift_from_txt.tif")
    shift_info = imreg_warp.warp_with_tie_points(
        ref_img,
        flt_img,
        shift_from_txt,
        rsgis_txt,
        transform_type=imreg.TRANSFORM_SHIFT,
        match_ref_grid=True,
        calc_stats=True,
    )
    assert shift_info["transform_type"] == imreg.TRANSFORM_SHIFT
    assert shift_info["x_offset_map"] == pytest.approx(-30.0, abs=2.5)
    assert shift_info["y_offset_map"] == pytest.approx(20.0, abs=2.5)

    # Test CSV and Parquet tie-point outputs
    csv_out = str(tmp_path / "tp.csv")
    imreg_tp.generate_tie_points(
        ref_img,
        flt_img,
        out_gcp_file=csv_out,
        method=imreg.TIE_METHOD_PHASE_CORR,
        pixel_gap=40,
        win_size=48,
        search_area=8,
        use_pbar=False,
    )
    assert len(read_tie_points_file(csv_out)) >= 4

    if importlib.util.find_spec("pyarrow") is not None:
        pq_out = str(tmp_path / "tp.parquet")
        imreg_tp.generate_tie_points(
            ref_img,
            flt_img,
            out_gcp_file=pq_out,
            method=imreg.TIE_METHOD_PHASE_CORR,
            pixel_gap=40,
            win_size=48,
            search_area=8,
            use_pbar=False,
        )
        assert len(read_tie_points_file(pq_out)) >= 4


def test_filter_tie_points_outliers():
    """Test RANSAC and residual/MAD outlier filtering on affine & poly2 models."""
    rng = np.random.default_rng(99)
    fx = rng.uniform(500000.0, 501000.0, 30)
    fy = rng.uniform(5199000.0, 5200000.0, 30)
    # True affine mapping with slight rotation + scale + translation
    u = fx - 500500.0
    v = fy - 5199500.0
    rx = 500500.0 + 1.002 * u - 0.015 * v - 15.0
    ry = 5199500.0 + 0.015 * u + 0.998 * v + 22.0

    # Inject 5 gross outliers
    rx_noisy = rx.copy()
    ry_noisy = ry.copy()
    outlier_idx = [2, 7, 13, 19, 25]
    rx_noisy[outlier_idx] += np.array([150.0, -220.0, 80.0, -140.0, 300.0])
    ry_noisy[outlier_idx] += np.array([-180.0, 90.0, 210.0, -95.0, -250.0])

    tie_df = pd.DataFrame(
        {
            "ref_x": rx_noisy,
            "ref_y": ry_noisy,
            "flt_x": fx,
            "flt_y": fy,
            "flt_pxl_x": (fx - 500000.0) / 10.0,
            "flt_pxl_y": (5200000.0 - fy) / 10.0,
            "ref_pxl_x": (rx_noisy - 500000.0) / 10.0,
            "ref_pxl_y": (5200000.0 - ry_noisy) / 10.0,
        }
    )

    inliers_ransac, diag_ransac = imreg_tp.filter_tie_points_outliers(
        tie_df,
        method=imreg.OUTLIER_METHOD_RANSAC,
        model=imreg.TRANSFORM_AFFINE,
        residual_thres=1.0,
        rnd_seed=42,
        return_model=True,
    )
    assert diag_ransac["n_inliers"] == 25
    assert diag_ransac["n_outliers"] == 5
    assert diag_ransac["rmse"] < 1e-6
    assert "residual" in inliers_ransac.columns

    inliers_res, diag_res = imreg_tp.filter_tie_points_outliers(
        tie_df,
        method=imreg.OUTLIER_METHOD_RESIDUAL,
        model=imreg.TRANSFORM_AFFINE,
        residual_thres=1.0,
        return_model=True,
    )
    assert diag_res["n_inliers"] == 25
    assert len(inliers_res) == 25

    inliers_mad, diag_mad = imreg_tp.filter_tie_points_outliers(
        tie_df,
        method=imreg.OUTLIER_METHOD_MAD,
        model=imreg.TRANSFORM_POLY2,
        residual_thres=1.0,
        return_model=True,
    )
    assert diag_mad["n_inliers"] == 25
    assert len(inliers_mad) == 25


def test_gcp_embedding_and_coregister_images(tmp_path):
    """Test gcp_to_gdal, vector GCPs, warp_with_gcps, and coregister_images."""
    arr = _make_textured_array(140, 140, seed=55)
    ref_img = str(tmp_path / "ref.tif")
    flt_img = str(tmp_path / "flt.tif")
    _write_raster(ref_img, arr, origin_x=500000.0, origin_y=5200000.0, res=10.0)
    # Floating image shifted by +20m X, -30m Y
    _write_raster(flt_img, arr, origin_x=500020.0, origin_y=5199970.0, res=10.0)

    # 1. End-to-end co-registration with phase correlation + RANSAC + affine warp
    coreg_out = str(tmp_path / "coreg_affine.tif")
    gcp_out = str(tmp_path / "coreg_gcps.csv")
    summary = imreg_warp.coregister_images(
        in_ref_img=ref_img,
        in_float_img=flt_img,
        output_img=coreg_out,
        tie_method=imreg.TIE_METHOD_PHASE_CORR,
        outlier_method=imreg.OUTLIER_METHOD_RANSAC,
        transform_type=imreg.TRANSFORM_AFFINE,
        pixel_gap=30,
        win_size=40,
        search_area=8,
        residual_thres=2.0,
        out_gcp_file=gcp_out,
        rnd_seed=0,
        use_pbar=False,
    )
    assert summary["n_inliers"] >= 4
    assert summary["rmse"] < 2.0

    with rasterio.open(ref_img) as r_src, rasterio.open(coreg_out) as c_src:
        assert c_src.width == r_src.width
        assert c_src.height == r_src.height
        assert c_src.transform == r_src.transform
        r_arr = r_src.read(1)[10:-10, 10:-10]
        c_arr = c_src.read(1)[10:-10, 10:-10]
        mask = np.isfinite(r_arr) & np.isfinite(c_arr) & (c_arr != -9999.0)
        corr = float(np.corrcoef(r_arr[mask], c_arr[mask])[0, 1])
        assert corr > 0.99

    # 2. Vector GCP embedding + warp_with_gcps
    pts_gdf = gpd.GeoDataFrame(
        {
            "x_match": [500000.0, 501400.0, 500000.0, 501400.0],
            "y_match": [5200000.0, 5200000.0, 5198600.0, 5198600.0],
        },
        geometry=[
            Point(500020.0, 5199970.0),
            Point(501420.0, 5199970.0),
            Point(500020.0, 5198570.0),
            Point(501420.0, 5198570.0),
        ],
        crs="EPSG:32630",
    )
    vec_gcps_img = str(tmp_path / "flt_vec_gcps.tif")
    imreg_warp.add_vec_pts_as_gcps_to_img(
        flt_img, vec_gcps_img, pts_gdf, gcp_x_col="x_match", gcp_y_col="y_match"
    )
    warped_vec = str(tmp_path / "warped_vec.tif")
    imreg_warp.warp_with_gcps(
        ref_img,
        vec_gcps_img,
        warped_vec,
        interp_method=rsgispy.INTERP_NEAREST_NEIGHBOUR,
        poly_order=1,
        calc_stats=True,
    )
    with rasterio.open(ref_img) as r_src, rasterio.open(warped_vec) as w_src:
        r_sub = r_src.read(1)[5:-5, 5:-5]
        w_sub = w_src.read(1)[5:-5, 5:-5]
        assert np.allclose(r_sub, w_sub, atol=1e-4)

    # 3. Shift-mode warp_with_tie_points (header-only & calc_stats=True)
    shift_hdr = str(tmp_path / "shift_hdr.tif")
    shift_info = imreg_warp.warp_with_tie_points(
        ref_img,
        flt_img,
        shift_hdr,
        gcp_out,
        transform_type=imreg.TRANSFORM_SHIFT,
        match_ref_grid=False,
        calc_stats=True,
    )
    assert shift_info["transform_type"] == imreg.TRANSFORM_SHIFT
    with rasterio.open(ref_img) as r_src, rasterio.open(shift_hdr) as s_src:
        assert s_src.transform.c == pytest.approx(r_src.transform.c, abs=1.0)
        assert s_src.transform.f == pytest.approx(r_src.transform.f, abs=1.0)

    # 4. Similarity-mode warp_with_tie_points
    sim_out = str(tmp_path / "sim_warped.tif")
    sim_info = imreg_warp.warp_with_tie_points(
        ref_img,
        flt_img,
        sim_out,
        gcp_out,
        transform_type=imreg.TRANSFORM_SIMILARITY,
        calc_stats=False,
    )
    assert sim_info["transform_type"] == imreg.TRANSFORM_SIMILARITY

    # 5. TPS warp_with_tie_points
    tps_out = str(tmp_path / "tps_warped.tif")
    tps_info = imreg_warp.warp_with_tie_points(
        ref_img,
        flt_img,
        tps_out,
        gcp_out,
        transform_type=imreg.TRANSFORM_TPS,
        calc_stats=False,
    )
    assert tps_info["transform_type"] == imreg.TRANSFORM_TPS
    with rasterio.open(ref_img) as r_src, rasterio.open(tps_out) as t_src:
        assert t_src.width == r_src.width
        assert t_src.height == r_src.height


def test_imageregistration_errors(tmp_path):
    """Test error handling for non-overlapping rasters and invalid arguments."""
    arr = _make_textured_array(32, 32, seed=1)
    r1 = str(tmp_path / "r1.tif")
    r2 = str(tmp_path / "r2.tif")
    _write_raster(r1, arr, origin_x=500000.0, origin_y=5200000.0, res=10.0)
    _write_raster(r2, arr, origin_x=600000.0, origin_y=5200000.0, res=10.0)

    with pytest.raises(rsgispy.RSGISPyException):
        imreg_shift.find_image_offset(r1, r2)
    with pytest.raises(rsgispy.RSGISPyException):
        imreg_shift.find_image_offset(r1, r1, metric_type="unknown_metric")
    with pytest.raises(rsgispy.RSGISPyException):
        imreg_warp.warp_with_gcps(r1, r1, str(tmp_path / "out.tif"))
