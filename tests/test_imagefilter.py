"""
Tests for :mod:`rsgispy.imagefilter`.
"""

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

import rsgispy
import rsgispy.imagefilter as imf
import rsgispy.imageprocess as img_proc


def _write_raster(path, data, nodata=None, descriptions=None):
    """Write a 2D or 3D GeoTIFF raster for filter testing."""
    arr = np.asarray(data)
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    count, height, width = arr.shape
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": count,
        "dtype": str(arr.dtype),
        "crs": "EPSG:32630",
        "transform": from_origin(500000.0, 5200000.0, 10.0, 10.0),
    }
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr)
        if descriptions is not None:
            for idx, desc in enumerate(descriptions, start=1):
                dst.set_band_description(idx, desc)


def test_window_stat_and_difference_filters(tmp_path):
    """Test statistical, difference, variance, Kuwahara, and Lee filters."""
    in_path = tmp_path / "in.tif"
    arr = np.arange(1, 26, dtype=np.float32).reshape(5, 5)
    # Place a nodata pixel at corner (0, 0) while keeping centre tile clean
    arr[0, 0] = -9999.0
    arr2 = arr * 2.0
    _write_raster(
        in_path,
        np.stack([arr, arr2], axis=0),
        nodata=-9999.0,
        descriptions=["B1", "B2"],
    )

    # Interior 3x3 around (2, 2) has values [7,8,9, 12,13,14, 17,18,19]
    win_vals = np.array([7, 8, 9, 12, 13, 14, 17, 18, 19], dtype=np.float64)

    img_proc.set_default_block_size(3)
    try:
        checks = [
            (imf.apply_mean_filter, np.mean(win_vals)),
            (imf.apply_median_filter, np.median(win_vals)),
            (imf.apply_min_filter, np.min(win_vals)),
            (imf.apply_max_filter, np.max(win_vals)),
            (imf.apply_range_filter, np.max(win_vals) - np.min(win_vals)),
            (imf.apply_total_filter, np.sum(win_vals)),
            (imf.apply_stddev_filter, np.std(win_vals)),
            (imf.apply_texture_var_filter, np.var(win_vals)),
            (imf.apply_mean_diff_filter, np.mean(win_vals - 13.0)),
            (imf.apply_mean_diff_abs_filter, np.mean(np.abs(win_vals - 13.0))),
            (imf.apply_total_diff_filter, np.sum(win_vals - 13.0)),
            (imf.apply_total_diff_abs_filter, np.sum(np.abs(win_vals - 13.0))),
            (imf.apply_coeff_of_var_filter, np.std(win_vals) / np.mean(win_vals)),
            (imf.apply_norm_var_filter, np.var(win_vals) / (np.mean(win_vals) ** 2)),
            (
                imf.apply_norm_var_sqrt_filter,
                np.std(win_vals) / abs(np.mean(win_vals)),
            ),
            (
                imf.apply_norm_var_ln_filter,
                np.log1p(np.var(win_vals) / (np.mean(win_vals) ** 2)),
            ),
        ]
        for idx, (fn, expected) in enumerate(checks):
            out_path = tmp_path / f"stat_{idx}.tif"
            fn(str(in_path), str(out_path), filter_size=3, use_pbar=False)
            with rasterio.open(out_path) as src:
                out = src.read()
                assert list(src.descriptions) == ["B1", "B2"]
                assert src.nodata == -9999.0
            assert out[0, 0, 0] == -9999.0
            assert out[0, 2, 2] == pytest.approx(expected, rel=1e-5, abs=1e-5)
    finally:
        img_proc.reset_default_block_size()

    # Mode filter on categorical data
    cat_in = tmp_path / "cat.tif"
    cat_out = tmp_path / "cat_mode.tif"
    cat = np.array(
        [
            [2, 2, 3, 3, 3],
            [2, 2, 2, 3, 3],
            [1, 2, 2, 1, 3],
            [1, 1, 1, 4, 4],
            [1, 1, 4, 4, 4],
        ],
        dtype=np.uint8,
    )
    _write_raster(cat_in, cat)
    imf.apply_mode_filter(
        str(cat_in),
        str(cat_out),
        filter_size=3,
        datatype=rsgispy.TYPE_8UINT,
        use_pbar=False,
    )
    with rasterio.open(cat_out) as src:
        assert src.read(1)[1, 1] == 2
        assert src.read(1)[3, 3] == 4

    # Kuwahara filter preserves step edge without blurring across it
    step_in = tmp_path / "step.tif"
    step_out = tmp_path / "step_kuw.tif"
    step = np.zeros((9, 9), dtype=np.float32)
    step[:, 5:] = 100.0
    _write_raster(step_in, step)
    imf.apply_kuwahara_filter(str(step_in), str(step_out), filter_size=5, use_pbar=False)
    with rasterio.open(step_out) as src:
        kuw = src.read(1)
    assert kuw[4, 4] == pytest.approx(0.0, abs=1e-6)
    assert kuw[4, 5] == pytest.approx(100.0, abs=1e-6)

    # Lee filter smooths noise inside uniform regions and retains strong targets
    sar_in = tmp_path / "sar.tif"
    sar_out = tmp_path / "sar_lee.tif"
    rng = np.random.default_rng(123)
    sar = rng.normal(50.0, 2.0, size=(12, 12)).astype(np.float32)
    sar[6, 6] = 500.0
    _write_raster(sar_in, sar)
    imf.apply_lee_filter(
        str(sar_in), str(sar_out), filter_size=5, n_looks=4.0, use_pbar=False
    )
    with rasterio.open(sar_out) as src:
        lee = src.read(1)
    assert np.std(lee[:4, :4]) < np.std(sar[:4, :4])
    assert lee[6, 6] > 200.0


def test_edge_gaussian_laplacian_and_custom_kernel_filters(tmp_path):
    """Test Sobel, Prewitt, Gaussian, Laplacian, and custom kernel filters."""
    step_in = tmp_path / "edge_step.tif"
    step = np.zeros((9, 9), dtype=np.float32)
    step[:, 4:] = 10.0
    _write_raster(step_in, step)

    sobel_x = tmp_path / "sobel_x.tif"
    sobel_y = tmp_path / "sobel_y.tif"
    sobel_m = tmp_path / "sobel_m.tif"
    prewitt_x = tmp_path / "prewitt_x.tif"
    prewitt_y = tmp_path / "prewitt_y.tif"
    prewitt_m = tmp_path / "prewitt_m.tif"

    img_proc.set_default_block_size(5)
    try:
        imf.apply_sobel_x_filter(str(step_in), str(sobel_x), use_pbar=False)
        imf.apply_sobel_y_filter(str(step_in), str(sobel_y), use_pbar=False)
        imf.apply_sobel_filter(str(step_in), str(sobel_m), use_pbar=False)
        imf.apply_prewitt_x_filter(str(step_in), str(prewitt_x), use_pbar=False)
        imf.apply_prewitt_y_filter(str(step_in), str(prewitt_y), use_pbar=False)
        imf.apply_prewitt_filter(str(step_in), str(prewitt_m), use_pbar=False)
    finally:
        img_proc.reset_default_block_size()

    with rasterio.open(sobel_x) as sx, rasterio.open(sobel_y) as sy, rasterio.open(
        sobel_m
    ) as sm:
        gx = sx.read(1)
        gy = sy.read(1)
        gm = sm.read(1)
    assert gx[4, 3] == pytest.approx(40.0, abs=1e-4)
    assert gy[4, 3] == pytest.approx(0.0, abs=1e-4)
    assert gm[4, 3] == pytest.approx(40.0, abs=1e-4)
    assert gx[4, 1] == pytest.approx(0.0, abs=1e-4)

    with rasterio.open(prewitt_x) as px, rasterio.open(prewitt_y) as py, rasterio.open(
        prewitt_m
    ) as pm:
        assert px.read(1)[4, 3] == pytest.approx(30.0, abs=1e-4)
        assert py.read(1)[4, 3] == pytest.approx(0.0, abs=1e-4)
        assert pm.read(1)[4, 3] == pytest.approx(30.0, abs=1e-4)

    # Constant image checks for Gaussian smooth, 1st/2nd derivatives, Laplacian
    const_in = tmp_path / "const.tif"
    _write_raster(const_in, np.full((7, 7), 42.0, dtype=np.float32))
    gs_out = tmp_path / "gs.tif"
    g1_out = tmp_path / "g1.tif"
    g2_out = tmp_path / "g2.tif"
    lap_out = tmp_path / "lap.tif"
    imf.apply_gaussian_smooth_filter(
        str(const_in), str(gs_out), filter_size=5, stddev_x=1.0, stddev_y=1.5,
        filter_angle=30.0, use_pbar=False
    )
    imf.apply_gaussian_1st_deriv_filter(
        str(step_in), str(g1_out), filter_size=5, stddev_x=1.0, stddev_y=1.0,
        filter_angle=0.0, use_pbar=False
    )
    imf.apply_gaussian_2nd_deriv_filter(
        str(const_in), str(g2_out), filter_size=5, stddev_x=1.0, stddev_y=1.0,
        filter_angle=45.0, use_pbar=False
    )
    imf.apply_laplacian_filter(
        str(const_in), str(lap_out), filter_size=5, stddev=1.0, use_pbar=False
    )

    with rasterio.open(gs_out) as s_gs:
        assert s_gs.read(1) == pytest.approx(42.0, abs=1e-4)
    with rasterio.open(g1_out) as s_g1:
        g1 = s_g1.read(1)
        assert g1[4, 1] == pytest.approx(0.0, abs=1e-4)
        assert abs(g1[4, 3]) > 1.0
    with rasterio.open(g2_out) as s_g2, rasterio.open(lap_out) as s_lap:
        assert s_g2.read(1) == pytest.approx(0.0, abs=1e-4)
        assert s_lap.read(1) == pytest.approx(0.0, abs=1e-4)

    # Custom kernel with normalisation
    kern_out = tmp_path / "kern.tif"
    imf.apply_kernel_filter(
        str(const_in),
        str(kern_out),
        kernel=[[1.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 1.0]],
        normalise=True,
        use_pbar=False,
    )
    with rasterio.open(kern_out) as sk:
        assert sk.read(1) == pytest.approx(42.0, abs=1e-4)

    # Parameter validation errors
    with pytest.raises(rsgispy.RSGISPyException):
        imf.apply_mean_filter(str(const_in), str(tmp_path / "bad.tif"), filter_size=4)
    with pytest.raises(rsgispy.RSGISPyException):
        imf.apply_lee_filter(str(const_in), str(tmp_path / "bad_lee.tif"), n_looks=0)
    with pytest.raises(rsgispy.RSGISPyException):
        imf.apply_kernel_filter(
            str(const_in),
            str(tmp_path / "bad_k.tif"),
            kernel=[[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]],
            normalise=True,
        )
