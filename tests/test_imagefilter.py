"""
Tests for :mod:`rsgispy.imagefilter`.
"""

import inspect

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
            (imf.stats.apply_mean_filter, np.mean(win_vals)),
            (imf.stats.apply_median_filter, np.median(win_vals)),
            (imf.stats.apply_min_filter, np.min(win_vals)),
            (imf.stats.apply_max_filter, np.max(win_vals)),
            (imf.stats.apply_range_filter, np.max(win_vals) - np.min(win_vals)),
            (imf.stats.apply_total_filter, np.sum(win_vals)),
            (imf.stats.apply_stddev_filter, np.std(win_vals)),
            (imf.stats.apply_texture_var_filter, np.var(win_vals)),
            (imf.stats.apply_mean_diff_filter, np.mean(win_vals - 13.0)),
            (imf.stats.apply_mean_diff_abs_filter, np.mean(np.abs(win_vals - 13.0))),
            (imf.stats.apply_total_diff_filter, np.sum(win_vals - 13.0)),
            (imf.stats.apply_total_diff_abs_filter, np.sum(np.abs(win_vals - 13.0))),
            (imf.stats.apply_coeff_of_var_filter, np.std(win_vals) / np.mean(win_vals)),
            (imf.stats.apply_norm_var_filter, np.var(win_vals) / (np.mean(win_vals) ** 2)),
            (
                imf.stats.apply_norm_var_sqrt_filter,
                np.std(win_vals) / abs(np.mean(win_vals)),
            ),
            (
                imf.stats.apply_norm_var_ln_filter,
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
    imf.stats.apply_mode_filter(
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
    imf.smooth.apply_kuwahara_filter(str(step_in), str(step_out), filter_size=5, use_pbar=False)
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
    imf.sar.apply_lee_filter(
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
        imf.edge.apply_sobel_x_filter(str(step_in), str(sobel_x), use_pbar=False)
        imf.edge.apply_sobel_y_filter(str(step_in), str(sobel_y), use_pbar=False)
        imf.edge.apply_sobel_filter(str(step_in), str(sobel_m), use_pbar=False)
        imf.edge.apply_prewitt_x_filter(str(step_in), str(prewitt_x), use_pbar=False)
        imf.edge.apply_prewitt_y_filter(str(step_in), str(prewitt_y), use_pbar=False)
        imf.edge.apply_prewitt_filter(str(step_in), str(prewitt_m), use_pbar=False)
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
    imf.smooth.apply_gaussian_smooth_filter(
        str(const_in), str(gs_out), filter_size=5, stddev_x=1.0, stddev_y=1.5,
        filter_angle=30.0, use_pbar=False
    )
    imf.edge.apply_gaussian_1st_deriv_filter(
        str(step_in), str(g1_out), filter_size=5, stddev_x=1.0, stddev_y=1.0,
        filter_angle=0.0, use_pbar=False
    )
    imf.edge.apply_gaussian_2nd_deriv_filter(
        str(const_in), str(g2_out), filter_size=5, stddev_x=1.0, stddev_y=1.0,
        filter_angle=45.0, use_pbar=False
    )
    imf.edge.apply_laplacian_filter(
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
    imf.smooth.apply_kernel_filter(
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
        imf.stats.apply_mean_filter(str(const_in), str(tmp_path / "bad.tif"), filter_size=4)
    with pytest.raises(rsgispy.RSGISPyException):
        imf.sar.apply_lee_filter(str(const_in), str(tmp_path / "bad_lee.tif"), n_looks=0)
    with pytest.raises(rsgispy.RSGISPyException):
        imf.smooth.apply_kernel_filter(
            str(const_in),
            str(tmp_path / "bad_k.tif"),
            kernel=[[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]],
            normalise=True,
        )


def test_imagefilter_submodules_and_sar_filters(tmp_path):
    """Test submodule imports and all SAR backscatter speckle filters."""
    import rsgispy.imagefilter.edge as if_edge
    import rsgispy.imagefilter.sar as if_sar
    import rsgispy.imagefilter.smooth as if_smooth
    import rsgispy.imagefilter.stats as if_stats
    import rsgispy.imagefilter.texture as if_tex

    # Package exports only its submodules, not individual filter functions
    assert imf.__all__ == ["stats", "smooth", "edge", "sar", "texture", "chained"]
    assert imf.stats is if_stats
    assert imf.smooth is if_smooth
    assert imf.edge is if_edge
    assert imf.sar is if_sar
    assert imf.texture is if_tex
    assert not hasattr(imf, "apply_mean_filter")

    # Synthetic SAR intensity image: low region + high region + strong point target + nodata
    sar_in = tmp_path / "sar_test.tif"
    rng = np.random.default_rng(77)
    sar = rng.normal(50.0, 6.0, size=(20, 20)).astype(np.float32)
    sar[:, 10:] = rng.normal(200.0, 20.0, size=(20, 10)).astype(np.float32)
    sar[4, 4] = 800.0  # Point target inside uniform low region
    sar[0, 0] = -9999.0
    _write_raster(sar_in, sar, nodata=-9999.0, descriptions=["VV"])

    img_proc.set_default_block_size(8)
    try:
        for name, fn, extra in [
            ("elee", if_sar.apply_enhanced_lee_filter, {"n_looks": 4.0, "damping": 1.0}),
            ("frost", if_sar.apply_frost_filter, {"damping": 1.0}),
            ("efrost", if_sar.apply_enhanced_frost_filter, {"n_looks": 4.0, "damping": 1.0}),
            ("kuan", if_sar.apply_kuan_filter, {"n_looks": 4.0}),
            ("gmap", if_sar.apply_gamma_map_filter, {"n_looks": 4.0}),
            ("reflee", if_sar.apply_refined_lee_filter, {"n_looks": 4.0}),
        ]:
            out_p = tmp_path / f"sar_{name}.tif"
            fn(
                str(sar_in),
                str(out_p),
                filter_size=5,
                use_pbar=False,
                **extra,
            )
            with rasterio.open(out_p) as src:
                out = src.read(1)
                assert list(src.descriptions) == ["VV"]
                assert src.nodata == -9999.0
            assert out[0, 0] == -9999.0
            # Homogeneous corner variance reduced compared to raw input
            assert np.std(out[12:18, 2:8]) < np.std(sar[12:18, 2:8])
            # Point target preserved well above background mean (50)
            assert out[4, 4] > 200.0
    finally:
        img_proc.reset_default_block_size()

    # Step-edge test for Refined Lee vs uniform blurring
    edge_in = tmp_path / "sar_step.tif"
    edge_arr = np.full((15, 15), 20.0, dtype=np.float32)
    edge_arr[:, 8:] = 200.0
    _write_raster(edge_in, edge_arr)
    ref_out = tmp_path / "sar_reflee_step.tif"
    if_sar.apply_refined_lee_filter(
        str(edge_in), str(ref_out), filter_size=7, n_looks=4.0, use_pbar=False
    )
    with rasterio.open(ref_out) as src:
        ref = src.read(1)
    assert ref[7, 7] == pytest.approx(20.0, abs=1e-4)
    assert ref[7, 8] == pytest.approx(200.0, abs=1e-4)


def test_glcm_texture_filters(tmp_path):
    """Test GLCM texture properties against skimage and multi-band/nodata behaviour."""
    import rsgispy.imagefilter.texture as if_tex

    sk_feat = pytest.importorskip("skimage.feature")

    rng = np.random.default_rng(12)
    img_arr = rng.integers(0, 8, size=(12, 12), dtype=np.uint8).astype(np.float32)
    img2 = np.full((12, 12), 3.0, dtype=np.float32)
    in_p = tmp_path / "glcm_in.tif"
    _write_raster(
        in_p,
        np.stack([img_arr, img2], axis=0),
        descriptions=["Red", "Const"],
    )

    out_p = tmp_path / "glcm_out.tif"
    props = [
        if_tex.GLCM_CONTRAST,
        if_tex.GLCM_DISSIMILARITY,
        if_tex.GLCM_HOMOGENEITY,
        if_tex.GLCM_ASM,
        if_tex.GLCM_ENERGY,
        if_tex.GLCM_CORRELATION,
        if_tex.GLCM_ENTROPY,
        if_tex.GLCM_MEAN,
        if_tex.GLCM_VARIANCE,
        if_tex.GLCM_MAX_PROB,
    ]
    img_proc.set_default_block_size(5)
    try:
        if_tex.apply_glcm_filter(
            str(in_p),
            str(out_p),
            properties=props,
            filter_size=5,
            distances=1,
            angles=(0.0, 45.0, 90.0, 135.0),
            n_levels=8,
            min_val=0.0,
            max_val=7.0,
            symmetric=True,
            average_angles=True,
            use_pbar=False,
        )
    finally:
        img_proc.reset_default_block_size()

    with rasterio.open(out_p) as src:
        assert src.count == 20
        assert src.descriptions[0] == "Red_glcm_contrast"
        assert src.descriptions[9] == "Red_glcm_max_prob"
        assert src.descriptions[10] == "Const_glcm_contrast"
        data = src.read()

    # Exact comparison on interior 5x5 patch at (6, 6) against skimage
    patch = img_arr[4:9, 4:9].astype(np.uint8)
    glcm = sk_feat.graycomatrix(
        patch,
        distances=[1],
        angles=[0.0, np.pi / 4.0, np.pi / 2.0, 3.0 * np.pi / 4.0],
        levels=8,
        symmetric=True,
        normed=True,
    )
    for b_idx, prop in enumerate(
        ["contrast", "dissimilarity", "homogeneity", "ASM", "energy", "correlation"]
    ):
        exp_val = float(np.mean(sk_feat.graycoprops(glcm, prop)))
        assert data[b_idx, 6, 6] == pytest.approx(exp_val, rel=1e-5, abs=1e-5)

    # Constant second band: contrast=0, homogeneity=1, energy=1, entropy=0, max_prob=1
    assert data[10, 6, 6] == pytest.approx(0.0, abs=1e-5)
    assert data[12, 6, 6] == pytest.approx(1.0, abs=1e-5)
    assert data[14, 6, 6] == pytest.approx(1.0, abs=1e-5)
    assert data[16, 6, 6] == pytest.approx(0.0, abs=1e-5)
    assert data[19, 6, 6] == pytest.approx(1.0, abs=1e-5)

    # Non-averaged multi-angle output + nodata mask check
    nd_in = tmp_path / "glcm_nd.tif"
    arr_nd = img_arr.copy()
    arr_nd[0, 0] = -9999.0
    _write_raster(nd_in, arr_nd, nodata=-9999.0, descriptions=["B1"])
    out_sep = tmp_path / "glcm_sep.tif"
    if_tex.apply_glcm_filter(
        str(nd_in),
        str(out_sep),
        properties="contrast",
        filter_size=5,
        distances=[1, 2],
        angles=[0.0, 90.0],
        n_levels=8,
        average_angles=False,
        use_pbar=False,
    )
    with rasterio.open(out_sep) as src:
        assert src.count == 4
        assert list(src.descriptions) == [
            "B1_glcm_contrast_d1_a0",
            "B1_glcm_contrast_d1_a90",
            "B1_glcm_contrast_d2_a0",
            "B1_glcm_contrast_d2_a90",
        ]
        assert src.read(1)[0, 0] == -9999.0

    # Validation checks
    with pytest.raises(rsgispy.RSGISPyException):
        if_tex.apply_glcm_filter(str(nd_in), str(tmp_path / "err1.tif"), properties="bad_prop")
    with pytest.raises(rsgispy.RSGISPyException):
        if_tex.apply_glcm_filter(str(nd_in), str(tmp_path / "err2.tif"), filter_size=3, distances=3)
    with pytest.raises(rsgispy.RSGISPyException):
        if_tex.apply_glcm_filter(str(nd_in), str(tmp_path / "err3.tif"), min_val=0.0, max_val=None)


def test_imagefilter_v1_signatures(tmp_path):
    """Verify v1.0 keyword-only tail on all raster-producing imagefilter functions."""
    kw_tail = {
        "gdalformat": "GTIFF",
        "datatype": None,
        "no_data_val": None,
        "calc_stats": True,
        "use_pbar": True,
    }
    submodules = [imf.stats, imf.smooth, imf.edge, imf.sar, imf.texture, imf.chained]
    checked = 0
    for mod in submodules:
        for name in mod.__all__:
            if not name.startswith("apply_"):
                continue
            fn = getattr(mod, name)
            sig = inspect.signature(fn)
            params = list(sig.parameters.values())
            assert params[0].name == "input_img"
            assert params[1].name == "output_img"
            for kw_name, kw_default in kw_tail.items():
                assert kw_name in sig.parameters, f"{mod.__name__}.{name} missing {kw_name}"
                p = sig.parameters[kw_name]
                assert p.kind == inspect.Parameter.KEYWORD_ONLY, (
                    f"{mod.__name__}.{name}.{kw_name} must be keyword-only"
                )
                assert p.default == kw_default
            checked += 1

    # 17 stats + 4 smooth + 9 edge + 7 sar + 1 texture + 1 chained = 39 functions
    assert checked == 39

    # Passing gdalformat positionally must raise TypeError
    in_p = tmp_path / "sig_in.tif"
    _write_raster(in_p, np.ones((5, 5), dtype=np.float32))
    with pytest.raises(TypeError):
        imf.stats.apply_mean_filter(str(in_p), str(tmp_path / "sig_out.tif"), 3, "GTIFF")
    with pytest.raises(TypeError):
        imf.edge.apply_sobel_filter(str(in_p), str(tmp_path / "sig_sob.tif"), "GTIFF")


def _edge_conv_reference(arr, kernel, flip):
    """Reference 2D convolution (``flip=True``) or correlation of ``arr``
    with an edge-padded border."""
    k = np.asarray(kernel, dtype=np.float64)
    if flip:
        k = k[::-1, ::-1]
    kh, kw = k.shape
    pad = np.pad(arr.astype(np.float64), ((kh // 2, kh // 2), (kw // 2, kw // 2)), mode="edge")
    out = np.zeros(arr.shape, dtype=np.float64)
    for i in range(kh):
        for j in range(kw):
            out += k[i, j] * pad[i:i + arr.shape[0], j:j + arr.shape[1]]
    return out


def test_kernel_filter_convolves_and_correlation_filter_correlates(tmp_path):
    """apply_kernel_filter is a true convolution (kernel flipped in both
    axes); apply_kernel_correlation_filter applies the kernel unflipped."""
    arr = np.arange(1, 36, dtype=np.float32).reshape(5, 7)
    in_p = tmp_path / "shift_in.tif"
    _write_raster(in_p, arr)

    def _run(fn, kernel, tag):
        out_p = tmp_path / f"{tag}.tif"
        fn(str(in_p), str(out_p), kernel, use_pbar=False)
        with rasterio.open(out_p) as src:
            return src.read(1).astype(np.float64)

    shift = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]
    conv = _run(imf.smooth.apply_kernel_filter, shift, "conv_shift")
    corr = _run(imf.smooth.apply_kernel_correlation_filter, shift, "corr_shift")
    # Convolution moves the image one column right: out[r, c] = in[r, c - 1].
    np.testing.assert_array_equal(conv[:, 1:], arr[:, :-1])
    np.testing.assert_array_equal(conv[:, 0], arr[:, 0])
    # Correlation moves it one column left: out[r, c] = in[r, c + 1].
    np.testing.assert_array_equal(corr[:, :-1], arr[:, 1:])
    np.testing.assert_array_equal(corr[:, -1], arr[:, -1])

    # A kernel asymmetric in both axes, with mixed weights.
    asym = [[1.0, 2.0, 0.0], [0.0, 0.0, 3.0], [0.0, -1.0, 0.5]]
    conv = _run(imf.smooth.apply_kernel_filter, asym, "conv_asym")
    corr = _run(imf.smooth.apply_kernel_correlation_filter, asym, "corr_asym")
    np.testing.assert_allclose(conv, _edge_conv_reference(arr, asym, True), rtol=1e-6)
    np.testing.assert_allclose(corr, _edge_conv_reference(arr, asym, False), rtol=1e-6)

    # A symmetric kernel gives the same result either way.
    sharpen = [[0.0, -1.0, 0.0], [-1.0, 5.0, -1.0], [0.0, -1.0, 0.0]]
    np.testing.assert_array_equal(
        _run(imf.smooth.apply_kernel_filter, sharpen, "conv_sharp"),
        _run(imf.smooth.apply_kernel_correlation_filter, sharpen, "corr_sharp"),
    )
