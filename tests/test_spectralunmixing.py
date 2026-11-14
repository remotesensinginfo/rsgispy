"""
Tests for :mod:`rsgispy.imagecalc.spectralunmixing`.
"""

import inspect

import geopandas
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

import rsgispy
import rsgispy.imagecalc.spectralunmixing as su
import rsgispy.zonalstats as zs


def _write_raster(path, data, origin=(0.0, 100.0), res=10.0, nodata=None):
    """Write a 3D ``(bands, height, width)`` GeoTIFF raster."""
    count, height, width = data.shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=str(data.dtype),
        crs="EPSG:32630",
        transform=from_origin(origin[0], origin[1], res, res),
        nodata=nodata,
    ) as dst:
        dst.write(data)


def test_exports_and_default_gdalformat():
    """Verify recommended/optional unmixing exports and GTIFF defaults."""
    funcs = [
        su.spec_unmix_ucls,
        su.exhcon_linear_spec_unmix,
        su.rescale_unmixing_results,
        su.predict_refl_linear_unmixing,
        su.calc_unmixing_rmse_residual_err,
        su.summarise_multi_endmember_linear_unmixing,
        su.calc_ppi,
    ]
    if su.HAVE_SCIPY:
        funcs.extend([su.spec_unmix_nnls, su.spec_unmix_fcls])
    if su.HAVE_PYSPTOOLS:
        funcs.extend([su.spec_unmix_spts_ucls, su.spec_unmix_spts_nnls])
        if su.HAVE_CVXOPT:
            funcs.append(su.spec_unmix_spts_fcls)
    if su.HAVE_PYMCR:
        funcs.extend([su.spec_unmix_pymcr_nnls, su.spec_unmix_pymcr_fcls])

    for fn in funcs:
        sig = inspect.signature(fn)
        assert sig.parameters["gdalformat"].default == "GTIFF"


def test_mtxt_roundtrip_and_equality(tmp_path):
    """Verify .mtxt write, read (with gain/weight), plot, and equality checks."""
    em_file = tmp_path / "em.mtxt"
    em_file2 = tmp_path / "em2.mtxt"
    plot_file = tmp_path / "em.png"

    em = np.array([[0.2, 0.4, 0.8], [0.6, 0.2, 0.1]], dtype=np.float64)
    su.write_endmembers_mtxt(em, str(em_file))
    su.write_endmembers_mtxt(em + 0.00001, str(em_file2))

    m, n, read_em = su.read_endmembers_mtxt(str(em_file))
    assert (m, n) == (2, 3)
    np.testing.assert_allclose(read_em, em)

    m_g, n_g, read_gw = su.read_endmembers_mtxt(str(em_file), gain=2.0, weight=5.0)
    assert (m_g, n_g) == (2, 4)
    np.testing.assert_allclose(read_gw[:, :3], em / 2.0)
    np.testing.assert_allclose(read_gw[:, 3], [5.0, 5.0])

    assert su.are_endmembers_equal(str(em_file), str(em_file2), flt_dif=0.0001)
    assert not su.are_endmembers_equal(str(em_file), str(em_file2), flt_dif=1e-7)

    su.plot_endmembers(
        str(em_file),
        ["Soil", "Veg"],
        str(plot_file),
        plot_title="Test Endmembers",
        wavelengths=[480, 560, 660],
    )
    assert plot_file.exists() and plot_file.stat().st_size > 0


def test_extract_avg_endmembers_and_zonalstats_constants(tmp_path):
    """Verify extract_avg_endmembers extracts per-polygon mean spectra cleanly."""
    assert zs.METHOD_POLYCONTAINSPIXELCENTER == 1
    assert zs.METHOD_POLYOVERLAPSPIXEL == 2

    img_path = tmp_path / "img.tif"
    vec_path = tmp_path / "rois.gpkg"
    out_base = tmp_path / "extracted_em"

    # 4x4 grid, origin (0, 40), res=10 -> x in [0, 40], y in [0, 40]
    arr = np.zeros((3, 4, 4), dtype=np.float32)
    arr[:, :2, :2] = np.array([0.2, 0.4, 0.8])[:, None, None]
    arr[:, 2:, 2:] = np.array([0.6, 0.2, 0.1])[:, None, None]
    _write_raster(img_path, arr, origin=(0.0, 40.0), res=10.0)

    gdf = geopandas.GeoDataFrame(
        {"cls": ["A", "B"]},
        geometry=[box(0.0, 20.0, 20.0, 40.0), box(20.0, 0.0, 40.0, 20.0)],
        crs="EPSG:32630",
    )
    gdf.to_file(vec_path, driver="GPKG", layer="rois")

    su.extract_avg_endmembers(
        str(img_path),
        str(vec_path),
        "rois",
        str(out_base),
        pxl_in_poly_method=zs.METHOD_POLYCONTAINSPIXELCENTER,
    )

    mtxt_path = tmp_path / "extracted_em.mtxt"
    assert mtxt_path.exists()
    m, n, em = su.read_endmembers_mtxt(str(mtxt_path))
    assert (m, n) == (2, 3)
    np.testing.assert_allclose(em[0], [0.2, 0.4, 0.8], atol=1e-6)
    np.testing.assert_allclose(em[1], [0.6, 0.2, 0.1], atol=1e-6)


def test_exhcon_unmixing_and_downstream_metrics(tmp_path):
    """Verify exhaustive constrained unmixing, rescaling, prediction, error, and PPI."""
    img_path = tmp_path / "refl.tif"
    em_path = tmp_path / "em.mtxt"
    exh_path = tmp_path / "exh.tif"
    resc_path = tmp_path / "resc.tif"
    pred_path = tmp_path / "pred.tif"
    err_path = tmp_path / "err.tif"
    ppi_path = tmp_path / "ppi.tif"

    em1 = np.array([0.2, 0.4, 0.8], dtype=np.float64)
    em2 = np.array([0.6, 0.2, 0.1], dtype=np.float64)
    su.write_endmembers_mtxt(np.vstack([em1, em2]), str(em_path))

    # Build a 2x2 image: exact EM1, exact EM2, 50/50 mix (of unnormed), zeros
    arr = np.zeros((3, 2, 2), dtype=np.float32)
    arr[:, 0, 0] = em1
    arr[:, 0, 1] = em2
    arr[:, 1, 0] = 0.5 * em1 + 0.5 * em2
    arr[:, 1, 1] = [0.0, 0.0, 0.0]
    _write_raster(img_path, arr, origin=(0.0, 20.0), res=10.0)

    su.exhcon_linear_spec_unmix(
        str(img_path),
        str(exh_path),
        gdalformat="GTIFF",
        datatype=rsgispy.TYPE_32FLOAT,
        endmember_file=str(em_path),
        step_res=0.1,
        use_pbar=False,
    )
    with rasterio.open(exh_path) as src:
        out_exh = src.read()
    assert out_exh.shape == (3, 2, 2)
    np.testing.assert_allclose(out_exh[:, 0, 0], [1.0, 0.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(out_exh[:, 0, 1], [0.0, 1.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(out_exh[:, 1, 1], [0.0, 0.0, 0.0], atol=1e-6)
    assert abs((out_exh[0, 1, 0] + out_exh[1, 1, 0]) - 1.0) < 1e-5

    # Test rescale_unmixing_results with negative and >1 values
    raw_coef_path = tmp_path / "raw_coef.tif"
    raw_coef = np.array(
        [[[0.6, -0.2], [0.3, 0.0]], [[0.6, 0.8], [0.7, 0.0]]],
        dtype=np.float32,
    )
    _write_raster(raw_coef_path, raw_coef, origin=(0.0, 20.0), res=10.0)
    su.rescale_unmixing_results(
        str(raw_coef_path), str(resc_path), calc_stats=False, use_pbar=False
    )
    with rasterio.open(resc_path) as src:
        resc = src.read()
    np.testing.assert_allclose(resc[:, 0, 0], [0.5, 0.5], atol=1e-6)
    np.testing.assert_allclose(resc[:, 0, 1], [0.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(resc[:, 1, 1], [0.0, 0.0], atol=1e-6)

    # Exact abundance raster -> predicted reflectance and zero error
    exact_coef = np.array(
        [[[1.0, 0.0], [0.5, 0.0]], [[0.0, 1.0], [0.5, 0.0]]],
        dtype=np.float32,
    )
    exact_coef_path = tmp_path / "exact_coef.tif"
    _write_raster(exact_coef_path, exact_coef, origin=(0.0, 20.0), res=10.0)

    su.predict_refl_linear_unmixing(
        str(exact_coef_path),
        str(em_path),
        str(pred_path),
        calc_stats=False,
        use_pbar=False,
    )
    with rasterio.open(pred_path) as src:
        pred = src.read()
    np.testing.assert_allclose(pred, arr, atol=1e-6)

    su.calc_unmixing_rmse_residual_err(
        str(img_path),
        str(exact_coef_path),
        str(em_path),
        str(err_path),
        calc_stats=True,
        use_pbar=False,
    )
    with rasterio.open(err_path) as src:
        err = src.read()
        assert list(src.descriptions) == ["RMSE", "RMSPE", "Residual"]
    np.testing.assert_allclose(err, 0.0, atol=1e-5)

    # PPI test
    su.calc_ppi(
        str(img_path),
        str(ppi_path),
        niters=50,
        seed=42,
        calc_stats=False,
        use_pbar=False,
    )
    with rasterio.open(ppi_path) as src:
        ppi = src.read(1)
        assert src.dtypes[0] == "uint16"
    assert int(np.sum(ppi)) > 0


def test_summarise_multi_endmember_linear_unmixing(tmp_path):
    """Verify multi-model summary picks the lower-RMSPE model per pixel."""
    refl_path = tmp_path / "refl.tif"
    em_a_path = tmp_path / "em_a.mtxt"
    em_b_path = tmp_path / "em_b.mtxt"
    unmix_a_path = tmp_path / "unmix_a.tif"
    unmix_b_path = tmp_path / "unmix_b.tif"
    out_unmix = tmp_path / "best_unmix.tif"
    out_ref = tmp_path / "best_ref.tif"

    em_veg = np.array([0.1, 0.3, 0.8], dtype=np.float64)
    em_soil = np.array([0.5, 0.4, 0.2], dtype=np.float64)
    em_water = np.array([0.2, 0.1, 0.05], dtype=np.float64)

    su.write_endmembers_mtxt(np.vstack([em_veg, em_soil]), str(em_a_path))
    su.write_endmembers_mtxt(np.vstack([em_veg, em_water]), str(em_b_path))

    # Left half: pure soil (perfectly explained by model A); right half: pure water (model B)
    refl = np.zeros((3, 2, 2), dtype=np.float32)
    refl[:, :, 0] = em_soil[:, None]
    refl[:, :, 1] = em_water[:, None]
    _write_raster(refl_path, refl, origin=(0.0, 20.0), res=10.0)

    coef_a = np.zeros((2, 2, 2), dtype=np.float32)
    coef_a[1, :, 0] = 1.0  # model A gets col 0 exact (Soil=1) and fails col 1 (0)
    _write_raster(unmix_a_path, coef_a, origin=(0.0, 20.0), res=10.0)

    coef_b = np.zeros((2, 2, 2), dtype=np.float32)
    coef_b[1, :, 1] = 1.0  # model B gets col 1 exact (Water=1) and fails col 0
    _write_raster(unmix_b_path, coef_b, origin=(0.0, 20.0), res=10.0)

    info_a = su.ImageEndmemberInfo(str(unmix_a_path), str(em_a_path), ["Veg", "Soil"])
    info_b = su.ImageEndmemberInfo(str(unmix_b_path), str(em_b_path), ["Veg", "Water"])

    su.summarise_multi_endmember_linear_unmixing(
        str(refl_path),
        [info_a, info_b],
        str(out_unmix),
        str(out_ref),
        tmp_dir=str(tmp_path / "tmp"),
        calc_stats=True,
        use_pbar=False,
    )

    with rasterio.open(out_ref) as src:
        ref = src.read(1)
    assert np.all(ref[:, 0] == 1)
    assert np.all(ref[:, 1] == 2)

    with rasterio.open(out_unmix) as src:
        comb = src.read()
        assert list(src.descriptions) == ["Veg", "Soil", "Water"]
    np.testing.assert_allclose(comb[:, 0, 0], [0.0, 1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(comb[:, 0, 1], [0.0, 0.0, 1.0], atol=1e-6)


@pytest.mark.skipif(not su.HAVE_PYSPTOOLS, reason="pysptools not installed")
def test_pysptools_unmixers(tmp_path):
    """Verify PySptools UCLS, NNLS, and FCLS end-to-end when installed."""
    img_path = tmp_path / "refl.tif"
    msk_path = tmp_path / "msk.tif"
    em_path = tmp_path / "em.mtxt"

    em1 = np.array([0.2, 0.4, 0.8], dtype=np.float64)
    em2 = np.array([0.6, 0.2, 0.1], dtype=np.float64)
    su.write_endmembers_mtxt(np.vstack([em1, em2]), str(em_path))

    arr = np.zeros((3, 2, 2), dtype=np.float32)
    arr[:, 0, 0] = 0.3 * em1 + 0.7 * em2
    arr[:, 0, 1] = em1
    arr[:, 1, 0] = em2
    arr[:, 1, 1] = 0.5 * em1 + 0.5 * em2
    _write_raster(img_path, arr, origin=(0.0, 20.0), res=10.0)

    msk = np.ones((1, 2, 2), dtype=np.uint8)
    _write_raster(msk_path, msk, origin=(0.0, 20.0), res=10.0)

    out_ucls = tmp_path / "ucls.tif"
    su.spec_unmix_spts_ucls(
        str(img_path), str(msk_path), 1, str(out_ucls), str(em_path),
        calc_stats=False, use_pbar=False,
    )
    with rasterio.open(out_ucls) as src:
        np.testing.assert_allclose(src.read()[:, 0, 0], [0.3, 0.7], atol=1e-3)

    out_nnls = tmp_path / "nnls.tif"
    su.spec_unmix_spts_nnls(
        str(img_path), str(msk_path), 1, str(out_nnls), str(em_path),
        weight=10.0, calc_stats=False, use_pbar=False,
    )
    with rasterio.open(out_nnls) as src:
        np.testing.assert_allclose(src.read()[:, 0, 0], [0.3, 0.7], atol=1e-2)

    if su.HAVE_CVXOPT:
        out_fcls = tmp_path / "fcls.tif"
        su.spec_unmix_spts_fcls(
            str(img_path), str(msk_path), 1, str(out_fcls), str(em_path),
            calc_stats=False, use_pbar=False,
        )
        with rasterio.open(out_fcls) as src:
            np.testing.assert_allclose(src.read()[:, 0, 0], [0.3, 0.7], atol=1e-2)


@pytest.mark.skipif(not su.HAVE_PYMCR, reason="pymcr not installed")
def test_pymcr_unmixers(tmp_path):
    """Verify PyMCR NNLS and FCLS end-to-end when installed."""
    img_path = tmp_path / "refl.tif"
    msk_path = tmp_path / "msk.tif"
    em_path = tmp_path / "em.mtxt"

    em1 = np.array([0.2, 0.4, 0.8], dtype=np.float64)
    em2 = np.array([0.6, 0.2, 0.1], dtype=np.float64)
    su.write_endmembers_mtxt(np.vstack([em1, em2]), str(em_path))

    arr = np.zeros((3, 2, 2), dtype=np.float32)
    arr[:, 0, 0] = 0.3 * em1 + 0.7 * em2
    arr[:, 0, 1] = em1
    arr[:, 1, 0] = em2
    arr[:, 1, 1] = 0.5 * em1 + 0.5 * em2
    _write_raster(img_path, arr, origin=(0.0, 20.0), res=10.0)

    msk = np.ones((1, 2, 2), dtype=np.uint8)
    _write_raster(msk_path, msk, origin=(0.0, 20.0), res=10.0)

    out_nnls = tmp_path / "pymcr_nnls.tif"
    su.spec_unmix_pymcr_nnls(
        str(img_path), str(msk_path), 1, str(out_nnls), str(em_path),
        calc_stats=False, use_pbar=False,
    )
    with rasterio.open(out_nnls) as src:
        np.testing.assert_allclose(src.read()[:, 0, 0], [0.3, 0.7], atol=1e-2)

    out_fcls = tmp_path / "pymcr_fcls.tif"
    su.spec_unmix_pymcr_fcls(
        str(img_path), str(msk_path), 1, str(out_fcls), str(em_path),
        calc_stats=False, use_pbar=False,
    )
    with rasterio.open(out_fcls) as src:
        np.testing.assert_allclose(src.read()[:, 0, 0], [0.3, 0.7], atol=1e-2)


def test_recommended_numpy_ucls_unmixer(tmp_path):
    """Verify recommended pure-NumPy UCLS unmixing with and without a mask."""
    img_path = tmp_path / "refl.tif"
    msk_path = tmp_path / "msk.tif"
    em_path = tmp_path / "em.mtxt"

    em1 = np.array([0.2, 0.4, 0.8], dtype=np.float64)
    em2 = np.array([0.6, 0.2, 0.1], dtype=np.float64)
    em3 = np.array([0.3, 0.7, 0.2], dtype=np.float64)
    su.write_endmembers_mtxt(np.vstack([em1, em2, em3]), str(em_path))

    arr = np.zeros((3, 2, 2), dtype=np.float32)
    arr[:, 0, 0] = 0.2 * em1 + 0.5 * em2 + 0.3 * em3
    arr[:, 0, 1] = 0.8 * em1 + 0.2 * em2
    arr[:, 1, 0] = em3
    arr[:, 1, 1] = [0.0, 0.0, 0.0]
    _write_raster(img_path, arr, origin=(0.0, 20.0), res=10.0)

    msk = np.array([[[1, 1], [0, 1]]], dtype=np.uint8)
    _write_raster(msk_path, msk, origin=(0.0, 20.0), res=10.0)

    out_masked = tmp_path / "numpy_ucls.tif"
    su.spec_unmix_ucls(
        str(img_path),
        str(msk_path),
        1,
        str(out_masked),
        str(em_path),
        calc_stats=False,
        use_pbar=False,
    )
    with rasterio.open(out_masked) as src:
        res = src.read()
    np.testing.assert_allclose(res[:, 0, 0], [0.2, 0.5, 0.3], atol=1e-5)
    np.testing.assert_allclose(res[:, 0, 1], [0.8, 0.2, 0.0], atol=1e-5)
    np.testing.assert_allclose(res[:, 1, 0], [0.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(res[:, 1, 1], [0.0, 0.0, 0.0], atol=1e-6)

    out_nomask = tmp_path / "numpy_ucls_nomask.tif"
    su.spec_unmix_ucls(
        str(img_path),
        None,
        1,
        str(out_nomask),
        str(em_path),
        weight=10.0,
        calc_stats=False,
        use_pbar=False,
    )
    with rasterio.open(out_nomask) as src:
        res_nm = src.read()
    np.testing.assert_allclose(res_nm[:, 1, 0], [0.0, 0.0, 1.0], atol=1e-5)
    np.testing.assert_allclose(res_nm[:, 1, 1], [0.0, 0.0, 0.0], atol=1e-6)


@pytest.mark.skipif(not su.HAVE_SCIPY, reason="scipy not installed")
def test_recommended_scipy_nnls_fcls_unmixers(tmp_path):
    """Verify recommended SciPy NNLS and FCLS unmixers including constraints."""
    img_path = tmp_path / "refl.tif"
    msk_path = tmp_path / "msk.tif"
    em_path = tmp_path / "em.mtxt"

    em1 = np.array([0.1, 0.2, 0.7, 0.3], dtype=np.float64)
    em2 = np.array([0.6, 0.3, 0.1, 0.2], dtype=np.float64)
    em3 = np.array([0.2, 0.6, 0.3, 0.7], dtype=np.float64)
    su.write_endmembers_mtxt(np.vstack([em1, em2, em3]), str(em_path))

    arr = np.zeros((4, 2, 2), dtype=np.float32)
    arr[:, 0, 0] = 0.25 * em1 + 0.45 * em2 + 0.30 * em3
    arr[:, 0, 1] = em1
    # Scaled sum != 1 pixel to test NNLS vs FCLS constraints
    arr[:, 1, 0] = 1.4 * em2 + 0.2 * em3
    # Perturbed out-of-simplex pixel
    arr[:, 1, 1] = np.clip(1.2 * em1 - 0.15 * em2 + 0.05 * em3, 0.01, None)
    _write_raster(img_path, arr, origin=(0.0, 20.0), res=10.0)

    msk = np.ones((1, 2, 2), dtype=np.uint8)
    _write_raster(msk_path, msk, origin=(0.0, 20.0), res=10.0)

    out_nnls = tmp_path / "scipy_nnls.tif"
    su.spec_unmix_nnls(
        str(img_path),
        str(msk_path),
        1,
        str(out_nnls),
        str(em_path),
        calc_stats=False,
        use_pbar=False,
    )
    with rasterio.open(out_nnls) as src:
        nnls_arr = src.read()
    assert np.all(nnls_arr >= -1e-7)
    np.testing.assert_allclose(nnls_arr[:, 0, 0], [0.25, 0.45, 0.30], atol=1e-5)
    np.testing.assert_allclose(nnls_arr[:, 1, 0], [0.0, 1.4, 0.2], atol=1e-5)

    out_fcls = tmp_path / "scipy_fcls.tif"
    su.spec_unmix_fcls(
        str(img_path),
        str(msk_path),
        1,
        str(out_fcls),
        str(em_path),
        calc_stats=False,
        use_pbar=False,
    )
    with rasterio.open(out_fcls) as src:
        fcls_arr = src.read()
    assert np.all(fcls_arr >= -1e-7)
    np.testing.assert_allclose(np.sum(fcls_arr, axis=0), 1.0, atol=1e-5)
    np.testing.assert_allclose(fcls_arr[:, 0, 0], [0.25, 0.45, 0.30], atol=1e-5)
    np.testing.assert_allclose(fcls_arr[:, 0, 1], [1.0, 0.0, 0.0], atol=1e-5)

    # Scale-invariance check (reflectance scaled by 10000 with gain=1)
    scaled_img = tmp_path / "refl_scaled.tif"
    scaled_em = tmp_path / "em_scaled.mtxt"
    _write_raster(scaled_img, (arr * 10000.0).astype(np.float32), origin=(0.0, 20.0), res=10.0)
    su.write_endmembers_mtxt(np.vstack([em1, em2, em3]) * 10000.0, str(scaled_em))
    out_fcls_scaled = tmp_path / "scipy_fcls_scaled.tif"
    su.spec_unmix_fcls(
        str(scaled_img),
        None,
        1,
        str(out_fcls_scaled),
        str(scaled_em),
        calc_stats=False,
        use_pbar=False,
    )
    with rasterio.open(out_fcls_scaled) as src:
        fcls_scaled = src.read()
    np.testing.assert_allclose(fcls_scaled, fcls_arr, atol=1e-5)
