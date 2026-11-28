"""
Tests for :mod:`rsgispy.imagecalc.comparison`: image differences, equality,
correlation, mutual information, earth mover's distance and similarity
metrics.
"""

import numpy as np
import pytest
import rasterio

import rsgispy
import rsgispy.imagecalc.comparison as ic_cmp

from conftest import write_raster


def _write_raster(
    path,
    data,
    origin=(0.0, 100.0),
    pixel_size=(10.0, 10.0),
    crs="EPSG:32630",
    dtype="float32",
    nodata=None,
):
    """Write a 3D ``(bands, height, width)`` NumPy array to a GeoTIFF."""
    return write_raster(
        path,
        data,
        dtype=dtype,
        nodata=nodata,
        crs=crs,
        origin=origin,
        pixel_size=pixel_size,
    )


def test_comparison_functions(tmp_path):
    """Test difference, equality, correlation, mutual info, and EMD."""
    a_p = tmp_path / "a.tif"
    b_p = tmp_path / "b.tif"
    diff_p = tmp_path / "diff.tif"

    x_vals = np.arange(1, 26, dtype=np.float32).reshape(1, 5, 5)
    y_vals = (x_vals * 2.0 + 1.0).astype(np.float32)
    y_vals[0, 4, 4] = -9999.0

    _write_raster(a_p, x_vals, nodata=-9999.0)
    _write_raster(b_p, y_vals, nodata=-9999.0)

    ic_cmp.calc_img_difference(str(a_p), str(b_p), str(diff_p), use_pbar=False)
    with rasterio.open(diff_p) as src:
        np.testing.assert_allclose(src.read(1)[0, 0], 1.0 - 3.0)

    eq, prop = ic_cmp.are_imgs_equal(str(a_p), str(a_p), use_pbar=False)
    assert eq is True and prop == pytest.approx(1.0)
    # flt_dif=0 tests for identical values.
    eq_0, prop_0 = ic_cmp.are_imgs_equal(str(a_p), str(a_p), flt_dif=0.0, use_pbar=False)
    assert eq_0 is True and prop_0 == pytest.approx(1.0)
    eq_ab, _ = ic_cmp.are_imgs_equal(str(a_p), str(b_p), flt_dif=0.0, use_pbar=False)
    assert eq_ab is False

    eq_b, prop_b = ic_cmp.are_img_bands_equal(
        str(a_p), 1, str(b_p), 1, prop_eql=0.5, flt_dif=100.0, use_pbar=False
    )
    assert eq_b is True and prop_b < 1.0

    # Uint8 underflow regression check (2 vs 5 within flt_dif=4.0)
    u1_p = tmp_path / "u1.tif"
    u2_p = tmp_path / "u2.tif"
    _write_raster(u1_p, np.full((1, 3, 3), 2, dtype=np.uint8), dtype="uint8")
    _write_raster(u2_p, np.full((1, 3, 3), 5, dtype=np.uint8), dtype="uint8")
    u_eq, u_prop = ic_cmp.are_imgs_equal(
        str(u1_p), str(u2_p), flt_dif=4.0, use_pbar=False
    )
    assert u_eq is True and u_prop == pytest.approx(1.0)
    with pytest.raises(rsgispy.RSGISPyException):
        ic_cmp._calc_imgs_equal(str(u1_p), str(u2_p), img_ref_band=1, img_cmp_band=None)

    corr, pval = ic_cmp.calc_img_correlation(
        str(a_p),
        str(b_p),
        1,
        1,
        corr_stat_method=rsgispy.STATS_CORR_PEARSONS,
    )
    assert corr == pytest.approx(1.0, abs=1e-6)
    assert pval < 1e-6

    corr_sp, _ = ic_cmp.calc_img_correlation(
        str(a_p), str(b_p), 1, 1, corr_stat_method=rsgispy.STATS_CORR_SPEARMAN
    )
    assert corr_sp == pytest.approx(1.0, abs=1e-6)

    mi_val = ic_cmp.calc_img_mutual_info(
        str(a_p), str(b_p), 1, 1, hist_bins=10, img_a_min=0, img_a_max=30, img_b_min=0, img_b_max=60
    )
    assert mi_val is not None and mi_val > 0.0

    emd_same = ic_cmp.calc_img_earth_move_dist(str(a_p), str(a_p), 1, 1)
    assert emd_same == pytest.approx(0.0)
    emd_diff = ic_cmp.calc_img_earth_move_dist(
        str(a_p), str(b_p), 1, 1, hist_a_bins=10, hist_b_bins=10, use_glb_range=True
    )
    assert emd_diff is not None and emd_diff >= 0.0


def test_calc_img_similarity_metrics(tmp_path):
    """Test calc_img_similarity_metrics with default/custom metrics, bands, and partial overlap."""
    img_a = tmp_path / "sim_a.tif"
    img_b = tmp_path / "sim_b.tif"
    img_b_shifted = tmp_path / "sim_b_partial.tif"

    rng = np.random.default_rng(21)
    band1 = (np.arange(1, 37, dtype=np.float32).reshape(6, 6) + rng.normal(0, 0.1, (6, 6))).astype(
        np.float32
    )
    band2 = (band1 * 2.0 + 5.0).astype(np.float32)
    arr_a = np.stack([band1, band2], axis=0)
    arr_b = np.stack([band1 * 3.0 + 2.0, band2 + 4.0], axis=0)
    arr_b[0, 0, 0] = -9999.0

    _write_raster(img_a, arr_a, origin=(0.0, 60.0), pixel_size=(10.0, 10.0), nodata=-9999.0)
    _write_raster(img_b, arr_b, origin=(0.0, 60.0), pixel_size=(10.0, 10.0), nodata=-9999.0)
    _write_raster(
        img_b_shifted,
        arr_b[:, :4, :4],
        origin=(20.0, 40.0),
        pixel_size=(10.0, 10.0),
        nodata=-9999.0,
    )

    # Default metric suite across both matching bands
    default_scores = ic_cmp.calc_img_similarity_metrics(str(img_a), str(img_b))
    assert set(default_scores.keys()) == {
        "correlation",
        "euclidean",
        "sqdiff",
        "manhattan",
        "mutual_info",
        "norm_mutual_info",
    }
    assert default_scores["correlation"] > 0.999

    # Custom subset of metrics mixing constants and strings on a single band pair
    custom_scores = ic_cmp.calc_img_similarity_metrics(
        str(img_a),
        str(img_b),
        metrics=[
            rsgispy.METRIC_CORRELATION,
            "manhattan",
            rsgispy.METRIC_SQDIFF,
            rsgispy.METRIC_PHASE_CORR,
        ],
        img_a_bands=2,
        img_b_bands=[2],
        hist_bins=12,
    )
    assert list(custom_scores.keys()) == [
        "correlation",
        "manhattan",
        "sqdiff",
        "phase_correlation",
    ]
    assert custom_scores["correlation"] == pytest.approx(1.0, abs=1e-5)
    assert custom_scores["manhattan"] == pytest.approx(4.0, abs=1e-4)
    assert custom_scores["sqdiff"] == pytest.approx(16.0, abs=1e-3)
    assert custom_scores["phase_correlation"] > 0.3

    # Partial overlap check
    part_scores = ic_cmp.calc_img_similarity_metrics(
        str(img_a),
        str(img_b_shifted),
        metrics=rsgispy.METRIC_EUCLIDEAN,
        img_a_bands=[2],
        img_b_bands=[2],
    )
    assert part_scores["euclidean"] >= 0.0

    # Invalid metrics list raises
    with pytest.raises(rsgispy.RSGISPyException):
        ic_cmp.calc_img_similarity_metrics(str(img_a), str(img_b), metrics=[])
    with pytest.raises(rsgispy.RSGISPyException):
        ic_cmp.calc_img_similarity_metrics(str(img_a), str(img_b), metrics=["corelation"])
