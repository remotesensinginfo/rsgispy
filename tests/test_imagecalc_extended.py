"""
Tests for extended rsgispy.imagecalc modules: stats, comparison, encoding,
rescale, distance, fill, and maths.calc_area_under_curve.
"""

import importlib
import inspect
import sys

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

import rsgispy
import rsgispy.imagecalc.comparison as ic_cmp
import rsgispy.imagecalc.distance as ic_dist
import rsgispy.imagecalc.encoding as ic_enc
import rsgispy.imagecalc.fill as ic_fill
import rsgispy.imagecalc.maths as ic_maths
import rsgispy.imagecalc.rescale as ic_resc
import rsgispy.imagecalc.stats as ic_stats


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
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    count, height, width = data.shape
    transform = from_origin(origin[0], origin[1], pixel_size[0], pixel_size[1])
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data.astype(dtype))


def test_default_gdalformat_is_gtiff():
    """Verify all newly imported functions default gdalformat to 'GTIFF'."""
    funcs = [
        ic_stats.image_pixel_column_summary,
        ic_stats.calc_multi_img_band_stats,
        ic_stats.get_img_sum_stats_in_pxl,
        ic_stats.calc_mask_img_pxl_val_prob,
        ic_cmp.calc_img_difference,
        ic_maths.calc_area_under_curve,
        ic_enc.recode_int_raster,
        ic_enc.one_hot_category_encoding,
        ic_resc.calc_img_rescale,
        ic_resc.rescale_img_pxl_vals,
        ic_dist.calc_dist_to_img_vals,
        ic_dist.calc_dist_to_img_vals_tiled,
        ic_dist.buffer_img_pxl_vals,
        ic_fill.calc_fill_regions_knn,
    ]
    for fn in funcs:
        sig = inspect.signature(fn)
        assert sig.parameters["gdalformat"].default == "GTIFF"


def test_image_pixel_column_summary(tmp_path):
    """Test StatsSummary and image_pixel_column_summary canonical bands and nodata."""
    in_img = tmp_path / "col.tif"
    out_img = tmp_path / "col_sum.tif"
    arr = np.array(
        [
            [[1.0, -9999.0], [2.0, 4.0]],
            [[2.0, -9999.0], [2.0, 6.0]],
            [[10.0, -9999.0], [5.0, -9999.0]],
        ],
        dtype=np.float32,
    )
    _write_raster(in_img, arr, nodata=-9999.0)

    ss = ic_stats.StatsSummary(
        calc_min=True,
        calc_max=True,
        calc_sum=True,
        calc_mean=True,
        calc_stdev=True,
        calc_median=True,
        calc_mode=True,
    )
    ic_stats.image_pixel_column_summary(
        str(in_img),
        str(out_img),
        ss,
        datatype=rsgispy.TYPE_32FLOAT,
        no_data_val=-9999.0,
        use_no_data=True,
        use_pbar=False,
    )

    with rasterio.open(out_img) as src:
        assert src.descriptions == (
            "Min",
            "Max",
            "Mean",
            "Median",
            "Mode",
            "Sum",
            "StdDev",
        )
        out = src.read()
        # Pixel (0, 0): [1, 2, 10]
        np.testing.assert_allclose(out[0, 0, 0], 1.0)
        np.testing.assert_allclose(out[1, 0, 0], 10.0)
        np.testing.assert_allclose(out[2, 0, 0], 13.0 / 3.0, rtol=1e-5)
        np.testing.assert_allclose(out[3, 0, 0], 2.0)
        np.testing.assert_allclose(out[5, 0, 0], 13.0)
        np.testing.assert_allclose(
            out[6, 0, 0], np.std([1.0, 2.0, 10.0], ddof=1), rtol=1e-5
        )
        # Pixel (0, 1): all nodata -> 0.0
        np.testing.assert_allclose(out[:, 0, 1], np.zeros(7))

    with pytest.raises(rsgispy.RSGISPyException):
        ic_stats.image_pixel_column_summary(
            str(in_img),
            str(tmp_path / "bad.tif"),
            ic_stats.StatsSummary(),
            use_pbar=False,
        )


def test_calc_multi_img_band_stats(tmp_path):
    """Test multi-image per-band statistics across all SUMTYPE_* options."""
    p1 = tmp_path / "m1.tif"
    p2 = tmp_path / "m2.tif"
    p3 = tmp_path / "m3.tif"

    _write_raster(
        p1,
        np.array([[[1.0, -99.0], [3.0, 4.0]], [[10.0, 20.0], [30.0, 40.0]]]),
        nodata=-99.0,
    )
    _write_raster(
        p2,
        np.array([[[3.0, 5.0], [3.0, 8.0]], [[20.0, 20.0], [30.0, 40.0]]]),
        nodata=-99.0,
    )
    _write_raster(
        p3,
        np.array([[[5.0, 5.0], [9.0, -99.0]], [[30.0, 20.0], [30.0, 40.0]]]),
        nodata=-99.0,
    )

    expected_b1_00 = {
        rsgispy.SUMTYPE_MIN: 1.0,
        rsgispy.SUMTYPE_MAX: 5.0,
        rsgispy.SUMTYPE_MEAN: 3.0,
        rsgispy.SUMTYPE_MEDIAN: 3.0,
        rsgispy.SUMTYPE_SUM: 9.0,
        rsgispy.SUMTYPE_COUNT: 3.0,
        rsgispy.SUMTYPE_RANGE: 4.0,
        rsgispy.SUMTYPE_STDDEV: 2.0,
    }
    for stat_id, exp_val in expected_b1_00.items():
        out_p = tmp_path / f"out_{stat_id}.tif"
        ic_stats.calc_multi_img_band_stats(
            [str(p1), str(p2), str(p3)],
            str(out_p),
            stat_id,
            no_data_val=-99.0,
            use_no_data=True,
            use_pbar=False,
        )
        with rasterio.open(out_p) as src:
            assert src.count == 2
            np.testing.assert_allclose(src.read(1)[0, 0], exp_val, rtol=1e-5)

    # Mode at (0, 1): [-99, 5, 5] -> 5.0, Count -> 2.0
    out_mode = tmp_path / "out_mode.tif"
    ic_stats.calc_multi_img_band_stats(
        [str(p1), str(p2), str(p3)],
        str(out_mode),
        rsgispy.SUMTYPE_MODE,
        no_data_val=-99.0,
        use_no_data=True,
        use_pbar=False,
    )
    with rasterio.open(out_mode) as src:
        np.testing.assert_allclose(src.read(1)[0, 1], 5.0)


def test_get_img_sum_stats_in_pxl(tmp_path):
    """Test high-resolution to reference pixel zonal summary aggregation."""
    ref_p = tmp_path / "ref.tif"
    stats_p = tmp_path / "stats.tif"
    out_p = tmp_path / "inpxl.tif"

    _write_raster(
        ref_p,
        np.ones((1, 2, 2), dtype=np.uint8),
        origin=(0.0, 40.0),
        pixel_size=(20.0, 20.0),
        dtype="uint8",
    )
    stats_arr = np.array(
        [
            [
                [1.0, 2.0, 10.0, 20.0],
                [3.0, 4.0, 30.0, -99.0],
                [5.0, 5.0, 0.0, 0.0],
                [5.0, 9.0, 0.0, 0.0],
            ]
        ],
        dtype=np.float32,
    )
    _write_raster(
        stats_p,
        stats_arr,
        origin=(0.0, 40.0),
        pixel_size=(10.0, 10.0),
        dtype="float32",
        nodata=-99.0,
    )

    ic_stats.get_img_sum_stats_in_pxl(
        str(ref_p),
        str(stats_p),
        str(out_p),
        sum_stats=[
            rsgispy.SUMTYPE_MEAN,
            rsgispy.SUMTYPE_MAX,
            rsgispy.SUMTYPE_MEDIAN,
            rsgispy.SUMTYPE_COUNT,
        ],
        io_grid_x=1,
        io_grid_y=1,
        use_pbar=False,
    )

    with rasterio.open(out_p) as src:
        assert src.count == 4
        out = src.read()
        np.testing.assert_allclose(out[0, 0, 0], 2.5)
        np.testing.assert_allclose(out[0, 0, 1], 20.0)
        np.testing.assert_allclose(out[1, 0, 1], 30.0)
        np.testing.assert_allclose(out[2, 1, 0], 5.0)
        np.testing.assert_allclose(out[3, 0, 1], 3.0)

    # Equal resolution must raise
    with pytest.raises(rsgispy.RSGISPyException):
        ic_stats.get_img_sum_stats_in_pxl(
            str(stats_p),
            str(stats_p),
            str(tmp_path / "err.tif"),
            use_pbar=False,
        )


def test_calc_mask_img_pxl_val_prob(tmp_path):
    """Test joint probability calculation from masked region."""
    img_p = tmp_path / "prob_in.tif"
    msk_p = tmp_path / "prob_msk.tif"
    out_p = tmp_path / "prob_out.tif"

    img_arr = np.array(
        [
            [[1.0, 1.0, 5.0], [1.0, 5.0, 9.0]],
            [[2.0, 2.0, 6.0], [2.0, 6.0, 10.0]],
        ],
        dtype=np.float32,
    )
    msk_arr = np.array([[[1, 1, 1], [1, 0, 0]]], dtype=np.uint8)
    _write_raster(img_p, img_arr, nodata=-9999.0)
    _write_raster(msk_p, msk_arr, dtype="uint8", nodata=0)

    ic_stats.calc_mask_img_pxl_val_prob(
        str(img_p),
        [1, 2],
        str(msk_p),
        1,
        str(out_p),
        bin_widths=[2.0, 2.0],
        use_no_data=True,
        rescale_probs=True,
        use_pbar=False,
    )
    with rasterio.open(out_p) as src:
        assert src.count == 1
        out = src.read(1)
        # Inside mask: three (1, 2) vs one (5, 6) -> rescaled 1.0 and 1/3
        np.testing.assert_allclose(out[0, 0], 1.0)
        np.testing.assert_allclose(out[0, 2], 1.0 / 3.0, rtol=1e-5)
        # Outside sample range (9, 10) -> 0.0
        np.testing.assert_allclose(out[1, 2], 0.0)

    # Auto bin widths on constant image must not crash/segfault
    const_p = tmp_path / "const.tif"
    _write_raster(const_p, np.ones((1, 2, 3), dtype=np.float32), nodata=-9999.0)
    out_const = tmp_path / "prob_const.tif"
    ic_stats.calc_mask_img_pxl_val_prob(
        str(const_p), [1], str(msk_p), 1, str(out_const), use_pbar=False
    )
    with rasterio.open(out_const) as src:
        np.testing.assert_allclose(src.read(1), np.ones((2, 3), dtype=np.float32))


def test_recode_and_one_hot(tmp_path):
    """Test recode_int_raster and one_hot_category_encoding."""
    in_p = tmp_path / "cats.tif"
    rec_keep = tmp_path / "rec_keep.tif"
    rec_drop = tmp_path / "rec_drop.tif"
    oh_auto = tmp_path / "oh_auto.tif"
    oh_dict = tmp_path / "oh_dict.tif"

    cats = np.array([[[1, 2, 3], [2, 99, 1]]], dtype=np.int32)
    _write_raster(in_p, cats, dtype="int32", nodata=99)

    ic_enc.recode_int_raster(
        str(in_p),
        str(rec_keep),
        {1: 10, 2: 20},
        keep_vals_not_in_dict=True,
        datatype=rsgispy.TYPE_16UINT,
        use_pbar=False,
    )
    with rasterio.open(rec_keep) as src:
        np.testing.assert_array_equal(
            src.read(1), np.array([[10, 20, 3], [20, 99, 10]], dtype=np.uint16)
        )

    ic_enc.recode_int_raster(
        str(in_p),
        str(rec_drop),
        {1: 10},
        keep_vals_not_in_dict=False,
        datatype=rsgispy.TYPE_8UINT,
        use_pbar=False,
    )
    with rasterio.open(rec_drop) as src:
        np.testing.assert_array_equal(
            src.read(1), np.array([[10, 0, 0], [0, 0, 10]], dtype=np.uint8)
        )

    # Auto one-hot excluding nodata=99 -> values [1, 2, 3] => 3 bands
    ic_enc.one_hot_category_encoding(
        str(in_p), str(oh_auto), no_data_val=99, use_pbar=False
    )
    with rasterio.open(oh_auto) as src:
        assert src.count == 3
        assert src.dtypes[0] == "uint8"
        out = src.read()
        np.testing.assert_array_equal(out[0], np.array([[1, 0, 0], [0, 0, 1]]))
        np.testing.assert_array_equal(out[1], np.array([[0, 1, 0], [1, 0, 0]]))
        np.testing.assert_array_equal(out[2], np.array([[0, 0, 1], [0, 0, 0]]))

    # Explicit recode_dict mapping and invalid band 0 guard
    ic_enc.one_hot_category_encoding(
        str(in_p), str(oh_dict), recode_dict={3: 1, 1: 2}, use_pbar=False
    )
    with rasterio.open(oh_dict) as src:
        assert src.count == 2
        out = src.read()
        np.testing.assert_array_equal(out[0], np.array([[0, 0, 1], [0, 0, 0]]))
        np.testing.assert_array_equal(out[1], np.array([[1, 0, 0], [0, 0, 1]]))

    with pytest.raises(rsgispy.RSGISPyException):
        ic_enc.one_hot_category_encoding(
            str(in_p), str(tmp_path / "bad_oh.tif"), recode_dict={1: 0}, use_pbar=False
        )


def test_rescale_functions(tmp_path):
    """Test calc_img_rescale, ImageBandRescale, and rescale_img_pxl_vals."""
    p1 = tmp_path / "r1.tif"
    p2 = tmp_path / "r2.tif"
    out_gain = tmp_path / "out_gain.tif"
    out_minmax = tmp_path / "out_minmax.tif"

    _write_raster(
        p1,
        np.array([[[1.0, 2.0], [5.0, -9999.0]]], dtype=np.float32),
        nodata=-9999.0,
    )
    _write_raster(
        p2,
        np.array([[[5.0, 8.0], [11.0, -9999.0]]], dtype=np.float32),
        nodata=-9999.0,
    )

    ic_resc.calc_img_rescale(
        [str(p1), str(p2)],
        str(out_gain),
        datatype=rsgispy.TYPE_32FLOAT,
        c_no_data_val=-9999.0,
        c_offset=5.0,
        c_gain=3.0,
        n_no_data_val=-1.0,
        n_offset=11.0,
        n_gain=7.0,
        use_pbar=False,
    )
    with rasterio.open(out_gain) as src:
        assert src.count == 2
        assert src.nodata == -1.0
        np.testing.assert_allclose(
            src.read(1),
            np.array([[1.6666666, 4.0], [11.0, -1.0]], dtype=np.float32),
            rtol=1e-5,
        )

    cfg = ic_resc.ImageBandRescale(
        band=1,
        in_min=0.0,
        in_max=10.0,
        no_data_val=-9999.0,
        out_min=0.0,
        out_max=100.0,
        out_no_data=-99.0,
    )
    assert "Band 1" in str(cfg) and "Band 1" in repr(cfg)
    ic_resc.rescale_img_pxl_vals(
        str(p1),
        str(out_minmax),
        datatype=rsgispy.TYPE_32FLOAT,
        band_rescale_objs=[cfg],
        trim_to_limits=True,
        use_pbar=False,
    )
    with rasterio.open(out_minmax) as src:
        np.testing.assert_allclose(
            src.read(1),
            np.array([[10.0, 20.0], [50.0, -99.0]], dtype=np.float32),
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


def test_calc_area_under_curve(tmp_path):
    """Test calc_area_under_curve weighted band sum."""
    in_p = tmp_path / "auc_in.tif"
    out_p = tmp_path / "auc_out.tif"
    arr = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[2.0, 2.0], [2.0, 2.0]],
        ],
        dtype=np.float32,
    )
    _write_raster(in_p, arr)

    ic_maths.calc_area_under_curve(
        str(in_p), [0.5, 2.0], str(out_p), use_pbar=False
    )
    with rasterio.open(out_p) as src:
        np.testing.assert_allclose(
            src.read(1), np.array([[4.5, 5.0], [5.5, 6.0]], dtype=np.float32)
        )


def test_distance_and_buffer_functions(tmp_path):
    """Test calc_dist_to_img_vals, calc_dist_to_img_vals_tiled, and buffer_img_pxl_vals."""
    in_p = tmp_path / "dist_in.tif"
    d_full = tmp_path / "d_full.tif"
    d_tiled = tmp_path / "d_tiled.tif"
    buf_p = tmp_path / "buf.tif"

    arr = np.zeros((1, 8, 8), dtype=np.int16)
    arr[0, 0, 0] = 1
    arr[0, 5, 5] = 2
    arr[0, 7, 7] = -1
    _write_raster(in_p, arr, origin=(0.0, 80.0), pixel_size=(10.0, 10.0), dtype="int16", nodata=-1)

    ic_dist.calc_dist_to_img_vals(
        str(in_p),
        str(d_full),
        pxl_vals=[1, 2],
        max_dist=25.0,
        no_data_val=-1,
        out_no_data_val=-999.0,
        unit_geo=True,
    )
    ic_dist.calc_dist_to_img_vals_tiled(
        str(in_p),
        str(d_tiled),
        pxl_vals=[1, 2],
        max_dist=25.0,
        no_data_val=-1,
        out_no_data_val=-999.0,
        unit_geo=True,
        tile_size=4,
        n_cores=1,
    )
    with rasterio.open(d_full) as s1, rasterio.open(d_tiled) as s2:
        a1 = s1.read(1)
        a2 = s2.read(1)
        np.testing.assert_allclose(a1, a2)
        assert a1[0, 0] == pytest.approx(0.0)
        assert a1[0, 1] == pytest.approx(10.0)
        assert a1[1, 1] == pytest.approx(10.0 * np.sqrt(2.0), rel=1e-5)
        assert a1[7, 7] == pytest.approx(-999.0)

    # Also verify multi-core tiled execution produces identical result
    d_tiled_mc = tmp_path / "d_tiled_mc.tif"
    ic_dist.calc_dist_to_img_vals_tiled(
        str(in_p),
        str(d_tiled_mc),
        pxl_vals=[1, 2],
        max_dist=25.0,
        no_data_val=-1,
        out_no_data_val=-999.0,
        unit_geo=True,
        tile_size=4,
        n_cores=2,
    )
    with rasterio.open(d_full) as s1, rasterio.open(d_tiled_mc) as s3:
        np.testing.assert_allclose(s1.read(1), s3.read(1))

    # Empty target mask must return out_no_data_val everywhere (no corner artefact)
    d_empty = tmp_path / "d_empty.tif"
    ic_dist.calc_dist_to_img_vals(
        str(in_p),
        str(d_empty),
        pxl_vals=[999],
        max_dist=25.0,
        out_no_data_val=-999.0,
    )
    with rasterio.open(d_empty) as s_emp:
        np.testing.assert_allclose(s_emp.read(1), np.full((8, 8), -999.0, dtype=np.float32))

    # Buffer mask (< 15m around value 1 -> centre + 4-neighbours within 10m)
    ic_dist.buffer_img_pxl_vals(
        str(in_p),
        str(buf_p),
        pxl_vals=[1],
        buf_thres=15.0,
        tmp_dir=str(tmp_path),
        unit_geo=True,
    )
    with rasterio.open(buf_p) as s_buf:
        b = s_buf.read(1)
        assert b.dtype == np.uint8
        assert b[0, 0] == 1 and b[0, 1] == 1 and b[1, 1] == 1
        assert b[0, 2] == 0


def test_calc_fill_regions_knn(tmp_path):
    """Test KNN spatial region fill across summary metrics."""
    ref_p = tmp_path / "knn_ref.tif"
    mask_p = tmp_path / "knn_mask.tif"
    out_p = tmp_path / "knn_out.tif"

    ref_arr = np.array(
        [
            [
                [10, 10, -99, -99],
                [10, -99, -99, 30],
                [-99, -99, 30, 30],
                [-99, 30, 30, 30],
            ]
        ],
        dtype=np.int32,
    )
    fill_mask = np.array(
        [
            [
                [0, 0, 1, 1],
                [0, 1, 1, 0],
                [1, 1, 0, 0],
                [1, 0, 0, 0],
            ]
        ],
        dtype=np.int32,
    )
    _write_raster(ref_p, ref_arr, dtype="int32", nodata=-99)
    _write_raster(mask_p, fill_mask, dtype="int32")

    ic_fill.calc_fill_regions_knn(
        str(ref_p),
        -99,
        str(mask_p),
        1,
        str(out_p),
        k=3,
        summary=rsgispy.SUMTYPE_MODE,
        use_pbar=False,
    )
    with rasterio.open(out_p) as src:
        out = src.read(1)
        # Non-fill pixels remain 0; (1, 1) nearest 3 valid neighbours are all 10
        assert out[0, 0] == 0
        assert out[1, 1] == 10
        assert out[2, 1] == 30


def test_optional_scipy_sklearn_guards(monkeypatch):
    """Verify optional scipy/sklearn dependencies guard functions cleanly."""
    # Test stats pure-numpy mode fallback when HAVE_SCIPY is False
    monkeypatch.setattr(ic_stats, "HAVE_SCIPY", False)
    col_data = np.array([[1.0, 5.0], [2.0, 5.0], [2.0, np.nan]])
    modes = ic_stats._calc_mode_columns(col_data)  # pylint: disable=protected-access
    np.testing.assert_allclose(modes, np.array([2.0, 5.0]))

    # Simulate missing scipy and sklearn during fresh module import
    with monkeypatch.context() as ctx:
        for mod_name in (
            "scipy",
            "scipy.stats",
            "scipy.ndimage",
            "scipy.spatial",
            "sklearn",
            "sklearn.metrics",
        ):
            ctx.setitem(sys.modules, mod_name, None)

        ctx.setattr(rsgispy.imagecalc, "comparison", ic_cmp)
        ctx.setattr(rsgispy.imagecalc, "distance", ic_dist)
        ctx.setattr(rsgispy.imagecalc, "fill", ic_fill)

        for mod_name in (
            "rsgispy.imagecalc.comparison",
            "rsgispy.imagecalc.distance",
            "rsgispy.imagecalc.fill",
        ):
            ctx.delitem(sys.modules, mod_name, raising=False)

        cmp_fresh = importlib.import_module("rsgispy.imagecalc.comparison")
        dist_fresh = importlib.import_module("rsgispy.imagecalc.distance")
        fill_fresh = importlib.import_module("rsgispy.imagecalc.fill")

        assert cmp_fresh.HAVE_SCIPY is False
        assert cmp_fresh.HAVE_SKLEARN is False
        assert not hasattr(cmp_fresh, "calc_img_correlation")
        assert not hasattr(cmp_fresh, "calc_img_earth_move_dist")
        assert not hasattr(cmp_fresh, "calc_img_mutual_info")
        assert hasattr(cmp_fresh, "calc_img_difference")

        assert dist_fresh.HAVE_SCIPY is False
        assert not hasattr(dist_fresh, "calc_dist_to_img_vals")
        assert not hasattr(dist_fresh, "buffer_img_pxl_vals")

        assert fill_fresh.HAVE_SCIPY is False
        assert not hasattr(fill_fresh, "calc_fill_regions_knn")
