"""
Tests for :mod:`rsgispy.imagecalc.stats`: per-pixel stack summaries,
multi-image band statistics, zonal summaries in reference pixels, masked
probabilities, histograms and pixel counts.
"""


import numpy as np
import pytest
import rasterio

import rsgispy
import rsgispy.imagecalc.stats as ic_stats

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
        # Population standard deviation by default (as numpy and GDAL).
        np.testing.assert_allclose(
            out[6, 0, 0], np.std([1.0, 2.0, 10.0]), rtol=1e-5
        )
        # Pixel (0, 1): all nodata -> the output nodata, set on the output
        assert src.nodata == -9999.0
        np.testing.assert_allclose(out[:, 0, 1], np.full(7, -9999.0))

    # Without no_data_val the header value is used, and zeros are data.
    zero_img = tmp_path / "col_zero.tif"
    zarr = arr.copy()
    zarr[0, 1, 0] = 0.0
    _write_raster(zero_img, zarr, nodata=-9999.0)
    ic_stats.image_pixel_column_summary(
        str(zero_img), str(tmp_path / "col_hdr.tif"),
        ic_stats.StatsSummary(calc_min=True, calc_mean=True), use_pbar=False,
    )
    with rasterio.open(tmp_path / "col_hdr.tif") as src:
        assert src.nodata == -9999.0
        hdr = src.read()
    np.testing.assert_allclose(hdr[:, 1, 0], [0.0, 7.0 / 3.0], rtol=1e-6)
    np.testing.assert_allclose(hdr[:, 0, 1], [-9999.0, -9999.0])

    # ddof=1 gives the sample standard deviation.
    ic_stats.image_pixel_column_summary(
        str(in_img), str(tmp_path / "sample.tif"), ss, datatype=rsgispy.TYPE_32FLOAT,
        no_data_val=-9999.0, use_no_data=True, use_pbar=False, ddof=1,
    )
    with rasterio.open(tmp_path / "sample.tif") as src:
        np.testing.assert_allclose(
            src.read(7)[0, 0], np.std([1.0, 2.0, 10.0], ddof=1), rtol=1e-5
        )

    with pytest.raises(rsgispy.RSGISPyException):
        ic_stats.image_pixel_column_summary(
            str(in_img),
            str(tmp_path / "bad.tif"),
            ic_stats.StatsSummary(),
            use_pbar=False,
        )
    with pytest.raises(rsgispy.RSGISPyException):
        ic_stats.image_pixel_column_summary(
            str(in_img), str(tmp_path / "bad_ddof.tif"), ss, use_pbar=False, ddof=-1
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
        rsgispy.SUMTYPE_STDDEV: float(np.std([1.0, 3.0, 5.0])),
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

    out_sd1 = tmp_path / "out_sd_ddof1.tif"
    ic_stats.calc_multi_img_band_stats(
        [str(p1), str(p2), str(p3)], str(out_sd1), rsgispy.SUMTYPE_STDDEV,
        no_data_val=-99.0, use_no_data=True, use_pbar=False, ddof=1,
    )
    with rasterio.open(out_sd1) as src:
        np.testing.assert_allclose(src.read(1)[0, 0], 2.0, rtol=1e-5)

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

    for ddof in (0, 1):
        sd_p = tmp_path / f"inpxl_sd{ddof}.tif"
        ic_stats.get_img_sum_stats_in_pxl(
            str(ref_p), str(stats_p), str(sd_p), sum_stats=[rsgispy.SUMTYPE_STDDEV],
            io_grid_x=1, io_grid_y=1, use_pbar=False, ddof=ddof,
        )
        with rasterio.open(sd_p) as src:
            np.testing.assert_allclose(
                src.read(1)[0, 0], np.std([1.0, 2.0, 3.0, 4.0], ddof=ddof), rtol=1e-5
            )

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
        str(msk_p),
        str(out_p),
        [1, 2],
        1,
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
        str(const_p), str(msk_p), str(out_const), [1], 1, use_pbar=False
    )
    with rasterio.open(out_const) as src:
        np.testing.assert_allclose(src.read(1), np.ones((2, 3), dtype=np.float32))


def test_get_img_histogram_continuous_options(tmp_path):
    """Test get_img_histogram with n_bins, bin_size, auto bins, val_range, and mask."""
    import rsgispy.imagecalc as imagecalc

    assert imagecalc.get_img_histogram is ic_stats.get_img_histogram

    in_img = tmp_path / "cont.tif"
    msk_img = tmp_path / "msk.tif"
    arr = np.array(
        [
            [
                [0.1, 0.2, 0.4, 0.9],
                [0.15, -9999.0, 0.6, 1.0],
                [0.35, 0.45, 0.75, np.nan],
            ]
        ],
        dtype=np.float32,
    )
    msk = np.array(
        [
            [
                [1, 1, 1, 1],
                [1, 1, 0, 1],
                [1, 1, 1, 1],
            ]
        ],
        dtype=np.uint8,
    )
    _write_raster(in_img, arr, dtype="float32", nodata=-9999.0)
    _write_raster(msk_img, msk, dtype="uint8", nodata=0)

    # Fixed bin_size with explicit val_range
    h_bs = ic_stats.get_img_histogram(
        in_img,
        img_band=1,
        bin_size=0.25,
        val_range=(0.0, 1.0),
        use_pbar=False,
    )
    assert h_bs["n_bins"] == 4
    assert h_bs["bin_width"] == pytest.approx(0.25)
    assert h_bs["total_count"] == 10
    assert h_bs["bin_edges"] == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])
    assert h_bs["bin_centres"] == pytest.approx([0.125, 0.375, 0.625, 0.875])
    assert h_bs["counts"].tolist() == [3, 3, 1, 3]
    assert np.sum(h_bs["proportions"]) == pytest.approx(1.0)
    assert np.sum(h_bs["density"] * h_bs["bin_widths"]) == pytest.approx(1.0)
    assert len(h_bs["bins"]) == 4
    assert h_bs["bins"][0]["count"] == 3
    assert h_bs["bins"][0]["lower_edge"] == pytest.approx(0.0)
    assert h_bs["bins"][0]["upper_edge"] == pytest.approx(0.25)

    # Fixed n_bins + mask image (excludes pixel with value 0.6)
    h_nb = ic_stats.get_img_histogram(
        in_img,
        img_band=1,
        n_bins=2,
        val_range=(0.0, 1.0),
        in_msk_img=msk_img,
        img_msk_val=1,
        use_pbar=False,
    )
    assert h_nb["n_bins"] == 2
    assert h_nb["total_count"] == 9
    assert h_nb["counts"].tolist() == [6, 3]

    # Auto Scott binning without explicit n_bins / bin_size
    h_auto = ic_stats.get_img_histogram(in_img, use_pbar=False)
    assert h_auto["n_bins"] >= 1
    assert h_auto["total_count"] == 10
    assert h_auto["counts"].sum() == 10


def test_get_img_histogram_integer_bins_and_errors(tmp_path):
    """Test get_img_histogram integer_bins=True on categorical rasters and error cases."""
    cat_img = tmp_path / "cat.tif"
    arr = np.array(
        [
            [
                [1, 1, 2, 0],
                [4, 2, 1, 0],
                [4, 4, 2, 1],
            ]
        ],
        dtype=np.uint8,
    )
    _write_raster(cat_img, arr, dtype="uint8", nodata=0)

    # Integer bins with nodata=0 excluded: values 1, 2, 4 -> bins [1, 2, 3, 4]
    h_cat = ic_stats.get_img_histogram(
        cat_img,
        img_band=1,
        integer_bins=True,
        use_pbar=False,
    )
    assert h_cat["integer_bins"] is True
    assert h_cat["n_bins"] == 4
    assert h_cat["bin_width"] == pytest.approx(1.0)
    assert h_cat["min_val"] == 1
    assert h_cat["max_val"] == 4
    assert h_cat["bin_values"].tolist() == [1, 2, 3, 4]
    assert h_cat["bin_centres"] == pytest.approx([1.0, 2.0, 3.0, 4.0])
    assert h_cat["bin_edges"] == pytest.approx([0.5, 1.5, 2.5, 3.5, 4.5])
    assert h_cat["counts"].tolist() == [4, 3, 0, 3]
    assert h_cat["total_count"] == 10
    assert [b["value"] for b in h_cat["bins"]] == [1, 2, 3, 4]
    assert [b["count"] for b in h_cat["bins"]] == [4, 3, 0, 3]

    # Constant float raster
    const_img = tmp_path / "const.tif"
    _write_raster(const_img, np.full((1, 3, 3), 5.0, dtype=np.float32))
    h_const = ic_stats.get_img_histogram(const_img, n_bins=3, use_pbar=False)
    assert h_const["n_bins"] == 3
    assert h_const["total_count"] == 9
    assert h_const["counts"].sum() == 9

    # Non-integer float raster with integer_bins=True raises error
    flt_img = tmp_path / "flt.tif"
    _write_raster(flt_img, np.array([[[1.2, 2.5], [3.0, 4.0]]], dtype=np.float32))
    with pytest.raises(rsgispy.RSGISPyException, match="non-integer"):
        ic_stats.get_img_histogram(flt_img, integer_bins=True, use_pbar=False)

    # Validation errors
    with pytest.raises(rsgispy.RSGISPyException):
        ic_stats.get_img_histogram(cat_img, n_bins=5, bin_size=1.0)
    with pytest.raises(rsgispy.RSGISPyException):
        ic_stats.get_img_histogram(cat_img, integer_bins=True, n_bins=4)
    with pytest.raises(rsgispy.RSGISPyException):
        ic_stats.get_img_histogram(cat_img, n_bins=0)
    with pytest.raises(rsgispy.RSGISPyException):
        ic_stats.get_img_histogram(cat_img, bin_size=-1.0)
    with pytest.raises(rsgispy.RSGISPyException):
        ic_stats.get_img_histogram(cat_img, val_range=(5.0, 1.0))


def test_stack_stats_write_nodata_where_no_input_is_valid(tmp_path):
    """H14: pixels with no valid input get the output nodata, which is set."""
    nd = -99.0
    p1 = tmp_path / "n1.tif"
    p2 = tmp_path / "n2.tif"
    _write_raster(p1, np.array([[[nd, 2.0], [0.0, 4.0]]]), nodata=nd)
    _write_raster(p2, np.array([[[nd, 6.0], [0.0, nd]]]), nodata=nd)

    out_p = tmp_path / "mean.tif"
    ic_stats.calc_multi_img_band_stats(
        [str(p1), str(p2)], str(out_p), rsgispy.SUMTYPE_MEAN,
        use_no_data=True, use_pbar=False,
    )
    with rasterio.open(out_p) as src:
        assert src.nodata == nd
        np.testing.assert_allclose(src.read(1), [[nd, 4.0], [0.0, 4.0]])

    # An integer output that cannot hold -99 uses the type's maximum.
    out_u8 = tmp_path / "max_u8.tif"
    ic_stats.calc_multi_img_band_stats(
        [str(p1), str(p2)], str(out_u8), rsgispy.SUMTYPE_MAX,
        use_no_data=True, datatype=rsgispy.TYPE_8UINT, use_pbar=False,
    )
    with rasterio.open(out_u8) as src:
        assert src.nodata == 255
        np.testing.assert_array_equal(src.read(1), [[255, 6], [0, 4]])

    stack_p = tmp_path / "stack.tif"
    _write_raster(stack_p, np.array([[[nd, 1.0]], [[nd, 3.0]]]), nodata=nd)
    col_p = tmp_path / "col.tif"
    ic_stats.image_pixel_column_summary(
        str(stack_p), str(col_p), ic_stats.StatsSummary(calc_mean=True),
        no_data_val=nd, use_pbar=False,
    )
    with rasterio.open(col_p) as src:
        assert src.nodata == nd
        np.testing.assert_allclose(src.read(1), [[nd, 2.0]])


def test_count_pxls_of_val_single_band(tmp_path):
    b1 = np.array([[1, 1, 2], [2, 3, 3]], dtype=np.uint8)
    b2 = np.array([[3, 3, 3], [3, 1, 1]], dtype=np.uint8)
    in_img = write_raster(tmp_path / "cnt.tif", np.stack([b1, b2], axis=0), origin=(500000.0, 5800000.0))

    c1 = ic_stats.count_pxls_of_val(in_img, [1, 2, 3], img_band=1, use_pbar=False)
    c2 = ic_stats.count_pxls_of_val(in_img, [1, 2, 3], img_band=2, use_pbar=False)
    call = ic_stats.count_pxls_of_val(
        in_img, [1, 2, 3], img_band=None, use_pbar=False
    )
    np.testing.assert_array_equal(c1, [2, 2, 2])
    np.testing.assert_array_equal(c2, [2, 0, 4])
    np.testing.assert_array_equal(call, [4, 2, 6])
