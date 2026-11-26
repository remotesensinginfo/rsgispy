"""
Tests for extended rsgispy.imagecalc modules: stats, comparison, encoding,
rescale, distance, and fill.
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
        ic_enc.recode_int_raster,
        ic_enc.one_hot_category_encoding,
        ic_resc.calc_img_rescale,
        ic_resc.rescale_img_pxl_vals,
        ic_dist.calc_dist_to_img_vals,
        ic_dist.calc_dist_to_img_vals_tiled,
        ic_dist.buffer_img_pxl_vals,
        ic_fill.calc_fill_regions_knn,
        ic_fill.region_growing_fill,
        ic_fill.region_growing_fill_tiled,
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


def test_rescale_img_pxl_vals_no_data(tmp_path):
    """rescale_img_pxl_vals keeps valid zeros and writes out_no_data."""
    # A distance raster: 0 m is valid, no header no data value.
    dist = np.array([[[0.0, 500.0], [1000.0, 2000.0]]], dtype=np.float32)
    dist_img = tmp_path / "dist.tif"
    _write_raster(dist_img, dist)

    out_img = tmp_path / "dist_memb.tif"
    ic_resc.rescale_img_pxl_vals(
        str(dist_img),
        str(out_img),
        [ic_resc.ImageBandRescale(band=1, in_min=0.0, in_max=1000.0,
                                  out_min=1.0, out_max=0.0)],
        trim_to_limits=True,
        use_pbar=False,
    )
    with rasterio.open(out_img) as src:
        # Default out_no_data for a float output is -9999.
        assert src.nodata == -9999.0
        np.testing.assert_allclose(src.read(1), [[1.0, 0.5], [0.0, 0.0]])

    # The input band's no data value is used when no_data_val is None.
    dist_nd_img = tmp_path / "dist_nd.tif"
    dist_nd = np.array([[[0.0, 500.0], [5000.0, 2000.0]]], dtype=np.float32)
    _write_raster(dist_nd_img, dist_nd, nodata=5000.0)
    out_nd_img = tmp_path / "dist_nd_memb.tif"
    ic_resc.rescale_img_pxl_vals(
        str(dist_nd_img),
        str(out_nd_img),
        [ic_resc.ImageBandRescale(band=1, in_min=0.0, in_max=1000.0,
                                  out_min=1.0, out_max=0.0, out_no_data=-1.0)],
        use_pbar=False,
    )
    with rasterio.open(out_nd_img) as src:
        assert src.nodata == -1.0
        np.testing.assert_allclose(src.read(1), [[1.0, 0.5], [-1.0, 0.0]])

    # An explicit no_data_val overrides the input band's value.
    ic_resc.rescale_img_pxl_vals(
        str(dist_nd_img),
        str(out_nd_img),
        [ic_resc.ImageBandRescale(band=1, in_min=0.0, in_max=1000.0,
                                  no_data_val=0.0, out_min=1.0, out_max=0.0,
                                  out_no_data=-1.0)],
        use_pbar=False,
    )
    with rasterio.open(out_nd_img) as src:
        np.testing.assert_allclose(src.read(1), [[-1.0, 0.5], [0.0, 0.0]])

    # Integer output: the default out_no_data must fit the data type.
    out_u8 = tmp_path / "dist_u8.tif"
    ic_resc.rescale_img_pxl_vals(
        str(dist_nd_img),
        str(out_u8),
        [ic_resc.ImageBandRescale(band=1, in_min=0.0, in_max=1000.0,
                                  out_min=0.0, out_max=100.0)],
        datatype=rsgispy.TYPE_8UINT,
        use_pbar=False,
    )
    with rasterio.open(out_u8) as src:
        assert src.nodata == 255.0
        np.testing.assert_array_equal(src.read(1), [[0, 50], [255, 100]])

    # An explicit out_no_data the output data type cannot hold is an error.
    with pytest.raises(rsgispy.RSGISPyException):
        ic_resc.rescale_img_pxl_vals(
            str(dist_img),
            str(tmp_path / "bad_u8.tif"),
            [ic_resc.ImageBandRescale(band=1, in_min=0.0, in_max=1000.0,
                                      out_min=0.0, out_max=100.0,
                                      out_no_data=-9999.0)],
            datatype=rsgispy.TYPE_8UINT,
            use_pbar=False,
        )

    # Different out_no_data values per band: kept per band where the format
    # supports it, an error for GeoTIFF (one no data value per file).
    two_img = tmp_path / "two.tif"
    _write_raster(two_img, np.concatenate([dist_nd, dist_nd]), nodata=5000.0)
    cfgs = [
        ic_resc.ImageBandRescale(band=1, in_min=0.0, in_max=1000.0,
                                 out_min=0.0, out_max=1.0, out_no_data=-1.0),
        ic_resc.ImageBandRescale(band=2, in_min=0.0, in_max=1000.0,
                                 out_min=0.0, out_max=1.0, out_no_data=-2.0),
    ]
    with pytest.raises(rsgispy.RSGISPyException):
        ic_resc.rescale_img_pxl_vals(
            str(two_img), str(tmp_path / "two_out.tif"), cfgs, use_pbar=False
        )
    if "KEA" in rasterio.drivers.raster_driver_extensions().values():
        two_kea = tmp_path / "two_out.kea"
        ic_resc.rescale_img_pxl_vals(
            str(two_img), str(two_kea), cfgs, gdalformat="KEA", use_pbar=False
        )
        with rasterio.open(two_kea) as src:
            assert src.nodatavals == (-1.0, -2.0)
            np.testing.assert_allclose(src.read(2), [[0.0, 0.5], [-2.0, 1.0]])


def test_normalise_img_pxl_vals(tmp_path):
    """Test calc_img_norm_params, apply_img_norm_params, and normalise_img_pxl_vals."""
    in_p = tmp_path / "norm_in.tif"
    b1 = np.arange(1, 26, dtype=np.float32).reshape(5, 5)
    b2 = (b1 * 10.0).astype(np.float32)
    b1[0, 0] = -9999.0
    b2[0, 0] = -9999.0
    b1[4, 4] = np.nan
    _write_raster(in_p, np.stack([b1, b2]), nodata=-9999.0)

    # 1. Split calc -> save JSON -> apply from file with linear stretch + custom bounds
    stats_file = tmp_path / "linear_stats.json"
    params = ic_resc.calc_img_norm_params(
        str(in_p),
        method=rsgispy.IMG_STRETCH_LINEAR,
        out_min=0.0,
        out_max=100.0,
        out_stats_file=stats_file,
        use_pbar=False,
    )
    assert stats_file.exists()
    assert params["method"] == "linear"

    out_lin = tmp_path / "out_linear.tif"
    ic_resc.apply_img_norm_params(
        str(in_p),
        str(out_lin),
        stats_file=stats_file,
        out_no_data_val=-9999.0,
        use_pbar=False,
    )
    with rasterio.open(out_lin) as src:
        arr = src.read()
        assert arr[0, 0, 0] == -9999.0
        assert arr[0, 4, 4] == -9999.0
        valid_b1 = arr[0][arr[0] != -9999.0]
        assert np.min(valid_b1) == pytest.approx(0.0, abs=1e-5)
        assert np.max(valid_b1) == pytest.approx(100.0, abs=1e-5)

    # 2. Mapping stretches (stdev, cumulative, user) and z-score
    for meth, extra_kw in [
        ("stdev", {"n_stdevs": 1.5}),
        ("cumulative", {"lower": 5.0, "upper": 95.0}),
        ("user", {"in_min_max": [{"min": 0.0, "max": 30.0}, {"min": 0.0, "max": 300.0}]}),
        ("none", {}),
    ]:
        out_m = tmp_path / f"out_{meth}.tif"
        ic_resc.normalise_img_pxl_vals(
            str(in_p),
            str(out_m),
            method=meth,
            out_min=0.0,
            out_max=1.0,
            use_pbar=False,
            **extra_kw,
        )
        assert out_m.exists()

    out_z = tmp_path / "out_zscore.tif"
    ic_resc.normalise_img_pxl_vals(
        str(in_p), str(out_z), method="zscore", use_pbar=False
    )
    with rasterio.open(out_z) as src:
        z_b1 = src.read(1)
        valid_z = z_b1[z_b1 != -9999.0]
        assert np.mean(valid_z) == pytest.approx(0.0, abs=1e-5)
        assert np.std(valid_z) == pytest.approx(1.0, abs=1e-5)

    # Z-score mapped to custom [out_min, out_max]
    out_z_mm = tmp_path / "out_zscore_mm.tif"
    ic_resc.normalise_img_pxl_vals(
        str(in_p),
        str(out_z_mm),
        method="zscore",
        out_min=-1.0,
        out_max=1.0,
        use_pbar=False,
    )
    with rasterio.open(out_z_mm) as src:
        zm = src.read(1)
        valid_zm = zm[zm != -9999.0]
        assert np.min(valid_zm) == pytest.approx(-1.0, abs=1e-5)
        assert np.max(valid_zm) == pytest.approx(1.0, abs=1e-5)

    # 3. Additional non-sklearn scaling/normalisation methods
    for meth in (
        "robust_zscore",
        "decimal_scaling",
        "log",
        "log1p",
        "sigmoid",
        "tanh",
        "l2",
        "softmax",
    ):
        out_extra = tmp_path / f"out_{meth}.tif"
        ic_resc.normalise_img_pxl_vals(
            str(in_p), str(out_extra), method=meth, use_pbar=False
        )
        with rasterio.open(out_extra) as src:
            assert src.count == 2
            assert src.read(1)[0, 0] == -9999.0

    # 4. Scikit-learn scalers and transformers (including JSON save/load roundtrip)
    pytest.importorskip("sklearn")
    for sk_meth in (
        "StandardScaler",
        "MinMaxScaler",
        "MaxAbsScaler",
        "RobustScaler",
        "PowerTransformer",
        "QuantileTransformer",
        "Normalizer",
    ):
        sk_stats = tmp_path / f"stats_{sk_meth}.json"
        sk_out = tmp_path / f"out_{sk_meth}.tif"
        ic_resc.calc_img_norm_params(
            str(in_p),
            method=sk_meth,
            out_stats_file=sk_stats,
            rnd_seed=42,
            use_pbar=False,
        )
        ic_resc.apply_img_norm_params(
            str(in_p),
            str(sk_out),
            stats_file=sk_stats,
            use_pbar=False,
        )
        with rasterio.open(sk_out) as src:
            d = src.read(1)
            assert d[0, 0] == -9999.0
            valid_d = d[d != -9999.0]
            assert np.all(np.isfinite(valid_d))
            if sk_meth == "MinMaxScaler":
                assert np.min(valid_d) == pytest.approx(0.0, abs=1e-5)
                assert np.max(valid_d) == pytest.approx(1.0, abs=1e-5)
            elif sk_meth == "MaxAbsScaler":
                assert np.max(np.abs(valid_d)) == pytest.approx(1.0, abs=1e-5)

    # Validation errors
    with pytest.raises(rsgispy.RSGISPyException):
        ic_resc.calc_img_norm_params(str(in_p), method="unknown_algo")
    with pytest.raises(rsgispy.RSGISPyException):
        ic_resc.calc_img_norm_params(str(in_p), out_min=0.0, out_max=None)
    with pytest.raises(rsgispy.RSGISPyException):
        ic_resc.apply_img_norm_params(str(in_p), str(tmp_path / "err.tif"))


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
        str(mask_p),
        str(out_p),
        -99,
        1,
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
        for fn_name in (
            "calc_img_correlation",
            "calc_img_earth_move_dist",
            "calc_img_mutual_info",
            "calc_img_difference",
        ):
            assert hasattr(cmp_fresh, fn_name)
            assert fn_name in cmp_fresh.__all__
        with pytest.raises(rsgispy.RSGISPyException, match="pip install"):
            cmp_fresh.calc_img_correlation("a.tif", "b.tif", 1, 1)
        with pytest.raises(rsgispy.RSGISPyException, match="pip install"):
            cmp_fresh.calc_img_earth_move_dist("a.tif", "b.tif", 1, 1)
        with pytest.raises(rsgispy.RSGISPyException, match="pip install"):
            cmp_fresh.calc_img_mutual_info("a.tif", "b.tif", 1, 1)

        assert dist_fresh.HAVE_SCIPY is False
        for fn_name in ("calc_dist_to_img_vals", "buffer_img_pxl_vals"):
            assert hasattr(dist_fresh, fn_name)
            assert fn_name in dist_fresh.__all__
        with pytest.raises(rsgispy.RSGISPyException, match="pip install"):
            dist_fresh.calc_dist_to_img_vals("in.tif", "out.tif", [1])
        with pytest.raises(rsgispy.RSGISPyException, match="pip install"):
            dist_fresh.buffer_img_pxl_vals("in.tif", "out.tif", [1], 10.0)

        assert fill_fresh.HAVE_SCIPY is False
        assert hasattr(fill_fresh, "calc_fill_regions_knn")
        assert hasattr(fill_fresh, "region_growing_fill")
        assert hasattr(fill_fresh, "region_growing_fill_tiled")
        assert "calc_fill_regions_knn" in fill_fresh.__all__
        assert "region_growing_fill" in fill_fresh.__all__
        assert "region_growing_fill_tiled" in fill_fresh.__all__
        with pytest.raises(rsgispy.RSGISPyException, match="pip install"):
            fill_fresh.calc_fill_regions_knn("in.tif", "fill.tif", "out.tif", 0, 1)


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


def test_region_growing_fill_balanced_voronoi(tmp_path):
    """Test balanced nearest-boundary region growing on diagonal and cardinal seeds."""
    reg_p = tmp_path / "rg_reg_diag.tif"
    mask_p = tmp_path / "rg_mask_diag.tif"
    out_p = tmp_path / "rg_out_diag.tif"

    reg_arr = np.zeros((1, 9, 9), dtype=np.uint8)
    reg_arr[0, 0, 0] = 4
    reg_arr[0, 8, 8] = 34
    mask_arr = np.ones((1, 9, 9), dtype=np.uint8)

    _write_raster(reg_p, reg_arr, dtype="uint8", nodata=0)
    _write_raster(mask_p, mask_arr, dtype="uint8", nodata=0)

    ic_fill.region_growing_fill(
        reg_p,
        mask_p,
        out_p,
        calc_stats=True,
        use_pbar=False,
    )
    with rasterio.open(out_p) as src:
        out = src.read(1)
        assert out.dtype == np.uint8
        assert src.nodata == 0
        tags = src.tags(1)
        assert float(tags["STATISTICS_MINIMUM"]) == 4.0
        assert float(tags["STATISTICS_MAXIMUM"]) == 34.0

    rows, cols = np.indices((9, 9))
    assert np.all(out[rows + cols < 8] == 4)
    assert np.all(out[rows + cols > 8] == 34)
    assert out[0, 0] == 4
    assert out[8, 8] == 34
    assert np.all(np.isin(out, [4, 34]))

    # Non-axis-aligned triangle check: (0,0)=1, (0,8)=2, (8,4)=3 -> central pixel (3,4)
    # Nearest to (8,4) is dist 5; to (0,0)/(0,8) is dist sqrt(9+16)=5 -> closer to (0,x) at row 2
    reg_tri = np.zeros((1, 9, 9), dtype=np.uint8)
    reg_tri[0, 0, 0] = 1
    reg_tri[0, 0, 8] = 2
    reg_tri[0, 8, 4] = 3
    reg_tri_p = tmp_path / "rg_tri.tif"
    out_tri_p = tmp_path / "rg_tri_out.tif"
    _write_raster(reg_tri_p, reg_tri, dtype="uint8", nodata=0)
    ic_fill.region_growing_fill(
        reg_tri_p, mask_p, out_tri_p, calc_stats=False, use_pbar=False
    )
    with rasterio.open(out_tri_p) as src:
        tri = src.read(1)
    assert tri[1, 1] == 1
    assert tri[1, 7] == 2
    assert tri[6, 4] == 3


def test_region_growing_fill_barriers_and_disconnected(tmp_path):
    """Test geodesic barrier routing and disconnected component isolation."""
    reg_p = tmp_path / "rg_reg_barrier.tif"
    mask_p = tmp_path / "rg_mask_barrier.tif"
    out8_p = tmp_path / "rg_out_barrier8.tif"
    out4_p = tmp_path / "rg_out_barrier4.tif"

    # Left block is connected except separated by a wall with an opening only at bottom,
    # plus a completely disconnected island on cols 6..7.
    mask = np.array(
        [
            [
                [1, 1, 0, 1, 1, 0, 1, 1],
                [1, 1, 0, 1, 1, 0, 1, 1],
                [1, 1, 0, 1, 1, 0, 1, 1],
                [1, 1, 0, 1, 1, 0, 0, 0],
                [1, 1, 1, 1, 1, 0, 1, 1],
            ]
        ],
        dtype=np.uint8,
    )
    # Seed 4 at (0, 0), Seed 34 at (3, 4) — 34 must dominate upper pocket (0, 3) around wall
    reg = np.full((1, 5, 8), -99, dtype=np.int32)
    reg[0, 0, 0] = 4
    reg[0, 3, 4] = 34

    _write_raster(reg_p, reg, dtype="int32", nodata=-99)
    _write_raster(mask_p, mask, dtype="uint8", nodata=0)

    ic_fill.region_growing_fill(
        reg_p,
        mask_p,
        out8_p,
        ref_no_data=-99,
        fill_region_val=1,
        connectivity=8,
        calc_stats=False,
        use_pbar=False,
    )
    with rasterio.open(out8_p) as src:
        out8 = src.read(1)
        assert out8.dtype == np.int32
        assert src.nodata == -99

    # Wall and non-mask pixels remain -99
    assert np.all(out8[:, 2][:4] == -99)
    assert np.all(out8[:, 5] == -99)
    # Disconnected unseeded islands (cols 6..7) remain unfilled (-99)
    assert np.all(out8[:, 6:] == -99)
    # Upper pocket behind the barrier is claimed by 34 (geodesically much closer than 4)
    assert out8[0, 3] == 34
    assert out8[0, 4] == 34
    # Left side claimed by 4 above the opening; 34 reaches (4, 1) first (17 vs 22)
    assert np.all(out8[:4, :2] == 4)
    assert out8[4, 0] == 4
    assert out8[4, 1] == 34

    # 4-connectivity check with diagonal-only gap across barrier
    mask_diag = np.array(
        [
            [
                [1, 1, 0, 1],
                [1, 0, 1, 1],
                [1, 1, 0, 1],
            ]
        ],
        dtype=np.uint8,
    )
    reg_diag = np.zeros((1, 3, 4), dtype=np.uint8)
    reg_diag[0, 0, 0] = 7
    reg_diag_p = tmp_path / "rg_reg_4c.tif"
    mask_diag_p = tmp_path / "rg_mask_4c.tif"
    _write_raster(reg_diag_p, reg_diag, dtype="uint8", nodata=0)
    _write_raster(mask_diag_p, mask_diag, dtype="uint8", nodata=0)

    ic_fill.region_growing_fill(
        reg_diag_p,
        mask_diag_p,
        out4_p,
        connectivity=4,
        calc_stats=False,
        use_pbar=False,
    )
    with rasterio.open(out4_p) as src:
        out4 = src.read(1)
    # Under 4-connectivity, (0,1)->(1,2) cannot cross diagonal-only choke point
    assert out4[0, 1] == 7
    assert out4[2, 1] == 7
    assert out4[1, 2] == 0
    assert out4[0, 3] == 0


def test_region_growing_fill_adjacent_overlap_max_iters_and_nodata(tmp_path):
    """Test adjacent masks, spatial intersection offsets, max_iters, and float/NaN handling."""
    # 1) Adjacent seed outside fill_mask == 2 + max_iters
    reg = np.zeros((1, 7, 7), dtype=np.uint8)
    reg[0, 3, 3] = 5
    mask = np.full((1, 7, 7), 2, dtype=np.uint8)
    mask[0, 3, 3] = 0  # seed sits just outside fill_mask==2

    reg_p = tmp_path / "rg_adj_reg.tif"
    mask_p = tmp_path / "rg_adj_mask.tif"
    out_p = tmp_path / "rg_adj_out.tif"
    _write_raster(reg_p, reg, dtype="uint8", nodata=0)
    _write_raster(mask_p, mask, dtype="uint8", nodata=0)

    ic_fill.region_growing_fill(
        reg_p,
        mask_p,
        out_p,
        fill_region_val=2,
        max_iters=1,
        no_data_val=99,
        calc_stats=False,
        use_pbar=False,
    )
    with rasterio.open(out_p) as src:
        out = src.read(1)
        assert src.nodata == 99
    # Centre (outside fill_mask==2) stays nodata=99; 3x3 ring around it gets 5; rest stays 99
    assert out[3, 3] == 99
    assert out[2, 2] == 5 and out[3, 4] == 5 and out[4, 4] == 5
    assert out[1, 3] == 99 and out[0, 0] == 99

    # max_iters=0 does not expand beyond initial seed pixels inside fill_mask
    out_zero_p = tmp_path / "rg_zero_out.tif"
    mask_full = np.full((1, 7, 7), 2, dtype=np.uint8)
    mask_full_p = tmp_path / "rg_full_mask.tif"
    _write_raster(mask_full_p, mask_full, dtype="uint8", nodata=0)
    ic_fill.region_growing_fill(
        reg_p,
        mask_full_p,
        out_zero_p,
        fill_region_val=2,
        max_iters=0,
        calc_stats=False,
        use_pbar=False,
    )
    with rasterio.open(out_zero_p) as src:
        out_zero = src.read(1)
    assert out_zero[3, 3] == 5
    assert int(np.count_nonzero(out_zero == 5)) == 1

    # 2) Partial spatial overlap with pixel offsets
    reg_big = np.zeros((1, 6, 6), dtype=np.uint8)
    reg_big[0, 1, 1] = 11
    reg_big[0, 4, 4] = 22
    mask_shifted = np.ones((1, 4, 4), dtype=np.uint8)
    reg_big_p = tmp_path / "rg_big.tif"
    mask_shifted_p = tmp_path / "rg_shifted.tif"
    out_overlap_p = tmp_path / "rg_overlap.tif"
    _write_raster(
        reg_big_p,
        reg_big,
        origin=(0.0, 60.0),
        pixel_size=(10.0, 10.0),
        dtype="uint8",
        nodata=0,
    )
    _write_raster(
        mask_shifted_p,
        mask_shifted,
        origin=(10.0, 50.0),
        pixel_size=(10.0, 10.0),
        dtype="uint8",
        nodata=0,
    )
    ic_fill.region_growing_fill(
        reg_big_p,
        mask_shifted_p,
        out_overlap_p,
        calc_stats=False,
        use_pbar=False,
    )
    with rasterio.open(out_overlap_p) as src:
        ov = src.read(1)
        assert ov.shape == (4, 4)
        assert ov[0, 0] == 11
        assert ov[3, 3] == 22

    # 3) Float raster with NaN nodata + works without scipy
    flt_reg = np.full((1, 4, 4), np.nan, dtype=np.float32)
    flt_reg[0, 0, 0] = 1.5
    flt_reg[0, 3, 3] = 2.5
    flt_mask = np.ones((1, 4, 4), dtype=np.uint8)
    flt_reg_p = tmp_path / "rg_flt_reg.tif"
    flt_mask_p = tmp_path / "rg_flt_mask.tif"
    flt_out_p = tmp_path / "rg_flt_out.tif"
    _write_raster(flt_reg_p, flt_reg, dtype="float32", nodata=np.nan)
    _write_raster(flt_mask_p, flt_mask, dtype="uint8", nodata=0)

    ic_fill.region_growing_fill(
        flt_reg_p, flt_mask_p, flt_out_p, calc_stats=False, use_pbar=False
    )
    with rasterio.open(flt_out_p) as src:
        fout = src.read(1)
        assert fout.dtype == np.float32
        assert fout[0, 0] == pytest.approx(1.5)
        assert fout[3, 3] == pytest.approx(2.5)
        assert np.all(np.isin(fout, [1.5, 2.5]))

    # Casting NaN-nodata float seed raster to uint8 falls back out nodata to 0
    uint_out_p = tmp_path / "rg_flt_to_uint.tif"
    ic_fill.region_growing_fill(
        flt_reg_p,
        flt_mask_p,
        uint_out_p,
        datatype=rsgispy.TYPE_8UINT,
        calc_stats=False,
        use_pbar=False,
    )
    with rasterio.open(uint_out_p) as src:
        assert src.dtypes[0] == "uint8"
        assert src.nodata == 0


def test_region_growing_fill_errors(tmp_path):
    """Test input validation errors in region_growing_fill."""
    reg_p = tmp_path / "rg_err_reg.tif"
    mask_p = tmp_path / "rg_err_mask.tif"
    mask_crs_p = tmp_path / "rg_err_crs.tif"
    empty_reg_p = tmp_path / "rg_err_empty.tif"
    out_p = tmp_path / "rg_err_out.tif"

    reg = np.array([[[0, 1], [0, 0]]], dtype=np.uint8)
    mask = np.ones((1, 2, 2), dtype=np.uint8)
    empty_reg = np.zeros((1, 2, 2), dtype=np.uint8)

    _write_raster(reg_p, reg, dtype="uint8", nodata=0)
    _write_raster(mask_p, mask, dtype="uint8", nodata=0)
    _write_raster(mask_crs_p, mask, crs="EPSG:4326", dtype="uint8", nodata=0)
    _write_raster(empty_reg_p, empty_reg, dtype="uint8", nodata=0)

    with pytest.raises(rsgispy.RSGISPyException):
        ic_fill.region_growing_fill(tmp_path / "missing.tif", mask_p, out_p)

    with pytest.raises(rsgispy.RSGISPyException):
        ic_fill.region_growing_fill(reg_p, tmp_path / "missing_mask.tif", out_p)

    with pytest.raises(rsgispy.RSGISPyException, match="connectivity"):
        ic_fill.region_growing_fill(reg_p, mask_p, out_p, connectivity=6)

    with pytest.raises(rsgispy.RSGISPyException, match="max_iters"):
        ic_fill.region_growing_fill(reg_p, mask_p, out_p, max_iters=-1)

    with pytest.raises(rsgispy.RSGISPyException, match="projection"):
        ic_fill.region_growing_fill(reg_p, mask_crs_p, out_p)

    with pytest.raises(rsgispy.RSGISPyException, match="No valid seed"):
        ic_fill.region_growing_fill(empty_reg_p, mask_p, out_p)


def _split_raster_into_tiles(
    tmp_path,
    prefix,
    reg_arr,
    mask_arr,
    row_ranges,
    col_ranges,
    *,
    origin=(0.0, 250.0),
    pixel_size=(10.0, 10.0),
    dtype="uint8",
    reg_nodata=0,
):
    """Split 3D arrays into non-overlapping tile GeoTIFFs along (row_ranges, col_ranges)."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    reg_tiles = []
    mask_tiles = []
    out_tiles = []
    slices = []
    ox, oy = origin
    px, py = pixel_size

    for t_idx, (r0, r1) in enumerate(row_ranges):
        for c_idx, (c0, c1) in enumerate(col_ranges):
            tile_orig = (ox + c0 * px, oy - r0 * py)
            rp = tmp_path / f"{prefix}_reg_{t_idx}_{c_idx}.tif"
            mp = tmp_path / f"{prefix}_msk_{t_idx}_{c_idx}.tif"
            op = tmp_path / f"{prefix}_out_{t_idx}_{c_idx}.tif"
            _write_raster(
                rp,
                reg_arr[:, r0:r1, c0:c1],
                origin=tile_orig,
                pixel_size=pixel_size,
                dtype=dtype,
                nodata=reg_nodata,
            )
            _write_raster(
                mp,
                mask_arr[:, r0:r1, c0:c1],
                origin=tile_orig,
                pixel_size=pixel_size,
                dtype="uint8",
                nodata=0,
            )
            reg_tiles.append(rp)
            mask_tiles.append(mp)
            out_tiles.append(op)
            slices.append((r0, r1, c0, c1))

    return reg_tiles, mask_tiles, out_tiles, slices


def _mosaic_tile_outputs(out_tiles, slices, shape, dtype):
    """Assemble output tile GeoTIFFs back into a 2D NumPy array."""
    mosaic = np.zeros(shape, dtype=dtype)
    for op, (r0, r1, c0, c1) in zip(out_tiles, slices):
        with rasterio.open(op) as src:
            mosaic[r0:r1, c0:c1] = src.read(1)
    return mosaic


def test_region_growing_fill_tiled_matches_global(tmp_path):
    """Test tile-list region_growing_fill_tiled matches global output for 1 and 2 cores."""
    reg_p = tmp_path / "rgt_reg.tif"
    mask_p = tmp_path / "rgt_mask.tif"
    global_p = tmp_path / "rgt_global.tif"

    # 15x15 grid split into irregular 4x4 tiles (widths/heights mixing 4 and 3)
    # Seed 4 at (1, 1) and seed 34 at (13, 12), separated by a serpentine wall that forces
    # waves to travel right, down, left, AND up across multiple empty-seed tiles.
    # Also includes a completely empty-mask region (rows 0..2, cols 12..14) and an
    # unreachable island at (0, 14).
    mask = np.ones((1, 15, 15), dtype=np.uint8)
    mask[0, :11, 5] = 0
    mask[0, 4:, 10] = 0
    mask[0, :3, 12:] = 0
    mask[0, 0, 14] = 1

    reg = np.zeros((1, 15, 15), dtype=np.uint8)
    reg[0, 1, 1] = 4
    reg[0, 13, 12] = 34

    origin = (0.0, 250.0)
    _write_raster(reg_p, reg, origin=origin, dtype="uint8", nodata=0)
    _write_raster(mask_p, mask, origin=origin, dtype="uint8", nodata=0)

    ic_fill.region_growing_fill(
        reg_p, mask_p, global_p, connectivity=8, calc_stats=False, use_pbar=False
    )
    with rasterio.open(global_p) as s_glob:
        g_arr = s_glob.read(1)

    row_ranges = [(0, 4), (4, 7), (7, 12), (12, 15)]
    col_ranges = [(0, 3), (3, 7), (7, 11), (11, 15)]
    reg_tiles, msk_tiles, out1_tiles, slices = _split_raster_into_tiles(
        tmp_path,
        "irreg",
        reg,
        mask,
        row_ranges,
        col_ranges,
        origin=origin,
        dtype="uint8",
        reg_nodata=0,
    )

    # Shuffle tile order to verify order-independence
    perm = list(reversed(range(len(reg_tiles))))
    reg_shuf = [reg_tiles[i] for i in perm]
    msk_shuf = [msk_tiles[i] for i in perm]
    out1_shuf = [out1_tiles[i] for i in perm]
    slices_shuf = [slices[i] for i in perm]

    ic_fill.region_growing_fill_tiled(
        reg_shuf,
        msk_shuf,
        out1_shuf,
        connectivity=8,
        n_cores=1,
        calc_stats=True,
        use_pbar=False,
    )
    t1_arr = _mosaic_tile_outputs(out1_shuf, slices_shuf, (15, 15), np.uint8)
    np.testing.assert_array_equal(t1_arr, g_arr)
    assert t1_arr[0, 14] == 0

    with rasterio.open(out1_shuf[-1]) as s_tile:
        assert s_tile.dtypes[0] == "uint8"
        assert s_tile.nodata == 0
        tags = s_tile.tags(1)
        assert float(tags["STATISTICS_MINIMUM"]) == 4.0

    # Parallel execution with n_cores=2
    out2_shuf = [tmp_path / f"irreg_out2_{i}.tif" for i in range(len(reg_shuf))]
    ic_fill.region_growing_fill_tiled(
        reg_shuf,
        msk_shuf,
        out2_shuf,
        connectivity=8,
        n_cores=2,
        calc_stats=False,
        use_pbar=False,
    )
    t2_arr = _mosaic_tile_outputs(out2_shuf, slices_shuf, (15, 15), np.uint8)
    np.testing.assert_array_equal(t2_arr, g_arr)

    # Test 4-connectivity + max_iters=3 + n_cores=2 matches global identically
    glob4_p = tmp_path / "rgt_glob4.tif"
    ic_fill.region_growing_fill(
        reg_p,
        mask_p,
        glob4_p,
        connectivity=4,
        max_iters=3,
        calc_stats=False,
        use_pbar=False,
    )
    out4_tiles = [tmp_path / f"irreg_out4_{i}.tif" for i in range(len(reg_tiles))]
    ic_fill.region_growing_fill_tiled(
        reg_tiles,
        msk_tiles,
        out4_tiles,
        connectivity=4,
        max_iters=3,
        n_cores=2,
        calc_stats=False,
        use_pbar=False,
    )
    t4_arr = _mosaic_tile_outputs(out4_tiles, slices, (15, 15), np.uint8)
    with rasterio.open(glob4_p) as s_g4:
        np.testing.assert_array_equal(t4_arr, s_g4.read(1))

    # Extreme 1x1 tile files on a 5x5 int32 grid with non-zero nodata (-99)
    small_reg_p = tmp_path / "rgt_small_reg.tif"
    small_msk_p = tmp_path / "rgt_small_msk.tif"
    small_g_p = tmp_path / "rgt_small_g.tif"
    s_reg = np.zeros((1, 5, 5), dtype=np.int32)
    s_reg[0, 0, 0] = 10
    s_reg[0, 4, 4] = 20
    s_msk = np.ones((1, 5, 5), dtype=np.uint8)
    _write_raster(small_reg_p, s_reg, origin=origin, dtype="int32", nodata=-99)
    _write_raster(small_msk_p, s_msk, origin=origin, dtype="uint8", nodata=0)

    ic_fill.region_growing_fill(
        small_reg_p,
        small_msk_p,
        small_g_p,
        ref_no_data=-99,
        calc_stats=False,
        use_pbar=False,
    )
    one_by_one = [(i, i + 1) for i in range(5)]
    s_reg_t, s_msk_t, s_out_t, s_slices = _split_raster_into_tiles(
        tmp_path,
        "px1",
        s_reg,
        s_msk,
        one_by_one,
        one_by_one,
        origin=origin,
        dtype="int32",
        reg_nodata=-99,
    )
    ic_fill.region_growing_fill_tiled(
        s_reg_t,
        s_msk_t,
        s_out_t,
        ref_no_data=-99,
        n_cores=2,
        calc_stats=True,
        use_pbar=False,
    )
    small_mosaic = _mosaic_tile_outputs(s_out_t, s_slices, (5, 5), np.int32)
    with rasterio.open(small_g_p) as sg, rasterio.open(s_out_t[0]) as st0:
        assert st0.nodata == -99
        np.testing.assert_array_equal(small_mosaic, sg.read(1))


def test_region_growing_fill_tiled_unseeded_eez_outside_seeds_and_errors(tmp_path):
    """Test unseeded EEZ tiles, outside seeds, sparse omitted tiles, tmp_dir cleanup, and errors."""
    # Build a 2x2 layout of 6x6 tiles:
    # Tile 0 (TL): seed 4 at (1, 1) outside fill_mask
    # Tile 1 (TR) & Tile 2 (BL): EEZ tiles with NO seeds
    # Tile 3 (BR): seed 34 at (4, 3) (asymmetric to avoid exact equidistant ties)
    reg_tiles = []
    msk_tiles = []
    origins = [(0.0, 120.0), (60.0, 120.0), (0.0, 60.0), (60.0, 60.0)]
    for idx, orig in enumerate(origins):
        r_arr = np.zeros((1, 6, 6), dtype=np.uint8)
        m_arr = np.ones((1, 6, 6), dtype=np.uint8)
        if idx == 0:
            r_arr[0, 1, 1] = 4
            m_arr[0, 1, 1] = 0  # seed outside fill_mask
        elif idx == 3:
            r_arr[0, 4, 3] = 34

        rp = tmp_path / f"eez_reg_{idx}.tif"
        mp = tmp_path / f"eez_msk_{idx}.tif"
        _write_raster(
            rp, r_arr, origin=orig, pixel_size=(10.0, 10.0), dtype="uint8", nodata=0
        )
        _write_raster(
            mp, m_arr, origin=orig, pixel_size=(10.0, 10.0), dtype="uint8", nodata=0
        )
        reg_tiles.append(rp)
        msk_tiles.append(mp)

    # Reference global 12x12 arrays
    full_reg = np.zeros((1, 12, 12), dtype=np.uint8)
    full_reg[0, 1, 1] = 4
    full_reg[0, 10, 9] = 34
    full_msk = np.ones((1, 12, 12), dtype=np.uint8)
    full_msk[0, 1, 1] = 0
    full_reg_p = tmp_path / "eez_full_reg.tif"
    full_msk_p = tmp_path / "eez_full_msk.tif"
    glob_out = tmp_path / "eez_global.tif"
    _write_raster(
        full_reg_p,
        full_reg,
        origin=(0.0, 120.0),
        pixel_size=(10.0, 10.0),
        dtype="uint8",
        nodata=0,
    )
    _write_raster(
        full_msk_p,
        full_msk,
        origin=(0.0, 120.0),
        pixel_size=(10.0, 10.0),
        dtype="uint8",
        nodata=0,
    )
    ic_fill.region_growing_fill(
        full_reg_p, full_msk_p, glob_out, calc_stats=False, use_pbar=False
    )
    with rasterio.open(glob_out) as sg:
        g_arr = sg.read(1)

    work_parent = tmp_path / "custom_tmp"
    out_tiles = [tmp_path / "nested_out" / f"eez_out_{idx}.tif" for idx in range(4)]
    ic_fill.region_growing_fill_tiled(
        reg_tiles,
        msk_tiles,
        out_tiles,
        n_cores=2,
        tmp_dir=work_parent,
        calc_stats=False,
        use_pbar=False,
    )

    tile_slices = [(0, 6, 0, 6), (0, 6, 6, 12), (6, 12, 0, 6), (6, 12, 6, 12)]
    t_arr = _mosaic_tile_outputs(out_tiles, tile_slices, (12, 12), np.uint8)
    np.testing.assert_array_equal(t_arr, g_arr)
    assert t_arr[1, 1] == 0  # outside fill_mask cleared to nodata
    assert t_arr[0, 11] == 4 or t_arr[11, 0] == 34  # unseeded EEZ tiles filled

    # Temporary workspace inside work_parent was cleaned up automatically
    assert work_parent.exists()
    assert list(work_parent.iterdir()) == []

    # Sparse L-shaped subset (3 of 4 tiles, omitting tile 2) with n_cores=-1
    sparse_out = [tmp_path / f"sparse_out_{i}.tif" for i in (0, 1, 3)]
    ic_fill.region_growing_fill_tiled(
        [reg_tiles[0], reg_tiles[1], reg_tiles[3]],
        [msk_tiles[0], msk_tiles[1], msk_tiles[3]],
        sparse_out,
        n_cores=-1,
        calc_stats=False,
        use_pbar=False,
    )
    with rasterio.open(sparse_out[1]) as s_tr:
        tr_arr = s_tr.read(1)
        assert np.all(np.isin(tr_arr, [4, 34]))

    # Validation errors
    err_outs = [tmp_path / f"rgt_err_{i}.tif" for i in range(4)]
    with pytest.raises(rsgispy.RSGISPyException, match="empty"):
        ic_fill.region_growing_fill_tiled([], [], [])
    with pytest.raises(rsgispy.RSGISPyException, match="same length"):
        ic_fill.region_growing_fill_tiled(reg_tiles, msk_tiles[:2], err_outs)
    with pytest.raises(rsgispy.RSGISPyException, match="max_sweeps"):
        ic_fill.region_growing_fill_tiled(
            reg_tiles, msk_tiles, err_outs, max_sweeps=-1
        )
    with pytest.raises(rsgispy.RSGISPyException, match="connectivity"):
        ic_fill.region_growing_fill_tiled(
            reg_tiles, msk_tiles, err_outs, connectivity=6
        )
    with pytest.raises(rsgispy.RSGISPyException, match="max_iters"):
        ic_fill.region_growing_fill_tiled(
            reg_tiles, msk_tiles, err_outs, max_iters=-2
        )

    # Overlapping tile check
    ov_reg = tmp_path / "ov_reg.tif"
    ov_msk = tmp_path / "ov_msk.tif"
    _write_raster(
        ov_reg,
        np.zeros((1, 6, 6), dtype=np.uint8),
        origin=(30.0, 120.0),
        pixel_size=(10.0, 10.0),
        dtype="uint8",
        nodata=0,
    )
    _write_raster(
        ov_msk,
        np.ones((1, 6, 6), dtype=np.uint8),
        origin=(30.0, 120.0),
        pixel_size=(10.0, 10.0),
        dtype="uint8",
        nodata=0,
    )
    with pytest.raises(rsgispy.RSGISPyException, match="overlap"):
        ic_fill.region_growing_fill_tiled(
            [reg_tiles[0], ov_reg],
            [msk_tiles[0], ov_msk],
            err_outs[:2],
        )

    # Unaligned pixel grid origin
    unaligned_reg = tmp_path / "unaligned_reg.tif"
    unaligned_msk = tmp_path / "unaligned_msk.tif"
    _write_raster(
        unaligned_reg,
        np.zeros((1, 6, 6), dtype=np.uint8),
        origin=(63.0, 120.0),
        pixel_size=(10.0, 10.0),
        dtype="uint8",
        nodata=0,
    )
    _write_raster(
        unaligned_msk,
        np.ones((1, 6, 6), dtype=np.uint8),
        origin=(63.0, 120.0),
        pixel_size=(10.0, 10.0),
        dtype="uint8",
        nodata=0,
    )
    with pytest.raises(rsgispy.RSGISPyException, match="aligned"):
        ic_fill.region_growing_fill_tiled(
            [reg_tiles[0], unaligned_reg],
            [msk_tiles[0], unaligned_msk],
            err_outs[:2],
        )

    # Different CRS between tiles
    crs_reg = tmp_path / "crs_reg.tif"
    crs_msk = tmp_path / "crs_msk.tif"
    _write_raster(
        crs_reg,
        np.zeros((1, 6, 6), dtype=np.uint8),
        origin=(60.0, 120.0),
        pixel_size=(10.0, 10.0),
        crs="EPSG:32631",
        dtype="uint8",
        nodata=0,
    )
    _write_raster(
        crs_msk,
        np.ones((1, 6, 6), dtype=np.uint8),
        origin=(60.0, 120.0),
        pixel_size=(10.0, 10.0),
        crs="EPSG:32631",
        dtype="uint8",
        nodata=0,
    )
    with pytest.raises(rsgispy.RSGISPyException, match="CRS"):
        ic_fill.region_growing_fill_tiled(
            [reg_tiles[0], crs_reg],
            [msk_tiles[0], crs_msk],
            err_outs[:2],
        )

    # No valid seeds across all tiles
    with pytest.raises(rsgispy.RSGISPyException, match="No valid seed"):
        ic_fill.region_growing_fill_tiled(
            [reg_tiles[1], reg_tiles[2]],
            [msk_tiles[1], msk_tiles[2]],
            err_outs[:2],
            use_pbar=False,
        )


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
