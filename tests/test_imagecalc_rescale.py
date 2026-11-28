"""
Tests for :mod:`rsgispy.imagecalc.rescale`: rescaling and normalising pixel
values.
"""

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

import rsgispy
import rsgispy.imagecalc.rescale as ic_resc


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
