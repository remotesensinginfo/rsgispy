"""
Tests for rsgispy and rsgispy.imagecalc.band_maths.
"""

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

import rsgispy
import rsgispy.imagecalc
import rsgispy.imageprocess


def _create_test_raster(
    path,
    bands_data,
    origin=(0.0, 100.0),
    pixel_size=(1.0, 1.0),
    crs="EPSG:4326",
    dtype="float32",
):
    """Helper to write a multi-band test GeoTIFF."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    count, height, width = bands_data.shape
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
    ) as dst:
        dst.write(bands_data.astype(dtype))


def test_datatype_helpers():
    """Test data type conversion helper functions in rsgispy."""
    assert rsgispy.get_rsgis_datatype("float32") == rsgispy.TYPE_32FLOAT
    assert rsgispy.get_rsgis_datatype("byte") == rsgispy.TYPE_8UINT
    assert rsgispy.get_rsgis_datatype("uint8") == rsgispy.TYPE_8UINT
    assert rsgispy.get_rsgis_datatype("int16") == rsgispy.TYPE_16INT

    assert rsgispy.get_rasterio_datatype(rsgispy.TYPE_32FLOAT) == "float32"
    assert rsgispy.get_rasterio_datatype(rsgispy.TYPE_8UINT) == "uint8"
    assert rsgispy.get_numpy_datatype(rsgispy.TYPE_32FLOAT) == np.float32
    assert rsgispy.get_numpy_datatype(rsgispy.TYPE_16UINT) == np.uint16
    assert rsgispy.get_rsgis_datatype_str(rsgispy.TYPE_32FLOAT) == "32 bit FLOAT"

    with pytest.raises(rsgispy.RSGISPyException):
        rsgispy.get_rsgis_datatype("unknown_type")
    with pytest.raises(rsgispy.RSGISPyException):
        rsgispy.get_rasterio_datatype(rsgispy.TYPE_UNDEFINED)
    with pytest.raises(rsgispy.RSGISPyException):
        rsgispy.get_numpy_datatype(999)
    with pytest.raises(rsgispy.RSGISPyException):
        rsgispy.get_rsgis_datatype_str(999)


def test_band_defn():
    """Test BandDefn initialisation, equality, repr, and to_dict."""
    b1 = rsgispy.imagecalc.BandDefn(band_name="b1", input_img="img.tif", img_band=1)
    b2 = rsgispy.imageprocess.BandDefn(band_name="b1", input_img="img.tif", img_band=1)
    b3 = rsgispy.imagecalc.BandDefn(band_name="b2", input_img="img.tif", img_band=2)

    assert b1 == b2
    assert b1 != b3
    assert b1 != "not_a_band_defn"
    assert b1.to_dict() == {
        "band_name": "b1",
        "input_img": "img.tif",
        "img_band": 1,
    }
    assert "BandDefn(" in repr(b1)


def test_band_maths_basic_multiplication(tmp_path):
    """Test basic two-band multiplication with band_maths and band_math alias."""
    in_img = tmp_path / "img.tif"
    out_img1 = tmp_path / "out1.tif"
    out_img2 = tmp_path / "out2.tif"

    b1_arr = np.arange(1, 26, dtype=np.float32).reshape(5, 5)
    b2_arr = np.full((5, 5), 3.5, dtype=np.float32)
    _create_test_raster(in_img, np.stack([b1_arr, b2_arr]))

    band_defns = [
        rsgispy.imagecalc.BandDefn(band_name="b1", input_img=str(in_img), img_band=1),
        rsgispy.imagecalc.BandDefn(band_name="b2", input_img=str(in_img), img_band=2),
    ]

    rsgispy.imagecalc.band_maths(
        output_img=str(out_img1),
        exp="b1*b2",
        gdalformat="GTIFF",
        datatype=rsgispy.TYPE_32FLOAT,
        band_defs=band_defns,
    )
    rsgispy.imagecalc.band_math(
        output_img=str(out_img2),
        exp="b1*b2",
        gdalformat="GTIFF",
        datatype=rsgispy.TYPE_32FLOAT,
        band_defs=band_defns,
        exp_band_name=True,
    )

    expected = b1_arr * b2_arr
    with rasterio.open(out_img1) as src1:
        assert src1.count == 1
        assert src1.dtypes == ("float32",)
        assert src1.descriptions == (None,)
        np.testing.assert_allclose(src1.read(1), expected)

    with rasterio.open(out_img2) as src2:
        assert src2.descriptions == ("b1*b2",)
        np.testing.assert_allclose(src2.read(1), expected)


def test_band_maths_ternary_and_uint8(tmp_path):
    """Test muParser ternary conditional expression and uint8 output casting."""
    in_img = tmp_path / "img.tif"
    out_img = tmp_path / "out.tif"

    b1_arr = np.array([[0, 1, 2], [1, 0, 5], [3, 4, 1]], dtype=np.uint8)
    b2_arr = np.array([[0, 0, 1], [0, 2, 1], [1, 0, 0]], dtype=np.uint8)
    _create_test_raster(in_img, np.stack([b1_arr, b2_arr]), dtype="uint8")

    band_defns = [
        rsgispy.imagecalc.BandDefn(band_name="b1", input_img=str(in_img), img_band=1),
        rsgispy.imagecalc.BandDefn(band_name="b2", input_img=str(in_img), img_band=2),
    ]

    rsgispy.imagecalc.band_maths(
        output_img=str(out_img),
        exp="(b1==1) || (b2==1) ? 1 : 0",
        gdalformat="GTIFF",
        datatype=rsgispy.TYPE_8UINT,
        band_defs=band_defns,
    )

    expected = np.where((b1_arr == 1) | (b2_arr == 1), 1, 0).astype(np.uint8)
    with rasterio.open(out_img) as src:
        assert src.dtypes == ("uint8",)
        np.testing.assert_array_equal(src.read(1), expected)


def test_band_maths_spatial_overlap(tmp_path):
    """Test band_maths across two input rasters with partially overlapping extents."""
    img1 = tmp_path / "img1.tif"
    img2 = tmp_path / "img2.tif"
    out_img = tmp_path / "out_overlap.tif"

    # img1: origin (0, 10), 10x10 -> x in [0, 10], y in [0, 10]
    arr1 = np.arange(100, dtype=np.float32).reshape(10, 10)
    _create_test_raster(img1, arr1[np.newaxis, :, :], origin=(0.0, 10.0))

    # img2: origin (2, 8), 10x10 -> x in [2, 12], y in [-2, 8]
    # Intersection is x in [2, 10] (width 8), y in [0, 8] (height 8)
    arr2 = np.full((10, 10), 10.0, dtype=np.float32)
    _create_test_raster(img2, arr2[np.newaxis, :, :], origin=(2.0, 8.0))

    band_defns = [
        rsgispy.imagecalc.BandDefn(band_name="b1", input_img=str(img1), img_band=1),
        rsgispy.imagecalc.BandDefn(band_name="b2", input_img=str(img2), img_band=1),
    ]

    rsgispy.imagecalc.band_maths(
        output_img=str(out_img),
        exp="b1 + b2",
        gdalformat="GTIFF",
        datatype=rsgispy.TYPE_32FLOAT,
        band_defs=band_defns,
    )

    with rasterio.open(out_img) as src:
        assert src.shape == (8, 8)
        assert (
            src.bounds.left,
            src.bounds.bottom,
            src.bounds.right,
            src.bounds.top,
        ) == (
            2.0,
            0.0,
            10.0,
            8.0,
        )
        np.testing.assert_allclose(src.read(1), arr1[2:10, 2:10] + arr2[0:8, 0:8])


def test_band_maths_multi_block_and_output_exists(tmp_path):
    """Test multi-block windowed processing (>256x256) and output_exists=True."""
    in_img = tmp_path / "large.tif"
    out_img = tmp_path / "large_out.tif"

    b1_arr = np.linspace(1.0, 100.0, 300 * 300, dtype=np.float32).reshape(300, 300)
    _create_test_raster(in_img, b1_arr[np.newaxis, :, :])
    _create_test_raster(out_img, np.zeros((1, 300, 300), dtype=np.float32))

    band_defns = [
        rsgispy.imagecalc.BandDefn(band_name="b1", input_img=str(in_img), img_band=1)
    ]

    rsgispy.imagecalc.band_maths(
        output_img=str(out_img),
        exp="sqrt(b1) + 2.0",
        gdalformat="GTIFF",
        datatype=rsgispy.TYPE_32FLOAT,
        band_defs=band_defns,
        output_exists=True,
    )

    expected = np.sqrt(b1_arr) + 2.0
    with rasterio.open(out_img) as src:
        np.testing.assert_allclose(src.read(1), expected, rtol=1e-5)


def test_band_maths_validation_errors(tmp_path):
    """Test error handling for invalid inputs, expressions, and non-overlapping images."""
    in_img = tmp_path / "img.tif"
    no_overlap_img = tmp_path / "no_overlap.tif"
    diff_res_img = tmp_path / "diff_res.tif"
    out_img = tmp_path / "out.tif"

    _create_test_raster(in_img, np.ones((1, 5, 5), dtype=np.float32), origin=(0.0, 5.0))
    _create_test_raster(
        no_overlap_img, np.ones((1, 5, 5), dtype=np.float32), origin=(50.0, 55.0)
    )
    _create_test_raster(
        diff_res_img,
        np.ones((1, 5, 5), dtype=np.float32),
        origin=(0.0, 5.0),
        pixel_size=(2.0, 2.0),
    )

    # Empty band_defs
    with pytest.raises(rsgispy.RSGISPyException):
        rsgispy.imagecalc.band_maths(
            str(out_img), "b1", "GTIFF", rsgispy.TYPE_32FLOAT, []
        )

    # Non-existent file
    with pytest.raises(rsgispy.RSGISPyException):
        rsgispy.imagecalc.band_maths(
            str(out_img),
            "b1",
            "GTIFF",
            rsgispy.TYPE_32FLOAT,
            [rsgispy.imagecalc.BandDefn("b1", str(tmp_path / "missing.tif"), 1)],
        )

    # Out-of-range band index
    with pytest.raises(rsgispy.RSGISPyException):
        rsgispy.imagecalc.band_maths(
            str(out_img),
            "b1",
            "GTIFF",
            rsgispy.TYPE_32FLOAT,
            [rsgispy.imagecalc.BandDefn("b1", str(in_img), 5)],
        )

    # Invalid expression / undefined variable
    with pytest.raises(rsgispy.RSGISPyException):
        rsgispy.imagecalc.band_maths(
            str(out_img),
            "b1 + unknown_var",
            "GTIFF",
            rsgispy.TYPE_32FLOAT,
            [rsgispy.imagecalc.BandDefn("b1", str(in_img), 1)],
        )

    # Non-overlapping rasters
    with pytest.raises(rsgispy.RSGISPyException):
        rsgispy.imagecalc.band_maths(
            str(out_img),
            "b1 + b2",
            "GTIFF",
            rsgispy.TYPE_32FLOAT,
            [
                rsgispy.imagecalc.BandDefn("b1", str(in_img), 1),
                rsgispy.imagecalc.BandDefn("b2", str(no_overlap_img), 1),
            ],
        )

    # Different pixel resolutions
    with pytest.raises(rsgispy.RSGISPyException):
        rsgispy.imagecalc.band_maths(
            str(out_img),
            "b1 + b2",
            "GTIFF",
            rsgispy.TYPE_32FLOAT,
            [
                rsgispy.imagecalc.BandDefn("b1", str(in_img), 1),
                rsgispy.imagecalc.BandDefn("b2", str(diff_res_img), 1),
            ],
        )


def test_tqdm_progress_bar_and_notebook_detection(tmp_path, capsys, monkeypatch):
    """Test tqdm progress bar suppression (use_pbar=False) and notebook detection."""
    in_img = tmp_path / "img.tif"
    out_img_quiet = tmp_path / "out_quiet.tif"
    out_img_pbar = tmp_path / "out_pbar.tif"

    _create_test_raster(in_img, np.ones((1, 5, 5), dtype=np.float32))
    band_defns = [
        rsgispy.imagecalc.BandDefn(band_name="b1", input_img=str(in_img), img_band=1)
    ]

    assert rsgispy.is_notebook() is False

    # Suppress progress bar with use_pbar=False
    capsys.readouterr()
    rsgispy.imagecalc.band_maths(
        output_img=str(out_img_quiet),
        exp="b1 * 2",
        gdalformat="GTIFF",
        datatype=rsgispy.TYPE_32FLOAT,
        band_defs=band_defns,
        use_pbar=False,
    )
    captured_quiet = capsys.readouterr()
    assert captured_quiet.err == ""

    # Enable progress bar with use_pbar=True
    rsgispy.imagecalc.band_maths(
        output_img=str(out_img_pbar),
        exp="b1 * 2",
        gdalformat="GTIFF",
        datatype=rsgispy.TYPE_32FLOAT,
        band_defs=band_defns,
        use_pbar=True,
    )
    captured_pbar = capsys.readouterr()
    assert "100%" in captured_pbar.err or "1/1" in captured_pbar.err

    # Simulate ZMQInteractiveShell (Jupyter notebook)
    class ZMQInteractiveShell:  # pylint: disable=too-few-public-methods
        """Dummy Jupyter shell class."""

    monkeypatch.setattr(
        rsgispy, "get_ipython", ZMQInteractiveShell, raising=False
    )
    assert rsgispy.is_notebook() is True
    assert callable(rsgispy.get_tqdm())


def test_image_math_and_image_band_math(tmp_path):
    """Test image_math (per-band expression) and image_band_math (multi-band to 1 band)."""
    in_img = tmp_path / "multi.tif"
    out_im = tmp_path / "out_im.tif"
    out_ibm = tmp_path / "out_ibm.tif"

    b1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b2 = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    b3 = np.array([[2.0, 2.0], [2.0, 2.0]], dtype=np.float32)
    stack = np.stack([b1, b2, b3])
    _create_test_raster(in_img, stack)

    rsgispy.imagecalc.image_math(
        input_img=str(in_img),
        output_img=str(out_im),
        exp="b1 * 10",
        gdalformat="GTIFF",
        datatype=rsgispy.TYPE_32FLOAT,
        exp_band_name=True,
        use_pbar=False,
    )
    with rasterio.open(out_im) as src:
        assert src.count == 3
        assert src.descriptions == ("b1 * 10", "b1 * 10", "b1 * 10")
        np.testing.assert_allclose(src.read(), stack * 10.0)

    # Update existing output_img with output_exists=True
    rsgispy.imagecalc.image_math(
        input_img=str(in_img),
        output_img=str(out_im),
        exp="b1 + 1",
        gdalformat="GTIFF",
        datatype=rsgispy.TYPE_32FLOAT,
        output_exists=True,
        use_pbar=False,
    )
    with rasterio.open(out_im) as src:
        np.testing.assert_allclose(src.read(), stack + 1.0)

    rsgispy.imagecalc.image_band_math(
        input_img=str(in_img),
        output_img=str(out_ibm),
        exp="(b1 + b2) / b3",
        gdalformat="GTIFF",
        datatype=rsgispy.TYPE_32FLOAT,
        exp_band_name=True,
        use_pbar=False,
    )
    with rasterio.open(out_ibm) as src:
        assert src.count == 1
        assert src.descriptions == ("(b1 + b2) / b3",)
        np.testing.assert_allclose(src.read(1), (b1 + b2) / b3)

    with pytest.raises(rsgispy.RSGISPyException):
        rsgispy.imagecalc.image_math(
            str(tmp_path / "missing.tif"), str(out_im), "b1", "GTIFF", rsgispy.TYPE_32FLOAT
        )
    with pytest.raises(rsgispy.RSGISPyException):
        rsgispy.imagecalc.image_band_math(
            str(tmp_path / "missing.tif"), str(out_ibm), "b1", "GTIFF", rsgispy.TYPE_32FLOAT
        )


def test_band_percentiles(tmp_path):
    """Test calc_band_percentile and calc_band_percentile_msk."""
    in_img = tmp_path / "vals.tif"
    msk_img = tmp_path / "msk.tif"

    b1 = np.array([[0.0, 10.0, 20.0], [30.0, 40.0, 50.0]], dtype=np.float32)
    b2 = np.array([[100.0, 200.0, 300.0], [400.0, 500.0, 0.0]], dtype=np.float32)
    _create_test_raster(in_img, np.stack([b1, b2]))

    msk = np.array([[1, 1, 1], [0, 1, 1]], dtype=np.uint8)
    _create_test_raster(msk_img, msk[np.newaxis, :, :], dtype="uint8")

    # Band 1 without 0.0: [10, 20, 30, 40, 50] -> median 30.0
    # Band 2 without 0.0: [100, 200, 300, 400, 500] -> median 300.0
    pcts = rsgispy.imagecalc.calc_band_percentile(str(in_img), 0.5, no_data_val=0.0)
    assert pcts == pytest.approx([30.0, 300.0])

    # All values without no_data_val
    pcts_all = rsgispy.imagecalc.calc_band_percentile(str(in_img), 0.5, no_data_val=None)
    assert pcts_all == pytest.approx([25.0, 250.0])

    with pytest.raises(rsgispy.RSGISPyException):
        rsgispy.imagecalc.calc_band_percentile(str(in_img), 1.5, no_data_val=None)

    # Masked pixels where msk == 1 and neither band == 0.0:
    # (0,1)->(10,200), (0,2)->(20,300), (1,1)->(40,500)
    msk_pcts = rsgispy.imagecalc.calc_band_percentile_msk(
        str(in_img),
        str(msk_img),
        msk_val=1,
        percentiles=[0.0, 50.0, 100.0],
        no_data_val=0.0,
        use_pbar=False,
    )
    assert msk_pcts.shape == (3, 2)
    np.testing.assert_allclose(msk_pcts[:, 0], [10.0, 20.0, 40.0])
    np.testing.assert_allclose(msk_pcts[:, 1], [200.0, 300.0, 500.0])

    with pytest.raises(rsgispy.RSGISPyException):
        rsgispy.imagecalc.calc_band_percentile_msk(
            str(in_img), str(msk_img), msk_val=99, percentiles=[50.0], use_pbar=False
        )


def test_imgs_and_band_pxl_percentiles(tmp_path):
    """Test calc_imgs_pxl_percentiles and calc_img_band_pxl_percentiles."""
    img1 = tmp_path / "img1.tif"
    img2 = tmp_path / "img2.tif"
    img_diff_bands = tmp_path / "img_1b.tif"
    out_all = tmp_path / "out_all.tif"
    out_per_band = tmp_path / "out_per_band.tif"

    a1 = np.array([[10.0, 0.0], [20.0, 30.0]], dtype=np.float32)
    a2 = np.array([[50.0, 60.0], [0.0, 70.0]], dtype=np.float32)
    b1 = np.array([[30.0, 40.0], [0.0, 50.0]], dtype=np.float32)
    b2 = np.array([[70.0, 80.0], [90.0, 100.0]], dtype=np.float32)
    _create_test_raster(img1, np.stack([a1, a2]))
    _create_test_raster(img2, np.stack([b1, b2]))
    _create_test_raster(img_diff_bands, a1[np.newaxis, :, :])

    rsgispy.imagecalc.calc_imgs_pxl_percentiles(
        input_imgs=[str(img1), str(img2)],
        percentiles=[0.0, 100.0],
        output_img=str(out_all),
        gdalformat="GTIFF",
        no_data_val=0.0,
        use_pbar=False,
    )
    with rasterio.open(out_all) as src:
        assert src.count == 2
        res_all = src.read()
        # Pixel (0,0): [10, 50, 30, 70] -> min=10, max=70
        # Pixel (0,1): [60, 40, 80] (0 ignored) -> min=40, max=80
        assert res_all[0, 0, 0] == pytest.approx(10.0)
        assert res_all[1, 0, 0] == pytest.approx(70.0)
        assert res_all[0, 0, 1] == pytest.approx(40.0)
        assert res_all[1, 0, 1] == pytest.approx(80.0)

    rsgispy.imagecalc.calc_img_band_pxl_percentiles(
        input_imgs=[str(img1), str(img2)],
        percentiles=[50.0, 100.0],
        output_img=str(out_per_band),
        gdalformat="GTIFF",
        no_data_val=0.0,
        use_pbar=False,
    )
    with rasterio.open(out_per_band) as src:
        assert src.count == 4
        res_pb = src.read()
        # Band 1 percentiles (bands 0, 1): pixel (0,0) -> [10, 30] -> 20, 30
        assert res_pb[0, 0, 0] == pytest.approx(20.0)
        assert res_pb[1, 0, 0] == pytest.approx(30.0)
        # Band 2 percentiles (bands 2, 3): pixel (1,0) -> [90] (0 ignored) -> 90, 90
        assert res_pb[2, 1, 0] == pytest.approx(90.0)
        assert res_pb[3, 1, 0] == pytest.approx(90.0)

    with pytest.raises(rsgispy.RSGISPyException):
        rsgispy.imagecalc.calc_img_band_pxl_percentiles(
            [str(img1), str(img_diff_bands)], [50.0], str(out_per_band), "GTIFF", use_pbar=False
        )
    with pytest.raises(rsgispy.RSGISPyException):
        rsgispy.imagecalc.calc_imgs_pxl_percentiles(
            [str(img1), str(img2)], [-1.0], str(out_all), "GTIFF", use_pbar=False
        )


def test_prop_true_count_and_unique(tmp_path):
    """Test calc_prop_true_exp, count_pxls_of_val, and get_unique_values."""
    img = tmp_path / "cls.tif"
    vld_img = tmp_path / "vld.tif"

    b1 = np.array([[1, 2, 3], [2, 1, 4]], dtype=np.int16)
    b2 = np.array([[5, 2, 1], [1, 3, 2]], dtype=np.int16)
    _create_test_raster(img, np.stack([b1, b2]), dtype="int16")

    vld = np.array([[1, 1, 0], [1, 1, 0]], dtype=np.uint8)
    _create_test_raster(vld_img, vld[np.newaxis, :, :], dtype="uint8")

    bdefs = [rsgispy.imagecalc.BandDefn("b1", str(img), 1)]
    prop_all = rsgispy.imagecalc.calc_prop_true_exp("b1 <= 2 ? 1 : 0", bdefs, use_pbar=False)
    assert prop_all == pytest.approx(4.0 / 6.0)

    # Inside vld == 1 (4 pixels: 1, 2, 2, 1): all <= 2 -> proportion 1.0
    prop_vld = rsgispy.imagecalc.calc_prop_true_exp(
        "b1 <= 2 ? 1 : 0", bdefs, in_vld_img=str(vld_img), use_pbar=False
    )
    assert prop_vld == pytest.approx(1.0)

    counts_b1 = rsgispy.imagecalc.count_pxls_of_val(
        str(img), vals=[1, 2, 99], img_band=1, use_pbar=False
    )
    assert counts_b1.dtype == np.int64
    np.testing.assert_array_equal(counts_b1, [2, 2, 0])

    counts_all = rsgispy.imagecalc.count_pxls_of_val(
        str(img), vals=[1, 2, 5], img_band=None, use_pbar=False
    )
    np.testing.assert_array_equal(counts_all, [4, 4, 1])

    uniq1 = rsgispy.imagecalc.get_unique_values(str(img), img_band=1)
    uniq2 = rsgispy.imagecalc.get_unique_values(str(img), img_band=2)
    np.testing.assert_array_equal(uniq1, [1, 2, 3, 4])
    np.testing.assert_array_equal(uniq2, [1, 2, 3, 5])

    with pytest.raises(rsgispy.RSGISPyException):
        rsgispy.imagecalc.count_pxls_of_val(str(img), vals=[], img_band=1, use_pbar=False)
    with pytest.raises(rsgispy.RSGISPyException):
        rsgispy.imagecalc.count_pxls_of_val(str(img), vals=[1], img_band=5, use_pbar=False)
    with pytest.raises(rsgispy.RSGISPyException):
        rsgispy.imagecalc.get_unique_values(str(img), img_band=5)
