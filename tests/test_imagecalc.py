"""
Tests for rsgispy and rsgispy.imagecalc.maths.band_maths.
"""

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

import rsgispy
import rsgispy.imagecalc.maths
import rsgispy.imagecalc.percentiles
import rsgispy.imagecalc.stats
import rsgispy.imageprocess
import rsgispy.imagecalc._norm_helpers as norm_helpers
import rsgispy.imageprocess as imgproc
from rasterio.transform import Affine
from pathlib import Path


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
    b1 = rsgispy.imagecalc.maths.BandDefn(band_name="b1", input_img="img.tif", img_band=1)
    b2 = rsgispy.imageprocess.BandDefn(band_name="b1", input_img="img.tif", img_band=1)
    b3 = rsgispy.imagecalc.maths.BandDefn(band_name="b2", input_img="img.tif", img_band=2)

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
    """Test basic two-band multiplication with band_maths."""
    in_img = tmp_path / "img.tif"
    out_img1 = tmp_path / "out1.tif"
    out_img2 = tmp_path / "out2.tif"

    b1_arr = np.arange(1, 26, dtype=np.float32).reshape(5, 5)
    b2_arr = np.full((5, 5), 3.5, dtype=np.float32)
    _create_test_raster(in_img, np.stack([b1_arr, b2_arr]))

    band_defns = [
        rsgispy.imagecalc.maths.BandDefn(band_name="b1", input_img=str(in_img), img_band=1),
        rsgispy.imagecalc.maths.BandDefn(band_name="b2", input_img=str(in_img), img_band=2),
    ]

    rsgispy.imagecalc.maths.band_maths(
        output_img=str(out_img1),
        exp="b1*b2",
        gdalformat="GTIFF",
        datatype=rsgispy.TYPE_32FLOAT,
        band_defs=band_defns,
    )
    rsgispy.imagecalc.maths.band_maths(
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
        rsgispy.imagecalc.maths.BandDefn(band_name="b1", input_img=str(in_img), img_band=1),
        rsgispy.imagecalc.maths.BandDefn(band_name="b2", input_img=str(in_img), img_band=2),
    ]

    rsgispy.imagecalc.maths.band_maths(
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
        rsgispy.imagecalc.maths.BandDefn(band_name="b1", input_img=str(img1), img_band=1),
        rsgispy.imagecalc.maths.BandDefn(band_name="b2", input_img=str(img2), img_band=1),
    ]

    rsgispy.imagecalc.maths.band_maths(
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
        rsgispy.imagecalc.maths.BandDefn(band_name="b1", input_img=str(in_img), img_band=1)
    ]

    rsgispy.imagecalc.maths.band_maths(
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
        rsgispy.imagecalc.maths.band_maths(
            str(out_img), "b1", [], gdalformat="GTIFF", datatype=rsgispy.TYPE_32FLOAT
        )

    # Non-existent file
    with pytest.raises(rsgispy.RSGISPyException):
        rsgispy.imagecalc.maths.band_maths(
            str(out_img),
            "b1",
            [rsgispy.imagecalc.maths.BandDefn("b1", str(tmp_path / "missing.tif"), 1)],
            gdalformat="GTIFF",
            datatype=rsgispy.TYPE_32FLOAT,
        )

    # Out-of-range band index
    with pytest.raises(rsgispy.RSGISPyException):
        rsgispy.imagecalc.maths.band_maths(
            str(out_img),
            "b1",
            [rsgispy.imagecalc.maths.BandDefn("b1", str(in_img), 5)],
            gdalformat="GTIFF",
            datatype=rsgispy.TYPE_32FLOAT,
        )

    # Invalid expression / undefined variable
    with pytest.raises(rsgispy.RSGISPyException):
        rsgispy.imagecalc.maths.band_maths(
            str(out_img),
            "b1 + unknown_var",
            [rsgispy.imagecalc.maths.BandDefn("b1", str(in_img), 1)],
            gdalformat="GTIFF",
            datatype=rsgispy.TYPE_32FLOAT,
        )

    # Non-overlapping rasters
    with pytest.raises(rsgispy.RSGISPyException):
        rsgispy.imagecalc.maths.band_maths(
            str(out_img),
            "b1 + b2",
            [
                rsgispy.imagecalc.maths.BandDefn("b1", str(in_img), 1),
                rsgispy.imagecalc.maths.BandDefn("b2", str(no_overlap_img), 1),
            ],
            gdalformat="GTIFF",
            datatype=rsgispy.TYPE_32FLOAT,
        )

    # Different pixel resolutions
    with pytest.raises(rsgispy.RSGISPyException):
        rsgispy.imagecalc.maths.band_maths(
            str(out_img),
            "b1 + b2",
            [
                rsgispy.imagecalc.maths.BandDefn("b1", str(in_img), 1),
                rsgispy.imagecalc.maths.BandDefn("b2", str(diff_res_img), 1),
            ],
            gdalformat="GTIFF",
            datatype=rsgispy.TYPE_32FLOAT,
        )


def test_tqdm_progress_bar_and_notebook_detection(tmp_path, capsys, monkeypatch):
    """Test tqdm progress bar suppression (use_pbar=False) and notebook detection."""
    in_img = tmp_path / "img.tif"
    out_img_quiet = tmp_path / "out_quiet.tif"
    out_img_pbar = tmp_path / "out_pbar.tif"

    _create_test_raster(in_img, np.ones((1, 5, 5), dtype=np.float32))
    band_defns = [
        rsgispy.imagecalc.maths.BandDefn(band_name="b1", input_img=str(in_img), img_band=1)
    ]

    assert rsgispy.is_notebook() is False

    # Suppress progress bar with use_pbar=False
    capsys.readouterr()
    rsgispy.imagecalc.maths.band_maths(
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
    rsgispy.imagecalc.maths.band_maths(
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

    rsgispy.imagecalc.maths.image_math(
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
    rsgispy.imagecalc.maths.image_math(
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

    rsgispy.imagecalc.maths.image_band_math(
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
        rsgispy.imagecalc.maths.image_math(
            str(tmp_path / "missing.tif"),
            str(out_im),
            "b1",
            gdalformat="GTIFF",
            datatype=rsgispy.TYPE_32FLOAT,
        )
    with pytest.raises(rsgispy.RSGISPyException):
        rsgispy.imagecalc.maths.image_band_math(
            str(tmp_path / "missing.tif"),
            str(out_ibm),
            "b1",
            gdalformat="GTIFF",
            datatype=rsgispy.TYPE_32FLOAT,
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
    # Percentiles are in [0, 100] (as calc_band_percentile_msk and
    # calc_imgs_pxl_percentiles), computed block by block.
    pcts = rsgispy.imagecalc.percentiles.calc_band_percentile(str(in_img), 50.0, no_data_val=0.0)
    assert pcts == pytest.approx([30.0, 300.0])

    # All values without no_data_val
    pcts_all = rsgispy.imagecalc.percentiles.calc_band_percentile(
        str(in_img), 50.0, no_data_val=None
    )
    assert pcts_all == pytest.approx([25.0, 250.0])

    with pytest.raises(rsgispy.RSGISPyException):
        rsgispy.imagecalc.percentiles.calc_band_percentile(str(in_img), 150.0, no_data_val=None)
    # A value in (0, 1) now means a percentile below 1: warn about the units.
    with pytest.warns(UserWarning, match="0-100"):
        rsgispy.imagecalc.percentiles.calc_band_percentile(str(in_img), 0.5, no_data_val=None)

    # Masked pixels where msk == 1 and neither band == 0.0:
    # (0,1)->(10,200), (0,2)->(20,300), (1,1)->(40,500)
    msk_pcts = rsgispy.imagecalc.percentiles.calc_band_percentile_msk(
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
        rsgispy.imagecalc.percentiles.calc_band_percentile_msk(
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

    rsgispy.imagecalc.percentiles.calc_imgs_pxl_percentiles(
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

    rsgispy.imagecalc.percentiles.calc_img_band_pxl_percentiles(
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
        rsgispy.imagecalc.percentiles.calc_img_band_pxl_percentiles(
            [str(img1), str(img_diff_bands)],
            str(out_per_band),
            [50.0],
            gdalformat="GTIFF",
            use_pbar=False,
        )
    with pytest.raises(rsgispy.RSGISPyException):
        rsgispy.imagecalc.percentiles.calc_imgs_pxl_percentiles(
            [str(img1), str(img2)],
            str(out_all),
            [-1.0],
            gdalformat="GTIFF",
            use_pbar=False,
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

    bdefs = [rsgispy.imagecalc.maths.BandDefn("b1", str(img), 1)]
    prop_all = rsgispy.imagecalc.maths.calc_prop_true_exp("b1 <= 2 ? 1 : 0", bdefs, use_pbar=False)
    assert prop_all == pytest.approx(4.0 / 6.0)

    # Inside vld == 1 (4 pixels: 1, 2, 2, 1): all <= 2 -> proportion 1.0
    prop_vld = rsgispy.imagecalc.maths.calc_prop_true_exp(
        "b1 <= 2 ? 1 : 0", bdefs, in_vld_img=str(vld_img), use_pbar=False
    )
    assert prop_vld == pytest.approx(1.0)

    counts_b1 = rsgispy.imagecalc.stats.count_pxls_of_val(
        str(img), vals=[1, 2, 99], img_band=1, use_pbar=False
    )
    assert counts_b1.dtype == np.int64
    np.testing.assert_array_equal(counts_b1, [2, 2, 0])

    counts_all = rsgispy.imagecalc.stats.count_pxls_of_val(
        str(img), vals=[1, 2, 5], img_band=None, use_pbar=False
    )
    np.testing.assert_array_equal(counts_all, [4, 4, 1])

    uniq1 = rsgispy.imagecalc.stats.get_unique_values(str(img), img_band=1)
    uniq2 = rsgispy.imagecalc.stats.get_unique_values(str(img), img_band=2)
    np.testing.assert_array_equal(uniq1, [1, 2, 3, 4])
    np.testing.assert_array_equal(uniq2, [1, 2, 3, 5])

    with pytest.raises(rsgispy.RSGISPyException):
        rsgispy.imagecalc.stats.count_pxls_of_val(str(img), vals=[], img_band=1, use_pbar=False)
    with pytest.raises(rsgispy.RSGISPyException):
        rsgispy.imagecalc.stats.count_pxls_of_val(str(img), vals=[1], img_band=5, use_pbar=False)
    with pytest.raises(rsgispy.RSGISPyException):
        rsgispy.imagecalc.stats.get_unique_values(str(img), img_band=5)


def test_band_maths_multi_output(tmp_path):
    """Test multi-output comma-separated pymu-eval expressions in band_maths and image_band_math."""
    in_img = tmp_path / "multi_in.tif"
    out_img = tmp_path / "multi_out.tif"
    out_exp_names = tmp_path / "multi_exp_names.tif"
    out_ibm = tmp_path / "multi_ibm.tif"

    green = np.array([[0.1, 0.3, 0.0], [0.2, 0.4, 0.5]], dtype=np.float32)
    red = np.array([[0.2, 0.2, 0.0], [0.3, 0.1, 0.4]], dtype=np.float32)
    nir = np.array([[0.8, 0.6, 0.0], [0.5, 0.7, 0.6]], dtype=np.float32)
    _create_test_raster(in_img, np.stack([green, red, nir]))

    band_defs = [
        rsgispy.imagecalc.maths.BandDefn("green", str(in_img), 1),
        rsgispy.imagecalc.maths.BandDefn("red", str(in_img), 2),
        rsgispy.imagecalc.maths.BandDefn("nir", str(in_img), 3),
    ]

    # 1. Multi-line expression with conditionals, multi-arg functions, and a scalar constant
    multi_exp = """
        (nir + red) != 0 ? (nir - red) / (nir + red) : -9999,
        min(green, red, nir),
        max(green, red, nir),
        2 * pi
    """
    rsgispy.imagecalc.maths.band_maths(
        output_img=str(out_img),
        exp=multi_exp,
        band_defs=band_defs,
        band_names=["NDVI", "min_val", "max_val", "two_pi"],
        gdalformat="GTIFF",
        datatype=rsgispy.TYPE_32FLOAT,
        use_pbar=False,
    )

    exp_ndvi = np.where((nir + red) != 0, (nir - red) / (nir + red), -9999.0)
    exp_min = np.minimum(np.minimum(green, red), nir)
    exp_max = np.maximum(np.maximum(green, red), nir)
    exp_two_pi = np.full_like(green, 2.0 * np.pi)

    with rasterio.open(out_img) as src:
        assert src.count == 4
        assert src.descriptions == ("NDVI", "min_val", "max_val", "two_pi")
        np.testing.assert_allclose(src.read(1), exp_ndvi, rtol=1e-5)
        np.testing.assert_allclose(src.read(2), exp_min, rtol=1e-5)
        np.testing.assert_allclose(src.read(3), exp_max, rtol=1e-5)
        np.testing.assert_allclose(src.read(4), exp_two_pi, rtol=1e-5)

    # 2. exp_band_name=True splits top-level sub-expressions while preserving inner commas
    rsgispy.imagecalc.maths.band_maths(
        output_img=str(out_exp_names),
        exp="min(green, red, nir), avg(green, red, nir)",
        band_defs=band_defs,
        exp_band_name=True,
        use_pbar=False,
    )
    with rasterio.open(out_exp_names) as src:
        assert src.count == 2
        assert src.descriptions == ("min(green, red, nir)", "avg(green, red, nir)")
        np.testing.assert_allclose(src.read(1), exp_min, rtol=1e-5)
        np.testing.assert_allclose(src.read(2), (green + red + nir) / 3.0, rtol=1e-5)

    # 3. image_band_math forwards multi-output and band_names
    rsgispy.imagecalc.maths.image_band_math(
        input_img=str(in_img),
        output_img=str(out_ibm),
        exp="b3 + b2, b3 - b2",
        band_names=["sum_nr", "diff_nr"],
        use_pbar=False,
    )
    with rasterio.open(out_ibm) as src:
        assert src.count == 2
        assert src.descriptions == ("sum_nr", "diff_nr")
        np.testing.assert_allclose(src.read(1), nir + red, rtol=1e-5)
        np.testing.assert_allclose(src.read(2), nir - red, rtol=1e-5)

    # 4. Multi-block (>256x256) with variable re-assignment across outputs & output_exists
    large_in = tmp_path / "large_multi_in.tif"
    large_out = tmp_path / "large_multi_out.tif"
    b_nir = np.linspace(2.0, 10.0, 300 * 300, dtype=np.float32).reshape(300, 300)
    b_red = np.linspace(1.0, 5.0, 300 * 300, dtype=np.float32).reshape(300, 300)
    _create_test_raster(large_in, np.stack([b_nir, b_red]))
    _create_test_raster(large_out, np.zeros((3, 300, 300), dtype=np.float32))

    large_defs = [
        rsgispy.imagecalc.maths.BandDefn("nir", str(large_in), 1),
        rsgispy.imagecalc.maths.BandDefn("red", str(large_in), 2),
    ]
    rsgispy.imagecalc.maths.band_maths(
        output_img=str(large_out),
        exp="nir = nir * 0.5, red = red * 0.5, (nir - red) / (nir + red)",
        band_defs=large_defs,
        output_exists=True,
        band_names=["nir_half", "red_half", "ndvi"],
        use_pbar=False,
    )
    with rasterio.open(large_out) as src:
        assert src.count == 3
        np.testing.assert_allclose(src.read(1), b_nir * 0.5, rtol=1e-5)
        np.testing.assert_allclose(src.read(2), b_red * 0.5, rtol=1e-5)
        np.testing.assert_allclose(
            src.read(3), (b_nir - b_red) / (b_nir + b_red), rtol=1e-5
        )

    # 5. Validation errors: mismatched output_exists band count and bad band_names
    with pytest.raises(rsgispy.RSGISPyException, match="band\\(s\\)"):
        rsgispy.imagecalc.maths.band_maths(
            output_img=str(out_ibm),
            exp="green, red, nir",
            band_defs=band_defs,
            output_exists=True,
            use_pbar=False,
        )
    with pytest.raises(rsgispy.RSGISPyException, match="band_names"):
        rsgispy.imagecalc.maths.band_maths(
            output_img=str(tmp_path / "bad_names.tif"),
            exp="green + red, nir - red",
            band_defs=band_defs,
            band_names=["only_one"],
            use_pbar=False,
        )
    with pytest.raises(rsgispy.RSGISPyException, match="non-empty"):
        rsgispy.imagecalc.maths.band_maths(
            output_img=str(tmp_path / "empty_name.tif"),
            exp="green + red, nir - red",
            band_defs=band_defs,
            band_names=["ok", "  "],
            use_pbar=False,
        )


def test_band_maths_intermediate_variables(tmp_path):
    """Test creating and reusing new intermediate variables in band_maths and siblings."""
    in_img = tmp_path / "s2_tiled.tif"
    out_img = tmp_path / "interm_multi.tif"
    out_single = tmp_path / "interm_single.tif"
    out_ibm = tmp_path / "interm_ibm.tif"
    out_im = tmp_path / "interm_im.tif"

    # Non-multiple of 256 so full tiles and smaller remainder edge tiles are tested
    rng = np.random.default_rng(42)
    red = rng.uniform(0.05, 0.4, size=(270, 260)).astype(np.float32)
    nir = rng.uniform(0.1, 0.9, size=(270, 260)).astype(np.float32)
    swir1 = rng.uniform(0.05, 0.5, size=(270, 260)).astype(np.float32)
    _create_test_raster(in_img, np.stack([red, nir, swir1]))

    band_defs = [
        rsgispy.imagecalc.maths.BandDefn("red", str(in_img), 1),
        rsgispy.imagecalc.maths.BandDefn("nir", str(in_img), 2),
        rsgispy.imagecalc.maths.BandDefn("swir1", str(in_img), 3),
    ]

    # 1. Exact multi-output pattern defining new intermediate `ndvi` and reusing it
    exp = (
        "ndvi = (nir - red) / (nir + red + 1e-6), "
        "(nir - swir1) / (nir + swir1 + 1e-6), "
        "ndvi >= 0.5 ? 1 : 0"
    )
    rsgispy.imagecalc.maths.band_maths(
        output_img=str(out_img),
        exp=exp,
        band_defs=band_defs,
        band_names=["NDVI", "NDMI", "VegMask"],
        gdalformat="GTIFF",
        datatype=rsgispy.TYPE_32FLOAT,
        use_pbar=False,
    )

    exp_ndvi = (nir - red) / (nir + red + 1e-6)
    exp_ndmi = (nir - swir1) / (nir + swir1 + 1e-6)
    exp_mask = np.where(exp_ndvi >= 0.5, 1.0, 0.0)

    with rasterio.open(out_img) as src:
        assert src.count == 3
        assert src.descriptions == ("NDVI", "NDMI", "VegMask")
        np.testing.assert_allclose(src.read(1), exp_ndvi, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(src.read(2), exp_ndmi, rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(src.read(3), exp_mask)

    # 2. Inline parenthesized assignment producing a single output band
    rsgispy.imagecalc.maths.band_maths(
        output_img=str(out_single),
        exp="(ndvi = (nir - red) / (nir + red + 1e-6)) >= 0.5 ? 1 : 0",
        band_defs=band_defs[:2],
        use_pbar=False,
    )
    with rasterio.open(out_single) as src:
        assert src.count == 1
        np.testing.assert_array_equal(src.read(1), exp_mask)

    # 3. image_band_math, image_math, and calc_prop_true_exp with intermediate variables
    rsgispy.imagecalc.maths.image_band_math(
        input_img=str(in_img),
        output_img=str(out_ibm),
        exp="ndvi = (b2 - b1) / (b2 + b1 + 1e-6), ndvi >= 0.5 ? 1 : 0",
        band_names=["NDVI", "VegMask"],
        use_pbar=False,
    )
    with rasterio.open(out_ibm) as src:
        assert src.count == 2
        np.testing.assert_allclose(src.read(1), exp_ndvi, rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(src.read(2), exp_mask)

    rsgispy.imagecalc.maths.image_math(
        input_img=str(in_img),
        output_img=str(out_im),
        exp="scaled = b1 * 2.0, scaled + 1.0",
        use_pbar=False,
    )
    with rasterio.open(out_im) as src:
        assert src.count == 3
        np.testing.assert_allclose(src.read(1), red * 2.0 + 1.0, rtol=1e-5)
        np.testing.assert_allclose(src.read(2), nir * 2.0 + 1.0, rtol=1e-5)

    prop = rsgispy.imagecalc.maths.calc_prop_true_exp(
        exp="ndvi = (nir - red) / (nir + red + 1e-6), ndvi >= 0.5 ? 1 : 0",
        band_defs=band_defs[:2],
        use_pbar=False,
    )
    assert prop == pytest.approx(float(np.mean(exp_mask)))

    # 4. Undefined identifiers on RHS, read-before-assign, and initial self-reference
    #    must still raise RSGISPyException rather than defaulting to zero
    for bad_exp in [
        "ndvi = (nir - unknown_band) / (nir + red + 1e-6), ndvi >= 0.5 ? 1 : 0",
        "ndvi + 1, ndvi = nir - red",
        "ndvi = ndvi + 1",
    ]:
        with pytest.raises(rsgispy.RSGISPyException):
            rsgispy.imagecalc.maths.band_maths(
                output_img=str(tmp_path / "bad_interm.tif"),
                exp=bad_exp,
                band_defs=band_defs,
                use_pbar=False,
            )


def test_band_maths_no_data_val_and_calc_stats(tmp_path):
    """band_maths, image_band_math and image_math write no_data_val and stats."""
    in_img = tmp_path / "nd_in.tif"
    b1 = np.array([[0.0, 2.0, 4.0], [6.0, 0.0, 10.0]], dtype=np.float32)
    b2 = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=np.float32)
    _create_test_raster(in_img, np.stack([b1, b2]))
    band_defs = [
        rsgispy.imagecalc.maths.BandDefn("b1", str(in_img), 1),
        rsgispy.imagecalc.maths.BandDefn("b2", str(in_img), 2),
    ]

    # Defaults are unchanged: no output no data value and no statistics.
    out_def = tmp_path / "nd_default.tif"
    rsgispy.imagecalc.maths.band_maths(
        str(out_def), "b1 == 0 ? -9999 : b1 * b2", band_defs, use_pbar=False
    )
    with rasterio.open(out_def) as src:
        assert src.nodata is None
        assert "STATISTICS_MINIMUM" not in src.tags(1)

    # Single output band: no data written and excluded from the statistics.
    out_single = tmp_path / "nd_single.tif"
    rsgispy.imagecalc.maths.band_maths(
        str(out_single),
        "b1 == 0 ? -9999 : b1 * b2",
        band_defs,
        no_data_val=-9999,
        calc_stats=True,
        use_pbar=False,
    )
    with rasterio.open(out_single) as src:
        assert src.nodata == -9999.0
        tags = src.tags(1)
        np.testing.assert_allclose(
            src.read(1), np.where(b1 == 0, -9999.0, b1 * b2)
        )
    assert float(tags["STATISTICS_MINIMUM"]) == pytest.approx(2.0)
    assert float(tags["STATISTICS_MAXIMUM"]) == pytest.approx(20.0)

    # Updating an existing output can change its no data value.
    rsgispy.imagecalc.maths.band_maths(
        str(out_single),
        "b1 == 0 ? -1 : b1 * b2",
        band_defs,
        output_exists=True,
        no_data_val=-1,
        use_pbar=False,
    )
    with rasterio.open(out_single) as src:
        assert src.nodata == -1.0

    # Multi-output expression: every band gets the no data value.
    out_multi = tmp_path / "nd_multi.tif"
    rsgispy.imagecalc.maths.band_maths(
        str(out_multi),
        "b1 == 0 ? 255 : b1, b1 == 0 ? 255 : b2 * 10",
        band_defs,
        band_names=["b1_vld", "b2_x10"],
        datatype=rsgispy.TYPE_8UINT,
        no_data_val=255,
        calc_stats=True,
        use_pbar=False,
    )
    with rasterio.open(out_multi) as src:
        assert src.nodatavals == (255.0, 255.0)
        assert float(src.tags(1)["STATISTICS_MINIMUM"]) == pytest.approx(2.0)
        assert float(src.tags(1)["STATISTICS_MAXIMUM"]) == pytest.approx(10.0)
        assert float(src.tags(2)["STATISTICS_MINIMUM"]) == pytest.approx(10.0)
        assert float(src.tags(2)["STATISTICS_MAXIMUM"]) == pytest.approx(20.0)

    # image_band_math forwards both options.
    out_ibm = tmp_path / "nd_ibm.tif"
    rsgispy.imagecalc.maths.image_band_math(
        str(in_img),
        str(out_ibm),
        "b1 == 0 ? -9999 : b1 / b2, b1 == 0 ? -9999 : b2",
        no_data_val=-9999,
        calc_stats=True,
        use_pbar=False,
    )
    with rasterio.open(out_ibm) as src:
        assert src.nodatavals == (-9999.0, -9999.0)
        assert float(src.tags(1)["STATISTICS_MINIMUM"]) == pytest.approx(2.0)
        assert float(src.tags(2)["STATISTICS_MAXIMUM"]) == pytest.approx(2.0)

    # image_math applies the expression and the no data value to every band.
    out_im = tmp_path / "nd_im.tif"
    rsgispy.imagecalc.maths.image_math(
        str(in_img),
        str(out_im),
        "b1 == 0 ? -1 : b1 + 1",
        no_data_val=-1,
        calc_stats=True,
        use_pbar=False,
    )
    with rasterio.open(out_im) as src:
        assert src.nodatavals == (-1.0, -1.0)
        assert float(src.tags(1)["STATISTICS_MINIMUM"]) == pytest.approx(3.0)
        assert float(src.tags(2)["STATISTICS_MINIMUM"]) == pytest.approx(2.0)

    # A no data value that the output data type cannot hold is an error.
    with pytest.raises(rsgispy.RSGISPyException):
        rsgispy.imagecalc.maths.band_maths(
            str(tmp_path / "nd_bad.tif"),
            "b1",
            band_defs,
            datatype=rsgispy.TYPE_8UINT,
            no_data_val=-9999,
            use_pbar=False,
        )
    with pytest.raises(rsgispy.RSGISPyException):
        rsgispy.imagecalc.maths.image_math(
            str(in_img),
            str(tmp_path / "nd_bad_im.tif"),
            "b1",
            datatype=rsgispy.TYPE_16UINT,
            no_data_val=0.5,
            use_pbar=False,
        )


def test_calc_band_percentile_block_by_block(tmp_path, monkeypatch):
    """calc_band_percentile reads windows, with the same values as a full read."""
    in_img = tmp_path / "big.tif"
    rng = np.random.default_rng(4)
    arr = rng.normal(100.0, 20.0, size=(2, 40, 37)).astype(np.float32)
    arr[0, :3, :] = -1.0
    _create_test_raster(in_img, arr)
    monkeypatch.setenv("RSGISPY_BLOCK_SIZE", "8")
    full_reads = []
    real_read = rasterio.io.DatasetReader.read

    def _spy(self, *args, **kwargs):
        if kwargs.get("window") is None:
            full_reads.append(args)
        return real_read(self, *args, **kwargs)

    monkeypatch.setattr(rasterio.io.DatasetReader, "read", _spy)
    pcts = rsgispy.imagecalc.percentiles.calc_band_percentile(
        str(in_img), 90.0, no_data_val=-1.0
    )
    assert not full_reads
    b0 = arr[0][arr[0] != -1.0].astype(np.float64)
    assert pcts == pytest.approx([np.percentile(b0, 90.0),
                                  np.percentile(arr[1].astype(np.float64), 90.0)])


def _write_raster(
    path: Path,
    arr: np.ndarray,
    *,
    transform: Affine = from_origin(500000.0, 5800000.0, 10.0, 10.0),
    crs: object = "EPSG:32630",
    nodata=None,
) -> str:
    data = arr[np.newaxis, :, :] if arr.ndim == 2 else arr
    profile = {
        "driver": "GTiff",
        "height": int(data.shape[1]),
        "width": int(data.shape[2]),
        "count": int(data.shape[0]),
        "dtype": str(data.dtype),
        "crs": crs,
        "transform": transform,
    }
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
    return str(path)


def test_subsample_band_values_tiled_reservoir(tmp_path, monkeypatch):
    monkeypatch.setenv(imgproc.ENV_VAR_BLOCK_SIZE, "8")
    arr = np.arange(1, 20 * 20 + 1, dtype=np.float32).reshape(20, 20)
    arr[0, 0] = -999.0
    arr[1, 1] = np.nan
    in_img = _write_raster(tmp_path / "res.tif", arr, nodata=-999.0)

    with rasterio.open(in_img) as src:
        s1 = norm_helpers.subsample_band_values(
            src, band_idx=1, eff_nd=-999.0, max_samples=50, rnd_seed=42
        )
        s2 = norm_helpers.subsample_band_values(
            src, band_idx=1, eff_nd=-999.0, max_samples=50, rnd_seed=42
        )
        assert s1.shape == (50,)
        np.testing.assert_array_equal(s1, s2)
        assert np.all(np.isfinite(s1))
        assert not np.any(s1 == -999.0)
        # Samples should span across multiple tiles (not just first 50 pixels)
        assert float(np.max(s1)) > 100.0
