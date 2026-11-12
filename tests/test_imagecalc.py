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

