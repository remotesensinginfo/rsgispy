"""
Tests for rsgispy.imageutils image creation option helpers, image statistics,
thematic colour tables, overview pyramids, and integration with band_maths.
"""

import numpy as np
import pytest
import rasterio
from rasterio.enums import ColorInterp, Compression
from rasterio.transform import from_origin

import rsgispy
import rsgispy.imagecalc
import rsgispy.imageutils


def _create_simple_raster(path, data, dtype="float32"):
    """Write a single-band or multi-band test GeoTIFF."""
    if data.ndim == 2:
        data = data[np.newaxis, :, :]
    count, height, width = data.shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=dtype,
        crs="EPSG:4326",
        transform=from_origin(0.0, float(height), 1.0, 1.0),
    ) as dst:
        dst.write(data.astype(dtype))


def test_env_vars_gtiff_and_cog_helpers(monkeypatch):
    """Test setting and retrieving GTiff and COG creation options via environment variables."""
    monkeypatch.delenv("RSGISLIB_IMG_CRT_OPTS_GTIFF", raising=False)
    monkeypatch.delenv("RSGISPY_IMG_CRT_OPTS_GTIFF", raising=False)
    monkeypatch.delenv("RSGISLIB_IMG_CRT_OPTS_COG", raising=False)
    monkeypatch.delenv("RSGISPY_IMG_CRT_OPTS_COG", raising=False)

    assert rsgispy.imageutils.get_gdal_img_creation_opts("GTIFF") == {}

    rsgispy.imageutils.set_env_vars_deflate_gtiff_outs(bigtiff=False)
    opts = rsgispy.imageutils.get_gdal_img_creation_opts("GTIFF")
    assert opts == {
        "TILED": "YES",
        "COMPRESS": "DEFLATE",
        "INTERLEAVE": "BAND",
        "BIGTIFF": "IF_SAFER",
    }

    rsgispy.imageutils.set_env_vars_lzw_gtiff_outs(bigtiff=True)
    opts = rsgispy.imageutils.get_gdal_img_creation_opts("gtiff")
    assert opts == {
        "TILED": "YES",
        "COMPRESS": "LZW",
        "INTERLEAVE": "BAND",
        "BIGTIFF": "YES",
    }

    rsgispy.imageutils.set_env_vars_zstd_gtiff_outs(bigtiff=False)
    opts = rsgispy.imageutils.get_gdal_img_creation_opts("GTIFF")
    assert opts["COMPRESS"] == "ZSTD"
    assert opts["BIGTIFF"] == "IF_SAFER"

    rsgispy.imageutils.set_env_vars_lzw_cog_outs(bigtiff=False)
    assert rsgispy.imageutils.get_gdal_img_creation_opts("COG") == {
        "COMPRESS": "LZW",
        "BIGTIFF": "IF_SAFER",
    }

    rsgispy.imageutils.set_env_vars_deflate_cog_outs(bigtiff=True)
    assert rsgispy.imageutils.get_gdal_img_creation_opts("COG") == {
        "COMPRESS": "DEFLATE",
        "BIGTIFF": "YES",
    }

    rsgispy.imageutils.set_img_creation_opts(
        "GTIFF", {"COMPRESS": "DEFLATE", "PREDICTOR": 2}
    )
    assert rsgispy.imageutils.get_gdal_img_creation_opts("GTIFF") == {
        "COMPRESS": "DEFLATE",
        "PREDICTOR": "2",
    }


def test_band_maths_with_creation_options(tmp_path, monkeypatch):
    """Test band_maths applies both env var creation options and explicit creation_opts."""
    monkeypatch.delenv("RSGISLIB_IMG_CRT_OPTS_GTIFF", raising=False)
    monkeypatch.delenv("RSGISPY_IMG_CRT_OPTS_GTIFF", raising=False)

    in_img = tmp_path / "in.tif"
    out_deflate = tmp_path / "out_deflate.tif"
    out_lzw = tmp_path / "out_lzw.tif"

    data = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
    _create_simple_raster(in_img, data)
    band_defs = [
        rsgispy.imagecalc.BandDefn(band_name="b1", input_img=str(in_img), img_band=1)
    ]

    # 1. Via set_env_vars_deflate_gtiff_outs
    rsgispy.imageutils.set_env_vars_deflate_gtiff_outs()
    rsgispy.imagecalc.band_maths(
        output_img=str(out_deflate),
        exp="b1 * 2.0",
        gdalformat="GTIFF",
        datatype=rsgispy.TYPE_32FLOAT,
        band_defs=band_defs,
        use_pbar=False,
    )

    with rasterio.open(out_deflate) as src:
        assert src.compression == Compression.deflate
        assert src.profile["tiled"] is True
        np.testing.assert_allclose(src.read(1), data * 2.0)

    # 2. Via explicit creation_opts overriding env vars
    rsgispy.imagecalc.band_maths(
        output_img=str(out_lzw),
        exp="b1 + 5.0",
        gdalformat="GTIFF",
        datatype=rsgispy.TYPE_32FLOAT,
        band_defs=band_defs,
        use_pbar=False,
        creation_opts={"COMPRESS": "LZW", "TILED": "YES"},
    )

    with rasterio.open(out_lzw) as src:
        assert src.compression == Compression.lzw
        assert src.profile["tiled"] is True
        np.testing.assert_allclose(src.read(1), data + 5.0)


def test_pop_thmt_img_stats(tmp_path):
    """Test pop_thmt_img_stats on uint8, uint16, and int16 thematic rasters."""
    thmt_u8_img = tmp_path / "thematic_u8.tif"
    thmt_u16_img = tmp_path / "thematic_u16.tif"
    thmt_i16_img = tmp_path / "thematic_i16.tif"
    float_img = tmp_path / "float.tif"

    # 1. uint8 thematic image (200x200 so overview level 4 is built)
    arr_u8 = np.zeros((200, 200), dtype=np.uint8)
    arr_u8[10:100, 10:100] = 1
    arr_u8[100:190, 100:190] = 2
    arr_u8[50:80, 120:160] = 3
    _create_simple_raster(thmt_u8_img, arr_u8, dtype="uint8")

    rsgispy.imageutils.pop_thmt_img_stats(
        str(thmt_u8_img),
        add_clr_tab=True,
        calc_pyramids=True,
        ignore_zero=True,
        use_pbar=False,
    )

    with rasterio.open(thmt_u8_img) as src:
        tags = src.tags(1)
        assert tags["LAYER_TYPE"] == "thematic"
        assert tags["STATISTICS_MINIMUM"] == "1"
        assert tags["STATISTICS_MAXIMUM"] == "3"
        assert tags["STATISTICS_EXCLUDEDVALUES"] == "0"
        assert tags["STATISTICS_HISTOBINFUNCTION"] == "direct"
        assert tags["STATISTICS_HISTONUMBINS"] == "256"
        assert src.nodata == 0.0
        assert src.colorinterp == (ColorInterp.palette,)
        cmap = src.colormap(1)
        assert cmap[0] == (0, 0, 0, 0)
        assert cmap[1][3] == 255
        assert src.overviews(1) == [4]

    # 2. uint16 thematic image with class ID > 255
    arr_u16 = np.zeros((200, 200), dtype=np.uint16)
    arr_u16[10:100, 10:100] = 5
    arr_u16[100:190, 100:190] = 300
    _create_simple_raster(thmt_u16_img, arr_u16, dtype="uint16")

    rsgispy.imageutils.pop_thmt_img_stats(
        str(thmt_u16_img),
        add_clr_tab=True,
        calc_pyramids=True,
        ignore_zero=True,
        use_pbar=False,
    )

    with rasterio.open(thmt_u16_img) as src:
        tags = src.tags(1)
        assert tags["LAYER_TYPE"] == "thematic"
        assert tags["STATISTICS_MINIMUM"] == "5"
        assert tags["STATISTICS_MAXIMUM"] == "300"
        assert tags["STATISTICS_HISTOMIN"] == "0"
        assert tags["STATISTICS_HISTOMAX"] == "300"
        assert tags["STATISTICS_HISTONUMBINS"] == "301"
        assert src.nodata == 0.0
        assert src.colorinterp == (ColorInterp.palette,)
        cmap = src.colormap(1)
        assert cmap[300][3] == 255
        assert src.overviews(1) == [4]

    # 3. int16 thematic image
    arr_i16 = np.zeros((200, 200), dtype=np.int16)
    arr_i16[10:100, 10:100] = 2
    arr_i16[100:190, 100:190] = 450
    _create_simple_raster(thmt_i16_img, arr_i16, dtype="int16")

    rsgispy.imageutils.pop_thmt_img_stats(
        str(thmt_i16_img),
        add_clr_tab=True,
        calc_pyramids=True,
        ignore_zero=True,
        use_pbar=False,
    )

    with rasterio.open(thmt_i16_img) as src:
        tags = src.tags(1)
        assert tags["LAYER_TYPE"] == "thematic"
        assert tags["STATISTICS_MINIMUM"] == "2"
        assert tags["STATISTICS_MAXIMUM"] == "450"
        assert tags["STATISTICS_HISTOMIN"] == "0"
        assert tags["STATISTICS_HISTOMAX"] == "450"
        assert tags["STATISTICS_HISTONUMBINS"] == "451"
        assert src.nodata == 0.0
        assert src.overviews(1) == [4]

    # 4. Non-integer dataset should raise RSGISPyException
    _create_simple_raster(float_img, np.ones((10, 10), dtype=np.float32), dtype="float32")
    with pytest.raises(rsgispy.RSGISPyException):
        rsgispy.imageutils.pop_thmt_img_stats(str(float_img), use_pbar=False)


def test_pop_img_stats(tmp_path):
    """Test pop_img_stats and calc_basic_img_band_stats on a continuous raster."""
    cont_img = tmp_path / "continuous.tif"

    arr1 = np.linspace(0.0, 100.0, 200 * 200, dtype=np.float32).reshape(200, 200)
    arr1[0, :] = -999.0  # No-data row
    arr2 = np.full((200, 200), 42.0, dtype=np.float32)
    _create_simple_raster(cont_img, np.stack([arr1, arr2]), dtype="float32")

    rsgispy.imageutils.pop_img_stats(
        str(cont_img),
        no_data_val=-999.0,
        calc_pyramids=True,
        ignore_cog=False,
        use_approx=False,
        use_pbar=False,
    )

    valid_b1 = arr1[arr1 != -999.0]
    with rasterio.open(cont_img) as src:
        assert src.nodata == -999.0
        assert src.overviews(1) == [4]

        tags1 = src.tags(1)
        assert np.isclose(float(tags1["STATISTICS_MINIMUM"]), float(valid_b1.min()))
        assert np.isclose(float(tags1["STATISTICS_MAXIMUM"]), float(valid_b1.max()))
        assert np.isclose(float(tags1["STATISTICS_MEAN"]), float(valid_b1.mean()), rtol=1e-4)
        assert np.isclose(float(tags1["STATISTICS_STDDEV"]), float(valid_b1.std()), rtol=1e-4)
        assert tags1["STATISTICS_EXCLUDEDVALUES"] == "-999.0"
        assert tags1["STATISTICS_HISTOBINFUNCTION"] == "Linear"
        assert tags1["STATISTICS_HISTONUMBINS"] == "256"

        # Band 2 has constant value 42.0 -> 1 bin
        tags2 = src.tags(2)
        assert float(tags2["STATISTICS_MINIMUM"]) == 42.0
        assert float(tags2["STATISTICS_MAXIMUM"]) == 42.0
        assert float(tags2["STATISTICS_MEAN"]) == 42.0
        assert float(tags2["STATISTICS_STDDEV"]) == 0.0
        assert tags2["STATISTICS_HISTONUMBINS"] == "1"
