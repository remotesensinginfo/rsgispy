"""
Tests for rsgispy.imageutils image creation option helpers and integration with band_maths.
"""

import numpy as np
import rasterio
from rasterio.enums import Compression
from rasterio.transform import from_origin

import rsgispy
import rsgispy.imagecalc
import rsgispy.imageutils


def _create_simple_raster(path, data):
    """Write a single-band float32 test GeoTIFF."""
    height, width = data.shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(0.0, 100.0, 1.0, 1.0),
    ) as dst:
        dst.write(data.astype("float32"), 1)


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
        assert src.is_tiled is True
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
        assert src.is_tiled is True
        np.testing.assert_allclose(src.read(1), data + 5.0)
