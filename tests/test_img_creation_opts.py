"""
Tests for the default GDAL image creation options (tiled, DEFLATE-compressed
GeoTIFF), the order of precedence between the built-in defaults and the
``RSGISPY_IMG_CRT_OPTS_<FORMAT>`` / ``RSGISLIB_IMG_CRT_OPTS_<FORMAT>``
environment variables, the opt-out helpers, and the raster writes that do not
go through :func:`rsgispy.imageprocess.open_output_dataset`.
"""

import os
import subprocess
import sys

import numpy as np
import pytest
import rasterio
from rasterio.enums import Compression
from rasterio.transform import from_origin

import rsgispy
import rsgispy.imagecalc.maths
import rsgispy.imageprocess
import rsgispy.imageutils.envopts
import rsgispy.imageutils.stats
import rsgispy.imageutils.warp

_GTIFF_DEFAULTS = {
    "TILED": "YES",
    "COMPRESS": "DEFLATE",
    "INTERLEAVE": "BAND",
    "BIGTIFF": "IF_SAFER",
}
_ENV_FMTS = ("GTIFF", "COG", "KEA")


@pytest.fixture(name="clean_crt_env")
def fixture_clean_crt_env(monkeypatch):
    """Remove every creation-option environment variable for the test, then restore."""
    for fmt in _ENV_FMTS:
        for prefix in ("RSGISPY", "RSGISLIB"):
            name = f"{prefix}_IMG_CRT_OPTS_{fmt}"
            # setenv first so monkeypatch records (and later restores) the
            # original state even when the variable is set inside the test.
            monkeypatch.setenv(name, "")
            monkeypatch.delenv(name)
    yield monkeypatch


def _write_raster(path, data, dtype="float32", **extra):
    """Write a small striped, uncompressed GeoTIFF."""
    if data.ndim == 2:
        data = data[np.newaxis]
    count, height, width = data.shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=dtype,
        crs="EPSG:32630",
        transform=from_origin(500000.0, 5200000.0, 10.0, 10.0),
        **extra,
    ) as dst:
        dst.write(data.astype(dtype))


def _band_maths(in_img, out_img, **kwargs):
    band_defs = [rsgispy.imageprocess.BandDefn("b1", str(in_img), 1)]
    rsgispy.imagecalc.maths.band_maths(
        output_img=str(out_img),
        exp="b1 * 2",
        gdalformat=kwargs.pop("gdalformat", "GTIFF"),
        datatype=rsgispy.TYPE_32FLOAT,
        band_defs=band_defs,
        use_pbar=False,
        **kwargs,
    )


def test_default_creation_opts_constant_and_resolver(clean_crt_env):
    """With nothing configured GTIFF gets the built-in defaults; other formats get none."""
    _ = clean_crt_env
    assert rsgispy.imageprocess.DEFAULT_IMG_CREATION_OPTS["GTIFF"] == _GTIFF_DEFAULTS
    assert (
        rsgispy.imageutils.envopts.DEFAULT_IMG_CREATION_OPTS
        is rsgispy.imageprocess.DEFAULT_IMG_CREATION_OPTS
    )
    assert rsgispy.imageprocess.get_gdal_img_creation_opts("GTIFF") == _GTIFF_DEFAULTS
    assert rsgispy.imageprocess.get_gdal_img_creation_opts(" gtiff ") == _GTIFF_DEFAULTS
    # The GTIFF defaults must not leak into COG (whose driver rejects TILED)
    # or any other format.
    assert rsgispy.imageprocess.get_gdal_img_creation_opts("COG") == {}
    assert rsgispy.imageprocess.get_gdal_img_creation_opts("KEA") == {}

    # The returned dictionary is a copy: changing it does not change the defaults.
    opts = rsgispy.imageprocess.get_gdal_img_creation_opts("GTIFF")
    opts["COMPRESS"] = "NONE"
    assert rsgispy.imageprocess.get_gdal_img_creation_opts("GTIFF") == _GTIFF_DEFAULTS


def test_creation_opts_precedence(clean_crt_env):
    """RSGISPY_* beats RSGISLIB_* beats the default; env vars replace, not merge."""
    monkeypatch = clean_crt_env
    monkeypatch.setenv("RSGISLIB_IMG_CRT_OPTS_GTIFF", "COMPRESS=LZW")
    assert rsgispy.imageprocess.get_gdal_img_creation_opts("GTIFF") == {
        "COMPRESS": "LZW"
    }

    monkeypatch.setenv("RSGISPY_IMG_CRT_OPTS_GTIFF", "TILED=YES:COMPRESS=ZSTD")
    assert rsgispy.imageprocess.get_gdal_img_creation_opts("GTIFF") == {
        "TILED": "YES",
        "COMPRESS": "ZSTD",
    }

    monkeypatch.delenv("RSGISPY_IMG_CRT_OPTS_GTIFF")
    monkeypatch.delenv("RSGISLIB_IMG_CRT_OPTS_GTIFF")
    assert rsgispy.imageprocess.get_gdal_img_creation_opts("GTIFF") == _GTIFF_DEFAULTS


def test_creation_opts_empty_string_opts_out(clean_crt_env):
    """An environment variable set to an empty string means no creation options."""
    monkeypatch = clean_crt_env
    monkeypatch.setenv("RSGISPY_IMG_CRT_OPTS_GTIFF", "")
    monkeypatch.setenv("RSGISLIB_IMG_CRT_OPTS_GTIFF", "COMPRESS=LZW")
    assert rsgispy.imageprocess.get_gdal_img_creation_opts("GTIFF") == {}

    monkeypatch.delenv("RSGISPY_IMG_CRT_OPTS_GTIFF")
    monkeypatch.setenv("RSGISLIB_IMG_CRT_OPTS_GTIFF", "  ")
    assert rsgispy.imageprocess.get_gdal_img_creation_opts("GTIFF") == {}


def test_set_and_reset_img_creation_opts(clean_crt_env):
    """set_img_creation_opts(fmt, {}) opts out; reset_img_creation_opts restores defaults."""
    _ = clean_crt_env
    envopts = rsgispy.imageutils.envopts
    envopts.set_img_creation_opts("GTIFF", {})
    assert os.environ["RSGISPY_IMG_CRT_OPTS_GTIFF"] == ""
    assert os.environ["RSGISLIB_IMG_CRT_OPTS_GTIFF"] == ""
    assert envopts.get_gdal_img_creation_opts("GTIFF") == {}

    envopts.reset_img_creation_opts("gtiff")
    assert "RSGISPY_IMG_CRT_OPTS_GTIFF" not in os.environ
    assert "RSGISLIB_IMG_CRT_OPTS_GTIFF" not in os.environ
    assert envopts.get_gdal_img_creation_opts("GTIFF") == _GTIFF_DEFAULTS

    envopts.set_env_vars_lzw_gtiff_outs()
    envopts.set_env_vars_zstd_cog_outs()
    envopts.set_img_creation_opts("KEA", {"DEFLATE": 5})
    envopts.reset_img_creation_opts()
    for fmt in _ENV_FMTS:
        assert f"RSGISPY_IMG_CRT_OPTS_{fmt}" not in os.environ
        assert f"RSGISLIB_IMG_CRT_OPTS_{fmt}" not in os.environ
    assert envopts.get_gdal_img_creation_opts("GTIFF") == _GTIFF_DEFAULTS
    assert envopts.get_gdal_img_creation_opts("COG") == {}


def test_import_does_not_set_environment_variables():
    """Importing rsgispy must not write creation options into os.environ."""
    env = {
        k: v
        for k, v in os.environ.items()
        if "_IMG_CRT_OPTS_" not in k
    }
    code = (
        "import os\n"
        "import rsgispy.imageprocess, rsgispy.imageutils.envopts\n"
        "print(sorted(k for k in os.environ if '_IMG_CRT_OPTS_' in k))\n"
    )
    res = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert res.stdout.strip() == "[]"


def test_band_maths_gtiff_default_is_tiled_deflate(tmp_path, clean_crt_env):
    """A GeoTIFF from band_maths with nothing configured is tiled and DEFLATE-compressed."""
    _ = clean_crt_env
    in_img = tmp_path / "in.tif"
    out_img = tmp_path / "out.tif"
    _write_raster(in_img, np.arange(2 * 300 * 300, dtype=np.float32).reshape(2, 300, 300))
    _band_maths(in_img, out_img)
    with rasterio.open(out_img) as src:
        assert src.profile["tiled"] is True
        assert src.compression == Compression.deflate
        assert src.block_shapes[0] == (256, 256)
        np.testing.assert_allclose(
            src.read(1), np.arange(300 * 300, dtype=np.float32).reshape(300, 300) * 2
        )


def test_explicit_creation_opts_and_opt_out(tmp_path, clean_crt_env):
    """Explicit creation_opts are merged on top; an empty env var gives a plain GeoTIFF."""
    monkeypatch = clean_crt_env
    in_img = tmp_path / "in.tif"
    _write_raster(in_img, np.ones((64, 64), dtype=np.float32))

    out_lzw = tmp_path / "out_lzw.tif"
    _band_maths(in_img, out_lzw, creation_opts={"COMPRESS": "LZW"})
    with rasterio.open(out_lzw) as src:
        assert src.compression == Compression.lzw
        assert src.profile["tiled"] is True

    monkeypatch.setenv("RSGISPY_IMG_CRT_OPTS_GTIFF", "")
    out_plain = tmp_path / "out_plain.tif"
    _band_maths(in_img, out_plain)
    with rasterio.open(out_plain) as src:
        assert src.compression is None
        assert src.profile["tiled"] is False

    # An environment variable replaces the default (it is not merged with it).
    monkeypatch.setenv("RSGISPY_IMG_CRT_OPTS_GTIFF", "COMPRESS=LZW")
    out_env = tmp_path / "out_env.tif"
    _band_maths(in_img, out_env)
    with rasterio.open(out_env) as src:
        assert src.compression == Compression.lzw
        assert src.profile["tiled"] is False


def test_default_opts_with_kea_and_output_exists(tmp_path, clean_crt_env):
    """The GTIFF defaults do not break KEA output or output_exists=True updates."""
    _ = clean_crt_env
    in_img = tmp_path / "in.tif"
    _write_raster(in_img, np.full((40, 50), 3.0, dtype=np.float32))

    out_tif = tmp_path / "out.tif"
    _band_maths(in_img, out_tif)
    band_defs = [rsgispy.imageprocess.BandDefn("b1", str(in_img), 1)]
    rsgispy.imagecalc.maths.band_maths(
        output_img=str(out_tif),
        exp="b1 + 1",
        gdalformat="GTIFF",
        datatype=rsgispy.TYPE_32FLOAT,
        band_defs=band_defs,
        output_exists=True,
        use_pbar=False,
    )
    with rasterio.open(out_tif) as src:
        assert src.compression == Compression.deflate
        np.testing.assert_allclose(src.read(1), 4.0)

    with rasterio.Env() as env:
        has_kea = "KEA" in env.drivers()
    if has_kea:
        out_kea = tmp_path / "out.kea"
        _band_maths(in_img, out_kea, gdalformat="KEA")
        with rasterio.open(out_kea) as src:
            assert src.driver == "KEA"
            np.testing.assert_allclose(src.read(1), 6.0)


def test_gdal_translate_uses_creation_opts_resolver(tmp_path, clean_crt_env):
    """gdal_translate GeoTIFF output follows the resolver (defaults and env vars)."""
    monkeypatch = clean_crt_env
    in_img = tmp_path / "in.tif"
    _write_raster(in_img, np.arange(300 * 300, dtype=np.float32).reshape(300, 300))

    out_default = tmp_path / "out_default.tif"
    rsgispy.imageutils.warp.gdal_translate(
        str(in_img), str(out_default), calc_stats=False, use_pbar=False
    )
    with rasterio.open(out_default) as src:
        assert src.profile["tiled"] is True
        assert src.compression == Compression.deflate

    monkeypatch.setenv("RSGISPY_IMG_CRT_OPTS_GTIFF", "TILED=YES:COMPRESS=LZW")
    out_env = tmp_path / "out_env.tif"
    rsgispy.imageutils.warp.gdal_translate(
        str(in_img), str(out_env), calc_stats=False, use_pbar=False
    )
    with rasterio.open(out_env) as src:
        assert src.compression == Compression.lzw

    out_explicit = tmp_path / "out_explicit.tif"
    rsgispy.imageutils.warp.gdal_translate(
        str(in_img),
        str(out_explicit),
        options="-co COMPRESS=ZSTD -co TILED=YES",
        calc_stats=False,
        use_pbar=False,
    )
    with rasterio.open(out_explicit) as src:
        assert src.compression == Compression.zstd


def test_multi_core_internal_overviews_use_default_opts(tmp_path, clean_crt_env):
    """Rewriting an image with internal overviews keeps it tiled and DEFLATE-compressed."""
    _ = clean_crt_env
    in_img = tmp_path / "in.tif"
    _write_raster(in_img, np.random.default_rng(0).random((300, 300)).astype("float32"))
    rsgispy.imageutils.stats.build_img_pyramids_multi_core(
        str(in_img), levels=[2, 4], external=False, use_pbar=False
    )
    with rasterio.open(in_img) as src:
        assert src.overviews(1) == [2, 4]
        assert src.profile["tiled"] is True
        assert src.compression == Compression.deflate
