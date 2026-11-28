"""
Tests for Cloud Optimised GeoTIFF (COG) outputs: COGs are written block by
block to a temporary tiled GeoTIFF next to the output and converted when the
writer closes (no in-memory ``BufferedDatasetWriter``), statistics are carried
into the COG, existing COGs are never edited in place, and no temporary files
are left behind.
"""

from contextlib import ExitStack
import functools
import importlib.util
import os
import tempfile
import uuid
import warnings
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.enums import ColorInterp
from rasterio.io import BufferedDatasetWriter
import rasterio.shutil
from rasterio.transform import from_origin

import rsgispy
import rsgispy.imagecalc.maths
import rsgispy.imagecalc.percentiles
import rsgispy.imageprocess
import rsgispy.imageprocess._io as rsgis_io
import rsgispy.imageutils.bands
import rsgispy.imageutils.envopts
import rsgispy.imageutils.masking
import rsgispy.imageutils.mosaic
import rsgispy.imageutils.rasterise
import rsgispy.imageutils.stats
import rsgispy.imageutils.warp

try:
    from osgeo_utils.samples.validate_cloud_optimized_geotiff import (
        validate as _gdal_validate_cog,
    )
except ImportError:  # pragma: no cover - GDAL Python bindings are optional
    _gdal_validate_cog = None

WIDTH = 600
HEIGHT = 520


@functools.lru_cache(maxsize=None)
def _gdal_records_overview_resampling():
    """
    Whether this GDAL stores ``OVERVIEW_RESAMPLING`` in a COG's
    ``IMAGE_STRUCTURE`` metadata (GDAL 3.12 does; some older versions, e.g.
    those bundled in rasterio wheels, do not). Where it is not recorded, the
    tests check the overview values instead.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        src = os.path.join(tmp_dir, "src.tif")
        with rasterio.open(
            src, "w", driver="GTiff", width=1024, height=1024, count=1,
            dtype="uint8", crs="EPSG:32630",
            transform=from_origin(500000.0, 5200000.0, 10.0, 10.0),
        ) as dst:
            dst.write(np.zeros((1, 1024, 1024), dtype=np.uint8))
        cog = os.path.join(tmp_dir, "cog.tif")
        rasterio.shutil.copy(src, cog, driver="COG", OVERVIEW_RESAMPLING="AVERAGE")
        with rasterio.open(cog) as cog_src:
            return cog_src.tags(ns="IMAGE_STRUCTURE").get("OVERVIEW_RESAMPLING") == "AVERAGE"


def _assert_overview_resampling(src, expected):
    """Check the recorded overview resampling, where this GDAL records it."""
    if _gdal_records_overview_resampling():
        assert src.tags(ns="IMAGE_STRUCTURE").get("OVERVIEW_RESAMPLING") == expected


@pytest.fixture(name="clean_crt_env")
def fixture_clean_crt_env(monkeypatch):
    """Remove the creation-option environment variables for the test, then restore."""
    for fmt in ("GTIFF", "COG"):
        for prefix in ("RSGISPY", "RSGISLIB"):
            name = f"{prefix}_IMG_CRT_OPTS_{fmt}"
            monkeypatch.setenv(name, "")
            monkeypatch.delenv(name)
    yield monkeypatch


def _write_raster(path, data, dtype="float32", nodata=None, x0=500000.0):
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
        transform=from_origin(x0, 5200000.0, 10.0, 10.0),
        nodata=nodata,
    ) as dst:
        dst.write(data.astype(dtype))


def _ramp(n_bands=1, height=HEIGHT, width=WIDTH):
    base = np.add.outer(np.arange(height), np.arange(width)).astype(np.float32)
    return np.stack([base + 1000.0 * b for b in range(n_bands)])


def _assert_valid_cog(path, stats=True, n_bands=1):
    with rasterio.open(path) as src:
        assert src.driver == "GTiff"
        assert src.tags(ns="IMAGE_STRUCTURE").get("LAYOUT") == "COG"
        assert src.overviews(1), "a COG of this size should have overviews"
        if stats:
            for bidx in range(1, n_bands + 1):
                tags = src.tags(bidx)
                assert "STATISTICS_MEAN" in tags
                assert "STATISTICS_MINIMUM" in tags
    if _gdal_validate_cog is not None:
        with warnings.catch_warnings():
            # osgeo.gdal warns that its exception mode is not set explicitly.
            warnings.simplefilter("ignore", FutureWarning)
            val_warnings, errors, _ = _gdal_validate_cog(str(path), full_check=True)
        assert errors == []
        assert val_warnings == []


def _assert_no_temp_files(out_dir, expected):
    """Only the expected files remain (no staging GeoTIFFs or sidecars)."""
    assert sorted(os.listdir(out_dir)) == sorted(expected)


def test_open_output_dataset_cog_is_not_buffered(tmp_path, clean_crt_env):
    """A COG output is written to an on-disk GeoTIFF and converted on close."""
    _ = clean_crt_env
    in_img = tmp_path / "in.tif"
    _write_raster(in_img, _ramp())
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    out_cog = out_dir / "out.tif"

    with ExitStack() as stack:
        src = stack.enter_context(rasterio.open(in_img))
        dst = rsgispy.imageprocess.create_output_from_dataset(
            stack, src, str(out_cog), "COG", "float32"
        )
        assert not isinstance(dst, BufferedDatasetWriter)
        assert dst.driver == "GTiff"
        assert dst.profile["tiled"] is True
        # The output is only created when the writer closes.
        assert not out_cog.exists()
        assert len(os.listdir(out_dir)) == 1
        dst.nodata = -9999.0
        dst.set_band_description(1, "ramp")
        for _, win in src.block_windows(1):
            dst.write(src.read(window=win), window=win)

    _assert_valid_cog(out_cog, stats=False)
    _assert_no_temp_files(out_dir, ["out.tif"])
    with rasterio.open(out_cog) as cog:
        assert cog.nodata == -9999.0
        assert cog.descriptions == ("ramp",)
        np.testing.assert_array_equal(cog.read(), _ramp())


def test_open_output_dataset_cog_removes_temp_on_error(tmp_path, clean_crt_env):
    """An exception while writing leaves neither an output nor a temporary file."""
    _ = clean_crt_env
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    grid = rsgispy.imageprocess.IntersectionGrid(
        width=64,
        height=64,
        transform=from_origin(0.0, 64.0, 1.0, 1.0),
        crs=rasterio.crs.CRS.from_epsg(32630),
        offsets={},
    )
    with pytest.raises(RuntimeError, match="boom"):
        with ExitStack() as stack:
            dst = rsgispy.imageprocess.open_output_dataset(
                stack, str(out_dir / "out.tif"), "COG", "float32", grid
            )
            dst.write(np.ones((1, 64, 64), dtype=np.float32))
            raise RuntimeError("boom")
    _assert_no_temp_files(out_dir, [])

    # band_maths with an expression that fails part-way through.
    in_img = tmp_path / "in.tif"
    _write_raster(in_img, _ramp())
    band_defs = [rsgispy.imageprocess.BandDefn("b1", str(in_img), 1)]
    with pytest.raises(Exception):
        rsgispy.imagecalc.maths.band_maths(
            output_img=str(out_dir / "bm.tif"),
            exp="b1 + undefined_var",
            gdalformat="COG",
            band_defs=band_defs,
            use_pbar=False,
        )
    _assert_no_temp_files(out_dir, [])


def test_band_maths_cog_then_pop_img_stats(tmp_path, clean_crt_env):
    """band_maths writes a valid COG; pop_img_stats on it keeps it a valid COG."""
    _ = clean_crt_env
    in_img = tmp_path / "in.tif"
    _write_raster(in_img, _ramp())
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    out_cog = out_dir / "bm.tif"
    band_defs = [rsgispy.imageprocess.BandDefn("b1", str(in_img), 1)]
    rsgispy.imagecalc.maths.band_maths(
        output_img=str(out_cog),
        exp="b1 * 2",
        gdalformat="COG",
        datatype=rsgispy.TYPE_32FLOAT,
        band_defs=band_defs,
        use_pbar=False,
    )
    _assert_valid_cog(out_cog, stats=False)
    with rasterio.open(out_cog) as src:
        orig_overviews = src.overviews(1)
        # GeoTIFF defaults (TILED, INTERLEAVE=BAND) must not leak into COGs.
        assert src.block_shapes[0] == (512, 512)

    rsgispy.imageutils.stats.pop_img_stats(
        str(out_cog), no_data_val=-1.0, use_pbar=False
    )
    _assert_valid_cog(out_cog, stats=True)
    _assert_no_temp_files(out_dir, ["bm.tif"])
    with rasterio.open(out_cog) as src:
        assert src.nodata == -1.0
        assert src.overviews(1) == orig_overviews
        assert float(src.tags(1)["STATISTICS_MAXIMUM"]) == pytest.approx(
            2.0 * (HEIGHT - 1 + WIDTH - 1)
        )
        np.testing.assert_array_equal(src.read(1), _ramp()[0] * 2)


def test_band_maths_cog_with_no_data_and_stats(tmp_path, clean_crt_env):
    """band_maths(calc_stats=True, no_data_val=...) writes a valid COG with stats."""
    _ = clean_crt_env
    in_img = tmp_path / "in.tif"
    _write_raster(in_img, _ramp())
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    out_cog = out_dir / "bm.tif"
    band_defs = [rsgispy.imageprocess.BandDefn("b1", str(in_img), 1)]
    rsgispy.imagecalc.maths.band_maths(
        output_img=str(out_cog),
        exp="b1 < 10 ? -1 : b1",
        band_defs=band_defs,
        gdalformat="COG",
        datatype=rsgispy.TYPE_32FLOAT,
        no_data_val=-1.0,
        calc_stats=True,
        use_pbar=False,
    )
    _assert_valid_cog(out_cog, stats=True)
    _assert_no_temp_files(out_dir, ["bm.tif"])
    with rasterio.open(out_cog) as src:
        assert src.nodata == -1.0
        assert float(src.tags(1)["STATISTICS_MINIMUM"]) == pytest.approx(10.0)


def test_band_maths_cog_output_exists_update(tmp_path, clean_crt_env):
    """output_exists=True on an existing COG rewrites it as a valid COG."""
    _ = clean_crt_env
    in_img = tmp_path / "in.tif"
    _write_raster(in_img, _ramp())
    out_cog = tmp_path / "bm.tif"
    band_defs = [rsgispy.imageprocess.BandDefn("b1", str(in_img), 1)]
    kwargs = dict(
        output_img=str(out_cog),
        gdalformat="COG",
        datatype=rsgispy.TYPE_32FLOAT,
        band_defs=band_defs,
        use_pbar=False,
    )
    rsgispy.imagecalc.maths.band_maths(exp="b1 * 2", **kwargs)
    rsgispy.imagecalc.maths.band_maths(exp="b1 + 1", output_exists=True, **kwargs)
    _assert_valid_cog(out_cog, stats=False)
    _assert_no_temp_files(tmp_path, ["bm.tif", "in.tif"])
    with rasterio.open(out_cog) as src:
        np.testing.assert_array_equal(src.read(1), _ramp()[0] + 1)


def test_create_img_mosaic_cog_with_stats(tmp_path, clean_crt_env):
    """create_img_mosaic(..., gdalformat='COG') with the default calc_stats=True."""
    _ = clean_crt_env
    tile_a = tmp_path / "a.tif"
    tile_b = tmp_path / "b.tif"
    _write_raster(tile_a, np.full((2, 400, 300), 5.0), nodata=0.0)
    _write_raster(tile_b, np.full((2, 400, 300), 9.0), nodata=0.0, x0=503000.0)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    out_cog = out_dir / "mosaic.tif"
    rsgispy.imageutils.mosaic.create_img_mosaic(
        [str(tile_a), str(tile_b)],
        str(out_cog),
        background_val=0.0,
        skip_val=0.0,
        gdalformat="COG",
        use_pbar=False,
    )
    _assert_valid_cog(out_cog, stats=True, n_bands=2)
    _assert_no_temp_files(out_dir, ["mosaic.tif"])
    with rasterio.open(out_cog) as src:
        assert src.width == 600
        assert src.nodata == 0.0
        assert float(src.tags(1)["STATISTICS_MEAN"]) == pytest.approx(7.0)


def test_gdal_translate_cog_with_stats(tmp_path, clean_crt_env):
    """gdal_translate to COG with calc_stats=True writes statistics into the COG."""
    monkeypatch = clean_crt_env
    in_img = tmp_path / "in.tif"
    _write_raster(in_img, _ramp(2), nodata=-1.0)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    out_cog = out_dir / "trans.tif"
    rsgispy.imageutils.warp.gdal_translate(
        str(in_img),
        str(out_cog),
        options={"COMPRESS": "DEFLATE", "PREDICTOR": "YES", "TILED": "YES"},
        gdalformat="COG",
        calc_stats=True,
        use_pbar=False,
    )
    _assert_valid_cog(out_cog, stats=True, n_bands=2)
    _assert_no_temp_files(out_dir, ["trans.tif"])
    with rasterio.open(out_cog) as src:
        assert src.compression.name.upper() == "DEFLATE"
        assert src.tags(ns="IMAGE_STRUCTURE").get("PREDICTOR") == "3"
        np.testing.assert_array_equal(src.read(), _ramp(2))

    # COG creation options from the environment apply.
    monkeypatch.setenv("RSGISPY_IMG_CRT_OPTS_COG", "COMPRESS=LZW:BLOCKSIZE=256")
    out_lzw = out_dir / "trans_lzw.tif"
    rsgispy.imageutils.warp.gdal_translate(
        str(in_img), str(out_lzw), gdalformat="COG", use_pbar=False
    )
    _assert_valid_cog(out_lzw, stats=True, n_bands=2)
    with rasterio.open(out_lzw) as src:
        assert src.compression.name.upper() == "LZW"
        assert src.block_shapes[0] == (256, 256)


def test_pop_img_stats_existing_cog(tmp_path, clean_crt_env):
    """pop_img_stats on an existing COG rebuilds it instead of editing it in place."""
    _ = clean_crt_env
    src_img = tmp_path / "src.tif"
    _write_raster(src_img, _ramp(2))
    cog = tmp_path / "existing_cog.tif"
    rasterio.shutil.copy(
        str(src_img), str(cog), driver="COG", COMPRESS="ZSTD", PREDICTOR="YES",
        OVERVIEW_RESAMPLING="AVERAGE",
    )
    rsgispy.imageutils.stats.pop_img_stats(str(cog), use_pbar=False)
    _assert_valid_cog(cog, stats=True, n_bands=2)
    _assert_no_temp_files(tmp_path, ["existing_cog.tif", "src.tif"])
    with rasterio.open(cog) as src:
        struct = src.tags(ns="IMAGE_STRUCTURE")
        # The original layout and compression settings are kept.
        assert src.compression.name.upper() == "ZSTD"
        assert struct.get("PREDICTOR") == "3"
        _assert_overview_resampling(src, "AVERAGE")
        np.testing.assert_array_equal(src.read(), _ramp(2))

    # maybe_pop_stats (used after many writers) also keeps the COG layout.
    rsgispy.imageprocess.maybe_pop_stats(
        str(cog), True, no_data_val=0.0, use_pbar=False, gdalformat="COG"
    )
    _assert_valid_cog(cog, stats=True, n_bands=2)
    with rasterio.open(cog) as src:
        assert src.nodata == 0.0


def test_pop_thmt_img_stats_existing_cog(tmp_path, clean_crt_env):
    """pop_thmt_img_stats on an existing COG adds a colour table and keeps the layout."""
    _ = clean_crt_env
    src_img = tmp_path / "cls.tif"
    cls = (np.add.outer(np.arange(HEIGHT), np.arange(WIDTH)) // 100 % 5).astype(np.uint8)
    _write_raster(src_img, cls, dtype="uint8")
    cog = tmp_path / "cls_cog.tif"
    rasterio.shutil.copy(str(src_img), str(cog), driver="COG")
    rsgispy.imageutils.stats.pop_thmt_img_stats(str(cog), use_pbar=False)
    _assert_valid_cog(cog, stats=False)
    _assert_no_temp_files(tmp_path, ["cls.tif", "cls_cog.tif"])
    with rasterio.open(cog) as src:
        assert "STATISTICS_MAXIMUM" in src.tags(1)
        assert src.colorinterp[0] == ColorInterp.palette
        assert src.nodata == 0.0
        # Thematic overviews use nearest neighbour.
        _assert_overview_resampling(src, "NEAREST")
        assert set(np.unique(src.read(1, out_shape=(HEIGHT // 4, WIDTH // 4)))) <= set(
            range(5)
        )


def test_tile_graph_disk_sink_cog(tmp_path, clean_crt_env):
    """A TileGraph DiskSink COG output with band names and statistics."""
    _ = clean_crt_env
    in_img = tmp_path / "in.tif"
    _write_raster(in_img, _ramp())
    out_cog = tmp_path / "tg.tif"
    src_node = rsgispy.imageprocess.RasterSource(str(in_img), bands=[1])
    sink = rsgispy.imageprocess.DiskSink(
        rsgispy.imageprocess.PointwiseOp([src_node], lambda arr: arr * 3.0),
        str(out_cog),
        gdalformat="COG",
        nodata=-1.0,
        band_names=["tripled"],
        calc_stats=True,
    )
    rsgispy.imageprocess.run_tile_graph(sink, use_pbar=False)
    _assert_valid_cog(out_cog, stats=True)
    _assert_no_temp_files(tmp_path, ["in.tif", "tg.tif"])
    with rasterio.open(out_cog) as src:
        assert src.descriptions == ("tripled",)
        assert src.nodata == -1.0
        np.testing.assert_allclose(src.read(1), _ramp()[0] * 3.0)


def test_resample_and_reproject_cog(tmp_path, clean_crt_env):
    """resample_img_to_match / reproject_image write valid COGs with statistics."""
    _ = clean_crt_env
    ref_img = tmp_path / "ref.tif"
    proc_img = tmp_path / "proc.tif"
    _write_raster(ref_img, np.zeros((HEIGHT, WIDTH)))
    _write_raster(proc_img, np.full((HEIGHT // 2, WIDTH // 2), 42.0), nodata=-999.0)
    with rasterio.open(proc_img, "r+") as dst:
        dst.transform = from_origin(500000.0, 5200000.0, 20.0, 20.0)

    resamp = tmp_path / "resamp.tif"
    rsgispy.imageutils.warp.resample_img_to_match(
        str(ref_img), str(proc_img), str(resamp), gdalformat="COG"
    )
    _assert_valid_cog(resamp, stats=True)
    with rasterio.open(resamp) as src:
        assert src.width == WIDTH
        np.testing.assert_allclose(src.read(1), 42.0)

    reproj = tmp_path / "reproj.tif"
    rsgispy.imageutils.warp.reproject_image(
        str(proc_img),
        str(reproj),
        out_epsg=3857,
        gdalformat="COG",
        no_data_val=-999.0,
        out_pxl_res="auto",
    )
    with rasterio.open(reproj) as src:
        assert src.tags(ns="IMAGE_STRUCTURE").get("LAYOUT") == "COG"
        assert "STATISTICS_MEAN" in src.tags(1)
    _assert_no_temp_files(
        tmp_path, ["proc.tif", "ref.tif", "reproj.tif", "resamp.tif"]
    )


def test_gdal_warp_cog_with_cutline(tmp_path, clean_crt_env):
    """gdal_warp to COG, including the cutline step, gives a valid COG."""
    _ = clean_crt_env
    import geopandas as gpd  # pylint: disable=import-outside-toplevel
    from shapely.geometry import box  # pylint: disable=import-outside-toplevel

    in_img = tmp_path / "in.tif"
    _write_raster(in_img, _ramp(), nodata=-1.0)
    cut = tmp_path / "cut.gpkg"
    gpd.GeoDataFrame(
        geometry=[box(500500.0, 5196000.0, 504000.0, 5199500.0)], crs="EPSG:32630"
    ).to_file(cut)
    out_cog = tmp_path / "warp.tif"
    rsgispy.imageutils.warp.gdal_warp(
        str(in_img),
        str(out_cog),
        gdalformat="COG",
        cutline=str(cut),
        use_pbar=False,
    )
    _assert_valid_cog(out_cog, stats=False)
    with rasterio.open(out_cog) as src:
        data = src.read(1)
        assert data[0, 0] == -1.0
        assert data[HEIGHT // 2, WIDTH // 2] == _ramp()[0][HEIGHT // 2, WIDTH // 2]
    _assert_no_temp_files(tmp_path, ["cut.gpkg", "in.tif", "warp.tif"])


def test_include_imgs_into_existing_cog(tmp_path, clean_crt_env):
    """include_imgs updates an existing COG without breaking its layout."""
    _ = clean_crt_env
    base_src = tmp_path / "base_src.tif"
    _write_raster(base_src, np.zeros((HEIGHT, WIDTH)))
    base_cog = tmp_path / "base_cog.tif"
    rasterio.shutil.copy(str(base_src), str(base_cog), driver="COG")
    patch = tmp_path / "patch.tif"
    _write_raster(patch, np.full((50, 60), 7.0))
    rsgispy.imageutils.mosaic.include_imgs(
        str(base_cog), [str(patch)], use_pbar=False
    )
    _assert_valid_cog(base_cog, stats=False)
    _assert_no_temp_files(tmp_path, ["base_cog.tif", "base_src.tif", "patch.tif"])
    with rasterio.open(base_cog) as src:
        data = src.read(1)
        assert np.all(data[:50, :60] == 7.0)
        assert np.all(data[50:, :] == 0.0)


def test_cloud_mask_cog_with_colour_table(tmp_path, clean_crt_env):
    """The CloudSEN12 mask colour table and statistics end up in a valid COG."""
    # pylint: disable=import-outside-toplevel,protected-access
    _ = clean_crt_env
    import rsgispy.imagecalibration.cloudmasking as cldmsk

    in_img = tmp_path / "in.tif"
    data = np.random.default_rng(0).integers(1, 4000, size=(13, HEIGHT, WIDTH))
    _write_raster(in_img, data.astype(np.uint16), dtype="uint16")
    out_cog = tmp_path / "clouds.tif"

    def _predict(arr):
        return (arr[0] >= 0.2).astype(np.uint8)

    cldmsk._apply_cloudsen12_cloud_mask(
        input_img=str(in_img),
        output_img=str(out_cog),
        product="toa",
        model_name="UNetMobV2_V2",
        img_bands=None,
        weights_dir=None,
        download_weights=False,
        device="cpu",
        scale_factor=10000.0,
        add_offset=0.0,
        no_data_val=0.0,
        tile_size=256,
        tile_overlap=0,
        gdalformat="COG",
        calc_stats=True,
        use_pbar=False,
        creation_opts=None,
        predict_fn=_predict,
    )
    _assert_valid_cog(out_cog, stats=False)
    _assert_no_temp_files(tmp_path, ["clouds.tif", "in.tif"])
    with rasterio.open(out_cog) as src:
        assert "STATISTICS_MAXIMUM" in src.tags(1)
        assert src.colormap(1)[cldmsk.CLOUDSEN12_THICK_CLOUD][:3] == (220, 220, 220)
        assert src.descriptions[0] == "CloudSEN12"


def test_apply_offset_to_image_cog(tmp_path, clean_crt_env):
    """apply_offset_to_image (copy_raster_with_updates) writes a valid COG."""
    # pylint: disable=import-outside-toplevel
    _ = clean_crt_env
    import rsgispy.imageregistration.shift

    in_img = tmp_path / "in.tif"
    _write_raster(in_img, _ramp(), nodata=-1.0)
    out_cog = tmp_path / "shifted.tif"
    rsgispy.imageregistration.shift.apply_offset_to_image(
        str(in_img), str(out_cog), x_offset=20.0, gdalformat="COG", use_pbar=False
    )
    _assert_valid_cog(out_cog, stats=True)
    _assert_no_temp_files(tmp_path, ["in.tif", "shifted.tif"])
    with rasterio.open(out_cog) as src:
        assert src.transform.c == pytest.approx(500020.0)
        np.testing.assert_array_equal(src.read(1), _ramp()[0])


def test_cog_helpers(tmp_path, clean_crt_env):
    """is_cog_layout, resolve_cog_creation_opts, StagedOutput and update_cog."""
    monkeypatch = clean_crt_env
    # Small tiled GeoTIFFs can report LAYOUT=COG in recent GDAL versions but
    # are ordinary GeoTIFFs that may be edited in place.
    small = tmp_path / "small.tif"
    _write_raster(small, np.ones((40, 50)))
    tiled_small = tmp_path / "tiled_small.tif"
    band_defs = [rsgispy.imageprocess.BandDefn("b1", str(small), 1)]
    rsgispy.imagecalc.maths.band_maths(
        output_img=str(tiled_small), exp="b1", band_defs=band_defs, use_pbar=False
    )
    assert not rsgispy.imageprocess.is_cog_layout(tiled_small)
    rsgispy.imageutils.stats.pop_img_stats(str(tiled_small), use_pbar=False)
    with rasterio.open(tiled_small) as src:
        assert "STATISTICS_MEAN" in src.tags(1)
    assert not rsgispy.imageprocess.is_cog_layout(tmp_path / "missing.tif")
    with pytest.raises(rsgispy.RSGISPyException, match="not a COG"):
        rsgispy.imageprocess.update_cog(str(small), lambda path: None)

    monkeypatch.setenv("RSGISPY_IMG_CRT_OPTS_COG", "COMPRESS=ZSTD:LEVEL=9")
    assert rsgispy.imageprocess.resolve_cog_creation_opts(
        {"tiled": "YES", "BLOCKXSIZE": 256, "BLOCKYSIZE": 256}
    ) == {"COMPRESS": "ZSTD", "LEVEL": "9", "BLOCKSIZE": 256}

    # A StagedOutput for a non-COG format writes the output directly.
    x_img = str(tmp_path / "x.tif")
    with rsgispy.imageprocess.StagedOutput(x_img, "GTIFF") as stage:
        assert stage.path == x_img
        assert not stage.is_cog
        assert stage.write_opts["COMPRESS"] == "DEFLATE"

    # A failing update leaves the COG unchanged and removes the temporary file.
    cog = tmp_path / "cog.tif"
    rasterio.shutil.copy(str(small), str(cog), driver="COG")
    assert rsgispy.imageprocess.is_cog_layout(cog)
    with open(cog, "rb") as fobj:
        before = fobj.read()

    def _fail(path):
        raise RuntimeError(f"cannot update {path}")

    with pytest.raises(RuntimeError, match="cannot update"):
        rsgispy.imageprocess.update_cog(str(cog), _fail)
    with open(cog, "rb") as fobj:
        assert fobj.read() == before
    _assert_no_temp_files(tmp_path, ["cog.tif", "small.tif", "tiled_small.tif"])


def _rgb_uint8(height=HEIGHT, width=WIDTH):
    """Smooth 3-band 8-bit image (JPEG-compressible)."""
    rows, cols = np.mgrid[0:height, 0:width]
    base = (np.sin(cols / 37.0) + np.cos(rows / 23.0)) * 60.0 + 128.0
    return np.stack(
        [base, np.roll(base, 50, axis=0), np.roll(base, 90, axis=1)]
    ).astype(np.uint8)


def _count_cog_conversions(monkeypatch):
    """Count the COG conversions (write_cog calls, also made by update_cog)."""
    counts = {"write_cog": 0, "update_cog": 0}
    orig_write = rsgis_io.write_cog
    orig_update = rsgispy.imageutils.stats.update_cog

    def _write_cog(*args, **kwargs):
        counts["write_cog"] += 1
        return orig_write(*args, **kwargs)

    def _update_cog(*args, **kwargs):
        counts["update_cog"] += 1
        return orig_update(*args, **kwargs)

    monkeypatch.setattr(rsgis_io, "write_cog", _write_cog)
    monkeypatch.setattr(rsgispy.imageutils.stats, "update_cog", _update_cog)
    return counts


def _try_cog_copy(src_path, dst_path, **opts):
    """Create a COG, returning False if GDAL lacks the requested codec."""
    try:
        rasterio.shutil.copy(str(src_path), str(dst_path), driver="COG", **opts)
    except Exception:  # pylint: disable=broad-except
        return False
    return True


def test_cog_rebuild_keeps_lossy_compression_settings(tmp_path, clean_crt_env):
    """Rebuilding a COG keeps JPEG/WEBP quality and the LERC maximum error."""
    _ = clean_crt_env
    rgb_img = tmp_path / "rgb.tif"
    _write_raster(rgb_img, _rgb_uint8(), dtype="uint8")
    jpeg_cog = tmp_path / "jpeg.tif"
    rasterio.shutil.copy(
        str(rgb_img), str(jpeg_cog), driver="COG", COMPRESS="JPEG", QUALITY="98"
    )
    opts = rsgispy.imageprocess.cog_creation_opts_from_file(jpeg_cog)
    assert opts["COMPRESS"] == "JPEG"
    assert opts["QUALITY"] == "98"
    rsgispy.imageutils.stats.pop_img_stats(str(jpeg_cog), use_pbar=False)
    _assert_valid_cog(jpeg_cog, stats=True, n_bands=3)
    with rasterio.open(jpeg_cog) as src:
        assert src.tags(ns="IMAGE_STRUCTURE").get("JPEG_QUALITY") == "98"

    # WEBP: QUALITY from WEBP_LEVEL, QUALITY=100 (lossless) when no level.
    for quality, lossless in (("60", False), ("100", True)):
        webp_cog = tmp_path / f"webp_{quality}.tif"
        if not _try_cog_copy(rgb_img, webp_cog, COMPRESS="WEBP", QUALITY=quality):
            continue
        opts = rsgispy.imageprocess.cog_creation_opts_from_file(webp_cog)
        assert opts["COMPRESS"] == "WEBP"
        assert opts["QUALITY"] == quality
        rsgispy.imageutils.stats.pop_img_stats(str(webp_cog), use_pbar=False)
        with rasterio.open(webp_cog) as src:
            struct = src.tags(ns="IMAGE_STRUCTURE")
            if lossless:
                assert struct.get("COMPRESSION_REVERSIBILITY") == "LOSSLESS"
                np.testing.assert_array_equal(src.read(), _rgb_uint8())
            else:
                assert struct.get("WEBP_LEVEL") == quality

    flt_img = tmp_path / "flt.tif"
    _write_raster(flt_img, _ramp())
    lerc_cog = tmp_path / "lerc.tif"
    if _try_cog_copy(
        flt_img, lerc_cog, COMPRESS="LERC_DEFLATE", MAX_Z_ERROR="0.5"
    ):
        opts = rsgispy.imageprocess.cog_creation_opts_from_file(lerc_cog)
        assert opts["COMPRESS"] == "LERC_DEFLATE"
        assert opts["MAX_Z_ERROR"] == "0.5"
        rsgispy.imageutils.stats.pop_img_stats(str(lerc_cog), use_pbar=False)
        with rasterio.open(lerc_cog) as src:
            assert src.tags(ns="IMAGE_STRUCTURE").get("MAX_Z_ERROR") == "0.5"


def test_cog_outputs_with_stats_are_converted_once(tmp_path, clean_crt_env):
    """Functions writing COGs with calc_stats=True convert once and keep QUALITY."""
    # pylint: disable=import-outside-toplevel
    import geopandas as gpd
    from shapely.geometry import box

    monkeypatch = clean_crt_env
    counts = _count_cog_conversions(monkeypatch)
    jpeg_opts = {"COMPRESS": "JPEG", "QUALITY": "98"}
    in_img = tmp_path / "rgb.tif"
    _write_raster(in_img, _rgb_uint8(), dtype="uint8")
    msk_img = tmp_path / "msk.tif"
    msk = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    msk[:100, :150] = 1
    _write_raster(msk_img, msk, dtype="uint8")
    vec_file = tmp_path / "zones.gpkg"
    gpd.GeoDataFrame(
        geometry=[box(501000.0, 5197000.0, 502000.0, 5198000.0)], crs="EPSG:32630"
    ).to_file(vec_file, layer="zones")
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    def _check(out_img, n_bands):
        assert counts == {"write_cog": 1, "update_cog": 0}
        _assert_valid_cog(out_img, stats=True, n_bands=n_bands)
        with rasterio.open(out_img) as src:
            assert src.tags(ns="IMAGE_STRUCTURE").get("JPEG_QUALITY") == "98"
        counts.update(write_cog=0, update_cog=0)

    out_sel = out_dir / "sel.tif"
    rsgispy.imageutils.bands.select_img_bands(
        str(in_img), str(out_sel), [1, 2, 3], gdalformat="COG",
        calc_stats=True, use_pbar=False, creation_opts=jpeg_opts,
    )
    _check(out_sel, 3)

    out_msk = out_dir / "masked.tif"
    rsgispy.imageutils.masking.mask_img(
        str(in_img), str(msk_img), str(out_msk), 0, 1, gdalformat="COG",
        calc_stats=True, use_pbar=False, creation_opts=jpeg_opts,
    )
    _check(out_msk, 3)

    out_pct = out_dir / "pct.tif"
    rsgispy.imagecalc.percentiles.calc_imgs_pxl_percentiles(
        [str(in_img)], str(out_pct), [50.0], gdalformat="COG",
        no_data_val=0, calc_stats=True, use_pbar=False, creation_opts=jpeg_opts,
    )
    _check(out_pct, 1)

    # create_dist_zones_to_vec_lyr needs scipy (the 'stats' extra).
    has_scipy = importlib.util.find_spec("scipy") is not None
    if has_scipy:
        out_zones = out_dir / "zones.tif"
        rsgispy.imageutils.rasterise.create_dist_zones_to_vec_lyr(
            str(vec_file), "zones", str(in_img), str(out_zones),
            recode_lut=[(1, (0.0, 500.0)), (2, (500.0, 2000.0))], gdalformat="COG",
            calc_stats=True, use_pbar=False, creation_opts=jpeg_opts,
        )
        _check(out_zones, 1)

    # Thematic output: LAYER_TYPE and thematic statistics are added before
    # the conversion (set_img_thematic on the COG used to fail) and the
    # overviews use nearest neighbour.
    nan_img = tmp_path / "nan.tif"
    nan_data = _ramp()
    nan_data[0, :50, :] = np.nan
    _write_raster(nan_img, nan_data)
    out_fin = out_dir / "finite.tif"
    rsgispy.imageutils.masking.gen_finite_mask(
        str(nan_img), str(out_fin), gdalformat="COG", calc_stats=True,
        use_pbar=False,
    )
    assert counts == {"write_cog": 1, "update_cog": 0}
    _assert_valid_cog(out_fin, stats=False)
    with rasterio.open(out_fin) as src:
        assert src.tags(1).get("LAYER_TYPE") == "thematic"
        assert "STATISTICS_MAXIMUM" in src.tags(1)
        _assert_overview_resampling(src, "NEAREST")
    with rasterio.open(out_fin, overview_level=0) as ovr:
        assert set(np.unique(ovr.read(1)).tolist()) <= {0, 1}
    expected_files = ["finite.tif", "masked.tif", "pct.tif", "sel.tif"]
    if has_scipy:
        expected_files.append("zones.tif")
    _assert_no_temp_files(out_dir, expected_files)


def _write_pam_category_names(img_path, names):
    """Write band 1 category names to the PAM sidecar of ``img_path``."""
    root = ET.Element("PAMDataset")
    band = ET.SubElement(root, "PAMRasterBand", band="1")
    cats = ET.SubElement(band, "CategoryNames")
    for name in names:
        ET.SubElement(cats, "Category").text = name
    ET.ElementTree(root).write(f"{img_path}.aux.xml")


def _read_pam_category_names(img_path):
    root = ET.parse(f"{img_path}.aux.xml").getroot()
    return [cat.text for cat in root.iter("Category")]


def test_cog_rebuild_keeps_pam_sidecar(tmp_path, clean_crt_env):
    """Metadata GDAL keeps in the .aux.xml sidecar (category names) survives a rebuild."""
    _ = clean_crt_env
    cls_img = tmp_path / "cls.tif"
    cls = (np.add.outer(np.arange(HEIGHT), np.arange(WIDTH)) // 100 % 3 + 1)
    _write_raster(cls_img, cls, dtype="uint8")
    cog = tmp_path / "cls_cog.tif"
    rasterio.shutil.copy(str(cls_img), str(cog), driver="COG")
    names = ["nd", "water", "forest", "urban"]
    _write_pam_category_names(cog, names)

    rsgispy.imageutils.stats.pop_img_stats(str(cog), use_pbar=False)
    _assert_valid_cog(cog, stats=True)
    _assert_no_temp_files(tmp_path, ["cls.tif", "cls_cog.tif", "cls_cog.tif.aux.xml"])
    assert _read_pam_category_names(cog) == names

    # A stale sidecar is still removed when the new COG has none.
    rsgispy.imageprocess.write_cog(str(cls_img), str(cog))
    _assert_no_temp_files(tmp_path, ["cls.tif", "cls_cog.tif"])


def _class_grid(dtype):
    classes = np.array([1, 100, 1000])
    idx = np.add.outer(np.arange(HEIGHT) // 7, np.arange(WIDTH) // 5) % 3
    return classes[idx].astype(dtype)


def _cog_grid():
    return rsgispy.imageprocess.IntersectionGrid(
        width=WIDTH,
        height=HEIGHT,
        transform=from_origin(500000.0, 5200000.0, 10.0, 10.0),
        crs=CRS.from_epsg(32630),
        offsets={},
    )


@pytest.mark.parametrize("dtype", ["int16", "int32"])
def test_thematic_cog_overviews_use_nearest(tmp_path, clean_crt_env, dtype):
    """Thematic COGs without a colour table get nearest-neighbour overviews."""
    _ = clean_crt_env
    cls = _class_grid(dtype)
    cases = (
        ("stats", True, None, "NEAREST"),
        ("no_stats", False, None, "NEAREST"),
        ("mode", True, {"OVERVIEW_RESAMPLING": "MODE"}, "MODE"),
    )
    for name, calc_stats, opts, exp_resamp in cases:
        out_cog = tmp_path / f"cls_{name}.tif"
        with ExitStack() as stack:
            dst = rsgispy.imageprocess.open_output_dataset(
                stack, str(out_cog), "COG", dtype, _cog_grid(), creation_opts=opts
            )
            rsgispy.imageprocess.pop_stats_on_close(
                dst, calc_stats, thematic=True, use_pbar=False
            )
            dst.write(cls, 1)
        _assert_valid_cog(out_cog, stats=False)
        with rasterio.open(out_cog) as src:
            _assert_overview_resampling(src, exp_resamp)
            assert ("STATISTICS_MAXIMUM" in src.tags(1)) == calc_stats
        with rasterio.open(out_cog, overview_level=0) as ovr:
            assert set(np.unique(ovr.read(1)).tolist()) <= {1, 100, 1000}

    # write_cog uses NEAREST for a GeoTIFF whose bands are all thematic.
    thm_img = tmp_path / "thm.tif"
    _write_raster(thm_img, cls, dtype=dtype)
    with rasterio.open(thm_img, "r+") as dst:
        dst.update_tags(1, LAYER_TYPE="thematic")
    thm_cog = tmp_path / "thm_cog.tif"
    rsgispy.imageprocess.write_cog(str(thm_img), str(thm_cog))
    with rasterio.open(thm_cog) as src:
        _assert_overview_resampling(src, "NEAREST")
    with rasterio.open(thm_cog, overview_level=0) as ovr:
        assert set(np.unique(ovr.read(1)).tolist()) <= {1, 100, 1000}


def test_cog_output_to_gdal_virtual_path(tmp_path, clean_crt_env):
    """COG outputs to /vsimem/ paths are staged locally and leave no temporary files."""
    monkeypatch = clean_crt_env
    staged = []
    orig_staging_path = rsgis_io._staging_path  # pylint: disable=protected-access

    def _record_staging_path(output_img):
        path = orig_staging_path(output_img)
        staged.append(path)
        return path

    monkeypatch.setattr(rsgis_io, "_staging_path", _record_staging_path)
    in_img = tmp_path / "in.tif"
    _write_raster(in_img, _ramp())
    vsi_dir = f"/vsimem/rsgispy_test_{uuid.uuid4().hex}"
    out_cog = f"{vsi_dir}/out_cog.tif"
    direct_cog = f"{vsi_dir}/direct_cog.tif"
    band_defs = [rsgispy.imageprocess.BandDefn("b1", str(in_img), 1)]
    try:
        for factor in (2, 3):  # the second call replaces the output
            rsgispy.imagecalc.maths.band_maths(
                output_img=out_cog,
                exp=f"b1 * {factor}",
                band_defs=band_defs,
                gdalformat="COG",
                datatype=rsgispy.TYPE_32FLOAT,
                calc_stats=True,
                use_pbar=False,
            )
        with rasterio.open(out_cog) as src:
            assert src.tags(ns="IMAGE_STRUCTURE").get("LAYOUT") == "COG"
            assert src.overviews(1)
            assert "STATISTICS_MEAN" in src.tags(1)
            np.testing.assert_array_equal(src.read(1), _ramp()[0] * 3)
        rsgispy.imageprocess.write_cog(str(in_img), direct_cog)
        with rasterio.open(direct_cog) as src:
            assert src.tags(ns="IMAGE_STRUCTURE").get("LAYOUT") == "COG"

        # The temporary GeoTIFFs were local files and have been deleted.
        assert staged
        for path in staged:
            assert not path.startswith("/vsi")
            assert not any(
                os.path.exists(path + ext) for ext in ("", ".aux.xml", ".ovr", ".msk")
            )
        try:
            from osgeo import gdal  # pylint: disable=import-outside-toplevel
        except ImportError:  # pragma: no cover - GDAL Python bindings are optional
            gdal = None
        # Only check the listing when osgeo shares rasterio's GDAL (/vsimem/).
        if gdal is not None and gdal.VSIStatL(out_cog) is not None:
            assert sorted(gdal.ReadDir(vsi_dir)) == ["direct_cog.tif", "out_cog.tif"]
    finally:
        for path in (out_cog, direct_cog):
            if rasterio.shutil.exists(path):
                rasterio.shutil.delete(path)


def test_set_img_pxl_values_keeps_cog_layout(tmp_path, clean_crt_env):
    """set_img_pxl_values on a COG rebuilds it instead of breaking the layout."""
    import rsgispy.imageutils.arrays  # pylint: disable=import-outside-toplevel

    src = tmp_path / "ramp.tif"
    _write_raster(src, _ramp())
    cog = str(tmp_path / "ramp_cog.tif")
    rsgispy.imagecalc.maths.band_maths(
        cog, "b1", [rsgispy.imagecalc.maths.BandDefn("b1", str(src), 1)],
        gdalformat="COG", calc_stats=True, use_pbar=False,
    )
    rsgispy.imageutils.arrays.set_img_pxl_values(cog, 1, [3, 7], [2, 5], pxl_value=-5)
    _assert_valid_cog(cog, stats=False)
    with rasterio.open(cog) as ds:
        arr = ds.read(1)
    assert arr[2, 3] == -5 and arr[5, 7] == -5
