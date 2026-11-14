"""
Tests for rsgispy.imageutils image creation option helpers, image statistics,
thematic colour tables, overview pyramids, and integration with band_maths.
"""

import json

import geopandas
import numpy as np
import pytest
import rasterio
from rasterio.enums import ColorInterp, Compression
from rasterio.transform import from_origin
from shapely.geometry import box

import rsgispy
import rsgispy.imagecalc.maths
import rsgispy.imageutils.blank
import rsgispy.imageutils.colourtable
import rsgispy.imageutils.envopts
import rsgispy.imageutils.masking
import rsgispy.imageutils.metadata
import rsgispy.imageutils.mosaic
import rsgispy.imageutils.stats
import rsgispy.imageutils.subsetting
import rsgispy.imageutils.tiling
import rsgispy.imageutils.vrt
import rsgispy.imageutils.warp


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

    assert rsgispy.imageutils.envopts.get_gdal_img_creation_opts("GTIFF") == {}

    rsgispy.imageutils.envopts.set_env_vars_deflate_gtiff_outs(bigtiff=False)
    opts = rsgispy.imageutils.envopts.get_gdal_img_creation_opts("GTIFF")
    assert opts == {
        "TILED": "YES",
        "COMPRESS": "DEFLATE",
        "INTERLEAVE": "BAND",
        "BIGTIFF": "IF_SAFER",
    }

    rsgispy.imageutils.envopts.set_env_vars_lzw_gtiff_outs(bigtiff=True)
    opts = rsgispy.imageutils.envopts.get_gdal_img_creation_opts("gtiff")
    assert opts == {
        "TILED": "YES",
        "COMPRESS": "LZW",
        "INTERLEAVE": "BAND",
        "BIGTIFF": "YES",
    }

    rsgispy.imageutils.envopts.set_env_vars_zstd_gtiff_outs(bigtiff=False)
    opts = rsgispy.imageutils.envopts.get_gdal_img_creation_opts("GTIFF")
    assert opts["COMPRESS"] == "ZSTD"
    assert opts["BIGTIFF"] == "IF_SAFER"

    rsgispy.imageutils.envopts.set_env_vars_lzw_cog_outs(bigtiff=False)
    assert rsgispy.imageutils.envopts.get_gdal_img_creation_opts("COG") == {
        "COMPRESS": "LZW",
        "BIGTIFF": "IF_SAFER",
    }

    rsgispy.imageutils.envopts.set_env_vars_deflate_cog_outs(bigtiff=True)
    assert rsgispy.imageutils.envopts.get_gdal_img_creation_opts("COG") == {
        "COMPRESS": "DEFLATE",
        "BIGTIFF": "YES",
    }

    rsgispy.imageutils.envopts.set_img_creation_opts(
        "GTIFF", {"COMPRESS": "DEFLATE", "PREDICTOR": 2}
    )
    assert rsgispy.imageutils.envopts.get_gdal_img_creation_opts("GTIFF") == {
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
        rsgispy.imagecalc.maths.BandDefn(band_name="b1", input_img=str(in_img), img_band=1)
    ]

    # 1. Via set_env_vars_deflate_gtiff_outs
    rsgispy.imageutils.envopts.set_env_vars_deflate_gtiff_outs()
    rsgispy.imagecalc.maths.band_maths(
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
    rsgispy.imagecalc.maths.band_maths(
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

    rsgispy.imageutils.stats.pop_thmt_img_stats(
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

    rsgispy.imageutils.stats.pop_thmt_img_stats(
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

    rsgispy.imageutils.stats.pop_thmt_img_stats(
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
        rsgispy.imageutils.stats.pop_thmt_img_stats(str(float_img), use_pbar=False)


def test_pop_img_stats(tmp_path):
    """Test pop_img_stats and calc_basic_img_band_stats on a continuous raster."""
    cont_img = tmp_path / "continuous.tif"

    arr1 = np.linspace(0.0, 100.0, 200 * 200, dtype=np.float32).reshape(200, 200)
    arr1[0, :] = -999.0  # No-data row
    arr2 = np.full((200, 200), 42.0, dtype=np.float32)
    _create_simple_raster(cont_img, np.stack([arr1, arr2]), dtype="float32")

    rsgispy.imageutils.stats.pop_img_stats(
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


def test_create_blank_and_copy_and_bbox_and_buf_imgs(tmp_path, monkeypatch):
    """Test create_blank_img, create_copy_img, create_blank_buf_img_from_ref_img, and create_blank_img_from_bbox."""
    monkeypatch.delenv("RSGISLIB_IMG_CRT_OPTS_GTIFF", raising=False)
    monkeypatch.delenv("RSGISPY_IMG_CRT_OPTS_GTIFF", raising=False)
    rsgispy.imageutils.envopts.set_env_vars_deflate_gtiff_outs()

    blank_img = tmp_path / "blank.tif"
    copy_img = tmp_path / "copy.tif"
    buf_pxl_img = tmp_path / "buf_pxl.tif"
    buf_spt_img = tmp_path / "buf_spt.tif"
    bbox_img = tmp_path / "bbox.tif"

    # 1. create_blank_img using epsg_code
    rsgispy.imageutils.blank.create_blank_img(
        output_img=str(blank_img),
        n_bands=2,
        width=50,
        height=40,
        tl_x=500000.0,
        tl_y=5200000.0,
        res_x=10.0,
        res_y=-10.0,
        pxl_val=-999.0,
        epsg_code=32630,
        gdalformat="GTIFF",
        datatype=rsgispy.TYPE_32FLOAT,
        no_data_val=-999.0,
    )

    with rasterio.open(blank_img) as src:
        assert src.count == 2
        assert src.width == 50
        assert src.height == 40
        assert src.nodata == -999.0
        assert src.crs.to_epsg() == 32630
        assert src.transform.c == 500000.0
        assert src.transform.f == 5200000.0
        assert src.transform.a == 10.0
        assert src.transform.e == -10.0
        assert src.compression == Compression.deflate
        assert src.profile["tiled"] is True
        np.testing.assert_allclose(src.read(), -999.0)

    # 2. create_copy_img
    rsgispy.imageutils.blank.create_copy_img(
        input_img=str(blank_img),
        output_img=str(copy_img),
        n_bands=3,
        pxl_val=7.0,
        gdalformat="GTIFF",
        datatype=rsgispy.TYPE_16UINT,
    )

    with rasterio.open(copy_img) as src:
        assert src.count == 3
        assert src.width == 50
        assert src.height == 40
        assert src.dtypes[0] == "uint16"
        assert src.crs.to_epsg() == 32630
        assert src.compression == Compression.deflate
        assert src.profile["tiled"] is True
        assert np.all(src.read() == 7)

    # 3. create_blank_buf_img_from_ref_img (pixel buffer and spatial buffer)
    rsgispy.imageutils.blank.create_blank_buf_img_from_ref_img(
        input_img=str(blank_img),
        output_img=str(buf_pxl_img),
        gdalformat="GTIFF",
        datatype=rsgispy.TYPE_32FLOAT,
        buf_pxl_ext=5,
    )
    with rasterio.open(buf_pxl_img) as src:
        assert src.width == 60
        assert src.height == 50
        assert src.transform.c == 500000.0 - 50.0
        assert src.transform.f == 5200000.0 + 50.0
        assert src.nodata == -999.0
        assert src.compression == Compression.deflate

    rsgispy.imageutils.blank.create_blank_buf_img_from_ref_img(
        input_img=str(blank_img),
        output_img=str(buf_spt_img),
        gdalformat="GTIFF",
        datatype=rsgispy.TYPE_32FLOAT,
        buf_spt_ext=100.0,
    )
    with rasterio.open(buf_spt_img) as src:
        assert src.width == 70
        assert src.height == 60
        assert src.transform.c == 500000.0 - 100.0
        assert src.transform.f == 5200000.0 + 100.0
        assert src.compression == Compression.deflate

    with pytest.raises(rsgispy.RSGISPyException):
        rsgispy.imageutils.blank.create_blank_buf_img_from_ref_img(
            str(blank_img), str(buf_pxl_img), "GTIFF", rsgispy.TYPE_32FLOAT
        )

    # 4. create_blank_img_from_bbox
    rsgispy.imageutils.blank.create_blank_img_from_bbox(
        bbox=(500000.0, 500250.0, 5199700.0, 5200000.0),
        epsg_code=32630,
        output_img=str(bbox_img),
        out_img_res=10.0,
        out_img_pxl_val=1.0,
        out_img_n_bands=1,
        gdalformat="GTIFF",
        datatype=rsgispy.TYPE_8UINT,
        snap_to_grid=True,
    )
    with rasterio.open(bbox_img) as src:
        assert src.width == 25
        assert src.height == 30
        assert src.crs.to_epsg() == 32630
        assert src.compression == Compression.deflate
        assert np.all(src.read(1) == 1)


def test_resample_img_to_match_and_reproject_image(tmp_path, monkeypatch):
    """Test resample_img_to_match and reproject_image with EPSG codes and env creation opts."""
    monkeypatch.delenv("RSGISLIB_IMG_CRT_OPTS_GTIFF", raising=False)
    monkeypatch.delenv("RSGISPY_IMG_CRT_OPTS_GTIFF", raising=False)
    rsgispy.imageutils.envopts.set_env_vars_deflate_gtiff_outs()

    ref_img = tmp_path / "ref.tif"
    proc_img = tmp_path / "proc.tif"
    resamp_img = tmp_path / "resampled.tif"
    reproj_img = tmp_path / "reprojected.tif"
    reproj_default_opts_img = tmp_path / "reprojected_deflate.tif"

    # Reference image at 10m resolution (20x20)
    rsgispy.imageutils.blank.create_blank_img(
        output_img=str(ref_img),
        n_bands=1,
        width=20,
        height=20,
        tl_x=500000.0,
        tl_y=5200000.0,
        res_x=10.0,
        res_y=-10.0,
        pxl_val=0.0,
        epsg_code=32630,
        gdalformat="GTIFF",
        datatype=rsgispy.TYPE_32FLOAT,
    )

    # Process image at 20m resolution (10x10) with constant value 42.0
    rsgispy.imageutils.blank.create_blank_img(
        output_img=str(proc_img),
        n_bands=2,
        width=10,
        height=10,
        tl_x=500000.0,
        tl_y=5200000.0,
        res_x=20.0,
        res_y=-20.0,
        pxl_val=42.0,
        epsg_code=32630,
        gdalformat="GTIFF",
        datatype=rsgispy.TYPE_32FLOAT,
        no_data_val=-999.0,
    )

    # 1. resample_img_to_match (uses env var creation options when options=None)
    rsgispy.imageutils.warp.resample_img_to_match(
        in_ref_img=str(ref_img),
        in_process_img=str(proc_img),
        output_img=str(resamp_img),
        gdalformat="GTIFF",
        interp_method=rsgispy.INTERP_NEAREST_NEIGHBOUR,
    )

    with rasterio.open(resamp_img) as src:
        assert src.count == 2
        assert src.width == 20
        assert src.height == 20
        assert src.transform.a == 10.0
        assert src.nodata == -999.0
        assert src.compression == Compression.deflate
        assert src.profile["tiled"] is True
        np.testing.assert_allclose(src.read(), 42.0)

    # 2. reproject_image from EPSG:32630 to EPSG:3857 using env var options
    rsgispy.imageutils.warp.reproject_image(
        input_img=str(proc_img),
        output_img=str(reproj_default_opts_img),
        out_epsg=3857,
        gdalformat="GTIFF",
        interp_method=rsgispy.INTERP_BILINEAR,
        no_data_val=-999.0,
        out_pxl_res="auto",
        snap_to_grid=True,
    )

    with rasterio.open(reproj_default_opts_img) as src:
        assert src.crs.to_epsg() == 3857
        assert src.compression == Compression.deflate
        assert src.profile["tiled"] is True

    # 3. reproject_image with explicit gdal_options overriding env vars
    rsgispy.imageutils.warp.reproject_image(
        input_img=str(proc_img),
        output_img=str(reproj_img),
        out_epsg=3857,
        gdalformat="GTIFF",
        interp_method=rsgispy.INTERP_BILINEAR,
        no_data_val=-999.0,
        out_pxl_res="auto",
        snap_to_grid=True,
        gdal_options=["COMPRESS=LZW", "TILED=YES"],
    )

    with rasterio.open(reproj_img) as src:
        assert src.count == 2
        assert src.crs.to_epsg() == 3857
        assert src.nodata == -999.0
        assert src.compression == Compression.lzw
        data = src.read(1)
        valid = data[data != -999.0]
        assert valid.size > 0
        np.testing.assert_allclose(valid, 42.0, rtol=1e-4)


def test_create_stack_mosaic_and_band_subset_vrts(tmp_path):
    """Test create_stack_images_vrt, create_mosaic_images_vrt, and create_vrt_band_subset."""
    img1 = tmp_path / "img1.tif"
    img2 = tmp_path / "img2.tif"
    stack_vrt = tmp_path / "stack.vrt"
    mosaic_vrt = tmp_path / "mosaic.vrt"
    mosaic_pf_vrt = tmp_path / "mosaic_pf.vrt"
    subset_vrt = tmp_path / "subset.vrt"

    # img1: 10x10 at x=[500000, 500100], y=[5199900, 5200000], 2 bands (values 10, 20)
    with rasterio.open(
        img1,
        "w",
        driver="GTiff",
        width=10,
        height=10,
        count=2,
        dtype="uint16",
        crs="EPSG:32630",
        transform=from_origin(500000.0, 5200000.0, 10.0, 10.0),
        nodata=0,
    ) as dst:
        dst.write(np.full((10, 10), 10, dtype="uint16"), 1)
        dst.write(np.full((10, 10), 20, dtype="uint16"), 2)
        dst.descriptions = ("b1_a", "b2_a")

    # img2: 10x10 overlapping half of img1 at x=[500050, 500150], y=[5199900, 5200000], 2 bands (values 30, 40)
    with rasterio.open(
        img2,
        "w",
        driver="GTiff",
        width=10,
        height=10,
        count=2,
        dtype="uint16",
        crs="EPSG:32630",
        transform=from_origin(500050.0, 5200000.0, 10.0, 10.0),
        nodata=0,
    ) as dst:
        dst.write(np.full((10, 10), 30, dtype="uint16"), 1)
        dst.write(np.full((10, 10), 40, dtype="uint16"), 2)
        dst.descriptions = ("b1_b", "b2_b")

    # 1. create_stack_images_vrt with custom band_names, options, and resolution
    rsgispy.imageutils.vrt.create_stack_images_vrt(
        input_imgs=[str(img1), str(img2)],
        out_vrt_file=str(stack_vrt),
        resolution="highest",
        interp_method=rsgispy.INTERP_NEAREST_NEIGHBOUR,
        band_names=["S1", "S2", "S3", "S4"],
        options={"BLOCKXSIZE": 128, "BLOCKYSIZE": 128},
    )

    with rasterio.open(stack_vrt) as src:
        assert src.count == 4
        assert src.width == 15
        assert src.height == 10
        assert src.crs.to_epsg() == 32630
        assert src.descriptions == ("S1", "S2", "S3", "S4")
        assert src.block_shapes[0] == (128, 128)
        data = src.read()
        # Left 5 columns: only img1 present
        assert np.all(data[0, :, :5] == 10)
        assert np.all(data[1, :, :5] == 20)
        assert np.all(data[2, :, :5] == 0)
        # Right 5 columns: only img2 present
        assert np.all(data[0, :, 10:] == 0)
        assert np.all(data[2, :, 10:] == 30)
        assert np.all(data[3, :, 10:] == 40)

    # 2. create_mosaic_images_vrt with custom extent, resolution, align_out_pxls, and add_alpha
    rsgispy.imageutils.vrt.create_mosaic_images_vrt(
        input_imgs=[str(img1), str(img2)],
        out_vrt_file=str(mosaic_vrt),
        vrt_extent=(500000.0, 5199900.0, 500150.0, 5200000.0),
        vrt_out_res_x=10.0,
        vrt_out_res_y=10.0,
        interp_method=rsgispy.INTERP_BILINEAR,
        align_out_pxls=True,
        add_alpha=True,
    )

    with rasterio.open(mosaic_vrt) as src:
        assert src.count == 3  # 2 bands (uint16) + 1 alpha band (Byte)
        assert src.width == 15
        assert src.height == 10
        band1_data = src.read(1)
        alpha_data = src.read(3)
        # Left 5 cols from img1 (10), right 10 cols overwritten by img2 (30)
        assert np.all(band1_data[:, :5] == 10)
        assert np.all(band1_data[:, 5:] == 30)
        assert np.all(alpha_data == 255)

    # 3. create_mosaic_images_vrt with pixel_function="sum" and band_list=[1]
    rsgispy.imageutils.vrt.create_mosaic_images_vrt(
        input_imgs=[str(img1), str(img2)],
        out_vrt_file=str(mosaic_pf_vrt),
        band_list=[1],
        pixel_function="sum",
        hide_nodata=True,
    )

    with rasterio.open(mosaic_pf_vrt) as src:
        assert src.count == 1
        assert src.nodata is None
        data = src.read(1)
        assert np.all(data[:, :5] == 10)
        assert np.all(data[:, 5:10] == 40)  # 10 + 30 in overlap region
        assert np.all(data[:, 10:] == 30)

    # 4. create_vrt_band_subset with reversed band order [2, 1] and custom band_names
    rsgispy.imageutils.vrt.create_vrt_band_subset(
        input_img=str(img1),
        img_bands=[2, 1],
        out_vrt_img=str(subset_vrt),
        band_names=["SecondBand", "FirstBand"],
        vrt_nodata=999,
        write_abs_path=True,
    )

    with rasterio.open(subset_vrt) as src:
        assert src.count == 2
        assert src.width == 10
        assert src.height == 10
        assert src.nodata == 999.0
        assert src.descriptions == ("SecondBand", "FirstBand")
        assert np.all(src.read(1) == 20)
        assert np.all(src.read(2) == 10)

    # Verify validation error on out-of-range band
    with pytest.raises(rsgispy.RSGISPyException):
        rsgispy.imageutils.vrt.create_vrt_band_subset(
            input_img=str(img1),
            img_bands=[3],
            out_vrt_img=str(subset_vrt),
        )


def test_img_metadata_nodata_res_size_bbox_and_format(tmp_path):
    """Test nodata, resolution, size, band count, bbox, files, format, and thematic metadata."""
    img_path = tmp_path / "meta.tif"
    arr = np.ones((2, 20, 30), dtype=np.uint8)
    with rasterio.open(
        img_path,
        "w",
        driver="GTiff",
        width=30,
        height=20,
        count=2,
        dtype="uint8",
        crs="EPSG:32630",
        transform=from_origin(500000.0, 5200000.0, 10.0, 10.0),
    ) as dst:
        dst.write(arr)

    assert rsgispy.imageutils.metadata.get_img_no_data_value(str(img_path)) is None
    rsgispy.imageutils.metadata.set_img_no_data_value(str(img_path), 255.0)
    assert rsgispy.imageutils.metadata.get_img_no_data_value(str(img_path), img_band=1) == 255.0
    assert rsgispy.imageutils.metadata.get_img_no_data_value(str(img_path), img_band=2) == 255.0

    rsgispy.imageutils.metadata.set_img_no_data_value(str(img_path), 0.0, img_band=2)
    assert rsgispy.imageutils.metadata.get_img_no_data_value(str(img_path), img_band=2) == 0.0

    with pytest.raises(rsgispy.RSGISPyException):
        rsgispy.imageutils.metadata.get_img_no_data_value(str(img_path), img_band=3)
    with pytest.raises(rsgispy.RSGISPyException):
        rsgispy.imageutils.metadata.set_img_no_data_value(str(img_path), 0.0, img_band=0)

    assert rsgispy.imageutils.metadata.get_img_res(str(img_path)) == (10.0, -10.0)
    assert rsgispy.imageutils.metadata.get_img_res(str(img_path), abs_vals=True) == (10.0, 10.0)
    assert rsgispy.imageutils.metadata.get_img_size(str(img_path)) == (30, 20)
    assert rsgispy.imageutils.metadata.get_img_band_count(str(img_path)) == 2
    assert rsgispy.imageutils.metadata.get_img_bbox(str(img_path)) == (
        500000.0,
        500300.0,
        5199800.0,
        5200000.0,
    )
    assert rsgispy.imageutils.metadata.get_gdal_format_name(str(img_path)) == "GTIFF"
    assert str(img_path) in rsgispy.imageutils.metadata.get_img_files(str(img_path))

    assert rsgispy.imageutils.colourtable.is_img_thematic(str(img_path)) is False
    rsgispy.imageutils.colourtable.set_img_thematic(str(img_path))
    assert rsgispy.imageutils.colourtable.is_img_thematic(str(img_path), img_band=1) is True
    assert rsgispy.imageutils.colourtable.is_img_thematic(str(img_path), img_band=2) is True


def test_img_spatial_coords_subset_and_reproj_bbox(tmp_path):
    """Test get_img_bbox_in_proj, get_img_subset_pxl_bbox, get_img_pxl_spatial_coords, and get_img_pxl_coords."""
    img_path = tmp_path / "coords.tif"
    with rasterio.open(
        img_path,
        "w",
        driver="GTiff",
        width=40,
        height=30,
        count=1,
        dtype="uint8",
        crs="EPSG:32630",
        transform=from_origin(500000.0, 5200000.0, 10.0, 10.0),
    ) as dst:
        dst.write(np.ones((1, 30, 40), dtype="uint8"))

    # Reprojected bbox to WGS84 (EPSG:4326)
    wgs84_bbox = rsgispy.imageutils.metadata.get_img_bbox_in_proj(str(img_path), out_epsg=4326)
    assert len(wgs84_bbox) == 4
    assert -10.0 < wgs84_bbox[0] < wgs84_bbox[1] < 0.0
    assert 40.0 < wgs84_bbox[2] < wgs84_bbox[3] < 50.0

    # Subset pixel bbox (partially outside -> clipped to image bounds)
    pxl_bbox = rsgispy.imageutils.metadata.get_img_subset_pxl_bbox(
        str(img_path), (499950.0, 500150.0, 5199800.0, 5199950.0)
    )
    assert pxl_bbox == (0, 15, 5, 20)

    with pytest.raises(rsgispy.RSGISPyException):
        rsgispy.imageutils.metadata.get_img_subset_pxl_bbox(
            str(img_path), (600000.0, 600100.0, 5199800.0, 5199900.0)
        )

    # Pixel bbox back to spatial coordinates
    spt_bbox = rsgispy.imageutils.metadata.get_img_pxl_spatial_coords(str(img_path), pxl_bbox)
    assert spt_bbox == (500000.0, 500150.0, 5199800.0, 5199950.0)

    # Point coordinate conversion
    xs, ys = rsgispy.imageutils.metadata.get_img_pxl_coords(
        str(img_path),
        np.array([500005.0, 500105.0]),
        np.array([5199995.0, 5199895.0]),
    )
    np.testing.assert_array_equal(xs, np.array([1, 11]))
    np.testing.assert_array_equal(ys, np.array([1, 11]))

    with pytest.raises(rsgispy.RSGISPyException):
        rsgispy.imageutils.metadata.get_img_pxl_coords(
            str(img_path), np.array([499000.0]), np.array([5199900.0])
        )
    with pytest.raises(rsgispy.RSGISPyException):
        rsgispy.imageutils.metadata.get_img_pxl_coords(
            str(img_path), np.array([500100.0]), np.array([5100000.0])
        )


def test_proj_spatial_assignment_copy_utm_and_gcps(tmp_path):
    """Test projection/spatial assignment & copying, EPSG/WKT/UTM queries, and GCP helpers."""
    ref_img = tmp_path / "ref_utm.tif"
    target_img = tmp_path / "target.tif"
    wkt_file = tmp_path / "proj.wkt"

    with rasterio.open(
        ref_img,
        "w",
        driver="GTiff",
        width=10,
        height=10,
        count=1,
        dtype="uint8",
        crs="EPSG:32605",
        transform=from_origin(400000.0, 6500000.0, 20.0, 20.0),
    ) as dst:
        dst.write(np.zeros((1, 10, 10), dtype="uint8"))

    # Start target with no CRS and unit transform
    with rasterio.open(
        target_img,
        "w",
        driver="GTiff",
        width=10,
        height=10,
        count=1,
        dtype="uint8",
    ) as dst:
        dst.write(np.zeros((1, 10, 10), dtype="uint8"))

    assert rsgispy.imageutils.metadata.get_wkt_proj_from_img(str(target_img)) == ""
    assert rsgispy.imageutils.metadata.get_epsg_proj_from_img(str(target_img)) is None
    assert rsgispy.imageutils.metadata.get_utm_zone(str(target_img)) is None

    # Assign WKT via string (WGS 84 / UTM zone 55S -> EPSG:32755)
    wkt_55s = rasterio.crs.CRS.from_epsg(32755).to_wkt()
    rsgispy.imageutils.metadata.assign_proj(str(target_img), wkt_str=wkt_55s)
    assert rsgispy.imageutils.metadata.get_epsg_proj_from_img(str(target_img)) == 32755
    assert rsgispy.imageutils.metadata.get_utm_zone(str(target_img)) == "55S"
    assert "32755" in rsgispy.imageutils.metadata.get_wkt_proj_from_img(str(target_img))

    # Assign WKT via file (EPSG:4326 -> not UTM -> None)
    wkt_file.write_text(
        rasterio.crs.CRS.from_epsg(4326).to_wkt(), encoding="utf-8"
    )
    rsgispy.imageutils.metadata.assign_proj(str(target_img), wkt_file=str(wkt_file))
    assert rsgispy.imageutils.metadata.get_epsg_proj_from_img(str(target_img)) == 4326
    assert rsgispy.imageutils.metadata.get_utm_zone(str(target_img)) is None

    # Assign via EPSG code (WGS 84 / UTM zone 30N -> EPSG:32630)
    rsgispy.imageutils.metadata.assign_proj(str(target_img), epsg_code=32630)
    assert rsgispy.imageutils.metadata.get_epsg_proj_from_img(str(target_img)) == 32630
    assert rsgispy.imageutils.metadata.get_utm_zone(str(target_img)) == "30N"

    with pytest.raises(rsgispy.RSGISPyException):
        rsgispy.imageutils.metadata.assign_proj(str(target_img))

    with pytest.raises(rsgispy.RSGISPyException):
        rsgispy.imageutils.metadata.assign_proj(
            str(target_img), wkt_str=wkt_55s, epsg_code=32630
        )

    # assign_spatial_info (partial & full updates)
    rsgispy.imageutils.metadata.assign_spatial_info(
        str(target_img),
        tl_x=100.0,
        tl_y=200.0,
        res_x=5.0,
        res_y=-5.0,
        rot_x=0.0,
        rot_y=0.0,
    )
    rsgispy.imageutils.metadata.assign_spatial_info(str(target_img), tl_x=150.0)
    assert rsgispy.imageutils.metadata.get_img_bbox(str(target_img)) == (
        150.0,
        200.0,
        150.0,
        200.0,
    )

    # copy_proj_from_img
    rsgispy.imageutils.metadata.copy_proj_from_img(str(target_img), str(ref_img))
    assert rsgispy.imageutils.metadata.get_epsg_proj_from_img(str(target_img)) == 32605
    assert rsgispy.imageutils.metadata.get_utm_zone(str(target_img)) == "05N"
    assert rsgispy.imageutils.metadata.get_img_res(str(target_img)) == (5.0, -5.0)

    # copy_spatial_and_proj_from_img
    rsgispy.imageutils.metadata.copy_spatial_and_proj_from_img(str(target_img), str(ref_img))
    assert rsgispy.imageutils.metadata.get_img_res(str(target_img)) == (20.0, -20.0)
    assert rsgispy.imageutils.metadata.get_img_bbox(str(target_img)) == (
        400000.0,
        400200.0,
        6499800.0,
        6500000.0,
    )

    # GCP helpers: has_gcps & copy_gcps
    gcp_src = tmp_path / "gcp_src.tif"
    gcp_dst = tmp_path / "gcp_dst.tif"
    gcps = [
        rasterio.control.GroundControlPoint(row=0, col=0, x=10.0, y=20.0, z=0.0, id="1"),
        rasterio.control.GroundControlPoint(row=9, col=9, x=19.0, y=11.0, z=0.0, id="2"),
    ]
    with rasterio.open(
        gcp_src,
        "w",
        driver="GTiff",
        width=10,
        height=10,
        count=1,
        dtype="uint8",
        gcps=gcps,
        crs="EPSG:4326",
    ) as dst:
        dst.write(np.zeros((1, 10, 10), dtype="uint8"))

    with rasterio.open(
        gcp_dst,
        "w",
        driver="GTiff",
        width=10,
        height=10,
        count=1,
        dtype="uint8",
    ) as dst:
        dst.write(np.zeros((1, 10, 10), dtype="uint8"))

    assert rsgispy.imageutils.metadata.has_gcps(str(ref_img)) is False
    assert rsgispy.imageutils.metadata.has_gcps(str(gcp_src)) is True
    assert rsgispy.imageutils.metadata.has_gcps(str(gcp_dst)) is False

    rsgispy.imageutils.metadata.copy_gcps(str(gcp_src), str(gcp_dst))
    assert rsgispy.imageutils.metadata.has_gcps(str(gcp_dst)) is True
    with rasterio.open(gcp_dst) as ds_gcp:
        copied_gcps, copied_crs = ds_gcp.gcps
        assert len(copied_gcps) == 2
        assert copied_crs.to_epsg() == 4326


def _write_geo_raster(
    path,
    data,
    tl_x=500000.0,
    tl_y=5200000.0,
    res=10.0,
    epsg=32630,
    dtype="float32",
    nodata=None,
    descriptions=None,
):
    """Helper to write a positioned rasterio GeoTIFF for tests."""
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
        crs=f"EPSG:{epsg}",
        transform=from_origin(tl_x, tl_y, res, res),
        nodata=nodata,
    ) as dst:
        dst.write(data.astype(dtype))
        if descriptions:
            for bidx, desc in enumerate(descriptions, start=1):
                dst.set_band_description(bidx, desc)


def test_gdal_warp(tmp_path):
    """Test gdal_warp reprojection preserves values and assigns target EPSG."""
    src_img = tmp_path / "warp_src.tif"
    dst_img = tmp_path / "warp_dst.tif"
    arr = np.full((1, 20, 20), 7.5, dtype=np.float32)
    _write_geo_raster(
        src_img,
        arr,
        tl_x=500000.0,
        tl_y=5200000.0,
        res=10.0,
        epsg=32630,
        nodata=-999.0,
    )

    rsgispy.imageutils.warp.gdal_warp(
        input_img=str(src_img),
        output_img=str(dst_img),
        out_epsg=3857,
        interp_method=rsgispy.INTERP_NEAREST_NEIGHBOUR,
        gdalformat="GTIFF",
        use_multi_threaded=True,
    )

    with rasterio.open(dst_img) as src:
        assert src.crs.to_epsg() == 3857
        assert src.nodata == -999.0
        vals = src.read(1)
        assert np.any(np.isclose(vals, 7.5))


def test_create_img_mosaic(tmp_path):
    """Test create_img_mosaic with overlap_behaviour 0, 1 (min), and 2 (max)."""
    img_a = tmp_path / "mosaic_a.tif"
    img_b = tmp_path / "mosaic_b.tif"

    # img_a covers cols 0..10 (500000..500100), rows 0..10 (5200000..5199900)
    arr_a = np.full((2, 10, 10), 10.0, dtype=np.float32)
    arr_a[0, 0:2, :] = 0.0  # skip pixels in band 1
    arr_a[1, :, :] = 100.0
    _write_geo_raster(
        img_a,
        arr_a,
        tl_x=500000.0,
        tl_y=5200000.0,
        res=10.0,
        descriptions=["b1", "b2"],
    )

    # img_b shifts right by 5 pixels (500050..500150), overlapping cols 5..10
    arr_b = np.full((2, 10, 10), 20.0, dtype=np.float32)
    arr_b[0, 5:, :] = 5.0  # lower than 10.0 on bottom half, higher on top half
    arr_b[1, :, :] = 200.0
    _write_geo_raster(
        img_b, arr_b, tl_x=500050.0, tl_y=5200000.0, res=10.0
    )

    # Behaviour 0: overwrite
    out_0 = tmp_path / "mosaic_0.tif"
    rsgispy.imageutils.mosaic.create_img_mosaic(
        [str(img_a), str(img_b)],
        str(out_0),
        background_val=-99.0,
        skip_val=0.0,
        skip_band=1,
        overlap_behaviour=0,
        gdalformat="GTIFF",
        datatype=rsgispy.TYPE_32FLOAT,
    )
    with rasterio.open(out_0) as src:
        assert src.width == 15
        assert src.height == 10
        assert src.descriptions == ("b1", "b2")
        b1 = src.read(1)
        b2 = src.read(2)
        assert b1[0, 0] == -99.0  # skipped row from A not covered by B
        assert b1[3, 2] == 10.0  # non-overlap A
        assert b1[3, 7] == 20.0  # overlap overwritten by B
        assert b2[3, 7] == 200.0

    # Behaviour 1: min of skip_band
    out_1 = tmp_path / "mosaic_1.tif"
    rsgispy.imageutils.mosaic.create_img_mosaic(
        [str(img_a), str(img_b)],
        str(out_1),
        background_val=-99.0,
        skip_val=0.0,
        skip_band=1,
        overlap_behaviour=1,
        gdalformat="GTIFF",
        datatype=rsgispy.TYPE_32FLOAT,
    )
    with rasterio.open(out_1) as src:
        b1 = src.read(1)
        assert b1[3, 7] == 10.0  # min(10, 20) == 10
        assert b1[7, 7] == 5.0  # min(10, 5) == 5

    # Behaviour 2: max of skip_band
    out_2 = tmp_path / "mosaic_2.tif"
    rsgispy.imageutils.mosaic.create_img_mosaic(
        [str(img_a), str(img_b)],
        str(out_2),
        background_val=-99.0,
        skip_val=0.0,
        skip_band=1,
        overlap_behaviour=2,
        gdalformat="GTIFF",
        datatype=rsgispy.TYPE_32FLOAT,
    )
    with rasterio.open(out_2) as src:
        b1 = src.read(1)
        assert b1[3, 7] == 20.0  # max(10, 20) == 20
        assert b1[7, 7] == 10.0  # max(10, 5) == 10


def test_include_imgs_and_with_overlap(tmp_path):
    """Test include_imgs (skip_val and input_bands) and include_imgs_with_overlap."""
    base_img = tmp_path / "base.tif"
    patch_img = tmp_path / "patch.tif"
    ovl_tile = tmp_path / "ovl_tile.tif"

    base_arr = np.ones((2, 10, 10), dtype=np.float32) * 5.0
    _write_geo_raster(base_img, base_arr, tl_x=0.0, tl_y=10.0, res=1.0)

    # 3-band 4x4 patch placed at cols 2..6, rows 2..6; select bands [3, 1], skip_val=-1
    patch_arr = np.zeros((3, 4, 4), dtype=np.float32)
    patch_arr[0, :, :] = 11.0
    patch_arr[2, :, :] = 33.0
    patch_arr[2, 0, 0] = -1.0  # should preserve base 5.0 at (row=2, col=2) band 1
    _write_geo_raster(patch_img, patch_arr, tl_x=2.0, tl_y=8.0, res=1.0)

    rsgispy.imageutils.mosaic.include_imgs(
        input_img=str(base_img),
        include_imgs=[str(patch_img)],
        input_bands=[3, 1],
        skip_val=-1.0,
    )
    with rasterio.open(base_img) as src:
        out = src.read()
        assert out[0, 2, 2] == 5.0  # skipped
        assert out[0, 3, 3] == 33.0  # from band 3
        assert out[1, 2, 2] == 11.0  # from band 1
        assert out[0, 0, 0] == 5.0  # outside patch

    # include_imgs_with_overlap: 6x6 tile with overlap=1 -> 4x4 core inserted at cols 4..8, rows 4..8
    ovl_arr = np.full((2, 6, 6), 99.0, dtype=np.float32)
    ovl_arr[:, 0, :] = -50.0  # border should be cropped away
    ovl_arr[:, -1, :] = -50.0
    ovl_arr[:, :, 0] = -50.0
    ovl_arr[:, :, -1] = -50.0
    # Crop origin should be (4, 6) -> tile origin must be (4-1, 6+1) = (3.0, 7.0)
    _write_geo_raster(ovl_tile, ovl_arr, tl_x=3.0, tl_y=7.0, res=1.0)

    rsgispy.imageutils.mosaic.include_imgs_with_overlap(
        input_img=str(base_img),
        include_imgs=[str(ovl_tile)],
        overlap=1,
    )
    with rasterio.open(base_img) as src:
        out = src.read()
        assert np.all(out[:, 4:8, 4:8] == 99.0)
        assert out[0, 3, 4] == 33.0  # border -50 did NOT overwrite row 3


def test_combine_imgs_to_band_and_combine_binary_masks(tmp_path):
    """Test combine_imgs_to_band and combine_binary_masks with JSON LUT."""
    img1 = tmp_path / "comb1.tif"
    img2 = tmp_path / "comb2.tif"
    comb_out = tmp_path / "comb_out.tif"

    a1 = np.zeros((4, 4), dtype=np.uint8)
    a1[0, :] = 1
    a2 = np.zeros((4, 4), dtype=np.uint8)
    a2[:, 0] = 2
    _write_geo_raster(img1, a1, dtype="uint8")
    _write_geo_raster(img2, a2, dtype="uint8")

    rsgispy.imageutils.mosaic.combine_imgs_to_band(
        [str(img1), str(img2)],
        str(comb_out),
        gdalformat="GTIFF",
        datatype=rsgispy.TYPE_8UINT,
        no_data_val=0.0,
    )
    with rasterio.open(comb_out) as src:
        assert src.count == 1
        out = src.read(1)
        assert out[0, 2] == 1
        assert out[2, 0] == 2
        assert out[0, 0] == 2  # img2 overrides img1 at non-nodata overlap
        assert out[2, 2] == 0

    # combine_binary_masks
    msk1 = tmp_path / "msk_water.tif"
    msk2 = tmp_path / "msk_forest.tif"
    packed_out = tmp_path / "packed_msk.tif"
    lut_json = tmp_path / "packed_lut.json"

    m1 = np.zeros((4, 4), dtype=np.uint8)
    m1[0:2, :] = 1
    m2 = np.zeros((4, 4), dtype=np.uint8)
    m2[:, 0:2] = 1
    _write_geo_raster(msk1, m1, dtype="uint8")
    _write_geo_raster(msk2, m2, dtype="uint8")

    rsgispy.imageutils.mosaic.combine_binary_masks(
        {"water": str(msk1), "forest": str(msk2)},
        str(packed_out),
        str(lut_json),
        gdalformat="GTIFF",
    )

    assert rsgispy.imageutils.colourtable.is_img_thematic(str(packed_out))
    with open(lut_json, "r", encoding="utf-8") as fp:
        lut = json.load(fp)
    with rasterio.open(packed_out) as src:
        arr = src.read(1)
        val_both = str(int(arr[0, 0]))
        val_water = str(int(arr[0, 3]))
        val_forest = str(int(arr[3, 0]))
        val_none = str(int(arr[3, 3]))
        assert lut[val_both] == {"water": 1, "forest": 1}
        assert lut[val_water] == {"water": 1, "forest": 0}
        assert lut[val_forest] == {"water": 0, "forest": 1}
        assert lut[val_none] == {"water": 0, "forest": 0}


def test_tiling_functions(tmp_path):
    """Test create_tiles, create_tiles_multi_core, and create_raster_tiles_bbox."""
    src_img = tmp_path / "tile_src.tif"
    arr = np.arange(12 * 10, dtype=np.uint16).reshape(12, 10)
    _write_geo_raster(src_img, arr, tl_x=100.0, tl_y=200.0, res=2.0, dtype="uint16")

    # 1. create_tiles without offset
    tile_files = rsgispy.imageutils.tiling.create_tiles(
        input_img=str(src_img),
        out_img_base=str(tmp_path / "tiles"),
        tile_width=5,
        tile_height=6,
        tile_overlap=1,
        offset_tiles=False,
        gdalformat="GTIFF",
        datatype=rsgispy.TYPE_16UINT,
        out_img_ext="tif",
    )
    assert len(tile_files) == 4
    with rasterio.open(tile_files[0]) as src:
        # First tile has overlap 1 on right/bottom -> width=6, height=7
        assert src.width == 6
        assert src.height == 7
        np.testing.assert_array_equal(src.read(1), arr[:7, :6])

    # 2. create_tiles with offset_tiles=True
    off_files = rsgispy.imageutils.tiling.create_tiles(
        input_img=str(src_img),
        out_img_base=str(tmp_path / "offtiles"),
        tile_width=6,
        tile_height=6,
        tile_overlap=0,
        offset_tiles=True,
        gdalformat="GTIFF",
        datatype=rsgispy.TYPE_16UINT,
        out_img_ext="tif",
    )
    assert len(off_files) == 9  # 3x3 offset grid for 10x12 image with 6x6 tiles

    # 3. create_tiles_multi_core (2 cores)
    mc_base = tmp_path / "mc_tile"
    rsgispy.imageutils.tiling.create_tiles_multi_core(
        input_img=str(src_img),
        out_img_base=str(mc_base),
        width=6,
        height=7,
        gdalformat="GTIFF",
        datatype=rsgispy.TYPE_16UINT,
        out_img_ext="tif",
        n_cores=2,
    )
    t_x1y1 = tmp_path / "mc_tile_x1y1.tif"
    t_x2y2 = tmp_path / "mc_tile_x2y2.tif"
    assert t_x1y1.exists()
    assert t_x2y2.exists()
    with rasterio.open(t_x1y1) as s1, rasterio.open(t_x2y2) as s2:
        assert (s1.width, s1.height) == (6, 7)
        assert (s2.width, s2.height) == (4, 5)
        np.testing.assert_array_equal(s2.read(1), arr[7:, 6:])

    # 4. create_raster_tiles_bbox
    tile_dir = tmp_path / "bbox_tiles"
    tile_dir.mkdir()
    rsgispy.imageutils.tiling.create_raster_tiles_bbox(
        bbox=(500000.0, 500200.0, 5199800.0, 5200000.0),
        tile_x_size=100.0,
        tile_y_size=100.0,
        out_img_res=10.0,
        out_epsg=32630,
        out_tile_dir=str(tile_dir),
        out_tile_name="grid_{}",
        out_img_ext="tif",
        gdalformat="GTIFF",
        datatype=rsgispy.TYPE_8UINT,
        out_img_pxl_val=3.0,
        out_img_n_bands=2,
    )
    created = sorted(p.name for p in tile_dir.glob("*.tif"))
    assert created == ["grid_0_0.tif", "grid_0_1.tif", "grid_1_0.tif", "grid_1_1.tif"]
    with rasterio.open(tile_dir / "grid_0_0.tif") as src:
        assert src.count == 2
        assert (src.width, src.height) == (10, 10)
        assert np.all(src.read() == 3)


def test_colour_interp_thematic_and_colour_table(tmp_path):
    """Test colour interp, thematic toggles, and colour table definition/retrieval."""
    img_path = tmp_path / "thematic_clr.tif"
    arr = np.array([[0, 1], [2, 3]], dtype=np.uint8)
    _write_geo_raster(img_path, arr, dtype="uint8")

    # Thematic toggle
    rsgispy.imageutils.colourtable.set_img_thematic(str(img_path))
    assert rsgispy.imageutils.colourtable.is_img_thematic(str(img_path)) is True
    rsgispy.imageutils.colourtable.set_img_not_thematic(str(img_path))
    assert rsgispy.imageutils.colourtable.is_img_thematic(str(img_path)) is False

    # Colour interp get/set
    rsgispy.imageutils.colourtable.set_img_band_colour_interp(
        str(img_path), 1, int(ColorInterp.red.value)
    )
    assert (
        rsgispy.imageutils.colourtable.get_img_band_colour_interp(str(img_path), 1)
        == int(ColorInterp.red.value)
    )

    # Missing colour table raises RSGISPyException
    with pytest.raises(rsgispy.RSGISPyException):
        rsgispy.imageutils.colourtable.get_colour_tab_info(str(img_path), img_band=1)

    # Define colour table with mixed hex, RGB, and RGBA
    clr_lut = {
        0: (0, 0, 0, 0),
        1: "#ff0000",
        2: [0, 255, 0],
        3: [0, 0, 255, 128],
    }
    rsgispy.imageutils.colourtable.define_colour_table(str(img_path), clr_lut, img_band=1)
    assert (
        rsgispy.imageutils.colourtable.get_img_band_colour_interp(str(img_path), 1)
        == int(ColorInterp.palette.value)
    )

    info = rsgispy.imageutils.colourtable.get_colour_tab_info(str(img_path), img_band=1)
    # GeoTIFF colour table stores RGB (alpha defaults to 255 unless index == nodata)
    assert info[0]["red"] == 0 and info[0]["green"] == 0 and info[0]["blue"] == 0
    assert info[1] == {"red": 255, "green": 0, "blue": 0, "alpha": 255}
    assert info[2] == {"red": 0, "green": 255, "blue": 0, "alpha": 255}
    assert info[3]["red"] == 0 and info[3]["green"] == 0 and info[3]["blue"] == 255

    info_def = rsgispy.imageutils.colourtable.get_colour_tab_info_for_def(
        str(img_path), img_band=1
    )
    assert info_def[1] == [255, 0, 0, 255]
    assert info_def[2] == [0, 255, 0, 255]


def test_masking_functions(tmp_path):
    """Test mask_img, gen_finite_mask, gen_valid_mask, gen_img_edge_mask, mask_all_band_zero_vals, create_valid_mask."""
    in_img = tmp_path / "in_vals.tif"
    msk_img = tmp_path / "in_msk.tif"
    masked_out = tmp_path / "masked.tif"
    finite_out = tmp_path / "finite_msk.tif"
    valid_out = tmp_path / "valid_msk.tif"
    edge_out = tmp_path / "edge_msk.tif"
    zero_out = tmp_path / "zero_vals.tif"
    cvalid_out = tmp_path / "create_valid.tif"

    data = np.array(
        [
            [
                [10.0, 20.0, 0.0, 0.0],
                [5.0, np.nan, 0.0, 15.0],
                [0.0, 0.0, 30.0, 40.0],
                [1.0, 2.0, 3.0, np.inf],
            ],
            [
                [0.0, 10.0, 0.0, 12.0],
                [2.0, 4.0, 0.0, 0.0],
                [0.0, 8.0, 6.0, 0.0],
                [4.0, 5.0, 6.0, 7.0],
            ],
        ],
        dtype=np.float32,
    )
    _write_geo_raster(in_img, data, nodata=0.0)

    msk_arr = np.array(
        [
            [0, 1, 2, 0],
            [1, 0, 2, 3],
            [0, 0, 1, 2],
            [3, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    _write_geo_raster(msk_img, msk_arr, dtype="uint8")

    # 1. mask_img with list of mask values
    rsgispy.imageutils.masking.mask_img(
        str(in_img),
        str(msk_img),
        str(masked_out),
        "GTIFF",
        rsgispy.TYPE_32FLOAT,
        out_value=-999.0,
        mask_value=[1, 2],
    )
    with rasterio.open(masked_out) as src:
        res = src.read()
        assert res[0, 0, 0] == 10.0
        assert res[0, 0, 1] == -999.0
        assert res[1, 0, 2] == -999.0
        assert res[0, 1, 3] == 15.0  # mask == 3, kept

    # 2. gen_finite_mask
    rsgispy.imageutils.masking.gen_finite_mask(
        str(in_img), str(finite_out), gdalformat="GTIFF"
    )
    assert rsgispy.imageutils.colourtable.is_img_thematic(str(finite_out))
    with rasterio.open(finite_out) as src:
        fm = src.read(1)
        assert fm[0, 0] == 1
        assert fm[1, 1] == 0  # nan
        assert fm[3, 3] == 0  # inf

    # 3. gen_valid_mask (pass single path string)
    rsgispy.imageutils.masking.gen_valid_mask(
        str(msk_img), str(valid_out), gdalformat="GTIFF", no_data_val=0.0
    )
    assert rsgispy.imageutils.colourtable.is_img_thematic(str(valid_out))
    with rasterio.open(valid_out) as src:
        vm = src.read(1)
        assert vm[0, 0] == 0
        assert vm[0, 1] == 1

    # 4. gen_img_edge_mask
    rsgispy.imageutils.masking.gen_img_edge_mask(
        str(msk_img), str(edge_out), gdalformat="GTIFF", n_edge_pxls=1
    )
    with rasterio.open(edge_out) as src:
        em = src.read(1)
        expected_edge = np.array(
            [
                [1, 1, 1, 1],
                [1, 0, 0, 1],
                [1, 0, 0, 1],
                [1, 1, 1, 1],
            ],
            dtype=np.uint8,
        )
        np.testing.assert_array_equal(em, expected_edge)

    # 5. mask_all_band_zero_vals
    finite_only = tmp_path / "finite_only.tif"
    f_data = np.array(
        [
            [[10.0, 0.0, 0.0], [5.0, 0.0, 2.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 4.0], [0.0, 0.0, 3.0], [0.0, 1.0, 0.0]],
        ],
        dtype=np.float32,
    )
    _write_geo_raster(finite_only, f_data, dtype="float32")
    rsgispy.imageutils.masking.mask_all_band_zero_vals(
        str(finite_only), str(zero_out), gdalformat="GTIFF", out_val=1.0
    )
    with rasterio.open(zero_out) as src:
        zo = src.read()
        # (0,0) has sum 10 > 0, band 2 is 0 -> replaced with 1.0
        assert zo[0, 0, 0] == 10.0
        assert zo[1, 0, 0] == 1.0
        # (0,1) has both bands 0 -> stays 0
        assert zo[0, 0, 1] == 0.0
        assert zo[1, 0, 1] == 0.0

    # 6. create_valid_mask using ImageBandInfo
    img_b1 = tmp_path / "cvalid_b1.tif"
    img_b2 = tmp_path / "cvalid_b2.tif"
    _write_geo_raster(
        img_b1,
        np.array([[1, -99, 2], [3, 4, 5]], dtype=np.int16),
        dtype="int16",
        nodata=-99.0,
    )
    _write_geo_raster(
        img_b2,
        np.array([[10, 20, 0], [30, 40, 50]], dtype=np.uint8),
        dtype="uint8",
        nodata=None,  # defaults to 0.0 check
    )
    band_info = [
        rsgispy.imageutils.masking.ImageBandInfo(
            file_name=str(img_b1), name="b1", bands=[1]
        ),
        rsgispy.imageutils.masking.ImageBandInfo(
            file_name=str(img_b2), name="b2", bands=[1]
        ),
    ]
    rsgispy.imageutils.masking.create_valid_mask(
        band_info,
        str(cvalid_out),
        gdalformat="GTIFF",
        tmp_dir=str(tmp_path / "tmp"),
    )
    with rasterio.open(cvalid_out) as src:
        cv = src.read(1)
        expected_cv = np.array([[1, 0, 0], [1, 1, 1]], dtype=np.uint8)
        np.testing.assert_array_equal(cv, expected_cv)


def test_subset_functions(tmp_path):
    """Test subset_bbox, subset_to_img, and subset with GeoPackage vector."""
    in_img = tmp_path / "sub_in.tif"
    roi_img = tmp_path / "sub_roi.tif"
    vec_file = tmp_path / "sub_roi.gpkg"
    out_bbox = tmp_path / "out_bbox.tif"
    out_img = tmp_path / "out_roi_img.tif"
    out_vec = tmp_path / "out_vec.tif"

    # 10x10 image at 10m res: X=[500000, 500100], Y=[5199900, 5200000]
    arr = np.arange(100, dtype=np.float32).reshape(10, 10)
    _write_geo_raster(
        in_img,
        arr,
        tl_x=500000.0,
        tl_y=5200000.0,
        res=10.0,
        nodata=-999.0,
        descriptions=["val"],
    )

    # 1. subset_bbox -> cols 2..5 (500020..500050), rows 3..7 (5199930..5199970)
    rsgispy.imageutils.subsetting.subset_bbox(
        str(in_img),
        str(out_bbox),
        "GTIFF",
        rsgispy.TYPE_32FLOAT,
        min_x=500020.0,
        max_x=500050.0,
        min_y=5199930.0,
        max_y=5199970.0,
    )
    with rasterio.open(out_bbox) as src:
        assert (src.width, src.height) == (3, 4)
        assert src.nodata == -999.0
        assert src.descriptions == ("val",)
        np.testing.assert_allclose(src.read(1), arr[3:7, 2:5])

    # 2. subset_to_img
    _write_geo_raster(
        roi_img,
        np.zeros((2, 2), dtype=np.uint8),
        tl_x=500030.0,
        tl_y=5199980.0,
        res=10.0,
        dtype="uint8",
    )
    rsgispy.imageutils.subsetting.subset_to_img(
        str(in_img), str(roi_img), str(out_img), "GTIFF", rsgispy.TYPE_32FLOAT
    )
    with rasterio.open(out_img) as src:
        assert (src.width, src.height) == (2, 2)
        np.testing.assert_allclose(src.read(1), arr[2:4, 3:5])

    # 3. subset by vector layer
    gdf = geopandas.GeoDataFrame(
        {"id": [1]},
        geometry=[box(500010.0, 5199950.0, 500060.0, 5199990.0)],
        crs="EPSG:32630",
    )
    gdf.to_file(vec_file, layer="roi", driver="GPKG")

    rsgispy.imageutils.subsetting.subset(
        str(in_img),
        str(vec_file),
        "roi",
        str(out_vec),
        "GTIFF",
        rsgispy.TYPE_32FLOAT,
    )
    with rasterio.open(out_vec) as src:
        assert (src.width, src.height) == (5, 4)
        np.testing.assert_allclose(src.read(1), arr[1:5, 1:6])
