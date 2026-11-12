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


def test_create_blank_and_copy_and_bbox_and_buf_imgs(tmp_path, monkeypatch):
    """Test create_blank_img, create_copy_img, create_blank_buf_img_from_ref_img, and create_blank_img_from_bbox."""
    monkeypatch.delenv("RSGISLIB_IMG_CRT_OPTS_GTIFF", raising=False)
    monkeypatch.delenv("RSGISPY_IMG_CRT_OPTS_GTIFF", raising=False)
    rsgispy.imageutils.set_env_vars_deflate_gtiff_outs()

    blank_img = tmp_path / "blank.tif"
    copy_img = tmp_path / "copy.tif"
    buf_pxl_img = tmp_path / "buf_pxl.tif"
    buf_spt_img = tmp_path / "buf_spt.tif"
    bbox_img = tmp_path / "bbox.tif"

    # 1. create_blank_img using epsg_code
    rsgispy.imageutils.create_blank_img(
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
    rsgispy.imageutils.create_copy_img(
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
    rsgispy.imageutils.create_blank_buf_img_from_ref_img(
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

    rsgispy.imageutils.create_blank_buf_img_from_ref_img(
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
        rsgispy.imageutils.create_blank_buf_img_from_ref_img(
            str(blank_img), str(buf_pxl_img), "GTIFF", rsgispy.TYPE_32FLOAT
        )

    # 4. create_blank_img_from_bbox
    rsgispy.imageutils.create_blank_img_from_bbox(
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
    rsgispy.imageutils.set_env_vars_deflate_gtiff_outs()

    ref_img = tmp_path / "ref.tif"
    proc_img = tmp_path / "proc.tif"
    resamp_img = tmp_path / "resampled.tif"
    reproj_img = tmp_path / "reprojected.tif"
    reproj_default_opts_img = tmp_path / "reprojected_deflate.tif"

    # Reference image at 10m resolution (20x20)
    rsgispy.imageutils.create_blank_img(
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
    rsgispy.imageutils.create_blank_img(
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
    rsgispy.imageutils.resample_img_to_match(
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
    rsgispy.imageutils.reproject_image(
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
    rsgispy.imageutils.reproject_image(
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
    rsgispy.imageutils.create_stack_images_vrt(
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
    rsgispy.imageutils.create_mosaic_images_vrt(
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
    rsgispy.imageutils.create_mosaic_images_vrt(
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
    rsgispy.imageutils.create_vrt_band_subset(
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
        rsgispy.imageutils.create_vrt_band_subset(
            input_img=str(img1),
            img_bands=[3],
            out_vrt_img=str(subset_vrt),
        )

