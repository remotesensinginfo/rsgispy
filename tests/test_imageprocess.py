"""
Tests for rsgispy.imageprocess utilities and block processing.
"""

from contextlib import ExitStack
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

import rsgispy
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


def test_normalise_band_defs(tmp_path):
    """Test normalise_band_defs with objects, dicts, duplicates, and invalid inputs."""
    img = tmp_path / "img.tif"
    _create_test_raster(img, np.ones((2, 4, 4), dtype=np.float32))

    # Deduplication of identical definitions and dict support
    norm = rsgispy.imageprocess.normalise_band_defs(
        [
            rsgispy.imageprocess.BandDefn("b1", str(img), 1),
            rsgispy.imageprocess.BandDefn("b1", str(img), 1),
            {"band_name": "b2", "input_img": str(img), "img_band": 2},
        ]
    )
    assert len(norm) == 2
    assert norm[0] == rsgispy.imageprocess.BandDefn("b1", str(img), 1)
    assert norm[1] == rsgispy.imageprocess.BandDefn("b2", str(img), 2)

    # Duplicate band_name pointing to different band raises exception
    with pytest.raises(rsgispy.RSGISPyException):
        rsgispy.imageprocess.normalise_band_defs(
            [
                rsgispy.imageprocess.BandDefn("b1", str(img), 1),
                rsgispy.imageprocess.BandDefn("b1", str(img), 2),
            ]
        )

    # Invalid band_name, input_img, and img_band
    with pytest.raises(rsgispy.RSGISPyException):
        rsgispy.imageprocess.normalise_band_defs(
            [rsgispy.imageprocess.BandDefn("", str(img), 1)]
        )
    with pytest.raises(rsgispy.RSGISPyException):
        rsgispy.imageprocess.normalise_band_defs(
            [rsgispy.imageprocess.BandDefn("b1", None, 1)]
        )
    with pytest.raises(rsgispy.RSGISPyException):
        rsgispy.imageprocess.normalise_band_defs(
            [rsgispy.imageprocess.BandDefn("b1", str(img), 0)]
        )


def test_validate_crs_and_res_and_intersection(tmp_path):
    """Test validate_crs_and_res and compute_spatial_intersection edge cases."""
    img1 = tmp_path / "img1.tif"
    img2 = tmp_path / "img2.tif"
    img_diff_crs = tmp_path / "img_diff_crs.tif"

    _create_test_raster(img1, np.ones((1, 10, 10)), origin=(0.0, 10.0), crs="EPSG:4326")
    _create_test_raster(img2, np.ones((1, 10, 10)), origin=(3.0, 7.0), crs="EPSG:4326")
    _create_test_raster(
        img_diff_crs, np.ones((1, 10, 10)), origin=(0.0, 10.0), crs="EPSG:3857"
    )

    with pytest.raises(rsgispy.RSGISPyException):
        rsgispy.imageprocess.validate_crs_and_res({})
    with pytest.raises(rsgispy.RSGISPyException):
        rsgispy.imageprocess.compute_spatial_intersection({})

    with ExitStack() as stack:
        ds1 = stack.enter_context(rasterio.open(img1))
        ds2 = stack.enter_context(rasterio.open(img2))
        ds_crs = stack.enter_context(rasterio.open(img_diff_crs))

        rsgispy.imageprocess.validate_crs_and_res({str(img1): ds1, str(img2): ds2})
        grid = rsgispy.imageprocess.compute_spatial_intersection(
            {str(img1): ds1, str(img2): ds2}
        )
        assert grid.width == 7
        assert grid.height == 7
        assert grid.offsets[str(img1)] == (3, 3)
        assert grid.offsets[str(img2)] == (0, 0)

        with pytest.raises(rsgispy.RSGISPyException):
            rsgispy.imageprocess.validate_crs_and_res(
                {str(img1): ds1, str(img_diff_crs): ds_crs}
            )


def test_cast_result_block():
    """Test cast_result_block with scalar, 2D/3D arrays, and NaN/Inf integer conversion."""
    scalar_block = rsgispy.imageprocess.cast_result_block(
        5.0, win_h=3, win_w=4, out_np_dtype=np.dtype(np.float32)
    )
    assert scalar_block.shape == (3, 4)
    assert scalar_block.dtype == np.float32
    assert np.all(scalar_block == 5.0)

    arr_with_nan = np.array([[np.nan, np.inf], [-np.inf, 7.0]])
    int_block = rsgispy.imageprocess.cast_result_block(
        arr_with_nan, win_h=2, win_w=2, out_np_dtype=np.dtype(np.uint8)
    )
    assert int_block.dtype == np.uint8
    np.testing.assert_array_equal(int_block, np.array([[0, 0], [0, 7]], dtype=np.uint8))


def test_process_blocks_single_and_multiband(tmp_path):
    """Test process_blocks with custom single-band and multi-band callables."""
    in_img = tmp_path / "in_custom.tif"
    out_single = tmp_path / "out_single.tif"
    out_multi = tmp_path / "out_multi.tif"

    b1_arr = np.arange(1, 17, dtype=np.float32).reshape(4, 4)
    b2_arr = np.full((4, 4), 2.0, dtype=np.float32)
    _create_test_raster(in_img, np.stack([b1_arr, b2_arr]))

    norm_defs = rsgispy.imageprocess.normalise_band_defs(
        [
            {"band_name": "nir", "input_img": str(in_img), "img_band": 1},
            {"band_name": "red", "input_img": str(in_img), "img_band": 2},
        ]
    )

    def custom_ndvi(block_vars):
        return (block_vars["nir"] - block_vars["red"]) / (
            block_vars["nir"] + block_vars["red"]
        )

    def custom_two_band(block_vars):
        return np.stack(
            [
                block_vars["nir"] + block_vars["red"],
                block_vars["nir"] - block_vars["red"],
            ],
            axis=0,
        )

    with ExitStack() as stack:
        datasets = rsgispy.imageprocess.open_input_datasets(stack, norm_defs)
        grid = rsgispy.imageprocess.compute_spatial_intersection(datasets)

        dst_single = rsgispy.imageprocess.open_output_dataset(
            stack=stack,
            output_img=str(out_single),
            gdalformat="GTIFF",
            out_rio_dtype="float32",
            grid=grid,
            count=1,
        )
        rsgispy.imageprocess.process_blocks(
            dst=dst_single,
            datasets=datasets,
            norm_defs=norm_defs,
            grid=grid,
            block_func=custom_ndvi,
            out_np_dtype=np.dtype(np.float32),
            block_size=2,
        )

        dst_multi = rsgispy.imageprocess.open_output_dataset(
            stack=stack,
            output_img=str(out_multi),
            gdalformat="GTIFF",
            out_rio_dtype="float32",
            grid=grid,
            count=2,
        )
        rsgispy.imageprocess.process_blocks(
            dst=dst_multi,
            datasets=datasets,
            norm_defs=norm_defs,
            grid=grid,
            block_func=custom_two_band,
            out_np_dtype=np.dtype(np.float32),
            block_size=2,
        )

    with rasterio.open(out_single) as src1:
        np.testing.assert_allclose(
            src1.read(1), (b1_arr - b2_arr) / (b1_arr + b2_arr), rtol=1e-6
        )

    with rasterio.open(out_multi) as src2:
        assert src2.count == 2
        np.testing.assert_allclose(src2.read(1), b1_arr + b2_arr)
        np.testing.assert_allclose(src2.read(2), b1_arr - b2_arr)


def test_open_output_dataset_existing_mismatch(tmp_path):
    """Test open_output_dataset error when output_exists=True has mismatched dimensions."""
    in_img = tmp_path / "in.tif"
    out_img = tmp_path / "out_wrong_size.tif"

    _create_test_raster(in_img, np.ones((1, 6, 6), dtype=np.float32))
    _create_test_raster(out_img, np.ones((1, 4, 4), dtype=np.float32))

    norm_defs = rsgispy.imageprocess.normalise_band_defs(
        [rsgispy.imageprocess.BandDefn("b1", str(in_img), 1)]
    )

    with ExitStack() as stack:
        datasets = rsgispy.imageprocess.open_input_datasets(stack, norm_defs)
        grid = rsgispy.imageprocess.compute_spatial_intersection(datasets)
        with pytest.raises(rsgispy.RSGISPyException):
            rsgispy.imageprocess.open_output_dataset(
                stack=stack,
                output_img=str(out_img),
                gdalformat="GTIFF",
                out_rio_dtype="float32",
                grid=grid,
                output_exists=True,
            )


def test_runtime_default_block_size_override(tmp_path, monkeypatch):
    """Test overriding and resetting DEFAULT_BLOCK_SIZE at runtime via env vars and helpers."""
    monkeypatch.delenv("RSGISPY_BLOCK_SIZE", raising=False)
    monkeypatch.delenv("RSGISLIB_BLOCK_SIZE", raising=False)

    assert (
        rsgispy.imageprocess.get_default_block_size()
        == rsgispy.imageprocess.DEFAULT_BLOCK_SIZE
    )

    in_img = tmp_path / "block_test.tif"
    _create_test_raster(in_img, np.ones((1, 10, 10), dtype=np.float32))

    # 1. Override via set_default_block_size (sets RSGISPY_BLOCK_SIZE env var)
    rsgispy.imageprocess.set_default_block_size(4)
    assert rsgispy.imageprocess.get_default_block_size() == 4
    blocks = list(
        rsgispy.imageprocess.iter_single_image_band_blocks(
            str(in_img), use_pbar=False
        )
    )
    # 10x10 image with block size 4 -> 3x3 = 9 blocks, first block is 4x4
    assert len(blocks) == 9
    assert blocks[0][0].width == 4
    assert blocks[0][0].height == 4

    # 2. Reset removes env var override and restores DEFAULT_BLOCK_SIZE (512)
    rsgispy.imageprocess.reset_default_block_size()
    assert rsgispy.imageprocess.get_default_block_size() == 512
    blocks_default = list(
        rsgispy.imageprocess.iter_single_image_band_blocks(
            str(in_img), use_pbar=False
        )
    )
    assert len(blocks_default) == 1

    # 3. Override directly via RSGISLIB_BLOCK_SIZE env var
    monkeypatch.setenv("RSGISLIB_BLOCK_SIZE", "5")
    assert rsgispy.imageprocess.get_default_block_size() == 5
    blocks_rsgislib = list(
        rsgispy.imageprocess.iter_single_image_band_blocks(
            str(in_img), use_pbar=False
        )
    )
    assert len(blocks_rsgislib) == 4  # 2x2 grid of 5x5 blocks

    # 4. RSGISPY_BLOCK_SIZE takes precedence over RSGISLIB_BLOCK_SIZE
    monkeypatch.setenv("RSGISPY_BLOCK_SIZE", "3")
    assert rsgispy.imageprocess.get_default_block_size() == 3

    # 5. Direct mutation of DEFAULT_BLOCK_SIZE works when env vars are unset
    rsgispy.imageprocess.reset_default_block_size()
    monkeypatch.setattr(rsgispy.imageprocess, "DEFAULT_BLOCK_SIZE", 5)
    assert rsgispy.imageprocess.get_default_block_size() == 5
    blocks_mutated = list(
        rsgispy.imageprocess.iter_single_image_band_blocks(
            str(in_img), use_pbar=False
        )
    )
    assert len(blocks_mutated) == 4

    # 6. Validation errors for invalid block sizes
    for bad_val in (0, -4, True, 3.5):
        with pytest.raises(rsgispy.RSGISPyException):
            rsgispy.imageprocess.set_default_block_size(bad_val)

    monkeypatch.setenv("RSGISPY_BLOCK_SIZE", "not_an_int")
    with pytest.raises(rsgispy.RSGISPyException):
        rsgispy.imageprocess.get_default_block_size()

    monkeypatch.setenv("RSGISPY_BLOCK_SIZE", "0")
    with pytest.raises(rsgispy.RSGISPyException):
        rsgispy.imageprocess.get_default_block_size()


def test_open_multi_image_datasets(tmp_path):
    """Test open_multi_image_datasets with unique/duplicate paths and errors."""
    img1 = tmp_path / "img1.tif"
    img2 = tmp_path / "img2.tif"
    _create_test_raster(img1, np.ones((1, 4, 4), dtype=np.float32))
    _create_test_raster(img2, np.full((2, 4, 4), 2.0, dtype=np.float32))

    with ExitStack() as stack:
        datasets, paths = rsgispy.imageprocess.open_multi_image_datasets(
            stack, [str(img1), str(img2), str(img1)]
        )
        assert len(datasets) == 2
        assert paths == [str(img1), str(img2), str(img1)]
        assert datasets[str(img1)].count == 1
        assert datasets[str(img2)].count == 2

    with ExitStack() as stack:
        with pytest.raises(rsgispy.RSGISPyException):
            rsgispy.imageprocess.open_multi_image_datasets(stack, [])
        with pytest.raises(rsgispy.RSGISPyException):
            rsgispy.imageprocess.open_multi_image_datasets(
                stack, [str(tmp_path / "missing.tif")]
            )


def test_dataset_and_validation_helpers(tmp_path):
    """Test IntersectionGrid.from_dataset, bbox, exists, and band helpers."""
    img = tmp_path / "sample.tif"
    _create_test_raster(
        img, np.ones((2, 4, 5), dtype=np.float32), origin=(10.0, 50.0), pixel_size=(2.0, 2.0)
    )
    rsgispy.imageprocess.ensure_file_exists(str(img))
    with pytest.raises(rsgispy.RSGISPyException):
        rsgispy.imageprocess.ensure_file_exists(str(tmp_path / "missing.tif"))

    with rasterio.open(img) as src:
        rsgispy.imageprocess.validate_band_index(1, src.count, str(img))
        rsgispy.imageprocess.validate_band_index(2, src.count, str(img))
        with pytest.raises(rsgispy.RSGISPyException):
            rsgispy.imageprocess.validate_band_index(0, src.count, str(img))
        with pytest.raises(rsgispy.RSGISPyException):
            rsgispy.imageprocess.validate_band_index(3, src.count, str(img))

        assert rsgispy.imageprocess.bbox_from_dataset(src) == pytest.approx(
            (10.0, 20.0, 42.0, 50.0)
        )
        grid = rsgispy.imageprocess.IntersectionGrid.from_dataset(src)
        assert grid.width == 5
        assert grid.height == 4
        assert grid.transform == src.transform
        assert grid.crs == src.crs
        assert not grid.offsets


def test_iter_dataset_windows():
    """Test iter_dataset_windows window tiling and validation."""
    wins = list(
        rsgispy.imageprocess.iter_dataset_windows(
            height=5, width=7, block_size=3, use_pbar=False
        )
    )
    assert [(w.col_off, w.row_off, w.width, w.height) for w in wins] == [
        (0, 0, 3, 3),
        (3, 0, 3, 3),
        (6, 0, 1, 3),
        (0, 3, 3, 2),
        (3, 3, 3, 2),
        (6, 3, 1, 2),
    ]

    with pytest.raises(rsgispy.RSGISPyException):
        list(rsgispy.imageprocess.iter_dataset_windows(5, 5, block_size=0))


def test_create_output_from_dataset(tmp_path):
    """Test create_output_from_dataset matching source grid/CRS and custom band count."""
    in_img = tmp_path / "ref.tif"
    out_same = tmp_path / "out_same.tif"
    out_single = tmp_path / "out_single.tif"
    _create_test_raster(
        in_img,
        np.ones((3, 5, 6), dtype=np.float32),
        origin=(100.0, 200.0),
        pixel_size=(5.0, 5.0),
        crs="EPSG:32630",
    )

    with ExitStack() as stack:
        src = stack.enter_context(rasterio.open(in_img, "r"))
        dst_same = rsgispy.imageprocess.create_output_from_dataset(
            stack=stack,
            src=src,
            output_img=str(out_same),
            gdalformat="GTIFF",
            out_rio_dtype="float32",
        )
        dst_same.write(np.full((3, 5, 6), 2.0, dtype=np.float32))

        dst_single = rsgispy.imageprocess.create_output_from_dataset(
            stack=stack,
            src=src,
            output_img=str(out_single),
            gdalformat="GTIFF",
            out_rio_dtype="uint8",
            count=1,
        )
        dst_single.write(np.full((5, 6), 7, dtype=np.uint8), 1)

    with rasterio.open(out_same) as s_out, rasterio.open(in_img) as s_in:
        assert s_out.count == 3
        assert s_out.width == s_in.width and s_out.height == s_in.height
        assert s_out.transform == s_in.transform
        assert s_out.crs == s_in.crs
        assert np.all(s_out.read() == 2.0)

    with rasterio.open(out_single) as o_out:
        assert o_out.count == 1
        assert o_out.dtypes == ("uint8",)
        assert np.all(o_out.read(1) == 7)
