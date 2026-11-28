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
    nodata=None,
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
        nodata=nodata,
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

    # 2. Reset removes env var override and restores DEFAULT_BLOCK_SIZE (256)
    rsgispy.imageprocess.reset_default_block_size()
    assert rsgispy.imageprocess.get_default_block_size() == 256
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


def test_iter_dataset_windows_block_width_and_pbar(monkeypatch):
    """Test rectangular block_width windows and use_pbar toggling."""
    wins = list(
        rsgispy.imageprocess.iter_dataset_windows(
            height=5, width=7, block_size=2, block_width=7, use_pbar=False
        )
    )
    assert [(w.col_off, w.row_off, w.width, w.height) for w in wins] == [
        (0, 0, 7, 2),
        (0, 2, 7, 2),
        (0, 4, 7, 1),
    ]

    with pytest.raises(rsgispy.RSGISPyException):
        list(
            rsgispy.imageprocess.iter_dataset_windows(
                5, 5, block_size=2, block_width=0, use_pbar=False
            )
        )

    calls = []

    class _RecorderPbar:
        def __init__(self, *args, **kwargs):
            calls.append(kwargs.get("disable"))
            self.updates = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def update(self, n=1):
            self.updates += n

    monkeypatch.setattr(rsgispy, "get_tqdm", lambda: _RecorderPbar)
    list(rsgispy.imageprocess.iter_dataset_windows(4, 4, block_size=2))
    list(rsgispy.imageprocess.iter_dataset_windows(4, 4, block_size=2, use_pbar=False))
    assert calls == [False, True]


def test_read_window_with_halo_and_cast_clamp(tmp_path):
    """Test read_window_with_halo edge padding and cast_result_block clamp_int."""
    in_img = tmp_path / "halo.tif"
    data = np.arange(1, 25, dtype=np.float32).reshape((2, 3, 4))
    _create_test_raster(in_img, data)

    with rasterio.open(in_img, "r") as src:
        blk_2d = rsgispy.imageprocess.read_window_with_halo(
            src,
            1,
            rasterio.windows.Window(0, 0, 2, 2),
            pad=1,
            out_dtype=np.float32,
        )
        assert blk_2d.shape == (4, 4)
        # Top-left corner repeats pixel (0, 0) == 1.0
        assert blk_2d[0, 0] == pytest.approx(1.0)
        assert blk_2d[1, 1] == pytest.approx(1.0)
        assert blk_2d[2, 2] == pytest.approx(6.0)

        blk_3d = rsgispy.imageprocess.read_window_with_halo(
            src,
            None,
            rasterio.windows.Window(2, 1, 2, 2),
            pad=1,
            out_dtype=np.float32,
        )
        assert blk_3d.shape == (2, 4, 4)
        # Bottom-right corner repeats last pixel of each band (12.0, 24.0)
        assert blk_3d[0, 3, 3] == pytest.approx(12.0)
        assert blk_3d[1, 3, 3] == pytest.approx(24.0)

    raw = np.array([[-5.0, 128.0], [300.0, np.nan]], dtype=np.float64)
    clamped = rsgispy.imageprocess.cast_result_block(
        raw, 2, 2, np.dtype(np.uint8), clamp_int=True
    )
    np.testing.assert_array_equal(
        clamped, np.array([[0, 128], [255, 0]], dtype=np.uint8)
    )


def test_validate_odd_window_size_and_halo_runner(tmp_path):
    """Test odd window validation, halo padding, and run_window_halo_blocks."""
    assert rsgispy.imageprocess.validate_odd_window_size(3) == 3
    assert rsgispy.imageprocess.validate_odd_window_size(5, min_size=5, name="op") == 5
    for bad in (1, 2, 4, -3, True, 3.5):
        with pytest.raises(rsgispy.RSGISPyException):
            rsgispy.imageprocess.validate_odd_window_size(bad)

    in_img = tmp_path / "halo_runner.tif"
    data = np.arange(1, 25, dtype=np.float32).reshape((2, 3, 4))
    _create_test_raster(in_img, data)

    with rasterio.open(in_img, "r") as src:
        blk_const = rsgispy.imageprocess.read_window_with_halo(
            src,
            1,
            rasterio.windows.Window(0, 0, 2, 2),
            pad=(2, 1),
            out_dtype=np.float32,
            pad_mode="constant",
            fill_value=-1.0,
        )
        assert blk_const.shape == (6, 4)
        assert blk_const[0, 0] == pytest.approx(-1.0)
        assert blk_const[2, 1] == pytest.approx(1.0)

    out_per_band = tmp_path / "out_halo_band.tif"
    rsgispy.imageprocess.run_window_halo_blocks(
        str(in_img),
        str(out_per_band),
        win_shape=(3, 3),
        block_fn=lambda h: h[1:-1, 1:-1] * 2.0,
        per_band=True,
        pad_mode="constant",
        fill_value=0.0,
        full_width_strips=True,
        use_pbar=False,
    )
    with rasterio.open(out_per_band) as dst_b:
        np.testing.assert_allclose(dst_b.read(), data * 2.0)

    out_cube = tmp_path / "out_halo_cube.tif"
    rsgispy.imageprocess.run_window_halo_blocks(
        str(in_img),
        str(out_cube),
        win_shape=(3, 5),
        block_fn=lambda h: h[:, 1:-1, 2:-2] + 1.0,
        per_band=False,
        use_pbar=False,
    )
    with rasterio.open(out_cube) as dst_c:
        np.testing.assert_allclose(dst_c.read(), data + 1.0)


def test_overlap_and_band_pair_helpers(tmp_path):
    """Test resolve_band_pair, grids_compatible, compute_overlap_bbox, extract_overlap_arrays."""
    a_p = tmp_path / "a.tif"
    b_p = tmp_path / "b.tif"
    c_p = tmp_path / "c_nonoverlap.tif"
    d_p = tmp_path / "d_coarse.tif"

    a_data = np.arange(1, 51, dtype=np.float32).reshape(2, 5, 5)
    b_data = np.arange(101, 151, dtype=np.float32).reshape(2, 5, 5)
    _create_test_raster(
        a_p, a_data, origin=(0.0, 50.0), pixel_size=(10.0, 10.0), crs="EPSG:32630"
    )
    _create_test_raster(
        b_p, b_data, origin=(20.0, 30.0), pixel_size=(10.0, 10.0), crs="EPSG:32630"
    )
    _create_test_raster(
        c_p, a_data, origin=(500.0, 50.0), pixel_size=(10.0, 10.0), crs="EPSG:32630"
    )
    _create_test_raster(
        d_p,
        np.ones((1, 4, 4), dtype=np.float32),
        origin=(0.0, 50.0),
        pixel_size=(12.5, 12.5),
        crs="EPSG:32630",
    )

    with ExitStack() as stack:
        sa = stack.enter_context(rasterio.open(a_p))
        sb = stack.enter_context(rasterio.open(b_p))
        sc = stack.enter_context(rasterio.open(c_p))
        sd = stack.enter_context(rasterio.open(d_p))

        assert rsgispy.imageprocess.resolve_band_pair(
            sa, sb, None, None, str(a_p), str(b_p)
        ) == ([1, 2], [1, 2])
        assert rsgispy.imageprocess.resolve_band_pair(
            sa, sb, [2], [1], str(a_p), str(b_p)
        ) == ([2], [1])
        with pytest.raises(rsgispy.RSGISPyException):
            rsgispy.imageprocess.resolve_band_pair(
                sa, sb, [1, 2], [1], str(a_p), str(b_p)
            )
        with pytest.raises(rsgispy.RSGISPyException):
            rsgispy.imageprocess.resolve_band_pair(
                sa, sb, [5], [1], str(a_p), str(b_p)
            )

        assert rsgispy.imageprocess.grids_compatible(sa, sb) is True
        assert rsgispy.imageprocess.grids_compatible(sa, sd) is False

        bbox = rsgispy.imageprocess.compute_overlap_bbox(sa, sb, str(a_p), str(b_p))
        assert bbox == pytest.approx((20.0, 0.0, 50.0, 30.0))
        with pytest.raises(rsgispy.RSGISPyException):
            rsgispy.imageprocess.compute_overlap_bbox(sa, sc, str(a_p), str(c_p))

        ov_same = rsgispy.imageprocess.extract_overlap_arrays(
            sa, sb, [1], [1], str(a_p), str(b_p)
        )
        assert ov_same["same_grid"] is True
        assert (ov_same["height"], ov_same["width"]) == (3, 3)
        np.testing.assert_allclose(ov_same["ref_arr"][0], a_data[0, 2:5, 2:5])
        np.testing.assert_allclose(ov_same["flt_arr"][0], b_data[0, 0:3, 0:3])

        ov_reproj = rsgispy.imageprocess.extract_overlap_arrays(
            sa, sd, [1], [1], str(a_p), str(d_p)
        )
        assert ov_reproj["same_grid"] is False
        assert ov_reproj["ref_arr"].shape == (1, 5, 5)
        assert np.all(ov_reproj["flt_valid"])


def test_build_valid_mask_and_read_raster_band(tmp_path):
    """Test build_valid_mask and read_raster_band across integer and float rasters."""
    int_arr = np.array([[0, 1, 2], [2, 0, 3]], dtype=np.uint8)
    mask_int = rsgispy.imageprocess.build_valid_mask(int_arr, no_data_val=0)
    assert mask_int.tolist() == [[False, True, True], [True, False, True]]

    flt_arr = np.array([[np.nan, -9999.0, 1.5], [2.0, np.inf, 0.0]], dtype=np.float32)
    mask_flt = rsgispy.imageprocess.build_valid_mask(flt_arr, no_data_val=-9999.0)
    assert mask_flt.tolist() == [[False, False, True], [True, False, True]]

    raster_path = tmp_path / "band_read.tif"
    _create_test_raster(
        raster_path,
        np.stack([int_arr.astype(np.float32), flt_arr], axis=0),
        pixel_size=(10.0, 10.0),
        nodata=-9999.0,
    )
    band_data, crs, transform, valid = rsgispy.imageprocess.read_raster_band(
        raster_path, img_band=2
    )
    assert band_data.shape == (2, 3)
    assert crs is not None
    assert transform.a == pytest.approx(10.0)
    assert valid.tolist() == [[False, False, True], [True, False, True]]

    with pytest.raises(rsgispy.RSGISPyException):
        rsgispy.imageprocess.read_raster_band(raster_path, img_band=3)
    with pytest.raises(rsgispy.RSGISPyException):
        rsgispy.imageprocess.read_raster_band(tmp_path / "missing.tif", img_band=1)


def test_extract_roi_feature_matrix():
    """Test extract_roi_feature_matrix filtering NaN, nodata, and empty ROIs."""
    b1 = np.array([[1.0, np.nan], [-9999.0, 4.0]], dtype=np.float32)
    b2 = np.array([[10, 20], [30, 40]], dtype=np.int16)
    roi = np.array([0, 1, 2, 3], dtype=np.int64)

    feats, valid_idx = rsgispy.imageprocess.extract_roi_feature_matrix(
        [(b1, np.float32(np.nan)), (b2, 40)], roi
    )
    assert valid_idx.tolist() == [0, 2]
    np.testing.assert_allclose(feats, np.array([[1.0, 10.0], [-9999.0, 30.0]]))

    feats_nd, valid_idx_nd = rsgispy.imageprocess.extract_roi_feature_matrix(
        [(b1, -9999.0), (b2, None)], roi
    )
    assert valid_idx_nd.tolist() == [0, 3]
    np.testing.assert_allclose(feats_nd, np.array([[1.0, 10.0], [4.0, 40.0]]))

    empty_feats, empty_idx = rsgispy.imageprocess.extract_roi_feature_matrix(
        [(b1, None), (b2, None)], np.array([], dtype=np.int64)
    )
    assert empty_feats.shape == (0, 2)
    assert empty_idx.size == 0


# --------------------------------------------------------------------------
# H2: integer outputs are rounded and clamped, never wrapped or truncated
# --------------------------------------------------------------------------

_H2_VALS = np.array([[[200.4, 400.0, -2.0, 99.6], [7.5, 255.6, -300.0, 12.49]]])


def _h2_cases():
    # pylint: disable=import-outside-toplevel
    import rsgispy.changedetect  # noqa: F401
    from rsgispy.imagecalc import comparison, maths, rescale, stats
    from rsgispy.imagecalibration import radiometry
    from rsgispy.imagespectral import indices
    from rsgispy.imageutils import bands, mosaic

    def _rescale(inp, out, dt):
        objs = [rescale.ImageBandRescale(1, 0.0, 1.0, -1000.0, 0.0, 1.0, 77.0)]
        rescale.rescale_img_pxl_vals(inp, out, objs, False, datatype=dt, use_pbar=False)

    return {
        "band_maths": lambda inp, zero, out, dt: maths.band_maths(
            out, "b1", [maths.BandDefn("b1", inp, 1)], datatype=dt, use_pbar=False
        ),
        "image_math": lambda inp, zero, out, dt: maths.image_math(
            inp, out, "b1", datatype=dt, use_pbar=False
        ),
        "rescale_img_pxl_vals": lambda inp, zero, out, dt: _rescale(inp, out, dt),
        "calc_img_rescale": lambda inp, zero, out, dt: rescale.calc_img_rescale(
            inp, out, -1000.0, 0.0, 1.0, 77.0, 0.0, 1.0, datatype=dt, use_pbar=False
        ),
        "image_pixel_column_summary": lambda inp, zero, out, dt: (
            stats.image_pixel_column_summary(
                inp, out, stats.StatsSummary(calc_mean=True), False,
                datatype=dt, use_pbar=False,
            )
        ),
        "calc_img_difference": lambda inp, zero, out, dt: (
            comparison.calc_img_difference(inp, zero, out, datatype=dt, use_pbar=False)
        ),
        "calc_area_under_curve": lambda inp, zero, out, dt: (
            indices.calc_area_under_curve(
                inp, out, [1.0], method="rectangle", datatype=dt, use_pbar=False
            )
        ),
        "stack_img_bands": lambda inp, zero, out, dt: bands.stack_img_bands(
            [inp], out, datatype=dt, calc_stats=False, use_pbar=False
        ),
        "create_img_mosaic": lambda inp, zero, out, dt: mosaic.create_img_mosaic(
            [inp], out, 0.0, -1000.0, datatype=dt, calc_stats=False, use_pbar=False
        ),
        "radiance_to_toa_refl": lambda inp, zero, out, dt: (
            radiometry.radiance_to_toa_refl(
                inp, out, 20.0, 2020, 6, 21, 30.0, [100.0],
                datatype=dt, calc_stats=False, use_pbar=False,
            )
        ),
    }


@pytest.mark.parametrize("case", sorted(_h2_cases()))
def test_integer_outputs_round_and_clamp(tmp_path, case):
    """Each caller's uint8 output equals clip(rint(float64 output), 0, 255)."""
    in_path = str(tmp_path / "in.tif")
    zero_path = str(tmp_path / "zero.tif")
    _create_test_raster(in_path, _H2_VALS, dtype="float64")
    _create_test_raster(zero_path, np.zeros_like(_H2_VALS), dtype="float64")
    func = _h2_cases()[case]
    ref_path = str(tmp_path / "ref.tif")
    int_path = str(tmp_path / "int.tif")
    func(in_path, zero_path, ref_path, rsgispy.TYPE_64FLOAT)
    func(in_path, zero_path, int_path, rsgispy.TYPE_8UINT)
    with rasterio.open(ref_path) as ds:
        ref = ds.read(1)
    with rasterio.open(int_path) as ds:
        got = ds.read(1)
    assert got.dtype == np.uint8
    expected = np.clip(np.rint(ref), 0, 255)
    assert (ref > 255).any() and (ref < 0).any(), "case must cover both limits"
    np.testing.assert_array_equal(got, expected)


def test_round_clamp_to_dtype_policy():
    """Round, clamp, map non-finite to nodata, and keep valid values off nodata."""
    arr = np.array([200.4, 400.0, -2.0, np.nan, 0.2, 254.7])
    np.testing.assert_array_equal(
        rsgispy.imageprocess.round_clamp_to_dtype(arr, np.uint8),
        [200, 255, 0, 0, 0, 255],
    )
    valid = np.array([True, True, True, True, True, False])
    np.testing.assert_array_equal(
        rsgispy.imageprocess.round_clamp_to_dtype(arr, np.uint8, 0, valid),
        [200, 255, 1, 0, 1, 0],
    )
    np.testing.assert_array_equal(
        rsgispy.imageprocess.round_clamp_to_dtype(arr, np.uint8, 255, valid),
        [200, 254, 0, 255, 0, 255],
    )
    with pytest.warns(DeprecationWarning):
        rsgispy.imageprocess.cast_result_block(arr, 1, 6, np.uint8, clamp_int=False)


def test_compute_spatial_intersection_rejects_subpixel_misalignment(tmp_path):
    """H9: grids offset by a fraction of a pixel raise, pointing to resampling."""
    a_path = str(tmp_path / "a.tif")
    b_path = str(tmp_path / "b.tif")
    c_path = str(tmp_path / "c.tif")
    data = np.ones((1, 6, 6), dtype=np.float32)
    _create_test_raster(a_path, data, origin=(0.0, 100.0))
    _create_test_raster(b_path, data, origin=(0.4, 100.0))  # 0.4 pixel shift
    _create_test_raster(c_path, data, origin=(2.0, 99.0))  # whole pixels: fine
    with rasterio.open(a_path) as ds_a, rasterio.open(b_path) as ds_b:
        with pytest.raises(rsgispy.RSGISPyException, match="resample_img_to_match"):
            rsgispy.imageprocess.compute_spatial_intersection(
                {a_path: ds_a, b_path: ds_b}
            )
    with rasterio.open(a_path) as ds_a, rasterio.open(c_path) as ds_c:
        grid = rsgispy.imageprocess.compute_spatial_intersection(
            {a_path: ds_a, c_path: ds_c}
        )
        assert (grid.width, grid.height) == (4, 5)
        assert grid.offsets[a_path] == (2, 1)
