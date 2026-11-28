"""
Tests for :mod:`rsgispy.imagecalc.fill`: KNN region filling and region
growing fill (global and tiled).
"""


import numpy as np
import pytest
import rasterio

import rsgispy
import rsgispy.imagecalc.fill as ic_fill

from conftest import write_raster


def _write_raster(
    path,
    data,
    origin=(0.0, 100.0),
    pixel_size=(10.0, 10.0),
    crs="EPSG:32630",
    dtype="float32",
    nodata=None,
):
    """Write a 3D ``(bands, height, width)`` NumPy array to a GeoTIFF."""
    return write_raster(
        path,
        data,
        dtype=dtype,
        nodata=nodata,
        crs=crs,
        origin=origin,
        pixel_size=pixel_size,
    )


def _split_raster_into_tiles(
    tmp_path,
    prefix,
    reg_arr,
    mask_arr,
    row_ranges,
    col_ranges,
    *,
    origin=(0.0, 250.0),
    pixel_size=(10.0, 10.0),
    dtype="uint8",
    reg_nodata=0,
):
    """Split 3D arrays into non-overlapping tile GeoTIFFs along (row_ranges, col_ranges)."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    reg_tiles = []
    mask_tiles = []
    out_tiles = []
    slices = []
    ox, oy = origin
    px, py = pixel_size

    for t_idx, (r0, r1) in enumerate(row_ranges):
        for c_idx, (c0, c1) in enumerate(col_ranges):
            tile_orig = (ox + c0 * px, oy - r0 * py)
            rp = tmp_path / f"{prefix}_reg_{t_idx}_{c_idx}.tif"
            mp = tmp_path / f"{prefix}_msk_{t_idx}_{c_idx}.tif"
            op = tmp_path / f"{prefix}_out_{t_idx}_{c_idx}.tif"
            _write_raster(
                rp,
                reg_arr[:, r0:r1, c0:c1],
                origin=tile_orig,
                pixel_size=pixel_size,
                dtype=dtype,
                nodata=reg_nodata,
            )
            _write_raster(
                mp,
                mask_arr[:, r0:r1, c0:c1],
                origin=tile_orig,
                pixel_size=pixel_size,
                dtype="uint8",
                nodata=0,
            )
            reg_tiles.append(rp)
            mask_tiles.append(mp)
            out_tiles.append(op)
            slices.append((r0, r1, c0, c1))

    return reg_tiles, mask_tiles, out_tiles, slices


def _mosaic_tile_outputs(out_tiles, slices, shape, dtype):
    """Assemble output tile GeoTIFFs back into a 2D NumPy array."""
    mosaic = np.zeros(shape, dtype=dtype)
    for op, (r0, r1, c0, c1) in zip(out_tiles, slices):
        with rasterio.open(op) as src:
            mosaic[r0:r1, c0:c1] = src.read(1)
    return mosaic


def test_calc_fill_regions_knn(tmp_path):
    """Test KNN spatial region fill across summary metrics."""
    ref_p = tmp_path / "knn_ref.tif"
    mask_p = tmp_path / "knn_mask.tif"
    out_p = tmp_path / "knn_out.tif"

    ref_arr = np.array(
        [
            [
                [10, 10, -99, -99],
                [10, -99, -99, 30],
                [-99, -99, 30, 30],
                [-99, 30, 30, 30],
            ]
        ],
        dtype=np.int32,
    )
    fill_mask = np.array(
        [
            [
                [0, 0, 1, 1],
                [0, 1, 1, 0],
                [1, 1, 0, 0],
                [1, 0, 0, 0],
            ]
        ],
        dtype=np.int32,
    )
    _write_raster(ref_p, ref_arr, dtype="int32", nodata=-99)
    _write_raster(mask_p, fill_mask, dtype="int32")

    ic_fill.calc_fill_regions_knn(
        str(ref_p),
        str(mask_p),
        str(out_p),
        -99,
        1,
        k=3,
        summary=rsgispy.SUMTYPE_MODE,
        use_pbar=False,
    )
    with rasterio.open(out_p) as src:
        out = src.read(1)
        # Non-fill pixels remain 0; (1, 1) nearest 3 valid neighbours are all 10
        assert out[0, 0] == 0
        assert out[1, 1] == 10
        assert out[2, 1] == 30


def test_region_growing_fill_balanced_voronoi(tmp_path):
    """Test balanced nearest-boundary region growing on diagonal and cardinal seeds."""
    reg_p = tmp_path / "rg_reg_diag.tif"
    mask_p = tmp_path / "rg_mask_diag.tif"
    out_p = tmp_path / "rg_out_diag.tif"

    reg_arr = np.zeros((1, 9, 9), dtype=np.uint8)
    reg_arr[0, 0, 0] = 4
    reg_arr[0, 8, 8] = 34
    mask_arr = np.ones((1, 9, 9), dtype=np.uint8)

    _write_raster(reg_p, reg_arr, dtype="uint8", nodata=0)
    _write_raster(mask_p, mask_arr, dtype="uint8", nodata=0)

    ic_fill.region_growing_fill(
        reg_p,
        mask_p,
        out_p,
        calc_stats=True,
        use_pbar=False,
    )
    with rasterio.open(out_p) as src:
        out = src.read(1)
        assert out.dtype == np.uint8
        assert src.nodata == 0
        tags = src.tags(1)
        assert float(tags["STATISTICS_MINIMUM"]) == 4.0
        assert float(tags["STATISTICS_MAXIMUM"]) == 34.0

    rows, cols = np.indices((9, 9))
    assert np.all(out[rows + cols < 8] == 4)
    assert np.all(out[rows + cols > 8] == 34)
    assert out[0, 0] == 4
    assert out[8, 8] == 34
    assert np.all(np.isin(out, [4, 34]))

    # Non-axis-aligned triangle check: (0,0)=1, (0,8)=2, (8,4)=3 -> central pixel (3,4)
    # Nearest to (8,4) is dist 5; to (0,0)/(0,8) is dist sqrt(9+16)=5 -> closer to (0,x) at row 2
    reg_tri = np.zeros((1, 9, 9), dtype=np.uint8)
    reg_tri[0, 0, 0] = 1
    reg_tri[0, 0, 8] = 2
    reg_tri[0, 8, 4] = 3
    reg_tri_p = tmp_path / "rg_tri.tif"
    out_tri_p = tmp_path / "rg_tri_out.tif"
    _write_raster(reg_tri_p, reg_tri, dtype="uint8", nodata=0)
    ic_fill.region_growing_fill(
        reg_tri_p, mask_p, out_tri_p, calc_stats=False, use_pbar=False
    )
    with rasterio.open(out_tri_p) as src:
        tri = src.read(1)
    assert tri[1, 1] == 1
    assert tri[1, 7] == 2
    assert tri[6, 4] == 3


def test_region_growing_fill_barriers_and_disconnected(tmp_path):
    """Test geodesic barrier routing and disconnected component isolation."""
    reg_p = tmp_path / "rg_reg_barrier.tif"
    mask_p = tmp_path / "rg_mask_barrier.tif"
    out8_p = tmp_path / "rg_out_barrier8.tif"
    out4_p = tmp_path / "rg_out_barrier4.tif"

    # Left block is connected except separated by a wall with an opening only at bottom,
    # plus a completely disconnected island on cols 6..7.
    mask = np.array(
        [
            [
                [1, 1, 0, 1, 1, 0, 1, 1],
                [1, 1, 0, 1, 1, 0, 1, 1],
                [1, 1, 0, 1, 1, 0, 1, 1],
                [1, 1, 0, 1, 1, 0, 0, 0],
                [1, 1, 1, 1, 1, 0, 1, 1],
            ]
        ],
        dtype=np.uint8,
    )
    # Seed 4 at (0, 0), Seed 34 at (3, 4) — 34 must dominate upper pocket (0, 3) around wall
    reg = np.full((1, 5, 8), -99, dtype=np.int32)
    reg[0, 0, 0] = 4
    reg[0, 3, 4] = 34

    _write_raster(reg_p, reg, dtype="int32", nodata=-99)
    _write_raster(mask_p, mask, dtype="uint8", nodata=0)

    ic_fill.region_growing_fill(
        reg_p,
        mask_p,
        out8_p,
        ref_no_data=-99,
        fill_region_val=1,
        connectivity=8,
        calc_stats=False,
        use_pbar=False,
    )
    with rasterio.open(out8_p) as src:
        out8 = src.read(1)
        assert out8.dtype == np.int32
        assert src.nodata == -99

    # Wall and non-mask pixels remain -99
    assert np.all(out8[:, 2][:4] == -99)
    assert np.all(out8[:, 5] == -99)
    # Disconnected unseeded islands (cols 6..7) remain unfilled (-99)
    assert np.all(out8[:, 6:] == -99)
    # Upper pocket behind the barrier is claimed by 34 (geodesically much closer than 4)
    assert out8[0, 3] == 34
    assert out8[0, 4] == 34
    # Left side claimed by 4 above the opening; 34 reaches (4, 1) first (17 vs 22)
    assert np.all(out8[:4, :2] == 4)
    assert out8[4, 0] == 4
    assert out8[4, 1] == 34

    # 4-connectivity check with diagonal-only gap across barrier
    mask_diag = np.array(
        [
            [
                [1, 1, 0, 1],
                [1, 0, 1, 1],
                [1, 1, 0, 1],
            ]
        ],
        dtype=np.uint8,
    )
    reg_diag = np.zeros((1, 3, 4), dtype=np.uint8)
    reg_diag[0, 0, 0] = 7
    reg_diag_p = tmp_path / "rg_reg_4c.tif"
    mask_diag_p = tmp_path / "rg_mask_4c.tif"
    _write_raster(reg_diag_p, reg_diag, dtype="uint8", nodata=0)
    _write_raster(mask_diag_p, mask_diag, dtype="uint8", nodata=0)

    ic_fill.region_growing_fill(
        reg_diag_p,
        mask_diag_p,
        out4_p,
        connectivity=4,
        calc_stats=False,
        use_pbar=False,
    )
    with rasterio.open(out4_p) as src:
        out4 = src.read(1)
    # Under 4-connectivity, (0,1)->(1,2) cannot cross diagonal-only choke point
    assert out4[0, 1] == 7
    assert out4[2, 1] == 7
    assert out4[1, 2] == 0
    assert out4[0, 3] == 0


def test_region_growing_fill_adjacent_overlap_max_iters_and_nodata(tmp_path):
    """Test adjacent masks, spatial intersection offsets, max_iters, and float/NaN handling."""
    # 1) Adjacent seed outside fill_mask == 2 + max_iters
    reg = np.zeros((1, 7, 7), dtype=np.uint8)
    reg[0, 3, 3] = 5
    mask = np.full((1, 7, 7), 2, dtype=np.uint8)
    mask[0, 3, 3] = 0  # seed sits just outside fill_mask==2

    reg_p = tmp_path / "rg_adj_reg.tif"
    mask_p = tmp_path / "rg_adj_mask.tif"
    out_p = tmp_path / "rg_adj_out.tif"
    _write_raster(reg_p, reg, dtype="uint8", nodata=0)
    _write_raster(mask_p, mask, dtype="uint8", nodata=0)

    ic_fill.region_growing_fill(
        reg_p,
        mask_p,
        out_p,
        fill_region_val=2,
        max_iters=1,
        no_data_val=99,
        calc_stats=False,
        use_pbar=False,
    )
    with rasterio.open(out_p) as src:
        out = src.read(1)
        assert src.nodata == 99
    # Centre (outside fill_mask==2) stays nodata=99; 3x3 ring around it gets 5; rest stays 99
    assert out[3, 3] == 99
    assert out[2, 2] == 5 and out[3, 4] == 5 and out[4, 4] == 5
    assert out[1, 3] == 99 and out[0, 0] == 99

    # max_iters=0 does not expand beyond initial seed pixels inside fill_mask
    out_zero_p = tmp_path / "rg_zero_out.tif"
    mask_full = np.full((1, 7, 7), 2, dtype=np.uint8)
    mask_full_p = tmp_path / "rg_full_mask.tif"
    _write_raster(mask_full_p, mask_full, dtype="uint8", nodata=0)
    ic_fill.region_growing_fill(
        reg_p,
        mask_full_p,
        out_zero_p,
        fill_region_val=2,
        max_iters=0,
        calc_stats=False,
        use_pbar=False,
    )
    with rasterio.open(out_zero_p) as src:
        out_zero = src.read(1)
    assert out_zero[3, 3] == 5
    assert int(np.count_nonzero(out_zero == 5)) == 1

    # 2) Partial spatial overlap with pixel offsets
    reg_big = np.zeros((1, 6, 6), dtype=np.uint8)
    reg_big[0, 1, 1] = 11
    reg_big[0, 4, 4] = 22
    mask_shifted = np.ones((1, 4, 4), dtype=np.uint8)
    reg_big_p = tmp_path / "rg_big.tif"
    mask_shifted_p = tmp_path / "rg_shifted.tif"
    out_overlap_p = tmp_path / "rg_overlap.tif"
    _write_raster(
        reg_big_p,
        reg_big,
        origin=(0.0, 60.0),
        pixel_size=(10.0, 10.0),
        dtype="uint8",
        nodata=0,
    )
    _write_raster(
        mask_shifted_p,
        mask_shifted,
        origin=(10.0, 50.0),
        pixel_size=(10.0, 10.0),
        dtype="uint8",
        nodata=0,
    )
    ic_fill.region_growing_fill(
        reg_big_p,
        mask_shifted_p,
        out_overlap_p,
        calc_stats=False,
        use_pbar=False,
    )
    with rasterio.open(out_overlap_p) as src:
        ov = src.read(1)
        assert ov.shape == (4, 4)
        assert ov[0, 0] == 11
        assert ov[3, 3] == 22

    # 3) Float raster with NaN nodata + works without scipy
    flt_reg = np.full((1, 4, 4), np.nan, dtype=np.float32)
    flt_reg[0, 0, 0] = 1.5
    flt_reg[0, 3, 3] = 2.5
    flt_mask = np.ones((1, 4, 4), dtype=np.uint8)
    flt_reg_p = tmp_path / "rg_flt_reg.tif"
    flt_mask_p = tmp_path / "rg_flt_mask.tif"
    flt_out_p = tmp_path / "rg_flt_out.tif"
    _write_raster(flt_reg_p, flt_reg, dtype="float32", nodata=np.nan)
    _write_raster(flt_mask_p, flt_mask, dtype="uint8", nodata=0)

    ic_fill.region_growing_fill(
        flt_reg_p, flt_mask_p, flt_out_p, calc_stats=False, use_pbar=False
    )
    with rasterio.open(flt_out_p) as src:
        fout = src.read(1)
        assert fout.dtype == np.float32
        assert fout[0, 0] == pytest.approx(1.5)
        assert fout[3, 3] == pytest.approx(2.5)
        assert np.all(np.isin(fout, [1.5, 2.5]))

    # Casting NaN-nodata float seed raster to uint8 falls back out nodata to 0
    uint_out_p = tmp_path / "rg_flt_to_uint.tif"
    ic_fill.region_growing_fill(
        flt_reg_p,
        flt_mask_p,
        uint_out_p,
        datatype=rsgispy.TYPE_8UINT,
        calc_stats=False,
        use_pbar=False,
    )
    with rasterio.open(uint_out_p) as src:
        assert src.dtypes[0] == "uint8"
        assert src.nodata == 0


def test_region_growing_fill_errors(tmp_path):
    """Test input validation errors in region_growing_fill."""
    reg_p = tmp_path / "rg_err_reg.tif"
    mask_p = tmp_path / "rg_err_mask.tif"
    mask_crs_p = tmp_path / "rg_err_crs.tif"
    empty_reg_p = tmp_path / "rg_err_empty.tif"
    out_p = tmp_path / "rg_err_out.tif"

    reg = np.array([[[0, 1], [0, 0]]], dtype=np.uint8)
    mask = np.ones((1, 2, 2), dtype=np.uint8)
    empty_reg = np.zeros((1, 2, 2), dtype=np.uint8)

    _write_raster(reg_p, reg, dtype="uint8", nodata=0)
    _write_raster(mask_p, mask, dtype="uint8", nodata=0)
    _write_raster(mask_crs_p, mask, crs="EPSG:4326", dtype="uint8", nodata=0)
    _write_raster(empty_reg_p, empty_reg, dtype="uint8", nodata=0)

    with pytest.raises(rsgispy.RSGISPyException):
        ic_fill.region_growing_fill(tmp_path / "missing.tif", mask_p, out_p)

    with pytest.raises(rsgispy.RSGISPyException):
        ic_fill.region_growing_fill(reg_p, tmp_path / "missing_mask.tif", out_p)

    with pytest.raises(rsgispy.RSGISPyException, match="connectivity"):
        ic_fill.region_growing_fill(reg_p, mask_p, out_p, connectivity=6)

    with pytest.raises(rsgispy.RSGISPyException, match="max_iters"):
        ic_fill.region_growing_fill(reg_p, mask_p, out_p, max_iters=-1)

    with pytest.raises(rsgispy.RSGISPyException, match="projection"):
        ic_fill.region_growing_fill(reg_p, mask_crs_p, out_p)

    with pytest.raises(rsgispy.RSGISPyException, match="No valid seed"):
        ic_fill.region_growing_fill(empty_reg_p, mask_p, out_p)


@pytest.fixture
def irreg_tiled_scene(tmp_path):
    """A 15 x 15 seed / mask scene, its global 8-connected fill and its split into
    irregular tiles (in raster order and shuffled)."""
    reg_p = tmp_path / "rgt_reg.tif"
    mask_p = tmp_path / "rgt_mask.tif"
    global_p = tmp_path / "rgt_global.tif"

    # 15x15 grid split into irregular 4x4 tiles (widths/heights mixing 4 and 3)
    # Seed 4 at (1, 1) and seed 34 at (13, 12), separated by a serpentine wall that forces
    # waves to travel right, down, left, AND up across multiple empty-seed tiles.
    # Also includes a completely empty-mask region (rows 0..2, cols 12..14) and an
    # unreachable island at (0, 14).
    mask = np.ones((1, 15, 15), dtype=np.uint8)
    mask[0, :11, 5] = 0
    mask[0, 4:, 10] = 0
    mask[0, :3, 12:] = 0
    mask[0, 0, 14] = 1

    reg = np.zeros((1, 15, 15), dtype=np.uint8)
    reg[0, 1, 1] = 4
    reg[0, 13, 12] = 34

    origin = (0.0, 250.0)
    _write_raster(reg_p, reg, origin=origin, dtype="uint8", nodata=0)
    _write_raster(mask_p, mask, origin=origin, dtype="uint8", nodata=0)

    ic_fill.region_growing_fill(
        reg_p, mask_p, global_p, connectivity=8, calc_stats=False, use_pbar=False
    )
    with rasterio.open(global_p) as s_glob:
        g_arr = s_glob.read(1)

    row_ranges = [(0, 4), (4, 7), (7, 12), (12, 15)]
    col_ranges = [(0, 3), (3, 7), (7, 11), (11, 15)]
    reg_tiles, msk_tiles, out1_tiles, slices = _split_raster_into_tiles(
        tmp_path,
        "irreg",
        reg,
        mask,
        row_ranges,
        col_ranges,
        origin=origin,
        dtype="uint8",
        reg_nodata=0,
    )

    # Shuffle tile order to verify order-independence
    perm = list(reversed(range(len(reg_tiles))))
    reg_shuf = [reg_tiles[i] for i in perm]
    msk_shuf = [msk_tiles[i] for i in perm]
    out1_shuf = [out1_tiles[i] for i in perm]
    slices_shuf = [slices[i] for i in perm]
    return {
        "reg_p": reg_p,
        "mask_p": mask_p,
        "origin": origin,
        "g_arr": g_arr,
        "reg_tiles": reg_tiles,
        "msk_tiles": msk_tiles,
        "slices": slices,
        "reg_shuf": reg_shuf,
        "msk_shuf": msk_shuf,
        "out1_shuf": out1_shuf,
        "slices_shuf": slices_shuf,
    }


@pytest.mark.slow
def test_region_growing_fill_tiled_matches_global_one_core(irreg_tiled_scene):
    """Tile-list region_growing_fill_tiled (shuffled tiles, 1 core) matches the
    global output, leaves the unreachable island empty and writes statistics."""
    scn = irreg_tiled_scene
    g_arr = scn["g_arr"]
    out1_shuf = scn["out1_shuf"]
    slices_shuf = scn["slices_shuf"]
    ic_fill.region_growing_fill_tiled(
        scn["reg_shuf"],
        scn["msk_shuf"],
        out1_shuf,
        connectivity=8,
        n_cores=1,
        calc_stats=True,
        use_pbar=False,
    )
    t1_arr = _mosaic_tile_outputs(out1_shuf, slices_shuf, (15, 15), np.uint8)
    np.testing.assert_array_equal(t1_arr, g_arr)
    assert t1_arr[0, 14] == 0

    with rasterio.open(out1_shuf[-1]) as s_tile:
        assert s_tile.dtypes[0] == "uint8"
        assert s_tile.nodata == 0
        tags = s_tile.tags(1)
        assert float(tags["STATISTICS_MINIMUM"]) == 4.0


@pytest.mark.slow
def test_region_growing_fill_tiled_matches_global_two_cores(tmp_path, irreg_tiled_scene):
    """Tile-list region_growing_fill_tiled (shuffled tiles, 2 cores) matches the global output."""
    scn = irreg_tiled_scene
    g_arr = scn["g_arr"]
    reg_shuf = scn["reg_shuf"]
    slices_shuf = scn["slices_shuf"]
    # Parallel execution with n_cores=2
    out2_shuf = [tmp_path / f"irreg_out2_{i}.tif" for i in range(len(reg_shuf))]
    ic_fill.region_growing_fill_tiled(
        reg_shuf,
        scn["msk_shuf"],
        out2_shuf,
        connectivity=8,
        n_cores=2,
        calc_stats=False,
        use_pbar=False,
    )
    t2_arr = _mosaic_tile_outputs(out2_shuf, slices_shuf, (15, 15), np.uint8)
    np.testing.assert_array_equal(t2_arr, g_arr)


@pytest.mark.slow
def test_region_growing_fill_tiled_4_connectivity_max_iters_matches_global(
    tmp_path, irreg_tiled_scene
):
    """4-connectivity with max_iters=3 on 2 cores matches the global output identically."""
    scn = irreg_tiled_scene
    reg_tiles = scn["reg_tiles"]
    msk_tiles = scn["msk_tiles"]
    slices = scn["slices"]
    # Test 4-connectivity + max_iters=3 + n_cores=2 matches global identically
    glob4_p = tmp_path / "rgt_glob4.tif"
    ic_fill.region_growing_fill(
        scn["reg_p"],
        scn["mask_p"],
        glob4_p,
        connectivity=4,
        max_iters=3,
        calc_stats=False,
        use_pbar=False,
    )
    out4_tiles = [tmp_path / f"irreg_out4_{i}.tif" for i in range(len(reg_tiles))]
    ic_fill.region_growing_fill_tiled(
        reg_tiles,
        msk_tiles,
        out4_tiles,
        connectivity=4,
        max_iters=3,
        n_cores=2,
        calc_stats=False,
        use_pbar=False,
    )
    t4_arr = _mosaic_tile_outputs(out4_tiles, slices, (15, 15), np.uint8)
    with rasterio.open(glob4_p) as s_g4:
        np.testing.assert_array_equal(t4_arr, s_g4.read(1))


@pytest.mark.slow
def test_region_growing_fill_tiled_one_pixel_tiles_int32_nodata(tmp_path):
    """1 x 1 pixel tiles of an int32 grid with nodata -99 match the global output."""
    origin = (0.0, 250.0)
    # Extreme 1x1 tile files on a 5x5 int32 grid with non-zero nodata (-99)
    small_reg_p = tmp_path / "rgt_small_reg.tif"
    small_msk_p = tmp_path / "rgt_small_msk.tif"
    small_g_p = tmp_path / "rgt_small_g.tif"
    s_reg = np.zeros((1, 5, 5), dtype=np.int32)
    s_reg[0, 0, 0] = 10
    s_reg[0, 4, 4] = 20
    s_msk = np.ones((1, 5, 5), dtype=np.uint8)
    _write_raster(small_reg_p, s_reg, origin=origin, dtype="int32", nodata=-99)
    _write_raster(small_msk_p, s_msk, origin=origin, dtype="uint8", nodata=0)

    ic_fill.region_growing_fill(
        small_reg_p,
        small_msk_p,
        small_g_p,
        ref_no_data=-99,
        calc_stats=False,
        use_pbar=False,
    )
    one_by_one = [(i, i + 1) for i in range(5)]
    s_reg_t, s_msk_t, s_out_t, s_slices = _split_raster_into_tiles(
        tmp_path,
        "px1",
        s_reg,
        s_msk,
        one_by_one,
        one_by_one,
        origin=origin,
        dtype="int32",
        reg_nodata=-99,
    )
    ic_fill.region_growing_fill_tiled(
        s_reg_t,
        s_msk_t,
        s_out_t,
        ref_no_data=-99,
        n_cores=2,
        calc_stats=True,
        use_pbar=False,
    )
    small_mosaic = _mosaic_tile_outputs(s_out_t, s_slices, (5, 5), np.int32)
    with rasterio.open(small_g_p) as sg, rasterio.open(s_out_t[0]) as st0:
        assert st0.nodata == -99
        np.testing.assert_array_equal(small_mosaic, sg.read(1))


@pytest.fixture
def eez_tiles(tmp_path):
    """A 2 x 2 layout of 6 x 6 seed / mask tiles, two of them without seeds."""
    # Build a 2x2 layout of 6x6 tiles:
    # Tile 0 (TL): seed 4 at (1, 1) outside fill_mask
    # Tile 1 (TR) & Tile 2 (BL): EEZ tiles with NO seeds
    # Tile 3 (BR): seed 34 at (4, 3) (asymmetric to avoid exact equidistant ties)
    reg_tiles = []
    msk_tiles = []
    origins = [(0.0, 120.0), (60.0, 120.0), (0.0, 60.0), (60.0, 60.0)]
    for idx, orig in enumerate(origins):
        r_arr = np.zeros((1, 6, 6), dtype=np.uint8)
        m_arr = np.ones((1, 6, 6), dtype=np.uint8)
        if idx == 0:
            r_arr[0, 1, 1] = 4
            m_arr[0, 1, 1] = 0  # seed outside fill_mask
        elif idx == 3:
            r_arr[0, 4, 3] = 34

        rp = tmp_path / f"eez_reg_{idx}.tif"
        mp = tmp_path / f"eez_msk_{idx}.tif"
        _write_raster(
            rp, r_arr, origin=orig, pixel_size=(10.0, 10.0), dtype="uint8", nodata=0
        )
        _write_raster(
            mp, m_arr, origin=orig, pixel_size=(10.0, 10.0), dtype="uint8", nodata=0
        )
        reg_tiles.append(rp)
        msk_tiles.append(mp)
    return reg_tiles, msk_tiles


def test_region_growing_fill_tiled_unseeded_eez_and_outside_seeds(tmp_path, eez_tiles):
    """Unseeded EEZ tiles are filled, seeds outside the fill mask are cleared, the
    result matches the global fill and the tmp_dir workspace is cleaned up."""
    reg_tiles, msk_tiles = eez_tiles
    # Reference global 12x12 arrays
    full_reg = np.zeros((1, 12, 12), dtype=np.uint8)
    full_reg[0, 1, 1] = 4
    full_reg[0, 10, 9] = 34
    full_msk = np.ones((1, 12, 12), dtype=np.uint8)
    full_msk[0, 1, 1] = 0
    full_reg_p = tmp_path / "eez_full_reg.tif"
    full_msk_p = tmp_path / "eez_full_msk.tif"
    glob_out = tmp_path / "eez_global.tif"
    _write_raster(
        full_reg_p,
        full_reg,
        origin=(0.0, 120.0),
        pixel_size=(10.0, 10.0),
        dtype="uint8",
        nodata=0,
    )
    _write_raster(
        full_msk_p,
        full_msk,
        origin=(0.0, 120.0),
        pixel_size=(10.0, 10.0),
        dtype="uint8",
        nodata=0,
    )
    ic_fill.region_growing_fill(
        full_reg_p, full_msk_p, glob_out, calc_stats=False, use_pbar=False
    )
    with rasterio.open(glob_out) as sg:
        g_arr = sg.read(1)

    work_parent = tmp_path / "custom_tmp"
    out_tiles = [tmp_path / "nested_out" / f"eez_out_{idx}.tif" for idx in range(4)]
    ic_fill.region_growing_fill_tiled(
        reg_tiles,
        msk_tiles,
        out_tiles,
        n_cores=2,
        tmp_dir=work_parent,
        calc_stats=False,
        use_pbar=False,
    )

    tile_slices = [(0, 6, 0, 6), (0, 6, 6, 12), (6, 12, 0, 6), (6, 12, 6, 12)]
    t_arr = _mosaic_tile_outputs(out_tiles, tile_slices, (12, 12), np.uint8)
    np.testing.assert_array_equal(t_arr, g_arr)
    assert t_arr[1, 1] == 0  # outside fill_mask cleared to nodata
    assert t_arr[0, 11] == 4 or t_arr[11, 0] == 34  # unseeded EEZ tiles filled

    # Temporary workspace inside work_parent was cleaned up automatically
    assert work_parent.exists()
    assert list(work_parent.iterdir()) == []


def test_region_growing_fill_tiled_sparse_tile_subset(tmp_path, eez_tiles):
    """A sparse L-shaped subset of the tiles (one omitted) fills with n_cores=-1."""
    reg_tiles, msk_tiles = eez_tiles
    # Sparse L-shaped subset (3 of 4 tiles, omitting tile 2) with n_cores=-1
    sparse_out = [tmp_path / f"sparse_out_{i}.tif" for i in (0, 1, 3)]
    ic_fill.region_growing_fill_tiled(
        [reg_tiles[0], reg_tiles[1], reg_tiles[3]],
        [msk_tiles[0], msk_tiles[1], msk_tiles[3]],
        sparse_out,
        n_cores=-1,
        calc_stats=False,
        use_pbar=False,
    )
    with rasterio.open(sparse_out[1]) as s_tr:
        tr_arr = s_tr.read(1)
        assert np.all(np.isin(tr_arr, [4, 34]))


def test_region_growing_fill_tiled_argument_errors(tmp_path, eez_tiles):
    """Empty or mismatched tile lists and invalid parameters raise."""
    reg_tiles, msk_tiles = eez_tiles
    # Validation errors
    err_outs = [tmp_path / f"rgt_err_{i}.tif" for i in range(4)]
    with pytest.raises(rsgispy.RSGISPyException, match="empty"):
        ic_fill.region_growing_fill_tiled([], [], [])
    with pytest.raises(rsgispy.RSGISPyException, match="same length"):
        ic_fill.region_growing_fill_tiled(reg_tiles, msk_tiles[:2], err_outs)
    with pytest.raises(rsgispy.RSGISPyException, match="max_sweeps"):
        ic_fill.region_growing_fill_tiled(
            reg_tiles, msk_tiles, err_outs, max_sweeps=-1
        )
    with pytest.raises(rsgispy.RSGISPyException, match="connectivity"):
        ic_fill.region_growing_fill_tiled(
            reg_tiles, msk_tiles, err_outs, connectivity=6
        )
    with pytest.raises(rsgispy.RSGISPyException, match="max_iters"):
        ic_fill.region_growing_fill_tiled(
            reg_tiles, msk_tiles, err_outs, max_iters=-2
        )


def test_region_growing_fill_tiled_tile_grid_errors(tmp_path, eez_tiles):
    """Overlapping, unaligned, mixed-CRS and seedless tile sets raise."""
    reg_tiles, msk_tiles = eez_tiles
    err_outs = [tmp_path / f"rgt_err_{i}.tif" for i in range(4)]
    # Overlapping tile check
    ov_reg = tmp_path / "ov_reg.tif"
    ov_msk = tmp_path / "ov_msk.tif"
    _write_raster(
        ov_reg,
        np.zeros((1, 6, 6), dtype=np.uint8),
        origin=(30.0, 120.0),
        pixel_size=(10.0, 10.0),
        dtype="uint8",
        nodata=0,
    )
    _write_raster(
        ov_msk,
        np.ones((1, 6, 6), dtype=np.uint8),
        origin=(30.0, 120.0),
        pixel_size=(10.0, 10.0),
        dtype="uint8",
        nodata=0,
    )
    with pytest.raises(rsgispy.RSGISPyException, match="overlap"):
        ic_fill.region_growing_fill_tiled(
            [reg_tiles[0], ov_reg],
            [msk_tiles[0], ov_msk],
            err_outs[:2],
        )

    # Unaligned pixel grid origin
    unaligned_reg = tmp_path / "unaligned_reg.tif"
    unaligned_msk = tmp_path / "unaligned_msk.tif"
    _write_raster(
        unaligned_reg,
        np.zeros((1, 6, 6), dtype=np.uint8),
        origin=(63.0, 120.0),
        pixel_size=(10.0, 10.0),
        dtype="uint8",
        nodata=0,
    )
    _write_raster(
        unaligned_msk,
        np.ones((1, 6, 6), dtype=np.uint8),
        origin=(63.0, 120.0),
        pixel_size=(10.0, 10.0),
        dtype="uint8",
        nodata=0,
    )
    with pytest.raises(rsgispy.RSGISPyException, match="aligned"):
        ic_fill.region_growing_fill_tiled(
            [reg_tiles[0], unaligned_reg],
            [msk_tiles[0], unaligned_msk],
            err_outs[:2],
        )

    # Different CRS between tiles
    crs_reg = tmp_path / "crs_reg.tif"
    crs_msk = tmp_path / "crs_msk.tif"
    _write_raster(
        crs_reg,
        np.zeros((1, 6, 6), dtype=np.uint8),
        origin=(60.0, 120.0),
        pixel_size=(10.0, 10.0),
        crs="EPSG:32631",
        dtype="uint8",
        nodata=0,
    )
    _write_raster(
        crs_msk,
        np.ones((1, 6, 6), dtype=np.uint8),
        origin=(60.0, 120.0),
        pixel_size=(10.0, 10.0),
        crs="EPSG:32631",
        dtype="uint8",
        nodata=0,
    )
    with pytest.raises(rsgispy.RSGISPyException, match="CRS"):
        ic_fill.region_growing_fill_tiled(
            [reg_tiles[0], crs_reg],
            [msk_tiles[0], crs_msk],
            err_outs[:2],
        )

    # No valid seeds across all tiles
    with pytest.raises(rsgispy.RSGISPyException, match="No valid seed"):
        ic_fill.region_growing_fill_tiled(
            [reg_tiles[1], reg_tiles[2]],
            [msk_tiles[1], msk_tiles[2]],
            err_outs[:2],
            use_pbar=False,
        )


def test_calc_fill_regions_knn_nan_fill_val_and_int_clamp(tmp_path):
    ref_arr = np.full((8, 8), 300.0, dtype=np.float32)  # exceeds uint8 max 255
    ref_arr[0, 0] = -1.0
    mask_arr = np.ones((8, 8), dtype=np.float32)
    mask_arr[3:5, 3:5] = np.nan

    ref_img = write_raster(tmp_path / "knn_ref.tif", ref_arr, nodata=-1.0, origin=(500000.0, 5800000.0))
    msk_img = write_raster(tmp_path / "knn_msk.tif", mask_arr, origin=(500000.0, 5800000.0))
    out_img = str(tmp_path / "knn_out.tif")

    ic_fill.calc_fill_regions_knn(
        in_ref_img=ref_img,
        in_fill_regions_img=msk_img,
        output_img=out_img,
        ref_no_data=-1.0,
        fill_region_val=float("nan"),
        k=3,
        summary=rsgispy.SUMTYPE_MEAN,
        datatype=rsgispy.TYPE_8UINT,
        use_pbar=False,
    )
    with rasterio.open(out_img) as dst:
        out_arr = dst.read(1)
        assert out_arr.dtype == np.uint8
        # Filled NaN region clamped safely to 255 rather than wrapping mod 256
        assert np.all(out_arr[3:5, 3:5] == 255)
        assert out_arr[0, 0] == 0
