"""
Unit tests for classification post-processing functions in
:mod:`rsgispy.classification.classpostprocess` (majority modal filtering and
connected-component sieve clump elimination).
"""

from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pytest
import rasterio

import rsgispy
import rsgispy.classification as rsgis_cls
import rsgispy.classification.classpostprocess as classpost
import rsgispy.imagefilter.stats as imagefilter_stats
from rsgispy.imageprocess import get_default_block_size, set_default_block_size
from rsgispy.imageutils.colourtable import is_img_thematic

from conftest import write_raster


def _write_raster(
    path: Path,
    bands: Sequence[np.ndarray],
    nodata: Optional[float] = None,
    descriptions: Optional[Sequence[str]] = None,
    colormap: Optional[Dict[int, tuple]] = None,
    cmap_band: int = 1,
) -> Path:
    """Create a synthetic GeoTIFF raster for classification post-processing tests."""
    write_raster(
        path,
        list(bands),
        dtype=bands[0].dtype,
        nodata=nodata,
        origin=(500000.0, 6000000.0),
        band_names=descriptions,
        colormap=colormap,
        colormap_band=cmap_band,
    )
    return path


def _read_band(path: Path, band: int = 1) -> np.ndarray:
    """Read a single raster band into a NumPy array."""
    with rasterio.open(path, "r") as src:
        return src.read(band)


def test_classification_reexports() -> None:
    """Top-level classification package re-exports both post-processing filters."""
    assert rsgis_cls.apply_majority_filter is classpost.apply_majority_filter
    assert rsgis_cls.apply_sieve_filter is classpost.apply_sieve_filter


def test_majority_filter_3x3_and_5x5_smooths_noise(tmp_path: Path) -> None:
    """3x3 and 5x5 majority filters remove isolated noise speckles."""
    arr = np.full((7, 7), 1, dtype=np.uint8)
    arr[3, 3] = 2  # single-pixel speckle in centre
    arr[0, 6] = 3  # corner speckle

    in_img = _write_raster(tmp_path / "cls.tif", [arr], descriptions=["ClassMap"])
    out3 = tmp_path / "maj3.tif"
    out5 = tmp_path / "maj5.tif"

    classpost.apply_majority_filter(
        in_img, out3, filter_size=3, calc_stats=False, use_pbar=False
    )
    res3 = _read_band(out3)
    assert res3.dtype == np.uint8
    assert res3[3, 3] == 1
    assert res3[0, 6] == 1
    assert np.all(res3 == 1)
    assert is_img_thematic(str(out3))
    with rasterio.open(out3, "r") as src:
        assert src.descriptions[0] == "ClassMap"

    # 2x2 block in centre (4 pixels of class 2 out of 25 -> replaced by 1 under 5x5)
    arr[2:4, 2:4] = 2
    in_img2 = _write_raster(tmp_path / "cls2.tif", [arr])
    classpost.apply_majority_filter(
        in_img2, out5, filter_size=5, calc_stats=False, use_pbar=False
    )
    res5 = _read_band(out5)
    assert np.all(res5 == 1)


def test_majority_filter_keep_centre_on_tie_and_matches_imagefilter(
    tmp_path: Path,
) -> None:
    """Tie handling preserves centre when requested or matches apply_mode_filter."""
    # Construct a 3x3 patch with a 4-vs-4 tie around centre pixel:
    # Neighbours around (1,1) have four 2s and four 5s -> tie if centre=5? No:
    # Make window contain: four 1s, four 3s, and one NaN nodata, with centre=3 -> 4 vs 4 tie.
    arr = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, 3.0, 3.0],
            [np.nan, 3.0, 3.0],
        ],
        dtype=np.float32,
    )
    in_img = _write_raster(tmp_path / "tie.tif", [arr])
    out_keep = tmp_path / "tie_keep.tif"
    out_break = tmp_path / "tie_break.tif"
    out_mode = tmp_path / "mode_ref.tif"

    classpost.apply_majority_filter(
        in_img,
        out_keep,
        filter_size=3,
        keep_centre_on_tie=True,
        calc_stats=False,
        use_pbar=False,
    )
    classpost.apply_majority_filter(
        in_img,
        out_break,
        filter_size=3,
        keep_centre_on_tie=False,
        calc_stats=False,
        use_pbar=False,
    )
    imagefilter_stats.apply_mode_filter(
        str(in_img),
        str(out_mode),
        filter_size=3,
        datatype=rsgispy.TYPE_32FLOAT,
        calc_stats=False,
        use_pbar=False,
    )

    res_keep = _read_band(out_keep)
    res_break = _read_band(out_break)
    res_mode = _read_band(out_mode)

    assert res_keep[1, 1] == 3.0
    assert res_break[1, 1] == 1.0
    assert np.allclose(res_break, res_mode, equal_nan=True)


def test_majority_filter_edge_padding_and_nodata_class_zero(
    tmp_path: Path,
) -> None:
    """Out-of-bounds edges do not bias counts, and class 0 is valid when nodata unset."""
    # Top edge has classes 1 and 2; class 0 must NOT get invented by edge padding,
    # while real class 0 inside the image stays valid when nodata metadata is None.
    arr = np.array(
        [
            [2, 2, 1, 1],
            [2, 2, 1, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    in_img = _write_raster(tmp_path / "zero_cls.tif", [arr], nodata=None)
    out_img = tmp_path / "zero_cls_maj.tif"

    classpost.apply_majority_filter(
        in_img,
        out_img,
        filter_size=3,
        no_data_val=None,
        calc_stats=False,
        use_pbar=False,
    )
    res = _read_band(out_img)
    # Top-left pixel (0,0) has 4 neighbours of class 2 -> stays 2
    assert res[0, 0] == 2
    # Bottom-left region stays 0
    assert res[3, 0] == 0
    assert res[3, 1] == 0
    with rasterio.open(out_img, "r") as src:
        assert src.nodata is None


def test_majority_filter_nodata_multiband_colourtable_and_tiling(
    tmp_path: Path,
) -> None:
    """Verify nodata preservation, band selection, colour table copy, and tiled parity."""
    rng = np.random.default_rng(42)
    b1 = np.zeros((10, 12), dtype=np.uint8)
    b2 = rng.choice([1, 2, 3], size=(10, 12)).astype(np.uint8)
    b2[0:3, 0:3] = 255  # nodata region
    cmap = {
        1: (255, 0, 0, 255),
        2: (0, 255, 0, 255),
        3: (0, 0, 255, 255),
        255: (0, 0, 0, 0),
    }

    in_img = _write_raster(
        tmp_path / "mb.tif",
        [b1, b2],
        nodata=255,
        descriptions=["Ignore", "LandCover"],
    )
    in_ctab = _write_raster(
        tmp_path / "sb_ctab.tif",
        [b2],
        nodata=255,
        descriptions=["LandCover"],
        colormap=cmap,
        cmap_band=1,
    )
    out_single = tmp_path / "maj_single.tif"
    out_tiled = tmp_path / "maj_tiled.tif"
    out_ctab = tmp_path / "maj_ctab.tif"

    prev_bs = get_default_block_size()
    try:
        set_default_block_size(64)
        classpost.apply_majority_filter(
            in_img,
            out_single,
            filter_size=3,
            img_band=2,
            calc_stats=True,
            copy_colour_table=False,
            use_pbar=False,
        )
        set_default_block_size(4)
        classpost.apply_majority_filter(
            in_img,
            out_tiled,
            filter_size=3,
            img_band=2,
            calc_stats=False,
            copy_colour_table=False,
            use_pbar=False,
        )
        classpost.apply_majority_filter(
            in_ctab,
            out_ctab,
            filter_size=3,
            img_band=1,
            calc_stats=True,
            copy_colour_table=True,
            use_pbar=False,
        )
    finally:
        set_default_block_size(prev_bs)

    res_single = _read_band(out_single)
    res_tiled = _read_band(out_tiled)
    np.testing.assert_array_equal(res_single, res_tiled)
    np.testing.assert_array_equal(res_single, _read_band(out_ctab))
    assert np.all(res_single[0:3, 0:3] == 255)
    assert is_img_thematic(str(out_single))

    with rasterio.open(out_single, "r") as src:
        assert src.count == 1
        assert src.nodata == 255
        assert src.descriptions[0] == "LandCover"
    with rasterio.open(out_ctab, "r") as src:
        assert src.colormap(1)[2] == (0, 255, 0, 255)


def test_majority_filter_dtypes_and_validation(tmp_path: Path) -> None:
    """Validate error handling and int32/float32 support for apply_majority_filter."""
    arr_i32 = np.full((5, 5), 100000, dtype=np.int32)
    arr_i32[2, 2] = -500
    arr_i32[0, 0] = -9999
    in_i32 = _write_raster(tmp_path / "i32.tif", [arr_i32], nodata=-9999)
    out_i32 = tmp_path / "i32_out.tif"

    classpost.apply_majority_filter(
        in_i32, out_i32, filter_size=3, calc_stats=True, use_pbar=False
    )
    res_i32 = _read_band(out_i32)
    assert res_i32.dtype == np.int32
    assert res_i32[2, 2] == 100000
    assert res_i32[0, 0] == -9999

    with pytest.raises(rsgispy.RSGISPyException):
        classpost.apply_majority_filter(in_i32, out_i32, filter_size=4)
    with pytest.raises(rsgispy.RSGISPyException):
        classpost.apply_majority_filter(in_i32, out_i32, filter_size=1)
    with pytest.raises(rsgispy.RSGISPyException):
        classpost.apply_majority_filter(in_i32, out_i32, img_band=2)
    with pytest.raises(rsgispy.RSGISPyException):
        classpost.apply_majority_filter(tmp_path / "missing.tif", out_i32)


def test_sieve_filter_4_vs_8_connectivity_diagonal_bridge(
    tmp_path: Path,
) -> None:
    """Diagonal 2-pixel bridge of class 2 is <3 for 4-conn and kept for 8-conn."""
    # Inside 6x6 background of class 1, two pixels of class 2 touch diagonally at (2,2) and (3,3).
    # With size_threshold=2? Wait: make a 3-pixel diagonal chain? Let's use 2-pixel diagonal
    # and threshold=2: 4-conn gives two 1-px clumps (<2 -> merged to 1), 8-conn gives 2 px (kept).
    arr = np.ones((6, 6), dtype=np.uint8)
    arr[2, 2] = 2
    arr[3, 3] = 2

    in_img = _write_raster(tmp_path / "diag.tif", [arr])
    out4 = tmp_path / "sieve_c4.tif"
    out8 = tmp_path / "sieve_c8.tif"

    classpost.apply_sieve_filter(
        in_img,
        out4,
        size_threshold=2,
        connectivity=4,
        in_memory=True,
        calc_stats=False,
        use_pbar=False,
    )
    classpost.apply_sieve_filter(
        in_img,
        out8,
        size_threshold=2,
        connectivity=8,
        in_memory=True,
        calc_stats=False,
        use_pbar=False,
    )

    res4 = _read_band(out4)
    res8 = _read_band(out8)
    assert np.all(res4 == 1)
    assert res8[2, 2] == 2 and res8[3, 3] == 2
    assert int(np.count_nonzero(res8 == 2)) == 2


def test_sieve_filter_majority_neighbour_and_tiebreak(tmp_path: Path) -> None:
    """Sieve merges small clump into neighbour with largest shared boundary contact."""
    # Build a vertical boundary between class 10 (left, 4 cols) and class 20 (right, 4 cols).
    # Place a 2-pixel horizontal clump of class 5 protruding mostly into class 20:
    # row 2: [10, 10, 10, 10, 20, 20, 20, 20]
    # row 3: [10, 10, 10,  5,  5, 20, 20, 20] -> wait: count edges cleanly:
    # Put 1x2 clump of class 5 inside class 20 touching class 10 only on its left edge (1 edge)
    # and class 20 on top (2), bottom (2), right (1) => 5 edges with 20 vs 1 edge with 10 -> 20.
    arr = np.full((6, 8), 20, dtype=np.uint8)
    arr[:, :3] = 10  # 18 pixels of class 10
    arr[2, 3:5] = 5  # 2 pixels of class 5; left touches 10 (1 edge), 5 edges touch 20
    in_img = _write_raster(tmp_path / "nbr.tif", [arr])
    out_img = tmp_path / "nbr_sieved.tif"

    classpost.apply_sieve_filter(
        in_img,
        out_img,
        size_threshold=4,
        connectivity=4,
        in_memory=True,
        calc_stats=False,
        use_pbar=False,
    )
    res = _read_band(out_img)
    assert np.all(res[2, 3:5] == 20)

    # Tie-break by larger neighbour clump when boundary contact count is equal:
    # class 30 (12 px) touches (2,2) on top & left (2 edges) and
    # class 15 (8 px) touches (2,2) on bottom & right (2 edges) -> class 30 wins.
    arr_sz = np.full((5, 5), 0, dtype=np.uint8)  # 0 is nodata
    arr_sz[0:2, 0:4] = 30  # 8 px
    arr_sz[2:4, 0:2] = 30  # +4 = 12 px, touches (2,2) on top & left (2 edges)
    arr_sz[2:5, 3:5] = 15  # 6 px
    arr_sz[3:5, 2] = 15  # +2 = 8 px, touches (2,2) on bottom & right (2 edges)
    arr_sz[2, 2] = 99  # 1 px tied with 2 edges to 30 (size 12) and 2 edges to 15 (size 8)
    in_sz = _write_raster(tmp_path / "tie_sz.tif", [arr_sz], nodata=0)
    out_sz = tmp_path / "tie_sz_out.tif"
    classpost.apply_sieve_filter(
        in_sz,
        out_sz,
        size_threshold=4,
        connectivity=4,
        in_memory=True,
        calc_stats=False,
        use_pbar=False,
    )
    res_sz = _read_band(out_sz)
    assert res_sz[2, 2] == 30


def test_sieve_filter_nested_clumps_and_isolated_island(
    tmp_path: Path,
) -> None:
    """Nested small clumps resolve into surrounding kept class; isolated islands stay valid."""
    arr = np.full((7, 7), 0, dtype=np.uint8)
    # Kept background clump of class 1 (5x4 = 20 px), containing nested 1-px speckles 2 & 3
    arr[1:6, 1:5] = 1
    arr[3, 2] = 2
    arr[3, 3] = 3

    # Isolated 2-px island of class 4 surrounded solely by NoData (0)
    arr[0, 5:7] = 4

    in_img = _write_raster(tmp_path / "nested.tif", [arr], nodata=0)
    out_img = tmp_path / "nested_sieved.tif"
    classpost.apply_sieve_filter(
        in_img,
        out_img,
        size_threshold=5,
        connectivity=4,
        in_memory=True,
        calc_stats=False,
        use_pbar=False,
    )
    res = _read_band(out_img)
    # Nested speckles 2 and 3 merge into 1; surrounding nodata stays 0; isolated island stays 4
    assert np.all(res[1:6, 1:5] == 1)
    assert np.all(res[0, 5:7] == 4)
    assert res[0, 0] == 0
    assert res[1, 5] == 0
    assert res[1, 6] == 0


def test_sieve_filter_in_memory_vs_tiled_parity(tmp_path: Path) -> None:
    """Tiled streaming sieve with small blocks is bit-identical to in-memory sieve."""
    rng = np.random.default_rng(12345)
    arr = rng.choice([0, 1, 2, 3, 4], size=(17, 19), p=[0.1, 0.3, 0.25, 0.2, 0.15]).astype(
        np.uint8
    )
    # Explicitly add a cross-tile diagonal bridge crossing block boundary (3,3)-(4,4)
    arr[3:5, 3:5] = np.array([[2, 1], [1, 2]], dtype=np.uint8)

    in_img = _write_raster(
        tmp_path / "parity.tif",
        [arr],
        nodata=0,
        descriptions=["Habitat"],
        colormap={
            0: (0, 0, 0, 0),
            1: (10, 20, 30, 255),
            2: (40, 50, 60, 255),
            3: (70, 80, 90, 255),
            4: (100, 110, 120, 255),
        },
    )

    for conn in (4, 8):
        for bs in (3, 4, 5):
            out_mem = tmp_path / f"mem_c{conn}_b{bs}.tif"
            out_tile = tmp_path / f"tile_c{conn}_b{bs}.tif"

            classpost.apply_sieve_filter(
                in_img,
                out_mem,
                size_threshold=6,
                connectivity=conn,
                in_memory=True,
                calc_stats=False,
                copy_colour_table=True,
                use_pbar=False,
            )
            classpost.apply_sieve_filter(
                in_img,
                out_tile,
                size_threshold=6,
                connectivity=conn,
                in_memory=False,
                block_size=bs,
                calc_stats=False,
                copy_colour_table=True,
                use_pbar=False,
            )

            np.testing.assert_array_equal(_read_band(out_mem), _read_band(out_tile))
            with rasterio.open(out_tile, "r") as src:
                assert src.nodata == 0
                assert src.descriptions[0] == "Habitat"
                assert src.colormap(1)[4] == (100, 110, 120, 255)


def test_sieve_filter_float_nan_class_zero_empty_and_validation(
    tmp_path: Path,
) -> None:
    """Verify float32 NaN preservation, class 0 validity, all-nodata inputs, and errors."""
    # 1. Float32 with NaN invalid area + valid classes including 0.0
    arr_f = np.full((6, 6), 1.0, dtype=np.float32)
    arr_f[2, 2] = 0.0  # 1-px speckle of class 0.0 -> merged to 1.0
    arr_f[0:2, 0:2] = np.nan
    in_f = _write_raster(tmp_path / "f32.tif", [arr_f], nodata=None)
    out_f_mem = tmp_path / "f32_mem.tif"
    out_f_tile = tmp_path / "f32_tile.tif"

    classpost.apply_sieve_filter(
        in_f,
        out_f_mem,
        size_threshold=3,
        connectivity=4,
        in_memory=True,
        calc_stats=False,
        use_pbar=False,
    )
    classpost.apply_sieve_filter(
        in_f,
        out_f_tile,
        size_threshold=3,
        connectivity=4,
        in_memory=False,
        block_size=3,
        calc_stats=False,
        use_pbar=False,
    )
    res_mem = _read_band(out_f_mem)
    res_tile = _read_band(out_f_tile)
    assert res_mem.dtype == np.float32
    assert np.all(np.isnan(res_mem[0:2, 0:2]))
    assert res_mem[2, 2] == 1.0
    np.testing.assert_allclose(res_mem, res_tile, equal_nan=True)

    # 2. All-nodata image (both in-memory and tiled) + size_threshold == 1 passthrough
    arr_nd = np.full((6, 6), 255, dtype=np.uint8)
    in_nd = _write_raster(tmp_path / "all_nd.tif", [arr_nd], nodata=255)
    out_nd_mem = tmp_path / "all_nd_mem.tif"
    out_nd_tile = tmp_path / "all_nd_tile.tif"
    classpost.apply_sieve_filter(
        in_nd,
        out_nd_mem,
        size_threshold=4,
        in_memory=True,
        calc_stats=False,
        use_pbar=False,
    )
    classpost.apply_sieve_filter(
        in_nd,
        out_nd_tile,
        size_threshold=4,
        in_memory=False,
        block_size=3,
        calc_stats=False,
        use_pbar=False,
    )
    assert np.all(_read_band(out_nd_mem) == 255)
    assert np.all(_read_band(out_nd_tile) == 255)

    out_t1 = tmp_path / "t1.tif"
    classpost.apply_sieve_filter(
        in_f, out_t1, size_threshold=1, in_memory=True, calc_stats=False, use_pbar=False
    )
    np.testing.assert_allclose(_read_band(out_t1), arr_f, equal_nan=True)

    # 3. Validation errors
    with pytest.raises(rsgispy.RSGISPyException):
        classpost.apply_sieve_filter(in_nd, out_nd_mem, size_threshold=0)
    with pytest.raises(rsgispy.RSGISPyException):
        classpost.apply_sieve_filter(in_nd, out_nd_mem, size_threshold=True)
    with pytest.raises(rsgispy.RSGISPyException):
        classpost.apply_sieve_filter(in_nd, out_nd_mem, size_threshold=4, connectivity=6)
    with pytest.raises(rsgispy.RSGISPyException):
        classpost.apply_sieve_filter(
            in_nd, out_nd_mem, size_threshold=4, connectivity=True
        )
    with pytest.raises(rsgispy.RSGISPyException):
        classpost.apply_sieve_filter(
            in_nd, out_nd_mem, size_threshold=4, block_size=0
        )
    with pytest.raises(rsgispy.RSGISPyException):
        classpost.apply_sieve_filter(tmp_path / "missing.tif", out_nd_mem, size_threshold=4)
