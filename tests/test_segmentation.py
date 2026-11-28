"""
Tests for rsgispy.segmentation.clump (clump and union_of_clumps).
"""

import os
import time

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

import rsgispy
import rsgispy.segmentation as rseg_pkg
import rsgispy.segmentation._helpers as seg_helpers
import rsgispy.segmentation.clump as rseg_clump

from conftest import write_raster


def _write_raster(
    path: str,
    arr: np.ndarray,
    origin_x: float = 0.0,
    origin_y: float = 100.0,
    pxl_size: float = 10.0,
    crs: str = "EPSG:32630",
) -> None:
    """Helper to write a 2D or 3D GeoTIFF raster."""
    return write_raster(path, arr, crs=crs, origin=(origin_x, origin_y), pixel_size=pxl_size)


def test_clump_4connectivity_nodata_and_no_pam(tmp_path):
    """Test 4-connectivity, diagonal separation, no_data_val, and no PAM files."""
    arr = np.array(
        [
            [1, 0, 1, 2, 2],
            [0, 1, 0, 2, 0],
            [3, 3, 0, 0, 4],
            [3, 0, 4, 4, 0],
            [0, 5, 5, 5, 5],
        ],
        dtype=np.uint8,
    )
    in_img = str(tmp_path / "in.tif")
    _write_raster(in_img, arr)

    # With no_data_val=0.0: diagonals do not merge, 0 pixels stay 0.
    out_nd0 = str(tmp_path / "clump_nd0.tif")
    rseg_clump.clump(
        in_img,
        out_nd0,
        gdalformat="GTIFF",
        in_memory=True,
        no_data_val=0.0,
        use_pbar=False,
    )
    expected_nd0 = np.array(
        [
            [1, 0, 2, 3, 3],
            [0, 4, 0, 3, 0],
            [5, 5, 0, 0, 6],
            [5, 0, 7, 7, 0],
            [0, 8, 8, 8, 8],
        ],
        dtype=np.uint32,
    )
    with rasterio.open(out_nd0) as src:
        assert src.dtypes == ("uint32",)
        np.testing.assert_array_equal(src.read(1), expected_nd0)
    assert not os.path.exists(f"{out_nd0}.aux.xml")

    # With no_data_val=None: 0 pixels are clumped normally (1..15).
    out_none = str(tmp_path / "clump_none.tif")
    rseg_clump.clump(
        in_img,
        out_none,
        gdalformat="GTIFF",
        in_memory=True,
        no_data_val=None,
        use_pbar=False,
    )
    expected_none = np.array(
        [
            [1, 2, 3, 4, 4],
            [5, 6, 7, 4, 8],
            [9, 9, 7, 7, 10],
            [9, 11, 12, 12, 13],
            [14, 15, 15, 15, 15],
        ],
        dtype=np.uint32,
    )
    with rasterio.open(out_none) as src:
        np.testing.assert_array_equal(src.read(1), expected_none)
    assert not os.path.exists(f"{out_none}.aux.xml")


def test_clump_in_memory_vs_tiled_and_pure_python_fallback(
    tmp_path, monkeypatch
):
    """
    Verify in_memory=True, two-pass tiled in_memory=False (crossing tile
    seams via U-shapes/spirals), and the zero-SciPy fallback produce identical
    uint32 outputs with no PAM files.
    """
    rng = np.random.default_rng(123)
    arr = np.repeat(
        np.repeat(rng.integers(0, 5, size=(10, 11), dtype=np.uint16), 4, axis=0),
        4,
        axis=1,
    )
    # Add U-shape spanning across row 16 and col 16 tile boundaries.
    arr[12:20, 14] = 99
    arr[19, 14:21] = 99
    arr[12:20, 20] = 99
    # Add nodata patch covering an entire 16x16 corner tile.
    arr[0:16, 0:16] = 0

    in_img = str(tmp_path / "grid.tif")
    _write_raster(in_img, arr)

    out_mem = str(tmp_path / "mem.tif")
    out_tiled = str(tmp_path / "tiled.tif")
    out_noscp = str(tmp_path / "noscp.tif")
    out_noscp_tiled = str(tmp_path / "noscp_tiled.tif")

    rseg_clump.clump(
        in_img,
        out_mem,
        gdalformat="GTIFF",
        in_memory=True,
        no_data_val=0.0,
        use_pbar=False,
    )
    rseg_clump.clump(
        in_img,
        out_tiled,
        gdalformat="GTIFF",
        in_memory=False,
        no_data_val=0.0,
        block_size=16,
        use_pbar=False,
    )

    monkeypatch.setattr(seg_helpers, "HAVE_SCIPY", False)
    rseg_clump.clump(
        in_img,
        out_noscp,
        gdalformat="GTIFF",
        in_memory=True,
        no_data_val=0.0,
        use_pbar=False,
    )
    rseg_clump.clump(
        in_img,
        out_noscp_tiled,
        gdalformat="GTIFF",
        in_memory=False,
        no_data_val=0.0,
        block_size=16,
        use_pbar=False,
    )

    with (
        rasterio.open(out_mem) as s_mem,
        rasterio.open(out_tiled) as s_tld,
        rasterio.open(out_noscp) as s_ns,
        rasterio.open(out_noscp_tiled) as s_nst,
    ):
        a_mem = s_mem.read(1)
        np.testing.assert_array_equal(a_mem, s_tld.read(1))
        np.testing.assert_array_equal(a_mem, s_ns.read(1))
        np.testing.assert_array_equal(a_mem, s_nst.read(1))

    for out_p in (out_mem, out_tiled, out_noscp, out_noscp_tiled):
        assert not os.path.exists(f"{out_p}.aux.xml")


def test_clump_edge_cases_and_multiband(tmp_path):
    """Test all-nodata, 1xN, Nx1, float rasters, and multi-band band-1 rule."""
    # All nodata
    nd_img = str(tmp_path / "all_nd.tif")
    _write_raster(nd_img, np.zeros((10, 12), dtype=np.uint8))
    out_nd = str(tmp_path / "out_all_nd.tif")
    rseg_clump.clump(
        nd_img,
        out_nd,
        in_memory=False,
        no_data_val=0.0,
        block_size=4,
        use_pbar=False,
    )
    with rasterio.open(out_nd) as src:
        np.testing.assert_array_equal(src.read(1), np.zeros((10, 12)))
    assert not os.path.exists(f"{out_nd}.aux.xml")

    # 1x7 row and 7x1 column
    row_img = str(tmp_path / "row.tif")
    col_img = str(tmp_path / "col.tif")
    _write_raster(row_img, np.array([[1, 1, 0, 2, 2, 2, 1]], dtype=np.uint8))
    _write_raster(
        col_img, np.array([[1], [1], [0], [2], [2], [0], [1]], dtype=np.uint8)
    )
    out_row = str(tmp_path / "out_row.tif")
    out_col = str(tmp_path / "out_col.tif")
    rseg_clump.clump(row_img, out_row, no_data_val=0.0, use_pbar=False)
    rseg_clump.clump(col_img, out_col, no_data_val=0.0, use_pbar=False)
    with rasterio.open(out_row) as src:
        np.testing.assert_array_equal(
            src.read(1), np.array([[1, 1, 0, 2, 2, 2, 3]], dtype=np.uint32)
        )
    with rasterio.open(out_col) as src:
        np.testing.assert_array_equal(
            src.read(1),
            np.array([[1], [1], [0], [2], [2], [0], [3]], dtype=np.uint32),
        )

    # Multi-band image uses band 1 only (matching rsgislib), and float32 works
    mb = np.zeros((2, 4, 4), dtype=np.float32)
    mb[0] = np.array(
        [
            [1.5, 1.5, 2.5, 2.5],
            [1.5, 1.5, 2.5, 2.5],
            [3.5, 3.5, 4.5, 4.5],
            [3.5, 3.5, 4.5, 4.5],
        ],
        dtype=np.float32,
    )
    mb[1] = np.arange(16, dtype=np.float32).reshape(4, 4)
    mb_img = str(tmp_path / "mb.tif")
    _write_raster(mb_img, mb)
    out_mb = str(tmp_path / "out_mb.tif")
    rseg_clump.clump(
        mb_img,
        out_mb,
        in_memory=True,
        no_data_val=None,
        use_pbar=False,
    )
    with rasterio.open(out_mb) as src:
        assert int(src.read(1).max()) == 4


def test_union_of_clumps_semantics_multiband_overlap_and_tiled(tmp_path):
    """
    Test union_of_clumps all-nodata rule, multi-band inputs, spatial
    intersection cropping, tiled vs in-memory parity, and no PAM files.
    """
    a = np.array([[0, 1, 1], [0, 2, 0], [3, 3, 0]], dtype=np.uint8)
    b = np.array([[0, 5, 6], [7, 5, 0], [5, 5, 0]], dtype=np.uint8)
    pa = str(tmp_path / "a.tif")
    pb = str(tmp_path / "b.tif")
    _write_raster(pa, a)
    _write_raster(pb, b)

    out_nd0 = str(tmp_path / "u_nd0.tif")
    rseg_clump.union_of_clumps(
        [pa, pb],
        out_nd0,
        gdalformat="GTIFF",
        no_data_val=0.0,
        in_memory=True,
        use_pbar=False,
    )
    expected_nd0 = np.array([[0, 1, 2], [3, 4, 0], [5, 5, 0]], dtype=np.uint32)
    with rasterio.open(out_nd0) as src:
        np.testing.assert_array_equal(src.read(1), expected_nd0)
    assert not os.path.exists(f"{out_nd0}.aux.xml")

    # Multi-band image + offset raster intersection (12x24)
    mb = np.stack(
        [
            np.repeat(
                np.repeat(np.array([[1, 2], [3, 4]], dtype=np.uint8), 12, 0),
                12,
                1,
            ),
            np.repeat(
                np.repeat(np.array([[1, 2, 1, 2]] * 4, dtype=np.uint8), 6, 0),
                6,
                1,
            ),
        ]
    )
    c = np.repeat(
        np.repeat(np.array([[10, 10], [20, 10]], dtype=np.uint16), 12, 0),
        12,
        1,
    )
    pmb = str(tmp_path / "umb.tif")
    pc = str(tmp_path / "uc.tif")
    _write_raster(pmb, mb, origin_x=0.0, origin_y=240.0, pxl_size=10.0)
    _write_raster(pc, c, origin_x=120.0, origin_y=240.0, pxl_size=10.0)

    out_u_mem = str(tmp_path / "u_mem.tif")
    out_u_tld = str(tmp_path / "u_tld.tif")
    rseg_clump.union_of_clumps(
        [pmb, pc],
        out_u_mem,
        no_data_val=None,
        in_memory=True,
        use_pbar=False,
    )
    rseg_clump.union_of_clumps(
        [pmb, pc],
        out_u_tld,
        no_data_val=None,
        in_memory=False,
        block_size=5,
        use_pbar=False,
    )
    with rasterio.open(out_u_mem) as s1, rasterio.open(out_u_tld) as s2:
        assert s1.width == 12 and s1.height == 24
        np.testing.assert_array_equal(s1.read(1), s2.read(1))
    assert not os.path.exists(f"{out_u_mem}.aux.xml")
    assert not os.path.exists(f"{out_u_tld}.aux.xml")


def test_segmentation_validation_and_performance(tmp_path):
    """Verify input validation errors and fast O(N) runtime on 1024x1024."""
    with pytest.raises(rsgispy.RSGISPyException):
        rseg_clump.clump(
            str(tmp_path / "missing.tif"), str(tmp_path / "out.tif")
        )
    with pytest.raises(rsgispy.RSGISPyException):
        rseg_clump.union_of_clumps([], str(tmp_path / "out.tif"))

    rng = np.random.default_rng(99)
    big = np.repeat(
        np.repeat(
            rng.integers(1, 20, size=(128, 128), dtype=np.uint16), 8, axis=0
        ),
        8,
        axis=1,
    )
    big_img = str(tmp_path / "big.tif")
    _write_raster(big_img, big)
    with pytest.raises(rsgispy.RSGISPyException):
        rseg_clump.clump(big_img, str(tmp_path / "bad_blk.tif"), block_size=0)

    out_big = str(tmp_path / "big_clump.tif")
    t0 = time.perf_counter()
    rseg_clump.clump(
        big_img, out_big, in_memory=True, no_data_val=0.0, use_pbar=False
    )
    elapsed = time.perf_counter() - t0
    assert elapsed < 2.0
    with rasterio.open(out_big) as src:
        assert src.read(1).max() > 1000


def test_eliminate_small_segments_masks_reference_nodata(tmp_path):
    """The reference-image nodata is excluded from the merge statistics."""
    from rsgispy.segmentation.utils import (  # pylint: disable=import-outside-toplevel
        eliminate_small_segments_array,
    )

    # Segment 2 (one pixel, value 10) touches segment 1 (mean 10 over its
    # valid pixels, but half of it is nodata -9999) and segment 3 (mean 50).
    labels = np.array([[1, 1, 2, 3, 3], [1, 1, 3, 3, 3]], dtype=np.int32)
    ref = np.array([[-9999.0, -9999.0, 10.0, 50.0, 50.0],
                    [10.0, 10.0, 50.0, 50.0, 50.0]])
    out = eliminate_small_segments_array(labels, ref, min_pixels=2,
                                         data_no_data_val=-9999.0)
    assert out[0, 2] == out[0, 0]  # merged with segment 1, the closer mean
    # Without masking, segment 1's mean is about -4995 and it merges into 3.
    out_raw = eliminate_small_segments_array(labels, ref, min_pixels=2)
    assert out_raw[0, 2] == out_raw[0, 3]

    # The file function takes the reference nodata from its header.
    lbl_path = str(tmp_path / "lbl.tif")
    ref_path = str(tmp_path / "ref.tif")
    for path, arr, dtype, nd in ((lbl_path, labels, "uint32", 0),
                                 (ref_path, ref, "float32", -9999.0)):
        with rasterio.open(path, "w", driver="GTiff", width=5, height=2, count=1,
                           dtype=dtype, crs="EPSG:32630", nodata=nd,
                           transform=from_origin(0.0, 20.0, 10.0, 10.0)) as dst:
            dst.write(arr.astype(dtype), 1)
    out_path = str(tmp_path / "out.tif")
    rseg_pkg.eliminate_small_segments(lbl_path, out_path, ref_path, 2,
                                      calc_stats=False, use_pbar=False)
    with rasterio.open(out_path) as ds:
        res = ds.read(1)
    assert res[0, 2] == res[0, 0]


def test_clump_id_overflow_raises(tmp_path):
    # 20x20 grid with unique pixel value per cell => 400 clumps (> uint8 max 255)
    vals = np.arange(1, 401, dtype=np.uint16).reshape(20, 20)
    in_img = write_raster(tmp_path / "clump_in.tif", vals, origin=(500000.0, 5800000.0), nodata=0)

    # In-memory overflow
    with pytest.raises(
        rsgispy.RSGISPyException, match="exceeds maximum representable value"
    ):
        rseg_clump.clump(
            input_img=in_img,
            output_img=str(tmp_path / "clump_mem.tif"),
            in_memory=True,
            datatype=rsgispy.TYPE_8UINT,
            calc_stats=False,
            use_pbar=False,
        )

    # Tiled overflow (block_size=8 forces multi-tile path)
    with pytest.raises(
        rsgispy.RSGISPyException, match="exceeds maximum representable value"
    ):
        rseg_clump.clump(
            input_img=in_img,
            output_img=str(tmp_path / "clump_tiled.tif"),
            in_memory=False,
            block_size=8,
            datatype=rsgispy.TYPE_8UINT,
            calc_stats=False,
            use_pbar=False,
        )
