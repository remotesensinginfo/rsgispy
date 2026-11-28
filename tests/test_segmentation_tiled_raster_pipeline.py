"""
Direct tests of :func:`rsgispy.segmentation.tiling.run_tiled_raster_pipeline`
with a trivial worker (connected regions of equal value), so the expected
segments are known exactly: the tiled, seam-stitched result must match the
single-pass result, IDs must be ``1..N``, no data must stay 0 and
``n_jobs > 1`` must give the same labels as ``n_jobs=1``.
"""

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from scipy import ndimage

from rsgispy.segmentation.tiling import run_tiled_raster_pipeline

H, W = 40, 48


def equal_value_worker(halo_arr, halo_valid, tile, params):
    """Label connected regions of equal band 1 value in the tile core."""
    del params
    core = halo_arr[0, tile.core_row_slice, tile.core_col_slice]
    valid = halo_valid[tile.core_row_slice, tile.core_col_slice]
    labels = np.zeros(core.shape, dtype=np.int64)
    n_next = 0
    for val in np.unique(core[valid]):
        lab, n_lab = ndimage.label((core == val) & valid)
        labels[lab > 0] = lab[lab > 0] + n_next
        n_next += n_lab
    return labels, None


def _scene():
    """Three blocks whose edges do not fall on the 16 pixel tile edges, and
    a no data corner (value 0)."""
    arr = np.full((H, W), 10, dtype=np.uint8)
    arr[:, 21:] = 50
    arr[27:, 21:] = 90
    arr[:5, :5] = 0
    return arr


@pytest.fixture(name="img")
def fixture_img(tmp_path):
    path = str(tmp_path / "in.tif")
    with rasterio.open(
        path, "w", driver="GTiff", width=W, height=H, count=1, dtype="uint8",
        crs="EPSG:32630", transform=from_origin(400000.0, 6000000.0, 10.0, 10.0),
        nodata=0,
    ) as dst:
        dst.write(_scene(), 1)
    return path


def _run(img, out, **kwargs):
    opts = dict(
        bands=[1], worker_fn=equal_value_worker, params={}, min_pixels=1,
        calc_stats=False, use_pbar=False,
    )
    opts.update(kwargs)
    run_tiled_raster_pipeline(img, out, **opts)
    with rasterio.open(out) as src:
        assert src.nodata == 0
        assert (src.width, src.height) == (W, H)
        return src.read(1)


def _expected_regions():
    arr = _scene()
    return [(arr == v) for v in (10, 50, 90)]


def _assert_partition(labels):
    arr = _scene()
    assert np.all(labels[arr == 0] == 0)
    assert np.all(labels[arr > 0] > 0)
    ids = np.unique(labels[labels > 0])
    np.testing.assert_array_equal(ids, np.arange(1, len(ids) + 1))  # 1..N
    return ids


def test_single_pass_matches_regions(img, tmp_path):
    labels = _run(img, str(tmp_path / "single.tif"))  # one tile (small image)
    ids = _assert_partition(labels)
    assert len(ids) == 3
    for region in _expected_regions():
        assert len(np.unique(labels[region])) == 1


def test_tiled_stitched_matches_single_pass(img, tmp_path):
    single = _run(img, str(tmp_path / "single.tif"))
    tiled = _run(img, str(tmp_path / "tiled.tif"), tile_size=16, overlap=2,
                 stitch_seams=True, seam_mode="spectral", seam_threshold=1.0)
    _assert_partition(tiled)
    # Same partition (labels may be numbered differently).
    pairs = set(zip(single.ravel().tolist(), tiled.ravel().tolist()))
    assert len(pairs) == len(np.unique(single)) == len(np.unique(tiled))


def test_tiled_without_stitching_keeps_tile_pieces(img, tmp_path):
    labels = _run(img, str(tmp_path / "raw.tif"), tile_size=16, stitch_seams=False)
    ids = _assert_partition(labels)
    # Each region is split at the tile edges (rows/cols 16 and 32) and no
    # segment crosses a tile edge.
    assert len(ids) > 3
    tile_id = (np.arange(H)[:, None] // 16) * 10 + (np.arange(W)[None, :] // 16)
    for seg_id in ids:
        assert len(np.unique(tile_id[labels == seg_id])) == 1
        # ... and stays within one region.
        assert len(np.unique(_scene()[labels == seg_id])) == 1
    # The 10-valued block spans 3 x 2 tiles -> 6 pieces.
    assert len(np.unique(labels[_scene() == 10])) == 6


def test_small_segments_eliminated(img, tmp_path):
    # Without stitching the 90 block's pieces in row tile 1 (rows 27-31) are
    # 5 rows tall; min_pixels larger than them merges them into a neighbour.
    raw = _run(img, str(tmp_path / "raw.tif"), tile_size=16, stitch_seams=False)
    merged = _run(img, str(tmp_path / "merged.tif"), tile_size=16,
                  stitch_seams=False, min_pixels=60)
    _assert_partition(merged)
    assert len(np.unique(merged)) < len(np.unique(raw))
    counts = np.bincount(merged.ravel())[1:]
    assert counts.min() >= 60


def test_n_jobs_identical(img, tmp_path):
    kw = dict(tile_size=16, overlap=2, seam_threshold=1.0)
    one = _run(img, str(tmp_path / "j1.tif"), n_jobs=1, **kw)
    two = _run(img, str(tmp_path / "j2.tif"), n_jobs=2, **kw)
    np.testing.assert_array_equal(one, two)


def test_stats_written(img, tmp_path):
    out = str(tmp_path / "stats.tif")
    _run(img, out, tile_size=16, calc_stats=True)
    with rasterio.open(out) as src:
        tags = src.tags(1)
        assert src.dtypes[0].startswith("uint")
    # Statistics over the 3 stitched segments, excluding no data (0).
    assert tags["STATISTICS_EXCLUDEDVALUES"] == "0"
    assert float(tags["STATISTICS_MINIMUM"]) == 1.0
    assert float(tags["STATISTICS_MAXIMUM"]) == 3.0
    hist = [int(v) for v in tags["STATISTICS_HISTOBINVALUES"].strip("|").split("|")]
    assert sum(hist) == H * W - 25  # all pixels but the 5 x 5 no data corner
