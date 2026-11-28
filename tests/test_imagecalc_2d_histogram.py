"""
Tests for :func:`rsgispy.imagecalc.stats.get_img_2d_histogram`.
"""

import numpy as np
import pytest

import rsgispy
import rsgispy.imagecalc.stats as ic_stats

from conftest import write_raster


def _write_raster(path, data, nodata=None, dtype="float32", origin=(0.0, 1000.0)):
    """Write a ``(bands, height, width)`` array to a GeoTIFF."""
    return write_raster(path, data, dtype=dtype, nodata=nodata, origin=origin)


@pytest.fixture(name="hist_imgs")
def fixture_hist_imgs(tmp_path):
    """A 3-band float image with no data and NaN, a second image and a mask."""
    rng = np.random.default_rng(42)
    height, width = 37, 53
    data = rng.normal(100.0, 25.0, size=(3, height, width)).round(1)
    data[0, 0:4, 0:6] = -999.0
    data[1, 10:12, 20:30] = -999.0
    data[2, 5, 5] = np.nan
    data[0, 30, 40] = np.nan
    img = _write_raster(tmp_path / "img.tif", data, nodata=-999.0)

    y_data = rng.uniform(0.0, 1.0, size=(1, height, width))
    y_data[0, 15:18, 2:9] = -1.0
    y_img = _write_raster(tmp_path / "y_img.tif", y_data, nodata=-1.0)

    msk = (rng.uniform(size=(1, height, width)) > 0.3).astype(np.uint8)
    msk_img = _write_raster(tmp_path / "msk.tif", msk, dtype="uint8")
    return {
        "img": img,
        "data": data,
        "y_img": y_img,
        "y_data": y_data,
        "msk_img": msk_img,
        "msk": msk,
    }


def _valid(*arrs, nodata=(-999.0, -999.0)):
    """Boolean mask of the pixels where every array is finite and not no data."""
    vld = np.ones(arrs[0].shape, dtype=bool)
    for arr, nd_val in zip(arrs, nodata):
        vld &= np.isfinite(arr) & (arr != nd_val)
    return vld


def test_2d_hist_matches_numpy_auto_range(hist_imgs):
    """Without ranges the min/max of the valid pixels are used."""
    x_arr = hist_imgs["data"][0].astype(np.float32).astype(np.float64)
    y_arr = hist_imgs["data"][1].astype(np.float32).astype(np.float64)
    vld = _valid(x_arr, y_arr)
    xv, yv = x_arr[vld], y_arr[vld]
    ref = np.histogram2d(
        xv, yv, bins=32, range=[[xv.min(), xv.max()], [yv.min(), yv.max()]]
    )
    out = ic_stats.get_img_2d_histogram(
        hist_imgs["img"], 1, 2, bins=32, use_pbar=False
    )
    assert out[0].dtype == np.float64
    assert out[0].shape == (32, 32)
    np.testing.assert_array_equal(out[0], ref[0])
    np.testing.assert_array_equal(out[1], ref[1])
    np.testing.assert_array_equal(out[2], ref[2])
    assert out[0].sum() == vld.sum()


def test_2d_hist_matches_numpy_with_ranges(hist_imgs):
    """Pixels outside the ranges are not counted; bins can differ per axis."""
    x_arr = hist_imgs["data"][0].astype(np.float32).astype(np.float64)
    y_arr = hist_imgs["data"][2].astype(np.float32).astype(np.float64)
    vld = _valid(x_arr, y_arr)
    x_rng, y_rng = (60.0, 140.0), (70.0, 120.0)
    ref = np.histogram2d(
        x_arr[vld], y_arr[vld], bins=[20, 10], range=[list(x_rng), list(y_rng)]
    )
    out = ic_stats.get_img_2d_histogram(
        hist_imgs["img"], 1, 3, bins=[20, 10], x_range=x_rng, y_range=y_rng,
        use_pbar=False,
    )
    for got, exp in zip(out, ref):
        np.testing.assert_array_equal(got, exp)
    assert out[0].shape == (20, 10)


def test_2d_hist_one_range_given(hist_imgs):
    """The missing range comes from the valid pixels inside the given one."""
    x_arr = hist_imgs["data"][0].astype(np.float32).astype(np.float64)
    y_arr = hist_imgs["data"][1].astype(np.float32).astype(np.float64)
    vld = _valid(x_arr, y_arr)
    x_rng = (80.0, 120.0)
    vld &= (x_arr >= x_rng[0]) & (x_arr <= x_rng[1])
    yv = y_arr[vld]
    ref = np.histogram2d(
        x_arr[vld], yv, bins=16, range=[list(x_rng), [yv.min(), yv.max()]]
    )
    out = ic_stats.get_img_2d_histogram(
        hist_imgs["img"], 1, 2, bins=16, x_range=x_rng, use_pbar=False
    )
    for got, exp in zip(out, ref):
        np.testing.assert_array_equal(got, exp)


def test_2d_hist_mask_second_image_and_nodata(hist_imgs):
    """A second image (its own header no data) and a mask."""
    x_arr = hist_imgs["data"][0].astype(np.float32).astype(np.float64)
    y_arr = hist_imgs["y_data"][0].astype(np.float32).astype(np.float64)
    vld = _valid(x_arr, y_arr, nodata=(-999.0, -1.0)) & (hist_imgs["msk"][0] == 1)
    x_rng, y_rng = (0.0, 200.0), (0.0, 1.0)
    ref = np.histogram2d(
        x_arr[vld], y_arr[vld], bins=25, range=[list(x_rng), list(y_rng)]
    )
    out = ic_stats.get_img_2d_histogram(
        hist_imgs["img"], 1, 1, y_img=hist_imgs["y_img"], bins=25,
        x_range=x_rng, y_range=y_rng, in_msk_img=hist_imgs["msk_img"],
        use_pbar=False,
    )
    for got, exp in zip(out, ref):
        np.testing.assert_array_equal(got, exp)

    # Mask value 0 counts the other pixels.
    vld0 = _valid(x_arr, y_arr, nodata=(-999.0, -1.0)) & (hist_imgs["msk"][0] == 0)
    ref0 = np.histogram2d(
        x_arr[vld0], y_arr[vld0], bins=25, range=[list(x_rng), list(y_rng)]
    )
    out0 = ic_stats.get_img_2d_histogram(
        hist_imgs["img"], 1, 1, y_img=hist_imgs["y_img"], bins=25,
        x_range=x_rng, y_range=y_rng, in_msk_img=hist_imgs["msk_img"],
        img_msk_val=0, use_pbar=False,
    )
    np.testing.assert_array_equal(out0[0], ref0[0])


def test_2d_hist_explicit_nodata(tmp_path):
    """``no_data_val`` excludes pixels of an image without a header no data."""
    data = np.array(
        [[[0, 1, 2, 3], [4, 5, 0, 7]], [[0, 2, 2, 0], [1, 1, 3, 3]]], dtype=np.uint16
    )
    img = _write_raster(tmp_path / "nd.tif", data, dtype="uint16")
    x_arr = data[0].astype(np.float64)
    y_arr = data[1].astype(np.float64)
    vld = (x_arr != 0) & (y_arr != 0)
    ref = np.histogram2d(
        x_arr[vld], y_arr[vld], bins=4, range=[[0.0, 8.0], [0.0, 4.0]]
    )
    out = ic_stats.get_img_2d_histogram(
        img, 1, 2, bins=4, x_range=(0, 8), y_range=(0, 4), no_data_val=0,
        use_pbar=False,
    )
    np.testing.assert_array_equal(out[0], ref[0])
    assert out[0].sum() == 5

    # Without no data every pixel is counted.
    out_all = ic_stats.get_img_2d_histogram(
        img, 1, 2, bins=4, x_range=(0, 8), y_range=(0, 4), use_pbar=False
    )
    assert out_all[0].sum() == 8


def test_2d_hist_density(hist_imgs):
    """``density=True`` matches numpy's density."""
    x_arr = hist_imgs["data"][0].astype(np.float32).astype(np.float64)
    y_arr = hist_imgs["data"][1].astype(np.float32).astype(np.float64)
    vld = _valid(x_arr, y_arr)
    rng = [[0.0, 200.0], [0.0, 200.0]]
    ref = np.histogram2d(x_arr[vld], y_arr[vld], bins=20, range=rng, density=True)
    out = ic_stats.get_img_2d_histogram(
        hist_imgs["img"], 1, 2, bins=20, x_range=(0, 200), y_range=(0, 200),
        density=True, use_pbar=False,
    )
    np.testing.assert_allclose(out[0], ref[0], rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("block_size", [1, 7, 16, 64, 1024])
def test_2d_hist_block_size_invariant(hist_imgs, block_size):
    """The histogram is identical for any block size."""
    ref = ic_stats.get_img_2d_histogram(
        hist_imgs["img"], 1, 2, bins=40, in_msk_img=hist_imgs["msk_img"],
        block_size=4096, use_pbar=False,
    )
    out = ic_stats.get_img_2d_histogram(
        hist_imgs["img"], 1, 2, bins=40, in_msk_img=hist_imgs["msk_img"],
        block_size=block_size, use_pbar=False,
    )
    for got, exp in zip(out, ref):
        np.testing.assert_array_equal(got, exp)


def test_2d_hist_errors(hist_imgs, tmp_path):
    """Grid mismatch, bad bands, bad ranges and no valid pixels raise."""
    shifted = _write_raster(
        tmp_path / "shift.tif", hist_imgs["y_data"], nodata=-1.0, origin=(10.0, 1000.0)
    )
    with pytest.raises(rsgispy.RSGISPyException, match="same pixel grid"):
        ic_stats.get_img_2d_histogram(
            hist_imgs["img"], 1, 1, y_img=shifted, use_pbar=False
        )
    small_msk = _write_raster(
        tmp_path / "small_msk.tif", np.ones((1, 5, 5), dtype=np.uint8), dtype="uint8"
    )
    with pytest.raises(rsgispy.RSGISPyException, match="same pixel grid"):
        ic_stats.get_img_2d_histogram(
            hist_imgs["img"], 1, 2, in_msk_img=small_msk, use_pbar=False
        )
    with pytest.raises(rsgispy.RSGISPyException):
        ic_stats.get_img_2d_histogram(hist_imgs["img"], 1, 5, use_pbar=False)
    with pytest.raises(rsgispy.RSGISPyException, match="x_range"):
        ic_stats.get_img_2d_histogram(
            hist_imgs["img"], 1, 2, x_range=(5.0, 1.0), use_pbar=False
        )
    all_nd = _write_raster(
        tmp_path / "all_nd.tif", np.full((2, 4, 4), -999.0), nodata=-999.0
    )
    with pytest.raises(rsgispy.RSGISPyException, match="No valid pixels"):
        ic_stats.get_img_2d_histogram(all_nd, 1, 2, use_pbar=False)
    # With both ranges given, an image without valid pixels gives zero counts.
    hist, _, _ = ic_stats.get_img_2d_histogram(
        all_nd, 1, 2, bins=3, x_range=(0, 1), y_range=(0, 1), use_pbar=False
    )
    assert hist.shape == (3, 3) and hist.sum() == 0
