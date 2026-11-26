"""
Tests for the vectorised per-pixel percentile calculation
(:func:`rsgispy.tools.stats.compute_nanpercentiles`) and the raster
percentile composites in :mod:`rsgispy.imagecalc.percentiles` (band names,
no-data value, statistics and block-wise processing).
"""

import time
import warnings

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

import rsgispy
import rsgispy.imagecalc.percentiles as img_pct
import rsgispy.imageprocess
import rsgispy.tools.stats as rstats

PCTS = [0.0, 100.0, 50.0, 10.0, 90.0, 2.5, 33.3, 66.7, 99.9, 12.345]


def _np_ref(stack, percentiles):
    """The reference result: NumPy's (slow) nanpercentile along axis 0."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanpercentile(stack, percentiles, axis=0)


def _random_stack(shape, dtype, nan_frac, seed):
    rng = np.random.default_rng(seed)
    arr = rng.normal(100.0, 30.0, size=shape)
    if np.issubdtype(np.dtype(dtype), np.integer):
        return np.round(arr).astype(dtype)
    arr = arr.astype(dtype)
    arr[rng.random(shape) < nan_frac] = np.nan
    if shape[0] > 1 and arr.ndim == 3:
        arr[:, 0, 0] = np.nan  # all NaN
        arr[:, 0, 1] = np.nan
        arr[0, 0, 1] = 42.0  # n == 1
        arr[:, 1, 0] = 7.0  # all identical
    return arr


@pytest.mark.parametrize("n_layers", [1, 2, 3, 7, 12])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("nan_frac", [0.0, 0.3, 0.8])
def test_compute_nanpercentiles_matches_numpy(n_layers, dtype, nan_frac):
    stack = _random_stack((n_layers, 23, 19), dtype, nan_frac, seed=n_layers)
    expected = _np_ref(stack, PCTS)
    orig = stack.copy()
    result = rstats.compute_nanpercentiles(stack, PCTS)
    assert result.shape == (len(PCTS), 23, 19)
    np.testing.assert_array_equal(np.isnan(result), np.isnan(expected))
    # compute_nanpercentiles interpolates in float64. Some NumPy versions
    # interpolate float32 input in float32, so allow float32 rounding there.
    tol = 1e-12 if dtype == np.float64 else 1e-6
    np.testing.assert_allclose(result, expected, rtol=tol, atol=tol, equal_nan=True)
    # The input is not modified
    np.testing.assert_array_equal(stack, orig)


def test_compute_nanpercentiles_special_pixels():
    stack = _random_stack((5, 4, 4), np.float64, 0.0, seed=1)
    result = rstats.compute_nanpercentiles(stack, [0, 50, 100])
    assert np.all(np.isnan(result[:, 0, 0]))  # n == 0 gives NaN
    assert result[:, 0, 1] == pytest.approx([42.0, 42.0, 42.0])  # n == 1
    assert result[:, 1, 0] == pytest.approx([7.0, 7.0, 7.0])
    # The percentiles of a pixel are ordered
    valid = ~np.isnan(result[0])
    assert np.all(result[0][valid] <= result[1][valid])
    assert np.all(result[1][valid] <= result[2][valid])


@pytest.mark.parametrize("dtype", [np.int16, np.uint16, np.int32])
def test_compute_nanpercentiles_integer_input(dtype):
    stack = np.abs(_random_stack((6, 9, 11), dtype, 0.0, seed=4))
    expected = _np_ref(stack.astype(np.float64), PCTS)
    result = rstats.compute_nanpercentiles(stack, PCTS)
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int16])
def test_compute_nanpercentiles_no_data(dtype):
    rng = np.random.default_rng(7)
    stack = rng.integers(1, 500, size=(8, 10, 12)).astype(dtype)
    stack[rng.random(stack.shape) < 0.4] = -99
    stack[:, 2, 3] = -99  # all no data
    orig = stack.copy()
    with_nan = stack.astype(np.float64)
    with_nan[with_nan == -99] = np.nan
    expected = _np_ref(with_nan, PCTS)
    expected[np.isnan(expected)] = -99.0
    result = rstats.compute_nanpercentiles(stack, PCTS, no_data_val=-99)
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)
    assert np.all(result[:, 2, 3] == -99.0)
    np.testing.assert_array_equal(stack, orig)


def test_compute_nanpercentiles_no_data_float32_fraction():
    """A fractional no-data value is matched in the array's own precision."""
    stack = np.full((3, 2, 2), 0.1, dtype=np.float32)
    stack[0] = 5.0
    result = rstats.compute_nanpercentiles(stack, [50.0], no_data_val=0.1)
    assert result[0] == pytest.approx(np.full((2, 2), 5.0))


def test_compute_nanpercentiles_shapes():
    stack = _random_stack((4, 6, 5), np.float64, 0.2, seed=3)
    # A scalar percentile on a 3D stack gives (1, height, width)
    res = rstats.compute_nanpercentiles(stack, 50.0)
    assert res.shape == (1, 6, 5)
    np.testing.assert_allclose(res[0], _np_ref(stack, 50.0), equal_nan=True)
    # 1D input
    vec = np.array([3.0, np.nan, 1.0, 2.0])
    assert rstats.compute_nanpercentiles(vec, [0, 50, 100]) == pytest.approx([1.0, 2.0, 3.0])
    assert float(rstats.compute_nanpercentiles(vec, 50)) == pytest.approx(2.0)


def test_compute_nanpercentiles_is_fast():
    """np.nanpercentile(axis=0) takes several seconds for this stack."""
    stack = _random_stack((10, 400, 400), np.float32, 0.2, seed=5)
    t_start = time.perf_counter()
    res = rstats.compute_nanpercentiles(stack, [10, 25, 50, 75, 90])
    elapsed = time.perf_counter() - t_start
    assert res.shape == (5, 400, 400)
    assert elapsed < 1.0


# ---------------------------------------------------------------------------
# Raster composites
# ---------------------------------------------------------------------------


def _write_img(path, arr, dtype="float32", nodata=None, names=None):
    count, height, width = arr.shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=dtype,
        crs="EPSG:32630",
        transform=from_origin(500000.0, 5700000.0, 10.0, 10.0),
        nodata=nodata,
    ) as dst:
        dst.write(arr.astype(dtype))
        if names is not None:
            dst.descriptions = tuple(names)
    return str(path)


def _make_dates(tmp_path, n_dates=5, n_bands=2, shape=(37, 41), dtype="float32", names=None):
    rng = np.random.default_rng(11)
    imgs = []
    arrs = []
    for idx in range(n_dates):
        arr = rng.integers(1, 1000, size=(n_bands,) + shape).astype(np.float64)
        arr[:, rng.random(shape) < 0.3] = 0.0  # cloud: no data in every band
        arr[:, :3, :3] = 0.0  # no data on every date
        arrs.append(arr)
        imgs.append(_write_img(tmp_path / f"d{idx}.tif", arr, dtype=dtype, nodata=0, names=names))
    return imgs, np.stack(arrs)  # (dates, bands, rows, cols)


@pytest.fixture
def small_blocks():
    rsgispy.imageprocess.set_default_block_size(16)
    yield 16
    rsgispy.imageprocess.reset_default_block_size()


def test_calc_imgs_pxl_percentiles_output(tmp_path, small_blocks):
    imgs, arrs = _make_dates(tmp_path, n_bands=1)
    out_img = str(tmp_path / "pcts.tif")
    pcts = [10, 50, 90, 2.5]
    img_pct.calc_imgs_pxl_percentiles(imgs, out_img, pcts, no_data_val=0, use_pbar=False)

    stack = arrs[:, 0].copy()
    stack[stack == 0] = np.nan
    expected = _np_ref(stack, pcts)
    expected[np.isnan(expected)] = 0.0
    with rasterio.open(out_img) as src:
        assert src.count == len(pcts)
        assert src.descriptions == ("p10", "p50", "p90", "p2.5")
        assert src.nodata == 0
        res = src.read()
        stats_tags = src.tags(1)
    np.testing.assert_allclose(res, expected.astype(np.float32), rtol=1e-6)
    assert np.all(res[:, :3, :3] == 0)
    # Statistics are calculated by default and ignore the no-data value
    assert "STATISTICS_MEAN" in stats_tags
    assert float(stats_tags["STATISTICS_MINIMUM"]) > 0


def test_calc_img_band_pxl_percentiles_output(tmp_path, small_blocks):
    imgs, arrs = _make_dates(tmp_path, n_bands=2, names=("Blue", "Green"))
    out_img = str(tmp_path / "band_pcts.tif")
    img_pct.calc_img_band_pxl_percentiles(imgs, out_img, [25, 75], no_data_val=0, use_pbar=False)

    with rasterio.open(out_img) as src:
        assert src.count == 4
        assert src.descriptions == ("Blue_p25", "Blue_p75", "Green_p25", "Green_p75")
        assert src.nodata == 0
        res = src.read()
        assert "STATISTICS_MEAN" in src.tags(3)
    for b_idx in range(2):
        stack = arrs[:, b_idx].copy()
        stack[stack == 0] = np.nan
        expected = _np_ref(stack, [25, 75])
        expected[np.isnan(expected)] = 0.0
        np.testing.assert_allclose(res[2 * b_idx:2 * b_idx + 2], expected.astype(np.float32), rtol=1e-6)


def test_pxl_percentiles_default_band_names_and_no_stats(tmp_path):
    imgs, _ = _make_dates(tmp_path, n_dates=3, n_bands=2, shape=(12, 10))
    out_img = str(tmp_path / "band_pcts.tif")
    img_pct.calc_img_band_pxl_percentiles(
        imgs, out_img, [50], no_data_val=0, calc_stats=False, use_pbar=False
    )
    with rasterio.open(out_img) as src:
        assert src.descriptions == ("b1_p50", "b2_p50")
        assert src.nodata == 0
        assert "STATISTICS_MEAN" not in src.tags(1)

    out_img2 = str(tmp_path / "pcts.tif")
    img_pct.calc_imgs_pxl_percentiles(
        imgs, out_img2, [50], no_data_val=0, calc_stats=False, use_pbar=False
    )
    with rasterio.open(out_img2) as src:
        assert src.descriptions == ("p50",)
        assert "STATISTICS_MEAN" not in src.tags(1)


def test_pxl_percentiles_float_no_data(tmp_path):
    """A float no-data value (e.g. NDVI -999) is written to the output."""
    rng = np.random.default_rng(2)
    imgs = []
    for idx in range(4):
        arr = rng.uniform(-1, 1, size=(1, 9, 8))
        arr[0, :2, :2] = -999.0
        imgs.append(_write_img(tmp_path / f"ndvi{idx}.tif", arr, nodata=-999))
    out_img = str(tmp_path / "ndvi_pcts.tif")
    img_pct.calc_imgs_pxl_percentiles(imgs, out_img, [10, 90], no_data_val=-999, use_pbar=False)
    with rasterio.open(out_img) as src:
        assert src.nodata == -999
        res = src.read()
        assert float(src.tags(1)["STATISTICS_MINIMUM"]) >= -1.0
    assert np.all(res[:, :2, :2] == -999)
    assert np.all(np.abs(res[:, 2:, 2:]) <= 1.0)


def test_pxl_percentiles_no_data_not_representable(tmp_path):
    imgs, _ = _make_dates(tmp_path, n_dates=2, n_bands=1, shape=(6, 6), dtype="uint16")
    with pytest.raises(rsgispy.RSGISPyException):
        img_pct.calc_imgs_pxl_percentiles(
            imgs, str(tmp_path / "out.tif"), [50], no_data_val=-999, use_pbar=False
        )


def test_pxl_percentiles_are_block_wise(tmp_path, small_blocks, monkeypatch):
    """The percentiles are calculated block by block, not on the whole image."""
    imgs, _ = _make_dates(tmp_path, n_dates=3, n_bands=2, shape=(40, 37))
    shapes = []
    orig = img_pct.compute_nanpercentiles

    def _record(stack_arr, *args, **kwargs):
        shapes.append(stack_arr.shape)
        return orig(stack_arr, *args, **kwargs)

    monkeypatch.setattr(img_pct, "compute_nanpercentiles", _record)
    img_pct.calc_imgs_pxl_percentiles(
        imgs, str(tmp_path / "a.tif"), [50], no_data_val=0, calc_stats=False, use_pbar=False
    )
    assert len(shapes) == 9  # 3 x 3 blocks of at most 16 x 16 pixels
    assert all(shp[1] <= 16 and shp[2] <= 16 for shp in shapes)
    assert shapes[0][0] == 6  # 3 images x 2 bands pooled

    shapes.clear()
    img_pct.calc_img_band_pxl_percentiles(
        imgs, str(tmp_path / "b.tif"), [50], no_data_val=0, calc_stats=False, use_pbar=False
    )
    assert len(shapes) == 18  # 9 blocks x 2 bands
    assert all(shp == (3,) + shp[1:] and shp[1] <= 16 and shp[2] <= 16 for shp in shapes)
