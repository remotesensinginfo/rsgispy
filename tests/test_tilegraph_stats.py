"""
Tests for the tile-graph statistics sinks (:class:`StatsSink`,
:class:`ZonalStatsSink`), compared with NumPy on the whole arrays.
"""

import os

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

import rsgispy
from rsgispy.imageprocess import (
    ArraySource,
    MemorySink,
    RasterSource,
    RasterStats,
    StatsSink,
    TileGraph,
    ZonalStatsSink,
)
from rsgispy.imagefilter.stats import mean_filter_node

BLOCK_SIZES = [7, 16, 64, 256]


def _make_values(seed=3, shape=(3, 53, 47)):
    rng = np.random.default_rng(seed)
    arr = rng.normal(100.0, 25.0, size=shape)
    arr[0, 5:9, 3:30] = -9999.0
    arr[1, 10:12, :] = np.nan
    arr[2, :, 40:] = -9999.0
    return arr


def _np_stats(arr, nodata, ddof=0):
    out = []
    for band in arr:
        vals = band[np.isfinite(band)]
        if nodata is not None:
            vals = vals[vals != nodata]
        out.append(
            (
                vals.size,
                vals.min() if vals.size else np.nan,
                vals.max() if vals.size else np.nan,
                vals.mean() if vals.size else np.nan,
                vals.std(ddof=ddof) if vals.size > ddof else np.nan,
                vals,
            )
        )
    return out


@pytest.mark.parametrize("block_size", BLOCK_SIZES)
def test_stats_sink_matches_numpy_with_nodata(block_size):
    arr = _make_values()
    src = ArraySource(arr, nodata=-9999.0, band_names=["a", "b", "c"])
    res = TileGraph(
        StatsSink(src, bins=12, hist_range=(0.0, 200.0), name="st"),
        block_size=block_size,
    ).run(use_pbar=False)
    stats = res["st"]
    assert isinstance(stats, RasterStats)
    assert stats.band_names == ["a", "b", "c"]
    assert stats.nodata == -9999.0
    for b_idx, (cnt, vmin, vmax, vmean, vstd, vals) in enumerate(
        _np_stats(arr, -9999.0)
    ):
        assert stats.count[b_idx] == cnt
        assert stats.min[b_idx] == vmin
        assert stats.max[b_idx] == vmax
        np.testing.assert_allclose(stats.mean[b_idx], vmean, rtol=1e-12)
        np.testing.assert_allclose(stats.std[b_idx], vstd, rtol=1e-10)
        hist, edges = np.histogram(vals, bins=12, range=(0.0, 200.0))
        np.testing.assert_array_equal(stats.hist[b_idx], hist)
        np.testing.assert_array_equal(stats.bin_edges, edges)
    df = stats.to_dataframe()
    assert list(df.columns) == ["count", "min", "max", "mean", "std"]
    assert list(df.index) == ["a", "b", "c"]


def test_stats_sink_explicit_nodata_edges_ddof_and_nan_nodata():
    arr = _make_values(seed=11)
    # No metadata no-data: only NaN excluded unless no_data_val is given.
    src = ArraySource(arr)
    edges = [-50.0, 0.0, 50.0, 100.0, 150.0, 300.0]
    res = TileGraph(
        [
            StatsSink(src, name="all", ddof=1),
            StatsSink(src, no_data_val=-9999.0, bins=edges, name="nd"),
        ],
        block_size=16,
    ).run(use_pbar=False)
    for b_idx, ref in enumerate(_np_stats(arr, None, ddof=1)):
        assert res["all"].count[b_idx] == ref[0]
        np.testing.assert_allclose(res["all"].std[b_idx], ref[4], rtol=1e-10)
    for b_idx, ref in enumerate(_np_stats(arr, -9999.0)):
        np.testing.assert_allclose(res["nd"].mean[b_idx], ref[3], rtol=1e-12)
        hist, _ = np.histogram(ref[5], bins=edges)
        np.testing.assert_array_equal(res["nd"].hist[b_idx], hist)
    assert res["all"].hist is None

    # NaN no-data value.
    nan_src = ArraySource(arr[1:2], nodata=float("nan"))
    st = TileGraph(StatsSink(nan_src), block_size=20).run(use_pbar=False)["stats_0"]
    ref = _np_stats(arr[1:2], None)[0]
    assert st.count[0] == ref[0]
    np.testing.assert_allclose(st.mean[0], ref[3], rtol=1e-12)


def test_stats_sink_empty_band_and_integer_input():
    arr = np.full((2, 30, 30), 5, dtype=np.int16)
    arr[1] = 0
    arr[0, :3, :] = 7
    src = ArraySource(arr, nodata=0)
    st = TileGraph(StatsSink(src), block_size=8).run(use_pbar=False)["stats_0"]
    assert st.count.tolist() == [900, 0]
    assert st.min[0] == 5 and st.max[0] == 7
    assert np.isnan(st.mean[1]) and np.isnan(st.std[1])
    np.testing.assert_allclose(st.mean[0], arr[0].mean(), rtol=1e-12)


def test_stats_sink_alongside_other_sinks_in_a_chain(tmp_path):
    arr = _make_values(seed=5)[0]
    path = os.path.join(tmp_path, "in.tif")
    with rasterio.open(
        path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
        count=1, dtype="float32", crs="EPSG:32630",
        transform=from_origin(0, 0, 10, 10), nodata=-9999.0,
    ) as dst:
        dst.write(arr.astype(np.float32)[np.newaxis])
    src = RasterSource(path)
    smooth = mean_filter_node(src, 3, no_data_val=-9999.0)
    res = TileGraph(
        [MemorySink(smooth, name="img"), StatsSink(smooth, name="st")], block_size=16
    ).run(use_pbar=False)
    img = res["img"][0]
    vals = img[np.isfinite(img) & (img != -9999.0)]
    assert res["st"].count[0] == vals.size
    np.testing.assert_allclose(res["st"].mean[0], vals.mean(), rtol=1e-12)
    assert res["st"].min[0] == vals.min()


def test_stats_sink_parameter_errors():
    src = ArraySource(np.zeros((5, 5)))
    with pytest.raises(rsgispy.RSGISPyException, match="hist_range"):
        StatsSink(src, bins=10)
    with pytest.raises(rsgispy.RSGISPyException, match="hist_range"):
        StatsSink(src, hist_range=(0, 1))
    with pytest.raises(rsgispy.RSGISPyException, match="bins"):
        StatsSink(src, bins=0, hist_range=(0, 1))
    with pytest.raises(rsgispy.RSGISPyException, match="bin edges"):
        StatsSink(src, bins=[3.0, 1.0])
    with pytest.raises(rsgispy.RSGISPyException, match="ddof"):
        StatsSink(src, ddof=-1)


def _np_zonal(labels, values, label_nd, value_nd):
    rows = {}
    for lab in np.unique(labels[np.isfinite(labels)]):
        if label_nd is not None and lab == label_nd:
            continue
        rows[lab] = []
        for band in values:
            sel = (labels == lab) & np.isfinite(band)
            if value_nd is not None:
                sel &= band != value_nd
            vals = band[sel]
            rows[lab].append(vals)
    return rows


@pytest.mark.parametrize("block_size", BLOCK_SIZES)
def test_zonal_stats_sink_matches_numpy(block_size):
    rng = np.random.default_rng(7)
    labels = rng.integers(0, 9, size=(53, 47)).astype(np.int32)
    labels[20:30, 20:30] = 42
    values = _make_values(seed=9)[:2]
    lab_src = ArraySource(labels, nodata=0)
    val_src = ArraySource(values, nodata=-9999.0, band_names=["red", "nir"])
    df = TileGraph(
        ZonalStatsSink(lab_src, val_src, name="z"), block_size=block_size
    ).run(use_pbar=False)["z"]
    assert isinstance(df, pd.DataFrame)
    ref = _np_zonal(labels, values, 0, -9999.0)
    assert list(df.index) == sorted(ref)
    assert df.index.dtype == np.int64
    for lab, band_vals in ref.items():
        for name, vals in zip(["red", "nir"], band_vals):
            row = df.loc[lab]
            assert row[f"{name}_count"] == vals.size
            if vals.size:
                np.testing.assert_allclose(row[f"{name}_mean"], vals.mean(), rtol=1e-12)
                np.testing.assert_allclose(row[f"{name}_std"], vals.std(), rtol=1e-9)
                assert row[f"{name}_min"] == vals.min()
                assert row[f"{name}_max"] == vals.max()
            else:
                assert np.isnan(row[f"{name}_mean"])


def test_zonal_stats_sink_single_band_columns_and_explicit_nodata():
    labels = np.repeat(np.arange(1, 6, dtype=np.float64), 20).reshape(10, 10)
    values = np.arange(100, dtype=np.float64).reshape(10, 10)
    values[0, 0] = -1.0
    df = TileGraph(
        ZonalStatsSink(
            ArraySource(labels), ArraySource(values), no_data_val=-1.0,
            label_no_data_val=5,
        ),
        block_size=3,
    ).run(use_pbar=False)["zonal_0"]
    assert list(df.columns) == ["count", "mean", "min", "max", "std"]
    assert list(df.index) == [1, 2, 3, 4]
    assert df.loc[1, "count"] == 19
    np.testing.assert_allclose(df.loc[2, "mean"], values[2:4].mean())


def test_zonal_stats_sink_requires_single_band_labels():
    with pytest.raises(rsgispy.RSGISPyException, match="1 band"):
        TileGraph(
            ZonalStatsSink(ArraySource(np.zeros((2, 5, 5))), ArraySource(np.zeros((5, 5))))
        )
