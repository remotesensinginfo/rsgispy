"""
Tests for parallel tile evaluation, ``TileGraph.run(n_jobs=...)``: results
must be identical to ``n_jobs=1``.
"""

import os

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

import rsgispy
from rsgispy.imageprocess import (
    BandMathOp,
    DiskSink,
    MemorySink,
    PointwiseOp,
    RasterSource,
    SelectBands,
    StatsSink,
    TileGraph,
    run_tile_graph,
)
from rsgispy.classification.classpostprocess import majority_filter_node
from rsgispy.elevation.terrain import slope_node
from rsgispy.imagefilter.sar import refined_lee_filter_node
from rsgispy.imagefilter.stats import mean_filter_node, median_filter_node


def _write(path, arr, nodata=None):
    with rasterio.open(
        path, "w", driver="GTiff", height=arr.shape[1], width=arr.shape[2],
        count=arr.shape[0], dtype=str(arr.dtype), crs="EPSG:32630",
        transform=from_origin(500000.0, 5000000.0, 10.0, 10.0), nodata=nodata,
    ) as dst:
        dst.write(arr)
    return path


@pytest.fixture(name="img")
def _img(tmp_path):
    rng = np.random.default_rng(1)
    arr = rng.gamma(3.0, 30.0, size=(3, 101, 87)).astype(np.float32)
    arr[:, 10:14, 20:40] = -9999.0
    return _write(os.path.join(tmp_path, "in.tif"), arr, nodata=-9999.0)


def _build(img, out_path):
    src = RasterSource(img)
    b1 = SelectBands(src, 1)
    b2 = RasterSource(img, bands=2)
    med = median_filter_node(b1, 5, no_data_val=-9999.0)
    chain = mean_filter_node(med, 3, no_data_val=-9999.0)
    ratio = BandMathOp({"a": chain, "b": b2}, "a / (b + 1)", nodata=-9999.0)
    rl = refined_lee_filter_node(RasterSource(img, bands=3), 7, no_data_val=-9999.0)
    slp = slope_node(b2)
    cls = PointwiseOp(chain, lambda a: np.digitize(a, [50, 100, 150]).astype(np.float64))
    maj = majority_filter_node(cls, 3)
    return [
        DiskSink(ratio, out_path),
        MemorySink(rl, name="rl"),
        MemorySink(slp, name="slope"),
        MemorySink(maj, name="maj"),
        StatsSink(chain, bins=10, hist_range=(0, 300), name="stats"),
    ]


@pytest.mark.parametrize("block_size", [16, 37])
def test_parallel_results_identical_to_serial(img, tmp_path, block_size):
    results = {}
    disk = {}
    for n_jobs in (1, 2, 4, 8):
        out = os.path.join(tmp_path, f"out_{n_jobs}.tif")
        results[n_jobs] = TileGraph(_build(img, out), block_size=block_size).run(
            use_pbar=False, n_jobs=n_jobs
        )
        with rasterio.open(out) as ds:
            disk[n_jobs] = ds.read()
    ref = results[1]
    for n_jobs in (2, 4, 8):
        np.testing.assert_array_equal(disk[n_jobs], disk[1])
        for key in ("rl", "slope", "maj"):
            np.testing.assert_array_equal(results[n_jobs][key], ref[key])
        st, st_ref = results[n_jobs]["stats"], ref["stats"]
        for field in ("count", "min", "max", "mean", "std", "hist"):
            np.testing.assert_array_equal(getattr(st, field), getattr(st_ref, field))


def test_parallel_progress_bar_and_run_tile_graph(img):
    src = RasterSource(img, bands=1)
    out1 = run_tile_graph(MemorySink(src, name="x"), block_size=20, use_pbar=True)
    out4 = run_tile_graph(
        MemorySink(src, name="x"), block_size=20, use_pbar=True, n_jobs=4
    )
    np.testing.assert_array_equal(out1["x"], out4["x"])
    out_all = TileGraph(MemorySink(src, name="x"), block_size=20).run(
        use_pbar=False, n_jobs=None
    )
    np.testing.assert_array_equal(out1["x"], out_all["x"])


def test_parallel_error_propagates(img):
    def _boom(arr):
        if np.nanmax(arr) > 0:
            raise ValueError("boom")
        return arr

    node = PointwiseOp(RasterSource(img, bands=1), _boom)
    with pytest.raises(ValueError, match="boom"):
        TileGraph(MemorySink(node), block_size=16).run(use_pbar=False, n_jobs=4)


@pytest.mark.parametrize("bad", [0, -2, 1.5, True, "2"])
def test_invalid_n_jobs(img, bad):
    graph = TileGraph(MemorySink(RasterSource(img, bands=1)), block_size=32)
    with pytest.raises(rsgispy.RSGISPyException, match="n_jobs"):
        graph.run(use_pbar=False, n_jobs=bad)
