"""
Tests for the tile-graph improvements: band names (``StackBands`` /
``SelectBands`` ``band_names``, ``SetBandNames``), ``DiskSink`` data types
from the node metadata, ``SampleSink`` and ``norm_params_from_stats``.
"""

# pylint: disable=redefined-outer-name,missing-function-docstring
# pylint: disable=import-outside-toplevel

import json
import os

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

import rsgispy
from rsgispy.imageprocess import (
    ArraySource,
    DiskSink,
    MemorySink,
    RasterSource,
    SampleSink,
    SelectBands,
    SetBandNames,
    SetNoData,
    StackBands,
    StatsSink,
    TileGraph,
    datatype_from_meta,
    infer_node_meta,
    NodeMeta,
)
from rsgispy.imagefilter.stats import median_filter_node, stddev_filter_node

TRANSFORM = from_origin(500000.0, 5200000.0, 10.0, 10.0)
H, W = 47, 61


def _write(path, arr, nodata=None, dtype=None):
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[np.newaxis]
    with rasterio.open(
        path, "w", driver="GTiff", height=arr.shape[1], width=arr.shape[2],
        count=arr.shape[0], dtype=dtype or arr.dtype, crs="EPSG:32630",
        transform=TRANSFORM, nodata=nodata,
    ) as dst:
        dst.write(arr)
    return str(path)


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("tg_improve")
    rng = np.random.default_rng(7)
    vals = (rng.gamma(2.0, 300.0, size=(2, H, W))).astype(np.float32)
    vals[:, :, :3] = -9999.0
    vals[1, 10, 20] = np.nan
    labels = np.zeros((H, W), dtype=np.uint8)
    pick = rng.random((H, W)) < 0.3
    labels[pick] = rng.integers(1, 4, size=int(pick.sum()))
    labels[5, :] = 0
    dem = np.add.outer(np.arange(H) * 2.0, np.arange(W) * 1.5).astype(np.float32)
    return {
        "img": _write(tmp / "img.tif", vals, nodata=-9999.0),
        "labels": _write(tmp / "labels.tif", labels, nodata=0),
        "dem": _write(tmp / "dem.tif", dem, nodata=-9999.0),
        "vals": vals,
        "labels_arr": labels,
    }


# --------------------------------------------------------------------------
# Band names
# --------------------------------------------------------------------------


def test_stack_and_select_band_names(scene, tmp_path):
    src = RasterSource(scene["img"])
    stack = StackBands([median_filter_node(src, 3), stddev_filter_node(src, 3)],
                       band_names=["a_med", "b_med", "a_sd", "b_sd"])
    assert infer_node_meta(stack).band_names == ("a_med", "b_med", "a_sd", "b_sd")
    sel = SelectBands(stack, [3, 1], band_names=["x", "y"])
    assert infer_node_meta(sel).band_names == ("x", "y")
    renamed = SetBandNames(SelectBands(stack, 2), ["only"])
    assert infer_node_meta(renamed).band_names == ("only",)

    out = str(tmp_path / "stack.tif")
    graph = TileGraph([DiskSink(stack, out),
                       StatsSink(sel, name="st"),
                       DiskSink(renamed, str(tmp_path / "r.tif"), band_names=["override"])])
    assert "a_med" in graph.describe()
    res = graph.run(use_pbar=False)
    assert res["st"].band_names == ["x", "y"]
    with rasterio.open(out) as ds:
        assert ds.descriptions == ("a_med", "b_med", "a_sd", "b_sd")
    with rasterio.open(tmp_path / "r.tif") as ds:
        assert ds.descriptions == ("override",)  # a sink's band_names win


def test_band_names_wrong_length_raises(scene):
    src = RasterSource(scene["img"])
    with pytest.raises(rsgispy.RSGISPyException, match="band name"):
        TileGraph(MemorySink(StackBands([src], band_names=["one"])))
    with pytest.raises(rsgispy.RSGISPyException, match="band name"):
        TileGraph(MemorySink(SetBandNames(src, ["a", "b", "c"])))
    with pytest.raises(rsgispy.RSGISPyException, match="band name"):
        SelectBands(src, [1, 2], band_names=["a"])


def test_set_band_names_keeps_pixels(scene):
    src = RasterSource(scene["img"])
    res = TileGraph([MemorySink(src, name="a"),
                     MemorySink(SetBandNames(src, ["p", "q"]), name="b")]).run(use_pbar=False)
    np.testing.assert_array_equal(res["a"], res["b"])


# --------------------------------------------------------------------------
# DiskSink data type from the metadata
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype,expected",
    [
        (np.uint8, rsgispy.TYPE_8UINT),
        (np.int16, rsgispy.TYPE_16INT),
        (np.uint32, rsgispy.TYPE_32UINT),
        (np.float32, rsgispy.TYPE_32FLOAT),
        (np.float64, rsgispy.TYPE_32FLOAT),
        (np.bool_, rsgispy.TYPE_8UINT),
        (None, rsgispy.TYPE_32FLOAT),
    ],
)
def test_datatype_from_meta(dtype, expected):
    meta = NodeMeta(dtype=None if dtype is None else np.dtype(dtype))
    assert datatype_from_meta(meta) == expected


def test_disk_sink_datatype_from_input(scene, tmp_path):
    from rsgispy.classification.classpostprocess import majority_filter_node
    from rsgispy.elevation.illumination import hillshade_node

    dem = RasterSource(scene["dem"])
    labels = RasterSource(scene["labels"])
    arr16 = ArraySource(np.arange(H * W, dtype=np.int16).reshape(1, H, W),
                        transform=TRANSFORM, crs="EPSG:32630")
    sinks = {
        "hs": (hillshade_node(dem, 315.0, 45.0), "uint8"),
        "maj": (majority_filter_node(labels, 3, dtype=np.uint8), "uint8"),
        "i16": (arr16, "int16"),
        "f64": (median_filter_node(dem, 3), "float32"),
    }
    graph = TileGraph([DiskSink(node, str(tmp_path / f"{key}.tif"))
                       for key, (node, _) in sinks.items()]
                      + [DiskSink(dem, str(tmp_path / "dem64.tif"),
                                  datatype=rsgispy.TYPE_64FLOAT)])
    assert "datatype=uint8 (from input)" in graph.describe()
    graph.run(use_pbar=False)
    for key, (_, dtype) in sinks.items():
        with rasterio.open(tmp_path / f"{key}.tif") as ds:
            assert ds.dtypes[0] == dtype, key
    with rasterio.open(tmp_path / "dem64.tif") as ds:
        assert ds.dtypes[0] == "float64"  # an explicit datatype wins


def test_majority_dtype_blocks_and_chain(scene):
    """majority_filter_node(dtype=uint8) gives uint8 blocks; chaining works."""
    from rsgispy.classification.classpostprocess import (
        apply_majority_filter,
        majority_filter_node,
    )

    labels = RasterSource(scene["labels"])
    once = majority_filter_node(labels, 3, dtype=np.uint8)
    twice = majority_filter_node(once, 5, dtype=np.uint8)
    res = TileGraph([MemorySink(once, name="a"), MemorySink(twice, name="b")],
                    block_size=16).run(use_pbar=False)
    assert res["a"].dtype == np.uint8 and res["b"].dtype == np.uint8
    as_float = TileGraph(
        MemorySink(majority_filter_node(majority_filter_node(labels, 3), 5), name="f"),
        block_size=16,
    ).run(use_pbar=False)["f"]
    np.testing.assert_array_equal(res["b"], as_float.astype(np.uint8))
    assert apply_majority_filter  # the file function is unchanged (golden tests)


def test_predict_thematic_uint8_only_when_exact(scene):
    pytest.importorskip("sklearn")
    from sklearn.tree import DecisionTreeClassifier
    from rsgispy.ml.predict import predict_raster_node

    src = RasterSource(scene["img"])
    rng = np.random.default_rng(0)
    clf = DecisionTreeClassifier(max_depth=3, random_state=0).fit(
        rng.random((60, 2)) * 1000, rng.integers(1, 4, 60))
    them = predict_raster_node(src, clf, thematic=True)
    big = predict_raster_node(src, clf, thematic=True, lut={1: 300})
    plain = predict_raster_node(src, clf)
    assert infer_node_meta(them).dtype == np.uint8
    assert infer_node_meta(big).dtype == np.float64
    assert infer_node_meta(plain).dtype == np.float64
    res = TileGraph([MemorySink(them, name="t"), MemorySink(big, name="b")]).run(use_pbar=False)
    assert res["t"].dtype == np.uint8
    assert 300 in np.unique(res["b"])


# --------------------------------------------------------------------------
# SampleSink
# --------------------------------------------------------------------------


def _expected_samples(scene):
    vals = scene["vals"].astype(np.float64)
    labels = scene["labels_arr"]
    valid = (labels != 0) & np.all(np.isfinite(vals) & (vals != -9999.0), axis=0)
    return valid, vals, labels


def test_sample_sink_all_pixels(scene):
    src = SetBandNames(RasterSource(scene["img"]), ["b1v", "b2v"])
    df = TileGraph(SampleSink(src, RasterSource(scene["labels"]), name="s"),
                   block_size=16).run(use_pbar=False)["s"]
    valid, vals, labels = _expected_samples(scene)
    assert list(df.columns) == ["class_name", "out_cls_id", "internal_cls_id",
                                "x", "y", "b1v", "b2v"]
    assert len(df) == int(valid.sum())
    rows, cols = np.nonzero(valid)
    order = np.lexsort((rows * W + cols, labels[rows, cols]))
    np.testing.assert_array_equal(df["out_cls_id"].to_numpy(), labels[rows, cols][order])
    np.testing.assert_array_equal(df["b1v"].to_numpy(), vals[0][rows, cols][order])
    np.testing.assert_array_equal(df["b2v"].to_numpy(), vals[1][rows, cols][order])
    xs, ys = rasterio.transform.xy(TRANSFORM, rows[order], cols[order])
    np.testing.assert_allclose(df["x"].to_numpy(), xs)
    np.testing.assert_allclose(df["y"].to_numpy(), ys)
    assert set(df["internal_cls_id"]) == {0, 1, 2}
    assert df["class_name"].iloc[0] == "1"


@pytest.mark.parametrize("block_size,n_jobs", [(256, 1), (16, 1), (23, 2), (50, 3)])
def test_sample_sink_subset_independent_of_tiling(scene, block_size, n_jobs):
    src = RasterSource(scene["img"])
    labels = RasterSource(scene["labels"])
    ref = TileGraph(SampleSink(src, labels, max_per_class=25, name="s"),
                    block_size=64).run(use_pbar=False)["s"]
    got = TileGraph(SampleSink(src, labels, max_per_class=25, name="s"),
                    block_size=block_size).run(use_pbar=False, n_jobs=n_jobs)["s"]
    assert ref.groupby("out_cls_id").size().to_dict() == {1: 25, 2: 25, 3: 25}
    assert got.equals(ref)
    other = TileGraph(SampleSink(src, labels, max_per_class=25, rnd_seed=7, name="s"),
                      block_size=block_size).run(use_pbar=False)["s"]
    assert not other.equals(ref)  # another seed, another sample


def test_sample_sink_options(scene):
    src = RasterSource(scene["img"])
    labels = RasterSource(scene["labels"])
    df = TileGraph(
        SampleSink(src, labels, max_per_class={1: 5, 2: 7}, include_xy=False,
                   class_names={1: "Water", 2: "Forest", 3: "Grass"}, name="s"),
    ).run(use_pbar=False)["s"]
    counts = df.groupby("class_name").size().to_dict()
    valid, _, labels_arr = _expected_samples(scene)
    assert counts["Water"] == 5 and counts["Forest"] == 7
    assert counts["Grass"] == int((valid & (labels_arr == 3)).sum())  # not limited
    assert "x" not in df.columns
    # No data: with SetNoData(None) the -9999 strip is sampled as values.
    df_all = TileGraph(SampleSink(SetNoData(src, None), labels, name="s")).run(
        use_pbar=False)["s"]
    assert (df_all["b1"] == -9999.0).any()
    assert len(df_all) > int(valid.sum())


def test_sample_sink_repeated_names_raise(scene):
    src = RasterSource(scene["img"])
    feats = StackBands([src, median_filter_node(src, 3)])  # b1, b2, b1, b2
    with pytest.raises(rsgispy.RSGISPyException, match="repeated names"):
        TileGraph(SampleSink(SetBandNames(feats, ["a", "a", "b", "c"]),
                             RasterSource(scene["labels"]))).run(use_pbar=False)


def test_sample_sink_round_trip_and_model(scene, tmp_path):
    pytest.importorskip("sklearn")
    pytest.importorskip("pyarrow")
    from sklearn.tree import DecisionTreeClassifier
    from rsgispy.classification.classdata import read_cls_samples, write_cls_samples

    src = SetBandNames(RasterSource(scene["img"]), ["f1", "f2"])
    df = TileGraph(SampleSink(src, RasterSource(scene["labels"]), max_per_class=30,
                              name="s")).run(use_pbar=False)["s"]
    out = os.path.join(str(tmp_path), "samples.parquet")
    write_cls_samples(df, out)
    back = read_cls_samples(out)
    assert list(back.columns) == list(df.columns)
    np.testing.assert_array_equal(back["f1"].to_numpy(), df["f1"].to_numpy())
    clf = DecisionTreeClassifier(random_state=0).fit(df[["f1", "f2"]], df["out_cls_id"])
    assert set(clf.classes_) == {1, 2, 3}


# --------------------------------------------------------------------------
# norm_params_from_stats
# --------------------------------------------------------------------------

EXACT_METHODS = ["none", "linear", "stdev", "zscore", "decimal_scaling",
                 "standard_scaler", "minmax_scaler", "maxabs_scaler"]
HIST_METHODS = ["cumulative", "robust_zscore", "robust_scaler"]


@pytest.fixture(scope="module")
def norm_stats(scene):
    src = RasterSource(scene["img"])
    top = float(np.nanmax(scene["vals"])) + 1.0
    stats = TileGraph(StatsSink(src, bins=4000, hist_range=(-1.0, top), name="s"),
                      block_size=16).run(use_pbar=False)["s"]
    return src, stats, (top + 1.0) / 4000


def _close_params(got, ref, rtol):
    for g_band, r_band in zip(got["bands"], ref["bands"]):
        assert set(g_band) == set(r_band)
        for key, r_val in r_band.items():
            g_val = g_band[key]
            if isinstance(r_val, float) and isinstance(g_val, float):
                assert g_val == pytest.approx(r_val, rel=rtol, abs=1e-12), key
            else:
                assert g_val == r_val, key


@pytest.mark.parametrize("method", EXACT_METHODS + ["user"])
def test_norm_params_from_stats_exact(scene, norm_stats, method):
    pytest.importorskip("sklearn")
    from rsgispy.imagecalc.rescale import (
        calc_img_norm_params,
        img_norm_params_node,
        norm_params_from_stats,
    )

    src, stats, _ = norm_stats
    kw = {"in_min_max": {"min": 10.0, "max": 900.0}} if method == "user" else {}
    ref = calc_img_norm_params(scene["img"], method=method, use_pbar=False, **kw)
    got = norm_params_from_stats(stats, method=method, **kw)
    assert got["method"] == ref["method"]
    assert got["selected_bands"] == ref["selected_bands"]
    _close_params(got, ref, rtol=1e-12)
    res = TileGraph([MemorySink(img_norm_params_node(src, ref), name="r"),
                     MemorySink(img_norm_params_node(src, got), name="g")]).run(use_pbar=False)
    np.testing.assert_allclose(res["g"], res["r"], rtol=1e-6, atol=1e-6, equal_nan=True)


@pytest.mark.parametrize("method", HIST_METHODS)
def test_norm_params_from_stats_histogram(scene, norm_stats, method):
    """
    Percentiles from the histogram are off by at most the share of pixels in
    one bin: the fraction of pixels below each estimate is within that share
    of the requested fraction.
    """
    pytest.importorskip("sklearn")
    from rsgispy.imagecalc.rescale import calc_img_norm_params, norm_params_from_stats

    _, stats, _ = norm_stats
    ref = calc_img_norm_params(scene["img"], method=method, use_pbar=False)
    got = norm_params_from_stats(stats, method=method)
    for b_idx, (g_band, r_band) in enumerate(zip(got["bands"], ref["bands"])):
        vals = scene["vals"][b_idx].astype(np.float64)
        vals = vals[np.isfinite(vals) & (vals != -9999.0)]
        bin_share = float(stats.hist[b_idx].max()) / vals.size + 1.0 / vals.size
        if method == "cumulative":
            checks = [(g_band["stretch_min"], 0.02), (g_band["stretch_max"], 0.98)]
        else:
            key = "median" if method == "robust_zscore" else "center"
            checks = [(g_band[key], 0.5)]
            assert g_band["scale"] == pytest.approx(r_band["scale"], rel=0.02)
        for est, frac in checks:
            assert abs(float(np.mean(vals <= est)) - frac) <= bin_share, (est, frac)


def test_norm_params_from_stats_errors(norm_stats):
    from rsgispy.imagecalc.rescale import norm_params_from_stats

    src, stats, _ = norm_stats
    for method in ("log", "l2", "softmax", "quantile_transformer"):
        with pytest.raises(rsgispy.RSGISPyException, match="Supported methods"):
            norm_params_from_stats(stats, method=method)
    no_hist = TileGraph(StatsSink(src, name="s")).run(use_pbar=False)["s"]
    with pytest.raises(rsgispy.RSGISPyException, match="histogram"):
        norm_params_from_stats(no_hist, method="cumulative")
    short = TileGraph(StatsSink(src, bins=10, hist_range=(0.0, 10.0), name="s")).run(
        use_pbar=False)["s"]
    with pytest.raises(rsgispy.RSGISPyException, match="hist_range"):
        norm_params_from_stats(short, method="robust_zscore")
    ddof1 = TileGraph(StatsSink(src, ddof=1, name="s")).run(use_pbar=False)["s"]
    with pytest.raises(rsgispy.RSGISPyException, match="ddof"):
        norm_params_from_stats(ddof1, method="zscore")


def test_norm_params_from_stats_json_round_trip(scene, norm_stats, tmp_path):
    from rsgispy.imagecalc.rescale import (
        apply_img_norm_params,
        img_norm_params_node,
        norm_params_from_stats,
    )

    src, stats, _ = norm_stats
    json_file = str(tmp_path / "params.json")
    params = norm_params_from_stats(stats, method="zscore", out_stats_file=json_file)
    with open(json_file, encoding="utf-8") as f_obj:
        assert json.load(f_obj)["bands"] == json.loads(json.dumps(params["bands"]))
    res = TileGraph([MemorySink(img_norm_params_node(src, params), name="a"),
                     MemorySink(img_norm_params_node(src, stats_file=json_file), name="b")]
                    ).run(use_pbar=False)
    np.testing.assert_array_equal(res["a"], res["b"])
    out = str(tmp_path / "z.tif")
    apply_img_norm_params(scene["img"], out, stats_file=json_file, use_pbar=False)
    with rasterio.open(out) as ds:
        np.testing.assert_allclose(ds.read(), res["a"].astype(np.float32), equal_nan=True)
