"""
Tests for the tile-graph prediction nodes of :mod:`rsgispy.ml.predict`
(:func:`feature_stack_nodes`, :func:`predict_raster_node`,
:func:`predict_proba_raster_node`, :func:`predict_quantiles_raster_node`):
they must give the same pixels as the file-based prediction functions, and
a chain feature nodes -> prediction -> majority filter -> DiskSink must
equal the file functions run one after another.
"""

import numpy as np
import pytest
import rasterio

import rsgispy
import rsgispy.classification.classpostprocess as cp
from golden_morph_class_ml import _feature_stack, block_size_env, make_inputs
from rsgispy.imageprocess import (
    DiskSink,
    MemorySink,
    RasterSource,
    SelectBands,
    StackBands,
    TileGraph,
)
from rsgispy.ml import predict as mp


@pytest.fixture(scope="module")
def ml_inputs(tmp_path_factory):
    work = tmp_path_factory.mktemp("predict_nodes")
    return work, make_inputs(work)


def _read(path):
    with rasterio.open(path) as src:
        return src.read(), src.dtypes[0], src.nodata, src.descriptions


def _run_nodes(node, dtype, block_size):
    return TileGraph(MemorySink(node, dtype=dtype, name="o"), block_size=block_size).run(
        use_pbar=False
    )["o"]


@pytest.mark.parametrize("block_size", [16, 23])
def test_predict_raster_node_matches_file(ml_inputs, tmp_path, block_size):
    """predict_raster_node == predict_raster (masks, LUT, threshold, scaler, df, multi-output)."""
    _, inp = ml_inputs
    models = inp["models"]
    stack = _feature_stack(inp)
    mask = RasterSource(inp["mask"], bands=1)
    cases = [
        ("clf", {"thematic": True}, {}, "uint8"),
        ("clf", {"thematic": True, "lut": {1: 10, 2: 20, 3: 30, 4: 40}}, {"mask": True}, "uint8"),
        ("bin", {"threshold": 0.5}, {}, "uint8"),
        ("rf", {"no_data_val": -1.0}, {}, "float32"),
        ("clf_sc", {"thematic": True, "scaler": models["scaler"]}, {}, "uint8"),
        ("df_reg", {}, {"mask": True}, "float32"),
        ("rf2", {}, {"mask": True}, "float32"),
    ]
    for idx, (model, kw, extra, dtype) in enumerate(cases):
        out = str(tmp_path / f"p{idx}.tif")
        file_kw = dict(kw)
        if extra.get("mask"):
            file_kw["in_msk_img"] = inp["mask"]
        with block_size_env(block_size):
            mp.predict_raster(models[model], stack, out, use_pbar=False, **file_kw)
        ref, ref_dt, ref_nd, _ = _read(out)
        assert ref_dt == dtype
        node = mp.predict_raster_node(
            mp.feature_stack_nodes(stack),
            models[model],
            mask=mask if extra.get("mask") else None,
            **kw,
        )
        got = _run_nodes(node, np.dtype(dtype), block_size)
        np.testing.assert_array_equal(got, ref, err_msg=f"case {idx} ({model})")
        graph = TileGraph(MemorySink(node))
        assert graph.node_meta[node].nodata == ref_nd


def test_predict_proba_and_quantile_nodes_match_file(ml_inputs, tmp_path):
    """Probability (with class and entropy bands) and quantile nodes == file functions."""
    _, inp = ml_inputs
    models = inp["models"]
    stack = _feature_stack(inp)
    lut = {1: 11, 2: 12, 3: 13, 4: 14}
    paths = {k: str(tmp_path / f"{k}.tif") for k in ("prob", "cls", "ent")}
    mp.predict_proba_raster(
        models["clf"], stack, paths["prob"], out_class_img=paths["cls"],
        out_entropy_img=paths["ent"], lut=lut, use_pbar=False,
    )
    probs = mp.predict_proba_raster_node(
        stack.to_tile_nodes(), models["clf"], lut=lut, with_class=True, with_entropy=True
    )
    n_cls = len(models["clf"].classes_)
    graph = TileGraph(
        [
            MemorySink(SelectBands(probs, list(range(1, n_cls + 1))), np.float32, name="p"),
            MemorySink(SelectBands(probs, n_cls + 1), np.int32, name="c"),
            MemorySink(SelectBands(probs, n_cls + 2), np.float32, name="e"),
        ],
        block_size=16,
    )
    assert graph.node_meta[probs].band_names[-2:] == ("predicted_class", "shannon_entropy")
    res = graph.run(use_pbar=False)
    np.testing.assert_array_equal(res["p"], _read(paths["prob"])[0])
    np.testing.assert_array_equal(res["c"], _read(paths["cls"])[0])
    np.testing.assert_array_equal(res["e"], _read(paths["ent"])[0])

    for key, est, kw in (
        ("ens", models["rf"], {"quantiles": (0.1, 0.5, 0.9)}),
        ("conf", models["lin"], {"conformal_residuals": models["residuals"], "alpha": 0.2}),
        ("map", {0.1: models["q_lo"], 0.9: models["q_hi"]}, {}),
    ):
        out = str(tmp_path / f"q_{key}.tif")
        mp.predict_quantiles_raster(est, stack, out, in_msk_img=inp["mask"], use_pbar=False, **kw)
        ref, _, _, ref_names = _read(out)
        node = mp.predict_quantiles_raster_node(
            mp.feature_stack_nodes(stack), est, mask=RasterSource(inp["mask"]), **kw
        )
        assert tuple(TileGraph(MemorySink(node)).node_meta[node].band_names) == ref_names
        np.testing.assert_array_equal(_run_nodes(node, np.float32, 23), ref)


@pytest.mark.parametrize("block_size", [16, 256])
def test_chain_features_predict_majority_disk(ml_inputs, tmp_path, block_size):
    """feature nodes -> predict_raster_node -> majority_filter_node -> DiskSink
    == predict_raster then apply_majority_filter."""
    _, inp = ml_inputs
    clf = inp["models"]["clf"]
    stack = _feature_stack(inp)
    pred_file = str(tmp_path / "pred.tif")
    maj_file = str(tmp_path / "maj.tif")
    with block_size_env(block_size):
        mp.predict_raster(clf, stack, pred_file, in_msk_img=inp["mask"], thematic=True, use_pbar=False)
        cp.apply_majority_filter(pred_file, maj_file, 3, use_pbar=False)

    cls = mp.predict_raster_node(
        mp.feature_stack_nodes(stack), clf, thematic=True, mask=RasterSource(inp["mask"], bands=1)
    )
    smooth = cp.majority_filter_node(cls, 3, dtype=np.uint8, no_data_val=0)
    out = str(tmp_path / "chain.tif")
    TileGraph(
        DiskSink(smooth, out, datatype=rsgispy.TYPE_8UINT), block_size=block_size
    ).run(use_pbar=False)
    got, got_dt, got_nd, _ = _read(out)
    ref, ref_dt, ref_nd, _ = _read(maj_file)
    np.testing.assert_array_equal(got, ref)
    assert (got_dt, got_nd) == (ref_dt, ref_nd)


def test_feature_stack_nodes_meta_and_errors(ml_inputs):
    """Feature nodes carry the stack's names; mismatched feature counts raise."""
    _, inp = ml_inputs
    stack = _feature_stack(inp)
    feats = mp.feature_stack_nodes(stack)
    stacked = StackBands(feats)
    graph = TileGraph(MemorySink(stacked))
    assert list(graph.node_meta[stacked].band_names) == stack.feature_names
    with pytest.raises(rsgispy.RSGISPyException, match="expects 4 features"):
        TileGraph(MemorySink(mp.predict_raster_node(feats[:1], inp["models"]["clf"])))
    with pytest.raises(rsgispy.RSGISPyException):
        mp.predict_proba_raster_node(feats, inp["models"]["lin"])
    with pytest.raises(rsgispy.RSGISPyException):
        mp.predict_quantiles_raster_node(feats, inp["models"]["lin"])
    multi = mp.predict_raster_node(feats, inp["models"]["rf2"], n_outputs=1)
    with pytest.raises(rsgispy.RSGISPyException, match="n_outputs"):
        TileGraph(MemorySink(multi)).run(use_pbar=False)
