"""
Tests for the per-pixel classification confidence outputs:
:func:`rsgispy.ml.predict.calc_proba_confidence`, ``out_conf_img`` of
:func:`rsgispy.ml.predict.predict_proba_raster` and of the
``apply_*_classifier`` functions (sklearn, LightGBM, XGBoost, CatBoost),
the ``with_conf`` band of :func:`rsgispy.ml.predict.predict_proba_raster_node`
and :func:`rsgispy.classification.classaccuracymetrics.calc_confidence_reliability`.

The stub classifiers return the feature values as their class probabilities,
so the expected confidences are computed by hand from known probabilities.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from shapely.geometry import Point

import rsgispy
import rsgispy.classification.classaccuracymetrics as cam
import rsgispy.classification.classdata as rcd
import rsgispy.classification.classsklearn as rcsk
from rsgispy.imageprocess import MemorySink, RasterSource, SelectBands, TileGraph
from rsgispy.ml import predict as mp
from rsgispy.ml.stack import BandSpec, ImageFeatureStack

from conftest import write_raster

_ND = -9999.0

# Multi-class pixels (row-major on a 2 x 3 grid); the last pixel is masked out.
_MC_PROBS = np.array(
    [
        [0.7, 0.2, 0.1],
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        [1.0, 0.0, 0.0],
        [0.4, 0.4, 0.2],
        [0.2, 0.3, 0.5],
        [0.6, 0.3, 0.1],
    ]
)
# Hand-computed confidences of the first five pixels.
_MC_EXPECT = {
    # p1 - p2
    "margin": [0.5, 0.0, 1.0, 0.0, 0.2],
    # (p1 - 1/3) / (2/3)
    "max_prob": [0.55, 0.0, 1.0, 0.1, 0.25],
    # 1 - H / ln(3)
    "entropy": [0.2701533008, 0.0, 1.0, 0.0397702821, 0.0627694368],
}

# Binary positive-class probabilities on a 2 x 3 grid; the last pixel is masked.
_BIN_P = np.array([0.9, 0.5, 0.2, 1.0, 0.65, 0.3])
_BIN_EXPECT = {
    # |2p - 1|
    "margin": [0.8, 0.0, 0.6, 1.0, 0.3],
    # (max(p, 1 - p) - 0.5) / 0.5 (same as the margin for two classes)
    "max_prob": [0.8, 0.0, 0.6, 1.0, 0.3],
    # 1 - H2(p)
    "entropy": [0.5310044064, 0.0, 0.2780719051, 1.0, 0.0659319446],
}


class _ProbaStub:
    """Classifier whose class probabilities are the input features."""

    def __init__(self, classes, binary=False):
        self.classes_ = np.asarray(classes)
        self.binary = binary
        self.n_features_in_ = 1 if binary else len(classes)

    def fit(self, *_args, **_kwargs):
        return self

    def predict_proba(self, x_in):
        x_arr = np.asarray(x_in, dtype=np.float64)
        if self.binary:
            pos = x_arr[:, 0]
            return np.column_stack([1.0 - pos, pos])
        return x_arr

    def predict(self, x_in):
        return self.classes_[np.argmax(self.predict_proba(x_in), axis=1)]


def _write(path: Path, arr: np.ndarray, nodata=None, dtype=None) -> str:
    return write_raster(path, arr, dtype=dtype, nodata=nodata, origin=(500000.0, 5600000.0))


def _read1(path):
    with rasterio.open(path) as src:
        return src.read(1), src.nodata, src.dtypes[0], src.descriptions


@pytest.fixture()
def mc_inputs(tmp_path):
    feat = _MC_PROBS.T.reshape(3, 2, 3).astype(np.float32)
    feat_img = _write(tmp_path / "mc_feat.tif", feat)
    msk = np.ones((2, 3), dtype=np.uint8)
    msk[1, 2] = 0
    msk_img = _write(tmp_path / "msk.tif", msk, nodata=0)
    return tmp_path, feat_img, msk_img


@pytest.fixture()
def bin_inputs(tmp_path):
    feat_img = _write(tmp_path / "bin_feat.tif", _BIN_P.reshape(2, 3).astype(np.float32))
    msk = np.ones((2, 3), dtype=np.uint8)
    msk[1, 2] = 0
    msk_img = _write(tmp_path / "msk.tif", msk, nodata=0)
    return tmp_path, feat_img, msk_img


def _mc_cfg(feat_img):
    class_info = {
        "A": rcd.ClassInfoObj(out_id=1, red=255, green=0, blue=0),
        "B": rcd.ClassInfoObj(out_id=2, red=0, green=255, blue=0),
        "C": rcd.ClassInfoObj(out_id=3, red=0, green=0, blue=255),
    }
    band_info = [
        rcd.ImageBandInfo(feat_img, name="P", bands=[1, 2, 3], band_names=["pa", "pb", "pc"])
    ]
    return class_info, band_info


def _bin_cfg(feat_img):
    class_info = {
        "Neg": rcd.ClassInfoObj(out_id=1, red=255, green=0, blue=0),
        "Pos": rcd.ClassInfoObj(out_id=2, red=0, green=255, blue=0),
    }
    band_info = [rcd.ImageBandInfo(feat_img, name="P", bands=[1], band_names=["p"])]
    return class_info, band_info


# ---------------------------------------------------------------------------
# calc_proba_confidence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", list(mp.CONF_METHODS))
def test_calc_proba_confidence_multiclass(method):
    got = mp.calc_proba_confidence(_MC_PROBS[:5], method)
    np.testing.assert_allclose(got, _MC_EXPECT[method], atol=1e-8)
    assert got.dtype == np.float64


@pytest.mark.parametrize("method", list(mp.CONF_METHODS))
def test_calc_proba_confidence_binary(method):
    probs_2d = np.column_stack([1.0 - _BIN_P[:5], _BIN_P[:5]])
    np.testing.assert_allclose(
        mp.calc_proba_confidence(probs_2d, method), _BIN_EXPECT[method], atol=1e-8
    )
    # A 1D array is the positive-class probability.
    np.testing.assert_allclose(
        mp.calc_proba_confidence(_BIN_P[:5], method), _BIN_EXPECT[method], atol=1e-8
    )


def test_calc_proba_confidence_default_and_invalid():
    np.testing.assert_allclose(
        mp.calc_proba_confidence(_MC_PROBS[:5]), _MC_EXPECT["margin"], atol=1e-12
    )
    with pytest.raises(rsgispy.RSGISPyException, match="conf_method"):
        mp.calc_proba_confidence(_MC_PROBS, "variance")


# ---------------------------------------------------------------------------
# predict_proba_raster / predict_proba_raster_node
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", list(mp.CONF_METHODS))
def test_predict_proba_raster_conf(mc_inputs, method):
    work, feat_img, msk_img = mc_inputs
    stack = ImageFeatureStack([BandSpec(feat_img, name="p")])
    est = _ProbaStub([1, 2, 3])
    out_conf = work / f"conf_{method}.tif"
    mp.predict_proba_raster(
        est,
        stack,
        work / "probs.tif",
        in_msk_img=msk_img,
        out_conf_img=out_conf,
        conf_method=method,
        use_pbar=False,
    )
    arr, nodata, dtype, desc = _read1(out_conf)
    assert dtype == "float32"
    assert nodata == _ND
    assert desc == (f"confidence_{method}",)
    np.testing.assert_allclose(arr.ravel()[:5], _MC_EXPECT[method], atol=1e-6)
    assert arr[1, 2] == _ND


def test_predict_proba_raster_unchanged_without_conf(mc_inputs):
    work, feat_img, msk_img = mc_inputs
    stack = ImageFeatureStack([BandSpec(feat_img, name="p")])
    est = _ProbaStub([1, 2, 3])
    kw = {"in_msk_img": msk_img, "use_pbar": False, "calc_stats": False}
    mp.predict_proba_raster(
        est, stack, work / "p_a.tif", out_class_img=work / "c_a.tif",
        out_entropy_img=work / "e_a.tif", **kw,
    )
    mp.predict_proba_raster(
        est, stack, work / "p_b.tif", out_class_img=work / "c_b.tif",
        out_entropy_img=work / "e_b.tif", out_conf_img=work / "conf_b.tif", **kw,
    )
    for name in ("p", "c", "e"):
        with rasterio.open(work / f"{name}_a.tif") as s_a, rasterio.open(
            work / f"{name}_b.tif"
        ) as s_b:
            np.testing.assert_array_equal(s_a.read(), s_b.read())
    # Entropy confidence == 1 - entropy raster.
    ent, _, _, _ = _read1(work / "e_a.tif")
    mp.predict_proba_raster(
        est, stack, work / "p_c.tif", out_conf_img=work / "conf_c.tif",
        conf_method="entropy", **kw,
    )
    conf, _, _, _ = _read1(work / "conf_c.tif")
    valid = ent != _ND
    np.testing.assert_allclose(conf[valid], 1.0 - ent[valid], atol=1e-6)


def test_predict_proba_raster_conf_invalid_method(mc_inputs):
    work, feat_img, _ = mc_inputs
    stack = ImageFeatureStack([BandSpec(feat_img, name="p")])
    with pytest.raises(rsgispy.RSGISPyException, match="conf_method"):
        mp.predict_proba_raster(
            _ProbaStub([1, 2, 3]), stack, work / "p.tif",
            out_conf_img=work / "c.tif", conf_method="bogus", use_pbar=False,
        )


@pytest.mark.parametrize("method", list(mp.CONF_METHODS))
def test_predict_proba_raster_node_conf(mc_inputs, method):
    _, feat_img, msk_img = mc_inputs
    est = _ProbaStub([1, 2, 3])
    stack = ImageFeatureStack([BandSpec(feat_img, name="p")])
    node = mp.predict_proba_raster_node(
        mp.feature_stack_nodes(stack),
        est,
        with_entropy=True,
        with_conf=True,
        conf_method=method,
        mask=RasterSource(msk_img, bands=1),
    )
    graph = TileGraph(MemorySink(SelectBands(node, 5), name="o"))
    assert graph.node_meta[node].band_names[-1] == f"confidence_{method}"
    arr = graph.run(use_pbar=False)["o"][0]
    np.testing.assert_allclose(arr.ravel()[:5], _MC_EXPECT[method], atol=1e-6)
    assert arr[1, 2] == _ND
    with pytest.raises(rsgispy.RSGISPyException, match="conf_method"):
        mp.predict_proba_raster_node(
            mp.feature_stack_nodes(stack), est, with_conf=True, conf_method="x"
        )


# ---------------------------------------------------------------------------
# apply_*_classifier (shared engine)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", list(mp.CONF_METHODS))
def test_apply_sklearn_classifier_conf_multiclass(mc_inputs, method):
    work, feat_img, msk_img = mc_inputs
    class_info, band_info = _mc_cfg(feat_img)
    out_conf = work / f"cls_conf_{method}.tif"
    rcsk.apply_sklearn_classifier(
        cls_cfg=class_info,
        sk_classifier=_ProbaStub([1, 2, 3]),
        in_msk_img=msk_img,
        img_msk_val=1,
        output_img=work / "cls.tif",
        img_band_info=band_info,
        use_internal_id=False,
        out_conf_img=out_conf,
        conf_method=method,
        use_pbar=False,
    )
    arr, nodata, dtype, desc = _read1(out_conf)
    assert dtype == "float32"
    assert nodata == _ND
    assert desc == (f"confidence_{method}",)
    np.testing.assert_allclose(arr.ravel()[:5], _MC_EXPECT[method], atol=1e-6)
    assert arr[1, 2] == _ND


def test_apply_sklearn_classifier_conf_only_and_unchanged(mc_inputs):
    work, feat_img, msk_img = mc_inputs
    class_info, band_info = _mc_cfg(feat_img)
    kw = {
        "cls_cfg": class_info,
        "sk_classifier": _ProbaStub([1, 2, 3]),
        "in_msk_img": msk_img,
        "img_msk_val": 1,
        "img_band_info": band_info,
        "use_internal_id": False,
        "use_pbar": False,
        "calc_stats": False,
    }
    rcsk.apply_sklearn_classifier(
        output_img=work / "cls_a.tif", out_score_img=work / "scr_a.tif", **kw
    )
    rcsk.apply_sklearn_classifier(
        output_img=work / "cls_b.tif",
        out_score_img=work / "scr_b.tif",
        out_conf_img=work / "conf_b.tif",
        **kw,
    )
    for name in ("cls", "scr"):
        with rasterio.open(work / f"{name}_a.tif") as s_a, rasterio.open(
            work / f"{name}_b.tif"
        ) as s_b:
            np.testing.assert_array_equal(s_a.read(), s_b.read())
            assert s_a.nodata == s_b.nodata
    # The confidence raster alone is a valid output.
    rcsk.apply_sklearn_classifier(out_conf_img=work / "conf_only.tif", **kw)
    arr, _, _, _ = _read1(work / "conf_only.tif")
    np.testing.assert_allclose(arr.ravel()[:5], _MC_EXPECT["margin"], atol=1e-6)


def test_apply_sklearn_classifier_conf_errors(mc_inputs):
    work, feat_img, msk_img = mc_inputs
    class_info, band_info = _mc_cfg(feat_img)

    class _NoProba:
        classes_ = np.array([1, 2, 3])

        def predict(self, x_in):
            return np.ones(len(x_in), dtype=np.int64)

    kw = {
        "cls_cfg": class_info,
        "in_msk_img": msk_img,
        "img_msk_val": 1,
        "img_band_info": band_info,
        "use_internal_id": False,
        "use_pbar": False,
    }
    with pytest.raises(rsgispy.RSGISPyException, match="out_conf_img"):
        rcsk.apply_sklearn_classifier(
            sk_classifier=_NoProba(), out_conf_img=work / "c.tif", **kw
        )
    with pytest.raises(rsgispy.RSGISPyException, match="conf_method"):
        rcsk.apply_sklearn_classifier(
            sk_classifier=_ProbaStub([1, 2, 3]),
            output_img=work / "o.tif",
            conf_method="nope",
            **kw,
        )


@pytest.mark.parametrize("method", list(mp.CONF_METHODS))
def test_apply_classifier_raster_conf_binary(bin_inputs, method):
    """Binary engine path (used by the GBDT binary apply functions)."""
    from rsgispy.classification._cls_common import (  # noqa: PLC0415
        apply_classifier_raster,
    )

    work, feat_img, msk_img = bin_inputs
    class_info, band_info = _bin_cfg(feat_img)
    out_conf = work / f"bin_conf_{method}.tif"
    apply_classifier_raster(
        cls_cfg=class_info,
        estimator=_ProbaStub([1, 2], binary=True),
        in_msk_img=msk_img,
        img_msk_val=1,
        output_img=work / "bin_cls.tif",
        img_band_info=band_info,
        use_internal_id=False,
        binary=True,
        pos_class="Pos",
        pred_threshold=0.6,
        out_conf_img=out_conf,
        conf_method=method,
        use_pbar=False,
    )
    arr, nodata, _, _ = _read1(out_conf)
    assert nodata == _ND
    np.testing.assert_allclose(arr.ravel()[:5], _BIN_EXPECT[method], atol=1e-6)
    assert arr[1, 2] == _ND


def _gbdt_cases():
    import rsgispy.classification.classcatboost as rccb  # noqa: PLC0415
    import rsgispy.classification.classlightgbm as rclgb  # noqa: PLC0415
    import rsgispy.classification.classxgboost as rcxgb  # noqa: PLC0415

    cases = []
    if rclgb.HAVE_LIGHTGBM:
        import lightgbm  # noqa: PLC0415

        cases.append(
            (
                "lightgbm",
                rclgb,
                "lgb_classifier",
                lambda: lightgbm.LGBMClassifier(
                    n_estimators=5, min_child_samples=2, verbose=-1
                ),
            )
        )
    if rcxgb.HAVE_XGBOOST:
        import xgboost  # noqa: PLC0415

        cases.append(
            ("xgboost", rcxgb, "xgb_classifier", lambda: xgboost.XGBClassifier(n_estimators=5))
        )
    if rccb.HAVE_CATBOOST:
        import catboost  # noqa: PLC0415

        cases.append(
            (
                "catboost",
                rccb,
                "cb_classifier",
                lambda: catboost.CatBoostClassifier(
                    iterations=5, verbose=False, allow_writing_files=False
                ),
            )
        )
    return cases


@pytest.mark.parametrize("binary", [False, True])
def test_gbdt_apply_conf_passthrough(tmp_path, binary):
    """GBDT apply functions write the margin of their own score raster."""
    cases = _gbdt_cases()
    if not cases:
        pytest.skip("no GBDT backend available")
    rng = np.random.default_rng(0)
    feat = rng.uniform(0.0, 1.0, size=(2, 8, 8)).astype(np.float32)
    feat_img = _write(tmp_path / "gb_feat.tif", feat)
    msk = np.ones((8, 8), dtype=np.uint8)
    msk[0, :] = 0
    msk_img = _write(tmp_path / "gb_msk.tif", msk, nodata=0)
    n_cls = 2 if binary else 3
    x_train = rng.uniform(0.0, 1.0, size=(90, 2))
    y_train = np.minimum((x_train[:, 0] * n_cls).astype(int), n_cls - 1)
    flip = rng.random(90) < 0.2
    y_train[flip] = rng.integers(0, n_cls, size=int(flip.sum()))
    class_info = {
        f"c{i}": rcd.ClassInfoObj(out_id=i + 1, red=10 * i, green=0, blue=0)
        for i in range(n_cls)
    }
    band_info = [rcd.ImageBandInfo(feat_img, name="F", bands=[1, 2], band_names=["f1", "f2"])]
    for name, module, arg_name, make in cases:
        model = make().fit(x_train, y_train)
        func = getattr(
            module, f"apply_{name}_{'binary' if binary else 'multiclass'}_classifier"
        )
        out_scr = tmp_path / f"{name}_{binary}_scr.tif"
        out_conf = tmp_path / f"{name}_{binary}_conf.tif"
        kw = {"score_all_classes": True} if binary else {}
        func(
            cls_cfg=class_info,
            **{arg_name: model},
            in_msk_img=msk_img,
            img_msk_val=1,
            out_score_img=out_scr,
            img_band_info=band_info,
            out_conf_img=out_conf,
            use_pbar=False,
            calc_stats=False,
            **kw,
        )
        with rasterio.open(out_scr) as src:
            probs = src.read().reshape(n_cls, -1).T
        conf, nodata, _, _ = _read1(out_conf)
        assert nodata == _ND
        assert np.all(conf[0] == _ND), name
        sorted_p = np.sort(probs[8:], axis=1)
        np.testing.assert_allclose(
            conf.ravel()[8:], sorted_p[:, -1] - sorted_p[:, -2], atol=1e-6, err_msg=name
        )


# ---------------------------------------------------------------------------
# calc_confidence_reliability
# ---------------------------------------------------------------------------


def test_calc_confidence_reliability(tmp_path):
    conf = np.array([[0.05, 0.15, 0.95, 0.85], [0.55, 0.45, 0.95, _ND]], dtype=np.float32)
    cls = np.array([[1, 2, 1, 2], [1, 1, 2, 2]], dtype=np.uint8)
    conf_img = _write(tmp_path / "conf.tif", conf, nodata=_ND)
    cls_img = _write(tmp_path / "cls.tif", cls, nodata=0)
    # Pixel centres: x = 500005 + 10 * col, y = 5599995 - 10 * row.
    pts = [
        (0, 0, 2),  # conf 0.05, wrong
        (0, 1, 2),  # conf 0.15, correct
        (0, 2, 1),  # conf 0.95, correct
        (1, 2, 2),  # conf 0.95, correct
        (0, 3, 2),  # conf 0.85, correct
        (1, 0, 2),  # conf 0.55, wrong
        (1, 1, 1),  # conf 0.45, correct
        (1, 3, 2),  # conf nodata -> ignored
        (5, 5, 1),  # outside -> ignored
    ]
    gdf = gpd.GeoDataFrame(
        {"ref": [p[2] for p in pts]},
        geometry=[Point(500005.0 + 10 * c, 5599995.0 - 10 * r) for r, c, _ in pts],
        crs="EPSG:32630",
    )
    out_csv = tmp_path / "rel" / "rel.csv"
    rel = cam.calc_confidence_reliability(
        gdf, None, "ref", conf_img, cls_img, n_bins=2, out_csv_file=out_csv
    )
    assert list(rel["n_pts"]) == [3, 4]
    assert list(rel["n_correct"]) == [2, 3]
    np.testing.assert_allclose(rel["accuracy"], [2 / 3, 0.75])
    np.testing.assert_allclose(rel["mean_conf"], [(0.05 + 0.15 + 0.45) / 3, 0.825], atol=1e-6)
    assert rel.attrs["n_pts"] == 7
    assert rel.attrs["overall_accuracy"] == pytest.approx(5 / 7)
    gap = (3 * abs(2 / 3 - 0.65 / 3) + 4 * abs(0.75 - 0.825)) / 7
    assert rel.attrs["mean_abs_gap"] == pytest.approx(gap, abs=1e-6)
    assert out_csv.exists()

    rel10 = cam.calc_confidence_reliability(gdf, None, "ref", conf_img, cls_img)
    assert len(rel10) == 10
    assert int(rel10["n_pts"].sum()) == 7
    assert np.isnan(rel10.loc[2, "accuracy"])
    with pytest.raises(rsgispy.RSGISPyException, match="n_bins"):
        cam.calc_confidence_reliability(gdf, None, "ref", conf_img, cls_img, n_bins=0)
    with pytest.raises(rsgispy.RSGISPyException, match="missing"):
        cam.calc_confidence_reliability(gdf, None, "missing", conf_img, cls_img)
