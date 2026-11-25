"""
Tests for rsgispy.classification.classsegments (object-based / segment
classification).
"""

import json
import os
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
import rasterio

import rsgispy
import rsgispy.classification.classsegments as cseg
import rsgispy.classification.classsklearn as csk
from rsgispy.classification.classaccuracymetrics import calc_acc_stats_ptonly
from rsgispy.classification.classdata import (
    ClassificationConfig,
    ClassInfoObj,
    ImageBandInfo,
    read_cls_config,
)
from rsgispy.ml.scaling import FeatureScaler
from seg_ml_scene import CLASS_IDS, make_seg_scene, pix_centre, seg_class_truth

sklearn = pytest.importorskip("sklearn")
from sklearn.cluster import KMeans  # noqa: E402  pylint: disable=wrong-import-position
from sklearn.ensemble import RandomForestClassifier  # noqa: E402

COLOURS = {"Water": "#0000ff", "Forest": "#228b22", "Grass": "#adff2f", "Urban": "#808080"}


def _cfg(scene, classes=None, grass_key="train_grass"):
    classes = classes or list(CLASS_IDS)
    cls_info = {
        name: ClassInfoObj(
            out_id=10 * CLASS_IDS[name],
            vec_file=scene[grass_key if name == "Grass" else f"train_{name.lower()}"],
            colour=COLOURS[name],
        )
        for name in classes
    }
    return ClassificationConfig(
        [
            ImageBandInfo(scene["s2"], name="s2", band_names=["Blue", "Green", "Red", "NIR"]),
            ImageBandInfo(scene["slope"], name="slope", band_names=["slope"]),
        ],
        cls_info,
    )


@pytest.fixture(name="seg_cfg")
def fixture_seg_cfg(tmp_path):
    scene = make_seg_scene(str(tmp_path / "scene"))
    cfg_path = str(tmp_path / "cfg.json")
    cfg = cseg.create_cls_segment_features(
        _cfg(scene),
        scene["segs"],
        str(tmp_path / "segs.parquet"),
        out_cfg_file=cfg_path,
        band_opts={"s2": {"no_data_val": 0}},
        use_pbar=False,
    )
    return {"scene": scene, "cfg": cfg, "cfg_path": cfg_path, "dir": tmp_path}


def _train(seg_cfg):
    samples = cseg.extract_class_segment_samples(seg_cfg["cfg_path"])
    model, _ = csk.train_sklearn_classifier(
        seg_cfg["cfg_path"],
        RandomForestClassifier(n_estimators=30, random_state=1),
        samples,
        return_model=True,
    )
    return model, samples


def test_config_written_with_seg_info(seg_cfg):
    back = read_cls_config(seg_cfg["cfg_path"])
    assert back.seg_info is not None
    assert back.seg_info.label_img == os.path.abspath(seg_cfg["scene"]["segs"])
    assert "NIR_mean" in back.seg_info.feature_cols


def test_extract_samples_dedup_and_schema(seg_cfg):
    samples = cseg.extract_class_segment_samples(seg_cfg["cfg_path"])
    rep = samples.attrs["report"]
    # Water: 6 points, 2 in segment 1 -> 5 samples; seg 1 has 2 hits
    assert rep["n_samples_per_class"]["Water"] == 5
    assert samples.loc[samples["seg_id"] == 1, "n_hits"].item() == 2
    assert samples["seg_id"].is_unique
    for col in ("out_cls_id", "internal_cls_id", "class_name", "train_grp", "pxl_count"):
        assert col in samples.columns
    truth = seg_class_truth()
    assert (samples["out_cls_id"] // 10 == samples["seg_id"].map(truth)).all()
    assert set(samples["out_cls_id"]) == {10, 20, 30, 40}
    assert samples["internal_cls_id"].between(0, 3).all()


def test_polygon_assignment_rules(seg_cfg, tmp_path):
    scene = seg_cfg["scene"]
    cfg = seg_cfg["cfg"]
    seg = cfg.seg_info.copy()
    poly_cfg = ClassificationConfig(
        cfg.img_band_info,
        {
            "Grass": ClassInfoObj(out_id=30, vec_file=scene["train_grass_poly"]),
            "Water": ClassInfoObj(out_id=10, vec_file=scene["train_water"]),
        },
        seg_info=seg,
    )
    s_overlap = cseg.extract_class_segment_samples(poly_cfg)
    grass = s_overlap[s_overlap["class_name"] == "Grass"]
    assert len(grass) == 2  # 100%, 100% (and a 40% segment excluded)
    assert grass["train_grp"].nunique() == 1  # same polygon -> same group
    s_low = cseg.extract_class_segment_samples(poly_cfg, min_overlap=0.3)
    assert (s_low["class_name"] == "Grass").sum() == 3
    s_cent = cseg.extract_class_segment_samples(poly_cfg, poly_assign="centroid")
    assert (s_cent["class_name"] == "Grass").sum() == 2


def test_conflicts_and_reprojection(seg_cfg, tmp_path):
    scene = seg_cfg["scene"]
    cfg = seg_cfg["cfg"]
    # Two Forest points in Water segment 1 (which has 2 Water points)
    conf = gpd.GeoDataFrame(
        {"c": [1, 1]}, geometry=[pix_centre(1, 1), pix_centre(3, 3)], crs="EPSG:32630"
    )
    conf_path = str(tmp_path / "conf.gpkg")
    conf.to_file(conf_path)
    conf_cfg = ClassificationConfig(
        cfg.img_band_info,
        {
            "Water": ClassInfoObj(out_id=10, vec_file=scene["train_water"]),
            "Forest": ClassInfoObj(out_id=20, vec_file=conf_path),
        },
        seg_info=cfg.seg_info,
    )
    with pytest.warns(UserWarning, match="more than one class"):
        dropped = cseg.extract_class_segment_samples(conf_cfg)
    assert 1 not in dropped["seg_id"].tolist()
    assert dropped.attrs["report"]["n_conflict_segments"] == 1
    # Majority: 2 Water vs 2 Forest -> tie -> dropped; add a third Forest
    conf3 = pd.concat([conf, conf.iloc[[0]]])
    conf3.to_file(conf_path)
    with pytest.warns(UserWarning):
        major = cseg.extract_class_segment_samples(conf_cfg, conflict="majority")
    assert major.loc[major["seg_id"] == 1, "class_name"].item() == "Forest"

    wgs_path = str(tmp_path / "water_wgs84.gpkg")
    gpd.read_file(scene["train_water"]).to_crs("EPSG:4326").to_file(wgs_path)
    wgs_cfg = ClassificationConfig(
        cfg.img_band_info,
        {
            "Water": ClassInfoObj(out_id=10, vec_file=wgs_path),
            "Forest": ClassInfoObj(out_id=20, vec_file=scene["train_forest"]),
        },
        seg_info=cfg.seg_info,
    )
    with pytest.raises(rsgispy.RSGISPyException, match="reproj_vec"):
        cseg.extract_class_segment_samples(wgs_cfg)
    reproj = cseg.extract_class_segment_samples(wgs_cfg, reproj_vec=True)
    assert (reproj["class_name"] == "Water").sum() == 5


def test_split_group_aware_and_small_classes(seg_cfg):
    samples = cseg.extract_class_segment_samples(seg_cfg["cfg_path"])
    two_water = samples.loc[samples["class_name"] == "Water", "seg_id"].iloc[:2]
    samples.loc[samples["seg_id"].isin(two_water), "train_grp"] = "g"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        train, test, valid = cseg.split_class_segment_samples(
            samples, test_prop=0.2, valid_prop=0.2, rnd_seed=1
        )
    assert len(train) + len(test) + len(valid) == len(samples)
    for part in (train, test, valid):
        assert set(part["class_name"]) == set(CLASS_IDS)
    grp_sets = [set(p.loc[p["train_grp"] == "g", "seg_id"]) for p in (train, test, valid)]
    assert sum(1 for g in grp_sets if g) == 1  # group kept together
    tiny = samples[samples["class_name"] != "Water"].copy()
    tiny = pd.concat([tiny, samples[samples["class_name"] == "Water"].iloc[:2]])
    with pytest.raises(rsgispy.RSGISPyException, match="fewer than"):
        cseg.split_class_segment_samples(tiny, test_prop=0.2, valid_prop=0.2)
    with pytest.warns(UserWarning, match="only"):
        cseg.split_class_segment_samples(samples, test_prop=0.2, valid_prop=0.0)


@pytest.mark.parametrize("out_ext", ["parquet", "gpkg"])
def test_apply_rasterise_and_accuracy(seg_cfg, out_ext):
    model, _ = _train(seg_cfg)
    out_vec = str(seg_cfg["dir"] / f"cls.{out_ext}")
    cseg.apply_sklearn_classifier_segments(
        seg_cfg["cfg_path"], model, out_vec, out_probs="all", chunk_size=37, use_pbar=False
    )
    res = gpd.read_file(out_vec) if out_ext == "gpkg" else gpd.read_parquet(out_vec)
    truth = seg_class_truth()
    nd_seg = (55 // 5) * 16 + 1  # all-no-data segment (missing='drop')
    res = res.set_index("seg_id")
    assert res.loc[nd_seg, "cls_id"] == 0 and pd.isna(res.loc[nd_seg, "cls_name"])
    others = res.drop(index=nd_seg)
    assert (others["cls_id"] // 10 == others.index.map(truth)).mean() > 0.95
    row = others.loc[others["cls_id"] == 20].iloc[0]
    assert row["cls_name"] == "Forest" and row["cls_colour"] == "#228b22"
    assert (row["cls_red"], row["cls_green"], row["cls_blue"]) == (34, 139, 34)
    prob_cols = [f"cls_prob_{n}" for n in CLASS_IDS]
    np.testing.assert_allclose(others[prob_cols].sum(axis=1), 1.0, rtol=1e-5)
    assert np.isnan(res.loc[nd_seg, "cls_prob_Water"])
    assert "NIR_mean" in res.columns  # copy_cols default copies the input columns

    out_img = str(seg_cfg["dir"] / f"cls_{out_ext}.tif")
    out_prob = str(seg_cfg["dir"] / f"prob_{out_ext}.tif")
    cseg.rasterise_segment_classification(
        seg_cfg["cfg_path"], out_vec, out_img, out_score_img=out_prob, use_pbar=False
    )
    with rasterio.open(seg_cfg["scene"]["segs"]) as lsrc:
        labels = lsrc.read(1)
    with rasterio.open(out_img) as src:
        arr = src.read(1)
        assert src.dtypes[0] == "uint8" and src.nodata == 0
        assert src.colormap(1)[10][:3] == (0, 0, 255)
    lut = res["cls_id"].to_dict()
    np.testing.assert_array_equal(arr, np.vectorize(lut.get)(labels))
    with rasterio.open(out_prob) as src:
        assert src.count == 4 and src.descriptions[0] == "prob_Water"

    pts = cseg.pop_segment_cls_accuracy_pts(
        seg_cfg["cfg_path"], out_vec, seg_cfg["scene"]["acc_pts"]
    )
    pts["ref_out"] = pts["ref_cls"] * 10
    stats = calc_acc_stats_ptonly(pts, None, "ref_out", "seg_cls")
    assert stats is not None and "confusion_matrix" in json.dumps(list(stats.keys()))


def test_apply_chunk_invariance_recoding_and_provenance(seg_cfg):
    model, _ = _train(seg_cfg)
    outs = []
    for chunk in (1, 13, 1000):
        out = str(seg_cfg["dir"] / f"c{chunk}.parquet")
        cseg.apply_sklearn_classifier_segments(
            seg_cfg["cfg_path"], model, out, out_probs="max", chunk_size=chunk,
            copy_cols=[], use_pbar=False,
        )
        outs.append(gpd.read_parquet(out))
    for other in outs[1:]:
        pd.testing.assert_frame_equal(
            pd.DataFrame(outs[0].drop(columns="geometry")),
            pd.DataFrame(other.drop(columns="geometry")),
        )
    assert list(outs[0].columns) == [
        "seg_id", "geometry", "cls_id", "cls_name", "cls_colour", "cls_red",
        "cls_green", "cls_blue", "cls_prob_max",
    ]
    meta = json.loads(
        pq.ParquetFile(seg_cfg["dir"] / "c1.parquet").schema_arrow.metadata[
            b"rsgispy_provenance"
        ]
    )
    assert meta["task"] == "classification"
    assert meta["classes"]["Forest"] == 20
    assert meta["feature_cols"] == list(model.feature_names_in_)
    # Without recoding, internal IDs are written
    out_int = str(seg_cfg["dir"] / "int.parquet")
    cseg.apply_sklearn_classifier_segments(
        seg_cfg["cfg_path"], model, out_int, recode_to_out_ids=False,
        out_no_data_val=99, use_pbar=False,
    )
    ids = set(gpd.read_parquet(out_int)["cls_id"])
    assert ids <= {0, 1, 2, 3, 99} and 99 in ids


def test_binary_threshold(seg_cfg):
    cfg = seg_cfg["cfg"]
    bin_cfg = ClassificationConfig(
        cfg.img_band_info,
        {n: cfg.class_info[n] for n in ("Water", "Forest")},
        seg_info=cfg.seg_info,
    )
    samples = cseg.extract_class_segment_samples(bin_cfg)
    model, _ = csk.train_sklearn_classifier(
        bin_cfg, RandomForestClassifier(n_estimators=20, random_state=0), samples,
        return_model=True,
    )
    out = str(seg_cfg["dir"] / "bin.parquet")
    from rsgispy.classification._cls_common import (  # pylint: disable=import-outside-toplevel
        apply_classifier_vector,
    )

    apply_classifier_vector(
        bin_cfg, model, out, binary=True, pos_class="Forest", pred_threshold=0.99,
        out_probs="pos", use_pbar=False,
    )
    res = gpd.read_parquet(out)
    assert "cls_prob_Forest" in res.columns
    forest = res["cls_prob_Forest"] >= 0.99
    assert (res.loc[forest, "cls_id"] == 20).all()
    assert (res.loc[~forest & res["cls_prob_Forest"].notna(), "cls_id"] == 10).all()


@pytest.mark.parametrize("backend", ["xgboost", "lightgbm", "catboost"])
def test_gbdt_segments_with_nan_features(seg_cfg, backend):
    pytest.importorskip(backend)
    import importlib  # pylint: disable=import-outside-toplevel

    mod = importlib.import_module(f"rsgispy.classification.class{backend}")
    cfg = read_cls_config(seg_cfg["cfg_path"])
    cfg.seg_info.missing = "nan"
    samples = cseg.extract_class_segment_samples(cfg)
    # Segment workflows give few samples: allow small leaves (LightGBM's
    # default min_data_in_leaf=20 would prevent any split here).
    params = {"min_child_samples": 2} if backend == "lightgbm" else None
    model, _ = getattr(mod, f"train_{backend}_multiclass_classifier")(
        cfg, samples, model_params=params, return_model=True
    )
    out = str(seg_cfg["dir"] / f"{backend}.parquet")
    getattr(cseg, f"apply_{backend}_multiclass_classifier_segments")(
        cfg, model, out, out_probs="max", use_pbar=False
    )
    res = gpd.read_parquet(out).set_index("seg_id")
    # With missing='nan' the no-data segment is still classified
    assert (res["cls_id"] != 0).all()
    truth = seg_class_truth()
    assert (res["cls_id"] // 10 == res.index.map(truth)).mean() > 0.9


def test_cluster_segments(seg_cfg):
    out = str(seg_cfg["dir"] / "clusters.parquet")
    km = cseg.cluster_segments(
        seg_cfg["cfg_path"], KMeans(n_clusters=4, n_init=4, random_state=0), out,
        scaler=FeatureScaler(method="zscore"), n_sample=150, rnd_seed=1, use_pbar=False,
    )
    assert hasattr(km, "cluster_centers_")
    res = gpd.read_parquet(out)
    assert set(res["cluster_id"]) <= {0, 1, 2, 3, 4}
    assert (res["cluster_id"] == 0).sum() == 1  # the no-data segment
    out_img = str(seg_cfg["dir"] / "clusters.tif")
    cseg.rasterise_segment_classification(
        None, out, out_img, cls_col="cluster_id", label_img=seg_cfg["scene"]["segs"],
        use_pbar=False,
    )
    with rasterio.open(out_img) as src:
        assert set(np.unique(src.read(1))) == set(res["cluster_id"])
        assert src.colormap(1) is not None


def test_requires_seg_info(seg_cfg):
    plain = _cfg(seg_cfg["scene"])
    with pytest.raises(rsgispy.RSGISPyException, match="seg_info"):
        cseg.extract_class_segment_samples(plain)
