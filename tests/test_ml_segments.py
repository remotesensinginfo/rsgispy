"""
Tests for rsgispy.ml.vector (chunked vector I/O and prediction) and
rsgispy.ml.segments (segment feature creation, config and scene substitution).
"""

import json
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
import rasterio

import rsgispy
from rsgispy.classification.classdata import (
    ClassificationConfig,
    ClassInfoObj,
    ImageBandInfo,
    read_cls_config,
    write_cls_config,
)
from rsgispy.ml import segments as mseg
from rsgispy.ml import vector as mvec
from rsgispy.regression.regdata import RegressionConfig, read_reg_config, write_reg_config
from seg_ml_scene import WIDTH, SEG, make_seg_scene


@pytest.fixture(name="scene")
def fixture_scene(tmp_path):
    return make_seg_scene(str(tmp_path / "scene"))


def _img_info(scene):
    return [
        ImageBandInfo(scene["s2"], name="s2", bands=[1, 2, 3, 4],
                      band_names=["Blue", "Green", "Red", "NIR"]),
        ImageBandInfo(scene["slope"], name="slope", bands=[1], band_names=["slope"]),
    ]


def _features(scene, tmp_path, **kwargs):
    return mseg.create_segment_features(
        scene["segs"],
        _img_info(scene),
        str(tmp_path / "segs.parquet"),
        band_opts={"s2": {"no_data_val": 0}},
        use_pbar=False,
        **kwargs,
    )


def test_create_segment_features(scene, tmp_path):
    seg_info = _features(scene, tmp_path, out_table_file=str(tmp_path / "feats.parquet"))
    n_segs = (60 // SEG) * (WIDTH // SEG)
    gdf = gpd.read_parquet(seg_info.seg_vec_file)
    assert len(gdf) == n_segs
    assert gdf["seg_id"].is_monotonic_increasing
    assert gdf.crs.to_epsg() == 32630
    exp_stat_cols = [
        f"{b}_{s}"
        for b in ("Blue", "Green", "Red", "NIR", "slope")
        for s in ("mean", "stddev", "min", "max")
    ]
    assert seg_info.feature_cols == exp_stat_cols + [
        f"shp_{m}" for m in mseg.DEFAULT_SHAPE_METRICS
    ]
    for col in ["seg_id", "pxl_count", "valid"] + seg_info.feature_cols:
        assert col in gdf.columns
    assert (gdf["pxl_count"] == SEG * SEG).all()
    assert gdf["shp_area"].tolist() == pytest.approx([SEG * SEG * 100.0] * n_segs)
    # The no-data segment (rows 55-59, cols 0-4): s2 stats are no data, slope valid
    nd_seg = (55 // SEG) * (WIDTH // SEG) + 1
    row = gdf.set_index("seg_id").loc[nd_seg]
    assert row["NIR_mean"] == -9999.0
    assert row["valid"] == 1
    # Feature table is in the same order without geometry
    tbl = pd.read_parquet(seg_info.feat_table_file)
    assert "geometry" not in tbl.columns
    assert tbl["seg_id"].tolist() == gdf["seg_id"].tolist()
    assert seg_info.band_specs[0].no_data_val == 0
    assert seg_info.model_features == seg_info.feature_cols


def test_create_segment_features_gpkg_and_no_features(scene, tmp_path):
    seg_info = mseg.create_segment_features(
        scene["segs"],
        _img_info(scene)[1:],
        str(tmp_path / "segs.gpkg"),
        out_table_file=str(tmp_path / "f.parquet"),
        vec_inc_features=False,
        shape_metrics=[],
        stats=["mean", "median"],
        use_pbar=False,
    )
    gdf = gpd.read_file(seg_info.seg_vec_file)
    assert list(gdf.columns) == ["seg_id", "pxl_count", "valid", "geometry"]
    assert seg_info.feature_cols == ["slope_mean", "slope_median"]
    assert seg_info.feature_source == (seg_info.feat_table_file, None)
    with pytest.raises(rsgispy.RSGISPyException, match="out_table_file"):
        mseg.create_segment_features(
            scene["segs"], _img_info(scene), str(tmp_path / "x.parquet"),
            vec_inc_features=False, use_pbar=False,
        )


def test_multipart_segments_are_dissolved(tmp_path):
    lab = np.ones((10, 10), dtype=np.uint16)
    lab[:, 4:6] = 2  # label 1 split in two parts by label 2
    img = np.arange(100, dtype=np.float32).reshape(10, 10)
    from seg_ml_scene import _write  # pylint: disable=import-outside-toplevel

    lab_p = _write(str(tmp_path / "l.tif"), lab)
    img_p = _write(str(tmp_path / "i.tif"), img)
    seg_info = mseg.create_segment_features(
        lab_p, [img_p], str(tmp_path / "s.parquet"), use_pbar=False
    )
    gdf = gpd.read_parquet(seg_info.seg_vec_file)
    assert gdf["seg_id"].tolist() == [1, 2]
    assert gdf.geometry.iloc[0].geom_type == "MultiPolygon"
    assert gdf.loc[0, "pxl_count"] == 80


def test_config_json_round_trip(scene, tmp_path):
    cls_info = {"Water": ClassInfoObj(out_id=1, vec_file=scene["train_water"])}
    cfg = ClassificationConfig(_img_info(scene), cls_info)
    old_json = str(tmp_path / "old.json")
    write_cls_config(old_json, cfg)
    with open(old_json, encoding="utf-8") as fobj:
        assert "seg_info" not in json.load(fobj)
    assert read_cls_config(old_json).seg_info is None

    seg_info = _features(scene, tmp_path, percentiles=[10])
    seg_info.selected_features = seg_info.feature_cols[:3]
    cfg.seg_info = seg_info
    new_json = str(tmp_path / "new.json")
    write_cls_config(new_json, cfg)
    back = read_cls_config(new_json)
    assert back == cfg
    assert back.seg_info.model_features == seg_info.feature_cols[:3]
    assert back.seg_info.band_specs[0].no_data_val == 0

    reg_cfg = RegressionConfig(_img_info(scene), "canopy_cover", seg_info=seg_info)
    reg_json = str(tmp_path / "reg.json")
    write_reg_config(reg_json, reg_cfg)
    assert read_reg_config(reg_json) == reg_cfg
    with pytest.raises(rsgispy.RSGISPyException, match="poly_assign"):
        mseg.SegmentFeatureConfig(poly_assign="nope")


def test_scene_substitution(scene, tmp_path):
    scene_b = make_seg_scene(str(tmp_path / "scene_b"), seed=5, tag="b")
    seg_info = _features(scene, tmp_path)
    seg_info.selected_features = ["NIR_mean", "Red_mean", "shp_area"]
    cfg = ClassificationConfig(
        _img_info(scene),
        {"Water": ClassInfoObj(out_id=1, vec_file=scene["train_water"])},
        seg_info=seg_info,
    )
    new_cfg = mseg.substitute_scene(
        cfg, {"s2": scene_b["s2"], "slope": scene_b["slope"]}, label_img=scene_b["segs"]
    )
    assert new_cfg.img_band_info[0].file_name == scene_b["s2"]
    assert cfg.img_band_info[0].file_name == scene["s2"]  # original unchanged
    new_seg = mseg.create_segment_features(
        new_cfg.seg_info.label_img,
        None,
        str(tmp_path / "segs_b.parquet"),
        template=new_cfg.seg_info,
        use_pbar=False,
    )
    assert new_seg.feature_cols == seg_info.feature_cols
    assert new_seg.model_features == ["NIR_mean", "Red_mean", "shp_area"]
    assert new_seg.band_specs[0].file_name == scene_b["s2"]
    gdf_b = gpd.read_parquet(new_seg.seg_vec_file)
    assert set(seg_info.feature_cols) <= set(gdf_b.columns)

    with pytest.raises(rsgispy.RSGISPyException, match="No new image path"):
        mseg.substitute_scene(cfg, {"s2": scene_b["s2"]})
    with pytest.raises(rsgispy.RSGISPyException, match="unknown image"):
        mseg.substitute_scene(
            cfg, {"s2": scene_b["s2"], "slope": scene_b["slope"], "x": "y.tif"}
        )
    # An image with too few bands is rejected
    with pytest.raises(rsgispy.RSGISPyException, match="bands"):
        mseg.substitute_scene(cfg, {"s2": scene_b["slope"], "slope": scene_b["slope"]})
    # A template whose model feature cannot be produced is rejected
    bad = new_cfg.seg_info.copy()
    bad.selected_features = ["NIR_mean", "not_a_feature"]
    with pytest.raises(rsgispy.RSGISPyException, match="not_a_feature"):
        mseg.create_segment_features(
            bad.label_img, None, str(tmp_path / "bad.parquet"), template=bad,
            use_pbar=False,
        )


class _SumModel:
    """Minimal estimator summing the first two features."""

    def predict(self, x_arr):
        return np.asarray(x_arr)[:, 0] + np.asarray(x_arr)[:, 1]


@pytest.mark.parametrize("chunk_size", [1, 7, 50, 1000])
@pytest.mark.parametrize("out_ext", ["parquet", "gpkg"])
def test_predict_vector_chunk_invariance(scene, tmp_path, chunk_size, out_ext):
    seg_info = _features(scene, tmp_path)
    out = str(tmp_path / f"pred.{out_ext}")
    mvec.predict_vector(
        _SumModel(),
        seg_info.seg_vec_file,
        out,
        ["NIR_mean", "Red_mean"],
        id_col="seg_id",
        valid_col="valid",
        missing="drop",
        feat_no_data_val=-9999.0,
        out_no_data_val=-1.0,
        chunk_size=chunk_size,
        copy_cols=[],
        metadata=mvec.build_provenance(model=_SumModel(), feature_cols=["NIR_mean"]),
        use_pbar=False,
    )
    res = gpd.read_file(out) if out_ext == "gpkg" else gpd.read_parquet(out)
    src = gpd.read_parquet(seg_info.seg_vec_file)
    assert list(res.columns[:1]) == ["seg_id"] and "prediction" in res.columns
    assert "NIR_mean" not in res.columns  # copy_cols=[] keeps only the ID
    exp = np.where(
        src["NIR_mean"] == -9999.0, -1.0, src["NIR_mean"] + src["Red_mean"]
    )
    np.testing.assert_allclose(res["prediction"], exp)
    assert all(res.geometry.geom_equals(src.geometry))
    if out_ext == "parquet":
        meta = pq.ParquetFile(out).schema_arrow.metadata
        assert json.loads(meta[b"rsgispy_provenance"])["feature_cols"] == ["NIR_mean"]
        assert b"geo" in meta


def test_predict_vector_table_with_geometry_layer(scene, tmp_path):
    seg_info = _features(scene, tmp_path, out_table_file=str(tmp_path / "t.parquet"))
    rev = gpd.read_parquet(seg_info.seg_vec_file)[["seg_id", "geometry"]].iloc[::-1]
    rev_path = str(tmp_path / "rev.parquet")
    rev.to_parquet(rev_path)
    for geom_file in (seg_info.seg_vec_file, rev_path):
        out = str(tmp_path / "p.parquet")
        mvec.predict_vector(
            _SumModel(), seg_info.feat_table_file, out, ["NIR_mean", "Red_mean"],
            id_col="seg_id", geom_vec_file=geom_file, copy_cols=[], chunk_size=13,
            overwrite=True, use_pbar=False,
        )
        res = gpd.read_parquet(out).set_index("seg_id").sort_index()
        tbl = pd.read_parquet(seg_info.feat_table_file).set_index("seg_id")
        np.testing.assert_allclose(res["prediction"], tbl["NIR_mean"] + tbl["Red_mean"])


def test_only_requested_columns_are_read(scene, tmp_path, monkeypatch):
    seg_info = _features(scene, tmp_path)
    calls = []
    orig = pq.ParquetFile.iter_batches

    def spy(self, *args, **kwargs):
        calls.append(kwargs.get("columns"))
        return orig(self, *args, **kwargs)

    monkeypatch.setattr(pq.ParquetFile, "iter_batches", spy)
    chunks = list(
        mvec.iter_vec_chunks(seg_info.seg_vec_file, ["seg_id", "NIR_mean"], chunk_size=40)
    )
    assert calls == [["seg_id", "NIR_mean"]]
    assert sum(len(c) for c in chunks) == 192
    assert all(list(c.columns) == ["seg_id", "NIR_mean"] for c in chunks)
    calls.clear()
    mvec.predict_vector(
        _SumModel(), seg_info.seg_vec_file, str(tmp_path / "o.parquet"),
        ["NIR_mean", "Red_mean"], id_col="seg_id", valid_col="valid",
        copy_cols=[], use_pbar=False,
    )
    assert calls == [["seg_id", "valid", "NIR_mean", "Red_mean", "geometry"]]


def test_failed_write_leaves_no_output(scene, tmp_path):
    seg_info = _features(scene, tmp_path)
    out = str(tmp_path / "fail.parquet")

    def boom(df):
        raise ValueError("model failure")

    with pytest.raises(ValueError, match="model failure"):
        mvec.map_vector_chunks(
            seg_info.seg_vec_file, out, ["NIR_mean"], boom, use_pbar=False
        )
    assert not os.path.exists(out)
    assert [f for f in os.listdir(tmp_path) if f.startswith(".fail")] == []


def test_prepare_feature_matrix_policies():
    df = pd.DataFrame(
        {"a": [1.0, np.nan, -9999.0, 4.0], "b": [1.0, 2.0, 3.0, 4.0], "v": [1, 1, 1, 0]}
    )
    x_df, ok = mvec.prepare_feature_matrix(df, ["a", "b"], "v", "nan", feat_no_data_val=-9999)
    assert ok.tolist() == [True, True, True, False]
    assert np.isnan(x_df["a"].iloc[1]) and np.isnan(x_df["a"].iloc[2])
    x_df, ok = mvec.prepare_feature_matrix(df, ["a", "b"], "v", "drop", feat_no_data_val=-9999)
    assert ok.tolist() == [True, False, False, False]
    x_df, ok = mvec.prepare_feature_matrix(df, ["a", "b"], None, "fill", fill_value=7.0,
                                           feat_no_data_val=-9999)
    assert x_df["a"].tolist() == [1.0, 7.0, 7.0, 4.0]
    with pytest.raises(rsgispy.RSGISPyException, match="missing"):
        mvec.prepare_feature_matrix(df, ["a"], None, "bad")
    with pytest.raises(rsgispy.RSGISPyException, match="'c'"):
        mvec.prepare_feature_matrix(df, ["c"])


def test_read_vec_layer_info_and_label_raster(scene, tmp_path):
    seg_info = _features(scene, tmp_path)
    info = mvec.read_vec_layer_info(seg_info.seg_vec_file)
    assert info.n_features == 192 and info.geom_col == "geometry"
    assert "NIR_mean" in info.columns
    with rasterio.open(scene["segs"]) as src:
        assert src.dtypes[0] == "uint32"
