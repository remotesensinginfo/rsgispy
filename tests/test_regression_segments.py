"""
Tests for rsgispy.regression.regsegments (object-based / segment regression).
"""

import json

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
import rasterio
from shapely.geometry import box

import rsgispy
import rsgispy.regression.regsegments as rseg
import rsgispy.regression.regsklearn as rsk
from rsgispy.regression.regdata import ImageBandInfo, RegressionConfig, read_reg_config
from seg_ml_scene import RES, X0, Y0, make_seg_scene

pytest.importorskip("sklearn")
from sklearn.ensemble import (  # noqa: E402  pylint: disable=wrong-import-position
    RandomForestRegressor,
)


@pytest.fixture(name="reg")
def fixture_reg(tmp_path):
    scene = make_seg_scene(str(tmp_path / "scene"))
    cfg = RegressionConfig(
        [
            ImageBandInfo(scene["s2"], name="s2", band_names=["Blue", "Green", "Red", "NIR"]),
            ImageBandInfo(scene["slope"], name="slope", band_names=["slope"]),
        ],
        target_cols="canopy_cover",
        vec_file=scene["plots"],
    )
    cfg_path = str(tmp_path / "reg.json")
    cfg = rseg.create_reg_segment_features(
        cfg, scene["segs"], str(tmp_path / "segs.parquet"), out_cfg_file=cfg_path,
        band_opts={"s2": {"no_data_val": 0}}, use_pbar=False,
    )
    return {"scene": scene, "cfg": cfg, "cfg_path": cfg_path, "dir": tmp_path}


def test_extract_reg_samples_and_aggregation(reg):
    samples = rseg.extract_reg_segment_samples(reg["cfg_path"])
    assert read_reg_config(reg["cfg_path"]).seg_info is not None
    assert samples["seg_id"].is_unique
    # Plots (2, 2) and (2, 3) are both in segment 1 -> averaged, n_ref = 2
    row = samples.set_index("seg_id").loc[1]
    assert row["n_ref"] == 2
    plots = gpd.read_file(reg["scene"]["plots"])
    assert row["canopy_cover"] == pytest.approx(plots["canopy_cover"].iloc[[0, -1]].mean())
    for col in ["canopy_cover", "train_grp", "pxl_count"] + reg["cfg"].seg_info.model_features:
        assert col in samples.columns
    med = rseg.extract_reg_segment_samples(reg["cfg_path"], reg_agg="median")
    assert len(med) == len(samples)


def test_polygon_reference_values(reg, tmp_path):
    poly = gpd.GeoDataFrame(
        {"canopy_cover": [55.0]},
        geometry=[box(X0 + 40 * RES, Y0 - 10 * RES, X0 + 50 * RES, Y0)],
        crs="EPSG:32630",
    )
    poly_path = str(tmp_path / "ref_poly.gpkg")
    poly.to_file(poly_path)
    samples = rseg.extract_reg_segment_samples(reg["cfg"], vec_file=poly_path)
    assert len(samples) == 4  # 2 x 2 segments fully covered
    assert (samples["canopy_cover"] == 55.0).all()
    assert samples["train_grp"].nunique() == 1
    # One reference polygon = one group, which cannot fill two sets
    with pytest.raises(rsgispy.RSGISPyException, match="1 group"):
        rseg.split_reg_segment_samples(samples, test_prop=0.25, valid_prop=0.0)
    tr, te, _ = rseg.split_reg_segment_samples(
        samples, test_prop=0.25, valid_prop=0.0, group_col=None, rnd_seed=0
    )
    assert (len(tr), len(te)) == (3, 1)


def test_train_apply_rasterise(reg):
    tr, te, _ = rseg.extract_and_split_reg_segment_samples(
        reg["cfg_path"], test_prop=0.3, valid_prop=0.0, rnd_seed=2
    )
    assert len(tr) > len(te) > 0
    model, metrics = rsk.train_sklearn_regressor(
        reg["cfg_path"], RandomForestRegressor(n_estimators=40, random_state=0), tr,
        test_data=te, return_model=True,
    )
    assert "test_r2" in metrics or any("r2" in k for k in metrics)
    out = str(reg["dir"] / "pred.parquet")
    rseg.apply_sklearn_regressor_segments(
        reg["cfg_path"], model, out, out_quantiles=[0.1, 0.9], chunk_size=29,
        copy_cols=[], use_pbar=False,
    )
    res = gpd.read_parquet(out)
    assert list(res.columns) == [
        "seg_id", "geometry", "pred_canopy_cover", "pred_canopy_cover_q0.1",
        "pred_canopy_cover_q0.9",
    ]
    valid = res["pred_canopy_cover"].notna()
    assert (~valid).sum() == 1  # all-no-data segment
    assert (res.loc[valid, "pred_canopy_cover_q0.1"]
            <= res.loc[valid, "pred_canopy_cover_q0.9"] + 1e-9).all()
    meta = json.loads(pq.ParquetFile(out).schema_arrow.metadata[b"rsgispy_provenance"])
    assert meta["task"] == "regression" and meta["targets"] == ["canopy_cover"]

    # Per-segment uncertainty from the spread of the trees.
    out_u = str(reg["dir"] / "pred_uncert.parquet")
    rseg.apply_sklearn_regressor_segments(
        reg["cfg_path"], model, out_u, out_uncert_method="std", chunk_size=29,
        copy_cols=[], use_pbar=False,
    )
    res_u = gpd.read_parquet(out_u)
    assert list(res_u.columns) == [
        "seg_id", "geometry", "pred_canopy_cover", "pred_canopy_cover_std",
    ]
    ok = res_u["pred_canopy_cover"].notna()
    assert res_u.loc[~ok, "pred_canopy_cover_std"].isna().all()
    assert (res_u.loc[ok, "pred_canopy_cover_std"] >= 0).all()
    assert res_u.loc[ok, "pred_canopy_cover_std"].max() > 0

    out_img = str(reg["dir"] / "pred.tif")
    rseg.rasterise_segment_regression(reg["cfg_path"], out, out_img, use_pbar=False)
    with rasterio.open(reg["scene"]["segs"]) as lsrc:
        labels = lsrc.read(1)
    with rasterio.open(out_img) as src:
        arr = src.read(1)
        assert src.descriptions == ("pred_canopy_cover",)
        assert src.nodata == -9999.0 and src.dtypes[0] == "float32"
    lut = res.set_index("seg_id")["pred_canopy_cover"].fillna(-9999.0).to_dict()
    np.testing.assert_allclose(arr, np.vectorize(lut.get)(labels), rtol=1e-6)


def test_multi_target_and_gbdt(reg):
    cfg = RegressionConfig(
        reg["cfg"].img_band_info, ["canopy_cover", "canopy_x2"],
        vec_file=reg["scene"]["plots"], seg_info=reg["cfg"].seg_info,
    )
    samples = rseg.extract_reg_segment_samples(cfg)
    model, _ = rsk.train_sklearn_regressor(
        cfg, RandomForestRegressor(n_estimators=20, random_state=0), samples,
        return_model=True,
    )
    out = str(reg["dir"] / "multi.gpkg")
    rseg.apply_sklearn_regressor_segments(cfg, model, out, copy_cols=[], use_pbar=False)
    res = gpd.read_file(out)
    ok = res["pred_canopy_cover"].notna()
    np.testing.assert_allclose(
        res.loc[ok, "pred_canopy_x2"], 2 * res.loc[ok, "pred_canopy_cover"], rtol=0.15, atol=2
    )
    out_img = str(reg["dir"] / "multi.tif")
    rseg.rasterise_segment_regression(cfg, out, out_img, use_pbar=False)
    with rasterio.open(out_img) as src:
        assert src.count == 2

    xgb = pytest.importorskip("xgboost")
    del xgb
    import rsgispy.regression.regxgboost as rxgb  # pylint: disable=import-outside-toplevel

    nan_cfg = read_reg_config(reg["cfg_path"])
    nan_cfg.seg_info.missing = "nan"
    nan_samples = rseg.extract_reg_segment_samples(nan_cfg)
    xmodel, _ = rxgb.train_xgboost_regressor(nan_cfg, nan_samples, return_model=True)
    out_x = str(reg["dir"] / "xgb.parquet")
    rseg.apply_xgboost_regressor_segments(nan_cfg, xmodel, out_x, use_pbar=False)
    assert gpd.read_parquet(out_x)["pred_canopy_cover"].notna().all()


def test_errors(reg):
    with pytest.raises(rsgispy.RSGISPyException, match="reg_agg"):
        rseg.extract_reg_segment_samples(reg["cfg_path"], reg_agg="max")
    samples = rseg.extract_reg_segment_samples(reg["cfg_path"])
    model, _ = rsk.train_sklearn_regressor(
        reg["cfg_path"], RandomForestRegressor(n_estimators=5), samples, return_model=True
    )
    from sklearn.linear_model import LinearRegression  # pylint: disable=import-outside-toplevel

    lin = LinearRegression().fit(
        samples[reg["cfg"].seg_info.model_features], samples["canopy_cover"]
    )
    with pytest.raises(rsgispy.RSGISPyException, match="tree ensemble"):
        rseg.apply_sklearn_regressor_segments(
            reg["cfg_path"], lin, str(reg["dir"] / "q.parquet"), out_quantiles=[0.5],
            use_pbar=False,
        )
    with pytest.raises(rsgispy.RSGISPyException, match="ensemble"):
        rseg.apply_sklearn_regressor_segments(
            reg["cfg_path"], lin, str(reg["dir"] / "u.parquet"),
            out_uncert_method="std", use_pbar=False,
        )
    del model
    assert isinstance(samples, pd.DataFrame)
