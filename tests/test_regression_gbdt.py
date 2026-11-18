"""
Tests for rsgispy.regression.regxgboost, reglightgbm, and regcatboost:
Optuna hyperparameter search, univariate and multivariate training with early
stopping, GPU detection & fallback warnings, persistence, and raster application.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from sklearn.multioutput import MultiOutputRegressor

import rsgispy.regression.regcatboost as rcb
import rsgispy.regression.regdata as rrd
import rsgispy.regression.reglightgbm as rlgb
import rsgispy.regression.regxgboost as rxgb


def _write_raster(
    path: Path,
    arr: np.ndarray,
    origin_x: float = 500000.0,
    origin_y: float = 5600000.0,
    pxl_size: float = 10.0,
    crs: str = "EPSG:32630",
    nodata: float = -999.0,
) -> str:
    """Write a 2D or 3D GeoTIFF raster for testing."""
    if arr.ndim == 2:
        count = 1
        height, width = arr.shape
        write_arr = arr[None, :, :]
    else:
        count, height, width = arr.shape
        write_arr = arr

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=count,
        dtype=str(arr.dtype),
        crs=crs,
        transform=from_origin(origin_x, origin_y, pxl_size, pxl_size),
        nodata=nodata,
    ) as dst:
        dst.write(write_arr)
    return str(path)


def _build_gbdt_fixture(tmp_path: Path):
    """Build synthetic regression scene and tabular train/valid/test splits."""
    h, w = 12, 12
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    b1 = 10.0 + 3.0 * xx
    b2 = 20.0 + 4.0 * yy
    b1[1, 1] = -999.0

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[1:h - 1, 1:w - 1] = 1

    img_path = _write_raster(tmp_path / "scene.tif", np.stack([b1, b2]), nodata=-999.0)
    msk_path = _write_raster(tmp_path / "mask.tif", mask, nodata=0.0)

    rng = np.random.default_rng(99)
    n_rows = 120
    x1 = rng.uniform(10.0, 45.0, size=n_rows)
    x2 = rng.uniform(20.0, 65.0, size=n_rows)
    y1 = 0.8 * x1 + 0.6 * x2 + rng.normal(0.0, 0.4, size=n_rows)
    y2 = 5.0 + 0.3 * x2 - 0.1 * x1 + rng.normal(0.0, 0.3, size=n_rows)

    df = pd.DataFrame({"B1": x1, "B2": x2, "canopy_cover": y1, "canopy_height": y2})
    train_df = df.iloc[:80].reset_index(drop=True)
    valid_df = df.iloc[80:100].reset_index(drop=True)
    test_df = df.iloc[100:].reset_index(drop=True)

    band_info = [
        rrd.ImageBandInfo(img_path, name="S2", bands=[1, 2], band_names=["B1", "B2"])
    ]
    uni_cfg = rrd.RegressionConfig(band_info, target_cols="canopy_cover")
    mv_cfg = rrd.RegressionConfig(
        band_info, target_cols=["canopy_cover", "canopy_height"]
    )
    return uni_cfg, mv_cfg, train_df, valid_df, test_df, msk_path


def test_xgboost_univariate_and_multivariate(tmp_path: Path):
    """Test XGBoost regression Optuna tuning, training, save/load, and apply."""
    pytest.importorskip("xgboost")
    pytest.importorskip("optuna")

    assert isinstance(rxgb.is_xgboost_gpu_available(), bool)
    assert "n_estimators" in rxgb.get_xgboost_optuna_param_space(extended=False)
    assert "grow_policy" in rxgb.get_xgboost_optuna_param_space(extended=True)

    uni_cfg, mv_cfg, train_df, valid_df, test_df, msk_path = _build_gbdt_fixture(
        tmp_path
    )

    _, opt_res = rxgb.optimise_xgboost_regressor(
        reg_cfg=uni_cfg,
        train_data=train_df,
        valid_data=valid_df,
        param_space={"n_estimators": {"type": "int", "low": 15, "high": 30}},
        early_stopping_rounds=5,
        n_trials=2,
        refit=False,
        return_results=True,
        rnd_seed=42,
    )
    assert "best_params" in opt_res

    model_json = tmp_path / "xgb_uni.json"
    model, metrics = rxgb.train_xgboost_regressor(
        reg_cfg=uni_cfg,
        train_data=train_df,
        test_data=test_df,
        valid_data=valid_df,
        model_params=opt_res["best_params"],
        early_stopping_rounds=5,
        out_model_file=model_json,
        rnd_seed=42,
    )
    assert model_json.exists()
    assert metrics["test_r2"] > 0.7

    reloaded = rxgb.load_xgboost_regressor(model_json)
    assert np.allclose(
        reloaded.predict(test_df[["B1", "B2"]]),
        model.predict(test_df[["B1", "B2"]]),
    )

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        out_uni = tmp_path / "xgb_uni.tif"
        rxgb.apply_xgboost_regressor(
            reg_cfg=uni_cfg,
            xgb_regressor=model_json,
            in_msk_img=msk_path,
            img_msk_val=1,
            out_img=out_uni,
            use_gpu=True,
            use_pbar=False,
        )
    with rasterio.open(out_uni) as src:
        assert src.count == 1
        assert src.descriptions == ("canopy_cover",)
        assert np.isfinite(src.read(1)[5, 5])

    # Multivariate XGBoost
    mv_model, mv_metrics = rxgb.train_xgboost_regressor(
        reg_cfg=mv_cfg,
        train_data=train_df,
        test_data=test_df,
        model_params={"n_estimators": 25, "max_depth": 3},
        rnd_seed=42,
    )
    assert mv_metrics["test_r2"] > 0.7
    out_mv = tmp_path / "xgb_mv.tif"
    rxgb.apply_xgboost_regressor(
        reg_cfg=mv_cfg,
        xgb_regressor=mv_model,
        in_msk_img=msk_path,
        img_msk_val=1,
        out_img=out_mv,
        use_pbar=False,
    )
    with rasterio.open(out_mv) as src:
        assert src.count == 2
        assert src.descriptions == ("canopy_cover", "canopy_height")


def test_lightgbm_univariate_and_multivariate(tmp_path: Path):
    """Test LightGBM regression Optuna tuning, MultiOutput wrapping, save/load, apply."""
    pytest.importorskip("lightgbm")
    pytest.importorskip("optuna")

    assert isinstance(rlgb.is_lightgbm_gpu_available(), bool)
    assert "num_leaves" in rlgb.get_lightgbm_optuna_param_space(extended=False)
    assert "extra_trees" in rlgb.get_lightgbm_optuna_param_space(extended=True)
    assert "class_weight" not in rlgb.get_lightgbm_optuna_param_space(extended=True)

    uni_cfg, mv_cfg, train_df, valid_df, test_df, msk_path = _build_gbdt_fixture(
        tmp_path
    )

    best_est = rlgb.optimise_lightgbm_regressor(
        reg_cfg=uni_cfg,
        train_data=train_df,
        valid_data=valid_df,
        param_space={
            "n_estimators": {"type": "int", "low": 15, "high": 30},
            "min_child_samples": 3,
        },
        early_stopping_rounds=5,
        n_trials=2,
        rnd_seed=42,
    )
    assert hasattr(best_est, "predict")

    # Multivariate LightGBM -> MultiOutputRegressor with per-target early stopping
    model_path = tmp_path / "lgb_mv.joblib"
    mv_model, mv_metrics = rlgb.train_lightgbm_regressor(
        reg_cfg=mv_cfg,
        train_data=train_df,
        test_data=test_df,
        valid_data=valid_df,
        model_params={"n_estimators": 30, "min_child_samples": 3},
        early_stopping_rounds=5,
        out_model_file=model_path,
        rnd_seed=42,
    )
    assert isinstance(mv_model, MultiOutputRegressor)
    assert model_path.exists()
    assert mv_metrics["test_r2"] > 0.65

    out_mv = tmp_path / "lgb_mv.tif"
    rlgb.apply_lightgbm_regressor(
        reg_cfg=mv_cfg,
        lgb_regressor=model_path,
        in_msk_img=msk_path,
        img_msk_val=1,
        out_img=out_mv,
        use_pbar=False,
    )
    with rasterio.open(out_mv) as src:
        assert src.count == 2
        assert src.descriptions == ("canopy_cover", "canopy_height")
        assert np.isfinite(src.read(1)[5, 5])


def test_catboost_univariate_and_multivariate(tmp_path: Path):
    """Test CatBoost regression Optuna tuning, MultiRMSE multivariate, save/load, apply."""
    pytest.importorskip("catboost")
    pytest.importorskip("optuna")

    assert isinstance(rcb.is_catboost_gpu_available(), bool)
    assert "iterations" in rcb.get_catboost_optuna_param_space(extended=False)
    ext_space = rcb.get_catboost_optuna_param_space(extended=True)
    assert callable(ext_space)

    uni_cfg, mv_cfg, train_df, valid_df, test_df, msk_path = _build_gbdt_fixture(
        tmp_path
    )

    _, opt_res = rcb.optimise_catboost_regressor(
        reg_cfg=uni_cfg,
        train_data=train_df,
        valid_data=valid_df,
        param_space={"iterations": {"type": "int", "low": 20, "high": 35}},
        early_stopping_rounds=5,
        n_trials=2,
        refit=False,
        return_results=True,
        rnd_seed=42,
    )
    assert "best_params" in opt_res

    model_cbm = tmp_path / "cb_uni.cbm"
    uni_model, uni_metrics = rcb.train_catboost_regressor(
        reg_cfg=uni_cfg,
        train_data=train_df,
        test_data=test_df,
        valid_data=valid_df,
        model_params=opt_res["best_params"],
        early_stopping_rounds=5,
        out_model_file=model_cbm,
        rnd_seed=42,
    )
    assert model_cbm.exists()
    assert uni_metrics["test_r2"] > 0.7

    reloaded = rcb.load_catboost_regressor(model_cbm)
    assert np.allclose(
        reloaded.predict(test_df[["B1", "B2"]]),
        uni_model.predict(test_df[["B1", "B2"]]),
    )

    # Native multivariate CatBoost (MultiRMSE)
    mv_model, mv_metrics = rcb.train_catboost_regressor(
        reg_cfg=mv_cfg,
        train_data=train_df,
        test_data=test_df,
        model_params={"iterations": 40, "depth": 4},
        rnd_seed=42,
    )
    assert mv_metrics["test_r2"] > 0.65

    out_mv = tmp_path / "cb_mv.tif"
    rcb.apply_catboost_regressor(
        reg_cfg=mv_cfg,
        cb_regressor=mv_model,
        in_msk_img=msk_path,
        img_msk_val=1,
        out_img=out_mv,
        use_pbar=False,
    )
    with rasterio.open(out_mv) as src:
        assert src.count == 2
        assert src.descriptions == ("canopy_cover", "canopy_height")
        assert np.isfinite(src.read(1)[5, 5])
