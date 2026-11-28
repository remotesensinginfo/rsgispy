"""
Tests for rsgispy.regression.regsklearn: standard parameter spaces,
Grid/Random/Optuna optimisation, univariate and multivariate training (native
and MultiOutputRegressor auto-wrapping), persistence, and block-wise raster
prediction.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

import rsgispy
import rsgispy.imageutils.colourtable as rct
import rsgispy.regression.regdata as rrd
import rsgispy.regression.regsklearn as rsk


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


def _build_reg_scene(tmp_path: Path):
    """Build synthetic multi-resolution regression scene and tabular splits."""
    h, w = 16, 16
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    b1 = 10.0 + 3.0 * xx
    b2 = 20.0 + 4.0 * yy
    b1[2, 2] = -999.0  # injected nodata inside ROI mask

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[1:h - 1, 1:w - 1] = 1

    coarse = 5.0 + 2.0 * np.mgrid[0:8, 0:8][0].astype(np.float32)

    s2_img = _write_raster(tmp_path / "s2.tif", np.stack([b1, b2]), nodata=-999.0)
    coarse_img = _write_raster(
        tmp_path / "coarse.tif", coarse, pxl_size=20.0, nodata=-999.0
    )
    msk_img = _write_raster(tmp_path / "mask.tif", mask, nodata=0.0)

    rng = np.random.default_rng(123)
    n_tot = 120
    f1 = rng.uniform(10.0, 55.0, size=n_tot)
    f2 = rng.uniform(20.0, 80.0, size=n_tot)
    f3 = rng.uniform(5.0, 20.0, size=n_tot)
    y1 = np.clip(
        0.9 * f1 + 0.5 * f2 - 0.3 * f3 + rng.normal(0.0, 0.4, n_tot), 0.0, 100.0
    )
    y2 = 2.0 + 0.25 * f2 + 0.1 * f3 + rng.normal(0.0, 0.2, n_tot)

    full_df = pd.DataFrame(
        {
            "B1": f1,
            "B2": f2,
            "C1": f3,
            "canopy_cover": y1,
            "canopy_height": y2,
        }
    )
    train_df = full_df.iloc[:80].reset_index(drop=True)
    valid_df = full_df.iloc[80:100].reset_index(drop=True)
    test_df = full_df.iloc[100:].reset_index(drop=True)

    img_band_info = [
        rrd.ImageBandInfo(s2_img, name="S2", bands=[1, 2], band_names=["B1", "B2"]),
        rrd.ImageBandInfo(coarse_img, name="Coarse", bands=[1], band_names=["C1"]),
    ]
    uni_cfg = rrd.RegressionConfig(img_band_info=img_band_info, target_cols="canopy_cover")
    mv_cfg = rrd.RegressionConfig(
        img_band_info=img_band_info,
        target_cols=["canopy_cover", "canopy_height"],
    )
    return uni_cfg, mv_cfg, train_df, valid_df, test_df, msk_img


def test_built_in_regressor_param_spaces():
    """Verify get_sklearn_regressor_param_space covers aliases and wrappers."""
    aliases = (
        "rf", "et", "gb", "hgb", "svm", "mlp", "knn",
        "ridge", "lasso", "enet", "dt", "ada",
    )
    for alias in aliases:
        opt_def = rsk.get_sklearn_regressor_param_space(alias, search_method="optuna")
        opt_ext = rsk.get_sklearn_regressor_param_space(
            alias, search_method="optuna", extended=True
        )
        cv_def = rsk.get_sklearn_regressor_param_space(alias, search_method="grid")
        assert callable(opt_def) or isinstance(opt_def, dict)
        assert callable(opt_ext) or isinstance(opt_ext, dict)
        assert isinstance(cv_def, dict) and cv_def

    pipe = Pipeline([("scaler", StandardScaler()), ("reg", Ridge())])
    pipe_space = rsk.get_sklearn_regressor_param_space(pipe, search_method="grid")
    assert "reg__alpha" in pipe_space

    mv = MultiOutputRegressor(SVR())
    mv_space = rsk.get_sklearn_regressor_param_space(mv, search_method="random")
    assert "estimator__C" in mv_space

    with pytest.raises(rsgispy.RSGISPyException):
        rsk.get_sklearn_regressor_param_space("unknown_regressor")


def test_sklearn_univariate_search_train_and_apply(tmp_path: Path):
    """Test Optuna/Grid search, persistence, training, and single-band raster apply."""
    uni_cfg, _, train_df, valid_df, test_df, msk_img = _build_reg_scene(tmp_path)

    # Grid search using custom dict
    gs_est, gs_res = rsk.perform_sklearn_regressor_param_search(
        reg_cfg=uni_cfg,
        sk_regressor=RandomForestRegressor(random_state=42),
        train_data=train_df,
        search_method="grid",
        param_space={"n_estimators": [10, 20], "max_depth": [3, 6]},
        scoring="r2",
        cv=3,
        return_results=True,
        rnd_seed=42,
    )
    assert gs_res["search_method"] == "grid"
    assert hasattr(gs_est, "predict")

    # Optuna search using built-in default space + validation split
    params_json = tmp_path / "rf_best.json"
    results_csv = tmp_path / "rf_trials.csv"
    best_rf, opt_res = rsk.perform_sklearn_regressor_param_search(
        reg_cfg=uni_cfg,
        sk_regressor=RandomForestRegressor(random_state=42),
        train_data=train_df,
        valid_data=valid_df,
        search_method="optuna",
        param_space=None,
        n_trials=3,
        refit=False,
        out_results_file=results_csv,
        out_params_json=params_json,
        return_results=True,
        rnd_seed=42,
    )
    assert results_csv.exists() and params_json.exists()
    assert opt_res["param_space_mode"] == "default"
    saved_meta = json.loads(params_json.read_text(encoding="utf-8"))
    assert "best_params" in saved_meta

    # Train & persist
    model_file = tmp_path / "rf_model.joblib"
    metrics_file = tmp_path / "rf_metrics.json"
    fitted_rf, metrics = rsk.train_sklearn_regressor(
        reg_cfg=uni_cfg,
        sk_regressor=best_rf,
        train_data=train_df,
        test_data=test_df,
        valid_data=valid_df,
        out_model_file=model_file,
        out_metrics_file=metrics_file,
        return_model=True,
        rnd_seed=42,
    )
    assert model_file.exists() and metrics_file.exists()
    assert metrics["train_r2"] > 0.7
    assert metrics["test_r2"] > 0.6
    assert "valid_rmse" in metrics

    loaded = rsk.load_sklearn_regressor(model_file)
    assert np.allclose(
        loaded.predict(test_df[["B1", "B2", "C1"]]),
        fitted_rf.predict(test_df[["B1", "B2", "C1"]]),
    )

    # Apply to multi-resolution Landsat/Coarse scene
    out_tif = tmp_path / "canopy_pred.tif"
    rsk.apply_sklearn_regressor(
        reg_cfg=uni_cfg,
        sk_regressor=model_file,
        in_msk_img=msk_img,
        img_msk_val=1,
        output_img=out_tif,
        out_no_data_val=-9999.0,
        block_size=8,
        use_pbar=False,
    )
    assert out_tif.exists()
    with rasterio.open(out_tif) as src:
        assert src.count == 1
        assert src.dtypes[0] == "float32"
        assert src.descriptions == ("canopy_cover",)
        arr = src.read(1)
        assert arr[0, 0] == pytest.approx(-9999.0)
        assert arr[2, 2] == pytest.approx(-9999.0)
        assert np.isfinite(arr[5, 5]) and arr[5, 5] > 0.0
    assert not rct.is_img_thematic(str(out_tif))


def test_sklearn_multivariate_native_and_autowrap(tmp_path: Path):
    """Test multivariate training/apply for native ET and auto-wrapped GBR/SVR."""
    _, mv_cfg, train_df, valid_df, test_df, msk_img = _build_reg_scene(tmp_path)

    # Native multi-output regressor (ExtraTreesRegressor)
    mv_et, et_metrics = rsk.train_sklearn_regressor(
        reg_cfg=mv_cfg,
        sk_regressor=ExtraTreesRegressor(n_estimators=25, random_state=42),
        train_data=train_df,
        test_data=test_df,
        return_model=True,
    )
    assert "test_canopy_cover_r2" in et_metrics
    assert "test_canopy_height_rmse" in et_metrics
    assert et_metrics["test_r2"] > 0.7

    # Apply multivariate model -> 2-band GeoTIFF
    out_mv_tif = tmp_path / "forest_structure.tif"
    rsk.apply_sklearn_regressor(
        reg_cfg=mv_cfg,
        sk_regressor=mv_et,
        in_msk_img=msk_img,
        img_msk_val=1,
        output_img=out_mv_tif,
        use_pbar=False,
    )
    with rasterio.open(out_mv_tif) as src:
        assert src.count == 2
        assert src.descriptions == ("canopy_cover", "canopy_height")
        b1 = src.read(1)
        b2 = src.read(2)
        assert b1[0, 0] == pytest.approx(-9999.0)
        assert b2[2, 2] == pytest.approx(-9999.0)
        assert b1[5, 5] > 0.0 and b2[5, 5] > 0.0

    # Non-native multi-output regressor (GradientBoostingRegressor) -> auto-wrapped
    wrapped_gbr, gbr_metrics = rsk.train_sklearn_regressor(
        reg_cfg=mv_cfg,
        sk_regressor=GradientBoostingRegressor(n_estimators=25, random_state=42),
        train_data=train_df,
        test_data=test_df,
        return_model=True,
    )
    assert isinstance(wrapped_gbr, MultiOutputRegressor)
    assert gbr_metrics["test_r2"] > 0.7

    # Auto-wrap with return_model=False raises clear exception
    with pytest.raises(rsgispy.RSGISPyException):
        rsk.train_sklearn_regressor(
            reg_cfg=mv_cfg,
            sk_regressor=GradientBoostingRegressor(n_estimators=15),
            train_data=train_df,
            return_model=False,
        )

    # Search auto-wraps SVR & auto-prefixes unprefixed custom dict
    searched_svr = rsk.perform_sklearn_regressor_param_search(
        reg_cfg=mv_cfg,
        sk_regressor=SVR(),
        train_data=train_df,
        valid_data=valid_df,
        search_method="random",
        param_space={"C": [1.0, 10.0], "kernel": ["rbf"]},
        n_iter=2,
        rnd_seed=42,
    )
    assert isinstance(searched_svr, MultiOutputRegressor)


def _pixel_member_preds(model, x_row):
    """Per-member predictions (members, targets) of one feature row."""
    return np.stack(
        [np.asarray(t.predict(x_row), dtype=np.float64).reshape(-1) for t in model.estimators_]
    )


def test_sklearn_regressor_apply_uncertainty(tmp_path: Path):
    """out_uncert_img writes the spread of ensemble member predictions."""
    from rsgispy.ml.predict import UNCERT_METHODS, calc_ensemble_uncertainty

    uni_cfg, mv_cfg, train_df, _, _, msk_img = _build_reg_scene(tmp_path)
    feats = ["B1", "B2", "C1"]
    rf = RandomForestRegressor(n_estimators=15, random_state=1).fit(
        train_df[feats], train_df["canopy_cover"]
    )
    # Pixel (5, 5): B1 = 10 + 3 * 5, B2 = 20 + 4 * 5, C1 (20 m pixel row 2) = 5 + 2 * 2.
    x_px = np.array([[25.0, 40.0, 9.0]])
    members = _pixel_member_preds(rf, x_px)[:, 0]

    assert UNCERT_METHODS == ("std", "iqr", "cv")
    expected = {
        "std": np.std(members),
        "iqr": np.subtract(*np.percentile(members, [75, 25])),
        "cv": np.std(members) / abs(np.mean(members)),
    }
    for method, exp_val in expected.items():
        out_tif = tmp_path / f"pred_{method}.tif"
        unc_tif = tmp_path / f"uncert_{method}.tif"
        rsk.apply_sklearn_regressor(
            reg_cfg=uni_cfg,
            sk_regressor=rf,
            in_msk_img=msk_img,
            img_msk_val=1,
            output_img=out_tif,
            block_size=8,
            use_pbar=False,
            out_uncert_img=unc_tif,
            uncert_method=method,
        )
        with rasterio.open(unc_tif) as src:
            assert src.count == 1
            assert src.dtypes[0] == "float32"
            assert src.nodata == pytest.approx(-9999.0)
            assert src.descriptions == (f"canopy_cover_{method}",)
            arr = src.read(1)
        assert arr[0, 0] == pytest.approx(-9999.0)
        assert arr[2, 2] == pytest.approx(-9999.0)
        assert arr[5, 5] == pytest.approx(exp_val, rel=1e-5)
        valid = arr != -9999.0
        assert np.all(arr[valid] >= 0.0)
        with rasterio.open(out_tif) as src:
            pred = src.read(1)
        assert pred[5, 5] == pytest.approx(np.mean(members), rel=1e-5)

    # The public helper on a (members, samples, targets) array.
    mem = np.array([[[1.0]], [[3.0]]])
    np.testing.assert_allclose(calc_ensemble_uncertainty(mem, "std"), [[1.0]])
    np.testing.assert_allclose(calc_ensemble_uncertainty(mem, "cv"), [[0.5]])

    # Multi-target: native multi-output forest, MultiOutputRegressor and pipeline.
    mv_et = ExtraTreesRegressor(n_estimators=10, random_state=0).fit(
        train_df[feats], train_df[["canopy_cover", "canopy_height"]]
    )
    mo_rf = MultiOutputRegressor(RandomForestRegressor(n_estimators=8, random_state=0)).fit(
        train_df[feats], train_df[["canopy_cover", "canopy_height"]]
    )
    for name, model in (("et", mv_et), ("mo", mo_rf)):
        unc_tif = tmp_path / f"mv_uncert_{name}.tif"
        rsk.apply_sklearn_regressor(
            reg_cfg=mv_cfg, sk_regressor=model, in_msk_img=msk_img, img_msk_val=1,
            output_img=tmp_path / f"mv_pred_{name}.tif", use_pbar=False,
            out_uncert_img=unc_tif,
        )
        with rasterio.open(unc_tif) as src:
            assert src.count == 2
            assert src.descriptions == ("canopy_cover_std", "canopy_height_std")
            px = src.read()[:, 5, 5]
        if name == "et":
            exp = np.std(_pixel_member_preds(mv_et, x_px), axis=0)
        else:
            exp = [np.std(_pixel_member_preds(sub, x_px)) for sub in mo_rf.estimators_]
        np.testing.assert_allclose(px, exp, rtol=1e-5)

    pipe = Pipeline(
        [("scale", StandardScaler()), ("rf", RandomForestRegressor(n_estimators=8, random_state=0))]
    ).fit(train_df[feats], train_df["canopy_cover"])
    rsk.apply_sklearn_regressor(
        reg_cfg=uni_cfg, sk_regressor=pipe, in_msk_img=msk_img, img_msk_val=1,
        output_img=tmp_path / "pipe_pred.tif", use_pbar=False,
        out_uncert_img=tmp_path / "pipe_uncert.tif",
    )
    x_scaled = pipe[:-1].transform(pd.DataFrame(x_px, columns=feats))
    with rasterio.open(tmp_path / "pipe_uncert.tif") as src:
        assert src.read(1)[5, 5] == pytest.approx(
            np.std(_pixel_member_preds(pipe[-1], x_scaled)), rel=1e-5
        )

    # Models without independent members fail before any output is written.
    ridge = Ridge().fit(train_df[feats], train_df["canopy_cover"])
    gbr = GradientBoostingRegressor(n_estimators=5).fit(
        train_df[feats], train_df["canopy_cover"]
    )
    for model in (ridge, gbr):
        bad_out = tmp_path / f"bad_{type(model).__name__}.tif"
        with pytest.raises(rsgispy.RSGISPyException, match="ensemble"):
            rsk.apply_sklearn_regressor(
                reg_cfg=uni_cfg, sk_regressor=model, in_msk_img=msk_img,
                img_msk_val=1, output_img=bad_out, use_pbar=False,
                out_uncert_img=tmp_path / "bad_uncert.tif",
            )
        assert not bad_out.exists()
    with pytest.raises(rsgispy.RSGISPyException, match="uncert_method"):
        rsk.apply_sklearn_regressor(
            reg_cfg=uni_cfg, sk_regressor=rf, in_msk_img=msk_img, img_msk_val=1,
            output_img=tmp_path / "x.tif", use_pbar=False,
            out_uncert_img=tmp_path / "x_unc.tif", uncert_method="var",
        )


def test_regressor_param_search_with_spatial_and_group_splitters():
    """rsgispy.ml.cv splitters are passed directly with coords= and groups=."""
    from rsgispy.ml.cv import PolygonGroupKFold, SpatialBlockKFold

    rng = np.random.default_rng(1)
    rows = []
    for p_idx in range(12):
        target = 10.0 * p_idx
        for _ in range(6):
            rows.append(
                {
                    "cover": target,
                    "train_grp": f"poly{p_idx}",
                    "x": 1000.0 * (p_idx % 4) + rng.uniform(0, 30),
                    "y": 1000.0 * (p_idx // 4) + rng.uniform(0, 30),
                    "f1": target / 10.0 + rng.normal(0, 0.2),
                    "f2": rng.normal(0, 1),
                }
            )
    train_df = pd.DataFrame(rows)
    common = dict(
        reg_cfg=None,
        sk_regressor=RandomForestRegressor(n_estimators=10, random_state=0),
        train_data=train_df,
        target_cols="cover",
        param_space={"max_depth": [2, None]},
        feature_cols=["f1", "f2"],
        rnd_seed=1,
        return_results=True,
    )
    spatial_cv = SpatialBlockKFold(n_splits=3, block_size=1500.0, rnd_seed=3)
    _, res = rsk.perform_sklearn_regressor_param_search(
        cv=spatial_cv, coords=("x", "y"), **common
    )
    expected = list(
        spatial_cv.split(train_df, coords=train_df[["x", "y"]].to_numpy())
    )
    assert len(res["search_obj"].cv) == len(expected)
    for (tr_a, va_a), (tr_b, va_b) in zip(res["search_obj"].cv, expected):
        np.testing.assert_array_equal(tr_a, tr_b)
        np.testing.assert_array_equal(va_a, va_b)

    grp = train_df["train_grp"].to_numpy()
    for cv in (PolygonGroupKFold(n_splits=3), 3):
        _, res = rsk.perform_sklearn_regressor_param_search(
            cv=cv, groups="train_grp", **common
        )
        for tr_idx, va_idx in res["search_obj"].cv:
            assert set(grp[tr_idx]).isdisjoint(grp[va_idx])

    pytest.importorskip("optuna")
    best = rsk.perform_sklearn_regressor_param_search(
        reg_cfg=None,
        sk_regressor=RandomForestRegressor(n_estimators=10, random_state=0),
        train_data=train_df,
        target_cols="cover",
        param_space={"max_depth": [2, 4]},
        feature_cols=["f1", "f2"],
        search_method="optuna",
        n_trials=2,
        cv=spatial_cv,
        coords=train_df[["x", "y"]].to_numpy(),
        use_pbar=False,
    )
    assert isinstance(best, RandomForestRegressor)
    with pytest.raises(rsgispy.RSGISPyException, match="coords"):
        rsk.perform_sklearn_regressor_param_search(cv=spatial_cv, **common)
