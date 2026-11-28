"""
Round-trip tests for the GBDT regressor save / load functions
(``save_{catboost,xgboost,lightgbm}_regressor`` and the matching ``load_*``):
fit a tiny model, save it in each supported format, load it back and check it
gives the same predictions. Each library is optional, so its tests are
skipped when it is not installed.
"""

import os

import numpy as np
import pandas as pd
import pytest

import rsgispy


def _data(n_targets=1):
    rng = np.random.default_rng(0)
    x_df = pd.DataFrame(rng.uniform(0, 1, (60, 3)), columns=["b1", "b2", "b3"])
    y_arr = 3.0 * x_df["b1"].to_numpy() - x_df["b2"].to_numpy()
    if n_targets > 1:
        y_arr = np.column_stack([y_arr, 2.0 * y_arr + 1.0])
    return x_df, y_arr


def _check_round_trip(model, loaded, x_df):
    assert type(loaded) is type(model)
    pred, pred_back = model.predict(x_df), loaded.predict(x_df)
    assert np.all(np.isfinite(pred))
    np.testing.assert_allclose(pred_back, pred, rtol=1e-6, atol=1e-6)
    assert np.std(pred) > 0  # a fitted (not constant) model was saved


@pytest.mark.parametrize("ext", [".json", ".ubj", ".joblib", ".pkl"])
def test_xgboost_regressor_round_trip(tmp_path, ext):
    xgb = pytest.importorskip("xgboost")
    import rsgispy.regression.regxgboost as rxgb  # noqa: E402

    x_df, y_arr = _data()
    model = xgb.XGBRegressor(n_estimators=10, max_depth=2).fit(x_df, y_arr)
    out_file = tmp_path / "sub" / f"xgb_reg{ext}"  # parent created on save
    rxgb.save_xgboost_regressor(model, out_file)
    assert out_file.is_file() and out_file.stat().st_size > 0
    loaded = rxgb.load_xgboost_regressor(str(out_file))
    _check_round_trip(model, loaded, x_df)
    with pytest.raises(rsgispy.RSGISPyException):
        rxgb.load_xgboost_regressor(tmp_path / "missing.json")


@pytest.mark.parametrize("ext", [".cbm", ".json", ".joblib"])
def test_catboost_regressor_round_trip(tmp_path, ext):
    cb = pytest.importorskip("catboost")
    import rsgispy.regression.regcatboost as rcb  # noqa: E402

    x_df, y_arr = _data()
    model = cb.CatBoostRegressor(
        iterations=10, depth=2, verbose=False, allow_writing_files=False
    ).fit(x_df, y_arr)
    out_file = tmp_path / "sub" / f"cb_reg{ext}"
    rcb.save_catboost_regressor(model, out_file)
    assert out_file.is_file() and out_file.stat().st_size > 0
    loaded = rcb.load_catboost_regressor(out_file)
    _check_round_trip(model, loaded, x_df)
    with pytest.raises(rsgispy.RSGISPyException):
        rcb.load_catboost_regressor(tmp_path / "missing.cbm")


def test_catboost_multi_target_round_trip(tmp_path):
    cb = pytest.importorskip("catboost")
    import rsgispy.regression.regcatboost as rcb  # noqa: E402

    x_df, y_arr = _data(n_targets=2)
    model = cb.CatBoostRegressor(
        iterations=10, depth=2, loss_function="MultiRMSE", verbose=False,
        allow_writing_files=False,
    ).fit(x_df, y_arr)
    out_file = tmp_path / "cb_multi.cbm"
    rcb.save_catboost_regressor(model, out_file)
    loaded = rcb.load_catboost_regressor(out_file)
    assert loaded.predict(x_df).shape == (len(x_df), 2)
    _check_round_trip(model, loaded, x_df)


@pytest.mark.parametrize("multi", [False, True])
def test_lightgbm_regressor_round_trip(tmp_path, multi):
    lgb = pytest.importorskip("lightgbm")
    from sklearn.multioutput import MultiOutputRegressor  # noqa: E402

    import rsgispy.regression.reglightgbm as rlgb  # noqa: E402

    x_df, y_arr = _data(n_targets=2 if multi else 1)
    base = lgb.LGBMRegressor(n_estimators=10, min_child_samples=5, verbose=-1)
    model = (MultiOutputRegressor(base) if multi else base).fit(x_df, y_arr)
    out_file = tmp_path / "sub" / "lgb_reg.joblib"
    rlgb.save_lightgbm_regressor(model, out_file)
    assert out_file.is_file() and out_file.stat().st_size > 0
    loaded = rlgb.load_lightgbm_regressor(str(out_file))
    _check_round_trip(model, loaded, x_df)
    if multi:
        assert loaded.predict(x_df).shape == (len(x_df), 2)
    with pytest.raises(rsgispy.RSGISPyException):
        rlgb.load_lightgbm_regressor(os.path.join(tmp_path, "missing.joblib"))


@pytest.mark.parametrize(
    "mod_name, flag, save_fn, load_fn",
    [
        ("regxgboost", "HAVE_XGBOOST", "save_xgboost_regressor", "load_xgboost_regressor"),
        ("regcatboost", "HAVE_CATBOOST", "save_catboost_regressor", "load_catboost_regressor"),
        ("reglightgbm", "HAVE_LIGHTGBM", "save_lightgbm_regressor", "load_lightgbm_regressor"),
    ],
)
def test_save_load_need_the_library(tmp_path, monkeypatch, mod_name, flag, save_fn, load_fn):
    import importlib  # noqa: E402

    mod = importlib.import_module(f"rsgispy.regression.{mod_name}")
    monkeypatch.setattr(mod, flag, False)
    out_file = tmp_path / "model.joblib"
    with pytest.raises(rsgispy.RSGISPyException):
        getattr(mod, save_fn)(object(), out_file)
    assert not out_file.exists()
    out_file.write_bytes(b"x")
    with pytest.raises(rsgispy.RSGISPyException):
        getattr(mod, load_fn)(out_file)
