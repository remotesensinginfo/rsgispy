"""
Tests for the in-house MaxEnt implementation
(:class:`rsgispy.modelling.species_distribution.MaxentClassifier`,
:class:`~rsgispy.modelling.species_distribution.MaxentFeatureTransformer`,
:func:`~rsgispy.modelling.species_distribution.tune_maxent_beta`).

The tests are self-contained: synthetic data only, and no MaxEnt package
(``elapid``, ``maxnet``, ``glmnet``) or output saved from one.
"""

import pickle

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio

import rsgispy
import rsgispy.modelling.species_distribution as sdm

pytest.importorskip("sklearn")
from sklearn.base import clone  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from sdm_scene import make_sdm_scene  # noqa: E402


# --------------------------------------------------------------------------
# Feature transformer
# --------------------------------------------------------------------------


def test_feature_values_hand_computed():
    x_df = pd.DataFrame({"t": [0.0, 5.0, 10.0]})
    ftr = sdm.MaxentFeatureTransformer(
        feature_types="lqht", n_hinge_knots=3, n_threshold_knots=2
    ).fit(x_df)
    names = list(ftr.get_feature_names_out())
    assert names == [
        "t", "t^2", "hinge_fwd(t,0)", "hinge_fwd(t,5)", "hinge_rev(t,5)",
        "hinge_rev(t,10)", "thr(t,3.33333)", "thr(t,6.66667)",
    ]
    feats = ftr.transform(x_df)
    expected = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
            [0.5, 0.25, 0.5, 0.0, 0.0, 0.5, 1.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
        ]
    )
    np.testing.assert_allclose(feats, expected)
    assert list(ftr.feature_classes_) == ["l", "q", "h", "h", "h", "h", "t", "t"]


def test_product_categorical_and_clamp():
    x_df = pd.DataFrame({"a": [0.0, 2.0, 4.0], "b": [10.0, 20.0, 30.0], "lc": [1, 2, 1]})
    ftr = sdm.MaxentFeatureTransformer(feature_types="lp", categorical_vars=["lc"]).fit(x_df)
    assert list(ftr.get_feature_names_out()) == ["a", "b", "a*b", "cat(lc=1)", "cat(lc=2)"]
    new = pd.DataFrame({"a": [8.0, 2.0], "b": [20.0, 0.0], "lc": [3, 2]})
    clamped = ftr.transform(new)
    # a=8 clamped to 4 (scaled 1); b=0 clamped to 10 (scaled 0); lc=3 unseen.
    np.testing.assert_allclose(
        clamped, [[1.0, 0.5, 0.5, 0.0, 0.0], [0.5, 0.0, 0.0, 0.0, 1.0]]
    )
    raw = sdm.MaxentFeatureTransformer(
        feature_types="l", clamp=False, categorical_vars=["lc"]
    ).fit(x_df).transform(new)
    # Training range is used for scaling, not the new data's range.
    np.testing.assert_allclose(raw[:, :2], [[2.0, 0.5], [0.5, -0.5]])


def test_auto_feature_types():
    assert [sdm.maxent_default_feature_types(n) for n in (5, 9, 10, 14, 15, 79, 80)] == [
        "l", "l", "lq", "lq", "lqh", "lqh", "lqph",
    ]
    x_df = pd.DataFrame({"a": np.arange(30.0), "b": np.arange(30.0) ** 0.5})
    y_arr = np.r_[np.ones(12), np.zeros(18)]
    ftr = sdm.MaxentFeatureTransformer().fit(x_df, y_arr)
    assert ftr.feature_types_ == "lq"
    with pytest.raises(rsgispy.RSGISPyException, match="auto"):
        sdm.MaxentFeatureTransformer().fit(x_df)
    with pytest.raises(rsgispy.RSGISPyException):
        sdm.MaxentFeatureTransformer(feature_types="lx").fit(x_df)


# --------------------------------------------------------------------------
# Synthetic point process
# --------------------------------------------------------------------------


def _ipp_data(n_bg=6000, n_pres=200, seed=0, a=1.5, c=0.15):
    """Presences from an inhomogeneous Poisson process with intensity
    exp(a x1 - c x1^2) over a uniform background; x2 is irrelevant."""
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0, 10, n_bg)
    x2 = rng.uniform(0, 10, n_bg)
    lam = np.exp(a * x1 - c * x1**2)
    pres = rng.choice(n_bg, n_pres, p=lam / lam.sum())
    x_df = pd.DataFrame({"x1": np.r_[x1[pres], x1], "x2": np.r_[x2[pres], x2]})
    y_arr = np.r_[np.ones(n_pres, dtype=int), np.zeros(n_bg, dtype=int)]
    return x_df, y_arr


def test_recovers_known_response():
    x_df, y_arr = _ipp_data()
    mdl = sdm.MaxentClassifier(feature_types="lq").fit(x_df, y_arr)
    grid = pd.DataFrame({"x1": np.linspace(0, 10, 201), "x2": 5.0})
    peak = grid["x1"].iloc[int(np.argmax(mdl.predict_raw(grid)))]
    # The true peak is at x1 = 5. Over data seeds 0-7 the fitted peak (on a
    # 0.05 grid) was within 0.15 of it (0.05 for seed 0), so abs=0.25 allows
    # sampling noise but fails for a peak shifted by more than 5 grid steps.
    assert peak == pytest.approx(1.5 / (2 * 0.15), abs=0.25)
    coefs = dict(zip(mdl.selected_features_["feature"], mdl.selected_features_["coef"]))
    # x2 is irrelevant: its coefficients were at most 9.5 % of the x1 ones
    # over seeds 0-7 (2.6 % for seed 0), so 15 % leaves a margin.
    assert abs(coefs.get("x2", 0.0)) < 0.15 * abs(coefs["x1"])
    assert abs(coefs.get("x2^2", 0.0)) < 0.15 * abs(coefs["x1^2"])
    assert roc_auc_score(y_arr, mdl.predict_proba(x_df)[:, 1]) > 0.7


def test_output_properties():
    x_df, y_arr = _ipp_data(n_bg=2000, n_pres=80, seed=1)
    mdl = sdm.MaxentClassifier().fit(x_df, y_arr)
    assert mdl.transformer_.feature_types_ == "lqph"
    raw_bg = mdl.predict_raw(x_df[y_arr == 0])
    assert raw_bg.sum() == pytest.approx(1.0)
    proba = mdl.predict_proba(x_df)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    assert proba.min() >= 0.0 and proba.max() <= 1.0
    link = mdl.decision_function(x_df)
    order = np.argsort(link)
    assert np.all(np.diff(proba[order, 1]) >= -1e-12)
    np.testing.assert_allclose(
        proba[:, 1], 1.0 - np.exp(-np.exp(link + mdl.entropy_)), rtol=1e-10
    )
    logi = clone(mdl).set_params(output="logistic").fit(x_df, y_arr)
    logi_link = logi.decision_function(x_df) + logi.entropy_
    np.testing.assert_allclose(
        logi.predict_proba(x_df)[:, 1], 1.0 / (1.0 + np.exp(-logi_link)), rtol=1e-10
    )
    assert set(np.unique(mdl.predict(x_df))) <= {0, 1}
    assert list(mdl.classes_) == [0, 1]


def test_weighted_lasso_matches_direct_minimisation():
    scipy_opt = pytest.importorskip("scipy.optimize")
    x_df, y_arr = _ipp_data(n_bg=400, n_pres=40, seed=2)
    mdl = sdm.MaxentClassifier(feature_types="lq", bg_weight=20.0).fit(x_df, y_arr)
    feats = mdl.transformer_.transform(x_df)
    pen = mdl.penalties_
    wgt = np.where(y_arr == 1, 1.0, 20.0)
    n_pres = int(y_arr.sum())
    n_feat = feats.shape[1]

    def objective(theta):
        b0, pos, neg = theta[0], theta[1 : n_feat + 1], theta[n_feat + 1 :]
        eta = b0 + feats @ (pos - neg)
        loss = np.sum(wgt * (np.logaddexp(0.0, eta) - y_arr * eta))
        resid = wgt * (1.0 / (1.0 + np.exp(-eta)) - y_arr)
        g_beta = feats.T @ resid
        grad = np.r_[resid.sum(), g_beta + n_pres * pen, -g_beta + n_pres * pen]
        return loss + n_pres * np.sum(pen * (pos + neg)), grad

    bounds = [(None, None)] + [(0.0, None)] * (2 * n_feat)
    res = scipy_opt.minimize(
        objective, np.zeros(2 * n_feat + 1), jac=True, method="L-BFGS-B",
        bounds=bounds, options={"maxiter": 20000, "ftol": 1e-14, "gtol": 1e-10},
    )
    direct = res.x[1 : n_feat + 1] - res.x[n_feat + 1 :]
    np.testing.assert_allclose(mdl.coef_, direct, atol=0.05 * np.max(np.abs(direct)))


def test_regularisation_strength():
    x_df, y_arr = _ipp_data(n_bg=2000, n_pres=100, seed=3)
    n_coefs = [
        int(np.count_nonzero(sdm.MaxentClassifier(feature_types="lqh", beta_multiplier=b).fit(x_df, y_arr).coef_))
        for b in (0.5, 2.0, 8.0)
    ]
    assert n_coefs[0] >= n_coefs[1] >= n_coefs[2]
    flat = sdm.MaxentClassifier(feature_types="lqh", beta_multiplier=1e4).fit(x_df, y_arr)
    assert np.count_nonzero(flat.coef_) == 0
    proba = flat.predict_proba(x_df)[:, 1]
    assert np.allclose(proba, proba[0])


def test_presence_weights_remove_sampling_bias():
    # Habitat is uniform in x1, but surveys favoured high x1.
    rng = np.random.default_rng(4)
    n_bg = 4000
    x1 = rng.uniform(0, 10, n_bg)
    bias = np.exp(x1 / 3.0)
    pres = rng.choice(n_bg, 300, p=bias / bias.sum())
    x_df = pd.DataFrame({"x1": np.r_[x1[pres], x1]})
    y_arr = np.r_[np.ones(300, dtype=int), np.zeros(n_bg, dtype=int)]
    plain = sdm.MaxentClassifier(feature_types="l").fit(x_df, y_arr)
    inv = 1.0 / bias[pres]
    wgts = np.r_[inv / inv.mean(), np.ones(n_bg)]
    weighted = sdm.MaxentClassifier(feature_types="l").fit(x_df, y_arr, sample_weight=wgts)
    assert plain.coef_[0] > 2.0
    assert abs(weighted.coef_[0]) < 0.5 * plain.coef_[0]


def test_validation_and_sklearn_compat(tmp_path):
    x_df, y_arr = _ipp_data(n_bg=500, n_pres=30, seed=5)
    mdl = sdm.MaxentClassifier(beta_multiplier=2.0, random_state=1)
    cloned = clone(mdl)
    assert cloned.get_params() == mdl.get_params()
    mdl.fit(x_df, y_arr)
    assert list(mdl.feature_names_in_) == ["x1", "x2"] and mdl.n_features_in_ == 2
    restored = pickle.loads(pickle.dumps(mdl))
    np.testing.assert_allclose(restored.predict_proba(x_df), mdl.predict_proba(x_df))
    arr_mdl = clone(mdl).fit(x_df.to_numpy(), y_arr)
    np.testing.assert_allclose(arr_mdl.coef_, mdl.coef_)
    assert not hasattr(arr_mdl, "feature_names_in_")
    with pytest.raises(rsgispy.RSGISPyException, match="presences"):
        clone(mdl).fit(x_df.iloc[:503], y_arr[:503] * 0 + np.r_[np.ones(3, int), np.zeros(500, int)])
    with pytest.raises(rsgispy.RSGISPyException, match="binary"):
        clone(mdl).fit(x_df, y_arr * 2)
    with pytest.raises(rsgispy.RSGISPyException, match="background"):
        clone(mdl).fit(x_df.iloc[:30], y_arr[:30])
    with pytest.raises(rsgispy.RSGISPyException, match="not fitted"):
        sdm.MaxentClassifier().predict_proba(x_df)


# --------------------------------------------------------------------------
# Tuning and integration with the SDM workflow
# --------------------------------------------------------------------------


def _samples_gdf(x_df, y_arr, seed=0):
    rng = np.random.default_rng(seed)
    n_rows = len(x_df)
    gdf = gpd.GeoDataFrame(
        x_df.assign(clsid=y_arr),
        geometry=gpd.points_from_xy(rng.uniform(0, 10000, n_rows), rng.uniform(0, 10000, n_rows)),
        crs="EPSG:32630",
    )
    return gdf


def test_tune_maxent_beta():
    x_df, y_arr = _ipp_data(n_bg=1500, n_pres=100, seed=6)
    gdf = _samples_gdf(x_df, y_arr)
    res_df, best = sdm.tune_maxent_beta(
        gdf, None, ["x1", "x2"], beta_multipliers=(0.5, 2.0), feature_types=("l", "lq"),
        cv=3, use_pbar=False,
    )
    assert len(res_df) == 4 and {"auc_mean", "auc_sd", "n_coefs"} <= set(res_df.columns)
    assert best == {
        "beta_multiplier": float(res_df.loc[res_df["auc_mean"].idxmax(), "beta_multiplier"]),
        "feature_types": res_df.loc[res_df["auc_mean"].idxmax(), "feature_types"],
    }
    aicc_df, aicc_best = sdm.tune_maxent_beta(
        gdf, None, ["x1", "x2"], beta_multipliers=(1.0, 4.0), feature_types=("lq",),
        metric="aicc", use_pbar=False,
    )
    assert "aicc" in aicc_df.columns and aicc_best["feature_types"] == "lq"
    with pytest.raises(rsgispy.RSGISPyException):
        sdm.tune_maxent_beta(gdf, None, ["x1"], metric="bic", use_pbar=False)


def test_end_to_end_with_sdm_functions(tmp_path):
    scene = make_sdm_scene(str(tmp_path / "scene"))
    env_vars = {
        "temp": sdm.EnvVarInfo(name="temp", file=scene["climate"], band=1),
        "precip": sdm.EnvVarInfo(name="precip", file=scene["climate"], band=2),
        "ndvi": sdm.EnvVarInfo(name="ndvi", file=scene["ndvi"], band=1),
    }
    bg_gdf = sdm.gen_pseudo_absences_smpls(
        scene["roi_mask"], 1, n_smpls=1500, xtra_n_smpls=500, rnd_seed=1, use_pbar=False
    )
    pres_gdf = sdm.extract_env_var_data(env_vars, scene["presences"], "presences", use_pbar=False)
    bg_gdf = sdm.extract_env_var_data(env_vars, bg_gdf, None, use_pbar=False)
    comb_gdf, analysis_vars = sdm.combine_presence_absence_data(
        pres_gdf, None, bg_gdf, None, env_vars
    )
    train_gdf, test_gdf = sdm.create_train_test_sets(comb_gdf, None, rnd_seed=2)
    mdl, _, _, auc = sdm.fit_sklearn_mdl(
        sdm.MaxentClassifier(), train_gdf, None, test_gdf, None, analysis_vars,
        return_model=True,
    )
    assert auc > 0.8 and len(mdl.selected_features_) > 0
    out_img = str(tmp_path / "maxent_prob.tif")
    sdm.pred_sklearn_mdl_prob(
        mdl, scene["roi_mask"], 1, env_vars, analysis_vars, out_img, use_pbar=False
    )
    with rasterio.open(out_img) as src:
        vals = src.read(1)
    valid = vals != -1.0
    assert valid.any() and vals[valid].min() >= 0.0 and vals[valid].max() <= 1.0
    mdls = sdm.fit_kfold_sklearn_mdls(
        sdm.MaxentClassifier(feature_types="lq"), train_gdf, None, test_gdf, None,
        analysis_vars, n_kfolds=2, rnd_seed=1, use_pbar=False,
    )
    assert len(mdls) == 2
    resp = sdm.sklearn_mdl_variable_response_curves(
        mdl, train_gdf, None, analysis_vars, None, None, n_samples=10
    )
    assert "temp_resp" in resp.columns
