"""
Tests for spatial autocorrelation (``rsgispy.vector.stats.calc_morans_i`` /
``calc_correlogram``), SDM residual autocorrelation, null-model AUC
significance and the extra pseudo-absence strategies.
"""

import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point

import rsgispy
import rsgispy.modelling.species_distribution as sdm
import rsgispy.vector.stats as vstats

pytest.importorskip("sklearn")
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

CRS = "EPSG:32630"


def _pts(vals, xy, crs=CRS, col="val"):
    return gpd.GeoDataFrame(
        {col: vals}, geometry=gpd.points_from_xy(xy[:, 0], xy[:, 1]), crs=crs
    )


# --------------------------------------------------------------------------
# Moran's I and correlograms
# --------------------------------------------------------------------------


def test_morans_i_random_near_expectation():
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 1000, (300, 2))
    i_vals = [
        vstats.calc_morans_i(_pts(rng.normal(size=300), xy), None, "val", n_perms=0)["I"]
        for _ in range(20)
    ]
    assert np.mean(i_vals) == pytest.approx(-1.0 / 299, abs=0.02)


def test_morans_i_trend_and_reproducible():
    rng = np.random.default_rng(1)
    xy = rng.uniform(0, 1000, (300, 2))
    gdf = _pts(xy[:, 0] / 100.0 + rng.normal(0, 0.5, 300), xy)
    res = vstats.calc_morans_i(gdf, None, "val", n_perms=199, rnd_seed=3)
    assert res["I"] > 0.7 and res["p_value"] == pytest.approx(1.0 / 200)
    assert res["z_score"] > 5 and res["n"] == 300 and res["n_islands"] == 0
    again = vstats.calc_morans_i(gdf, None, "val", n_perms=199, rnd_seed=3)
    assert again == res
    knn = vstats.calc_morans_i(gdf, None, "val", weights="knn", k=6, n_perms=0)
    assert knn["I"] > 0.7 and knn["n_pairs"] == 300 * 6
    neg = _pts(np.tile([1.0, -1.0], 50), np.column_stack([np.arange(100) * 10.0, np.zeros(100)]))
    res_neg = vstats.calc_morans_i(neg, None, "val", dist=10.0, n_perms=99, rnd_seed=1)
    assert res_neg["I"] < -0.9 and res_neg["p_value"] == pytest.approx(0.01)


def test_morans_i_hand_worked():
    # Four points on a line 1 m apart, neighbours within 1 m, values 1, 2, 3, 4.
    xy = np.column_stack([np.arange(4.0), np.zeros(4)])
    res = vstats.calc_morans_i(_pts([1.0, 2.0, 3.0, 4.0], xy), None, "val", dist=1.0, n_perms=0)
    z_vals = np.array([-1.5, -0.5, 0.5, 1.5])
    w_mat = np.array(
        [[0, 1, 0, 0], [0.5, 0, 0.5, 0], [0, 0.5, 0, 0.5], [0, 0, 1, 0]]
    )
    expected = 4 / w_mat.sum() * z_vals @ w_mat @ z_vals / (z_vals @ z_vals)
    assert res["I"] == pytest.approx(expected)


def test_morans_i_warnings_and_errors():
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [100.0, 0.0]])
    with pytest.warns(UserWarning, match="islands"):
        vstats.calc_morans_i(_pts([1.0, 2.0, 3.0, 4.0], xy), None, "val", dist=1.5, n_perms=0)
    gdf = _pts([1.0, np.nan, 3.0, 4.0, 5.0], np.column_stack([np.arange(5.0), np.zeros(5)]))
    with pytest.warns(UserWarning, match="missing"):
        res = vstats.calc_morans_i(gdf, None, "val", n_perms=0)
    assert res["n"] == 4
    geo = _pts([1.0, 2.0, 3.0], np.array([[0.0, 50.0], [0.1, 50.0], [0.2, 50.0]]), crs="EPSG:4326")
    with pytest.raises(rsgispy.RSGISPyException, match="geographic"):
        vstats.calc_morans_i(geo, None, "val")
    const = _pts([1.0, 1.0, 1.0], np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]))
    with pytest.raises(rsgispy.RSGISPyException, match="equal"):
        vstats.calc_morans_i(const, None, "val")
    with pytest.raises(rsgispy.RSGISPyException):
        vstats.calc_morans_i(gdf, None, "val", weights="queen")


def test_correlogram(tmp_path):
    rng = np.random.default_rng(2)
    xy = rng.uniform(0, 1000, (300, 2))
    gdf = _pts(np.sin(xy[:, 0] / 150.0) + rng.normal(0, 0.2, 300), xy)
    plot_file = str(tmp_path / "corr.png")
    corr = vstats.calc_correlogram(
        gdf, None, "val", n_classes=6, max_dist=600.0, n_perms=49, rnd_seed=1,
        plot_file=plot_file,
    )
    assert list(corr.columns) == [
        "dist_lower", "dist_upper", "dist_mid", "n_pairs", "morans_i",
        "expected_i", "p_value",
    ]
    assert len(corr) == 6 and (tmp_path / "corr.png").exists()
    assert corr["morans_i"].iloc[0] > 0.5
    assert corr["morans_i"].iloc[0] > corr["morans_i"].iloc[2]
    custom = vstats.calc_correlogram(
        gdf, None, "val", dist_classes=[0, 100, 300], n_perms=0
    )
    assert list(custom["dist_upper"]) == [100, 300]
    with pytest.raises(rsgispy.RSGISPyException):
        vstats.calc_correlogram(gdf, None, "val", dist_classes=[100, 50])


# --------------------------------------------------------------------------
# SDM residual autocorrelation and null-model AUC
# --------------------------------------------------------------------------


def _sdm_samples(n=400, seed=0):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 10000, (n, 2))
    v1 = rng.normal(0, 1, n)
    hidden = np.sin(xy[:, 0] / 1500.0) * 2.0  # spatial driver not in the model
    clsid = (v1 + hidden + rng.normal(0, 0.5, n) > 0.5).astype(int)
    gdf = gpd.GeoDataFrame(
        {"v1": v1, "v2": rng.normal(0, 1, n), "clsid": clsid},
        geometry=gpd.points_from_xy(xy[:, 0], xy[:, 1]),
        crs=CRS,
    )
    return gdf


def test_residual_autocorrelation():
    gdf = _sdm_samples()
    mdl = LogisticRegression().fit(gdf[["v1"]], gdf["clsid"])
    corr = sdm.calc_residual_autocorrelation(
        mdl, gdf, None, ["v1"], n_classes=5, max_dist=2500.0, n_perms=49, rnd_seed=1
    )
    assert corr["morans_i"].iloc[0] > 0.2
    assert corr["p_value"].iloc[0] <= 0.05


def test_null_model_auc_modes(tmp_path):
    rng = np.random.default_rng(4)
    bg = _sdm_samples(300, seed=4).drop(columns=["clsid"])
    test = _sdm_samples(200, seed=5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sdm.calc_null_model_auc(
            LogisticRegression(), bg, None, test, None, ["v1", "v2"], n_pres=20,
            n_reps=30, observed_auc=0.9, rnd_seed=1, use_pbar=False,
            plot_file=str(tmp_path / "null.png"),
        )
    assert res["mode"] == "test" and res["null_aucs"].shape == (30,)
    assert abs(res["null_mean"] - 0.5) < 0.1
    assert res["p_value"] == pytest.approx((1 + np.sum(res["null_aucs"] >= 0.9)) / 31)
    assert res["p_value"] < 0.1 and (tmp_path / "null.png").exists()

    again = sdm.calc_null_model_auc(
        LogisticRegression(), bg, None, test, None, ["v1", "v2"], n_pres=20,
        n_reps=30, rnd_seed=1, use_pbar=False,
    )
    np.testing.assert_array_equal(again["null_aucs"], res["null_aucs"])

    # Training AUC of a flexible model is far above 0.5 with no signal at all.
    train_res = sdm.calc_null_model_auc(
        RandomForestClassifier(n_estimators=20, random_state=0), bg, None, None, None,
        ["v1", "v2"], n_pres=10, n_reps=10, rnd_seed=2, use_pbar=False,
    )
    assert train_res["mode"] == "train" and train_res["null_mean"] > 0.9

    cv_res = sdm.calc_null_model_auc(
        LogisticRegression(), bg, None, None, None, ["v1", "v2"], n_pres=40,
        n_reps=5, cv=3, rnd_seed=3, use_pbar=False,
    )
    assert cv_res["mode"] == "cv" and np.isfinite(cv_res["null_mean"])


def test_null_model_auc_validation():
    bg = _sdm_samples(50).drop(columns=["clsid"])
    test = _sdm_samples(40, seed=1)
    with pytest.raises(rsgispy.RSGISPyException):
        sdm.calc_null_model_auc(
            LogisticRegression(), bg, None, test, None, ["v1"], n_pres=60, use_pbar=False
        )
    with pytest.raises(rsgispy.RSGISPyException, match="not both"):
        sdm.calc_null_model_auc(
            LogisticRegression(), bg, None, test, None, ["v1"], n_pres=10, cv=3,
            use_pbar=False,
        )


# --------------------------------------------------------------------------
# Pseudo-absence strategies
# --------------------------------------------------------------------------


SIZE, RES, X0, Y0 = 50, 100.0, 400000.0, 5800000.0


def _raster(path, arr, nodata=None):
    with rasterio.open(
        path, "w", driver="GTiff", height=SIZE, width=SIZE, count=1, dtype=arr.dtype,
        crs=CRS, transform=from_origin(X0, Y0, RES, RES), nodata=nodata,
    ) as dst:
        dst.write(arr[np.newaxis])
    return str(path)


def _pa_scene(tmp_path):
    msk = np.ones((SIZE, SIZE), np.uint8)
    rows, cols = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    env_vars = {
        "v1": sdm.EnvVarInfo(name="v1", file=_raster(tmp_path / "v1.tif", rows, -9999.0)),
        "v2": sdm.EnvVarInfo(name="v2", file=_raster(tmp_path / "v2.tif", cols, -9999.0)),
    }
    pres = gpd.GeoDataFrame(
        geometry=[
            Point(X0 + (c + 0.5) * RES, Y0 - (r + 0.5) * RES)
            for r, c in [(20, 20), (22, 24), (25, 22), (24, 26), (21, 23)]
        ],
        crs=CRS,
    )
    return _raster(tmp_path / "msk.tif", msk), env_vars, pres


def test_pseudo_absence_distance_band(tmp_path):
    msk_img, _, pres = _pa_scene(tmp_path)
    pts = sdm.gen_pseudo_absences_smpls(
        msk_img, 1, n_smpls=200, xtra_n_smpls=3000, presence_smpls_vec_file=pres,
        presence_smpls_dist_thres=300.0, presence_smpls_max_dist=1000.0,
        rnd_seed=1, use_pbar=False,
    )
    dists = np.array([pres.distance(g).min() for g in pts.geometry])
    assert len(pts) == 200
    assert np.all(dists > 300.0) and np.all(dists <= 1000.0)


def test_pseudo_absence_sre(tmp_path):
    msk_img, env_vars, pres = _pa_scene(tmp_path)
    pts = sdm.gen_pseudo_absences_smpls(
        msk_img, 1, n_smpls=300, xtra_n_smpls=1000, presence_smpls_vec_file=pres,
        presence_smpls_dist_thres=0.0, env_profile="sre",
        env_profile_quantiles=(0.0, 1.0), env_vars=env_vars,
        analysis_vars=["v1", "v2"], rnd_seed=2, use_pbar=False,
    )
    rows = np.floor((Y0 - pts.geometry.y) / RES)
    cols = np.floor((pts.geometry.x - X0) / RES)
    inside = (rows >= 20) & (rows <= 25) & (cols >= 20) & (cols <= 26)
    assert len(pts) == 300 and not inside.any()
    assert list(pts.columns) == ["geometry"]


def test_pseudo_absence_strategy_validation(tmp_path):
    msk_img, env_vars, pres = _pa_scene(tmp_path)
    bad_args = [
        {"presence_smpls_max_dist": 500.0},
        {"presence_smpls_vec_file": pres, "presence_smpls_max_dist": 500.0,
         "presence_smpls_dist_thres": 800.0},
        {"presence_smpls_vec_file": pres, "env_profile": "sre"},
        {"presence_smpls_vec_file": pres, "env_profile": "bioclim",
         "env_vars": env_vars, "analysis_vars": ["v1"]},
        {"presence_smpls_vec_file": pres, "env_profile": "sre", "env_vars": env_vars,
         "analysis_vars": ["v1"], "env_profile_quantiles": (0.9, 0.1)},
    ]
    for kwargs in bad_args:
        with pytest.raises(rsgispy.RSGISPyException):
            sdm.gen_pseudo_absences_smpls(msk_img, 1, n_smpls=10, use_pbar=False, **kwargs)
