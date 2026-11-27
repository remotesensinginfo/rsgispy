"""
Tests for SDM evaluation metrics (``calc_boyce_index``,
``calc_boyce_index_from_raster``, ``calc_sdm_eval_metrics``) and spatial
cross-validation in ``fit_kfold_sklearn_mdls`` /
``fit_kfold_sklearn_sgl_cls_mdls``.
"""

import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

import rsgispy
import rsgispy.modelling.species_distribution as sdm
from rsgispy.ml.cv import SpatialBlockKFold
from rsgispy.tools.accuracy import accuracy_scores_binary

pytest.importorskip("sklearn")
from sklearn.ensemble import IsolationForest  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupKFold, KFold  # noqa: E402

CRS = "EPSG:32630"


# --------------------------------------------------------------------------
# Boyce index
# --------------------------------------------------------------------------


def _pres_bg(seed=0, power=3.0, n_bg=5000, n_pres=500):
    rng = np.random.default_rng(seed)
    bkg = rng.uniform(0.0, 1.0, n_bg)
    prob = bkg**power / np.sum(bkg**power)
    pres = rng.choice(bkg, n_pres, p=prob)
    return pres, bkg


def test_boyce_good_model_near_one():
    pres, bkg = _pres_bg()
    res = sdm.calc_boyce_index(pres, bkg)
    assert res["boyce"] > 0.95
    assert res["bin_mids"].size == 101 and res["pe_ratio"].size == 101
    # Inverted scores give a strongly negative index.
    assert sdm.calc_boyce_index(1.0 - pres, 1.0 - bkg)["boyce"] < -0.9


def test_boyce_random_near_zero():
    rng = np.random.default_rng(1)
    bkg = rng.uniform(0.0, 1.0, 20000)
    vals = [
        sdm.calc_boyce_index(rng.uniform(0.0, 1.0, 5000), bkg, window_width=0.2)["boyce"]
        for _ in range(10)
    ]
    assert abs(float(np.mean(vals))) < 0.3


def test_boyce_hand_worked_example():
    # Range 0..1; 3 windows of width 0.5: [0, 0.5], [0.25, 0.75], [0.5, 1].
    bkg = np.array([0.0, 0.1, 0.3, 0.6, 0.8, 1.0])
    pres = np.array([0.6, 0.8, 1.0, 0.3])
    res = sdm.calc_boyce_index(
        pres, bkg, n_bins=3, window_width=0.5, method="pearson", rm_duplicates=False
    )
    np.testing.assert_allclose(res["bin_mids"], [0.25, 0.5, 0.75])
    # P: [1/4, 2/4, 3/4]; E: [3/6, 2/6, 3/6] -> P/E = [0.5, 1.5, 1.5]
    np.testing.assert_allclose(res["pe_ratio"], [0.5, 1.5, 1.5])
    assert res["boyce"] == pytest.approx(np.corrcoef([0.25, 0.5, 0.75], [0.5, 1.5, 1.5])[0, 1])
    # With duplicates removed the third window is dropped: two points -> r = 1.
    res_dup = sdm.calc_boyce_index(
        pres, bkg, n_bins=3, window_width=0.5, method="pearson"
    )
    assert res_dup["boyce"] == pytest.approx(1.0)


def test_boyce_plot_and_validation(tmp_path):
    pres, bkg = _pres_bg()
    plot_file = str(tmp_path / "boyce.png")
    sdm.calc_boyce_index(pres, bkg, plot_file=plot_file)
    assert (tmp_path / "boyce.png").exists()
    for kwargs in ({"n_bins": 1}, {"window_width": 0.0}, {"method": "kendall"}):
        with pytest.raises(rsgispy.RSGISPyException):
            sdm.calc_boyce_index(pres, bkg, **kwargs)
    with pytest.raises(rsgispy.RSGISPyException):
        sdm.calc_boyce_index([0.5], [0.5, 0.5])


def test_boyce_from_raster_matches_in_memory(tmp_path):
    rng = np.random.default_rng(5)
    size = 60
    vals = rng.uniform(0.0, 1.0, (size, size)).astype(np.float32)
    vals[:3, :] = -1.0  # no data
    img = str(tmp_path / "pred.tif")
    with rasterio.open(
        img, "w", driver="GTiff", height=size, width=size, count=1,
        dtype="float32", crs=CRS, transform=from_origin(0.0, 600.0, 10.0, 10.0),
        nodata=-1.0, tiled=True, blockxsize=16, blockysize=16,
    ) as dst:
        dst.write(vals[np.newaxis])
    valid = vals != -1.0
    rows, cols = np.nonzero(valid)
    prob = vals[valid] ** 3
    pick = rng.choice(rows.size, 150, replace=False, p=prob / prob.sum())
    pres_gdf = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(cols[pick] * 10.0 + 5.0, 600.0 - rows[pick] * 10.0 - 5.0),
        crs=CRS,
    )
    res_r = sdm.calc_boyce_index_from_raster(img, pres_gdf, use_pbar=False)
    res_m = sdm.calc_boyce_index(vals[rows[pick], cols[pick]], vals[valid])
    assert res_r["boyce"] == pytest.approx(res_m["boyce"])
    np.testing.assert_allclose(res_r["pe_ratio"], res_m["pe_ratio"], equal_nan=True)
    assert res_r["boyce"] > 0.8


# --------------------------------------------------------------------------
# calc_sdm_eval_metrics
# --------------------------------------------------------------------------


def test_sdm_eval_metrics_matches_components():
    pres, bkg = _pres_bg()
    y_true = np.r_[np.ones(pres.size), np.zeros(bkg.size)]
    scores = np.r_[pres, bkg]
    met = sdm.calc_sdm_eval_metrics(y_true, scores, threshold=0.5)
    assert met["auc"] == pytest.approx(roc_auc_score(y_true, scores))
    names, vals = accuracy_scores_binary(y_true.astype(int), (scores >= 0.5).astype(int))
    acc = dict(zip(names, vals))
    assert met["tss"] == pytest.approx(acc["Hanssen-Kuiper skill"])
    assert met["kappa"] == pytest.approx(acc["Cohen Kappa"])
    assert met["omission_rate"] == pytest.approx(1.0 - acc["Sensitivity"])
    assert met["commission_rate"] == pytest.approx(1.0 - acc["Specificity"])
    assert met["threshold"] == 0.5 and "boyce" not in met
    assert met["n_presence"] == pres.size and met["n_absence"] == bkg.size

    auto = sdm.calc_sdm_eval_metrics(y_true, scores, bg_scores="absences")
    assert auto["threshold"] == pytest.approx(sdm.calc_binary_threshold(y_true, scores))
    assert auto["boyce"] == pytest.approx(sdm.calc_boyce_index(pres, bkg)["boyce"])
    with_bg = sdm.calc_sdm_eval_metrics(y_true, scores, bg_scores=bkg)
    assert with_bg["boyce"] == pytest.approx(auto["boyce"])
    # -1 / 1 labels work too.
    alt = sdm.calc_sdm_eval_metrics(np.where(y_true == 1, 1, -1), scores, threshold=0.5)
    assert alt["auc"] == pytest.approx(met["auc"])


def test_sdm_eval_metrics_validation():
    with pytest.raises(rsgispy.RSGISPyException):
        sdm.calc_sdm_eval_metrics([1, 0], [0.5])
    with pytest.raises(rsgispy.RSGISPyException):
        sdm.calc_sdm_eval_metrics([1, 0], [0.5, np.nan])
    with pytest.raises(rsgispy.RSGISPyException):
        sdm.calc_sdm_eval_metrics([1, 0], [0.6, 0.4], threshold_method="bogus")
    with pytest.raises(rsgispy.RSGISPyException):
        sdm.calc_sdm_eval_metrics([1, 0], [0.6, 0.4], bg_scores="everything")


# --------------------------------------------------------------------------
# Spatial cross-validation in the k-fold fitting functions
# --------------------------------------------------------------------------


def _spatial_samples(n=400, seed=0):
    rng = np.random.default_rng(seed)
    x_crd = rng.uniform(0.0, 10000.0, n)
    y_crd = rng.uniform(0.0, 10000.0, n)
    v1 = x_crd / 1000.0 + rng.normal(0, 1, n)
    v2 = y_crd / 2000.0 + rng.normal(0, 1, n)
    clsid = (v1 + v2 + rng.normal(0, 1.5, n) > 7.5).astype(int)
    gdf = gpd.GeoDataFrame(
        {"v1": v1, "v2": v2, "clsid": clsid, "grp": (x_crd // 2500).astype(int)},
        geometry=gpd.points_from_xy(x_crd, y_crd),
        crs=CRS,
    )
    return gdf.iloc[:300].copy(), gdf.iloc[300:].copy()


def test_kfold_default_unchanged_and_scores(capsys):
    train, test = _spatial_samples()
    mdls_a = sdm.fit_kfold_sklearn_mdls(
        LogisticRegression(), train, None, test, None, ["v1", "v2"], n_kfolds=4,
        rnd_seed=3, use_pbar=False,
    )
    out_a = capsys.readouterr().out
    mdls_b, scores = sdm.fit_kfold_sklearn_mdls(
        LogisticRegression(), train, None, test, None, ["v1", "v2"], n_kfolds=4,
        rnd_seed=3, use_pbar=False, return_scores=True,
    )
    out_b = capsys.readouterr().out
    assert out_a == out_b and "Validation" not in out_a
    for mdl_a, mdl_b in zip(mdls_a, mdls_b):
        np.testing.assert_array_equal(mdl_a.coef_, mdl_b.coef_)
    assert list(scores.columns) == [
        "fold", "n_train", "n_val", "train_auc", "train_tss", "val_auc",
        "val_tss", "val_boyce", "test_auc", "test_tss",
    ]
    assert len(scores) == 4 and scores["val_auc"].isna().all()
    assert scores["test_auc"].between(0.5, 1.0).all()
    assert (scores["n_train"] == 180).all()


def test_kfold_spatial_cv(capsys):
    train, test = _spatial_samples()
    cv = SpatialBlockKFold(n_splits=4, block_size=2500.0, stratify=True, rnd_seed=1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mdls, scores = sdm.fit_kfold_sklearn_mdls(
            LogisticRegression(), train, None, test, None, ["v1", "v2"],
            rnd_seed=3, use_pbar=False, cv=cv, return_scores=True,
        )
    assert "Validation (out-of-fold) ROC" in capsys.readouterr().out
    assert len(mdls) == 4 and len(scores) == 4
    assert (scores["n_train"] + scores["n_val"] == len(train)).all()
    assert scores[["val_auc", "val_tss", "test_auc"]].notna().all().all()
    # Each member is fitted only on its fold's training rows.
    coords = np.column_stack([train.geometry.x, train.geometry.y])
    splits = list(cv.split(train[["v1", "v2"]], train["clsid"].to_numpy(), coords=coords))
    for (tr_idx, _), mdl in zip(splits, mdls):
        ref = LogisticRegression().fit(train[["v1", "v2"]].iloc[tr_idx], train["clsid"].iloc[tr_idx])
        np.testing.assert_allclose(mdl.coef_, ref.coef_, rtol=1e-6)

    # Integer cv -> SpatialBlockKFold; ignored arguments warn.
    with pytest.warns(UserWarning, match="ignored"):
        mdls_int = sdm.fit_kfold_sklearn_mdls(
            LogisticRegression(), train, None, test, None, ["v1", "v2"],
            n_kfolds=3, use_pbar=False, cv=3,
        )
    assert len(mdls_int) == 3


def test_kfold_group_and_sklearn_splitters():
    train, test = _spatial_samples()
    mdls, scores = sdm.fit_kfold_sklearn_mdls(
        LogisticRegression(), train, None, test, None, ["v1", "v2"], use_pbar=False,
        cv=GroupKFold(n_splits=4), groups_col="grp", return_scores=True,
    )
    assert len(mdls) == 4
    # Each held-out fold is one whole group.
    assert set(scores["n_val"].astype(int)) == set(train["grp"].value_counts().to_numpy())
    mdls_k = sdm.fit_kfold_sklearn_mdls(
        LogisticRegression(), train, None, test, None, ["v1", "v2"], use_pbar=False,
        cv=KFold(n_splits=5, shuffle=True, random_state=0),
    )
    assert len(mdls_k) == 5


def test_kfold_cv_errors():
    train, test = _spatial_samples()
    for bad_cv in (1, "blocks", True):
        with pytest.raises(rsgispy.RSGISPyException, match="cv must be"):
            sdm.fit_kfold_sklearn_mdls(
                LogisticRegression(), train, None, test, None, ["v1", "v2"],
                use_pbar=False, cv=bad_cv,
            )
    # A fold whose held-out rows contain no presences raises a clear error.
    lopsided = train.copy()
    lopsided["clsid"] = np.where(lopsided.geometry.x < 5000.0, 0, lopsided["clsid"])
    lopsided.loc[lopsided.index[lopsided.geometry.x >= 5000.0][:5], "clsid"] = 0
    with pytest.raises(rsgispy.RSGISPyException, match="no presences"):
        sdm.fit_kfold_sklearn_mdls(
            LogisticRegression(), lopsided, None, test, None, ["v1", "v2"],
            use_pbar=False, cv=GroupKFold(n_splits=4), groups_col="grp",
        )


def test_kfold_sgl_cls_spatial_cv():
    train, test = _spatial_samples()
    pres_only = train[train["clsid"] == 1]
    mdls_plain = sdm.fit_kfold_sklearn_sgl_cls_mdls(
        IsolationForest(n_estimators=20), pres_only, None, test, None, ["v1", "v2"],
        n_kfolds=2, rnd_seed=1, use_pbar=False,
    )
    assert len(mdls_plain) == 2
    with pytest.raises(rsgispy.RSGISPyException, match="class column"):
        sdm.fit_kfold_sklearn_sgl_cls_mdls(
            IsolationForest(), pres_only.drop(columns=["clsid"]), None, test, None,
            ["v1", "v2"], use_pbar=False, cv=3,
        )
    cv = SpatialBlockKFold(n_splits=3, block_size=2500.0, stratify=True, rnd_seed=2)
    mdls, scores = sdm.fit_kfold_sklearn_sgl_cls_mdls(
        IsolationForest(n_estimators=20), train, None, test, None, ["v1", "v2"],
        rnd_seed=1, use_pbar=False, cv=cv, return_scores=True,
    )
    assert len(mdls) == 3
    # Members are fitted on the presences of each training fold only.
    coords = np.column_stack([train.geometry.x, train.geometry.y])
    y_arr = train["clsid"].to_numpy()
    for (tr_idx, _), n_tr in zip(cv.split(train, y_arr, coords=coords), scores["n_train"]):
        assert n_tr == int(np.sum(y_arr[tr_idx] == 1))
    assert scores["val_auc"].notna().all()
    assert isinstance(scores, pd.DataFrame)
