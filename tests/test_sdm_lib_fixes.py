"""Tests for stratified spatial block CV with rare classes, the SHAP linear
explainer perturbation option / linear-model detection, and the Sobol CSV
outputs keeping the variable names."""

import os

import numpy as np
import pandas as pd
import pytest

import rsgispy
from rsgispy.ml.cv import SpatialBlockKFold

sklearn = pytest.importorskip("sklearn")
from sklearn.linear_model import LogisticRegression  # noqa: E402


def _find_tutorial_train():
    """Find the case-study training set in a sibling rsgispy-tutorials checkout."""
    rel = os.path.join(
        "rsgispy-tutorials", "11_case_study_species_distribution", "outputs",
        "sdm_LogReg_outs", "sdm_train.gpkg",
    )
    cur = os.path.dirname(os.path.abspath(__file__))
    for _ in range(6):
        cur = os.path.dirname(cur)
        cand = os.path.join(cur, rel)
        if os.path.exists(cand):
            return cand
    return ""


TUTORIAL_TRAIN = _find_tutorial_train()


def _imbalanced_coords():
    """7000 background points over 10 x 10 km and 23 presences in 10 blocks
    (one block holding 9), as in the Snowdon beetle case study."""
    rng = np.random.default_rng(3)
    bg = rng.uniform(0.0, 10000.0, size=(7000, 2))
    pres_counts = [9, 3, 2, 2, 1, 1, 1, 1, 1, 2]
    pres = []
    for i, n_p in enumerate(pres_counts):
        cx, cy = (i % 5) * 2000.0 + 500.0, (i // 5) * 4000.0 + 500.0
        pres.append(rng.uniform(-400.0, 400.0, size=(n_p, 2)) + [cx, cy])
    pres = np.vstack(pres)
    coords = np.vstack([bg, pres])
    y = np.r_[np.zeros(len(bg), dtype=int), np.ones(len(pres), dtype=int)]
    return coords, y


@pytest.mark.parametrize("n_splits", [3, 5, 10])
def test_stratified_block_cv_spreads_rare_class(n_splits):
    coords, y = _imbalanced_coords()
    cv = SpatialBlockKFold(
        n_splits=n_splits, block_size=1000.0, stratify=True, rnd_seed=42
    )
    folds = list(cv.split(coords, y, coords=coords))
    assert len(folds) == n_splits
    pres_per_fold = [int((y[val] == 1).sum()) for _, val in folds]
    assert all(n_p > 0 for n_p in pres_per_fold), pres_per_fold
    assert sum(pres_per_fold) == int(y.sum())
    # The majority class stays balanced across folds
    bg_per_fold = np.array([int((y[val] == 0).sum()) for _, val in folds])
    assert bg_per_fold.max() - bg_per_fold.min() < 0.1 * bg_per_fold.mean()


def test_stratified_block_cv_fewer_rare_blocks_than_folds():
    # Only 2 blocks hold presences: 2 folds get them, the rest none, no error.
    coords, y = _imbalanced_coords()
    keep = (y == 0) | (np.arange(len(y)) < 7000 + 12)
    coords, y = coords[keep], y[keep]
    cv = SpatialBlockKFold(n_splits=5, block_size=1000.0, stratify=True, rnd_seed=1)
    pres_per_fold = [int((y[val] == 1).sum()) for _, val in cv.split(coords, y, coords=coords)]
    assert sum(pres_per_fold) == int(y.sum())
    assert sum(n_p > 0 for n_p in pres_per_fold) == 2


@pytest.mark.skipif(
    not os.path.exists(TUTORIAL_TRAIN), reason="rsgispy-tutorials not available"
)
def test_stratified_block_cv_tutorial_training_set():
    gpd = pytest.importorskip("geopandas")
    gdf = gpd.read_file(TUTORIAL_TRAIN, layer="train")
    y = gdf["clsid"].to_numpy()
    coords = np.c_[gdf.geometry.x, gdf.geometry.y]
    cv = SpatialBlockKFold(n_splits=5, block_size=1000.0, stratify=True, rnd_seed=42)
    pres_per_fold = [int((y[val] == 1).sum()) for _, val in cv.split(coords, y, coords=coords)]
    assert all(n_p > 0 for n_p in pres_per_fold), pres_per_fold


# --------------------------------------------------------------------------
# SHAP
# --------------------------------------------------------------------------


def _correlated_logreg():
    rng = np.random.default_rng(0)
    a = rng.normal(size=300)
    b = a + 0.1 * rng.normal(size=300)
    x_df = pd.DataFrame({"a": a, "b": b})
    y = (a + 0.3 * rng.normal(size=300) > 0).astype(int)
    mdl = LogisticRegression().fit(x_df, y)
    mdl.coef_[0, 1] = 0.0  # b has no effect on the model
    return mdl, x_df, y


def test_shap_linear_perturbation_option():
    pytest.importorskip("shap")
    from rsgispy.tools._ml_explain import run_shap_explainer

    mdl, x_df, _ = _correlated_logreg()
    kw = dict(
        est_cls_obj=mdl, train_x=x_df, analysis_vars=["a", "b"],
        feature_name="test", extra="modelling", use_linear_explainer=True,
    )
    interv = run_shap_explainer(**kw)
    corr = run_shap_explainer(**kw, linear_perturbation="correlation_dependent")
    interv_b = float(interv.set_index(interv.columns[0]).iloc[:, 0]["b"])
    corr_b = float(corr.set_index(corr.columns[0]).iloc[:, 0]["b"])
    assert interv_b == pytest.approx(0.0, abs=1e-10)
    assert corr_b > 0.1

    with pytest.raises(rsgispy.RSGISPyException):
        run_shap_explainer(**kw, linear_perturbation="bad")


def test_shap_default_is_interventional_in_public_api(tmp_path, monkeypatch):
    pytest.importorskip("shap")
    import geopandas as gpd

    import rsgispy.modelling.species_distribution as sdm
    import rsgispy.tools._ml_explain as ml_explain

    # shap_sklearn_mdl_explainer returns None, so capture the SHAP values it
    # computes (the same object the summary plot is drawn from).
    captured = {}
    orig_mean_abs = ml_explain._compute_shap_mean_abs_df

    def _spy(raw_shap, feature_names, class_labels=None):
        captured["raw"] = raw_shap
        captured["summary"] = orig_mean_abs(raw_shap, feature_names, class_labels)
        return captured["summary"]

    monkeypatch.setattr(ml_explain, "_compute_shap_mean_abs_df", _spy)

    mdl, x_df, _ = _correlated_logreg()
    gdf = gpd.GeoDataFrame(
        x_df, geometry=gpd.points_from_xy(x_df["a"], x_df["b"]), crs="EPSG:27700"
    )
    out_plot = tmp_path / "shap_summary.png"
    sdm.shap_sklearn_mdl_explainer(
        mdl, gdf, None, ["a", "b"], shap_summary_plot=str(out_plot),
        use_linear_explainer=True, linear_perturbation="interventional",
    )
    assert out_plot.exists() and out_plot.stat().st_size > 0

    raw = captured["raw"]
    vals = np.asarray(raw.values, dtype=float)
    assert vals.shape == (len(x_df), 2)  # one value per sample and variable
    assert np.all(np.isfinite(vals))
    assert list(raw.feature_names) == ["a", "b"]
    # Interventional linear SHAP: value = coef * (x - mean(x)), so "b"
    # (coefficient forced to 0) gets exactly 0 and "a" is coef_a * (a - mean).
    np.testing.assert_allclose(vals[:, 1], 0.0, atol=1e-10)
    np.testing.assert_allclose(
        vals[:, 0], mdl.coef_[0, 0] * (x_df["a"] - x_df["a"].mean()), rtol=1e-6,
        atol=1e-10,
    )
    # Additivity: base value + sum of SHAP values = the explained log-odds.
    base = np.broadcast_to(np.asarray(raw.base_values, dtype=float), (len(x_df),))
    np.testing.assert_allclose(
        base + vals.sum(axis=1), mdl.decision_function(x_df), rtol=1e-6, atol=1e-8
    )
    summary = captured["summary"]
    assert list(summary["feature"]) == ["a", "b"]  # sorted by importance
    assert summary["mean_abs_shap"].iloc[0] > 0.1
    assert summary["mean_abs_shap"].iloc[1] == pytest.approx(0.0, abs=1e-10)


def test_shap_auto_explainer_linear_detection():
    shap = pytest.importorskip("shap")
    from rsgispy.modelling.species_distribution import MaxentClassifier
    from rsgispy.tools._ml_explain import _build_shap_explainer, _is_linear_in_inputs

    mdl, x_df, y = _correlated_logreg()
    assert _is_linear_in_inputs(mdl, x_df, "classification")
    expl = _build_shap_explainer(mdl, x_df, ["a", "b"], use_tree_explainer=None)
    assert isinstance(expl, shap.LinearExplainer)

    # MaxentClassifier has coef_ but for the derived features, not the inputs:
    # it must fall back to the model-agnostic explainer.
    mx = MaxentClassifier().fit(x_df, y)
    assert hasattr(mx, "coef_")
    assert not _is_linear_in_inputs(mx, x_df, "classification")
    expl = _build_shap_explainer(
        mx, x_df.iloc[:20], ["a", "b"], use_tree_explainer=None
    )
    assert not isinstance(expl, shap.LinearExplainer)
    assert isinstance(expl, shap.KernelExplainer)


# --------------------------------------------------------------------------
# Sobol
# --------------------------------------------------------------------------


def test_sobol_csvs_keep_variable_names(tmp_path):
    pytest.importorskip("SALib")
    import geopandas as gpd

    import rsgispy.modelling.species_distribution as sdm

    mdl, x_df, _ = _correlated_logreg()
    x_df["c"] = np.random.default_rng(1).normal(size=len(x_df))
    mdl = LogisticRegression().fit(x_df, (x_df["a"] > 0).astype(int))
    gdf = gpd.GeoDataFrame(
        x_df, geometry=gpd.points_from_xy(x_df["a"], x_df["b"]), crs="EPSG:27700"
    )
    files = {k: tmp_path / f"sobol_{k}.csv" for k in ("total", "first", "second")}
    total_si, first_si, second_si = sdm.salib_sklearn_mdl_sensitivity(
        mdl, gdf, None, ["a", "b", "c"], n_samples=64,
        sobol_overall_file=str(files["total"]),
        sobol_first_file=str(files["first"]),
        sobol_second_file=str(files["second"]),
    )
    total_csv = pd.read_csv(files["total"])
    first_csv = pd.read_csv(files["first"])
    second_csv = pd.read_csv(files["second"])
    assert list(total_csv["variable"]) == ["a", "b", "c"]
    assert list(first_csv["variable"]) == ["a", "b", "c"]
    assert {"ST", "ST_conf"} <= set(total_csv.columns)
    assert {"S1", "S1_conf"} <= set(first_csv.columns)
    pairs = list(zip(second_csv["variable_1"], second_csv["variable_2"]))
    assert pairs == [("a", "b"), ("a", "c"), ("b", "c")]
    np.testing.assert_allclose(total_csv["ST"], total_si["ST"].to_numpy())
    assert total_si.loc["a", "ST"] > total_si.loc["b", "ST"]
    assert len(second_si) == 3
