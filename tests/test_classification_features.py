"""
Tests for rsgispy.classification.classfeatures: exploratory plots, correlation &
VIF multicollinearity diagnostics, feature selection methods, feature importance,
jackknife analysis, response curves, SHAP contributions, and SALib Sobol
sensitivity.
"""

import importlib.util
import inspect
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

import rsgispy
import rsgispy.classification as rcls
import rsgispy.classification.classdata as rcd
import rsgispy.classification.classfeatures as rcf
import rsgispy.classification.classsklearn as rcsk

_HAS_PARQUET = (
    importlib.util.find_spec("pyarrow") is not None
    or importlib.util.find_spec("fastparquet") is not None
)


def _make_synthetic_dataset(
    tmp_path: Path,
    n_per_class: int = 40,
    seed: int = 42,
    include_collinear: bool = True,
):
    """Build deterministic 3-class and 2-class tabular datasets + configs."""
    rng = np.random.default_rng(seed)

    # Three well-separated classes using f_signal1, f_signal2, plus noisy & collinear cols.
    centers = [
        (0, 10, "Forest", (1.0, 8.0, 2.0)),
        (1, 50, "Water", (8.0, 1.0, 5.0)),
        (2, 300, "Urban", (8.0, 8.0, 8.0)),
    ]

    rows = []
    for int_id, out_id, cls_name, (mu1, mu2, mu3) in centers:
        f1 = rng.normal(loc=mu1, scale=0.45, size=n_per_class)
        f2 = rng.normal(loc=mu2, scale=0.45, size=n_per_class)
        f3 = rng.normal(loc=mu3, scale=0.60, size=n_per_class)
        f_noise = rng.normal(loc=5.0, scale=0.1, size=n_per_class)
        for idx in range(n_per_class):
            row = {
                "f1": float(f1[idx]),
                "f2": float(f2[idx]),
                "f3": float(f3[idx]),
                "f_noise": float(f_noise[idx]),
                "internal_cls_id": int_id,
                "out_cls_id": out_id,
                "class_name": cls_name,
            }
            if include_collinear:
                row["f1_collinear"] = float(2.0 * f1[idx] + 1.0)
            rows.append(row)

    full_df = pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    split_idx = int(0.7 * len(full_df))
    train_df = full_df.iloc[:split_idx].reset_index(drop=True)
    test_df = full_df.iloc[split_idx:].reset_index(drop=True)

    feat_cols = ["f1", "f2", "f3", "f_noise"] + (
        ["f1_collinear"] if include_collinear else []
    )

    dummy_tif = str(tmp_path / "dummy.tif")
    dummy_vec = str(tmp_path / "dummy.gpkg")
    img_info = [
        rcd.ImageBandInfo(
            dummy_tif,
            name="FeatStack",
            bands=list(range(1, len(feat_cols) + 1)),
            band_names=feat_cols,
        )
    ]
    class_info = {
        "Forest": rcd.ClassInfoObj(
            out_id=10, internal_id=0, vec_file=dummy_vec, vec_lyr="forest", green=180
        ),
        "Water": rcd.ClassInfoObj(
            out_id=50, internal_id=1, vec_file=dummy_vec, vec_lyr="water", blue=255
        ),
        "Urban": rcd.ClassInfoObj(
            out_id=300, internal_id=2, vec_file=dummy_vec, vec_lyr="urban", red=220
        ),
    }

    cfg_path = tmp_path / "cls_cfg.json"
    cfg = rcd.write_cls_config(cfg_path, img_info, class_info)

    ext = ".parquet" if _HAS_PARQUET else ".csv"
    train_pq = tmp_path / f"train_tbl{ext}"
    test_pq = tmp_path / f"test_tbl{ext}"
    train_csv = tmp_path / "train.csv"
    rcd.write_cls_samples(train_df, train_pq)
    rcd.write_cls_samples(test_df, test_pq)
    rcd.write_cls_samples(train_df, train_csv, out_format="csv")

    return {
        "cfg": cfg,
        "cfg_path": cfg_path,
        "class_info": class_info,
        "feat_cols": feat_cols,
        "train_df": train_df,
        "test_df": test_df,
        "train_pq": train_pq,
        "test_pq": test_pq,
        "train_csv": train_csv,
    }


def test_classfeatures_module_layout():
    """Verify classfeatures exposes expected API and is not re-exported at top level."""
    assert inspect.ismodule(rcf)
    expected = {
        "comparison_box_plots",
        "correlation_matrix",
        "calc_vif_multicollinearity",
        "select_features_cqv",
        "select_features_corr",
        "select_features_vif",
        "select_features_univariate",
        "select_features_rfe",
        "select_features_from_model",
        "select_features_boruta_shap",
        "calc_feature_importance",
        "calc_jackknife_feature_importance",
        "calc_feature_response_curves",
        "calc_shap_feature_contributions",
        "calc_salib_feature_sensitivity",
    }
    assert set(rcf.__all__) == expected
    for name in expected:
        assert callable(getattr(rcf, name))
        assert not hasattr(rcls, name)


def test_diagnostics_boxplots_corr_and_vif(tmp_path):
    """Test comparison_box_plots, correlation_matrix, and calc_vif_multicollinearity."""
    fix = _make_synthetic_dataset(tmp_path)

    # 1. Violin plots grouped by class name column.
    violin_dir = tmp_path / "violins"
    rcf.comparison_box_plots(
        fix["train_pq"],
        cls_cfg=fix["cfg_path"],
        feature_cols=["f1", "f2"],
        out_dir=violin_dir,
        box_plots=False,
        use_pbar=False,
    )
    assert (violin_dir / "f1_comp_plot.png").exists()
    assert (violin_dir / "f2_comp_plot.png").exists()

    # 2. Box plots mapped from class_info when cls_name column is stripped.
    no_name_df = fix["train_df"].drop(columns=["class_name"])
    box_dir = tmp_path / "boxes"
    rcf.comparison_box_plots(
        no_name_df,
        cls_cfg=fix["class_info"],
        feature_cols=["f1", "f3"],
        out_dir=box_dir,
        box_plots=True,
        use_pbar=False,
    )
    assert (box_dir / "f1_comp_plot.png").exists()
    assert (box_dir / "f3_comp_plot.png").exists()

    # 3. Grouping by ID column without class_info.
    id_box_dir = tmp_path / "id_boxes"
    rcf.comparison_box_plots(
        no_name_df,
        cls_cfg=None,
        feature_cols=["f1"],
        out_dir=id_box_dir,
        use_class_names=False,
        use_internal_id=True,
        box_plots=True,
        use_pbar=False,
    )
    assert (id_box_dir / "f1_comp_plot.png").exists()

    # 4. Correlation matrix (Pearson & Spearman) + CSV & heatmap PNG.
    corr_csv = tmp_path / "corr.csv"
    corr_png = tmp_path / "corr.png"
    corr_df = rcf.correlation_matrix(
        fix["train_pq"],
        cls_cfg=fix["cfg_path"],
        method="spearman",
        out_corr_file=corr_csv,
        out_plt_file=corr_png,
    )
    assert corr_csv.exists()
    assert corr_png.exists()
    assert corr_df.loc["f1", "f1_collinear"] == pytest.approx(1.0, abs=1e-6)

    with pytest.raises(rsgispy.RSGISPyException):
        rcf.correlation_matrix(fix["train_df"], method="invalid_corr")

    # 5. VIF multicollinearity.
    vif_csv = tmp_path / "vif.csv"
    vif_series = rcf.calc_vif_multicollinearity(
        fix["train_df"],
        cls_cfg=fix["cfg"],
        out_vif_file=vif_csv,
    )
    assert vif_csv.exists()
    assert set(vif_series.index) == set(fix["feat_cols"])
    assert plt.get_fignums() == []


def test_feature_selection_cqv_corr_and_vif(tmp_path):
    """Test CQV, correlation cluster, and iterative VIF feature selection."""
    fix = _make_synthetic_dataset(tmp_path, include_collinear=True)

    # Build a dataframe with varied CQV across features (all positive).
    cqv_df = pd.DataFrame(
        {
            "const_like": np.linspace(100.0, 102.0, 30),
            "mid_spread": np.linspace(10.0, 30.0, 30),
            "wide_spread": np.linspace(1.0, 50.0, 30),
            "internal_cls_id": [0] * 10 + [1] * 10 + [2] * 10,
        }
    )
    kept_cqv = rcf.select_features_cqv(
        cqv_df,
        feature_cols=["const_like", "mid_spread", "wide_spread"],
        low_threshold=0.20,
        high_threshold=0.70,
    )
    assert "mid_spread" in kept_cqv
    assert "const_like" not in kept_cqv

    with pytest.raises(rsgispy.RSGISPyException):
        rcf.select_features_cqv(cqv_df, low_threshold=0.8, high_threshold=0.2)

    # Correlation clustering selection across 5 features.
    selected_corr = rcf.select_features_corr(
        fix["train_pq"],
        cls_cfg=fix["cfg"],
        n_min_clusters=2,
        n_max_clusters=5,
        use_pbar=False,
    )
    assert isinstance(selected_corr, list)
    assert 2 <= len(selected_corr) <= 4
    assert all(col in fix["feat_cols"] for col in selected_corr)

    with pytest.raises(rsgispy.RSGISPyException):
        rcf.select_features_corr(
            fix["train_df"],
            cls_cfg=fix["cfg"],
            feature_cols=["f1", "f2"],
            use_pbar=False,
        )

    # Iterative VIF pruning drops exact linear collinearity (f1 or f1_collinear).
    vif_out_csv = tmp_path / "selected_vif.csv"
    kept_vif, final_vif = rcf.select_features_vif(
        fix["train_df"],
        cls_cfg=fix["cfg"],
        vif_threshold=5.0,
        min_features=2,
        out_vif_file=vif_out_csv,
        return_vif=True,
    )
    assert vif_out_csv.exists()
    assert len(kept_vif) < len(fix["feat_cols"])
    assert not ("f1" in kept_vif and "f1_collinear" in kept_vif)
    assert len(final_vif) == len(kept_vif)

    with pytest.raises(rsgispy.RSGISPyException):
        rcf.select_features_vif(fix["train_df"], vif_threshold=0.0)
    with pytest.raises(rsgispy.RSGISPyException):
        rcf.select_features_vif(fix["train_df"], min_features=1)


def test_feature_selection_univariate_rfe_and_from_model(tmp_path):
    """Test univariate tests, RFE/RFECV, and SelectFromModel/LassoLars."""
    fix = _make_synthetic_dataset(tmp_path, include_collinear=False)

    # 1. Univariate f_classif (k=2) & scores CSV.
    uni_csv = tmp_path / "uni_scores.csv"
    sel_f, df_f = rcf.select_features_univariate(
        fix["cfg_path"],
        fix["train_pq"],
        score_func="f_classif",
        k=2,
        out_scores_file=uni_csv,
        return_scores=True,
    )
    assert uni_csv.exists()
    assert len(sel_f) == 2
    assert set(sel_f).issubset({"f1", "f2", "f3"})
    assert list(df_f.columns) == ["feature", "score", "p_value", "selected", "rank"]

    # Univariate mutual_info_classif with percentile.
    sel_mi = rcf.select_features_univariate(
        fix["cfg"],
        fix["train_csv"],
        score_func="mutual_info_classif",
        percentile=50,
        rnd_seed=42,
    )
    assert len(sel_mi) == 2

    # Univariate chi2 requires non-negative values; shift values positive to verify.
    pos_df = fix["train_df"].copy()
    for col in ["f1", "f2", "f3", "f_noise"]:
        pos_df[col] = pos_df[col] - pos_df[col].min() + 0.1
    sel_chi = rcf.select_features_univariate(
        fix["cfg"], pos_df, score_func="chi2", k=2
    )
    assert len(sel_chi) == 2

    neg_df = fix["train_df"].copy()
    neg_df.loc[0, "f1"] = -5.0
    with pytest.raises(rsgispy.RSGISPyException):
        rcf.select_features_univariate(fix["cfg"], neg_df, score_func="chi2", k=2)

    # 2. RFE (fixed n_features_to_select).
    rfe_csv = tmp_path / "rfe_rank.csv"
    sel_rfe, rfe_df = rcf.select_features_rfe(
        fix["cfg"],
        RandomForestClassifier(n_estimators=20),
        fix["train_df"],
        n_features_to_select=2,
        rnd_seed=42,
        out_scores_file=rfe_csv,
        return_scores=True,
    )
    assert rfe_csv.exists()
    assert len(sel_rfe) == 2
    assert "f_noise" not in sel_rfe
    assert (rfe_df["ranking"] >= 1).all()

    # RFECV with CV plot (test step=1 and step=2).
    rfecv_png = tmp_path / "rfecv.png"
    sel_rfecv = rcf.select_features_rfe(
        fix["cfg"],
        RandomForestClassifier(n_estimators=20),
        fix["train_pq"],
        use_cv=True,
        cv=3,
        step=1,
        min_features_to_select=1,
        rnd_seed=42,
        out_cv_plot=rfecv_png,
    )
    assert rfecv_png.exists()
    assert len(sel_rfecv) >= 1

    rfecv_step2_png = tmp_path / "rfecv_step2.png"
    sel_rfecv_s2 = rcf.select_features_rfe(
        fix["cfg"],
        RandomForestClassifier(n_estimators=15),
        fix["train_pq"],
        use_cv=True,
        cv=3,
        step=2,
        min_features_to_select=1,
        rnd_seed=42,
        out_cv_plot=rfecv_step2_png,
    )
    assert rfecv_step2_png.exists()
    assert len(sel_rfecv_s2) >= 1

    # 3. SelectFromModel (unfitted + prefit + linear coef_ + LassoLars).
    sfm_csv = tmp_path / "sfm.csv"
    sel_sfm, sfm_df = rcf.select_features_from_model(
        fix["cfg"],
        RandomForestClassifier(n_estimators=30),
        fix["train_df"],
        threshold="mean",
        rnd_seed=42,
        out_scores_file=sfm_csv,
        return_scores=True,
    )
    assert sfm_csv.exists()
    assert len(sel_sfm) >= 1
    assert "importance" in sfm_df.columns

    fitted_clf, _ = rcsk.train_sklearn_classifier(
        fix["cfg"],
        RandomForestClassifier(n_estimators=25),
        fix["train_df"],
        rnd_seed=42,
        return_model=True,
    )
    sel_prefit = rcf.select_features_from_model(
        fix["cfg"],
        fitted_clf,
        fix["train_df"],
        prefit=True,
        max_features=2,
    )
    assert 1 <= len(sel_prefit) <= 2

    sel_lr = rcf.select_features_from_model(
        fix["cfg"],
        LogisticRegression(max_iter=500),
        fix["train_df"],
        threshold="median",
        rnd_seed=42,
    )
    assert len(sel_lr) >= 1

    lasso_csv = tmp_path / "lasso.csv"
    sel_lasso, lasso_df = rcf.select_features_from_model(
        fix["cfg"],
        None,
        fix["train_df"],
        use_lassolars=True,
        out_scores_file=lasso_csv,
        return_scores=True,
    )
    assert lasso_csv.exists()
    assert isinstance(sel_lasso, list)
    assert set(lasso_df.columns) == {"feature", "selected"}
    assert plt.get_fignums() == []


def test_feature_importance_and_jackknife(tmp_path):
    """Test permutation/model feature importance and jackknife evaluation."""
    fix = _make_synthetic_dataset(tmp_path, include_collinear=False)

    clf, _ = rcsk.train_sklearn_classifier(
        fix["cfg"],
        RandomForestClassifier(n_estimators=40),
        fix["train_pq"],
        rnd_seed=42,
        return_model=True,
    )

    # Method='both' with CSV & plot.
    imp_csv = tmp_path / "imp_both.csv"
    imp_png = tmp_path / "imp_both.png"
    imp_df = rcf.calc_feature_importance(
        fix["cfg"],
        clf,
        fix["test_pq"],
        method="both",
        n_repeats=5,
        rnd_seed=42,
        out_importance_file=imp_csv,
        out_plot_file=imp_png,
    )
    assert imp_csv.exists()
    assert imp_png.exists()
    assert {
        "feature",
        "model_importance",
        "importance_mean",
        "importance_std",
        "rank",
    }.issubset(imp_df.columns)
    assert imp_df.iloc[-1]["feature"] == "f_noise"

    # Method='model' on estimator without native importance fails cleanly.
    knn, _ = rcsk.train_sklearn_classifier(
        fix["cfg"],
        KNeighborsClassifier(n_neighbors=3),
        fix["train_df"],
        return_model=True,
    )
    with pytest.raises(rsgispy.RSGISPyException):
        rcf.calc_feature_importance(fix["cfg"], knn, fix["test_df"], method="model")

    knn_perm = rcf.calc_feature_importance(
        fix["cfg"], knn, fix["test_df"], method="permutation", n_repeats=3, rnd_seed=42
    )
    assert "importance_mean" in knn_perm.columns

    # Jackknife feature importance.
    jack_csv = tmp_path / "jackknife.csv"
    acc_png = tmp_path / "jack_acc.png"
    f1_png = tmp_path / "jack_f1.png"
    jack_df = rcf.calc_jackknife_feature_importance(
        fix["cfg_path"],
        RandomForestClassifier(n_estimators=25),
        fix["train_pq"],
        fix["test_pq"],
        rnd_seed=42,
        out_var_importance_file=jack_csv,
        out_importance_acc_plot=acc_png,
        out_importance_f1_plot=f1_png,
        use_pbar=False,
    )
    assert jack_csv.exists()
    assert acc_png.exists()
    assert f1_png.exists()
    assert list(jack_df.index) == fix["feat_cols"]
    assert {
        "train_acc_leave_var_out",
        "test_acc_leave_var_out",
        "test_bal_acc_leave_var_out",
        "test_f1_macro_leave_var_out",
        "test_kappa_leave_var_out",
        "test_acc_only_var",
        "all_vars_test_accuracy",
    }.issubset(jack_df.columns)
    assert plt.get_fignums() == []


def test_feature_response_curves(tmp_path):
    """Test multiclass, target-class, and decision-function response curves."""
    fix = _make_synthetic_dataset(tmp_path, include_collinear=False)

    rf_clf, _ = rcsk.train_sklearn_classifier(
        fix["cfg"],
        RandomForestClassifier(n_estimators=25),
        fix["train_df"],
        rnd_seed=42,
        return_model=True,
    )

    # Multiclass response curves (all classes + single target_class by out_id).
    resp_csv = tmp_path / "resp_all.csv"
    resp_dir = tmp_path / "resp_plots"
    resp_df = rcf.calc_feature_response_curves(
        fix["cfg"],
        rf_clf,
        fix["train_df"],
        n_samples=25,
        out_file=resp_csv,
        response_plots_dir=resp_dir,
        use_pbar=False,
    )
    assert resp_csv.exists()
    assert len(resp_df) == 25
    assert "f1_Forest_prob" in resp_df.columns
    assert "f1_Water_prob" in resp_df.columns
    assert "f1_Urban_prob" in resp_df.columns
    assert (resp_dir / "f1_response_curve.png").exists()

    # Resolve target_class via output ID (300 -> internal ID 2 -> Urban).
    resp_urban = rcf.calc_feature_response_curves(
        fix["cfg"],
        rf_clf,
        fix["train_df"],
        target_class=300,
        summary_stat="median",
        n_samples=15,
        use_pbar=False,
    )
    assert "f1_Urban_prob" in resp_urban.columns
    assert "f1_Forest_prob" not in resp_urban.columns

    # Non-predict_proba fallback (LinearSVC with decision_function).
    svc_clf, _ = rcsk.train_sklearn_classifier(
        fix["cfg"],
        LinearSVC(max_iter=2000),
        fix["train_df"],
        rnd_seed=42,
        return_model=True,
    )
    resp_svc = rcf.calc_feature_response_curves(
        fix["cfg"], svc_clf, fix["train_df"], n_samples=10, use_pbar=False
    )
    assert len(resp_svc) == 10
    assert plt.get_fignums() == []


def test_shap_feature_contributions(tmp_path):
    """Test multiclass and binary SHAP contributions when shap is installed."""
    pytest.importorskip("shap")
    fix = _make_synthetic_dataset(tmp_path, include_collinear=False)

    rf_clf, _ = rcsk.train_sklearn_classifier(
        fix["cfg"],
        RandomForestClassifier(n_estimators=25),
        fix["train_df"],
        rnd_seed=42,
        return_model=True,
    )

    # Multiclass SHAP contributions (target_class=None -> multiclass summary bar + plots).
    shap_csv = tmp_path / "shap_summary.csv"
    shap_sum_png = tmp_path / "shap_summary.png"
    shap_heat_png = tmp_path / "shap_heatmap.png"
    shap_scat_dir = tmp_path / "shap_scatter"
    shap_dep_dir = tmp_path / "shap_depend"
    shap_df = rcf.calc_shap_feature_contributions(
        fix["cfg"],
        rf_clf,
        fix["train_df"],
        target_class=None,
        shap_summary_plot=shap_sum_png,
        shap_heatmap_plot=shap_heat_png,
        shap_scatter_plots_dir=shap_scat_dir,
        shap_depend_plots_dir=shap_dep_dir,
        out_shap_file=shap_csv,
        subsample_n_smpls=18,
        use_tree_explainer=True,
        rnd_seed=42,
    )
    assert shap_csv.exists()
    assert shap_sum_png.exists()
    assert shap_heat_png.exists()
    assert (shap_scat_dir / "shap_scatter_f1.png").exists()
    assert (shap_dep_dir / "shap_depends_f1.png").exists()
    assert {
        "feature",
        "mean_abs_shap",
        "mean_abs_shap_Forest",
        "mean_abs_shap_Water",
        "mean_abs_shap_Urban",
    }.issubset(shap_df.columns)

    # Multiclass SHAP focused on target_class='Water'.
    shap_water_png = tmp_path / "shap_water_summary.png"
    rcf.calc_shap_feature_contributions(
        fix["cfg"],
        rf_clf,
        fix["train_df"],
        target_class="Water",
        shap_summary_plot=shap_water_png,
        subsample_n_smpls=15,
        use_tree_explainer=True,
        rnd_seed=42,
    )
    assert shap_water_png.exists()

    # Binary LogisticRegression with LinearExplainer.
    bin_df = fix["train_df"][fix["train_df"]["internal_cls_id"] < 2].copy()
    bin_classes = {k: fix["class_info"][k] for k in ("Forest", "Water")}
    lr_clf, _ = rcsk.train_sklearn_classifier(
        bin_classes,
        LogisticRegression(max_iter=500),
        bin_df,
        feature_cols=["f1", "f2", "f3", "f_noise"],
        use_internal_id=True,
        rnd_seed=42,
        return_model=True,
    )
    bin_shap = rcf.calc_shap_feature_contributions(
        None,
        lr_clf,
        bin_df,
        feature_cols=["f1", "f2", "f3", "f_noise"],
        shap_summary_plot=tmp_path / "bin_shap.png",
        subsample_n_smpls=15,
        use_linear_explainer=True,
        rnd_seed=42,
    )
    assert (tmp_path / "bin_shap.png").exists()
    assert "mean_abs_shap" in bin_shap.columns

    # KernelExplainer fallback with small subsample.
    kern_shap = rcf.calc_shap_feature_contributions(
        bin_classes,
        lr_clf,
        bin_df.iloc[:12],
        feature_cols=["f1", "f2", "f3", "f_noise"],
        subsample_n_smpls=12,
        rnd_seed=42,
    )
    assert "mean_abs_shap" in kern_shap.columns
    assert plt.get_fignums() == []


def _shap_scale_data():
    """Small 3-class / 2-class dataset for the SHAP dependence scale tests."""
    rng = np.random.default_rng(0)
    x_df = pd.DataFrame(rng.normal(size=(200, 3)), columns=["a", "b", "c"])
    y_multi = (x_df["a"] > 0.3).astype(int) + (x_df["b"] > 0).astype(int)
    y_bin = (x_df["a"] + x_df["b"] > 0).astype(int)
    return x_df, y_multi.to_numpy(), y_bin.to_numpy()


def _xgb_classifier():
    xgb = pytest.importorskip("xgboost")
    return xgb.XGBClassifier(n_estimators=20, max_depth=3, random_state=0)


@pytest.mark.parametrize(
    "est_factory, multiclass, out_idx",
    [
        (_xgb_classifier, True, 0),
        (_xgb_classifier, True, 2),
        (_xgb_classifier, False, -1),
        (lambda: RandomForestClassifier(n_estimators=15, random_state=0), True, 1),
        (lambda: RandomForestClassifier(n_estimators=15, random_state=0), False, 1),
    ],
)
def test_shap_dependence_model_fn_on_shap_scale(est_factory, multiclass, out_idx):
    """
    The partial dependence model function must be on the scale of the SHAP
    values (log-odds margin for XGBoost, probability for a random forest), so
    the SHAP values drawn over the dependence curve are comparable with it.
    """
    pytest.importorskip("shap")
    from rsgispy.tools._ml_explain import (
        _build_shap_explainer,
        _resolve_dependence_model_fn,
    )

    x_df, y_multi, y_bin = _shap_scale_data()
    est = est_factory()
    est.fit(x_df, y_multi if multiclass else y_bin)
    expl = _build_shap_explainer(est, x_df, list(x_df.columns), use_tree_explainer=True)(x_df)
    if np.asarray(expl.values).ndim == 3:
        expl = expl[:, :, out_idx]
    else:
        out_idx = -1

    model_fn, on_scale = _resolve_dependence_model_fn(
        est, x_df, expl, "classification", out_idx
    )
    assert on_scale
    explained = np.asarray(expl.values).sum(axis=1) + np.asarray(expl.base_values)
    np.testing.assert_allclose(model_fn(x_df), explained, rtol=1e-3, atol=1e-3)
    if type(est).__name__ == "XGBClassifier":
        # The margin, not a probability: values outside [0, 1] must occur.
        out_vals = model_fn(x_df)
        assert out_vals.min() < 0.0 or out_vals.max() > 1.0


def test_shap_dependence_model_fn_unmatched_scale_warns():
    """When no model output matches the SHAP values, warn and drop the overlay."""
    pytest.importorskip("shap")
    from types import SimpleNamespace

    from rsgispy.tools._ml_explain import _resolve_dependence_model_fn

    x_df, _, y_bin = _shap_scale_data()
    est = RandomForestClassifier(n_estimators=10, random_state=0).fit(x_df, y_bin)
    fake_expl = SimpleNamespace(
        values=np.zeros((len(x_df), 3)), base_values=np.full(len(x_df), 50.0)
    )
    with pytest.warns(UserWarning, match="without the SHAP values"):
        model_fn, on_scale = _resolve_dependence_model_fn(
            est, x_df, fake_expl, "classification", -1
        )
    assert not on_scale
    np.testing.assert_allclose(model_fn(x_df), est.predict_proba(x_df)[:, -1])


def test_shap_depend_plots_xgboost_multiclass(tmp_path):
    """SHAP dependence plots for a multi-class XGBoost model, without a scale warning."""
    pytest.importorskip("shap")
    fix = _make_synthetic_dataset(tmp_path, include_collinear=False)
    xgb_clf = _xgb_classifier()
    feats = ["f1", "f2", "f3"]
    xgb_clf.fit(fix["train_df"][feats], fix["train_df"]["internal_cls_id"])

    dep_dir = tmp_path / "shap_depend_xgb"
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*without the SHAP values.*")
        rcf.calc_shap_feature_contributions(
            fix["cfg"],
            xgb_clf,
            fix["train_df"],
            feature_cols=feats,
            target_class="Water",
            shap_depend_plots_dir=dep_dir,
            use_tree_explainer=True,
            rnd_seed=42,
        )
    assert {p.name for p in dep_dir.iterdir()} == {f"shap_depends_{f}.png" for f in feats}
    assert plt.get_fignums() == []


def test_salib_feature_sensitivity(tmp_path):
    """Test SALib Sobol sensitivity analysis when SALib is installed."""
    pytest.importorskip("SALib")
    fix = _make_synthetic_dataset(tmp_path, include_collinear=False)

    rf_clf, _ = rcsk.train_sklearn_classifier(
        fix["cfg"],
        RandomForestClassifier(n_estimators=25),
        fix["train_df"],
        rnd_seed=42,
        return_model=True,
    )

    sobol_png = tmp_path / "sobol.png"
    sobol_tot_csv = tmp_path / "sobol_total.csv"
    sobol_1st_csv = tmp_path / "sobol_first.csv"
    sobol_2nd_csv = tmp_path / "sobol_second.csv"
    total_si, first_si, second_si = rcf.calc_salib_feature_sensitivity(
        fix["cfg_path"],
        rf_clf,
        fix["train_pq"],
        target_class="Forest",
        n_samples=64,
        sobol_plot_file=sobol_png,
        sobol_overall_file=sobol_tot_csv,
        sobol_first_file=sobol_1st_csv,
        sobol_second_file=sobol_2nd_csv,
    )
    assert sobol_png.exists()
    assert sobol_tot_csv.exists()
    assert sobol_1st_csv.exists()
    assert sobol_2nd_csv.exists()
    assert "ST" in total_si.columns
    assert "S1" in first_si.columns
    assert "S2" in second_si.columns
    assert plt.get_fignums() == []


def test_select_features_boruta_shap(tmp_path):
    """Test BorutaShap feature selection, scores table, boxplot, and validation."""
    pytest.importorskip("shap")
    pytest.importorskip("scipy")
    fix = _make_synthetic_dataset(tmp_path, include_collinear=True)

    scores_csv = tmp_path / "boruta_shap_scores.csv"
    box_png = tmp_path / "boruta_shap_plot.png"

    # All-relevant selection retains informative & correlated signal features
    # while rejecting pure noise (f_noise).
    selected, scores_df = rcf.select_features_boruta_shap(
        fix["cfg_path"],
        fix["train_pq"],
        sk_classifier=RandomForestClassifier(n_estimators=40),
        n_trials=20,
        pvalue=0.05,
        rnd_seed=42,
        out_scores_file=scores_csv,
        out_plot_file=box_png,
        use_pbar=False,
        return_scores=True,
    )
    assert scores_csv.exists()
    assert box_png.exists()
    assert {"f1", "f2"}.issubset(selected)
    assert "f1_collinear" in selected
    assert "f_noise" not in selected
    assert {
        "feature",
        "decision",
        "selected",
        "hits",
        "n_trials_evaluated",
        "mean_importance",
        "median_importance",
        "p_value_accept",
        "rank",
    }.issubset(scores_df.columns)
    noise_dec = scores_df.loc[
        scores_df["feature"] == "f_noise", "decision"
    ].iloc[0]
    assert noise_dec == "Rejected"

    # Default sk_classifier=None + target_class + subsampling + tentative_rough_fix.
    sel_target = rcf.select_features_boruta_shap(
        fix["cfg"],
        fix["train_df"],
        feature_cols=["f1", "f2", "f3", "f_noise"],
        n_trials=12,
        target_class="Forest",
        subsample_n_smpls=30,
        tentative_rough_fix=True,
        rnd_seed=42,
        use_pbar=False,
    )
    assert "f1" in sel_target
    assert "f_noise" not in sel_target

    # Validation checks.
    with pytest.raises(rsgispy.RSGISPyException):
        rcf.select_features_boruta_shap(
            fix["cfg"], fix["train_df"], n_trials=0, use_pbar=False
        )
    with pytest.raises(rsgispy.RSGISPyException):
        rcf.select_features_boruta_shap(
            fix["cfg"], fix["train_df"], pvalue=1.0, use_pbar=False
        )
    with pytest.raises(rsgispy.RSGISPyException):
        rcf.select_features_boruta_shap(
            fix["cfg"],
            fix["train_df"],
            target_class="NoSuchClass",
            use_pbar=False,
        )
    assert plt.get_fignums() == []


def test_classfeatures_validation_errors(tmp_path):
    """Verify clear RSGISPyException validation on invalid inputs."""
    fix = _make_synthetic_dataset(tmp_path, include_collinear=False)
    empty_df = fix["train_df"].iloc[0:0].copy()

    with pytest.raises(rsgispy.RSGISPyException):
        rcf.correlation_matrix(empty_df, cls_cfg=fix["cfg"])

    with pytest.raises(rsgispy.RSGISPyException):
        rcf.calc_feature_importance(
            fix["cfg"], RandomForestClassifier(), fix["test_df"]
        )

    rf_clf, _ = rcsk.train_sklearn_classifier(
        fix["cfg"],
        RandomForestClassifier(n_estimators=10),
        fix["train_df"],
        return_model=True,
    )
    with pytest.raises(rsgispy.RSGISPyException):
        rcf.calc_feature_response_curves(
            fix["cfg"], rf_clf, fix["train_df"], target_class="MissingClass"
        )
    with pytest.raises(rsgispy.RSGISPyException):
        rcf.calc_shap_feature_contributions(
            fix["cfg"],
            rf_clf,
            fix["train_df"],
            use_tree_explainer=True,
            use_linear_explainer=True,
        )
    with pytest.raises(rsgispy.RSGISPyException):
        rcf.calc_jackknife_feature_importance(
            fix["cfg"],
            RandomForestClassifier(),
            fix["train_df"],
            fix["test_df"],
            feature_cols=["f1"],
            use_pbar=False,
        )
