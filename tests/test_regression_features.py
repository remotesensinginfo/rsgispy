"""
Tests for rsgispy.regression.regfeatures: scatter plots, collinearity metrics,
feature selection methods, permutation & native feature importance, jackknife
analysis, response curves (marginal & PDP, univariate & multivariate), SHAP,
and SALib Sobol sensitivity.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR

import rsgispy
import rsgispy.regression.regdata as rrd
import rsgispy.regression.regfeatures as rrf


def _make_reg_feature_dataset(tmp_path: Path, seed: int = 42):
    """Build synthetic collinear univariate & multivariate regression dataset."""
    rng = np.random.default_rng(seed)
    n_rows = 90
    f1 = rng.uniform(0.0, 10.0, size=n_rows)
    f2 = rng.uniform(5.0, 25.0, size=n_rows)
    f3 = rng.normal(50.0, 5.0, size=n_rows)
    f_noise = rng.normal(10.0, 0.2, size=n_rows)
    f1_col = 2.0 * f1 + 1.0  # exact collinear twin of f1

    canopy_cover = 2.5 * f1 + 1.8 * f2 + 0.2 * f3 + rng.normal(0.0, 0.5, size=n_rows)
    canopy_height = 0.6 * f2 - 0.3 * f1 + rng.normal(0.0, 0.3, size=n_rows)

    full_df = pd.DataFrame(
        {
            "f1": f1,
            "f2": f2,
            "f3": f3,
            "f_noise": f_noise,
            "f1_collinear": f1_col,
            "canopy_cover": canopy_cover,
            "canopy_height": canopy_height,
        }
    )
    train_df = full_df.iloc[:65].reset_index(drop=True)
    test_df = full_df.iloc[65:].reset_index(drop=True)

    feat_cols = ["f1", "f2", "f3", "f_noise", "f1_collinear"]
    band_info = [
        rrd.ImageBandInfo(
            file_name=str(tmp_path / "dummy.tif"),
            name="Feat",
            bands=list(range(1, len(feat_cols) + 1)),
            band_names=feat_cols,
        )
    ]
    uni_cfg = rrd.RegressionConfig(band_info, target_cols="canopy_cover")
    mv_cfg = rrd.RegressionConfig(
        band_info, target_cols=["canopy_cover", "canopy_height"]
    )
    return uni_cfg, mv_cfg, train_df, test_df, feat_cols


def test_scatter_corr_vif_and_unsupervised_selection(tmp_path: Path):
    """Test scatter plots, correlation matrix, VIF, CQV, and corr clustering."""
    _, mv_cfg, train_df, _, feat_cols = _make_reg_feature_dataset(tmp_path)

    scatter_dir = tmp_path / "scatter"
    rrf.target_scatter_plots(
        train_df,
        reg_cfg=mv_cfg,
        feature_cols=["f1", "f2"],
        out_dir=scatter_dir,
        use_pbar=False,
    )
    assert (scatter_dir / "f1_canopy_cover_scatter.png").exists()
    assert (scatter_dir / "f2_canopy_height_scatter.png").exists()

    corr_csv = tmp_path / "corr.csv"
    corr_png = tmp_path / "corr.png"
    corr_df = rrf.correlation_matrix(
        train_df,
        reg_cfg=mv_cfg,
        include_targets=True,
        out_corr_file=corr_csv,
        out_plt_file=corr_png,
    )
    assert corr_csv.exists() and corr_png.exists()
    assert "canopy_cover" in corr_df.columns
    assert corr_df.loc["f1", "f1_collinear"] == pytest.approx(1.0)

    vif_csv = tmp_path / "vif.csv"
    vif_s = rrf.calc_vif_multicollinearity(
        train_df, reg_cfg=mv_cfg, out_vif_file=vif_csv
    )
    assert vif_csv.exists()
    assert len(vif_s) == len(feat_cols)

    cqv_sel = rrf.select_features_cqv(
        train_df, reg_cfg=mv_cfg, low_threshold=0.0, high_threshold=1.0
    )
    assert set(cqv_sel).issubset(set(feat_cols))

    vif_sel, final_vif = rrf.select_features_vif(
        train_df, reg_cfg=mv_cfg, vif_threshold=5.0, min_features=2, return_vif=True
    )
    assert len(vif_sel) >= 2 and len(vif_sel) < len(feat_cols)
    assert len(final_vif) == len(vif_sel)

    corr_sel = rrf.select_features_corr(
        train_df,
        reg_cfg=mv_cfg,
        n_min_clusters=2,
        n_max_clusters=4,
        use_pbar=False,
    )
    assert 2 <= len(corr_sel) < len(feat_cols)


def test_supervised_selection_and_importances(tmp_path: Path):
    """Test univariate, RFE/RFECV, from_model/LassoLars, importance, and jackknife."""
    uni_cfg, mv_cfg, train_df, test_df, _ = _make_reg_feature_dataset(tmp_path)

    # Univariate selection (f_regression, mutual_info, multivariate mean)
    uni_sel, uni_df = rrf.select_features_univariate(
        uni_cfg,
        train_df,
        score_func="f_regression",
        k=3,
        out_scores_file=tmp_path / "uni_scores.csv",
        return_scores=True,
    )
    assert len(uni_sel) == 3
    assert "f2" in uni_sel
    assert (tmp_path / "uni_scores.csv").exists()
    assert list(uni_df["rank"]) == [1, 2, 3, 4, 5]

    mi_mv_sel = rrf.select_features_univariate(
        mv_cfg,
        train_df,
        score_func="mutual_info_regression",
        k=2,
        rnd_seed=42,
    )
    assert len(mi_mv_sel) == 2

    # RFE + RFECV
    rfe_sel = rrf.select_features_rfe(
        uni_cfg,
        RandomForestRegressor(n_estimators=20, random_state=42),
        train_df,
        n_features_to_select=3,
        rnd_seed=42,
    )
    assert len(rfe_sel) == 3

    rfecv_plot = tmp_path / "rfecv.png"
    rfecv_sel, rfecv_df = rrf.select_features_rfe(
        mv_cfg,
        RandomForestRegressor(n_estimators=20, random_state=42),
        train_df,
        use_cv=True,
        cv=3,
        scoring="r2",
        out_cv_plot=rfecv_plot,
        return_scores=True,
        rnd_seed=42,
    )
    assert rfecv_plot.exists()
    assert len(rfecv_sel) >= 1 and len(rfecv_df) == 5

    # From model & LassoLars
    fm_sel = rrf.select_features_from_model(
        uni_cfg,
        RandomForestRegressor(n_estimators=20, random_state=42),
        train_df,
        rnd_seed=42,
    )
    assert len(fm_sel) >= 1

    ll_sel = rrf.select_features_from_model(
        mv_cfg, None, train_df, use_lassolars=True
    )
    assert len(ll_sel) >= 1

    # Feature importance (univariate & multivariate with target_var)
    fitted_mv = RandomForestRegressor(n_estimators=30, random_state=42).fit(
        train_df[["f1", "f2", "f3", "f_noise", "f1_collinear"]],
        train_df[["canopy_cover", "canopy_height"]],
    )
    imp_png = tmp_path / "imp.png"
    imp_df = rrf.calc_feature_importance(
        mv_cfg,
        fitted_mv,
        test_df,
        target_var="canopy_cover",
        method="both",
        n_repeats=3,
        rnd_seed=42,
        out_importance_file=tmp_path / "imp.csv",
        out_plot_file=imp_png,
    )
    assert imp_png.exists()
    assert "model_importance" in imp_df.columns
    assert "importance_mean" in imp_df.columns

    # Jackknife
    r2_plot = tmp_path / "jack_r2.png"
    rmse_plot = tmp_path / "jack_rmse.png"
    jack_df = rrf.calc_jackknife_feature_importance(
        uni_cfg,
        RandomForestRegressor(n_estimators=20, random_state=42),
        train_df,
        test_df,
        out_var_importance_file=tmp_path / "jack.csv",
        out_importance_r2_plot=r2_plot,
        out_importance_rmse_plot=rmse_plot,
        use_pbar=False,
    )
    assert r2_plot.exists() and rmse_plot.exists()
    assert "test_r2_leave_var_out" in jack_df.columns
    assert "test_rmse_only_var" in jack_df.columns


def test_response_curves_univariate_and_multivariate(tmp_path: Path):
    """Test marginal & PDP response curves with single/multi-panel & grid plots."""
    uni_cfg, mv_cfg, train_df, _, feat_cols = _make_reg_feature_dataset(tmp_path)
    x_tr = train_df[feat_cols]

    # Univariate marginal with observed sample scatter overlay & summary matrix
    rf_uni = RandomForestRegressor(n_estimators=25, random_state=42).fit(
        x_tr, train_df["canopy_cover"]
    )
    uni_dir = tmp_path / "resp_uni"
    uni_grid = tmp_path / "resp_uni_grid.png"
    uni_csv = tmp_path / "resp_uni.csv"
    resp_uni = rrf.calc_feature_response_curves(
        reg_cfg=uni_cfg,
        sk_regressor=rf_uni,
        train_data=train_df,
        feature_cols=["f1", "f2"],
        n_samples=25,
        summary_stat="median",
        method="marginal",
        plot_samples=True,
        out_file=uni_csv,
        response_plots_dir=uni_dir,
        out_grid_plot=uni_grid,
        use_pbar=False,
    )
    assert uni_csv.exists() and uni_grid.exists()
    assert (uni_dir / "f1_response_curve.png").exists()
    assert "f1_canopy_cover_pred" in resp_uni.columns
    assert len(resp_uni) == 25
    # Response should increase strongly with f2
    assert resp_uni["f2_canopy_cover_pred"].iloc[-1] > resp_uni["f2_canopy_cover_pred"].iloc[0]

    # Multivariate PDP with multi_panel=True & out_grid_plot
    rf_mv = RandomForestRegressor(n_estimators=25, random_state=42).fit(
        x_tr, train_df[["canopy_cover", "canopy_height"]]
    )
    mv_dir = tmp_path / "resp_mv"
    mv_grid = tmp_path / "resp_mv_grid.png"
    resp_mv = rrf.calc_feature_response_curves(
        reg_cfg=mv_cfg,
        sk_regressor=rf_mv,
        train_data=train_df,
        feature_cols=["f1", "f2"],
        n_samples=20,
        method="pdp",
        pdp_background_samples=30,
        multi_panel=True,
        plot_samples=True,
        response_plots_dir=mv_dir,
        out_grid_plot=mv_grid,
        rnd_seed=42,
        use_pbar=False,
    )
    assert mv_grid.exists()
    assert (mv_dir / "f1_response_curve.png").exists()
    assert "f1_canopy_cover_pred" in resp_mv.columns
    assert "f1_canopy_height_pred" in resp_mv.columns

    # Filter to single target_vars subset
    resp_sub = rrf.calc_feature_response_curves(
        reg_cfg=mv_cfg,
        sk_regressor=rf_mv,
        train_data=train_df,
        feature_cols=["f2"],
        target_vars="canopy_height",
        n_samples=15,
        use_pbar=False,
    )
    assert "f2_canopy_height_pred" in resp_sub.columns
    assert "f2_canopy_cover_pred" not in resp_sub.columns

    with pytest.raises(rsgispy.RSGISPyException):
        rrf.calc_feature_response_curves(
            mv_cfg, rf_mv, train_df, target_vars="unknown_target", use_pbar=False
        )


def test_shap_boruta_shap_and_salib_optional(tmp_path: Path):
    """Test SHAP contributions, BorutaShap selection, and SALib Sobol sensitivity."""
    uni_cfg, mv_cfg, train_df, _, feat_cols = _make_reg_feature_dataset(tmp_path)
    x_tr = train_df[feat_cols]

    if importlib.util.find_spec("shap") is not None:
        # Univariate SHAP summary + scatter + depend plots
        ridge_uni = Ridge().fit(x_tr, train_df["canopy_cover"])
        shap_sum = tmp_path / "shap_sum.png"
        shap_df = rrf.calc_shap_feature_contributions(
            uni_cfg,
            ridge_uni,
            train_df,
            shap_summary_plot=shap_sum,
            out_shap_file=tmp_path / "shap.csv",
            subsample_n_smpls=30,
            use_linear_explainer=True,
            rnd_seed=42,
        )
        assert shap_sum.exists() and (tmp_path / "shap.csv").exists()
        assert "mean_abs_shap" in shap_df.columns

        # Multivariate MultiOutputRegressor SHAP with TreeExplainer
        mo_rf = MultiOutputRegressor(SVR(kernel="linear")).fit(
            x_tr, train_df[["canopy_cover", "canopy_height"]]
        )
        assert hasattr(mo_rf, "predict")
        rf_mv = RandomForestRegressor(n_estimators=15, random_state=42).fit(
            x_tr, train_df[["canopy_cover", "canopy_height"]]
        )
        mv_shap_df = rrf.calc_shap_feature_contributions(
            mv_cfg,
            rf_mv,
            train_df,
            target_var="canopy_cover",
            subsample_n_smpls=25,
            use_tree_explainer=True,
            rnd_seed=42,
        )
        assert "mean_abs_shap_canopy_cover" in mv_shap_df.columns

        if importlib.util.find_spec("scipy") is not None:
            bs_plot = tmp_path / "boruta_shap.png"
            bs_sel, bs_df = rrf.select_features_boruta_shap(
                uni_cfg,
                train_df,
                sk_regressor=RandomForestRegressor(n_estimators=20, random_state=42),
                n_trials=6,
                tentative_rough_fix=True,
                out_scores_file=tmp_path / "boruta.csv",
                out_plot_file=bs_plot,
                rnd_seed=42,
                use_pbar=False,
                return_scores=True,
            )
            assert bs_plot.exists() and (tmp_path / "boruta.csv").exists()
            assert isinstance(bs_sel, list) and len(bs_df) == len(feat_cols)

    if importlib.util.find_spec("SALib") is not None:
        rf_mv = RandomForestRegressor(n_estimators=15, random_state=42).fit(
            x_tr, train_df[["canopy_cover", "canopy_height"]]
        )
        total_si, first_si, second_si = rrf.calc_salib_feature_sensitivity(
            mv_cfg,
            rf_mv,
            train_df,
            target_var="canopy_cover",
            n_samples=16,
            sobol_plot_file=tmp_path / "sobol.png",
            sobol_overall_file=tmp_path / "sobol_tot.csv",
        )
        assert (tmp_path / "sobol.png").exists()
        assert (tmp_path / "sobol_tot.csv").exists()
        assert len(total_si) == len(feat_cols)
        assert len(first_si) == len(feat_cols)
        assert not second_si.empty
