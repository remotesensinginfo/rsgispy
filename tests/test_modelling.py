"""
Unit tests for rsgispy.modelling and rsgispy.modelling.species_distribution.
"""

import pathlib

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point
from sklearn.ensemble import (
    GradientBoostingClassifier,
    IsolationForest,
    RandomForestClassifier,
)
from sklearn.model_selection import GridSearchCV

import rsgispy
import rsgispy.modelling as modelling_pkg
import rsgispy.modelling.species_distribution as sdm
from rsgispy.modelling._helpers import resolve_analysis_vars

matplotlib.use("Agg")


def _write_raster(
    path: str,
    arr: np.ndarray,
    origin_x: float = 500000.0,
    origin_y: float = 5200000.0,
    res: float = 10.0,
    crs: str = "EPSG:32630",
    nodata: float = -9999.0,
) -> None:
    """Write a 2D or 3D NumPy array to a GeoTIFF."""
    data = arr[None, :, :] if arr.ndim == 2 else arr
    count, height, width = data.shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=str(data.dtype),
        crs=crs,
        transform=from_origin(origin_x, origin_y, res, res),
        nodata=nodata,
    ) as dst:
        dst.write(data)


def _create_sdm_fixture(tmp_path):
    """Create a synthetic raster/vector species distribution test dataset."""
    h, w = 40, 40
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    # North-west has high var1 & low var2 (favourable niche); south-east opposite
    var1 = (100.0 - 1.5 * xx - 1.5 * yy).astype(np.float32)
    var2 = (10.0 + 1.2 * xx + 0.8 * yy).astype(np.float32)
    var3 = (50.0 + 0.3 * xx - 0.2 * yy).astype(np.float32)

    # Mask: outer 2px border is 0, central 36x36 is 1
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[2:h - 2, 2:w - 2] = 1

    # Inject a NaN and a raster nodata pixel inside ROI
    var1[5, 5] = np.nan
    var2[6, 6] = -9999.0

    cov_path = str(tmp_path / "covariates.tif")
    var3_path = str(tmp_path / "var3.tif")
    msk_path = str(tmp_path / "roi_mask.tif")

    _write_raster(cov_path, np.stack([var1, var2], axis=0), nodata=-9999.0)
    _write_raster(var3_path, var3, nodata=-9999.0)
    _write_raster(msk_path, mask, nodata=0.0)

    env_vars = {
        "VarOne": sdm.EnvVarInfo(
            name="VarOne",
            file=cov_path,
            band=1,
            data_type=rsgispy.VAR_TYPE_CONTINUOUS,
            min_vld_val=-500.0,
            max_vld_val=500.0,
        ),
        "VarTwo": sdm.EnvVarInfo(
            name="VarTwo",
            file=cov_path,
            band=2,
            data_type=rsgispy.VAR_TYPE_CONTINUOUS,
        ),
        "CatThree": sdm.EnvVarInfo(
            name="CatThree",
            file=var3_path,
            band=1,
            data_type=rsgispy.VAR_TYPE_CATEGORICAL,
        ),
    }

    # Presence points clustered in north-west (rows 3..15, cols 3..15)
    rng = np.random.default_rng(101)
    n_pres = 25
    pres_cols = rng.uniform(3.5, 14.5, n_pres)
    pres_rows = rng.uniform(3.5, 14.5, n_pres)
    pres_pts = [
        Point(500000.0 + c * 10.0, 5200000.0 - r * 10.0)
        for c, r in zip(pres_cols, pres_rows)
    ]
    pres_gdf = gpd.GeoDataFrame(
        {"species_id": np.arange(n_pres)}, geometry=pres_pts, crs="EPSG:32630"
    )

    # Absence points clustered in south-east (rows 22..36, cols 22..36)
    abs_cols = rng.uniform(22.5, 36.5, 35)
    abs_rows = rng.uniform(22.5, 36.5, 35)
    abs_pts = [
        Point(500000.0 + c * 10.0, 5200000.0 - r * 10.0)
        for c, r in zip(abs_cols, abs_rows)
    ]
    abs_gdf = gpd.GeoDataFrame(
        {"abs_id": np.arange(35)}, geometry=abs_pts, crs="EPSG:32630"
    )

    return {
        "env_vars": env_vars,
        "mask_img": msk_path,
        "cov_img": cov_path,
        "var3_img": var3_path,
        "pres_gdf": pres_gdf,
        "abs_gdf": abs_gdf,
    }


def test_package_layout_and_constants():
    """Verify clean modelling module structure and VAR_TYPE_* constants."""
    assert rsgispy.VAR_TYPE_UNDEFINED == 0
    assert rsgispy.VAR_TYPE_CONTINUOUS == 1
    assert rsgispy.VAR_TYPE_CATEGORICAL == 2

    mod_dir = pathlib.Path(modelling_pkg.__file__).parent
    py_files = sorted(p.name for p in mod_dir.glob("*.py"))
    assert py_files == [
        "__init__.py",
        "_helpers.py",
        "_maxent.py",
        "_sdm_bias.py",
        "_sdm_data.py",
        "_sdm_eval.py",
        "_sdm_models.py",
        "_sdm_novelty.py",
        "_sdm_predict.py",
        "_sdm_projection.py",
        "species_distribution.py",
    ]

    info = sdm.EnvVarInfo(
        name="Temp",
        file="temp.tif",
        band=2,
        data_type=rsgispy.VAR_TYPE_CONTINUOUS,
        min_vld_val=-20.0,
        max_vld_val=50.0,
    )
    info.set_norm_mean_std(12.5, 3.0)
    assert info.norm_mean == 12.5
    assert info.norm_std == 3.0
    assert "Continuous" in str(info)
    clone_info = sdm.EnvVarInfo(
        name="Temp",
        file="temp.tif",
        band=2,
        data_type=rsgispy.VAR_TYPE_CONTINUOUS,
        min_vld_val=-20.0,
        max_vld_val=50.0,
        norm_mean=12.5,
        norm_std=3.0,
    )
    assert info == clone_info


def test_create_finite_mask(tmp_path):
    """Verify create_finite_mask intersects ROI mask and per-band finite/nodata checks."""
    fx = _create_sdm_fixture(tmp_path)
    out_mask = str(tmp_path / "finite_mask.tif")

    sdm.create_finite_mask(
        in_msk_img=fx["mask_img"],
        img_msk_val=1,
        env_vars=fx["env_vars"],
        analysis_vars=["VarOne", "VarTwo", "CatThree"],
        output_img=out_mask,
        calc_stats=False,
        use_pbar=False,
    )

    with rasterio.open(out_mask) as src:
        arr = src.read(1)
        assert arr.dtype == np.uint8
        # Outside ROI border must be 0
        assert np.all(arr[0, :] == 0)
        assert np.all(arr[-1, :] == 0)
        # NaN in band 1 and -9999 in band 2 must both be masked out (AND semantics)
        assert arr[5, 5] == 0
        assert arr[6, 6] == 0
        # Healthy interior pixels must be 1
        assert arr[10, 10] == 1


def test_gen_pseudo_absences_smpls(tmp_path):
    """Verify pseudo-absence sampling stays inside mask and outside exclusion buffer."""
    fx = _create_sdm_fixture(tmp_path)
    valid_mask = str(tmp_path / "valid_mask.tif")
    sdm.create_finite_mask(
        fx["mask_img"],
        1,
        fx["env_vars"],
        ["VarOne", "VarTwo"],
        valid_mask,
        calc_stats=False,
        use_pbar=False,
    )

    out_gpkg = str(tmp_path / "pseudo_abs.gpkg")
    pts_gdf = sdm.gen_pseudo_absences_smpls(
        in_msk_img=valid_mask,
        img_msk_val=1,
        out_vec_file=None,
        n_smpls=25,
        xtra_n_smpls=300,
        presence_smpls_vec_file=fx["pres_gdf"],
        presence_smpls_dist_thres=50.0,
        rnd_seed=42,
        use_pbar=False,
    )
    assert pts_gdf is not None
    assert 0 < len(pts_gdf) <= 25
    assert list(pts_gdf.columns) == ["geometry"]

    # Check every pseudo-absence point is strictly > 50m from all presence points
    pres_union = fx["pres_gdf"].geometry.union_all()
    assert np.all(pts_gdf.geometry.distance(pres_union) > 50.0)

    # Also test writing to file
    res = sdm.gen_pseudo_absences_smpls(
        in_msk_img=valid_mask,
        img_msk_val=1,
        out_vec_file=out_gpkg,
        out_vec_lyr="pseudo_abs",
        out_format="GPKG",
        n_smpls=15,
        xtra_n_smpls=200,
        rnd_seed=42,
        use_pbar=False,
    )
    assert res is None
    loaded = gpd.read_file(out_gpkg, layer="pseudo_abs")
    assert len(loaded) == 15


def test_extract_combine_split_and_normalise(tmp_path):
    """Verify covariate extraction, combining, splitting, and normalisation."""
    fx = _create_sdm_fixture(tmp_path)
    env_vars = fx["env_vars"]

    pres_ext = sdm.extract_env_var_data(
        env_vars, fx["pres_gdf"], use_pbar=False
    )
    abs_ext = sdm.extract_env_var_data(
        env_vars, fx["abs_gdf"], use_pbar=False
    )
    assert pres_ext is not None and abs_ext is not None
    # Mixed-case variable names must be preserved
    for col in ("VarOne", "VarTwo", "CatThree"):
        assert col in pres_ext.columns
        assert col in abs_ext.columns

    # Combine with equalisation
    comb_gdf, analysis_vars = sdm.combine_presence_absence_data(
        pres_ext,
        None,
        abs_ext,
        None,
        env_vars,
        equalise_smpls=True,
        rnd_seed=42,
    )
    assert analysis_vars == ["VarOne", "VarTwo", "CatThree"]
    assert len(comb_gdf) == 50  # 25 + 25
    assert comb_gdf.index.is_unique
    assert (comb_gdf["clsid"] == 1).sum() == 25
    assert (comb_gdf["clsid"] == 0).sum() == 25

    # Stratified split
    train_gdf, test_gdf = sdm.create_train_test_sets(
        comb_gdf, None, split_prop=0.8, rnd_seed=42
    )
    assert len(train_gdf) == 40
    assert len(test_gdf) == 10
    assert (train_gdf["clsid"] == 1).sum() == 20
    assert (test_gdf["clsid"] == 1).sum() == 5

    # Add a constant zero-std continuous var check
    const_gdf = comb_gdf.copy()
    const_gdf["ConstVar"] = 7.0
    vars_with_const = dict(env_vars)
    vars_with_const["ConstVar"] = sdm.EnvVarInfo(
        name="ConstVar",
        file=fx["var3_img"],
        band=1,
        data_type=rsgispy.VAR_TYPE_CONTINUOUS,
    )
    sdm.pop_normalise_coeffs(vars_with_const, const_gdf)
    assert vars_with_const["ConstVar"].norm_mean == 7.0
    assert vars_with_const["ConstVar"].norm_std == 1.0
    assert vars_with_const["CatThree"].norm_mean == 0.0
    assert vars_with_const["CatThree"].norm_std == 1.0

    norm_df = sdm.apply_normalise_coeffs(vars_with_const, const_gdf)
    assert norm_df is not None
    assert np.isclose(norm_df["VarOne"].mean(), 0.0, atol=1e-7)
    assert np.isclose(norm_df["VarOne"].std(), 1.0, atol=1e-7)
    assert np.all(norm_df["ConstVar"] == 0.0)
    # Categorical column must remain un-normalised
    assert np.allclose(norm_df["CatThree"], const_gdf["CatThree"])


def test_diagnostics_plots_correlation_and_vif(tmp_path):
    """Verify comparison plots, correlation matrix, VIF, and figure cleanup."""
    fx = _create_sdm_fixture(tmp_path)
    env_vars = {
        "VarOne": fx["env_vars"]["VarOne"],
        "VarTwo": fx["env_vars"]["VarTwo"],
        "CatThree": fx["env_vars"]["CatThree"],
    }
    pres_ext = sdm.extract_env_var_data(env_vars, fx["pres_gdf"], use_pbar=False)
    abs_ext = sdm.extract_env_var_data(env_vars, fx["abs_gdf"], use_pbar=False)
    comb_gdf, _ = sdm.combine_presence_absence_data(
        pres_ext, None, abs_ext, None, env_vars
    )

    violin_dir = str(tmp_path / "violins")
    box_dir = str(tmp_path / "boxes")
    sdm.comparison_box_plots(env_vars, comb_gdf, out_dir=violin_dir, box_plots=False)
    sdm.comparison_box_plots(env_vars, comb_gdf, out_dir=box_dir, box_plots=True)
    for var_name in ("VarOne", "VarTwo", "CatThree"):
        assert (tmp_path / "violins" / f"{var_name}_comp_plot.png").is_file()
        assert (tmp_path / "boxes" / f"{var_name}_comp_plot.png").is_file()

    corr_csv = str(tmp_path / "corr.csv")
    corr_png = str(tmp_path / "corr.png")
    corr_df = sdm.correlation_matrix(
        env_vars, comb_gdf, out_corr_file=corr_csv, out_plt_file=corr_png
    )
    assert corr_df.shape == (3, 3)
    assert (tmp_path / "corr.csv").is_file()
    assert (tmp_path / "corr.png").is_file()

    # Default stays Pearson; rank-based methods are available via ``method``.
    num_vars = ["VarOne", "VarTwo", "CatThree"]
    np.testing.assert_allclose(
        corr_df.to_numpy(),
        comb_gdf[num_vars].corr(method="pearson").to_numpy(),
    )
    for corr_method in ("spearman", "kendall"):
        rank_df = sdm.correlation_matrix(
            env_vars, comb_gdf, out_corr_file=None, out_plt_file=None,
            method=corr_method,
        )
        np.testing.assert_allclose(
            rank_df.to_numpy(),
            comb_gdf[num_vars].corr(method=corr_method).to_numpy(),
        )
    with pytest.raises(rsgispy.RSGISPyException, match="correlation method"):
        sdm.correlation_matrix(
            env_vars, comb_gdf, out_corr_file=None, out_plt_file=None,
            method="cosine",
        )

    vif_csv = str(tmp_path / "vif.csv")
    vif_series = sdm.calc_vif_multicollinearity(
        ["VarOne", "VarTwo", "CatThree"], comb_gdf, out_vif_file=vif_csv
    )
    assert len(vif_series) == 3
    assert (tmp_path / "vif.csv").is_file()
    assert plt.get_fignums() == []


def test_sklearn_loop_isolation_and_fitting(tmp_path):
    """
    Verify estimators passed in loops are cleanly cloned so warm_start or prior
    fitted state never leaks across iterations.
    """
    fx = _create_sdm_fixture(tmp_path)
    env_vars = fx["env_vars"]
    pres_ext = sdm.extract_env_var_data(env_vars, fx["pres_gdf"], use_pbar=False)
    abs_ext = sdm.extract_env_var_data(env_vars, fx["abs_gdf"], use_pbar=False)
    comb_gdf, analysis_vars = sdm.combine_presence_absence_data(
        pres_ext, None, abs_ext, None, env_vars, equalise_smpls=True, rnd_seed=7
    )
    train_gdf, test_gdf = sdm.create_train_test_sets(
        comb_gdf, None, split_prop=0.7, rnd_seed=7
    )

    # 1. Warm-start RandomForest in a loop with varying feature subsets
    base_rf = RandomForestClassifier(n_estimators=5, warm_start=True, random_state=42)
    loop_models = []
    for sub_vars in (["VarOne", "VarTwo"], ["VarOne", "VarTwo", "CatThree"]):
        mdl, tr_acc, te_acc, auc_val = sdm.fit_sklearn_mdl(
            base_rf,
            train_gdf,
            None,
            test_gdf,
            None,
            sub_vars,
            return_model=True,
        )
        assert tr_acc > 0.8 and te_acc > 0.8 and auc_val > 0.9
        # Every cloned model must strictly have 5 trees (not 5 then 10)
        assert len(mdl.estimators_) == 5
        loop_models.append(mdl)

    # Input template estimator was never mutated when return_model=True
    assert not hasattr(base_rf, "estimators_")
    assert loop_models[0].n_features_in_ == 2
    assert loop_models[1].n_features_in_ == 3

    # 2. When return_model=False, reusing an already-fitted estimator with
    # different features or warm_start=True must still replace state cleanly.
    reused_gb = GradientBoostingClassifier(
        n_estimators=4, warm_start=True, random_state=11
    )
    roc_plot = str(tmp_path / "roc.png")
    tr_1, te_1, auc_1 = sdm.fit_sklearn_mdl(
        reused_gb,
        train_gdf,
        None,
        test_gdf,
        None,
        ["VarOne", "VarTwo"],
        roc_curve_plot=roc_plot,
        return_model=False,
    )
    assert len(reused_gb.estimators_) == 4
    assert reused_gb.n_features_in_ == 2
    assert (tmp_path / "roc.png").is_file()

    tr_2, te_2, auc_2 = sdm.fit_sklearn_mdl(
        reused_gb,
        train_gdf,
        None,
        test_gdf,
        None,
        ["VarOne", "VarTwo", "CatThree"],
        return_model=False,
    )
    assert len(reused_gb.estimators_) == 4
    assert reused_gb.n_features_in_ == 3
    assert min(tr_1, te_1, auc_1, tr_2, te_2, auc_2) > 0.8

    # 3. Hyperparameter search isolation
    search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        {"n_estimators": [5, 10]},
        cv=3,
    )
    best_est, best_params = sdm.search_mdl_params(
        search, train_gdf, None, analysis_vars
    )
    assert best_params["n_estimators"] in (5, 10)
    assert not hasattr(search, "best_estimator_")
    assert hasattr(best_est, "estimators_")

    # 4. Single-class model fitting
    pres_train = train_gdf[train_gdf["clsid"] == 1].copy()
    oc_template = IsolationForest(n_estimators=10, contamination=0.05, random_state=42)
    oc_mdl, oc_acc, oc_auc = sdm.fit_sklearn_sgl_cls_mdl(
        oc_template,
        pres_train,
        None,
        test_gdf,
        None,
        analysis_vars,
        return_model=True,
    )
    assert 0.0 <= oc_acc <= 1.0 and 0.0 <= oc_auc <= 1.0
    assert not hasattr(oc_template, "estimators_")
    assert hasattr(oc_mdl, "estimators_")

    # 5. K-fold ensembles preserve reproducibility without touching np.random global state
    np.random.seed(999)
    expected_rand = float(np.random.rand())
    np.random.seed(999)
    kfold_mdls = sdm.fit_kfold_sklearn_mdls(
        RandomForestClassifier(n_estimators=5, warm_start=True, random_state=1),
        train_gdf,
        None,
        test_gdf,
        None,
        analysis_vars,
        n_kfolds=4,
        fold_prop=0.7,
        rnd_seed=123,
        use_pbar=False,
    )
    assert float(np.random.rand()) == expected_rand
    assert len(kfold_mdls) == 4
    assert all(len(m.estimators_) == 5 for m in kfold_mdls)
    assert kfold_mdls[0].random_state != kfold_mdls[1].random_state

    oc_kfold = sdm.fit_kfold_sklearn_sgl_cls_mdls(
        IsolationForest(n_estimators=6, contamination=0.05, random_state=1),
        pres_train,
        None,
        test_gdf,
        None,
        analysis_vars,
        n_kfolds=3,
        fold_prop=0.8,
        rnd_seed=123,
        use_pbar=False,
    )
    assert len(oc_kfold) == 3
    assert plt.get_fignums() == []


def test_variable_response_and_jackknife(tmp_path):
    """Verify marginal response curves (incl. constant feature) and jackknife importance."""
    fx = _create_sdm_fixture(tmp_path)
    env_vars = fx["env_vars"]
    pres_ext = sdm.extract_env_var_data(env_vars, fx["pres_gdf"], use_pbar=False)
    abs_ext = sdm.extract_env_var_data(env_vars, fx["abs_gdf"], use_pbar=False)
    comb_gdf, analysis_vars = sdm.combine_presence_absence_data(
        pres_ext, None, abs_ext, None, env_vars, equalise_smpls=True, rnd_seed=5
    )
    comb_gdf["FlatVar"] = 3.0
    train_gdf, test_gdf = sdm.create_train_test_sets(
        comb_gdf, None, split_prop=0.75, rnd_seed=5
    )

    clf = RandomForestClassifier(n_estimators=10, random_state=5)
    fitted_clf, _, _, _ = sdm.fit_sklearn_mdl(
        clf,
        train_gdf,
        None,
        test_gdf,
        None,
        analysis_vars + ["FlatVar"],
        return_model=True,
    )

    resp_csv = str(tmp_path / "resp.csv")
    resp_dir = str(tmp_path / "resp_plots")
    resp_df = sdm.sklearn_mdl_variable_response_curves(
        fitted_clf,
        train_gdf,
        None,
        analysis_vars + ["FlatVar"],
        output_file=resp_csv,
        response_plots_dir=resp_dir,
        n_samples=25,
    )
    assert len(resp_df) == 25
    assert "FlatVar_vals" in resp_df.columns
    assert np.all(resp_df["FlatVar_vals"] == 3.0)
    assert (tmp_path / "resp.csv").is_file()
    assert (tmp_path / "resp_plots" / "VarOne_response_curve.png").is_file()

    jack_csv = str(tmp_path / "jackknife.csv")
    auc_png = str(tmp_path / "jack_auc.png")
    test_png = str(tmp_path / "jack_test.png")
    imp_df = sdm.sklearn_jackknife_var_importance(
        RandomForestClassifier(n_estimators=8, warm_start=True, random_state=5),
        train_gdf,
        None,
        test_gdf,
        None,
        analysis_vars,
        out_var_importance_file=jack_csv,
        out_importance_auc_plot=auc_png,
        out_importance_test_plot=test_png,
        use_pbar=False,
    )
    assert list(imp_df.index) == analysis_vars
    assert (tmp_path / "jackknife.csv").is_file()
    assert (tmp_path / "jack_auc.png").is_file()
    assert (tmp_path / "jack_test.png").is_file()
    assert plt.get_fignums() == []


def test_shap_plots_do_not_draw_on_open_figure(tmp_path):
    """
    SHAP plots get their own figure: an open figure (e.g. one returned by
    plot_roc_curve, which uses constrained layout) is left unchanged and
    SHAP's tight_layout call does not fail on it.
    """
    pytest.importorskip("shap")
    fx = _create_sdm_fixture(tmp_path)
    env_vars = fx["env_vars"]
    pres_ext = sdm.extract_env_var_data(env_vars, fx["pres_gdf"], use_pbar=False)
    abs_ext = sdm.extract_env_var_data(env_vars, fx["abs_gdf"], use_pbar=False)
    comb_gdf, analysis_vars = sdm.combine_presence_absence_data(
        pres_ext, None, abs_ext, None, env_vars, equalise_smpls=True, rnd_seed=3
    )
    train_gdf, test_gdf = sdm.create_train_test_sets(
        comb_gdf, None, split_prop=0.8, rnd_seed=3
    )
    clf, _, _, _ = sdm.fit_sklearn_mdl(
        RandomForestClassifier(n_estimators=10, random_state=3),
        train_gdf,
        None,
        test_gdf,
        None,
        analysis_vars,
        return_model=True,
    )
    roc_fig, roc_ax = sdm.plot_roc_curve(
        np.array([0.0, 0.5, 1.0]), np.array([0.0, 0.8, 1.0]), 0.8
    )
    assert roc_fig.get_layout_engine() is not None
    sdm.shap_sklearn_mdl_explainer(
        clf,
        train_gdf,
        None,
        analysis_vars,
        shap_summary_plot=str(tmp_path / "shap_summary.png"),
        shap_heatmap_plot=str(tmp_path / "shap_heat.png"),
        subsample_n_smpls=15,
        use_tree_explainer=True,
    )
    assert (tmp_path / "shap_summary.png").is_file()
    assert (tmp_path / "shap_heat.png").is_file()
    assert plt.get_fignums() == [roc_fig.number]
    assert roc_fig.axes == [roc_ax]


def test_optional_shap_and_salib(tmp_path):
    """Verify optional SHAP and SALib sensitivity functions when installed."""
    fx = _create_sdm_fixture(tmp_path)
    env_vars = fx["env_vars"]
    pres_ext = sdm.extract_env_var_data(env_vars, fx["pres_gdf"], use_pbar=False)
    abs_ext = sdm.extract_env_var_data(env_vars, fx["abs_gdf"], use_pbar=False)
    comb_gdf, analysis_vars = sdm.combine_presence_absence_data(
        pres_ext, None, abs_ext, None, env_vars, equalise_smpls=True, rnd_seed=3
    )
    train_gdf, test_gdf = sdm.create_train_test_sets(
        comb_gdf, None, split_prop=0.8, rnd_seed=3
    )
    clf, _, _, _ = sdm.fit_sklearn_mdl(
        RandomForestClassifier(n_estimators=10, random_state=3),
        train_gdf,
        None,
        test_gdf,
        None,
        analysis_vars,
        return_model=True,
    )

    shap_mod = pytest.importorskip("shap")
    assert shap_mod is not None
    shap_summary = str(tmp_path / "shap_summary.png")
    shap_heat = str(tmp_path / "shap_heat.png")
    shap_scatter = str(tmp_path / "shap_scatter")
    shap_depend = str(tmp_path / "shap_depend")
    sdm.shap_sklearn_mdl_explainer(
        clf,
        train_gdf,
        None,
        analysis_vars,
        shap_summary_plot=shap_summary,
        shap_heatmap_plot=shap_heat,
        shap_scatter_plots_dir=shap_scatter,
        shap_depend_plots_dir=shap_depend,
        subsample_n_smpls=15,
        use_tree_explainer=True,
    )
    assert (tmp_path / "shap_summary.png").is_file()
    assert (tmp_path / "shap_heat.png").is_file()
    assert (tmp_path / "shap_scatter" / "shap_scatter_VarOne.png").is_file()
    assert (tmp_path / "shap_depend" / "shap_depends_VarOne.png").is_file()

    salib_mod = pytest.importorskip("SALib")
    assert salib_mod is not None
    sobol_png = str(tmp_path / "sobol.png")
    sobol_total = str(tmp_path / "sobol_total.csv")
    sobol_first = str(tmp_path / "sobol_first.csv")
    sobol_second = str(tmp_path / "sobol_second.csv")
    tot_df, fst_df, sec_df = sdm.salib_sklearn_mdl_sensitivity(
        clf,
        train_gdf,
        None,
        analysis_vars,
        n_samples=16,
        sobol_plot_file=sobol_png,
        sobol_overall_file=sobol_total,
        sobol_first_file=sobol_first,
        sobol_second_file=sobol_second,
    )
    assert not tot_df.empty and not fst_df.empty and not sec_df.empty
    assert (tmp_path / "sobol.png").is_file()
    assert (tmp_path / "sobol_total.csv").is_file()
    assert plt.get_fignums() == []


def test_raster_predictions_single_and_ensemble(tmp_path):
    """Verify all 8 spatial prediction functions using streaming imageprocess blocks."""
    fx = _create_sdm_fixture(tmp_path)
    env_vars = fx["env_vars"]

    valid_mask = str(tmp_path / "valid_mask.tif")
    sdm.create_finite_mask(
        fx["mask_img"],
        1,
        env_vars,
        ["VarOne", "VarTwo", "CatThree"],
        valid_mask,
        calc_stats=False,
        use_pbar=False,
    )

    pres_ext = sdm.extract_env_var_data(env_vars, fx["pres_gdf"], use_pbar=False)
    abs_ext = sdm.extract_env_var_data(env_vars, fx["abs_gdf"], use_pbar=False)
    comb_gdf, analysis_vars = sdm.combine_presence_absence_data(
        pres_ext, None, abs_ext, None, env_vars, equalise_smpls=True, rnd_seed=9
    )
    sdm.pop_normalise_coeffs(env_vars, comb_gdf)
    norm_comb = sdm.apply_normalise_coeffs(env_vars, comb_gdf)
    assert norm_comb is not None

    train_gdf, test_gdf = sdm.create_train_test_sets(
        norm_comb, None, split_prop=0.8, rnd_seed=9
    )
    pres_train = train_gdf[train_gdf["clsid"] == 1].copy()

    # Binary classifier and ensemble
    clf, _, _, _ = sdm.fit_sklearn_mdl(
        RandomForestClassifier(n_estimators=12, random_state=9),
        train_gdf,
        None,
        test_gdf,
        None,
        analysis_vars,
        return_model=True,
    )
    ens_mdls = sdm.fit_kfold_sklearn_mdls(
        RandomForestClassifier(n_estimators=8, random_state=9),
        train_gdf,
        None,
        test_gdf,
        None,
        analysis_vars,
        n_kfolds=3,
        rnd_seed=9,
        use_pbar=False,
    )

    # Single-class classifier and ensemble
    oc_mdl, _, _ = sdm.fit_sklearn_sgl_cls_mdl(
        IsolationForest(n_estimators=10, contamination=0.05, random_state=9),
        pres_train,
        None,
        test_gdf,
        None,
        analysis_vars,
        return_model=True,
    )
    oc_ens = sdm.fit_kfold_sklearn_sgl_cls_mdls(
        IsolationForest(n_estimators=8, contamination=0.05, random_state=9),
        pres_train,
        None,
        test_gdf,
        None,
        analysis_vars,
        n_kfolds=3,
        rnd_seed=9,
        use_pbar=False,
    )

    prob_img = str(tmp_path / "prob.tif")
    cls_img = str(tmp_path / "cls.tif")
    oc_prob_img = str(tmp_path / "oc_prob.tif")
    oc_cls_img = str(tmp_path / "oc_cls.tif")
    ens_prob_img = str(tmp_path / "ens_prob.tif")
    ens_cls_img = str(tmp_path / "ens_cls.tif")
    oc_ens_prob_img = str(tmp_path / "oc_ens_prob.tif")
    oc_ens_cls_img = str(tmp_path / "oc_ens_cls.tif")

    # Force multi-block iteration via block_size=16 (exercise calc_stats=True once)
    sdm.pred_sklearn_mdl_prob(
        clf, valid_mask, 1, env_vars, analysis_vars, prob_img,
        normalise_data=True, calc_stats=True, block_size=16, use_pbar=False,
    )
    sdm.pred_sklearn_mdl_cls(
        clf, valid_mask, 1, env_vars, analysis_vars, cls_img,
        normalise_data=True, calc_stats=False, block_size=16, use_pbar=False,
    )
    sdm.pred_sklearn_sgl_cls_mdl_prob(
        oc_mdl, valid_mask, 1, env_vars, analysis_vars, oc_prob_img,
        normalise_data=True, calc_stats=False, block_size=16, use_pbar=False,
    )
    sdm.pred_sklearn_sgl_cls_mdl_cls(
        oc_mdl, valid_mask, 1, env_vars, analysis_vars, oc_cls_img,
        normalise_data=True, calc_stats=False, block_size=16, use_pbar=False,
    )
    sdm.pred_ensemble_sklearn_mdls_prob(
        ens_mdls, valid_mask, 1, env_vars, analysis_vars, ens_prob_img,
        normalise_data=True, calc_stats=False, block_size=16, use_pbar=False,
    )
    sdm.pred_ensemble_sklearn_mdls_cls(
        ens_mdls, valid_mask, 1, env_vars, analysis_vars, ens_cls_img,
        normalise_data=True, calc_stats=False, block_size=16, use_pbar=False,
    )
    sdm.pred_ensemble_sklearn_sgl_cls_mdls_prob(
        oc_ens, valid_mask, 1, env_vars, analysis_vars, oc_ens_prob_img,
        normalise_data=True, calc_stats=False, block_size=16, use_pbar=False,
    )
    sdm.pred_ensemble_sklearn_sgl_cls_mdls_cls(
        oc_ens, valid_mask, 1, env_vars, analysis_vars, oc_ens_cls_img,
        normalise_data=True, calc_stats=False, block_size=16, use_pbar=False,
    )

    with rasterio.open(valid_mask) as msrc:
        vmask = msrc.read(1) == 1

    with rasterio.open(prob_img) as src:
        prob_arr = src.read(1)
        assert prob_arr.dtype == np.float32
        assert np.all(prob_arr[~vmask] == -1.0)
        # NW presence corner must have higher probability than SE absence corner
        assert np.mean(prob_arr[8:13, 8:13]) > np.mean(prob_arr[27:32, 27:32])

    with rasterio.open(cls_img) as src:
        cls_arr = src.read(1)
        assert cls_arr.dtype == np.uint8
        assert set(np.unique(cls_arr)).issubset({0, 1})
        assert np.all(cls_arr[~vmask] == 0)

    with rasterio.open(oc_prob_img) as src:
        oc_prob = src.read(1)
        assert oc_prob.dtype == np.float32
        assert np.all(oc_prob[~vmask] == -9999.0)
        assert np.all(oc_prob[vmask] > -9999.0)

    with rasterio.open(oc_cls_img) as src:
        oc_cls = src.read(1)
        assert oc_cls.dtype == np.uint8
        assert set(np.unique(oc_cls)).issubset({0, 1})

    with rasterio.open(ens_prob_img) as src:
        eprob = src.read(1)
        assert eprob.dtype == np.float32
        assert np.all(eprob[~vmask] == -1.0)
        assert np.all((eprob[vmask] >= 0.0) & (eprob[vmask] <= 1.0))

    with rasterio.open(ens_cls_img) as src:
        ecls = src.read(1)
        assert ecls.dtype == np.float32
        # Nodata outside mask must strictly be -1.0 (not 0.0!)
        assert np.all(ecls[~vmask] == -1.0)
        assert np.all((ecls[vmask] >= 0.0) & (ecls[vmask] <= 1.0))

    with rasterio.open(oc_ens_prob_img) as src:
        oep = src.read(1)
        assert oep.dtype == np.float32
        assert np.all(oep[~vmask] == -9999.0)

    with rasterio.open(oc_ens_cls_img) as src:
        oec = src.read(1)
        assert oec.dtype == np.float32
        assert np.all(oec[~vmask] == -1.0)
        assert np.all((oec[vmask] >= 0.0) & (oec[vmask] <= 1.0))


def _write_gtiff(
    path: str,
    arr: np.ndarray,
    *,
    origin_x: float = 500000.0,
    origin_y: float = 5200000.0,
    res: float = 10.0,
    crs: str = "EPSG:32630",
    nodata: float = -9999.0,
) -> None:
    """Write a 2D or 3D NumPy array to a GeoTIFF."""
    data = arr[None, :, :] if arr.ndim == 2 else arr
    count, height, width = data.shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=str(data.dtype),
        crs=crs,
        transform=from_origin(origin_x, origin_y, res, res),
        nodata=nodata,
    ) as dst:
        dst.write(data)


def test_modelling_valid_range_crs_alignment_resolve_vars_and_distinct_nodata(tmp_path):
    """
    Verify modelling valid-range filtering, mismatched dict-key/info.name resolution,
    CRS alignment in combine_presence_absence_data, k-fold non-duplicate class
    preservation, and distinct uint8 nodata=255 in class predictions.
    """
    height, width = 24, 24
    yy_raw, xx_raw = np.mgrid[0:height, 0:width]
    yy = yy_raw.astype(np.float32)
    xx = xx_raw.astype(np.float32)
    v1 = (50.0 + xx - yy).astype(np.float32)
    v2 = (20.0 + 0.5 * xx + 0.5 * yy).astype(np.float32)
    # Inject out-of-valid-range values inside ROI (finite and != raster nodata)
    v1[4, 4] = -5.0  # below min_vld_val=0.0
    v1[5, 5] = 200.0  # above max_vld_val=100.0

    mask = np.zeros((height, width), dtype=np.uint8)
    mask[2:22, 2:22] = 1

    cov_path = str(tmp_path / "cov.tif")
    msk_path = str(tmp_path / "msk.tif")
    _write_gtiff(cov_path, np.stack([v1, v2], axis=0), nodata=-9999.0)
    _write_gtiff(msk_path, mask, nodata=0)

    # Dict keys deliberately differ from EnvVarInfo.name
    env_vars = {
        "key_one": sdm.EnvVarInfo(
            name="VarOne",
            file=cov_path,
            band=1,
            data_type=rsgispy.VAR_TYPE_CONTINUOUS,
            min_vld_val=0.0,
            max_vld_val=100.0,
        ),
        "key_two": sdm.EnvVarInfo(
            name="VarTwo",
            file=cov_path,
            band=2,
            data_type=rsgispy.VAR_TYPE_CONTINUOUS,
        ),
    }

    # 1. resolve_analysis_vars works with None, info.name, and dict keys
    assert resolve_analysis_vars(env_vars) == ["VarOne", "VarTwo"]
    assert resolve_analysis_vars(env_vars, ["VarOne", "VarTwo"]) == ["VarOne", "VarTwo"]
    assert resolve_analysis_vars(env_vars, ["key_one", "key_two"]) == ["VarOne", "VarTwo"]

    # 2. create_finite_mask enforces min_vld_val and max_vld_val
    finite_msk = str(tmp_path / "finite_mask.tif")
    sdm.create_finite_mask(
        msk_path,
        1,
        env_vars,
        ["VarOne", "VarTwo"],
        finite_msk,
        calc_stats=False,
        use_pbar=False,
    )
    with rasterio.open(finite_msk) as src:
        fmask = src.read(1)
        assert fmask[4, 4] == 0
        assert fmask[5, 5] == 0
        assert fmask[10, 10] == 1

    # 3. combine_presence_absence_data reprojects absence_gdf when CRSs differ
    pres_pts = [Point(500000.0 + c * 10.0, 5200000.0 - r * 10.0) for c, r in zip(
        np.linspace(3.5, 9.5, 10), np.linspace(3.5, 9.5, 10)
    )]
    abs_pts = [Point(500000.0 + c * 10.0, 5200000.0 - r * 10.0) for c, r in zip(
        np.linspace(14.5, 20.5, 10), np.linspace(14.5, 20.5, 10)
    )]
    pres_gdf = gpd.GeoDataFrame(
        {"VarOne": np.linspace(60.0, 70.0, 10), "VarTwo": np.linspace(22.0, 26.0, 10)},
        geometry=pres_pts,
        crs="EPSG:32630",
    )
    abs_gdf_utm = gpd.GeoDataFrame(
        {"VarOne": np.linspace(30.0, 40.0, 10), "VarTwo": np.linspace(34.0, 38.0, 10)},
        geometry=abs_pts,
        crs="EPSG:32630",
    )
    abs_gdf_wgs84 = abs_gdf_utm.to_crs("EPSG:4326")

    comb_gdf, analysis_vars = sdm.combine_presence_absence_data(
        pres_gdf, None, abs_gdf_wgs84, None, env_vars
    )
    assert analysis_vars == ["VarOne", "VarTwo"]
    assert comb_gdf.crs == pres_gdf.crs
    # Reprojected absence coordinates must match original UTM coordinates
    abs_rows = comb_gdf[comb_gdf["clsid"] == 0].reset_index(drop=True)
    assert np.allclose(abs_rows.geometry.x.to_numpy(), abs_gdf_utm.geometry.x.to_numpy(), atol=1e-3)
    assert np.allclose(abs_rows.geometry.y.to_numpy(), abs_gdf_utm.geometry.y.to_numpy(), atol=1e-3)

    # Mismatched defined vs undefined CRS must raise RSGISPyException
    abs_no_crs = abs_gdf_utm.set_crs(None, allow_override=True)
    with pytest.raises(rsgispy.RSGISPyException, match="CRS"):
        sdm.combine_presence_absence_data(pres_gdf, None, abs_no_crs, None, env_vars)

    # 4. fit_kfold_sklearn_mdls validates fold size >= n_classes and avoids duplicates
    with pytest.raises(rsgispy.RSGISPyException, match="fold size"):
        sdm.fit_kfold_sklearn_mdls(
            RandomForestClassifier(n_estimators=5, random_state=1),
            comb_gdf,
            None,
            comb_gdf,
            None,
            analysis_vars,
            n_kfolds=2,
            fold_prop=0.05,  # 20 * 0.05 = 1 < 2 classes
            sel_replacement=False,
            rnd_seed=1,
            use_pbar=False,
        )

    # Direct check on class-preservation helper with extreme imbalance & no replacement
    train_y_imb = np.array([0] * 19 + [1])
    bad_draw = np.array([0, 1, 2, 3])
    fixed_idx = sdm._ensure_all_classes_in_fold(  # pylint: disable=protected-access
        bad_draw, train_y_imb, np.array([0, 1]), np.random.default_rng(0)
    )
    assert set(np.unique(train_y_imb[fixed_idx])) == {0, 1}
    assert len(np.unique(fixed_idx)) == len(fixed_idx) == len(bad_draw)

    # 5. Distinct nodata=255 and valid-range filtering in build_env_block_matrix / pred_*_cls
    clf, _, _, _ = sdm.fit_sklearn_mdl(
        RandomForestClassifier(n_estimators=10, random_state=3),
        comb_gdf,
        None,
        comb_gdf,
        None,
        analysis_vars,
        return_model=True,
    )
    oc_mdl, _, _ = sdm.fit_sklearn_sgl_cls_mdl(
        IsolationForest(n_estimators=10, contamination=0.1, random_state=3),
        comb_gdf[comb_gdf["clsid"] == 1],
        None,
        comb_gdf,
        None,
        analysis_vars,
        return_model=True,
    )

    cls_255 = str(tmp_path / "cls_nd255.tif")
    oc_cls_255 = str(tmp_path / "oc_cls_nd255.tif")
    # Pass raw msk_path (which does NOT mask [4,4] or [5,5]) to prove build_env_block_matrix
    # enforces min_vld_val/max_vld_val during prediction too!
    sdm.pred_sklearn_mdl_cls(
        clf,
        msk_path,
        1,
        env_vars,
        analysis_vars,
        cls_255,
        calc_stats=False,
        use_pbar=False,
        out_no_data_val=255,
    )
    sdm.pred_sklearn_sgl_cls_mdl_cls(
        oc_mdl,
        msk_path,
        1,
        env_vars,
        analysis_vars,
        oc_cls_255,
        calc_stats=False,
        use_pbar=False,
        out_no_data_val=255,
    )

    with rasterio.open(cls_255) as src:
        arr = src.read(1)
        assert arr.dtype == np.uint8
        assert src.nodata == 255
        assert arr[0, 0] == 255
        # Out-of-valid-range pixels inside ROI must also be marked nodata (255)
        assert arr[4, 4] == 255
        assert arr[5, 5] == 255
        assert arr[10, 10] in (0, 1)

    with rasterio.open(oc_cls_255) as src:
        oc_arr = src.read(1)
        assert oc_arr.dtype == np.uint8
        assert src.nodata == 255
        assert oc_arr[0, 0] == 255
        assert oc_arr[4, 4] == 255
        assert oc_arr[5, 5] == 255
        assert oc_arr[10, 10] in (0, 1)
