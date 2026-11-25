"""
Worked example: species distribution modelling (SDM) with rsgispy.

This script is included step by step in the documentation guide
``docs/guides/species_distribution_modelling.rst`` and is run end-to-end by
the test suite (``tests/test_species_distribution_modelling_example.py``) on
small synthetic data, so the documented code is always tested.

Run it on your own data with, e.g.::

    python species_distribution_modelling.py \\
        --env elevation=dem_100m.tif:1 temp=climate_100m.tif:1 \\
        precip=climate_100m.tif:2 ndvi=ndvi_100m.tif:1 \\
        --roi study_area_mask.tif --presences species_records.gpkg \\
        --drop-vars elevation --out-dir sdm_outputs
"""

import argparse
import importlib.util
import os
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


def run_workflow(
    env_imgs: Mapping[str, Tuple[str, int]],
    roi_mask_img: str,
    presence_vec: str,
    out_dir: str,
    presence_lyr: Optional[str] = None,
    drop_vars: Sequence[str] = (),
    n_pseudo_abs: int = 10000,
    pseudo_abs_dist: float = 1000.0,
    n_trees: int = 500,
    n_kfolds: int = 10,
    cv_block_size: Optional[float] = None,
    salib_n_samples: int = 1024,
    use_pbar: bool = True,
) -> Dict[str, Any]:
    """
    Run the worked example. ``env_imgs`` maps each environmental variable
    name to ``(raster file, band)``; all rasters must be on the same projected
    grid as ``roi_mask_img`` (1 inside the study area). ``presence_vec`` holds
    the species presence points. ``drop_vars`` lists variables removed after
    the collinearity checks.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-statements,import-outside-toplevel
    os.makedirs(out_dir, exist_ok=True)
    out: Dict[str, Any] = {}

    # [step-1-env-vars]
    import rsgispy
    import rsgispy.modelling.species_distribution as sdm

    env_vars = {
        name: sdm.EnvVarInfo(
            name=name,
            file=img_file,
            band=band,
            data_type=rsgispy.VAR_TYPE_CONTINUOUS,
        )
        for name, (img_file, band) in env_imgs.items()
    }
    var_names = list(env_vars)

    # 1 where the pixel is in the study area AND every covariate is valid.
    vld_msk_img = os.path.join(out_dir, "valid_mask.tif")
    sdm.create_finite_mask(roi_mask_img, 1, env_vars, var_names, vld_msk_img,
                           use_pbar=use_pbar)
    # [end-step-1-env-vars]
    out["vld_msk_img"] = vld_msk_img

    # [step-2-pseudo-absences]
    abs_vec = os.path.join(out_dir, "pseudo_absences.gpkg")
    sdm.gen_pseudo_absences_smpls(
        vld_msk_img,
        1,
        out_vec_file=abs_vec,
        out_vec_lyr="absences",
        n_smpls=n_pseudo_abs,  # number of pseudo-absences to keep
        xtra_n_smpls=3 * n_pseudo_abs,  # extra candidates (some are rejected)
        presence_smpls_vec_file=presence_vec,
        presence_smpls_vec_lyr=presence_lyr,
        presence_smpls_dist_thres=pseudo_abs_dist,  # metres from any presence
        out_format="GPKG",
        rnd_seed=42,
        use_pbar=use_pbar,
    )
    # [end-step-2-pseudo-absences]

    # [step-3-extract]
    pres_gdf = sdm.extract_env_var_data(env_vars, presence_vec, presence_lyr,
                                        use_pbar=use_pbar)
    abs_gdf = sdm.extract_env_var_data(env_vars, abs_vec, "absences",
                                       use_pbar=use_pbar)
    # Drop points outside the rasters or on no data (filled with -9999).
    pres_gdf = pres_gdf[(pres_gdf[var_names] != -9999).all(axis=1)]
    abs_gdf = abs_gdf[(abs_gdf[var_names] != -9999).all(axis=1)]
    print(len(pres_gdf), "presences and", len(abs_gdf), "pseudo-absences")

    data_vec = os.path.join(out_dir, "sdm_samples.gpkg")
    analysis_vars = sdm.combine_presence_absence_data(
        pres_gdf, None, abs_gdf, None, env_vars,
        out_vec_file=data_vec, out_vec_lyr="samples",
        cls_col="clsid",  # 1 = presence, 0 = pseudo-absence
    )
    print(analysis_vars)
    # [end-step-3-extract]
    out["n_pres"] = len(pres_gdf)
    out["n_abs"] = len(abs_gdf)

    # [step-4-explore]
    plots_dir = os.path.join(out_dir, "plots")
    sdm.comparison_box_plots(env_vars, data_vec, "samples",
                             out_dir=os.path.join(plots_dir, "violins"))
    corr_df = sdm.correlation_matrix(
        env_vars, data_vec, "samples",
        out_corr_file=os.path.join(out_dir, "correlation_matrix.csv"),
        out_plt_file=os.path.join(plots_dir, "correlation_matrix.png"),
        fig_width=6, fig_height=5,
    )
    vif = sdm.calc_vif_multicollinearity(
        env_vars, data_vec, "samples",
        out_vif_file=os.path.join(out_dir, "vif.csv"),
    )
    print(corr_df.round(2))
    print(vif.round(1))
    # Remove strongly collinear variables (e.g. |r| > 0.7 or VIF > 10).
    analysis_vars = [var for var in analysis_vars if var not in drop_vars]
    # [end-step-4-explore]
    out["vif"] = vif

    # [step-5-split]
    train_vec = os.path.join(out_dir, "sdm_train.gpkg")
    test_vec = os.path.join(out_dir, "sdm_test.gpkg")
    sdm.create_train_test_sets(
        data_vec, "samples",
        train_vec, "train",
        test_vec, "test",
        split_prop=0.7,  # 70 % of each class for training
        rnd_seed=42,
    )
    # [end-step-5-split]

    # [step-6-model]
    from sklearn.ensemble import RandomForestClassifier

    rf = RandomForestClassifier(
        n_estimators=n_trees,
        min_samples_leaf=3,  # smoother response than fully grown trees
        class_weight="balanced",  # presences are usually the minority
        n_jobs=-1,
        random_state=42,
    )
    rf_mdl, train_acc, test_acc, test_auc = sdm.fit_sklearn_mdl(
        rf, train_vec, "train", test_vec, "test", analysis_vars,
        roc_curve_plot=os.path.join(plots_dir, "roc_curve.png"),
        return_model=True,
    )
    print(f"Single model: train acc {train_acc:.3f}, test acc {test_acc:.3f}, "
          f"test AUC {test_auc:.3f}")

    # An ensemble of models, each fitted to a random 60 % of the training set.
    mdls = sdm.fit_kfold_sklearn_mdls(
        rf, train_vec, "train", test_vec, "test", analysis_vars,
        n_kfolds=n_kfolds,
        fold_prop=0.6,
        rnd_seed=42,
        use_pbar=use_pbar,
    )
    # [end-step-6-model]
    out["test_auc"] = test_auc
    out["n_mdls"] = len(mdls)

    # [step-7-evaluate]
    import geopandas as gpd
    import numpy as np
    from rsgispy.tools.accuracy import bin_accuracy_scores_prob

    test_gdf = gpd.read_file(test_vec, layer="test")
    test_y = test_gdf["clsid"].to_numpy()
    # Median presence probability over the ensemble (as in the raster step).
    test_prob = np.median(
        [mdl.predict_proba(test_gdf[analysis_vars])[:, 1] for mdl in mdls], axis=0
    )
    names, scores = bin_accuracy_scores_prob(test_y, test_prob)
    ens_metrics = dict(zip(names, scores))
    print(ens_metrics)  # Test AUC, point-biserial r, Brier score, ...
    # [end-step-7-evaluate]
    out["ens_metrics"] = ens_metrics

    # [step-8-spatial-cv]
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from rsgispy.ml.cv import SpatialBlockKFold

    data_gdf = gpd.read_file(data_vec, layer="samples")
    data_x = data_gdf[analysis_vars]
    data_y = data_gdf["clsid"].to_numpy()
    coords = np.column_stack([data_gdf.geometry.x, data_gdf.geometry.y])

    rnd_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rnd_auc = cross_val_score(rf, data_x, data_y, cv=rnd_cv, scoring="roc_auc")

    # Blocks of cv_block_size metres (None = chosen from the sample extent).
    blk_cv = SpatialBlockKFold(n_splits=5, block_size=cv_block_size,
                               stratify=True, rnd_seed=42)
    blk_splits = list(blk_cv.split(data_x, data_y, coords=coords))
    blk_auc = cross_val_score(rf, data_x, data_y, cv=blk_splits, scoring="roc_auc")
    print(f"Random CV AUC  {rnd_auc.mean():.3f} +/- {rnd_auc.std():.3f}")
    print(f"Spatial CV AUC {blk_auc.mean():.3f} +/- {blk_auc.std():.3f}")
    # [end-step-8-spatial-cv]
    out["rnd_cv_auc"] = float(rnd_auc.mean())
    out["blk_cv_auc"] = float(blk_auc.mean())

    # [step-9-interpret]
    resp_df = sdm.sklearn_mdl_variable_response_curves(
        rf_mdl, train_vec, "train", analysis_vars,
        os.path.join(out_dir, "response_curves.csv"),
        os.path.join(plots_dir, "response_curves"),
        n_samples=100,
    )
    jk_df = sdm.sklearn_jackknife_var_importance(
        rf, train_vec, "train", test_vec, "test", analysis_vars,
        os.path.join(out_dir, "jackknife_importance.csv"),
        out_importance_auc_plot=os.path.join(plots_dir, "jackknife_auc.png"),
        use_pbar=use_pbar,
    )
    print(jk_df[["roc_test_acc_leave_var_out", "roc_test_acc_only_var"]])

    if importlib.util.find_spec("shap") is not None:  # optional dependency
        sdm.shap_sklearn_mdl_explainer(
            rf_mdl, train_vec, "train", analysis_vars,
            shap_summary_plot=os.path.join(plots_dir, "shap_summary.png"),
            subsample_n_smpls=500,
            use_tree_explainer=True,
        )
    if importlib.util.find_spec("SALib") is not None:  # optional dependency
        total_si, _, _ = sdm.salib_sklearn_mdl_sensitivity(
            rf_mdl, train_vec, "train", analysis_vars,
            n_samples=salib_n_samples,  # a power of 2
            sobol_plot_file=os.path.join(plots_dir, "sobol.png"),
        )
        print(total_si)
    # [end-step-9-interpret]
    out["resp_df"] = resp_df
    out["jk_df"] = jk_df

    # [step-10-predict]
    prob_img = os.path.join(out_dir, "sdm_probability.tif")
    sdm.pred_ensemble_sklearn_mdls_prob(
        mdls, vld_msk_img, 1, env_vars, analysis_vars, prob_img,
        use_pbar=use_pbar,
    )
    # Proportion of the ensemble predicting presence (model agreement).
    agree_img = os.path.join(out_dir, "sdm_agreement.tif")
    sdm.pred_ensemble_sklearn_mdls_cls(
        mdls, vld_msk_img, 1, env_vars, analysis_vars, agree_img,
        use_pbar=use_pbar,
    )
    # [end-step-10-predict]
    out["prob_img"] = prob_img
    out["agree_img"] = agree_img

    # [step-11-threshold]
    from sklearn.metrics import roc_curve
    from rsgispy.imagecalc.maths import BandDefn, band_maths, calc_prop_true_exp
    from rsgispy.imageutils.stats import pop_thmt_img_stats
    from rsgispy.tools.accuracy import accuracy_scores_binary

    # Threshold maximising the true skill statistic (TSS = sens + spec - 1).
    fpr, tpr, thresholds = roc_curve(test_y, test_prob)
    prob_thres = float(thresholds[np.argmax(tpr - fpr)])
    names, scores = accuracy_scores_binary(test_y, (test_prob >= prob_thres) * 1)
    bin_metrics = dict(zip(names, scores))
    print(f"Threshold {prob_thres:.3f}: TSS {bin_metrics['Hanssen-Kuiper skill']:.3f}")

    # 0 = outside the study area, 1 = unsuitable, 2 = suitable.
    pa_img = os.path.join(out_dir, "sdm_presence_absence.tif")
    band_defs = [BandDefn("prob", prob_img, 1)]
    band_maths(pa_img, f"prob < 0 ? 0 : (prob >= {prob_thres} ? 2 : 1)",
               band_defs, datatype=rsgispy.TYPE_8UINT, use_pbar=use_pbar)
    pop_thmt_img_stats(pa_img, add_clr_tab=True, ignore_zero=True,
                       use_pbar=use_pbar)
    suit_prop = calc_prop_true_exp(f"prob >= {prob_thres} ? 1 : 0", band_defs,
                                   in_vld_img=vld_msk_img, use_pbar=use_pbar)
    print(f"{suit_prop * 100:.1f} % of the study area is predicted suitable")
    # [end-step-11-threshold]
    out["prob_thres"] = prob_thres
    out["bin_metrics"] = bin_metrics
    out["pa_img"] = pa_img
    out["suit_prop"] = suit_prop
    return out


def main() -> None:
    """Command line interface for the worked example."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    parser.add_argument("--env", nargs="+", required=True,
                        help="NAME=raster_file[:band] entries")
    parser.add_argument("--roi", required=True, help="Study area mask (1 = in)")
    parser.add_argument("--presences", required=True, help="Presence points")
    parser.add_argument("--presences-lyr", default=None, help="Presence layer")
    parser.add_argument("--drop-vars", nargs="*", default=[],
                        help="Variables to exclude after the collinearity checks")
    parser.add_argument("--n-pseudo-abs", type=int, default=10000,
                        help="Number of pseudo-absences")
    parser.add_argument("--pseudo-abs-dist", type=float, default=1000.0,
                        help="Minimum distance (m) of pseudo-absences from presences")
    parser.add_argument("--cv-block-size", type=float, default=None,
                        help="Spatial CV block size (m)")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    args = parser.parse_args()
    env_imgs = {}
    for entry in args.env:
        name, spec = entry.split("=", 1)
        img_file, _, band = spec.partition(":")
        env_imgs[name] = (img_file, int(band) if band else 1)
    run_workflow(env_imgs, args.roi, args.presences, args.out_dir,
                 presence_lyr=args.presences_lyr,
                 drop_vars=args.drop_vars,
                 n_pseudo_abs=args.n_pseudo_abs,
                 pseudo_abs_dist=args.pseudo_abs_dist,
                 cv_block_size=args.cv_block_size)


if __name__ == "__main__":
    main()
