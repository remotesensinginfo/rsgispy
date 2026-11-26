"""
Worked example: pixel-based regression (canopy cover from field plots) with
rsgispy.

This script is included step by step in the documentation guide
``docs/guides/pixel_regression_workflow.rst`` and is run end-to-end by the
test suite (``tests/test_pixel_regression_workflow_example.py``) on small
synthetic data, so the documented code is always tested.

Run it on your own data with, e.g.::

    python pixel_regression_workflow.py --s2 s2_sref_10m.tif --dem dem_30m.tif \\
        --plots canopy_plots.gpkg --valid-plots canopy_plots_valid.gpkg \\
        --out-dir outputs
"""

import argparse
import importlib.util
import os
from typing import Any, Dict, Sequence


def run_workflow(
    s2_img: str,
    dem_img: str,
    plots_vec: str,
    valid_plots_vec: str,
    out_dir: str,
    target_col: str = "canopy_cover",
    s2_bands: Sequence[int] = (1, 2, 3, 7, 9, 10),
    s2_band_names: Sequence[str] = ("Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"),
    cv_block_size: float = 2000.0,
    n_trees: int = 500,
    use_gbdt: bool = True,
    n_kfold_repeats: int = 10,
    use_pbar: bool = True,
) -> Dict[str, Any]:
    """
    Run the worked example. ``plots_vec`` holds the training plots and
    ``valid_plots_vec`` independent validation plots, both with the measured
    value in ``target_col`` (here canopy cover, %). ``use_gbdt=False`` skips
    the XGBoost model (optional dependency).
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-statements,import-outside-toplevel
    os.makedirs(out_dir, exist_ok=True)
    out: Dict[str, Any] = {}

    # [step-1-features]
    import rsgispy
    from rsgispy.elevation.terrain import slope
    from rsgispy.imagecalc.calcindices import calc_indices
    from rsgispy.imageutils.vrt import create_stack_images_vrt, create_vrt_band_subset
    from rsgispy.imageutils.warp import resample_img_to_match

    idx_img = os.path.join(out_dir, "s2_indices.tif")
    calc_indices(
        s2_img,
        idx_img,
        band_names={b: n.lower() for b, n in zip(s2_bands, s2_band_names)},
        indices=["NDVI", "EVI", "NDMI"],
        in_gain=10000.0,
        no_data_val=0,
        out_no_data_val=-9999.0,
        use_pbar=use_pbar,
    )
    slope_dem_img = os.path.join(out_dir, "slope_dem_grid.tif")
    slope(dem_img, slope_dem_img, no_data_val=-9999.0, use_pbar=use_pbar)
    slope_img = os.path.join(out_dir, "slope.tif")
    resample_img_to_match(s2_img, slope_dem_img, slope_img,
                          interp_method=rsgispy.INTERP_BILINEAR)
    # One virtual stack (no copy) of the uint16 bands and float32 layers.
    s2_vrt = os.path.join(out_dir, "s2_bands.vrt")
    create_vrt_band_subset(s2_img, s2_vrt, list(s2_bands))
    feat_names = list(s2_band_names) + ["NDVI", "EVI", "NDMI", "slope"]
    feat_vrt = os.path.join(out_dir, "features.vrt")
    create_stack_images_vrt([s2_vrt, idx_img, slope_img], feat_vrt,
                            band_names=feat_names)
    # [end-step-1-features]

    # [step-2-config]
    from rsgispy.classification.classdata import ImageBandInfo
    from rsgispy.regression.regdata import RegressionConfig, write_reg_config

    img_band_info = [
        ImageBandInfo(feat_vrt, name="features",
                      bands=list(range(1, len(feat_names) + 1)),
                      band_names=feat_names),
    ]
    cfg_file = os.path.join(out_dir, "reg_config.json")
    reg_cfg = write_reg_config(
        cfg_file, RegressionConfig(img_band_info, target_col, vec_file=plots_vec)
    )
    print(reg_cfg.get_band_names(), reg_cfg.target_cols)
    # [end-step-2-config]
    out["cfg_file"] = cfg_file

    # [step-3-samples]
    from rsgispy.regression.regdata import extract_and_split_reg_samples

    train_file = os.path.join(out_dir, "train.parquet")
    test_file = os.path.join(out_dir, "test.parquet")
    calib_file = os.path.join(out_dir, "calib.parquet")
    train_df, test_df, calib_df = extract_and_split_reg_samples(
        reg_cfg,
        test_prop=0.2,
        valid_prop=0.2,  # used below to calibrate the prediction intervals
        out_train_file=train_file,
        out_test_file=test_file,
        out_valid_file=calib_file,
        target_min=0.0,  # drop impossible values (cover is 0-100 %)
        target_max=100.0,
        stratify_col=target_col,  # every split covers the full range
        n_bins=5,
        include_coords=True,
        group_col="train_grp",  # all pixels of a plot go to one table
        rnd_seed=42,
        use_pbar=use_pbar,
    )
    print(train_df[target_col].describe())
    print(len(train_df), "train,", len(test_df), "test,", len(calib_df), "calib")
    # [end-step-3-samples]
    out["n_train"] = len(train_df)

    # [step-4-explore]
    import rsgispy.regression.regfeatures as rfeat

    explore_dir = os.path.join(out_dir, "explore")
    rfeat.target_scatter_plots(train_file, reg_cfg,
                               out_dir=os.path.join(explore_dir, "scatter"),
                               use_pbar=use_pbar)
    corr_df = rfeat.correlation_matrix(
        train_file, reg_cfg, include_targets=True,
        out_corr_file=os.path.join(explore_dir, "correlation.csv"),
        out_plt_file=os.path.join(explore_dir, "correlation.png"),
    )
    print(corr_df[target_col].drop(target_col).sort_values().round(2))
    print(rfeat.select_features_vif(train_file, reg_cfg, vif_threshold=10.0))
    # [end-step-4-explore]

    # [step-5-train]
    from sklearn.ensemble import RandomForestRegressor
    from rsgispy.ml.cv import SpatialBlockKFold
    import rsgispy.regression.regsklearn as rsk

    rf_model = rsk.perform_sklearn_regressor_param_search(
        cfg_file,
        RandomForestRegressor(n_estimators=n_trees, n_jobs=-1, random_state=42),
        train_file,
        param_space={"max_features": [0.33, 0.66, 1.0],
                     "min_samples_leaf": [1, 3, 10]},
        search_method="grid",
        cv=SpatialBlockKFold(n_splits=4, block_size=cv_block_size, rnd_seed=42),
        coords=("x", "y"),  # spatially blocked folds from the x/y columns
        scoring="neg_root_mean_squared_error",
        use_pbar=use_pbar,
    )
    rf_file = os.path.join(out_dir, "rf_model.joblib")
    rf_metrics = rsk.train_sklearn_regressor(
        cfg_file, rf_model, train_file, test_data=test_file,
        out_model_file=rf_file,
        out_metrics_file=os.path.join(out_dir, "rf_metrics.json"),
    )
    print({k: round(v, 3) for k, v in rf_metrics.items()})
    if use_gbdt:
        import rsgispy.regression.regxgboost as rxgb

        _, xgb_metrics = rxgb.train_xgboost_regressor(
            cfg_file, train_file, test_data=test_file, valid_data=calib_file,
            model_params={"n_estimators": 500, "learning_rate": 0.05,
                          "max_depth": 4},
            early_stopping_rounds=30,
            out_model_file=os.path.join(out_dir, "xgb_model.json"),
        )
        print({k: round(v, 3) for k, v in xgb_metrics.items()})
    # [end-step-5-train]
    out["rf_metrics"] = rf_metrics

    # [step-6-spatial-cv]
    import pandas as pd
    from rsgispy.ml.cv import nested_spatial_cv

    all_df = pd.concat([train_df, test_df, calib_df], ignore_index=True)
    feats = reg_cfg.get_band_names()
    ncv = nested_spatial_cv(
        lambda params: RandomForestRegressor(
            n_estimators=n_trees, n_jobs=-1, random_state=42, **params
        ),
        all_df[feats],
        all_df[target_col],
        task="regression",
        coords=all_df[["x", "y"]].to_numpy(),
        outer_cv=SpatialBlockKFold(n_splits=4, block_size=cv_block_size,
                                   rnd_seed=1),
        scoring="neg_rmse",
        use_pbar=use_pbar,
    )
    print(ncv.overall_metrics)  # r2, rmse and mae of the out-of-fold predictions
    # [end-step-6-spatial-cv]
    out["nested_cv"] = ncv

    # [step-6-repeated-kfold]
    kf_metrics, _ = rsk.perform_kfold_fit(
        RandomForestRegressor(n_estimators=n_trees, n_jobs=-1, random_state=42),
        all_df[feats],
        all_df[target_col],
        n_splits=5,
        n_repeats=n_kfold_repeats,  # each repeat a different random split
        groups=all_df["train_grp"],  # the pixels of a plot stay together
        use_pbar=use_pbar,
    )
    kf_summary = rsk.summarise_kfold_metrics(kf_metrics)
    print(kf_summary.loc[target_col].loc[["r2", "rmse", "nrmse"]].round(3))
    # [end-step-6-repeated-kfold]
    out["kfold_summary"] = kf_summary

    # [step-7-interpret]
    rf_fitted = rsk.load_sklearn_regressor(rf_file)
    interp_dir = os.path.join(out_dir, "interpret")
    imp_df = rfeat.calc_feature_importance(
        cfg_file, rf_fitted, test_file, method="both", rnd_seed=42,
        out_importance_file=os.path.join(interp_dir, "importance.csv"),
        out_plot_file=os.path.join(interp_dir, "importance.png"),
    )
    print(imp_df)
    rfeat.calc_feature_response_curves(
        cfg_file, rf_fitted, train_file,
        method="pdp",  # partial dependence over a sample of training rows
        out_file=os.path.join(interp_dir, "response.csv"),
        out_grid_plot=os.path.join(interp_dir, "response_grid.png"),
        plot_samples=True,
        rnd_seed=42,
        use_pbar=use_pbar,
    )
    # [end-step-7-interpret]
    out["importance"] = imp_df

    # [step-8-apply]
    vld_msk_img = os.path.join(out_dir, "valid_mask.tif")
    reg_cfg.to_feature_stack().create_valid_mask(vld_msk_img, use_pbar=use_pbar)
    pred_img = os.path.join(out_dir, "canopy_cover.tif")
    rsk.apply_sklearn_regressor(cfg_file, rf_file, vld_msk_img, 1, pred_img,
                                use_pbar=use_pbar)
    # [end-step-8-apply]
    out["pred_img"] = pred_img

    # [step-9-intervals]
    import numpy as np
    from rsgispy.ml.predict import predict_quantiles_raster

    # (a) Spread of the predictions of the individual trees.
    rf_q_img = os.path.join(out_dir, "canopy_cover_rf_quantiles.tif")
    predict_quantiles_raster(rf_fitted, reg_cfg.to_feature_stack(), rf_q_img,
                             quantiles=(0.05, 0.5, 0.95),
                             in_msk_img=vld_msk_img, use_pbar=use_pbar)
    # (b) Split-conformal 90 % interval calibrated on the held-out samples.
    calib_resid = np.abs(
        calib_df[target_col].to_numpy() - rf_fitted.predict(calib_df[feats])
    )
    conf_img = os.path.join(out_dir, "canopy_cover_conformal90.tif")
    predict_quantiles_raster(rf_fitted, reg_cfg.to_feature_stack(), conf_img,
                             conformal_residuals=calib_resid, alpha=0.1,
                             in_msk_img=vld_msk_img, use_pbar=use_pbar)
    # [end-step-9-intervals]
    out["conf_img"] = conf_img

    # [step-10-validate]
    import rsgispy.regression.regmetrics as rmet

    val_pts = rmet.pop_reg_img_pred_pts(pred_img, valid_plots_vec,
                                        pred_col="pred", use_pbar=use_pbar)
    val_pts = rmet.pop_reg_img_pred_pts(conf_img, val_pts,
                                        pred_cols=["lower", "mid", "upper"],
                                        use_pbar=use_pbar)
    # Plots outside the valid mask are read as NaN (no data): remove them.
    val_pts = val_pts.dropna(subset=["pred", "lower", "upper"])
    val_metrics = rmet.calc_reg_metrics(
        val_pts, ref_col=target_col, pred_col="pred",
        out_json_file=os.path.join(out_dir, "valid_metrics.json"),
    )
    print(val_metrics)
    rmet.plot_pred_vs_obs(val_pts, ref_col=target_col, pred_col="pred",
                          out_plot_file=os.path.join(out_dir, "pred_vs_obs.png"))
    rmet.plot_residuals(val_pts, ref_col=target_col, pred_col="pred",
                        out_plot_file=os.path.join(out_dir, "residuals.png"))
    in_interval = val_pts[target_col].between(val_pts["lower"], val_pts["upper"])
    print(f"90 % interval coverage: {in_interval.mean():.2f}")
    val_pts["residual"] = val_pts["pred"] - val_pts[target_col]
    val_pts.to_file(os.path.join(out_dir, "valid_plots_residuals.gpkg"))
    # [end-step-10-validate]
    out["val_metrics"] = val_metrics
    out["coverage"] = float(in_interval.mean())

    # [step-11-map]
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from rsgispy.imagecalc.maths import BandDefn, band_maths
    from rsgispy.mapping.raster import create_raster_cmap_img_map

    # Width of the tree-spread interval (q95 - q05), keeping no data as 0.
    width_img = os.path.join(out_dir, "canopy_cover_rf_q90_width.tif")
    band_maths(width_img, "q05 == -9999 ? 0 : q95 - q05",
               [BandDefn("q05", rf_q_img, 1), BandDefn("q95", rf_q_img, 3)],
               use_pbar=use_pbar)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    create_raster_cmap_img_map(axs[0], pred_img, 1, title_str="Canopy cover (%)",
                               cmap_name="YlGn", norm_vmin=0, norm_vmax=100)
    create_raster_cmap_img_map(axs[1], width_img, 1,
                               title_str="q05-q95 width (%)", cmap_name="magma")
    map_file = os.path.join(out_dir, "canopy_cover_map.png")
    fig.savefig(map_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    # [end-step-11-map]
    out["map_file"] = map_file
    return out


def main() -> None:
    """Command line interface for the worked example."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    parser.add_argument("--s2", required=True, help="Sentinel-2 SREF stack")
    parser.add_argument("--dem", required=True, help="DEM (projected, metres)")
    parser.add_argument("--plots", required=True, help="Training plots")
    parser.add_argument("--valid-plots", required=True, help="Validation plots")
    parser.add_argument("--target", default="canopy_cover",
                        help="Target column in the plot files")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    args = parser.parse_args()
    run_workflow(args.s2, args.dem, args.plots, args.valid_plots, args.out_dir,
                 target_col=args.target,
                 use_gbdt=importlib.util.find_spec("xgboost") is not None)


if __name__ == "__main__":
    main()
