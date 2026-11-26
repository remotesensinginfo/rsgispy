"""
Worked example: pixel-based supervised land cover classification with rsgispy.

This script is included step by step in the documentation guide
``docs/guides/pixel_classification_workflow.rst`` and is run end-to-end by
the test suite (``tests/test_pixel_classification_workflow_example.py``) on
small synthetic data, so the documented code is always tested.

Run it on your own data with, e.g.::

    python pixel_classification_workflow.py --s2 s2_sref_10m.tif \\
        --dem dem_30m.tif --train Water=water.gpkg Forest=forest.gpkg \\
        Grass=grass.gpkg Urban=urban.gpkg --acc-pts acc_pts_labelled.gpkg \\
        --out-dir outputs
"""

import argparse
import importlib.util
import os
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


def run_workflow(
    s2_img: str,
    dem_img: str,
    train_vecs: Mapping[str, Tuple[str, int, str]],
    out_dir: str,
    acc_pts_vec: Optional[str] = None,
    s2_bands: Sequence[int] = (1, 2, 3, 7, 9, 10),
    s2_band_names: Sequence[str] = ("Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"),
    max_cls_smpls: int = 20000,
    cv_block_size: float = 2000.0,
    n_trees: int = 300,
    n_trials: int = 50,
    mmu_pxls: int = 10,
    n_acc_smpls: int = 500,
    salib_n_samples: int = 256,
    use_glcm: bool = True,
    use_gbdt: bool = True,
    use_boruta: bool = True,
    use_shap: bool = True,
    use_salib: bool = True,
    use_pbar: bool = True,
) -> Dict[str, Any]:
    """
    Run the worked example. ``train_vecs`` maps each class name to
    ``(training vector file, output class ID, colour)``. ``acc_pts_vec`` holds
    labelled accuracy assessment points with the reference class name in a
    ``ref_name`` column; if it is ``None`` the accuracy assessment is skipped
    (the stratified random sample to label is still created). The ``use_*``
    options switch off steps that need optional dependencies.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-statements,import-outside-toplevel
    # pylint: disable=too-many-branches
    os.makedirs(out_dir, exist_ok=True)
    out: Dict[str, Any] = {}

    # [step-1-features]
    import rsgispy
    from rsgispy.elevation.terrain import slope
    from rsgispy.imagecalc.calcindices import calc_indices
    from rsgispy.imagefilter.texture import apply_glcm_filter
    from rsgispy.imageutils.vrt import create_stack_images_vrt, create_vrt_band_subset
    from rsgispy.imageutils.warp import resample_img_to_match

    # Spectral indices (input reflectance is scaled by 10000, no data = 0).
    idx_img = os.path.join(out_dir, "s2_indices.tif")
    calc_indices(
        s2_img,
        idx_img,
        band_names={b: n.lower() for b, n in zip(s2_bands, s2_band_names)},
        indices=["NDVI", "MNDWI", "NDBI"],
        in_gain=10000.0,
        no_data_val=0,
        out_no_data_val=-9999.0,
        use_pbar=use_pbar,
    )

    # Slope on the DEM's own grid, then resampled onto the Sentinel-2 grid.
    slope_dem_img = os.path.join(out_dir, "slope_dem_grid.tif")
    slope(dem_img, slope_dem_img, unit="degrees", no_data_val=-9999.0,
          use_pbar=use_pbar)
    slope_img = os.path.join(out_dir, "slope.tif")
    resample_img_to_match(s2_img, slope_dem_img, slope_img,
                          interp_method=rsgispy.INTERP_BILINEAR)

    # The Sentinel-2 bands to use, as a virtual image (no copy).
    s2_vrt = os.path.join(out_dir, "s2_bands.vrt")
    create_vrt_band_subset(s2_img, s2_vrt, list(s2_bands))
    feat_imgs = [s2_vrt, idx_img, slope_img]
    feat_names = list(s2_band_names) + ["NDVI", "MNDWI", "NDBI", "slope"]
    if use_glcm:
        # Texture of the NIR band in a 5 x 5 pixel window.
        nir_vrt = os.path.join(out_dir, "s2_nir.vrt")
        create_vrt_band_subset(s2_img, nir_vrt,
                               [s2_bands[s2_band_names.index("NIR")]])
        glcm_img = os.path.join(out_dir, "nir_glcm.tif")
        apply_glcm_filter(nir_vrt, glcm_img, properties=["homogeneity", "entropy"],
                          filter_size=5, no_data_val=0, use_pbar=use_pbar)
        feat_imgs.append(glcm_img)
        feat_names += ["NIR_homogeneity", "NIR_entropy"]

    # One virtual stack (no copy) of every feature: the uint16 reflectance and
    # the float32 derived layers, each band keeping its own no data value.
    feat_vrt = os.path.join(out_dir, "features.vrt")
    create_stack_images_vrt(feat_imgs, feat_vrt, band_names=feat_names)
    # [end-step-1-features]
    out["feat_vrt"] = feat_vrt

    # [step-2-config]
    from rsgispy.classification.classdata import (
        ClassificationConfig,
        ClassInfoObj,
        ImageBandInfo,
        read_cls_config,
        write_cls_config,
    )

    img_band_info = [
        ImageBandInfo(feat_vrt, name="features",
                      bands=list(range(1, len(feat_names) + 1)),
                      band_names=feat_names),
    ]
    class_info = {
        cls_name: ClassInfoObj(out_id=out_id, vec_file=vec_file, colour=colour)
        for cls_name, (vec_file, out_id, colour) in train_vecs.items()
    }
    cfg_file = os.path.join(out_dir, "cls_config.json")
    write_cls_config(cfg_file, ClassificationConfig(img_band_info, class_info))
    cls_cfg = read_cls_config(cfg_file)
    print(cls_cfg.get_band_names())
    print(cls_cfg.get_class_names_lut())
    # [end-step-2-config]
    out["cfg_file"] = cfg_file

    # [step-3-samples]
    from rsgispy.classification.classdata import extract_and_split_class_samples

    train_file = os.path.join(out_dir, "train.parquet")
    valid_file = os.path.join(out_dir, "valid.parquet")
    test_file = os.path.join(out_dir, "test.parquet")
    train_df, test_df, valid_df = extract_and_split_class_samples(
        cls_cfg,
        test_prop=0.2,
        valid_prop=0.2,
        out_train_file=train_file,
        out_test_file=test_file,
        out_valid_file=valid_file,
        max_cls_smpls=max_cls_smpls,  # cap large classes before splitting
        include_coords=True,  # keep x/y for spatial cross-validation
        group_col="train_grp",  # each polygon goes to one table only
        rnd_seed=42,
        use_pbar=use_pbar,
    )
    print(train_df["class_name"].value_counts())
    for name, part in (("train", train_df), ("valid", valid_df), ("test", test_df)):
        print(name, len(part), "pixels from",
              part.groupby("class_name")["train_grp"].nunique().to_dict(),
              "polygons")
    # [end-step-3-samples]
    out["n_train"] = len(train_df)
    out["split_polygons"] = [set(df["train_grp"]) for df in (train_df, valid_df, test_df)]

    # [step-4-explore]
    import rsgispy.classification.classfeatures as cfeat

    explore_dir = os.path.join(out_dir, "explore")
    cfeat.comparison_box_plots(train_file, cls_cfg, out_dir=explore_dir,
                               use_pbar=use_pbar)
    corr_df = cfeat.correlation_matrix(
        train_file,
        cls_cfg,
        out_corr_file=os.path.join(explore_dir, "correlation.csv"),
        out_plt_file=os.path.join(explore_dir, "correlation.png"),
    )
    vif = cfeat.calc_vif_multicollinearity(
        train_file, cls_cfg, out_vif_file=os.path.join(explore_dir, "vif.csv")
    )
    print(corr_df.round(2))
    print(vif.sort_values(ascending=False).round(1))
    # [end-step-4-explore]
    out["vif"] = vif

    # [step-5-select]
    from sklearn.ensemble import RandomForestClassifier

    def new_rf(**params: Any) -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=params.pop("n_estimators", n_trees),
            class_weight="balanced", n_jobs=-1, random_state=42, **params,
        )

    selections = {
        "corr": cfeat.select_features_corr(
            train_file, cls_cfg, n_min_clusters=3,
            n_max_clusters=len(cls_cfg.get_band_names()) - 1, use_pbar=use_pbar,
        ),
        "vif": cfeat.select_features_vif(train_file, cls_cfg, vif_threshold=10.0),
        "rfe": cfeat.select_features_rfe(
            cls_cfg, new_rf(), train_file, use_cv=True, cv=3,
            min_features_to_select=3, rnd_seed=42,
        ),
        "model": cfeat.select_features_from_model(
            cls_cfg, new_rf(), train_file, threshold="median",
            rnd_seed=42,
        ),
    }
    if use_boruta:
        selections["boruta"] = cfeat.select_features_boruta_shap(
            cls_cfg, train_file, new_rf(), n_trials=30,
            rnd_seed=42, use_pbar=use_pbar,
        )
    for method, feats in selections.items():
        print(f"{method}: {feats}")

    # Apply a selection: keep only the selected bands of each image.
    selected = selections["rfe"]
    sel_band_info = [
        ImageBandInfo(
            info.file_name,
            name=info.name,
            bands=[b for b, n in zip(info.bands, info.band_names) if n in selected],
            band_names=[n for n in info.band_names if n in selected],
        )
        for info in cls_cfg.img_band_info
        if set(info.band_names) & set(selected)
    ]
    sel_cfg_file = os.path.join(out_dir, "cls_config_selected.json")
    sel_cfg = write_cls_config(sel_cfg_file, sel_band_info, cls_cfg.class_info)
    print("Model features:", sel_cfg.get_band_names())
    # [end-step-5-select]
    out["selections"] = selections

    # [step-6-train-rf]
    import rsgispy.classification.classsklearn as csk
    from rsgispy.ml.cv import SpatialBlockKFold

    spatial_cv = SpatialBlockKFold(n_splits=4, block_size=cv_block_size,
                                   stratify=True, rnd_seed=42)
    rf_model = csk.perform_sklearn_classifier_param_search(
        sel_cfg_file,
        new_rf(),
        train_file,
        param_space={
            "max_features": ["sqrt", 0.5, 1.0],
            "min_samples_leaf": [1, 5, 20],
        },
        search_method="grid",
        cv=spatial_cv,  # spatially blocked folds ...
        coords=("x", "y"),  # ... from the pixel coordinates in train_file
        scoring="balanced_accuracy",
        out_params_json=os.path.join(out_dir, "rf_best_params.json"),
        use_pbar=use_pbar,
    )
    rf_file = os.path.join(out_dir, "rf_model.joblib")
    rf_metrics = csk.train_sklearn_classifier(
        sel_cfg_file,
        rf_model,
        train_file,
        test_data=test_file,
        valid_data=valid_file,
        out_model_file=rf_file,
        out_metrics_file=os.path.join(out_dir, "rf_metrics.json"),
    )
    print({k: round(v, 3) for k, v in rf_metrics.items() if "accuracy" in k})
    # [end-step-6-train-rf]
    out["rf_metrics"] = rf_metrics

    # [step-6b-train-gbdt]
    xgb_file = os.path.join(out_dir, "xgb_model.json")
    if use_gbdt:
        import rsgispy.classification.classxgboost as cxgb

        xgb_model = cxgb.optimise_xgboost_multiclass_classifier(
            sel_cfg_file,
            train_file,
            valid_data=valid_file,  # score each trial on the validation set
            n_trials=n_trials,
            early_stopping_rounds=20,
            rnd_seed=42,
            use_pbar=use_pbar,
        )
        _, xgb_metrics = cxgb.train_xgboost_multiclass_classifier(
            sel_cfg_file,
            train_file,
            test_data=test_file,
            xgb_classifier=xgb_model,
            out_model_file=xgb_file,
        )
        print({k: round(v, 3) for k, v in xgb_metrics.items() if "accuracy" in k})
    # [end-step-6b-train-gbdt]

    # [step-6c-nested-cv]
    import pandas as pd
    from rsgispy.ml.cv import nested_spatial_cv

    all_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)
    feats = sel_cfg.get_band_names()
    ncv = nested_spatial_cv(
        lambda params: new_rf(**params),
        all_df[feats],
        all_df["internal_cls_id"],
        coords=all_df[["x", "y"]].to_numpy(),
        outer_cv=SpatialBlockKFold(n_splits=4, block_size=cv_block_size,
                                   stratify=True, rnd_seed=1),
        param_space={"min_samples_leaf": [1, 5, 20]},
        scoring="accuracy",
        n_trials=3,
        use_pbar=use_pbar,
    )
    print(f"Spatial CV accuracy: {ncv.mean_score:.3f} +/- {ncv.std_score:.3f}")
    print(ncv.overall_metrics)
    # [end-step-6c-nested-cv]
    out["nested_cv"] = ncv

    # [step-7-interpret]
    rf_fitted = csk.load_sklearn_classifier(rf_file)
    interp_dir = os.path.join(out_dir, "interpret")
    imp_df = cfeat.calc_feature_importance(
        sel_cfg_file, rf_fitted, test_file, method="both", rnd_seed=42,
        out_importance_file=os.path.join(interp_dir, "importance.csv"),
        out_plot_file=os.path.join(interp_dir, "importance.png"),
    )
    print(imp_df)
    jk_df = cfeat.calc_jackknife_feature_importance(
        sel_cfg_file, new_rf(), train_file, test_file,
        out_var_importance_file=os.path.join(interp_dir, "jackknife.csv"),
        out_importance_acc_plot=os.path.join(interp_dir, "jackknife_acc.png"),
        use_pbar=use_pbar,
    )
    print(jk_df)
    cfeat.calc_feature_response_curves(
        sel_cfg_file, rf_fitted, train_file,
        out_file=os.path.join(interp_dir, "response_forest.csv"),
        response_plots_dir=os.path.join(interp_dir, "response_forest"),
        target_class="Forest",  # probability of the Forest class
        use_pbar=use_pbar,
    )
    if use_shap:
        cfeat.calc_shap_feature_contributions(
            sel_cfg_file, rf_fitted, test_file, target_class="Water",
            shap_summary_plot=os.path.join(interp_dir, "shap_water.png"),
            subsample_n_smpls=500, use_tree_explainer=True, rnd_seed=42,
        )
    if use_salib:
        sobol_total, _, _ = cfeat.calc_salib_feature_sensitivity(
            sel_cfg_file, rf_fitted, train_file, target_class="Water",
            n_samples=salib_n_samples,
            sobol_plot_file=os.path.join(interp_dir, "sobol_water.png"),
        )
        print(sobol_total)
    # [end-step-7-interpret]
    out["importance"] = imp_df

    # [step-8-apply]
    vld_msk_img = os.path.join(out_dir, "valid_mask.tif")
    sel_cfg.to_feature_stack().create_valid_mask(vld_msk_img, use_pbar=use_pbar)

    cls_img = os.path.join(out_dir, "rf_classes.tif")
    csk.apply_sklearn_classifier(
        sel_cfg_file,
        rf_file,
        vld_msk_img,
        1,  # classify pixels where the mask is 1
        output_img=cls_img,
        out_score_img=os.path.join(out_dir, "rf_class_probs.tif"),
        use_pbar=use_pbar,
    )
    if use_gbdt:
        cxgb.apply_xgboost_multiclass_classifier(
            sel_cfg_file, xgb_file, vld_msk_img, 1,
            output_img=os.path.join(out_dir, "xgb_classes.tif"),
            use_pbar=use_pbar,
        )
    # [end-step-8-apply]
    out["cls_img"] = cls_img

    # [step-9-postprocess]
    from rsgispy.classification.classpostprocess import apply_postprocess_pipeline

    cls_pp_img = os.path.join(out_dir, "rf_classes_mmu.tif")
    apply_postprocess_pipeline(
        cls_img,
        cls_pp_img,
        majority_sizes=[3],  # one 3 x 3 majority (modal) filter pass
        sieve_size_threshold=mmu_pxls,  # minimum mapping unit in pixels
        sieve_connectivity=8,
        use_pbar=use_pbar,
    )
    # [end-step-9-postprocess]
    out["cls_pp_img"] = cls_pp_img

    # [step-10-accuracy]
    import rsgispy.classification.classaccuracymetrics as cam
    import rsgispy.classification.classrefpts as crefpts

    cls_names_lut = sel_cfg.get_class_names_lut()
    crefpts.create_stratified_random_ref_smpls_darts(
        cls_pp_img,
        n_smpls=n_acc_smpls,
        out_vec_file=os.path.join(out_dir, "acc_pts_to_label.gpkg"),
        min_cls_smpls=50,  # so that rare classes get enough points
        class_names_lut=cls_names_lut,
        rnd_seed=42,
    )
    if acc_pts_vec is not None:  # the labelled points (``ref_name`` column)
        acc_pts = crefpts.pop_class_info_accuracy_pts(
            cls_pp_img, acc_pts_vec, None, None, None, cls_names_lut,
            img_cls_col="map_name", use_pbar=use_pbar,
        )
        acc_stats = cam.calc_acc_stats_img(
            acc_pts, None, "ref_name", "map_name", cls_pp_img, cls_names_lut,
            out_json_file=os.path.join(out_dir, "acc_stats.json"),
            out_csv_file=os.path.join(out_dir, "acc_stats.csv"),
            use_pbar=use_pbar,
        )
        acc_ci = cam.calc_acc_stats_img_bootstrap_conf(
            acc_pts, None, "ref_name", "map_name", cls_pp_img, cls_names_lut,
            out_json_file=os.path.join(out_dir, "acc_stats_conf.json"),
            sample_frac=1.0,  # resample all n points (standard bootstrap)
            bootstrap_n=200, rnd_seed=42, use_pbar=use_pbar,
        )
        # Area-weighted overall accuracy and its bootstrap interval.
        print(acc_stats["quantity_metrics"]["Proportion Correct (C)"],
              acc_ci["quantity_metrics"]["Proportion Correct (C)"])
        out["acc_stats"] = acc_stats
        out["acc_ci"] = acc_ci
    # [end-step-10-accuracy]

    # [step-11-map]
    import matplotlib

    matplotlib.use("Agg")  # no display needed
    import matplotlib.pyplot as plt
    from rsgispy.mapping.legend import add_categorical_legend
    from rsgispy.mapping.raster import create_thematic_raster_map

    fig, (ax, leg_ax) = plt.subplots(
        1, 2, figsize=(8, 5), gridspec_kw={"width_ratios": [4, 1]}
    )
    create_thematic_raster_map(ax, cls_pp_img, title_str="Land cover")
    add_categorical_legend(leg_ax, classes_info=sel_cfg, overlay=False,
                           title="Class")
    map_file = os.path.join(out_dir, "land_cover_map.png")
    fig.savefig(map_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    # [end-step-11-map]
    out["map_file"] = map_file

    # [step-12-cluster]
    from sklearn.cluster import KMeans
    from rsgispy.classification.clustersklearn import img_pixel_sample_cluster

    clu_img = os.path.join(out_dir, "kmeans_clusters.tif")
    img_pixel_sample_cluster(
        sel_cfg_file,
        clu_img,
        clusterer=KMeans(n_clusters=8, n_init=10, random_state=42),
        n_img_smpl=10,  # fit on every 10th valid pixel
        in_msk_img=vld_msk_img,
        img_msk_val=1,
        rnd_seed=42,
        use_pbar=use_pbar,
    )
    # [end-step-12-cluster]
    out["clu_img"] = clu_img
    return out


def main() -> None:
    """Command line interface for the worked example."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    parser.add_argument("--s2", required=True, help="Sentinel-2 SREF stack")
    parser.add_argument("--dem", required=True, help="DEM (projected, metres)")
    parser.add_argument("--train", nargs="+", required=True,
                        help="CLASS=vector_file entries")
    parser.add_argument("--acc-pts", default=None,
                        help="Labelled accuracy points (ref_name column)")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    args = parser.parse_args()
    colours = ["#1f78b4", "#33a02c", "#b2df8a", "#808080", "#e31a1c", "#ff7f00"]
    train = {}
    for idx, entry in enumerate(args.train):
        name, path = entry.split("=", 1)
        train[name] = (path, idx + 1, colours[idx % len(colours)])

    def has(mod: str) -> bool:
        return importlib.util.find_spec(mod) is not None

    run_workflow(args.s2, args.dem, train, args.out_dir, acc_pts_vec=args.acc_pts,
                 use_gbdt=has("xgboost"), use_boruta=has("shap"),
                 use_shap=has("shap"), use_salib=has("SALib"))


if __name__ == "__main__":
    main()
