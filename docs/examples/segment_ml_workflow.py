"""
Worked example: object-based (segment) classification and regression with
rsgispy.

This script is included step by step in the documentation guide
``docs/guides/segment_ml_workflow.rst`` and is run end-to-end by the test
suite (``tests/test_segment_ml_workflow_example.py``) on small synthetic data,
so the documented code is always tested.

Run it on your own data with, e.g.::

    python segment_ml_workflow.py --s2 s2_sref_10m.tif --slope slope.tif \\
        --train Water=water.gpkg Forest=forest.gpkg Grass=grass.gpkg \\
        Urban=urban.gpkg --plots canopy_plots.gpkg --acc-pts acc_pts.gpkg \\
        --s2-new s2_sref_10m_2024.tif --slope-new slope.tif --out-dir outputs
"""

import argparse
import os
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


def run_workflow(
    s2_img: str,
    slope_img: str,
    train_vecs: Mapping[str, Tuple[str, int, str]],
    plots_vec: Optional[str],
    acc_pts_vec: str,
    out_dir: str,
    s2_img_new: Optional[str] = None,
    slope_img_new: Optional[str] = None,
    s2_bands: Sequence[int] = (1, 2, 3, 7, 9, 10),
    s2_band_names: Sequence[str] = ("Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"),
    n_clusters: int = 60,
    min_pixels: int = 50,
    sampling: int = 100,
    use_gbdt: bool = True,
    use_pbar: bool = True,
) -> Dict[str, Any]:
    """
    Run the worked example. ``train_vecs`` maps each class name to
    ``(training vector file, output class ID, colour)``; ``plots_vec`` holds
    canopy cover plots (``canopy_cover`` column; ``None`` skips the
    regression) and ``acc_pts_vec`` accuracy points with the reference class
    name in ``ref_name``.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-statements,import-outside-toplevel
    os.makedirs(out_dir, exist_ok=True)
    out: Dict[str, Any] = {}

    # [step-1-segmentation]
    import rsgispy.segmentation

    segs_img = os.path.join(out_dir, "s2_segs.tif")
    rsgispy.segmentation.shepherd_segmentation(
        s2_img,
        segs_img,
        n_clusters=n_clusters,  # number of spectral clusters (K-means)
        min_pixels=min_pixels,  # minimum segment size in pixels
        sampling=sampling,  # use every n-th pixel to fit the clusters
        bands=list(s2_bands),  # bands used to segment
        no_data_val=0,
        use_pbar=use_pbar,
    )
    # [end-step-1-segmentation]

    # [step-2-features]
    from rsgispy.classification.classdata import (
        ClassificationConfig,
        ClassInfoObj,
        ImageBandInfo,
    )
    import rsgispy.classification.classsegments as cseg

    img_band_info = [
        ImageBandInfo(s2_img, name="s2", bands=list(s2_bands),
                      band_names=list(s2_band_names)),
        ImageBandInfo(slope_img, name="slope", bands=[1], band_names=["slope"]),
    ]
    class_info = {
        cls_name: ClassInfoObj(out_id=out_id, vec_file=vec_file, colour=colour)
        for cls_name, (vec_file, out_id, colour) in train_vecs.items()
    }
    cls_cfg = ClassificationConfig(img_band_info, class_info)

    cfg_file = os.path.join(out_dir, "cls_seg_config.json")
    cls_cfg = cseg.create_cls_segment_features(
        cls_cfg,
        segs_img,
        os.path.join(out_dir, "s2_segs.parquet"),  # GeoParquet (snappy)
        out_cfg_file=cfg_file,
        stats=["mean", "stddev", "min", "max"],
        shape_metrics=["area", "compactness", "solidity", "elongation"],
        band_opts={"s2": {"no_data_val": 0}},  # S2 stack no data value
        min_seg_pxls=1,
        use_pbar=use_pbar,
    )
    print(cls_cfg.seg_info.feature_cols)
    # [end-step-2-features]
    out["cfg_file"] = cfg_file

    # [step-3-config]
    import json

    with open(cfg_file, encoding="utf-8") as f_obj:
        seg_json = json.load(f_obj)["seg_info"]
    keys = ("label_img", "seg_vec_file", "id_col", "stats", "shape_metrics",
            "missing")
    print({key: seg_json[key] for key in keys})
    # [end-step-3-config]

    # [step-4-samples]
    train_file = os.path.join(out_dir, "seg_train.parquet")
    test_file = os.path.join(out_dir, "seg_test.parquet")
    train_df, test_df, _ = cseg.extract_and_split_class_segment_samples(
        cfg_file,
        out_train_file=train_file,
        out_test_file=test_file,
        test_prop=0.3,
        valid_prop=0.0,  # few samples: use a test set (and CV) only
        rnd_seed=42,
    )
    print(train_df["class_name"].value_counts())
    print(len(train_df), "training and", len(test_df), "test segments")
    # [end-step-4-samples]
    out["n_train"] = len(train_df)

    # [step-5-features-explore]
    from rsgispy.classification.classfeatures import select_features_corr

    selected = select_features_corr(train_file, cfg_file, n_min_clusters=2,
                                    n_max_clusters=6, use_pbar=use_pbar)
    print("Selected features:", selected)
    # This example keeps all the features; to use the selection instead:
    # cls_cfg.seg_info.selected_features = selected
    # [end-step-5-features-explore]

    # [step-6-train]
    from sklearn.ensemble import RandomForestClassifier
    import rsgispy.classification.classsklearn as csk

    rf_file = os.path.join(out_dir, "rf_model.joblib")
    metrics = csk.train_sklearn_classifier(
        cfg_file,
        RandomForestClassifier(n_estimators=200, class_weight="balanced",
                               random_state=42),
        train_file,
        test_data=test_file,
        out_model_file=rf_file,
    )
    print({k: round(v, 3) for k, v in metrics.items() if "accuracy" in k})
    # [end-step-6-train]
    out["rf_metrics"] = metrics

    # [step-6b-train-gbdt]
    if use_gbdt:
        import rsgispy.classification.classxgboost as cxgb

        xgb_model, xgb_metrics = cxgb.train_xgboost_multiclass_classifier(
            cfg_file, train_file, test_data=test_file, return_model=True
        )
        print({k: round(v, 3) for k, v in xgb_metrics.items() if "accuracy" in k})
    # [end-step-6b-train-gbdt]

    # [step-7-apply]
    cls_vec = os.path.join(out_dir, "s2_segs_classified.parquet")
    cseg.apply_sklearn_classifier_segments(
        cfg_file,
        rf_file,
        cls_vec,
        out_probs="max",  # add the maximum class probability
        copy_cols=[],  # keep only seg_id, geometry and the outputs
        chunk_size=100_000,
        use_pbar=use_pbar,
    )
    import geopandas as gpd

    print(gpd.read_parquet(cls_vec).head())
    # [end-step-7-apply]
    out["cls_vec"] = cls_vec

    # [step-8-rasterise]
    cls_img = os.path.join(out_dir, "s2_segs_classified.tif")
    cseg.rasterise_segment_classification(cfg_file, cls_vec, cls_img,
                                          use_pbar=use_pbar)
    # Store the class names (with the colours) in the GeoTIFF.
    from rsgispy.imageutils.colourtable import set_thematic_classes

    set_thematic_classes(cls_img, {out_id: (cls_name, colour) for cls_name,
                                   (_, out_id, colour) in train_vecs.items()})
    # [end-step-8-rasterise]
    out["cls_img"] = cls_img

    # [step-9-accuracy]
    import rsgispy.classification.classaccuracymetrics as cam

    acc_pts = cseg.pop_segment_cls_accuracy_pts(cfg_file, cls_vec, acc_pts_vec)
    acc_pts = acc_pts[acc_pts["seg_cls_name"].notna()]  # drop unclassified
    pt_stats = cam.calc_acc_stats_ptonly(acc_pts, None, "ref_name", "seg_cls_name")
    area_stats = cam.calc_acc_stats_img(
        acc_pts,
        None,
        "ref_name",
        "seg_cls_name",
        cls_img,
        class_names_lut=cls_cfg.get_class_names_lut(),
        use_pbar=use_pbar,
    )
    # Unweighted point accuracy, then the area-weighted proportion correct
    print(pt_stats["accuracy"],
          area_stats["quantity_metrics"]["Proportion Correct (C)"])
    # [end-step-9-accuracy]
    out["pt_stats"] = pt_stats

    # [step-10-new-scene]
    if s2_img_new is not None:
        from rsgispy.ml.segments import create_segment_features, substitute_scene

        segs_new = os.path.join(out_dir, "s2_new_segs.tif")
        rsgispy.segmentation.shepherd_segmentation(
            s2_img_new, segs_new, n_clusters=n_clusters, min_pixels=min_pixels,
            sampling=sampling, bands=list(s2_bands), no_data_val=0,
            use_pbar=use_pbar,
        )
        new_cfg = substitute_scene(
            cls_cfg,
            {"s2": s2_img_new, "slope": slope_img_new or slope_img},
            label_img=segs_new,
        )
        new_cfg.seg_info = create_segment_features(
            segs_new,
            None,
            os.path.join(out_dir, "s2_new_segs.parquet"),
            template=new_cfg.seg_info,  # same features as the trained model
            use_pbar=use_pbar,
        )
        new_cls_vec = os.path.join(out_dir, "s2_new_segs_classified.parquet")
        cseg.apply_sklearn_classifier_segments(new_cfg, rf_file, new_cls_vec,
                                               copy_cols=[], use_pbar=use_pbar)
        cseg.rasterise_segment_classification(
            new_cfg, new_cls_vec, os.path.join(out_dir, "s2_new_classified.tif"),
            use_pbar=use_pbar,
        )
        out["new_cls_vec"] = new_cls_vec
    # [end-step-10-new-scene]

    # [step-11-regression]
    if plots_vec is not None:
        from sklearn.ensemble import RandomForestRegressor
        from rsgispy.regression.regdata import RegressionConfig
        import rsgispy.regression.regsegments as rsegs
        import rsgispy.regression.regsklearn as rsk

        # Reuse the classification segment features for the regression.
        reg_cfg = RegressionConfig(
            img_band_info, "canopy_cover", vec_file=plots_vec,
            seg_info=cls_cfg.seg_info,
        )
        reg_train, reg_test, _ = rsegs.extract_and_split_reg_segment_samples(
            reg_cfg, test_prop=0.3, valid_prop=0.0, rnd_seed=42
        )
        rf_reg, reg_metrics = rsk.train_sklearn_regressor(
            reg_cfg, RandomForestRegressor(n_estimators=200, random_state=42),
            reg_train, test_data=reg_test, return_model=True,
        )
        print({k: round(v, 3) for k, v in reg_metrics.items() if "r2" in k})
        reg_vec = os.path.join(out_dir, "s2_segs_canopy.parquet")
        rsegs.apply_sklearn_regressor_segments(
            reg_cfg, rf_reg, reg_vec, out_quantiles=[0.05, 0.95], copy_cols=[],
            use_pbar=use_pbar,
        )
        rsegs.rasterise_segment_regression(
            reg_cfg, reg_vec, os.path.join(out_dir, "canopy_cover.tif"),
            pred_cols=["pred_canopy_cover", "pred_canopy_cover_q0.05",
                       "pred_canopy_cover_q0.95"],
            use_pbar=use_pbar,
        )
        out["reg_vec"] = reg_vec
    # [end-step-11-regression]

    # [step-12-clustering]
    from sklearn.cluster import KMeans
    from rsgispy.ml.scaling import FeatureScaler

    clu_vec = os.path.join(out_dir, "s2_segs_clusters.parquet")
    cseg.cluster_segments(
        cfg_file,
        KMeans(n_clusters=4, n_init=10, random_state=42),
        clu_vec,
        scaler=FeatureScaler(method="zscore"),  # put features on one scale
        copy_cols=[],
        use_pbar=use_pbar,
    )
    cseg.rasterise_segment_classification(
        None, clu_vec, os.path.join(out_dir, "s2_segs_clusters.tif"),
        cls_col="cluster_id", label_img=segs_img, use_pbar=use_pbar,
    )
    # [end-step-12-clustering]
    out["clu_vec"] = clu_vec
    return out


def main() -> None:
    """Command line interface for the worked example."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--s2", required=True, help="Sentinel-2 SREF stack")
    parser.add_argument("--slope", required=True, help="Slope image")
    parser.add_argument("--train", nargs="+", required=True,
                        help="CLASS=vector_file entries")
    parser.add_argument("--plots", default=None,
                        help="Canopy cover plots (optional, for the regression)")
    parser.add_argument("--acc-pts", required=True, help="Accuracy points")
    parser.add_argument("--s2-new", default=None, help="Second scene S2 stack")
    parser.add_argument("--slope-new", default=None, help="Second scene slope")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    args = parser.parse_args()
    colours = ["#1f78b4", "#33a02c", "#b2df8a", "#808080", "#e31a1c", "#ff7f00"]
    train = {}
    for idx, entry in enumerate(args.train):
        name, path = entry.split("=", 1)
        train[name] = (path, idx + 1, colours[idx % len(colours)])
    run_workflow(args.s2, args.slope, train, args.plots, args.acc_pts, args.out_dir,
                 s2_img_new=args.s2_new, slope_img_new=args.slope_new)


if __name__ == "__main__":
    main()
