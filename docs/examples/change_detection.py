"""
Worked example: detecting forest loss between two dates with rsgispy.

This script is included step by step in the documentation guide
``docs/guides/change_detection.rst`` and is run end-to-end by the test suite
(``tests/test_change_detection_example.py``) on small synthetic data, so the
documented code is always tested.

Run it on your own data with, e.g.::

    python change_detection.py --s2-t1 s2_sref_2019.tif --s2-t2 s2_sref_2024.tif \\
        --lc landcover_2019.tif --forest-id 2 --acc-pts acc_pts_labelled.gpkg \\
        --out-dir outputs

The two images must be co-registered surface reflectance on the same pixel
grid (e.g. 10-band Sentinel-2 stacks from
``rsgispy.imagecalibration.sensorlvl2data.create_stacked_sen2_esa_l2a_img``,
no data value 0), and the land cover map must be on that grid too.
"""

import argparse
import importlib.util
import os
from typing import Any, Dict, Mapping, Optional, Sequence

# Band layout of the 10-band Sentinel-2 stack (B2, B3, B4, B5, B6, B7, B8,
# B8A, B11, B12): 1-based band index -> calc_indices band name.
S2_BAND_MAP = {1: "blue", 2: "green", 3: "red", 7: "nir", 9: "swir1", 10: "swir2"}


def run_workflow(
    s2_img_t1: str,
    s2_img_t2: str,
    lc_img: str,
    out_dir: str,
    forest_id: int = 2,
    acc_pts_vec: Optional[str] = None,
    acc_pts_lyr: Optional[str] = None,
    ref_col: str = "ref_cls",
    band_map: Optional[Mapping[int, str]] = None,
    cva_bands: Sequence[int] = (3, 7, 9, 10),
    thres_method: str = "li",
    n_acc_pts: int = 500,
    use_pbar: bool = True,
) -> Dict[str, Any]:
    """
    Run the worked example. ``lc_img`` is a land cover map of the first date
    in which ``forest_id`` is forest; ``acc_pts_vec`` (optional) holds
    accuracy points labelled with the reference class name ("Change" or
    "No change") in ``ref_col``. ``band_map`` maps band indices to
    ``calc_indices`` band names (default: ``S2_BAND_MAP``).
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-statements,import-outside-toplevel
    os.makedirs(out_dir, exist_ok=True)
    band_map = S2_BAND_MAP if band_map is None else band_map
    out: Dict[str, Any] = {}

    # [step-1-prepare]
    import rsgispy
    from rsgispy.imagecalc.calcindices import calc_indices
    from rsgispy.imagecalc.maths import BandDefn, band_maths

    # Valid where both dates have data (no data = 0 in the S2 stacks).
    vld_img = os.path.join(out_dir, "valid_t1_t2.tif")
    band_maths(
        vld_img,
        "(t1 > 0) && (t2 > 0) ? 1 : 0",
        [BandDefn("t1", s2_img_t1, 1), BandDefn("t2", s2_img_t2, 1)],
        datatype=rsgispy.TYPE_8UINT,
        use_pbar=use_pbar,
    )

    idx_imgs = {}
    for date, s2_img in (("t1", s2_img_t1), ("t2", s2_img_t2)):
        idx_imgs[date] = os.path.join(out_dir, f"indices_{date}.tif")
        calc_indices(
            s2_img,
            idx_imgs[date],
            band_names=dict(band_map),
            indices=["NDVI", "NBR"],  # band 1 NDVI, band 2 NBR
            in_gain=10000.0,  # reflectance x 10000 -> 0-1
            out_no_data_val=-999.0,
            use_pbar=use_pbar,
        )
    # [end-step-1-prepare]
    out["vld_img"] = vld_img

    # [step-2-difference]
    import rsgispy.changedetect.imgchngmthds as icm

    # Difference = date 1 - date 2, so vegetation loss is positive.
    diff_img = os.path.join(out_dir, "indices_diff.tif")
    icm.image_difference(
        idx_imgs["t1"], idx_imgs["t2"], diff_img,
        vld_msk_img=vld_img,
        no_data_val=-999.0,
        use_pbar=use_pbar,
    )

    # Ratio = date 1 / date 2 for SWIR1, which rises when forest is cleared.
    swir1 = [band for band, name in band_map.items() if name == "swir1"]
    ratio_img = os.path.join(out_dir, "swir1_ratio.tif")
    icm.image_ratio(
        s2_img_t1, s2_img_t2, ratio_img,
        vld_msk_img=vld_img,
        img_base_bands=swir1,
        img_chng_bands=swir1,
        no_data_val=-999.0,
        use_pbar=use_pbar,
    )
    # [end-step-2-difference]
    out["diff_img"] = diff_img

    # [step-3-threshold]
    from rsgispy.imagecalc.maths import calc_prop_true_exp
    from rsgispy.imageutils.sampling import extract_img_pxl_vals_in_msk
    from rsgispy.imageutils.stats import pop_thmt_img_stats
    import rsgispy.tools.thresholding as thres
    from rsgispy.tools.plotting import plot_histogram_threshold

    dndvi = extract_img_pxl_vals_in_msk(diff_img, [1], vld_img, 1,
                                        use_pbar=use_pbar)[:, 0]
    thresholds = {
        "otsu": thres.calc_otsu_threshold(dndvi),
        "yen": thres.calc_yen_threshold(dndvi),
        "li": thres.calc_li_threshold(dndvi),
    }
    print({name: round(val, 3) for name, val in thresholds.items()})
    dndvi_thres = thresholds[thres_method]
    plot_histogram_threshold(dndvi, os.path.join(out_dir, "dndvi_hist.png"),
                             threshold=dndvi_thres, title_str="dNDVI")

    # 0 = no data, 1 = no change, 2 = vegetation loss.
    chng_img = os.path.join(out_dir, "dndvi_change.tif")
    band_maths(
        chng_img,
        f"vld == 1 ? (dndvi > {dndvi_thres} ? 2 : 1) : 0",
        [BandDefn("vld", vld_img, 1), BandDefn("dndvi", diff_img, 1)],
        datatype=rsgispy.TYPE_8UINT,
        use_pbar=use_pbar,
    )
    pop_thmt_img_stats(chng_img, add_clr_tab=True, ignore_zero=True,
                       use_pbar=use_pbar)
    chng_prop = calc_prop_true_exp(f"dndvi > {dndvi_thres} ? 1 : 0",
                                   [BandDefn("dndvi", diff_img, 1)],
                                   in_vld_img=vld_img, use_pbar=use_pbar)
    print(f"Vegetation loss in {chng_prop * 100:.1f} % of the valid area")
    # [end-step-3-threshold]
    out["thresholds"] = thresholds
    out["chng_img"] = chng_img
    out["chng_prop"] = chng_prop

    # [step-4-cva]
    cva_img = os.path.join(out_dir, "cva_change.tif")
    cva_dist_img = os.path.join(out_dir, "cva_magnitude.tif")
    icm.change_vector_analysis(
        s2_img_t1, s2_img_t2, cva_img, cva_dist_img,
        vld_msk_img=vld_img,
        img_base_bands=list(cva_bands),  # e.g. Red, NIR, SWIR1, SWIR2
        img_chng_bands=list(cva_bands),
        use_pbar=use_pbar,
    )
    # CVA gives the size of the change; the dNDVI sign gives its direction.
    # 0 = no data, 1 = no change, 2 = change with vegetation loss,
    # 3 = other change.
    cva_dir_img = os.path.join(out_dir, "cva_change_direction.tif")
    band_maths(
        cva_dir_img,
        "cva == 1 ? (dndvi > 0 ? 2 : 3) : (cva == 0 ? 1 : 0)",
        [BandDefn("cva", cva_img, 1), BandDefn("dndvi", diff_img, 1)],
        datatype=rsgispy.TYPE_8UINT,
        use_pbar=use_pbar,
    )
    # [end-step-4-cva]
    out["cva_img"] = cva_img
    out["cva_dir_img"] = cva_dir_img

    # [step-5-sfa-mad]
    if icm.HAVE_SCIPY:
        sfa_img = os.path.join(out_dir, "sfa_change.tif")
        icm.slow_feature_analysis(
            s2_img_t1, s2_img_t2, sfa_img,
            os.path.join(out_dir, "sfa_distance.tif"),
            vld_msk_img=vld_img,
            img_base_bands=list(cva_bands),
            img_chng_bands=list(cva_bands),
            use_pbar=use_pbar,
        )
        mad_img = os.path.join(out_dir, "irmad_change.tif")
        icm.multivariate_alteration_detection(
            s2_img_t1, s2_img_t2, mad_img,
            os.path.join(out_dir, "irmad_chi2.tif"),
            os.path.join(out_dir, "irmad_variates.tif"),
            vld_msk_img=vld_img,
            img_base_bands=list(cva_bands),
            img_chng_bands=list(cva_bands),
            use_kmeans=False,  # Otsu threshold on the chi-square distance
            use_pbar=use_pbar,
        )
        out["sfa_img"] = sfa_img
        out["mad_img"] = mad_img
    # [end-step-5-sfa-mad]

    # [step-6-class-outliers]
    import rsgispy.changedetect.pxloutlierchng as poc

    # Forest pixels (from the date 1 land cover map) with unusually low
    # NDVI on date 2. Output: 0 = not forest, 1 = forest, 2 = forest loss.
    forest_loss_img = os.path.join(out_dir, "forest_loss_otsu.tif")
    otsu_thres = poc.find_class_otsu_outliers(
        idx_imgs["t2"], lc_img, forest_loss_img,
        low_thres=True,  # low NDVI values are the outliers
        img_mask_val=forest_id,
        img_band=1,  # NDVI
        plot_thres_file=os.path.join(out_dir, "forest_ndvi_t2_otsu.png"),
        use_pbar=use_pbar,
    )
    li_thres = poc.find_class_li_outliers(
        idx_imgs["t2"], lc_img, os.path.join(out_dir, "forest_loss_li.tif"),
        low_thres=True, img_mask_val=forest_id, img_band=1, use_pbar=use_pbar,
    )
    print(f"Forest NDVI thresholds: Otsu {otsu_thres:.3f}, Li {li_thres:.3f}")

    if importlib.util.find_spec("pyod") is not None:  # optional dependency
        from pyod.models.iforest import IForest

        poc.find_class_pyod_outliers(
            s2_img_t2, lc_img, os.path.join(out_dir, "forest_loss_iforest.tif"),
            IForest(contamination=0.1, random_state=42),
            img_mask_val=forest_id,
            img_bands=list(cva_bands),
            use_pbar=use_pbar,
        )
    # [end-step-6-class-outliers]
    out["forest_loss_img"] = forest_loss_img
    out["otsu_thres"] = otsu_thres

    # [step-7-accuracy]
    import rsgispy.classification.classaccuracymetrics as cam
    import rsgispy.classification.classrefpts as crp

    cls_lut = {1: "No change", 2: "Change"}
    # A stratified random sample of the change map to label (e.g. by visual
    # interpretation of very high resolution imagery), at least 50 per class.
    crp.create_stratified_random_ref_smpls_darts(
        chng_img, n_acc_pts, os.path.join(out_dir, "acc_pts_to_label.gpkg"),
        out_vec_lyr="acc_pts",
        min_cls_smpls=50,
        cls_no_data=0,
        class_names_lut=cls_lut,
        rnd_seed=42,
    )
    if acc_pts_vec is not None:  # points that have already been labelled
        acc_gdf = crp.pop_class_info_accuracy_pts(
            chng_img, acc_pts_vec, acc_pts_lyr, None, None, cls_lut,
            img_cls_col="map_cls", use_pbar=use_pbar,
        )
        acc_gdf = acc_gdf[acc_gdf["map_cls"].isin(list(cls_lut.values()))]
        acc_stats = cam.calc_acc_stats_img(
            acc_gdf, None, ref_col, "map_cls", chng_img, cls_lut,
            out_json_file=os.path.join(out_dir, "change_accuracy.json"),
            use_pbar=use_pbar,
        )
        # Area-weighted overall accuracy and estimated class proportions.
        print(acc_stats["quantity_metrics"]["Proportion Correct (C)"],
              acc_stats["est_prop_cls_area"])
        out["acc_stats"] = acc_stats
    # [end-step-7-accuracy]
    return out


def main() -> None:
    """Command line interface for the worked example."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    parser.add_argument("--s2-t1", required=True, help="Date 1 S2 SREF stack")
    parser.add_argument("--s2-t2", required=True, help="Date 2 S2 SREF stack")
    parser.add_argument("--lc", required=True, help="Date 1 land cover map")
    parser.add_argument("--forest-id", type=int, default=2,
                        help="Forest pixel value in the land cover map")
    parser.add_argument("--acc-pts", default=None,
                        help="Labelled accuracy points (ref_cls column)")
    parser.add_argument("--thres-method", default="li",
                        choices=["otsu", "yen", "li"], help="dNDVI threshold")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    args = parser.parse_args()
    run_workflow(args.s2_t1, args.s2_t2, args.lc, args.out_dir,
                 forest_id=args.forest_id, acc_pts_vec=args.acc_pts,
                 thres_method=args.thres_method)


if __name__ == "__main__":
    main()
