"""
Worked example: co-registering an image to a reference image with rsgispy.

This script is included step by step in the documentation guide
``docs/guides/image_coregistration.rst`` and is run end-to-end by the test
suite (``tests/test_image_coregistration_example.py``) on a small synthetic
image pair, so the documented code is always tested.

Run it on your own data with, e.g.::

    python image_coregistration.py --ref ref_s2_10m.tif \\
        --float drone_ortho_10m.tif --band 7 --flt-band 4 \\
        --gcps float_gcps.gpkg --out-dir outputs
"""

import argparse
import os
from typing import Any, Dict, Optional


def run_workflow(
    ref_img: str,
    flt_img: str,
    out_dir: str,
    gcp_vec: Optional[str] = None,
    ref_band: int = 1,
    flt_band: int = 1,
    win_size: int = 64,
    pixel_gap: int = 50,
    search_area: int = 15,
    residual_thres: float = 10.0,
    use_pbar: bool = True,
) -> Dict[str, Any]:
    """
    Run the worked example: register ``flt_img`` (the floating image) to
    ``ref_img`` (the reference image) using band ``flt_band`` and
    ``ref_band``. ``gcp_vec`` optionally holds ground control points
    digitised on the floating image, with their true coordinates in the
    ``x_match`` and ``y_match`` columns. ``residual_thres`` is in map units.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-statements,import-outside-toplevel
    os.makedirs(out_dir, exist_ok=True)
    out: Dict[str, Any] = {}

    # [step-1-measure]
    import rsgispy
    import rsgispy.imageregistration.shift as imreg_shift
    from rsgispy.imagecalc.comparison import calc_img_similarity_metrics

    metrics = [rsgispy.METRIC_CORRELATION, rsgispy.METRIC_NORM_MUTUAL_INFO]
    before = calc_img_similarity_metrics(ref_img, flt_img, metrics=metrics,
                                         img_a_bands=[ref_band],
                                         img_b_bands=[flt_band])
    offset = imreg_shift.find_image_offset(
        ref_img,
        flt_img,
        ref_img_bands=[ref_band],
        flt_img_bands=[flt_band],
        metric_type=rsgispy.METRIC_CORRELATION,
        x_search=search_area,  # largest shift tested (pixels)
        y_search=search_area,
        sub_pxl_res=4,  # 0.25 pixel precision
        return_details=True,
    )
    print(before)
    print(offset)
    # [end-step-1-measure]
    out["before"] = before
    out["offset"] = offset

    # [step-2-shift]
    shift_img = os.path.join(out_dir, "flt_shifted.tif")
    imreg_shift.apply_offset_to_image(
        flt_img,
        shift_img,
        x_offset=offset["x_offset_map"],  # added to the image origin
        y_offset=offset["y_offset_map"],
        use_pbar=use_pbar,
    )
    # [end-step-2-shift]
    out["shift_img"] = shift_img

    # [step-3-tie-points]
    import rsgispy.imageregistration as imreg
    import rsgispy.imageregistration.tiepoints as imreg_tp

    tie_df = imreg_tp.generate_tie_points(
        ref_img,
        flt_img,
        method=imreg.TIE_METHOD_TEMPLATE_MATCH,
        ref_band=ref_band,
        flt_band=flt_band,
        pixel_gap=pixel_gap,  # spacing of the candidate points (pixels)
        win_size=win_size,  # matching window (pixels)
        search_area=search_area,  # search radius (pixels)
        threshold=0.6,  # minimum correlation of a match
        metric_type=rsgispy.METRIC_CORRELATION,
        use_pbar=use_pbar,
    )
    print(len(tie_df), "tie points")
    print(tie_df[["ref_x", "ref_y", "dx_map", "dy_map", "metric"]].describe())
    # [end-step-3-tie-points]
    out["n_tie_points"] = len(tie_df)

    # [step-4-filter]
    inliers_df, diag = imreg_tp.filter_tie_points_outliers(
        tie_df,
        method=imreg.OUTLIER_METHOD_RANSAC,
        model=imreg.TRANSFORM_AFFINE,
        residual_thres=residual_thres,  # map units (metres)
        rnd_seed=42,
        return_model=True,
    )
    print({k: v for k, v in diag.items() if k != "model"})
    tie_file = os.path.join(out_dir, "tie_points_inliers.csv")
    inliers_df.to_csv(tie_file, index=False)  # can be passed to the warp
    # [end-step-4-filter]
    out["diag"] = diag
    out["tie_file"] = tie_file

    # [step-5-choose-model]
    import numpy as np

    # Hold back every third tie point as an independent check point.
    is_chk = np.arange(len(inliers_df)) % 3 == 0
    fit_df, chk_df = inliers_df[~is_chk], inliers_df[is_chk]
    chk_rmse = {}
    for model in (imreg.TRANSFORM_SHIFT, imreg.TRANSFORM_AFFINE,
                  imreg.TRANSFORM_POLY2):
        mdl = imreg_tp.fit_geometric_model(fit_df["flt_x"], fit_df["flt_y"],
                                           fit_df["ref_x"], fit_df["ref_y"],
                                           model=model)
        pred_x, pred_y = imreg_tp.evaluate_geometric_model(
            mdl, chk_df["flt_x"], chk_df["flt_y"])
        chk_rmse[model] = float(np.sqrt(np.mean(
            (pred_x - chk_df["ref_x"]) ** 2 + (pred_y - chk_df["ref_y"]) ** 2)))
    print("Check point RMSE (m):", chk_rmse)
    # [end-step-5-choose-model]
    out["chk_rmse"] = chk_rmse

    # [step-6-warp]
    import rsgispy.imageregistration.warping as imreg_warp

    warp_img = os.path.join(out_dir, "flt_warped_affine.tif")
    warp_info = imreg_warp.warp_with_tie_points(
        ref_img,
        flt_img,
        warp_img,
        inliers_df,
        interp_method=rsgispy.INTERP_CUBIC,
        transform_type=imreg.TRANSFORM_AFFINE,
        match_ref_grid=True,  # output on the reference image pixel grid
        no_data_val=0,
        use_pbar=use_pbar,
    )
    print(warp_info)
    # [end-step-6-warp]
    out["warp_img"] = warp_img

    # [step-7-coregister]
    coreg_img = os.path.join(out_dir, "flt_coregistered.tif")
    coreg_info = imreg_warp.coregister_images(
        ref_img,
        flt_img,
        coreg_img,
        tie_method=imreg.TIE_METHOD_TEMPLATE_MATCH,
        outlier_method=imreg.OUTLIER_METHOD_RANSAC,
        transform_type=imreg.TRANSFORM_AFFINE,
        ref_band=ref_band,
        flt_band=flt_band,
        pixel_gap=pixel_gap,
        win_size=win_size,
        search_area=search_area,
        threshold=0.6,
        residual_thres=residual_thres,
        interp_method=rsgispy.INTERP_CUBIC,
        no_data_val=0,
        rnd_seed=42,
        use_pbar=use_pbar,
    )
    print(coreg_info)
    # [end-step-7-coregister]
    out["coreg_img"] = coreg_img
    out["coreg_info"] = coreg_info

    # [step-8-gcps]
    if gcp_vec is not None:
        gcp_img = os.path.join(out_dir, "flt_with_gcps.tif")
        imreg_warp.add_vec_pts_as_gcps_to_img(
            flt_img,
            gcp_img,
            gcp_vec,
            gcp_x_col="x_match",  # true X of each point
            gcp_y_col="y_match",  # true Y of each point
            calc_stats=False,
        )
        gcp_warp_img = os.path.join(out_dir, "flt_warped_gcps.tif")
        imreg_warp.warp_with_gcps(
            ref_img,
            gcp_img,
            gcp_warp_img,
            interp_method=rsgispy.INTERP_CUBIC,
            poly_order=1,  # affine: needs at least 3 well-spread GCPs
            no_data_val=0,
            use_pbar=use_pbar,
        )
        out["gcp_warp_img"] = gcp_warp_img
    # [end-step-8-gcps]

    # [step-9-assess]
    from rsgispy.imagecalc.comparison import calc_img_difference

    results = {"before": flt_img, "shift": shift_img, "affine": warp_img}
    if gcp_vec is not None:
        results["gcps"] = out["gcp_warp_img"]
    assess = {}
    for name, img in results.items():
        scores = calc_img_similarity_metrics(ref_img, img, metrics=metrics,
                                             img_a_bands=[ref_band],
                                             img_b_bands=[flt_band])
        resid = imreg_shift.find_image_offset(
            ref_img, img, [ref_band], [flt_band], x_search=5, y_search=5,
            return_details=True)
        scores["resid_offset_m"] = float(np.hypot(resid["x_offset_map"],
                                                  resid["y_offset_map"]))
        assess[name] = scores
        print(name, {k: round(v, 3) for k, v in scores.items()})
    diff_img = os.path.join(out_dir, "ref_minus_affine.tif")
    calc_img_difference(ref_img, warp_img, diff_img, use_pbar=use_pbar)
    # [end-step-9-assess]
    out["assess"] = assess
    out["diff_img"] = diff_img
    return out


def main() -> None:
    """Command line interface for the worked example."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    parser.add_argument("--ref", required=True, help="Reference image")
    parser.add_argument("--float", required=True, help="Image to register")
    parser.add_argument("--band", type=int, default=1,
                        help="Reference image band used for matching")
    parser.add_argument("--flt-band", type=int, default=None,
                        help="Floating image band (default: same as --band)")
    parser.add_argument("--gcps", default=None,
                        help="GCP points (x_match/y_match columns)")
    parser.add_argument("--win-size", type=int, default=64,
                        help="Matching window size (pixels)")
    parser.add_argument("--search", type=int, default=15,
                        help="Search radius (pixels)")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    args = parser.parse_args()
    run_workflow(args.ref, args.float, args.out_dir, gcp_vec=args.gcps,
                 ref_band=args.band, flt_band=args.flt_band or args.band,
                 win_size=args.win_size, search_area=args.search)


if __name__ == "__main__":
    main()
