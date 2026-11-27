"""
Worked example: wetland mapping with linear spectral unmixing, following the
TropWet method (Hardy et al., 2020), with rsgispy.

This script is included step by step in the documentation guide
``docs/guides/spectral_unmixing.rst`` and is run end-to-end by the test suite
(``tests/test_spectral_unmixing_example.py``) on small synthetic data, so the
documented code is always tested.

Run it on your own data with, e.g.::

    python spectral_unmixing.py --sref ls_20200115.tif ls_20200131.tif \\
        ls_20200216.tif --dem srtm_90m.tif --out-dir tropwet
"""

import argparse
import os
from typing import Any, Dict, Optional, Sequence

# [step-7-class-table]
# TropWet classes: pixel value -> (name, colour). 0 is no data.
TROPWET_CLASSES = {
    1: ("Open Water (OW)", "#08519C"),
    2: ("Emergent Flooded Vegetation (EFV)", "#3FB8AF"),
    3: ("Wet Bare Sand (WBS)", "#B39C4D"),
    4: ("Dense Vegetation (GV)", "#1B7837"),
    5: ("Dry Sparse Vegetation (DSV)", "#A6D96A"),
    6: ("Dry Bare Sand (DBS)", "#F4E2B8"),
    7: ("Mixed / Other", "#BDBDBD"),
}
# [end-step-7-class-table]


def run_workflow(
    sref_imgs: Sequence[str],
    dem_img: str,
    out_dir: str,
    no_data_val: float = 0,
    refl_scale: float = 10000.0,
    wavelengths: Optional[Sequence[float]] = (482, 561, 655, 865, 1609, 2201),
    min_em_pxls: int = 10,
    max_slope: float = 2.5,
    use_pbar: bool = True,
) -> Dict[str, Any]:
    """
    Run the worked example. ``sref_imgs`` are cloud-masked surface
    reflectance images of the same area on different dates (or a single
    composite), each with the bands Blue, Green, Red, NIR, SWIR1 and SWIR2 in
    that order, scaled by ``refl_scale`` (reflectance x 10000 by default) and
    with ``no_data_val`` in every band of masked pixels. ``dem_img`` is a
    digital elevation model (e.g. SRTM) covering the same area, in metres.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-statements,import-outside-toplevel
    os.makedirs(out_dir, exist_ok=True)
    out: Dict[str, Any] = {}

    # [step-1-composite]
    import numpy as np
    import pandas as pd
    import rasterio
    import rsgispy
    import rsgispy.imagecalc.maths as img_maths
    from rsgispy.imageutils.composite import create_percentile_composite

    band_names = ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"]
    nd = no_data_val

    def prep_refl(in_img: str, out_img: str) -> str:
        """Keep no data as 0 and clip valid values to >= 1 (see the guide)."""
        defs = [img_maths.BandDefn(f"b{idx}", in_img, idx) for idx in range(1, 7)]
        vld = " && ".join(f"b{idx} != {nd}" for idx in range(1, 7))
        exp = ", ".join([f"(vld = {vld}) ? max(b1, 1) : 0"]
                        + [f"vld ? max(b{idx}, 1) : 0" for idx in range(2, 7)])
        img_maths.band_maths(out_img, exp, defs, datatype=rsgispy.TYPE_16UINT,
                             band_names=band_names, no_data_val=0,
                             use_pbar=use_pbar)
        return out_img

    if len(sref_imgs) > 1:
        # Per-pixel, per-band median of the valid observations (an
        # observation is valid where all six bands are valid).
        median_img = os.path.join(out_dir, "sref_median.tif")
        create_percentile_composite(
            sref_imgs, median_img, [50], no_data_val=nd,
            datatype=rsgispy.TYPE_16UINT,  # rounded to integers
            use_pbar=use_pbar)
    else:
        median_img = sref_imgs[0]  # already a composite
    comp_img = prep_refl(median_img, os.path.join(out_dir, "sref_composite.tif"))
    # [end-step-1-composite]
    out["comp_img"] = comp_img

    # [step-2-indices]
    from rsgispy.imagecalc.calcindices import calc_indices

    idx_img = os.path.join(out_dir, "composite_indices.tif")
    calc_indices(
        comp_img, idx_img,
        band_names={1: "blue", 2: "green", 3: "red", 4: "nir", 5: "swir1",
                    6: "swir2"},
        indices=["NDVI", "MNDWI"],  # (NIR-R)/(NIR+R), (G-SWIR1)/(G+SWIR1)
        in_gain=refl_scale,  # to reflectance 0-1
        no_data_val=0,  # output no data is -999
        use_pbar=use_pbar,
    )
    # [end-step-2-indices]
    out["idx_img"] = idx_img

    # [step-3-endmembers]
    import rsgispy.imagespectral.endmembers as spec_em
    from rsgispy.imagecalc.stats import count_pxls_of_val
    from rsgispy.imageutils.sampling import extract_img_pxl_vals_in_msk

    # TropWet Table 1 thresholds: 1 = Water, 2 = Vegetation, 3 = Sand.
    em_rules = {
        "Water": "mndwi > 0.5",
        "Vegetation": "ndvi > 0.7",
        "Sand": "ndvi > 0.16 && ndvi < 0.17 && mndwi > -0.28 && mndwi < 0.26",
    }
    em_msk_img = os.path.join(out_dir, "endmember_pxls.tif")
    img_maths.band_maths(
        em_msk_img,
        f"({em_rules['Water']}) ? 1 : (({em_rules['Vegetation']}) ? 2 : "
        f"(({em_rules['Sand']}) ? 3 : 0))",
        [img_maths.BandDefn("ndvi", idx_img, 1),
         img_maths.BandDefn("mndwi", idx_img, 2)],
        datatype=rsgispy.TYPE_8UINT, use_pbar=use_pbar,
    )
    em_counts = dict(zip(em_rules, count_pxls_of_val(
        em_msk_img, [1, 2, 3], img_band=1, use_pbar=use_pbar).tolist()))
    print("Pure pixels per endmember:", em_counts)
    too_few = [name for name, cnt in em_counts.items() if cnt < min_em_pxls]
    if too_few:
        raise rsgispy.RSGISPyException(
            f"Too few pure pixels for {too_few}: check the thresholds, the "
            "date range and the reflectance scaling.")

    # Endmember spectrum = mean of the selected pure pixels.
    em_df = pd.DataFrame({
        name: extract_img_pxl_vals_in_msk(
            comp_img, list(range(1, 7)), em_msk_img, val, no_data_val=0,
            use_pbar=use_pbar).mean(axis=0)
        for val, name in enumerate(em_rules, start=1)
    }, index=pd.Index(band_names, name="band"))
    print(em_df.round(0))  # rows = bands, columns = endmembers
    spec_em.write_endmembers_df(em_df, os.path.join(out_dir, "endmembers.csv"))
    out["em_plot"] = os.path.join(out_dir, "unmix_endmembers.png")
    fig, ax = spec_em.plot_endmembers(
        em_df, gain=refl_scale, plot_title="TropWet endmembers",
        wavelengths=wavelengths, out_plot_file=out["em_plot"],
    )
    # [end-step-3-endmembers]
    out["em_counts"] = em_counts
    out["em_msk_img"] = em_msk_img
    out["em_df"] = em_df

    # [step-4-unmix]
    import rsgispy.imagespectral.unmixing as unmix

    def unmix_pct(refl_img: str, out_base: str) -> str:
        """FCLS unmixing (no data -9999), then fractions as % (no data -999)."""
        frac_img = f"{out_base}_fcls.tif"
        unmix.spec_unmix_fcls(refl_img, frac_img, em_df, use_pbar=use_pbar)
        pct_img = f"{out_base}_pct.tif"
        img_maths.band_maths(
            pct_img,
            ", ".join(f"{name} == -9999 ? -999 : 100 * {name}"
                      for name in em_df.columns),
            [img_maths.BandDefn(name, frac_img, idx)
             for idx, name in enumerate(em_df.columns, start=1)],
            band_names=list(em_df.columns), no_data_val=-999, use_pbar=use_pbar,
        )
        return pct_img

    pct_img = unmix_pct(comp_img, os.path.join(out_dir, "composite"))

    # For comparison: unconstrained fractions can be negative or exceed one.
    ucls_img = os.path.join(out_dir, "composite_ucls.tif")
    unmix.spec_unmix_ucls(comp_img, ucls_img, em_df, use_pbar=use_pbar)

    # Quality check: model error per pixel (band 1 = RMSE in image units).
    err = {}
    for method, frac_img in {"fcls": os.path.join(out_dir, "composite_fcls.tif"),
                             "ucls": ucls_img}.items():
        err_img = os.path.join(out_dir, f"composite_{method}_err.tif")
        unmix.calc_unmixing_rmse_residual_err(comp_img, frac_img, err_img, em_df,
                                              use_pbar=use_pbar)
        with rasterio.open(frac_img) as src:
            fracs = src.read().reshape(src.count, -1)
            vld = fracs[0] != src.nodata  # the pixels that were unmixed
        with rasterio.open(err_img) as src:
            rmse = src.read(1).ravel()
        err[method] = {"neg_pct": 100 * np.mean(np.any(fracs[:, vld] < 0, axis=0)),
                       "sum_min": fracs[:, vld].sum(axis=0).min(),
                       "sum_max": fracs[:, vld].sum(axis=0).max(),
                       "rmse_median": np.median(rmse[vld]),
                       "rmse_p99": np.percentile(rmse[vld], 99)}
    print(pd.DataFrame(err).T.round(2))
    # [end-step-4-unmix]
    out["pct_img"] = pct_img
    out["ucls_img"] = ucls_img
    out["fcls_err_img"] = os.path.join(out_dir, "composite_fcls_err.tif")
    out["unmix_err"] = err

    # [step-5-burn]
    from rsgispy.imagecalc.stats import calc_basic_img_band_stats

    # Burned Area Index (Chuvieco et al., 2002) on reflectance 0-1; 0 = no data.
    bai_img = os.path.join(out_dir, "composite_bai.tif")
    img_maths.band_maths(
        bai_img,
        f"red > 0 ? 1 / ((0.1 - red / {refl_scale})^2 "
        f"+ (0.06 - nir / {refl_scale})^2) : 0",
        [img_maths.BandDefn("red", comp_img, 3),
         img_maths.BandDefn("nir", comp_img, 4)],
        use_pbar=use_pbar,
    )
    _, _, bai_mean, bai_std = calc_basic_img_band_stats(
        bai_img, no_data_val=0, use_pbar=use_pbar)
    bai_thres = float(bai_mean[0] + bai_std[0])
    print(f"BAI threshold (mean + 1 sd): {bai_thres:.1f}")

    # False water: bright BAI, similar water and sand fractions
    # (-0.5 < NDWS < 0.5) and little vegetation (V < 10 %).
    pct_defs = [img_maths.BandDefn(name, pct_img, idx)
                for idx, name in enumerate(["W", "V", "S"], start=1)]
    burn_img = os.path.join(out_dir, "composite_burn_flag.tif")
    img_maths.band_maths(
        burn_img,
        f"(W >= 0 && W + S > 0 && bai > {bai_thres} && abs((W - S) / (W + S)) < 0.5"
        " && V < 10) ? 1 : 0",
        pct_defs + [img_maths.BandDefn("bai", bai_img, 1)],
        datatype=rsgispy.TYPE_8UINT, use_pbar=use_pbar,
    )
    # Reassign the water fraction of flagged pixels to sand.
    corr_img = os.path.join(out_dir, "composite_pct_burn_corr.tif")
    img_maths.band_maths(
        corr_img, "burn ? 0 : W, V, burn ? S + W : S",
        pct_defs + [img_maths.BandDefn("burn", burn_img, 1)],
        band_names=list(em_df.columns), no_data_val=-999, use_pbar=use_pbar,
    )
    n_burn = count_pxls_of_val(burn_img, [1], img_band=1, use_pbar=use_pbar)[0]
    print(f"{n_burn} pixels corrected for burn scars")
    # [end-step-5-burn]
    out["bai_thres"] = bai_thres
    out["burn_img"] = burn_img
    out["corr_img"] = corr_img

    # [step-6-slope]
    from rsgispy.elevation.terrain import slope
    from rsgispy.imageutils.warp import resample_img_to_match

    # Slope on the DEM's own grid, then resampled to the image grid.
    dem_slope_img = os.path.join(out_dir, "dem_slope.tif")
    slope(dem_img, dem_slope_img, unit="degrees", use_pbar=use_pbar)
    slope_img = os.path.join(out_dir, "slope.tif")
    resample_img_to_match(comp_img, dem_slope_img, slope_img,
                          interp_method=rsgispy.INTERP_BILINEAR,
                          use_pbar=use_pbar)
    # [end-step-6-slope]
    out["slope_img"] = slope_img

    # [step-7-classify]
    from rsgispy.imageutils.colourtable import set_thematic_classes

    # TropWet Table 3 (fractions in %), tested in this order.
    rules = [
        (1, "W >= 75"),
        (2, "W + V >= 75 && V >= 25 && V < 75 && W >= 25 && W < 75"),
        (3, "W + S >= 75 && W >= 25 && W < 75 && S >= 25 && S < 75"),
        (4, "V >= 75"),
        (5, "V + S >= 75 && V >= 25 && V < 75 && S >= 25 && S < 75"),
        (6, "S >= 75"),
    ]
    cls_exp = "W < 0 ? 0 : "  # no data
    for code, rule in rules:
        cls_exp += f"(({rule}) ? {code} : "
    cls_exp += "7" + ")" * len(rules)  # anything else: Mixed / Other
    # Wetland classes (OW, EFV, WBS) on slopes > max_slope become Mixed / Other.
    cls_img = os.path.join(out_dir, "tropwet_classes.tif")
    img_maths.band_maths(
        cls_img,
        f"((cls = {cls_exp}) >= 1 && cls <= 3 && slope > {max_slope}) ? 7 : cls",
        [img_maths.BandDefn(name, corr_img, idx)
         for idx, name in enumerate(["W", "V", "S"], start=1)]
        + [img_maths.BandDefn("slope", slope_img, 1)],
        datatype=rsgispy.TYPE_8UINT, band_names=["tropwet_class"],
        no_data_val=0, use_pbar=use_pbar,
    )
    # Thematic, with the class names and colours, statistics and pyramids.
    set_thematic_classes(cls_img, TROPWET_CLASSES)

    # Area per class.
    with rasterio.open(cls_img) as src:
        pxl_ha = abs(src.res[0] * src.res[1]) / 10000.0
    counts = count_pxls_of_val(cls_img, list(TROPWET_CLASSES), img_band=1,
                               use_pbar=use_pbar)
    cls_area = pd.DataFrame(
        {"class": [name for name, _ in TROPWET_CLASSES.values()],
         "pixels": counts, "area_ha": counts * pxl_ha,
         "percent": 100 * counts / max(counts.sum(), 1)},
        index=pd.Index(list(TROPWET_CLASSES), name="value"))
    cls_area.to_csv(os.path.join(out_dir, "tropwet_class_areas.csv"))
    print(cls_area.round(1))
    # [end-step-7-classify]
    out["cls_img"] = cls_img
    out["cls_area"] = cls_area

    # [step-8-water-freq]
    # Unmix every date with the composite's endmembers; W >= 75 % = water.
    # Per date: 0 = no observation, 1 = not water, 2 = water.
    obs_defs = []
    for idx, sref_img in enumerate(sref_imgs, start=1):
        date_base = os.path.join(out_dir, f"date{idx}")
        date_refl = prep_refl(sref_img, f"{date_base}_sref.tif")
        date_pct = unmix_pct(date_refl, date_base)
        date_obs = f"{date_base}_water.tif"
        img_maths.band_maths(
            date_obs, "W < 0 ? 0 : (W >= 75 ? 2 : 1)",
            [img_maths.BandDefn("W", date_pct, 1)],
            datatype=rsgispy.TYPE_8UINT, use_pbar=use_pbar,
        )
        obs_defs.append(img_maths.BandDefn(f"d{idx}", date_obs, 1))

    n_obs = " + ".join(f"({defn.band_name} > 0)" for defn in obs_defs)
    n_wat = " + ".join(f"({defn.band_name} == 2)" for defn in obs_defs)
    freq_img = os.path.join(out_dir, "water_frequency.tif")
    img_maths.band_maths(
        freq_img,
        f"nobs = {n_obs}, nwat = {n_wat}, nobs > 0 ? 100 * nwat / nobs : -1",
        obs_defs, band_names=["n_obs", "n_water", "water_freq_pct"],
        no_data_val=-1, use_pbar=use_pbar,
    )
    # [end-step-8-water-freq]
    out["freq_img"] = freq_img

    # [step-9-map]
    import matplotlib.pyplot as plt
    import rsgispy.mapping.elements as map_elements
    import rsgispy.mapping.legend as map_legend
    import rsgispy.mapping.raster as map_raster

    fig, axs = plt.subplots(2, 2, figsize=(9, 7.5))
    # (a) Sand, Vegetation, Water fractions as red, green, blue (0-100 %).
    map_raster.create_raster_img_map(
        axs[0, 0], corr_img, img_bands=[3, 2, 1],
        img_stch=rsgispy.IMG_STRETCH_USER,
        stch_min_max_vals=[{"min": 0.0, "max": 100.0}] * 3,
        show_map_axis=False, scale_bar_loc="lower left",
        title_str="Sand / Vegetation / Water (R/G/B)",
    )
    # (b) Water frequency across the dates.
    map_raster.create_raster_cmap_img_map(
        axs[0, 1], freq_img, img_band=3, cmap_name="Blues",
        norm_img_vals=True, norm_vmin=0.0, norm_vmax=100.0,
        show_map_axis=False, scale_bar_loc="lower left",
        title_str="Water frequency",
        show_colour_bar=True, cbar_label="% of observations",
    )
    # (c) TropWet classes, using the colour table stored in the image.
    map_raster.create_thematic_raster_map(
        axs[1, 0], cls_img, show_map_axis=False, scale_bar_loc="lower left",
        title_str="TropWet classes",
    )
    map_legend.add_categorical_legend(axs[1, 1], classes_info=TROPWET_CLASSES,
                                      overlay=False, title="Class")
    for ax, lbl in zip([axs[0, 0], axs[0, 1], axs[1, 0]], "abc"):
        map_elements.add_axis_label(ax, lbl)
    out["map_file"] = os.path.join(out_dir, "unmix_tropwet_maps.png")
    fig.savefig(out["map_file"], dpi=110, bbox_inches="tight")
    plt.close(fig)
    # [end-step-9-map]
    return out


def main() -> None:
    """Command line interface for the worked example."""
    import matplotlib  # pylint: disable=import-outside-toplevel

    matplotlib.use("Agg")  # write files only; no window needed
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    parser.add_argument("--sref", required=True, nargs="+",
                        help="Surface reflectance images (one per date) or a composite")
    parser.add_argument("--dem", required=True, help="DEM (e.g. SRTM), metres")
    parser.add_argument("--no-data", type=float, default=0,
                        help="No data value of the reflectance images")
    parser.add_argument("--refl-scale", type=float, default=10000.0,
                        help="Scale factor of the reflectance (e.g. 10000)")
    parser.add_argument("--max-slope", type=float, default=2.5,
                        help="Maximum slope (degrees) for wetland classes")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    args = parser.parse_args()
    run_workflow(args.sref, args.dem, args.out_dir, no_data_val=args.no_data,
                 refl_scale=args.refl_scale, max_slope=args.max_slope)


if __name__ == "__main__":
    main()
