"""
Worked example: multi-criteria analysis with band maths, mapping where
mangrove restoration is biophysically feasible and most likely to succeed.

This script is included step by step in the documentation guide
``docs/guides/multi_criteria_analysis.rst`` and is run end-to-end by the test
suite (``tests/test_multi_criteria_analysis_example.py``) on small synthetic
data, so the documented code is always tested.

The criteria, thresholds and weights are ILLUSTRATIVE ONLY. A real study must
justify each of them from the literature and from local knowledge.

Run it on your own data with, e.g.::

    python multi_criteria_analysis.py --s2 s2_sref_10m.tif --dem dem_msl.tif \\
        --mng-now gmw_2020.tif --mng-hist gmw_1996.tif --ponds ponds.gpkg \\
        --settlements settlements.gpkg --protected wdpa.gpkg \\
        --admin admin_units.gpkg --out-dir outputs

Add ``--osm-settlements`` to download the settlements from OpenStreetMap
instead (needs the ``osmnx`` package and internet access).
"""

import argparse
import os
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

DEF_S2_BANDS = {1: "blue", 2: "green", 3: "red", 7: "nir", 9: "swir1", 10: "swir2"}


def download_osm_settlements(ref_img: str, out_vec_file: str) -> str:
    """
    Optional: download residential areas from OpenStreetMap for the extent of
    ``ref_img`` and save them in the image's projection (needs ``osmnx``).
    """
    # pylint: disable=import-outside-toplevel
    # [step-0-osm]
    import rsgispy
    import rsgispy.dataaccess.openstreetmap as osm
    import rsgispy.imageutils.metadata as img_meta

    img_epsg = img_meta.get_epsg_proj_from_img(ref_img)
    osm.get_osm_to_file(
        out_vec_file,
        "settlements",
        bbox=img_meta.get_img_bbox(ref_img),  # (MinX, MaxX, MinY, MaxY)
        osm_tags={"landuse": "residential"},
        rsgis_geom_type=rsgispy.GEOM_POLY,
        bbox_epsg=img_epsg,
        out_vec_epsg=img_epsg,  # same projection as the image grid
    )
    # [end-step-0-osm]
    return out_vec_file


def run_workflow(
    s2_img: str,
    dem_img: str,
    mng_now_img: str,
    mng_hist_img: str,
    ponds_vec: str,
    settle_vec: str,
    protect_vec: str,
    admin_vec: str,
    out_dir: str,
    s2_bands: Optional[Mapping[int, str]] = None,
    dem_no_data: float = -9999.0,
    elev_frame: Tuple[float, float, float, float] = (0.0, 0.3, 1.0, 1.5),
    class_breaks: Tuple[float, float] = (0.45, 0.65),
    min_area_ha: float = 0.5,
    use_pbar: bool = True,
) -> Dict[str, Any]:
    """
    Run the worked example.

    ``s2_img`` is a Sentinel-2 surface reflectance stack (reflectance x 10000,
    no data 0) whose bands are named by ``s2_bands``; ``dem_img`` is elevation
    in metres relative to mean sea level; ``mng_now_img`` / ``mng_hist_img``
    are current and baseline mangrove extent masks (1 = mangrove).
    ``ponds_vec`` holds aquaculture ponds (``yrs_aband``: years since they
    were abandoned, 0 = active), ``settle_vec`` built-up areas,
    ``protect_vec`` protected areas (WDPA ``IUCN_CAT``) and ``admin_vec`` the
    reporting units (``name``). ``elev_frame`` gives the elevations (m) where
    the elevation membership starts to rise, reaches 1, starts to fall and
    reaches 0; ``class_breaks`` the score thresholds between low/medium and
    medium/high potential.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-statements,import-outside-toplevel
    os.makedirs(out_dir, exist_ok=True)
    s2_bands = dict(DEF_S2_BANDS if s2_bands is None else s2_bands)
    out: Dict[str, Any] = {}

    def opath(name: str) -> str:
        return os.path.join(out_dir, name)

    # [step-1-inputs]
    import rsgispy
    import rsgispy.imagecalc.calcindices
    import rsgispy.imageutils.warp
    import rsgispy.elevation.terrain

    # The Sentinel-2 image defines the analysis grid (10 m). Continuous data
    # (elevation) is resampled bilinearly; categorical data (the mangrove
    # masks) must use nearest neighbour so no new values are invented.
    rsgispy.imageutils.warp.resample_img_to_match(
        s2_img, dem_img, opath("dem_10m.tif"),
        interp_method=rsgispy.INTERP_BILINEAR,
        datatype=rsgispy.TYPE_32FLOAT, no_data_val=dem_no_data, use_pbar=use_pbar,
    )
    for name, in_img in (("mng_now", mng_now_img), ("mng_hist", mng_hist_img)):
        rsgispy.imageutils.warp.resample_img_to_match(
            s2_img, in_img, opath(f"{name}_10m.tif"),
            interp_method=rsgispy.INTERP_NEAREST_NEIGHBOUR,
            datatype=rsgispy.TYPE_8UINT, use_pbar=use_pbar,
        )

    # Mangrove and water indices (input reflectance x 10000, no data 0).
    rsgispy.imagecalc.calcindices.calc_indices(
        s2_img, opath("s2_indices.tif"), band_names=s2_bands,
        indices=["MVI", "CMRI", "MNDWI"], in_gain=10000.0, no_data_val=0,
        use_pbar=use_pbar,
    )
    rsgispy.elevation.terrain.slope(
        opath("dem_10m.tif"), opath("slope_10m.tif"), unit="degrees",
        no_data_val=dem_no_data, use_pbar=use_pbar,
    )
    # [end-step-1-inputs]

    # [step-2-constraints]
    import rsgispy.imageutils.rasterise
    from rsgispy.imagecalc.maths import BandDefn, band_maths

    # Vector layers onto the same grid: settlements as a 0/1 mask, ponds as a
    # code from the years since abandonment (1 = active, 2 = abandoned < 5
    # years, 3 = abandoned >= 5 years, 0 = no pond).
    rsgispy.imageutils.rasterise.rasterise_vec_lyr(
        settle_vec, None, s2_img, opath("settlements.tif"), burn_val=1,
        use_pbar=use_pbar,
    )
    rsgispy.imageutils.rasterise.create_vec_range_lut_score_img(
        ponds_vec, None, "yrs_aband", s2_img, opath("ponds.tif"),
        scrs_lut={1: (0.0, 1.0), 2: (1.0, 5.0), 3: (5.0, 1000.0)},
        use_pbar=use_pbar,
    )

    blue_band = next(idx for idx, name in s2_bands.items() if name == "blue")
    lyr_defs = [
        BandDefn("blue", s2_img, blue_band),
        BandDefn("elev", opath("dem_10m.tif"), 1),
        BandDefn("gmw", opath("mng_now_10m.tif"), 1),
        BandDefn("mvi", opath("s2_indices.tif"), 1),
        BandDefn("cmri", opath("s2_indices.tif"), 2),
        BandDefn("mndwi", opath("s2_indices.tif"), 3),
        BandDefn("pond", opath("ponds.tif"), 1),
    ]
    # One pass, three output bands. `valid` is defined in the first
    # sub-expression and reused by the next two.
    band_maths(
        opath("layers.tif"),
        f"""
        valid = (blue > 0 && elev != {dem_no_data}) ? 1 : 0,
        valid == 1 && (gmw == 1 || (mvi >= 4.5 && cmri >= 1.0)) ? 1 : 0,
        valid == 1 && mndwi > 0 && pond != 1 ? 1 : 0
        """,
        lyr_defs,
        band_names=["valid", "mangrove", "water"],
        datatype=rsgispy.TYPE_8UINT,
        use_pbar=use_pbar,
    )

    # The constraint mask: 1 where restoration is feasible, 0 where excluded.
    z_low, _, _, z_high = elev_frame
    cnst_defs = [
        BandDefn("valid", opath("layers.tif"), 1),
        BandDefn("mng", opath("layers.tif"), 2),
        BandDefn("elev", opath("dem_10m.tif"), 1),
        BandDefn("settle", opath("settlements.tif"), 1),
        BandDefn("pond", opath("ponds.tif"), 1),
    ]
    band_maths(
        opath("feasible.tif"),
        f"valid == 1 && mng == 0 && settle == 0 && pond != 1"
        f" && elev >= {z_low} && elev <= {z_high} ? 1 : 0",
        cnst_defs,
        band_names=["feasible"],
        datatype=rsgispy.TYPE_8UINT,
        use_pbar=use_pbar,
    )
    # [end-step-2-constraints]

    # [step-3-factors]
    import rsgispy.imagecalc.distance
    import rsgispy.imagecalc.rescale

    # Distance (m) to existing mangroves (layers band 2) and to open water
    # (band 3), capped at 5 km (pixels further away are set to 5000).
    for name, band in (("mangrove", 2), ("water", 3)):
        rsgispy.imagecalc.distance.calc_dist_to_img_vals(
            opath("layers.tif"), opath(f"dist_{name}.tif"), pxl_vals=1,
            img_band=band, max_dist=5000.0, use_pbar=use_pbar,
        )

    # A simple linear membership with rescale: 0 m -> 1, >= 1000 m -> 0.
    # Pixels beyond max_dist (the distance image's no data value) are
    # written as no data (-9999, the default out_no_data).
    rsgispy.imagecalc.rescale.rescale_img_pxl_vals(
        opath("dist_water.tif"), opath("f_water.tif"),
        [rsgispy.imagecalc.rescale.ImageBandRescale(
            band=1, in_min=0.0, in_max=1000.0, out_min=1.0, out_max=0.0)],
        trim_to_limits=True,
        use_pbar=use_pbar,
    )

    # A categorical score from a vector attribute (percent; 0 = unprotected).
    rsgispy.imageutils.rasterise.create_vec_lst_lut_score_img(
        protect_vec, None, "IUCN_CAT", s2_img, opath("protection.tif"),
        scrs_lut=[("Ia", 100), ("Ib", 100), ("II", 100), ("III", 80),
                  ("IV", 80), ("V", 60), ("VI", 60), ("Not Reported", 50),
                  ("Not Applicable", 50), ("Not Assigned", 50)],
        use_pbar=use_pbar,
    )

    # Fuzzy membership (0-1) for each factor, all in one band_maths pass.
    z_0, z_1, z_2, z_3 = elev_frame
    fac_defs = [
        BandDefn("elev", opath("dem_10m.tif"), 1),
        BandDefn("slope", opath("slope_10m.tif"), 1),
        BandDefn("dmng", opath("dist_mangrove.tif"), 1),
        BandDefn("fwater", opath("f_water.tif"), 1),
        BandDefn("mng", opath("layers.tif"), 2),
        BandDefn("mng_hist", opath("mng_hist_10m.tif"), 1),
        BandDefn("pond", opath("ponds.tif"), 1),
        BandDefn("prot", opath("protection.tif"), 1),
    ]
    factor_names = ["f_elev", "f_loss", "f_dmng", "f_water", "f_pond", "f_prot",
                    "f_slope"]
    factor_exps = [
        # trapezoid: rises z_0 -> z_1, flat to z_2, falls to z_3
        f"max(0, min((elev - {z_0}) / {z_1 - z_0}, 1, ({z_3} - elev) / {z_3 - z_2}))",
        # former mangrove that has been lost since the baseline year
        "mng_hist == 1 && mng == 0 ? 1 : 0.4",
        # decreasing sigmoid: 0.5 at 500 m from mangroves (propagule supply)
        "1 / (1 + exp((dmng - 500) / 100))",
        # already 0-1 (linear, from rescale_img_pxl_vals); no data (further
        # than max_dist from water) -> 0
        "fwater < 0 ? 0 : fwater",
        # categorical: abandoned >= 5 yr, < 5 yr, no pond
        "pond == 3 ? 1 : (pond == 2 ? 0.8 : 0.5)",
        "prot == 0 ? 0.3 : prot / 100",
        # linear decreasing: 1 on flat ground, 0 at >= 5 degrees; the slope
        # no data value (-9999) must not become a high membership
        "slope < 0 ? 0 : max(0, 1 - slope / 5)",
    ]
    band_maths(
        opath("factors.tif"), ",\n".join(factor_exps), fac_defs,
        band_names=factor_names, datatype=rsgispy.TYPE_32FLOAT,
        use_pbar=use_pbar,
    )
    # [end-step-3-factors]
    out["factors_img"] = opath("factors.tif")

    # [step-4-weights]
    import numpy as np

    # Pairwise comparison (Saaty scale): pairwise[i, j] = how much more
    # important factor i is than factor j (1 = equal ... 9 = extreme).
    pairwise = np.array([
        # elev loss dmng water pond  prot slope
        [1,    2,   3,   3,    4,    5,   5],  # f_elev
        [1/2,  1,   2,   2,    3,    4,   4],  # f_loss
        [1/3,  1/2, 1,   1,    2,    3,   3],  # f_dmng
        [1/3,  1/2, 1,   1,    2,    3,   3],  # f_water
        [1/4,  1/3, 1/2, 1/2,  1,    2,   2],  # f_pond
        [1/5,  1/4, 1/3, 1/3,  1/2,  1,   1],  # f_prot
        [1/5,  1/4, 1/3, 1/3,  1/2,  1,   1],  # f_slope
    ])
    eig_vals, eig_vecs = np.linalg.eig(pairwise)
    i_max = int(np.argmax(eig_vals.real))
    ahp_w = np.abs(eig_vecs[:, i_max].real)
    ahp_w = ahp_w / ahp_w.sum()
    n_fac = len(factor_names)
    cons_idx = (eig_vals[i_max].real - n_fac) / (n_fac - 1)
    cons_ratio = cons_idx / 1.32  # Saaty's random index for n = 7
    weights = {name: round(float(w), 4) for name, w in zip(factor_names, ahp_w)}
    print(weights, f"CR = {cons_ratio:.3f}")  # CR < 0.1 is acceptable
    # [end-step-4-weights]
    out["weights"] = weights
    out["cons_ratio"] = float(cons_ratio)

    # [step-5-combine]
    import rsgispy.imageutils.colourtable

    def calc_potential(wgts: Mapping[str, float], out_img: str) -> None:
        """Weighted linear combination x constraint -> class + score."""
        comb_defs = [
            BandDefn(name, opath("factors.tif"), idx)
            for idx, name in enumerate(factor_names, start=1)
        ]
        comb_defs += [BandDefn("feasible", opath("feasible.tif"), 1),
                      BandDefn("valid", opath("layers.tif"), 1)]
        wlc = " + ".join(f"{wgts[name]} * {name}" for name in factor_names)
        lo_brk, hi_brk = class_breaks
        band_maths(
            out_img,
            # band 1: 0 no data, 1 excluded, 2 low, 3 medium, 4 high
            f"(s = feasible * ({wlc})) >= {hi_brk} ? 4 :"
            f" (s >= {lo_brk} ? 3 : (feasible == 1 ? 2 : valid)),"
            # band 2: score x 100 (0 where excluded, 255 = no data)
            " valid == 1 ? rint(100 * s) : 255",
            comb_defs,
            band_names=["potential", "score"],
            datatype=rsgispy.TYPE_8UINT,
            no_data_val=255,
            use_pbar=use_pbar,
        )

    potential_img = opath("restoration_potential.tif")
    calc_potential(weights, potential_img)
    cls_colours = {1: "#d9d9d9", 2: "#fee391", 3: "#78c679", 4: "#006837"}
    rsgispy.imageutils.colourtable.define_colour_table(potential_img, cls_colours)
    # [end-step-5-combine]
    out["potential_img"] = potential_img

    # [step-6-sensitivity]
    import pandas as pd
    from rsgispy.imagecalc.maths import calc_prop_true_exp

    weight_sets = {"equal": {name: round(1 / n_fac, 4) for name in factor_names}}
    for name in factor_names:  # change one weight by +/- 25 % and renormalise
        for change in (0.75, 1.25):
            alt = dict(weights, **{name: weights[name] * change})
            weight_sets[f"{name} x{change}"] = {
                k: round(v / sum(alt.values()), 4) for k, v in alt.items()
            }

    sens_rows = []
    alt_img = opath("sensitivity_tmp.tif")
    for set_name, wgts in weight_sets.items():
        calc_potential(wgts, alt_img)
        cmp_defs = [BandDefn("base", potential_img, 1), BandDefn("alt", alt_img, 1)]
        sens_rows.append({
            "weights": set_name,
            # proportion of the feasible pixels that change class
            "changed": calc_prop_true_exp("base != alt ? 1 : 0", cmp_defs,
                                          in_vld_img=opath("feasible.tif"),
                                          use_pbar=False),
            "base_high": calc_prop_true_exp("base == 4 ? 1 : 0", cmp_defs,
                                            in_vld_img=opath("feasible.tif"),
                                            use_pbar=False),
            "alt_high": calc_prop_true_exp("alt == 4 ? 1 : 0", cmp_defs,
                                           in_vld_img=opath("feasible.tif"),
                                           use_pbar=False),
        })
    os.remove(alt_img)
    sens_df = pd.DataFrame(sens_rows).sort_values("changed", ascending=False)
    print(sens_df.round(3).to_string(index=False))
    # [end-step-6-sensitivity]
    out["sensitivity"] = sens_df

    # [step-7-summaries]
    import rsgispy.imageutils.metadata
    import rsgispy.vector.create
    import rsgispy.vector.geoms
    import rsgispy.zonalstats.polygon as zonal

    x_res, y_res = rsgispy.imageutils.metadata.get_img_res(
        potential_img, abs_vals=True)
    pxl_ha = (x_res * y_res) / 10000.0
    classes = {1: "excluded", 2: "low", 3: "medium", 4: "high"}
    admin_gdf = zonal.calc_zonal_band_pxl_counts(
        admin_vec, potential_img, 1, classes, total_field="n_pxls",
        use_pbar=use_pbar,
    )
    for cls_name in classes.values():
        admin_gdf[f"{cls_name}_ha"] = admin_gdf[cls_name] * pxl_ha
    # mean / max score (x 100) of the pixels that are not excluded
    admin_gdf = zonal.calc_zonal_band_stats(
        admin_gdf, potential_img, 2, min_thres=1, max_thres=100,
        out_no_data_val=-1, mean_field="score_mean", max_field="score_max",
        use_pbar=use_pbar,
    )
    admin_gdf.to_file(opath("admin_summary.gpkg"), driver="GPKG")
    print(admin_gdf.drop(columns="geometry").round(2).to_string())

    # High potential areas as polygons, keeping those >= min_area_ha.
    cls_polys = rsgispy.vector.create.polygonise_raster_to_vec_lyr(
        potential_img, img_band=1, use_8_conn=True,
    )
    high_polys = rsgispy.vector.geoms.rm_polys_area(
        cls_polys[cls_polys["PXLVAL"] == 4], None, min_area_ha * 10000.0,
        less_than=False,  # keep polygons larger than the threshold
    )
    high_polys["area_ha"] = high_polys.area / 10000.0
    high_polys.to_file(opath("high_potential.gpkg"), driver="GPKG")
    # [end-step-7-summaries]
    out["admin_gdf"] = admin_gdf
    out["high_polys"] = high_polys

    # [step-8-maps]
    import matplotlib.pyplot as plt
    import rsgispy.mapping.raster as map_raster

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    map_raster.create_raster_cmap_img_map(
        ax1, potential_img, 2, title_str="Restoration score (x 100)",
        cmap_name="YlGn", norm_img_vals=True, norm_vmin=1, norm_vmax=100,
        vals_under_white=True,  # excluded pixels (score 0) in white
        show_map_axis=False,
        show_colour_bar=True, cbar_label="Score (x 100)",
    )
    map_raster.create_thematic_raster_map(
        ax2, potential_img, title_str="Restoration potential",
        show_map_axis=False,
        show_legend=True, legend_overlay=True, legend_loc="lower right",
        classes_info={1: "Excluded", 2: "Low", 3: "Medium", 4: "High"},
    )
    fig.savefig(opath("restoration_potential.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    # [end-step-8-maps]
    out["map_png"] = opath("restoration_potential.png")
    return out


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Command line interface for the worked example."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    parser.add_argument("--s2", required=True, help="Sentinel-2 SREF stack")
    parser.add_argument("--dem", required=True,
                        help="DEM (m relative to mean sea level)")
    parser.add_argument("--mng-now", required=True, help="Current mangrove mask")
    parser.add_argument("--mng-hist", required=True,
                        help="Baseline (historical) mangrove mask")
    parser.add_argument("--ponds", required=True,
                        help="Aquaculture ponds (yrs_aband column)")
    parser.add_argument("--settlements", default=None,
                        help="Settlement polygons")
    parser.add_argument("--osm-settlements", action="store_true",
                        help="Download the settlements from OpenStreetMap")
    parser.add_argument("--protected", required=True,
                        help="Protected areas (IUCN_CAT column)")
    parser.add_argument("--admin", required=True,
                        help="Reporting units (name column)")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    args = parser.parse_args(argv)
    settle_vec = args.settlements
    if args.osm_settlements:
        os.makedirs(args.out_dir, exist_ok=True)
        settle_vec = download_osm_settlements(
            args.s2, os.path.join(args.out_dir, "osm_settlements.gpkg"))
    if settle_vec is None:
        parser.error("give --settlements or --osm-settlements")
    run_workflow(args.s2, args.dem, args.mng_now, args.mng_hist, args.ponds,
                 settle_vec, args.protected, args.admin, args.out_dir)


if __name__ == "__main__":
    main()
