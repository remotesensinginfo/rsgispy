"""
Worked example: preparing Sentinel-1 (SAR) backscatter for analysis with
rsgispy (speckle filtering, dB conversion, SAR indices, texture and a feature
stack for classification).

This script is included step by step in the documentation guide
``docs/guides/sar_processing.rst`` and is run end-to-end by the test suite
(``tests/test_sar_processing_example.py``) on small synthetic data, so the
documented code is always tested.

Run it on your own data with, e.g.::

    python sar_processing.py --s1 s1_sigma0_vv_vh.tif --in-units power \\
        --train train_areas.gpkg --homog homog_areas.gpkg --out-dir outputs
"""

import argparse
import os
from typing import Any, Dict, Optional, Sequence


def run_workflow(
    s1_img: str,
    train_vec: str,
    homog_vec: str,
    out_dir: str,
    in_units: str = "power",
    pol_names: Sequence[str] = ("VV", "VH"),
    in_no_data_val: Optional[float] = None,
    n_looks: float = 4.4,
    filter_size: int = 7,
    glcm_size: int = 7,
    db_range: Sequence[float] = (-30.0, 5.0),
    fig_file: Optional[str] = None,
    use_pbar: bool = True,
) -> Dict[str, Any]:
    """
    Run the worked example. ``s1_img`` is a dual-polarisation backscatter
    image (``pol_names`` gives the polarisation of each band) in linear power
    (``in_units='power'``, no data value 0) or decibels (``in_units='dB'``).
    ``train_vec`` holds training polygons with ``cls_id`` and ``cls_name``
    columns and ``homog_vec`` polygons of homogeneous areas (``area_id``)
    used to measure the speckle reduction. If ``fig_file`` is given, a PNG
    comparing the raw and filtered VV backscatter is written to it.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-statements,import-outside-toplevel
    os.makedirs(out_dir, exist_ok=True)
    out: Dict[str, Any] = {}

    # [step-1-linear]
    import rsgispy
    from rsgispy.imagecalc.maths import BandDefn, band_maths

    pols = [pol.lower() for pol in pol_names]  # band variable names: vv, vh
    if in_units.lower() == "db":
        # dB -> linear power: 10^(dB / 10); no data (here -9999) -> 0.
        nd_db = -9999.0 if in_no_data_val is None else in_no_data_val
        lin_img = os.path.join(out_dir, "s1_linear.tif")
        band_maths(
            lin_img,
            ", ".join(f"{p} != {nd_db} ? 10^({p} / 10) : 0" for p in pols),
            [BandDefn(p, s1_img, idx) for idx, p in enumerate(pols, start=1)],
            band_names=list(pol_names),
            datatype=rsgispy.TYPE_32FLOAT,
            no_data_val=0.0,
            calc_stats=True,
            use_pbar=use_pbar,
        )
    else:
        lin_img = s1_img  # already linear power (sigma0 / gamma0)
    # [end-step-1-linear]
    out["lin_img"] = lin_img

    # [step-2-speckle]
    import rsgispy.imagefilter.sar as sar_filters

    filt_imgs = {
        "lee": os.path.join(out_dir, "s1_lee.tif"),
        "refined_lee": os.path.join(out_dir, "s1_refined_lee.tif"),
        "frost": os.path.join(out_dir, "s1_frost.tif"),
        "gamma_map": os.path.join(out_dir, "s1_gamma_map.tif"),
    }
    sar_filters.apply_lee_filter(
        lin_img, filt_imgs["lee"], filter_size=filter_size, n_looks=n_looks,
        use_pbar=use_pbar,
    )
    sar_filters.apply_refined_lee_filter(
        lin_img, filt_imgs["refined_lee"], filter_size=filter_size,
        n_looks=n_looks, use_pbar=use_pbar,
    )
    sar_filters.apply_frost_filter(
        lin_img, filt_imgs["frost"], filter_size=filter_size, damping=1.0,
        use_pbar=use_pbar,
    )
    sar_filters.apply_gamma_map_filter(
        lin_img, filt_imgs["gamma_map"], filter_size=filter_size,
        n_looks=n_looks, use_pbar=use_pbar,
    )
    # [end-step-2-speckle]

    # [step-3-check]
    import pandas as pd
    from rsgispy.imageutils.rasterise import rasterise_vec_lyr
    from rsgispy.ml.stack import BandSpec
    from rsgispy.zonalstats.labels import calc_label_zonal_stats

    homog_img = os.path.join(out_dir, "homog_areas.tif")
    rasterise_vec_lyr(
        homog_vec, None, lin_img, homog_img, att_column="area_id",
        datatype=rsgispy.TYPE_16UINT, use_pbar=use_pbar,
    )
    rows = []
    for name, img in {"raw": lin_img, **filt_imgs}.items():
        stats = calc_label_zonal_stats(
            homog_img,
            BandSpec(img, bands=(1,), band_names=(pol_names[0],)),
            stats=["mean", "stddev"],
            id_col="area_id",
            use_pbar=use_pbar,
        )
        mean = stats[f"{pol_names[0]}_mean"]
        rows.append(
            pd.DataFrame(
                {
                    "filter": name,
                    "area_id": stats["area_id"],
                    "mean": mean,
                    # Equivalent number of looks: higher = less speckle.
                    "enl": (mean / stats[f"{pol_names[0]}_stddev"]) ** 2,
                }
            )
        )
    enl_df = pd.concat(rows, ignore_index=True)
    print(enl_df.round(4).to_string(index=False))
    # [end-step-3-check]
    out["enl"] = enl_df

    # [step-4-db]
    filt_img = filt_imgs["refined_lee"]  # the filter chosen from step 3
    db_img = os.path.join(out_dir, "s1_refined_lee_db.tif")
    band_maths(
        db_img,
        ", ".join(f"{p} > 0 ? 10 * log10({p}) : -9999" for p in pols),
        [BandDefn(p, filt_img, idx) for idx, p in enumerate(pols, start=1)],
        band_names=[f"{pol}_dB" for pol in pol_names],
        datatype=rsgispy.TYPE_32FLOAT,
        no_data_val=-9999.0,
        calc_stats=True,
        use_pbar=use_pbar,
    )
    # [end-step-4-db]
    out["db_img"] = db_img

    # [step-5-indices]
    from rsgispy.imagecalc.calcindices import (
        calc_sar_indices,
        get_sar_index_equation,
    )

    idx_names = ["RVI_VV_VH", "CR_VH_VV_dB", "DpRVIc_VV_VH", "NDPI_VV_VH"]
    for idx_name in idx_names:
        print(idx_name, "=", get_sar_index_equation(idx_name))
    idx_img = os.path.join(out_dir, "s1_indices.tif")
    calc_sar_indices(
        filt_img,
        idx_img,
        band_names=dict(enumerate(pol_names, start=1)),
        indices=idx_names,
        in_units="power",  # the filtered image is linear power
        no_data_val=0.0,
        out_no_data_val=-9999.0,
        use_pbar=use_pbar,
    )
    # [end-step-5-indices]
    out["idx_img"] = idx_img

    # [step-6-texture]
    import rsgispy.imagefilter.texture as tex

    glcm_img = os.path.join(out_dir, "s1_glcm.tif")
    tex.apply_glcm_filter(
        db_img,
        glcm_img,
        properties=[tex.GLCM_CONTRAST, tex.GLCM_HOMOGENEITY, tex.GLCM_ENTROPY],
        filter_size=glcm_size,
        n_levels=32,
        min_val=db_range[0],  # fixed dB range, so the grey levels mean
        max_val=db_range[1],  # the same thing in every scene
        no_data_val=-9999.0,
        use_pbar=use_pbar,
    )
    # [end-step-6-texture]
    out["glcm_img"] = glcm_img

    # [step-7-chain]
    from rsgispy.imagefilter.chained import FilterStep, apply_filter_chain

    chain_img = os.path.join(out_dir, "s1_lee_db_chain.tif")
    apply_filter_chain(
        lin_img,
        chain_img,
        steps=[
            FilterStep("lee", filter_size=filter_size,
                       params={"n_looks": n_looks}),
            FilterStep("band_math",
                       params={"expr": "x > 0 ? 10 * log10(x) : -9999"}),
        ],
        no_data_val=0.0,
        use_pbar=use_pbar,
    )
    # [end-step-7-chain]
    out["chain_img"] = chain_img

    # [step-8-features]
    from rsgispy.imageutils.metadata import get_band_names
    from rsgispy.imageutils.vrt import create_stack_images_vrt

    feat_imgs = [db_img, idx_img, glcm_img]
    feat_names = [name for img in feat_imgs for name in get_band_names(img)]
    feat_vrt = os.path.join(out_dir, "s1_features.vrt")
    create_stack_images_vrt(feat_imgs, feat_vrt, band_names=feat_names)

    cls_img = os.path.join(out_dir, "train_areas.tif")
    rasterise_vec_lyr(
        train_vec, None, lin_img, cls_img, att_column="cls_id",
        datatype=rsgispy.TYPE_8UINT, use_pbar=use_pbar,
    )
    cls_stats = calc_label_zonal_stats(
        cls_img,
        BandSpec(feat_vrt, band_names=tuple(feat_names), no_data_val=-9999.0),
        stats=["mean", "stddev"],
        id_col="cls_id",
        use_pbar=use_pbar,
    )
    print(feat_names)
    print(cls_stats.round(3).T.to_string())
    # [end-step-8-features]
    out["feat_vrt"] = feat_vrt
    out["feat_names"] = feat_names
    out["cls_stats"] = cls_stats

    if fig_file is not None:
        # [step-9-figure]
        import matplotlib

        matplotlib.use("Agg")  # no display needed
        import matplotlib.pyplot as plt
        from rsgispy.mapping.raster import create_raster_cmap_img_map

        fig, axs = plt.subplots(1, 2, figsize=(8, 3.4))
        for ax, (img, title) in zip(
            axs, [(lin_img, "VV (raw)"), (filt_img, "VV (refined Lee)")]
        ):
            img = create_raster_cmap_img_map(
                ax, img, 1, title_str=title, cmap_name="gray",
                norm_img_vals=True,
                use_log_norm=True,  # a log scale shows the data as dB
                norm_vmin=10 ** (-25 / 10),  # -25 dB
                norm_vmax=1.0,  # 0 dB
                show_map_axis=False,
            )
        fig.colorbar(img, ax=axs,  # one colour bar for both panels
                     label="sigma0 (linear, log scale)", shrink=0.8)
        fig.savefig(fig_file, dpi=100, bbox_inches="tight")
        plt.close(fig)
        # [end-step-9-figure]
        out["fig_file"] = fig_file
    return out


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Command line interface for running the example on your own data."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    parser.add_argument("--s1", required=True,
                        help="Dual-pol backscatter image (e.g. VV, VH).")
    parser.add_argument("--in-units", default="power", choices=["power", "dB"],
                        help="Units of the input image (Default: power).")
    parser.add_argument("--pols", nargs="+", default=["VV", "VH"],
                        help="Polarisation of each band (Default: VV VH).")
    parser.add_argument("--no-data", type=float, default=None,
                        help="No data value of a dB input (Default: -9999).")
    parser.add_argument("--train", required=True,
                        help="Training polygons (cls_id, cls_name columns).")
    parser.add_argument("--homog", required=True,
                        help="Homogeneous area polygons (area_id column).")
    parser.add_argument("--n-looks", type=float, default=4.4,
                        help="Equivalent number of looks (Default: 4.4).")
    parser.add_argument("--filter-size", type=int, default=7,
                        help="Speckle filter window size (Default: 7).")
    parser.add_argument("--glcm-size", type=int, default=7,
                        help="GLCM texture window size (Default: 7).")
    parser.add_argument("--figure", default=None,
                        help="Optional PNG comparing raw and filtered VV.")
    parser.add_argument("--out-dir", required=True, help="Output directory.")
    args = parser.parse_args(argv)
    run_workflow(
        args.s1,
        args.train,
        args.homog,
        args.out_dir,
        in_units=args.in_units,
        pol_names=args.pols,
        in_no_data_val=args.no_data,
        n_looks=args.n_looks,
        filter_size=args.filter_size,
        glcm_size=args.glcm_size,
        fig_file=args.figure,
    )


if __name__ == "__main__":
    main()
