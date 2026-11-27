"""
Run the change detection worked example (``change_detection.py``) on the
Landsat images of the Rufiji delta (Tanzania) from the rsgispy tutorials
(https://github.com/remotesensinginfo/rsgispy-tutorials, ``data/chg_data``): a
Landsat 5 TM image of 16 July 1997 and a Landsat 8 OLI image of 8 June 2018
(surface reflectance x 1000, 30 m, UTM zone 37S), the mangrove habitat layer
and the training polygons used by the section 6 notebooks. Mangrove loss
takes the place of forest loss. The figures and printed outputs in
``docs/guides/change_detection.rst`` were made with this script::

    python change_detection_tutorial.py --data-dir rsgispy-tutorials/data \\
        --out-dir chng_tutorial
"""

import argparse
import os
from typing import Any, Dict

import change_detection

# The subset of the delta mapped in the section 6 notebooks
# (MinX, MaxX, MinY, MaxY; EPSG:32637).
SUB_BBOX = (523000, 550000, -877000, -854000)


def run(data_dir: str, out_dir: str, use_pbar: bool = True) -> Dict[str, Any]:
    """Prepare the tutorial inputs and run the change detection worked example."""
    # pylint: disable=import-outside-toplevel
    # [tutorial-data]
    import rsgispy
    import rsgispy.classification.classxgboost as cxgb
    from rsgispy.classification.classdata import (
        ClassificationConfig, ClassInfoObj, ImageBandInfo,
        extract_and_split_class_samples)
    from rsgispy.imagecalc.maths import BandDefn, band_maths
    from rsgispy.imageutils.colourtable import set_thematic_classes
    from rsgispy.imageutils.rasterise import rasterise_vec_lyr

    chg_dir = os.path.join(data_dir, "chg_data")
    ls97_img = os.path.join(
        chg_dir, "LS5TM_19970716_vmsk_mclds_topshad_rad_srefdem_stdsref_subset.tif")
    ls18_img = os.path.join(
        chg_dir, "LS8_20180608_vmsk_mclds_topshad_rad_srefdem_stdsref_subset.tif")
    prep_dir = os.path.join(out_dir, "inputs")
    os.makedirs(prep_dir, exist_ok=True)

    # Analyse only the mangrove habitat: over the whole scene (sea, farmland,
    # forest) IR-MAD and SFA lock on to one land cover (see notebook 05).
    hab_img = os.path.join(prep_dir, "mangrove_habitat.tif")
    rasterise_vec_lyr(os.path.join(chg_dir, "mangrove_habitat.geojson"), None,
                      ls97_img, hab_img, burn_val=1, use_pbar=use_pbar)

    # The land cover map of 1997 (1 = mangroves, 2 = other terrestrial,
    # 3 = water): an XGBoost classification of the habitat in the 1997 image,
    # trained on the polygons of notebook 01 (default XGBoost parameters).
    cls_cfg = ClassificationConfig(
        [ImageBandInfo(ls97_img, name="ls97", bands=list(range(1, 7)),
                       band_names=["Blue", "Green", "Red", "NIR", "SWIR1",
                                   "SWIR2"])],
        {name: ClassInfoObj(out_id=out_id, colour=colour, vec_file=os.path.join(
            chg_dir, "training", f"{name}.geojson"))
         for name, out_id, colour in (("mangroves", 1, (0, 160, 0)),
                                      ("other_terrestrial", 2, (180, 180, 180)),
                                      ("water", 3, (0, 0, 255)))},
    )
    train_df, test_df, _ = extract_and_split_class_samples(
        cls_cfg, test_prop=0.2, valid_prop=0.0, max_cls_smpls=10000,
        rnd_seed=42, use_pbar=use_pbar,
    )
    xgb_mdl, xgb_metrics = cxgb.train_xgboost_multiclass_classifier(
        cls_cfg, train_df, test_data=test_df, rnd_seed=42)
    print(f"1997 classification test accuracy: {xgb_metrics['test_accuracy']:.3f}")
    lc_img = os.path.join(prep_dir, "landcover_1997.tif")
    cxgb.apply_xgboost_multiclass_classifier(
        cls_cfg, xgb_mdl, hab_img, 1, output_img=lc_img, use_pbar=use_pbar)
    set_thematic_classes(lc_img, {1: ("Mangroves", "#00a000"),
                                  2: ("Other terrestrial", "#b4b4b4"),
                                  3: ("Water", "#0000ff")})

    # Both images with the same 6 bands (Blue, Green, Red, NIR, SWIR1, SWIR2;
    # Landsat 8 has an extra coastal band 1), set to no data (0) outside the
    # habitat and where the 1997 map is water: vegetation can only be lost
    # where there was some, and the NDVI of the river channels varies with
    # the sediment load and the tide.
    ls_imgs = {}
    for name, img, bands in (("ls97", ls97_img, range(1, 7)),
                             ("ls18", ls18_img, range(2, 8))):
        ls_imgs[name] = os.path.join(prep_dir, f"{name}_6bands_land.tif")
        band_maths(
            ls_imgs[name],
            ", ".join(f"(lc == 1 || lc == 2) ? b{band} : 0" for band in bands),
            [BandDefn("lc", lc_img, 1)]
            + [BandDefn(f"b{band}", img, band) for band in bands],
            datatype=rsgispy.TYPE_16UINT, no_data_val=0, use_pbar=use_pbar,
        )

    out = change_detection.run_workflow(
        ls_imgs["ls97"],
        ls_imgs["ls18"],
        lc_img,
        out_dir,
        forest_id=1,  # mangroves
        band_map={1: "blue", 2: "green", 3: "red", 4: "nir", 5: "swir1",
                  6: "swir2"},
        cva_bands=(3, 4, 5, 6),  # Red, NIR, SWIR1, SWIR2
        in_gain=1000.0,  # these images are reflectance x 1000
        use_pbar=use_pbar,
    )
    # [end-tutorial-data]
    out["ls_imgs"] = ls_imgs
    out["lc_img"] = lc_img
    out["maps_png"] = _map_figure(out, ls_imgs, out_dir)
    return out


def _map_figure(out: Dict[str, Any], ls_imgs: Dict[str, str], out_dir: str) -> str:
    """The images, land cover and change maps of a subset of the delta."""
    # pylint: disable=import-outside-toplevel
    import matplotlib.pyplot as plt
    from PIL import Image
    from rsgispy.mapping.layout import MapPanel, create_map_panels

    from rsgispy.imageutils.colourtable import set_thematic_classes

    # The same mangrove colour as the land cover map.
    set_thematic_classes(out["forest_loss_img"],
                         {1: ("Mangroves", "#00a000"), 2: ("Mangrove loss", "#e31a1c")})
    out_png = os.path.join(out_dir, "change_maps.png")
    legend = {"show_legend": True, "legend_overlay": True,
              "legend_loc": "lower right", "legend_kwargs": {"fontsize": 9}}
    fig, _ = create_map_panels(
        [
            MapPanel("rgb", ls_imgs["ls97"], bands=[4, 5, 3],
                     title="Landsat 5 TM, July 1997"),
            MapPanel("rgb", ls_imgs["ls18"], bands=[4, 5, 3],
                     title="Landsat 8 OLI, June 2018"),
            MapPanel("thematic", out["lc_img"], title="Land cover 1997 (XGBoost)",
                     map_kw=legend),
            MapPanel("thematic", out["chng_img"], title="dNDVI > Li threshold",
                     map_kw=legend),
            MapPanel("thematic", out["cva_dir_img"], title="CVA + sign of dNDVI",
                     map_kw=legend),
            MapPanel("thematic", out["forest_loss_img"],
                     title="1997 mangroves: 2018 NDVI < Otsu",
                     map_kw=legend),
        ],
        ncols=3,
        bbox=SUB_BBOX,
        figsize=(13, 8.5),
        show_legend=False,  # one legend in each map instead
        out_plot_file=out_png,
        out_dpi=100,
    )
    plt.close(fig)
    # An adaptive 256 colour palette keeps the PNG small.
    with Image.open(out_png) as img:
        img.convert("RGB").quantize(colors=256).save(out_png, optimize=True)
    return out_png


def main() -> None:
    """Command line interface."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    parser.add_argument("--data-dir", required=True,
                        help="The data folder of the rsgispy-tutorials repository")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    args = parser.parse_args()
    run(args.data_dir, args.out_dir)


if __name__ == "__main__":
    main()
