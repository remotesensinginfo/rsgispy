"""
Run the spectral unmixing worked example (``spectral_unmixing.py``) on the
Landsat images of the Rufiji delta, Tanzania, from the rsgispy tutorials
(https://github.com/remotesensinginfo/rsgispy-tutorials, ``data/chg_data/``): a
Landsat 5 TM image from 16 July 1997 and a Landsat 8 OLI image from
8 June 2018, both surface reflectance x 1000 with no data 0. The figures and
printed outputs in ``docs/guides/spectral_unmixing.rst`` were made with this
script::

    python spectral_unmixing_tutorial.py --data-dir rsgispy-tutorials/data \\
        --out-dir unmix_tutorial
"""

import argparse
import os
from typing import Any, Dict

import spectral_unmixing


def run(data_dir: str, out_dir: str, use_pbar: bool = True) -> Dict[str, Any]:
    """Prepare the tutorial inputs and run the spectral unmixing worked example."""
    # [tutorial-data]
    import rsgispy
    from rsgispy.imagecalc.maths import BandDefn, band_maths
    from rsgispy.imageutils.bands import select_img_bands
    from rsgispy.imageutils.subsetting import subset_bbox

    os.makedirs(out_dir, exist_ok=True)
    chg_dir = os.path.join(data_dir, "chg_data")

    # The northern Rufiji delta (the subset used in the change detection
    # notebooks) and the six bands Blue, Green, Red, NIR, SWIR1, SWIR2:
    # bands 1-6 of Landsat 5 TM, bands 2-7 of Landsat 8 OLI.
    sub_bbox = (523000, 550000, -877000, -854000)  # min x, max x, min y, max y
    sref_imgs = []
    for name, bands in [("LS5TM_19970716", [1, 2, 3, 4, 5, 6]),
                        ("LS8_20180608", [2, 3, 4, 5, 6, 7])]:
        sub_img = os.path.join(out_dir, f"{name}_sub.tif")
        subset_bbox(os.path.join(
            chg_dir, f"{name}_vmsk_mclds_topshad_rad_srefdem_stdsref_subset.tif"),
            sub_img, *sub_bbox, calc_stats=False, use_pbar=use_pbar)
        sref_img = os.path.join(out_dir, f"{name}_sref.tif")
        select_img_bands(sub_img, sref_img, bands, calc_stats=False,
                         use_pbar=use_pbar)
        sref_imgs.append(sref_img)

    # The tutorial data has no DEM of the delta. A flat DEM (0 m everywhere)
    # switches the slope mask off; the delta is flat anyway.
    dem_img = os.path.join(out_dir, "flat_dem.tif")
    band_maths(dem_img, "0 * b1", [BandDefn("b1", sref_imgs[0], 1)],
               datatype=rsgispy.TYPE_32FLOAT, use_pbar=use_pbar)

    out = spectral_unmixing.run_workflow(
        sref_imgs,
        dem_img,
        out_dir,
        no_data_val=0,
        refl_scale=1000.0,  # these images are reflectance x 1000
        use_pbar=use_pbar,
    )
    # [end-tutorial-data]
    return out


def main() -> None:
    """Command line interface."""
    import matplotlib  # pylint: disable=import-outside-toplevel

    matplotlib.use("Agg")  # write files only; no window needed
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    parser.add_argument("--data-dir", required=True,
                        help="The data folder of the rsgispy-tutorials repository")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    args = parser.parse_args()
    run(args.data_dir, args.out_dir)


if __name__ == "__main__":
    main()
