"""
Run the image co-registration worked example (``image_coregistration.py``)
on two Sentinel-2 images of Aberystwyth from the rsgispy tutorials
(https://github.com/petebunting/rsgispy-tutorials, ``data/``): the 20 m image
of 29 June 2018 (``sen2_20180629_t30uvd_orb037_osgb_stdsref_20m.tif``) is the
reference, and the 10 m image of 27 May 2021 (``sen2_20210527_aber.tif``),
resampled to 20 m with a known shift and rotation added, is the floating
image. The figures and printed outputs in
``docs/guides/image_coregistration.rst`` were made with this script::

    python image_coregistration_tutorial.py --data-dir rsgispy-tutorials/data \\
        --out-dir coreg_tutorial
"""

import argparse
import os
from typing import Any, Dict

import image_coregistration


def run(data_dir: str, out_dir: str, use_pbar: bool = True) -> Dict[str, Any]:
    """Prepare the tutorial image pair and run the co-registration example."""
    # [tutorial-data]
    import geopandas as gpd
    import numpy as np
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import Affine
    from rasterio.warp import reproject
    from rsgispy.imageutils.subsetting import subset_bbox

    os.makedirs(out_dir, exist_ok=True)
    img_2021 = os.path.join(data_dir, "sen2_20210527_aber.tif")

    # Reference: the 2018 image (20 m), cut to the extent of the 2021 image.
    ref_img = os.path.join(out_dir, "s2_20180629_ref.tif")
    with rasterio.open(img_2021) as src:
        bounds = src.bounds
    subset_bbox(
        os.path.join(data_dir, "sen2_20180629_t30uvd_orb037_osgb_stdsref_20m.tif"),
        ref_img, bounds.left, bounds.right, bounds.bottom, bounds.top,
        use_pbar=use_pbar)

    # Floating: the 2021 image (10 m) on the reference grid (20 m, averaged)
    # with a known misregistration: features moved 23 m east and 14 m south,
    # then rotated by 0.3 degrees (clockwise) about the image centre.
    with rasterio.open(ref_img) as ref:
        profile = ref.profile
        centre = ref.xy(ref.height / 2, ref.width / 2)
    true_to_flt = (Affine.rotation(-0.3, pivot=centre)
                   @ Affine.translation(23.0, -14.0))
    flt_img = os.path.join(out_dir, "s2_20210527_flt.tif")
    with rasterio.open(img_2021) as src, rasterio.open(flt_img, "w", **profile) as dst:
        for band in range(1, src.count + 1):
            arr = np.zeros((profile["height"], profile["width"]), dtype=np.uint16)
            reproject(src.read(band), arr, src_transform=true_to_flt @ src.transform,
                      src_crs=src.crs, src_nodata=0, dst_transform=profile["transform"],
                      dst_crs=profile["crs"], dst_nodata=0,
                      resampling=Resampling.average)
            dst.write(arr, band)
        dst.descriptions = src.descriptions

    # GCPs on the floating image, with their true coordinates (x_match,
    # y_match) from the known misregistration, spread over the image.
    true_x = bounds.left + np.array([0.1, 0.5, 0.9, 0.15, 0.85, 0.5]) * (bounds.right - bounds.left)
    true_y = bounds.top - np.array([0.1, 0.15, 0.1, 0.9, 0.85, 0.5]) * (bounds.top - bounds.bottom)
    flt_x, flt_y = true_to_flt @ (true_x, true_y)
    gcp_vec = os.path.join(out_dir, "flt_gcps.gpkg")
    gpd.GeoDataFrame({"x_match": true_x, "y_match": true_y},
                     geometry=gpd.points_from_xy(flt_x, flt_y),
                     crs=profile["crs"]).to_file(gcp_vec)

    out = image_coregistration.run_workflow(
        ref_img,
        flt_img,
        out_dir,
        gcp_vec=gcp_vec,
        ref_band=8,  # NIR (B8A) in both images
        flt_band=8,
        win_size=64,  # pixels of 20 m
        pixel_gap=50,
        search_area=15,
        residual_thres=10.0,  # metres: half a pixel
        use_pbar=use_pbar,
    )
    # [end-tutorial-data]
    return out


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
