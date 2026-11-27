"""
Run the pixel-based classification worked example
(``pixel_classification_workflow.py``) on the data of the rsgispy tutorials
(https://github.com/petebunting/rsgispy-tutorials): the 20 m Sentinel-2 image
of 29 June 2018 (``data/sen2_20180629_t30uvd_orb037_osgb_stdsref_20m.tif``),
the training polygons of
``04_image_classification/03_supervised_pixel_classification.ipynb``
(``data/cls_data/aber_sen2_cls_training.gpkg``) and the 10 m DEM
(``data/dem_10m_osgb.tif``). The printed outputs and the figure in
``docs/guides/pixel_classification_workflow.rst`` were made with this script::

    python pixel_classification_workflow_tutorial.py \\
        --data-dir rsgispy-tutorials/data --out-dir pixel_cls_tutorial
"""

import argparse
import importlib.util
import os
from typing import Any, Dict

import pixel_classification_workflow


def run(data_dir: str, out_dir: str, use_pbar: bool = True) -> Dict[str, Any]:
    """Prepare the tutorial inputs and run the pixel classification example."""
    # [tutorial-data]
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import box
    from rsgispy.imageutils.subsetting import subset_bbox

    os.makedirs(out_dir, exist_ok=True)

    # The DEM covers the west of the Sentinel-2 image: subset the image to a
    # 36 x 30 km area (1800 x 1500 pixels) around Aberystwyth inside the DEM.
    bbox = (255000, 291000, 269000, 299000)  # min x, max x, min y, max y
    s2_img = os.path.join(out_dir, "sen2_20180629_aber_20m.tif")
    subset_bbox(os.path.join(data_dir,
                             "sen2_20180629_t30uvd_orb037_osgb_stdsref_20m.tif"),
                s2_img, *bbox, use_pbar=use_pbar)

    # The classes, layers and colours of the tutorial notebook; Grass_Long
    # and Grass_Short are merged, as only 3 Grass_Long polygons are in the
    # area. Each class's polygons inside the area are written to one file.
    train_gpkg = os.path.join(data_dir, "cls_data", "aber_sen2_cls_training.gpkg")
    classes = {  # class name: (layers, output class ID, colour)
        "Artificial_Surfaces": (["Artificial_Surfaces"], 1, (180, 180, 180)),
        "Bare_Rock_Sand": (["Bare_Rock_Sand"], 2, (131, 250, 252)),
        "Conifer_Forest": (["Conifer_Forest"], 3, (0, 105, 16)),
        "Deciduous_Forest": (["Deciduous_Forest"], 4, (62, 168, 78)),
        "Grass": (["Grass_Long", "Grass_Short"], 5, (247, 176, 32)),
        "NonPhoto_Veg": (["NonPhotosynthetic_Vegetation"], 7, (156, 255, 215)),
        "Scrub": (["Scrub"], 8, (225, 114, 247)),
        "Water": (["Water_Training"], 9, (0, 0, 255)),
        "Bracken": (["Bracken"], 10, (140, 255, 0)),
    }
    area = box(bbox[0], bbox[2], bbox[1], bbox[3])
    train_vecs = {}
    for cls_name, (lyrs, cls_id, colour) in classes.items():
        polys = pd.concat([gpd.read_file(train_gpkg, layer=lyr) for lyr in lyrs],
                          ignore_index=True)
        polys = polys[polys.within(area)]
        vec_file = os.path.join(out_dir, f"train_{cls_name.lower()}.gpkg")
        polys.to_file(vec_file)
        train_vecs[cls_name] = (vec_file, cls_id, colour)

    def has(mod: str) -> bool:
        return importlib.util.find_spec(mod) is not None

    out = pixel_classification_workflow.run_workflow(
        s2_img,
        os.path.join(data_dir, "dem_10m_osgb.tif"),
        train_vecs,
        out_dir,
        refl_gain=1000.0,  # this image is reflectance x 1000
        n_trees=100,
        n_trials=20,
        mmu_pxls=5,  # 0.2 ha with 20 m pixels
        response_class="Conifer_Forest",
        shap_class="Water",
        use_gbdt=has("xgboost"),
        use_boruta=has("shap"),
        use_shap=has("shap"),
        use_salib=has("SALib"),
        use_pbar=use_pbar,
    )
    # [end-tutorial-data]
    out["fig_file"] = make_figures(out)
    return out


def make_figures(out: Dict[str, Any]) -> str:
    """Copy the example's map (``land_cover_map.png``) to the figure of the guide."""
    import shutil  # pylint: disable=import-outside-toplevel

    fig_file = os.path.join(os.path.dirname(out["map_file"]),
                            "pixel_classification_map.png")
    shutil.copyfile(out["map_file"], fig_file)
    _shrink_png(fig_file)
    return fig_file


def _shrink_png(png_file: str, max_bytes: int = 400_000) -> None:
    """Reduce a PNG to 256 colours (with Pillow, if installed) when it is too large."""
    if os.path.getsize(png_file) <= max_bytes or importlib.util.find_spec("PIL") is None:
        return
    from PIL import Image

    with Image.open(png_file) as img:
        small = img.convert("RGB").quantize(colors=256, method=Image.Quantize.MEDIANCUT,
                                            dither=Image.Dither.NONE)
    small.save(png_file, optimize=True)


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
