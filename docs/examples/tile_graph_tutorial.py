"""
Run the tile-graph worked example (``tile_graph.py``) on the data of the
rsgispy tutorials (https://github.com/remotesensinginfo/rsgispy-tutorials): the
20 m Sentinel-2 image of 29 June 2018
(``data/sen2_20180629_t30uvd_orb037_osgb_stdsref_20m.tif``), the training
polygons of ``09_tile_graph_workflows/01_indices_classification_filtering.ipynb``
(``data/cls_data/aber_sen2_cls_training.gpkg``) and the 10 m DEM
(``data/dem_10m_osgb.tif``). The printed outputs and the figure in
``docs/guides/tile_graph.rst`` were made with this script::

    python tile_graph_tutorial.py --data-dir rsgispy-tutorials/data \\
        --out-dir tile_graph_tutorial
"""

import argparse
import importlib.util
import os
from typing import Any, Dict

import tile_graph


def run(data_dir: str, out_dir: str, use_pbar: bool = True) -> Dict[str, Any]:
    """Prepare the tutorial inputs and run the tile-graph example."""
    # [tutorial-data]
    import geopandas as gpd
    import pandas as pd
    import rsgispy
    from rsgispy.imageutils.rasterise import rasterise_vec_lyr
    from rsgispy.imageutils.subsetting import subset_bbox
    from rsgispy.imageutils.warp import resample_img_to_match

    os.makedirs(out_dir, exist_ok=True)

    # A 36 x 30 km area (1800 x 1500 pixels) around Aberystwyth, inside the DEM.
    s2_img = os.path.join(out_dir, "sen2_20180629_aber_20m.tif")
    subset_bbox(os.path.join(data_dir,
                             "sen2_20180629_t30uvd_orb037_osgb_stdsref_20m.tif"),
                s2_img, 255000, 291000, 269000, 299000, use_pbar=use_pbar)

    # The DEM on the image's 20 m grid (a tile graph does not resample).
    dem_img = os.path.join(out_dir, "dem_20m.tif")
    resample_img_to_match(s2_img, os.path.join(data_dir, "dem_10m_osgb.tif"),
                          dem_img, interp_method=rsgispy.INTERP_BILINEAR,
                          use_pbar=use_pbar)

    # The training polygons (one layer per class) as one raster of class IDs,
    # 0 = unlabelled; the classes and colours of the tutorial notebook.
    train_gpkg = os.path.join(data_dir, "cls_data", "aber_sen2_cls_training.gpkg")
    classes = {  # class ID: (name, colour, layer)
        1: ("Artificial_Surfaces", "#b4b4b4", "Artificial_Surfaces"),
        2: ("Bare_Rock_Sand", "#83fafc", "Bare_Rock_Sand"),
        3: ("Conifer_Forest", "#006910", "Conifer_Forest"),
        4: ("Deciduous_Forest", "#3ea84e", "Deciduous_Forest"),
        5: ("Grass_Long", "#f7b020", "Grass_Long"),
        6: ("Grass_Short", "#fce76d", "Grass_Short"),
        7: ("NonPhoto_Veg", "#9cffd7", "NonPhotosynthetic_Vegetation"),
        8: ("Scrub", "#e172f7", "Scrub"),
        9: ("Water", "#0000ff", "Water_Training"),
        10: ("Bracken", "#8cff00", "Bracken"),
    }
    polys = pd.concat(
        [gpd.read_file(train_gpkg, layer=lyr).assign(cls_id=cls_id)
         for cls_id, (_, _, lyr) in classes.items()],
        ignore_index=True,
    )
    train_img = os.path.join(out_dir, "training_labels.tif")
    rasterise_vec_lyr(gpd.GeoDataFrame(polys, crs=polys.crs), None, s2_img,
                      train_img, att_column="cls_id", use_pbar=use_pbar)

    out = tile_graph.run_workflow(
        s2_img,
        dem_img,
        train_img,
        out_dir,
        band_names={1: "Blue", 2: "Green", 3: "Red", 7: "NIR"},  # B2, B3, B4, B8
        classes={cls_id: (name, clr) for cls_id, (name, clr, _) in classes.items()},
        n_jobs=4,
        use_pbar=use_pbar,
    )
    # [end-tutorial-data]
    out["fig_file"] = make_figures(out_dir)
    return out


def make_figures(out_dir: str) -> str:
    """Draw the figure of the guide (the image and the class map) in ``out_dir``."""
    # pylint: disable=import-outside-toplevel
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import rsgispy.mapping.layout as map_layout

    fig_file = os.path.join(out_dir, "tile_graph_cls_map.png")
    fig, _ = map_layout.create_map_panels(
        [
            map_layout.MapPanel("rgb", os.path.join(out_dir, "sen2_20180629_aber_20m.tif"),
                                bands=[8, 9, 3], title="Sentinel-2 (NIR, SWIR1, red)"),
            map_layout.MapPanel("thematic", os.path.join(out_dir, "s2_cls.tif"),
                                title="Class map from the graph (s2_cls.tif)"),
        ],
        figsize=(11, 4.4),
        legend_title="Class",
        out_plot_file=fig_file,
        out_dpi=100,
    )
    plt.close(fig)
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
