"""
Run the object-based (segment) worked example (``segment_ml_workflow.py``) on
the data of the rsgispy tutorials
(https://github.com/petebunting/rsgispy-tutorials): the 20 m Sentinel-2 image
of 29 June 2018 (``data/sen2_20180629_t30uvd_orb037_osgb_stdsref_20m.tif``),
the training polygons of
``04_image_classification/04_object_based_classification.ipynb``
(``data/cls_data/aber_sen2_cls_training.gpkg``), the 10 m DEM
(``data/dem_10m_osgb.tif``) for the slope and, as the second scene, the 10 m
Sentinel-2 image of 27 May 2021 (``data/sen2_20210527_aber.tif``). The
printed outputs and the figures in ``docs/guides/segment_ml_workflow.rst``
were made with this script::

    python segment_ml_workflow_tutorial.py --data-dir rsgispy-tutorials/data \\
        --out-dir segment_ml_tutorial --fig-dir docs/guides/images
"""

import argparse
import importlib.util
import os
from typing import Any, Dict

import segment_ml_workflow


def run(data_dir: str, out_dir: str, use_pbar: bool = True) -> Dict[str, Any]:
    """Prepare the tutorial inputs and run the segment worked example."""
    # [tutorial-data]
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import box
    import rsgispy
    from rsgispy.elevation.terrain import slope
    from rsgispy.imageutils.subsetting import subset_bbox
    from rsgispy.imageutils.warp import resample_img_to_match

    os.makedirs(out_dir, exist_ok=True)
    s2_2018 = os.path.join(data_dir, "sen2_20180629_t30uvd_orb037_osgb_stdsref_20m.tif")

    # The DEM covers the west of the Sentinel-2 image: subset the image to a
    # 36 x 30 km area (1800 x 1500 pixels) around Aberystwyth inside the DEM.
    bbox = (255000, 291000, 269000, 299000)  # min x, max x, min y, max y
    s2_img = os.path.join(out_dir, "sen2_20180629_aber_20m.tif")
    subset_bbox(s2_2018, s2_img, *bbox, use_pbar=use_pbar)
    # The slope (degrees) on the Sentinel-2 grid.
    dem_slope_img = os.path.join(out_dir, "dem_10m_slope.tif")
    slope(os.path.join(data_dir, "dem_10m_osgb.tif"), dem_slope_img,
          no_data_val=-9999.0, use_pbar=use_pbar)
    slope_img = os.path.join(out_dir, "slope_20m.tif")
    resample_img_to_match(s2_img, dem_slope_img, slope_img,
                          interp_method=rsgispy.INTERP_AVERAGE, use_pbar=use_pbar)

    # The second scene: the 10 m image of 27 May 2021 (a 9 x 9 km area around
    # Aberystwyth) averaged onto the 20 m grid of the training image, so the
    # segments and their features are comparable.
    new_grid_img = os.path.join(out_dir, "sen2_20180629_aber_new_grid.tif")
    subset_bbox(s2_2018, new_grid_img, 257070, 266360, 276270, 285720,
                use_pbar=use_pbar)
    s2_img_new = os.path.join(out_dir, "sen2_20210527_aber_20m.tif")
    resample_img_to_match(new_grid_img, os.path.join(data_dir, "sen2_20210527_aber.tif"),
                          s2_img_new, interp_method=rsgispy.INTERP_AVERAGE,
                          use_pbar=use_pbar)
    slope_img_new = os.path.join(out_dir, "slope_20m_new.tif")
    resample_img_to_match(s2_img_new, dem_slope_img, slope_img_new,
                          interp_method=rsgispy.INTERP_AVERAGE, use_pbar=use_pbar)

    # The classes, layers and colours of the tutorial notebook (Grass_Long and
    # Grass_Short merged, as in the pixel classification guide). The training
    # polygons were drawn for a pixel classification and are often smaller
    # than a segment, so each polygon gives one point (inside it). Two thirds
    # of the polygons of each class train the model; the points of the other
    # third are the accuracy points.
    train_gpkg = os.path.join(data_dir, "cls_data", "aber_sen2_cls_training.gpkg")
    classes = {  # class name: (layers, output class ID, colour)
        "Artificial_Surfaces": (["Artificial_Surfaces"], 1, "#b4b4b4"),
        "Bare_Rock_Sand": (["Bare_Rock_Sand"], 2, "#83fafc"),
        "Conifer_Forest": (["Conifer_Forest"], 3, "#006910"),
        "Deciduous_Forest": (["Deciduous_Forest"], 4, "#3ea84e"),
        "Grass": (["Grass_Long", "Grass_Short"], 5, "#f7b020"),
        "NonPhoto_Veg": (["NonPhotosynthetic_Vegetation"], 7, "#9cffd7"),
        "Scrub": (["Scrub"], 8, "#e172f7"),
        "Water": (["Water_Training"], 9, "#0000ff"),
        "Bracken": (["Bracken"], 10, "#8cff00"),
    }
    area = box(bbox[0], bbox[2], bbox[1], bbox[3])
    train_vecs, acc_pts = {}, []
    for cls_name, (lyrs, cls_id, colour) in classes.items():
        polys = pd.concat([gpd.read_file(train_gpkg, layer=lyr) for lyr in lyrs],
                          ignore_index=True)
        polys = polys[polys.within(area)].sample(frac=1.0, random_state=42)
        pts = gpd.GeoDataFrame(geometry=polys.representative_point(), crs=polys.crs)
        n_acc = len(pts) // 3
        vec_file = os.path.join(out_dir, f"train_{cls_name.lower()}.gpkg")
        pts.iloc[n_acc:].to_file(vec_file)
        train_vecs[cls_name] = (vec_file, cls_id, colour)
        acc_pts.append(pts.iloc[:n_acc].assign(ref_name=cls_name))
    acc_pts_vec = os.path.join(out_dir, "acc_pts.gpkg")
    pd.concat(acc_pts, ignore_index=True).to_file(acc_pts_vec)

    out = segment_ml_workflow.run_workflow(
        s2_img,
        slope_img,
        train_vecs,
        None,  # no canopy cover plots in the tutorial data: no regression
        acc_pts_vec,
        out_dir,
        s2_img_new=s2_img_new,
        slope_img_new=slope_img_new,
        n_clusters=60,
        min_pixels=10,  # 0.4 ha with 20 m pixels
        use_gbdt=importlib.util.find_spec("xgboost") is not None,
        use_pbar=use_pbar,
    )
    # [end-tutorial-data]
    return out


def make_figures(out_dir: str, fig_dir: str) -> None:
    """Draw the figures of the guide from the outputs of :func:`run`."""
    # pylint: disable=import-outside-toplevel
    import geopandas as gpd
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    from rsgispy.mapping.layout import MapPanel, create_map_panels

    def save(fig, name):
        # 256 colours keep the PNG small with no visible change.
        png = os.path.join(fig_dir, name)
        fig.savefig(png, dpi=100, bbox_inches="tight")
        plt.close(fig)
        Image.open(png).convert("RGB").quantize(256).save(png, optimize=True)

    os.makedirs(fig_dir, exist_ok=True)
    s2_img = os.path.join(out_dir, "sen2_20180629_aber_20m.tif")
    nir_swir_red = [7, 9, 3]

    # The image, the segments and their classification, east of Aberystwyth.
    bbox = (262000, 266000, 278000, 282000)
    segs = gpd.read_parquet(os.path.join(out_dir, "s2_segs.parquet"),
                            columns=["geometry"])
    segs = segs.cx[bbox[0]:bbox[1], bbox[2]:bbox[3]]
    seg_lines = gpd.GeoDataFrame(geometry=segs.boundary, crs=segs.crs)
    fig, _ = create_map_panels(
        [
            MapPanel("rgb", s2_img, bands=nir_swir_red,
                     title="Sentinel-2, 29 June 2018"),
            MapPanel("vector", vec=seg_lines, base_img=s2_img,
                     base_bands=nir_swir_red,
                     map_kw={"vec_line_clrs": "yellow", "vec_line_widths": 0.4},
                     title="Segments"),
            MapPanel("thematic", os.path.join(out_dir, "s2_segs_classified.tif"),
                     title="Segment classification"),
        ],
        bbox=bbox,
        figsize=(14, 4.9),
        legend_title="Land cover",
    )
    save(fig, "segment_ml_classification.png")

    # The model applied to the image of another date (step 10).
    new_img = os.path.join(out_dir, "sen2_20210527_aber_20m.tif")
    bbox = (257100, 266300, 276300, 285700)
    fig, _ = create_map_panels(
        [
            MapPanel("rgb", new_img, bands=nir_swir_red,
                     title="Sentinel-2, 27 May 2021"),
            MapPanel("thematic", os.path.join(out_dir, "s2_segs_classified.tif"),
                     title="Classification, 29 June 2018"),
            MapPanel("thematic", os.path.join(out_dir, "s2_new_classified.tif"),
                     title="Same model, 27 May 2021"),
        ],
        bbox=bbox,
        figsize=(14, 4.9),
        legend_title="Land cover",
    )
    save(fig, "segment_ml_new_scene.png")


def main() -> None:
    """Command line interface."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    parser.add_argument("--data-dir", required=True,
                        help="The data folder of the rsgispy-tutorials repository")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--fig-dir", default=None,
                        help="Also draw the figures of the guide in this directory")
    args = parser.parse_args()
    run(args.data_dir, args.out_dir)
    if args.fig_dir:
        make_figures(args.out_dir, args.fig_dir)


if __name__ == "__main__":
    main()
