"""
Run the species distribution modelling worked example
(``species_distribution_modelling.py``) on the Snowdon beetle
(*Chrysolina cerealis*) data from the rsgispy tutorials
(https://github.com/petebunting/rsgispy-tutorials, ``data/sdm``): the
sightings CSV, the 20 m elevation and mean winter temperature rasters, the
Phase 1 habitat map and the study area in Snowdonia. The figures and printed
outputs in ``docs/guides/species_distribution_modelling.rst`` were made with
this script::

    python species_distribution_modelling_tutorial.py \\
        --data-dir rsgispy-tutorials/data --out-dir sdm_tutorial
"""

import argparse
import os
from typing import Any, Dict

import species_distribution_modelling


def run(data_dir: str, out_dir: str, use_pbar: bool = True) -> Dict[str, Any]:
    """Prepare the tutorial inputs and run the SDM worked example."""
    # pylint: disable=import-outside-toplevel
    # [tutorial-data]
    import geopandas as gpd
    import pandas as pd
    import rasterio
    import rsgispy
    from rsgispy.imagecalc.distance import calc_dist_to_img_vals
    from rsgispy.imageutils.rasterise import rasterise_vec_lyr

    env_dir = os.path.join(data_dir, "sdm", "environmental_data")
    elev_img = os.path.join(env_dir, "Elevation.tif")  # 20 m, OSGB (EPSG:27700)
    os.makedirs(out_dir, exist_ok=True)

    # Sightings (WGS84 latitude/longitude) to British National Grid points.
    # Drop records located to 2 km or worse (e.g. the centre of a 10 km
    # square), then keep one record per 20 m pixel: most of the 119 records
    # are repeat visits to the same few sites.
    rec_df = pd.read_csv(os.path.join(data_dir, "sdm", "sightings",
                                      "Chrysolina_cerealis.csv"))
    rec_gdf = gpd.GeoDataFrame(
        rec_df, crs="EPSG:4326",
        geometry=gpd.points_from_xy(rec_df["Longitude (WGS84)"],
                                    rec_df["Latitude (WGS84)"]),
    ).to_crs("EPSG:27700")
    rec_gdf = rec_gdf[rec_gdf["Coordinate uncertainty (m)"] <= 1000]
    with rasterio.open(elev_img) as src:
        rec_gdf["pxl"] = [src.index(x, y) for x, y in
                          zip(rec_gdf.geometry.x, rec_gdf.geometry.y)]
    pres_gdf = rec_gdf.drop_duplicates("pxl")[["Year", "geometry"]]
    presence_vec = os.path.join(out_dir, "presences.gpkg")
    pres_gdf.to_file(presence_vec, layer="presences")

    # Study area mask (1 inside) on the elevation grid.
    roi_img = os.path.join(out_dir, "study_area.tif")
    rasterise_vec_lyr(os.path.join(env_dir, "Study_Area.geojson"), None,
                      elev_img, roi_img, burn_val=1,
                      datatype=rsgispy.TYPE_8UINT, use_pbar=use_pbar)

    # Phase 1 habitat map rasterised from its 'Class' column, then the
    # distance (m) to the habitats that matter to the beetle, which lives on
    # wild thyme on mountain ledges, grassland and rocks.
    phase1_img = os.path.join(out_dir, "phase1.tif")
    rasterise_vec_lyr(os.path.join(env_dir, "Phase1_Veg.geojson"), None,
                      elev_img, phase1_img, att_column="Class",
                      datatype=rsgispy.TYPE_8UINT, use_pbar=use_pbar)
    habitats = {
        "ledges_dist": [14],  # C.2 upland species-rich ledges
        "calc_grass_dist": [8, 20],  # B.3.1 calcareous grassland, D.6 mosaic
        "rock_dist": [33, 36, 37],  # I.1, I.1.4, I.1.4.1 rock exposures
        "scree_dist": [34, 35],  # I.1.2, I.1.2.1 scree
        "flush_dist": [24, 25],  # E.2, E.2.1 flushes and springs
        "impr_grass_dist": [9],  # B.4 improved grassland
    }
    env_imgs = {
        "elevation": (elev_img, 1),
        "winter_temp": (os.path.join(env_dir, "Mean_Winter_Temp.tif"), 1),
    }
    for name, cls_ids in habitats.items():
        dist_img = os.path.join(out_dir, f"{name}.tif")
        calc_dist_to_img_vals(phase1_img, dist_img, pxl_vals=cls_ids,
                              out_no_data_val=-9999, use_pbar=use_pbar)
        env_imgs[name] = (dist_img, 1)

    out = species_distribution_modelling.run_workflow(
        env_imgs,
        roi_img,
        presence_vec,
        out_dir,
        presence_lyr="presences",
        drop_vars=("elevation",),  # r = -0.95 with winter temperature
        n_pseudo_abs=1000,
        pseudo_abs_dist=250.0,
        n_trees=500,
        n_kfolds=10,
        cv_block_size=1000.0,  # 1 km blocks
        salib_n_samples=1024,
        use_pbar=use_pbar,
    )
    # [end-tutorial-data]
    make_figures(out_dir)
    return out


def make_figures(out_dir: str) -> None:
    """
    Draw the guide's figures (``sdm_response_temp.png`` and
    ``sdm_probability.png``, copied to ``docs/guides/images``) into
    ``out_dir`` from the outputs of :func:`run`.
    """
    # pylint: disable=import-outside-toplevel
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import rasterio

    # Response to winter temperature, with the training presences as ticks.
    resp_df = pd.read_csv(os.path.join(out_dir, "response_curves.csv"))
    train_gdf = gpd.read_file(os.path.join(out_dir, "sdm_train.gpkg"), layer="train")
    pres_temp = train_gdf.loc[train_gdf["clsid"] == 1, "winter_temp"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(resp_df["winter_temp_vals"], resp_df["winter_temp_resp"], lw=2,
            color="#1f6fb4")
    ax.plot(pres_temp, np.full(len(pres_temp), 0.02), "|", color="k", ms=12,
            label="Training presences")
    ax.set_xlabel("Mean winter temperature (°C)")
    ax.set_ylabel("Presence probability")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    ax.set_title("Response to winter temperature (random forest)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "sdm_response_temp.png"), dpi=100)
    plt.close(fig)

    # Median presence probability of the ensemble, with the presences.
    with rasterio.open(os.path.join(out_dir, "sdm_probability.tif")) as src:
        prob = src.read(1, masked=True)
        bnds = src.bounds
    pres_gdf = gpd.read_file(os.path.join(out_dir, "presences.gpkg"),
                             layer="presences")
    fig, ax = plt.subplots(figsize=(8, 5.6))
    img = ax.imshow(prob, cmap="viridis", vmin=0, vmax=1,
                    extent=(bnds.left, bnds.right, bnds.bottom, bnds.top))
    ax.scatter(pres_gdf.geometry.x, pres_gdf.geometry.y, s=18, facecolor="none",
               edgecolor="red", lw=1.2, label="Presences (one per pixel)")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.ticklabel_format(style="plain")
    ax.legend(loc="lower right")
    cbar = fig.colorbar(img, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Median presence probability")
    ax.set_title("Snowdon beetle habitat suitability (ensemble median)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "sdm_probability.png"), dpi=100)
    plt.close(fig)


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
