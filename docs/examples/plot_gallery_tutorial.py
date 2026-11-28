"""
Draw the plot gallery (``plot_gallery.py``) with real data from the rsgispy
tutorials (https://github.com/remotesensinginfo/rsgispy-tutorials) wherever a plot
is about data:

* the maps: the Aberystwyth maps of ``making_maps_tutorial.py`` (Sentinel-2,
  the land cover classified from it, NDVI, the 10 m DEM and rivers derived
  from it);
* the histograms: the MNDWI of the Sentinel-2 scene
  (``data/sen2_20180629_t30uvd_orb037_osgb_stdsref_20m.tif``) and the NDVI
  difference of two Landsat images (``data/chg_data``);
* the spectra and the feature space: the HyMap image
  (``data/hymap_injune_subset.tif``, ``data/hymap_wavelengths.json``,
  ``data/hymap_ppi_endmembers.geojson``);
* the semivariogram: elevations from the DEM (``data/dem_10m_osgb.tif``);
* the regression plots and the correlation heatmap: the forest plots and
  LiDAR metrics (``data/lidar``).

The other plots (ROC curve, feature importances, Q-Q plot of skewed values)
keep the synthetic values of ``plot_gallery.py``. The images in
``docs/guides/plot_gallery.rst`` were made with this script::

    python plot_gallery_tutorial.py --data-dir rsgispy-tutorials/data \\
        --out-dir gallery_tutorial
"""

import argparse
import os
from typing import Any, Dict

import making_maps_tutorial
import plot_gallery


def run(data_dir: str, out_dir: str, use_pbar: bool = True,
        dpi: int = 90) -> Dict[str, Any]:
    """Prepare the tutorial data and draw the plot gallery."""
    # pylint: disable=too-many-locals,import-outside-toplevel
    # [tutorial-data]
    import geopandas as gpd
    import numpy as np
    import pandas as pd
    import rasterio
    import rsgispy.imagespectral.endmembers as spec_em
    import rsgispy.tools.stats as rstats
    import rsgispy.zonalstats
    from rsgispy.imagecalc.calcindices import calc_indices
    from rsgispy.imageutils.arrays import get_img_pxl_column
    from rsgispy.imageutils.metadata import get_img_pxl_coords
    from rsgispy.tools.jsonio import read_json_to_dict
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.model_selection import KFold, cross_val_predict

    os.makedirs(out_dir, exist_ok=True)
    # The synthetic data, for the plots that are not about data.
    d = plot_gallery.make_inputs(os.path.join(out_dir, "_scene"))

    # Maps: the inputs of the Aberystwyth maps of the making maps guide.
    maps = making_maps_tutorial.run(data_dir, os.path.join(out_dir, "maps"),
                                    use_pbar=use_pbar, dpi=50)
    d.update(s2=maps["s2_img"], cls=maps["cls_img"], classes_info=maps["classes_info"],
             score=maps["ndvi_img"], score_title="NDVI", score_label="NDVI",
             dem=maps["dem_img"], regions=maps["regions"],
             rivers=gpd.read_file(maps["rivers_vec"]))

    # Image histogram: the MNDWI of the Sentinel-2 scene, with the water
    # thresholds tried in the rule-based classification notebook.
    mndwi_img = os.path.join(out_dir, "sen2_20180629_mndwi.tif")
    calc_indices(os.path.join(data_dir, "sen2_20180629_t30uvd_orb037_osgb_stdsref_20m.tif"),
                 mndwi_img, band_names={2: "green", 9: "swir1"}, indices=["GMNDWI"],
                 no_data_val=0, out_no_data_val=-999, use_pbar=use_pbar)
    d["img_histogram"] = {
        "input_img": mndwi_img, "n_bins": 200, "val_range": (-1, 1),
        "thresholds": [-0.25, 0.1, 0.5],
        "threshold_labels": ["Rule base (-0.25)", "Second try (0.1)", "First try (0.5)"],
        "title": "Sentinel-2 MNDWI with water thresholds", "x_label": "MNDWI",
    }

    # Histogram and threshold: the NDVI difference (1997 - 2018) of two
    # Landsat images of a mangrove coast, with the threshold the image to
    # image change notebook uses for mangrove loss.
    ls_imgs = {"LS5TM_19970716": (3, 4), "LS8_20180608": (4, 5)}  # red, NIR bands
    ndvi = {}
    for name, bands in ls_imgs.items():
        ls_file = f"{name}_vmsk_mclds_topshad_rad_srefdem_stdsref_subset.tif"
        with rasterio.open(os.path.join(data_dir, "chg_data", ls_file)) as src:
            red, nir = src.read(bands).astype(np.float32)
        ndvi[name] = np.divide(nir - red, nir + red, where=(red > 0) & (nir > 0),
                               out=np.full(red.shape, np.nan, np.float32))  # NaN: no data
    dndvi = ndvi["LS5TM_19970716"] - ndvi["LS8_20180608"]
    dndvi = dndvi[np.isfinite(dndvi)]
    d["hist_threshold"] = {"data": dndvi, "threshold": 0.35, "n_bins": 100,
                           "threshold_label": "Mangrove loss (0.35)", "x_label": "dNDVI",
                           "title_str": "Landsat NDVI difference (1997 - 2018)"}

    # Spectra: HyMap. Mean spectra in the PPI endmember polygons, and the
    # spectra of three pixels (vegetation, bare soil, shadow), as in the
    # spectral unmixing notebook. The image is reflectance x 10000.
    hymap_img = os.path.join(data_dir, "hymap_injune_subset.tif")
    wvls = read_json_to_dict(os.path.join(data_dir, "hymap_wavelengths.json"))["wavelength"]
    em_names = ["Photosynthetic", "Bare/Soil", "Shadow"]
    em_df = spec_em.extract_avg_endmembers(
        hymap_img, os.path.join(data_dir, "hymap_ppi_endmembers.geojson"),
        endmember_names=em_names,
    )
    d["endmembers"] = {"endmembers": em_df, "wavelengths": wvls, "gain": 100,
                       "y_label": "Reflectance (%)", "plot_title": "HyMap PPI endmembers"}
    d["img_2d_histogram"] = {
        "input_img": hymap_img, "x_band": 42, "y_band": 13, "bins": 100,
        "x_range": (1, 6000), "y_range": (1, 6000), "val_scale": 100,
        "x_label": "NIR reflectance (%)", "y_label": "Red reflectance (%)",
        "title": "HyMap feature space: NIR v red",
        # The endmembers in bands 42 and 13 (rows 41 and 12 of em_df).
        "points": {n: (em_df[n].iloc[41], em_df[n].iloc[12]) for n in em_names},
    }
    x_pxls, y_pxls = get_img_pxl_coords(hymap_img, [546933.6, 547573.3, 547903.6],
                                        [7165115.7, 7164588.9, 7161428.6])
    pxl_spectra = {n: get_img_pxl_column(hymap_img, int(x), int(y)) / 100.0
                   for n, x, y in zip(em_names, x_pxls, y_pxls)}
    d["spectra"] = {"spectra": pxl_spectra, "wavelengths": wvls,
                    "y_label": "Reflectance (%)", "title": "HyMap pixel spectra"}
    d["continuum_removal"] = {
        "spectrum": pxl_spectra["Photosynthetic"] / 100.0, "wavelengths": wvls,
        "start_wvl": 530, "stop_wvl": 760, "title": "HyMap vegetation pixel",
    }

    # Semivariogram: the elevation of 2000 random points within 10 km of
    # (266500, 278100), as in the semivariograms notebook (land points only).
    rng = np.random.default_rng(42)
    dist, angle = 10000 * np.sqrt(rng.uniform(size=2000)), rng.uniform(0, 2 * np.pi, 2000)
    pts = np.column_stack([266500 + dist * np.cos(angle), 278100 + dist * np.sin(angle)])
    with rasterio.open(os.path.join(data_dir, "dem_10m_osgb.tif")) as src:
        elev = np.array([val[0] for val in src.sample(pts)])
    land = elev > 0
    d["semivariogram"] = {
        "vario_df": rstats.calc_semivariogram(pts[land], elev[land], n_lags=25),
        "title": "Semivariogram of elevation", "dist_label": "Lag distance (m)",
    }

    # Regression: stem volume and mean DBH of 166 forest plots from LiDAR
    # metrics, with the Extra Trees parameters of the regression notebooks.
    # Each plot is predicted by a model fitted without it (5-fold).
    lidar_dir = os.path.join(data_dir, "lidar")
    plots = pd.read_csv(os.path.join(lidar_dir, "Forest_Plot_Metrics_LassoLars_Sel.csv"),
                        index_col=0)
    feats = list(plots.columns[6:])
    targets = {"Vol / ha": "Stem volume (m³ ha⁻¹)", "Mean DBH": "Mean DBH (cm)"}
    et_params = read_json_to_dict(os.path.join(lidar_dir, "regrs_opt_params_et.json"))
    pred = cross_val_predict(ExtraTreesRegressor(**et_params, random_state=0),
                             plots[feats], plots[list(targets)],
                             cv=KFold(5, shuffle=True, random_state=0))
    eval_df = pd.DataFrame({"vol_obs": plots["Vol / ha"], "vol_pred": pred[:, 0],
                            "dbh_obs": plots["Mean DBH"], "dbh_pred": pred[:, 1]})
    d["regression"] = {"in_data": eval_df, "ref_cols": ["vol_obs", "dbh_obs"],
                       "pred_cols": ["vol_pred", "dbh_pred"],
                       "target_names": list(targets.values())}
    d["corr_df"] = plots[feats].corr()  # the LiDAR metrics

    out = plot_gallery.run_gallery(out_dir, dpi=dpi, inputs=d)
    # [end-tutorial-data]
    return out


def main() -> None:
    """Command line interface."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    parser.add_argument("--data-dir", required=True,
                        help="The data folder of the rsgispy-tutorials repository")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--dpi", type=int, default=90, help="Resolution of the PNGs")
    args = parser.parse_args()
    run(args.data_dir, args.out_dir, dpi=args.dpi)


if __name__ == "__main__":
    main()
