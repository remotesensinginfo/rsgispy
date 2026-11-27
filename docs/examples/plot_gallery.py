"""
Plot gallery: draws the output of every rsgispy plotting function on small
synthetic data and writes one PNG per plot. The images are shown in
``docs/guides/plot_gallery.rst`` and the script is run by the test suite
(``tests/test_plot_gallery_example.py``).

The synthetic scene comes from ``tests/maps_scene.py`` (the data of the
"Making maps" guide); the tables (regression results, endmembers, ROC curve,
feature importances) are made here with a fixed random seed. All the data of
the plots are in one dictionary (``make_inputs``), so they can be replaced:
the images in the guide were drawn with real data from the rsgispy tutorials
where a plot is about data (``plot_gallery_tutorial.py``).

Draw the gallery from the synthetic data with::

    python docs/examples/plot_gallery.py --out-dir plot_gallery --dpi 90
"""

import argparse
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # no display needed

# pylint: disable=wrong-import-position
import geopandas as gpd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import rsgispy  # noqa: E402
import rsgispy.imagespectral.endmembers as spec_em  # noqa: E402
import rsgispy.imagespectral.spectra as spec_spectra  # noqa: E402
import rsgispy.imageutils.metadata  # noqa: E402
import rsgispy.mapping.elements as map_elements  # noqa: E402
import rsgispy.mapping.layout as map_layout  # noqa: E402
import rsgispy.mapping.legend as map_legend  # noqa: E402
import rsgispy.mapping.raster as map_raster  # noqa: E402
import rsgispy.mapping.vector as map_vector  # noqa: E402
import rsgispy.modelling.species_distribution as sdm  # noqa: E402
import rsgispy.regression.regmetrics as rmet  # noqa: E402
import rsgispy.tools._ml_explain as ml_explain  # noqa: E402
import rsgispy.tools.plotting as rplot  # noqa: E402
import rsgispy.tools.stats as rstats  # noqa: E402

TESTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, "tests")

CLASS_NAMES = {1: "Water", 2: "Forest", 3: "Grass", 4: "Urban"}


def make_inputs(scene_dir: str, seed: int = 42) -> Dict[str, Any]:
    """
    The data of every gallery plot, from the synthetic scene of
    ``tests/maps_scene.py`` (written to ``scene_dir``) and synthetic tables.
    Each plot reads its entry of the returned dictionary, so any entry can be
    replaced with real data (see ``plot_gallery_tutorial.py``).
    """
    # pylint: disable=too-many-locals
    sys.path.insert(0, TESTS_DIR)
    from maps_scene import make_maps_scene  # pylint: disable=import-outside-toplevel,import-error

    scene = make_maps_scene(scene_dir)
    rng = np.random.default_rng(seed)
    obs = rng.uniform(5.0, 95.0, 150)
    # Tree-model-like predictions: shrunk towards the mean, with noise.
    pred = 50.0 + 0.8 * (obs - 50.0) + rng.normal(0.0, 6.0, obs.size)
    height_obs = 0.3 * obs + rng.normal(0.0, 2.0, obs.size)
    height_pred = height_obs + rng.normal(0.5, 2.5, obs.size)
    eval_df = pd.DataFrame({
        "cover_obs": obs, "cover_pred": pred,
        "height_obs": height_obs, "height_pred": height_pred,
    })
    em_df = pd.DataFrame(
        {
            "Water": [0.06, 0.05, 0.03, 0.01, 0.005, 0.003],
            "Vegetation": [0.03, 0.07, 0.04, 0.40, 0.20, 0.09],
            "Sand": [0.12, 0.17, 0.22, 0.28, 0.35, 0.30],
        },
        index=pd.Index(["blue", "green", "red", "nir", "swir1", "swir2"], name="band"),
    )
    y_true = np.concatenate([np.zeros(200), np.ones(200)])
    scores = np.concatenate([rng.beta(2, 5, 200), rng.beta(5, 2, 200)])
    feats = ["NDVI", "NDWI", "Elevation", "Slope", "B11", "B12", "Texture"]
    # Features from two latent factors, so the correlations vary in sign and size.
    latent = rng.normal(size=(300, 2))
    loadings = np.array([[0.9, 0.1], [-0.8, 0.3], [0.2, 0.9], [0.1, 0.7],
                         [-0.6, -0.2], [-0.5, 0.1], [0.4, -0.4]])
    corr_df = pd.DataFrame(
        latent @ loadings.T + rng.normal(0.0, 0.4, (300, len(feats))), columns=feats
    ).corr()
    lags = np.linspace(40.0, 1000.0, 20)
    vario_df = pd.DataFrame({
        "distance": lags,
        "semivariance": 1.0 - np.exp(-lags / 300.0) + rng.normal(0.0, 0.03, 20),
        "count": np.linspace(400, 2000, 20).astype(int),
    })
    importances = rng.uniform(0.01, 0.3, len(feats))
    importance_errs = rng.uniform(0.005, 0.03, len(feats))
    dndvi = np.concatenate([rng.normal(0.0, 0.05, 4000), rng.normal(0.35, 0.08, 600)])
    wvls = np.arange(400.0, 1001.0, 5.0)
    regions = gpd.read_file(scene["regions"])
    regions["forest_pct"] = [62.0, 35.0, 48.0, 12.0]
    return {
        # Histograms
        "hist_threshold": {"data": dndvi, "threshold": 0.18, "x_label": "dNDVI",
                           "title_str": "dNDVI with the Otsu threshold"},
        "img_histogram": {
            "input_img": scene["score"], "n_bins": 60, "val_range": (0.0, 1.0),
            "thresholds": [0.4, 0.7], "threshold_labels": ["Moderate (0.4)", "High (0.7)"],
            "title": "Suitability score with class thresholds", "x_label": "Score (0-1)",
        },
        "img_2d_histogram": {
            "input_img": scene["s2"], "x_band": 4, "y_band": 3, "bins": 100,
            "x_range": (0, 4500), "y_range": (0, 4500), "val_scale": 100,
            "x_label": "NIR reflectance (%)", "y_label": "Red reflectance (%)",
            "title": "Feature space: NIR v red",
            # The class spectra of the synthetic scene, as "endmembers".
            "points": {"Water": (150, 300), "Forest": (3500, 300), "Grass": (2500, 600),
                       "Urban": (1800, 1400)},
        },
        # Spectra
        "endmembers": {"endmembers": em_df, "plot_title": "Endmember spectra",
                       "wavelengths": [482.0, 562.0, 655.0, 865.0, 1609.0, 2201.0]},
        "spectra": {
            "spectra": pd.DataFrame({f"{c} µg/cm²": _leaf_spectrum(wvls, c)
                                     for c in [10, 20, 40, 60, 80]}, index=wvls),
            "cmap_name": "viridis", "legend_title": "Chlorophyll", "title": "Leaf spectra",
        },
        "continuum_removal": {"spectrum": _leaf_spectrum(wvls, 40), "wavelengths": wvls,
                              "start_wvl": 530, "stop_wvl": 760,
                              "title": "Chlorophyll 40 µg/cm²"},
        "semivariogram": {"vario_df": vario_df, "title": "Empirical semivariogram",
                          "dist_label": "Distance (m)"},
        # Regression: one or more targets (observed and predicted columns)
        "regression": {"in_data": eval_df, "ref_cols": ["cover_obs", "height_obs"],
                       "pred_cols": ["cover_pred", "height_pred"],
                       "target_names": ["Canopy cover (%)", "Height (m)"]},
        "qq_values": rng.gamma(2.0, 2.0, 200),
        # Classification and feature diagnostics
        "roc": sdm.calc_binary_roc_auc(y_true, scores),
        "corr_df": corr_df,
        "feats": feats,
        "importances": (importances, importance_errs),
        # Maps
        "s2": scene["s2"], "rgb_bands": [3, 2, 1], "cls": scene["cls"],
        "classes_info": CLASS_NAMES, "score": scene["score"],
        "score_title": "Suitability score", "score_label": "Score (0-1)",
        "dem": scene["dem"], "regions": regions, "rivers": gpd.read_file(scene["rivers"]),
    }


def _leaf_spectrum(wvls: np.ndarray, chl: float) -> np.ndarray:
    """A synthetic leaf reflectance spectrum for a chlorophyll content."""
    red_edge = 0.05 + 0.42 / (1.0 + np.exp(-(wvls - (700.0 + 0.2 * chl)) / 15.0))
    chl_abs = 1.0 - (0.3 + 0.006 * chl) * np.exp(-0.5 * ((wvls - 670.0) / 30.0) ** 2)
    blue_abs = 1.0 - 0.5 * np.exp(-0.5 * ((wvls - 450.0) / 40.0) ** 2)
    green = 1.0 + 0.6 * np.exp(-0.5 * ((wvls - 550.0) / 25.0) ** 2) * (1.0 - chl / 120.0)
    return np.clip(red_edge * chl_abs * blue_abs * green, 0.01, None)


def _gallery_plots(d: Dict[str, Any]) -> List[Tuple[str, Callable[[], object]]]:
    """(name, function returning a figure) for every plot in the gallery."""
    # pylint: disable=too-many-locals,too-many-statements
    bbox = rsgispy.imageutils.metadata.get_img_bbox(d["s2"])
    reg = d["regression"]
    reg_one = {"in_data": reg["in_data"], "ref_col": reg["ref_cols"][0],
               "pred_col": reg["pred_cols"][0], "target_names": reg["target_names"][:1]}

    def hist_threshold():
        fig, _ = rplot.plot_histogram_threshold(**d["hist_threshold"])
        return fig

    def img_histogram():
        fig, _ = rplot.plot_img_histogram(**d["img_histogram"], use_pbar=False)
        return fig

    def img_2d_histogram():
        fig, _ = rplot.plot_2d_img_histogram(**d["img_2d_histogram"], use_pbar=False)
        return fig

    def roc_curve():
        fpr, tpr, roc_auc = d["roc"]
        fig, _ = sdm.plot_roc_curve(fpr, tpr, roc_auc, label="Ensemble")
        return fig

    def endmembers():
        fig, _ = spec_em.plot_endmembers(**d["endmembers"])
        fig.axes[0].set_xlabel("Wavelength (nm)")
        return fig

    def spectra_plot():
        fig, _ = spec_spectra.plot_spectra(**d["spectra"])
        return fig

    def continuum_removal():
        cr_kw = dict(d["continuum_removal"])
        title = cr_kw.pop("title")
        fig, _ = spec_spectra.plot_continuum_removal(
            spec_spectra.calc_continuum_removal(**cr_kw), title=title
        )
        return fig

    def semivariogram():
        fig, _ = rstats.plot_semivariogram(**d["semivariogram"])
        return fig

    def pred_vs_obs():
        fig, _ = rmet.plot_pred_vs_obs(**reg_one)
        return fig

    def pred_vs_obs_multi():
        fig, _ = rmet.plot_pred_vs_obs(**reg)
        return fig

    def residuals():
        fig, _ = rmet.plot_residuals(**reg_one)
        return fig

    def residuals_multi():
        fig, _ = rmet.plot_residuals(**reg)
        return fig

    def qq_residuals():
        fig, _ = rmet.plot_qq_residuals(**reg_one)
        return fig

    def qq_values():
        fig, _ = rplot.plot_qq(d["qq_values"], title="Q-Q plot of skewed values")
        return fig

    def correlation_heatmap():
        fig, _ = ml_explain.plot_correlation_heatmap(d["corr_df"])
        return fig

    def jackknife_bars():
        n_feats = len(d["feats"])
        fig, _ = ml_explain.plot_jackknife_bars(
            d["feats"], np.linspace(0.80, 0.86, n_feats), np.linspace(0.40, 0.70, n_feats),
            0.88, "Jackknife accuracy", "Test accuracy",
        )
        return fig

    def importance_bars():
        importances, errors = d["importances"]
        fig, _ = ml_explain.plot_feature_importance_bars(
            d["feats"], importances, errors=errors,
            title="Permutation importance", xlabel="Mean decrease in accuracy",
        )
        return fig

    def boruta_boxplots():
        rng = np.random.default_rng(1)
        history = {f: list(rng.normal(m, 0.02, 30)) for f, m in
                   zip(d["feats"], [0.3, 0.25, 0.2, 0.05, 0.12, 0.02, 0.08])}
        status = dict(zip(d["feats"], ["Confirmed", "Confirmed", "Confirmed", "Rejected",
                                       "Tentative", "Rejected", "Tentative"]))
        fig, _ = ml_explain._plot_boruta_shap_boxplots(  # pylint: disable=protected-access
            history, status, list(rng.normal(0.1, 0.02, 30)), list(rng.normal(0.04, 0.01, 30))
        )
        return fig

    def map_figure(draw: Callable[[plt.Axes], object]) -> Callable[[], object]:
        def _make():
            fig, ax = plt.subplots(figsize=(5.5, 4.4))
            draw(ax)
            return fig
        return _make

    def rgb_map(ax):
        map_raster.create_raster_img_map(
            ax, d["s2"], img_bands=d["rgb_bands"], img_stch=rsgispy.IMG_STRETCH_CUMULATIVE,
            title_str="Sentinel-2 true colour", show_north_arrow=True,
            scale_bar_loc="lower right", no_data_out_clr=(1.0, 1.0, 1.0),
        )

    def cmap_map(ax):
        map_raster.create_raster_cmap_img_map(
            ax, d["score"], 1, norm_img_vals=True, norm_vmin=0.0, norm_vmax=1.0,
            title_str=d["score_title"], show_colour_bar=True, cbar_label=d["score_label"],
            scale_bar_loc="lower right",
        )

    def thematic_map(ax):
        map_raster.create_thematic_raster_map(
            ax, d["cls"], title_str="Land cover", show_legend=True,
            classes_info=d["classes_info"], legend_title="Land cover", legend_position="bottom",
            legend_ncols=2, scale_bar_loc="lower right",
        )

    def shaded_cmap_map(ax):
        map_raster.create_shaded_cmap_img_map(
            ax, d["dem"], dem_img=d["dem"], cmap_name="viridis",
            title_str="Elevation over a hillshade", cbar_label="Elevation (m)",
            scale_bar_loc="lower right",
        )

    def hillshade_map(ax):
        map_raster.create_thematic_raster_map(
            ax, d["cls"], title_str="Land cover with a hillshade",
            scale_bar_loc="lower right", show_north_arrow=True,
        )
        map_raster.add_hillshade(ax, dem_img=d["dem"], alpha=0.35)

    def vector_map(ax):
        map_raster.create_raster_img_map(
            ax, d["s2"], img_bands=d["rgb_bands"], img_stch=rsgispy.IMG_STRETCH_CUMULATIVE,
            show_scale_bar=False,
        )
        map_vector.create_vec_lyr_map(
            ax, [d["regions"], d["rivers"]], bbox, vec_fill_clrs=["none", "none"],
            vec_line_clrs=["yellow", "deepskyblue"], vec_line_widths=[1.0, 2.0],
            plot_zorders=[2, 3], title_str="Regions and rivers", scale_bar_loc="lower right",
        )

    def choropleth_map(ax):
        map_vector.create_choropleth_vec_lyr_map(
            ax, d["regions"], "forest_pct", bbox, vec_fill_cmap="viridis", vec_line_width=0.6,
            title_str="Forest cover by region", show_colour_bar=True, cbar_label="Forest (%)",
            scale_bar_loc="lower right",
        )

    def elements_map(ax):
        map_raster.create_thematic_raster_map(ax, d["cls"], show_scale_bar=False,
                                              title_str="Map elements")
        map_elements.add_scale_bar(ax, epsg_code=bbox.epsg, loc="lower right",
                                   style=map_elements.SCALEBAR_STYLE_ALTERNATING)
        map_elements.add_north_arrow(ax, loc="upper right",
                                     style=map_elements.NORTH_ARROW_STYLE_COMPASS)
        # A box a quarter of the map's width and height, at its centre.
        x_mid, y_mid = (bbox[0] + bbox[1]) / 2, (bbox[2] + bbox[3]) / 2
        dx, dy = (bbox[1] - bbox[0]) / 8, (bbox[3] - bbox[2]) / 8
        map_elements.draw_bboxes_to_axis(
            ax, [(x_mid - dx, x_mid + dx, y_mid - dy, y_mid + dy)], bbox_labels=["Zoom"],
            rect_clr="red", line_width=1.5, lbl_pos="above", lbl_font_size=9,
            lbl_font_clr="red", lbl_fill_clr="white",
        )
        map_legend.add_categorical_legend(ax, classes_info=d["classes_info"], input_img=d["cls"],
                                          loc="lower left", title="Land cover", ncols=2,
                                          fontsize=7)
        map_elements.add_axis_label(ax, "(a)")

    def map_panels():
        fig, _ = map_layout.create_map_panels(
            [
                {"panel_type": "rgb", "input_img": d["s2"], "bands": d["rgb_bands"],
                 "title": "Sentinel-2", "map_kw": {"no_data_out_clr": (1.0, 1.0, 1.0)}},
                {"panel_type": "thematic", "input_img": d["cls"], "title": "Land cover"},
                {"panel_type": "vector", "vec": d["rivers"], "base_img": d["s2"],
                 "base_bands": d["rgb_bands"], "title": "Rivers over the image",
                 "map_kw": {"vec_line_clrs": "deepskyblue", "vec_line_widths": 1.5}},
            ],
            classes_info=d["classes_info"], legend_title="Land cover", legend_position="bottom",
            scale_bar_loc="lower right",
        )
        return fig

    return [
        ("plot_histogram_threshold", hist_threshold),
        ("plot_img_histogram", img_histogram),
        ("plot_2d_img_histogram", img_2d_histogram),
        ("plot_roc_curve", roc_curve),
        ("plot_endmembers", endmembers),
        ("plot_spectra", spectra_plot),
        ("plot_continuum_removal", continuum_removal),
        ("plot_semivariogram", semivariogram),
        ("plot_pred_vs_obs", pred_vs_obs),
        ("plot_pred_vs_obs_multi", pred_vs_obs_multi),
        ("plot_residuals", residuals),
        ("plot_residuals_multi", residuals_multi),
        ("plot_qq_residuals", qq_residuals),
        ("plot_qq", qq_values),
        ("plot_correlation_heatmap", correlation_heatmap),
        ("plot_jackknife_bars", jackknife_bars),
        ("plot_feature_importance_bars", importance_bars),
        ("plot_boruta_shap_boxplots", boruta_boxplots),
        ("create_raster_img_map", map_figure(rgb_map)),
        ("create_raster_cmap_img_map", map_figure(cmap_map)),
        ("create_thematic_raster_map", map_figure(thematic_map)),
        ("create_shaded_cmap_img_map", map_figure(shaded_cmap_map)),
        ("add_hillshade", map_figure(hillshade_map)),
        ("create_vec_lyr_map", map_figure(vector_map)),
        ("create_choropleth_vec_lyr_map", map_figure(choropleth_map)),
        ("map_elements", map_figure(elements_map)),
        ("create_map_panels", map_panels),
    ]


def run_gallery(out_dir: str, scene_dir: Optional[str] = None, dpi: int = 90,
                inputs: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """
    Write every gallery plot to ``out_dir`` and return ``{name: png path}``.
    ``inputs`` is the data of the plots (by default ``make_inputs``, the
    synthetic data, with the scene written to ``scene_dir``).
    """
    os.makedirs(out_dir, exist_ok=True)
    if inputs is None:
        inputs = make_inputs(scene_dir or os.path.join(out_dir, "_scene"))
    out: Dict[str, str] = {}
    for name, make_fig in _gallery_plots(inputs):
        fig = make_fig()
        out[name] = rplot.save_plot(fig, os.path.join(out_dir, f"{name}.png"), out_dpi=dpi, close=True)
    return out


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    parser.add_argument("--out-dir", default="plot_gallery", help="Output directory for the PNGs.")
    parser.add_argument("--scene-dir", default=None, help="Directory for the synthetic data.")
    parser.add_argument("--dpi", type=int, default=90, help="Resolution of the PNGs.")
    args = parser.parse_args()
    for name, path in run_gallery(args.out_dir, args.scene_dir, args.dpi).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
