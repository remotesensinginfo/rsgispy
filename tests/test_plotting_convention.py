"""
Tests that every rsgispy plotting function follows the plotting conventions
(see ``docs/rsgispy_tools.rst``, "Plotting conventions"):

* functions that make a figure return ``(fig, ax)`` or ``(fig, axs)``, draw
  on a given axis without creating a new figure, save only when
  ``out_plot_file`` is given, and never close the figure they return or the
  one they were given;
* functions that draw on a given axis (maps and map elements) return the
  artist the caller needs next and create no figure.

To cover a new plotting function, add a case to ``FIGURE_CASES`` (a callable
taking ``ax`` and ``out_plot_file``) or ``AXIS_CASES`` (a callable taking
``ax`` and returning the documented artist type check).
"""

import os

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import Collection
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.legend import Legend
from matplotlib.patches import Rectangle
from matplotlib.text import Text
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

import rsgispy.imagespectral.endmembers as em
import rsgispy.mapping.elements as map_elements
import rsgispy.mapping.legend as map_legend
import rsgispy.mapping.raster as map_raster
import rsgispy.mapping.vector as map_vector
import rsgispy.modelling.species_distribution as sdm
import rsgispy.regression.regmetrics as rm
import rsgispy.tools._ml_explain as ml_explain
import rsgispy.tools.plotting as rplot
import rsgispy.tools.stats as rstats

matplotlib.use("Agg")

X_MIN, Y_MAX, RES = 500000.0, 5200000.0, 10.0
BBOX = (X_MIN, X_MIN + 200.0, Y_MAX - 200.0, Y_MAX)


@pytest.fixture(scope="module")
def plot_data(tmp_path_factory):
    """Small rasters, vectors and tables used by the plotting cases."""
    data_dir = tmp_path_factory.mktemp("plot_data")
    rng = np.random.default_rng(42)
    profile = {
        "driver": "GTiff",
        "width": 20,
        "height": 20,
        "crs": "EPSG:32630",
        "transform": from_origin(X_MIN, Y_MAX, RES, RES),
    }

    cont_img = os.path.join(data_dir, "cont.tif")
    with rasterio.open(cont_img, "w", count=3, dtype="float32", **profile) as dst:
        dst.write(rng.uniform(1.0, 100.0, (3, 20, 20)).astype(np.float32))

    dem_img = os.path.join(data_dir, "dem.tif")
    rows, cols = np.mgrid[0:20, 0:20]
    with rasterio.open(dem_img, "w", count=1, dtype="float32", **profile) as dst:
        dst.write((50.0 + 5.0 * np.sin(cols / 3.0) + rows).astype(np.float32)[np.newaxis])

    cls_img = os.path.join(data_dir, "cls.tif")
    cls_arr = np.repeat(np.arange(1, 5, dtype=np.uint8), 100).reshape(20, 20)
    with rasterio.open(cls_img, "w", count=1, dtype="uint8", nodata=0, **profile) as dst:
        dst.write(cls_arr[np.newaxis])
        dst.write_colormap(
            1,
            {
                0: (0, 0, 0, 0),
                1: (0, 114, 178, 255),
                2: (213, 94, 0, 255),
                3: (0, 158, 115, 255),
                4: (230, 159, 0, 255),
            },
        )

    polys = gpd.GeoDataFrame(
        {"val": [1.0, 2.0, 3.0]},
        geometry=[
            box(X_MIN + 10, Y_MAX - 90, X_MIN + 90, Y_MAX - 10),
            box(X_MIN + 110, Y_MAX - 90, X_MIN + 190, Y_MAX - 10),
            box(X_MIN + 10, Y_MAX - 190, X_MIN + 190, Y_MAX - 110),
        ],
        crs="EPSG:32630",
    )

    obs = rng.uniform(0.0, 100.0, 80)
    eval_df = pd.DataFrame(
        {
            "obs": obs,
            "pred": obs + rng.normal(0.0, 5.0, obs.size),
            "obs2": obs / 2.0,
            "pred2": obs / 2.0 + rng.normal(0.0, 3.0, obs.size),
        }
    )
    em_df = pd.DataFrame(
        {"Veg": [0.05, 0.08, 0.04, 0.45], "Soil": [0.15, 0.2, 0.25, 0.3]},
        index=pd.Index(["b", "g", "r", "nir"], name="band"),
    )
    vario_df = pd.DataFrame(
        {
            "distance": np.linspace(10.0, 100.0, 10),
            "semivariance": np.linspace(0.2, 1.0, 10),
            "count": np.arange(10, 20),
        }
    )
    corr_df = pd.DataFrame(rng.normal(size=(50, 4)), columns=list("abcd")).corr()
    return {
        "cont_img": cont_img,
        "dem_img": dem_img,
        "cls_img": cls_img,
        "polys": polys,
        "eval_df": eval_df,
        "em_df": em_df,
        "vario_df": vario_df,
        "corr_df": corr_df,
        "vals": rng.normal(0.0, 1.0, 300),
    }


def _fpr_tpr():
    fpr = np.array([0.0, 0.1, 0.3, 1.0])
    tpr = np.array([0.0, 0.6, 0.9, 1.0])
    return fpr, tpr


# Each case: (id, number of axes used, callable(data, ax, out_plot_file)).
# ``ax`` is one axis, or a list of axes when more than one is used.
FIGURE_CASES = [
    (
        "plot_histogram_threshold",
        1,
        lambda d, ax, out: rplot.plot_histogram_threshold(
            d["vals"], 0.5, ax=ax, out_plot_file=out
        ),
    ),
    (
        "plot_roc_curve",
        1,
        lambda d, ax, out: sdm.plot_roc_curve(*_fpr_tpr(), 0.8, ax=ax, out_plot_file=out),
    ),
    (
        "plot_endmembers",
        1,
        lambda d, ax, out: em.plot_endmembers(d["em_df"], ax=ax, out_plot_file=out),
    ),
    (
        "plot_semivariogram",
        2,
        lambda d, ax, out: rstats.plot_semivariogram(
            d["vario_df"], axs=ax, out_plot_file=out
        ),
    ),
    (
        "plot_pred_vs_obs",
        1,
        lambda d, ax, out: rm.plot_pred_vs_obs(
            d["eval_df"], ref_col="obs", pred_col="pred", axs=ax, out_plot_file=out
        ),
    ),
    (
        "plot_residuals",
        2,
        lambda d, ax, out: rm.plot_residuals(
            d["eval_df"], ref_col="obs", pred_col="pred", axs=ax, out_plot_file=out
        ),
    ),
    (
        "plot_residuals_no_hist",
        1,
        lambda d, ax, out: rm.plot_residuals(
            d["eval_df"], ref_col="obs", pred_col="pred", show_hist=False, axs=ax,
            out_plot_file=out,
        ),
    ),
    (
        "plot_qq_residuals",
        1,
        lambda d, ax, out: rm.plot_qq_residuals(
            d["eval_df"], ref_col="obs", pred_col="pred", axs=ax, out_plot_file=out
        ),
    ),
    (
        "plot_qq",
        1,
        lambda d, ax, out: rplot.plot_qq(d["vals"], ax=ax, out_plot_file=out),
    ),
    (
        "plot_correlation_heatmap",
        1,
        lambda d, ax, out: ml_explain.plot_correlation_heatmap(
            d["corr_df"], ax=ax, out_plot_file=out
        ),
    ),
    (
        "plot_jackknife_bars",
        1,
        lambda d, ax, out: ml_explain.plot_jackknife_bars(
            ["a", "b"], [0.7, 0.6], [0.5, 0.4], 0.8, "Jackknife", "Accuracy",
            ax=ax, out_plot_file=out,
        ),
    ),
    (
        "plot_feature_importance_bars",
        1,
        lambda d, ax, out: ml_explain.plot_feature_importance_bars(
            ["a", "b", "c"], [0.2, 0.5, 0.3], ax=ax, out_plot_file=out
        ),
    ),
]


def _check_fig_axes(fig, axes, n_axes):
    assert isinstance(fig, Figure)
    ax_list = np.asarray(axes, dtype=object).ravel().tolist() if not isinstance(
        axes, Axes
    ) else [axes]
    assert len(ax_list) >= n_axes
    for ax in ax_list:
        assert isinstance(ax, Axes)
        assert ax.get_figure() is fig
    return ax_list


@pytest.mark.parametrize(
    "case_id,n_axes,func", FIGURE_CASES, ids=[c[0] for c in FIGURE_CASES]
)
def test_figure_function_convention(plot_data, tmp_path, case_id, n_axes, func):
    """Return (fig, ax/axs), draw on a given axis, save only when asked, never close."""
    # pylint: disable=unused-argument
    plt.close("all")

    # 1. A new figure, nothing saved.
    result = func(plot_data, None, None)
    assert isinstance(result, tuple) and len(result) == 2
    fig, axes = result
    _check_fig_axes(fig, axes, n_axes)
    assert plt.get_fignums() == [fig.number]
    assert not os.listdir(tmp_path)
    plt.close(fig)

    # 2. Draw on given axes: no new figure, the given figure is returned.
    given_fig, given_axs = plt.subplots(n_axes, 1, squeeze=False)
    given = given_axs[0, 0] if n_axes == 1 else list(given_axs[:, 0])
    fig2, axes2 = func(plot_data, given, None)
    assert fig2 is given_fig
    ax_list = _check_fig_axes(fig2, axes2, n_axes)
    assert set(ax_list) <= set(given_axs.ravel().tolist())
    assert plt.get_fignums() == [given_fig.number]

    # 3. Save when out_plot_file is given; neither figure is closed.
    out_file = os.path.join(tmp_path, "sub", f"{case_id}.png")
    fig3, _ = func(plot_data, None, out_file)
    assert os.path.isfile(out_file) and os.path.getsize(out_file) > 0
    assert plt.fignum_exists(fig3.number)
    assert plt.fignum_exists(given_fig.number)
    plt.close("all")


# Each case: (id, callable(data, ax) -> result, check(result, ax)).
AXIS_CASES = [
    (
        "create_raster_img_map",
        lambda d, ax: map_raster.create_raster_img_map(
            ax, d["cont_img"], [1, 2, 3], rsgispy.IMG_STRETCH_LINEAR
        ),
        lambda res, ax: isinstance(res, AxesImage) and res in ax.images,
    ),
    (
        "create_raster_cmap_img_map",
        lambda d, ax: map_raster.create_raster_cmap_img_map(
            ax, d["cont_img"], 1, show_colour_bar=True, cbar_label="Value"
        ),
        lambda res, ax: isinstance(res, AxesImage) and res.colorbar is not None,
    ),
    (
        "create_thematic_raster_map",
        lambda d, ax: map_raster.create_thematic_raster_map(ax, d["cls_img"]),
        lambda res, ax: isinstance(res, AxesImage) and res in ax.images,
    ),
    (
        "create_thematic_raster_map_list",
        lambda d, ax: map_raster.create_thematic_raster_map(
            ax, [d["cls_img"], d["cls_img"]]
        ),
        lambda res, ax: isinstance(res, list)
        and len(res) == 2
        and all(isinstance(img, AxesImage) for img in res),
    ),
    (
        "add_hillshade",
        lambda d, ax: map_raster.add_hillshade(ax, dem_img=d["dem_img"]),
        lambda res, ax: isinstance(res, AxesImage) and res in ax.images,
    ),
    (
        "create_shaded_cmap_img_map",
        lambda d, ax: map_raster.create_shaded_cmap_img_map(
            ax, d["cont_img"], dem_img=d["dem_img"], cbar_label="Value"
        ),
        lambda res, ax: isinstance(res, AxesImage)
        and res is ax.images[0]
        and res.colorbar is not None,
    ),
    (
        "create_vec_lyr_map",
        lambda d, ax: map_vector.create_vec_lyr_map(ax, d["polys"], BBOX),
        lambda res, ax: isinstance(res, list)
        and len(res) >= 1
        and all(isinstance(c, Collection) for c in res),
    ),
    (
        "create_choropleth_vec_lyr_map",
        lambda d, ax: map_vector.create_choropleth_vec_lyr_map(
            ax, d["polys"], "val", BBOX, show_colour_bar=True, cbar_label="Value"
        ),
        lambda res, ax: isinstance(res, Collection) and res.colorbar is not None,
    ),
    (
        "add_scale_bar",
        lambda d, ax: map_elements.add_scale_bar(ax, epsg_code=32630, bbox=BBOX),
        lambda res, ax: isinstance(res, map_elements.MapScaleBar),
    ),
    (
        "add_north_arrow",
        lambda d, ax: map_elements.add_north_arrow(ax),
        lambda res, ax: isinstance(res, map_elements.MapNorthArrow),
    ),
    (
        "draw_bboxes_to_axis",
        lambda d, ax: map_elements.draw_bboxes_to_axis(ax, [BBOX], bbox_labels=["A"]),
        lambda res, ax: len(res) == 1 and isinstance(res[0], Rectangle),
    ),
    (
        "add_axis_label",
        lambda d, ax: map_elements.add_axis_label(ax, "(a)"),
        lambda res, ax: isinstance(res, Text) and res.get_text() == "(a)",
    ),
    (
        "add_categorical_legend",
        lambda d, ax: map_legend.add_categorical_legend(
            ax, classes_info={1: "A", 2: "B"}, input_img=d["cls_img"]
        ),
        lambda res, ax: isinstance(res, Legend),
    ),
]


@pytest.mark.parametrize("case_id,func,check", AXIS_CASES, ids=[c[0] for c in AXIS_CASES])
def test_axis_function_convention(plot_data, case_id, func, check):
    """Map functions draw on the given axis, create no figure and return their artist."""
    # pylint: disable=unused-argument
    plt.close("all")
    fig, ax = plt.subplots()
    res = func(plot_data, ax)
    assert check(res, ax)
    assert plt.get_fignums() == [fig.number]
    plt.close(fig)


def test_save_plot_and_fig_helpers(tmp_path):
    """save_plot creates directories and leaves the figure open unless asked."""
    fig, ax = rplot.get_fig_ax(None, figsize=(3, 2))
    assert ax.get_figure() is fig
    assert rplot.get_fig_ax(ax) == (fig, ax)
    out = rplot.save_plot(fig, tmp_path / "a" / "b.png", out_dpi=50)
    assert os.path.isfile(out) and plt.fignum_exists(fig.number)
    rplot.save_plot(fig, tmp_path / "c.png", out_dpi=50, close=True)
    assert not plt.fignum_exists(fig.number)

    fig, axs = rplot.get_fig_axs(None, nrows=2, ncols=3)
    assert axs.shape == (2, 3)
    fig2, axs2 = rplot.get_fig_axs(list(axs.ravel()), nrows=3, ncols=2)
    assert fig2 is fig and axs2.shape == (3, 2)
    with pytest.raises(rsgispy.RSGISPyException):
        rplot.get_fig_axs(list(axs.ravel()), nrows=1, ncols=2)

    txt = rplot.add_stats_text_box(axs[0, 0], ["RMSE = 1.0", "n = 5"])
    assert txt.txt.get_text() == "RMSE = 1.0\nn = 5"
    plt.close("all")


def test_batch_plot_functions_close_their_figures(tmp_path):
    """Functions that write one figure per feature close each one and return the files."""
    plt.close("all")
    data_df = pd.DataFrame(
        {"a": np.arange(20.0), "b": np.arange(20.0) ** 2, "cls": [1, 2] * 10}
    )
    out_files = ml_explain.plot_class_feature_distributions(
        data_df, ["a", "b"], "cls", out_dir=tmp_path, use_pbar=False
    )
    assert len(out_files) == 2 and all(os.path.isfile(f) for f in out_files)
    assert not plt.get_fignums()


import rsgispy  # noqa: E402  pylint: disable=wrong-import-position
