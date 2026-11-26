"""
Tests for rsgispy.regression.regmetrics: raster point prediction sampling,
univariate & multivariate regression accuracy metrics, bootstrap confidence
intervals, and predicted-vs-observed / residual plots.
"""

from pathlib import Path

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point, box

import rsgispy
import rsgispy.regression.regmetrics as rrm

matplotlib.use("Agg")


def _write_pred_raster(tmp_path: Path):
    """Write 2-band synthetic prediction raster with nodata & descriptions."""
    b1 = np.full((10, 10), 45.0, dtype=np.float32)
    b2 = np.full((10, 10), 18.0, dtype=np.float32)
    for row in range(10):
        b1[row, :] = 10.0 + 8.0 * row
        b2[row, :] = 5.0 + 2.0 * row
    b1[0, 0] = -9999.0
    b2[0, 0] = -9999.0

    pred_path = tmp_path / "pred_forest.tif"
    with rasterio.open(
        pred_path,
        "w",
        driver="GTiff",
        width=10,
        height=10,
        count=2,
        dtype="float32",
        crs="EPSG:32630",
        transform=from_origin(500000.0, 5600000.0, 10.0, 10.0),
        nodata=-9999.0,
    ) as dst:
        dst.write(np.stack([b1, b2]))
        dst.set_band_description(1, "canopy_cover")
        dst.set_band_description(2, "canopy_height")
    return pred_path


def test_pop_reg_img_pred_pts_and_metrics(tmp_path: Path):
    """Test raster prediction sampling at points and univariate/multivariate metrics."""
    pred_tif = _write_pred_raster(tmp_path)

    pts = []
    ref_cover = []
    ref_height = []
    for row in range(1, 9):
        px_x = 500005.0 + row * 10.0
        px_y = 5599995.0 - row * 10.0
        pts.append(Point(px_x, px_y))
        ref_cover.append(10.0 + 8.0 * row + (0.5 if row % 2 == 0 else -0.5))
        ref_height.append(5.0 + 2.0 * row + (0.2 if row % 2 == 0 else -0.2))

    # Add nodata pixel location [0, 0] to verify NaN handling
    pts.append(Point(500005.0, 5599995.0))
    ref_cover.append(10.0)
    ref_height.append(5.0)

    pts_vec = tmp_path / "val_pts.gpkg"
    gpd.GeoDataFrame(
        {"canopy_cover": ref_cover, "canopy_height": ref_height},
        geometry=pts,
        crs="EPSG:32630",
    ).to_file(pts_vec, layer="val_pts", driver="GPKG")

    # Sample both bands into GeoDataFrame
    pop_vec = tmp_path / "val_pts_pop.gpkg"
    rrm.pop_reg_img_pred_pts(
        input_reg_img=pred_tif,
        pts_vec_file=pts_vec,
        pts_vec_lyr="val_pts",
        out_vec_file=pop_vec,
        out_vec_lyr="val_pts",
        use_pbar=False,
    )
    assert pop_vec.exists()
    pop_gdf = gpd.read_file(pop_vec, layer="val_pts")
    assert "pred_canopy_cover" in pop_gdf.columns
    assert "pred_canopy_height" in pop_gdf.columns
    assert np.isnan(pop_gdf["pred_canopy_cover"].iloc[-1])

    # Univariate metrics with JSON and CSV output
    uni_json = tmp_path / "uni_metrics.json"
    uni_csv = tmp_path / "uni_metrics.csv"
    uni_stats = rrm.calc_reg_metrics(
        in_data=pop_vec,
        vec_lyr="val_pts",
        ref_col="canopy_cover",
        pred_col="pred_canopy_cover",
        out_json_file=uni_json,
        out_csv_file=uni_csv,
    )
    assert uni_json.exists() and uni_csv.exists()
    assert uni_stats["n_samples"] == 8  # nodata point automatically filtered
    assert uni_stats["r2"] > 0.99
    assert uni_stats["rmse"] == pytest.approx(0.5)
    assert uni_stats["mae"] == pytest.approx(0.5)

    # Multivariate metrics & CSV
    mv_csv = tmp_path / "mv_metrics.csv"
    mv_stats = rrm.calc_reg_metrics(
        in_data=pop_gdf,
        ref_cols=["canopy_cover", "canopy_height"],
        pred_cols=["pred_canopy_cover", "pred_canopy_height"],
        out_csv_file=mv_csv,
    )
    assert mv_csv.exists()
    assert mv_stats["canopy_cover_rmse"] == pytest.approx(0.5)
    assert mv_stats["canopy_height_rmse"] == pytest.approx(0.2)

    # Non-point geometry error
    poly_gdf = gpd.GeoDataFrame(
        {"a": [1]}, geometry=[box(500000, 5599900, 500050, 5599950)], crs="EPSG:32630"
    )
    with pytest.raises(rsgispy.RSGISPyException):
        rrm.pop_reg_img_pred_pts(pred_tif, poly_gdf, use_pbar=False)


def test_bootstrap_conf_and_diagnostic_plots(tmp_path: Path):
    """Test bootstrap confidence intervals and Pred-vs-Obs / residual plots."""
    rng = np.random.default_rng(7)
    n = 50
    obs1 = np.linspace(5.0, 95.0, n)
    pred1 = obs1 + rng.normal(0.0, 2.0, n)
    obs2 = np.linspace(2.0, 30.0, n)
    pred2 = obs2 + rng.normal(0.2, 0.8, n)

    eval_df = pd.DataFrame(
        {
            "obs_cover": obs1,
            "pred_cover": pred1,
            "obs_height": obs2,
            "pred_height": pred2,
        }
    )

    ci_json = tmp_path / "bs_ci.json"
    ci_stats = rrm.calc_reg_stats_ptonly_bootstrap_conf(
        in_data=eval_df,
        ref_cols=["obs_cover", "obs_height"],
        pred_cols=["pred_cover", "pred_height"],
        target_names=["cover", "height"],
        out_json_file=ci_json,
        sample_frac=0.5,
        bootstrap_n=40,
        rnd_seed=42,
        use_pbar=False,
    )
    assert ci_json.exists()
    assert "r2" in ci_stats
    assert "cover_rmse" in ci_stats
    assert "95th" in ci_stats["cover_rmse"]

    pvo_uni = tmp_path / "pvo_uni.png"
    res_uni = tmp_path / "res_uni.png"
    pvo_mv = tmp_path / "pvo_mv.png"
    res_mv = tmp_path / "res_mv.png"

    m1 = rrm.calc_reg_metrics(eval_df, ref_col="obs_cover", pred_col="pred_cover")
    fig1, axs1 = rrm.plot_pred_vs_obs(
        eval_df, ref_col="obs_cover", pred_col="pred_cover", out_plot_file=pvo_uni
    )
    fig2, axs2 = rrm.plot_residuals(
        eval_df, ref_col="obs_cover", pred_col="pred_cover", out_plot_file=res_uni
    )
    fig3, axs3 = rrm.plot_pred_vs_obs(
        eval_df,
        ref_cols=["obs_cover", "obs_height"],
        pred_cols=["pred_cover", "pred_height"],
        target_names=["Canopy Cover", "Canopy Height"],
        out_plot_file=pvo_mv,
    )
    fig4, axs4 = rrm.plot_residuals(
        eval_df,
        ref_cols=["obs_cover", "obs_height"],
        pred_cols=["pred_cover", "pred_height"],
        target_names=["Canopy Cover", "Canopy Height"],
        out_plot_file=res_mv,
    )
    assert all(p.exists() for p in (pvo_uni, res_uni, pvo_mv, res_mv))
    assert m1["r2"] > 0.9
    assert axs1.shape == (1, 1) and axs2.shape[0] == 1
    assert axs3.size >= 2 and axs4.size >= 2
    for fig in (fig1, fig2, fig3, fig4):
        plt.close(fig)
