rsgispy.regression.regmetrics
=============================

Accuracy Assessment & Diagnostic Plots Example
----------------------------------------------

The following example samples predicted forest canopy cover values from the
output Landsat raster at independent validation points, computes continuous
regression accuracy metrics and bootstrap confidence intervals, and creates
predicted-vs-observed and residual diagnostic plots.

.. code-block:: python

    import rsgispy.regression.regmetrics as rm

    # 1. Sample predicted raster at validation points
    rm.pop_reg_img_pred_pts(
        input_reg_img="landsat_canopy_cover_rf.tif",
        pts_vec_file="canopy_val_pts.gpkg",
        pts_vec_lyr="val_pts",
        out_vec_file="canopy_val_pts_eval.gpkg",
        out_vec_lyr="val_pts",
        pred_col="pred_canopy_cover",
    )

    # 2. Compute continuous metrics and bootstrap confidence intervals
    metrics = rm.calc_reg_metrics(
        in_data="canopy_val_pts_eval.gpkg",
        vec_lyr="val_pts",
        ref_col="canopy_cover",
        pred_col="pred_canopy_cover",
        out_json_file="canopy_val_metrics.json",
        out_csv_file="canopy_val_metrics.csv",
    )
    ci_stats = rm.calc_reg_stats_ptonly_bootstrap_conf(
        in_data="canopy_val_pts_eval.gpkg",
        vec_lyr="val_pts",
        ref_col="canopy_cover",
        pred_col="pred_canopy_cover",
        bootstrap_n=500,
        rnd_seed=42,
    )

    # 3. Plot predicted vs observed scatter and residual diagnostics;
    #    each returns (fig, axs) and saves when out_plot_file is given.
    fig, axs = rm.plot_pred_vs_obs(
        "canopy_val_pts_eval.gpkg",
        vec_lyr="val_pts",
        ref_col="canopy_cover",
        pred_col="pred_canopy_cover",
        out_plot_file="canopy_pred_vs_obs.png",
    )
    fig, axs = rm.plot_residuals(
        "canopy_val_pts_eval.gpkg",
        vec_lyr="val_pts",
        ref_col="canopy_cover",
        pred_col="pred_canopy_cover",
        out_plot_file="canopy_residuals.png",
    )

API Reference
-------------

.. automodule:: rsgispy.regression.regmetrics
   :members:
   :undoc-members:
   :show-inheritance:
