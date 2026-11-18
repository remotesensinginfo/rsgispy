rsgispy.regression.regfeatures
==============================

Response Function Curves & Feature Analysis Example
---------------------------------------------------

The example below visualises predictor-to-target relationships, checks collinearity,
selects non-redundant bands, and computes response function curves relating predicted
forest canopy cover (``canopy_cover``) to the Landsat spectral bands.

.. code-block:: python

    import rsgispy.regression.regfeatures as rf

    # 1. Scatter diagnostics & target correlation matrix
    rf.target_scatter_plots("train_samples.parquet", reg_cfg=reg_cfg, out_dir="scatter_plots")
    rf.correlation_matrix("train_samples.parquet", reg_cfg=reg_cfg, include_targets=True)

    # 2. Compute feature importance and marginal / partial-dependence response curves
    imp_df = rf.calc_feature_importance(reg_cfg, fitted_rf, "test_samples.parquet", method="both")
    resp_df = rf.calc_feature_response_curves(
        reg_cfg=reg_cfg,
        sk_regressor=fitted_rf,
        train_data="train_samples.parquet",
        method="pdp",
        plot_samples=True,
        out_file="canopy_response_curves.csv",
        response_plots_dir="response_plots",
        out_grid_plot="canopy_response_matrix.png",
    )

API Reference
-------------

.. automodule:: rsgispy.regression.regfeatures
   :members:
   :undoc-members:
   :show-inheritance:
