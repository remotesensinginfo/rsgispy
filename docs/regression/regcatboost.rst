rsgispy.regression.regcatboost
==============================

Landsat Forest Canopy Cover Example
-----------------------------------

The following example tunes, trains, and applies a CatBoost regressor to predict
forest canopy cover percentage (``canopy_cover``) from 6-band Landsat imagery.
When multiple targets are specified in ``target_cols``, CatBoost automatically
uses ``loss_function="MultiRMSE"`` for native multi-output regression.

.. code-block:: python

    import rsgispy.regression.regdata as rd
    import rsgispy.regression.regcatboost as rcb

    reg_cfg = rd.RegressionConfig(
        img_band_info=[
            rd.ImageBandInfo(
                file_name="landsat_sr.tif",
                name="lsat",
                bands=[1, 2, 3, 4, 5, 6],
                band_names=["blue", "green", "red", "nir", "swir1", "swir2"],
            )
        ],
        target_cols="canopy_cover",
        vec_file="canopy_cover_samples.gpkg",
    )
    train_df, test_df, valid_df = rd.extract_and_split_reg_samples(
        reg_cfg, test_prop=0.2, valid_prop=0.2, rnd_seed=42
    )
    _, opt_res = rcb.optimise_catboost_regressor(
        reg_cfg=reg_cfg,
        train_data=train_df,
        valid_data=valid_df,
        early_stopping_rounds=20,
        n_trials=30,
        refit=False,
        return_results=True,
        rnd_seed=42,
    )
    model, metrics = rcb.train_catboost_regressor(
        reg_cfg=reg_cfg,
        train_data=train_df,
        test_data=test_df,
        valid_data=valid_df,
        model_params=opt_res["best_params"],
        early_stopping_rounds=20,
        out_model_file="landsat_canopy_cb.cbm",
        rnd_seed=42,
    )
    rcb.apply_catboost_regressor(
        reg_cfg=reg_cfg,
        cb_regressor=model,
        in_msk_img="landsat_valid_mask.tif",
        img_msk_val=1,
        output_img="landsat_canopy_cover_cb.tif",
    )

Optuna Hyperparameter Search Spaces
-----------------------------------

The default search space tunes 8 core parameters (``iterations``,
``learning_rate``, ``depth``, ``l2_leaf_reg``, ``random_strength``,
``bagging_temperature``, ``rsm``, and ``border_count``). Passing
``extended_param_space=True`` activates conditional tree/bootstrap tuning
(``bootstrap_type``, ``grow_policy``, ``boosting_type``, ``max_leaves``,
``min_data_in_leaf``, ``model_size_reg``, etc.). Inspect spaces with
:func:`~rsgispy.regression.regcatboost.get_catboost_optuna_param_space`.

API Reference
-------------

.. automodule:: rsgispy.regression.regcatboost
   :members:
   :undoc-members:
   :show-inheritance:
