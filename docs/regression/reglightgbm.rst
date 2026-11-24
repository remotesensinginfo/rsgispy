rsgispy.regression.reglightgbm
==============================

Landsat Forest Canopy Cover Example
-----------------------------------

The following example tunes, trains, and applies a LightGBM regressor to predict
forest canopy cover percentage (``canopy_cover``) from 6-band Landsat imagery.
When multiple targets are specified in ``target_cols``, LightGBM automatically
wraps per-target boosters via ``MultiOutputRegressor`` while supporting early
stopping per target.

.. code-block:: python

    import rsgispy.regression.regdata as rd
    import rsgispy.regression.reglightgbm as rlgb

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
    _, opt_res = rlgb.optimise_lightgbm_regressor(
        reg_cfg=reg_cfg,
        train_data=train_df,
        valid_data=valid_df,
        early_stopping_rounds=20,
        n_trials=30,
        refit=False,
        return_results=True,
        rnd_seed=42,
    )
    model, metrics = rlgb.train_lightgbm_regressor(
        reg_cfg=reg_cfg,
        train_data=train_df,
        test_data=test_df,
        valid_data=valid_df,
        model_params=opt_res["best_params"],
        early_stopping_rounds=20,
        out_model_file="landsat_canopy_lgb.joblib",
        rnd_seed=42,
    )
    rlgb.apply_lightgbm_regressor(
        reg_cfg=reg_cfg,
        lgb_regressor=model,
        in_msk_img="landsat_valid_mask.tif",
        img_msk_val=1,
        output_img="landsat_canopy_cover_lgb.tif",
    )

Optuna Hyperparameter Search Spaces
-----------------------------------

The default search space tunes 10 core parameters (``n_estimators``,
``learning_rate``, ``num_leaves``, ``max_depth``, ``min_child_samples``,
``min_split_gain``, ``subsample``, ``colsample_bytree``, ``reg_alpha``, and
``reg_lambda`` with ``subsample_freq=1``). Passing ``extended_param_space=True``
enables wider ranges plus ``min_child_weight``, ``subsample_freq``, ``max_bin``,
``subsample_for_bin``, ``path_smooth``, and ``extra_trees``. Inspect spaces with
:func:`~rsgispy.regression.reglightgbm.get_lightgbm_optuna_param_space`.

API Reference
-------------

.. automodule:: rsgispy.regression.reglightgbm
   :members:
   :undoc-members:
   :show-inheritance:
