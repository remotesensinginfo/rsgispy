rsgispy.regression.regxgboost
=============================

Landsat Forest Canopy Cover Example
-----------------------------------

The following example tunes, trains, and applies an XGBoost regressor to predict
forest canopy cover percentage (``canopy_cover``) from 6-band Landsat surface
reflectance imagery and reference estimates in a vector layer. Multi-target
(multivariate) regression is also supported natively by specifying multiple
columns in ``target_cols`` (for example, ``["canopy_cover", "canopy_height"]``).

.. code-block:: python

    import rsgispy.regression.regdata as rd
    import rsgispy.regression.regxgboost as rxgb

    # 1. Define Landsat bands and target canopy cover vector layer
    img_band_info = [
        rd.ImageBandInfo(
            file_name="landsat_sr.tif",
            name="lsat",
            bands=[1, 2, 3, 4, 5, 6],
            band_names=["blue", "green", "red", "nir", "swir1", "swir2"],
        )
    ]
    reg_cfg = rd.RegressionConfig(
        img_band_info=img_band_info,
        target_cols="canopy_cover",
        vec_file="canopy_cover_samples.gpkg",
        vec_lyr="canopy_cover_samples",
    )

    # 2. Extract and split samples into train / test / validation sets
    train_df, test_df, valid_df = rd.extract_and_split_reg_samples(
        reg_cfg,
        test_prop=0.2,
        valid_prop=0.2,
        target_min=0.0,
        target_max=100.0,
        rnd_seed=42,
    )

    # 3. Optimise XGBoost parameters using Bayesian updating (Optuna)
    _, opt_res = rxgb.optimise_xgboost_regressor(
        reg_cfg=reg_cfg,
        train_data=train_df,
        valid_data=valid_df,
        early_stopping_rounds=20,
        scoring="r2",
        n_trials=30,
        refit=False,
        out_params_json="xgb_best_params.json",
        return_results=True,
        rnd_seed=42,
    )

    # 4. Train and evaluate the optimised regressor
    model, metrics = rxgb.train_xgboost_regressor(
        reg_cfg=reg_cfg,
        train_data=train_df,
        test_data=test_df,
        valid_data=valid_df,
        model_params=opt_res["best_params"],
        early_stopping_rounds=20,
        out_model_file="landsat_canopy_xgb.json",
        out_metrics_file="landsat_canopy_xgb_metrics.json",
        rnd_seed=42,
    )

    # 5. Apply the model to predict canopy cover across the Landsat scene
    rxgb.apply_xgboost_regressor(
        reg_cfg=reg_cfg,
        xgb_regressor=model,
        in_msk_img="landsat_valid_mask.tif",
        img_msk_val=1,
        output_img="landsat_canopy_cover_xgb.tif",
    )

Optuna Hyperparameter Search Spaces
-----------------------------------

By default (``param_space=None``, ``extended_param_space=False``),
:func:`~rsgispy.regression.regxgboost.optimise_xgboost_regressor` optimises 9
core XGBoost parameters (``n_estimators``, ``learning_rate``, ``max_depth``,
``min_child_weight``, ``gamma``, ``subsample``, ``colsample_bytree``,
``reg_alpha``, and ``reg_lambda``). Passing ``extended_param_space=True`` adds
``colsample_bylevel``, ``colsample_bynode``, ``max_delta_step``, ``max_bin``,
``grow_policy``, and ``max_leaves``. Both spaces can also be inspected via
:func:`~rsgispy.regression.regxgboost.get_xgboost_optuna_param_space`.

GPU Acceleration
----------------

All optimise, train, and apply functions accept ``use_gpu=True`` and an optional
``gpu_device_id`` index, checked via
:func:`~rsgispy.regression.regxgboost.is_xgboost_gpu_available`. When no
compatible CUDA GPU is available, execution emits a ``UserWarning`` and falls
back cleanly to CPU.

API Reference
-------------

.. automodule:: rsgispy.regression.regxgboost
   :members:
   :undoc-members:
   :show-inheritance:
