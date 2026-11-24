rsgispy.classification.classxgboost
===================================

Landsat Classification Example
------------------------------

The following example classifies a 6-band Landsat surface reflectance scene into
``Mangroves`` (``out_id=10``), ``Water`` (``out_id=20``), and ``Other``
(``out_id=30``) classes. Training pixels are extracted from per-class vector
files, hyperparameters are tuned via Bayesian optimisation (Optuna TPE), and the
final XGBoost model is trained, evaluated, and applied to the Landsat scene.

.. code-block:: python

    import rsgispy.classification.classdata as cd
    import rsgispy.classification.classxgboost as cxgb

    # 1. Define Landsat bands and classes with vector training samples
    img_band_info = [
        cd.ImageBandInfo(
            file_name="landsat_sr.tif",
            name="lsat",
            bands=[1, 2, 3, 4, 5, 6],
            band_names=["blue", "green", "red", "nir", "swir1", "swir2"],
        )
    ]
    class_info = {
        "Mangroves": cd.ClassInfoObj(
            out_id=10, vec_file="mangroves_samples.gpkg", red=0, green=150, blue=60
        ),
        "Water": cd.ClassInfoObj(
            out_id=20, vec_file="water_samples.gpkg", red=0, green=90, blue=220
        ),
        "Other": cd.ClassInfoObj(
            out_id=30, vec_file="other_samples.gpkg", red=180, green=140, blue=80
        ),
    }
    cls_cfg = cd.ClassificationConfig(
        img_band_info=img_band_info,
        class_info=class_info,
    )

    # 2. Extract and split samples into train / test / validation sets
    train_df, test_df, valid_df = cd.extract_and_split_class_samples(
        cls_cfg,
        test_prop=0.2,
        valid_prop=0.2,
        out_train_file="train_samples.parquet",
        out_test_file="test_samples.parquet",
        out_valid_file="valid_samples.parquet",
        rnd_seed=42,
    )

    # 3. Optimise XGBoost parameters using Bayesian updating (Optuna)
    _, opt_res = cxgb.optimise_xgboost_multiclass_classifier(
        cls_cfg=cls_cfg,
        train_data=train_df,
        valid_data=valid_df,
        early_stopping_rounds=20,
        scoring="f1_macro",
        n_trials=30,
        refit=False,
        out_params_json="xgb_best_params.json",
        return_results=True,
        rnd_seed=42,
    )

    # 4. Train and evaluate the optimised classifier
    model, metrics = cxgb.train_xgboost_multiclass_classifier(
        cls_cfg=cls_cfg,
        train_data=train_df,
        test_data=test_df,
        valid_data=valid_df,
        model_params=opt_res["best_params"],
        early_stopping_rounds=20,
        out_model_file="landsat_xgb_model.json",
        out_metrics_file="landsat_xgb_metrics.json",
        rnd_seed=42,
    )

    # 5. Apply the model to classify the Landsat scene (outputs IDs 10, 20, 30)
    cxgb.apply_xgboost_multiclass_classifier(
        cls_cfg=cls_cfg,
        xgb_classifier=model,
        in_msk_img="landsat_valid_mask.tif",
        img_msk_val=1,
        output_img="landsat_xgb_cls.tif",
        out_score_img="landsat_xgb_probs.tif",
    )

Optuna Hyperparameter Search Spaces
-----------------------------------

By default (``param_space=None``, ``extended_param_space=False``),
:func:`~rsgispy.classification.classxgboost.optimise_xgboost_binary_classifier`
and :func:`~rsgispy.classification.classxgboost.optimise_xgboost_multiclass_classifier`
optimise the 9 core XGBoost parameters listed below. Passing
``extended_param_space=True`` (or ``param_space='extended'``) enables the
expanded 15-parameter multi-class / 16-parameter binary search space. Both spaces
can also be inspected via
:func:`~rsgispy.classification.classxgboost.get_xgboost_optuna_param_space`.

Default Search Space (Core Key Parameters)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 18 22 40

   * - Parameter
     - Type / Scale
     - Range / Choices
     - Description
   * - ``n_estimators``
     - int (step 25)
     - ``50`` to ``500``
     - Number of boosting rounds (trees).
   * - ``learning_rate``
     - float (log)
     - ``0.01`` to ``0.3``
     - Boosting step size shrinkage (``eta``).
   * - ``max_depth``
     - int
     - ``3`` to ``10``
     - Maximum depth of each decision tree.
   * - ``min_child_weight``
     - int
     - ``1`` to ``5``
     - Minimum sum of instance Hessian weight needed in a child.
   * - ``gamma``
     - float (log)
     - ``1e-4`` to ``1.0``
     - Minimum loss reduction required to make a further partition.
   * - ``subsample``
     - float
     - ``0.5`` to ``1.0``
     - Subsample ratio of training rows per tree.
   * - ``colsample_bytree``
     - float
     - ``0.5`` to ``1.0``
     - Subsample ratio of features when constructing each tree.
   * - ``reg_alpha``
     - float (log)
     - ``1e-4`` to ``1.0``
     - L1 regularisation term on leaf weights.
   * - ``reg_lambda``
     - float (log)
     - ``1e-4`` to ``10.0``
     - L2 regularisation term on leaf weights.

Extended Search Space (``extended_param_space=True``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 18 22 40

   * - Parameter
     - Type / Scale
     - Range / Choices
     - Description
   * - ``n_estimators``
     - int (step 25)
     - ``50`` to ``1000``
     - Wider boosting round range for slower learning rates.
   * - ``learning_rate``
     - float (log)
     - ``0.005`` to ``0.3``
     - Finer minimum step size shrinkage.
   * - ``max_depth``
     - int
     - ``3`` to ``12``
     - Deeper maximum tree depth range.
   * - ``min_child_weight``
     - float (log)
     - ``0.1`` to ``10.0``
     - Continuous Hessian leaf threshold.
   * - ``gamma``
     - float (log)
     - ``1e-4`` to ``5.0``
     - Wider split-pruning regularisation range.
   * - ``subsample``
     - float
     - ``0.5`` to ``1.0``
     - Row subsampling ratio per tree.
   * - ``colsample_bytree``
     - float
     - ``0.4`` to ``1.0``
     - Feature subsampling ratio per tree.
   * - ``colsample_bylevel``
     - float
     - ``0.4`` to ``1.0``
     - Feature subsampling ratio for each tree depth level.
   * - ``colsample_bynode``
     - float
     - ``0.4`` to ``1.0``
     - Feature subsampling ratio at each individual split node.
   * - ``reg_alpha``
     - float (log)
     - ``1e-5`` to ``5.0``
     - Wider L1 weight regularisation range.
   * - ``reg_lambda``
     - float (log)
     - ``1e-5`` to ``20.0``
     - Wider L2 weight regularisation range.
   * - ``max_delta_step``
     - int
     - ``0`` to ``10``
     - Maximum delta step per leaf output (helps imbalanced classes).
   * - ``max_bin``
     - int (step 64)
     - ``64`` to ``512``
     - Maximum histogram bins for continuous feature discretisation.
   * - ``grow_policy``
     - categorical
     - ``['depthwise', 'lossguide']``
     - Tree construction order (closest-to-root vs highest-loss-gain leaf).
   * - ``max_leaves``
     - int
     - ``0`` to ``128``
     - Maximum number of leaves (``0`` = unlimited, used with ``lossguide``).
   * - ``scale_pos_weight``
     - float (log)
     - ``0.5`` to ``10.0``
     - Positive-class gradient weight scaling (**binary tasks only**).

GPU Acceleration
----------------

All optimise, train, and apply functions accept ``use_gpu=True`` and an optional
``gpu_device_id`` index. GPU support is checked via
:func:`~rsgispy.classification.classxgboost.is_xgboost_gpu_available`. When a
CUDA-capable GPU is available, ``tree_method='hist'`` and ``device='cuda'``
(or ``'cuda:<gpu_device_id>'``) are configured automatically; when no compatible
GPU is detected, a ``UserWarning`` is emitted and execution falls back to CPU.

API Reference
-------------

.. automodule:: rsgispy.classification.classxgboost
   :members:
   :undoc-members:
   :show-inheritance:
