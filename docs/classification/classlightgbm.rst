rsgispy.classification.classlightgbm
====================================

Landsat Classification Example
------------------------------

The following example classifies a 6-band Landsat surface reflectance scene into
``Mangroves`` (``out_id=10``), ``Water`` (``out_id=20``), and ``Other``
(``out_id=30``) classes. Training pixels are extracted from per-class vector
files, hyperparameters are tuned via Bayesian optimisation (Optuna TPE), and the
final LightGBM model is trained, evaluated, and applied to the Landsat scene.

.. code-block:: python

    import rsgispy.classification.classdata as cd
    import rsgispy.classification.classlightgbm as clgb

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

    # 3. Optimise LightGBM parameters using Bayesian updating (Optuna)
    _, opt_res = clgb.optimise_lightgbm_multiclass_classifier(
        cls_cfg=cls_cfg,
        train_data=train_df,
        valid_data=valid_df,
        early_stopping_rounds=20,
        scoring="f1_macro",
        n_trials=30,
        refit=False,
        out_params_json="lgb_best_params.json",
        return_results=True,
        rnd_seed=42,
    )

    # 4. Train and evaluate the optimised classifier
    model, metrics = clgb.train_lightgbm_multiclass_classifier(
        cls_cfg=cls_cfg,
        train_data=train_df,
        test_data=test_df,
        valid_data=valid_df,
        model_params=opt_res["best_params"],
        early_stopping_rounds=20,
        out_model_file="landsat_lgb_model.joblib",
        out_metrics_file="landsat_lgb_metrics.json",
        rnd_seed=42,
    )

    # 5. Apply the model to classify the Landsat scene (outputs IDs 10, 20, 30)
    clgb.apply_lightgbm_multiclass_classifier(
        cls_cfg=cls_cfg,
        lgb_classifier=model,
        in_msk_img="landsat_valid_mask.tif",
        img_msk_val=1,
        out_class_img="landsat_lgb_cls.tif",
        out_score_img="landsat_lgb_probs.tif",
    )

Optuna Hyperparameter Search Spaces
-----------------------------------

By default (``param_space=None``, ``extended_param_space=False``),
:func:`~rsgispy.classification.classlightgbm.optimise_lightgbm_binary_classifier`
and :func:`~rsgispy.classification.classlightgbm.optimise_lightgbm_multiclass_classifier`
optimise the 10 core LightGBM parameters listed below (with
``subsample_freq=1`` fixed so row subsampling is active on every iteration).
Passing ``extended_param_space=True`` (or ``param_space='extended'``) enables
the expanded 17-parameter search space. Both spaces can also be inspected via
:func:`~rsgispy.classification.classlightgbm.get_lightgbm_optuna_param_space`.

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
     - Number of boosted trees.
   * - ``learning_rate``
     - float (log)
     - ``0.01`` to ``0.3``
     - Boosting shrinkage rate.
   * - ``num_leaves``
     - int
     - ``15`` to ``127``
     - Maximum number of leaves in one tree (primary complexity control).
   * - ``max_depth``
     - int
     - ``3`` to ``12``
     - Maximum tree depth limit.
   * - ``min_child_samples``
     - int
     - ``5`` to ``50``
     - Minimum number of training samples required in a leaf.
   * - ``min_split_gain``
     - float (log)
     - ``1e-4`` to ``1.0``
     - Minimum loss reduction required to split an internal node.
   * - ``subsample``
     - float
     - ``0.5`` to ``1.0``
     - Row subsampling fraction per bagging iteration.
   * - ``subsample_freq``
     - fixed
     - ``1``
     - Fixed to ``1`` so LightGBM bagging is active when ``subsample < 1.0``.
   * - ``colsample_bytree``
     - float
     - ``0.5`` to ``1.0``
     - Feature subsampling fraction per tree.
   * - ``reg_alpha``
     - float (log)
     - ``1e-4`` to ``10.0``
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
     - Wider boosting round range for finer learning rates.
   * - ``learning_rate``
     - float (log)
     - ``0.005`` to ``0.3``
     - Finer minimum boosting shrinkage rate.
   * - ``num_leaves``
     - int
     - ``8`` to ``255``
     - Expanded leaf count range for leaf-wise tree growth.
   * - ``max_depth``
     - int
     - ``3`` to ``15``
     - Expanded tree depth limit.
   * - ``min_child_samples``
     - int
     - ``3`` to ``100``
     - Expanded minimum leaf sample threshold.
   * - ``min_child_weight``
     - float (log)
     - ``1e-4`` to ``10.0``
     - Minimum sum of Hessian weights required in a leaf.
   * - ``min_split_gain``
     - float (log)
     - ``1e-4`` to ``5.0``
     - Minimum gain required to partition a leaf.
   * - ``subsample``
     - float
     - ``0.5`` to ``1.0``
     - Row subsampling fraction.
   * - ``subsample_freq``
     - int
     - ``1`` to ``7``
     - Frequency (in trees) for performing row bagging.
   * - ``colsample_bytree``
     - float
     - ``0.4`` to ``1.0``
     - Feature subsampling fraction per tree.
   * - ``reg_alpha``
     - float (log)
     - ``1e-5`` to ``20.0``
     - Wider L1 leaf weight regularisation.
   * - ``reg_lambda``
     - float (log)
     - ``1e-5`` to ``20.0``
     - Wider L2 leaf weight regularisation.
   * - ``max_bin``
     - int (step 64)
     - ``63`` to ``511``
     - Maximum histogram bins for continuous feature values.
   * - ``subsample_for_bin``
     - int (step 20000)
     - ``20000`` to ``300000``
     - Number of samples used to construct feature histograms.
   * - ``path_smooth``
     - float
     - ``0.0`` to ``10.0``
     - Tree-path smoothing regularisation to reduce overfitting on small leaves.
   * - ``extra_trees``
     - categorical
     - ``[False, True]``
     - Randomised split thresholds (Extremely Randomised Trees mode).
   * - ``class_weight``
     - categorical
     - ``[None, 'balanced']``
     - Inverse-frequency class weighting for imbalanced datasets.

GPU Acceleration
----------------

All optimise, train, and apply functions accept ``use_gpu=True`` and an optional
``gpu_device_id`` index. GPU support is checked via
:func:`~rsgispy.classification.classlightgbm.is_lightgbm_gpu_available` (probing
CUDA when NVIDIA drivers are present, or OpenCL ``device='gpu'`` otherwise).
When OpenCL ``device='gpu'`` is active, ``max_bin`` is automatically clamped to
at most ``255`` for compatibility. If no compatible GPU is detected, a
``UserWarning`` is emitted and execution falls back to CPU.

API Reference
-------------

.. automodule:: rsgispy.classification.classlightgbm
   :members:
   :undoc-members:
   :show-inheritance:
