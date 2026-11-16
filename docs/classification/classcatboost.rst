rsgispy.classification.classcatboost
====================================

Optuna Hyperparameter Search Spaces
-----------------------------------

By default (``param_space=None``, ``extended_param_space=False``),
:func:`~rsgispy.classification.classcatboost.optimise_catboost_binary_classifier`
and :func:`~rsgispy.classification.classcatboost.optimise_catboost_multiclass_classifier`
optimise the 8 core CatBoost parameters listed below. Passing
``extended_param_space=True`` (or ``param_space='extended'``) activates a larger
conditional search space that respects CatBoost parameter compatibility rules
(pairing ``bagging_temperature`` with ``Bayesian`` bootstrap, ``subsample``
with ``Bernoulli``/``MVS``, ``boosting_type`` with ``SymmetricTree``, and
``max_leaves`` with ``Lossguide``). Both spaces can be retrieved via
:func:`~rsgispy.classification.classcatboost.get_catboost_optuna_param_space`.

Default Search Space (Core Key Parameters)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 18 22 40

   * - Parameter
     - Type / Scale
     - Range / Choices
     - Description
   * - ``iterations``
     - int (step 25)
     - ``50`` to ``500``
     - Maximum number of oblivious/decision trees.
   * - ``learning_rate``
     - float (log)
     - ``0.01`` to ``0.3``
     - Gradient boosting step size.
   * - ``depth``
     - int
     - ``3`` to ``8``
     - Tree depth.
   * - ``l2_leaf_reg``
     - float (log)
     - ``1.0`` to ``10.0``
     - L2 regularisation coefficient of the cost function.
   * - ``random_strength``
     - float (log)
     - ``1e-3`` to ``10.0``
     - Randomness added when scoring split candidates to prevent overfitting.
   * - ``bagging_temperature``
     - float
     - ``0.0`` to ``1.0``
     - Intensity of Bayesian bootstrap weighting (``0`` = uniform weights).
   * - ``rsm``
     - float
     - ``0.5`` to ``1.0``
     - Random subspace method (feature subsampling ratio per split).
   * - ``border_count``
     - int
     - ``32`` to ``255``
     - Number of splits for numerical feature quantisation.

Extended Search Space (``extended_param_space=True``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 18 22 40

   * - Parameter
     - Type / Scale
     - Range / Choices
     - Description
   * - ``iterations``
     - int (step 25)
     - ``50`` to ``1000``
     - Expanded boosting iteration range.
   * - ``learning_rate``
     - float (log)
     - ``0.005`` to ``0.3``
     - Finer minimum step size.
   * - ``depth``
     - int
     - ``3`` to ``10``
     - Deeper maximum tree depth.
   * - ``l2_leaf_reg``
     - float (log)
     - ``1e-2`` to ``20.0``
     - Wider L2 leaf regularisation range.
   * - ``random_strength``
     - float (log)
     - ``1e-3`` to ``10.0``
     - Split-score randomness magnitude.
   * - ``rsm``
     - float
     - ``0.4`` to ``1.0``
     - Feature subsampling fraction per split.
   * - ``border_count``
     - int
     - ``32`` to ``255``
     - Numerical feature quantisation bin count.
   * - ``min_data_in_leaf``
     - int
     - ``1`` to ``50``
     - Minimum training samples in a leaf (active for ``Depthwise`` / ``Lossguide``).
   * - ``leaf_estimation_iterations``
     - int
     - ``1`` to ``10``
     - Number of gradient/Newton steps used to calculate leaf values.
   * - ``leaf_estimation_method``
     - categorical
     - ``['Newton', 'Gradient']``
     - Leaf value estimation method.
   * - ``model_size_reg``
     - float
     - ``0.0`` to ``2.0``
     - Model size regularisation coefficient.
   * - ``auto_class_weights``
     - categorical
     - ``['None', 'Balanced', 'SqrtBalanced']``
     - Automatic class-imbalance weighting scheme.
   * - ``score_function``
     - categorical
     - ``['Cosine', 'L2']``
     - Criterion used to select the best tree split.
   * - ``bootstrap_type``
     - categorical
     - ``['Bayesian', 'Bernoulli', 'MVS']``
     - Bootstrap sampling method for object weights.
   * - ``bagging_temperature``
     - float (conditional)
     - ``0.0`` to ``2.0``
     - Sampled when ``bootstrap_type == 'Bayesian'``.
   * - ``subsample``
     - float (conditional)
     - ``0.5`` to ``1.0``
     - Sampled when ``bootstrap_type`` is ``'Bernoulli'`` or ``'MVS'``.
   * - ``grow_policy``
     - categorical
     - ``['SymmetricTree', 'Depthwise', 'Lossguide']``
     - Tree growing strategy (oblivious symmetric vs depth-wise vs loss-guided).
   * - ``boosting_type``
     - categorical (conditional)
     - ``['Ordered', 'Plain']``
     - Sampled when ``grow_policy == 'SymmetricTree'``.
   * - ``max_leaves``
     - int (conditional)
     - ``16`` to ``128``
     - Sampled when ``grow_policy == 'Lossguide'``.

GPU Acceleration
----------------

All optimise, train, and apply functions accept ``use_gpu=True`` and an optional
``gpu_device_id`` index. GPU support is checked via
:func:`~rsgispy.classification.classcatboost.is_catboost_gpu_available`. When a
GPU is available, ``task_type='GPU'`` (and ``devices=str(gpu_device_id)`` when
specified) is enabled automatically, CPU-only ``rsm`` is omitted, and
``bootstrap_type='MVS'`` is mapped to ``'Bernoulli'``. Inference passes
``task_type='GPU'`` directly to ``predict()`` / ``predict_proba()``. If no
compatible GPU is detected, a ``UserWarning`` is emitted and execution falls
back to CPU.

API Reference
-------------

.. automodule:: rsgispy.classification.classcatboost
   :members:
   :undoc-members:
   :show-inheritance:
