rsgispy.classification.classxgboost
===================================

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
     - ``1`` to ``10``
     - Minimum sum of instance Hessian weight needed in a child.
   * - ``gamma``
     - float (log)
     - ``1e-4`` to ``5.0``
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
     - ``0.5`` to ``20.0``
     - Continuous Hessian leaf threshold.
   * - ``gamma``
     - float (log)
     - ``1e-4`` to ``10.0``
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
     - ``1e-5`` to ``20.0``
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
