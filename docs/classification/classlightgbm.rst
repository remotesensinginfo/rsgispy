rsgispy.classification.classlightgbm
====================================

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

API Reference
-------------

.. automodule:: rsgispy.classification.classlightgbm
   :members:
   :undoc-members:
   :show-inheritance:
