rsgispy.classification.classsklearn
===================================

Landsat Classification Example
------------------------------

The following example classifies a 6-band Landsat surface reflectance scene into
``Mangroves`` (``out_id=10``), ``Water`` (``out_id=20``), and ``Other``
(``out_id=30``) classes. Training pixels are extracted from per-class vector
files, hyperparameters are tuned via Bayesian optimisation (Optuna TPE), and the
final Random Forest model is trained, evaluated, and applied to the Landsat
scene.

.. code-block:: python

    from sklearn.ensemble import RandomForestClassifier
    import rsgispy.classification.classdata as cd
    import rsgispy.classification.classsklearn as csk

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

    # 3. Optimise Random Forest parameters using Bayesian updating (Optuna)
    # (omitting param_space automatically uses the built-in RandomForest space)
    best_rf = csk.perform_sklearn_classifier_param_search(
        cls_cfg=cls_cfg,
        sk_classifier=RandomForestClassifier(n_jobs=-1, random_state=42),
        train_data=train_df,
        valid_data=valid_df,
        search_method="optuna",
        scoring="f1_macro",
        n_trials=30,
        refit=False,
        out_params_json="rf_best_params.json",
        rnd_seed=42,
    )

    # 4. Train and evaluate the optimised classifier
    fitted_rf, metrics = csk.train_sklearn_classifier(
        cls_cfg=cls_cfg,
        sk_classifier=best_rf,
        train_data=train_df,
        test_data=test_df,
        valid_data=valid_df,
        out_model_file="landsat_rf_model.joblib",
        out_metrics_file="landsat_rf_metrics.json",
        return_model=True,
        rnd_seed=42,
    )

    # 5. Apply the model to classify the Landsat scene (outputs IDs 10, 20, 30)
    csk.apply_sklearn_classifier(
        cls_cfg=cls_cfg,
        sk_classifier=fitted_rf,
        in_msk_img="landsat_valid_mask.tif",
        img_msk_val=1,
        output_img="landsat_rf_cls.tif",
        out_score_img="landsat_rf_probs.tif",
    )

Standard Parameter Search Spaces
--------------------------------

When ``param_space=None`` (or ``'default'`` / ``'extended'``) is passed to
:func:`~rsgispy.classification.classsklearn.perform_sklearn_classifier_param_search`,
built-in standard search spaces are automatically selected for the classifiers
below (including ``Pipeline`` objects whose final step is one of them, with
``<step>__`` prefixes applied automatically). All spaces can also be inspected
or customised via
:func:`~rsgispy.classification.classsklearn.get_sklearn_classifier_param_space`.

.. list-table::
   :header-rows: 1
   :widths: 26 14 30 30

   * - Classifier
     - Aliases
     - Default Parameters (``extended=False``)
     - Extended Parameters (``extended=True``)
   * - ``RandomForestClassifier``
     - ``'rf'``
     - ``n_estimators`` (50..500), ``max_depth`` (4..32), ``min_samples_split`` (2..20), ``min_samples_leaf`` (1..10), ``max_features`` (``'sqrt'``, ``'log2'``, ``None``)
     - Widens tree/depth ranges (``n_estimators`` 50..1000, ``max_depth`` 3..40) and adds ``criterion``, ``bootstrap``, ``class_weight``, ``ccp_alpha``, numeric ``max_features``
   * - ``ExtraTreesClassifier``
     - ``'et'``
     - ``n_estimators`` (50..500), ``max_depth`` (4..32), ``min_samples_split`` (2..20), ``min_samples_leaf`` (1..10), ``max_features``
     - Widens tree/depth ranges and adds ``criterion``, ``bootstrap``, ``class_weight``, ``ccp_alpha``, numeric ``max_features``
   * - ``GradientBoostingClassifier``
     - ``'gb'``, ``'gbc'``
     - ``n_estimators`` (50..400), ``learning_rate`` (0.01..0.3 log), ``max_depth`` (2..8), ``min_samples_split`` (2..20), ``min_samples_leaf`` (1..15), ``subsample`` (0.5..1.0)
     - Widens boosting ranges and adds ``max_features``, ``ccp_alpha``
   * - ``HistGradientBoostingClassifier``
     - ``'hgb'``, ``'hgbc'``
     - ``learning_rate`` (0.01..0.3 log), ``max_iter`` (50..500), ``max_leaf_nodes`` (15..127), ``max_depth`` (3..15), ``min_samples_leaf`` (5..50), ``l2_regularization`` (1e-4..10.0 log)
     - Widens tree/regularisation ranges and adds ``max_bins`` (64..255), ``class_weight``
   * - ``SVC``
     - ``'svm'``, ``'svc'``
     - ``C`` (0.1..100.0 log), ``kernel`` (``'rbf'``, ``'linear'``), ``gamma`` (``'scale'``, ``'auto'``), ``class_weight``
     - Conditional search over ``C`` (0.01..1000.0 log), ``kernel`` (``'rbf'``, ``'poly'``, ``'sigmoid'``, ``'linear'``), heuristic/numeric ``gamma``, ``degree``, ``coef0``, ``class_weight``
   * - ``MLPClassifier``
     - ``'mlp'``
     - ``hidden_layer_sizes`` (32..256), ``activation`` (``'relu'``, ``'tanh'``), ``alpha`` (1e-5..1e-1 log), ``learning_rate_init`` (1e-4..1e-2 log)
     - Conditional multi-layer networks (1..3 layers of 32..256 units), ``activation`` (adds ``'logistic'``), ``alpha``, ``learning_rate_init``, ``learning_rate``, ``early_stopping``
   * - ``KNeighborsClassifier``
     - ``'knn'``
     - ``n_neighbors`` (3..25 step 2), ``weights`` (``'uniform'``, ``'distance'``), ``p`` (1..2)
     - Widens ``n_neighbors`` (1..51 step 2), ``p`` (1..3), and adds ``leaf_size`` (15..60)
   * - ``LogisticRegression``
     - ``'logreg'``
     - ``C`` (1e-3..100.0 log), ``class_weight`` (``None``, ``'balanced'``)
     - Widens ``C`` (1e-4..100.0 log) and adds ``solver`` (``'lbfgs'``, ``'newton-cg'``, ``'saga'``), ``fit_intercept``, ``tol``
   * - ``DecisionTreeClassifier``
     - ``'dt'``
     - ``criterion`` (``'gini'``, ``'entropy'``), ``max_depth`` (3..30), ``min_samples_split`` (2..20), ``min_samples_leaf`` (1..15)
     - Widens ranges and adds ``'log_loss'``, ``splitter``, ``max_features``, ``class_weight``, ``ccp_alpha``
   * - ``AdaBoostClassifier``
     - ``'ada'``, ``'adaboost'``
     - ``n_estimators`` (50..400), ``learning_rate`` (0.01..2.0 log)
     - Widens ``n_estimators`` (50..800) and ``learning_rate`` (0.005..2.0 log)

Repeated k-fold and Jackknife Assessment
----------------------------------------

As for regression (:func:`~rsgispy.regression.regsklearn.perform_kfold_fit`),
:func:`~rsgispy.classification.classsklearn.perform_kfold_fit` fits the
classifier to every fold of a repeated (by default stratified) k-fold and
returns the accuracy, balanced accuracy, macro F1 and Cohen's kappa of each
fold with the out-of-fold predictions;
:func:`~rsgispy.classification.classsklearn.summarise_kfold_metrics`
summarises them. With ``groups`` (for example the polygon each pixel sample
came from) a group is never split across the train and test folds.
:func:`~rsgispy.classification.classsklearn.perform_jackknife_fit` leaves out
one sample (or group) at a time, giving the pooled leave-one-out metrics, the
influence of each sample (``delta_cv_accuracy``, ``delta_fit_accuracy``,
``pred_change_frac``) and, for linear classifiers, jackknife standard errors
of the coefficients of each class.

.. code-block:: python

    import pandas
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    import rsgispy.classification.classsklearn as csk

    df = pandas.read_parquet("train.parquet")
    x = df[["blue", "green", "red", "nir"]]
    y = df["class_name"]

    # 5 folds repeated 10 times; writes kfold_rf_metrics.csv and kfold_rf_predictions.csv.
    metrics, preds = csk.perform_kfold_fit(
        RandomForestClassifier(n_estimators=200, random_state=0),
        x,
        y,
        n_repeats=10,
        groups=df["poly_id"],
        out_dir="kfold_outputs",
        out_prefix="kfold_rf",
    )
    print(csk.summarise_kfold_metrics(metrics))

    jk = csk.perform_jackknife_fit(LogisticRegression(max_iter=1000), x, y)
    print(jk["metrics"]["accuracy"])
    print(jk["influence"].nlargest(5, "delta_cv_accuracy"))  # likely label errors

API Reference
-------------

.. automodule:: rsgispy.classification.classsklearn
   :members:
   :undoc-members:
   :show-inheritance:
