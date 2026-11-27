rsgispy.regression
==================

.. seealso::

   Guides: :doc:`guides/pixel_regression_workflow`, :doc:`guides/segment_ml_workflow`

   Tutorials: :tutorial-dir:`section 5, regression <05_regression>`, :tutorial-dir:`ICESat-2 bathymetry case study <10_case_study_icesat2_bathymetry>` (:doc:`tutorials`)

.. automodule:: rsgispy.regression
   :no-index:

Submodules
----------

* :doc:`regression/regdata` — Image/band and continuous target training definitions, JSON configuration I/O, point/polygon vector sample extraction, combining sample tables, and Parquet/tabular splitting.
* :doc:`regression/regsklearn` — Scikit-learn parameter search (Grid, Random, Optuna) with built-in standard regressor search spaces, univariate/multivariate training, and raster prediction.
* :doc:`regression/regxgboost` — XGBoost univariate and multivariate Optuna hyperparameter optimisation, regressor training, and block-wise image regression.
* :doc:`regression/reglightgbm` — LightGBM univariate and multivariate Optuna hyperparameter optimisation, regressor training, and block-wise image regression.
* :doc:`regression/regcatboost` — CatBoost univariate and multivariate Optuna hyperparameter optimisation, regressor training, and block-wise image regression.
* :doc:`regression/regfeatures` — Scatter diagnostics, collinearity analysis (correlation, VIF), feature selection (CQV, correlation, VIF, univariate, RFE/RFECV, model/LassoLars, BorutaShap), response function curves, permutation/model importance, jackknife analysis, SHAP, and SALib Sobol sensitivity.
* :doc:`regression/regsegments` — Object-based (segment) regression: segment feature creation, reference value aggregation per segment, group-aware splitting, chunked application of univariate/multi-target regressors to segment vector layers (with tree-ensemble quantiles), and rasterising. See the :doc:`guides/segment_ml_workflow` guide.
* :doc:`regression/regmetrics` — Predicted raster point sampling, regression accuracy statistics (R², RMSE, MAE, Bias, Explained Variance, Pearson r), bootstrap confidence intervals, and diagnostic scatter/residual plots.

.. toctree::
   :maxdepth: 1

   regression/regdata
   regression/regsklearn
   regression/regxgboost
   regression/reglightgbm
   regression/regcatboost
   regression/regfeatures
   regression/regsegments
   regression/regmetrics
