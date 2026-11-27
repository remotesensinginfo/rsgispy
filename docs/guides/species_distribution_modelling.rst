Species distribution modelling
==============================

A species distribution model (SDM) relates where a species has been recorded
to the environmental conditions at those places, and then uses that
relationship to predict how suitable every other place in a study area is.
The result is a map of relative habitat suitability, often turned into a
presence/absence map with a threshold. SDMs are used to find unsurveyed
populations, to plan surveys and protected areas, to assess invasion risk and
to project ranges under future climates.

This guide walks through a complete SDM workflow with
:mod:`rsgispy.modelling.species_distribution`: preparing the environmental
layers, generating pseudo-absences, extracting and checking the training
data, fitting an ensemble of scikit-learn models, evaluating them honestly
(including with spatial cross-validation), interpreting them and predicting a
suitability map.

Most species records are **presence-only** (herbarium and museum records,
citizen science and GBIF downloads): we know where the species was seen, but
not where it was looked for and not found. The workflow here therefore
contrasts the presences with *pseudo-absences* (also called background
points) sampled from the study area. Keep in mind what that means for the
output: the model learns how presence locations differ from the study area as
a whole, so the predicted value is a *relative* suitability index between 0
and 1, not the probability that the species occurs in a pixel. If you have
real absences from systematic surveys, use them instead of pseudo-absences
(see step 3).

.. seealso::

   The rsgispy tutorials cover this in notebooks you can run:
   :tutorial:`Prepare the datasets <11_case_study_species_distribution/01_prepare_datasets.ipynb>`
   and
   :tutorial:`Species distribution modelling <11_case_study_species_distribution/02_species_distribution_modelling.ipynb>`
   (section 11, the Snowdon beetle case study, with a logistic regression).
   See :doc:`../tutorials` for how to get them and their data.

.. contents:: Contents
   :local:
   :depth: 1

Installation
------------

The workflow needs the ``modelling`` extra, which installs ``scikit-learn``,
``scipy``, ``SALib`` and ``shap``:

.. code-block:: bash

   pip install "rsgispy[modelling]"

``SALib`` (Sobol sensitivity analysis) and ``shap`` (SHAP explanations) are
only used in step 9; the example script skips those parts if the packages are
not installed.

Overview
--------

.. code-block:: text

   Presence points + environmental rasters (one grid) + study area mask
            │
            ▼
   1. Environmental variables ──► EnvVarInfo per layer; valid_mask.tif
            │
            ▼
   2. Pseudo-absences ──────────► pseudo_absences.gpkg (random background)
            │
            ▼
   3. Extract & combine ────────► sdm_samples.gpkg: covariates + clsid (1/0)
            │
            ▼
   4. Explore & check collinearity (violin plots, correlation, VIF)
            │
            ▼
   5. Train / test split ───────► sdm_train.gpkg, sdm_test.gpkg
            │
            ▼
   6. Fit a model and an ensemble (random forest)
            │
            ├─► 7. Evaluate (AUC, Brier score, ROC curve)
            ├─► 8. Spatial cross-validation (random vs spatial blocks)
            ├─► 9. Interpret (response curves, jackknife, SHAP, Sobol)
            ▼
   10. Predict ─────────────────► sdm_probability.tif, sdm_agreement.tif
            │
            ▼
   11. Threshold ───────────────► sdm_presence_absence.tif (0/1/2)

The code in this guide comes from the example script
``docs/examples/species_distribution_modelling.py`` (included below step by
step). Its ``run_workflow`` function takes a dictionary of environmental
rasters, a study area mask and a layer of presence points. All layers, and the
study area mask (``1`` inside the study area, ``0`` outside), must be on
**the same pixel grid** (CRS, resolution and extent). Resample them first if
necessary, with :func:`rsgispy.imageutils.warp.resample_img_to_match`
(bilinear for continuous layers, nearest neighbour for categorical ones).

The figures and printed outputs on this page come from the Snowdon beetle
(*Chrysolina cerealis*) data of the rsgispy tutorials: a 170 km² study area in
Snowdonia (:tutorial-data:`sdm/environmental_data/Study_Area.geojson`), the
beetle records (:tutorial-data:`sdm/sightings/Chrysolina_cerealis.csv`),
20 m rasters of elevation
(:tutorial-data:`sdm/environmental_data/Elevation.tif`) and mean winter
temperature (:tutorial-data:`sdm/environmental_data/Mean_Winter_Temp.tif`) on
the British National Grid, and the Natural Resources Wales Phase 1 habitat map
(:tutorial-data:`sdm/environmental_data/Phase1_Veg.geojson`). The beetle is
rare: in Britain it lives only on a few mountains of Snowdonia, where it
feeds on wild thyme on mountain ledges, grassland and rocks. As in the
tutorial notebooks, the habitat map is turned into rasters of the
**distance to** each habitat of interest, because a record a few metres from a
habitat patch says more about the species than the habitat class of the
record's own pixel. The script below prepares those inputs and runs the
example; the rsgispy test suite runs the same ``run_workflow`` code on a small
synthetic scene (``tests/sdm_scene.py``).

.. literalinclude:: ../examples/species_distribution_modelling_tutorial.py
   :language: python
   :start-after: # [tutorial-data]
   :end-before: # [end-tutorial-data]
   :dedent:

The preparation of the records matters as much as the modelling. Of the 119
records, 19 are located to 2 km or worse (some at the centre of a 10 km
square), which is meaningless at 20 m, and the remaining 100 records fall in
only 34 pixels: they are repeat visits to a few well-known sites. Keeping one
record per pixel leaves 34 presences.

Step 1 — Environmental variables
--------------------------------

Each environmental layer is described by an
:class:`~rsgispy.modelling.species_distribution.EnvVarInfo`: the column name
used in the sample tables and by the model (``name``), the raster file and
band, and the variable type. Continuous variables
(``rsgispy.VAR_TYPE_CONTINUOUS``) can be z-score normalised later;
categorical variables (``rsgispy.VAR_TYPE_CATEGORICAL``, e.g. a soil or land
cover class) are passed to the model unchanged as integer codes, which tree
models handle well. ``min_vld_val`` and ``max_vld_val`` optionally define a
valid range, so that values outside it (e.g. negative precipitation in a badly
interpolated climate layer) are treated as missing.

:func:`~rsgispy.modelling.species_distribution.create_finite_mask` then writes
a ``uint8`` mask that is ``1`` where the pixel is inside the study area **and**
every variable is finite, not the raster's no data value and inside its valid
range, and ``0`` elsewhere. This mask is used for the pseudo-absences and the
prediction, so neither ever lands on missing data.

.. literalinclude:: ../examples/species_distribution_modelling.py
   :language: python
   :start-after: # [step-1-env-vars]
   :end-before: # [end-step-1-env-vars]
   :dedent: 4

The **study area** matters more than any model setting. Pseudo-absences are
drawn from it, so a study area much larger than the region the species could
have reached (for example a whole continent for a species of one mountain
range) makes presences trivially easy to tell apart from the background:
AUC is high, but the model has only learnt "mountains versus lowlands" and
says little about habitat within the range. Restrict the study area to the
region accessible to the species (e.g. the ecoregions with records, or a
buffer of some tens of kilometres around them).

Step 2 — Pseudo-absences
------------------------

:func:`~rsgispy.modelling.species_distribution.gen_pseudo_absences_smpls`
draws random points inside the valid mask. It first creates
``n_smpls + xtra_n_smpls`` random candidate points over the mask's bounding
box, removes those outside the mask (value ``img_msk_val``) and, if presence
points are given, those within ``presence_smpls_dist_thres`` metres of any
presence, and keeps a random ``n_smpls`` of the rest. If many candidates are
rejected (a small study area inside a large bounding box, or a large exclusion
distance), increase ``xtra_n_smpls``; check the printed number of
pseudo-absences in step 3.

.. literalinclude:: ../examples/species_distribution_modelling.py
   :language: python
   :start-after: # [step-2-pseudo-absences]
   :end-before: # [end-step-2-pseudo-absences]
   :dedent: 4

Choosing the pseudo-absence strategy:

* **How many?** For models that estimate a smooth density over the
  background (MaxEnt-like methods, GLMs/GAMs), use many (around 10,000) so the
  background is well described. For classification trees and random forests,
  a large excess of pseudo-absences makes the model predict "absent" almost
  everywhere; use a similar number to the presences (``equalise_smpls=True``
  in :func:`~rsgispy.modelling.species_distribution.combine_presence_absence_data`),
  or keep more and balance the classes with ``class_weight='balanced'`` as in
  step 6. A good compromise is several replicate sets of pseudo-absences, each
  giving one model of an ensemble.
* **Exclusion distance** (``presence_smpls_dist_thres``). A buffer of one or
  two pixels around presences avoids pseudo-absences in the same pixel as a
  presence (contradictory training data). Large buffers (several kilometres)
  remove suitable habitat from the background, which makes the problem easier
  and inflates evaluation scores; use them only if you have a reason (e.g.
  dispersal limitation).
* **Sampling bias.** Records cluster along roads, near towns and in well-known
  sites. Random pseudo-absences then teach the model the survey effort as well
  as the habitat. Mitigations are thinning the presences (step 3) and, when
  records of related species collected with the same methods are available,
  using those locations as "target-group" background (load them as the
  absence layer in step 3 instead of the random points).

Step 3 — Extracting and combining the samples
---------------------------------------------

:func:`~rsgispy.modelling.species_distribution.extract_env_var_data` adds one
column per environmental variable to a point layer, reading the value of the
pixel under each point (all bands of one file are read in one pass). Points
outside the rasters, or on no data or out-of-range values, get
``out_no_data_val`` (default ``-9999``); those rows are dropped here. Points
in another CRS can be reprojected on the fly with ``reproj_vec=True``.

:func:`~rsgispy.modelling.species_distribution.combine_presence_absence_data`
then stacks the presences (``clsid = 1``) and pseudo-absences
(``clsid = 0``) into one layer with only the variable columns, the class
column and the geometry, and returns the ordered list of variable names
(``analysis_vars``) used by every later step. With ``out_vec_file=None`` it
returns ``(GeoDataFrame, analysis_vars)`` instead of writing a file. If you
have true absences from surveys, pass them as the absence layer instead of the
pseudo-absences.

.. literalinclude:: ../examples/species_distribution_modelling.py
   :language: python
   :start-after: # [step-3-extract]
   :end-before: # [end-step-3-extract]
   :dedent: 4

.. code-block:: text

   34 presences and 1000 pseudo-absences
   ['elevation', 'winter_temp', 'ledges_dist', 'calc_grass_dist', 'rock_dist', 'scree_dist', 'flush_dist', 'impr_grass_dist']

**Thin the presences** before this step if records are clustered: several
records in one pixel are duplicates as far as the model is concerned, and a
dense cluster of records from one well-surveyed site makes that site dominate
the model. Keeping one record per pixel (or per few pixels) is a simple and
effective first step; for the Snowdon beetle it reduces 100 usable records to
34 presences (see the tutorial script above). With 34 presences and 1000
pseudo-absences, the classes are very unbalanced, which the random forest
handles with ``class_weight='balanced'`` (step 6).

Step 4 — Exploring the data and checking collinearity
-----------------------------------------------------

Before modelling, look at how the presences differ from the background, and
check for strongly correlated variables:

* :func:`~rsgispy.modelling.species_distribution.comparison_box_plots` writes
  one violin plot (or box plot with ``box_plots=True``) per variable,
  comparing the presences and pseudo-absences. Variables where the two
  distributions overlap completely are unlikely to help.
* :func:`~rsgispy.modelling.species_distribution.correlation_matrix` returns
  (and saves as CSV and a heatmap) the Pearson correlations between the
  variables.
* :func:`~rsgispy.modelling.species_distribution.calc_vif_multicollinearity`
  returns the variance inflation factor (VIF) of each variable: how well it
  is predicted by all the others together. VIF catches collinearity between
  combinations of variables that pairwise correlations miss.

.. literalinclude:: ../examples/species_distribution_modelling.py
   :language: python
   :start-after: # [step-4-explore]
   :end-before: # [end-step-4-explore]
   :dedent: 4

.. code-block:: text

                    elevation  winter_temp  ledges_dist  calc_grass_dist  rock_dist  scree_dist  flush_dist  impr_grass_dist
   elevation             1.00        -0.95        -0.43            -0.22       0.04       -0.48        0.27             0.63
   winter_temp          -0.95         1.00         0.46             0.34      -0.12        0.45       -0.21            -0.69
   ledges_dist          -0.43         0.46         1.00             0.32      -0.02        0.68       -0.11            -0.38
   calc_grass_dist      -0.22         0.34         0.32             1.00       0.02        0.26        0.03            -0.23
   rock_dist             0.04        -0.12        -0.02             0.02       1.00        0.34        0.07             0.09
   scree_dist           -0.48         0.45         0.68             0.26       0.34        1.00       -0.12            -0.36
   flush_dist            0.27        -0.21        -0.11             0.03       0.07       -0.12        1.00             0.18
   impr_grass_dist       0.63        -0.69        -0.38            -0.23       0.09       -0.36        0.18             1.00
   elevation          13.1
   winter_temp        15.2
   ledges_dist         2.3
   calc_grass_dist     1.4
   rock_dist           1.5
   scree_dist          2.9
   flush_dist          1.1
   impr_grass_dist     2.0
   Name: VIF, dtype: float64

(In a narrow terminal pandas abbreviates the correlation table with ``...``;
``pandas.set_option("display.max_columns", None)`` prints all the columns.)

Common rules of thumb are to keep only one of each pair with ``|r| > 0.7`` and
to remove variables until every VIF is below 10 (or 5 for a stricter check).
Here mean winter temperature falls with elevation (``r = -0.95``), and these
two are the only variables
with a VIF above 10. The distances to scree and to species-rich ledges are
also related (``r = 0.68``: ledges are often next to scree), but below the
thresholds. Collinear variables do not reduce the predictive
accuracy of a random forest much, but they split the importance between them,
make response curves hard to interpret, and make the model unreliable when it
is applied to places or times where the correlation is different (e.g. future
climates). When you choose which variable of a pair to keep, prefer the one
with the more direct ecological effect: here temperature, not elevation, so the
example is run with ``drop_vars=['elevation']`` (as in the tutorial
notebook).

Step 5 — Training and test sets
-------------------------------

:func:`~rsgispy.modelling.species_distribution.create_train_test_sets` splits
the samples into training and test sets, separately for presences and
pseudo-absences so both sets have the same class proportions (here 70 % of
each class for training). The test set is used to evaluate the models in
steps 6, 7 and 11.

.. literalinclude:: ../examples/species_distribution_modelling.py
   :language: python
   :start-after: # [step-5-split]
   :end-before: # [end-step-5-split]
   :dedent: 4

This is a *random* split: test points can be a few metres from training
points. Step 8 explains why that gives an optimistic estimate of accuracy, and
how to get a more honest one.

Step 6 — Fitting the models
---------------------------

Any scikit-learn classifier with ``predict_proba`` can be used. A random
forest is a robust default: it captures non-linear responses and interactions,
needs no normalisation and handles categorical codes. ``min_samples_leaf``
above 1 gives smoother, less over-fitted responses, and
``class_weight='balanced'`` compensates for unequal numbers of presences and
pseudo-absences.

* :func:`~rsgispy.modelling.species_distribution.fit_sklearn_mdl` fits one
  model on the training set and returns the training and test accuracy and
  the test ROC AUC (``roc_curve_plot`` also writes the model's ROC curve).
  With ``return_model=True`` the fitted model is also returned (as the first
  value) and the estimator you passed in is left unchanged.
* :func:`~rsgispy.modelling.species_distribution.fit_kfold_sklearn_mdls` fits
  ``n_kfolds`` models, each on a random ``fold_prop`` fraction of the
  training samples (drawn with replacement by default, like bagging), and
  prints the minimum, median and maximum ROC AUC and accuracies of the models
  on the test set. The ensemble is used for the prediction; the median of
  its members is more stable than any single model, and the spread between
  them shows how sensitive the result is to the training data. (Despite the
  name, these are not cross-validation folds: every model is evaluated on the
  same test set.)

.. literalinclude:: ../examples/species_distribution_modelling.py
   :language: python
   :start-after: # [step-6-model]
   :end-before: # [end-step-6-model]
   :dedent: 4

.. code-block:: text

   Classifier Train Score = 98.62%
   Classifier Test Score = 96.46%
   Single model: train acc 0.986, test acc 0.965, test AUC 0.986
   ROC: 0.9515151515151515 - 0.9706060606060606 - 0.9818181818181818
   Train: 0.976905311778291 - 0.9919168591224019 - 0.9953810623556582
   Test: 0.954983922829582 - 0.9694533762057878 - 0.977491961414791

With 96.5 % of the test points being pseudo-absences, an accuracy of 0.965 is
no better than predicting "absent" everywhere; the AUC, which does not depend
on the class proportions, is the more useful number. The test set has only 11
presences, so all the test scores here are uncertain: one presence more or
less correctly ranked changes the AUC by a few hundredths.

To tune the hyperparameters, wrap the model in a scikit-learn search object and
use :func:`~rsgispy.modelling.species_distribution.search_mdl_params`, e.g.
``best_rf, params = sdm.search_mdl_params(GridSearchCV(rf, {"min_samples_leaf":
[1, 3, 10]}, scoring="roc_auc"), train_vec, "train", analysis_vars)``, then use
``best_rf`` in place of ``rf``. Other options:

* **Models that need scaled inputs** (logistic regression, SVMs, neural
  networks): compute the mean and standard deviation of each continuous
  variable from the training data with
  :func:`~rsgispy.modelling.species_distribution.pop_normalise_coeffs`
  (stored on the ``EnvVarInfo`` objects), normalise the sample tables with
  :func:`~rsgispy.modelling.species_distribution.apply_normalise_coeffs`, and
  pass ``normalise_data=True`` to the prediction functions in step 10 so the
  rasters are scaled in the same way.
* **Presence-only (one-class) models** such as ``IsolationForest`` or
  ``OneClassSVM`` are fitted to the presences alone with
  :func:`~rsgispy.modelling.species_distribution.fit_sklearn_sgl_cls_mdl` /
  :func:`~rsgispy.modelling.species_distribution.fit_kfold_sklearn_sgl_cls_mdls`
  and applied with the ``*_sgl_cls_*`` prediction functions. They need no
  pseudo-absences, but usually discriminate less well.

Step 7 — Evaluating the ensemble
--------------------------------

The ensemble prediction for a sample is the median of its members' presence
probabilities (the same rule used for the raster in step 10).
:func:`~rsgispy.modelling.species_distribution.get_positive_class_probs`
returns a model's presence probabilities: the ``predict_proba`` column of
class ``1``, whatever the order of the classes.
:func:`rsgispy.tools.accuracy.bin_accuracy_scores_prob` compares the ensemble
prediction with the test labels:

* **Test AUC** — the area under the ROC curve: the probability that a randomly
  chosen presence gets a higher score than a randomly chosen pseudo-absence.
  0.5 is no better than random, 1 is perfect. With pseudo-absences the
  maximum achievable AUC is below 1, because some pseudo-absences fall in
  suitable habitat.
* **Point-biserial r** — the correlation between the labels and the scores.
* **Average precision** — the area under the precision–recall curve; more
  informative than AUC when presences are rare.
* **Brier score** and **log-loss** — errors of the probabilities themselves
  (0 is perfect); lower is better.

.. literalinclude:: ../examples/species_distribution_modelling.py
   :language: python
   :start-after: # [step-7-evaluate]
   :end-before: # [end-step-7-evaluate]
   :dedent: 4

.. code-block:: text

   {'Test AUC': 0.981, 'Point-Biserial r': 0.658, 'Av. Precision': 0.718, 'Brier Score': 0.022, 'Log-Loss Score': 0.075}

The ensemble ranks the presences well (AUC 0.98), but the average precision
(0.72) is much lower: with 11 presences among 311 test points, even a small
false positive rate puts many pseudo-absences among the highest scores. The
low Brier score mostly reflects the many pseudo-absences correctly given
probabilities near 0.

:func:`~rsgispy.modelling.species_distribution.calc_binary_roc_auc` returns
the ensemble's ROC curve and AUC, and
:func:`~rsgispy.modelling.species_distribution.plot_roc_curve` plots the
curve, returns ``(fig, ax)`` and saves it to ``plots/roc_curve.png``. It shows the trade-off between the true
positive rate (sensitivity) and the false positive rate (1 − specificity) over
all thresholds; a curve close to the top-left corner is good.

Step 8 — Spatial cross-validation
---------------------------------

**Why random cross-validation inflates AUC.** Environmental conditions are
spatially autocorrelated: points close to each other have similar values, and
presence records are usually clustered. With a random split, most test points
have a training point nearby with almost the same covariate values, so a
flexible model such as a random forest can score well by *remembering* the
training data rather than by learning a relationship that transfers to new
places. The test score then measures interpolation between nearby samples,
not prediction into unsampled areas, which is what the map is used for. The
result is optimistic AUCs and, when tuning, a preference for over-fitted
models.

Spatial cross-validation keeps nearby samples in the same fold.
:class:`rsgispy.ml.cv.SpatialBlockKFold` divides the study area into square
blocks of ``block_size`` metres and assigns whole blocks to folds; with
``stratify=True`` the blocks are assigned so that the folds have similar
numbers of presences (a fold can still have none if the presences are in few
blocks).
Its splits can be passed to any scikit-learn function that takes a ``cv``
argument. Here the same random forest is evaluated with 5-fold random and
5-fold spatial block cross-validation on all the samples:

.. literalinclude:: ../examples/species_distribution_modelling.py
   :language: python
   :start-after: # [step-8-spatial-cv]
   :end-before: # [end-step-8-spatial-cv]
   :dedent: 4

.. code-block:: text

   Random CV AUC  0.973 +/- 0.020
   Spatial CV AUC 0.929 +/- 0.033

The Snowdon beetle example uses 1 km blocks (``cv_block_size=1000.0``). The
spatial AUC is 0.04 lower than the random one: when whole 1 km blocks are left
out, the model has to predict presences at sites it has not seen, and it does
less well. With more strongly clustered records, spatial AUCs 0.05–0.2 lower
than random ones are common. The larger standard deviation of the spatial
scores is expected: blocks differ in how hard they are to predict. Report the
spatial cross-validation score as the accuracy of the model.

Choosing ``block_size``: blocks should be larger than the distance over which
the covariates (or the model residuals) are autocorrelated, which can be
estimated from an empirical semivariogram
(:func:`rsgispy.vector.stats.calc_empirical_semivariogram`), but small enough
that there are several blocks per fold. Blocks that are too large leave whole
environmental zones out of the training data and make the score pessimistic,
and with clustered presences they can leave a fold without any presence, so
its AUC cannot be computed and ``cross_val_score`` returns ``nan``. Here most
of the 34 presences are within 2 km of each other: with 2 km blocks two of
the five folds have no presence, while with 1 km blocks every fold has at
least two. Check the number of presences in each test fold
(``[data_y[test].sum() for _, test in blk_splits]``).
With ``block_size=None`` a size is chosen from the sample extent. Other
splitters in :mod:`rsgispy.ml.cv` are k-means clusters of the sample locations
(``method='kmeans'``), a buffer around each test point
(:class:`~rsgispy.ml.cv.SpatialBufferCV`), and
:func:`~rsgispy.ml.cv.nested_spatial_cv`, which tunes hyperparameters in an
inner spatial loop so the outer score is not biased by the tuning.

Step 9 — Interpreting the model
-------------------------------

A model that predicts well can still be ecologically implausible (e.g. by
relying on a sampling bias). Four complementary tools show what the model has
learnt:

* **Response curves** —
  :func:`~rsgispy.modelling.species_distribution.sklearn_mdl_variable_response_curves`
  varies one variable over its range (``n_samples`` steps) while holding the
  others at their mean value at the presences, and records the predicted
  probability. The returned table (also written as CSV) has ``<var>_vals`` and
  ``<var>_resp`` columns, and a plot per variable is written to the plots
  directory. Check that the shapes make sense (e.g. a temperature optimum,
  increasing response to precipitation). Responses beyond the range of the
  training data are extrapolation; random forests simply stay flat there.
* **Jackknife importance** —
  :func:`~rsgispy.modelling.species_distribution.sklearn_jackknife_var_importance`
  refits the model leaving each variable out (``*_leave_var_out`` columns)
  and with each variable alone (``*_only_var``), and reports training and
  test accuracy and test AUC for each. A variable with a high AUC on its own
  carries a lot of information; a variable whose removal lowers the AUC most
  carries information that no other variable has.
* **SHAP** —
  :func:`~rsgispy.modelling.species_distribution.shap_sklearn_mdl_explainer`
  (optional ``shap`` package) splits each prediction into contributions of the
  variables; the summary plot shows which variables matter most and in which
  direction, sample by sample.
* **Sobol sensitivity** —
  :func:`~rsgispy.modelling.species_distribution.salib_sklearn_mdl_sensitivity`
  (optional ``SALib`` package) samples the variables over their training
  ranges and decomposes the variance of the predicted probability. ``S1`` is a
  variable's effect on its own, ``ST`` its total effect including
  interactions; ``ST`` much larger than ``S1`` indicates strong interactions.
  The model is evaluated ``n_samples × (2 × n_vars + 2)`` times, and
  ``n_samples`` should be a power of 2. Wide ``ST_conf`` intervals mean
  ``n_samples`` is too small.

.. literalinclude:: ../examples/species_distribution_modelling.py
   :language: python
   :start-after: # [step-9-interpret]
   :end-before: # [end-step-9-interpret]
   :dedent: 4

.. code-block:: text

                    roc_test_acc_leave_var_out  roc_test_acc_only_var
   winter_temp                        0.983939               0.870152
   ledges_dist                        0.966364               0.919091
   calc_grass_dist                    0.986667               0.627121
   rock_dist                          0.983939               0.653636
   scree_dist                         0.984848               0.668636
   flush_dist                         0.980000               0.826970
   impr_grass_dist                    0.986667               0.565000
                          ST   ST_conf
   winter_temp      0.309467  0.097853
   ledges_dist      0.776935  0.188845
   calc_grass_dist  0.120805  0.065092
   rock_dist        0.017680  0.005980
   scree_dist       0.040084  0.017268
   flush_dist       0.081254  0.035577
   impr_grass_dist  0.059072  0.028876

.. figure:: images/sdm_response_temp.png
   :width: 80%
   :alt: Response curve of the Snowdon beetle presence probability to mean
         winter temperature, high between about 2.7 and 3.05 degrees C

   Response of the random forest to mean winter temperature, with the other
   variables held at their mean at the presences. The predicted probability is
   high (about 0.85–0.9) between about 2.7 and 3.05 °C, the range of the
   training presences (ticks at the bottom), lower on the coldest summits and
   low in the warmer valleys. Random forest responses are step-like and flat
   beyond the last split. (Drawn from ``response_curves.csv`` by
   ``make_figures`` in the tutorial script; the example itself writes
   ``plots/response_curves/winter_temp_response_curve.png``.)

For the Snowdon beetle the distance to upland species-rich ledges is the most
informative variable alone (AUC 0.92), removing it lowers the AUC most (from
0.986 with all the variables to 0.966), and it has the largest total Sobol
index; its response curve (``ledges_dist_resp``) is about 0.87 up to 600 m
from a ledge, falls in steps between 600 and 900 m, and is below 0.1 beyond
900 m. Winter temperature is
second (AUC 0.87 alone; Sobol ``ST`` 0.31), then the distance to flushes and
springs. Removing any other single variable barely changes the AUC, partly
because of correlated partners (scree and ledges, ``r = 0.68``) and partly
because the 11 test presences cannot resolve small differences. The Sobol
indices change a little from run to run (the SALib samples are not seeded),
and the wide ``ST_conf`` of ``ledges_dist`` shows that more samples would be
needed for precise values; the ranking is stable.

**When the methods disagree**, it is
usually because of correlated variables (jackknife "leave out" scores barely
change when a correlated partner can take over, while SHAP and Sobol may split
the effect between them) or interactions (which Sobol ``ST`` and SHAP capture
but one-at-a-time response curves do not). Use the jackknife for "what can I
remove?", response curves and SHAP for "how does the suitability change?", and
Sobol indices for "which variable drives the variation of the prediction
across the study area?".

Step 10 — Predicting the suitability map
----------------------------------------

:func:`~rsgispy.modelling.species_distribution.pred_ensemble_sklearn_mdls_prob`
applies every model of the ensemble to every valid pixel (``valid_mask.tif ==
1``) and writes the **median presence probability** as a ``float32`` raster
with values from 0 to 1 and no data value ``-1`` outside the mask. The rasters
are read block by block, so the size of the study area is limited only by
disk space and time.

:func:`~rsgispy.modelling.species_distribution.pred_ensemble_sklearn_mdls_cls`
writes the **proportion of models predicting presence** (each with its default
0.5 cut-off). It is a simple measure of agreement: values near 0 or 1 mean
the models agree; values near 0.5 mark areas where the prediction depends on
which training samples were used and should be treated with caution.

.. literalinclude:: ../examples/species_distribution_modelling.py
   :language: python
   :start-after: # [step-10-predict]
   :end-before: # [end-step-10-predict]
   :dedent: 4

The variable names passed as ``analysis_vars`` must be in the same order as
for training. Single models are applied with
:func:`~rsgispy.modelling.species_distribution.pred_sklearn_mdl_prob` and
:func:`~rsgispy.modelling.species_distribution.pred_sklearn_mdl_cls`. For
anything these functions do not cover (e.g. the mean instead of the median of
the ensemble, or a hand-written suitability index),
:func:`~rsgispy.modelling.species_distribution.predict_raster_from_env_vars`
applies your own function to the variable table of each block of valid pixels.
To display the map, use
:func:`rsgispy.mapping.raster.create_raster_cmap_img_map` with a sequential
colour map.

.. figure:: images/sdm_probability.png
   :width: 100%
   :alt: Map of the median presence probability of the Snowdon beetle over the
         study area in Snowdonia, with the presence records

   Median presence probability of the 10-model ensemble for the Snowdon
   beetle (``sdm_probability.tif``), with the 34 presences (red circles). The
   suitable areas are around the cluster of records in the south-west of the
   study area and a few smaller patches elsewhere; most of the study area
   has a probability near 0.

Step 11 — Thresholding to presence/absence
------------------------------------------

Many uses (area of suitable habitat, overlap with protected areas, survey
planning) need a binary map. The threshold should be chosen from the data, not
fixed at 0.5: with unequal numbers of presences and pseudo-absences, or
balanced class weights, 0.5 has no special meaning. Common choices are:

* **maximum TSS** — the threshold that maximises the true skill statistic
  (sensitivity + specificity − 1), equivalent to the point of the ROC curve
  furthest above the diagonal. TSS, unlike kappa, does not depend on
  prevalence. This is used in the example.
* **sensitivity = specificity** — balances omission and commission errors.
* **10th percentile training presence** — the probability exceeded by 90 % of
  the training presences (``numpy.percentile(train_probs, 10)``); a
  conservative choice that accepts 10 % omission, common with presence-only
  data.
* **minimum training presence** — the lowest probability at any presence;
  very inclusive, sensitive to outliers.

The example computes the maximum-TSS threshold from the ensemble predictions
of the test set (step 7) with
:func:`~rsgispy.modelling.species_distribution.calc_binary_threshold`
(``method="max_tss"``; ``method="sens_eq_spec"`` gives the sensitivity =
specificity threshold), reports the binary accuracy at that threshold with
:func:`rsgispy.tools.accuracy.accuracy_scores_binary` (the "Hanssen-Kuiper
skill" is the TSS), applies it to the probability raster with
:func:`rsgispy.imagecalc.maths.band_maths` and computes the proportion of the
study area predicted suitable with
:func:`rsgispy.imagecalc.maths.calc_prop_true_exp`. The output is a ``uint8``
thematic raster: ``0`` outside the study area, ``1`` unsuitable, ``2``
suitable, with statistics and a colour table added by
:func:`rsgispy.imageutils.stats.pop_thmt_img_stats`.

.. literalinclude:: ../examples/species_distribution_modelling.py
   :language: python
   :start-after: # [step-11-threshold]
   :end-before: # [end-step-11-threshold]
   :dedent: 4

.. code-block:: text

   Threshold 0.189: TSS 0.923
   8.8 % of the study area is predicted suitable

At the maximum-TSS threshold of 0.189, all 11 test presences are above the
threshold (sensitivity 1.0) and 92.3 % of the test pseudo-absences below it
(specificity 0.923). The suitable area is 8.8 % of the 170 km² valid area,
about 15 km². With so few test presences, the threshold itself is
uncertain: a different split could move it noticeably.

The threshold is chosen and evaluated on the same test set here, which flatters
the TSS slightly; for a strict assessment choose it on one set (e.g. the
out-of-fold predictions of spatial cross-validation) and report the accuracy on
another. Multiply ``suit_prop`` by the valid area (number of pixels with
``valid_mask.tif == 1`` × pixel area) to get the suitable area in m².

Performance and large datasets
------------------------------

* **Prediction** — the cost is roughly (number of models) × (number of valid
  pixels) × (cost of one prediction). Rasters are processed in blocks
  (``block_size`` argument of the prediction functions), so memory is bounded;
  use ``n_jobs=-1`` in the random forest to use all cores, fewer trees
  (200–500 is usually enough) and fewer ensemble members (5–10) for large
  areas. Test the workflow on a coarser grid or a small subset first.
* **Covariate layers** — every ``EnvVarInfo`` can point to a different file;
  they are read together, so they must be on the same grid. A VRT stack
  (:func:`rsgispy.imageutils.vrt.create_stack_images_vrt`) avoids copying the
  data when bands are spread over many files. Store large layers as tiled,
  compressed GeoTIFFs (COGs) for fast block reads.
* **Number of samples** — extraction and model fitting scale with the number
  of points. Tens of thousands of pseudo-absences are rarely needed for tree
  models.
* **Interpretation** — jackknife importance fits ``2 × n_vars + 1`` models;
  Sobol analysis evaluates ``n_samples × (2 × n_vars + 2)`` predictions;
  SHAP is slow for large forests (use ``subsample_n_smpls``). Run them on a
  single model, not the ensemble.

Common problems
---------------

* **"No pseudo-absence samples fell within the specified mask region"** or
  far fewer pseudo-absences than requested — the mask value is wrong (it is
  ``img_msk_val``, usually ``1``), the mask covers a small part of its
  bounding box (increase ``xtra_n_smpls``), or the exclusion distance is too
  large for the study area.
* **Presence points in a different CRS** —
  :func:`~rsgispy.modelling.species_distribution.gen_pseudo_absences_smpls`
  reprojects the presences to the mask's CRS, but
  :func:`~rsgispy.modelling.species_distribution.extract_env_var_data` needs
  ``reproj_vec=True`` (and ``vec_def_epsg`` for files without a CRS);
  otherwise it raises an exception saying that the raster and vector layers
  have different projections. Check that the printed number of
  presences in step 3 is what you expect.
* **Rasters on different grids** — resample them to one grid first with
  :func:`rsgispy.imageutils.warp.resample_img_to_match`; the prediction reads
  the intersection of the inputs, block by block, and assumes they align.
* **No data values** — every layer must have its no data value set in the
  file (or use ``min_vld_val`` / ``max_vld_val``), otherwise fill values such
  as ``-3.4e38`` or ``-32768`` are treated as real data. The ``-9999`` used for
  missing values in the sample tables must not be a valid value of any
  variable; change ``out_no_data_val`` if it is.
* **Every prediction is close to 0 (or 1)** — very unbalanced classes; use
  ``class_weight='balanced'`` or ``equalise_smpls=True``, and choose the
  threshold from the data (step 11).
* **AUC near 1 with a large study area** — see the study area discussion in
  step 1 and spatial cross-validation in step 8.
* **"Computed fold size is smaller than the number of classes"** — too few
  training samples for ``fold_prop``; increase ``fold_prop`` or the number of
  samples.
* **Optional dependencies** — ``shap`` and ``SALib`` raise an exception naming
  the missing package and the ``modelling`` extra; install them with
  ``pip install "rsgispy[modelling]"``.
* **Variable order** — the prediction functions build the feature table in
  the order of ``analysis_vars``; pass the same list used for training.

Related guides
--------------

* :doc:`pixel_classification_workflow`
* :doc:`pixel_regression_workflow`
* :doc:`making_maps`
