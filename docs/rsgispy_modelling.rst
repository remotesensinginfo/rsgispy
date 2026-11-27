rsgispy.modelling
=================

.. seealso::

   Guide: :doc:`guides/species_distribution_modelling`

   Tutorials: :tutorial-dir:`species distribution modelling case study <11_case_study_species_distribution>` (:doc:`tutorials`)

   Thinning (spatial rarefaction) of occurrence records:
   :func:`rsgispy.vector.geoms.thin_pts_by_raster_grid`,
   :func:`rsgispy.vector.geoms.thin_pts_min_dist`,
   :func:`rsgispy.vector.geoms.thin_vector_geoms` and
   :func:`rsgispy.modelling.species_distribution.thin_pts_env_space`.
   Spatial cross-validation: :mod:`rsgispy.ml.cv` (``cv`` in
   :func:`rsgispy.modelling.species_distribution.fit_kfold_sklearn_mdls`).
   Evaluation: :func:`rsgispy.modelling.species_distribution.calc_sdm_eval_metrics`
   and the continuous Boyce index
   (:func:`rsgispy.modelling.species_distribution.calc_boyce_index`).

   Sampling bias: :func:`rsgispy.modelling.species_distribution.create_bias_grid`,
   ``bias_img`` in :func:`rsgispy.modelling.species_distribution.gen_pseudo_absences_smpls`,
   ``fixed_var_vals`` in the ``pred_*`` functions and
   :func:`rsgispy.modelling.species_distribution.calc_bias_weights` with
   ``sample_weight_col``.

.. automodule:: rsgispy.modelling
   :no-index:

Submodules
----------

* :doc:`modelling/species_distribution` — Species distribution modelling workflows using ``scikit-learn`` and :mod:`rsgispy.imageprocess`.

.. toctree::
   :maxdepth: 1

   modelling/species_distribution
