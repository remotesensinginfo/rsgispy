rsgispy.vectorstats
===================

The :mod:`rsgispy.vectorstats` subpackage provides spatial metrics, geometric
measurements, feature counts, and geostatistical summaries on vector datasets
using ``geopandas`` without ``osgeo`` dependencies.

Examples
--------

Calculating polygon areas and counting features per attribute value:

.. code-block:: python

   import rsgispy.vectorstats.metrics

   rsgispy.vectorstats.metrics.calc_vec_area(
       "polys.gpkg", "polys", "area_m2", "polys_area.gpkg", "polys"
   )
   counts = rsgispy.vectorstats.metrics.count_feats_per_att_val(
       "polys.gpkg", "polys", "class_name"
   )

Computing an empirical semi-variogram from a point GPKG layer:

.. code-block:: python

   import rsgispy.vectorstats.variogram

   vario_df = rsgispy.vectorstats.variogram.calc_empirical_semivariogram(
       vec_file="samples.gpkg",
       vec_lyr="samples",
       vals_col="height",
       n_lags=20,
   )

Geometric Metrics & Feature Counts
----------------------------------

* :doc:`vectorstats/metrics` — Polygon area, hole area, line length, point distances/angles, point-in-polygon counts, radius counts, and attribute value counts.

.. toctree::
   :maxdepth: 1

   vectorstats/metrics

Geostatistics & Variograms
--------------------------

* :doc:`vectorstats/variogram` — Empirical semi-variogram computation.

.. toctree::
   :maxdepth: 1

   vectorstats/variogram
