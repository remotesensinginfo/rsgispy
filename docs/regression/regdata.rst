rsgispy.regression.regdata
==========================

Combining Sample Tables
-----------------------

:func:`~rsgispy.regression.regdata.combine_reg_samples` combines regression
sample tables (e.g. segment and pixel samples) whose targets and features mean
the same thing, checking that the feature columns agree:

.. code-block:: python

   import rsgispy.regression.regdata as rd

   smpls_df = rd.combine_reg_samples(
       [seg_smpls_df, pxl_smpls_df],
       "canopy_cover",
       feat_cols=feat_cols,
       source_names=["segment", "pixel"],
       source_col="source",
   )

API
---

.. automodule:: rsgispy.regression.regdata
   :members:
   :undoc-members:
   :show-inheritance:
