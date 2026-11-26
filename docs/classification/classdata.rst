rsgispy.classification.classdata
================================

Combining Pixel and Segment Samples
-----------------------------------

A segment classification often has few training samples (one per segment).
When the model only uses features that mean the same thing for a pixel and a
segment (e.g. the segment mean of each band and the pixel value), pixel
samples can be added with
:func:`~rsgispy.classification.classdata.combine_cls_samples`. It checks that
the feature columns and the class IDs agree and can record the source of each
sample (``source_col``) for a grouped split:

.. code-block:: python

   import rsgispy.classification.classdata as cd
   import rsgispy.classification.classsegments as cseg

   feat_cols = cls_cfg.seg_info.feature_cols  # e.g. ['Blue_mean', ...]
   seg_df = cseg.extract_class_segment_samples(cls_seg_cfg_file)
   pxl_df = cd.extract_class_training_samples(
       [cd.ImageBandInfo("sen2.tif", name="sen2", band_names=feat_cols)],
       class_info,
   )
   smpls_df = cd.combine_cls_samples(
       [seg_df, pxl_df],
       feat_cols=feat_cols,
       source_names=["segment", "pixel"],
       source_col="source",
   )

API
---

.. automodule:: rsgispy.classification.classdata
   :members:
   :undoc-members:
   :show-inheritance:
