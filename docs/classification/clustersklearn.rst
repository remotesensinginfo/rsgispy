rsgispy.classification.clustersklearn
=====================================

Applying a Fitted Clusterer to Another Image
--------------------------------------------

:func:`~rsgispy.classification.clustersklearn.img_pixel_sample_cluster` fits a
clusterer on a sample of an image's pixels; save it with ``out_model_file``
(or get it with ``return_model=True``) and apply it to another image with
:func:`~rsgispy.classification.clustersklearn.apply_img_pixel_clusterer`, so
the cluster IDs mean the same thing in both outputs. Both functions take an
image path (all bands), a list of
:class:`~rsgispy.classification.classdata.ImageBandInfo` or a classification
config. Only clusterers with ``predict()`` can be applied (e.g. ``KMeans``,
``Birch``, ``MeanShift``, ``GaussianMixture``):

.. code-block:: python

   from sklearn.cluster import KMeans
   import rsgispy.classification.clustersklearn as clsk

   clsk.img_pixel_sample_cluster(
       "sen2_2018.tif",
       "clusters_2018.tif",
       clusterer=KMeans(n_clusters=10, n_init=10, random_state=42),
       no_data_val=0,
       out_model_file="kmeans.joblib",
   )
   clsk.apply_img_pixel_clusterer(
       "sen2_2019.tif", "kmeans.joblib", "clusters_2019.tif", no_data_val=0
   )

API
---

.. automodule:: rsgispy.classification.clustersklearn
   :members:
   :undoc-members:
   :show-inheritance:
