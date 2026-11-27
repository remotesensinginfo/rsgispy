Image co-registration
=====================

Two images of the same place rarely line up exactly. Images from different
dates, sensors or processing chains can be offset by a fraction of a pixel to
many pixels, and drone orthomosaics or historical aerial photographs can be
offset, rotated or distorted. *Co-registration* moves one image (the
**floating** image) so that it matches another (the **reference** image).

Why it matters:

* **Change detection** — a shift of half a pixel creates "change" along every
  edge in the image (field boundaries, roads, coastlines), which can be larger
  than the real change you are looking for.
* **Time series and composites** — misaligned dates blur composites and mix
  neighbouring land covers in a pixel's time series.
* **Combining datasets** — training data, field plots or a DEM only describe
  the right pixels if the images are in the right place.

rsgispy offers three levels of co-registration in
:doc:`rsgispy.imageregistration <../rsgispy_imageregistration>`:

* a **whole-image shift**
  (:func:`~rsgispy.imageregistration.shift.find_image_offset` and
  :func:`~rsgispy.imageregistration.shift.apply_offset_to_image`) — fast,
  no resampling, enough when the images only differ by a translation;
* **tie points** matched automatically across the image
  (:func:`~rsgispy.imageregistration.tiepoints.generate_tie_points`), filtered
  for outliers and used to fit a geometric model (shift, similarity, affine,
  polynomial or thin-plate spline) with which the image is warped onto the
  reference grid
  (:func:`~rsgispy.imageregistration.warping.warp_with_tie_points`, or the
  one-call :func:`~rsgispy.imageregistration.warping.coregister_images`);
* **ground control points** (GCPs) measured by hand or in the field, attached
  from a point vector layer
  (:func:`~rsgispy.imageregistration.warping.add_vec_pts_as_gcps_to_img`) and
  used for the warp (:func:`~rsgispy.imageregistration.warping.warp_with_gcps`).

This guide shows all three and how to check the result. The
:doc:`analysis-ready data guide <analysis_ready_data>` uses
:func:`~rsgispy.imageregistration.warping.coregister_images` to align the
dates of a Sentinel-2 time series before compositing.

.. seealso::

   The rsgispy tutorials do not have a co-registration notebook, but the
   images used on this page come from them: the 2021 Sentinel-2 image of
   Aberystwyth is introduced in
   :tutorial:`Image basics and NDVI <01_getting_started/01_image_basics_and_ndvi.ipynb>`
   (section 1), and
   :tutorial:`Image-to-image change detection <06_change_detection/03_image_to_image_change.ipynb>`
   (section 6) compares two dates pixel by pixel, which only works for
   co-registered images. See :doc:`../tutorials` for how to get the
   notebooks and their data.

.. contents:: Contents
   :local:
   :depth: 1

Installation
------------

Image co-registration only needs the core dependencies (``rasterio``,
``numpy``, ``pandas``, ``geopandas``). The ``numba`` extra speeds up template
matching considerably:

.. code-block:: bash

   pip install "rsgispy[numba]"

Overview
--------

.. code-block:: text

   reference image + floating image (+ GCP points, optional)
            │
            ▼
   1. Measure the misregistration ──► similarity metrics, whole-image offset
            │
            ├─► 2. Whole-image shift ──► flt_shifted.tif (header only)
            ▼
   3. Tie points ─────────────────────► table: ref_x/y, flt_x/y, dx/dy, metric
            ▼
   4. Outlier filtering (RANSAC) ─────► inlier tie points, RMSE
            ▼
   5. Choosing the geometric model ───► check point RMSE per model
            ▼
   6. Warp with tie points ───────────► flt_warped_affine.tif (reference grid)
   7. (or all of 3, 4, 6 in one call) ─► flt_coregistered.tif
   8. Warp with GCPs (optional) ──────► flt_warped_gcps.tif
            ▼
   9. Assess ─────────────────────────► metrics and residual offset per output

The code in this guide comes from the example script
``docs/examples/image_coregistration.py`` (included below step by step), which
is run by the rsgispy test suite on a synthetic 160 × 160 pixel, 10 m image
pair. The outputs on this page come from two Sentinel-2 images of
Aberystwyth from the rsgispy tutorials:

* the **reference** is the 20 m image of 29 June 2018,
  :tutorial-data:`sen2_20180629_t30uvd_orb037_osgb_stdsref_20m.tif`, cut to
  the extent of the other image (466 × 473 pixels, 9.3 × 9.5 km);
* the **floating** image is the 10 m image of 27 May 2021,
  :tutorial-data:`sen2_20210527_aber.tif`, averaged to the 20 m reference
  grid. The two products are already well aligned (a whole-image offset of
  at most a quarter of a pixel, 5 m), so to have something to correct the
  script adds a **known misregistration** while resampling: the features
  move 23 m east and 14 m south and are rotated by 0.3° clockwise about the
  image centre, which moves the corners by up to 35 m more. The GCPs (step
  8) are six points spread over the image whose true coordinates come from
  the same known distortion.

Everything else is real: three years of land change, a different season
and atmosphere, and the sea, which has no texture to match. The NIR band (B8A, band 8) is used in both images:

.. literalinclude:: ../examples/image_coregistration_tutorial.py
   :language: python
   :start-after: # [tutorial-data]
   :end-before: # [end-tutorial-data]
   :dedent: 4

Run the example on your own data from the command line, for example:

.. code-block:: bash

   python image_coregistration.py --ref ref_s2_10m.tif --float new_10m.tif \
       --band 7 --flt-band 4 --gcps float_gcps.gpkg --out-dir outputs

Choosing the reference and floating images
------------------------------------------

* The **reference** should be the image with the best known geolocation: an
  orthorectified product (e.g. Sentinel-2 or Landsat Level-2, which are
  aligned to a global reference), an orthophoto, or the clearest date of a
  time series.
* Both images must be in the same, projected CRS (metres), at similar
  resolutions. If they are not, reproject or resample the floating image first
  (:func:`rsgispy.imageutils.warp.reproject_image`,
  :func:`rsgispy.imageutils.warp.resample_img_to_match`); a large resolution
  difference makes matching unreliable.
* Choose one band from each image that shows the same features with plenty of
  texture: NIR or red for vegetated land, SWIR for arid areas, the panchromatic
  or green band for urban areas. The bands do not have to have the same
  values, only the same pattern (correlation is insensitive to gain and
  offset; mutual information also copes with inverted or non-linear
  relationships, e.g. between optical and radar images).
* Mask clouds (set them to no data) first; clouds and their shadows are
  different in every image and produce false matches.

Step 1 — Measuring the misregistration
--------------------------------------

Before changing anything, measure how well the images match.
:func:`rsgispy.imagecalc.comparison.calc_img_similarity_metrics` calculates
similarity metrics over the overlapping, valid pixels of the two images
without writing anything (correlation: 1 = identical pattern; normalised
mutual information: 0 = independent, higher is better).

:func:`rsgispy.imageregistration.shift.find_image_offset` then searches for
the single shift (up to ``x_search`` / ``y_search`` pixels in each direction)
that maximises the similarity, refined to ``1 / sub_pxl_res`` of a pixel.
``metric_type`` selects the similarity measure: correlation
(``rsgispy.METRIC_CORRELATION``) for images from the same or similar sensors,
mutual information (``rsgispy.METRIC_MUTUAL_INFO`` /
``METRIC_NORM_MUTUAL_INFO``) for different sensors, and phase correlation
(``rsgispy.METRIC_PHASE_CORR``), which does not search (``x_search`` /
``y_search`` are not used), so it is fast for large shifts. With
``return_details=True`` it returns the offset in map units and pixels, the
metric value at the best shift and the number of pixels compared.

.. literalinclude:: ../examples/image_coregistration.py
   :language: python
   :start-after: # [step-1-measure]
   :end-before: # [end-step-1-measure]
   :dedent: 4

Output for the tutorial pair (rounded):

.. code-block:: text

   {'correlation': 0.829, 'norm_mutual_info': 0.150}
   {'x_offset_map': -20.0, 'y_offset_map': 20.0, 'x_offset_pxl': -1.0,
    'y_offset_pxl': 1.0, 'metric_value': 0.846, 'metric_name': 'correlation',
    'n_valid_pxls': 219376}

The offset is the correction to apply to the floating image: features
are about 20 m too far east and 20 m too far south, so the image must move
−20 m in X and +20 m in Y. The expected correction is about −23 m and +19 m
(the added 23 m and 14 m, plus the 5 m by which the two dates already
differed in Y); the estimate is limited to 0.25 pixels (5 m) and affected by
the rotation. The correlation is well below 1 even after the shift
(``metric_value``): the land changed between the dates. If the best
shift is at the edge of the search window (±``search_area`` pixels in the
example), the true offset
may be larger: increase ``x_search`` / ``y_search``.

Step 2 — Applying a whole-image shift
-------------------------------------

:func:`rsgispy.imageregistration.shift.apply_offset_to_image` adds the offset
to the image origin in the file header; the pixel values are not resampled,
so the operation is instant and lossless. The shifted image is then no longer
on the reference pixel grid (unless the shift is a whole number of pixels);
resample it with :func:`rsgispy.imageutils.warp.resample_img_to_match` if the
grids must match.

.. literalinclude:: ../examples/image_coregistration.py
   :language: python
   :start-after: # [step-2-shift]
   :end-before: # [end-step-2-shift]
   :dedent: 4

A whole-image shift is all that is needed for images that differ only by a
translation, such as two dates of the same Sentinel-2 tile. When the
misregistration varies across the image (rotation, scale or terrain
distortion), use tie points.

Step 3 — Generating tie points
------------------------------

:func:`rsgispy.imageregistration.tiepoints.generate_tie_points` places
candidate points on a grid every ``pixel_gap`` pixels and, for each, finds the
location in the floating image that best matches a ``win_size`` × ``win_size``
window of the reference image. Candidates are skipped where more than 20 % of
the window is no data, where the window has too little texture (standard
deviation below ``sd_ref_thres`` / ``sd_flt_thres``), or where the best match
scores below ``threshold``. Two matching methods:

* ``TIE_METHOD_TEMPLATE_MATCH`` slides the window over a search area of
  ±``search_area`` pixels and computes ``metric_type`` (correlation by
  default; mutual information for different sensors) at every position, then
  refines the peak to ``1 / sub_pxl_res`` of a pixel. It is robust and
  accurate; its cost grows with ``search_area²``, so keep the search area
  close to the largest expected shift (e.g. twice the offset from step 1).
* ``TIE_METHOD_PHASE_CORR`` uses FFT phase correlation of the two windows,
  refined to ``1 / sub_pxl_res`` of a pixel. It does not search, so it is
  fast whatever the shift, but the shift must be less than about a quarter of
  ``win_size`` (e.g. 8 pixels for 32-pixel windows); ``search_area`` is only
  used to reject matches with a larger shift (set it to 0 to keep them all).
  Windows of 32 pixels or more give unbiased offsets: for the synthetic pair
  the mean tie point offset is within 0.5 m of the true 23 m with 32, 64 and
  128 pixel windows. The match score (``metric``) is the height of the
  correlation peak (1 for identical windows); unrelated windows of smooth
  images can score 0.3–0.7 with 32-pixel windows (0.2–0.4 with 64), so
  raise ``threshold`` (default 0.15) to 0.6–0.8 to reject poor matches.
  Phase correlation is more sensitive to noise than template matching: for
  noisy or weakly textured images (the individual tie points of the
  synthetic pair are about twice as precise with template matching), or
  images from different sensors, use template matching.

Choosing the parameters:

* ``win_size`` — large enough to contain several distinctive features
  (typically 32–128 pixels); larger windows give more reliable but fewer and
  less local matches.
* ``pixel_gap`` — sets the number of tie points. Aim for tens to hundreds of
  points spread over the whole image; more are only useful for polynomial or
  thin-plate spline models.
* ``threshold`` — the minimum correlation of an accepted match (default 0.4
  for template matching). Raise it (e.g. 0.6–0.8) to reject poor matches early
  when the images are similar; lower it for images from different sensors.

The result is a pandas ``DataFrame`` with one row per tie point: the
map coordinates in the reference (``ref_x``, ``ref_y``) and floating
(``flt_x``, ``flt_y``) images, the same in pixel coordinates
(``ref_pxl_x`` …), the offset ``dx_map``/``dy_map`` (reference minus floating,
in map units) and ``dx_pxl``/``dy_pxl`` (pixels), and the match score
``metric``. Pass ``out_gcp_file`` to write the tie points to a file instead
(CSV, Parquet, a vector layer such as GeoPackage, or the ENVI / RSGISLib GCP
text formats selected with ``output_type``, e.g.
``rsgispy.imageregistration.TYPE_ENVI_IMG2IMG``).

.. literalinclude:: ../examples/image_coregistration.py
   :language: python
   :start-after: # [step-3-tie-points]
   :end-before: # [end-step-3-tie-points]
   :dedent: 4

**Check:** the offsets should be consistent (a small standard deviation of
``dx_map``/``dy_map``) or vary smoothly across the image; view the tie points
in a GIS over both images. Here 26 tie points are matched (windows on the
sea, or where the land changed between the dates, are rejected),
with a mean offset of about −27 m / +11 m. The offsets vary from −45 to −5 m
in X and from −5 to +40 m in Y across the image because of the rotation.

Step 4 — Removing outlier tie points
------------------------------------

Some matches are wrong: repeated patterns (rows of crops, buildings), land
change, clouds or water. :func:`rsgispy.imageregistration.tiepoints.filter_tie_points_outliers`
fits a geometric model (``model``) and removes the tie points that do not
agree with it:

* ``OUTLIER_METHOD_RANSAC`` (recommended) repeatedly fits the model to random
  minimal subsets of the points and keeps the largest set that agrees within
  ``residual_thres`` (map units). It copes with a large proportion of
  outliers. Set ``rnd_seed`` for reproducible results.
* ``OUTLIER_METHOD_RESIDUAL`` fits all the points and removes the worst point
  until every residual is below ``residual_thres``.
* ``OUTLIER_METHOD_MAD`` is like the residual method but with a threshold
  based on the median absolute deviation of the residuals, which adapts to
  the data.

Choose ``residual_thres`` from the accuracy you need and the pixel size: one
pixel is a good start; lower it when the tie points are precise (the
tutorial pair uses 10 m, half of its 20 m pixels).
With ``return_model=True`` the function also returns diagnostics: the number
of input points, inliers and outliers, and the RMSE and maximum residual of
the inliers (map units). The inliers have an extra ``residual`` column. They
are saved here as a CSV file, which can be passed (like the data frame) to the
warping functions.

.. literalinclude:: ../examples/image_coregistration.py
   :language: python
   :start-after: # [step-4-filter]
   :end-before: # [end-step-4-filter]
   :dedent: 4

.. code-block:: text

   {'n_input': 26, 'n_inliers': 25, 'n_outliers': 1, 'rmse': 4.26,
    'max_residual': 6.90}

One tie point disagrees with the affine model by more than 10 m and is
removed; the other 25 fit it with an RMSE of 4.3 m (about a fifth of a
pixel).

**Check:** the inlier RMSE should be a fraction of a pixel. If most points are
outliers, the images may not overlap where you think, the search area may be
too small, or the model may be too simple (e.g. a shift for an image that is
rotated).

Step 5 — Choosing the geometric model
-------------------------------------

The geometric model maps floating image coordinates to reference
coordinates:

* ``TRANSFORM_SHIFT`` — a translation (2 parameters). Images from the same
  orbit and processing chain.
* ``TRANSFORM_SIMILARITY`` — translation, rotation and a uniform scale (4).
* ``TRANSFORM_AFFINE`` / ``TRANSFORM_POLY1`` — adds shear and different X
  and Y scales (6). A good default for satellite images from different
  sensors or orbits.
* ``TRANSFORM_POLY2`` / ``TRANSFORM_POLY3`` — 2nd/3rd order polynomials (12 /
  20 parameters) for smoothly varying distortions (e.g. scanned aerial
  photographs). They need many well-spread points and can behave badly
  outside the area covered by the points.
* ``TRANSFORM_TPS`` — a thin-plate spline that fits every tie point exactly;
  only for dense, reliable points (e.g. local terrain distortion).

More complex models always fit the tie points better, so the fit RMSE alone
cannot be used to choose one. Instead, keep some tie points out of the fit
as independent **check points**:
:func:`~rsgispy.imageregistration.tiepoints.fit_geometric_model` fits a model
to the remaining points and
:func:`~rsgispy.imageregistration.tiepoints.evaluate_geometric_model` predicts
the reference coordinates of the check points. Choose the simplest model
whose check point error is close to the lowest.

.. literalinclude:: ../examples/image_coregistration.py
   :language: python
   :start-after: # [step-5-choose-model]
   :end-before: # [end-step-5-choose-model]
   :dedent: 4

.. code-block:: text

   Check point RMSE (m): {'shift': 18.6, 'affine': 5.7, 'poly2': 5.5}

The shift cannot represent the rotation (its error is almost a pixel at the
check points), while the affine and second order polynomial models are both
well below a pixel; the affine model is chosen as the simpler of the two.

Step 6 — Warping with the tie points
------------------------------------

:func:`rsgispy.imageregistration.warping.warp_with_tie_points` attaches the
tie points to the floating image as GCPs and resamples it with GDAL using the
chosen ``transform_type``. With ``match_ref_grid=True`` (the default) the
output has exactly the extent, pixel size and grid of the reference image, so
the two can be compared pixel by pixel; pixels outside the floating image are
set to ``no_data_val``. ``interp_method`` controls the resampling:
``rsgispy.INTERP_NEAREST_NEIGHBOUR`` for classified or other categorical
images, ``INTERP_BILINEAR`` or ``INTERP_CUBIC`` (sharper) for continuous
data. The function returns the transform type, the number of GCPs and the fit
RMSE.

.. literalinclude:: ../examples/image_coregistration.py
   :language: python
   :start-after: # [step-6-warp]
   :end-before: # [end-step-6-warp]
   :dedent: 4

Step 7 — The one-call workflow
------------------------------

:func:`rsgispy.imageregistration.warping.coregister_images` runs steps 3, 4
and 6 in one call with the same parameters, and returns the tie point
diagnostics (``n_tie_points``, ``n_inliers``, ``n_outliers``, ``rmse``,
``max_residual``). Use it in batch processing once you have chosen the
parameters with the step-by-step functions; ``out_gcp_file`` saves the inlier
tie points for checking.

.. literalinclude:: ../examples/image_coregistration.py
   :language: python
   :start-after: # [step-7-coregister]
   :end-before: # [end-step-7-coregister]
   :dedent: 4

Step 8 — Ground control points from a vector layer
--------------------------------------------------

When automatic matching is not possible — very different images (a
historical map, a scanned photograph), large changes, or when the reference
is a set of field (GNSS) measurements rather than an image — use ground
control points. Digitise each point on a recognisable feature in the
**floating** image (the point geometry) and record its true coordinates in
two attribute columns (here ``x_match`` and ``y_match``, in the floating
image's CRS unless ``gcp_epsg`` is given).
:func:`rsgispy.imageregistration.warping.add_vec_pts_as_gcps_to_img` stores
the points as GCPs in a copy of the image, and
:func:`rsgispy.imageregistration.warping.warp_with_gcps` warps it onto the
reference grid with a polynomial of order ``poly_order`` (1 = affine, needs
at least 3 points; 2 needs at least 6; 3 needs at least 10) or a thin-plate
spline (``use_tps=True``). Spread the points over the whole image, including
the corners; points clustered in one area give a model that is only valid
there.

.. literalinclude:: ../examples/image_coregistration.py
   :language: python
   :start-after: # [step-8-gcps]
   :end-before: # [end-step-8-gcps]
   :dedent: 4

GCPs stored in a text file (for example from ENVI or RSGISLib) or a table
can be attached with :func:`rsgispy.imageregistration.warping.gcp_to_gdal`
instead.

Step 9 — Assessing the result
-----------------------------

Three checks, all computed against the reference image:

* the **similarity metrics** from step 1 should increase
  (:func:`~rsgispy.imagecalc.comparison.calc_img_similarity_metrics`);
* the **residual offset** — running
  :func:`~rsgispy.imageregistration.shift.find_image_offset` on the
  registered image — should be close to zero (here it is measured to 0.25
  pixels, i.e. 5 m);
* a **difference image**
  (:func:`rsgispy.imagecalc.comparison.calc_img_difference`, reference minus
  registered) should not show the bright and dark fringes along edges that
  are typical of misregistration; what remains should be real change and the
  radiometric differences between the images.

.. literalinclude:: ../examples/image_coregistration.py
   :language: python
   :start-after: # [step-9-assess]
   :end-before: # [end-step-9-assess]
   :dedent: 4

.. code-block:: text

   before {'correlation': 0.829, 'norm_mutual_info': 0.15, 'resid_offset_m': 28.284}
   shift {'correlation': 0.846, 'norm_mutual_info': 0.161, 'resid_offset_m': 5.0}
   affine {'correlation': 0.857, 'norm_mutual_info': 0.182, 'resid_offset_m': 0.0}
   gcps {'correlation': 0.857, 'norm_mutual_info': 0.183, 'resid_offset_m': 5.0}

The whole-image shift removes most of the offset, but the affine warp from
the tie points also corrects the rotation and gives the best match, with no
measurable residual offset. The metrics improve much less than they would
for two copies of the same image, because most of the difference between the
dates is real change; the residual offset is the clearer check. The GCP warp
matches as well but is still 5 m off: its points only describe the
misregistration that the script added, not the 5 m by which the two dates
already differed, which the tie points measured from the images. For
a more independent check, measure the positions of a few features that were
not used as tie points or GCPs in both images and report their RMSE.

Performance and large datasets
------------------------------

* **Whole-image offset** —
  :func:`~rsgispy.imageregistration.shift.find_image_offset` reads the whole
  overlap into memory and evaluates every shift in the search window, so for
  large images use a subset or a coarser copy of both images (e.g. a 4×
  average-resampled VRT) to find the approximate offset, then refine it with
  tie points.
* **Tie points** — the cost is proportional to the number of candidates
  (image area / ``pixel_gap²``) times the window and search area. A few
  hundred to a few thousand candidates are enough for most models: increase
  ``pixel_gap`` for large images. Template matching uses a compiled kernel
  when ``numba`` is installed (for mutual information and for windows that
  contain no data; correlation on complete windows is vectorised with numpy
  either way). The tie points are calculated on a single band
  read into memory over the overlap; for very large images, generate the tie
  points on a reduced resolution copy. The warping functions use the tie
  points' floating image *pixel* coordinates (``flt_pxl_x``, ``flt_pxl_y``),
  so before warping the full resolution image, recalculate them from the map
  coordinates (``flt_x``, ``flt_y``) with the full image's transform (e.g.
  ``~src.transform * (flt_x, flt_y)`` with rasterio).
* **Warping** — the warps use GDAL, multi-threaded by default
  (``use_multi_threaded=True``), and write compressed outputs according to
  :mod:`rsgispy.imageutils.envopts`. Set ``calc_stats=False`` for
  intermediate files.
* **Batches** — choose the parameters on one or two image pairs with the
  step-by-step functions, then use
  :func:`~rsgispy.imageregistration.warping.coregister_images` for the rest,
  in parallel (one process per image).

Common problems
---------------

* **"No tie points were matched"** — the search area is smaller than the
  offset (measure it first with ``find_image_offset``), ``threshold`` is too
  high for the images, the windows are mostly no data or cloud, or the band
  has little texture (water, snow, uniform fields). Increase ``search_area``,
  lower ``threshold``, use a larger ``win_size`` or a different band.
* **"RANSAC failed to find a consensus set"** or most points are outliers —
  ``residual_thres`` is too small, the model is too simple for the distortion,
  or the matches are dominated by a repeated pattern. Look at the tie points
  in a GIS.
* **"Transform is not solvable"** (from GDAL) — the tie points are too few or
  lie on a line (e.g. only one row of clear windows). Use a smaller
  ``pixel_gap``, a simpler model (``TRANSFORM_SHIFT``) or add GCPs.
* **The result is worse in some areas** — a polynomial or thin-plate spline
  extrapolates badly outside the area covered by points. Use an affine model
  or add points near the edges.
* **Different CRSs** — shift and similarity transforms require the same CRS
  in both images; reproject the floating image first
  (:func:`rsgispy.imageutils.warp.reproject_image`). Affine, polynomial and
  TPS warps can work across CRSs, but matching is more reliable on the same
  grid.
* **Different resolutions** — resample the finer image to the coarser
  resolution before matching (average or bilinear resampling). The tie points
  then refer to the pixels of the resampled copy; to warp the original image,
  recalculate ``flt_pxl_x`` / ``flt_pxl_y`` as described under "Performance
  and large datasets".
* **No data** — set the no data value in both image headers (or pass
  ``ref_no_data`` / ``flt_no_data``) so that image borders and masked clouds
  are ignored, and pass ``no_data_val`` to the warp so the areas outside the
  floating image are marked as no data.
* **Categorical images** — warp classifications and masks with
  ``rsgispy.INTERP_NEAREST_NEIGHBOUR``; register them with the transform
  found for the corresponding continuous image (tie points cannot be matched
  reliably on class labels).

Related guides
--------------

* :doc:`analysis_ready_data`
* :doc:`change_detection`
