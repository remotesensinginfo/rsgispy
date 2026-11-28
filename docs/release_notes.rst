Release notes
=============

0.5.0
-----

This release fixes the bugs found in a full review of the package and makes
the API more consistent. Several fixes change results; check the sections
below if you upgrade from 0.4.x.

Integer outputs, clamping and no data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Integer outputs are rounded and clamped.** Every function that writes an
  integer raster now rounds to the nearest integer and clamps to the data
  type's range (previously values were truncated and could wrap around, e.g.
  ``400`` written as ``uint8`` became ``144``). Where a valid value would land
  on the no-data value it is moved one step away. The ``clamp_int``
  argument was removed. Outputs that previously truncated (e.g. slope or
  filters written as integers) can differ by 1.
* **One default output no-data value.** ``-9999`` for floating-point outputs,
  the type's minimum (signed) or maximum (unsigned) for integer outputs, and
  ``0`` for thematic class maps (:func:`rsgispy.imageprocess.default_out_nodata`).
  The optical and SAR indices previously used ``-999``.
* ``Cast`` and ``DiskSink`` remap a no-data value that the output type cannot
  hold (e.g. ``-9999`` to ``uint8``) to the type's default and record it.
* Per-pixel statistics (``calc_multi_img_band_stats``,
  ``image_pixel_column_summary``, ``get_img_sum_stats_in_pxl``) write and set
  the no-data value where no input is valid (previously ``0``).
* ``calc_area_under_curve`` masks no data and now uses the trapezoid rule
  between the band centres (``method='rectangle'`` keeps the old sum).
* ``resample_img_to_match`` and ``reproject_image`` take the source no-data
  value from the input header; without one, no pixels are masked.
* Change detection (CVA, SFA, IR-MAD, difference, ratio) compares a pixel only
  when every band is valid on both dates.
* Thematic statistics are exact for integer ranges wider than 65536 values.
* The per-pixel standard deviation (``image_pixel_column_summary``,
  ``calc_multi_img_band_stats``, ``get_img_sum_stats_in_pxl``) is the
  population one by default, as for ``calc_basic_img_band_stats``; pass
  ``ddof=1`` for the sample one (the previous behaviour).

Grids, CRSs and vectors
~~~~~~~~~~~~~~~~~~~~~~~

* Inputs whose pixel grids are offset by a fraction of a pixel now raise
  (resample them first) instead of being combined with a silent shift.
* Slope, aspect, hillshade, the incidence/exitance angles and ``shadow_mask``
  raise for a DEM in a geographic CRS (use the ``*_pxl_res_img`` variants).
* Vector spatial predicates, spatial joins, point counts and near distances
  reproject the second layer to the first one's CRS.
* ``vec_def_epsg`` is only used when the vector layer has no CRS.
* In-place ``*_file`` updates write back to the layer that was read.
* Empty vector results are written as empty layers (with a warning); empty
  zonal statistics inputs give empty results.
* ``calc_zonal_band_pxl_counts`` counts only valid pixels in its total and
  writes the number of no-data pixels to ``nodata_field``
  (``'nodata_count'``).
* The polygon zonal statistics take ``reproj_vec``.
* ``merge_utm_vecs_wgs84`` splits features at WGS84 zone boundaries and keeps
  point and line layers.
* UTM helpers handle longitude 180, the Norway and Svalbard exceptions and
  bounding boxes crossing the antimeridian; ``utm_from_epsg`` validates its
  input.

Machine learning
~~~~~~~~~~~~~~~~

* An integer ``cv`` gives shuffled (and, for classification, stratified)
  folds; fold assignments differ from earlier versions.
* The Optuna ``direction`` defaults to the direction of the metric
  (``rmse``/``mae`` are minimised); a conflicting direction raises.
* Prediction no data never collides with an output class.
* Area-weighted commission/omission are percentages; the old proportions are
  ``commission_area_prop`` / ``omission_area_prop``.
* ``lassolars_feature_selection`` standardises predictors and fits each target.
* New: PDP options for classification ``calc_feature_response_curves``,
  ``method`` for the SDM ``correlation_matrix``, repeated k-fold and jackknife
  fitting for classifiers, and per-pixel ensemble uncertainty
  (``out_uncert_img``) for regressors.

API changes
~~~~~~~~~~~

* ``calc_band_percentile`` and ``perform_dos_calc(dark_obj_percentile=...)``
  take percentiles in ``[0, 100]`` (they took 0-1); a value between 0 and 1
  gives a warning.
* ``polygonise_raster_to_vec_lyr`` uses the argument order of
  ``polygonise_raster_tiled`` (``input_img`` first, ``out_format``
  keyword-only) and takes ``no_data_val``. The old RSGISLib order is no
  longer accepted.
* ``kernel`` filters are true convolutions; ``apply_kernel_correlation_filter``
  gives the old correlation. Window filters, GLCM and morphology take
  ``border_mode``.
* ``DpRVI_VV``, ``DpRVI_HH`` and ``RVI4S1`` compute the published DpRVI
  (Mandal et al. 2020) instead of the simple dual-pol RVI.
* The ``rsgispy.zonalstats.METHOD_*`` constants were removed, and
  ``extract_avg_endmembers(pxl_in_poly_method=...)`` is replaced by
  ``all_touched`` and ``small_poly_fallback``.
* Otsu, Yen, ISODATA and the histogram cross-entropy use 256 histogram bins
  by default (``bins='fd'`` gives the previous Freedman-Diaconis rule), so
  thresholds can shift slightly.
* ``find_files_ext`` and ``find_files_mpaths_ext`` return a sorted list of
  paths (previously a dictionary keyed by file name, which lost same-named
  files in different folders).
* ``is_summer_winter`` treats the hemispheres as mirror images (April to
  September is the southern winter; it was May to October).
* New: ``extract_segment_stats(seg_no_data_val=...)``, a COG guard in
  ``set_img_pxl_values``, ``rsgispy.tools.files.safe_join``.

Other fixes
~~~~~~~~~~~

* Remote file names can no longer write outside the download folder (FTP,
  NASA CMR); FTP downloads use ``.incomplete`` files.
* Copernicus downloads refresh their token and continue after network errors.
* Tiled segmentation gives connected segments; strict local minima are found
  on the image border; ``eliminate_small_segments`` masks reference no data.
* File locks are atomic; ``delete_file_with_basename`` matches only the base
  name; date helpers reject non-positive intervals.
