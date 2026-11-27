Migrating from RSGISLib
=======================

rsgispy is a pure Python replacement for
`RSGISLib <http://www.rsgislib.org>`_. Most RSGISLib functions have an
rsgispy equivalent, usually with the same name and arguments, so porting a
script is mostly a matter of changing imports. This guide lists what is
different: where functions have moved, which were renamed or have different
arguments, how data types, formats and output options are given, and what has
not been ported.

Why migrate?

* rsgispy installs with ``pip`` and has no compiled RSGISLib library or GDAL
  Python bindings (``osgeo``) to build; it uses ``rasterio``, ``geopandas``,
  ``pyogrio``, ``shapely``, ``pyproj`` and ``numpy``.
* Vector functions accept and return :class:`geopandas.GeoDataFrame` objects
  as well as files, so steps can be chained in memory.
* Newer functionality (multi-output band maths, spectral and SAR index
  libraries, label raster zonal statistics, object-based machine learning
  using vectors rather than raster attribute tables (RATs), hydrology,
  geomedian, medoid and percentile composites)
  is only in rsgispy.

.. seealso::

   The `RSGISLib tutorials <https://github.com/remotesensinginfo/rsgislib-tutorials>`_
   have been ported to rsgispy as the
   `rsgispy tutorials <https://github.com/remotesensinginfo/rsgispy-tutorials>`_
   (see :doc:`../tutorials`). Comparing a notebook in the two repositories
   shows the port of a complete analysis, and the notebooks point out the
   differences in "RSGISLib users" notes.

.. contents:: Contents
   :local:
   :depth: 1

Installation
------------

.. code-block:: bash

   pip install rsgispy

Optional functionality is installed with extras, for example
``pip install "rsgispy[classification,vector]"``; see :doc:`../index` for the
full list. RSGISLib and rsgispy do not depend on each other and can be
installed in the same environment, which is useful while porting: run the old
and new versions of a step on the same data and compare the outputs (for
example with :func:`rsgispy.imagecalc.comparison.are_imgs_equal`).

Overview of the differences
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Topic
     - RSGISLib
     - rsgispy
   * - Imports
     - ``import rsgislib.imagecalc``; most functions in one module.
     - Functions are in submodules: ``import rsgispy.imagecalc.maths``.
   * - Output format
     - ``gdalformat`` argument, often positional; KEA commonly used.
     - ``gdalformat`` keyword-only, default ``"GTIFF"``.
   * - Data types
     - ``rsgislib.TYPE_*`` constants.
     - ``rsgispy.TYPE_*`` constants with the same names.
   * - Output options
     - Positional arguments.
     - Keyword-only (after ``*`` in the signature): ``gdalformat``,
       ``datatype``, ``calc_stats``, ``use_pbar``, ``creation_opts``, ...
       All functions have a ``use_pbar`` argument to disable the progress
       bar, which is useful when running functions within a loop, as the
       progress bars can make the output unreadable.
   * - Vector inputs
     - File path and layer name.
     - File path and layer name, or a GeoDataFrame (layer ``None``).
   * - Raster attribute tables
     - ``rsgislib.rastergis`` (KEA).
     - Not ported: label rasters plus tables (Parquet); class names and
       colours stored in the image
       (:func:`rsgispy.imageutils.colourtable.set_thematic_classes`).
   * - Endmembers
     - ``.mtxt`` text files.
     - :class:`pandas.DataFrame` or CSV/Parquet/Excel; convert ``.mtxt``
       files once.

Imports and module layout
-------------------------

RSGISLib modules such as ``rsgislib.imagecalc`` and ``rsgislib.imageutils``
hold hundreds of functions each. In rsgispy the same functions are grouped
into submodules by topic (``rsgispy.imagecalc.maths``,
``rsgispy.imagecalc.stats``, ``rsgispy.imageutils.metadata``, ...), and the
three RSGISLib vector modules are replaced by one package,
:mod:`rsgispy.vector`. The package ``__init__`` modules do not re-export
every function, so import the submodule:

.. code-block:: python

   # RSGISLib
   import rsgislib.imageutils
   rsgislib.imageutils.get_img_res("s2_sref.kea")

.. code-block:: python

   # rsgispy
   import rsgispy.imageutils.metadata

   print(rsgispy.imageutils.metadata.get_img_res("s2_sref.tif"))

The tables in :ref:`rsgislib-migration-tables` give the new module of each
common function. For anything not listed, search the API reference (the
search box finds functions by name) or the module pages, for example
:doc:`../rsgispy_imageutils`.

Band maths and BandDefn
-----------------------

:class:`rsgispy.imagecalc.maths.BandDefn` takes the same three values as in
RSGISLib (variable name, image, 1-based band). The function is called
:func:`rsgispy.imagecalc.maths.band_maths` (not ``band_math``) and its
arguments are in a different order: the band definitions come third, and the
format and data type are keyword-only.

.. code-block:: python

   # RSGISLib
   import rsgislib
   import rsgislib.imagecalc

   band_defs = [
       rsgislib.imagecalc.BandDefn("red", "s2_sref.kea", 3),
       rsgislib.imagecalc.BandDefn("nir", "s2_sref.kea", 4),
   ]
   rsgislib.imagecalc.band_math(
       "ndvi.kea", "(nir - red) / (nir + red)", "KEA", rsgislib.TYPE_32FLOAT,
       band_defs,
   )

.. code-block:: python

   # rsgispy
   import rsgispy
   import rsgispy.imagecalc.maths

   band_defs = [
       rsgispy.imagecalc.maths.BandDefn("red", "s2_sref.tif", 3),
       rsgispy.imagecalc.maths.BandDefn("nir", "s2_sref.tif", 4),
   ]
   rsgispy.imagecalc.maths.band_maths(
       "ndvi.tif",
       "(nir - red) / (nir + red)",
       band_defs,
       gdalformat="GTIFF",
       datatype=rsgispy.TYPE_32FLOAT,
       use_pbar=False,
   )

The expressions use the same muParser syntax (``?:`` conditionals, ``&&``,
``||``, ``min``, ``max``, ...), evaluated by
`pymu-eval <https://github.com/remotesensinginfo/pymu-eval>`_, so existing
expressions can be copied unchanged. rsgispy adds multi-output expressions
(``"expr1, expr2"`` writes two bands) and intermediate variables; see
:doc:`../rsgispy_imagecalc`. For spectral indices, use
:func:`rsgispy.imagecalc.calcindices.calc_indices` rather than writing the
expression yourself.

Data types, formats and creation options
----------------------------------------

**Data types.** Use the ``rsgispy.TYPE_*`` constants (``TYPE_8UINT``,
``TYPE_16UINT``, ``TYPE_32INT``, ``TYPE_32FLOAT``, ...), which have the same
names as the RSGISLib constants, so ``rsgislib.TYPE_32FLOAT`` becomes
``rsgispy.TYPE_32FLOAT``. Use the names rather than the integer values.
:func:`rsgispy.imageutils.metadata.get_rsgislib_datatype_from_img` returns
the constant for an existing image, and :func:`rsgispy.get_numpy_datatype`
converts a constant to a NumPy type. Interpolation (``rsgispy.INTERP_*``) and
summary statistic (``rsgispy.SUMTYPE_*``) constants are also in the top-level
:mod:`rsgispy` module.

**Formats.** ``gdalformat`` takes the same GDAL driver names (``"GTIFF"``,
``"COG"``, ``"KEA"``, ...), but the default is ``"GTIFF"`` everywhere. Images
are written with rasterio, so only drivers in rasterio's GDAL build can be
used. KEA in particular needs a GDAL with the KEA driver (for example from
conda-forge with ``kealib``); the GDAL in the rasterio wheels from PyPI may
not include it. GeoTIFF or COG is recommended for new work.

**Creation options and block size.** The default GeoTIFF creation options
differ from RSGISLib's. With no environment variable set, RSGISLib writes
GDAL's default GeoTIFF (striped and uncompressed), whereas rsgispy writes
tiled, DEFLATE-compressed GeoTIFFs (``TILED=YES``, ``COMPRESS=DEFLATE``,
``INTERLEAVE=BAND``, ``BIGTIFF=IF_SAFER``; see
:data:`rsgispy.imageprocess.DEFAULT_IMG_CREATION_OPTS`). rsgispy builds this
default into :func:`rsgispy.imageprocess.get_gdal_img_creation_opts` and does
not set any environment variables, so importing rsgispy does not change the
output of RSGISLib functions in the same script. Creation options set with
environment variables are still read and replace the default:
``RSGISPY_IMG_CRT_OPTS_<FORMAT>`` first, then the RSGISLib name
``RSGISLIB_IMG_CRT_OPTS_<FORMAT>`` (for example
``RSGISLIB_IMG_CRT_OPTS_GTIFF="TILED=YES:COMPRESS=LZW"``); an empty variable
means no creation options, as in RSGISLib. The helper functions keep their
RSGISLib names, e.g.
:func:`rsgispy.imageutils.envopts.set_env_vars_lzw_gtiff_outs`, and set both
variables; :func:`rsgispy.imageutils.envopts.set_img_creation_opts` with an
empty dictionary gives plain GeoTIFFs and
:func:`rsgispy.imageutils.envopts.reset_img_creation_opts` restores the
rsgispy default. Many functions also take
``creation_opts={"COMPRESS": "LZW"}`` for a single output, merged on top.
The processing block size is read from ``RSGISPY_BLOCK_SIZE`` or
``RSGISLIB_BLOCK_SIZE`` (:data:`rsgispy.imageprocess.ENV_VAR_BLOCK_SIZE`,
:data:`rsgispy.imageprocess.ENV_VAR_RSGISLIB_BLOCK_SIZE`), or set with
:func:`rsgispy.imageprocess.set_default_block_size`.

**Statistics and pyramids.** Many rsgispy functions that write an image have
``calc_stats=True`` by default, so a separate call to
:func:`rsgispy.imageutils.stats.pop_img_stats` is often no longer needed.
Note that ``pop_img_stats`` has no ``use_no_data`` argument: its second
argument is ``no_data_val``, so pass it by keyword.

Keyword-only output options
---------------------------

In rsgispy, the options that control how an output is written (format, data
type, no data value, statistics, progress bar, creation options) come after
a ``*`` in the signature and must be given by name. An RSGISLib call that
passes them by position usually fails with a ``TypeError``:

.. code-block:: text

   >>> rsgispy.elevation.terrain.slope("dem.tif", "slope.tif", "degrees", "GTIFF")
   TypeError: slope() takes from 2 to 3 positional arguments but 4 were given

Fix the call by naming the options:

.. code-block:: python

   import rsgispy
   import rsgispy.imagecalc.maths

   rsgispy.imagecalc.maths.image_math(
       "s2_sref.tif", "s2_refl.tif", "b1 / 10000.0",
       gdalformat="GTIFF", datatype=rsgispy.TYPE_32FLOAT, use_pbar=False,
   )

This is not guaranteed when the positional arguments themselves changed.
For example, the RSGISLib call ``pop_img_stats(img, True, 0, True)`` runs
in rsgispy without an error but uses ``True`` as the no data value, and
``band_maths(out, exp, "KEA", rsgispy.TYPE_32FLOAT, band_defs)`` passes
``"KEA"`` as the band definitions (see :ref:`rsgislib-changed-table`).
The "Notes" column in :ref:`rsgislib-same-name-table` shows which functions
have keyword-only ``gdalformat`` and ``datatype`` (read from their
signatures). When porting, it is safest to name every argument after the
first two or three.

Vector inputs and GeoDataFrames
-------------------------------

Vector functions take the same ``vec_file, vec_lyr`` pair as in RSGISLib,
but ``vec_file`` can also be a :class:`geopandas.GeoDataFrame` (with
``vec_lyr=None``). When the output file is ``None``, most functions return a
GeoDataFrame instead of writing a file, so intermediate layers do not need
to be written to disk. Zonal statistics functions with a ``_file`` suffix
still update the input file in place when no ``out_vec_file`` is given, as
in RSGISLib; the versions without the suffix take and return a GeoDataFrame:

.. code-block:: python

   import geopandas as gpd
   import rsgispy.zonalstats.polygon

   fields_gdf = gpd.read_file("fields.gpkg")
   fields_gdf = rsgispy.zonalstats.polygon.calc_zonal_band_stats(
       fields_gdf, "ndvi.tif", img_band=1, min_thres=-1, max_thres=1,
       out_no_data_val=-999, mean_field="ndvi_mean", use_pbar=False,
   )
   print(fields_gdf[["ndvi_mean"]].head())

Vector layers are read and written with ``pyogrio``; GeoPackage is the
default format and GeoParquet (``.parquet``) is supported throughout. Layers
in different CRSs are usually reprojected or rejected with an error rather
than combined silently. See :doc:`zonal_stats_and_vectors` for more vector
and zonal statistics recipes.

Endmembers from .mtxt files
---------------------------

RSGISLib stored endmember spectra in ``.mtxt`` matrix files. rsgispy uses a
:class:`pandas.DataFrame` with one row per band and one column per
endmember, which can be saved as CSV, Parquet or Excel.
:func:`rsgispy.imagespectral.endmembers.convert_mtxt_to_df` converts an old
file once (``gain`` divides the values, e.g. to rescale reflectance):

.. code-block:: python

   import rsgispy.imagespectral.endmembers
   import rsgispy.imagespectral.unmixing

   em_df = rsgispy.imagespectral.endmembers.convert_mtxt_to_df(
       "endmembers.mtxt",
       endmember_names=["vegetation", "soil"],
       band_names=["Blue", "Green", "Red", "NIR"],
       out_file="endmembers.csv",
   )
   rsgispy.imagespectral.unmixing.spec_unmix_ucls(
       "s2_sref.tif", "s2_unmix.tif", em_df, use_pbar=False
   )

The unmixing functions are in :mod:`rsgispy.imagespectral.unmixing` and
accept the DataFrame or the converted file. Morphology operator files
(``.gmtxt``, :func:`rsgispy.imagemorphology.create_circular_op`) and the
RSGISLib GCP text formats
(:data:`rsgispy.imageregistration.TYPE_RSGIS_IMG2MAP`,
:data:`rsgispy.imageregistration.TYPE_RSGIS_MAPOFFS`) are still supported.

What is not ported
------------------

* **Raster attribute tables** (``rsgislib.rastergis``) and other
  KEA-specific functionality. Object-based workflows use a label raster and a
  table of statistics per label instead:
  :func:`rsgispy.zonalstats.labels.calc_label_zonal_stats` replaces
  populating a RAT with statistics, and
  :mod:`rsgispy.classification.classsegments` classifies the segments as a
  vector layer. See :doc:`segment_ml_workflow`. The class names and colours
  of a thematic image (``rastergis.set_class_names_colours``) are set with
  :func:`rsgispy.imageutils.colourtable.set_thematic_classes`, which writes
  the colour table and stores the names in the GeoTIFF without a RAT.
* **GDAL/OGR objects.** Functions that took or returned ``osgeo`` dataset or
  layer objects take file paths (or GeoDataFrames) instead; there is no
  ``osgeo`` dependency.
* **camelCase names** from RSGISLib 4 and earlier (``bandMath``,
  ``imageMath``, ...). Use the snake_case names from the tables below.
* **Anything else.** If a function you use is not in the tables or the API
  reference, check the RSGISLib documentation for what it does and look for
  the rsgispy function by topic; if there is no equivalent, open an issue at
  https://github.com/remotesensinginfo/rsgispy/issues.

.. _rsgislib-migration-tables:

Function and module tables
--------------------------

The tables are generated by ``docs/examples/gen_rsgislib_migration_table.py``,
which locates each function in rsgispy and reads the notes from its
signature; ``tests/test_migrating_from_rsgislib_guide.py`` checks that every
rsgispy name exists and that the tables are up to date. The RSGISLib names
are the RSGISLib 5 names where known; where the RSGISLib arguments are not
certain, the notes say to check the RSGISLib documentation.

.. begin-generated-tables

.. _rsgislib-module-table:

Module mapping
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - RSGISLib module
     - rsgispy module(s)
     - Notes
   * - ``rsgislib``
     - :mod:`rsgispy`
     - ``TYPE_*``, ``INTERP_*``, ``SUMTYPE_*`` and other constants.
   * - ``rsgislib.imagecalc``
     - :mod:`rsgispy.imagecalc`
     - Split into submodules (``maths``, ``stats``, ``percentiles``, ``distance``, ...).
   * - ``rsgislib.imagecalc.calcindices``
     - :mod:`rsgispy.imagecalc.calcindices`
     - One function for all indices.
   * - ``rsgislib.imageutils``
     - :mod:`rsgispy.imageutils`
     - Split into submodules (``metadata``, ``bands``, ``blank``, ``masking``, ``warp``, ...).
   * - ``rsgislib.vectorutils``
     - :mod:`rsgispy.vector`, :mod:`rsgispy.vectorprocess`
     - ``rsgispy.vector.{io,create,geoms,attrs,stats,selection}``.
   * - ``rsgislib.vectorattrs``
     - :mod:`rsgispy.vector.attrs`
     -
   * - ``rsgislib.vectorgeoms``
     - :mod:`rsgispy.vector.geoms`
     -
   * - ``rsgislib.zonalstats``
     - :mod:`rsgispy.zonalstats`
     - ``polygon``, ``point`` and the new ``labels`` submodule.
   * - ``rsgislib.rastergis``
     - :mod:`rsgispy.zonalstats.labels`, :mod:`rsgispy.imageutils.colourtable`
     - Not ported; label raster statistics written to tables instead, and class names and colours stored in the image.
   * - ``rsgislib.imageutils.imagecomp``
     - :mod:`rsgispy.imageutils.composite`
     - Also geomedian, medoid, percentile and quality mosaic composites.
   * - ``rsgislib.segmentation``
     - :mod:`rsgispy.segmentation`
     - One submodule per algorithm.
   * - ``rsgislib.classification``
     - :mod:`rsgispy.classification`, :mod:`rsgispy.ml`
     - Configuration objects; shared ML tools in ``rsgispy.ml``.
   * - ``rsgislib.elevation``
     - :mod:`rsgispy.elevation`
     - ``terrain``, ``illumination``, ``filtering``, ``hydrology``, ...
   * - ``rsgislib.imagefilter``
     - :mod:`rsgispy.imagefilter`
     - ``stats``, ``smooth``, ``edge``, ``sar``, ``texture``, ``chained``.
   * - ``rsgislib.imagemorphology``
     - :mod:`rsgispy.imagemorphology`
     -
   * - ``rsgislib.imagecalibration``
     - :mod:`rsgispy.imagecalibration`
     -
   * - ``rsgislib.imageregistration``
     - :mod:`rsgispy.imageregistration`
     - ``shift``, ``tiepoints``, ``warping``.
   * - ``rsgislib.tools``
     - :mod:`rsgispy.tools`
     - e.g. ``tools.filetools`` is ``rsgispy.tools.files``; ``tools.utils`` is split (``jsonio``, ``strings``, ``dicts``, ...).

.. _rsgislib-changed-table:

Renamed functions and changed arguments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 32 33 35

   * - RSGISLib
     - rsgispy
     - What changed
   * - ``rsgislib.imagecalc.band_math``
     - :func:`rsgispy.imagecalc.maths.band_maths`
     - Renamed (``maths``). ``band_defs`` is the third positional argument; ``gdalformat`` and ``datatype`` are keyword-only.
   * - ``rsgislib.imagecalc.bandMath (RSGISLib 4)``
     - :func:`rsgispy.imagecalc.maths.band_maths`
     - As above; no camelCase names in rsgispy.
   * - ``rsgislib.imagecalc.imageMath (RSGISLib 4)``
     - :func:`rsgispy.imagecalc.maths.image_math`
     - As above.
   * - ``rsgislib.imagecalc.calcindices.calc_ndvi (and the other per-index functions)``
     - :func:`rsgispy.imagecalc.calcindices.calc_indices`
     - One function: ``indices=['NDVI', ...]`` and ``band_names`` mapping band index to band name (e.g. ``{3: 'red', 4: 'nir'}``).
   * - ``rsgislib.segmentation.shepherdseg.run_shepherd_segmentation``
     - :func:`rsgispy.segmentation.shepherd.shepherd_segmentation`
     - No temporary directory or KEA intermediate files; check the argument names against the RSGISLib docs.
   * - ``rsgislib.vectorutils.createrasters.rasterise_vec_lyr``
     - :func:`rsgispy.imageutils.rasterise.rasterise_vec_lyr`
     - Moved to ``imageutils``; output options keyword-only.
   * - ``rsgislib.rastergis.populate_rat_with_stats``
     - :func:`rsgispy.zonalstats.labels.calc_label_zonal_stats`
     - Statistics go to a table (one row per label) instead of a raster attribute table; join to polygons with :func:`rsgispy.zonalstats.labels.join_label_stats_to_vec`.
   * - ``rsgislib.imageutils.pop_img_stats``
     - :func:`rsgispy.imageutils.stats.pop_img_stats`
     - No ``use_no_data`` flag: ``no_data_val`` is the second argument (``None`` uses the image's no data value). Pass it by keyword.
   * - ``rsgislib.imageutils.stack_img_bands``
     - :func:`rsgispy.imageutils.bands.stack_img_bands`
     - rsgispy order is ``(input_imgs, output_img, band_names, skip_value)``; compare with your RSGISLib calls (check the RSGISLib docs) and use keyword arguments.
   * - ``rsgislib.imageutils.create_blank_img``
     - :func:`rsgispy.imageutils.blank.create_blank_img`
     - Projection given as ``epsg_code`` instead of WKT, after a new fill value argument ``pxl_val`` (check the RSGISLib docs for the arguments it replaces); use keyword arguments.
   * - ``rsgislib.classification.classsklearn.train_sklearn_classifier``
     - :func:`rsgispy.classification.classsklearn.train_sklearn_classifier`
     - Same name; takes a :class:`rsgispy.classification.classdata.ClassificationConfig` and sample tables, so the arguments differ.
   * - ``rsgislib.classification.classsklearn.apply_sklearn_classifier``
     - :func:`rsgispy.classification.classsklearn.apply_sklearn_classifier`
     - As above.
   * - ``rsgislib.imageutils.imagecomp.create_max_ndvi_composite``
     - :func:`rsgispy.imageutils.composite.create_max_ndvi_composite`
     - Same name; ``(input_imgs, output_img, red_band, nir_band)`` with an optional ``out_ref_img`` (a thematic GeoTIFF of the 1-based image index with a colour table, not a KEA RAT) and no ``tmp_dir``. A preset of :func:`rsgispy.imageutils.composite.create_quality_mosaic`.
   * - ``rsgislib.imageutils.imagecomp.create_max_ndvi_ndwi_composite``
     - :func:`rsgispy.imageutils.composite.create_max_ndvi_ndwi_composite`
     - Same name; no ``ref_sp_img``: the images must already be on one grid (:func:`rsgispy.imageutils.warp.resample_img_to_match`). ``r_band``, ``n_band``, ``s_band`` become ``red_band``, ``nir_band``, ``swir_band`` (the RSGISLib NDWI and land / water rule); the outputs ``out_ref_img`` and ``out_lw_msk_img`` are optional keywords. A preset of :func:`rsgispy.imageutils.composite.create_land_water_composite`.
   * - ``rsgislib.imagecalc.get_img_idx_for_stat and rsgislib.imageutils.create_ref_img_composite_img``
     - :func:`rsgispy.imageutils.composite.create_quality_mosaic`
     - One call: the observation with the ``mode='max'``, ``'min'`` or ``'percentile'`` value of a quality (a band, an expression or a callable) of each image; the index image is ``out_ref_img``.
   * - ``rsgislib.imagecalc.get_2d_img_histogram``
     - :func:`rsgispy.imagecalc.stats.get_img_2d_histogram`
     - Returns ``(hist, x_edges, y_edges)`` arrays instead of writing an image and returning the bin widths; ``bins``, ``x_range``, ``y_range`` and ``density`` as :func:`numpy.histogram2d`; no scale or offset.
   * - ``rsgislib.rastergis.set_class_names_colours``
     - :func:`rsgispy.imageutils.colourtable.set_thematic_classes`
     - No RAT or column name: ``classes`` maps each pixel value to ``(name, colour)``; the colours go in the colour table and the names in a band tag, read with :func:`rsgispy.imageutils.colourtable.get_thematic_classes`.
   * - ``rsgislib.segmentation.label_pixels_from_cluster_centres``
     - :func:`rsgispy.classification.clustersklearn.apply_img_pixel_clusterer`
     - Takes a fitted scikit-learn clusterer (e.g. ``KMeans``, which labels each pixel with its nearest centre) or its ``.joblib`` file instead of a cluster centres file; clusters are numbered from ``out_id_offset``.
   * - ``rsgislib.regression.regresssklearn.perform_kfold_fit``
     - :func:`rsgispy.regression.regsklearn.perform_kfold_fit`
     - Same name; ``n_repeats`` (default 20, seeded with ``rnd_seed``) replaces ``repeats`` / ``shuffle``, a scikit-learn ``Pipeline`` replaces ``data_scaler``, and ``groups`` keeps groups together. Returns ``(metrics, residuals)`` dictionaries of DataFrames per target; summarise with :func:`rsgispy.regression.regsklearn.summarise_kfold_metrics`.

.. _rsgislib-same-name-table:

Functions with the same name in a new module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 38 42 20

   * - RSGISLib
     - rsgispy
     - Notes (from the signature)
   * - ``rsgislib.imagecalc.BandDefn``
     - :class:`rsgispy.imagecalc.maths.BandDefn`
     -
   * - ``rsgislib.imagecalc.image_math``
     - :func:`rsgispy.imagecalc.maths.image_math`
     - ``gdalformat``, ``datatype`` keyword-only
   * - ``rsgislib.imagecalc.image_band_math``
     - :func:`rsgispy.imagecalc.maths.image_band_math`
     - ``gdalformat``, ``datatype`` keyword-only
   * - ``rsgislib.imagecalc.calc_prop_true_exp``
     - :func:`rsgispy.imagecalc.maths.calc_prop_true_exp`
     -
   * - ``rsgislib.imagecalc.calc_dist_to_img_vals``
     - :func:`rsgispy.imagecalc.distance.calc_dist_to_img_vals`
     - ``gdalformat`` keyword-only
   * - ``rsgislib.imagecalc.count_pxls_of_val``
     - :func:`rsgispy.imagecalc.stats.count_pxls_of_val`
     -
   * - ``rsgislib.imagecalc.get_unique_values``
     - :func:`rsgispy.imagecalc.stats.get_unique_values`
     -
   * - ``rsgislib.imageutils.create_copy_img``
     - :func:`rsgispy.imageutils.blank.create_copy_img`
     - ``gdalformat``, ``datatype`` keyword-only
   * - ``rsgislib.imageutils.set_band_names``
     - :func:`rsgispy.imageutils.metadata.set_band_names`
     -
   * - ``rsgislib.imageutils.get_band_names``
     - :func:`rsgispy.imageutils.metadata.get_band_names`
     -
   * - ``rsgislib.imageutils.get_img_res``
     - :func:`rsgispy.imageutils.metadata.get_img_res`
     -
   * - ``rsgislib.imageutils.get_img_size``
     - :func:`rsgispy.imageutils.metadata.get_img_size`
     -
   * - ``rsgislib.imageutils.get_img_band_count``
     - :func:`rsgispy.imageutils.metadata.get_img_band_count`
     -
   * - ``rsgislib.imageutils.get_img_no_data_value``
     - :func:`rsgispy.imageutils.metadata.get_img_no_data_value`
     -
   * - ``rsgislib.imageutils.set_img_no_data_value``
     - :func:`rsgispy.imageutils.metadata.set_img_no_data_value`
     -
   * - ``rsgislib.imageutils.get_img_bbox``
     - :func:`rsgispy.imageutils.metadata.get_img_bbox`
     -
   * - ``rsgislib.imageutils.get_epsg_proj_from_img``
     - :func:`rsgispy.imageutils.metadata.get_epsg_proj_from_img`
     -
   * - ``rsgislib.imageutils.get_rsgislib_datatype_from_img``
     - :func:`rsgispy.imageutils.metadata.get_rsgislib_datatype_from_img`
     -
   * - ``rsgislib.imageutils.resample_img_to_match``
     - :func:`rsgispy.imageutils.warp.resample_img_to_match`
     - ``gdalformat``, ``datatype`` keyword-only
   * - ``rsgislib.imageutils.reproject_image``
     - :func:`rsgispy.imageutils.warp.reproject_image`
     - ``gdalformat``, ``datatype`` keyword-only
   * - ``rsgislib.imageutils.mask_img``
     - :func:`rsgispy.imageutils.masking.mask_img`
     - ``gdalformat``, ``datatype`` keyword-only
   * - ``rsgislib.imageutils.gen_valid_mask``
     - :func:`rsgispy.imageutils.masking.gen_valid_mask`
     - ``gdalformat``, ``datatype`` keyword-only
   * - ``rsgislib.imageutils.subset_bbox``
     - :func:`rsgispy.imageutils.subsetting.subset_bbox`
     - ``gdalformat``, ``datatype`` keyword-only
   * - ``rsgislib.imageutils.subset_to_img``
     - :func:`rsgispy.imageutils.subsetting.subset_to_img`
     - ``gdalformat``, ``datatype`` keyword-only
   * - ``rsgislib.imageutils.create_img_mosaic``
     - :func:`rsgispy.imageutils.mosaic.create_img_mosaic`
     - ``gdalformat``, ``datatype`` keyword-only
   * - ``rsgislib.imageutils.include_imgs``
     - :func:`rsgispy.imageutils.mosaic.include_imgs`
     -
   * - ``rsgislib.imageutils.create_stack_images_vrt``
     - :func:`rsgispy.imageutils.vrt.create_stack_images_vrt`
     -
   * - ``rsgislib.imageutils.define_colour_table``
     - :func:`rsgispy.imageutils.colourtable.define_colour_table`
     -
   * - ``rsgislib.imageutils.set_img_thematic``
     - :func:`rsgispy.imageutils.colourtable.set_img_thematic`
     -
   * - ``rsgislib.imageutils.set_env_vars_lzw_gtiff_outs``
     - :func:`rsgispy.imageutils.envopts.set_env_vars_lzw_gtiff_outs`
     -
   * - ``rsgislib.imageutils.set_env_vars_deflate_gtiff_outs``
     - :func:`rsgispy.imageutils.envopts.set_env_vars_deflate_gtiff_outs`
     -
   * - ``rsgislib.vectorutils.get_vec_feat_count``
     - :func:`rsgispy.vector.io.get_vec_feat_count`
     - accepts a GeoDataFrame
   * - ``rsgislib.vectorutils.get_vec_lyr_cols``
     - :func:`rsgispy.vector.io.get_vec_lyr_cols`
     - accepts a GeoDataFrame
   * - ``rsgislib.vectorutils.get_vec_lyrs_lst``
     - :func:`rsgispy.vector.io.get_vec_lyrs_lst`
     - accepts a GeoDataFrame
   * - ``rsgislib.vectorutils.get_epsg_proj_from_vec``
     - :func:`rsgispy.vector.io.get_epsg_proj_from_vec`
     - accepts a GeoDataFrame
   * - ``rsgislib.vectorutils.reproj_vec_lyr``
     - :func:`rsgispy.vector.io.reproj_vec_lyr`
     - accepts a GeoDataFrame
   * - ``rsgislib.vectorutils.merge_vec_lyrs``
     - :func:`rsgispy.vector.io.merge_vec_lyrs`
     - can return a GeoDataFrame
   * - ``rsgislib.vectorutils.vector_translate``
     - :func:`rsgispy.vector.io.vector_translate`
     - accepts a GeoDataFrame
   * - ``rsgislib.vectorutils.delete_vec_file``
     - :func:`rsgispy.vector.io.delete_vec_file`
     -
   * - ``rsgislib.vectorutils.spatial_select``
     - :func:`rsgispy.vector.selection.spatial_select`
     - accepts a GeoDataFrame
   * - ``rsgislib.vectorutils.create_train_test_smpls``
     - :func:`rsgispy.vector.selection.create_train_test_smpls`
     - accepts a GeoDataFrame
   * - ``rsgislib.vectorutils.createvectors.polygonise_raster_to_vec_lyr``
     - :func:`rsgispy.vector.create.polygonise_raster_to_vec_lyr`
     - can return a GeoDataFrame
   * - ``rsgislib.vectorattrs.write_vec_column``
     - :func:`rsgispy.vector.attrs.write_vec_column`
     - accepts a GeoDataFrame
   * - ``rsgislib.vectorattrs.read_vec_column``
     - :func:`rsgispy.vector.attrs.read_vec_column`
     - accepts a GeoDataFrame
   * - ``rsgislib.vectorattrs.read_vec_columns``
     - :func:`rsgispy.vector.attrs.read_vec_columns`
     - accepts a GeoDataFrame
   * - ``rsgislib.vectorattrs.add_fid_col``
     - :func:`rsgispy.vector.attrs.add_fid_col`
     - accepts a GeoDataFrame
   * - ``rsgislib.vectorgeoms.buffer_vec_layer_gp``
     - :func:`rsgispy.vector.geoms.buffer_vec_layer_gp`
     - accepts a GeoDataFrame
   * - ``rsgislib.vectorgeoms.explode_vec_lyr``
     - :func:`rsgispy.vector.geoms.explode_vec_lyr`
     - accepts a GeoDataFrame
   * - ``rsgislib.vectorgeoms.delete_polygon_holes``
     - :func:`rsgispy.vector.geoms.delete_polygon_holes`
     - accepts a GeoDataFrame
   * - ``rsgislib.zonalstats.calc_zonal_band_stats_file``
     - :func:`rsgispy.zonalstats.polygon.calc_zonal_band_stats_file`
     - ``out_vec_file`` keyword-only; accepts a GeoDataFrame
   * - ``rsgislib.zonalstats.calc_zonal_band_stats_test_poly_pts_file``
     - :func:`rsgispy.zonalstats.polygon.calc_zonal_band_stats_test_poly_pts_file`
     - ``out_vec_file`` keyword-only; accepts a GeoDataFrame
   * - ``rsgislib.zonalstats.ext_point_band_values_file``
     - :func:`rsgispy.zonalstats.point.ext_point_band_values_file`
     - ``out_vec_file`` keyword-only; accepts a GeoDataFrame
   * - ``rsgislib.elevation.slope``
     - :func:`rsgispy.elevation.terrain.slope`
     - ``gdalformat``, ``datatype`` keyword-only
   * - ``rsgislib.elevation.aspect``
     - :func:`rsgispy.elevation.terrain.aspect`
     - ``gdalformat``, ``datatype`` keyword-only
   * - ``rsgislib.elevation.hillshade``
     - :func:`rsgispy.elevation.illumination.hillshade`
     - ``gdalformat``, ``datatype`` keyword-only
   * - ``rsgislib.elevation.shadow_mask``
     - :func:`rsgispy.elevation.illumination.shadow_mask`
     - ``gdalformat``, ``datatype`` keyword-only
   * - ``rsgislib.elevation.local_incidence_angle``
     - :func:`rsgispy.elevation.illumination.local_incidence_angle`
     - ``gdalformat``, ``datatype`` keyword-only
   * - ``rsgislib.elevation.local_exitance_angle``
     - :func:`rsgispy.elevation.illumination.local_exitance_angle`
     - ``gdalformat``, ``datatype`` keyword-only
   * - ``rsgislib.elevation.fill_dem_soille_gratin_1994``
     - :func:`rsgispy.elevation.filtering.fill_dem_soille_gratin_1994`
     - ``gdalformat``, ``datatype`` keyword-only
   * - ``rsgislib.elevation.dtm_aspect_median_filter``
     - :func:`rsgispy.elevation.filtering.dtm_aspect_median_filter`
     - ``gdalformat``, ``datatype`` keyword-only
   * - ``rsgislib.segmentation.clump``
     - :func:`rsgispy.segmentation.clump.clump`
     - ``gdalformat``, ``datatype`` keyword-only
   * - ``rsgislib.segmentation.union_of_clumps``
     - :func:`rsgispy.segmentation.clump.union_of_clumps`
     - ``gdalformat``, ``datatype`` keyword-only
   * - ``rsgislib.imagefilter.apply_median_filter``
     - :func:`rsgispy.imagefilter.stats.apply_median_filter`
     - ``gdalformat``, ``datatype`` keyword-only
   * - ``rsgislib.imagefilter.apply_mean_filter``
     - :func:`rsgispy.imagefilter.stats.apply_mean_filter`
     - ``gdalformat``, ``datatype`` keyword-only
   * - ``rsgislib.imagemorphology.create_circular_op``
     - :func:`rsgispy.imagemorphology.create_circular_op`
     -
   * - ``rsgislib.imagemorphology.image_dilate``
     - :func:`rsgispy.imagemorphology.image_dilate`
     - ``gdalformat``, ``datatype`` keyword-only
   * - ``rsgislib.imagemorphology.image_erode``
     - :func:`rsgispy.imagemorphology.image_erode`
     - ``gdalformat``, ``datatype`` keyword-only
   * - ``rsgislib.imagemorphology.image_opening``
     - :func:`rsgispy.imagemorphology.image_opening`
     - ``gdalformat``, ``datatype`` keyword-only
   * - ``rsgislib.imagemorphology.image_closing``
     - :func:`rsgispy.imagemorphology.image_closing`
     - ``gdalformat``, ``datatype`` keyword-only
   * - ``rsgislib.classification.clustersklearn.img_pixel_cluster``
     - :func:`rsgispy.classification.clustersklearn.img_pixel_cluster`
     - ``gdalformat``, ``datatype`` keyword-only
   * - ``rsgislib.classification.clustersklearn.img_pixel_sample_cluster``
     - :func:`rsgispy.classification.clustersklearn.img_pixel_sample_cluster`
     - ``gdalformat``, ``datatype`` keyword-only
   * - ``rsgislib.imageregistration.find_image_offset``
     - :func:`rsgispy.imageregistration.shift.find_image_offset`
     -
   * - ``rsgislib.tools.filetools.get_file_basename``
     - :func:`rsgispy.tools.files.get_file_basename`
     -
   * - ``rsgislib.tools.filetools.find_file``
     - :func:`rsgispy.tools.files.find_file`
     -
   * - ``rsgislib.tools.filetools.delete_file_with_basename``
     - :func:`rsgispy.tools.files.delete_file_with_basename`
     -
   * - ``rsgislib.tools.utils.read_json_to_dict``
     - :func:`rsgispy.tools.jsonio.read_json_to_dict`
     -
   * - ``rsgislib.tools.utils.write_dict_to_json``
     - :func:`rsgispy.tools.jsonio.write_dict_to_json`
     -
   * - ``rsgislib.tools.utils.uid_generator``
     - :func:`rsgispy.tools.strings.uid_generator`
     -

.. end-generated-tables

Common problems
---------------

* **ModuleNotFoundError: rsgispy.vectorutils** — the RSGISLib vector modules
  are now :mod:`rsgispy.vector` (``io``, ``create``, ``geoms``, ``attrs``,
  ``stats``, ``selection``); see the module table.
* **AttributeError: module 'rsgispy.imageutils' has no attribute ...** — the
  function is in a submodule; import e.g. ``rsgispy.imageutils.metadata``.
* **TypeError: ... positional arguments but ... were given** — an output
  option (``gdalformat``, ``datatype``, ...) was passed by position; name it.
* **Output written as GeoTIFF instead of KEA** — the default ``gdalformat``
  is ``"GTIFF"``; pass ``gdalformat="KEA"`` if your GDAL has the KEA driver.
* **Different numbers from RSGISLib** — check no data handling first:
  many rsgispy functions read the no data value from the image header,
  and statistics exclude it. Then check pixel selection rules (zonal
  statistics use pixels whose centre is inside the polygon) and
  interpolation (``rsgispy.INTERP_*``).
* **Unrecognised band maths expression** — pymu-eval follows the muParser
  syntax; check operator precedence and use parentheses around conditions,
  e.g. ``(b1 > 0) && (b2 > 0) ? 1 : 0``.

Related guides
--------------

* :doc:`analysis_ready_data`
* :doc:`pixel_classification_workflow`
* :doc:`large_rasters`
