rsgispy Documentation
=====================

**rsgispy** is a pure Python replacement for `RSGISLib <http://www.rsgislib.org>`_
designed to be easy to install and maintain while preserving a familiar API for
remote sensing and GIS raster processing using ``rasterio``, ``numpy``, ``pyproj``,
``geopandas``, and ``pymu-eval``.

Installation
------------

Minimal Installation
~~~~~~~~~~~~~~~~~~~~

Install ``rsgispy`` with only its core dependencies (``geopandas``,
``matplotlib``, ``numpy``, ``pandas``, ``pymu-eval``, ``pyogrio``,
``pyproj``, ``rasterio``, ``shapely``, and ``tqdm``) using ``pip``:

.. code-block:: bash

   pip install .

This provides core raster and vector I/O, image maths, spectral indices,
morphology, elevation derivatives, and basic vector operations.

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~

Additional functionality uses optional third-party packages defined under
``[project.optional-dependencies]`` in ``pyproject.toml``:

.. list-table::
   :header-rows: 1
   :widths: 14 36 50

   * - Extra
     - Packages Installed
     - Features / Modules Enabled
   * - ``classification``
     - ``optuna``, ``pyarrow``, ``scikit-learn``, ``scipy``
     - Supervised image classification, hyperparameter tuning, accuracy
       assessment, and post-classification majority/sieve filtering.
   * - ``regression``
     - ``optuna``, ``pyarrow``, ``scikit-learn``, ``scipy``
     - Continuous univariate and multivariate image regression with
       ``scikit-learn`` (plus optional GBDT, SHAP, and SALib extras).
   * - ``gbdt``
     - ``xgboost``, ``lightgbm``, ``catboost``, ``optuna``, ``scikit-learn``
     - Gradient-boosted decision tree classifiers and regressors (also
       installable individually as ``xgboost``, ``lightgbm``, or ``catboost``).
   * - ``modelling``
     - ``SALib``, ``scikit-learn``, ``scipy``, ``shap``
     - Species distribution modelling, SHAP model explanations, and SALib
       Sobol global sensitivity analysis.
   * - ``stats``
     - ``scikit-gstat``, ``scikit-learn``, ``scipy``
     - Empirical semi-variograms, accuracy/feature selection, thresholding,
       KNN gap filling, distance transforms, and spatial interpolation.
   * - ``vector``
     - ``alphashape``, ``h3``, ``openpyxl``, ``pyarrow``, ``rtree``,
       ``xlsxwriter``
     - Alpha shapes, H3 hexagon grids, Excel table I/O, GeoParquet I/O, and
       R-tree spatial indexing.
   * - ``mapping``
     - ``contextily``, ``mpl-scatter-density``, ``owslib``
     - Tile basemaps, scatter density plots, and WMTS layer retrieval.
   * - ``dataaccess``
     - ``beautifulsoup4``, ``osmnx``, ``planet``, ``pycurl``, ``pysondb``,
       ``requests``
     - Copernicus OData, NASA CMR, OpenStreetMap, Planet API, HTTP/FTP
       tracking databases, and web file discovery.
   * - ``imaging``
     - ``h5py``, ``imageio``, ``pillow``, ``pyexiftool``, ``pymupdf``,
       ``qrcode``, ``scikit-image``
     - KEA/HDF5 validation, drone/EXIF metadata, animated GIFs, PDF page/image
       utilities, QR codes, and least-cost path graph routing.
   * - ``unmixing``
     - ``cvxopt``, ``pyMCR``, ``pysptools``, ``scikit-learn``, ``scipy``
     - Full optional spectral unmixing backends (``pysptools`` / ``pyMCR``)
       and SciPy NNLS/FCLS unmixing (recommended NumPy/SciPy unmixing only
       requires ``scipy``, also included in ``stats``), ISFA/IR-MAD change
       detection, and K-means change thresholding.
   * - ``jupyter``
     - ``ipywidgets``
     - Interactive Jupyter notebook progress bars.
   * - ``all``
     - ``classification``, ``regression``, ``gbdt``, ``stats``, ``vector``,
       ``mapping``, ``dataaccess``, ``imaging``, ``unmixing``, ``jupyter``,
       ``modelling``
     - All optional runtime dependencies.
   * - ``test``
     - ``flake8``, ``pylint``, ``pytest``, ``pytest-cov``,
       ``classification``, ``regression``, ``stats``
     - Running the test suite and linters.
   * - ``docs``
     - ``sphinx``, ``sphinx-rtd-theme``
     - Building HTML documentation with Sphinx.
   * - ``dev``
     - ``all``, ``test``, ``docs``
     - Full editable development environment with all optional, test, and
       documentation dependencies.

Install one or more extras by passing them in square brackets:

.. code-block:: bash

   # Install specific optional feature groups
   pip install ".[vector]"
   pip install ".[stats,vector,mapping]"

   # Install with all optional runtime dependencies
   pip install ".[all]"

   # Editable installs for testing, documentation, or full development
   pip install -e ".[test]"
   pip install -e ".[all,test]"
   pip install -e ".[dev]"

.. note::

   On Linux, installing the ``dataaccess`` or ``all`` extras may require
   ``libcurl4-openssl-dev`` and ``libssl-dev`` if ``pycurl`` is built from
   source. Functions using ``pyexiftool`` (in ``imaging`` / ``all``) also
   require the external ``exiftool`` command-line tool on ``PATH``.

Quickstart
----------

Evaluating a mathematical expression across image bands with
:func:`rsgispy.imagecalc.maths.band_maths`:

.. code-block:: python

   import rsgispy
   import rsgispy.imagecalc.maths

   band_defns = [
       rsgispy.imagecalc.maths.BandDefn(band_name="b1", input_img="img.kea", img_band=1),
       rsgispy.imagecalc.maths.BandDefn(band_name="b2", input_img="img.kea", img_band=2),
   ]
   rsgispy.imagecalc.maths.band_maths(
       output_img="out.tif",
       exp="b1*b2",
       band_defs=band_defns,
       gdalformat="GTIFF",
       datatype=rsgispy.TYPE_32FLOAT,
   )

Output image creation options
-----------------------------

By default, GeoTIFF outputs (``gdalformat="GTIFF"``) are tiled (GDAL's default
256 x 256 tiles) and DEFLATE-compressed, with the creation options
``TILED=YES``, ``COMPRESS=DEFLATE``, ``INTERLEAVE=BAND`` and
``BIGTIFF=IF_SAFER``. Other formats (for example ``"KEA"`` and ``"COG"``) use
their GDAL driver defaults. rsgispy does not set any environment variables when
it is imported; the defaults are listed in
``rsgispy.imageprocess.DEFAULT_IMG_CREATION_OPTS``.

The options for a format are taken from, in order:

1. the ``RSGISPY_IMG_CRT_OPTS_<FORMAT>`` environment variable (for example
   ``RSGISPY_IMG_CRT_OPTS_GTIFF="TILED=YES:COMPRESS=ZSTD"``), if set;
2. otherwise ``RSGISLIB_IMG_CRT_OPTS_<FORMAT>``, if set;
3. otherwise the built-in default.

A variable that is set replaces the default rather than adding to it, and an
empty variable means "no creation options". The ``creation_opts`` argument of
individual functions is merged on top. To change the options from Python, use
:mod:`rsgispy.imageutils.envopts`:

.. code-block:: python

   import rsgispy.imageutils.envopts

   # ZSTD instead of DEFLATE
   rsgispy.imageutils.envopts.set_env_vars_zstd_gtiff_outs()
   # Plain striped, uncompressed GeoTIFFs (the GDAL default)
   rsgispy.imageutils.envopts.set_img_creation_opts("GTIFF", {})
   # Back to the built-in defaults
   rsgispy.imageutils.envopts.reset_img_creation_opts()

Guides
------

The guides walk through complete analyses, from input data to final
products, and show how functions from several subpackages fit together.
Each guide's code is taken from a runnable example script in
``docs/examples/`` that is tested on small synthetic data.

* :doc:`guides/migrating_from_rsgislib` — Moving existing RSGISLib scripts to rsgispy.
* :doc:`guides/analysis_ready_data` — Downloading, stacking and cloud masking Sentinel-2, and building cloud-free composites.
* :doc:`guides/image_coregistration` — Measuring and correcting misregistration with shifts, tie points and GCPs.
* :doc:`guides/pixel_classification_workflow` — Pixel-based land cover classification, from feature selection and model interpretation to accuracy assessment.
* :doc:`guides/pixel_regression_workflow` — Mapping a continuous variable from field plots, with prediction intervals.
* :doc:`guides/segment_ml_workflow` — Object-based (segment) classification and regression.
* :doc:`guides/change_detection` — Image differencing, change vector analysis, IR-MAD and class-based change.
* :doc:`guides/species_distribution_modelling` — Presence/pseudo-absence models, spatial cross-validation and habitat suitability maps.
* :doc:`guides/spectral_unmixing` — Wetland mapping with automatic endmember selection and linear unmixing, following the TropWet method.
* :doc:`guides/sar_processing` — Speckle filtering, SAR indices and texture for SAR backscatter.
* :doc:`guides/hydrology_workflow` — DEM conditioning, flow routing, streams, catchments, HAND and terrain indices.
* :doc:`guides/multi_criteria_analysis` — A multi-criteria analysis with ``band_maths``: mangrove restoration potential.
* :doc:`guides/zonal_stats_and_vectors` — Summarising rasters by polygons, points, label rasters and hexagon grids.
* :doc:`guides/making_maps` — Publication-quality maps with legends, scale bars and inset maps.
* :doc:`guides/large_rasters` — Block processing, tile graphs, VRTs, tiling and parallel processing for large data.

Package Overview
----------------

* :doc:`rsgispy` — Core data type constants, exception class, and interpolation mappings.
* :doc:`rsgispy_changedetect` — Multi-temporal change detection (CVA, ISFA, IR-MAD) and class outlier detection.
* :doc:`rsgispy_classification` — Supervised classification (``scikit-learn``, XGBoost, LightGBM, CatBoost), clustering, feature selection, reference point sampling, accuracy statistics, and post-classification majority/sieve filtering.
* :doc:`rsgispy_cli` — Standardised command-line tools for raster/vector inspection, statistics, COG conversion, LUT creation, dataset validation, hashing, and CRS conversion.
* :doc:`rsgispy_dataaccess` — Clients for Copernicus OData, NASA CMR, OSM, Planet, and USGS M2M.
* :doc:`rsgispy_droneutils` — Drone photo organisation and metadata utilities.
* :doc:`rsgispy_elevation` — Terrain derivatives, illumination, DEM filtering, fusion, and roughness.
* :doc:`rsgispy_imagecalc` — Band maths, spectral indices, least-cost path, statistics, percentiles, comparison, encoding, and distance.
* :doc:`rsgispy_imagecalibration` — Radiometric calibration, TOA reflectance, brightness temperature, DOS, solar angles, and Landsat/Sentinel-2 stacking.
* :doc:`rsgispy_imagefilter` — Spatial moving-window statistical, edge, Gaussian, Kuwahara, Lee SAR, and kernel filters.
* :doc:`rsgispy_imagemorphology` — Morphological raster filtering (dilation, erosion, opening, closing).
* :doc:`rsgispy_imageprocess` — Block-processing engine, spatial intersection, and dataset validation.
* :doc:`rsgispy_imageregistration` — Image registration, tie-point co-registration, outlier filtering, whole-image shift estimation, and GCP warping.
* :doc:`rsgispy_imagespectral` — Spectral classification (SAM, SCM), linear unmixing, endmembers, PPI, PCA, MNF, Tasseled Cap, histogram matching, whitening, and smoothing.
* :doc:`rsgispy_imageutils` — Raster creation, rasterisation, metadata, masking, mosaicking, tiling, VRT, and warping.
* :doc:`rsgispy_mapping` — Cartographic map composition, scale bars, north arrows, raster/vector/basemap layers, and categorical legends.
* :doc:`rsgispy_ml` — Unified machine-learning core: :class:`~rsgispy.ml.stack.ImageFeatureStack`, :class:`~rsgispy.ml.scaling.FeatureScaler`, tiled raster inference, spatial cross-validation, and Optuna tuning.
* :doc:`rsgispy_modelling` — Environmental and species distribution modelling with ``scikit-learn`` and :mod:`rsgispy.imageprocess`.
* :doc:`rsgispy_regression` — Continuous univariate and multivariate image regression with ``scikit-learn``, XGBoost, LightGBM, and CatBoost, response curves, feature selection, and accuracy metrics.
* :doc:`rsgispy_segmentation` — Connected-component clumping (:func:`~rsgispy.segmentation.clump.clump`) and union of clumps (:func:`~rsgispy.segmentation.clump.union_of_clumps`).
* :doc:`rsgispy_tools` — General geospatial, geometry, projection, plotting, and file utilities.
* :doc:`rsgispy_vector` — Consolidated vector I/O, geometry creation, geometric operations, attribute tables, spatial statistics, and selection (accepting file paths or GeoDataFrames).
* :doc:`rsgispy_vectorprocess` — Shared low-level vector I/O, CRS, bounding-box, and attribute helpers.
* :doc:`rsgispy_zonalstats` — Polygon and point zonal statistics extraction.

.. toctree::
   :maxdepth: 2
   :caption: Guides

   guides/migrating_from_rsgislib
   guides/analysis_ready_data
   guides/image_coregistration
   guides/pixel_classification_workflow
   guides/pixel_regression_workflow
   guides/segment_ml_workflow
   guides/change_detection
   guides/species_distribution_modelling
   guides/spectral_unmixing
   guides/sar_processing
   guides/hydrology_workflow
   guides/multi_criteria_analysis
   guides/zonal_stats_and_vectors
   guides/making_maps
   guides/large_rasters

.. toctree::
   :maxdepth: 2
   :caption: AI Tooling

   gemini_skill

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   rsgispy
   rsgispy_changedetect
   rsgispy_classification
   rsgispy_cli
   rsgispy_dataaccess
   rsgispy_droneutils
   rsgispy_elevation
   rsgispy_imagecalc
   rsgispy_imagecalibration
   rsgispy_imagefilter
   rsgispy_imagemorphology
   rsgispy_imageprocess
   rsgispy_imageregistration
   rsgispy_imagespectral
   rsgispy_imageutils
   rsgispy_mapping
   rsgispy_ml
   rsgispy_modelling
   rsgispy_regression
   rsgispy_segmentation
   rsgispy_tools
   rsgispy_vector
   rsgispy_vectorprocess
   rsgispy_zonalstats

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
