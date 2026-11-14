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
``matplotlib``, ``matplotlib-scalebar``, ``numpy``, ``pandas``, ``pymu-eval``,
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
     - ``stats``, ``vector``, ``mapping``, ``dataaccess``, ``imaging``,
       ``unmixing``, ``jupyter``
     - All optional runtime dependencies.
   * - ``test``
     - ``flake8``, ``pylint``, ``pytest``, ``pytest-cov``, ``stats``
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
       gdalformat="GTIFF",
       datatype=rsgispy.TYPE_32FLOAT,
       band_defs=band_defns,
   )

Package Overview
----------------

* :doc:`rsgispy` — Core data type constants, exception class, and interpolation mappings.
* :doc:`rsgispy_changedetect` — Multi-temporal change detection (CVA, ISFA, IR-MAD) and class outlier detection.
* :doc:`rsgispy_dataaccess` — Clients for Copernicus OData, NASA CMR, OSM, Planet, and USGS M2M.
* :doc:`rsgispy_droneutils` — Drone photo organisation and metadata utilities.
* :doc:`rsgispy_elevation` — Terrain derivatives, illumination, DEM filtering, fusion, and roughness.
* :doc:`rsgispy_imagecalc` — Band maths, spectral indices, least-cost path, statistics, percentiles, comparison, encoding, and distance.
* :doc:`rsgispy_imagefilter` — Spatial moving-window statistical, edge, Gaussian, Kuwahara, Lee SAR, and kernel filters.
* :doc:`rsgispy_imagemorphology` — Morphological raster filtering (dilation, erosion, opening, closing).
* :doc:`rsgispy_imageprocess` — Block-processing engine, spatial intersection, and dataset validation.
* :doc:`rsgispy_imagespectral` — Spectral classification (SAM, SCM), linear unmixing, endmembers, PPI, PCA, MNF, Tasseled Cap, histogram matching, whitening, and smoothing.
* :doc:`rsgispy_imageutils` — Raster creation, rasterisation, metadata, masking, mosaicking, tiling, VRT, and warping.
* :doc:`rsgispy_tools` — General geospatial, geometry, projection, plotting, mapping, and file utilities.
* :doc:`rsgispy_vectorattrs` — Vector attribute table, column, and spatial join utilities.
* :doc:`rsgispy_vectorgeoms` — Vector geometry creation, grids, polygonisation, manipulation, overlay, clipping, transects, and spatial predicates.
* :doc:`rsgispy_vectorprocess` — Shared low-level vector I/O, CRS, bounding-box, and attribute helpers.
* :doc:`rsgispy_vectorstats` — Spatial metrics, area/length/distance calculations, attribute counts, and empirical semi-variograms for vector datasets.
* :doc:`rsgispy_vectorutils` — Vector metadata, translation, reprojection, merging, splitting, and selection utilities.
* :doc:`rsgispy_zonalstats` — Polygon and point zonal statistics extraction.

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   rsgispy
   rsgispy_changedetect
   rsgispy_dataaccess
   rsgispy_droneutils
   rsgispy_elevation
   rsgispy_imagecalc
   rsgispy_imagefilter
   rsgispy_imagemorphology
   rsgispy_imageprocess
   rsgispy_imagespectral
   rsgispy_imageutils
   rsgispy_tools
   rsgispy_vectorattrs
   rsgispy_vectorgeoms
   rsgispy_vectorprocess
   rsgispy_vectorstats
   rsgispy_vectorutils
   rsgispy_zonalstats

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
