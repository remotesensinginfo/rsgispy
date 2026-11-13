rsgispy Documentation
=====================

**rsgispy** is a pure Python replacement for `RSGISLib <http://www.rsgislib.org>`_
designed to be easy to install and maintain while preserving a familiar API for
remote sensing and GIS raster processing using ``rasterio``, ``numpy``, ``pyproj``,
``geopandas``, and ``pymu-eval``.

Installation
------------

Install ``rsgispy`` using ``pip``:

.. code-block:: bash

   pip install .

To install with testing and documentation dependencies:

.. code-block:: bash

   pip install -e ".[test,docs]"

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
* :doc:`rsgispy_dataaccess` — Clients for Copernicus OData, NASA CMR, OSM, Planet, and USGS M2M.
* :doc:`rsgispy_droneutils` — Drone photo organisation and metadata utilities.
* :doc:`rsgispy_elevation` — Terrain derivatives, illumination, DEM filtering, fusion, and roughness.
* :doc:`rsgispy_imagecalc` — Band maths, statistics, percentiles, comparison, encoding, and distance.
* :doc:`rsgispy_imagemorphology` — Morphological raster filtering (dilation, erosion, opening, closing).
* :doc:`rsgispy_imageprocess` — Block-processing engine, spatial intersection, and dataset validation.
* :doc:`rsgispy_imageutils` — Raster creation, metadata, masking, mosaicking, tiling, VRT, and warping.
* :doc:`rsgispy_tools` — General geospatial, geometry, projection, plotting, mapping, and file utilities.
* :doc:`rsgispy_zonalstats` — Polygon and point zonal statistics extraction.

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   rsgispy
   rsgispy_dataaccess
   rsgispy_droneutils
   rsgispy_elevation
   rsgispy_imagecalc
   rsgispy_imagemorphology
   rsgispy_imageprocess
   rsgispy_imageutils
   rsgispy_tools
   rsgispy_zonalstats

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
