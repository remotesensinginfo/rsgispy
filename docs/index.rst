rsgispy Documentation
=====================

**rsgispy** is a pure Python replacement for `RSGISLib <http://www.rsgislib.org>`_ designed to be easy to install and maintain while preserving a familiar API for remote sensing and GIS raster processing.

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

Evaluating a mathematical expression across image bands with :func:`rsgispy.imagecalc.band_maths`:

.. code-block:: python

   import rsgispy
   import rsgispy.imagecalc

   band_defns = [
       rsgispy.imagecalc.BandDefn(band_name="b1", input_img="img.kea", img_band=1),
       rsgispy.imagecalc.BandDefn(band_name="b2", input_img="img.kea", img_band=2),
   ]
   rsgispy.imagecalc.band_maths(
       output_img="out.tif",
       exp="b1*b2",
       gdalformat="GTIFF",
       datatype=rsgispy.TYPE_32FLOAT,
       band_defs=band_defns,
   )

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   rsgispy
   rsgispy_imagecalc
   rsgispy_imageprocess

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
