rsgispy Base Module
===================

The top-level :mod:`rsgispy` module defines package constants (data types,
interpolation methods, image stretches, pan-sharpening resolution modes, vector
geometry types, rounding modes, summary statistic types, and correlation methods),
the base :class:`~rsgispy.RSGISPyException` class, and helper functions to convert
between RSGIS constants, ``rasterio``/GDAL strings, and ``numpy`` dtypes.

Constants
---------

Data Type Constants
~~~~~~~~~~~~~~~~~~~

* ``rsgispy.TYPE_UNDEFINED = 0``
* ``rsgispy.TYPE_8INT = 1``
* ``rsgispy.TYPE_16INT = 2``
* ``rsgispy.TYPE_32INT = 3``
* ``rsgispy.TYPE_64INT = 4``
* ``rsgispy.TYPE_8UINT = 5``
* ``rsgispy.TYPE_16UINT = 6``
* ``rsgispy.TYPE_32UINT = 7``
* ``rsgispy.TYPE_64UINT = 8``
* ``rsgispy.TYPE_32FLOAT = 9``
* ``rsgispy.TYPE_64FLOAT = 10``

Interpolation Constants
~~~~~~~~~~~~~~~~~~~~~~~

* ``rsgispy.INTERP_NEAREST_NEIGHBOUR = 0``
* ``rsgispy.INTERP_BILINEAR = 1``
* ``rsgispy.INTERP_CUBIC = 2``
* ``rsgispy.INTERP_CUBICSPLINE = 3``
* ``rsgispy.INTERP_LANCZOS = 4``
* ``rsgispy.INTERP_AVERAGE = 5``
* ``rsgispy.INTERP_MODE = 6``
* ``rsgispy.INTERP_MIN = 7``
* ``rsgispy.INTERP_MAX = 8``
* ``rsgispy.INTERP_MEDIAN = 9``
* ``rsgispy.INTERP_Q1 = 10``
* ``rsgispy.INTERP_Q3 = 11``
* ``rsgispy.INTERP_SUM = 12``

Image Stretch Constants
~~~~~~~~~~~~~~~~~~~~~~~

* ``rsgispy.IMG_STRETCH_NONE = 0``
* ``rsgispy.IMG_STRETCH_USER = 1``
* ``rsgispy.IMG_STRETCH_LINEAR = 2``
* ``rsgispy.IMG_STRETCH_STDEV = 3``
* ``rsgispy.IMG_STRETCH_CUMULATIVE = 4``

Pan-Sharpening Resolution Constants
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``rsgispy.SHARP_RES_IGNORE = 0``
* ``rsgispy.SHARP_RES_LOW = 1``
* ``rsgispy.SHARP_RES_HIGH = 2``

Vector Geometry Type Constants
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``rsgispy.GEOM_PT = 1``
* ``rsgispy.GEOM_LINE = 2``
* ``rsgispy.GEOM_POLY = 3``
* ``rsgispy.GEOM_MPT = 4``
* ``rsgispy.GEOM_MLINE = 5``
* ``rsgispy.GEOM_MPOLY = 6``

Rounding Mode Constants
~~~~~~~~~~~~~~~~~~~~~~~

* ``rsgispy.ROUND_NEAREST = 1``
* ``rsgispy.ROUND_UP = 2``
* ``rsgispy.ROUND_DOWN = 3``

Summary Statistic Constants
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``rsgispy.SUMTYPE_UNDEFINED = 0``
* ``rsgispy.SUMTYPE_MODE = 1``
* ``rsgispy.SUMTYPE_MEAN = 2``
* ``rsgispy.SUMTYPE_MEDIAN = 3``
* ``rsgispy.SUMTYPE_MIN = 4``
* ``rsgispy.SUMTYPE_MAX = 5``
* ``rsgispy.SUMTYPE_STDDEV = 6``
* ``rsgispy.SUMTYPE_COUNT = 7``
* ``rsgispy.SUMTYPE_RANGE = 8``
* ``rsgispy.SUMTYPE_SUM = 9``

Correlation Statistics Constants
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``rsgispy.STATS_CORR_UNDEFINED = 0``
* ``rsgispy.STATS_CORR_PEARSONS = 1``
* ``rsgispy.STATS_CORR_SPEARMAN = 2``
* ``rsgispy.STATS_CORR_KENDALL_TAU = 3``
* ``rsgispy.STATS_CORR_POINT_BISERIAL = 4``

Version
~~~~~~~

* ``rsgispy.__version__`` — Package version string (e.g. ``"0.0.1"``).

Example
-------

Converting ``rsgispy`` data type and interpolation constants:

.. code-block:: python

   import rsgispy

   rio_dtype = rsgispy.get_rasterio_datatype(rsgispy.TYPE_32FLOAT)
   np_dtype = rsgispy.get_numpy_datatype(rsgispy.TYPE_32FLOAT)
   rsgis_type = rsgispy.get_rsgis_datatype("int16")
   interp = rsgispy.get_rasterio_interp_type(rsgispy.INTERP_BILINEAR)

API Reference
-------------

.. automodule:: rsgispy
   :members:
   :undoc-members:
   :show-inheritance:
