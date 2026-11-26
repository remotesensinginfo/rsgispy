rsgispy.elevation
=================

The :mod:`rsgispy.elevation` subpackage provides terrain and digital elevation
model (DEM) analysis utilities using ``rasterio`` and ``numpy``.

.. seealso::

   Guide: :doc:`guides/hydrology_workflow`

Terrain Derivatives & Illumination
----------------------------------

* :doc:`elevation/terrain` — Slope, aspect, and aspect categorisation.
* :doc:`elevation/illumination` — Hillshade, shadow masks, local incidence and exitance angles.
* :doc:`elevation/roughness` — Surface roughness profile and Smith roughness metrics.

.. toctree::
   :maxdepth: 1

   elevation/terrain
   elevation/illumination
   elevation/roughness

Chaining
--------

The terrain derivatives can also be used as nodes of an in-memory
:class:`~rsgispy.imageprocess.TileGraph`, chained with filters, band maths
and other nodes in one pass without intermediate files (e.g. smoothing →
slope → classes). Each has a node builder next to its file-based function,
named after it with ``_node``, which gives the same result (the file-based
functions are built on them):

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - File-based function
     - Node builder
   * - :func:`~rsgispy.elevation.terrain.slope`
     - :func:`~rsgispy.elevation.terrain.slope_node`
   * - :func:`~rsgispy.elevation.terrain.slope_pxl_res_img`
     - :func:`~rsgispy.elevation.terrain.slope_pxl_res_img_node`
   * - :func:`~rsgispy.elevation.terrain.aspect`
     - :func:`~rsgispy.elevation.terrain.aspect_node`
   * - :func:`~rsgispy.elevation.terrain.aspect_pxl_res_img`
     - :func:`~rsgispy.elevation.terrain.aspect_pxl_res_img_node`
   * - :func:`~rsgispy.elevation.terrain.categorise_aspect`
     - :func:`~rsgispy.elevation.terrain.categorise_aspect_node`
   * - :func:`~rsgispy.elevation.illumination.hillshade`
     - :func:`~rsgispy.elevation.illumination.hillshade_node`
   * - :func:`~rsgispy.elevation.illumination.hillshade_pxl_res_img`
     - :func:`~rsgispy.elevation.illumination.hillshade_pxl_res_img_node`
   * - :func:`~rsgispy.elevation.illumination.local_incidence_angle`
     - :func:`~rsgispy.elevation.illumination.local_incidence_angle_node`
   * - :func:`~rsgispy.elevation.illumination.local_exitance_angle`
     - :func:`~rsgispy.elevation.illumination.local_exitance_angle_node`

The builders take a single-band DEM node (e.g. ``RasterSource("dem.tif",
bands=1)``); the ``*_pxl_res_img_node`` builders take the pixel-resolution
node (bands 1 and 2: East-West and North-South resolution, on the DEM's
grid) as their second argument. The Horn 3x3 nodes have a halo radius of 1
and repeat the edge pixels at the image borders, and take the pixel size
from the DEM's geotransform. Unlike the filter builders, their
``no_data_val`` defaults to the no-data value of the input node (the DEM
file's, or the one a previous node sets), as the file-based functions read
it from the DEM header. The slope, aspect and angle nodes output
``float64`` (write them as ``rsgispy.TYPE_32FLOAT`` to match the file-based
functions); the hillshade and aspect-class nodes output ``uint8``.

.. code-block:: python

   import rsgispy
   from rsgispy.imageprocess import BandMathOp, DiskSink, RasterSource, TileGraph
   from rsgispy.imagefilter.stats import mean_filter_node
   from rsgispy.elevation.terrain import slope_node

   dem = RasterSource("dem.tif", bands=1)
   smooth = mean_filter_node(dem, 5, no_data_val=-9999)
   slp = slope_node(smooth, "degrees")
   classes = BandMathOp({"s": slp}, "(s >= 5) + (s >= 15) + (s >= 30)")
   TileGraph(
       DiskSink(classes, "slope_classes.tif", datatype=rsgispy.TYPE_8UINT)
   ).run()

:func:`~rsgispy.elevation.illumination.shadow_mask` is not chainable: it
traces solar rays across the whole DEM (a pixel can be shaded by terrain
any distance away), so it cannot be computed tile by tile with a fixed
halo. The hydrological flow routing and accumulation functions are
whole-image operations for the same reason. See :doc:`rsgispy_imageprocess`
for the tile graph and the other chainable operations.

DEM Filtering & Fusion
----------------------

* :doc:`elevation/filtering` — Aspect median filtering, depression filling, and detrending.
* :doc:`elevation/fusion` — FFT-based DEM fusion.

.. toctree::
   :maxdepth: 1

   elevation/filtering
   elevation/fusion

Hydrological Analysis
---------------------

* :doc:`elevation/hydrology` — Sink identification, depression filling and breaching,
  stream burning, flat resolution, D8/D-Infinity/MFD flow routing, flow accumulation,
  stream ordering and vectorisation, watershed delineation, HAND flood inundation,
  TWI, SPI, and RUSLE LS factor.

.. toctree::
   :maxdepth: 1

   elevation/hydrology

