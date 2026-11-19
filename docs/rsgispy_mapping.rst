rsgispy.mapping
===============

The :mod:`rsgispy.mapping` subpackage provides cartographic map composition
and visualisation utilities using ``matplotlib``, ``rasterio``, ``geopandas``,
and ``pyproj``, organised into thematic submodules:

* :doc:`mapping/layout` — Figure sizing, axis extent, tick spacing, and inset overview maps.
* :doc:`mapping/elements` — Scale bars, north arrows, bounding boxes, and axis labels.
* :doc:`mapping/raster` — Continuous, colour-mapped, and thematic raster map layers.
* :doc:`mapping/vector` — Vector, choropleth, and point-density map layers.
* :doc:`mapping/basemaps` — Contextily tile basemaps and WMTS layer mapping.
* :doc:`mapping/legend` — Categorical and classification map legends.

Example
-------

Creating a thematic land-cover map with a companion legend, north arrow, and
custom scale bar:

.. code-block:: python

   import matplotlib.pyplot as plt
   import rsgispy.mapping.elements as elements
   import rsgispy.mapping.raster as map_raster

   fig, ax = plt.subplots(figsize=(8, 6))
   map_raster.create_thematic_raster_map(
       ax=ax,
       input_img="landcover.kea",
       title_str="Land Cover Classification",
       scale_bar_style=elements.SCALEBAR_STYLE_ALTERNATING,
       show_north_arrow=True,
       north_arrow_style=elements.NORTH_ARROW_STYLE_COMPASS,
       show_legend=True,
       classes_info={1: "Forest", 2: "Grassland", 3: "Water"},
   )

Submodules
----------

.. toctree::
   :maxdepth: 1

   mapping/layout
   mapping/elements
   mapping/raster
   mapping/vector
   mapping/basemaps
   mapping/legend
