Making maps
===========

Most analyses end with a map: a true colour image of the study area, a land
cover classification, a continuous score, or statistics summarised by region.
:doc:`rsgispy.mapping <../rsgispy_mapping>` draws rasters and vectors onto ordinary Matplotlib
axes, and adds the cartographic elements a figure for a report or paper needs
(scale bars, north arrows, legends, inset overview maps and basemaps).

This guide shows how to make each type of map, how to choose stretches,
colour maps and legends, and how to export a figure at publication quality.
It uses the outputs of the other guides: the land cover map from
:doc:`pixel_classification_workflow`, a continuous score such as the
restoration potential from :doc:`multi_criteria_analysis`, and vector layers
such as the streams and catchments from :doc:`hydrology_workflow`.

Why use ``rsgispy.mapping`` rather than plain Matplotlib?

* Rasters are read at the right place in map coordinates (only the part
  inside the bounding box you ask for), stretched for display, and drawn with
  the correct extent, so raster and vector layers line up.
* Scale bars are computed from the coordinate reference system (CRS), so they
  are correct in metres even for maps in degrees.
* Thematic rasters are drawn with the colour table stored in the file, and the
  legend can be built from the same colour table and your class names.
* Every function draws onto an axis you created, so you keep full control of
  the figure layout and can mix rsgispy layers with any other Matplotlib
  plotting.

.. seealso::

   The rsgispy tutorials cover this in notebooks you can run:
   :tutorial:`an introduction to Matplotlib <02_making_maps/01_intro_matplotlib.ipynb>`,
   :tutorial:`maps from EO data <02_making_maps/02_maps_from_eo_data.ipynb>`,
   :tutorial:`maps from vector data <02_making_maps/03_maps_from_vector_data.ipynb>` and
   :tutorial:`combining data sources <02_making_maps/04_combining_data_sources.ipynb>`
   (section 2), and
   :tutorial:`supervised pixel classification <04_image_classification/03_supervised_pixel_classification.ipynb>`,
   which makes the land cover map used here. See :doc:`../tutorials` for how to
   get them and their data.

.. contents:: Contents
   :local:
   :depth: 1

Installation
------------

The mapping functions need ``matplotlib`` (and ``geopandas`` for vector
layers). Basemaps from web tile services need ``contextily``, and point
density maps need ``mpl-scatter-density``; both are in the ``mapping`` extra:

.. code-block:: bash

   pip install "rsgispy[mapping]"

The basemap and WMTS steps need an internet connection. All other steps work
offline.

Overview
--------

.. code-block:: text

   1. Layout: extent, figure size, ticks
            │
            ├─► 2. RGB image (s2_sref.tif) ──────────► stretch comparison, RGB map
            ├─► 3. continuous raster (score.tif) ────► map + colour bar
            │      + DEM (dem.tif) ──────────────────► hillshaded maps
            ├─► 4. thematic raster (land_cover.tif) ─► map + categorical legend
            ├─► 5. vectors (regions, rivers) ────────► overlay map, choropleth
            │
            ▼
   6. Composed map: scale bar, north arrow, zoom box
            │
            ▼
   7. Inset overview map (overview.gpkg)
            │
            ▼
   8. Export: PNG at 300 dpi, PDF; zoomed panel
            │
            ▼
   9. Basemaps: contextily tiles, WMTS  [optional, needs the internet]

The code in this guide comes from the example script
``docs/examples/making_maps.py`` (included below step by step), which the
rsgispy test suite runs on synthetic data. The inputs are:

* ``s2_sref.tif`` — a Sentinel-2 surface reflectance stack (reflectance x
  10000, no data 0), for example from
  :func:`~rsgispy.imagecalibration.sensorlvl2data.create_stacked_sen2_esa_l2a_img`;
* ``land_cover.tif`` — a classification with a colour table, as written by
  :func:`~rsgispy.classification.classsklearn.apply_sklearn_classifier`;
* ``score.tif`` — any continuous single-band raster, for example a 0–1 score;
* ``dem.tif`` — an optional DEM (elevation in metres) for the hillshaded maps;
* ``regions.gpkg`` and ``rivers.gpkg`` — polygon and line layers in the same
  CRS as the rasters;
* ``overview.gpkg`` — a small-scale layer (e.g. country outlines, in any CRS)
  for the inset map.

The figures and printed outputs on this page come from the data of the
rsgispy tutorials, for a 24 x 18 km area around Aberystwyth (Wales) in the
British National Grid (EPSG:27700): the coast, the town, the Rheidol and
Ystwyth valleys and the conifer plantations to the east. The script
``docs/examples/making_maps_tutorial.py`` prepares the inputs and runs the
example:

* the RGB maps use the 20 m Sentinel-2 image of 29 June 2018
  (:tutorial-data:`sen2_20180629_t30uvd_orb037_osgb_stdsref_20m.tif`, ten
  bands from Blue to SWIR2, reflectance x 1000), cut to the area;
* the land cover is the random forest classification of the
  :tutorial:`supervised pixel classification notebook <04_image_classification/03_supervised_pixel_classification.ipynb>`:
  the same ten classes, colours, training polygons
  (:tutorial-data:`cls_data/aber_sen2_cls_training.gpkg`) and classifier
  parameters (100 trees, maximum depth 16), trained on 350 pixels per class
  from the whole image and applied to the area. The notebook reports a test
  accuracy of 0.99, optimistic because the test pixels come from the same
  polygons as the training pixels, and the map has not been checked against
  independent data, so treat it as an illustration;
* the continuous raster is the NDVI, (NIR − red) / (NIR + red), with water
  (from the land cover) set to no data;
* the DEM is :tutorial-data:`dem_10m_osgb.tif`, cut to the area, with the sea
  (0 m in this DEM) set to no data;
* the rivers are the channels draining more than 10 km², derived from the
  whole DEM at 20 m with :mod:`rsgispy.elevation.hydrology` (as in
  :doc:`hydrology_workflow`);
* the regions are the 108 squares of a 2 km grid, named by row (A–I, from
  the north) and column (1–12, from the west);
* the inset map uses the country outlines of
  :tutorial-data:`country.geojson`.

.. literalinclude:: ../examples/making_maps_tutorial.py
   :language: python
   :start-after: # [tutorial-data]
   :end-before: # [end-tutorial-data]
   :dedent: 4

The test suite runs the same example on a small synthetic scene
(``tests/maps_scene.py``), so the code on this page is always tested.

A short note on Matplotlib: a *figure* is the whole image and holds one or
more *axes* (map panels). Every rsgispy mapping function takes the axis to
draw on as its first argument, and you save the figure with
``fig.savefig()``.

Step 1 — Layout: extent, figure size and ticks
----------------------------------------------

Maps are drawn in the CRS of the data, so the extent is a bounding box
``(MinX, MaxX, MinY, MaxY)`` in map units (metres for a projected CRS).
:func:`~rsgispy.imageutils.metadata.get_img_bbox` returns the extent of an
image as a ``rsgispy.BBox``, a tuple that also records the EPSG code
(``bbox.epsg``).

To avoid white space around the map, the figure should have the same aspect
ratio as the area mapped. Fix the figure width (the width of the page or
column it will be printed in: about 6.7 inches for the full text width of an
A4 journal page, 3.3 inches for one column) and let
:func:`~rsgispy.mapping.layout.calc_y_fig_size` work out the height.

The tick spacing (in map units) sets where the coordinate labels go;
:func:`~rsgispy.mapping.layout.define_map_tick_spacing` applies it after a
layer has been drawn. Here it defaults to about a quarter of the map width,
rounded to one significant figure. For maps of a sub-area (a zoom panel), use
a smaller bounding box; every drawing function takes a ``bbox`` argument and
:func:`~rsgispy.mapping.layout.define_axis_extent` sets an axis to a given
box with an equal aspect ratio.

.. literalinclude:: ../examples/making_maps.py
   :language: python
   :start-after: # [step-1-layout]
   :end-before: # [end-step-1-layout]
   :dedent: 4

The printed line is a quick check that the extent is in metres (a width of a
few degrees means the data are in a geographic CRS; see `Common problems`_):

.. code-block:: text

   Map extent 24020 m wide; figure 6.7 x 5.03 in; ticks every 6000 m

Step 2 — RGB images and stretches
---------------------------------

:func:`~rsgispy.mapping.raster.create_raster_img_map` draws one band (in
grey) or three bands (as red, green and blue). Reflectance values must be
*stretched* to the 0–1 display range, and the choice of stretch changes the
look of the image a lot:

``rsgispy.IMG_STRETCH_LINEAR``
   The band minimum and maximum map to black and white
   (:func:`~rsgispy.tools.plotting.linear_stretch_np_arr`). A few very bright
   pixels (cloud, roofs) make the rest of the image dark.
``rsgispy.IMG_STRETCH_STDEV``
   Mean ± ``stch_n_stdevs`` standard deviations
   (:func:`~rsgispy.tools.plotting.stdev_stretch_np_arr`); 2 is a good start,
   use more for less contrast.
``rsgispy.IMG_STRETCH_CUMULATIVE``
   The 2nd and 98th percentiles
   (:func:`~rsgispy.tools.plotting.cumulative_stretch_np_arr`). Robust to
   outliers and usually the best default for a single image.
``rsgispy.IMG_STRETCH_USER``
   Fixed ``min``/``max`` values for each band
   (:func:`~rsgispy.tools.plotting.manual_stretch_np_arr`). Use this to compare
   images: the same reflectance has the same colour in every map, which is not
   true of the statistical stretches. For Sentinel-2 surface reflectance x
   10000, 0–1500 (visible bands) or 0–3000 (with NIR/SWIR) are typical; the
   tutorial image is reflectance x 1000, so the figures use 0–150
   (``user_stretch`` of the example).
``rsgispy.IMG_STRETCH_NONE``
   No stretch; for data already scaled to 0–1 (or 0–255 integers).

The statistics are computed from the pixels inside the bounding box only, so
a zoomed-in map is stretched for its own area. No data pixels (the image's no
data value, or ``no_data_val``) are excluded from the statistics and drawn in
``no_data_out_clr`` (white here); otherwise they are transparent.

For band combinations, use the band numbers of your stack: ``(3, 2, 1)`` is
true colour for a stack that starts Blue, Green, Red (as the tutorial image
does); NIR, Red, Green is the standard false colour composite in which
vegetation is red (``(4, 3, 2)`` for a Blue, Green, Red, NIR stack,
``(7, 3, 2)`` for the ten-band tutorial image). Panel labels
are added with :func:`~rsgispy.mapping.elements.add_axis_label`.

.. literalinclude:: ../examples/making_maps.py
   :language: python
   :start-after: # [step-2-rgb]
   :end-before: # [end-step-2-rgb]
   :dedent: 4

.. figure:: images/maps_rgb_stretches.png
   :width: 100%
   :alt: The Sentinel-2 image of Aberystwyth with four stretches: linear (almost black), standard deviation, cumulative and user.

   The same image with four stretches. The linear stretch is dominated by a
   few very bright pixels (maxima of 800–950, while 98 % of the pixels are
   below 90–160), so the image is almost black. The statistical stretches (b,
   c) stretch each band to its own range, which makes the grassland bright
   green; the user stretch, the same for the three bands, keeps the relative
   brightness of the bands (the grassland looks browner) and keeps the
   colours comparable between images. Contains modified Copernicus Sentinel
   data (2018).

Step 3 — Continuous single-band maps
------------------------------------

:func:`~rsgispy.mapping.raster.create_raster_cmap_img_map` draws one band
with a Matplotlib colour map. Pixels equal to the band's no data value are
transparent. ``show_colour_bar=True`` (with ``cbar_label``) adds a colour bar
beside the map. The function returns the image artist
(``matplotlib.image.AxesImage``), so you can also draw the colour bar
yourself, e.g. one bar for several panels with ``fig.colorbar(img, ax=axs)``.
Every map function returns the artist it drew (see
:ref:`plotting_conventions`).

The map functions draw in the rsgispy house style (one box style for the
scale bar, north arrow and legends, and readable tick labels) without
changing your Matplotlib settings; see the :doc:`plot_gallery`.

Choosing a colour map:

* **Sequential** data (a fraction, a score, elevation): use a perceptually
  uniform map such as ``viridis``, ``cividis`` (also safe for red–green colour
  blindness) or a single-hue map such as ``YlGn`` or ``Blues``.
* **Diverging** data with a meaningful centre (a change, an anomaly, a
  residual): ``RdBu`` or ``BrBG``, with ``norm_vmin = -norm_vmax`` so the
  centre is zero.
* Avoid ``jet`` and ``rainbow``: they create false boundaries.

Set ``norm_img_vals=True`` with fixed ``norm_vmin``/``norm_vmax`` (here 0 and
1) so the colours have the same meaning in every map and the colour bar shows
the full range. Without ``norm_vmin``/``norm_vmax`` the range is the data
minimum and maximum. The normalisation (and so ``use_log_norm``) is only used
when ``norm_img_vals=True``; with the default ``False`` the colours span the
image's own range, so set it to ``True`` whenever you add a colour bar.
``use_log_norm=True`` suits strongly
skewed data (e.g. flow accumulation), and ``vals_under_white=True`` draws
values below ``norm_vmin`` in white.

.. literalinclude:: ../examples/making_maps.py
   :language: python
   :start-after: # [step-3-continuous]
   :end-before: # [end-step-3-continuous]
   :dedent: 4

.. figure:: images/maps_score.png
   :width: 80%
   :alt: The NDVI of the Aberystwyth area drawn with the viridis colour map and a colour bar.

   The NDVI with a fixed 0–1 colour bar range: highest (about 0.85–0.9,
   yellow) in the woods and long grass, about 0.7 on short grass and scrub,
   and lowest on the town and the beaches (0.2–0.4). Water (no data: the
   sea, the lakes and the wider rivers) is transparent. Contains modified
   Copernicus Sentinel data (2018).

Relief: hillshaded maps
~~~~~~~~~~~~~~~~~~~~~~~

A semi-transparent hillshade over a map shows the terrain under it, which
helps to read hydrology, geomorphology or land cover maps.
:func:`~rsgispy.mapping.raster.add_hillshade` draws a hillshade over what is
already on an axis, from a hillshade image (e.g. made by
:func:`rsgispy.elevation.illumination.hillshade`) or straight from a DEM
(``dem_img``): the hillshade of the map's bounding box is then made in memory,
with the sun at ``azimuth`` and ``zenith`` (default 315° and 45°), and no file
is written. :func:`~rsgispy.mapping.raster.create_shaded_cmap_img_map` draws a
colour-mapped raster (e.g. elevation, flow accumulation or a wetness index)
with the hillshade over it, a colour bar, a scale bar and a north arrow, and
returns the colour-mapped image artist. ``hshade_alpha`` (``alpha`` of
``add_hillshade``) sets how strongly the relief shows: 0.3–0.5 keeps the
colours readable.

When a large raster is drawn on a small map, Matplotlib resamples it, and
features one pixel wide (stream channels, roads) can disappear if the colours
are resampled. The raster map functions resample the data values instead
(``interp_stage="data"``, the default), which keeps them visible; set
``interpolation="nearest"`` to show the pixels as they are, and ``img_alpha``
to make any raster layer semi-transparent.

.. literalinclude:: ../examples/making_maps.py
   :language: python
   :start-after: # [step-3-relief]
   :end-before: # [end-step-3-relief]
   :dedent: 8

.. figure:: images/maps_relief.png
   :width: 100%
   :alt: Elevation of the Aberystwyth area shaded by its hillshade, and the land cover with a hillshade over it.

   a) The elevation (sea level to 750 m on the uplands in the north-east)
   shaded by its own hillshade: the valleys of the Rheidol and the Ystwyth run
   west to the sea at Aberystwyth; b) a hillshade over the land cover. Both
   hillshades were made in memory from the DEM. The sea is no data in the DEM
   (white in a). Contains modified Copernicus Sentinel data (2018).

Step 4 — Thematic maps and categorical legends
----------------------------------------------

:func:`~rsgispy.mapping.raster.create_thematic_raster_map` draws a
classification with the **colour table stored in the image**; pixel value 0
is transparent (``alpha_backgd=True``). If an image has no colour table, add
one first, e.g. with :func:`rsgispy.imageutils.colourtable.define_colour_table`
(the classification functions in rsgispy write it for you from the class
colours in the configuration).

With ``show_legend=True`` a categorical legend is built from the colour table
and ``classes_info``, which gives the class names and can be:

* a ``{pixel value: name}`` dictionary (as here);
* a ``{pixel value: (name, colour)}`` dictionary, to override the colours;
* a :class:`~rsgispy.classification.classdata.ClassificationConfig` or the
  path of its JSON file, so the legend uses exactly the names and colours of
  the classification.

Where the legend goes:

* ``legend_position='right'`` (default), ``'bottom'``, ``'left'`` or
  ``'top'`` adds a separate legend axis next to the map, so the legend never
  hides part of the map. ``'bottom'`` suits wide maps and many classes.
* ``legend_overlay=True`` puts the legend inside the map at ``legend_loc``,
  for maps with an empty corner.
* ``legend_ax=`` writes the legend into an axis you created, for multi-panel
  figures that share one legend.

The number of rows and columns is chosen from the space available and the
label lengths, or set with ``legend_ncols``/``legend_nrows`` (here three
columns, so the ten classes of the tutorial data fit in four rows). Further options go
in ``legend_kwargs`` and are passed to
:func:`~rsgispy.mapping.legend.add_categorical_legend`; for example
``only_present_in_img=True`` leaves out classes that do not occur in the
mapped area (useful for zoomed maps), and ``show_pxl_vals=True`` adds the
pixel values to the labels. For a legend on its own, e.g. for slides or a
poster, use :func:`~rsgispy.mapping.legend.save_categorical_legend_img`.

.. literalinclude:: ../examples/making_maps.py
   :language: python
   :start-after: # [step-4-thematic]
   :end-before: # [end-step-4-thematic]
   :dedent: 4

.. figure:: images/maps_land_cover.png
   :width: 70%
   :alt: The land cover map of the Aberystwyth area with a legend of ten classes below it.

   The random forest land cover map with its legend in a companion axis
   below the map. Short grass covers 46 % of the area, scrub 15 %, long
   grass 10 % and forest 13 % (conifer 5 %, deciduous 8 %). Contains
   modified Copernicus Sentinel data (2018).

Step 5 — Vector layers and choropleth maps
------------------------------------------

:func:`~rsgispy.mapping.vector.create_vec_lyr_map` draws one or more
GeoDataFrames (all in the CRS of the map). Give one value per layer for the
fill colours, line colours, widths, transparency (``vec_fill_alphas``) and
``plot_zorders``. The z-order sets the drawing order: rasters are drawn at
z-order 1 by default, so vectors over an image need a higher value.
Polygons and points are filled with ``vec_fill_clrs`` and outlined with
``vec_line_clrs``; use a fill colour of ``'none'`` to draw polygon outlines
only. Lines are drawn in ``vec_line_clrs``, and a layer that mixes geometry
types is split by type, so each part gets the right colours.

A **choropleth** colours each polygon by an attribute.
:func:`~rsgispy.mapping.vector.create_choropleth_vec_lyr_map` takes the column
name, a colour map and an optional normalisation (use a fixed range, e.g. 0
to 100 % with ``matplotlib.colors.Normalize``, so the colour bar matches
the map). Here the attribute is the percentage of forest pixels in each
region (``forest_vals``: the conifer and deciduous forest classes, 3 and 4),
counted from the land cover map with
:func:`~rsgispy.zonalstats.polygon.calc_zonal_band_pxl_counts`, which writes
one column of pixel counts per class value; any column
works, e.g. mean values from
:func:`~rsgispy.zonalstats.polygon.calc_zonal_band_stats` or the area of each
restoration potential class from :doc:`multi_criteria_analysis`.

Choropleths should show **rates or proportions** (percentage forest, people
per km²), not totals, because large polygons would otherwise dominate the
map. Use a sequential colour map for one-sided values and a diverging one for
values around a reference.

.. literalinclude:: ../examples/making_maps.py
   :language: python
   :start-after: # [step-5-vectors]
   :end-before: # [end-step-5-vectors]
   :dedent: 4

The printed table is a check that the zonal statistics worked (percentages
between 0 and 100, and different between regions). pandas shows the first
and last five of the 108 squares:

.. code-block:: text

       name  forest_pct
   0     A1        0.00
   1     A2        0.87
   2     A3        1.15
   3     A4        4.86
   4     A5        5.72
   ..   ...         ...
   103   I8       64.66
   104   I9       15.89
   105  I10        2.28
   106  I11       24.45
   107  I12        2.72

   [108 rows x 2 columns]

The percentages are of the whole square, so the squares on the coast (A1 is
all sea) include the sea. Half the squares have less than 9 % forest; the
most wooded are in the south (I8, 65 %) and the east (E12, 60 %), with the
conifer plantations.

.. figure:: images/maps_choropleth.png
   :width: 70%
   :alt: A 2 km grid over the Aberystwyth area, each square coloured by its percentage of forest.

   Choropleth of the percentage of forest in each 2 km square, on a fixed 0
   to 100 % scale. Contains modified Copernicus Sentinel data (2018).

To show point density (e.g. thousands of observations), use
:func:`~rsgispy.mapping.vector.create_vec_pt_density_map`, which needs an axis
created with ``projection='scatter_density'`` (``mpl-scatter-density``
package).

Step 6 — Scale bars, north arrows and zoom boxes
------------------------------------------------

Every drawing function adds a scale bar by default (``show_scale_bar``,
``scale_bar_loc``, ``scale_bar_style``, ``scale_bar_units``) and can add a north
arrow (``show_north_arrow``). When several layers are drawn on one axis, turn
these off in the layer functions and add them once with the element functions,
which have more options:

* :func:`~rsgispy.mapping.elements.add_scale_bar` — the length is chosen as
  a "nice" number close to ``length_fraction`` (default a quarter) of the map
  width, or set with ``fixed_value``. Styles are
  ``SCALEBAR_STYLE_SOLID``, ``SCALEBAR_STYLE_ALTERNATING`` (black and white
  segments; ``n_segments``) and ``SCALEBAR_STYLE_TICKS``; units are metres or
  kilometres (chosen from the map width if ``units=None``). Pass the map's
  ``epsg_code``: for a geographic CRS (degrees) the bar is converted to
  metres at the centre latitude.
* :func:`~rsgispy.mapping.elements.add_north_arrow` — styles
  ``NORTH_ARROW_STYLE_TRIANGLE``, ``_ARROW``, ``_COMPASS`` and ``_SIMPLE``.
  In a projected CRS grid north and true north differ slightly away from the
  central meridian; ``rotation`` (degrees) can correct this for large areas.
* :func:`~rsgispy.mapping.elements.draw_bboxes_to_axis` — rectangles (with
  optional labels), for example to show where zoomed-in panels or field
  sites are.

The constants are defined in :mod:`rsgispy.mapping.elements`. Locations use
the Matplotlib legend names (``'upper left'``, ``'lower right'`` …).

.. literalinclude:: ../examples/making_maps.py
   :language: python
   :start-after: # [step-6-elements]
   :end-before: # [end-step-6-elements]
   :dedent: 4

Step 7 — Inset overview maps
----------------------------

An overview (locator) map shows where the study area is.
:func:`~rsgispy.mapping.layout.get_overview_info` takes the map's bounding box
and EPSG code and returns the extent of the overview (the centre of the
study area ± ``overview_buf``, in the units of the overview CRS, so degrees
for an EPSG:4326 layer of countries) and the centre point.
:func:`~rsgispy.mapping.layout.add_overview_maps` then adds an inset axis
(size and offsets in fractions of the main axis, placed with ``loc`` like a
legend) with the overview layer and a marker at the study area. It returns
the inset axis, so you can add more to it (e.g. a rectangle with
:func:`~rsgispy.mapping.elements.draw_bboxes_to_axis` instead of a point for
larger study areas). Choose ``overview_buf`` so the inset shows a recognisable
area: a country for a local study, a continent for a national one.

.. literalinclude:: ../examples/making_maps.py
   :language: python
   :start-after: # [step-7-overview]
   :end-before: # [end-step-7-overview]
   :dedent: 4

Step 8 — Exporting at publication resolution
--------------------------------------------

* Set the **figure size in inches at the final printed size** (Step 1) and
  keep font sizes at 8–10 points; do not resize the image afterwards, or the
  text size changes.
* **PNG/TIFF** (raster): use ``dpi=300`` for print (600 or more for line art
  with thin lines), 100–150 for web pages and slides. File size grows with
  the square of the dpi.
* **PDF/SVG** (vector): lines, text and vector layers stay sharp at any
  zoom, and journals often prefer them. Raster layers in the figure are
  embedded at the figure's ``dpi``.
* ``bbox_inches='tight'`` removes white space around the figure (and makes
  sure companion legend axes are included); ``pad_inches`` sets the margin.
* Close figures you have saved with ``plt.close(fig)``, especially in loops,
  or memory use grows.

A zoomed-in panel uses the same functions with a smaller ``bbox``; the
stretch is then computed for the zoomed area.

.. literalinclude:: ../examples/making_maps.py
   :language: python
   :start-after: # [step-8-export]
   :end-before: # [end-step-8-export]
   :dedent: 4

.. figure:: images/maps_final.png
   :width: 80%
   :alt: Land cover map of the Aberystwyth area with rivers, compass, scale bar, zoom box and an inset map of Great Britain and Ireland.

   The composed map from Steps 6–8: land cover, rivers, alternating scale
   bar, compass north arrow, zoom box and an inset overview map (±5° around
   the study area, from the country outlines). Contains modified Copernicus
   Sentinel data (2018).

Several maps side by side
-------------------------

To compare maps of the same area (an image and its classification, two
classifications, a change distance and the change over the image), use
:func:`~rsgispy.mapping.layout.create_map_panels`. It takes a list of panels,
each a ``dict`` (or a :class:`~rsgispy.mapping.layout.MapPanel`) naming the
panel type (``'rgb'``, ``'cmap'``, ``'thematic'`` or ``'vector'``), its data and
a title, and draws them with the functions of Steps 2–5 over one ``bbox``.
A panel can have a ``base_img`` drawn below it (e.g. a change mask over the
image); ``map_kw`` passes other arguments to the drawing function. The
thematic panels share one legend (``legend_position`` ``'right'`` or
``'bottom'``), ``'cmap'`` panels can have a colour bar, and one scale bar and
north arrow are drawn on the first panel. ``ncols`` sets the panels per row.
It returns ``(fig, axs)``, with one axis per panel.

.. literalinclude:: ../examples/making_maps.py
   :language: python
   :start-after: # [step-panels]
   :end-before: # [end-step-panels]
   :dedent: 4

.. figure:: images/maps_panels.png
   :width: 100%
   :alt: Three map panels of the Rheidol valley: the Sentinel-2 image, the land cover and the NDVI.

   The zoom area (the Rheidol valley east of Aberystwyth) as three panels,
   with one legend, one scale bar and one north arrow. The conifer and
   deciduous woods on the valley sides stand out in all three. Contains
   modified Copernicus Sentinel data (2018).

Step 9 — Basemaps (optional, online)
------------------------------------

Basemaps give context (roads, place names, terrain) behind vector data or a
semi-transparent raster. Two sources are supported:

* :func:`~rsgispy.mapping.basemaps.add_contextily_basemap` adds web map tiles
  (OpenStreetMap, OpenTopoMap, Esri imagery and many others in
  ``contextily.providers``) *under* the layers already on the axis. Draw your
  layers first so the axis extent is set, and give the map's EPSG code; the
  tiles are reprojected to it. ``cx_zoom_lvl='auto'`` picks the tile zoom from
  the extent; a fixed level gives a consistent look across maps. Respect the
  provider's attribution (shown by default) and usage policy.
* :func:`~rsgispy.mapping.basemaps.create_wmts_img_map` reads an image from a
  WMTS service (for example a national aerial photography service or a
  satellite mosaic) for the bounding box, reprojects it if needed and draws
  it like an RGB image. ``wmts_zoom_level`` controls the resolution.

Both need the internet, so they are optional in the example (the tests
replace them with stand-ins). For maps that must be reproducible, prefer your
own imagery (Step 2), as tile services change over time.

.. literalinclude:: ../examples/making_maps.py
   :language: python
   :start-after: # [step-9-basemap]
   :end-before: # [end-step-9-basemap]
   :dedent: 4

Regenerating the figures
------------------------

The figures on this page are made by running the example on the tutorial
data at 100 dpi (see `Overview`_), with the data folder of a clone of the
rsgispy tutorials. From ``docs/examples``:

.. code-block:: bash

   python making_maps_tutorial.py --data-dir ~/rsgispy-tutorials/data \
       --out-dir maps_tutorial

It takes about half a minute. PNGs over 400 KB were reduced to 256 colours
to keep them small:

.. code-block:: python

   from PIL import Image

   img = Image.open("maps_tutorial/maps_panels.png").convert("RGB")
   img.quantize(colors=256, method=Image.Quantize.FASTOCTREE).save(
       "docs/guides/images/maps_panels.png", optimize=True)

Performance and large datasets
------------------------------

* **Only the bounding box is read.** The raster functions read the window
  inside ``bbox``, so maps of a small area of a large image are fast. For a
  map of a whole large image every pixel is read at full resolution (image
  overviews/pyramids are not used), which needs a lot of memory and gives no
  visible benefit. Make a reduced-resolution copy first, close to the display
  resolution (a 6.7 inch map at 300 dpi is about 2000 pixels wide), e.g. with
  :func:`rsgispy.imageutils.warp.gdal_translate` (``out_size`` or
  ``target_res``, with ``resample_alg=rsgispy.INTERP_AVERAGE`` for continuous
  data and nearest neighbour for thematic data).
* **VRTs** work as inputs, so a mosaic of tiles can be mapped without copying
  (:mod:`rsgispy.imageutils.vrt`).
* **Large vector layers** — pass ``sub_in_vec=True`` to
  :func:`~rsgispy.mapping.vector.create_vec_lyr_map` to drop features outside
  the bounding box before drawing, simplify geometries for small-scale maps
  (``gdf.simplify(tolerance)``), or rasterise very dense layers.
* **Many maps in a loop** (e.g. one per region or date) — create one figure
  per map, save it, and close it with ``plt.close(fig)``; use the ``Agg``
  backend (``matplotlib.use("Agg")``) on servers without a display.
* **PDF/SVG with large rasters** can become very large files; use PNG for
  image-heavy maps, or lower the ``dpi`` of the embedded rasters.

Common problems
---------------

* **Layers do not line up / a layer is missing** — all rasters and vectors on
  one axis must be in the same CRS. Reproject vectors with
  ``gdf.to_crs(epsg)`` and rasters with
  :func:`rsgispy.imageutils.warp.reproject_image`. A layer drawn outside the
  axis extent is simply not visible.
* **The scale bar is wrong or very long** — the map is in a geographic CRS
  (degrees) and the EPSG code is unknown, so degrees were treated as metres.
  Make sure the image or GeoDataFrame has a CRS (or pass ``epsg_code`` to
  :func:`~rsgispy.mapping.elements.add_scale_bar`), or better, map in a
  projected CRS.
* **The image is black, white or washed out** — wrong stretch or no data not
  set. Set the image no data value (or ``no_data_val``) so fill values (e.g.
  0 or -9999) are excluded from the stretch statistics, and try
  ``IMG_STRETCH_CUMULATIVE``.
* **"Do not recognise stretch method"** — ``img_stch`` must be one of the
  ``rsgispy.IMG_STRETCH_*`` constants; ``IMG_STRETCH_USER`` also needs
  ``stch_min_max_vals`` (one ``{'min': …, 'max': …}`` dictionary per band).
* **A thematic image raises an exception about the colour table** — the image
  has no colour table; add one (Step 4).
* **Legend labels are numbers or classes are missing** — give
  ``classes_info`` with the names. Classes without a colour table entry, and
  class 0 (``ignore_zero=True``), are left out.
* **"contextily is required"** — install the ``mapping`` extra; basemaps
  also need an internet connection (and some providers need an API key).

Related guides
--------------

* :doc:`pixel_classification_workflow`
* :doc:`multi_criteria_analysis`
* :doc:`hydrology_workflow`
