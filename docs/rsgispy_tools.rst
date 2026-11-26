rsgispy.tools
=============

The :mod:`rsgispy.tools` subpackage provides plotting, file, hashing,
archive, text/JSON I/O, geometry, projection, UTM, statistics, feature
selection, thresholding, accuracy, tile cache, image/PDF, web, HTTP, FTP, and
WMTS utilities without requiring GDAL's ``osgeo`` bindings.

.. _plotting_conventions:

Plotting conventions
--------------------

Every rsgispy plotting function follows the same rules, so plots can be
combined in your own figures, shown in a notebook and saved the same way.
The helpers named here are in :mod:`rsgispy.tools.plotting`.

**Functions that make a figure** (``plot_*``, e.g.
:func:`~rsgispy.tools.plotting.plot_histogram_threshold`,
:func:`~rsgispy.regression.regmetrics.plot_residuals`):

* take ``ax=None`` (``axs=None`` for several panels) and draw on it when
  given; otherwise they create a figure of a sensible ``figsize``;
* take ``out_plot_file=None`` and ``out_dpi=300``: when a file is given the
  figure is saved with ``bbox_inches="tight"``
  (:func:`~rsgispy.tools.plotting.save_plot`). Saving does not close the
  figure, so it still shows in a notebook;
* return ``(fig, ax)``, or ``(fig, axs)`` for several panels (``axs`` is a
  NumPy array of axes);
* never call ``plt.show()`` and never close the figure they return, or one
  that was passed in.

In a notebook, assign the result (``fig, ax = rplot.plot_...(...)``); the
figure is then shown once, at the end of the cell. Close figures you no longer
need with ``plt.close(fig)``.

**Functions that draw on a given axis** (maps and map elements, ``create_*``
and ``add_*`` in :doc:`rsgispy.mapping <rsgispy_mapping>`) return what the caller needs next:
the image artist (``matplotlib.image.AxesImage``) for raster maps, so
``fig.colorbar(img)`` works; the collections for vector maps; and the legend,
scale bar, north arrow, rectangles or text for map elements. The
colour-mapped map functions take ``show_colour_bar=False`` and
``cbar_label=None`` to draw their colour bar themselves.

**Functions that compute data and can also plot it** (``calc_*``, ``run_*``,
``select_*``, the model fitting functions) keep returning their data. Their
plotting is done by a separate function that follows the rules above; their
``out_*plot*`` arguments call it, save the figure and close it (so loops over
many models or variables do not keep figures in memory). To see the plot in
a notebook, call the plotting function on the returned data, e.g.
:func:`~rsgispy.tools.stats.plot_semivariogram` on the result of
:func:`~rsgispy.tools.stats.calc_semivariogram`, or
:func:`~rsgispy.regression.regmetrics.plot_pred_vs_obs` next to
:func:`~rsgispy.regression.regmetrics.calc_reg_metrics`.

**Functions that write one plot per feature or sample to a directory** (e.g.
:func:`~rsgispy.regression.regfeatures.target_scatter_plots`,
:func:`~rsgispy.classification.classfeatures.comparison_box_plots`) save and
close each figure and return the list of files written.

**House style.** Every plotting function applies the rsgispy house style
(:func:`~rsgispy.tools.plotting.get_plot_style_rc`) with
``matplotlib.rc_context`` while it runs, so your global ``rcParams`` are
never changed (importing rsgispy changes nothing either). Use
:func:`~rsgispy.tools.plotting.rsgispy_plot_style` to draw your own plots
in the same style; the colours are
:data:`~rsgispy.tools.plotting.CATEGORICAL_COLOURS` (Okabe–Ito),
:data:`~rsgispy.tools.plotting.SEQUENTIAL_CMAP` and
:data:`~rsgispy.tools.plotting.DIVERGING_CMAP`. The :doc:`guides/plot_gallery`
shows every plot.

**Writing a new plotting function** in rsgispy: decorate it with
:func:`~rsgispy.tools.plotting.use_plot_style` (``plot_type="map"`` for maps
and map elements), get the figure with
:func:`~rsgispy.tools.plotting.get_fig_ax` or
:func:`~rsgispy.tools.plotting.get_fig_axs`, save with
:func:`~rsgispy.tools.plotting.save_plot`, return ``(fig, ax)``, and add a
case to ``tests/test_plotting_convention.py``.

.. code-block:: python

   import rsgispy.tools.plotting as rplot

   @rplot.use_plot_style
   def plot_values(vals, *, title=None, figsize=(6.4, 4.2), ax=None,
                   out_plot_file=None, out_dpi=rplot.PLOT_DPI):
       fig, ax = rplot.get_fig_ax(ax, figsize=figsize)
       ax.plot(vals)
       if title is not None:
           ax.set_title(title)
       if out_plot_file is not None:
           rplot.save_plot(fig, out_plot_file, out_dpi=out_dpi)
       return fig, ax

Files, Text, JSON, Archives & System
------------------------------------

* :doc:`tools/files` — Path inspection, searching, locks, sorting, and temp directory contexts.
* :doc:`tools/tables` — Exporting pandas DataFrames to Parquet (with compression) and tabular formats.
* :doc:`tools/textio` — Reading/writing text files, line lists, and prettifying XML.
* :doc:`tools/jsonio` — Reading and writing JSON and gzipped JSON files.
* :doc:`tools/archives` — Creating and extracting zip, tar, tar.gz, and tar.bz2 archives.
* :doc:`tools/hashing` — SHA, MD5, BLAKE2, and SHA3 file/string hashing.
* :doc:`tools/credentials` — Base64 encoding and username/password credential files.
* :doc:`tools/system` — Environment variables and available CPU core count.
* :doc:`tools/sensors` — Landsat MTL and Sentinel-2 L2A XML metadata header parsers.

.. toctree::
   :maxdepth: 1

   tools/files
   tools/tables
   tools/textio
   tools/jsonio
   tools/archives
   tools/hashing
   tools/credentials
   tools/system
   tools/sensors

Geometry, Projections, UTM & Tile Caches
----------------------------------------

* :doc:`tools/geometry` — Bounding box operations, grid snapping, and coordinate reprojection.
* :doc:`tools/projection` — WKT/EPSG/CRS conversions and geodesic distances.
* :doc:`tools/utm` — UTM zone lookups and lat/lon <-> UTM conversions.
* :doc:`tools/tilecache` — Web Mercator XYZ/TMS/Quadkey slippy-map tile calculations.

.. toctree::
   :maxdepth: 1

   tools/geometry
   tools/projection
   tools/utm
   tools/tilecache

Visualisation & Imaging
-----------------------

* :doc:`tools/plotting` — The house plot style, figure and saving helpers, raster arrays and image stretches for Matplotlib.
* :doc:`tools/colours` — Hex and RGB colour conversions.
* :doc:`tools/imaging` — GIF animations, EXIF reading, PDF image extraction, and QR codes.
* :doc:`tools/wmts` — Querying WMTS capabilities and downloading WMTS imagery.

.. toctree::
   :maxdepth: 1

   tools/plotting
   tools/colours
   tools/imaging
   tools/wmts

Maths, Statistics, Thresholding & Accuracy
------------------------------------------

* :doc:`tools/maths` — Numeric comparisons, rounding, bounds checks, and powersets.
* :doc:`tools/stats` — Masking, semivariograms, standardisation, nan-percentiles, and similarity metrics.
* :doc:`tools/thresholding` — Otsu, Yen, ISODATA, cross-entropy, Li, and kurtosis/skew thresholds.
* :doc:`tools/featureselection` — VIF, CQV, correlation, LassoLars, and Breusch-Pagan tests.
* :doc:`tools/accuracy` — Binary accuracy, probability scores, and bias-variance decomposition.

.. toctree::
   :maxdepth: 1

   tools/maths
   tools/stats
   tools/thresholding
   tools/featureselection
   tools/accuracy

Strings, Dates & Dictionaries
-----------------------------

* :doc:`tools/strings` — UID generation, numeric parsing, zero-padding, and string cleaning.
* :doc:`tools/dates` — Day deltas, month ranges, season checks, and datetime intervals.
* :doc:`tools/dicts` — Nested dictionary path queries and typed value extraction.

.. toctree::
   :maxdepth: 1

   tools/strings
   tools/dates
   tools/dicts

Web, HTTP, FTP & Dataset Validation
-----------------------------------

* :doc:`tools/http` — HTTP checks, JSON requests, file downloads, and listing databases.
* :doc:`tools/ftp` — FTP directory listings, downloads, uploads, and listing databases.
* :doc:`tools/web` — Scraping web pages for image and file links.
* :doc:`tools/checkdatasets` — Raster, vector, and HDF5 integrity validation and comparison.

.. toctree::
   :maxdepth: 1

   tools/http
   tools/ftp
   tools/web
   tools/checkdatasets
