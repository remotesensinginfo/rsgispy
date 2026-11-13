rsgispy.tools
=============

The :mod:`rsgispy.tools` subpackage provides plotting, mapping, file, hashing,
archive, text/JSON I/O, geometry, projection, UTM, statistics, feature
selection, thresholding, accuracy, tile cache, image/PDF, web, HTTP, FTP, and
WMTS utilities without requiring GDAL's ``osgeo`` bindings.

Files, Text, JSON, Archives & System
------------------------------------

* :doc:`tools/files` — Path inspection, searching, locks, sorting, and temp directory contexts.
* :doc:`tools/textio` — Reading/writing text files, line lists, and prettifying XML.
* :doc:`tools/jsonio` — Reading and writing JSON and gzipped JSON files.
* :doc:`tools/archives` — Creating and extracting zip, tar, tar.gz, and tar.bz2 archives.
* :doc:`tools/hashing` — SHA, MD5, BLAKE2, and SHA3 file/string hashing.
* :doc:`tools/credentials` — Base64 encoding and username/password credential files.
* :doc:`tools/system` — Environment variables and available CPU core count.

.. toctree::
   :maxdepth: 1

   tools/files
   tools/textio
   tools/jsonio
   tools/archives
   tools/hashing
   tools/credentials
   tools/system

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

Visualisation, Cartography & Imaging
------------------------------------

* :doc:`tools/plotting` — Reading raster arrays and applying image stretches for Matplotlib.
* :doc:`tools/mapping` — Creating raster, thematic, choropleth, density, and WMTS maps.
* :doc:`tools/colours` — Hex and RGB colour conversions.
* :doc:`tools/imaging` — GIF animations, EXIF reading, PDF image extraction, and QR codes.
* :doc:`tools/wmts` — Querying WMTS capabilities and downloading WMTS imagery.

.. toctree::
   :maxdepth: 1

   tools/plotting
   tools/mapping
   tools/colours
   tools/imaging
   tools/wmts

Maths, Statistics, Thresholding & Accuracy
------------------------------------------

* :doc:`tools/maths` — Numeric comparisons, rounding, bounds checks, and powersets.
* :doc:`tools/stats` — Masking, semivariograms, standardisation, and nan-percentiles.
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
