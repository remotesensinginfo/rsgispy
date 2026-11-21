rsgispy.cli
===========

The :mod:`rsgispy.cli` subpackage provides 11 pure-Python command-line tools
ported from RSGISLib with standardised option names, multi-file/glob input
handling, JSON output modes where applicable, and zero ``osgeo`` or external
shell dependencies.

Each tool is installed as a console script under both its extensionless name
(e.g., ``rsgisimginfo``) and its legacy ``.py`` alias (e.g.,
``rsgisimginfo.py``).

Standardised Option Conventions
-------------------------------

* ``-i`` / ``--input`` — Input file(s) or glob pattern(s) (handles both
  shell-expanded lists and quoted globs).
* ``-o`` / ``--output`` — Output file or directory path.
* ``-l`` / ``--layer`` — Vector layer name (defaults to the first layer).
* ``-f`` / ``--format`` — Output file or CRS format.
* ``-j`` / ``--n-cores`` — Worker process count for parallel operations.
* ``--no-pbar`` — Suppress progress bars.
* ``--json`` — Emit machine-readable JSON output.
* ``-V`` / ``--version`` — Print tool and ``rsgispy`` version.

Available Command-Line Tools
----------------------------

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Command
     - Description
   * - ``rsgisbatchconvert2cog``
     - Convert raster images to Cloud-Optimised GeoTIFFs (COGs) with optional
       multi-core processing and existing-file validation.
   * - ``rsgisbuildimglut``
     - Build a vector bounding-box spatial look-up table (LUT) from raster
       images.
   * - ``rsgiscalcimgstats``
     - Populate continuous or thematic image statistics, histograms, and
       pyramids.
   * - ``rsgischkgdalfile``
     - Validate GDAL raster or vector datasets and optionally delete corrupt
       files.
   * - ``rsgiscopybandnames``
     - Copy band names (descriptions) from a reference image to one or more
       target images.
   * - ``rsgisfilehash``
     - Compute MD5, SHA, SHA3, or BLAKE2 cryptographic file hashes.
   * - ``rsgisimginfo``
     - Display complete, compact raster metadata (format, grid, pixel size,
       extent, CRS, overviews, and per-band summary table) in a single open.
   * - ``rsgisproj``
     - Convert or inspect CRS definitions from rasters, vectors, EPSG, WKT,
       PROJ4, ESRI ``.prj``, or UTM zones.
   * - ``rsgisuserpassfile``
     - Create a base64-encoded username/password credentials file.
   * - ``rsgisvectools``
     - Inspect vector datasets (OGR formats and GeoParquet): compact summary,
       layers, columns, feature count, CRS, bbox, geometry type, and unique
       attribute values.
   * - ``rsgiswgs84imgres``
     - Calculate approximate pixel resolution in metres at the centre latitude
       of geographic (WGS84) raster images.

Module Reference
----------------

.. automodule:: rsgispy.cli.batchconvert2cog
   :members:
   :undoc-members:

.. automodule:: rsgispy.cli.buildimglut
   :members:
   :undoc-members:

.. automodule:: rsgispy.cli.calcimgstats
   :members:
   :undoc-members:

.. automodule:: rsgispy.cli.chkgdalfile
   :members:
   :undoc-members:

.. automodule:: rsgispy.cli.copybandnames
   :members:
   :undoc-members:

.. automodule:: rsgispy.cli.filehash
   :members:
   :undoc-members:

.. automodule:: rsgispy.cli.imginfo
   :members:
   :undoc-members:

.. automodule:: rsgispy.cli.proj
   :members:
   :undoc-members:

.. automodule:: rsgispy.cli.userpassfile
   :members:
   :undoc-members:

.. automodule:: rsgispy.cli.vectools
   :members:
   :undoc-members:

.. automodule:: rsgispy.cli.wgs84imgres
   :members:
   :undoc-members:
