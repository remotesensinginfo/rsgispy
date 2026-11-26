Processing large rasters efficiently
====================================

Earth observation data are big: a single Sentinel-2 tile at 10 m is
10,980 x 10,980 pixels, a national mosaic can be hundreds of thousands of
pixels across, and a stack of dates or features multiplies that again. Most
rsgispy functions never read a whole image into memory. They read, process
and write the image in *blocks* (windows) of a few hundred pixels, so they
work on images much larger than the computer's memory. This guide explains
how that works and which choices affect speed, memory use and file sizes:

* how to estimate the memory a step needs;
* the processing block size and the output creation options (compression,
  tiling, BigTIFF);
* virtual rasters (VRTs) instead of copying data;
* writing your own block-wise calculations with
  :func:`rsgispy.imageprocess.process_blocks`;
* chaining several operations in memory with a tile graph
  (:class:`~rsgispy.imageprocess.TileGraph`) and filter chains
  (:mod:`rsgispy.imagefilter.chained`), without intermediate files;
* splitting an image into tiles for parallel or cluster processing, and
  mosaicking the results;
* the ``n_jobs`` and ``numba`` options of the segmentation, zonal statistics
  and polygonisation functions;
* choosing output formats: GeoTIFF, Cloud Optimised GeoTIFF (COG), KEA and
  GeoParquet.

The example script runs each technique on a small image so that it can be
tested; the text gives the guidance for real data.

.. contents:: Contents
   :local:
   :depth: 1

Installation
------------

The block processing, VRT, tiling and tile graph functions only need the core
rsgispy dependencies. ``numba`` (the ``numba`` extra) compiles the slow loops
in the segmentation, polygonisation, GLCM texture, mode filter, hydrology and
other functions, and ``pyarrow`` (in the ``vector`` extra) is needed to write
GeoParquet:

.. code-block:: bash

   pip install "rsgispy[numba,vector]"

Overview
--------

.. code-block:: text

   Sentinel-2 SREF image (large)
            │
            ▼
   1. Memory estimate ──────────► full image vs one block (MiB)
   2. Block size + creation options (environment variables)
            │
            ├─► 3. VRT band subset (no copy) ─► 4. process_blocks ─► NDVI
            ├─► 5. Tile graph: NDVI + filtered NIR in one pass, no temp files
            ├─► 6. Filter chain: median → Sobel on NDVI
            ├─► 7. Tiles (parallel) → per-tile NDVI → mosaic VRT
            │            │
            │            └─► 8. Final GeoTIFF (or COG)
            └─► 9. Tiled multi-core segmentation → zonal statistics
                    (Parquet) → polygons (GeoParquet)

The code in this guide comes from the example script
``docs/examples/large_rasters.py``, which the rsgispy test suite runs on a
small synthetic image. The input is a surface reflectance image with no data
value 0 (e.g. from
:func:`~rsgispy.imagecalibration.sensorlvl2data.create_stacked_sen2_esa_l2a_img`;
see :doc:`analysis_ready_data`), in a projected CRS.

Step 1 — Estimating memory
--------------------------

Before running a step on a big image, estimate how much memory it would need
if the image were read at once, and how much one block needs. The full image
needs ``columns x rows x bands x bytes per value``; most rsgispy functions
read blocks as 64-bit floats (8 bytes per value), whatever the data type on
disk, so a block needs ``block size² x bands x 8`` bytes, plus a few
temporary arrays of the same size while it is processed.

.. literalinclude:: ../examples/large_rasters.py
   :language: python
   :start-after: # [step-1-memory]
   :end-before: # [end-step-1-memory]
   :dedent: 4

For a 10,980 x 10,980 pixel, 10-band ``uint16`` Sentinel-2 image this gives
about 2.2 GiB as ``uint16`` (9.0 GiB as float64), but only 5 MiB for a
256 x 256 block of all 10 bands (as float64). Block processing therefore keeps the memory
use small and constant; what grows with the image size is the processing
time and, for some functions, memory per *object* (e.g. the number of
segments or polygons), not per pixel.

Functions that return or work on the whole image as one numpy array are the
exception: check the full-image estimate before calling them, and subset or
sample the image first if it does not fit.

Moving-window functions (filters, texture, terrain) need more: each block is
read with a *halo* of ``filter_size // 2`` pixels and, for many statistics,
a temporary array with ``filter_size²`` values per pixel is created. Their
memory grows with ``block size² x filter_size²`` (see
`Performance and large datasets`_ for measurements).

Step 2 — Block size and creation options
----------------------------------------

**Block size.** ``rsgispy.imageprocess.DEFAULT_BLOCK_SIZE`` is 256
pixels: one 256 x 256 tile of a tiled GeoTIFF and a quarter of a 512 x 512
COG tile. It was 512 up to rsgispy 0.3; 256 pixel blocks were as fast or
faster in the measurements below and need a quarter of the memory for window
filters. Change it for the whole session with
:func:`~rsgispy.imageprocess.set_default_block_size`, or by setting the
``RSGISPY_BLOCK_SIZE`` environment variable (``RSGISLIB_BLOCK_SIZE`` is also
read) before running a script; :func:`~rsgispy.imageprocess.get_default_block_size`
returns the value in use and :func:`~rsgispy.imageprocess.reset_default_block_size`
restores the default. Functions with an explicit ``block_size`` argument
(:func:`~rsgispy.imageprocess.process_blocks`,
:class:`~rsgispy.imageprocess.TileGraph`,
:func:`~rsgispy.imagefilter.chained.apply_filter_chain`,
:func:`~rsgispy.zonalstats.labels.calc_label_zonal_stats`) use that value
instead; without it, :class:`~rsgispy.imageprocess.TileGraph` and
:func:`~rsgispy.imagefilter.chained.apply_filter_chain` also use the session
default.

How to choose it:

* **256–1024** is a good range for point-wise calculations (band maths,
  indices). Very small blocks (e.g. 64) are slower because of the overhead per
  block; very large blocks use more memory without being faster.
* **Smaller blocks (256) for moving-window filters with large windows**,
  because their memory grows with the square of both the block size and the
  window size.
* Keep the block size a multiple of the internal tile size of your files
  (usually 256 or 512) so each block reads whole tiles. GeoTIFF outputs are
  tiled with GDAL's default 256 x 256 tiles (see below), so the default 256
  pixel block covers exactly one tile; pass ``BLOCKXSIZE`` and
  ``BLOCKYSIZE`` in the creation options for other tile sizes (multiples of
  16).

**Creation options.** GeoTIFF outputs are tiled and DEFLATE-compressed by
default: ``TILED=YES``, ``COMPRESS=DEFLATE``, ``INTERLEAVE=BAND`` and
``BIGTIFF=IF_SAFER``. Tiled files are fast to read in blocks, compression
typically makes files several times smaller, and ``BIGTIFF=IF_SAFER``
switches to BigTIFF when a file may exceed the 4 GB limit of classic TIFF.
Other formats (KEA, COG, ...) use their GDAL driver defaults. To change the
options, set the ``RSGISPY_IMG_CRT_OPTS_<FORMAT>`` environment variable
(colon-separated ``KEY=VALUE`` pairs; ``RSGISLIB_IMG_CRT_OPTS_<FORMAT>`` is
also read), which replaces the default, or pass ``creation_opts`` to a
function, which is merged on top (``creation_opts`` with ``BLOCKYSIZE`` but
no ``TILED`` or ``BLOCKXSIZE``, e.g. ``{"BLOCKYSIZE": 1}``, give a striped
GeoTIFF with that many rows per strip). The helpers in
:mod:`rsgispy.imageutils.envopts` set common choices:

* :func:`~rsgispy.imageutils.envopts.set_env_vars_zstd_gtiff_outs` or
  :func:`~rsgispy.imageutils.envopts.set_env_vars_lzw_gtiff_outs` for other
  compression; ``bigtiff=True`` forces BigTIFF.
* :func:`~rsgispy.imageutils.envopts.set_img_creation_opts` for any other
  set of options and format; ``set_img_creation_opts("GTIFF", {})`` gives
  plain striped, uncompressed GeoTIFFs.
* :func:`~rsgispy.imageutils.envopts.reset_img_creation_opts` restores the
  defaults.

.. literalinclude:: ../examples/large_rasters.py
   :language: python
   :start-after: # [step-2-settings]
   :end-before: # [end-step-2-settings]
   :dedent: 4

This prints the block size and
``{'TILED': 'YES', 'COMPRESS': 'DEFLATE', 'INTERLEAVE': 'BAND', 'BIGTIFF': 'IF_SAFER'}``.
The block size and the ``set_*`` helpers use environment variables, so they
also apply to every rsgispy function called later in the same process (and
to worker processes started from it). The built-in defaults do not: rsgispy
does not set any environment variables itself.

Step 3 — Virtual rasters instead of copies
------------------------------------------

A VRT is a small XML file that describes a raster made from other rasters:
a subset of bands, a stack of several files, or a mosaic of tiles. Reading a
VRT reads the source files directly, so a VRT costs no disk space and no time
to create, and it is always up to date with its sources. Use one instead of
writing a new file whenever a step only needs to *rearrange* data:

* :func:`~rsgispy.imageutils.vrt.create_vrt_band_subset` — some bands of an
  image, in any order;
* :func:`~rsgispy.imageutils.vrt.create_stack_images_vrt` — the bands of
  several images (on the same grid) as one multi-band image, e.g. a feature
  stack for classification;
* :func:`~rsgispy.imageutils.vrt.create_mosaic_images_vrt` — tiles or
  scenes as one mosaic (see step 7).

.. literalinclude:: ../examples/large_rasters.py
   :language: python
   :start-after: # [step-3-vrt]
   :end-before: # [end-step-3-vrt]
   :dedent: 4

``red_nir.vrt`` is a 2-band image (``Red``, ``NIR``) that can be passed to
any rsgispy function. VRTs refer to their source files by path:
:func:`~rsgispy.imageutils.vrt.create_vrt_band_subset` writes absolute paths
by default, while the stack and mosaic functions write paths relative to the
VRT (``write_abs_path=False``), so keep those VRTs next to their sources, or
pass ``write_abs_path=True`` if the VRT will be moved on its own.
VRTs that resample or reproject on the fly are convenient but repeat that
work every time they are read; write a real file if it is read many times.

Step 4 — Your own block-wise calculations
-----------------------------------------

:func:`~rsgispy.imagecalc.maths.band_maths` covers most per-pixel
calculations with an expression. When you need numpy (or any Python code),
use the same machinery directly:

1. describe the input bands with :class:`~rsgispy.imageprocess.BandDefn`
   (a variable name, a file and a band);
2. open them with :func:`~rsgispy.imageprocess.open_input_datasets` and
   compute the common pixel grid with
   :func:`~rsgispy.imageprocess.compute_spatial_intersection` (the inputs
   must have the same CRS and pixel size; the output covers their overlap);
3. create the output with :func:`~rsgispy.imageprocess.open_output_dataset`
   (it applies the creation options from step 2);
4. call :func:`~rsgispy.imageprocess.process_blocks` with a function that
   receives a dictionary of 2D ``float64`` arrays (one per variable) for each
   block and returns the output block (2D for one band, ``(bands, rows,
   cols)`` for several).

.. literalinclude:: ../examples/large_rasters.py
   :language: python
   :start-after: # [step-4-blocks]
   :end-before: # [end-step-4-blocks]
   :dedent: 4

The block function must only use the pixels of its block: it is called once
per block and never sees the whole image, so image-wide values (a mean, a
percentile for a stretch) must be computed first (e.g. with
:func:`~rsgispy.imageutils.stats.pop_img_stats` or
:func:`~rsgispy.imageprocess.compute_dataset_band_stats`) and passed in.
Handle no data explicitly (here, pixels where either input is 0 become
-9999), set the output no data value, then compute statistics and pyramids
with :func:`~rsgispy.imageutils.stats.pop_img_stats`. For moving-window
calculations, use :func:`~rsgispy.imageprocess.run_window_halo_blocks` or the
tile graph (step 5), which read the extra halo pixels around each block.

Step 5 — Chaining operations in memory with a tile graph
--------------------------------------------------------

A typical chain of steps (filter, filter again, combine with the original,
calculate an index) normally writes an intermediate file after every step,
and reads it again for the next one. For large images this disk traffic
often takes longer than the calculations. A
:class:`~rsgispy.imageprocess.TileGraph` runs the whole chain tile by tile in
memory and writes only the final outputs. The building blocks are:

* sources — :class:`~rsgispy.imageprocess.RasterSource` (bands of a file)
  and :class:`~rsgispy.imageprocess.ArraySource` (a numpy array);
* operations — :class:`~rsgispy.imageprocess.BandMathOp` (a band maths
  expression over named inputs), :class:`~rsgispy.imageprocess.PointwiseOp`
  (a Python function per pixel block) and
  :class:`~rsgispy.imageprocess.WindowOp` (a moving-window function with a
  halo);
* filter nodes — each filter that can be chained has a node builder next to
  its file-based function, named after it without ``apply_`` and with
  ``_node`` (e.g. :func:`~rsgispy.imagefilter.stats.median_filter_node`,
  :func:`~rsgispy.imagefilter.sar.lee_filter_node`,
  :func:`~rsgispy.imagefilter.edge.sobel_filter_node`; see
  :doc:`../rsgispy_imageprocess` for the list);
* band nodes — :class:`~rsgispy.imageprocess.SelectBands` (pick bands of a
  node by 1-based index) and :class:`~rsgispy.imageprocess.StackBands`
  (stack the bands of several nodes, keeping their band names);
* :func:`rsgispy.imagefilter.chained.build_filter_chain_node`, which adds a
  sequence of filters (:class:`~rsgispy.imagefilter.chained.FilterStep`:
  window statistics such as ``'median'`` and ``'mean'``, ``'gaussian'``,
  ``'sobel'``, the ``'lee'`` speckle filter, kernels and band maths) to a
  graph and returns the last node;
* sinks — :class:`~rsgispy.imageprocess.DiskSink` (write a raster) and
  :class:`~rsgispy.imageprocess.MemorySink` (return a numpy array from
  :meth:`~rsgispy.imageprocess.TileGraph.run`).

The graph works out how many extra halo pixels each source must read: here
the 3 x 3 median and the 5 x 5 mean need a 1 + 2 = 3 pixel halo around every
tile, so the result is identical to running the filters one after the other
on the whole image. Several sinks can share the same sources, so this one
pass writes both the NDVI and a high-pass filtered NIR band (the NIR band
minus its smoothed version, which enhances edges and small features).

.. literalinclude:: ../examples/large_rasters.py
   :language: python
   :start-after: # [step-5-tilegraph]
   :end-before: # [end-step-5-tilegraph]
   :dedent: 4

All sources must be on the same pixel grid: the graph checks their size,
transform and CRS when it is built and raises an error naming the rasters
that differ (use
:func:`~rsgispy.imageutils.warp.resample_img_to_match` first). Sources on the
same file, such as ``red`` and ``nir`` here, share one open file and are read
together for each tile. Every node also carries its number of bands, band
names, data type and no data value, so a
:class:`~rsgispy.imageprocess.DiskSink` without ``band_names`` or ``nodata``
uses those of its input, and ``print(graph.describe())`` shows the whole
graph with the halo and metadata of each node. With ``nodata`` set on a :class:`~rsgispy.imageprocess.BandMathOp`,
pixels that are no data in any input stay no data in the output (the NDVI no
data value is 0 here, the input's; use a ternary expression as in step 7 if
you need a different value). The NDVI from the graph equals the one from
step 4 wherever the input is valid, which the test checks.

Step 6 — Filter chains
----------------------

When the chain is a sequence of filters on one image,
:func:`~rsgispy.imagefilter.chained.apply_filter_chain` builds and runs the
tile graph for you: give the input, the output and the steps. Here the NDVI
is median filtered (to remove isolated noisy pixels) and then the Sobel
gradient magnitude is calculated, which highlights boundaries between
vegetated and non-vegetated areas.

.. literalinclude:: ../examples/large_rasters.py
   :language: python
   :start-after: # [step-6-filter-chain]
   :end-before: # [end-step-6-filter-chain]
   :dedent: 4

The no data value is read from the input (``-9999`` here) and honoured by
every step. A ``'band_math'`` step can use ``x`` (the result of the previous
step), ``input`` (the original image) and any earlier step given a ``name``,
e.g. ``FilterStep('band_math', params={'expr': 'input - x'})`` after a
Gaussian smoothing gives an unsharp mask.

Step 7 — Tiling and mosaicking
------------------------------

Block processing bounds the memory of a single process, but it still runs on
one core and one computer. For very large areas, split the image into tiles
with :func:`~rsgispy.imageutils.tiling.create_tiles_multi_core` (``n_cores``
worker processes; tiles named ``<base>_x<col>y<row>.tif``), process each tile
as an independent job (in a loop, with :mod:`multiprocessing`, or as separate
jobs on a cluster, e.g. with ``pb_process_tools``), and mosaic the results.
:func:`~rsgispy.imageutils.tiling.create_raster_tiles_bbox` creates tiles
for given bounding boxes instead.

.. literalinclude:: ../examples/large_rasters.py
   :language: python
   :start-after: # [step-7-tiles]
   :end-before: # [end-step-7-tiles]
   :dedent: 4

The mosaic is a VRT (step 3), so it costs nothing to create; write it to a
real file with :func:`~rsgispy.imageutils.warp.gdal_translate` (step 8) or
:func:`~rsgispy.imageutils.mosaic.create_img_mosaic` when it is final. For
point-wise operations, such as this NDVI, the mosaic equals the result from
the whole image. Operations that look at neighbouring pixels (filters,
texture, segmentation) give different results at the tile edges unless each
tile is processed with an overlap that is cut off afterwards; prefer the
functions that handle this for you (the block and tile graph functions,
and the ``tile_size`` / ``overlap`` options of the segmentation functions in
step 9).

How big should tiles be? Large enough that the per-tile overheads (starting
a job, opening files, building statistics) are small compared with the
processing time — typically a few thousand pixels across (e.g. 5,000 x
5,000) — and small enough that the number of tiles is at least the number of
workers. With ``n_cores > 1``, call the function from inside an
``if __name__ == "__main__":`` block (see its documentation).

Step 8 — Writing the final product
----------------------------------

The VRT mosaic from step 7 refers to every tile file. Write the final
product as a single GeoTIFF with
:func:`~rsgispy.imageutils.warp.gdal_translate`, here directly from the tile
mosaic VRT, and check the file with
:func:`~rsgispy.tools.checkdatasets.check_gdal_image_file`:

.. literalinclude:: ../examples/large_rasters.py
   :language: python
   :start-after: # [step-8-final]
   :end-before: # [end-step-8-final]
   :dedent: 4

``check_gdal_image_file`` returns ``(True, '')`` for a valid file. rsgispy
writes GeoTIFFs tiled and DEFLATE-compressed by default; options passed to a
function (here ``PREDICTOR=3``, which helps the compression of floating-point
data) are added to the defaults, and the ``envopts`` functions change the
defaults for every later output in the same process.

For products that are shared, served or read over HTTP from cloud storage,
write a Cloud Optimised GeoTIFF (COG) instead, with ``gdalformat="COG"``. A
COG is a GeoTIFF with internal tiles, overviews and a layout that lets
programs read just the part and resolution they need. Useful COG options:
``COMPRESS`` (``DEFLATE``, ``ZSTD``, or ``LZW``; ``JPEG`` or ``WEBP`` for
lossy RGB previews), ``PREDICTOR=YES`` (helps compression of continuous
data), ``OVERVIEW_RESAMPLING`` (``AVERAGE`` or ``CUBIC`` for continuous
data, ``NEAREST`` or ``MODE`` for classes) and ``BLOCKSIZE`` (default 512).
The ``rsgisbatchconvert2cog`` command line tool converts many files at once.
The ``set_env_vars_*_cog_outs`` functions in
:mod:`rsgispy.imageutils.envopts` set COG creation options for functions
that write ``gdalformat='COG'``; the GeoTIFF defaults are never applied to
COGs.

**How rsgispy writes COGs.** GDAL can only create a COG by copying a
complete image, so when an rsgispy function writes ``gdalformat='COG'``
(including :func:`~rsgispy.imageutils.warp.gdal_translate`), it writes the
data block by block to a temporary tiled GeoTIFF next to the output (in the
local temporary directory for GDAL virtual file system outputs such as
``/vsis3/...`` or ``/vsimem/...``), adds the statistics to it
(``calc_stats=True``), and only then converts it with GDAL's COG driver,
which builds the overviews (with ``NEAREST`` resampling for thematic outputs
unless ``OVERVIEW_RESAMPLING`` is set); the temporary file is deleted, also
when an error occurs. Memory use is the same as for a GeoTIFF output,
but the output directory needs free space for the temporary file as well as
the COG, and the conversion adds roughly the time of one more read and write
of the image. The statistics are stored in the COG without breaking its
layout. Your own block loops get the same behaviour from
:func:`~rsgispy.imageprocess.open_output_dataset`; register statistics with
:func:`~rsgispy.imageprocess.pop_stats_on_close` inside the ``with`` block.
Calling :func:`~rsgispy.imageutils.stats.pop_img_stats` (or another function
that edits the file) on an existing COG never edits it in place: the COG is
rebuilt through a temporary GeoTIFF
(:func:`~rsgispy.imageprocess.update_cog`) with the compression, quality and
layout settings recorded in the file, which keeps it valid but costs a full
re-encode (and another round of lossy compression for JPEG or WEBP), so
prefer ``calc_stats=True`` when the COG is written.

**Choosing output formats**

* **GeoTIFF** (``gdalformat='GTIFF'``, tiled and DEFLATE-compressed by
  default) for intermediate and working files: fast block-wise reading and
  writing, readable by everything.
* **COG** for final, shared or cloud-hosted products.
* **KEA** (``gdalformat='KEA'``; an HDF5-based format) supports raster
  attribute tables with many columns, compression and large files, and is
  convenient for segmentations and thematic maps used with RSGISLib-style
  workflows; fewer programs can read it (see the KEA note under
  `Common problems`_).
* **VRT** for band subsets, stacks and mosaics of existing files.
* **GeoParquet** (``.parquet``) for vector outputs and attribute tables with
  many rows (segments, polygons, samples): compressed, columnar (reading a
  few columns is fast) and much faster to write and read than GeoPackage or
  shapefiles. Use GeoPackage when the data must be edited in a desktop GIS.

Step 9 — Tiled, multi-core segmentation, statistics and polygons
----------------------------------------------------------------

Segmenting an image, summarising each segment and turning the segments into
polygons are the steps that most often run out of memory or time on large
images, because their cost grows with the number of objects as well as the
number of pixels. The rsgispy functions for these steps have options to
bound memory and use several cores:

* the segmentation functions in :doc:`rsgispy.segmentation <../rsgispy_segmentation>` (here
  :func:`~rsgispy.segmentation.shepherd.shepherd_segmentation`) take ``tile_size``
  (segment the image tile by tile; scenes larger than 16 million pixels are
  tiled automatically), ``overlap`` (a halo around each tile so segments near
  tile edges see their neighbours), ``stitch_seams`` (join segments cut by
  tile edges), ``n_jobs`` (tiles processed in parallel worker processes; the
  result is the same for any ``n_jobs``) and ``use_numba``;
* :func:`~rsgispy.zonalstats.labels.calc_label_zonal_stats` computes the
  statistics of every segment in a single block-wise pass over the label
  image (``block_size``), and writes the table to Parquet with ``out_file``;
* :func:`~rsgispy.vector.create.polygonise_raster_tiled` polygonises in
  tiles (``tile_size``) on ``n_jobs`` cores with a ``numba`` kernel and
  streams the polygons to GeoParquet, so memory does not depend on the image
  size (see :ref:`polygonise-benchmark` for timings).
  :func:`~rsgispy.segmentation.utils.polygonise_segments` calls it with the
  options for segment label images.

.. literalinclude:: ../examples/large_rasters.py
   :language: python
   :start-after: # [step-9-segments]
   :end-before: # [end-step-9-segments]
   :dedent: 4

The outputs are the label image ``s2_segs.tif`` (segment IDs ``1..N``, 0 = no
data), a Parquet table with one row per segment (``seg_id``, ``pxl_count``,
and ``<feature>_mean`` / ``<feature>_stddev`` for each band, where the
feature name is ``<image name>_b<band>``, e.g. ``s2_sref_10m_b8_mean`` for
``s2_sref_10m.tif``, unless
``name`` or ``band_names`` are given to the ``BandSpec``) and a GeoParquet layer
with one (Multi)Polygon per segment, sorted by ``seg_id``. Check that the
number of rows in the table and the layer equals the largest segment ID. The
:doc:`segment_ml_workflow` guide continues from here to classify the
segments.

**numba.** rsgispy uses ``numba`` when it is installed and falls back to
pure Python/numpy otherwise, giving the same results. The first call of a
compiled function in a new process takes a few seconds to compile it (many
kernels are cached on disk after the first run), so the speed-up shows on large images,
not small tests. Set the environment variable ``RSGISPY_DISABLE_NUMBA=1`` to
turn compilation off (e.g. to check whether a problem comes from numba), or
pass ``use_numba=False`` to the segmentation functions.

**n_jobs.** Worker processes take a second or two to start and each holds
its own tile, so:

* use ``n_jobs=1`` for small images (up to a few million pixels);
* for large images, use up to the number of physical (performance) cores,
  and check that ``n_jobs x memory per worker`` fits in memory;
* do not combine many worker processes with multi-threaded libraries in each
  worker (numpy/BLAS, numba's ``parallel`` kernels): rsgispy's segmentation
  workers limit BLAS/OpenMP to one thread each to avoid oversubscription,
  and you can set ``NUMBA_NUM_THREADS`` / ``OMP_NUM_THREADS`` yourself for
  your own code;
* on macOS, Windows and recent Python versions on Linux, worker processes
  re-import the calling script, so call these functions from inside an
  ``if __name__ == "__main__":`` block.

Performance and large datasets
------------------------------

**Measured memory and time against block size.** Peak memory allocated by
Python and numpy (measured with :mod:`tracemalloc`, so excluding GDAL's
cache) and wall-clock time, for a 4096 x 4096 pixel single-band ``float32``
image (64 MiB) on an Apple M2 Max laptop (96 GB RAM), with this version of
rsgispy and numba installed:

.. list-table::
   :header-rows: 1
   :widths: 46 18 18 18

   * - Function
     - Block size
     - Time
     - Peak memory
   * - :func:`~rsgispy.imagecalc.maths.band_maths` (dB conversion)
     - 64 / 256 / 512 / 1024 / 4096
     - 0.73 / 0.25 / 0.28 / 0.33 / 0.78 s
     - 1 / 4 / 14 / 53 / 656 MiB
   * - :func:`~rsgispy.imagefilter.sar.apply_lee_filter` (7 x 7)
     - 256 / 512 / 1024
     - 6.0 / 6.8 / 6.8 s
     - 105 / 419 / 1675 MiB
   * - :func:`~rsgispy.imagefilter.chained.apply_filter_chain` (``'lee'``,
       7 x 7)
     - 256 / 512 / 1024
     - 4.7 / 7.5 / 6.5 s
     - 106 / 424 / 1696 MiB

Point-wise calculations need only a few MiB per block and are fastest with
blocks of 256–1024 pixels. The 7 x 7 window filter needed about 400 MiB with
512 pixel blocks (about six times the size of the whole image) and four times
as much with 1024 pixel blocks, without being faster: use 256 pixel blocks
for window filters with large windows, especially when running several
processes at once.

**Other tips**

* **Use VRTs** for band subsets, stacks and mosaics instead of copying data
  (step 3), and write a real file only for final products or when a VRT that
  resamples is read many times.
* **Avoid intermediate files** with a tile graph or a filter chain (steps 5
  and 6).
* **Store data with the right type**: reflectance as ``uint16`` scaled by
  10,000 rather than ``float32`` halves the file size; class maps as
  ``uint8``.
* **Compute statistics and pyramids once**, at the end:
  :func:`~rsgispy.imageutils.stats.pop_img_stats` reads the whole image;
  pass ``calc_stats=False`` to intermediate steps, and use
  ``use_multi_core_pyramids=True`` with ``n_cores`` for very large images.
* **Local disks** are much faster than network drives for intermediate
  files; on a cluster, write intermediates to the node's local scratch.
* **Zonal statistics**: for many zones, rasterise them to a label image and
  use :func:`~rsgispy.zonalstats.labels.calc_label_zonal_stats` (one pass
  over the image) rather than polygon zonal statistics, which read the image
  once per polygon (see :doc:`zonal_stats_and_vectors`).
* **Batch processing** of many scenes or tiles: ``pb_process_tools``
  generates and runs one job per file on a workstation (GNU Parallel) or a
  cluster (Slurm).

Common problems
---------------

* **Memory errors with window filters** — reduce the block size
  (``set_default_block_size(256)``) or the window size, and run fewer
  processes at once.
* **Results differ at tile edges** — the operation uses neighbouring pixels
  and the tiles were processed separately; use an overlap, or the block, tile
  graph and segmentation functions, which handle halos.
* **"not on the same grid" / different sizes** — VRT stacks, tile graphs and
  :func:`~rsgispy.imageprocess.compute_spatial_intersection` need the same
  CRS and pixel size; resample with
  :func:`~rsgispy.imageutils.warp.resample_img_to_match` first.
* **Files larger than 4 GB fail to write** — use ``BIGTIFF=IF_SAFER`` or
  ``BIGTIFF=YES`` (the GeoTIFF default and the ``set_env_vars_*`` helpers
  set ``IF_SAFER``; for COGs set it with the ``set_env_vars_*_cog_outs``
  helpers).
* **A COG is "not a valid COG" after processing** — it was edited in place
  by a tool other than rsgispy (for example with
  ``IGNORE_COG_LAYOUT_BREAK=YES``, or by building overviews with
  :func:`~rsgispy.imageutils.stats.build_img_pyramids`). rsgispy's statistics
  functions rebuild a COG instead; to repair a file, convert it again with
  ``rasterio.shutil.copy("in.tif", "out.tif", driver="COG")``.
* **No data in outputs** — custom block functions must handle no data
  themselves and set the output no data value; check with
  ``rasterio.open(path).nodata`` and the image statistics.
* **Multi-core functions hang or fail with ``RuntimeError`` at start-up** —
  the call is not inside ``if __name__ == "__main__":``.
* **Slow first call** — numba is compiling; later calls (and later runs,
  for cached kernels) are fast.
* **Changing the no data value of a KEA file fails** — GDAL's KEA driver
  cannot replace a no data value once a file has one, so
  :func:`~rsgispy.imageutils.stats.pop_img_stats` or
  :func:`~rsgispy.imageutils.metadata.set_img_no_data_value` with a
  *different* value raises an error naming the old and new values (setting
  the same value again, as the ``calc_stats=True`` step does, is fine). Pass
  the right ``no_data_val`` when the KEA file is written.
* **Environment settings leak between steps** — the block size and creation
  options set with the ``set_*`` helpers are environment variables; reset them
  with :func:`~rsgispy.imageprocess.reset_default_block_size` and
  :func:`~rsgispy.imageutils.envopts.reset_img_creation_opts`, or set them
  again.

Related guides
--------------

* :doc:`analysis_ready_data`
* :doc:`zonal_stats_and_vectors`
* :doc:`segment_ml_workflow`
