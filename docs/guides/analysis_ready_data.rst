Analysis-ready data: from download to a cloud-free composite
============================================================

Most analyses start with the same job: turning a set of satellite scenes into
images that can be used directly — surface reflectance in one file per
scene, with clouds and their shadows removed, on a common pixel grid, and
usually combined into a single cloud-free composite for the period of
interest. Data prepared like this is called *analysis-ready data* (ARD). This
guide walks through that preparation in rsgispy for Sentinel-2 Level-2A data,
from searching for and downloading scenes to a checked, compressed GeoTIFF
composite. The same steps apply to Landsat Collection 2
Level-2 data (see the notes in each step).

Why each stage matters:

* **Stacking** — Level-2A products are delivered as a folder of separate
  band files at 10, 20 and 60 m with an integer encoding (and, since
  processing baseline 04.00, an offset). Stacking puts the bands you need into
  one image at one resolution with a documented scaling, so every later tool
  can read it.
* **Cloud masking** — a single unmasked cloud in one date turns into a bright
  artefact in a composite, an abrupt "change" in a time series or a false
  class in a classification.
* **Co-registration** — images of different dates are not perfectly aligned
  (typically by a fraction of a pixel up to a pixel or two). Misalignment
  blurs composites and creates false change along every edge.
* **Compositing** — no single date is cloud-free everywhere; a per-pixel
  median or percentile of the clear observations gives a complete image that
  is also robust to undetected clouds and shadows.
* **Checked outputs** — tiled, compressed GeoTIFFs with statistics and
  overviews open quickly in GIS software; checking the files catches
  truncated or corrupt outputs before they reach later stages.

The outputs of this guide are the inputs of the other guides: for example the
:doc:`object-based classification <segment_ml_workflow>`, the
:doc:`multi-criteria analysis <multi_criteria_analysis>`, the
:doc:`change detection <change_detection>` and the
:doc:`map making <making_maps>` guides.

.. contents:: Contents
   :local:
   :depth: 1

Installation
------------

Searching for and downloading data needs the ``dataaccess`` extra
(``requests``); the CloudSEN12 cloud masking needs the ``cloudmask`` extra
(PyTorch and ``segmentation-models-pytorch``). Everything else uses the core
dependencies:

.. code-block:: bash

   pip install "rsgispy[dataaccess]"
   # optional: the CloudSEN12 deep learning cloud masking
   pip install "rsgispy[dataaccess,cloudmask]"

On a machine with a GPU, install the PyTorch build for your CUDA version first
(see the PyTorch installation instructions); a CPU build works but is slower.

Overview
--------

.. code-block:: text

   bbox + dates ─► 1. Search and download (optional) ─► *.SAFE.zip ─► *.SAFE
                                                                        │
   *.SAFE (per tile) ───────────────────────────────────────────────────┤
                                                                        ▼
   2. Stack ──────────► <scene>_10m_sref.tif (10 bands, reflectance x 10000)
        │               <scene>_10m_clouds.tif, _vld.tif, _scl.tif
        ▼               (<scene>_10m_cloudsen12_sref.tif, 12 bands)
   3. Cloud mask ─────► <scene>_10m_sref_clrsky.tif (clouds = 0 = no data)
        ▼
   4. Co-register ────► <scene>_10m_sref_clrsky_coreg.tif (to the clearest date)
        ▼
   5. Indices ────────► <scene>_..._ndvi.tif
        ▼
   6. Composite ──────► <tile>_sref_median.tif, <tile>_ndvi_p10_p50_p90.tif,
        │               <tile>_n_clear_obs.tif
        ▼
   7. Mosaic tiles ───► sref_mosaic.tif, ndvi_mosaic.tif (+ VRTs)
        ▼
   8. Check outputs

The code in this guide comes from the example script
``docs/examples/analysis_ready_data.py`` (included below step by step), which
is run by the rsgispy test suite on small synthetic ``.SAFE`` products (two
overlapping tiles with three dates each, one of them misregistered). In the
script, ``scenes`` is a dictionary mapping each Sentinel-2 tile name (e.g.
``'T30UVD'``) to its ``.SAFE`` folders, and every output is written under
``out_dir``. Run it on your own data from the command line, for example:

.. code-block:: bash

   # scenes you have already downloaded and unzipped
   python analysis_ready_data.py --safe S2*_MSIL2A_*.SAFE --out-dir outputs

   # search for and download the scenes first
   export RSGIS_COP_USER="you@example.com"
   export RSGIS_COP_PASS="your-password"
   python analysis_ready_data.py --bbox -4.2 -3.9 52.3 52.5 \
       --dates 2024-06-01 2024-08-31 --max-cloud 40 --out-dir outputs

Step 1 — Finding and downloading scenes
---------------------------------------

:func:`rsgispy.dataaccess.copernicus_odata.query_scn_lst` searches the
Copernicus Data Space Ecosystem (CDSE) catalogue. The search area is a
bounding box in WGS84 degrees, given in the order
``(x_min, x_max, y_min, y_max)`` (longitude range first, then latitude), not
the ``(min_x, min_y, max_x, max_y)`` order used by many other tools. Choosing
the search parameters:

* **Dates** — wide enough to give several clear observations of every pixel.
  In cloudy regions a summer composite may need three months; in arid regions
  a few weeks is enough. Keep the period within one season if you want the
  composite to represent that season's land cover.
* **cloud_cover** — the maximum cloud cover of the *whole scene* (0–100 %).
  A scene with 60 % cloud may still be clear over your area, so do not set it
  too low; 40–60 % is a reasonable start, and the per-pixel cloud mask in
  step 3 does the real work.
* **product_type** — ``RSGIS_ODATA_PROD_TYPE_S2_MSI_2A`` restricts the search
  to Level-2A (surface reflectance) products; without it, Level-1C
  (top-of-atmosphere) products are also returned.
* **max_n_rslts** — the maximum number of scenes returned (the API returns
  results in pages; use ``start_n`` to get the next page).

Each result is a dictionary of catalogue metadata (``Name``, ``ContentDate``,
``Footprint``, download ``Locations``, …). Print the names and check the
dates and tiles before downloading.

Downloading needs a (free) CDSE account. Do not write your password into
scripts: :func:`rsgispy.tools.credentials.resolve_env_user_pass` reads the
username and password from environment variables (here ``RSGIS_COP_USER``
and ``RSGIS_COP_PASS``, the variables
:func:`~rsgispy.dataaccess.copernicus_odata.get_access_token` also reads by
default) and, with ``required=True``, raises a clear error if they are not
set. :func:`~rsgispy.dataaccess.copernicus_odata.download_scns` downloads each
scene as ``<name>.SAFE.zip``, verifies the MD5 checksum and skips files that
already exist, so an interrupted download can simply be run again. Each zip is
then extracted with :func:`rsgispy.tools.archives.unzip_file`, and the tile
name (the sixth part of the product name, e.g. ``T30UVD``) is used to group
the scenes by tile.

.. literalinclude:: ../examples/analysis_ready_data.py
   :language: python
   :start-after: # [step-1-download]
   :end-before: # [end-step-1-download]
   :dedent: 4

For Landsat, :mod:`rsgispy.dataaccess.usgs_m2m` (USGS EarthExplorer
machine-to-machine API, e.g.
:func:`~rsgispy.dataaccess.usgs_m2m.usgs_search`) and
:mod:`rsgispy.dataaccess.nasa_cmr` (NASA Common Metadata Repository) provide
the equivalent searches and downloads.

.. note::

   Access tokens are only valid for a short time, so get a new token (call
   ``get_access_token`` again) for each batch of downloads rather than
   reusing one for hours. The test suite does not download anything: it
   replaces the search and download functions with fakes that copy zipped
   synthetic products.

Step 2 — Stacking the bands
---------------------------

:func:`rsgispy.imagecalibration.sensorlvl2data.create_stacked_sen2_esa_l2a_img`
reads the product metadata (``MTD_MSIL2A.xml``) and writes, for each scene:

* ``<scene>_10m_sref.tif`` — 10 bands (``Blue``, ``Green``, ``Red``,
  ``RE_B5``, ``RE_B6``, ``RE_B7``, ``NIR_B8``, ``NIR_B8A``, ``SWIR1``,
  ``SWIR2``; B02–B08, B8A, B11, B12) as 32-bit integers equal to
  reflectance × ``scale_factor`` (10000), with no data = 0. The processing
  baseline ≥ 04.00 offset (``BOA_ADD_OFFSET``, −1000) is applied for you, so
  values from different baselines are directly comparable. Valid values are
  typically 0–10000 (bright targets such as snow or cloud can exceed 10000).
* ``<scene>_10m_scl.tif`` — the Scene Classification Layer (SCL) from the
  L2A processor.
* ``<scene>_10m_clouds.tif`` — a cloud mask derived from the SCL:
  ``1`` = cloud (SCL medium/high probability cloud and thin cirrus),
  ``2`` = cloud shadow, ``0`` = clear (stored as no data).
* ``<scene>_10m_vld.tif`` — ``1`` where the scene has data (SCL ≠ 0).
* with ``inc_cloudsen12_img=True``, ``<scene>_10m_cloudsen12_sref.tif`` — the
  12 bands (adding B01 and B09) in the order required by the CloudSEN12 cloud
  masking in step 3.

With ``out_res_10m=True`` the 20 m bands are resampled to 10 m and, by default
(``sharpen_10m=True``), sharpened using the 10 m bands, which keeps field
boundaries crisp in the red-edge and SWIR bands. Use ``out_res_10m=False`` for
a 20 m stack (four times fewer pixels) when 10 m detail is not needed.

.. literalinclude:: ../examples/analysis_ready_data.py
   :language: python
   :start-after: # [step-2-stack]
   :end-before: # [end-step-2-stack]
   :dedent: 4

**Check:** open a stack in a GIS (or read it with rasterio) and check that
there are 10 named bands, that vegetation has a NIR value of roughly
2500–4500 and water a NIR value below about 500. Values around −1000 or a
uniform shift of all bands usually means the baseline offset has been applied
twice (for example to data that was already converted elsewhere).

**Landsat.** For Landsat 8/9 Collection 2 Level-2 archives, use
``create_stacked_sref_ls_oli_cl2_lv2_img(archive, out_dir, tmp_dir)``
(:func:`~rsgispy.imagecalibration.sensorlvl2data.create_stacked_sref_ls_oli_cl2_lv2_img`;
:func:`~rsgispy.imagecalibration.sensorlvl2data.create_stacked_sref_ls_4_5_7_lv2_img`
for Landsat 4, 5 and 7). It writes the stacked reflectance (×10000, no data
= −9999) and ``_clouds`` / ``_vld_msk`` images with the same coding as above
(1 = cloud, 2 = shadow), so step 3 works unchanged apart from the no data
value. To use the individual quality flags (cirrus, snow, confidence levels),
pass ``inc_full_qa_pxl_img=True``, or decode a ``QA_PIXEL`` file with
:func:`~rsgispy.imagecalibration.sensorlvl2data.parse_landsat_c2_qa_pixel_img`,
which writes a 12-band image (``Fill``, ``DilatedCloud``, ``Cirrus``,
``Cloud``, ``CloudShadow``, ``Snow``, ``Clear``, ``Water`` and four
confidence bands).

Step 3 — Cloud and shadow masking
---------------------------------

There are two sources of cloud masks:

* **The SCL** (the default here) comes with every L2A product and needs no
  extra software. It is reasonable for thick clouds but often misses thin
  cloud and cloud edges, and confuses bright surfaces (buildings, sand,
  snow) with cloud.
* **CloudSEN12** —
  :func:`rsgispy.imagecalibration.cloudmasking.apply_cloudsen12_s2_sref_cloud_mask`
  runs a deep learning model trained on the CloudSEN12 dataset. It is
  considerably more accurate, particularly for thin cloud and cloud shadow,
  and is the better choice for compositing and change detection. It needs the
  ``cloudmask`` extra and downloads the model weights (once) into a cache
  folder (``RSGISPY_CLOUDSEN12_WEIGHTS_DIR``, or ``~/.cache/rsgispy/cloudsen12``); pass
  ``weights_dir`` and ``download_weights=False`` on machines without internet
  access. Its input is the 12-band ``_cloudsen12_sref`` stack from step 2; the
  stack already has the BOA offsets applied, so pass ``add_offset=0`` and the
  same ``scale_factor``. The output is ``0`` = no data, ``1`` = clear,
  ``2`` = thick cloud, ``3`` = thin cloud and ``4`` = cloud shadow. Use
  ``device='cuda'`` (or ``'mps'`` on Apple silicon, or ``'auto'``) when a GPU
  is available; on a CPU a full tile takes several minutes.

Either mask is applied with :func:`rsgispy.imageutils.masking.mask_img`, which
sets every band to the no data value (0) where the mask has one of the cloud
values. The proportion of clear pixels in each masked image is calculated
with :func:`rsgispy.imagecalc.maths.calc_prop_true_exp` and used in the next
step to choose a reference image; it is also a quick way to drop scenes that
are almost entirely cloudy.

.. literalinclude:: ../examples/analysis_ready_data.py
   :language: python
   :start-after: # [step-3-cloud-mask]
   :end-before: # [end-step-3-cloud-mask]
   :dedent: 4

The printed proportions look like this (synthetic data):

.. code-block:: text

   {'S2A_MSIL2A_20240602T112119_..._T30UVD_..._10m_sref_clrsky.tif': 0.905,
    'S2A_MSIL2A_20240612T112119_..._T30UVD_..._10m_sref_clrsky.tif': 0.902,
    'S2A_MSIL2A_20240622T112119_..._T30UVD_..._10m_sref_clrsky.tif': 0.777, ...}

**Check:** view each ``_clrsky`` image over the original stack. Cloud edges
and thin haze are the usual failures; if they remain, use CloudSEN12, and
consider growing the mask by a few pixels before applying it (for example a
dilation with :mod:`rsgispy.imagemorphology`), which removes most cloud edge
and shadow edge pixels at the cost of a few clear ones.

Step 4 — Co-registering the dates
---------------------------------

Sentinel-2 images of the same tile are on the same 10 m grid, but the image
content of different dates can be offset by a fraction of a pixel up to about
a pixel (more for older products and in steep terrain). Each date is
registered to a reference date — here the clearest scene of the tile —
with :func:`rsgispy.imageregistration.warping.coregister_images`, which:

1. matches a grid of windows (``win_size`` pixels, every ``pixel_gap``
   pixels) between the two images on one band (here NIR, which has the most
   texture over land). Windows with too much no data (clouds) or too little
   texture are skipped, which is why the images are masked *before* they are
   co-registered: cloud edges would otherwise give false matches;
2. removes inconsistent tie points with RANSAC (``residual_thres`` in metres);
3. resamples the image onto the reference image grid (``match_ref_grid=True``
   by default), keeping no data as 0.

``TRANSFORM_SHIFT`` (a translation) is appropriate for images of the same
tile and orbit; use ``TRANSFORM_AFFINE`` or a polynomial for images from
different sensors or orbits. The inlier tie points are written to
``out_gcp_file`` so they can be inspected in a GIS. If an image cannot be
registered (usually because it is too cloudy to find tie points), the
example keeps the unregistered image and prints a message.

The match is measured before and after registration with
:func:`rsgispy.imagecalc.comparison.calc_img_similarity_metrics` (here the
correlation of the NIR band, ignoring no data).

.. literalinclude:: ../examples/analysis_ready_data.py
   :language: python
   :start-after: # [step-4-coregister]
   :end-before: # [end-step-4-coregister]
   :dedent: 4

Typical output for the synthetic date that was offset by 15 m east and 10 m
south:

.. code-block:: text

   {'n_tie_points': 6, 'n_inliers': 6, 'n_outliers': 0, 'rmse': 0.0,
    'max_residual': 0.0, 'transform_type': 'shift', ...,
    'corr_before': 0.887, 'corr_after': 1.0}

**Check:** the correlation should increase (or stay the same for a date that
was already aligned), the RMSE of the inliers should be well below a pixel
(10 m), and most tie points should be inliers. A large number of outliers or
an RMSE of several pixels points to poor matches (clouds, snow or land
change); increase ``win_size`` or use a different band. The example's
defaults (``win_size=64``, ``pixel_gap=250``) give about 1900 candidate tie
points on a full Sentinel-2 tile, which is plenty for a translation; a smaller
gap gives more points but takes proportionally longer (the test uses
``win_size=32`` and ``pixel_gap=16`` on its 96 × 96 pixel images). The
:doc:`image co-registration guide
<image_coregistration>` explains the tie point methods, the geometric models
and how to assess the result in detail.

Step 5 — Spectral indices
-------------------------

:func:`rsgispy.imagecalc.calcindices.calc_indices` calculates any of the
indices in its registry (NDVI, NDWI, NBR, EVI, red-edge indices, …) in one
pass. ``band_names`` maps the stack's band numbers to the canonical band
names (note ``re1``–``re3`` for the red-edge bands and ``nir8a`` for B8A),
``in_gain=10000`` converts the stored values to reflectance, input no data
(0, including the masked clouds) is ignored and written as ``-999``. Each
output band is named after its index.

Indices are calculated here for each date so that their distribution over
time can be summarised in step 6. Calculating an index from a reflectance
composite gives a different result from compositing the index (the median of
the NDVI values is not the NDVI of the median reflectances); both are valid,
but use one consistently.

.. literalinclude:: ../examples/analysis_ready_data.py
   :language: python
   :start-after: # [step-5-indices]
   :end-before: # [end-step-5-indices]
   :dedent: 4

Step 6 — Compositing
--------------------

Two percentile functions in :mod:`rsgispy.imagecalc.percentiles` build the
composites; both ignore no data values, so each pixel is summarised from its
clear observations only:

* :func:`~rsgispy.imagecalc.percentiles.calc_img_band_pxl_percentiles`
  calculates the percentiles **band by band** across the input images. With
  ``[50]`` the output has one band per input band: the median reflectance
  composite. The median is robust to the occasional cloud or shadow that the
  mask missed. Each band is summarised independently, so a composite pixel's
  spectrum can mix dates; use a narrow date range if that matters.
* :func:`~rsgispy.imagecalc.percentiles.calc_imgs_pxl_percentiles` pools
  **all bands of all inputs** for each pixel and writes one band per
  percentile. Given the single-band NDVI images it gives the 10th, 50th and
  90th percentiles of NDVI over the period — useful classification features
  that describe seasonality (a crop field and a forest can have the same
  median NDVI but very different 10th percentiles).

The outputs take the data type of the first input. The percentile
functions set the no data value, calculate statistics and overview pyramids
(``calc_stats=True``), and name the output bands after the percentile
(``p10``, ...) and, for
:func:`~rsgispy.imagecalc.percentiles.calc_img_band_pxl_percentiles`, the
input band (``Blue_p50``, ...). The example renames them with
:func:`rsgispy.imageutils.metadata.set_band_names`, to the input band names
for the median composite and to ``NDVI_P10``, ... for the NDVI percentiles.

The number of clear observations per pixel is calculated with
:func:`rsgispy.imagecalc.maths.band_maths` (the expression
``(d0 > 0) + (d1 > 0) + …`` counts the dates with data). It is the most useful
quality layer of a composite: pixels with zero observations are no data in
the composite, and pixels with one or two observations are the most likely to
contain undetected cloud.

.. literalinclude:: ../examples/analysis_ready_data.py
   :language: python
   :start-after: # [step-6-composite]
   :end-before: # [end-step-6-composite]
   :dedent: 4

**Check:** view the median composite as an RGB image (bands 3, 2, 1) next to
the ``n_clear_obs`` image. Remaining bright or dark blobs coincide with low
observation counts; extend the date range or improve the cloud mask there.
Reflectance values should be in the same range as the individual dates.

Step 7 — Mosaicking tiles
-------------------------

Sentinel-2 tiles overlap by about 10 km, and an area of interest often covers
several tiles. Two ways of combining the per-tile composites:

* :func:`rsgispy.imageutils.vrt.create_mosaic_images_vrt` writes a GDAL
  virtual raster (VRT): a small XML file that refers to the tile images and
  takes no time or disk space to create. It is the best choice while you are
  still working with the data, and every rsgispy function can read it.
* :func:`rsgispy.imageutils.mosaic.create_img_mosaic` writes a new image.
  ``skip_val`` (with ``skip_band``) marks no data pixels that should not
  overwrite data from another tile, and ``overlap_behaviour`` decides what
  happens where tiles overlap (``0``: the later image wins; ``1``/``2``: keep
  the minimum/maximum of ``skip_band``).

The final mosaics are written as GeoTIFFs, which rsgispy creates tiled and
DEFLATE-compressed by default. :mod:`rsgispy.imageutils.envopts` changes the
creation options used by all rsgispy functions in the same Python process:
for example :func:`~rsgispy.imageutils.envopts.set_env_vars_zstd_gtiff_outs`
for faster ZSTD compression where it is supported, or
:func:`~rsgispy.imageutils.envopts.set_img_creation_opts` for any other
options, e.g. ``{"COMPRESS": "DEFLATE", "PREDICTOR": "2"}``. Band names are
copied from the composites, and statistics and overviews are calculated
(``calc_stats=True``, the default).

To publish a final product as a Cloud Optimised GeoTIFF (COG), pass
``gdalformat="COG"`` instead (see :doc:`large_rasters` for how rsgispy writes
COGs).

.. literalinclude:: ../examples/analysis_ready_data.py
   :language: python
   :start-after: # [step-7-mosaic]
   :end-before: # [end-step-7-mosaic]
   :dedent: 4

All the images being mosaicked must have the same CRS, resolution and band
count. Tiles in different UTM zones (e.g. ``T30…`` and ``T31…``) must first be
reprojected to one CRS (see "Common problems").

Step 8 — Checking the outputs
-----------------------------

:func:`rsgispy.tools.checkdatasets.check_gdal_image_file` opens a file and
checks that it has the expected number of bands, a projection (and, with
``epsg_code``, the expected EPSG code) and, with ``read_img=True``, that
randomly chosen pixels can be decoded. It returns ``(file_ok, error_message)``.
Run it on every final product, particularly after long batch runs where a
process may have been killed part way through writing a file.

.. literalinclude:: ../examples/analysis_ready_data.py
   :language: python
   :start-after: # [step-8-check]
   :end-before: # [end-step-8-check]
   :dedent: 4

For many files,
:func:`~rsgispy.tools.checkdatasets.run_check_gdal_image_files` checks a list
of files in separate processes with a time-out (so a file that hangs the
reader cannot stop the check) and can delete the files that fail
(``rm_err=True``) so they are recreated by the next run; call it inside an
``if __name__ == "__main__":`` block.
:func:`~rsgispy.tools.checkdatasets.cmp_to_ref_imgs` checks that a set of
outputs has the same size, extent and EPSG code as a set of reference
images.

Performance and large datasets
------------------------------

* **Scale** — a Sentinel-2 tile is 10980 × 10980 pixels at 10 m (about 120
  million pixels per band); a 10-band 32-bit stack is about 4.8 GB
  uncompressed. Use ``out_res_10m=False`` (20 m) when 10 m detail is not
  needed, and delete the intermediate files (``_sref``, ``_clrsky``) once the
  composite is made.
* **Parallel processing** — stacking, masking and co-registration are
  independent for each scene, so run the scenes in parallel (one process per
  scene, e.g. with :mod:`multiprocessing` or a batch queue) rather than in one
  loop. ``use_multi_core=True`` in
  :func:`~rsgispy.imagecalibration.sensorlvl2data.create_stacked_sen2_esa_l2a_img`
  uses several threads for the resampling.
* **Compositing time** — the percentile functions read and process the
  images block by block, so memory use is small, and the per-pixel
  percentiles are vectorised (about 1 s for 10 dates of 2048 × 2048 pixels
  and five percentiles on an Apple M2 Max). For a large area, composite each
  tile separately (in parallel).
* **CloudSEN12** — use a GPU (``device='cuda'`` or ``'mps'``). ``tile_size``
  (default 512) sets the size of the blocks passed to the model; larger tiles
  are faster on a GPU with enough memory. ``tile_overlap`` (default 32)
  avoids edge effects between blocks.
* **VRTs instead of copies** — mosaics, band subsets
  (:func:`rsgispy.imageutils.vrt.create_vrt_band_subset`) and stacks
  (:func:`rsgispy.imageutils.vrt.create_stack_images_vrt`) can be VRTs until
  the final product is written.
* **Output formats and compression** — write intermediate files as GeoTIFFs
  (tiled and DEFLATE-compressed by default), and use COGs
  (``gdalformat="COG"``) for final products that are shared or served.
  ZSTD compresses and decompresses faster than Deflate where GDAL supports
  it. For outputs larger than 4 GB, pass
  ``bigtiff=True`` to the ``envopts`` functions.
* **Statistics and pyramids** — :func:`~rsgispy.imageutils.stats.pop_img_stats`
  with ``use_multi_core_pyramids=True`` and ``n_cores`` builds the overviews
  of large images in parallel.

Common problems
---------------

* **The search returns no scenes** — check the bounding box order
  ``(x_min, x_max, y_min, y_max)`` in WGS84 degrees, that both dates are
  given, and that ``cloud_cover`` is not too strict.
* **"username must be provided or set via the RSGIS_COP_USER environment
  variable"** — the credentials are not set in the environment of the Python
  process (e.g. a batch job or IDE that does not read your shell profile).
* **A downloaded zip fails to extract** — the download was interrupted.
  Delete the file and run the download again (complete downloads are
  verified with their MD5 checksum and kept).
* **"Missing Sentinel-2 band file(s)"** — the ``.SAFE`` folder is incomplete,
  or ``inc_cloudsen12_img=True`` was used with a product that lacks B01 or
  B09. Re-download the product.
* **CloudSEN12 errors** — "requires 12 bands" means the 10-band ``_sref``
  stack was passed instead of the ``_cloudsen12_sref`` stack (or use
  ``img_bands``). A missing ``torch`` or ``segmentation_models_pytorch``
  means the ``cloudmask`` extra is not installed. On machines without
  internet access, copy the weights into ``weights_dir`` and set
  ``download_weights=False``.
* **No data handling** — the Sentinel-2 stacks use 0 as no data and the
  Landsat stacks −9999; the masking, percentile and mosaic steps must be given
  the matching value. A composite with black (zero) holes means some pixels
  had no clear observation (check ``n_clear_obs``); a composite with
  unexpectedly low values usually means the no data value was not ignored.
* **Co-registration fails or makes things worse** — too few tie points
  (very cloudy dates, snow, water bodies) or land change between the dates.
  Use the clearest scene as the reference, a band with texture (NIR or SWIR),
  a larger ``win_size``, or skip co-registration for dates that are already
  aligned (correlation before ≈ after).
* **CRS and resolution mismatches** — the percentile, masking and mosaic
  functions work on the intersection of the inputs and expect the same grid.
  Tiles in different UTM zones must be reprojected to one CRS first, e.g. with
  :func:`rsgispy.imageutils.warp.reproject_image`, and images from another
  source put on the same grid with
  :func:`rsgispy.imageutils.warp.resample_img_to_match` (nearest neighbour for
  masks and classes, bilinear or cubic for reflectance).
* **Memory** — the functions in this guide process images in blocks, so
  memory is rarely a problem (``gdalformat='COG'`` outputs are also written
  block by block, through a temporary GeoTIFF); disk space for the
  intermediate files usually is.

Related guides
--------------

* :doc:`image_coregistration`
* :doc:`pixel_classification_workflow`
* :doc:`change_detection`
* :doc:`large_rasters`
