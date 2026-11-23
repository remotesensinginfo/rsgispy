rsgispy.elevation.hydrology
===========================

The :mod:`rsgispy.elevation.hydrology` module provides a pure-Python hydrological
analysis suite covering DEM conditioning, flow routing and accumulation, stream
network extraction and ordering, watershed and subcatchment delineation,
Height Above Nearest Drainage (HAND), and terrain wetness/erosivity indices.

.. important::

   All functions in :mod:`rsgispy.elevation.hydrology` require input rasters in
   a **projected coordinate reference system with linear units in metres** (for
   example, UTM). Geographic (latitude/longitude) rasters raise
   :class:`~rsgispy.RSGISPyException` and should be reprojected beforehand with
   :func:`rsgispy.imageutils.warp.reproject_image`.

End-to-End DEM Hydrology Workflow
---------------------------------

A typical catchment analysis starts from a raw projected DEM and derives:

1. A **hydrologically conditioned DEM** free of spurious sinks and closed pits.
2. **Flow direction** and **flow accumulation** grids.
3. A raster and vector **stream network** with Strahler/Shreve stream orders.
4. **Basins, gauged watersheds, and reach subcatchments** (raster and vector polygons).
5. **HAND** (Height Above Nearest Drainage) and optional static flood depth.
6. **Topographic Wetness Index (TWI)** from specific catchment area and terrain slope.

.. code-block:: text

   Raw Projected DEM (metres)
          │
          ▼
    condition_dem() ──────────────────────────────────────────────┐
          │                                                       │
          ├──► flow_direction("d8") ──► flow_accumulation()       │
          │          │                        │                   │
          │          │                        ▼                   │
          │          │                 extract_streams()          │
          │          │                        │                   │
          │          ├────────────────────────┼───────────────────┤
          │          │                        ▼                   │
          │          ├──► stream_link() / stream_order()          │
          │          ├──► vectorise_streams()                     │
          │          ├──► delineate_basins() / watersheds()       │
          │          │         └──► polygonise_watersheds()       │
          │          └──► calc_hand()                             │
          │                                                       │
          └──► terrain.slope() + specific catchment area ──► calc_twi()

Step 1: Condition the DEM and Compute Flow Grids
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use :func:`condition_dem` (combining optional stream burning, Lindsay 2016
least-cost depression breaching, and Barnes et al. 2014 Priority-Flood filling)
followed by :func:`flow_direction` and :func:`flow_accumulation`.

.. code-block:: python

   import os
   import rsgispy.elevation.hydrology as hydro

   os.makedirs("hydro_out", exist_ok=True)
   raw_dem = "dem_utm.tif"

   # 1. Breach minor dams/road embankments and fill remaining closed sinks
   cond_dem = "hydro_out/dem_cond.tif"
   hydro.condition_dem(
       input_img=raw_dem,
       output_img=cond_dem,
       method="breach_and_fill",
       max_breach_depth=5.0,
       max_breach_length=25,
   )

   # 2. D8 flow direction (with Barnes 2014 flat resolution) and accumulation
   fdir_d8 = "hydro_out/flow_dir_d8.tif"
   facc_cells = "hydro_out/flow_acc_cells.tif"
   hydro.flow_direction(cond_dem, fdir_d8, method="d8", flat_routing=True)
   hydro.flow_accumulation(fdir_d8, facc_cells, out_units="cells")

.. note::

   If you already have a mapped hydrography vector layer, pass
   ``streams_vec="mapped_rivers.gpkg"`` to :func:`condition_dem` (or call
   :func:`burn_streams` directly) to enforce known channels before routing.

Step 2: Extract, Order, and Vectorise Streams
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Threshold the flow accumulation grid with :func:`extract_streams` (either in
pixel counts with ``threshold_units="cells"`` or drainage area in square metres
with ``threshold_units="area"``), assign unique reach IDs and Strahler orders,
and export a topologically linked vector network with :func:`vectorise_streams`.

.. code-block:: python

   streams_bin = "hydro_out/streams.tif"
   stream_links = "hydro_out/stream_links.tif"
   stream_strahler = "hydro_out/stream_strahler.tif"
   streams_gpkg = "hydro_out/streams.gpkg"

   # Extract channels draining at least 50,000 m^2 (5 ha)
   hydro.extract_streams(
       input_img=facc_cells,
       output_img=streams_bin,
       threshold=50000.0,
       threshold_units="area",
   )

   # Optional raster reach IDs and Strahler stream orders
   hydro.stream_link(streams_bin, fdir_d8, stream_links)
   hydro.stream_order(streams_bin, fdir_d8, stream_strahler, method="strahler")

   # Vectorise into topologically connected reaches with DEM & accumulation stats
   hydro.vectorise_streams(
       streams_img=streams_bin,
       flow_dir_img=fdir_d8,
       output_vec=streams_gpkg,
       dem_img=cond_dem,
       flow_acc_img=facc_cells,
       order_method="strahler",
   )

The output ``streams.gpkg`` contains ``link_id``, ``down_link_id``, ``order``,
``length_m``, ``elev_start``, ``elev_end``, ``drop_m``, ``slope``, and
``max_acc`` attributes, with upstream endpoints snapped directly to downstream
junction coordinates.

Step 3: Delineate Basins, Gauged Watersheds, and Subcatchments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can partition the entire domain into drainage basins using
:func:`delineate_basins`, delineate reach-level lateral subcatchments from
``stream_links.tif`` using :func:`delineate_subcatchments`, or delineate custom
catchments upstream of gauge/pour-point coordinates (after snapping them to the
highest-accumulation stream cell within ``snap_dist`` metres via
:func:`snap_pour_points`).

.. code-block:: python

   # Option A: All complete drainage basins draining to domain outlets
   basins_tif = "hydro_out/basins.tif"
   basins_gpkg = "hydro_out/basins.gpkg"
   hydro.delineate_basins(fdir_d8, basins_tif)
   hydro.polygonise_watersheds(basins_tif, basins_gpkg, dem_img=cond_dem)

   # Option B: Reach-level lateral subcatchments for every link in stream_links
   subcatchments_tif = "hydro_out/subcatchments.tif"
   subcatchments_gpkg = "hydro_out/subcatchments.gpkg"
   hydro.delineate_subcatchments(fdir_d8, stream_links, subcatchments_tif)
   hydro.polygonise_watersheds(
       subcatchments_tif, subcatchments_gpkg, dem_img=cond_dem
   )

   # Option C: Delineate watersheds for specific gauge/pour points
   pour_pts_snapped = "hydro_out/pour_points_snapped.gpkg"
   watersheds_tif = "hydro_out/watersheds.tif"
   watersheds_gpkg = "hydro_out/watersheds.gpkg"

   hydro.snap_pour_points(
       pour_points_vec="gauges.gpkg",
       flow_acc_img=facc_cells,
       output_vec=pour_pts_snapped,
       snap_dist=50.0,
   )
   hydro.delineate_watersheds(
       flow_dir_img=fdir_d8,
       pour_points_vec=pour_pts_snapped,
       output_img=watersheds_tif,
       id_field="gauge_id",
   )
   hydro.polygonise_watersheds(watersheds_tif, watersheds_gpkg, dem_img=cond_dem)

Passing ``dem_img=cond_dem`` to :func:`polygonise_watersheds` populates
``area_m2``, ``perimeter_m``, ``elev_min``, ``elev_mean``, and ``elev_max``
attributes on each dissolved polygon.

Step 4: Calculate HAND and Flood Inundation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`calc_hand` traces every hillslope pixel down the D8 drainage network to
its nearest stream cell and computes the vertical elevation difference
(:math:`Z_{\text{cell}} - Z_{\text{drainage}}`). The resulting HAND surface can
be passed directly to :func:`flood_inundation_from_hand` to estimate static
inundation depth for a given stage height.

.. code-block:: python

   hand_tif = "hydro_out/hand.tif"
   flood_2m_tif = "hydro_out/flood_depth_2m.tif"

   hydro.calc_hand(
       dem_img=cond_dem,
       flow_dir_img=fdir_d8,
       streams_img=streams_bin,
       output_img=hand_tif,
   )
   hydro.flood_inundation_from_hand(
       input_img=hand_tif,
       output_img=flood_2m_tif,
       water_depth=2.0,
   )

Step 5: Calculate Topographic Wetness Index (TWI)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Topographic Wetness Index is defined as
:math:`\text{TWI} = \ln(a / \tan\beta)`, where :math:`a` is specific catchment
area (:math:`\text{m}^2\,\text{m}^{-1}`, upslope contributing area per unit
contour length) and :math:`\beta` is local terrain slope.

For smooth hillslope wetness patterns, D-Infinity (:func:`flow_direction` with
``method="dinf"``) or MFD (``method="mfd"``) is generally preferred over D8,
paired with :func:`rsgispy.elevation.terrain.slope` and
:func:`rsgispy.imagecalc.maths.image_band_math` to convert contributing area
(:math:`\text{m}^2`) to specific catchment area (:math:`a = A / \Delta x`):

.. code-block:: python

   import rsgispy
   from rsgispy.elevation import terrain
   from rsgispy.imagecalc.maths import image_band_math
   from rsgispy.imageutils.metadata import get_img_res

   fdir_dinf = "hydro_out/flow_dir_dinf.tif"
   facc_area = "hydro_out/flow_acc_dinf_area.tif"
   sca_tif = "hydro_out/sca_dinf.tif"
   slope_tif = "hydro_out/slope_deg.tif"
   twi_tif = "hydro_out/twi.tif"

   # 1. Compute D-Infinity flow accumulation in m^2
   hydro.flow_direction(cond_dem, fdir_dinf, method="dinf")
   hydro.flow_accumulation(fdir_dinf, facc_area, out_units="area")

   # 2. Convert contributing area (m^2) to specific catchment area (m^2 / m)
   x_res, _ = get_img_res(cond_dem, abs_vals=True)
   image_band_math(
       input_img=facc_area,
       output_img=sca_tif,
       exp=f"b1 / {x_res}",
       datatype=rsgispy.TYPE_32FLOAT,
   )

   # 3. Calculate terrain slope in degrees and TWI
   terrain.slope(cond_dem, slope_tif, unit="degrees")
   hydro.calc_twi(
       specific_catchment_area_img=sca_tif,
       slope_img=slope_tif,
       output_img=twi_tif,
       slope_units="degrees",
       min_slope=0.001,
   )

.. tip::

   The same ``sca_tif`` and ``slope_tif`` inputs can also be passed directly to
   :func:`calc_spi` (Stream Power Index) and :func:`calc_ls_factor` (RUSLE
   slope-length factor).

API Reference
-------------

.. automodule:: rsgispy.elevation.hydrology
   :members:
   :undoc-members:
   :show-inheritance:
