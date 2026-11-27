# DEM Elevation, Hydrology, Segmentation, Spatial Filters & Morphology

Covers `rsgispy.elevation.*`, `rsgispy.segmentation.*`, `rsgispy.imagefilter.*`, and `rsgispy.imagemorphology` in `rsgispy` v0.4.0.

---

## 1. `rsgispy.elevation.*`

> Note: Functions in `elevation.{terrain,illumination,filtering,fusion,hydrology}` accept keyword-only `*, gdalformat='GTIFF', datatype=None, no_data_val=None, calc_stats=True, use_pbar=True` (no `creation_opts`).

### Terrain Derivatives (`rsgispy.elevation.terrain`)
- `slope(input_img, output_img, unit='degrees', *, ...)` (`unit`: `'degrees'` or `'radians'`)
- `slope_pxl_res_img(in_dem_img, in_pxl_res_img, output_img, unit='degrees', *, ...)` (for geographic / varying pixel-size DEMs)
- `aspect(input_img, output_img, *, ...)`
- `aspect_pxl_res_img(in_dem_img, in_pxl_res_img, output_img, *, ...)`
- `categorise_aspect(input_img, output_img, *, ...)`

### Solar Illumination (`rsgispy.elevation.illumination`)
- `hillshade(input_img, output_img, azimuth, zenith, *, ...)`
- `hillshade_pxl_res_img(in_dem_img, in_pxl_res_img, output_img, azimuth, zenith, *, ...)`
- `shadow_mask(input_img, output_img, azimuth, zenith, max_height, *, ...)`
- `local_incidence_angle(input_img, output_img, azimuth, zenith, *, ...)`
- `local_exitance_angle(input_img, output_img, azimuth, zenith, *, ...)`

### Chaining terrain in a tile graph (node builders)
Each terrain/illumination function above (except `shadow_mask`) has a node builder for an in-memory `TileGraph` (one pass, no intermediate files; same output as the file function, which is built on it). Signature `builder(dem_node, <params>, *, no_data_val=None, name=None)`; `no_data_val=None` uses the **input node's** no-data value (unlike the filter builders):
- `terrain.slope_node(dem, unit='degrees')`, `terrain.aspect_node(dem)`, `terrain.categorise_aspect_node(aspect_node)` (pixel-wise, `uint8`, no nodata)
- `terrain.slope_pxl_res_img_node(dem, pxl_res, unit='degrees')`, `terrain.aspect_pxl_res_img_node(dem, pxl_res)` (`pxl_res`: node with EW/NS resolution in bands 1/2, same grid as the DEM)
- `illumination.hillshade_node(dem, azimuth, zenith)` (`uint8`), `illumination.hillshade_pxl_res_img_node(dem, pxl_res, azimuth, zenith)`
- `illumination.local_incidence_angle_node(dem, azimuth, zenith)`, `illumination.local_exitance_angle_node(dem, azimuth, zenith)`

The DEM node must be single-band (`RasterSource("dem.tif", bands=1)` or `SelectBands`). Horn 3x3 nodes: halo radius 1, edge padding, pixel size from the DEM geotransform (an `ArraySource` needs a `transform`). Slope/aspect/angle nodes output `float64` (write with `DiskSink(..., datatype=rsgispy.TYPE_32FLOAT)` to match the file functions); hillshade writes as `rsgispy.TYPE_8UINT`. `shadow_mask` is not chainable (rays cross the whole DEM).
```python
import rsgispy
from rsgispy.imageprocess import BandMathOp, DiskSink, RasterSource, TileGraph
from rsgispy.imagefilter.stats import mean_filter_node
from rsgispy.elevation.terrain import slope_node

dem = RasterSource("dem.tif", bands=1)
slp = slope_node(mean_filter_node(dem, 5, no_data_val=-9999), "degrees")
classes = BandMathOp({"s": slp}, "(s >= 5) + (s >= 15) + (s >= 30)")
TileGraph(DiskSink(classes, "slope_classes.tif", datatype=rsgispy.TYPE_8UINT)).run()
```

### Filtering & Fusion (`rsgispy.elevation.{filtering,fusion}`)
- `filtering.dtm_aspect_median_filter(in_dem_img, in_aspect_img, output_img, aspect_range, win_h_size, *, ...)`
- `filtering.fill_dem_soille_gratin_1994(in_dem_img, in_vld_img, output_img, *, ...)`
- `filtering.plane_fit_detrend_dem(input_img, output_img, win_size, *, ...)`
- `filtering.resampling_detrend_dem(input_img, output_img, resmp_res, low_resmp_mthd=5, up_smpl_mthd=3, tmp_dir=None, *, ...)`
- `fusion.fft_dem_fusion(high_res_dem_img, low_res_dem_img, fused_dem_img, c_size=20, *, ...)`

### Surface Roughness Profiles (`rsgispy.elevation.roughness`)
Follows dual path / `GeoDataFrame` pattern (`out_vec_file=None` returns `GeoDataFrame`). Works with point/transect sample boxes (`calc_pt_win_smpls`):
- Classes: `CalcProfileRoughMetrics(detrend=True, detrend_poly_order=1, create_plots=False, out_plot_path=None)`, `CalcMunroRoughnessMetric(...)`, `CalcSmithRoughnessMetric(detrend=True, ...)`
- Functions: `calc_simple_roughness_profile_metrics(...)`, `calc_munro_roughness_profile_metrics(...)`, `calc_smith_roughness_metrics(...)`, `calc_all_roughness_profile_metrics(...)`

### Hydrology (`rsgispy.elevation.hydrology`)
> **Numba behaviour**: Graph/priority-queue hydrology functions default to `allow_pure_python=False`, which requires `numba` (`HAS_NUMBA = True`) and raises `RSGISPyException` if `numba` is not installed unless `allow_pure_python=True` is passed explicitly.

- **Sink & DEM conditioning**:
  - `identify_sinks(input_img, output_img, output_depth_img=None, allow_pure_python=False, *, ...)`
  - `fill_depressions(input_img, output_img, resolve_flats=True, allow_pure_python=False, *, ...)`
  - `breach_depressions(input_img, output_img, max_depth=None, max_length=None, fill_remaining=True, allow_pure_python=False, *, ...)` (least total cut from each pit to lower ground or to the DEM edge / no data; without `max_depth` / `max_length` every pit is breached however deep, so set them for real DEMs or use `condition_dem`)
  - `burn_streams(dem_img, streams_vec, output_img, burn_depth=5.0, buffer_dist=None, smooth_drop=2.0, *, ...)`
  - `resolve_flats(input_img, output_img, allow_pure_python=False, *, ...)`
  - `condition_dem(input_img, output_img, method='breach_and_fill', max_breach_depth=None, max_breach_length=None, streams_vec=None, burn_depth=5.0, buffer_dist=None, allow_pure_python=False, *, ...)`
- **Flow routing**:
  - `flow_direction(input_img, output_img, method='d8', flat_routing=True, mfd_exponent=1.1, allow_pure_python=False, *, ...)` (`method`: `'d8'`, `'dinf'`, `'mfd'`)
  - `flow_accumulation(input_img, output_img, weight_img=None, out_units='cells', allow_pure_python=False, *, ...)`
  - `flow_length(input_img, output_img, direction='downstream', weight_img=None, allow_pure_python=False, *, ...)`
  - `flow_distance(flow_dir_img, streams_img, output_img, dem_img=None, distance_type='horizontal', allow_pure_python=False, *, ...)`
  - `trace_flow_path(flow_dir_img, pour_points_vec, out_vec_file, direction='downstream', flow_acc_img=None, *, out_format=None, out_vec_lyr=None, overwrite=True, no_data_val=None, use_pbar=True)`
- **Streams & watersheds**:
  - `extract_streams(input_img, output_img, threshold, threshold_units='cells', *, acc_units='auto', ...)` — `flow_accumulation` writes the tag `RSGISPY_FLOW_ACC_UNITS` (`cells`/`area`); `acc_units='auto'` reads it (untagged = cells) and converts the threshold, so cells and m² accumulations give the same network; a contradicting `acc_units` raises. `calc_twi`/`calc_spi`/`calc_ls_factor` raise if given a tagged flow accumulation (they need specific catchment area, m²/m).
  - `stream_link(streams_img, flow_dir_img, output_img, allow_pure_python=False, *, ...)`
  - `stream_order(streams_img, flow_dir_img, output_img, method='strahler', allow_pure_python=False, *, ...)`
  - `vectorise_streams(streams_img, flow_dir_img, out_vec_file, dem_img=None, flow_acc_img=None, order_method='strahler', allow_pure_python=False, *, ...)`
  - `snap_pour_points(pour_points_vec, flow_acc_img, out_vec_file, snap_dist, streams_img=None, *, ...)`
  - `delineate_watersheds(flow_dir_img, pour_points_vec, output_img, id_field=None, allow_pure_python=False, *, ...)`
  - `delineate_basins(input_img, output_img, allow_pure_python=False, *, ...)`
  - `delineate_subcatchments(flow_dir_img, stream_link_img, output_img, allow_pure_python=False, *, ...)`
  - `polygonise_watersheds(watershed_img, out_vec_file, dem_img=None, dissolve=True, simplify_tol=None, *, ...)`
- **Flood & terrain indices**:
  - `calc_hand(dem_img, flow_dir_img, streams_img, output_img, allow_pure_python=False, *, ...)` (Height Above Nearest Drainage)
  - `flood_inundation_from_hand(input_img, output_img, water_depth, *, ...)`
  - `calc_twi(specific_catchment_area_img, slope_img, output_img, slope_units='degrees', min_slope=0.001, *, ...)`
  - `calc_spi(specific_catchment_area_img, slope_img, output_img, slope_units='degrees', *, ...)`
  - `calc_ls_factor(specific_catchment_area_img, slope_img, output_img, slope_units='degrees', m=0.4, n=1.3, *, ...)`

---

## 2. `rsgispy.segmentation.*`

All major algorithms provide both file-based (`*_segmentation`) and NumPy array (`*_segmentation_array`) entrypoints with optional tiled + multi-core execution (`tile_size`, `overlap`, `n_jobs`, `stitch_seams`) and `use_numba=True` (use `is_numba_available()` / `HAVE_NUMBA` to inspect; emits warning and falls back if Numba is unavailable).

- **Algorithms** (top-level `rsgispy.segmentation` and dedicated submodules):
  - `shepherd.shepherd_segmentation(input_img, output_img, n_clusters=60, min_pixels=100, dist_thres=None, sampling=100, max_iter=50, bands=None, tile_size=None, overlap=32, n_jobs=1, stitch_seams=True, rnd_seed=42, use_numba=True, *, gdalformat='GTIFF', datatype=None, no_data_val=None, calc_stats=True, use_pbar=True)` + `shepherd_segmentation_array(...)`
  - `region_merge.region_merge_segmentation(..., scale=25.0, compactness=0.5, shape_weight=0.1, min_pixels=10, ...)` + `region_merge_segmentation_array(...)`
  - `slic.slic_segmentation(..., n_segments=500, segment_size=None, compactness=10.0, max_iter=10, sigma=0.0, enforce_connectivity=True, min_pixels=10, ...)` + `slic_segmentation_array(...)`
  - `graph.graph_segmentation(..., scale=100.0, sigma=0.8, min_pixels=50, ...)` + `graph_segmentation_array(...)`
  - `watershed.watershed_segmentation(..., markers_img=None, min_distance=10, gradient_sigma=1.0, compactness=0.0, min_pixels=10, ...)` + `watershed_segmentation_array(...)`
- **Clumping (`rsgispy.segmentation.clump`)**:
  - `clump(input_img, output_img, in_memory=False, block_size=None, *, ...)`
  - `union_of_clumps(input_imgs, output_img, in_memory=None, block_size=None, *, ...)`
- **Utilities & post-processing (`rsgispy.segmentation.utils` & top-level `rsgispy.segmentation`)**:
  - `relabel_segments(input_img, output_img, contiguous=True, *, ...)` + `relabel_segments_array(...)`
  - `eliminate_small_segments(input_img, output_img, ref_img=None, min_pixels=10, bands=None, use_numba=True, *, ...)` + `eliminate_small_segments_array(...)`
  - `polygonise_segments(input_img, out_vec_file=None, out_vec_lyr=None, out_format='GPKG', img_band=1, no_data_val=0, segment_id_field='seg_id', use_8_conn=False, out_compress=None, *, tile_size=2048, n_jobs=1, backend='auto', use_pbar=False) -> GeoDataFrame | None` — one feature per connected region, sorted by segment ID; uses `rsgispy.vector.create.polygonise_raster_tiled` (tiled, multi-core, streamed to a new output file). For one (Multi)Polygon per segment ID call `polygonise_raster_tiled(..., no_data_val=0, dissolve_by_value=True, sort_by_value=True)`.
  - `extract_segment_stats(segments_img, feature_img, out_vec_file=None, out_vec_lyr=None, out_format='GPKG', bands=None, min_thres=-1e12, max_thres=1e12, out_no_data_val=-9999.0, segment_id_field='seg_id', use_pbar=True, out_compress=None) -> GeoDataFrame | None` (uses the fast `zonalstats.labels.calc_label_zonal_stats` when the feature image is on the segment grid).
  - For object-based classification/regression with segments use `rsgispy.classification.classsegments` / `rsgispy.regression.regsegments` (see `ml_classification_regression_and_sdm.md`).
- **Tiling primitives (`rsgispy.segmentation.tiling`)**: `TileWindow`, `plan_tiles`, `normalise_tile_size`, `extract_halo_tile`, `crop_halo_labels`, `offset_tile_labels`, `reconcile_seam_equivalences`, `run_tiled_array_pipeline`, `run_tiled_raster_pipeline`.

---

## 3. `rsgispy.imagefilter.*`

> Standard keyword-only args: `*, gdalformat='GTIFF', datatype=None, no_data_val=None, calc_stats=True, use_pbar=True` (except `chained.apply_filter_chain` which also accepts `creation_opts=None`).

- **Window statistics (`stats`)**: `apply_median_filter`, `apply_mean_filter`, `apply_min_filter`, `apply_max_filter`, `apply_mode_filter`, `apply_stddev_filter`, `apply_range_filter`, `apply_total_filter`, `apply_mean_diff_filter`, `apply_mean_diff_abs_filter`, `apply_total_diff_filter`, `apply_total_diff_abs_filter`, `apply_coeff_of_var_filter`, `apply_norm_var_filter`, `apply_norm_var_sqrt_filter`, `apply_norm_var_ln_filter`, `apply_texture_var_filter` (default `filter_size=3`).
- **Smoothing (`smooth`)**: `apply_kuwahara_filter(..., filter_size=5)`, `apply_gaussian_smooth_filter(..., filter_size=3, stddev_x=1.0, stddev_y=1.0, filter_angle=0.0)`, `apply_kernel_filter(input_img, output_img, kernel, normalise=False, *, ...)`.
- **Edge detection (`edge`)**: `apply_sobel_filter`, `apply_sobel_x_filter`, `apply_sobel_y_filter`, `apply_prewitt_filter`, `apply_prewitt_x_filter`, `apply_prewitt_y_filter`, `apply_gaussian_1st_deriv_filter`, `apply_gaussian_2nd_deriv_filter`, `apply_laplacian_filter`.
- **SAR speckle (`sar`)**: `apply_lee_filter(..., filter_size=3, n_looks=1.0)`, `apply_enhanced_lee_filter(..., filter_size=5, n_looks=1.0, damping=1.0)`, `apply_frost_filter(..., filter_size=5, damping=1.0)`, `apply_enhanced_frost_filter(..., filter_size=5, n_looks=1.0, damping=1.0)`, `apply_kuan_filter(..., filter_size=5, n_looks=1.0)`, `apply_gamma_map_filter(..., filter_size=5, n_looks=1.0)`, `apply_refined_lee_filter(..., filter_size=7, n_looks=1.0)`.
- **Node builders (tile-graph chaining)**: every `apply_<name>_filter` has `<name>_filter_node(source, <same processing params as the apply_* function>, *, no_data_val=None, name=None) -> TileNode` in the same module (a `WindowOp`, `float64` blocks, same bands/band names as `source`, halo = half the window). The `apply_*` functions are wrappers running `RasterSource -> builder -> DiskSink`, so a builder is bit-identical to its function (`no_data_val=None` uses the source node's no-data value; `SetNoData(src, None)` disables it). File options (`gdalformat`, `datatype`, `calc_stats`) go on the `DiskSink`.
  - `stats`: `median_filter_node`, `mean_filter_node`, `min_filter_node`, `max_filter_node`, `mode_filter_node`, `stddev_filter_node`, `range_filter_node`, `total_filter_node`, `mean_diff_filter_node`, `mean_diff_abs_filter_node`, `total_diff_filter_node`, `total_diff_abs_filter_node`, `coeff_of_var_filter_node`, `norm_var_filter_node`, `norm_var_sqrt_filter_node`, `norm_var_ln_filter_node`, `texture_var_filter_node` (all `(source, filter_size=3, ...)`).
  - `smooth`: `kuwahara_filter_node(source, filter_size=5)`, `gaussian_smooth_filter_node(source, filter_size=3, stddev_x=1.0, stddev_y=1.0, filter_angle=0.0)`, `kernel_filter_node(source, kernel, normalise=False, *, zero_dc=None, ...)`.
  - `edge`: `sobel_filter_node(source)`, `sobel_x_filter_node(source)`, `sobel_y_filter_node(source)`, `prewitt_filter_node(source)`, `prewitt_x_filter_node(source)`, `prewitt_y_filter_node(source)`, `gaussian_1st_deriv_filter_node(source, filter_size=3, stddev_x=1.0, stddev_y=1.0, filter_angle=0.0)`, `gaussian_2nd_deriv_filter_node(...)` (same params), `laplacian_filter_node(source, filter_size=3, stddev=1.0)`.
  - `sar`: `lee_filter_node(source, filter_size=3, n_looks=1.0)`, `enhanced_lee_filter_node(source, filter_size=5, n_looks=1.0, damping=1.0)`, `frost_filter_node(source, filter_size=5, damping=1.0)`, `enhanced_frost_filter_node(source, filter_size=5, n_looks=1.0, damping=1.0)`, `kuan_filter_node(source, filter_size=5, n_looks=1.0)`, `gamma_map_filter_node(source, filter_size=5, n_looks=1.0)`, `refined_lee_filter_node(source, filter_size=7, n_looks=1.0)`.
  - `texture`: `glcm_filter_node(source, properties=GLCM_ALL_PROPERTIES, filter_size=5, distances=(1,), angles=(0.0, 45.0, 90.0, 135.0), n_levels=32, min_val=None, max_val=None, symmetric=True, average_angles=True, *, band_bounds=None, ...)`. The grey-level range must be known before the graph runs: pass `min_val`+`max_val` (all bands) or `band_bounds=[(min, max), ...]` (one per band), else `RSGISPyException`. `calc_glcm_band_bounds(input_img, no_data_val=None, use_pbar=True)` gives the per-band ranges `apply_glcm_filter` uses by default. Output: `len(properties)` bands per input band (x offsets if `average_angles=False`), named `<band>_glcm_<property>`.

```python
from rsgispy.imageprocess import BandMathOp, DiskSink, RasterSource, TileGraph
from rsgispy.imagefilter.sar import lee_filter_node
from rsgispy.imagefilter.stats import median_filter_node

sar = RasterSource("sar_linear.tif", nodata=0.0)
lee = lee_filter_node(sar, filter_size=5, n_looks=4, no_data_val=0.0)
db = BandMathOp({"x": lee}, "x > 0 ? 10 * log10(x) : -9999", nodata=-9999)
smooth = median_filter_node(db, 3, no_data_val=-9999)
TileGraph(DiskSink(smooth, "sar_lee_db_med.tif")).run()  # one pass, no temp files

# Refined Lee -> GLCM texture with the ranges apply_glcm_filter would use
import rsgispy.imagefilter.texture as tex
from rsgispy.imagefilter.sar import refined_lee_filter_node

rlee = refined_lee_filter_node(RasterSource("sar_linear.tif"), 7, n_looks=4, no_data_val=0.0)
glcm = tex.glcm_filter_node(rlee, [tex.GLCM_CONTRAST, tex.GLCM_ENTROPY], min_val=0.0, max_val=0.5, no_data_val=0.0)
TileGraph(DiskSink(glcm, "sar_rlee_glcm.tif")).run()
```
- **GLCM texture (`texture`; requires `imaging` / `scikit-image`)**:
  - Constants: `GLCM_CONTRAST='contrast'`, `GLCM_DISSIMILARITY='dissimilarity'`, `GLCM_HOMOGENEITY='homogeneity'`, `GLCM_ASM='asm'`, `GLCM_ENERGY='energy'`, `GLCM_CORRELATION='correlation'`, `GLCM_ENTROPY='entropy'`, `GLCM_MEAN='mean'`, `GLCM_VARIANCE='variance'`, `GLCM_MAX_PROB='max_prob'`, `GLCM_ALL_PROPERTIES`.
  - `apply_glcm_filter(input_img, output_img, properties=GLCM_ALL_PROPERTIES, filter_size=5, distances=(1,), angles=(0.0, 45.0, 90.0, 135.0), n_levels=32, min_val=None, max_val=None, symmetric=True, average_angles=True, *, ...)`
- **In-memory fused chains (`chained`)**:
  - `FilterStep(kind: str, filter_size: int = 3, params: Dict[str, Any] = {}, name: Optional[str] = None)`
  - Supported `kind` values:
    - Any of the 18 window stat/kuwahara names (`'median'`, `'mean'`, `'min'`, `'max'`, `'mode'`, `'stddev'`, `'range'`, `'total'`, `'mean_diff'`, `'mean_diff_abs'`, `'total_diff'`, `'total_diff_abs'`, `'coeff_of_var'`, `'norm_var'`, `'norm_var_sqrt'`, `'norm_var_ln'`, `'variance'` (= `texture_var_filter_node`), `'kuwahara'`); each kind calls the matching node builder
    - SAR: `'lee'`, `'kuan'`, `'gamma_map'`, `'refined_lee'` (`params={'n_looks': 1.0}`), `'enhanced_lee'`, `'enhanced_frost'` (`params={'n_looks': 1.0, 'damping': 1.0}`), `'frost'` (`params={'damping': 1.0}`)
    - `'gaussian'`, `'gaussian_1st_deriv'`, `'gaussian_2nd_deriv'` (`params={'stddev': 1.0}` or `stddev_x`, `stddev_y`, `filter_angle`)
    - `'laplacian'` (`params={'stddev': 1.0}`)
    - `'sobel'`, `'prewitt'` (magnitudes), `'sobel_x'`, `'sobel_y'`, `'prewitt_x'`, `'prewitt_y'`
    - `'glcm'` (`params` = the `glcm_filter_node` args: `properties`, `distances`, `angles`, `n_levels`, `symmetric`, `average_angles`, and `min_val`+`max_val` or `band_bounds` (required); output band names come from the node)
    - `'kernel'` (`params={'kernel': array_2d, 'zero_dc': False}`)
    - `'band_math'` (`params={'expr': "..."}` — binds `'x'` to current step and any prior named step `step.name`)
    - `'pointwise'` (`params={'func': callable_arr_fn}`)
  - `apply_filter_chain(input_img, output_img, steps, block_size=None, *, gdalformat='GTIFF', datatype=None, no_data_val=None, calc_stats=True, use_pbar=True, creation_opts=None)`
  - `build_filter_chain_node(source, steps, *, no_data_val=None) -> TileNode` (composable inside `TileGraph`).

---

## 4. `rsgispy.imagemorphology`

- `create_circular_op(output_file, op_size)` (creates ASCII structuring element file for `use_op_file=True`; otherwise uses circular structuring element of `op_size`).
- **Basic operators**: `image_dilate`, `image_erode`, `image_gradient`
- **Combined-out variants**: `image_dilate_combined_out`, `image_erode_combined_out`, `image_gradient_combined_out`
- **Local minima**: `image_local_minima(..., sequencial_out=False, allow_equal=True, ...)`, `image_local_minima_combined_out(...)`
- **Compound operators**: `image_opening(..., niters=1)`, `image_closing(..., niters=1)`, `image_black_top_hat`, `image_white_top_hat`
- Common parameters: `(input_img, output_img, morph_op_file='', use_op_file=False, op_size=3, [tmp_img=None], *, gdalformat='GTIFF', datatype=None, no_data_val=None, calc_stats=True, use_pbar=True)`.
- **Tile-graph nodes** (chain with filters / band maths / `majority_filter_node` in one `TileGraph` pass; the file functions run these same nodes): `image_dilate_node`, `image_erode_node`, `image_gradient_node` `(source, morph_op_file='', use_op_file=False, op_size=3, *, combined=False, dtype=None, no_data_val=None, name=None)`; `image_local_minima_node(source, sequencial_out=False, allow_equal=True, morph_op_file='', use_op_file=False, op_size=3, *, combined=False, no_data_val=None)` (binary only: `sequencial_out=True` raises, numbering is not tile-local); `image_opening_node`, `image_closing_node` `(source, ..., op_size=3, niters=1, *, dtype=None, no_data_val=None)`; `image_white_top_hat_node`, `image_black_top_hat_node`. `no_data_val` is the input no-data (default: the source node's); pass `dtype=np.uint8` etc. when writing integers so opening/closing steps round as the file functions do.

```python
import numpy as np
import rsgispy
from rsgispy.imageprocess import DiskSink, RasterSource, TileGraph
from rsgispy.imagemorphology import image_closing_node, image_opening_node

mask = RasterSource("cloud_mask.tif")
clean = image_closing_node(image_opening_node(mask, op_size=3, dtype=np.uint8), op_size=5, dtype=np.uint8)
TileGraph(DiskSink(clean, "cloud_mask_clean.tif", datatype=rsgispy.TYPE_8UINT)).run()
```
