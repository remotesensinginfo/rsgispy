---
name: rsgispy
description: Generate idiomatic Python code using rsgispy (v0.3.x), the pure-Python replacement for RSGISLib built on rasterio, geopandas, shapely, pyproj, numpy, pandas, and pymu-eval. Use whenever the user asks to write, debug, or migrate remote sensing, Earth observation, GIS, raster processing, satellite radiometric calibration or CloudSEN12 cloud masking, image co-registration, spectral or SAR index calculation (calc_indices, calc_sar_indices), band_maths / vector_maths expressions, zonal statistics, DEM terrain derivatives or hydrology, spatial/SAR filtering, segmentation, spectral unmixing, vector geometry/attribute operations, cartographic mapping, data access, CLI tools, or ML classification/regression/SDM pipelines using rsgispy or migrating from RSGISLib.
---

# rsgispy Coding Guide

`rsgispy` (v0.3.1) is a **pure-Python** replacement for RSGISLib with **zero** `osgeo` or `rsgislib` dependencies. Full online documentation is available at <https://rsgispy.readthedocs.io>.

> Always verify exact module paths, function names, and parameter names against the tables below or the bundled files in `references/`—do **not** invent RSGISLib-style camelCase names (`bandMath`, `calcSlope`) or guess unverified identifiers.

## Core Conventions (Must Follow)

1. **No `osgeo` / `rsgislib` imports**: `rsgispy` is built on `rasterio`, `geopandas`, `pyogrio`, `shapely`, `pyproj`, `numpy`, `pandas`, `matplotlib`, `tqdm`, and `pymu-eval`.
2. **1-based band indexing**: Raster band indices (`img_band`, `bands`, `band_names={1: "blue", ...}`, `BandDefn(..., img_band=1)`) are **1-based**, matching GDAL/rasterio conventions.
3. **Use `rsgispy` constants, not raw integers**:
   - **Data types**: `rsgispy.TYPE_UNDEFINED`, `TYPE_8INT`, `TYPE_16INT`, `TYPE_32INT`, `TYPE_64INT`, `TYPE_8UINT`, `TYPE_16UINT`, `TYPE_32UINT`, `TYPE_64UINT`, `TYPE_32FLOAT`, `TYPE_64FLOAT`
   - **Resampling**: `rsgispy.INTERP_NEAREST_NEIGHBOUR`, `INTERP_BILINEAR`, `INTERP_CUBIC`, `INTERP_CUBICSPLINE`, `INTERP_LANCZOS`, `INTERP_AVERAGE`, `INTERP_MODE`, `INTERP_MIN`, `INTERP_MAX`, `INTERP_MEDIAN`, `INTERP_Q1`, `INTERP_Q3`, `INTERP_SUM`
   - **Image stretches (mapping)**: `rsgispy.IMG_STRETCH_NONE`, `IMG_STRETCH_USER`, `IMG_STRETCH_LINEAR`, `IMG_STRETCH_STDEV`, `IMG_STRETCH_CUMULATIVE`
   - **Pan-sharpening resolution**: `rsgispy.SHARP_RES_IGNORE`, `SHARP_RES_LOW`, `SHARP_RES_HIGH`
   - **Geometry types**: `rsgispy.GEOM_PT`, `GEOM_LINE`, `GEOM_POLY`, `GEOM_MPT`, `GEOM_MLINE`, `GEOM_MPOLY`
   - **Rounding / Stats / Correlation / Metrics / Variables**: `ROUND_{NEAREST,UP,DOWN}`, `SUMTYPE_{UNDEFINED,MODE,MEAN,MEDIAN,MIN,MAX,STDDEV,COUNT,RANGE,SUM}`, `STATS_CORR_{UNDEFINED,PEARSONS,SPEARMAN,KENDALL_TAU,POINT_BISERIAL}`, `METRIC_{EUCLIDEAN,SQDIFF,MANHATTAN,CORRELATION,MUTUAL_INFO,NORM_MUTUAL_INFO,PHASE_CORR}`, `VAR_TYPE_{UNDEFINED,CONTINUOUS,CATEGORICAL}`
   - **Typed `BBox` (`rsgispy.BBox`)**: Immutable `tuple` subclass storing RSGIS order `(west, east, south, north)` / `(xmin, xmax, ymin, ymax)` with optional `epsg`. Use `bbox.as_geo()` for `(minx, miny, maxx, maxy)` (Shapely/Rasterio order) and `BBox.from_geo(bounds, epsg=...)` / `BBox.from_rsgis(bbox, epsg=...)`.
   - **Top-level converters & exceptions**: `rsgispy.RSGISPyException`, `get_rsgis_datatype`, `rsgis_dtype_from_np_dtype`, `get_gdal_datatype_name`, `get_gdal_datatype_enum`, `get_rasterio_datatype`, `get_numpy_datatype`, `get_rsgis_datatype_str`, `get_rasterio_interp_type`, `get_tqdm`, `is_notebook`, `check_optional_dep`, `get_rsgispy_version`.
4. **Standardised keyword-only raster output options**: Most raster-writing functions accept keyword-only options after `*`:
   `*, gdalformat="GTIFF", datatype=..., no_data_val=..., calc_stats=..., use_pbar=..., creation_opts=...`
   (Note: `band_maths` and `image_band_math` take `gdalformat`, `datatype`, `use_pbar`, `creation_opts`, `band_names`; `calc_indices` / `calc_sar_indices` take `no_data_val` and `out_no_data_val`; `elevation` and `segmentation` functions omit `creation_opts`; default `gdalformat` is `"GTIFF"`, **not** `"KEA"`).
5. **Dual path / `GeoDataFrame` vector API**: Most functions in `rsgispy.vector.*`, `rsgispy.zonalstats.*`, and `rsgispy.elevation.roughness` accept either a file path (`str | os.PathLike`) or a `geopandas.GeoDataFrame`. Passing `out_vec_file=None` returns an in-memory `GeoDataFrame` instead of writing to disk. GeoParquet (`.parquet` / `.geoparquet`) is supported with `out_compress` (`"snappy"`, `"gzip"`, `"zstd"`, etc.).
6. **Optional extras (`pip install "rsgispy[<extra>]"`)**:
   - Core install covers raster/vector I/O, `imagecalc`, `imagecalibration` (radiometry, solar angles, Landsat C2, Sentinel-2 L2A), `imageregistration`, `imageutils`, `imagefilter` (except GLCM), `imagemorphology`, `elevation`, `changedetect`, `zonalstats`, `droneutils`, and `cli`.
   - Extras: `numba` (JIT hydrology/segmentation/terrain acceleration), `classification`, `regression`, `xgboost`, `lightgbm`, `catboost`, `gbdt` (`xgboost`+`lightgbm`+`catboost`), `stats` (`scipy`, `scikit-learn`, `scikit-gstat`), `vector` (`alphashape`, `h3`, `openpyxl`, `pyarrow`, `rtree`, `xlsxwriter`), `mapping` (`contextily`, `mpl-scatter-density`, `owslib`), `dataaccess` (`beautifulsoup4`, `osmnx`, `planet`, `pycurl`, `pysondb`, `requests`), `imaging` (`h5py`, `imageio`, `pillow`, `pyexiftool`, `pymupdf`, `qrcode`, `scikit-image`—needed for `apply_glcm_filter`), `unmixing` (`cvxopt`, `pyMCR`, `pysptools`, `scikit-learn`, `scipy`), `cloudmask` (`requests`, `segmentation-models-pytorch`, `torch`—for CloudSEN12), `jupyter` (`ipywidgets`), `modelling` (`SALib`, `shap`, `scikit-learn`, `scipy`), `all` (everything except `cloudmask`/`test`/`docs`), `dev`.

## Subpackage Routing Table

| Task | Module | Key Entrypoints | Reference File |
| --- | --- | --- | --- |
| Band math (`pymu-eval`) | `rsgispy.imagecalc.maths` | `BandDefn`, `band_maths`, `image_band_math`, `image_math`, `calc_prop_true_exp` | `raster_and_indices.md` |
| Optical (97) & SAR (63) indices | `rsgispy.imagecalc.calcindices` | `calc_indices`, `calc_sar_indices`, `get_available_*`, `get_*_req_bands`, `get_*_equation` | `raster_and_indices.md` |
| Stats, histograms, percentiles | `rsgispy.imagecalc.{stats,percentiles}` | `StatsSummary`, `calc_basic_img_band_stats`, `get_img_histogram`, `calc_multi_img_band_stats`, `calc_band_percentile` | `raster_and_indices.md` |
| Rescale, compare, fill, dist, encode | `rsgispy.imagecalc.*` | `normalise_img_pxl_vals`, `are_imgs_equal`, `calc_img_similarity_metrics`, `region_growing_fill`, `calc_dist_to_img_vals`, `recode_int_raster`, `perform_least_cost_path_calc`, `calc_pt_smpl_img_vals` | `raster_and_indices.md` |
| Metadata, masking, warp, tile, VRT | `rsgispy.imageutils.*` | `ImageBandInfo`, `mask_img`, `create_valid_mask`, `reproject_image`, `resample_img_to_match`, `gdal_warp`, `gdal_translate`, `rasterise_vec_lyr`, `create_img_mosaic`, `subset`, `stack_img_bands`, `create_tiles`, `pop_img_stats`, `create_img_extent_lut` | `raster_and_indices.md` |
| Radiometry, L1/L2 ingest, CloudSEN12 | `rsgispy.imagecalibration.*` | `radiance_to_toa_refl`, `landsat_thermal_rad_to_brightness`, `perform_dos_calc`, `calc_solar_azimuth_zenith`, `create_stacked_toa_ls_oli_cl2_lv1_img`, `create_stacked_sref_ls_*`, `create_stacked_sen2_esa_l2a_img`, `apply_cloudsen12_s2_*` | `calibration_registration_and_spectral.md` |
| Image registration & tie points | `rsgispy.imageregistration.*` | `find_image_offset`, `apply_offset_to_image`, `generate_tie_points`, `filter_tie_points_outliers`, `warp_with_gcps`, `warp_with_tie_points`, `coregister_images` | `calibration_registration_and_spectral.md` |
| Change detection & outliers | `rsgispy.changedetect.*` | `image_difference`, `image_ratio`, `change_vector_analysis`, `multivariate_alteration_detection`, `slow_feature_analysis`, `find_class_{otsu,li,kurt_skew,pyod}_outliers` | `calibration_registration_and_spectral.md` |
| Spectral transforms & unmixing | `rsgispy.imagespectral.*` | `calc_tasseled_cap`, `calc_pca`, `calc_mnf`, `histogram_matching`, `spec_unmix_{ucls,nnls,fcls}`, `spectral_angle_mapper`, `extract_avg_endmembers` | `calibration_registration_and_spectral.md` |
| DEM terrain, illumination, roughness | `rsgispy.elevation.*` | `slope`, `aspect`, `categorise_aspect`, `hillshade`, `shadow_mask`, `local_incidence_angle`, `fill_dem_soille_gratin_1994`, `fft_dem_fusion`, `calc_all_roughness_profile_metrics` | `elevation_segmentation_and_filters.md` |
| DEM hydrology & flood indices | `rsgispy.elevation.hydrology` | `condition_dem`, `fill_depressions`, `breach_depressions`, `flow_direction`, `flow_accumulation`, `extract_streams`, `stream_order`, `delineate_watersheds`, `calc_hand`, `calc_twi` | `elevation_segmentation_and_filters.md` |
| Spatial, SAR & texture filters | `rsgispy.imagefilter.*` | `apply_gaussian_smooth_filter`, `apply_lee_filter`, `apply_refined_lee_filter`, `apply_mean_filter`, `apply_glcm_filter`, `FilterStep`, `apply_filter_chain` | `elevation_segmentation_and_filters.md` |
| Morphological operators | `rsgispy.imagemorphology` | `create_circular_op`, `image_dilate`, `image_erode`, `image_opening`, `image_closing`, `image_gradient`, `image_black_top_hat`, `image_white_top_hat` | `elevation_segmentation_and_filters.md` |
| Segmentation & clumping | `rsgispy.segmentation.*` | `shepherd_segmentation`, `region_merge_segmentation`, `slic_segmentation`, `graph_segmentation`, `watershed_segmentation`, `clump.clump`, `relabel_segments`, `polygonise_segments`, `extract_segment_stats` | `elevation_segmentation_and_filters.md` |
| Vector I/O, geoms, attrs, select | `rsgispy.vector.*` | `vector_maths`, `read_vec_columns`, `reproj_vec_lyr`, `polygonise_raster_to_vec_lyr`, `clip_vec_lyr`, `calc_vec_area`, `spatial_select`, `create_hex_grid_polys` | `vector_and_zonalstats.md` |
| Zonal statistics (polygon & point) | `rsgispy.zonalstats.*` | `calc_zonal_band_stats_file`, `calc_zonal_band_stats`, `export_poly_pxls_to_df`, `ext_point_band_values_file` | `vector_and_zonalstats.md` |
| Shared low-level raster/vector helpers | `rsgispy.{imageprocess,vectorprocess}` | `TileGraph`, `BandMathOp`, `build_valid_mask`, `resolve_out_dtype`, `read_vec_gdf`, `finish_vec_output` | `raster_and_indices.md` / `vector_and_zonalstats.md` |
| Unified ML core (stack, CV, tune) | `rsgispy.ml.*` | `BandSpec`, `ImageFeatureStack`, `FeatureScaler`, `SpatialBlockKFold`, `SpatialBufferCV`, `nested_spatial_cv`, `tune_estimator`, `predict_raster`, `predict_proba_raster` | `ml_classification_regression_and_sdm.md` |
| Supervised & unsupervised classification | `rsgispy.classification.*` | `ClassInfoObj`, `ClassificationConfig`, `extract_and_split_class_samples`, `train_{sklearn,xgboost,lightgbm,catboost}_*`, `img_pixel_cluster`, `calc_acc_stats_img`, `apply_majority_filter` | `ml_classification_regression_and_sdm.md` |
| Continuous regression | `rsgispy.regression.*` | `RegressionConfig`, `extract_and_split_reg_samples`, `train_{sklearn,xgboost,lightgbm,catboost}_regressor`, `apply_*_regressor`, `calc_reg_metrics` | `ml_classification_regression_and_sdm.md` |
| Species distribution & XAI | `rsgispy.modelling.species_distribution` | `EnvVarInfo`, `gen_pseudo_absences_smpls`, `extract_env_var_data`, `fit_sklearn_mdl`, `fit_sklearn_sgl_cls_mdl`, `shap_sklearn_mdl_explainer`, `salib_sklearn_mdl_sensitivity`, `pred_sklearn_mdl_prob` | `ml_classification_regression_and_sdm.md` |
| Cartographic mapping | `rsgispy.mapping.*` | `create_raster_img_map`, `create_thematic_raster_map`, `create_vec_lyr_map`, `add_contextily_basemap`, `add_categorical_legend`, `MapScaleBar`, `MapNorthArrow` | `tools_mapping_dataaccess_and_cli.md` |
| Satellite / OSM data access | `rsgispy.dataaccess.*` | `copernicus_odata`, `nasa_cmr`, `openstreetmap`, `planet_data_api`, `usgs_m2m` | `tools_mapping_dataaccess_and_cli.md` |
| Drone flightline & EXIF utilities | `rsgispy.droneutils` | `split_photos_by_time`, `create_flightline_vec`, `define_extern_gps_in_imgs` | `tools_mapping_dataaccess_and_cli.md` |
| General & sensor tools | `rsgispy.tools.*` | `sensors` (`read_landsat_mtl_to_dict`, `read_sen2_l2a_mtd_to_dict`), `geometry`, `utm`, `projection`, `dates`, `files`, `tables`, `thresholding`, `stats`, `plotting` | `tools_mapping_dataaccess_and_cli.md` |
| Command-line tools | `rsgispy.cli.*` | `rsgisimginfo`, `rsgiscalcimgstats`, `rsgisvectools`, `rsgisbatchconvert2cog`, `rsgisbuildimglut`, `rsgischkgdalfile`, `rsgisfilehash`, `rsgisproj`, `rsgisuserpassfile`, `rsgiswgs84imgres`, `rsgiscopybandnames` | `tools_mapping_dataaccess_and_cli.md` |

## Essential Patterns

### 1. Band Maths (Single & Multi-Output `pymu-eval`)

Separate multiple output bands with **top-level commas** (commas inside `min(...)`, `max(...)`, `avg(...)` stay function arguments). Both existing band re-assignments and new intermediate variables (`ndvi = ...`) emit an output band and persist for later outputs:

```python
import rsgispy
from rsgispy.imagecalc.maths import BandDefn, band_maths

band_defs = [
    BandDefn(band_name="red", input_img="s2.tif", img_band=3),
    BandDefn(band_name="nir", input_img="s2.tif", img_band=4),
    BandDefn(band_name="swir1", input_img="s2.tif", img_band=5),
]
band_maths(
    output_img="veg_stack.tif",
    exp=(
        "ndvi = (nir - red) / (nir + red + 1e-6), "
        "(nir - swir1) / (nir + swir1 + 1e-6), "
        "ndvi >= 0.5 ? 1 : 0"
    ),
    band_defs=band_defs,
    band_names=["NDVI", "NDWI_Gao", "VegMask"],
    gdalformat="GTIFF",
    datatype=rsgispy.TYPE_32FLOAT,
)
```

### 2. Optical & SAR Spectral Indices

```python
import rsgispy
from rsgispy.imagecalc.calcindices import calc_indices, calc_sar_indices

# Optical: Sentinel-2 L2A scaled by 10000 -> int16 scaled by 1000
calc_indices(
    input_img="s2_sr.tif",
    output_img="optical_indices.tif",
    band_names={1: "blue", 2: "green", 3: "red", 4: "re1", 5: "nir", 6: "swir1", 7: "swir2"},
    indices=["NDVI", "NDWI_McFeeters", "MNDWI", "EVI", "MVI", "NDMaI"],
    in_gain=10000.0,
    out_gain=1000.0,
    gdalformat="GTIFF",
    datatype=rsgispy.TYPE_16INT,
)

# SAR: Dual-pol Sentinel-1 in dB (auto-converts dB <-> linear power per index)
calc_sar_indices(
    input_img="s1_grd_db.tif",
    output_img="sar_indices.tif",
    band_names={1: "vv", 2: "vh"},
    indices=["RVI_VV_VH", "DpRVIc_VV_VH", "DPSVI", "VDDPI", "CR_VH_VV_dB", "SDWI"],
    in_units="dB",
    gdalformat="GTIFF",
    datatype=rsgispy.TYPE_32FLOAT,
)
```

### 3. Sentinel-2 L2A Ingestion & CloudSEN12 Cloud Masking

```python
import rsgispy
from rsgispy.imagecalibration.sensorlvl2data import create_stacked_sen2_esa_l2a_img
from rsgispy.imagecalibration.cloudmasking import apply_cloudsen12_s2_sref_cloud_mask

create_stacked_sen2_esa_l2a_img(
    input_safe="S2A_MSIL2A_...SAFE",
    out_dir="s2_out",
    tmp_dir="s2_tmp",
    inc_cloudsen12_img=True,
)
# Uses the *_cloudsen12_sref.tif (12 bands: B01..B8A, B09, B11, B12)
apply_cloudsen12_s2_sref_cloud_mask(
    input_img="s2_out/scene_cloudsen12_sref.tif",
    output_img="s2_out/scene_cloudsen12_mask.tif",
    model_name="cloudsen12l2a",
    scale_factor=10000.0,
)
```

### 4. Vector Attribute Maths (Single & Multi-Output)

```python
from rsgispy.vector.attrs import vector_maths

vector_maths(
    vec_file="plots.gpkg",
    vec_lyr="plots",
    out_col=["agb_t", "carbon_t"],
    exp="agb = density * area_ha, agb * 0.47",
    out_vec_file="plots_out.gpkg",
    out_vec_lyr="plots",
)
```

## RSGISLib Migration Pitfalls (Avoid These Mistakes)

- **Default format**: Default `gdalformat` across `rsgispy` is `"GTIFF"` (not `"KEA"`).
- **British English identifiers**: Several constants, functions, and kwargs use British spelling:
  - `rsgispy.INTERP_NEAREST_NEIGHBOUR` (and `RSGIS_INTERP_*` aliases do not exist; use `rsgispy.INTERP_*`).
  - `rsgispy.zonalstats.METHOD_POLYCONTAINSPIXELCENTRE`, `METHOD_PIXELCONTAINSPOLYCENTRE`.
  - `keep_centre_on_tie` (`apply_majority_filter`), `copy_colour_table` (`classpostprocess`), `define_colour_table` / `get_colour_tab_info` (`imageutils.colourtable`), `colour` / `alt_colour` / `box_colour` / `swatch_edge_colour` / `colour_lut` (`mapping`, `ClassInfoObj`), `normalise_*` / `equalise_smpls` / `polygonise_*` / `vectorise_*` / `rasterise_*` / `categorise_aspect`.
  - `optimise_*` (`classcatboost`, `classlightgbm`, `classxgboost`, `regcatboost`, `reglightgbm`, `regxgboost`).
- **Single-class SDM naming**: Use `fit_sklearn_sgl_cls_mdl`, `fit_kfold_sklearn_sgl_cls_mdls`, `pred_sklearn_sgl_cls_mdl_prob`, `pred_sklearn_sgl_cls_mdl_cls`, `pred_ensemble_sklearn_sgl_cls_mdls_prob`, `pred_ensemble_sklearn_sgl_cls_mdls_cls` (not `singcls` or `single_class`).
- **Raster maths**: Use `rsgispy.imagecalc.maths.band_maths` / `image_band_math` (not `bandMath` or `imageMath`).
- **Calibration**: Use `rsgispy.imagecalibration` (`radiometry`, `solarangles`, `sensorlvl1data`, `sensorlvl2data`, `cloudmasking`) rather than `rsgislib.imagecalibration`; `radiance_to_toa_refl`, `perform_dos_calc`, and `landsat_thermal_rad_to_brightness` require explicit parameters / `LandsatThermalCoeffs`.
- **Zonal stats**: Use `rsgispy.zonalstats.polygon.calc_zonal_band_stats_file` / `calc_zonal_band_stats` and `rsgispy.zonalstats.point.ext_point_band_values_file` (not `calc_poly_zonal_stats` or `extract_img_vals_at_pts`).
- **Elevation**: Use `rsgispy.elevation.terrain.slope` / `aspect` (not `calc_slope` / `calc_aspect`), and `rsgispy.elevation.hydrology` (note `allow_pure_python=False` raises `RSGISPyException` if Numba is missing for priority-queue/topological kernels unless `allow_pure_python=True`).
- **Image filters**: Prefix `apply_*` in `rsgispy.imagefilter.{stats,smooth,edge,sar,texture}` or fuse in-memory with `FilterStep` + `apply_filter_chain`.
- **Morphology**: Under `rsgispy.imagemorphology` (not `rsgispy.morphology`).
- **Vector ops**: Under `rsgispy.vector.{io,create,geoms,attrs,stats,selection}` (not `vectorutils` or `vectorattrs`).
- **Optical index disambiguation**: Use `NDWI_McFeeters` for `(green - nir) / (green + nir)`, `NDWI_Gao` (or `NDMI`) for `(nir - swir1) / (nir + swir1)`, and `NDMaI` for `(swir2 - green) / (swir2 + green)`.

## Detailed Reference Documentation

Read the relevant reference file(s) in `references/` when writing non-trivial workflows:
- `references/raster_and_indices.md` — Top-level `BBox` / `imageprocess` / `TileGraph`, `band_maths`, all 97 optical & 63 SAR indices (equations + bands + params), `imagecalc.*` (`stats`, `percentiles`, `rescale`, `comparison`, `distance`, `encoding`, `fill`, `leastcostpath`, `calc_pt_win_smpls`), and all `imageutils.*` submodules.
- `references/calibration_registration_and_spectral.md` — `imagecalibration` (radiometry, solar angles, Landsat C2 L1/L2, Sentinel-2 L2A, CloudSEN12), `imageregistration` (`shift`, `tiepoints`, `warping`), `changedetect`, and `imagespectral`.
- `references/elevation_segmentation_and_filters.md` — `elevation` (`terrain`, `illumination`, `filtering`, `fusion`, `roughness`, `hydrology`), `segmentation` (`clump`, `shepherd`, `region_merge`, `slic`, `graph`, `watershed`, `tiling`, `utils`), `imagefilter`, and `imagemorphology`.
- `references/vector_and_zonalstats.md` — `vectorprocess` and all `vector.{io,create,geoms,attrs,stats,selection}` and `zonalstats.{polygon,point}` APIs.
- `references/ml_classification_regression_and_sdm.md` — `ml`, `classification` (sklearn/GBDT/clustering/accuracy/post-processing/features), `regression`, and `modelling.species_distribution`.
- `references/tools_mapping_dataaccess_and_cli.md` — `mapping`, `dataaccess`, `droneutils`, all 29 `tools.*` modules, and all 11 CLI tools with exact flags.
