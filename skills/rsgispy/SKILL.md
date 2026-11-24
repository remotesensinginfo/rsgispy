---
name: rsgispy
description: Generate idiomatic Python code using rsgispy, the pure-Python replacement for RSGISLib built on rasterio, geopandas, shapely, pyproj, numpy, pandas, and pymu-eval. Use whenever the user asks to write, debug, or migrate remote sensing, Earth observation, GIS, raster processing, spectral or SAR index calculation (calc_indices, calc_sar_indices), band_maths / vector_maths expressions, zonal statistics, DEM terrain derivatives, image filtering, change detection, vector geometry/attribute operations, cartographic mapping, or ML classification/regression pipelines using rsgispy or migrating from RSGISLib.
---

# rsgispy Coding Guide

`rsgispy` (v0.2.2+) is a **pure-Python** replacement for RSGISLib with **zero** `osgeo` or `rsgislib` dependencies. Full online documentation is available at <https://rsgispy.readthedocs.io>.

> Always verify exact module paths and function names against the tables below or the reference files in `references/`—do **not** invent RSGISLib-style camelCase names (`bandMath`, `calcSlope`) or guess unverified function names.

## Core Conventions (Must Follow)

1. **No `osgeo` / `rsgislib` imports**: `rsgispy` uses `rasterio`, `geopandas`, `shapely`, `pyproj`, `numpy`, `pandas`, and `pymu-eval`.
2. **1-based band indexing**: Raster band indices (`img_band`, `bands`, `band_names={1: "blue", ...}`) are **1-based**, matching GDAL/rasterio conventions.
3. **Use `rsgispy` constants, not raw integers**:
   - Data types: `rsgispy.TYPE_8INT`, `TYPE_16INT`, `TYPE_32INT`, `TYPE_64INT`, `TYPE_8UINT`, `TYPE_16UINT`, `TYPE_32UINT`, `TYPE_64UINT`, `TYPE_32FLOAT`, `TYPE_64FLOAT`
   - Resampling: `rsgispy.INTERP_NEAREST_NEIGHBOUR`, `INTERP_BILINEAR`, `INTERP_CUBIC`, `INTERP_CUBICSPLINE`, `INTERP_LANCZOS`, `INTERP_AVERAGE`, `INTERP_MODE`
   - Image stretches (mapping): `rsgispy.IMG_STRETCH_NONE`, `IMG_STRETCH_USER`, `IMG_STRETCH_LINEAR`, `IMG_STRETCH_STDEV`, `IMG_STRETCH_CUMULATIVE`
   - Geometry types: `rsgispy.GEOM_PT`, `GEOM_LINE`, `GEOM_POLY`, `GEOM_MPT`, `GEOM_MLINE`, `GEOM_MPOLY`
4. **Standardised keyword-only raster output options**: Most raster-writing functions accept these keyword-only parameters after `*`:
   `*, gdalformat="GTIFF", datatype=..., no_data_val=..., calc_stats=..., use_pbar=..., creation_opts=...`
   (Note: `band_maths` and `image_band_math` accept `gdalformat`, `datatype`, `use_pbar`, `creation_opts`, `band_names`; `calc_indices` / `calc_sar_indices` use `no_data_val` and `out_no_data_val`.)
5. **Dual path / `GeoDataFrame` vector API**: Most functions in `rsgispy.vector.*` and `rsgispy.zonalstats.*` accept either a file path (`str | Path`) or a `geopandas.GeoDataFrame`. Passing `out_vec_file=None` (where supported) returns an in-memory `GeoDataFrame` instead of writing to disk.
6. **Optional extras**: Core install (`pip install rsgispy`) covers raster/vector I/O, `band_maths`, `vector_maths`, `calc_indices`, `calc_sar_indices`, `elevation`, `imagemorphology`, `imageutils`, and `zonalstats`. ML/GBDT, SHAP/SALib, GLCM/scikit-image, `dataaccess`, and `mapping` basemaps use extras: `classification`, `regression`, `gbdt`, `modelling`, `stats`, `vector`, `mapping`, `dataaccess`, `imaging`, `unmixing`, or `all`.

## Subpackage Routing Table

| Task | Module | Key Entrypoints |
| --- | --- | --- |
| Band math (`pymu-eval`) | `rsgispy.imagecalc.maths` | `BandDefn`, `band_maths`, `image_band_math`, `image_math`, `calc_prop_true_exp` |
| Optical indices (97) | `rsgispy.imagecalc.calcindices` | `calc_indices`, `get_available_indices`, `get_index_req_bands`, `get_index_equation` |
| SAR indices (63) | `rsgispy.imagecalc.calcindices` | `calc_sar_indices`, `get_available_sar_indices`, `get_sar_index_req_bands`, `get_sar_index_units` |
| Raster stats / histograms / percentiles | `rsgispy.imagecalc.stats`, `percentiles` | `calc_basic_img_band_stats`, `get_img_histogram`, `calc_multi_img_band_stats`, `calc_band_percentile` |
| Comparison / rescale / fill / dist | `rsgispy.imagecalc.{comparison,rescale,fill,distance,encoding}` | `are_imgs_equal`, `rescale_img_pxl_vals`, `region_growing_fill`, `calc_dist_to_img_vals`, `recode_int_raster` |
| Metadata, masking, mosaic, warp, tile, VRT | `rsgispy.imageutils.*` | `ImageBandInfo`, `mask_img`, `create_valid_mask`, `reproject_image`, `resample_img_to_match`, `rasterise_vec_lyr`, `create_img_mosaic`, `subset`, `subset_bbox`, `stack_img_bands`, `create_tiles`, `pop_img_stats` |
| Spatial & SAR filters | `rsgispy.imagefilter.*` | `apply_gaussian_smooth_filter`, `apply_lee_filter`, `apply_refined_lee_filter`, `apply_mean_filter`, `apply_glcm_filter`, `FilterStep`, `apply_filter_chain` |
| Morphology | `rsgispy.imagemorphology` | `create_circular_op`, `image_dilate`, `image_erode`, `image_opening`, `image_closing` |
| DEM terrain & illumination | `rsgispy.elevation.{terrain,illumination,filtering,roughness}` | `slope`, `aspect`, `categorise_aspect`, `hillshade`, `shadow_mask`, `local_incidence_angle`, `fill_dem_soille_gratin_1994` |
| DEM hydrology | `rsgispy.elevation.hydrology` | `identify_sinks`, `fill_depressions`, `breach_depressions`, `burn_streams`, `resolve_flats`, `condition_dem`, `flow_direction`, `flow_accumulation`, `flow_length`, `flow_distance`, `trace_flow_path`, `extract_streams`, `stream_link`, `stream_order`, `vectorise_streams`, `snap_pour_points`, `delineate_watersheds`, `delineate_basins`, `delineate_subcatchments`, `polygonise_watersheds`, `calc_hand`, `flood_inundation_from_hand`, `calc_twi`, `calc_spi`, `calc_ls_factor` |
| Change detection | `rsgispy.changedetect.{imgchngmthds,pxloutlierchng}` | `image_difference`, `image_ratio`, `change_vector_analysis`, `multivariate_alteration_detection`, `find_class_otsu_outliers` |
| Spectral transforms & unmixing | `rsgispy.imagespectral.{transforms,unmixing,classification}` | `calc_tasseled_cap`, `calc_pca`, `spec_unmix_fcls`, `spec_unmix_nnls`, `spectral_angle_mapper` |
| Segmentation | `rsgispy.segmentation.*` | `shepherd_segmentation`, `region_merge_segmentation`, `slic_segmentation`, `graph_segmentation`, `watershed_segmentation`, `eliminate_small_segments`, `relabel_segments`, `polygonise_segments`, `extract_segment_stats`, `clump.clump`, `clump.union_of_clumps` |
| Vector I/O, geoms, attrs, stats, select | `rsgispy.vector.{io,create,geoms,attrs,stats,selection}` | `vector_maths`, `read_vec_columns`, `write_vec_column`, `reproj_vec_lyr`, `polygonise_raster_to_vec_lyr`, `clip_vec_lyr`, `calc_vec_area`, `spatial_select` |
| Zonal statistics | `rsgispy.zonalstats.{polygon,point}` | `calc_zonal_band_stats_file`, `calc_zonal_band_stats`, `export_poly_pxls_to_df`, `ext_point_band_values_file` |
| Unified ML core | `rsgispy.ml.{stack,scaling,predict,cv,tune}` | `BandSpec`, `ImageFeatureStack`, `FeatureScaler`, `predict_raster` |
| Supervised classification | `rsgispy.classification.*` | `ClassInfoObj`, `ClassificationConfig`, `extract_and_split_class_samples`, `train_sklearn_classifier`, `apply_sklearn_classifier`, `calc_acc_stats_img`, `apply_majority_filter`, `apply_sieve_filter` |
| Continuous regression | `rsgispy.regression.*` | `RegressionConfig`, `extract_and_split_reg_samples`, `train_sklearn_regressor`, `apply_sklearn_regressor`, `calc_reg_metrics` |
| Species distribution & SHAP | `rsgispy.modelling.species_distribution` | `extract_env_var_data`, `fit_sklearn_mdl`, `shap_sklearn_mdl_explainer`, `salib_sklearn_mdl_sensitivity`, `pred_sklearn_mdl_prob` |
| Cartographic maps | `rsgispy.mapping.{raster,vector,basemaps,elements,legend}` | `create_raster_img_map`, `create_thematic_raster_map`, `create_vec_lyr_map`, `add_contextily_basemap` |
| Satellite / OSM data access | `rsgispy.dataaccess.*` | `copernicus_odata`, `nasa_cmr`, `openstreetmap`, `planet_data_api`, `usgs_m2m` |
| Drone utilities | `rsgispy.droneutils` | `split_photos_by_time`, `create_flightline_vec`, `define_extern_gps_in_imgs` |

## Essential Patterns

### 1. Band Maths (Single & Multi-Output `pymu-eval`)

Separate multiple output bands with **top-level commas** (commas inside `min(...)`, `max(...)`, `avg(...)` stay function arguments). Both existing band re-assignments and new intermediate variables (`ndvi = ...`) can be defined and reused by later outputs:

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

# SAR: Dual-pol Sentinel-1 in dB (automatically converts dB <-> linear power per index)
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

### 3. Vector Attribute Maths (Single & Multi-Output)

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

## RSGISLib Migration Pitfalls (Common Wrong Names to Avoid)

- **Raster maths**: Use `rsgispy.imagecalc.maths.band_maths` / `image_band_math` (not `bandMath` or `imageMath`).
- **Zonal stats**: Use `rsgispy.zonalstats.polygon.calc_zonal_band_stats_file` or `calc_zonal_band_stats` and `rsgispy.zonalstats.point.ext_point_band_values_file` (not `calc_poly_zonal_stats` or `extract_img_vals_at_pts`).
- **Elevation**: Use `rsgispy.elevation.terrain.slope` / `aspect` (not `calc_slope` / `calc_aspect`).
- **Image filters**: Functions are prefixed with `apply_*` in `rsgispy.imagefilter.{stats,smooth,edge,sar,texture}` (e.g., `apply_gaussian_smooth_filter`, `apply_lee_filter`, `apply_mean_filter`).
- **Morphology**: Use `rsgispy.imagemorphology.image_dilate`, `image_erode`, `image_opening`, `image_closing` (not `rsgispy.morphology`).
- **Vector I/O & attributes**: Consolidated under `rsgispy.vector.{io,create,geoms,attrs,stats,selection}` (not `rsgislib.vectorutils` or `rsgislib.vectorattrs`).
- **Optical index disambiguation**: Use `NDWI_McFeeters` for `(green - nir) / (green + nir)`, `NDWI_Gao` (or `NDMI`) for `(nir - swir1) / (nir + swir1)`, and `NDMaI` for Shi et al. (2016) `(swir2 - green) / (swir2 + green)`.

## CLI Tools

Installed console scripts (also available with `.py` suffix):
`rsgisimginfo`, `rsgiscalcimgstats`, `rsgisvectools`, `rsgisbatchconvert2cog`, `rsgisbuildimglut`, `rsgischkgdalfile`, `rsgisfilehash`, `rsgisproj`, `rsgisuserpassfile`, `rsgiswgs84imgres`, `rsgiscopybandnames`.

## Detailed Reference Documentation

Read these bundled reference files when writing non-trivial workflows:
- `references/raster_and_indices.md` — Band/image maths, all 97 optical & 63 SAR index conventions, `imageutils`, `imagefilter`, `imagemorphology`, `elevation`, `changedetect`, `imagespectral`, and `segmentation`.
- `references/vector_zonal_and_ml.md` — `rsgispy.vector`, `rsgispy.zonalstats`, `rsgispy.ml`, `rsgispy.classification`, `rsgispy.regression`, `rsgispy.modelling`, `rsgispy.mapping`, `rsgispy.dataaccess`, and CLI usage.
