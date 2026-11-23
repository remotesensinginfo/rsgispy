# Raster Processing, Maths, Indices & Elevation Reference

All symbols below are verified against `rsgispy` v0.2.2+.

## 1. Band & Image Maths (`rsgispy.imagecalc.maths`)

- `BandDefn(band_name=None, input_img=None, img_band=None)` (`img_band` is 1-based).
- `band_maths(output_img, exp, band_defs, exp_band_name=False, output_exists=False, *, gdalformat="GTIFF", datatype=rsgispy.TYPE_32FLOAT, use_pbar=True, creation_opts=None, band_names=None)`
- `image_band_math(input_img, output_img, exp, exp_band_name=False, output_exists=False, *, gdalformat="GTIFF", datatype=rsgispy.TYPE_32FLOAT, use_pbar=True, creation_opts=None, band_names=None)` — auto-binds bands of a single image as `b1`, `b2`, ..., `bN`.
- `image_math(input_img, output_img, exp, ...)` — applies expression element-wise per band.
- `calc_prop_true_exp(exp, band_defs, in_valid_img=None)` — returns proportion of valid pixels where `exp` evaluates true.

### Multi-Output `pymu-eval` Expressions

Separate output bands with **top-level commas**. Commas inside function calls (`min(b1, b2)`, `avg(b1, b2, b3)`) remain argument separators. Both input variable reassignments and new intermediate variables (`ndvi = ...`) persist across subsequent comma-separated outputs (each top-level assignment also emits an output band at its position; wrap in parentheses inside an expression to assign without adding a band):

```python
import rsgispy
from rsgispy.imagecalc.maths import image_band_math

image_band_math(
    input_img="s2_scaled.tif",
    output_img="scaled_and_indices.tif",
    exp=(
        "b1 = b1 * 0.0001, "                 # output band 1: scaled Blue
        "b2 = b2 * 0.0001, "                 # output band 2: scaled Green
        "b3 = b3 * 0.0001, "                 # output band 3: scaled Red
        "b4 = b4 * 0.0001, "                 # output band 4: scaled NIR
        "ndvi = (b4 - b3) / (b4 + b3), "     # output band 5: NDVI (stored in `ndvi`)
        "(b2 - b4) / (b2 + b4), "            # output band 6: NDWI_McFeeters
        "ndvi >= 0.5 ? 1 : 0"                # output band 7: binary veg mask reusing `ndvi`
    ),
    band_names=["Blue", "Green", "Red", "NIR", "NDVI", "NDWI", "VegMask"],
    gdalformat="GTIFF",
    datatype=rsgispy.TYPE_32FLOAT,
)
```

## 2. Optical & SAR Spectral Indices (`rsgispy.imagecalc.calcindices`)

### Optical Indices (`calc_indices`)

```python
calc_indices(
    input_img, output_img, band_names, indices,
    in_gain=1.0, in_offset=0.0, out_gain=1.0, out_offset=0.0,
    index_params=None, *, gdalformat="GTIFF", datatype=rsgispy.TYPE_32FLOAT,
    in_no_data_val=None, out_no_data_val=None, calc_stats=True,
    use_pbar=True, creation_opts=None,
)
```

- `band_names`: `{1: "blue", 2: "green", ...}`. Recognised canonical band names: `coastal`, `blue`, `green`, `yellow`, `red`, `re1`, `re2`, `re3`, `nir`, `nir_narrow` (or `nir2`), `swir1`, `swir2`, `pan`, `lwir1`, `lwir2`, plus hyperspectral narrowbands (`r445`, `r550`, `r559`, `r680`, `r700`, `r705`, `r740`, `r750`, `r800`, `r900`, `r970`, `r1510`, `r1680`, `r1754`, `r2133`, `r2203`, etc.).
- Conversion pipeline: `physical = (dn - in_offset) / in_gain`; `output = (index_val * out_gain) + out_offset` (clamped & rounded when writing integer `datatype`).
- `index_params`: per-index parameter overrides, e.g. `{"SAVI": {"L": 0.5}, "EVI": {"G": 2.5, "C1": 6.0, "C2": 7.5, "L": 1.0}}`.
- Discovery helpers: `get_available_indices()`, `get_index_req_bands(name)`, `get_index_equation(name)`.
- Disambiguation highlights (97 total):
  - `NDWI_McFeeters` (`NDWI`, `GNDWI`): `(green - nir) / (green + nir)`
  - `NDWI_Gao` (`NDMI`, `NDII`, `LSWI`): `(nir - swir1) / (nir + swir1)`
  - `NDMaI` (`NDWVI`): `(swir2 - green) / (swir2 + green)` (Shi et al., 2016 mangrove index)
  - `MVI` vs `MVI_Abs`: signed vs `|nir - green| / |swir1 - green|` (Baloloy et al., 2020)
  - `MNDWI` (`GMNDWI`, `NDSI`): `(green - swir1) / (green + swir1)`
  - `WBI_BlueNIR` (`WBI`): `blue / nir` (distinct from narrowband `WBI_900_970`)

### SAR Indices (`calc_sar_indices`)

```python
calc_sar_indices(
    input_img, output_img, band_names, indices, in_units,
    in_gain=1.0, in_offset=0.0, out_gain=1.0, out_offset=0.0,
    index_params=None, *, gdalformat="GTIFF", datatype=rsgispy.TYPE_32FLOAT,
    in_no_data_val=None, out_no_data_val=None, calc_stats=True,
    use_pbar=True, creation_opts=None,
)
```

- `in_units`: **Required** — `"dB"` or `"power"`. `calc_sar_indices` converts pixel values automatically between dB (`10 * log10(power)`) and linear power (`10 ** (dB / 10)`) as required by each index.
- Recognised SAR band names: `vv`, `vh`, `hh`, `hv` (case-insensitive).
- `index_params`: e.g. `{"SDWI_C": {"C": 0.008}}` (for `SDWI` constant, default `C = 1.0` in `SDWI`).
- Discovery helpers: `get_available_sar_indices()`, `get_sar_index_req_bands(name)`, `get_sar_index_equation(name)`, `get_sar_index_units(name)`.

### Band Histograms (`rsgispy.imagecalc.stats.get_img_histogram`)

`get_img_histogram(input_img, img_band=1, n_bins=None, bin_size=None, integer_bins=False, val_range=None, no_data_val=None, use_no_data=True, in_msk_img=None, img_msk_val=1, use_pbar=True)` computes streaming band histograms and returns `"counts"`, `"proportions"`, `"density"`, `"bin_edges"`, `"bin_centres"`, `"bin_widths"`, `"bin_width"`, `"n_bins"`, `"total_count"`, and per-bin records in `"bins"` (plus `"bin_values"` when `integer_bins=True` for single-integer/categorical datasets):

```python
from rsgispy.imagecalc.stats import get_img_histogram

# Continuous histogram with fixed bin width
h_cont = get_img_histogram("ndvi.tif", img_band=1, bin_size=0.1, val_range=(-1.0, 1.0))

# Categorical integer dataset (one bin per integer value)
h_cat = get_img_histogram("landcover.tif", img_band=1, integer_bins=True)
```

## 3. Image Utilities (`rsgispy.imageutils.*`)

- **Metadata (`metadata`)**: `get_img_band_count(img)`, `get_img_res(img)`, `get_img_bbox(img)`, `get_img_no_data_value(img, img_band=1)`, `get_epsg_proj_from_img(img)`, `get_band_names(img)`, `set_band_names(img, band_names)`.
- **Statistics (`stats`)**: `pop_img_stats(input_img, no_data_val=None, calc_pyramids=True, ...)`, `pop_thmt_img_stats(input_img, ...)`.
- **Masking (`masking`)**:
  - `ImageBandInfo(file_name, name=None, bands=None, band_names=None)`
  - `mask_img(input_img, in_msk_img, output_img, out_value, mask_value, ...)`
  - `gen_valid_mask(input_imgs, output_img, ...)` / `create_valid_mask(img_band_info, output_img, ...)`
  - `mask_all_band_zero_vals(input_img, output_img, ...)`
- **Warping & Resampling (`warp`)**:
  - `reproject_image(input_img, output_img, out_epsg, interp_method=rsgispy.INTERP_NEAREST_NEIGHBOUR, out_pxl_res="image", ...)`
  - `resample_img_to_match(in_ref_img, in_process_img, output_img, interp_method=0, ...)`
  - `gdal_warp(...)`, `gdal_translate(...)`
- **Rasterisation (`rasterise`)**:
  - `rasterise_vec_lyr(vec_file, vec_lyr, input_img, output_img, burn_val=1, att_column=None, use_vec_extent=False, thematic=True, ...)`
- **Subsetting, Bands, Mosaics, Tiling & VRT (`subsetting`, `bands`, `mosaic`, `tiling`, `vrt`)**:
  - `subset(input_img, vec_file, vec_lyr, output_img, ...)`
  - `subset_bbox(input_img, output_img, min_x, max_x, min_y, max_y, ...)`
  - `subset_to_img(input_img, in_roi_img, output_img, ...)`
  - `select_img_bands(input_img, output_img, bands, ...)`
  - `stack_img_bands(input_imgs, output_img, band_names=None, skip_value=None, ...)`
  - `create_img_mosaic(input_imgs, output_img, background_val=0.0, skip_val=0.0, skip_band=1, overlap_behaviour=0, ...)`
  - `create_tiles(input_img, out_img_base, tile_width, tile_height, tile_overlap=0, ...)`
  - `create_mosaic_images_vrt(input_imgs, out_vrt_file, ...)`, `create_stack_images_vrt(input_imgs, out_vrt_file, ...)`
- **Colour Tables (`colourtable`)**: `define_colour_table(input_img, clr_lut, img_band=1)`, `set_img_thematic(input_img)`.

## 4. Filtering, Morphology, Elevation, Change Detection & Spectral

- **Image Filtering (`rsgispy.imagefilter.*`)**:
  - Smooth: `apply_gaussian_smooth_filter`, `apply_kernel_filter`, `apply_kuwahara_filter`
  - SAR: `apply_lee_filter`, `apply_enhanced_lee_filter`, `apply_refined_lee_filter`, `apply_frost_filter`, `apply_gamma_map_filter`, `apply_kuan_filter`
  - Stats / Edge / Texture: `apply_mean_filter`, `apply_median_filter`, `apply_stddev_filter`, `apply_sobel_filter`, `apply_glcm_filter`
  - Chained in-memory: `FilterStep(kind, filter_size=3, params=...)` + `apply_filter_chain(input_img, output_img, steps, ...)`
- **Morphology (`rsgispy.imagemorphology`)**:
  `create_circular_op`, `image_dilate`, `image_erode`, `image_opening`, `image_closing`, `image_gradient`, `image_black_top_hat`, `image_white_top_hat`.
- **Elevation (`rsgispy.elevation.*`)**:
  - `terrain`: `slope(input_img, output_img, unit="degrees", ...)`, `aspect(input_img, output_img, ...)`, `categorise_aspect(...)`
  - `illumination`: `hillshade(input_img, output_img, azimuth, zenith, ...)`, `shadow_mask(...)`, `local_incidence_angle(...)`
  - `filtering`: `fill_dem_soille_gratin_1994`, `dtm_aspect_median_filter`, `plane_fit_detrend_dem`
  - `hydrology` (`rsgispy.elevation.hydrology`, projected CRS in metres required; `allow_pure_python=False` for topological/PQ ops):
    - Conditioning: `identify_sinks`, `fill_depressions`, `breach_depressions`, `burn_streams`, `resolve_flats`, `condition_dem`
    - Flow routing: `flow_direction` (`method="d8"|"dinf"|"mfd"`), `flow_accumulation` (`out_units="cells"|"area"`), `flow_length`, `flow_distance`, `trace_flow_path`
    - Streams: `extract_streams`, `stream_link`, `stream_order` (`"strahler"|"shreve"`), `vectorise_streams`
    - Watersheds: `snap_pour_points`, `delineate_watersheds`, `delineate_basins`, `delineate_subcatchments`, `polygonise_watersheds`
    - Terrain indices: `calc_hand`, `flood_inundation_from_hand`, `calc_twi`, `calc_spi`, `calc_ls_factor`
- **Change Detection (`rsgispy.changedetect.*`)**:
  - `imgchngmthds`: `image_difference`, `image_ratio`, `change_vector_analysis`, `multivariate_alteration_detection`, `slow_feature_analysis`
  - `pxloutlierchng`: `find_class_otsu_outliers`, `find_class_li_outliers`, `find_class_kurt_skew_outliers`, `find_class_pyod_outliers`
- **Spectral (`rsgispy.imagespectral.*`)**:
  - `transforms`: `calc_tasseled_cap`, `calc_pca`, `calc_mnf`, `histogram_matching`, `whiten_image`
  - `unmixing`: `spec_unmix_fcls`, `spec_unmix_nnls`
  - `classification`: `spectral_angle_mapper`, `spectral_correlation_mapper`
- **Segmentation (`rsgispy.segmentation.*`)**:
  - **Connected-component clumping (`rsgispy.segmentation.clump`)**: `clump(input_img, output_img, in_memory=False, ...)`, `union_of_clumps(input_imgs, output_img, ...)`.
  - **Remote sensing segmentation (`rsgispy.segmentation`)** — all raster functions follow `(input_img, output_img, ..., *, gdalformat="GTIFF", datatype=None, no_data_val=None, calc_stats=True, use_pbar=True)`, with matching in-memory `*_array(...)` functions and bounded-memory tiling (`tile_size`, `overlap`, `n_workers`, `stitch_seams=True`, `use_numba=True`):
    - `shepherd_segmentation` / `shepherd_segmentation_array`: Shepherd et al. (2019) global K-means + connected-component clumping + iterative small-segment elimination (`n_clusters=60, min_pixels=100, dist_thres=None, sampling=100, max_iter=50, random_state=42`).
    - `region_merge_segmentation` / `region_merge_segmentation_array`: Baatz & Schäpe multiresolution FNEA region merging (`scale=25.0, compactness=0.5, shape_weight=0.1, min_pixels=10`).
    - `slic_segmentation` / `slic_segmentation_array`: Global grid-aligned SLIC superpixels (`n_segments=500, segment_size=None, compactness=10.0, max_iter=10, sigma=0.0, enforce_connectivity=True, min_pixels=10`).
    - `graph_segmentation` / `graph_segmentation_array`: Felzenszwalb–Huttenlocher graph-based segmentation (`scale=100.0, sigma=0.8, min_pixels=50`).
    - `watershed_segmentation` / `watershed_segmentation_array`: Seeded or auto-seeded compact gradient watershed (`markers_img=None, min_distance=10, gradient_sigma=1.0, compactness=0.0, min_pixels=10`).
  - **Post-processing & GEOBIA helpers (`rsgispy.segmentation.utils`)**: `relabel_segments`, `relabel_segments_array`, `eliminate_small_segments`, `eliminate_small_segments_array`, `polygonise_segments`, `extract_segment_stats`, `is_numba_available`.
