# Image Calibration, Co-Registration, Change Detection & Spectral Processing

Covers `rsgispy.imagecalibration.*`, `rsgispy.imageregistration.*`, `rsgispy.changedetect.*`, and `rsgispy.imagespectral.*` in `rsgispy` v0.3.1.

---

## 1. `rsgispy.imagecalibration.*`

### Radiometry & Solar Geometry (`rsgispy.imagecalibration.radiometry` & top-level `rsgispy.imagecalibration`)
- `LandsatThermalCoeffs(band_name: str, img_band: int, k1: float, k2: float)` (1-based `img_band`)
- `get_julian_day(year: int, month: int, day: int) -> int`
- `calc_solar_distance(julian_day: int | float) -> float` (Earth–Sun distance in AU)
- `get_esun_value(radiance, toa_refl, day, month, year, solar_zenith) -> float`
- `visibility_to_aot(vis: float) -> float`
- `radiance_to_toa_refl(input_img, output_img, scale_factor, year, month, day, solar_zenith, solar_irradiance, *, gdalformat='GTIFF', datatype=9, no_data_val=None, calc_stats=True, use_pbar=True, creation_opts=None)`
  - `solar_irradiance`: sequence of per-band floats (or objects/dicts with `.irradiance` / `['irradiance']`) matching band order.
- `landsat_thermal_rad_to_brightness(input_img, output_img, band_defs: Sequence[LandsatThermalCoeffs], scale_factor=1.0, *, gdalformat='GTIFF', datatype=9, no_data_val=None, calc_stats=True, use_pbar=True, creation_opts=None)`
- `perform_dos_calc(input_img, output_img, non_negative=True, no_data_val=0.0, dark_obj_refl_val=0.0, dark_obj_percentile=0.01, copy_band_names=True, *, gdalformat='GTIFF', datatype=None, calc_stats=True, use_pbar=True, creation_opts=None)`

### Solar Angles (`rsgispy.imagecalibration.solarangles`)
- `get_solar_irr_convention_solar_azimuth_from_usgs(solar_az: float) -> float`
- `get_solar_irr_convention_solar_azimuth_from_trad(solar_az: float) -> float`
- `calc_solar_azimuth_zenith(input_img, img_datetime: datetime.datetime, output_img, gdalformat='GTIFF', *, datatype=9, calc_stats=True, use_pbar=True, creation_opts=None)` — writes 2-band raster `[SolarAzimuth, SolarZenith]` in degrees.

### Landsat Collection 2 & Sentinel-2 L2A Ingestion (`sensorlvl1data`, `sensorlvl2data`)
- `sensorlvl1data.create_stacked_toa_ls_oli_cl2_lv1_img(input_file, out_dir, tmp_dir, scale_factor=10000, gdalformat='GTIFF', delete_inter_data=True, inc_full_qa_pxl_img=False, *, calc_stats=True, use_pbar=True, creation_opts=None)` — pass MTL path (`*_MTL.txt` / `.json`).
- `sensorlvl2data.parse_landsat_c2_qa_pixel_img(input_img, output_img, gdalformat='GTIFF', *, calc_stats=True, use_pbar=True, creation_opts=None)`
- `sensorlvl2data.create_stacked_sref_ls_oli_cl2_lv2_img(input_file, out_dir, tmp_dir, scale_factor=10000, gdalformat='GTIFF', delete_inter_data=True, inc_full_qa_pxl_img=False, *, ...)` (Landsat 8/9 OLI C2 L2)
- `sensorlvl2data.create_stacked_sref_ls_4_5_7_lv2_img(input_file, out_dir, tmp_dir, scale_factor=10000, gdalformat='GTIFF', delete_inter_data=True, inc_full_qa_pxl_img=False, *, ...)` (Landsat 4/5/7 TM/ETM+ C2 L2)
- `sensorlvl2data.create_stacked_sen2_esa_l2a_img(input_safe, out_dir, tmp_dir, out_res_10m=True, sharpen_10m=True, scale_factor=10000, gdalformat='GTIFF', datatype=3, delete_inter_data=True, resample_10m_method=2, resample_20m_method=5, use_multi_core=False, *, inc_cloudsen12_img=False, calc_stats=True, use_pbar=True, creation_opts=None)`
  - Setting `inc_cloudsen12_img=True` additionally writes `*_cloudsen12_sref` in exact 12-band CloudSEN12 L2A order (`B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12`).

### CloudSEN12 Cloud Masking (`rsgispy.imagecalibration.cloudmasking`; requires `cloudmask` extra)
- **Classes**: `CLOUDSEN12_NO_DATA = 0`, `CLOUDSEN12_CLEAR = 1`, `CLOUDSEN12_THICK_CLOUD = 2`, `CLOUDSEN12_THIN_CLOUD = 3`, `CLOUDSEN12_CLOUD_SHADOW = 4`.
- **Models & Band Orders**:
  - `CLOUDSEN12_MODELS`: `'UNetMobV2_V1'` (TOA/L1C), `'UNetMobV2_V2'` (TOA/L1C), `'cloudsen12l2a'` (SREF/L2A).
  - `CLOUDSEN12_S2_L1C_BANDS` (13 bands): `('B01','B02','B03','B04','B05','B06','B07','B08','B8A','B09','B10','B11','B12')`.
  - `CLOUDSEN12_S2_L2A_BANDS` (12 bands): `('B01','B02','B03','B04','B05','B06','B07','B08','B8A','B09','B11','B12')`.
- **Functions**:
  - `get_cloudsen12_weights_dir() -> str`, `download_cloudsen12_weights(model_name='UNetMobV2_V2', out_dir=None, overwrite=False, use_pbar=True, timeout=60.0) -> str`, `get_cloudsen12_weights(...) -> str`, `get_torch_device(device='cpu')`
  - `apply_cloudsen12_s2_toa_cloud_mask(input_img, output_img, model_name='UNetMobV2_V2', img_bands=None, weights_dir=None, download_weights=True, device='cpu', scale_factor=10000.0, add_offset=0.0, no_data_val=0.0, tile_size=512, tile_overlap=32, *, gdalformat='GTIFF', calc_stats=True, use_pbar=True, creation_opts=None)`
  - `apply_cloudsen12_s2_sref_cloud_mask(input_img, output_img, model_name='cloudsen12l2a', img_bands=None, ...)` (same defaults except `model_name='cloudsen12l2a'`).

---

## 2. `rsgispy.imageregistration.*`

### Constants (`rsgispy.imageregistration`)
- GCP/Output types: `TYPE_ENVI_IMG2IMG = 1`, `TYPE_ENVI_IMG2MAP = 2`, `TYPE_RSGIS_IMG2MAP = 3`, `TYPE_RSGIS_MAPOFFS = 4`
- Tie-point methods: `TIE_METHOD_PHASE_CORR = 'phase_correlation'`, `TIE_METHOD_TEMPLATE_MATCH = 'template_matching'`
- Outlier filters: `OUTLIER_METHOD_RANSAC = 'ransac'`, `OUTLIER_METHOD_RESIDUAL = 'residual'`, `OUTLIER_METHOD_MAD = 'mad'`
- Transforms: `TRANSFORM_SHIFT = 'shift'`, `TRANSFORM_SIMILARITY = 'similarity'`, `TRANSFORM_AFFINE = 'affine'`, `TRANSFORM_POLY1 = 'poly1'`, `TRANSFORM_POLY2 = 'poly2'`, `TRANSFORM_POLY3 = 'poly3'`, `TRANSFORM_TPS = 'tps'`

### Submodules
- **`shift`**:
  - `find_image_offset(in_ref_img, in_float_img, ref_img_bands=None, flt_img_bands=None, metric_type=4, x_search=15, y_search=15, sub_pxl_res=4, hist_bins=64, ref_no_data=None, flt_no_data=None, out_map_units=False, return_details=False) -> (dx, dy) | Dict` (`metric_type`: `rsgispy.METRIC_*` or string; default `4` is `METRIC_MUTUAL_INFO`)
  - `apply_offset_to_image(input_img, output_img, x_offset=0.0, y_offset=0.0, in_pxl_units=False, *, gdalformat='GTIFF', datatype=None, no_data_val=None, calc_stats=True, use_pbar=True)`
- **`tiepoints`**:
  - `generate_tie_points(in_ref_img, in_float_img, out_gcp_file=None, method='phase_correlation', ref_band=1, flt_band=1, pixel_gap=50, win_size=64, search_area=15, threshold=None, sd_ref_thres=1.0, sd_flt_thres=1.0, sub_pxl_res=4, metric_type=4, output_type=3, hist_bins=64, ref_no_data=None, flt_no_data=None, use_pbar=True) -> pd.DataFrame | None`
  - `basic_registration(in_ref_img, in_float_img, out_gcp_file, pixel_gap, threshold, win_size, search_area, sd_ref_thres, sd_flt_thres, sub_pxl_res, metric_type=4, output_type=3, ref_band=1, flt_band=1, use_pbar=True)`
  - `filter_tie_points_outliers(tie_points, out_gcp_file=None, method='ransac', model='affine', residual_thres=2.0, coord_space='map', max_trials=1000, min_samples=None, mad_scale=3.0, output_type=3, rnd_seed=None, return_model=False)`
  - `fit_geometric_model(src_x, src_y, dst_x, dst_y, model='affine') -> Dict[str, Any]`
  - `evaluate_geometric_model(model_dict, src_x, src_y) -> (x_pred, y_pred)`
- **`warping`**:
  - `gcp_to_gdal(input_img, output_img, in_gcp_file, gcp_epsg=None, *, gcp_crs=None, ...)`
  - `add_vec_pts_as_gcps_to_img(input_img, output_img, vec_file, vec_lyr=None, gcp_x_col='x_match', gcp_y_col='y_match', gcp_z_col=None, gcp_epsg=None, *, ...)`
  - `warp_with_gcps(in_ref_img, in_process_img, output_img, interp_method=1, use_tps=False, use_poly=True, poly_order=3, use_multi_threaded=True, refine_gcps=None, *, ...)`
  - `warp_with_tie_points(in_ref_img, in_float_img, output_img, tie_points, interp_method=1, transform_type='affine', poly_order=1, use_tps=False, match_ref_grid=True, refine_gcps=None, use_multi_threaded=True, *, ...) -> Dict[str, Any]`
  - `coregister_images(in_ref_img, in_float_img, output_img, tie_method='phase_correlation', outlier_method='ransac', transform_type='affine', ref_band=1, flt_band=1, pixel_gap=50, win_size=64, search_area=15, threshold=None, sd_ref_thres=1.0, sd_flt_thres=1.0, sub_pxl_res=4, metric_type=4, residual_thres=2.0, interp_method=1, match_ref_grid=True, out_gcp_file=None, rnd_seed=None, *, ...) -> Dict[str, Any]` (end-to-end single-call pipeline).

---

## 3. `rsgispy.changedetect.*`

- **`imgchngmthds`**:
  - `image_difference(in_base_img, in_chng_img, out_diff_img, vld_msk_img=None, vld_msk_val=1, apply_std=False, img_base_bands=None, img_chng_bands=None, *, gdalformat='GTIFF', datatype=None, no_data_val=None, base_img_no_data=0, chng_img_no_data=0, calc_stats=True, use_pbar=True)`
  - With no `vld_msk_img`, every change-detection function treats a pixel as valid only when **both** dates have data (any band finite and not no data); pass `vld_msk_img` to add cloud masks.
  - `image_ratio(...)` (same signature except `out_ratio_img`)
  - `change_vector_analysis(in_base_img, in_chng_img, out_chng_img, out_chng_dist_img, ..., use_kmeans=False, *)`
  - `slow_feature_analysis(in_base_img, in_chng_img, out_chng_img, out_chng_dist_img, ..., sfa_max_iter=50, sfa_epsilon=1e-6, sfa_norm_trans=False, sfa_regular=False, use_kmeans=False, *)`
  - `multivariate_alteration_detection(in_base_img, in_chng_img, out_chng_img, out_chng_dist_img, out_chng_mads_img, ..., mad_max_iter=50, mad_epsilon=1e-6, use_kmeans=True, *)` (IR-MAD)
- **`pxloutlierchng`**:
  - `find_class_pyod_outliers(input_img, in_msk_img, output_img, pyod_obj, out_scores_img=None, img_mask_val=1, img_bands=None, *, ...)`
  - `find_class_otsu_outliers(input_img, in_msk_img, output_img, low_thres, img_mask_val=1, img_band=1, plot_thres_file=None, *, ...) -> float`
  - `find_class_li_outliers(input_img, in_msk_img, output_img, low_thres, tolerance=None, init_thres=None, img_mask_val=1, img_band=1, plot_thres_file=None, *, ...) -> float`
  - `find_class_kurt_skew_outliers(input_img, in_msk_img, output_img, vld_min, vld_max, init_thres, low_thres, contamination=10.0, only_kurtosis=False, img_mask_val=1, img_band=1, plot_thres_file=None, *, ...) -> float`

---

## 4. `rsgispy.imagespectral.*`

- **`transforms`**:
  - `get_tasseled_cap_sensors() -> List[str]` (`['landsat5_tm', 'landsat7_etm', 'landsat8_oli', 'landsat9_oli', 'modis_nbar', 'sentinel2_msi']`)
  - `calc_tasseled_cap(input_img, output_img, sensor='landsat8_oli', bands=None, coeffs=None, component_names=None, gain=1.0, offset=0.0, valid_msk_img=None, valid_msk_val=1, *, gdalformat='GTIFF', datatype=9, no_data_val=None, out_no_data_val=None, calc_stats=True, use_pbar=True)`
  - `calc_pca(input_img, output_img, n_components=None, valid_msk_img=None, valid_msk_val=1, bands=None, standardise=False, out_stats_json=None, *, ...)` -> `Dict[str, Any]`
  - `calc_mnf(input_img, output_img, n_components=None, valid_msk_img=None, valid_msk_val=1, bands=None, noise_direction='both', out_stats_json=None, *, ...)` -> `Dict[str, Any]`
  - `whiten_image(input_img, output_img, valid_msk_img, valid_msk_val=1, *, gdalformat='GTIFF', use_pbar=True)`
  - `histogram_matching(input_img, ref_img, output_img, bands=None, ref_bands=None, n_bins=4096, ref_no_data_val=None, valid_msk_img=None, valid_msk_val=1, ref_valid_msk_img=None, ref_valid_msk_val=1, *, ...)`
- **`preprocessing`**: `spectral_smoothing(..., win_len=5, polyorder=3)`, `mask_outliers_data_values(..., lower=5, upper=95)`, `polyfill_nan_data_values(..., band_vals, valid_msk_img, valid_msk_val=1, polyorder=3, mean_abs_diff=None)`
- **`indices`**: `calc_area_under_curve(input_img, output_img, band_widths, *, ...)`
- **`endmembers`**: `ImageEndmemberInfo(in_unmix_img, endmembers, endmember_names=None, file_format=None)`, `extract_avg_endmembers(input_img, vec_file, vec_lyr=None, out_file=None, class_col=None, endmember_names=None, file_format=None, pxl_in_poly_method=1) -> pd.DataFrame`, `read_endmembers_df`, `write_endmembers_df`, `coerce_endmembers_df`, `prepare_endmembers_array`, `convert_mtxt_to_df`, `plot_endmembers`, `are_endmembers_equal`, `calc_ppi(input_img, output_img, niters=1000, lthres=0, uthres=0, img_gain=1, rnd_seed=None, *, ...)`
- **`classification`**: `spectral_angle_mapper(input_img, output_img, endmembers, out_class_img=None, out_min_angle_img=None, img_gain=1, endmembers_gain=1, file_format=None, degrees=False, max_angle=None, valid_msk_img=None, valid_msk_val=1, *, ...)`, `spectral_correlation_mapper(..., out_class_img=None, out_max_corr_img=None, ..., min_corr=None, ...)`
- **`unmixing`** (note `unmixing` extra needed for `pyMCR` / `pysptools` variants):
  - Built-in / NumPy+SciPy: `exhcon_linear_spec_unmix`, `spec_unmix_ucls`, `spec_unmix_nnls`, `spec_unmix_fcls` — all unmixing and post-unmixing functions take keyword-only `out_no_data_val` (default `-9999.0`; a fraction of 0 is valid). Pixels outside the mask, non-finite, equal to the input no data, or with any band ≤ 0 are not unmixed and are written as no data.
  - `pysptools`: `spec_unmix_spts_{ucls,nnls,fcls}`
  - `pyMCR`: `spec_unmix_pymcr_{nnls,fcls}`
  - Post-unmixing: `rescale_unmixing_results`, `predict_refl_linear_unmixing`, `calc_unmixing_rmse_residual_err`, `summarise_multi_endmember_linear_unmixing(input_img, out_unmix_img, out_ref_img, in_unmixed_datasets: Sequence[ImageEndmemberInfo], tmp_dir='tmp', *, ...)`
