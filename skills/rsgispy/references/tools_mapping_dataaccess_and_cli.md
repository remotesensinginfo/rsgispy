# Mapping, Data Access, Drone Utilities, General Tools & CLI

Covers `rsgispy.mapping.*`, `rsgispy.dataaccess.*`, `rsgispy.droneutils`, all 29 `rsgispy.tools.*` modules, and all 11 CLI entrypoints in `rsgispy` v0.4.1.

---

## 1. `rsgispy.mapping.*` (Matplotlib Cartographic Figures)

> **BBox order & British spelling**: `bbox` arguments expect RSGIS order `(xmin, xmax, ymin, ymax)` (`bbox.as_rsgis()`). Note British spelling on `colour`, `alt_colour`, `box_colour`, `swatch_edge_colour`, and `colour_lut`.

- **Plotting convention** (all of `mapping`, `tools.plotting` and every `plot_*`): figure functions take `ax=None`/`axs=None`, `out_plot_file=None`, `out_dpi=300` and return `(fig, ax)` / `(fig, axs)` (saving never closes the figure; assign the result in notebooks). Map functions draw on the given `ax` and return their artist. Compute functions (`calc_*`, `run_*`) keep returning data; plot it with the matching `plot_*` (e.g. `tools.stats.plot_semivariogram(vario_df)`). Helpers: `tools.plotting.get_fig_ax`, `get_fig_axs`, `save_plot(fig, out_plot_file, out_dpi=300, close=False)`, `add_stats_text_box`. House style: `with tools.plotting.rsgispy_plot_style(plot_type='chart'|'map'):` for your own plots (never changes global rcParams); `@use_plot_style` decorator for new plotting functions; colours `CATEGORICAL_COLOURS` (Okabe-Ito), `SEQUENTIAL_CMAP='viridis'`, `DIVERGING_CMAP='RdBu_r'`.
- **Raster layers (`mapping.raster`)**:
  - `create_raster_img_map(ax, input_img, img_bands, img_stch, bbox=None, title_str=None, show_scale_bar=True, use_grid=False, show_map_axis=True, no_data_val=None, no_data_out_clr=None, stch_min_max_vals=None, stch_n_stdevs=2.0, img_alpha=1.0, scale_bar_loc='upper right', plot_zorder=1.0, scale_bar_style='solid', scale_bar_units=None, show_north_arrow=False, north_arrow_style='triangle', north_arrow_loc='upper left', interpolation=None, interp_stage='data')` -> `AxesImage` (`img_stch`: `rsgispy.IMG_STRETCH_*`)
  - `create_thematic_raster_map(ax, input_img, bbox=None, title_str=None, alpha_backgd=True, img_alpha=1.0, ctab_alpha=True, show_scale_bar=True, ..., show_legend=False, classes_info=None, legend_position='right', legend_title=None, legend_ncols=None, legend_nrows=None, legend_overlay=False, legend_loc='lower right', legend_kwargs=None, legend_ax=None, interpolation=None, interp_stage='data')` -> `AxesImage` (a list for a list of images)
  - `create_raster_cmap_img_map(ax, input_img, img_band, bbox=None, title_str=None, ..., cmap_name='viridis', norm_img_vals=False, use_log_norm=False, norm_vmin=None, norm_vmax=None, vals_under_white=False, print_norm_vals=False, ..., show_colour_bar=False, cbar_label=None, img_alpha=1.0, interpolation=None, interp_stage='data')` -> `AxesImage` (use `fig.colorbar(img)`; with `show_colour_bar=True` it is `img.colorbar`)
  - All three resample on the data values by default (`interp_stage='data'`), so one-pixel features (stream channels) stay visible on a small map; `interpolation='nearest'` shows the pixels as they are.
  - `add_hillshade(ax, hshade_img=None, *, dem_img=None, bbox=None, alpha=0.4, azimuth=315.0, zenith=45.0, zorder=2, interpolation=None, interp_stage='data')` -> `AxesImage`: a grey hillshade over what is on `ax`, from a hillshade image or made in memory from a DEM (no file; equals `elevation.illumination.hillshade`); no data transparent. Give exactly one of `hshade_img` / `dem_img`.
  - `create_shaded_cmap_img_map(ax, input_img, hshade_img=None, *, dem_img=None, img_band=1, bbox=None, cmap_name='viridis', vmin=None, vmax=None, use_log_norm=False, vals_under_white=False, hshade_alpha=0.4, azimuth=315.0, zenith=45.0, show_colour_bar=True, cbar_label=None, title_str=None, show_scale_bar=True, show_north_arrow=True, show_map_axis=False, ...)` -> the colour-mapped `AxesImage` (hillshade is `ax.images[-1]`), e.g. `create_shaded_cmap_img_map(ax, "flow_acc.tif", dem_img="dem.tif", cmap_name="Blues", vmin=1, use_log_norm=True, cbar_label="Upslope cells")`.
- **Vector layers (`mapping.vector`)**:
  - `create_vec_lyr_map(ax, gp_vecs, bbox, title_str=None, vec_fill_clrs='grey', vec_line_clrs='black', vec_line_widths=0.25, vec_fill_alphas=1.0, vec_markersize=None, show_scale_bar=True, ...)` -> list of collections
  - Line layers are drawn in `vec_line_clrs` (polygons/points use `vec_fill_clrs` with `vec_line_clrs` edges; mixed layers are split by geometry type).
  - `create_choropleth_vec_lyr_map(ax, gp_vec, vec_col, bbox, title_str=None, vec_fill_cmap=None, vec_var_norm=None, ..., show_colour_bar=False, cbar_label=None)` -> first collection (for `fig.colorbar`)
  - `create_vec_pt_density_map(ax, gp_vec, bbox, title_str=None, cmap_name='viridis', use_log_norm=False, density_norm_vmin=1.0, density_norm_vmax=None, density_dpi=72, ..., show_colour_bar=False, cbar_label=None)` -> density artist
- **Basemaps (`mapping.basemaps`)**:
  - `add_contextily_basemap(ax, epsg_crs, cx_src, cx_zoom_lvl='auto', cx_attribution=None, cx_att_size=8)` -> `AxesImage` (requires `mapping` extra / `contextily`)
  - `create_wmts_img_map(ax, wmts_url, wmts_lyr, bbox, bbox_epsg, wmts_zoom_level=None, ...)` -> `AxesImage`
- **Cartographic elements (`mapping.elements`)**:
  - Constants: `SCALEBAR_STYLE_{SOLID,ALTERNATING,TICKS}`, `SCALEBAR_UNIT_{METRES,KILOMETRES}`, `NORTH_ARROW_STYLE_{TRIANGLE,ARROW,COMPASS,SIMPLE}`.
  - Classes & helpers: `MapScaleBar`, `MapNorthArrow`, `add_scale_bar(...)`, `add_north_arrow(...)`, `draw_bboxes_to_axis(...)` -> list of `Rectangle`, `add_axis_label(...)` -> `Text`.
- **Layout (`mapping.layout`)**: `calc_y_fig_size(bbox, fig_x_size)`, `define_axis_extent(ax, bbox)`, `define_map_tick_spacing(ax, tick_spacing)`, `get_overview_info(...)`, `add_overview_maps(...)`.
  - `create_map_panels(panels, ncols=None, bbox=None, figsize=None, *, suptitle=None, show_map_axis=False, show_scale_bar=True, scale_bar_loc='upper right', show_north_arrow=True, north_arrow_loc='upper left', show_legend=True, classes_info=None, legend_title=None, legend_position='right'|'bottom', axs=None, out_plot_file=None, out_dpi=300) -> (fig, axs)` — maps side by side over one extent (an image and its classifications, a change distance and the change over the image). Each panel is a `MapPanel` or dict: `panel_type` `'rgb'` (`bands`, `img_stch`, default cumulative) / `'cmap'` (`bands`=one band, `cmap_name`, `vmin`, `vmax`, `show_colour_bar`, `cbar_label`) / `'thematic'` (`classes_info`, `in_legend`) / `'vector'` (`vec`), plus `input_img`, `title`, `base_img`/`base_bands`/`base_stch` (drawn below), `map_kw` (extra args for the drawing function). One shared legend of the thematic classes found in `bbox` (`fig.legends[-1]`), one scale bar and north arrow on the first panel; `axs[i]` is panel i.
    ```python
    fig, axs = map_layout.create_map_panels(
        [{"panel_type": "rgb", "input_img": "s2.tif", "bands": [8, 9, 3], "title": "Sentinel-2"},
         {"panel_type": "thematic", "input_img": "cls.tif", "title": "Classification"},
         {"panel_type": "thematic", "input_img": "chng.tif", "base_img": "s2.tif", "base_bands": [8, 9, 3], "in_legend": False}],
        bbox=sub_bbox, legend_title="Land cover")
    ```
- **Categorical legends (`mapping.legend`)**: `calc_categorical_legend_layout`, `create_categorical_legend_patches`, `add_categorical_legend`, `add_categorical_legend_axis`, `save_categorical_legend_img`.

---

## 2. `rsgispy.dataaccess.*`

- **`copernicus_odata`** (CDSE):
  - Sensor constants: `RSGIS_ODATA_SEN1=1`, `RSGIS_ODATA_SEN1_RTC=11`, `RSGIS_ODATA_SEN2=2`, `RSGIS_ODATA_SEN3=3`, `RSGIS_ODATA_SEN5P=5`, `RSGIS_ODATA_SEN6=6`, `RSGIS_ODATA_COP_DEM=100`, `RSGIS_ODATA_SMOS=101`; orbit/order (`RSGIS_ODATA_ORBIT_DIR_{ASC,DESC}`, `RSGIS_ODATA_ORDERBY_{ASC,DESC}`); product types `RSGIS_ODATA_PROD_TYPE_S1_*`, `S2_MSI_{1C,2A}`, `S3_*`, `S5P_*`.
  - Functions: `get_sensor_collection_name`, `get_access_token(username=None, password=None)`, `query_scn_lst(sensor, bbox, start_date=None, end_date=None, cloud_cover=None, orbit_dir=None, product_type=None, order_by=1, max_n_rslts=25, start_n=None)`, `query_scn(scn_name)`, `download_scn(access_token, scn_info, out_path, use_pbar=True)`, `download_scns(...)`.
- **`usgs_m2m`** (USGS M2M):
  - `DATA_PRODUCTS`, `USGS_WRS1`, `USGS_WRS2`, `usgs_login`, `usgs_logout`, `can_user_dwnld`, `can_user_order`, `get_wrs_pt`, `get_wrs_bbox`, `get_dataset_info`, `get_earth_explorer_datasets`, `usgs_search`, `get_all_usgs_search`, `get_download_ids`, `create_scene_list`, `remove_scene_list`, `check_dwnld_opts`, `request_downloads`.
- **`nasa_cmr`**: `SessionWithHeaderNASARedirection`, `get_total_file_size`, `get_prods_info`, `check_prod_version_avail`, `get_max_prod_version`, `find_granules`, `find_all_granules`, `cmr_download_file_http`, `create_cmr_dwnld_db`, `download_granules_use_dwnld_db`.
- **`planet_data_api`**: `RSGIS_PLANET_ITEM_*`, `RSGIS_PLANET_ASSET_*`, `RSGIS_PLANET_BUNDLE_*`, `get_{item,asset,bundle}_type_str`, `planet_auth(username=None, password=None, api_key=None)`, `run_search_planet_items`, `run_download_and_validate_item`, `run_create_planet_order`, `run_download_planet_order`, `run_get_planet_orders`, `run_cancel_planet_orders`.
- **`openstreetmap`**: `get_osm_geom_type_name`, `get_osm_gdf(bbox, osm_tags, rsgis_geom_type=3, bbox_epsg=4326, clip_to_roi=False, col_filter_regex=None)`, `get_osm_to_file`, `get_osm_multi_to_file`.

---

## 3. `rsgispy.droneutils`

- `ImageTimeCluster(cluster_id=None, init_time=None)` (`.add_image_to_cluster`, `.does_cluster_overlap`, `.merge_cluster`, `.within_cluster`)
- `split_photos_by_time(input_imgs, output_base_dir, time_split_secs=60, date_time_tag='DateTime', date_time_format='%Y:%m:%d %H:%M:%S', out_date_time_format='%Y%m%d_%H%M%S', use_pbar=True)`
- `create_flightline_vec(input_imgs, pts_vec_file=None, pts_vec_lyr=None, line_vec_file=None, line_vec_lyr=None, out_format='GPKG', out_compress=None)`
- `create_flightline_extern_gps_vec(gps_data_df, input_imgs_dir, pts_vec_file=None, pts_vec_lyr=None, line_vec_file=None, line_vec_lyr=None, out_format='GPKG', out_compress=None)`
- `define_extern_gps_in_imgs(gps_data_df, input_imgs_dir, flying_height, gimbal_roll, gimbal_pitch, gimbal_yaw, overwrite=False, use_pbar=True)`

---

## 4. `rsgispy.tools.*` (All 29 Utility Modules)

- **`sensors`**: `read_landsat_mtl_to_dict(in_mtl_file)`, `read_sen2_l2a_mtd_to_dict(mtd_header_file)`.
- **`geometry`**: `reproj_bbox`, `reproj_bbox_epsg`, `do_bboxes_intersect`, `does_bbox_contain`, `calc_bbox_area`, `bbox_equal`, `bbox_intersection`, `bboxes_intersection`, `buffer_bbox`, `find_bbox_union`, `unwrap_wgs84_bbox`, `find_common_extent_on_grid`, `find_extent_on_grid`, `find_extent_on_whole_num_grid`, `get_bbox_grid`, `get_bbox_geojson_poly`, `get_bbox_centre_pt`, `reproj_point`, `reproj_point_to_wgs84`, `calc_pt_distance`, `pt_in_bbox`, `find_point_on_whole_num_grid`, `get_bbox_wkt_poly`, `degrees_to_metres`.
- **`utm`**: `min_max_eastings_for_lat`, `parse_utm_zone_str`, `epsg_for_utm(zone, hemisphere)`, `utm_from_epsg(epsg_code)`, `split_wgs84_bbox_utm_zones`, `check_valid_zone`, `to_latlon`, `from_latlon`, `latitude_to_zone_letter`, `latlon_to_zone_number`, `latlon_arr_to_utm_zone_number`, `latlon_to_mode_utm_zone_number`, `zone_number_to_central_longitude`.
- **`projection`**: `get_epsg_code_from_wkt`, `get_wkt_from_epsg_code`, `get_crs_obj`, `degrees_to_metres`, `metres_to_degrees`, `get_deg_coord_as_str`, `great_circle_distance`.
- **`tilecache`**: `TILE_SIZE=256`, `resolution`, `truncate_lng_lat`, `get_tile_for_point`, `get_tile_tl_lonlat`, `get_tile_bbox`, `flip_xyz_tms_tiles`, `get_web_mercator_xy`, `get_xy_mercator_lonlat`, `xy_mercator_bounds`, `get_quadkey_for_tile_path`, `get_tile_path_from_quadkey`, `get_tiles`, `convert_between_tms_xyz`.
- **`files`**: `get_file_basename`, `is_path_valid`, `is_path_sibling_creatable`, `does_path_exists_or_creatable`, `get_dir_name`, `split_path_all`, `delete_file_silent`, `delete_file_with_basename`, `rm_files_size_gt`, `find_file`, `find_file_none`, `find_files_ext`, `find_files_mpaths_ext`, `find_first_file`, `get_files_mod_time`, `find_files_size_limits`, `is_cmd_tool_avail`, `file_is_hidden`, `get_dir_list`, `get_dir_list_srch`, `convert_file_size_units`, `get_file_size`, `get_file_mod_time`, `get_file_lock`, `release_file_lock`, `clean_file_locks`, `sort_imgs_to_dirs_utm`, `natural_sort_file_names`, `sort_file_by_datetime`, `work_dir_context`, `ensure_parent_dir`.
- **`archives`**: `create_directory_archive`, `untar_file`, `untar_gz_file`, `unzip_file`, `untar_bz_file`, `create_targz_arch`.
- **`checkdatasets`**: `check_gdal_image_file`, `run_check_gdal_image_file`, `run_check_gdal_image_files`, `check_gdal_vector_file`, `run_check_gdal_vector_file`, `run_check_gdal_vector_files`, `cmp_to_ref_imgs`, `check_hdf5_file`, `run_check_hdf5_file`, `run_check_hdf5_files`.
- **`jsonio`**: `RSGISNumpyArrayEncoder`, `to_json_compatible`, `write_dict_to_json`, `dict_to_json_str`, `write_dict_to_json_gz`, `read_json_to_dict`, `read_gz_json_to_dict`.
- **`tables`**: `resolve_table_format`, `resolve_table_output_spec`, `write_dataframe_table`, `read_dataframe_table`.
- **`textio`**: `read_text_file`, `read_text_file_no_new_lines`, `read_text_file_to_list`, `write_list_to_file`, `write_data_to_file`, `create_ascii_text_file`, `prettify_xml_file`.
- **`dicts`**: `dict_struct_does_path_exist`, `dict_struct_get_{str,boolean,date,datetime,str_list,numeric,list}_value`.
- **`strings`**: `uid_generator`, `str_to_float`, `str_to_int`, `zero_pad_num_str`, `remove_repeated_chars`, `check_str`.
- **`dates`**: `get_days_since`, `get_days_since_date`, `find_month_end_date`, `create_year_month_start_end_lst`, `create_year_month_n_months_lst`, `is_summer_winter`, `get_datetime_increment_lst`.
- **`colours`**: `hex_to_rgb`, `rgb_to_hex`.
- **`credentials`**: `encode_base64_text`, `decode_base64_text`, `create_username_password_file`, `get_username_password`, `resolve_env_credential`, `resolve_env_user_pass`.
- **`hashing`**: `create_{md5,sha1,sha224,sha256,sha384,sha512,blake2b,blake2s,sha3_224,sha3_256,sha3_384,sha3_512}_hash`.
- **`http` / `ftp` / `web` / `wmts`**:
  - `http`: `RSGISPyResponseException`, `wget_download_file`, `check_url_exists`, `check_http_response`, `send_http_json_request`, `download_file_http`, `create_file_listings_db`, `download_http_files_use_lst_db`.
  - `ftp`: `get_ftp_file_listings`, `download_ftp_file`, `download_curl_ftp_file`, `upload_ftp_file`, `create_file_listings_db`, `filter_file_listings_db`, `download_files_use_lst_db`.
  - `web`: `find_web_imgs`, `find_web_files`.
  - `wmts`: `get_wmts_layer_list`, `get_wmts_as_img`.
- **`stats` / `thresholding` / `accuracy` / `featureselection`**:
  - `stats`: `mask_data_to_valid`, `calc_semivariogram`, `plot_semivariogram(vario_df, *, ..., axs=None, out_plot_file=None) -> (fig, axs)`, `standardise_img_data`, `normalise_img_data`, `validate_percentiles_0_100`, `compute_nanpercentiles`, `resolve_metric_type`, `calc_similarity_metric`, `phase_cross_correlation_2d` (integer peak, then re-correlation on the overlap with DFT-upsampled sub-pixel refinement; unbiased for windows ≥ 32 px with shifts below about a quarter of the window). `compute_nanpercentiles` is vectorised and matches `np.nanpercentile`.
  - `thresholding`: `get_nbins_histogram`, `get_bin_centres`, `calc_{otsu,yen,isodata,li,kurt_skew}_threshold`, `calc_hist_cross_entropy`. Otsu, Yen, ISODATA and cross-entropy take `bins=256` (int >= 2, or `"fd"` for Freedman-Diaconis).
  - `accuracy`: `bias_score`, `decompose_bias_variance`, `accuracy_scores_binary`, `bin_accuracy_scores_prob`.
  - `featureselection`: `cqv_threshold`, `calc_pandas_vif`, `lassolars_feature_selection`, `corr_feature_selection`, `breusch_pagan_test`.
- **`plotting` / `imaging` / `maths` / `system`**:
  - `plotting`: `get_raster_mpl_imshow`, `get_thematic_raster_mpl_imshow`, `limit_range_np_arr`, `{manual,linear,cumulative,stdev}_stretch_np_arr`, `plot_histogram_threshold(data, threshold=None, *, title_str=None, x_label=None, ..., ax=None, out_plot_file=None, out_dpi=300) -> (fig, ax)`, `plot_img_histogram(input_img, img_band=1, *, n_bins=200, val_range=None, no_data_val=None, in_msk_img=None, img_msk_val=1, thresholds=None, threshold_labels=None, log_y=False, title=None, x_label=None, hist_data=None, ax=None, out_plot_file=None, use_pbar=True) -> (fig, ax)` (block-wise `imagecalc.stats.get_img_histogram`; any number of labelled threshold lines, same look as `plot_histogram_threshold`; `hist_data=` a precomputed histogram dict), `plot_2d_img_histogram(input_img, x_band, y_band, *, y_img=None, bins=256, x_range=None, y_range=None, no_data_val=None, in_msk_img=None, img_msk_val=1, val_scale=1.0, normalise=True, log_scale=True, cmap='viridis', x_label=None, y_label=None, title=None, hist_data=None, points=None, point_labels=None, show_colour_bar=True, ax=None, out_plot_file=None, use_pbar=True) -> (fig, ax)` (feature space via `get_img_2d_histogram`; values / `val_scale` (100: reflectance x 10,000 -> %); ranges and `points` in image units; `points={"Veg": (4200, 300)}` marks endmembers; empty bins white; labels default to band names), `draw_histogram(ax, counts, bin_edges, thresholds=None, threshold_labels=None, hist_clr='#9a9a9a', log_y=False)`, `plot_qq(values, *, dist='norm', show_shapiro=True, title=None, ax=None, out_plot_file=None) -> (fig, ax)` (Shapiro-Wilk p < 0.05 = not normal; `describe_shapiro`, `draw_qq`), `get_fig_ax`, `get_fig_axs`, `save_plot`, `add_stats_text_box`.
  - `imaging`: `list_exif_info_pil`, `animate_img_set`, `list_exif_info`, `get_exif_info`, `extract_images_from_pdf`, `add_pdf_blank_pages`, `create_qr_code`.
  - `maths`: `round_to_nearest_val`, `is_number`, `similar_numeric_vals`, `in_bounds`, `mixed_signs`, `negative`, `is_odd`, `powerset_iter`, `powerset_lst`, `create_var_list`.
  - `system`: `get_environment_variable`, `num_process_cores`.

---

## 5. `rsgispy.cli.*` (11 Command-Line Entrypoints)

All commands support `-V/--version` and `-h/--help`, and also expose a `.py` alias (e.g., `rsgisimginfo.py`, `rsgiscalcimgstats.py`).

| Command | Module | Required Flags | Key Optional Flags |
| --- | --- | --- | --- |
| `rsgisimginfo` | `cli.imginfo` | `-i PATH_OR_GLOB...` | `--wkt`, `--proj4`, `--metadata`, `--full-metadata`, `--calc-stats`, `--json` |
| `rsgiscalcimgstats` | `cli.calcimgstats` | `-i PATH_OR_GLOB...` | `-t/--thematic`, `-n/--nodata VALUE`, `--include-zero`, `--no-pyramids`, `--no-clrtab`, `--approx`, `--ignore-cog`, `-j/--n-cores N`, `--tile-size PX`, `--no-pbar` |
| `rsgisvectools` | `cli.vectools` | `-i PATH` | `-l/--layer`, `-c/--column`, `--layers`, `--columns`, `--n-feats`, `--proj`, `--bbox`, `--geom`, `--unique-vals`, `--json` |
| `rsgisbatchconvert2cog` | `cli.batchconvert2cog` | `-i PATH_OR_GLOB...`, `-o DIR` | `--compress {LZW,DEFLATE,ZSTD,NONE}`, `--resampling {auto,nearest,bilinear,cubic,average,mode}`, `--datatype DTYPE`, `--byte`, `--bigtiff {IF_SAFER,YES,NO,IF_NEEDED}`, `--skip-existing`, `--check-existing`, `-j/--n-cores N`, `--no-pbar` |
| `rsgisbuildimglut` | `cli.buildimglut` | `-i PATH_OR_GLOB...`, `-o PATH` | `-l/--layer`, `-f/--format` (default `GPKG`), `--wgs84`, `--ignore-noproj`, `--overwrite`, `--no-pbar` |
| `rsgischkgdalfile` | `cli.chkgdalfile` | `-i PATH_OR_GLOB...` | `--type {raster,vector}`, `--no-sort`, `--mod-date YYYY-MM-DD`, `--rm-err`, `--print-names`, `-q/--quiet`, `--chk-proj`, `--epsg CODE`, `--max-size BYTES`, `--timeout SEC`, `--no-band-check`, `--n-bands N`, `--read-img`, `--sample-pixels N`, `--chk-sum`, `--multi-file` |
| `rsgiscopybandnames` | `cli.copybandnames` | `-r/--ref PATH`, `-i PATH_OR_GLOB...` | `-v/--verbose` |
| `rsgisfilehash` | `cli.filehash` | `-i PATH_OR_GLOB...` | `-a/--algorithm {md5,sha1,sha224,sha256,sha384,sha512,blake2b,blake2s,sha3_224,sha3_256,sha3_384,sha3_512}`, `--json` |
| `rsgisproj` | `cli.proj` | One of `--image`, `--vector`, `--epsg`, `--wkt`, `--wkt-file`, `--proj4`, `--proj4-file`, `--esri-file`, `--utm ZONE` | `-l/--layer`, `-f/--format {epsg,wkt,wkt2,wkt-pretty,proj4,json}...`, `-o/--output PATH` |
| `rsgisuserpassfile` | `cli.userpassfile` | `-o/--output PATH` | `-u/--username USER`, `--overwrite` |
| `rsgiswgs84imgres` | `cli.wgs84imgres` | `-i PATH_OR_GLOB...` | `--json` |
