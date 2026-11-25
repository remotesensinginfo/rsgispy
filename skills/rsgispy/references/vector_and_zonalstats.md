# Vector Operations & Zonal Statistics

Covers `rsgispy.vectorprocess`, `rsgispy.vector.{io,create,geoms,attrs,stats,selection}`, and `rsgispy.zonalstats.{polygon,point,labels}` in `rsgispy` v0.3.1.

> **Dual Path / `GeoDataFrame` API**: Almost every vector and zonalstats function accepts `str | os.PathLike | geopandas.GeoDataFrame`. Pass `out_vec_file=None` to return an in-memory `GeoDataFrame` instead of writing to disk. GeoParquet (`.parquet` / `.geoparquet`) is supported directly via `out_compress` (`'snappy'`, `'gzip'`, `'zstd'`, etc.).

---

## 1. `rsgispy.vectorprocess` (Shared Low-Level Vector Core)

- **Format & I/O**: `is_geoparquet_file`, `is_geoparquet_format`, `infer_parquet_compression`, `check_format_name(vec_file_format)`, `get_file_vec_extension(vec_format)`, `delete_vec_file(vec_file, feedback=False)`, `prepare_out_vec_file`, `list_vec_layers`, `resolve_vec_layer`, `prepare_optional_out_vec`, `resolve_out_lyr_name`, `is_multi_layer_driver`, `read_vec_info(vec_file, vec_lyr=None)` -> `Dict[str, Any]`, `read_vec_gdf(vec_file, vec_lyr=None, epsg=None, bbox=None, copy=True)` -> `GeoDataFrame`, `write_vec_gdf(gdf, out_vec_file, out_vec_lyr=None, out_format=None, overwrite=False, out_compress=None, replace_layer=False)`, `finish_vec_output(gdf, out_vec_file, out_vec_lyr=None, out_format=None, overwrite=False, out_compress=None)`.
- **CRS, BBox & geometry helpers**: `parse_crs_input(crs_val)`, `crs_to_epsg`, `resolve_vector_crs`, `align_gdf_crs`, `ensure_matching_crs`, `bbox_to_rsgis`, `get_gdf_bbox` -> `rsgispy.BBox`, `get_gdf_geom_type` -> `rsgispy.GEOM_*`, `validate_columns_exist`, `cast_col_data`, `to_python_scalar`, `build_atts_dataframe`, `pt_xy`, `extract_geom_pts`.

---

## 2. `rsgispy.vector.*` Submodules

### `rsgispy.vector.io` (Metadata, Reprojection, Merge, Split, GPX)
- **Inspect & metadata**: `VecLayersInfoObj(vec_file=None, vec_lyr=None, vec_out_lyr=None)`, `check_format_name`, `get_file_vec_extension`, `delete_vec_file`, `get_wkt_proj_from_vec`, `get_epsg_proj_from_vec`, `get_vec_lyr_bbox` -> `BBox`, `get_vec_lyr_bbox_in_proj` -> `BBox`, `get_vec_feat_count`, `get_geom_type_name`, `parse_geom_type_label`, `get_vec_lyr_geom_type`, `get_vec_lyrs_lst`, `get_vec_lyr_cols`.
- **Copy, translate & project**: `create_copy_vec_lyr`, `vector_translate`, `reproj_vec_lyr(vec_file, out_vec_file=None, out_vec_lyr=None, epsg_code=None, wkt_str=None, *, out_format='GPKG', vec_lyr=None, in_epsg_code=None, in_wkt_str=None, del_exist_vec=False, out_compress=None)`, `reproj_wgs84_vec_to_utm(..., use_hemi=False)`, `redefine_vec_lyr_proj(vec_file, vec_lyr, epsg_code, out_vec_file=None, out_vec_lyr=None, *, ...)`.
- **Merge & split**: `merge_vec_files(vec_files, ...)`, `merge_vec_lyrs(vecs_dict, ...)`, `merge_vec_file_lyrs(vec_file, ...)`, `merge_to_multi_lyr_vec(input_file_lyrs, out_vec_file=None, *, out_format='GPKG', overwrite=True, rename_dup_lyrs=False, use_pbar=True, out_compress=None)`, `check_polys_wgs84_bounds_geometry`, `merge_utm_vecs_wgs84`, `split_vec_lyr(..., n_feats, ...)`, `split_feats_to_mlyrs`, `split_by_attribute(vec_file, vec_lyr, split_col_name, multi_layers=True, ...)`, `convert_gpx_to_vec(gpx_file, pts_vec_file=None, pts_vec_lyr=None, line_vec_file=None, line_vec_lyr=None, *, out_format='GPKG', out_epsg=4326, split_segments=False, include_z=False, ...)`.

### `rsgispy.vector.attrs` (Columns, Joins, Tables & `vector_maths`)
- **Expression math (`pymu-eval`)**:
  - `vector_maths(vec_file, vec_lyr, out_col: str | Sequence[str], exp: str, out_vec_file=None, out_vec_lyr=None, *, out_format='GPKG', vec_cols=None, out_dtype=None, del_exist_vec=False, out_compress=None)`
  - Supports single or top-level comma-separated multi-output expressions; assignments persist for later outputs (same rules as `band_maths`).
- **Read / write / schema**: `write_vec_column`, `write_vec_column_to_layer`, `read_vec_column`, `read_vec_columns`, `drop_vec_cols`, `rename_vec_cols`, `sort_vec_lyr`, `get_unq_col_values`, `get_vec_cols_as_array`.
- **Export**: `export_vec_attrs_to_csv`, `export_vec_attrs_to_excel(..., out_sheet_name='Sheet1')`, `export_vec_attrs_to_parquet(..., gzip_output=True, *, out_compress=None)`.
- **Populate columns**: `add_fid_col`, `add_unq_numeric_col`, `add_numeric_col`, `add_numeric_col_lut`, `add_numeric_col_range_lut`, `add_numeric_col_from_lst_lut`, `add_string_col`, `add_string_col_lut`, `add_ran_vals_cols`, `create_name_col`, `check_str_col`, `create_date_col`, `find_replace_str_vec_lyr`, `create_angle_sets`, `create_orthogonal_angle_sets`, `pop_bbox_cols`, `add_geom_bbox_cols`.
- **Spatial joins**: `annotate_vec_selection`, `perform_spatial_join(..., join_how='inner', join_op='within', vec_base_epsg=None, vec_join_epsg=None, out_compress=None)`.

### `rsgispy.vector.create` (Raster-to-Vector, Grids, BBoxes, Sampling Points)
- `polygonise_raster_to_vec_lyr(out_vec_file, out_vec_lyr, out_format, input_img, img_band=1, mask_img=None, mask_band=1, *, replace_file=True, replace_lyr=True, pxl_val_fieldname='PXLVAL', use_8_conn=False, out_compress=None)` — whole band and all polygons in memory, one core; fine for small rasters.
- `polygonise_raster_tiled(input_img, out_vec_file, out_vec_lyr=None, img_band=1, mask_img=None, mask_band=1, no_data_val=None, pxl_val_fieldname='PXLVAL', use_8_conn=False, dissolve_by_value=False, sort_by_value=False, inc_pxl_count=False, tile_size=2048, n_jobs=1, backend='auto', *, out_format=None, out_compress='snappy', chunk_size=100_000, pxl_count_fieldname='pxl_count', tmp_dir=None, overwrite=False, use_pbar=True) -> GeoDataFrame | None` — use for rasters with many polygons (segmentations, fine classifications): tiled, multi-core (`n_jobs`), streams GeoParquet/any OGR format to disk; same polygons as `polygonise_raster_to_vec_lyr` for any tile size/`n_jobs` (8-connected diagonal regions written as valid MultiPolygons). `backend`: `'numba'` (compiled kernel, fastest), `'rasterio'` (GDAL per tile) or `'auto'`. `dissolve_by_value=True` + `sort_by_value=True` → one feature per segment ID sorted by ID (out-of-core). `out_vec_file=None` returns a GeoDataFrame (all in memory). `no_data_val=None` polygonises every pixel (pass e.g. `0` for segment rasters).
- `vectorise_pxls_to_pts(input_img, img_band, img_msk_val, out_vec_file=None, out_vec_lyr=None, *, ...)`
- `extract_img_footprint(input_img, out_vec_file=None, out_vec_lyr=None, *, out_format='GPKG', no_data_val=None, tmp_dir=None, reproj_to=None, use_pbar=True, out_compress=None)`
- `create_poly_vec_for_lst_bboxs`, `define_grid(bbox, x_size, y_size, in_epsg_code, ..., utm_grid=False, utm_hemi=False)`, `create_poly_vec_bboxs`, `write_pts_to_vec`, `create_bboxs_for_pts`, `create_wgs84_vec_grid`, `create_vec_for_imgs`, `create_hex_grid_bbox`, `create_hex_grid_poly`, `create_hex_grid_polys`, `create_random_pts_in_radius`, `create_random_pts_in_bbox`, `create_lines_vec`, `create_img_transects`, `create_bbox_vec_lyr`.

### `rsgispy.vector.geoms` (Overlays, Clips, Buffers, Morphometrics, Transects)
- **Overlays & clip**: `vec_lyr_intersection_gp`, `vec_lyr_difference_gp`, `vec_lyr_sym_difference_gp`, `vec_lyr_identity_gp`, `vec_lyr_union_gp`, `clip_vec_lyr`, `clip_and_merge_with_roi`, `clip_vec_into_sets`.
- **Transformations**: `convert_polys_to_lines_gp`, `calc_poly_centroids`, `explode_vec_lyr`, `explode_vec_files`, `simplify_geometries(..., tolerance, ...)`, `delete_polygon_holes(..., area_thres=None, ...)`, `buffer_vec_layer_gp(..., buf_dist, ..., buf_res=16, buf_sgl_sided=False, cap_style=1, join_style=1)`, `split_vec_by_grid`, `rm_polys_area(..., area_thres, ..., less_than=True)`, `vec_lyr_dissolve(vec_file, vec_lyr, vec_col, ...)`, `thin_vector_geoms`, `shiftxy_vec_lyr`, `create_alpha_shape` (requires `alphashape` extra).
- **Lines, points & coastal transects**: `get_pt_on_line`, `find_pt_to_side`, `create_orthg_lines`, `split_lines_to_reg_pts`, `closest_line_intersection`, `line_intersection_range`, `scnd_line_intersection_range`, `pts_to_line_geoms`, `create_angle_lines_from_points`, `estimate_future_coastline`.
- **Spatial predicates & R-tree**: `get_geom_pts`, `get_vec_lyr_as_pts`, `get_geoms_as_bboxs`, `bbox_intersects_vec_lyr`, `create_rtree_index`, `bbox_intersects_index`, `vec_intersects_vec`, `vec_overlaps_vec`, `vec_within_vec`, `vec_contains_vec`, `vec_touches_vec`, `vec_crosses_vec`.

### `rsgispy.vector.selection`
- `does_vmsk_img_intersect`, `spatial_select(..., predicate='intersects')`, `spatial_select_bbox`, `subset_vec_lyr_to_feat_bboxs`, `get_att_lst_select_feats`, `get_att_lst_select_bbox_feats`, `subset_by_attribute(..., match_type='equals')`, `select_feats_str_search`, `drop_rows_by_attribute`, `drop_rows_by_multi_attributes`, `rm_attrib_duplicates(..., keep_rows='First')`, `match_closest_vec_pts(..., tolerance=None, cp_match_atts=False, ...)`, `split_vec_lyr_random_subset`, `vec_lyr_random_subset`, `create_train_test_smpls`, `create_train_test_valid_smpls`, `create_n_random_subsets`.

### `rsgispy.vector.stats`
- **Shape metrics**: `calc_poly_shape_metrics(vec_file, vec_lyr, metrics=None, col_prefix='', out_vec_file=None, out_vec_lyr=None, proj_epsg=None, allow_geographic=False, make_valid=False, *, out_format='GPKG', out_compress=None, overwrite=False)` and the vectorised chunk-level `calc_shape_metrics_geoseries(geoms, metrics=None, col_prefix='', make_valid=False) -> DataFrame`. `SHAPE_METRICS`: area, perimeter, n_parts, n_holes, hole_area, n_vertices, compactness (Polsby-Popper), shape_index, fractal_dim, convexity, solidity, rectangularity, mrr_length, mrr_width, elongation, orientation, roundness, equiv_diameter. Geographic CRS raises unless `proj_epsg` (EPSG or `'utm'`) or `allow_geographic=True`; empty/invalid/non-polygon geometries give NaN.
- `calc_vec_area`, `calc_vec_length`, `get_poly_hole_area`, `calc_vec_pt_dist_angle`, `calc_near_dist_to_feats`, `count_pt_intersects`, `calc_npts_in_radius(..., n_cores=1)`, `count_feats_per_att_val`, `calc_empirical_semivariogram`.

---

## 3. `rsgispy.zonalstats.*`

### Constants (`rsgispy.zonalstats`)
- Pixel-in-polygon methods (British spelling on `*CENTRE`):
  - `METHOD_POLYCONTAINSPIXEL = 0`
  - `METHOD_POLYCONTAINSPIXELCENTRE = 1`
  - `METHOD_POLYOVERLAPSPIXEL = 2`
  - `METHOD_POLYOVERLAPSORCONTAINSPIXEL = 3`
  - `METHOD_PIXELCONTAINSPOLY = 4`
  - `METHOD_PIXELCONTAINSPOLYCENTRE = 5`
  - `METHOD_ADAPTIVE = 6`
  - `METHOD_ENVELOPE = 7`
  - `METHOD_PIXELAREAINPOLY = 8`
  - `METHOD_POLYAREAINPIXEL = 9`

### Shared Helpers (`rsgispy.zonalstats`)
- `open_validated_zonal_inputs`, `open_validated_zonal_multiband_inputs`, `sample_point_pixel(src, img_band, x_coord, y_coord, clamp=False)`.

### Polygon Zonal Stats (`rsgispy.zonalstats.polygon`)
Each operation provides a file/in-memory entrypoint (`*_file`, returning `GeoDataFrame` if `out_vec_file=None`) and a `GeoDataFrame`-first variant without `_file`:
- `calc_zonal_band_stats_file(vec_file, vec_lyr, input_img, img_band, min_thres, max_thres, out_no_data_val, min_field=None, max_field=None, mean_field=None, stddev_field=None, sum_field=None, count_field=None, mode_field=None, median_field=None, vec_def_epsg=None, *, use_pbar=True, out_compress=None, out_vec_file=None, out_vec_lyr=None)` & `calc_zonal_band_stats(...)`
- `calc_zonal_poly_pts_band_stats_file(vec_file, vec_lyr, input_img, img_band, out_field, vec_def_epsg=None, *, ...)` & `calc_zonal_poly_pts_band_stats(...)` (samples centroid / representative point per polygon)
- `calc_zonal_band_stats_test_poly_pts_file(..., percentile=None, percentile_field=None, ...)` & `calc_zonal_band_stats_test_poly_pts(...)`
- `calc_zonal_band_pxl_counts_file(vec_file, vec_lyr, input_img, img_band, vals_lut: Dict[int, str], total_field='pxl_count', vec_def_epsg=None, *, ...)` & `calc_zonal_band_pxl_counts(...)`
- `export_poly_pxls_to_df(vec_gdf, input_img, img_bands=None, out_file=None, out_format=None, out_compress=None, band_names=None, attr_cols=None, mask_nodata=True, all_touched=False, include_coords=False, reproj_vec=False, vec_def_epsg=None, *, vec_lyr=None, use_pbar=True) -> pd.DataFrame | None`

### Label Raster Zonal Stats (`rsgispy.zonalstats.labels`, also exported from `rsgispy.zonalstats`)
One streamed pass over a label (segment/clump) raster; ~90x faster than polygon zonal stats at 40k zones; memory bounded by the number of labels. Value images must be on the label raster's grid.
- `calc_label_zonal_stats(label_img, band_specs, stats=None, percentiles=None, label_band=1, label_no_data_val=0, out_no_data_val=-9999.0, id_col='seg_id', col_pattern='{feat}_{stat}', quantile_method='auto', hist_bins=1000, categories=None, cat_props=True, pxl_count_col='pxl_count', inc_bbox=False, inc_centroid=False, dense_max_labels=50_000_000, *, block_size=None, out_file=None, out_format=None, out_compress='snappy', use_pbar=True) -> DataFrame` — `band_specs`: `BandSpec`/`ImageBandInfo`/paths (BandSpec gives `no_data_val`, `min_vld_val`, `max_vld_val`, `data_type`); stats: count, sum, mean, stddev, min, max, median, mode (+ percentiles). Categorical bands (`data_type=rsgispy.VAR_TYPE_CATEGORICAL`) give mode + `<feat>_prop_<cat>`; categories in `df.attrs['categories']`. uint64/sparse IDs use a compact mapping.
- `join_label_stats_to_vec(vec_file, stats_df, vec_lyr=None, id_col='seg_id', vec_id_col=None, out_vec_file=None, out_vec_lyr=None, *, out_format=None, out_compress=None, overwrite=False)`
- `rasterise_label_values(label_img, ids, values, output_img, label_band=1, label_no_data_val=0, out_no_data_val=0, band_names=None, thematic=False, clr_tab=None, dense_max_labels=..., *, gdalformat='GTIFF', datatype=None, calc_stats=True, block_size=None, use_pbar=True, creation_opts=None)` — label->value lookup, no geometry burning.

### Point Value Extraction (`rsgispy.zonalstats.point`)
- `ext_point_band_values_file(vec_file, vec_lyr, input_img, img_band: int | Sequence[int] | None, min_thres, max_thres, out_no_data_val, out_field: str | Sequence[str] | None = None, reproj_vec=False, vec_def_epsg=None, *, use_pbar=True, out_compress=None, out_vec_file=None, out_vec_lyr=None)` & `ext_point_band_values(...)`
  - Supports single band (`int`) or multi-band (`Sequence[int]` / `None` for all bands) in a single pass with `out_field` as matching `str` or `Sequence[str]`.
