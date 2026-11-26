# ML Pipeline Core, Classification, Regression & Species Distribution Modelling

Covers `rsgispy.ml.*`, `rsgispy.classification.*`, `rsgispy.regression.*`, and `rsgispy.modelling.species_distribution` in `rsgispy` v0.3.1.

---

## 1. Unified ML Core (`rsgispy.ml.*`)

### Feature Stacks (`rsgispy.ml.stack`)
- `BandSpec(file_name, name=None, bands=None, band_names=None, data_type=0, min_vld_val=None, max_vld_val=None, no_data_val=None)`
  - Methods: `.from_dict()`, `.to_dict()`, `.from_image_band_info(info)`, `.to_image_band_info()`, `.from_env_var_info(env_info)`.
- `ImageFeatureStack(specs, scaler=None)`
  - Constructors: `ImageFeatureStack.from_image_band_infos(img_band_info, scaler=None)`, `ImageFeatureStack.from_env_vars(env_vars, scaler=None)`, `ImageFeatureStack.from_dict()`, `ImageFeatureStack.load(json_file)`.
  - Methods: `.save(json_file)`, `.to_dict()`, `.to_image_band_infos()`, `.get_bbox()`, `.resolve_band_specs()`, `.resolve_band_nodata_list()`, `.create_valid_mask(output_img, ...)`, `.extract_points(vec_file, vec_lyr=None, keep_cols=None, drop_invalid=True, coord_cols=('x', 'y')) -> pd.DataFrame`, `.extract_polygons(vec_file, vec_lyr=None, keep_cols=None, poly_id_col='poly_fid', drop_invalid=True, coord_cols=('x', 'y'), all_touched=False) -> pd.DataFrame`, `.iter_blocks(...)`.

### Scaling (`rsgispy.ml.scaling`)
- `ALL_NORM_METHODS` (23 methods): `'none'`, `'linear'`, `'stdev'`, `'cumulative'`, `'user'`, `'zscore'`, `'robust_zscore'`, `'decimal_scaling'`, `'log'`, `'log1p'`, `'sigmoid'`, `'tanh'`, `'softmax'`, `'l1'`, `'l2'`, `'max_norm'`, `'standard_scaler'`, `'minmax_scaler'`, `'maxabs_scaler'`, `'robust_scaler'`, `'power_transformer'`, `'quantile_transformer'`, `'normalizer'`.
- `FeatureScaler(method='zscore', out_min=None, out_max=None, trim_to_limits=None, n_stdevs=2.0, lower_pct=2.0, upper_pct=98.0, in_min_max=None, method_kwargs=None, rnd_seed=42)` (scikit-learn `BaseEstimator` compatible: `.fit()`, `.transform()`, `.fit_transform()`, `.fit_from_stack(feature_stack, max_samples=500000, use_pbar=True)`, `.clone_unfitted()`, `.get_params()`, `.set_params()`, `.save(json_file)`, `.load(json_file)`).

### Spatial Cross-Validation (`rsgispy.ml.cv`)
- `SpatialBlockKFold(n_splits=5, block_size=None, method='grid', stratify=False, shuffle=True, rnd_seed=42)`
- `SpatialBufferCV(n_splits=5, buffer_dist=100.0, method='kmeans', block_size=None, rnd_seed=42)`
- `PolygonGroupKFold(n_splits=5, stratify=False, shuffle=True, rnd_seed=42)`
- `NestedCVResult` dataclass (`fold_scores`, `mean_score`, `std_score`, `oof_predictions`, `oof_probabilities`, `best_params_per_fold`, `fold_metrics`, `overall_metrics`).
- `nested_spatial_cv(estimator_factory, X, y, *, task='classification', coords=None, groups=None, outer_cv=None, inner_cv=None, param_space=None, scaler=None, scoring=None, n_trials=20, timeout=None, direction='maximize', rnd_seed=42, use_pbar=True) -> NestedCVResult`.

### Optuna Tuning (`rsgispy.ml.tune`)
- `TuneResult` dataclass (`best_params`, `best_score`, `best_estimator`, `best_scaler`, `study`, `trials_df`).
- `get_default_param_space(model_type, task='classification', extended=False, binary=True)`
- `suggest_params_from_space(trial, param_space)`
- `tune_estimator(estimator_factory, X, y, param_space, *, task='classification', cv=5, coords=None, groups=None, scaler=None, scoring=None, n_trials=30, timeout=None, direction='maximize', pruner=None, sampler=None, refit=True, rnd_seed=42, use_pbar=True) -> TuneResult`

### Tiled Raster Inference (`rsgispy.ml.predict`)
- `predict_raster(estimator, feature_stack, output_img, in_msk_img=None, img_msk_val=1, scaler=None, lut=None, threshold=None, thematic=False, band_names=None, block_size=None, *, gdalformat='GTIFF', datatype=None, no_data_val=None, calc_stats=True, use_pbar=True, creation_opts=None)`
- `predict_proba_raster(estimator, feature_stack, output_img, ..., class_names=None, out_class_img=None, out_entropy_img=None, lut=None, scale_to_pct=False, ...)`
- `predict_quantiles_raster(estimators_or_model, feature_stack, output_img, quantiles=(0.05, 0.5, 0.95), ..., conformal_residuals=None, alpha=0.1, ...)`
- **Tile-graph nodes** (chain features -> model -> majority filter in one `TileGraph` pass, no intermediate files; same pixels as the file functions):
  - `feature_stack_nodes(stack) -> List[TileNode]` (or `stack.to_tile_nodes()`): one node per `BandSpec`, with the stack's feature names, no-data and `min/max_vld_val` (out of range -> NaN). All rasters (and the mask) must be on the same pixel grid (the file functions also handle intersections/reprojection). The stack's scaler is not applied: pass `scaler=stack.scaler`.
  - `predict_raster_node(features, estimator, scaler=None, lut=None, threshold=None, thematic=False, band_names=None, *, mask=None, img_msk_val=1, n_outputs=None, no_data_val=None)` (no data default 0 if thematic/threshold else -9999).
  - `predict_proba_raster_node(features, estimator, scaler=None, class_names=None, lut=None, scale_to_pct=False, *, with_class=False, with_entropy=False, mask=None, ...)` (extra bands appended; split with `SelectBands`).
  - `predict_quantiles_raster_node(features, estimators_or_model, quantiles=(0.05, 0.5, 0.95), scaler=None, conformal_residuals=None, alpha=0.1, *, mask=None, ...)`.
  - `features` = any node(s): filters, indices, `BandMathOp`, `StackBands`; a pixel is predicted if inside `mask == img_msk_val` and all features are finite and != their node's no-data.

```python
import numpy as np
import rsgispy
from rsgispy.classification import majority_filter_node
from rsgispy.imageprocess import DiskSink, RasterSource, TileGraph
from rsgispy.ml.predict import feature_stack_nodes, predict_raster_node

cls = predict_raster_node(feature_stack_nodes(stack), clf, scaler=stack.scaler, thematic=True,
                          mask=RasterSource("valid_mask.tif", bands=1))
smooth = majority_filter_node(cls, 3, dtype=np.uint8, no_data_val=0)
TileGraph(DiskSink(smooth, "landcover.tif", datatype=rsgispy.TYPE_8UINT)).run()
```

### Chunked Vector I/O & Prediction (`rsgispy.ml.vector`)
- `read_vec_layer_info(vec_file, vec_lyr=None) -> VecLayerInfo`, `iter_vec_batches(vec_file, columns=None, vec_lyr=None, read_geometry=False, chunk_size=100_000)` (Arrow), `iter_vec_chunks(...)` (pandas, never geometry).
- `ChunkedVecWriter(out_file, out_vec_lyr=None, out_format=None, out_compress='snappy', geom_col='geometry', crs=None, geometry_type='Unknown', geo_metadata=None, metadata=None, overwrite=False)` — GeoParquet via pyarrow / OGR via pyogrio Arrow, WKB pass-through, temp file + atomic move, provenance metadata.
- `map_vector_chunks(in_vec_file, out_vec_file, read_cols, chunk_fn, *, in_vec_lyr=None, geom_vec_file=None, geom_vec_lyr=None, id_col=None, copy_cols=None, chunk_size=100_000, ...)`, `prepare_feature_matrix(df, feature_cols, valid_col=None, missing='nan'|'drop'|'fill', fill_value=0.0, feat_no_data_val=None, scaler=None)`, `predict_vector(estimator, in_vec_file, out_vec_file, feature_cols, *, out_col='prediction', ...)`, `build_provenance(...)`.

### Segment Features & Config (`rsgispy.ml.segments`)
- `SegmentFeatureConfig` (stored as `seg_info` in `ClassificationConfig`/`RegressionConfig` JSON; omitted when `None`): label_img, seg_vec_file, feat_table_file, band_specs, id_col='seg_id', valid_col='valid', stats, percentiles, shape_metrics, categories, feature_cols, selected_features (`.model_features`), feat_no_data_val=-9999, min_seg_pxls, missing='drop' ('nan' for GBDT, 'fill'), poly_assign='overlap'|'centroid', min_overlap=0.5, conflict='drop'|'majority', reg_agg='mean'|'median'.
- `create_segment_features(label_img, band_specs, out_vec_file, out_vec_lyr=None, out_table_file=None, vec_inc_features=True, stats=None, percentiles=None, shape_metrics=None, shape_proj_epsg=None, band_opts=None, template=None, *, ..., chunk_size=100_000, tile_size=2048, n_jobs=1, use_pbar=True) -> SegmentFeatureConfig` — columns `<feat>_<stat>`, `shp_<metric>`, `pxl_count`, `valid`; GeoParquet default. Segment polygons come from the tiled `polygonise_raster_tiled` (`n_jobs` cores) and are streamed/joined chunk by chunk (not held in memory).
- `substitute_scene(cfg, img_files: {name: path}, label_img=None, seg_vec_file=None, ...)` + `create_segment_features(..., template=new_cfg.seg_info)` to apply a model to a new scene.
- `assign_vectors_to_segments`, `read_segment_features`, `apply_missing_policy`.
- With `seg_info`, training functions default `feature_cols` to `seg_info.model_features` and accept NaN features when `missing='nan'`.

---

## 2. `rsgispy.classification.*`

### Config & Training Samples (`rsgispy.classification.classdata`)
- `ImageBandInfo` (from `imageutils.masking`): `(file_name, name=None, bands=None, band_names=None)`
- `ClassInfoObj(out_id, vec_file=None, vec_lyr=None, red=None, green=None, blue=None, internal_id=None, name=None, colour=None)`
- `ClassificationConfig(img_band_info, class_info, out_id_col='out_cls_id', internal_id_col='internal_cls_id', class_name_col='class_name', seg_info=None)` (`.save()`, `.load()`, `.to_feature_stack()`, `.get_colour_table()`, `.get_class_names_lut()`, `.get_internal_to_out_lut()`, `.get_out_to_internal_lut()`).
- Functions: `normalise_class_info`, `resolve_img_band_info`, `write_cls_config`, `read_cls_config`, `write_cls_samples`, `read_cls_samples`, `extract_class_training_samples(..., *, poly_id_col=None)`, `split_class_samples(..., *, group_col=None)`, `extract_and_split_class_samples(..., *, poly_id_col=None, group_col=None)` (`group_col='train_grp'` records `'<class>:poly<n>'` IDs and sends whole polygons to train/test/valid, stratified by class), `get_cls_id_luts`, `recode_internal_to_out_ids`, `recode_cls_img_to_out_ids`.

### Classifiers (`classsklearn`, `classxgboost`, `classlightgbm`, `classcatboost`)
- **scikit-learn (`classsklearn`)**: `get_sklearn_classifier_param_space`, `save_sklearn_classifier`, `load_sklearn_classifier`, `perform_sklearn_classifier_param_search(..., *, groups=None, coords=None)` (pass a column name / labels for `groups` and `('x', 'y')` or an (n, 2) array for `coords` so `rsgispy.ml.cv` splitters such as `SpatialBlockKFold` / `PolygonGroupKFold` can be the `cv`), `train_sklearn_classifier`, `apply_sklearn_classifier`.
- **GBDTs (`class{xgboost,lightgbm,catboost}`)**: each provides:
  - `is_{xgboost,lightgbm,catboost}_gpu_available`
  - `save_*_classifier`, `load_*_classifier`
  - `get_*_optuna_param_space(extended=False, ...)`
  - `optimise_*_binary_classifier` & `optimise_*_multiclass_classifier` (note British spelling: `optimise_*`)
  - `train_*_binary_classifier` & `train_*_multiclass_classifier`
  - `apply_*_binary_classifier` & `apply_*_multiclass_classifier`

### Object-based / Segment Classification (`classsegments`)
- `create_cls_segment_features(cls_cfg, label_img, out_vec_file, out_cfg_file=None, **create_segment_features_kwargs) -> ClassificationConfig`
- `extract_class_segment_samples(cls_cfg, out_file=None, poly_assign=None, min_overlap=None, conflict=None, min_seg_pxls=None, reproj_vec=False, vec_def_epsg=None, *, seg_info=None, ...)` — one sample per segment, columns out_cls_id/internal_cls_id/class_name/seg_id/train_grp/n_hits + features; `df.attrs['report']`.
- `split_class_segment_samples(samples_df, train_prop=None, test_prop=0.2, valid_prop=0.2, ..., group_col='train_grp', rnd_seed=None)` (group-aware, stratified), `extract_and_split_class_segment_samples(cls_cfg, out_train_file, out_test_file, out_valid_file, ...)`.
- Train with the existing `train_*`/`optimise_*`/`perform_sklearn_classifier_param_search` (pass the config + segment sample tables).
- `apply_sklearn_classifier_segments(cls_cfg, sk_classifier, out_vec_file, out_probs=None|'all'|'max'|'pos', **kw)`, `apply_{xgboost,lightgbm,catboost}_{binary,multiclass}_classifier_segments(cls_cfg, classifier, out_vec_file, out_probs=None, pos_class=None, pred_threshold=None, *, use_gpu=False, ...)` — chunked (`chunk_size`), reads only ID/valid/features, writes `cls_id`, `cls_name`, `cls_colour`, `cls_red/green/blue`, `cls_prob_*`; `copy_cols=[]` keeps only seg_id.
- `rasterise_segment_classification(cls_cfg_or_None, cls_vec_file, output_img, cls_col='cls_id', out_score_img=None, score_cols=None, label_img=None, clr_lut=None, *, ...)`, `cluster_segments(cfg_or_seg_info, clusterer, out_vec_file, n_sample=None, scaler=None, out_col='cluster_id', ...)`, `pop_segment_cls_accuracy_pts(cls_cfg, cls_vec_file, acc_pts_vec_file, ...)` (then `calc_acc_stats_ptonly` / `calc_acc_stats_img`).

### Unsupervised Clustering (`clustersklearn`)
- `img_pixel_cluster(img_band_info, output_img, clusterer=None, in_msk_img=None, img_msk_val=1, no_data_val=None, *, gdalformat='GTIFF', datatype=None, out_no_data_val=None, out_id_offset=1, use_mean_shift_est_band_width=False, ...)`
- `img_pixel_sample_cluster(..., n_img_smpl=100, max_samples=None, ...)` (fits on random sample, predicts tiled across raster).

### Feature Analysis, Accuracy, Reference Points & Post-Processing
- **`classfeatures`**: `comparison_box_plots`, `correlation_matrix`, `calc_vif_multicollinearity`, `select_features_cqv`, `select_features_corr`, `select_features_vif`, `select_features_univariate`, `select_features_rfe`, `select_features_from_model`, `select_features_boruta_shap`, `calc_feature_importance`, `calc_jackknife_feature_importance`, `calc_feature_response_curves`, `calc_shap_feature_contributions`, `calc_salib_feature_sensitivity`.
- **`classrefpts`**: `create_random_ref_smpls_darts`, `create_stratified_random_ref_smpls_darts`, `create_stratified_random_ref_smpls_lut`, `pop_class_info_accuracy_pts`.
- **`classaccuracymetrics`**: `calc_acc_stats_img`, `calc_acc_stats_ptonly`, `calc_acc_stats_img_bootstrap_conf`, `calc_acc_stats_ptonly_bootstrap_conf`.
- **`classpostprocess`** (also top-level `rsgispy.classification.{apply_majority_filter,apply_sieve_filter,majority_filter_node}`): `apply_majority_filter(..., filter_size=3, img_band=1, keep_centre_on_tie=True, copy_colour_table=True, *)`, `apply_sieve_filter(..., size_threshold, connectivity=4, img_band=1, ...)`, `apply_postprocess_pipeline(..., majority_sizes=(), sieve_size_threshold=None, ...)`; tile-graph node `majority_filter_node(source, filter_size=3, keep_centre_on_tie=True, *, dtype=None, no_data_val=None)` (filters every band of `source`; `no_data_val=None` = the source node's no-data value; pass the output dtype). The sieve is whole-image (connected components) and has no node.

---

## 3. `rsgispy.regression.*`

- **`regdata`**: `RegressionConfig(img_band_info, target_cols, vec_file=None, vec_lyr=None, seg_info=None)`, `normalise_target_cols`, `write_reg_config`, `read_reg_config`, `write_reg_samples`, `read_reg_samples`, `extract_reg_training_samples(..., *, poly_id_col=None)`, `split_reg_samples(..., *, group_col=None)`, `extract_and_split_reg_samples(..., *, poly_id_col=None, group_col=None)`; `regsklearn.perform_sklearn_regressor_param_search(..., *, groups=None, coords=None)`.
- **`regsklearn`**: `get_sklearn_regressor_param_space`, `save_sklearn_regressor`, `load_sklearn_regressor`, `perform_sklearn_regressor_param_search`, `train_sklearn_regressor`, `apply_sklearn_regressor`.
- **GBDTs (`reg{xgboost,lightgbm,catboost}`)**: each provides `is_*_gpu_available`, `save_*_regressor`, `load_*_regressor`, `get_*_optuna_param_space`, `optimise_*_regressor`, `train_*_regressor`, `apply_*_regressor`.
- **`regfeatures`**: `target_scatter_plots`, `correlation_matrix`, `calc_vif_multicollinearity`, `select_features_{cqv,corr,vif,univariate,rfe,from_model,boruta_shap}`, `calc_feature_importance`, `calc_jackknife_feature_importance`, `calc_feature_response_curves`, `calc_shap_feature_contributions`, `calc_salib_feature_sensitivity`.
- **`regsegments`** (object-based): `create_reg_segment_features`, `extract_reg_segment_samples(reg_cfg, out_file=None, vec_file=None, ..., reg_agg=None)` (mean/median per segment, `n_ref`), `split_reg_segment_samples`, `extract_and_split_reg_segment_samples`, `apply_sklearn_regressor_segments(reg_cfg, sk_regressor, out_vec_file, out_quantiles=None, **kw)` (`pred_<target>`, tree-ensemble quantiles), `apply_{xgboost,lightgbm,catboost}_regressor_segments`, `rasterise_segment_regression(reg_cfg, reg_vec_file, output_img, pred_cols=None, ...)`.
- **`regmetrics`**: `calc_reg_metrics`, `pop_reg_img_pred_pts`, `calc_reg_stats_ptonly_bootstrap_conf`, `plot_pred_vs_obs`, `plot_residuals`.

---

## 4. `rsgispy.modelling.species_distribution`

- `EnvVarInfo(name=None, file=None, band=1, data_type=0, min_vld_val=None, max_vld_val=None, norm_mean=None, norm_std=None)` (`.to_band_spec()`, `.set_norm_mean_std()`)
- **Sampling & prep**: `create_finite_mask`, `gen_pseudo_absences_smpls`, `extract_env_var_data`, `combine_presence_absence_data(..., equalise_smpls=False, cls_col='clsid')`, `create_train_test_sets`, `pop_normalise_coeffs`, `apply_normalise_coeffs`.
- **Exploratory / collinearity**: `comparison_box_plots`, `correlation_matrix`, `calc_vif_multicollinearity`.
- **Model fitting & XAI**:
  - `search_mdl_params`, `fit_sklearn_mdl`, `fit_kfold_sklearn_mdls`
  - **Single-class SDM naming** (note `sgl_cls`): `fit_sklearn_sgl_cls_mdl`, `fit_kfold_sklearn_sgl_cls_mdls`
  - `shap_sklearn_mdl_explainer`, `salib_sklearn_mdl_sensitivity`, `sklearn_mdl_variable_response_curves`, `sklearn_jackknife_var_importance`
- **Evaluation & thresholds**: `calc_binary_roc_auc(y_true, scores) -> (fpr, tpr, auc)`, `calc_binary_threshold(y_true, scores, *, method='max_tss')` (`'sens_eq_spec'` also available), `plot_roc_curve(fpr, tpr, roc_auc, out_file=None, *, label='ROC Curve', title=..., dpi=150)`, `get_positive_class_probs(est_cls_obj, x_data, pos_label=1)`, `calc_single_class_predictions(est_cls_obj, x_data)`.
- **Raster prediction with a custom function**: `predict_raster_from_env_vars(in_msk_img, img_msk_val, env_vars, analysis_vars, output_img, predict_fn, *, gdalformat='GTIFF', datatype=None, no_data_val=-1.0, fill_val=None, normalise_data=False, thematic=False, calc_stats=True, block_size=None, use_pbar=True)`.
- **Prediction**: `pred_sklearn_mdl_prob`, `pred_sklearn_mdl_cls`, `pred_sklearn_sgl_cls_mdl_prob`, `pred_sklearn_sgl_cls_mdl_cls`, `pred_ensemble_sklearn_mdls_prob`, `pred_ensemble_sklearn_mdls_cls`, `pred_ensemble_sklearn_sgl_cls_mdls_prob`, `pred_ensemble_sklearn_sgl_cls_mdls_cls`.
