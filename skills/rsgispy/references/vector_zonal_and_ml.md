# Vector, Zonal Stats, Machine Learning, Mapping & CLI Reference

All symbols below are verified against `rsgispy` v0.2.2+.

## 1. Vector Processing (`rsgispy.vector.*`)

Every function in `rsgispy.vector` accepts either a file path (`str | Path`) or a `geopandas.GeoDataFrame`. Pass `out_vec_file=None` to get a `GeoDataFrame` back in memory instead of writing to disk.

- **I/O & CRS (`rsgispy.vector.io`)**:
  - `get_vec_lyrs_lst(vec_file)` / `get_vec_lyr_cols(vec_file, vec_lyr=None)` / `get_vec_feat_count(...)` / `get_epsg_proj_from_vec(...)`
  - `reproj_vec_lyr(vec_file, out_vec_file=None, out_vec_lyr=None, epsg_code=None, ..., out_format="GPKG", vec_lyr=None)`
  - `merge_vec_files(vec_files, out_vec_file=None, out_vec_lyr=None, out_format="GPKG", out_epsg=None, ...)`
  - `vector_translate(vec_file, vec_lyr, out_vec_file=None, out_vec_lyr=None, out_format="GPKG", ...)`
- **Creation (`rsgispy.vector.create`)**:
  - `polygonise_raster_to_vec_lyr(out_vec_file, out_vec_lyr, out_format, input_img, img_band=1, mask_img=None, mask_band=1, ..., pxl_val_fieldname="PXLVAL")`
  - `extract_img_footprint(input_img, out_vec_file=None, out_vec_lyr=None, out_format="GPKG", ...)`
  - `create_hex_grid_poly(poly_series, hex_scale)` (requires `vector` extra for `h3`)
  - `define_grid(...)`, `create_poly_vec_bboxs(...)`, `create_random_pts_in_bbox(...)`
- **Geometries (`rsgispy.vector.geoms`)**:
  - `clip_vec_lyr(vec_file, vec_lyr, vec_roi_file, vec_roi_lyr, out_vec_file=None, out_vec_lyr=None, out_format="GPKG")`
  - `buffer_vec_layer_gp(vec_file, vec_lyr, buf_dist, out_vec_file=None, out_vec_lyr=None, out_format="GPKG", ...)`
  - `vec_lyr_intersection_gp`, `vec_lyr_difference_gp`, `vec_lyr_union_gp`, `simplify_geometries`, `delete_polygon_holes`, `calc_poly_centroids`, `explode_vec_lyr`
- **Attributes & `vector_maths` (`rsgispy.vector.attrs`)**:
  - `read_vec_columns(vec_file, vec_lyr=None, cols=None)`, `read_vec_column(...)`, `write_vec_column(...)`, `drop_vec_cols(...)`, `rename_vec_cols(...)`, `export_vec_attrs_to_csv(...)`, `export_vec_attrs_to_parquet(...)`
  - `vector_maths(vec_file, vec_lyr, out_col, exp, out_vec_file=None, out_vec_lyr=None, out_format="GPKG", vec_cols=None, out_dtype=None, del_exist_vec=False, out_compress=None)`:
    - Automatically binds all numeric/boolean columns with valid muParser names (`[A-Za-z_][A-Za-z0-9_]*`).
    - Pass `vec_cols={"alias": "Column With Spaces"}` to bind non-identifier column names.
    - Pass `out_col=["c1", "c2"]` and top-level comma-separated `exp="expr1, expr2"` for multi-column calculation in a single call; new intermediate variables (e.g., `ndvi = ..., ndvi >= 0.5 ? 1 : 0`) can be assigned and reused by later sub-expressions.
- **Stats & Selection (`rsgispy.vector.stats`, `selection`)**:
  - `calc_vec_area(vec_file, vec_lyr, out_col, out_vec_file=None, ...)`
  - `calc_vec_length(vec_file, vec_lyr, out_col, out_vec_file=None, ...)`
  - `calc_near_dist_to_feats(vec_file, vec_lyr, vec_dist_file, vec_dist_lyr, out_vec_file=None, ..., out_col="dist")`
  - `spatial_select(vec_file, vec_lyr, vec_roi_file, vec_roi_lyr, out_vec_file=None, ..., predicate="intersects")`
  - `subset_by_attribute(vec_file, vec_lyr, sub_col, sub_vals, out_vec_file=None, ..., match_type="equals")`
  - `create_train_test_smpls(...)`

## 2. Zonal Statistics (`rsgispy.zonalstats`)

### Polygon Zonal Stats (`rsgispy.zonalstats.polygon`)

```python
from rsgispy.zonalstats.polygon import calc_zonal_band_stats_file, export_poly_pxls_to_df

calc_zonal_band_stats_file(
    vec_file="polygons.gpkg",
    vec_lyr="polygons",
    input_img="ndvi.tif",
    img_band=1,
    min_thres=-1.0,
    max_thres=1.0,
    out_no_data_val=-9999.0,
    mean_field="ndvi_mean",
    stddev_field="ndvi_std",
    min_field="ndvi_min",
    max_field="ndvi_max",
    count_field="ndvi_cnt",
    out_vec_file="polygons_stats.gpkg",
    out_vec_lyr="polygons",
)
```

Use `calc_zonal_band_stats(vec_gdf, ...)` to work directly in memory, or `export_poly_pxls_to_df(...)` to extract per-pixel DataFrames within polygons.

### Point Extraction (`rsgispy.zonalstats.point`)

```python
from rsgispy.zonalstats.point import ext_point_band_values_file

ext_point_band_values_file(
    vec_file="sample_pts.gpkg",
    vec_lyr="sample_pts",
    input_img="s2_stack.tif",
    img_band=[1, 2, 3, 4],
    min_thres=0.0,
    max_thres=10000.0,
    out_no_data_val=-9999.0,
    out_field=["b1", "b2", "b3", "b4"],
    out_vec_file="sample_pts_vals.gpkg",
    out_vec_lyr="sample_pts",
)
```

## 3. Machine Learning, Classification & Regression

### Unified ML Core (`rsgispy.ml`)

- `BandSpec` & `ImageFeatureStack(specs, scaler=None)` (`rsgispy.ml.stack`)
- `FeatureScaler(method="zscore", ...)` (`rsgispy.ml.scaling`)
- `predict_raster(estimator, feature_stack, output_img, in_msk_img=None, img_msk_val=1, scaler=None, thematic=False, ..., gdalformat="GTIFF", datatype=None)` (`rsgispy.ml.predict`)

### Supervised Classification (`rsgispy.classification`)

Requires `classification` extra (plus `gbdt` for XGBoost/LightGBM/CatBoost).

```python
import rsgispy
from sklearn.ensemble import RandomForestClassifier
from rsgispy.imageutils.masking import ImageBandInfo
from rsgispy.classification.classdata import (
    ClassInfoObj,
    ClassificationConfig,
    extract_and_split_class_samples,
)
from rsgispy.classification.classsklearn import (
    train_sklearn_classifier,
    apply_sklearn_classifier,
)
from rsgispy.classification.classpostprocess import apply_majority_filter

cfg = ClassificationConfig(
    img_band_info=[ImageBandInfo(file_name="s2.tif", name="s2", bands=[1, 2, 3, 4], band_names=["B", "G", "R", "NIR"])],
    class_info={
        "Water": ClassInfoObj(out_id=1, vec_file="train.gpkg", vec_lyr="water", red=0, green=0, blue=255),
        "Forest": ClassInfoObj(out_id=2, vec_file="train.gpkg", vec_lyr="forest", red=0, green=150, blue=0),
    },
)
train_df, test_df, valid_df = extract_and_split_class_samples(cfg, test_prop=0.2, valid_prop=0.2, rnd_seed=42)
clf, metrics = train_sklearn_classifier(cfg, RandomForestClassifier(n_estimators=200, random_state=42), train_df, test_df, valid_df, return_model=True)
apply_sklearn_classifier(cfg, clf, in_msk_img="valid_msk.tif", img_msk_val=1, out_class_img="cls.tif", gdalformat="GTIFF", datatype=rsgispy.TYPE_8UINT)
apply_majority_filter("cls.tif", "cls_filt.tif", filter_size=3, gdalformat="GTIFF", datatype=rsgispy.TYPE_8UINT)
```

Accuracy assessment: `rsgispy.classification.classaccuracymetrics.calc_acc_stats_img(...)` / `calc_acc_stats_ptonly(...)`.

### Continuous Regression (`rsgispy.regression`)

- `RegressionConfig(img_band_info, target_cols, vec_file=..., vec_lyr=...)` (`rsgispy.regression.regdata`)
- `extract_and_split_reg_samples(...)` -> `train_sklearn_regressor(...)` -> `apply_sklearn_regressor(...)` (`rsgispy.regression.regsklearn`)
- `calc_reg_metrics(...)` (`rsgispy.regression.regmetrics`)

## 4. Cartographic Mapping (`rsgispy.mapping`)

```python
import matplotlib.pyplot as plt
import rsgispy
from rsgispy.mapping.raster import create_raster_img_map, create_thematic_raster_map

fig, ax = plt.subplots(figsize=(8, 8))
create_raster_img_map(
    ax=ax,
    input_img="s2_rgb.tif",
    img_bands=[3, 2, 1],
    img_stch=rsgispy.IMG_STRETCH_CUMULATIVE,
    title_str="Sentinel-2 True Colour",
    show_scale_bar=True,
    show_north_arrow=True,
)
fig.savefig("map.png", dpi=200, bbox_inches="tight")
```

## 5. CLI Tools (`rsgispy.cli`)

Common command-line tools installed with `rsgispy`:
- `rsgisimginfo` — print raster dimensions, CRS, bands, stats, nodata
- `rsgiscalcimgstats` — populate image statistics and overviews/pyramids
- `rsgisvectools` — inspect vector layers, columns, CRS, feature counts
- `rsgisbatchconvert2cog` — batch convert rasters to Cloud-Optimised GeoTIFFs
- `rsgisbuildimglut` — build spatial footprint LUT GeoPackage for raster directories
- `rsgischkgdalfile` / `rsgisfilehash` / `rsgisproj` / `rsgiswgs84imgres` / `rsgiscopybandnames` / `rsgisuserpassfile`
