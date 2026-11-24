"""
Targeted regression tests for ML, classification, regression, and modelling
code-review fixes.
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point
from sklearn.cluster import KMeans
from sklearn.ensemble import (
    GradientBoostingRegressor,
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold

import rsgispy
from rsgispy.classification.classdata import split_class_samples
from rsgispy.classification.clustersklearn import img_pixel_cluster
from rsgispy.imageutils.masking import ImageBandInfo
from rsgispy.ml.cv import nested_spatial_cv
from rsgispy.ml.predict import predict_quantiles_raster, predict_raster
from rsgispy.ml.stack import BandSpec, ImageFeatureStack
from rsgispy.ml.tune import tune_estimator
from rsgispy.modelling._helpers import resolve_analysis_vars
import rsgispy.modelling.species_distribution as sdm
from rsgispy.regression.regdata import split_reg_samples
from rsgispy.regression.regsklearn import (
    load_sklearn_regressor,
    save_sklearn_regressor,
)


def _write_gtiff(
    path: str,
    arr: np.ndarray,
    *,
    origin_x: float = 500000.0,
    origin_y: float = 5200000.0,
    res: float = 10.0,
    crs: str = "EPSG:32630",
    nodata: float = -9999.0,
) -> None:
    """Write a 2D or 3D NumPy array to a GeoTIFF."""
    data = arr[None, :, :] if arr.ndim == 2 else arr
    count, height, width = data.shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=str(data.dtype),
        crs=crs,
        transform=from_origin(origin_x, origin_y, res, res),
        nodata=nodata,
    ) as dst:
        dst.write(data)


def test_multitile_clustering_preserves_pixel_positions(tmp_path):
    """Verify img_pixel_cluster maps labels back to exact pixel positions across tiles."""
    height, width = 20, 20
    # Four quadrants with cleanly separated spectral values across multiple 8x8 tiles
    b1 = np.zeros((height, width), dtype=np.float32)
    b2 = np.zeros((height, width), dtype=np.float32)
    b1[:10, :10], b2[:10, :10] = 0.0, 0.0
    b1[:10, 10:], b2[:10, 10:] = 100.0, 0.0
    b1[10:, :10], b2[10:, :10] = 0.0, 100.0
    b1[10:, 10:], b2[10:, 10:] = 100.0, 100.0

    # Inject a nodata hole to ensure sparse pixel indexing stays aligned
    b1[3, 4] = -9999.0
    b2[15, 16] = -9999.0

    img_path = str(tmp_path / "quadrants.tif")
    out_single = str(tmp_path / "cluster_single.tif")
    out_multi = str(tmp_path / "cluster_multi.tif")
    _write_gtiff(img_path, np.stack([b1, b2], axis=0), nodata=-9999.0)
    band_info = [ImageBandInfo(file_name=img_path, name="quad", bands=[1, 2])]

    img_pixel_cluster(
        band_info,
        out_single,
        clusterer=KMeans(n_clusters=4, random_state=42, n_init=10),
        out_no_data_val=0,
        calc_stats=False,
        block_size=64,
        use_pbar=False,
    )
    img_pixel_cluster(
        band_info,
        out_multi,
        clusterer=KMeans(n_clusters=4, random_state=42, n_init=10),
        out_no_data_val=0,
        calc_stats=False,
        block_size=8,
        use_pbar=False,
    )

    with rasterio.open(out_single) as src1, rasterio.open(out_multi) as src2:
        arr_single = src1.read(1)
        arr_multi = src2.read(1)

    assert arr_multi[3, 4] == 0
    assert arr_multi[15, 16] == 0

    # Each quadrant (excluding nodata pixels) must have a single uniform cluster ID
    quad_slices = [
        (slice(0, 10), slice(0, 10)),
        (slice(0, 10), slice(10, 20)),
        (slice(10, 20), slice(0, 10)),
        (slice(10, 20), slice(10, 20)),
    ]
    quad_labels = []
    for r_sl, c_sl in quad_slices:
        vals = arr_multi[r_sl, c_sl]
        valid_vals = vals[vals != 0]
        uniq = np.unique(valid_vals)
        assert uniq.size == 1
        quad_labels.append(int(uniq[0]))
    assert len(set(quad_labels)) == 4
    # Single-tile and multi-tile partitions must agree up to label permutation
    for r_sl, c_sl in quad_slices:
        s_vals = arr_single[r_sl, c_sl]
        assert np.unique(s_vals[s_vals != 0]).size == 1


def test_duplicate_index_split_no_leakage():
    """Verify split_class_samples and split_reg_samples prevent leakage with duplicate indices."""
    n_rows = 40
    # Deliberately create non-unique index labels (e.g., from pd.concat without ignore_index)
    dup_index = [i % 5 for i in range(n_rows)]
    cls_df = pd.DataFrame(
        {
            "row_uid": np.arange(n_rows),
            "b1": np.linspace(0.0, 1.0, n_rows),
            "internal_cls_id": [0] * 20 + [1] * 20,
            "class_name": ["A"] * 20 + ["B"] * 20,
        },
        index=dup_index,
    )
    tr_cls, te_cls, va_cls = split_class_samples(
        cls_df, train_prop=0.6, test_prop=0.2, valid_prop=0.2, rnd_seed=42
    )
    assert len(tr_cls) + len(te_cls) + len(va_cls) == n_rows
    cls_sets = [set(tr_cls["row_uid"]), set(te_cls["row_uid"]), set(va_cls["row_uid"])]
    assert cls_sets[0].isdisjoint(cls_sets[1])
    assert cls_sets[0].isdisjoint(cls_sets[2])
    assert cls_sets[1].isdisjoint(cls_sets[2])
    assert cls_sets[0] | cls_sets[1] | cls_sets[2] == set(range(n_rows))

    reg_df = pd.DataFrame(
        {
            "row_uid": np.arange(n_rows),
            "b1": np.linspace(0.0, 10.0, n_rows),
            "target": np.linspace(1.0, 100.0, n_rows),
        },
        index=dup_index,
    )
    tr_reg, te_reg, va_reg = split_reg_samples(
        reg_df,
        train_prop=0.6,
        test_prop=0.2,
        valid_prop=0.2,
        stratify_col="target",
        n_bins=4,
        rnd_seed=42,
    )
    assert len(tr_reg) + len(te_reg) + len(va_reg) == n_rows
    reg_sets = [set(tr_reg["row_uid"]), set(te_reg["row_uid"]), set(va_reg["row_uid"])]
    assert reg_sets[0].isdisjoint(reg_sets[1])
    assert reg_sets[0].isdisjoint(reg_sets[2])
    assert reg_sets[1].isdisjoint(reg_sets[2])
    assert reg_sets[0] | reg_sets[1] | reg_sets[2] == set(range(n_rows))


def test_feature_stack_non_compatible_grid(tmp_path):
    """Verify ImageFeatureStack handles non-compatible rasters via on-the-fly reprojection."""
    b1_path = str(tmp_path / "fine.tif")
    b2_path = str(tmp_path / "coarse.tif")
    _write_gtiff(
        b1_path,
        np.full((12, 12), 10.0, dtype=np.float32),
        origin_x=500000.0,
        origin_y=5200000.0,
        res=10.0,
    )
    # Different resolution (20m) and shifted extent covering the same footprint
    _write_gtiff(
        b2_path,
        np.full((8, 8), 25.0, dtype=np.float32),
        origin_x=499990.0,
        origin_y=5200010.0,
        res=20.0,
    )

    stack = ImageFeatureStack(
        [
            BandSpec(file_name=b1_path, name="fine_b", bands=(1,)),
            BandSpec(file_name=b2_path, name="coarse_b", bands=(1,)),
        ]
    )
    n_valid = 0
    for _win, _grid, valid_idx, x_valid in stack.iter_blocks(
        block_size=6, use_pbar=False
    ):
        n_valid += int(valid_idx.size)
        if valid_idx.size > 0:
            assert np.allclose(x_valid[:, 0], 10.0)
            assert np.allclose(x_valid[:, 1], 25.0)
    assert n_valid == 144


def test_optuna_fixed_param_refit_and_verbosity_restore():
    """Verify tune_estimator preserves fixed callable params on refit and restores verbosity."""
    optuna = pytest.importorskip("optuna")
    orig_verbosity = optuna.logging.get_verbosity()
    optuna.logging.set_verbosity(optuna.logging.INFO)
    try:
        rng = np.random.default_rng(7)
        x_mat = rng.normal(size=(60, 3))
        y_vec = (x_mat[:, 0] + 0.5 * x_mat[:, 1] > 0.0).astype(int)

        def _space(trial):
            return {
                "C": trial.suggest_float("C", 0.1, 10.0, log=True),
                "max_iter": 250,
                "random_state": 17,
            }

        res = tune_estimator(
            lambda p: LogisticRegression(**p),
            x_mat,
            y_vec,
            _space,
            task="classification",
            cv=3,
            n_trials=3,
            rnd_seed=42,
            refit=True,
            use_pbar=False,
        )
        assert optuna.logging.get_verbosity() == optuna.logging.INFO
        assert res.best_params["max_iter"] == 250
        assert res.best_params["random_state"] == 17
        assert res.best_estimator is not None
        assert res.best_estimator.max_iter == 250
        assert res.best_estimator.random_state == 17
    finally:
        optuna.logging.set_verbosity(orig_verbosity)


def test_ml_predict_empty_first_tile_lut_promotion_and_gbr_quantile_guard(tmp_path):
    """
    Verify predict_raster handles an empty first tile, promotes LUT dtypes
    above uint8, and predict_quantiles_raster rejects GradientBoostingRegressor.
    """
    height, width = 16, 16
    b1 = np.full((height, width), 2.0, dtype=np.float32)
    b2 = np.full((height, width), 4.0, dtype=np.float32)
    mask = np.zeros((height, width), dtype=np.uint8)
    # Leave top-left 8x8 tile completely masked out (empty first tile)
    mask[8:, 8:] = 1

    feat_path = str(tmp_path / "feat.tif")
    msk_path = str(tmp_path / "msk.tif")
    _write_gtiff(feat_path, np.stack([b1, b2], axis=0), nodata=-9999.0)
    _write_gtiff(msk_path, mask, nodata=0)

    stack = ImageFeatureStack(
        [
            BandSpec(file_name=feat_path, name="b1", bands=(1,)),
            BandSpec(file_name=feat_path, name="b2", bands=(2,)),
        ]
    )

    # 1. Multi-output regressor with empty first tile (must infer 2 bands from later tile)
    rng = np.random.default_rng(12)
    x_tr = rng.uniform(0.0, 5.0, size=(30, 2))
    y_multi = np.column_stack([x_tr[:, 0] * 2.0, x_tr[:, 1] * 3.0])
    reg_multi = RandomForestRegressor(n_estimators=5, random_state=12).fit(x_tr, y_multi)

    out_multi = str(tmp_path / "pred_multi_empty_first.tif")
    predict_raster(
        reg_multi,
        stack,
        out_multi,
        in_msk_img=msk_path,
        img_msk_val=1,
        block_size=8,
        calc_stats=False,
        use_pbar=False,
    )
    with rasterio.open(out_multi) as src:
        assert src.count == 2
        arr = src.read()
        assert np.all(arr[:, :8, :8] == -9999.0)
        assert np.all(arr[:, 8:, 8:] > -9999.0)

    # 2. LUT dtype promotion: classifier predicts uint8 (0, 1), LUT maps to >255 (1000, 2000)
    y_cls = (x_tr[:, 0] > 2.5).astype(np.uint8)
    clf = RandomForestClassifier(n_estimators=5, random_state=12).fit(x_tr, y_cls)
    out_lut = str(tmp_path / "pred_lut_promoted.tif")
    predict_raster(
        clf,
        stack,
        out_lut,
        in_msk_img=msk_path,
        img_msk_val=1,
        lut={0: 1000, 1: 2000},
        datatype=rsgispy.TYPE_16UINT,
        no_data_val=0,
        block_size=8,
        calc_stats=False,
        use_pbar=False,
    )
    with rasterio.open(out_lut) as src:
        lut_arr = src.read(1)
        assert lut_arr.dtype == np.uint16
        assert np.all(lut_arr[:8, :8] == 0)
        assert set(np.unique(lut_arr[8:, 8:])).issubset({1000, 2000})

    # 3. GBR quantile guard: GradientBoostingRegressor must be rejected for empirical quantiles
    gbr = GradientBoostingRegressor(n_estimators=5, random_state=12).fit(x_tr, y_multi[:, 0])
    out_q = str(tmp_path / "gbr_quantiles.tif")
    with pytest.raises(rsgispy.RSGISPyException, match="GradientBoostingRegressor"):
        predict_quantiles_raster(
            gbr,
            stack,
            out_q,
            quantiles=(0.1, 0.5, 0.9),
            block_size=8,
            calc_stats=False,
            use_pbar=False,
        )


def test_nested_spatial_cv_missing_fold_class():
    """Verify nested_spatial_cv aligns oof_probs when a fold omits a global class."""
    rng = np.random.default_rng(21)
    # 3 classes: class 2 appears ONLY in group 0 (absent from training in fold 0)
    x_mat = rng.normal(size=(30, 3))
    y_vec = np.array([0] * 5 + [1] * 5 + [2] * 5 + [0] * 8 + [1] * 7)
    groups = np.array([0] * 15 + [1] * 15)

    res = nested_spatial_cv(
        lambda _p: RandomForestClassifier(n_estimators=10, random_state=21),
        x_mat,
        y_vec,
        task="classification",
        groups=groups,
        outer_cv=GroupKFold(n_splits=2),
        rnd_seed=21,
    )
    assert res.oof_probabilities is not None
    assert res.oof_probabilities.shape == (30, 3)
    assert np.all(np.isfinite(res.oof_probabilities))
    # Fold validating group 0 never saw class 2 in training, so col 2 must be 0.0
    assert np.all(res.oof_probabilities[groups == 0, 2] == 0.0)
    assert np.allclose(res.oof_probabilities.sum(axis=1), 1.0)


def test_modelling_valid_range_crs_alignment_resolve_vars_and_distinct_nodata(tmp_path):
    """
    Verify modelling valid-range filtering, mismatched dict-key/info.name resolution,
    CRS alignment in combine_presence_absence_data, k-fold non-duplicate class
    preservation, and distinct uint8 nodata=255 in class predictions.
    """
    height, width = 24, 24
    yy_raw, xx_raw = np.mgrid[0:height, 0:width]
    yy = yy_raw.astype(np.float32)
    xx = xx_raw.astype(np.float32)
    v1 = (50.0 + xx - yy).astype(np.float32)
    v2 = (20.0 + 0.5 * xx + 0.5 * yy).astype(np.float32)
    # Inject out-of-valid-range values inside ROI (finite and != raster nodata)
    v1[4, 4] = -5.0  # below min_vld_val=0.0
    v1[5, 5] = 200.0  # above max_vld_val=100.0

    mask = np.zeros((height, width), dtype=np.uint8)
    mask[2:22, 2:22] = 1

    cov_path = str(tmp_path / "cov.tif")
    msk_path = str(tmp_path / "msk.tif")
    _write_gtiff(cov_path, np.stack([v1, v2], axis=0), nodata=-9999.0)
    _write_gtiff(msk_path, mask, nodata=0)

    # Dict keys deliberately differ from EnvVarInfo.name
    env_vars = {
        "key_one": sdm.EnvVarInfo(
            name="VarOne",
            file=cov_path,
            band=1,
            data_type=rsgispy.VAR_TYPE_CONTINUOUS,
            min_vld_val=0.0,
            max_vld_val=100.0,
        ),
        "key_two": sdm.EnvVarInfo(
            name="VarTwo",
            file=cov_path,
            band=2,
            data_type=rsgispy.VAR_TYPE_CONTINUOUS,
        ),
    }

    # 1. resolve_analysis_vars works with None, info.name, and dict keys
    assert resolve_analysis_vars(env_vars) == ["VarOne", "VarTwo"]
    assert resolve_analysis_vars(env_vars, ["VarOne", "VarTwo"]) == ["VarOne", "VarTwo"]
    assert resolve_analysis_vars(env_vars, ["key_one", "key_two"]) == ["VarOne", "VarTwo"]

    # 2. create_finite_mask enforces min_vld_val and max_vld_val
    finite_msk = str(tmp_path / "finite_mask.tif")
    sdm.create_finite_mask(
        msk_path,
        1,
        env_vars,
        ["VarOne", "VarTwo"],
        finite_msk,
        calc_stats=False,
        use_pbar=False,
    )
    with rasterio.open(finite_msk) as src:
        fmask = src.read(1)
        assert fmask[4, 4] == 0
        assert fmask[5, 5] == 0
        assert fmask[10, 10] == 1

    # 3. combine_presence_absence_data reprojects absence_gdf when CRSs differ
    pres_pts = [Point(500000.0 + c * 10.0, 5200000.0 - r * 10.0) for c, r in zip(
        np.linspace(3.5, 9.5, 10), np.linspace(3.5, 9.5, 10)
    )]
    abs_pts = [Point(500000.0 + c * 10.0, 5200000.0 - r * 10.0) for c, r in zip(
        np.linspace(14.5, 20.5, 10), np.linspace(14.5, 20.5, 10)
    )]
    pres_gdf = gpd.GeoDataFrame(
        {"VarOne": np.linspace(60.0, 70.0, 10), "VarTwo": np.linspace(22.0, 26.0, 10)},
        geometry=pres_pts,
        crs="EPSG:32630",
    )
    abs_gdf_utm = gpd.GeoDataFrame(
        {"VarOne": np.linspace(30.0, 40.0, 10), "VarTwo": np.linspace(34.0, 38.0, 10)},
        geometry=abs_pts,
        crs="EPSG:32630",
    )
    abs_gdf_wgs84 = abs_gdf_utm.to_crs("EPSG:4326")

    comb_gdf, analysis_vars = sdm.combine_presence_absence_data(
        pres_gdf, None, abs_gdf_wgs84, None, env_vars
    )
    assert analysis_vars == ["VarOne", "VarTwo"]
    assert comb_gdf.crs == pres_gdf.crs
    # Reprojected absence coordinates must match original UTM coordinates
    abs_rows = comb_gdf[comb_gdf["clsid"] == 0].reset_index(drop=True)
    assert np.allclose(abs_rows.geometry.x.to_numpy(), abs_gdf_utm.geometry.x.to_numpy(), atol=1e-3)
    assert np.allclose(abs_rows.geometry.y.to_numpy(), abs_gdf_utm.geometry.y.to_numpy(), atol=1e-3)

    # Mismatched defined vs undefined CRS must raise RSGISPyException
    abs_no_crs = abs_gdf_utm.set_crs(None, allow_override=True)
    with pytest.raises(rsgispy.RSGISPyException, match="CRS"):
        sdm.combine_presence_absence_data(pres_gdf, None, abs_no_crs, None, env_vars)

    # 4. fit_kfold_sklearn_mdls validates fold size >= n_classes and avoids duplicates
    with pytest.raises(rsgispy.RSGISPyException, match="fold size"):
        sdm.fit_kfold_sklearn_mdls(
            RandomForestClassifier(n_estimators=5, random_state=1),
            comb_gdf,
            None,
            comb_gdf,
            None,
            analysis_vars,
            n_kfolds=2,
            fold_prop=0.05,  # 20 * 0.05 = 1 < 2 classes
            sel_replacement=False,
            rnd_seed=1,
            use_pbar=False,
        )

    # Direct check on class-preservation helper with extreme imbalance & no replacement
    train_y_imb = np.array([0] * 19 + [1])
    bad_draw = np.array([0, 1, 2, 3])
    fixed_idx = sdm._ensure_all_classes_in_fold(  # pylint: disable=protected-access
        bad_draw, train_y_imb, np.array([0, 1]), np.random.default_rng(0)
    )
    assert set(np.unique(train_y_imb[fixed_idx])) == {0, 1}
    assert len(np.unique(fixed_idx)) == len(fixed_idx) == len(bad_draw)

    # 5. Distinct nodata=255 and valid-range filtering in build_env_block_matrix / pred_*_cls
    clf, _, _, _ = sdm.fit_sklearn_mdl(
        RandomForestClassifier(n_estimators=10, random_state=3),
        comb_gdf,
        None,
        comb_gdf,
        None,
        analysis_vars,
        return_model=True,
    )
    oc_mdl, _, _ = sdm.fit_sklearn_sgl_cls_mdl(
        IsolationForest(n_estimators=10, contamination=0.1, random_state=3),
        comb_gdf[comb_gdf["clsid"] == 1],
        None,
        comb_gdf,
        None,
        analysis_vars,
        return_model=True,
    )

    cls_255 = str(tmp_path / "cls_nd255.tif")
    oc_cls_255 = str(tmp_path / "oc_cls_nd255.tif")
    # Pass raw msk_path (which does NOT mask [4,4] or [5,5]) to prove build_env_block_matrix
    # enforces min_vld_val/max_vld_val during prediction too!
    sdm.pred_sklearn_mdl_cls(
        clf,
        msk_path,
        1,
        env_vars,
        analysis_vars,
        cls_255,
        calc_stats=False,
        use_pbar=False,
        out_no_data_val=255,
    )
    sdm.pred_sklearn_sgl_cls_mdl_cls(
        oc_mdl,
        msk_path,
        1,
        env_vars,
        analysis_vars,
        oc_cls_255,
        calc_stats=False,
        use_pbar=False,
        out_no_data_val=255,
    )

    with rasterio.open(cls_255) as src:
        arr = src.read(1)
        assert arr.dtype == np.uint8
        assert src.nodata == 255
        assert arr[0, 0] == 255
        # Out-of-valid-range pixels inside ROI must also be marked nodata (255)
        assert arr[4, 4] == 255
        assert arr[5, 5] == 255
        assert arr[10, 10] in (0, 1)

    with rasterio.open(oc_cls_255) as src:
        oc_arr = src.read(1)
        assert oc_arr.dtype == np.uint8
        assert src.nodata == 255
        assert oc_arr[0, 0] == 255
        assert oc_arr[4, 4] == 255
        assert oc_arr[5, 5] == 255
        assert oc_arr[10, 10] in (0, 1)


def test_regression_model_save_load_roundtrip(tmp_path):
    """Verify sklearn regression models round-trip cleanly via save/load_sklearn_regressor."""
    rng = np.random.default_rng(99)
    x_tr = rng.normal(size=(40, 3))
    y_tr = 2.0 * x_tr[:, 0] - 0.5 * x_tr[:, 1] + 1.5
    reg = Ridge(alpha=0.5).fit(x_tr, y_tr)

    model_path = tmp_path / "subdir" / "ridge_reg.joblib"
    save_sklearn_regressor(reg, model_path)
    assert model_path.is_file()

    loaded = load_sklearn_regressor(model_path)
    assert np.allclose(loaded.predict(x_tr), reg.predict(x_tr))

    with pytest.raises(rsgispy.RSGISPyException):
        load_sklearn_regressor(tmp_path / "does_not_exist.joblib")
