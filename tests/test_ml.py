"""Tests for the unified rsgispy.ml subpackage."""

import os

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from scipy.spatial.distance import cdist
from shapely.geometry import Point, box
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge

import rsgispy
import rsgispy.ml as ml
from rsgispy.classification.classdata import (
    ClassInfoObj,
    ClassificationConfig,
    ImageBandInfo,
)
from rsgispy.ml.cv import (
    NestedCVResult,
    PolygonGroupKFold,
    SpatialBlockKFold,
    SpatialBufferCV,
    nested_spatial_cv,
)
from rsgispy.ml.predict import (
    predict_proba_raster,
    predict_quantiles_raster,
    predict_raster,
)
from rsgispy.ml.scaling import FeatureScaler
from rsgispy.ml.stack import BandSpec, ImageFeatureStack
from rsgispy.ml.tune import TuneResult, get_default_param_space, tune_estimator
from rsgispy.modelling._helpers import EnvVarInfo
from rsgispy.regression.regdata import RegressionConfig


def _write_test_raster(
    path: str,
    data: np.ndarray,
    nodata: float = -9999.0,
    x_origin: float = 500000.0,
    y_origin: float = 5800000.0,
    px_size: float = 10.0,
    epsg: int = 32630,
) -> str:
    """Create a multi-band float32 GeoTIFF for ML tests."""
    if data.ndim == 2:
        data = data[np.newaxis, :, :]
    bands, height, width = data.shape
    transform = from_origin(x_origin, y_origin, px_size, px_size)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=bands,
        dtype=str(data.dtype),
        crs=f"EPSG:{epsg}",
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data)
    return path


@pytest.fixture
def synthetic_stack_files(tmp_path):
    """Create a 2-image synthetic feature stack (3 bands total) with nodata pixels."""
    rng = np.random.default_rng(123)
    h, w = 24, 24
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    b1 = (xx * 2.0 + rng.normal(0.0, 0.5, size=(h, w))).astype(np.float32)
    b2 = (yy * 1.5 + rng.normal(0.0, 0.5, size=(h, w))).astype(np.float32)
    b3 = (0.5 * xx + 0.5 * yy + 10.0 + rng.normal(0.0, 0.2, size=(h, w))).astype(
        np.float32
    )

    # Inject nodata and NaN at known corners
    b1[0, 0] = -9999.0
    b2[0, 1] = np.nan
    b3[h - 1, w - 1] = -9999.0

    img1 = _write_test_raster(
        str(tmp_path / "feat1.tif"), np.stack([b1, b2], axis=0), nodata=-9999.0
    )
    img2 = _write_test_raster(
        str(tmp_path / "feat2.tif"), b3[np.newaxis, :, :], nodata=-9999.0
    )

    # Create a mask raster (1 everywhere except bottom row)
    msk = np.ones((h, w), dtype=np.uint8)
    msk[h - 1, :] = 0
    msk_img = _write_test_raster(
        str(tmp_path / "mask.tif"), msk, nodata=0
    )

    return {
        "img1": img1,
        "img2": img2,
        "msk_img": msk_img,
        "arrays": (b1, b2, b3),
        "shape": (h, w),
    }


def test_ml_package_exports_submodules_only():
    """rsgispy.ml exports only its 7 submodules."""
    assert ml.__all__ == [
        "stack", "scaling", "predict", "cv", "tune", "vector", "segments"
    ]
    for sub in ml.__all__:
        assert hasattr(ml, sub)


def test_feature_stack_construct_bbox_mask_and_interop(synthetic_stack_files, tmp_path):
    """Test BandSpec, ImageFeatureStack, BBox, valid mask, JSON round-trip, and interop."""
    img1 = synthetic_stack_files["img1"]
    img2 = synthetic_stack_files["img2"]

    stack = ImageFeatureStack(
        [
            BandSpec(img1, name="s2", bands=(1, 2), band_names=("red", "nir")),
            BandSpec(img2, name="dem", bands=(1,), band_names=("elev",)),
        ]
    )
    assert stack.n_features == 3
    assert stack.feature_names == ["red", "nir", "elev"]

    bbox = stack.get_bbox()
    assert isinstance(bbox, rsgispy.BBox)
    assert bbox.epsg == 32630
    assert bbox.west == pytest.approx(500000.0)
    assert bbox.east == pytest.approx(500240.0)

    # Valid mask generation
    vld_mask_path = str(tmp_path / "valid_mask.tif")
    stack.create_valid_mask(vld_mask_path, block_size=8, use_pbar=False)
    with rasterio.open(vld_mask_path) as src:
        vld = src.read(1)
    assert vld.shape == (24, 24)
    assert vld[0, 0] == 0
    assert vld[0, 1] == 0
    assert vld[23, 23] == 0
    assert vld[12, 12] == 1

    # JSON save/load round-trip
    stack_json = str(tmp_path / "stack.json")
    stack.save(stack_json)
    loaded = ImageFeatureStack.load(stack_json)
    assert loaded.feature_names == stack.feature_names
    assert loaded.n_features == 3

    # Interop with ImageBandInfo, ClassificationConfig, RegressionConfig, EnvVarInfo
    ibis = stack.to_image_band_infos()
    assert len(ibis) == 2
    assert isinstance(ibis[0], ImageBandInfo)
    stack_from_ibi = ImageFeatureStack.from_image_band_infos(ibis)
    assert stack_from_ibi.n_features == 3

    cls_cfg = ClassificationConfig(
        img_band_info=ibis,
        class_info={"forest": ClassInfoObj(out_id=1, name="forest", colour="#228B22")},
    )
    assert cls_cfg.to_feature_stack().n_features == 3

    reg_cfg = RegressionConfig(img_band_info=ibis, target_cols="agb")
    assert reg_cfg.to_feature_stack().n_features == 3

    ev = EnvVarInfo(name="elev", file=img2, band=1)
    bs = ev.to_band_spec()
    assert isinstance(bs, BandSpec)
    assert bs.bands == (1,)
    stack_ev = ImageFeatureStack.from_env_vars([ev])
    assert stack_ev.feature_names == ["elev"]


def test_feature_stack_extract_points_and_polygons(synthetic_stack_files):
    """Test point and polygon extraction with coordinates and polygon FIDs."""
    img1 = synthetic_stack_files["img1"]
    img2 = synthetic_stack_files["img2"]
    stack = ImageFeatureStack(
        [
            BandSpec(img1, name="s2", bands=(1, 2), band_names=("red", "nir")),
            BandSpec(img2, name="dem", bands=(1,), band_names=("elev",)),
        ]
    )

    # Points inside valid region plus one point on nodata (0, 0)
    pts = gpd.GeoDataFrame(
        {
            "label": [1, 2, 1, 2],
            "geometry": [
                Point(500005.0, 5799995.0),  # (row 0, col 0) -> nodata!
                Point(500055.0, 5799945.0),  # (row 5, col 5)
                Point(500125.0, 5799875.0),  # (row 12, col 12)
                Point(500185.0, 5799815.0),  # (row 18, col 18)
            ],
        },
        crs="EPSG:32630",
    )
    pt_df = stack.extract_points(pts, keep_cols=["label"], drop_invalid=True)
    assert len(pt_df) == 3
    assert set(["red", "nir", "elev", "label", "x", "y"]).issubset(pt_df.columns)

    # Polygons covering disjoint patches
    polys = gpd.GeoDataFrame(
        {
            "cls": [1, 2],
            "geometry": [
                box(500020.0, 5799940.0, 500060.0, 5799980.0),
                box(500120.0, 5799840.0, 500160.0, 5799880.0),
            ],
        },
        crs="EPSG:32630",
    )
    poly_df = stack.extract_polygons(polys, keep_cols=["cls"], poly_id_col="poly_fid")
    assert len(poly_df) > 10
    assert set(poly_df["poly_fid"].unique()) == {0, 1}
    assert set(["red", "nir", "elev", "cls", "x", "y", "poly_fid"]).issubset(
        poly_df.columns
    )


def test_feature_stack_mixed_dtype_vrt(synthetic_stack_files, tmp_path):
    """Points and polygons from a VRT mixing uint16 and float32 bands."""
    from rsgispy.imageutils.vrt import create_stack_images_vrt

    h, w = synthetic_stack_files["shape"]
    refl = (np.arange(h * w, dtype=np.uint16).reshape(h, w) + 1).astype(np.uint16)
    refl[5, 5] = 0
    refl_img = _write_test_raster(str(tmp_path / "refl.tif"), refl, nodata=0)
    vrt = str(tmp_path / "mixed.vrt")
    create_stack_images_vrt([refl_img, synthetic_stack_files["img2"]], vrt)
    names = ("refl", "elev")
    vrt_stack = ImageFeatureStack([BandSpec(vrt, name="v", band_names=names)])
    sep_stack = ImageFeatureStack(
        [
            BandSpec(refl_img, name="r", band_names=("refl",)),
            BandSpec(synthetic_stack_files["img2"], name="e", band_names=("elev",)),
        ]
    )
    pts = gpd.GeoDataFrame(
        geometry=[
            Point(500055.0, 5799945.0),  # (row 5, col 5) -> refl no data
            Point(500125.0, 5799875.0),
            Point(500235.0, 5799765.0),  # (row 23, col 23) -> elev no data
            Point(500185.0, 5799815.0),
        ],
        crs="EPSG:32630",
    )
    pt_vrt = vrt_stack.extract_points(pts, keep_cols=[])
    pt_sep = sep_stack.extract_points(pts, keep_cols=[])
    assert len(pt_vrt) == 2
    pd.testing.assert_frame_equal(pt_vrt, pt_sep)

    polys = gpd.GeoDataFrame(
        geometry=[box(500020.0, 5799940.0, 500060.0, 5799980.0)], crs="EPSG:32630"
    )
    pd.testing.assert_frame_equal(
        vrt_stack.extract_polygons(polys, keep_cols=[]),
        sep_stack.extract_polygons(polys, keep_cols=[]),
    )


@pytest.mark.parametrize(
    "method",
    [
        "zscore",
        "robust_zscore",
        "linear",
        "stdev",
        "cumulative",
        "standard_scaler",
        "minmax_scaler",
        "maxabs_scaler",
        "robust_scaler",
        "power_transformer",
        "quantile_transformer",
        "log1p",
        "sigmoid",
        "tanh",
        "decimal_scaling",
        "l1",
        "l2",
        "max_norm",
        "softmax",
    ],
)
def test_feature_scaler_methods_and_persistence(method, tmp_path):
    """FeatureScaler supports all scaling/normalisation methods and JSON round-trip."""
    rng = np.random.default_rng(7)
    x_arr = np.abs(rng.normal(loc=50.0, scale=15.0, size=(80, 3))) + 1.0
    x_df = pd.DataFrame(x_arr, columns=["b1", "b2", "b3"])

    kwargs = {"method_kwargs": {"n_quantiles": 20}} if method == "quantile_transformer" else {}
    scaler = FeatureScaler(method=method, **kwargs)
    out_arr = scaler.fit_transform(x_arr)
    assert isinstance(out_arr, np.ndarray)
    assert out_arr.shape == x_arr.shape
    assert np.all(np.isfinite(out_arr))

    # DataFrame preservation
    out_df = scaler.transform(x_df)
    assert isinstance(out_df, pd.DataFrame)
    assert list(out_df.columns) == ["b1", "b2", "b3"]
    np.testing.assert_allclose(out_df.to_numpy(), out_arr, atol=1e-6)

    # JSON persistence round-trip
    json_path = str(tmp_path / f"scaler_{method}.json")
    scaler.save(json_path)
    reloaded = FeatureScaler.load(json_path)
    out_reloaded = reloaded.transform(x_arr)
    np.testing.assert_allclose(out_reloaded, out_arr, atol=1e-5)

    # Clone & params
    cloned = scaler.clone_unfitted()
    assert not cloned.is_fitted
    assert cloned.get_params()["method"] == method


def test_feature_scaler_fit_from_stack(synthetic_stack_files):
    """FeatureScaler.fit_from_stack streams valid pixels for band-wise and vector methods."""
    stack = ImageFeatureStack(
        [
            BandSpec(synthetic_stack_files["img1"], bands=(1, 2)),
            BandSpec(synthetic_stack_files["img2"], bands=(1,)),
        ]
    )
    sc_z = FeatureScaler(method="zscore").fit_from_stack(stack, use_pbar=False)
    assert sc_z.is_fitted
    assert sc_z.n_features_in_ == 3

    # Verify iter_blocks applies scaler on the fly
    scaled_chunks = []
    for _, _, _, x_blk in stack.iter_blocks(
        block_size=8, scaler=sc_z, apply_scaler=True, use_pbar=False
    ):
        scaled_chunks.append(x_blk)
    all_scaled = np.vstack(scaled_chunks)
    assert np.all(np.isfinite(all_scaled))
    assert np.abs(np.mean(all_scaled[:, 0])) < 0.1

    # Pixel-vector method via fit_from_stack
    sc_l2 = FeatureScaler(method="l2").fit_from_stack(stack, use_pbar=False)
    assert sc_l2.is_fitted
    sample = np.array([[3.0, 4.0, 0.0]])
    np.testing.assert_allclose(sc_l2.transform(sample), [[0.6, 0.8, 0.0]], atol=1e-6)


def test_predict_raster_proba_and_quantiles(synthetic_stack_files, tmp_path):
    """Test tiled classification, probability/entropy, regression, and quantile prediction."""
    b1, b2, b3 = synthetic_stack_files["arrays"]
    stack = ImageFeatureStack(
        [
            BandSpec(synthetic_stack_files["img1"], bands=(1, 2)),
            BandSpec(synthetic_stack_files["img2"], bands=(1,)),
        ]
    )

    # Build clean training data from valid pixels
    valid = np.isfinite(b1) & (b1 != -9999.0) & np.isfinite(b2) & (b3 != -9999.0)
    X = np.column_stack([b1[valid], b2[valid], b3[valid]])
    y_cls = np.where(X[:, 0] + X[:, 1] > 40.0, 1, 0)
    y_reg = 0.3 * X[:, 0] - 0.2 * X[:, 1] + 0.5 * X[:, 2]

    scaler = FeatureScaler(method="standard_scaler").fit(X)
    X_s = scaler.transform(X)

    clf = RandomForestClassifier(n_estimators=20, max_depth=4, random_state=42).fit(
        X_s, y_cls
    )

    # 1) Classification predict_raster with LUT and on-the-fly mask (in_msk_img=None)
    cls_out = str(tmp_path / "cls_pred.tif")
    predict_raster(
        estimator=clf,
        feature_stack=stack,
        output_img=cls_out,
        scaler=scaler,
        lut={0: 10, 1: 20},
        thematic=True,
        block_size=8,
        no_data_val=0,
        use_pbar=False,
    )
    with rasterio.open(cls_out) as src:
        cls_arr = src.read(1)
    assert cls_arr[0, 0] == 0
    assert cls_arr[0, 1] == 0
    assert cls_arr[23, 23] == 0
    assert set(np.unique(cls_arr[valid])).issubset({10, 20})

    # 2) predict_proba_raster + out_class_img + out_entropy_img in one pass
    proba_out = str(tmp_path / "proba.tif")
    hard_out = str(tmp_path / "hard.tif")
    ent_out = str(tmp_path / "entropy.tif")
    predict_proba_raster(
        estimator=clf,
        feature_stack=stack,
        output_img=proba_out,
        in_msk_img=synthetic_stack_files["msk_img"],
        scaler=scaler,
        class_names=["bg", "fg"],
        out_class_img=hard_out,
        out_entropy_img=ent_out,
        lut={0: 1, 1: 2},
        block_size=8,
        use_pbar=False,
    )
    with rasterio.open(proba_out) as src:
        probs = src.read()
        assert src.count == 2
    assert os.path.exists(hard_out)
    assert os.path.exists(ent_out)
    # Probabilities at valid interior pixel sum to ~1
    assert probs[:, 12, 12].sum() == pytest.approx(1.0, abs=1e-4)
    # Masked bottom row has nodata
    with rasterio.open(hard_out) as src:
        hard_arr = src.read(1)
    assert hard_arr[23, 5] == 0

    # 3) predict_quantiles_raster: ensemble empirical quantiles
    reg = RandomForestRegressor(n_estimators=20, max_depth=5, random_state=42).fit(
        X_s, y_reg
    )
    q_ens_out = str(tmp_path / "q_ens.tif")
    predict_quantiles_raster(
        estimators_or_model=reg,
        feature_stack=stack,
        output_img=q_ens_out,
        quantiles=(0.1, 0.5, 0.9),
        scaler=scaler,
        block_size=8,
        use_pbar=False,
    )
    with rasterio.open(q_ens_out) as src:
        q_arr = src.read()
        assert src.count == 3
    assert q_arr[0, 12, 12] <= q_arr[1, 12, 12] <= q_arr[2, 12, 12]

    # 4) predict_quantiles_raster: dict of quantile models
    m_lo = Ridge().fit(X_s, y_reg - 1.0)
    m_hi = Ridge().fit(X_s, y_reg + 1.0)
    q_dict_out = str(tmp_path / "q_dict.tif")
    predict_quantiles_raster(
        estimators_or_model={0.1: m_lo, 0.9: m_hi},
        feature_stack=stack,
        output_img=q_dict_out,
        scaler=scaler,
        block_size=8,
        use_pbar=False,
    )
    with rasterio.open(q_dict_out) as src:
        qd = src.read()
        assert src.count == 2
    assert qd[0, 12, 12] < qd[1, 12, 12]

    # 5) predict_quantiles_raster: split-conformal intervals
    m_pt = Ridge().fit(X_s, y_reg)
    residuals = np.abs(y_reg - m_pt.predict(X_s))
    q_conf_out = str(tmp_path / "q_conf.tif")
    predict_quantiles_raster(
        estimators_or_model=m_pt,
        feature_stack=stack,
        output_img=q_conf_out,
        scaler=scaler,
        conformal_residuals=residuals,
        alpha=0.1,
        block_size=8,
        use_pbar=False,
    )
    with rasterio.open(q_conf_out) as src:
        qc = src.read()
        assert src.count == 3
    assert qc[0, 12, 12] < qc[1, 12, 12] < qc[2, 12, 12]


@pytest.mark.parametrize("method", ["grid", "checkerboard", "kmeans"])
def test_spatial_block_kfold(method):
    """SpatialBlockKFold covers every sample once with disjoint train/test splits."""
    rng = np.random.default_rng(11)
    n = 120
    coords = rng.uniform(0.0, 1000.0, size=(n, 2))
    X = rng.normal(size=(n, 3))
    y = (coords[:, 0] > 500.0).astype(int)

    splitter = SpatialBlockKFold(
        n_splits=4, method=method, stratify=(method != "checkerboard"), rnd_seed=42
    )
    seen = []
    for tr, te in splitter.split(X, y, coords=coords):
        assert len(tr) > 0 and len(te) > 0
        assert len(np.intersect1d(tr, te)) == 0
        seen.extend(te.tolist())
    assert sorted(seen) == list(range(n))


def test_spatial_buffer_cv_enforces_minimum_distance():
    """SpatialBufferCV removes all training points within buffer_dist of validation points."""
    rng = np.random.default_rng(19)
    coords = rng.uniform(0.0, 1000.0, size=(150, 2))
    X = rng.normal(size=(150, 3))
    y = rng.integers(0, 2, size=150)

    buf_dist = 150.0
    splitter = SpatialBufferCV(
        n_splits=5, buffer_dist=buf_dist, method="kmeans", rnd_seed=42
    )
    fold_count = 0
    for tr, te in splitter.split(X, y, coords=coords):
        fold_count += 1
        assert len(tr) > 0 and len(te) > 0
        assert len(np.intersect1d(tr, te)) == 0
        dists = cdist(coords[tr], coords[te])
        assert float(dists.min()) > buf_dist
    assert fold_count == 5


def test_polygon_group_kfold_zero_leakage():
    """PolygonGroupKFold guarantees no polygon ID appears in both train and test."""
    rng = np.random.default_rng(23)
    groups = np.repeat(np.arange(12), 10)
    X = rng.normal(size=(120, 4))
    y = (groups % 2).astype(int)

    for stratify in (False, True):
        splitter = PolygonGroupKFold(n_splits=4, stratify=stratify, rnd_seed=42)
        seen = []
        for tr, te in splitter.split(X, y, groups=groups):
            assert len(set(groups[tr]).intersection(set(groups[te]))) == 0
            seen.extend(te.tolist())
        assert sorted(seen) == list(range(120))


def test_tune_estimator_and_nested_spatial_cv():
    """Test Optuna tune_estimator and nested_spatial_cv for classification & regression."""
    rng = np.random.default_rng(31)
    n = 100
    coords = rng.uniform(0.0, 1000.0, size=(n, 2))
    X = rng.normal(size=(n, 4))
    y_cls = (X[:, 0] + 0.5 * X[:, 1] > 0.0).astype(int)
    y_reg = 2.0 * X[:, 0] - 1.5 * X[:, 1] + rng.normal(0.0, 0.2, size=n)

    # Check default param spaces
    for model_type in ("rf", "et", "hist_gb", "xgb", "lgbm", "catboost"):
        sp = get_default_param_space(model_type, task="classification")
        assert isinstance(sp, dict) and len(sp) > 0

    param_space = {
        "n_estimators": ("int", 10, 30),
        "max_depth": ("int", 2, 6),
    }

    # tune_estimator with spatial CV and fold-local scaling
    tune_res = tune_estimator(
        estimator_factory=lambda p: RandomForestClassifier(**p, random_state=42),
        X=X,
        y=y_cls,
        param_space=param_space,
        task="classification",
        cv=SpatialBlockKFold(n_splits=3, method="kmeans", rnd_seed=42),
        coords=coords,
        scaler=FeatureScaler(method="standard_scaler"),
        n_trials=4,
        use_pbar=False,
    )
    assert isinstance(tune_res, TuneResult)
    assert tune_res.best_estimator is not None
    assert tune_res.best_scaler is not None and tune_res.best_scaler.is_fitted
    assert len(tune_res.trials_df) == 4

    # nested_spatial_cv classification
    ncv_cls = nested_spatial_cv(
        estimator_factory=lambda p: RandomForestClassifier(**p, random_state=42),
        X=X,
        y=y_cls,
        task="classification",
        coords=coords,
        outer_cv=SpatialBlockKFold(n_splits=3, method="grid", rnd_seed=42),
        inner_cv=SpatialBlockKFold(n_splits=2, method="kmeans", rnd_seed=42),
        param_space=param_space,
        scaler=FeatureScaler(method="zscore"),
        n_trials=3,
        use_pbar=False,
    )
    assert isinstance(ncv_cls, NestedCVResult)
    assert len(ncv_cls.fold_scores) == 3
    assert ncv_cls.oof_predictions.shape == (n,)
    assert "accuracy" in ncv_cls.overall_metrics

    # nested_spatial_cv regression without tuning
    ncv_reg = nested_spatial_cv(
        estimator_factory=lambda p: Ridge(**p),
        X=X,
        y=y_reg,
        task="regression",
        coords=coords,
        outer_cv=SpatialBlockKFold(n_splits=3, method="kmeans", rnd_seed=42),
        scaler=FeatureScaler(method="robust_zscore"),
        use_pbar=False,
    )
    assert len(ncv_reg.fold_scores) == 3
    assert "r2" in ncv_reg.overall_metrics
    assert ncv_reg.overall_metrics["r2"] > 0.8
