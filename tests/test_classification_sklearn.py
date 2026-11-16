"""
Tests for rsgispy.classification.classsklearn: parameter optimisation (Grid Search,
Random Search, Optuna), classifier training, persistence, and image pixel
classification outputting class IDs and probability scores.
"""

import importlib.util
import inspect
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point, box
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

import rsgispy
import rsgispy.classification as rcls
import rsgispy.classification.classdata as rcd
import rsgispy.classification.classsklearn as rcsk
import rsgispy.imageutils.colourtable as rct

_HAS_PARQUET = (
    importlib.util.find_spec("pyarrow") is not None
    or importlib.util.find_spec("fastparquet") is not None
)


def _write_raster(
    path: Path,
    arr: np.ndarray,
    origin_x: float = 500000.0,
    origin_y: float = 5600000.0,
    pxl_size: float = 10.0,
    crs: str = "EPSG:32630",
    nodata: float = -999.0,
) -> str:
    """Write a 2D or 3D GeoTIFF raster for testing."""
    if arr.ndim == 2:
        count = 1
        height, width = arr.shape
        write_arr = arr[None, :, :]
    else:
        count, height, width = arr.shape
        write_arr = arr

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=count,
        dtype=str(arr.dtype),
        crs=crs,
        transform=from_origin(origin_x, origin_y, pxl_size, pxl_size),
        nodata=nodata,
    ) as dst:
        dst.write(write_arr)
    return str(path)


def _create_cls_fixture(tmp_path: Path):
    """Build a synthetic 3-class remote sensing scene, config, and parquet splits."""
    h, w = 18, 18
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    # Distinct spectral clusters:
    # Top third (rows 0..5): low b1, high b2 (Forest, out_id=10)
    # Middle third (rows 6..11): high b1, low b2 (Water, out_id=50)
    # Bottom third (rows 12..17): high b1, high b2 (Urban, out_id=300 -> uint16)
    b1 = np.zeros((h, w), dtype=np.float32)
    b2 = np.zeros((h, w), dtype=np.float32)

    b1[0:6, :] = 10.0 + 0.2 * xx[0:6, :]
    b2[0:6, :] = 90.0 + 0.2 * yy[0:6, :]

    b1[6:12, :] = 90.0 + 0.2 * xx[6:12, :]
    b2[6:12, :] = 10.0 + 0.2 * yy[6:12, :]

    b1[12:18, :] = 90.0 + 0.2 * xx[12:18, :]
    b2[12:18, :] = 90.0 + 0.2 * yy[12:18, :]

    # Inject a nodata pixel inside ROI mask to verify nodata exclusion.
    b1[3, 3] = -999.0

    # Valid ROI mask (outer 1px border excluded = 0, interior = 1).
    mask = np.zeros((h, w), dtype=np.uint8)
    h_end = h - 1
    w_end = w - 1
    mask[1:h_end, 1:w_end] = 1

    # Additional 20m coarse raster (9x9) to test multi-resolution inference.
    coarse = np.full((9, 9), 5.0, dtype=np.float32)
    coarse[3:6, :] = 50.0
    coarse[6:9, :] = 95.0

    s2_img = _write_raster(tmp_path / "s2.tif", np.stack([b1, b2]), nodata=-999.0)
    coarse_img = _write_raster(
        tmp_path / "coarse.tif", coarse, pxl_size=20.0, nodata=-999.0
    )
    msk_img = _write_raster(tmp_path / "mask.tif", mask, nodata=0.0)

    # Polygon and point training vectors covering each class zone.
    forest_vec = str(tmp_path / "forest.gpkg")
    gpd.GeoDataFrame(
        {"cid": [1]},
        geometry=[box(500020.0, 5599940.0, 500160.0, 5599990.0)],
        crs="EPSG:32630",
    ).to_file(forest_vec, layer="forest", driver="GPKG")

    water_vec = str(tmp_path / "water.gpkg")
    gpd.GeoDataFrame(
        {"cid": [2]},
        geometry=[box(500020.0, 5599880.0, 500160.0, 5599930.0)],
        crs="EPSG:32630",
    ).to_file(water_vec, layer="water", driver="GPKG")

    urban_pts = [
        Point(500015.0 + c * 10.0, 5599875.0 - r * 10.0)
        for r in range(5)
        for c in range(10)
    ]
    urban_vec = str(tmp_path / "urban.gpkg")
    gpd.GeoDataFrame(
        {"cid": list(range(len(urban_pts)))},
        geometry=urban_pts,
        crs="EPSG:32630",
    ).to_file(urban_vec, layer="urban", driver="GPKG")

    img_band_info = [
        rcd.ImageBandInfo(s2_img, name="S2", bands=[1, 2], band_names=["B1", "B2"]),
        rcd.ImageBandInfo(coarse_img, name="Coarse", bands=[1], band_names=["C1"]),
    ]
    class_info = {
        "Forest": rcd.ClassInfoObj(
            out_id=10, vec_file=forest_vec, vec_lyr="forest", red=0, green=180, blue=0
        ),
        "Water": rcd.ClassInfoObj(
            out_id=50, vec_file=water_vec, vec_lyr="water", red=0, green=0, blue=255
        ),
        "Urban": rcd.ClassInfoObj(
            out_id=300, vec_file=urban_vec, vec_lyr="urban", red=220, green=40, blue=40
        ),
    }

    cfg_path = tmp_path / "cls_cfg.json"
    cfg = rcd.write_cls_config(cfg_path, img_band_info, class_info)

    ext = ".parquet" if _HAS_PARQUET else ".csv"
    train_pq = tmp_path / f"train{ext}"
    test_pq = tmp_path / f"test{ext}"
    valid_pq = tmp_path / f"valid{ext}"
    train_df, test_df, valid_df = rcd.extract_and_split_class_samples(
        cfg,
        train_prop=0.6,
        test_prop=0.2,
        valid_prop=0.2,
        out_train_file=train_pq,
        out_test_file=test_pq,
        out_valid_file=valid_pq,
        rnd_seed=42,
        use_pbar=False,
    )

    return {
        "cfg": cfg,
        "cfg_path": cfg_path,
        "s2_img": s2_img,
        "coarse_img": coarse_img,
        "msk_img": msk_img,
        "train_df": train_df,
        "test_df": test_df,
        "valid_df": valid_df,
        "train_pq": train_pq,
        "test_pq": test_pq,
        "valid_pq": valid_pq,
    }


def test_classsklearn_module_layout():
    """Verify classsklearn exports cleanly and excludes legacy RAT functions."""
    assert inspect.ismodule(rcsk)
    assert not hasattr(rcls, "train_sklearn_classifier")
    assert not hasattr(rcls, "apply_sklearn_classifier")
    assert not hasattr(rcsk, "apply_sklearn_classifier_rat")
    for fn_name in (
        "perform_sklearn_classifier_param_search",
        "train_sklearn_classifier",
        "apply_sklearn_classifier",
        "save_sklearn_classifier",
        "load_sklearn_classifier",
    ):
        assert callable(getattr(rcsk, fn_name))


def test_param_search_grid_random_and_optuna(tmp_path):
    """Test grid search, random search, and Optuna Bayesian optimisation."""
    fix = _create_cls_fixture(tmp_path)
    cfg_path = fix["cfg_path"]
    train_pq = fix["train_pq"]
    valid_pq = fix["valid_pq"]

    # 1. Grid Search (with CV on train_data) + results/params persistence.
    grid_csv = tmp_path / "grid_results.csv"
    grid_json = tmp_path / "grid_params.json"
    grid_clf, grid_res = rcsk.perform_sklearn_classifier_param_search(
        cls_cfg=cfg_path,
        sk_classifier=RandomForestClassifier(),
        train_data=train_pq,
        param_space={"n_estimators": [10, 25], "max_depth": [2, 5]},
        search_method="grid",
        cv=3,
        rnd_seed=42,
        out_results_file=grid_csv,
        out_params_json=grid_json,
        return_results=True,
    )
    assert isinstance(grid_clf, RandomForestClassifier)
    assert grid_res["search_method"] == "grid"
    assert grid_res["best_score"] >= 0.9
    assert len(grid_res["results_df"]) == 4
    assert grid_csv.exists()
    saved_params = json.loads(grid_json.read_text(encoding="utf-8"))
    assert saved_params["search_method"] == "grid"
    assert set(saved_params["best_params"]) == {"n_estimators", "max_depth"}

    # 2. Random Search (evaluating against valid_data).
    rnd_clf, rnd_res = rcsk.perform_sklearn_classifier_param_search(
        cls_cfg=fix["cfg"],
        sk_classifier=RandomForestClassifier(),
        train_data=fix["train_df"],
        valid_data=valid_pq,
        param_space={
            "n_estimators": [5, 15, 25, 35],
            "max_depth": [2, 4, 6, None],
        },
        search_method="random",
        n_iter=4,
        rnd_seed=7,
        return_results=True,
    )
    assert isinstance(rnd_clf, RandomForestClassifier)
    assert rnd_res["search_method"] == "random"
    assert len(rnd_res["results_df"]) == 4
    assert rnd_res["best_score"] >= 0.9

    # 3. Pre-built GridSearchCV passthrough.
    gs_obj = GridSearchCV(
        RandomForestClassifier(random_state=42),
        {"n_estimators": [10, 20]},
        cv=3,
    )
    gs_clf = rcsk.perform_sklearn_classifier_param_search(
        cls_cfg=fix["cfg"],
        sk_classifier=gs_obj,
        train_data=fix["train_df"],
        rnd_seed=42,
    )
    assert gs_clf.n_estimators in (10, 20)

    # 4. Optuna Bayesian optimisation using declarative dictionary space.
    if importlib.util.find_spec("optuna") is not None:
        ext = ".parquet" if _HAS_PARQUET else ".csv"
        opt_pq = tmp_path / f"optuna_trials{ext}"
        opt_clf, opt_res = rcsk.perform_sklearn_classifier_param_search(
            cls_cfg=cfg_path,
            sk_classifier=RandomForestClassifier(),
            train_data=train_pq,
            valid_data=valid_pq,
            param_space={
                "n_estimators": (10, 40),
                "max_depth": {"type": "int", "low": 2, "high": 8, "step": 2},
                "criterion": ["gini", "entropy"],
                "min_samples_split": 2,
            },
            search_method="optuna",
            n_trials=8,
            rnd_seed=42,
            out_results_file=opt_pq,
            return_results=True,
        )
        assert isinstance(opt_clf, RandomForestClassifier)
        assert opt_res["search_method"] == "optuna"
        assert len(opt_res["results_df"]) == 8
        assert opt_pq.exists()
        assert opt_res["best_score"] >= 0.9
        assert opt_clf.min_samples_split == 2

        # 5. Optuna Bayesian optimisation using callable trial function.
        opt_callable_clf = rcsk.perform_sklearn_classifier_param_search(
            cls_cfg=fix["cfg"],
            sk_classifier=SVC(probability=True),
            train_data=fix["train_df"],
            param_space=lambda trial: {
                "C": trial.suggest_float("C", 0.1, 10.0, log=True),
                "kernel": trial.suggest_categorical("kernel", ["linear", "rbf"]),
            },
            search_method="optuna",
            cv=3,
            n_trials=6,
            rnd_seed=11,
        )
        assert isinstance(opt_callable_clf, SVC)

    with pytest.raises(rsgispy.RSGISPyException):
        rcsk.perform_sklearn_classifier_param_search(
            fix["cfg"],
            RandomForestClassifier(),
            fix["train_df"],
            param_space={"n_estimators": [10]},
            search_method="invalid_method",
        )


def test_train_and_apply_sklearn_classifier_ids_and_probs(tmp_path):
    """
    Test training, persistence, and scene classification producing both
    recoded unsigned output class IDs and probability score rasters.
    """
    fix = _create_cls_fixture(tmp_path)
    model_path = tmp_path / "rf_model.joblib"
    metrics_path = tmp_path / "train_metrics.json"

    base_clf = RandomForestClassifier(n_estimators=25, random_state=42)
    metrics = rcsk.train_sklearn_classifier(
        cls_cfg=fix["cfg_path"],
        sk_classifier=base_clf,
        train_data=fix["train_pq"],
        test_data=fix["test_pq"],
        valid_data=fix["valid_pq"],
        rnd_seed=42,
        out_model_file=model_path,
        out_metrics_file=metrics_path,
        return_model=False,
    )
    assert metrics["train_accuracy"] == pytest.approx(1.0)
    assert metrics["test_accuracy"] >= 0.95
    assert metrics["valid_accuracy"] >= 0.95
    assert model_path.exists()
    assert json.loads(metrics_path.read_text(encoding="utf-8"))["test_accuracy"] >= 0.95

    # Clean isolated return_model=True should not mutate original template.
    unfitted_tpl = RandomForestClassifier(n_estimators=15)
    isolated_clf, iso_metrics = rcsk.train_sklearn_classifier(
        cls_cfg=fix["cfg"],
        sk_classifier=unfitted_tpl,
        train_data=fix["train_df"],
        return_model=True,
    )
    assert not hasattr(unfitted_tpl, "estimators_")
    assert hasattr(isolated_clf, "estimators_")
    assert iso_metrics["train_accuracy"] == pytest.approx(1.0)

    # Apply classifier from saved joblib path + saved config JSON, outputting
    # both class IDs (recoded to out_ids 10, 50, 300 -> uint16) and 3-band probs.
    out_cls = tmp_path / "classified_ids.tif"
    out_score = tmp_path / "classified_probs.tif"
    rcsk.apply_sklearn_classifier(
        cls_cfg=fix["cfg_path"],
        sk_classifier=model_path,
        in_msk_img=fix["msk_img"],
        img_msk_val=1,
        out_class_img=out_cls,
        out_score_img=out_score,
        block_size=8,
        use_pbar=False,
    )

    with rasterio.open(out_cls, "r") as src:
        assert src.dtypes[0] == "uint16"
        assert src.nodata == 0
        cls_arr = src.read(1)
        # Border outside mask and nodata pixel (3, 3) must be nodata=0.
        assert cls_arr[0, 0] == 0
        assert cls_arr[3, 3] == 0
        # Verify recoded out_ids per spectral zone.
        assert np.all(cls_arr[1:6, 1:17][np.arange(5) != 2, :] != 0)
        assert set(np.unique(cls_arr[1:6, 5:15])) == {10}
        assert set(np.unique(cls_arr[6:12, 5:15])) == {50}
        assert set(np.unique(cls_arr[12:17, 5:15])) == {300}

    assert rct.is_img_thematic(str(out_cls))
    clr_tab = rct.get_colour_tab_info(str(out_cls))
    assert clr_tab[10]["green"] == 180
    assert clr_tab[50]["blue"] == 255
    assert clr_tab[300]["red"] == 220

    with rasterio.open(out_score, "r") as src:
        assert src.count == 3
        assert src.dtypes[0] == "float32"
        assert src.nodata == pytest.approx(-1.0)
        assert list(src.descriptions) == ["Forest", "Water", "Urban"]
        probs = src.read()
        # Masked border and nodata pixel must have score nodata=-1.0.
        assert np.all(probs[:, 0, 0] == pytest.approx(-1.0))
        assert np.all(probs[:, 3, 3] == pytest.approx(-1.0))
        # Valid ROI pixels have probabilities in [0, 1] summing to ~1.0,
        # and band argmax matches class zone.
        assert probs[:, 2, 10].sum() == pytest.approx(1.0, abs=1e-5)
        assert np.argmax(probs[:, 2, 10]) == 0  # Forest
        assert np.argmax(probs[:, 8, 10]) == 1  # Water
        assert np.argmax(probs[:, 14, 10]) == 2  # Urban

    # Single-band max probability output only (out_class_img=None, score_max_only=True).
    out_max_score = tmp_path / "max_prob.tif"
    rcsk.apply_sklearn_classifier(
        cls_cfg=fix["cfg"],
        sk_classifier=base_clf,
        in_msk_img=fix["msk_img"],
        img_msk_val=1,
        out_class_img=None,
        out_score_img=out_max_score,
        score_max_only=True,
        use_pbar=False,
    )
    with rasterio.open(out_max_score, "r") as src:
        assert src.count == 1
        assert src.descriptions == ("max_class_probability",)
        max_p = src.read(1)
        assert max_p[0, 0] == pytest.approx(-1.0)
        assert 0.5 <= max_p[8, 10] <= 1.0


def test_apply_class_zero_and_xgboost_workflow(tmp_path):
    """
    Verify out_id=0 collision avoidance, class_info dict input, and XGBoost
    0-based consecutive internal ID training -> recoded output ID raster.
    """
    xgboost = pytest.importorskip("xgboost")

    b1 = np.tile(np.linspace(0.0, 100.0, 10, dtype=np.float32), (10, 1))
    img_path = _write_raster(tmp_path / "grad.tif", b1, nodata=-999.0)
    msk_arr = np.ones((10, 10), dtype=np.uint8)
    msk_arr[0, 0] = 0
    msk_path = _write_raster(tmp_path / "msk.tif", msk_arr, nodata=0.0)

    train_df = pd.DataFrame(
        {
            "class_name": ["ZeroCls"] * 8 + ["HighCls"] * 8,
            "out_cls_id": [0] * 8 + [20] * 8,
            "internal_cls_id": [0] * 8 + [1] * 8,
            "B1": [5.0 + i for i in range(8)] + [85.0 + i for i in range(8)],
        }
    )

    class_info = {
        "ZeroCls": rcd.ClassInfoObj(out_id=0, red=10, green=20, blue=30),
        "HighCls": rcd.ClassInfoObj(out_id=20, red=200, green=100, blue=50),
    }
    img_band_info = [rcd.ImageBandInfo(img_path, name="S2", bands=[1], band_names=["B1"])]

    xgb_clf = xgboost.XGBClassifier(
        n_estimators=10,
        max_depth=2,
        learning_rate=0.3,
        random_state=42,
        eval_metric="logloss",
    )
    metrics = rcsk.train_sklearn_classifier(
        cls_cfg=class_info,
        sk_classifier=xgb_clf,
        train_data=train_df,
        feature_cols=["B1"],
        use_internal_id=True,
    )
    assert metrics["train_accuracy"] == pytest.approx(1.0)

    out_cls = tmp_path / "xgb_cls.tif"
    out_probs = tmp_path / "xgb_probs.tif"
    rcsk.apply_sklearn_classifier(
        cls_cfg=class_info,
        sk_classifier=xgb_clf,
        in_msk_img=msk_path,
        img_msk_val=1,
        out_class_img=out_cls,
        out_score_img=out_probs,
        img_band_info=img_band_info,
        use_pbar=False,
    )
    with rasterio.open(out_cls, "r") as src:
        # Since out_id=0 is used by ZeroCls, automatic nodata avoids 0!
        assert src.nodata == 255
        arr = src.read(1)
        assert arr[0, 0] == 255
        assert arr[5, 1] == 0
        assert arr[5, 8] == 20


def test_apply_sklearn_validation_errors(tmp_path):
    """Verify clear errors for missing outputs, unfitted models, or missing predict_proba."""
    fix = _create_cls_fixture(tmp_path)

    with pytest.raises(rsgispy.RSGISPyException):
        rcsk.apply_sklearn_classifier(
            fix["cfg"],
            RandomForestClassifier(),
            fix["msk_img"],
            1,
            out_class_img=None,
            out_score_img=None,
        )

    # Unfitted estimator raises RSGISPyException.
    with pytest.raises(rsgispy.RSGISPyException):
        rcsk.apply_sklearn_classifier(
            fix["cfg"],
            RandomForestClassifier(),
            fix["msk_img"],
            1,
            out_class_img=tmp_path / "unfitted.tif",
        )

    # SVC(probability=False) lacks predict_proba when out_score_img is requested.
    svc = SVC(probability=False, random_state=42)
    rcsk.train_sklearn_classifier(fix["cfg"], svc, fix["train_df"])
    with pytest.raises(rsgispy.RSGISPyException):
        rcsk.apply_sklearn_classifier(
            fix["cfg"],
            svc,
            fix["msk_img"],
            1,
            out_class_img=tmp_path / "svc.tif",
            out_score_img=tmp_path / "svc_prob.tif",
        )
