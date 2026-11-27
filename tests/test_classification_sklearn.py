"""
Tests for rsgispy.classification.classsklearn: parameter optimisation (Grid Search,
Random Search, Optuna), classifier training, persistence, and image pixel
classification outputting class IDs and probability scores.
"""

import importlib.util
import inspect
import json
from pathlib import Path
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point, box
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

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
        "get_sklearn_classifier_param_space",
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
        output_img=out_cls,
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

    # Single-band max probability output only (output_img=None, score_max_only=True).
    out_max_score = tmp_path / "max_prob.tif"
    rcsk.apply_sklearn_classifier(
        cls_cfg=fix["cfg"],
        sk_classifier=base_clf,
        in_msk_img=fix["msk_img"],
        img_msk_val=1,
        output_img=None,
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
    import rsgispy.classification.classxgboost as rcxgb  # noqa: PLC0415

    if not rcxgb.HAVE_XGBOOST:
        pytest.skip("xgboost is not available or failed to load")
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
        output_img=out_cls,
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
            output_img=None,
            out_score_img=None,
        )

    # Unfitted estimator raises RSGISPyException.
    with pytest.raises(rsgispy.RSGISPyException):
        rcsk.apply_sklearn_classifier(
            fix["cfg"],
            RandomForestClassifier(),
            fix["msk_img"],
            1,
            output_img=tmp_path / "unfitted.tif",
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
            output_img=tmp_path / "svc.tif",
            out_score_img=tmp_path / "svc_prob.tif",
        )


def test_get_sklearn_classifier_param_space():
    """Verify built-in parameter spaces across classifiers, aliases, and Pipelines."""
    classifiers_and_aliases = [
        (RandomForestClassifier(), "rf"),
        (ExtraTreesClassifier(), "et"),
        (GradientBoostingClassifier(), "gbc"),
        (HistGradientBoostingClassifier(), "hgbc"),
        (SVC(), "svm"),
        (MLPClassifier(), "mlp"),
        (KNeighborsClassifier(), "knn"),
        (LogisticRegression(), "logreg"),
        (DecisionTreeClassifier(), "dt"),
        (AdaBoostClassifier(), "ada"),
    ]

    for est, alias in classifiers_and_aliases:
        cls_type = type(est)
        for method in ("optuna", "grid", "random"):
            for extended in (False, True):
                sp_inst = rcsk.get_sklearn_classifier_param_space(
                    est, search_method=method, extended=extended
                )
                sp_cls = rcsk.get_sklearn_classifier_param_space(
                    cls_type, search_method=method, extended=extended
                )
                sp_alias = rcsk.get_sklearn_classifier_param_space(
                    alias, search_method=method, extended=extended
                )
                if callable(sp_inst):
                    assert callable(sp_cls)
                    assert callable(sp_alias)
                else:
                    assert isinstance(sp_inst, dict) and len(sp_inst) >= 2
                    assert sp_inst == sp_cls == sp_alias

        # Grid / Random discrete lists must be valid non-empty parameter names on estimator.
        valid_est_params = set(est.get_params().keys())
        for extended in (False, True):
            cv_sp = rcsk.get_sklearn_classifier_param_space(
                est, search_method="grid", extended=extended
            )
            assert isinstance(cv_sp, dict)
            assert set(cv_sp.keys()).issubset(valid_est_params)
            assert all(isinstance(vals, list) and len(vals) >= 2 for vals in cv_sp.values())

    # Pipeline prefixing (both dict space and callable extended Optuna space).
    rf_pipe = Pipeline([("scaler", StandardScaler()), ("clf", RandomForestClassifier())])
    rf_pipe_sp = rcsk.get_sklearn_classifier_param_space(rf_pipe, search_method="grid")
    assert all(k.startswith("clf__") for k in rf_pipe_sp)

    nested_pipe = Pipeline(
        [
            ("outer", Pipeline([("scaler", StandardScaler()), ("svc", SVC())])),
        ]
    )
    nested_cv_sp = rcsk.get_sklearn_classifier_param_space(
        nested_pipe, search_method="random", extended=True
    )
    assert all(k.startswith("outer__svc__") for k in nested_cv_sp)

    optuna = pytest.importorskip("optuna")
    mlp_pipe = Pipeline([("scaler", StandardScaler()), ("mlp", MLPClassifier())])
    mlp_sampler = rcsk.get_sklearn_classifier_param_space(
        mlp_pipe, search_method="optuna", extended=True
    )
    assert callable(mlp_sampler)
    study = optuna.create_study(direction="maximize")
    trial = study.ask()
    sampled = mlp_sampler(trial)
    assert all(k.startswith("mlp__") for k in sampled)
    assert isinstance(sampled["mlp__hidden_layer_sizes"], tuple)
    assert set(sampled.keys()).issubset(set(mlp_pipe.get_params().keys()))

    # Unsupported classifier and invalid search method raise RSGISPyException.
    with pytest.raises(rsgispy.RSGISPyException, match="Supported classifiers"):
        rcsk.get_sklearn_classifier_param_space(GaussianNB())
    with pytest.raises(rsgispy.RSGISPyException):
        rcsk.get_sklearn_classifier_param_space(RandomForestClassifier(), search_method="bad")


def test_param_search_builtin_spaces(tmp_path):
    """Test perform_sklearn_classifier_param_search auto-selected built-in spaces."""
    fix = _create_cls_fixture(tmp_path)

    # 1. Grid & Random search with param_space=None on LogisticRegression and DecisionTree.
    dt_clf, dt_res = rcsk.perform_sklearn_classifier_param_search(
        cls_cfg=fix["cfg"],
        sk_classifier=DecisionTreeClassifier(),
        train_data=fix["train_df"],
        valid_data=fix["valid_df"],
        search_method="random",
        n_iter=5,
        rnd_seed=42,
        return_results=True,
    )
    assert isinstance(dt_clf, DecisionTreeClassifier)
    assert dt_res["param_space_mode"] == "default"
    assert dt_res["best_score"] >= 0.85

    lr_pipe = Pipeline(
        [("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=500))]
    )
    lr_clf, lr_res = rcsk.perform_sklearn_classifier_param_search(
        cls_cfg=fix["cfg"],
        sk_classifier=lr_pipe,
        train_data=fix["train_df"],
        valid_data=fix["valid_df"],
        search_method="grid",
        rnd_seed=42,
        return_results=True,
    )
    assert isinstance(lr_clf, Pipeline)
    assert lr_res["param_space_mode"] == "default"
    assert all(k.startswith("lr__") for k in lr_res["best_params"])
    assert lr_res["best_score"] >= 0.9

    # 2. Optuna with param_space=None, "extended", and extended_param_space=True.
    if importlib.util.find_spec("optuna") is not None:
        rf_json = tmp_path / "rf_auto_params.json"
        rf_clf, rf_res = rcsk.perform_sklearn_classifier_param_search(
            cls_cfg=fix["cfg"],
            sk_classifier=RandomForestClassifier(),
            train_data=fix["train_df"],
            valid_data=fix["valid_df"],
            search_method="optuna",
            n_trials=5,
            rnd_seed=42,
            out_params_json=rf_json,
            return_results=True,
        )
        assert isinstance(rf_clf, RandomForestClassifier)
        assert rf_res["param_space_mode"] == "default"
        saved_meta = json.loads(rf_json.read_text(encoding="utf-8"))
        assert saved_meta["param_space_mode"] == "default"

        svc_pipe = Pipeline(
            [("scaler", StandardScaler()), ("svc", SVC(probability=True))]
        )
        svc_clf, svc_res = rcsk.perform_sklearn_classifier_param_search(
            cls_cfg=fix["cfg"],
            sk_classifier=svc_pipe,
            train_data=fix["train_df"],
            valid_data=fix["valid_df"],
            param_space="extended",
            search_method="optuna",
            n_trials=6,
            rnd_seed=19,
            return_results=True,
        )
        assert isinstance(svc_clf, Pipeline)
        assert svc_res["param_space_mode"] == "extended"
        assert all(k.startswith("svc__") for k in svc_res["best_params"])
        assert "svc__kernel" in svc_res["best_params"]

        mlp_pipe = Pipeline(
            [("scaler", StandardScaler()), ("mlp", MLPClassifier(max_iter=300))]
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mlp_clf, mlp_res = rcsk.perform_sklearn_classifier_param_search(
                cls_cfg=fix["cfg"],
                sk_classifier=mlp_pipe,
                train_data=fix["train_df"],
                valid_data=fix["valid_df"],
                search_method="optuna",
                n_trials=4,
                rnd_seed=7,
                extended_param_space=True,
                return_results=True,
            )
        optuna_user_warns = [
            w
            for w in caught
            if issubclass(w.category, UserWarning) and "optuna" in (w.filename or "").lower()
        ]
        assert not optuna_user_warns
        assert isinstance(mlp_clf, Pipeline)
        assert mlp_res["param_space_mode"] == "extended"
        assert isinstance(mlp_clf.named_steps["mlp"].hidden_layer_sizes, tuple)

    # 3. Validation errors for unsupported estimator or conflicting space arguments.
    with pytest.raises(rsgispy.RSGISPyException):
        rcsk.perform_sklearn_classifier_param_search(
            fix["cfg"],
            GaussianNB(),
            fix["train_df"],
            param_space=None,
            search_method="grid",
        )
    with pytest.raises(rsgispy.RSGISPyException):
        rcsk.perform_sklearn_classifier_param_search(
            fix["cfg"],
            RandomForestClassifier(),
            fix["train_df"],
            param_space="default",
            extended_param_space=True,
        )
    with pytest.raises(rsgispy.RSGISPyException):
        rcsk.perform_sklearn_classifier_param_search(
            fix["cfg"],
            RandomForestClassifier(),
            fix["train_df"],
            param_space={"n_estimators": [10]},
            extended_param_space=True,
        )
    with pytest.raises(rsgispy.RSGISPyException):
        rcsk.perform_sklearn_classifier_param_search(
            fix["cfg"],
            RandomForestClassifier(),
            fix["train_df"],
            param_space="unknown_preset",
        )


def test_optuna_progress_bar_option(tmp_path, monkeypatch):
    """Verify Optuna optimisation enables tqdm progress bar by default and respects use_pbar."""
    optuna = pytest.importorskip("optuna")
    fix = _create_cls_fixture(tmp_path)

    recorded = []
    orig_optimize = optuna.Study.optimize

    def _spy_optimize(self, func, *args, **kwargs):
        recorded.append(kwargs.get("show_progress_bar"))
        return orig_optimize(self, func, *args, **kwargs)

    monkeypatch.setattr(optuna.Study, "optimize", _spy_optimize)

    rcsk.perform_sklearn_classifier_param_search(
        cls_cfg=fix["cfg"],
        sk_classifier=RandomForestClassifier(random_state=0),
        train_data=fix["train_df"],
        valid_data=fix["valid_df"],
        search_method="optuna",
        n_trials=2,
        rnd_seed=42,
    )
    rcsk.perform_sklearn_classifier_param_search(
        cls_cfg=fix["cfg"],
        sk_classifier=RandomForestClassifier(random_state=0),
        train_data=fix["train_df"],
        valid_data=fix["valid_df"],
        search_method="optuna",
        n_trials=2,
        rnd_seed=42,
        use_pbar=False,
    )
    assert recorded == [True, False]


def _grouped_cls_samples(n_polys: int = 12, n_pxls: int = 8) -> pd.DataFrame:
    """Pixel samples of two classes from ``n_polys`` polygons spread in space."""
    rng = np.random.default_rng(0)
    rows = []
    for p_idx in range(n_polys):
        cls_id = p_idx % 2
        cx, cy = 1000.0 * (p_idx % 4), 1000.0 * (p_idx // 4)
        for _ in range(n_pxls):
            rows.append(
                {
                    "class_name": "A" if cls_id == 0 else "B",
                    "out_cls_id": cls_id + 1,
                    "internal_cls_id": cls_id,
                    "train_grp": f"{'AB'[cls_id]}:poly{p_idx}",
                    "x": cx + rng.uniform(0, 30),
                    "y": cy + rng.uniform(0, 30),
                    "f1": cls_id * 5.0 + rng.normal(0, 1),
                    "f2": rng.normal(0, 1),
                }
            )
    return pd.DataFrame(rows)


def _assert_splits_keep_groups(splits, groups):
    """No group in both the training and validation indices of a fold."""
    groups = np.asarray(groups)
    assert len(splits) >= 2
    for tr_idx, val_idx in splits:
        assert len(tr_idx) and len(val_idx)
        assert set(groups[tr_idx]).isdisjoint(groups[val_idx])


def test_param_search_with_spatial_and_group_splitters():
    """rsgispy.ml.cv splitters are passed directly with coords= and groups=."""
    from rsgispy.ml.cv import PolygonGroupKFold, SpatialBlockKFold

    train_df = _grouped_cls_samples()
    cls_info = {
        "A": rcd.ClassInfoObj(out_id=1),
        "B": rcd.ClassInfoObj(out_id=2),
    }
    common = dict(
        cls_cfg=cls_info,
        sk_classifier=RandomForestClassifier(n_estimators=10, random_state=0),
        train_data=train_df,
        param_space={"max_depth": [2, None]},
        feature_cols=["f1", "f2"],
        rnd_seed=1,
        return_results=True,
    )
    xy = train_df[["x", "y"]].to_numpy()

    # Spatial blocks: coords as two column names or as an (n, 2) array.
    spatial_cv = SpatialBlockKFold(n_splits=3, block_size=1500.0, rnd_seed=3)
    expected = list(spatial_cv.split(train_df, coords=xy))
    for coords in (("x", "y"), xy):
        _, res = rcsk.perform_sklearn_classifier_param_search(
            cv=spatial_cv, coords=coords, **common
        )
        cv_used = res["search_obj"].cv
        assert len(cv_used) == len(expected)
        for (tr_a, va_a), (tr_b, va_b) in zip(cv_used, expected):
            np.testing.assert_array_equal(tr_a, tr_b)
            np.testing.assert_array_equal(va_a, va_b)

    # Polygon groups: whole polygons in each fold (column name or array).
    for groups in ("train_grp", train_df["train_grp"].to_numpy()):
        _, res = rcsk.perform_sklearn_classifier_param_search(
            cv=PolygonGroupKFold(n_splits=3, stratify=True), groups=groups,
            **common,
        )
        _assert_splits_keep_groups(res["search_obj"].cv, train_df["train_grp"])
        assert res["best_score"] > 0.8

    # An integer cv with groups uses polygon-grouped folds.
    _, res = rcsk.perform_sklearn_classifier_param_search(
        cv=3, groups="train_grp", **common
    )
    _assert_splits_keep_groups(res["search_obj"].cv, train_df["train_grp"])

    # A pre-configured GridSearchCV with a group splitter.
    gs_obj = GridSearchCV(
        RandomForestClassifier(n_estimators=10, random_state=0),
        {"max_depth": [2, None]},
        cv=PolygonGroupKFold(n_splits=3),
    )
    _, res = rcsk.perform_sklearn_classifier_param_search(
        cls_info, gs_obj, train_df, feature_cols=["f1", "f2"],
        groups="train_grp", return_results=True,
    )
    _assert_splits_keep_groups(res["search_obj"].cv, train_df["train_grp"])

    # Optuna scores the trials on the same folds.
    if importlib.util.find_spec("optuna") is not None:
        best, res = rcsk.perform_sklearn_classifier_param_search(
            cls_info,
            RandomForestClassifier(n_estimators=10, random_state=0),
            train_df,
            param_space={"max_depth": [2, 4]},
            search_method="optuna",
            n_trials=2,
            cv=PolygonGroupKFold(n_splits=3),
            groups="train_grp",
            feature_cols=["f1", "f2"],
            return_results=True,
            use_pbar=False,
        )
        assert isinstance(best, RandomForestClassifier)
        assert res["best_score"] > 0.8

    # Missing or invalid coords / groups give clear errors.
    with pytest.raises(rsgispy.RSGISPyException, match="coords"):
        rcsk.perform_sklearn_classifier_param_search(cv=spatial_cv, **common)
    with pytest.raises(rsgispy.RSGISPyException, match="groups"):
        rcsk.perform_sklearn_classifier_param_search(
            cv=PolygonGroupKFold(n_splits=3), **common
        )
    with pytest.raises(rsgispy.RSGISPyException):
        rcsk.perform_sklearn_classifier_param_search(
            cv=3, groups="missing_col", **common
        )
    with pytest.raises(rsgispy.RSGISPyException):
        rcsk.perform_sklearn_classifier_param_search(
            cv=3, groups=np.arange(5), **common
        )
    with pytest.raises(rsgispy.RSGISPyException):
        rcsk.perform_sklearn_classifier_param_search(
            cv=3, coords=("x", "y", "z"), **common
        )


def test_optuna_search_rejects_minimize_with_sklearn_scorer(tmp_path):
    """sklearn scorers are greater-is-better, so direction='minimize' would pick the worst model."""
    pytest.importorskip("optuna")
    fix = _create_cls_fixture(tmp_path)
    with pytest.raises(rsgispy.RSGISPyException, match="greater-is-better"):
        rcsk.perform_sklearn_classifier_param_search(
            cls_cfg=fix["cfg"],
            sk_classifier=RandomForestClassifier(),
            train_data=fix["train_df"],
            param_space={"n_estimators": (5, 10)},
            search_method="optuna",
            scoring="accuracy",
            direction="minimize",
            n_trials=2,
            cv=3,
            rnd_seed=1,
        )


def test_check_sklearn_scorer_direction():
    """String scorers must be maximised; callables may use either direction."""
    from rsgispy.tools._ml_tune import check_sklearn_scorer_direction

    check_sklearn_scorer_direction("neg_root_mean_squared_error", "maximize")
    check_sklearn_scorer_direction(lambda est, x, y: 0.0, "minimize")
    with pytest.raises(rsgispy.RSGISPyException, match="neg_"):
        check_sklearn_scorer_direction("neg_mean_absolute_error", "minimize")
