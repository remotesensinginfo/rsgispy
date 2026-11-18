"""
Tests for GBDT classification submodules:
- rsgispy.classification.classxgboost
- rsgispy.classification.classlightgbm
- rsgispy.classification.classcatboost
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

import rsgispy
import rsgispy.classification.classcatboost as rccb
import rsgispy.classification.classdata as rcd
import rsgispy.classification.classlightgbm as rclgb
import rsgispy.classification.classxgboost as rcxgb
import rsgispy.imageutils.colourtable as rct

_HAS_OPTUNA = importlib.util.find_spec("optuna") is not None


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


def _create_binary_fixture(tmp_path: Path):
    """Build synthetic 2-class imagery, mask, ClassificationConfig, and splits."""
    h, w = 16, 16
    b1 = np.zeros((h, w), dtype=np.float32)
    b2 = np.zeros((h, w), dtype=np.float32)

    # Top half: Background (out_id=10, internal_id=0)
    b1[0:8, :] = 10.0
    b2[0:8, :] = 90.0
    # Bottom half: Target (out_id=50, internal_id=1)
    b1[8:16, :] = 90.0
    b2[8:16, :] = 10.0
    # Inject a nodata pixel inside ROI
    b1[2, 2] = -999.0

    mask = np.zeros((h, w), dtype=np.uint8)
    h_end = h - 1
    w_end = w - 1
    mask[1:h_end, 1:w_end] = 1

    feat_img = _write_raster(
        tmp_path / "bin_feat.tif", np.stack([b1, b2]), nodata=-999.0
    )
    msk_img = _write_raster(tmp_path / "bin_mask.tif", mask, nodata=0.0)

    img_band_info = [
        rcd.ImageBandInfo(feat_img, name="S2", bands=[1, 2], band_names=["B1", "B2"])
    ]
    class_info = {
        "Background": rcd.ClassInfoObj(out_id=10, red=100, green=100, blue=100),
        "Target": rcd.ClassInfoObj(out_id=50, red=20, green=200, blue=40),
    }
    cfg = rcd.ClassificationConfig(img_band_info=img_band_info, class_info=class_info)

    rng = np.random.default_rng(42)

    def _make_split(n_per_cls: int, spread: float = 3.0) -> pd.DataFrame:
        b1_0 = 10.0 + rng.normal(0.0, spread, n_per_cls)
        b2_0 = 90.0 + rng.normal(0.0, spread, n_per_cls)
        b1_1 = 90.0 + rng.normal(0.0, spread, n_per_cls)
        b2_1 = 10.0 + rng.normal(0.0, spread, n_per_cls)
        return pd.DataFrame(
            {
                "class_name": ["Background"] * n_per_cls + ["Target"] * n_per_cls,
                "out_cls_id": [10] * n_per_cls + [50] * n_per_cls,
                "internal_cls_id": [0] * n_per_cls + [1] * n_per_cls,
                "B1": np.concatenate([b1_0, b1_1]),
                "B2": np.concatenate([b2_0, b2_1]),
            }
        )

    return {
        "cfg": cfg,
        "msk_img": msk_img,
        "train_df": _make_split(30, spread=4.0),
        "test_df": _make_split(12, spread=1.5),
        "valid_df": _make_split(12, spread=1.5),
    }


def _create_multiclass_fixture(tmp_path: Path):
    """Build synthetic 3-class imagery, mask, ClassificationConfig, and splits."""
    h, w = 18, 18
    b1 = np.zeros((h, w), dtype=np.float32)
    b2 = np.zeros((h, w), dtype=np.float32)

    # Forest (out_id=10), Water (out_id=50), Urban (out_id=300 -> uint16)
    b1[0:6, :] = 10.0
    b2[0:6, :] = 90.0
    b1[6:12, :] = 90.0
    b2[6:12, :] = 10.0
    b1[12:18, :] = 90.0
    b2[12:18, :] = 90.0
    b1[2, 2] = -999.0

    mask = np.zeros((h, w), dtype=np.uint8)
    h_end = h - 1
    w_end = w - 1
    mask[1:h_end, 1:w_end] = 1

    feat_img = _write_raster(
        tmp_path / "mc_feat.tif", np.stack([b1, b2]), nodata=-999.0
    )
    msk_img = _write_raster(tmp_path / "mc_mask.tif", mask, nodata=0.0)

    img_band_info = [
        rcd.ImageBandInfo(feat_img, name="S2", bands=[1, 2], band_names=["B1", "B2"])
    ]
    class_info = {
        "Forest": rcd.ClassInfoObj(out_id=10, red=0, green=180, blue=0),
        "Water": rcd.ClassInfoObj(out_id=50, red=0, green=0, blue=255),
        "Urban": rcd.ClassInfoObj(out_id=300, red=220, green=40, blue=40),
    }
    cfg = rcd.ClassificationConfig(img_band_info=img_band_info, class_info=class_info)

    rng = np.random.default_rng(7)

    def _make_split(n_per_cls: int) -> pd.DataFrame:
        b1_vals = np.concatenate(
            [
                10.0 + rng.normal(0.0, 1.5, n_per_cls),
                90.0 + rng.normal(0.0, 1.5, n_per_cls),
                90.0 + rng.normal(0.0, 1.5, n_per_cls),
            ]
        )
        b2_vals = np.concatenate(
            [
                90.0 + rng.normal(0.0, 1.5, n_per_cls),
                10.0 + rng.normal(0.0, 1.5, n_per_cls),
                90.0 + rng.normal(0.0, 1.5, n_per_cls),
            ]
        )
        return pd.DataFrame(
            {
                "class_name": (
                    ["Forest"] * n_per_cls
                    + ["Water"] * n_per_cls
                    + ["Urban"] * n_per_cls
                ),
                "out_cls_id": (
                    [10] * n_per_cls + [50] * n_per_cls + [300] * n_per_cls
                ),
                "internal_cls_id": (
                    [0] * n_per_cls + [1] * n_per_cls + [2] * n_per_cls
                ),
                "B1": b1_vals,
                "B2": b2_vals,
            }
        )

    return {
        "cfg": cfg,
        "msk_img": msk_img,
        "train_df": _make_split(20),
        "test_df": _make_split(10),
        "valid_df": _make_split(10),
    }


def _backend_specs():
    """Return per-library callbacks for parametrised GBDT workflow tests."""
    return [
        pytest.param(
            {
                "pkg": "xgboost",
                "mod": rcxgb,
                "ext": ".json",
                "opt_bin": rcxgb.optimise_xgboost_binary_classifier,
                "opt_mc": rcxgb.optimise_xgboost_multiclass_classifier,
                "train_bin": rcxgb.train_xgboost_binary_classifier,
                "train_mc": rcxgb.train_xgboost_multiclass_classifier,
                "apply_bin": rcxgb.apply_xgboost_binary_classifier,
                "apply_mc": rcxgb.apply_xgboost_multiclass_classifier,
                "save": rcxgb.save_xgboost_classifier,
                "load": rcxgb.load_xgboost_classifier,
                "cls_kw": "xgb_classifier",
                "fast_params": {
                    "n_estimators": 15,
                    "max_depth": 3,
                    "learning_rate": 0.2,
                },
                "bin_space": {
                    "n_estimators": (8, 20),
                    "max_depth": (2, 4),
                    "learning_rate": (0.05, 0.3),
                },
            },
            id="xgboost",
        ),
        pytest.param(
            {
                "pkg": "lightgbm",
                "mod": rclgb,
                "ext": ".joblib",
                "opt_bin": rclgb.optimise_lightgbm_binary_classifier,
                "opt_mc": rclgb.optimise_lightgbm_multiclass_classifier,
                "train_bin": rclgb.train_lightgbm_binary_classifier,
                "train_mc": rclgb.train_lightgbm_multiclass_classifier,
                "apply_bin": rclgb.apply_lightgbm_binary_classifier,
                "apply_mc": rclgb.apply_lightgbm_multiclass_classifier,
                "save": rclgb.save_lightgbm_classifier,
                "load": rclgb.load_lightgbm_classifier,
                "cls_kw": "lgb_classifier",
                "fast_params": {
                    "n_estimators": 15,
                    "max_depth": 3,
                    "num_leaves": 7,
                    "min_child_samples": 3,
                    "learning_rate": 0.2,
                },
                "bin_space": {
                    "n_estimators": (8, 20),
                    "num_leaves": (4, 10),
                    "min_child_samples": 3,
                },
            },
            id="lightgbm",
        ),
        pytest.param(
            {
                "pkg": "catboost",
                "mod": rccb,
                "ext": ".cbm",
                "opt_bin": rccb.optimise_catboost_binary_classifier,
                "opt_mc": rccb.optimise_catboost_multiclass_classifier,
                "train_bin": rccb.train_catboost_binary_classifier,
                "train_mc": rccb.train_catboost_multiclass_classifier,
                "apply_bin": rccb.apply_catboost_binary_classifier,
                "apply_mc": rccb.apply_catboost_multiclass_classifier,
                "save": rccb.save_catboost_classifier,
                "load": rccb.load_catboost_classifier,
                "cls_kw": "cb_classifier",
                "fast_params": {
                    "iterations": 15,
                    "depth": 3,
                    "learning_rate": 0.2,
                },
                "bin_space": {
                    "iterations": (8, 20),
                    "depth": (2, 4),
                    "learning_rate": (0.05, 0.3),
                },
            },
            id="catboost",
        ),
    ]


@pytest.mark.parametrize("spec", _backend_specs())
def test_gbdt_binary_optimise_train_and_apply(tmp_path: Path, spec: dict):
    """Verify binary Optuna search, training, persistence, and scene application."""
    pytest.importorskip(spec["pkg"])
    fix = _create_binary_fixture(tmp_path)

    model_path = tmp_path / f"bin_model{spec['ext']}"
    metrics_path = tmp_path / "bin_metrics.json"
    trained, metrics = spec["train_bin"](
        cls_cfg=fix["cfg"],
        train_data=fix["train_df"],
        test_data=fix["test_df"],
        valid_data=fix["valid_df"],
        model_params=spec["fast_params"],
        early_stopping_rounds=5,
        rnd_seed=42,
        out_model_file=model_path,
        out_metrics_file=metrics_path,
    )
    assert metrics["train_accuracy"] >= 0.95
    assert metrics["test_accuracy"] >= 0.95
    assert metrics["test_roc_auc"] >= 0.95
    assert metrics["valid_f1_binary"] >= 0.95
    assert "test_logloss" in metrics
    assert model_path.exists()
    assert json.loads(metrics_path.read_text(encoding="utf-8"))["test_roc_auc"] >= 0.95

    # Persistence round-trip (native or joblib + explicit joblib)
    jb_path = tmp_path / "bin_model_alt.joblib"
    spec["save"](trained, jb_path)
    reloaded = spec["load"](model_path)
    reloaded_jb = spec["load"](jb_path)
    assert np.allclose(
        trained.predict_proba(fix["test_df"][["B1", "B2"]]),
        reloaded.predict_proba(fix["test_df"][["B1", "B2"]]),
        atol=1e-5,
    )
    assert np.allclose(
        trained.predict_proba(fix["test_df"][["B1", "B2"]]),
        reloaded_jb.predict_proba(fix["test_df"][["B1", "B2"]]),
        atol=1e-5,
    )

    # Apply binary classifier from saved model path
    out_cls = tmp_path / "bin_cls.tif"
    out_prob = tmp_path / "bin_prob.tif"
    apply_kwargs = {
        "cls_cfg": fix["cfg"],
        spec["cls_kw"]: model_path,
        "in_msk_img": fix["msk_img"],
        "img_msk_val": 1,
        "out_class_img": out_cls,
        "out_score_img": out_prob,
        "pred_threshold": 0.5,
        "block_size": 8,
        "use_pbar": False,
    }
    spec["apply_bin"](**apply_kwargs)

    with rasterio.open(out_cls, "r") as src:
        assert src.dtypes[0] == "uint8"
        assert src.nodata == 0
        arr = src.read(1)
        assert arr[0, 0] == 0
        assert arr[2, 2] == 0
        assert set(np.unique(arr[1:8, 4:12])) == {10}
        assert set(np.unique(arr[8:15, 4:12])) == {50}

    assert rct.is_img_thematic(str(out_cls))
    clr_tab = rct.get_colour_tab_info(str(out_cls))
    assert clr_tab[50]["green"] == 200

    with rasterio.open(out_prob, "r") as src:
        assert src.count == 1
        assert src.dtypes[0] == "float32"
        assert src.nodata == pytest.approx(-1.0)
        assert src.descriptions == ("prob_Target",)
        prob = src.read(1)
        assert prob[0, 0] == pytest.approx(-1.0)
        assert prob[2, 2] == pytest.approx(-1.0)
        assert prob[4, 8] < 0.25
        assert prob[11, 8] > 0.75

    # Two-band binary probability output when score_all_classes=True
    out_prob2 = tmp_path / "bin_prob_2band.tif"
    spec["apply_bin"](
        cls_cfg=fix["cfg"],
        in_msk_img=fix["msk_img"],
        img_msk_val=1,
        out_class_img=None,
        out_score_img=out_prob2,
        score_all_classes=True,
        use_pbar=False,
        **{spec["cls_kw"]: trained},
    )
    with rasterio.open(out_prob2, "r") as src:
        assert src.count == 2
        assert list(src.descriptions) == ["Background", "Target"]

    if _HAS_OPTUNA:
        opt_csv = tmp_path / "opt_bin.csv"
        opt_json = tmp_path / "opt_bin_params.json"
        opt_mdl = tmp_path / f"opt_bin_model{spec['ext']}"
        best_est, res = spec["opt_bin"](
            cls_cfg=fix["cfg"],
            train_data=fix["train_df"],
            valid_data=fix["valid_df"],
            param_space=spec["bin_space"],
            early_stopping_rounds=4,
            n_trials=4,
            rnd_seed=42,
            out_results_file=opt_csv,
            out_params_json=opt_json,
            out_model_file=opt_mdl,
            return_results=True,
        )
        assert res["search_method"] == "optuna"
        assert res["best_score"] >= 0.9
        assert len(res["results_df"]) == 4
        assert opt_csv.exists()
        assert opt_mdl.exists()
        saved_meta = json.loads(opt_json.read_text(encoding="utf-8"))
        assert saved_meta["task"] == "binary"
        assert saved_meta["library"] == spec["pkg"]
        assert best_est.predict(fix["test_df"][["B1", "B2"]]).shape[0] == len(
            fix["test_df"]
        )


@pytest.mark.parametrize("spec", _backend_specs())
def test_gbdt_multiclass_optimise_train_and_apply(tmp_path: Path, spec: dict):
    """Verify multi-class Optuna CV search, training, and recoded uint16 output."""
    pytest.importorskip(spec["pkg"])
    fix = _create_multiclass_fixture(tmp_path)

    model, metrics = spec["train_mc"](
        cls_cfg=fix["cfg"],
        train_data=fix["train_df"],
        test_data=fix["test_df"],
        model_params=spec["fast_params"],
        rnd_seed=42,
    )
    assert metrics["train_accuracy"] >= 0.95
    assert metrics["test_accuracy"] >= 0.95

    out_cls = tmp_path / "mc_cls.tif"
    out_probs = tmp_path / "mc_probs.tif"
    spec["apply_mc"](
        cls_cfg=fix["cfg"],
        in_msk_img=fix["msk_img"],
        img_msk_val=1,
        out_class_img=out_cls,
        out_score_img=out_probs,
        block_size=8,
        use_pbar=False,
        **{spec["cls_kw"]: model},
    )

    with rasterio.open(out_cls, "r") as src:
        assert src.dtypes[0] == "uint16"
        assert src.nodata == 0
        arr = src.read(1)
        assert arr[0, 0] == 0
        assert arr[2, 2] == 0
        assert set(np.unique(arr[1:6, 5:14])) == {10}
        assert set(np.unique(arr[6:12, 5:14])) == {50}
        assert set(np.unique(arr[12:17, 5:14])) == {300}

    with rasterio.open(out_probs, "r") as src:
        assert src.count == 3
        assert list(src.descriptions) == ["Forest", "Water", "Urban"]
        probs = src.read()
        assert np.argmax(probs[:, 3, 10]) == 0
        assert np.argmax(probs[:, 9, 10]) == 1
        assert np.argmax(probs[:, 15, 10]) == 2
        assert probs[:, 9, 10].sum() == pytest.approx(1.0, abs=1e-4)

    # Single-band max probability output
    out_max = tmp_path / "mc_max_prob.tif"
    spec["apply_mc"](
        cls_cfg=fix["cfg"],
        in_msk_img=fix["msk_img"],
        img_msk_val=1,
        out_class_img=None,
        out_score_img=out_max,
        score_max_only=True,
        use_pbar=False,
        **{spec["cls_kw"]: model},
    )
    with rasterio.open(out_max, "r") as src:
        assert src.count == 1
        assert src.descriptions == ("max_class_probability",)
        assert src.read(1)[9, 10] > 0.6

    if _HAS_OPTUNA:
        # Cross-validated Optuna search using callable param_space
        cv_clf, cv_res = spec["opt_mc"](
            cls_cfg=fix["cfg"],
            train_data=fix["train_df"],
            param_space=lambda _trial: dict(spec["fast_params"]),
            cv=3,
            n_trials=3,
            rnd_seed=42,
            return_results=True,
        )
        assert cv_res["best_score"] >= 0.9
        assert cv_clf is not None


def test_gbdt_default_optuna_search_spaces(tmp_path: Path):
    """Verify default built-in Optuna search spaces work when param_space=None."""
    if not _HAS_OPTUNA:
        pytest.skip("optuna not installed")
    fix = _create_binary_fixture(tmp_path)

    for pkg, opt_fn, fixed in [
        ("xgboost", rcxgb.optimise_xgboost_binary_classifier, {"n_estimators": 10}),
        (
            "lightgbm",
            rclgb.optimise_lightgbm_binary_classifier,
            {"n_estimators": 10, "min_child_samples": 3},
        ),
        ("catboost", rccb.optimise_catboost_binary_classifier, {"iterations": 10}),
    ]:
        pytest.importorskip(pkg)
        clf, res = opt_fn(
            cls_cfg=fix["cfg"],
            train_data=fix["train_df"],
            valid_data=fix["valid_df"],
            fixed_params=fixed,
            n_trials=2,
            rnd_seed=11,
            return_results=True,
        )
        assert clf is not None
        assert res["param_space_mode"] == "default"


def test_gbdt_extended_optuna_search_spaces(tmp_path: Path):
    """Verify get_*_optuna_param_space helpers and extended_param_space=True."""
    xgb_def = rcxgb.get_xgboost_optuna_param_space(extended=False, binary=True)
    xgb_ext_bin = rcxgb.get_xgboost_optuna_param_space(extended=True, binary=True)
    xgb_ext_mc = rcxgb.get_xgboost_optuna_param_space(extended=True, binary=False)
    assert {"gamma", "min_child_weight", "reg_alpha", "reg_lambda"}.issubset(
        xgb_def
    )
    assert len(xgb_ext_bin) > len(xgb_def)
    assert "scale_pos_weight" in xgb_ext_bin
    assert "scale_pos_weight" not in xgb_ext_mc
    assert {"colsample_bylevel", "colsample_bynode", "grow_policy"}.issubset(
        xgb_ext_mc
    )

    lgb_def = rclgb.get_lightgbm_optuna_param_space(extended=False)
    lgb_ext = rclgb.get_lightgbm_optuna_param_space(extended=True)
    assert {"num_leaves", "min_split_gain", "subsample_freq"}.issubset(lgb_def)
    assert lgb_def["subsample_freq"] == 1
    assert len(lgb_ext) > len(lgb_def)
    assert {"max_bin", "path_smooth", "extra_trees", "class_weight"}.issubset(
        lgb_ext
    )

    cb_def = rccb.get_catboost_optuna_param_space(extended=False)
    cb_ext = rccb.get_catboost_optuna_param_space(extended=True)
    assert isinstance(cb_def, dict)
    assert {"rsm", "border_count", "bagging_temperature", "random_strength"}.issubset(
        cb_def
    )
    assert callable(cb_ext)

    if not _HAS_OPTUNA:
        pytest.skip("optuna not installed")

    bin_fix = _create_binary_fixture(tmp_path)
    mc_fix = _create_multiclass_fixture(tmp_path)

    for pkg, opt_bin, opt_mc, fixed in [
        (
            "xgboost",
            rcxgb.optimise_xgboost_binary_classifier,
            rcxgb.optimise_xgboost_multiclass_classifier,
            {"n_estimators": 10},
        ),
        (
            "lightgbm",
            rclgb.optimise_lightgbm_binary_classifier,
            rclgb.optimise_lightgbm_multiclass_classifier,
            {"n_estimators": 10, "min_child_samples": 3},
        ),
        (
            "catboost",
            rccb.optimise_catboost_binary_classifier,
            rccb.optimise_catboost_multiclass_classifier,
            {"iterations": 10, "depth": 4},
        ),
    ]:
        pytest.importorskip(pkg)
        clf_bin, res_bin = opt_bin(
            cls_cfg=bin_fix["cfg"],
            train_data=bin_fix["train_df"],
            valid_data=bin_fix["valid_df"],
            extended_param_space=True,
            fixed_params=fixed,
            n_trials=6,
            rnd_seed=17,
            return_results=True,
        )
        assert clf_bin is not None
        assert res_bin["param_space_mode"] == "extended"

        clf_mc, res_mc = opt_mc(
            cls_cfg=mc_fix["cfg"],
            train_data=mc_fix["train_df"],
            valid_data=mc_fix["valid_df"],
            param_space="extended",
            fixed_params=fixed,
            n_trials=5,
            rnd_seed=23,
            return_results=True,
        )
        assert clf_mc is not None
        assert res_mc["param_space_mode"] == "extended"

    if importlib.util.find_spec("xgboost") is not None:
        with pytest.raises(rsgispy.RSGISPyException, match="not both"):
            rcxgb.optimise_xgboost_binary_classifier(
                cls_cfg=bin_fix["cfg"],
                train_data=bin_fix["train_df"],
                param_space={"max_depth": (2, 4)},
                extended_param_space=True,
                n_trials=2,
            )
        with pytest.raises(rsgispy.RSGISPyException, match="Unsupported param_space"):
            rcxgb.optimise_xgboost_binary_classifier(
                cls_cfg=bin_fix["cfg"],
                train_data=bin_fix["train_df"],
                param_space="unknown_preset",
                n_trials=2,
            )


def test_gbdt_validation_and_missing_dep_errors(tmp_path: Path, monkeypatch):
    """Verify explicit validation and optional dependency errors across GBDTs."""
    mc_fix = _create_multiclass_fixture(tmp_path)
    bin_fix = _create_binary_fixture(tmp_path)

    # Binary functions reject 3-class config
    if importlib.util.find_spec("xgboost") is not None:
        with pytest.raises(rsgispy.RSGISPyException, match="exactly 2 classes"):
            rcxgb.train_xgboost_binary_classifier(
                cls_cfg=mc_fix["cfg"], train_data=mc_fix["train_df"]
            )

        # Invalid pred_threshold
        trained, _ = rcxgb.train_xgboost_binary_classifier(
            cls_cfg=bin_fix["cfg"],
            train_data=bin_fix["train_df"],
            model_params={"n_estimators": 5, "max_depth": 2},
        )
        with pytest.raises(rsgispy.RSGISPyException, match="pred_threshold"):
            rcxgb.apply_xgboost_binary_classifier(
                cls_cfg=bin_fix["cfg"],
                xgb_classifier=trained,
                in_msk_img=bin_fix["msk_img"],
                img_msk_val=1,
                out_class_img=tmp_path / "bad.tif",
                pred_threshold=1.5,
            )

        # Non-0-based labels rejected cleanly for XGBoost
        with pytest.raises(rsgispy.RSGISPyException, match="consecutive 0-based"):
            rcxgb.train_xgboost_multiclass_classifier(
                cls_cfg=mc_fix["cfg"],
                train_data=mc_fix["train_df"],
                use_internal_id=False,
            )

    # Multi-class requires >= 2 classes
    single_cfg = {
        "OnlyOne": rcd.ClassInfoObj(out_id=1),
    }
    if importlib.util.find_spec("lightgbm") is not None:
        with pytest.raises(rsgispy.RSGISPyException, match="at least 2 classes"):
            rclgb.train_lightgbm_multiclass_classifier(
                cls_cfg=single_cfg, train_data=mc_fix["train_df"]
            )

    # Missing library dependency errors
    monkeypatch.setattr(rcxgb, "HAVE_XGBOOST", False)
    with pytest.raises(rsgispy.RSGISPyException, match="xgboost"):
        rcxgb.train_xgboost_binary_classifier(bin_fix["cfg"], bin_fix["train_df"])

    monkeypatch.setattr(rclgb, "HAVE_LIGHTGBM", False)
    with pytest.raises(rsgispy.RSGISPyException, match="lightgbm"):
        rclgb.train_lightgbm_binary_classifier(bin_fix["cfg"], bin_fix["train_df"])

    monkeypatch.setattr(rccb, "HAVE_CATBOOST", False)
    with pytest.raises(rsgispy.RSGISPyException, match="catboost"):
        rccb.train_catboost_binary_classifier(bin_fix["cfg"], bin_fix["train_df"])


def test_gbdt_gpu_availability_and_validation(monkeypatch):
    """Verify GPU availability probes, mocked deps, and gpu_device_id validation."""
    for fn, mod, attr in [
        (rcxgb.is_xgboost_gpu_available, rcxgb, "HAVE_XGBOOST"),
        (rclgb.is_lightgbm_gpu_available, rclgb, "HAVE_LIGHTGBM"),
        (rccb.is_catboost_gpu_available, rccb, "HAVE_CATBOOST"),
    ]:
        res = fn()
        assert isinstance(res, bool)
        with pytest.raises(rsgispy.RSGISPyException, match="gpu_device_id"):
            fn(gpu_device_id=-1)
        with pytest.raises(rsgispy.RSGISPyException, match="gpu_device_id"):
            fn(gpu_device_id=True)
        with monkeypatch.context() as mp:
            mp.setattr(mod, attr, False)
            assert fn() is False

    # CatBoost device-index count checks
    if importlib.util.find_spec("catboost") is not None:
        with monkeypatch.context() as mp:
            mp.setattr(rccb, "HAVE_CATBOOST", True)
            mp.setattr(rccb, "_cb_get_gpu_device_count", lambda: 2)
            assert rccb.is_catboost_gpu_available() is True
            assert rccb.is_catboost_gpu_available(0) is True
            assert rccb.is_catboost_gpu_available(1) is True
            assert rccb.is_catboost_gpu_available(2) is False


@pytest.mark.parametrize("spec", _backend_specs())
def test_gbdt_gpu_fallback_and_validation(tmp_path: Path, monkeypatch, spec: dict):
    """Verify use_gpu=True falls back cleanly to CPU when no GPU is detected."""
    pytest.importorskip(spec["pkg"])
    fix = _create_binary_fixture(tmp_path)

    # Invalid gpu_device_id raises before training
    with pytest.raises(rsgispy.RSGISPyException, match="gpu_device_id"):
        spec["train_bin"](
            cls_cfg=fix["cfg"],
            train_data=fix["train_df"],
            model_params=spec["fast_params"],
            use_gpu=False,
            gpu_device_id=-1,
        )
    with pytest.raises(rsgispy.RSGISPyException, match="gpu_device_id"):
        spec["train_bin"](
            cls_cfg=fix["cfg"],
            train_data=fix["train_df"],
            model_params=spec["fast_params"],
            use_gpu=True,
            gpu_device_id=True,
        )

    # Ensure no GPU is reported so fallback path is exercised deterministically
    from dataclasses import replace  # noqa: PLC0415

    backend_attr = {
        "xgboost": "_XGB_BACKEND",
        "lightgbm": "_LGB_BACKEND",
        "catboost": "_CB_BACKEND",
    }[spec["pkg"]]
    backend_obj = getattr(spec["mod"], backend_attr)
    monkeypatch.setattr(
        spec["mod"],
        backend_attr,
        replace(backend_obj, gpu_available=lambda _dev_id: False),
    )

    # Train with use_gpu=True -> UserWarning + successful CPU fallback
    with pytest.warns(UserWarning, match="Falling back to CPU"):
        model, metrics = spec["train_bin"](
            cls_cfg=fix["cfg"],
            train_data=fix["train_df"],
            test_data=fix["test_df"],
            model_params=spec["fast_params"],
            rnd_seed=42,
            use_gpu=True,
        )
    assert metrics["test_accuracy"] >= 0.9

    # Apply with use_gpu=True -> UserWarning + successful CPU fallback
    out_cls = tmp_path / f"fb_{spec['pkg']}_cls.tif"
    with pytest.warns(UserWarning, match="Falling back to CPU"):
        spec["apply_bin"](
            cls_cfg=fix["cfg"],
            in_msk_img=fix["msk_img"],
            img_msk_val=1,
            out_class_img=out_cls,
            use_pbar=False,
            use_gpu=True,
            **{spec["cls_kw"]: model},
        )
    assert out_cls.exists()

    # Optimise with use_gpu=True -> UserWarning + JSON/result metadata recorded
    if _HAS_OPTUNA:
        opt_json = tmp_path / f"fb_{spec['pkg']}_params.json"
        with pytest.warns(UserWarning, match="Falling back to CPU"):
            _, res = spec["opt_bin"](
                cls_cfg=fix["cfg"],
                train_data=fix["train_df"],
                valid_data=fix["valid_df"],
                param_space=lambda _trial: dict(spec["fast_params"]),
                n_trials=2,
                rnd_seed=42,
                out_params_json=opt_json,
                return_results=True,
                use_gpu=True,
            )
        assert res["use_gpu_requested"] is True
        assert res["gpu_used"] is False
        saved_meta = json.loads(opt_json.read_text(encoding="utf-8"))
        assert saved_meta["use_gpu_requested"] is True
        assert saved_meta["gpu_used"] is False


def test_gbdt_gpu_parameter_mapping_and_inference(tmp_path: Path, monkeypatch):
    """Verify backend-specific GPU param mapping, prediction prep, and CV n_jobs=1."""
    # pylint: disable=protected-access
    # 1. XGBoost parameter mapping and inference device switching
    xgb_gpu_params = rcxgb._apply_xgb_gpu_params(
        {"n_estimators": 10, "gpu_id": 0, "tree_method": "approx"},
        gpu_device_id=1,
    )
    assert xgb_gpu_params["tree_method"] == "hist"
    assert xgb_gpu_params["device"] == "cuda:1"
    assert "gpu_id" not in xgb_gpu_params

    class _DummyXGBEst:
        def __init__(self, device="cpu"):
            self._device = device

        def get_params(self):
            return {"device": self._device}

        def set_params(self, **kwargs):
            if "device" in kwargs:
                self._device = kwargs["device"]
            return self

    dummy_cpu_xgb = _DummyXGBEst(device="cpu")
    eval_est, pred_kw, cleanup = rcxgb._prepare_xgb_predict_estimator(
        dummy_cpu_xgb, gpu_active=True, gpu_device_id=0
    )
    assert eval_est is dummy_cpu_xgb
    assert pred_kw == {}
    assert dummy_cpu_xgb.get_params()["device"] == "cuda:0"
    assert cleanup is not None
    cleanup()
    assert dummy_cpu_xgb.get_params()["device"] == "cpu"

    dummy_cuda_xgb = _DummyXGBEst(device="cuda:0")
    _, _, cleanup_cpu = rcxgb._prepare_xgb_predict_estimator(
        dummy_cuda_xgb, gpu_active=False, gpu_device_id=None
    )
    assert dummy_cuda_xgb.get_params()["device"] == "cpu"
    assert cleanup_cpu is not None
    cleanup_cpu()
    assert dummy_cuda_xgb.get_params()["device"] == "cuda:0"

    # 2. LightGBM parameter mapping (OpenCL gpu clamps max_bin <= 255; cuda keeps 511)
    monkeypatch.setattr(rclgb, "_detect_lightgbm_gpu_device", lambda _dev=None: "gpu")
    lgb_opencl = rclgb._apply_lgb_gpu_params(
        {"device_type": "cpu", "max_bin": 511}, gpu_device_id=0
    )
    assert "device_type" not in lgb_opencl
    assert lgb_opencl["device"] == "gpu"
    assert lgb_opencl["gpu_device_id"] == 0
    assert lgb_opencl["max_bin"] == 255

    monkeypatch.setattr(rclgb, "_detect_lightgbm_gpu_device", lambda _dev=None: "cuda")
    lgb_cuda = rclgb._apply_lgb_gpu_params(
        {"device_type": "cpu", "max_bin": 511}, gpu_device_id=2
    )
    assert lgb_cuda["device"] == "cuda"
    assert lgb_cuda["gpu_device_id"] == 2
    assert lgb_cuda["max_bin"] == 511

    # 3. CatBoost parameter mapping & predict_kwargs forwarding
    cb_gpu = rccb._apply_cb_gpu_params(
        {
            "iterations": 20,
            "rsm": 0.8,
            "bootstrap_type": "MVS",
            "subsample": 0.7,
        },
        gpu_device_id=1,
    )
    assert cb_gpu["task_type"] == "GPU"
    assert cb_gpu["devices"] == "1"
    assert "rsm" not in cb_gpu
    assert cb_gpu["bootstrap_type"] == "Bernoulli"
    assert cb_gpu["subsample"] == pytest.approx(0.7)

    _, cb_kw_on, cb_clean_on = rccb._prepare_cb_predict_estimator(
        object(), gpu_active=True, _gpu_device_id=0
    )
    assert cb_kw_on == {"task_type": "GPU"}
    assert cb_clean_on is None

    _, cb_kw_off, _ = rccb._prepare_cb_predict_estimator(
        object(), gpu_active=False, _gpu_device_id=None
    )
    assert cb_kw_off == {}

    # 4. Verify mocked GPU Optuna CV uses n_jobs=1 on cross_val_score and records gpu_used=True
    if _HAS_OPTUNA and importlib.util.find_spec("xgboost") is not None:
        import rsgispy.classification._gbdt as _gbdt  # noqa: PLC0415

        fix = _create_binary_fixture(tmp_path)
        cv_n_jobs_seen = []

        class _StubXGB:
            def __init__(self, **params):
                self.params = dict(params)
                self._device = params.get("device", "cpu")

            def get_params(self, deep=False):  # noqa: ARG002
                return dict(self.params, device=self._device)

            def set_params(self, **kwargs):
                self.params.update(kwargs)
                if "device" in kwargs:
                    self._device = kwargs["device"]
                return self

            def fit(self, x_arr, y_arr):  # noqa: ARG002
                return self

            def predict(self, x_arr):
                return np.zeros(len(x_arr), dtype=np.int32)

            def predict_proba(self, x_arr):
                probs = np.full((len(x_arr), 2), 0.5, dtype=np.float32)
                return probs

        def _fake_cv_score(est, x_arr, y_arr, cv=None, scoring=None, n_jobs=None):
            # noqa: ARG001
            cv_n_jobs_seen.append((n_jobs, est.get_params().get("device")))
            return np.array([0.95, 0.95])

        mock_backend = GBDTBackendSpec = type(rcxgb._XGB_BACKEND)(
            name=rcxgb._XGB_BACKEND.name,
            package_name=rcxgb._XGB_BACKEND.package_name,
            extra=rcxgb._XGB_BACKEND.extra,
            is_available=lambda: True,
            gpu_available=lambda _dev: True,
            apply_gpu_params=rcxgb._apply_xgb_gpu_params,
            seed_param=rcxgb._XGB_BACKEND.seed_param,
            jobs_param=rcxgb._XGB_BACKEND.jobs_param,
            default_binary_params=rcxgb._XGB_BACKEND.default_binary_params,
            default_multiclass_params=rcxgb._XGB_BACKEND.default_multiclass_params,
            default_binary_space=rcxgb._XGB_BACKEND.default_binary_space,
            default_multiclass_space=rcxgb._XGB_BACKEND.default_multiclass_space,
            extended_binary_space=rcxgb._XGB_BACKEND.extended_binary_space,
            extended_multiclass_space=rcxgb._XGB_BACKEND.extended_multiclass_space,
            build_estimator=lambda p, _base: _StubXGB(**p),
            fit_estimator=lambda est, x_df, y_arr, *_rest: est.fit(x_df, y_arr),
            load_model=rcxgb._XGB_BACKEND.load_model,
            save_model=rcxgb._XGB_BACKEND.save_model,
            prepare_predict_estimator=rcxgb._prepare_xgb_predict_estimator,
            require_consecutive_zero_labels=True,
        )
        assert GBDTBackendSpec is not None
        monkeypatch.setattr(rcxgb, "_XGB_BACKEND", mock_backend)
        monkeypatch.setattr(
            _gbdt.sklearn.model_selection, "cross_val_score", _fake_cv_score
        )
        _, opt_res = rcxgb.optimise_xgboost_binary_classifier(
            cls_cfg=fix["cfg"],
            train_data=fix["train_df"],
            valid_data=None,
            param_space=lambda _t: {"n_estimators": 5},
            cv=2,
            n_trials=2,
            n_jobs=4,
            rnd_seed=42,
            use_gpu=True,
            gpu_device_id=0,
            return_results=True,
        )
        assert opt_res["use_gpu_requested"] is True
        assert opt_res["gpu_used"] is True
        assert cv_n_jobs_seen == [(1, "cuda:0"), (1, "cuda:0")]

    # 5. Verify CatBoost predict_kwargs forwarding in _cls_common and apply_gbdt_classifier
    import rsgispy.classification._cls_common as _cls_common  # noqa: PLC0415

    recorded_predict_kwargs = []

    class _StubCBEst:
        def predict(self, x_arr, **kwargs):
            recorded_predict_kwargs.append(("predict", dict(kwargs)))
            return np.zeros(len(x_arr), dtype=np.int32)

        def predict_proba(self, x_arr, **kwargs):
            recorded_predict_kwargs.append(("predict_proba", dict(kwargs)))
            return np.tile(np.array([[0.2, 0.8]], dtype=np.float32), (len(x_arr), 1))

    fix_cb = _create_binary_fixture(tmp_path)
    _cls_common.calc_cls_split_metrics(
        _StubCBEst(),
        fix_cb["test_df"][["B1", "B2"]],
        fix_cb["test_df"]["internal_cls_id"].to_numpy(),
        "test",
        binary=True,
        pos_train_id=1,
        predict_kwargs={"task_type": "GPU"},
    )
    assert ("predict", {"task_type": "GPU"}) in recorded_predict_kwargs
    assert ("predict_proba", {"task_type": "GPU"}) in recorded_predict_kwargs


def test_xgboost_feature_order_subset_and_id_mismatch_guards(tmp_path: Path):
    """
    Verify raster apply aligns reordered/subset feature_cols, raises when
    use_internal_id mismatches trained classes, and preserves multiclass
    objective when a default XGBClassifier instance is supplied.
    """
    xgb = pytest.importorskip("xgboost")
    fix = _create_multiclass_fixture(tmp_path)

    # 1. Passing a default XGBClassifier() (objective='binary:logistic')
    #    to train_xgboost_multiclass_classifier must preserve multi:softprob
    base_clf = xgb.XGBClassifier(n_estimators=15, max_depth=3)
    trained_mc, mc_metrics = rcxgb.train_xgboost_multiclass_classifier(
        cls_cfg=fix["cfg"],
        train_data=fix["train_df"],
        xgb_classifier=base_clf,
        rnd_seed=42,
    )
    assert trained_mc.get_params()["objective"] == "multi:softprob"
    assert mc_metrics["train_accuracy"] >= 0.95

    # 2. Training with reversed feature_cols=['B2', 'B1'] and applying to a
    #    raster with ['B1', 'B2'] must reorder raster bands to match
    rev_mdl, _ = rcxgb.train_xgboost_multiclass_classifier(
        cls_cfg=fix["cfg"],
        train_data=fix["train_df"],
        feature_cols=["B2", "B1"],
        model_params={"n_estimators": 20, "max_depth": 3},
        rnd_seed=42,
    )
    out_rev = tmp_path / "mc_rev_cls.tif"
    rcxgb.apply_xgboost_multiclass_classifier(
        cls_cfg=fix["cfg"],
        xgb_classifier=rev_mdl,
        in_msk_img=fix["msk_img"],
        img_msk_val=1,
        out_class_img=out_rev,
        use_pbar=False,
    )
    with rasterio.open(out_rev, "r") as src:
        arr_rev = src.read(1)
        assert set(np.unique(arr_rev[1:6, 5:14])) == {10}
        assert set(np.unique(arr_rev[6:12, 5:14])) == {50}
        assert set(np.unique(arr_rev[12:17, 5:14])) == {300}

    # 3. Training on a 1-feature subset feature_cols=['B1'] (e.g. after feature
    #    selection) and applying with 2-band cls_cfg must subset raster bands
    sub_mdl, _ = rcxgb.train_xgboost_multiclass_classifier(
        cls_cfg=fix["cfg"],
        train_data=fix["train_df"],
        feature_cols=["B1"],
        model_params={"n_estimators": 20, "max_depth": 3},
        rnd_seed=42,
    )
    out_sub = tmp_path / "mc_sub_cls.tif"
    rcxgb.apply_xgboost_multiclass_classifier(
        cls_cfg=fix["cfg"],
        xgb_classifier=sub_mdl,
        in_msk_img=fix["msk_img"],
        img_msk_val=1,
        out_class_img=out_sub,
        use_pbar=False,
    )
    with rasterio.open(out_sub, "r") as src:
        arr_sub = src.read(1)
        assert len(np.unique(arr_sub[arr_sub > 0])) >= 2

    # 4. Mismatched use_internal_id between train (True -> [0, 1, 2]) and apply
    #    (False -> expects [10, 50, 300]) raises a clear RSGISPyException
    with pytest.raises(
        rsgispy.RSGISPyException,
        match="do not match any expected target class IDs",
    ):
        rcxgb.apply_xgboost_multiclass_classifier(
            cls_cfg=fix["cfg"],
            xgb_classifier=trained_mc,
            in_msk_img=fix["msk_img"],
            img_msk_val=1,
            out_class_img=tmp_path / "mc_mismatch.tif",
            use_internal_id=False,
            use_pbar=False,
        )

    # 5. Model trained on a feature name missing from a 1-band img_band_info
    #    raises a clear RSGISPyException
    one_band_info = [
        rcd.ImageBandInfo(
            file_name=fix["cfg"].img_band_info[0].file_name,
            name="feat",
            bands=[1],
            band_names=["B1"],
        )
    ]
    with pytest.raises(
        rsgispy.RSGISPyException,
        match="do not match the 1 configured image band",
    ):
        rcxgb.apply_xgboost_multiclass_classifier(
            cls_cfg=fix["cfg"],
            xgb_classifier=trained_mc,
            in_msk_img=fix["msk_img"],
            img_msk_val=1,
            img_band_info=one_band_info,
            out_class_img=tmp_path / "mc_missing_band.tif",
            use_pbar=False,
        )

    # 6. Applying with an img_msk_val that matches 0 valid pixels emits a
    #    UserWarning explaining why the output raster contains only nodata
    with pytest.warns(UserWarning, match="No valid pixels were classified"):
        rcxgb.apply_xgboost_multiclass_classifier(
            cls_cfg=fix["cfg"],
            xgb_classifier=trained_mc,
            in_msk_img=fix["msk_img"],
            img_msk_val=99,
            out_class_img=tmp_path / "mc_empty_msk.tif",
            calc_img_stats=False,
            use_pbar=False,
        )
