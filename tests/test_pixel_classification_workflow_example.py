"""
Run the documented pixel-based classification worked example
(``docs/examples/pixel_classification_workflow.py``) end-to-end on synthetic
data so the code shown in ``docs/guides/pixel_classification_workflow.rst``
stays correct.
"""

import importlib.util
import os

import pytest
import rasterio

from pixel_ml_scene import CLASS_COLOURS, CLASS_IDS, make_pixel_scene

pytest.importorskip("sklearn")
pytest.importorskip("optuna")
pytest.importorskip("matplotlib")

EXAMPLE = os.path.join(
    os.path.dirname(__file__), os.pardir, "docs", "examples",
    "pixel_classification_workflow.py",
)


def _load_example():
    spec = importlib.util.spec_from_file_location(
        "pixel_classification_workflow", EXAMPLE
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _has(mod_name):
    return importlib.util.find_spec(mod_name) is not None


def test_worked_example_runs(tmp_path):
    """The example runs end to end and gives sensible outputs."""
    mod = _load_example()
    scene = make_pixel_scene(str(tmp_path / "scene"))
    train = {
        name: (scene[f"train_{name.lower()}"], 10 * cls_id, CLASS_COLOURS[name])
        for name, cls_id in CLASS_IDS.items()
    }
    out = mod.run_workflow(
        scene["s2"],
        scene["dem"],
        train,
        str(tmp_path / "out"),
        acc_pts_vec=scene["acc_pts"],
        cv_block_size=200.0,
        n_trees=20,
        n_trials=3,
        mmu_pxls=4,
        n_acc_smpls=60,
        salib_n_samples=32,
        use_gbdt=_has("xgboost"),
        use_boruta=_has("shap"),
        use_shap=_has("shap"),
        use_salib=_has("SALib"),
        use_pbar=False,
    )
    assert out["n_train"] > 0
    # The grouped split never puts one training polygon in two tables.
    train_p, valid_p, test_p = out["split_polygons"]
    assert not (train_p & valid_p or train_p & test_p or valid_p & test_p)
    assert len(train_p | valid_p | test_p) == 2 * 3 * len(CLASS_IDS)
    with rasterio.open(out["feat_vrt"]) as src:
        assert {"uint16", "float32"} <= set(src.dtypes)  # one mixed-type stack
    assert {"corr", "vif", "rfe", "model"} <= set(out["selections"])
    assert out["rf_metrics"]["test_accuracy"] > 0.9
    assert out["nested_cv"].mean_score > 0.8
    assert {"feature", "importance_mean"} <= set(out["importance"].columns)
    for key in ("cls_img", "cls_pp_img"):
        with rasterio.open(out[key]) as src:
            assert src.colormap(1)[10][:3] == (31, 120, 180)  # Water colour
    with rasterio.open(os.path.join(str(tmp_path / "out"), "rf_class_probs.tif")) as src:
        assert src.count == len(CLASS_IDS)
    assert out["acc_stats"]["accuracy"] > 0.9
    assert os.path.exists(out["map_file"])
    assert os.path.exists(out["clu_img"])
    assert os.path.exists(str(tmp_path / "out" / "acc_pts_to_label.gpkg"))
