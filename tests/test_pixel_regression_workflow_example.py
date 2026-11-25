"""
Run the documented pixel-based regression worked example
(``docs/examples/pixel_regression_workflow.py``) end-to-end on synthetic data
so the code shown in ``docs/guides/pixel_regression_workflow.rst`` stays
correct.
"""

import importlib.util
import os

import geopandas as gpd
import pytest
import rasterio

from pixel_ml_scene import make_pixel_scene

pytest.importorskip("sklearn")
pytest.importorskip("matplotlib")

EXAMPLE = os.path.join(
    os.path.dirname(__file__), os.pardir, "docs", "examples",
    "pixel_regression_workflow.py",
)
GUIDE = os.path.join(
    os.path.dirname(__file__), os.pardir, "docs", "guides",
    "pixel_regression_workflow.rst",
)


def _load_example():
    spec = importlib.util.spec_from_file_location("pixel_regression_workflow", EXAMPLE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_worked_example_runs(tmp_path):
    """The example runs end to end and gives sensible outputs."""
    mod = _load_example()
    scene = make_pixel_scene(str(tmp_path / "scene"))
    out = mod.run_workflow(
        scene["s2"],
        scene["dem"],
        scene["plots"],
        scene["plots_valid"],
        str(tmp_path / "out"),
        cv_block_size=200.0,
        n_trees=50,
        use_gbdt=importlib.util.find_spec("xgboost") is not None,
        use_pbar=False,
    )
    assert out["n_train"] > 0
    assert out["rf_metrics"]["test_r2"] > 0.8
    assert out["nested_cv"].overall_metrics["r2"] > 0.8
    assert out["val_metrics"]["r2"] > 0.8
    assert 0.75 <= out["coverage"] <= 1.0
    with rasterio.open(out["pred_img"]) as src:
        assert src.count == 1
    with rasterio.open(out["conf_img"]) as src:
        assert src.count == 3
    resid = gpd.read_file(str(tmp_path / "out" / "valid_plots_residuals.gpkg"))
    assert "residual" in resid.columns
    assert os.path.exists(out["map_file"])


def test_guide_markers_exist():
    """Every literalinclude marker used by the guide exists in the script."""
    with open(EXAMPLE, encoding="utf-8") as f_obj:
        script = f_obj.read()
    with open(GUIDE, encoding="utf-8") as f_obj:
        guide = f_obj.read()
    markers = [
        line.split(":start-after:")[1].strip()
        for line in guide.splitlines()
        if ":start-after:" in line
    ]
    assert len(markers) >= 10
    for marker in markers:
        assert marker in script
        assert marker.replace("[step-", "[end-step-") in script
