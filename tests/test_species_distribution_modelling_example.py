"""
Run the documented species distribution modelling worked example
(``docs/examples/species_distribution_modelling.py``) end-to-end on synthetic
data so the code shown in ``docs/guides/species_distribution_modelling.rst``
stays correct.
"""

import importlib.util
import os

import matplotlib
import numpy as np
import pytest
import rasterio

from sdm_scene import make_sdm_scene

pytest.importorskip("sklearn")
pytest.importorskip("scipy")
matplotlib.use("Agg")

EXAMPLE = os.path.join(
    os.path.dirname(__file__), os.pardir, "docs", "examples",
    "species_distribution_modelling.py",
)
GUIDE = os.path.join(
    os.path.dirname(__file__), os.pardir, "docs", "guides",
    "species_distribution_modelling.rst",
)


def _load_example():
    spec = importlib.util.spec_from_file_location("sdm_example", EXAMPLE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_worked_example_runs(tmp_path):
    mod = _load_example()
    scene = make_sdm_scene(str(tmp_path / "scene"))
    env_imgs = {
        "elevation": (scene["elevation"], 1),
        "temp": (scene["climate"], 1),
        "precip": (scene["climate"], 2),
        "ndvi": (scene["ndvi"], 1),
    }
    out_dir = tmp_path / "out"
    out = mod.run_workflow(
        env_imgs,
        scene["roi_mask"],
        scene["presences"],
        str(out_dir),
        presence_lyr="presences",
        drop_vars=("elevation",),
        n_pseudo_abs=300,
        pseudo_abs_dist=150.0,
        n_trees=30,
        n_kfolds=4,
        cv_block_size=1000.0,
        salib_n_samples=64,
        use_pbar=False,
    )
    assert out["n_pres"] > 50
    assert out["n_abs"] > 200
    assert out["n_mdls"] == 4
    assert out["test_auc"] > 0.8
    assert out["ens_metrics"]["Test AUC"] > 0.8
    assert out["ens_auc"] == pytest.approx(out["ens_metrics"]["Test AUC"], abs=1e-6)
    # Random CV is optimistic relative to spatial block CV (usually).
    assert 0.5 < out["blk_cv_auc"] <= 1.0
    assert set(out["jk_df"].index) == {"temp", "precip", "ndvi"}
    assert any(col.endswith("_resp") for col in out["resp_df"].columns)
    assert os.path.exists(out_dir / "plots" / "roc_curve.png")

    with rasterio.open(out["prob_img"]) as src:
        prob = src.read(1)
        assert src.nodata == -1.0
    with rasterio.open(out["vld_msk_img"]) as src:
        vld = src.read(1)
    assert np.all(prob[vld == 1] >= 0.0) and np.all(prob[vld == 1] <= 1.0)
    assert np.all(prob[vld == 0] == -1.0)
    assert 0.0 < out["prob_thres"] < 1.0
    assert out["bin_metrics"]["Hanssen-Kuiper skill"] > 0.5
    with rasterio.open(out["pa_img"]) as src:
        pa_arr = src.read(1)
    assert set(np.unique(pa_arr)) <= {0, 1, 2}
    assert np.all(pa_arr[vld == 0] == 0)
    assert 0.0 < out["suit_prop"] < 1.0


def test_guide_markers_exist():
    """Every literalinclude marker used by the guide exists in the script."""
    with open(EXAMPLE, encoding="utf-8") as f_obj:
        script = f_obj.read()
    with open(GUIDE, encoding="utf-8") as f_obj:
        guide = f_obj.read()
    markers = [
        line.split(":start-after:")[1].strip()
        for line in guide.splitlines()
        if ":start-after:" in line and "[step-" in line
    ]
    assert len(markers) >= 11
    for marker in markers:
        assert marker in script
        assert marker.replace("[step-", "[end-step-") in script
