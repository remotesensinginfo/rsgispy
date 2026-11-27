"""
Run the documented change detection worked example
(``docs/examples/change_detection.py``) end-to-end on synthetic data so the
code shown in ``docs/guides/change_detection.rst`` stays correct.
"""

import importlib.util
import os

import matplotlib
import numpy as np
import pytest
import rasterio

from rsgispy.imageutils.colourtable import get_thematic_classes

from change_scene import make_change_scene

pytest.importorskip("sklearn")
matplotlib.use("Agg")

EXAMPLE = os.path.join(
    os.path.dirname(__file__), os.pardir, "docs", "examples", "change_detection.py"
)
GUIDE = os.path.join(
    os.path.dirname(__file__), os.pardir, "docs", "guides", "change_detection.rst"
)


def _load_example():
    spec = importlib.util.spec_from_file_location("change_detection", EXAMPLE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _read(path):
    with rasterio.open(path) as src:
        return src.read(1)


def test_worked_example_runs(tmp_path):
    mod = _load_example()
    scene = make_change_scene(str(tmp_path / "scene"))
    out = mod.run_workflow(
        scene["s2_t1"],
        scene["s2_t2"],
        scene["lc_t1"],
        str(tmp_path / "out"),
        forest_id=2,
        acc_pts_vec=scene["acc_pts"],
        acc_pts_lyr="acc_pts",
        band_map={1: "blue", 2: "green", 3: "red", 4: "nir", 5: "swir1",
                  6: "swir2"},
        cva_bands=(3, 4, 5, 6),
        n_acc_pts=100,
        use_pbar=False,
    )
    ref = _read(scene["ref_chng"])
    truth = ref == 2

    vld = _read(out["vld_img"])
    assert np.all(vld[ref == 0] == 0)  # cloud hole is not valid

    chng = _read(out["chng_img"])
    assert np.all(chng[ref == 0] == 0)
    assert np.mean(chng[truth] == 2) > 0.95
    assert np.mean(chng[ref == 1] == 1) > 0.98
    assert 0.02 < out["chng_prop"] < 0.08
    assert set(out["thresholds"]) == {"otsu", "yen", "li"}

    classes = get_thematic_classes(out["chng_img"])
    assert classes[2][0] == "Vegetation loss"

    cva_dir = _read(out["cva_dir_img"])
    assert np.mean(cva_dir[truth] == 2) > 0.95
    assert get_thematic_classes(out["cva_dir_img"])[3][0] == "Other change"
    cva_info = out["cva_info"]
    assert cva_info["threshold_method"] == "otsu"
    assert 0 < cva_info["n_change"] < cva_info["n_valid"]

    if "mad_img" in out:
        mad = _read(out["mad_img"])
        assert np.mean(mad[truth] == 1) > 0.9
        assert np.all(mad[ref == 0] == 255)
        assert out["mad_info"]["n_iter"] >= 1
        assert isinstance(out["mad_info"]["converged"], bool)

    loss = _read(out["forest_loss_img"])
    assert np.mean(loss[truth] == 2) > 0.95
    assert np.all(loss[:30, :40] == 0)  # water is not forest

    acc = out["acc_stats"]
    assert acc["accuracy"] > 0.95
    assert os.path.exists(tmp_path / "out" / "acc_pts_to_label.gpkg")


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
    assert len(markers) >= 7
    for marker in markers:
        assert marker in script
        assert marker.replace("[step-", "[end-step-") in script
