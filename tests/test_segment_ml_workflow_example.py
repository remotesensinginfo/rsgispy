"""
Run the documented object-based (segment) ML worked example
(``docs/examples/segment_ml_workflow.py``) end-to-end on synthetic data so the
code shown in ``docs/guides/segment_ml_workflow.rst`` stays correct.
"""

import importlib.util
import os

import geopandas as gpd
import pytest
import rasterio

from rsgispy.imageutils.colourtable import get_thematic_classes

from seg_ml_scene import CLASS_IDS, make_seg_scene

pytest.importorskip("sklearn")

pytestmark = pytest.mark.slow

EXAMPLE = os.path.join(
    os.path.dirname(__file__), os.pardir, "docs", "examples", "segment_ml_workflow.py"
)


def _load_example():
    spec = importlib.util.spec_from_file_location("segment_ml_workflow", EXAMPLE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_worked_example_runs(tmp_path):
    mod = _load_example()
    scene = make_seg_scene(str(tmp_path / "a"), seed=0, tag="a")
    scene_b = make_seg_scene(str(tmp_path / "b"), seed=9, tag="b")
    colours = {"Water": "#1f78b4", "Forest": "#33a02c", "Grass": "#b2df8a",
               "Urban": "#808080"}
    train = {
        name: (scene[f"train_{name.lower()}"], 10 * CLASS_IDS[name], colours[name])
        for name in CLASS_IDS
    }
    has_xgb = importlib.util.find_spec("xgboost") is not None
    out = mod.run_workflow(
        scene["s2"],
        scene["slope"],
        train,
        scene["plots"],
        scene["acc_pts"],
        str(tmp_path / "out"),
        s2_img_new=scene_b["s2"],
        slope_img_new=scene_b["slope"],
        s2_bands=(1, 2, 3, 4),
        s2_band_names=("Blue", "Green", "Red", "NIR"),
        n_clusters=30,
        min_pixels=8,
        sampling=1,
        use_gbdt=has_xgb,
        use_pbar=False,
    )
    assert out["n_train"] > 0
    assert out["pt_stats"]["accuracy"] > 0.9
    cls = gpd.read_parquet(out["cls_vec"])
    assert {"seg_id", "cls_id", "cls_name", "cls_prob_max"} <= set(cls.columns)
    with rasterio.open(out["cls_img"]) as src:
        assert src.colormap(1) is not None
    names = {name for name, _ in get_thematic_classes(out["cls_img"]).values()}
    assert names == set(CLASS_IDS)
    assert os.path.exists(out["new_cls_vec"])
    assert "pred_canopy_cover" in gpd.read_parquet(out["reg_vec"]).columns
    assert "cluster_id" in gpd.read_parquet(out["clu_vec"]).columns
