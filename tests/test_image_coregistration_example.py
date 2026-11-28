"""
Run the documented image co-registration worked example
(``docs/examples/image_coregistration.py``) end-to-end on a synthetic image
pair so the code shown in ``docs/guides/image_coregistration.rst`` stays
correct.
"""

import importlib.util
import os

import pandas as pd
import rasterio

from coreg_scene import SHIFT, make_coreg_scene

import pytest  # noqa: E402

pytestmark = pytest.mark.slow

EXAMPLE = os.path.join(
    os.path.dirname(__file__), os.pardir, "docs", "examples", "image_coregistration.py"
)


def _load_example():
    spec = importlib.util.spec_from_file_location("image_coregistration", EXAMPLE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_worked_example_runs(tmp_path):
    mod = _load_example()
    scene = make_coreg_scene(str(tmp_path / "in"))
    out = mod.run_workflow(
        scene["ref"],
        scene["flt"],
        str(tmp_path / "out"),
        gcp_vec=scene["gcps"],
        ref_band=2,
        flt_band=2,
        win_size=32,
        pixel_gap=24,
        search_area=6,
        use_pbar=False,
    )
    # Step 1: the whole-image offset undoes the simulated shift (to 0.5 px)
    assert abs(out["offset"]["x_offset_map"] + SHIFT[0]) <= 5.0
    assert abs(out["offset"]["y_offset_map"] + SHIFT[1]) <= 5.0

    # Steps 3-5: tie points, filtering and model choice
    assert out["n_tie_points"] >= 10
    assert out["diag"]["n_inliers"] >= 10
    assert len(pd.read_csv(out["tie_file"])) == out["diag"]["n_inliers"]
    chk = out["chk_rmse"]
    assert chk["affine"] < chk["shift"]  # the rotation needs an affine model

    # Steps 6-8: outputs on the reference grid
    with rasterio.open(scene["ref"]) as ref, rasterio.open(out["warp_img"]) as reg:
        assert reg.transform == ref.transform and reg.shape == ref.shape
    assert out["coreg_info"]["n_inliers"] >= 10
    assert os.path.exists(out["gcp_warp_img"])

    # Step 9: every method improves the match; residual offset is sub-pixel
    assess = out["assess"]
    assert assess["before"]["resid_offset_m"] > 20.0
    for name in ("shift", "affine", "gcps"):
        assert assess[name]["norm_mutual_info"] > assess["before"]["norm_mutual_info"]
        assert assess[name]["resid_offset_m"] < 5.0
    assert assess["affine"]["correlation"] > assess["shift"]["correlation"]
    assert os.path.exists(out["diff_img"])
