"""
Run the documented SAR processing example (``docs/examples/sar_processing.py``)
end-to-end on synthetic data so the code shown in
``docs/guides/sar_processing.rst`` stays correct.
"""

import importlib.util
import os

import numpy as np
import pytest
import rasterio

from sar_scene import CLASS_DB, CLASS_IDS, ENL, make_sar_scene

pytestmark = pytest.mark.slow

EXAMPLE = os.path.join(
    os.path.dirname(__file__), os.pardir, "docs", "examples", "sar_processing.py"
)


def _load_example():
    spec = importlib.util.spec_from_file_location("sar_processing", EXAMPLE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("in_db", [False, True])
def test_worked_example_runs(tmp_path, in_db):
    mod = _load_example()
    scene = make_sar_scene(str(tmp_path / "scene"), seed=0, in_db=in_db)
    out = mod.run_workflow(
        scene["s1"],
        scene["train"],
        scene["homog"],
        str(tmp_path / "out"),
        in_units="dB" if in_db else "power",
        fig_file=None if in_db else str(tmp_path / "sar_fig.png"),
        use_pbar=False,
    )
    if not in_db:
        assert os.path.getsize(out["fig_file"]) > 0
    # Speckle check: the raw image has ENL close to the simulated looks and
    # every filter increases it without changing the mean backscatter.
    enl = out["enl"]
    raw = enl[enl["filter"] == "raw"].set_index("area_id")
    assert np.all(np.abs(raw["enl"] - ENL) < 1.5)
    for name in ("lee", "refined_lee", "frost", "gamma_map"):
        filt = enl[enl["filter"] == name].set_index("area_id")
        assert np.all(filt["enl"] > 3 * raw["enl"])
        np.testing.assert_allclose(filt["mean"], raw["mean"], rtol=0.05)

    with rasterio.open(out["db_img"]) as src:
        assert src.nodata == -9999.0
        assert src.descriptions == ("VV_dB", "VH_dB")
        db_vals = src.read()
    assert np.all(db_vals[:, :, :2] == -9999.0)  # no data edge kept

    with rasterio.open(out["idx_img"]) as src:
        assert src.descriptions == (
            "RVI_VV_VH", "CR_VH_VV_dB", "DpRVIc_VV_VH", "NDPI_VV_VH"
        )
    with rasterio.open(out["glcm_img"]) as src:
        assert src.count == 6
    with rasterio.open(out["chain_img"]) as src:
        assert src.count == 2

    # Class means of the dB features match the simulated backscatter.
    stats = out["cls_stats"].set_index("cls_id")
    assert len(out["feat_names"]) == 12
    for cls_name, (vv_db, vh_db) in CLASS_DB.items():
        row = stats.loc[CLASS_IDS[cls_name]]
        assert row["VV_dB_mean"] == pytest.approx(vv_db, abs=1.0)
        assert row["VH_dB_mean"] == pytest.approx(vh_db, abs=1.0)
