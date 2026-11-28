"""
Run the documented multi-criteria analysis worked example
(``docs/examples/multi_criteria_analysis.py``) end-to-end on a synthetic
coastal scene so the code shown in
``docs/guides/multi_criteria_analysis.rst`` stays correct.
"""

import importlib.util
import os

import geopandas as gpd
import numpy as np
import pytest
import rasterio

from mca_scene import make_mca_scene

pytest.importorskip("scipy")  # calc_dist_to_img_vals

EXAMPLE = os.path.join(
    os.path.dirname(__file__), os.pardir, "docs", "examples",
    "multi_criteria_analysis.py",
)


def _load_example():
    spec = importlib.util.spec_from_file_location("multi_criteria_analysis", EXAMPLE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_worked_example_runs(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    mod = _load_example()
    scene = make_mca_scene(str(tmp_path / "in"))
    out = mod.run_workflow(
        scene["s2"],
        scene["dem"],
        scene["mng_now"],
        scene["mng_hist"],
        scene["ponds"],
        scene["settlements"],
        scene["protected"],
        scene["admin"],
        str(tmp_path / "out"),
        s2_bands={1: "blue", 2: "green", 3: "red", 4: "nir", 5: "swir1", 6: "swir2"},
        use_pbar=False,
    )

    # AHP weights sum to one and the pairwise matrix is consistent.
    assert sum(out["weights"].values()) == pytest.approx(1.0, abs=1e-3)
    assert out["cons_ratio"] < 0.1

    with rasterio.open(out["factors_img"]) as src:
        assert src.count == 7
        fac = src.read()
    assert np.nanmin(fac) >= 0.0 and np.nanmax(fac) <= 1.0

    with rasterio.open(out["potential_img"]) as src:
        assert src.descriptions == ("potential", "score")
        assert src.colormap(1)[4][:3] == (0, 104, 55)
        pot = src.read(1)
        score = src.read(2)
    # No data strip on the western edge, current mangroves and active ponds
    # are excluded, the abandoned (former mangrove) ponds have high potential.
    assert np.all(pot[:, :2] == 0) and np.all(score[:, :2] == 255)
    assert np.all(pot[2:24, 36:45] == 1)  # current mangrove
    assert np.all(pot[36:58, 43:49] == 1)  # active ponds
    assert np.all(pot[36:58, 35:41] == 4)  # abandoned ponds
    assert np.all(score[pot == 1] == 0)
    assert set(np.unique(pot)) <= {0, 1, 2, 3, 4}

    sens = out["sensitivity"]
    assert len(sens) == 15
    assert sens["changed"].between(0.0, 1.0).all()

    admin = out["admin_gdf"]
    assert {"high_ha", "medium_ha", "low_ha", "score_mean"} <= set(admin.columns)
    assert admin["high_ha"].sum() > 0
    assert len(out["high_polys"]) >= 1
    assert (out["high_polys"]["area_ha"] >= 0.5).all()
    assert len(gpd.read_file(os.path.join(str(tmp_path / "out"),
                                          "high_potential.gpkg"))) >= 1
    assert os.path.getsize(out["map_png"]) > 0
