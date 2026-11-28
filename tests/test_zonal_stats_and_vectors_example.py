"""
Run the documented zonal statistics and vector analysis worked example
(``docs/examples/zonal_stats_and_vectors.py``) end-to-end on synthetic data so
the code shown in ``docs/guides/zonal_stats_and_vectors.rst`` stays correct.
"""

import importlib.util
import os

import geopandas as gpd
import numpy as np
import pytest

from zonal_scene import FIELD_PX, make_zonal_scene

pytest.importorskip("h3")
pytest.importorskip("pyarrow")

EXAMPLE = os.path.join(
    os.path.dirname(__file__), os.pardir, "docs", "examples",
    "zonal_stats_and_vectors.py",
)


def _load_example():
    spec = importlib.util.spec_from_file_location("zonal_stats_and_vectors", EXAMPLE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_worked_example_runs(tmp_path):
    """The worked example runs and its outputs are consistent."""
    mod = _load_example()
    scene = make_zonal_scene(str(tmp_path / "in"))
    out = mod.run_workflow(
        scene["s2"],
        scene["fields"],
        scene["plots"],
        scene["roi"],
        str(tmp_path / "out"),
        s2_bands=(1, 2, 3, 4),
        hex_res=10,
        use_pbar=False,
    )
    # Polygon and label-raster statistics agree.
    assert out["max_diff"] < 1e-6

    fields = gpd.read_file(out["fields_ndvi_vec"])
    assert {"ndvi_mean", "ndvi_std", "ndvi_count"} <= set(fields.columns)
    assert fields["ndvi_mean"].between(-1, 1).all()
    # The no data strip on the right edge is excluded from the counts.
    assert fields["ndvi_count"].min() < FIELD_PX * FIELD_PX
    assert fields["ndvi_count"].max() == FIELD_PX * FIELD_PX

    pts = gpd.read_file(out["plots_vals_vec"])
    assert {"blue", "green", "red", "nir"} <= set(pts.columns)
    assert (pts["nir"] > 0).all()

    stats = out["stats_df"]
    assert {"field_id", "pxl_count", "NIR_mean", "ndvi_median"} <= set(stats.columns)
    assert len(stats) == 12

    hexes = gpd.read_file(out["hex_vec"])
    assert len(hexes) > 3
    assert {"hex_id", "ndvi_mean", "n_plots"} <= set(hexes.columns)
    roi_pts = gpd.read_file(scene["plots"]).clip(gpd.read_file(scene["roi"]))
    assert hexes["n_plots"].sum() >= len(roi_pts)

    shape = gpd.read_file(out["fields_shape_vec"])
    np.testing.assert_allclose(shape["shp_area"], (FIELD_PX * 10.0) ** 2)
    np.testing.assert_allclose(shape["shp_compactness"], np.pi / 4)

    assert len(out["roi_fields_gdf"]) == 6

    train = gpd.read_file(out["train_vec"])
    test = gpd.read_file(out["test_vec"])
    assert len(train) + len(test) == len(pts)
    assert 0.2 < len(test) / len(pts) < 0.4
