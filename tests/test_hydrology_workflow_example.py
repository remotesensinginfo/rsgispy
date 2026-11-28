"""
Run the documented DEM hydrology worked example
(``docs/examples/hydrology_workflow.py``) end-to-end on a synthetic catchment
so the code shown in ``docs/guides/hydrology_workflow.rst`` stays correct.
"""

import importlib.util
import os

import geopandas as gpd
import pytest
import rasterio

from hydro_scene import LC_NAMES, OUT_EPSG, make_hydro_scene

pytest.importorskip("matplotlib")

pytestmark = pytest.mark.slow

EXAMPLE = os.path.join(
    os.path.dirname(__file__), os.pardir, "docs", "examples", "hydrology_workflow.py"
)


def _load_example():
    spec = importlib.util.spec_from_file_location("hydrology_workflow", EXAMPLE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_worked_example_runs(tmp_path):
    """The example runs on the synthetic catchment and gives sensible results."""
    mod = _load_example()
    scene = make_hydro_scene(str(tmp_path / "in"))
    out = mod.run_workflow(
        scene["dem"],
        scene["gauges"],
        str(tmp_path / "out"),
        out_epsg=OUT_EPSG,
        out_res=10.0,
        rivers_vec=scene["rivers"],
        landcover_img=scene["landcover"],
        lc_names=LC_NAMES,
        stream_threshold=20000.0,  # 2 ha = 200 pixels of 10 m
        snap_dist=40.0,
        use_pbar=False,
    )
    with rasterio.open(out["cond_dem"]) as src:
        assert src.crs.to_epsg() == OUT_EPSG
        assert src.res == (10.0, 10.0)

    # The embankment and the pit are found and removed.
    assert out["n_sinks"] >= 2
    assert out["changes"]["fill"] > out["changes"]["breach"] > 0.0
    assert out["n_flat"] > 0  # the filled depressions are flat

    # Two tributaries join one main channel (Strahler order 2).
    streams = gpd.read_file(out["streams_vec"])
    assert len(streams) == 3
    assert streams["order"].max() == 2

    # Nested gauged watersheds: the downstream one is incremental.
    wsheds = gpd.read_file(out["wsheds_vec"]).set_index("watershed_id")
    assert sorted(wsheds.index) == [1, 2]
    assert wsheds.loc[1, "area_m2"] > wsheds.loc[2, "area_m2"]
    ws_stats = out["ws_stats"].set_index("watershed_id")
    props = ws_stats[["prop_forest", "prop_grassland", "prop_urban"]].sum(axis=1)
    assert props.round(6).eq(1.0).all()
    assert ws_stats.loc[1, "prop_urban"] > 0.0
    assert ws_stats["slope_mean"].between(1.0, 45.0).all()

    # Inundated area grows with the stage height.
    areas = [out["flood_area_ha"][stage] for stage in sorted(out["flood_area_ha"])]
    assert areas[0] > 0.0
    assert all(b > a for a, b in zip(areas, areas[1:]))

    with rasterio.open(out["twi_img"]) as src:
        assert src.nodata is not None
    assert os.path.getsize(out["map_png"]) > 0
