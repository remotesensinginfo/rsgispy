"""
Run the documented large raster processing example
(``docs/examples/large_rasters.py``) end-to-end on synthetic data so the code
shown in ``docs/guides/large_rasters.rst`` stays correct.
"""

import importlib.util
import os

import geopandas as gpd
import numpy as np
import pytest
import rasterio

from seg_ml_scene import HEIGHT, WIDTH, make_seg_scene

EXAMPLE = os.path.join(
    os.path.dirname(__file__), os.pardir, "docs", "examples", "large_rasters.py"
)
GUIDE = os.path.join(
    os.path.dirname(__file__), os.pardir, "docs", "guides", "large_rasters.rst"
)
# Environment variables the example sets (restored after the test).
ENV_VARS = (
    "RSGISPY_BLOCK_SIZE",
    "RSGISLIB_BLOCK_SIZE",
    "RSGISPY_IMG_CRT_OPTS_GTIFF",
    "RSGISLIB_IMG_CRT_OPTS_GTIFF",
)


def _load_example():
    spec = importlib.util.spec_from_file_location("large_rasters", EXAMPLE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def restore_env():
    """Restore the block size / creation option variables after the test."""
    saved = {key: os.environ.get(key) for key in ENV_VARS}
    yield
    for key, val in saved.items():
        if val is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = val


def _read(path):
    with rasterio.open(path) as src:
        return src.read(1), src.nodata


def test_worked_example_runs(tmp_path, restore_env):
    # pylint: disable=redefined-outer-name,unused-argument
    mod = _load_example()
    scene = make_seg_scene(str(tmp_path / "scene"), seed=0, tag="a")
    out = mod.run_workflow(
        scene["s2"],
        str(tmp_path / "out"),
        red_band=3,
        nir_band=4,
        seg_bands=(1, 2, 3, 4),
        block_size=16,  # several blocks on the small test image
        tile_size=32,  # 3 x 2 tiles
        n_jobs=1,
        use_pbar=False,
    )
    assert out["full_mib"] == pytest.approx(HEIGHT * WIDTH * 4 * 2 / 2**20)
    assert out["block_mib"] == pytest.approx(16 * 16 * 4 * 8 / 2**20)

    # Block processing, the tile graph and the tiled mosaic all give the
    # same NDVI (the no data area is -9999 in ndvi.tif and 0 in the graph).
    ndvi, ndvi_nd = _read(out["ndvi_img"])
    assert ndvi_nd == -9999.0
    valid = ndvi != -9999.0
    assert (~valid).sum() == 25  # the 5 x 5 no data patch
    assert np.all((ndvi[valid] > -1.0) & (ndvi[valid] < 1.0))
    tg_ndvi, _ = _read(out["tg_ndvi_img"])
    np.testing.assert_allclose(tg_ndvi[valid], ndvi[valid], atol=1e-6)
    assert np.all(tg_ndvi[~valid] == 0)
    mosaic, _ = _read(out["ndvi_mosaic_vrt"])
    np.testing.assert_allclose(mosaic, ndvi, atol=1e-6)
    assert out["n_tiles"] == 6

    final, _ = _read(out["ndvi_final"])
    np.testing.assert_allclose(final, ndvi, atol=1e-6)
    with rasterio.open(out["ndvi_final"]) as src:
        assert src.profile.get("tiled")
        assert src.tags(ns="IMAGE_STRUCTURE").get("COMPRESSION") == "DEFLATE"
        assert src.tags(ns="IMAGE_STRUCTURE").get("PREDICTOR") == "3"
        assert "STATISTICS_MEAN" in src.tags(1)

    hp_img, _ = _read(out["tg_hp_img"])
    assert hp_img.shape == (HEIGHT, WIDTH)
    edges, _ = _read(out["ndvi_edges_img"])
    assert edges.shape == (HEIGHT, WIDTH)

    with rasterio.open(out["segs_img"]) as src:
        labels = src.read(1)
    assert labels.max() == len(out["seg_stats"])
    assert {"seg_id", "pxl_count"} <= set(out["seg_stats"].columns)
    segs = gpd.read_parquet(out["segs_vec"])
    assert len(segs) == len(out["seg_stats"])
    assert segs["seg_id"].is_monotonic_increasing


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
    assert len(markers) >= 9
    for marker in markers:
        assert marker in script
        assert marker.replace("[step-", "[end-step-") in script
