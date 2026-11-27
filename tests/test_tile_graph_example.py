"""
Run the documented tile-graph worked example (``docs/examples/tile_graph.py``)
end-to-end on synthetic data so the code shown in
``docs/guides/tile_graph.rst`` stays correct. The example itself asserts that
the graph matches the file-based functions run one after another, that the
result does not depend on the block size and that ``n_jobs=2`` gives the same
result as one thread.
"""

import importlib.util
import os

import numpy as np
import pytest
import rasterio

import rsgispy
from rsgispy.imageutils.colourtable import get_thematic_classes
from tile_graph_scene import make_tile_graph_scene

pytest.importorskip("sklearn")

EXAMPLE = os.path.join(
    os.path.dirname(__file__), os.pardir, "docs", "examples", "tile_graph.py"
)
GUIDE = os.path.join(
    os.path.dirname(__file__), os.pardir, "docs", "guides", "tile_graph.rst"
)


def _load_example():
    spec = importlib.util.spec_from_file_location("tile_graph_example", EXAMPLE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    return make_tile_graph_scene(str(tmp_path_factory.mktemp("tg_scene")))


# 64 and 50 split the 120 x 150 scene into several tiles (50 does not divide
# the width); n_jobs=2 runs the tiles on two threads.
@pytest.mark.parametrize("block_size,n_jobs", [(64, 2), (50, 1)])
def test_worked_example_runs(scene, tmp_path, block_size, n_jobs):
    mod = _load_example()
    out = mod.run_workflow(
        scene["s2"], scene["dem"], scene["train"], str(tmp_path / "out"),
        block_size=block_size, n_jobs=n_jobs, use_pbar=False,
    )
    assert out["n_train"] > 100
    samples = out["samples"]
    assert list(samples.columns) == ["class_name", "out_cls_id", "internal_cls_id",
                                     "x", "y", "NDVI_med", "NDWI_med", "NDVI_sd",
                                     "NDWI_sd"]
    assert set(samples["out_cls_id"]) == {1, 2, 3, 4}

    with rasterio.open(out["cls_img"]) as src, \
            rasterio.open(scene["classes"]) as ref:
        cls = src.read(1)
        truth = ref.read(1)
        assert src.dtypes[0] == "uint8"
        assert src.nodata == 0
        assert src.colormap(1)[1][:3] == (31, 120, 180)
    assert get_thematic_classes(out["cls_img"]) == mod.CLASSES
    valid = cls > 0
    assert not valid[:, :2].any()  # the no data strip stays no data
    assert (cls[valid] == truth[valid]).mean() > 0.95

    with rasterio.open(out["feats_img"]) as src:
        assert src.count == 4
        assert src.descriptions == ("NDVI_med", "NDWI_med", "NDVI_sd", "NDWI_sd")
        assert src.nodata == -999.0
    with rasterio.open(out["hshade_img"]) as src:
        assert src.dtypes[0] == "uint8"
    for key in ("slope_img", "sharp_img", "dog_img", "dem_edges_img",
                "steep_veg_img", "indices_z_img"):
        assert os.path.exists(out[key]), key

    nodata = out["nodata"]
    np.testing.assert_array_equal(nodata["meta"], nodata["explicit"])
    assert (nodata["meta"] != nodata["ignored"]).any()  # 0 used as a value
    assert "halo=(4, 4)" in out["halo_graph"]  # 1 + 2 + 1 pixels at the source

    stats = out["stats"]
    assert list(stats["indices"].to_dataframe().index) == ["NDVI", "NDWI"]
    assert stats["indices"].hist.shape == (2, 20)
    per_class = stats["per_class"]
    assert set(per_class.index) == {1, 2, 3, 4}
    assert per_class.loc[2, "NDVI_mean"] > per_class.loc[1, "NDVI_mean"]

    with rasterio.open(out["indices_z_img"]) as src:
        assert src.descriptions == ("NDVI", "NDWI")
        for band in (1, 2):
            z_val = src.read(band)
            z_val = z_val[(z_val != src.nodata) & np.isfinite(z_val)]
            assert abs(float(z_val.mean())) < 1e-3
            assert float(z_val.std()) == pytest.approx(1.0, rel=1e-3)

    with rasterio.open(out["steep_veg_img"]) as src:
        assert set(np.unique(src.read(1))) <= {0, 1, 255}


def test_grid_mismatch_is_reported(scene, tmp_path):
    """Sources on different grids raise an error naming both rasters."""
    from rsgispy.imageprocess import MemorySink, PointwiseOp, RasterSource, TileGraph

    shifted = str(tmp_path / "dem_shifted.tif")
    with rasterio.open(scene["dem"]) as src:
        profile = src.profile
        data = src.read()
    trans = profile["transform"]  # shift by 3 pixels to the east
    profile["transform"] = rasterio.Affine(trans.a, trans.b, trans.c + 3 * trans.a,
                                           trans.d, trans.e, trans.f)
    with rasterio.open(shifted, "w", **profile) as dst:
        dst.write(data)
    node = PointwiseOp([RasterSource(scene["dem"]), RasterSource(shifted)],
                       lambda a, b: a - b)
    with pytest.raises(rsgispy.RSGISPyException, match="same pixel grid"):
        TileGraph(MemorySink(node))


def test_guide_markers_exist():
    """Every literalinclude marker used by the guide exists in the script."""
    with open(EXAMPLE, encoding="utf-8") as f_obj:
        script = f_obj.read()
    with open(GUIDE, encoding="utf-8") as f_obj:
        guide = f_obj.read()
    # Only the markers of the blocks included from the example script (the
    # guide also includes the tutorial-data script).
    markers = []
    included = ""
    for line in guide.splitlines():
        if "literalinclude::" in line:
            included = line.split("literalinclude::")[1].strip()
        elif ":start-after:" in line and included.endswith("/tile_graph.py"):
            markers.append(line.split(":start-after:")[1].strip())
    assert len(markers) >= 11
    for marker in markers:
        assert marker in script
        assert marker.replace("[step-", "[end-step-") in script
