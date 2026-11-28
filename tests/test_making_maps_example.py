"""
Run the documented map-making worked example (``docs/examples/making_maps.py``)
end-to-end on synthetic data so the code shown in
``docs/guides/making_maps.rst`` stays correct. No network access is used: the
contextily basemap is replaced by a stub module and the WMTS download by an
autospec mock (which still checks the call's arguments against the real
function signature).
"""

import importlib.util
import os
import sys
import types
from unittest.mock import MagicMock, patch

import matplotlib

matplotlib.use("Agg")

# pylint: disable=wrong-import-position,wrong-import-order
import pytest  # noqa: E402

import rsgispy.mapping.basemaps  # noqa: E402
from maps_scene import CLASS_NAMES, make_maps_scene  # noqa: E402

pytestmark = pytest.mark.slow

EXAMPLE = os.path.join(
    os.path.dirname(__file__), os.pardir, "docs", "examples", "making_maps.py"
)


def _load_example():
    spec = importlib.util.spec_from_file_location("making_maps", EXAMPLE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_worked_example_runs(tmp_path):
    """The example runs end to end and writes every figure."""
    mod = _load_example()
    scene = make_maps_scene(str(tmp_path / "data"), height=40, width=60)

    fake_cx = types.ModuleType("contextily")
    fake_cx.add_basemap = MagicMock()
    fake_cx.providers = MagicMock()
    with patch.dict(sys.modules, {"contextily": fake_cx}), patch.object(
        rsgispy.mapping.basemaps, "create_wmts_img_map", autospec=True
    ) as fake_wmts:
        out = mod.run_workflow(
            scene["s2"],
            scene["cls"],
            scene["score"],
            scene["regions"],
            scene["rivers"],
            scene["overview"],
            str(tmp_path / "maps"),
            CLASS_NAMES,
            dpi=50,
            basemap=True,
            wmts_url="https://example.com/wmts/WMTSCapabilities.xml",
            wmts_lyr="mosaic",
            dem_img=scene["dem"],
        )
    fake_cx.add_basemap.assert_called_once()
    assert fake_cx.add_basemap.call_args.kwargs["crs"] == "epsg:32630"
    fake_wmts.assert_called_once()

    assert out["fig_height"] == pytest.approx(6.7 * 40 / 60)
    for key in ("rgb_stretches", "rgb_map", "score_map", "relief_map", "cls_map", "cls_legend",
                "vec_map", "choropleth_map", "final_map", "final_map_pdf",
                "zoom_map", "panels_map", "basemap_map", "wmts_map"):
        assert os.path.getsize(out[key]) > 0, key
    regions = out["regions"]
    assert regions["forest_pct"].between(0, 100).all()
    assert regions["total_pxls"].sum() > 0
