"""
Run the plot gallery script (``docs/examples/plot_gallery.py``), which draws
every rsgispy plotting function, and check that each image is written and no
figure is left open.
"""

import importlib.util
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

import pytest  # noqa: E402

pytestmark = pytest.mark.slow

EXAMPLE = os.path.join(
    os.path.dirname(__file__), os.pardir, "docs", "examples", "plot_gallery.py"
)
GUIDE = os.path.join(
    os.path.dirname(__file__), os.pardir, "docs", "guides", "plot_gallery.rst"
)


def _load_example():
    spec = importlib.util.spec_from_file_location("plot_gallery", EXAMPLE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_plot_gallery_runs(tmp_path):
    """Every gallery plot is written and closed, and each is shown in the guide."""
    plt.close("all")
    gallery = _load_example()
    out = gallery.run_gallery(str(tmp_path / "gallery"), scene_dir=str(tmp_path / "scene"), dpi=40)
    assert len(out) >= 15
    for path in out.values():
        assert os.path.isfile(path) and os.path.getsize(path) > 0
    assert not plt.get_fignums()

    with open(GUIDE, encoding="utf-8") as guide_fp:
        guide_txt = guide_fp.read()
    for name in out:
        assert f"images/gallery/{name}.png" in guide_txt, name
