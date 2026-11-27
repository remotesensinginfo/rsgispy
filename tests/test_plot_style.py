"""
Tests for the rsgispy house plot style (:mod:`rsgispy.tools.plotting`):
the style context restores ``rcParams``, importing rsgispy does not change
them, and plotting functions apply the style only while they run.
"""

import subprocess
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

import rsgispy
import rsgispy.tools.plotting as rplot

matplotlib.use("Agg")


def _rc_snapshot():
    return {key: repr(val) for key, val in matplotlib.rcParams.items()}


def test_style_context_restores_rcparams():
    """rsgispy_plot_style applies the house style and restores rcParams afterwards."""
    before = _rc_snapshot()
    with rplot.rsgispy_plot_style():
        assert matplotlib.rcParams["axes.spines.top"] is False
        assert matplotlib.rcParams["figure.constrained_layout.use"] is True
        assert matplotlib.rcParams["image.cmap"] == rplot.SEQUENTIAL_CMAP
        colours = [c["color"] for c in matplotlib.rcParams["axes.prop_cycle"]]
        assert colours == rplot.CATEGORICAL_COLOURS
    assert _rc_snapshot() == before

    with rplot.rsgispy_plot_style("map", rc={"font.size": 7}):
        assert matplotlib.rcParams["axes.spines.top"] is True
        assert matplotlib.rcParams["axes.grid"] is False
        assert matplotlib.rcParams["font.size"] == 7
    assert _rc_snapshot() == before

    # Restored even when the body raises.
    with pytest.raises(ValueError):
        with rplot.rsgispy_plot_style():
            raise ValueError("boom")
    assert _rc_snapshot() == before

    with pytest.raises(rsgispy.RSGISPyException):
        rplot.get_plot_style_rc("poster")


def test_use_plot_style_decorator():
    """The decorator applies the style while the function runs, and only then."""
    before = _rc_snapshot()
    seen = {}

    @rplot.use_plot_style
    def _chart():
        seen["chart_top"] = matplotlib.rcParams["axes.spines.top"]
        return 1

    @rplot.use_plot_style(plot_type="map")
    def _map():
        seen["map_grid"] = matplotlib.rcParams["axes.grid"]
        return 2

    assert _chart() == 1 and _map() == 2
    assert seen == {"chart_top": False, "map_grid": False}
    assert _chart.__name__ == "_chart"
    assert _rc_snapshot() == before
    with pytest.raises(rsgispy.RSGISPyException):
        rplot.use_plot_style(plot_type="poster")


def test_plotting_function_applies_style_without_leaking():
    """A figure made by an rsgispy plot has the house style; rcParams are unchanged."""
    before = _rc_snapshot()
    fig, ax = rplot.plot_histogram_threshold(np.random.default_rng(0).normal(size=200), 0.5)
    assert not ax.spines["top"].get_visible()
    assert fig.get_layout_engine() is not None
    assert _rc_snapshot() == before
    plt.close(fig)


def test_import_does_not_change_rcparams():
    """Importing rsgispy and its plotting modules leaves rcParams unchanged."""
    code = (
        "import matplotlib\n"
        "before = {k: repr(v) for k, v in matplotlib.rcParams.items()}\n"
        "import rsgispy, rsgispy.tools.plotting, rsgispy.mapping, rsgispy.tools.stats\n"
        "import rsgispy.regression.regmetrics, rsgispy.imagespectral.endmembers\n"
        "after = {k: repr(v) for k, v in matplotlib.rcParams.items()}\n"
        "changed = sorted(k for k in before if before[k] != after[k])\n"
        "print(changed)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True,
        env={**__import__("os").environ, "MPLBACKEND": "Agg"},
    )
    assert out.stdout.strip() == "[]", out.stdout + out.stderr
