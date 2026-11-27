"""
Shared pytest fixtures.

rsgispy plotting functions return their figures open (so they show in a
notebook), so a test or worked example that does not close a figure would
leave it open for the next test in the same process. Every figure still open
after a test is closed here, so tests that check that no figures are left
open only see their own.
"""

import sys

import pytest


@pytest.fixture(autouse=True)
def _close_open_figures():
    """Close all Matplotlib figures left open by a test."""
    yield
    plt_mod = sys.modules.get("matplotlib.pyplot")
    if plt_mod is not None:
        plt_mod.close("all")
