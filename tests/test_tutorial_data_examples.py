"""
Run the guide examples on the rsgispy tutorial data
(``docs/examples/*_tutorial.py``), which made the figures and printed
outputs shown in the guides.

The tutorial data (about 1 GB, stored with Git LFS) is not part of this
repository, so these tests are skipped unless the ``RSGISPY_TUTORIAL_DATA``
environment variable gives the ``data`` folder of a clone of
https://github.com/remotesensinginfo/rsgispy-tutorials:

.. code-block:: bash

   RSGISPY_TUTORIAL_DATA=~/rsgispy-tutorials/data pytest tests/test_tutorial_data_examples.py
"""

import glob
import importlib.util
import os
import sys

import pytest

pytestmark = [pytest.mark.slow, pytest.mark.tutorial_data]

EXAMPLES_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, "docs", "examples")
)
TUTORIAL_SCRIPTS = sorted(glob.glob(os.path.join(EXAMPLES_DIR, "*_tutorial.py")))
DATA_DIR = os.environ.get("RSGISPY_TUTORIAL_DATA")


def _load(path: str):
    # The scripts import the guide example next to them (e.g. ``import
    # hydrology_workflow``), as they do when run with ``python``.
    if EXAMPLES_DIR not in sys.path:
        sys.path.insert(0, EXAMPLES_DIR)
    name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_tutorial_scripts_define_run():
    """Every tutorial data script has a ``run(data_dir, out_dir)`` function."""
    for path in TUTORIAL_SCRIPTS:
        with open(path, encoding="utf-8") as f:
            assert "\ndef run(" in f.read(), path


@pytest.mark.skipif(
    not DATA_DIR or not os.path.isdir(DATA_DIR),
    reason="set RSGISPY_TUTORIAL_DATA to the rsgispy-tutorials data folder",
)
@pytest.mark.parametrize(
    "path", TUTORIAL_SCRIPTS, ids=[os.path.basename(p) for p in TUTORIAL_SCRIPTS]
)
def test_tutorial_data_example_runs(path, tmp_path):
    """The guide example runs on the tutorial data."""
    mod = _load(path)
    out = mod.run(DATA_DIR, str(tmp_path), use_pbar=False)
    assert isinstance(out, dict) and out, "run() must return a non-empty dict"
    assert all(isinstance(key, str) for key in out)
    # Keys the tutorial scripts add to the guide workflow's outputs.
    script = os.path.basename(path)
    for key in _TUTORIAL_KEYS.get(script, ()):
        assert key in out, f"{script}: missing output {key!r}"
    # Every output path written into out_dir exists and is not empty, and at
    # least one output was written there.
    out_paths = [p for p in _iter_str_values(out) if _under(p, str(tmp_path))]
    assert out_paths, f"{script}: no outputs written to out_dir"
    for out_path in out_paths:
        assert os.path.exists(out_path), f"{script}: {out_path} not written"
        if os.path.isfile(out_path):
            assert os.path.getsize(out_path) > 0, f"{script}: {out_path} is empty"


# Outputs each tutorial script's run() adds itself (the rest come from the
# guide workflow it calls and are checked by that workflow's own test).
_TUTORIAL_KEYS = {
    "change_detection_tutorial.py": ("ls_imgs", "lc_img", "maps_png"),
    "making_maps_tutorial.py": ("s2_img", "cls_img", "ndvi_img", "dem_img"),
    "pixel_classification_workflow_tutorial.py": ("fig_file",),
    "tile_graph_tutorial.py": ("fig_file",),
}


def _iter_str_values(obj):
    """Yield the string values in a (nested) dict / list / tuple."""
    if isinstance(obj, (str, os.PathLike)):
        yield os.fspath(obj)
    elif isinstance(obj, dict):
        for val in obj.values():
            yield from _iter_str_values(val)
    elif isinstance(obj, (list, tuple)):
        for val in obj:
            yield from _iter_str_values(val)


def _under(path: str, root: str) -> bool:
    try:
        return os.path.commonpath([os.path.abspath(path), os.path.abspath(root)]) == (
            os.path.abspath(root)
        )
    except ValueError:
        return False
