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
    assert isinstance(out, dict)
