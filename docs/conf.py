"""
Configuration file for the Sphinx documentation builder for rsgispy.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

import rsgispy  # noqa: E402

# -- Project information -----------------------------------------------------
project = "rsgispy"
copyright = "2026, Pete Bunting"  # pylint: disable=redefined-builtin
author = "Pete Bunting"
version = rsgispy.__version__
release = rsgispy.__version__

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.extlinks",
]

# Links to the rsgispy tutorials (Jupyter notebooks with real data):
#   :tutorial:`Hydrological analysis <08_hydrology/01_hydrological_analysis.ipynb>`
#   :tutorial-dir:`section 5 <05_regression>` (a folder)
#   :tutorial-data:`dem_10m_osgb.tif`
TUTORIALS_URL = "https://github.com/remotesensinginfo/rsgispy-tutorials"
extlinks = {
    "tutorial": (f"{TUTORIALS_URL}/blob/main/%s", "%s"),
    "tutorial-dir": (f"{TUTORIALS_URL}/tree/main/%s", "%s"),
    "tutorial-data": (f"{TUTORIALS_URL}/blob/main/data/%s", "%s"),
}

templates_path = []
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autodoc_member_order = "bysource"
autodoc_typehints = "description"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "rasterio": ("https://rasterio.readthedocs.io/en/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# -- Options for HTML output -------------------------------------------------
try:
    import sphinx_rtd_theme  # noqa: F401  # pylint: disable=unused-import

    html_theme = "sphinx_rtd_theme"
except ImportError:
    html_theme = "alabaster"

html_static_path = []
