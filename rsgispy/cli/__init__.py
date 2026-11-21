"""
Command-line interface tools for ``rsgispy``.

All CLI tools share standardised argument naming conventions (``-i/--input``,
``-o/--output``, ``-l/--layer``, ``-f/--format``, ``-j/--n-cores``,
``--no-pbar``, ``--json``, ``-V/--version``) and expose both a parser builder
and ``main(argv=None) -> int`` for programmatic use and testing.
"""

import importlib
from typing import Any

__all__ = [
    "batchconvert2cog",
    "buildimglut",
    "calcimgstats",
    "chkgdalfile",
    "copybandnames",
    "filehash",
    "imginfo",
    "proj",
    "userpassfile",
    "vectools",
    "wgs84imgres",
]


def __getattr__(name: str) -> Any:
    """Lazily load CLI submodules to avoid ``python -m`` import warnings."""
    if name in __all__:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
