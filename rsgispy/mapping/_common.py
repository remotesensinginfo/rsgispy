"""
Internal shared placement and validation helpers for :mod:`rsgispy.mapping`.
"""

from typing import Union

from matplotlib.offsetbox import AnchoredOffsetbox

import rsgispy

_LOC_ALIASES = {
    "top right": "upper right",
    "top left": "upper left",
    "bottom left": "lower left",
    "bottom right": "lower right",
    "top center": "upper center",
    "top centre": "upper center",
    "upper centre": "upper center",
    "bottom center": "lower center",
    "bottom centre": "lower center",
    "lower centre": "lower center",
    "center": "center",
    "centre": "center",
    "left": "center left",
    "centre left": "center left",
    "centre right": "center right",
}

_CODE_TO_LOC = {code: name for name, code in AnchoredOffsetbox.codes.items()}


def resolve_mpl_location(loc: Union[str, int]) -> str:
    """Validate and normalise a Matplotlib legend-style placement string or int code."""
    if isinstance(loc, int) and not isinstance(loc, bool):
        if loc in _CODE_TO_LOC:
            return _CODE_TO_LOC[loc]
        valid_codes = ", ".join(str(c) for c in sorted(_CODE_TO_LOC))
        raise rsgispy.RSGISPyException(
            f"Invalid placement location code '{loc}'. Valid codes are: {valid_codes}."
        )
    if not isinstance(loc, str):
        raise rsgispy.RSGISPyException(
            f"Location must be a string or integer code, got {type(loc).__name__}."
        )
    norm_loc = " ".join(loc.strip().lower().split())
    norm_loc = _LOC_ALIASES.get(norm_loc, norm_loc)
    if norm_loc not in AnchoredOffsetbox.codes:
        valid = ", ".join(sorted(AnchoredOffsetbox.codes))
        raise rsgispy.RSGISPyException(
            f"Invalid placement location '{loc}'. Valid options are: {valid}."
        )
    return norm_loc
