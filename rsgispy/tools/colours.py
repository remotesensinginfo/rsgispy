"""
The tools.colours module provides utilities for converting between colour
representations such as hexadecimal strings and RGB tuples.
"""

from typing import Tuple, Union

import rsgispy

__all__ = [
    "hex_to_rgb",
    "rgb_to_hex",
]


def hex_to_rgb(hex_str: str) -> Tuple[int, int, int]:
    """
    Convert a hexadecimal colour string (``#RRGGBB`` or ``RRGGBB``) to an
    ``(R, G, B)`` tuple in ``0..255``.

    :param hex_str: 6-character or 7-character (leading ``#``) hex colour string.
    :return: ``(R, G, B)`` integer tuple.

    .. code:: python

        import rsgispy.tools.colours
        rgb = rsgispy.tools.colours.hex_to_rgb("#ff8800")
    """
    if not isinstance(hex_str, str):
        raise rsgispy.RSGISPyException(
            f"Hex colour must be a string, got {type(hex_str).__name__}."
        )
    trimmed = hex_str.strip()
    cleaned = trimmed[1:] if trimmed.startswith("#") else trimmed
    if len(cleaned) != 6:
        raise rsgispy.RSGISPyException(
            "String must be of length 6 or 7 if starting with #"
        )
    try:
        return (
            int(cleaned[0:2], 16),
            int(cleaned[2:4], 16),
            int(cleaned[4:6], 16),
        )
    except ValueError as err:
        raise rsgispy.RSGISPyException(
            f"Invalid hexadecimal colour string: {hex_str!r}."
        ) from err


def rgb_to_hex(
    r: Union[int, float],
    g: Union[int, float],
    b: Union[int, float],
    normalised: bool = False,
) -> str:
    """
    Convert ``(R, G, B)`` colour components to a ``#rrggbb`` hex string.

    :param r: Red value (``0..255``, or ``0..1`` when ``normalised=True``).
    :param g: Green value (``0..255``, or ``0..1`` when ``normalised=True``).
    :param b: Blue value (``0..255``, or ``0..1`` when ``normalised=True``).
    :param normalised: Whether inputs are normalised in ``0..1``.
    :return: Hex colour string ``#rrggbb``.

    .. code:: python

        import rsgispy.tools.colours
        hex_str = rsgispy.tools.colours.rgb_to_hex(255, 136, 0)
    """
    if normalised:
        r = int(r * 255)
        g = int(g * 255)
        b = int(b * 255)

    for label, val in (("Red", r), ("Green", g), ("Blue", b)):
        if not 0 <= val <= 255:
            raise rsgispy.RSGISPyException(
                f"{label} value must be between 0-255 or 0-1 if normalised"
            )

    return f"#{int(r):02x}{int(g):02x}{int(b):02x}"
