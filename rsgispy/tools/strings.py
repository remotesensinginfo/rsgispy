"""
The tools.strings module provides string parsing, formatting, cleaning,
and unique identifier generation utilities.
"""

import math
import string
from typing import List, Optional
import uuid

import rsgispy

__all__ = [
    "uid_generator",
    "str_to_float",
    "str_to_int",
    "zero_pad_num_str",
    "remove_repeated_chars",
    "check_str",
]


def uid_generator(size: int = 6) -> str:
    """
    Generate a random string of the specified length based on UUID4.

    :param size: Length of the returned string.
    :return: Random hex string of length ``size``.
    """
    random_str = str(uuid.uuid4()).replace("-", "")
    return random_str[0:size]


def str_to_float(str_val: str, err_val: Optional[float] = None) -> float:
    """
    Convert a string to a float.

    If conversion fails and ``err_val`` is provided, ``float(err_val)`` is
    returned; otherwise :class:`rsgispy.RSGISPyException` is raised.

    :param str_val: String to convert to float.
    :param err_val: Fallback value returned on conversion failure.
    :return: Floating-point value.
    """
    cleaned = str(str_val).strip()
    try:
        return float(cleaned)
    except ValueError as err:
        if err_val is not None:
            return float(err_val)
        raise rsgispy.RSGISPyException(
            f"Could not convert string to float: '{cleaned}'."
        ) from err


def str_to_int(str_val: str, err_val: Optional[int] = None) -> int:
    """
    Convert a string to an int, rounding via float if needed.

    If conversion fails and ``err_val`` is provided, ``int(err_val)`` is
    returned; otherwise :class:`rsgispy.RSGISPyException` is raised.

    :param str_val: String to convert to int.
    :param err_val: Fallback value returned on conversion failure.
    :return: Integer value.
    """
    cleaned = str(str_val).strip()
    try:
        return int(cleaned)
    except ValueError:
        try:
            tmp_float = float(cleaned)
            return int(math.floor(tmp_float + 0.5))
        except ValueError as err:
            if err_val is not None:
                return int(err_val)
            raise rsgispy.RSGISPyException(
                f"Could not convert string to int: '{cleaned}'."
            ) from err


def zero_pad_num_str(
    num_val: float,
    str_len: int = 3,
    round_num: bool = False,
    round_n_digts: int = 0,
    integerise: bool = False,
    absolute: bool = False,
    gain: float = 1,
) -> str:
    """
    Zero-pad a number to produce a fixed-width string.

    :param num_val: Number value to format.
    :param str_len: Minimum character length of the output string.
    :param round_num: Whether to round ``num_val``.
    :param round_n_digts: Number of decimal digits when rounding.
    :param integerise: Whether to multiply by ``gain`` and cast to ``int``.
    :param absolute: Whether to take the absolute value first.
    :param gain: Multiplier applied before integerising.
    :return: Zero-padded string representation.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    if absolute:
        num_val = abs(num_val)
    if round_num:
        num_val = round(num_val, round_n_digts)
    if integerise:
        num_val = int(num_val * gain)

    return f"{num_val}".zfill(str_len)


def remove_repeated_chars(str_val: str, repeat_char: str) -> str:
    """
    Collapse consecutive occurrences of ``repeat_char`` in ``str_val`` to one.

    :param str_val: Input string.
    :param repeat_char: Single character to deduplicate.
    :return: String with repeated ``repeat_char`` collapsed.
    """
    if len(repeat_char) != 1:
        raise rsgispy.RSGISPyException(
            "The repeat character has multiple characters."
        )
    out_chars: List[str] = []
    prev_char = ""
    for char in str_val:
        if char == repeat_char:
            if char != prev_char:
                out_chars.append(char)
        else:
            out_chars.append(char)
        prev_char = char
    return "".join(out_chars)


def check_str(
    str_val: Optional[str],
    rm_non_ascii: bool = False,
    rm_dashs: bool = False,
    rm_spaces: bool = False,
    rm_punc: bool = False,
) -> str:
    """
    Clean a string by optionally stripping non-ASCII characters, converting
    dashes/spaces to underscores, and removing punctuation.

    :param str_val: Input string to clean (``None`` treated as empty string).
    :param rm_non_ascii: Remove non-ASCII characters when True.
    :param rm_dashs: Replace ``-`` with ``_`` and collapse repeats when True.
    :param rm_spaces: Replace spaces with ``_`` and collapse repeats when True.
    :param rm_punc: Remove punctuation other than ``_`` and ``-`` when True.
    :return: Cleaned string.
    """
    str_val_tmp = "" if str_val is None else str_val.strip()

    if rm_non_ascii:
        allowed = set(string.ascii_letters + string.punctuation + string.digits + " ")
        str_val_tmp = "".join(c for c in str_val_tmp if c in allowed)

    if rm_dashs:
        str_val_tmp = remove_repeated_chars(str_val_tmp.replace("-", "_"), "_")

    if rm_spaces:
        str_val_tmp = remove_repeated_chars(str_val_tmp.replace(" ", "_"), "_")

    if rm_punc:
        for punct in string.punctuation:
            if punct not in ("_", "-"):
                str_val_tmp = str_val_tmp.replace(punct, "")
        str_val_tmp = remove_repeated_chars(str_val_tmp, "_")

    return str_val_tmp
