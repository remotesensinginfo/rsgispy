"""
The tools.dicts module provides utilities for querying and validating values
within nested dictionary structures (such as parsed JSON responses and
configurations).
"""

import collections.abc
import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import rsgispy
import rsgispy.tools.maths

__all__ = [
    "dict_struct_does_path_exist",
    "dict_struct_get_str_value",
    "dict_struct_get_boolean_value",
    "dict_struct_get_date_value",
    "dict_struct_get_datetime_value",
    "dict_struct_get_str_list_value",
    "dict_struct_get_numeric_value",
    "dict_struct_get_list_value",
]


def _dict_struct_traverse(
    dict_struct_obj: Dict[str, Any],
    tree_sequence: Sequence[str],
) -> Tuple[Any, str]:
    """Traverse nested dicts along ``tree_sequence``, raising if a key is missing."""
    curr_obj: Any = dict_struct_obj
    steps_str = ""
    for tree_step in tree_sequence:
        steps_str = f"{steps_str}:{tree_step}"
        if isinstance(curr_obj, collections.abc.Mapping) and tree_step in curr_obj:
            curr_obj = curr_obj[tree_step]
        else:
            raise rsgispy.RSGISPyException(f"Could not find '{steps_str}'")
    return curr_obj, steps_str


def dict_struct_does_path_exist(
    dict_struct_obj: Dict[str, Any],
    tree_sequence: Sequence[str],
) -> bool:
    """
    Test whether a key path exists within a nested dictionary structure.

    :param dict_struct_obj: Nested dictionary (e.g., parsed JSON).
    :param tree_sequence: Sequence of keys specifying the path.
    :return: True if the full path exists, False otherwise.
    """
    curr_obj: Any = dict_struct_obj
    for tree_step in tree_sequence:
        if isinstance(curr_obj, collections.abc.Mapping) and tree_step in curr_obj:
            curr_obj = curr_obj[tree_step]
        else:
            return False
    return True


def dict_struct_get_str_value(
    dict_struct_obj: Dict[str, Any],
    tree_sequence: Sequence[str],
    valid_values: Optional[Sequence[str]] = None,
) -> str:
    """
    Retrieve a string value from a nested dictionary structure.

    :param dict_struct_obj: Nested dictionary (e.g., parsed JSON).
    :param tree_sequence: Sequence of keys specifying the path.
    :param valid_values: Optional allowed values; raises if the retrieved value
                         is not among them.
    :return: Retrieved string value.
    """
    curr_obj, _ = _dict_struct_traverse(dict_struct_obj, tree_sequence)
    if valid_values is not None and curr_obj not in valid_values:
        raise rsgispy.RSGISPyException(
            f"'{curr_obj}' is not within the list of valid values."
        )
    return curr_obj


def dict_struct_get_boolean_value(
    dict_struct_obj: Dict[str, Any],
    tree_sequence: Sequence[str],
) -> bool:
    """
    Retrieve a boolean value from a nested dictionary structure.

    :param dict_struct_obj: Nested dictionary (e.g., parsed JSON).
    :param tree_sequence: Sequence of keys specifying the path.
    :return: Retrieved boolean value.
    """
    curr_obj, _ = _dict_struct_traverse(dict_struct_obj, tree_sequence)
    if isinstance(curr_obj, bool):
        return curr_obj
    raise rsgispy.RSGISPyException(f"'{curr_obj}' is not 'True' or 'False'.")


def dict_struct_get_date_value(
    dict_struct_obj: Dict[str, Any],
    tree_sequence: Sequence[str],
    date_format: Union[str, List[str]] = "%Y-%m-%d",
) -> datetime.date:
    """
    Retrieve and parse a ``datetime.date`` value from a nested dictionary.

    :param dict_struct_obj: Nested dictionary (e.g., parsed JSON).
    :param tree_sequence: Sequence of keys specifying the path.
    :param date_format: Format string or list of format strings passed to
                        ``datetime.datetime.strptime``.
    :return: Parsed ``datetime.date`` object.
    """
    curr_obj, _ = _dict_struct_traverse(dict_struct_obj, tree_sequence)

    if isinstance(date_format, str):
        try:
            return datetime.datetime.strptime(curr_obj, date_format).date()
        except Exception as err:
            raise rsgispy.RSGISPyException(str(err)) from err

    if isinstance(date_format, list):
        last_err: Optional[Exception] = None
        for fmt_str in date_format:
            try:
                return datetime.datetime.strptime(curr_obj, fmt_str).date()
            except Exception as err:  # pylint: disable=broad-exception-caught
                last_err = err
        raise rsgispy.RSGISPyException(str(last_err)) from last_err

    raise rsgispy.RSGISPyException(
        "Do not know what the type is of date_format variable."
    )


def dict_struct_get_datetime_value(
    dict_struct_obj: Dict[str, Any],
    tree_sequence: Sequence[str],
    date_time_format: Union[str, List[str]] = "%Y-%m-%dT%H:%M:%S.%f",
) -> datetime.datetime:
    """
    Retrieve and parse a ``datetime.datetime`` value from a nested dictionary.

    :param dict_struct_obj: Nested dictionary (e.g., parsed JSON).
    :param tree_sequence: Sequence of keys specifying the path.
    :param date_time_format: Format string or list of format strings passed to
                             ``datetime.datetime.strptime``.
    :return: Parsed ``datetime.datetime`` object.
    """
    curr_obj, _ = _dict_struct_traverse(dict_struct_obj, tree_sequence)
    cleaned_val = str(curr_obj).replace("Z", "")

    if isinstance(date_time_format, str):
        try:
            return datetime.datetime.strptime(cleaned_val, date_time_format)
        except Exception as err:
            raise rsgispy.RSGISPyException(str(err)) from err

    if isinstance(date_time_format, list):
        last_err: Optional[Exception] = None
        for fmt_str in date_time_format:
            try:
                return datetime.datetime.strptime(cleaned_val, fmt_str)
            except Exception as err:  # pylint: disable=broad-exception-caught
                last_err = err
        raise rsgispy.RSGISPyException(str(last_err)) from last_err

    raise rsgispy.RSGISPyException(
        "Do not know what the type is of date_time_format variable."
    )


def dict_struct_get_str_list_value(
    dict_struct_obj: Dict[str, Any],
    tree_sequence: Sequence[str],
    valid_values: Optional[Sequence[str]] = None,
) -> List[str]:
    """
    Retrieve a list of string values from a nested dictionary structure.

    :param dict_struct_obj: Nested dictionary (e.g., parsed JSON).
    :param tree_sequence: Sequence of keys specifying the path.
    :param valid_values: Optional allowed string values; raises if any item
                         is not a string or not in ``valid_values``.
    :return: List of strings.
    """
    curr_obj, _ = _dict_struct_traverse(dict_struct_obj, tree_sequence)
    if not isinstance(curr_obj, list):
        raise rsgispy.RSGISPyException("Retrieved value is not a list.")
    if valid_values is not None:
        for val in curr_obj:
            if not isinstance(val, str):
                raise rsgispy.RSGISPyException(f"'{val}' is not of type string.")
            if val not in valid_values:
                raise rsgispy.RSGISPyException(
                    f"'{val}' is not within the list of valid values."
                )
    return curr_obj


def dict_struct_get_numeric_value(
    dict_struct_obj: Dict[str, Any],
    tree_sequence: Sequence[str],
    valid_lower: Optional[float] = None,
    valid_upper: Optional[float] = None,
) -> Union[int, float]:
    """
    Retrieve a numeric value from a nested dictionary structure.

    :param dict_struct_obj: Nested dictionary (e.g., parsed JSON).
    :param tree_sequence: Sequence of keys specifying the path.
    :param valid_lower: Optional minimum valid bound.
    :param valid_upper: Optional maximum valid bound.
    :return: Numeric ``int`` or ``float`` value.
    """
    curr_obj, steps_str = _dict_struct_traverse(dict_struct_obj, tree_sequence)
    if not rsgispy.tools.maths.is_number(curr_obj):
        raise rsgispy.RSGISPyException(
            f"The identified value is not numeric '{steps_str}'"
        )

    out_value: Union[int, float]
    if isinstance(curr_obj, int):
        out_value = int(curr_obj)
    else:
        out_value = float(curr_obj)

    if valid_lower is not None and out_value < valid_lower:
        raise rsgispy.RSGISPyException(
            f"'{out_value}' is less than the defined valid range."
        )
    if valid_upper is not None and out_value > valid_upper:
        raise rsgispy.RSGISPyException(
            f"'{out_value}' is higher than the defined valid range."
        )
    return out_value


def dict_struct_get_list_value(
    dict_struct_obj: Dict[str, Any],
    tree_sequence: Sequence[str],
) -> List[Any]:
    """
    Retrieve a list value from a nested dictionary structure.

    :param dict_struct_obj: Nested dictionary (e.g., parsed JSON).
    :param tree_sequence: Sequence of keys specifying the path.
    :return: Retrieved list.
    """
    curr_obj, _ = _dict_struct_traverse(dict_struct_obj, tree_sequence)
    if not isinstance(curr_obj, list):
        raise rsgispy.RSGISPyException("Retrieved value is not a list.")
    return curr_obj
