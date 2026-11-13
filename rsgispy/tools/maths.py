"""
The tools.maths module contains utility functions for mathematical, numeric,
and combinatorial operations.
"""

import collections.abc
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union

import numpy

import rsgispy

__all__ = [
    "round_to_nearest_val",
    "is_number",
    "similar_numeric_vals",
    "in_bounds",
    "mixed_signs",
    "negative",
    "is_odd",
    "powerset_iter",
    "powerset_lst",
    "create_var_list",
]


def round_to_nearest_val(
    input_val: Union[numpy.ndarray, int, float, List[int], List[float]],
    round_val: Union[int, float] = 1,
    round_direction: int = rsgispy.ROUND_NEAREST,
) -> Union[numpy.ndarray, float]:
    """
    Round elements of a number or NumPy array to the nearest multiple of
    ``round_val``, with options for rounding up, down, or to the nearest value.

    For example, if ``round_val=5``, numbers are rounded to intervals of 5.

    :param input_val: Scalar, list, or NumPy array of numbers.
    :param round_val: Interval value to round to (Default = ``1``).
    :param round_direction: Rounding mode (:data:`rsgispy.ROUND_NEAREST`,
                            :data:`rsgispy.ROUND_UP`, or
                            :data:`rsgispy.ROUND_DOWN`; Default = ``ROUND_NEAREST``).
    :return: Rounded scalar or NumPy array.
    """
    if round_val == 0:
        raise rsgispy.RSGISPyException("round_val must be non-zero.")

    if isinstance(input_val, list):
        input_val = numpy.array(input_val, dtype=float)

    round_val_flt = float(round_val)
    if round_direction == rsgispy.ROUND_NEAREST:
        return numpy.round(input_val / round_val_flt) * round_val
    if round_direction == rsgispy.ROUND_UP:
        return numpy.ceil(input_val / round_val_flt) * round_val
    if round_direction == rsgispy.ROUND_DOWN:
        return numpy.floor(input_val / round_val_flt) * round_val
    raise rsgispy.RSGISPyException(
        f"Invalid round_direction: '{round_direction}'. Please use "
        "rsgispy.ROUND_NEAREST, rsgispy.ROUND_UP, or rsgispy.ROUND_DOWN."
    )


def is_number(str_val: Union[str, float, int]) -> bool:
    """
    Test whether the input value represents a number.

    :param str_val: Input string or numeric value to test.
    :return: True if convertible to float or complex, False otherwise.
    """
    if isinstance(str_val, bool):
        return False
    try:
        float(str_val)  # type: ignore[arg-type]
    except (ValueError, TypeError):
        try:
            complex(str_val)  # type: ignore[arg-type]
        except (ValueError, TypeError):
            return False
    return True


def similar_numeric_vals(
    val_a: Union[numpy.ndarray, Sequence[Union[float, int]], float, int],
    val_b: Union[numpy.ndarray, Sequence[Union[float, int]], float, int],
    eql_thres: Union[float, int] = 0.1,
    abs_diff: bool = True,
) -> bool:
    """
    Test whether two numbers or arrays have similar values within a threshold.

    Calculates ``val_a - val_b`` (or ``abs(val_a - val_b)`` when ``abs_diff``
    is True) and checks if all differences are strictly less than ``eql_thres``.

    :param val_a: First scalar, sequence, or NumPy array.
    :param val_b: Second scalar, sequence, or NumPy array.
    :param eql_thres: Threshold below which values are considered similar.
    :param abs_diff: Use absolute difference between ``val_a`` and ``val_b``.
    :return: True if values are within ``eql_thres``, False otherwise.
    """
    if isinstance(val_a, collections.abc.Sequence):
        val_a = numpy.array(val_a)
    if isinstance(val_b, collections.abc.Sequence):
        val_b = numpy.array(val_b)

    diff = numpy.abs(val_a - val_b) if abs_diff else (val_a - val_b)
    similar = diff < eql_thres

    if isinstance(similar, numpy.ndarray):
        return bool(similar.all())
    return bool(similar)


def in_bounds(
    x: Union[numpy.ndarray, Sequence[float], float],
    lower: float,
    upper: float,
    upper_strict: bool = False,
) -> bool:
    """
    Check whether a value or array of values falls within ``[lower, upper]``
    (or ``[lower, upper)`` when ``upper_strict`` is True).

    :param x: Scalar, sequence, or NumPy array of values.
    :param lower: Lower inclusive bound.
    :param upper: Upper bound.
    :param upper_strict: Use strict ``< upper`` instead of ``<= upper``.
    :return: True if all values lie within bounds.
    """
    if upper_strict:
        return bool(lower <= numpy.min(x) and numpy.max(x) < upper)
    return bool(lower <= numpy.min(x) and numpy.max(x) <= upper)


def mixed_signs(x: Union[numpy.ndarray, Sequence[float]]) -> bool:
    """
    Check whether an array or sequence contains both negative and non-negative values.

    :param x: Sequence or NumPy array of numeric values.
    :return: True if minimum < 0 and maximum >= 0.
    """
    min_val = numpy.min(x)
    max_val = numpy.max(x)
    return bool(min_val < 0 <= max_val)


def negative(x: Union[numpy.ndarray, Sequence[float]]) -> bool:
    """
    Check whether all values in ``x`` are strictly negative.

    :param x: Sequence or NumPy array of numeric values.
    :return: True if the maximum value is < 0.
    """
    return bool(numpy.max(x) < 0)


def is_odd(number: float) -> bool:
    """
    Test whether a number is odd.

    :param number: Number to test.
    :return: True if ``number % 2 != 0``, False otherwise.
    """
    return (number % 2) != 0


def powerset_iter(in_set: List[Any]) -> Iterator[List[Any]]:
    """
    Return an iterator yielding all subsets of ``in_set`` (the powerset).

    :param in_set: Input list of items.
    :return: Iterator over subset lists.
    """
    if len(in_set) <= 1:
        yield in_set
        yield []
    else:
        for item in powerset_iter(in_set[1:]):
            yield [in_set[0]] + item
            yield item


def powerset_lst(in_set: List[Any], min_items: int = 0) -> List[List[Any]]:
    """
    Return a list of all subsets of ``in_set`` (the powerset).

    :param in_set: Input list of items.
    :param min_items: Minimum number of items required in each returned subset.
                      Ignored if <= 0 (Default = 0).
    :return: List of subset lists.
    """
    out_pset: List[List[Any]] = []
    for subset in powerset_iter(in_set):
        if min_items <= 0 or len(subset) >= min_items:
            out_pset.append(subset)
    return out_pset


def create_var_list(
    in_vals_lsts: Optional[Dict[str, Sequence[Any]]],
    val_dict: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Produce a list of dictionaries containing all Cartesian combinations of
    parameter lists given in ``in_vals_lsts``.

    :param in_vals_lsts: Dictionary mapping parameter names to lists of values.
    :param val_dict: Internal accumulator dictionary used during recursion.
    :return: List of dictionaries with one single value per key.
    """
    out_vars: List[Dict[str, Any]] = []
    if (in_vals_lsts is None) and (val_dict is not None):
        out_vars.append(dict(val_dict))
    elif in_vals_lsts is not None and len(in_vals_lsts) > 0:
        keys = list(in_vals_lsts.keys())
        key = keys[0]
        vals_arr = in_vals_lsts[key]
        next_vals_lsts: Optional[Dict[str, Sequence[Any]]] = {
            ckey: in_vals_lsts[ckey] for ckey in keys[1:]
        }
        if len(next_vals_lsts) == 0:
            next_vals_lsts = None

        base_dict = dict(val_dict) if val_dict is not None else {}
        for val in vals_arr:
            c_val_dict = dict(base_dict)
            c_val_dict[key] = val
            out_vars.extend(create_var_list(next_vals_lsts, c_val_dict))
    return out_vars
