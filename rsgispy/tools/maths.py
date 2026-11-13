"""
The tools.maths module contains utility functions for mathematical operations.
"""

from typing import List, Union

import numpy

import rsgispy

__all__ = [
    "round_to_nearest_val",
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
