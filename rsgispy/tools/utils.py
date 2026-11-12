"""
The tools.utils module contains general utility functions for strings, JSON,
files, dates, numeric checks, and base64 encoding.
"""

# pylint: disable=too-many-lines

import base64
import calendar
import collections.abc
import datetime
import gzip
import json
import math
import os
import string
import uuid
import xml.etree.ElementTree as ET
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy

import rsgispy

__all__ = [
    "RSGISNumpyArrayEncoder",
    "uid_generator",
    "is_number",
    "similar_numeric_vals",
    "str_to_float",
    "str_to_int",
    "zero_pad_num_str",
    "powerset_iter",
    "powerset_lst",
    "get_environment_variable",
    "num_process_cores",
    "read_text_file",
    "read_text_file_no_new_lines",
    "read_text_file_to_list",
    "write_list_to_file",
    "write_data_to_file",
    "write_dict_to_json",
    "dict_to_json_str",
    "write_dict_to_json_gz",
    "read_json_to_dict",
    "read_gz_json_to_dict",
    "create_var_list",
    "in_bounds",
    "mixed_signs",
    "negative",
    "is_odd",
    "hex_to_rgb",
    "rgb_to_hex",
    "remove_repeated_chars",
    "check_str",
    "create_ascii_text_file",
    "get_days_since",
    "get_days_since_date",
    "find_month_end_date",
    "create_year_month_start_end_lst",
    "create_year_month_n_months_lst",
    "is_summer_winter",
    "get_datetime_increment_lst",
    "dict_struct_does_path_exist",
    "dict_struct_get_str_value",
    "dict_struct_get_boolean_value",
    "dict_struct_get_date_value",
    "dict_struct_get_datetime_value",
    "dict_struct_get_str_list_value",
    "dict_struct_get_numeric_value",
    "dict_struct_get_list_value",
    "encode_base64_text",
    "decode_base64_text",
    "create_username_password_file",
    "get_username_password",
    "prettify_xml_file",
]


class RSGISNumpyArrayEncoder(json.JSONEncoder):
    """JSON encoder supporting NumPy scalar/array types and datetime objects."""

    def default(self, o: Any) -> Any:
        encoded: Any = None
        handled = True
        if isinstance(o, numpy.integer):
            encoded = int(o)
        elif isinstance(o, numpy.floating):
            encoded = float(o)
        elif isinstance(o, numpy.complexfloating):
            encoded = float(o.real) if o.imag == 0 else str(o)
        elif isinstance(o, numpy.ndarray):
            encoded = o.tolist()
        elif isinstance(o, numpy.bool_):
            encoded = bool(o)
        elif isinstance(o, numpy.bytes_):
            encoded = str(o)
        elif isinstance(o, numpy.void):
            encoded = None
        elif isinstance(o, (datetime.datetime, datetime.date)):
            encoded = o.isoformat()
        elif isinstance(o, datetime.timedelta):
            encoded = str(o)
        elif hasattr(o, "to_json"):
            encoded = o.to_json()
        else:
            handled = False

        if handled:
            return encoded
        return super().default(o)


def uid_generator(size: int = 6) -> str:
    """
    Generate a random string of the specified length based on UUID4.

    :param size: Length of the returned string.
    :return: Random hex string of length ``size``.
    """
    random_str = str(uuid.uuid4()).replace("-", "")
    return random_str[0:size]


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


def get_environment_variable(var: str) -> Optional[str]:
    """
    Get an environment variable value, or ``None`` if it is not set.

    :param var: Environment variable name.
    :return: Variable value string or ``None``.
    """
    return os.environ.get(var)


def num_process_cores() -> int:
    """
    Return the number of CPU cores available on the machine.

    :return: Integer CPU count (defaults to 1 if undetectable).
    """
    return os.cpu_count() or 1


def read_text_file(input_file: str) -> str:
    """
    Read a text file into a single string.

    :param input_file: Path to the input text file.
    :return: File contents string.
    """
    with open(input_file, "r", encoding="utf-8") as data_file:
        return data_file.read()


def read_text_file_no_new_lines(input_file: str) -> str:
    """
    Read a text file into a single string with line breaks and leading/trailing
    line whitespace stripped.

    :param input_file: Path to the input text file.
    :return: Concatenated file contents string.
    """
    with open(input_file, "r", encoding="utf-8") as data_file:
        return "".join(line.strip() for line in data_file)


def read_text_file_to_list(
    input_file: str,
    include_empty_lines: bool = False,
) -> List[str]:
    """
    Read a text file into a list of stripped lines.

    :param input_file: Path to the input text file.
    :param include_empty_lines: Include blank lines when True (Default = False).
    :return: List of line strings.
    """
    out_list: List[str] = []
    with open(input_file, "r", encoding="utf-8") as data_file:
        for line in data_file:
            stripped = line.strip()
            if include_empty_lines or stripped != "":
                out_list.append(stripped)
    return out_list


def write_list_to_file(data_lst: Sequence[Any], out_file: str) -> None:
    """
    Write a sequence of values to a text file, one item per line.

    :param data_lst: Sequence of items to write.
    :param out_file: Path to the output text file.
    """
    with open(out_file, "w", encoding="utf-8") as out_fp:
        for item in data_lst:
            out_fp.write(f"{item}\n")


def write_data_to_file(data_val: Any, out_file: str) -> None:
    """
    Write ``str(data_val)`` followed by a newline to an output text file.

    :param data_val: Value to write.
    :param out_file: Path to the output text file.
    """
    with open(out_file, "w", encoding="utf-8") as out_fp:
        out_fp.write(f"{data_val}\n")


def write_dict_to_json(
    data_dict: Union[Dict[str, Any], List[Any]],
    out_file: str,
) -> None:
    """
    Write a dictionary (or list) to a pretty-printed JSON file.

    :param data_dict: Data structure to serialise to JSON.
    :param out_file: Output JSON file path.
    """
    with open(out_file, "w", encoding="utf-8") as out_fp:
        json.dump(
            data_dict,
            out_fp,
            cls=RSGISNumpyArrayEncoder,
            sort_keys=True,
            indent=4,
            separators=(",", ": "),
            ensure_ascii=False,
        )


def dict_to_json_str(data_dict: Union[Dict[str, Any], List[Any]]) -> str:
    """
    Serialise a dictionary (or list) to a compact JSON string.

    :param data_dict: Data structure to serialise to JSON.
    :return: Compact JSON string.
    """
    return json.dumps(
        data_dict,
        cls=RSGISNumpyArrayEncoder,
        sort_keys=True,
        indent=None,
        separators=(",", ": "),
        ensure_ascii=False,
    )


def write_dict_to_json_gz(
    data_dict: Union[Dict[str, Any], List[Any]],
    out_file: str,
    encoding: str = "utf-8",
) -> None:
    """
    Write a dictionary (or list) to a gzip-compressed JSON file.

    :param data_dict: Data structure to serialise to JSON.
    :param out_file: Output ``.json.gz`` file path.
    :param encoding: Text encoding (Default = ``'utf-8'``).
    """
    json_str = json.dumps(
        data_dict,
        cls=RSGISNumpyArrayEncoder,
        sort_keys=True,
        indent=4,
        separators=(",", ": "),
        ensure_ascii=False,
    )
    json_bytes = json_str.encode(encoding)
    with gzip.GzipFile(out_file, "w") as fout:
        fout.write(json_bytes)


def read_json_to_dict(input_file: str) -> Any:
    """
    Read a JSON file and return the parsed Python object.

    :param input_file: Input JSON file path.
    :return: Parsed JSON dict or list.
    """
    with open(input_file, "r", encoding="utf-8") as in_fp:
        return json.load(in_fp)


def read_gz_json_to_dict(input_file: str, encoding: str = "utf-8") -> Any:
    """
    Read a gzip-compressed JSON file and return the parsed Python object.

    :param input_file: Input ``.json.gz`` file path.
    :param encoding: Text encoding (Default = ``'utf-8'``).
    :return: Parsed JSON dict or list.
    """
    with gzip.GzipFile(input_file, "r") as f_in:
        json_bytes = f_in.read()
    return json.loads(json_bytes.decode(encoding))


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


def hex_to_rgb(hex_str: str) -> Tuple[int, int, int]:
    """
    Convert a hexadecimal colour string (``#RRGGBB`` or ``RRGGBB``) to an
    ``(R, G, B)`` tuple in ``0..255``.

    :param hex_str: 6-character or 7-character (leading ``#``) hex colour string.
    :return: ``(R, G, B)`` integer tuple.
    """
    cleaned = hex_str[1:] if hex_str.startswith("#") else hex_str
    if len(cleaned) != 6:
        raise rsgispy.RSGISPyException(
            "String must be of length 6 or 7 if starting with #"
        )
    return (
        int(cleaned[0:2], 16),
        int(cleaned[2:4], 16),
        int(cleaned[4:6], 16),
    )


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


def create_ascii_text_file(input_file: str, output_file: str) -> None:
    """
    Copy ``input_file`` to ``output_file`` while removing non-ASCII characters.

    :param input_file: Input text file path.
    :param output_file: Output ASCII text file path.
    """
    txt_lst = read_text_file_to_list(input_file, True)
    out_txt_lst = [check_str(line, rm_non_ascii=True) for line in txt_lst]
    write_list_to_file(out_txt_lst, output_file)


def get_days_since(year: int, day_of_year: int, base_date: datetime.date) -> int:
    """
    Calculate the number of days from ``base_date`` to a given ``year`` and
    1-based ``day_of_year``.

    :param year: Target year (e.g., 2020).
    :param day_of_year: Day within the year (1..366).
    :param base_date: Reference ``datetime.date``.
    :return: Number of days elapsed since ``base_date``.
    """
    if year < base_date.year:
        raise rsgispy.RSGISPyException(
            "The year specified is before the base date."
        )
    date_val = datetime.date(year=int(year), month=1, day=1) + datetime.timedelta(
        days=int(day_of_year - 1)
    )
    return (date_val - base_date).days


def get_days_since_date(
    year: int,
    month: int,
    day: int,
    base_date: datetime.date,
) -> int:
    """
    Calculate the number of days from ``base_date`` to ``year-month-day``.

    :param year: Target year (e.g., 2020).
    :param month: Target month (1..12).
    :param day: Target day of month (1..31).
    :param base_date: Reference ``datetime.date``.
    :return: Number of days elapsed since ``base_date``.
    """
    if year < base_date.year:
        raise rsgispy.RSGISPyException(
            "The year specified is before the base date."
        )
    date_val = datetime.date(year=int(year), month=int(month), day=int(day))
    return (date_val - base_date).days


def find_month_end_date(year: int, month: int) -> int:
    """
    Return the last day number of ``month`` in ``year``.

    :param year: Year integer (e.g., 2019).
    :param month: Month integer (1..12).
    :return: Number of the last day of the month.
    """
    _, n_days = calendar.monthrange(year, month)
    return int(n_days)


def create_year_month_start_end_lst(
    start_year: int,
    start_month: int,
    end_year: int,
    end_month: int,
) -> List[Tuple[int, int]]:
    """
    Create an inclusive list of ``(year, month)`` tuples between start and end.

    :param start_year: Start year.
    :param start_month: Start month (1..12).
    :param end_year: End year.
    :param end_month: End month (1..12).
    :return: List of ``(year, month)`` tuples.
    """
    out_year_mnt_lst: List[Tuple[int, int]] = []
    for year in range(int(start_year), int(end_year) + 1):
        m_start = int(start_month) if year == start_year else 1
        m_end = int(end_month) if year == end_year else 12
        for month in range(m_start, m_end + 1):
            out_year_mnt_lst.append((year, month))
    return out_year_mnt_lst


def create_year_month_n_months_lst(
    start_year: int,
    start_month: int,
    n_months_fwd: int,
    n_months_bwd: int = 0,
) -> List[Tuple[int, int]]:
    """
    Create a list of ``(year, month)`` tuples spanning ``n_months_bwd`` months
    before and ``n_months_fwd`` months after ``(start_year, start_month)``.

    :param start_year: Reference start year (>= 0).
    :param start_month: Reference start month (1..12).
    :param n_months_fwd: Number of months forward from the start month.
    :param n_months_bwd: Number of months backward from the start month.
    :return: List of ``(year, month)`` tuples.
    """
    if start_year < 0:
        raise rsgispy.RSGISPyException("Year must be positive")
    if (start_month < 1) or (start_month > 12):
        raise rsgispy.RSGISPyException("Month must be between 1-12")

    base_idx = int(start_year) * 12 + (int(start_month) - 1)
    out_year_mnt_lst: List[Tuple[int, int]] = []
    for offset in range(-int(n_months_bwd), int(n_months_fwd) + 1):
        year_val, month_zero = divmod(base_idx + offset, 12)
        out_year_mnt_lst.append((year_val, month_zero + 1))
    return out_year_mnt_lst


def is_summer_winter(lat: float, date: datetime.date) -> int:
    """
    Determine whether ``date`` falls in summer (1) or winter (2) at latitude ``lat``.

    Southern Hemisphere Winter (``lat < 0``): May - October (months 5..10).
    Northern Hemisphere Summer (``lat >= 0``): April - September (months 4..9).

    :param lat: Latitude in decimal degrees.
    :param date: ``datetime.date`` (or ``datetime.datetime``) object.
    :return: ``1`` for summer, ``2`` for winter.
    """
    if lat < 0:
        if 4 < date.month < 11:
            return 2
        return 1
    if 3 < date.month < 10:
        return 1
    return 2


def get_datetime_increment_lst(
    start_date: datetime.datetime,
    end_date: datetime.datetime,
    interval: datetime.timedelta,
) -> List[datetime.datetime]:
    """
    Return a list of datetimes starting after ``start_date`` in steps of
    ``interval`` until reaching or exceeding ``end_date``.

    :param start_date: Start datetime of the sequence.
    :param end_date: End datetime of the sequence.
    :param interval: Step ``datetime.timedelta``.
    :return: List of ``datetime.datetime`` objects.
    """
    date_lst: List[datetime.datetime] = []
    c_date = start_date
    while c_date < end_date:
        c_date = c_date + interval
        date_lst.append(c_date)
    return date_lst


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
    if not is_number(curr_obj):
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


def encode_base64_text(input_txt: str) -> str:
    """
    Encode ``input_txt`` as a base64 ASCII string.

    :param input_txt: Plain text string to encode.
    :return: Base64-encoded string.
    """
    return base64.b64encode(input_txt.encode("utf-8")).decode("utf-8")


def decode_base64_text(in_encoded_txt: str) -> str:
    """
    Decode a base64 string produced by :func:`encode_base64_text`.

    :param in_encoded_txt: Base64-encoded string.
    :return: Decoded plain text string.
    """
    return base64.b64decode(in_encoded_txt.encode("utf-8")).decode("utf-8")


def create_username_password_file(
    username: str,
    password: str,
    out_file: str,
) -> None:
    """
    Write a two-line base64-obfuscated username/password file.

    :param username: Username string.
    :param password: Password string.
    :param out_file: Output file path.
    """
    out_data = [encode_base64_text(username), encode_base64_text(password)]
    write_list_to_file(out_data, out_file)


def get_username_password(input_file: str) -> Tuple[str, str]:
    """
    Read and decode a username/password file created by
    :func:`create_username_password_file`.

    :param input_file: Input credentials file path.
    :return: ``(username, password)`` tuple.
    """
    in_data_lst = read_text_file_to_list(input_file)
    if len(in_data_lst) != 2:
        raise rsgispy.RSGISPyException(
            "There should just be two lines in username/password file."
        )
    return (
        decode_base64_text(in_data_lst[0]),
        decode_base64_text(in_data_lst[1]),
    )


def prettify_xml_file(input_file: str, output_file: str) -> None:
    """
    Prettify an XML file by inserting standard indentation and newlines.

    :param input_file: Input XML file path.
    :param output_file: Output formatted XML file path.
    """
    xml_str = read_text_file_no_new_lines(input_file)
    xml_elements = ET.XML(xml_str)
    ET.indent(xml_elements)
    pretty_xml = ET.tostring(xml_elements, encoding="unicode")
    write_data_to_file(pretty_xml, output_file)
