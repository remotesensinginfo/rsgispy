"""
The tools.jsonio module provides JSON and gzipped-JSON serialisation and
deserialisation utilities, including support for NumPy scalars/arrays and
datetime objects.
"""

import datetime
import gzip
import json
from typing import Any, Dict, List, Union

import numpy as np

__all__ = [
    "RSGISNumpyArrayEncoder",
    "to_json_compatible",
    "write_dict_to_json",
    "dict_to_json_str",
    "write_dict_to_json_gz",
    "read_json_to_dict",
    "read_gz_json_to_dict",
]


class RSGISNumpyArrayEncoder(json.JSONEncoder):
    """JSON encoder supporting NumPy scalar/array types and datetime objects."""

    def default(self, o: Any) -> Any:
        encoded: Any = None
        handled = True
        if isinstance(o, np.integer):
            encoded = int(o)
        elif isinstance(o, np.floating):
            encoded = float(o)
        elif isinstance(o, np.complexfloating):
            encoded = float(o.real) if o.imag == 0 else str(o)
        elif isinstance(o, np.ndarray):
            encoded = o.tolist()
        elif isinstance(o, np.bool_):
            encoded = bool(o)
        elif isinstance(o, np.bytes_):
            encoded = str(o)
        elif isinstance(o, np.void):
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


_NUMPY_JSON_ENCODER = RSGISNumpyArrayEncoder()


def to_json_compatible(data: Any) -> Any:
    """
    Recursively convert NumPy scalars/arrays, datetime objects, and ``to_json``
    objects into native JSON-serialisable Python types using
    :class:`RSGISNumpyArrayEncoder`.

    :param data: Arbitrary Python/NumPy object or nested container.
    :return: JSON-compatible Python representation.

    .. code:: python

        import numpy as np
        import rsgispy.tools.jsonio

        clean = rsgispy.tools.jsonio.to_json_compatible(
            {"n": np.int64(10), "vals": np.array([1.5, 2.5])}
        )
    """
    if isinstance(data, dict):
        return {str(k): to_json_compatible(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [to_json_compatible(v) for v in data]
    if data is None or type(data) in (bool, int, float, str):
        return data
    try:
        encoded = _NUMPY_JSON_ENCODER.default(data)
    except TypeError:
        return data
    if isinstance(encoded, (dict, list, tuple)):
        return to_json_compatible(encoded)
    return encoded


def write_dict_to_json(
    data_dict: Union[Dict[str, Any], List[Any]],
    out_file: str,
) -> None:
    """
    Write a dictionary (or list) to a pretty-printed JSON file.

    :param data_dict: Data structure to serialise to JSON.
    :param out_file: Output JSON file path.

    .. code:: python

        import rsgispy.tools.jsonio
        rsgispy.tools.jsonio.write_dict_to_json({"a": 1, "b": 2}, "out.json")
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

    .. code:: python

        import rsgispy.tools.jsonio
        json_str = rsgispy.tools.jsonio.dict_to_json_str({"a": 1, "b": 2})
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

    .. code:: python

        import rsgispy.tools.jsonio
        rsgispy.tools.jsonio.write_dict_to_json_gz({"a": 1}, "out.json.gz")
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

    .. code:: python

        import rsgispy.tools.jsonio
        data = rsgispy.tools.jsonio.read_json_to_dict("out.json")
    """
    with open(input_file, "r", encoding="utf-8") as in_fp:
        return json.load(in_fp)


def read_gz_json_to_dict(input_file: str, encoding: str = "utf-8") -> Any:
    """
    Read a gzip-compressed JSON file and return the parsed Python object.

    :param input_file: Input ``.json.gz`` file path.
    :param encoding: Text encoding (Default = ``'utf-8'``).
    :return: Parsed JSON dict or list.

    .. code:: python

        import rsgispy.tools.jsonio
        data = rsgispy.tools.jsonio.read_gz_json_to_dict("out.json.gz")
    """
    with gzip.GzipFile(input_file, "r") as f_in:
        json_bytes = f_in.read()
    return json.loads(json_bytes.decode(encoding))
