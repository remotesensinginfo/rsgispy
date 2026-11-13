"""
The tools.jsonio module provides JSON and gzipped-JSON serialisation and
deserialisation utilities, including support for NumPy scalars/arrays and
datetime objects.
"""

import datetime
import gzip
import json
from typing import Any, Dict, List, Union

import numpy

__all__ = [
    "RSGISNumpyArrayEncoder",
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
