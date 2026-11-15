"""
The tools.tables module provides helpers for resolving tabular output formats and
writing :class:`pandas.DataFrame` objects to disk (including compressed Parquet
and common text/binary table formats).
"""

import os
from typing import Dict, Optional, Tuple, Union

import pandas as pd

import rsgispy
from rsgispy.vectorprocess import infer_parquet_compression, is_geoparquet_file

__all__ = [
    "resolve_table_format",
    "resolve_table_output_spec",
    "write_dataframe_table",
]

_TABLE_EXT_TO_FORMAT: Dict[str, str] = {
    ".csv": "csv",
    ".tsv": "tsv",
    ".txt": "tsv",
    ".parquet": "parquet",
    ".pq": "parquet",
    ".feather": "feather",
    ".ftr": "feather",
    ".xlsx": "excel",
    ".xls": "excel",
    ".xlsm": "excel",
    ".json": "json",
    ".pkl": "pickle",
    ".pickle": "pickle",
    ".h5": "hdf",
    ".hdf5": "hdf",
}

_TABLE_EXTS: Tuple[str, ...] = tuple(
    ext for ext in _TABLE_EXT_TO_FORMAT if ext not in (".txt", ".json")
)

_TABLE_FORMAT_ALIASES: Dict[str, str] = {
    "csv": "csv",
    "tsv": "tsv",
    "tab": "tsv",
    "parquet": "parquet",
    "pq": "parquet",
    "feather": "feather",
    "ftr": "feather",
    "excel": "excel",
    "xlsx": "excel",
    "xls": "excel",
    "json": "json",
    "pickle": "pickle",
    "pkl": "pickle",
    "hdf": "hdf",
    "hdf5": "hdf",
    "h5": "hdf",
}

_TEXT_COMP_EXTS = (".gz", ".gzip", ".bz2", ".xz", ".zst", ".zip")


def resolve_table_format(
    out_file: Union[str, os.PathLike],
    out_format: Optional[str] = None,
) -> str:
    """
    Resolve the normalised table format name from ``out_format`` or file suffix.

    :param out_file: Table file path.
    :param out_format: Optional explicit format name overriding extension
                       detection.
    :return: Canonical format name (e.g., ``'csv'``, ``'parquet'``, ``'excel'``).

    .. code:: python

        import rsgispy.tools.tables

        fmt = rsgispy.tools.tables.resolve_table_format("data.parquet.zstd")
    """
    path_str = os.fspath(out_file)
    lower_path = path_str.lower().strip()

    if out_format is not None:
        key = out_format.strip().lower().lstrip(".")
        if key not in _TABLE_FORMAT_ALIASES:
            raise rsgispy.RSGISPyException(
                f"Unsupported table format '{out_format}'. Supported "
                f"formats: {sorted(set(_TABLE_FORMAT_ALIASES.values()))}."
            )
        return _TABLE_FORMAT_ALIASES[key]

    if is_geoparquet_file(path_str):
        return "parquet"

    ext = os.path.splitext(lower_path)[1]
    if ext in _TEXT_COMP_EXTS:
        base_ext = os.path.splitext(lower_path[: -len(ext)])[1]
        ext = base_ext if base_ext in (".csv", ".tsv", ".txt") else ext
    if ext not in _TABLE_EXT_TO_FORMAT:
        raise rsgispy.RSGISPyException(
            f"Cannot infer table format from extension '{ext}' in "
            f"'{path_str}'. Specify format explicitly."
        )
    return _TABLE_EXT_TO_FORMAT[ext]


def resolve_table_output_spec(
    out_file: Union[str, os.PathLike],
    out_format: Optional[str] = None,
    out_compress: Optional[Union[str, bool]] = None,
) -> Tuple[str, Optional[str]]:
    """
    Resolve ``(format_name, compression_codec)`` for an output tabular file.

    Supports Parquet compression options (explicit codecs and suffix inference
    such as ``.parquet.gzip`` or ``.parquet.zstd``) as well as CSV, TSV,
    Feather, Excel, JSON, Pickle, and HDF5.

    :param out_file: Output table file path.
    :param out_format: Optional output table format overriding extension
                       detection (e.g., ``'parquet'``, ``'csv'``, ``'tsv'``,
                       ``'feather'``, ``'excel'``, ``'json'``, ``'pickle'``,
                       ``'hdf'``).
    :param out_compress: Optional compression setting. For Parquet: ``None`` to
                         infer from extension or write uncompressed, ``True``
                         for ``'snappy'``, ``False``/``'none'`` for uncompressed,
                         or codec name (``'snappy'``, ``'gzip'``, ``'zstd'``,
                         ``'brotli'``, ``'lz4'``). For CSV/TSV: ``None`` uses
                         ``'infer'``; ``True`` uses ``'gzip'``; ``False`` or
                         ``'none'`` disables compression; or a pandas-supported
                         compression string.
    :return: Tuple of ``(resolved_format, compression_codec)``.

    .. code:: python

        import rsgispy.tools.tables

        fmt, comp = rsgispy.tools.tables.resolve_table_output_spec(
            "pixels.parquet.zstd"
        )
    """
    path_str = os.fspath(out_file)
    fmt = resolve_table_format(path_str, out_format=out_format)

    if fmt == "parquet":
        comp = infer_parquet_compression(path_str, out_compress=out_compress)
    elif out_compress is None:
        comp = "infer" if fmt in ("csv", "tsv") else None
    elif isinstance(out_compress, bool):
        comp = "gzip" if out_compress else None
    elif isinstance(out_compress, str):
        c_val = out_compress.strip().lower()
        comp = None if c_val in ("none", "uncompressed", "") else c_val
    else:
        raise rsgispy.RSGISPyException(
            f"Unsupported out_compress setting: {out_compress!r}"
        )
    return fmt, comp


def write_dataframe_table(
    df: pd.DataFrame,
    out_file: Union[str, os.PathLike],
    out_format: Optional[str] = None,
    out_compress: Optional[Union[str, bool]] = None,
) -> None:
    """
    Write a :class:`pandas.DataFrame` to ``out_file`` in Parquet (with optional
    compression) or another common tabular format.

    Creates parent directories if they do not already exist.

    :param df: Input :class:`pandas.DataFrame` to export.
    :param out_file: Output file path (e.g., ``'data.parquet'``,
                     ``'data.parquet.zstd'``, ``'data.csv'``).
    :param out_format: Optional output table format overriding extension
                       detection (e.g., ``'parquet'``, ``'csv'``, ``'tsv'``,
                       ``'feather'``, ``'excel'``, ``'json'``, ``'pickle'``,
                       ``'hdf'``).
    :param out_compress: Optional compression setting passed through
                         :func:`resolve_table_output_spec`.

    .. code:: python

        import pandas as pd
        import rsgispy.tools.tables

        df = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
        rsgispy.tools.tables.write_dataframe_table(
            df, "table.parquet", out_compress="zstd"
        )
    """
    # pylint: disable=too-many-branches
    path_str = os.fspath(out_file)
    fmt, comp = resolve_table_output_spec(
        path_str, out_format=out_format, out_compress=out_compress
    )

    out_dir = os.path.dirname(os.path.abspath(path_str))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    if fmt == "parquet":
        try:
            df.to_parquet(path_str, index=False, compression=comp)
        except ImportError as err:
            raise rsgispy.RSGISPyException(
                "Writing Parquet files requires 'pyarrow' or 'fastparquet'."
            ) from err
    elif fmt == "csv":
        df.to_csv(path_str, index=False, compression=comp)
    elif fmt == "tsv":
        df.to_csv(path_str, sep="\t", index=False, compression=comp)
    elif fmt == "feather":
        try:
            df.to_feather(path_str)
        except ImportError as err:
            raise rsgispy.RSGISPyException(
                "Writing Feather files requires 'pyarrow'."
            ) from err
    elif fmt == "excel":
        try:
            df.to_excel(path_str, index=False, engine="openpyxl")
        except ImportError:
            try:
                df.to_excel(path_str, index=False, engine="xlsxwriter")
            except ImportError as err:
                raise rsgispy.RSGISPyException(
                    "Writing Excel files requires 'openpyxl' or 'xlsxwriter'."
                ) from err
    elif fmt == "json":
        df.to_json(path_str, orient="records", indent=2)
    elif fmt == "pickle":
        df.to_pickle(path_str)
    else:  # hdf
        try:
            df.to_hdf(path_str, key="data", mode="w")
        except ImportError as err:
            raise rsgispy.RSGISPyException(
                "Writing HDF5 files requires 'tables' (PyTables)."
            ) from err
