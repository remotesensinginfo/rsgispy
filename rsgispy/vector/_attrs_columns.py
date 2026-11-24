"""
The vector.attrs module provides functions to add, derive, clean, and
categorise columns in vector attribute tables using ``geopandas``.
"""

import datetime
import os
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import pymu_eval

import rsgispy
import rsgispy.tools.jsonio
import rsgispy.tools.maths
import rsgispy.tools.strings
from rsgispy.vectorprocess import (
    VectorInput,
    _resolve_cast_kind,
    cast_col_data,
    read_vec_gdf,
    to_python_scalar,
    validate_columns_exist,
    write_vec_gdf,
)

__all__ = [
    "add_fid_col",
    "add_unq_numeric_col",
    "add_numeric_col",
    "add_numeric_col_lut",
    "add_numeric_col_range_lut",
    "add_numeric_col_from_lst_lut",
    "add_string_col",
    "add_string_col_lut",
    "add_ran_vals_cols",
    "create_name_col",
    "check_str_col",
    "create_date_col",
    "find_replace_str_vec_lyr",
    "create_angle_sets",
    "create_orthogonal_angle_sets",
    "vector_maths",
]


def add_fid_col(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    *,
    out_format: str = "GPKG",
    out_col: str = "fid",
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Add a 1-based unique integer feature ID column to a vector layer.

    :param vec_file: Input vector file path or :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param out_vec_file: Output vector file path, or ``None`` to return an
                         in-memory :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param out_col: Output FID column name (Default = ``'fid'``).
    :param out_compress: Optional GeoParquet compression setting (``None`` to
                         infer from extension or write uncompressed, ``True``
                         for ``'snappy'``, ``False``/``'none'`` for uncompressed,
                         or codec name ``'snappy'``, ``'gzip'``, ``'zstd'``,
                         ``'brotli'``, ``'lz4'``).
    :return: Updated :class:`geopandas.GeoDataFrame` when ``out_vec_file`` is
             ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.vector.attrs

        rsgispy.vector.attrs.add_fid_col(
            "in.gpkg", "polys", "out.gpkg", "polys", out_col="feat_id"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    base_gdf = read_vec_gdf(vec_file, vec_lyr)
    base_gdf[out_col] = np.arange(1, len(base_gdf) + 1, 1, dtype=int)
    if out_vec_file is None:
        return base_gdf
    write_vec_gdf(
        base_gdf,
        out_vec_file,
        out_vec_lyr,
        out_format=out_format,
        out_compress=out_compress,
    )
    return None


def add_unq_numeric_col(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    unq_col: str,
    out_col: str,
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    *,
    out_format: str = "GPKG",
    lut_json_file: Optional[Union[str, os.PathLike]] = None,
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Add a sequential integer ID column (starting at ``1``) mapped from unique
    values in ``unq_col``, optionally writing the lookup table to JSON.

    :param vec_file: Input vector file path or :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param unq_col: Source column whose unique values define the mapping.
    :param out_col: Output numeric column name.
    :param out_vec_file: Output vector file path, or ``None`` to return an
                         in-memory :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param lut_json_file: Optional path to write the JSON lookup table.
    :param out_compress: Optional GeoParquet compression setting (``None`` to
                         infer from extension or write uncompressed, ``True``
                         for ``'snappy'``, ``False``/``'none'`` for uncompressed,
                         or codec name ``'snappy'``, ``'gzip'``, ``'zstd'``,
                         ``'brotli'``, ``'lz4'``).
    :return: Updated :class:`geopandas.GeoDataFrame` when ``out_vec_file`` is
             ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.vector.attrs

        rsgispy.vector.attrs.add_unq_numeric_col(
            "in.gpkg", "polys", "species", "sp_id", "out.gpkg", "polys"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    base_gdf = read_vec_gdf(vec_file, vec_lyr)
    validate_columns_exist(base_gdf, [unq_col])

    unq_vals = base_gdf[unq_col].dropna().unique()
    na_mask = base_gdf[unq_col].isna()
    base_gdf[out_col] = np.zeros(len(base_gdf), dtype=int)
    lut: Dict[str, Dict[Any, Any]] = {"id": {}, "val": {}}

    idx = 0
    for raw_val in unq_vals:
        idx += 1
        py_val = to_python_scalar(raw_val)
        sel_rows = base_gdf[unq_col] == raw_val
        base_gdf.loc[sel_rows, out_col] = idx
        lut["id"][str(idx)] = py_val
        lut["val"][str(py_val)] = idx

    if bool(na_mask.any()):
        idx += 1
        base_gdf.loc[na_mask, out_col] = idx
        lut["id"][str(idx)] = None
        lut["val"]["None"] = idx

    if lut_json_file is not None:
        rsgispy.tools.jsonio.write_dict_to_json(lut, os.fspath(lut_json_file))
    if out_vec_file is None:
        return base_gdf
    write_vec_gdf(
        base_gdf,
        out_vec_file,
        out_vec_lyr,
        out_format=out_format,
        out_compress=out_compress,
    )
    return None


def add_numeric_col(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    out_col: str,
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    out_val: float = 1,
    *,
    out_format: str = "GPKG",
    out_col_int: bool = False,
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Add a constant numeric column to a vector layer.

    :param vec_file: Input vector file path or :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param out_col: Output numeric column name.
    :param out_vec_file: Output vector file path, or ``None`` to return an
                         in-memory :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name.
    :param out_val: Constant value assigned to all rows (Default = ``1``).
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param out_col_int: If ``True``, store values as integers; otherwise float
                        (Default = ``False``).
    :param out_compress: Optional GeoParquet compression setting (``None`` to
                         infer from extension or write uncompressed, ``True``
                         for ``'snappy'``, ``False``/``'none'`` for uncompressed,
                         or codec name ``'snappy'``, ``'gzip'``, ``'zstd'``,
                         ``'brotli'``, ``'lz4'``).
    :return: Updated :class:`geopandas.GeoDataFrame` when ``out_vec_file`` is
             ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.vector.attrs

        rsgispy.vector.attrs.add_numeric_col(
            "in.gpkg", "polys", "weight", "out.gpkg", "polys", out_val=1.5
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    base_gdf = read_vec_gdf(vec_file, vec_lyr)
    dtype = int if out_col_int else float
    base_gdf[out_col] = np.full(len(base_gdf), out_val, dtype=dtype)
    if out_vec_file is None:
        return base_gdf
    write_vec_gdf(
        base_gdf,
        out_vec_file,
        out_vec_lyr,
        out_format=out_format,
        out_compress=out_compress,
    )
    return None


def add_numeric_col_lut(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    ref_col: str,
    val_lut: Dict[Any, Union[int, float]],
    out_col: str,
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    *,
    out_format: str = "GPKG",
    col_dtype: str = "int",
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Add a numeric column derived from ``ref_col`` via a dictionary lookup table.

    :param vec_file: Input vector file path or :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param ref_col: Reference column name whose values match keys in ``val_lut``.
    :param val_lut: Dictionary mapping reference values to output numbers.
    :param out_col: Output numeric column name.
    :param out_vec_file: Output vector file path, or ``None`` to return an
                         in-memory :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param col_dtype: Output column data type (``'int'`` or ``'float'``;
                      Default = ``'int'``).
    :param out_compress: Optional GeoParquet compression setting (``None`` to
                         infer from extension or write uncompressed, ``True``
                         for ``'snappy'``, ``False``/``'none'`` for uncompressed,
                         or codec name ``'snappy'``, ``'gzip'``, ``'zstd'``,
                         ``'brotli'``, ``'lz4'``).
    :return: Updated :class:`geopandas.GeoDataFrame` when ``out_vec_file`` is
             ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.vector.attrs

        rsgispy.vector.attrs.add_numeric_col_lut(
            "in.gpkg",
            "polys",
            "cls",
            {"Forest": 1, "Water": 2},
            "cls_id",
            "out.gpkg",
            "polys",
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    dtype_key = col_dtype.lower()
    if dtype_key not in ("int", "float"):
        raise rsgispy.RSGISPyException(
            f"col_dtype must be 'int' or 'float', got '{col_dtype}'."
        )

    base_gdf = read_vec_gdf(vec_file, vec_lyr)
    validate_columns_exist(base_gdf, [ref_col])

    np_dtype = int if dtype_key == "int" else float
    base_gdf[out_col] = np.zeros(len(base_gdf), dtype=np_dtype)
    for lut_key, out_val in val_lut.items():
        sel_rows = base_gdf[ref_col] == lut_key
        base_gdf.loc[sel_rows, out_col] = out_val

    if out_vec_file is None:
        return base_gdf
    write_vec_gdf(
        base_gdf,
        out_vec_file,
        out_vec_lyr,
        out_format=out_format,
        out_compress=out_compress,
    )
    return None


def add_numeric_col_range_lut(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    vec_col: str,
    out_vec_file: Optional[Union[str, os.PathLike]],
    out_vec_lyr: Optional[str],
    out_vec_col: str,
    val_lut: Dict[int, Tuple[float, float]],
    *,
    out_format: str = "GPKG",
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Add a numeric class column using half-open ``[lower, upper)`` value ranges.

    Features outside all configured intervals receive ``0``.

    :param vec_file: Input vector file path or :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param vec_col: Numeric source column name to classify.
    :param out_vec_file: Output vector file path, or ``None`` to return an
                         in-memory :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name.
    :param out_vec_col: Output numeric column name.
    :param val_lut: Dictionary mapping output integer code to ``(low, high)``
                    bounds checked as ``low <= val < high``.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param out_compress: Optional GeoParquet compression setting (``None`` to
                         infer from extension or write uncompressed, ``True``
                         for ``'snappy'``, ``False``/``'none'`` for uncompressed,
                         or codec name ``'snappy'``, ``'gzip'``, ``'zstd'``,
                         ``'brotli'``, ``'lz4'``).
    :return: Updated :class:`geopandas.GeoDataFrame` when ``out_vec_file`` is
             ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.vector.attrs

        rsgispy.vector.attrs.add_numeric_col_range_lut(
            "in.gpkg",
            "polys",
            "elev",
            "out.gpkg",
            "polys",
            "elev_cls",
            {1: (0.0, 100.0), 2: (100.0, 500.0)},
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    data_gdf = read_vec_gdf(vec_file, vec_lyr)
    validate_columns_exist(data_gdf, [vec_col])

    out_arr = np.zeros(len(data_gdf), dtype=int)
    vals_arr = data_gdf[vec_col].to_numpy()
    for out_code, (low_val, high_val) in val_lut.items():
        out_arr[(vals_arr >= low_val) & (vals_arr < high_val)] = out_code

    data_gdf[out_vec_col] = out_arr
    if out_vec_file is None:
        return data_gdf
    write_vec_gdf(
        data_gdf,
        out_vec_file,
        out_vec_lyr,
        out_format=out_format,
        out_compress=out_compress,
    )
    return None


def add_numeric_col_from_lst_lut(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    ref_col: str,
    vals_lut: Sequence[Tuple[Union[str, int], int]],
    out_col: str,
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    *,
    out_format: str = "GPKG",
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Add an integer column derived from ``ref_col`` using a sequence of
    ``(match_value, output_int)`` pairs.

    :param vec_file: Input vector file path or :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param ref_col: Reference column name.
    :param vals_lut: Sequence of ``(lookup_value, out_int)`` tuples.
    :param out_col: Output numeric column name.
    :param out_vec_file: Output vector file path, or ``None`` to return an
                         in-memory :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param out_compress: Optional GeoParquet compression setting (``None`` to
                         infer from extension or write uncompressed, ``True``
                         for ``'snappy'``, ``False``/``'none'`` for uncompressed,
                         or codec name ``'snappy'``, ``'gzip'``, ``'zstd'``,
                         ``'brotli'``, ``'lz4'``).
    :return: Updated :class:`geopandas.GeoDataFrame` when ``out_vec_file`` is
             ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.vector.attrs

        rsgispy.vector.attrs.add_numeric_col_from_lst_lut(
            "in.gpkg",
            "polys",
            "label",
            [("A", 1), ("B", 2)],
            "label_id",
            "out.gpkg",
            "polys",
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    base_gdf = read_vec_gdf(vec_file, vec_lyr)
    validate_columns_exist(base_gdf, [ref_col])

    out_vals = np.zeros(len(base_gdf), dtype=int)
    ref_vals = base_gdf[ref_col].to_numpy()
    for match_val, out_int in vals_lut:
        out_vals[ref_vals == match_val] = out_int

    base_gdf[out_col] = out_vals
    if out_vec_file is None:
        return base_gdf
    write_vec_gdf(
        base_gdf,
        out_vec_file,
        out_vec_lyr,
        out_format=out_format,
        out_compress=out_compress,
    )
    return None


def add_string_col(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    out_col: str,
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    out_val: str = "str_val",
    *,
    out_format: str = "GPKG",
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Add a constant string column to a vector layer.

    :param vec_file: Input vector file path or :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param out_col: Output string column name.
    :param out_vec_file: Output vector file path, or ``None`` to return an
                         in-memory :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name.
    :param out_val: Constant string assigned to all rows (Default = ``'str_val'``).
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param out_compress: Optional GeoParquet compression setting (``None`` to
                         infer from extension or write uncompressed, ``True``
                         for ``'snappy'``, ``False``/``'none'`` for uncompressed,
                         or codec name ``'snappy'``, ``'gzip'``, ``'zstd'``,
                         ``'brotli'``, ``'lz4'``).
    :return: Updated :class:`geopandas.GeoDataFrame` when ``out_vec_file`` is
             ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.vector.attrs

        rsgispy.vector.attrs.add_string_col(
            "in.gpkg", "polys", "source", "out.gpkg", "polys", out_val="survey_2026"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    base_gdf = read_vec_gdf(vec_file, vec_lyr)
    base_gdf[out_col] = [str(out_val)] * len(base_gdf)
    if out_vec_file is None:
        return base_gdf
    write_vec_gdf(
        base_gdf,
        out_vec_file,
        out_vec_lyr,
        out_format=out_format,
        out_compress=out_compress,
    )
    return None


def add_string_col_lut(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    ref_col: str,
    val_lut: Dict[Any, str],
    out_col: str,
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    *,
    out_format: str = "GPKG",
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Add a string column derived from ``ref_col`` via a dictionary lookup table.

    :param vec_file: Input vector file path or :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param ref_col: Reference column name whose values match keys in ``val_lut``.
    :param val_lut: Dictionary mapping reference values to output strings.
    :param out_col: Output string column name.
    :param out_vec_file: Output vector file path, or ``None`` to return an
                         in-memory :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param out_compress: Optional GeoParquet compression setting (``None`` to
                         infer from extension or write uncompressed, ``True``
                         for ``'snappy'``, ``False``/``'none'`` for uncompressed,
                         or codec name ``'snappy'``, ``'gzip'``, ``'zstd'``,
                         ``'brotli'``, ``'lz4'``).
    :return: Updated :class:`geopandas.GeoDataFrame` when ``out_vec_file`` is
             ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.vector.attrs

        rsgispy.vector.attrs.add_string_col_lut(
            "in.gpkg",
            "polys",
            "cls_id",
            {1: "Forest", 2: "Water"},
            "cls_name",
            "out.gpkg",
            "polys",
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    base_gdf = read_vec_gdf(vec_file, vec_lyr)
    validate_columns_exist(base_gdf, [ref_col])

    base_gdf[out_col] = ""
    for lut_key, out_str in val_lut.items():
        sel_rows = base_gdf[ref_col] == lut_key
        base_gdf.loc[sel_rows, out_col] = str(out_str)

    if out_vec_file is None:
        return base_gdf
    write_vec_gdf(
        base_gdf,
        out_vec_file,
        out_vec_lyr,
        out_format=out_format,
        out_compress=out_compress,
    )
    return None


def add_ran_vals_cols(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    *,
    out_format: str = "GPKG",
    out_col: str = "ran_vals",
    rnd_seed: Optional[int] = None,
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Add a column populated with random integers in ``[1, 10 * n_rows)``.

    Note that random values are not guaranteed to be unique across rows.

    :param vec_file: Input vector file path or :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param out_vec_file: Output vector file path, or ``None`` to return an
                         in-memory :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param out_col: Output column name (Default = ``'ran_vals'``).
    :param rnd_seed: Optional seed for NumPy's random generator.
    :param out_compress: Optional GeoParquet compression setting (``None`` to
                         infer from extension or write uncompressed, ``True``
                         for ``'snappy'``, ``False``/``'none'`` for uncompressed,
                         or codec name ``'snappy'``, ``'gzip'``, ``'zstd'``,
                         ``'brotli'``, ``'lz4'``).
    :return: Updated :class:`geopandas.GeoDataFrame` when ``out_vec_file`` is
             ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.vector.attrs

        rsgispy.vector.attrs.add_ran_vals_cols(
            "in.gpkg", "polys", "out.gpkg", "polys", rnd_seed=42
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    data_gdf = read_vec_gdf(vec_file, vec_lyr)
    rng = np.random.default_rng(seed=rnd_seed)
    n_rows = len(data_gdf)
    high = max(2, n_rows * 10)
    data_gdf[out_col] = rng.integers(low=1, high=high, size=n_rows)
    if out_vec_file is None:
        return data_gdf
    write_vec_gdf(
        data_gdf,
        out_vec_file,
        out_vec_lyr,
        out_format=out_format,
        out_compress=out_compress,
    )
    return None


def _format_coord_component(
    val: float,
    coord_gain: float,
    zero_pad: int,
    round_n_digits: int,
    int_coords: bool,
) -> str:
    """Format a single X or Y coordinate scalar for :func:`create_name_col`."""
    scaled = val * coord_gain
    if zero_pad > 0:
        return rsgispy.tools.strings.zero_pad_num_str(
            scaled,
            str_len=zero_pad,
            round_num=False,
            round_n_digts=round_n_digits,
            integerise=int_coords,
        )
    return f"{int(scaled)}"


def create_name_col(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    *,
    out_format: str = "GPKG",
    out_col: str = "names",
    x_col: str = "MinX",
    y_col: str = "MaxY",
    prefix: str = "",
    postfix: str = "",
    coords_lat_lon: bool = True,
    int_coords: bool = True,
    coord_gain: float = 1.0,
    zero_x_pad: int = 0,
    zero_y_pad: int = 0,
    round_n_digts: int = 0,
    non_neg: bool = False,
    replace_dec_pt: bool = True,
    dec_pt_val: str = "",
    use_pbar: bool = True,
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Create a feature name column formatted from X and Y coordinate attributes.

    :param vec_file: Input vector file path or :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param out_vec_file: Output vector file path, or ``None`` to return an
                         in-memory :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param out_col: Output name column (Default = ``'names'``).
    :param x_col: Column containing X coordinates (Default = ``'MinX'``).
    :param y_col: Column containing Y coordinates (Default = ``'MaxY'``).
    :param prefix: Optional name prefix.
    :param postfix: Optional name suffix.
    :param coords_lat_lon: If ``True``, format with ``N/S`` and ``E/W``
                           hemisphere prefixes; otherwise ``E<x>N<y>``
                           (Default = ``True``).
    :param int_coords: If ``True``, integerise coordinates before formatting
                       (Default = ``True``).
    :param coord_gain: Multiplier applied to coordinates prior to formatting
                       (Default = ``1.0``).
    :param zero_x_pad: Zero-padding width for X when ``> 0`` (Default = ``0``).
    :param zero_y_pad: Zero-padding width for Y when ``> 0`` (Default = ``0``).
    :param round_n_digts: Significant decimal digits used with padding
                          (Default = ``0``).
    :param non_neg: If ``True``, use absolute coordinate magnitudes with sign
                    captured via hemisphere letters (Default = ``False``).
    :param replace_dec_pt: Replace decimal points with ``dec_pt_val``
                           (Default = ``True``).
    :param dec_pt_val: Replacement string for decimal points (Default = ``''``).
    :param use_pbar: If ``True``, display a tqdm progress bar while generating
                     names (Default = ``True``).
    :param out_compress: Optional GeoParquet compression setting (``None`` to
                         infer from extension or write uncompressed, ``True``
                         for ``'snappy'``, ``False``/``'none'`` for uncompressed,
                         or codec name ``'snappy'``, ``'gzip'``, ``'zstd'``,
                         ``'brotli'``, ``'lz4'``).
    :return: Updated :class:`geopandas.GeoDataFrame` when ``out_vec_file`` is
             ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.vector.attrs

        rsgispy.vector.attrs.create_name_col(
            "tiles.gpkg",
            "tiles",
            "tiles_named.gpkg",
            "tiles",
            prefix="tile_",
            coords_lat_lon=False,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    base_gdf = read_vec_gdf(vec_file, vec_lyr)
    validate_columns_exist(base_gdf, [x_col, y_col])

    tqdm_cls = rsgispy.get_tqdm()
    x_vals = base_gdf[x_col].tolist()
    y_vals = base_gdf[y_col].tolist()
    names: List[str] = []

    for x_val, y_val in tqdm_cls(
        zip(x_vals, y_vals), total=len(base_gdf), disable=not use_pbar
    ):
        x_neg = False
        y_neg = False
        if non_neg:
            if x_val < 0:
                x_neg = True
                x_val = -x_val
            if y_val < 0:
                y_neg = True
                y_val = -y_val

        x_str = _format_coord_component(
            x_val, coord_gain, zero_x_pad, round_n_digts, int_coords
        )
        y_str = _format_coord_component(
            y_val, coord_gain, zero_y_pad, round_n_digts, int_coords
        )

        if coords_lat_lon:
            hemi = "S" if y_neg else "N"
            east_west = "W" if x_neg else "E"
            name = f"{prefix}{hemi}{y_str}{east_west}{x_str}{postfix}"
        else:
            name = f"{prefix}E{x_str}N{y_str}{postfix}"

        if replace_dec_pt:
            name = name.replace(".", dec_pt_val)
        names.append(name)

    base_gdf[out_col] = names
    if out_vec_file is None:
        return base_gdf
    write_vec_gdf(
        base_gdf,
        out_vec_file,
        out_vec_lyr,
        out_format=out_format,
        out_compress=out_compress,
    )
    return None


def check_str_col(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    vec_col: str,
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    *,
    out_format: str = "GPKG",
    rm_non_ascii: bool = True,
    rm_dashs: bool = False,
    rm_spaces: bool = False,
    rm_punc: bool = False,
    use_pbar: bool = True,
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Sanitise values in a string column using :func:`rsgispy.tools.strings.check_str`.

    :param vec_file: Input vector file path or :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param vec_col: String column name to clean.
    :param out_vec_file: Output vector file path, or ``None`` to return an
                         in-memory :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param rm_non_ascii: Remove non-ASCII characters (Default = ``True``).
    :param rm_dashs: Replace dashes with underscores (Default = ``False``).
    :param rm_spaces: Replace spaces with underscores (Default = ``False``).
    :param rm_punc: Remove punctuation other than ``_`` and ``-``
                    (Default = ``False``).
    :param use_pbar: If ``True``, display a tqdm progress bar while cleaning
                     values (Default = ``True``).
    :param out_compress: Optional GeoParquet compression setting (``None`` to
                         infer from extension or write uncompressed, ``True``
                         for ``'snappy'``, ``False``/``'none'`` for uncompressed,
                         or codec name ``'snappy'``, ``'gzip'``, ``'zstd'``,
                         ``'brotli'``, ``'lz4'``).
    :return: Updated :class:`geopandas.GeoDataFrame` when ``out_vec_file`` is
             ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.vector.attrs

        rsgispy.vector.attrs.check_str_col(
            "in.gpkg", "polys", "name", "out.gpkg", "polys", rm_spaces=True
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    base_gdf = read_vec_gdf(vec_file, vec_lyr)
    validate_columns_exist(base_gdf, [vec_col])

    tqdm_cls = rsgispy.get_tqdm()
    raw_vals = base_gdf[vec_col].tolist()
    cleaned = [
        rsgispy.tools.strings.check_str(
            None if val is None else str(val),
            rm_non_ascii=rm_non_ascii,
            rm_dashs=rm_dashs,
            rm_spaces=rm_spaces,
            rm_punc=rm_punc,
        )
        for val in tqdm_cls(raw_vals, total=len(raw_vals), disable=not use_pbar)
    ]

    base_gdf[vec_col] = cleaned
    if out_vec_file is None:
        return base_gdf
    write_vec_gdf(
        base_gdf,
        out_vec_file,
        out_vec_lyr,
        out_format=out_format,
        out_compress=out_compress,
    )
    return None


def create_date_col(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    year_col: str,
    month_col: str,
    day_col: str,
    out_date_col: str,
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    *,
    out_format: str = "GPKG",
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Create a date/datetime column from integer year, month, and day columns.

    :param vec_file: Input vector file path or :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param year_col: Column containing integer years.
    :param month_col: Column containing integer months (1–12).
    :param day_col: Column containing integer days of month (1–31).
    :param out_date_col: Output date column name.
    :param out_vec_file: Output vector file path, or ``None`` to return an
                         in-memory :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param out_compress: Optional GeoParquet compression setting (``None`` to
                         infer from extension or write uncompressed, ``True``
                         for ``'snappy'``, ``False``/``'none'`` for uncompressed,
                         or codec name ``'snappy'``, ``'gzip'``, ``'zstd'``,
                         ``'brotli'``, ``'lz4'``).
    :return: Updated :class:`geopandas.GeoDataFrame` when ``out_vec_file`` is
             ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.vector.attrs

        rsgispy.vector.attrs.create_date_col(
            "in.gpkg", "pts", "yr", "mo", "dy", "obs_date", "out.gpkg", "pts"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    data_gdf = read_vec_gdf(vec_file, vec_lyr)
    validate_columns_exist(data_gdf, [year_col, month_col, day_col])

    out_dates = [
        datetime.datetime(int(y), int(m), int(d))
        for y, m, d in zip(
            data_gdf[year_col], data_gdf[month_col], data_gdf[day_col]
        )
    ]
    data_gdf[out_date_col] = out_dates
    if out_vec_file is None:
        return data_gdf
    write_vec_gdf(
        data_gdf,
        out_vec_file,
        out_vec_lyr,
        out_format=out_format,
        out_compress=out_compress,
    )
    return None


def find_replace_str_vec_lyr(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    out_vec_file: Optional[Union[str, os.PathLike]],
    out_vec_lyr: Optional[str],
    cols: Sequence[str],
    find_replace: Dict[str, str],
    *,
    out_format: str = "GPKG",
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Perform substring find-and-replace across specified string columns.

    :param vec_file: Input vector file path or :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param out_vec_file: Output vector file path, or ``None`` to return an
                         in-memory :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name.
    :param cols: Sequence of string column names to update.
    :param find_replace: Mapping ``{"find_str": "replacement_str"}``.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param out_compress: Optional GeoParquet compression setting (``None`` to
                         infer from extension or write uncompressed, ``True``
                         for ``'snappy'``, ``False``/``'none'`` for uncompressed,
                         or codec name ``'snappy'``, ``'gzip'``, ``'zstd'``,
                         ``'brotli'``, ``'lz4'``).
    :return: Updated :class:`geopandas.GeoDataFrame` when ``out_vec_file`` is
             ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.vector.attrs

        rsgispy.vector.attrs.find_replace_str_vec_lyr(
            "in.gpkg", "polys", "out.gpkg", "polys", ["cls"], {"N/A": "Unknown"}
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    base_gdf = read_vec_gdf(vec_file, vec_lyr)
    validate_columns_exist(base_gdf, cols)

    for col in cols:
        series = base_gdf[col].astype(str)
        for find_val, repl_val in find_replace.items():
            series = series.str.replace(find_val, repl_val, regex=False)
        base_gdf[col] = series

    if out_vec_file is None:
        return base_gdf
    write_vec_gdf(
        base_gdf,
        out_vec_file,
        out_vec_lyr,
        out_format=out_format,
        out_compress=out_compress,
    )
    return None


def create_angle_sets(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    angle_col: str,
    start_angle: int,
    angle_set_width: int,
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    *,
    out_format: str = "GPKG",
    out_angle_set_col: str = "angle_set",
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Partition point/feature azimuth angles (0–360 degrees) into mirrored
    angular sectors (each sector includes its opposite 180-degree direction).

    :param vec_file: Input vector file path or :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param angle_col: Column containing azimuth angles in ``[0, 360]`` degrees.
    :param start_angle: Starting angle in ``[0, 360]`` degrees.
    :param angle_set_width: Angular sector width in degrees; must divide ``180``.
    :param out_vec_file: Output vector file path, or ``None`` to return an
                         in-memory :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param out_angle_set_col: Output integer sector ID column
                              (Default = ``'angle_set'``).
    :param out_compress: Optional GeoParquet compression setting (``None`` to
                         infer from extension or write uncompressed, ``True``
                         for ``'snappy'``, ``False``/``'none'`` for uncompressed,
                         or codec name ``'snappy'``, ``'gzip'``, ``'zstd'``,
                         ``'brotli'``, ``'lz4'``).
    :return: Updated :class:`geopandas.GeoDataFrame` when ``out_vec_file`` is
             ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.vector.attrs

        rsgispy.vector.attrs.create_angle_sets(
            "pts.gpkg", "pts", "angle", 0, 45, "pts_sets.gpkg", "pts"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    if angle_set_width <= 0 or (180 % angle_set_width) != 0:
        raise rsgispy.RSGISPyException(
            "angle_set_width must divide evenly into 180 "
            "(e.g., 10, 30, 45, 90)."
        )
    if (start_angle < 0) or (start_angle > 360):
        raise rsgispy.RSGISPyException("start_angle must be between 0 and 360.")

    def _wrap_start(arr: np.ndarray) -> np.ndarray:
        out = arr.copy()
        out[out > 360] -= 360
        out[out == 360] = 0
        return out

    def _wrap_end(arr: np.ndarray) -> np.ndarray:
        out = arr.copy()
        out[out > 360] -= 360
        out[out == 0] = 360
        return out

    base_starts = start_angle + np.arange(0, 180, angle_set_width)
    p1_starts = _wrap_start(base_starts)
    p1_ends = _wrap_end(p1_starts + angle_set_width)
    p2_starts = _wrap_start(p1_starts + 180)
    p2_ends = _wrap_end(p1_ends + 180)

    data_gdf = read_vec_gdf(vec_file, vec_lyr)
    validate_columns_exist(data_gdf, [angle_col])
    angles = data_gdf[angle_col].to_numpy(dtype=float, copy=True)
    angles[angles == 360.0] = 0.0
    angle_sets = np.zeros_like(angles, dtype=int)

    for idx, (s1, e1, s2, e2) in enumerate(
        zip(p1_starts, p1_ends, p2_starts, p2_ends), start=1
    ):
        for start_a, end_a in ((s1, e1), (s2, e2)):
            if start_a > end_a:
                angle_sets[(angles >= start_a) & (angles <= 360)] = idx
                angle_sets[(angles >= 0) & (angles < end_a)] = idx
            else:
                angle_sets[(angles >= start_a) & (angles < end_a)] = idx

    data_gdf[out_angle_set_col] = angle_sets
    if out_vec_file is None:
        return data_gdf
    write_vec_gdf(
        data_gdf,
        out_vec_file,
        out_vec_lyr,
        out_format=out_format,
        out_compress=out_compress,
    )
    return None


def _split_wrapped_interval(start_ang: int, end_ang: int) -> List[Tuple[int, int]]:
    """Normalise an angle interval possibly crossing 0/360 into ``[0, 360]`` pairs."""
    if end_ang == 0:
        end_ang = 360
    if start_ang > 360 and end_ang > 360:
        start_ang -= 360
        end_ang -= 360
        if start_ang > 360 and end_ang > 360:
            raise rsgispy.RSGISPyException(
                f"Angle bounds ({start_ang}, {end_ang}) are out of range."
            )

    if start_ang >= 0 and end_ang <= 360:
        return [(start_ang, end_ang)]
    if start_ang < 0 and end_ang <= 360:
        return [(360 + start_ang, 360), (0, end_ang)]
    if start_ang >= 0 and end_ang > 360:
        return [(start_ang, 360), (0, end_ang - 360)]
    raise rsgispy.RSGISPyException(
        f"Invalid angle bounds ({start_ang}, {end_ang})."
    )


def create_orthogonal_angle_sets(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    angle_col: str,
    start_angle: int,
    angle_half_width: int,
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    *,
    out_format: str = "GPKG",
    out_angle_set_col: str = "angle_set",
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Create two orthogonal mirrored angle sets (labelled ``1`` and ``2``;
    outside angles receive ``0``).

    :param vec_file: Input vector file path or :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param angle_col: Column containing azimuth angles in ``[0, 360]`` degrees.
    :param start_angle: Central angle of set 1 in ``[0, 180]`` degrees.
    :param angle_half_width: Half-width of each angular sector in ``[0, 45]``
                             degrees.
    :param out_vec_file: Output vector file path, or ``None`` to return an
                         in-memory :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param out_angle_set_col: Output integer sector ID column
                              (Default = ``'angle_set'``).
    :param out_compress: Optional GeoParquet compression setting (``None`` to
                         infer from extension or write uncompressed, ``True``
                         for ``'snappy'``, ``False``/``'none'`` for uncompressed,
                         or codec name ``'snappy'``, ``'gzip'``, ``'zstd'``,
                         ``'brotli'``, ``'lz4'``).
    :return: Updated :class:`geopandas.GeoDataFrame` when ``out_vec_file`` is
             ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.vector.attrs

        rsgispy.vector.attrs.create_orthogonal_angle_sets(
            "pts.gpkg", "pts", "angle", 0, 20, "pts_orth.gpkg", "pts"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    if (start_angle < 0) or (start_angle > 180):
        raise rsgispy.RSGISPyException("start_angle must be between 0 and 180.")
    if (angle_half_width < 0) or (angle_half_width > 45):
        raise rsgispy.RSGISPyException(
            "angle_half_width must be between 0 and 45."
        )

    set1_ranges = _split_wrapped_interval(
        start_angle - angle_half_width, start_angle + angle_half_width
    ) + _split_wrapped_interval(
        start_angle + 180 - angle_half_width, start_angle + 180 + angle_half_width
    )

    orthog = start_angle + 90
    set2_ranges = _split_wrapped_interval(
        orthog - angle_half_width, orthog + angle_half_width
    ) + _split_wrapped_interval(
        orthog + 180 - angle_half_width, orthog + 180 + angle_half_width
    )

    data_gdf = read_vec_gdf(vec_file, vec_lyr)
    validate_columns_exist(data_gdf, [angle_col])

    angles = data_gdf[angle_col].to_numpy()
    angle_sets = np.zeros_like(angles, dtype=int)

    for low_a, high_a in set1_ranges:
        angle_sets[(angles >= low_a) & (angles <= high_a)] = 1
    for low_a, high_a in set2_ranges:
        angle_sets[(angles >= low_a) & (angles <= high_a)] = 2

    data_gdf[out_angle_set_col] = angle_sets
    if out_vec_file is None:
        return data_gdf
    write_vec_gdf(
        data_gdf,
        out_vec_file,
        out_vec_lyr,
        out_format=out_format,
        out_compress=out_compress,
    )
    return None


_MUPARSER_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_MUPARSER_RESERVED_CONSTS = {"_pi", "_e"}


def _series_to_float64(series: pd.Series, *, strict: bool) -> Optional[np.ndarray]:
    """Convert a pandas Series to a float64 NumPy array, or ``None`` if skipped."""
    if pd.api.types.is_numeric_dtype(series.dtype) or pd.api.types.is_bool_dtype(
        series.dtype
    ):
        return series.to_numpy(dtype=np.float64, na_value=np.nan)

    if pd.api.types.is_object_dtype(series.dtype):
        non_na = series.dropna()
        if all(
            isinstance(val, (int, float, np.number, bool, np.bool_))
            for val in non_na
        ):
            return series.to_numpy(dtype=np.float64, na_value=np.nan)

    if strict:
        try:
            numeric = pd.to_numeric(series, errors="raise")
            return numeric.to_numpy(dtype=np.float64, na_value=np.nan)
        except (TypeError, ValueError) as err:
            raise rsgispy.RSGISPyException(
                f"Column '{series.name}' is not numeric and cannot be used "
                f"in vector_maths: {err}"
            ) from err
    return None


def _collect_vector_maths_vars(
    base_gdf: gpd.GeoDataFrame,
    vec_cols: Optional[Union[Sequence[str], Mapping[str, str]]],
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """Collect ``{var_name: float64_array}`` bindings and skipped column names."""
    geom_col = base_gdf.geometry.name
    var_arrays: Dict[str, np.ndarray] = {}
    skipped_cols: List[str] = []

    if vec_cols is None:
        for col in base_gdf.columns:
            if col == geom_col:
                continue
            col_name = str(col)
            if (
                not _MUPARSER_IDENT_RE.match(col_name)
                or col_name in _MUPARSER_RESERVED_CONSTS
            ):
                skipped_cols.append(col_name)
                continue
            arr = _series_to_float64(base_gdf[col], strict=False)
            if arr is None:
                skipped_cols.append(col_name)
            else:
                var_arrays[col_name] = arr
        return var_arrays, skipped_cols

    if isinstance(vec_cols, Mapping):
        col_map = {str(var_name): str(col_name) for var_name, col_name in vec_cols.items()}
        for var_name in col_map:
            if (
                not _MUPARSER_IDENT_RE.match(var_name)
                or var_name in _MUPARSER_RESERVED_CONSTS
            ):
                raise rsgispy.RSGISPyException(
                    f"Invalid expression variable name '{var_name}' in vec_cols."
                )
        validate_columns_exist(base_gdf, list(col_map.values()))
        for var_name, col_name in col_map.items():
            arr = _series_to_float64(base_gdf[col_name], strict=True)
            if arr is not None:
                var_arrays[var_name] = arr
        return var_arrays, skipped_cols

    if isinstance(vec_cols, str):
        col_names = [vec_cols]
    elif isinstance(vec_cols, Sequence) and not isinstance(vec_cols, bytes):
        col_names = [str(col) for col in vec_cols]
    else:
        raise rsgispy.RSGISPyException(
            "vec_cols must be None, a sequence of column names, or a mapping "
            "of {var_name: col_name}."
        )

    validate_columns_exist(base_gdf, col_names)
    for col_name in col_names:
        if (
            not _MUPARSER_IDENT_RE.match(col_name)
            or col_name in _MUPARSER_RESERVED_CONSTS
        ):
            raise rsgispy.RSGISPyException(
                f"Column name '{col_name}' is not a valid muParser identifier; "
                "pass vec_cols={'var_name': col_name} to alias it."
            )
        arr = _series_to_float64(base_gdf[col_name], strict=True)
        if arr is not None:
            var_arrays[col_name] = arr
    return var_arrays, skipped_cols


def _cast_vector_maths_result(
    result_arr: np.ndarray,
    out_dtype: Optional[Any],
) -> Any:
    """Cast evaluated float64 results according to ``out_dtype``."""
    if out_dtype is None:
        return result_arr

    is_np_dtype = isinstance(out_dtype, np.dtype) or (
        isinstance(out_dtype, type) and issubclass(out_dtype, np.generic)
    )
    kind = _resolve_cast_kind(out_dtype)
    if kind is None:
        raise rsgispy.RSGISPyException(
            f"Unsupported out_dtype '{out_dtype}' for vector_maths."
        )

    if kind == "float":
        if is_np_dtype and np.issubdtype(np.dtype(out_dtype), np.floating):
            return result_arr.astype(np.dtype(out_dtype))
        return result_arr.astype(np.float64)

    if kind == "int":
        has_non_finite = bool(result_arr.size > 0 and not np.all(np.isfinite(result_arr)))
        if has_non_finite:
            return pd.array(
                [pd.NA if not np.isfinite(val) else int(val) for val in result_arr],
                dtype="Int64",
            )
        if is_np_dtype and np.issubdtype(np.dtype(out_dtype), np.integer):
            return result_arr.astype(np.dtype(out_dtype))
        return result_arr.astype(np.int64)

    if kind == "bool":
        has_nan = bool(result_arr.size > 0 and np.any(np.isnan(result_arr)))
        if has_nan:
            return pd.array(
                [pd.NA if np.isnan(val) else bool(val) for val in result_arr],
                dtype="boolean",
            )
        return result_arr.astype(bool)

    return cast_col_data(result_arr, str)


def _normalise_vector_maths_out_cols(
    out_col: Union[str, Sequence[str]],
) -> List[str]:
    """Validate and normalise ``out_col`` into a list of unique column names."""
    if isinstance(out_col, str):
        if not out_col.strip():
            raise rsgispy.RSGISPyException(
                "Output column 'out_col' must be a non-empty string or "
                "sequence of non-empty strings."
            )
        return [out_col]

    if isinstance(out_col, Sequence) and not isinstance(
        out_col, (bytes, bytearray)
    ):
        if not out_col:
            raise rsgispy.RSGISPyException(
                "Output column 'out_col' must be a non-empty string or "
                "sequence of non-empty strings."
            )
        out_cols: List[str] = []
        seen: set = set()
        for col in out_col:
            if not isinstance(col, str) or not col.strip():
                raise rsgispy.RSGISPyException(
                    "Each entry in 'out_col' must be a non-empty string."
                )
            if col in seen:
                raise rsgispy.RSGISPyException(
                    f"Duplicate output column name '{col}' in 'out_col'."
                )
            seen.add(col)
            out_cols.append(col)
        return out_cols

    raise rsgispy.RSGISPyException(
        "Output column 'out_col' must be a non-empty string or "
        "sequence of non-empty strings."
    )


def _resolve_vector_maths_out_dtypes(
    out_dtype: Optional[Any],
    out_cols: Sequence[str],
) -> List[Optional[Any]]:
    """Resolve ``out_dtype`` into one dtype specification per output column."""
    if out_dtype is None:
        return [None] * len(out_cols)
    if isinstance(out_dtype, Mapping):
        unknown = [key for key in out_dtype if key not in out_cols]
        if unknown:
            raise rsgispy.RSGISPyException(
                f"Unknown column name(s) in out_dtype mapping: {unknown}. "
                f"Expected keys from {list(out_cols)}."
            )
        return [out_dtype.get(col, None) for col in out_cols]
    if isinstance(out_dtype, (list, tuple)):
        if len(out_dtype) != len(out_cols):
            raise rsgispy.RSGISPyException(
                f"Length of out_dtype ({len(out_dtype)}) must match the "
                f"number of output columns ({len(out_cols)})."
            )
        return list(out_dtype)
    return [out_dtype] * len(out_cols)


def _eval_vector_maths_expr(
    exp: str,
    var_arrays: Mapping[str, np.ndarray],
    skipped_cols: Sequence[str],
) -> List[Any]:
    """Compile and evaluate a single- or multi-output muParser expression."""
    parser = pymu_eval.Parser()
    if "pi" not in var_arrays:
        parser.define_const("pi", pymu_eval.MathImpl.CONST_PI)
    if "e" not in var_arrays:
        parser.define_const("e", pymu_eval.MathImpl.CONST_E)

    for var_name, arr in var_arrays.items():
        try:
            parser.define_var(var_name, arr)
        except pymu_eval.ParserError as err:
            raise rsgispy.RSGISPyException(
                f"Invalid column variable name '{var_name}': {err}"
            ) from err

    rsgispy.tools.maths.configure_muparser_assignment_vars(parser, exp)

    try:
        parser.set_expr(exp)
        with np.errstate(all="ignore"):
            return parser.eval_multi()
    except pymu_eval.ParserError as err:
        hint = ""
        if skipped_cols:
            used_skipped = [
                col for col in skipped_cols if re.search(rf"\b{re.escape(col)}\b", exp)
            ]
            if used_skipped:
                hint = (
                    f" Note that column(s) {used_skipped} could not be bound "
                    "automatically (non-numeric or non-identifier name)."
                )
        raise rsgispy.RSGISPyException(
            f"Error parsing or evaluating expression '{exp}': {err}{hint}"
        ) from err


def vector_maths(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    out_col: Union[str, Sequence[str]],
    exp: str,
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    *,
    out_format: str = "GPKG",
    vec_cols: Optional[Union[Sequence[str], Mapping[str, str]]] = None,
    out_dtype: Optional[Any] = None,
    del_exist_vec: bool = False,
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Calculate one or more new (or overwrite existing) vector attribute columns
    from existing columns using a ``pymu-eval`` (muParser) expression.

    When ``vec_cols`` is ``None`` (the default), all numeric and boolean
    attribute columns whose names are valid muParser identifiers
    (``[A-Za-z_][A-Za-z0-9_]*``) are bound automatically. Use ``vec_cols`` as
    a sequence of column names to restrict the bound columns, or as a mapping
    ``{var_name: col_name}`` to alias column names containing spaces, hyphens,
    or other characters not permitted in muParser identifiers.

    To calculate multiple output columns in a single expression, pass a
    sequence of column names to ``out_col`` and separate the sub-expressions
    in ``exp`` with top-level commas (e.g., ``"expr1, expr2, expr3"``). Each
    sub-expression is assigned positionally to the corresponding column in
    ``out_col``. Commas inside function calls (such as ``min(b1, b2)`` or
    ``avg(b1, b2, b3)``) remain function argument separators and do not start
    a new output column. Sub-expressions may also span multiple lines, use
    ternary conditionals (``cond ? val_true : val_false``), re-assign a bound
    variable, or define arbitrary new intermediate variables (e.g.,
    ``ndvi = (nir - red) / (nir + red + 1e-6)``) for reuse by subsequent
    sub-expressions. Note that a top-level assignment sub-expression both
    binds the variable and writes that value to its positional column in
    ``out_col``; to define an intermediate without adding an output column,
    assign inside parentheses (e.g.,
    ``"(ndvi = (nir - red) / (nir + red + 1e-6)) >= 0.5 ? 1 : 0"``).

    :param vec_file: Input vector file path or :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param out_col: Output attribute column name (``str``) for a single-output
                    expression, or a sequence of unique column names matching
                    the number of comma-separated outputs in ``exp``.
    :param exp: Mathematical expression string in muParser syntax
                (e.g., ``'(nir - red) / (nir + red)'``,
                ``'val > 0 ? sqrt(val) : 0'``, or a comma-separated
                multi-output expression such as
                ``'ndvi = (nir - red) / (nir + red + 1e-6), ndvi >= 0.5 ? 1 : 0'``).
    :param out_vec_file: Output vector file path, or ``None`` to return an
                         in-memory :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param vec_cols: Optional sequence of column names to expose, or mapping
                     ``{var_name: col_name}`` binding expression variable names
                     to column names. If ``None``, all eligible numeric columns
                     are exposed by name.
    :param out_dtype: Optional output column type specification: ``None`` for
                      ``float64``, a single type (e.g., ``float``, ``int``,
                      ``bool``, ``np.float32``) applied to all outputs, a
                      sequence of per-output types matching ``out_col``, or a
                      mapping ``{col_name: dtype}`` for selected output columns.
    :param del_exist_vec: If ``True``, delete ``out_vec_file`` if it exists
                          (Default = ``False``).
    :param out_compress: Optional GeoParquet compression setting (``None`` to
                         infer from extension or write uncompressed, ``True``
                         for ``'snappy'``, ``False``/``'none'`` for uncompressed,
                         or codec name ``'snappy'``, ``'gzip'``, ``'zstd'``,
                         ``'brotli'``, ``'lz4'``).
    :return: Updated :class:`geopandas.GeoDataFrame` when ``out_vec_file`` is
             ``None``, otherwise ``None``.

    .. code:: python

        import numpy as np
        import rsgispy.vector.attrs

        # Single output column
        gdf = rsgispy.vector.attrs.vector_maths(
            "polys.gpkg",
            "polys",
            out_col="ndvi",
            exp="(nir - red) / (nir + red)",
        )

        # Multiple output columns defining and reusing an intermediate variable
        multi_exp = (
            "ndvi = (nir - red) / (nir + red + 1e-6), "
            "(nir + red) / 2, "
            "ndvi >= 0.5 ? 1 : 0"
        )
        gdf = rsgispy.vector.attrs.vector_maths(
            "polys.gpkg",
            "polys",
            out_col=["ndvi", "mean_nr", "is_veg"],
            exp=multi_exp,
            out_dtype=[np.float32, np.float32, int],
        )

        # Multi-argument functions (inner commas are function arguments;
        # top-level commas produce separate columns)
        gdf = rsgispy.vector.attrs.vector_maths(
            "polys.gpkg",
            "polys",
            out_col=["band_min", "band_max", "band_avg"],
            exp="min(blue, green, red), max(blue, green, red), avg(blue, green, red)",
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    out_cols = _normalise_vector_maths_out_cols(out_col)
    col_dtypes = _resolve_vector_maths_out_dtypes(out_dtype, out_cols)
    if not exp or not isinstance(exp, str) or not exp.strip():
        raise rsgispy.RSGISPyException(
            "Expression 'exp' must be a non-empty string."
        )

    base_gdf = read_vec_gdf(vec_file, vec_lyr)
    geom_col = base_gdf.geometry.name
    if geom_col in out_cols:
        raise rsgispy.RSGISPyException(
            f"Output column '{geom_col}' cannot overwrite the geometry column."
        )

    var_arrays, skipped_cols = _collect_vector_maths_vars(base_gdf, vec_cols)
    raw_results = _eval_vector_maths_expr(exp, var_arrays, skipped_cols)

    if len(raw_results) != len(out_cols):
        raise rsgispy.RSGISPyException(
            f"Expression '{exp}' produced {len(raw_results)} output(s), "
            f"but {len(out_cols)} output column name(s) were provided "
            f"({out_cols})."
        )

    n_rows = len(base_gdf)
    for col_name, col_dtype, raw_result in zip(out_cols, col_dtypes, raw_results):
        if np.ndim(raw_result) == 0:
            result_arr = np.full(n_rows, float(raw_result), dtype=np.float64)
        else:
            result_arr = np.asarray(raw_result, dtype=np.float64)
            if result_arr.shape != (n_rows,):
                raise rsgispy.RSGISPyException(
                    f"Evaluated result shape {result_arr.shape} for column "
                    f"'{col_name}' does not match feature count ({n_rows})."
                )
        base_gdf[col_name] = _cast_vector_maths_result(result_arr, col_dtype)

    if out_vec_file is None:
        return base_gdf
    write_vec_gdf(
        base_gdf,
        out_vec_file,
        out_vec_lyr,
        out_format=out_format,
        overwrite=del_exist_vec,
        out_compress=out_compress,
    )
    return None
