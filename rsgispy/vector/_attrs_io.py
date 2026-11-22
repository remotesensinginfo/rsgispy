"""
The vector.attrs module provides reading, writing, sorting, renaming,
dropping, and tabular export functions for vector attribute tables using
``geopandas``.
"""

import os
from typing import Any, Dict, List, Optional, Sequence, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio

try:
    import pyarrow.parquet as pq

    HAVE_PYARROW = True
except ImportError:
    pq = None
    HAVE_PYARROW = False

import rsgispy
import rsgispy.tools.stats
from rsgispy.vectorprocess import (
    VectorInput,
    cast_col_data,
    infer_parquet_compression,
    is_geoparquet_file,
    read_vec_gdf,
    read_vec_info,
    to_python_scalar,
    validate_columns_exist,
    write_vec_gdf,
)

__all__ = [
    "write_vec_column",
    "write_vec_column_to_layer",
    "read_vec_column",
    "read_vec_columns",
    "drop_vec_cols",
    "rename_vec_cols",
    "sort_vec_lyr",
    "get_unq_col_values",
    "get_vec_cols_as_array",
    "export_vec_attrs_to_csv",
    "export_vec_attrs_to_excel",
    "export_vec_attrs_to_parquet",
]


def _read_vec_attrs_df(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Read attribute columns from a vector file or GeoDataFrame without loading
    geometries for file inputs.
    """
    if isinstance(vec_file, gpd.GeoDataFrame):
        if columns is not None:
            req_cols = list(columns)
            validate_columns_exist(vec_file, req_cols)
            return pd.DataFrame(vec_file[req_cols])
        geom_col = vec_file.geometry.name
        if geom_col in vec_file.columns:
            return pd.DataFrame(vec_file.drop(columns=[geom_col]))
        return pd.DataFrame(vec_file.copy())

    if not isinstance(vec_file, (str, os.PathLike)):
        raise rsgispy.RSGISPyException(
            "Input must be a geopandas.GeoDataFrame or a vector file path."
        )

    vec_path = os.fspath(vec_file)
    info = read_vec_info(vec_path, vec_lyr)
    raw_fields = info.get("fields")
    avail_cols = [] if raw_fields is None else [str(col) for col in raw_fields]

    if columns is not None:
        req_cols = list(columns)
        missing = [col for col in req_cols if col not in avail_cols]
        if missing:
            raise rsgispy.RSGISPyException(
                f"Column(s) not found in vector layer: {missing}"
            )
    else:
        req_cols = avail_cols

    if is_geoparquet_file(vec_path):
        if not HAVE_PYARROW or pq is None:
            raise rsgispy.RSGISPyException(
                "Reading GeoParquet attributes requires 'pyarrow'."
            )
        if not req_cols:
            return pd.DataFrame(index=pd.RangeIndex(int(info.get("features", 0))))
        return pq.read_table(vec_path, columns=req_cols).to_pandas()

    return pyogrio.read_dataframe(
        vec_path,
        layer=info["layer_name"],
        read_geometry=False,
        columns=req_cols,
    )


def write_vec_column_to_layer(
    out_vec_lyr_obj: VectorInput,
    att_column: str,
    att_col_datatype: Any,
    att_col_data: Sequence[Any],
    vec_lyr: Optional[str] = None,
    out_format: Optional[str] = None,
    out_compress: Optional[Union[str, bool]] = None,
) -> gpd.GeoDataFrame:
    """
    Write or overwrite an attribute column on an in-memory
    :class:`geopandas.GeoDataFrame` or a vector file on disk.

    :param out_vec_lyr_obj: Target :class:`geopandas.GeoDataFrame` or vector
                            file path to update.
    :param att_column: Name of the output attribute column.
    :param att_col_datatype: Target column type (e.g., ``int``, ``float``,
                             ``str``, ``'int'``, ``'float'``, ``'str'``, or
                             ``None`` to preserve input values).
    :param att_col_data: Sequence of values matching the feature count.
    :param vec_lyr: Layer name when ``out_vec_lyr_obj`` is a vector file path.
    :param out_format: Output format driver when writing back to a file path
                       (inferred from extension if ``None``).
    :param out_compress: Optional GeoParquet compression setting when writing
                         to a file path.
    :return: Updated :class:`geopandas.GeoDataFrame`.

    .. code:: python

        import geopandas as gpd
        import rsgispy.vector.attrs

        gdf = gpd.read_file("polys.gpkg", layer="polys")
        rsgispy.vector.attrs.write_vec_column_to_layer(
            gdf, "score", float, [0.5] * len(gdf)
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    if out_vec_lyr_obj is None:
        raise rsgispy.RSGISPyException("The layer passed in is None.")

    is_gdf = isinstance(out_vec_lyr_obj, gpd.GeoDataFrame)
    if is_gdf:
        gdf = out_vec_lyr_obj
    elif isinstance(out_vec_lyr_obj, (str, os.PathLike)):
        gdf = read_vec_gdf(out_vec_lyr_obj, vec_lyr=vec_lyr, copy=True)
    else:
        raise rsgispy.RSGISPyException(
            "out_vec_lyr_obj must be a geopandas.GeoDataFrame or a vector file path."
        )

    num_feats = len(gdf)
    if len(att_col_data) != num_feats:
        raise rsgispy.RSGISPyException(
            f"The number of features ({num_feats}) and length of the input "
            f"data ({len(att_col_data)}) are not equal."
        )

    target_col = att_column
    for existing in gdf.columns:
        if existing.lower() == att_column.lower():
            target_col = existing
            break

    gdf[target_col] = cast_col_data(att_col_data, att_col_datatype)
    if not is_gdf:
        write_vec_gdf(
            gdf,
            out_vec_file=out_vec_lyr_obj,  # type: ignore[arg-type]
            out_vec_lyr=vec_lyr,
            out_format=out_format,
            overwrite=False,
            replace_layer=True,
            out_compress=out_compress,
        )
    return gdf


def write_vec_column(
    out_vec_file: VectorInput,
    out_vec_lyr: Optional[str],
    att_column: str,
    att_col_datatype: Any,
    att_col_data: Sequence[Any],
    out_format: Optional[str] = None,
    out_compress: Optional[Union[str, bool]] = None,
    dest_vec_file: Optional[Union[str, os.PathLike]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Write or update an attribute column in an existing vector file or
    :class:`geopandas.GeoDataFrame` using ``geopandas``.

    When ``out_vec_file`` is an in-memory :class:`geopandas.GeoDataFrame` and
    ``dest_vec_file`` is ``None``, a copy with the added/updated column is
    returned without writing to disk.

    :param out_vec_file: Path to the vector file to update, or input
                         :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Layer name within ``out_vec_file`` (or ``None``).
    :param att_column: Name of the output attribute column.
    :param att_col_datatype: Target column type (e.g., ``int``, ``float``,
                             ``str``, ``'int'``, ``'float'``, ``'str'``, or
                             ``None`` to preserve input values).
    :param att_col_data: Sequence of values matching the feature count.
    :param out_format: Optional output driver name (inferred from file
                       extension when ``None``).
    :param out_compress: Optional GeoParquet compression setting (``None`` to
                         infer from extension or write uncompressed, ``True``
                         for ``'snappy'``, ``False``/``'none'`` for uncompressed,
                         or codec name ``'snappy'``, ``'gzip'``, ``'zstd'``,
                         ``'brotli'``, ``'lz4'``).
    :param dest_vec_file: Optional separate output file path.
    :return: Updated :class:`geopandas.GeoDataFrame` when ``out_vec_file`` is a
             ``GeoDataFrame`` and ``dest_vec_file`` is ``None``, otherwise
             ``None``.

    .. code:: python

        import rsgispy.vector.attrs

        rsgispy.vector.attrs.write_vec_column(
            "polys.gpkg", "polys", "class_id", int, [1, 2, 3]
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    is_gdf_input = isinstance(out_vec_file, gpd.GeoDataFrame)
    gdf = read_vec_gdf(out_vec_file, out_vec_lyr)
    write_vec_column_to_layer(gdf, att_column, att_col_datatype, att_col_data)
    if dest_vec_file is not None:
        write_vec_gdf(
            gdf,
            dest_vec_file,
            out_vec_lyr,
            out_format=out_format,
            out_compress=out_compress,
        )
        return None
    if is_gdf_input:
        return gdf
    write_vec_gdf(
        gdf,
        out_vec_file,  # type: ignore[arg-type]
        out_vec_lyr,
        out_format=out_format,
        replace_layer=True,
        out_compress=out_compress,
    )
    return None


def read_vec_column(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    att_column: str,
) -> List[Any]:
    """
    Read a single attribute column from a vector file or
    :class:`geopandas.GeoDataFrame` into a Python list.

    :param vec_file: Input vector file path or :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param att_column: Name of the attribute column to read.
    :return: List of column values in feature order.

    .. code:: python

        import rsgispy.vector.attrs

        vals = rsgispy.vector.attrs.read_vec_column(
            "polys.gpkg", "polys", "class_name"
        )
    """
    df = _read_vec_attrs_df(vec_file, vec_lyr, columns=[att_column])
    return [to_python_scalar(v) for v in df[att_column].tolist()]


def read_vec_columns(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    att_columns: Sequence[str],
) -> List[Dict[str, Any]]:
    """
    Read multiple attribute columns from a vector file or
    :class:`geopandas.GeoDataFrame` as a list of dicts.

    :param vec_file: Input vector file path or :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param att_columns: Sequence of attribute column names to read.
    :return: List of dictionaries keyed by column name, one per feature.

    .. code:: python

        import rsgispy.vector.attrs

        rows = rsgispy.vector.attrs.read_vec_columns(
            "polys.gpkg", "polys", ["id", "class_name"]
        )
    """
    sub_df = _read_vec_attrs_df(vec_file, vec_lyr, columns=att_columns)
    records: List[Dict[str, Any]] = []
    for row in sub_df.to_dict(orient="records"):
        records.append({str(k): to_python_scalar(v) for k, v in row.items()})
    return records


def drop_vec_cols(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    drop_cols: Sequence[str],
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    out_format: str = "GPKG",
    chk_cols_present: bool = True,
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Remove specified columns from a vector layer and write a new vector file
    or return an updated :class:`geopandas.GeoDataFrame`.

    :param vec_file: Input vector file path or :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param drop_cols: Sequence of column names to drop.
    :param out_vec_file: Output vector file path, or ``None`` to return an
                         in-memory :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param chk_cols_present: If ``True``, silently ignore column names not
                             present in the layer (Default = ``True``).
    :param out_compress: Optional GeoParquet compression setting (``None`` to
                         infer from extension or write uncompressed, ``True``
                         for ``'snappy'``, ``False``/``'none'`` for uncompressed,
                         or codec name ``'snappy'``, ``'gzip'``, ``'zstd'``,
                         ``'brotli'``, ``'lz4'``).
    :return: Updated :class:`geopandas.GeoDataFrame` when ``out_vec_file`` is
             ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.vector.attrs

        rsgispy.vector.attrs.drop_vec_cols(
            "in.gpkg", "polys", ["tmp_col"], "out.gpkg", "polys"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    data_gdf = read_vec_gdf(vec_file, vec_lyr)

    if chk_cols_present:
        drop_set = set(drop_cols)
        cols_to_drop = [c for c in data_gdf.columns if c in drop_set]
    else:
        cols_to_drop = list(drop_cols)
        validate_columns_exist(data_gdf, cols_to_drop)

    if cols_to_drop:
        data_gdf = data_gdf.drop(columns=cols_to_drop)

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


def rename_vec_cols(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    rname_cols_lut: Dict[str, str],
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    out_format: str = "GPKG",
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Rename columns in a vector layer and write a new vector file or return
    an updated :class:`geopandas.GeoDataFrame`.

    :param vec_file: Input vector file path or :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param rname_cols_lut: Mapping ``{"old_name": "new_name"}``.
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

        rsgispy.vector.attrs.rename_vec_cols(
            "in.gpkg", "polys", {"old_c": "new_c"}, "out.gpkg", "polys"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    data_gdf = read_vec_gdf(vec_file, vec_lyr)
    data_gdf = data_gdf.rename(columns=rname_cols_lut)
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


def sort_vec_lyr(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    out_vec_file: Optional[Union[str, os.PathLike]],
    out_vec_lyr: Optional[str],
    sort_by: Union[str, List[str]],
    ascending: Union[bool, List[bool]],
    out_format: str = "GPKG",
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Sort features in a vector layer by one or more attribute columns.

    :param vec_file: Input vector file path or :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param out_vec_file: Output vector file path, or ``None`` to return an
                         in-memory :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name.
    :param sort_by: Single column name or list of column names to sort by.
    :param ascending: Boolean or list of booleans matching ``sort_by``
                      (``True`` = ascending, ``False`` = descending).
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param out_compress: Optional GeoParquet compression setting (``None`` to
                         infer from extension or write uncompressed, ``True``
                         for ``'snappy'``, ``False``/``'none'`` for uncompressed,
                         or codec name ``'snappy'``, ``'gzip'``, ``'zstd'``,
                         ``'brotli'``, ``'lz4'``).
    :return: Sorted :class:`geopandas.GeoDataFrame` when ``out_vec_file`` is
             ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.vector.attrs

        rsgispy.vector.attrs.sort_vec_lyr(
            "in.gpkg", "polys", "sorted.gpkg", "polys", "area", False
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    if isinstance(sort_by, list):
        if not isinstance(ascending, list):
            raise rsgispy.RSGISPyException(
                "If sort_by is a list then ascending must be too."
            )
        if len(sort_by) != len(ascending):
            raise rsgispy.RSGISPyException(
                "If lists, the length of sort_by and ascending must be the same."
            )

    base_gdf = read_vec_gdf(vec_file, vec_lyr, copy=False)
    cols_to_check = [sort_by] if isinstance(sort_by, str) else sort_by
    validate_columns_exist(base_gdf, cols_to_check)

    sorted_gdf = base_gdf.sort_values(by=sort_by, ascending=ascending).copy()
    if out_vec_file is None:
        return sorted_gdf
    write_vec_gdf(
        sorted_gdf,
        out_vec_file,
        out_vec_lyr,
        out_format=out_format,
        out_compress=out_compress,
    )
    return None


def get_unq_col_values(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    col_name: str,
) -> np.ndarray:
    """
    Return the unique values within an attribute column.

    :param vec_file: Input vector file path or :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param col_name: Name of the column to inspect.
    :return: NumPy array of unique values in ``col_name``.

    .. code:: python

        import rsgispy.vector.attrs

        classes = rsgispy.vector.attrs.get_unq_col_values(
            "polys.gpkg", "polys", "landcover"
        )
    """
    df = _read_vec_attrs_df(vec_file, vec_lyr, columns=[col_name])
    return np.asarray(df[col_name].unique())


def get_vec_cols_as_array(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    cols: Sequence[str],
    lower_limit: Optional[float] = None,
    upper_limit: Optional[float] = None,
) -> np.ndarray:
    """
    Read numeric attribute columns from a vector layer into a 2D ``float``
    NumPy array, optionally filtering out rows outside ``(lower_limit,
    upper_limit)`` or containing non-finite values.

    :param vec_file: Input vector file path or :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param cols: Sequence of column names to read.
    :param lower_limit: Optional exclusive lower bound applied across all columns.
    :param upper_limit: Optional exclusive upper bound applied across all columns.
    :return: 2D NumPy float array of shape ``(n_valid_rows, len(cols))``.

    .. code:: python

        import rsgispy.vector.attrs

        arr = rsgispy.vector.attrs.get_vec_cols_as_array(
            "samples.gpkg", "samples", ["b1", "b2"], lower_limit=0.0
        )
    """
    df = _read_vec_attrs_df(vec_file, vec_lyr, columns=cols)
    out_arr = df.loc[:, list(cols)].to_numpy(dtype=float)
    return rsgispy.tools.stats.mask_data_to_valid(
        out_arr, lower_limit=lower_limit, upper_limit=upper_limit
    )


def export_vec_attrs_to_csv(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    output_file: Union[str, os.PathLike],
) -> None:
    """
    Export the attribute table of a vector layer (excluding geometry) to CSV.

    :param vec_file: Input vector file path or :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param output_file: Output CSV file path.

    .. code:: python

        import rsgispy.vector.attrs

        rsgispy.vector.attrs.export_vec_attrs_to_csv(
            "polys.gpkg", "polys", "attrs.csv"
        )
    """
    data_df = _read_vec_attrs_df(vec_file, vec_lyr)
    data_df.to_csv(os.fspath(output_file))


def export_vec_attrs_to_excel(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    output_file: Union[str, os.PathLike],
    out_sheet_name: str = "Sheet1",
) -> None:
    """
    Export the attribute table of a vector layer (excluding geometry) to an
    Excel ``.xlsx`` workbook.

    :param vec_file: Input vector file path or :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param output_file: Output Excel file path.
    :param out_sheet_name: Worksheet name (Default = ``'Sheet1'``).

    .. code:: python

        import rsgispy.vector.attrs

        rsgispy.vector.attrs.export_vec_attrs_to_excel(
            "polys.gpkg", "polys", "attrs.xlsx"
        )
    """
    out_path = os.fspath(output_file)
    data_df = _read_vec_attrs_df(vec_file, vec_lyr)
    try:
        data_df.to_excel(
            out_path, sheet_name=out_sheet_name, engine="xlsxwriter"
        )
    except ImportError:
        try:
            data_df.to_excel(
                out_path, sheet_name=out_sheet_name, engine="openpyxl"
            )
        except ImportError as err:
            raise rsgispy.RSGISPyException(
                "Writing Excel files requires 'xlsxwriter' or 'openpyxl'."
            ) from err


def export_vec_attrs_to_parquet(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    output_file: Union[str, os.PathLike],
    gzip_output: bool = True,
    out_compress: Optional[Union[str, bool]] = None,
) -> None:
    """
    Export the attribute table of a vector layer (excluding geometry) to a
    Parquet file.

    :param vec_file: Input vector file path or :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param output_file: Output Parquet file path.
    :param gzip_output: If ``True`` and ``out_compress`` is ``None``, compress
                        using ``gzip`` (Default = ``True``).
    :param out_compress: Optional Parquet compression setting (overrides
                         ``gzip_output`` when provided: ``True`` for ``'snappy'``,
                         ``False``/``'none'`` for uncompressed, or codec name
                         ``'snappy'``, ``'gzip'``, ``'zstd'``, ``'brotli'``,
                         ``'lz4'``).

    .. code:: python

        import rsgispy.vector.attrs

        rsgispy.vector.attrs.export_vec_attrs_to_parquet(
            "polys.gpkg", "polys", "attrs.parquet"
        )
    """
    out_path = os.fspath(output_file)
    data_df = _read_vec_attrs_df(vec_file, vec_lyr)
    if out_compress is not None:
        comp = infer_parquet_compression(out_path, out_compress=out_compress)
    elif gzip_output:
        comp = "gzip"
    else:
        comp = infer_parquet_compression(out_path, out_compress=None)
    try:
        data_df.to_parquet(out_path, compression=comp)
    except ImportError as err:
        raise rsgispy.RSGISPyException(
            "Writing Parquet files requires 'pyarrow' or 'fastparquet'."
        ) from err
