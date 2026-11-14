"""
The vectorattrs.io module provides reading, writing, sorting, renaming,
dropping, and tabular export functions for vector attribute tables using
``geopandas``.
"""

from typing import Any, Dict, List, Optional, Sequence, Union

import geopandas
import numpy

import rsgispy
import rsgispy.tools.stats
from rsgispy.vectorprocess import (
    cast_col_data,
    read_vec_gdf,
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


def write_vec_column_to_layer(
    out_vec_lyr_obj: geopandas.GeoDataFrame,
    att_column: str,
    att_col_datatype: Any,
    att_col_data: Sequence[Any],
) -> geopandas.GeoDataFrame:
    """
    Write or overwrite an attribute column on an in-memory
    :class:`geopandas.GeoDataFrame`.

    :param out_vec_lyr_obj: Target :class:`geopandas.GeoDataFrame` to update.
    :param att_column: Name of the output attribute column.
    :param att_col_datatype: Target column type (e.g., ``int``, ``float``,
                             ``str``, ``'int'``, ``'float'``, ``'str'``, or
                             ``None`` to preserve input values).
    :param att_col_data: Sequence of values matching ``len(out_vec_lyr_obj)``.
    :return: Updated :class:`geopandas.GeoDataFrame`.

    .. code:: python

        import geopandas
        import rsgispy.vectorattrs.io

        gdf = geopandas.read_file("polys.gpkg", layer="polys")
        rsgispy.vectorattrs.io.write_vec_column_to_layer(
            gdf, "score", float, [0.5] * len(gdf)
        )
    """
    if out_vec_lyr_obj is None:
        raise rsgispy.RSGISPyException("The layer passed in is None.")
    if not isinstance(out_vec_lyr_obj, geopandas.GeoDataFrame):
        raise rsgispy.RSGISPyException(
            "out_vec_lyr_obj must be a geopandas.GeoDataFrame."
        )

    num_feats = len(out_vec_lyr_obj)
    if len(att_col_data) != num_feats:
        raise rsgispy.RSGISPyException(
            f"The number of features ({num_feats}) and length of the input "
            f"data ({len(att_col_data)}) are not equal."
        )

    target_col = att_column
    for existing in out_vec_lyr_obj.columns:
        if existing.lower() == att_column.lower():
            target_col = existing
            break

    out_vec_lyr_obj[target_col] = cast_col_data(att_col_data, att_col_datatype)
    return out_vec_lyr_obj


def write_vec_column(
    out_vec_file: str,
    out_vec_lyr: Optional[str],
    att_column: str,
    att_col_datatype: Any,
    att_col_data: Sequence[Any],
    out_format: Optional[str] = None,
) -> None:
    """
    Write or update an attribute column in an existing vector file using
    ``geopandas``.

    :param out_vec_file: Path to the vector file to update.
    :param out_vec_lyr: Layer name within ``out_vec_file`` (or ``None``).
    :param att_column: Name of the output attribute column.
    :param att_col_datatype: Target column type (e.g., ``int``, ``float``,
                             ``str``, ``'int'``, ``'float'``, ``'str'``, or
                             ``None`` to preserve input values).
    :param att_col_data: Sequence of values matching the feature count.
    :param out_format: Optional output driver name (inferred from file
                       extension when ``None``).

    .. code:: python

        import rsgispy.vectorattrs.io

        rsgispy.vectorattrs.io.write_vec_column(
            "polys.gpkg", "polys", "class_id", int, [1, 2, 3]
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    gdf = read_vec_gdf(out_vec_file, out_vec_lyr)
    write_vec_column_to_layer(gdf, att_column, att_col_datatype, att_col_data)
    write_vec_gdf(gdf, out_vec_file, out_vec_lyr, out_format=out_format)


def read_vec_column(
    vec_file: str,
    vec_lyr: Optional[str],
    att_column: str,
) -> List[Any]:
    """
    Read a single attribute column from a vector file into a Python list.

    :param vec_file: Path to the input vector file.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param att_column: Name of the attribute column to read.
    :return: List of column values in feature order.

    .. code:: python

        import rsgispy.vectorattrs.io

        vals = rsgispy.vectorattrs.io.read_vec_column(
            "polys.gpkg", "polys", "class_name"
        )
    """
    gdf = read_vec_gdf(vec_file, vec_lyr)
    if att_column not in gdf.columns:
        raise rsgispy.RSGISPyException(
            f"The specified column '{att_column}' does not exist in the input "
            "layer; check case as drivers may be case-sensitive."
        )
    return [to_python_scalar(v) for v in gdf[att_column].tolist()]


def read_vec_columns(
    vec_file: str,
    vec_lyr: Optional[str],
    att_columns: Sequence[str],
) -> List[Dict[str, Any]]:
    """
    Read multiple attribute columns from a vector file as a list of dicts.

    :param vec_file: Path to the input vector file.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param att_columns: Sequence of attribute column names to read.
    :return: List of dictionaries keyed by column name, one per feature.

    .. code:: python

        import rsgispy.vectorattrs.io

        rows = rsgispy.vectorattrs.io.read_vec_columns(
            "polys.gpkg", "polys", ["id", "class_name"]
        )
    """
    gdf = read_vec_gdf(vec_file, vec_lyr)
    validate_columns_exist(gdf, att_columns)

    sub_df = gdf[list(att_columns)]
    records: List[Dict[str, Any]] = []
    for row in sub_df.to_dict(orient="records"):
        records.append({str(k): to_python_scalar(v) for k, v in row.items()})
    return records


def drop_vec_cols(
    vec_file: str,
    vec_lyr: Optional[str],
    drop_cols: Sequence[str],
    out_vec_file: str,
    out_vec_lyr: Optional[str] = None,
    out_format: str = "GPKG",
    chk_cols_present: bool = True,
) -> None:
    """
    Remove specified columns from a vector layer and write a new vector file.

    :param vec_file: Input vector file path.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param drop_cols: Sequence of column names to drop.
    :param out_vec_file: Output vector file path.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param chk_cols_present: If ``True``, silently ignore column names not
                             present in the layer (Default = ``True``).

    .. code:: python

        import rsgispy.vectorattrs.io

        rsgispy.vectorattrs.io.drop_vec_cols(
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

    write_vec_gdf(data_gdf, out_vec_file, out_vec_lyr, out_format=out_format)


def rename_vec_cols(
    vec_file: str,
    vec_lyr: Optional[str],
    rname_cols_lut: Dict[str, str],
    out_vec_file: str,
    out_vec_lyr: Optional[str] = None,
    out_format: str = "GPKG",
) -> None:
    """
    Rename columns in a vector layer and write a new vector file.

    :param vec_file: Input vector file path.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param rname_cols_lut: Mapping ``{"old_name": "new_name"}``.
    :param out_vec_file: Output vector file path.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).

    .. code:: python

        import rsgispy.vectorattrs.io

        rsgispy.vectorattrs.io.rename_vec_cols(
            "in.gpkg", "polys", {"old_c": "new_c"}, "out.gpkg", "polys"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    data_gdf = read_vec_gdf(vec_file, vec_lyr)
    data_gdf = data_gdf.rename(columns=rname_cols_lut)
    write_vec_gdf(data_gdf, out_vec_file, out_vec_lyr, out_format=out_format)


def sort_vec_lyr(
    vec_file: str,
    vec_lyr: Optional[str],
    out_vec_file: str,
    out_vec_lyr: Optional[str],
    sort_by: Union[str, List[str]],
    ascending: Union[bool, List[bool]],
    out_format: str = "GPKG",
) -> None:
    """
    Sort features in a vector layer by one or more attribute columns.

    :param vec_file: Input vector file path.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param out_vec_file: Output vector file path.
    :param out_vec_lyr: Output vector layer name.
    :param sort_by: Single column name or list of column names to sort by.
    :param ascending: Boolean or list of booleans matching ``sort_by``
                      (``True`` = ascending, ``False`` = descending).
    :param out_format: Output vector format driver (Default = ``'GPKG'``).

    .. code:: python

        import rsgispy.vectorattrs.io

        rsgispy.vectorattrs.io.sort_vec_lyr(
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

    base_gdf = read_vec_gdf(vec_file, vec_lyr)
    cols_to_check = [sort_by] if isinstance(sort_by, str) else sort_by
    validate_columns_exist(base_gdf, cols_to_check)

    sorted_gdf = base_gdf.sort_values(by=sort_by, ascending=ascending)
    write_vec_gdf(sorted_gdf, out_vec_file, out_vec_lyr, out_format=out_format)


def get_unq_col_values(
    vec_file: str,
    vec_lyr: Optional[str],
    col_name: str,
) -> numpy.ndarray:
    """
    Return the unique values within an attribute column.

    :param vec_file: Input vector file path.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param col_name: Name of the column to inspect.
    :return: NumPy array of unique values in ``col_name``.

    .. code:: python

        import rsgispy.vectorattrs.io

        classes = rsgispy.vectorattrs.io.get_unq_col_values(
            "polys.gpkg", "polys", "landcover"
        )
    """
    gdf = read_vec_gdf(vec_file, vec_lyr)
    validate_columns_exist(gdf, [col_name])
    return numpy.asarray(gdf[col_name].unique())


def get_vec_cols_as_array(
    vec_file: str,
    vec_lyr: Optional[str],
    cols: Sequence[str],
    lower_limit: Optional[float] = None,
    upper_limit: Optional[float] = None,
) -> numpy.ndarray:
    """
    Read numeric attribute columns from a vector layer into a 2D ``float``
    NumPy array, optionally filtering out rows outside ``(lower_limit,
    upper_limit)`` or containing non-finite values.

    :param vec_file: Input vector file path.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param cols: Sequence of column names to read.
    :param lower_limit: Optional exclusive lower bound applied across all columns.
    :param upper_limit: Optional exclusive upper bound applied across all columns.
    :return: 2D NumPy float array of shape ``(n_valid_rows, len(cols))``.

    .. code:: python

        import rsgispy.vectorattrs.io

        arr = rsgispy.vectorattrs.io.get_vec_cols_as_array(
            "samples.gpkg", "samples", ["b1", "b2"], lower_limit=0.0
        )
    """
    gdf = read_vec_gdf(vec_file, vec_lyr)
    validate_columns_exist(gdf, cols)
    out_arr = gdf.loc[:, list(cols)].to_numpy(dtype=float)
    return rsgispy.tools.stats.mask_data_to_valid(
        out_arr, lower_limit=lower_limit, upper_limit=upper_limit
    )


def export_vec_attrs_to_csv(
    vec_file: str,
    vec_lyr: Optional[str],
    output_file: str,
) -> None:
    """
    Export the attribute table of a vector layer (excluding geometry) to CSV.

    :param vec_file: Input vector file path.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param output_file: Output CSV file path.

    .. code:: python

        import rsgispy.vectorattrs.io

        rsgispy.vectorattrs.io.export_vec_attrs_to_csv(
            "polys.gpkg", "polys", "attrs.csv"
        )
    """
    data_df = read_vec_gdf(vec_file, vec_lyr)
    if "geometry" in data_df.columns:
        data_df = data_df.drop(columns=["geometry"])
    data_df.to_csv(output_file)


def export_vec_attrs_to_excel(
    vec_file: str,
    vec_lyr: Optional[str],
    output_file: str,
    out_sheet_name: str = "Sheet1",
) -> None:
    """
    Export the attribute table of a vector layer (excluding geometry) to an
    Excel ``.xlsx`` workbook.

    :param vec_file: Input vector file path.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param output_file: Output Excel file path.
    :param out_sheet_name: Worksheet name (Default = ``'Sheet1'``).

    .. code:: python

        import rsgispy.vectorattrs.io

        rsgispy.vectorattrs.io.export_vec_attrs_to_excel(
            "polys.gpkg", "polys", "attrs.xlsx"
        )
    """
    data_df = read_vec_gdf(vec_file, vec_lyr)
    if "geometry" in data_df.columns:
        data_df = data_df.drop(columns=["geometry"])
    try:
        data_df.to_excel(
            output_file, sheet_name=out_sheet_name, engine="xlsxwriter"
        )
    except ImportError:
        try:
            data_df.to_excel(
                output_file, sheet_name=out_sheet_name, engine="openpyxl"
            )
        except ImportError as err:
            raise rsgispy.RSGISPyException(
                "Writing Excel files requires 'xlsxwriter' or 'openpyxl'."
            ) from err


def export_vec_attrs_to_parquet(
    vec_file: str,
    vec_lyr: Optional[str],
    output_file: str,
    gzip_output: bool = True,
) -> None:
    """
    Export the attribute table of a vector layer (excluding geometry) to a
    Parquet file.

    :param vec_file: Input vector file path.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param output_file: Output Parquet file path.
    :param gzip_output: If ``True``, compress using ``gzip`` (Default = ``True``).

    .. code:: python

        import rsgispy.vectorattrs.io

        rsgispy.vectorattrs.io.export_vec_attrs_to_parquet(
            "polys.gpkg", "polys", "attrs.parquet"
        )
    """
    data_df = read_vec_gdf(vec_file, vec_lyr)
    if "geometry" in data_df.columns:
        data_df = data_df.drop(columns=["geometry"])
    try:
        if gzip_output:
            data_df.to_parquet(output_file, compression="gzip")
        else:
            data_df.to_parquet(output_file)
    except ImportError as err:
        raise rsgispy.RSGISPyException(
            "Writing Parquet files requires 'pyarrow' or 'fastparquet'."
        ) from err
