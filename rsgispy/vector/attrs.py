"""
Attribute table I/O, column manipulation, tabular export (CSV/Excel/Parquet),
and spatial attribute enrichment/joins for :mod:`rsgispy.vector.attrs`.
"""

from rsgispy.vector._attrs_columns import (
    add_fid_col,
    add_numeric_col,
    add_numeric_col_from_lst_lut,
    add_numeric_col_lut,
    add_numeric_col_range_lut,
    add_ran_vals_cols,
    add_string_col,
    add_string_col_lut,
    add_unq_numeric_col,
    check_str_col,
    create_angle_sets,
    create_date_col,
    create_name_col,
    create_orthogonal_angle_sets,
    find_replace_str_vec_lyr,
    vector_maths,
)
from rsgispy.vector._attrs_io import (
    drop_vec_cols,
    export_vec_attrs_to_csv,
    export_vec_attrs_to_excel,
    export_vec_attrs_to_parquet,
    get_unq_col_values,
    get_vec_cols_as_array,
    read_vec_column,
    read_vec_columns,
    rename_vec_cols,
    sort_vec_lyr,
    write_vec_column,
    write_vec_column_to_layer,
)
from rsgispy.vector._attrs_spatial import (
    add_geom_bbox_cols,
    annotate_vec_selection,
    perform_spatial_join,
    pop_bbox_cols,
)

__all__ = [
    # Attribute I/O
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
    # Column manipulation
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
    # Spatial attribute enrichment
    "pop_bbox_cols",
    "add_geom_bbox_cols",
    "annotate_vec_selection",
    "perform_spatial_join",
]
