"""
The vectorattrs.columns module provides functions to add, derive, clean, and
categorise columns in vector attribute tables using ``geopandas``.
"""

import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy

import rsgispy
import rsgispy.tools.jsonio
import rsgispy.tools.strings
from rsgispy.vectorprocess import (
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
]


def add_fid_col(
    vec_file: str,
    vec_lyr: Optional[str],
    out_vec_file: str,
    out_vec_lyr: Optional[str],
    out_format: str = "GPKG",
    out_col: str = "fid",
) -> None:
    """
    Add a 1-based unique integer feature ID column to a vector layer.

    :param vec_file: Input vector file path.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param out_vec_file: Output vector file path.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param out_col: Output FID column name (Default = ``'fid'``).

    .. code:: python

        import rsgispy.vectorattrs.columns

        rsgispy.vectorattrs.columns.add_fid_col(
            "in.gpkg", "polys", "out.gpkg", "polys", out_col="feat_id"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    base_gdf = read_vec_gdf(vec_file, vec_lyr)
    base_gdf[out_col] = numpy.arange(1, len(base_gdf) + 1, 1, dtype=int)
    write_vec_gdf(base_gdf, out_vec_file, out_vec_lyr, out_format=out_format)


def add_unq_numeric_col(
    vec_file: str,
    vec_lyr: Optional[str],
    unq_col: str,
    out_col: str,
    out_vec_file: str,
    out_vec_lyr: Optional[str],
    out_format: str = "GPKG",
    lut_json_file: Optional[str] = None,
) -> None:
    """
    Add a sequential integer ID column (starting at ``1``) mapped from unique
    values in ``unq_col``, optionally writing the lookup table to JSON.

    :param vec_file: Input vector file path.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param unq_col: Source column whose unique values define the mapping.
    :param out_col: Output numeric column name.
    :param out_vec_file: Output vector file path.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param lut_json_file: Optional path to write the JSON lookup table.

    .. code:: python

        import rsgispy.vectorattrs.columns

        rsgispy.vectorattrs.columns.add_unq_numeric_col(
            "in.gpkg", "polys", "species", "sp_id", "out.gpkg", "polys"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    base_gdf = read_vec_gdf(vec_file, vec_lyr)
    validate_columns_exist(base_gdf, [unq_col])

    unq_vals = base_gdf[unq_col].unique()
    base_gdf[out_col] = numpy.zeros(len(base_gdf), dtype=int)
    lut: Dict[str, Dict[Any, Any]] = {"id": {}, "val": {}}

    for idx, raw_val in enumerate(unq_vals, start=1):
        py_val = to_python_scalar(raw_val)
        sel_rows = base_gdf[unq_col] == raw_val
        base_gdf.loc[sel_rows, out_col] = idx
        lut["id"][str(idx)] = py_val
        lut["val"][str(py_val)] = idx

    write_vec_gdf(base_gdf, out_vec_file, out_vec_lyr, out_format=out_format)
    if lut_json_file is not None:
        rsgispy.tools.jsonio.write_dict_to_json(lut, lut_json_file)


def add_numeric_col(
    vec_file: str,
    vec_lyr: Optional[str],
    out_col: str,
    out_vec_file: str,
    out_vec_lyr: Optional[str],
    out_val: float = 1,
    out_format: str = "GPKG",
    out_col_int: bool = False,
) -> None:
    """
    Add a constant numeric column to a vector layer.

    :param vec_file: Input vector file path.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param out_col: Output numeric column name.
    :param out_vec_file: Output vector file path.
    :param out_vec_lyr: Output vector layer name.
    :param out_val: Constant value assigned to all rows (Default = ``1``).
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param out_col_int: If ``True``, store values as integers; otherwise float
                        (Default = ``False``).

    .. code:: python

        import rsgispy.vectorattrs.columns

        rsgispy.vectorattrs.columns.add_numeric_col(
            "in.gpkg", "polys", "weight", "out.gpkg", "polys", out_val=1.5
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    base_gdf = read_vec_gdf(vec_file, vec_lyr)
    dtype = int if out_col_int else float
    base_gdf[out_col] = numpy.full(len(base_gdf), out_val, dtype=dtype)
    write_vec_gdf(base_gdf, out_vec_file, out_vec_lyr, out_format=out_format)


def add_numeric_col_lut(
    vec_file: str,
    vec_lyr: Optional[str],
    ref_col: str,
    val_lut: Dict[Any, Union[int, float]],
    out_col: str,
    out_vec_file: str,
    out_vec_lyr: Optional[str],
    out_format: str = "GPKG",
    col_dtype: str = "int",
) -> None:
    """
    Add a numeric column derived from ``ref_col`` via a dictionary lookup table.

    :param vec_file: Input vector file path.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param ref_col: Reference column name whose values match keys in ``val_lut``.
    :param val_lut: Dictionary mapping reference values to output numbers.
    :param out_col: Output numeric column name.
    :param out_vec_file: Output vector file path.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param col_dtype: Output column data type (``'int'`` or ``'float'``;
                      Default = ``'int'``).

    .. code:: python

        import rsgispy.vectorattrs.columns

        rsgispy.vectorattrs.columns.add_numeric_col_lut(
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
    base_gdf[out_col] = numpy.zeros(len(base_gdf), dtype=np_dtype)
    for lut_key, out_val in val_lut.items():
        sel_rows = base_gdf[ref_col] == lut_key
        base_gdf.loc[sel_rows, out_col] = out_val

    write_vec_gdf(base_gdf, out_vec_file, out_vec_lyr, out_format=out_format)


def add_numeric_col_range_lut(
    vec_file: str,
    vec_lyr: Optional[str],
    vec_col: str,
    out_vec_file: str,
    out_vec_lyr: Optional[str],
    out_vec_col: str,
    val_lut: Dict[int, Tuple[float, float]],
    out_format: str = "GPKG",
) -> None:
    """
    Add a numeric class column using half-open ``[lower, upper)`` value ranges.

    Features outside all configured intervals receive ``0``.

    :param vec_file: Input vector file path.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param vec_col: Numeric source column name to classify.
    :param out_vec_file: Output vector file path.
    :param out_vec_lyr: Output vector layer name.
    :param out_vec_col: Output numeric column name.
    :param val_lut: Dictionary mapping output integer code to ``(low, high)``
                    bounds checked as ``low <= val < high``.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).

    .. code:: python

        import rsgispy.vectorattrs.columns

        rsgispy.vectorattrs.columns.add_numeric_col_range_lut(
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

    out_arr = numpy.zeros(len(data_gdf), dtype=int)
    vals_arr = data_gdf[vec_col].to_numpy()
    for out_code, (low_val, high_val) in val_lut.items():
        out_arr[(vals_arr >= low_val) & (vals_arr < high_val)] = out_code

    data_gdf[out_vec_col] = out_arr
    write_vec_gdf(data_gdf, out_vec_file, out_vec_lyr, out_format=out_format)


def add_numeric_col_from_lst_lut(
    vec_file: str,
    vec_lyr: Optional[str],
    ref_col: str,
    vals_lut: Sequence[Tuple[Union[str, int], int]],
    out_col: str,
    out_vec_file: str,
    out_vec_lyr: Optional[str],
    out_format: str = "GPKG",
) -> None:
    """
    Add an integer column derived from ``ref_col`` using a sequence of
    ``(match_value, output_int)`` pairs.

    :param vec_file: Input vector file path.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param ref_col: Reference column name.
    :param vals_lut: Sequence of ``(lookup_value, out_int)`` tuples.
    :param out_col: Output numeric column name.
    :param out_vec_file: Output vector file path.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).

    .. code:: python

        import rsgispy.vectorattrs.columns

        rsgispy.vectorattrs.columns.add_numeric_col_from_lst_lut(
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

    out_vals = numpy.zeros(len(base_gdf), dtype=int)
    ref_vals = base_gdf[ref_col].to_numpy()
    for match_val, out_int in vals_lut:
        out_vals[ref_vals == match_val] = out_int

    base_gdf[out_col] = out_vals
    write_vec_gdf(base_gdf, out_vec_file, out_vec_lyr, out_format=out_format)


def add_string_col(
    vec_file: str,
    vec_lyr: Optional[str],
    out_col: str,
    out_vec_file: str,
    out_vec_lyr: Optional[str],
    out_val: str = "str_val",
    out_format: str = "GPKG",
) -> None:
    """
    Add a constant string column to a vector layer.

    :param vec_file: Input vector file path.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param out_col: Output string column name.
    :param out_vec_file: Output vector file path.
    :param out_vec_lyr: Output vector layer name.
    :param out_val: Constant string assigned to all rows (Default = ``'str_val'``).
    :param out_format: Output vector format driver (Default = ``'GPKG'``).

    .. code:: python

        import rsgispy.vectorattrs.columns

        rsgispy.vectorattrs.columns.add_string_col(
            "in.gpkg", "polys", "source", "out.gpkg", "polys", out_val="survey_2026"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    base_gdf = read_vec_gdf(vec_file, vec_lyr)
    base_gdf[out_col] = [str(out_val)] * len(base_gdf)
    write_vec_gdf(base_gdf, out_vec_file, out_vec_lyr, out_format=out_format)


def add_string_col_lut(
    vec_file: str,
    vec_lyr: Optional[str],
    ref_col: str,
    val_lut: Dict[Any, str],
    out_col: str,
    out_vec_file: str,
    out_vec_lyr: Optional[str],
    out_format: str = "GPKG",
) -> None:
    """
    Add a string column derived from ``ref_col`` via a dictionary lookup table.

    :param vec_file: Input vector file path.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param ref_col: Reference column name whose values match keys in ``val_lut``.
    :param val_lut: Dictionary mapping reference values to output strings.
    :param out_col: Output string column name.
    :param out_vec_file: Output vector file path.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).

    .. code:: python

        import rsgispy.vectorattrs.columns

        rsgispy.vectorattrs.columns.add_string_col_lut(
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

    write_vec_gdf(base_gdf, out_vec_file, out_vec_lyr, out_format=out_format)


def add_ran_vals_cols(
    vec_file: str,
    vec_lyr: Optional[str],
    out_vec_file: str,
    out_vec_lyr: Optional[str],
    out_format: str = "GPKG",
    out_col: str = "ran_vals",
    rnd_seed: Optional[int] = None,
) -> None:
    """
    Add a column populated with random integers in ``[1, 10 * n_rows)``.

    Note that random values are not guaranteed to be unique across rows.

    :param vec_file: Input vector file path.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param out_vec_file: Output vector file path.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param out_col: Output column name (Default = ``'ran_vals'``).
    :param rnd_seed: Optional seed for NumPy's random generator.

    .. code:: python

        import rsgispy.vectorattrs.columns

        rsgispy.vectorattrs.columns.add_ran_vals_cols(
            "in.gpkg", "polys", "out.gpkg", "polys", rnd_seed=42
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    data_gdf = read_vec_gdf(vec_file, vec_lyr)
    rng = numpy.random.default_rng(seed=rnd_seed)
    n_rows = len(data_gdf)
    high = max(2, n_rows * 10)
    data_gdf[out_col] = rng.integers(low=1, high=high, size=n_rows)
    write_vec_gdf(data_gdf, out_vec_file, out_vec_lyr, out_format=out_format)


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
    vec_file: str,
    vec_lyr: Optional[str],
    out_vec_file: str,
    out_vec_lyr: Optional[str],
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
) -> None:
    """
    Create a feature name column formatted from X and Y coordinate attributes.

    :param vec_file: Input vector file path.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param out_vec_file: Output vector file path.
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

    .. code:: python

        import rsgispy.vectorattrs.columns

        rsgispy.vectorattrs.columns.create_name_col(
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
    write_vec_gdf(base_gdf, out_vec_file, out_vec_lyr, out_format=out_format)


def check_str_col(
    vec_file: str,
    vec_lyr: Optional[str],
    vec_col: str,
    out_vec_file: str,
    out_vec_lyr: Optional[str],
    out_format: str = "GPKG",
    rm_non_ascii: bool = True,
    rm_dashs: bool = False,
    rm_spaces: bool = False,
    rm_punc: bool = False,
    use_pbar: bool = True,
) -> None:
    """
    Sanitise values in a string column using :func:`rsgispy.tools.strings.check_str`.

    :param vec_file: Input vector file path.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param vec_col: String column name to clean.
    :param out_vec_file: Output vector file path.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param rm_non_ascii: Remove non-ASCII characters (Default = ``True``).
    :param rm_dashs: Replace dashes with underscores (Default = ``False``).
    :param rm_spaces: Replace spaces with underscores (Default = ``False``).
    :param rm_punc: Remove punctuation other than ``_`` and ``-``
                    (Default = ``False``).
    :param use_pbar: If ``True``, display a tqdm progress bar while cleaning
                     values (Default = ``True``).

    .. code:: python

        import rsgispy.vectorattrs.columns

        rsgispy.vectorattrs.columns.check_str_col(
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
    write_vec_gdf(base_gdf, out_vec_file, out_vec_lyr, out_format=out_format)


def create_date_col(
    vec_file: str,
    vec_lyr: Optional[str],
    year_col: str,
    month_col: str,
    day_col: str,
    out_date_col: str,
    out_vec_file: str,
    out_vec_lyr: Optional[str],
    out_format: str = "GPKG",
) -> None:
    """
    Create a date/datetime column from integer year, month, and day columns.

    :param vec_file: Input vector file path.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param year_col: Column containing integer years.
    :param month_col: Column containing integer months (1–12).
    :param day_col: Column containing integer days of month (1–31).
    :param out_date_col: Output date column name.
    :param out_vec_file: Output vector file path.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).

    .. code:: python

        import rsgispy.vectorattrs.columns

        rsgispy.vectorattrs.columns.create_date_col(
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
    write_vec_gdf(data_gdf, out_vec_file, out_vec_lyr, out_format=out_format)


def find_replace_str_vec_lyr(
    vec_file: str,
    vec_lyr: Optional[str],
    out_vec_file: str,
    out_vec_lyr: Optional[str],
    cols: Sequence[str],
    find_replace: Dict[str, str],
    out_format: str = "GPKG",
) -> None:
    """
    Perform substring find-and-replace across specified string columns.

    :param vec_file: Input vector file path.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param out_vec_file: Output vector file path.
    :param out_vec_lyr: Output vector layer name.
    :param cols: Sequence of string column names to update.
    :param find_replace: Mapping ``{"find_str": "replacement_str"}``.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).

    .. code:: python

        import rsgispy.vectorattrs.columns

        rsgispy.vectorattrs.columns.find_replace_str_vec_lyr(
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

    write_vec_gdf(base_gdf, out_vec_file, out_vec_lyr, out_format=out_format)


def create_angle_sets(
    vec_file: str,
    vec_lyr: Optional[str],
    angle_col: str,
    start_angle: int,
    angle_set_width: int,
    out_vec_file: str,
    out_vec_lyr: Optional[str],
    out_format: str = "GPKG",
    out_angle_set_col: str = "angle_set",
) -> None:
    """
    Partition point/feature azimuth angles (0–360 degrees) into mirrored
    angular sectors (each sector includes its opposite 180-degree direction).

    :param vec_file: Input vector file path.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param angle_col: Column containing azimuth angles in ``[0, 360]`` degrees.
    :param start_angle: Starting angle in ``[0, 360]`` degrees.
    :param angle_set_width: Angular sector width in degrees; must divide ``180``.
    :param out_vec_file: Output vector file path.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param out_angle_set_col: Output integer sector ID column
                              (Default = ``'angle_set'``).

    .. code:: python

        import rsgispy.vectorattrs.columns

        rsgispy.vectorattrs.columns.create_angle_sets(
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

    def _wrap_start(arr: numpy.ndarray) -> numpy.ndarray:
        out = arr.copy()
        out[out > 360] -= 360
        out[out == 360] = 0
        return out

    def _wrap_end(arr: numpy.ndarray) -> numpy.ndarray:
        out = arr.copy()
        out[out > 360] -= 360
        out[out == 0] = 360
        return out

    base_starts = start_angle + numpy.arange(0, 180, angle_set_width)
    p1_starts = _wrap_start(base_starts)
    p1_ends = _wrap_end(p1_starts + angle_set_width)
    p2_starts = _wrap_start(p1_starts + 180)
    p2_ends = _wrap_end(p1_ends + 180)

    data_gdf = read_vec_gdf(vec_file, vec_lyr)
    validate_columns_exist(data_gdf, [angle_col])
    angles = data_gdf[angle_col].to_numpy()
    angle_sets = numpy.zeros_like(angles, dtype=int)

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
    write_vec_gdf(data_gdf, out_vec_file, out_vec_lyr, out_format=out_format)


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
    vec_file: str,
    vec_lyr: Optional[str],
    angle_col: str,
    start_angle: int,
    angle_half_width: int,
    out_vec_file: str,
    out_vec_lyr: Optional[str],
    out_format: str = "GPKG",
    out_angle_set_col: str = "angle_set",
) -> None:
    """
    Create two orthogonal mirrored angle sets (labelled ``1`` and ``2``;
    outside angles receive ``0``).

    :param vec_file: Input vector file path.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param angle_col: Column containing azimuth angles in ``[0, 360]`` degrees.
    :param start_angle: Central angle of set 1 in ``[0, 180]`` degrees.
    :param angle_half_width: Half-width of each angular sector in ``[0, 45]``
                             degrees.
    :param out_vec_file: Output vector file path.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param out_angle_set_col: Output integer sector ID column
                              (Default = ``'angle_set'``).

    .. code:: python

        import rsgispy.vectorattrs.columns

        rsgispy.vectorattrs.columns.create_orthogonal_angle_sets(
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
    angle_sets = numpy.zeros_like(angles, dtype=int)

    for low_a, high_a in set1_ranges:
        angle_sets[(angles >= low_a) & (angles <= high_a)] = 1
    for low_a, high_a in set2_ranges:
        angle_sets[(angles >= low_a) & (angles <= high_a)] = 2

    data_gdf[out_angle_set_col] = angle_sets
    write_vec_gdf(data_gdf, out_vec_file, out_vec_lyr, out_format=out_format)
