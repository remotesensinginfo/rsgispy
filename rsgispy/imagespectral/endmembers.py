# pylint: disable=too-many-lines
"""
The imagespectral.endmembers module provides spectral endmember table I/O
(:class:`pandas.DataFrame` and legacy ``.mtxt`` conversion), polygon-based
endmember extraction, spectral signature plotting, endmember equality checks,
and Pixel Purity Index (PPI) calculation.

Endmembers are stored and passed as :class:`pandas.DataFrame` tables (in
memory or read from / written to disk in the user's chosen format such as
CSV, TSV, Parquet, Feather, Excel, JSON, Pickle, or HDF5) with **spectral
bands as rows and endmember class names as columns**.
"""

from contextlib import ExitStack
import json
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
import rasterio.windows

import rsgispy
from rsgispy.imageprocess import create_output_from_dataset, ensure_file_exists
from rsgispy.imageutils.stats import pop_img_stats
from rsgispy.tools.geometry import do_bboxes_intersect, reproj_bbox
from rsgispy.tools.tables import resolve_table_format
from rsgispy.vectorprocess import (
    VectorInput,
    get_gdf_bbox,
    read_vec_gdf,
    resolve_vector_crs,
)

EndmemberInput = Union[pd.DataFrame, str, os.PathLike]

__all__: List[str] = [
    "EndmemberInput",
    "ImageEndmemberInfo",
    "extract_avg_endmembers",
    "read_endmembers_df",
    "write_endmembers_df",
    "coerce_endmembers_df",
    "prepare_endmembers_array",
    "convert_mtxt_to_df",
    "plot_endmembers",
    "are_endmembers_equal",
    "calc_ppi",
]


class ImageEndmemberInfo:
    """
    Container associating an unmixed abundance image with the endmembers
    :class:`pandas.DataFrame` (or file path) and optional endmember class
    names used to produce it.

    :param in_unmix_img: Path to the linear unmixed abundance image file.
    :param endmembers: :class:`pandas.DataFrame` (columns = class names,
                       rows = bands) or file path readable by
                       :func:`read_endmembers_df`.
    :param endmember_names: Optional ordered sequence of endmember names
                            matching the bands of ``in_unmix_img``. Inferred
                            from the DataFrame columns when ``None``.
    :param file_format: Optional format override when ``endmembers`` is a file
                        path (e.g., ``'csv'``, ``'parquet'``, ``'excel'``,
                        ``'json'``, ``'feather'``, ``'pickle'``).

    .. code:: python

        import rsgispy.imagespectral.endmembers as em

        info = em.ImageEndmemberInfo(
            in_unmix_img="unmix.tif",
            endmembers="endmembers.csv",
        )
    """

    def __init__(
        self,
        in_unmix_img: str,
        endmembers: EndmemberInput,
        endmember_names: Optional[Sequence[str]] = None,
        file_format: Optional[str] = None,
    ) -> None:
        self.in_unmix_img = in_unmix_img
        self.endmembers = endmembers
        self.file_format = file_format
        if endmember_names is not None:
            self.endmember_names: Optional[List[str]] = [
                str(n) for n in endmember_names
            ]
        elif isinstance(endmembers, pd.DataFrame):
            self.endmember_names = [str(c) for c in endmembers.columns]
        else:
            self.endmember_names = None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ImageEndmemberInfo):
            return NotImplemented
        if isinstance(self.endmembers, pd.DataFrame) or isinstance(
            other.endmembers, pd.DataFrame
        ):
            if not (
                isinstance(self.endmembers, pd.DataFrame)
                and isinstance(other.endmembers, pd.DataFrame)
            ):
                return False
            em_eq = self.endmembers.equals(other.endmembers)
        else:
            em_eq = os.fspath(self.endmembers) == os.fspath(other.endmembers)
        return (
            self.in_unmix_img == other.in_unmix_img
            and em_eq
            and self.endmember_names == other.endmember_names
            and self.file_format == other.file_format
        )

    def __repr__(self) -> str:
        return (
            f"ImageEndmemberInfo(in_unmix_img={self.in_unmix_img!r}, "
            f"endmembers={self.endmembers!r}, "
            f"endmember_names={self.endmember_names!r}, "
            f"file_format={self.file_format!r})"
        )


def _resolve_table_format(
    file_path: Union[str, os.PathLike],
    file_format: Optional[str] = None,
) -> str:
    """Resolve normalised table format from explicit ``file_format`` or suffix."""
    return resolve_table_format(file_path, out_format=file_format)


def _normalise_endmembers_df(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalise an endmembers DataFrame (rows=bands, cols=classes)."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise rsgispy.RSGISPyException(
            "Endmembers DataFrame must be a non-empty pandas.DataFrame."
        )
    out = df.copy()
    out.columns = [str(col) for col in out.columns]
    if len(set(out.columns)) != len(out.columns):
        raise rsgispy.RSGISPyException(
            "Endmembers DataFrame column (class) names must be unique."
        )
    try:
        for col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="raise").astype(
                np.float64
            )
    except (ValueError, TypeError) as err:
        raise rsgispy.RSGISPyException(
            f"All endmember class columns must contain numeric values: {err}"
        ) from err
    if out.index.name is None:
        out.index = pd.Index(out.index, name="band")
    return out


def _promote_band_index_if_present(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect an explicit band/unnamed index column loaded from CSV/TSV/Excel and
    restore it as the DataFrame index; keep all columns when none is present.
    """
    if df.shape[1] >= 2:
        first_col = str(df.columns[0])
        is_idx_name = first_col.lower() in (
            "band",
            "__band__",
        ) or first_col.startswith("Unnamed:")
        is_non_numeric = not pd.api.types.is_numeric_dtype(df.iloc[:, 0])
        if is_idx_name or is_non_numeric:
            idx_vals = df.iloc[:, 0].astype(str)
            rest = df.iloc[:, 1:].copy()
            rest.index = pd.Index(idx_vals, name="band")
            return rest
    out = df.copy()
    out.index = pd.Index(
        [f"band_{i + 1}" for i in range(len(out))], name="band"
    )
    return out


def read_endmembers_df(
    endmembers_file: Union[str, os.PathLike],
    file_format: Optional[str] = None,
    gain: float = 1,
    sheet_name: Union[int, str] = 0,
) -> pd.DataFrame:
    """
    Read spectral endmembers from a tabular file into a :class:`pandas.DataFrame`
    with spectral bands as rows and endmember class names as columns.

    Supported formats (specified via ``file_format`` or inferred from file
    extension): ``'csv'``, ``'tsv'``, ``'parquet'``, ``'feather'``,
    ``'excel'`` (``.xlsx``/``.xls``), ``'json'``, ``'pickle'``, and ``'hdf'``.

    :param endmembers_file: Path to the input endmembers file.
    :param file_format: Optional format name overriding extension detection
                        (Default = ``None``).
    :param gain: Optional gain divisor applied to spectral values when
                 ``gain > 1`` (Default = ``1``).
    :param sheet_name: Worksheet name or 0-based index when reading Excel files
                       (Default = ``0``).
    :return: :class:`pandas.DataFrame` of shape ``(n_bands, n_classes)`` where
             column names are the endmember class names.

    .. code:: python

        import rsgispy.imagespectral.endmembers as em

        em_df = em.read_endmembers_df("endmembers.csv")
        em_pq = em.read_endmembers_df("endmembers.parquet")
    """
    # pylint: disable=too-many-branches
    path_str = os.fspath(endmembers_file)
    ensure_file_exists(path_str, "Endmembers file")
    fmt = _resolve_table_format(path_str, file_format)

    if fmt == "csv":
        raw_df = pd.read_csv(path_str)
        df = _promote_band_index_if_present(raw_df)
    elif fmt == "tsv":
        raw_df = pd.read_csv(path_str, sep="\t")
        df = _promote_band_index_if_present(raw_df)
    elif fmt == "parquet":
        try:
            df = pd.read_parquet(path_str)
        except ImportError as err:
            raise rsgispy.RSGISPyException(
                "Reading Parquet endmember files requires 'pyarrow' or 'fastparquet'."
            ) from err
    elif fmt == "feather":
        try:
            raw_df = pd.read_feather(path_str)
        except ImportError as err:
            raise rsgispy.RSGISPyException(
                "Reading Feather endmember files requires 'pyarrow'."
            ) from err
        if "__band__" in raw_df.columns:
            idx = pd.Index(raw_df["__band__"].astype(str), name="band")
            df = raw_df.drop(columns=["__band__"]).set_index(idx)
        else:
            df = raw_df
    elif fmt == "excel":
        try:
            raw_df = pd.read_excel(path_str, sheet_name=sheet_name)
        except ImportError as err:
            raise rsgispy.RSGISPyException(
                "Reading Excel endmember files requires 'openpyxl'."
            ) from err
        df = _promote_band_index_if_present(raw_df)
    elif fmt == "json":
        with open(path_str, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict) and {"columns", "data"} <= set(
            payload.keys()
        ):
            df = pd.DataFrame(
                payload["data"],
                columns=payload["columns"],
                index=pd.Index(payload["index"], name="band")
                if "index" in payload
                else None,
            )
        else:
            df = pd.read_json(path_str, orient="columns")
    elif fmt == "pickle":
        df = pd.read_pickle(path_str)
    else:  # hdf
        try:
            df = pd.read_hdf(path_str)
        except ImportError as err:
            raise rsgispy.RSGISPyException(
                "Reading HDF5 endmember files requires 'tables' (PyTables)."
            ) from err

    out_df = _normalise_endmembers_df(df)
    if gain > 1:
        out_df = out_df / float(gain)
    return out_df


def write_endmembers_df(
    endmembers_df: pd.DataFrame,
    out_file: Union[str, os.PathLike],
    file_format: Optional[str] = None,
    sheet_name: str = "endmembers",
) -> None:
    """
    Write an endmembers :class:`pandas.DataFrame` (bands as rows, endmember
    class names as columns) to ``out_file`` in the user's chosen format.

    Supported formats (specified via ``file_format`` or inferred from file
    extension): ``'csv'``, ``'tsv'``, ``'parquet'``, ``'feather'``,
    ``'excel'`` (``.xlsx``/``.xls``), ``'json'``, ``'pickle'``, and ``'hdf'``.

    :param endmembers_df: :class:`pandas.DataFrame` with spectral bands as rows
                          and endmember class names as columns.
    :param out_file: Output file path.
    :param file_format: Optional format name overriding extension detection
                        (Default = ``None``).
    :param sheet_name: Worksheet name when writing Excel files
                       (Default = ``'endmembers'``).

    .. code:: python

        import pandas as pd
        import rsgispy.imagespectral.endmembers as em

        em_df = pd.DataFrame(
            {"Veg": [0.05, 0.45, 0.20], "Soil": [0.30, 0.35, 0.40]},
            index=["blue", "nir", "swir1"],
        )
        em.write_endmembers_df(em_df, "endmembers.csv")
        em.write_endmembers_df(em_df, "endmembers.parquet")
    """
    # pylint: disable=too-many-branches
    path_str = os.fspath(out_file)
    fmt = _resolve_table_format(path_str, file_format)
    df = _normalise_endmembers_df(endmembers_df)

    out_dir = os.path.dirname(os.path.abspath(path_str))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    if fmt == "csv":
        df.to_csv(path_str, index=True)
    elif fmt == "tsv":
        df.to_csv(path_str, sep="\t", index=True)
    elif fmt == "parquet":
        try:
            df.to_parquet(path_str, index=True)
        except ImportError as err:
            raise rsgispy.RSGISPyException(
                "Writing Parquet endmember files requires 'pyarrow' or 'fastparquet'."
            ) from err
    elif fmt == "feather":
        df_fth = df.reset_index(drop=True).copy()
        df_fth.insert(0, "__band__", df.index.astype(str))
        try:
            df_fth.to_feather(path_str)
        except ImportError as err:
            raise rsgispy.RSGISPyException(
                "Writing Feather endmember files requires 'pyarrow'."
            ) from err
    elif fmt == "excel":
        try:
            df.to_excel(
                path_str, sheet_name=sheet_name, index=True, engine="openpyxl"
            )
        except ImportError:
            try:
                df.to_excel(
                    path_str,
                    sheet_name=sheet_name,
                    index=True,
                    engine="xlsxwriter",
                )
            except ImportError as err:
                raise rsgispy.RSGISPyException(
                    "Writing Excel endmember files requires 'openpyxl' or 'xlsxwriter'."
                ) from err
    elif fmt == "json":
        df.to_json(path_str, orient="split", indent=2)
    elif fmt == "pickle":
        df.to_pickle(path_str)
    else:  # hdf
        try:
            df.to_hdf(path_str, key="endmembers", mode="w")
        except ImportError as err:
            raise rsgispy.RSGISPyException(
                "Writing HDF5 endmember files requires 'tables' (PyTables)."
            ) from err


def convert_mtxt_to_df(
    mtxt_file: Union[str, os.PathLike],
    endmember_names: Optional[Sequence[str]] = None,
    band_names: Optional[Sequence[str]] = None,
    gain: float = 1,
    out_file: Optional[Union[str, os.PathLike]] = None,
    file_format: Optional[str] = None,
) -> pd.DataFrame:
    """
    Convert a legacy RSGISLib ``.mtxt`` endmembers matrix file into a
    :class:`pandas.DataFrame` with spectral bands as rows and endmember class
    names as columns.

    Optionally writes the converted DataFrame to ``out_file`` in any format
    supported by :func:`write_endmembers_df`.

    :param mtxt_file: Path to the legacy ``.mtxt`` endmembers file.
    :param endmember_names: Optional sequence of class names matching the ``m``
                            endmembers in ``mtxt_file``. When ``None``, columns
                            are named ``endmember_1``, ..., ``endmember_m``.
    :param band_names: Optional sequence of band labels matching the ``n`` bands
                       in ``mtxt_file``. When ``None``, rows are indexed
                       ``band_1``, ..., ``band_n``.
    :param gain: Optional gain divisor applied to endmember spectra when
                 ``gain > 1`` (Default = ``1``).
    :param out_file: Optional output file path to write the converted DataFrame.
    :param file_format: Optional format override when ``out_file`` is given
                        (Default = ``None``).
    :return: Converted :class:`pandas.DataFrame` of shape
             ``(n_bands, m_endmembers)``.

    .. code:: python

        import rsgispy.imagespectral.endmembers as em

        em_df = em.convert_mtxt_to_df(
            "old_endmembers.mtxt",
            endmember_names=["Veg", "Soil", "Water"],
            out_file="endmembers.csv",
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    path_str = os.fspath(mtxt_file)
    ensure_file_exists(path_str, "Legacy endmembers (.mtxt) file")
    with open(path_str, "r", encoding="utf-8") as handle:
        data_lst = [line.strip() for line in handle if line.strip() != ""]

    if len(data_lst) < 3:
        raise rsgispy.RSGISPyException(
            f"Invalid endmembers file format in '{path_str}'."
        )

    m = int(data_lst[0].split("=")[-1])
    n = int(data_lst[1].split("=")[-1])
    raw_vals = [val.strip() for val in data_lst[2].split(",") if val.strip() != ""]
    if len(raw_vals) != m * n:
        raise rsgispy.RSGISPyException(
            f"Expected {m * n} values in '{path_str}', found {len(raw_vals)}."
        )

    # Legacy .mtxt stores values in column-major order across (m, n): band-outer,
    # endmember-inner. Transpose to (n_bands, m_endmembers) for DataFrame.
    end_mem_arr = np.zeros((m, n), dtype=np.float64)
    idx = 0
    for n_idx in range(n):
        for m_idx in range(m):
            end_mem_arr[m_idx, n_idx] = float(raw_vals[idx])
            idx += 1

    if gain > 1:
        end_mem_arr = end_mem_arr / float(gain)

    if endmember_names is not None:
        if len(endmember_names) != m:
            raise rsgispy.RSGISPyException(
                f"Length of endmember_names ({len(endmember_names)}) does not "
                f"match endmember count ({m}) in '{path_str}'."
            )
        col_names = [str(name) for name in endmember_names]
    else:
        col_names = [f"endmember_{i + 1}" for i in range(m)]

    if band_names is not None:
        if len(band_names) != n:
            raise rsgispy.RSGISPyException(
                f"Length of band_names ({len(band_names)}) does not match "
                f"band count ({n}) in '{path_str}'."
            )
        row_idx = pd.Index([str(b) for b in band_names], name="band")
    else:
        row_idx = pd.Index([f"band_{i + 1}" for i in range(n)], name="band")

    df = _normalise_endmembers_df(
        pd.DataFrame(end_mem_arr.T, index=row_idx, columns=col_names)
    )
    if out_file is not None:
        write_endmembers_df(df, out_file, file_format=file_format)
    return df


def coerce_endmembers_df(
    endmembers: EndmemberInput,
    gain: float = 1,
    file_format: Optional[str] = None,
) -> pd.DataFrame:
    """Resolve an in-memory DataFrame or file path into a validated DataFrame."""
    if isinstance(endmembers, pd.DataFrame):
        df = _normalise_endmembers_df(endmembers)
        if gain > 1:
            df = df / float(gain)
        return df
    return read_endmembers_df(endmembers, file_format=file_format, gain=gain)


def prepare_endmembers_array(
    endmembers: EndmemberInput,
    gain: float = 1,
    weight: Optional[float] = None,
    file_format: Optional[str] = None,
) -> Tuple[List[str], int, int, np.ndarray]:
    """
    Convert an endmembers DataFrame or file into solver matrix shape
    ``(m_endmembers, n_bands [+ 1 if weight])`` plus class names.
    """
    df = coerce_endmembers_df(endmembers, gain=gain, file_format=file_format)
    names = [str(col) for col in df.columns]
    # DataFrame has shape (n_bands, m_endmembers); transpose to (m, n_bands)
    arr = df.to_numpy(dtype=np.float64).T
    m, n = arr.shape
    if weight is not None:
        weights = np.full((m, 1), float(weight), dtype=np.float64)
        arr = np.hstack([arr, weights])
        n += 1
    return names, m, n, arr


def plot_endmembers(
    endmembers: EndmemberInput,
    out_plot_file: str,
    endmember_names: Optional[Sequence[str]] = None,
    plot_title: Optional[str] = None,
    gain: float = 1,
    wavelengths: Optional[Sequence[float]] = None,
    file_format: Optional[str] = None,
) -> None:
    """
    Plot spectral endmember signatures from a :class:`pandas.DataFrame` (or
    endmembers file) and save the figure to disk.

    :param endmembers: :class:`pandas.DataFrame` (columns = class names,
                       rows = bands) or file path readable by
                       :func:`read_endmembers_df`.
    :param out_plot_file: Output image path for the saved figure.
    :param endmember_names: Optional legend labels overriding DataFrame column
                            names (must match endmember count if supplied).
    :param plot_title: Optional plot title string.
    :param gain: Optional gain divisor applied to endmember spectra
                 (Default = ``1``).
    :param wavelengths: Optional sequence of band wavelengths (band indices used
                        when ``None``).
    :param file_format: Optional format override when ``endmembers`` is a file
                        path (Default = ``None``).

    .. code:: python

        import rsgispy.imagespectral.endmembers as em

        em.plot_endmembers(
            "endmembers.csv",
            "endmembers.png",
            plot_title="Endmember Signatures",
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    # pylint: disable=import-outside-toplevel
    import matplotlib.pyplot as plt

    df = coerce_endmembers_df(endmembers, gain=gain, file_format=file_format)
    n_bands, n_endmembers = df.shape
    if endmember_names is not None:
        if len(endmember_names) != n_endmembers:
            raise rsgispy.RSGISPyException(
                "The list of names provided does not match the number of "
                "endmembers in the input table."
            )
        labels = [str(name) for name in endmember_names]
    else:
        labels = [str(col) for col in df.columns]

    x_axis_lbl = "Wavelength"
    if wavelengths is not None:
        if not isinstance(wavelengths, (list, tuple, np.ndarray)):
            raise rsgispy.RSGISPyException(
                "If provided the wavelengths variable must be a sequence."
            )
        if len(wavelengths) != n_bands:
            raise rsgispy.RSGISPyException(
                "The number of wavelengths provided is not equal to the "
                "number of image bands in the endmembers."
            )
        x_vals: Sequence[float] = list(wavelengths)
    else:
        x_vals = list(range(n_bands))
        x_axis_lbl = "Band"

    arr = df.to_numpy(dtype=np.float64)
    fig = plt.figure(figsize=(7, 5), dpi=300)
    try:
        ax1 = fig.add_subplot(111)
        for i in range(n_endmembers):
            ax1.plot(x_vals, arr[:, i], label=labels[i])

        if plot_title is not None:
            ax1.set_title(plot_title)
        ax1.set_xlabel(x_axis_lbl)
        ax1.set_ylabel("Reflectance")
        ax1.legend()
        fig.savefig(out_plot_file)
    finally:
        plt.close(fig)


def are_endmembers_equal(
    ref_endmembers: EndmemberInput,
    cmp_endmembers: EndmemberInput,
    flt_dif: float = 0.0001,
    check_names: bool = True,
    ref_file_format: Optional[str] = None,
    cmp_file_format: Optional[str] = None,
) -> bool:
    """
    Compare two endmember tables (:class:`pandas.DataFrame` objects or file
    paths) for shape, optional class column names, and element-wise equality
    within ``flt_dif``.

    :param ref_endmembers: Reference endmembers DataFrame or file path.
    :param cmp_endmembers: Comparison endmembers DataFrame or file path.
    :param flt_dif: Maximum allowed absolute floating-point difference
                    (Default = ``0.0001``).
    :param check_names: If ``True``, also require matching endmember column names
                        in identical order (Default = ``True``).
    :param ref_file_format: Optional format override for ``ref_endmembers``.
    :param cmp_file_format: Optional format override for ``cmp_endmembers``.
    :return: ``True`` if shapes (and optionally column names) match and all
             entries differ by at most ``flt_dif``, ``False`` otherwise.

    .. code:: python

        import rsgispy.imagespectral.endmembers as em

        same = em.are_endmembers_equal("em_ref.csv", "em_cmp.parquet")
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    ref_df = coerce_endmembers_df(ref_endmembers, file_format=ref_file_format)
    cmp_df = coerce_endmembers_df(cmp_endmembers, file_format=cmp_file_format)

    if ref_df.shape != cmp_df.shape:
        return False
    if check_names and list(ref_df.columns) != list(cmp_df.columns):
        return False

    return bool(
        np.all(
            np.abs(
                ref_df.to_numpy(dtype=np.float64)
                - cmp_df.to_numpy(dtype=np.float64)
            )
            <= flt_dif
        )
    )


def _feature_window(
    geom_bounds: Tuple[float, float, float, float],
    transform: rasterio.Affine,
    width: int,
    height: int,
) -> Window:
    """Compute a clipped integer pixel Window covering ``geom_bounds``."""
    min_x, min_y, max_x, max_y = geom_bounds
    raw = rasterio.windows.from_bounds(
        min_x, min_y, max_x, max_y, transform=transform
    )
    c0 = max(0, int(math.floor(raw.col_off)))
    r0 = max(0, int(math.floor(raw.row_off)))
    c1 = min(width, int(math.ceil(raw.col_off + raw.width)))
    r1 = min(height, int(math.ceil(raw.row_off + raw.height)))
    return Window(c0, r0, max(0, c1 - c0), max(0, r1 - r0))


def extract_avg_endmembers(
    input_img: str,
    vec_file: VectorInput,
    vec_lyr: Optional[str] = None,
    out_file: Optional[Union[str, os.PathLike]] = None,
    class_col: Optional[str] = None,
    endmember_names: Optional[Sequence[str]] = None,
    file_format: Optional[str] = None,
    pxl_in_poly_method: int = 1,
) -> pd.DataFrame:
    """
    Extract the mean spectral signature per class/polygon in ``vec_file`` and
    return a :class:`pandas.DataFrame` (bands as rows, class names as columns).

    When ``class_col`` is provided, polygons sharing the same class name are
    aggregated together to compute the class mean spectrum. When ``out_file``
    is provided, the DataFrame is also written to disk using
    :func:`write_endmembers_df`.

    :param input_img: Path to the input multi-band raster image file.
    :param vec_file: Path to the input vector polygon file or an in-memory
                     :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Name of the vector layer (or ``None`` for default layer).
    :param out_file: Optional output file path in any format supported by
                     :func:`write_endmembers_df`.
    :param class_col: Optional attribute column in ``vec_file`` providing the
                      endmember class name for each polygon.
    :param endmember_names: Optional explicit sequence of class names matching
                            the polygon count (used when ``class_col`` is
                            ``None``).
    :param file_format: Optional format override when ``out_file`` is provided.
    :param pxl_in_poly_method: Pixel-in-polygon selection constant from
                               :mod:`rsgispy.zonalstats` (``METHOD_*``).
                               Default = ``1`` (pixel centre within polygon).
    :return: :class:`pandas.DataFrame` of shape ``(n_bands, n_classes)``.

    .. code:: python

        import rsgispy.imagespectral.endmembers as em
        import rsgispy.zonalstats as zs

        em_df = em.extract_avg_endmembers(
            "image.tif",
            "rois.gpkg",
            "rois",
            out_file="endmembers.csv",
            class_col="landcover",
            pxl_in_poly_method=zs.METHOD_POLYCONTAINSPIXELCENTER,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    ensure_file_exists(input_img, "Input image")

    gdf = read_vec_gdf(vec_file, vec_lyr)
    if gdf.empty:
        vec_desc = (
            vec_lyr
            or (
                "<GeoDataFrame>"
                if isinstance(vec_file, gpd.GeoDataFrame)
                else vec_file
            )
        )
        raise rsgispy.RSGISPyException(
            f"Vector layer '{vec_desc}' contains no features."
        )

    if class_col is not None:
        if class_col not in gdf.columns:
            raise rsgispy.RSGISPyException(
                f"Column '{class_col}' was not found in vector layer."
            )
        poly_labels = [str(val) for val in gdf[class_col].tolist()]
    elif endmember_names is not None:
        if len(endmember_names) != len(gdf):
            raise rsgispy.RSGISPyException(
                f"Length of endmember_names ({len(endmember_names)}) does not "
                f"match feature count ({len(gdf)})."
            )
        poly_labels = [str(name) for name in endmember_names]
    else:
        poly_labels = [f"endmember_{i + 1}" for i in range(len(gdf))]

    all_touched = pxl_in_poly_method in (2, 3, 7, 8, 9)

    with rasterio.open(input_img, "r") as src:
        img_crs = (
            pyproj.CRS.from_user_input(src.crs)
            if src.crs is not None
            else None
        )
        geom_series: gpd.GeoSeries = gdf.geometry
        if gdf.crs is not None and img_crs is not None:
            vec_crs = resolve_vector_crs(gdf)
            if vec_crs != img_crs:
                geom_series = geom_series.to_crs(img_crs)
                vec_bbox = reproj_bbox(get_gdf_bbox(gdf), vec_crs, img_crs)
            else:
                vec_bbox = get_gdf_bbox(gdf)
        else:
            vec_bbox = get_gdf_bbox(gdf)

        bounds = src.bounds
        img_bbox = (
            float(bounds.left),
            float(bounds.right),
            float(bounds.bottom),
            float(bounds.top),
        )
        if not do_bboxes_intersect(img_bbox, vec_bbox):
            raise rsgispy.RSGISPyException(
                "The vector layer and image do not overlap."
            )

        descs = src.descriptions or ()
        if len(descs) == src.count and all(
            d is not None and str(d).strip() for d in descs
        ):
            band_labels = [str(d).strip() for d in descs]
        else:
            band_labels = [f"band_{i + 1}" for i in range(src.count)]

        nodata_val = src.nodata
        class_pxls: Dict[str, List[np.ndarray]] = {}
        class_order: List[str] = []

        for idx, (geom, cls_lbl) in enumerate(zip(geom_series, poly_labels)):
            if geom is None or geom.is_empty:
                raise rsgispy.RSGISPyException(
                    f"Polygon feature at index {idx} is empty or null."
                )

            win = _feature_window(
                geom.bounds, src.transform, src.width, src.height
            )
            if win.width <= 0 or win.height <= 0:
                raise rsgispy.RSGISPyException(
                    f"Polygon feature at index {idx} does not intersect any raster pixels."
                )

            arr = src.read(window=win).astype(np.float64, copy=False)
            sub_transform = rasterio.windows.transform(win, src.transform)
            rv = rasterize(
                [(geom, 1)],
                out_shape=(int(win.height), int(win.width)),
                transform=sub_transform,
                fill=0,
                dtype="uint8",
                all_touched=all_touched,
            )
            mask = rv == 1
            if not np.any(mask) and pxl_in_poly_method == 6:
                rv = rasterize(
                    [(geom, 1)],
                    out_shape=(int(win.height), int(win.width)),
                    transform=sub_transform,
                    fill=0,
                    dtype="uint8",
                    all_touched=True,
                )
                mask = rv == 1

            pxls = arr[:, mask].T  # (n_pxls, n_bands)
            vld = np.all(np.isfinite(pxls), axis=1)
            if nodata_val is not None and not (
                isinstance(nodata_val, float) and math.isnan(nodata_val)
            ):
                vld &= np.all(pxls != nodata_val, axis=1)
            pxls = pxls[vld]

            if pxls.shape[0] == 0 and pxl_in_poly_method in (1, 5, 6):
                centroid = geom.centroid
                row, col = src.index(centroid.x, centroid.y)
                if 0 <= row < src.height and 0 <= col < src.width:
                    cent_win = Window(int(col), int(row), 1, 1)
                    c_arr = (
                        src.read(window=cent_win).astype(np.float64).ravel()
                    )
                    c_vld = bool(np.all(np.isfinite(c_arr)))
                    if (
                        c_vld
                        and nodata_val is not None
                        and not (
                            isinstance(nodata_val, float)
                            and math.isnan(nodata_val)
                        )
                    ):
                        c_vld = bool(np.all(c_arr != nodata_val))
                    if c_vld:
                        pxls = c_arr[np.newaxis, :]

            if pxls.shape[0] == 0:
                raise rsgispy.RSGISPyException(
                    f"No valid pixels fell within polygon feature at index {idx}."
                )

            if cls_lbl not in class_pxls:
                class_pxls[cls_lbl] = []
                class_order.append(cls_lbl)
            class_pxls[cls_lbl].append(pxls)

    em_cols: Dict[str, np.ndarray] = {
        cls_lbl: np.mean(np.vstack(class_pxls[cls_lbl]), axis=0)
        for cls_lbl in class_order
    }
    em_df = _normalise_endmembers_df(
        pd.DataFrame(em_cols, index=pd.Index(band_labels, name="band"))
    )
    if out_file is not None:
        write_endmembers_df(em_df, out_file, file_format=file_format)
    return em_df


def calc_ppi(
    input_img: str,
    output_img: str,
    gdalformat: str = "GTIFF",
    niters: int = 1000,
    lthres: float = 0,
    uthres: float = 0,
    img_gain: float = 1,
    seed: Optional[int] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Calculate the Pixel Purity Index (PPI) using random linear projections of
    mean-centred spectral bands (Boardman et al., 1995).

    :param input_img: Path to the input multi-band raster image file.
    :param output_img: Path to the output ``uint16`` PPI count raster file.
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param niters: Number of random skewer projections (Default = ``1000``).
    :param lthres: Lower extreme threshold distance in scaled image units
                   (Default = ``0``).
    :param uthres: Upper extreme threshold distance in scaled image units
                   (Default = ``0``).
    :param img_gain: Positive gain divisor applied to input reflectance values
                     when not equal to ``1`` (Default = ``1``).
    :param seed: Optional random seed for reproducible projections.
    :param calc_stats: If ``True``, compute statistics and pyramids on
                       ``output_img`` (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar during iterations
                     (Default = ``True``).

    .. code:: python

        import rsgispy.imagespectral.endmembers as em

        em.calc_ppi(
            "pca_img.tif",
            "ppi_out.tif",
            gdalformat="GTIFF",
            niters=500,
            seed=42,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    ensure_file_exists(input_img, "Input image")
    if niters < 1:
        raise rsgispy.RSGISPyException("niters must be at least 1.")
    if img_gain <= 0:
        raise rsgispy.RSGISPyException("img_gain must be positive.")

    pbar_cls = rsgispy.get_tqdm()
    with ExitStack() as stack:
        src = stack.enter_context(rasterio.open(input_img, "r"))
        n_bands = src.count
        height = src.height
        width = src.width
        n_pxls = height * width
        nodatavals = src.nodatavals

        img_data = np.zeros((n_bands, n_pxls), dtype=np.float32)
        img_msk = np.ones(n_pxls, dtype=bool)

        for b_idx in pbar_cls(
            range(n_bands), desc="Importing bands", disable=not use_pbar
        ):
            band_arr = src.read(b_idx + 1).ravel().astype(np.float32, copy=False)
            nd = nodatavals[b_idx]
            b_vld = np.isfinite(band_arr)
            if nd is not None and not (
                isinstance(nd, float) and math.isnan(nd)
            ):
                b_vld &= band_arr != nd
            img_msk &= b_vld

            if img_gain != 1.0:
                band_arr = band_arr / float(img_gain)
            band_tmp = np.where(b_vld, band_arr, np.nan)
            band_mean = float(np.nanmean(band_tmp)) if np.any(b_vld) else 0.0
            img_data[b_idx] = band_arr - band_mean

        vld_data = np.ascontiguousarray(img_data[:, img_msk].T)
        counts = np.zeros(vld_data.shape[0], dtype=np.uint16)

        if vld_data.shape[0] > 0:
            rng = np.random.default_rng(seed)
            batch_size = 256
            batch_starts = range(0, niters, batch_size)
            for b_start in pbar_cls(
                batch_starts, desc="Calculating PPI", disable=not use_pbar
            ):
                cur_batch = min(batch_size, niters - b_start)
                skewers = (
                    rng.random((cur_batch, n_bands)).T.astype(
                        np.float32, copy=False
                    )
                    - np.float32(0.5)
                )
                proj = vld_data @ skewers  # (n_valid, cur_batch)
                if lthres == 0:
                    min_idx = np.argmin(proj, axis=0)
                    np.add.at(counts, min_idx, 1)
                else:
                    s_min = np.min(proj, axis=0, keepdims=True)
                    counts += np.sum(
                        proj <= (s_min + lthres), axis=1
                    ).astype(np.uint16)

                if uthres == 0:
                    max_idx = np.argmax(proj, axis=0)
                    np.add.at(counts, max_idx, 1)
                else:
                    s_max = np.max(proj, axis=0, keepdims=True)
                    counts += np.sum(
                        proj >= (s_max - uthres), axis=1
                    ).astype(np.uint16)

        out_full = np.zeros(n_pxls, dtype=np.uint16)
        out_full[img_msk] = counts
        out_img_arr = out_full.reshape((height, width))

        dst = create_output_from_dataset(
            stack=stack,
            src=src,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype="uint16",
            count=1,
        )
        dst.write(out_img_arr, 1)

    if calc_stats:
        pop_img_stats(
            output_img,
            no_data_val=0,
            calc_pyramids=True,
            use_pbar=use_pbar,
        )
