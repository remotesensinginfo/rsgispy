# pylint: disable=too-many-lines
"""
The imagecalc.spectralunmixing module provides linear spectral unmixing,
endmember extraction/comparison, pixel purity index (PPI), error evaluation,
and multi-endmember unmixing summary utilities using ``rasterio``, ``numpy``,
and ``pandas``.

Endmembers are stored and passed as :class:`pandas.DataFrame` tables (in
memory or read from / written to disk in the user's chosen format such as
CSV, TSV, Parquet, Feather, Excel, JSON, Pickle, or HDF5) with **spectral
bands as rows and endmember class names as columns**.

**Recommended unmixing functions:**

* :func:`spec_unmix_ucls` — vectorised unconstrained least squares (core
  ``numpy`` only)
* :func:`spec_unmix_nnls` — direct Lawson–Hanson non-negative least squares
  (requires ``scipy``)
* :func:`spec_unmix_fcls` — scale-invariant fully constrained least squares
  (non-negative and sum-to-one, requires ``scipy``)

Functions wrapping ``pysptools``, ``cvxopt``, and ``pymcr`` are also retained
and exported when those optional dependencies are installed.
"""

from contextlib import ExitStack
import json
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple, Union

import geopandas
import numpy as np
import pandas
import pyproj
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
import rasterio.windows

import rsgispy
from rsgispy.imageprocess import (
    cast_result_block,
    compute_spatial_intersection,
    create_output_from_dataset,
    ensure_file_exists,
    iter_dataset_windows,
    open_multi_image_datasets,
    open_output_dataset,
)
from rsgispy.imageutils.metadata import set_band_names
from rsgispy.imageutils.stats import pop_img_stats, pop_thmt_img_stats
from rsgispy.tools.geometry import do_bboxes_intersect, reproj_bbox
from rsgispy.vectorprocess import get_gdf_bbox, read_vec_gdf, resolve_vector_crs

try:
    import scipy.optimize

    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

try:
    import pysptools.abundance_maps.amaps as _spts_amaps  # pylint: disable=import-error

    HAVE_PYSPTOOLS = True
except ImportError:
    _spts_amaps = None
    HAVE_PYSPTOOLS = False

try:
    import cvxopt as _cvxopt  # noqa: F401  # pylint: disable=import-error,unused-import

    HAVE_CVXOPT = True
except ImportError:
    HAVE_CVXOPT = False

try:
    from pymcr.constraints import (  # pylint: disable=import-error
        ConstraintNonneg,
        ConstraintNorm,
    )
    from pymcr.mcr import McrAR  # pylint: disable=import-error
    from pymcr.regressors import NNLS as _PyMcrNNLS  # pylint: disable=import-error

    HAVE_PYMCR = True
except ImportError:
    HAVE_PYMCR = False

EndmemberInput = Union[pandas.DataFrame, str, os.PathLike]

_EXT_TO_FORMAT: Dict[str, str] = {
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

_FORMAT_ALIASES: Dict[str, str] = {
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

__all__: List[str] = [
    "ImageEndmemberInfo",
    "extract_avg_endmembers",
    "read_endmembers_df",
    "write_endmembers_df",
    "convert_mtxt_to_df",
    "plot_endmembers",
    "are_endmembers_equal",
    "spec_unmix_ucls",
    "exhcon_linear_spec_unmix",
    "rescale_unmixing_results",
    "predict_refl_linear_unmixing",
    "calc_unmixing_rmse_residual_err",
    "summarise_multi_endmember_linear_unmixing",
    "calc_ppi",
]

if HAVE_SCIPY:
    __all__.extend(["spec_unmix_nnls", "spec_unmix_fcls"])

if HAVE_PYSPTOOLS:
    __all__.extend(["spec_unmix_spts_ucls", "spec_unmix_spts_nnls"])
    if HAVE_CVXOPT:
        __all__.append("spec_unmix_spts_fcls")

if HAVE_PYMCR:
    __all__.extend(["spec_unmix_pymcr_nnls", "spec_unmix_pymcr_fcls"])


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

        import rsgispy.imagecalc.spectralunmixing as su

        info = su.ImageEndmemberInfo(
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
            self.endmember_names: Optional[List[str]] = [str(n) for n in endmember_names]
        elif isinstance(endmembers, pandas.DataFrame):
            self.endmember_names = [str(c) for c in endmembers.columns]
        else:
            self.endmember_names = None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ImageEndmemberInfo):
            return NotImplemented
        if isinstance(self.endmembers, pandas.DataFrame) or isinstance(
            other.endmembers, pandas.DataFrame
        ):
            if not (
                isinstance(self.endmembers, pandas.DataFrame)
                and isinstance(other.endmembers, pandas.DataFrame)
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
    if file_format is not None:
        key = file_format.strip().lower().lstrip(".")
        if key in _FORMAT_ALIASES:
            return _FORMAT_ALIASES[key]
        raise rsgispy.RSGISPyException(
            f"Unsupported endmember file format '{file_format}'. Supported "
            f"formats: {sorted(set(_FORMAT_ALIASES.values()))}."
        )
    ext = os.path.splitext(os.fspath(file_path))[1].lower()
    if ext in _EXT_TO_FORMAT:
        return _EXT_TO_FORMAT[ext]
    raise rsgispy.RSGISPyException(
        f"Cannot infer endmember file format from extension '{ext}' in "
        f"'{os.fspath(file_path)}'. Specify 'file_format' explicitly."
    )


def _normalise_endmembers_df(df: pandas.DataFrame) -> pandas.DataFrame:
    """Validate and normalise an endmembers DataFrame (rows=bands, cols=classes)."""
    if not isinstance(df, pandas.DataFrame) or df.empty:
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
            out[col] = pandas.to_numeric(out[col], errors="raise").astype(np.float64)
    except (ValueError, TypeError) as err:
        raise rsgispy.RSGISPyException(
            f"All endmember class columns must contain numeric values: {err}"
        ) from err
    if out.index.name is None:
        out.index = pandas.Index(out.index, name="band")
    return out


def _promote_band_index_if_present(df: pandas.DataFrame) -> pandas.DataFrame:
    """
    Detect an explicit band/unnamed index column loaded from CSV/TSV/Excel and
    restore it as the DataFrame index; keep all columns when none is present.
    """
    if df.shape[1] >= 2:
        first_col = str(df.columns[0])
        is_idx_name = first_col.lower() in ("band", "__band__") or first_col.startswith(
            "Unnamed:"
        )
        is_non_numeric = not pandas.api.types.is_numeric_dtype(df.iloc[:, 0])
        if is_idx_name or is_non_numeric:
            idx_vals = df.iloc[:, 0].astype(str)
            rest = df.iloc[:, 1:].copy()
            rest.index = pandas.Index(idx_vals, name="band")
            return rest
    out = df.copy()
    out.index = pandas.Index(
        [f"band_{i + 1}" for i in range(len(out))], name="band"
    )
    return out


def read_endmembers_df(
    endmembers_file: Union[str, os.PathLike],
    file_format: Optional[str] = None,
    gain: float = 1,
    sheet_name: Union[int, str] = 0,
) -> pandas.DataFrame:
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

        import rsgispy.imagecalc.spectralunmixing as su

        em_df = su.read_endmembers_df("endmembers.csv")
        em_pq = su.read_endmembers_df("endmembers.parquet")
    """
    # pylint: disable=too-many-branches
    path_str = os.fspath(endmembers_file)
    ensure_file_exists(path_str, "Endmembers file")
    fmt = _resolve_table_format(path_str, file_format)

    if fmt == "csv":
        raw_df = pandas.read_csv(path_str)
        df = _promote_band_index_if_present(raw_df)
    elif fmt == "tsv":
        raw_df = pandas.read_csv(path_str, sep="\t")
        df = _promote_band_index_if_present(raw_df)
    elif fmt == "parquet":
        try:
            df = pandas.read_parquet(path_str)
        except ImportError as err:
            raise rsgispy.RSGISPyException(
                "Reading Parquet endmember files requires 'pyarrow' or 'fastparquet'."
            ) from err
    elif fmt == "feather":
        try:
            raw_df = pandas.read_feather(path_str)
        except ImportError as err:
            raise rsgispy.RSGISPyException(
                "Reading Feather endmember files requires 'pyarrow'."
            ) from err
        if "__band__" in raw_df.columns:
            idx = pandas.Index(raw_df["__band__"].astype(str), name="band")
            df = raw_df.drop(columns=["__band__"]).set_index(idx)
        else:
            df = raw_df
    elif fmt == "excel":
        try:
            raw_df = pandas.read_excel(path_str, sheet_name=sheet_name)
        except ImportError as err:
            raise rsgispy.RSGISPyException(
                "Reading Excel endmember files requires 'openpyxl'."
            ) from err
        df = _promote_band_index_if_present(raw_df)
    elif fmt == "json":
        with open(path_str, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict) and {"columns", "data"} <= set(payload.keys()):
            df = pandas.DataFrame(
                payload["data"],
                columns=payload["columns"],
                index=pandas.Index(payload["index"], name="band")
                if "index" in payload
                else None,
            )
        else:
            df = pandas.read_json(path_str, orient="columns")
    elif fmt == "pickle":
        df = pandas.read_pickle(path_str)
    else:  # hdf
        try:
            df = pandas.read_hdf(path_str)
        except ImportError as err:
            raise rsgispy.RSGISPyException(
                "Reading HDF5 endmember files requires 'tables' (PyTables)."
            ) from err

    out_df = _normalise_endmembers_df(df)
    if gain > 1:
        out_df = out_df / float(gain)
    return out_df


def write_endmembers_df(
    endmembers_df: pandas.DataFrame,
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
        import rsgispy.imagecalc.spectralunmixing as su

        em_df = pd.DataFrame(
            {"Veg": [0.05, 0.45, 0.20], "Soil": [0.30, 0.35, 0.40]},
            index=["blue", "nir", "swir1"],
        )
        su.write_endmembers_df(em_df, "endmembers.csv")
        su.write_endmembers_df(em_df, "endmembers.parquet")
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
            df.to_excel(path_str, sheet_name=sheet_name, index=True, engine="openpyxl")
        except ImportError:
            try:
                df.to_excel(
                    path_str, sheet_name=sheet_name, index=True, engine="xlsxwriter"
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
) -> pandas.DataFrame:
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

        import rsgispy.imagecalc.spectralunmixing as su

        em_df = su.convert_mtxt_to_df(
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
        row_idx = pandas.Index([str(b) for b in band_names], name="band")
    else:
        row_idx = pandas.Index([f"band_{i + 1}" for i in range(n)], name="band")

    df = _normalise_endmembers_df(
        pandas.DataFrame(end_mem_arr.T, index=row_idx, columns=col_names)
    )
    if out_file is not None:
        write_endmembers_df(df, out_file, file_format=file_format)
    return df


def _coerce_endmembers_df(
    endmembers: EndmemberInput,
    gain: float = 1,
    file_format: Optional[str] = None,
) -> pandas.DataFrame:
    """Resolve an in-memory DataFrame or file path into a validated DataFrame."""
    if isinstance(endmembers, pandas.DataFrame):
        df = _normalise_endmembers_df(endmembers)
        if gain > 1:
            df = df / float(gain)
        return df
    return read_endmembers_df(endmembers, file_format=file_format, gain=gain)


def _prepare_endmembers_array(
    endmembers: EndmemberInput,
    gain: float = 1,
    weight: Optional[float] = None,
    file_format: Optional[str] = None,
) -> Tuple[List[str], int, int, np.ndarray]:
    """
    Convert an endmembers DataFrame or file into solver matrix shape
    ``(m_endmembers, n_bands [+ 1 if weight])`` plus class names.
    """
    df = _coerce_endmembers_df(endmembers, gain=gain, file_format=file_format)
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

        import rsgispy.imagecalc.spectralunmixing as su

        su.plot_endmembers(
            "endmembers.csv",
            "endmembers.png",
            plot_title="Endmember Signatures",
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    # pylint: disable=import-outside-toplevel
    import matplotlib.pyplot as plt

    df = _coerce_endmembers_df(endmembers, gain=gain, file_format=file_format)
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

        import rsgispy.imagecalc.spectralunmixing as su

        same = su.are_endmembers_equal("em_ref.csv", "em_cmp.parquet")
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    ref_df = _coerce_endmembers_df(ref_endmembers, file_format=ref_file_format)
    cmp_df = _coerce_endmembers_df(cmp_endmembers, file_format=cmp_file_format)

    if ref_df.shape != cmp_df.shape:
        return False
    if check_names and list(ref_df.columns) != list(cmp_df.columns):
        return False

    return bool(
        np.all(
            np.abs(
                ref_df.to_numpy(dtype=np.float64) - cmp_df.to_numpy(dtype=np.float64)
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
    raw = rasterio.windows.from_bounds(min_x, min_y, max_x, max_y, transform=transform)
    c0 = max(0, int(math.floor(raw.col_off)))
    r0 = max(0, int(math.floor(raw.row_off)))
    c1 = min(width, int(math.ceil(raw.col_off + raw.width)))
    r1 = min(height, int(math.ceil(raw.row_off + raw.height)))
    return Window(c0, r0, max(0, c1 - c0), max(0, r1 - r0))


def extract_avg_endmembers(
    input_img: str,
    vec_file: str,
    vec_lyr: Optional[str] = None,
    out_file: Optional[Union[str, os.PathLike]] = None,
    class_col: Optional[str] = None,
    endmember_names: Optional[Sequence[str]] = None,
    file_format: Optional[str] = None,
    pxl_in_poly_method: int = 1,
) -> pandas.DataFrame:
    """
    Extract the mean spectral signature per class/polygon in ``vec_file`` and
    return a :class:`pandas.DataFrame` (bands as rows, class names as columns).

    When ``class_col`` is provided, polygons sharing the same class name are
    aggregated together to compute the class mean spectrum. When ``out_file``
    is provided, the DataFrame is also written to disk using
    :func:`write_endmembers_df`.

    :param input_img: Path to the input multi-band raster image file.
    :param vec_file: Path to the input vector polygon file.
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

        import rsgispy.imagecalc.spectralunmixing as su
        import rsgispy.zonalstats as zs

        em_df = su.extract_avg_endmembers(
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
    ensure_file_exists(vec_file, "Input vector")

    gdf = read_vec_gdf(vec_file, vec_lyr)
    if gdf.empty:
        raise rsgispy.RSGISPyException(
            f"Vector layer '{vec_lyr or vec_file}' contains no features."
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
        img_crs = pyproj.CRS.from_user_input(src.crs) if src.crs is not None else None
        geom_series: geopandas.GeoSeries = gdf.geometry
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
            raise rsgispy.RSGISPyException("The vector layer and image do not overlap.")

        descs = src.descriptions or ()
        if len(descs) == src.count and all(d is not None and str(d).strip() for d in descs):
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

            win = _feature_window(geom.bounds, src.transform, src.width, src.height)
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
                    c_arr = src.read(window=cent_win).astype(np.float64).ravel()
                    if np.all(np.isfinite(c_arr)):
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
        pandas.DataFrame(em_cols, index=pandas.Index(band_labels, name="band"))
    )
    if out_file is not None:
        write_endmembers_df(em_df, out_file, file_format=file_format)
    return em_df


def _build_simplex_weights(n_endmembers: int, step_res: float) -> np.ndarray:
    """
    Generate all non-negative endmember weight combinations summing to 1.0
    on a simplex grid of resolution ``step_res``.
    """
    if step_res <= 0.0 or step_res > 1.0:
        raise rsgispy.RSGISPyException(
            f"step_res must be in (0, 1], got {step_res}."
        )

    n_steps = int(round(1.0 / step_res))
    if math.isclose(n_steps * step_res, 1.0, rel_tol=1e-6, abs_tol=1e-7):
        rows: List[Tuple[int, ...]] = []

        def _gen_int(pos: int, remaining: int, cur: List[int]) -> None:
            if pos == n_endmembers - 1:
                rows.append(tuple(cur + [remaining]))
                return
            for val in range(remaining + 1):
                cur.append(val)
                _gen_int(pos + 1, remaining - val, cur)
                cur.pop()

        _gen_int(0, n_steps, [])
        return np.asarray(rows, dtype=np.float64) / float(n_steps)

    grid_vals: List[List[float]] = []
    steps = np.arange(0.0, 1.0 + 1e-9, step_res)

    def _gen_flt(pos: int, rem: float, cur: List[float]) -> None:
        if pos == n_endmembers - 1:
            if -1e-6 <= rem <= 1.0 + 1e-6:
                grid_vals.append(cur + [max(0.0, rem)])
            return
        for v in steps:
            if v <= rem + 1e-6:
                cur.append(float(v))
                _gen_flt(pos + 1, rem - float(v), cur)
                cur.pop()

    _gen_flt(0, 1.0, [])
    if not grid_vals:
        raise rsgispy.RSGISPyException(
            f"No valid simplex weights generated for step_res={step_res}."
        )
    return np.asarray(grid_vals, dtype=np.float64)


def exhcon_linear_spec_unmix(
    input_img: str,
    output_img: str,
    gdalformat: str = "GTIFF",
    datatype: int = rsgispy.TYPE_32FLOAT,
    endmembers: Optional[EndmemberInput] = None,
    step_res: float = 0.1,
    gain: float = 1.0,
    offset: float = 0.0,
    file_format: Optional[str] = None,
    use_pbar: bool = True,
) -> None:
    """
    Perform exhaustive constrained linear spectral unmixing of ``input_img``.

    Both pixel spectra and endmember spectra are L2-normalised prior to
    searching all non-negative abundance combinations summing to 1.0 at grid
    step ``step_res``. The output raster contains ``m + 1`` bands named after
    the endmember columns followed by ``RMSE`` (scaled via
    ``offset + gain * value``; zero-norm / invalid pixels are written as ``0``).

    :param input_img: Path to the input multi-band raster image file.
    :param output_img: Path to the output unmixed raster image file.
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param datatype: ``rsgispy.TYPE_*`` output data type constant
                     (Default = ``rsgispy.TYPE_32FLOAT``).
    :param endmembers: :class:`pandas.DataFrame` (columns = class names,
                       rows = bands) or file path readable by
                       :func:`read_endmembers_df`.
    :param step_res: Search step resolution in ``(0, 1]`` (Default = ``0.1``).
    :param gain: Multiplicative gain applied to output values
                 (Default = ``1.0``).
    :param offset: Additive offset applied to output values (Default = ``0.0``).
    :param file_format: Optional format override when ``endmembers`` is a file
                        path (Default = ``None``).
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).

    .. code:: python

        import rsgispy
        import rsgispy.imagecalc.spectralunmixing as su

        su.exhcon_linear_spec_unmix(
            input_img="s2_refl.tif",
            output_img="unmixed.tif",
            gdalformat="GTIFF",
            datatype=rsgispy.TYPE_32FLOAT,
            endmembers="endmembers.csv",
            step_res=0.1,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    ensure_file_exists(input_img, "Input image")
    if endmembers is None:
        raise rsgispy.RSGISPyException("endmembers must be provided.")

    em_names, m, n_bands, em_arr = _prepare_endmembers_array(
        endmembers, file_format=file_format
    )
    em_norms = np.linalg.norm(em_arr, axis=1, keepdims=True)
    if np.any(em_norms == 0):
        raise rsgispy.RSGISPyException(
            "One or more endmember spectra have zero norm."
        )
    em_normed = em_arr / em_norms

    cand_weights = _build_simplex_weights(m, float(step_res))  # (C, m)
    cand_spectra = cand_weights @ em_normed  # (C, n_bands)
    cand_sq_norms = np.sum(cand_spectra * cand_spectra, axis=1)  # (C,)
    cand_spectra_t = cand_spectra.T  # (n_bands, C)

    out_rio_dtype = rsgispy.get_rasterio_datatype(datatype)
    out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(datatype))
    n_out_bands = m + 1
    chunk_size = 8192

    with ExitStack() as stack:
        src = stack.enter_context(rasterio.open(input_img, "r"))
        if src.count != n_bands:
            raise rsgispy.RSGISPyException(
                f"Input image '{input_img}' has {src.count} bands, but "
                f"endmembers expect {n_bands} bands."
            )

        dst = create_output_from_dataset(
            stack=stack,
            src=src,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype=out_rio_dtype,
            count=n_out_bands,
        )
        for b_idx, b_name in enumerate(em_names + ["RMSE"], start=1):
            dst.set_band_description(b_idx, b_name)

        for win in iter_dataset_windows(src.height, src.width, use_pbar=use_pbar):
            win_h = int(win.height)
            win_w = int(win.width)
            blk = src.read(window=win).astype(np.float64, copy=False)
            flat = np.moveaxis(blk, 0, -1).reshape(-1, n_bands)

            norms = np.linalg.norm(flat, axis=1)
            valid = np.all(np.isfinite(flat), axis=1) & (norms > 0.0)
            out_flat = np.zeros((flat.shape[0], n_out_bands), dtype=np.float64)

            if np.any(valid):
                v_idx = np.flatnonzero(valid)
                pxl_normed = flat[v_idx] / norms[v_idx, np.newaxis]
                pxl_sq = np.sum(pxl_normed * pxl_normed, axis=1)

                n_valid = v_idx.size
                for start in range(0, n_valid, chunk_size):
                    end = min(start + chunk_size, n_valid)
                    sub_p = pxl_normed[start:end]
                    sub_sq = pxl_sq[start:end, np.newaxis] + cand_sq_norms[np.newaxis, :]
                    sq_dists = sub_sq - 2.0 * (sub_p @ cand_spectra_t)
                    best_c = np.argmin(sq_dists, axis=1)
                    min_sq = np.maximum(0.0, sq_dists[np.arange(end - start), best_c])
                    rmse = np.sqrt(min_sq / float(n_bands))

                    out_idx = v_idx[start:end]
                    out_flat[out_idx, :m] = offset + (gain * cand_weights[best_c])
                    out_flat[out_idx, m] = offset + (gain * rmse)

            out_blk = np.moveaxis(
                out_flat.reshape((win_h, win_w, n_out_bands)), -1, 0
            )
            dst.write(
                cast_result_block(
                    out_blk, win_h, win_w, out_np_dtype, clamp_int=False
                ),
                window=win,
            )


def _run_windowed_unmixer(
    input_img: str,
    valid_msk_img: Optional[str],
    valid_msk_val: int,
    output_img: str,
    endmembers: EndmemberInput,
    solver_fn,
    gdalformat: str = "GTIFF",
    gain: float = 1,
    weight: Optional[float] = None,
    file_format: Optional[str] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """Shared windowed linear spectral unmixing runner."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-branches
    em_names, m, _, endmembers_arr = _prepare_endmembers_array(
        endmembers, gain=gain, weight=weight, file_format=file_format
    )

    in_paths = [input_img] if valid_msk_img is None else [input_img, valid_msk_img]
    with ExitStack() as stack:
        datasets, paths = open_multi_image_datasets(stack, in_paths)
        src_img = datasets[paths[0]]
        src_msk = datasets[paths[1]] if valid_msk_img is not None else None
        grid = compute_spatial_intersection(datasets)

        expected_bands = endmembers_arr.shape[1] - (1 if weight is not None else 0)
        if src_img.count != expected_bands:
            raise rsgispy.RSGISPyException(
                f"Input image '{input_img}' has {src_img.count} bands, expected "
                f"{expected_bands}."
            )

        dst = open_output_dataset(
            stack=stack,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype="float32",
            grid=grid,
            count=m,
        )
        for b_idx, b_name in enumerate(em_names, start=1):
            dst.set_band_description(b_idx, b_name)

        img_xoff, img_yoff = grid.offsets[paths[0]]
        msk_xoff, msk_yoff = (
            grid.offsets[paths[1]] if valid_msk_img is not None else (0, 0)
        )

        for win in iter_dataset_windows(grid.height, grid.width, use_pbar=use_pbar):
            win_h = int(win.height)
            win_w = int(win.width)
            img_win = Window(
                int(win.col_off) + img_xoff,
                int(win.row_off) + img_yoff,
                win_w,
                win_h,
            )

            out_flat = np.zeros((win_h * win_w, m), dtype=np.float32)
            if src_msk is not None:
                msk_win = Window(
                    int(win.col_off) + msk_xoff,
                    int(win.row_off) + msk_yoff,
                    win_w,
                    win_h,
                )
                msk_vld = src_msk.read(1, window=msk_win).ravel() == valid_msk_val
            else:
                msk_vld = np.ones(win_h * win_w, dtype=bool)

            if np.any(msk_vld):
                img_blk = src_img.read(window=img_win).astype(np.float64, copy=False)
                img_flat = np.moveaxis(img_blk, 0, -1).reshape(-1, src_img.count)

                vld = (
                    msk_vld
                    & np.all(np.isfinite(img_flat), axis=1)
                    & np.all(img_flat > 0.0, axis=1)
                )
                if np.any(vld):
                    vld_pxls = img_flat[vld]
                    if gain > 1:
                        vld_pxls = vld_pxls / float(gain)
                    if weight is not None:
                        w_col = np.full(
                            (vld_pxls.shape[0], 1), float(weight), dtype=np.float64
                        )
                        vld_pxls = np.hstack([vld_pxls, w_col])

                    abundances = solver_fn(vld_pxls, endmembers_arr)
                    out_flat[vld] = abundances.astype(np.float32, copy=False)

            out_blk = np.moveaxis(out_flat.reshape((win_h, win_w, m)), -1, 0)
            dst.write(out_blk, window=win)

    if calc_stats:
        pop_img_stats(
            output_img,
            no_data_val=0,
            calc_pyramids=True,
            use_pbar=use_pbar,
        )


def _numpy_ucls_solver(vld_pxls: np.ndarray, endmembers_arr: np.ndarray) -> np.ndarray:
    """Vectorised unconstrained least squares solver using ``numpy.linalg.lstsq``."""
    abundances, _, _, _ = np.linalg.lstsq(
        endmembers_arr.T, vld_pxls.T, rcond=None
    )
    return abundances.T


def spec_unmix_ucls(
    input_img: str,
    valid_msk_img: Optional[str],
    valid_msk_val: int,
    output_img: str,
    endmembers: EndmemberInput,
    gdalformat: str = "GTIFF",
    gain: float = 1,
    weight: Optional[float] = None,
    file_format: Optional[str] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Perform unconstrained least-squares (UCLS) linear spectral
    unmixing using vectorised ``numpy.linalg.lstsq``.

    Requires only core ``numpy`` (no optional unmixing packages needed).

    :param input_img: Path to the input image file.
    :param valid_msk_img: Optional path to the valid pixel mask image file
                          (if ``None``, all finite positive pixels are unmixed).
    :param valid_msk_val: Pixel value in ``valid_msk_img`` marking valid pixels.
    :param output_img: Path to the output unmixed abundance image file.
    :param endmembers: :class:`pandas.DataFrame` (columns = class names,
                       rows = bands) or file path readable by
                       :func:`read_endmembers_df`.
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param gain: Gain divisor applied to image and endmember spectra when
                 greater than ``1`` (Default = ``1``).
    :param weight: Optional Scarth et al. (2010) sum-to-one weight.
    :param file_format: Optional format override when ``endmembers`` is a file
                        path (Default = ``None``).
    :param calc_stats: If ``True``, populate image statistics and pyramids
                       (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).

    .. code:: python

        import rsgispy.imagecalc.spectralunmixing as su

        su.spec_unmix_ucls(
            "img.tif", "mask.tif", 1, "unmix_ucls.tif", "endmembers.csv"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    _run_windowed_unmixer(
        input_img=input_img,
        valid_msk_img=valid_msk_img,
        valid_msk_val=valid_msk_val,
        output_img=output_img,
        endmembers=endmembers,
        solver_fn=_numpy_ucls_solver,
        gdalformat=gdalformat,
        gain=gain,
        weight=weight,
        file_format=file_format,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )


if HAVE_SCIPY:

    def _scipy_nnls_solver(
        vld_pxls: np.ndarray, endmembers_arr: np.ndarray
    ) -> np.ndarray:
        """Direct Lawson–Hanson NNLS solver using ``scipy.optimize.nnls``."""
        end_t = endmembers_arr.T
        n_pxls = vld_pxls.shape[0]
        m = endmembers_arr.shape[0]
        out = np.empty((n_pxls, m), dtype=np.float64)
        for idx in range(n_pxls):
            out[idx] = scipy.optimize.nnls(end_t, vld_pxls[idx])[0]
        return out

    def _scipy_fcls_solver(
        vld_pxls: np.ndarray, endmembers_arr: np.ndarray
    ) -> np.ndarray:
        """
        Scale-invariant Heinz & Chang (2001) FCLS solver using
        ``scipy.optimize.nnls``.
        """
        m = endmembers_arr.shape[0]
        n_pxls = vld_pxls.shape[0]
        scale = max(float(np.max(np.abs(endmembers_arr))), 1e-12)
        delta = 1e-5 / scale
        end_aug = np.vstack(
            [delta * endmembers_arr.T, np.ones((1, m), dtype=np.float64)]
        )
        pxls_aug = np.hstack(
            [delta * vld_pxls, np.ones((n_pxls, 1), dtype=np.float64)]
        )
        out = np.empty((n_pxls, m), dtype=np.float64)
        for idx in range(n_pxls):
            out[idx] = scipy.optimize.nnls(end_aug, pxls_aug[idx])[0]
        out = np.clip(out, 0.0, None)
        row_sums = np.sum(out, axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(row_sums > 0.0, out / row_sums, 0.0)
        return out

    def spec_unmix_nnls(
        input_img: str,
        valid_msk_img: Optional[str],
        valid_msk_val: int,
        output_img: str,
        endmembers: EndmemberInput,
        gdalformat: str = "GTIFF",
        gain: float = 1,
        weight: Optional[float] = None,
        file_format: Optional[str] = None,
        calc_stats: bool = True,
        use_pbar: bool = True,
    ) -> None:
        """
        **[Recommended]** Perform non-negative least-squares (NNLS) linear
        spectral unmixing using ``scipy.optimize.nnls``.

        .. tip::

           This NumPy/SciPy function is the recommended implementation for NNLS
           unmixing in ``rsgispy``. Unlike ``pysptools.abundance_maps.amaps.NNLS``,
           it solves :math:`\\min_{x \\ge 0} \\| E^T x - y \\|_2` directly
           without squaring the system via normal equations, and requires only
           ``scipy`` (no ``pysptools`` or ``pyMCR``).

        :param input_img: Path to the input image file.
        :param valid_msk_img: Optional path to the valid pixel mask image file
                              (if ``None``, all finite positive pixels are
                              unmixed).
        :param valid_msk_val: Pixel value in ``valid_msk_img`` marking valid
                              pixels.
        :param output_img: Path to the output unmixed abundance image file.
        :param endmembers: :class:`pandas.DataFrame` (columns = class names,
                           rows = bands) or file path readable by
                           :func:`read_endmembers_df`.
        :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
        :param gain: Gain divisor applied to image and endmember spectra when
                     greater than ``1`` (Default = ``1``).
        :param weight: Optional Scarth et al. (2010) sum-to-one weight.
        :param file_format: Optional format override when ``endmembers`` is a
                            file path (Default = ``None``).
        :param calc_stats: If ``True``, populate image statistics and pyramids
                           (Default = ``True``).
        :param use_pbar: If ``True``, display a tqdm progress bar during
                         processing (Default = ``True``).

        .. code:: python

            import rsgispy.imagecalc.spectralunmixing as su

            su.spec_unmix_nnls(
                "img.tif", "mask.tif", 1, "unmix_nnls.tif", "endmembers.csv"
            )
        """
        # pylint: disable=too-many-arguments,too-many-positional-arguments
        _run_windowed_unmixer(
            input_img=input_img,
            valid_msk_img=valid_msk_img,
            valid_msk_val=valid_msk_val,
            output_img=output_img,
            endmembers=endmembers,
            solver_fn=_scipy_nnls_solver,
            gdalformat=gdalformat,
            gain=gain,
            weight=weight,
            file_format=file_format,
            calc_stats=calc_stats,
            use_pbar=use_pbar,
        )

    def spec_unmix_fcls(
        input_img: str,
        valid_msk_img: Optional[str],
        valid_msk_val: int,
        output_img: str,
        endmembers: EndmemberInput,
        gdalformat: str = "GTIFF",
        gain: float = 1,
        file_format: Optional[str] = None,
        calc_stats: bool = True,
        use_pbar: bool = True,
    ) -> None:
        """
        **[Recommended]** Perform fully constrained least-squares (FCLS:
        non-negative and sum-to-one) linear spectral unmixing using a
        scale-invariant Heinz & Chang (2001) formulation via
        ``scipy.optimize.nnls``.

        .. tip::

           This NumPy/SciPy function is the recommended implementation for FCLS
           unmixing in ``rsgispy``. It avoids the compiled ``cvxopt`` dependency
           required by :func:`spec_unmix_spts_fcls` while delivering high
           accuracy and substantially faster runtime.

        :param input_img: Path to the input image file.
        :param valid_msk_img: Optional path to the valid pixel mask image file
                              (if ``None``, all finite positive pixels are
                              unmixed).
        :param valid_msk_val: Pixel value in ``valid_msk_img`` marking valid
                              pixels.
        :param output_img: Path to the output unmixed abundance image file.
        :param endmembers: :class:`pandas.DataFrame` (columns = class names,
                           rows = bands) or file path readable by
                           :func:`read_endmembers_df`.
        :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
        :param gain: Gain divisor applied to image and endmember spectra when
                     greater than ``1`` (Default = ``1``).
        :param file_format: Optional format override when ``endmembers`` is a
                            file path (Default = ``None``).
        :param calc_stats: If ``True``, populate image statistics and pyramids
                           (Default = ``True``).
        :param use_pbar: If ``True``, display a tqdm progress bar during
                         processing (Default = ``True``).

        .. code:: python

            import rsgispy.imagecalc.spectralunmixing as su

            su.spec_unmix_fcls(
                "img.tif", "mask.tif", 1, "unmix_fcls.tif", "endmembers.csv"
            )
        """
        # pylint: disable=too-many-arguments,too-many-positional-arguments
        _run_windowed_unmixer(
            input_img=input_img,
            valid_msk_img=valid_msk_img,
            valid_msk_val=valid_msk_val,
            output_img=output_img,
            endmembers=endmembers,
            solver_fn=_scipy_fcls_solver,
            gdalformat=gdalformat,
            gain=gain,
            weight=None,
            file_format=file_format,
            calc_stats=calc_stats,
            use_pbar=use_pbar,
        )


if HAVE_PYSPTOOLS:

    def spec_unmix_spts_ucls(
        input_img: str,
        valid_msk_img: Optional[str],
        valid_msk_val: int,
        output_img: str,
        endmembers: EndmemberInput,
        gdalformat: str = "GTIFF",
        gain: float = 1,
        weight: Optional[float] = None,
        file_format: Optional[str] = None,
        calc_stats: bool = True,
        use_pbar: bool = True,
    ) -> None:
        """
        Perform unconstrained least-squares (UCLS) spectral unmixing using
        ``pysptools``.

        .. note::

           Prefer the recommended pure-NumPy :func:`spec_unmix_ucls` function
           for general workflows; this function is retained for ``pysptools``
           compatibility.

        :param input_img: Path to the input image file.
        :param valid_msk_img: Optional path to the valid pixel mask image file.
        :param valid_msk_val: Pixel value in ``valid_msk_img`` marking valid pixels.
        :param output_img: Path to the output unmixed abundance image file.
        :param endmembers: :class:`pandas.DataFrame` (columns = class names,
                           rows = bands) or file path readable by
                           :func:`read_endmembers_df`.
        :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
        :param gain: Gain divisor applied to image and endmember spectra when
                     greater than ``1`` (Default = ``1``).
        :param weight: Optional Scarth et al. (2010) sum-to-one weight.
        :param file_format: Optional format override when ``endmembers`` is a
                            file path (Default = ``None``).
        :param calc_stats: If ``True``, populate image statistics and pyramids
                           (Default = ``True``).
        :param use_pbar: If ``True``, display a tqdm progress bar during processing
                         (Default = ``True``).

        .. code:: python

            import rsgispy.imagecalc.spectralunmixing as su

            su.spec_unmix_spts_ucls(
                "img.tif", "mask.tif", 1, "unmix_ucls.tif", "endmembers.csv"
            )
        """
        # pylint: disable=too-many-arguments,too-many-positional-arguments
        _run_windowed_unmixer(
            input_img=input_img,
            valid_msk_img=valid_msk_img,
            valid_msk_val=valid_msk_val,
            output_img=output_img,
            endmembers=endmembers,
            solver_fn=_spts_amaps.UCLS,
            gdalformat=gdalformat,
            gain=gain,
            weight=weight,
            file_format=file_format,
            calc_stats=calc_stats,
            use_pbar=use_pbar,
        )

    def spec_unmix_spts_nnls(
        input_img: str,
        valid_msk_img: Optional[str],
        valid_msk_val: int,
        output_img: str,
        endmembers: EndmemberInput,
        gdalformat: str = "GTIFF",
        gain: float = 1,
        weight: Optional[float] = None,
        file_format: Optional[str] = None,
        calc_stats: bool = True,
        use_pbar: bool = True,
    ) -> None:
        """
        Perform non-negative least-squares (NNLS) spectral unmixing using
        ``pysptools``.

        .. note::

           Prefer the recommended SciPy :func:`spec_unmix_nnls` function for
           general workflows; ``spec_unmix_nnls`` solves the direct NNLS
           problem without squaring normal equations.

        :param input_img: Path to the input image file.
        :param valid_msk_img: Optional path to the valid pixel mask image file.
        :param valid_msk_val: Pixel value in ``valid_msk_img`` marking valid pixels.
        :param output_img: Path to the output unmixed abundance image file.
        :param endmembers: :class:`pandas.DataFrame` (columns = class names,
                           rows = bands) or file path readable by
                           :func:`read_endmembers_df`.
        :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
        :param gain: Gain divisor applied to image and endmember spectra when
                     greater than ``1`` (Default = ``1``).
        :param weight: Optional Scarth et al. (2010) sum-to-one weight.
        :param file_format: Optional format override when ``endmembers`` is a
                            file path (Default = ``None``).
        :param calc_stats: If ``True``, populate image statistics and pyramids
                           (Default = ``True``).
        :param use_pbar: If ``True``, display a tqdm progress bar during processing
                         (Default = ``True``).

        .. code:: python

            import rsgispy.imagecalc.spectralunmixing as su

            su.spec_unmix_spts_nnls(
                "img.tif", "mask.tif", 1, "unmix_nnls.tif", "endmembers.csv"
            )
        """
        # pylint: disable=too-many-arguments,too-many-positional-arguments
        _run_windowed_unmixer(
            input_img=input_img,
            valid_msk_img=valid_msk_img,
            valid_msk_val=valid_msk_val,
            output_img=output_img,
            endmembers=endmembers,
            solver_fn=_spts_amaps.NNLS,
            gdalformat=gdalformat,
            gain=gain,
            weight=weight,
            file_format=file_format,
            calc_stats=calc_stats,
            use_pbar=use_pbar,
        )

    if HAVE_CVXOPT:

        def spec_unmix_spts_fcls(
            input_img: str,
            valid_msk_img: Optional[str],
            valid_msk_val: int,
            output_img: str,
            endmembers: EndmemberInput,
            gdalformat: str = "GTIFF",
            gain: float = 1,
            file_format: Optional[str] = None,
            calc_stats: bool = True,
            use_pbar: bool = True,
        ) -> None:
            """
            Perform fully constrained least-squares (FCLS: non-negative and
            sum-to-one) spectral unmixing using ``pysptools`` and ``cvxopt``.

            .. note::

               Prefer the recommended SciPy :func:`spec_unmix_fcls` function
               for general workflows; it does not require ``cvxopt`` or
               ``pysptools`` and runs substantially faster.

            :param input_img: Path to the input image file.
            :param valid_msk_img: Optional path to the valid pixel mask image
                                  file.
            :param valid_msk_val: Pixel value in ``valid_msk_img`` marking valid
                                  pixels.
            :param output_img: Path to the output unmixed abundance image file.
            :param endmembers: :class:`pandas.DataFrame` (columns = class names,
                               rows = bands) or file path readable by
                               :func:`read_endmembers_df`.
            :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
            :param gain: Gain divisor applied to image and endmember spectra when
                         greater than ``1`` (Default = ``1``).
            :param file_format: Optional format override when ``endmembers`` is
                                a file path (Default = ``None``).
            :param calc_stats: If ``True``, populate image statistics and
                               pyramids (Default = ``True``).
            :param use_pbar: If ``True``, display a tqdm progress bar during
                             processing (Default = ``True``).

            .. code:: python

                import rsgispy.imagecalc.spectralunmixing as su

                su.spec_unmix_spts_fcls(
                    "img.tif", "mask.tif", 1, "unmix_fcls.tif", "endmembers.csv"
                )
            """
            # pylint: disable=too-many-arguments,too-many-positional-arguments
            _run_windowed_unmixer(
                input_img=input_img,
                valid_msk_img=valid_msk_img,
                valid_msk_val=valid_msk_val,
                output_img=output_img,
                endmembers=endmembers,
                solver_fn=_spts_amaps.FCLS,
                gdalformat=gdalformat,
                gain=gain,
                weight=None,
                file_format=file_format,
                calc_stats=calc_stats,
                use_pbar=use_pbar,
            )


if HAVE_PYMCR:

    def _pymcr_nnls_solver(vld_pxls: np.ndarray, endmembers_arr: np.ndarray) -> np.ndarray:
        """Fit PyMCR NNLS model and return abundance matrix."""
        mcrar = McrAR(
            max_iter=100,
            st_regr=_PyMcrNNLS(),
            c_regr=_PyMcrNNLS(),
            c_constraints=[ConstraintNonneg()],
        )
        mcrar.fit(vld_pxls, ST=endmembers_arr)
        return mcrar.C_opt_

    def _pymcr_fcls_solver(vld_pxls: np.ndarray, endmembers_arr: np.ndarray) -> np.ndarray:
        """Fit PyMCR FCLS model and return abundance matrix."""
        mcrar = McrAR(
            max_iter=200,
            st_regr=_PyMcrNNLS(),
            c_regr=_PyMcrNNLS(),
            c_constraints=[ConstraintNonneg(), ConstraintNorm()],
        )
        mcrar.fit(vld_pxls, ST=endmembers_arr)
        return mcrar.C_opt_

    def spec_unmix_pymcr_nnls(
        input_img: str,
        valid_msk_img: Optional[str],
        valid_msk_val: int,
        output_img: str,
        endmembers: EndmemberInput,
        gdalformat: str = "GTIFF",
        gain: float = 1,
        weight: Optional[float] = None,
        file_format: Optional[str] = None,
        calc_stats: bool = True,
        use_pbar: bool = True,
    ) -> None:
        """
        Perform non-negative least-squares (NNLS) spectral unmixing using
        ``pymcr``.

        .. note::

           Prefer the recommended SciPy :func:`spec_unmix_nnls` function for
           general workflows.

        :param input_img: Path to the input image file.
        :param valid_msk_img: Optional path to the valid pixel mask image file.
        :param valid_msk_val: Pixel value in ``valid_msk_img`` marking valid pixels.
        :param output_img: Path to the output unmixed abundance image file.
        :param endmembers: :class:`pandas.DataFrame` (columns = class names,
                           rows = bands) or file path readable by
                           :func:`read_endmembers_df`.
        :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
        :param gain: Gain divisor applied to image and endmember spectra when
                     greater than ``1`` (Default = ``1``).
        :param weight: Optional Scarth et al. (2010) sum-to-one weight.
        :param file_format: Optional format override when ``endmembers`` is a
                            file path (Default = ``None``).
        :param calc_stats: If ``True``, populate image statistics and pyramids
                           (Default = ``True``).
        :param use_pbar: If ``True``, display a tqdm progress bar during processing
                         (Default = ``True``).

        .. code:: python

            import rsgispy.imagecalc.spectralunmixing as su

            su.spec_unmix_pymcr_nnls(
                "img.tif", "mask.tif", 1, "unmix_pymcr_nnls.tif", "endmembers.csv"
            )
        """
        # pylint: disable=too-many-arguments,too-many-positional-arguments
        _run_windowed_unmixer(
            input_img=input_img,
            valid_msk_img=valid_msk_img,
            valid_msk_val=valid_msk_val,
            output_img=output_img,
            endmembers=endmembers,
            solver_fn=_pymcr_nnls_solver,
            gdalformat=gdalformat,
            gain=gain,
            weight=weight,
            file_format=file_format,
            calc_stats=calc_stats,
            use_pbar=use_pbar,
        )

    def spec_unmix_pymcr_fcls(
        input_img: str,
        valid_msk_img: Optional[str],
        valid_msk_val: int,
        output_img: str,
        endmembers: EndmemberInput,
        gdalformat: str = "GTIFF",
        gain: float = 1,
        file_format: Optional[str] = None,
        calc_stats: bool = True,
        use_pbar: bool = True,
    ) -> None:
        """
        Perform fully constrained least-squares (FCLS: non-negative and
        sum-to-one) spectral unmixing using ``pymcr``.

        .. note::

           Prefer the recommended SciPy :func:`spec_unmix_fcls` function for
           general workflows.

        :param input_img: Path to the input image file.
        :param valid_msk_img: Optional path to the valid pixel mask image file.
        :param valid_msk_val: Pixel value in ``valid_msk_img`` marking valid pixels.
        :param output_img: Path to the output unmixed abundance image file.
        :param endmembers: :class:`pandas.DataFrame` (columns = class names,
                           rows = bands) or file path readable by
                           :func:`read_endmembers_df`.
        :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
        :param gain: Gain divisor applied to image and endmember spectra when
                     greater than ``1`` (Default = ``1``).
        :param file_format: Optional format override when ``endmembers`` is a
                            file path (Default = ``None``).
        :param calc_stats: If ``True``, populate image statistics and pyramids
                           (Default = ``True``).
        :param use_pbar: If ``True``, display a tqdm progress bar during processing
                         (Default = ``True``).

        .. code:: python

            import rsgispy.imagecalc.spectralunmixing as su

            su.spec_unmix_pymcr_fcls(
                "img.tif", "mask.tif", 1, "unmix_pymcr_fcls.tif", "endmembers.csv"
            )
        """
        # pylint: disable=too-many-arguments,too-many-positional-arguments
        _run_windowed_unmixer(
            input_img=input_img,
            valid_msk_img=valid_msk_img,
            valid_msk_val=valid_msk_val,
            output_img=output_img,
            endmembers=endmembers,
            solver_fn=_pymcr_fcls_solver,
            gdalformat=gdalformat,
            gain=gain,
            weight=None,
            file_format=file_format,
            calc_stats=calc_stats,
            use_pbar=use_pbar,
        )


def rescale_unmixing_results(
    input_img: str,
    output_img: str,
    gdalformat: str = "GTIFF",
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Clamp negative abundances to zero and normalise each pixel's abundances
    so they sum to ``1.0`` across bands.

    :param input_img: Path to the input unmixed abundance image file.
    :param output_img: Path to the output rescaled abundance image file.
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param calc_stats: If ``True``, populate image statistics and pyramids
                       (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).

    .. code:: python

        import rsgispy.imagecalc.spectralunmixing as su

        su.rescale_unmixing_results("unmix_ucls.tif", "unmix_rescaled.tif")
    """
    ensure_file_exists(input_img, "Input image")

    with ExitStack() as stack:
        src = stack.enter_context(rasterio.open(input_img, "r"))
        dst = create_output_from_dataset(
            stack=stack,
            src=src,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype="float32",
            count=src.count,
        )

        for win in iter_dataset_windows(src.height, src.width, use_pbar=use_pbar):
            blk = src.read(window=win).astype(np.float32, copy=True)
            blk[blk < 0.0] = 0.0
            band_sum = np.sum(blk, axis=0, keepdims=True)
            out_blk = np.zeros_like(blk, dtype=np.float32)
            with np.errstate(divide="ignore", invalid="ignore"):
                np.divide(blk, band_sum, out=out_blk, where=band_sum > 0.0)
            out_blk = np.where(np.isfinite(out_blk), out_blk, 0.0).astype(
                np.float32, copy=False
            )
            dst.write(out_blk, window=win)

    if calc_stats:
        pop_img_stats(
            output_img,
            no_data_val=0,
            calc_pyramids=True,
            use_pbar=use_pbar,
        )


def predict_refl_linear_unmixing(
    in_unmix_coefs_img: str,
    endmembers: EndmemberInput,
    output_img: str,
    gdalformat: str = "GTIFF",
    file_format: Optional[str] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Predict per-band reflectance values from unmixed abundance coefficients and
    an endmembers :class:`pandas.DataFrame` (or file path).

    :param in_unmix_coefs_img: Path to the input abundance coefficients image.
    :param endmembers: :class:`pandas.DataFrame` (columns = class names,
                       rows = bands) or file path readable by
                       :func:`read_endmembers_df`.
    :param output_img: Path to the output predicted reflectance image.
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param file_format: Optional format override when ``endmembers`` is a file
                        path (Default = ``None``).
    :param calc_stats: If ``True``, populate image statistics and pyramids
                       (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).

    .. code:: python

        import rsgispy.imagecalc.spectralunmixing as su

        su.predict_refl_linear_unmixing(
            "unmix.tif", "endmembers.csv", "pred_refl.tif"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    ensure_file_exists(in_unmix_coefs_img, "Input unmixing coefficients image")
    em_df = _coerce_endmembers_df(endmembers, file_format=file_format)
    endmembers_arr = em_df.to_numpy(dtype=np.float64).T
    m, n_bands = endmembers_arr.shape
    band_names = [str(idx) for idx in em_df.index]

    with ExitStack() as stack:
        src = stack.enter_context(rasterio.open(in_unmix_coefs_img, "r"))
        if src.count < m:
            raise rsgispy.RSGISPyException(
                f"Unmixing image '{in_unmix_coefs_img}' has {src.count} bands, "
                f"but endmembers define {m}."
            )

        dst = create_output_from_dataset(
            stack=stack,
            src=src,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype="float32",
            count=n_bands,
        )
        for b_idx, b_name in enumerate(band_names, start=1):
            dst.set_band_description(b_idx, b_name)

        for win in iter_dataset_windows(src.height, src.width, use_pbar=use_pbar):
            win_h = int(win.height)
            win_w = int(win.width)
            blk = src.read(list(range(1, m + 1)), window=win).astype(
                np.float64, copy=False
            )
            coefs_flat = np.moveaxis(blk, 0, -1).reshape(-1, m)
            pred_flat = (coefs_flat @ endmembers_arr).astype(np.float32, copy=False)
            out_blk = np.moveaxis(
                pred_flat.reshape((win_h, win_w, n_bands)), -1, 0
            )
            dst.write(out_blk, window=win)

    if calc_stats:
        pop_img_stats(
            output_img,
            no_data_val=0,
            calc_pyramids=True,
            use_pbar=use_pbar,
        )


def calc_unmixing_rmse_residual_err(
    input_img: str,
    input_unmix_img: str,
    endmembers: EndmemberInput,
    output_img: str,
    gdalformat: str = "GTIFF",
    file_format: Optional[str] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Calculate linear unmixing prediction RMSE, RMSPE, and mean absolute residual
    error against the original reference image.

    Output bands are: Band 1 ``RMSE``, Band 2 ``RMSPE`` (percentage), and
    Band 3 ``Residual``.

    :param input_img: Path to the original reference reflectance image file.
    :param input_unmix_img: Path to the unmixed abundance coefficients image.
    :param endmembers: :class:`pandas.DataFrame` (columns = class names,
                       rows = bands) or file path readable by
                       :func:`read_endmembers_df`.
    :param output_img: Path to the 3-band output error image file.
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param file_format: Optional format override when ``endmembers`` is a file
                        path (Default = ``None``).
    :param calc_stats: If ``True``, assign band names and calculate image
                       statistics and pyramids (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).

    .. code:: python

        import rsgispy.imagecalc.spectralunmixing as su

        su.calc_unmixing_rmse_residual_err(
            "s2_refl.tif", "unmix.tif", "endmembers.csv", "unmix_err.tif"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    _, m, n_bands, endmembers_arr = _prepare_endmembers_array(
        endmembers, file_format=file_format
    )

    with ExitStack() as stack:
        datasets, paths = open_multi_image_datasets(stack, [input_img, input_unmix_img])
        src_orig = datasets[paths[0]]
        src_unmix = datasets[paths[1]]
        grid = compute_spatial_intersection(datasets)

        if src_orig.count != n_bands:
            raise rsgispy.RSGISPyException(
                f"Input image '{input_img}' has {src_orig.count} bands, "
                f"expected {n_bands}."
            )
        if src_unmix.count < m:
            raise rsgispy.RSGISPyException(
                f"Unmixed image '{input_unmix_img}' has {src_unmix.count} bands, "
                f"expected at least {m}."
            )

        dst = open_output_dataset(
            stack=stack,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype="float32",
            grid=grid,
            count=3,
        )

        orig_xoff, orig_yoff = grid.offsets[paths[0]]
        unmix_xoff, unmix_yoff = grid.offsets[paths[1]]

        for win in iter_dataset_windows(grid.height, grid.width, use_pbar=use_pbar):
            win_h = int(win.height)
            win_w = int(win.width)
            orig_win = Window(
                int(win.col_off) + orig_xoff,
                int(win.row_off) + orig_yoff,
                win_w,
                win_h,
            )
            unmix_win = Window(
                int(win.col_off) + unmix_xoff,
                int(win.row_off) + unmix_yoff,
                win_w,
                win_h,
            )

            orig_blk = src_orig.read(window=orig_win).astype(np.float64, copy=False)
            unmix_blk = src_unmix.read(
                list(range(1, m + 1)), window=unmix_win
            ).astype(np.float64, copy=False)

            orig_flat = np.moveaxis(orig_blk, 0, -1).reshape(-1, n_bands)
            unmix_flat = np.moveaxis(unmix_blk, 0, -1).reshape(-1, m)

            vld = np.logical_not(np.all(orig_flat == 0.0, axis=1)) & np.all(
                np.isfinite(orig_flat), axis=1
            )
            out_flat = np.zeros((orig_flat.shape[0], 3), dtype=np.float32)

            if np.any(vld):
                v_orig = orig_flat[vld]
                v_coef = unmix_flat[vld]
                pred_refl = v_coef @ endmembers_arr
                diff = v_orig - pred_refl

                with np.errstate(divide="ignore", invalid="ignore"):
                    norm_diff = np.where(v_orig != 0.0, diff / v_orig, 0.0)

                rmse = np.sqrt(np.sum(diff * diff, axis=1) / float(n_bands))
                rmspe = np.sqrt(np.sum(norm_diff * norm_diff, axis=1) / float(n_bands)) * 100.0
                residual = np.sum(np.abs(diff), axis=1) / float(n_bands)

                err_stack = np.stack([rmse, rmspe, residual], axis=-1)
                err_stack = np.nan_to_num(err_stack, nan=0.0, posinf=0.0, neginf=0.0)
                out_flat[vld] = err_stack.astype(np.float32, copy=False)

            out_blk = np.moveaxis(out_flat.reshape((win_h, win_w, 3)), -1, 0)
            dst.write(out_blk, window=win)

    if calc_stats:
        set_band_names(output_img, ["RMSE", "RMSPE", "Residual"], feedback=False)
        pop_img_stats(
            output_img,
            no_data_val=None,
            calc_pyramids=True,
            use_pbar=use_pbar,
        )


def summarise_multi_endmember_linear_unmixing(
    input_img: str,
    in_unmixed_datasets: Sequence[ImageEndmemberInfo],
    out_unmix_img: str,
    out_ref_img: str,
    tmp_dir: str = "tmp",
    gdalformat: str = "GTIFF",
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Merge multiple linear spectral unmixing results by selecting the model with
    the lowest Root Mean Square Percentage Error (RMSPE) at each pixel.

    :param input_img: Path to the original input reflectance image file.
    :param in_unmixed_datasets: Sequence of :class:`ImageEndmemberInfo`
                                objects describing each candidate unmixing run.
    :param out_unmix_img: Path to the combined output abundance image file.
    :param out_ref_img: Path to the 1-based reference index image indicating
                        which candidate dataset achieved the lowest RMSPE per
                        pixel (``0`` for invalid pixels).
    :param tmp_dir: Retained for API compatibility (Default = ``'tmp'``).
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param calc_stats: If ``True``, set band names and compute statistics and
                       pyramids (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).

    .. code:: python

        import rsgispy.imagecalc.spectralunmixing as su

        info_a = su.ImageEndmemberInfo("unmix_a.tif", "em_a.csv")
        info_b = su.ImageEndmemberInfo("unmix_b.tif", "em_b.csv")
        su.summarise_multi_endmember_linear_unmixing(
            "s2_refl.tif",
            [info_a, info_b],
            "unmix_best.tif",
            "unmix_ref.tif",
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    _ = tmp_dir
    if not isinstance(in_unmixed_datasets, (list, tuple)) or not in_unmixed_datasets:
        raise rsgispy.RSGISPyException(
            "in_unmixed_datasets must be a non-empty list of ImageEndmemberInfo."
        )

    endmember_names: List[str] = []
    em_infos: List[Tuple[int, int, np.ndarray]] = []
    resolved_ds_names: List[List[str]] = []
    img_paths = [input_img]

    for ds_info in in_unmixed_datasets:
        if not isinstance(ds_info, ImageEndmemberInfo):
            raise rsgispy.RSGISPyException(
                "in_unmixed_datasets must contain ImageEndmemberInfo instances."
            )
        em_names, em_m, em_n, em_arr = _prepare_endmembers_array(
            ds_info.endmembers, file_format=ds_info.file_format
        )
        if ds_info.endmember_names is not None:
            if not isinstance(ds_info.endmember_names, (list, tuple)):
                raise rsgispy.RSGISPyException(
                    "unmixed_dataset.endmember_names must be a list or tuple."
                )
            if em_m != len(ds_info.endmember_names):
                raise rsgispy.RSGISPyException(
                    f"Endmember count ({em_m}) does not match "
                    f"endmember_names length ({len(ds_info.endmember_names)})."
                )
            ds_names = [str(name) for name in ds_info.endmember_names]
        else:
            ds_names = em_names

        resolved_ds_names.append(ds_names)
        em_infos.append((em_m, em_n, em_arr))
        img_paths.append(ds_info.in_unmix_img)
        for name in ds_names:
            if name not in endmember_names:
                endmember_names.append(name)

    name_to_band: Dict[str, int] = {
        name: idx for idx, name in enumerate(endmember_names)
    }
    n_out_bands = len(endmember_names)
    n_models = len(in_unmixed_datasets)
    ref_dtype = "uint8" if n_models <= 254 else "uint16"

    with ExitStack() as stack:
        datasets, resolved_paths = open_multi_image_datasets(stack, img_paths)
        src_orig = datasets[resolved_paths[0]]
        for ds_idx, ds_info in enumerate(in_unmixed_datasets):
            u_ds = datasets[resolved_paths[ds_idx + 1]]
            em_m, em_n, _ = em_infos[ds_idx]
            ds_names = resolved_ds_names[ds_idx]
            if u_ds.count != len(ds_names):
                raise rsgispy.RSGISPyException(
                    f"Band count ({u_ds.count}) of '{ds_info.in_unmix_img}' "
                    f"does not match endmember names ({len(ds_names)})."
                )
            if em_m != u_ds.count:
                raise rsgispy.RSGISPyException(
                    f"Band count ({u_ds.count}) of '{ds_info.in_unmix_img}' "
                    f"does not match endmember count ({em_m})."
                )
            if em_n != src_orig.count:
                raise rsgispy.RSGISPyException(
                    f"Input image '{input_img}' band count ({src_orig.count}) "
                    f"does not match endmember band count ({em_n})."
                )

        grid = compute_spatial_intersection(datasets)
        dst_unmix = open_output_dataset(
            stack=stack,
            output_img=out_unmix_img,
            gdalformat=gdalformat,
            out_rio_dtype="float32",
            grid=grid,
            count=n_out_bands,
        )
        for b_idx, b_name in enumerate(endmember_names, start=1):
            dst_unmix.set_band_description(b_idx, b_name)

        dst_ref = open_output_dataset(
            stack=stack,
            output_img=out_ref_img,
            gdalformat=gdalformat,
            out_rio_dtype=ref_dtype,
            grid=grid,
            count=1,
        )

        orig_xoff, orig_yoff = grid.offsets[resolved_paths[0]]
        model_band_maps = [
            [name_to_band[name] for name in ds_names]
            for ds_names in resolved_ds_names
        ]

        for win in iter_dataset_windows(grid.height, grid.width, use_pbar=use_pbar):
            win_h = int(win.height)
            win_w = int(win.width)
            n_pxls = win_h * win_w

            orig_win = Window(
                int(win.col_off) + orig_xoff,
                int(win.row_off) + orig_yoff,
                win_w,
                win_h,
            )
            orig_blk = src_orig.read(window=orig_win).astype(np.float64, copy=False)
            orig_flat = np.moveaxis(orig_blk, 0, -1).reshape(-1, src_orig.count)
            vld = np.logical_not(np.all(orig_flat == 0.0, axis=1)) & np.all(
                np.isfinite(orig_flat), axis=1
            )

            best_rmspe = np.full(n_pxls, np.inf, dtype=np.float64)
            best_ref = np.zeros(n_pxls, dtype=np.int32)
            best_abund = np.zeros((n_pxls, n_out_bands), dtype=np.float32)

            if np.any(vld):
                v_orig = orig_flat[vld]
                for ds_idx in range(n_models):
                    u_path = resolved_paths[ds_idx + 1]
                    u_ds = datasets[u_path]
                    u_xoff, u_yoff = grid.offsets[u_path]
                    u_win = Window(
                        int(win.col_off) + u_xoff,
                        int(win.row_off) + u_yoff,
                        win_w,
                        win_h,
                    )
                    u_blk = u_ds.read(window=u_win).astype(np.float64, copy=False)
                    u_flat = np.moveaxis(u_blk, 0, -1).reshape(-1, u_ds.count)
                    v_coef = u_flat[vld]

                    _, n_bands_m, em_arr = em_infos[ds_idx]
                    pred_refl = v_coef @ em_arr
                    diff = v_orig - pred_refl
                    with np.errstate(divide="ignore", invalid="ignore"):
                        norm_diff = np.where(v_orig != 0.0, diff / v_orig, 0.0)
                    rmspe = (
                        np.sqrt(np.sum(norm_diff * norm_diff, axis=1) / float(n_bands_m))
                        * 100.0
                    )
                    rmspe = np.where(np.isfinite(rmspe), rmspe, np.inf)

                    v_idx = np.flatnonzero(vld)
                    better = rmspe < best_rmspe[v_idx]
                    if np.any(better):
                        win_pxls = v_idx[better]
                        best_rmspe[win_pxls] = rmspe[better]
                        best_ref[win_pxls] = ds_idx + 1
                        best_abund[win_pxls, :] = 0.0
                        for local_b, global_b in enumerate(model_band_maps[ds_idx]):
                            best_abund[win_pxls, global_b] = v_coef[better, local_b]

            out_unmix_blk = np.moveaxis(
                best_abund.reshape((win_h, win_w, n_out_bands)), -1, 0
            )
            dst_unmix.write(out_unmix_blk, window=win)
            dst_ref.write(
                best_ref.astype(ref_dtype).reshape((1, win_h, win_w)), window=win
            )

    if calc_stats:
        set_band_names(out_unmix_img, endmember_names, feedback=False)
        pop_img_stats(
            out_unmix_img,
            no_data_val=999,
            calc_pyramids=True,
            use_pbar=use_pbar,
        )
        pop_thmt_img_stats(
            out_ref_img,
            add_clr_tab=True,
            calc_pyramids=True,
            ignore_zero=True,
            use_pbar=use_pbar,
        )


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
    :param img_gain: Gain divisor applied to input reflectance values when
                     greater than ``1`` (Default = ``1``).
    :param seed: Optional random seed for reproducible projections.
    :param calc_stats: If ``True``, compute statistics and pyramids on
                       ``output_img`` (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar during iterations
                     (Default = ``True``).

    .. code:: python

        import rsgispy.imagecalc.spectralunmixing as su

        su.calc_ppi(
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
            if nd is not None and not (isinstance(nd, float) and math.isnan(nd)):
                b_vld &= band_arr != nd
            img_msk &= b_vld

            if img_gain > 1:
                band_arr = band_arr / float(img_gain)
            band_tmp = np.where(b_vld, band_arr, np.nan)
            band_mean = float(np.nanmean(band_tmp)) if np.any(b_vld) else 0.0
            img_data[b_idx] = band_arr - band_mean

        vld_data = img_data[:, img_msk].T  # (n_valid, n_bands)
        counts = np.zeros(vld_data.shape[0], dtype=np.uint16)

        if vld_data.shape[0] > 0:
            rng = np.random.default_rng(seed)
            for _ in pbar_cls(
                range(niters), desc="Calculating PPI", disable=not use_pbar
            ):
                skewer = rng.random(n_bands) - 0.5
                proj = vld_data @ skewer
                s_min = float(np.min(proj))
                s_max = float(np.max(proj))
                if lthres == 0:
                    counts[int(np.argmin(proj))] += 1
                else:
                    counts[proj <= (s_min + lthres)] += 1

                if uthres == 0:
                    counts[int(np.argmax(proj))] += 1
                else:
                    counts[proj >= (s_max - uthres)] += 1

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
