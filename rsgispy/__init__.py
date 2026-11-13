"""
rsgispy: Pure Python replacement for the RSGISLib library.
Provides remote sensing and GIS software tools and utilities.

Data Types for images:

    * TYPE_UNDEFINED = 0
    * TYPE_8INT = 1
    * TYPE_16INT = 2
    * TYPE_32INT = 3
    * TYPE_64INT = 4
    * TYPE_8UINT = 5
    * TYPE_16UINT = 6
    * TYPE_32UINT = 7
    * TYPE_64UINT = 8
    * TYPE_32FLOAT = 9
    * TYPE_64FLOAT = 10
"""

from typing import Union
import numpy as np
from rasterio.enums import Resampling

__version__ = "0.0.1"

TYPE_UNDEFINED = 0
TYPE_8INT = 1
TYPE_16INT = 2
TYPE_32INT = 3
TYPE_64INT = 4
TYPE_8UINT = 5
TYPE_16UINT = 6
TYPE_32UINT = 7
TYPE_64UINT = 8
TYPE_32FLOAT = 9
TYPE_64FLOAT = 10

INTERP_NEAREST_NEIGHBOUR = 0
INTERP_BILINEAR = 1
INTERP_CUBIC = 2
INTERP_CUBICSPLINE = 3
INTERP_LANCZOS = 4
INTERP_AVERAGE = 5
INTERP_MODE = 6
INTERP_MIN = 7
INTERP_MAX = 8
INTERP_MEDIAN = 9
INTERP_Q1 = 10
INTERP_Q3 = 11
INTERP_SUM = 12

IMG_STRETCH_NONE = 0
IMG_STRETCH_USER = 1
IMG_STRETCH_LINEAR = 2
IMG_STRETCH_STDEV = 3
IMG_STRETCH_CUMULATIVE = 4

SHARP_RES_IGNORE = 0
SHARP_RES_LOW = 1
SHARP_RES_HIGH = 2

GEOM_PT = 1
GEOM_LINE = 2
GEOM_POLY = 3
GEOM_MPT = 4
GEOM_MLINE = 5
GEOM_MPOLY = 6

ROUND_NEAREST = 1
ROUND_UP = 2
ROUND_DOWN = 3

SUMTYPE_UNDEFINED = 0
SUMTYPE_MODE = 1
SUMTYPE_MEAN = 2
SUMTYPE_MEDIAN = 3
SUMTYPE_MIN = 4
SUMTYPE_MAX = 5
SUMTYPE_STDDEV = 6
SUMTYPE_COUNT = 7
SUMTYPE_RANGE = 8
SUMTYPE_SUM = 9

STATS_CORR_UNDEFINED = 0
STATS_CORR_PEARSONS = 1
STATS_CORR_SPEARMAN = 2
STATS_CORR_KENDALL_TAU = 3
STATS_CORR_POINT_BISERIAL = 4

_RSGIS_TO_RASTERIO_INTERP = {
    INTERP_NEAREST_NEIGHBOUR: Resampling.nearest,
    INTERP_BILINEAR: Resampling.bilinear,
    INTERP_CUBIC: Resampling.cubic,
    INTERP_CUBICSPLINE: Resampling.cubic_spline,
    INTERP_LANCZOS: Resampling.lanczos,
    INTERP_AVERAGE: Resampling.average,
    INTERP_MODE: Resampling.mode,
    INTERP_MIN: Resampling.min,
    INTERP_MAX: Resampling.max,
    INTERP_MEDIAN: Resampling.med,
    INTERP_Q1: Resampling.q1,
    INTERP_Q3: Resampling.q3,
    INTERP_SUM: Resampling.sum,
}

_RSGIS_TO_RASTERIO_DTYPE = {
    TYPE_8INT: "int8",
    TYPE_16INT: "int16",
    TYPE_32INT: "int32",
    TYPE_64INT: "int64",
    TYPE_8UINT: "uint8",
    TYPE_16UINT: "uint16",
    TYPE_32UINT: "uint32",
    TYPE_64UINT: "uint64",
    TYPE_32FLOAT: "float32",
    TYPE_64FLOAT: "float64",
}

_RSGIS_TO_NUMPY_DTYPE = {
    TYPE_8INT: np.int8,
    TYPE_16INT: np.int16,
    TYPE_32INT: np.int32,
    TYPE_64INT: np.int64,
    TYPE_8UINT: np.uint8,
    TYPE_16UINT: np.uint16,
    TYPE_32UINT: np.uint32,
    TYPE_64UINT: np.uint64,
    TYPE_32FLOAT: np.float32,
    TYPE_64FLOAT: np.float64,
}

_RSGIS_TO_STR_DTYPE = {
    TYPE_8INT: "8 bit INTEGER",
    TYPE_16INT: "16 bit INTEGER",
    TYPE_32INT: "32 bit INTEGER",
    TYPE_64INT: "64 bit INTEGER",
    TYPE_8UINT: "8 bit UNSIGNED INTEGER (BYTE)",
    TYPE_16UINT: "16 bit UNSIGNED INTEGER",
    TYPE_32UINT: "32 bit UNSIGNED INTEGER",
    TYPE_64UINT: "64 bit UNSIGNED INTEGER",
    TYPE_32FLOAT: "32 bit FLOAT",
    TYPE_64FLOAT: "64 bit FLOAT",
}

_STR_TO_RSGIS_DTYPE = {
    "int8": TYPE_8INT,
    "int16": TYPE_16INT,
    "int32": TYPE_32INT,
    "int64": TYPE_64INT,
    "byte": TYPE_8UINT,
    "uint8": TYPE_8UINT,
    "uint16": TYPE_16UINT,
    "uint32": TYPE_32UINT,
    "uint64": TYPE_64UINT,
    "float32": TYPE_32FLOAT,
    "float64": TYPE_64FLOAT,
}


class RSGISPyException(Exception):
    """
    A class representing an RSGISPy exception.
    """

    def __init__(self, value: str):
        """
        Init for the RSGISPyException class.
        """
        super().__init__(value)
        self.value = value

    def __str__(self) -> str:
        """
        Return a string representation of the exception.
        """
        return repr(self.value)


def get_rsgis_datatype(gdal_type: str) -> int:
    """
    Convert from a GDAL/rasterio data type string to an RSGIS data type int.

    :param gdal_type: Data type name string (e.g., 'float32', 'uint8', 'byte').
    :return: RSGIS data type constant (e.g., rsgispy.TYPE_32FLOAT).
    """
    key = gdal_type.lower().strip()
    if key not in _STR_TO_RSGIS_DTYPE:
        raise RSGISPyException(
            f"The data type '{gdal_type}' is unknown / not supported."
        )
    return _STR_TO_RSGIS_DTYPE[key]


def get_rasterio_datatype(rsgislib_datatype: int) -> str:
    """
    Convert from an RSGIS data type int to a rasterio/GDAL dtype string.

    :param rsgislib_datatype: RSGIS data type constant (e.g., rsgispy.TYPE_32FLOAT).
    :return: Rasterio dtype string (e.g., 'float32').
    """
    if rsgislib_datatype not in _RSGIS_TO_RASTERIO_DTYPE:
        raise RSGISPyException(
            f"The data type '{rsgislib_datatype}' is unknown / not supported."
        )
    return _RSGIS_TO_RASTERIO_DTYPE[rsgislib_datatype]


def get_numpy_datatype(rsgislib_datatype: int):
    """
    Convert from an RSGIS data type int to a NumPy dtype.

    :param rsgislib_datatype: RSGIS data type constant (e.g., rsgispy.TYPE_32FLOAT).
    :return: NumPy dtype class (e.g., numpy.float32).
    """
    if rsgislib_datatype not in _RSGIS_TO_NUMPY_DTYPE:
        raise RSGISPyException("Data type was not recognised.")
    return _RSGIS_TO_NUMPY_DTYPE[rsgislib_datatype]


def get_rsgis_datatype_str(rsgislib_datatype: int) -> str:
    """
    Get a human-readable string representation of the RSGIS data type.

    :param rsgislib_datatype: RSGIS data type constant.
    :return: Human-readable string description.
    """
    if rsgislib_datatype not in _RSGIS_TO_STR_DTYPE:
        raise RSGISPyException(
            f"The data type '{rsgislib_datatype}' is unknown / not supported."
        )
    return _RSGIS_TO_STR_DTYPE[rsgislib_datatype]


def get_rasterio_interp_type(
    rsgis_interp_method: Union[int, Resampling],
) -> Resampling:
    """
    Convert an ``rsgispy.INTERP_*`` (or ``rsgislib.INTERP_*``) constant to a
    ``rasterio.enums.Resampling`` enum member.

    :param rsgis_interp_method: Interpolation constant or ``Resampling`` enum.
    :return: Corresponding ``rasterio.enums.Resampling`` value.
    """
    if isinstance(rsgis_interp_method, Resampling):
        return rsgis_interp_method
    if rsgis_interp_method not in _RSGIS_TO_RASTERIO_INTERP:
        raise RSGISPyException(
            "Interpolation method was not recognised or known."
        )
    return _RSGIS_TO_RASTERIO_INTERP[rsgis_interp_method]


def is_notebook() -> bool:
    """
    Determine whether code is executing inside a Jupyter notebook environment.

    :return: True if running within a Jupyter notebook or qtconsole, False otherwise.
    """
    try:
        shell = get_ipython().__class__.__name__  # type: ignore[name-defined]
        if shell == "ZMQInteractiveShell":
            return True
        return False
    except NameError:
        return False


def get_tqdm():
    """
    Return the appropriate tqdm progress bar callable for the current environment
    (Jupyter notebook widget when inside a notebook and ipywidgets is available,
    or standard terminal tqdm otherwise).

    :return: tqdm class/callable.
    """
    if is_notebook():
        try:
            import ipywidgets  # noqa: F401  # pylint: disable=unused-import,import-outside-toplevel
            import tqdm.notebook  # pylint: disable=import-outside-toplevel

            return tqdm.notebook.tqdm
        except ImportError:
            pass
    import tqdm  # pylint: disable=import-outside-toplevel

    return tqdm.tqdm


def get_rsgispy_version() -> str:
    """
    Return the installed ``rsgispy`` version string.

    :return: Version string (e.g., ``'0.0.1'``).
    """
    return __version__
