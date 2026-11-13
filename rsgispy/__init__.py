"""
rsgispy: Pure Python replacement for the RSGISLib library.
Provides remote sensing and GIS software tools and utilities.

Data Types for images:

    * ``TYPE_UNDEFINED = 0``
    * ``TYPE_8INT = 1``
    * ``TYPE_16INT = 2``
    * ``TYPE_32INT = 3``
    * ``TYPE_64INT = 4``
    * ``TYPE_8UINT = 5``
    * ``TYPE_16UINT = 6``
    * ``TYPE_32UINT = 7``
    * ``TYPE_64UINT = 8``
    * ``TYPE_32FLOAT = 9``
    * ``TYPE_64FLOAT = 10``

Interpolation methods:

    * ``INTERP_NEAREST_NEIGHBOUR = 0``
    * ``INTERP_BILINEAR = 1``
    * ``INTERP_CUBIC = 2``
    * ``INTERP_CUBICSPLINE = 3``
    * ``INTERP_LANCZOS = 4``
    * ``INTERP_AVERAGE = 5``
    * ``INTERP_MODE = 6``
    * ``INTERP_MIN = 7``
    * ``INTERP_MAX = 8``
    * ``INTERP_MEDIAN = 9``
    * ``INTERP_Q1 = 10``
    * ``INTERP_Q3 = 11``
    * ``INTERP_SUM = 12``

Image stretch methods:

    * ``IMG_STRETCH_NONE = 0``
    * ``IMG_STRETCH_USER = 1``
    * ``IMG_STRETCH_LINEAR = 2``
    * ``IMG_STRETCH_STDEV = 3``
    * ``IMG_STRETCH_CUMULATIVE = 4``

Pan-sharpening resolution options:

    * ``SHARP_RES_IGNORE = 0``
    * ``SHARP_RES_LOW = 1``
    * ``SHARP_RES_HIGH = 2``

Vector geometry types:

    * ``GEOM_PT = 1``
    * ``GEOM_LINE = 2``
    * ``GEOM_POLY = 3``
    * ``GEOM_MPT = 4``
    * ``GEOM_MLINE = 5``
    * ``GEOM_MPOLY = 6``

Rounding modes:

    * ``ROUND_NEAREST = 1``
    * ``ROUND_UP = 2``
    * ``ROUND_DOWN = 3``

Summary statistic types:

    * ``SUMTYPE_UNDEFINED = 0``
    * ``SUMTYPE_MODE = 1``
    * ``SUMTYPE_MEAN = 2``
    * ``SUMTYPE_MEDIAN = 3``
    * ``SUMTYPE_MIN = 4``
    * ``SUMTYPE_MAX = 5``
    * ``SUMTYPE_STDDEV = 6``
    * ``SUMTYPE_COUNT = 7``
    * ``SUMTYPE_RANGE = 8``
    * ``SUMTYPE_SUM = 9``

Correlation statistics methods:

    * ``STATS_CORR_UNDEFINED = 0``
    * ``STATS_CORR_PEARSONS = 1``
    * ``STATS_CORR_SPEARMAN = 2``
    * ``STATS_CORR_KENDALL_TAU = 3``
    * ``STATS_CORR_POINT_BISERIAL = 4``
"""

from typing import Union
import numpy as np
from rasterio.enums import Resampling

#: Package version string for ``rsgispy``.
__version__ = "0.0.2"

#: Undefined raster pixel data type (``0``).
TYPE_UNDEFINED = 0
#: Signed 8-bit integer raster data type (``1``).
TYPE_8INT = 1
#: Signed 16-bit integer raster data type (``2``).
TYPE_16INT = 2
#: Signed 32-bit integer raster data type (``3``).
TYPE_32INT = 3
#: Signed 64-bit integer raster data type (``4``).
TYPE_64INT = 4
#: Unsigned 8-bit integer (byte) raster data type (``5``).
TYPE_8UINT = 5
#: Unsigned 16-bit integer raster data type (``6``).
TYPE_16UINT = 6
#: Unsigned 32-bit integer raster data type (``7``).
TYPE_32UINT = 7
#: Unsigned 64-bit integer raster data type (``8``).
TYPE_64UINT = 8
#: 32-bit floating-point raster data type (``9``).
TYPE_32FLOAT = 9
#: 64-bit floating-point raster data type (``10``).
TYPE_64FLOAT = 10

#: Nearest-neighbour resampling interpolation (``0``).
INTERP_NEAREST_NEIGHBOUR = 0
#: Bilinear resampling interpolation (``1``).
INTERP_BILINEAR = 1
#: Cubic convolution resampling interpolation (``2``).
INTERP_CUBIC = 2
#: Cubic B-spline resampling interpolation (``3``).
INTERP_CUBICSPLINE = 3
#: Lanczos windowed sinc resampling interpolation (``4``).
INTERP_LANCZOS = 4
#: Average of contributing pixels resampling (``5``).
INTERP_AVERAGE = 5
#: Mode (most frequent value) resampling (``6``).
INTERP_MODE = 6
#: Minimum value resampling (``7``).
INTERP_MIN = 7
#: Maximum value resampling (``8``).
INTERP_MAX = 8
#: Median value resampling (``9``).
INTERP_MEDIAN = 9
#: First quartile (25th percentile) resampling (``10``).
INTERP_Q1 = 10
#: Third quartile (75th percentile) resampling (``11``).
INTERP_Q3 = 11
#: Sum of contributing pixels resampling (``12``).
INTERP_SUM = 12

#: No image display stretch applied (``0``).
IMG_STRETCH_NONE = 0
#: User-defined min/max image display stretch (``1``).
IMG_STRETCH_USER = 1
#: Linear min-to-max image display stretch (``2``).
IMG_STRETCH_LINEAR = 2
#: Standard-deviation image display stretch (``3``).
IMG_STRETCH_STDEV = 3
#: Cumulative percentile (2%–98%) image display stretch (``4``).
IMG_STRETCH_CUMULATIVE = 4

#: Ignore resolution setting when pan-sharpening (``0``).
SHARP_RES_IGNORE = 0
#: Output pan-sharpened image at low-resolution band scale (``1``).
SHARP_RES_LOW = 1
#: Output pan-sharpened image at high-resolution band scale (``2``).
SHARP_RES_HIGH = 2

#: Point vector geometry type constant (``1``).
GEOM_PT = 1
#: LineString vector geometry type constant (``2``).
GEOM_LINE = 2
#: Polygon vector geometry type constant (``3``).
GEOM_POLY = 3
#: MultiPoint vector geometry type constant (``4``).
GEOM_MPT = 4
#: MultiLineString vector geometry type constant (``5``).
GEOM_MLINE = 5
#: MultiPolygon vector geometry type constant (``6``).
GEOM_MPOLY = 6

#: Round floating-point values to nearest integer (``1``).
ROUND_NEAREST = 1
#: Round floating-point values up (ceiling) (``2``).
ROUND_UP = 2
#: Round floating-point values down (floor) (``3``).
ROUND_DOWN = 3

#: Undefined summary statistic type (``0``).
SUMTYPE_UNDEFINED = 0
#: Mode summary statistic type (``1``).
SUMTYPE_MODE = 1
#: Mean summary statistic type (``2``).
SUMTYPE_MEAN = 2
#: Median summary statistic type (``3``).
SUMTYPE_MEDIAN = 3
#: Minimum summary statistic type (``4``).
SUMTYPE_MIN = 4
#: Maximum summary statistic type (``5``).
SUMTYPE_MAX = 5
#: Standard deviation summary statistic type (``6``).
SUMTYPE_STDDEV = 6
#: Pixel count summary statistic type (``7``).
SUMTYPE_COUNT = 7
#: Value range (max - min) summary statistic type (``8``).
SUMTYPE_RANGE = 8
#: Sum summary statistic type (``9``).
SUMTYPE_SUM = 9

#: Undefined correlation method (``0``).
STATS_CORR_UNDEFINED = 0
#: Pearson linear correlation coefficient (``1``).
STATS_CORR_PEARSONS = 1
#: Spearman rank-order correlation coefficient (``2``).
STATS_CORR_SPEARMAN = 2
#: Kendall's tau rank correlation coefficient (``3``).
STATS_CORR_KENDALL_TAU = 3
#: Point-biserial correlation coefficient (``4``).
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

    .. code:: python

        import rsgispy
        dtype = rsgispy.get_rsgis_datatype("float32")
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

    .. code:: python

        import rsgispy
        rio_dtype = rsgispy.get_rasterio_datatype(rsgispy.TYPE_32FLOAT)
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

    .. code:: python

        import rsgispy
        np_dtype = rsgispy.get_numpy_datatype(rsgispy.TYPE_32FLOAT)
    """
    if rsgislib_datatype not in _RSGIS_TO_NUMPY_DTYPE:
        raise RSGISPyException("Data type was not recognised.")
    return _RSGIS_TO_NUMPY_DTYPE[rsgislib_datatype]


def get_rsgis_datatype_str(rsgislib_datatype: int) -> str:
    """
    Get a human-readable string representation of the RSGIS data type.

    :param rsgislib_datatype: RSGIS data type constant.
    :return: Human-readable string description.

    .. code:: python

        import rsgispy
        dtype_str = rsgispy.get_rsgis_datatype_str(rsgispy.TYPE_32FLOAT)
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

    .. code:: python

        import rsgispy
        interp = rsgispy.get_rasterio_interp_type(rsgispy.INTERP_BILINEAR)
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

    .. code:: python

        import rsgispy
        in_nb = rsgispy.is_notebook()
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

    .. code:: python

        import rsgispy
        pbar = rsgispy.get_tqdm()
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

    .. code:: python

        import rsgispy
        version = rsgispy.get_rsgispy_version()
    """
    return __version__
