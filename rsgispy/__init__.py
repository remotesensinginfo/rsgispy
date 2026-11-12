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

import numpy as np

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
