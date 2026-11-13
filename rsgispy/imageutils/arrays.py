"""
The imageutils.arrays module provides NumPy array <-> raster dataset creation
and pixel/window extraction utilities using ``rasterio``.
"""

from contextlib import ExitStack
import math
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np
import rasterio
import rasterio.crs
from rasterio.transform import Affine
from rasterio.windows import Window

import rsgispy
from rsgispy.imageprocess import (
    IntersectionGrid,
    ensure_file_exists,
    open_output_dataset,
    validate_band_index,
)
from rsgispy.imageutils._helpers import normalise_creation_options
from rsgispy.imageutils.metadata import (
    get_img_bbox,
    get_img_res,
    get_img_size,
    get_wkt_proj_from_img,
)
from rsgispy.tools.geometry import does_bbox_contain

__all__ = [
    "create_img_from_array",
    "create_img_from_array_ref_img",
    "get_img_coords_pxl_values",
    "get_img_pxl_values",
    "set_img_pxl_values",
    "get_img_pxl_column",
    "get_img_column",
    "get_img_band_pxl_data",
    "get_img_data_as_arr",
]


def create_img_from_array(
    data_arr: np.ndarray,
    output_img: str,
    tl_x: float,
    tl_y: float,
    out_img_res_x: float,
    out_img_res_y: float,
    wkt_string: Optional[str],
    gdalformat: str,
    datatype: int,
    options: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
    no_data_val: Optional[float] = None,
) -> None:
    """
    Create a raster image from a 2D ``(height, width)`` or 3D
    ``(bands, height, width)`` NumPy array with the given spatial header.

    :param data_arr: Input NumPy array of shape ``(bands, height, width)`` or
                     ``(height, width)``.
    :param output_img: Path to the output image file.
    :param tl_x: Top-left corner X coordinate.
    :param tl_y: Top-left corner Y coordinate.
    :param out_img_res_x: Output pixel resolution in the X axis.
    :param out_img_res_y: Output pixel resolution in the Y axis (negative for north-up).
    :param wkt_string: WKT projection string (or ``None`` / empty string).
    :param gdalformat: Output GDAL format name (e.g., ``'KEA'``, ``'GTIFF'``).
    :param datatype: ``rsgispy.TYPE_*`` constant for the output image.
    :param options: Optional GDAL creation options (dict or sequence of ``'KEY=VALUE'``).
    :param no_data_val: Optional output no-data value.

    .. code:: python

        import numpy as np
        import rsgispy
        import rsgispy.imageutils.arrays

        data = np.zeros((3, 100, 100), dtype=np.float32)
        rsgispy.imageutils.arrays.create_img_from_array(
            data, "out.tif", 500000.0, 5600000.0, 10.0, -10.0, None,
            "GTIFF", rsgispy.TYPE_32FLOAT,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals,c-extension-no-member
    arr = np.asarray(data_arr)
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    elif arr.ndim != 3:
        raise rsgispy.RSGISPyException(
            "The dimensions of the input array must be either 2 or 3."
        )

    n_bands, height, width = arr.shape
    out_rio_dtype = rsgispy.get_rasterio_datatype(datatype)
    out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(datatype))
    creation_opts = normalise_creation_options(options)

    crs_obj = (
        rasterio.crs.CRS.from_wkt(wkt_string.strip())
        if wkt_string and wkt_string.strip()
        else None
    )
    transform = Affine(
        float(out_img_res_x),
        0.0,
        float(tl_x),
        0.0,
        float(out_img_res_y),
        float(tl_y),
    )
    grid = IntersectionGrid(
        width=width,
        height=height,
        transform=transform,
        crs=crs_obj,
        offsets={},
    )

    with ExitStack() as stack:
        dst = open_output_dataset(
            stack=stack,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype=out_rio_dtype,
            grid=grid,
            count=n_bands,
            output_exists=False,
            creation_opts=creation_opts,
        )
        if no_data_val is not None:
            dst.nodata = no_data_val
        dst.write(arr.astype(out_np_dtype, copy=False))


def create_img_from_array_ref_img(
    data_arr: np.ndarray,
    output_img: str,
    ref_img: str,
    gdalformat: str,
    datatype: int,
    options: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
    no_data_val: Optional[float] = None,
) -> None:
    """
    Create a raster image from a NumPy array using spatial header information
    from ``ref_img``.

    :param data_arr: Input NumPy array of shape ``(bands, height, width)`` or
                     ``(height, width)``.
    :param output_img: Path to the output image file.
    :param ref_img: Reference image file path with matching width and height.
    :param gdalformat: Output GDAL format name (e.g., ``'KEA'``, ``'GTIFF'``).
    :param datatype: ``rsgispy.TYPE_*`` constant for the output image.
    :param options: Optional GDAL creation options (dict or sequence of ``'KEY=VALUE'``).
    :param no_data_val: Optional output no-data value.

    .. code:: python

        import numpy as np
        import rsgispy
        import rsgispy.imageutils.arrays

        data = np.ones((100, 100), dtype=np.uint8)
        rsgispy.imageutils.arrays.create_img_from_array_ref_img(
            data, "out.tif", "ref.tif", "GTIFF", rsgispy.TYPE_8UINT
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    arr = np.asarray(data_arr)
    if arr.ndim not in (2, 3):
        raise rsgispy.RSGISPyException(
            "The dimensions of the input array must be either 2 or 3."
        )
    height, width = arr.shape[-2], arr.shape[-1]

    if (width, height) != get_img_size(ref_img):
        raise rsgispy.RSGISPyException(
            "The input reference image size must be the same as the array - "
            "otherwise BBOX and resolution will not match."
        )

    img_bbox = get_img_bbox(ref_img)
    x_res, y_res = get_img_res(ref_img, abs_vals=False)

    create_img_from_array(
        data_arr=arr,
        output_img=output_img,
        tl_x=img_bbox[0],
        tl_y=img_bbox[3],
        out_img_res_x=x_res,
        out_img_res_y=y_res,
        wkt_string=get_wkt_proj_from_img(ref_img),
        gdalformat=gdalformat,
        datatype=datatype,
        options=options,
        no_data_val=no_data_val,
    )


def get_img_coords_pxl_values(
    input_img: str,
    img_band: int,
    x_coords: Union[np.ndarray, Sequence[float]],
    y_coords: Union[np.ndarray, Sequence[float]],
) -> np.ndarray:
    """
    Extract pixel values from ``img_band`` of ``input_img`` at spatial
    coordinates ``(x_coords, y_coords)`` in the image CRS.

    :param input_img: Path to the input image file.
    :param img_band: 1-based band index.
    :param x_coords: Array or sequence of spatial X coordinates.
    :param y_coords: Array or sequence of spatial Y coordinates.
    :return: 1D NumPy float array of extracted pixel values.

    .. code:: python

        import rsgispy.imageutils.arrays

        vals = rsgispy.imageutils.arrays.get_img_coords_pxl_values(
            "img.tif", 1, [500050.0, 500100.0], [5599950.0, 5599900.0]
        )
    """
    x_arr = np.asarray(x_coords, dtype=np.float64)
    y_arr = np.asarray(y_coords, dtype=np.float64)
    if x_arr.shape != y_arr.shape:
        raise rsgispy.RSGISPyException(
            "The X and Y image coordinates are not the same."
        )

    img_bbox = get_img_bbox(input_img)
    img_res_x, img_res_y = get_img_res(input_img, abs_vals=True)
    x_pxl_locs = np.floor((x_arr - img_bbox[0]) / img_res_x).astype(int)
    y_pxl_locs = np.floor((img_bbox[3] - y_arr) / img_res_y).astype(int)

    return get_img_pxl_values(input_img, img_band, x_pxl_locs, y_pxl_locs)


def get_img_pxl_values(
    input_img: str,
    img_band: int,
    x_coords: Union[np.ndarray, Sequence[int]],
    y_coords: Union[np.ndarray, Sequence[int]],
) -> np.ndarray:
    """
    Extract pixel values from ``img_band`` of ``input_img`` at pixel coordinates
    ``(x_coords, y_coords)``.

    :param input_img: Path to the input image file.
    :param img_band: 1-based band index.
    :param x_coords: Array or sequence of X pixel indices (``0 <= x < width``).
    :param y_coords: Array or sequence of Y pixel indices (``0 <= y < height``).
    :return: 1D NumPy float array of extracted pixel values.

    .. code:: python

        import rsgispy.imageutils.arrays

        vals = rsgispy.imageutils.arrays.get_img_pxl_values(
            "img.tif", 1, [10, 20], [15, 25]
        )
    """
    ensure_file_exists(input_img)
    x_arr = np.asarray(x_coords, dtype=np.int64)
    y_arr = np.asarray(y_coords, dtype=np.int64)
    if x_arr.shape != y_arr.shape:
        raise rsgispy.RSGISPyException(
            "The X and Y image coordinates are not the same."
        )

    with rasterio.open(input_img, "r") as src:
        validate_band_index(img_band, src.count, input_img)
        if np.any((x_arr < 0) | (x_arr >= src.width)) or np.any(
            (y_arr < 0) | (y_arr >= src.height)
        ):
            raise rsgispy.RSGISPyException(
                "Pixel coordinates are outside the image dimensions."
            )
        img_data = src.read(img_band)

    return img_data[y_arr, x_arr].astype(float)


def set_img_pxl_values(
    input_img: str,
    img_band: int,
    x_coords: Union[np.ndarray, Sequence[int]],
    y_coords: Union[np.ndarray, Sequence[int]],
    pxl_value: float = 1,
) -> None:
    """
    Set defined pixel coordinates ``(x_coords, y_coords)`` in ``img_band`` of
    ``input_img`` to ``pxl_value`` in place.

    :param input_img: Path to the input image file.
    :param img_band: 1-based band index.
    :param x_coords: Array or sequence of X pixel indices.
    :param y_coords: Array or sequence of Y pixel indices.
    :param pxl_value: Value to assign to the specified pixels (Default = ``1``).

    .. code:: python

        import rsgispy.imageutils.arrays

        rsgispy.imageutils.arrays.set_img_pxl_values(
            "img.tif", 1, [10, 20], [15, 25], pxl_value=0
        )
    """
    ensure_file_exists(input_img)
    x_arr = np.asarray(x_coords, dtype=np.int64)
    y_arr = np.asarray(y_coords, dtype=np.int64)
    if x_arr.shape != y_arr.shape:
        raise rsgispy.RSGISPyException(
            "The X and Y image coordinates are not the same."
        )

    with rasterio.open(input_img, "r+") as dst:
        validate_band_index(img_band, dst.count, input_img)
        if np.any((x_arr < 0) | (x_arr >= dst.width)) or np.any(
            (y_arr < 0) | (y_arr >= dst.height)
        ):
            raise rsgispy.RSGISPyException(
                "Pixel coordinates are outside the image dimensions."
            )
        img_data = dst.read(img_band)
        img_data[y_arr, x_arr] = pxl_value
        dst.write(img_data, img_band)


def get_img_pxl_column(
    input_img: str, x_pxl_coord: int, y_pxl_coord: int
) -> np.ndarray:
    """
    Extract all band values for a single pixel ``(x_pxl_coord, y_pxl_coord)``.

    :param input_img: Path to the input image file.
    :param x_pxl_coord: X pixel coordinate (0-based column index).
    :param y_pxl_coord: Y pixel coordinate (0-based row index).
    :return: 1D float32 NumPy array of length ``n_bands``.

    .. code:: python

        import rsgispy.imageutils.arrays

        pxl_col = rsgispy.imageutils.arrays.get_img_pxl_column("img.tif", 10, 15)
    """
    ensure_file_exists(input_img)
    col_idx = int(x_pxl_coord)
    row_idx = int(y_pxl_coord)
    with rasterio.open(input_img, "r") as src:
        if (
            col_idx < 0
            or col_idx >= src.width
            or row_idx < 0
            or row_idx >= src.height
        ):
            raise rsgispy.RSGISPyException(
                f"Pixel ({col_idx}, {row_idx}) is outside image dimensions "
                f"({src.width} x {src.height})."
            )
        vals = src.read(window=Window(col_idx, row_idx, 1, 1))
    return vals[:, 0, 0].astype(np.float32)


def get_img_column(
    input_img: str, x_coord: float, y_coord: float
) -> np.ndarray:
    """
    Extract all band values for a single pixel at spatial coordinate
    ``(x_coord, y_coord)`` in ``input_img``'s CRS.

    :param input_img: Path to the input image file.
    :param x_coord: Spatial X coordinate.
    :param y_coord: Spatial Y coordinate.
    :return: 1D float32 NumPy array of length ``n_bands``.

    .. code:: python

        import rsgispy.imageutils.arrays

        pxl_col = rsgispy.imageutils.arrays.get_img_column(
            "img.tif", 500050.0, 5599950.0
        )
    """
    img_bbox = get_img_bbox(input_img)
    img_res_x, img_res_y = get_img_res(input_img, abs_vals=True)
    x_pxl_coord = int(math.floor((float(x_coord) - img_bbox[0]) / img_res_x))
    y_pxl_coord = int(math.floor((img_bbox[3] - float(y_coord)) / img_res_y))
    return get_img_pxl_column(input_img, x_pxl_coord, y_pxl_coord)


def get_img_band_pxl_data(
    input_img: str, img_band: int, bbox_sub: Sequence[float]
) -> np.ndarray:
    """
    Read a spatial subset ``bbox_sub`` ``(MinX, MaxX, MinY, MaxY)`` from band
    ``img_band`` of ``input_img`` into a 2D NumPy array.

    :param input_img: Path to the input image file.
    :param img_band: 1-based band index.
    :param bbox_sub: Bounding box ``(MinX, MaxX, MinY, MaxY)`` contained within
                     ``input_img``.
    :return: 2D NumPy array ``(height, width)`` with the subset pixel values.

    .. code:: python

        import rsgispy.imageutils.arrays

        arr = rsgispy.imageutils.arrays.get_img_band_pxl_data(
            "img.tif", 1, [500000.0, 500500.0, 5599500.0, 5600000.0]
        )
    """
    # pylint: disable=too-many-locals
    ensure_file_exists(input_img)
    sub_tuple = (
        float(bbox_sub[0]),
        float(bbox_sub[1]),
        float(bbox_sub[2]),
        float(bbox_sub[3]),
    )
    img_x_res, img_y_res = get_img_res(input_img, abs_vals=True)
    img_bbox = get_img_bbox(input_img)
    img_x_size, img_y_size = get_img_size(input_img)

    with rasterio.open(input_img, "r") as src:
        validate_band_index(img_band, src.count, input_img)
        if not does_bbox_contain(img_bbox, sub_tuple):
            raise rsgispy.RSGISPyException("Subset is not within the image.")

        sub_width = min(
            img_x_size, int(math.ceil((sub_tuple[1] - sub_tuple[0]) / img_x_res))
        )
        sub_height = min(
            img_y_size, int(math.ceil((sub_tuple[3] - sub_tuple[2]) / img_y_res))
        )
        if sub_width < 1 or sub_height < 1:
            raise rsgispy.RSGISPyException(
                f"Subset size is less than 1: {sub_width} x {sub_height}."
            )

        pxl_x_off = (
            0
            if sub_width == img_x_size
            else int(math.floor(((sub_tuple[0] - img_bbox[0]) / img_x_res) + 0.5))
        )
        pxl_y_off = (
            0
            if sub_height == img_y_size
            else int(math.floor(((img_bbox[3] - sub_tuple[3]) / img_y_res) + 0.5))
        )

        return src.read(
            img_band,
            window=Window(pxl_x_off, pxl_y_off, sub_width, sub_height),
        )


def get_img_data_as_arr(
    input_img: str, img_bands: Optional[Sequence[int]] = None
) -> np.ndarray:
    """
    Read the full image data from ``input_img`` into a 3D NumPy array
    ``(bands, height, width)``.

    :param input_img: Path to the input image file.
    :param img_bands: Optional sequence of 1-based band indices to read in order.
                      If ``None``, all bands are read.
    :return: 3D NumPy array ``(bands, height, width)``.

    .. code:: python

        import rsgispy.imageutils.arrays

        arr = rsgispy.imageutils.arrays.get_img_data_as_arr("img.tif", [1, 2, 3])
    """
    ensure_file_exists(input_img)
    with rasterio.open(input_img, "r") as src:
        if img_bands is None:
            return src.read()
        band_list = [int(b) for b in img_bands]
        for bidx in band_list:
            validate_band_index(bidx, src.count, input_img)
        return np.stack([src.read(bidx) for bidx in band_list], axis=0)
