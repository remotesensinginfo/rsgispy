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
import rasterio.errors
from rasterio.transform import Affine
import rasterio.windows
from rasterio.windows import Window

import rsgispy
from rsgispy.imageprocess import (
    is_cog_layout,
    update_cog,
    IntersectionGrid,
    ensure_file_exists,
    open_output_dataset,
    pop_stats_on_close,
    validate_band_index,
)
from rsgispy.imageutils._helpers import (
    normalise_creation_options,
    rsgis_dtype_from_np_dtype,
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
    wkt_string: Optional[str] = None,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
    creation_opts: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
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
    :param gdalformat: Output GDAL format name (e.g., ``'KEA'``, ``'GTIFF'``;
                       Default = ``'GTIFF'``).
    :param datatype: Optional ``rsgispy.TYPE_*`` constant for the output image
                     (defaults to matching ``data_arr.dtype``).
    :param no_data_val: Optional output no-data value (Default = ``None``).
    :param calc_stats: If ``True``, compute image statistics and pyramids
                       (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar when computing
                     statistics (Default = ``True``).
    :param creation_opts: Optional GDAL creation options (dict or sequence of
                          ``'KEY=VALUE'``).

    .. code:: python

        import numpy as np
        import rsgispy
        import rsgispy.imageutils.arrays

        data = np.zeros((3, 100, 100), dtype=np.float32)
        rsgispy.imageutils.arrays.create_img_from_array(
            data, "out.tif", 500000.0, 5600000.0, 10.0, -10.0, None,
            gdalformat="GTIFF", datatype=rsgispy.TYPE_32FLOAT,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,c-extension-no-member
    arr = np.asarray(data_arr)
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    elif arr.ndim != 3:
        raise rsgispy.RSGISPyException(
            "The dimensions of the input array must be either 2 or 3."
        )

    n_bands, height, width = arr.shape
    eff_dtype = (
        rsgis_dtype_from_np_dtype(arr.dtype)
        if datatype is None
        else datatype
    )
    out_rio_dtype = rsgispy.get_rasterio_datatype(eff_dtype)
    out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(eff_dtype))
    norm_opts = normalise_creation_options(creation_opts)

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
            creation_opts=norm_opts,
        )
        if no_data_val is not None:
            dst.nodata = no_data_val
        pop_stats_on_close(
            dst, calc_stats, no_data_val=no_data_val, use_pbar=use_pbar
        )
        dst.write(arr.astype(out_np_dtype, copy=False))


def create_img_from_array_ref_img(
    data_arr: np.ndarray,
    ref_img: str,
    output_img: str,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
    creation_opts: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
) -> None:
    """
    Create a raster image from a NumPy array using spatial header information
    from ``ref_img``.

    :param data_arr: Input NumPy array of shape ``(bands, height, width)`` or
                     ``(height, width)``.
    :param ref_img: Reference image file path with matching width and height.
    :param output_img: Path to the output image file.
    :param gdalformat: Output GDAL format name (e.g., ``'KEA'``, ``'GTIFF'``;
                       Default = ``'GTIFF'``).
    :param datatype: Optional ``rsgispy.TYPE_*`` constant for the output image
                     (defaults to matching ``data_arr.dtype``).
    :param no_data_val: Optional output no-data value (Default = ``None``).
    :param calc_stats: If ``True``, compute image statistics and pyramids
                       (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar when computing
                     statistics (Default = ``True``).
    :param creation_opts: Optional GDAL creation options (dict or sequence of
                          ``'KEY=VALUE'``).

    .. code:: python

        import numpy as np
        import rsgispy
        import rsgispy.imageutils.arrays

        data = np.ones((100, 100), dtype=np.uint8)
        rsgispy.imageutils.arrays.create_img_from_array_ref_img(
            data, "ref.tif", "out.tif",
            gdalformat="GTIFF", datatype=rsgispy.TYPE_8UINT,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    ensure_file_exists(ref_img, label="Reference image")
    arr = np.asarray(data_arr)
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    elif arr.ndim != 3:
        raise rsgispy.RSGISPyException(
            "The dimensions of the input array must be either 2 or 3."
        )
    n_bands, height, width = arr.shape

    eff_dtype = (
        rsgis_dtype_from_np_dtype(arr.dtype)
        if datatype is None
        else datatype
    )
    out_rio_dtype = rsgispy.get_rasterio_datatype(eff_dtype)
    out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(eff_dtype))
    norm_opts = normalise_creation_options(creation_opts)

    with ExitStack() as stack:
        ref_src = stack.enter_context(rasterio.open(ref_img, "r"))
        if (width, height) != (ref_src.width, ref_src.height):
            raise rsgispy.RSGISPyException(
                "The input reference image size must be the same as the array - "
                "otherwise BBOX and resolution will not match."
            )
        grid = IntersectionGrid(
            width=width,
            height=height,
            transform=ref_src.transform,
            crs=ref_src.crs,
            offsets={},
        )
        dst = open_output_dataset(
            stack=stack,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype=out_rio_dtype,
            grid=grid,
            count=n_bands,
            output_exists=False,
            creation_opts=norm_opts,
        )
        if no_data_val is not None:
            dst.nodata = no_data_val
        pop_stats_on_close(
            dst, calc_stats, no_data_val=no_data_val, use_pbar=use_pbar
        )
        dst.write(arr.astype(out_np_dtype, copy=False))


def _read_band_pxls(
    src: Any, img_band: int, rows: np.ndarray, cols: np.ndarray
) -> np.ndarray:
    """
    Values of ``img_band`` at pixel ``(rows, cols)`` (already range-checked),
    read block by block: only the blocks that contain a point are read, so
    sampling a few points does not read the whole band.
    """
    out = np.empty(rows.shape, dtype=src.dtypes[img_band - 1])
    if rows.size == 0:
        return out
    blk_h, blk_w = src.block_shapes[img_band - 1]
    blk_r = rows // blk_h
    blk_c = cols // blk_w
    blk_ids = blk_r * (src.width // blk_w + 1) + blk_c
    uniq, inv = np.unique(blk_ids, return_inverse=True)
    if uniq.size * blk_h * blk_w >= src.width * src.height:
        return src.read(img_band)[rows, cols]
    inv = inv.ravel()
    for u_idx in range(uniq.size):
        sel = inv == u_idx
        r0 = int(blk_r.ravel()[sel][0]) * blk_h
        c0 = int(blk_c.ravel()[sel][0]) * blk_w
        win = Window(c0, r0, min(blk_w, src.width - c0), min(blk_h, src.height - r0))
        blk = src.read(img_band, window=win)
        out.ravel()[sel] = blk[rows.ravel()[sel] - r0, cols.ravel()[sel] - c0]
    return out


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
    ensure_file_exists(input_img)
    x_arr = np.asarray(x_coords, dtype=np.float64)
    y_arr = np.asarray(y_coords, dtype=np.float64)
    if x_arr.shape != y_arr.shape:
        raise rsgispy.RSGISPyException(
            "The X and Y image coordinates are not the same."
        )

    with rasterio.open(input_img, "r") as src:
        validate_band_index(img_band, src.count, input_img)
        col_f, row_f = ~src.transform @ (x_arr, y_arr)
        x_pxl_locs = np.floor(col_f).astype(np.int64)
        y_pxl_locs = np.floor(row_f).astype(np.int64)
        if np.any((x_pxl_locs < 0) | (x_pxl_locs >= src.width)) or np.any(
            (y_pxl_locs < 0) | (y_pxl_locs >= src.height)
        ):
            raise rsgispy.RSGISPyException(
                "Pixel coordinates are outside the image dimensions."
            )
        vals = _read_band_pxls(src, img_band, y_pxl_locs, x_pxl_locs)

    return vals.astype(float)


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
        vals = _read_band_pxls(src, img_band, y_arr, x_arr)

    return vals.astype(float)


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

    A Cloud Optimised GeoTIFF is not edited in place (which would break its
    layout): it is rebuilt with :func:`rsgispy.imageprocess.update_cog`.

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

    if is_cog_layout(input_img):
        update_cog(
            input_img,
            lambda tmp_img: set_img_pxl_values(
                tmp_img, img_band, x_arr, y_arr, pxl_value=pxl_value
            ),
        )
        return

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
    ensure_file_exists(input_img)
    with rasterio.open(input_img, "r") as src:
        col_f, row_f = ~src.transform @ (float(x_coord), float(y_coord))
        col_idx = int(math.floor(col_f))
        row_idx = int(math.floor(row_f))
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

    with rasterio.open(input_img, "r") as src:
        validate_band_index(img_band, src.count, input_img)
        bounds = src.bounds
        img_bbox = (
            float(bounds.left),
            float(bounds.right),
            float(bounds.bottom),
            float(bounds.top),
        )
        if not does_bbox_contain(img_bbox, sub_tuple):
            raise rsgispy.RSGISPyException("Subset is not within the image.")

        full_win = Window(0, 0, src.width, src.height)
        try:
            raw_win = rasterio.windows.from_bounds(
                left=sub_tuple[0],
                bottom=sub_tuple[2],
                right=sub_tuple[1],
                top=sub_tuple[3],
                transform=src.transform,
            ).intersection(full_win)
        except rasterio.errors.WindowError as err:
            raise rsgispy.RSGISPyException(
                "Subset does not overlap valid image pixels."
            ) from err

        col_off = int(math.floor(raw_win.col_off + 1e-9))
        row_off = int(math.floor(raw_win.row_off + 1e-9))
        col_end = int(math.ceil(raw_win.col_off + raw_win.width - 1e-9))
        row_end = int(math.ceil(raw_win.row_off + raw_win.height - 1e-9))
        sub_width = col_end - col_off
        sub_height = row_end - row_off
        if sub_width < 1 or sub_height < 1:
            raise rsgispy.RSGISPyException(
                f"Subset size is less than 1: {sub_width} x {sub_height}."
            )

        clamped_win = Window(col_off, row_off, sub_width, sub_height).intersection(
            full_win
        )
        return src.read(img_band, window=clamped_win)


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
