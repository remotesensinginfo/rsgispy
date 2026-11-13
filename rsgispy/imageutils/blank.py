"""
The imageutils.blank module provides functions for creating blank, copied,
buffered, and bbox-defined raster datasets using ``rasterio``.
"""

from contextlib import ExitStack
import math
import os
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.windows import Window

import rsgispy
from rsgispy.imageprocess import (
    IntersectionGrid,
    get_default_block_size,
    open_output_dataset,
)
from rsgispy.imageutils._helpers import (
    get_epsg_from_dataset,
    normalise_creation_options,
    parse_epsg_crs,
)

__all__ = [
    "create_blank_img",
    "create_copy_img",
    "create_blank_buf_img_from_ref_img",
    "create_blank_img_from_bbox",
]


def _find_extent_on_grid(
    base_extent: Tuple[float, float, float, float],
    base_grid: float,
    full_contain: bool = True,
) -> Tuple[float, float, float, float]:
    """
    Snap ``base_extent`` ``(x_min, x_max, y_min, y_max)`` onto a grid of
    resolution ``base_grid``.
    """
    # pylint: disable=too-many-locals
    x_min, x_max, y_min, y_max = (float(v) for v in base_extent[:4])
    x_min_floor = int(math.floor(x_min))
    y_max_ceil = int(math.ceil(y_max))
    n_x_cells = math.floor((x_min - x_min_floor) / base_grid)
    n_y_cells = math.floor((y_max_ceil - y_max) / base_grid)
    x_min_out = x_min_floor + (n_x_cells * base_grid)
    y_max_out = y_max_ceil - (n_y_cells * base_grid)

    round_fn = math.ceil if full_contain else math.floor
    n_pxl_x = round_fn((x_max - x_min_out) / base_grid)
    n_pxl_y = round_fn(abs(y_max_out - y_min) / base_grid)
    return (
        x_min_out,
        x_min_out + (n_pxl_x * base_grid),
        y_max_out - (n_pxl_y * base_grid),
        y_max_out,
    )


def create_blank_img(
    output_img: str,
    n_bands: int,
    width: int,
    height: int,
    tl_x: float,
    tl_y: float,
    res_x: float,
    res_y: float,
    pxl_val: float,
    epsg_code: int,
    gdalformat: str = "GTIFF",
    datatype: int = rsgispy.TYPE_32FLOAT,
    options: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
    no_data_val: Optional[float] = None,
) -> None:
    """
    Create a new blank raster image with the specified dimensions, resolution,
    EPSG coordinate system, and initial pixel value using ``rasterio``.

    If ``options`` is ``None`` or empty, any format creation options configured via
    environment variables (e.g., :func:`set_env_vars_deflate_gtiff_outs`) are
    automatically applied.

    :param output_img: Path to the output image file.
    :param n_bands: Number of image bands in the output image.
    :param width: Width of the output image in pixels.
    :param height: Height of the output image in pixels.
    :param tl_x: Top-left pixel X coordinate (eastings).
    :param tl_y: Top-left pixel Y coordinate (northings).
    :param res_x: Pixel resolution along the X axis.
    :param res_y: Pixel resolution along the Y axis (positive or negative).
    :param pxl_val: Initial pixel value for all bands.
    :param epsg_code: EPSG integer code defining the output projection (e.g., 32630).
    :param gdalformat: Output GDAL format name (e.g., 'GTIFF', 'KEA').
    :param datatype: ``rsgispy.TYPE_*`` constant for the output image.
    :param options: Optional GDAL creation options (dict or list of ``'KEY=VALUE'``).
    :param no_data_val: Optional no-data value to set on the output dataset.

    .. code:: python

        import rsgispy
        import rsgispy.imageutils.blank

        rsgispy.imageutils.blank.create_blank_img(
            "blank.tif", 1, 100, 100, 500000.0, 5600000.0, 10.0, -10.0,
            0.0, 32630, "GTIFF", rsgispy.TYPE_32FLOAT,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    if n_bands < 1 or width < 1 or height < 1:
        raise rsgispy.RSGISPyException(
            "n_bands, width, and height must all be >= 1."
        )

    out_rio_dtype = rsgispy.get_rasterio_datatype(datatype)
    out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(datatype))
    crs_obj = parse_epsg_crs(epsg_code)
    transform = Affine(
        abs(float(res_x)),
        0.0,
        float(tl_x),
        0.0,
        -abs(float(res_y)),
        float(tl_y),
    )
    grid = IntersectionGrid(
        crs=crs_obj,
        transform=transform,
        width=int(width),
        height=int(height),
        offsets={},
    )
    creation_opts = normalise_creation_options(options)

    with ExitStack() as stack:
        dst = open_output_dataset(
            stack,
            output_img,
            gdalformat,
            out_rio_dtype,
            grid,
            int(n_bands),
            False,
            creation_opts,
        )
        if no_data_val is not None:
            dst.nodata = no_data_val

        block_size = get_default_block_size()
        for row_off in range(0, grid.height, block_size):
            win_h = min(block_size, grid.height - row_off)
            for col_off in range(0, grid.width, block_size):
                win_w = min(block_size, grid.width - col_off)
                win = Window(col_off, row_off, win_w, win_h)
                block = np.full(
                    (int(n_bands), win_h, win_w), pxl_val, dtype=out_np_dtype
                )
                dst.write(block, window=win)


def create_copy_img(
    input_img: str,
    output_img: str,
    n_bands: int,
    pxl_val: float,
    gdalformat: str,
    datatype: int,
    options: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
    no_data_val: Optional[float] = None,
) -> None:
    """
    Create a new blank raster image matching the spatial extent, resolution,
    dimensions, and EPSG projection of ``input_img``.

    :param input_img: Path to the reference input image.
    :param output_img: Path to the output image file.
    :param n_bands: Number of image bands in the output image.
    :param pxl_val: Initial pixel value for all bands.
    :param gdalformat: Output GDAL format name (e.g., 'GTIFF', 'KEA').
    :param datatype: ``rsgispy.TYPE_*`` constant for the output image.
    :param options: Optional GDAL creation options (dict or list of ``'KEY=VALUE'``).
    :param no_data_val: Optional no-data value to set on the output dataset.

    .. code:: python

        import rsgispy
        import rsgispy.imageutils.blank

        rsgispy.imageutils.blank.create_copy_img(
            "in_img.tif", "out_img.tif", 1, 0, "GTIFF", rsgispy.TYPE_8UINT
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    if not os.path.exists(input_img):
        raise rsgispy.RSGISPyException(
            f"Input image file does not exist: '{input_img}'."
        )

    with rasterio.open(input_img, "r") as src:
        width = src.width
        height = src.height
        tl_x = src.transform.c
        tl_y = src.transform.f
        res_x = src.transform.a
        res_y = src.transform.e
        epsg_code = get_epsg_from_dataset(src, input_img)

    create_blank_img(
        output_img=output_img,
        n_bands=n_bands,
        width=width,
        height=height,
        tl_x=tl_x,
        tl_y=tl_y,
        res_x=res_x,
        res_y=res_y,
        pxl_val=pxl_val,
        epsg_code=epsg_code,
        gdalformat=gdalformat,
        datatype=datatype,
        options=options,
        no_data_val=no_data_val,
    )


def create_blank_buf_img_from_ref_img(
    input_img: str,
    output_img: str,
    gdalformat: str,
    datatype: int,
    buf_pxl_ext: Optional[int] = None,
    buf_spt_ext: Optional[float] = None,
    no_data_val: Optional[float] = None,
    options: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
) -> None:
    """
    Create a new blank image based on ``input_img`` buffered on all sides by
    either ``buf_pxl_ext`` pixels or ``buf_spt_ext`` spatial units.

    :param input_img: Path to the reference input image.
    :param output_img: Path to the output image file.
    :param gdalformat: Output GDAL format name (e.g., 'GTIFF', 'KEA').
    :param datatype: ``rsgispy.TYPE_*`` constant for the output image.
    :param buf_pxl_ext: Buffer amount in pixels (mutually exclusive with ``buf_spt_ext``).
    :param buf_spt_ext: Buffer amount in spatial projection units (mutually exclusive
                        with ``buf_pxl_ext``).
    :param no_data_val: Optional no-data value. If ``None``, read from ``input_img``.
    :param options: Optional GDAL creation options (dict or list of ``'KEY=VALUE'``).

    .. code:: python

        import rsgispy
        import rsgispy.imageutils.blank

        rsgispy.imageutils.blank.create_blank_buf_img_from_ref_img(
            "in_img.tif", "buf_img.tif", "GTIFF", rsgispy.TYPE_32FLOAT,
            buf_pxl_ext=10, no_data_val=0.0,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    if (buf_pxl_ext is None) and (buf_spt_ext is None):
        raise rsgispy.RSGISPyException(
            "You must specify either the buf_pxl_ext or buf_spt_ext value."
        )
    if (buf_pxl_ext is not None) and (buf_spt_ext is not None):
        raise rsgispy.RSGISPyException(
            "You cannot specify both the buf_pxl_ext or buf_spt_ext value."
        )
    if not os.path.exists(input_img):
        raise rsgispy.RSGISPyException(
            f"Input image file does not exist: '{input_img}'."
        )

    with rasterio.open(input_img, "r") as src:
        if no_data_val is None:
            no_data_val = src.nodata
            if no_data_val is None:
                raise rsgispy.RSGISPyException(
                    "You must specify a no data value"
                )

        x_res = src.transform.a
        y_res = src.transform.e
        x_res_abs = abs(x_res)
        y_res_abs = abs(y_res)
        x_in_size = src.width
        y_in_size = src.height
        bounds = src.bounds
        n_bands = src.count
        epsg_code = get_epsg_from_dataset(src, input_img)

    if buf_spt_ext is not None:
        buf_pxl_x = math.ceil(buf_spt_ext / x_res_abs)
        buf_pxl_y = math.ceil(buf_spt_ext / y_res_abs)
    else:
        assert buf_pxl_ext is not None
        buf_pxl_x = int(buf_pxl_ext)
        buf_pxl_y = int(buf_pxl_ext)

    create_blank_img(
        output_img=output_img,
        n_bands=n_bands,
        width=x_in_size + (2 * buf_pxl_x),
        height=y_in_size + (2 * buf_pxl_y),
        tl_x=bounds.left - (buf_pxl_x * x_res_abs),
        tl_y=bounds.top + (buf_pxl_y * y_res_abs),
        res_x=x_res,
        res_y=y_res,
        pxl_val=float(no_data_val),
        epsg_code=epsg_code,
        gdalformat=gdalformat,
        datatype=datatype,
        options=options,
        no_data_val=float(no_data_val),
    )


def create_blank_img_from_bbox(
    bbox: Tuple[float, float, float, float],
    epsg_code: int,
    output_img: str,
    out_img_res: float,
    out_img_pxl_val: float,
    out_img_n_bands: int,
    gdalformat: str,
    datatype: int,
    snap_to_grid: bool = False,
    options: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
    no_data_val: Optional[float] = None,
) -> None:
    """
    Create a new blank image file from a bounding box ``(xMin, xMax, yMin, yMax)``
    and an EPSG code.

    :param bbox: Bounding box ``(xMin, xMax, yMin, yMax)`` defining the output extent.
    :param epsg_code: EPSG integer code defining the output projection (e.g., 32630).
    :param output_img: Path to the output image file.
    :param out_img_res: Output pixel resolution (square pixels).
    :param out_img_pxl_val: Initial pixel value for all bands.
    :param out_img_n_bands: Number of image bands in the output image.
    :param gdalformat: Output GDAL format name (e.g., 'GTIFF', 'KEA').
    :param datatype: ``rsgispy.TYPE_*`` constant for the output image.
    :param snap_to_grid: Whether to snap the bounding box onto a grid of
                         whole multiples of ``out_img_res`` (Default = False).
    :param options: Optional GDAL creation options (dict or list of ``'KEY=VALUE'``).
    :param no_data_val: Optional no-data value to set on the output dataset.

    .. code:: python

        import rsgispy
        import rsgispy.imageutils.blank

        rsgispy.imageutils.blank.create_blank_img_from_bbox(
            (500000.0, 501000.0, 5599000.0, 5600000.0),
            32630, "blank_bbox.tif", 10.0, 0.0, 1, "GTIFF", rsgispy.TYPE_8UINT,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    if snap_to_grid:
        bbox = _find_extent_on_grid(bbox, out_img_res, full_contain=True)

    x_min, x_max, y_min, y_max = (float(v) for v in bbox[:4])
    width_coord = x_max - x_min
    height_coord = y_max - y_min

    width_rem = abs(
        width_coord / out_img_res - math.floor(width_coord / out_img_res)
    )
    height_rem = abs(
        height_coord / out_img_res - math.floor(height_coord / out_img_res)
    )

    width = (
        int(math.floor(width_coord / out_img_res))
        if width_rem < 0.0001
        else int(math.ceil(width_coord / out_img_res))
    )
    height = (
        int(math.floor(height_coord / out_img_res))
        if height_rem < 0.0001
        else int(math.ceil(height_coord / out_img_res))
    )

    create_blank_img(
        output_img=output_img,
        n_bands=out_img_n_bands,
        width=width,
        height=height,
        tl_x=x_min,
        tl_y=y_max,
        res_x=out_img_res,
        res_y=-out_img_res,
        pxl_val=out_img_pxl_val,
        epsg_code=epsg_code,
        gdalformat=gdalformat,
        datatype=datatype,
        options=options,
        no_data_val=no_data_val,
    )
