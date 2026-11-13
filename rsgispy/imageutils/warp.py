"""
The imageutils.warp module provides raster resampling and reprojection
functions using ``rasterio.warp.reproject``.
"""

import os
from typing import Any, Mapping, Optional, Sequence, Union

import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject as rio_reproject
from rasterio.warp import transform as rio_transform_coords

import rsgispy
from rsgispy.imageutils._helpers import parse_epsg_crs
from rsgispy.imageutils.blank import create_blank_img, create_copy_img

__all__ = [
    "resample_img_to_match",
    "reproject_image",
    "gdal_warp",
]


def resample_img_to_match(
    in_ref_img: str,
    in_process_img: str,
    output_img: str,
    gdalformat: str,
    interp_method: Union[int, Resampling] = rsgispy.INTERP_NEAREST_NEIGHBOUR,
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    multicore: bool = False,
    options: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
) -> None:
    """
    Resample ``in_process_img`` to match the projection, extent, and pixel size
    of ``in_ref_img`` using ``rasterio.warp.reproject``.

    :param in_ref_img: Reference image defining the target grid and projection.
    :param in_process_img: Input image to be resampled.
    :param output_img: Path to the output resampled image file.
    :param gdalformat: Output GDAL format name (e.g., 'GTIFF', 'KEA').
    :param interp_method: Interpolation method ``rsgispy.INTERP_*`` or
                          ``rasterio.enums.Resampling`` (Default = INTERP_NEAREST_NEIGHBOUR).
    :param datatype: Optional ``rsgispy.TYPE_*`` constant for the output image
                     (defaults to the data type of ``in_process_img``).
    :param no_data_val: Optional no-data value (defaults to ``in_process_img`` nodata).
    :param multicore: Whether to use multiple CPU threads for warping (Default = False).
    :param options: Optional GDAL creation options (dict or list of ``'KEY=VALUE'``).
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    if not os.path.exists(in_ref_img):
        raise rsgispy.RSGISPyException(
            f"Reference image file does not exist: '{in_ref_img}'."
        )
    if not os.path.exists(in_process_img):
        raise rsgispy.RSGISPyException(
            f"Input process image file does not exist: '{in_process_img}'."
        )

    resampling_alg = rsgispy.get_rasterio_interp_type(interp_method)

    with rasterio.open(in_process_img, "r") as src_proc:
        num_bands = src_proc.count
        if no_data_val is None:
            no_data_val = src_proc.nodata
        if datatype is None:
            datatype = rsgispy.get_rsgis_datatype(src_proc.dtypes[0])

    back_val = float(no_data_val) if no_data_val is not None else 0.0

    create_copy_img(
        input_img=in_ref_img,
        output_img=output_img,
        n_bands=num_bands,
        pxl_val=back_val,
        gdalformat=gdalformat,
        datatype=datatype,
        options=options,
        no_data_val=no_data_val,
    )

    num_threads = (os.cpu_count() or 1) if multicore else 1
    band_indices = list(range(1, num_bands + 1))

    with rasterio.open(in_process_img, "r") as src_proc, rasterio.open(
        output_img, "r+"
    ) as dst:
        rio_reproject(
            source=rasterio.band(src_proc, band_indices),
            destination=rasterio.band(dst, band_indices),
            resampling=resampling_alg,
            src_nodata=no_data_val,
            dst_nodata=no_data_val,
            num_threads=num_threads,
        )


def _resolve_reproject_res(
    out_pxl_res: Union[float, int, str],
    in_res: float,
    x_size: int,
    y_size: int,
    tl_x_out: float,
    tl_y_out: float,
    br_x_out: float,
    br_y_out: float,
) -> float:
    """
    Determine the output pixel resolution for :func:`reproject_image`.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    res_str = str(out_pxl_res).strip()
    try:
        return abs(float(res_str))
    except ValueError:
        pass

    if res_str == "image":
        return in_res
    if res_str == "auto":
        x_out_res = abs(br_x_out - tl_x_out) / x_size
        y_out_res = abs(tl_y_out - br_y_out) / y_size
        return min(x_out_res, y_out_res)

    raise rsgispy.RSGISPyException(
        f"Was not able to define the output resolution. Check input: '{res_str}'"
    )


def reproject_image(
    input_img: str,
    output_img: str,
    out_epsg: int,
    gdalformat: str = "GTIFF",
    interp_method: Union[int, Resampling] = rsgispy.INTERP_NEAREST_NEIGHBOUR,
    in_epsg: Optional[int] = None,
    no_data_val: float = 0.0,
    out_pxl_res: Union[float, int, str] = "image",
    snap_to_grid: bool = True,
    multicore: bool = False,
    gdal_options: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
) -> None:
    """
    Reproject ``input_img`` to the coordinate system specified by ``out_epsg``
    using ``rasterio.warp.reproject``.

    :param input_img: Path to the input image file.
    :param output_img: Path to the output image file.
    :param out_epsg: EPSG integer code representing the output projection (e.g., 3857).
    :param gdalformat: Output GDAL format name (Default = 'GTIFF').
    :param interp_method: Interpolation method ``rsgispy.INTERP_*`` or
                          ``rasterio.enums.Resampling`` (Default = INTERP_NEAREST_NEIGHBOUR).
    :param in_epsg: Optional EPSG integer code overriding the input image projection.
    :param no_data_val: No-data value for the reprojected image (Default = 0.0).
    :param out_pxl_res: Output resolution mode: ``'image'`` (match input resolution),
                        ``'auto'`` (maintain input dimensions), or a numeric value
                        for square pixel resolution (Default = 'image').
    :param snap_to_grid: Whether to snap the top-left pixel to a multiple of the
                         output pixel resolution (Default = True).
    :param multicore: Whether to use multiple CPU threads for warping (Default = False).
    :param gdal_options: Optional GDAL creation options (dict or list of ``'KEY=VALUE'``).
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals,too-many-statements
    if not os.path.exists(input_img):
        raise rsgispy.RSGISPyException(
            f"The input image file does not exist: '{input_img}'"
        )

    resampling_alg = rsgispy.get_rasterio_interp_type(interp_method)
    out_crs = parse_epsg_crs(out_epsg)
    if out_crs is None:
        raise rsgispy.RSGISPyException(
            "An output EPSG code (out_epsg) must be provided."
        )

    with rasterio.open(input_img, "r") as src:
        in_crs = parse_epsg_crs(in_epsg) if in_epsg is not None else src.crs
        if in_crs is None:
            raise rsgispy.RSGISPyException(
                f"Could not determine projection for input image: '{input_img}'"
            )

        x_pxl_res = abs(src.transform.a)
        y_pxl_res = abs(src.transform.e)
        in_res = min(x_pxl_res, y_pxl_res)
        x_size, y_size = src.width, src.height
        tl_x_in, tl_y_in = src.transform.c, src.transform.f
        br_x_in = tl_x_in + (x_size * x_pxl_res)
        br_y_in = tl_y_in - (y_size * y_pxl_res)
        num_bands = src.count
        rsgis_dtype = rsgispy.get_rsgis_datatype(src.dtypes[0])

    xs_out, ys_out = rio_transform_coords(
        in_crs,
        out_crs,
        [tl_x_in, br_x_in, br_x_in, tl_x_in],
        [tl_y_in, br_y_in, tl_y_in, br_y_in],
    )
    x_min, x_max = float(min(xs_out)), float(max(xs_out))
    y_min, y_max = float(min(ys_out)), float(max(ys_out))

    out_res = _resolve_reproject_res(
        out_pxl_res=out_pxl_res,
        in_res=in_res,
        x_size=x_size,
        y_size=y_size,
        tl_x_out=float(xs_out[0]),
        tl_y_out=float(ys_out[0]),
        br_x_out=float(xs_out[1]),
        br_y_out=float(ys_out[1]),
    )

    out_tl_x, out_tl_y = x_min, y_max
    out_width = int(round((x_max - x_min) / out_res)) + 1
    out_height = int(round((y_max - y_min) / out_res)) + 1

    if snap_to_grid:
        out_tl_x = (out_tl_x - (out_tl_x % out_res)) - (5 * out_res)
        out_tl_y = ((out_tl_y - (out_tl_y % out_res)) + out_res) + (5 * out_res)
        out_width = int(round((x_max - x_min) / out_res)) + 10
        out_height = int(round((y_max - y_min) / out_res)) + 10

    create_blank_img(
        output_img=output_img,
        n_bands=num_bands,
        width=out_width,
        height=out_height,
        tl_x=out_tl_x,
        tl_y=out_tl_y,
        res_x=out_res,
        res_y=-out_res,
        pxl_val=float(no_data_val),
        epsg_code=int(out_epsg),
        gdalformat=gdalformat,
        datatype=rsgis_dtype,
        options=gdal_options,
        no_data_val=float(no_data_val),
    )

    num_threads = (os.cpu_count() or 1) if multicore else 1
    band_indices = list(range(1, num_bands + 1))

    with rasterio.open(input_img, "r") as src, rasterio.open(
        output_img, "r+"
    ) as dst:
        rio_reproject(
            source=rasterio.band(src, band_indices),
            destination=rasterio.band(dst, band_indices),
            src_crs=in_crs,
            dst_crs=out_crs,
            resampling=resampling_alg,
            src_nodata=no_data_val,
            dst_nodata=no_data_val,
            num_threads=num_threads,
        )


def gdal_warp(
    input_img: str,
    output_img: str,
    out_epsg: int,
    interp_method: Union[int, Resampling] = rsgispy.INTERP_NEAREST_NEIGHBOUR,
    gdalformat: str = "KEA",
    use_multi_threaded: bool = True,
    options: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
) -> None:
    """
    Reproject ``input_img`` to ``out_epsg`` using ``rasterio.warp.reproject``
    with automatic output resolution.

    :param input_img: Path to the input image file.
    :param output_img: Path to the output warped image file.
    :param out_epsg: Target EPSG integer code.
    :param interp_method: Interpolation method ``rsgispy.INTERP_*`` or
                          ``rasterio.enums.Resampling`` (Default = INTERP_NEAREST_NEIGHBOUR).
    :param gdalformat: Output GDAL format name (Default = ``'KEA'``).
    :param use_multi_threaded: Use multiple threads when warping (Default = ``True``).
    :param options: Optional GDAL creation options (dict or sequence of ``'KEY=VALUE'``).
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    if not os.path.exists(input_img):
        raise rsgispy.RSGISPyException(
            f"Input image file does not exist: '{input_img}'."
        )

    with rasterio.open(input_img, "r") as src:
        src_nd = src.nodata

    no_data_val = 0.0 if src_nd is None else float(src_nd)
    reproject_image(
        input_img=input_img,
        output_img=output_img,
        out_epsg=int(out_epsg),
        gdalformat=gdalformat,
        interp_method=interp_method,
        no_data_val=no_data_val,
        out_pxl_res="auto",
        snap_to_grid=False,
        multicore=use_multi_threaded,
        gdal_options=options,
    )
