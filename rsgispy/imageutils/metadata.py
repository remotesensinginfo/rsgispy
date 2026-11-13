"""
Raster metadata, spatial geometry, projection, and GCP utilities for
:mod:`rsgispy.imageutils`.
"""

import math
import os
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import pyproj
import rasterio
import rasterio.crs
from rasterio.transform import Affine

import rsgispy
from rsgispy.imageprocess import bbox_from_dataset
from rsgispy.imageutils._helpers import (
    ensure_img_exists as _ensure_img_exists,
    parse_epsg_crs as _parse_epsg_crs,
    validate_band_index as _validate_band_index,
)
from rsgispy.tools.geometrytools import (
    bbox_intersection,
    do_bboxes_intersect,
    reproj_bbox,
)


def get_img_no_data_value(
    input_img: str, img_band: int = 1
) -> Optional[float]:
    """
    Retrieve the no-data value for an image band (default band 1).

    :param input_img: Path to the input image file.
    :param img_band: 1-based band index for which the no-data value is returned
                     (Default = ``1``).
    :return: No-data value for the band, or ``None`` if unset.

    .. code:: python

        import rsgispy.imageutils.metadata
        no_data_val = rsgispy.imageutils.metadata.get_img_no_data_value("img.tif")
    """
    _ensure_img_exists(input_img)
    with rasterio.open(input_img, "r") as src:
        _validate_band_index(img_band, src.count, input_img)
        return src.nodatavals[img_band - 1]


def set_img_no_data_value(
    input_img: str, no_data_val: float, img_band: Optional[int] = None
) -> None:
    """
    Set the no-data value on ``input_img`` (for a specific band or all bands).

    :param input_img: Path to the input image file.
    :param no_data_val: No-data value to assign.
    :param img_band: Optional 1-based band index. If ``None``, assigns to all
                     bands (Default = ``None``).

    .. code:: python

        import rsgispy.imageutils.metadata
        rsgispy.imageutils.metadata.set_img_no_data_value("img.tif", 0.0, 1)
    """
    _ensure_img_exists(input_img)
    with rasterio.open(input_img, "r+") as dst:
        if img_band is None:
            dst.nodata = no_data_val
        else:
            _validate_band_index(img_band, dst.count, input_img)
            if dst.count == 1:
                dst.nodata = no_data_val
            else:
                vals = list(dst.nodatavals)
                vals[img_band - 1] = no_data_val
                # pylint: disable=protected-access
                dst._set_nodatavals(vals)


def get_img_res(
    input_img: str, abs_vals: bool = False
) -> Tuple[float, float]:
    """
    Retrieve the pixel resolution ``(x_res, y_res)`` of ``input_img``.

    :param input_img: Path to the input image file.
    :param abs_vals: If ``True``, return positive absolute resolution values
                     (Default = ``False``).
    :return: Tuple ``(x_res, y_res)``.

    .. code:: python

        import rsgispy.imageutils.metadata
        x_res, y_res = rsgispy.imageutils.metadata.get_img_res("img.tif")
    """
    _ensure_img_exists(input_img)
    with rasterio.open(input_img, "r") as src:
        x_res = float(src.transform.a)
        y_res = float(src.transform.e)
    if abs_vals:
        return abs(x_res), abs(y_res)
    return x_res, y_res


def get_img_size(input_img: str) -> Tuple[int, int]:
    """
    Retrieve the image dimensions in pixels as ``(x_size, y_size)``.

    :param input_img: Path to the input image file.
    :return: Tuple ``(width, height)`` in pixels.

    .. code:: python

        import rsgispy.imageutils.metadata
        x_size, y_size = rsgispy.imageutils.metadata.get_img_size("img.tif")
    """
    _ensure_img_exists(input_img)
    with rasterio.open(input_img, "r") as src:
        return int(src.width), int(src.height)


def get_img_band_count(input_img: str) -> int:
    """
    Retrieve the number of bands in ``input_img``.

    :param input_img: Path to the input image file.
    :return: Number of raster bands.

    .. code:: python

        import rsgispy.imageutils.metadata
        n_bands = rsgispy.imageutils.metadata.get_img_band_count("img.tif")
    """
    _ensure_img_exists(input_img)
    with rasterio.open(input_img, "r") as src:
        return int(src.count)


def get_img_bbox(input_img: str) -> Tuple[float, float, float, float]:
    """
    Retrieve the bounding box of ``input_img`` in its spatial coordinate system.

    :param input_img: Path to the input image file.
    :return: Bounding box ``(MinX, MaxX, MinY, MaxY)``.

    .. code:: python

        import rsgispy.imageutils.metadata
        bbox = rsgispy.imageutils.metadata.get_img_bbox("img.tif")
    """
    _ensure_img_exists(input_img)
    with rasterio.open(input_img, "r") as src:
        return bbox_from_dataset(src)


def get_img_bbox_in_proj(
    input_img: str, out_epsg: int
) -> Tuple[float, float, float, float]:
    """
    Retrieve the bounding box of ``input_img`` reprojected into ``out_epsg``.

    :param input_img: Path to the input image file.
    :param out_epsg: Target projection EPSG integer code.
    :return: Reprojected bounding box ``(MinX, MaxX, MinY, MaxY)``.

    .. code:: python

        import rsgispy.imageutils.metadata
        bbox = rsgispy.imageutils.metadata.get_img_bbox_in_proj("img.tif", out_epsg=27700)
    """
    in_wkt = get_wkt_proj_from_img(input_img)
    if not in_wkt:
        raise rsgispy.RSGISPyException(
            f"Image '{input_img}' does not have a CRS/projection defined."
        )
    img_bbox = get_img_bbox(input_img)
    return reproj_bbox(
        img_bbox,
        pyproj.CRS.from_wkt(in_wkt),
        pyproj.CRS.from_epsg(int(out_epsg)),
    )


def get_img_subset_pxl_bbox(
    input_img: str,
    sub_bbox: Union[Tuple[float, float, float, float], List[float]],
) -> Tuple[int, int, int, int]:
    """
    Compute the pixel bounding box ``(min_x_pxl, max_x_pxl, min_y_pxl, max_y_pxl)``
    corresponding to ``sub_bbox`` clipped to the extent of ``input_img``.

    :param input_img: Path to the input image file.
    :param sub_bbox: Spatial bounding box ``(MinX, MaxX, MinY, MaxY)`` in the
                     image CRS.
    :return: Pixel bounding box ``(min_x_pxl, max_x_pxl, min_y_pxl, max_y_pxl)``.
    """
    img_bbox = get_img_bbox(input_img)
    x_res, y_res = get_img_res(input_img, abs_vals=True)
    sub_tuple = (
        float(sub_bbox[0]),
        float(sub_bbox[1]),
        float(sub_bbox[2]),
        float(sub_bbox[3]),
    )

    if not do_bboxes_intersect(img_bbox, sub_tuple):
        raise rsgispy.RSGISPyException(
            "The subset BBOX does not intersect with the image bbox - "
            "is the projection of the sub_bbox correct?"
        )

    sub_inter = bbox_intersection(img_bbox, sub_tuple)

    min_x_pxl = math.floor(((sub_inter[0] - img_bbox[0]) / x_res) + 0.5)
    max_x_pxl = math.floor(((sub_inter[1] - img_bbox[0]) / x_res) + 0.5)
    min_y_pxl = math.floor(((img_bbox[3] - sub_inter[3]) / y_res) + 0.5)
    max_y_pxl = math.floor(((img_bbox[3] - sub_inter[2]) / y_res) + 0.5)

    return (min_x_pxl, max_x_pxl, min_y_pxl, max_y_pxl)


def get_img_pxl_spatial_coords(
    input_img: str,
    sub_pxl_bbox: Union[Tuple[float, float, float, float], List[float]],
) -> Tuple[float, float, float, float]:
    """
    Convert a pixel-coordinate bounding box ``(min_x_pxl, max_x_pxl, min_y_pxl, max_y_pxl)``
    into spatial coordinates ``(MinX, MaxX, MinY, MaxY)`` in ``input_img``'s CRS.

    :param input_img: Path to the input image file.
    :param sub_pxl_bbox: Pixel bounding box ``(min_x, max_x, min_y, max_y)``.
    :return: Spatial bounding box ``(MinX, MaxX, MinY, MaxY)``.
    """
    img_bbox = get_img_bbox(input_img)
    x_res, y_res = get_img_res(input_img, abs_vals=True)

    min_x = (sub_pxl_bbox[0] * x_res) + img_bbox[0]
    max_x = (sub_pxl_bbox[1] * x_res) + img_bbox[0]
    min_y = img_bbox[3] - (sub_pxl_bbox[3] * y_res)
    max_y = img_bbox[3] - (sub_pxl_bbox[2] * y_res)

    return (float(min_x), float(max_x), float(min_y), float(max_y))


def get_img_pxl_coords(
    input_img: str,
    x_coords: Union[np.ndarray, Sequence[float]],
    y_coords: Union[np.ndarray, Sequence[float]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert spatial coordinates ``(x_coords, y_coords)`` within ``input_img``'s
    extent into integer pixel coordinates ``(x_pxl_coords, y_pxl_coords)``.

    :param input_img: Path to the input image file.
    :param x_coords: Array or sequence of X spatial coordinates.
    :param y_coords: Array or sequence of Y spatial coordinates.
    :return: Tuple ``(x_pxl_coords, y_pxl_coords)`` of integer NumPy arrays.
    """
    _ensure_img_exists(input_img)
    x_arr = np.asarray(x_coords, dtype=np.float64)
    y_arr = np.asarray(y_coords, dtype=np.float64)

    with rasterio.open(input_img, "r") as src:
        tl_x = float(src.transform.c)
        tl_y = float(src.transform.f)
        x_res = abs(float(src.transform.a))
        y_res = abs(float(src.transform.e))
        br_x = tl_x + (x_res * src.width)
        br_y = tl_y - (y_res * src.height)

    if np.any((x_arr < tl_x) | (x_arr > br_x)):
        raise rsgispy.RSGISPyException(
            "Coordinates outside the image extent were passed (x-axis)"
        )
    if np.any((y_arr < br_y) | (y_arr > tl_y)):
        raise rsgispy.RSGISPyException(
            "Coordinates outside the image extent were passed (y-axis)"
        )

    x_pxl_coords = np.floor(((x_arr - tl_x) / x_res) + 0.5).astype(int)
    y_pxl_coords = np.floor(((tl_y - y_arr) / y_res) + 0.5).astype(int)
    return x_pxl_coords, y_pxl_coords


def get_img_files(input_img: str) -> List[str]:
    """
    Return the list of files associated with ``input_img`` (e.g., data and
    sidecar header/overview files).

    :param input_img: Path to the input image file.
    :return: List of associated file paths.
    """
    _ensure_img_exists(input_img)
    with rasterio.open(input_img, "r") as src:
        return list(src.files)


def has_gcps(input_img: str) -> bool:
    """
    Test whether ``input_img`` contains ground control points (GCPs).

    :param input_img: Path to the input image file.
    :return: ``True`` if the image has one or more GCPs, ``False`` otherwise.
    """
    _ensure_img_exists(input_img)
    with rasterio.open(input_img, "r") as src:
        gcps, _ = src.gcps
        return len(gcps) > 0


def copy_gcps(input_img: str, output_img: str) -> None:
    """
    Copy ground control points (GCPs) and their CRS from ``input_img`` to
    ``output_img``.

    :param input_img: Path to the source image containing GCPs.
    :param output_img: Path to the target image to update with GCPs.
    """
    _ensure_img_exists(input_img)
    _ensure_img_exists(output_img)
    with rasterio.open(input_img, "r") as src:
        gcps, gcp_crs = src.gcps

    if gcps:
        with rasterio.open(output_img, "r+") as dst:
            dst.gcps = (gcps, gcp_crs)


def get_gdal_format_name(input_img: str) -> str:
    """
    Return the uppercase GDAL/rasterio driver short name for ``input_img``
    (e.g., ``'GTIFF'`` or ``'KEA'``).

    :param input_img: Path to the input image file.
    :return: Uppercase format/driver name.

    .. code:: python

        import rsgispy.imageutils.metadata
        gdal_format = rsgispy.imageutils.metadata.get_gdal_format_name("img.tif")
    """
    _ensure_img_exists(input_img)
    with rasterio.open(input_img, "r") as src:
        return str(src.driver).upper()


def assign_proj(
    input_img: str,
    wkt_str: Optional[str] = None,
    wkt_file: Optional[str] = None,
    epsg_code: Optional[int] = None,
) -> None:
    """
    Assign a projection to ``input_img`` in place using a WKT string, a WKT
    file, or an EPSG integer code.

    :param input_img: Path to the input image file.
    :param wkt_str: WKT projection string. If ``None``, ``wkt_file`` or
                    ``epsg_code`` is used.
    :param wkt_file: Path to a text file containing the WKT string (ignored
                     when ``wkt_str`` is provided).
    :param epsg_code: EPSG integer code for the target projection (e.g., ``4326``
                      or ``32630``). Cannot be combined with ``wkt_str`` or
                      ``wkt_file``.

    .. code:: python

        import rsgispy.imageutils.metadata
        rsgispy.imageutils.metadata.assign_proj("img.tif", epsg_code=32630)
    """
    # pylint: disable=c-extension-no-member
    _ensure_img_exists(input_img)
    if epsg_code is not None:
        if wkt_str is not None or wkt_file is not None:
            raise rsgispy.RSGISPyException(
                "Cannot specify 'epsg_code' together with 'wkt_str' or 'wkt_file'."
            )
        crs_obj = _parse_epsg_crs(epsg_code)
    else:
        if wkt_str is None:
            if wkt_file is None:
                raise rsgispy.RSGISPyException(
                    "Must specify 'epsg_code', 'wkt_str', or 'wkt_file'."
                )
            if not os.path.exists(wkt_file):
                raise rsgispy.RSGISPyException(
                    f"WKT file does not exist: '{wkt_file}'."
                )
            with open(wkt_file, "r", encoding="utf-8") as handle:
                wkt_val = handle.read().strip()
        else:
            wkt_val = str(wkt_str).strip()

        crs_obj = (
            rasterio.crs.CRS.from_wkt(wkt_val)
            if wkt_val
            else rasterio.crs.CRS()
        )

    with rasterio.open(input_img, "r+") as dst:
        dst.crs = crs_obj


def assign_spatial_info(
    input_img: str,
    tl_x: Optional[float] = None,
    tl_y: Optional[float] = None,
    res_x: Optional[float] = None,
    res_y: Optional[float] = None,
    rot_x: Optional[float] = None,
    rot_y: Optional[float] = None,
) -> None:
    """
    Assign or update geotransform spatial information on ``input_img`` in place.
    Any parameter passed as ``None`` retains the existing geotransform value.

    :param input_img: Path to the input image file.
    :param tl_x: Top-left X coordinate.
    :param tl_y: Top-left Y coordinate.
    :param res_x: Pixel width (X resolution).
    :param res_y: Pixel height (Y resolution, typically negative for north-up).
    :param rot_x: X rotation parameter.
    :param rot_y: Y rotation parameter.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    _ensure_img_exists(input_img)
    with rasterio.open(input_img, "r+") as dst:
        g_c, g_a, g_b, g_f, g_d, g_e = dst.transform.to_gdal()
        if tl_x is not None:
            g_c = float(tl_x)
        if tl_y is not None:
            g_f = float(tl_y)
        if res_x is not None:
            g_a = float(res_x)
        if res_y is not None:
            g_e = float(res_y)
        if rot_x is not None:
            g_b = float(rot_x)
        if rot_y is not None:
            g_d = float(rot_y)
        dst.transform = Affine.from_gdal(g_c, g_a, g_b, g_f, g_d, g_e)


def copy_proj_from_img(input_img: str, ref_img: str) -> None:
    """
    Copy the CRS projection from ``ref_img`` onto ``input_img``.

    :param input_img: Path to the target image file to update.
    :param ref_img: Path to the reference image file.
    """
    # pylint: disable=c-extension-no-member
    _ensure_img_exists(ref_img)
    _ensure_img_exists(input_img)
    with rasterio.open(ref_img, "r") as ref_ds:
        ref_crs = ref_ds.crs

    with rasterio.open(input_img, "r+") as dst:
        dst.crs = ref_crs if ref_crs is not None else rasterio.crs.CRS()


def copy_spatial_and_proj_from_img(input_img: str, ref_img: str) -> None:
    """
    Copy both the affine geotransform and CRS projection from ``ref_img`` onto
    ``input_img``.

    :param input_img: Path to the target image file to update.
    :param ref_img: Path to the reference image file.
    """
    # pylint: disable=c-extension-no-member
    _ensure_img_exists(ref_img)
    _ensure_img_exists(input_img)
    with rasterio.open(ref_img, "r") as ref_ds:
        ref_crs = ref_ds.crs
        ref_transform = ref_ds.transform

    with rasterio.open(input_img, "r+") as dst:
        dst.transform = ref_transform
        dst.crs = ref_crs if ref_crs is not None else rasterio.crs.CRS()


def get_wkt_proj_from_img(input_img: str) -> str:
    """
    Return the WKT projection string of ``input_img`` (empty string if unset).

    :param input_img: Path to the input image file.
    :return: WKT projection string.
    """
    _ensure_img_exists(input_img)
    with rasterio.open(input_img, "r") as src:
        if src.crs is None:
            return ""
        return src.crs.to_wkt()


def get_epsg_proj_from_img(input_img: str) -> Optional[int]:
    """
    Return the EPSG code of ``input_img``'s CRS, or ``None`` if unavailable.

    :param input_img: Path to the input image file.
    :return: EPSG integer code or ``None``.
    """
    _ensure_img_exists(input_img)
    try:
        with rasterio.open(input_img, "r") as src:
            if src.crs is None:
                return None
            epsg_code = src.crs.to_epsg()
            if epsg_code is None:
                epsg_code = pyproj.CRS.from_user_input(src.crs).to_epsg()
            return int(epsg_code) if epsg_code is not None else None
    except Exception:  # pylint: disable=broad-exception-caught
        return None


def get_utm_zone(input_img: str) -> Optional[str]:
    """
    Return the UTM zone string (e.g., ``'30N'``, ``'05N'``, ``'55S'``) of
    ``input_img`` if it is in a UTM coordinate reference system, else ``None``.

    :param input_img: Path to the input image file.
    :return: Zero-padded UTM zone string or ``None``.
    """
    _ensure_img_exists(input_img)
    with rasterio.open(input_img, "r") as src:
        if src.crs is None:
            return None
        crs_obj = pyproj.CRS.from_user_input(src.crs)

    if not crs_obj.is_projected:
        return None
    utm_zone = crs_obj.utm_zone
    if not utm_zone:
        return None
    zone_num = int(utm_zone[:-1])
    hemi = utm_zone[-1].upper()
    return f"{zone_num:02d}{hemi}"
