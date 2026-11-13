"""
The imageutils.subsetting module provides raster spatial subsetting functions
by bounding box, reference raster image, or vector layer using ``rasterio`` and
``geopandas``.
"""

import math
from typing import Any, Mapping, Optional, Sequence, Union

import geopandas
import rasterio

import rsgispy
from rsgispy.imageprocess import bbox_from_dataset, ensure_file_exists
from rsgispy.imageutils._helpers import copy_raster_window
from rsgispy.tools.geometry import (
    bbox_intersection,
    do_bboxes_intersect,
    reproj_bbox,
)

__all__ = [
    "subset_bbox",
    "subset_to_img",
    "subset",
]


def subset_bbox(
    input_img: str,
    output_img: str,
    gdalformat: str,
    datatype: int,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
    options: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
) -> None:
    """
    Subset ``input_img`` to the bounding box ``(min_x, max_x, min_y, max_y)``,
    snapping cleanly onto the input pixel grid.

    :param input_img: Path to the input image file.
    :param output_img: Path to the output subset image file.
    :param gdalformat: Output GDAL format name (e.g., ``'KEA'``, ``'GTIFF'``).
    :param datatype: ``rsgispy.TYPE_*`` constant for the output image.
    :param min_x: Minimum X coordinate in ``input_img`` CRS.
    :param max_x: Maximum X coordinate in ``input_img`` CRS.
    :param min_y: Minimum Y coordinate in ``input_img`` CRS.
    :param max_y: Maximum Y coordinate in ``input_img`` CRS.
    :param options: Optional GDAL creation options mapping.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    ensure_file_exists(input_img)
    if float(min_x) >= float(max_x) or float(min_y) >= float(max_y):
        raise rsgispy.RSGISPyException(
            f"Invalid bounding box ({min_x}, {max_x}, {min_y}, {max_y}): "
            "min must be strictly less than max."
        )

    with rasterio.open(input_img, "r") as src:
        img_bbox = bbox_from_dataset(src)
        sub_bbox = (float(min_x), float(max_x), float(min_y), float(max_y))

        if not do_bboxes_intersect(img_bbox, sub_bbox):
            raise rsgispy.RSGISPyException(
                "The subset BBOX does not intersect with the image bbox - "
                "is the projection of the sub_bbox correct?"
            )

        sub_inter = bbox_intersection(img_bbox, sub_bbox)
        x_res = abs(float(src.transform.a))
        y_res = abs(float(src.transform.e))

        min_x_pxl = max(
            0, math.floor(((sub_inter[0] - img_bbox[0]) / x_res) + 0.5)
        )
        max_x_pxl = min(
            src.width, math.floor(((sub_inter[1] - img_bbox[0]) / x_res) + 0.5)
        )
        min_y_pxl = max(
            0, math.floor(((img_bbox[3] - sub_inter[3]) / y_res) + 0.5)
        )
        max_y_pxl = min(
            src.height, math.floor(((img_bbox[3] - sub_inter[2]) / y_res) + 0.5)
        )

    out_w = max_x_pxl - min_x_pxl
    out_h = max_y_pxl - min_y_pxl
    if out_w <= 0 or out_h <= 0:
        raise rsgispy.RSGISPyException(
            "Subset region contains zero pixels within the input image grid."
        )

    copy_raster_window(
        input_img=input_img,
        output_img=output_img,
        col_off=min_x_pxl,
        row_off=min_y_pxl,
        width=out_w,
        height=out_h,
        gdalformat=gdalformat,
        datatype=datatype,
        options=options,
    )


def subset_to_img(
    input_img: str,
    in_roi_img: str,
    output_img: str,
    gdalformat: str,
    datatype: int,
) -> None:
    """
    Subset ``input_img`` to the spatial bounding box of ``in_roi_img``.

    :param input_img: Path to the input image file to subset.
    :param in_roi_img: Path to the ROI image defining the target extent.
    :param output_img: Path to the output subset image file.
    :param gdalformat: Output GDAL format name (e.g., ``'KEA'``, ``'GTIFF'``).
    :param datatype: ``rsgispy.TYPE_*`` constant for the output image.
    """
    ensure_file_exists(input_img)
    ensure_file_exists(in_roi_img, label="ROI image")

    with rasterio.open(input_img, "r") as in_ds:
        in_crs = in_ds.crs
    with rasterio.open(in_roi_img, "r") as roi_ds:
        roi_crs = roi_ds.crs
        roi_bbox = bbox_from_dataset(roi_ds)

    if in_crs is not None and roi_crs is not None and in_crs != roi_crs:
        roi_bbox = reproj_bbox(roi_bbox, roi_crs, in_crs)

    subset_bbox(
        input_img=input_img,
        output_img=output_img,
        gdalformat=gdalformat,
        datatype=datatype,
        min_x=roi_bbox[0],
        max_x=roi_bbox[1],
        min_y=roi_bbox[2],
        max_y=roi_bbox[3],
    )


def subset(
    input_img: str,
    vec_file: str,
    vec_lyr: Optional[str],
    output_img: str,
    gdalformat: str,
    datatype: int,
) -> None:
    """
    Subset ``input_img`` to the bounding box of a vector file/layer.

    :param input_img: Path to the input image file to subset.
    :param vec_file: Path to the vector file defining the subset region.
    :param vec_lyr: Optional layer name within ``vec_file`` (or ``None`` for default).
    :param output_img: Path to the output subset image file.
    :param gdalformat: Output GDAL format name (e.g., ``'KEA'``, ``'GTIFF'``).
    :param datatype: ``rsgispy.TYPE_*`` constant for the output image.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    ensure_file_exists(input_img)
    ensure_file_exists(vec_file, label="Vector")

    if vec_lyr is not None:
        gdf = geopandas.read_file(vec_file, layer=vec_lyr)
    else:
        gdf = geopandas.read_file(vec_file)

    if gdf.empty:
        raise rsgispy.RSGISPyException(
            f"Vector layer '{vec_lyr}' in '{vec_file}' contains no features."
        )

    with rasterio.open(input_img, "r") as in_ds:
        in_crs = in_ds.crs

    if in_crs is not None and gdf.crs is not None and gdf.crs != in_crs:
        gdf = gdf.to_crs(in_crs)

    min_x, min_y, max_x, max_y = (float(v) for v in gdf.total_bounds)
    subset_bbox(
        input_img=input_img,
        output_img=output_img,
        gdalformat=gdalformat,
        datatype=datatype,
        min_x=min_x,
        max_x=max_x,
        min_y=min_y,
        max_y=max_y,
    )
