"""
The imageutils.checks module provides raster comparison, projection, resolution,
and spatial intersection check functions using ``rasterio``.
"""

from typing import List, Optional, Sequence, Tuple, Union

import rasterio

import rsgispy
from rsgispy.imageprocess import bbox_from_dataset, ensure_file_exists
from rsgispy.imageutils.metadata import (
    get_epsg_proj_from_img,
    get_img_band_count,
    get_img_bbox,
    get_img_res,
    get_wkt_proj_from_img,
)
from rsgispy.tools.geometrytools import (
    bbox_equal,
    do_bboxes_intersect,
    reproj_bbox,
)

__all__ = [
    "do_img_res_match",
    "check_img_lst",
    "check_img_file_comparison",
    "do_images_overlap",
    "do_gdal_layers_have_same_proj",
    "test_img_lst_intersects",
]


def do_img_res_match(in_a_img: str, in_b_img: str) -> bool:
    """
    Test whether two images have the same pixel resolution.

    :param in_a_img: First input image file path.
    :param in_b_img: Second input image file path.
    :return: ``True`` if both X and Y resolutions match, ``False`` otherwise.
    """
    img1_x_res, img1_y_res = get_img_res(in_a_img)
    img2_x_res, img2_y_res = get_img_res(in_b_img)
    return (img1_x_res == img2_x_res) and (img1_y_res == img2_y_res)


def check_img_lst(
    input_imgs: Sequence[str],
    exp_x_res: float,
    exp_y_res: float,
    bbox: Optional[Union[Tuple[float, float, float, float], Sequence[float]]] = None,
    print_errors: bool = True,
    abs_res: bool = True,
) -> List[str]:
    """
    Filter a sequence of images to those matching the expected resolution and
    optionally intersecting ``bbox``.

    :param input_imgs: Sequence of input image file paths.
    :param exp_x_res: Expected image resolution along the X axis.
    :param exp_y_res: Expected image resolution along the Y axis.
    :param bbox: Optional bounding box ``(MinX, MaxX, MinY, MaxY)`` to test
                 for intersection.
    :param print_errors: If ``True``, print details for images failing checks
                         (Default = ``True``).
    :param abs_res: If ``True``, compare absolute pixel resolution values
                    (Default = ``True``).
    :return: List of image paths that pass all checks.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    if abs_res:
        exp_x_res = abs(exp_x_res)
        exp_y_res = abs(exp_y_res)

    bbox_tuple = (
        (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        if bbox is not None
        else None
    )

    out_imgs: List[str] = []
    for img in input_imgs:
        img_res = get_img_res(img, abs_vals=abs_res)
        img_bbox = get_img_bbox(img) if bbox_tuple is not None else None
        if (img_res[0] != exp_x_res) or (img_res[1] != exp_y_res):
            if print_errors:
                print(f"{img} has resolution: {img_res}")
        elif bbox_tuple is not None and img_bbox is not None and not do_bboxes_intersect(
            bbox_tuple, img_bbox
        ):
            if print_errors:
                print(f"{img} has BBOX: {img_bbox}")
        else:
            out_imgs.append(img)
    return out_imgs


def check_img_file_comparison(
    in_base_img: str,
    in_comp_img: str,
    test_n_bands: bool = False,
    test_eql_bbox: bool = False,
    print_errors: bool = True,
) -> bool:
    """
    Test whether two images are comparable in resolution, bounding box
    intersection, and optionally bounding box equality and band count.

    :param in_base_img: Base input image path.
    :param in_comp_img: Comparison input image path.
    :param test_n_bands: If ``True``, require equal band counts (Default = ``False``).
    :param test_eql_bbox: If ``True``, require identical bounding boxes (Default = ``False``).
    :param print_errors: If ``True``, print mismatch details (Default = ``True``).
    :return: ``True`` if images pass all enabled checks, ``False`` otherwise.
    """
    imgs_match = True

    if not do_img_res_match(in_base_img, in_comp_img):
        if print_errors:
            print(f"Base Image Res: {get_img_res(in_base_img)}")
            print(f"Comp Image Res: {get_img_res(in_comp_img)}")
        imgs_match = False

    base_bbox = get_img_bbox(in_base_img)
    comp_bbox = get_img_bbox(in_comp_img)
    if not do_bboxes_intersect(base_bbox, comp_bbox):
        if print_errors:
            print(f"Base Image BBOX: {base_bbox}")
            print(f"Comp Image BBOX: {comp_bbox}")
        imgs_match = False

    if test_eql_bbox and not bbox_equal(base_bbox, comp_bbox):
        if print_errors:
            print(f"Base Image BBOX: {base_bbox}")
            print(f"Comp Image BBOX: {comp_bbox}")
        imgs_match = False

    if test_n_bands:
        base_nbands = get_img_band_count(in_base_img)
        comp_nbands = get_img_band_count(in_comp_img)
        if base_nbands != comp_nbands:
            if print_errors:
                print(f"Base Image n-bands: {base_nbands}")
                print(f"Comp Image n-bands: {comp_nbands}")
            imgs_match = False

    return imgs_match


def do_gdal_layers_have_same_proj(in_a_img: str, in_b_img: str) -> bool:
    """
    Test whether two raster layers share the same projection / coordinate system.

    :param in_a_img: First input image file path.
    :param in_b_img: Second input image file path.
    :return: ``True`` if projections match, ``False`` otherwise.
    """
    epsg_a = get_epsg_proj_from_img(in_a_img)
    epsg_b = get_epsg_proj_from_img(in_b_img)
    if epsg_a is not None and epsg_b is not None:
        return epsg_a == epsg_b
    if epsg_a != epsg_b:
        return False
    return get_wkt_proj_from_img(in_a_img) == get_wkt_proj_from_img(in_b_img)


def do_images_overlap(
    in_a_img: str, in_b_img: str, over_thres: float = 0.0
) -> bool:
    """
    Test whether two raster images spatially overlap by more than ``over_thres``
    in the coordinate system of ``in_a_img``.

    :param in_a_img: First input image file path.
    :param in_b_img: Second input image file path.
    :param over_thres: Minimum required overlap width/height in ``in_a_img`` CRS
                       units (Default = ``0.0``).
    :return: ``True`` if the images overlap by more than ``over_thres``, else ``False``.
    """
    ensure_file_exists(in_a_img)
    ensure_file_exists(in_b_img)

    proj_same = do_gdal_layers_have_same_proj(in_a_img, in_b_img)
    with rasterio.open(in_a_img, "r") as src_a, rasterio.open(
        in_b_img, "r"
    ) as src_b:
        bbox_a = bbox_from_dataset(src_a)
        bbox_b = bbox_from_dataset(src_b)
        crs_a = src_a.crs
        crs_b = src_b.crs

    if not proj_same:
        if crs_a is None or crs_b is None:
            raise rsgispy.RSGISPyException(
                "Could not retrieve CRS for both images to test overlap."
            )
        bbox_b = reproj_bbox(bbox_b, crs_b, crs_a)

    x_overlap = min(bbox_a[1], bbox_b[1]) - max(bbox_a[0], bbox_b[0])
    y_overlap = min(bbox_a[3], bbox_b[3]) - max(bbox_a[2], bbox_b[2])
    return (x_overlap > over_thres) and (y_overlap > over_thres)


def test_img_lst_intersects(
    input_imgs: Sequence[str], stop_err: bool = False
) -> None:
    """
    Test a sequence of images against the first image for matching projection,
    intersecting bounding boxes, and matching pixel resolutions.

    :param input_imgs: Sequence of image file paths.
    :param stop_err: If ``True``, raise :class:`rsgispy.RSGISPyException` on the
                     first mismatch; otherwise print diagnostics (Default = ``False``).
    """
    if not input_imgs:
        return

    ref_img = input_imgs[0]
    print(ref_img)
    ref_bbox = get_img_bbox(ref_img)
    ref_proj = get_epsg_proj_from_img(ref_img)
    ref_res = get_img_res(ref_img)
    print("\tReference Image")

    for img in input_imgs[1:]:
        print(img)
        img_bbox = get_img_bbox(img)
        img_proj = get_epsg_proj_from_img(img)
        img_res = get_img_res(img)

        if ref_proj != img_proj:
            msg = "Projection does not match the reference (i.e., first image)"
            print(f"\t{msg}")
            print(f"\tRef (first) Image: {ref_img}")
            print(f"\tRef (first) EPSG: {ref_proj}")
            print(f"\tImage: {img}")
            print(f"\tImage EPSG: {img_proj}")
            if stop_err:
                raise rsgispy.RSGISPyException(msg)
        elif not do_bboxes_intersect(ref_bbox, img_bbox):
            msg = "BBOX does not intersect the reference (i.e., first image)"
            print(f"\t{msg}")
            print(f"\tRef (first) Image: {ref_img}")
            print("\tRef (first) BBOX:", ref_bbox)
            print(f"\tImage: {img}")
            print("\tImage BBOX: ", img_bbox)
            if stop_err:
                raise rsgispy.RSGISPyException(msg)
        elif (img_res[0] != ref_res[0]) or (img_res[1] != ref_res[1]):
            msg = "Image resolution does not match the reference (i.e., first image)"
            print(f"\t{msg}")
            print(f"\tRef (first) Image: {ref_img}")
            print("\tRef (first) Res: ", ref_res)
            print(f"\tImage: {img}")
            print("\tImage Res: ", img_res)
            if stop_err:
                raise rsgispy.RSGISPyException(msg)
        else:
            print("\tOK")
