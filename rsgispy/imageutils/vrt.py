"""
GDAL VRT generation utilities (stacking, mosaicking, band subsetting)
for :mod:`rsgispy.imageutils`.
"""

from contextlib import ExitStack
import math
import os
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union
import xml.etree.ElementTree as ET

import rasterio
import rasterio.crs
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.vrt import _gdal_typename

import rsgispy
from rsgispy.imageutils._helpers import (
    normalise_creation_options as _normalise_creation_options,
    parse_epsg_crs as _parse_epsg_crs,
)


def _resampling_to_vrt_name(
    interp_method: Union[int, str, Resampling],
) -> str:
    """
    Convert an interpolation method to a GDAL VRT ``resampling`` attribute string.
    """
    if isinstance(interp_method, str):
        return interp_method.strip().lower().replace("_", "")
    resamp = rsgispy.get_rasterio_interp_type(interp_method)
    if resamp == Resampling.nearest:
        return "nearest"
    return resamp.name.replace("_", "")


def _compute_vrt_grid(
    datasets: Sequence[rasterio.DatasetReader],
    resolution: str = "average",
    vrt_extent: Optional[Tuple[float, float, float, float]] = None,
    vrt_out_res_x: Optional[float] = None,
    vrt_out_res_y: Optional[float] = None,
    align_out_pxls: bool = False,
    allow_proj_diff: bool = False,
) -> Tuple[int, int, Affine, Optional[rasterio.crs.CRS]]:
    """
    Calculate the VRT output dimensions, affine transform, and CRS from input datasets.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals,too-many-branches
    if not datasets:
        raise rsgispy.RSGISPyException(
            "At least one input image must be provided to create a VRT."
        )

    if (vrt_out_res_x is None) != (vrt_out_res_y is None):
        raise rsgispy.RSGISPyException(
            "The X and Y resolution cannot be defined independently."
        )

    ref_crs = datasets[0].crs
    if not allow_proj_diff:
        for ds in datasets[1:]:
            if ds.crs != ref_crs:
                raise rsgispy.RSGISPyException(
                    f"Input dataset '{ds.name}' projection ({ds.crs}) does not match "
                    f"reference projection ({ref_crs}). Set allow_proj_diff=True to override."
                )

    x_res_list = [abs(ds.transform.a) for ds in datasets]
    y_res_list = [abs(ds.transform.e) for ds in datasets]

    res_mode = resolution.strip().lower()
    if vrt_out_res_x is not None and vrt_out_res_y is not None:
        res_x = abs(float(vrt_out_res_x))
        res_y = abs(float(vrt_out_res_y))
    elif res_mode == "highest":
        res_x = min(x_res_list)
        res_y = min(y_res_list)
    elif res_mode == "lowest":
        res_x = max(x_res_list)
        res_y = max(y_res_list)
    elif res_mode in ("average", "user"):
        if res_mode == "user":
            raise rsgispy.RSGISPyException(
                "resolution='user' requires vrt_out_res_x and vrt_out_res_y to be specified."
            )
        res_x = sum(x_res_list) / len(x_res_list)
        res_y = sum(y_res_list) / len(y_res_list)
    else:
        raise rsgispy.RSGISPyException(
            f"Unsupported resolution mode '{resolution}'. Choose from "
            "'average', 'highest', 'lowest', or 'user'."
        )

    if vrt_extent is not None:
        if len(vrt_extent) < 4:
            raise rsgispy.RSGISPyException(
                "vrt_extent must contain 4 values: (min_x, min_y, max_x, max_y)."
            )
        min_x, min_y, max_x, max_y = (float(v) for v in vrt_extent[:4])
    else:
        min_x = min(ds.bounds.left for ds in datasets)
        min_y = min(ds.bounds.bottom for ds in datasets)
        max_x = max(ds.bounds.right for ds in datasets)
        max_y = max(ds.bounds.top for ds in datasets)

    if align_out_pxls:
        min_x = math.floor(min_x / res_x) * res_x
        min_y = math.floor(min_y / res_y) * res_y
        max_x = math.ceil(max_x / res_x) * res_x
        max_y = math.ceil(max_y / res_y) * res_y

    width = max(1, int(round((max_x - min_x) / res_x)))
    height = max(1, int(round((max_y - min_y) / res_y)))
    transform = Affine(res_x, 0.0, min_x, 0.0, -res_y, max_y)
    return width, height, transform, ref_crs


def _format_vrt_source_path(
    input_img: str, out_vrt_file: str, write_abs_path: bool
) -> Tuple[str, str]:
    """
    Return ``(source_filename_text, relativeToVRT_attr)`` for a VRT source.
    """
    abs_in = os.path.abspath(input_img)
    if write_abs_path:
        return abs_in, "0"
    vrt_dir = os.path.dirname(os.path.abspath(out_vrt_file))
    try:
        rel_path = os.path.relpath(abs_in, vrt_dir)
        return rel_path, "1"
    except ValueError:
        return abs_in, "0"


def _append_complex_source(
    band_elem: ET.Element,
    src_ds: rasterio.DatasetReader,
    src_band: int,
    src_path: str,
    rel_attr: str,
    vrt_transform: Affine,
    resampling_str: str = "nearest",
    src_nodata: Optional[float] = None,
    scale_offset: Optional[float] = None,
    scale_ratio: Optional[float] = None,
) -> None:
    """
    Append a ``<ComplexSource>`` element to a ``<VRTRasterBand>`` element.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    cs_elem = ET.SubElement(band_elem, "ComplexSource")
    if resampling_str and resampling_str != "nearest":
        cs_elem.attrib["resampling"] = resampling_str

    sf_elem = ET.SubElement(cs_elem, "SourceFilename")
    sf_elem.attrib["relativeToVRT"] = rel_attr
    sf_elem.text = src_path

    sb_elem = ET.SubElement(cs_elem, "SourceBand")
    sb_elem.text = str(src_band)

    dtype_str = _gdal_typename(src_ds.dtypes[src_band - 1])
    block_y, block_x = (
        src_ds.block_shapes[src_band - 1]
        if src_ds.block_shapes and len(src_ds.block_shapes) >= src_band
        else (src_ds.height, src_ds.width)
    )
    sp_elem = ET.SubElement(cs_elem, "SourceProperties")
    sp_elem.attrib["RasterXSize"] = str(src_ds.width)
    sp_elem.attrib["RasterYSize"] = str(src_ds.height)
    sp_elem.attrib["DataType"] = dtype_str
    sp_elem.attrib["BlockXSize"] = str(block_x)
    sp_elem.attrib["BlockYSize"] = str(block_y)

    src_rect = ET.SubElement(cs_elem, "SrcRect")
    src_rect.attrib["xOff"] = "0"
    src_rect.attrib["yOff"] = "0"
    src_rect.attrib["xSize"] = str(src_ds.width)
    src_rect.attrib["ySize"] = str(src_ds.height)

    dst_x_off = (src_ds.transform.c - vrt_transform.c) / vrt_transform.a
    dst_y_off = (src_ds.transform.f - vrt_transform.f) / vrt_transform.e
    dst_x_size = (src_ds.width * abs(src_ds.transform.a)) / abs(vrt_transform.a)
    dst_y_size = (src_ds.height * abs(src_ds.transform.e)) / abs(
        vrt_transform.e
    )

    dst_rect = ET.SubElement(cs_elem, "DstRect")
    dst_rect.attrib["xOff"] = f"{dst_x_off:g}"
    dst_rect.attrib["yOff"] = f"{dst_y_off:g}"
    dst_rect.attrib["xSize"] = f"{dst_x_size:g}"
    dst_rect.attrib["ySize"] = f"{dst_y_size:g}"

    if src_nodata is not None:
        nd_elem = ET.SubElement(cs_elem, "NODATA")
        nd_elem.text = f"{src_nodata:g}"

    if scale_offset is not None:
        so_elem = ET.SubElement(cs_elem, "ScaleOffset")
        so_elem.text = f"{scale_offset:g}"
    if scale_ratio is not None:
        sr_elem = ET.SubElement(cs_elem, "ScaleRatio")
        sr_elem.text = f"{scale_ratio:g}"


def _write_vrt_xml(vrt_root: ET.Element, out_vrt_file: str) -> None:
    """
    Indent and write a ``<VRTDataset>`` XML tree to ``out_vrt_file``.
    """
    ET.indent(vrt_root, space="  ")
    xml_bytes = ET.tostring(vrt_root, encoding="utf-8", xml_declaration=False)
    with open(out_vrt_file, "wb") as out_f:
        out_f.write(xml_bytes)
        out_f.write(b"\n")


def _apply_vrt_band_creation_opts(
    band_elem: ET.Element,
    creation_opts: Optional[Dict[str, Any]],
) -> None:
    """
    Apply VRT band creation options (such as ``BLOCKXSIZE`` and ``BLOCKYSIZE``)
    to a ``<VRTRasterBand>`` element.
    """
    if not creation_opts:
        return
    for key, val in creation_opts.items():
        key_up = str(key).upper()
        if key_up == "BLOCKXSIZE":
            band_elem.attrib["blockXSize"] = str(val)
        elif key_up == "BLOCKYSIZE":
            band_elem.attrib["blockYSize"] = str(val)


def create_stack_images_vrt(
    input_imgs: Sequence[str],
    out_vrt_file: str,
    vrt_extent: Optional[Tuple[float, float, float, float]] = None,
    vrt_out_res_x: Optional[float] = None,
    vrt_out_res_y: Optional[float] = None,
    resolution: str = "average",
    interp_method: Union[int, str, Resampling] = rsgispy.INTERP_NEAREST_NEIGHBOUR,
    align_out_pxls: bool = False,
    src_nodata: Optional[float] = None,
    vrt_nodata: Optional[float] = None,
    hide_nodata: bool = False,
    allow_proj_diff: bool = False,
    out_epsg: Optional[int] = None,
    write_abs_path: bool = False,
    band_names: Optional[Sequence[str]] = None,
    options: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
) -> None:
    """
    Create a GDAL VRT file by stacking bands from a sequence of input images into
    a multi-band VRT dataset using ``rasterio`` to inspect image metadata.

    :param input_imgs: Sequence of input image file paths.
    :param out_vrt_file: Output file path for the generated VRT.
    :param vrt_extent: Optional extent ``(minX, minY, maxX, maxY)`` for the VRT image.
    :param vrt_out_res_x: Optional X resolution of the output VRT.
    :param vrt_out_res_y: Optional Y resolution of the output VRT.
    :param resolution: Resolution strategy when ``vrt_out_res_x``/``y`` are not set:
                       ``'average'``, ``'highest'``, ``'lowest'``, or ``'user'``
                       (Default = ``'average'``).
    :param interp_method: Interpolation method ``rsgispy.INTERP_*`` or
                          ``rasterio.enums.Resampling`` (Default = INTERP_NEAREST_NEIGHBOUR).
    :param align_out_pxls: Whether to align output bounds to multiples of the output
                           pixel resolution (Default = False).
    :param src_nodata: Optional override for source image no-data values.
    :param vrt_nodata: Optional override for VRT band-level no-data values.
    :param hide_nodata: Whether to hide the no-data value on VRT bands (Default = False).
    :param allow_proj_diff: Whether to allow input images with differing projections
                            without raising an error (Default = False).
    :param out_epsg: Optional EPSG code to assign as the output VRT projection.
    :param write_abs_path: Whether to write absolute paths to source images in the VRT
                           rather than relative paths (Default = False).
    :param band_names: Optional sequence of band descriptions for the output stacked bands.
    :param options: Optional VRT creation options
                    (e.g., ``{'BLOCKXSIZE': 256, 'BLOCKYSIZE': 256}``).
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    if not input_imgs:
        raise rsgispy.RSGISPyException(
            "input_imgs must contain at least one image path."
        )
    for img_path in input_imgs:
        if not os.path.exists(img_path):
            raise rsgispy.RSGISPyException(
                f"Input image file does not exist: '{img_path}'."
            )

    resampling_str = _resampling_to_vrt_name(interp_method)
    creation_opts = _normalise_creation_options(options)

    with ExitStack() as stack:
        datasets = [
            stack.enter_context(rasterio.open(p, "r")) for p in input_imgs
        ]
        width, height, transform, ref_crs = _compute_vrt_grid(
            datasets=datasets,
            resolution=resolution,
            vrt_extent=vrt_extent,
            vrt_out_res_x=vrt_out_res_x,
            vrt_out_res_y=vrt_out_res_y,
            align_out_pxls=align_out_pxls,
            allow_proj_diff=allow_proj_diff,
        )

        out_crs = _parse_epsg_crs(out_epsg) if out_epsg is not None else ref_crs

        vrt_root = ET.Element(
            "VRTDataset",
            attrib={"rasterXSize": str(width), "rasterYSize": str(height)},
        )
        if out_crs is not None:
            srs_elem = ET.SubElement(
                vrt_root, "SRS", attrib={"dataAxisToSRSAxisMapping": "1,2"}
            )
            srs_elem.text = out_crs.to_wkt()

        gt_elem = ET.SubElement(vrt_root, "GeoTransform")
        gt_elem.text = ", ".join(f"{v:.16e}" for v in transform.to_gdal())

        out_band_idx = 1
        for img_path, ds in zip(input_imgs, datasets):
            src_path, rel_attr = _format_vrt_source_path(
                img_path, out_vrt_file, write_abs_path
            )
            nodata_list = (
                ds.nodatavals if ds.nodatavals else [ds.nodata] * ds.count
            )
            descriptions = (
                ds.descriptions if ds.descriptions else (None,) * ds.count
            )

            for b_idx in range(1, ds.count + 1):
                dtype_str = _gdal_typename(ds.dtypes[b_idx - 1])
                band_elem = ET.SubElement(
                    vrt_root,
                    "VRTRasterBand",
                    attrib={"dataType": dtype_str, "band": str(out_band_idx)},
                )
                _apply_vrt_band_creation_opts(band_elem, creation_opts)

                desc = (
                    band_names[out_band_idx - 1]
                    if band_names and len(band_names) >= out_band_idx
                    else descriptions[b_idx - 1]
                )
                if desc:
                    desc_elem = ET.SubElement(band_elem, "Description")
                    desc_elem.text = str(desc)

                eff_src_nd = (
                    src_nodata
                    if src_nodata is not None
                    else nodata_list[b_idx - 1]
                )
                eff_vrt_nd = (
                    vrt_nodata if vrt_nodata is not None else eff_src_nd
                )
                if eff_vrt_nd is not None:
                    nd_elem = ET.SubElement(band_elem, "NoDataValue")
                    nd_elem.text = f"{eff_vrt_nd:g}"
                if hide_nodata:
                    hide_elem = ET.SubElement(band_elem, "HideNoDataValue")
                    hide_elem.text = "1"

                _append_complex_source(
                    band_elem=band_elem,
                    src_ds=ds,
                    src_band=b_idx,
                    src_path=src_path,
                    rel_attr=rel_attr,
                    vrt_transform=transform,
                    resampling_str=resampling_str,
                    src_nodata=eff_src_nd,
                )
                out_band_idx += 1

    _write_vrt_xml(vrt_root, out_vrt_file)


def create_mosaic_images_vrt(
    input_imgs: Sequence[str],
    out_vrt_file: str,
    vrt_extent: Optional[Tuple[float, float, float, float]] = None,
    vrt_out_res_x: Optional[float] = None,
    vrt_out_res_y: Optional[float] = None,
    interp_method: Union[int, str, Resampling] = rsgispy.INTERP_NEAREST_NEIGHBOUR,
    align_out_pxls: bool = False,
    resolution: str = "average",
    src_nodata: Optional[float] = None,
    vrt_nodata: Optional[float] = None,
    hide_nodata: bool = False,
    add_alpha: bool = False,
    band_list: Optional[Sequence[int]] = None,
    allow_proj_diff: bool = False,
    out_epsg: Optional[int] = None,
    write_abs_path: bool = False,
    pixel_function: Optional[str] = None,
    pixel_function_args: Optional[Mapping[str, Any]] = None,
    options: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
) -> None:
    """
    Create a GDAL VRT file by mosaicking a sequence of input images using ``rasterio``
    to inspect image metadata.

    :param input_imgs: Sequence of input image file paths.
    :param out_vrt_file: Output file path for the generated VRT.
    :param vrt_extent: Optional extent ``(minX, minY, maxX, maxY)`` for the VRT image.
    :param vrt_out_res_x: Optional X resolution of the output VRT.
    :param vrt_out_res_y: Optional Y resolution of the output VRT.
    :param interp_method: Interpolation method ``rsgispy.INTERP_*`` or
                          ``rasterio.enums.Resampling`` (Default = INTERP_NEAREST_NEIGHBOUR).
    :param align_out_pxls: Whether to align output bounds to multiples of the output
                           pixel resolution (Default = False).
    :param resolution: Resolution strategy when ``vrt_out_res_x``/``y`` are not set:
                       ``'average'``, ``'highest'``, ``'lowest'``, or ``'user'``
                       (Default = ``'average'``).
    :param src_nodata: Optional override for source image no-data values.
    :param vrt_nodata: Optional override for VRT band-level no-data values.
    :param hide_nodata: Whether to hide the no-data value on VRT bands (Default = False).
    :param add_alpha: Whether to add an alpha mask band to the output VRT (Default = False).
    :param band_list: Optional 1-based band numbers from the input images to include
                      in the mosaic.
    :param allow_proj_diff: Whether to allow input images with differing projections
                            without raising an error (Default = False).
    :param out_epsg: Optional EPSG code to assign as the output VRT projection.
    :param write_abs_path: Whether to write absolute paths to source images in the VRT
                           rather than relative paths (Default = False).
    :param pixel_function: Optional GDAL built-in pixel function name (e.g., ``'max'``,
                           ``'min'``, ``'mean'``, ``'median'``, ``'mode'``, ``'sum'``)
                           for combining overlapping pixels.
    :param pixel_function_args: Optional dictionary of arguments for ``pixel_function``.
    :param options: Optional VRT creation options
                    (e.g., ``{'BLOCKXSIZE': 256, 'BLOCKYSIZE': 256}``).
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals,too-many-branches,too-many-statements
    if not input_imgs:
        raise rsgispy.RSGISPyException(
            "input_imgs must contain at least one image path."
        )
    for img_path in input_imgs:
        if not os.path.exists(img_path):
            raise rsgispy.RSGISPyException(
                f"Input image file does not exist: '{img_path}'."
            )

    resampling_str = _resampling_to_vrt_name(interp_method)
    creation_opts = _normalise_creation_options(options)

    with ExitStack() as stack:
        datasets = [
            stack.enter_context(rasterio.open(p, "r")) for p in input_imgs
        ]
        width, height, transform, ref_crs = _compute_vrt_grid(
            datasets=datasets,
            resolution=resolution,
            vrt_extent=vrt_extent,
            vrt_out_res_x=vrt_out_res_x,
            vrt_out_res_y=vrt_out_res_y,
            align_out_pxls=align_out_pxls,
            allow_proj_diff=allow_proj_diff,
        )

        ref_ds = datasets[0]
        if band_list is not None:
            selected_bands = [int(b) for b in band_list]
        else:
            selected_bands = list(range(1, ref_ds.count + 1))

        for ds in datasets:
            for b_idx in selected_bands:
                if b_idx < 1 or b_idx > ds.count:
                    raise rsgispy.RSGISPyException(
                        f"Band {b_idx} is out of range for image '{ds.name}' "
                        f"with {ds.count} band(s)."
                    )

        out_crs = _parse_epsg_crs(out_epsg) if out_epsg is not None else ref_crs

        vrt_root = ET.Element(
            "VRTDataset",
            attrib={"rasterXSize": str(width), "rasterYSize": str(height)},
        )
        if out_crs is not None:
            srs_elem = ET.SubElement(
                vrt_root, "SRS", attrib={"dataAxisToSRSAxisMapping": "1,2"}
            )
            srs_elem.text = out_crs.to_wkt()

        gt_elem = ET.SubElement(vrt_root, "GeoTransform")
        gt_elem.text = ", ".join(f"{v:.16e}" for v in transform.to_gdal())

        ref_nodatavals = (
            ref_ds.nodatavals
            if ref_ds.nodatavals
            else [ref_ds.nodata] * ref_ds.count
        )
        ref_descriptions = (
            ref_ds.descriptions
            if ref_ds.descriptions
            else (None,) * ref_ds.count
        )

        for out_band_idx, src_b_idx in enumerate(selected_bands, start=1):
            dtype_str = _gdal_typename(ref_ds.dtypes[src_b_idx - 1])
            band_attrib: Dict[str, str] = {
                "dataType": dtype_str,
                "band": str(out_band_idx),
            }
            if pixel_function:
                band_attrib["subClass"] = "VRTDerivedRasterBand"

            band_elem = ET.SubElement(
                vrt_root, "VRTRasterBand", attrib=band_attrib
            )
            _apply_vrt_band_creation_opts(band_elem, creation_opts)

            desc = ref_descriptions[src_b_idx - 1]
            if desc:
                desc_elem = ET.SubElement(band_elem, "Description")
                desc_elem.text = str(desc)

            eff_ref_nd = (
                src_nodata
                if src_nodata is not None
                else ref_nodatavals[src_b_idx - 1]
            )
            eff_vrt_nd = vrt_nodata if vrt_nodata is not None else eff_ref_nd
            if eff_vrt_nd is not None:
                nd_elem = ET.SubElement(band_elem, "NoDataValue")
                nd_elem.text = f"{eff_vrt_nd:g}"
            if hide_nodata:
                hide_elem = ET.SubElement(band_elem, "HideNoDataValue")
                hide_elem.text = "1"

            if ref_ds.colorinterp and len(ref_ds.colorinterp) >= src_b_idx:
                ci_name = ref_ds.colorinterp[src_b_idx - 1].name.capitalize()
                if ci_name != "Undefined":
                    ci_elem = ET.SubElement(band_elem, "ColorInterp")
                    ci_elem.text = ci_name

            for img_path, ds in zip(input_imgs, datasets):
                src_path, rel_attr = _format_vrt_source_path(
                    img_path, out_vrt_file, write_abs_path
                )
                ds_nodatavals = (
                    ds.nodatavals if ds.nodatavals else [ds.nodata] * ds.count
                )
                eff_src_nd = (
                    src_nodata
                    if src_nodata is not None
                    else ds_nodatavals[src_b_idx - 1]
                )
                _append_complex_source(
                    band_elem=band_elem,
                    src_ds=ds,
                    src_band=src_b_idx,
                    src_path=src_path,
                    rel_attr=rel_attr,
                    vrt_transform=transform,
                    resampling_str=resampling_str,
                    src_nodata=eff_src_nd,
                )

            if pixel_function:
                pf_elem = ET.SubElement(band_elem, "PixelFunctionType")
                pf_elem.text = str(pixel_function)
                if pixel_function_args:
                    pfa_elem = ET.SubElement(band_elem, "PixelFunctionArguments")
                    for arg_k, arg_v in pixel_function_args.items():
                        pfa_elem.attrib[str(arg_k)] = str(arg_v)
                skip_elem = ET.SubElement(band_elem, "SkipNonContributingSources")
                skip_elem.text = "true"

        if add_alpha:
            alpha_band_idx = len(selected_bands) + 1
            alpha_elem = ET.SubElement(
                vrt_root,
                "VRTRasterBand",
                attrib={"dataType": "Byte", "band": str(alpha_band_idx)},
            )
            _apply_vrt_band_creation_opts(alpha_elem, creation_opts)
            ci_elem = ET.SubElement(alpha_elem, "ColorInterp")
            ci_elem.text = "Alpha"
            for img_path, ds in zip(input_imgs, datasets):
                src_path, rel_attr = _format_vrt_source_path(
                    img_path, out_vrt_file, write_abs_path
                )
                _append_complex_source(
                    band_elem=alpha_elem,
                    src_ds=ds,
                    src_band=1,
                    src_path=src_path,
                    rel_attr=rel_attr,
                    vrt_transform=transform,
                    resampling_str=resampling_str,
                    scale_offset=255.0,
                    scale_ratio=0.0,
                )

    _write_vrt_xml(vrt_root, out_vrt_file)


def create_vrt_band_subset(
    input_img: str,
    img_bands: Sequence[int],
    out_vrt_img: str,
    vrt_extent: Optional[Tuple[float, float, float, float]] = None,
    vrt_out_res_x: Optional[float] = None,
    vrt_out_res_y: Optional[float] = None,
    interp_method: Union[int, str, Resampling] = rsgispy.INTERP_NEAREST_NEIGHBOUR,
    align_out_pxls: bool = False,
    src_nodata: Optional[float] = None,
    vrt_nodata: Optional[float] = None,
    hide_nodata: bool = False,
    out_epsg: Optional[int] = None,
    write_abs_path: bool = True,
    band_names: Optional[Sequence[str]] = None,
    options: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
) -> None:
    """
    Create a GDAL VRT for ``input_img`` containing only the bands specified in
    ``img_bands`` (in the order provided), using ``rasterio`` to inspect image metadata.

    :param input_img: Path to the input image file.
    :param img_bands: Sequence of 1-based band numbers in the order they should appear
                      in the output VRT.
    :param out_vrt_img: Path to the output VRT file.
    :param vrt_extent: Optional extent ``(minX, minY, maxX, maxY)`` for the VRT image.
    :param vrt_out_res_x: Optional X resolution of the output VRT.
    :param vrt_out_res_y: Optional Y resolution of the output VRT.
    :param interp_method: Interpolation method ``rsgispy.INTERP_*`` or
                          ``rasterio.enums.Resampling`` (Default = INTERP_NEAREST_NEIGHBOUR).
    :param align_out_pxls: Whether to align output bounds to multiples of the output
                           pixel resolution (Default = False).
    :param src_nodata: Optional override for source image no-data values.
    :param vrt_nodata: Optional override for VRT band-level no-data values.
    :param hide_nodata: Whether to hide the no-data value on VRT bands (Default = False).
    :param out_epsg: Optional EPSG code to assign as the output VRT projection.
    :param write_abs_path: Whether to write the absolute path to ``input_img`` in the VRT
                           (Default = True, matching ``rsgislib``).
    :param band_names: Optional sequence of band descriptions for the output VRT bands.
    :param options: Optional VRT creation options
                    (e.g., ``{'BLOCKXSIZE': 256, 'BLOCKYSIZE': 256}``).
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    if not os.path.exists(input_img):
        raise rsgispy.RSGISPyException(
            f"Input image file does not exist: '{input_img}'."
        )
    if not img_bands:
        raise rsgispy.RSGISPyException(
            "img_bands must contain at least one 1-based band index."
        )

    resampling_str = _resampling_to_vrt_name(interp_method)
    creation_opts = _normalise_creation_options(options)

    with rasterio.open(input_img, "r") as ds:
        for b_idx in img_bands:
            if b_idx < 1 or b_idx > ds.count:
                raise rsgispy.RSGISPyException(
                    f"Band {b_idx} is out of range for image '{input_img}' "
                    f"with {ds.count} band(s)."
                )

        width, height, transform, ref_crs = _compute_vrt_grid(
            datasets=[ds],
            resolution="average",
            vrt_extent=vrt_extent,
            vrt_out_res_x=vrt_out_res_x,
            vrt_out_res_y=vrt_out_res_y,
            align_out_pxls=align_out_pxls,
            allow_proj_diff=True,
        )

        out_crs = _parse_epsg_crs(out_epsg) if out_epsg is not None else ref_crs

        vrt_root = ET.Element(
            "VRTDataset",
            attrib={"rasterXSize": str(width), "rasterYSize": str(height)},
        )
        if out_crs is not None:
            srs_elem = ET.SubElement(
                vrt_root, "SRS", attrib={"dataAxisToSRSAxisMapping": "1,2"}
            )
            srs_elem.text = out_crs.to_wkt()

        gt_elem = ET.SubElement(vrt_root, "GeoTransform")
        gt_elem.text = ", ".join(f"{v:.16e}" for v in transform.to_gdal())

        src_path, rel_attr = _format_vrt_source_path(
            input_img, out_vrt_img, write_abs_path
        )
        nodata_list = (
            ds.nodatavals if ds.nodatavals else [ds.nodata] * ds.count
        )
        descriptions = (
            ds.descriptions if ds.descriptions else (None,) * ds.count
        )

        for out_band_idx, src_b_idx in enumerate(img_bands, start=1):
            dtype_str = _gdal_typename(ds.dtypes[src_b_idx - 1])
            band_elem = ET.SubElement(
                vrt_root,
                "VRTRasterBand",
                attrib={"dataType": dtype_str, "band": str(out_band_idx)},
            )
            _apply_vrt_band_creation_opts(band_elem, creation_opts)

            desc = (
                band_names[out_band_idx - 1]
                if band_names and len(band_names) >= out_band_idx
                else descriptions[src_b_idx - 1]
            )
            if desc:
                desc_elem = ET.SubElement(band_elem, "Description")
                desc_elem.text = str(desc)

            eff_src_nd = (
                src_nodata
                if src_nodata is not None
                else nodata_list[src_b_idx - 1]
            )
            eff_vrt_nd = vrt_nodata if vrt_nodata is not None else eff_src_nd
            if eff_vrt_nd is not None:
                nd_elem = ET.SubElement(band_elem, "NoDataValue")
                nd_elem.text = f"{eff_vrt_nd:g}"
            if hide_nodata:
                hide_elem = ET.SubElement(band_elem, "HideNoDataValue")
                hide_elem.text = "1"

            if ds.colorinterp and len(ds.colorinterp) >= src_b_idx:
                ci_name = ds.colorinterp[src_b_idx - 1].name.capitalize()
                if ci_name != "Undefined":
                    ci_elem = ET.SubElement(band_elem, "ColorInterp")
                    ci_elem.text = ci_name

            _append_complex_source(
                band_elem=band_elem,
                src_ds=ds,
                src_band=src_b_idx,
                src_path=src_path,
                rel_attr=rel_attr,
                vrt_transform=transform,
                resampling_str=resampling_str,
                src_nodata=eff_src_nd,
            )

    _write_vrt_xml(vrt_root, out_vrt_img)
