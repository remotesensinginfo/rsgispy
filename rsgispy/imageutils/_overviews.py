"""
Internal helpers for tiled, multi-core raster overview pyramid generation in
:mod:`rsgispy.imageutils.stats`.
"""

import math
import os
from typing import Any, Dict, Optional, Sequence, Tuple, Union
import xml.etree.ElementTree as ET

import numpy as np
import rasterio
import rasterio.shutil
from rasterio.enums import ColorInterp, Resampling
from rasterio.transform import Affine
from rasterio.windows import Window

import rsgispy
from rsgispy import get_gdal_datatype_name as _gdal_typename
from rsgispy.imageutils.vrt import (
    _resampling_to_vrt_name,
    _write_vrt_xml,
)
from rsgispy.tools.files import ensure_parent_dir

_VALID_OVR_RESAMPLING = frozenset(
    {
        "nearest",
        "bilinear",
        "cubic",
        "cubicspline",
        "lanczos",
        "average",
        "mode",
        "gauss",
        "max",
        "min",
        "med",
        "q1",
        "q3",
        "sum",
        "rms",
    }
)


def validate_positive_int(val: Any, label: str, min_val: int = 1) -> int:
    """Validate an integer argument is non-bool and ``>= min_val``."""
    if isinstance(val, bool) or not isinstance(val, (int, np.integer)):
        raise rsgispy.RSGISPyException(
            f"{label} must be an integer, got {val!r}."
        )
    int_val = int(val)
    if int_val < min_val:
        raise rsgispy.RSGISPyException(
            f"{label} must be at least {min_val}, got {int_val}."
        )
    return int_val


def normalise_ovr_resampling(
    resampling_method: Union[int, str, Resampling],
) -> str:
    """Convert and validate overview resampling method to a GDAL VRT name."""
    resamp_name = _resampling_to_vrt_name(resampling_method)
    if resamp_name not in _VALID_OVR_RESAMPLING:
        raise rsgispy.RSGISPyException(
            f"Unsupported overview resampling method '{resampling_method}'."
        )
    return resamp_name


def format_nodata_text(val: Any) -> str:
    """Format a scalar nodata value accurately for VRT XML text."""
    if isinstance(val, (int, np.integer)):
        return str(int(val))
    fval = float(val)
    if math.isnan(fval):
        return "nan"
    if fval.is_integer() and abs(fval) < 1e16:
        return str(int(fval))
    return repr(fval)


def uniform_nodata(nodatavals: Sequence[Any]) -> Any:
    """Return a single nodata value when all bands agree, else ``None``."""
    if not nodatavals:
        return None
    first = nodatavals[0]
    if first is None:
        return None
    first_nan = isinstance(first, float) and math.isnan(first)
    for val in nodatavals[1:]:
        if val is None:
            return None
        val_nan = isinstance(val, float) and math.isnan(val)
        if first_nan != val_nan:
            return None
        if not first_nan and float(val) != float(first):
            return None
    return first


def _build_tile_vrt_xml(
    src_path: str,
    src_rect_vals: Tuple[float, float, float, float],
    tile_w: int,
    tile_h: int,
    dtypes: Sequence[str],
    nodatavals: Sequence[Any],
    resamp_name: str,
    is_base_src: bool,
    crs_wkt: Optional[str],
    tile_transform: Optional[Affine],
) -> str:
    """Build an in-memory VRT XML string mapping a sub-pixel SrcRect to one tile."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    vrt_root = ET.Element(
        "VRTDataset", rasterXSize=str(tile_w), rasterYSize=str(tile_h)
    )
    if crs_wkt:
        ET.SubElement(vrt_root, "SRS").text = crs_wkt
    if tile_transform is not None:
        ET.SubElement(vrt_root, "GeoTransform").text = (
            f"{tile_transform.c:.17g}, {tile_transform.a:.17g}, "
            f"{tile_transform.b:.17g}, {tile_transform.f:.17g}, "
            f"{tile_transform.d:.17g}, {tile_transform.e:.17g}"
        )

    abs_src = os.path.abspath(src_path)
    sx_off, sy_off, sx_size, sy_size = src_rect_vals
    for b_idx, dtype_str in enumerate(dtypes, start=1):
        band_elem = ET.SubElement(
            vrt_root,
            "VRTRasterBand",
            dataType=_gdal_typename(dtype_str),
            band=str(b_idx),
        )
        nd_val = nodatavals[b_idx - 1]
        nd_text = format_nodata_text(nd_val) if nd_val is not None else None
        if nd_text is not None:
            ET.SubElement(band_elem, "NoDataValue").text = nd_text

        cs_elem = ET.SubElement(band_elem, "ComplexSource")
        if resamp_name != "nearest":
            cs_elem.attrib["resampling"] = resamp_name

        sf_elem = ET.SubElement(cs_elem, "SourceFilename", relativeToVRT="0")
        sf_elem.text = abs_src
        ET.SubElement(cs_elem, "SourceBand").text = str(b_idx)

        if is_base_src:
            oo_elem = ET.SubElement(cs_elem, "OpenOptions")
            ET.SubElement(oo_elem, "OOI", key="OVERVIEW_LEVEL").text = "NONE"

        ET.SubElement(
            cs_elem,
            "SrcRect",
            xOff=f"{sx_off:.17g}",
            yOff=f"{sy_off:.17g}",
            xSize=f"{sx_size:.17g}",
            ySize=f"{sy_size:.17g}",
        )
        ET.SubElement(
            cs_elem,
            "DstRect",
            xOff="0",
            yOff="0",
            xSize=str(tile_w),
            ySize=str(tile_h),
        )
        if nd_text is not None:
            ET.SubElement(cs_elem, "NODATA").text = nd_text

    return ET.tostring(vrt_root, encoding="unicode")


def render_ovr_tile_worker(
    task: Dict[str, Any],
) -> Tuple[str, int, int, int, int]:
    """
    Top-level picklable worker rendering a single overview tile from an exact
    floating-point ``SrcRect`` VRT into a temporary GeoTIFF file.
    """
    tile_w = task["tile_w"]
    tile_h = task["tile_h"]
    col_off = task["col_off"]
    row_off = task["row_off"]
    tile_xfm = (
        Affine(*task["xfm_vals"]) if task["xfm_vals"] is not None else None
    )
    vrt_xml = _build_tile_vrt_xml(
        src_path=task["src_path"],
        src_rect_vals=task["src_rect"],
        tile_w=tile_w,
        tile_h=tile_h,
        dtypes=task["dtypes"],
        nodatavals=task["nodatavals"],
        resamp_name=task["resamp_name"],
        is_base_src=task["is_base_src"],
        crs_wkt=task["crs_wkt"],
        tile_transform=tile_xfm,
    )
    with rasterio.open(vrt_xml, "r") as src:
        data = src.read()

    create_kwargs: Dict[str, Any] = {
        "driver": "GTiff",
        "width": tile_w,
        "height": tile_h,
        "count": len(task["dtypes"]),
        "dtype": task["dtypes"][0],
        "compress": task["compress"],
        "interleave": "BAND",
    }
    if task["no_data_val"] is not None:
        create_kwargs["nodata"] = task["no_data_val"]
    if task["crs_wkt"] is not None:
        create_kwargs["crs"] = task["crs_wkt"]
    if tile_xfm is not None:
        create_kwargs["transform"] = tile_xfm

    with rasterio.open(task["tile_path"], "w", **create_kwargs) as dst:
        dst.write(data)

    return task["tile_path"], col_off, row_off, tile_w, tile_h


def stitch_scale_tiles(
    lvl_path: str,
    tile_results: Sequence[Tuple[str, int, int, int, int]],
    ovr_w: int,
    ovr_h: int,
    n_bands: int,
    dtype_str: str,
    no_data_val: Any,
    crs_wkt: Optional[str],
    lvl_transform: Optional[Affine],
    compress: str,
    ovr_block_size: int,
) -> None:
    """Combine rendered scale tiles into one tiled GeoTIFF and delete tile files."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    create_kwargs: Dict[str, Any] = {
        "driver": "GTiff",
        "width": ovr_w,
        "height": ovr_h,
        "count": n_bands,
        "dtype": dtype_str,
        "tiled": True,
        "blockxsize": ovr_block_size,
        "blockysize": ovr_block_size,
        "compress": compress,
        "interleave": "BAND",
        "BIGTIFF": "IF_SAFER",
    }
    if no_data_val is not None:
        create_kwargs["nodata"] = no_data_val
    if crs_wkt is not None:
        create_kwargs["crs"] = crs_wkt
    if lvl_transform is not None:
        create_kwargs["transform"] = lvl_transform

    with rasterio.open(lvl_path, "w", **create_kwargs) as dst:
        for tile_path, col_off, row_off, tile_w, tile_h in tile_results:
            with rasterio.open(tile_path, "r") as tsrc:
                arr = tsrc.read()
            dst.write(arr, window=Window(col_off, row_off, tile_w, tile_h))
            os.remove(tile_path)


def write_ovr_bundle_vrt(
    vrt_path: str,
    primary_path: str,
    primary_w: int,
    primary_h: int,
    overview_paths: Sequence[str],
    dtypes: Sequence[str],
    nodatavals: Sequence[Any],
    crs_wkt: Optional[str],
    primary_transform: Optional[Affine],
    primary_is_base: bool,
    colour_interp: Optional[Sequence[ColorInterp]] = None,
    colour_maps: Optional[Dict[int, Dict[int, Tuple[int, int, int, int]]]] = None,
) -> None:
    """
    Build a VRT binding ``primary_path`` with explicit ``<Overview>`` entries
    pointing to precomputed scale rasters.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-statements
    vrt_root = ET.Element("VRTDataset")
    vrt_root.attrib["rasterXSize"] = str(primary_w)
    vrt_root.attrib["rasterYSize"] = str(primary_h)

    if crs_wkt:
        srs_elem = ET.SubElement(vrt_root, "SRS")
        srs_elem.text = crs_wkt
    if primary_transform is not None:
        gt_elem = ET.SubElement(vrt_root, "GeoTransform")
        gt_elem.text = (
            f"{primary_transform.c:.17g}, {primary_transform.a:.17g}, "
            f"{primary_transform.b:.17g}, {primary_transform.f:.17g}, "
            f"{primary_transform.d:.17g}, {primary_transform.e:.17g}"
        )

    abs_primary = os.path.abspath(primary_path)
    abs_overviews = [os.path.abspath(p) for p in overview_paths]

    for b_idx, dtype_str in enumerate(dtypes, start=1):
        band_elem = ET.SubElement(vrt_root, "VRTRasterBand")
        band_elem.attrib["dataType"] = _gdal_typename(dtype_str)
        band_elem.attrib["band"] = str(b_idx)

        nd_val = nodatavals[b_idx - 1]
        if nd_val is not None:
            nd_elem = ET.SubElement(band_elem, "NoDataValue")
            nd_elem.text = format_nodata_text(nd_val)

        if colour_interp is not None and b_idx - 1 < len(colour_interp):
            ci_name = colour_interp[b_idx - 1].name
            if ci_name in (
                "undefined",
                "gray",
                "palette",
                "red",
                "green",
                "blue",
                "alpha",
                "hue",
                "saturation",
                "lightness",
                "cyan",
                "magenta",
                "yellow",
                "black",
            ):
                ci_elem = ET.SubElement(band_elem, "ColorInterp")
                ci_elem.text = ci_name.capitalize()

        if colour_maps and b_idx in colour_maps:
            cmap = colour_maps[b_idx]
            ct_elem = ET.SubElement(band_elem, "ColorTable")
            max_idx = max(cmap.keys()) if cmap else -1
            for entry_idx in range(max_idx + 1):
                rgba = cmap.get(entry_idx, (0, 0, 0, 255))
                ent_elem = ET.SubElement(ct_elem, "Entry")
                ent_elem.attrib["c1"] = str(int(rgba[0]))
                ent_elem.attrib["c2"] = str(int(rgba[1]))
                ent_elem.attrib["c3"] = str(int(rgba[2]))
                ent_elem.attrib["c4"] = str(int(rgba[3]))

        ss_elem = ET.SubElement(band_elem, "SimpleSource")
        sf_elem = ET.SubElement(ss_elem, "SourceFilename")
        sf_elem.attrib["relativeToVRT"] = "0"
        sf_elem.text = abs_primary
        sb_elem = ET.SubElement(ss_elem, "SourceBand")
        sb_elem.text = str(b_idx)

        if primary_is_base:
            oo_elem = ET.SubElement(ss_elem, "OpenOptions")
            ooi_elem = ET.SubElement(oo_elem, "OOI")
            ooi_elem.attrib["key"] = "OVERVIEW_LEVEL"
            ooi_elem.text = "NONE"

        src_rect = ET.SubElement(ss_elem, "SrcRect")
        src_rect.attrib["xOff"] = "0"
        src_rect.attrib["yOff"] = "0"
        src_rect.attrib["xSize"] = str(primary_w)
        src_rect.attrib["ySize"] = str(primary_h)

        dst_rect = ET.SubElement(ss_elem, "DstRect")
        dst_rect.attrib["xOff"] = "0"
        dst_rect.attrib["yOff"] = "0"
        dst_rect.attrib["xSize"] = str(primary_w)
        dst_rect.attrib["ySize"] = str(primary_h)

        for ovr_p in abs_overviews:
            ov_elem = ET.SubElement(band_elem, "Overview")
            ov_sf = ET.SubElement(ov_elem, "SourceFilename")
            ov_sf.attrib["relativeToVRT"] = "0"
            ov_sf.text = ovr_p
            ov_sb = ET.SubElement(ov_elem, "SourceBand")
            ov_sb.text = str(b_idx)

    _write_vrt_xml(vrt_root, vrt_path)


def finalise_external_ovr(
    out_ovr_path: str,
    work_dir: str,
    scale_files: Sequence[Tuple[int, int, int, str, Optional[Affine]]],
    dtypes: Sequence[str],
    nodatavals: Sequence[Any],
    crs_wkt: Optional[str],
    comp_val: str,
    ovr_blk: int,
) -> None:
    """Combine precomputed scale TIFFs into an external ``.ovr`` GeoTIFF."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    ensure_parent_dir(out_ovr_path)
    _, first_w, first_h, first_path, first_xfm = scale_files[0]
    ovr_vrt = os.path.join(work_dir, "combine_ovr.vrt")
    write_ovr_bundle_vrt(
        vrt_path=ovr_vrt,
        primary_path=first_path,
        primary_w=first_w,
        primary_h=first_h,
        overview_paths=[entry[3] for entry in scale_files[1:]],
        dtypes=dtypes,
        nodatavals=nodatavals,
        crs_wkt=crs_wkt,
        primary_transform=first_xfm,
        primary_is_base=False,
    )
    staging_ovr = os.path.join(
        os.path.dirname(os.path.abspath(out_ovr_path)),
        f".tmp_{os.getpid()}_{os.path.basename(out_ovr_path)}",
    )
    try:
        with rasterio.Env(
            COMPRESS_OVERVIEW=comp_val,
            GDAL_TIFF_OVR_BLOCKSIZE=str(ovr_blk),
        ):
            rasterio.shutil.copy(
                ovr_vrt,
                staging_ovr,
                driver="GTiff",
                COPY_SRC_OVERVIEWS="YES",
                TILED="YES",
                BLOCKXSIZE=ovr_blk,
                BLOCKYSIZE=ovr_blk,
                COMPRESS=comp_val,
                INTERLEAVE="BAND",
                BIGTIFF="IF_SAFER",
            )
        os.replace(staging_ovr, out_ovr_path)
    finally:
        if os.path.exists(staging_ovr):
            os.remove(staging_ovr)


def finalise_internal_ovr(
    dest_path: str,
    input_path: str,
    work_dir: str,
    base_w: int,
    base_h: int,
    scale_files: Sequence[Tuple[int, int, int, str, Optional[Affine]]],
    dtypes: Sequence[str],
    nodatavals: Sequence[Any],
    crs_wkt: Optional[str],
    base_transform: Affine,
    has_geo: bool,
    colour_interp: Sequence[ColorInterp],
    colour_maps: Dict[int, Dict[int, Tuple[int, int, int, int]]],
    base_tags: Dict[str, str],
    band_tags: Sequence[Dict[str, str]],
    descriptions: Sequence[Optional[str]],
    comp_val: str,
    ovr_blk: int,
    remove_stale_ovr: bool,
) -> None:
    """Combine base image and precomputed scale TIFFs into internal GeoTIFF overviews."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    ensure_parent_dir(dest_path)
    full_vrt = os.path.join(work_dir, "combine_internal.vrt")
    write_ovr_bundle_vrt(
        vrt_path=full_vrt,
        primary_path=input_path,
        primary_w=base_w,
        primary_h=base_h,
        overview_paths=[entry[3] for entry in scale_files],
        dtypes=dtypes,
        nodatavals=nodatavals,
        crs_wkt=crs_wkt,
        primary_transform=base_transform if has_geo else None,
        primary_is_base=True,
        colour_interp=colour_interp,
        colour_maps=colour_maps,
    )
    staging_img = os.path.join(
        os.path.dirname(os.path.abspath(dest_path)),
        f".tmp_{os.getpid()}_{os.path.basename(dest_path)}",
    )
    try:
        with rasterio.Env(
            COMPRESS_OVERVIEW=comp_val,
            GDAL_TIFF_OVR_BLOCKSIZE=str(ovr_blk),
        ):
            rasterio.shutil.copy(
                full_vrt,
                staging_img,
                driver="GTiff",
                COPY_SRC_OVERVIEWS="YES",
                TILED="YES",
                COMPRESS=comp_val,
                INTERLEAVE="BAND",
                BIGTIFF="IF_SAFER",
            )
        with rasterio.open(
            staging_img, "r+", IGNORE_COG_LAYOUT_BREAK="YES"
        ) as out_dst:
            if colour_interp:
                out_dst.colorinterp = list(colour_interp)
            if base_tags:
                out_dst.update_tags(**base_tags)
            for b_idx in range(1, len(dtypes) + 1):
                if band_tags[b_idx - 1]:
                    out_dst.update_tags(b_idx, **band_tags[b_idx - 1])
                if descriptions[b_idx - 1]:
                    out_dst.set_band_description(b_idx, descriptions[b_idx - 1])
        os.replace(staging_img, dest_path)
    finally:
        if os.path.exists(staging_img):
            os.remove(staging_img)

    if remove_stale_ovr:
        stale_ovr = f"{input_path}.ovr"
        if os.path.exists(stale_ovr):
            os.remove(stale_ovr)
