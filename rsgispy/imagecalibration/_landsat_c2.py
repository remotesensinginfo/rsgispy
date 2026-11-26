"""
Shared internal helpers for USGS Landsat Collection 2 Level-1 and Level-2
product extraction, stacking, radiometric scaling, and pixel QA parsing.
"""

from contextlib import ExitStack
from dataclasses import dataclass
import math
import os
import shutil
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import rasterio
from rasterio.windows import Window

import rsgispy
from rsgispy.imageprocess import (
    cast_result_block,
    compute_spatial_intersection,
    ensure_file_exists,
    iter_dataset_windows,
    normalise_creation_options,
    open_multi_image_datasets,
    open_output_dataset,
    pop_stats_on_close,
)
from rsgispy.imageutils.metadata import get_file_img_extension
from rsgispy.tools.archives import (
    untar_bz_file,
    untar_file,
    untar_gz_file,
    unzip_file,
)
from rsgispy.tools.dicts import (
    dict_struct_does_path_exist,
    dict_struct_get_numeric_value,
    dict_struct_get_str_value,
)
from rsgispy.tools.files import find_first_file
from rsgispy.tools.sensors import read_landsat_mtl_to_dict
from rsgispy.tools.strings import uid_generator

__all__ = [
    "QA_BAND_NAMES",
    "LS_OLI_LV1_SPEC",
    "LS_OLI_LV2_SPEC",
    "LS_TM_LV2_SPEC",
    "parse_landsat_c2_qa_pixel",
    "process_landsat_c2_archive",
]

QA_BAND_NAMES: Tuple[str, ...] = (
    "Fill",
    "DilatedCloud",
    "Cirrus",
    "Cloud",
    "CloudShadow",
    "Snow",
    "Clear",
    "Water",
    "CloudConfidence",
    "CloudShadowConfidence",
    "SnowIceConfidence",
    "CirrusConfidence",
)

_QA_SINGLE_BITS: Tuple[Tuple[str, int], ...] = (
    ("Fill", 0),
    ("DilatedCloud", 1),
    ("Cirrus", 2),
    ("Cloud", 3),
    ("CloudShadow", 4),
    ("Snow", 5),
    ("Clear", 6),
    ("Water", 7),
)

_QA_CONFIDENCE_SHIFTS: Tuple[Tuple[str, int], ...] = (
    ("CloudConfidence", 8),
    ("CloudShadowConfidence", 10),
    ("SnowIceConfidence", 12),
    ("CirrusConfidence", 14),
)


@dataclass(frozen=True)
class LandsatC2ProductSpec:
    """Configuration describing a Landsat Collection 2 product family."""

    band_numbers: Tuple[str, ...]
    band_names: Tuple[str, ...]
    rescaling_group: str
    out_suffix: str
    divide_by_sin_sun_elev: bool = False


LS_OLI_LV1_SPEC = LandsatC2ProductSpec(
    band_numbers=("1", "2", "3", "4", "5", "6", "7"),
    band_names=("coastal", "blue", "green", "red", "nir", "swir1", "swir2"),
    rescaling_group="LEVEL1_RADIOMETRIC_RESCALING",
    out_suffix="toa",
    divide_by_sin_sun_elev=True,
)

LS_OLI_LV2_SPEC = LandsatC2ProductSpec(
    band_numbers=("1", "2", "3", "4", "5", "6", "7"),
    band_names=("coastal", "blue", "green", "red", "nir", "swir1", "swir2"),
    rescaling_group="LEVEL2_SURFACE_REFLECTANCE_PARAMETERS",
    out_suffix="sref",
    divide_by_sin_sun_elev=False,
)

LS_TM_LV2_SPEC = LandsatC2ProductSpec(
    band_numbers=("1", "2", "3", "4", "5", "7"),
    band_names=("blue", "green", "red", "nir", "swir1", "swir2"),
    rescaling_group="LEVEL2_SURFACE_REFLECTANCE_PARAMETERS",
    out_suffix="sref",
    divide_by_sin_sun_elev=False,
)


def parse_landsat_c2_qa_pixel(
    input_img: str,
    output_img: str,
    gdalformat: str = "GTIFF",
    *,
    calc_stats: bool = True,
    use_pbar: bool = True,
    creation_opts: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
) -> None:
    """
    Decode a Landsat Collection 2 ``QA_PIXEL`` bitmask raster into a 12-band
    ``uint8`` raster.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    ensure_file_exists(input_img, label="Landsat QA_PIXEL image")
    norm_opts = normalise_creation_options(creation_opts)

    with ExitStack() as stack:
        src = stack.enter_context(rasterio.open(input_img, "r"))
        if not np.issubdtype(np.dtype(src.dtypes[0]), np.integer):
            raise rsgispy.RSGISPyException(
                f"Landsat QA_PIXEL image '{input_img}' must have an integer "
                f"data type, got '{src.dtypes[0]}'."
            )

        dst = open_output_dataset(
            stack=stack,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype="uint8",
            grid=compute_spatial_intersection({input_img: src}),
            count=len(QA_BAND_NAMES),
            output_exists=False,
            creation_opts=norm_opts,
        )
        dst.nodata = 0
        pop_stats_on_close(
            dst, calc_stats, no_data_val=0, thematic=False, use_pbar=use_pbar
        )
        for bidx, bname in enumerate(QA_BAND_NAMES, start=1):
            dst.set_band_description(bidx, bname)

        for win in iter_dataset_windows(src.height, src.width, use_pbar=use_pbar):
            qa_arr = src.read(1, window=win).astype(np.uint32, copy=False)
            out_blk = np.empty(
                (len(QA_BAND_NAMES), int(win.height), int(win.width)),
                dtype=np.uint8,
            )
            band_i = 0
            for _, bit in _QA_SINGLE_BITS:
                out_blk[band_i] = ((qa_arr >> bit) & 1).astype(np.uint8)
                band_i += 1
            for _, shift in _QA_CONFIDENCE_SHIFTS:
                out_blk[band_i] = (((qa_arr >> shift) & 0x3) + 1).astype(
                    np.uint8
                )
                band_i += 1
            dst.write(out_blk, window=win)


def _extract_archive(input_file: str, dest_dir: str) -> str:
    """Extract a ``.tar``, ``.tar.gz``/``.tgz``, ``.tar.bz2``, or ``.zip`` archive."""
    lower_name = input_file.lower()
    if lower_name.endswith((".tar.gz", ".tgz")):
        return untar_gz_file(input_file, dest_dir, gen_arch_dir=True)
    if lower_name.endswith((".tar.bz2", ".tbz2")):
        return untar_bz_file(input_file, dest_dir, gen_arch_dir=True)
    if lower_name.endswith(".zip"):
        return unzip_file(input_file, dest_dir, gen_arch_dir=True)
    return untar_file(input_file, dest_dir, gen_arch_dir=True)


def _find_qa_pixel_filename(ls_head_info: Mapping[str, Any]) -> str:
    """Retrieve the Collection 2 pixel QA filename from parsed MTL metadata."""
    for qa_key in ("FILE_NAME_QUALITY_L1_PIXEL", "FILE_NAME_QUALITY_L2_PIXEL"):
        if dict_struct_does_path_exist(
            dict(ls_head_info), ["PRODUCT_CONTENTS", qa_key]
        ):
            return dict_struct_get_str_value(
                dict(ls_head_info), ["PRODUCT_CONTENTS", qa_key]
            )
    raise rsgispy.RSGISPyException(
        "Could not find pixel QA image filename ('FILE_NAME_QUALITY_L1_PIXEL' "
        "or 'FILE_NAME_QUALITY_L2_PIXEL') in Landsat MTL header."
    )


def _write_landsat_cloud_and_valid_masks(
    qa_pxl_img_path: str,
    cloud_msk_img: str,
    vld_msk_img: str,
    gdalformat: str,
    calc_stats: bool,
    use_pbar: bool,
    creation_opts: Optional[Dict[str, str]],
) -> None:
    """Derive ``_clouds`` and ``_vld_msk`` rasters directly from raw ``QA_PIXEL``."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    with ExitStack() as stack:
        src = stack.enter_context(rasterio.open(qa_pxl_img_path, "r"))
        grid = compute_spatial_intersection({qa_pxl_img_path: src})
        cloud_dst = open_output_dataset(
            stack=stack,
            output_img=cloud_msk_img,
            gdalformat=gdalformat,
            out_rio_dtype="uint8",
            grid=grid,
            count=1,
            output_exists=False,
            creation_opts=creation_opts,
        )
        cloud_dst.nodata = 0
        pop_stats_on_close(cloud_dst, calc_stats, thematic=True, use_pbar=use_pbar)

        vld_dst = open_output_dataset(
            stack=stack,
            output_img=vld_msk_img,
            gdalformat=gdalformat,
            out_rio_dtype="uint8",
            grid=grid,
            count=1,
            output_exists=False,
            creation_opts=creation_opts,
        )
        vld_dst.nodata = 0
        pop_stats_on_close(vld_dst, calc_stats, thematic=True, use_pbar=use_pbar)

        for win in iter_dataset_windows(src.height, src.width, use_pbar=use_pbar):
            qa_arr = src.read(1, window=win).astype(np.uint32, copy=False)
            fill_bit = (qa_arr & 1) != 0
            # DilatedCloud (bit 1), Cirrus (bit 2), Cloud (bit 3) -> 1; CloudShadow (bit 4) -> 2
            cloud_or_cirrus = (qa_arr & ((1 << 1) | (1 << 2) | (1 << 3))) != 0
            cloud_shadow = (qa_arr & (1 << 4)) != 0

            cloud_out = np.zeros(qa_arr.shape, dtype=np.uint8)
            cloud_out[cloud_shadow] = 2
            cloud_out[cloud_or_cirrus] = 1
            cloud_dst.write(cloud_out, 1, window=win)

            vld_out = np.where(fill_bit, 0, 1).astype(np.uint8)
            vld_dst.write(vld_out, 1, window=win)


def process_landsat_c2_archive(
    input_file: str,
    out_dir: str,
    tmp_dir: str,
    spec: LandsatC2ProductSpec,
    scale_factor: int = 10000,
    gdalformat: str = "GTIFF",
    delete_inter_data: bool = True,
    inc_full_qa_pxl_img: bool = False,
    *,
    calc_stats: bool = True,
    use_pbar: bool = True,
    creation_opts: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
) -> None:
    """
    Extract a USGS Landsat Collection 2 archive and produce a calibrated,
    stacked multi-band ``int32`` raster plus cloud and valid-data masks.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-statements
    ensure_file_exists(input_file, label="Input Landsat archive")
    sf_val = float(scale_factor)
    if not math.isfinite(sf_val) or sf_val <= 0.0:
        raise rsgispy.RSGISPyException(
            f"scale_factor must be finite and > 0, got {scale_factor!r}."
        )
    norm_opts = normalise_creation_options(creation_opts)
    out_img_ext = get_file_img_extension(gdalformat)

    out_dir_abs = os.path.abspath(out_dir)
    tmp_dir_abs = os.path.abspath(tmp_dir)
    os.makedirs(out_dir_abs, exist_ok=True)
    os.makedirs(tmp_dir_abs, exist_ok=True)

    uid_tmp_dir = os.path.join(tmp_dir_abs, f"ls_c2_{uid_generator(8)}")
    os.makedirs(uid_tmp_dir, exist_ok=True)

    try:
        raw_dir = _extract_archive(input_file, uid_tmp_dir)
        mtl_header = find_first_file(raw_dir, "*MTL.txt", rtn_except=False)
        if mtl_header is None:
            raise rsgispy.RSGISPyException(
                f"Could not find the MTL header inside '{input_file}'."
            )
        mtl_dir = os.path.dirname(mtl_header)
        ls_head_info = read_landsat_mtl_to_dict(mtl_header)

        ls_prod_id = dict_struct_get_str_value(
            ls_head_info, ["PRODUCT_CONTENTS", "LANDSAT_PRODUCT_ID"]
        ).lower()

        geom_denom = 1.0
        if spec.divide_by_sin_sun_elev:
            sun_elev = float(
                dict_struct_get_numeric_value(
                    ls_head_info, ["IMAGE_ATTRIBUTES", "SUN_ELEVATION"]
                )
            )
            geom_denom = math.sin(math.radians(sun_elev))
            if geom_denom <= 0.0:
                raise rsgispy.RSGISPyException(
                    f"SUN_ELEVATION must be > 0 degrees for TOA conversion, "
                    f"got {sun_elev}."
                )

        band_entries = []
        for bnum in spec.band_numbers:
            band_file = dict_struct_get_str_value(
                ls_head_info, ["PRODUCT_CONTENTS", f"FILE_NAME_BAND_{bnum}"]
            )
            mult = float(
                dict_struct_get_numeric_value(
                    ls_head_info,
                    [spec.rescaling_group, f"REFLECTANCE_MULT_BAND_{bnum}"],
                )
            )
            add = float(
                dict_struct_get_numeric_value(
                    ls_head_info,
                    [spec.rescaling_group, f"REFLECTANCE_ADD_BAND_{bnum}"],
                )
            )
            band_path = os.path.join(mtl_dir, band_file)
            ensure_file_exists(band_path, label=f"Landsat band {bnum}")
            band_entries.append((band_path, mult, add))

        qa_fname = _find_qa_pixel_filename(ls_head_info)
        qa_pxl_img_path = os.path.join(mtl_dir, qa_fname)
        ensure_file_exists(qa_pxl_img_path, label="Landsat QA_PIXEL image")

        output_img = os.path.join(
            out_dir_abs, f"{ls_prod_id}_{spec.out_suffix}.{out_img_ext}"
        )
        in_paths = [entry[0] for entry in band_entries]
        out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(rsgispy.TYPE_32INT))
        out_rio_dtype = rsgispy.get_rasterio_datatype(rsgispy.TYPE_32INT)
        nd_val = -9999

        with ExitStack() as stack:
            datasets, paths = open_multi_image_datasets(stack, in_paths)
            grid = compute_spatial_intersection(datasets)
            dst = open_output_dataset(
                stack=stack,
                output_img=output_img,
                gdalformat=gdalformat,
                out_rio_dtype=out_rio_dtype,
                grid=grid,
                count=len(band_entries),
                output_exists=False,
                creation_opts=norm_opts,
            )
            dst.nodata = nd_val
            pop_stats_on_close(
                dst,
                calc_stats,
                no_data_val=nd_val,
                thematic=False,
                use_pbar=use_pbar,
            )
            for bidx, bname in enumerate(spec.band_names, start=1):
                dst.set_band_description(bidx, bname)

            for dst_win in iter_dataset_windows(
                grid.height, grid.width, use_pbar=use_pbar
            ):
                col_off, row_off = int(dst_win.col_off), int(dst_win.row_off)
                win_w, win_h = int(dst_win.width), int(dst_win.height)
                out_blk = np.empty(
                    (len(band_entries), win_h, win_w), dtype=out_np_dtype
                )
                for b_i, (bpath, mult, add) in enumerate(band_entries):
                    ds = datasets[bpath]
                    off_x, off_y = grid.offsets[bpath]
                    src_win = Window(
                        off_x + col_off, off_y + row_off, win_w, win_h
                    )
                    raw_arr = ds.read(1, window=src_win)
                    valid = raw_arr != 0
                    scaled = (
                        ((raw_arr.astype(np.float64) * mult) + add) / geom_denom
                    ) * sf_val
                    band_out = np.where(valid, scaled, float(nd_val))
                    out_blk[b_i] = cast_result_block(
                        band_out,
                        win_h,
                        win_w,
                        out_np_dtype,
                        clamp_int=True,
                    )
                dst.write(out_blk, window=dst_win)
            _ = paths

        if inc_full_qa_pxl_img:
            qa_pxl_mband_img = os.path.join(
                out_dir_abs, f"{ls_prod_id}_pxl_qa.{out_img_ext}"
            )
            parse_landsat_c2_qa_pixel(
                qa_pxl_img_path,
                qa_pxl_mband_img,
                gdalformat=gdalformat,
                calc_stats=calc_stats,
                use_pbar=use_pbar,
                creation_opts=norm_opts,
            )

        cloud_msk_img = os.path.join(
            out_dir_abs, f"{ls_prod_id}_clouds.{out_img_ext}"
        )
        vld_msk_img = os.path.join(
            out_dir_abs, f"{ls_prod_id}_vld_msk.{out_img_ext}"
        )
        _write_landsat_cloud_and_valid_masks(
            qa_pxl_img_path=qa_pxl_img_path,
            cloud_msk_img=cloud_msk_img,
            vld_msk_img=vld_msk_img,
            gdalformat=gdalformat,
            calc_stats=calc_stats,
            use_pbar=use_pbar,
            creation_opts=norm_opts,
        )
    finally:
        if delete_inter_data and os.path.exists(uid_tmp_dir):
            shutil.rmtree(uid_tmp_dir, ignore_errors=True)
