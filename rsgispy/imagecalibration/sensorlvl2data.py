# pylint: disable=too-many-lines
"""
The imagecalibration.sensorlvl2data module provides utilities for ingesting
Level-2 surface reflectance products (USGS Landsat Collection 2 Level-2 and
ESA Sentinel-2 Level-2A SAFE) and parsing Landsat Collection 2 ``QA_PIXEL``
bitmasks.
"""

from contextlib import ExitStack
import math
import os
import shutil
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import rasterio
from rasterio.enums import Resampling

import rsgispy
from rsgispy.imagecalibration._landsat_c2 import (
    LS_OLI_LV2_SPEC,
    LS_TM_LV2_SPEC,
    parse_landsat_c2_qa_pixel,
    process_landsat_c2_archive,
)
from rsgispy.imageprocess import (
    build_valid_mask,
    cast_result_block,
    compute_spatial_intersection,
    iter_dataset_windows,
    normalise_creation_options,
    open_output_dataset,
    pop_stats_on_close,
    resolve_out_dtype,
)
from rsgispy.imageutils.bands import (
    SharpBandInfo,
    sharpen_low_res_bands,
    stack_img_bands,
)
from rsgispy.imageutils.metadata import get_file_img_extension
from rsgispy.imageutils.stats import pop_thmt_img_stats
from rsgispy.imageutils.vrt import create_stack_images_vrt
from rsgispy.imageutils.warp import resample_img_to_match
from rsgispy.tools.files import find_first_file, get_file_basename
from rsgispy.tools.sensors import read_sen2_l2a_mtd_to_dict
from rsgispy.tools.strings import uid_generator

__all__ = [
    "parse_landsat_c2_qa_pixel_img",
    "create_stacked_sref_ls_oli_cl2_lv2_img",
    "create_stacked_sref_ls_4_5_7_lv2_img",
    "create_stacked_sen2_esa_l2a_img",
]

_SEN2_OUT_BAND_ORDER: Tuple[str, ...] = (
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B11",
    "B12",
)

_SEN2_OUT_BAND_NAMES: Tuple[str, ...] = (
    "Blue",
    "Green",
    "Red",
    "RE_B5",
    "RE_B6",
    "RE_B7",
    "NIR_B8",
    "NIR_B8A",
    "SWIR1",
    "SWIR2",
)

# Band order expected by the CloudSEN12 L2A cloud masking model
# (see :mod:`rsgispy.imagecalibration.cloudmasking`).
_SEN2_CLDSEN12_BAND_ORDER: Tuple[str, ...] = (
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B11",
    "B12",
)

_SEN2_CLDSEN12_BAND_NAMES: Tuple[str, ...] = (
    "Coastal",
    "Blue",
    "Green",
    "Red",
    "RE_B5",
    "RE_B6",
    "RE_B7",
    "NIR_B8",
    "NIR_B8A",
    "WV_B9",
    "SWIR1",
    "SWIR2",
)


def parse_landsat_c2_qa_pixel_img(
    input_img: str,
    output_img: str,
    gdalformat: str = "GTIFF",
    *,
    calc_stats: bool = True,
    use_pbar: bool = True,
    creation_opts: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
) -> None:
    """
    Parse a bit-encoded USGS Landsat Collection 2 ``QA_PIXEL`` raster into a
    12-band ``uint8`` output image.

    The output bands are:

    1. ``Fill`` — No-data fill regions (0–1)
    2. ``DilatedCloud`` — Cloud dilation regions (0–1)
    3. ``Cirrus`` — Cirrus clouds present (0–1)
    4. ``Cloud`` — Clouds present (0–1)
    5. ``CloudShadow`` — Cloud shadow present (0–1)
    6. ``Snow`` — Snow/ice present (0–1)
    7. ``Clear`` — Clear sky pixel (0–1)
    8. ``Water`` — Water present (0–1)
    9. ``CloudConfidence`` — Cloud confidence (1 = None, 2 = Low, 3 = Medium, 4 = High)
    10. ``CloudShadowConfidence`` — Cloud shadow confidence (1–4)
    11. ``SnowIceConfidence`` — Snow/ice confidence (1–4)
    12. ``CirrusConfidence`` — Cirrus confidence (1–4)

    :param input_img: Path to the input Landsat ``QA_PIXEL`` image.
    :param output_img: Path to the output 12-band ``uint8`` QA image.
    :param gdalformat: Output GDAL driver format (Default = ``'GTIFF'``).
    :param calc_stats: If ``True``, populate image statistics and pyramids
                       (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).
    :param creation_opts: Optional GDAL creation options (dict or sequence of
                          ``'KEY=VALUE'``).

    .. code:: python

        import rsgispy.imagecalibration.sensorlvl2data as l2

        l2.parse_landsat_c2_qa_pixel_img("LC08_QA_PIXEL.tif", "qa_bands.tif")
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    parse_landsat_c2_qa_pixel(
        input_img=input_img,
        output_img=output_img,
        gdalformat=gdalformat,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
        creation_opts=creation_opts,
    )


def create_stacked_sref_ls_oli_cl2_lv2_img(
    input_file: str,
    out_dir: str,
    tmp_dir: str,
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
    Extract USGS Landsat Collection 2 Level-2 (Landsat 8/9 OLI) data from an
    archive and create a single stacked multi-band surface reflectance raster
    (32-bit integer, nodata = ``-9999``), along with cloud and valid-pixel masks
    derived from ``QA_PIXEL``.

    :param input_file: File path to the input Landsat Collection 2 archive.
    :param out_dir: Output directory where calibrated rasters and masks will be
                    written.
    :param tmp_dir: Temporary working directory for archive extraction.
    :param scale_factor: Multiplicative scale factor applied to surface
                         reflectance values (Default = ``10000``).
    :param gdalformat: Output GDAL driver format (Default = ``'GTIFF'``).
    :param delete_inter_data: Delete extracted intermediate files when done
                              (Default = ``True``).
    :param inc_full_qa_pxl_img: If ``True``, also write the decoded 12-band
                                pixel QA raster (Default = ``False``).
    :param calc_stats: If ``True``, populate image statistics and pyramids
                       (Default = ``True``).
    :param use_pbar: If ``True``, display tqdm progress bars (Default = ``True``).
    :param creation_opts: Optional GDAL creation options (dict or sequence of
                          ``'KEY=VALUE'``).

    .. code:: python

        import rsgispy.imagecalibration.sensorlvl2data as l2

        l2.create_stacked_sref_ls_oli_cl2_lv2_img(
            "LC08_L2SP_135046_20210319_20210328_02_T1.tar",
            "outputs",
            "tmp",
            gdalformat="GTIFF",
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    process_landsat_c2_archive(
        input_file=input_file,
        out_dir=out_dir,
        tmp_dir=tmp_dir,
        spec=LS_OLI_LV2_SPEC,
        scale_factor=scale_factor,
        gdalformat=gdalformat,
        delete_inter_data=delete_inter_data,
        inc_full_qa_pxl_img=inc_full_qa_pxl_img,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
        creation_opts=creation_opts,
    )


def create_stacked_sref_ls_4_5_7_lv2_img(
    input_file: str,
    out_dir: str,
    tmp_dir: str,
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
    Extract USGS Landsat Collection 2 Level-2 (Landsat 4, 5, and 7 TM/ETM+)
    data from an archive and create a single stacked multi-band surface
    reflectance raster (32-bit integer, nodata = ``-9999``), along with cloud
    and valid-pixel masks derived from ``QA_PIXEL``.

    :param input_file: File path to the input Landsat Collection 2 archive.
    :param out_dir: Output directory where calibrated rasters and masks will be
                    written.
    :param tmp_dir: Temporary working directory for archive extraction.
    :param scale_factor: Multiplicative scale factor applied to surface
                         reflectance values (Default = ``10000``).
    :param gdalformat: Output GDAL driver format (Default = ``'GTIFF'``).
    :param delete_inter_data: Delete extracted intermediate files when done
                              (Default = ``True``).
    :param inc_full_qa_pxl_img: If ``True``, also write the decoded 12-band
                                pixel QA raster (Default = ``False``).
    :param calc_stats: If ``True``, populate image statistics and pyramids
                       (Default = ``True``).
    :param use_pbar: If ``True``, display tqdm progress bars (Default = ``True``).
    :param creation_opts: Optional GDAL creation options (dict or sequence of
                          ``'KEY=VALUE'``).

    .. code:: python

        import rsgispy.imagecalibration.sensorlvl2data as l2

        l2.create_stacked_sref_ls_4_5_7_lv2_img(
            "LT05_L2SP_204024_20060715_20200831_02_T1.tar",
            "outputs",
            "tmp",
            gdalformat="GTIFF",
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    process_landsat_c2_archive(
        input_file=input_file,
        out_dir=out_dir,
        tmp_dir=tmp_dir,
        spec=LS_TM_LV2_SPEC,
        scale_factor=scale_factor,
        gdalformat=gdalformat,
        delete_inter_data=delete_inter_data,
        inc_full_qa_pxl_img=inc_full_qa_pxl_img,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
        creation_opts=creation_opts,
    )


def _resolve_s2_img_path(input_safe_abs: str, rel_entry: str) -> str:
    """Resolve an ``IMAGE_FILE`` path inside a Sentinel-2 ``.SAFE`` directory."""
    direct = os.path.join(input_safe_abs, rel_entry)
    if os.path.isfile(direct):
        return direct
    for ext in (".jp2", ".tif", ".tiff"):
        candidate = f"{direct}{ext}"
        if os.path.isfile(candidate):
            return candidate
    raise rsgispy.RSGISPyException(
        f"Sentinel-2 image file '{rel_entry}' was not found under "
        f"'{input_safe_abs}'."
    )


def _match_band_token(img_entry: str, band_code: str) -> bool:
    """Return True if ``band_code`` appears as an underscore-delimited filename token."""
    stem = os.path.splitext(os.path.basename(img_entry))[0]
    tokens = stem.split("_")
    return band_code in tokens


def _collect_sen2_band_paths(
    input_safe_abs: str, img_files: Sequence[str]
) -> Tuple[
    Dict[str, Optional[str]],
    Dict[str, Optional[str]],
    Optional[str],
    Dict[str, Optional[str]],
]:
    """
    Map Sentinel-2 10 m, 20 m, SCL and coarse (B01/B09) ``IMAGE_FILE`` paths
    without stopping early. For B01 and B09 the finest available resolution
    (20 m, then 60 m) is selected.
    """
    # pylint: disable=too-many-branches
    coarse_imgs: Dict[str, Optional[str]] = {"B01": None, "B09": None}
    coarse_res: Dict[str, int] = {}
    imgs_10m: Dict[str, Optional[str]] = {
        b: None for b in ("B02", "B03", "B04", "B08")
    }
    all_20m_imgs: Dict[str, Optional[str]] = {
        b: None
        for b in ("B02", "B03", "B04", "B05", "B06", "B07", "B8A", "B11", "B12")
    }
    scl_20m_img: Optional[str] = None

    for img in img_files:
        norm_img = img.replace("\\", "/")
        fname_tokens = set(os.path.splitext(os.path.basename(norm_img))[0].split("_"))
        is_10m = "10m" in fname_tokens or "/R10m/" in norm_img
        is_20m = "20m" in fname_tokens or "/R20m/" in norm_img
        is_60m = "60m" in fname_tokens or "/R60m/" in norm_img
        if is_20m or is_60m:
            img_res = 20 if is_20m else 60
            for band in coarse_imgs:
                if _match_band_token(norm_img, band) and img_res < coarse_res.get(
                    band, 999
                ):
                    coarse_imgs[band] = _resolve_s2_img_path(input_safe_abs, img)
                    coarse_res[band] = img_res
        if is_10m:
            for band in imgs_10m:
                if _match_band_token(norm_img, band):
                    imgs_10m[band] = _resolve_s2_img_path(input_safe_abs, img)
                    break
        elif is_20m:
            if _match_band_token(norm_img, "SCL"):
                scl_20m_img = _resolve_s2_img_path(input_safe_abs, img)
                continue
            for band in all_20m_imgs:
                if _match_band_token(norm_img, band):
                    all_20m_imgs[band] = _resolve_s2_img_path(input_safe_abs, img)
                    break

    return imgs_10m, all_20m_imgs, scl_20m_img, coarse_imgs


def _write_sen2_scl_derivatives(
    scl_src_img: str,
    out_scl_img: Optional[str],
    out_clouds_img: str,
    out_vld_img: str,
    gdalformat: str,
    calc_stats: bool,
    use_pbar: bool,
    creation_opts: Optional[Dict[str, str]],
) -> None:
    """Write thematic Sentinel-2 SCL copy (optional), clouds, and valid masks."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    is_cog = gdalformat.upper().strip() == "COG"
    with ExitStack() as stack:
        src = stack.enter_context(rasterio.open(scl_src_img, "r"))
        grid = compute_spatial_intersection({scl_src_img: src})
        scl_dst = (
            open_output_dataset(
                stack=stack,
                output_img=out_scl_img,
                gdalformat=gdalformat,
                out_rio_dtype="uint8",
                grid=grid,
                count=1,
                output_exists=False,
                creation_opts=creation_opts,
            )
            if out_scl_img is not None
            else None
        )
        clouds_dst = open_output_dataset(
            stack=stack,
            output_img=out_clouds_img,
            gdalformat=gdalformat,
            out_rio_dtype="uint8",
            grid=grid,
            count=1,
            output_exists=False,
            creation_opts=creation_opts,
        )
        clouds_dst.nodata = 0

        vld_dst = open_output_dataset(
            stack=stack,
            output_img=out_vld_img,
            gdalformat=gdalformat,
            out_rio_dtype="uint8",
            grid=grid,
            count=1,
            output_exists=False,
            creation_opts=creation_opts,
        )
        vld_dst.nodata = 0

        for win in iter_dataset_windows(src.height, src.width, use_pbar=use_pbar):
            scl_arr = src.read(1, window=win).astype(np.uint8, copy=False)
            if scl_dst is not None:
                scl_dst.write(scl_arr, 1, window=win)

            clouds_out = np.zeros(scl_arr.shape, dtype=np.uint8)
            clouds_out[scl_arr == 3] = 2
            clouds_out[(scl_arr == 8) | (scl_arr == 9) | (scl_arr == 10)] = 1
            clouds_dst.write(clouds_out, 1, window=win)

            # SCL 0 is NO_DATA; non-zero classes represent valid scene pixels.
            vld_out = np.where(scl_arr != 0, 1, 0).astype(np.uint8)
            vld_dst.write(vld_out, 1, window=win)

    if calc_stats:
        if out_scl_img is not None:
            pop_thmt_img_stats(
                out_scl_img,
                add_clr_tab=True,
                calc_pyramids=True,
                ignore_zero=False,
                use_pbar=use_pbar,
                ignore_cog=is_cog,
            )
        pop_thmt_img_stats(
            out_clouds_img,
            add_clr_tab=True,
            calc_pyramids=True,
            ignore_zero=True,
            use_pbar=use_pbar,
            ignore_cog=is_cog,
        )
        pop_thmt_img_stats(
            out_vld_img,
            add_clr_tab=True,
            calc_pyramids=True,
            ignore_zero=True,
            use_pbar=use_pbar,
            ignore_cog=is_cog,
        )


def _finalise_sen2_sref_output(
    band_sources: Sequence[Tuple[str, int]],
    out_sref_img: str,
    band_offsets: Sequence[float],
    band_names: Sequence[str],
    quant_boa_val: float,
    scale_factor: float,
    no_data_val: float,
    gdalformat: str,
    datatype: int,
    calc_stats: bool,
    use_pbar: bool,
    creation_opts: Optional[Dict[str, str]],
) -> None:
    """
    Apply Sentinel-2 quantification scaling & BOA offsets and set metadata.

    ``band_sources`` gives, for each output band, the ``(image_path, band_index)``
    to read; all source images must share the same pixel grid.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    _, out_rio_dtype, out_np_dtype = resolve_out_dtype(
        datatype, default_dtype=rsgispy.TYPE_32INT
    )
    scale_ratio = float(scale_factor) / float(quant_boa_val)
    if not len(band_sources) == len(band_offsets) == len(band_names):
        raise rsgispy.RSGISPyException(
            "band_sources, band_offsets and band_names must have the same length."
        )

    with ExitStack() as stack:
        srcs: Dict[str, Any] = {}
        for img_path, _ in band_sources:
            if img_path not in srcs:
                srcs[img_path] = stack.enter_context(rasterio.open(img_path, "r"))
        ref_src = srcs[band_sources[0][0]]
        for img_path, img_src in srcs.items():
            if (img_src.height, img_src.width) != (ref_src.height, ref_src.width):
                raise rsgispy.RSGISPyException(
                    f"Image '{img_path}' does not match the pixel grid of "
                    f"'{band_sources[0][0]}'."
                )
        grid = compute_spatial_intersection({band_sources[0][0]: ref_src})
        dst = open_output_dataset(
            stack=stack,
            output_img=out_sref_img,
            gdalformat=gdalformat,
            out_rio_dtype=out_rio_dtype,
            grid=grid,
            count=len(band_sources),
            output_exists=False,
            creation_opts=creation_opts,
        )
        dst.nodata = no_data_val
        pop_stats_on_close(
            dst,
            calc_stats,
            no_data_val=no_data_val,
            thematic=False,
            use_pbar=use_pbar,
        )
        for bidx, bname in enumerate(band_names, start=1):
            dst.set_band_description(bidx, bname)

        for win in iter_dataset_windows(
            ref_src.height, ref_src.width, use_pbar=use_pbar
        ):
            blk = np.stack(
                [
                    srcs[img_path].read(band, window=win, out_dtype=np.float64)
                    for img_path, band in band_sources
                ],
                axis=0,
            )
            out_blk = np.empty_like(blk, dtype=np.float64)
            for b_i, add_offset in enumerate(band_offsets):
                arr = blk[b_i]
                valid = build_valid_mask(arr, no_data_val=no_data_val)
                scaled = (arr + float(add_offset)) * scale_ratio
                out_blk[b_i] = np.where(valid, scaled, no_data_val)
            dst.write(
                cast_result_block(
                    out_blk,
                    int(win.height),
                    int(win.width),
                    out_np_dtype,
                    clamp_int=True,
                ),
                window=win,
            )


def create_stacked_sen2_esa_l2a_img(
    input_safe: str,
    out_dir: str,
    tmp_dir: str,
    out_res_10m: bool = True,
    sharpen_10m: bool = True,
    scale_factor: int = 10000,
    gdalformat: str = "GTIFF",
    datatype: int = rsgispy.TYPE_32INT,
    delete_inter_data: bool = True,
    resample_10m_method: Union[int, Resampling] = rsgispy.INTERP_CUBIC,
    resample_20m_method: Union[int, Resampling] = rsgispy.INTERP_AVERAGE,
    use_multi_core: bool = False,
    *,
    inc_cloudsen12_img: bool = False,
    calc_stats: bool = True,
    use_pbar: bool = True,
    creation_opts: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
) -> None:
    """
    Create a stacked multi-band Sentinel-2 Level-2A surface reflectance image and
    accompanying SCL, cloud, and valid-pixel masks from an ESA ``.SAFE``
    directory.

    Optionally (``inc_cloudsen12_img=True``) also create a 12-band surface
    reflectance stack (``*_cloudsen12_sref``) with the bands
    ``B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12`` in the order
    required as input to
    :func:`rsgispy.imagecalibration.cloudmasking.apply_cloudsen12_s2_sref_cloud_mask`.

    Resamples 20 m bands to 10 m (with optional Shepherd & Dymond band
    sharpening using B02, B03, B04, B08 when ``sharpen_10m=True``) when
    ``out_res_10m=True``, or resamples 10 m B08 to 20 m when
    ``out_res_10m=False``. Automatically accounts for Processing Baseline >=
    04.00 ``BOA_ADD_OFFSET`` values if present in ``MTD_MSIL2A.xml``.

    :param input_safe: Input ESA Sentinel-2 L2A ``.SAFE`` directory path.
    :param out_dir: Output directory where stacked imagery and masks are written.
    :param tmp_dir: Temporary working directory for intermediate files.
    :param out_res_10m: If ``True``, output 10 m resolution products; otherwise
                        output 20 m products (Default = ``True``).
    :param sharpen_10m: If ``True`` and ``out_res_10m=True``, sharpen the 20 m
                        bands using high-resolution 10 m bands (Default = ``True``).
    :param scale_factor: Multiplicative scale factor applied to surface
                         reflectance outputs (Default = ``10000``).
    :param gdalformat: Output GDAL driver format (Default = ``'GTIFF'``).
    :param datatype: Output ``rsgispy.TYPE_*`` constant
                     (Default = :data:`rsgispy.TYPE_32INT`).
    :param delete_inter_data: Delete intermediate files when finished
                              (Default = ``True``).
    :param resample_10m_method: Resampling method used for 20 m -> 10 m bands
                                when ``sharpen_10m=False``
                                (Default = :data:`rsgispy.INTERP_CUBIC`).
    :param resample_20m_method: Resampling method used for 10 m -> 20 m band
                                B08 when ``out_res_10m=False``
                                (Default = :data:`rsgispy.INTERP_AVERAGE`).
    :param use_multi_core: If ``True``, use multiple threads during resampling
                           (Default = ``False``).
    :param inc_cloudsen12_img: If ``True``, also output a 12-band image
                               (``<scene>_<res>_cloudsen12_sref``) suitable
                               as input to the CloudSEN12 SREF cloud masking
                               (``apply_cloudsen12_s2_sref_cloud_mask`` in
                               :mod:`rsgispy.imagecalibration.cloudmasking`).
                               Bands B01 and B09 are resampled to the output
                               resolution using ``resample_10m_method`` (and
                               are not sharpened). The same ``scale_factor``,
                               BOA offsets, ``datatype`` and no-data value as
                               the main stack are used, so pass the same
                               ``scale_factor`` (and ``add_offset=0``) to the
                               cloud masking function (Default = ``False``).
    :param calc_stats: If ``True``, populate image statistics and pyramids
                       (Default = ``True``).
    :param use_pbar: If ``True``, display tqdm progress bars (Default = ``True``).
    :param creation_opts: Optional GDAL creation options (dict or sequence of
                          ``'KEY=VALUE'``).

    .. code:: python

        import rsgispy.imagecalibration.sensorlvl2data as l2

        l2.create_stacked_sen2_esa_l2a_img(
            "S2A_MSIL2A_20200621T112119.SAFE",
            "outputs",
            "tmp",
            gdalformat="GTIFF",
        )

    To also create the input for the CloudSEN12 cloud masking:

    .. code:: python

        import rsgispy.imagecalibration.cloudmasking as cldmsk
        import rsgispy.imagecalibration.sensorlvl2data as l2

        l2.create_stacked_sen2_esa_l2a_img(
            "S2A_MSIL2A_20200621T112119.SAFE",
            "outputs",
            "tmp",
            inc_cloudsen12_img=True,
        )
        cldmsk.apply_cloudsen12_s2_sref_cloud_mask(
            "outputs/S2A_MSIL2A_20200621T112119_10m_cloudsen12_sref.tif",
            "outputs/S2A_MSIL2A_20200621T112119_10m_cloudsen12.tif",
            scale_factor=10000,
            device="auto",
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    if not os.path.isdir(input_safe):
        raise rsgispy.RSGISPyException(
            f"Input Sentinel-2 SAFE directory does not exist: '{input_safe}'."
        )
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

    uid_tmp_dir = os.path.join(tmp_dir_abs, f"sen2_{uid_generator(8)}")
    os.makedirs(uid_tmp_dir, exist_ok=True)

    try:
        input_safe_abs = os.path.abspath(input_safe)
        direct_mtd = os.path.join(input_safe_abs, "MTD_MSIL2A.xml")
        mtd_header_file = (
            direct_mtd
            if os.path.isfile(direct_mtd)
            else find_first_file(input_safe_abs, "MTD_MSIL2A.xml", rtn_except=False)
        )
        if mtd_header_file is None:
            raise rsgispy.RSGISPyException(
                "Could not find the Sentinel-2 MTD_MSIL2A.xml header."
            )

        s2_hdr_info = read_sen2_l2a_mtd_to_dict(mtd_header_file)
        granules = s2_hdr_info["Product_Info"]["Product_Organisation"][
            "Granule_List"
        ]
        if len(granules) != 1:
            raise rsgispy.RSGISPyException(
                f"Expected a single granule in '{mtd_header_file}', "
                f"found {len(granules)}."
            )

        prod_uri = s2_hdr_info["Product_Info"].get("PRODUCT_URI", "")
        scn_file_name = (
            prod_uri.replace(".SAFE", "")
            if prod_uri
            else get_file_basename(input_safe_abs)
        )

        img_char = s2_hdr_info["Product_Image_Characteristics"]
        no_data_val = float(img_char["Special_Values"].get("NODATA", 0.0))
        quant_boa_val = float(
            img_char["Quantification_values"]["BOA_QUANTIFICATION_VALUE"]
        )
        if quant_boa_val <= 0.0:
            raise rsgispy.RSGISPyException(
                f"BOA_QUANTIFICATION_VALUE must be > 0, got {quant_boa_val}."
            )

        boa_offsets_by_id = img_char.get("BOA_Add_Offsets", {})
        phys_band_lut = s2_hdr_info.get("Band_LUT", {}).get("Physical_Band", {})
        offsets_by_band: Dict[str, float] = {}
        for bcode in _SEN2_CLDSEN12_BAND_ORDER:
            # Physical band key in S2 MTD may be 'B8' for 'B08'
            b_id = phys_band_lut.get(bcode)
            if b_id is None and bcode.startswith("B0"):
                b_id = phys_band_lut.get(f"B{bcode[2:]}")
            offsets_by_band[bcode] = (
                float(boa_offsets_by_id.get(b_id, 0.0))
                if b_id is not None
                else 0.0
            )
        band_offsets: List[float] = [
            offsets_by_band[b] for b in _SEN2_OUT_BAND_ORDER
        ]

        imgs_10m, all_20m_imgs, scl_20m_img, coarse_imgs = (
            _collect_sen2_band_paths(input_safe_abs, granules[0]["IMAGE_FILES"])
        )
        if inc_cloudsen12_img:
            missing_coarse = [b for b, p in coarse_imgs.items() if p is None]
            if missing_coarse:
                raise rsgispy.RSGISPyException(
                    f"Missing Sentinel-2 band file(s) {missing_coarse} required "
                    "for the CloudSEN12 stack (inc_cloudsen12_img=True)."
                )
        if scl_20m_img is None:
            raise rsgispy.RSGISPyException(
                "Could not find Sentinel-2 20 m SCL image in MTD_MSIL2A.xml."
            )

        res_tag = "10m" if out_res_10m else "20m"
        out_sref_img = os.path.join(
            out_dir_abs, f"{scn_file_name}_{res_tag}_sref.{out_img_ext}"
        )
        out_scl_img = os.path.join(
            out_dir_abs, f"{scn_file_name}_{res_tag}_scl.{out_img_ext}"
        )
        out_clouds_img = os.path.join(
            out_dir_abs, f"{scn_file_name}_{res_tag}_clouds.{out_img_ext}"
        )
        out_vld_img = os.path.join(
            out_dir_abs, f"{scn_file_name}_{res_tag}_vld.{out_img_ext}"
        )
        out_cldsen12_img = os.path.join(
            out_dir_abs, f"{scn_file_name}_{res_tag}_cloudsen12_sref.{out_img_ext}"
        )
        tmp_sref_img = os.path.join(
            uid_tmp_dir, f"{scn_file_name}_{res_tag}_raw_sref.tif"
        )

        if out_res_10m:
            missing_10m = [b for b, p in imgs_10m.items() if p is None]
            comp_20m_bands = ("B05", "B06", "B07", "B8A", "B11", "B12")
            missing_20m = [b for b in comp_20m_bands if all_20m_imgs[b] is None]
            if missing_10m or missing_20m:
                raise rsgispy.RSGISPyException(
                    f"Missing Sentinel-2 band file(s): 10m={missing_10m}, "
                    f"20m={missing_20m}."
                )

            ref_img = imgs_10m["B02"]
            assert ref_img is not None
            band_interp = (
                rsgispy.INTERP_NEAREST_NEIGHBOUR
                if sharpen_10m
                else resample_10m_method
            )
            stacked_10m: Dict[str, str] = {
                b: p for b, p in imgs_10m.items() if p is not None
            }
            for bcode in comp_20m_bands:
                src_20m = all_20m_imgs[bcode]
                assert src_20m is not None
                bname_10m = get_file_basename(src_20m).replace("20m", "10m")
                out_10m = os.path.join(uid_tmp_dir, f"{bname_10m}.tif")
                resample_img_to_match(
                    ref_img,
                    src_20m,
                    out_10m,
                    interp_method=band_interp,
                    multicore=use_multi_core,
                    gdalformat="GTIFF",
                    datatype=rsgispy.TYPE_32FLOAT,
                    no_data_val=no_data_val,
                    calc_stats=False,
                    use_pbar=False,
                )
                stacked_10m[bcode] = out_10m

            input_imgs = [stacked_10m[b] for b in _SEN2_OUT_BAND_ORDER]
            if sharpen_10m:
                tmp_vrt_img = os.path.join(
                    uid_tmp_dir, f"{scn_file_name}_tmp_stck.vrt"
                )
                create_stack_images_vrt(input_imgs, tmp_vrt_img)
                high_res_bands = {"B02", "B03", "B04", "B08"}
                band_shp_info = [
                    SharpBandInfo(
                        band=idx,
                        status=(
                            rsgispy.SHARP_RES_HIGH
                            if bcode in high_res_bands
                            else rsgispy.SHARP_RES_LOW
                        ),
                        name=bname,
                    )
                    for idx, (bcode, bname) in enumerate(
                        zip(_SEN2_OUT_BAND_ORDER, _SEN2_OUT_BAND_NAMES),
                        start=1,
                    )
                ]
                sharpen_low_res_bands(
                    tmp_vrt_img,
                    tmp_sref_img,
                    band_info=band_shp_info,
                    win_size=7,
                    gdalformat="GTIFF",
                    datatype=rsgispy.TYPE_32FLOAT,
                    no_data_val=no_data_val,
                    calc_stats=False,
                    use_pbar=use_pbar,
                )
            else:
                stack_img_bands(
                    input_imgs,
                    tmp_sref_img,
                    band_names=list(_SEN2_OUT_BAND_NAMES),
                    skip_value=no_data_val,
                    gdalformat="GTIFF",
                    datatype=rsgispy.TYPE_32FLOAT,
                    no_data_val=no_data_val,
                    calc_stats=False,
                    use_pbar=use_pbar,
                )

            tmp_scl_10m = os.path.join(
                uid_tmp_dir, f"{scn_file_name}_10m_scl_tmp.tif"
            )
            resample_img_to_match(
                ref_img,
                scl_20m_img,
                tmp_scl_10m,
                interp_method=rsgispy.INTERP_NEAREST_NEIGHBOUR,
                multicore=use_multi_core,
                gdalformat="GTIFF",
                datatype=rsgispy.TYPE_8UINT,
                no_data_val=0,
                calc_stats=False,
                use_pbar=False,
            )
            _write_sen2_scl_derivatives(
                scl_src_img=tmp_scl_10m,
                out_scl_img=out_scl_img,
                out_clouds_img=out_clouds_img,
                out_vld_img=out_vld_img,
                gdalformat=gdalformat,
                calc_stats=calc_stats,
                use_pbar=use_pbar,
                creation_opts=norm_opts,
            )
        else:
            missing_20m = [b for b, p in all_20m_imgs.items() if p is None]
            if imgs_10m["B08"] is None or missing_20m:
                raise rsgispy.RSGISPyException(
                    f"Missing Sentinel-2 band file(s): B08={imgs_10m['B08']}, "
                    f"20m={missing_20m}."
                )
            ref_img = all_20m_imgs["B02"]
            assert ref_img is not None
            b08_img = imgs_10m["B08"]
            assert b08_img is not None

            b08_name_20m = get_file_basename(b08_img).replace("10m", "20m")
            out_b08_20m_img = os.path.join(uid_tmp_dir, f"{b08_name_20m}.tif")
            resample_img_to_match(
                ref_img,
                b08_img,
                out_b08_20m_img,
                interp_method=resample_20m_method,
                multicore=use_multi_core,
                gdalformat="GTIFF",
                datatype=rsgispy.TYPE_32FLOAT,
                no_data_val=no_data_val,
                calc_stats=False,
                use_pbar=False,
            )
            stacked_20m: Dict[str, str] = {
                b: p for b, p in all_20m_imgs.items() if p is not None
            }
            stacked_20m["B08"] = out_b08_20m_img
            input_imgs = [stacked_20m[b] for b in _SEN2_OUT_BAND_ORDER]

            stack_img_bands(
                input_imgs,
                tmp_sref_img,
                band_names=list(_SEN2_OUT_BAND_NAMES),
                skip_value=no_data_val,
                gdalformat="GTIFF",
                datatype=rsgispy.TYPE_32FLOAT,
                no_data_val=no_data_val,
                calc_stats=False,
                use_pbar=use_pbar,
            )
            _write_sen2_scl_derivatives(
                scl_src_img=scl_20m_img,
                out_scl_img=out_scl_img,
                out_clouds_img=out_clouds_img,
                out_vld_img=out_vld_img,
                gdalformat=gdalformat,
                calc_stats=calc_stats,
                use_pbar=use_pbar,
                creation_opts=norm_opts,
            )

        _finalise_sen2_sref_output(
            band_sources=[
                (tmp_sref_img, bidx)
                for bidx in range(1, len(_SEN2_OUT_BAND_ORDER) + 1)
            ],
            out_sref_img=out_sref_img,
            band_offsets=band_offsets,
            band_names=_SEN2_OUT_BAND_NAMES,
            quant_boa_val=quant_boa_val,
            scale_factor=sf_val,
            no_data_val=no_data_val,
            gdalformat=gdalformat,
            datatype=datatype,
            calc_stats=calc_stats,
            use_pbar=use_pbar,
            creation_opts=norm_opts,
        )

        if inc_cloudsen12_img:
            coarse_srcs: Dict[str, str] = {}
            for bcode, coarse_img in coarse_imgs.items():
                assert coarse_img is not None
                out_coarse = os.path.join(
                    uid_tmp_dir, f"{scn_file_name}_{bcode}_{res_tag}.tif"
                )
                resample_img_to_match(
                    ref_img,
                    coarse_img,
                    out_coarse,
                    interp_method=resample_10m_method,
                    multicore=use_multi_core,
                    gdalformat="GTIFF",
                    datatype=rsgispy.TYPE_32FLOAT,
                    no_data_val=no_data_val,
                    calc_stats=False,
                    use_pbar=False,
                )
                coarse_srcs[bcode] = out_coarse

            cldsen12_sources: List[Tuple[str, int]] = []
            for bcode in _SEN2_CLDSEN12_BAND_ORDER:
                if bcode in coarse_srcs:
                    cldsen12_sources.append((coarse_srcs[bcode], 1))
                else:
                    cldsen12_sources.append(
                        (tmp_sref_img, _SEN2_OUT_BAND_ORDER.index(bcode) + 1)
                    )
            _finalise_sen2_sref_output(
                band_sources=cldsen12_sources,
                out_sref_img=out_cldsen12_img,
                band_offsets=[
                    offsets_by_band[b] for b in _SEN2_CLDSEN12_BAND_ORDER
                ],
                band_names=_SEN2_CLDSEN12_BAND_NAMES,
                quant_boa_val=quant_boa_val,
                scale_factor=sf_val,
                no_data_val=no_data_val,
                gdalformat=gdalformat,
                datatype=datatype,
                calc_stats=calc_stats,
                use_pbar=use_pbar,
                creation_opts=norm_opts,
            )
    finally:
        if delete_inter_data and os.path.exists(uid_tmp_dir):
            shutil.rmtree(uid_tmp_dir, ignore_errors=True)
