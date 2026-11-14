"""
The imageutils.warp module provides raster resampling and reprojection
functions using ``rasterio.warp.reproject``.
"""
# pylint: disable=too-many-lines

from contextlib import ExitStack
import math
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import rasterio
from rasterio.control import GroundControlPoint
from rasterio.enums import ColorInterp, Resampling
from rasterio.warp import reproject as rio_reproject
from rasterio.warp import transform as rio_transform_coords
from rasterio.windows import Window

import rsgispy
from rsgispy.imageprocess import (
    IntersectionGrid,
    cast_result_block,
    ensure_file_exists,
    iter_dataset_windows,
    open_output_dataset,
    validate_band_index,
)
from rsgispy.imageutils._helpers import parse_epsg_crs
from rsgispy.imageutils._translate import (
    TranslateConfig,
    _parse_crs_input,
    build_translate_config,
    expand_colormap_block,
    has_per_dataset_mask,
    resolve_output_grid,
    resolve_scale_specs,
    resolve_src_window,
    validate_requested_bands,
)
from rsgispy.imageutils._warp_opts import (
    WarpConfig,
    WarpGrid,
    apply_cutline_to_dataset,
    build_transformer_and_warp_kwargs,
    build_warp_config,
    resolve_warp_grid,
)
from rsgispy.imageutils.blank import create_blank_img, create_copy_img
from rsgispy.imageutils.stats import pop_img_stats

__all__ = [
    "resample_img_to_match",
    "reproject_image",
    "gdal_warp",
    "gdal_translate",
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

    .. code:: python

        import rsgispy
        import rsgispy.imageutils.warp
        rsgispy.imageutils.warp.resample_img_to_match(
            "ref_img.tif",
            "in_img.tif",
            "resampled_img.tif",
            "GTIFF",
            interp_method=rsgispy.INTERP_BILINEAR,
        )
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

    .. code:: python

        import rsgispy
        import rsgispy.imageutils.warp
        rsgispy.imageutils.warp.reproject_image(
            "input_img.tif",
            "reproj_img.tif",
            out_epsg=32630,
            interp_method=rsgispy.INTERP_BILINEAR,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-statements
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


def _resolve_warp_bands_and_alpha(
    first_src: rasterio.io.DatasetReader,
    cfg: WarpConfig,
) -> Tuple[List[int], List[int], int, int]:
    """
    Resolve ``(src_data_bands, dst_data_bands, src_alpha_band, dst_alpha_band)``.
    """
    # pylint: disable=too-many-branches
    has_alpha_ci = ColorInterp.alpha in first_src.colorinterp
    auto_alpha_idx = (
        first_src.colorinterp.index(ColorInterp.alpha) + 1
        if has_alpha_ci
        else 0
    )

    if cfg.src_alpha is True:
        src_alpha_band = auto_alpha_idx if auto_alpha_idx else first_src.count
    elif cfg.src_alpha is False:
        src_alpha_band = 0
    else:
        src_alpha_band = auto_alpha_idx if cfg.src_bands is None else 0

    if cfg.src_bands is not None:
        src_data_bands = list(cfg.src_bands)
        for bidx in src_data_bands:
            validate_band_index(bidx, first_src.count, first_src.name)
    elif src_alpha_band > 0:
        src_data_bands = [
            b for b in range(1, first_src.count + 1) if b != src_alpha_band
        ]
        if not src_data_bands:
            src_data_bands = [1]
            src_alpha_band = 0
    else:
        src_data_bands = list(range(1, first_src.count + 1))

    if cfg.dst_bands is not None:
        if len(cfg.dst_bands) != len(src_data_bands):
            raise rsgispy.RSGISPyException(
                "Number of dst_bands must match number of src_bands."
            )
        dst_data_bands = list(cfg.dst_bands)
        for bidx in dst_data_bands:
            if bidx < 1:
                raise rsgispy.RSGISPyException(
                    f"Destination band index must be >= 1, got {bidx}."
                )
    else:
        dst_data_bands = list(range(1, len(src_data_bands) + 1))

    add_dst_alpha = cfg.dst_alpha or (
        cfg.src_alpha is None and auto_alpha_idx > 0 and cfg.src_bands is None
    )
    dst_alpha_band = (max(dst_data_bands) + 1) if add_dst_alpha else 0
    return src_data_bands, dst_data_bands, src_alpha_band, dst_alpha_band


def _write_warp_metadata(
    dst: rasterio.io.DatasetWriter,
    src_datasets: Sequence[rasterio.io.DatasetReader],
    src_data_bands: Sequence[int],
    dst_data_bands: Sequence[int],
    dst_alpha_band: int,
    cfg: WarpConfig,
) -> None:
    """Copy metadata tags, band descriptions, and colorinterp to ``dst``."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-branches,too-many-locals
    first_src = src_datasets[0]
    if not cfg.no_metadata:
        merged_tags: Dict[str, str] = {}
        for s_idx, src in enumerate(src_datasets):
            s_tags = src.tags()
            for key, val in s_tags.items():
                if s_idx == 0 or key not in merged_tags:
                    merged_tags[key] = val
                elif (
                    cfg.meta_conflict_value is not None
                    and merged_tags[key] != val
                ):
                    merged_tags[key] = cfg.meta_conflict_value
        if merged_tags:
            dst.update_tags(**merged_tags)

        for sb, db in zip(src_data_bands, dst_data_bands):
            if db <= dst.count:
                desc = first_src.descriptions[sb - 1]
                if desc:
                    dst.set_band_description(db, desc)
                btags = first_src.tags(sb)
                if btags:
                    dst.update_tags(db, **btags)

    if cfg.set_ci or dst_alpha_band > 0:
        ci_list = list(dst.colorinterp)
        for sb, db in zip(src_data_bands, dst_data_bands):
            if db <= dst.count:
                ci_list[db - 1] = first_src.colorinterp[sb - 1]
        if 0 < dst_alpha_band <= dst.count:
            ci_list[dst_alpha_band - 1] = ColorInterp.alpha
        dst.colorinterp = ci_list


def _resolve_band_nodata(
    nodata_cfg: Optional[Union[str, Tuple[float, ...]]],
    has_cfg: bool,
    fallback_vals: Sequence[Optional[float]],
    n_bands: int,
) -> List[Optional[float]]:
    """Expand scalar or per-band nodata setting to length ``n_bands``."""
    if has_cfg:
        if nodata_cfg == "none":
            return [None] * n_bands
        assert isinstance(nodata_cfg, tuple)
        if len(nodata_cfg) == 1:
            return [float(nodata_cfg[0])] * n_bands
        if len(nodata_cfg) == n_bands:
            return [float(v) for v in nodata_cfg]
        raise rsgispy.RSGISPyException(
            f"Expected 1 or {n_bands} nodata values, got {len(nodata_cfg)}."
        )
    return [
        float(v) if v is not None else None for v in fallback_vals[:n_bands]
    ]


def _reproject_single_source(
    src: rasterio.io.DatasetReader,
    dst: rasterio.io.DatasetWriter,
    s_crs: Optional[Any],
    gref: Dict[str, Any],
    dst_crs: Optional[Any],
    src_data_bands: Sequence[int],
    dst_data_bands: Sequence[int],
    src_alpha_band: int,
    dst_alpha_band: int,
    init_dest: bool,
    cfg: WarpConfig,
    extra_kw: Dict[str, Any],
) -> None:
    """Warp a single opened source dataset onto ``dst``."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-branches
    src_nd_list = _resolve_band_nodata(
        cfg.src_nodata,
        cfg.has_src_nodata,
        [src.nodatavals[b - 1] for b in src_data_bands],
        len(src_data_bands),
    )
    dst_fallback = (
        list(dst.nodatavals)
        if any(v is not None for v in dst.nodatavals)
        else src_nd_list
    )
    dst_nd_list = _resolve_band_nodata(
        cfg.dst_nodata,
        cfg.has_dst_nodata,
        [dst_fallback[min(i, len(dst_fallback) - 1)] for i in range(len(dst_data_bands))],
        len(dst_data_bands),
    )

    reproj_kw: Dict[str, Any] = dict(extra_kw)
    if "gcps" in gref:
        reproj_kw["gcps"] = gref["gcps"]
    elif "rpcs" in gref:
        reproj_kw["rpcs"] = gref["rpcs"]
    if s_crs is not None:
        reproj_kw["src_crs"] = s_crs
    if dst_crs is not None:
        reproj_kw["dst_crs"] = dst_crs

    ignore_src_nd = cfg.has_src_nodata and cfg.src_nodata == "none"
    uniform_nd = (len(set(src_nd_list)) <= 1) and (len(set(dst_nd_list)) <= 1)

    if ignore_src_nd:
        src_arr = src.read(list(src_data_bands))
        arr_kw = dict(reproj_kw)
        if "gcps" not in arr_kw and "rpcs" not in arr_kw:
            arr_kw["src_transform"] = src.transform
        d_nd = dst_nd_list[0]
        if dst_alpha_band > 0:
            arr_kw["dst_alpha"] = dst_alpha_band
        rio_reproject(
            source=src_arr,
            destination=rasterio.band(dst, list(dst_data_bands)),
            src_nodata=None,
            dst_nodata=d_nd,
            resampling=cfg.resampling,
            num_threads=cfg.num_threads,
            init_dest_nodata=init_dest,
            tolerance=cfg.err_threshold,
            warp_mem_limit=cfg.warp_mem_limit,
            **arr_kw,
        )
        return

    if uniform_nd:
        b_kw = dict(reproj_kw)
        if src_alpha_band > 0:
            b_kw["src_alpha"] = src_alpha_band
        if dst_alpha_band > 0:
            b_kw["dst_alpha"] = dst_alpha_band
        rio_reproject(
            source=rasterio.band(src, list(src_data_bands)),
            destination=rasterio.band(dst, list(dst_data_bands)),
            src_nodata=src_nd_list[0],
            dst_nodata=dst_nd_list[0],
            resampling=cfg.resampling,
            num_threads=cfg.num_threads,
            init_dest_nodata=init_dest,
            tolerance=cfg.err_threshold,
            warp_mem_limit=cfg.warp_mem_limit,
            **b_kw,
        )
        return

    for idx, (sb, db, s_nd, d_nd) in enumerate(
        zip(src_data_bands, dst_data_bands, src_nd_list, dst_nd_list)
    ):
        b_kw = dict(reproj_kw)
        if src_alpha_band > 0:
            b_kw["src_alpha"] = src_alpha_band
        if dst_alpha_band > 0 and idx == 0:
            b_kw["dst_alpha"] = dst_alpha_band
        rio_reproject(
            source=rasterio.band(src, [sb]),
            destination=rasterio.band(dst, [db]),
            src_nodata=s_nd,
            dst_nodata=d_nd,
            resampling=cfg.resampling,
            num_threads=cfg.num_threads,
            init_dest_nodata=init_dest,
            tolerance=cfg.err_threshold,
            warp_mem_limit=cfg.warp_mem_limit,
            **b_kw,
        )


def gdal_warp(
    input_img: Union[str, os.PathLike, Sequence[Union[str, os.PathLike]]],
    output_img: Union[str, os.PathLike],
    out_epsg: Optional[Union[int, str, Any]] = None,
    interp_method: Union[int, Resampling, str] = rsgispy.INTERP_NEAREST_NEIGHBOUR,
    gdalformat: str = "GTIFF",
    use_multi_threaded: bool = True,
    options: Optional[Union[str, Sequence[str], Mapping[str, Any]]] = None,
    *,
    in_format: Optional[str] = None,
    datatype: Optional[Union[int, str]] = None,
    working_datatype: Optional[Union[int, str]] = None,
    overwrite: Optional[bool] = None,
    src_srs: Optional[Union[int, str, Any]] = None,
    t_srs: Optional[Union[int, str, Any]] = None,
    src_coord_epoch: Optional[float] = None,
    dst_coord_epoch: Optional[float] = None,
    coord_transform: Optional[str] = None,
    target_res: Optional[Union[str, Tuple[float, float]]] = None,
    target_size: Optional[Tuple[int, int]] = None,
    target_extent: Optional[Tuple[float, float, float, float]] = None,
    target_extent_srs: Optional[Union[int, str, Any]] = None,
    tap: bool = False,
    src_nodata: Optional[Union[float, int, str, Sequence[float]]] = None,
    dst_nodata: Optional[Union[float, int, str, Sequence[float]]] = None,
    src_bands: Optional[Sequence[int]] = None,
    dst_bands: Optional[Sequence[int]] = None,
    src_alpha: Optional[bool] = None,
    dst_alpha: bool = False,
    ovr: Optional[Union[int, str]] = None,
    err_threshold: Optional[float] = None,
    warp_mem_limit: Optional[int] = None,
    tps: bool = False,
    rpc: bool = False,
    geoloc: bool = False,
    order: Optional[int] = None,
    refine_gcps: Optional[
        Union[float, Tuple[float, Optional[int]]]
    ] = None,
    warp_options: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
    transformer_options: Optional[
        Union[Sequence[str], Mapping[str, Any]]
    ] = None,
    vshift: Optional[bool] = None,
    cutline: Optional[Any] = None,
    cutline_srs: Optional[Union[int, str, Any]] = None,
    cutline_layer: Optional[str] = None,
    cutline_where: Optional[str] = None,
    cutline_sql: Optional[str] = None,
    cutline_blend: Optional[float] = None,
    crop_to_cutline: bool = False,
    no_metadata: bool = False,
    meta_conflict_value: Optional[str] = None,
    set_ci: bool = False,
    open_options: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
    dst_open_options: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
    use_pbar: bool = False,
) -> None:
    """
    Reproject, resample, mosaic, and/or clip one or more input rasters using
    ``rasterio.warp.reproject`` with ``gdalwarp`` CLI option support.

    CLI flags from ``gdalwarp`` are accepted both as Python keyword arguments
    and inside a CLI string or token sequence passed to ``options``.

    :param input_img: Path (or sequence of paths) to input raster file(s).
    :param output_img: Path to the output warped raster file.
    :param out_epsg: Target EPSG integer code or CRS definition (optional if
                     ``t_srs`` is given or keeping source CRS).
    :param interp_method: Resampling method (``-r``): ``rsgispy.INTERP_*``,
                          ``Resampling``, or CLI string (Default = nearest).
    :param gdalformat: Output GDAL format name (``-of``, Default = ``'GTIFF'``).
    :param use_multi_threaded: Use multiple CPU threads (``-multi``, Default = ``True``).
    :param options: Creation options (dict/list) or ``gdalwarp`` CLI option
                    string (e.g., ``'-t_srs EPSG:4326 -tr 0.001 0.001 -tap'``).
    :param in_format: Optional input driver name (``-if``).
    :param datatype: Output data type ``rsgispy.TYPE_*`` or GDAL name (``-ot``).
    :param working_datatype: Intermediate working data type (``-wt``).
    :param overwrite: Overwrite existing output file (``-overwrite``, Default = ``True``;
                      set ``False`` to warp into an existing destination raster).
    :param src_srs: Override source CRS (``-s_srs``).
    :param t_srs: Target CRS definition (``-t_srs``), overrides ``out_epsg``.
    :param src_coord_epoch: Source coordinate epoch (``-s_coord_epoch``).
    :param dst_coord_epoch: Target coordinate epoch (``-t_coord_epoch``).
    :param coord_transform: Explicit PROJ pipeline transformation string (``-ct``).
    :param target_res: Target resolution ``(xres, yres)`` or ``'square'`` (``-tr``).
    :param target_size: Target output dimensions ``(width, height)`` with optional
                        ``0`` to preserve aspect ratio (``-ts``).
    :param target_extent: Target extent ``(xmin, ymin, xmax, ymax)`` (``-te``).
    :param target_extent_srs: CRS of ``target_extent`` coordinates (``-te_srs``).
    :param tap: Align target extent coordinates to multiples of resolution (``-tap``).
    :param src_nodata: Source nodata value(s) or ``'none'`` (``-srcnodata``).
    :param dst_nodata: Destination nodata value(s) or ``'none'`` (``-dstnodata``).
    :param src_bands: 1-based source band indices to warp (``-srcband``).
    :param dst_bands: 1-based destination band indices (``-dstband``).
    :param src_alpha: Use/force source alpha band (``True`` = ``-srcalpha``,
                      ``False`` = ``-nosrcalpha``, ``None`` = auto).
    :param dst_alpha: Create an output alpha band (``-dstalpha``).
    :param ovr: Source overview level selection (``-ovr``).
    :param err_threshold: Transformer error threshold in pixels (``-et``).
    :param warp_mem_limit: Warp memory limit in MB (``-wm``).
    :param tps: Use thin-plate spline GCP transformer (``-tps``).
    :param rpc: Use RPC transformer (``-rpc``).
    :param geoloc: Use geolocation arrays transformer (``-geoloc``).
    :param order: GCP polynomial order 1, 2, or 3 (``-order``).
    :param refine_gcps: Tolerance (and optional min GCPs) for outlier removal
                        (``-refine_gcps``).
    :param warp_options: Extra GDAL warp options ``NAME=VALUE`` (``-wo``).
    :param transformer_options: Extra GDAL transformer options (``-to``).
    :param vshift: Enable or disable vertical datum grid shift (``-vshift`` /
                   ``-novshiftgrid``).
    :param cutline: Cutline vector path, WKT/GeoJSON string, or shapely/GeoPandas
                    geometry (``-cutline``).
    :param cutline_srs: Override CRS of ``cutline`` geometries (``-cutline_srs``).
    :param cutline_layer: Layer name in cutline datasource (``-cl``).
    :param cutline_where: Attribute WHERE filter for cutline (``-cwhere``).
    :param cutline_sql: SQL query for cutline datasource (``-csql``).
    :param cutline_blend: Feathering blend distance in pixels inside cutline
                          (``-cblend``).
    :param crop_to_cutline: Crop output extent to cutline bounding box
                            (``-crop_to_cutline``).
    :param no_metadata: Do not copy metadata from source datasets (``-nomd``).
    :param meta_conflict_value: Replacement value for conflicting metadata across
                                multiple inputs (``-cvmd``).
    :param set_ci: Set output color interpretation from source bands (``-setci``).
    :param open_options: Source dataset open options (``-oo``).
    :param dst_open_options: Existing destination open options (``-doo``).
    :param use_pbar: Display a tqdm progress bar during cutline masking.

    .. code:: python

        import rsgispy.imageutils.warp
        rsgispy.imageutils.warp.gdal_warp(
            "input_img.tif",
            "warped_img.tif",
            out_epsg=4326,
            target_res=(0.001, 0.001),
            tap=True,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    if isinstance(input_img, (str, bytes, os.PathLike)):
        in_paths = [os.fspath(input_img)]
    else:
        in_paths = [os.fspath(p) for p in input_img]
    if not in_paths:
        raise rsgispy.RSGISPyException("At least one input image is required.")
    for p in in_paths:
        ensure_file_exists(p)

    out_path = os.fspath(output_img)
    cfg = build_warp_config(
        out_epsg=out_epsg,
        interp_method=interp_method,
        gdalformat=gdalformat,
        use_multi_threaded=use_multi_threaded,
        options=options,
        in_format=in_format,
        datatype=datatype,
        working_datatype=working_datatype,
        overwrite=overwrite,
        src_srs=src_srs,
        t_srs=t_srs,
        src_coord_epoch=src_coord_epoch,
        dst_coord_epoch=dst_coord_epoch,
        coord_transform=coord_transform,
        target_res=target_res,
        target_size=target_size,
        target_extent=target_extent,
        target_extent_srs=target_extent_srs,
        tap=tap,
        src_nodata=src_nodata,
        dst_nodata=dst_nodata,
        src_bands=src_bands,
        dst_bands=dst_bands,
        src_alpha=src_alpha,
        dst_alpha=dst_alpha,
        ovr=ovr,
        err_threshold=err_threshold,
        warp_mem_limit=warp_mem_limit,
        tps=tps,
        rpc=rpc,
        geoloc=geoloc,
        order=order,
        refine_gcps=refine_gcps,
        warp_options=warp_options,
        transformer_options=transformer_options,
        vshift=vshift,
        cutline=cutline,
        cutline_srs=cutline_srs,
        cutline_layer=cutline_layer,
        cutline_where=cutline_where,
        cutline_sql=cutline_sql,
        cutline_blend=cutline_blend,
        crop_to_cutline=crop_to_cutline,
        no_metadata=no_metadata,
        meta_conflict_value=meta_conflict_value,
        set_ci=set_ci,
        open_options=open_options,
        dst_open_options=dst_open_options,
        use_pbar=use_pbar,
    )

    open_kwargs: Dict[str, Any] = dict(cfg.open_opts)
    if cfg.in_format is not None:
        open_kwargs["driver"] = cfg.in_format
    if cfg.ovr is not None and "OVERVIEW_LEVEL" not in open_kwargs:
        open_kwargs["OVERVIEW_LEVEL"] = cfg.ovr

    extra_kw = build_transformer_and_warp_kwargs(cfg)
    cutline_geoms: Optional[List[Any]] = None
    cutline_fill: List[float] = [0.0]
    dst_alpha_band = 0

    with ExitStack() as stack:
        src_datasets = [
            stack.enter_context(rasterio.open(p, "r", **open_kwargs))
            for p in in_paths
        ]
        first_src = src_datasets[0]
        (
            src_data_bands,
            dst_data_bands,
            src_alpha_band,
            dst_alpha_band,
        ) = _resolve_warp_bands_and_alpha(first_src, cfg)

        grid_res: WarpGrid = resolve_warp_grid(src_datasets, cfg, extra_kw)
        dst_crs = grid_res.dst_crs
        cutline_geoms = grid_res.cutline_geoms

        update_existing = (not cfg.overwrite) and os.path.exists(out_path)
        if update_existing:
            dst = stack.enter_context(
                rasterio.open(out_path, "r+", **cfg.dst_open_opts)
            )
            dst_crs = dst.crs
            out_nd_vals = [
                float(v) if v is not None else 0.0 for v in dst.nodatavals
            ]
            init_first = False
        else:
            out_dtype = (
                cfg.datatype
                if cfg.datatype is not None
                else rsgispy.get_rsgis_datatype(
                    first_src.dtypes[src_data_bands[0] - 1]
                )
            )
            out_rio_dtype = rsgispy.get_rasterio_datatype(out_dtype)
            out_count = max(*dst_data_bands, dst_alpha_band)
            out_grid = IntersectionGrid(
                width=grid_res.width,
                height=grid_res.height,
                transform=grid_res.transform,
                crs=dst_crs,
                offsets={},
            )
            dst = open_output_dataset(
                stack=stack,
                output_img=out_path,
                gdalformat=cfg.gdalformat,
                out_rio_dtype=out_rio_dtype,
                grid=out_grid,
                count=out_count,
                creation_opts=cfg.creation_opts if cfg.creation_opts else None,
            )
            dst_nd_fallback = (
                cfg.src_nodata
                if cfg.has_src_nodata and cfg.src_nodata != "none"
                else (
                    float(first_src.nodata)
                    if first_src.nodata is not None
                    else None
                )
            )
            if cfg.has_dst_nodata:
                if cfg.dst_nodata != "none":
                    assert isinstance(cfg.dst_nodata, tuple)
                    dst.nodata = float(cfg.dst_nodata[0])
            elif dst_nd_fallback is not None:
                val = (
                    dst_nd_fallback[0]
                    if isinstance(dst_nd_fallback, tuple)
                    else dst_nd_fallback
                )
                dst.nodata = float(val)

            _write_warp_metadata(
                dst=dst,
                src_datasets=src_datasets,
                src_data_bands=src_data_bands,
                dst_data_bands=dst_data_bands,
                dst_alpha_band=dst_alpha_band,
                cfg=cfg,
            )
            out_nd_vals = [
                float(v) if v is not None else 0.0 for v in dst.nodatavals
            ]
            init_first = True

        cutline_fill = out_nd_vals if out_nd_vals else [0.0]

        for s_idx, (src, s_crs, gref) in enumerate(
            zip(src_datasets, grid_res.src_crs_list, grid_res.src_georef_list)
        ):
            _reproject_single_source(
                src=src,
                dst=dst,
                s_crs=s_crs,
                gref=gref,
                dst_crs=dst_crs,
                src_data_bands=src_data_bands,
                dst_data_bands=dst_data_bands,
                src_alpha_band=src_alpha_band,
                dst_alpha_band=dst_alpha_band,
                init_dest=init_first and (s_idx == 0),
                cfg=cfg,
                extra_kw=extra_kw,
            )

    if cutline_geoms is not None:
        with rasterio.open(out_path, "r+", **cfg.dst_open_opts) as dst_cut:
            apply_cutline_to_dataset(
                dst=dst_cut,
                cutline_geoms=cutline_geoms,
                fill_values=cutline_fill,
                dst_alpha_band=dst_alpha_band,
                blend_dist=cfg.cutline_blend,
                use_pbar=cfg.use_pbar,
            )


gdalwarp = gdal_warp


def _apply_scale_and_unscale(
    block: np.ndarray,
    src: rasterio.io.DatasetReader,
    src_bands: Sequence[int],
    cfg: TranslateConfig,
    scale_specs: Optional[List[Optional[Tuple[float, float, float, float, float]]]],
    out_nodata: Optional[float],
) -> np.ndarray:
    """Apply ``-unscale``, ``-scale``, and ``-exponent`` per band preserving nodata."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-branches
    out_block = block.astype(np.float64, copy=True)
    for pos, bidx in enumerate(src_bands):
        band_arr = out_block[pos]
        b_src_nd = src.nodatavals[bidx - 1]
        if b_src_nd is not None:
            if math.isnan(float(b_src_nd)):
                nd_mask = np.isnan(band_arr)
            else:
                nd_mask = band_arr == float(b_src_nd)
        elif np.issubdtype(block.dtype, np.floating):
            nd_mask = np.isnan(band_arr)
        else:
            nd_mask = None

        if cfg.unscale:
            s_val = float(src.scales[bidx - 1])
            o_val = float(src.offsets[bidx - 1])
            if s_val != 1.0 or o_val != 0.0:
                band_arr = (band_arr * s_val) + o_val

        if scale_specs is not None and scale_specs[pos] is not None:
            smin, smax, dmin, dmax, exp_val = scale_specs[pos]  # type: ignore[misc]
            if smax == smin:
                band_arr = np.full_like(band_arr, dmin, dtype=np.float64)
            else:
                norm = (band_arr - smin) / (smax - smin)
                if exp_val != 1.0:
                    norm = np.power(np.maximum(norm, 0.0), exp_val)
                band_arr = dmin + (dmax - dmin) * norm

        if nd_mask is not None and np.any(nd_mask):
            fill_nd = (
                float(out_nodata)
                if out_nodata is not None
                else (float(b_src_nd) if b_src_nd is not None else np.nan)
            )
            band_arr[nd_mask] = fill_nd
        out_block[pos] = band_arr
    return out_block


def _read_translate_window(
    src: rasterio.io.DatasetReader,
    read_bands: Sequence[int],
    src_win: Window,
    out_w: int,
    out_h: int,
    out_win: Window,
    resampling: Resampling,
) -> np.ndarray:
    """Read an output block window from ``src``, resampling if output size differs."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    win_w = int(round(src_win.width))
    win_h = int(round(src_win.height))
    c0 = int(round(src_win.col_off))
    r0 = int(round(src_win.row_off))
    ow = int(out_win.width)
    oh = int(out_win.height)
    oc = int(out_win.col_off)
    orow = int(out_win.row_off)

    if win_w == out_w and win_h == out_h:
        r_win = Window(c0 + oc, r0 + orow, ow, oh)
        return src.read(list(read_bands), window=r_win)

    scale_x = float(win_w) / float(out_w)
    scale_y = float(win_h) / float(out_h)
    r_win = Window(
        c0 + (oc * scale_x),
        r0 + (orow * scale_y),
        ow * scale_x,
        oh * scale_y,
    )
    return src.read(
        list(read_bands),
        window=r_win,
        out_shape=(len(read_bands), oh, ow),
        resampling=resampling,
    )


def _write_translate_metadata(
    dst: Any,
    src: rasterio.io.DatasetReader,
    src_bands: Sequence[int],
    out_count: int,
    out_rio_dtype: str,
    out_nodata: Optional[float],
    out_crs: Optional[Any],
    out_w: int,
    out_h: int,
    src_win: Window,
    cfg: TranslateConfig,
) -> None:
    """Apply nodata, tags, band metadata, colorinterp, scales/offsets, and GCPs."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    if out_nodata is not None:
        dst.nodata = out_nodata

    if not cfg.no_metadata:
        tags = dict(src.tags())
        if cfg.no_xmp:
            tags.pop("xml:XMP", None)
        if tags:
            dst.update_tags(**tags)
    if cfg.metadata:
        dst.update_tags(**cfg.metadata)
    for dom, dom_tags in cfg.domain_metadata.items():
        if cfg.no_xmp and dom == "xml:XMP":
            continue
        dst.update_tags(ns=dom, **dom_tags)

    if cfg.expand is None:
        for out_bidx, src_bidx in enumerate(src_bands, start=1):
            desc = src.descriptions[src_bidx - 1]
            if desc:
                dst.set_band_description(out_bidx, desc)
            if not cfg.no_metadata:
                btags = src.tags(src_bidx)
                if btags:
                    dst.update_tags(out_bidx, **btags)
        if out_rio_dtype in ("uint8", "uint16"):
            for out_bidx, src_bidx in enumerate(src_bands, start=1):
                try:
                    cmap = src.colormap(src_bidx)
                    if cmap:
                        dst.write_colormap(out_bidx, cmap)
                except ValueError:
                    pass

    if cfg.color_interp_list is not None:
        if len(cfg.color_interp_list) != out_count:
            raise rsgispy.RSGISPyException(
                f"color_interp has {len(cfg.color_interp_list)} items, "
                f"expected {out_count}."
            )
        dst.colorinterp = list(cfg.color_interp_list)
    else:
        if cfg.expand == "gray":
            ci_list = [ColorInterp.gray]
        elif cfg.expand == "rgb":
            ci_list = [ColorInterp.red, ColorInterp.green, ColorInterp.blue]
        elif cfg.expand == "rgba":
            ci_list = [
                ColorInterp.red,
                ColorInterp.green,
                ColorInterp.blue,
                ColorInterp.alpha,
            ]
        else:
            ci_list = [src.colorinterp[b - 1] for b in src_bands]
        for b_idx, ci_val in cfg.color_interp_per_band.items():
            validate_band_index(b_idx, out_count, "output")
            ci_list[b_idx - 1] = ci_val
        dst.colorinterp = ci_list

    if cfg.band_scale is not None:
        if isinstance(cfg.band_scale, (int, float, np.number)):
            dst.scales = tuple([float(cfg.band_scale)] * out_count)
        else:
            if len(cfg.band_scale) != out_count:
                raise rsgispy.RSGISPyException(
                    "band_scale sequence length must match output band count."
                )
            dst.scales = tuple(float(v) for v in cfg.band_scale)
    elif not cfg.unscale and cfg.expand is None:
        src_scales = tuple(float(src.scales[b - 1]) for b in src_bands)
        if any(s != 1.0 for s in src_scales):
            dst.scales = src_scales

    if cfg.band_offset is not None:
        if isinstance(cfg.band_offset, (int, float, np.number)):
            dst.offsets = tuple([float(cfg.band_offset)] * out_count)
        else:
            if len(cfg.band_offset) != out_count:
                raise rsgispy.RSGISPyException(
                    "band_offset sequence length must match output band count."
                )
            dst.offsets = tuple(float(v) for v in cfg.band_offset)
    elif not cfg.unscale and cfg.expand is None:
        src_offsets = tuple(float(src.offsets[b - 1]) for b in src_bands)
        if any(o != 0.0 for o in src_offsets):
            dst.offsets = src_offsets

    if cfg.gcps:
        gcp_crs = out_crs if out_crs is not None else src.gcps[1]
        dst.gcps = (cfg.gcps, gcp_crs)
    elif not cfg.no_gcp and cfg.ullr is None and cfg.geotransform is None:
        src_gcps, src_gcp_crs = src.gcps
        if src_gcps:
            win_w = int(round(src_win.width))
            win_h = int(round(src_win.height))
            sx = float(out_w) / float(win_w)
            sy = float(out_h) / float(win_h)
            adj_gcps = [
                GroundControlPoint(
                    row=(g.row - float(src_win.row_off)) * sy,
                    col=(g.col - float(src_win.col_off)) * sx,
                    x=g.x,
                    y=g.y,
                    z=g.z,
                    id=g.id,
                    info=g.info,
                )
                for g in src_gcps
            ]
            dst.gcps = (adj_gcps, out_crs if cfg.has_out_srs else src_gcp_crs)


def gdal_translate(
    input_img: str,
    output_img: str,
    gdalformat: str = "GTIFF",
    datatype: Optional[Union[int, str]] = None,
    options: Union[str, Sequence[str], Mapping[str, Any]] = "",
    *,
    in_format: Optional[str] = None,
    bands: Optional[Sequence[int]] = None,
    src_win: Optional[Sequence[Union[int, float]]] = None,
    proj_win: Optional[Sequence[float]] = None,
    proj_win_srs: Optional[Union[int, str, Any]] = None,
    error_partially_outside: bool = False,
    error_completely_outside: bool = False,
    out_size: Optional[
        Tuple[Union[int, float, str], Union[int, float, str]]
    ] = None,
    target_res: Optional[Tuple[float, float]] = None,
    ovr: Optional[Union[int, str]] = None,
    resample_alg: Union[int, Resampling, str] = rsgispy.INTERP_NEAREST_NEIGHBOUR,
    out_srs: Optional[Union[int, str, Any]] = None,
    ullr: Optional[Tuple[float, float, float, float]] = None,
    geotransform: Optional[Sequence[float]] = None,
    no_data_val: Optional[Union[float, int, str]] = None,
    unscale: bool = False,
    scale_params: Optional[
        Union[bool, Sequence[Union[float, Sequence[float]]]]
    ] = None,
    exponent: Optional[Union[float, Sequence[float]]] = None,
    band_scale: Optional[Union[float, Sequence[float]]] = None,
    band_offset: Optional[Union[float, Sequence[float]]] = None,
    expand: Optional[str] = None,
    color_interp: Optional[
        Union[
            str,
            ColorInterp,
            Sequence[Union[str, int, ColorInterp]],
            Mapping[int, Union[str, int, ColorInterp]],
        ]
    ] = None,
    mask_band: Optional[Union[int, str]] = None,
    no_gcp: bool = False,
    gcps: Optional[Sequence[Sequence[float]]] = None,
    stats: bool = False,
    approx_stats: bool = False,
    metadata: Optional[Union[Sequence[str], Mapping[str, str]]] = None,
    domain_metadata: Optional[Mapping[str, Mapping[str, str]]] = None,
    no_metadata: bool = False,
    no_xmp: bool = False,
    open_options: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
    strict: bool = False,
    use_pbar: bool = True,
) -> None:
    """
    Convert ``input_img`` to another GDAL/rasterio format, subset, resample,
    rescale, and/or update spatial and band metadata without ``osgeo.gdal``.

    When ``gdalformat='GTIFF'`` and no creation options are provided via
    ``options``, tiled LZW BigTIFF creation options are applied automatically.
    CLI flags from ``gdal_translate`` are supported both as Python keyword
    arguments and inside a CLI string/list passed to ``options``.

    :param input_img: Path to the input raster image file.
    :param output_img: Path to the output raster image file.
    :param gdalformat: Output GDAL format name (``-of``, Default = ``'GTIFF'``).
    :param datatype: Optional ``rsgispy.TYPE_*`` constant or GDAL type string
                     (``-ot``, e.g., ``'Float32'``, ``'Byte'``).
    :param options: Creation options or ``gdal_translate`` CLI option string /
                    sequence / dict (e.g., ``'-co TILED=YES -b 1 -stats'``).
    :param in_format: Optional input driver name (``-if``).
    :param bands: 1-based band indices to select/reorder (``-b``).
    :param src_win: Pixel subset window ``(xoff, yoff, xsize, ysize)`` (``-srcwin``).
    :param proj_win: Projected subset window ``(ulx, uly, lrx, lry)`` (``-projwin``).
    :param proj_win_srs: CRS for ``proj_win`` if different from source CRS
                         (``-projwin_srs``).
    :param error_partially_outside: Raise if window extends outside raster (``-epo``).
    :param error_completely_outside: Raise if window is outside raster (``-eco``).
    :param out_size: Output dimensions ``(xsize, ysize)`` as ints, ``'N%'`` strings,
                     or ``0`` to preserve aspect ratio (``-outsize``).
    :param target_res: Target pixel resolution ``(xres, yres)`` (``-tr``).
    :param ovr: Source overview level selection (``-ovr``, e.g., ``0``, ``'AUTO'``,
                ``'NONE'``).
    :param resample_alg: Resampling algorithm (``-r``): ``rsgispy.INTERP_*``,
                         ``Resampling``, or string (Default = nearest).
    :param out_srs: Assign/override output CRS (``-a_srs``), or ``'none'`` to clear.
    :param ullr: Assign georeferenced bounds ``(ulx, uly, lrx, lry)`` (``-a_ullr``).
    :param geotransform: Assign 6-element GDAL geotransform (``-a_gt``).
    :param no_data_val: Assign output nodata value, or ``'none'`` to unset
                        (``-a_nodata``).
    :param unscale: Apply source band scale and offset to pixel values (``-unscale``).
    :param scale_params: Linear rescaling ``[src_min, src_max]``,
                         ``[src_min, src_max, dst_min, dst_max]``, per-band lists,
                         or ``True``/``[]`` for auto min/max (``-scale``).
    :param exponent: Power exponent(s) for non-linear ``scale_params`` (``-exponent``).
    :param band_scale: Assign output band scale metadata (``-a_scale``).
    :param band_offset: Assign output band offset metadata (``-a_offset``).
    :param expand: Expand single-band indexed colormap to ``'gray'``, ``'rgb'``,
                   or ``'rgba'`` (``-expand``).
    :param color_interp: Assign band color interpretation(s) (``-colorinterp``).
    :param mask_band: Output per-dataset mask: ``'none'``, ``'auto'``, or 1-based
                      source band index (``-mask``).
    :param no_gcp: Drop source GCPs (``-nogcp``).
    :param gcps: Sequence of GCP tuples ``(pixel, line, easting, northing[, z])``
                 (``-gcp``).
    :param stats: Compute and write band statistics tags (``-stats``).
    :param approx_stats: Write statistics marked as approximate (``-approx_stats``).
    :param metadata: Dataset metadata tags ``KEY=VALUE`` or dict (``-mo``).
    :param domain_metadata: Domain-namespaced tags ``{domain: {key: val}}`` (``-dmo``).
    :param no_metadata: Strip source dataset/band metadata tags (``-nomd``).
    :param no_xmp: Strip ``xml:XMP`` metadata domain (``-noxmp``).
    :param open_options: Input dataset open options (``-oo``).
    :param strict: Raise on integer range overflow when casting (``-strict``).
    :param use_pbar: Display a tqdm progress bar (Default = True).

    .. code:: python

        import rsgispy.imageutils.warp
        rsgispy.imageutils.warp.gdal_translate(
            "input_img.kea",
            "output_img.tif",
            gdalformat="GTIFF",
            bands=[3, 2, 1],
            src_win=(0, 0, 512, 512),
            stats=True,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    ensure_file_exists(input_img)
    cfg = build_translate_config(
        gdalformat=gdalformat,
        datatype=datatype,
        options=options,
        in_format=in_format,
        bands=bands,
        src_win=src_win,
        proj_win=proj_win,
        proj_win_srs=proj_win_srs,
        error_partially_outside=error_partially_outside,
        error_completely_outside=error_completely_outside,
        out_size=out_size,
        target_res=target_res,
        ovr=ovr,
        resample_alg=resample_alg,
        out_srs=out_srs,
        ullr=ullr,
        geotransform=geotransform,
        no_data_val=no_data_val,
        unscale=unscale,
        scale_params=scale_params,
        exponent=exponent,
        band_scale=band_scale,
        band_offset=band_offset,
        expand=expand,
        color_interp=color_interp,
        mask_band=mask_band,
        no_gcp=no_gcp,
        gcps=gcps,
        stats=stats,
        approx_stats=approx_stats,
        metadata=metadata,
        domain_metadata=domain_metadata,
        no_metadata=no_metadata,
        no_xmp=no_xmp,
        open_options=open_options,
        strict=strict,
        use_pbar=use_pbar,
    )

    copy_overviews = (
        str(cfg.creation_opts.pop("COPY_SRC_OVERVIEWS", "NO")).upper() == "YES"
    )
    open_kwargs: Dict[str, Any] = dict(cfg.open_opts)
    if cfg.in_format is not None:
        open_kwargs["driver"] = cfg.in_format
    if cfg.ovr is not None and "OVERVIEW_LEVEL" not in open_kwargs:
        open_kwargs["OVERVIEW_LEVEL"] = cfg.ovr

    with ExitStack() as stack:
        src = stack.enter_context(rasterio.open(input_img, "r", **open_kwargs))
        src_bands = validate_requested_bands(src, cfg, input_img)
        src_win_res = resolve_src_window(src, cfg)
        out_w, out_h, out_t = resolve_output_grid(src_win_res, src.transform, cfg)
        out_crs = _parse_crs_input(cfg.out_srs) if cfg.has_out_srs else src.crs

        cmap_expand: Optional[Dict[int, Tuple[int, int, int, int]]] = None
        if cfg.expand is not None:
            if len(src_bands) != 1:
                raise rsgispy.RSGISPyException(
                    "-expand requires a single input band with a color table."
                )
            try:
                cmap_expand = src.colormap(src_bands[0])
            except ValueError as err:
                raise rsgispy.RSGISPyException(
                    "Input band has no color table for -expand."
                ) from err
            out_count = {"gray": 1, "rgb": 3, "rgba": 4}[cfg.expand]
        else:
            out_count = len(src_bands)

        if cfg.datatype is not None:
            out_dtype = cfg.datatype
        elif cfg.expand is not None:
            out_dtype = rsgispy.TYPE_8UINT
        elif cfg.unscale and any(
            src.scales[b - 1] != 1.0 or src.offsets[b - 1] != 0.0
            for b in src_bands
        ):
            out_dtype = rsgispy.TYPE_64FLOAT
        else:
            out_dtype = rsgispy.get_rsgis_datatype(src.dtypes[src_bands[0] - 1])

        out_rio_dtype = rsgispy.get_rasterio_datatype(out_dtype)
        out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(out_dtype))

        if cfg.has_no_data:
            if cfg.no_data_val == "none":
                out_nodata: Optional[float] = None
            else:
                out_nodata = float(cfg.no_data_val)  # type: ignore[arg-type]
        else:
            out_nodata = float(src.nodata) if src.nodata is not None else None

        scale_specs = resolve_scale_specs(src, src_bands, src_win_res, cfg)

        if isinstance(cfg.mask_band, int):
            validate_band_index(cfg.mask_band, src.count, input_img)

        out_grid = IntersectionGrid(
            width=out_w,
            height=out_h,
            transform=out_t,
            crs=out_crs,
            offsets={},
        )
        dst = open_output_dataset(
            stack=stack,
            output_img=output_img,
            gdalformat=cfg.gdalformat,
            out_rio_dtype=out_rio_dtype,
            grid=out_grid,
            count=out_count,
            creation_opts=cfg.creation_opts if cfg.creation_opts else None,
        )

        _write_translate_metadata(
            dst=dst,
            src=src,
            src_bands=src_bands,
            out_count=out_count,
            out_rio_dtype=out_rio_dtype,
            out_nodata=out_nodata,
            out_crs=out_crs,
            out_w=out_w,
            out_h=out_h,
            src_win=src_win_res,
            cfg=cfg,
        )

        read_resampling = (
            Resampling.nearest if cfg.expand is not None else cfg.resampling
        )
        do_math = cfg.unscale or (scale_specs is not None)
        is_int_out = np.issubdtype(out_np_dtype, np.integer)
        int_info = np.iinfo(out_np_dtype) if is_int_out else None

        for out_win in iter_dataset_windows(
            out_h, out_w, use_pbar=cfg.use_pbar
        ):
            data = _read_translate_window(
                src=src,
                read_bands=src_bands,
                src_win=src_win_res,
                out_w=out_w,
                out_h=out_h,
                out_win=out_win,
                resampling=read_resampling,
            )
            if do_math:
                data = _apply_scale_and_unscale(
                    data, src, src_bands, cfg, scale_specs, out_nodata
                )
            if cfg.expand is not None and cmap_expand is not None:
                data = expand_colormap_block(data, cmap_expand, cfg.expand)
            if int_info is not None:
                if do_math:
                    data = np.rint(data)
                if cfg.strict and np.any(
                    (data < int_info.min) | (data > int_info.max)
                ):
                    raise rsgispy.RSGISPyException(
                        "Output pixel values exceed target integer data type "
                        "range (-strict)."
                    )
                if do_math:
                    data = np.clip(data, int_info.min, int_info.max)
            dst.write(
                cast_result_block(
                    data, int(out_win.height), int(out_win.width), out_np_dtype
                ),
                window=out_win,
            )

            if cfg.mask_band in (None, "auto") and has_per_dataset_mask(src):
                win_w = int(round(src_win_res.width))
                win_h = int(round(src_win_res.height))
                c0 = int(round(src_win_res.col_off))
                r0 = int(round(src_win_res.row_off))
                if win_w == out_w and win_h == out_h:
                    mw = Window(
                        c0 + int(out_win.col_off),
                        r0 + int(out_win.row_off),
                        int(out_win.width),
                        int(out_win.height),
                    )
                    m_arr = src.dataset_mask(window=mw)
                else:
                    sx = float(win_w) / float(out_w)
                    sy = float(win_h) / float(out_h)
                    mw = Window(
                        c0 + (int(out_win.col_off) * sx),
                        r0 + (int(out_win.row_off) * sy),
                        int(out_win.width) * sx,
                        int(out_win.height) * sy,
                    )
                    m_arr = src.dataset_mask(
                        window=mw,
                        out_shape=(int(out_win.height), int(out_win.width)),
                    )
                dst.write_mask(m_arr, window=out_win)
            elif isinstance(cfg.mask_band, int):
                m_blk = _read_translate_window(
                    src=src,
                    read_bands=[cfg.mask_band],
                    src_win=src_win_res,
                    out_w=out_w,
                    out_h=out_h,
                    out_win=out_win,
                    resampling=Resampling.nearest,
                )[0]
                m_u8 = np.where(m_blk != 0, np.uint8(255), np.uint8(0))
                dst.write_mask(m_u8, window=out_win)

        if copy_overviews and src.overviews(src_bands[0]):
            valid_ovrs = [
                lvl
                for lvl in src.overviews(src_bands[0])
                if lvl < min(out_w, out_h)
            ]
            if valid_ovrs:
                dst.build_overviews(valid_ovrs, cfg.resampling)

    if cfg.stats or cfg.approx_stats:
        pop_img_stats(
            output_img,
            no_data_val=out_nodata,
            calc_pyramids=False,
            use_approx=cfg.approx_stats,
            use_pbar=False,
        )
