"""
The imageutils module contains general image utility functions, including
functions for configuring GDAL/rasterio image creation options via environment
variables and populating image statistics, colour tables, and pyramids.
"""

# pylint: disable=too-many-lines
from contextlib import ExitStack
import math
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import xml.etree.ElementTree as ET

import numpy as np
import rasterio
import rasterio.crs
from rasterio.enums import ColorInterp, Resampling
from rasterio.transform import Affine
from rasterio.vrt import _gdal_typename
from rasterio.warp import reproject as rio_reproject
from rasterio.warp import transform as rio_transform_coords
from rasterio.windows import Window

import rsgispy
from rsgispy.imagecalc import calc_basic_img_band_stats
from rsgispy.imageprocess import (
    _INT_DTYPES,
    IntersectionGrid,
    get_default_block_size,
    get_gdal_img_creation_opts,
    iter_single_image_band_blocks,
    open_output_dataset,
    reset_default_block_size,
    set_default_block_size,
)

__all__ = [
    "get_default_block_size",
    "set_default_block_size",
    "reset_default_block_size",
    "get_gdal_img_creation_opts",
    "set_img_creation_opts",
    "set_env_vars_lzw_gtiff_outs",
    "set_env_vars_deflate_gtiff_outs",
    "set_env_vars_zstd_gtiff_outs",
    "set_env_vars_lzw_cog_outs",
    "set_env_vars_deflate_cog_outs",
    "set_img_thematic",
    "build_img_pyramids",
    "pop_thmt_img_stats",
    "pop_img_stats",
    "create_blank_img",
    "create_copy_img",
    "create_blank_buf_img_from_ref_img",
    "create_blank_img_from_bbox",
    "resample_img_to_match",
    "reproject_image",
    "create_stack_images_vrt",
    "create_mosaic_images_vrt",
    "create_vrt_band_subset",
]


def set_img_creation_opts(
    gdalformat: str, creation_opts: Mapping[str, Any]
) -> None:
    """
    Set the environment variables ``RSGISLIB_IMG_CRT_OPTS_<GDAL_FORMAT>`` and
    ``RSGISPY_IMG_CRT_OPTS_<GDAL_FORMAT>`` from a mapping of creation options.

    :param gdalformat: GDAL format string (e.g., 'GTIFF', 'COG').
    :param creation_opts: Mapping of GDAL creation option keys and values.
    """
    fmt_upper = gdalformat.upper().strip()
    opts_str = ":".join(f"{k}={v}" for k, v in creation_opts.items())
    os.environ[f"RSGISLIB_IMG_CRT_OPTS_{fmt_upper}"] = opts_str
    os.environ[f"RSGISPY_IMG_CRT_OPTS_{fmt_upper}"] = opts_str


def set_env_vars_lzw_gtiff_outs(bigtiff: bool = False) -> None:
    """
    Set environment variables such that outputted GeoTIFF files are created
    tiled and compressed using the LZW algorithm.

    :param bigtiff: If True, GTIFF files will be created as BigTIFF (``BIGTIFF=YES``),
                    otherwise ``BIGTIFF=IF_SAFER`` (Default = False).

    .. code:: python

        import rsgispy.imageutils
        rsgispy.imageutils.set_env_vars_lzw_gtiff_outs()
    """
    bigtiff_val = "YES" if bigtiff else "IF_SAFER"
    opts_str = f"TILED=YES:COMPRESS=LZW:INTERLEAVE=BAND:BIGTIFF={bigtiff_val}"
    os.environ["RSGISLIB_IMG_CRT_OPTS_GTIFF"] = opts_str
    os.environ["RSGISPY_IMG_CRT_OPTS_GTIFF"] = opts_str


def set_env_vars_deflate_gtiff_outs(bigtiff: bool = False) -> None:
    """
    Set environment variables such that outputted GeoTIFF files are created
    tiled and compressed using the Deflate (zlib) algorithm.

    :param bigtiff: If True, GTIFF files will be created as BigTIFF (``BIGTIFF=YES``),
                    otherwise ``BIGTIFF=IF_SAFER`` (Default = False).

    .. code:: python

        import rsgispy.imageutils
        rsgispy.imageutils.set_env_vars_deflate_gtiff_outs()
    """
    bigtiff_val = "YES" if bigtiff else "IF_SAFER"
    opts_str = (
        f"TILED=YES:COMPRESS=DEFLATE:INTERLEAVE=BAND:BIGTIFF={bigtiff_val}"
    )
    os.environ["RSGISLIB_IMG_CRT_OPTS_GTIFF"] = opts_str
    os.environ["RSGISPY_IMG_CRT_OPTS_GTIFF"] = opts_str


def set_env_vars_zstd_gtiff_outs(bigtiff: bool = False) -> None:
    """
    Set environment variables such that outputted GeoTIFF files are created
    tiled and compressed using the ZSTD algorithm.

    :param bigtiff: If True, GTIFF files will be created as BigTIFF (``BIGTIFF=YES``),
                    otherwise ``BIGTIFF=IF_SAFER`` (Default = False).
    """
    bigtiff_val = "YES" if bigtiff else "IF_SAFER"
    opts_str = f"TILED=YES:COMPRESS=ZSTD:INTERLEAVE=BAND:BIGTIFF={bigtiff_val}"
    os.environ["RSGISLIB_IMG_CRT_OPTS_GTIFF"] = opts_str
    os.environ["RSGISPY_IMG_CRT_OPTS_GTIFF"] = opts_str


def set_env_vars_lzw_cog_outs(bigtiff: bool = False) -> None:
    """
    Set environment variables such that outputted Cloud Optimized GeoTIFF (COG)
    files are compressed using the LZW algorithm.

    :param bigtiff: If True, COG files will be created as BigTIFF (``BIGTIFF=YES``),
                    otherwise ``BIGTIFF=IF_SAFER`` (Default = False).
    """
    bigtiff_val = "YES" if bigtiff else "IF_SAFER"
    opts_str = f"COMPRESS=LZW:BIGTIFF={bigtiff_val}"
    os.environ["RSGISLIB_IMG_CRT_OPTS_COG"] = opts_str
    os.environ["RSGISPY_IMG_CRT_OPTS_COG"] = opts_str


def set_env_vars_deflate_cog_outs(bigtiff: bool = False) -> None:
    """
    Set environment variables such that outputted Cloud Optimized GeoTIFF (COG)
    files are compressed using the Deflate algorithm.

    :param bigtiff: If True, COG files will be created as BigTIFF (``BIGTIFF=YES``),
                    otherwise ``BIGTIFF=IF_SAFER`` (Default = False).
    """
    bigtiff_val = "YES" if bigtiff else "IF_SAFER"
    opts_str = f"COMPRESS=DEFLATE:BIGTIFF={bigtiff_val}"
    os.environ["RSGISLIB_IMG_CRT_OPTS_COG"] = opts_str
    os.environ["RSGISPY_IMG_CRT_OPTS_COG"] = opts_str


def set_img_thematic(input_img: str) -> None:
    """
    Set all image bands in ``input_img`` to have ``LAYER_TYPE=thematic`` metadata.

    :param input_img: Path to the image file to update.
    """
    if not os.path.exists(input_img):
        raise rsgispy.RSGISPyException(
            f"Input image file does not exist: '{input_img}'."
        )
    try:
        with rasterio.open(input_img, "r+") as dst:
            for bidx in range(1, dst.count + 1):
                dst.update_tags(bidx, LAYER_TYPE="thematic")
    except Exception as err:
        raise rsgispy.RSGISPyException(
            f"Could not update thematic metadata on '{input_img}': {err}"
        ) from err


def build_img_pyramids(
    input_img: str,
    resampling_method: Resampling = Resampling.average,
    ignore_cog: bool = False,
) -> None:
    """
    Build overview pyramids on ``input_img`` using ``rasterio``.

    :param input_img: Path to the image file to update.
    :param resampling_method: Overview resampling algorithm (Default = Resampling.average).
    :param ignore_cog: If True, pass ``IGNORE_COG_LAYOUT_BREAK=YES`` when opening.
    """
    open_opts: Dict[str, Any] = {}
    if ignore_cog:
        open_opts["IGNORE_COG_LAYOUT_BREAK"] = "YES"

    with rasterio.open(input_img, "r+", **open_opts) as dst:
        min_size = min(dst.width, dst.height)
        pyd_lvls = [
            lvl
            for lvl in (4, 8, 16, 32, 64, 128, 256, 512)
            if (min_size // lvl) > 33
        ]
        if not pyd_lvls:
            return

        if dst.driver and dst.driver.upper() == "GTIFF":
            with rasterio.Env(COMPRESS_OVERVIEW="LZW"):
                dst.build_overviews(pyd_lvls, resampling_method)
        else:
            dst.build_overviews(pyd_lvls, resampling_method)


def _calc_int_band_hists(
    input_img: str, dtype_str: str, use_pbar: bool = True
) -> List[Tuple[np.ndarray, int, int]]:
    """
    Accumulate direct integer histograms ``(hist, hist_min, hist_max)`` across
    blocks for an integer image (``uint8``, ``uint16``, ``int16``, etc.).
    """
    # pylint: disable=too-many-locals
    with rasterio.open(input_img, "r") as src:
        n_bands = src.count

    if dtype_str == "uint8":
        bounds = [(0, 255) for _ in range(n_bands)]
    else:
        band_mins = [0 for _ in range(n_bands)]
        band_maxs = [0 for _ in range(n_bands)]
        first_seen = [True for _ in range(n_bands)]
        for _, block_arrays in iter_single_image_band_blocks(
            input_img=input_img,
            read_dtype=np.int64,
            use_pbar=use_pbar,
        ):
            for idx, band_arr in enumerate(block_arrays):
                if band_arr.size == 0:
                    continue
                b_min = int(band_arr.min())
                b_max = int(band_arr.max())
                if first_seen[idx]:
                    band_mins[idx] = min(0, b_min)
                    band_maxs[idx] = max(0, b_max)
                    first_seen[idx] = False
                else:
                    band_mins[idx] = min(band_mins[idx], b_min)
                    band_maxs[idx] = max(band_maxs[idx], b_max)
        bounds = [
            (
                band_mins[idx],
                min(band_maxs[idx], band_mins[idx] + 65535),
            )
            for idx in range(n_bands)
        ]

    hists = [
        np.zeros((h_max - h_min) + 1, dtype=np.int64)
        for h_min, h_max in bounds
    ]
    for _, block_arrays in iter_single_image_band_blocks(
        input_img=input_img,
        read_dtype=np.int64,
        use_pbar=use_pbar,
    ):
        for idx, band_arr in enumerate(block_arrays):
            h_min, h_max = bounds[idx]
            n_bins = (h_max - h_min) + 1
            shifted = band_arr.ravel() - h_min
            valid = shifted[(shifted >= 0) & (shifted < n_bins)]
            if valid.size > 0:
                hists[idx] += np.bincount(valid, minlength=n_bins)

    return [
        (hists[idx], bounds[idx][0], bounds[idx][1]) for idx in range(n_bands)
    ]


def _stats_from_int_hist(
    hist: np.ndarray, hist_min: int, hist_max: int, ignore_zero: bool
) -> Tuple[int, int, float, float, int, int]:
    """
    Compute ``(min, max, mean, stddev, mode, median)`` from a direct integer
    histogram covering ``[hist_min, hist_max]``.
    """
    # pylint: disable=too-many-locals
    vals = np.arange(hist_min, hist_max + 1, dtype=np.float64)
    stat_counts = hist.copy()
    if ignore_zero and hist_min <= 0 <= hist_max:
        stat_counts[0 - hist_min] = 0

    total_valid = int(stat_counts.sum())
    if total_valid == 0:
        min_val = 0
        max_val = 0
        mean_val = 0.0
        stddev_val = 0.0
    else:
        nz_indices = np.nonzero(stat_counts)[0]
        min_val = int(vals[nz_indices[0]])
        max_val = int(vals[nz_indices[-1]])
        mean_val = float(np.sum(vals * stat_counts) / total_valid)
        variance = float(
            np.sum(((vals - mean_val) ** 2) * stat_counts) / total_valid
        )
        stddev_val = float(np.sqrt(variance))

    mode_val = int(vals[int(np.argmax(hist))])
    if int(hist.sum()) > 0:
        mid_num = hist.sum() / 2.0
        median_idx = int(np.nonzero(hist.cumsum() >= mid_num)[0][0])
        median_val = int(vals[median_idx])
    else:
        median_val = 0

    return min_val, max_val, mean_val, stddev_val, mode_val, median_val


def pop_thmt_img_stats(
    input_img: str,
    add_clr_tab: bool = True,
    calc_pyramids: bool = True,
    ignore_zero: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Populate an integer thematic input image (e.g., ``uint8``, ``uint16``, ``int16``
    classification) with header statistics, a colour table, and overview pyramids
    using ``rasterio``.

    :param input_img: Path to the input thematic image file.
    :param add_clr_tab: Whether to compute statistics and add a colour table (Default = True).
    :param calc_pyramids: Whether to build overview pyramids (Default = True).
    :param ignore_zero: Whether to treat pixel value 0 as no-data (Default = True).
    :param use_pbar: Whether to display a tqdm progress bar (Default = True).

    .. code:: python

        import rsgispy.imageutils
        rsgispy.imageutils.pop_thmt_img_stats("land_cover_cls_img.tif")
    """
    # pylint: disable=too-many-locals
    if not os.path.exists(input_img):
        raise rsgispy.RSGISPyException(
            f"Input image file does not exist: '{input_img}'."
        )

    with rasterio.open(input_img, "r") as src:
        dtype_str = src.dtypes[0]
        if dtype_str not in _INT_DTYPES:
            raise rsgispy.RSGISPyException(
                "This function only supports integer datasets"
            )

    set_img_thematic(input_img)

    if add_clr_tab:
        hist_info = _calc_int_band_hists(
            input_img, dtype_str=dtype_str, use_pbar=use_pbar
        )
        with rasterio.open(input_img, "r+") as dst:
            if ignore_zero:
                dst.nodata = 0

            for idx, (hist, h_min, h_max) in enumerate(hist_info):
                bidx = idx + 1
                n_bins = (h_max - h_min) + 1
                min_v, max_v, mean_v, std_v, mode_v, med_v = (
                    _stats_from_int_hist(hist, h_min, h_max, ignore_zero)
                )
                tags: Dict[str, str] = {
                    "STATISTICS_MINIMUM": f"{min_v}",
                    "STATISTICS_MAXIMUM": f"{max_v}",
                    "STATISTICS_MEAN": f"{mean_v}",
                    "STATISTICS_STDDEV": f"{std_v}",
                    "STATISTICS_SKIPFACTORX": "1",
                    "STATISTICS_SKIPFACTORY": "1",
                    "STATISTICS_HISTOBINFUNCTION": "direct",
                    "STATISTICS_MODE": f"{mode_v}",
                    "STATISTICS_MEDIAN": f"{med_v}",
                    "STATISTICS_HISTOBINVALUES": (
                        "|".join(map(str, hist.tolist())) + "|"
                    ),
                    "STATISTICS_HISTOMIN": f"{h_min}",
                    "STATISTICS_HISTOMAX": f"{h_max}",
                    "STATISTICS_HISTONUMBINS": f"{n_bins}",
                }
                if ignore_zero:
                    tags["STATISTICS_EXCLUDEDVALUES"] = "0"
                dst.update_tags(bidx, **tags)

                n_clr_entries = (
                    256
                    if dtype_str == "uint8"
                    else min(max(h_max, 0) + 1, 65536)
                )
                clr_tab: Dict[int, Tuple[int, int, int, int]] = {}
                for val_i in range(n_clr_entries):
                    if val_i == 0 and ignore_zero:
                        clr_tab[0] = (0, 0, 0, 0)
                    else:
                        rnd_rgb = np.random.randint(1, 255, 3, dtype=np.uint8)
                        clr_tab[val_i] = (
                            int(rnd_rgb[0]),
                            int(rnd_rgb[1]),
                            int(rnd_rgb[2]),
                            255,
                        )
                dst.write_colormap(bidx, clr_tab)

            dst.colorinterp = [ColorInterp.palette] * dst.count

    if calc_pyramids:
        build_img_pyramids(input_img, resampling_method=Resampling.nearest)


def _calc_linear_band_hists(
    input_img: str,
    img_min: np.ndarray,
    img_max: np.ndarray,
    no_data_val: Optional[float],
    use_pbar: bool = True,
) -> List[Tuple[np.ndarray, float, float, float, int]]:
    """
    Compute 256-bin linear histograms across blocks for each band of an image.
    """
    # pylint: disable=too-many-locals
    n_bands = len(img_min)
    hist_params: List[Tuple[float, float, float, float, int]] = []
    hists: List[np.ndarray] = []
    for idx in range(n_bands):
        h_min = float(img_min[idx])
        h_max = float(img_max[idx])
        c_min = h_min
        c_max = h_max
        n_bins = 256
        if c_min == c_max:
            c_max = c_max + 0.5
            n_bins = 1
        hist_params.append((h_min, h_max, c_min, c_max, n_bins))
        hists.append(np.zeros(n_bins, dtype=np.int64))

    for _, block_arrays in iter_single_image_band_blocks(
        input_img=input_img, use_pbar=use_pbar
    ):
        for idx, band_arr in enumerate(block_arrays):
            data = band_arr.ravel()
            if no_data_val is not None:
                data = data[data != no_data_val]
            data = data[np.isfinite(data)]
            if data.size > 0:
                _, _, c_min, c_max, n_bins = hist_params[idx]
                counts, _ = np.histogram(
                    data, bins=n_bins, range=(c_min, c_max)
                )
                hists[idx] += counts

    results: List[Tuple[np.ndarray, float, float, float, int]] = []
    for idx in range(n_bands):
        h_min, h_max, c_min, c_max, n_bins = hist_params[idx]
        h_step = float(c_max - c_min) / n_bins
        results.append((hists[idx], h_min, h_max, h_step, n_bins))

    return results


def pop_img_stats(
    input_img: str,
    no_data_val: Optional[float] = None,
    calc_pyramids: bool = True,
    ignore_cog: bool = False,
    use_approx: bool = False,
    use_pbar: bool = True,
) -> None:
    """
    Calculate and populate band statistics, histogram metadata, and overview
    pyramids on ``input_img`` using ``rasterio`` and :mod:`rsgispy.imagecalc`.

    :param input_img: Path to the input image file.
    :param no_data_val: Optional no-data value to set and exclude from statistics.
    :param calc_pyramids: Whether to build overview pyramids (Default = True).
    :param ignore_cog: Whether to ignore breaking COG layout when modifying a COG
                       file in place (Default = False).
    :param use_approx: Whether to record statistics as approximate in metadata
                       (Default = False).
    :param use_pbar: Whether to display a tqdm progress bar (Default = True).

    .. code:: python

        import rsgispy.imageutils
        rsgispy.imageutils.pop_img_stats("input_img.tif", no_data_val=0.0)
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    use_no_data = no_data_val is not None
    img_min, img_max, img_mean, img_std = calc_basic_img_band_stats(
        input_img=input_img,
        no_data_val=no_data_val,
        use_no_data=use_no_data,
        use_pbar=use_pbar,
    )

    hist_results = _calc_linear_band_hists(
        input_img=input_img,
        img_min=img_min,
        img_max=img_max,
        no_data_val=no_data_val,
        use_pbar=use_pbar,
    )

    open_opts: Dict[str, Any] = {}
    if ignore_cog:
        open_opts["IGNORE_COG_LAYOUT_BREAK"] = "YES"

    float_dtypes = {"float32", "float64"}

    with rasterio.open(input_img, "r+", **open_opts) as dst:
        dst.nodata = no_data_val

        for idx in range(dst.count):
            bidx = idx + 1
            is_float = dst.dtypes[idx] in float_dtypes
            hist, h_min, h_max, h_step, n_bins = hist_results[idx]

            tags: Dict[str, str] = {
                "STATISTICS_MINIMUM": f"{float(img_min[idx])}",
                "STATISTICS_MAXIMUM": f"{float(img_max[idx])}",
                "STATISTICS_MEAN": f"{float(img_mean[idx])}",
                "STATISTICS_STDDEV": f"{float(img_std[idx])}",
                "STATISTICS_HISTOBINVALUES": (
                    "|".join(map(str, hist.tolist())) + "|"
                ),
                "STATISTICS_HISTOMIN": repr(h_min),
                "STATISTICS_HISTOMAX": repr(h_max),
                "STATISTICS_HISTONUMBINS": repr(int(n_bins)),
                "STATISTICS_HISTOBINFUNCTION": "Linear",
            }
            if no_data_val is not None:
                tags["STATISTICS_EXCLUDEDVALUES"] = repr(no_data_val)

            if use_approx:
                tags["STATISTICS_APPROXIMATE"] = "YES"
            else:
                tags["STATISTICS_SKIPFACTORX"] = "1"
                tags["STATISTICS_SKIPFACTORY"] = "1"

            if int(hist.sum()) > 0:
                middle_num = hist.sum() / 2.0
                median_bin = int(np.nonzero(hist.cumsum() >= middle_num)[0][0])
                median_val = median_bin * h_step + h_min
            else:
                median_val = h_min

            mode_bin = int(np.argmax(hist))
            mode_val = mode_bin * h_step + h_min

            if is_float:
                tags["STATISTICS_MEDIAN"] = repr(float(median_val))
                tags["STATISTICS_MODE"] = repr(float(mode_val))
            else:
                tags["STATISTICS_MEDIAN"] = repr(int(round(median_val)))
                tags["STATISTICS_MODE"] = repr(int(round(mode_val)))

            dst.update_tags(bidx, **tags)

    if calc_pyramids:
        build_img_pyramids(
            input_img,
            resampling_method=Resampling.average,
            ignore_cog=ignore_cog,
        )


def _normalise_creation_options(
    options: Optional[Union[Sequence[str], Mapping[str, Any]]],
) -> Optional[Dict[str, Any]]:
    """
    Convert ``options`` given as either a sequence of ``'KEY=VALUE'`` strings
    or a dictionary into a ``Dict[str, Any]`` mapping, or ``None`` if empty.
    """
    if not options:
        return None
    if isinstance(options, Mapping):
        return dict(options) if options else None
    parsed: Dict[str, Any] = {}
    for item in options:
        item_str = str(item).strip()
        if "=" in item_str:
            key, val = item_str.split("=", 1)
            parsed[key.strip()] = val.strip()
    return parsed if parsed else None


def _parse_epsg_crs(epsg_code: Optional[int]) -> Optional[Any]:
    """
    Resolve a ``rasterio.crs.CRS`` from an integer ``epsg_code``.
    """
    # pylint: disable=c-extension-no-member
    if epsg_code is None:
        return None
    try:
        return rasterio.crs.CRS.from_epsg(int(epsg_code))
    except Exception as err:
        raise rsgispy.RSGISPyException(
            f"Could not parse EPSG code '{epsg_code}': {err}"
        ) from err


def _get_epsg_from_dataset(src: Any, input_img: str) -> int:
    """
    Extract the integer EPSG code from an opened rasterio dataset ``src``.
    """
    epsg_code = src.crs.to_epsg() if src.crs is not None else None
    if epsg_code is None:
        raise rsgispy.RSGISPyException(
            f"Could not determine EPSG code from input image '{input_img}'."
        )
    return int(epsg_code)


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
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    if n_bands < 1 or width < 1 or height < 1:
        raise rsgispy.RSGISPyException(
            "n_bands, width, and height must all be >= 1."
        )

    out_rio_dtype = rsgispy.get_rasterio_datatype(datatype)
    out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(datatype))
    crs_obj = _parse_epsg_crs(epsg_code)
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
    creation_opts = _normalise_creation_options(options)

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
        epsg_code = _get_epsg_from_dataset(src, input_img)

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
        epsg_code = _get_epsg_from_dataset(src, input_img)

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
    out_crs = _parse_epsg_crs(out_epsg)
    if out_crs is None:
        raise rsgispy.RSGISPyException(
            "An output EPSG code (out_epsg) must be provided."
        )

    with rasterio.open(input_img, "r") as src:
        in_crs = _parse_epsg_crs(in_epsg) if in_epsg is not None else src.crs
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
