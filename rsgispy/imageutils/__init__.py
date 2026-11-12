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

import numpy as np
import rasterio
import rasterio.crs
from rasterio.enums import ColorInterp, Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject as rio_reproject
from rasterio.warp import transform as rio_transform_coords
from rasterio.windows import Window

import rsgispy
from rsgispy.imagecalc import calc_basic_img_band_stats
from rsgispy.imageprocess import (
    DEFAULT_BLOCK_SIZE,
    IntersectionGrid,
    get_gdal_img_creation_opts,
    iter_single_image_band_blocks,
    open_output_dataset,
)

__all__ = [
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


_INT_DTYPES = {"uint8", "int8", "uint16", "int16", "uint32", "int32"}


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
    base_extent: Sequence[float],
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

        for row_off in range(0, grid.height, DEFAULT_BLOCK_SIZE):
            win_h = min(DEFAULT_BLOCK_SIZE, grid.height - row_off)
            for col_off in range(0, grid.width, DEFAULT_BLOCK_SIZE):
                win_w = min(DEFAULT_BLOCK_SIZE, grid.width - col_off)
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
    bbox: Sequence[float],
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
