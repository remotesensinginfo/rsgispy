"""
The imageutils module contains general image utility functions, including
functions for configuring GDAL/rasterio image creation options via environment
variables and populating image statistics, colour tables, and pyramids.
"""

import os
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import rasterio
from rasterio.enums import ColorInterp, Resampling

import rsgispy
from rsgispy.imagecalc import calc_basic_img_band_stats
from rsgispy.imageprocess import (
    get_gdal_img_creation_opts,
    iter_single_image_band_blocks,
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
