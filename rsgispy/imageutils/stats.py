"""
The imageutils.stats module provides functions for building raster overview
pyramids and populating thematic and continuous image statistics and histograms
using ``rasterio``.
"""

import math
import multiprocessing
import os
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import rasterio
from rasterio.enums import ColorInterp, Resampling
from rasterio.transform import Affine

import rsgispy
from rsgispy.imagecalc.stats import calc_basic_img_band_stats
from rsgispy.imageprocess import (
    _INT_DTYPES,
    ensure_file_exists,
    iter_single_image_band_blocks,
)
from rsgispy.imageutils._overviews import (
    finalise_external_ovr,
    finalise_internal_ovr,
    normalise_ovr_resampling,
    render_ovr_tile_worker,
    stitch_scale_tiles,
    uniform_nodata,
    validate_positive_int,
)
from rsgispy.imageutils.colourtable import set_img_thematic

__all__ = [
    "build_img_pyramids",
    "build_img_pyramids_multi_core",
    "pop_thmt_img_stats",
    "pop_img_stats",
]


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

    .. code:: python

        import rsgispy.imageutils.stats
        rsgispy.imageutils.stats.build_img_pyramids("input_img.tif")
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


def build_img_pyramids_multi_core(
    input_img: Union[str, os.PathLike],
    resampling_method: Union[int, str, Resampling] = Resampling.average,
    levels: Optional[Sequence[int]] = None,
    tile_size: int = 1024,
    n_cores: int = 1,
    external: bool = True,
    output_img: Optional[Union[str, os.PathLike]] = None,
    tmp_dir: Optional[Union[str, os.PathLike]] = None,
    compress: str = "LZW",
    ovr_block_size: int = 128,
    cascade: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Build image overview pyramids using multiple CPU cores by tiling each
    overview scale into temporary GeoTIFF tiles and combining them at the end.

    Designed for very large raster datasets: at each overview scale, the output
    overview grid is partitioned into ``tile_size x tile_size`` tiles rendered
    in parallel across ``n_cores`` processes into temporary files, combined per
    scale, and finally assembled into either an external GDAL ``.ovr`` sidecar
    file (``external=True``, default — avoids rewriting the huge base raster)
    or internal GeoTIFF overviews (``external=False``).

    :param input_img: Path to the input image file.
    :param resampling_method: Overview resampling algorithm as a
                              ``rasterio.enums.Resampling`` enum,
                              ``rsgispy.INTERP_*`` integer, or string name
                              (Default = ``Resampling.average``).
    :param levels: Optional sequence of integer decimation factors ``>= 2``
                   (e.g., ``[2, 4, 8, 16]``). When ``None`` (default), uses
                   ``[4, 8, 16, 32, 64, 128, 256, 512]`` filtered to
                   ``(min(width, height) // lvl) > 33``.
    :param tile_size: Overview tile width and height in pixels at each scale
                      (Default = ``1024``).
    :param n_cores: Number of worker processes (Default = ``1``).
    :param external: If ``True`` (default), write an external GDAL ``.ovr``
                     overview file (``<input_img>.ovr``) without modifying
                     ``input_img``. If ``False``, embed internal GeoTIFF
                     overviews into ``output_img`` (or replace ``input_img``
                     atomically when ``output_img=None``).
    :param output_img: Optional output path. When ``external=True``, overrides
                       the default ``<input_img>.ovr`` path; when
                       ``external=False``, writes a new GeoTIFF with internal
                       overviews instead of replacing ``input_img``.
    :param tmp_dir: Optional scratch directory for temporary scale/tile files
                    (e.g., fast local SSD). Automatically cleaned up on exit.
    :param compress: Compression algorithm for overview TIFFs (Default = ``'LZW'``).
    :param ovr_block_size: Internal TIFF tile block size (power of 2 between
                           ``64`` and ``4096``, Default = ``128``).
    :param cascade: If ``True`` (default), generate scale ``lvl`` from the
                    largest previously built divisor scale (``lvl % prev == 0``)
                    for faster hierarchical downsampling; if ``False``, sample
                    every scale directly from ``input_img``.
    :param use_pbar: Display a tqdm progress bar per overview scale
                     (Default = ``True``).

    .. code:: python

        from rasterio.enums import Resampling
        import rsgispy.imageutils.stats as rstats

        rstats.build_img_pyramids_multi_core(
            "large_mosaic.tif",
            resampling_method=Resampling.average,
            tile_size=1024,
            n_cores=8,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    input_path = os.fspath(input_img)
    ensure_file_exists(input_path, label="Input image")

    resamp_name = normalise_ovr_resampling(resampling_method)
    t_size = validate_positive_int(tile_size, "tile_size", min_val=1)
    eff_cores = validate_positive_int(n_cores, "n_cores", min_val=1)
    ovr_blk = validate_positive_int(ovr_block_size, "ovr_block_size", min_val=64)
    if ovr_blk > 4096 or (ovr_blk & (ovr_blk - 1)) != 0:
        raise rsgispy.RSGISPyException(
            "ovr_block_size must be a power of 2 between 64 and 4096, "
            f"got {ovr_blk}."
        )
    comp_val = str(compress).strip().upper() if compress else "NONE"

    with rasterio.open(input_path, "r") as src:
        base_w = int(src.width)
        base_h = int(src.height)
        n_bands = int(src.count)
        dtypes = list(src.dtypes)
        nodatavals = list(src.nodatavals)
        crs = src.crs
        base_transform = src.transform
        has_geo = (crs is not None) or (base_transform != Affine.identity())
        crs_wkt = crs.to_wkt() if crs is not None else None
        colorinterp = list(src.colorinterp)
        colormaps: Dict[int, Dict[int, Tuple[int, int, int, int]]] = {}
        for b_idx in range(1, n_bands + 1):
            if colorinterp[b_idx - 1] == ColorInterp.palette:
                try:
                    colormaps[b_idx] = src.colormap(b_idx)
                except ValueError:
                    pass
        base_tags = src.tags()
        band_tags = [src.tags(b_idx) for b_idx in range(1, n_bands + 1)]
        descriptions = list(src.descriptions)

    if levels is None:
        min_size = min(base_w, base_h)
        resolved_levels = [
            lvl
            for lvl in (4, 8, 16, 32, 64, 128, 256, 512)
            if (min_size // lvl) > 33
        ]
        if not resolved_levels:
            return
    else:
        if isinstance(levels, (str, bytes)) or not levels:
            raise rsgispy.RSGISPyException(
                "levels must be a non-empty sequence of integers >= 2."
            )
        resolved_levels = sorted(
            {validate_positive_int(lvl, "levels entry", min_val=2) for lvl in levels}
        )

    uniform_nd = uniform_nodata(nodatavals)
    scratch_root: Optional[str] = None
    if tmp_dir is not None:
        scratch_root = os.fspath(tmp_dir)
        os.makedirs(scratch_root, exist_ok=True)

    tqdm_cls = rsgispy.get_tqdm()
    with tempfile.TemporaryDirectory(
        prefix="rsgispy_ovr_", dir=scratch_root
    ) as work_dir:
        scale_files: List[Tuple[int, int, int, str, Optional[Affine]]] = []
        for lvl in resolved_levels:
            ovr_w = max(1, math.ceil(base_w / lvl))
            ovr_h = max(1, math.ceil(base_h / lvl))

            src_path = input_path
            src_w = base_w
            src_h = base_h
            is_base_src = True
            if cascade:
                for prev_lvl, prev_w, prev_h, prev_path, _ in reversed(scale_files):
                    if lvl % prev_lvl == 0:
                        src_path = prev_path
                        src_w = prev_w
                        src_h = prev_h
                        is_base_src = False
                        break

            lvl_xfm: Optional[Affine] = None
            if has_geo:
                lvl_xfm = base_transform @ Affine.scale(
                    base_w / float(ovr_w), base_h / float(ovr_h)
                )

            scale_x = src_w / float(ovr_w)
            scale_y = src_h / float(ovr_h)
            tasks: List[Dict[str, Any]] = []
            for row_off in range(0, ovr_h, t_size):
                cur_h = min(t_size, ovr_h - row_off)
                for col_off in range(0, ovr_w, t_size):
                    cur_w = min(t_size, ovr_w - col_off)
                    tile_path = os.path.join(
                        work_dir, f"lvl_{lvl}_r{row_off}_c{col_off}.tif"
                    )
                    tile_xfm_tuple: Optional[
                        Tuple[float, float, float, float, float, float]
                    ] = None
                    if lvl_xfm is not None:
                        t_xfm = lvl_xfm @ Affine.translation(col_off, row_off)
                        tile_xfm_tuple = (
                            float(t_xfm.a),
                            float(t_xfm.b),
                            float(t_xfm.c),
                            float(t_xfm.d),
                            float(t_xfm.e),
                            float(t_xfm.f),
                        )
                    tasks.append(
                        {
                            "src_path": src_path,
                            "src_rect": (
                                col_off * scale_x,
                                row_off * scale_y,
                                cur_w * scale_x,
                                cur_h * scale_y,
                            ),
                            "tile_path": tile_path,
                            "col_off": col_off,
                            "row_off": row_off,
                            "tile_w": cur_w,
                            "tile_h": cur_h,
                            "dtypes": dtypes,
                            "nodatavals": nodatavals,
                            "nodata_val": uniform_nd,
                            "resamp_name": resamp_name,
                            "is_base_src": is_base_src,
                            "crs_wkt": crs_wkt,
                            "xfm_vals": tile_xfm_tuple,
                            "compress": comp_val,
                        }
                    )

            if eff_cores > 1 and len(tasks) > 1:
                with multiprocessing.Pool(processes=eff_cores) as pool:
                    tile_results = list(
                        tqdm_cls(
                            pool.imap(render_ovr_tile_worker, tasks),
                            total=len(tasks),
                            desc=f"Overview 1:{lvl}",
                            unit="tile",
                            disable=not use_pbar,
                        )
                    )
            else:
                tile_results = [
                    render_ovr_tile_worker(task)
                    for task in tqdm_cls(
                        tasks,
                        desc=f"Overview 1:{lvl}",
                        unit="tile",
                        disable=not use_pbar,
                    )
                ]

            lvl_path = os.path.join(work_dir, f"lvl_{lvl}.tif")
            stitch_scale_tiles(
                lvl_path=lvl_path,
                tile_results=tile_results,
                ovr_w=ovr_w,
                ovr_h=ovr_h,
                n_bands=n_bands,
                dtype_str=dtypes[0],
                nodata_val=uniform_nd,
                crs_wkt=crs_wkt,
                lvl_transform=lvl_xfm,
                compress=comp_val,
                ovr_block_size=ovr_blk,
            )
            scale_files.append((lvl, ovr_w, ovr_h, lvl_path, lvl_xfm))

        if external:
            out_ovr_path = (
                os.fspath(output_img)
                if output_img is not None
                else f"{input_path}.ovr"
            )
            finalise_external_ovr(
                out_ovr_path=out_ovr_path,
                work_dir=work_dir,
                scale_files=scale_files,
                dtypes=dtypes,
                nodatavals=nodatavals,
                crs_wkt=crs_wkt,
                comp_val=comp_val,
                ovr_blk=ovr_blk,
            )
        else:
            dest_path = (
                os.fspath(output_img) if output_img is not None else input_path
            )
            finalise_internal_ovr(
                dest_path=dest_path,
                input_path=input_path,
                work_dir=work_dir,
                base_w=base_w,
                base_h=base_h,
                scale_files=scale_files,
                dtypes=dtypes,
                nodatavals=nodatavals,
                crs_wkt=crs_wkt,
                base_transform=base_transform,
                has_geo=has_geo,
                colorinterp=colorinterp,
                colormaps=colormaps,
                base_tags=base_tags,
                band_tags=band_tags,
                descriptions=descriptions,
                comp_val=comp_val,
                ovr_blk=ovr_blk,
                remove_stale_ovr=(output_img is None),
            )


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
    use_multi_core_pyramids: bool = False,
    n_cores: int = 1,
    tile_size: int = 1024,
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
    :param use_multi_core_pyramids: If ``True``, use
                                    :func:`build_img_pyramids_multi_core` instead
                                    of :func:`build_img_pyramids` when
                                    ``calc_pyramids=True`` (Default = ``False``).
    :param n_cores: Number of worker processes when
                    ``use_multi_core_pyramids=True`` (Default = ``1``).
    :param tile_size: Overview tile width and height in pixels at each scale
                      when ``use_multi_core_pyramids=True`` (Default = ``1024``).

    .. code:: python

        import rsgispy.imageutils.stats
        rsgispy.imageutils.stats.pop_thmt_img_stats("land_cover_cls_img.tif")
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
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
        if use_multi_core_pyramids:
            build_img_pyramids_multi_core(
                input_img,
                resampling_method=Resampling.nearest,
                tile_size=tile_size,
                n_cores=n_cores,
                use_pbar=use_pbar,
            )
        else:
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
    use_multi_core_pyramids: bool = False,
    n_cores: int = 1,
    tile_size: int = 1024,
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
    :param use_multi_core_pyramids: If ``True``, use
                                    :func:`build_img_pyramids_multi_core` instead
                                    of :func:`build_img_pyramids` when
                                    ``calc_pyramids=True`` (Default = ``False``).
    :param n_cores: Number of worker processes when
                    ``use_multi_core_pyramids=True`` (Default = ``1``).
    :param tile_size: Overview tile width and height in pixels at each scale
                      when ``use_multi_core_pyramids=True`` (Default = ``1024``).

    .. code:: python

        import rsgispy.imageutils.stats
        rsgispy.imageutils.stats.pop_img_stats("input_img.tif", no_data_val=0.0)
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
        if use_multi_core_pyramids:
            build_img_pyramids_multi_core(
                input_img,
                resampling_method=Resampling.average,
                tile_size=tile_size,
                n_cores=n_cores,
                use_pbar=use_pbar,
            )
        else:
            build_img_pyramids(
                input_img,
                resampling_method=Resampling.average,
                ignore_cog=ignore_cog,
            )
