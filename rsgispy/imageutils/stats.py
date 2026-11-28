"""
The imageutils.stats module provides functions for building raster overview
pyramids and populating thematic and continuous image statistics and histograms
using ``rasterio``.
"""

from contextlib import ExitStack
import math
import multiprocessing
import os
import tempfile
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import rasterio
import rasterio.errors
from rasterio.enums import ColorInterp, Resampling
from rasterio.transform import Affine

import rsgispy
from rsgispy.imagecalc.stats import calc_basic_img_band_stats
from rsgispy.imageprocess import (
    build_valid_mask,
    ensure_file_exists,
    is_cog_layout,
    iter_single_image_band_blocks,
    set_dataset_nodata,
    update_cog,
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
                     When ``external=False`` the rewritten base image instead
                     takes the GeoTIFF creation options of
                     :func:`rsgispy.imageprocess.get_gdal_img_creation_opts`
                     (tiled and DEFLATE-compressed by default).
    :param ovr_block_size: Internal TIFF tile block size (power of 2 between
                           ``64`` and ``4096``, Default = ``128``).
    :param cascade: If ``True`` (default), generate scale ``lvl`` from the
                    largest previously built divisor scale (``lvl % prev == 0``)
                    for faster hierarchical downsampling; if ``False``, sample
                    every scale directly from ``input_img``.
    :param use_pbar: Display a tqdm progress bar per overview scale
                     (Default = ``True``).

    .. note::

        When calling this function from a Python script with ``n_cores > 1``,
        wrap the call inside an ``if __name__ == "__main__":`` guard. On macOS,
        Windows, and Python 3.14+ on Linux, Python's ``multiprocessing`` uses
        ``spawn`` (or ``forkserver``), which re-imports the calling script in
        every spawned worker process. Without the guard, each worker
        re-executes the top-level call and fails during process bootstrapping
        with a ``RuntimeError``.

    .. code:: python

        from rasterio.enums import Resampling
        import rsgispy.imageutils.stats as rstats

        if __name__ == "__main__":
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
        colour_interp = list(src.colorinterp)
        colour_maps: Dict[int, Dict[int, Tuple[int, int, int, int]]] = {}
        for b_idx in range(1, n_bands + 1):
            if colour_interp[b_idx - 1] == ColorInterp.palette:
                try:
                    colour_maps[b_idx] = src.colormap(b_idx)
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
        need_pool = eff_cores > 1 and any(
            (
                math.ceil(max(1, math.ceil(base_w / lvl)) / t_size)
                * math.ceil(max(1, math.ceil(base_h / lvl)) / t_size)
            )
            > 1
            for lvl in resolved_levels
        )
        with ExitStack() as pool_stack:
            pool = (
                pool_stack.enter_context(
                    multiprocessing.Pool(processes=eff_cores)
                )
                if need_pool
                else None
            )
            for lvl in resolved_levels:
                ovr_w = max(1, math.ceil(base_w / lvl))
                ovr_h = max(1, math.ceil(base_h / lvl))

                src_path = input_path
                src_w = base_w
                src_h = base_h
                is_base_src = True
                if cascade:
                    for (
                        prev_lvl,
                        prev_w,
                        prev_h,
                        prev_path,
                        _,
                    ) in reversed(scale_files):
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
                            t_xfm = lvl_xfm @ Affine.translation(
                                col_off, row_off
                            )
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
                                "no_data_val": uniform_nd,
                                "resamp_name": resamp_name,
                                "is_base_src": is_base_src,
                                "crs_wkt": crs_wkt,
                                "xfm_vals": tile_xfm_tuple,
                                "compress": comp_val,
                            }
                        )

                if pool is not None and len(tasks) > 1:
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
                    no_data_val=uniform_nd,
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
                colour_interp=colour_interp,
                colour_maps=colour_maps,
                base_tags=base_tags,
                band_tags=band_tags,
                descriptions=descriptions,
                comp_val=comp_val,
                ovr_blk=ovr_blk,
                remove_stale_ovr=(output_img is None),
            )


#: Widest value range stored as a direct (one bin per value) histogram;
#: wider ranges get a linear histogram with this many bins.
MAX_DIRECT_HIST_BINS = 65536


def _merge_value_counts(
    acc: Optional[Tuple[np.ndarray, np.ndarray]],
    vals: np.ndarray,
    counts: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Merge sorted ``(values, counts)`` into the accumulator ``acc``."""
    if acc is None:
        return vals, counts.astype(np.int64)
    all_v = np.concatenate([acc[0], vals])
    all_c = np.concatenate([acc[1], counts.astype(np.int64)])
    uniq, inv = np.unique(all_v, return_inverse=True)
    return uniq, np.bincount(inv.ravel(), weights=all_c).astype(np.int64)


def _calc_int_band_value_counts(
    input_img: str, dtype_str: str, use_pbar: bool = True
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Exact sorted ``(values, counts)`` of every value in each band of an
    integer image, accumulated block by block (memory scales with the number
    of distinct values, not the value range).
    """
    with rasterio.open(input_img, "r") as src:
        n_bands = src.count
    read_np_dtype = np.uint64 if dtype_str == "uint64" else np.int64
    accs: List[Optional[Tuple[np.ndarray, np.ndarray]]] = [None] * n_bands
    for _, block_arrays in iter_single_image_band_blocks(
        input_img=input_img,
        read_dtype=read_np_dtype,
        use_pbar=use_pbar,
    ):
        for idx, band_arr in enumerate(block_arrays):
            if band_arr.size == 0:
                continue
            vals, counts = np.unique(band_arr.ravel(), return_counts=True)
            accs[idx] = _merge_value_counts(accs[idx], vals, counts)
    empty = (np.zeros(0, dtype=read_np_dtype), np.zeros(0, dtype=np.int64))
    return [acc if acc is not None else empty for acc in accs]


def _hist_from_value_counts(
    vals: np.ndarray, counts: np.ndarray, dtype_str: str
) -> Tuple[np.ndarray, Any, Any, str]:
    """
    Histogram ``(hist, hist_min, hist_max, bin_function)`` of a band. The
    range always includes 0 (and is ``0..255`` for ``uint8``). Ranges of up
    to :data:`MAX_DIRECT_HIST_BINS` values give a ``'direct'`` histogram (one
    bin per value); wider ranges a ``'linear'`` one with that many bins.
    """
    if dtype_str == "uint8":
        h_min, h_max = 0, 255
    elif vals.size == 0:
        h_min, h_max = 0, 0
    else:
        h_min, h_max = min(0, int(vals[0])), max(0, int(vals[-1]))
    n_vals = (h_max - h_min) + 1
    if n_vals <= MAX_DIRECT_HIST_BINS:
        hist = np.zeros(n_vals, dtype=np.int64)
        if vals.size:
            idx = (vals.astype(np.float64) - h_min).astype(np.intp)
            np.add.at(hist, idx, counts)
        return hist, h_min, h_max, "direct"
    n_bins = MAX_DIRECT_HIST_BINS
    width = (float(h_max) - float(h_min) + 1.0) / n_bins
    idx = np.minimum(
        ((vals.astype(np.float64) - h_min) / width).astype(np.intp), n_bins - 1
    )
    hist = np.bincount(idx, weights=counts, minlength=n_bins).astype(np.int64)
    return hist, h_min, h_max, "linear"


def _stats_from_value_counts(
    vals: np.ndarray, counts: np.ndarray, ignore_zero: bool
) -> Tuple[int, int, float, float, int, int]:
    """
    Exact ``(min, max, mean, stddev, mode, median)`` of a band from its
    sorted ``(values, counts)``; the standard deviation is the population
    one (``ddof=0``), as in RSGISLib.
    """
    stat_counts = counts.copy()
    if ignore_zero:
        stat_counts[vals == 0] = 0
    total_valid = int(stat_counts.sum())
    vals_f = vals.astype(np.float64)
    if total_valid == 0:
        min_val, max_val, mean_val, stddev_val = 0, 0, 0.0, 0.0
    else:
        nz = np.nonzero(stat_counts)[0]
        min_val = int(vals[nz[0]])
        max_val = int(vals[nz[-1]])
        mean_val = float(np.sum(vals_f * stat_counts) / total_valid)
        variance = float(np.sum(((vals_f - mean_val) ** 2) * stat_counts) / total_valid)
        stddev_val = float(np.sqrt(variance))
    active = stat_counts if total_valid > 0 else counts
    if active.size == 0 or int(active.sum()) == 0:
        return min_val, max_val, mean_val, stddev_val, 0, 0
    mode_val = int(vals[int(np.argmax(active))])
    median_idx = int(np.nonzero(active.cumsum() >= active.sum() / 2.0)[0][0])
    return min_val, max_val, mean_val, stddev_val, mode_val, int(vals[median_idx])


def _thematic_colour_table(
    n_entries: int, ignore_zero: bool
) -> Dict[int, Tuple[int, int, int, int]]:
    """
    Random colour table for values ``0..n_entries-1`` with a fixed seed, so a
    value gets the same colour every run (and in every image). Values below
    65536 keep the colours of earlier rsgispy versions; the rest are drawn
    in one call.
    """
    clr_rng = np.random.default_rng(42)
    clr_tab: Dict[int, Tuple[int, int, int, int]] = {}
    n_loop = min(n_entries, 65536)
    for val_i in range(n_loop):
        rnd_rgb = clr_rng.integers(1, 255, 3, dtype=np.uint8)
        if val_i == 0 and ignore_zero:
            clr_tab[0] = (0, 0, 0, 0)
        else:
            clr_tab[val_i] = (int(rnd_rgb[0]), int(rnd_rgb[1]), int(rnd_rgb[2]), 255)
    if n_entries > n_loop:
        extra = clr_rng.integers(1, 255, (n_entries - n_loop, 3), dtype=np.uint8)
        for off, (red, grn, blu) in enumerate(extra.tolist()):
            clr_tab[n_loop + off] = (red, grn, blu, 255)
    return clr_tab


def pop_thmt_img_stats(
    input_img: str,
    add_clr_tab: bool = True,
    calc_pyramids: bool = True,
    ignore_zero: bool = True,
    use_pbar: bool = True,
    use_multi_core_pyramids: bool = False,
    n_cores: int = 1,
    tile_size: int = 1024,
    ignore_cog: bool = False,
) -> None:
    """
    Populate an integer thematic input image (e.g., ``uint8``, ``uint16``, ``int16``
    classification) with header statistics, a colour table, and overview pyramids
    using ``rasterio``.

    :param input_img: Path to the input thematic image file.
    :param add_clr_tab: Whether to compute statistics and add a colour table (Default = True).
    :param calc_pyramids: Whether to build overview pyramids (Default = True).
    :param ignore_zero: Whether to treat pixel value 0 as no-data and set it as
                        the image no-data value (Default = True). For a KEA file
                        that already has a different no-data value this raises
                        :class:`rsgispy.RSGISPyException` (see
                        :func:`rsgispy.imageprocess.set_dataset_nodata`).
    :param use_pbar: Whether to display a tqdm progress bar (Default = True).
    :param use_multi_core_pyramids: If ``True``, use
                                    :func:`build_img_pyramids_multi_core` instead
                                    of :func:`build_img_pyramids` when
                                    ``calc_pyramids=True`` (Default = ``False``).
    :param n_cores: Number of worker processes when
                    ``use_multi_core_pyramids=True`` (Default = ``1``).
    :param tile_size: Overview tile width and height in pixels at each scale
                      when ``use_multi_core_pyramids=True`` (Default = ``1024``).
    :param ignore_cog: Kept for compatibility. Cloud Optimised GeoTIFFs are
                       detected from the file and never edited in place.

    .. note::

        A Cloud Optimised GeoTIFF (see
        :func:`rsgispy.imageprocess.is_cog_layout`) is not edited in place,
        which would break its layout. It is rebuilt with
        :func:`rsgispy.imageprocess.update_cog`: copied to a temporary
        GeoTIFF, given the statistics and colour table, and converted back
        with its original compression and block size. Its overviews are
        rebuilt with nearest-neighbour resampling (unless they already use
        ``NEAREST`` or ``MODE``) and rsgispy's own pyramids are not added.
        The rebuild decodes and re-encodes the whole image. To avoid it,
        calculate the statistics when the COG is written (``calc_stats=True``
        on rsgispy functions).

    .. note::

        When calling this function from a Python script with
        ``calc_pyramids=True``, ``use_multi_core_pyramids=True``, and
        ``n_cores > 1``, wrap the call inside an
        ``if __name__ == "__main__":`` guard. On macOS, Windows, and Python
        3.14+ on Linux, Python's ``multiprocessing`` uses ``spawn`` (or
        ``forkserver``), which re-imports the calling script in every spawned
        worker process. Without the guard, each worker re-executes the
        top-level call and fails during process bootstrapping with a
        ``RuntimeError``.

    .. code:: python

        import rsgispy.imageutils.stats

        if __name__ == "__main__":
            rsgispy.imageutils.stats.pop_thmt_img_stats(
                "land_cover_cls_img.tif",
                use_multi_core_pyramids=True,
                n_cores=4,
            )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    if not os.path.exists(input_img):
        raise rsgispy.RSGISPyException(
            f"Input image file does not exist: '{input_img}'."
        )

    with rasterio.open(input_img, "r") as src:
        dtype_str = src.dtypes[0]
        if not np.issubdtype(np.dtype(dtype_str), np.integer):
            raise rsgispy.RSGISPyException(
                "This function only supports integer datasets"
            )
        ovr_resamp = str(
            src.tags(ns="IMAGE_STRUCTURE").get("OVERVIEW_RESAMPLING", "")
        ).upper()

    if is_cog_layout(input_img):
        update_cog(
            input_img,
            lambda tmp_img: pop_thmt_img_stats(
                tmp_img,
                add_clr_tab=add_clr_tab,
                calc_pyramids=False,
                ignore_zero=ignore_zero,
                use_pbar=use_pbar,
            ),
            overview_resampling=(
                None if ovr_resamp in ("NEAREST", "MODE") else "NEAREST"
            ),
        )
        return

    set_img_thematic(input_img, ignore_cog=ignore_cog)

    open_opts: Dict[str, Any] = {}
    if ignore_cog:
        open_opts["IGNORE_COG_LAYOUT_BREAK"] = "YES"

    if add_clr_tab:
        val_counts = _calc_int_band_value_counts(
            input_img, dtype_str=dtype_str, use_pbar=use_pbar
        )
        with rasterio.open(input_img, "r+", **open_opts) as dst:
            if ignore_zero:
                set_dataset_nodata(dst, 0)

            for idx, (vals, counts) in enumerate(val_counts):
                bidx = idx + 1
                hist, h_min, h_max, bin_fn = _hist_from_value_counts(
                    vals, counts, dtype_str
                )
                min_v, max_v, mean_v, std_v, mode_v, med_v = (
                    _stats_from_value_counts(vals, counts, ignore_zero)
                )
                tags: Dict[str, str] = {
                    "STATISTICS_MINIMUM": f"{min_v}",
                    "STATISTICS_MAXIMUM": f"{max_v}",
                    "STATISTICS_MEAN": f"{mean_v}",
                    "STATISTICS_STDDEV": f"{std_v}",
                    "STATISTICS_SKIPFACTORX": "1",
                    "STATISTICS_SKIPFACTORY": "1",
                    "STATISTICS_HISTOBINFUNCTION": bin_fn,
                    "STATISTICS_MODE": f"{mode_v}",
                    "STATISTICS_MEDIAN": f"{med_v}",
                    "STATISTICS_HISTOBINVALUES": (
                        "|".join(map(str, hist.tolist())) + "|"
                    ),
                    "STATISTICS_HISTOMIN": f"{h_min}",
                    "STATISTICS_HISTOMAX": f"{h_max}",
                    "STATISTICS_HISTONUMBINS": f"{hist.size}",
                }
                if ignore_zero:
                    tags["STATISTICS_EXCLUDEDVALUES"] = "0"
                dst.update_tags(bidx, **tags)

                # One entry for every value up to the band maximum.
                n_clr_entries = 256 if dtype_str == "uint8" else max(h_max, 0) + 1
                try:
                    dst.write_colormap(
                        bidx, _thematic_colour_table(n_clr_entries, ignore_zero)
                    )
                    # Some drivers (e.g. GeoTIFF for 32-bit types) silently
                    # store no colour table; reading it back raises.
                    dst.colormap(bidx)
                except (ValueError, rasterio.errors.RasterioError) as err:
                    warnings.warn(
                        f"Could not write a colour table for band {bidx} of "
                        f"'{input_img}' ({dtype_str}, {n_clr_entries} entries): "
                        f"{err}. The statistics were written; use KEA (or a "
                        "uint8/uint16 GeoTIFF) for a colour table.",
                        UserWarning,
                        stacklevel=2,
                    )

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
            build_img_pyramids(
                input_img,
                resampling_method=Resampling.nearest,
                ignore_cog=ignore_cog,
            )


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
            valid = build_valid_mask(band_arr, no_data_val)
            if np.any(valid):
                data = band_arr[valid]
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
    :param no_data_val: Optional no-data value to set and exclude from
                        statistics (if ``None``, the image's own no-data value
                        is excluded). Setting the value the image already has
                        writes nothing, so this works for KEA files; a KEA file
                        whose existing no-data value differs raises
                        :class:`rsgispy.RSGISPyException` because GDAL's KEA
                        driver cannot replace it (see
                        :func:`rsgispy.imageprocess.set_dataset_nodata`).
    :param calc_pyramids: Whether to build overview pyramids (Default = True).
    :param ignore_cog: Kept for compatibility. Cloud Optimised GeoTIFFs are
                       detected from the file and never edited in place.
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

    .. note::

        A Cloud Optimised GeoTIFF (see
        :func:`rsgispy.imageprocess.is_cog_layout`) is not edited in place,
        which would break its layout. It is rebuilt with
        :func:`rsgispy.imageprocess.update_cog`: copied to a temporary
        GeoTIFF next to it, given the statistics, and converted back with its
        original compression, predictor, block size and overview resampling.
        The COG keeps overviews built by the COG driver; rsgispy's own
        pyramids are not added. The rebuild decodes and re-encodes the whole
        image, so it takes about as long as creating the COG and needs
        temporary disk space for one compressed copy. To avoid it, calculate
        the statistics when the COG is written (``calc_stats=True`` on rsgispy
        functions, which do this before the COG is created).

    .. note::

        When calling this function from a Python script with
        ``calc_pyramids=True``, ``use_multi_core_pyramids=True``, and
        ``n_cores > 1``, wrap the call inside an
        ``if __name__ == "__main__":`` guard. On macOS, Windows, and Python
        3.14+ on Linux, Python's ``multiprocessing`` uses ``spawn`` (or
        ``forkserver``), which re-imports the calling script in every spawned
        worker process. Without the guard, each worker re-executes the
        top-level call and fails during process bootstrapping with a
        ``RuntimeError``.

    .. code:: python

        import rsgispy.imageutils.stats

        if __name__ == "__main__":
            rsgispy.imageutils.stats.pop_img_stats(
                "input_img.tif",
                no_data_val=0.0,
                use_multi_core_pyramids=True,
                n_cores=4,
            )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    if is_cog_layout(input_img):
        update_cog(
            input_img,
            lambda tmp_img: pop_img_stats(
                tmp_img,
                no_data_val=no_data_val,
                calc_pyramids=False,
                use_approx=use_approx,
                use_pbar=use_pbar,
            ),
        )
        return

    eff_no_data = no_data_val
    if eff_no_data is None and os.path.exists(input_img):
        with rasterio.open(input_img, "r") as src_hdr:
            eff_no_data = src_hdr.nodata

    use_no_data = eff_no_data is not None
    img_min, img_max, img_mean, img_std = calc_basic_img_band_stats(
        input_img=input_img,
        no_data_val=eff_no_data,
        use_no_data=use_no_data,
        use_pbar=use_pbar,
    )

    hist_results = _calc_linear_band_hists(
        input_img=input_img,
        img_min=img_min,
        img_max=img_max,
        no_data_val=eff_no_data,
        use_pbar=use_pbar,
    )

    open_opts: Dict[str, Any] = {}
    if ignore_cog:
        open_opts["IGNORE_COG_LAYOUT_BREAK"] = "YES"

    float_dtypes = {"float32", "float64"}

    with rasterio.open(input_img, "r+", **open_opts) as dst:
        if no_data_val is not None:
            set_dataset_nodata(dst, no_data_val)

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
