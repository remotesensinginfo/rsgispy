"""
Tiled / windowed segmentation planner, halo extraction & cropping, per-tile ID
offsetting, cross-tile seam reconciliation, and bounded-memory streaming
pipelines for :mod:`rsgispy.segmentation`.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from dataclasses import dataclass
import os
import tempfile
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import rasterio
from rasterio.windows import Window

import rsgispy
from rsgispy._numba import use_compiled
from rsgispy._unionfind import uf_min_root_kernel
from rsgispy.imageprocess import (
    IntersectionGrid,
    build_valid_mask,
    get_default_block_size,
    iter_dataset_windows,
    open_output_dataset,
)
from rsgispy.imageutils.stats import pop_img_stats
from rsgispy.segmentation.utils import (
    call_kernel,
    check_label_overflow,
    eliminate_small_segments_array,
    relabel_segments_array,
    resolve_input_nodata,
    resolve_label_dtype,
    validate_n_jobs,
    worker_init_single_thread,
    _eliminate_small_regions_kernel,
)

__all__ = [
    "TileWindow",
    "normalise_tile_size",
    "plan_tiles",
    "extract_halo_tile",
    "crop_halo_labels",
    "offset_tile_labels",
    "reconcile_seam_equivalences",
    "run_tiled_array_pipeline",
    "run_tiled_raster_pipeline",
]

_AUTO_SINGLE_PASS_MAX_PXLS = 16_777_216


@dataclass(frozen=True)
class TileWindow:
    """
    Descriptor for a core tile window and its halo-padded read window.

    * ``tile_id``: Deterministic 0-based sequential tile index in row-major order.
    * ``grid_row``, ``grid_col``: 0-based tile indices in the tile grid.
    * ``row_off``, ``col_off``, ``height``, ``width``: Core tile origin and
      dimensions in full-image pixel coordinates.
    * ``halo_row_off``, ``halo_col_off``, ``halo_height``, ``halo_width``:
      Halo-expanded window clipped to ``[0, full_height) x [0, full_width)``.
    * ``core_row_slice``, ``core_col_slice``: Slices inside the halo array
      extracting the core tile.
    """

    tile_id: int
    grid_row: int
    grid_col: int
    row_off: int
    col_off: int
    height: int
    width: int
    halo_row_off: int
    halo_col_off: int
    halo_height: int
    halo_width: int
    core_row_slice: slice
    core_col_slice: slice

    def core_rio_window(self) -> Window:
        """Return the ``rasterio.windows.Window`` for the core tile."""
        return Window(self.col_off, self.row_off, self.width, self.height)

    def halo_rio_window(self) -> Window:
        """Return the ``rasterio.windows.Window`` for the halo-padded tile."""
        return Window(
            self.halo_col_off,
            self.halo_row_off,
            self.halo_width,
            self.halo_height,
        )


def normalise_tile_size(
    tile_size: Optional[Union[int, Tuple[int, int]]],
) -> Optional[Tuple[int, int]]:
    """
    Validate and normalise ``tile_size`` into ``(tile_h, tile_w)`` or ``None``.
    """
    if tile_size is None:
        return None
    if isinstance(tile_size, (int, np.integer)) and not isinstance(
        tile_size, bool
    ):
        val = int(tile_size)
        if val <= 0:
            raise rsgispy.RSGISPyException(
                f"tile_size must be positive, got {tile_size}."
            )
        return (val, val)
    if isinstance(tile_size, (tuple, list)) and len(tile_size) == 2:
        t_h, t_w = int(tile_size[0]), int(tile_size[1])
        if t_h <= 0 or t_w <= 0:
            raise rsgispy.RSGISPyException(
                f"tile_size dimensions must be positive, got {tile_size!r}."
            )
        return (t_h, t_w)
    raise rsgispy.RSGISPyException(
        f"tile_size must be None, a positive int, or a (height, width) tuple, got {tile_size!r}."
    )


def plan_tiles(
    height: int,
    width: int,
    tile_size: Optional[Union[int, Tuple[int, int]]] = None,
    overlap: int = 0,
) -> List[TileWindow]:
    """
    Partition a ``(height, width)`` grid into core tiles with halo overlap.

    Parameters
    ----------
    height, width : int
        Full raster spatial dimensions.
    tile_size : int or tuple[int, int] or None, optional
        Core tile dimensions ``(tile_h, tile_w)`` (or square ``int``). When
        ``None``, a single tile covering the full grid is returned.
    overlap : int, optional
        Number of halo overlap pixels added around each core tile edge before
        clipping to the image extent (Default = ``0``).

    Returns
    -------
    list of TileWindow
        Ordered list of tile descriptors in row-major order.
    """
    # pylint: disable=too-many-locals
    if height <= 0 or width <= 0:
        raise rsgispy.RSGISPyException(
            f"Image dimensions must be positive, got ({height}, {width})."
        )
    if overlap < 0:
        raise rsgispy.RSGISPyException(
            f"overlap must be >= 0, got {overlap}."
        )

    norm_ts = normalise_tile_size(tile_size)
    if norm_ts is None:
        return [
            TileWindow(
                tile_id=0,
                grid_row=0,
                grid_col=0,
                row_off=0,
                col_off=0,
                height=height,
                width=width,
                halo_row_off=0,
                halo_col_off=0,
                halo_height=height,
                halo_width=width,
                core_row_slice=slice(0, height),
                core_col_slice=slice(0, width),
            )
        ]

    tile_h, tile_w = norm_ts
    tiles: List[TileWindow] = []
    tile_id = 0
    g_row = 0
    for row_off in range(0, height, tile_h):
        cur_h = min(tile_h, height - row_off)
        halo_r0 = max(0, row_off - overlap)
        halo_r1 = min(height, row_off + cur_h + overlap)
        core_r0 = row_off - halo_r0
        core_r1 = core_r0 + cur_h

        g_col = 0
        for col_off in range(0, width, tile_w):
            cur_w = min(tile_w, width - col_off)
            halo_c0 = max(0, col_off - overlap)
            halo_c1 = min(width, col_off + cur_w + overlap)
            core_c0 = col_off - halo_c0
            core_c1 = core_c0 + cur_w

            tiles.append(
                TileWindow(
                    tile_id=tile_id,
                    grid_row=g_row,
                    grid_col=g_col,
                    row_off=row_off,
                    col_off=col_off,
                    height=cur_h,
                    width=cur_w,
                    halo_row_off=halo_r0,
                    halo_col_off=halo_c0,
                    halo_height=halo_r1 - halo_r0,
                    halo_width=halo_c1 - halo_c0,
                    core_row_slice=slice(core_r0, core_r1),
                    core_col_slice=slice(core_c0, core_c1),
                )
            )
            tile_id += 1
            g_col += 1
        g_row += 1
    return tiles


def extract_halo_tile(data: np.ndarray, tile: TileWindow) -> np.ndarray:
    """Extract the halo-padded slice for ``tile`` from a 2D or 3D array."""
    r0 = tile.halo_row_off
    r1 = r0 + tile.halo_height
    c0 = tile.halo_col_off
    c1 = c0 + tile.halo_width
    if data.ndim == 2:
        return data[r0:r1, c0:c1]
    if data.ndim == 3:
        return data[:, r0:r1, c0:c1]
    raise rsgispy.RSGISPyException(
        f"Expected 2D or 3D array, got {data.ndim}D array."
    )


def crop_halo_labels(halo_labels: np.ndarray, tile: TileWindow) -> np.ndarray:
    """Crop halo-padded 2D labels back to ``tile``'s non-overlapping core."""
    return halo_labels[tile.core_row_slice, tile.core_col_slice]


def offset_tile_labels(
    labels: np.ndarray,
    offset: int,
    no_data_val: int = 0,
) -> Tuple[np.ndarray, int]:
    """
    Shift positive labels in ``labels`` by ``offset`` while keeping
    ``no_data_val`` unchanged. Returns ``(shifted_labels, new_max_id)``.
    """
    out = labels.astype(np.int64, copy=True)
    valid = out != no_data_val
    if no_data_val >= 0:
        valid &= out > 0
    if not np.any(valid):
        return out, offset
    out[valid] += int(offset)
    return out, int(out[valid].max())


def _reconcile_seam_equivalences_py(
    n_labels: int,
    merge_pairs: Sequence[Tuple[int, int]],
) -> np.ndarray:
    """Pure-Python union-find fallback for :func:`reconcile_seam_equivalences`."""
    parent = np.arange(n_labels + 1, dtype=np.int64)
    for raw_u, raw_v in merge_pairs:
        u = int(raw_u)
        v = int(raw_v)
        if u <= 0 or v <= 0 or u > n_labels or v > n_labels:
            continue
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        if u != v:
            if u < v:
                parent[v] = u
            else:
                parent[u] = v

    for idx in range(1, n_labels + 1):
        root = idx
        while parent[root] != root:
            root = parent[root]
        parent[idx] = root
    return parent


def reconcile_seam_equivalences(
    n_labels: int,
    merge_pairs: Union[Sequence[Tuple[int, int]], np.ndarray],
) -> np.ndarray:
    """
    Resolve cross-tile segment equivalence pairs ``(u, v)`` into a deterministic
    1D lookup table mapping ``1..n_labels`` to canonical component roots (with
    ``0`` preserved). Each label maps to the minimum label in its component.

    ``merge_pairs`` may be a sequence of pairs or an ``(N, 2)`` integer array.
    Pairs referencing labels outside ``1..n_labels`` are ignored. Uses a
    ``numba``-compiled union-by-rank / path-halving kernel when available
    (suitable for ``10^5 - 10^7`` labels) and a pure-Python fallback otherwise.
    """
    n_labels = max(int(n_labels), 0)
    if not use_compiled(uf_min_root_kernel, "reconcile_seam_equivalences"):
        if isinstance(merge_pairs, np.ndarray):
            merge_pairs = merge_pairs.reshape(-1, 2).tolist()
        return _reconcile_seam_equivalences_py(n_labels, merge_pairs)
    pairs = np.asarray(merge_pairs, dtype=np.int64).reshape(-1, 2)
    return uf_min_root_kernel(
        np.ascontiguousarray(pairs[:, 0]),
        np.ascontiguousarray(pairs[:, 1]),
        n_labels + 1,
        1,
    )


@dataclass
class _TileSeamRecord:
    """Boundary strips and per-segment summaries stored per completed tile."""

    tile: TileWindow
    offset: int
    max_local_id: int
    top_ids: np.ndarray
    bot_ids: np.ndarray
    left_ids: np.ndarray
    right_ids: np.ndarray
    top_vals: np.ndarray
    bot_vals: np.ndarray
    left_vals: np.ndarray
    right_vals: np.ndarray
    top_aux: Optional[np.ndarray]
    bot_aux: Optional[np.ndarray]
    left_aux: Optional[np.ndarray]
    right_aux: Optional[np.ndarray]
    counts: np.ndarray
    band_sums: np.ndarray


def _collect_tile_seam_record(
    tile: TileWindow,
    offset: int,
    core_labels: np.ndarray,
    core_arr: np.ndarray,
    core_aux: Optional[np.ndarray],
    seam_stds: Optional[np.ndarray] = None,
) -> _TileSeamRecord:
    """Summarise a tile's border pixels and per-segment band sums."""
    local_max = int(core_labels.max()) if core_labels.size > 0 else 0
    n_bands = core_arr.shape[0]
    scales = (
        np.where(seam_stds > 1e-12, seam_stds, 1.0)
        if seam_stds is not None
        else np.ones(n_bands, dtype=np.float64)
    )
    if local_max > 0:
        flat_lbl = core_labels.ravel()
        counts = np.bincount(flat_lbl, minlength=local_max + 1).astype(
            np.int64
        )
        counts[0] = 0
        band_sums = np.zeros((local_max + 1, n_bands), dtype=np.float64)
        for b_i in range(n_bands):
            b_vals = (
                np.where(np.isfinite(core_arr[b_i]), core_arr[b_i], 0.0).ravel()
                / scales[b_i]
            )
            band_sums[:, b_i] = np.bincount(
                flat_lbl, weights=b_vals, minlength=local_max + 1
            )
    else:
        counts = np.zeros(1, dtype=np.int64)
        band_sums = np.zeros((1, n_bands), dtype=np.float64)

    inv_scales = (1.0 / scales)[:, np.newaxis]
    return _TileSeamRecord(
        tile=tile,
        offset=offset,
        max_local_id=local_max,
        top_ids=core_labels[0, :].copy(),
        bot_ids=core_labels[-1, :].copy(),
        left_ids=core_labels[:, 0].copy(),
        right_ids=core_labels[:, -1].copy(),
        top_vals=core_arr[:, 0, :].copy() * inv_scales,
        bot_vals=core_arr[:, -1, :].copy() * inv_scales,
        left_vals=core_arr[:, :, 0].copy() * inv_scales,
        right_vals=core_arr[:, :, -1].copy() * inv_scales,
        top_aux=core_aux[0, :].copy() if core_aux is not None else None,
        bot_aux=core_aux[-1, :].copy() if core_aux is not None else None,
        left_aux=core_aux[:, 0].copy() if core_aux is not None else None,
        right_aux=core_aux[:, -1].copy() if core_aux is not None else None,
        counts=counts,
        band_sums=band_sums,
    )


def _gather_seam_pairs_and_merges(
    records_by_grid: Dict[Tuple[int, int], _TileSeamRecord],
    total_raw_labels: int,
    n_bands: int,
    seam_mode: str,
    seam_threshold: float,
    stitch_seams: bool,
    min_pixels: int,
    use_numba: bool,
) -> np.ndarray:
    """
    Build global segment LUT across tiles: reconcile seams, optionally merge
    sub-threshold border segments across tile edges, and relabel ``1..N``.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-branches
    if total_raw_labels == 0:
        return np.zeros(1, dtype=np.uint32)

    global_counts = np.zeros(total_raw_labels + 1, dtype=np.int64)
    global_sums = np.zeros((total_raw_labels + 1, n_bands), dtype=np.float64)
    for rec in records_by_grid.values():
        if rec.max_local_id > 0:
            s_idx = rec.offset + 1
            e_idx = rec.offset + rec.max_local_id + 1
            global_counts[s_idx:e_idx] = rec.counts[1:]
            global_sums[s_idx:e_idx, :] = rec.band_sums[1:, :]

    # Accumulate all seam adjacency pixel pairs in deterministic traversal
    # order, then reduce per unique edge (u < v) with ``np.bincount`` (which
    # sums sequentially in input order, matching a per-pixel dict reduction).
    seam_u: List[np.ndarray] = []
    seam_v: List[np.ndarray] = []
    seam_d: List[np.ndarray] = []
    exact_u: List[np.ndarray] = []
    exact_v: List[np.ndarray] = []

    def _inspect_seam(
        ids_a: np.ndarray,
        off_a: int,
        vals_a: np.ndarray,
        aux_a: Optional[np.ndarray],
        ids_b: np.ndarray,
        off_b: int,
        vals_b: np.ndarray,
        aux_b: Optional[np.ndarray],
    ) -> None:
        valid_seam = (ids_a > 0) & (ids_b > 0)
        if not np.any(valid_seam):
            return
        ga = ids_a[valid_seam].astype(np.int64) + off_a
        gb = ids_b[valid_seam].astype(np.int64) + off_b
        diff_vec = vals_a[:, valid_seam] - vals_b[:, valid_seam]
        px_dists = np.sqrt(np.sum(diff_vec * diff_vec, axis=0))

        if (
            stitch_seams
            and seam_mode == "exact_aux"
            and aux_a is not None
            and aux_b is not None
        ):
            aa = aux_a[valid_seam]
            ab = aux_b[valid_seam]
            match_mask = (aa > 0) & (aa == ab)
            if np.any(match_mask):
                exact_u.append(ga[match_mask])
                exact_v.append(gb[match_mask])

        seam_u.append(ga)
        seam_v.append(gb)
        seam_d.append(np.asarray(px_dists, dtype=np.float64))

    for (g_r, g_c), rec in sorted(records_by_grid.items()):
        right_rec = records_by_grid.get((g_r, g_c + 1))
        if right_rec is not None:
            _inspect_seam(
                rec.right_ids,
                rec.offset,
                rec.right_vals,
                rec.right_aux,
                right_rec.left_ids,
                right_rec.offset,
                right_rec.left_vals,
                right_rec.left_aux,
            )
        bot_rec = records_by_grid.get((g_r + 1, g_c))
        if bot_rec is not None:
            _inspect_seam(
                rec.bot_ids,
                rec.offset,
                rec.bot_vals,
                rec.bot_aux,
                bot_rec.top_ids,
                bot_rec.offset,
                bot_rec.top_vals,
                bot_rec.top_aux,
            )

    # Unique undirected seam edges (sorted by (u, v)) with summed pixel
    # distances and pixel counts.
    if seam_u:
        all_u = np.concatenate(seam_u)
        all_v = np.concatenate(seam_v)
        all_d = np.concatenate(seam_d)
        lo = np.minimum(all_u, all_v)
        hi = np.maximum(all_u, all_v)
        key_base = np.int64(total_raw_labels + 1)
        uniq_keys, inv = np.unique(lo * key_base + hi, return_inverse=True)
        inv = inv.ravel()
        edge_u = uniq_keys // key_base
        edge_v = uniq_keys % key_base
        edge_diff_sums = np.bincount(inv, weights=all_d, minlength=uniq_keys.size)
        edge_pixel_counts = np.bincount(inv, minlength=uniq_keys.size)
    else:
        edge_u = np.empty(0, dtype=np.int64)
        edge_v = np.empty(0, dtype=np.int64)
        edge_diff_sums = np.empty(0, dtype=np.float64)
        edge_pixel_counts = np.empty(0, dtype=np.int64)
    has_edges = edge_u.size > 0

    pair_u: List[np.ndarray] = list(exact_u)
    pair_v: List[np.ndarray] = list(exact_v)
    if stitch_seams and seam_mode == "spectral" and has_edges:
        mean_edge_dist = edge_diff_sums / edge_pixel_counts.astype(np.float64)
        cu = np.maximum(1, global_counts[edge_u]).astype(np.float64)
        cv = np.maximum(1, global_counts[edge_v]).astype(np.float64)
        mean_u = global_sums[edge_u] / cu[:, np.newaxis]
        mean_v = global_sums[edge_v] / cv[:, np.newaxis]
        seg_dist = np.sqrt(np.sum((mean_u - mean_v) ** 2, axis=1))
        sel = (mean_edge_dist <= seam_threshold) & (
            seg_dist <= seam_threshold * 1.5
        )
        if np.any(sel):
            pair_u.append(edge_u[sel])
            pair_v.append(edge_v[sel])

    if pair_u:
        merge_pairs = np.stack(
            [np.concatenate(pair_u), np.concatenate(pair_v)], axis=1
        ).astype(np.int64, copy=False)
    else:
        merge_pairs = np.empty((0, 2), dtype=np.int64)

    seam_parent = reconcile_seam_equivalences(total_raw_labels, merge_pairs)

    # Update counts & band_sums onto seam roots (sequential bincount
    # accumulation in label order)
    n_lut = total_raw_labels + 1
    roots = seam_parent[1:]
    merged_counts = np.zeros(n_lut, dtype=np.int64)
    np.add.at(merged_counts, roots, global_counts[1:])
    merged_sums = np.zeros((n_lut, n_bands), dtype=np.float64)
    for b_i in range(n_bands):
        merged_sums[:, b_i] = np.bincount(
            roots, weights=global_sums[1:, b_i], minlength=n_lut
        )

    # Merge any remaining sub-threshold segments clipped by tile boundaries
    if min_pixels > 1 and has_edges:
        ru = seam_parent[edge_u]
        rv = seam_parent[edge_v]
        diff_root = ru != rv
        if np.any(diff_root):
            u_arr = np.minimum(ru, rv)[diff_root].astype(np.int64)
            v_arr = np.maximum(ru, rv)[diff_root].astype(np.int64)
            elim_parent = call_kernel(
                _eliminate_small_regions_kernel,
                use_numba,
                u_arr,
                v_arr,
                merged_counts,
                merged_sums,
                True,
                int(min_pixels),
                32,
            )
        else:
            elim_parent = np.arange(total_raw_labels + 1, dtype=np.int64)
    else:
        elim_parent = np.arange(total_raw_labels + 1, dtype=np.int64)

    # Compose seam_parent -> elim_parent -> contiguous 1..N LUT
    final_roots = elim_parent[seam_parent]
    valid_roots = np.unique(final_roots[1:])
    valid_roots = valid_roots[valid_roots > 0]
    if min_pixels > 1 and valid_roots.size > 0:
        valid_roots = valid_roots[merged_counts[valid_roots] >= min_pixels]

    lut = np.zeros(total_raw_labels + 1, dtype=np.uint32)
    if valid_roots.size > 0:
        remap = np.zeros(total_raw_labels + 1, dtype=np.uint32)
        remap[valid_roots] = np.arange(
            1, valid_roots.size + 1, dtype=np.uint32
        )
        lut[1:] = remap[final_roots[1:]]
    return lut


def _array_tile_task(
    args: Tuple[
        TileWindow,
        np.ndarray,
        np.ndarray,
        Callable[..., Tuple[np.ndarray, Optional[np.ndarray]]],
        Dict[str, Any],
    ],
) -> Tuple[int, np.ndarray, Optional[np.ndarray]]:
    """Top-level picklable worker wrapper for array tile segmentation."""
    tile, halo_arr, halo_valid, worker_fn, params = args
    core_labels, aux_core = worker_fn(halo_arr, halo_valid, tile, params)
    return tile.tile_id, core_labels, aux_core


def _raster_tile_task(
    args: Tuple[
        TileWindow,
        str,
        List[int],
        Optional[float],
        Callable[..., Tuple[np.ndarray, Optional[np.ndarray]]],
        Dict[str, Any],
    ],
) -> Tuple[int, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Top-level picklable worker wrapper reading one halo tile from disk."""
    tile, input_img, bands, eff_nd, worker_fn, params = args
    halo_win = tile.halo_rio_window()
    with rasterio.open(input_img, "r") as src:
        halo_arr = src.read(bands, window=halo_win, out_dtype=np.float64)
    halo_valid = np.ones((tile.halo_height, tile.halo_width), dtype=bool)
    for b_i in range(len(bands)):
        halo_valid &= build_valid_mask(halo_arr[b_i], no_data_val=eff_nd)

    core_labels, aux_core = worker_fn(halo_arr, halo_valid, tile, params)
    core_arr = halo_arr[:, tile.core_row_slice, tile.core_col_slice]
    return tile.tile_id, core_labels, core_arr, aux_core


def run_tiled_array_pipeline(
    arr_3d: np.ndarray,
    valid_mask: np.ndarray,
    worker_fn: Callable[..., Tuple[np.ndarray, Optional[np.ndarray]]],
    params: Dict[str, Any],
    tile_size: Optional[Union[int, Tuple[int, int]]] = None,
    overlap: int = 0,
    n_jobs: int = 1,
    stitch_seams: bool = True,
    seam_mode: str = "spectral",
    seam_threshold: float = 1.0,
    min_pixels: int = 10,
    use_numba: bool = True,
    use_pbar: bool = False,
) -> np.ndarray:
    """
    Execute a segmentation worker over an in-memory ``(B, H, W)`` array using
    optional tiling with halo overlap, multi-core worker processes, seam
    reconciliation, and contiguous ``1..N`` relabelling.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    workers = validate_n_jobs(n_jobs)
    _, height, width = arr_3d.shape
    tiles = plan_tiles(height, width, tile_size=tile_size, overlap=overlap)

    if len(tiles) == 1:
        core_labels, _ = worker_fn(arr_3d, valid_mask, tiles[0], params)
        if min_pixels > 1:
            return eliminate_small_segments_array(
                core_labels,
                data=arr_3d,
                min_pixels=min_pixels,
                no_data_val=0,
                use_numba=use_numba,
            )
        return relabel_segments_array(
            core_labels, no_data_val=0, contiguous=True
        )

    tasks = [
        (
            tile,
            extract_halo_tile(arr_3d, tile),
            extract_halo_tile(valid_mask, tile),
            worker_fn,
            params,
        )
        for tile in tiles
    ]

    results_by_id: Dict[int, Tuple[np.ndarray, Optional[np.ndarray]]] = {}
    tqdm_cls = rsgispy.get_tqdm()
    if workers == 1:
        for task in tqdm_cls(tasks, disable=not use_pbar, desc="Segmenting tiles"):
            tid, c_lbl, c_aux = _array_tile_task(task)
            results_by_id[tid] = (c_lbl, c_aux)
    else:
        with ProcessPoolExecutor(
            max_workers=workers, initializer=worker_init_single_thread
        ) as executor:
            futs = [executor.submit(_array_tile_task, t) for t in tasks]
            for fut in tqdm_cls(
                as_completed(futs),
                total=len(futs),
                disable=not use_pbar,
                desc="Segmenting tiles",
            ):
                tid, c_lbl, c_aux = fut.result()
                results_by_id[tid] = (c_lbl, c_aux)

    out_labels = np.zeros((height, width), dtype=np.int64)
    records_by_grid: Dict[Tuple[int, int], _TileSeamRecord] = {}
    current_offset = 0
    for tile in tiles:
        c_lbl, c_aux = results_by_id[tile.tile_id]
        c_lbl_int = c_lbl.astype(np.int64, copy=False)
        shifted, new_max = offset_tile_labels(c_lbl_int, current_offset)
        r0 = tile.row_off
        r1 = tile.row_off + tile.height
        c0 = tile.col_off
        c1 = tile.col_off + tile.width
        out_labels[r0:r1, c0:c1] = shifted
        core_arr = arr_3d[:, r0:r1, c0:c1]
        records_by_grid[(tile.grid_row, tile.grid_col)] = (
            _collect_tile_seam_record(
                tile=tile,
                offset=current_offset,
                core_labels=c_lbl_int,
                core_arr=core_arr,
                core_aux=c_aux,
                seam_stds=params.get("seam_stds"),
            )
        )
        current_offset = new_max

    lut = _gather_seam_pairs_and_merges(
        records_by_grid=records_by_grid,
        total_raw_labels=current_offset,
        n_bands=arr_3d.shape[0],
        seam_mode=seam_mode,
        seam_threshold=seam_threshold,
        stitch_seams=stitch_seams,
        min_pixels=min_pixels,
        use_numba=use_numba,
    )
    mapped = lut[out_labels]
    return relabel_segments_array(mapped, no_data_val=0, contiguous=True)


def run_tiled_raster_pipeline(
    input_img: Union[str, os.PathLike],
    output_img: Union[str, os.PathLike],
    bands: Sequence[int],
    worker_fn: Callable[..., Tuple[np.ndarray, Optional[np.ndarray]]],
    params: Dict[str, Any],
    tile_size: Optional[Union[int, Tuple[int, int]]] = None,
    overlap: int = 0,
    n_jobs: int = 1,
    stitch_seams: bool = True,
    seam_mode: str = "spectral",
    seam_threshold: float = 1.0,
    min_pixels: int = 10,
    use_numba: bool = True,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Execute a file-backed tiled segmentation pipeline with bounded memory,
    optional multi-core tile execution, two-pass temporary GeoTIFF LUT
    remapping, and standard ``rsgispy`` statistics population.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-branches
    workers = validate_n_jobs(n_jobs)
    in_path = os.fspath(input_img)
    out_path = os.fspath(output_img)
    _, out_rio_dtype, out_np_dtype = resolve_label_dtype(datatype)

    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with rasterio.open(in_path, "r") as src:
        height = int(src.height)
        width = int(src.width)
        eff_nd = resolve_input_nodata(src, no_data_val)
        grid = IntersectionGrid.from_dataset(src)

    eff_tile_size = tile_size
    if eff_tile_size is None and (height * width) > _AUTO_SINGLE_PASS_MAX_PXLS:
        eff_tile_size = get_default_block_size()

    tiles = plan_tiles(height, width, tile_size=eff_tile_size, overlap=overlap)
    band_lst = [int(b) for b in bands]

    # Single-tile fast path
    if len(tiles) == 1:
        with rasterio.open(in_path, "r") as src:
            arr_3d = src.read(band_lst, out_dtype=np.float64)
        valid_mask = np.ones((height, width), dtype=bool)
        for b_i in range(len(band_lst)):
            valid_mask &= build_valid_mask(arr_3d[b_i], no_data_val=eff_nd)

        labels = run_tiled_array_pipeline(
            arr_3d=arr_3d,
            valid_mask=valid_mask,
            worker_fn=worker_fn,
            params=params,
            tile_size=None,
            overlap=0,
            n_jobs=1,
            stitch_seams=stitch_seams,
            seam_mode=seam_mode,
            seam_threshold=seam_threshold,
            min_pixels=min_pixels,
            use_numba=use_numba,
            use_pbar=False,
        )
        max_id = int(labels.max()) if labels.size > 0 else 0
        check_label_overflow(max_id, out_np_dtype)
        with ExitStack() as stack:
            dst = open_output_dataset(
                stack=stack,
                output_img=out_path,
                gdalformat=gdalformat,
                out_rio_dtype=out_rio_dtype,
                grid=grid,
                count=1,
            )
            dst.nodata = 0
            dst.write(labels.astype(out_np_dtype, copy=False), 1)
        if calc_stats:
            pop_img_stats(
                out_path,
                no_data_val=0,
                calc_pyramids=True,
                use_pbar=use_pbar,
            )
        return

    # Bounded-memory two-pass tiled streaming pipeline
    tasks = [
        (tile, in_path, band_lst, eff_nd, worker_fn, params) for tile in tiles
    ]
    tqdm_cls = rsgispy.get_tqdm()
    records_by_grid: Dict[Tuple[int, int], _TileSeamRecord] = {}
    current_offset = 0

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_raw_img = os.path.join(tmp_dir, "raw_tile_segs.tif")
        with ExitStack() as pass1_stack:
            tmp_dst = open_output_dataset(
                stack=pass1_stack,
                output_img=tmp_raw_img,
                gdalformat="GTIFF",
                out_rio_dtype="uint32",
                grid=grid,
                count=1,
            )
            tmp_dst.nodata = 0

            def _record_tile(
                tile_obj: TileWindow,
                c_lbl: np.ndarray,
                c_arr: np.ndarray,
                c_aux: Optional[np.ndarray],
            ) -> None:
                nonlocal current_offset
                c_lbl_int = c_lbl.astype(np.int64, copy=False)
                shifted, new_max = offset_tile_labels(
                    c_lbl_int, current_offset
                )
                tmp_dst.write(
                    shifted.astype(np.uint32, copy=False),
                    1,
                    window=tile_obj.core_rio_window(),
                )
                records_by_grid[(tile_obj.grid_row, tile_obj.grid_col)] = (
                    _collect_tile_seam_record(
                        tile=tile_obj,
                        offset=current_offset,
                        core_labels=c_lbl_int,
                        core_arr=c_arr,
                        core_aux=c_aux,
                        seam_stds=params.get("seam_stds"),
                    )
                )
                current_offset = new_max

            if workers == 1:
                for task in tqdm_cls(
                    tasks, disable=not use_pbar, desc="Segmenting tiles"
                ):
                    tid, c_lbl, c_arr, c_aux = _raster_tile_task(task)
                    _record_tile(tiles[tid], c_lbl, c_arr, c_aux)
            else:
                # Stream results into deterministic tile order to keep IDs identical to n_jobs=1
                pending: Dict[
                    int, Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]
                ] = {}
                next_tid = 0
                with ProcessPoolExecutor(
                    max_workers=workers,
                    initializer=worker_init_single_thread,
                ) as executor:
                    futs = [
                        executor.submit(_raster_tile_task, t)
                        for t in tasks
                    ]
                    for fut in tqdm_cls(
                        as_completed(futs),
                        total=len(futs),
                        disable=not use_pbar,
                        desc="Segmenting tiles",
                    ):
                        tid, c_lbl, c_arr, c_aux = fut.result()
                        pending[tid] = (c_lbl, c_arr, c_aux)
                        while next_tid in pending:
                            p_lbl, p_arr, p_aux = pending.pop(next_tid)
                            _record_tile(
                                tiles[next_tid], p_lbl, p_arr, p_aux
                            )
                            next_tid += 1

        lut = _gather_seam_pairs_and_merges(
            records_by_grid=records_by_grid,
            total_raw_labels=current_offset,
            n_bands=len(band_lst),
            seam_mode=seam_mode,
            seam_threshold=seam_threshold,
            stitch_seams=stitch_seams,
            min_pixels=min_pixels,
            use_numba=use_numba,
        )
        max_out_id = int(lut.max()) if lut.size > 0 else 0
        check_label_overflow(max_out_id, out_np_dtype)
        lut_cast = lut.astype(out_np_dtype, copy=False)

        with ExitStack() as pass2_stack:
            raw_src = pass2_stack.enter_context(rasterio.open(tmp_raw_img, "r"))
            dst = open_output_dataset(
                stack=pass2_stack,
                output_img=out_path,
                gdalformat=gdalformat,
                out_rio_dtype=out_rio_dtype,
                grid=grid,
                count=1,
            )
            dst.nodata = 0
            for win in iter_dataset_windows(
                height, width, use_pbar=use_pbar
            ):
                raw_blk = raw_src.read(1, window=win)
                dst.write(lut_cast[raw_blk], 1, window=win)

    if calc_stats:
        pop_img_stats(
            out_path,
            no_data_val=0,
            calc_pyramids=True,
            use_pbar=use_pbar,
        )
