"""
High-performance internal helpers for :mod:`rsgispy.segmentation`.

Implements an :math:`O(N)` horizontal run-length-encoded (RLE) 4-connected
component labelling engine backed by :mod:`scipy.sparse.csgraph` when available
and a ``numba``-accelerated (or pure-Python) RLE union-find fallback when
``scipy`` is not installed, together with a two-pass tiled streaming pipeline
for out-of-core rasters.
"""

from contextlib import ExitStack
from dataclasses import dataclass
import math
import os
import tempfile
from typing import List, Optional, Sequence, Tuple

import numpy as np
import rasterio
from rasterio.io import DatasetReader
from rasterio.windows import Window

import rsgispy
from rsgispy._numba import use_compiled
from rsgispy._unionfind import (
    uf_first_appearance_labels_kernel,
    uf_min_root_kernel,
)
from rsgispy.imageprocess import (
    IntersectionGrid,
    get_default_block_size,
    iter_dataset_windows,
    open_output_dataset,
)
from rsgispy.imageutils.stats import pop_img_stats

try:
    from scipy.sparse import coo_matrix  # pylint: disable=import-error
    from scipy.sparse.csgraph import connected_components  # pylint: disable=import-error

    HAVE_SCIPY = True
except ImportError:
    coo_matrix = None  # type: ignore[assignment]
    connected_components = None  # type: ignore[assignment]
    HAVE_SCIPY = False

_AUTO_IN_MEM_MAX_PXLS = 16_777_216


@dataclass
class BandSource:
    """Descriptor for a single 1-based raster band participating in clumping."""

    dataset: DatasetReader
    path: str
    band_idx: int


def _nodata_match_mask(arr: np.ndarray, no_data_val: float) -> np.ndarray:
    """Return boolean mask where pixels match ``no_data_val`` (supports NaN)."""
    if isinstance(no_data_val, float) and math.isnan(no_data_val):
        if np.issubdtype(arr.dtype, np.floating):
            return np.isnan(arr)
        return np.zeros(arr.shape, dtype=bool)
    return arr == no_data_val


def _rle_union_find_scanline(
    u_arr: np.ndarray,
    v_arr: np.ndarray,
    n_nodes: int,
) -> Tuple[int, np.ndarray]:
    """
    Solve connected components for nodes ``0 .. n_nodes - 1`` using
    union-find that preserves scanline root order: a ``numba``-compiled
    union-by-rank / path-halving kernel when available, otherwise
    :func:`_rle_union_find_scanline_py`.

    Returns ``(n_comp, labels)`` where ``labels`` has shape ``(n_nodes,)``
    with component IDs ``0 .. n_comp - 1`` in first-appearance order.
    """
    if not use_compiled(uf_min_root_kernel, "_rle_union_find_scanline"):
        return _rle_union_find_scanline_py(u_arr, v_arr, n_nodes)
    lut = uf_min_root_kernel(
        np.ascontiguousarray(u_arr, dtype=np.int64),
        np.ascontiguousarray(v_arr, dtype=np.int64),
        int(n_nodes),
        0,
    )
    n_comp, labels = uf_first_appearance_labels_kernel(lut)
    return int(n_comp), labels


def _rle_union_find_scanline_py(
    u_arr: np.ndarray,
    v_arr: np.ndarray,
    n_nodes: int,
) -> Tuple[int, np.ndarray]:
    """
    Solve connected components for nodes ``0 .. n_nodes - 1`` in pure Python
    using path-halving union-find that preserves scanline root order.

    Returns ``(n_comp, labels)`` where ``labels`` has shape ``(n_nodes,)``
    with component IDs ``0 .. n_comp - 1`` in first-appearance order.
    """
    parent = list(range(n_nodes))
    for a, b in zip(u_arr.tolist(), v_arr.tolist()):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        while parent[b] != b:
            parent[b] = parent[parent[b]]
            b = parent[b]
        if a != b:
            if a < b:
                parent[b] = a
            else:
                parent[a] = b

    labels = [0] * n_nodes
    next_id = 0
    for idx in range(n_nodes):
        root = parent[idx]
        if root == idx:
            labels[idx] = next_id
            next_id += 1
        else:
            # Because merges always attach larger roots to smaller roots,
            # root < idx always holds and labels[root] is already resolved.
            labels[idx] = labels[root]

    return next_id, np.asarray(labels, dtype=np.int64)


def _solve_graph_components(
    n_nodes: int,
    u_arr: np.ndarray,
    v_arr: np.ndarray,
) -> Tuple[int, np.ndarray]:
    """
    Find connected component labels ``0 .. n_comp - 1`` for edges ``(u_arr, v_arr)``
    using SciPy Cython ``csgraph.connected_components`` when available, or
    pure-Python path-halving union-find otherwise.
    """
    if n_nodes == 0:
        return 0, np.empty(0, dtype=np.int64)
    if u_arr.size == 0:
        return n_nodes, np.arange(n_nodes, dtype=np.int64)
    if HAVE_SCIPY:
        ones = np.ones(u_arr.size, dtype=np.int8)
        adj = coo_matrix((ones, (u_arr, v_arr)), shape=(n_nodes, n_nodes))
        n_comp, labels = connected_components(
            adj, directed=False, return_labels=True
        )
        return int(n_comp), np.asarray(labels, dtype=np.int64)
    return _rle_union_find_scanline(u_arr, v_arr, n_nodes)


def label_4connected(
    valid: np.ndarray,
    horiz_eq: np.ndarray,
    vert_eq: np.ndarray,
    return_first_flat: bool = False,
) -> Tuple[np.ndarray, int, np.ndarray]:
    """
    Assign scanline-ordered 4-connected clump IDs ``1 .. n_clumps`` (and ``0``
    where ``valid`` is ``False``) from precomputed adjacency masks.

    Returns ``(clump_img, n_clumps, rep_flat_idx)`` where ``rep_flat_idx``
    holds the first scanline flat pixel index per clump when
    ``return_first_flat=True`` (otherwise empty).
    """
    # pylint: disable=too-many-locals
    height, width = valid.shape
    total_pxls = height * width
    empty_rep = np.empty(0, dtype=np.int64)
    if not np.any(valid):
        return np.zeros((height, width), dtype=np.uint32), 0, empty_rep

    starts = np.ones((height, width), dtype=bool)
    if width > 1:
        starts[:, 1:] = ~horiz_eq
    valid_starts = starts & valid
    n_runs = int(np.count_nonzero(valid_starts))
    idx_dtype = np.int32 if total_pxls < 2_000_000_000 else np.int64
    run_id_grid = np.cumsum(valid_starts.ravel(), dtype=idx_dtype).reshape(
        height, width
    )
    run_id_grid -= 1
    run_start_flat = (
        np.flatnonzero(valid_starts) if return_first_flat else empty_rep
    )

    if height > 1 and np.any(vert_eq):
        if width > 1:
            dedup = vert_eq.copy()
            dedup[:, 1:] &= ~(
                vert_eq[:, :-1] & horiz_eq[:-1, :] & horiz_eq[1:, :]
            )
        else:
            dedup = vert_eq
        edge_rows, edge_cols = np.nonzero(dedup)
        u_arr = run_id_grid[edge_rows, edge_cols]
        v_arr = run_id_grid[edge_rows + 1, edge_cols]
    else:
        u_arr = np.empty(0, dtype=idx_dtype)
        v_arr = np.empty(0, dtype=idx_dtype)

    all_valid = int(np.count_nonzero(valid)) == total_pxls

    if u_arr.size == 0:
        if all_valid:
            out_img = (run_id_grid + 1).astype(np.uint32, copy=False)
        else:
            out_img = np.zeros((height, width), dtype=np.uint32)
            out_img[valid] = (run_id_grid[valid] + 1).astype(np.uint32)
        return out_img, n_runs, run_start_flat

    n_comp, comp_labels = _solve_graph_components(n_runs, u_arr, v_arr)

    # Canonicalise component labels into strict row-major first-appearance order.
    first_run = np.empty(n_comp, dtype=np.int64)
    rev_runs = np.arange(n_runs - 1, -1, -1, dtype=np.int64)
    first_run[comp_labels[::-1]] = rev_runs
    order = np.argsort(first_run, kind="mergesort")
    canon_lut = np.empty(n_comp, dtype=np.uint32)
    canon_lut[order] = np.arange(1, n_comp + 1, dtype=np.uint32)
    run_to_clump = canon_lut[comp_labels]
    rep_flat_idx = (
        run_start_flat[first_run[order]] if return_first_flat else empty_rep
    )

    if all_valid:
        out_img = run_to_clump[run_id_grid]
    else:
        out_img = np.zeros((height, width), dtype=np.uint32)
        out_img[valid] = run_to_clump[run_id_grid[valid]]

    return out_img, n_comp, rep_flat_idx


def _compute_band_adjacency(
    band_arrays: Sequence[np.ndarray],
    shape: Tuple[int, int],
    no_data_val: Optional[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ``(valid, horiz_eq, vert_eq)`` across one or more 2D band arrays of
    shape ``(height, width)``.
    """
    height, width = shape
    horiz_eq = (
        np.ones((height, width - 1), dtype=bool)
        if width > 1
        else np.empty((height, 0), dtype=bool)
    )
    vert_eq = (
        np.ones((height - 1, width), dtype=bool)
        if height > 1
        else np.empty((0, width), dtype=bool)
    )
    all_nodata = (
        np.ones((height, width), dtype=bool)
        if no_data_val is not None
        else None
    )

    for arr in band_arrays:
        if width > 1:
            horiz_eq &= arr[:, :-1] == arr[:, 1:]
        if height > 1:
            vert_eq &= arr[:-1, :] == arr[1:, :]
        if all_nodata is not None and no_data_val is not None:
            all_nodata &= _nodata_match_mask(arr, no_data_val)

    if all_nodata is not None:
        valid = ~all_nodata
        if width > 1:
            horiz_eq &= valid[:, :-1] & valid[:, 1:]
        if height > 1:
            vert_eq &= valid[:-1, :] & valid[1:, :]
    else:
        valid = np.ones((height, width), dtype=bool)

    return valid, horiz_eq, vert_eq


def _read_source_window(
    band_src: BandSource,
    grid: IntersectionGrid,
    out_win: Window,
) -> np.ndarray:
    """Read a window from ``band_src`` aligned to ``grid`` in its native dtype."""
    base_col, base_row = grid.offsets[band_src.path]
    in_win = Window(
        base_col + int(out_win.col_off),
        base_row + int(out_win.row_off),
        int(out_win.width),
        int(out_win.height),
    )
    return band_src.dataset.read(band_src.band_idx, window=in_win)


def _check_clump_id_overflow(n_clumps: int, out_np_dtype: np.dtype) -> None:
    """Raise if ``n_clumps`` exceeds the maximum value representable in ``out_np_dtype``."""
    if np.issubdtype(out_np_dtype, np.integer):
        max_id = int(np.iinfo(out_np_dtype).max)
        if n_clumps > max_id:
            raise rsgispy.RSGISPyException(
                f"Clump count ({n_clumps}) exceeds maximum representable value "
                f"({max_id}) for output datatype '{out_np_dtype}'."
            )


def _run_in_memory(
    band_sources: Sequence[BandSource],
    grid: IntersectionGrid,
    output_img: str,
    gdalformat: str,
    no_data_val: Optional[float],
    out_rio_dtype: str = "uint32",
) -> None:
    """
    Execute single-pass vectorised 4-connected clumping in memory, streaming
    bands into adjacency masks one band at a time to minimise peak RAM.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    full_win = Window(0, 0, grid.width, grid.height)
    shape = (grid.height, grid.width)

    horiz_eq = (
        np.ones((grid.height, grid.width - 1), dtype=bool)
        if grid.width > 1
        else np.empty((grid.height, 0), dtype=bool)
    )
    vert_eq = (
        np.ones((grid.height - 1, grid.width), dtype=bool)
        if grid.height > 1
        else np.empty((0, grid.width), dtype=bool)
    )
    all_nodata = np.ones(shape, dtype=bool) if no_data_val is not None else None

    for b_src in band_sources:
        arr = _read_source_window(b_src, grid, full_win)
        if grid.width > 1:
            horiz_eq &= arr[:, :-1] == arr[:, 1:]
        if grid.height > 1:
            vert_eq &= arr[:-1, :] == arr[1:, :]
        if all_nodata is not None and no_data_val is not None:
            all_nodata &= _nodata_match_mask(arr, no_data_val)
        del arr

    if all_nodata is not None:
        valid = ~all_nodata
        if grid.width > 1:
            horiz_eq &= valid[:, :-1] & valid[:, 1:]
        if grid.height > 1:
            vert_eq &= valid[:-1, :] & valid[1:, :]
    else:
        valid = np.ones(shape, dtype=bool)

    out_img, n_clumps, _ = label_4connected(
        valid, horiz_eq, vert_eq, return_first_flat=False
    )
    out_np_dtype = np.dtype(out_rio_dtype)
    _check_clump_id_overflow(n_clumps, out_np_dtype)
    if out_img.dtype != out_np_dtype:
        out_img = out_img.astype(out_np_dtype)

    with ExitStack() as out_stack:
        dst = open_output_dataset(
            stack=out_stack,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype=out_rio_dtype,
            grid=grid,
            count=1,
        )
        dst.write(out_img, 1)


def _run_tiled(
    band_sources: Sequence[BandSource],
    grid: IntersectionGrid,
    output_img: str,
    gdalformat: str,
    no_data_val: Optional[float],
    block_size: int,
    use_pbar: bool,
    out_rio_dtype: str = "uint32",
) -> None:
    """
    Execute memory-bounded two-pass tiled 4-connected clumping.

    Pass 1 labels each tile in memory via :func:`label_4connected`, writes raw
    offset IDs to a temporary tiled GeoTIFF, and records cross-tile seam merges
    plus per-clump first-pixel positions. Pass 2 streams tiles through a
    single global lookup table so the output matches :func:`_run_in_memory`
    identically.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    # pylint: disable=too-many-branches,too-many-statements
    n_bands = len(band_sources)
    out_np_dtype = np.dtype(out_rio_dtype)
    u32_max = int(np.iinfo(np.uint32).max)

    with tempfile.TemporaryDirectory(prefix="rsgispy_clump_") as tmp_dir:
        tmp_raw_img = os.path.join(tmp_dir, "raw_clumps.tif")
        tile_dim = max(16, min(block_size, 512))
        tile_dim = ((tile_dim + 15) // 16) * 16
        tmp_opts = {
            "TILED": "YES",
            "BLOCKXSIZE": tile_dim,
            "BLOCKYSIZE": tile_dim,
            "BIGTIFF": "IF_NEEDED",
        }

        prev_row_ids = np.zeros(grid.width, dtype=np.uint32)
        prev_row_vals: List[Optional[np.ndarray]] = [None] * n_bands
        prev_col_ids = np.empty(0, dtype=np.uint32)
        prev_col_vals: List[np.ndarray] = []

        total_raw = 0
        raw_first_pos_chunks: List[np.ndarray] = []
        merge_u_chunks: List[np.ndarray] = []
        merge_v_chunks: List[np.ndarray] = []

        # Pass 1: Label tiles independently and collect seam merge edges.
        with ExitStack() as pass1_stack:
            tmp_dst = open_output_dataset(
                stack=pass1_stack,
                output_img=tmp_raw_img,
                gdalformat="GTIFF",
                out_rio_dtype="uint32",
                grid=grid,
                count=1,
                creation_opts=tmp_opts,
            )

            for out_win in iter_dataset_windows(
                grid.height,
                grid.width,
                block_size=block_size,
                use_pbar=use_pbar,
            ):
                col_off = int(out_win.col_off)
                row_off = int(out_win.row_off)
                win_w = int(out_win.width)
                win_h = int(out_win.height)
                if col_off == 0:
                    prev_col_ids = np.empty(0, dtype=np.uint32)
                    prev_col_vals = []

                tile_bands = [
                    _read_source_window(b_src, grid, out_win)
                    for b_src in band_sources
                ]
                valid, horiz_eq, vert_eq = _compute_band_adjacency(
                    tile_bands, (win_h, win_w), no_data_val
                )
                local_out, k_clumps, rep_local_flat = label_4connected(
                    valid, horiz_eq, vert_eq, return_first_flat=True
                )

                if k_clumps > 0:
                    if total_raw + k_clumps > u32_max:
                        raise rsgispy.RSGISPyException(
                            f"Intermediate tiled clump count ({total_raw + k_clumps}) "
                            f"exceeds maximum uint32 value ({u32_max})."
                        )
                    global_out = np.where(
                        local_out > 0, local_out + total_raw, np.uint32(0)
                    )
                    ly = rep_local_flat // win_w
                    lx = rep_local_flat % win_w
                    global_first = (row_off + ly) * grid.width + (col_off + lx)
                    raw_first_pos_chunks.append(global_first.astype(np.int64))
                    total_raw += k_clumps
                else:
                    global_out = local_out

                tmp_dst.write(global_out, 1, window=out_win)

                col_end = col_off + win_w
                if row_off > 0:
                    top_ids = global_out[0, :]
                    above_ids = prev_row_ids[col_off:col_end]
                    seam = (top_ids > 0) & (above_ids > 0)
                    for b_i, arr in enumerate(tile_bands):
                        p_row = prev_row_vals[b_i]
                        if p_row is None:
                            raise rsgispy.RSGISPyException(
                                f"Missing previous-row buffer for band {b_i} "
                                "during tiled clumping."
                            )
                        seam &= arr[0, :] == p_row[col_off:col_end]
                    if np.any(seam):
                        merge_u_chunks.append(above_ids[seam].astype(np.int64))
                        merge_v_chunks.append(top_ids[seam].astype(np.int64))

                if col_off > 0 and prev_col_ids.size > 0:
                    cur_left_ids = global_out[:, 0]
                    seam_col = (cur_left_ids > 0) & (prev_col_ids > 0)
                    for b_i, arr in enumerate(tile_bands):
                        seam_col &= arr[:, 0] == prev_col_vals[b_i]
                    if np.any(seam_col):
                        merge_u_chunks.append(
                            prev_col_ids[seam_col].astype(np.int64)
                        )
                        merge_v_chunks.append(
                            cur_left_ids[seam_col].astype(np.int64)
                        )

                prev_row_ids[col_off:col_end] = global_out[-1, :]
                for b_i, arr in enumerate(tile_bands):
                    if prev_row_vals[b_i] is None:
                        prev_row_vals[b_i] = np.empty(
                            grid.width, dtype=arr.dtype
                        )
                    prev_row_vals[b_i][col_off:col_end] = arr[-1, :]

                prev_col_ids = global_out[:, -1].copy()
                prev_col_vals = [arr[:, -1].copy() for arr in tile_bands]

        # Resolve global equivalences and canonicalise to scanline order.
        if total_raw == 0:
            with ExitStack() as out_stack:
                dst = open_output_dataset(
                    stack=out_stack,
                    output_img=output_img,
                    gdalformat=gdalformat,
                    out_rio_dtype=out_rio_dtype,
                    grid=grid,
                    count=1,
                )
                for out_win in iter_dataset_windows(
                    grid.height,
                    grid.width,
                    block_size=block_size,
                    use_pbar=use_pbar,
                ):
                    zeros = np.zeros(
                        (int(out_win.height), int(out_win.width)),
                        dtype=out_np_dtype,
                    )
                    dst.write(zeros, 1, window=out_win)
            return

        raw_first_pos = np.concatenate(raw_first_pos_chunks)
        if merge_u_chunks:
            u_all = np.concatenate(merge_u_chunks) - 1
            v_all = np.concatenate(merge_v_chunks) - 1
        else:
            u_all = np.empty(0, dtype=np.int64)
            v_all = np.empty(0, dtype=np.int64)

        n_comp, raw_comp_labels = _solve_graph_components(
            total_raw, u_all, v_all
        )
        _check_clump_id_overflow(n_comp, out_np_dtype)
        raw_order = np.argsort(raw_first_pos, kind="mergesort")
        comp_rep_raw = np.empty(n_comp, dtype=np.int64)
        comp_rep_raw[raw_comp_labels[raw_order[::-1]]] = raw_order[::-1]
        comp_order = np.argsort(raw_first_pos[comp_rep_raw], kind="mergesort")

        canon_comp = np.empty(n_comp, dtype=np.uint32)
        canon_comp[comp_order] = np.arange(1, n_comp + 1, dtype=np.uint32)

        lut = np.zeros(total_raw + 1, dtype=np.uint32)
        lut[1:] = canon_comp[raw_comp_labels]
        if lut.dtype != out_np_dtype:
            lut = lut.astype(out_np_dtype)

        # Pass 2: Remap raw tile IDs through global LUT into final output raster.
        with ExitStack() as pass2_stack:
            raw_src = pass2_stack.enter_context(
                rasterio.open(tmp_raw_img, "r")
            )
            dst = open_output_dataset(
                stack=pass2_stack,
                output_img=output_img,
                gdalformat=gdalformat,
                out_rio_dtype=out_rio_dtype,
                grid=grid,
                count=1,
            )
            for out_win in iter_dataset_windows(
                grid.height,
                grid.width,
                block_size=block_size,
                use_pbar=use_pbar,
            ):
                raw_block = raw_src.read(1, window=out_win)
                dst.write(lut[raw_block], 1, window=out_win)


def run_clump_pipeline(
    band_sources: Sequence[BandSource],
    grid: IntersectionGrid,
    output_img: str,
    gdalformat: str,
    in_memory: Optional[bool],
    no_data_val: Optional[float],
    block_size: Optional[int] = None,
    datatype: Optional[int] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Dispatch 4-connected clumping between fast single-pass in-memory execution
    and memory-bounded two-pass tiled execution.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    out_dir = os.path.dirname(os.path.abspath(output_img))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    blk_sz = (
        get_default_block_size() if block_size is None else int(block_size)
    )
    if blk_sz <= 0:
        raise rsgispy.RSGISPyException(
            f"block_size must be a positive integer, got {block_size}."
        )

    eff_dtype = rsgispy.TYPE_32UINT if datatype is None else datatype
    out_rio_dtype = rsgispy.get_rasterio_datatype(eff_dtype)

    fits_single_tile = grid.width <= blk_sz and grid.height <= blk_sz
    if in_memory is None:
        use_in_mem = (
            fits_single_tile
            or (grid.width * grid.height) <= _AUTO_IN_MEM_MAX_PXLS
        )
    else:
        use_in_mem = bool(in_memory) or fits_single_tile

    if use_in_mem:
        _run_in_memory(
            band_sources=band_sources,
            grid=grid,
            output_img=output_img,
            gdalformat=gdalformat,
            no_data_val=no_data_val,
            out_rio_dtype=out_rio_dtype,
        )
    else:
        _run_tiled(
            band_sources=band_sources,
            grid=grid,
            output_img=output_img,
            gdalformat=gdalformat,
            no_data_val=no_data_val,
            block_size=blk_sz,
            use_pbar=use_pbar,
            out_rio_dtype=out_rio_dtype,
        )

    if calc_stats:
        pop_img_stats(
            output_img,
            no_data_val=0,
            calc_pyramids=True,
            use_pbar=use_pbar,
        )
