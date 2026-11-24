"""
Internal helpers for classification raster post-processing (majority modal
filtering and connected-component sieve clump elimination).
"""

from contextlib import ExitStack
from dataclasses import dataclass
import os
import tempfile
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine

import rsgispy
from rsgispy._io import (
    DiskSink,
    HaloBorder,
    MemorySink,
    RasterSource,
    TileGraph,
    TileNode,
    WindowOp,
)
from rsgispy.imagefilter._helper import (
    _apply_tile_nd,
    _resolve_out_nodata,
    stat_mode,
    validate_filter_size,
)
from rsgispy.imageprocess import (
    _INT_DTYPES,
    IntersectionGrid,
    build_valid_mask,
    cast_result_block,
    ensure_file_exists,
    get_default_block_size,
    iter_dataset_windows,
    open_output_dataset,
    run_window_halo_blocks,
    validate_band_index,
)
from rsgispy.imageprocess import normalise_creation_options
from rsgispy.imageutils.colourtable import define_colour_table, set_img_thematic
from rsgispy.imageutils.metadata import set_band_names, set_img_no_data_value
from rsgispy.imageutils.stats import build_img_pyramids, pop_thmt_img_stats
from rsgispy.segmentation._helpers import (
    _AUTO_IN_MEM_MAX_PXLS,
    _solve_graph_components,
    label_4connected,
)
from rsgispy.tools.files import ensure_parent_dir


@dataclass
class PostprocessMeta:
    """Resolved input/output raster metadata for classification post-processing."""

    # pylint: disable=too-many-instance-attributes
    width: int
    height: int
    src_count: int
    crs: Any
    transform: Affine
    out_rio_dtype: str
    out_np_dtype: np.dtype
    eff_in_nd: Optional[float]
    out_nd: Optional[float]
    band_desc: str
    clr_lut: Optional[Dict[int, List[int]]]


def resolve_postprocess_io_meta(
    in_path: str,
    img_band: int,
    datatype: Optional[int],
    no_data_val: Optional[float],
    copy_colour_table: bool,
) -> PostprocessMeta:
    """Inspect ``in_path`` and resolve output dtype, nodata, and colour table."""
    ensure_file_exists(in_path, "Input image")
    if isinstance(img_band, bool) or not isinstance(img_band, (int, np.integer)):
        raise rsgispy.RSGISPyException(
            f"img_band must be an integer, got {img_band!r}."
        )
    band_idx = int(img_band)

    with rasterio.open(in_path, "r") as src:
        validate_band_index(band_idx, src.count, in_path)
        in_rio_dtype = src.dtypes[band_idx - 1]
        if datatype is None:
            out_rio_dtype = in_rio_dtype
            out_np_dtype = np.dtype(in_rio_dtype)
        else:
            out_rio_dtype = rsgispy.get_rasterio_datatype(datatype)
            out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(datatype))

        src_nd = src.nodatavals[band_idx - 1]
        eff_in_nd = (
            float(no_data_val)
            if no_data_val is not None
            else (float(src_nd) if src_nd is not None else None)
        )
        out_nd = _resolve_out_nodata(eff_in_nd, out_np_dtype)
        band_desc = src.descriptions[band_idx - 1] or ""

        clr_lut: Optional[Dict[int, List[int]]] = None
        if copy_colour_table:
            try:
                cmap = src.colormap(band_idx)
            except ValueError:
                cmap = None
            if cmap:
                clr_lut = {
                    int(k): (
                        [int(v[0]), int(v[1]), int(v[2]), int(v[3])]
                        if len(v) >= 4
                        else [int(v[0]), int(v[1]), int(v[2]), 255]
                    )
                    for k, v in cmap.items()
                }

        return PostprocessMeta(
            width=int(src.width),
            height=int(src.height),
            src_count=int(src.count),
            crs=src.crs,
            transform=src.transform,
            out_rio_dtype=out_rio_dtype,
            out_np_dtype=out_np_dtype,
            eff_in_nd=eff_in_nd,
            out_nd=out_nd,
            band_desc=band_desc,
            clr_lut=clr_lut,
        )


def finalise_postprocess_output(
    out_path: str,
    meta: PostprocessMeta,
    calc_stats: bool,
    use_pbar: bool,
) -> None:
    """Apply band description, thematic tags, stats/pyramids, and colour table."""
    if meta.band_desc:
        set_band_names(out_path, [meta.band_desc], feedback=False)

    set_img_thematic(out_path)
    if meta.out_nd is not None:
        set_img_no_data_value(out_path, meta.out_nd, img_band=1)

    can_ctab = meta.out_np_dtype in (
        np.dtype(np.uint8),
        np.dtype(np.uint16),
    )
    if calc_stats:
        if meta.out_rio_dtype in _INT_DTYPES:
            pop_thmt_img_stats(
                out_path,
                add_clr_tab=bool(can_ctab and meta.clr_lut is None),
                calc_pyramids=True,
                ignore_zero=(meta.out_nd == 0),
                use_pbar=use_pbar,
            )
        else:
            build_img_pyramids(out_path, resampling_method=Resampling.nearest)
        if meta.out_nd is not None:
            set_img_no_data_value(out_path, meta.out_nd, img_band=1)

    if meta.clr_lut is not None and can_ctab:
        define_colour_table(out_path, meta.clr_lut, img_band=1)


def run_majority_filter(
    in_path: str,
    out_path: str,
    filter_size: int,
    img_band: int,
    datatype: Optional[int],
    no_data_val: Optional[float],
    keep_centre_on_tie: bool,
    gdalformat: str,
    calc_stats: bool,
    copy_colour_table: bool,
    creation_opts: Optional[Mapping[str, Any]],
    use_pbar: bool,
) -> None:
    """Execute moving-window modal majority filtering on a classification band."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    size = validate_filter_size(filter_size)
    meta = resolve_postprocess_io_meta(
        in_path=in_path,
        img_band=img_band,
        datatype=datatype,
        no_data_val=no_data_val,
        copy_colour_table=copy_colour_table,
    )
    ensure_parent_dir(out_path)
    norm_opts = normalise_creation_options(creation_opts)

    pad = size // 2
    keep_tie = bool(keep_centre_on_tie)

    def _tile_cb(halo: np.ndarray) -> np.ndarray:
        win_h = halo.shape[0] - 2 * pad
        win_w = halo.shape[1] - 2 * pad
        y_end = pad + win_h
        x_end = pad + win_w

        valid_halo = build_valid_mask(halo, meta.eff_in_nd)
        centre = halo[pad:y_end, pad:x_end]
        centre_valid = valid_halo[pad:y_end, pad:x_end]
        wins = sliding_window_view(halo, (size, size))
        v_wins = sliding_window_view(valid_halo, (size, size))

        res = stat_mode(wins, centre, v_wins, centre_valid)
        if keep_tie:
            diff_mask = centre_valid & np.isfinite(res) & (res != centre)
            if np.any(diff_mask):
                w_sub = wins[diff_mask]
                v_sub = v_wins[diff_mask]
                m_cnt = np.sum(
                    v_sub & (w_sub == res[diff_mask][:, None, None]),
                    axis=(1, 2),
                )
                c_cnt = np.sum(
                    v_sub & (w_sub == centre[diff_mask][:, None, None]),
                    axis=(1, 2),
                )
                tied_with_centre = c_cnt >= m_cnt
                if np.any(tied_with_centre):
                    upd = res[diff_mask]
                    upd[tied_with_centre] = centre[diff_mask][tied_with_centre]
                    res[diff_mask] = upd

        return _apply_tile_nd(res, meta.out_nd, meta.out_np_dtype)

    run_window_halo_blocks(
        input_img=in_path,
        output_img=out_path,
        win_shape=(size, size),
        block_fn=_tile_cb,
        gdalformat=gdalformat,
        out_rio_dtype=meta.out_rio_dtype,
        out_np_dtype=meta.out_np_dtype,
        out_count=1,
        per_band=False,
        read_dtype=np.float64,
        pad_mode="constant",
        fill_value=np.nan,
        out_no_data_val=meta.out_nd,
        clamp_int=True,
        use_pbar=use_pbar,
        creation_opts=norm_opts,
        band_idx=int(img_band),
    )

    finalise_postprocess_output(
        out_path=out_path,
        meta=meta,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )


def _label_connected_2d(
    arr: np.ndarray,
    eff_in_nd: Optional[float],
    connectivity: int,
) -> Tuple[np.ndarray, int, np.ndarray]:
    """
    Label 4- or 8-connected components of equal valid class values in ``arr``.

    Returns ``(clump_img, n_clumps, rep_flat_idx)`` with ``uint32`` IDs
    ``1 .. n_clumps`` (and ``0`` where pixels are invalid/nodata).
    """
    # pylint: disable=too-many-locals
    valid = build_valid_mask(arr, eff_in_nd)
    height, width = arr.shape
    horiz_eq = (
        (arr[:, :-1] == arr[:, 1:]) & valid[:, :-1] & valid[:, 1:]
        if width > 1
        else np.empty((height, 0), dtype=bool)
    )
    vert_eq = (
        (arr[:-1, :] == arr[1:, :]) & valid[:-1, :] & valid[1:, :]
        if height > 1
        else np.empty((0, width), dtype=bool)
    )

    if connectivity == 4:
        return label_4connected(valid, horiz_eq, vert_eq, return_first_flat=True)

    # 8-connectivity using horizontal RLE + vertical & diagonal edges.
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
    run_start_flat = np.flatnonzero(valid_starts)

    u_chunks: List[np.ndarray] = []
    v_chunks: List[np.ndarray] = []

    if height > 1 and np.any(vert_eq):
        if width > 1:
            dedup = vert_eq.copy()
            dedup[:, 1:] &= ~(
                vert_eq[:, :-1] & horiz_eq[:-1, :] & horiz_eq[1:, :]
            )
        else:
            dedup = vert_eq
        er, ec = np.nonzero(dedup)
        u_chunks.append(run_id_grid[er, ec])
        v_chunks.append(run_id_grid[er + 1, ec])

    if height > 1 and width > 1:
        dr_eq = valid[:-1, :-1] & valid[1:, 1:] & (arr[:-1, :-1] == arr[1:, 1:])
        dr_eq &= ~(vert_eq[:, :-1] & horiz_eq[1:, :]) & ~(
            horiz_eq[:-1, :] & vert_eq[:, 1:]
        )
        if np.any(dr_eq):
            er, ec = np.nonzero(dr_eq)
            u_chunks.append(run_id_grid[er, ec])
            v_chunks.append(run_id_grid[er + 1, ec + 1])

        dl_eq = valid[:-1, 1:] & valid[1:, :-1] & (arr[:-1, 1:] == arr[1:, :-1])
        dl_eq &= ~(horiz_eq[:-1, :] & vert_eq[:, :-1]) & ~(
            vert_eq[:, 1:] & horiz_eq[1:, :]
        )
        if np.any(dl_eq):
            er, ec = np.nonzero(dl_eq)
            u_chunks.append(run_id_grid[er, ec + 1])
            v_chunks.append(run_id_grid[er + 1, ec])

    all_valid = int(np.count_nonzero(valid)) == total_pxls
    if not u_chunks:
        if all_valid:
            out_img = (run_id_grid + 1).astype(np.uint32, copy=False)
        else:
            out_img = np.zeros((height, width), dtype=np.uint32)
            out_img[valid] = (run_id_grid[valid] + 1).astype(np.uint32)
        return out_img, n_runs, run_start_flat

    u_arr = np.concatenate(u_chunks) if len(u_chunks) > 1 else u_chunks[0]
    v_arr = np.concatenate(v_chunks) if len(v_chunks) > 1 else v_chunks[0]
    n_comp, comp_labels = _solve_graph_components(n_runs, u_arr, v_arr)

    first_run = np.empty(n_comp, dtype=np.int64)
    rev_runs = np.arange(n_runs - 1, -1, -1, dtype=np.int64)
    first_run[comp_labels[::-1]] = rev_runs
    order = np.argsort(first_run, kind="mergesort")
    canon_lut = np.empty(n_comp, dtype=np.uint32)
    canon_lut[order] = np.arange(1, n_comp + 1, dtype=np.uint32)
    run_to_clump = canon_lut[comp_labels]
    rep_flat_idx = run_start_flat[first_run[order]]

    if all_valid:
        out_img = run_to_clump[run_id_grid]
    else:
        out_img = np.zeros((height, width), dtype=np.uint32)
        out_img[valid] = run_to_clump[run_id_grid[valid]]

    return out_img, n_comp, rep_flat_idx


def _append_pair_edges(
    a: np.ndarray,
    b: np.ndarray,
    u_list: List[np.ndarray],
    v_list: List[np.ndarray],
) -> None:
    """Append undirected cross-clump edges ``(a, b)`` where both > 0 and ``a != b``."""
    mask = (a > 0) & (b > 0) & (a != b)
    if np.any(mask):
        u_list.append(a[mask])
        v_list.append(b[mask])


def _extract_interior_half_edges(
    clump_img: np.ndarray,
    connectivity: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract undirected cross-clump edges inside a 2D clump array."""
    height, width = clump_img.shape
    u_list: List[np.ndarray] = []
    v_list: List[np.ndarray] = []

    if width > 1:
        _append_pair_edges(clump_img[:, :-1], clump_img[:, 1:], u_list, v_list)
    if height > 1:
        _append_pair_edges(clump_img[:-1, :], clump_img[1:, :], u_list, v_list)
    if connectivity == 8 and height > 1 and width > 1:
        _append_pair_edges(clump_img[:-1, :-1], clump_img[1:, 1:], u_list, v_list)
        _append_pair_edges(clump_img[:-1, 1:], clump_img[1:, :-1], u_list, v_list)

    if not u_list:
        empty = np.empty(0, dtype=np.uint32)
        return empty, empty
    return np.concatenate(u_list), np.concatenate(v_list)


def _aggregate_directed_edges(
    edge_u: np.ndarray,
    edge_v: np.ndarray,
    edge_w: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sum weights of duplicate directed ``(u, v)`` boundary edges."""
    if edge_u.size == 0:
        return edge_u, edge_v, edge_w
    keys = (edge_u.astype(np.uint64) << 32) | edge_v.astype(np.uint64)
    order = np.argsort(keys, kind="mergesort")
    k_s = keys[order]
    w_s = edge_w[order]

    is_new = np.empty(k_s.size, dtype=bool)
    is_new[0] = True
    is_new[1:] = k_s[1:] != k_s[:-1]
    starts = np.flatnonzero(is_new)
    w_sum = np.add.reduceat(w_s, starts)
    uniq_k = k_s[starts]
    return (
        (uniq_k >> 32).astype(np.int64),
        (uniq_k & 0xFFFFFFFF).astype(np.int64),
        w_sum.astype(np.int64),
    )


def _build_sym_weighted_edges(
    u_half: Sequence[np.ndarray],
    v_half: Sequence[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert undirected half-edge lists into aggregated directed weighted edges."""
    if not u_half:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, empty
    u_cat = np.concatenate(u_half) if len(u_half) > 1 else u_half[0]
    v_cat = np.concatenate(v_half) if len(v_half) > 1 else v_half[0]
    if u_cat.size == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, empty
    u_dir = np.concatenate([u_cat, v_cat])
    v_dir = np.concatenate([v_cat, u_cat])
    ones = np.ones(u_dir.size, dtype=np.int64)
    return _aggregate_directed_edges(u_dir, v_dir, ones)


def _select_majority_neighbours(
    u_sel: np.ndarray,
    v_sel: np.ndarray,
    w_sel: np.ndarray,
    clump_classes: np.ndarray,
    cur_sizes: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each source clump in ``u_sel``, choose the replacement clump ``target_v``
    belonging to the neighbouring class with maximum total boundary contact
    weight (tie-breaking by largest neighbour clump size, then smallest class ID).
    """
    # pylint: disable=too-many-locals
    cls_sel = clump_classes[v_sel]
    sz_sel = cur_sizes[v_sel]

    # Sort by (u_sel, cls_sel, -sz_sel, v_sel) so first row per (u, cls) has max size.
    order = np.lexsort((v_sel, -sz_sel, cls_sel, u_sel))
    u_s = u_sel[order]
    v_s = v_sel[order]
    c_s = cls_sel[order]
    w_s = w_sel[order]

    is_new = np.empty(u_s.size, dtype=bool)
    is_new[0] = True
    is_new[1:] = (u_s[1:] != u_s[:-1]) | (c_s[1:] != c_s[:-1])
    starts = np.flatnonzero(is_new)

    grp_u = u_s[starts]
    grp_cls = c_s[starts]
    grp_rep_v = v_s[starts]
    grp_rep_sz = cur_sizes[grp_rep_v]
    grp_w = np.add.reduceat(w_s, starts)

    win_order = np.lexsort((grp_rep_v, grp_cls, -grp_rep_sz, -grp_w, grp_u))
    u_w = grp_u[win_order]
    first_u = np.empty(u_w.size, dtype=bool)
    first_u[0] = True
    first_u[1:] = u_w[1:] != u_w[:-1]
    winners = win_order[first_u]
    return grp_u[winners], grp_rep_v[winners]


def _solve_sieve_merges(
    clump_sizes: np.ndarray,
    clump_classes: np.ndarray,
    edge_u: np.ndarray,
    edge_v: np.ndarray,
    edge_w: np.ndarray,
    size_threshold: int,
) -> np.ndarray:
    """
    Iteratively merge clumps smaller than ``size_threshold`` into their
    surrounding majority neighbour and return ``final_class_lut`` indexed by
    initial clump ID ``1 .. n_clumps``.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    n_clumps = clump_sizes.size - 1
    if n_clumps <= 0 or size_threshold <= 1 or edge_u.size == 0:
        return clump_classes.copy()

    cur_sizes = clump_sizes.astype(np.int64, copy=True)
    is_kept = cur_sizes >= size_threshold
    is_kept[0] = False
    is_small = (cur_sizes > 0) & (cur_sizes < size_threshold)
    is_small[0] = False

    parent = np.arange(n_clumps + 1, dtype=np.int64)
    cur_u = edge_u
    cur_v = edge_v
    cur_w = edge_w

    while np.any(is_small) and cur_u.size > 0:
        kept_mask = is_small[cur_u] & is_kept[cur_v]
        if np.any(kept_mask):
            target_u, target_v = _select_majority_neighbours(
                cur_u[kept_mask],
                cur_v[kept_mask],
                cur_w[kept_mask],
                clump_classes,
                cur_sizes,
            )
            step_lut = np.arange(n_clumps + 1, dtype=np.int64)
            step_lut[target_u] = target_v
            parent[target_u] = target_v
            np.add.at(cur_sizes, target_v, cur_sizes[target_u])
            cur_sizes[target_u] = 0
            is_small[target_u] = False
        else:
            # Fallback when remaining small clumps have no kept neighbours
            # (e.g., an isolated region where every component is < threshold).
            valid_nbr = is_small[cur_u] & (cur_sizes[cur_v] > 0)
            if not np.any(valid_nbr):
                break
            cand_u = np.unique(cur_u[valid_nbr])
            u_min = int(cand_u[np.argmin(cur_sizes[cand_u])])
            u_edges = valid_nbr & (cur_u == u_min)
            _, rep_arr = _select_majority_neighbours(
                cur_u[u_edges],
                cur_v[u_edges],
                cur_w[u_edges],
                clump_classes,
                cur_sizes,
            )
            rep_v = int(rep_arr[0])
            step_lut = np.arange(n_clumps + 1, dtype=np.int64)
            step_lut[u_min] = rep_v
            parent[u_min] = rep_v
            cur_sizes[rep_v] += cur_sizes[u_min]
            cur_sizes[u_min] = 0
            is_small[u_min] = False
            if cur_sizes[rep_v] >= size_threshold:
                is_kept[rep_v] = True
                is_small[rep_v] = False

        cur_u = step_lut[cur_u]
        cur_v = step_lut[cur_v]
        keep_edges = cur_u != cur_v
        if not np.any(keep_edges):
            break
        cur_u, cur_v, cur_w = _aggregate_directed_edges(
            cur_u[keep_edges], cur_v[keep_edges], cur_w[keep_edges]
        )

    # Path compression to resolve multi-hop merges to final root clump.
    root = parent.copy()
    while True:
        nxt = root[root]
        if np.array_equal(nxt, root):
            break
        root = nxt

    return clump_classes[root]


def _fill_nodata_value(meta: PostprocessMeta) -> float:
    """Resolve fill value for invalid/nodata pixels in sieve outputs."""
    if meta.out_nd is not None:
        return meta.out_nd
    if np.issubdtype(meta.out_np_dtype, np.floating):
        return float("nan")
    return 0


def _sieve_array_to_disk(
    arr: np.ndarray,
    grid: IntersectionGrid,
    out_path: str,
    size_threshold: int,
    connectivity: int,
    gdalformat: str,
    meta: PostprocessMeta,
    eff_in_nd: Optional[float],
    creation_opts: Optional[Dict[str, str]],
) -> None:
    """Sieve an in-memory 2D classification array and write it to ``out_path``."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    clump_img, n_clumps, rep_flat = _label_connected_2d(
        arr, eff_in_nd, connectivity
    )
    with ExitStack() as stack:
        dst = open_output_dataset(
            stack=stack,
            output_img=out_path,
            gdalformat=gdalformat,
            out_rio_dtype=meta.out_rio_dtype,
            grid=grid,
            count=1,
            creation_opts=creation_opts,
        )
        if meta.out_nd is not None:
            dst.nodata = meta.out_nd

        if size_threshold <= 1 or n_clumps == 0:
            if n_clumps == 0:
                out_arr = np.full(
                    arr.shape, _fill_nodata_value(meta), dtype=meta.out_np_dtype
                )
            else:
                out_arr = arr
            dst.write(
                cast_result_block(
                    out_arr, meta.height, meta.width, meta.out_np_dtype
                ),
                1,
            )
            return

        clump_sizes = np.bincount(clump_img.ravel(), minlength=n_clumps + 1)
        clump_classes = np.empty(n_clumps + 1, dtype=arr.dtype)
        clump_classes[0] = arr.ravel()[0]
        clump_classes[1:] = arr.ravel()[rep_flat]

        u_half, v_half = _extract_interior_half_edges(clump_img, connectivity)
        edge_u, edge_v, edge_w = _build_sym_weighted_edges(
            [u_half] if u_half.size else [],
            [v_half] if v_half.size else [],
        )
        final_class_lut = _solve_sieve_merges(
            clump_sizes=clump_sizes,
            clump_classes=clump_classes,
            edge_u=edge_u,
            edge_v=edge_v,
            edge_w=edge_w,
            size_threshold=size_threshold,
        )

        valid_mask = clump_img > 0
        if np.all(valid_mask):
            out_arr = final_class_lut[clump_img]
        else:
            fill_nd = _fill_nodata_value(meta)
            out_arr = np.full(arr.shape, fill_nd, dtype=meta.out_np_dtype)
            out_arr[valid_mask] = final_class_lut[clump_img[valid_mask]].astype(
                meta.out_np_dtype, copy=False
            )

        dst.write(
            cast_result_block(out_arr, meta.height, meta.width, meta.out_np_dtype),
            1,
        )


def _run_sieve_in_memory(
    in_path: str,
    out_path: str,
    img_band: int,
    size_threshold: int,
    connectivity: int,
    gdalformat: str,
    meta: PostprocessMeta,
    creation_opts: Optional[Dict[str, str]],
) -> None:
    """Single-pass exact in-memory sieve filtering."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    with rasterio.open(in_path, "r") as src:
        arr = src.read(img_band)
        grid = IntersectionGrid.from_dataset(src)

    _sieve_array_to_disk(
        arr=arr,
        grid=grid,
        out_path=out_path,
        size_threshold=size_threshold,
        connectivity=connectivity,
        gdalformat=gdalformat,
        meta=meta,
        eff_in_nd=meta.eff_in_nd,
        creation_opts=creation_opts,
    )


def _record_seam_merges(
    arr_tile: np.ndarray,
    global_out: np.ndarray,
    prev_row_ids: np.ndarray,
    prev_row_vals: Optional[np.ndarray],
    prev_col_ids: np.ndarray,
    prev_col_vals: Optional[np.ndarray],
    row_off: int,
    col_off: int,
    width: int,
    connectivity: int,
    merge_u: List[np.ndarray],
    merge_v: List[np.ndarray],
) -> None:
    """Record cross-tile equivalence merges for 4- and 8-connected components."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    win_h, win_w = global_out.shape
    col_end = col_off + win_w

    if row_off > 0 and prev_row_vals is not None:
        top_ids = global_out[0, :]
        top_vals = arr_tile[0, :]
        above_ids = prev_row_ids[col_off:col_end]
        above_vals = prev_row_vals[col_off:col_end]
        seam = (top_ids > 0) & (above_ids > 0) & (top_vals == above_vals)
        if np.any(seam):
            merge_u.append(above_ids[seam].astype(np.int64))
            merge_v.append(top_ids[seam].astype(np.int64))

        if connectivity == 8:
            # Up-left diagonal against previous row strip (includes col_off - 1 corner).
            if col_off > 0:
                ul_ids = prev_row_ids[col_off - 1 : col_end - 1]
                ul_vals = prev_row_vals[col_off - 1 : col_end - 1]
                s_ul = (top_ids > 0) & (ul_ids > 0) & (top_vals == ul_vals)
                if np.any(s_ul):
                    merge_u.append(ul_ids[s_ul].astype(np.int64))
                    merge_v.append(top_ids[s_ul].astype(np.int64))
            elif win_w > 1:
                ul_ids = prev_row_ids[0 : win_w - 1]
                ul_vals = prev_row_vals[0 : win_w - 1]
                s_ul = (
                    (top_ids[1:] > 0)
                    & (ul_ids > 0)
                    & (top_vals[1:] == ul_vals)
                )
                if np.any(s_ul):
                    merge_u.append(ul_ids[s_ul].astype(np.int64))
                    merge_v.append(top_ids[1:][s_ul].astype(np.int64))

            # Up-right diagonal against previous row strip (includes col_end corner).
            if col_end < width:
                ur_ids = prev_row_ids[col_off + 1 : col_end + 1]
                ur_vals = prev_row_vals[col_off + 1 : col_end + 1]
                s_ur = (top_ids > 0) & (ur_ids > 0) & (top_vals == ur_vals)
                if np.any(s_ur):
                    merge_u.append(ur_ids[s_ur].astype(np.int64))
                    merge_v.append(top_ids[s_ur].astype(np.int64))
            elif win_w > 1:
                ur_ids = prev_row_ids[col_off + 1 : col_end]
                ur_vals = prev_row_vals[col_off + 1 : col_end]
                s_ur = (
                    (top_ids[:-1] > 0)
                    & (ur_ids > 0)
                    & (top_vals[:-1] == ur_vals)
                )
                if np.any(s_ur):
                    merge_u.append(ur_ids[s_ur].astype(np.int64))
                    merge_v.append(top_ids[:-1][s_ur].astype(np.int64))

    if col_off > 0 and prev_col_ids.size > 0 and prev_col_vals is not None:
        left_ids = global_out[:, 0]
        left_vals = arr_tile[:, 0]
        seam_col = (
            (left_ids > 0) & (prev_col_ids > 0) & (left_vals == prev_col_vals)
        )
        if np.any(seam_col):
            merge_u.append(prev_col_ids[seam_col].astype(np.int64))
            merge_v.append(left_ids[seam_col].astype(np.int64))

        if connectivity == 8 and win_h > 1:
            s_diag_up = (
                (left_ids[1:] > 0)
                & (prev_col_ids[:-1] > 0)
                & (left_vals[1:] == prev_col_vals[:-1])
            )
            if np.any(s_diag_up):
                merge_u.append(prev_col_ids[:-1][s_diag_up].astype(np.int64))
                merge_v.append(left_ids[1:][s_diag_up].astype(np.int64))

            s_diag_dn = (
                (left_ids[:-1] > 0)
                & (prev_col_ids[1:] > 0)
                & (left_vals[:-1] == prev_col_vals[1:])
            )
            if np.any(s_diag_dn):
                merge_u.append(prev_col_ids[1:][s_diag_dn].astype(np.int64))
                merge_v.append(left_ids[:-1][s_diag_dn].astype(np.int64))


def _append_seam_boundary_edges(
    g_tile: np.ndarray,
    prev_row_global: np.ndarray,
    prev_col_global: np.ndarray,
    row_off: int,
    col_off: int,
    width: int,
    connectivity: int,
    u_half: List[np.ndarray],
    v_half: List[np.ndarray],
) -> None:
    """Collect cross-clump boundary edges along top and left tile seams."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    win_h, win_w = g_tile.shape
    col_end = col_off + win_w

    if row_off > 0:
        top_ids = g_tile[0, :]
        _append_pair_edges(
            prev_row_global[col_off:col_end], top_ids, u_half, v_half
        )
        if connectivity == 8:
            if col_off > 0:
                _append_pair_edges(
                    prev_row_global[col_off - 1 : col_end - 1],
                    top_ids,
                    u_half,
                    v_half,
                )
            elif win_w > 1:
                _append_pair_edges(
                    prev_row_global[0 : win_w - 1], top_ids[1:], u_half, v_half
                )
            if col_end < width:
                _append_pair_edges(
                    prev_row_global[col_off + 1 : col_end + 1],
                    top_ids,
                    u_half,
                    v_half,
                )
            elif win_w > 1:
                _append_pair_edges(
                    prev_row_global[col_off + 1 : col_end],
                    top_ids[:-1],
                    u_half,
                    v_half,
                )

    if col_off > 0 and prev_col_global.size > 0:
        left_ids = g_tile[:, 0]
        _append_pair_edges(prev_col_global, left_ids, u_half, v_half)
        if connectivity == 8 and win_h > 1:
            _append_pair_edges(
                prev_col_global[:-1], left_ids[1:], u_half, v_half
            )
            _append_pair_edges(
                prev_col_global[1:], left_ids[:-1], u_half, v_half
            )


def _run_sieve_tiled(
    in_path: str,
    out_path: str,
    img_band: int,
    size_threshold: int,
    connectivity: int,
    gdalformat: str,
    meta: PostprocessMeta,
    block_size: int,
    creation_opts: Optional[Dict[str, str]],
    use_pbar: bool,
) -> None:
    """
    Memory-bounded multi-pass tiled sieve filter producing bit-identical
    results to :func:`_run_sieve_in_memory`.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-statements
    grid = IntersectionGrid(
        width=meta.width,
        height=meta.height,
        transform=meta.transform,
        crs=meta.crs,
        offsets={in_path: (0, 0)},
    )

    with tempfile.TemporaryDirectory(prefix="rsgispy_sieve_") as tmp_dir:
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
        cur_row_ids = np.zeros(grid.width, dtype=np.uint32)
        prev_row_vals: Optional[np.ndarray] = None
        cur_row_vals: Optional[np.ndarray] = None
        prev_col_ids = np.empty(0, dtype=np.uint32)
        prev_col_vals: Optional[np.ndarray] = None
        cur_strip_row = 0

        total_raw = 0
        raw_class_chunks: List[np.ndarray] = []
        raw_size_chunks: List[np.ndarray] = []
        raw_first_pxl_chunks: List[np.ndarray] = []
        merge_u: List[np.ndarray] = []
        merge_v: List[np.ndarray] = []

        # Pass 1: Label each tile independently and record seam merges.
        with ExitStack() as p1_stack:
            src = p1_stack.enter_context(rasterio.open(in_path, "r"))
            in_dtype = np.dtype(src.dtypes[img_band - 1])
            cur_row_vals = np.empty(grid.width, dtype=in_dtype)
            tmp_dst = open_output_dataset(
                stack=p1_stack,
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
                col_end = col_off + win_w
                if row_off != cur_strip_row:
                    prev_row_ids, cur_row_ids = cur_row_ids, prev_row_ids
                    cur_row_ids.fill(0)
                    if prev_row_vals is None:
                        prev_row_vals = cur_row_vals.copy()
                    else:
                        prev_row_vals, cur_row_vals = cur_row_vals, prev_row_vals
                    cur_strip_row = row_off
                if col_off == 0:
                    prev_col_ids = np.empty(0, dtype=np.uint32)
                    prev_col_vals = None

                arr_tile = src.read(img_band, window=out_win)
                local_out, k_clumps, rep_local_flat = _label_connected_2d(
                    arr_tile, meta.eff_in_nd, connectivity
                )

                if k_clumps > 0:
                    global_out = np.where(
                        local_out > 0, local_out + total_raw, np.uint32(0)
                    )
                    raw_class_chunks.append(arr_tile.ravel()[rep_local_flat])
                    raw_size_chunks.append(
                        np.bincount(local_out.ravel(), minlength=k_clumps + 1)[1:]
                    )
                    rep_r = rep_local_flat // win_w
                    rep_c = rep_local_flat % win_w
                    raw_first_pxl_chunks.append(
                        (row_off + rep_r) * grid.width + (col_off + rep_c)
                    )
                    total_raw += k_clumps
                else:
                    global_out = local_out

                tmp_dst.write(global_out, 1, window=out_win)
                _record_seam_merges(
                    arr_tile=arr_tile,
                    global_out=global_out,
                    prev_row_ids=prev_row_ids,
                    prev_row_vals=prev_row_vals,
                    prev_col_ids=prev_col_ids,
                    prev_col_vals=prev_col_vals,
                    row_off=row_off,
                    col_off=col_off,
                    width=grid.width,
                    connectivity=connectivity,
                    merge_u=merge_u,
                    merge_v=merge_v,
                )

                cur_row_ids[col_off:col_end] = global_out[-1, :]
                cur_row_vals[col_off:col_end] = arr_tile[-1, :]
                prev_col_ids = global_out[:, -1].copy()
                prev_col_vals = arr_tile[:, -1].copy()

        # Resolve empty/trivial cases before scanning boundary graph.
        if total_raw == 0 or size_threshold <= 1:
            with ExitStack() as cp_stack:
                src = cp_stack.enter_context(rasterio.open(in_path, "r"))
                dst = open_output_dataset(
                    stack=cp_stack,
                    output_img=out_path,
                    gdalformat=gdalformat,
                    out_rio_dtype=meta.out_rio_dtype,
                    grid=grid,
                    count=1,
                    creation_opts=creation_opts,
                )
                if meta.out_nd is not None:
                    dst.nodata = meta.out_nd
                for out_win in iter_dataset_windows(
                    grid.height,
                    grid.width,
                    block_size=block_size,
                    use_pbar=use_pbar,
                ):
                    if total_raw == 0:
                        tile = np.full(
                            (int(out_win.height), int(out_win.width)),
                            _fill_nodata_value(meta),
                            dtype=meta.out_np_dtype,
                        )
                    else:
                        tile = src.read(img_band, window=out_win)
                    dst.write(
                        cast_result_block(
                            tile,
                            int(out_win.height),
                            int(out_win.width),
                            meta.out_np_dtype,
                        ),
                        1,
                        window=out_win,
                    )
            return

        raw_classes = np.concatenate(raw_class_chunks)
        raw_sizes = np.concatenate(raw_size_chunks)
        raw_first_pxl = np.concatenate(raw_first_pxl_chunks)
        if merge_u:
            u_all = np.concatenate(merge_u) - 1
            v_all = np.concatenate(merge_v) - 1
        else:
            u_all = np.empty(0, dtype=np.int64)
            v_all = np.empty(0, dtype=np.int64)

        n_comp, raw_comp_labels = _solve_graph_components(total_raw, u_all, v_all)
        # Canonicalize global components in strict row-major first-pixel order
        # so component IDs match single-pass in-memory execution.
        comp_first_pxl = np.full(
            n_comp, grid.height * grid.width, dtype=np.int64
        )
        np.minimum.at(comp_first_pxl, raw_comp_labels, raw_first_pxl)
        canon_order = np.argsort(comp_first_pxl, kind="mergesort")
        canon_lut = np.empty(n_comp, dtype=np.int64)
        canon_lut[canon_order] = np.arange(n_comp, dtype=np.int64)
        raw_comp_labels = canon_lut[raw_comp_labels]

        raw_to_comp = np.zeros(total_raw + 1, dtype=np.uint32)
        raw_to_comp[1:] = (raw_comp_labels + 1).astype(np.uint32)

        clump_sizes = np.bincount(
            raw_comp_labels + 1, weights=raw_sizes, minlength=n_comp + 1
        ).astype(np.int64)
        clump_classes = np.empty(n_comp + 1, dtype=in_dtype)
        clump_classes[0] = raw_classes[0]
        clump_classes[raw_comp_labels + 1] = raw_classes

        # Pass 2: Extract boundary edges between distinct global clumps.
        u_half_chunks: List[np.ndarray] = []
        v_half_chunks: List[np.ndarray] = []
        prev_row_global = np.zeros(grid.width, dtype=np.uint32)
        cur_row_global = np.zeros(grid.width, dtype=np.uint32)
        prev_col_global = np.empty(0, dtype=np.uint32)
        cur_strip_row = 0

        with rasterio.open(tmp_raw_img, "r") as raw_src:
            for out_win in iter_dataset_windows(
                grid.height,
                grid.width,
                block_size=block_size,
                use_pbar=use_pbar,
            ):
                col_off = int(out_win.col_off)
                row_off = int(out_win.row_off)
                win_w = int(out_win.width)
                col_end = col_off + win_w
                if row_off != cur_strip_row:
                    prev_row_global, cur_row_global = (
                        cur_row_global,
                        prev_row_global,
                    )
                    cur_row_global.fill(0)
                    cur_strip_row = row_off
                if col_off == 0:
                    prev_col_global = np.empty(0, dtype=np.uint32)

                g_tile = raw_to_comp[raw_src.read(1, window=out_win)]
                iu, iv = _extract_interior_half_edges(g_tile, connectivity)
                if iu.size > 0:
                    u_half_chunks.append(iu)
                    v_half_chunks.append(iv)
                _append_seam_boundary_edges(
                    g_tile=g_tile,
                    prev_row_global=prev_row_global,
                    prev_col_global=prev_col_global,
                    row_off=row_off,
                    col_off=col_off,
                    width=grid.width,
                    connectivity=connectivity,
                    u_half=u_half_chunks,
                    v_half=v_half_chunks,
                )
                cur_row_global[col_off:col_end] = g_tile[-1, :]
                prev_col_global = g_tile[:, -1].copy()

        edge_u, edge_v, edge_w = _build_sym_weighted_edges(
            u_half_chunks, v_half_chunks
        )
        final_class_lut = _solve_sieve_merges(
            clump_sizes=clump_sizes,
            clump_classes=clump_classes,
            edge_u=edge_u,
            edge_v=edge_v,
            edge_w=edge_w,
            size_threshold=size_threshold,
        )
        raw_final_lut = np.empty(total_raw + 1, dtype=meta.out_np_dtype)
        fill_nd = _fill_nodata_value(meta)
        raw_final_lut[0] = np.asarray(fill_nd, dtype=meta.out_np_dtype)
        raw_final_lut[1:] = final_class_lut[raw_to_comp[1:]].astype(
            meta.out_np_dtype, copy=False
        )

        # Pass 3: Stream final class values to output raster.
        with ExitStack() as p3_stack:
            raw_src = p3_stack.enter_context(rasterio.open(tmp_raw_img, "r"))
            dst = open_output_dataset(
                stack=p3_stack,
                output_img=out_path,
                gdalformat=gdalformat,
                out_rio_dtype=meta.out_rio_dtype,
                grid=grid,
                count=1,
                creation_opts=creation_opts,
            )
            if meta.out_nd is not None:
                dst.nodata = meta.out_nd
            for out_win in iter_dataset_windows(
                grid.height,
                grid.width,
                block_size=block_size,
                use_pbar=use_pbar,
            ):
                raw_blk = raw_src.read(1, window=out_win)
                dst.write(raw_final_lut[raw_blk], 1, window=out_win)


def run_sieve_filter(
    in_path: str,
    out_path: str,
    size_threshold: int,
    connectivity: int,
    img_band: int,
    datatype: Optional[int],
    no_data_val: Optional[float],
    gdalformat: str,
    in_memory: Optional[bool],
    block_size: Optional[int],
    calc_stats: bool,
    copy_colour_table: bool,
    creation_opts: Optional[Mapping[str, Any]],
    use_pbar: bool,
) -> None:
    """Validate parameters and dispatch in-memory or tiled sieve filtering."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    if isinstance(size_threshold, bool) or not isinstance(
        size_threshold, (int, np.integer)
    ):
        raise rsgispy.RSGISPyException(
            f"size_threshold must be an integer >= 1, got {size_threshold!r}."
        )
    thresh = int(size_threshold)
    if thresh < 1:
        raise rsgispy.RSGISPyException(
            f"size_threshold must be >= 1, got {thresh}."
        )

    if isinstance(connectivity, bool) or not isinstance(
        connectivity, (int, np.integer)
    ):
        raise rsgispy.RSGISPyException(
            f"connectivity must be 4 or 8, got {connectivity!r}."
        )
    conn = int(connectivity)
    if conn not in (4, 8):
        raise rsgispy.RSGISPyException(
            f"connectivity must be 4 or 8, got {conn}."
        )

    blk_sz = (
        get_default_block_size() if block_size is None else int(block_size)
    )
    if blk_sz <= 0:
        raise rsgispy.RSGISPyException(
            f"block_size must be a positive integer, got {block_size}."
        )

    meta = resolve_postprocess_io_meta(
        in_path=in_path,
        img_band=img_band,
        datatype=datatype,
        no_data_val=no_data_val,
        copy_colour_table=copy_colour_table,
    )
    ensure_parent_dir(out_path)
    norm_opts = normalise_creation_options(creation_opts)

    if in_memory is None:
        fits_single_tile = meta.width <= blk_sz and meta.height <= blk_sz
        use_in_mem = (
            fits_single_tile
            or (meta.width * meta.height) <= _AUTO_IN_MEM_MAX_PXLS
        )
    else:
        use_in_mem = bool(in_memory)

    if use_in_mem:
        _run_sieve_in_memory(
            in_path=in_path,
            out_path=out_path,
            img_band=img_band,
            size_threshold=thresh,
            connectivity=conn,
            gdalformat=gdalformat,
            meta=meta,
            creation_opts=norm_opts,
        )
    else:
        _run_sieve_tiled(
            in_path=in_path,
            out_path=out_path,
            img_band=img_band,
            size_threshold=thresh,
            connectivity=conn,
            gdalformat=gdalformat,
            meta=meta,
            block_size=blk_sz,
            creation_opts=norm_opts,
            use_pbar=use_pbar,
        )

    finalise_postprocess_output(
        out_path=out_path,
        meta=meta,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )


def _make_majority_window_node(
    source: TileNode,
    size: int,
    eff_nd: Optional[float],
    out_nd: Optional[float],
    out_np_dtype: np.dtype,
    keep_centre_on_tie: bool,
    name: str,
) -> WindowOp:
    """Build a :class:`WindowOp` applying modal majority voting on band 0."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    pad = size // 2
    keep_tie = bool(keep_centre_on_tie)

    def _tile_fn(arr: np.ndarray) -> np.ndarray:
        halo = arr[0]
        win_h = halo.shape[0] - 2 * pad
        win_w = halo.shape[1] - 2 * pad
        y_end = pad + win_h
        x_end = pad + win_w

        valid_halo = build_valid_mask(halo, eff_nd)
        centre = halo[pad:y_end, pad:x_end]
        centre_valid = valid_halo[pad:y_end, pad:x_end]
        wins = sliding_window_view(halo, (size, size))
        v_wins = sliding_window_view(valid_halo, (size, size))

        res = stat_mode(wins, centre, v_wins, centre_valid)
        if keep_tie:
            diff_mask = centre_valid & np.isfinite(res) & (res != centre)
            if np.any(diff_mask):
                w_sub = wins[diff_mask]
                v_sub = v_wins[diff_mask]
                m_cnt = np.sum(
                    v_sub & (w_sub == res[diff_mask][:, None, None]),
                    axis=(1, 2),
                )
                c_cnt = np.sum(
                    v_sub & (w_sub == centre[diff_mask][:, None, None]),
                    axis=(1, 2),
                )
                tied_with_centre = c_cnt >= m_cnt
                if np.any(tied_with_centre):
                    upd = res[diff_mask]
                    upd[tied_with_centre] = centre[diff_mask][tied_with_centre]
                    res[diff_mask] = upd

        out_2d = _apply_tile_nd(res, out_nd, out_np_dtype)
        return out_2d[np.newaxis, :, :].astype(np.float64, copy=False)

    return WindowOp(
        source=source,
        radius=(pad, pad),
        func=_tile_fn,
        border=HaloBorder(mode="constant", fill_value=np.nan),
        name=name,
    )


def run_postprocess_pipeline(
    in_path: str,
    out_path: str,
    majority_sizes: Sequence[int],
    sieve_size_threshold: Optional[int],
    sieve_connectivity: int,
    keep_centre_on_tie: bool,
    img_band: int,
    in_memory: Optional[bool],
    block_size: Optional[int],
    copy_colour_table: bool,
    gdalformat: str,
    datatype: Optional[int],
    no_data_val: Optional[float],
    calc_stats: bool,
    use_pbar: bool,
    creation_opts: Optional[Mapping[str, Any]],
) -> None:
    """
    Execute a halo-fused sequence of majority filter passes optionally followed
    by connected-component sieving with zero intermediate disk I/O (in memory).
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    # pylint: disable=too-many-branches
    maj_sizes = [validate_filter_size(s) for s in majority_sizes]
    if not maj_sizes and sieve_size_threshold is None:
        raise rsgispy.RSGISPyException(
            "At least one majority_sizes entry or sieve_size_threshold "
            "must be specified."
        )

    if not maj_sizes and sieve_size_threshold is not None:
        run_sieve_filter(
            in_path=in_path,
            out_path=out_path,
            size_threshold=sieve_size_threshold,
            connectivity=sieve_connectivity,
            img_band=img_band,
            datatype=datatype,
            no_data_val=no_data_val,
            gdalformat=gdalformat,
            in_memory=in_memory,
            block_size=block_size,
            calc_stats=calc_stats,
            copy_colour_table=copy_colour_table,
            creation_opts=creation_opts,
            use_pbar=use_pbar,
        )
        return

    thresh: Optional[int] = None
    conn = int(sieve_connectivity)
    if sieve_size_threshold is not None:
        if isinstance(sieve_size_threshold, bool) or not isinstance(
            sieve_size_threshold, (int, np.integer)
        ):
            raise rsgispy.RSGISPyException(
                "sieve_size_threshold must be an integer >= 1, "
                f"got {sieve_size_threshold!r}."
            )
        thresh = int(sieve_size_threshold)
        if thresh < 1:
            raise rsgispy.RSGISPyException(
                f"sieve_size_threshold must be >= 1, got {thresh}."
            )
        if isinstance(sieve_connectivity, bool) or conn not in (4, 8):
            raise rsgispy.RSGISPyException(
                f"sieve_connectivity must be 4 or 8, got {sieve_connectivity!r}."
            )

    blk_sz = (
        get_default_block_size() if block_size is None else int(block_size)
    )
    if blk_sz <= 0:
        raise rsgispy.RSGISPyException(
            f"block_size must be a positive integer, got {block_size}."
        )

    meta = resolve_postprocess_io_meta(
        in_path=in_path,
        img_band=img_band,
        datatype=datatype,
        no_data_val=no_data_val,
        copy_colour_table=copy_colour_table,
    )
    ensure_parent_dir(out_path)
    norm_opts = normalise_creation_options(creation_opts)
    out_rsgis_dt = (
        int(datatype)
        if datatype is not None
        else rsgispy.get_rsgis_datatype(meta.out_rio_dtype)
    )

    cur_node: TileNode = RasterSource(
        in_path, bands=(int(img_band),), dtype=np.float64, name="src"
    )
    for idx, size in enumerate(maj_sizes):
        stage_nd = meta.eff_in_nd if idx == 0 else meta.out_nd
        cur_node = _make_majority_window_node(
            source=cur_node,
            size=size,
            eff_nd=stage_nd,
            out_nd=meta.out_nd,
            out_np_dtype=meta.out_np_dtype,
            keep_centre_on_tie=keep_centre_on_tie,
            name=f"majority_{idx}_{size}x{size}",
        )

    if thresh is None:
        sink = DiskSink(
            source=cur_node,
            output_img=out_path,
            gdalformat=gdalformat,
            datatype=out_rsgis_dt,
            nodata=meta.out_nd,
            clamp_int=True,
            creation_opts=norm_opts,
            name="output",
        )
        TileGraph(sink, block_size=blk_sz).run(use_pbar=use_pbar)
    else:
        if in_memory is None:
            fits_single_tile = meta.width <= blk_sz and meta.height <= blk_sz
            use_in_mem = (
                fits_single_tile
                or (meta.width * meta.height) <= _AUTO_IN_MEM_MAX_PXLS
            )
        else:
            use_in_mem = bool(in_memory)

        if use_in_mem:
            mem_sink = MemorySink(
                source=cur_node,
                dtype=meta.out_np_dtype,
                clamp_int=True,
                name="maj_out",
            )
            TileGraph(mem_sink, block_size=blk_sz).run(use_pbar=use_pbar)
            if mem_sink.result is None:
                raise rsgispy.RSGISPyException(
                    "In-memory majority filter pipeline produced no output."
                )
            arr_2d = mem_sink.result[0]
            grid = IntersectionGrid(
                crs=meta.crs,
                transform=meta.transform,
                width=meta.width,
                height=meta.height,
                offsets={},
            )
            _sieve_array_to_disk(
                arr=arr_2d,
                grid=grid,
                out_path=out_path,
                size_threshold=thresh,
                connectivity=conn,
                gdalformat=gdalformat,
                meta=meta,
                eff_in_nd=meta.out_nd,
                creation_opts=norm_opts,
            )
        else:
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_maj = os.path.join(tmp_dir, "fused_majority.tif")
                tmp_sink = DiskSink(
                    source=cur_node,
                    output_img=tmp_maj,
                    gdalformat="GTIFF",
                    datatype=out_rsgis_dt,
                    nodata=meta.out_nd,
                    clamp_int=True,
                    name="tmp_maj",
                )
                TileGraph(tmp_sink, block_size=blk_sz).run(use_pbar=use_pbar)
                sieve_meta = PostprocessMeta(
                    width=meta.width,
                    height=meta.height,
                    src_count=1,
                    crs=meta.crs,
                    transform=meta.transform,
                    out_rio_dtype=meta.out_rio_dtype,
                    out_np_dtype=meta.out_np_dtype,
                    eff_in_nd=meta.out_nd,
                    out_nd=meta.out_nd,
                    band_desc=meta.band_desc,
                    clr_lut=meta.clr_lut,
                )
                _run_sieve_tiled(
                    in_path=tmp_maj,
                    out_path=out_path,
                    img_band=1,
                    size_threshold=thresh,
                    connectivity=conn,
                    gdalformat=gdalformat,
                    meta=sieve_meta,
                    block_size=blk_sz,
                    creation_opts=norm_opts,
                    use_pbar=use_pbar,
                )

    finalise_postprocess_output(
        out_path=out_path,
        meta=meta,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )
