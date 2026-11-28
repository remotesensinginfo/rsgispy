"""
Simple Linear Iterative Clustering (SLIC) superpixel segmentation for
multi-band remote sensing imagery with global coordinate-aligned seed grids
for seamless tiled execution.

Reference
---------
Achanta, R., Shaji, A., Smith, K., Lucchi, A., Fua, P., & Süsstrunk, S. (2012).
SLIC Superpixels Compared to State-of-the-Art Superpixel Methods. *IEEE
Transactions on Pattern Analysis and Machine Intelligence*, 34(11), 2274–2282.
"""

import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import rasterio

import rsgispy
from rsgispy._numba import optional_njit
from rsgispy.segmentation._helpers import label_4connected
from rsgispy.segmentation.tiling import (
    TileWindow,
    crop_halo_labels,
    split_core_components,
    run_tiled_array_pipeline,
    run_tiled_raster_pipeline,
)
from rsgispy.segmentation.utils import (
    call_kernel,
    compute_global_band_stats,
    compute_multiband_gradient,
    gaussian_smooth_nodata,
    make_njit,
    normalise_array_input,
    normalise_band_selection,
    standardise_bands,
)

__all__ = [
    "slic_segmentation",
    "slic_segmentation_array",
]


@make_njit
def _slic_iterate_kernel(
    std_arr: np.ndarray,
    valid_mask: np.ndarray,
    seed_rows: np.ndarray,
    seed_cols: np.ndarray,
    seed_spec: np.ndarray,
    grid_step: float,
    compactness: float,
    max_iter: int,
) -> np.ndarray:
    """
    Run SLIC assignment and cluster-centre updates inside one halo tile.
    Returns 2D integer array storing 0-based local cluster index (``-1`` for
    invalid pixels).
    """
    # pylint: disable=too-many-arguments,too-many-locals,too-many-branches
    n_bands, height, width = std_arr.shape
    n_seeds = seed_rows.shape[0]
    labels = np.full((height, width), -1, dtype=np.int64)
    distances = np.full((height, width), 1e300, dtype=np.float64)

    search_rad = max(2, int(math.ceil(grid_step * 1.5)))
    spatial_weight = (compactness / (10.0 * max(grid_step, 1.0))) ** 2
    inv_bands = 1.0 / float(n_bands)

    sum_r = np.zeros(n_seeds, dtype=np.float64)
    sum_c = np.zeros(n_seeds, dtype=np.float64)
    sum_spec = np.zeros((n_seeds, n_bands), dtype=np.float64)
    counts = np.zeros(n_seeds, dtype=np.int64)

    for _ in range(max_iter):
        for r in range(height):
            for c in range(width):
                distances[r, c] = 1e300
                labels[r, c] = -1

        for s_i in range(n_seeds):
            cr = seed_rows[s_i]
            cc = seed_cols[s_i]
            r0 = max(0, int(math.floor(cr)) - search_rad)
            r1 = min(height, int(math.ceil(cr)) + search_rad + 1)
            c0 = max(0, int(math.floor(cc)) - search_rad)
            c1 = min(width, int(math.ceil(cc)) + search_rad + 1)

            for r in range(r0, r1):
                dr = float(r) - cr
                dr_sq = dr * dr
                for c in range(c0, c1):
                    if not valid_mask[r, c]:
                        continue
                    dc = float(c) - cc
                    d_spec_sq = 0.0
                    for b_i in range(n_bands):
                        diff = std_arr[b_i, r, c] - seed_spec[s_i, b_i]
                        d_spec_sq += diff * diff
                    dist = d_spec_sq * inv_bands + spatial_weight * (
                        dr_sq + dc * dc
                    )
                    if dist < distances[r, c]:
                        distances[r, c] = dist
                        labels[r, c] = s_i

        # Recompute centroids
        for s_i in range(n_seeds):
            sum_r[s_i] = 0.0
            sum_c[s_i] = 0.0
            counts[s_i] = 0
            for b_i in range(n_bands):
                sum_spec[s_i, b_i] = 0.0

        for r in range(height):
            for c in range(width):
                lbl = labels[r, c]
                if lbl >= 0:
                    counts[lbl] += 1
                    sum_r[lbl] += float(r)
                    sum_c[lbl] += float(c)
                    for b_i in range(n_bands):
                        sum_spec[lbl, b_i] += std_arr[b_i, r, c]

        for s_i in range(n_seeds):
            cnt = counts[s_i]
            if cnt > 0:
                inv_c = 1.0 / float(cnt)
                seed_rows[s_i] = sum_r[s_i] * inv_c
                seed_cols[s_i] = sum_c[s_i] * inv_c
                for b_i in range(n_bands):
                    seed_spec[s_i, b_i] = sum_spec[s_i, b_i] * inv_c

    # Nearest fallback for any valid pixel not reached by 1.5*S windows
    for r in range(height):
        for c in range(width):
            if valid_mask[r, c] and labels[r, c] < 0:
                best_s = 0
                best_d = 1e300
                for s_i in range(n_seeds):
                    dr = float(r) - seed_rows[s_i]
                    dc = float(c) - seed_cols[s_i]
                    d_spec_sq = 0.0
                    for b_i in range(n_bands):
                        diff = std_arr[b_i, r, c] - seed_spec[s_i, b_i]
                        d_spec_sq += diff * diff
                    d_val = d_spec_sq * inv_bands + spatial_weight * (
                        dr * dr + dc * dc
                    )
                    if d_val < best_d:
                        best_d = d_val
                        best_s = s_i
                labels[r, c] = best_s

    return labels


@make_njit
def _absorb_small_slic_components_kernel(
    clumps: np.ndarray,
    seed_map: np.ndarray,
    clump_counts: np.ndarray,
    min_comp_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Merge 4-connected superpixel fragments smaller than ``min_comp_size``
    into an adjacent larger component and propagate its global ``seed_map`` ID.
    """
    height, width = clumps.shape
    n_clumps = clump_counts.shape[0]
    parent = np.arange(n_clumps, dtype=np.int64)
    clump_seed = np.zeros(n_clumps, dtype=np.int64)

    for r in range(height):
        for c in range(width):
            cid = clumps[r, c]
            if cid > 0 and clump_seed[cid] == 0:
                clump_seed[cid] = seed_map[r, c]

    for _ in range(16):
        merged_any = False
        for r in range(height):
            for c in range(width):
                cid = clumps[r, c]
                if cid <= 0:
                    continue
                rc = cid
                while parent[rc] != rc:
                    parent[rc] = parent[parent[rc]]
                    rc = parent[rc]
                if clump_counts[rc] >= min_comp_size:
                    continue

                best_nbr = 0
                best_cnt = -1
                if r > 0 and clumps[r - 1, c] > 0:
                    rn = clumps[r - 1, c]
                    while parent[rn] != rn:
                        rn = parent[rn]
                    if rn != rc and clump_counts[rn] > best_cnt:
                        best_cnt = clump_counts[rn]
                        best_nbr = rn
                if r + 1 < height and clumps[r + 1, c] > 0:
                    rn = clumps[r + 1, c]
                    while parent[rn] != rn:
                        rn = parent[rn]
                    if rn != rc and clump_counts[rn] > best_cnt:
                        best_cnt = clump_counts[rn]
                        best_nbr = rn
                if c > 0 and clumps[r, c - 1] > 0:
                    rn = clumps[r, c - 1]
                    while parent[rn] != rn:
                        rn = parent[rn]
                    if rn != rc and clump_counts[rn] > best_cnt:
                        best_cnt = clump_counts[rn]
                        best_nbr = rn
                if c + 1 < width and clumps[r, c + 1] > 0:
                    rn = clumps[r, c + 1]
                    while parent[rn] != rn:
                        rn = parent[rn]
                    if rn != rc and clump_counts[rn] > best_cnt:
                        best_cnt = clump_counts[rn]
                        best_nbr = rn

                if best_nbr > 0:
                    parent[rc] = best_nbr
                    clump_counts[best_nbr] += clump_counts[rc]
                    clump_counts[rc] = 0
                    merged_any = True
        if not merged_any:
            break

    for idx in range(1, n_clumps):
        root = idx
        while parent[root] != root:
            root = parent[root]
        parent[idx] = root
    return parent, clump_seed


@optional_njit(cache=True)
def _perturb_seeds_kernel(
    grad: np.ndarray,
    halo_valid: np.ndarray,
    nom_rows: np.ndarray,
    nom_cols: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For every nominal grid centre ``(nom_rows[i], nom_cols[j])`` (row-major
    flattened), find the valid pixel with minimum gradient inside the clipped
    3x3 neighbourhood (first minimum in ``dr``/``dc`` scan order). Returns
    ``(best_r, best_c)`` with ``-1`` where no valid pixel exists.

    Kept single-threaded (no ``prange``) because tile workers already run in
    parallel processes.
    """
    halo_h, halo_w = grad.shape
    n_r = nom_rows.shape[0]
    n_c = nom_cols.shape[0]
    best_r = np.full(n_r * n_c, -1, dtype=np.int64)
    best_c = np.full(n_r * n_c, -1, dtype=np.int64)
    for i in range(n_r):
        nom_r = nom_rows[i]
        for j in range(n_c):
            nom_c = nom_cols[j]
            b_r = -1
            b_c = -1
            best_g = 1e300
            for dr in range(-1, 2):
                rr = nom_r + dr
                if rr < 0 or rr >= halo_h:
                    continue
                for dc in range(-1, 2):
                    cc = nom_c + dc
                    if cc < 0 or cc >= halo_w or not halo_valid[rr, cc]:
                        continue
                    g_val = grad[rr, cc]
                    if g_val < best_g:
                        best_g = g_val
                        b_r = rr
                        b_c = cc
            best_r[i * n_c + j] = b_r
            best_c[i * n_c + j] = b_c
    return best_r, best_c


def _initialise_global_aligned_seeds(
    std_halo: np.ndarray,
    halo_valid: np.ndarray,
    tile: TileWindow,
    grid_step: float,
    full_width: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate SLIC cluster centres on the global pixel grid ``(grid_step x
    grid_step)``, perturb each inside a 3x3 valid neighbourhood to minimum
    multi-band gradient, and return ``(seed_rows, seed_cols, seed_spec, seed_ids)``.
    """
    # pylint: disable=too-many-locals
    n_bands, halo_h, halo_w = std_halo.shape
    grad = compute_multiband_gradient(std_halo, halo_valid)
    n_grid_cols = max(1, int(math.ceil(float(full_width) / grid_step)))

    g_r_min = max(
        0,
        int(math.floor((float(tile.halo_row_off) - grid_step) / grid_step)),
    )
    g_r_max = int(
        math.ceil((float(tile.halo_row_off + halo_h) + grid_step) / grid_step)
    )
    g_c_min = max(
        0,
        int(math.floor((float(tile.halo_col_off) - grid_step) / grid_step)),
    )
    g_c_max = int(
        math.ceil((float(tile.halo_col_off + halo_w) + grid_step) / grid_step)
    )

    # Nominal grid-centre rows/cols that fall within (or 1px outside) the
    # halo; rows and columns are filtered independently.
    gr_arr = np.arange(g_r_min, g_r_max + 1, dtype=np.int64)
    gc_arr = np.arange(g_c_min, g_c_max + 1, dtype=np.int64)
    nom_r_arr = np.array(
        [int(round((float(gr) + 0.5) * grid_step)) for gr in gr_arr.tolist()],
        dtype=np.int64,
    ) - int(tile.halo_row_off)
    nom_c_arr = np.array(
        [int(round((float(gc) + 0.5) * grid_step)) for gc in gc_arr.tolist()],
        dtype=np.int64,
    ) - int(tile.halo_col_off)
    keep_r = (nom_r_arr >= -1) & (nom_r_arr <= halo_h)
    keep_c = (nom_c_arr >= -1) & (nom_c_arr <= halo_w)
    gr_arr = gr_arr[keep_r]
    nom_r_arr = nom_r_arr[keep_r]
    gc_arr = gc_arr[keep_c]
    nom_c_arr = nom_c_arr[keep_c]

    best_r, best_c = _perturb_seeds_kernel(
        np.ascontiguousarray(grad, dtype=np.float64),
        np.ascontiguousarray(halo_valid, dtype=np.bool_),
        nom_r_arr,
        nom_c_arr,
    )
    found = best_r >= 0
    sel_r = best_r[found]
    sel_c = best_c[found]
    grid_ids = (gr_arr[:, np.newaxis] * n_grid_cols + gc_arr[np.newaxis, :] + 1).ravel()

    s_rows: List[float] = sel_r.astype(np.float64).tolist()
    s_cols: List[float] = sel_c.astype(np.float64).tolist()
    s_specs: List[np.ndarray] = list(std_halo[:, sel_r, sel_c].T.copy())
    s_ids: List[int] = grid_ids[found].tolist()

    if not s_rows:
        # Fallback: pick first valid pixel if grid is smaller than tile
        v_r, v_c = np.nonzero(halo_valid)
        r0 = int(v_r[0])
        c0 = int(v_c[0])
        s_rows.append(float(r0))
        s_cols.append(float(c0))
        s_specs.append(std_halo[:, r0, c0].copy())
        s_ids.append(1)

    return (
        np.asarray(s_rows, dtype=np.float64),
        np.asarray(s_cols, dtype=np.float64),
        np.asarray(s_specs, dtype=np.float64).reshape(-1, n_bands),
        np.asarray(s_ids, dtype=np.int64),
    )


def _slic_tile_worker(
    halo_arr: np.ndarray,
    halo_valid: np.ndarray,
    tile: TileWindow,
    params: Dict[str, Any],
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Worker running global-seed-aligned SLIC inside one halo-padded tile."""
    # pylint: disable=too-many-locals
    if not np.any(halo_valid):
        zeros = np.zeros((tile.height, tile.width), dtype=np.uint32)
        return zeros, zeros

    std_halo, _, _ = standardise_bands(
        halo_arr, halo_valid, means=params["means"], stds=params["stds"]
    )
    sigma = float(params["sigma"])
    if sigma > 0.0:
        std_halo = gaussian_smooth_nodata(std_halo, halo_valid, sigma=sigma)

    grid_step = float(params["grid_step"])
    use_numba = bool(params["use_numba"])
    s_rows, s_cols, s_spec, s_ids = _initialise_global_aligned_seeds(
        std_halo=std_halo,
        halo_valid=halo_valid,
        tile=tile,
        grid_step=grid_step,
        full_width=int(params["full_width"]),
    )

    raw_idx = call_kernel(
        _slic_iterate_kernel,
        use_numba,
        std_halo,
        halo_valid,
        s_rows,
        s_cols,
        s_spec,
        grid_step,
        float(params["compactness"]),
        int(params["max_iter"]),
    )

    seed_map = np.zeros(halo_valid.shape, dtype=np.int64)
    pos_mask = raw_idx >= 0
    seed_map[pos_mask] = s_ids[raw_idx[pos_mask]]

    if params["enforce_connectivity"]:
        _, height, width = std_halo.shape
        horiz_eq = (
            (seed_map[:, :-1] == seed_map[:, 1:])
            & (seed_map[:, :-1] > 0)
            & halo_valid[:, :-1]
            & halo_valid[:, 1:]
            if width > 1
            else np.empty((height, 0), dtype=bool)
        )
        vert_eq = (
            (seed_map[:-1, :] == seed_map[1:, :])
            & (seed_map[:-1, :] > 0)
            & halo_valid[:-1, :]
            & halo_valid[1:, :]
            if height > 1
            else np.empty((0, width), dtype=bool)
        )
        clumps, n_clumps, _ = label_4connected(
            halo_valid, horiz_eq, vert_eq, return_first_flat=False
        )
        min_comp = max(
            int(params["min_pixels"]),
            int(round(0.25 * grid_step * grid_step)),
        )
        if n_clumps > 0 and min_comp > 1:
            clump_counts = np.bincount(
                clumps.ravel(), minlength=n_clumps + 1
            ).astype(np.int64)
            clump_counts[0] = 0
            if np.any((clump_counts[1:] > 0) & (clump_counts[1:] < min_comp)):
                parent, clump_seed = call_kernel(
                    _absorb_small_slic_components_kernel,
                    use_numba,
                    clumps.astype(np.int64, copy=False),
                    seed_map,
                    clump_counts,
                    min_comp,
                )
                halo_labels = parent[clumps]
                seed_map = clump_seed[halo_labels]
            else:
                halo_labels = clumps
        else:
            halo_labels = clumps
    else:
        halo_labels = seed_map

    core_labels = split_core_components(crop_halo_labels(halo_labels, tile))
    core_seeds = crop_halo_labels(seed_map, tile)
    core_seeds = np.where(core_labels > 0, core_seeds, 0)
    return core_labels, core_seeds


def _validate_slic_params(
    n_segments: int,
    segment_size: Optional[float],
    compactness: float,
    max_iter: int,
    sigma: float,
    min_pixels: int,
) -> None:
    """Validate SLIC superpixel parameters."""
    if n_segments < 1:
        raise rsgispy.RSGISPyException(
            f"n_segments must be >= 1, got {n_segments}."
        )
    if segment_size is not None and segment_size < 2.0:
        raise rsgispy.RSGISPyException(
            f"segment_size must be >= 2.0 or None, got {segment_size}."
        )
    if compactness <= 0.0:
        raise rsgispy.RSGISPyException(
            f"compactness must be > 0, got {compactness}."
        )
    if max_iter < 1:
        raise rsgispy.RSGISPyException(
            f"max_iter must be >= 1, got {max_iter}."
        )
    if sigma < 0.0:
        raise rsgispy.RSGISPyException(f"sigma must be >= 0, got {sigma}.")
    if min_pixels < 1:
        raise rsgispy.RSGISPyException(
            f"min_pixels must be >= 1, got {min_pixels}."
        )


def slic_segmentation_array(
    data: np.ndarray,
    n_segments: int = 500,
    segment_size: Optional[float] = None,
    compactness: float = 10.0,
    max_iter: int = 10,
    sigma: float = 0.0,
    enforce_connectivity: bool = True,
    min_pixels: int = 10,
    tile_size: Optional[Union[int, Tuple[int, int]]] = None,
    overlap: Optional[int] = None,
    n_jobs: int = 1,
    stitch_seams: bool = True,
    use_numba: bool = True,
    no_data_val: Optional[float] = None,
    use_pbar: bool = False,
) -> np.ndarray:
    """
    Generate SLIC superpixels over an in-memory 2D ``(H, W)`` or 3D
    ``(B, H, W)`` array.

    Parameters
    ----------
    data : numpy.ndarray
        Input 2D ``(H, W)`` or 3D ``(B, H, W)`` array.
    n_segments : int, optional
        Approximate target number of superpixels (Default = ``500``). Ignored
        when ``segment_size`` is specified.
    segment_size : float or None, optional
        Explicit grid spacing ``S`` in pixels between initial superpixel
        centres (Default = ``None``; derived from ``n_segments``).
    compactness : float, optional
        Balance between colour proximity and spatial regularity
        (Default = ``10.0``). Larger values yield more square superpixels.
    max_iter : int, optional
        Number of k-means assignment/update iterations (Default = ``10``).
    sigma : float, optional
        Optional Gaussian pre-smoothing standard deviation in pixels
        (Default = ``0.0``).
    enforce_connectivity : bool, optional
        If ``True`` (Default), enforce 4-connected superpixels and absorb
        small orphaned fragments.
    min_pixels : int, optional
        Minimum segment pixel count (Default = ``10``).
    tile_size : int or tuple[int, int] or None, optional
        Core tile dimensions ``(tile_h, tile_w)`` for tiled processing.
    overlap : int or None, optional
        Halo overlap in pixels around each tile (Default = ``None``,
        automatically set to ``max(8, ceil(2 * S))``).
    n_jobs : int, optional
        Number of worker processes for tiled execution (Default = ``1``).
    stitch_seams : bool, optional
        Reconnect superpixels sharing identical global grid seed IDs across
        tile seams (Default = ``True``).
    use_numba : bool, optional
        Enable optional ``numba`` JIT acceleration when installed
        (Default = ``True``).
    no_data_val : float or None, optional
        Value masked as nodata (output label ``0``).
    use_pbar : bool, optional
        Display a progress bar across tiles (Default = ``False``).

    Returns
    -------
    numpy.ndarray
        2D ``uint32`` label array with ``1..N`` superpixel IDs and ``0`` for
        nodata.

    Examples
    --------
    >>> import numpy as np
    >>> from rsgispy.segmentation import slic_segmentation_array
    >>> img = np.random.default_rng(0).random((3, 32, 32))
    >>> seg = slic_segmentation_array(img, n_segments=16, min_pixels=4)
    >>> seg.shape
    (32, 32)
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    _validate_slic_params(
        n_segments, segment_size, compactness, max_iter, sigma, min_pixels
    )
    arr_3d, valid_mask = normalise_array_input(data, no_data_val=no_data_val)
    n_valid = int(np.count_nonzero(valid_mask))
    if n_valid == 0:
        return np.zeros(arr_3d.shape[1:], dtype=np.uint32)

    grid_step = (
        float(segment_size)
        if segment_size is not None
        else max(2.0, math.sqrt(float(n_valid) / float(n_segments)))
    )
    eff_overlap = (
        int(overlap)
        if overlap is not None
        else max(8, int(math.ceil(2.0 * grid_step)))
    )
    std_arr, means, stds = standardise_bands(arr_3d, valid_mask)
    params = {
        "means": np.zeros_like(means),
        "stds": np.ones_like(stds),
        "grid_step": grid_step,
        "full_width": arr_3d.shape[2],
        "compactness": float(compactness),
        "max_iter": int(max_iter),
        "sigma": float(sigma),
        "enforce_connectivity": bool(enforce_connectivity),
        "min_pixels": int(min_pixels),
        "use_numba": bool(use_numba),
    }
    return run_tiled_array_pipeline(
        arr_3d=std_arr,
        valid_mask=valid_mask,
        worker_fn=_slic_tile_worker,
        params=params,
        tile_size=tile_size,
        overlap=eff_overlap,
        n_jobs=n_jobs,
        stitch_seams=stitch_seams,
        seam_mode="exact_aux",
        min_pixels=int(min_pixels),
        use_numba=use_numba,
        use_pbar=use_pbar,
    )


def slic_segmentation(
    input_img: Union[str, os.PathLike],
    output_img: Union[str, os.PathLike],
    n_segments: int = 500,
    segment_size: Optional[float] = None,
    compactness: float = 10.0,
    max_iter: int = 10,
    sigma: float = 0.0,
    enforce_connectivity: bool = True,
    min_pixels: int = 10,
    bands: Optional[Sequence[int]] = None,
    tile_size: Optional[Union[int, Tuple[int, int]]] = None,
    overlap: Optional[int] = None,
    n_jobs: int = 1,
    stitch_seams: bool = True,
    use_numba: bool = True,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Generate SLIC superpixels for a multi-band remote sensing raster with
    global seed-grid alignment across tiles.

    Parameters
    ----------
    input_img : str or os.PathLike
        Path to the input multi-band raster.
    output_img : str or os.PathLike
        Path to the output superpixel label raster.
    n_segments : int, optional
        Approximate target number of superpixels (Default = ``500``). Ignored
        when ``segment_size`` is specified.
    segment_size : float or None, optional
        Explicit grid spacing ``S`` in pixels (Default = ``None``).
    compactness : float, optional
        Spatial regularity weight (Default = ``10.0``).
    max_iter : int, optional
        Number of SLIC assignment iterations (Default = ``10``).
    sigma : float, optional
        Optional nodata-aware Gaussian pre-smoothing sigma (Default = ``0.0``).
    enforce_connectivity : bool, optional
        Enforce 4-connected superpixels and absorb small disconnected pieces
        (Default = ``True``).
    min_pixels : int, optional
        Minimum segment pixel count (Default = ``10``).
    bands : sequence of int or None, optional
        1-based band indices to include (Default = all bands).
    tile_size : int or tuple[int, int] or None, optional
        Core tile size ``(tile_h, tile_w)`` for bounded-memory execution.
    overlap : int or None, optional
        Halo padding in pixels (Default = ``None``, set to
        ``max(8, ceil(2 * S))``).
    n_jobs : int, optional
        Number of worker processes for parallel tile processing
        (Default = ``1``).
    stitch_seams : bool, optional
        Reconnect matching global seed superpixels across tile seams
        (Default = ``True``).
    use_numba : bool, optional
        Enable ``numba`` JIT acceleration when available (Default = ``True``).
    gdalformat : str, optional
        Output GDAL/rasterio driver name (Default = ``'GTIFF'``).
    datatype : int or None, optional
        Optional ``rsgispy.TYPE_*`` integer output type (Default = ``None``,
        resolves to :data:`rsgispy.TYPE_32UINT`).
    no_data_val : float or None, optional
        Input pixel value treated as nodata (Default = ``None``, uses source
        nodata metadata).
    calc_stats : bool, optional
        If ``True`` (Default), compute statistics and pyramids.
    use_pbar : bool, optional
        If ``True`` (Default), display ``tqdm`` progress bars.

    Examples
    --------
    >>> import rsgispy.segmentation
    >>> # rsgispy.segmentation.slic_segmentation(
    ... #     "sentinel2.tif", "slic.tif", segment_size=20, compactness=10.0
    ... # )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    _validate_slic_params(
        n_segments, segment_size, compactness, max_iter, sigma, min_pixels
    )
    in_path = os.fspath(input_img)
    if not os.path.exists(in_path):
        raise rsgispy.RSGISPyException(f"Input image does not exist: {in_path}")

    with rasterio.open(in_path, "r") as src:
        sel_bands = normalise_band_selection(src, bands, in_path)
        full_height = int(src.height)
        full_width = int(src.width)

    means, stds, _, _, _ = compute_global_band_stats(
        input_img=in_path,
        bands=sel_bands,
        no_data_val=no_data_val,
        sampling=100,
        use_pbar=use_pbar,
    )
    grid_step = (
        float(segment_size)
        if segment_size is not None
        else max(
            2.0,
            math.sqrt(float(full_height * full_width) / float(n_segments)),
        )
    )
    eff_overlap = (
        int(overlap)
        if overlap is not None
        else max(8, int(math.ceil(2.0 * grid_step)))
    )
    params = {
        "means": means,
        "stds": stds,
        "seam_stds": stds,
        "grid_step": grid_step,
        "full_width": full_width,
        "compactness": float(compactness),
        "max_iter": int(max_iter),
        "sigma": float(sigma),
        "enforce_connectivity": bool(enforce_connectivity),
        "min_pixels": int(min_pixels),
        "use_numba": bool(use_numba),
    }
    run_tiled_raster_pipeline(
        input_img=in_path,
        output_img=output_img,
        bands=sel_bands,
        worker_fn=_slic_tile_worker,
        params=params,
        tile_size=tile_size,
        overlap=eff_overlap,
        n_jobs=n_jobs,
        stitch_seams=stitch_seams,
        seam_mode="exact_aux",
        min_pixels=int(min_pixels),
        use_numba=use_numba,
        gdalformat=gdalformat,
        datatype=datatype,
        no_data_val=no_data_val,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )
