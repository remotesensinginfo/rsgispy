"""
Multiresolution / region-merging segmentation (Baatz & Schäpe / FNEA style)
using Local Mutual Best Fitting (LMBF) over spectral and shape heterogeneity.

Reference
---------
Baatz, M., & Schäpe, A. (2000). Multiresolution Segmentation: An Optimization
Approach for High Quality Multi-Scale Image Segmentation. In *Angewandte
Geographische Informationsverarbeitung XII*, pp. 12–23.
"""

import math
import os
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import rasterio

import rsgispy
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
    make_njit,
    normalise_array_input,
    normalise_band_selection,
    standardise_bands,
)

__all__ = [
    "region_merge_segmentation",
    "region_merge_segmentation_array",
]


@make_njit
def _coalesce_edges_kernel(
    u_raw: np.ndarray,
    v_raw: np.ndarray,
    len_raw: np.ndarray,
    parent: np.ndarray,
    n_regions: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Remap edge endpoints to current roots, pack as ``(min_r, max_r)`` keys,
    sort, and sum shared boundary lengths for unique undirected edges.
    """
    n_edges = u_raw.shape[0]
    keys = np.empty(n_edges, dtype=np.int64)
    lens = np.empty(n_edges, dtype=np.int64)
    stride = n_regions + 1
    k_cnt = 0

    for idx in range(n_edges):
        ru = u_raw[idx]
        while parent[ru] != ru:
            parent[ru] = parent[parent[ru]]
            ru = parent[ru]
        rv = v_raw[idx]
        while parent[rv] != rv:
            parent[rv] = parent[parent[rv]]
            rv = parent[rv]
        if ru == rv:
            continue
        if ru > rv:
            tmp = ru
            ru = rv
            rv = tmp
        keys[k_cnt] = ru * stride + rv
        lens[k_cnt] = len_raw[idx]
        k_cnt += 1

    if k_cnt == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )

    order = np.argsort(keys[:k_cnt])
    u_out = np.empty(k_cnt, dtype=np.int64)
    v_out = np.empty(k_cnt, dtype=np.int64)
    l_out = np.empty(k_cnt, dtype=np.int64)

    out_cnt = 0
    prev_key = -1
    for i in range(k_cnt):
        idx = order[i]
        cur_key = keys[idx]
        if cur_key == prev_key:
            l_out[out_cnt - 1] += lens[idx]
        else:
            prev_key = cur_key
            u_out[out_cnt] = cur_key // stride
            v_out[out_cnt] = cur_key % stride
            l_out[out_cnt] = lens[idx]
            out_cnt += 1

    return u_out[:out_cnt], v_out[:out_cnt], l_out[:out_cnt]


@make_njit
def _lmbf_region_merge_kernel(
    u_edges: np.ndarray,
    v_edges: np.ndarray,
    edge_lens: np.ndarray,
    counts: np.ndarray,
    band_sums: np.ndarray,
    band_sq_sums: np.ndarray,
    min_r: np.ndarray,
    max_r: np.ndarray,
    min_c: np.ndarray,
    max_c: np.ndarray,
    perim: np.ndarray,
    scale_sq: float,
    compactness: float,
    shape_weight: float,
) -> np.ndarray:
    """
    Execute Local Mutual Best Fitting (LMBF) rounds using Baatz & Schäpe
    colour + shape heterogeneity until no adjacent pair satisfies ``cost <= scale_sq``.
    """
    # pylint: disable=too-many-arguments,too-many-locals,too-many-branches,too-many-statements
    n_regions = counts.shape[0]
    n_bands = band_sums.shape[1]
    w_colour = 1.0 - shape_weight
    w_shape = shape_weight
    inv_bands = 1.0 / float(n_bands)

    parent = np.arange(n_regions, dtype=np.int64)
    cur_u = u_edges
    cur_v = v_edges
    cur_l = edge_lens

    while cur_u.shape[0] > 0:
        n_cur = cur_u.shape[0]
        edge_costs = np.full(n_cur, 1e300, dtype=np.float64)
        best_nbr = np.zeros(n_regions, dtype=np.int64)
        best_cost = np.full(n_regions, 1e300, dtype=np.float64)

        for e_i in range(n_cur):
            ru = cur_u[e_i]
            rv = cur_v[e_i]
            nu = float(counts[ru])
            nv = float(counts[rv])
            nuv = nu + nv

            h_colour = 0.0
            for b_i in range(n_bands):
                su = band_sums[ru, b_i]
                sv = band_sums[rv, b_i]
                squ = band_sq_sums[ru, b_i]
                sqv = band_sq_sums[rv, b_i]

                var_u = max(0.0, (squ / nu) - (su / nu) ** 2)
                var_v = max(0.0, (sqv / nv) - (sv / nv) ** 2)
                suv = su + sv
                var_uv = max(0.0, ((squ + sqv) / nuv) - (suv / nuv) ** 2)
                h_colour += (
                    nuv * math.sqrt(var_uv)
                    - nu * math.sqrt(var_u)
                    - nv * math.sqrt(var_v)
                )
            h_colour *= inv_bands

            if w_shape > 0.0:
                lu = float(perim[ru])
                lv = float(perim[rv])
                luv = float(perim[ru] + perim[rv] - 2 * cur_l[e_i])

                bu = float(
                    2
                    * (
                        (max_r[ru] - min_r[ru] + 1)
                        + (max_c[ru] - min_c[ru] + 1)
                    )
                )
                bv = float(
                    2
                    * (
                        (max_r[rv] - min_r[rv] + 1)
                        + (max_c[rv] - min_c[rv] + 1)
                    )
                )
                mr_lo = min_r[ru] if min_r[ru] < min_r[rv] else min_r[rv]
                mr_hi = max_r[ru] if max_r[ru] > max_r[rv] else max_r[rv]
                mc_lo = min_c[ru] if min_c[ru] < min_c[rv] else min_c[rv]
                mc_hi = max_c[ru] if max_c[ru] > max_c[rv] else max_c[rv]
                buv = float(2 * ((mr_hi - mr_lo + 1) + (mc_hi - mc_lo + 1)))

                h_compact = math.sqrt(nuv) * luv - (
                    math.sqrt(nu) * lu + math.sqrt(nv) * lv
                )
                h_smooth = nuv * (luv / buv) - (
                    nu * (lu / bu) + nv * (lv / bv)
                )
                h_shape = (
                    compactness * h_compact + (1.0 - compactness) * h_smooth
                )
            else:
                h_shape = 0.0

            cost = w_colour * h_colour + w_shape * h_shape
            edge_costs[e_i] = cost

            if cost <= scale_sq:
                if cost < best_cost[ru] or (
                    cost == best_cost[ru]
                    and (best_nbr[ru] == 0 or rv < best_nbr[ru])
                ):
                    best_cost[ru] = cost
                    best_nbr[ru] = rv
                if cost < best_cost[rv] or (
                    cost == best_cost[rv]
                    and (best_nbr[rv] == 0 or ru < best_nbr[rv])
                ):
                    best_cost[rv] = cost
                    best_nbr[rv] = ru

        merged_any = False
        for e_i in range(n_cur):
            cost = edge_costs[e_i]
            if cost > scale_sq:
                continue
            ru = cur_u[e_i]
            rv = cur_v[e_i]
            if best_nbr[ru] != rv or best_nbr[rv] != ru:
                continue
            if parent[ru] != ru or parent[rv] != rv:
                continue

            # Keep lower ID as root deterministically
            keep_id = ru if ru < rv else rv
            drop_id = rv if ru < rv else ru

            parent[drop_id] = keep_id
            counts[keep_id] += counts[drop_id]
            counts[drop_id] = 0
            for b_i in range(n_bands):
                band_sums[keep_id, b_i] += band_sums[drop_id, b_i]
                band_sq_sums[keep_id, b_i] += band_sq_sums[drop_id, b_i]
            if min_r[drop_id] < min_r[keep_id]:
                min_r[keep_id] = min_r[drop_id]
            if max_r[drop_id] > max_r[keep_id]:
                max_r[keep_id] = max_r[drop_id]
            if min_c[drop_id] < min_c[keep_id]:
                min_c[keep_id] = min_c[drop_id]
            if max_c[drop_id] > max_c[keep_id]:
                max_c[keep_id] = max_c[drop_id]
            perim[keep_id] = perim[keep_id] + perim[drop_id] - 2 * cur_l[e_i]
            merged_any = True

        if not merged_any:
            break

        cur_u, cur_v, cur_l = _coalesce_edges_kernel(
            cur_u, cur_v, cur_l, parent, n_regions
        )

    for r_i in range(1, n_regions):
        root = r_i
        while parent[root] != root:
            root = parent[root]
        parent[r_i] = root
    return parent


def _region_merge_tile_worker(
    halo_arr: np.ndarray,
    halo_valid: np.ndarray,
    tile: TileWindow,
    params: Dict[str, Any],
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Worker running FNEA region merging inside one halo-padded tile."""
    # pylint: disable=too-many-locals
    if not np.any(halo_valid):
        return np.zeros((tile.height, tile.width), dtype=np.uint32), None

    std_halo, _, _ = standardise_bands(
        halo_arr, halo_valid, means=params["means"], stds=params["stds"]
    )
    n_bands, height, width = std_halo.shape
    use_numba: bool = bool(params["use_numba"])

    grid_ids = np.zeros((height, width), dtype=np.int64)
    valid_rows, valid_cols = np.nonzero(halo_valid)
    n_valid = int(valid_rows.size)
    region_ids = np.arange(1, n_valid + 1, dtype=np.int64)
    grid_ids[valid_rows, valid_cols] = region_ids

    n_regions = n_valid + 1
    counts = np.zeros(n_regions, dtype=np.int64)
    counts[1:] = 1
    band_sums = np.zeros((n_regions, n_bands), dtype=np.float64)
    band_sq_sums = np.zeros((n_regions, n_bands), dtype=np.float64)
    for b_i in range(n_bands):
        vals = std_halo[b_i, valid_rows, valid_cols]
        band_sums[1:, b_i] = vals
        band_sq_sums[1:, b_i] = vals * vals

    min_r = np.zeros(n_regions, dtype=np.int64)
    max_r = np.zeros(n_regions, dtype=np.int64)
    min_c = np.zeros(n_regions, dtype=np.int64)
    max_c = np.zeros(n_regions, dtype=np.int64)
    min_r[1:] = valid_rows
    max_r[1:] = valid_rows
    min_c[1:] = valid_cols
    max_c[1:] = valid_cols
    perim = np.zeros(n_regions, dtype=np.int64)
    perim[1:] = 4

    u_parts = []
    v_parts = []
    if width > 1:
        h_mask = halo_valid[:, :-1] & halo_valid[:, 1:]
        if np.any(h_mask):
            u_parts.append(grid_ids[:, :-1][h_mask])
            v_parts.append(grid_ids[:, 1:][h_mask])
    if height > 1:
        v_mask = halo_valid[:-1, :] & halo_valid[1:, :]
        if np.any(v_mask):
            u_parts.append(grid_ids[:-1, :][v_mask])
            v_parts.append(grid_ids[1:, :][v_mask])

    if u_parts:
        u_edges = np.concatenate(u_parts)
        v_edges = np.concatenate(v_parts)
        edge_lens = np.ones(u_edges.shape[0], dtype=np.int64)
        parent = call_kernel(
            _lmbf_region_merge_kernel,
            use_numba,
            u_edges,
            v_edges,
            edge_lens,
            counts,
            band_sums,
            band_sq_sums,
            min_r,
            max_r,
            min_c,
            max_c,
            perim,
            float(params["scale"]) ** 2,
            float(params["compactness"]),
            float(params["shape_weight"]),
        )
        halo_labels = parent[grid_ids]
    else:
        halo_labels = grid_ids

    core_labels = split_core_components(crop_halo_labels(halo_labels, tile))
    return core_labels, None


def _validate_region_merge_params(
    scale: float,
    compactness: float,
    shape_weight: float,
    min_pixels: int,
) -> None:
    """Validate multiresolution region-merging parameters."""
    if scale <= 0.0:
        raise rsgispy.RSGISPyException(f"scale must be > 0, got {scale}.")
    if not 0.0 <= compactness <= 1.0:
        raise rsgispy.RSGISPyException(
            f"compactness must be in [0, 1], got {compactness}."
        )
    if not 0.0 <= shape_weight <= 1.0:
        raise rsgispy.RSGISPyException(
            f"shape_weight must be in [0, 1], got {shape_weight}."
        )
    if min_pixels < 1:
        raise rsgispy.RSGISPyException(
            f"min_pixels must be >= 1, got {min_pixels}."
        )


def region_merge_segmentation_array(
    data: np.ndarray,
    scale: float = 25.0,
    compactness: float = 0.5,
    shape_weight: float = 0.1,
    min_pixels: int = 10,
    tile_size: Optional[Union[int, Tuple[int, int]]] = None,
    overlap: int = 32,
    n_jobs: int = 1,
    stitch_seams: bool = True,
    use_numba: bool = True,
    no_data_val: Optional[float] = None,
    use_pbar: bool = False,
) -> np.ndarray:
    """
    Segment an in-memory 2D ``(H, W)`` or 3D ``(B, H, W)`` array via
    Baatz & Schäpe multiresolution region merging (FNEA).

    Parameters
    ----------
    data : numpy.ndarray
        Input 2D ``(H, W)`` or 3D ``(B, H, W)`` array.
    scale : float, optional
        Heterogeneity threshold controlling segment size (Default = ``25.0``).
        Larger values produce coarser objects.
    compactness : float, optional
        Balance between compactness and smoothness within the shape
        heterogeneity term, in ``[0, 1]`` (Default = ``0.5``).
    shape_weight : float, optional
        Weight of shape heterogeneity relative to spectral colour heterogeneity,
        in ``[0, 1]`` (Default = ``0.1``).
    min_pixels : int, optional
        Minimum segment size in pixels after post-merge elimination
        (Default = ``10``).
    tile_size : int or tuple[int, int] or None, optional
        Optional core tile size ``(tile_h, tile_w)`` for tiled execution.
    overlap : int, optional
        Halo padding in pixels around each tile (Default = ``32``).
    n_jobs : int, optional
        Number of worker processes for parallel tile execution (Default = ``1``).
    stitch_seams : bool, optional
        Spectrally reconcile segments across tile boundaries (Default = ``True``).
    use_numba : bool, optional
        Use ``numba`` JIT acceleration when available (Default = ``True``).
    no_data_val : float or None, optional
        Value masked as nodata (output label ``0``).
    use_pbar : bool, optional
        Display a progress bar across tiles (Default = ``False``).

    Returns
    -------
    numpy.ndarray
        2D ``uint32`` label array with ``1..N`` segment IDs and ``0`` for nodata.

    Examples
    --------
    >>> import numpy as np
    >>> from rsgispy.segmentation import region_merge_segmentation_array
    >>> arr = np.zeros((2, 16, 16), dtype=np.float32)
    >>> arr[:, :, :8] = 10.0
    >>> arr[:, :, 8:] = 90.0
    >>> seg = region_merge_segmentation_array(arr, scale=20.0, min_pixels=10)
    >>> int(seg.max())
    2
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    _validate_region_merge_params(
        scale, compactness, shape_weight, min_pixels
    )
    arr_3d, valid_mask = normalise_array_input(data, no_data_val=no_data_val)
    if not np.any(valid_mask):
        return np.zeros(arr_3d.shape[1:], dtype=np.uint32)

    std_arr, means, stds = standardise_bands(arr_3d, valid_mask)
    params = {
        "means": np.zeros_like(means),
        "stds": np.ones_like(stds),
        "scale": float(scale),
        "compactness": float(compactness),
        "shape_weight": float(shape_weight),
        "use_numba": bool(use_numba),
    }
    seam_thres = min(2.0, max(0.3, float(scale) / 25.0))
    return run_tiled_array_pipeline(
        arr_3d=std_arr,
        valid_mask=valid_mask,
        worker_fn=_region_merge_tile_worker,
        params=params,
        tile_size=tile_size,
        overlap=overlap,
        n_jobs=n_jobs,
        stitch_seams=stitch_seams,
        seam_mode="spectral",
        seam_threshold=seam_thres,
        min_pixels=int(min_pixels),
        use_numba=use_numba,
        use_pbar=use_pbar,
    )


def region_merge_segmentation(
    input_img: Union[str, os.PathLike],
    output_img: Union[str, os.PathLike],
    scale: float = 25.0,
    compactness: float = 0.5,
    shape_weight: float = 0.1,
    min_pixels: int = 10,
    bands: Optional[Sequence[int]] = None,
    tile_size: Optional[Union[int, Tuple[int, int]]] = None,
    overlap: int = 32,
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
    Segment a multi-band remote sensing raster via Baatz & Schäpe
    multiresolution region merging (FNEA).

    Parameters
    ----------
    input_img : str or os.PathLike
        Path to the input multi-band raster.
    output_img : str or os.PathLike
        Path to the output segmentation label raster.
    scale : float, optional
        Heterogeneity threshold controlling segment size (Default = ``25.0``).
    compactness : float, optional
        Compactness vs. smoothness weight within the shape term, in ``[0, 1]``
        (Default = ``0.5``).
    shape_weight : float, optional
        Weight of shape heterogeneity vs. spectral colour heterogeneity, in
        ``[0, 1]`` (Default = ``0.1``).
    min_pixels : int, optional
        Minimum segment size in pixels (Default = ``10``).
    bands : sequence of int or None, optional
        1-based band indices to include (Default = all bands).
    tile_size : int or tuple[int, int] or None, optional
        Core tile size ``(tile_h, tile_w)`` for bounded-memory execution.
    overlap : int, optional
        Halo padding in pixels around each tile (Default = ``32``).
    n_jobs : int, optional
        Number of worker processes for parallel tile processing
        (Default = ``1``).
    stitch_seams : bool, optional
        Reconcile spectrally matching segments across tile seams
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
    >>> # rsgispy.segmentation.region_merge_segmentation(
    ... #     "sentinel2.tif", "fnea_segs.tif", scale=25.0, compactness=0.5
    ... # )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    _validate_region_merge_params(
        scale, compactness, shape_weight, min_pixels
    )
    in_path = os.fspath(input_img)
    if not os.path.exists(in_path):
        raise rsgispy.RSGISPyException(f"Input image does not exist: {in_path}")

    with rasterio.open(in_path, "r") as src:
        sel_bands = normalise_band_selection(src, bands, in_path)

    means, stds, _, _, _ = compute_global_band_stats(
        input_img=in_path,
        bands=sel_bands,
        no_data_val=no_data_val,
        sampling=100,
        use_pbar=use_pbar,
    )
    params = {
        "means": means,
        "stds": stds,
        "seam_stds": stds,
        "scale": float(scale),
        "compactness": float(compactness),
        "shape_weight": float(shape_weight),
        "use_numba": bool(use_numba),
    }
    seam_thres = min(2.0, max(0.3, float(scale) / 25.0))
    run_tiled_raster_pipeline(
        input_img=in_path,
        output_img=output_img,
        bands=sel_bands,
        worker_fn=_region_merge_tile_worker,
        params=params,
        tile_size=tile_size,
        overlap=overlap,
        n_jobs=n_jobs,
        stitch_seams=stitch_seams,
        seam_mode="spectral",
        seam_threshold=seam_thres,
        min_pixels=int(min_pixels),
        use_numba=use_numba,
        gdalformat=gdalformat,
        datatype=datatype,
        no_data_val=no_data_val,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )
