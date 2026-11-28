"""
Shepherd et al. (2019) iterative K-means and small-segment elimination
segmentation algorithm for remote sensing imagery.

Reference
---------
Shepherd, J. D., Bunting, P., & Dymond, J. R. (2019). Operational Large-Scale
Segmentation of Imagery Based on Iterative Elimination. *Remote Sensing*,
11(6), 658.
"""

import os
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import rasterio

import rsgispy
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
    extract_unique_region_edges,
    make_njit,
    normalise_array_input,
    normalise_band_selection,
    standardise_bands,
)

__all__ = [
    "shepherd_segmentation",
    "shepherd_segmentation_array",
]


@make_njit
def _assign_kmeans_kernel(
    std_arr: np.ndarray,
    valid_mask: np.ndarray,
    centroids: np.ndarray,
) -> np.ndarray:
    """
    Assign each valid pixel in ``std_arr`` ``(B, H, W)`` to its nearest
    centroid (1-based cluster IDs ``1..K``, with ``0`` for invalid pixels).
    """
    n_bands, height, width = std_arr.shape
    n_clusters = centroids.shape[0]
    out = np.zeros((height, width), dtype=np.int32)
    for row in range(height):
        for col in range(width):
            if not valid_mask[row, col]:
                continue
            best_k = 0
            best_dist = 1e300
            for k_i in range(n_clusters):
                dist_sq = 0.0
                for b_i in range(n_bands):
                    diff = std_arr[b_i, row, col] - centroids[k_i, b_i]
                    dist_sq += diff * diff
                if dist_sq < best_dist:
                    best_dist = dist_sq
                    best_k = k_i + 1
            out[row, col] = best_k
    return out


@make_njit
def _shepherd_eliminate_kernel(
    u_arr: np.ndarray,
    v_arr: np.ndarray,
    counts: np.ndarray,
    band_sums: np.ndarray,
    min_pixels: int,
    max_dist_sq: float,
    max_iter: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Iteratively eliminate segments smaller than ``min_pixels`` by merging them
    into their spectrally closest adjacent neighbour (optionally constrained
    by ``max_dist_sq >= 0``), up to ``max_iter`` elimination passes.
    """
    # pylint: disable=too-many-arguments,too-many-branches
    n_regions = counts.shape[0]
    n_edges = u_arr.shape[0]
    n_bands = band_sums.shape[1]
    parent = np.arange(n_regions, dtype=np.int64)
    has_limit = max_dist_sq >= 0.0

    for _ in range(max_iter):
        best_nbr = np.zeros(n_regions, dtype=np.int64)
        best_dist = np.full(n_regions, 1e300, dtype=np.float64)

        for e_i in range(n_edges):
            ru = u_arr[e_i]
            while parent[ru] != ru:
                parent[ru] = parent[parent[ru]]
                ru = parent[ru]
            rv = v_arr[e_i]
            while parent[rv] != rv:
                parent[rv] = parent[parent[rv]]
                rv = parent[rv]
            if ru == rv or counts[ru] <= 0 or counts[rv] <= 0:
                continue

            if counts[ru] < min_pixels or counts[rv] < min_pixels:
                dist_sq = 0.0
                inv_u = 1.0 / float(counts[ru])
                inv_v = 1.0 / float(counts[rv])
                for b_i in range(n_bands):
                    diff = (
                        band_sums[ru, b_i] * inv_u
                        - band_sums[rv, b_i] * inv_v
                    )
                    dist_sq += diff * diff
                if has_limit and dist_sq > max_dist_sq:
                    continue

                if counts[ru] < min_pixels:
                    if dist_sq < best_dist[ru] or (
                        dist_sq == best_dist[ru]
                        and (best_nbr[ru] == 0 or rv < best_nbr[ru])
                    ):
                        best_dist[ru] = dist_sq
                        best_nbr[ru] = rv
                if counts[rv] < min_pixels:
                    if dist_sq < best_dist[rv] or (
                        dist_sq == best_dist[rv]
                        and (best_nbr[rv] == 0 or ru < best_nbr[rv])
                    ):
                        best_dist[rv] = dist_sq
                        best_nbr[rv] = ru

        merged_any = False
        for r_i in range(1, n_regions):
            if (
                parent[r_i] != r_i
                or counts[r_i] <= 0
                or counts[r_i] >= min_pixels
            ):
                continue
            target = best_nbr[r_i]
            if target <= 0:
                continue
            while parent[target] != target:
                parent[target] = parent[parent[target]]
                target = parent[target]
            if target == r_i or counts[target] <= 0:
                continue

            if counts[target] < counts[r_i] or (
                counts[target] == counts[r_i] and r_i < target
            ):
                keep_id = r_i
                drop_id = target
            else:
                keep_id = target
                drop_id = r_i

            parent[drop_id] = keep_id
            counts[keep_id] += counts[drop_id]
            counts[drop_id] = 0
            for b_i in range(n_bands):
                band_sums[keep_id, b_i] += band_sums[drop_id, b_i]
            merged_any = True

        if not merged_any:
            break

    for r_i in range(1, n_regions):
        root = r_i
        while parent[root] != root:
            root = parent[root]
        parent[r_i] = root

    return parent, counts


def _fit_kmeans_fallback(
    samples: np.ndarray,
    n_clusters: int,
    max_iter: int,
    rnd_seed: Optional[int],
) -> np.ndarray:
    """Deterministic pure-NumPy K-means fallback when scikit-learn is absent."""
    rng = np.random.default_rng(rnd_seed)
    n_samples = samples.shape[0]
    init_idx = rng.choice(n_samples, size=n_clusters, replace=False)
    centroids = samples[init_idx].copy()

    for _ in range(max_iter):
        dists = np.sum(
            (samples[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2,
            axis=2,
        )
        labels = np.argmin(dists, axis=1)
        new_centroids = np.empty_like(centroids)
        for k_i in range(n_clusters):
            members = samples[labels == k_i]
            if members.shape[0] > 0:
                new_centroids[k_i] = members.mean(axis=0)
            else:
                new_centroids[k_i] = centroids[k_i]
        if np.allclose(centroids, new_centroids, atol=1e-6):
            centroids = new_centroids
            break
        centroids = new_centroids
    return centroids


def _fit_kmeans_centroids(
    samples: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    n_clusters: int,
    max_iter: int,
    rnd_seed: Optional[int],
) -> np.ndarray:
    """Standardise global samples and fit K-means centroids deterministically."""
    if samples.shape[0] == 0:
        return np.empty((0, means.shape[0]), dtype=np.float64)

    std_samples = (samples - means[np.newaxis, :]) / stds[np.newaxis, :]
    unique_samples = np.unique(std_samples, axis=0)
    eff_k = min(int(n_clusters), int(unique_samples.shape[0]))
    if eff_k <= 1:
        return unique_samples[:1].astype(np.float64)

    try:
        from sklearn.cluster import KMeans  # pylint: disable=import-outside-toplevel

        km = KMeans(
            n_clusters=eff_k,
            max_iter=max(10, int(max_iter) * 4),
            n_init=3,
            random_state=rnd_seed,
        )
        km.fit(std_samples)
        return np.asarray(km.cluster_centers_, dtype=np.float64)
    except ImportError:
        return _fit_kmeans_fallback(
            std_samples, eff_k, max(10, int(max_iter) * 4), rnd_seed
        )


def _shepherd_tile_worker(
    halo_arr: np.ndarray,
    halo_valid: np.ndarray,
    tile: TileWindow,
    params: Dict[str, Any],
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Worker: quantise halo pixels to global K-means centroids, clump, and eliminate."""
    # pylint: disable=too-many-locals
    centroids: np.ndarray = params["centroids"]
    if centroids.shape[0] == 0 or not np.any(halo_valid):
        zeros = np.zeros((tile.height, tile.width), dtype=np.uint32)
        return zeros, zeros

    std_halo, _, _ = standardise_bands(
        halo_arr, halo_valid, means=params["means"], stds=params["stds"]
    )
    use_numba: bool = bool(params["use_numba"])
    cluster_map = call_kernel(
        _assign_kmeans_kernel, use_numba, std_halo, halo_valid, centroids
    )

    _, height, width = std_halo.shape
    horiz_eq = (
        (cluster_map[:, :-1] == cluster_map[:, 1:])
        & (cluster_map[:, :-1] > 0)
        & halo_valid[:, :-1]
        & halo_valid[:, 1:]
        if width > 1
        else np.empty((height, 0), dtype=bool)
    )
    vert_eq = (
        (cluster_map[:-1, :] == cluster_map[1:, :])
        & (cluster_map[:-1, :] > 0)
        & halo_valid[:-1, :]
        & halo_valid[1:, :]
        if height > 1
        else np.empty((0, width), dtype=bool)
    )
    halo_clumps, n_clumps, _ = label_4connected(
        halo_valid, horiz_eq, vert_eq, return_first_flat=False
    )
    if n_clumps == 0:
        zeros = np.zeros((tile.height, tile.width), dtype=np.uint32)
        return zeros, zeros

    min_pixels = int(params["min_pixels"])
    dist_thres: Optional[float] = params["dist_thres"]
    max_iter = int(params["max_iter"])

    if min_pixels > 1:
        counts = np.bincount(
            halo_clumps.ravel(), minlength=n_clumps + 1
        ).astype(np.int64)
        counts[0] = 0
        if np.any((counts[1:] > 0) & (counts[1:] < min_pixels)):
            n_bands = std_halo.shape[0]
            band_sums = np.zeros((n_clumps + 1, n_bands), dtype=np.float64)
            flat_lbl = halo_clumps.ravel()
            for b_i in range(n_bands):
                band_sums[:, b_i] = np.bincount(
                    flat_lbl,
                    weights=std_halo[b_i].ravel(),
                    minlength=n_clumps + 1,
                )
            u_arr, v_arr = extract_unique_region_edges(halo_clumps)
            max_dist_sq = (
                -1.0 if dist_thres is None else float(dist_thres) ** 2
            )
            parent, updated_counts = call_kernel(
                _shepherd_eliminate_kernel,
                use_numba,
                u_arr,
                v_arr,
                counts,
                band_sums,
                min_pixels,
                max_dist_sq,
                max_iter,
            )
            lut = np.zeros(n_clumps + 1, dtype=np.int64)
            for r_i in range(1, n_clumps + 1):
                root = int(parent[r_i])
                if dist_thres is not None or updated_counts[root] >= min_pixels:
                    lut[r_i] = root
            halo_clumps = lut[halo_clumps]

    core_labels = split_core_components(crop_halo_labels(halo_clumps, tile))
    core_clusters = crop_halo_labels(cluster_map, tile)
    core_clusters = np.where(core_labels > 0, core_clusters, 0)
    return core_labels, core_clusters


def _validate_shepherd_params(
    n_clusters: int,
    min_pixels: int,
    dist_thres: Optional[float],
    sampling: int,
    max_iter: int,
) -> None:
    """Validate Shepherd segmentation parameters."""
    if n_clusters < 1:
        raise rsgispy.RSGISPyException(
            f"n_clusters must be >= 1, got {n_clusters}."
        )
    if min_pixels < 1:
        raise rsgispy.RSGISPyException(
            f"min_pixels must be >= 1, got {min_pixels}."
        )
    if dist_thres is not None and dist_thres < 0.0:
        raise rsgispy.RSGISPyException(
            f"dist_thres must be >= 0 or None, got {dist_thres}."
        )
    if sampling < 1:
        raise rsgispy.RSGISPyException(
            f"sampling must be >= 1, got {sampling}."
        )
    if max_iter < 1:
        raise rsgispy.RSGISPyException(
            f"max_iter must be >= 1, got {max_iter}."
        )


def shepherd_segmentation_array(
    data: np.ndarray,
    n_clusters: int = 60,
    min_pixels: int = 100,
    dist_thres: Optional[float] = None,
    sampling: int = 100,
    max_iter: int = 50,
    tile_size: Optional[Union[int, Tuple[int, int]]] = None,
    overlap: int = 32,
    n_jobs: int = 1,
    stitch_seams: bool = True,
    rnd_seed: Optional[int] = 42,
    use_numba: bool = True,
    no_data_val: Optional[float] = None,
    use_pbar: bool = False,
) -> np.ndarray:
    """
    Segment an in-memory 2D ``(H, W)`` or 3D ``(B, H, W)`` array using the
    Shepherd et al. (2019) iterative K-means and small-segment elimination
    algorithm.

    Parameters
    ----------
    data : numpy.ndarray
        Input 2D ``(H, W)`` or 3D ``(B, H, W)`` array.
    n_clusters : int, optional
        Number of K-means spectral clusters used to quantise pixels before
        connected-component clumping (Default = ``60``).
    min_pixels : int, optional
        Minimum segment size in pixels (Default = ``100``).
    dist_thres : float or None, optional
        Maximum standardised Euclidean spectral distance allowed when merging
        small segments during elimination, or ``None`` (Default) for unlimited
        merging into the closest neighbour.
    sampling : int, optional
        Pixel subsampling stride when fitting global K-means centroids
        (Default = ``100``).
    max_iter : int, optional
        Maximum elimination iterations (Default = ``50``).
    tile_size : int or tuple[int, int] or None, optional
        Optional core tile size ``(tile_h, tile_w)`` for tiled processing.
    overlap : int, optional
        Halo padding in pixels around each tile (Default = ``32``).
    n_jobs : int, optional
        Number of worker processes for tiled execution (Default = ``1``).
    stitch_seams : bool, optional
        Merge identical K-means cluster segments across tile boundaries
        (Default = ``True``).
    rnd_seed : int or None, optional
        Deterministic random seed for K-means initialisation (Default = ``42``).
    use_numba : bool, optional
        Enable optional ``numba`` JIT acceleration when installed
        (Default = ``True``).
    no_data_val : float or None, optional
        Pixel value masked as nodata (output label ``0``).
    use_pbar : bool, optional
        Display a progress bar across tiles (Default = ``False``).

    Returns
    -------
    numpy.ndarray
        2D ``uint32`` label array with ``1..N`` segment IDs and ``0`` for nodata.

    Notes
    -----
    Global band standardisation statistics and K-means centroids are estimated
    across the scene first so all tiles share an identical spectral codebook
    before parallel per-tile assignment and elimination.

    Examples
    --------
    >>> import numpy as np
    >>> from rsgispy.segmentation import shepherd_segmentation_array
    >>> img = np.zeros((2, 16, 16), dtype=np.float32)
    >>> img[:, :8, :] = 0.2
    >>> img[:, 8:, :] = 0.8
    >>> seg = shepherd_segmentation_array(img, n_clusters=2, min_pixels=10, sampling=1)
    >>> int(seg.max())
    2
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    _validate_shepherd_params(
        n_clusters, min_pixels, dist_thres, sampling, max_iter
    )
    arr_3d, valid_mask = normalise_array_input(data, no_data_val=no_data_val)
    if not np.any(valid_mask):
        return np.zeros(arr_3d.shape[1:], dtype=np.uint32)

    _, means, stds = standardise_bands(arr_3d, valid_mask)
    flat_valid = arr_3d[:, valid_mask].T
    samples = flat_valid[:: max(1, int(sampling))]
    if samples.shape[0] == 0:
        samples = flat_valid[:1]

    centroids = _fit_kmeans_centroids(
        samples=samples,
        means=means,
        stds=stds,
        n_clusters=n_clusters,
        max_iter=max_iter,
        rnd_seed=rnd_seed,
    )
    params = {
        "centroids": centroids,
        "means": means,
        "stds": stds,
        "seam_stds": stds,
        "min_pixels": int(min_pixels),
        "dist_thres": dist_thres,
        "max_iter": int(max_iter),
        "use_numba": bool(use_numba),
    }
    pipeline_min_px = 1 if dist_thres is not None else int(min_pixels)
    return run_tiled_array_pipeline(
        arr_3d=arr_3d,
        valid_mask=valid_mask,
        worker_fn=_shepherd_tile_worker,
        params=params,
        tile_size=tile_size,
        overlap=overlap,
        n_jobs=n_jobs,
        stitch_seams=stitch_seams,
        seam_mode="exact_aux",
        min_pixels=pipeline_min_px,
        use_numba=use_numba,
        use_pbar=use_pbar,
    )


def shepherd_segmentation(
    input_img: Union[str, os.PathLike],
    output_img: Union[str, os.PathLike],
    n_clusters: int = 60,
    min_pixels: int = 100,
    dist_thres: Optional[float] = None,
    sampling: int = 100,
    max_iter: int = 50,
    bands: Optional[Sequence[int]] = None,
    tile_size: Optional[Union[int, Tuple[int, int]]] = None,
    overlap: int = 32,
    n_jobs: int = 1,
    stitch_seams: bool = True,
    rnd_seed: Optional[int] = 42,
    use_numba: bool = True,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Segment a multi-band remote sensing raster using the Shepherd et al.
    (2019) iterative K-means and small-segment elimination algorithm.

    Parameters
    ----------
    input_img : str or os.PathLike
        Path to the input multi-band raster.
    output_img : str or os.PathLike
        Path to the output segmentation label raster.
    n_clusters : int, optional
        Number of K-means clusters (Default = ``60``).
    min_pixels : int, optional
        Minimum segment size in pixels (Default = ``100``).
    dist_thres : float or None, optional
        Maximum standardised Euclidean distance permitted when merging small
        segments during elimination (Default = ``None``).
    sampling : int, optional
        Subsampling stride used when estimating global K-means centroids
        (Default = ``100``).
    max_iter : int, optional
        Maximum elimination passes (Default = ``50``).
    bands : sequence of int or None, optional
        1-based band indices to include (Default = all bands).
    tile_size : int or tuple[int, int] or None, optional
        Core tile dimensions ``(tile_h, tile_w)`` for bounded-memory tiled
        execution (Default = ``None``; automatically tiles scenes larger than
        16M pixels).
    overlap : int, optional
        Halo overlap pixels around each tile (Default = ``32``).
    n_jobs : int, optional
        Number of worker processes for parallel tile processing
        (Default = ``1``).
    stitch_seams : bool, optional
        Reconnect segments sharing the same K-means cluster across tile seams
        (Default = ``True``).
    rnd_seed : int or None, optional
        Random seed for deterministic K-means clustering (Default = ``42``).
    use_numba : bool, optional
        Enable optional ``numba`` JIT acceleration when available
        (Default = ``True``).
    gdalformat : str, optional
        Output GDAL/rasterio driver name (Default = ``'GTIFF'``).
    datatype : int or None, optional
        Optional ``rsgispy.TYPE_*`` integer output type (Default = ``None``,
        resolves to :data:`rsgispy.TYPE_32UINT`).
    no_data_val : float or None, optional
        Input pixel value treated as nodata (Default = ``None``, uses source
        nodata metadata).
    calc_stats : bool, optional
        If ``True`` (Default), compute statistics and image pyramids.
    use_pbar : bool, optional
        If ``True`` (Default), display ``tqdm`` progress bars.

    Notes
    -----
    Stage 1 streams ``input_img`` block-by-block to compute global band means,
    standard deviations, and subsampled pixels for K-means clustering so all
    tiles share the same centroids. Stage 2 executes per-tile assignment,
    clumping, and elimination in parallel using ``n_jobs``.

    Examples
    --------
    >>> import rsgispy.segmentation
    >>> # rsgispy.segmentation.shepherd_segmentation(
    ... #     "sentinel2.tif", "segments.tif", n_clusters=60, min_pixels=100
    ... # )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    _validate_shepherd_params(
        n_clusters, min_pixels, dist_thres, sampling, max_iter
    )
    in_path = os.fspath(input_img)
    if not os.path.exists(in_path):
        raise rsgispy.RSGISPyException(f"Input image does not exist: {in_path}")

    with rasterio.open(in_path, "r") as src:
        sel_bands = normalise_band_selection(src, bands, in_path)

    means, stds, _, _, samples = compute_global_band_stats(
        input_img=in_path,
        bands=sel_bands,
        no_data_val=no_data_val,
        sampling=sampling,
        use_pbar=use_pbar,
    )
    centroids = _fit_kmeans_centroids(
        samples=samples,
        means=means,
        stds=stds,
        n_clusters=n_clusters,
        max_iter=max_iter,
        rnd_seed=rnd_seed,
    )
    params = {
        "centroids": centroids,
        "means": means,
        "stds": stds,
        "seam_stds": stds,
        "min_pixels": int(min_pixels),
        "dist_thres": dist_thres,
        "max_iter": int(max_iter),
        "use_numba": bool(use_numba),
    }
    pipeline_min_px = 1 if dist_thres is not None else int(min_pixels)
    run_tiled_raster_pipeline(
        input_img=in_path,
        output_img=output_img,
        bands=sel_bands,
        worker_fn=_shepherd_tile_worker,
        params=params,
        tile_size=tile_size,
        overlap=overlap,
        n_jobs=n_jobs,
        stitch_seams=stitch_seams,
        seam_mode="exact_aux",
        min_pixels=pipeline_min_px,
        use_numba=use_numba,
        gdalformat=gdalformat,
        datatype=datatype,
        no_data_val=no_data_val,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )
