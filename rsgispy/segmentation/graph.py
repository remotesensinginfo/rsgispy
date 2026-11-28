"""
Felzenszwalb & Huttenlocher (2004) graph-based image segmentation for
multi-band remote sensing imagery.

Reference
---------
Felzenszwalb, P. F., & Huttenlocher, D. P. (2004). Efficient Graph-Based
Image Segmentation. *International Journal of Computer Vision*, 59(2),
167–181.
"""

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
    gaussian_smooth_nodata,
    make_njit,
    normalise_array_input,
    normalise_band_selection,
    scale_bands_to_range,
)

__all__ = [
    "graph_segmentation",
    "graph_segmentation_array",
]


@make_njit
def _felzenszwalb_merge_kernel(
    u_sorted: np.ndarray,
    v_sorted: np.ndarray,
    w_sorted: np.ndarray,
    n_pixels: int,
    scale: float,
    min_pixels: int,
) -> np.ndarray:
    """
    Execute Felzenszwalb–Huttenlocher minimum-spanning-tree region merging and
    small-component absorption over edges sorted by increasing weight.
    """
    parent = np.arange(n_pixels, dtype=np.int64)
    sizes = np.ones(n_pixels, dtype=np.int64)
    thresh = np.full(n_pixels, float(scale), dtype=np.float64)
    n_edges = u_sorted.shape[0]

    for e_i in range(n_edges):
        ru = u_sorted[e_i]
        while parent[ru] != ru:
            parent[ru] = parent[parent[ru]]
            ru = parent[ru]

        rv = v_sorted[e_i]
        while parent[rv] != rv:
            parent[rv] = parent[parent[rv]]
            rv = parent[rv]

        if ru == rv:
            continue

        w = w_sorted[e_i]
        if w <= thresh[ru] and w <= thresh[rv]:
            if sizes[ru] < sizes[rv] or (sizes[ru] == sizes[rv] and rv < ru):
                keep_id = rv
                drop_id = ru
            else:
                keep_id = ru
                drop_id = rv
            parent[drop_id] = keep_id
            sizes[keep_id] += sizes[drop_id]
            thresh[keep_id] = w + float(scale) / float(sizes[keep_id])

    if min_pixels > 1:
        for e_i in range(n_edges):
            ru = u_sorted[e_i]
            while parent[ru] != ru:
                parent[ru] = parent[parent[ru]]
                ru = parent[ru]
            rv = v_sorted[e_i]
            while parent[rv] != rv:
                parent[rv] = parent[parent[rv]]
                rv = parent[rv]
            if ru == rv:
                continue
            if sizes[ru] < min_pixels or sizes[rv] < min_pixels:
                if sizes[ru] < sizes[rv] or (
                    sizes[ru] == sizes[rv] and rv < ru
                ):
                    keep_id = rv
                    drop_id = ru
                else:
                    keep_id = ru
                    drop_id = rv
                parent[drop_id] = keep_id
                sizes[keep_id] += sizes[drop_id]

    for idx in range(n_pixels):
        root = idx
        while parent[root] != root:
            root = parent[root]
        parent[idx] = root
    return parent


def _graph_tile_worker(
    halo_arr: np.ndarray,
    halo_valid: np.ndarray,
    tile: TileWindow,
    params: Dict[str, Any],
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Worker running Felzenszwalb–Huttenlocher segmentation on one halo tile."""
    # pylint: disable=too-many-locals
    if not np.any(halo_valid):
        return np.zeros((tile.height, tile.width), dtype=np.uint32), None

    scaled_halo, _, _ = scale_bands_to_range(
        halo_arr,
        halo_valid,
        lows=params["lows"],
        highs=params["highs"],
        target_max=255.0,
    )
    sigma = float(params["sigma"])
    if sigma > 0.0:
        scaled_halo = gaussian_smooth_nodata(
            scaled_halo, halo_valid, sigma=sigma
        )

    n_bands, height, width = scaled_halo.shape
    valid_rows, valid_cols = np.nonzero(halo_valid)
    n_valid = int(valid_rows.size)
    node_grid = np.full((height, width), -1, dtype=np.int64)
    node_grid[valid_rows, valid_cols] = np.arange(n_valid, dtype=np.int64)

    u_parts = []
    v_parts = []
    w_parts = []

    if width > 1:
        h_mask = halo_valid[:, :-1] & halo_valid[:, 1:]
        if np.any(h_mask):
            diff_h = scaled_halo[:, :, :-1] - scaled_halo[:, :, 1:]
            dist_h = np.sqrt(np.sum(diff_h * diff_h, axis=0) / float(n_bands))
            u_parts.append(node_grid[:, :-1][h_mask])
            v_parts.append(node_grid[:, 1:][h_mask])
            w_parts.append(dist_h[h_mask])

    if height > 1:
        v_mask = halo_valid[:-1, :] & halo_valid[1:, :]
        if np.any(v_mask):
            diff_v = scaled_halo[:, :-1, :] - scaled_halo[:, 1:, :]
            dist_v = np.sqrt(np.sum(diff_v * diff_v, axis=0) / float(n_bands))
            u_parts.append(node_grid[:-1, :][v_mask])
            v_parts.append(node_grid[1:, :][v_mask])
            w_parts.append(dist_v[v_mask])

    if u_parts:
        u_all = np.concatenate(u_parts)
        v_all = np.concatenate(v_parts)
        w_all = np.concatenate(w_parts)
        order = np.argsort(w_all, kind="mergesort")
        parent = call_kernel(
            _felzenszwalb_merge_kernel,
            bool(params["use_numba"]),
            u_all[order],
            v_all[order],
            w_all[order],
            n_valid,
            float(params["scale"]),
            int(params["min_pixels"]),
        )
    else:
        parent = np.arange(n_valid, dtype=np.int64)

    halo_labels = np.zeros((height, width), dtype=np.int64)
    halo_labels[valid_rows, valid_cols] = parent + 1
    core_labels = split_core_components(crop_halo_labels(halo_labels, tile))
    return core_labels, None


def _validate_graph_params(scale: float, sigma: float, min_pixels: int) -> None:
    """Validate Felzenszwalb graph segmentation parameters."""
    if scale <= 0.0:
        raise rsgispy.RSGISPyException(f"scale must be > 0, got {scale}.")
    if sigma < 0.0:
        raise rsgispy.RSGISPyException(f"sigma must be >= 0, got {sigma}.")
    if min_pixels < 1:
        raise rsgispy.RSGISPyException(
            f"min_pixels must be >= 1, got {min_pixels}."
        )


def graph_segmentation_array(
    data: np.ndarray,
    scale: float = 100.0,
    sigma: float = 0.8,
    min_pixels: int = 50,
    tile_size: Optional[Union[int, Tuple[int, int]]] = None,
    overlap: int = 32,
    n_jobs: int = 1,
    stitch_seams: bool = True,
    use_numba: bool = True,
    no_data_val: Optional[float] = None,
    use_pbar: bool = False,
) -> np.ndarray:
    """
    Segment an in-memory 2D ``(H, W)`` or 3D ``(B, H, W)`` array using the
    Felzenszwalb–Huttenlocher (2004) graph-based segmentation algorithm.

    Parameters
    ----------
    data : numpy.ndarray
        Input 2D ``(H, W)`` or 3D ``(B, H, W)`` array.
    scale : float, optional
        Observation scale parameter ``k`` controlling segment merging
        (Default = ``100.0``). Higher values produce larger segments.
    sigma : float, optional
        Nodata-aware Gaussian smoothing standard deviation applied prior to
        building pixel-edge weights (Default = ``0.8``).
    min_pixels : int, optional
        Minimum segment size enforced during post-merge absorption
        (Default = ``50``).
    tile_size : int or tuple[int, int] or None, optional
        Optional core tile size ``(tile_h, tile_w)`` for tiled processing.
    overlap : int, optional
        Halo padding in pixels around each tile (Default = ``32``).
    n_jobs : int, optional
        Number of worker processes for parallel tiled execution
        (Default = ``1``).
    stitch_seams : bool, optional
        Spectrally merge matching regions across tile seams (Default = ``True``).
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
        2D ``uint32`` label array with ``1..N`` segment IDs and ``0`` for
        nodata.

    Examples
    --------
    >>> import numpy as np
    >>> from rsgispy.segmentation import graph_segmentation_array
    >>> img = np.zeros((2, 16, 16), dtype=np.float32)
    >>> img[:, :8, :] = 10.0
    >>> img[:, 8:, :] = 200.0
    >>> seg = graph_segmentation_array(img, scale=50.0, sigma=0.0, min_pixels=10)
    >>> int(seg.max())
    2
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    _validate_graph_params(scale, sigma, min_pixels)
    arr_3d, valid_mask = normalise_array_input(data, no_data_val=no_data_val)
    if not np.any(valid_mask):
        return np.zeros(arr_3d.shape[1:], dtype=np.uint32)

    scaled_arr, lows, highs = scale_bands_to_range(
        arr_3d, valid_mask, target_max=255.0
    )
    params = {
        "lows": np.zeros_like(lows),
        "highs": np.full_like(highs, 255.0),
        "scale": float(scale),
        "sigma": float(sigma),
        "min_pixels": int(min_pixels),
        "use_numba": bool(use_numba),
    }
    seam_thres = min(30.0, max(2.0, float(scale) * 0.15))
    return run_tiled_array_pipeline(
        arr_3d=scaled_arr,
        valid_mask=valid_mask,
        worker_fn=_graph_tile_worker,
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


def graph_segmentation(
    input_img: Union[str, os.PathLike],
    output_img: Union[str, os.PathLike],
    scale: float = 100.0,
    sigma: float = 0.8,
    min_pixels: int = 50,
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
    Segment a multi-band remote sensing raster using the Felzenszwalb–
    Huttenlocher (2004) graph-based segmentation algorithm.

    Parameters
    ----------
    input_img : str or os.PathLike
        Path to the input multi-band raster.
    output_img : str or os.PathLike
        Path to the output segmentation label raster.
    scale : float, optional
        Observation scale parameter ``k`` controlling segment coarseness
        (Default = ``100.0``).
    sigma : float, optional
        Nodata-aware Gaussian smoothing sigma prior to edge weight calculation
        (Default = ``0.8``).
    min_pixels : int, optional
        Minimum segment size in pixels (Default = ``50``).
    bands : sequence of int or None, optional
        1-based band indices to include (Default = all bands).
    tile_size : int or tuple[int, int] or None, optional
        Core tile size ``(tile_h, tile_w)`` for bounded-memory execution.
    overlap : int, optional
        Halo padding in pixels around each tile (Default = ``32``).
    n_jobs : int, optional
        Number of worker processes for parallel tiled execution
        (Default = ``1``).
    stitch_seams : bool, optional
        Reconcile spectrally matching regions across tile seams
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
    >>> # rsgispy.segmentation.graph_segmentation(
    ... #     "sentinel2.tif", "graph_segs.tif", scale=100.0, min_pixels=50
    ... # )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    _validate_graph_params(scale, sigma, min_pixels)
    in_path = os.fspath(input_img)
    if not os.path.exists(in_path):
        raise rsgispy.RSGISPyException(f"Input image does not exist: {in_path}")

    with rasterio.open(in_path, "r") as src:
        sel_bands = normalise_band_selection(src, bands, in_path)
        auto_stride = max(1, (int(src.height) * int(src.width)) // 200_000)

    _, _, band_mins, band_maxs, samples = compute_global_band_stats(
        input_img=in_path,
        bands=sel_bands,
        no_data_val=no_data_val,
        sampling=auto_stride,
        use_pbar=use_pbar,
    )
    if samples.shape[0] > 0:
        lows = np.percentile(samples, 2.0, axis=0)
        highs = np.percentile(samples, 98.0, axis=0)
        deg_mask = highs <= lows
        lows = np.where(deg_mask, band_mins, lows)
        highs = np.where(deg_mask, band_maxs, highs)
    else:
        lows = band_mins
        highs = band_maxs
    highs = np.where(highs <= lows, lows + 1.0, highs)

    params = {
        "lows": lows,
        "highs": highs,
        "seam_stds": (highs - lows) / 255.0,
        "scale": float(scale),
        "sigma": float(sigma),
        "min_pixels": int(min_pixels),
        "use_numba": bool(use_numba),
    }
    scaled_thres = min(30.0, max(2.0, float(scale) * 0.15))
    run_tiled_raster_pipeline(
        input_img=in_path,
        output_img=output_img,
        bands=sel_bands,
        worker_fn=_graph_tile_worker,
        params=params,
        tile_size=tile_size,
        overlap=overlap,
        n_jobs=n_jobs,
        stitch_seams=stitch_seams,
        seam_mode="spectral",
        seam_threshold=scaled_thres,
        min_pixels=int(min_pixels),
        use_numba=use_numba,
        gdalformat=gdalformat,
        datatype=datatype,
        no_data_val=no_data_val,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )
