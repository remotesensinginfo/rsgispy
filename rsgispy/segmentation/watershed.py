"""
Seeded / marker-controlled compact watershed segmentation over multi-band
spectral gradient surfaces for remote sensing imagery.
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
    extract_halo_tile,
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
    relabel_segments_array,
    standardise_bands,
)

__all__ = [
    "watershed_segmentation",
    "watershed_segmentation_array",
]


@make_njit
def _suppress_local_minima_kernel(
    cand_rows: np.ndarray,
    cand_cols: np.ndarray,
    height: int,
    width: int,
    min_distance: int,
) -> np.ndarray:
    """
    Greedily filter candidate gradient-minimum pixels sorted by ascending
    gradient so kept seeds are separated by at least ``min_distance`` pixels.
    """
    n_cand = cand_rows.shape[0]
    markers = np.zeros((height, width), dtype=np.int64)
    blocked = np.zeros((height, width), dtype=np.uint8)
    next_id = 1
    rad = max(1, min_distance)

    for idx in range(n_cand):
        r = cand_rows[idx]
        c = cand_cols[idx]
        if blocked[r, c] != 0:
            continue
        markers[r, c] = next_id
        next_id += 1
        r0 = max(0, r - rad)
        r1 = min(height, r + rad + 1)
        c0 = max(0, c - rad)
        c1 = min(width, c + rad + 1)
        for rr in range(r0, r1):
            for cc in range(c0, c1):
                blocked[rr, cc] = 1

    return markers


@make_njit
def _compact_watershed_kernel(
    grad: np.ndarray,
    valid_mask: np.ndarray,
    markers: np.ndarray,
    compactness: float,
) -> np.ndarray:
    """
    Compact Meyer priority-queue watershed using a binary min-heap on
    ``(cost, fifo_age)`` over 4-connected valid pixels.
    """
    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    height, width = grad.shape
    labels = markers.copy()
    max_lbl = 0
    n_valid = 0
    for r in range(height):
        for c in range(width):
            if valid_mask[r, c]:
                n_valid += 1
                if labels[r, c] > max_lbl:
                    max_lbl = labels[r, c]
            else:
                labels[r, c] = 0

    if max_lbl <= 0 or n_valid == 0:
        return labels

    # Mean seed anchor coordinates for compactness distance penalty
    seed_sum_r = np.zeros(max_lbl + 1, dtype=np.float64)
    seed_sum_c = np.zeros(max_lbl + 1, dtype=np.float64)
    seed_cnt = np.zeros(max_lbl + 1, dtype=np.int64)
    for r in range(height):
        for c in range(width):
            lbl = labels[r, c]
            if lbl > 0:
                seed_sum_r[lbl] += float(r)
                seed_sum_c[lbl] += float(c)
                seed_cnt[lbl] += 1

    seed_r = np.zeros(max_lbl + 1, dtype=np.float64)
    seed_c = np.zeros(max_lbl + 1, dtype=np.float64)
    for lbl in range(1, max_lbl + 1):
        if seed_cnt[lbl] > 0:
            inv = 1.0 / float(seed_cnt[lbl])
            seed_r[lbl] = seed_sum_r[lbl] * inv
            seed_c[lbl] = seed_sum_c[lbl] * inv

    cap = 4 * n_valid + 16
    h_cost = np.empty(cap, dtype=np.float64)
    h_age = np.empty(cap, dtype=np.int64)
    h_r = np.empty(cap, dtype=np.int64)
    h_c = np.empty(cap, dtype=np.int64)
    h_lbl = np.empty(cap, dtype=np.int64)
    h_size = 0
    age_counter = 0

    # Push initial neighbours of all seed pixels
    for r in range(height):
        for c in range(width):
            lbl = labels[r, c]
            if lbl <= 0:
                continue
            sr = seed_r[lbl]
            sc = seed_c[lbl]
            for d_i in range(4):
                if d_i == 0:
                    nr = r - 1
                    nc = c
                elif d_i == 1:
                    nr = r + 1
                    nc = c
                elif d_i == 2:
                    nr = r
                    nc = c - 1
                else:
                    nr = r
                    nc = c + 1
                if (
                    0 <= nr < height
                    and 0 <= nc < width
                    and valid_mask[nr, nc]
                    and labels[nr, nc] == 0
                ):
                    cost = grad[nr, nc]
                    if compactness > 0.0:
                        dr = float(nr) - sr
                        dc = float(nc) - sc
                        cost += compactness * math.sqrt(dr * dr + dc * dc)
                    idx = h_size
                    h_size += 1
                    h_cost[idx] = cost
                    h_age[idx] = age_counter
                    age_counter += 1
                    h_r[idx] = nr
                    h_c[idx] = nc
                    h_lbl[idx] = lbl
                    while idx > 0:
                        par = (idx - 1) // 2
                        if h_cost[idx] < h_cost[par] or (
                            h_cost[idx] == h_cost[par]
                            and h_age[idx] < h_age[par]
                        ):
                            tc = h_cost[par]
                            ta = h_age[par]
                            tr = h_r[par]
                            tcol = h_c[par]
                            tl = h_lbl[par]
                            h_cost[par] = h_cost[idx]
                            h_age[par] = h_age[idx]
                            h_r[par] = h_r[idx]
                            h_c[par] = h_c[idx]
                            h_lbl[par] = h_lbl[idx]
                            h_cost[idx] = tc
                            h_age[idx] = ta
                            h_r[idx] = tr
                            h_c[idx] = tcol
                            h_lbl[idx] = tl
                            idx = par
                        else:
                            break

    while h_size > 0:
        cur_r = h_r[0]
        cur_c = h_c[0]
        cur_lbl = h_lbl[0]
        h_size -= 1
        if h_size > 0:
            h_cost[0] = h_cost[h_size]
            h_age[0] = h_age[h_size]
            h_r[0] = h_r[h_size]
            h_c[0] = h_c[h_size]
            h_lbl[0] = h_lbl[h_size]
            idx = 0
            while True:
                left = 2 * idx + 1
                if left >= h_size:
                    break
                right = left + 1
                smallest = left
                if right < h_size and (
                    h_cost[right] < h_cost[left]
                    or (
                        h_cost[right] == h_cost[left]
                        and h_age[right] < h_age[left]
                    )
                ):
                    smallest = right
                if h_cost[smallest] < h_cost[idx] or (
                    h_cost[smallest] == h_cost[idx]
                    and h_age[smallest] < h_age[idx]
                ):
                    tc = h_cost[idx]
                    ta = h_age[idx]
                    tr = h_r[idx]
                    tcol = h_c[idx]
                    tl = h_lbl[idx]
                    h_cost[idx] = h_cost[smallest]
                    h_age[idx] = h_age[smallest]
                    h_r[idx] = h_r[smallest]
                    h_c[idx] = h_c[smallest]
                    h_lbl[idx] = h_lbl[smallest]
                    h_cost[smallest] = tc
                    h_age[smallest] = ta
                    h_r[smallest] = tr
                    h_c[smallest] = tcol
                    h_lbl[smallest] = tl
                    idx = smallest
                else:
                    break

        if labels[cur_r, cur_c] != 0:
            continue
        labels[cur_r, cur_c] = cur_lbl
        sr = seed_r[cur_lbl]
        sc = seed_c[cur_lbl]

        for d_i in range(4):
            if d_i == 0:
                nr = cur_r - 1
                nc = cur_c
            elif d_i == 1:
                nr = cur_r + 1
                nc = cur_c
            elif d_i == 2:
                nr = cur_r
                nc = cur_c - 1
            else:
                nr = cur_r
                nc = cur_c + 1
            if (
                0 <= nr < height
                and 0 <= nc < width
                and valid_mask[nr, nc]
                and labels[nr, nc] == 0
            ):
                cost = grad[nr, nc]
                if compactness > 0.0:
                    dr = float(nr) - sr
                    dc = float(nc) - sc
                    cost += compactness * math.sqrt(dr * dr + dc * dc)
                idx = h_size
                h_size += 1
                h_cost[idx] = cost
                h_age[idx] = age_counter
                age_counter += 1
                h_r[idx] = nr
                h_c[idx] = nc
                h_lbl[idx] = cur_lbl
                while idx > 0:
                    par = (idx - 1) // 2
                    if h_cost[idx] < h_cost[par] or (
                        h_cost[idx] == h_cost[par] and h_age[idx] < h_age[par]
                    ):
                        tc = h_cost[par]
                        ta = h_age[par]
                        tr = h_r[par]
                        tcol = h_c[par]
                        tl = h_lbl[par]
                        h_cost[par] = h_cost[idx]
                        h_age[par] = h_age[idx]
                        h_r[par] = h_r[idx]
                        h_c[par] = h_c[idx]
                        h_lbl[par] = h_lbl[idx]
                        h_cost[idx] = tc
                        h_age[idx] = ta
                        h_r[idx] = tr
                        h_c[idx] = tcol
                        h_lbl[idx] = tl
                        idx = par
                    else:
                        break

    return labels


def _auto_generate_markers(
    grad: np.ndarray,
    valid_mask: np.ndarray,
    min_distance: int,
    use_numba: bool,
) -> np.ndarray:
    """Detect local gradient minima separated by ``min_distance`` pixels."""
    from scipy.ndimage import minimum_filter  # pylint: disable=import-outside-toplevel

    height, width = grad.shape
    grad_inf = np.where(valid_mask, grad, np.inf)
    win = max(3, 2 * int(min_distance) + 1)
    local_min = minimum_filter(grad_inf, size=win, mode="nearest")
    is_cand = valid_mask & (grad_inf <= local_min + 1e-12)
    cand_r, cand_c = np.nonzero(is_cand)
    if cand_r.size == 0:
        cand_r, cand_c = np.nonzero(valid_mask)

    cand_g = grad[cand_r, cand_c]
    order = np.lexsort((cand_c, cand_r, cand_g))
    return call_kernel(
        _suppress_local_minima_kernel,
        use_numba,
        cand_r[order],
        cand_c[order],
        height,
        width,
        int(min_distance),
    )


def _watershed_tile_worker(
    halo_arr: np.ndarray,
    halo_valid: np.ndarray,
    tile: TileWindow,
    params: Dict[str, Any],
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Worker running gradient computation and compact watershed on a halo tile."""
    # pylint: disable=too-many-locals
    if not np.any(halo_valid):
        zeros = np.zeros((tile.height, tile.width), dtype=np.uint32)
        return zeros, zeros

    std_halo, _, _ = standardise_bands(
        halo_arr, halo_valid, means=params["means"], stds=params["stds"]
    )
    sigma = float(params["gradient_sigma"])
    if sigma > 0.0:
        std_halo = gaussian_smooth_nodata(std_halo, halo_valid, sigma=sigma)

    grad = compute_multiband_gradient(std_halo, halo_valid)
    use_numba = bool(params["use_numba"])
    has_user_markers = False

    if params.get("markers_arr") is not None:
        markers_halo = extract_halo_tile(params["markers_arr"], tile).astype(
            np.int64, copy=False
        )
        markers_halo = np.where(halo_valid & (markers_halo > 0), markers_halo, 0)
        has_user_markers = True
    elif params.get("markers_img") is not None:
        with rasterio.open(params["markers_img"], "r") as msrc:
            markers_halo = msrc.read(1, window=tile.halo_rio_window()).astype(
                np.int64, copy=False
            )
        markers_halo = np.where(halo_valid & (markers_halo > 0), markers_halo, 0)
        has_user_markers = True
    else:
        markers_halo = _auto_generate_markers(
            grad=grad,
            valid_mask=halo_valid,
            min_distance=int(params["min_distance"]),
            use_numba=use_numba,
        )

    halo_labels = call_kernel(
        _compact_watershed_kernel,
        use_numba,
        grad,
        halo_valid,
        markers_halo,
        float(params["compactness"]),
    )

    # Ensure any isolated valid islands not reached in auto-seeded mode get labelled
    if not has_user_markers:
        unreached = halo_valid & (halo_labels == 0)
        if np.any(unreached):
            next_seed = int(halo_labels.max()) + 1
            extra_markers = _auto_generate_markers(
                grad=grad,
                valid_mask=unreached,
                min_distance=int(params["min_distance"]),
                use_numba=use_numba,
            )
            extra_markers = np.where(
                extra_markers > 0, extra_markers + next_seed - 1, 0
            )
            extra_labels = call_kernel(
                _compact_watershed_kernel,
                use_numba,
                grad,
                unreached,
                extra_markers,
                float(params["compactness"]),
            )
            halo_labels = np.where(unreached, extra_labels, halo_labels)

    cropped = crop_halo_labels(halo_labels, tile)
    if has_user_markers:
        aux_core = np.where(cropped > 0, cropped, 0).astype(np.int64)
        core_labels = relabel_segments_array(
            cropped, no_data_val=0, contiguous=True
        )
        return core_labels, aux_core

    core_labels = relabel_segments_array(
        cropped, no_data_val=0, contiguous=True
    )
    return core_labels, None


def _validate_watershed_params(
    min_distance: int,
    gradient_sigma: float,
    compactness: float,
    min_pixels: int,
) -> None:
    """Validate watershed segmentation parameters."""
    if min_distance < 1:
        raise rsgispy.RSGISPyException(
            f"min_distance must be >= 1, got {min_distance}."
        )
    if gradient_sigma < 0.0:
        raise rsgispy.RSGISPyException(
            f"gradient_sigma must be >= 0, got {gradient_sigma}."
        )
    if compactness < 0.0:
        raise rsgispy.RSGISPyException(
            f"compactness must be >= 0, got {compactness}."
        )
    if min_pixels < 1:
        raise rsgispy.RSGISPyException(
            f"min_pixels must be >= 1, got {min_pixels}."
        )


def watershed_segmentation_array(
    data: np.ndarray,
    markers: Optional[np.ndarray] = None,
    min_distance: int = 10,
    gradient_sigma: float = 1.0,
    compactness: float = 0.0,
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
    Segment an in-memory 2D ``(H, W)`` or 3D ``(B, H, W)`` array using seeded
    or automatically seeded compact watershed over a multi-band gradient
    magnitude surface.

    Parameters
    ----------
    data : numpy.ndarray
        Input 2D ``(H, W)`` or 3D ``(B, H, W)`` array.
    markers : numpy.ndarray or None, optional
        Optional 2D positive-integer marker array matching ``(H, W)``; ``<= 0``
        indicates unseeded pixels. When ``None`` (Default), seeds are detected
        automatically from local minima of the gradient surface.
    min_distance : int, optional
        Minimum spatial spacing in pixels between auto-detected seeds
        (Default = ``10``). Ignored when ``markers`` is provided.
    gradient_sigma : float, optional
        Nodata-aware Gaussian smoothing standard deviation applied before
        gradient computation (Default = ``1.0``).
    compactness : float, optional
        Spatial compactness penalty added to gradient flooding priority
        (Default = ``0.0``, standard watershed).
    min_pixels : int, optional
        Minimum segment size in pixels after post-watershed elimination
        (Default = ``10``).
    tile_size : int or tuple[int, int] or None, optional
        Optional core tile size ``(tile_h, tile_w)`` for tiled processing.
    overlap : int, optional
        Halo padding in pixels around each tile (Default = ``32``).
    n_jobs : int, optional
        Number of worker processes for tiled execution (Default = ``1``).
    stitch_seams : bool, optional
        Reconcile segments across tile seams (Default = ``True``). Uses exact
        marker ID matching when ``markers`` is provided, else spectral seam
        reconciliation.
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
    >>> from rsgispy.segmentation import watershed_segmentation_array
    >>> img = np.zeros((16, 16), dtype=np.float32)
    >>> img[:, 8:] = 10.0
    >>> markers = np.zeros((16, 16), dtype=np.int32)
    >>> markers[8, 2] = 1
    >>> markers[8, 13] = 2
    >>> seg = watershed_segmentation_array(img, markers=markers, min_pixels=1)
    >>> int(seg.max())
    2
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    _validate_watershed_params(
        min_distance, gradient_sigma, compactness, min_pixels
    )
    arr_3d, valid_mask = normalise_array_input(data, no_data_val=no_data_val)
    if markers is not None:
        m_arr = np.asarray(markers)
        if m_arr.shape != arr_3d.shape[1:]:
            raise rsgispy.RSGISPyException(
                f"markers shape {m_arr.shape} does not match input spatial "
                f"shape {arr_3d.shape[1:]}."
            )
    if not np.any(valid_mask):
        return np.zeros(arr_3d.shape[1:], dtype=np.uint32)

    std_arr, means, stds = standardise_bands(arr_3d, valid_mask)
    params: Dict[str, Any] = {
        "means": np.zeros_like(means),
        "stds": np.ones_like(stds),
        "markers_arr": markers,
        "min_distance": int(min_distance),
        "gradient_sigma": float(gradient_sigma),
        "compactness": float(compactness),
        "use_numba": bool(use_numba),
    }
    seam_mode = "exact_aux" if markers is not None else "spectral"
    return run_tiled_array_pipeline(
        arr_3d=std_arr,
        valid_mask=valid_mask,
        worker_fn=_watershed_tile_worker,
        params=params,
        tile_size=tile_size,
        overlap=overlap,
        n_jobs=n_jobs,
        stitch_seams=stitch_seams,
        seam_mode=seam_mode,
        seam_threshold=0.75,
        min_pixels=int(min_pixels),
        use_numba=use_numba,
        use_pbar=use_pbar,
    )


def watershed_segmentation(
    input_img: Union[str, os.PathLike],
    output_img: Union[str, os.PathLike],
    markers_img: Optional[Union[str, os.PathLike]] = None,
    min_distance: int = 10,
    gradient_sigma: float = 1.0,
    compactness: float = 0.0,
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
    Segment a multi-band remote sensing raster using seeded or auto-seeded
    compact watershed over a multi-band gradient surface.

    Parameters
    ----------
    input_img : str or os.PathLike
        Path to the input multi-band raster.
    output_img : str or os.PathLike
        Path to the output segmentation label raster.
    markers_img : str or os.PathLike or None, optional
        Optional single-band positive-integer marker raster aligned to
        ``input_img``. When ``None`` (Default), seeds are auto-detected from
        local minima of the smoothed gradient magnitude.
    min_distance : int, optional
        Minimum spatial spacing in pixels between auto-detected seeds
        (Default = ``10``).
    gradient_sigma : float, optional
        Nodata-aware Gaussian smoothing sigma prior to gradient magnitude
        computation (Default = ``1.0``).
    compactness : float, optional
        Spatial compactness weight added to flooding cost (Default = ``0.0``).
    min_pixels : int, optional
        Minimum segment size in pixels (Default = ``10``).
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
        Reconcile segments across tile seams (Default = ``True``).
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
    >>> # rsgispy.segmentation.watershed_segmentation(
    ... #     "sentinel2.tif", "ws_segs.tif", min_distance=10, gradient_sigma=1.0
    ... # )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    _validate_watershed_params(
        min_distance, gradient_sigma, compactness, min_pixels
    )
    in_path = os.fspath(input_img)
    if not os.path.exists(in_path):
        raise rsgispy.RSGISPyException(f"Input image does not exist: {in_path}")

    with rasterio.open(in_path, "r") as src:
        sel_bands = normalise_band_selection(src, bands, in_path)
        in_shape = (int(src.height), int(src.width))

    markers_path: Optional[str] = None
    if markers_img is not None:
        markers_path = os.fspath(markers_img)
        if not os.path.exists(markers_path):
            raise rsgispy.RSGISPyException(
                f"Markers image does not exist: {markers_path}"
            )
        with rasterio.open(markers_path, "r") as msrc:
            if (int(msrc.height), int(msrc.width)) != in_shape:
                raise rsgispy.RSGISPyException(
                    f"Markers image dimensions ({msrc.height}, {msrc.width}) "
                    f"do not match input image dimensions {in_shape}."
                )

    means, stds, _, _, _ = compute_global_band_stats(
        input_img=in_path,
        bands=sel_bands,
        no_data_val=no_data_val,
        sampling=100,
        use_pbar=use_pbar,
    )
    params: Dict[str, Any] = {
        "means": means,
        "stds": stds,
        "seam_stds": stds,
        "markers_img": markers_path,
        "min_distance": int(min_distance),
        "gradient_sigma": float(gradient_sigma),
        "compactness": float(compactness),
        "use_numba": bool(use_numba),
    }
    seam_mode = "exact_aux" if markers_path is not None else "spectral"
    run_tiled_raster_pipeline(
        input_img=in_path,
        output_img=output_img,
        bands=sel_bands,
        worker_fn=_watershed_tile_worker,
        params=params,
        tile_size=tile_size,
        overlap=overlap,
        n_jobs=n_jobs,
        stitch_seams=stitch_seams,
        seam_mode=seam_mode,
        seam_threshold=0.75,
        min_pixels=int(min_pixels),
        use_numba=use_numba,
        gdalformat=gdalformat,
        datatype=datatype,
        no_data_val=no_data_val,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )
