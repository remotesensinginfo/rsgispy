"""
Core utilities, optional ``numba`` acceleration, post-processing, and downstream
``rsgispy`` integration helpers for :mod:`rsgispy.segmentation`.
"""

import os
import tempfile
from typing import Any, List, Optional, Sequence, Tuple, Union

import geopandas as gpd
import numpy as np
import rasterio

import rsgispy
from rsgispy.imageprocess import (
    IntersectionGrid,
    OnlineStatsAccumulator,
    build_valid_cube_mask,
    ensure_file_exists,
    get_default_block_size,
    iter_dataset_windows,
    open_output_dataset,
    resolve_out_dtype,
    validate_band_index,
)
from rsgispy._numba import (
    HAVE_NUMBA,
    NUMBA_INSTALLED,
    make_njit,
    numba_enabled,
    warn_numba_fallback,
)
from rsgispy.imageutils.stats import pop_img_stats

__all__ = [
    "HAVE_NUMBA",
    "is_numba_available",
    "relabel_segments",
    "relabel_segments_array",
    "eliminate_small_segments",
    "eliminate_small_segments_array",
    "polygonise_segments",
    "extract_segment_stats",
]

_DEFAULT_SAMPLE_CAP = 250_000


def is_numba_available() -> bool:
    """
    Return whether optional ``numba`` JIT acceleration is installed.

    Returns
    -------
    bool
        ``True`` if :mod:`numba` can be imported, ``False`` otherwise.

    Examples
    --------
    >>> import rsgispy.segmentation.utils as seg_utils
    >>> isinstance(seg_utils.is_numba_available(), bool)
    True
    """
    return HAVE_NUMBA


def call_kernel(kernel: Any, use_numba: bool, *args: Any) -> Any:
    """
    Dispatch to a Numba-compiled kernel or its pure-Python fallback.

    Emits a :class:`RuntimeWarning` when ``use_numba=True`` is requested but
    ``numba`` is not installed; stays silent when ``use_numba=False`` is
    explicitly passed or acceleration is disabled via
    :func:`rsgispy._numba.disable_numba` / ``RSGISPY_DISABLE_NUMBA``.
    """
    if use_numba:
        if HAVE_NUMBA:
            if numba_enabled():
                return kernel(*args)
        elif not NUMBA_INSTALLED:
            warn_numba_fallback(
                getattr(kernel, "__name__", "segmentation"),
                stacklevel=3,
            )
    py_fn = getattr(kernel, "py_func", kernel)
    return py_fn(*args)


def worker_init_single_thread() -> None:
    """
    Initialise worker processes for multi-process execution.

    Pins BLAS/OpenMP/NumExpr thread pools to ``1`` inside worker processes
    to prevent CPU oversubscription when ``n_jobs > 1``. All internal
    ``numba`` kernels run without ``parallel=True`` so they remain single-
    threaded without mutating ``NUMBA_NUM_THREADS`` post-import.
    """
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"


def validate_n_jobs(n_jobs: int) -> int:
    """Validate and return integer worker count ``>= 1``."""
    if isinstance(n_jobs, bool) or not isinstance(
        n_jobs, (int, np.integer)
    ):
        raise rsgispy.RSGISPyException(
            f"n_jobs must be a positive integer, got {n_jobs!r}."
        )
    jobs = int(n_jobs)
    if jobs < 1:
        raise rsgispy.RSGISPyException(
            f"n_jobs must be >= 1, got {jobs}."
        )
    return jobs


def resolve_label_dtype(
    datatype: Optional[int],
) -> Tuple[int, str, np.dtype]:
    """
    Resolve an optional ``rsgispy.TYPE_*`` constant for integer label rasters,
    defaulting to :data:`rsgispy.TYPE_32UINT`.
    """
    return resolve_out_dtype(
        datatype,
        default_dtype=rsgispy.TYPE_32UINT,
        require_int=True,
    )


def check_label_overflow(max_label: int, out_np_dtype: np.dtype) -> None:
    """Raise :class:`rsgispy.RSGISPyException` if ``max_label`` overflows ``out_np_dtype``."""
    if max_label <= 0:
        return
    dtype_max = int(np.iinfo(out_np_dtype).max)
    if max_label > dtype_max:
        raise rsgispy.RSGISPyException(
            f"Maximum segment ID ({max_label}) exceeds maximum representable "
            f"value ({dtype_max}) for output datatype '{out_np_dtype}'."
        )


def resolve_input_nodata(
    src: rasterio.io.DatasetReader,
    no_data_val: Optional[float],
) -> Optional[float]:
    """Resolve explicit ``no_data_val`` or fall back to ``src.nodata``."""
    if no_data_val is not None:
        return float(no_data_val)
    return float(src.nodata) if src.nodata is not None else None


def normalise_band_selection(
    src: rasterio.io.DatasetReader,
    bands: Optional[Sequence[int]],
    img_path: str,
) -> List[int]:
    """Validate and return 1-based band indices to read from ``src``."""
    if bands is None:
        return list(range(1, src.count + 1))
    band_lst = [int(b) for b in bands]
    if not band_lst:
        raise rsgispy.RSGISPyException("bands must not be empty.")
    for b_idx in band_lst:
        validate_band_index(b_idx, src.count, img_path)
    return band_lst


def normalise_array_input(
    data: np.ndarray,
    no_data_val: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalise an input 2D ``(H, W)`` or 3D ``(B, H, W)`` NumPy array to
    ``(float64 (B, H, W), bool (H, W) valid_mask)``.
    """
    arr = np.asarray(data)
    if arr.ndim == 2:
        arr_3d = arr[np.newaxis, :, :].astype(np.float64, copy=False)
    elif arr.ndim == 3:
        arr_3d = arr.astype(np.float64, copy=False)
    else:
        raise rsgispy.RSGISPyException(
            f"Input array must be 2D (H, W) or 3D (B, H, W), got shape {arr.shape}."
        )
    if arr_3d.shape[0] == 0 or arr_3d.shape[1] == 0 or arr_3d.shape[2] == 0:
        raise rsgispy.RSGISPyException(
            f"Input array must be non-empty, got shape {arr.shape}."
        )

    valid_mask = build_valid_cube_mask(arr_3d, no_data_val=no_data_val, axis=0)
    return arr_3d, valid_mask


def standardise_bands(
    arr_3d: np.ndarray,
    valid_mask: np.ndarray,
    means: Optional[np.ndarray] = None,
    stds: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardise each band to zero mean and unit standard deviation over valid
    pixels. Invalid pixels are set to ``0.0`` in the returned copy.
    """
    n_bands = arr_3d.shape[0]
    out = np.zeros_like(arr_3d, dtype=np.float64)
    has_valid = bool(np.any(valid_mask))

    if means is None or stds is None:
        comp_means = np.zeros(n_bands, dtype=np.float64)
        comp_stds = np.ones(n_bands, dtype=np.float64)
        if has_valid:
            for b_idx in range(n_bands):
                vals = arr_3d[b_idx][valid_mask]
                comp_means[b_idx] = float(np.mean(vals))
                std_val = float(np.std(vals))
                comp_stds[b_idx] = std_val if std_val > 1e-12 else 1.0
    else:
        comp_means = np.asarray(means, dtype=np.float64)
        comp_stds = np.where(
            np.asarray(stds, dtype=np.float64) > 1e-12,
            np.asarray(stds, dtype=np.float64),
            1.0,
        )

    if has_valid:
        for b_idx in range(n_bands):
            out[b_idx][valid_mask] = (
                arr_3d[b_idx][valid_mask] - comp_means[b_idx]
            ) / comp_stds[b_idx]
    return out, comp_means, comp_stds


def scale_bands_to_range(
    arr_3d: np.ndarray,
    valid_mask: np.ndarray,
    lows: Optional[np.ndarray] = None,
    highs: Optional[np.ndarray] = None,
    target_max: float = 255.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Linearly scale each band from ``[low, high]`` into ``[0, target_max]`` so
    scale-dependent segmentation thresholds behave consistently across uint8,
    uint16 (0–10000), and float reflectance (0–1) imagery.
    """
    n_bands = arr_3d.shape[0]
    out = np.zeros_like(arr_3d, dtype=np.float64)
    has_valid = bool(np.any(valid_mask))

    if lows is None or highs is None:
        comp_lows = np.zeros(n_bands, dtype=np.float64)
        comp_highs = np.ones(n_bands, dtype=np.float64)
        if has_valid:
            for b_idx in range(n_bands):
                vals = arr_3d[b_idx][valid_mask]
                lo = float(np.percentile(vals, 2.0))
                hi = float(np.percentile(vals, 98.0))
                if hi <= lo:
                    lo = float(np.min(vals))
                    hi = float(np.max(vals))
                if hi <= lo:
                    hi = lo + 1.0
                comp_lows[b_idx] = lo
                comp_highs[b_idx] = hi
    else:
        comp_lows = np.asarray(lows, dtype=np.float64)
        comp_highs = np.asarray(highs, dtype=np.float64)

    if has_valid:
        for b_idx in range(n_bands):
            span = max(float(comp_highs[b_idx] - comp_lows[b_idx]), 1e-12)
            scaled = (arr_3d[b_idx][valid_mask] - comp_lows[b_idx]) / span
            out[b_idx][valid_mask] = np.clip(scaled, 0.0, 1.0) * target_max
    return out, comp_lows, comp_highs


def gaussian_smooth_nodata(
    arr_3d: np.ndarray,
    valid_mask: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """
    Apply nodata-aware spatial Gaussian smoothing to each band of ``arr_3d``
    ``(B, H, W)`` without bleeding masked values into valid pixels.
    """
    if sigma <= 0.0:
        return np.where(valid_mask[np.newaxis, :, :], arr_3d, 0.0)

    from scipy.ndimage import gaussian_filter  # pylint: disable=import-outside-toplevel

    weight = valid_mask.astype(np.float64)
    smooth_w = gaussian_filter(weight, sigma=float(sigma), mode="nearest")
    out = np.zeros_like(arr_3d, dtype=np.float64)
    valid_weight = valid_mask & (smooth_w > 1e-12)
    for b_idx in range(arr_3d.shape[0]):
        filled = np.where(valid_mask, arr_3d[b_idx], 0.0)
        smooth_b = gaussian_filter(filled, sigma=float(sigma), mode="nearest")
        out[b_idx, valid_weight] = (
            smooth_b[valid_weight] / smooth_w[valid_weight]
        )
    return out


def compute_multiband_gradient(
    arr_3d: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """
    Compute nodata-aware L2 multi-band gradient magnitude ``(H, W)`` using
    valid central/one-sided differences so nodata edges do not create spikes.
    """
    n_bands, height, width = arr_3d.shape
    grad_sq = np.zeros((height, width), dtype=np.float64)

    for b_idx in range(n_bands):
        band = arr_3d[b_idx]
        dy = np.zeros((height, width), dtype=np.float64)
        dx = np.zeros((height, width), dtype=np.float64)

        if height > 1:
            both_y = valid_mask[:-1, :] & valid_mask[1:, :]
            diff_y = np.where(both_y, band[1:, :] - band[:-1, :], 0.0)
            dy[:-1, :] += 0.5 * diff_y
            dy[1:, :] += 0.5 * diff_y
            # Double one-sided differences where the opposite neighbour is invalid
            only_down = both_y & (~np.vstack([np.zeros((1, width), dtype=bool), both_y[:-1, :]]))
            only_up = both_y & (~np.vstack([both_y[1:, :], np.zeros((1, width), dtype=bool)]))
            dy[:-1, :] = np.where(only_down, diff_y, dy[:-1, :])
            dy[1:, :] = np.where(only_up, diff_y, dy[1:, :])

        if width > 1:
            both_x = valid_mask[:, :-1] & valid_mask[:, 1:]
            diff_x = np.where(both_x, band[:, 1:] - band[:, :-1], 0.0)
            dx[:, :-1] += 0.5 * diff_x
            dx[:, 1:] += 0.5 * diff_x
            only_right = both_x & (~np.hstack([np.zeros((height, 1), dtype=bool), both_x[:, :-1]]))
            only_left = both_x & (~np.hstack([both_x[:, 1:], np.zeros((height, 1), dtype=bool)]))
            dx[:, :-1] = np.where(only_right, diff_x, dx[:, :-1])
            dx[:, 1:] = np.where(only_left, diff_x, dx[:, 1:])

        grad_sq += dy * dy + dx * dx

    return np.where(valid_mask, np.sqrt(grad_sq / max(1, n_bands)), 0.0)


def compute_global_band_stats(
    input_img: str,
    bands: Sequence[int],
    no_data_val: Optional[float] = None,
    sampling: int = 1,
    max_samples: int = _DEFAULT_SAMPLE_CAP,
    use_pbar: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Stream ``input_img`` block-by-block to compute exact global band means,
    standard deviations, min/max, and an optional deterministic subsampled
    feature matrix ``(n_samples, n_bands)``.
    """
    # pylint: disable=too-many-locals
    n_bands = len(bands)
    accum = OnlineStatsAccumulator(n_bands)
    sample_chunks: List[np.ndarray] = []
    stride = max(1, int(sampling))
    total_sampled = 0

    with rasterio.open(input_img, "r") as src:
        eff_nd = resolve_input_nodata(src, no_data_val)
        for win in iter_dataset_windows(
            src.height, src.width, use_pbar=use_pbar
        ):
            block = src.read(list(bands), window=win, out_dtype=np.float64)
            v_mask = build_valid_cube_mask(block, no_data_val=eff_nd, axis=0)

            if not np.any(v_mask):
                continue

            flat_valid = block[:, v_mask].T  # (n_valid, n_bands)
            accum.update_matrix(flat_valid)

            if total_sampled < max_samples:
                sub = flat_valid[::stride]
                if sub.shape[0] > 0:
                    take = min(sub.shape[0], max_samples - total_sampled)
                    sample_chunks.append(sub[:take])
                    total_sampled += take

    if int(accum.counts[0]) == 0:
        return (
            np.zeros(n_bands, dtype=np.float64),
            np.ones(n_bands, dtype=np.float64),
            np.zeros(n_bands, dtype=np.float64),
            np.ones(n_bands, dtype=np.float64),
            np.empty((0, n_bands), dtype=np.float64),
        )

    band_mins = accum.mins.copy()
    band_maxs = accum.maxs.copy()
    means = accum.means.copy()
    stds = accum.stds()
    stds = np.where(stds > 1e-12, stds, 1.0)
    samples = (
        np.vstack(sample_chunks)
        if sample_chunks
        else np.empty((0, n_bands), dtype=np.float64)
    )
    return means, stds, band_mins, band_maxs, samples


def write_label_raster_from_array(
    labels: np.ndarray,
    ref_img: str,
    output_img: str,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Write a 2D integer label array ``labels`` aligned to ``ref_img`` into
    ``output_img`` and optionally populate statistics/pyramids.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    _, rio_dtype, np_dtype = resolve_label_dtype(datatype)
    max_id = int(labels.max()) if labels.size > 0 else 0
    check_label_overflow(max_id, np_dtype)

    out_dir = os.path.dirname(os.path.abspath(output_img))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with rasterio.open(ref_img, "r") as src:
        grid = IntersectionGrid.from_dataset(src)

    out_arr = labels.astype(np_dtype, copy=False)
    with rasterio.Env():
        from contextlib import ExitStack  # pylint: disable=import-outside-toplevel

        with ExitStack() as stack:
            dst = open_output_dataset(
                stack=stack,
                output_img=output_img,
                gdalformat=gdalformat,
                out_rio_dtype=rio_dtype,
                grid=grid,
                count=1,
            )
            dst.nodata = 0
            dst.write(out_arr, 1)

    if calc_stats:
        pop_img_stats(
            output_img,
            no_data_val=0,
            calc_pyramids=True,
            use_pbar=use_pbar,
        )


def relabel_segments_array(
    labels: np.ndarray,
    no_data_val: int = 0,
    contiguous: bool = True,
) -> np.ndarray:
    """
    Relabel an integer segmentation array so valid regions have sequential
    positive IDs ``1..N`` (preserving ``no_data_val``).

    Parameters
    ----------
    labels : numpy.ndarray
        2D integer array of segment labels.
    no_data_val : int, optional
        Label value representing masked/invalid pixels (Default = ``0``).
    contiguous : bool, optional
        If ``True`` (Default), remap sorted unique valid IDs strictly to
        ``1..N``. If ``False``, preserve positive IDs and normalise any
        pixels matching ``no_data_val`` to ``0``.

    Returns
    -------
    numpy.ndarray
        Relabelled 2D array with dtype ``uint32`` (or ``int64`` if ``N`` exceeds
        ``uint32`` range).

    Examples
    --------
    >>> import numpy as np
    >>> from rsgispy.segmentation.utils import relabel_segments_array
    >>> arr = np.array([[0, 10, 10], [5, 5, 20]], dtype=np.int32)
    >>> relabel_segments_array(arr).tolist()
    [[0, 2, 2], [1, 1, 3]]
    """
    arr = np.asarray(labels)
    if arr.ndim != 2:
        raise rsgispy.RSGISPyException(
            f"labels must be a 2D array, got shape {arr.shape}."
        )
    if not contiguous:
        out = arr.astype(np.int64, copy=True)
        if no_data_val != 0:
            out[out == no_data_val] = 0
        out[out < 0] = 0
        max_v = int(out.max()) if out.size > 0 else 0
        return out.astype(
            np.uint32 if max_v <= np.iinfo(np.uint32).max else np.int64
        )

    valid = arr != no_data_val
    if no_data_val >= 0:
        valid &= arr > 0
    if not np.any(valid):
        return np.zeros(arr.shape, dtype=np.uint32)

    uniq = np.unique(arr[valid])
    n_seg = int(uniq.size)
    out_dtype = (
        np.uint32 if n_seg <= np.iinfo(np.uint32).max else np.int64
    )
    max_raw = int(uniq[-1])
    if max_raw <= 10_000_000 and int(uniq[0]) >= 0:
        lut = np.zeros(max_raw + 1, dtype=out_dtype)
        lut[uniq] = np.arange(1, n_seg + 1, dtype=out_dtype)
        out = np.zeros(arr.shape, dtype=out_dtype)
        out[valid] = lut[arr[valid]]
        return out

    idx = np.searchsorted(uniq, arr[valid]) + 1
    out = np.zeros(arr.shape, dtype=out_dtype)
    out[valid] = idx.astype(out_dtype)
    return out


def relabel_segments(
    input_img: Union[str, os.PathLike],
    output_img: Union[str, os.PathLike],
    contiguous: bool = True,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Stream-relabel a segmentation raster so valid segment IDs are contiguous
    integers ``1..N`` with no-data pixels mapped to ``0``.

    Parameters
    ----------
    input_img : str or os.PathLike
        Path to the input segmentation label raster (band 1).
    output_img : str or os.PathLike
        Path to the output relabelled raster.
    contiguous : bool, optional
        If ``True`` (Default), map unique valid IDs densely to ``1..N``. If
        ``False``, copy valid positive IDs while normalising ``no_data_val``
        to ``0``.
    gdalformat : str, optional
        Output GDAL/rasterio driver name (Default = ``'GTIFF'``).
    datatype : int or None, optional
        Optional ``rsgispy.TYPE_*`` integer output type (Default = ``None``,
        resolves to :data:`rsgispy.TYPE_32UINT`).
    no_data_val : float or None, optional
        Input pixel value treated as no-data and mapped to ``0``
        (Default = ``None``, falls back to ``input_img`` nodata or ``0``).
    calc_stats : bool, optional
        If ``True`` (Default), compute output statistics and pyramids.
    use_pbar : bool, optional
        If ``True`` (Default), display a ``tqdm`` progress bar.

    Examples
    --------
    >>> import rsgispy.segmentation
    >>> # rsgispy.segmentation.relabel_segments("raw_segs.tif", "clean_segs.tif")
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    in_path = os.fspath(input_img)
    out_path = os.fspath(output_img)
    ensure_file_exists(in_path, label="Input segmentation image")

    _, rio_dtype, np_dtype = resolve_label_dtype(datatype)
    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with rasterio.open(in_path, "r") as src:
        eff_nd = resolve_input_nodata(src, no_data_val)
        nd_int = 0 if eff_nd is None else int(eff_nd)
        grid = IntersectionGrid.from_dataset(src)

        uniq_chunks: List[np.ndarray] = []
        for win in iter_dataset_windows(
            src.height, src.width, use_pbar=use_pbar
        ):
            blk = src.read(1, window=win)
            v_mask = blk != nd_int
            if nd_int >= 0:
                v_mask &= blk > 0
            if np.any(v_mask):
                uniq_chunks.append(np.unique(blk[v_mask]))

        uniq = (
            np.unique(np.concatenate(uniq_chunks))
            if uniq_chunks
            else np.empty(0, dtype=np.int64)
        )
        n_seg = int(uniq.size)
        check_label_overflow(
            n_seg if contiguous else (int(uniq[-1]) if n_seg else 0),
            np_dtype,
        )

        from contextlib import ExitStack  # pylint: disable=import-outside-toplevel

        with ExitStack() as stack:
            dst = open_output_dataset(
                stack=stack,
                output_img=out_path,
                gdalformat=gdalformat,
                out_rio_dtype=rio_dtype,
                grid=grid,
                count=1,
            )
            dst.nodata = 0
            for win in iter_dataset_windows(
                src.height, src.width, use_pbar=use_pbar
            ):
                blk = src.read(1, window=win)
                v_mask = blk != nd_int
                if nd_int >= 0:
                    v_mask &= blk > 0
                out_blk = np.zeros(blk.shape, dtype=np_dtype)
                if np.any(v_mask):
                    if contiguous:
                        mapped = np.searchsorted(uniq, blk[v_mask]) + 1
                        out_blk[v_mask] = mapped.astype(np_dtype)
                    else:
                        out_blk[v_mask] = blk[v_mask].astype(np_dtype)
                dst.write(out_blk, 1, window=win)

    if calc_stats:
        pop_img_stats(
            out_path, no_data_val=0, calc_pyramids=True, use_pbar=use_pbar
        )


@make_njit
def _eliminate_small_regions_kernel(
    u_arr: np.ndarray,
    v_arr: np.ndarray,
    counts: np.ndarray,
    band_sums: np.ndarray,
    has_data: bool,
    min_pixels: int,
    max_passes: int = 32,
) -> np.ndarray:
    """
    Iteratively merge regions with ``0 < counts[r] < min_pixels`` into their
    best adjacent neighbour (minimum spectral Euclidean distance when
    ``has_data`` is True, else largest neighbour) using union-find.

    Returns ``parent`` array where isolated sub-threshold roots with no valid
    neighbours have ``counts[root] < min_pixels``.
    """
    # pylint: disable=too-many-arguments,too-many-branches
    n_regions = counts.shape[0]
    n_edges = u_arr.shape[0]
    n_bands = band_sums.shape[1]
    parent = np.arange(n_regions, dtype=np.int64)

    for _ in range(max_passes):
        best_nbr = np.zeros(n_regions, dtype=np.int64)
        best_score = np.full(n_regions, 1e300, dtype=np.float64)

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
                if has_data:
                    dist_sq = 0.0
                    inv_u = 1.0 / float(counts[ru])
                    inv_v = 1.0 / float(counts[rv])
                    for b_i in range(n_bands):
                        diff = (
                            band_sums[ru, b_i] * inv_u
                            - band_sums[rv, b_i] * inv_v
                        )
                        dist_sq += diff * diff
                    score_u = dist_sq
                    score_v = dist_sq
                else:
                    score_u = -float(counts[rv])
                    score_v = -float(counts[ru])

                if counts[ru] < min_pixels:
                    if score_u < best_score[ru] or (
                        score_u == best_score[ru]
                        and (best_nbr[ru] == 0 or rv < best_nbr[ru])
                    ):
                        best_score[ru] = score_u
                        best_nbr[ru] = rv

                if counts[rv] < min_pixels:
                    if score_v < best_score[rv] or (
                        score_v == best_score[rv]
                        and (best_nbr[rv] == 0 or ru < best_nbr[rv])
                    ):
                        best_score[rv] = score_v
                        best_nbr[rv] = ru

        merged_any = False
        for r_i in range(1, n_regions):
            if parent[r_i] != r_i or counts[r_i] <= 0 or counts[r_i] >= min_pixels:
                continue
            target = best_nbr[r_i]
            if target <= 0:
                continue
            while parent[target] != target:
                parent[target] = parent[parent[target]]
                target = parent[target]
            if target == r_i or counts[target] <= 0:
                continue

            # Keep larger region as root (tie-break to lower ID)
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
            if has_data:
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

    return parent


def extract_unique_region_edges(labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Extract sorted unique 4-connected region adjacency pairs ``(u, v)`` with ``0 < u < v``."""
    h_left = labels[:, :-1]
    h_right = labels[:, 1:]
    h_mask = (h_left > 0) & (h_right > 0) & (h_left != h_right)

    v_top = labels[:-1, :]
    v_bot = labels[1:, :]
    v_mask = (v_top > 0) & (v_bot > 0) & (v_top != v_bot)

    u_parts: List[np.ndarray] = []
    v_parts: List[np.ndarray] = []
    if np.any(h_mask):
        a = h_left[h_mask]
        b = h_right[h_mask]
        u_parts.append(np.minimum(a, b))
        v_parts.append(np.maximum(a, b))
    if np.any(v_mask):
        a = v_top[v_mask]
        b = v_bot[v_mask]
        u_parts.append(np.minimum(a, b))
        v_parts.append(np.maximum(a, b))

    if not u_parts:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    u_all = np.concatenate(u_parts).astype(np.int64, copy=False)
    v_all = np.concatenate(v_parts).astype(np.int64, copy=False)
    max_id = max(int(u_all.max()), int(v_all.max())) + 1
    keys = u_all * max_id + v_all
    uniq_keys = np.unique(keys)
    return uniq_keys // max_id, uniq_keys % max_id


def eliminate_small_segments_array(
    labels: np.ndarray,
    data: Optional[np.ndarray] = None,
    min_pixels: int = 10,
    no_data_val: int = 0,
    use_numba: bool = True,
) -> np.ndarray:
    """
    Eliminate segments smaller than ``min_pixels`` by merging them into their
    best 4-connected neighbour and relabelling contiguously from ``1``.

    When ``data`` is supplied, each small segment is merged into the adjacent
    neighbour with the closest mean spectral vector. When ``data`` is ``None``,
    each small segment is merged into its largest adjacent neighbour. Isolated
    sub-threshold islands with no valid neighbours are set to ``0``.

    Parameters
    ----------
    labels : numpy.ndarray
        2D integer array of segment IDs.
    data : numpy.ndarray or None, optional
        Optional 2D ``(H, W)`` or 3D ``(B, H, W)`` spectral array used to rank
        adjacent neighbours by Euclidean mean-spectrum distance.
    min_pixels : int, optional
        Minimum segment pixel count to retain (Default = ``10``).
    no_data_val : int, optional
        Label value treated as masked/invalid background (Default = ``0``).
    use_numba : bool, optional
        Use ``numba``-compiled merging loops when available (Default = ``True``).

    Returns
    -------
    numpy.ndarray
        Cleaned, contiguously relabelled 2D array (``0`` for nodata).

    Examples
    --------
    >>> import numpy as np
    >>> from rsgispy.segmentation.utils import eliminate_small_segments_array
    >>> lbl = np.array([[1, 1, 2], [1, 1, 1], [1, 1, 1]], dtype=np.int32)
    >>> eliminate_small_segments_array(lbl, min_pixels=4).tolist()
    [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    """
    # pylint: disable=too-many-locals
    if min_pixels <= 1:
        return relabel_segments_array(
            labels, no_data_val=no_data_val, contiguous=True
        )

    compact = relabel_segments_array(
        labels, no_data_val=no_data_val, contiguous=True
    ).astype(np.int64, copy=False)
    n_seg = int(compact.max()) if compact.size > 0 else 0
    if n_seg == 0:
        return np.zeros(compact.shape, dtype=np.uint32)

    counts = np.bincount(compact.ravel(), minlength=n_seg + 1).astype(np.int64)
    counts[0] = 0
    if not np.any((counts[1:] > 0) & (counts[1:] < min_pixels)):
        return relabel_segments_array(compact, no_data_val=0, contiguous=True)

    has_data = data is not None
    if has_data:
        arr_3d, _ = normalise_array_input(data, no_data_val=None)
        if arr_3d.shape[1:] != compact.shape:
            raise rsgispy.RSGISPyException(
                f"data spatial shape {arr_3d.shape[1:]} does not match "
                f"labels shape {compact.shape}."
            )
        n_bands = arr_3d.shape[0]
        band_sums = np.zeros((n_seg + 1, n_bands), dtype=np.float64)
        flat_lbl = compact.ravel()
        for b_i in range(n_bands):
            band_vals = np.where(
                np.isfinite(arr_3d[b_i]), arr_3d[b_i], 0.0
            ).ravel()
            band_sums[:, b_i] = np.bincount(
                flat_lbl, weights=band_vals, minlength=n_seg + 1
            )
    else:
        band_sums = np.zeros((n_seg + 1, 1), dtype=np.float64)

    u_arr, v_arr = extract_unique_region_edges(compact)
    parent = call_kernel(
        _eliminate_small_regions_kernel,
        use_numba,
        u_arr,
        v_arr,
        counts,
        band_sums,
        has_data,
        int(min_pixels),
        32,
    )

    lut = np.zeros(n_seg + 1, dtype=np.int64)
    for r_i in range(1, n_seg + 1):
        root = int(parent[r_i])
        if counts[root] >= min_pixels:
            lut[r_i] = root
    mapped = lut[compact]
    return relabel_segments_array(mapped, no_data_val=0, contiguous=True)


def eliminate_small_segments(
    input_img: Union[str, os.PathLike],
    output_img: Union[str, os.PathLike],
    ref_img: Optional[Union[str, os.PathLike]] = None,
    min_pixels: int = 10,
    bands: Optional[Sequence[int]] = None,
    use_numba: bool = True,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Eliminate segments smaller than ``min_pixels`` from a segmentation raster
    and write the cleaned, contiguously relabelled output raster.

    Parameters
    ----------
    input_img : str or os.PathLike
        Path to the input label raster (band 1).
    output_img : str or os.PathLike
        Path to the output cleaned label raster.
    ref_img : str or os.PathLike or None, optional
        Optional spectral reference image aligned to ``input_img`` used to
        merge small segments into their spectrally closest neighbour. When
        ``None``, small segments merge into their largest adjacent neighbour.
    min_pixels : int, optional
        Minimum segment size in pixels (Default = ``10``).
    bands : sequence of int or None, optional
        Optional 1-based band indices from ``ref_img`` (Default = all bands).
    use_numba : bool, optional
        Use ``numba`` acceleration when available (Default = ``True``).
    gdalformat : str, optional
        Output GDAL/rasterio driver name (Default = ``'GTIFF'``).
    datatype : int or None, optional
        Optional ``rsgispy.TYPE_*`` integer output type (Default = ``None``,
        resolves to :data:`rsgispy.TYPE_32UINT`).
    no_data_val : float or None, optional
        Input label value treated as nodata/background (Default = ``None``,
        falls back to ``input_img`` nodata or ``0``).
    calc_stats : bool, optional
        If ``True`` (Default), compute output statistics and pyramids.
    use_pbar : bool, optional
        If ``True`` (Default), display a ``tqdm`` progress bar.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    in_path = os.fspath(input_img)
    out_path = os.fspath(output_img)
    ensure_file_exists(in_path, label="Input segmentation image")

    with rasterio.open(in_path, "r") as src:
        eff_nd = resolve_input_nodata(src, no_data_val)
        nd_int = 0 if eff_nd is None else int(eff_nd)
        labels = src.read(1)

    ref_data: Optional[np.ndarray] = None
    if ref_img is not None:
        ref_path = os.fspath(ref_img)
        ensure_file_exists(ref_path, label="Reference spectral image")
        with rasterio.open(ref_path, "r") as ref_src:
            if (ref_src.height, ref_src.width) != labels.shape:
                raise rsgispy.RSGISPyException(
                    f"Reference image dimensions ({ref_src.height}, {ref_src.width}) "
                    f"do not match segmentation image ({labels.shape[0]}, {labels.shape[1]})."
                )
            sel_bands = normalise_band_selection(ref_src, bands, ref_path)
            ref_data = ref_src.read(sel_bands, out_dtype=np.float64)

    cleaned = eliminate_small_segments_array(
        labels=labels,
        data=ref_data,
        min_pixels=min_pixels,
        no_data_val=nd_int,
        use_numba=use_numba,
    )
    write_label_raster_from_array(
        labels=cleaned,
        ref_img=in_path,
        output_img=out_path,
        gdalformat=gdalformat,
        datatype=datatype,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )


def polygonise_segments(
    input_img: Union[str, os.PathLike],
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    out_format: str = "GPKG",
    img_band: int = 1,
    no_data_val: int = 0,
    segment_id_field: str = "seg_id",
    use_8_conn: bool = False,
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Polygonise a segmentation label raster into a vector layer using
    :func:`rsgispy.vector.create.polygonise_raster_to_vec_lyr`, excluding
    ``no_data_val`` pixels.

    Parameters
    ----------
    input_img : str or os.PathLike
        Path to the segmentation label raster.
    out_vec_file : str or os.PathLike or None, optional
        Output vector path, or ``None`` (Default) to return an in-memory
        :class:`geopandas.GeoDataFrame`.
    out_vec_lyr : str or None, optional
        Output layer name.
    out_format : str, optional
        Output OGR/vector format (Default = ``'GPKG'``).
    img_band : int, optional
        1-based band index containing segment IDs (Default = ``1``).
    no_data_val : int, optional
        Label value to omit from polygonisation (Default = ``0``).
    segment_id_field : str, optional
        Attribute field storing segment IDs (Default = ``'seg_id'``).
    use_8_conn : bool, optional
        Use 8-connectivity instead of 4-connectivity (Default = ``False``).
    out_compress : str or bool or None, optional
        Optional GeoParquet compression setting.

    Returns
    -------
    geopandas.GeoDataFrame or None
        In-memory GeoDataFrame when ``out_vec_file`` is ``None``, else ``None``.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    from rsgispy.vector.create import (  # pylint: disable=import-outside-toplevel
        polygonise_raster_to_vec_lyr,
    )
    from rsgispy.vectorprocess import (  # pylint: disable=import-outside-toplevel
        write_vec_gdf,
    )

    in_path = os.fspath(input_img)
    ensure_file_exists(in_path, label="Input segmentation image")

    with tempfile.TemporaryDirectory() as tmp_dir:
        mask_path = os.path.join(tmp_dir, "seg_mask.tif")
        with rasterio.open(in_path, "r") as src:
            grid = IntersectionGrid.from_dataset(src)
            from contextlib import ExitStack  # pylint: disable=import-outside-toplevel

            with ExitStack() as stack:
                msk_dst = open_output_dataset(
                    stack=stack,
                    output_img=mask_path,
                    gdalformat="GTIFF",
                    out_rio_dtype="uint8",
                    grid=grid,
                    count=1,
                )
                msk_dst.nodata = 0
                for win in iter_dataset_windows(
                    src.height, src.width, block_size=get_default_block_size(), use_pbar=False
                ):
                    blk = src.read(img_band, window=win)
                    msk_dst.write((blk != no_data_val).astype(np.uint8), 1, window=win)

        gdf = polygonise_raster_to_vec_lyr(
            out_vec_file=None,
            out_vec_lyr=out_vec_lyr,
            out_format=out_format,
            input_img=in_path,
            img_band=img_band,
            mask_img=mask_path,
            mask_band=1,
            pxl_val_fieldname=segment_id_field,
            use_8_conn=use_8_conn,
        )

    assert gdf is not None
    if out_vec_file is None:
        return gdf
    write_vec_gdf(
        gdf,
        out_vec_file=out_vec_file,
        out_vec_lyr=out_vec_lyr,
        out_format=out_format,
        out_compress=out_compress,
    )
    return None


def extract_segment_stats(
    segments_img: Union[str, os.PathLike],
    feature_img: Union[str, os.PathLike],
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    out_format: str = "GPKG",
    bands: Optional[Sequence[int]] = None,
    min_thres: float = -1e12,
    max_thres: float = 1e12,
    out_no_data_val: float = -9999.0,
    segment_id_field: str = "seg_id",
    use_pbar: bool = True,
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Polygonise ``segments_img`` and extract per-object band summary statistics
    (mean, stddev, min, max, count) from ``feature_img``.

    When ``feature_img`` is on the same pixel grid as ``segments_img`` the
    statistics are computed per segment ID directly from the label raster with
    :func:`rsgispy.zonalstats.labels.calc_label_zonal_stats` (much faster for
    large numbers of segments); otherwise the polygon zonal statistics in
    :func:`rsgispy.zonalstats.polygon.calc_zonal_band_stats_file` are used. In
    the fast path a segment ID made up of several disconnected parts receives
    the statistics of the whole segment on each of its polygons.

    Parameters
    ----------
    segments_img : str or os.PathLike
        Path to the input segmentation label raster.
    feature_img : str or os.PathLike
        Path to the multi-band imagery raster to summarise per segment.
    out_vec_file : str or os.PathLike or None, optional
        Output vector path, or ``None`` (Default) to return an in-memory
        :class:`geopandas.GeoDataFrame`.
    out_vec_lyr : str or None, optional
        Output vector layer name.
    out_format : str, optional
        Output vector format driver (Default = ``'GPKG'``).
    bands : sequence of int or None, optional
        1-based band indices of ``feature_img`` to summarise (Default = all).
    min_thres, max_thres : float, optional
        Inclusive valid pixel range in ``feature_img``.
    out_no_data_val : float, optional
        Fill value for segments without valid pixels (Default = ``-9999.0``).
    segment_id_field : str, optional
        Attribute column for segment IDs (Default = ``'seg_id'``).
    use_pbar : bool, optional
        Display progress bars (Default = ``True``).
    out_compress : str or bool or None, optional
        Optional GeoParquet compression setting.

    Returns
    -------
    geopandas.GeoDataFrame or None
        In-memory GeoDataFrame with per-band summary columns when
        ``out_vec_file`` is ``None``, else ``None``.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    from rsgispy.vectorprocess import (  # pylint: disable=import-outside-toplevel
        write_vec_gdf,
    )
    from rsgispy.zonalstats.polygon import (  # pylint: disable=import-outside-toplevel
        calc_zonal_band_stats_file,
    )

    from rsgispy.ml.stack import (  # pylint: disable=import-outside-toplevel
        BandSpec,
    )
    from rsgispy.zonalstats.labels import (  # pylint: disable=import-outside-toplevel
        _label_is_grid_compatible,
        calc_label_zonal_stats,
    )

    feat_path = os.fspath(feature_img)
    ensure_file_exists(feat_path, label="Feature image")
    with rasterio.open(feat_path, "r") as feat_src:
        sel_bands = normalise_band_selection(feat_src, bands, feat_path)

    gdf = polygonise_segments(
        segments_img,
        out_vec_file=None,
        segment_id_field=segment_id_field,
    )
    assert gdf is not None

    seg_path = os.fspath(segments_img)
    if _label_is_grid_compatible(seg_path, feat_path):
        # Fast path: one streamed pass over the label raster.
        stats_df = calc_label_zonal_stats(
            seg_path,
            BandSpec(
                file_name=feat_path,
                bands=tuple(sel_bands),
                band_names=tuple(f"b{b_idx}" for b_idx in sel_bands),
                min_vld_val=min_thres,
                max_vld_val=max_thres,
            ),
            stats=["min", "max", "mean", "stddev", "count"],
            label_no_data_val=0,
            out_no_data_val=out_no_data_val,
            id_col=segment_id_field,
            pxl_count_col=None,
            use_pbar=use_pbar,
        )
        rename = {}
        for b_idx in sel_bands:
            rename[f"b{b_idx}_stddev"] = f"b{b_idx}_std"
            rename[f"b{b_idx}_count"] = f"b{b_idx}_cnt"
        stats_df = stats_df.rename(columns=rename)
        for b_idx in sel_bands:
            stats_df[f"b{b_idx}_cnt"] = stats_df[f"b{b_idx}_cnt"].astype(np.float64)
        stats_df = stats_df.astype({segment_id_field: gdf[segment_id_field].dtype})
        gdf = gdf.merge(stats_df, on=segment_id_field, how="left")
        sel_bands = []

    for b_idx in sel_bands:
        gdf = calc_zonal_band_stats_file(
            vec_file=gdf,
            vec_lyr=None,
            input_img=feat_path,
            img_band=b_idx,
            min_thres=min_thres,
            max_thres=max_thres,
            out_no_data_val=out_no_data_val,
            min_field=f"b{b_idx}_min",
            max_field=f"b{b_idx}_max",
            mean_field=f"b{b_idx}_mean",
            stddev_field=f"b{b_idx}_std",
            count_field=f"b{b_idx}_cnt",
            use_pbar=use_pbar,
        )
        assert gdf is not None

    if out_vec_file is None:
        return gdf
    write_vec_gdf(
        gdf,
        out_vec_file=out_vec_file,
        out_vec_lyr=out_vec_lyr,
        out_format=out_format,
        out_compress=out_compress,
    )
    return None
