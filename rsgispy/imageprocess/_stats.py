"""
Online streaming band statistics and post-write statistics helpers for
:mod:`rsgispy.imageprocess`.
"""

from typing import Dict, Optional, Tuple

import numpy as np
from rasterio.io import DatasetReader, DatasetWriter

from rsgispy.imageprocess._io import get_output_stage, on_output_close
from rsgispy.imageprocess._masks import build_valid_mask


def maybe_pop_stats(
    output_img: str,
    calc_stats: bool,
    no_data_val: Optional[float] = None,
    thematic: bool = False,
    use_pbar: bool = True,
    ignore_cog: bool = False,
    gdalformat: Optional[str] = None,
    *,
    calc_pyramids: bool = True,
) -> None:
    """
    Populate continuous (:func:`rsgispy.imageutils.stats.pop_img_stats`) or
    thematic (:func:`rsgispy.imageutils.stats.pop_thmt_img_stats`) image
    statistics and pyramids on a closed image if ``calc_stats`` is True.

    An existing Cloud Optimised GeoTIFF is never edited in place: the
    statistics functions rebuild it through a temporary GeoTIFF, which costs
    an extra copy and COG conversion. For outputs written with
    :func:`rsgispy.imageprocess.open_output_dataset`, prefer
    :func:`pop_stats_on_close`, which calculates the statistics on the
    temporary GeoTIFF before the COG is created.

    :param output_img: Path to the image file.
    :param calc_stats: If ``False``, do nothing.
    :param no_data_val: Optional no-data value to set and exclude from
                        continuous statistics.
    :param thematic: If ``True``, calculate thematic statistics and a colour
                     table (0 is no data).
    :param use_pbar: Whether to display a tqdm progress bar (Default = True).
    :param ignore_cog: Kept for compatibility; COGs are detected and rebuilt
                       automatically.
    :param gdalformat: Kept for compatibility; COGs are detected from the file.
    :param calc_pyramids: Whether to build rsgispy's overview pyramids
                          (Default = True; not done for COGs, whose overviews
                          are built by the COG driver).

    .. code:: python

        import rsgispy.imageprocess

        rsgispy.imageprocess.maybe_pop_stats("out.tif", True, no_data_val=0)
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=unused-argument
    if not calc_stats:
        return

    # Lazy import avoids circular dependencies with stats -> imagecalc.
    from rsgispy.imageutils.stats import (  # pylint: disable=import-outside-toplevel
        pop_img_stats,
        pop_thmt_img_stats,
    )

    if thematic:
        pop_thmt_img_stats(
            input_img=output_img,
            add_clr_tab=True,
            calc_pyramids=calc_pyramids,
            ignore_zero=True,
            use_pbar=use_pbar,
        )
    else:
        pop_img_stats(
            input_img=output_img,
            no_data_val=no_data_val,
            calc_pyramids=calc_pyramids,
            use_pbar=use_pbar,
        )


def pop_stats_on_close(
    dst: DatasetWriter,
    calc_stats: bool,
    no_data_val: Optional[float] = None,
    thematic: bool = False,
    use_pbar: bool = True,
) -> None:
    """
    Schedule :func:`maybe_pop_stats` to run on the output of the open writer
    ``dst`` (returned by :func:`rsgispy.imageprocess.open_output_dataset`)
    once its ``ExitStack`` has closed it. Call this inside the ``with`` block
    instead of calling :func:`maybe_pop_stats` after it.

    For GeoTIFF, KEA and other formats this is the same as calling
    :func:`maybe_pop_stats` after the ``with`` block. For COG outputs, the
    statistics are calculated on the temporary tiled GeoTIFF before it is
    converted, so they are stored in the COG without rewriting it, and
    rsgispy's own pyramids are skipped because the COG driver builds the
    overviews.

    :param dst: Open writer returned by
                :func:`rsgispy.imageprocess.open_output_dataset` or
                :func:`rsgispy.imageprocess.create_output_from_dataset`.
    :param calc_stats: If ``False``, do nothing.
    :param no_data_val: Optional no-data value to set and exclude from
                        continuous statistics.
    :param thematic: If ``True``, calculate thematic statistics and a colour
                     table (0 is no data).
    :param use_pbar: Whether to display a tqdm progress bar (Default = True).

    .. code:: python

        from contextlib import ExitStack
        import rsgispy.imageprocess

        with ExitStack() as stack:
            dst = rsgispy.imageprocess.open_output_dataset(
                stack, "out_cog.tif", "COG", "float32", grid
            )
            rsgispy.imageprocess.pop_stats_on_close(dst, True, no_data_val=0)
            ...  # write blocks
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    if not calc_stats:
        return
    stage = get_output_stage(dst)
    is_cog = stage is not None and stage.is_cog

    def _pop_stats(path: str) -> None:
        maybe_pop_stats(
            path,
            True,
            no_data_val=no_data_val,
            thematic=thematic,
            use_pbar=use_pbar,
            calc_pyramids=not is_cog,
        )

    on_output_close(dst, _pop_stats)


def update_chan_moments(
    count_a: int,
    mean_a: Optional[np.ndarray],
    m2_a: Optional[np.ndarray],
    batch: np.ndarray,
    track_covariance: bool = False,
) -> Tuple[int, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Update running sample count, mean, and second central moment ``M2`` using
    the parallel batch algorithm of Chan et al. (1979).

    :param count_a: Running sample count before ``batch``.
    :param mean_a: Running mean vector ``(D,)`` (or ``None`` if ``count_a == 0``).
    :param m2_a: Running second central moment ``(D,)`` or scatter matrix
                 ``(D, D)`` when ``track_covariance=True``.
    :param batch: 1D ``(N,)`` or 2D ``(N, D)`` batch of finite valid observations.
    :param track_covariance: If ``True``, maintain the full ``(D, D)`` centred
                             scatter matrix ``M2``; otherwise maintain ``(D,)``
                             element-wise sum of squared deviations.
    :return: Updated ``(count_total, mean_new, m2_new)``.
    """
    arr = np.asarray(batch, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    n_b = int(arr.shape[0])
    if n_b == 0:
        return count_a, mean_a, m2_a
    mean_b = np.mean(arr, axis=0)
    centred_b = arr - mean_b
    if track_covariance:
        m2_b = centred_b.T @ centred_b
    else:
        m2_b = np.sum(centred_b * centred_b, axis=0)
    if count_a == 0 or mean_a is None or m2_a is None:
        return n_b, mean_b, m2_b
    delta = mean_b - mean_a
    n_total = count_a + n_b
    weight = float(count_a) * float(n_b) / float(n_total)
    mean_new = mean_a + delta * (float(n_b) / float(n_total))
    if track_covariance:
        m2_new = m2_a + m2_b + np.outer(delta, delta) * weight
    else:
        m2_new = m2_a + m2_b + (delta * delta) * weight
    return n_total, mean_new, m2_new


class OnlineStatsAccumulator:
    """
    Numerically stable streaming accumulator for per-band min, max, mean, and
    variance/standard deviation using Chan et al. (1979) / Welford updates.

    Supports both independent per-band 1D updates (:meth:`update_band`) and
    joint multi-band ``(N, B)`` matrix updates (:meth:`update_matrix`).
    """

    def __init__(self, n_bands: int = 1) -> None:
        self.n_bands = int(n_bands)
        self.counts = np.zeros(self.n_bands, dtype=np.int64)
        self.mins = np.full(self.n_bands, np.inf, dtype=np.float64)
        self.maxs = np.full(self.n_bands, -np.inf, dtype=np.float64)
        self.means = np.zeros(self.n_bands, dtype=np.float64)
        self.m2s = np.zeros(self.n_bands, dtype=np.float64)

    def update_band(self, band_idx: int, vals: np.ndarray) -> None:
        """Update running statistics for 0-based ``band_idx`` with 1D finite ``vals``."""
        arr = np.asarray(vals, dtype=np.float64).ravel()
        k = int(arr.size)
        if k == 0:
            return
        b_min = float(np.min(arr))
        b_max = float(np.max(arr))
        if b_min < self.mins[band_idx]:
            self.mins[band_idx] = b_min
        if b_max > self.maxs[band_idx]:
            self.maxs[band_idx] = b_max

        b_mean = float(np.mean(arr))
        b_m2 = float(np.sum((arr - b_mean) ** 2))
        n_prev = int(self.counts[band_idx])
        if n_prev == 0:
            self.means[band_idx] = b_mean
            self.m2s[band_idx] = b_m2
            self.counts[band_idx] = k
        else:
            n_new = n_prev + k
            delta = b_mean - self.means[band_idx]
            self.means[band_idx] += delta * (float(k) / float(n_new))
            self.m2s[band_idx] += b_m2 + (delta * delta) * (
                float(n_prev) * float(k) / float(n_new)
            )
            self.counts[band_idx] = n_new

    def update_matrix(self, flat_valid: np.ndarray) -> None:
        """Update all bands simultaneously from a 2D ``(n_samples, n_bands)`` matrix."""
        mat = np.asarray(flat_valid, dtype=np.float64)
        k = int(mat.shape[0])
        if k == 0:
            return
        self.mins = np.minimum(self.mins, np.min(mat, axis=0))
        self.maxs = np.maximum(self.maxs, np.max(mat, axis=0))
        b_mean = np.mean(mat, axis=0)
        b_m2 = np.sum(np.square(mat - b_mean), axis=0)
        n_prev = int(self.counts[0])
        if n_prev == 0:
            self.means = b_mean
            self.m2s = b_m2
            self.counts[:] = k
        else:
            n_new = n_prev + k
            delta = b_mean - self.means
            self.means += delta * (float(k) / float(n_new))
            self.m2s += b_m2 + np.square(delta) * (
                float(n_prev) * float(k) / float(n_new)
            )
            self.counts[:] = n_new

    def stds(self, ddof: int = 0) -> np.ndarray:
        """Return per-band standard deviations (``0.0`` where ``count <= ddof``)."""
        denom = np.maximum(self.counts - int(ddof), 1)
        return np.where(
            self.counts > int(ddof),
            np.sqrt(np.maximum(self.m2s / denom, 0.0)),
            0.0,
        )


def compute_dataset_band_stats(
    ds: DatasetReader,
    bidx: int,
    nodata: Optional[float] = None,
) -> Dict[str, Optional[float]]:
    """
    Stream blocks of 1-based band ``bidx`` from ``ds`` using
    :class:`OnlineStatsAccumulator` and return a dictionary of summary metrics.
    """
    acc = OnlineStatsAccumulator(n_bands=1)
    for _, window in ds.block_windows(bidx):
        block = ds.read(bidx, window=window, masked=False).astype(
            np.float64, copy=False
        )
        valid = build_valid_mask(block, nodata)
        if np.any(valid):
            acc.update_band(0, block[valid])

    count = int(acc.counts[0])
    if count == 0:
        return {
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "median": None,
            "mode": None,
            "valid_percent": 0.0,
        }

    total_px = ds.width * ds.height
    valid_pct = 100.0 * float(count) / float(total_px) if total_px > 0 else 0.0
    return {
        "min": float(acc.mins[0]),
        "max": float(acc.maxs[0]),
        "mean": float(acc.means[0]),
        "std": float(acc.stds()[0]),
        "median": None,
        "mode": None,
        "valid_percent": valid_pct,
    }
