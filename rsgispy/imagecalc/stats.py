# pylint: disable=too-many-lines
"""
The imagecalc.stats module provides basic raster band summary statistics,
pixel value counting, and unique value extraction using ``rasterio``.
"""

from contextlib import ExitStack
import math
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import rasterio
from rasterio.windows import Window

import rsgispy
from rsgispy.imageprocess import (
    OnlineStatsAccumulator,
    build_valid_cube_mask,
    build_valid_mask,
    cast_result_block,
    compute_spatial_intersection,
    ensure_file_exists,
    iter_dataset_windows,
    iter_single_image_band_blocks,
    open_multi_image_datasets,
    open_output_dataset,
    resolve_out_nodata,
    validate_band_index,
)

try:
    import scipy.stats  # pylint: disable=import-error

    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

__all__ = [
    "StatsSummary",
    "calc_basic_img_band_stats",
    "count_pxls_of_val",
    "get_unique_values",
    "image_pixel_column_summary",
    "calc_multi_img_band_stats",
    "get_img_sum_stats_in_pxl",
    "calc_mask_img_pxl_val_prob",
    "get_img_histogram",
    "get_img_2d_histogram",
]


def calc_basic_img_band_stats(
    input_img: str,
    no_data_val: Optional[float] = None,
    use_no_data: bool = True,
    use_pbar: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the minimum, maximum, mean, and standard deviation values
    for each band in the input image using block-wise iteration.

    :param input_img: Path to the input image file.
    :param no_data_val: Optional no-data value. If ``None`` and ``use_no_data`` is True,
                        reads the no-data value from the image header.
    :param use_no_data: Whether to exclude the no-data value from statistics (Default = True).
    :param use_pbar: Whether to display a tqdm progress bar (Default = True).
    :return: Tuple of ``(min_vals, max_vals, mean_vals, stddev_vals)`` NumPy arrays.
             ``stddev_vals`` is the population standard deviation
             (``ddof=0``, divided by ``n``), as in RSGISLib's image band
             statistics and GDAL.

    .. code:: python

        import rsgispy.imagecalc.stats

        mins, maxs, means, stds = rsgispy.imagecalc.stats.calc_basic_img_band_stats(
            input_img="img.tif",
            no_data_val=0.0,
        )
    """
    # pylint: disable=too-many-locals,too-many-branches
    ensure_file_exists(input_img)

    try:
        with rasterio.open(input_img, "r") as src:
            hdr_nodata = src.nodata
            n_bands = src.count
            rsgis_dt = rsgispy.get_rsgis_datatype(src.dtypes[0])
    except Exception as err:
        raise rsgispy.RSGISPyException(
            f"Could not open input image '{input_img}': {err}"
        ) from err

    if use_no_data and no_data_val is None:
        no_data_val = hdr_nodata
        if no_data_val is None:
            raise rsgispy.RSGISPyException(
                "A no data value needs to be specified either passed to the "
                "function or read from the input image header."
            )

    np_dt = rsgispy.get_numpy_datatype(rsgis_dt)
    accum = OnlineStatsAccumulator(n_bands)
    eff_nd = no_data_val if use_no_data else None

    for _, block_arrays in iter_single_image_band_blocks(
        input_img=input_img, read_dtype=np.float64, use_pbar=use_pbar
    ):
        for idx, band_arr in enumerate(block_arrays):
            valid = build_valid_mask(band_arr, no_data_val=eff_nd)
            if np.any(valid):
                accum.update_band(idx, band_arr[valid])

    min_vals = np.where(accum.counts > 0, accum.mins, 0.0).astype(np_dt)
    max_vals = np.where(accum.counts > 0, accum.maxs, 0.0).astype(np_dt)

    return min_vals, max_vals, accum.means.copy(), accum.stds()


def count_pxls_of_val(
    input_img: str,
    vals: Sequence[int],
    img_band: Optional[int] = None,
    use_pbar: bool = True,
) -> np.ndarray:
    """
    Count the number of pixels equal to each value in ``vals``.

    :param input_img: Path to the input image file.
    :param vals: Sequence of pixel values to count.
    :param img_band: 1-based band index, or ``None`` to count across all bands.
    :param use_pbar: If True, display a tqdm progress bar (Default = True).
    :return: 1D ``int64`` NumPy array of counts matching the order of ``vals``.

    .. code:: python

        import rsgispy.imagecalc.stats

        counts = rsgispy.imagecalc.stats.count_pxls_of_val(
            input_img="classes.tif",
            vals=[1, 2, 3],
            img_band=1,
        )
    """
    if vals is None:
        raise rsgispy.RSGISPyException("Input vals list must not be None.")
    if len(vals) == 0:
        raise rsgispy.RSGISPyException(
            "At least 1 value should be provided within the vals input variable."
        )
    ensure_file_exists(input_img)

    with rasterio.open(input_img, "r") as src:
        n_bands = src.count
        if img_band is not None and (img_band < 1 or img_band > n_bands):
            raise rsgispy.RSGISPyException(
                "The specified input image band is not within the input image."
            )

        out_vals = np.zeros(len(vals), dtype=np.int64)
        for win in iter_dataset_windows(
            src.height, src.width, use_pbar=use_pbar
        ):
            block_arr = (
                src.read(indexes=img_band, window=win)
                if img_band is not None
                else src.read(window=win)
            )
            for idx, val in enumerate(vals):
                out_vals[idx] += int(np.count_nonzero(block_arr == val))

    return out_vals


def get_unique_values(
    input_img: str,
    img_band: int = 1,
) -> np.ndarray:
    """
    Find the unique pixel values within a single image band.

    Note: the entire image band is read into memory.

    :param input_img: Path to the input image file.
    :param img_band: 1-based band index to process (Default = 1).
    :return: 1D sorted NumPy array of unique values.

    .. code:: python

        import rsgispy.imagecalc.stats

        unq_vals = rsgispy.imagecalc.stats.get_unique_values(
            input_img="classes.tif",
            img_band=1,
        )
    """
    ensure_file_exists(input_img)

    try:
        with rasterio.open(input_img, "r") as src:
            if img_band < 1 or img_band > src.count:
                raise rsgispy.RSGISPyException(
                    f"Could not open image band ({img_band})."
                )
            vals_arr = src.read(img_band)
    except rsgispy.RSGISPyException:
        raise
    except Exception as err:
        raise rsgispy.RSGISPyException(
            f"Could not open input image '{input_img}': {err}"
        ) from err

    return np.unique(vals_arr)


class StatsSummary:
    """
    Configuration object specifying summary statistics to calculate per pixel
    column in :func:`image_pixel_column_summary`.
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-instance-attributes,too-few-public-methods,redefined-builtin
    def __init__(
        self,
        min: float = 0.0,
        max: float = 0.0,
        sum: float = 0.0,
        median: float = 0.0,
        stdev: float = 0.0,
        mean: float = 0.0,
        mode: float = 0.0,
        calc_min: bool = False,
        calc_max: bool = False,
        calc_sum: bool = False,
        calc_mean: bool = False,
        calc_stdev: bool = False,
        calc_median: bool = False,
        calc_mode: bool = False,
    ) -> None:
        self.min = min
        self.max = max
        self.sum = sum
        self.mean = mean
        self.stdev = stdev
        self.median = median
        self.mode = mode
        self.calc_min = calc_min
        self.calc_max = calc_max
        self.calc_sum = calc_sum
        self.calc_mean = calc_mean
        self.calc_stdev = calc_stdev
        self.calc_median = calc_median
        self.calc_mode = calc_mode


def _calc_mode_columns(sub_2d: np.ndarray) -> np.ndarray:
    """Compute column-wise mode ignoring NaNs for a 2D ``(n_samples, n_cols)`` array."""
    if HAVE_SCIPY:
        mode_res = scipy.stats.mode(sub_2d, axis=0, nan_policy="omit").mode
        return np.asarray(mode_res, dtype=np.float64).reshape(-1)[: sub_2d.shape[1]]

    n_cols = sub_2d.shape[1]
    out = np.zeros(n_cols, dtype=np.float64)
    for col_idx in range(n_cols):
        col = sub_2d[:, col_idx]
        col = col[np.isfinite(col)]
        if col.size > 0:
            vals, counts = np.unique(col, return_counts=True)
            out[col_idx] = float(vals[int(np.argmax(counts))])
    return out


def _calc_stack_stat(stack_arr: np.ndarray, stat_type: int) -> np.ndarray:
    """
    Compute a summary statistic ``rsgispy.SUMTYPE_*`` along ``axis=0`` of a
    3D float64 array ``(n_samples, height, width)`` where invalid values are
    ``np.nan`` or non-finite. The standard deviation is the sample one
    (``ddof=1``), matching RSGISLib's pixel summaries (GSL ``gsl_stats_sd``);
    :func:`calc_basic_img_band_stats` uses the population one (``ddof=0``),
    matching RSGISLib's image band statistics.
    """
    # pylint: disable=too-many-return-statements
    valid_mask = build_valid_mask(stack_arr)
    counts = np.sum(valid_mask, axis=0)
    out_shape = stack_arr.shape[1:]

    if stat_type == rsgispy.SUMTYPE_COUNT:
        return counts.astype(np.float64)
    if stat_type == rsgispy.SUMTYPE_SUM:
        return np.where(valid_mask, stack_arr, 0.0).sum(axis=0)
    if stat_type == rsgispy.SUMTYPE_MIN:
        mins = np.min(np.where(valid_mask, stack_arr, np.inf), axis=0)
        return np.where(counts > 0, mins, 0.0)
    if stat_type == rsgispy.SUMTYPE_MAX:
        maxs = np.max(np.where(valid_mask, stack_arr, -np.inf), axis=0)
        return np.where(counts > 0, maxs, 0.0)
    if stat_type == rsgispy.SUMTYPE_RANGE:
        mins = np.min(np.where(valid_mask, stack_arr, np.inf), axis=0)
        maxs = np.max(np.where(valid_mask, stack_arr, -np.inf), axis=0)
        return np.where(counts > 0, maxs - mins, 0.0)
    if stat_type == rsgispy.SUMTYPE_MEAN:
        valid_sum = np.where(valid_mask, stack_arr, 0.0).sum(axis=0)
        return np.where(counts > 0, valid_sum / np.maximum(counts, 1), 0.0)
    if stat_type == rsgispy.SUMTYPE_STDDEV:
        valid_sum = np.where(valid_mask, stack_arr, 0.0).sum(axis=0)
        means = np.where(counts > 0, valid_sum / np.maximum(counts, 1), 0.0)
        sq_diff = np.where(valid_mask, (stack_arr - means[None, ...]) ** 2, 0.0).sum(
            axis=0
        )
        return np.where(
            counts > 1, np.sqrt(sq_diff / np.maximum(counts - 1, 1)), 0.0
        )
    if stat_type == rsgispy.SUMTYPE_MEDIAN:
        out = np.zeros(out_shape, dtype=np.float64)
        has_data = counts > 0
        if np.any(has_data):
            flat = stack_arr[:, has_data]
            out[has_data] = np.nanmedian(flat, axis=0)
        return out
    if stat_type == rsgispy.SUMTYPE_MODE:
        out = np.zeros(out_shape, dtype=np.float64)
        has_data = counts > 0
        if np.any(has_data):
            flat = stack_arr[:, has_data]
            out[has_data] = _calc_mode_columns(flat)
        return out

    raise rsgispy.RSGISPyException(
        f"Summary statistic type '{stat_type}' is not recognised."
    )


def _resolve_stats_out_nd(
    cand: Optional[float], out_np_dtype: np.dtype
) -> Optional[float]:
    """
    Output no-data value for the per-pixel summary statistics: ``cand``, or
    for an integer type that cannot hold it, the type's maximum (unsigned)
    or minimum (signed); ``None`` when ``cand`` is ``None``.
    """
    if cand is None:
        return None
    return resolve_out_nodata(
        out_np_dtype, cand, clamp_int=True, unsigned_clamp="max"
    )


def _stack_stat_block(
    stack_arr: np.ndarray,
    stat_type: int,
    out_np_dtype: np.dtype,
    out_nd: Optional[float],
) -> np.ndarray:
    """
    :func:`_calc_stack_stat` cast to ``out_np_dtype``, with pixels that have
    no valid input written as ``out_nd`` (counts are always written as is).
    """
    stat_arr = _calc_stack_stat(stack_arr, stat_type)
    win_h, win_w = int(stack_arr.shape[1]), int(stack_arr.shape[2])
    if out_nd is None or stat_type == rsgispy.SUMTYPE_COUNT:
        return cast_result_block(stat_arr, win_h, win_w, out_np_dtype)
    has_data = build_valid_mask(stack_arr).any(axis=0)
    return cast_result_block(
        np.where(has_data, stat_arr, out_nd), win_h, win_w, out_np_dtype
    )


def _mask_nodata_nan(
    arr: np.ndarray, no_data_val: Optional[float], use_no_data: bool
) -> np.ndarray:
    """Convert non-finite and (when enabled) ``no_data_val`` elements to ``np.nan``."""
    out = arr.astype(np.float64, copy=True)
    invalid = ~build_valid_mask(out, no_data_val if use_no_data else None)
    if np.any(invalid):
        out[invalid] = np.nan
    return out


def image_pixel_column_summary(
    input_img: str,
    output_img: str,
    sum_stats: StatsSummary,
    use_no_data: bool = True,
    *,
    gdalformat: str = "GTIFF",
    datatype: int = rsgispy.TYPE_32FLOAT,
    no_data_val: float = 0.0,
    use_pbar: bool = True,
    creation_opts: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Calculate summary statistics across all bands (pixel column) for each
    pixel in ``input_img``.

    Output bands are written in the canonical order: ``Min``, ``Max``, ``Mean``,
    ``Median``, ``Mode``, ``Sum``, ``StdDev`` for whichever corresponding
    ``calc_*`` flags are ``True`` on ``sum_stats``. ``StdDev`` is the sample
    standard deviation (``ddof=1``, divided by ``n - 1``; ``0`` for a single
    valid value), as in RSGISLib's pixel summaries, unlike
    :func:`calc_basic_img_band_stats`, which gives the population one.

    :param input_img: Path to the input multi-band image file.
    :param output_img: Path to the output summary image file.
    :param sum_stats: :class:`StatsSummary` specifying statistics to compute.
    :param use_no_data: If True, exclude ``no_data_val`` from statistics.
    :param gdalformat: GDAL/rasterio driver format for the output file
                       (Default = ``'GTIFF'``).
    :param datatype: Output data type constant from ``rsgispy.TYPE_*``
                     (Default = ``rsgispy.TYPE_32FLOAT``).
    :param no_data_val: Value treated as no-data when ``use_no_data`` is True.
                        It is also the output no-data value, written where no
                        band of a pixel is valid (for an integer ``datatype``
                        that cannot hold it, the type's max/min is used).
    :param use_pbar: If True, display a tqdm progress bar (Default = True).
    :param creation_opts: Optional mapping of GDAL creation options.

    .. code:: python

        import rsgispy
        import rsgispy.imagecalc.stats

        sum_stats = rsgispy.imagecalc.stats.StatsSummary(
            calc_min=True, calc_max=True, calc_mean=True
        )
        rsgispy.imagecalc.stats.image_pixel_column_summary(
            input_img="stack.tif",
            output_img="stack_summary.tif",
            sum_stats=sum_stats,
            gdalformat="GTIFF",
            datatype=rsgispy.TYPE_32FLOAT,
        )
    """
    # pylint: disable=too-many-arguments,too-many-locals
    if sum_stats is None:
        raise rsgispy.RSGISPyException("sum_stats must not be None.")

    metric_specs = [
        ("calc_min", "Min", rsgispy.SUMTYPE_MIN),
        ("calc_max", "Max", rsgispy.SUMTYPE_MAX),
        ("calc_mean", "Mean", rsgispy.SUMTYPE_MEAN),
        ("calc_median", "Median", rsgispy.SUMTYPE_MEDIAN),
        ("calc_mode", "Mode", rsgispy.SUMTYPE_MODE),
        ("calc_sum", "Sum", rsgispy.SUMTYPE_SUM),
        ("calc_stdev", "StdDev", rsgispy.SUMTYPE_STDDEV),
    ]
    selected: List[Tuple[str, int]] = [
        (label, stat_id)
        for attr, label, stat_id in metric_specs
        if getattr(sum_stats, attr, False)
    ]
    if not selected:
        raise rsgispy.RSGISPyException(
            "At least one statistic must be enabled in sum_stats."
        )

    ensure_file_exists(input_img)
    out_rio_dtype = rsgispy.get_rasterio_datatype(datatype)
    out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(datatype))

    with ExitStack() as stack:
        try:
            src = stack.enter_context(rasterio.open(input_img, "r"))
        except Exception as err:
            raise rsgispy.RSGISPyException(
                f"Could not open input image '{input_img}': {err}"
            ) from err

        grid = compute_spatial_intersection({input_img: src})
        dst = open_output_dataset(
            stack=stack,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype=out_rio_dtype,
            grid=grid,
            count=len(selected),
            creation_opts=creation_opts,
        )
        out_nd = _resolve_stats_out_nd(
            no_data_val if use_no_data else None, out_np_dtype
        )
        if out_nd is not None:
            dst.nodata = out_nd
        for out_band_idx, (label, _) in enumerate(selected, start=1):
            dst.set_band_description(out_band_idx, label)

        for out_win in iter_dataset_windows(
            grid.height, grid.width, use_pbar=use_pbar
        ):
            in_block = src.read(window=out_win, out_dtype=np.float64)
            masked_block = _mask_nodata_nan(in_block, no_data_val, use_no_data)
            out_bands = [
                _stack_stat_block(masked_block, stat_id, out_np_dtype, out_nd)
                for _, stat_id in selected
            ]
            dst.write(np.stack(out_bands, axis=0), window=out_win)


def calc_multi_img_band_stats(
    input_imgs: Sequence[str],
    output_img: str,
    summary_stat: int,
    use_no_data: bool = False,
    *,
    gdalformat: str = "GTIFF",
    datatype: int = rsgispy.TYPE_32FLOAT,
    no_data_val: Optional[float] = None,
    use_pbar: bool = True,
    creation_opts: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Calculate a summary statistic (``rsgispy.SUMTYPE_*``) across multiple
    input images on a per-band basis. ``SUMTYPE_STDDEV`` is the sample
    standard deviation (``ddof=1``), as in RSGISLib's pixel summaries.

    All input images must have the same number of bands. The output image has
    the same band count over the spatial intersection of ``input_imgs``.

    :param input_imgs: Sequence of input image paths with identical band counts.
    :param output_img: Path to the output image file.
    :param summary_stat: Summary statistic constant ``rsgispy.SUMTYPE_*``.
    :param use_no_data: If True, exclude ``no_data_val`` pixels (Default = False).
    :param gdalformat: GDAL/rasterio driver format for the output file
                       (Default = ``'GTIFF'``).
    :param datatype: Output data type constant from ``rsgispy.TYPE_*``
                     (Default = ``rsgispy.TYPE_32FLOAT``).
    :param no_data_val: Optional no-data value across all input images. If
                        ``None`` and ``use_no_data`` is True, read from the
                        first image header. It is also the output no-data
                        value, written where no input is valid (for an
                        integer ``datatype`` that cannot hold it, the type's
                        max/min is used); counts are written as they are.
    :param use_pbar: If True, display a tqdm progress bar (Default = True).
    :param creation_opts: Optional mapping of GDAL creation options.

    .. code:: python

        import rsgispy
        import rsgispy.imagecalc.stats

        rsgispy.imagecalc.stats.calc_multi_img_band_stats(
            input_imgs=["img1.tif", "img2.tif", "img3.tif"],
            output_img="mean_composite.tif",
            summary_stat=rsgispy.SUMTYPE_MEAN,
            gdalformat="GTIFF",
            datatype=rsgispy.TYPE_32FLOAT,
        )
    """
    # pylint: disable=too-many-arguments,too-many-locals
    valid_stats = (
        rsgispy.SUMTYPE_MIN,
        rsgispy.SUMTYPE_MAX,
        rsgispy.SUMTYPE_MEAN,
        rsgispy.SUMTYPE_MEDIAN,
        rsgispy.SUMTYPE_MODE,
        rsgispy.SUMTYPE_STDDEV,
        rsgispy.SUMTYPE_COUNT,
        rsgispy.SUMTYPE_RANGE,
        rsgispy.SUMTYPE_SUM,
    )
    if summary_stat not in valid_stats:
        raise rsgispy.RSGISPyException(
            f"Summary statistic type '{summary_stat}' is not recognised."
        )

    out_rio_dtype = rsgispy.get_rasterio_datatype(datatype)
    out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(datatype))

    with ExitStack() as stack:
        datasets, img_paths = open_multi_image_datasets(stack, input_imgs)
        n_bands = datasets[img_paths[0]].count
        for path in img_paths[1:]:
            if datasets[path].count != n_bands:
                raise rsgispy.RSGISPyException(
                    "All input images must have the same number of bands."
                )

        eff_nd = no_data_val
        if use_no_data and eff_nd is None:
            eff_nd = datasets[img_paths[0]].nodata
            if eff_nd is None:
                raise rsgispy.RSGISPyException(
                    "A no data value must be specified or present in image header."
                )

        grid = compute_spatial_intersection(datasets)
        dst = open_output_dataset(
            stack=stack,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype=out_rio_dtype,
            grid=grid,
            count=n_bands,
            creation_opts=creation_opts,
        )
        out_nd = _resolve_stats_out_nd(eff_nd, out_np_dtype)
        if out_nd is not None:
            dst.nodata = out_nd

        for out_win in iter_dataset_windows(
            grid.height, grid.width, use_pbar=use_pbar
        ):
            win_h = int(out_win.height)
            win_w = int(out_win.width)
            img_blocks = [
                datasets[path].read(
                    window=Window(
                        grid.offsets[path][0] + int(out_win.col_off),
                        grid.offsets[path][1] + int(out_win.row_off),
                        win_w,
                        win_h,
                    ),
                    out_dtype=np.float64,
                )
                for path in img_paths
            ]
            out_bands = []
            for b_idx in range(n_bands):
                band_stack = np.stack([blk[b_idx] for blk in img_blocks], axis=0)
                masked_stack = _mask_nodata_nan(band_stack, eff_nd, use_no_data)
                out_bands.append(
                    _stack_stat_block(
                        masked_stack, summary_stat, out_np_dtype, out_nd
                    )
                )
            dst.write(np.stack(out_bands, axis=0), window=out_win)


def get_img_sum_stats_in_pxl(
    in_ref_img: str,
    in_stats_img: str,
    output_img: str,
    sum_stats: Sequence[int] = (rsgispy.SUMTYPE_MEAN,),
    stats_img_band: int = 1,
    use_no_data: bool = True,
    io_grid_x: int = 16,
    io_grid_y: int = 16,
    *,
    gdalformat: str = "GTIFF",
    datatype: int = rsgispy.TYPE_32FLOAT,
    use_pbar: bool = True,
    creation_opts: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Summarise high-resolution pixel values from ``in_stats_img`` within each
    coarse pixel of ``in_ref_img`` (zonal aggregation on aligned integer-ratio grids).

    The resolution of ``in_ref_img`` must be strictly coarser than and an integer
    multiple of ``in_stats_img`` along both axes. Each entry in ``sum_stats``
    produces one output band. ``SUMTYPE_STDDEV`` is the sample standard
    deviation (``ddof=1``), as in RSGISLib's pixel summaries.

    :param in_ref_img: Path to the coarser-resolution reference image defining
                       output grid, extent, and resolution.
    :param in_stats_img: Path to the higher-resolution image to summarise.
    :param output_img: Path to the output image file.
    :param sum_stats: Sequence of ``rsgispy.SUMTYPE_*`` statistics to compute
                      (one band per statistic).
    :param stats_img_band: 1-based band index in ``in_stats_img`` (Default = 1).
    :param use_no_data: If True, exclude the ``in_stats_img`` header no-data value
                        (Default = True). That value is then also the output
                        no-data value, written for reference pixels with no
                        valid stats pixel.
    :param io_grid_x: Block width in reference pixels (Default = 16).
    :param io_grid_y: Block height in reference pixels (Default = 16).
    :param gdalformat: GDAL/rasterio driver format for the output file
                       (Default = ``'GTIFF'``).
    :param datatype: Output data type constant from ``rsgispy.TYPE_*``
                     (Default = ``rsgispy.TYPE_32FLOAT``).
    :param use_pbar: If True, display a tqdm progress bar (Default = True).
    :param creation_opts: Optional mapping of GDAL creation options.

    .. code:: python

        import rsgispy
        import rsgispy.imagecalc.stats

        rsgispy.imagecalc.stats.get_img_sum_stats_in_pxl(
            in_ref_img="coarse_grid.tif",
            in_stats_img="fine_res.tif",
            output_img="zonal_stats.tif",
            sum_stats=[rsgispy.SUMTYPE_MEAN, rsgispy.SUMTYPE_STDDEV],
            gdalformat="GTIFF",
            datatype=rsgispy.TYPE_32FLOAT,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    if not sum_stats:
        raise rsgispy.RSGISPyException(
            "sum_stats must contain at least one SUMTYPE_* value."
        )
    if io_grid_x <= 0 or io_grid_y <= 0:
        raise rsgispy.RSGISPyException("io_grid_x and io_grid_y must be > 0.")

    ensure_file_exists(in_ref_img, label="Reference image")
    ensure_file_exists(in_stats_img, label="Stats image")
    out_rio_dtype = rsgispy.get_rasterio_datatype(datatype)
    out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(datatype))

    with ExitStack() as stack:
        try:
            ref_src = stack.enter_context(rasterio.open(in_ref_img, "r"))
            stats_src = stack.enter_context(rasterio.open(in_stats_img, "r"))
        except Exception as err:
            raise rsgispy.RSGISPyException(
                f"Could not open input imagery: {err}"
            ) from err

        validate_band_index(stats_img_band, stats_src.count, in_stats_img)
        ref_x_res = abs(float(ref_src.transform.a))
        ref_y_res = abs(float(ref_src.transform.e))
        stats_x_res = abs(float(stats_src.transform.a))
        stats_y_res = abs(float(stats_src.transform.e))

        if ref_x_res <= stats_x_res:
            raise rsgispy.RSGISPyException(
                "Reference image is less than or equal resolution to stats image in X axis."
            )
        if ref_y_res <= stats_y_res:
            raise rsgispy.RSGISPyException(
                "Reference image is less than or equal resolution to stats image in Y axis."
            )

        ratio_x = ref_x_res / stats_x_res
        ratio_y = ref_y_res / stats_y_res
        factor_x = int(round(ratio_x))
        factor_y = int(round(ratio_y))
        if abs(ratio_x - factor_x) > 1e-4 or abs(ratio_y - factor_y) > 1e-4:
            raise rsgispy.RSGISPyException(
                "Stats image resolution must divide reference image resolution "
                "into an integer number of pixels."
            )

        stats_nd = stats_src.nodatavals[stats_img_band - 1]
        grid = compute_spatial_intersection({in_ref_img: ref_src})
        dst = open_output_dataset(
            stack=stack,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype=out_rio_dtype,
            grid=grid,
            count=len(sum_stats),
            creation_opts=creation_opts,
        )
        out_nd = _resolve_stats_out_nd(
            stats_nd if use_no_data else None, out_np_dtype
        )
        if out_nd is not None:
            dst.nodata = out_nd

        inv_stats_transform = ~stats_src.transform

        for out_win in iter_dataset_windows(
            grid.height,
            grid.width,
            block_size=io_grid_y,
            block_width=io_grid_x,
            use_pbar=use_pbar,
        ):
            col_off = int(out_win.col_off)
            row_off = int(out_win.row_off)
            win_w = int(out_win.width)
            win_h = int(out_win.height)

            tl_x, tl_y = ref_src.transform @ (col_off, row_off)
            s_col_f, s_row_f = inv_stats_transform @ (tl_x, tl_y)
            s_win = Window(
                int(round(s_col_f)),
                int(round(s_row_f)),
                win_w * factor_x,
                win_h * factor_y,
            )
            sub = stats_src.read(
                stats_img_band,
                window=s_win,
                boundless=True,
                fill_value=np.nan,
                out_dtype=np.float64,
            )
            sub = _mask_nodata_nan(sub, stats_nd, use_no_data)
            sub_stack = (
                sub.reshape(win_h, factor_y, win_w, factor_x)
                .transpose(1, 3, 0, 2)
                .reshape(factor_y * factor_x, win_h, win_w)
            )
            out_bands = [
                _stack_stat_block(sub_stack, stat_id, out_np_dtype, out_nd)
                for stat_id in sum_stats
            ]
            dst.write(np.stack(out_bands, axis=0), window=out_win)


def _estimate_bin_width(vals: np.ndarray, b_min: float, b_max: float) -> float:
    """Estimate histogram bin width using Freedman-Diaconis / Scott fallback."""
    if b_max <= b_min or vals.size <= 1:
        return 1.0
    n_cube = float(vals.size) ** (1.0 / 3.0)
    iqr = float(np.percentile(vals, 75) - np.percentile(vals, 25))
    if iqr > 0.0:
        width = (2.0 * iqr) / n_cube
        if width > 0.0:
            return width
    std = float(np.std(vals))
    if std > 0.0:
        width = (3.5 * std) / n_cube
        if width > 0.0:
            return width
    return (b_max - b_min) / 10.0


def calc_mask_img_pxl_val_prob(
    input_img: str,
    in_msk_img: str,
    output_img: str,
    img_bands: Sequence[int],
    msk_val: int,
    bin_widths: Optional[Sequence[float]] = None,
    use_no_data: bool = True,
    rescale_probs: bool = True,
    *,
    gdalformat: str = "GTIFF",
    use_pbar: bool = True,
    creation_opts: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Calculate the empirical joint n-dimensional probability of pixel values
    across ``img_bands`` based on the distribution inside ``in_msk_img == msk_val``.

    The output image is a single-band ``float32`` image.

    :param input_img: Path to the input image file.
    :param in_msk_img: Path to the mask image file (band 1 compared to ``msk_val``).
    :param output_img: Path to the output single-band probability image.
    :param img_bands: Sequence of 1-based band indices in ``input_img``.
    :param msk_val: Pixel value in ``in_msk_img`` defining the sample mask.
    :param bin_widths: Optional sequence of histogram bin widths (one per band).
                       Values ``<= 0`` or ``None`` trigger automatic estimation.
    :param use_no_data: If True, exclude band header no-data values (Default = True).
    :param rescale_probs: If True, rescale probabilities so the maximum is ``1.0``
                          (Default = True).
    :param gdalformat: GDAL/rasterio driver format for the output file
                       (Default = ``'GTIFF'``).
    :param use_pbar: If True, display a tqdm progress bar (Default = True).
    :param creation_opts: Optional mapping of GDAL creation options.

    .. code:: python

        import rsgispy.imagecalc.stats

        rsgispy.imagecalc.stats.calc_mask_img_pxl_val_prob(
            input_img="multispec.tif",
            in_msk_img="training_mask.tif",
            output_img="pxl_prob.tif",
            img_bands=[1, 2, 3],
            msk_val=1,
            gdalformat="GTIFF",
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-branches
    if not img_bands:
        raise rsgispy.RSGISPyException(
            "img_bands must specify at least one band."
        )
    if bin_widths is not None and len(bin_widths) != len(img_bands):
        raise rsgispy.RSGISPyException(
            "The number of bands specified and the histogram bin widths must be the same."
        )

    with ExitStack() as stack:
        datasets, _ = open_multi_image_datasets(stack, [input_img, in_msk_img])
        img_src = datasets[input_img]
        msk_src = datasets[in_msk_img]

        for band in img_bands:
            validate_band_index(band, img_src.count, input_img)

        band_nodatas = [img_src.nodatavals[b - 1] for b in img_bands]
        grid = compute_spatial_intersection(datasets)
        base_ci, base_ri = grid.offsets[input_img]
        base_cm, base_rm = grid.offsets[in_msk_img]

        # Pass 1: gather valid masked pixel vectors.
        samples: List[np.ndarray] = []
        for out_win in iter_dataset_windows(
            grid.height, grid.width, use_pbar=use_pbar
        ):
            win_h = int(out_win.height)
            win_w = int(out_win.width)
            msk_win = Window(
                base_cm + int(out_win.col_off),
                base_rm + int(out_win.row_off),
                win_w,
                win_h,
            )
            msk_sel = msk_src.read(1, window=msk_win).ravel() == msk_val
            if not np.any(msk_sel):
                continue

            img_win = Window(
                base_ci + int(out_win.col_off),
                base_ri + int(out_win.row_off),
                win_w,
                win_h,
            )
            blk = img_src.read(list(img_bands), window=img_win, out_dtype=np.float64)
            cols = np.column_stack([blk[i].ravel()[msk_sel] for i in range(len(img_bands))])

            valid = build_valid_cube_mask(cols, axis=1)
            if use_no_data:
                for dim_idx, nd_val in enumerate(band_nodatas):
                    if nd_val is not None:
                        valid &= build_valid_mask(
                            cols[:, dim_idx], no_data_val=float(nd_val)
                        )
            if np.any(valid):
                samples.append(cols[valid])

        if not samples:
            raise rsgispy.RSGISPyException(
                "No valid masked pixels found to build probability histogram."
            )

        all_samples = np.concatenate(samples, axis=0)
        n_samples, n_dims = all_samples.shape

        # Compute bin geometry per dimension.
        b_mins = np.min(all_samples, axis=0)
        b_maxs = np.max(all_samples, axis=0)
        widths = np.empty(n_dims, dtype=np.float64)
        n_bins = np.empty(n_dims, dtype=np.int64)
        for dim_idx in range(n_dims):
            req_w = (
                float(bin_widths[dim_idx])
                if bin_widths is not None
                else -1.0
            )
            if req_w <= 0.0:
                widths[dim_idx] = _estimate_bin_width(
                    all_samples[:, dim_idx],
                    float(b_mins[dim_idx]),
                    float(b_maxs[dim_idx]),
                )
            else:
                widths[dim_idx] = req_w

            span = float(b_maxs[dim_idx] - b_mins[dim_idx])
            if span <= 0.0:
                n_bins[dim_idx] = 1
            else:
                n_bins[dim_idx] = max(1, int(math.ceil(span / widths[dim_idx])))

        def _compute_bin_indices(
            arr_2d: np.ndarray,
        ) -> Tuple[np.ndarray, np.ndarray]:
            in_bounds = build_valid_cube_mask(arr_2d, axis=1)
            idx_mat = np.zeros(arr_2d.shape, dtype=np.int64)
            for dim_idx in range(n_dims):
                col = arr_2d[:, dim_idx]
                in_bounds &= (col >= b_mins[dim_idx]) & (col <= b_maxs[dim_idx])
                if n_bins[dim_idx] > 1:
                    raw_idx = np.floor(
                        (col - b_mins[dim_idx]) / widths[dim_idx]
                    ).astype(np.int64)
                    raw_idx = np.clip(raw_idx, 0, n_bins[dim_idx] - 1)
                    idx_mat[:, dim_idx] = raw_idx
            return idx_mat, in_bounds

        sample_bins, _ = _compute_bin_indices(all_samples)
        unq_bins, counts = np.unique(sample_bins, axis=0, return_counts=True)
        probs = counts.astype(np.float64) / float(n_samples)
        if rescale_probs and probs.size > 0:
            max_p = float(np.max(probs))
            if max_p > 0.0:
                probs = probs / max_p

        # Build fast sorted key lookup if flattened index fits in int64.
        use_flat_lookup = math.prod(int(nb) for nb in n_bins) < (2**62)
        if use_flat_lookup:
            dims_tuple = tuple(int(nb) for nb in n_bins)
            sorted_keys = np.ravel_multi_index(unq_bins.T, dims_tuple)
            order = np.argsort(sorted_keys)
            sorted_keys = sorted_keys[order]
            sorted_probs = probs[order]
        else:
            prob_dict: Dict[Tuple[int, ...], float] = {
                tuple(row.tolist()): float(p_val)
                for row, p_val in zip(unq_bins, probs)
            }

        # Pass 2: evaluate probability for all pixels in input_img.
        dst = open_output_dataset(
            stack=stack,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype="float32",
            grid=grid,
            count=1,
            creation_opts=creation_opts,
        )
        for out_win in iter_dataset_windows(
            grid.height, grid.width, use_pbar=use_pbar
        ):
            win_h = int(out_win.height)
            win_w = int(out_win.width)
            img_win = Window(
                base_ci + int(out_win.col_off),
                base_ri + int(out_win.row_off),
                win_w,
                win_h,
            )
            blk = img_src.read(list(img_bands), window=img_win, out_dtype=np.float64)
            flat_2d = np.column_stack([blk[i].ravel() for i in range(n_dims)])

            valid_pxls = build_valid_cube_mask(flat_2d, axis=1)
            if use_no_data:
                for dim_idx, nd_val in enumerate(band_nodatas):
                    if nd_val is not None:
                        valid_pxls &= build_valid_mask(
                            flat_2d[:, dim_idx], no_data_val=float(nd_val)
                        )

            idx_mat, in_bounds = _compute_bin_indices(flat_2d)
            lookup_mask = valid_pxls & in_bounds
            out_flat = np.zeros(win_h * win_w, dtype=np.float32)

            if np.any(lookup_mask):
                q_bins = idx_mat[lookup_mask]
                if use_flat_lookup:
                    q_keys = np.ravel_multi_index(q_bins.T, dims_tuple)
                    pos = np.searchsorted(sorted_keys, q_keys)
                    clipped_pos = np.minimum(pos, sorted_keys.size - 1)
                    matched = (pos < sorted_keys.size) & (
                        sorted_keys[clipped_pos] == q_keys
                    )
                    q_probs = np.where(
                        matched, sorted_probs[clipped_pos], 0.0
                    ).astype(np.float32)
                else:
                    q_probs = np.array(
                        [
                            prob_dict.get(tuple(row.tolist()), 0.0)
                            for row in q_bins
                        ],
                        dtype=np.float32,
                    )
                out_flat[lookup_mask] = q_probs

            dst.write(out_flat.reshape(win_h, win_w), 1, window=out_win)


_MAX_INTEGER_HIST_BINS = 1_000_000


def _validate_histogram_options(
    n_bins: Optional[int],
    bin_size: Optional[float],
    integer_bins: bool,
    val_range: Optional[Tuple[float, float]],
) -> Optional[Tuple[float, float]]:
    """Validate bin and range arguments for :func:`get_img_histogram`."""
    if integer_bins and (n_bins is not None or bin_size is not None):
        raise rsgispy.RSGISPyException(
            "integer_bins=True cannot be combined with n_bins or bin_size."
        )
    if n_bins is not None and bin_size is not None:
        raise rsgispy.RSGISPyException(
            "Specify either n_bins or bin_size, not both."
        )
    if n_bins is not None and int(n_bins) < 1:
        raise rsgispy.RSGISPyException("n_bins must be >= 1.")
    if bin_size is not None:
        bs_val = float(bin_size)
        if not math.isfinite(bs_val) or bs_val <= 0.0:
            raise rsgispy.RSGISPyException(
                "bin_size must be a finite value greater than 0."
            )

    norm_range: Optional[Tuple[float, float]] = None
    if val_range is not None:
        if len(val_range) != 2:
            raise rsgispy.RSGISPyException(
                "val_range must be a (min_val, max_val) 2-tuple."
            )
        r_min = float(val_range[0])
        r_max = float(val_range[1])
        if not (math.isfinite(r_min) and math.isfinite(r_max)) or r_min > r_max:
            raise rsgispy.RSGISPyException(
                "val_range must contain finite bounds with min_val <= max_val."
            )
        if integer_bins and (
            r_min != math.floor(r_min) or r_max != math.floor(r_max)
        ):
            raise rsgispy.RSGISPyException(
                "val_range bounds must be integers when integer_bins=True."
            )
        norm_range = (r_min, r_max)
    return norm_range


def _extract_valid_hist_pixels(
    img_src: Any,
    img_band: int,
    img_nodata: Optional[float],
    use_no_data: bool,
    msk_src: Optional[Any],
    img_msk_val: int,
    base_ci: int,
    base_ri: int,
    base_cm: int,
    base_rm: int,
    out_win: Window,
    val_range: Optional[Tuple[float, float]],
) -> np.ndarray:
    """Read a block and return 1D valid ``float64`` pixel values."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    win_h = int(out_win.height)
    win_w = int(out_win.width)
    img_win = Window(
        base_ci + int(out_win.col_off),
        base_ri + int(out_win.row_off),
        win_w,
        win_h,
    )
    blk = img_src.read(img_band, window=img_win, out_dtype=np.float64).ravel()
    msk_arr = None
    if msk_src is not None:
        msk_win = Window(
            base_cm + int(out_win.col_off),
            base_rm + int(out_win.row_off),
            win_w,
            win_h,
        )
        msk_arr = msk_src.read(1, window=msk_win).ravel()

    valid = build_valid_mask(
        blk,
        no_data_val=float(img_nodata) if (use_no_data and img_nodata is not None) else None,
        msk_arr=msk_arr,
        valid_msk_val=img_msk_val,
    )

    vals = blk[valid]
    if val_range is not None and vals.size > 0:
        vals = vals[(vals >= val_range[0]) & (vals <= val_range[1])]
    return vals


def _build_histogram_edges(
    v_min: float,
    v_max: float,
    n_valid: int,
    std_val: float,
    n_bins: Optional[int],
    bin_size: Optional[float],
    integer_bins: bool,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Construct bin edges (and integer bin labels when ``integer_bins=True``)."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    if integer_bins:
        min_int = int(round(v_min))
        max_int = int(round(v_max))
        span_bins = (max_int - min_int) + 1
        if span_bins > _MAX_INTEGER_HIST_BINS:
            raise rsgispy.RSGISPyException(
                f"Integer histogram span ({span_bins} bins from {min_int} to "
                f"{max_int}) exceeds limit ({_MAX_INTEGER_HIST_BINS}). Use "
                "n_bins or bin_size instead."
            )
        int_vals = np.arange(min_int, max_int + 1, dtype=np.int64)
        bin_edges = np.arange(min_int, max_int + 2, dtype=np.float64) - 0.5
        return bin_edges, int_vals

    if bin_size is not None:
        bs_val = float(bin_size)
        span = v_max - v_min
        if span <= 0.0:
            n_calc = 1
        else:
            n_calc = max(1, int(math.ceil(span / bs_val)))
        bin_edges = v_min + np.arange(n_calc + 1, dtype=np.float64) * bs_val
        if bin_edges[-1] < v_max:
            bin_edges = np.append(bin_edges, bin_edges[-1] + bs_val)
        return bin_edges, None

    if n_bins is not None:
        nb_val = int(n_bins)
    elif v_max <= v_min or n_valid <= 1 or std_val <= 0.0:
        nb_val = 1
    else:
        scott_w = (3.5 * std_val) / (float(n_valid) ** (1.0 / 3.0))
        if scott_w > 0.0:
            nb_val = max(1, min(1024, int(math.ceil((v_max - v_min) / scott_w))))
        else:
            nb_val = 1

    if v_max <= v_min:
        bin_edges = np.linspace(v_min - 0.5, v_max + 0.5, nb_val + 1, dtype=np.float64)
    else:
        bin_edges = np.linspace(v_min, v_max, nb_val + 1, dtype=np.float64)
    return bin_edges, None


def get_img_histogram(
    input_img: Union[str, os.PathLike],
    img_band: int = 1,
    n_bins: Optional[int] = None,
    bin_size: Optional[float] = None,
    integer_bins: bool = False,
    val_range: Optional[Tuple[float, float]] = None,
    no_data_val: Optional[float] = None,
    use_no_data: bool = True,
    in_msk_img: Optional[Union[str, os.PathLike]] = None,
    img_msk_val: int = 1,
    use_pbar: bool = True,
) -> Dict[str, Any]:
    """
    Calculate a histogram for a single raster band using streaming
    block-wise processing and return counts, bin boundaries, centres,
    widths, proportions, density, and per-bin summary records.

    Binning modes (mutually exclusive):

    * **Auto** (``n_bins=None``, ``bin_size=None``, ``integer_bins=False``):
      Estimates the bin count from the streaming standard deviation using
      Scott's rule.
    * **Fixed number of bins** (``n_bins=<int>``): Divides the data or
      ``val_range`` interval into ``n_bins`` equal-width bins.
    * **Fixed bin width** (``bin_size=<float>``): Builds contiguous bins of
      width ``bin_size`` covering the data or ``val_range`` interval.
    * **Single-integer / categorical bins** (``integer_bins=True``): Creates
      one unit-width bin per integer ``k`` (edges ``[k - 0.5, k + 0.5]``,
      centre ``k``) across ``[min_int, max_int]``, ideal for thematic or
      classified rasters.

    :param input_img: Path to the input raster image.
    :param img_band: 1-based band index to summarise (Default = ``1``).
    :param n_bins: Optional explicit number of equal-width bins (``>= 1``).
    :param bin_size: Optional explicit bin width (``> 0``).
    :param integer_bins: If ``True``, create one bin per integer value
                         (Default = ``False``).
    :param val_range: Optional ``(min_val, max_val)`` range restricting
                      included pixel values and defining outer bin bounds.
    :param no_data_val: Optional no-data value to exclude. If ``None`` and
                        ``use_no_data`` is ``True``, uses the band header
                        no-data value when present.
    :param use_no_data: If ``True``, exclude ``no_data_val`` pixels
                        (Default = ``True``).
    :param in_msk_img: Optional path to a mask raster (band 1 compared to
                       ``img_msk_val``).
    :param img_msk_val: Pixel value in ``in_msk_img`` identifying valid
                        pixels (Default = ``1``).
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).
    :return: Dictionary containing histogram arrays and bin metadata:

             * ``"band"`` (``int``): 1-based band index.
             * ``"n_bins"`` (``int``): Number of histogram bins.
             * ``"bin_width"`` (``float``): Scalar bin width.
             * ``"integer_bins"`` (``bool``): Whether integer binning was used.
             * ``"min_val"`` (``int | float``): Minimum bound of the histogram.
             * ``"max_val"`` (``int | float``): Maximum bound of the histogram.
             * ``"total_count"`` (``int``): Total valid pixels counted.
             * ``"counts"`` (:class:`numpy.ndarray`): 1D ``int64`` bin counts.
             * ``"proportions"`` (:class:`numpy.ndarray`): 1D ``float64``
               per-bin pixel proportions summing to ``1.0`` (when non-empty).
             * ``"density"`` (:class:`numpy.ndarray`): 1D ``float64``
               probability density (``counts / (total_count * bin_widths)``).
             * ``"bin_edges"`` (:class:`numpy.ndarray`): 1D ``float64`` edges
               of length ``n_bins + 1``.
             * ``"bin_centres"`` (:class:`numpy.ndarray`): 1D ``float64`` bin
               centres of length ``n_bins``.
             * ``"bin_widths"`` (:class:`numpy.ndarray`): 1D ``float64`` bin
               widths of length ``n_bins``.
             * ``"bin_values"`` (:class:`numpy.ndarray`, optional): 1D
               ``int64`` integer values when ``integer_bins=True``.
             * ``"bins"`` (``list[dict]``): Per-bin dictionaries with keys
               ``bin_index``, ``lower_edge``, ``upper_edge``, ``centre``,
               ``width``, ``count``, ``proportion``, ``density`` (plus
               ``value`` when ``integer_bins=True``).

    .. code:: python

        import rsgispy.imagecalc.stats as imgstats

        # Continuous histogram with fixed bin width
        ndvi_hist = imgstats.get_img_histogram(
            input_img="ndvi.tif",
            img_band=1,
            bin_size=0.1,
            val_range=(-1.0, 1.0),
        )

        # Categorical integer raster histogram (1 bin per class ID)
        cls_hist = imgstats.get_img_histogram(
            input_img="landcover.tif",
            img_band=1,
            integer_bins=True,
        )
        for bin_info in cls_hist["bins"]:
            print(bin_info["value"], bin_info["count"], bin_info["proportion"])
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-branches
    norm_range = _validate_histogram_options(
        n_bins=n_bins,
        bin_size=bin_size,
        integer_bins=integer_bins,
        val_range=val_range,
    )
    in_path = os.fspath(input_img)
    ensure_file_exists(in_path)
    msk_path = os.fspath(in_msk_img) if in_msk_img is not None else None
    if msk_path is not None:
        ensure_file_exists(msk_path)

    open_paths = [in_path] if msk_path is None else [in_path, msk_path]
    with ExitStack() as stack:
        datasets, _ = open_multi_image_datasets(stack, open_paths)
        img_src = datasets[in_path]
        validate_band_index(img_band, img_src.count, in_path)
        msk_src = datasets[msk_path] if msk_path is not None else None

        eff_nodata = (
            img_src.nodatavals[img_band - 1]
            if (use_no_data and no_data_val is None)
            else no_data_val
        )
        grid = compute_spatial_intersection(datasets)
        base_ci, base_ri = grid.offsets[in_path]
        base_cm, base_rm = grid.offsets[msk_path] if msk_path is not None else (0, 0)

        need_scan = (norm_range is None) or (
            n_bins is None and bin_size is None and not integer_bins
        )
        accum = OnlineStatsAccumulator(1)
        v_min = math.inf
        v_max = -math.inf

        if need_scan:
            for out_win in iter_dataset_windows(
                grid.height, grid.width, use_pbar=use_pbar
            ):
                vals = _extract_valid_hist_pixels(
                    img_src=img_src,
                    img_band=img_band,
                    img_nodata=eff_nodata,
                    use_no_data=use_no_data,
                    msk_src=msk_src,
                    img_msk_val=img_msk_val,
                    base_ci=base_ci,
                    base_ri=base_ri,
                    base_cm=base_cm,
                    base_rm=base_rm,
                    out_win=out_win,
                    val_range=norm_range,
                )
                if vals.size == 0:
                    continue
                if integer_bins and not np.all(vals == np.floor(vals)):
                    raise rsgispy.RSGISPyException(
                        "Input image contains non-integer values; integer_bins=True "
                        "requires integer pixel values."
                    )
                accum.update_band(0, vals)

            if int(accum.counts[0]) == 0 and norm_range is None:
                raise rsgispy.RSGISPyException(
                    "No valid pixels found in input image to compute histogram."
                )

        n_valid = int(accum.counts[0])
        if n_valid > 0:
            v_min = float(accum.mins[0])
            v_max = float(accum.maxs[0])
        if norm_range is not None:
            v_min, v_max = norm_range
        std_val = float(accum.stds()[0]) if n_valid > 0 else 0.0

        bin_edges, int_vals = _build_histogram_edges(
            v_min=v_min,
            v_max=v_max,
            n_valid=n_valid,
            std_val=std_val,
            n_bins=n_bins,
            bin_size=bin_size,
            integer_bins=integer_bins,
        )
        out_n_bins = int(bin_edges.size - 1)
        counts = np.zeros(out_n_bins, dtype=np.int64)
        min_int = int(int_vals[0]) if int_vals is not None else 0

        for out_win in iter_dataset_windows(
            grid.height, grid.width, use_pbar=use_pbar
        ):
            vals = _extract_valid_hist_pixels(
                img_src=img_src,
                img_band=img_band,
                img_nodata=eff_nodata,
                use_no_data=use_no_data,
                msk_src=msk_src,
                img_msk_val=img_msk_val,
                base_ci=base_ci,
                base_ri=base_ri,
                base_cm=base_cm,
                base_rm=base_rm,
                out_win=out_win,
                val_range=norm_range,
            )
            if vals.size == 0:
                continue
            if int_vals is not None:
                if not np.all(vals == np.floor(vals)):
                    raise rsgispy.RSGISPyException(
                        "Input image contains non-integer values; integer_bins=True "
                        "requires integer pixel values."
                    )
                shifted = vals.astype(np.int64) - min_int
                in_int_bounds = (shifted >= 0) & (shifted < out_n_bins)
                if np.any(in_int_bounds):
                    counts += np.bincount(
                        shifted[in_int_bounds].astype(np.intp, copy=False),
                        minlength=out_n_bins,
                    )
            else:
                blk_counts, _ = np.histogram(vals, bins=bin_edges)
                counts += blk_counts.astype(np.int64)

    total_count = int(np.sum(counts))
    bin_widths = np.diff(bin_edges)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    if total_count > 0:
        proportions = counts.astype(np.float64) / float(total_count)
        density = counts.astype(np.float64) / (float(total_count) * bin_widths)
    else:
        proportions = np.zeros(out_n_bins, dtype=np.float64)
        density = np.zeros(out_n_bins, dtype=np.float64)

    bins_info: List[Dict[str, Any]] = []
    for idx in range(out_n_bins):
        entry: Dict[str, Any] = {
            "bin_index": idx,
            "lower_edge": float(bin_edges[idx]),
            "upper_edge": float(bin_edges[idx + 1]),
            "centre": float(bin_centres[idx]),
            "width": float(bin_widths[idx]),
            "count": int(counts[idx]),
            "proportion": float(proportions[idx]),
            "density": float(density[idx]),
        }
        if int_vals is not None:
            entry["value"] = int(int_vals[idx])
        bins_info.append(entry)

    result: Dict[str, Any] = {
        "band": int(img_band),
        "n_bins": out_n_bins,
        "bin_width": float(bin_widths[0]) if out_n_bins > 0 else 0.0,
        "integer_bins": bool(integer_bins),
        "min_val": int(round(v_min)) if integer_bins else float(v_min),
        "max_val": int(round(v_max)) if integer_bins else float(v_max),
        "total_count": total_count,
        "counts": counts,
        "proportions": proportions,
        "density": density,
        "bin_edges": bin_edges,
        "bin_centres": bin_centres,
        "bin_widths": bin_widths,
        "bins": bins_info,
    }
    if int_vals is not None:
        result["bin_values"] = int_vals
    return result


def _check_same_pixel_grid(
    ref_src: Any, src: Any, ref_path: str, img_path: str
) -> None:
    """Raise if ``src`` is not on exactly the same pixel grid as ``ref_src``."""
    same_size = (ref_src.width == src.width) and (ref_src.height == src.height)
    same_trans = bool(
        np.allclose(
            np.asarray(ref_src.transform)[:6],
            np.asarray(src.transform)[:6],
            rtol=0.0,
            atol=1e-6 * max(abs(ref_src.transform.a), 1e-12),
        )
    )
    same_crs = (ref_src.crs is None and src.crs is None) or (
        ref_src.crs is not None and src.crs is not None and ref_src.crs == src.crs
    )
    if not (same_size and same_trans and same_crs):
        raise rsgispy.RSGISPyException(
            f"Image '{img_path}' is not on the same pixel grid (size, transform, "
            f"CRS) as '{ref_path}'. Resample it onto that grid first, e.g. with "
            "rsgispy.imageutils.warp.resample_img_to_match."
        )


def _validate_2d_hist_range(
    val_range: Optional[Tuple[float, float]], label: str
) -> Optional[Tuple[float, float]]:
    """Validate an optional ``(min, max)`` histogram range."""
    if val_range is None:
        return None
    if len(val_range) != 2:
        raise rsgispy.RSGISPyException(f"{label} must be a (min_val, max_val) 2-tuple.")
    r_min = float(val_range[0])
    r_max = float(val_range[1])
    if not (math.isfinite(r_min) and math.isfinite(r_max)) or r_min > r_max:
        raise rsgispy.RSGISPyException(
            f"{label} must contain finite bounds with min_val <= max_val."
        )
    return (r_min, r_max)


def _read_2d_hist_block(
    srcs: Tuple[Any, Any, Optional[Any]],
    bands: Tuple[int, int],
    nodata: Tuple[Optional[float], Optional[float]],
    img_msk_val: int,
    ranges: Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]],
    win: Window,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the 1D ``float64`` x and y values of the valid pixels in ``win``."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    x_src, y_src, msk_src = srcs
    x_blk = x_src.read(bands[0], window=win, out_dtype=np.float64).ravel()
    y_blk = y_src.read(bands[1], window=win, out_dtype=np.float64).ravel()
    msk_arr = msk_src.read(1, window=win).ravel() if msk_src is not None else None
    valid = build_valid_mask(
        x_blk, no_data_val=nodata[0], msk_arr=msk_arr, valid_msk_val=img_msk_val
    )
    valid &= build_valid_mask(y_blk, no_data_val=nodata[1])
    for vals, v_range in zip((x_blk, y_blk), ranges):
        if v_range is not None:
            valid &= (vals >= v_range[0]) & (vals <= v_range[1])
    return x_blk[valid], y_blk[valid]


def _scan_2d_hist_ranges(
    srcs: Tuple[Any, Any, Optional[Any]],
    bands: Tuple[int, int],
    nodata: Tuple[Optional[float], Optional[float]],
    img_msk_val: int,
    ranges: Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]],
    block_size: Optional[int],
    use_pbar: bool,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Fill the ``None`` ranges with the min/max of the valid pixels."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    mins = [math.inf, math.inf]
    maxs = [-math.inf, -math.inf]
    for win in iter_dataset_windows(
        srcs[0].height, srcs[0].width, block_size=block_size, use_pbar=use_pbar
    ):
        x_vals, y_vals = _read_2d_hist_block(
            srcs, bands, nodata, img_msk_val, ranges, win
        )
        if x_vals.size == 0:
            continue
        for idx, vals in enumerate((x_vals, y_vals)):
            mins[idx] = min(mins[idx], float(np.min(vals)))
            maxs[idx] = max(maxs[idx], float(np.max(vals)))
    if not math.isfinite(mins[0]):
        raise rsgispy.RSGISPyException(
            "No valid pixels found to compute the 2D histogram range."
        )
    x_rng = ranges[0] if ranges[0] is not None else (mins[0], maxs[0])
    y_rng = ranges[1] if ranges[1] is not None else (mins[1], maxs[1])
    return x_rng, y_rng


def get_img_2d_histogram(
    input_img: Union[str, os.PathLike],
    x_band: int,
    y_band: int,
    *,
    y_img: Optional[Union[str, os.PathLike]] = None,
    bins: Union[int, Sequence[int]] = 256,
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
    no_data_val: Optional[float] = None,
    in_msk_img: Optional[Union[str, os.PathLike]] = None,
    img_msk_val: int = 1,
    density: bool = False,
    block_size: Optional[int] = None,
    use_pbar: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the 2D histogram (feature space) of two image bands, block by
    block, so memory does not grow with the image size.

    The x values are band ``x_band`` of ``input_img`` and the y values are band
    ``y_band`` of ``y_img`` (an image on the same pixel grid) or, when
    ``y_img`` is ``None``, of ``input_img``. A pixel is counted when both
    values are valid (finite and not the no-data value), inside ``x_range``
    and ``y_range``, and the mask (if given) equals ``img_msk_val``. When a
    range is ``None``, a first pass finds the minimum and maximum of the
    valid pixels. The counts are accumulated tile by tile with fixed edges,
    so the result is identical to :func:`numpy.histogram2d` on the whole
    arrays of valid pixel values (with ``range=[x_range, y_range]``), and
    identical for any ``block_size``.

    This replaces ``rsgislib.imagecalc.get_2d_img_histogram``, which wrote the
    histogram to an image file. To plot it with ``matplotlib``, transpose it
    (it is indexed ``[x, y]``): ``ax.imshow(hist.T, origin='lower',
    extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])``.

    :param input_img: Path to the input image (x band and, by default, y band).
    :param x_band: 1-based band index of the x values in ``input_img``.
    :param y_band: 1-based band index of the y values in ``y_img`` (or in
                   ``input_img`` when ``y_img`` is ``None``).
    :param y_img: Optional second image, on the same pixel grid as
                  ``input_img``, providing ``y_band`` (Default = ``None``).
    :param bins: Number of bins for both axes, or ``[n_x_bins, n_y_bins]``
                 (Default = ``256``), as for :func:`numpy.histogram2d`.
    :param x_range: Optional ``(min_val, max_val)`` of the x axis. Pixels
                    outside it are not counted. When ``None``, the range of the
                    valid pixels.
    :param y_range: Optional ``(min_val, max_val)`` of the y axis, as
                    ``x_range``.
    :param no_data_val: Optional no-data value of both bands. When ``None``,
                        each band's no-data value from the image header (if
                        any) is used.
    :param in_msk_img: Optional mask image (band 1), on the same pixel grid as
                       ``input_img``; only pixels equal to ``img_msk_val`` are
                       counted.
    :param img_msk_val: Mask value of the pixels to count (Default = ``1``).
    :param density: If ``True``, return the probability density
                    (``counts / (total * bin_area)``, as
                    :func:`numpy.histogram2d` with ``density=True``) instead
                    of the counts (Default = ``False``).
    :param block_size: Optional processing block size in pixels (Default =
                       ``None``, the rsgispy default block size). The result
                       does not depend on it.
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).
    :return: ``(hist, x_edges, y_edges)``: ``hist`` is a ``float64`` array of
             shape ``(n_x_bins, n_y_bins)`` indexed ``[x, y]`` (counts, or
             densities with ``density=True``), and ``x_edges`` / ``y_edges``
             are the bin edges (lengths ``n_x_bins + 1`` and ``n_y_bins + 1``).

    .. code:: python

        import rsgispy.imagecalc.stats as imgstats

        # Feature space of the red (band 3) and NIR (band 8) bands
        hist, x_edges, y_edges = imgstats.get_img_2d_histogram(
            "sen2.tif", 3, 8, bins=200, x_range=(0, 5000), y_range=(0, 5000),
            no_data_val=0,
        )
        # Proportion of the valid pixels in each bin
        hist_norm = hist / hist.sum()
    """
    # pylint: disable=too-many-arguments,too-many-locals
    x_rng = _validate_2d_hist_range(x_range, "x_range")
    y_rng = _validate_2d_hist_range(y_range, "y_range")
    in_path = os.fspath(input_img)
    ensure_file_exists(in_path)
    y_path = os.fspath(y_img) if y_img is not None else in_path
    ensure_file_exists(y_path)
    msk_path = os.fspath(in_msk_img) if in_msk_img is not None else None
    if msk_path is not None:
        ensure_file_exists(msk_path)

    with ExitStack() as stack:
        x_src = stack.enter_context(rasterio.open(in_path, "r"))
        y_src = (
            x_src if y_img is None else stack.enter_context(rasterio.open(y_path, "r"))
        )
        msk_src = (
            stack.enter_context(rasterio.open(msk_path, "r"))
            if msk_path is not None
            else None
        )
        validate_band_index(x_band, x_src.count, in_path)
        validate_band_index(y_band, y_src.count, y_path)
        if y_img is not None:
            _check_same_pixel_grid(x_src, y_src, in_path, y_path)
        if msk_src is not None and msk_path is not None:
            _check_same_pixel_grid(x_src, msk_src, in_path, msk_path)

        if no_data_val is not None:
            nodata = (float(no_data_val), float(no_data_val))
        else:
            x_nd = x_src.nodatavals[x_band - 1]
            y_nd = y_src.nodatavals[y_band - 1]
            nodata = (
                float(x_nd) if x_nd is not None else None,
                float(y_nd) if y_nd is not None else None,
            )
        srcs = (x_src, y_src, msk_src)
        band_pair = (int(x_band), int(y_band))

        if x_rng is None or y_rng is None:
            x_rng, y_rng = _scan_2d_hist_ranges(
                srcs, band_pair, nodata, img_msk_val, (x_rng, y_rng),
                block_size, use_pbar,
            )

        hist_range = [list(x_rng), list(y_rng)]
        hist, x_edges, y_edges = np.histogram2d(
            np.empty(0), np.empty(0), bins=bins, range=hist_range
        )
        for win in iter_dataset_windows(
            x_src.height, x_src.width, block_size=block_size, use_pbar=use_pbar
        ):
            x_vals, y_vals = _read_2d_hist_block(
                srcs, band_pair, nodata, img_msk_val, (x_rng, y_rng), win
            )
            if x_vals.size == 0:
                continue
            blk_hist, _, _ = np.histogram2d(
                x_vals, y_vals, bins=bins, range=hist_range
            )
            hist += blk_hist

    if density:
        total = float(np.sum(hist))
        bin_area = np.diff(x_edges)[:, np.newaxis] * np.diff(y_edges)[np.newaxis, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            hist = hist / total / bin_area
    return hist, x_edges, y_edges
