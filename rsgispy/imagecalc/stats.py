"""
The imagecalc.stats module provides basic raster band summary statistics,
pixel value counting, and unique value extraction using ``rasterio``.
"""

from contextlib import ExitStack
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import rasterio
from rasterio.windows import Window

import rsgispy
from rsgispy.imageprocess import (
    cast_result_block,
    compute_spatial_intersection,
    ensure_file_exists,
    iter_dataset_windows,
    iter_single_image_band_blocks,
    open_multi_image_datasets,
    open_output_dataset,
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
    min_vals = np.zeros(n_bands, dtype=np_dt)
    max_vals = np.zeros(n_bands, dtype=np_dt)
    sum_vals = np.zeros(n_bands, dtype=np.float64)
    n_vals = np.zeros(n_bands, dtype=np.int64)
    first_arr = np.ones(n_bands, dtype=bool)

    for _, block_arrays in iter_single_image_band_blocks(
        input_img=input_img, use_pbar=use_pbar
    ):
        for idx, band_arr in enumerate(block_arrays):
            data = band_arr.ravel()
            if use_no_data and no_data_val is not None:
                data = data[data != no_data_val]
            data = data[np.isfinite(data)]
            if data.size > 0:
                d_min = data.min()
                d_max = data.max()
                if first_arr[idx]:
                    min_vals[idx] = d_min
                    max_vals[idx] = d_max
                    first_arr[idx] = False
                else:
                    if d_min < min_vals[idx]:
                        min_vals[idx] = d_min
                    if d_max > max_vals[idx]:
                        max_vals[idx] = d_max
                sum_vals[idx] += float(np.sum(data))
                n_vals[idx] += data.size

    mean_vals = np.where(n_vals > 0, sum_vals / np.maximum(n_vals, 1), 0.0)
    sq_diff_vals = np.zeros(n_bands, dtype=np.float64)

    for _, block_arrays in iter_single_image_band_blocks(
        input_img=input_img, use_pbar=use_pbar
    ):
        for idx, band_arr in enumerate(block_arrays):
            data = band_arr.ravel()
            if use_no_data and no_data_val is not None:
                data = data[data != no_data_val]
            data = data[np.isfinite(data)]
            if data.size > 0:
                sq_diff_vals[idx] += float(
                    np.sum(np.power(data - mean_vals[idx], 2))
                )

    stddev_vals = np.where(
        n_vals > 0, np.sqrt(sq_diff_vals / np.maximum(n_vals, 1)), 0.0
    )

    return min_vals, max_vals, mean_vals, stddev_vals


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
    for _, block_arrays in iter_single_image_band_blocks(
        input_img=input_img,
        read_dtype=None,
        use_pbar=use_pbar,
    ):
        target_arrays = (
            [block_arrays[img_band - 1]]
            if img_band is not None
            else block_arrays
        )
        for band_arr in target_arrays:
            for idx, val in enumerate(vals):
                out_vals[idx] += int(np.count_nonzero(band_arr == val))

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
    ``np.nan`` or non-finite.
    """
    # pylint: disable=too-many-return-statements
    valid_mask = np.isfinite(stack_arr)
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


def _mask_nodata_nan(
    arr: np.ndarray, no_data_val: Optional[float], use_no_data: bool
) -> np.ndarray:
    """Convert non-finite and (when enabled) ``no_data_val`` elements to ``np.nan``."""
    out = arr.astype(np.float64, copy=True)
    invalid = ~np.isfinite(out)
    if use_no_data and no_data_val is not None and not math.isnan(float(no_data_val)):
        invalid |= out == float(no_data_val)
    if np.any(invalid):
        out[invalid] = np.nan
    return out


def image_pixel_column_summary(
    input_img: str,
    output_img: str,
    sum_stats: StatsSummary,
    gdalformat: str = "GTIFF",
    datatype: int = rsgispy.TYPE_32FLOAT,
    no_data_val: float = 0.0,
    use_no_data: bool = True,
    use_pbar: bool = True,
    creation_opts: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Calculate summary statistics across all bands (pixel column) for each
    pixel in ``input_img``.

    Output bands are written in the canonical order: ``Min``, ``Max``, ``Mean``,
    ``Median``, ``Mode``, ``Sum``, ``StdDev`` for whichever corresponding
    ``calc_*`` flags are ``True`` on ``sum_stats``.

    :param input_img: Path to the input multi-band image file.
    :param output_img: Path to the output summary image file.
    :param sum_stats: :class:`StatsSummary` specifying statistics to compute.
    :param gdalformat: GDAL/rasterio driver format for the output file
                       (Default = ``'GTIFF'``).
    :param datatype: Output data type constant from ``rsgispy.TYPE_*``
                     (Default = ``rsgispy.TYPE_32FLOAT``).
    :param no_data_val: Value treated as no-data when ``use_no_data`` is True.
    :param use_no_data: If True, exclude ``no_data_val`` from statistics.
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
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
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
        for out_band_idx, (label, _) in enumerate(selected, start=1):
            dst.set_band_description(out_band_idx, label)

        for out_win in iter_dataset_windows(
            grid.height, grid.width, use_pbar=use_pbar
        ):
            win_h = int(out_win.height)
            win_w = int(out_win.width)
            in_block = src.read(window=out_win, out_dtype=np.float64)
            masked_block = _mask_nodata_nan(in_block, no_data_val, use_no_data)
            out_bands = [
                cast_result_block(
                    _calc_stack_stat(masked_block, stat_id),
                    win_h,
                    win_w,
                    out_np_dtype,
                )
                for _, stat_id in selected
            ]
            dst.write(np.stack(out_bands, axis=0), window=out_win)


def calc_multi_img_band_stats(
    input_imgs: Sequence[str],
    output_img: str,
    summary_stat: int,
    gdalformat: str = "GTIFF",
    datatype: int = rsgispy.TYPE_32FLOAT,
    no_data_val: Optional[float] = None,
    use_no_data: bool = False,
    use_pbar: bool = True,
    creation_opts: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Calculate a summary statistic (``rsgispy.SUMTYPE_*``) across multiple
    input images on a per-band basis.

    All input images must have the same number of bands. The output image has
    the same band count over the spatial intersection of ``input_imgs``.

    :param input_imgs: Sequence of input image paths with identical band counts.
    :param output_img: Path to the output image file.
    :param summary_stat: Summary statistic constant ``rsgispy.SUMTYPE_*``.
    :param gdalformat: GDAL/rasterio driver format for the output file
                       (Default = ``'GTIFF'``).
    :param datatype: Output data type constant from ``rsgispy.TYPE_*``
                     (Default = ``rsgispy.TYPE_32FLOAT``).
    :param no_data_val: Optional no-data value across all input images. If
                        ``None`` and ``use_no_data`` is True, read from the
                        first image header.
    :param use_no_data: If True, exclude ``no_data_val`` pixels (Default = False).
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
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
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
                stat_arr = _calc_stack_stat(masked_stack, summary_stat)
                out_bands.append(
                    cast_result_block(stat_arr, win_h, win_w, out_np_dtype)
                )
            dst.write(np.stack(out_bands, axis=0), window=out_win)


def get_img_sum_stats_in_pxl(
    in_ref_img: str,
    in_stats_img: str,
    output_img: str,
    gdalformat: str = "GTIFF",
    datatype: int = rsgispy.TYPE_32FLOAT,
    sum_stats: Sequence[int] = (rsgispy.SUMTYPE_MEAN,),
    stats_img_band: int = 1,
    use_no_data: bool = True,
    io_grid_x: int = 16,
    io_grid_y: int = 16,
    use_pbar: bool = True,
    creation_opts: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Summarise high-resolution pixel values from ``in_stats_img`` within each
    coarse pixel of ``in_ref_img`` (zonal aggregation on aligned integer-ratio grids).

    The resolution of ``in_ref_img`` must be strictly coarser than and an integer
    multiple of ``in_stats_img`` along both axes. Each entry in ``sum_stats``
    produces one output band.

    :param in_ref_img: Path to the coarser-resolution reference image defining
                       output grid, extent, and resolution.
    :param in_stats_img: Path to the higher-resolution image to summarise.
    :param output_img: Path to the output image file.
    :param gdalformat: GDAL/rasterio driver format for the output file
                       (Default = ``'GTIFF'``).
    :param datatype: Output data type constant from ``rsgispy.TYPE_*``
                     (Default = ``rsgispy.TYPE_32FLOAT``).
    :param sum_stats: Sequence of ``rsgispy.SUMTYPE_*`` statistics to compute
                      (one band per statistic).
    :param stats_img_band: 1-based band index in ``in_stats_img`` (Default = 1).
    :param use_no_data: If True, exclude the ``in_stats_img`` header no-data value
                        (Default = True).
    :param io_grid_x: Block width in reference pixels (Default = 16).
    :param io_grid_y: Block height in reference pixels (Default = 16).
    :param use_pbar: If True, display a tqdm progress bar (Default = True).
    :param creation_opts: Optional mapping of GDAL creation options.

    .. code:: python

        import rsgispy
        import rsgispy.imagecalc.stats

        rsgispy.imagecalc.stats.get_img_sum_stats_in_pxl(
            in_ref_img="coarse_grid.tif",
            in_stats_img="fine_res.tif",
            output_img="zonal_stats.tif",
            gdalformat="GTIFF",
            datatype=rsgispy.TYPE_32FLOAT,
            sum_stats=[rsgispy.SUMTYPE_MEAN, rsgispy.SUMTYPE_STDDEV],
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

        tqdm_cls = rsgispy.get_tqdm()
        total_blocks = math.ceil(grid.height / io_grid_y) * math.ceil(
            grid.width / io_grid_x
        )
        inv_stats_transform = ~stats_src.transform

        with tqdm_cls(total=total_blocks, disable=not use_pbar) as pbar:
            for row_off in range(0, grid.height, io_grid_y):
                win_h = min(io_grid_y, grid.height - row_off)
                for col_off in range(0, grid.width, io_grid_x):
                    win_w = min(io_grid_x, grid.width - col_off)
                    out_win = Window(col_off, row_off, win_w, win_h)

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
                        cast_result_block(
                            _calc_stack_stat(sub_stack, stat_id),
                            win_h,
                            win_w,
                            out_np_dtype,
                        )
                        for stat_id in sum_stats
                    ]
                    dst.write(np.stack(out_bands, axis=0), window=out_win)
                    pbar.update(1)


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
    img_bands: Sequence[int],
    in_msk_img: str,
    msk_val: int,
    output_img: str,
    gdalformat: str = "GTIFF",
    bin_widths: Optional[Sequence[float]] = None,
    use_no_data: bool = True,
    rescale_probs: bool = True,
    use_pbar: bool = True,
    creation_opts: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Calculate the empirical joint n-dimensional probability of pixel values
    across ``img_bands`` based on the distribution inside ``in_msk_img == msk_val``.

    The output image is a single-band ``float32`` image.

    :param input_img: Path to the input image file.
    :param img_bands: Sequence of 1-based band indices in ``input_img``.
    :param in_msk_img: Path to the mask image file (band 1 compared to ``msk_val``).
    :param msk_val: Pixel value in ``in_msk_img`` defining the sample mask.
    :param output_img: Path to the output single-band probability image.
    :param gdalformat: GDAL/rasterio driver format for the output file
                       (Default = ``'GTIFF'``).
    :param bin_widths: Optional sequence of histogram bin widths (one per band).
                       Values ``<= 0`` or ``None`` trigger automatic estimation.
    :param use_no_data: If True, exclude band header no-data values (Default = True).
    :param rescale_probs: If True, rescale probabilities so the maximum is ``1.0``
                          (Default = True).
    :param use_pbar: If True, display a tqdm progress bar (Default = True).
    :param creation_opts: Optional mapping of GDAL creation options.

    .. code:: python

        import rsgispy.imagecalc.stats

        rsgispy.imagecalc.stats.calc_mask_img_pxl_val_prob(
            input_img="multispec.tif",
            img_bands=[1, 2, 3],
            in_msk_img="training_mask.tif",
            msk_val=1,
            output_img="pxl_prob.tif",
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

            valid = np.all(np.isfinite(cols), axis=1)
            if use_no_data:
                for dim_idx, nd_val in enumerate(band_nodatas):
                    if nd_val is not None and not math.isnan(float(nd_val)):
                        valid &= cols[:, dim_idx] != float(nd_val)
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
            in_bounds = np.all(np.isfinite(arr_2d), axis=1)
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

            valid_pxls = np.all(np.isfinite(flat_2d), axis=1)
            if use_no_data:
                for dim_idx, nd_val in enumerate(band_nodatas):
                    if nd_val is not None and not math.isnan(float(nd_val)):
                        valid_pxls &= flat_2d[:, dim_idx] != float(nd_val)

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
