# pylint: disable=too-many-lines
"""
The imagefilter module provides spatial moving-window and convolution filters
for raster images using ``rasterio`` and ``numpy``.

All filters stream images window-by-window with spatial boundary halos, process
every band independently, preserve band descriptions and nodata masks, and
require only core ``rsgispy`` dependencies.
"""

from typing import List, Optional, Sequence, Union

import numpy as np

import rsgispy
from rsgispy.imagefilter._helper import (
    gaussian_kernel,
    laplacian_kernel,
    make_lee_stat_fn,
    prewitt_kernels,
    run_gradient_magnitude_filter,
    run_kernel_filter,
    run_window_stat_filter,
    sobel_kernels,
    stat_coeff_of_var,
    stat_kuwahara,
    stat_max,
    stat_mean,
    stat_mean_diff,
    stat_mean_diff_abs,
    stat_median,
    stat_min,
    stat_mode,
    stat_norm_var,
    stat_norm_var_ln,
    stat_norm_var_sqrt,
    stat_range,
    stat_stddev,
    stat_texture_var,
    stat_total,
    stat_total_diff,
    stat_total_diff_abs,
    validate_positive,
)

__all__: List[str] = [
    "apply_median_filter",
    "apply_mean_filter",
    "apply_min_filter",
    "apply_max_filter",
    "apply_mode_filter",
    "apply_stddev_filter",
    "apply_range_filter",
    "apply_mean_diff_filter",
    "apply_mean_diff_abs_filter",
    "apply_total_diff_filter",
    "apply_total_diff_abs_filter",
    "apply_coeff_of_var_filter",
    "apply_total_filter",
    "apply_norm_var_filter",
    "apply_norm_var_sqrt_filter",
    "apply_norm_var_ln_filter",
    "apply_texture_var_filter",
    "apply_kuwahara_filter",
    "apply_lee_filter",
    "apply_sobel_filter",
    "apply_sobel_x_filter",
    "apply_sobel_y_filter",
    "apply_prewitt_filter",
    "apply_prewitt_x_filter",
    "apply_prewitt_y_filter",
    "apply_gaussian_smooth_filter",
    "apply_gaussian_1st_deriv_filter",
    "apply_gaussian_2nd_deriv_filter",
    "apply_laplacian_filter",
    "apply_kernel_filter",
]


def apply_median_filter(
    input_img: str,
    output_img: str,
    filter_size: int = 3,
    gdalformat: str = "GTIFF",
    datatype: int = rsgispy.TYPE_32FLOAT,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Apply a moving-window median filter to each band of ``input_img``.

    :param input_img: Path to the input raster image.
    :param output_img: Path to the output filtered image.
    :param filter_size: Odd filter window width/height >= 3 (Default = ``3``).
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param datatype: Output pixel data type constant (Default = :data:`rsgispy.TYPE_32FLOAT`).
    :param no_data_val: Optional explicit input nodata value (Default = ``None``).
    :param calc_stats: If ``True``, compute output stats and pyramids (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).

    .. code:: python

        import rsgispy
        import rsgispy.imagefilter

        rsgispy.imagefilter.apply_median_filter(
            "in.tif", "out_median3.tif", filter_size=3
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    run_window_stat_filter(
        input_img=input_img,
        output_img=output_img,
        filter_size=filter_size,
        stat_fn=stat_median,
        gdalformat=gdalformat,
        datatype=datatype,
        no_data_val=no_data_val,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )


def apply_mean_filter(
    input_img: str,
    output_img: str,
    filter_size: int = 3,
    gdalformat: str = "GTIFF",
    datatype: int = rsgispy.TYPE_32FLOAT,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Apply a moving-window arithmetic mean filter to each band of ``input_img``.

    :param input_img: Path to the input raster image.
    :param output_img: Path to the output filtered image.
    :param filter_size: Odd filter window width/height >= 3 (Default = ``3``).
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param datatype: Output pixel data type constant (Default = :data:`rsgispy.TYPE_32FLOAT`).
    :param no_data_val: Optional explicit input nodata value (Default = ``None``).
    :param calc_stats: If ``True``, compute output stats and pyramids (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).

    .. code:: python

        import rsgispy.imagefilter

        rsgispy.imagefilter.apply_mean_filter("in.tif", "out_mean3.tif", filter_size=3)
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    run_window_stat_filter(
        input_img=input_img,
        output_img=output_img,
        filter_size=filter_size,
        stat_fn=stat_mean,
        gdalformat=gdalformat,
        datatype=datatype,
        no_data_val=no_data_val,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )


def apply_min_filter(
    input_img: str,
    output_img: str,
    filter_size: int = 3,
    gdalformat: str = "GTIFF",
    datatype: int = rsgispy.TYPE_32FLOAT,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Apply a moving-window minimum filter to each band of ``input_img``.

    :param input_img: Path to the input raster image.
    :param output_img: Path to the output filtered image.
    :param filter_size: Odd filter window width/height >= 3 (Default = ``3``).
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param datatype: Output pixel data type constant (Default = :data:`rsgispy.TYPE_32FLOAT`).
    :param no_data_val: Optional explicit input nodata value (Default = ``None``).
    :param calc_stats: If ``True``, compute output stats and pyramids (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).

    .. code:: python

        import rsgispy.imagefilter

        rsgispy.imagefilter.apply_min_filter("in.tif", "out_min3.tif", filter_size=3)
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    run_window_stat_filter(
        input_img=input_img,
        output_img=output_img,
        filter_size=filter_size,
        stat_fn=stat_min,
        gdalformat=gdalformat,
        datatype=datatype,
        no_data_val=no_data_val,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )


def apply_max_filter(
    input_img: str,
    output_img: str,
    filter_size: int = 3,
    gdalformat: str = "GTIFF",
    datatype: int = rsgispy.TYPE_32FLOAT,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Apply a moving-window maximum filter to each band of ``input_img``.

    :param input_img: Path to the input raster image.
    :param output_img: Path to the output filtered image.
    :param filter_size: Odd filter window width/height >= 3 (Default = ``3``).
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param datatype: Output pixel data type constant (Default = :data:`rsgispy.TYPE_32FLOAT`).
    :param no_data_val: Optional explicit input nodata value (Default = ``None``).
    :param calc_stats: If ``True``, compute output stats and pyramids (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).

    .. code:: python

        import rsgispy.imagefilter

        rsgispy.imagefilter.apply_max_filter("in.tif", "out_max3.tif", filter_size=3)
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    run_window_stat_filter(
        input_img=input_img,
        output_img=output_img,
        filter_size=filter_size,
        stat_fn=stat_max,
        gdalformat=gdalformat,
        datatype=datatype,
        no_data_val=no_data_val,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )


def apply_mode_filter(
    input_img: str,
    output_img: str,
    filter_size: int = 3,
    gdalformat: str = "GTIFF",
    datatype: int = rsgispy.TYPE_32FLOAT,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Apply a moving-window modal (most frequent value) filter to ``input_img``.

    Ties are broken deterministically by choosing the smallest modal value.

    :param input_img: Path to the input raster image.
    :param output_img: Path to the output filtered image.
    :param filter_size: Odd filter window width/height >= 3 (Default = ``3``).
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param datatype: Output pixel data type constant (Default = :data:`rsgispy.TYPE_32FLOAT`).
    :param no_data_val: Optional explicit input nodata value (Default = ``None``).
    :param calc_stats: If ``True``, compute output stats and pyramids (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).

    .. code:: python

        import rsgispy
        import rsgispy.imagefilter

        rsgispy.imagefilter.apply_mode_filter(
            "classes.tif", "out_mode3.tif", filter_size=3, datatype=rsgispy.TYPE_8UINT
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    run_window_stat_filter(
        input_img=input_img,
        output_img=output_img,
        filter_size=filter_size,
        stat_fn=stat_mode,
        gdalformat=gdalformat,
        datatype=datatype,
        no_data_val=no_data_val,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )


def apply_stddev_filter(
    input_img: str,
    output_img: str,
    filter_size: int = 3,
    gdalformat: str = "GTIFF",
    datatype: int = rsgispy.TYPE_32FLOAT,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Apply a moving-window standard deviation filter to ``input_img``.

    :param input_img: Path to the input raster image.
    :param output_img: Path to the output filtered image.
    :param filter_size: Odd filter window width/height >= 3 (Default = ``3``).
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param datatype: Output pixel data type constant (Default = :data:`rsgispy.TYPE_32FLOAT`).
    :param no_data_val: Optional explicit input nodata value (Default = ``None``).
    :param calc_stats: If ``True``, compute output stats and pyramids (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).

    .. code:: python

        import rsgispy.imagefilter

        rsgispy.imagefilter.apply_stddev_filter("in.tif", "out_std3.tif", filter_size=3)
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    run_window_stat_filter(
        input_img=input_img,
        output_img=output_img,
        filter_size=filter_size,
        stat_fn=stat_stddev,
        gdalformat=gdalformat,
        datatype=datatype,
        no_data_val=no_data_val,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )


def apply_range_filter(
    input_img: str,
    output_img: str,
    filter_size: int = 3,
    gdalformat: str = "GTIFF",
    datatype: int = rsgispy.TYPE_32FLOAT,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Apply a moving-window range (``max - min``) filter to ``input_img``.

    :param input_img: Path to the input raster image.
    :param output_img: Path to the output filtered image.
    :param filter_size: Odd filter window width/height >= 3 (Default = ``3``).
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param datatype: Output pixel data type constant (Default = :data:`rsgispy.TYPE_32FLOAT`).
    :param no_data_val: Optional explicit input nodata value (Default = ``None``).
    :param calc_stats: If ``True``, compute output stats and pyramids (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).

    .. code:: python

        import rsgispy.imagefilter

        rsgispy.imagefilter.apply_range_filter("in.tif", "out_range3.tif", filter_size=3)
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    run_window_stat_filter(
        input_img=input_img,
        output_img=output_img,
        filter_size=filter_size,
        stat_fn=stat_range,
        gdalformat=gdalformat,
        datatype=datatype,
        no_data_val=no_data_val,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )


def apply_total_filter(
    input_img: str,
    output_img: str,
    filter_size: int = 3,
    gdalformat: str = "GTIFF",
    datatype: int = rsgispy.TYPE_32FLOAT,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Apply a moving-window sum (total) filter to ``input_img``.

    :param input_img: Path to the input raster image.
    :param output_img: Path to the output filtered image.
    :param filter_size: Odd filter window width/height >= 3 (Default = ``3``).
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param datatype: Output pixel data type constant (Default = :data:`rsgispy.TYPE_32FLOAT`).
    :param no_data_val: Optional explicit input nodata value (Default = ``None``).
    :param calc_stats: If ``True``, compute output stats and pyramids (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).

    .. code:: python

        import rsgispy.imagefilter

        rsgispy.imagefilter.apply_total_filter("in.tif", "out_total3.tif", filter_size=3)
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    run_window_stat_filter(
        input_img=input_img,
        output_img=output_img,
        filter_size=filter_size,
        stat_fn=stat_total,
        gdalformat=gdalformat,
        datatype=datatype,
        no_data_val=no_data_val,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )


def apply_mean_diff_filter(
    input_img: str,
    output_img: str,
    filter_size: int = 3,
    gdalformat: str = "GTIFF",
    datatype: int = rsgispy.TYPE_32FLOAT,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Apply a moving-window mean signed difference filter (``mean(window - centre)``).

    :param input_img: Path to the input raster image.
    :param output_img: Path to the output filtered image.
    :param filter_size: Odd filter window width/height >= 3 (Default = ``3``).
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param datatype: Output pixel data type constant (Default = :data:`rsgispy.TYPE_32FLOAT`).
    :param no_data_val: Optional explicit input nodata value (Default = ``None``).
    :param calc_stats: If ``True``, compute output stats and pyramids (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).

    .. code:: python

        import rsgispy.imagefilter

        rsgispy.imagefilter.apply_mean_diff_filter("in.tif", "out_meandiff3.tif", filter_size=3)
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    run_window_stat_filter(
        input_img=input_img,
        output_img=output_img,
        filter_size=filter_size,
        stat_fn=stat_mean_diff,
        gdalformat=gdalformat,
        datatype=datatype,
        no_data_val=no_data_val,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )


def apply_mean_diff_abs_filter(
    input_img: str,
    output_img: str,
    filter_size: int = 3,
    gdalformat: str = "GTIFF",
    datatype: int = rsgispy.TYPE_32FLOAT,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Apply a moving-window mean absolute difference filter (``mean(|window - centre|)``).

    :param input_img: Path to the input raster image.
    :param output_img: Path to the output filtered image.
    :param filter_size: Odd filter window width/height >= 3 (Default = ``3``).
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param datatype: Output pixel data type constant (Default = :data:`rsgispy.TYPE_32FLOAT`).
    :param no_data_val: Optional explicit input nodata value (Default = ``None``).
    :param calc_stats: If ``True``, compute output stats and pyramids (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).

    .. code:: python

        import rsgispy.imagefilter

        rsgispy.imagefilter.apply_mean_diff_abs_filter(
            "in.tif", "out_meandiffabs3.tif", filter_size=3
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    run_window_stat_filter(
        input_img=input_img,
        output_img=output_img,
        filter_size=filter_size,
        stat_fn=stat_mean_diff_abs,
        gdalformat=gdalformat,
        datatype=datatype,
        no_data_val=no_data_val,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )


def apply_total_diff_filter(
    input_img: str,
    output_img: str,
    filter_size: int = 3,
    gdalformat: str = "GTIFF",
    datatype: int = rsgispy.TYPE_32FLOAT,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Apply a moving-window total signed difference filter (``sum(window - centre)``).

    :param input_img: Path to the input raster image.
    :param output_img: Path to the output filtered image.
    :param filter_size: Odd filter window width/height >= 3 (Default = ``3``).
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param datatype: Output pixel data type constant (Default = :data:`rsgispy.TYPE_32FLOAT`).
    :param no_data_val: Optional explicit input nodata value (Default = ``None``).
    :param calc_stats: If ``True``, compute output stats and pyramids (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).

    .. code:: python

        import rsgispy.imagefilter

        rsgispy.imagefilter.apply_total_diff_filter(
            "in.tif", "out_totaldiff3.tif", filter_size=3
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    run_window_stat_filter(
        input_img=input_img,
        output_img=output_img,
        filter_size=filter_size,
        stat_fn=stat_total_diff,
        gdalformat=gdalformat,
        datatype=datatype,
        no_data_val=no_data_val,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )


def apply_total_diff_abs_filter(
    input_img: str,
    output_img: str,
    filter_size: int = 3,
    gdalformat: str = "GTIFF",
    datatype: int = rsgispy.TYPE_32FLOAT,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Apply a moving-window total absolute difference filter (``sum(|window - centre|)``).

    :param input_img: Path to the input raster image.
    :param output_img: Path to the output filtered image.
    :param filter_size: Odd filter window width/height >= 3 (Default = ``3``).
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param datatype: Output pixel data type constant (Default = :data:`rsgispy.TYPE_32FLOAT`).
    :param no_data_val: Optional explicit input nodata value (Default = ``None``).
    :param calc_stats: If ``True``, compute output stats and pyramids (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).

    .. code:: python

        import rsgispy.imagefilter

        rsgispy.imagefilter.apply_total_diff_abs_filter(
            "in.tif", "out_totaldiffabs3.tif", filter_size=3
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    run_window_stat_filter(
        input_img=input_img,
        output_img=output_img,
        filter_size=filter_size,
        stat_fn=stat_total_diff_abs,
        gdalformat=gdalformat,
        datatype=datatype,
        no_data_val=no_data_val,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )


def apply_coeff_of_var_filter(
    input_img: str,
    output_img: str,
    filter_size: int = 3,
    gdalformat: str = "GTIFF",
    datatype: int = rsgispy.TYPE_32FLOAT,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Apply a moving-window coefficient of variation filter (``std / mean``).

    Outputs ``0`` where the local mean is ``0``.

    :param input_img: Path to the input raster image.
    :param output_img: Path to the output filtered image.
    :param filter_size: Odd filter window width/height >= 3 (Default = ``3``).
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param datatype: Output pixel data type constant (Default = :data:`rsgispy.TYPE_32FLOAT`).
    :param no_data_val: Optional explicit input nodata value (Default = ``None``).
    :param calc_stats: If ``True``, compute output stats and pyramids (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).

    .. code:: python

        import rsgispy.imagefilter

        rsgispy.imagefilter.apply_coeff_of_var_filter("in.tif", "out_cv3.tif", filter_size=3)
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    run_window_stat_filter(
        input_img=input_img,
        output_img=output_img,
        filter_size=filter_size,
        stat_fn=stat_coeff_of_var,
        gdalformat=gdalformat,
        datatype=datatype,
        no_data_val=no_data_val,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )


def apply_norm_var_filter(
    input_img: str,
    output_img: str,
    filter_size: int = 3,
    gdalformat: str = "GTIFF",
    datatype: int = rsgispy.TYPE_32FLOAT,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Apply a moving-window normalised variance filter (``var / mean**2``).

    :param input_img: Path to the input raster image.
    :param output_img: Path to the output filtered image.
    :param filter_size: Odd filter window width/height >= 3 (Default = ``3``).
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param datatype: Output pixel data type constant (Default = :data:`rsgispy.TYPE_32FLOAT`).
    :param no_data_val: Optional explicit input nodata value (Default = ``None``).
    :param calc_stats: If ``True``, compute output stats and pyramids (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).

    .. code:: python

        import rsgispy.imagefilter

        rsgispy.imagefilter.apply_norm_var_filter("in.tif", "out_normvar3.tif", filter_size=3)
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    run_window_stat_filter(
        input_img=input_img,
        output_img=output_img,
        filter_size=filter_size,
        stat_fn=stat_norm_var,
        gdalformat=gdalformat,
        datatype=datatype,
        no_data_val=no_data_val,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )


def apply_norm_var_sqrt_filter(
    input_img: str,
    output_img: str,
    filter_size: int = 3,
    gdalformat: str = "GTIFF",
    datatype: int = rsgispy.TYPE_32FLOAT,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Apply a moving-window square-root normalised variance filter
    (``sqrt(var / mean**2)``).

    :param input_img: Path to the input raster image.
    :param output_img: Path to the output filtered image.
    :param filter_size: Odd filter window width/height >= 3 (Default = ``3``).
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param datatype: Output pixel data type constant (Default = :data:`rsgispy.TYPE_32FLOAT`).
    :param no_data_val: Optional explicit input nodata value (Default = ``None``).
    :param calc_stats: If ``True``, compute output stats and pyramids (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).

    .. code:: python

        import rsgispy.imagefilter

        rsgispy.imagefilter.apply_norm_var_sqrt_filter(
            "in.tif", "out_normvarsqrt3.tif", filter_size=3
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    run_window_stat_filter(
        input_img=input_img,
        output_img=output_img,
        filter_size=filter_size,
        stat_fn=stat_norm_var_sqrt,
        gdalformat=gdalformat,
        datatype=datatype,
        no_data_val=no_data_val,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )


def apply_norm_var_ln_filter(
    input_img: str,
    output_img: str,
    filter_size: int = 3,
    gdalformat: str = "GTIFF",
    datatype: int = rsgispy.TYPE_32FLOAT,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Apply a moving-window natural-log normalised variance filter
    (``ln(1 + var / mean**2)``).

    :param input_img: Path to the input raster image.
    :param output_img: Path to the output filtered image.
    :param filter_size: Odd filter window width/height >= 3 (Default = ``3``).
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param datatype: Output pixel data type constant (Default = :data:`rsgispy.TYPE_32FLOAT`).
    :param no_data_val: Optional explicit input nodata value (Default = ``None``).
    :param calc_stats: If ``True``, compute output stats and pyramids (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).

    .. code:: python

        import rsgispy.imagefilter

        rsgispy.imagefilter.apply_norm_var_ln_filter(
            "in.tif", "out_normvarln3.tif", filter_size=3
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    run_window_stat_filter(
        input_img=input_img,
        output_img=output_img,
        filter_size=filter_size,
        stat_fn=stat_norm_var_ln,
        gdalformat=gdalformat,
        datatype=datatype,
        no_data_val=no_data_val,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )


def apply_texture_var_filter(
    input_img: str,
    output_img: str,
    filter_size: int = 3,
    gdalformat: str = "GTIFF",
    datatype: int = rsgispy.TYPE_32FLOAT,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Apply a moving-window local texture variance filter to ``input_img``.

    :param input_img: Path to the input raster image.
    :param output_img: Path to the output filtered image.
    :param filter_size: Odd filter window width/height >= 3 (Default = ``3``).
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param datatype: Output pixel data type constant (Default = :data:`rsgispy.TYPE_32FLOAT`).
    :param no_data_val: Optional explicit input nodata value (Default = ``None``).
    :param calc_stats: If ``True``, compute output stats and pyramids (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).

    .. code:: python

        import rsgispy.imagefilter

        rsgispy.imagefilter.apply_texture_var_filter("in.tif", "out_var3.tif", filter_size=3)
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    run_window_stat_filter(
        input_img=input_img,
        output_img=output_img,
        filter_size=filter_size,
        stat_fn=stat_texture_var,
        gdalformat=gdalformat,
        datatype=datatype,
        no_data_val=no_data_val,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )


def apply_kuwahara_filter(
    input_img: str,
    output_img: str,
    filter_size: int = 5,
    gdalformat: str = "GTIFF",
    datatype: int = rsgispy.TYPE_32FLOAT,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Apply an edge-preserving 4-quadrant Kuwahara smoothing filter.

    Divides the ``filter_size x filter_size`` window into four overlapping
    ``((filter_size + 1) // 2)`` quadrants and outputs the mean of the quadrant
    with the lowest local variance.

    :param input_img: Path to the input raster image.
    :param output_img: Path to the output filtered image.
    :param filter_size: Odd filter window width/height >= 3 (Default = ``5``).
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param datatype: Output pixel data type constant (Default = :data:`rsgispy.TYPE_32FLOAT`).
    :param no_data_val: Optional explicit input nodata value (Default = ``None``).
    :param calc_stats: If ``True``, compute output stats and pyramids (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).

    .. code:: python

        import rsgispy.imagefilter

        rsgispy.imagefilter.apply_kuwahara_filter("in.tif", "out_kuwahara5.tif", filter_size=5)
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    run_window_stat_filter(
        input_img=input_img,
        output_img=output_img,
        filter_size=filter_size,
        stat_fn=stat_kuwahara,
        gdalformat=gdalformat,
        datatype=datatype,
        no_data_val=no_data_val,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )


def apply_lee_filter(
    input_img: str,
    output_img: str,
    filter_size: int = 3,
    n_looks: float = 1.0,
    gdalformat: str = "GTIFF",
    datatype: int = rsgispy.TYPE_32FLOAT,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Apply a Lee (1980) adaptive SAR speckle filter to ``input_img``.

    Uses local window mean and variance against the expected speckle variance
    ``1 / n_looks`` to smooth homogeneous areas towards the local mean while
    preserving strong edges and point targets.

    :param input_img: Path to the input SAR raster image.
    :param output_img: Path to the output despeckled image.
    :param filter_size: Odd filter window width/height >= 3 (Default = ``3``).
    :param n_looks: Equivalent number of looks (> 0; Default = ``1.0``).
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param datatype: Output pixel data type constant (Default = :data:`rsgispy.TYPE_32FLOAT`).
    :param no_data_val: Optional explicit input nodata value (Default = ``None``).
    :param calc_stats: If ``True``, compute output stats and pyramids (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).

    .. code:: python

        import rsgispy.imagefilter

        rsgispy.imagefilter.apply_lee_filter("sar.tif", "sar_lee5.tif", filter_size=5, n_looks=4)
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    looks = validate_positive(n_looks, "n_looks")
    run_window_stat_filter(
        input_img=input_img,
        output_img=output_img,
        filter_size=filter_size,
        stat_fn=make_lee_stat_fn(looks),
        gdalformat=gdalformat,
        datatype=datatype,
        no_data_val=no_data_val,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )


def apply_sobel_filter(
    input_img: str,
    output_img: str,
    gdalformat: str = "GTIFF",
    datatype: int = rsgispy.TYPE_32FLOAT,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Apply a 3x3 Sobel edge-magnitude filter (``sqrt(Gx**2 + Gy**2)``).

    :param input_img: Path to the input raster image.
    :param output_img: Path to the output filtered image.
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param datatype: Output pixel data type constant (Default = :data:`rsgispy.TYPE_32FLOAT`).
    :param no_data_val: Optional explicit input nodata value (Default = ``None``).
    :param calc_stats: If ``True``, compute output stats and pyramids (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).

    .. code:: python

        import rsgispy.imagefilter

        rsgispy.imagefilter.apply_sobel_filter("in.tif", "out_sobel.tif")
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    kx, ky = sobel_kernels()
    run_gradient_magnitude_filter(
        input_img=input_img,
        output_img=output_img,
        kernel_x=kx,
        kernel_y=ky,
        gdalformat=gdalformat,
        datatype=datatype,
        no_data_val=no_data_val,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )


def apply_sobel_x_filter(
    input_img: str,
    output_img: str,
    gdalformat: str = "GTIFF",
    datatype: int = rsgispy.TYPE_32FLOAT,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Apply a horizontal (X-axis) 3x3 Sobel derivative filter.

    :param input_img: Path to the input raster image.
    :param output_img: Path to the output filtered image.
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param datatype: Output pixel data type constant (Default = :data:`rsgispy.TYPE_32FLOAT`).
    :param no_data_val: Optional explicit input nodata value (Default = ``None``).
    :param calc_stats: If ``True``, compute output stats and pyramids (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).

    .. code:: python

        import rsgispy.imagefilter

        rsgispy.imagefilter.apply_sobel_x_filter("in.tif", "out_sobel_x.tif")
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    kx, _ = sobel_kernels()
    run_kernel_filter(
        input_img=input_img,
        output_img=output_img,
        kernel=kx,
        zero_dc=True,
        gdalformat=gdalformat,
        datatype=datatype,
        no_data_val=no_data_val,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )


def apply_sobel_y_filter(
    input_img: str,
    output_img: str,
    gdalformat: str = "GTIFF",
    datatype: int = rsgispy.TYPE_32FLOAT,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Apply a vertical (Y-axis) 3x3 Sobel derivative filter.

    :param input_img: Path to the input raster image.
    :param output_img: Path to the output filtered image.
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param datatype: Output pixel data type constant (Default = :data:`rsgispy.TYPE_32FLOAT`).
    :param no_data_val: Optional explicit input nodata value (Default = ``None``).
    :param calc_stats: If ``True``, compute output stats and pyramids (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).

    .. code:: python

        import rsgispy.imagefilter

        rsgispy.imagefilter.apply_sobel_y_filter("in.tif", "out_sobel_y.tif")
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    _, ky = sobel_kernels()
    run_kernel_filter(
        input_img=input_img,
        output_img=output_img,
        kernel=ky,
        zero_dc=True,
        gdalformat=gdalformat,
        datatype=datatype,
        no_data_val=no_data_val,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )


def apply_prewitt_filter(
    input_img: str,
    output_img: str,
    gdalformat: str = "GTIFF",
    datatype: int = rsgispy.TYPE_32FLOAT,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Apply a 3x3 Prewitt edge-magnitude filter (``sqrt(Gx**2 + Gy**2)``).

    :param input_img: Path to the input raster image.
    :param output_img: Path to the output filtered image.
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param datatype: Output pixel data type constant (Default = :data:`rsgispy.TYPE_32FLOAT`).
    :param no_data_val: Optional explicit input nodata value (Default = ``None``).
    :param calc_stats: If ``True``, compute output stats and pyramids (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).

    .. code:: python

        import rsgispy.imagefilter

        rsgispy.imagefilter.apply_prewitt_filter("in.tif", "out_prewitt.tif")
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    kx, ky = prewitt_kernels()
    run_gradient_magnitude_filter(
        input_img=input_img,
        output_img=output_img,
        kernel_x=kx,
        kernel_y=ky,
        gdalformat=gdalformat,
        datatype=datatype,
        no_data_val=no_data_val,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )


def apply_prewitt_x_filter(
    input_img: str,
    output_img: str,
    gdalformat: str = "GTIFF",
    datatype: int = rsgispy.TYPE_32FLOAT,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Apply a horizontal (X-axis) 3x3 Prewitt derivative filter.

    :param input_img: Path to the input raster image.
    :param output_img: Path to the output filtered image.
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param datatype: Output pixel data type constant (Default = :data:`rsgispy.TYPE_32FLOAT`).
    :param no_data_val: Optional explicit input nodata value (Default = ``None``).
    :param calc_stats: If ``True``, compute output stats and pyramids (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).

    .. code:: python

        import rsgispy.imagefilter

        rsgispy.imagefilter.apply_prewitt_x_filter("in.tif", "out_prewitt_x.tif")
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    kx, _ = prewitt_kernels()
    run_kernel_filter(
        input_img=input_img,
        output_img=output_img,
        kernel=kx,
        zero_dc=True,
        gdalformat=gdalformat,
        datatype=datatype,
        no_data_val=no_data_val,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )


def apply_prewitt_y_filter(
    input_img: str,
    output_img: str,
    gdalformat: str = "GTIFF",
    datatype: int = rsgispy.TYPE_32FLOAT,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Apply a vertical (Y-axis) 3x3 Prewitt derivative filter.

    :param input_img: Path to the input raster image.
    :param output_img: Path to the output filtered image.
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param datatype: Output pixel data type constant (Default = :data:`rsgispy.TYPE_32FLOAT`).
    :param no_data_val: Optional explicit input nodata value (Default = ``None``).
    :param calc_stats: If ``True``, compute output stats and pyramids (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).

    .. code:: python

        import rsgispy.imagefilter

        rsgispy.imagefilter.apply_prewitt_y_filter("in.tif", "out_prewitt_y.tif")
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    _, ky = prewitt_kernels()
    run_kernel_filter(
        input_img=input_img,
        output_img=output_img,
        kernel=ky,
        zero_dc=True,
        gdalformat=gdalformat,
        datatype=datatype,
        no_data_val=no_data_val,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )


def apply_gaussian_smooth_filter(
    input_img: str,
    output_img: str,
    filter_size: int = 3,
    stddev_x: float = 1.0,
    stddev_y: float = 1.0,
    filter_angle: float = 0.0,
    gdalformat: str = "GTIFF",
    datatype: int = rsgispy.TYPE_32FLOAT,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Apply an oriented 2D anisotropic Gaussian smoothing filter.

    :param input_img: Path to the input raster image.
    :param output_img: Path to the output smoothed image.
    :param filter_size: Odd kernel width/height >= 3 (Default = ``3``).
    :param stddev_x: Standard deviation along the primary rotated axis (Default = ``1.0``).
    :param stddev_y: Standard deviation along the orthogonal axis (Default = ``1.0``).
    :param filter_angle: Orientation angle in degrees (Default = ``0.0``).
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param datatype: Output pixel data type constant (Default = :data:`rsgispy.TYPE_32FLOAT`).
    :param no_data_val: Optional explicit input nodata value (Default = ``None``).
    :param calc_stats: If ``True``, compute output stats and pyramids (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).

    .. code:: python

        import rsgispy.imagefilter

        rsgispy.imagefilter.apply_gaussian_smooth_filter(
            "in.tif", "out_gauss5.tif", filter_size=5, stddev_x=1.2, stddev_y=1.2
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    kernel = gaussian_kernel(
        filter_size, stddev_x, stddev_y, filter_angle, deriv_order=0
    )
    run_kernel_filter(
        input_img=input_img,
        output_img=output_img,
        kernel=kernel,
        zero_dc=False,
        gdalformat=gdalformat,
        datatype=datatype,
        no_data_val=no_data_val,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )


def apply_gaussian_1st_deriv_filter(
    input_img: str,
    output_img: str,
    filter_size: int = 3,
    stddev_x: float = 1.0,
    stddev_y: float = 1.0,
    filter_angle: float = 0.0,
    gdalformat: str = "GTIFF",
    datatype: int = rsgispy.TYPE_32FLOAT,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Apply an oriented 2D Gaussian first-derivative filter.

    :param input_img: Path to the input raster image.
    :param output_img: Path to the output filtered image.
    :param filter_size: Odd kernel width/height >= 3 (Default = ``3``).
    :param stddev_x: Standard deviation along the primary rotated axis (Default = ``1.0``).
    :param stddev_y: Standard deviation along the orthogonal axis (Default = ``1.0``).
    :param filter_angle: Orientation angle in degrees (Default = ``0.0``).
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param datatype: Output pixel data type constant (Default = :data:`rsgispy.TYPE_32FLOAT`).
    :param no_data_val: Optional explicit input nodata value (Default = ``None``).
    :param calc_stats: If ``True``, compute output stats and pyramids (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).

    .. code:: python

        import rsgispy.imagefilter

        rsgispy.imagefilter.apply_gaussian_1st_deriv_filter(
            "in.tif", "out_gauss1st.tif", filter_size=5, stddev_x=1.0, stddev_y=1.0
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    kernel = gaussian_kernel(
        filter_size, stddev_x, stddev_y, filter_angle, deriv_order=1
    )
    run_kernel_filter(
        input_img=input_img,
        output_img=output_img,
        kernel=kernel,
        zero_dc=True,
        gdalformat=gdalformat,
        datatype=datatype,
        no_data_val=no_data_val,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )


def apply_gaussian_2nd_deriv_filter(
    input_img: str,
    output_img: str,
    filter_size: int = 3,
    stddev_x: float = 1.0,
    stddev_y: float = 1.0,
    filter_angle: float = 0.0,
    gdalformat: str = "GTIFF",
    datatype: int = rsgispy.TYPE_32FLOAT,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Apply an oriented 2D Gaussian second-derivative filter.

    :param input_img: Path to the input raster image.
    :param output_img: Path to the output filtered image.
    :param filter_size: Odd kernel width/height >= 3 (Default = ``3``).
    :param stddev_x: Standard deviation along the primary rotated axis (Default = ``1.0``).
    :param stddev_y: Standard deviation along the orthogonal axis (Default = ``1.0``).
    :param filter_angle: Orientation angle in degrees (Default = ``0.0``).
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param datatype: Output pixel data type constant (Default = :data:`rsgispy.TYPE_32FLOAT`).
    :param no_data_val: Optional explicit input nodata value (Default = ``None``).
    :param calc_stats: If ``True``, compute output stats and pyramids (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).

    .. code:: python

        import rsgispy.imagefilter

        rsgispy.imagefilter.apply_gaussian_2nd_deriv_filter(
            "in.tif", "out_gauss2nd.tif", filter_size=5, stddev_x=1.0, stddev_y=1.0
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    kernel = gaussian_kernel(
        filter_size, stddev_x, stddev_y, filter_angle, deriv_order=2
    )
    run_kernel_filter(
        input_img=input_img,
        output_img=output_img,
        kernel=kernel,
        zero_dc=True,
        gdalformat=gdalformat,
        datatype=datatype,
        no_data_val=no_data_val,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )


def apply_laplacian_filter(
    input_img: str,
    output_img: str,
    filter_size: int = 3,
    stddev: float = 1.0,
    gdalformat: str = "GTIFF",
    datatype: int = rsgispy.TYPE_32FLOAT,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Apply a Laplacian-of-Gaussian (LoG) zero-sum edge/blob detection filter.

    :param input_img: Path to the input raster image.
    :param output_img: Path to the output filtered image.
    :param filter_size: Odd kernel width/height >= 3 (Default = ``3``).
    :param stddev: Gaussian standard deviation > 0 (Default = ``1.0``).
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param datatype: Output pixel data type constant (Default = :data:`rsgispy.TYPE_32FLOAT`).
    :param no_data_val: Optional explicit input nodata value (Default = ``None``).
    :param calc_stats: If ``True``, compute output stats and pyramids (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).

    .. code:: python

        import rsgispy.imagefilter

        rsgispy.imagefilter.apply_laplacian_filter(
            "in.tif", "out_laplacian.tif", filter_size=5, stddev=1.0
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    kernel = laplacian_kernel(filter_size, stddev)
    run_kernel_filter(
        input_img=input_img,
        output_img=output_img,
        kernel=kernel,
        zero_dc=True,
        gdalformat=gdalformat,
        datatype=datatype,
        no_data_val=no_data_val,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )


def apply_kernel_filter(
    input_img: str,
    output_img: str,
    kernel: Union[np.ndarray, Sequence[Sequence[float]]],
    gdalformat: str = "GTIFF",
    datatype: int = rsgispy.TYPE_32FLOAT,
    normalise: bool = False,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Convolve each band of ``input_img`` with a custom 2D odd-dimension kernel.

    :param input_img: Path to the input raster image.
    :param output_img: Path to the output convolved image.
    :param kernel: 2D array-like kernel with odd dimensions >= 3.
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param datatype: Output pixel data type constant (Default = :data:`rsgispy.TYPE_32FLOAT`).
    :param normalise: If ``True``, divide kernel weights by their sum before
                      convolution (raises if sum is ``0``; Default = ``False``).
    :param no_data_val: Optional explicit input nodata value (Default = ``None``).
    :param calc_stats: If ``True``, compute output stats and pyramids (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).

    .. code:: python

        import numpy as np
        import rsgispy.imagefilter

        sharpen = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=float)
        rsgispy.imagefilter.apply_kernel_filter("in.tif", "out_sharp.tif", sharpen)
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    k_arr = np.asarray(kernel, dtype=np.float64)
    if normalise:
        k_sum = float(np.sum(k_arr))
        if k_sum == 0.0:
            raise rsgispy.RSGISPyException(
                "Cannot normalise a convolution kernel whose entries sum to zero."
            )
        k_arr = k_arr / k_sum
    zero_dc = abs(float(np.sum(k_arr))) < 1e-12
    run_kernel_filter(
        input_img=input_img,
        output_img=output_img,
        kernel=k_arr,
        zero_dc=zero_dc,
        gdalformat=gdalformat,
        datatype=datatype,
        no_data_val=no_data_val,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )
