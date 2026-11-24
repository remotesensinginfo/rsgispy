"""
Statistical, dispersion, and local-difference moving-window raster filters.
"""

from typing import List, Optional

from rsgispy.imagefilter._helpers import (
    run_window_stat_filter,
    stat_coeff_of_var,
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
)

__all__: List[str] = [
    "apply_median_filter",
    "apply_mean_filter",
    "apply_min_filter",
    "apply_max_filter",
    "apply_mode_filter",
    "apply_stddev_filter",
    "apply_range_filter",
    "apply_total_filter",
    "apply_mean_diff_filter",
    "apply_mean_diff_abs_filter",
    "apply_total_diff_filter",
    "apply_total_diff_abs_filter",
    "apply_coeff_of_var_filter",
    "apply_norm_var_filter",
    "apply_norm_var_sqrt_filter",
    "apply_norm_var_ln_filter",
    "apply_texture_var_filter",
]


def apply_median_filter(
    input_img: str,
    output_img: str,
    filter_size: int = 3,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
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
    :param datatype: Output pixel data type constant (Default = ``None``,
                     resolves to :data:`rsgispy.TYPE_32FLOAT`).
    :param no_data_val: Optional explicit input nodata value (Default = ``None``).
    :param calc_stats: If ``True``, compute output stats and pyramids (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).

    .. code:: python

        import rsgispy
        import rsgispy.imagefilter.stats

        rsgispy.imagefilter.stats.apply_median_filter(
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
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
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
    :param datatype: Output pixel data type constant (Default = ``None``,
                     resolves to :data:`rsgispy.TYPE_32FLOAT`).
    :param no_data_val: Optional explicit input nodata value (Default = ``None``).
    :param calc_stats: If ``True``, compute output stats and pyramids (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).

    .. code:: python

        import rsgispy.imagefilter.stats

        rsgispy.imagefilter.stats.apply_mean_filter("in.tif", "out_mean3.tif", filter_size=3)
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
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
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
    :param datatype: Output pixel data type constant (Default = ``None``,
                     resolves to :data:`rsgispy.TYPE_32FLOAT`).
    :param no_data_val: Optional explicit input nodata value (Default = ``None``).
    :param calc_stats: If ``True``, compute output stats and pyramids (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).

    .. code:: python

        import rsgispy.imagefilter.stats

        rsgispy.imagefilter.stats.apply_min_filter("in.tif", "out_min3.tif", filter_size=3)
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
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
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
    :param datatype: Output pixel data type constant (Default = ``None``,
                     resolves to :data:`rsgispy.TYPE_32FLOAT`).
    :param no_data_val: Optional explicit input nodata value (Default = ``None``).
    :param calc_stats: If ``True``, compute output stats and pyramids (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).

    .. code:: python

        import rsgispy.imagefilter.stats

        rsgispy.imagefilter.stats.apply_max_filter("in.tif", "out_max3.tif", filter_size=3)
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
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
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
    :param datatype: Output pixel data type constant (Default = ``None``,
                     resolves to :data:`rsgispy.TYPE_32FLOAT`).
    :param no_data_val: Optional explicit input nodata value (Default = ``None``).
    :param calc_stats: If ``True``, compute output stats and pyramids (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).

    .. code:: python

        import rsgispy
        import rsgispy.imagefilter.stats

        rsgispy.imagefilter.stats.apply_mode_filter(
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
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
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
    :param datatype: Output pixel data type constant (Default = ``None``,
                     resolves to :data:`rsgispy.TYPE_32FLOAT`).
    :param no_data_val: Optional explicit input nodata value (Default = ``None``).
    :param calc_stats: If ``True``, compute output stats and pyramids (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).

    .. code:: python

        import rsgispy.imagefilter.stats

        rsgispy.imagefilter.stats.apply_stddev_filter("in.tif", "out_std3.tif", filter_size=3)
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
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
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
    :param datatype: Output pixel data type constant (Default = ``None``,
                     resolves to :data:`rsgispy.TYPE_32FLOAT`).
    :param no_data_val: Optional explicit input nodata value (Default = ``None``).
    :param calc_stats: If ``True``, compute output stats and pyramids (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).

    .. code:: python

        import rsgispy.imagefilter.stats

        rsgispy.imagefilter.stats.apply_range_filter("in.tif", "out_range3.tif", filter_size=3)
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
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
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
    :param datatype: Output pixel data type constant (Default = ``None``,
                     resolves to :data:`rsgispy.TYPE_32FLOAT`).
    :param no_data_val: Optional explicit input nodata value (Default = ``None``).
    :param calc_stats: If ``True``, compute output stats and pyramids (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).

    .. code:: python

        import rsgispy.imagefilter.stats

        rsgispy.imagefilter.stats.apply_total_filter("in.tif", "out_total3.tif", filter_size=3)
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
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
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
    :param datatype: Output pixel data type constant (Default = ``None``,
                     resolves to :data:`rsgispy.TYPE_32FLOAT`).
    :param no_data_val: Optional explicit input nodata value (Default = ``None``).
    :param calc_stats: If ``True``, compute output stats and pyramids (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).

    .. code:: python

        import rsgispy.imagefilter.stats

        rsgispy.imagefilter.stats.apply_mean_diff_filter(
            "in.tif", "out_meandiff3.tif", filter_size=3
        )
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
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
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
    :param datatype: Output pixel data type constant (Default = ``None``,
                     resolves to :data:`rsgispy.TYPE_32FLOAT`).
    :param no_data_val: Optional explicit input nodata value (Default = ``None``).
    :param calc_stats: If ``True``, compute output stats and pyramids (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).

    .. code:: python

        import rsgispy.imagefilter.stats

        rsgispy.imagefilter.stats.apply_mean_diff_abs_filter(
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
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
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
    :param datatype: Output pixel data type constant (Default = ``None``,
                     resolves to :data:`rsgispy.TYPE_32FLOAT`).
    :param no_data_val: Optional explicit input nodata value (Default = ``None``).
    :param calc_stats: If ``True``, compute output stats and pyramids (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).

    .. code:: python

        import rsgispy.imagefilter.stats

        rsgispy.imagefilter.stats.apply_total_diff_filter(
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
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
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
    :param datatype: Output pixel data type constant (Default = ``None``,
                     resolves to :data:`rsgispy.TYPE_32FLOAT`).
    :param no_data_val: Optional explicit input nodata value (Default = ``None``).
    :param calc_stats: If ``True``, compute output stats and pyramids (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).

    .. code:: python

        import rsgispy.imagefilter.stats

        rsgispy.imagefilter.stats.apply_total_diff_abs_filter(
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
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
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
    :param datatype: Output pixel data type constant (Default = ``None``,
                     resolves to :data:`rsgispy.TYPE_32FLOAT`).
    :param no_data_val: Optional explicit input nodata value (Default = ``None``).
    :param calc_stats: If ``True``, compute output stats and pyramids (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).

    .. code:: python

        import rsgispy.imagefilter.stats

        rsgispy.imagefilter.stats.apply_coeff_of_var_filter(
            "in.tif", "out_cv3.tif", filter_size=3
        )
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
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
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
    :param datatype: Output pixel data type constant (Default = ``None``,
                     resolves to :data:`rsgispy.TYPE_32FLOAT`).
    :param no_data_val: Optional explicit input nodata value (Default = ``None``).
    :param calc_stats: If ``True``, compute output stats and pyramids (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).

    .. code:: python

        import rsgispy.imagefilter.stats

        rsgispy.imagefilter.stats.apply_norm_var_filter(
            "in.tif", "out_normvar3.tif", filter_size=3
        )
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
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
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
    :param datatype: Output pixel data type constant (Default = ``None``,
                     resolves to :data:`rsgispy.TYPE_32FLOAT`).
    :param no_data_val: Optional explicit input nodata value (Default = ``None``).
    :param calc_stats: If ``True``, compute output stats and pyramids (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).

    .. code:: python

        import rsgispy.imagefilter.stats

        rsgispy.imagefilter.stats.apply_norm_var_sqrt_filter(
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
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
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
    :param datatype: Output pixel data type constant (Default = ``None``,
                     resolves to :data:`rsgispy.TYPE_32FLOAT`).
    :param no_data_val: Optional explicit input nodata value (Default = ``None``).
    :param calc_stats: If ``True``, compute output stats and pyramids (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).

    .. code:: python

        import rsgispy.imagefilter.stats

        rsgispy.imagefilter.stats.apply_norm_var_ln_filter(
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
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
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
    :param datatype: Output pixel data type constant (Default = ``None``,
                     resolves to :data:`rsgispy.TYPE_32FLOAT`).
    :param no_data_val: Optional explicit input nodata value (Default = ``None``).
    :param calc_stats: If ``True``, compute output stats and pyramids (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).

    .. code:: python

        import rsgispy.imagefilter.stats

        rsgispy.imagefilter.stats.apply_texture_var_filter(
            "in.tif", "out_var3.tif", filter_size=3
        )
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
