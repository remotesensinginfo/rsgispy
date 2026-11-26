# pylint: disable=too-many-lines
"""
Statistical, dispersion, and local-difference moving-window raster filters.
"""

from typing import List, Optional

from rsgispy._io import RasterSource, TileNode
from rsgispy.imagefilter._helpers import (
    run_filter_node,
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
    window_stat_node,
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
    "median_filter_node",
    "mean_filter_node",
    "min_filter_node",
    "max_filter_node",
    "mode_filter_node",
    "stddev_filter_node",
    "range_filter_node",
    "total_filter_node",
    "mean_diff_filter_node",
    "mean_diff_abs_filter_node",
    "total_diff_filter_node",
    "total_diff_abs_filter_node",
    "coeff_of_var_filter_node",
    "norm_var_filter_node",
    "norm_var_sqrt_filter_node",
    "norm_var_ln_filter_node",
    "texture_var_filter_node",
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
    src = RasterSource(input_img)
    run_filter_node(
        src,
        median_filter_node(
            src,
            filter_size=filter_size,
        ),
        output_img,
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
    src = RasterSource(input_img)
    run_filter_node(
        src,
        mean_filter_node(
            src,
            filter_size=filter_size,
        ),
        output_img,
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
    src = RasterSource(input_img)
    run_filter_node(
        src,
        min_filter_node(
            src,
            filter_size=filter_size,
        ),
        output_img,
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
    src = RasterSource(input_img)
    run_filter_node(
        src,
        max_filter_node(
            src,
            filter_size=filter_size,
        ),
        output_img,
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
    src = RasterSource(input_img)
    run_filter_node(
        src,
        mode_filter_node(
            src,
            filter_size=filter_size,
        ),
        output_img,
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
    src = RasterSource(input_img)
    run_filter_node(
        src,
        stddev_filter_node(
            src,
            filter_size=filter_size,
        ),
        output_img,
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
    src = RasterSource(input_img)
    run_filter_node(
        src,
        range_filter_node(
            src,
            filter_size=filter_size,
        ),
        output_img,
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
    src = RasterSource(input_img)
    run_filter_node(
        src,
        total_filter_node(
            src,
            filter_size=filter_size,
        ),
        output_img,
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
    src = RasterSource(input_img)
    run_filter_node(
        src,
        mean_diff_filter_node(
            src,
            filter_size=filter_size,
        ),
        output_img,
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
    src = RasterSource(input_img)
    run_filter_node(
        src,
        mean_diff_abs_filter_node(
            src,
            filter_size=filter_size,
        ),
        output_img,
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
    src = RasterSource(input_img)
    run_filter_node(
        src,
        total_diff_filter_node(
            src,
            filter_size=filter_size,
        ),
        output_img,
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
    src = RasterSource(input_img)
    run_filter_node(
        src,
        total_diff_abs_filter_node(
            src,
            filter_size=filter_size,
        ),
        output_img,
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
    src = RasterSource(input_img)
    run_filter_node(
        src,
        coeff_of_var_filter_node(
            src,
            filter_size=filter_size,
        ),
        output_img,
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
    src = RasterSource(input_img)
    run_filter_node(
        src,
        norm_var_filter_node(
            src,
            filter_size=filter_size,
        ),
        output_img,
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
    src = RasterSource(input_img)
    run_filter_node(
        src,
        norm_var_sqrt_filter_node(
            src,
            filter_size=filter_size,
        ),
        output_img,
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
    src = RasterSource(input_img)
    run_filter_node(
        src,
        norm_var_ln_filter_node(
            src,
            filter_size=filter_size,
        ),
        output_img,
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
    src = RasterSource(input_img)
    run_filter_node(
        src,
        texture_var_filter_node(
            src,
            filter_size=filter_size,
        ),
        output_img,
        gdalformat=gdalformat,
        datatype=datatype,
        no_data_val=no_data_val,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )


def median_filter_node(
    source: TileNode,
    filter_size: int = 3,
    *,
    no_data_val: Optional[float] = None,
    name: Optional[str] = None,
) -> TileNode:
    """
    Tile-graph node of :func:`apply_median_filter`: a moving-window median of
    each band of ``source``, for chaining in a
    :class:`~rsgispy.imageprocess.TileGraph` without intermediate files.

    The node outputs ``float64`` blocks with the bands (and band names) of
    ``source``; its halo radius is ``filter_size // 2``.

    :param source: Upstream :class:`~rsgispy.imageprocess.TileNode`.
    :param filter_size: Odd filter window width/height >= 3 (Default = ``3``).
    :param no_data_val: Optional input no-data value: these pixels are
                        ignored, and output pixels without a value are set to
                        it (Default = ``None``; not read from ``source``, so
                        pass the input's no-data value).
    :param name: Optional node name (Default = ``'median_filter'``).
    :return: :class:`~rsgispy.imageprocess.WindowOp` node.

    .. code:: python

        from rsgispy.imageprocess import DiskSink, RasterSource, TileGraph
        from rsgispy.imagefilter.stats import median_filter_node, mean_filter_node

        src = RasterSource("in.tif")
        node = median_filter_node(mean_filter_node(src, 3), filter_size=5)
        TileGraph(DiskSink(node, "out.tif")).run()
    """
    return window_stat_node(
        source,
        filter_size,
        stat_median,
        no_data_val=no_data_val,
        name=name or "median_filter",
    )


def mean_filter_node(
    source: TileNode,
    filter_size: int = 3,
    *,
    no_data_val: Optional[float] = None,
    name: Optional[str] = None,
) -> TileNode:
    """
    Tile-graph node of :func:`apply_mean_filter`: a moving-window arithmetic
    mean of each band of ``source``, for chaining in a
    :class:`~rsgispy.imageprocess.TileGraph` without intermediate files.

    The node outputs ``float64`` blocks with the bands (and band names) of
    ``source``; its halo radius is ``filter_size // 2``.

    :param source: Upstream :class:`~rsgispy.imageprocess.TileNode`.
    :param filter_size: Odd filter window width/height >= 3 (Default = ``3``).
    :param no_data_val: Optional input no-data value: these pixels are
                        ignored, and output pixels without a value are set to
                        it (Default = ``None``; not read from ``source``, so
                        pass the input's no-data value).
    :param name: Optional node name (Default = ``'mean_filter'``).
    :return: :class:`~rsgispy.imageprocess.WindowOp` node.

    .. code:: python

        from rsgispy.imageprocess import DiskSink, RasterSource, TileGraph
        from rsgispy.imagefilter.stats import mean_filter_node, median_filter_node

        src = RasterSource("in.tif")
        node = mean_filter_node(median_filter_node(src, 3), filter_size=5)
        TileGraph(DiskSink(node, "out.tif")).run()
    """
    return window_stat_node(
        source,
        filter_size,
        stat_mean,
        no_data_val=no_data_val,
        name=name or "mean_filter",
    )


def min_filter_node(
    source: TileNode,
    filter_size: int = 3,
    *,
    no_data_val: Optional[float] = None,
    name: Optional[str] = None,
) -> TileNode:
    """
    Tile-graph node of :func:`apply_min_filter`: a moving-window minimum of
    each band of ``source``, for chaining in a
    :class:`~rsgispy.imageprocess.TileGraph` without intermediate files.

    The node outputs ``float64`` blocks with the bands (and band names) of
    ``source``; its halo radius is ``filter_size // 2``.

    :param source: Upstream :class:`~rsgispy.imageprocess.TileNode`.
    :param filter_size: Odd filter window width/height >= 3 (Default = ``3``).
    :param no_data_val: Optional input no-data value: these pixels are
                        ignored, and output pixels without a value are set to
                        it (Default = ``None``; not read from ``source``, so
                        pass the input's no-data value).
    :param name: Optional node name (Default = ``'min_filter'``).
    :return: :class:`~rsgispy.imageprocess.WindowOp` node.

    .. code:: python

        from rsgispy.imageprocess import DiskSink, RasterSource, TileGraph
        from rsgispy.imagefilter.stats import min_filter_node, mean_filter_node

        src = RasterSource("in.tif")
        node = min_filter_node(mean_filter_node(src, 3), filter_size=5)
        TileGraph(DiskSink(node, "out.tif")).run()
    """
    return window_stat_node(
        source,
        filter_size,
        stat_min,
        no_data_val=no_data_val,
        name=name or "min_filter",
    )


def max_filter_node(
    source: TileNode,
    filter_size: int = 3,
    *,
    no_data_val: Optional[float] = None,
    name: Optional[str] = None,
) -> TileNode:
    """
    Tile-graph node of :func:`apply_max_filter`: a moving-window maximum of
    each band of ``source``, for chaining in a
    :class:`~rsgispy.imageprocess.TileGraph` without intermediate files.

    The node outputs ``float64`` blocks with the bands (and band names) of
    ``source``; its halo radius is ``filter_size // 2``.

    :param source: Upstream :class:`~rsgispy.imageprocess.TileNode`.
    :param filter_size: Odd filter window width/height >= 3 (Default = ``3``).
    :param no_data_val: Optional input no-data value: these pixels are
                        ignored, and output pixels without a value are set to
                        it (Default = ``None``; not read from ``source``, so
                        pass the input's no-data value).
    :param name: Optional node name (Default = ``'max_filter'``).
    :return: :class:`~rsgispy.imageprocess.WindowOp` node.

    .. code:: python

        from rsgispy.imageprocess import DiskSink, RasterSource, TileGraph
        from rsgispy.imagefilter.stats import max_filter_node, mean_filter_node

        src = RasterSource("in.tif")
        node = max_filter_node(mean_filter_node(src, 3), filter_size=5)
        TileGraph(DiskSink(node, "out.tif")).run()
    """
    return window_stat_node(
        source,
        filter_size,
        stat_max,
        no_data_val=no_data_val,
        name=name or "max_filter",
    )


def mode_filter_node(
    source: TileNode,
    filter_size: int = 3,
    *,
    no_data_val: Optional[float] = None,
    name: Optional[str] = None,
) -> TileNode:
    """
    Tile-graph node of :func:`apply_mode_filter`: a moving-window mode (most
    frequent value; ties go to the smallest value) of each band of ``source``,
    for chaining in a :class:`~rsgispy.imageprocess.TileGraph` without
    intermediate files.

    The node outputs ``float64`` blocks with the bands (and band names) of
    ``source``; its halo radius is ``filter_size // 2``.

    :param source: Upstream :class:`~rsgispy.imageprocess.TileNode`.
    :param filter_size: Odd filter window width/height >= 3 (Default = ``3``).
    :param no_data_val: Optional input no-data value: these pixels are
                        ignored, and output pixels without a value are set to
                        it (Default = ``None``; not read from ``source``, so
                        pass the input's no-data value).
    :param name: Optional node name (Default = ``'mode_filter'``).
    :return: :class:`~rsgispy.imageprocess.WindowOp` node.

    .. code:: python

        from rsgispy.imageprocess import DiskSink, RasterSource, TileGraph
        from rsgispy.imagefilter.stats import mode_filter_node, mean_filter_node

        src = RasterSource("in.tif")
        node = mode_filter_node(mean_filter_node(src, 3), filter_size=5)
        TileGraph(DiskSink(node, "out.tif")).run()
    """
    return window_stat_node(
        source,
        filter_size,
        stat_mode,
        no_data_val=no_data_val,
        name=name or "mode_filter",
    )


def stddev_filter_node(
    source: TileNode,
    filter_size: int = 3,
    *,
    no_data_val: Optional[float] = None,
    name: Optional[str] = None,
) -> TileNode:
    """
    Tile-graph node of :func:`apply_stddev_filter`: a moving-window standard
    deviation of each band of ``source``, for chaining in a
    :class:`~rsgispy.imageprocess.TileGraph` without intermediate files.

    The node outputs ``float64`` blocks with the bands (and band names) of
    ``source``; its halo radius is ``filter_size // 2``.

    :param source: Upstream :class:`~rsgispy.imageprocess.TileNode`.
    :param filter_size: Odd filter window width/height >= 3 (Default = ``3``).
    :param no_data_val: Optional input no-data value: these pixels are
                        ignored, and output pixels without a value are set to
                        it (Default = ``None``; not read from ``source``, so
                        pass the input's no-data value).
    :param name: Optional node name (Default = ``'stddev_filter'``).
    :return: :class:`~rsgispy.imageprocess.WindowOp` node.

    .. code:: python

        from rsgispy.imageprocess import DiskSink, RasterSource, TileGraph
        from rsgispy.imagefilter.stats import stddev_filter_node, mean_filter_node

        src = RasterSource("in.tif")
        node = stddev_filter_node(mean_filter_node(src, 3), filter_size=5)
        TileGraph(DiskSink(node, "out.tif")).run()
    """
    return window_stat_node(
        source,
        filter_size,
        stat_stddev,
        no_data_val=no_data_val,
        name=name or "stddev_filter",
    )


def range_filter_node(
    source: TileNode,
    filter_size: int = 3,
    *,
    no_data_val: Optional[float] = None,
    name: Optional[str] = None,
) -> TileNode:
    """
    Tile-graph node of :func:`apply_range_filter`: a moving-window range (``max
    - min``) of each band of ``source``, for chaining in a
    :class:`~rsgispy.imageprocess.TileGraph` without intermediate files.

    The node outputs ``float64`` blocks with the bands (and band names) of
    ``source``; its halo radius is ``filter_size // 2``.

    :param source: Upstream :class:`~rsgispy.imageprocess.TileNode`.
    :param filter_size: Odd filter window width/height >= 3 (Default = ``3``).
    :param no_data_val: Optional input no-data value: these pixels are
                        ignored, and output pixels without a value are set to
                        it (Default = ``None``; not read from ``source``, so
                        pass the input's no-data value).
    :param name: Optional node name (Default = ``'range_filter'``).
    :return: :class:`~rsgispy.imageprocess.WindowOp` node.

    .. code:: python

        from rsgispy.imageprocess import DiskSink, RasterSource, TileGraph
        from rsgispy.imagefilter.stats import range_filter_node, mean_filter_node

        src = RasterSource("in.tif")
        node = range_filter_node(mean_filter_node(src, 3), filter_size=5)
        TileGraph(DiskSink(node, "out.tif")).run()
    """
    return window_stat_node(
        source,
        filter_size,
        stat_range,
        no_data_val=no_data_val,
        name=name or "range_filter",
    )


def total_filter_node(
    source: TileNode,
    filter_size: int = 3,
    *,
    no_data_val: Optional[float] = None,
    name: Optional[str] = None,
) -> TileNode:
    """
    Tile-graph node of :func:`apply_total_filter`: a moving-window sum (total)
    of each band of ``source``, for chaining in a
    :class:`~rsgispy.imageprocess.TileGraph` without intermediate files.

    The node outputs ``float64`` blocks with the bands (and band names) of
    ``source``; its halo radius is ``filter_size // 2``.

    :param source: Upstream :class:`~rsgispy.imageprocess.TileNode`.
    :param filter_size: Odd filter window width/height >= 3 (Default = ``3``).
    :param no_data_val: Optional input no-data value: these pixels are
                        ignored, and output pixels without a value are set to
                        it (Default = ``None``; not read from ``source``, so
                        pass the input's no-data value).
    :param name: Optional node name (Default = ``'total_filter'``).
    :return: :class:`~rsgispy.imageprocess.WindowOp` node.

    .. code:: python

        from rsgispy.imageprocess import DiskSink, RasterSource, TileGraph
        from rsgispy.imagefilter.stats import total_filter_node, mean_filter_node

        src = RasterSource("in.tif")
        node = total_filter_node(mean_filter_node(src, 3), filter_size=5)
        TileGraph(DiskSink(node, "out.tif")).run()
    """
    return window_stat_node(
        source,
        filter_size,
        stat_total,
        no_data_val=no_data_val,
        name=name or "total_filter",
    )


def mean_diff_filter_node(
    source: TileNode,
    filter_size: int = 3,
    *,
    no_data_val: Optional[float] = None,
    name: Optional[str] = None,
) -> TileNode:
    """
    Tile-graph node of :func:`apply_mean_diff_filter`: a moving-window mean
    signed difference (``mean(window - centre)``) of each band of ``source``,
    for chaining in a :class:`~rsgispy.imageprocess.TileGraph` without
    intermediate files.

    The node outputs ``float64`` blocks with the bands (and band names) of
    ``source``; its halo radius is ``filter_size // 2``.

    :param source: Upstream :class:`~rsgispy.imageprocess.TileNode`.
    :param filter_size: Odd filter window width/height >= 3 (Default = ``3``).
    :param no_data_val: Optional input no-data value: these pixels are
                        ignored, and output pixels without a value are set to
                        it (Default = ``None``; not read from ``source``, so
                        pass the input's no-data value).
    :param name: Optional node name (Default = ``'mean_diff_filter'``).
    :return: :class:`~rsgispy.imageprocess.WindowOp` node.

    .. code:: python

        from rsgispy.imageprocess import DiskSink, RasterSource, TileGraph
        from rsgispy.imagefilter.stats import mean_diff_filter_node, mean_filter_node

        src = RasterSource("in.tif")
        node = mean_diff_filter_node(mean_filter_node(src, 3), filter_size=5)
        TileGraph(DiskSink(node, "out.tif")).run()
    """
    return window_stat_node(
        source,
        filter_size,
        stat_mean_diff,
        no_data_val=no_data_val,
        name=name or "mean_diff_filter",
    )


def mean_diff_abs_filter_node(
    source: TileNode,
    filter_size: int = 3,
    *,
    no_data_val: Optional[float] = None,
    name: Optional[str] = None,
) -> TileNode:
    """
    Tile-graph node of :func:`apply_mean_diff_abs_filter`: a moving-window mean
    absolute difference (``mean(|window - centre|)``) of each band of
    ``source``, for chaining in a :class:`~rsgispy.imageprocess.TileGraph`
    without intermediate files.

    The node outputs ``float64`` blocks with the bands (and band names) of
    ``source``; its halo radius is ``filter_size // 2``.

    :param source: Upstream :class:`~rsgispy.imageprocess.TileNode`.
    :param filter_size: Odd filter window width/height >= 3 (Default = ``3``).
    :param no_data_val: Optional input no-data value: these pixels are
                        ignored, and output pixels without a value are set to
                        it (Default = ``None``; not read from ``source``, so
                        pass the input's no-data value).
    :param name: Optional node name (Default = ``'mean_diff_abs_filter'``).
    :return: :class:`~rsgispy.imageprocess.WindowOp` node.

    .. code:: python

        from rsgispy.imageprocess import DiskSink, RasterSource, TileGraph
        from rsgispy.imagefilter.stats import mean_diff_abs_filter_node, mean_filter_node

        src = RasterSource("in.tif")
        node = mean_diff_abs_filter_node(mean_filter_node(src, 3), filter_size=5)
        TileGraph(DiskSink(node, "out.tif")).run()
    """
    return window_stat_node(
        source,
        filter_size,
        stat_mean_diff_abs,
        no_data_val=no_data_val,
        name=name or "mean_diff_abs_filter",
    )


def total_diff_filter_node(
    source: TileNode,
    filter_size: int = 3,
    *,
    no_data_val: Optional[float] = None,
    name: Optional[str] = None,
) -> TileNode:
    """
    Tile-graph node of :func:`apply_total_diff_filter`: a moving-window total
    signed difference (``sum(window - centre)``) of each band of ``source``,
    for chaining in a :class:`~rsgispy.imageprocess.TileGraph` without
    intermediate files.

    The node outputs ``float64`` blocks with the bands (and band names) of
    ``source``; its halo radius is ``filter_size // 2``.

    :param source: Upstream :class:`~rsgispy.imageprocess.TileNode`.
    :param filter_size: Odd filter window width/height >= 3 (Default = ``3``).
    :param no_data_val: Optional input no-data value: these pixels are
                        ignored, and output pixels without a value are set to
                        it (Default = ``None``; not read from ``source``, so
                        pass the input's no-data value).
    :param name: Optional node name (Default = ``'total_diff_filter'``).
    :return: :class:`~rsgispy.imageprocess.WindowOp` node.

    .. code:: python

        from rsgispy.imageprocess import DiskSink, RasterSource, TileGraph
        from rsgispy.imagefilter.stats import total_diff_filter_node, mean_filter_node

        src = RasterSource("in.tif")
        node = total_diff_filter_node(mean_filter_node(src, 3), filter_size=5)
        TileGraph(DiskSink(node, "out.tif")).run()
    """
    return window_stat_node(
        source,
        filter_size,
        stat_total_diff,
        no_data_val=no_data_val,
        name=name or "total_diff_filter",
    )


def total_diff_abs_filter_node(
    source: TileNode,
    filter_size: int = 3,
    *,
    no_data_val: Optional[float] = None,
    name: Optional[str] = None,
) -> TileNode:
    """
    Tile-graph node of :func:`apply_total_diff_abs_filter`: a moving-window
    total absolute difference (``sum(|window - centre|)``) of each band of
    ``source``, for chaining in a :class:`~rsgispy.imageprocess.TileGraph`
    without intermediate files.

    The node outputs ``float64`` blocks with the bands (and band names) of
    ``source``; its halo radius is ``filter_size // 2``.

    :param source: Upstream :class:`~rsgispy.imageprocess.TileNode`.
    :param filter_size: Odd filter window width/height >= 3 (Default = ``3``).
    :param no_data_val: Optional input no-data value: these pixels are
                        ignored, and output pixels without a value are set to
                        it (Default = ``None``; not read from ``source``, so
                        pass the input's no-data value).
    :param name: Optional node name (Default = ``'total_diff_abs_filter'``).
    :return: :class:`~rsgispy.imageprocess.WindowOp` node.

    .. code:: python

        from rsgispy.imageprocess import DiskSink, RasterSource, TileGraph
        from rsgispy.imagefilter.stats import total_diff_abs_filter_node, mean_filter_node

        src = RasterSource("in.tif")
        node = total_diff_abs_filter_node(mean_filter_node(src, 3), filter_size=5)
        TileGraph(DiskSink(node, "out.tif")).run()
    """
    return window_stat_node(
        source,
        filter_size,
        stat_total_diff_abs,
        no_data_val=no_data_val,
        name=name or "total_diff_abs_filter",
    )


def coeff_of_var_filter_node(
    source: TileNode,
    filter_size: int = 3,
    *,
    no_data_val: Optional[float] = None,
    name: Optional[str] = None,
) -> TileNode:
    """
    Tile-graph node of :func:`apply_coeff_of_var_filter`: a moving-window
    coefficient of variation (``std / mean``; ``0`` where the mean is ``0``) of
    each band of ``source``, for chaining in a
    :class:`~rsgispy.imageprocess.TileGraph` without intermediate files.

    The node outputs ``float64`` blocks with the bands (and band names) of
    ``source``; its halo radius is ``filter_size // 2``.

    :param source: Upstream :class:`~rsgispy.imageprocess.TileNode`.
    :param filter_size: Odd filter window width/height >= 3 (Default = ``3``).
    :param no_data_val: Optional input no-data value: these pixels are
                        ignored, and output pixels without a value are set to
                        it (Default = ``None``; not read from ``source``, so
                        pass the input's no-data value).
    :param name: Optional node name (Default = ``'coeff_of_var_filter'``).
    :return: :class:`~rsgispy.imageprocess.WindowOp` node.

    .. code:: python

        from rsgispy.imageprocess import DiskSink, RasterSource, TileGraph
        from rsgispy.imagefilter.stats import coeff_of_var_filter_node, mean_filter_node

        src = RasterSource("in.tif")
        node = coeff_of_var_filter_node(mean_filter_node(src, 3), filter_size=5)
        TileGraph(DiskSink(node, "out.tif")).run()
    """
    return window_stat_node(
        source,
        filter_size,
        stat_coeff_of_var,
        no_data_val=no_data_val,
        name=name or "coeff_of_var_filter",
    )


def norm_var_filter_node(
    source: TileNode,
    filter_size: int = 3,
    *,
    no_data_val: Optional[float] = None,
    name: Optional[str] = None,
) -> TileNode:
    """
    Tile-graph node of :func:`apply_norm_var_filter`: a moving-window
    normalised variance (``var / mean**2``) of each band of ``source``, for
    chaining in a :class:`~rsgispy.imageprocess.TileGraph` without intermediate
    files.

    The node outputs ``float64`` blocks with the bands (and band names) of
    ``source``; its halo radius is ``filter_size // 2``.

    :param source: Upstream :class:`~rsgispy.imageprocess.TileNode`.
    :param filter_size: Odd filter window width/height >= 3 (Default = ``3``).
    :param no_data_val: Optional input no-data value: these pixels are
                        ignored, and output pixels without a value are set to
                        it (Default = ``None``; not read from ``source``, so
                        pass the input's no-data value).
    :param name: Optional node name (Default = ``'norm_var_filter'``).
    :return: :class:`~rsgispy.imageprocess.WindowOp` node.

    .. code:: python

        from rsgispy.imageprocess import DiskSink, RasterSource, TileGraph
        from rsgispy.imagefilter.stats import norm_var_filter_node, mean_filter_node

        src = RasterSource("in.tif")
        node = norm_var_filter_node(mean_filter_node(src, 3), filter_size=5)
        TileGraph(DiskSink(node, "out.tif")).run()
    """
    return window_stat_node(
        source,
        filter_size,
        stat_norm_var,
        no_data_val=no_data_val,
        name=name or "norm_var_filter",
    )


def norm_var_sqrt_filter_node(
    source: TileNode,
    filter_size: int = 3,
    *,
    no_data_val: Optional[float] = None,
    name: Optional[str] = None,
) -> TileNode:
    """
    Tile-graph node of :func:`apply_norm_var_sqrt_filter`: a moving-window
    square-root normalised variance (``sqrt(var / mean**2)``) of each band of
    ``source``, for chaining in a :class:`~rsgispy.imageprocess.TileGraph`
    without intermediate files.

    The node outputs ``float64`` blocks with the bands (and band names) of
    ``source``; its halo radius is ``filter_size // 2``.

    :param source: Upstream :class:`~rsgispy.imageprocess.TileNode`.
    :param filter_size: Odd filter window width/height >= 3 (Default = ``3``).
    :param no_data_val: Optional input no-data value: these pixels are
                        ignored, and output pixels without a value are set to
                        it (Default = ``None``; not read from ``source``, so
                        pass the input's no-data value).
    :param name: Optional node name (Default = ``'norm_var_sqrt_filter'``).
    :return: :class:`~rsgispy.imageprocess.WindowOp` node.

    .. code:: python

        from rsgispy.imageprocess import DiskSink, RasterSource, TileGraph
        from rsgispy.imagefilter.stats import norm_var_sqrt_filter_node, mean_filter_node

        src = RasterSource("in.tif")
        node = norm_var_sqrt_filter_node(mean_filter_node(src, 3), filter_size=5)
        TileGraph(DiskSink(node, "out.tif")).run()
    """
    return window_stat_node(
        source,
        filter_size,
        stat_norm_var_sqrt,
        no_data_val=no_data_val,
        name=name or "norm_var_sqrt_filter",
    )


def norm_var_ln_filter_node(
    source: TileNode,
    filter_size: int = 3,
    *,
    no_data_val: Optional[float] = None,
    name: Optional[str] = None,
) -> TileNode:
    """
    Tile-graph node of :func:`apply_norm_var_ln_filter`: a moving-window
    natural-log normalised variance (``ln(1 + var / mean**2)``) of each band of
    ``source``, for chaining in a :class:`~rsgispy.imageprocess.TileGraph`
    without intermediate files.

    The node outputs ``float64`` blocks with the bands (and band names) of
    ``source``; its halo radius is ``filter_size // 2``.

    :param source: Upstream :class:`~rsgispy.imageprocess.TileNode`.
    :param filter_size: Odd filter window width/height >= 3 (Default = ``3``).
    :param no_data_val: Optional input no-data value: these pixels are
                        ignored, and output pixels without a value are set to
                        it (Default = ``None``; not read from ``source``, so
                        pass the input's no-data value).
    :param name: Optional node name (Default = ``'norm_var_ln_filter'``).
    :return: :class:`~rsgispy.imageprocess.WindowOp` node.

    .. code:: python

        from rsgispy.imageprocess import DiskSink, RasterSource, TileGraph
        from rsgispy.imagefilter.stats import norm_var_ln_filter_node, mean_filter_node

        src = RasterSource("in.tif")
        node = norm_var_ln_filter_node(mean_filter_node(src, 3), filter_size=5)
        TileGraph(DiskSink(node, "out.tif")).run()
    """
    return window_stat_node(
        source,
        filter_size,
        stat_norm_var_ln,
        no_data_val=no_data_val,
        name=name or "norm_var_ln_filter",
    )


def texture_var_filter_node(
    source: TileNode,
    filter_size: int = 3,
    *,
    no_data_val: Optional[float] = None,
    name: Optional[str] = None,
) -> TileNode:
    """
    Tile-graph node of :func:`apply_texture_var_filter`: a moving-window local
    texture variance of each band of ``source``, for chaining in a
    :class:`~rsgispy.imageprocess.TileGraph` without intermediate files.

    The node outputs ``float64`` blocks with the bands (and band names) of
    ``source``; its halo radius is ``filter_size // 2``.

    :param source: Upstream :class:`~rsgispy.imageprocess.TileNode`.
    :param filter_size: Odd filter window width/height >= 3 (Default = ``3``).
    :param no_data_val: Optional input no-data value: these pixels are
                        ignored, and output pixels without a value are set to
                        it (Default = ``None``; not read from ``source``, so
                        pass the input's no-data value).
    :param name: Optional node name (Default = ``'texture_var_filter'``).
    :return: :class:`~rsgispy.imageprocess.WindowOp` node.

    .. code:: python

        from rsgispy.imageprocess import DiskSink, RasterSource, TileGraph
        from rsgispy.imagefilter.stats import texture_var_filter_node, mean_filter_node

        src = RasterSource("in.tif")
        node = texture_var_filter_node(mean_filter_node(src, 3), filter_size=5)
        TileGraph(DiskSink(node, "out.tif")).run()
    """
    return window_stat_node(
        source,
        filter_size,
        stat_texture_var,
        no_data_val=no_data_val,
        name=name or "texture_var_filter",
    )
