"""
SAR backscatter adaptive speckle filters for raster imagery.
"""

from typing import List, Optional

import rsgispy
from rsgispy.imagefilter._helper import (
    make_lee_stat_fn,
    run_window_stat_filter,
    validate_positive,
)
from rsgispy.imagefilter._sar_helper import (
    make_enhanced_frost_stat_fn,
    make_enhanced_lee_stat_fn,
    make_frost_stat_fn,
    make_gamma_map_stat_fn,
    make_kuan_stat_fn,
    make_refined_lee_stat_fn,
)

__all__: List[str] = [
    "apply_lee_filter",
    "apply_enhanced_lee_filter",
    "apply_frost_filter",
    "apply_enhanced_frost_filter",
    "apply_kuan_filter",
    "apply_gamma_map_filter",
    "apply_refined_lee_filter",
]


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

        import rsgispy.imagefilter.sar

        rsgispy.imagefilter.sar.apply_lee_filter(
            "sar.tif", "sar_lee5.tif", filter_size=5, n_looks=4
        )
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


def apply_enhanced_lee_filter(
    input_img: str,
    output_img: str,
    filter_size: int = 5,
    n_looks: float = 1.0,
    damping: float = 1.0,
    gdalformat: str = "GTIFF",
    datatype: int = rsgispy.TYPE_32FLOAT,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Apply a Lopes et al. (1990) Enhanced Lee SAR adaptive speckle filter.

    Divides pixels into homogeneous (local mean), heterogeneous (exponentially
    damped Lee blend), and point-target (centre pixel preserved) regimes based
    on ``C_u = 1 / sqrt(n_looks)`` and ``C_max = sqrt(1 + 2 / n_looks)``.

    :param input_img: Path to the input SAR raster image.
    :param output_img: Path to the output despeckled image.
    :param filter_size: Odd filter window width/height >= 3 (Default = ``5``).
    :param n_looks: Equivalent number of looks (> 0; Default = ``1.0``).
    :param damping: Exponential damping factor (> 0; Default = ``1.0``).
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param datatype: Output pixel data type constant (Default = :data:`rsgispy.TYPE_32FLOAT`).
    :param no_data_val: Optional explicit input nodata value (Default = ``None``).
    :param calc_stats: If ``True``, compute output stats and pyramids (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).

    .. code:: python

        import rsgispy.imagefilter.sar

        rsgispy.imagefilter.sar.apply_enhanced_lee_filter(
            "sar.tif", "sar_elee5.tif", filter_size=5, n_looks=4, damping=1.0
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    looks = validate_positive(n_looks, "n_looks")
    damp = validate_positive(damping, "damping")
    run_window_stat_filter(
        input_img=input_img,
        output_img=output_img,
        filter_size=filter_size,
        stat_fn=make_enhanced_lee_stat_fn(looks, damping=damp),
        gdalformat=gdalformat,
        datatype=datatype,
        no_data_val=no_data_val,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )


def apply_frost_filter(
    input_img: str,
    output_img: str,
    filter_size: int = 5,
    damping: float = 1.0,
    gdalformat: str = "GTIFF",
    datatype: int = rsgispy.TYPE_32FLOAT,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Apply a Frost et al. (1982) exponentially damped SAR speckle filter.

    Computes spatially weighted window averages using kernel weights
    ``exp(-damping * (var / mean**2) * distance)`` so smooth regions average
    broadly while textured/edge pixels weight closely towards the centre pixel.

    :param input_img: Path to the input SAR raster image.
    :param output_img: Path to the output despeckled image.
    :param filter_size: Odd filter window width/height >= 3 (Default = ``5``).
    :param damping: Exponential damping factor (> 0; Default = ``1.0``).
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param datatype: Output pixel data type constant (Default = :data:`rsgispy.TYPE_32FLOAT`).
    :param no_data_val: Optional explicit input nodata value (Default = ``None``).
    :param calc_stats: If ``True``, compute output stats and pyramids (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).

    .. code:: python

        import rsgispy.imagefilter.sar

        rsgispy.imagefilter.sar.apply_frost_filter(
            "sar.tif", "sar_frost5.tif", filter_size=5, damping=1.0
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    damp = validate_positive(damping, "damping")
    run_window_stat_filter(
        input_img=input_img,
        output_img=output_img,
        filter_size=filter_size,
        stat_fn=make_frost_stat_fn(damping=damp),
        gdalformat=gdalformat,
        datatype=datatype,
        no_data_val=no_data_val,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )


def apply_enhanced_frost_filter(
    input_img: str,
    output_img: str,
    filter_size: int = 5,
    n_looks: float = 1.0,
    damping: float = 1.0,
    gdalformat: str = "GTIFF",
    datatype: int = rsgispy.TYPE_32FLOAT,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Apply a Lopes et al. (1990) Enhanced Frost SAR adaptive speckle filter.

    Combines the three-regime threshold structure of the Enhanced Lee filter
    with exponentially decaying spatial kernel weights in heterogeneous areas.

    :param input_img: Path to the input SAR raster image.
    :param output_img: Path to the output despeckled image.
    :param filter_size: Odd filter window width/height >= 3 (Default = ``5``).
    :param n_looks: Equivalent number of looks (> 0; Default = ``1.0``).
    :param damping: Exponential damping factor (> 0; Default = ``1.0``).
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param datatype: Output pixel data type constant (Default = :data:`rsgispy.TYPE_32FLOAT`).
    :param no_data_val: Optional explicit input nodata value (Default = ``None``).
    :param calc_stats: If ``True``, compute output stats and pyramids (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).

    .. code:: python

        import rsgispy.imagefilter.sar

        rsgispy.imagefilter.sar.apply_enhanced_frost_filter(
            "sar.tif", "sar_efrost5.tif", filter_size=5, n_looks=4, damping=1.0
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    looks = validate_positive(n_looks, "n_looks")
    damp = validate_positive(damping, "damping")
    run_window_stat_filter(
        input_img=input_img,
        output_img=output_img,
        filter_size=filter_size,
        stat_fn=make_enhanced_frost_stat_fn(looks, damping=damp),
        gdalformat=gdalformat,
        datatype=datatype,
        no_data_val=no_data_val,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )


def apply_kuan_filter(
    input_img: str,
    output_img: str,
    filter_size: int = 5,
    n_looks: float = 1.0,
    gdalformat: str = "GTIFF",
    datatype: int = rsgispy.TYPE_32FLOAT,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Apply a Kuan et al. (1985) adaptive MMSE SAR speckle filter.

    Uses the exact multiplicative-to-additive linear MMSE weight
    ``w = (1 - cu**2 / ci**2) / (1 + cu**2)`` with ``cu**2 = 1 / n_looks``.

    :param input_img: Path to the input SAR raster image.
    :param output_img: Path to the output despeckled image.
    :param filter_size: Odd filter window width/height >= 3 (Default = ``5``).
    :param n_looks: Equivalent number of looks (> 0; Default = ``1.0``).
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param datatype: Output pixel data type constant (Default = :data:`rsgispy.TYPE_32FLOAT`).
    :param no_data_val: Optional explicit input nodata value (Default = ``None``).
    :param calc_stats: If ``True``, compute output stats and pyramids (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).

    .. code:: python

        import rsgispy.imagefilter.sar

        rsgispy.imagefilter.sar.apply_kuan_filter(
            "sar.tif", "sar_kuan5.tif", filter_size=5, n_looks=4
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    looks = validate_positive(n_looks, "n_looks")
    run_window_stat_filter(
        input_img=input_img,
        output_img=output_img,
        filter_size=filter_size,
        stat_fn=make_kuan_stat_fn(looks),
        gdalformat=gdalformat,
        datatype=datatype,
        no_data_val=no_data_val,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )


def apply_gamma_map_filter(
    input_img: str,
    output_img: str,
    filter_size: int = 5,
    n_looks: float = 1.0,
    gdalformat: str = "GTIFF",
    datatype: int = rsgispy.TYPE_32FLOAT,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Apply a Lopes et al. (1993) Gamma Maximum A Posteriori (MAP) SAR speckle filter.

    Models radar backscatter intensity as Gamma-distributed and solves the
    closed-form MAP estimator inside heterogeneous regions while returning the
    local mean in homogeneous areas and the centre pixel for strong targets.

    :param input_img: Path to the input SAR raster image.
    :param output_img: Path to the output despeckled image.
    :param filter_size: Odd filter window width/height >= 3 (Default = ``5``).
    :param n_looks: Equivalent number of looks (> 0; Default = ``1.0``).
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param datatype: Output pixel data type constant (Default = :data:`rsgispy.TYPE_32FLOAT`).
    :param no_data_val: Optional explicit input nodata value (Default = ``None``).
    :param calc_stats: If ``True``, compute output stats and pyramids (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).

    .. code:: python

        import rsgispy.imagefilter.sar

        rsgispy.imagefilter.sar.apply_gamma_map_filter(
            "sar.tif", "sar_gmap5.tif", filter_size=5, n_looks=4
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    looks = validate_positive(n_looks, "n_looks")
    run_window_stat_filter(
        input_img=input_img,
        output_img=output_img,
        filter_size=filter_size,
        stat_fn=make_gamma_map_stat_fn(looks),
        gdalformat=gdalformat,
        datatype=datatype,
        no_data_val=no_data_val,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )


def apply_refined_lee_filter(
    input_img: str,
    output_img: str,
    filter_size: int = 7,
    n_looks: float = 1.0,
    gdalformat: str = "GTIFF",
    datatype: int = rsgispy.TYPE_32FLOAT,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Apply a Lee et al. (1999) edge-directed Refined Lee SAR speckle filter.

    Evaluates 8 directional half-window templates around each pixel to identify
    the strongest local edge gradient and applies the Lee MMSE estimator over
    the homogeneous side containing the centre pixel, preserving crisp step edges.

    :param input_img: Path to the input SAR raster image.
    :param output_img: Path to the output despeckled image.
    :param filter_size: Odd filter window width/height >= 3 (Default = ``7``).
    :param n_looks: Equivalent number of looks (> 0; Default = ``1.0``).
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param datatype: Output pixel data type constant (Default = :data:`rsgispy.TYPE_32FLOAT`).
    :param no_data_val: Optional explicit input nodata value (Default = ``None``).
    :param calc_stats: If ``True``, compute output stats and pyramids (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).

    .. code:: python

        import rsgispy.imagefilter.sar

        rsgispy.imagefilter.sar.apply_refined_lee_filter(
            "sar.tif", "sar_reflee7.tif", filter_size=7, n_looks=4
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    looks = validate_positive(n_looks, "n_looks")
    run_window_stat_filter(
        input_img=input_img,
        output_img=output_img,
        filter_size=filter_size,
        stat_fn=make_refined_lee_stat_fn(looks),
        gdalformat=gdalformat,
        datatype=datatype,
        no_data_val=no_data_val,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )
