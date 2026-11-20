"""
Smoothing and custom 2D convolution kernel filters for raster images.
"""

from typing import List, Optional, Sequence, Union

import numpy as np

import rsgispy
from rsgispy.imagefilter._helper import (
    gaussian_kernel,
    run_kernel_filter,
    run_window_stat_filter,
    stat_kuwahara,
)

__all__: List[str] = [
    "apply_kuwahara_filter",
    "apply_gaussian_smooth_filter",
    "apply_kernel_filter",
]


def apply_kuwahara_filter(
    input_img: str,
    output_img: str,
    filter_size: int = 5,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
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
    :param datatype: Output pixel data type constant (Default = ``None``,
                     resolves to :data:`rsgispy.TYPE_32FLOAT`).
    :param no_data_val: Optional explicit input nodata value (Default = ``None``).
    :param calc_stats: If ``True``, compute output stats and pyramids (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).

    .. code:: python

        import rsgispy.imagefilter.smooth

        rsgispy.imagefilter.smooth.apply_kuwahara_filter(
            "in.tif", "out_kuwahara5.tif", filter_size=5
        )
    """
    # pylint: disable=too-many-arguments
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


def apply_gaussian_smooth_filter(
    input_img: str,
    output_img: str,
    filter_size: int = 3,
    stddev_x: float = 1.0,
    stddev_y: float = 1.0,
    filter_angle: float = 0.0,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
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
    :param datatype: Output pixel data type constant (Default = ``None``,
                     resolves to :data:`rsgispy.TYPE_32FLOAT`).
    :param no_data_val: Optional explicit input nodata value (Default = ``None``).
    :param calc_stats: If ``True``, compute output stats and pyramids (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).

    .. code:: python

        import rsgispy.imagefilter.smooth

        rsgispy.imagefilter.smooth.apply_gaussian_smooth_filter(
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


def apply_kernel_filter(
    input_img: str,
    output_img: str,
    kernel: Union[np.ndarray, Sequence[Sequence[float]]],
    normalise: bool = False,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Convolve each band of ``input_img`` with a custom 2D odd-dimension kernel.

    :param input_img: Path to the input raster image.
    :param output_img: Path to the output convolved image.
    :param kernel: 2D array-like kernel with odd dimensions >= 3.
    :param normalise: If ``True``, divide kernel weights by their sum before
                      convolution (raises if sum is ``0``; Default = ``False``).
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param datatype: Output pixel data type constant (Default = ``None``,
                     resolves to :data:`rsgispy.TYPE_32FLOAT`).
    :param no_data_val: Optional explicit input nodata value (Default = ``None``).
    :param calc_stats: If ``True``, compute output stats and pyramids (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).

    .. code:: python

        import numpy as np
        import rsgispy.imagefilter.smooth

        sharpen = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=float)
        rsgispy.imagefilter.smooth.apply_kernel_filter("in.tif", "out_sharp.tif", sharpen)
    """
    # pylint: disable=too-many-arguments
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
