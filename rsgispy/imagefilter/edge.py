"""
Edge-detection and spatial derivative convolution filters for raster images.
"""

from typing import List, Optional

from rsgispy.imagefilter._helper import (
    gaussian_kernel,
    laplacian_kernel,
    prewitt_kernels,
    run_gradient_magnitude_filter,
    run_kernel_filter,
    sobel_kernels,
)

__all__: List[str] = [
    "apply_sobel_filter",
    "apply_sobel_x_filter",
    "apply_sobel_y_filter",
    "apply_prewitt_filter",
    "apply_prewitt_x_filter",
    "apply_prewitt_y_filter",
    "apply_gaussian_1st_deriv_filter",
    "apply_gaussian_2nd_deriv_filter",
    "apply_laplacian_filter",
]


def apply_sobel_filter(
    input_img: str,
    output_img: str,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Apply a 3x3 Sobel edge-magnitude filter (``sqrt(Gx**2 + Gy**2)``).

    :param input_img: Path to the input raster image.
    :param output_img: Path to the output filtered image.
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param datatype: Output pixel data type constant (Default = ``None``,
                     resolves to :data:`rsgispy.TYPE_32FLOAT`).
    :param no_data_val: Optional explicit input nodata value (Default = ``None``).
    :param calc_stats: If ``True``, compute output stats and pyramids (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).

    .. code:: python

        import rsgispy.imagefilter.edge

        rsgispy.imagefilter.edge.apply_sobel_filter("in.tif", "out_sobel.tif")
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
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Apply a horizontal (X-axis) 3x3 Sobel derivative filter.

    :param input_img: Path to the input raster image.
    :param output_img: Path to the output filtered image.
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param datatype: Output pixel data type constant (Default = ``None``,
                     resolves to :data:`rsgispy.TYPE_32FLOAT`).
    :param no_data_val: Optional explicit input nodata value (Default = ``None``).
    :param calc_stats: If ``True``, compute output stats and pyramids (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).

    .. code:: python

        import rsgispy.imagefilter.edge

        rsgispy.imagefilter.edge.apply_sobel_x_filter("in.tif", "out_sobel_x.tif")
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
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Apply a vertical (Y-axis) 3x3 Sobel derivative filter.

    :param input_img: Path to the input raster image.
    :param output_img: Path to the output filtered image.
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param datatype: Output pixel data type constant (Default = ``None``,
                     resolves to :data:`rsgispy.TYPE_32FLOAT`).
    :param no_data_val: Optional explicit input nodata value (Default = ``None``).
    :param calc_stats: If ``True``, compute output stats and pyramids (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).

    .. code:: python

        import rsgispy.imagefilter.edge

        rsgispy.imagefilter.edge.apply_sobel_y_filter("in.tif", "out_sobel_y.tif")
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
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Apply a 3x3 Prewitt edge-magnitude filter (``sqrt(Gx**2 + Gy**2)``).

    :param input_img: Path to the input raster image.
    :param output_img: Path to the output filtered image.
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param datatype: Output pixel data type constant (Default = ``None``,
                     resolves to :data:`rsgispy.TYPE_32FLOAT`).
    :param no_data_val: Optional explicit input nodata value (Default = ``None``).
    :param calc_stats: If ``True``, compute output stats and pyramids (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).

    .. code:: python

        import rsgispy.imagefilter.edge

        rsgispy.imagefilter.edge.apply_prewitt_filter("in.tif", "out_prewitt.tif")
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
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Apply a horizontal (X-axis) 3x3 Prewitt derivative filter.

    :param input_img: Path to the input raster image.
    :param output_img: Path to the output filtered image.
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param datatype: Output pixel data type constant (Default = ``None``,
                     resolves to :data:`rsgispy.TYPE_32FLOAT`).
    :param no_data_val: Optional explicit input nodata value (Default = ``None``).
    :param calc_stats: If ``True``, compute output stats and pyramids (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).

    .. code:: python

        import rsgispy.imagefilter.edge

        rsgispy.imagefilter.edge.apply_prewitt_x_filter("in.tif", "out_prewitt_x.tif")
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
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Apply a vertical (Y-axis) 3x3 Prewitt derivative filter.

    :param input_img: Path to the input raster image.
    :param output_img: Path to the output filtered image.
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param datatype: Output pixel data type constant (Default = ``None``,
                     resolves to :data:`rsgispy.TYPE_32FLOAT`).
    :param no_data_val: Optional explicit input nodata value (Default = ``None``).
    :param calc_stats: If ``True``, compute output stats and pyramids (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).

    .. code:: python

        import rsgispy.imagefilter.edge

        rsgispy.imagefilter.edge.apply_prewitt_y_filter("in.tif", "out_prewitt_y.tif")
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


def apply_gaussian_1st_deriv_filter(
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
    Apply an oriented 2D Gaussian first-derivative filter.

    :param input_img: Path to the input raster image.
    :param output_img: Path to the output filtered image.
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

        import rsgispy.imagefilter.edge

        rsgispy.imagefilter.edge.apply_gaussian_1st_deriv_filter(
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
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
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
    :param datatype: Output pixel data type constant (Default = ``None``,
                     resolves to :data:`rsgispy.TYPE_32FLOAT`).
    :param no_data_val: Optional explicit input nodata value (Default = ``None``).
    :param calc_stats: If ``True``, compute output stats and pyramids (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).

    .. code:: python

        import rsgispy.imagefilter.edge

        rsgispy.imagefilter.edge.apply_gaussian_2nd_deriv_filter(
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
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
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
    :param datatype: Output pixel data type constant (Default = ``None``,
                     resolves to :data:`rsgispy.TYPE_32FLOAT`).
    :param no_data_val: Optional explicit input nodata value (Default = ``None``).
    :param calc_stats: If ``True``, compute output stats and pyramids (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).

    .. code:: python

        import rsgispy.imagefilter.edge

        rsgispy.imagefilter.edge.apply_laplacian_filter(
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
