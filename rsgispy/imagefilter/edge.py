"""
Edge-detection and spatial derivative convolution filters for raster images.
"""

from typing import List, Optional

from rsgispy._io import TileNode
from rsgispy.imagefilter._helpers import (
    gaussian_kernel,
    gradient_magnitude_node,
    kernel_conv_node,
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
    "sobel_filter_node",
    "prewitt_filter_node",
    "laplacian_filter_node",
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


def sobel_filter_node(
    source: TileNode,
    *,
    no_data_val: Optional[float] = None,
    name: Optional[str] = None,
) -> TileNode:
    """
    Tile-graph node of :func:`apply_sobel_filter`: the 3x3 Sobel edge
    magnitude (``sqrt(Gx**2 + Gy**2)``) of each band of ``source``, for
    chaining in a :class:`~rsgispy.imageprocess.TileGraph` without
    intermediate files.

    The node outputs ``float64`` blocks with the bands (and band names) of
    ``source``; its halo radius is 1.

    :param source: Upstream :class:`~rsgispy.imageprocess.TileNode`.
    :param no_data_val: Optional input no-data value: these pixels are
                        ignored, and output pixels without a value are set to
                        it (Default = ``None``; not read from ``source``, so
                        pass the input's no-data value).
    :param name: Optional node name (Default = ``'sobel_filter'``).
    :return: :class:`~rsgispy.imageprocess.WindowOp` node.

    .. code:: python

        from rsgispy.imageprocess import DiskSink, RasterSource, TileGraph
        from rsgispy.imagefilter.edge import sobel_filter_node
        from rsgispy.imagefilter.stats import median_filter_node

        src = RasterSource("in.tif")
        edges = sobel_filter_node(median_filter_node(src, 3))
        TileGraph(DiskSink(edges, "median_sobel.tif")).run()
    """
    kx, ky = sobel_kernels()
    return gradient_magnitude_node(
        source,
        kx,
        ky,
        no_data_val=no_data_val,
        name=name or "sobel_filter",
    )


def prewitt_filter_node(
    source: TileNode,
    *,
    no_data_val: Optional[float] = None,
    name: Optional[str] = None,
) -> TileNode:
    """
    Tile-graph node of :func:`apply_prewitt_filter`: the 3x3 Prewitt edge
    magnitude (``sqrt(Gx**2 + Gy**2)``) of each band of ``source``, for
    chaining in a :class:`~rsgispy.imageprocess.TileGraph` without
    intermediate files.

    The node outputs ``float64`` blocks with the bands (and band names) of
    ``source``; its halo radius is 1.

    :param source: Upstream :class:`~rsgispy.imageprocess.TileNode`.
    :param no_data_val: Optional input no-data value: these pixels are
                        ignored, and output pixels without a value are set to
                        it (Default = ``None``; not read from ``source``, so
                        pass the input's no-data value).
    :param name: Optional node name (Default = ``'prewitt_filter'``).
    :return: :class:`~rsgispy.imageprocess.WindowOp` node.

    .. code:: python

        from rsgispy.imageprocess import DiskSink, RasterSource, TileGraph
        from rsgispy.imagefilter.edge import prewitt_filter_node
        from rsgispy.imagefilter.stats import median_filter_node

        src = RasterSource("in.tif")
        edges = prewitt_filter_node(median_filter_node(src, 3))
        TileGraph(DiskSink(edges, "median_prewitt.tif")).run()
    """
    kx, ky = prewitt_kernels()
    return gradient_magnitude_node(
        source,
        kx,
        ky,
        no_data_val=no_data_val,
        name=name or "prewitt_filter",
    )


def laplacian_filter_node(
    source: TileNode,
    filter_size: int = 3,
    stddev: float = 1.0,
    *,
    no_data_val: Optional[float] = None,
    name: Optional[str] = None,
) -> TileNode:
    """
    Tile-graph node of :func:`apply_laplacian_filter`: a zero-sum
    Laplacian-of-Gaussian (LoG) edge/blob filter of each band of ``source``,
    for chaining in a :class:`~rsgispy.imageprocess.TileGraph` without
    intermediate files.

    The node outputs ``float64`` blocks with the bands (and band names) of
    ``source``; its halo radius is half the kernel size.

    :param source: Upstream :class:`~rsgispy.imageprocess.TileNode`.
    :param filter_size: Odd kernel width/height >= 3 (Default = ``3``).
    :param stddev: Gaussian standard deviation > 0 (Default = ``1.0``).
    :param no_data_val: Optional input no-data value: these pixels are
                        ignored, and output pixels without a value are set to
                        it (Default = ``None``; not read from ``source``, so
                        pass the input's no-data value).
    :param name: Optional node name (Default = ``'laplacian_filter'``).
    :return: :class:`~rsgispy.imageprocess.WindowOp` node.

    .. code:: python

        from rsgispy.imageprocess import DiskSink, RasterSource, TileGraph
        from rsgispy.imagefilter.edge import laplacian_filter_node
        from rsgispy.imagefilter.smooth import gaussian_smooth_filter_node

        src = RasterSource("in.tif")
        log = laplacian_filter_node(gaussian_smooth_filter_node(src, 5), 5, stddev=1.0)
        TileGraph(DiskSink(log, "smooth_log.tif")).run()
    """
    kernel = laplacian_kernel(filter_size, stddev)
    return kernel_conv_node(
        source,
        kernel,
        zero_dc=True,
        no_data_val=no_data_val,
        name=name or "laplacian_filter",
    )
