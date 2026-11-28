"""
Smoothing and custom 2D convolution kernel filters for raster images.

Convolution and correlation
---------------------------

:func:`apply_kernel_filter` (and :func:`kernel_filter_node`) is a true 2D
convolution: the kernel is flipped in both axes before it is laid over each
window, ``out[r, c] = sum(k[i, j] * in[r - i + kh // 2, c - j + kw // 2])``.
:func:`apply_kernel_correlation_filter` (and
:func:`kernel_correlation_filter_node`) is a correlation: the kernel is laid
over the window as written, ``out[r, c] = sum(k[i, j] * in[r + i - kh // 2,
c + j - kw // 2])`` (what :func:`apply_kernel_filter` did up to rsgispy
0.4.1). The two give the same result for kernels that are symmetric under a
180 degree rotation (e.g. smoothing, sharpening and Laplacian kernels); for
an asymmetric kernel the result is mirrored: ``[[0, 0, 0], [0, 0, 1], [0, 0,
0]]`` moves the image one pixel right under convolution and one pixel left
under correlation.
"""

from typing import List, Optional, Sequence, Union

import numpy as np

import rsgispy
from rsgispy._io import RasterSource, TileNode
from rsgispy.imagefilter._helpers import (
    gaussian_kernel,
    kernel_conv_node,
    kernel_correlate_node,
    run_filter_node,
    stat_kuwahara,
    window_stat_node,
)

__all__: List[str] = [
    "apply_kuwahara_filter",
    "apply_gaussian_smooth_filter",
    "apply_kernel_filter",
    "apply_kernel_correlation_filter",
    "kuwahara_filter_node",
    "gaussian_smooth_filter_node",
    "kernel_filter_node",
    "kernel_correlation_filter_node",
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
    src = RasterSource(input_img)
    run_filter_node(
        src,
        kuwahara_filter_node(
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
    src = RasterSource(input_img)
    run_filter_node(
        src,
        gaussian_smooth_filter_node(
            src,
            filter_size=filter_size,
            stddev_x=stddev_x,
            stddev_y=stddev_y,
            filter_angle=filter_angle,
        ),
        output_img,
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

    This is a true convolution: the kernel is flipped in both axes before it
    is applied (see the module notes on convolution and correlation); use
    :func:`apply_kernel_correlation_filter` to apply it unflipped. Kernels
    symmetric under a 180 degree rotation give the same result with both.

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
    src = RasterSource(input_img)
    run_filter_node(
        src,
        kernel_filter_node(
            src,
            kernel=kernel,
            normalise=normalise,
        ),
        output_img,
        gdalformat=gdalformat,
        datatype=datatype,
        no_data_val=no_data_val,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )


def apply_kernel_correlation_filter(
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
    Correlate each band of ``input_img`` with a custom 2D odd-dimension
    kernel: the kernel is laid over each window as written, without the flip
    of a convolution, ``out[r, c] = sum(k[i, j] * in[r + i - kh // 2,
    c + j - kw // 2])`` (the behaviour of :func:`apply_kernel_filter` up to
    rsgispy 0.4.1). For a kernel symmetric under a 180 degree rotation this
    equals :func:`apply_kernel_filter`; an asymmetric kernel gives the
    mirrored result (see the module notes on convolution and correlation).

    :param input_img: Path to the input raster image.
    :param output_img: Path to the output correlated image.
    :param kernel: 2D array-like kernel with odd dimensions >= 3.
    :param normalise: If ``True``, divide kernel weights by their sum before
                      correlation (raises if sum is ``0``; Default = ``False``).
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param datatype: Output pixel data type constant (Default = ``None``,
                     resolves to :data:`rsgispy.TYPE_32FLOAT`).
    :param no_data_val: Optional explicit input nodata value (Default = ``None``).
    :param calc_stats: If ``True``, compute output stats and pyramids (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).

    .. code:: python

        import numpy as np
        import rsgispy.imagefilter.smooth

        # Template-style kernel: out[r, c] = in[r, c + 1] (one pixel left).
        shift = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
        rsgispy.imagefilter.smooth.apply_kernel_correlation_filter(
            "in.tif", "out_shift.tif", shift
        )
    """
    # pylint: disable=too-many-arguments
    src = RasterSource(input_img)
    run_filter_node(
        src,
        kernel_correlation_filter_node(
            src,
            kernel=kernel,
            normalise=normalise,
        ),
        output_img,
        gdalformat=gdalformat,
        datatype=datatype,
        no_data_val=no_data_val,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )


def kuwahara_filter_node(
    source: TileNode,
    filter_size: int = 5,
    *,
    no_data_val: Optional[float] = None,
    name: Optional[str] = None,
) -> TileNode:
    """
    Tile-graph node of :func:`apply_kuwahara_filter`: an edge-preserving
    4-quadrant Kuwahara smoothing of each band of ``source``, for chaining in
    a :class:`~rsgispy.imageprocess.TileGraph` without intermediate files.

    The node outputs ``float64`` blocks with the bands (and band names) of
    ``source``; its halo radius is half the window size.

    :param source: Upstream :class:`~rsgispy.imageprocess.TileNode`.
    :param filter_size: Odd filter window width/height >= 3 (Default = ``5``).
    :param no_data_val: Optional input no-data value: these pixels are
                        ignored, and output pixels without a value are set to
                        it (Default = ``None``: the no-data value of
                        ``source``'s metadata; see :ref:`tilegraph-nodata`).
    :param name: Optional node name (Default = ``'kuwahara_filter'``).
    :return: :class:`~rsgispy.imageprocess.WindowOp` node.

    .. code:: python

        from rsgispy.imageprocess import DiskSink, RasterSource, TileGraph
        from rsgispy.imagefilter.edge import sobel_filter_node
        from rsgispy.imagefilter.smooth import kuwahara_filter_node

        src = RasterSource("in.tif")
        edges = sobel_filter_node(kuwahara_filter_node(src, filter_size=5))
        TileGraph(DiskSink(edges, "kuwahara_sobel.tif")).run()
    """
    return window_stat_node(
        source,
        filter_size,
        stat_kuwahara,
        no_data_val=no_data_val,
        name=name or "kuwahara_filter",
    )


def gaussian_smooth_filter_node(
    source: TileNode,
    filter_size: int = 3,
    stddev_x: float = 1.0,
    stddev_y: float = 1.0,
    filter_angle: float = 0.0,
    *,
    no_data_val: Optional[float] = None,
    name: Optional[str] = None,
) -> TileNode:
    """
    Tile-graph node of :func:`apply_gaussian_smooth_filter`: an oriented 2D
    anisotropic Gaussian smoothing of each band of ``source``, for chaining
    in a :class:`~rsgispy.imageprocess.TileGraph` without intermediate files.

    The node outputs ``float64`` blocks with the bands (and band names) of
    ``source``; its halo radius is half the kernel size.

    :param source: Upstream :class:`~rsgispy.imageprocess.TileNode`.
    :param filter_size: Odd kernel width/height >= 3 (Default = ``3``).
    :param stddev_x: Standard deviation along the primary rotated axis (Default = ``1.0``).
    :param stddev_y: Standard deviation along the orthogonal axis (Default = ``1.0``).
    :param filter_angle: Orientation angle in degrees (Default = ``0.0``).
    :param no_data_val: Optional input no-data value: these pixels are
                        ignored, and output pixels without a value are set to
                        it (Default = ``None``: the no-data value of
                        ``source``'s metadata; see :ref:`tilegraph-nodata`).
    :param name: Optional node name (Default = ``'gaussian_smooth_filter'``).
    :return: :class:`~rsgispy.imageprocess.WindowOp` node.

    .. code:: python

        from rsgispy.imageprocess import DiskSink, RasterSource, TileGraph
        from rsgispy.imagefilter.edge import laplacian_filter_node
        from rsgispy.imagefilter.smooth import gaussian_smooth_filter_node

        src = RasterSource("dem.tif")
        smooth = gaussian_smooth_filter_node(src, 5, stddev_x=1.2, stddev_y=1.2)
        TileGraph(DiskSink(laplacian_filter_node(smooth, 5), "dem_log.tif")).run()
    """
    # pylint: disable=too-many-arguments
    kernel = gaussian_kernel(
        filter_size, stddev_x, stddev_y, filter_angle, deriv_order=0
    )
    return kernel_conv_node(
        source,
        kernel,
        zero_dc=False,
        no_data_val=no_data_val,
        name=name or "gaussian_smooth_filter",
    )


def kernel_filter_node(
    source: TileNode,
    kernel: Union[np.ndarray, Sequence[Sequence[float]]],
    normalise: bool = False,
    *,
    zero_dc: Optional[bool] = None,
    no_data_val: Optional[float] = None,
    name: Optional[str] = None,
) -> TileNode:
    """
    Tile-graph node of :func:`apply_kernel_filter`: convolves each band of
    ``source`` with a custom 2D odd-dimension kernel (a true convolution:
    the kernel is flipped in both axes; see
    :func:`kernel_correlation_filter_node` for the unflipped correlation),
    for chaining in a :class:`~rsgispy.imageprocess.TileGraph` without
    intermediate files.

    The node outputs ``float64`` blocks with the bands (and band names) of
    ``source``; its halo radius is half the kernel size.

    :param source: Upstream :class:`~rsgispy.imageprocess.TileNode`.
    :param kernel: 2D array-like kernel with odd dimensions >= 3.
    :param normalise: If ``True``, divide kernel weights by their sum before
                      convolution (raises if sum is ``0``; Default = ``False``).
    :param zero_dc: How pixels next to no data are handled: ``True`` treats
                    the kernel as zero-sum (edge detector; invalid neighbours
                    take the centre value), ``False`` re-weights the valid
                    neighbours of a non-negative kernel. ``None`` (default)
                    decides from the kernel sum, as :func:`apply_kernel_filter`
                    does.
    :param no_data_val: Optional input no-data value: these pixels are
                        ignored, and output pixels without a value are set to
                        it (Default = ``None``: the no-data value of
                        ``source``'s metadata; see :ref:`tilegraph-nodata`).
    :param name: Optional node name (Default = ``'kernel_filter'``).
    :return: :class:`~rsgispy.imageprocess.WindowOp` node.

    .. code:: python

        import numpy as np
        from rsgispy.imageprocess import DiskSink, RasterSource, TileGraph
        from rsgispy.imagefilter.smooth import kernel_filter_node
        from rsgispy.imagefilter.stats import median_filter_node

        sharpen = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=float)
        src = RasterSource("in.tif")
        node = kernel_filter_node(median_filter_node(src, 3), sharpen)
        TileGraph(DiskSink(node, "median_sharpen.tif")).run()
    """
    # pylint: disable=too-many-arguments
    return _custom_kernel_node(
        source,
        kernel,
        normalise,
        zero_dc,
        no_data_val=no_data_val,
        name=name or "kernel_filter",
        correlate=False,
    )


def kernel_correlation_filter_node(
    source: TileNode,
    kernel: Union[np.ndarray, Sequence[Sequence[float]]],
    normalise: bool = False,
    *,
    zero_dc: Optional[bool] = None,
    no_data_val: Optional[float] = None,
    name: Optional[str] = None,
) -> TileNode:
    """
    Tile-graph node of :func:`apply_kernel_correlation_filter`: correlates
    each band of ``source`` with a custom 2D odd-dimension kernel (applied
    as written, without the flip of :func:`kernel_filter_node`), for
    chaining in a :class:`~rsgispy.imageprocess.TileGraph` without
    intermediate files.

    The node outputs ``float64`` blocks with the bands (and band names) of
    ``source``; its halo radius is half the kernel size.

    :param source: Upstream :class:`~rsgispy.imageprocess.TileNode`.
    :param kernel: 2D array-like kernel with odd dimensions >= 3.
    :param normalise: If ``True``, divide kernel weights by their sum before
                      correlation (raises if sum is ``0``; Default = ``False``).
    :param zero_dc: How pixels next to no data are handled: ``True`` treats
                    the kernel as zero-sum (edge detector; invalid neighbours
                    take the centre value), ``False`` re-weights the valid
                    neighbours of a non-negative kernel. ``None`` (default)
                    decides from the kernel sum, as
                    :func:`apply_kernel_correlation_filter` does.
    :param no_data_val: Optional input no-data value: these pixels are
                        ignored, and output pixels without a value are set to
                        it (Default = ``None``: the no-data value of
                        ``source``'s metadata; see :ref:`tilegraph-nodata`).
    :param name: Optional node name (Default = ``'kernel_correlation_filter'``).
    :return: :class:`~rsgispy.imageprocess.WindowOp` node.

    .. code:: python

        import numpy as np
        from rsgispy.imageprocess import DiskSink, RasterSource, TileGraph
        from rsgispy.imagefilter.smooth import kernel_correlation_filter_node
        from rsgispy.imagefilter.stats import median_filter_node

        shift = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
        src = RasterSource("in.tif")
        node = kernel_correlation_filter_node(median_filter_node(src, 3), shift)
        TileGraph(DiskSink(node, "median_shift_left.tif")).run()
    """
    # pylint: disable=too-many-arguments
    return _custom_kernel_node(
        source,
        kernel,
        normalise,
        zero_dc,
        no_data_val=no_data_val,
        name=name or "kernel_correlation_filter",
        correlate=True,
    )


def _custom_kernel_node(
    source: TileNode,
    kernel: Union[np.ndarray, Sequence[Sequence[float]]],
    normalise: bool,
    zero_dc: Optional[bool],
    *,
    no_data_val: Optional[float],
    name: str,
    correlate: bool,
) -> TileNode:
    """
    Node of a custom kernel: normalises it when asked, decides ``zero_dc``
    from the kernel sum when ``None``, and builds a convolution
    (``correlate=False``) or correlation node.
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
    eff_zero_dc = (
        abs(float(np.sum(k_arr))) < 1e-12 if zero_dc is None else bool(zero_dc)
    )
    builder = kernel_correlate_node if correlate else kernel_conv_node
    return builder(
        source,
        k_arr,
        zero_dc=eff_zero_dc,
        no_data_val=no_data_val,
        name=name,
    )
