# pylint: disable=too-many-lines
"""
Public raster morphological operations for :mod:`rsgispy.imagemorphology`:
the file-based functions and their tile-graph node builders
(``image_*_node``). The file-based functions build
``RasterSource -> node -> DiskSink`` and run it (sequentially numbered local
minima excepted).
"""

from typing import Any, Optional, Tuple

import numpy as np

import rsgispy
from rsgispy._io import HaloBorder, TileNode, halo_border_from_mode, resolve_input_nodata
from rsgispy.imagemorphology._helpers import (
    MorphNodeNodata,
    _load_morph_operator,
    _run_top_hat,
    _run_two_step_seq,
    _run_window_morphology,
    local_minima_node,
    morph_reduce_node,
    morph_top_hat_nodes,
    morph_two_step_nodes,
)
from rsgispy.imageprocess import validate_odd_window_size


def create_circular_op(output_file: str, op_size: int) -> None:
    """
    Create a circular binary structuring-element matrix file (``.gmtxt``).

    If ``output_file`` does not end with ``.gmtxt``, ``.gmtxt`` is appended to
    match RSGISLib's behaviour.

    :param output_file: Output file path for the ``.gmtxt`` operator matrix.
    :param op_size: Odd integer width/height of the square operator matrix.

    .. code:: python

        import rsgispy.imagemorphology

        rsgispy.imagemorphology.create_circular_op(
            output_file="circ_op_5.gmtxt",
            op_size=5,
        )
    """
    size_int = validate_odd_window_size(op_size, min_size=1, name="op_size")

    out_path = output_file if output_file.endswith(".gmtxt") else f"{output_file}.gmtxt"
    radius = (size_int - 1) // 2
    radius_sq = (radius + 0.25) ** 2

    y_coords = np.arange(radius, -radius - 1, -1, dtype=np.int64)[:, np.newaxis]
    x_coords = np.arange(-radius, radius + 1, dtype=np.int64)[np.newaxis, :]
    mat = ((x_coords**2 + y_coords**2) <= radius_sq).astype(np.int64)

    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(f"m={size_int}\n")
        handle.write(f"n={size_int}\n")
        for row in mat:
            handle.write(",".join(str(int(v)) for v in row) + "\n")


def image_dilate(
    input_img: str,
    output_img: str,
    morph_op_file: str = "",
    use_op_file: bool = False,
    op_size: int = 3,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    border_mode: str = "edge",
    border_fill: float = 0.0,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Perform morphological dilation independently on each band of ``input_img``.

    :param input_img: Path to the input image file.
    :param output_img: Path to the output dilated image file.
    :param morph_op_file: Path to a ``.gmtxt`` binary structuring-element file
                          (Default = ``''``).
    :param use_op_file: If ``True``, load operator from ``morph_op_file``;
                        otherwise use a square operator of size ``op_size``
                        (Default = ``False``).
    :param op_size: Square operator size used when ``use_op_file`` is ``False``
                    (Default = ``3``).
    :param gdalformat: GDAL/rasterio output format (Default = ``'GTIFF'``).
    :param datatype: RSGIS data type constant (Default = ``None``, matches
                     ``input_img`` data type).
    :param no_data_val: Optional output no-data value (Default = ``None``,
                        inherits ``input_img`` header no-data when representable).
    :param border_mode: How windows are filled beyond the image edge:
                        ``'edge'`` (repeat the edge pixels; default),
                        ``'reflect'``, ``'symmetric'`` or ``'constant'``
                        (``border_fill``); see :ref:`filter-border-mode`.
    :param border_fill: Value beyond the image edge when ``border_mode``
                        is ``'constant'`` (Default = ``0.0``; NaN leaves
                        those pixels out, like no data).
    :param calc_stats: If ``True``, compute output stats and pyramids
                       (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).

    .. code:: python

        import rsgispy.imagemorphology

        rsgispy.imagemorphology.image_dilate(
            input_img="mask.tif",
            output_img="mask_dilate.tif",
            op_size=3,
        )
    """
    # pylint: disable=too-many-arguments
    op_mask = _load_morph_operator(morph_op_file, use_op_file, op_size)
    _run_window_morphology(
        input_img=input_img,
        output_img=output_img,
        op_mask=op_mask,
        gdalformat=gdalformat,
        datatype=datatype,
        op_kind="dilate",
        combined=False,
        no_data_val=no_data_val,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
        border=halo_border_from_mode(border_mode, border_fill),
    )


def image_erode(
    input_img: str,
    output_img: str,
    morph_op_file: str = "",
    use_op_file: bool = False,
    op_size: int = 3,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    border_mode: str = "edge",
    border_fill: float = 0.0,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Perform morphological erosion independently on each band of ``input_img``.

    :param input_img: Path to the input image file.
    :param output_img: Path to the output eroded image file.
    :param morph_op_file: Path to a ``.gmtxt`` binary structuring-element file
                          (Default = ``''``).
    :param use_op_file: If ``True``, load operator from ``morph_op_file``;
                        otherwise use a square operator of size ``op_size``
                        (Default = ``False``).
    :param op_size: Square operator size used when ``use_op_file`` is ``False``
                    (Default = ``3``).
    :param gdalformat: GDAL/rasterio output format (Default = ``'GTIFF'``).
    :param datatype: RSGIS data type constant (Default = ``None``, matches
                     ``input_img`` data type).
    :param no_data_val: Optional output no-data value (Default = ``None``,
                        inherits ``input_img`` header no-data when representable).
    :param border_mode: How windows are filled beyond the image edge:
                        ``'edge'`` (repeat the edge pixels; default),
                        ``'reflect'``, ``'symmetric'`` or ``'constant'``
                        (``border_fill``); see :ref:`filter-border-mode`.
    :param border_fill: Value beyond the image edge when ``border_mode``
                        is ``'constant'`` (Default = ``0.0``; NaN leaves
                        those pixels out, like no data).
    :param calc_stats: If ``True``, compute output stats and pyramids
                       (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).

    .. code:: python

        import rsgispy.imagemorphology

        rsgispy.imagemorphology.image_erode(
            input_img="mask.tif",
            output_img="mask_erode.tif",
            op_size=3,
        )
    """
    # pylint: disable=too-many-arguments
    op_mask = _load_morph_operator(morph_op_file, use_op_file, op_size)
    _run_window_morphology(
        input_img=input_img,
        output_img=output_img,
        op_mask=op_mask,
        gdalformat=gdalformat,
        datatype=datatype,
        op_kind="erode",
        combined=False,
        no_data_val=no_data_val,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
        border=halo_border_from_mode(border_mode, border_fill),
    )


def image_gradient(
    input_img: str,
    output_img: str,
    morph_op_file: str = "",
    use_op_file: bool = False,
    op_size: int = 3,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    border_mode: str = "edge",
    border_fill: float = 0.0,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Calculate morphological gradient (dilation minus erosion) independently for
    each band of ``input_img``.

    :param input_img: Path to the input image file.
    :param output_img: Path to the output morphological gradient image file.
    :param morph_op_file: Path to a ``.gmtxt`` binary structuring-element file
                          (Default = ``''``).
    :param use_op_file: If ``True``, load operator from ``morph_op_file``;
                        otherwise use a square operator of size ``op_size``
                        (Default = ``False``).
    :param op_size: Square operator size used when ``use_op_file`` is ``False``
                    (Default = ``3``).
    :param gdalformat: GDAL/rasterio output format (Default = ``'GTIFF'``).
    :param datatype: RSGIS data type constant (Default = ``None``, matches
                     ``input_img`` data type).
    :param no_data_val: Optional output no-data value (Default = ``None``,
                        inherits ``input_img`` header no-data when representable).
    :param border_mode: How windows are filled beyond the image edge:
                        ``'edge'`` (repeat the edge pixels; default),
                        ``'reflect'``, ``'symmetric'`` or ``'constant'``
                        (``border_fill``); see :ref:`filter-border-mode`.
    :param border_fill: Value beyond the image edge when ``border_mode``
                        is ``'constant'`` (Default = ``0.0``; NaN leaves
                        those pixels out, like no data).
    :param calc_stats: If ``True``, compute output stats and pyramids
                       (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).

    .. code:: python

        import rsgispy
        import rsgispy.imagemorphology

        rsgispy.imagemorphology.image_gradient(
            input_img="img.tif",
            output_img="img_grad.tif",
            op_size=3,
            datatype=rsgispy.TYPE_32FLOAT,
        )
    """
    # pylint: disable=too-many-arguments
    op_mask = _load_morph_operator(morph_op_file, use_op_file, op_size)
    _run_window_morphology(
        input_img=input_img,
        output_img=output_img,
        op_mask=op_mask,
        gdalformat=gdalformat,
        datatype=datatype,
        op_kind="gradient",
        combined=False,
        no_data_val=no_data_val,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
        border=halo_border_from_mode(border_mode, border_fill),
    )


def image_dilate_combined_out(
    input_img: str,
    output_img: str,
    morph_op_file: str = "",
    use_op_file: bool = False,
    op_size: int = 3,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    border_mode: str = "edge",
    border_fill: float = 0.0,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Perform morphological dilation across all bands of ``input_img`` and write
    a single combined output band.

    :param input_img: Path to the input image file.
    :param output_img: Path to the output single-band dilated image file.
    :param morph_op_file: Path to a ``.gmtxt`` binary structuring-element file
                          (Default = ``''``).
    :param use_op_file: If ``True``, load operator from ``morph_op_file``;
                        otherwise use a square operator of size ``op_size``
                        (Default = ``False``).
    :param op_size: Square operator size used when ``use_op_file`` is ``False``
                    (Default = ``3``).
    :param gdalformat: GDAL/rasterio output format (Default = ``'GTIFF'``).
    :param datatype: RSGIS data type constant (Default = ``None``, matches
                     ``input_img`` data type).
    :param no_data_val: Optional output no-data value (Default = ``None``,
                        inherits ``input_img`` header no-data when representable).
    :param border_mode: How windows are filled beyond the image edge:
                        ``'edge'`` (repeat the edge pixels; default),
                        ``'reflect'``, ``'symmetric'`` or ``'constant'``
                        (``border_fill``); see :ref:`filter-border-mode`.
    :param border_fill: Value beyond the image edge when ``border_mode``
                        is ``'constant'`` (Default = ``0.0``; NaN leaves
                        those pixels out, like no data).
    :param calc_stats: If ``True``, compute output stats and pyramids
                       (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).

    .. code:: python

        import rsgispy.imagemorphology

        rsgispy.imagemorphology.image_dilate_combined_out(
            input_img="masks.tif",
            output_img="masks_dilate_comb.tif",
            op_size=3,
        )
    """
    # pylint: disable=too-many-arguments
    op_mask = _load_morph_operator(morph_op_file, use_op_file, op_size)
    _run_window_morphology(
        input_img=input_img,
        output_img=output_img,
        op_mask=op_mask,
        gdalformat=gdalformat,
        datatype=datatype,
        op_kind="dilate",
        combined=True,
        no_data_val=no_data_val,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
        border=halo_border_from_mode(border_mode, border_fill),
    )


def image_erode_combined_out(
    input_img: str,
    output_img: str,
    morph_op_file: str = "",
    use_op_file: bool = False,
    op_size: int = 3,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    border_mode: str = "edge",
    border_fill: float = 0.0,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Perform morphological erosion across all bands of ``input_img`` and write a
    single combined output band.

    :param input_img: Path to the input image file.
    :param output_img: Path to the output single-band eroded image file.
    :param morph_op_file: Path to a ``.gmtxt`` binary structuring-element file
                          (Default = ``''``).
    :param use_op_file: If ``True``, load operator from ``morph_op_file``;
                        otherwise use a square operator of size ``op_size``
                        (Default = ``False``).
    :param op_size: Square operator size used when ``use_op_file`` is ``False``
                    (Default = ``3``).
    :param gdalformat: GDAL/rasterio output format (Default = ``'GTIFF'``).
    :param datatype: RSGIS data type constant (Default = ``None``, matches
                     ``input_img`` data type).
    :param no_data_val: Optional output no-data value (Default = ``None``,
                        inherits ``input_img`` header no-data when representable).
    :param border_mode: How windows are filled beyond the image edge:
                        ``'edge'`` (repeat the edge pixels; default),
                        ``'reflect'``, ``'symmetric'`` or ``'constant'``
                        (``border_fill``); see :ref:`filter-border-mode`.
    :param border_fill: Value beyond the image edge when ``border_mode``
                        is ``'constant'`` (Default = ``0.0``; NaN leaves
                        those pixels out, like no data).
    :param calc_stats: If ``True``, compute output stats and pyramids
                       (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).

    .. code:: python

        import rsgispy.imagemorphology

        rsgispy.imagemorphology.image_erode_combined_out(
            input_img="masks.tif",
            output_img="masks_erode_comb.tif",
            op_size=3,
        )
    """
    # pylint: disable=too-many-arguments
    op_mask = _load_morph_operator(morph_op_file, use_op_file, op_size)
    _run_window_morphology(
        input_img=input_img,
        output_img=output_img,
        op_mask=op_mask,
        gdalformat=gdalformat,
        datatype=datatype,
        op_kind="erode",
        combined=True,
        no_data_val=no_data_val,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
        border=halo_border_from_mode(border_mode, border_fill),
    )


def image_gradient_combined_out(
    input_img: str,
    output_img: str,
    morph_op_file: str = "",
    use_op_file: bool = False,
    op_size: int = 3,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    border_mode: str = "edge",
    border_fill: float = 0.0,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Calculate morphological gradient across all bands of ``input_img`` and write
    a single combined output band.

    :param input_img: Path to the input image file.
    :param output_img: Path to the output single-band gradient image file.
    :param morph_op_file: Path to a ``.gmtxt`` binary structuring-element file
                          (Default = ``''``).
    :param use_op_file: If ``True``, load operator from ``morph_op_file``;
                        otherwise use a square operator of size ``op_size``
                        (Default = ``False``).
    :param op_size: Square operator size used when ``use_op_file`` is ``False``
                    (Default = ``3``).
    :param gdalformat: GDAL/rasterio output format (Default = ``'GTIFF'``).
    :param datatype: RSGIS data type constant (Default = ``None``, matches
                     ``input_img`` data type).
    :param no_data_val: Optional output no-data value (Default = ``None``,
                        inherits ``input_img`` header no-data when representable).
    :param border_mode: How windows are filled beyond the image edge:
                        ``'edge'`` (repeat the edge pixels; default),
                        ``'reflect'``, ``'symmetric'`` or ``'constant'``
                        (``border_fill``); see :ref:`filter-border-mode`.
    :param border_fill: Value beyond the image edge when ``border_mode``
                        is ``'constant'`` (Default = ``0.0``; NaN leaves
                        those pixels out, like no data).
    :param calc_stats: If ``True``, compute output stats and pyramids
                       (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).

    .. code:: python

        import rsgispy
        import rsgispy.imagemorphology

        rsgispy.imagemorphology.image_gradient_combined_out(
            input_img="img.tif",
            output_img="img_grad_comb.tif",
            op_size=3,
            datatype=rsgispy.TYPE_32FLOAT,
        )
    """
    # pylint: disable=too-many-arguments
    op_mask = _load_morph_operator(morph_op_file, use_op_file, op_size)
    _run_window_morphology(
        input_img=input_img,
        output_img=output_img,
        op_mask=op_mask,
        gdalformat=gdalformat,
        datatype=datatype,
        op_kind="gradient",
        combined=True,
        no_data_val=no_data_val,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
        border=halo_border_from_mode(border_mode, border_fill),
    )


def image_local_minima(
    input_img: str,
    output_img: str,
    sequencial_out: bool = False,
    allow_equal: bool = True,
    morph_op_file: str = "",
    use_op_file: bool = False,
    op_size: int = 3,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Identify local minima independently in each band of ``input_img`` using the
    morphological structuring element.

    :param input_img: Path to the input image file.
    :param output_img: Path to the output local-minima image file.
    :param sequencial_out: If ``True``, number minima sequentially per band
                           (``1, 2, 3, ...``); if ``False``, write a binary
                           mask (``1`` / ``0``; Default = ``False``).
    :param allow_equal: If ``True``, neighbour pixels equal to the centre pixel
                        do not disqualify a local minimum (Default = ``True``).
    :param morph_op_file: Path to a ``.gmtxt`` binary structuring-element file
                          (Default = ``''``).
    :param use_op_file: If ``True``, load operator from ``morph_op_file``;
                        otherwise use a square operator of size ``op_size``
                        (Default = ``False``).
    :param op_size: Square operator size used when ``use_op_file`` is ``False``
                    (Default = ``3``).
    :param gdalformat: GDAL/rasterio output format (Default = ``'GTIFF'``).
    :param datatype: RSGIS data type constant (Default = ``None``, resolves to
                     :data:`rsgispy.TYPE_32UINT` when ``sequencial_out=True``
                     else :data:`rsgispy.TYPE_8UINT`).
    :param no_data_val: Optional output no-data value (Default = ``None``,
                        inherits ``input_img`` header no-data when representable).
    :param calc_stats: If ``True``, compute output stats and pyramids
                       (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).

    .. code:: python

        import rsgispy.imagemorphology

        rsgispy.imagemorphology.image_local_minima(
            input_img="dem.tif",
            output_img="minima.tif",
            sequencial_out=False,
            allow_equal=True,
            op_size=3,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    op_mask = _load_morph_operator(morph_op_file, use_op_file, op_size)
    default_dtype = rsgispy.TYPE_32UINT if sequencial_out else rsgispy.TYPE_8UINT
    _run_window_morphology(
        input_img=input_img,
        output_img=output_img,
        op_mask=op_mask,
        gdalformat=gdalformat,
        datatype=datatype,
        op_kind="local_minima",
        combined=False,
        sequencial_out=sequencial_out,
        allow_equal=allow_equal,
        no_data_val=no_data_val,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
        default_dtype=default_dtype,
    )


def image_local_minima_combined_out(
    input_img: str,
    output_img: str,
    sequencial_out: bool = False,
    allow_equal: bool = True,
    morph_op_file: str = "",
    use_op_file: bool = False,
    op_size: int = 3,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Identify local minima across all bands of ``input_img`` and write a single
    combined output band.

    :param input_img: Path to the input image file.
    :param output_img: Path to the output single-band local-minima image file.
    :param sequencial_out: If ``True``, number minima sequentially
                           (``1, 2, 3, ...``); if ``False``, write a binary
                           mask (``1`` / ``0``; Default = ``False``).
    :param allow_equal: If ``True``, neighbour pixels equal to the minimum band
                        centre value do not disqualify a local minimum
                        (Default = ``True``).
    :param morph_op_file: Path to a ``.gmtxt`` binary structuring-element file
                          (Default = ``''``).
    :param use_op_file: If ``True``, load operator from ``morph_op_file``;
                        otherwise use a square operator of size ``op_size``
                        (Default = ``False``).
    :param op_size: Square operator size used when ``use_op_file`` is ``False``
                    (Default = ``3``).
    :param gdalformat: GDAL/rasterio output format (Default = ``'GTIFF'``).
    :param datatype: RSGIS data type constant (Default = ``None``, resolves to
                     :data:`rsgispy.TYPE_32UINT` when ``sequencial_out=True``
                     else :data:`rsgispy.TYPE_8UINT`).
    :param no_data_val: Optional output no-data value (Default = ``None``,
                        inherits ``input_img`` header no-data when representable).
    :param calc_stats: If ``True``, compute output stats and pyramids
                       (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).

    .. code:: python

        import rsgispy.imagemorphology

        rsgispy.imagemorphology.image_local_minima_combined_out(
            input_img="bands.tif",
            output_img="minima_comb.tif",
            sequencial_out=False,
            allow_equal=True,
            op_size=3,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    op_mask = _load_morph_operator(morph_op_file, use_op_file, op_size)
    default_dtype = rsgispy.TYPE_32UINT if sequencial_out else rsgispy.TYPE_8UINT
    _run_window_morphology(
        input_img=input_img,
        output_img=output_img,
        op_mask=op_mask,
        gdalformat=gdalformat,
        datatype=datatype,
        op_kind="local_minima",
        combined=True,
        sequencial_out=sequencial_out,
        allow_equal=allow_equal,
        no_data_val=no_data_val,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
        default_dtype=default_dtype,
    )


def image_opening(
    input_img: str,
    output_img: str,
    morph_op_file: str = "",
    use_op_file: bool = False,
    op_size: int = 3,
    tmp_img: Optional[str] = None,
    niters: int = 1,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    border_mode: str = "edge",
    border_fill: float = 0.0,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Perform morphological opening (erosion followed by dilation) on ``input_img``.

    Executed in a single halo-fused tile pass in memory with zero intermediate
    disk I/O unless ``tmp_img`` is explicitly provided.

    :param input_img: Path to the input image file.
    :param output_img: Path to the output opened image file.
    :param morph_op_file: Path to a ``.gmtxt`` binary structuring-element file
                          (Default = ``''``).
    :param use_op_file: If ``True``, load operator from ``morph_op_file``;
                        otherwise use a square operator of size ``op_size``
                        (Default = ``False``).
    :param op_size: Square operator size used when ``use_op_file`` is ``False``
                    (Default = ``3``).
    :param tmp_img: Optional path to also save the intermediate eroded raster;
                    if empty (``''``) or ``None``, no intermediate file is
                    written (Default = ``None``).
    :param niters: Number of opening iterations (Default = ``1``).
    :param gdalformat: GDAL/rasterio output format (Default = ``'GTIFF'``).
    :param datatype: RSGIS data type constant (Default = ``None``, matches
                     ``input_img`` data type).
    :param no_data_val: Optional output no-data value (Default = ``None``,
                        inherits ``input_img`` header no-data when representable).
    :param border_mode: How windows are filled beyond the image edge:
                        ``'edge'`` (repeat the edge pixels; default),
                        ``'reflect'``, ``'symmetric'`` or ``'constant'``
                        (``border_fill``); see :ref:`filter-border-mode`.
    :param border_fill: Value beyond the image edge when ``border_mode``
                        is ``'constant'`` (Default = ``0.0``; NaN leaves
                        those pixels out, like no data).
    :param calc_stats: If ``True``, compute output stats and pyramids
                       (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).

    .. code:: python

        import rsgispy.imagemorphology

        rsgispy.imagemorphology.image_opening(
            input_img="mask.tif",
            output_img="mask_open.tif",
            op_size=3,
            niters=1,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    op_mask = _load_morph_operator(morph_op_file, use_op_file, op_size)
    _run_two_step_seq(
        input_img=input_img,
        output_img=output_img,
        tmp_img=tmp_img,
        op_mask=op_mask,
        gdalformat=gdalformat,
        datatype=datatype,
        first_op="erode",
        second_op="dilate",
        niters=niters,
        no_data_val=no_data_val,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
        border=halo_border_from_mode(border_mode, border_fill),
    )


def image_closing(
    input_img: str,
    output_img: str,
    morph_op_file: str = "",
    use_op_file: bool = False,
    op_size: int = 3,
    tmp_img: Optional[str] = None,
    niters: int = 1,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    border_mode: str = "edge",
    border_fill: float = 0.0,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Perform morphological closing (dilation followed by erosion) on ``input_img``.

    Executed in a single halo-fused tile pass in memory with zero intermediate
    disk I/O unless ``tmp_img`` is explicitly provided.

    :param input_img: Path to the input image file.
    :param output_img: Path to the output closed image file.
    :param morph_op_file: Path to a ``.gmtxt`` binary structuring-element file
                          (Default = ``''``).
    :param use_op_file: If ``True``, load operator from ``morph_op_file``;
                        otherwise use a square operator of size ``op_size``
                        (Default = ``False``).
    :param op_size: Square operator size used when ``use_op_file`` is ``False``
                    (Default = ``3``).
    :param tmp_img: Optional path to also save the intermediate dilated raster;
                    if empty (``''``) or ``None``, no intermediate file is
                    written (Default = ``None``).
    :param niters: Number of closing iterations (Default = ``1``).
    :param gdalformat: GDAL/rasterio output format (Default = ``'GTIFF'``).
    :param datatype: RSGIS data type constant (Default = ``None``, matches
                     ``input_img`` data type).
    :param no_data_val: Optional output no-data value (Default = ``None``,
                        inherits ``input_img`` header no-data when representable).
    :param border_mode: How windows are filled beyond the image edge:
                        ``'edge'`` (repeat the edge pixels; default),
                        ``'reflect'``, ``'symmetric'`` or ``'constant'``
                        (``border_fill``); see :ref:`filter-border-mode`.
    :param border_fill: Value beyond the image edge when ``border_mode``
                        is ``'constant'`` (Default = ``0.0``; NaN leaves
                        those pixels out, like no data).
    :param calc_stats: If ``True``, compute output stats and pyramids
                       (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).

    .. code:: python

        import rsgispy.imagemorphology

        rsgispy.imagemorphology.image_closing(
            input_img="mask.tif",
            output_img="mask_close.tif",
            op_size=3,
            niters=1,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    op_mask = _load_morph_operator(morph_op_file, use_op_file, op_size)
    _run_two_step_seq(
        input_img=input_img,
        output_img=output_img,
        tmp_img=tmp_img,
        op_mask=op_mask,
        gdalformat=gdalformat,
        datatype=datatype,
        first_op="dilate",
        second_op="erode",
        niters=niters,
        no_data_val=no_data_val,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
        border=halo_border_from_mode(border_mode, border_fill),
    )


def image_black_top_hat(
    input_img: str,
    output_img: str,
    morph_op_file: str = "",
    use_op_file: bool = False,
    op_size: int = 3,
    tmp_img: Optional[str] = None,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    border_mode: str = "edge",
    border_fill: float = 0.0,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Perform a morphological black top-hat transform (closing minus input image).

    Executed in a single halo-fused tile pass in memory with zero intermediate
    disk I/O unless ``tmp_img`` is explicitly provided.

    :param input_img: Path to the input image file.
    :param output_img: Path to the output black top-hat image file.
    :param morph_op_file: Path to a ``.gmtxt`` binary structuring-element file
                          (Default = ``''``).
    :param use_op_file: If ``True``, load operator from ``morph_op_file``;
                        otherwise use a square operator of size ``op_size``
                        (Default = ``False``).
    :param op_size: Square operator size used when ``use_op_file`` is ``False``
                    (Default = ``3``).
    :param tmp_img: Optional path to also save the intermediate closed raster;
                    if empty (``''``) or ``None``, no intermediate file is
                    written (Default = ``None``).
    :param gdalformat: GDAL/rasterio output format (Default = ``'GTIFF'``).
    :param datatype: RSGIS data type constant (Default = ``None``, matches
                     ``input_img`` data type).
    :param no_data_val: Optional output no-data value (Default = ``None``,
                        inherits ``input_img`` header no-data when representable).
    :param border_mode: How windows are filled beyond the image edge:
                        ``'edge'`` (repeat the edge pixels; default),
                        ``'reflect'``, ``'symmetric'`` or ``'constant'``
                        (``border_fill``); see :ref:`filter-border-mode`.
    :param border_fill: Value beyond the image edge when ``border_mode``
                        is ``'constant'`` (Default = ``0.0``; NaN leaves
                        those pixels out, like no data).
    :param calc_stats: If ``True``, compute output stats and pyramids
                       (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).

    .. code:: python

        import rsgispy
        import rsgispy.imagemorphology

        rsgispy.imagemorphology.image_black_top_hat(
            input_img="dem.tif",
            output_img="dem_bth.tif",
            op_size=5,
            datatype=rsgispy.TYPE_32FLOAT,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    op_mask = _load_morph_operator(morph_op_file, use_op_file, op_size)
    _run_top_hat(
        input_img=input_img,
        output_img=output_img,
        tmp_img=tmp_img,
        op_mask=op_mask,
        gdalformat=gdalformat,
        datatype=datatype,
        black=True,
        no_data_val=no_data_val,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
        border=halo_border_from_mode(border_mode, border_fill),
    )


def image_white_top_hat(
    input_img: str,
    output_img: str,
    morph_op_file: str = "",
    use_op_file: bool = False,
    op_size: int = 3,
    tmp_img: Optional[str] = None,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    border_mode: str = "edge",
    border_fill: float = 0.0,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Perform a morphological white top-hat transform (input image minus opening).

    Executed in a single halo-fused tile pass in memory with zero intermediate
    disk I/O unless ``tmp_img`` is explicitly provided.

    :param input_img: Path to the input image file.
    :param output_img: Path to the output white top-hat image file.
    :param morph_op_file: Path to a ``.gmtxt`` binary structuring-element file
                          (Default = ``''``).
    :param use_op_file: If ``True``, load operator from ``morph_op_file``;
                        otherwise use a square operator of size ``op_size``
                        (Default = ``False``).
    :param op_size: Square operator size used when ``use_op_file`` is ``False``
                    (Default = ``3``).
    :param tmp_img: Optional path to also save the intermediate opened raster;
                    if empty (``''``) or ``None``, no intermediate file is
                    written (Default = ``None``).
    :param gdalformat: GDAL/rasterio output format (Default = ``'GTIFF'``).
    :param datatype: RSGIS data type constant (Default = ``None``, matches
                     ``input_img`` data type).
    :param no_data_val: Optional output no-data value (Default = ``None``,
                        inherits ``input_img`` header no-data when representable).
    :param border_mode: How windows are filled beyond the image edge:
                        ``'edge'`` (repeat the edge pixels; default),
                        ``'reflect'``, ``'symmetric'`` or ``'constant'``
                        (``border_fill``); see :ref:`filter-border-mode`.
    :param border_fill: Value beyond the image edge when ``border_mode``
                        is ``'constant'`` (Default = ``0.0``; NaN leaves
                        those pixels out, like no data).
    :param calc_stats: If ``True``, compute output stats and pyramids
                       (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).

    .. code:: python

        import rsgispy
        import rsgispy.imagemorphology

        rsgispy.imagemorphology.image_white_top_hat(
            input_img="dem.tif",
            output_img="dem_wth.tif",
            op_size=5,
            datatype=rsgispy.TYPE_32FLOAT,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    op_mask = _load_morph_operator(morph_op_file, use_op_file, op_size)
    _run_top_hat(
        input_img=input_img,
        output_img=output_img,
        tmp_img=tmp_img,
        op_mask=op_mask,
        gdalformat=gdalformat,
        datatype=datatype,
        black=False,
        no_data_val=no_data_val,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
        border=halo_border_from_mode(border_mode, border_fill),
    )


# ---------------------------------------------------------------------------
# Tile-graph node builders
# ---------------------------------------------------------------------------


def _reduce_node(
    op_kind: str,
    source: TileNode,
    op_args: Tuple[str, bool, int],
    combined: bool,
    dtype: Any,
    no_data_val: Optional[float],
    name: Optional[str],
    *,
    border: HaloBorder,
) -> TileNode:
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    op_mask = _load_morph_operator(*op_args)
    nodata = MorphNodeNodata(resolve_input_nodata(source, no_data_val), dtype)
    return morph_reduce_node(
        source,
        op_mask,
        op_kind,
        nodata,
        combined=combined,
        in_nd=nodata.in_nd,
        border=border,
        name=name or f"image_{op_kind}",
    )


def image_dilate_node(
    source: TileNode,
    morph_op_file: str = "",
    use_op_file: bool = False,
    op_size: int = 3,
    *,
    combined: bool = False,
    dtype: Any = None,
    no_data_val: Optional[float] = None,
    border_mode: str = "edge",
    border_fill: float = 0.0,
    name: Optional[str] = None,
) -> TileNode:
    """
    Tile-graph node of :func:`image_dilate`: the maximum of the valid pixels
    under the structuring element, for each band of ``source`` (or across
    all bands with ``combined=True``, as :func:`image_dilate_combined_out`),
    for chaining in a :class:`~rsgispy.imageprocess.TileGraph` without
    intermediate files. The halo radius is half the operator size.

    The node outputs ``float64`` blocks with the bands (and band names) of
    ``source`` (one band when ``combined``). ``no_data_val`` is the input
    no-data value (by default that of ``source``'s metadata; see
    :ref:`tilegraph-nodata`): these pixels are ignored, and output pixels without a value are
    set to it (to NaN when it cannot be stored in ``dtype``; NaN becomes 0
    when written to an integer raster).

    :param source: Upstream :class:`~rsgispy.imageprocess.TileNode`.
    :param morph_op_file: Path to a ``.gmtxt`` structuring-element file
                          (Default = ``''``).
    :param use_op_file: Use ``morph_op_file`` instead of a square operator of
                        size ``op_size`` (Default = ``False``).
    :param op_size: Odd square operator size >= 3 (Default = ``3``).
    :param combined: Reduce across all bands into one output band
                     (Default = ``False``).
    :param dtype: NumPy dtype of the raster the result will be written to,
                  used to decide whether ``no_data_val`` can be kept
                  (Default = ``None``, ``float64``).
    :param no_data_val: Input no-data value (Default = ``None``: that of
                        ``source``'s metadata).
    :param border_mode: How windows are filled beyond the image edge:
                        ``'edge'`` (repeat the edge pixels; default),
                        ``'reflect'``, ``'symmetric'`` or ``'constant'``
                        (``border_fill``); see :ref:`filter-border-mode`.
    :param border_fill: Value beyond the image edge when ``border_mode``
                        is ``'constant'`` (Default = ``0.0``; NaN leaves
                        those pixels out, like no data).
    :param name: Optional node name (Default = ``'image_dilate'``).
    :return: :class:`~rsgispy.imageprocess.WindowOp` node.

    .. code:: python

        import rsgispy
        from rsgispy.imageprocess import DiskSink, RasterSource, TileGraph
        from rsgispy.imagemorphology import image_dilate_node, image_erode_node

        src = RasterSource("mask.tif")
        # Dilate then erode with different operators, without temporary files.
        node = image_erode_node(image_dilate_node(src, op_size=5), op_size=3)
        TileGraph(DiskSink(node, "mask_out.tif", datatype=rsgispy.TYPE_8UINT)).run()
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    return _reduce_node(
        "dilate", source, (morph_op_file, use_op_file, op_size), combined,
        dtype, no_data_val, name,
        border=halo_border_from_mode(border_mode, border_fill),
    )


def image_erode_node(
    source: TileNode,
    morph_op_file: str = "",
    use_op_file: bool = False,
    op_size: int = 3,
    *,
    combined: bool = False,
    dtype: Any = None,
    no_data_val: Optional[float] = None,
    border_mode: str = "edge",
    border_fill: float = 0.0,
    name: Optional[str] = None,
) -> TileNode:
    """
    Tile-graph node of :func:`image_erode`: the minimum of the valid pixels
    under the structuring element, for each band of ``source`` (or across
    all bands with ``combined=True``, as :func:`image_erode_combined_out`).
    The halo radius is half the operator size.

    The node outputs ``float64`` blocks with the bands (and band names) of
    ``source`` (one band when ``combined``); ``no_data_val`` and ``dtype``
    work as in :func:`image_dilate_node`.

    :param source: Upstream :class:`~rsgispy.imageprocess.TileNode`.
    :param morph_op_file: Path to a ``.gmtxt`` structuring-element file
                          (Default = ``''``).
    :param use_op_file: Use ``morph_op_file`` instead of a square operator of
                        size ``op_size`` (Default = ``False``).
    :param op_size: Odd square operator size >= 3 (Default = ``3``).
    :param combined: Reduce across all bands into one output band
                     (Default = ``False``).
    :param dtype: NumPy dtype of the raster the result will be written to
                  (Default = ``None``, ``float64``).
    :param no_data_val: Input no-data value (Default = ``None``: that of
                        ``source``'s metadata).
    :param border_mode: How windows are filled beyond the image edge:
                        ``'edge'`` (repeat the edge pixels; default),
                        ``'reflect'``, ``'symmetric'`` or ``'constant'``
                        (``border_fill``); see :ref:`filter-border-mode`.
    :param border_fill: Value beyond the image edge when ``border_mode``
                        is ``'constant'`` (Default = ``0.0``; NaN leaves
                        those pixels out, like no data).
    :param name: Optional node name (Default = ``'image_erode'``).
    :return: :class:`~rsgispy.imageprocess.WindowOp` node.

    .. code:: python

        from rsgispy.imageprocess import BandMathOp, DiskSink, RasterSource, TileGraph
        from rsgispy.imagemorphology import image_erode_node

        src = RasterSource("dem.tif", nodata=-9999)
        eroded = image_erode_node(src, op_size=7, no_data_val=-9999)
        # Height above the local minimum surface, in one pass.
        diff = BandMathOp({"d": src, "e": eroded}, "d - e", nodata=-9999)
        TileGraph(DiskSink(diff, "dem_minus_erosion.tif")).run()
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    return _reduce_node(
        "erode", source, (morph_op_file, use_op_file, op_size), combined,
        dtype, no_data_val, name,
        border=halo_border_from_mode(border_mode, border_fill),
    )


def image_gradient_node(
    source: TileNode,
    morph_op_file: str = "",
    use_op_file: bool = False,
    op_size: int = 3,
    *,
    combined: bool = False,
    dtype: Any = None,
    no_data_val: Optional[float] = None,
    border_mode: str = "edge",
    border_fill: float = 0.0,
    name: Optional[str] = None,
) -> TileNode:
    """
    Tile-graph node of :func:`image_gradient`: the morphological gradient
    (dilation minus erosion over the same structuring element) of each band
    of ``source`` (or across all bands with ``combined=True``, as
    :func:`image_gradient_combined_out`). The halo radius is half the
    operator size.

    The node outputs ``float64`` blocks with the bands (and band names) of
    ``source`` (one band when ``combined``); ``no_data_val`` and ``dtype``
    work as in :func:`image_dilate_node`.

    :param source: Upstream :class:`~rsgispy.imageprocess.TileNode`.
    :param morph_op_file: Path to a ``.gmtxt`` structuring-element file
                          (Default = ``''``).
    :param use_op_file: Use ``morph_op_file`` instead of a square operator of
                        size ``op_size`` (Default = ``False``).
    :param op_size: Odd square operator size >= 3 (Default = ``3``).
    :param combined: Reduce across all bands into one output band
                     (Default = ``False``).
    :param dtype: NumPy dtype of the raster the result will be written to
                  (Default = ``None``, ``float64``).
    :param no_data_val: Input no-data value (Default = ``None``: that of
                        ``source``'s metadata).
    :param border_mode: How windows are filled beyond the image edge:
                        ``'edge'`` (repeat the edge pixels; default),
                        ``'reflect'``, ``'symmetric'`` or ``'constant'``
                        (``border_fill``); see :ref:`filter-border-mode`.
    :param border_fill: Value beyond the image edge when ``border_mode``
                        is ``'constant'`` (Default = ``0.0``; NaN leaves
                        those pixels out, like no data).
    :param name: Optional node name (Default = ``'image_gradient'``).
    :return: :class:`~rsgispy.imageprocess.WindowOp` node.

    .. code:: python

        from rsgispy.imageprocess import DiskSink, RasterSource, TileGraph
        from rsgispy.imagefilter.stats import median_filter_node
        from rsgispy.imagemorphology import image_gradient_node

        src = RasterSource("s2_nir.tif")
        edges = image_gradient_node(median_filter_node(src, 5), op_size=3)
        TileGraph(DiskSink(edges, "nir_edges.tif")).run()
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    return _reduce_node(
        "gradient", source, (morph_op_file, use_op_file, op_size), combined,
        dtype, no_data_val, name,
        border=halo_border_from_mode(border_mode, border_fill),
    )


def image_local_minima_node(
    source: TileNode,
    sequencial_out: bool = False,
    allow_equal: bool = True,
    morph_op_file: str = "",
    use_op_file: bool = False,
    op_size: int = 3,
    *,
    combined: bool = False,
    no_data_val: Optional[float] = None,
    name: Optional[str] = None,
) -> TileNode:
    """
    Tile-graph node of :func:`image_local_minima` (binary output): ``1``
    where a valid pixel is a local minimum of its structuring-element
    neighbourhood (no valid neighbour is lower; with ``allow_equal=False``
    no valid neighbour is lower or equal), ``0`` elsewhere. With
    ``combined=True`` (as :func:`image_local_minima_combined_out`) the
    minimum across bands is tested and the node has one band.

    Sequential numbering of the minima (``sequencial_out=True`` in the file
    functions) depends on every pixel before the current one in raster scan
    order, so it is not tile-local and is not available as a node:
    ``sequencial_out=True`` raises :class:`rsgispy.RSGISPyException`.

    :param source: Upstream :class:`~rsgispy.imageprocess.TileNode`.
    :param sequencial_out: Must be ``False`` (Default = ``False``).
    :param allow_equal: Accept neighbours equal to the centre
                        (Default = ``True``).
    :param morph_op_file: Path to a ``.gmtxt`` structuring-element file
                          (Default = ``''``).
    :param use_op_file: Use ``morph_op_file`` instead of a square operator of
                        size ``op_size`` (Default = ``False``).
    :param op_size: Odd square operator size >= 3 (Default = ``3``).
    :param combined: Test the minimum across all bands (Default = ``False``).
    :param no_data_val: Input no-data value; these pixels are never minima
                        and are ignored as neighbours (Default = ``None``:
                        the no-data value of ``source``'s metadata).
    :param name: Optional node name (Default = ``'image_local_minima'``).
    :return: :class:`~rsgispy.imageprocess.WindowOp` node (``float64``
             blocks of 0 and 1).

    .. code:: python

        import rsgispy
        from rsgispy.imageprocess import DiskSink, RasterSource, TileGraph
        from rsgispy.imagefilter.stats import mean_filter_node
        from rsgispy.imagemorphology import image_local_minima_node

        dem = RasterSource("dem.tif")
        pits = image_local_minima_node(mean_filter_node(dem, 5), op_size=5)
        TileGraph(DiskSink(pits, "pits.tif", datatype=rsgispy.TYPE_8UINT)).run()
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    if sequencial_out:
        raise rsgispy.RSGISPyException(
            "image_local_minima_node does not support sequencial_out=True: "
            "sequential numbering is not tile-local. Use image_local_minima "
            "(file based) or number the binary minima afterwards."
        )
    op_mask = _load_morph_operator(morph_op_file, use_op_file, op_size)
    in_nd = resolve_input_nodata(source, no_data_val)
    return local_minima_node(
        source,
        op_mask,
        combined=combined,
        allow_equal=allow_equal,
        in_nd=float(in_nd) if in_nd is not None else None,
        name=name or "image_local_minima",
    )


def _two_step_node(
    source: TileNode,
    op_args: Tuple[str, bool, int],
    first_op: str,
    niters: int,
    dtype: Any,
    no_data_val: Optional[float],
    name: str,
    *,
    border: HaloBorder,
) -> TileNode:
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    op_mask = _load_morph_operator(*op_args)
    second_op = "dilate" if first_op == "erode" else "erode"
    _, result = morph_two_step_nodes(
        source, op_mask, first_op, second_op, niters,
        MorphNodeNodata(resolve_input_nodata(source, no_data_val), dtype), name,
        border=border,
    )
    return result


def image_opening_node(
    source: TileNode,
    morph_op_file: str = "",
    use_op_file: bool = False,
    op_size: int = 3,
    niters: int = 1,
    *,
    dtype: Any = None,
    no_data_val: Optional[float] = None,
    border_mode: str = "edge",
    border_fill: float = 0.0,
    name: Optional[str] = None,
) -> TileNode:
    """
    Tile-graph node of :func:`image_opening`: ``niters`` erosions followed
    by ``niters`` dilations of each band of ``source``. The node chains
    ``2 * niters`` window nodes, so the total halo is ``2 * niters`` times
    half the operator size.

    When ``dtype`` is an integer type, each step is rounded to it (as
    :func:`image_opening` does when it writes an integer raster);
    ``no_data_val`` works as in :func:`image_dilate_node`.

    :param source: Upstream :class:`~rsgispy.imageprocess.TileNode`.
    :param morph_op_file: Path to a ``.gmtxt`` structuring-element file
                          (Default = ``''``).
    :param use_op_file: Use ``morph_op_file`` instead of a square operator of
                        size ``op_size`` (Default = ``False``).
    :param op_size: Odd square operator size >= 3 (Default = ``3``).
    :param niters: Number of erosions (and of dilations) >= 1
                   (Default = ``1``).
    :param dtype: NumPy dtype of the raster the result will be written to
                  (Default = ``None``, ``float64``).
    :param no_data_val: Input no-data value (Default = ``None``: that of
                        ``source``'s metadata).
    :param border_mode: How windows are filled beyond the image edge:
                        ``'edge'`` (repeat the edge pixels; default),
                        ``'reflect'``, ``'symmetric'`` or ``'constant'``
                        (``border_fill``); see :ref:`filter-border-mode`.
    :param border_fill: Value beyond the image edge when ``border_mode``
                        is ``'constant'`` (Default = ``0.0``; NaN leaves
                        those pixels out, like no data).
    :param name: Optional node name prefix (Default = ``'image_opening'``).
    :return: Last node of the opening.

    .. code:: python

        import numpy as np
        import rsgispy
        from rsgispy.imageprocess import DiskSink, RasterSource, TileGraph
        from rsgispy.imagemorphology import image_closing_node, image_opening_node

        mask = RasterSource("cloud_mask.tif")
        # Remove specks, then fill small holes, in one pass.
        node = image_closing_node(
            image_opening_node(mask, op_size=3, dtype=np.uint8),
            op_size=5,
            dtype=np.uint8,
        )
        TileGraph(
            DiskSink(node, "cloud_mask_clean.tif", datatype=rsgispy.TYPE_8UINT)
        ).run()
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    return _two_step_node(
        source, (morph_op_file, use_op_file, op_size), "erode", niters,
        dtype, no_data_val, name or "image_opening",
        border=halo_border_from_mode(border_mode, border_fill),
    )


def image_closing_node(
    source: TileNode,
    morph_op_file: str = "",
    use_op_file: bool = False,
    op_size: int = 3,
    niters: int = 1,
    *,
    dtype: Any = None,
    no_data_val: Optional[float] = None,
    border_mode: str = "edge",
    border_fill: float = 0.0,
    name: Optional[str] = None,
) -> TileNode:
    """
    Tile-graph node of :func:`image_closing`: ``niters`` dilations followed
    by ``niters`` erosions of each band of ``source`` (total halo
    ``2 * niters`` times half the operator size). ``dtype`` and
    ``no_data_val`` work as in :func:`image_opening_node`.

    :param source: Upstream :class:`~rsgispy.imageprocess.TileNode`.
    :param morph_op_file: Path to a ``.gmtxt`` structuring-element file
                          (Default = ``''``).
    :param use_op_file: Use ``morph_op_file`` instead of a square operator of
                        size ``op_size`` (Default = ``False``).
    :param op_size: Odd square operator size >= 3 (Default = ``3``).
    :param niters: Number of dilations (and of erosions) >= 1
                   (Default = ``1``).
    :param dtype: NumPy dtype of the raster the result will be written to
                  (Default = ``None``, ``float64``).
    :param no_data_val: Input no-data value (Default = ``None``: that of
                        ``source``'s metadata).
    :param border_mode: How windows are filled beyond the image edge:
                        ``'edge'`` (repeat the edge pixels; default),
                        ``'reflect'``, ``'symmetric'`` or ``'constant'``
                        (``border_fill``); see :ref:`filter-border-mode`.
    :param border_fill: Value beyond the image edge when ``border_mode``
                        is ``'constant'`` (Default = ``0.0``; NaN leaves
                        those pixels out, like no data).
    :param name: Optional node name prefix (Default = ``'image_closing'``).
    :return: Last node of the closing.

    .. code:: python

        import rsgispy
        from rsgispy.imageprocess import DiskSink, RasterSource, TileGraph
        from rsgispy.classification.classpostprocess import majority_filter_node
        from rsgispy.imagemorphology import image_closing_node

        water = RasterSource("water_mask.tif")
        node = majority_filter_node(image_closing_node(water, op_size=3), 5)
        TileGraph(DiskSink(node, "water_clean.tif", datatype=rsgispy.TYPE_8UINT)).run()
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    return _two_step_node(
        source, (morph_op_file, use_op_file, op_size), "dilate", niters,
        dtype, no_data_val, name or "image_closing",
        border=halo_border_from_mode(border_mode, border_fill),
    )


def _top_hat_node(
    source: TileNode,
    op_args: Tuple[str, bool, int],
    black: bool,
    dtype: Any,
    no_data_val: Optional[float],
    name: str,
    *,
    border: HaloBorder,
) -> TileNode:
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    op_mask = _load_morph_operator(*op_args)
    _, result = morph_top_hat_nodes(
        source,
        op_mask,
        black,
        MorphNodeNodata(resolve_input_nodata(source, no_data_val), dtype),
        name,
        border=border,
    )
    return result


def image_white_top_hat_node(
    source: TileNode,
    morph_op_file: str = "",
    use_op_file: bool = False,
    op_size: int = 3,
    *,
    dtype: Any = None,
    no_data_val: Optional[float] = None,
    border_mode: str = "edge",
    border_fill: float = 0.0,
    name: Optional[str] = None,
) -> TileNode:
    """
    Tile-graph node of :func:`image_white_top_hat`: ``source`` minus its
    opening (bright features smaller than the structuring element). The
    node reads ``source`` both directly and through the opening; the halo is
    twice half the operator size. ``dtype`` and ``no_data_val`` work as in
    :func:`image_opening_node`.

    :param source: Upstream :class:`~rsgispy.imageprocess.TileNode`.
    :param morph_op_file: Path to a ``.gmtxt`` structuring-element file
                          (Default = ``''``).
    :param use_op_file: Use ``morph_op_file`` instead of a square operator of
                        size ``op_size`` (Default = ``False``).
    :param op_size: Odd square operator size >= 3 (Default = ``3``).
    :param dtype: NumPy dtype of the raster the result will be written to
                  (Default = ``None``, ``float64``).
    :param no_data_val: Input no-data value (Default = ``None``: that of
                        ``source``'s metadata).
    :param border_mode: How windows are filled beyond the image edge:
                        ``'edge'`` (repeat the edge pixels; default),
                        ``'reflect'``, ``'symmetric'`` or ``'constant'``
                        (``border_fill``); see :ref:`filter-border-mode`.
    :param border_fill: Value beyond the image edge when ``border_mode``
                        is ``'constant'`` (Default = ``0.0``; NaN leaves
                        those pixels out, like no data).
    :param name: Optional node name (Default = ``'image_white_top_hat'``).
    :return: :class:`~rsgispy.imageprocess.PointwiseOp` node.

    .. code:: python

        from rsgispy.imageprocess import DiskSink, RasterSource, TileGraph
        from rsgispy.imagefilter.stats import median_filter_node
        from rsgispy.imagemorphology import image_white_top_hat_node

        dsm = RasterSource("dsm.tif", nodata=-9999)
        smooth = median_filter_node(dsm, 3, no_data_val=-9999)
        objects = image_white_top_hat_node(smooth, op_size=11, no_data_val=-9999)
        TileGraph(DiskSink(objects, "object_heights.tif")).run()
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    return _top_hat_node(
        source, (morph_op_file, use_op_file, op_size), False, dtype,
        no_data_val, name or "image_white_top_hat",
        border=halo_border_from_mode(border_mode, border_fill),
    )


def image_black_top_hat_node(
    source: TileNode,
    morph_op_file: str = "",
    use_op_file: bool = False,
    op_size: int = 3,
    *,
    dtype: Any = None,
    no_data_val: Optional[float] = None,
    border_mode: str = "edge",
    border_fill: float = 0.0,
    name: Optional[str] = None,
) -> TileNode:
    """
    Tile-graph node of :func:`image_black_top_hat`: the closing of
    ``source`` minus ``source`` (dark features smaller than the structuring
    element). ``dtype`` and ``no_data_val`` work as in
    :func:`image_opening_node`.

    :param source: Upstream :class:`~rsgispy.imageprocess.TileNode`.
    :param morph_op_file: Path to a ``.gmtxt`` structuring-element file
                          (Default = ``''``).
    :param use_op_file: Use ``morph_op_file`` instead of a square operator of
                        size ``op_size`` (Default = ``False``).
    :param op_size: Odd square operator size >= 3 (Default = ``3``).
    :param dtype: NumPy dtype of the raster the result will be written to
                  (Default = ``None``, ``float64``).
    :param no_data_val: Input no-data value (Default = ``None``: that of
                        ``source``'s metadata).
    :param border_mode: How windows are filled beyond the image edge:
                        ``'edge'`` (repeat the edge pixels; default),
                        ``'reflect'``, ``'symmetric'`` or ``'constant'``
                        (``border_fill``); see :ref:`filter-border-mode`.
    :param border_fill: Value beyond the image edge when ``border_mode``
                        is ``'constant'`` (Default = ``0.0``; NaN leaves
                        those pixels out, like no data).
    :param name: Optional node name (Default = ``'image_black_top_hat'``).
    :return: :class:`~rsgispy.imageprocess.PointwiseOp` node.

    .. code:: python

        from rsgispy.imageprocess import DiskSink, RasterSource, TileGraph
        from rsgispy.imagemorphology import image_black_top_hat_node

        dem = RasterSource("dem.tif", nodata=-9999)
        hollows = image_black_top_hat_node(dem, op_size=9, no_data_val=-9999)
        TileGraph(DiskSink(hollows, "hollows.tif")).run()
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    return _top_hat_node(
        source, (morph_op_file, use_op_file, op_size), True, dtype,
        no_data_val, name or "image_black_top_hat",
        border=halo_border_from_mode(border_mode, border_fill),
    )
