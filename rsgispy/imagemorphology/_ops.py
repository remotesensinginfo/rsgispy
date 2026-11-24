"""
Public raster morphological operations for :mod:`rsgispy.imagemorphology`.
"""

from typing import Optional

import numpy as np

import rsgispy
from rsgispy.imagemorphology._helpers import (
    _load_morph_operator,
    _run_top_hat,
    _run_two_step_seq,
    _run_window_morphology,
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
    )
