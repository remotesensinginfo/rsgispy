"""
The imagecalc.maths module provides muParser-based band and image math
calculation functions using ``pymu_eval`` and ``rasterio``.
"""

from contextlib import ExitStack
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import pymu_eval
import rasterio
from rasterio.windows import Window

import rsgispy
import rsgispy.tools.maths
from rsgispy.imageprocess import (
    BandDefn,
    cast_result_block,
    compute_spatial_intersection,
    ensure_file_exists,
    fits_dtype_range,
    iter_dataset_windows,
    iter_image_blocks,
    normalise_band_defs,
    open_input_datasets,
    open_output_dataset,
    pop_stats_on_close,
    process_blocks,
    set_dataset_nodata,
)

__all__ = [
    "BandDefn",
    "band_maths",
    "image_math",
    "image_band_math",
    "calc_prop_true_exp",
]


def _split_top_level_exprs(exp: str) -> List[str]:
    """Split a comma-separated muParser expression at top-level commas."""
    parts: List[str] = []
    buf: List[str] = []
    depth = 0
    in_quote: Optional[str] = None
    escaped = False

    for ch in exp:
        if in_quote is not None:
            buf.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == in_quote:
                in_quote = None
            continue

        if ch in ('"', "'"):
            in_quote = ch
            buf.append(ch)
        elif ch == "(":
            depth += 1
            buf.append(ch)
        elif ch == ")":
            depth = max(0, depth - 1)
            buf.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)

    if buf or parts:
        parts.append("".join(buf).strip())
    return parts


def _normalise_out_band_names(
    band_names: Optional[Union[str, Sequence[str]]],
    n_outputs: int,
) -> Optional[List[str]]:
    """Validate optional output band names against the expression output count."""
    if band_names is None:
        return None
    if isinstance(band_names, str):
        resolved = [band_names]
    elif isinstance(band_names, Sequence) and not isinstance(
        band_names, (bytes, bytearray)
    ):
        resolved = [str(name) for name in band_names]
    else:
        raise rsgispy.RSGISPyException(
            "band_names must be None or a sequence of non-empty strings."
        )

    if any(not name or not name.strip() for name in resolved):
        raise rsgispy.RSGISPyException(
            "band_names must contain non-empty strings."
        )
    if len(resolved) != n_outputs:
        raise rsgispy.RSGISPyException(
            f"Length of band_names ({len(resolved)}) must match the number of "
            f"expression outputs ({n_outputs})."
        )
    return resolved


def _check_out_no_data_val(
    no_data_val: Optional[float], out_np_dtype: np.dtype
) -> Optional[float]:
    """
    Validate an optional output no-data value against the output data type.

    :param no_data_val: Requested output no-data value (or ``None``).
    :param out_np_dtype: NumPy dtype of the output raster.
    :return: ``no_data_val`` as a float, or ``None``.
    """
    if no_data_val is None:
        return None
    nd_val = float(no_data_val)
    if math.isnan(nd_val) and np.issubdtype(out_np_dtype, np.floating):
        return nd_val
    if not fits_dtype_range(nd_val, out_np_dtype):
        raise rsgispy.RSGISPyException(
            f"no_data_val ({no_data_val}) cannot be represented by the output "
            f"data type '{np.dtype(out_np_dtype).name}'."
        )
    return nd_val


def _set_out_no_data_val(dst: Any, no_data_val: Optional[float]) -> None:
    """Write ``no_data_val`` to every band of ``dst`` unless it is already set."""
    if no_data_val is None:
        return
    set_dataset_nodata(dst, no_data_val)


def _create_parser(exp: str, band_names: Sequence[str]) -> pymu_eval.Parser:
    """
    Create and validate a pymu_eval.Parser instance for the given expression and bands.

    :param exp: Mathematical expression string (muParser syntax).
    :param band_names: Variable names corresponding to the input bands.
    :return: Configured pymu_eval.Parser instance.
    """
    if not exp or not isinstance(exp, str) or not exp.strip():
        raise rsgispy.RSGISPyException(
            "Expression 'exp' must be a non-empty string."
        )

    parser = pymu_eval.Parser()
    band_name_set = set(band_names)
    if "pi" not in band_name_set:
        parser.define_const("pi", pymu_eval.MathImpl.CONST_PI)
    if "e" not in band_name_set:
        parser.define_const("e", pymu_eval.MathImpl.CONST_E)

    dummy = np.zeros((1, 1), dtype=np.float64)
    for name in band_names:
        try:
            parser.define_var(name, dummy)
        except pymu_eval.ParserError as err:
            raise rsgispy.RSGISPyException(
                f"Invalid band variable name '{name}': {err}"
            ) from err

    rsgispy.tools.maths.configure_muparser_assignment_vars(parser, exp)

    try:
        parser.set_expr(exp)
        with np.errstate(all="ignore"):
            parser.eval_multi()
    except pymu_eval.ParserError as err:
        raise rsgispy.RSGISPyException(
            f"Error parsing or evaluating expression '{exp}': {err}"
        ) from err

    return parser


def band_maths(
    output_img: str,
    exp: str,
    band_defs: Sequence[BandDefn],
    exp_band_name: bool = False,
    output_exists: bool = False,
    *,
    gdalformat: str = "GTIFF",
    datatype: int = rsgispy.TYPE_32FLOAT,
    use_pbar: bool = True,
    creation_opts: Optional[Mapping[str, Any]] = None,
    band_names: Optional[Sequence[str]] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = False,
) -> None:
    """
    Performs band math calculation using pymu-eval and rasterio.

    The syntax for the expression follows the muParser library syntax
    (supported by ``pymu_eval``). A single expression produces a 1-band output
    raster. To produce multiple output bands in a single pass, separate
    sub-expressions with top-level commas (e.g., ``"expr1, expr2, expr3"``):
    each sub-expression is written to its corresponding 1-based output band
    in order. Commas inside function calls (such as ``min(b1, b2)`` or
    ``avg(b1, b2, b3)``) remain function argument separators and do not start
    a new output band. Sub-expressions may also span multiple lines, use
    ternary conditionals (``cond ? val_true : val_false``), re-assign an input
    variable (e.g., ``b1 = b1 * 0.0001``), or define arbitrary new intermediate
    variables (e.g., ``ndvi = (nir - red) / (nir + red + 1e-6)``) for reuse
    by later sub-expressions. Note that a top-level assignment sub-expression
    both binds the variable and emits that value as its positional output band;
    to compute an intermediate without emitting an extra band, assign inside
    parentheses within a sub-expression (e.g.,
    ``"(ndvi = (nir - red) / (nir + red + 1e-6)) >= 0.5 ? 1 : 0"``).

    The input bands are read without masking, so their no-data values take
    part in the calculation unless the expression tests for them. Giving
    ``no_data_val`` writes that value as the no-data value of every output
    band, but does not change how the expression is evaluated: the expression
    must itself produce ``no_data_val`` for the pixels that should be no data
    (e.g. ``"b1 == 0 ? -9999 : b1 * 0.0001"``).

    :param output_img: Path to the output image file.
    :param exp: String containing the muParser expression to evaluate over the
                bands. Use top-level commas to separate multiple output
                sub-expressions (one output band per sub-expression).
    :param band_defs: Sequence of :class:`rsgispy.imagecalc.maths.BandDefn`
                      objects defining the inputs.
    :param exp_band_name: Optional bool specifying whether each output band
                          description should be set to its corresponding
                          expression (or sub-expression for multi-output
                          expressions) when ``band_names`` is not provided
                          (Default = False).
    :param output_exists: Optional bool specifying whether the output image already exists
                          and should be updated in place rather than overwritten (Default = False).
    :param gdalformat: GDAL/rasterio driver format for the output file
                       (Default = ``'GTIFF'``).
    :param datatype: Output data type constant from ``rsgispy.TYPE_*``
                     (Default = ``rsgispy.TYPE_32FLOAT``).
    :param use_pbar: Optional bool specifying whether to display a tqdm progress bar
                     during processing (Default = True).
    :param creation_opts: Optional mapping of GDAL creation options (e.g.,
                          ``{'COMPRESS': 'DEFLATE', 'TILED': 'YES'}``) overriding or
                          supplementing any options set via :mod:`rsgispy.imageutils.envopts`.
    :param band_names: Optional sequence of output band description names whose
                       length matches the number of outputs produced by ``exp``
                       (takes precedence over ``exp_band_name`` when supplied).
    :param no_data_val: Optional no-data value written to every output band
                        (Default = ``None``: no value is written, and an
                        existing output keeps its value). It must be
                        representable by ``datatype``; ``NaN`` is allowed for
                        floating-point outputs.
    :param calc_stats: If ``True``, calculate image statistics and pyramids
                       for the output, excluding ``no_data_val``
                       (Default = ``False``).

    .. code:: python

        import rsgispy
        import rsgispy.imagecalc.maths

        # Single-band output
        band_defns = [
            rsgispy.imagecalc.maths.BandDefn(band_name='b1', input_img='img.tif', img_band=1),
            rsgispy.imagecalc.maths.BandDefn(band_name='b2', input_img='img.tif', img_band=2),
        ]
        rsgispy.imagecalc.maths.band_maths(
            output_img='out.tif',
            exp='b1*b2',
            band_defs=band_defns,
            gdalformat='GTIFF',
            datatype=rsgispy.TYPE_32FLOAT,
        )

        # Multi-band output defining and reusing an intermediate variable
        idx_defs = [
            rsgispy.imagecalc.maths.BandDefn('red', 'sentinel2.tif', 3),
            rsgispy.imagecalc.maths.BandDefn('nir', 'sentinel2.tif', 4),
            rsgispy.imagecalc.maths.BandDefn('swir1', 'sentinel2.tif', 5),
        ]
        multi_exp = (
            "ndvi = (nir - red) / (nir + red + 1e-6), "
            "(nir - swir1) / (nir + swir1 + 1e-6), "
            "ndvi >= 0.5 ? 1 : 0"
        )
        rsgispy.imagecalc.maths.band_maths(
            output_img='indices.tif',
            exp=multi_exp,
            band_defs=idx_defs,
            band_names=['NDVI', 'NDMI', 'VegMask'],
            gdalformat='GTIFF',
            datatype=rsgispy.TYPE_32FLOAT,
        )

        # Multi-band summary using multi-argument functions (inner commas are
        # function arguments; top-level commas separate the 3 output bands)
        rsgispy.imagecalc.maths.band_maths(
            output_img='band_stats.tif',
            exp='min(red, nir, swir1), max(red, nir, swir1), avg(red, nir, swir1)',
            band_defs=idx_defs,
            band_names=['min_refl', 'max_refl', 'mean_refl'],
        )

        # The expression writes -9999 where the input is no data (0);
        # no_data_val records it in the output, and the statistics skip it
        rsgispy.imagecalc.maths.band_maths(
            output_img='ndvi.tif',
            exp='(red == 0 || nir == 0) ? -9999 : (nir - red) / (nir + red)',
            band_defs=idx_defs,
            band_names=['NDVI'],
            no_data_val=-9999,
            calc_stats=True,
        )
    """
    # pylint: disable=too-many-arguments,too-many-locals
    out_rio_dtype = rsgispy.get_rasterio_datatype(datatype)
    out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(datatype))
    out_no_data_val = _check_out_no_data_val(no_data_val, out_np_dtype)
    norm_defs = normalise_band_defs(band_defs)
    parser = _create_parser(exp, [b.band_name for b in norm_defs if b.band_name])
    n_outputs = parser.get_num_results()
    if n_outputs < 1:
        raise rsgispy.RSGISPyException(
            f"Expression '{exp}' did not produce any outputs."
        )
    out_band_names = _normalise_out_band_names(band_names, n_outputs)

    if n_outputs == 1:

        def _eval_block(block_vars: Dict[str, np.ndarray]) -> Any:
            with np.errstate(all="ignore"):
                return parser.eval(block_vars)

    else:

        def _eval_block(block_vars: Dict[str, np.ndarray]) -> np.ndarray:
            first_arr = next(iter(block_vars.values()))
            win_h, win_w = first_arr.shape
            with np.errstate(all="ignore"):
                results = parser.eval_multi(block_vars)
            bands = [
                (
                    np.full((win_h, win_w), float(res), dtype=np.float64)
                    if np.ndim(res) == 0
                    else np.asarray(res, dtype=np.float64)
                )
                for res in results
            ]
            return np.stack(bands, axis=0)

    with ExitStack() as stack:
        datasets = open_input_datasets(stack, norm_defs)
        grid = compute_spatial_intersection(datasets)
        dst = open_output_dataset(
            stack=stack,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype=out_rio_dtype,
            grid=grid,
            count=n_outputs,
            output_exists=output_exists,
            creation_opts=creation_opts,
        )
        if dst.count != n_outputs:
            raise rsgispy.RSGISPyException(
                f"Existing output image '{output_img}' has {dst.count} band(s), "
                f"but expression produces {n_outputs} output(s)."
            )
        _set_out_no_data_val(dst, out_no_data_val)
        pop_stats_on_close(
            dst, calc_stats, no_data_val=out_no_data_val, use_pbar=use_pbar
        )

        if out_band_names is not None:
            for band_idx, band_name in enumerate(out_band_names, start=1):
                dst.set_band_description(band_idx, band_name)
        elif exp_band_name:
            if n_outputs == 1:
                dst.set_band_description(1, exp)
            else:
                sub_exprs = _split_top_level_exprs(exp)
                descs = sub_exprs if len(sub_exprs) == n_outputs else [exp] * n_outputs
                for band_idx, desc in enumerate(descs, start=1):
                    dst.set_band_description(band_idx, desc)

        process_blocks(
            dst=dst,
            datasets=datasets,
            norm_defs=norm_defs,
            grid=grid,
            block_func=_eval_block,
            out_np_dtype=out_np_dtype,
            use_pbar=use_pbar,
        )


def image_math(
    input_img: str,
    output_img: str,
    exp: str,
    exp_band_name: bool = False,
    output_exists: bool = False,
    *,
    gdalformat: str = "GTIFF",
    datatype: int = rsgispy.TYPE_32FLOAT,
    use_pbar: bool = True,
    creation_opts: Optional[Mapping[str, Any]] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = False,
) -> None:
    """
    Perform image math calculations using ``pymu_eval`` and ``rasterio``.

    Produces an output image with the same number of bands as ``input_img``,
    applying the same muParser expression (using variable ``b1``) independently
    to each band. As in :func:`band_maths`, ``no_data_val`` is only written to
    the output; the expression must produce it for the no-data pixels.

    :param input_img: Path to the input image file.
    :param output_img: Path to the output image file.
    :param exp: muParser mathematical expression referencing ``b1``.
    :param exp_band_name: If True, set each output band description to ``exp`` (Default = False).
    :param output_exists: If True and ``output_img`` exists, update it in place (Default = False).
    :param gdalformat: GDAL/rasterio driver format for the output file (Default = ``'GTIFF'``).
    :param datatype: Output data type constant from ``rsgispy.TYPE_*``
                     (Default = ``rsgispy.TYPE_32FLOAT``).
    :param use_pbar: If True, display a tqdm progress bar during processing (Default = True).
    :param creation_opts: Optional mapping of GDAL creation options.
    :param no_data_val: Optional no-data value written to every output band
                        (Default = ``None``: no value is written). It must be
                        representable by ``datatype``.
    :param calc_stats: If ``True``, calculate image statistics and pyramids
                       for the output, excluding ``no_data_val``
                       (Default = ``False``).

    .. code:: python

        import rsgispy
        import rsgispy.imagecalc.maths

        rsgispy.imagecalc.maths.image_math(
            input_img="img.tif",
            output_img="out.tif",
            exp="b1 == 0 ? -9999 : b1 * 0.0001",
            gdalformat="GTIFF",
            datatype=rsgispy.TYPE_32FLOAT,
            no_data_val=-9999,
        )
    """
    # pylint: disable=too-many-arguments,too-many-locals
    ensure_file_exists(input_img)

    out_rio_dtype = rsgispy.get_rasterio_datatype(datatype)
    out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(datatype))
    out_no_data_val = _check_out_no_data_val(no_data_val, out_np_dtype)
    parser = _create_parser(exp, ["b1"])

    with ExitStack() as stack:
        try:
            src = stack.enter_context(rasterio.open(input_img, "r"))
        except Exception as err:
            raise rsgispy.RSGISPyException(
                f"Could not open input image '{input_img}': {err}"
            ) from err

        n_bands = src.count
        grid = compute_spatial_intersection({input_img: src})
        dst = open_output_dataset(
            stack=stack,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype=out_rio_dtype,
            grid=grid,
            count=n_bands,
            output_exists=output_exists,
            creation_opts=creation_opts,
        )
        _set_out_no_data_val(dst, out_no_data_val)
        pop_stats_on_close(
            dst, calc_stats, no_data_val=out_no_data_val, use_pbar=use_pbar
        )

        if exp_band_name:
            for band_idx in range(1, n_bands + 1):
                dst.set_band_description(band_idx, exp)

        for out_win in iter_dataset_windows(
            grid.height, grid.width, use_pbar=use_pbar
        ):
            win_h = int(out_win.height)
            win_w = int(out_win.width)
            in_block = src.read(window=out_win, out_dtype=np.float64)
            out_bands = [
                cast_result_block(
                    parser.eval({"b1": in_block[band_idx]}),
                    win_h,
                    win_w,
                    out_np_dtype,
                )
                for band_idx in range(n_bands)
            ]
            dst.write(np.stack(out_bands, axis=0), window=out_win)


def image_band_math(
    input_img: str,
    output_img: str,
    exp: str,
    exp_band_name: bool = False,
    output_exists: bool = False,
    *,
    gdalformat: str = "GTIFF",
    datatype: int = rsgispy.TYPE_32FLOAT,
    use_pbar: bool = True,
    creation_opts: Optional[Mapping[str, Any]] = None,
    band_names: Optional[Sequence[str]] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = False,
) -> None:
    """
    Perform image band math calculations using ``pymu_eval`` and ``rasterio``.

    Input bands are bound automatically to variables ``b1, b2, ..., bn``
    (1-based). A single expression produces a 1-band output image, while a
    comma-separated expression (e.g., ``"(b4 - b3) / (b4 + b3), (b2 - b4) / (b2 + b4)"``)
    produces one output band per sub-expression. Sub-expressions may also
    assign new intermediate variables (e.g., ``"ndvi = (b4 - b3) / (b4 + b3), ndvi > 0.4 ? 1 : 0"``)
    for reuse by later sub-expressions. As in :func:`band_maths`,
    ``no_data_val`` is only written to the output; the expression must produce
    it for the no-data pixels.

    :param input_img: Path to the input image file.
    :param output_img: Path to the output image file.
    :param exp: muParser expression referencing input bands as ``b1``, ``b2``,
                etc. Separate multiple sub-expressions with top-level commas to
                produce multiple output bands.
    :param exp_band_name: If True, set each output band description to its
                          corresponding expression/sub-expression when
                          ``band_names`` is not provided (Default = False).
    :param output_exists: If True and ``output_img`` exists, update it in place (Default = False).
    :param gdalformat: GDAL/rasterio driver format for the output file (Default = ``'GTIFF'``).
    :param datatype: Output data type constant from ``rsgispy.TYPE_*``
                     (Default = ``rsgispy.TYPE_32FLOAT``).
    :param use_pbar: If True, display a tqdm progress bar during processing (Default = True).
    :param creation_opts: Optional mapping of GDAL creation options.
    :param band_names: Optional sequence of output band description names matching
                       the number of outputs produced by ``exp``.
    :param no_data_val: Optional no-data value written to every output band
                        (Default = ``None``: no value is written). It must be
                        representable by ``datatype``.
    :param calc_stats: If ``True``, calculate image statistics and pyramids
                       for the output, excluding ``no_data_val``
                       (Default = ``False``).

    .. code:: python

        import rsgispy
        import rsgispy.imagecalc.maths

        # Single output band, -9999 where the input is no data (0)
        rsgispy.imagecalc.maths.image_band_math(
            input_img="multispec.tif",
            output_img="ndvi.tif",
            exp="(b3 == 0 || b4 == 0) ? -9999 : (b4 - b3) / (b4 + b3)",
            gdalformat="GTIFF",
            datatype=rsgispy.TYPE_32FLOAT,
            no_data_val=-9999,
            calc_stats=True,
        )

        # Multiple output bands in a single pass
        rsgispy.imagecalc.maths.image_band_math(
            input_img="multispec.tif",
            output_img="indices.tif",
            exp="(b4 - b3) / (b4 + b3), (b2 - b4) / (b2 + b4)",
            band_names=["NDVI", "NDWI"],
            gdalformat="GTIFF",
            datatype=rsgispy.TYPE_32FLOAT,
        )
    """
    # pylint: disable=too-many-arguments,too-many-locals
    ensure_file_exists(input_img)

    try:
        with rasterio.open(input_img, "r") as src:
            n_bands = src.count
    except Exception as err:
        raise rsgispy.RSGISPyException(
            f"Could not open input image '{input_img}': {err}"
        ) from err

    band_defs = [
        BandDefn(band_name=f"b{idx + 1}", input_img=input_img, img_band=idx + 1)
        for idx in range(n_bands)
    ]
    band_maths(
        output_img=output_img,
        exp=exp,
        band_defs=band_defs,
        exp_band_name=exp_band_name,
        output_exists=output_exists,
        gdalformat=gdalformat,
        datatype=datatype,
        use_pbar=use_pbar,
        creation_opts=creation_opts,
        band_names=band_names,
        no_data_val=no_data_val,
        calc_stats=calc_stats,
    )


def calc_prop_true_exp(
    exp: str,
    band_defs: Sequence[BandDefn],
    in_vld_img: Optional[str] = None,
    use_pbar: bool = True,
) -> float:
    """
    Calculate the proportion of pixels where ``exp`` evaluates to ``1``.

    :param exp: muParser expression over ``band_defs`` (``floor(result) == 1`` is true).
    :param band_defs: Sequence of :class:`rsgispy.imagecalc.maths.BandDefn` objects.
    :param in_vld_img: Optional valid-area mask image path (pixels == 1 in band 1).
    :param use_pbar: If True, display a tqdm progress bar (Default = True).
    :return: Proportion of true pixels in ``[0.0, 1.0]``.

    .. code:: python

        import rsgispy.imagecalc.maths

        band_defs = [
            rsgispy.imagecalc.maths.BandDefn("b1", "ndvi.tif", 1),
        ]
        prop = rsgispy.imagecalc.maths.calc_prop_true_exp(
            exp="b1 > 0.3 ? 1 : 0",
            band_defs=band_defs,
        )
    """
    # pylint: disable=too-many-locals
    norm_defs = normalise_band_defs(band_defs)
    parser = _create_parser(exp, [b.band_name for b in norm_defs if b.band_name])

    if in_vld_img is not None:
        ensure_file_exists(in_vld_img, label="Valid image")

    true_count = 0
    total_count = 0
    with ExitStack() as stack:
        datasets = open_input_datasets(stack, norm_defs)
        if in_vld_img is not None and in_vld_img not in datasets:
            datasets[in_vld_img] = stack.enter_context(
                rasterio.open(in_vld_img, "r")
            )
        grid = compute_spatial_intersection(datasets)

        for out_win, block_vars in iter_image_blocks(
            datasets=datasets,
            norm_defs=norm_defs,
            grid=grid,
            use_pbar=use_pbar,
        ):
            res = parser.eval(block_vars)
            win_h = int(out_win.height)
            win_w = int(out_win.width)
            if np.ndim(res) == 0:
                with np.errstate(invalid="ignore"):
                    is_one = (
                        math.floor(float(res)) == 1
                        if np.isfinite(res)
                        else False
                    )
                true_mask = np.full((win_h, win_w), is_one, dtype=bool)
            else:
                with np.errstate(invalid="ignore"):
                    true_mask = np.floor(np.asarray(res)) == 1

            if in_vld_img is not None:
                base_col, base_row = grid.offsets[in_vld_img]
                vld_win = Window(
                    base_col + int(out_win.col_off),
                    base_row + int(out_win.row_off),
                    win_w,
                    win_h,
                )
                vld_mask = datasets[in_vld_img].read(1, window=vld_win) == 1
                true_count += int(np.count_nonzero(true_mask & vld_mask))
                total_count += int(np.count_nonzero(vld_mask))
            else:
                true_count += int(np.count_nonzero(true_mask))
                total_count += int(true_mask.size)

    return float(true_count / total_count) if total_count > 0 else 0.0
