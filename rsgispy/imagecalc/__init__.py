"""
The imagecalc module contains functions for performing calculations on raster images.
"""

from contextlib import ExitStack
import os
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np
import pymu_eval
import rasterio

import rsgispy
from rsgispy.imageprocess import (
    BandDefn,
    compute_spatial_intersection,
    iter_single_image_band_blocks,
    normalise_band_defs,
    open_input_datasets,
    open_output_dataset,
    process_blocks,
)

__all__ = ["BandDefn", "band_maths", "band_math", "calc_basic_img_band_stats"]


def _create_parser(exp: str, band_names: Sequence[str]) -> pymu_eval.Parser:
    """
    Create and validate a pymu_eval.Parser instance for the given expression and bands.

    :param exp: Mathematical expression string (muParser syntax).
    :param band_names: Variable names corresponding to the input bands.
    :return: Configured pymu_eval.Parser instance.
    """
    if not exp or not isinstance(exp, str) or not exp.strip():
        raise rsgispy.RSGISPyException("Expression 'exp' must be a non-empty string.")

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

    try:
        parser.set_expr(exp)
        with np.errstate(all="ignore"):
            parser.eval()
    except pymu_eval.ParserError as err:
        raise rsgispy.RSGISPyException(
            f"Error parsing or evaluating expression '{exp}': {err}"
        ) from err

    return parser


def band_maths(
    output_img: str,
    exp: str,
    gdalformat: str,
    datatype: int,
    band_defs: Sequence[BandDefn],
    exp_band_name: bool = False,
    output_exists: bool = False,
    use_pbar: bool = True,
    creation_opts: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Performs band math calculation using pymu-eval and rasterio.

    The syntax for the expression follows the muParser library syntax
    (supported by ``pymu_eval``).

    :param output_img: Path to the output image file.
    :param exp: String containing the muParser expression to evaluate over the bands.
    :param gdalformat: GDAL/rasterio driver format for the output file (e.g., 'GTIFF', 'KEA').
    :param datatype: Output data type constant from ``rsgispy.TYPE_*`` (or ``rsgislib.TYPE_*``).
    :param band_defs: Sequence of :class:`rsgispy.imagecalc.BandDefn` objects defining the inputs.
    :param exp_band_name: Optional bool specifying whether the output band description
                          should be set to the expression string (Default = False).
    :param output_exists: Optional bool specifying whether the output image already exists
                          and should be updated in place rather than overwritten (Default = False).
    :param use_pbar: Optional bool specifying whether to display a tqdm progress bar
                     during processing (Default = True).
    :param creation_opts: Optional mapping of GDAL creation options (e.g.,
                          ``{'COMPRESS': 'DEFLATE', 'TILED': 'YES'}``) overriding or
                          supplementing any options set via :mod:`rsgispy.imageutils`.

    .. code:: python

        import rsgispy
        import rsgispy.imagecalc

        band_defns = list()
        band_defns.append(
            rsgispy.imagecalc.BandDefn(band_name='b1', input_img='img.kea', img_band=1)
        )
        band_defns.append(
            rsgispy.imagecalc.BandDefn(band_name='b2', input_img='img.kea', img_band=2)
        )
        rsgispy.imagecalc.band_maths(
            output_img='out.tif',
            exp='b1*b2',
            gdalformat='GTIFF',
            datatype=rsgispy.TYPE_32FLOAT,
            band_defs=band_defns,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    out_rio_dtype = rsgispy.get_rasterio_datatype(datatype)
    out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(datatype))
    norm_defs = normalise_band_defs(band_defs)
    parser = _create_parser(exp, [b.band_name for b in norm_defs if b.band_name])

    with ExitStack() as stack:
        datasets = open_input_datasets(stack, norm_defs)
        grid = compute_spatial_intersection(datasets)
        dst = open_output_dataset(
            stack=stack,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype=out_rio_dtype,
            grid=grid,
            count=1,
            output_exists=output_exists,
            creation_opts=creation_opts,
        )

        if exp_band_name:
            dst.set_band_description(1, exp)

        process_blocks(
            dst=dst,
            datasets=datasets,
            norm_defs=norm_defs,
            grid=grid,
            block_func=parser.eval,
            out_np_dtype=out_np_dtype,
            use_pbar=use_pbar,
        )


# Alias for backwards compatibility with rsgislib.imagecalc.band_math
band_math = band_maths


def calc_basic_img_band_stats(
    input_img: str,
    no_data_val: Optional[float] = None,
    use_no_data: bool = True,
    use_pbar: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the minimum, maximum, mean, and standard deviation values
    for each band in the input image using block-wise iteration.

    :param input_img: Path to the input image file.
    :param no_data_val: Optional no-data value. If ``None`` and ``use_no_data`` is True,
                        reads the no-data value from the image header.
    :param use_no_data: Whether to exclude the no-data value from statistics (Default = True).
    :param use_pbar: Whether to display a tqdm progress bar (Default = True).
    :return: Tuple of ``(min_vals, max_vals, mean_vals, stddev_vals)`` NumPy arrays.
    """
    # pylint: disable=too-many-locals,too-many-branches
    if not os.path.exists(input_img):
        raise rsgispy.RSGISPyException(
            f"Input image file does not exist: '{input_img}'."
        )

    try:
        with rasterio.open(input_img, "r") as src:
            hdr_nodata = src.nodata
            n_bands = src.count
            rsgis_dt = rsgispy.get_rsgis_datatype(src.dtypes[0])
    except Exception as err:
        raise rsgispy.RSGISPyException(
            f"Could not open input image '{input_img}': {err}"
        ) from err

    if use_no_data and no_data_val is None:
        no_data_val = hdr_nodata
        if no_data_val is None:
            raise rsgispy.RSGISPyException(
                "A no data value needs to be specified either passed to the "
                "function or read from the input image header."
            )

    np_dt = rsgispy.get_numpy_datatype(rsgis_dt)
    min_vals = np.zeros(n_bands, dtype=np_dt)
    max_vals = np.zeros(n_bands, dtype=np_dt)
    sum_vals = np.zeros(n_bands, dtype=np.float64)
    n_vals = np.zeros(n_bands, dtype=np.int64)
    first_arr = np.ones(n_bands, dtype=bool)

    for _, block_arrays in iter_single_image_band_blocks(
        input_img=input_img, use_pbar=use_pbar
    ):
        for idx, band_arr in enumerate(block_arrays):
            data = band_arr.ravel()
            if use_no_data and no_data_val is not None:
                data = data[data != no_data_val]
            data = data[np.isfinite(data)]
            if data.size > 0:
                d_min = data.min()
                d_max = data.max()
                if first_arr[idx]:
                    min_vals[idx] = d_min
                    max_vals[idx] = d_max
                    first_arr[idx] = False
                else:
                    if d_min < min_vals[idx]:
                        min_vals[idx] = d_min
                    if d_max > max_vals[idx]:
                        max_vals[idx] = d_max
                sum_vals[idx] += float(np.sum(data))
                n_vals[idx] += data.size

    mean_vals = np.where(n_vals > 0, sum_vals / np.maximum(n_vals, 1), 0.0)
    sq_diff_vals = np.zeros(n_bands, dtype=np.float64)

    for _, block_arrays in iter_single_image_band_blocks(
        input_img=input_img, use_pbar=use_pbar
    ):
        for idx, band_arr in enumerate(block_arrays):
            data = band_arr.ravel()
            if use_no_data and no_data_val is not None:
                data = data[data != no_data_val]
            data = data[np.isfinite(data)]
            if data.size > 0:
                sq_diff_vals[idx] += float(
                    np.sum(np.power(data - mean_vals[idx], 2))
                )

    stddev_vals = np.where(
        n_vals > 0, np.sqrt(sq_diff_vals / np.maximum(n_vals, 1)), 0.0
    )

    return min_vals, max_vals, mean_vals, stddev_vals
