"""
The imagecalc module contains functions for performing calculations on raster images.
"""

from contextlib import ExitStack
from typing import Sequence

import numpy as np
import pymu_eval

import rsgispy
from rsgispy.imageprocess import (
    BandDefn,
    compute_spatial_intersection,
    normalise_band_defs,
    open_input_datasets,
    open_output_dataset,
    process_blocks,
)

__all__ = ["BandDefn", "band_maths", "band_math"]


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
