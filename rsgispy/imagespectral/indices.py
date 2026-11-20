"""
The imagespectral.indices module provides spectral curve and index summary
utilities using ``rasterio`` and ``numpy``.
"""

from contextlib import ExitStack
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import rasterio

import rsgispy
from rsgispy.imageprocess import (
    cast_result_block,
    compute_spatial_intersection,
    ensure_file_exists,
    iter_dataset_windows,
    open_output_dataset,
)

__all__ = [
    "calc_area_under_curve",
]


def calc_area_under_curve(
    input_img: str,
    output_img: str,
    band_widths: Sequence[float],
    *,
    gdalformat: str = "GTIFF",
    datatype: int = rsgispy.TYPE_32FLOAT,
    use_pbar: bool = True,
    creation_opts: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Calculate the area under the curve defined by the input image bands and
    their corresponding ``band_widths`` (e.g., spectral or temporal widths).

    For each pixel, the output equals ``sum(band_i * band_widths[i])``.

    :param input_img: Path to the input image file.
    :param output_img: Path to the output single-band image file.
    :param band_widths: Sequence of float widths matching the number of bands
                        in ``input_img``.
    :param gdalformat: GDAL/rasterio driver format for the output file
                       (Default = ``'GTIFF'``).
    :param datatype: Output data type constant from ``rsgispy.TYPE_*``
                     (Default = ``rsgispy.TYPE_32FLOAT``).
    :param use_pbar: If True, display a tqdm progress bar (Default = True).
    :param creation_opts: Optional mapping of GDAL creation options.

    .. code:: python

        import rsgispy
        import rsgispy.imagespectral.indices

        rsgispy.imagespectral.indices.calc_area_under_curve(
            input_img="hyperspectral.tif",
            output_img="auc.tif",
            band_widths=[10.0, 10.0, 10.0, 10.0],
            gdalformat="GTIFF",
            datatype=rsgispy.TYPE_32FLOAT,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    if band_widths is None:
        raise rsgispy.RSGISPyException("Input band_widths list must not be None.")
    ensure_file_exists(input_img)

    out_rio_dtype = rsgispy.get_rasterio_datatype(datatype)
    out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(datatype))
    widths_arr = np.asarray(band_widths, dtype=np.float64)
    if widths_arr.ndim != 1:
        raise rsgispy.RSGISPyException("Input band_widths must be a 1D sequence.")

    with ExitStack() as stack:
        try:
            src = stack.enter_context(rasterio.open(input_img, "r"))
        except Exception as err:
            raise rsgispy.RSGISPyException(
                f"Could not open input image '{input_img}': {err}"
            ) from err

        if src.count != widths_arr.size:
            raise rsgispy.RSGISPyException(
                f"Length of band_widths ({widths_arr.size}) must be equal to "
                f"the number of bands in the input image (N Bands: {src.count})."
            )

        grid = compute_spatial_intersection({input_img: src})
        dst = open_output_dataset(
            stack=stack,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype=out_rio_dtype,
            grid=grid,
            count=1,
            creation_opts=creation_opts,
        )

        for out_win in iter_dataset_windows(
            grid.height, grid.width, use_pbar=use_pbar
        ):
            in_block = src.read(window=out_win, out_dtype=np.float64)
            area = np.tensordot(widths_arr, in_block, axes=([0], [0]))
            out_block = cast_result_block(
                area, int(out_win.height), int(out_win.width), out_np_dtype
            )
            dst.write(out_block, 1, window=out_win)
