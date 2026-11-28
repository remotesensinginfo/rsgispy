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
    build_valid_cube_mask,
    cast_result_block,
    compute_spatial_intersection,
    ensure_file_exists,
    iter_dataset_windows,
    open_output_dataset,
)

__all__ = [
    "calc_area_under_curve",
]


#: ``method`` values of :func:`calc_area_under_curve`.
AUC_METHODS = ("trapezoid", "rectangle")


def _auc_band_centres(
    band_widths: Optional[Sequence[float]],
    band_centres: Optional[Sequence[float]],
    n_bands: int,
) -> np.ndarray:
    """
    x positions of the bands for the trapezoid rule: ``band_centres`` if
    given, otherwise the centres of contiguous bands of ``band_widths``
    (``x_0 = w_0 / 2``, ``x_i = x_(i-1) + (w_(i-1) + w_i) / 2``).
    """
    if band_centres is not None:
        centres = np.asarray(band_centres, dtype=np.float64)
    else:
        widths = np.asarray(band_widths, dtype=np.float64)
        centres = np.cumsum(widths) - (widths / 2.0)
    if centres.ndim != 1 or centres.size != n_bands:
        raise rsgispy.RSGISPyException(
            f"band_centres must be a 1D sequence with one value per band "
            f"({n_bands})."
        )
    if np.any(np.diff(centres) <= 0):
        raise rsgispy.RSGISPyException("band_centres must be strictly increasing.")
    return centres


def calc_area_under_curve(
    input_img: str,
    output_img: str,
    band_widths: Optional[Sequence[float]],
    *,
    band_centres: Optional[Sequence[float]] = None,
    method: str = "trapezoid",
    no_data_val: Optional[float] = None,
    gdalformat: str = "GTIFF",
    datatype: int = rsgispy.TYPE_32FLOAT,
    use_pbar: bool = True,
    creation_opts: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Calculate the area under the curve defined by the input image bands (a
    spectral curve for reflectance data, or a temporal curve).

    * ``method='trapezoid'`` (default): the trapezoid rule
      (:func:`numpy.trapezoid`) between the band positions, which are
      ``band_centres`` if given (e.g. wavelengths), otherwise the centres of
      contiguous bands of ``band_widths``.
    * ``method='rectangle'``: ``sum(band_i * band_widths[i])``, as in
      RSGISLib's ``calc_area_under_curve`` (versions of rsgispy before 0.5
      always used this).

    A pixel is valid only when every band is finite and not equal to the
    no-data value; other pixels are written as the no-data value.

    :param input_img: Path to the input image file.
    :param output_img: Path to the output single-band image file.
    :param band_widths: Sequence of float widths matching the number of bands
                        in ``input_img`` (may be ``None`` with
                        ``method='trapezoid'`` when ``band_centres`` is given).
    :param band_centres: Optional strictly increasing band positions (e.g.
                         centre wavelengths) for the trapezoid rule.
    :param method: ``'trapezoid'`` (Default) or ``'rectangle'``.
    :param no_data_val: Optional no-data value of the input bands and the
                        output (Default = ``None``: the input header no-data
                        value; without one only non-finite values are
                        masked and no output no-data value is set).
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
            band_widths=None,
            band_centres=[490.0, 560.0, 665.0, 842.0],
            gdalformat="GTIFF",
            datatype=rsgispy.TYPE_32FLOAT,
        )
    """
    # pylint: disable=too-many-arguments,too-many-locals
    if method not in AUC_METHODS:
        raise rsgispy.RSGISPyException(
            f"method must be one of {AUC_METHODS}, got {method!r}."
        )
    if band_widths is None and (method == "rectangle" or band_centres is None):
        raise rsgispy.RSGISPyException(
            "Input band_widths list must not be None (unless band_centres is "
            "given with method='trapezoid')."
        )
    ensure_file_exists(input_img)

    out_rio_dtype = rsgispy.get_rasterio_datatype(datatype)
    out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(datatype))
    widths_arr = (
        None if band_widths is None else np.asarray(band_widths, dtype=np.float64)
    )
    if widths_arr is not None and widths_arr.ndim != 1:
        raise rsgispy.RSGISPyException("Input band_widths must be a 1D sequence.")

    with ExitStack() as stack:
        try:
            src = stack.enter_context(rasterio.open(input_img, "r"))
        except Exception as err:
            raise rsgispy.RSGISPyException(
                f"Could not open input image '{input_img}': {err}"
            ) from err

        if widths_arr is not None and src.count != widths_arr.size:
            raise rsgispy.RSGISPyException(
                f"Length of band_widths ({widths_arr.size}) must be equal to "
                f"the number of bands in the input image (N Bands: {src.count})."
            )
        centres = (
            _auc_band_centres(band_widths, band_centres, src.count)
            if method == "trapezoid"
            else None
        )
        eff_nd = no_data_val if no_data_val is not None else src.nodata

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
        if eff_nd is not None:
            dst.nodata = eff_nd

        for out_win in iter_dataset_windows(
            grid.height, grid.width, use_pbar=use_pbar
        ):
            in_block = src.read(window=out_win, out_dtype=np.float64)
            if centres is not None:
                area = np.trapezoid(in_block, x=centres, axis=0)
            else:
                area = np.tensordot(widths_arr, in_block, axes=([0], [0]))
            valid = build_valid_cube_mask(in_block, eff_nd)
            if eff_nd is not None:
                area = np.where(valid, area, float(eff_nd))
            else:
                area = np.where(valid, area, np.nan)
            out_block = cast_result_block(
                area, int(out_win.height), int(out_win.width), out_np_dtype
            )
            dst.write(out_block, 1, window=out_win)
