"""
Whole-image shift estimation and linear geotransform offset utilities for
:mod:`rsgispy.imageregistration`.
"""

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import rasterio
from rasterio.transform import Affine

import rsgispy
from rsgispy.imageprocess import (
    ensure_file_exists,
    extract_overlap_arrays,
    resolve_band_pair,
)
from rsgispy.imageregistration._helper import (
    copy_raster_with_updates,
    find_surface_extremum,
    refine_grid_subpixel,
    search_shift_surface,
)
from rsgispy.imageutils.stats import pop_img_stats
from rsgispy.tools.stats import phase_cross_correlation_2d, resolve_metric_type

__all__ = [
    "find_image_offset",
    "apply_offset_to_image",
]


def find_image_offset(
    in_ref_img: str,
    in_float_img: str,
    ref_img_bands: Optional[Sequence[int]] = None,
    flt_img_bands: Optional[Sequence[int]] = None,
    metric_type: Union[int, str] = rsgispy.METRIC_CORRELATION,
    x_search: int = 15,
    y_search: int = 15,
    sub_pxl_res: int = 4,
    hist_bins: int = 64,
    ref_no_data: Optional[float] = None,
    flt_no_data: Optional[float] = None,
    out_map_units: bool = False,
    return_details: bool = False,
) -> Union[Tuple[float, float], Dict[str, Any]]:
    """
    Estimate the global linear X/Y shift offset between ``in_float_img`` and
    ``in_ref_img`` across their overlapping region.

    Supports Pearson correlation (``rsgispy.METRIC_CORRELATION`` /
    ``'correlation'``), mutual information (``rsgispy.METRIC_MUTUAL_INFO`` /
    ``'mutual_info'``), normalised mutual information
    (``rsgispy.METRIC_NORM_MUTUAL_INFO`` / ``'norm_mutual_info'``), FFT phase
    cross-correlation (``rsgispy.METRIC_PHASE_CORR`` /
    ``'phase_correlation'``), Euclidean distance (``rsgispy.METRIC_EUCLIDEAN``),
    Manhattan distance (``rsgispy.METRIC_MANHATTAN``), and mean squared
    difference (``rsgispy.METRIC_SQDIFF``).

    By default returns ``(x_offset, y_offset)`` (matching
    ``rsgislib.imageregistration.find_image_offset`` in pixel units, or map
    units when ``out_map_units=True``). Set ``return_details=True`` to return a
    summary dictionary containing both map and pixel offsets, metric score, and
    valid pixel count.

    :param in_ref_img: Path to the reference raster image.
    :param in_float_img: Path to the floating raster image to align.
    :param ref_img_bands: 1-based band indices from ``in_ref_img`` (or ``None``
                          to use all matching bands).
    :param flt_img_bands: 1-based band indices from ``in_float_img`` (or ``None``
                          to use all matching bands).
    :param metric_type: Similarity/distance metric constant ``rsgispy.METRIC_*``
                        or name string (Default = :data:`rsgispy.METRIC_CORRELATION`).
    :param x_search: Maximum search shift in pixels along X for spatial metrics
                     (Default = ``15``).
    :param y_search: Maximum search shift in pixels along Y for spatial metrics
                     (Default = ``15``).
    :param sub_pxl_res: Sub-pixel refinement factor (e.g., ``4`` estimates shifts
                        to ``0.25`` pixels; Default = ``4``).
    :param hist_bins: Histogram bin count for mutual information metrics
                      (Default = ``64``).
    :param ref_no_data: Optional nodata override for ``in_ref_img``.
    :param flt_no_data: Optional nodata override for ``in_float_img``.
    :param out_map_units: If ``True`` (and ``return_details=False``), return
                          ``(x_offset_map, y_offset_map)`` instead of pixel
                          offsets (Default = ``False``).
    :param return_details: If ``True``, return a dictionary with
                           ``x_offset_map``, ``y_offset_map``, ``x_offset_pxl``,
                           ``y_offset_pxl``, ``metric_value``, ``metric_name``,
                           and ``n_valid_pxls`` (Default = ``False``).
    :return: ``(x_offset, y_offset)`` tuple when ``return_details=False``, or a
             diagnostics dictionary when ``return_details=True``.

    .. code:: python

        import rsgispy
        import rsgispy.imageregistration.shift as imreg_shift

        x_off, y_off = imreg_shift.find_image_offset(
            in_ref_img="ref.tif",
            in_float_img="float.tif",
            ref_img_bands=[1],
            flt_img_bands=[1],
            metric_type=rsgispy.METRIC_CORRELATION,
            x_search=10,
            y_search=10,
            sub_pxl_res=4,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    ref_path = os.fspath(in_ref_img)
    flt_path = os.fspath(in_float_img)
    ensure_file_exists(ref_path, label="Reference image")
    ensure_file_exists(flt_path, label="Floating image")
    metric_id, metric_name, maximise = resolve_metric_type(metric_type)

    if x_search < 0 or y_search < 0:
        raise rsgispy.RSGISPyException("x_search and y_search must be >= 0.")
    if sub_pxl_res < 1:
        raise rsgispy.RSGISPyException("sub_pxl_res must be >= 1.")

    with rasterio.open(ref_path, "r") as ref_src, rasterio.open(
        flt_path, "r"
    ) as flt_src:
        r_bands, f_bands = resolve_band_pair(
            ref_src, flt_src, ref_img_bands, flt_img_bands, ref_path, flt_path
        )
        ov = extract_overlap_arrays(
            ref_src,
            flt_src,
            r_bands,
            f_bands,
            ref_path,
            flt_path,
            ref_no_data=ref_no_data,
            flt_no_data=flt_no_data,
        )

    ov_transform = ov["ov_transform"]
    ref_valid = ov["ref_valid"]
    flt_valid = ov["flt_valid"]
    comb_valid = ref_valid & flt_valid
    n_valid = int(np.count_nonzero(comb_valid))

    if metric_id == rsgispy.METRIC_PHASE_CORR:
        if n_valid < 16:
            raise rsgispy.RSGISPyException(
                "Insufficient overlapping valid pixels for phase correlation."
            )
        row_shifts: List[float] = []
        col_shifts: List[float] = []
        peaks: List[float] = []
        for b_idx in range(len(r_bands)):
            r_b = ov["ref_arr"][b_idx].copy()
            f_b = ov["flt_arr"][b_idx].copy()
            r_mean = float(np.mean(r_b[ref_valid])) if np.any(ref_valid) else 0.0
            f_mean = float(np.mean(f_b[flt_valid])) if np.any(flt_valid) else 0.0
            r_b[~ref_valid] = r_mean
            f_b[~flt_valid] = f_mean
            dy, dx, pk = phase_cross_correlation_2d(
                r_b, f_b, upsample_factor=sub_pxl_res
            )
            row_shifts.append(dy)
            col_shifts.append(dx)
            peaks.append(pk)
        row_shift = float(np.mean(row_shifts))
        col_shift = float(np.mean(col_shifts))
        metric_val = float(np.mean(peaks))
    else:
        surface, counts = search_shift_surface(
            ov["ref_arr"],
            ov["flt_arr"],
            ref_valid,
            flt_valid,
            metric_id=metric_id,
            maximise=maximise,
            x_search=x_search,
            y_search=y_search,
            hist_bins=hist_bins,
        )
        iy, ix = find_surface_extremum(surface, maximise=maximise)
        iy_f, ix_f, metric_val = refine_grid_subpixel(
            surface, iy, ix, sub_pxl_res=sub_pxl_res, maximise=maximise
        )
        row_shift = iy_f - float(y_search)
        col_shift = ix_f - float(x_search)
        n_valid = int(counts[iy, ix])

    x_offset_map = float(col_shift * ov_transform.a)
    y_offset_map = float(row_shift * ov_transform.e)
    x_offset_pxl = float(x_offset_map / abs(ov_transform.a))
    y_offset_pxl = float(y_offset_map / abs(ov_transform.e))

    if return_details:
        return {
            "x_offset_map": x_offset_map,
            "y_offset_map": y_offset_map,
            "x_offset_pxl": x_offset_pxl,
            "y_offset_pxl": y_offset_pxl,
            "metric_value": float(metric_val),
            "metric_name": metric_name,
            "n_valid_pxls": n_valid,
        }
    if out_map_units:
        return x_offset_map, y_offset_map
    return x_offset_pxl, y_offset_pxl


def apply_offset_to_image(
    input_img: str,
    output_img: str,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
    in_pxl_units: bool = False,
    calc_stats: bool = True,
) -> None:
    """
    Apply a linear X, Y translation offset to the image geotransform header
    without resampling pixel values.

    :param input_img: Path to the input raster image.
    :param output_img: Path to the output shifted raster image.
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param datatype: Optional ``rsgispy.TYPE_*`` output data type (``None``
                     preserves input pixel dtype).
    :param x_offset: X offset to add to the image origin (in map units by
                     default, or pixels if ``in_pxl_units=True``).
    :param y_offset: Y offset to add to the image origin (in map units by
                     default, or pixels if ``in_pxl_units=True``).
    :param in_pxl_units: If ``True``, interpret ``(x_offset, y_offset)`` as
                         pixel offsets rather than map units (Default = ``False``).
    :param calc_stats: If ``True``, compute output image statistics and pyramids
                       (Default = ``True``).

    .. code:: python

        import rsgispy
        import rsgispy.imageregistration.shift as imreg_shift

        imreg_shift.apply_offset_to_image(
            input_img="float.tif",
            output_img="shifted.tif",
            gdalformat="GTIFF",
            datatype=rsgispy.TYPE_32FLOAT,
            x_offset=-20.0,
            y_offset=30.0,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    in_path = os.fspath(input_img)
    out_path = os.fspath(output_img)
    ensure_file_exists(in_path, label="Input image")

    with rasterio.open(in_path, "r") as src:
        t = src.transform
        if in_pxl_units:
            dx_map = float(x_offset) * abs(t.a)
            dy_map = float(y_offset) * abs(t.e)
        else:
            dx_map = float(x_offset)
            dy_map = float(y_offset)
        new_transform = Affine(t.a, t.b, t.c + dx_map, t.d, t.e, t.f + dy_map)

    nodata_val = copy_raster_with_updates(
        in_path,
        out_path,
        gdalformat=gdalformat,
        datatype=datatype,
        new_transform=new_transform,
    )

    if calc_stats:
        pop_img_stats(
            out_path,
            no_data_val=None if nodata_val is None else float(nodata_val),
            calc_pyramids=True,
            use_pbar=False,
        )
