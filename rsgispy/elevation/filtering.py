"""
The elevation.filtering module provides DEM filtering, depression filling,
and detrending functions using ``rasterio`` and ``numpy``.
"""

from contextlib import ExitStack
import heapq
import math
import os
from typing import Optional

import numpy as np
import rasterio
from rasterio.transform import Affine

import rsgispy
from rsgispy.elevation._helpers import (
    read_window_with_halo as _read_window_with_halo,
    valid_mask as _valid_mask,
)
from rsgispy.imageprocess import (
    IntersectionGrid,
    create_output_from_dataset,
    ensure_file_exists,
    iter_dataset_windows,
    open_output_dataset,
)
import rsgispy.imageutils.warp
import rsgispy.tools.files
import rsgispy.tools.strings

__all__ = [
    "dtm_aspect_median_filter",
    "fill_dem_soille_gratin_1994",
    "plane_fit_detreat_dem",
    "resampling_detread_dem",
]


def dtm_aspect_median_filter(
    in_dem_img: str,
    in_aspect_img: str,
    output_img: str,
    aspect_range: float,
    win_h_size: int,
    gdalformat: str,
) -> None:
    """
    Filter a DTM for noise using a median filter restricted to neighbouring pixels
    whose aspect lies within ``aspect_range`` degrees of the centre pixel.

    Falls back to the full window median of valid DTM pixels when no neighbours
    match the aspect criterion, or ``NaN`` if no valid pixels exist in the window.

    :param in_dem_img: Path to the input DTM image file.
    :param in_aspect_img: Path to the input aspect image file (in degrees).
    :param output_img: Path to the output filtered DTM image file.
    :param aspect_range: Angular tolerance around the centre pixel aspect (degrees).
    :param win_h_size: Half-window size in pixels (full window is ``2 * win_h_size + 1``).
    :param gdalformat: GDAL/rasterio driver format for the output file.

    .. code:: python

        import rsgispy.elevation.filtering

        rsgispy.elevation.filtering.dtm_aspect_median_filter(
            in_dem_img="dem.tif",
            in_aspect_img="aspect.tif",
            output_img="dem_filt.tif",
            aspect_range=20.0,
            win_h_size=2,
            gdalformat="GTIFF",
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    if win_h_size < 1:
        raise rsgispy.RSGISPyException("win_h_size must be at least 1.")
    if aspect_range < 0.0:
        raise rsgispy.RSGISPyException("aspect_range must be non-negative.")

    ensure_file_exists(in_dem_img, "Input DEM image")
    ensure_file_exists(in_aspect_img, "Input aspect image")

    win_size = (win_h_size * 2) + 1
    with ExitStack() as stack:
        dem_src = stack.enter_context(rasterio.open(in_dem_img, "r"))
        asp_src = stack.enter_context(rasterio.open(in_aspect_img, "r"))
        if dem_src.width != asp_src.width or dem_src.height != asp_src.height:
            raise rsgispy.RSGISPyException(
                "Input DEM and aspect images must have matching dimensions."
            )

        no_data_val = dem_src.nodata
        dst = create_output_from_dataset(
            stack=stack,
            src=dem_src,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype="float32",
            count=1,
        )

        for out_win in iter_dataset_windows(dem_src.height, dem_src.width):
            win_h = int(out_win.height)
            win_w = int(out_win.width)

            dem_halo = _read_window_with_halo(dem_src, 1, out_win, win_h_size)
            asp_halo = _read_window_with_halo(asp_src, 1, out_win, win_h_size)

            dem_wins = np.lib.stride_tricks.sliding_window_view(
                dem_halo, (win_size, win_size)
            ).reshape(win_h, win_w, -1)
            asp_wins = np.lib.stride_tricks.sliding_window_view(
                asp_halo, (win_size, win_size)
            ).reshape(win_h, win_w, -1)

            center_asp = asp_halo[
                win_h_size : win_h_size + win_h,
                win_h_size : win_h_size + win_w,
            ][..., None]
            ang_diff = np.abs(
                np.mod((asp_wins - center_asp) + 180.0, 360.0) - 180.0
            )
            asp_match = np.isfinite(ang_diff) & (ang_diff <= aspect_range)
            dem_valid = _valid_mask(dem_wins, no_data_val)

            sel_match = dem_valid & asp_match
            has_match = np.any(sel_match, axis=2)
            has_any = np.any(dem_valid, axis=2)

            out_blk = np.full((win_h, win_w), np.nan, dtype=np.float32)
            if np.any(has_match):
                matched_vals = np.where(
                    sel_match[has_match], dem_wins[has_match], np.nan
                )
                out_blk[has_match] = np.nanmedian(matched_vals, axis=1).astype(
                    np.float32
                )
            fallback_mask = (~has_match) & has_any
            if np.any(fallback_mask):
                fb_vals = np.where(
                    dem_valid[fallback_mask], dem_wins[fallback_mask], np.nan
                )
                out_blk[fallback_mask] = np.nanmedian(fb_vals, axis=1).astype(
                    np.float32
                )

            dst.write(out_blk, 1, window=out_win)


def fill_dem_soille_gratin_1994(
    in_dem_img: str,
    in_vld_img: str,
    output_img: str,
    gdalformat: str,
) -> None:
    """
    Fill local minima/depressions in an integer DEM using the Soille & Gratin
    (1994) hierarchical reconstruction algorithm.

    Reference: Soille, P., and Gratin, C. (1994). An efficient algorithm for
    drainage network extraction on DEMs. *Journal of Visual Communication and
    Image Representation*, 5(2), 181-189.

    :param in_dem_img: Path to the input integer-dtype single-band DEM image.
    :param in_vld_img: Path to a binary mask image specifying valid pixels (``1`` = valid).
    :param output_img: Path to the output filled DEM image file.
    :param gdalformat: GDAL/rasterio driver format for the output file.

    .. code:: python

        import rsgispy.elevation.filtering

        rsgispy.elevation.filtering.fill_dem_soille_gratin_1994(
            in_dem_img="dem_int.tif",
            in_vld_img="valid_mask.tif",
            output_img="dem_filled.tif",
            gdalformat="GTIFF",
        )
    """
    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    ensure_file_exists(in_dem_img, "Input DEM image")
    ensure_file_exists(in_vld_img, "Input valid mask image")

    with ExitStack() as stack:
        dem_src = stack.enter_context(rasterio.open(in_dem_img, "r"))
        vld_src = stack.enter_context(rasterio.open(in_vld_img, "r"))

        if dem_src.count != 1:
            raise rsgispy.RSGISPyException(
                "The image to be filled should only have 1 image band."
            )
        rsgis_dt = rsgispy.get_rsgis_datatype(dem_src.dtypes[0])
        out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(rsgis_dt))
        if not np.issubdtype(out_np_dtype, np.integer):
            raise rsgispy.RSGISPyException(
                "Input image must be of an integer data type."
            )
        if dem_src.width != vld_src.width or dem_src.height != vld_src.height:
            raise rsgispy.RSGISPyException(
                "Input DEM and valid area mask images must have matching dimensions."
            )

        dem = dem_src.read(1)
        vld = vld_src.read(1) == 1
        dem_nodata = dem_src.nodata

        stat_mask = vld.copy()
        if dem_nodata is not None:
            stat_mask &= dem != dem_nodata
        valid_vals = dem[stat_mask]
        if valid_vals.size == 0:
            raise rsgispy.RSGISPyException(
                "No valid DEM pixels found within the valid area mask."
            )

        min_val = int(valid_vals.min())
        max_val = int(valid_vals.max())
        mean_val = float(valid_vals.mean())
        std_val = float(valid_vals.std())
        if (mean_val - std_val) > min_val:
            border_val = int(math.floor((mean_val - std_val) + 0.5))
        else:
            border_val = int(math.floor(mean_val + 0.5))

        fill_nodata = int(dem_nodata) if dem_nodata is not None else 0
        out = np.full(dem.shape, fill_nodata, dtype=np.int64)
        out[vld] = max_val

        # Identify invalid pixels adjacent (8-connected) to valid region
        vld_pad = np.pad(vld, 1, mode="constant", constant_values=False)
        vld_nbr_wins = np.lib.stride_tricks.sliding_window_view(vld_pad, (3, 3))
        boundary = (~vld) & np.any(vld_nbr_wins, axis=(2, 3))

        heap: list = []
        bnd_r, bnd_c = np.where(boundary)
        if bnd_r.size > 0:
            out[bnd_r, bnd_c] = border_val
            for r_idx, c_idx in zip(bnd_r.tolist(), bnd_c.tolist()):
                heap.append((border_val, r_idx, c_idx))
        else:
            height, width = dem.shape
            edge_coords = (
                [(0, c) for c in range(width)]
                + [(height - 1, c) for c in range(width)]
                + [(r, 0) for r in range(height)]
                + [(r, width - 1) for r in range(height)]
            )
            for r_idx, c_idx in set(edge_coords):
                out[r_idx, c_idx] = border_val
                heap.append((border_val, r_idx, c_idx))

        heapq.heapify(heap)
        height, width = dem.shape
        while heap:
            h_cur, r_cur, c_cur = heapq.heappop(heap)
            r_min = max(0, r_cur - 1)
            r_max = min(height, r_cur + 2)
            c_min = max(0, c_cur - 1)
            c_max = min(width, c_cur + 2)
            for nr in range(r_min, r_max):
                for nc in range(c_min, c_max):
                    if (nr != r_cur or nc != c_cur) and vld[nr, nc] and out[nr, nc] == max_val:
                        new_val = max(h_cur, int(dem[nr, nc]))
                        out[nr, nc] = new_val
                        if new_val < max_val:
                            heapq.heappush(heap, (new_val, nr, nc))

        dst = create_output_from_dataset(
            stack=stack,
            src=dem_src,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype=rsgispy.get_rasterio_datatype(rsgis_dt),
            count=1,
        )
        if dem_nodata is not None:
            dst.nodata = dem_nodata
        dst.write(out.astype(out_np_dtype), 1)


def plane_fit_detreat_dem(
    input_img: str,
    output_img: str,
    gdalformat: str,
    win_size: int,
) -> None:
    """
    Detrend a DEM by fitting a local least-squares plane ``z = a*x + b*y + c``
    over each ``win_size x win_size`` window and outputting the fitted centre
    elevation ``c``.

    :param input_img: Path to the input DEM image file.
    :param output_img: Path to the output detrended DEM image file.
    :param gdalformat: GDAL/rasterio driver format for the output file.
    :param win_size: Odd window size in pixels (``>= 3``) for plane fitting.

    .. code:: python

        import rsgispy.elevation.filtering

        rsgispy.elevation.filtering.plane_fit_detreat_dem(
            input_img="dem.tif",
            output_img="dem_detrend.tif",
            gdalformat="GTIFF",
            win_size=11,
        )
    """
    # pylint: disable=too-many-locals
    if win_size < 3 or win_size % 2 == 0:
        raise rsgispy.RSGISPyException(
            "win_size must be an odd integer greater than or equal to 3."
        )
    ensure_file_exists(input_img)

    half = win_size // 2
    offs = np.arange(-half, half + 1, dtype=np.float64)
    gx, gy = np.meshgrid(offs, offs)
    flat_x = gx.ravel()
    flat_y = gy.ravel()
    flat_xx = flat_x * flat_x
    flat_yy = flat_y * flat_y
    flat_xy = flat_x * flat_y

    with ExitStack() as stack:
        src = stack.enter_context(rasterio.open(input_img, "r"))
        no_data_val = src.nodata
        dst = create_output_from_dataset(
            stack=stack,
            src=src,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype="float32",
            count=1,
        )

        for out_win in iter_dataset_windows(src.height, src.width):
            win_h = int(out_win.height)
            win_w = int(out_win.width)
            dem_halo = _read_window_with_halo(src, 1, out_win, pad=half)
            wins = np.lib.stride_tricks.sliding_window_view(
                dem_halo, (win_size, win_size)
            ).reshape(win_h, win_w, -1)

            valid = _valid_mask(wins, no_data_val).astype(np.float64)
            z_clean = np.where(valid > 0.0, wins, 0.0)

            s_n = valid.sum(axis=2)
            s_x = valid @ flat_x
            s_y = valid @ flat_y
            s_xx = valid @ flat_xx
            s_yy = valid @ flat_yy
            s_xy = valid @ flat_xy
            s_xz = (z_clean * flat_x).sum(axis=2)
            s_yz = (z_clean * flat_y).sum(axis=2)
            s_z = z_clean.sum(axis=2)

            # Solve 3x3 normal equations for c = (c0*s_xz + c1*s_yz + c2*s_z) / det
            det = (
                s_xx * ((s_yy * s_n) - (s_y * s_y))
                - s_xy * ((s_xy * s_n) - (s_x * s_y))
                + s_x * ((s_xy * s_y) - (s_yy * s_x))
            )
            num_c = (
                ((s_xy * s_y) - (s_x * s_yy)) * s_xz
                + ((s_xy * s_x) - (s_xx * s_y)) * s_yz
                + ((s_xx * s_yy) - (s_xy * s_xy)) * s_z
            )
            out_c = np.divide(
                num_c,
                det,
                out=np.zeros_like(det, dtype=np.float64),
                where=(s_n > 0) & (np.abs(det) > 1e-12),
            )
            dst.write(out_c.astype(np.float32), 1, window=out_win)


def resampling_detread_dem(
    input_img: str,
    output_img: str,
    resmp_res: float,
    gdalformat: str,
    no_data_val: Optional[float] = None,
    datatype: Optional[int] = None,
    tmp_dir: Optional[str] = None,
    low_resmp_mthd: int = rsgispy.INTERP_AVERAGE,
    up_smpl_mthd: int = rsgispy.INTERP_CUBICSPLINE,
) -> None:
    """
    Detrend a DEM by down-sampling to ``resmp_res`` and then up-sampling back to
    the original grid resolution.

    :param input_img: Input DEM image path (band 1 is used).
    :param output_img: Output detrended DEM image path.
    :param resmp_res: Intermediate coarse pixel resolution (positive spatial units).
    :param gdalformat: GDAL/rasterio driver format for the output file.
    :param no_data_val: DEM no-data value (read from ``input_img`` header if ``None``).
    :param datatype: Output ``rsgispy.TYPE_*`` constant (inferred from ``input_img`` if ``None``).
    :param tmp_dir: Optional directory for intermediate rasters (temporary directory
                    created and cleaned up automatically if ``None``).
    :param low_resmp_mthd: Interpolation method for down-sampling
                           (Default = :data:`rsgispy.INTERP_AVERAGE`).
    :param up_smpl_mthd: Interpolation method for up-sampling
                         (Default = :data:`rsgispy.INTERP_CUBICSPLINE`).

    .. code:: python

        import rsgispy.elevation.filtering

        rsgispy.elevation.filtering.resampling_detread_dem(
            input_img="dem.tif",
            output_img="dem_detrend.tif",
            resmp_res=100.0,
            gdalformat="GTIFF",
            no_data_val=-9999.0,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    if resmp_res <= 0.0:
        raise rsgispy.RSGISPyException("resmp_res must be greater than 0.")
    ensure_file_exists(input_img)

    with rasterio.open(input_img, "r") as src:
        if datatype is None:
            datatype = rsgispy.get_rsgis_datatype(src.dtypes[0])
        if no_data_val is None:
            no_data_val = src.nodata
        if no_data_val is None:
            raise rsgispy.RSGISPyException(
                "A no data value has not been defined. Either set in image header "
                "or pass to function."
            )
        left, bottom, right, top = src.bounds
        src_crs = src.crs

    coord_width = right - left
    coord_height = top - bottom
    img_width = int(math.ceil(coord_width / resmp_res) + 1)
    img_height = int(math.ceil(coord_height / resmp_res) + 1)

    int_base_name = rsgispy.tools.files.get_file_basename(input_img)
    uid_str = rsgispy.tools.strings.uid_generator()
    out_rio_dtype = rsgispy.get_rasterio_datatype(datatype)
    out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(datatype))

    with rsgispy.tools.files.work_dir_context(tmp_dir) as work_dir:
        base_img = os.path.join(work_dir, f"{int_base_name}_base_{uid_str}.tif")
        low_res_img = os.path.join(work_dir, f"{int_base_name}_low_res_{uid_str}.tif")

        coarse_transform = Affine(
            float(resmp_res), 0.0, float(left), 0.0, -float(resmp_res), float(top)
        )
        coarse_grid = IntersectionGrid(
            crs=src_crs,
            transform=coarse_transform,
            width=img_width,
            height=img_height,
            offsets={},
        )
        with ExitStack() as base_stack:
            base_dst = open_output_dataset(
                stack=base_stack,
                output_img=base_img,
                gdalformat="GTIFF",
                out_rio_dtype=out_rio_dtype,
                grid=coarse_grid,
                count=1,
            )
            base_dst.nodata = no_data_val
            base_dst.write(
                np.full((img_height, img_width), 0, dtype=out_np_dtype), 1
            )

        rsgispy.imageutils.warp.resample_img_to_match(
            base_img,
            input_img,
            low_res_img,
            gdalformat="GTIFF",
            interp_method=low_resmp_mthd,
            datatype=datatype,
            no_data_val=no_data_val,
            multicore=False,
        )

        rsgispy.imageutils.warp.resample_img_to_match(
            input_img,
            low_res_img,
            output_img,
            gdalformat=gdalformat,
            interp_method=up_smpl_mthd,
            datatype=datatype,
            no_data_val=no_data_val,
            multicore=False,
        )
