"""
The elevation.filtering module provides DEM filtering, depression filling,
and detrending functions using ``rasterio`` and ``numpy``.
"""

from contextlib import ExitStack
import heapq
import math
import os
import tempfile
from typing import Optional

import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import Affine
from rasterio.vrt import WarpedVRT

import rsgispy
from rsgispy._numba import (
    HAS_NUMBA,
    NUMBA_INSTALLED,
    numba_enabled,
    warn_numba_fallback,
)
from rsgispy.elevation._hydro_kernels import priority_flood_kernel
from rsgispy.imageprocess import (
    IntersectionGrid,
    build_valid_mask as _valid_mask,
    cast_result_block,
    create_output_from_dataset,
    ensure_file_exists,
    iter_dataset_windows,
    open_output_dataset,
    read_window_with_halo as _read_window_with_halo,
)
from rsgispy.imageutils.stats import pop_img_stats

__all__ = [
    "dtm_aspect_median_filter",
    "fill_dem_soille_gratin_1994",
    "plane_fit_detrend_dem",
    "resampling_detrend_dem",
]


def dtm_aspect_median_filter(
    in_dem_img: str,
    in_aspect_img: str,
    output_img: str,
    aspect_range: float,
    win_h_size: int,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
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
    :param gdalformat: GDAL/rasterio driver format for the output file
                       (Default = ``'GTIFF'``).
    :param datatype: Optional ``rsgispy.TYPE_*`` output data type constant
                     (Default = ``rsgispy.TYPE_32FLOAT`` when ``None``).
    :param no_data_val: Optional input DTM no-data value (read from header
                        when ``None``).
    :param calc_stats: If ``True``, populate band statistics and overview
                       pyramids on ``output_img`` (Default = ``True``).
    :param use_pbar: Show a progress bar during filtering (Default = ``True``).

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
    # pylint: disable=too-many-arguments,too-many-locals
    if win_h_size < 1:
        raise rsgispy.RSGISPyException("win_h_size must be at least 1.")
    if aspect_range < 0.0:
        raise rsgispy.RSGISPyException("aspect_range must be non-negative.")

    ensure_file_exists(in_dem_img, "Input DEM image")
    ensure_file_exists(in_aspect_img, "Input aspect image")

    eff_dtype = rsgispy.TYPE_32FLOAT if datatype is None else int(datatype)
    out_rio_dtype = rsgispy.get_rasterio_datatype(eff_dtype)
    out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(eff_dtype))

    win_size = (win_h_size * 2) + 1
    with ExitStack() as stack:
        dem_src = stack.enter_context(rasterio.open(in_dem_img, "r"))
        asp_src = stack.enter_context(rasterio.open(in_aspect_img, "r"))
        if dem_src.width != asp_src.width or dem_src.height != asp_src.height:
            raise rsgispy.RSGISPyException(
                "Input DEM and aspect images must have matching dimensions."
            )

        eff_nd = no_data_val if no_data_val is not None else dem_src.nodata
        dst = create_output_from_dataset(
            stack=stack,
            src=dem_src,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype=out_rio_dtype,
            count=1,
        )

        for out_win in iter_dataset_windows(
            dem_src.height, dem_src.width, use_pbar=use_pbar
        ):
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

            centre_asp = asp_halo[
                win_h_size : win_h_size + win_h,
                win_h_size : win_h_size + win_w,
            ][..., None]
            ang_diff = np.abs(
                np.mod((asp_wins - centre_asp) + 180.0, 360.0) - 180.0
            )
            asp_match = np.isfinite(ang_diff) & (ang_diff <= aspect_range)
            dem_valid = _valid_mask(dem_wins, eff_nd)

            sel_match = dem_valid & asp_match
            has_match = np.any(sel_match, axis=2)
            has_any = np.any(dem_valid, axis=2)

            out_blk = np.full((win_h, win_w), np.nan, dtype=np.float64)
            if np.any(has_match):
                matched_vals = np.where(
                    sel_match[has_match], dem_wins[has_match], np.nan
                )
                out_blk[has_match] = np.nanmedian(matched_vals, axis=1)
            fallback_mask = (~has_match) & has_any
            if np.any(fallback_mask):
                fb_vals = np.where(
                    dem_valid[fallback_mask], dem_wins[fallback_mask], np.nan
                )
                out_blk[fallback_mask] = np.nanmedian(fb_vals, axis=1)

            dst.write(
                cast_result_block(out_blk, win_h, win_w, out_np_dtype),
                1,
                window=out_win,
            )

    if calc_stats:
        pop_img_stats(
            input_img=output_img,
            no_data_val=None,
            calc_pyramids=True,
            ignore_cog=gdalformat.upper().strip() == "COG",
            use_pbar=use_pbar,
        )


def _priority_flood_fill_py(
    dem: np.ndarray,
    seed_mask: np.ndarray,
    visited: np.ndarray,
    out: np.ndarray,
) -> None:
    """
    Pure-Python ``heapq`` Soille & Gratin (1994) priority-flood over an
    integer DEM: seeds every ``seed_mask`` pixel at its own elevation, then
    floods 8-connected unvisited neighbours at ``max(h_cur, dem[nr, nc])``.
    ``visited`` (``True`` for invalid pixels) and ``out`` are updated in
    place. Used when numba is unavailable or the values do not fit exactly
    in ``float64``; otherwise the shared
    :func:`rsgispy.elevation._hydro_kernels.priority_flood_kernel` gives the
    same result (priority-flood output does not depend on tie order).
    """
    # pylint: disable=too-many-locals
    heap: list = []
    seed_r, seed_c = np.where(seed_mask)
    for r_idx, c_idx in zip(seed_r.tolist(), seed_c.tolist()):
        elev = int(dem[r_idx, c_idx])
        out[r_idx, c_idx] = elev
        visited[r_idx, c_idx] = True
        heap.append((elev, r_idx, c_idx))

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
                if not visited[nr, nc]:
                    visited[nr, nc] = True
                    new_val = max(h_cur, int(dem[nr, nc]))
                    out[nr, nc] = new_val
                    heapq.heappush(heap, (new_val, nr, nc))


def fill_dem_soille_gratin_1994(
    in_dem_img: str,
    in_vld_img: str,
    output_img: str,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
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
    :param gdalformat: GDAL/rasterio driver format for the output file
                       (Default = ``'GTIFF'``).
    :param datatype: Optional ``rsgispy.TYPE_*`` output data type constant
                     (preserves ``in_dem_img`` dtype when ``None``).
    :param no_data_val: Optional DEM no-data value (read from header when ``None``).
    :param calc_stats: If ``True``, populate band statistics and overview
                       pyramids on ``output_img`` (Default = ``True``).
    :param use_pbar: Show a progress bar when computing statistics
                     (Default = ``True``).

    .. code:: python

        import rsgispy.elevation.filtering

        rsgispy.elevation.filtering.fill_dem_soille_gratin_1994(
            in_dem_img="dem_int.tif",
            in_vld_img="valid_mask.tif",
            output_img="dem_filled.tif",
            gdalformat="GTIFF",
        )
    """
    # pylint: disable=too-many-arguments,too-many-locals,too-many-branches,too-many-statements
    ensure_file_exists(in_dem_img, "Input DEM image")
    ensure_file_exists(in_vld_img, "Input valid mask image")

    dem_nodata: Optional[float] = None
    with ExitStack() as stack:
        dem_src = stack.enter_context(rasterio.open(in_dem_img, "r"))
        vld_src = stack.enter_context(rasterio.open(in_vld_img, "r"))

        if dem_src.count != 1:
            raise rsgispy.RSGISPyException(
                "The image to be filled should only have 1 image band."
            )
        in_rsgis_dt = rsgispy.get_rsgis_datatype(dem_src.dtypes[0])
        in_np_dtype = np.dtype(rsgispy.get_numpy_datatype(in_rsgis_dt))
        if not np.issubdtype(in_np_dtype, np.integer):
            raise rsgispy.RSGISPyException(
                "Input image must be of an integer data type."
            )
        out_rsgis_dt = in_rsgis_dt if datatype is None else int(datatype)
        out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(out_rsgis_dt))
        if dem_src.width != vld_src.width or dem_src.height != vld_src.height:
            raise rsgispy.RSGISPyException(
                "Input DEM and valid area mask images must have matching dimensions."
            )

        dem = dem_src.read(1)
        vld = vld_src.read(1) == 1
        dem_nodata = no_data_val if no_data_val is not None else dem_src.nodata

        if dem_nodata is not None:
            vld = vld & (dem != dem_nodata)
        valid_vals = dem[vld]
        if valid_vals.size == 0:
            raise rsgispy.RSGISPyException(
                "No valid DEM pixels found within the valid area mask."
            )

        fill_nodata = int(dem_nodata) if dem_nodata is not None else 0
        out = np.full(dem.shape, fill_nodata, dtype=np.int64)
        visited = ~vld

        # Seed valid raster-edge pixels and valid pixels 8-adjacent to ~vld
        # at their true elevations dem[r, c]. Padding with False guarantees
        # all raster-edge valid pixels have at least one False neighbour.
        vld_pad = np.pad(vld, 1, mode="constant", constant_values=False)
        vld_nbr_wins = np.lib.stride_tricks.sliding_window_view(vld_pad, (3, 3))
        seed_mask = vld & np.any(~vld_nbr_wins, axis=(2, 3))

        # The shared float64 priority-flood (as used by the hydrology
        # functions) is exact while every value is within +/-2**53.
        fits_f64 = max(abs(int(valid_vals.min())), abs(int(valid_vals.max()))) <= 2**53
        if fits_f64 and HAS_NUMBA and numba_enabled():
            filled = priority_flood_kernel(dem.astype(np.float64), vld)
            out[vld] = np.rint(filled[vld]).astype(np.int64)
        else:
            if not NUMBA_INSTALLED:
                warn_numba_fallback("fill_dem_soille_gratin_1994", stacklevel=2)
            _priority_flood_fill_py(dem, seed_mask, visited, out)

        out[~vld] = fill_nodata

        dst = create_output_from_dataset(
            stack=stack,
            src=dem_src,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype=rsgispy.get_rasterio_datatype(out_rsgis_dt),
            count=1,
        )
        if dem_nodata is not None:
            dst.nodata = dem_nodata
        dst.write(out.astype(out_np_dtype), 1)

    if calc_stats:
        pop_img_stats(
            input_img=output_img,
            no_data_val=dem_nodata,
            calc_pyramids=True,
            ignore_cog=gdalformat.upper().strip() == "COG",
            use_pbar=use_pbar,
        )


def plane_fit_detrend_dem(
    input_img: str,
    output_img: str,
    win_size: int,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Detrend a DEM by fitting a local least-squares plane ``z = a*x + b*y + c``
    over each ``win_size x win_size`` window and outputting the fitted centre
    elevation ``c``.

    :param input_img: Path to the input DEM image file.
    :param output_img: Path to the output detrended DEM image file.
    :param win_size: Odd window size in pixels (``>= 3``) for plane fitting.
    :param gdalformat: GDAL/rasterio driver format for the output file
                       (Default = ``'GTIFF'``).
    :param datatype: Optional ``rsgispy.TYPE_*`` output data type constant
                     (Default = ``rsgispy.TYPE_32FLOAT`` when ``None``).
    :param no_data_val: Optional DEM no-data value (read from header when ``None``).
    :param calc_stats: If ``True``, populate band statistics and overview
                       pyramids on ``output_img`` (Default = ``True``).
    :param use_pbar: Show a progress bar during filtering (Default = ``True``).

    .. code:: python

        import rsgispy.elevation.filtering

        rsgispy.elevation.filtering.plane_fit_detrend_dem(
            input_img="dem.tif",
            output_img="dem_detrend.tif",
            win_size=11,
            gdalformat="GTIFF",
        )
    """
    # pylint: disable=too-many-arguments,too-many-locals
    if win_size < 3 or win_size % 2 == 0:
        raise rsgispy.RSGISPyException(
            "win_size must be an odd integer greater than or equal to 3."
        )
    ensure_file_exists(input_img)

    eff_dtype = rsgispy.TYPE_32FLOAT if datatype is None else int(datatype)
    out_rio_dtype = rsgispy.get_rasterio_datatype(eff_dtype)
    out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(eff_dtype))

    half = win_size // 2
    offs = np.arange(-half, half + 1, dtype=np.float64)
    gx, gy = np.meshgrid(offs, offs)
    flat_x = gx.ravel()
    flat_y = gy.ravel()
    flat_xx = flat_x * flat_x
    flat_yy = flat_y * flat_y
    flat_xy = flat_x * flat_y
    n_win = float(win_size * win_size)

    eff_nd: Optional[float] = None
    with ExitStack() as stack:
        src = stack.enter_context(rasterio.open(input_img, "r"))
        eff_nd = no_data_val if no_data_val is not None else src.nodata
        dst = create_output_from_dataset(
            stack=stack,
            src=src,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype=out_rio_dtype,
            count=1,
        )
        if eff_nd is not None:
            dst.nodata = eff_nd

        for out_win in iter_dataset_windows(
            src.height, src.width, use_pbar=use_pbar
        ):
            win_h = int(out_win.height)
            win_w = int(out_win.width)
            dem_halo = _read_window_with_halo(src, 1, out_win, pad=half)
            wins = np.lib.stride_tricks.sliding_window_view(
                dem_halo, (win_size, win_size)
            ).reshape(win_h, win_w, -1)

            v_bool = _valid_mask(wins, eff_nd)
            if np.all(v_bool):
                # When all window pixels are valid, symmetric coordinates sum to 0,
                # so the plane intercept c at (0, 0) equals the window mean.
                out_c = wins.sum(axis=2) / n_win
                dst.write(
                    cast_result_block(out_c, win_h, win_w, out_np_dtype),
                    1,
                    window=out_win,
                )
                continue

            valid = v_bool.astype(np.float64)
            z_clean = np.where(v_bool, wins, 0.0)

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
            fill_val = float(eff_nd) if eff_nd is not None else 0.0
            out_c = np.divide(
                num_c,
                det,
                out=np.full_like(det, fill_val, dtype=np.float64),
                where=(s_n > 0) & (np.abs(det) > 1e-12),
            )
            dst.write(
                cast_result_block(out_c, win_h, win_w, out_np_dtype),
                1,
                window=out_win,
            )

    if calc_stats:
        pop_img_stats(
            input_img=output_img,
            no_data_val=eff_nd,
            calc_pyramids=True,
            ignore_cog=gdalformat.upper().strip() == "COG",
            use_pbar=use_pbar,
        )


def resampling_detrend_dem(
    input_img: str,
    output_img: str,
    resmp_res: float,
    low_resmp_mthd: int = rsgispy.INTERP_AVERAGE,
    up_smpl_mthd: int = rsgispy.INTERP_CUBICSPLINE,
    tmp_dir: Optional[str] = None,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Detrend a DEM by down-sampling to ``resmp_res`` and then up-sampling back to
    the original grid resolution.

    :param input_img: Input DEM image path (band 1 is used).
    :param output_img: Output detrended DEM image path.
    :param resmp_res: Intermediate coarse pixel resolution (positive spatial units).
    :param low_resmp_mthd: Interpolation method for down-sampling
                           (Default = :data:`rsgispy.INTERP_AVERAGE`).
    :param up_smpl_mthd: Interpolation method for up-sampling
                         (Default = :data:`rsgispy.INTERP_CUBICSPLINE`).
    :param tmp_dir: Optional directory for intermediate rasters (retained for
                    API compatibility; resampling runs in memory).
    :param gdalformat: GDAL/rasterio driver format for the output file
                       (Default = ``'GTIFF'``).
    :param datatype: Output ``rsgispy.TYPE_*`` constant (inferred from
                     ``input_img`` if ``None``).
    :param no_data_val: DEM no-data value (read from ``input_img`` header if ``None``).
    :param calc_stats: If ``True``, populate band statistics and overview
                       pyramids on ``output_img`` (Default = ``True``).
    :param use_pbar: Show a progress bar when computing statistics
                     (Default = ``True``).

    .. code:: python

        import rsgispy.elevation.filtering

        rsgispy.elevation.filtering.resampling_detrend_dem(
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

    with ExitStack() as stack:
        src = stack.enter_context(rasterio.open(input_img, "r"))
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
        src_transform = src.transform
        src_height = int(src.height)
        src_width = int(src.width)

        coord_width = right - left
        coord_height = top - bottom
        img_width = int(math.ceil(coord_width / resmp_res) + 1)
        img_height = int(math.ceil(coord_height / resmp_res) + 1)

        out_rio_dtype = rsgispy.get_rasterio_datatype(datatype)
        out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(datatype))

        coarse_transform = Affine(
            float(resmp_res), 0.0, float(left), 0.0, -float(resmp_res), float(top)
        )
        coarse_profile = {
            "driver": "GTiff",
            "width": img_width,
            "height": img_height,
            "count": 1,
            "dtype": out_rio_dtype,
            "crs": src_crs,
            "transform": coarse_transform,
            "nodata": no_data_val,
        }

        low_vrt = stack.enter_context(
            WarpedVRT(
                src,
                crs=src_crs,
                transform=coarse_transform,
                width=img_width,
                height=img_height,
                resampling=rsgispy.get_rasterio_interp_type(low_resmp_mthd),
                src_nodata=no_data_val,
                nodata=no_data_val,
                dtype=out_rio_dtype,
            )
        )

        if tmp_dir is not None:
            os.makedirs(tmp_dir, exist_ok=True)
            tmp_ctx = stack.enter_context(
                tempfile.TemporaryDirectory(dir=tmp_dir)
            )
            coarse_path = os.path.join(tmp_ctx, "coarse_dem.tif")
            with rasterio.open(coarse_path, "w", **coarse_profile) as coarse_dst:
                for cwin in iter_dataset_windows(
                    img_height, img_width, use_pbar=False
                ):
                    cblk = low_vrt.read(1, window=cwin).astype(
                        out_np_dtype, copy=False
                    )
                    coarse_dst.write(cblk, 1, window=cwin)
            coarse_src = stack.enter_context(rasterio.open(coarse_path, "r"))
        else:
            mem_file = stack.enter_context(MemoryFile())
            with mem_file.open(**coarse_profile) as coarse_dst:
                for cwin in iter_dataset_windows(
                    img_height, img_width, use_pbar=False
                ):
                    cblk = low_vrt.read(1, window=cwin).astype(
                        out_np_dtype, copy=False
                    )
                    coarse_dst.write(cblk, 1, window=cwin)
            coarse_src = stack.enter_context(mem_file.open())

        up_vrt = stack.enter_context(
            WarpedVRT(
                coarse_src,
                crs=src_crs,
                transform=src_transform,
                width=src_width,
                height=src_height,
                resampling=rsgispy.get_rasterio_interp_type(up_smpl_mthd),
                src_nodata=no_data_val,
                nodata=no_data_val,
                dtype=out_rio_dtype,
            )
        )

        out_grid = IntersectionGrid(
            crs=src_crs,
            transform=src_transform,
            width=src_width,
            height=src_height,
            offsets={},
        )
        dst = open_output_dataset(
            stack=stack,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype=out_rio_dtype,
            grid=out_grid,
            count=1,
        )
        dst.nodata = no_data_val
        for out_win in iter_dataset_windows(
            src_height, src_width, use_pbar=use_pbar
        ):
            ublk = up_vrt.read(1, window=out_win).astype(
                out_np_dtype, copy=False
            )
            dst.write(ublk, 1, window=out_win)

    if calc_stats:
        pop_img_stats(
            input_img=output_img,
            no_data_val=no_data_val,
            calc_pyramids=True,
            ignore_cog=gdalformat.upper().strip() == "COG",
            use_pbar=use_pbar,
        )
