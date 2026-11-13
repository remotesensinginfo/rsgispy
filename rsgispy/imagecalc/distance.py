"""
The imagecalc.distance module provides Euclidean distance-to-value and binary
proximity buffering utilities using ``rasterio`` and ``scipy``.

Functions requiring ``scipy`` are only defined and exported when ``scipy`` is
installed.
"""

from contextlib import ExitStack
import math
from multiprocessing import Pool
import os
from typing import Any, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import rasterio
from rasterio.windows import Window

import rsgispy
from rsgispy.imageprocess import (
    compute_spatial_intersection,
    ensure_file_exists,
    open_output_dataset,
    validate_band_index,
)
from rsgispy.tools.system import num_process_cores

try:
    from scipy.ndimage import distance_transform_edt  # pylint: disable=import-error

    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

__all__: List[str] = []

if HAVE_SCIPY:
    __all__.extend(
        [
            "calc_dist_to_img_vals",
            "calc_dist_to_img_vals_tiled",
            "buffer_img_pxl_vals",
        ]
    )


def _normalise_pxl_vals(
    pxl_vals: Union[int, float, Sequence[Union[int, float]]],
) -> np.ndarray:
    """Normalise scalar or sequence ``pxl_vals`` into a 1D NumPy array."""
    if isinstance(pxl_vals, (list, tuple, set, np.ndarray)):
        arr = np.asarray(list(pxl_vals))
    else:
        arr = np.asarray([pxl_vals])
    if arr.size == 0:
        raise rsgispy.RSGISPyException("pxl_vals must contain at least one value.")
    return arr


if HAVE_SCIPY:

    def _compute_distance_block(
        band_arr: np.ndarray,
        target_vals: np.ndarray,
        sampling: Tuple[float, float],
        max_dist: Optional[float],
        no_data_val: Optional[float],
        out_no_data_val: float,
    ) -> np.ndarray:
        """
        Compute Euclidean distance to ``target_vals`` in ``band_arr``.
        """
        # pylint: disable=too-many-arguments,too-many-positional-arguments
        target_mask = np.isin(band_arr, target_vals)
        if no_data_val is not None:
            if np.isnan(no_data_val):
                nd_mask = np.isnan(band_arr)
            else:
                nd_mask = band_arr == no_data_val
            target_mask &= ~nd_mask
        else:
            nd_mask = np.zeros(band_arr.shape, dtype=bool)

        if not np.any(target_mask):
            return np.full(band_arr.shape, out_no_data_val, dtype=np.float32)

        dist = distance_transform_edt(~target_mask, sampling=sampling).astype(
            np.float32
        )
        if max_dist is not None:
            dist[dist > max_dist] = out_no_data_val
        if no_data_val is not None:
            dist[nd_mask] = out_no_data_val
        return dist

    def _dist_tile_worker(
        args: Tuple[
            str,
            int,
            np.ndarray,
            Tuple[float, float],
            float,
            Optional[float],
            float,
            int,
            int,
            int,
            int,
            int,
            int,
            int,
            int,
        ],
    ) -> Tuple[int, int, int, int, np.ndarray]:
        """
        Top-level picklable worker computing distance over an overlapped tile
        and returning only the unpadded core region.
        """
        (
            input_img,
            img_band,
            target_vals,
            sampling,
            max_dist,
            no_data_val,
            out_no_data_val,
            row_off,
            col_off,
            core_h,
            core_w,
            pad_row_off,
            pad_col_off,
            pad_h,
            pad_w,
        ) = args
        with rasterio.open(input_img, "r") as src:
            pad_win = Window(pad_col_off, pad_row_off, pad_w, pad_h)
            pad_arr = src.read(img_band, window=pad_win)

        pad_dist = _compute_distance_block(
            band_arr=pad_arr,
            target_vals=target_vals,
            sampling=sampling,
            max_dist=max_dist,
            no_data_val=no_data_val,
            out_no_data_val=out_no_data_val,
        )
        inner_r0 = row_off - pad_row_off
        inner_c0 = col_off - pad_col_off
        core_dist = pad_dist[
            inner_r0 : inner_r0 + core_h, inner_c0 : inner_c0 + core_w
        ]
        return row_off, col_off, core_h, core_w, core_dist

    def calc_dist_to_img_vals(
        input_img: str,
        output_img: str,
        pxl_vals: Union[int, float, Sequence[Union[int, float]]],
        img_band: int = 1,
        gdalformat: str = "GTIFF",
        max_dist: Optional[float] = None,
        no_data_val: Optional[float] = None,
        out_no_data_val: Optional[float] = None,
        unit_geo: bool = True,
        creation_opts: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """
        Calculate Euclidean distance to the nearest pixel having one of the
        values specified in ``pxl_vals``.

        :param input_img: Path to the input image file.
        :param output_img: Path to the output float32 distance image file.
        :param pxl_vals: Target pixel value or sequence of pixel values.
        :param img_band: 1-based band index to inspect (Default = ``1``).
        :param gdalformat: GDAL/rasterio driver format for the output file
                           (Default = ``'GTIFF'``).
        :param max_dist: Optional maximum distance threshold; distances greater
                         than ``max_dist`` are set to ``out_no_data_val``.
        :param no_data_val: Optional input no-data value excluded from calculation
                            and written as ``out_no_data_val``.
        :param out_no_data_val: Output no-data value (defaults to ``max_dist`` if
                                ``None``).
        :param unit_geo: If ``True``, return distances in CRS map units; if
                         ``False``, return distances in pixel units (Default = True).
        :param creation_opts: Optional mapping of GDAL creation options.
        """
        # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
        ensure_file_exists(input_img)
        target_vals = _normalise_pxl_vals(pxl_vals)

        eff_out_nd = out_no_data_val if out_no_data_val is not None else max_dist
        if eff_out_nd is None:
            raise rsgispy.RSGISPyException(
                "Either out_no_data_val or max_dist must be supplied."
            )

        with ExitStack() as stack:
            try:
                src = stack.enter_context(rasterio.open(input_img, "r"))
            except Exception as err:
                raise rsgispy.RSGISPyException(
                    f"Could not open input image '{input_img}': {err}"
                ) from err

            validate_band_index(img_band, src.count, input_img)
            if unit_geo:
                sampling = (abs(float(src.transform.e)), abs(float(src.transform.a)))
            else:
                sampling = (1.0, 1.0)

            band_arr = src.read(img_band)
            dist_arr = _compute_distance_block(
                band_arr=band_arr,
                target_vals=target_vals,
                sampling=sampling,
                max_dist=max_dist,
                no_data_val=no_data_val,
                out_no_data_val=float(eff_out_nd),
            )

            grid = compute_spatial_intersection({input_img: src})
            dst = open_output_dataset(
                stack=stack,
                output_img=output_img,
                gdalformat=gdalformat,
                out_rio_dtype="float32",
                grid=grid,
                count=1,
                creation_opts=creation_opts,
            )
            dst.nodata = float(eff_out_nd)
            dst.write(dist_arr, 1)

    def calc_dist_to_img_vals_tiled(
        input_img: str,
        output_img: str,
        pxl_vals: Union[int, float, Sequence[Union[int, float]]],
        img_band: int = 1,
        max_dist: float = 1000,
        no_data_val: Optional[float] = 1000,
        out_no_data_val: Optional[float] = None,
        gdalformat: str = "GTIFF",
        unit_geo: bool = True,
        tmp_dir: str = "tmp",
        tile_size: int = 2000,
        n_cores: int = -1,
        creation_opts: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """
        Calculate Euclidean distance to ``pxl_vals`` across overlapping tiles
        so large rasters can be processed without exceeding memory limits.

        :param input_img: Path to the input image file.
        :param output_img: Path to the output float32 distance image file.
        :param pxl_vals: Target pixel value or sequence of pixel values.
        :param img_band: 1-based band index to inspect (Default = ``1``).
        :param max_dist: Maximum distance to compute (Default = ``1000``).
        :param no_data_val: Optional input no-data value (Default = ``1000``).
        :param out_no_data_val: Output no-data value (defaults to ``max_dist`` if
                                ``None``).
        :param gdalformat: GDAL/rasterio driver format for the output file
                           (Default = ``'GTIFF'``).
        :param unit_geo: If ``True``, distances are in CRS map units; if ``False``,
                         in pixel units (Default = ``True``).
        :param tmp_dir: Retained for API compatibility with ``rsgislib``
                        (Default = ``'tmp'``).
        :param tile_size: Core tile size in pixels (Default = ``2000``).
        :param n_cores: Number of worker processes; ``<= 0`` uses all CPU cores
                        (Default = ``-1``).
        :param creation_opts: Optional mapping of GDAL creation options.
        """
        # pylint: disable=too-many-arguments,too-many-positional-arguments
        # pylint: disable=too-many-locals,unused-argument
        ensure_file_exists(input_img)
        target_vals = _normalise_pxl_vals(pxl_vals)
        if max_dist is None or max_dist <= 0:
            raise rsgispy.RSGISPyException("max_dist must be a positive number.")
        if tile_size <= 0:
            raise rsgispy.RSGISPyException("tile_size must be a positive integer.")

        eff_out_nd = float(out_no_data_val if out_no_data_val is not None else max_dist)
        eff_cores = num_process_cores() if n_cores <= 0 else int(n_cores)

        with ExitStack() as stack:
            try:
                src = stack.enter_context(rasterio.open(input_img, "r"))
            except Exception as err:
                raise rsgispy.RSGISPyException(
                    f"Could not open input image '{input_img}': {err}"
                ) from err

            validate_band_index(img_band, src.count, input_img)
            x_res = abs(float(src.transform.a))
            y_res = abs(float(src.transform.e))
            if unit_geo:
                sampling = (y_res, x_res)
                x_ovl = math.ceil(max_dist / x_res) + 10
                y_ovl = math.ceil(max_dist / y_res) + 10
            else:
                sampling = (1.0, 1.0)
                x_ovl = math.ceil(max_dist) + 10
                y_ovl = math.ceil(max_dist) + 10

            width = src.width
            height = src.height
            tasks = []
            for row_off in range(0, height, tile_size):
                core_h = min(tile_size, height - row_off)
                pad_r0 = max(0, row_off - y_ovl)
                pad_r1 = min(height, row_off + core_h + y_ovl)
                for col_off in range(0, width, tile_size):
                    core_w = min(tile_size, width - col_off)
                    pad_c0 = max(0, col_off - x_ovl)
                    pad_c1 = min(width, col_off + core_w + x_ovl)
                    tasks.append(
                        (
                            input_img,
                            img_band,
                            target_vals,
                            sampling,
                            float(max_dist),
                            no_data_val,
                            eff_out_nd,
                            row_off,
                            col_off,
                            core_h,
                            core_w,
                            pad_r0,
                            pad_c0,
                            pad_r1 - pad_r0,
                            pad_c1 - pad_c0,
                        )
                    )

            grid = compute_spatial_intersection({input_img: src})
            dst = open_output_dataset(
                stack=stack,
                output_img=output_img,
                gdalformat=gdalformat,
                out_rio_dtype="float32",
                grid=grid,
                count=1,
                creation_opts=creation_opts,
            )
            dst.nodata = eff_out_nd

            if eff_cores > 1 and len(tasks) > 1:
                with Pool(eff_cores) as pool:
                    results = pool.map(_dist_tile_worker, tasks)
                for row_off, col_off, core_h, core_w, core_dist in results:
                    dst.write(
                        core_dist,
                        1,
                        window=Window(col_off, row_off, core_w, core_h),
                    )
            else:
                for task in tasks:
                    row_off, col_off, core_h, core_w, core_dist = (
                        _dist_tile_worker(task)
                    )
                    dst.write(
                        core_dist,
                        1,
                        window=Window(col_off, row_off, core_w, core_h),
                    )

    def buffer_img_pxl_vals(
        input_img: str,
        output_img: str,
        pxl_vals: Union[int, float, Sequence[Union[int, float]]],
        buf_thres: float,
        tmp_dir: Optional[str] = None,
        gdalformat: str = "GTIFF",
        img_band: int = 1,
        unit_geo: bool = True,
        creation_opts: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """
        Buffer image pixel values by distance ``buf_thres`` to produce a binary
        ``uint8`` mask (``1`` within buffer, ``0`` outside).

        :param input_img: Path to the input image file.
        :param output_img: Path to the output binary mask image file.
        :param pxl_vals: Target pixel value or sequence of pixel values to buffer.
        :param buf_thres: Distance threshold strictly below which pixels are set
                          to ``1`` in the output mask.
        :param tmp_dir: Optional directory path checked for existence when
                        provided (retained for API compatibility).
        :param gdalformat: GDAL/rasterio driver format for the output file
                           (Default = ``'GTIFF'``).
        :param img_band: 1-based band index to buffer (Default = ``1``).
        :param unit_geo: If ``True``, ``buf_thres`` is in CRS map units; if
                         ``False``, in pixel units (Default = ``True``).
        :param creation_opts: Optional mapping of GDAL creation options.
        """
        # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
        if tmp_dir is not None and not os.path.exists(tmp_dir):
            raise rsgispy.RSGISPyException("The tmp_dir does not exist.")
        ensure_file_exists(input_img)
        target_vals = _normalise_pxl_vals(pxl_vals)

        with ExitStack() as stack:
            try:
                src = stack.enter_context(rasterio.open(input_img, "r"))
            except Exception as err:
                raise rsgispy.RSGISPyException(
                    f"Could not open input image '{input_img}': {err}"
                ) from err

            validate_band_index(img_band, src.count, input_img)
            if unit_geo:
                sampling = (abs(float(src.transform.e)), abs(float(src.transform.a)))
            else:
                sampling = (1.0, 1.0)

            band_arr = src.read(img_band)
            dist_arr = _compute_distance_block(
                band_arr=band_arr,
                target_vals=target_vals,
                sampling=sampling,
                max_dist=float(buf_thres) * 1.1,
                no_data_val=None,
                out_no_data_val=-999.0,
            )
            buf_mask = np.where(
                (dist_arr >= 0.0) & (dist_arr < float(buf_thres)), 1, 0
            ).astype(np.uint8)

            grid = compute_spatial_intersection({input_img: src})
            dst = open_output_dataset(
                stack=stack,
                output_img=output_img,
                gdalformat=gdalformat,
                out_rio_dtype="uint8",
                grid=grid,
                count=1,
                creation_opts=creation_opts,
            )
            dst.write(buf_mask, 1)
