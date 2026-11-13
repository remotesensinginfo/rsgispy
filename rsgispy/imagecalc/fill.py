"""
The imagecalc.fill module provides spatial k-nearest-neighbour region filling
utilities using ``rasterio`` and ``scipy``.

Functions requiring ``scipy`` are only defined and exported when ``scipy`` is
installed.
"""

from contextlib import ExitStack
from typing import Any, List, Mapping, Optional, Tuple, Union

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
from rsgispy.imageutils.checks import do_gdal_layers_have_same_proj

try:
    import scipy.spatial  # pylint: disable=import-error
    import scipy.stats  # pylint: disable=import-error

    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

__all__: List[str] = []

if HAVE_SCIPY:
    __all__.append("calc_fill_regions_knn")


def _summarise_neighbours(nbr_vals: np.ndarray, summary: int) -> np.ndarray:
    """
    Summarise a 2D array ``(n_pixels, k)`` of neighbour values using
    ``rsgispy.SUMTYPE_*``.
    """
    if summary == rsgispy.SUMTYPE_MODE:
        mode_res = scipy.stats.mode(nbr_vals, axis=1).mode
        return np.asarray(mode_res, dtype=np.float64).reshape(-1)[: nbr_vals.shape[0]]
    if summary == rsgispy.SUMTYPE_MEDIAN:
        return np.median(nbr_vals, axis=1)
    if summary == rsgispy.SUMTYPE_MAX:
        return np.max(nbr_vals, axis=1)
    if summary == rsgispy.SUMTYPE_MIN:
        return np.min(nbr_vals, axis=1)
    if summary == rsgispy.SUMTYPE_MEAN:
        return np.mean(nbr_vals, axis=1)
    if summary == rsgispy.SUMTYPE_SUM:
        return np.sum(nbr_vals, axis=1)
    if summary == rsgispy.SUMTYPE_RANGE:
        return np.max(nbr_vals, axis=1) - np.min(nbr_vals, axis=1)
    raise rsgispy.RSGISPyException("Summary method not recognised/available.")


def _pixel_centers_from_indices(
    transform: rasterio.Affine, rows: np.ndarray, cols: np.ndarray
) -> np.ndarray:
    """Compute ``(N, 2)`` pixel-centre ``(x, y)`` coordinates for ``(rows, cols)``."""
    col_f = cols.astype(np.float64) + 0.5
    row_f = rows.astype(np.float64) + 0.5
    xs = transform.c + col_f * transform.a + row_f * transform.b
    ys = transform.f + col_f * transform.d + row_f * transform.e
    return np.column_stack((xs, ys))


def _collect_reference_points(
    in_ref_img: str, ref_no_data: Union[int, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Read valid reference pixel centres and values from band 1 of ``in_ref_img``."""
    coords_list: List[np.ndarray] = []
    vals_list: List[np.ndarray] = []
    ref_nd_is_nan = np.isnan(ref_no_data)

    with rasterio.open(in_ref_img, "r") as ref_src:
        for win in iter_dataset_windows(
            ref_src.height, ref_src.width, use_pbar=False
        ):
            arr = ref_src.read(1, window=win, out_dtype=np.float64)
            if ref_nd_is_nan:
                valid = np.isfinite(arr)
            else:
                valid = np.isfinite(arr) & (arr != ref_no_data)
            if not np.any(valid):
                continue
            loc_rows, loc_cols = np.nonzero(valid)
            abs_rows = loc_rows + int(win.row_off)
            abs_cols = loc_cols + int(win.col_off)
            coords_list.append(
                _pixel_centers_from_indices(ref_src.transform, abs_rows, abs_cols)
            )
            vals_list.append(arr[valid])

    if not vals_list:
        raise rsgispy.RSGISPyException(
            "No valid reference pixels found in in_ref_img."
        )

    return np.concatenate(coords_list, axis=0), np.concatenate(vals_list, axis=0)


if HAVE_SCIPY:

    def calc_fill_regions_knn(
        in_ref_img: str,
        ref_no_data: Union[int, float],
        in_fill_regions_img: str,
        fill_region_val: Union[int, float],
        output_img: str,
        k: int = 5,
        summary: int = rsgispy.SUMTYPE_MODE,
        gdalformat: str = "GTIFF",
        datatype: int = rsgispy.TYPE_32INT,
        use_pbar: bool = True,
        creation_opts: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """
        Fill regions in ``in_fill_regions_img`` equal to ``fill_region_val``
        using spatial k-nearest neighbours sampled from ``in_ref_img``.

        Pixels outside the target fill regions are written as ``0``. Both input
        images must share the same CRS/projection.

        :param in_ref_img: Path to the reference image (can differ in resolution).
        :param ref_no_data: No-data value in ``in_ref_img`` excluded from KNN.
        :param in_fill_regions_img: Path to the mask image defining fill regions.
        :param fill_region_val: Pixel value in ``in_fill_regions_img`` to fill.
        :param output_img: Path to the output image file.
        :param k: Number of nearest neighbours (Default = ``5``).
        :param summary: Summary method ``rsgispy.SUMTYPE_*`` (Mode, Median, Max,
                        Min, Mean, Sum, or Range; Default = ``SUMTYPE_MODE``).
        :param gdalformat: GDAL/rasterio driver format for the output file
                           (Default = ``'GTIFF'``).
        :param datatype: Output data type constant from ``rsgispy.TYPE_*``
                         (Default = ``rsgispy.TYPE_32INT``).
        :param use_pbar: If True, display a tqdm progress bar (Default = True).
        :param creation_opts: Optional mapping of GDAL creation options.
        """
        # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
        ensure_file_exists(in_ref_img, label="Reference image")
        ensure_file_exists(in_fill_regions_img, label="Fill regions image")
        if k < 1:
            raise rsgispy.RSGISPyException("k must be >= 1.")

        valid_summaries = (
            rsgispy.SUMTYPE_MODE,
            rsgispy.SUMTYPE_MEDIAN,
            rsgispy.SUMTYPE_MAX,
            rsgispy.SUMTYPE_MIN,
            rsgispy.SUMTYPE_MEAN,
            rsgispy.SUMTYPE_SUM,
            rsgispy.SUMTYPE_RANGE,
        )
        if summary not in valid_summaries:
            raise rsgispy.RSGISPyException(
                "Summary method not recognised/available."
            )

        if not do_gdal_layers_have_same_proj(in_ref_img, in_fill_regions_img):
            raise rsgispy.RSGISPyException(
                "The reference image and fill regions image "
                "do not have the same projection."
            )

        ref_coords, ref_vals = _collect_reference_points(in_ref_img, ref_no_data)
        k_eff = min(int(k), int(ref_vals.size))
        kd_tree = scipy.spatial.cKDTree(ref_coords)

        out_rio_dtype = rsgispy.get_rasterio_datatype(datatype)
        out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(datatype))

        with ExitStack() as stack:
            try:
                fill_src = stack.enter_context(
                    rasterio.open(in_fill_regions_img, "r")
                )
            except Exception as err:
                raise rsgispy.RSGISPyException(
                    f"Could not open fill regions image '{in_fill_regions_img}': {err}"
                ) from err

            grid = compute_spatial_intersection({in_fill_regions_img: fill_src})
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
                win_h = int(out_win.height)
                win_w = int(out_win.width)
                in_block = fill_src.read(1, window=out_win)
                out_arr = np.zeros((win_h, win_w), dtype=out_np_dtype)

                fill_mask = in_block == fill_region_val
                if np.any(fill_mask):
                    loc_rows, loc_cols = np.nonzero(fill_mask)
                    abs_rows = loc_rows + int(out_win.row_off)
                    abs_cols = loc_cols + int(out_win.col_off)
                    query_pts = _pixel_centers_from_indices(
                        fill_src.transform, abs_rows, abs_cols
                    )
                    _, idxs = kd_tree.query(query_pts, k=k_eff)
                    nbr_vals = ref_vals[idxs].reshape(loc_rows.size, k_eff)
                    fill_vals = _summarise_neighbours(nbr_vals, summary)
                    out_arr[loc_rows, loc_cols] = fill_vals.astype(
                        out_np_dtype, copy=False
                    )

                dst.write(
                    cast_result_block(out_arr, win_h, win_w, out_np_dtype),
                    1,
                    window=out_win,
                )
