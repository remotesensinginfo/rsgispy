"""
The imagecalc.fill module provides spatial region filling utilities using
``rasterio``, including k-nearest-neighbour filling (``scipy``) and
balanced iterative region-growing filling (pure ``numpy``).
"""

from contextlib import ExitStack
import heapq
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import rasterio
from rasterio.windows import Window

import rsgispy
from rsgispy.imageprocess import (
    cast_result_block,
    compute_spatial_intersection,
    ensure_file_exists,
    iter_dataset_windows,
    open_output_dataset,
)
from rsgispy.imageutils._helpers import (
    maybe_pop_stats,
    normalise_creation_options,
    rsgis_dtype_from_np_dtype,
)
from rsgispy.imageutils.checks import do_gdal_layers_have_same_proj

try:
    import scipy.spatial  # pylint: disable=import-error
    import scipy.stats  # pylint: disable=import-error

    HAVE_SCIPY = True
except ImportError:
    scipy = None  # type: ignore[assignment]
    HAVE_SCIPY = False

__all__: List[str] = ["calc_fill_regions_knn", "region_growing_fill"]


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


def _pixel_centres_from_indices(
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
                _pixel_centres_from_indices(ref_src.transform, abs_rows, abs_cols)
            )
            vals_list.append(arr[valid])

    if not vals_list:
        raise rsgispy.RSGISPyException(
            "No valid reference pixels found in in_ref_img."
        )

    return np.concatenate(coords_list, axis=0), np.concatenate(vals_list, axis=0)


def calc_fill_regions_knn(
    in_ref_img: str,
    in_fill_regions_img: str,
    output_img: str,
    ref_no_data: Union[int, float],
    fill_region_val: Union[int, float],
    k: int = 5,
    summary: int = rsgispy.SUMTYPE_MODE,
    *,
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
    :param in_fill_regions_img: Path to the mask image defining fill regions.
    :param output_img: Path to the output image file.
    :param ref_no_data: No-data value in ``in_ref_img`` excluded from KNN.
    :param fill_region_val: Pixel value in ``in_fill_regions_img`` to fill.
    :param k: Number of nearest neighbours (Default = ``5``).
    :param summary: Summary method ``rsgispy.SUMTYPE_*`` (Mode, Median, Max,
                    Min, Mean, Sum, or Range; Default = ``SUMTYPE_MODE``).
    :param gdalformat: GDAL/rasterio driver format for the output file
                       (Default = ``'GTIFF'``).
    :param datatype: Output data type constant from ``rsgispy.TYPE_*``
                     (Default = ``rsgispy.TYPE_32INT``).
    :param use_pbar: If True, display a tqdm progress bar (Default = True).
    :param creation_opts: Optional mapping of GDAL creation options.

    .. code:: python

        import rsgispy
        import rsgispy.imagecalc.fill

        rsgispy.imagecalc.fill.calc_fill_regions_knn(
            in_ref_img="ref_classes.tif",
            in_fill_regions_img="fill_mask.tif",
            output_img="filled_classes.tif",
            ref_no_data=0,
            fill_region_val=1,
            k=5,
            summary=rsgispy.SUMTYPE_MODE,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    rsgispy.check_optional_dep(HAVE_SCIPY, "calc_fill_regions_knn", "scipy", "stats")
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
                query_pts = _pixel_centres_from_indices(
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


def _valid_seed_mask(
    reg_arr: np.ndarray, ref_no_data: Union[int, float]
) -> np.ndarray:
    """Return boolean mask of valid non-nodata seed pixels in ``reg_arr``."""
    if np.issubdtype(reg_arr.dtype, np.floating):
        if np.isnan(ref_no_data):
            return np.isfinite(reg_arr)
        return np.isfinite(reg_arr) & (reg_arr != ref_no_data)
    if np.isnan(ref_no_data):
        return np.ones(reg_arr.shape, dtype=bool)
    return reg_arr != ref_no_data


def _propagate_region_labels(
    reg_arr: np.ndarray,
    seed_mask: np.ndarray,
    fill_mask: np.ndarray,
    out_np_dtype: np.dtype,
    eff_out_nd: Union[int, float],
    connectivity: int,
    max_iters: Optional[int],
    use_pbar: bool,
) -> np.ndarray:
    """
    Propagate seed region labels across connected components of ``fill_mask``
    using pure-NumPy Dial-bucketed Chamfer (5, 7) / cardinal wavefronts.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    height, width = reg_arr.shape
    fill_target = fill_mask & (~seed_mask)
    pad_labels = np.full((height + 2, width + 2), eff_out_nd, dtype=out_np_dtype)
    pad_labels[1:-1, 1:-1][seed_mask] = reg_arr[seed_mask].astype(
        out_np_dtype, copy=False
    )

    n_to_fill = int(np.count_nonzero(fill_target))
    if n_to_fill > 0:
        inf_dist = np.iinfo(np.int32).max
        pad_dist = np.full((height + 2, width + 2), -1, dtype=np.int32)
        pad_dist[1:-1, 1:-1][fill_target] = inf_dist
        pad_dist[1:-1, 1:-1][seed_mask] = 0

        if connectivity == 8:
            offsets: Tuple[Tuple[int, int, int], ...] = (
                (-1, 0, 5),
                (1, 0, 5),
                (0, -1, 5),
                (0, 1, 5),
                (-1, -1, 7),
                (-1, 1, 7),
                (1, -1, 7),
                (1, 1, 7),
            )
            step_unit = 5
        else:
            offsets = ((-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1))
            step_unit = 1

        pad_target = pad_dist == inf_dist
        has_target_nbr = np.zeros((height, width), dtype=bool)
        for dr, dc, _ in offsets:
            has_target_nbr |= pad_target[
                1 + dr : 1 + dr + height, 1 + dc : 1 + dc + width
            ]
        del pad_target

        frontier_r, frontier_c = np.nonzero(seed_mask & has_target_nbr)
        del has_target_nbr

        if frontier_r.size > 0:
            frontier_r = frontier_r + 1
            frontier_c = frontier_c + 1
            max_dist = None if max_iters is None else int(max_iters) * step_unit
            buckets: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = {
                0: [(frontier_r, frontier_c)]
            }
            heap: List[int] = [0]
            tqdm_cls = rsgispy.get_tqdm()

            with tqdm_cls(
                total=n_to_fill, disable=not use_pbar, desc="Growing regions"
            ) as pbar:
                while heap:
                    cur_d = heapq.heappop(heap)
                    chunks = buckets.pop(cur_d)
                    if max_dist is not None and cur_d >= max_dist:
                        break

                    if len(chunks) == 1:
                        rs, cs = chunks[0]
                    else:
                        rs = np.concatenate([chunk[0] for chunk in chunks])
                        cs = np.concatenate([chunk[1] for chunk in chunks])

                    if cur_d > 0:
                        keep = pad_dist[rs, cs] == cur_d
                        if not np.any(keep):
                            continue
                        if not np.all(keep):
                            rs = rs[keep]
                            cs = cs[keep]

                    for dr, dc, weight in offsets:
                        nd_val = cur_d + weight
                        nr = rs + dr
                        nc = cs + dc
                        cur_nbr_dist = pad_dist[nr, nc]
                        improve = nd_val < cur_nbr_dist
                        if not np.any(improve):
                            continue

                        vr = rs[improve]
                        vc = cs[improve]
                        unr = nr[improve]
                        unc = nc[improve]

                        if use_pbar:
                            newly_claimed = int(
                                np.count_nonzero(cur_nbr_dist[improve] == inf_dist)
                            )
                            if newly_claimed:
                                pbar.update(newly_claimed)

                        pad_dist[unr, unc] = nd_val
                        pad_labels[unr, unc] = pad_labels[vr, vc]
                        if nd_val not in buckets:
                            buckets[nd_val] = [(unr, unc)]
                            heapq.heappush(heap, nd_val)
                        else:
                            buckets[nd_val].append((unr, unc))

    out_arr = pad_labels[1:-1, 1:-1]
    if not np.all(fill_mask):
        out_arr[~fill_mask] = eff_out_nd
    return out_arr


def region_growing_fill(
    in_regions_img: Union[str, os.PathLike],
    in_fill_regions_img: Union[str, os.PathLike],
    output_img: Union[str, os.PathLike],
    ref_no_data: Optional[Union[int, float]] = None,
    fill_region_val: Union[int, float] = 1,
    connectivity: int = 8,
    max_iters: Optional[int] = None,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
    creation_opts: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
) -> None:
    """
    Iteratively expand seed regions from ``in_regions_img`` into connected
    target regions defined in ``in_fill_regions_img`` using balanced
    nearest-boundary geodesic propagation.

    Starting from all valid (non-no-data) pixels in ``in_regions_img``, labels
    grow outward step by step strictly through pixels in
    ``in_fill_regions_img`` equal to ``fill_region_val``. When
    ``connectivity=8`` (default), Chamfer ``(5, 7)`` distance weights
    (ratio ``7/5 = 1.400`` approximating ``sqrt(2)``) are used so expanding
    fronts remain nearly circular and meet along balanced nearest boundaries.
    Expansion stops automatically once all reachable pixels in each connected
    target component are filled (or after ``max_iters`` steps). Disconnected
    components of ``fill_region_val`` with no bordering or overlapping seed
    pixels remain unfilled (set to the output no-data value), and pixels
    outside ``fill_region_val`` are also written as the output no-data value.

    :param in_regions_img: Path to the input seed regions image (band 1). All
                           non-no-data pixel values define unique region IDs.
    :param in_fill_regions_img: Path to the mask image defining target regions
                                to fill (band 1). Must share the same CRS and
                                pixel resolution as ``in_regions_img``.
    :param output_img: Path to the output filled regions image.
    :param ref_no_data: No-data value in ``in_regions_img`` excluded from seed
                        regions. If ``None`` (default), uses the input
                        dataset's header no-data value, or ``0`` if unset.
    :param fill_region_val: Pixel value in ``in_fill_regions_img`` defining the
                            fillable domain (Default = ``1``).
    :param connectivity: Neighbourhood connectivity: ``4`` (cardinal) or ``8``
                         (Chamfer ``(5, 7)`` balanced diagonal+cardinal,
                         Default = ``8``).
    :param max_iters: Optional maximum number of expansion steps (where each
                      step corresponds to one cardinal pixel distance). If
                      ``None`` (default), expands until all reachable pixels
                      in connected fill components are filled.
    :param gdalformat: GDAL/rasterio driver format for the output file
                       (Default = ``'GTIFF'``).
    :param datatype: Output data type constant from ``rsgispy.TYPE_*``. When
                     ``None`` (default), inferred from ``in_regions_img``.
    :param no_data_val: Output no-data value. When ``None`` (default), uses
                        ``ref_no_data`` (or ``0`` if ``ref_no_data`` is NaN
                        and the output data type is integer).
    :param calc_stats: If ``True``, populate image statistics and overview
                       pyramids on ``output_img`` (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar during propagation
                     (Default = ``True``).
    :param creation_opts: Optional GDAL creation options as a mapping or sequence
                          of ``'KEY=VALUE'`` strings.

    .. code:: python

        import rsgispy.imagecalc.fill

        rsgispy.imagecalc.fill.region_growing_fill(
            in_regions_img="sub_country_regions.tif",
            in_fill_regions_img="country_extent.tif",
            output_img="sub_country_filled.tif",
            fill_region_val=1,
            connectivity=8,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    reg_path = os.fspath(in_regions_img)
    mask_path = os.fspath(in_fill_regions_img)
    out_path = os.fspath(output_img)

    ensure_file_exists(reg_path, label="Regions image")
    ensure_file_exists(mask_path, label="Fill regions image")

    if connectivity not in (4, 8):
        raise rsgispy.RSGISPyException(
            f"connectivity must be 4 or 8, got {connectivity}."
        )
    if max_iters is not None and max_iters < 0:
        raise rsgispy.RSGISPyException("max_iters must be >= 0 or None.")

    if not do_gdal_layers_have_same_proj(reg_path, mask_path):
        raise rsgispy.RSGISPyException(
            "The regions image and fill regions image "
            "do not have the same projection."
        )

    norm_opts = normalise_creation_options(creation_opts)

    with ExitStack() as stack:
        try:
            reg_src = stack.enter_context(rasterio.open(reg_path, "r"))
        except Exception as err:
            raise rsgispy.RSGISPyException(
                f"Could not open regions image '{reg_path}': {err}"
            ) from err
        try:
            mask_src = stack.enter_context(rasterio.open(mask_path, "r"))
        except Exception as err:
            raise rsgispy.RSGISPyException(
                f"Could not open fill regions image '{mask_path}': {err}"
            ) from err

        grid = compute_spatial_intersection({reg_path: reg_src, mask_path: mask_src})
        reg_off_c, reg_off_r = grid.offsets[reg_path]
        mask_off_c, mask_off_r = grid.offsets[mask_path]

        reg_win = Window(reg_off_c, reg_off_r, grid.width, grid.height)
        mask_win = Window(mask_off_c, mask_off_r, grid.width, grid.height)

        reg_arr = reg_src.read(1, window=reg_win)
        mask_arr = mask_src.read(1, window=mask_win)

        eff_ref_nd: Union[int, float]
        if ref_no_data is not None:
            eff_ref_nd = ref_no_data
        elif reg_src.nodata is not None:
            eff_ref_nd = reg_src.nodata
        else:
            eff_ref_nd = 0

        seed_mask = _valid_seed_mask(reg_arr, eff_ref_nd)
        if not np.any(seed_mask):
            raise rsgispy.RSGISPyException(
                "No valid seed region pixels found in in_regions_img."
            )

        if np.issubdtype(mask_arr.dtype, np.floating) and np.isnan(fill_region_val):
            fill_mask = np.isnan(mask_arr)
        else:
            fill_mask = mask_arr == fill_region_val

        eff_dtype = (
            rsgis_dtype_from_np_dtype(reg_src.dtypes[0])
            if datatype is None
            else datatype
        )
        out_rio_dtype = rsgispy.get_rasterio_datatype(eff_dtype)
        out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(eff_dtype))

        if no_data_val is not None:
            eff_out_nd: Union[int, float] = no_data_val
        elif np.isnan(eff_ref_nd) and np.issubdtype(out_np_dtype, np.integer):
            eff_out_nd = 0
        else:
            eff_out_nd = eff_ref_nd

        if np.issubdtype(out_np_dtype, np.integer) and not np.isnan(eff_out_nd):
            eff_out_nd = int(eff_out_nd)

        out_arr = _propagate_region_labels(
            reg_arr=reg_arr,
            seed_mask=seed_mask,
            fill_mask=fill_mask,
            out_np_dtype=out_np_dtype,
            eff_out_nd=eff_out_nd,
            connectivity=connectivity,
            max_iters=max_iters,
            use_pbar=use_pbar,
        )

        dst = open_output_dataset(
            stack=stack,
            output_img=out_path,
            gdalformat=gdalformat,
            out_rio_dtype=out_rio_dtype,
            grid=grid,
            count=1,
            creation_opts=norm_opts,
        )
        dst.nodata = eff_out_nd
        dst.write(
            cast_result_block(out_arr, grid.height, grid.width, out_np_dtype),
            1,
        )

    maybe_pop_stats(
        output_img=out_path,
        calc_stats=calc_stats,
        no_data_val=float(eff_out_nd),
        thematic=True,
        use_pbar=use_pbar,
    )
