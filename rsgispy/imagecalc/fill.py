"""
The imagecalc.fill module provides spatial region filling utilities using
``rasterio``, including k-nearest-neighbour filling (``scipy``) and
balanced iterative region-growing filling (pure ``numpy``).
"""

from contextlib import ExitStack
import heapq
import math
from multiprocessing import Pool
import os
import tempfile
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import rasterio
from rasterio.crs import CRS  # pylint: disable=no-name-in-module
from rasterio.transform import Affine
from rasterio.windows import Window

import rsgispy
from rsgispy.imageprocess import (
    IntersectionGrid,
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
from rsgispy.tools.system import num_process_cores

try:
    import scipy.spatial  # pylint: disable=import-error
    import scipy.stats  # pylint: disable=import-error

    HAVE_SCIPY = True
except ImportError:
    scipy = None  # type: ignore[assignment]
    HAVE_SCIPY = False

__all__: List[str] = [
    "calc_fill_regions_knn",
    "region_growing_fill",
    "region_growing_fill_tiled",
]


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
            fill_mask = _fill_mask_from_arr(in_block, fill_region_val)
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
                out_f = np.zeros((win_h, win_w), dtype=np.float64)
                out_f[loc_rows, loc_cols] = fill_vals
                out_arr = cast_result_block(
                    out_f, win_h, win_w, out_np_dtype, clamp_int=True
                )
            else:
                out_arr = np.zeros((win_h, win_w), dtype=out_np_dtype)

            dst.write(out_arr, 1, window=out_win)


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


def _connectivity_spec(
    connectivity: int,
) -> Tuple[Tuple[Tuple[int, int, int], ...], int]:
    """Return ``(offsets, step_unit)`` for 4- or 8-connected Chamfer (5, 7) steps."""
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
        return offsets, 5
    return ((-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1)), 1


def _fill_mask_from_arr(
    mask_arr: np.ndarray, fill_region_val: Union[int, float]
) -> np.ndarray:
    """Return boolean mask of fillable pixels equal to ``fill_region_val``."""
    if math.isnan(float(fill_region_val)):
        if np.issubdtype(mask_arr.dtype, np.floating):
            return np.isnan(mask_arr)
        return np.zeros(mask_arr.shape, dtype=bool)
    return mask_arr == fill_region_val


def _resolve_out_dtype_and_nodata(
    reg_src: Any,
    ref_no_data: Optional[Union[int, float]],
    datatype: Optional[int],
    no_data_val: Optional[float],
) -> Tuple[Union[int, float], str, np.dtype, Union[int, float]]:
    """Resolve input nodata, output rasterio/numpy dtypes, and output nodata."""
    if ref_no_data is not None:
        eff_ref_nd: Union[int, float] = ref_no_data
    elif reg_src.nodata is not None:
        eff_ref_nd = reg_src.nodata
    else:
        eff_ref_nd = 0

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

    return eff_ref_nd, out_rio_dtype, out_np_dtype, eff_out_nd


def _run_dial_wavefront(
    pad_labels: np.ndarray,
    pad_dist: np.ndarray,
    buckets: Dict[int, List[Tuple[np.ndarray, np.ndarray]]],
    heap: List[int],
    offsets: Tuple[Tuple[int, int, int], ...],
    max_dist: Optional[int],
    inf_dist: int,
    pbar: Optional[Any] = None,
    core_bounds: Optional[Tuple[int, int]] = None,
) -> None:
    """Execute Dial-bucketed wavefront propagation in-place on ``pad_labels``/``pad_dist``."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
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
            if core_bounds is not None:
                c_h, c_w = core_bounds
                improve &= (nr >= 1) & (nr <= c_h) & (nc >= 1) & (nc <= c_w)
            if not np.any(improve):
                continue

            vr = rs[improve]
            vc = cs[improve]
            unr = nr[improve]
            unc = nc[improve]

            if pbar is not None:
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


def _seed_frontier_coords(
    seed_mask: np.ndarray,
    pad_dist: np.ndarray,
    offsets: Tuple[Tuple[int, int, int], ...],
    inf_dist: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return 1-based padded ``(rows, cols)`` of seed pixels bordering ``inf_dist``."""
    height, width = seed_mask.shape
    pad_target = pad_dist == inf_dist
    has_target_nbr = np.zeros((height, width), dtype=bool)
    for dr, dc, _ in offsets:
        has_target_nbr |= pad_target[
            1 + dr : 1 + dr + height, 1 + dc : 1 + dc + width
        ]
    frontier_r, frontier_c = np.nonzero(seed_mask & has_target_nbr)
    return frontier_r + 1, frontier_c + 1


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
        inf_dist = int(np.iinfo(np.int32).max)
        pad_dist = np.full((height + 2, width + 2), -1, dtype=np.int32)
        pad_dist[1:-1, 1:-1][fill_target] = inf_dist
        pad_dist[1:-1, 1:-1][seed_mask] = 0

        offsets, step_unit = _connectivity_spec(connectivity)
        frontier_r, frontier_c = _seed_frontier_coords(
            seed_mask, pad_dist, offsets, inf_dist
        )

        if frontier_r.size > 0:
            max_dist = None if max_iters is None else int(max_iters) * step_unit
            buckets: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = {
                0: [(frontier_r, frontier_c)]
            }
            heap: List[int] = [0]
            tqdm_cls = rsgispy.get_tqdm()
            with tqdm_cls(
                total=n_to_fill, disable=not use_pbar, desc="Growing regions"
            ) as pbar:
                _run_dial_wavefront(
                    pad_labels=pad_labels,
                    pad_dist=pad_dist,
                    buckets=buckets,
                    heap=heap,
                    offsets=offsets,
                    max_dist=max_dist,
                    inf_dist=inf_dist,
                    pbar=pbar if use_pbar else None,
                )

    out_arr = pad_labels[1:-1, 1:-1]
    if not np.all(fill_mask):
        out_arr[~fill_mask] = eff_out_nd
    return out_arr


def _extract_tile_edges(
    core_labels: np.ndarray, core_dist: np.ndarray
) -> Dict[str, Any]:
    """Extract 1-pixel border strips and corners of a tile's label and distance arrays."""
    return {
        "top_l": core_labels[0, :].copy(),
        "top_d": core_dist[0, :].copy(),
        "bot_l": core_labels[-1, :].copy(),
        "bot_d": core_dist[-1, :].copy(),
        "left_l": core_labels[:, 0].copy(),
        "left_d": core_dist[:, 0].copy(),
        "right_l": core_labels[:, -1].copy(),
        "right_d": core_dist[:, -1].copy(),
        "tl": (core_labels[0, 0], int(core_dist[0, 0])),
        "tr": (core_labels[0, -1], int(core_dist[0, -1])),
        "bl": (core_labels[-1, 0], int(core_dist[-1, 0])),
        "br": (core_labels[-1, -1], int(core_dist[-1, -1])),
    }


def _find_tile_links(
    tile_bounds: Sequence[Tuple[int, int, int, int]],
) -> List[Dict[str, Any]]:
    """
    Validate non-overlapping tile bounds ``(row0, col0, height, width)`` and
    precompute abutting edge and corner neighbour links for each tile.
    """
    # pylint: disable=too-many-locals,too-many-branches
    n_tiles = len(tile_bounds)
    links: List[Dict[str, Any]] = [
        {
            "top": [],
            "bot": [],
            "left": [],
            "right": [],
            "tl": None,
            "tr": None,
            "bl": None,
            "br": None,
        }
        for _ in range(n_tiles)
    ]

    order = sorted(range(n_tiles), key=lambda idx: tile_bounds[idx][1])
    for pos_a, idx_a in enumerate(order):
        r0_a, c0_a, h_a, w_a = tile_bounds[idx_a]
        r1_a = r0_a + h_a
        c1_a = c0_a + w_a

        for idx_b in order[pos_a + 1 :]:
            r0_b, c0_b, h_b, w_b = tile_bounds[idx_b]
            r1_b = r0_b + h_b
            c1_b = c0_b + w_b

            if c0_b > c1_a:
                break

            ov_r0 = max(r0_a, r0_b)
            ov_r1 = min(r1_a, r1_b)
            ov_c0 = max(c0_a, c0_b)
            ov_c1 = min(c1_a, c1_b)

            if ov_r0 < ov_r1 and ov_c0 < ov_c1:
                raise rsgispy.RSGISPyException(
                    f"Input tiles at indices {idx_a} and {idx_b} overlap spatially; "
                    "region_growing_fill_tiled requires non-overlapping tile extents."
                )

            # Vertical adjacency (B above A or A above B)
            if ov_c0 < ov_c1:
                sl_a = slice(ov_c0 - c0_a, ov_c1 - c0_a)
                sl_b = slice(ov_c0 - c0_b, ov_c1 - c0_b)
                if r1_b == r0_a:
                    links[idx_a]["top"].append((idx_b, sl_a, sl_b))
                    links[idx_b]["bot"].append((idx_a, sl_b, sl_a))
                elif r1_a == r0_b:
                    links[idx_a]["bot"].append((idx_b, sl_a, sl_b))
                    links[idx_b]["top"].append((idx_a, sl_b, sl_a))

            # Horizontal adjacency (B left of A or A left of B)
            if ov_r0 < ov_r1:
                sl_a = slice(ov_r0 - r0_a, ov_r1 - r0_a)
                sl_b = slice(ov_r0 - r0_b, ov_r1 - r0_b)
                if c1_b == c0_a:
                    links[idx_a]["left"].append((idx_b, sl_a, sl_b))
                    links[idx_b]["right"].append((idx_a, sl_b, sl_a))
                elif c1_a == c0_b:
                    links[idx_a]["right"].append((idx_b, sl_a, sl_b))
                    links[idx_b]["left"].append((idx_a, sl_b, sl_a))

            # Corner checks (both exact diagonal corner tiles and staggered edges)
            # A's TL corner (r0_a - 1, c0_a - 1) inside B -> B's BR
            if r0_b <= r0_a - 1 < r1_b and c0_b <= c0_a - 1 < c1_b:
                if r1_b == r0_a:
                    links[idx_a]["tl"] = (idx_b, "bot", (c0_a - 1) - c0_b)
                elif c1_b == c0_a:
                    links[idx_a]["tl"] = (idx_b, "right", (r0_a - 1) - r0_b)
            # A's TR corner (r0_a - 1, c1_a) inside B
            if r0_b <= r0_a - 1 < r1_b and c0_b <= c1_a < c1_b:
                if r1_b == r0_a:
                    links[idx_a]["tr"] = (idx_b, "bot", c1_a - c0_b)
                elif c0_b == c1_a:
                    links[idx_a]["tr"] = (idx_b, "left", (r0_a - 1) - r0_b)
            # A's BL corner (r1_a, c0_a - 1) inside B
            if r0_b <= r1_a < r1_b and c0_b <= c0_a - 1 < c1_b:
                if r0_b == r1_a:
                    links[idx_a]["bl"] = (idx_b, "top", (c0_a - 1) - c0_b)
                elif c1_b == c0_a:
                    links[idx_a]["bl"] = (idx_b, "right", r1_a - r0_b)
            # A's BR corner (r1_a, c1_a) inside B
            if r0_b <= r1_a < r1_b and c0_b <= c1_a < c1_b:
                if r0_b == r1_a:
                    links[idx_a]["br"] = (idx_b, "top", c1_a - c0_b)
                elif c0_b == c1_a:
                    links[idx_a]["br"] = (idx_b, "left", r1_a - r0_b)

            # Symmetric corners from B's perspective where B is to the right/abutting A
            if r0_a <= r0_b - 1 < r1_a and c0_a <= c0_b - 1 < c1_a:
                if r1_a == r0_b:
                    links[idx_b]["tl"] = (idx_a, "bot", (c0_b - 1) - c0_a)
                elif c1_a == c0_b:
                    links[idx_b]["tl"] = (idx_a, "right", (r0_b - 1) - r0_a)
            if r0_a <= r0_b - 1 < r1_a and c0_a <= c1_b < c1_a:
                if r1_a == r0_b:
                    links[idx_b]["tr"] = (idx_a, "bot", c1_b - c0_a)
                elif c0_a == c1_b:
                    links[idx_b]["tr"] = (idx_a, "left", (r0_b - 1) - r0_a)
            if r0_a <= r1_b < r1_a and c0_a <= c0_b - 1 < c1_a:
                if r0_a == r1_b:
                    links[idx_b]["bl"] = (idx_a, "top", (c0_b - 1) - c0_a)
                elif c1_a == c0_b:
                    links[idx_b]["bl"] = (idx_a, "right", r1_b - r0_a)
            if r0_a <= r1_b < r1_a and c0_a <= c1_b < c1_a:
                if r0_a == r1_b:
                    links[idx_b]["br"] = (idx_a, "top", c1_b - c0_a)
                elif c0_a == c1_b:
                    links[idx_b]["br"] = (idx_a, "left", r1_b - r0_a)

    return links


def _build_incoming_halo(
    tile_edges: Mapping[int, Optional[Dict[str, Any]]],
    tile_link: Mapping[str, Any],
    tile_h: int,
    tile_w: int,
    eff_out_nd: Union[int, float],
    out_np_dtype: np.dtype,
) -> Dict[str, Any]:
    """Assemble the 1-pixel incoming halo strips and corners for a single tile."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    halo: Dict[str, Any] = {
        "top_l": np.full(tile_w, eff_out_nd, dtype=out_np_dtype),
        "top_d": np.full(tile_w, -1, dtype=np.int32),
        "bot_l": np.full(tile_w, eff_out_nd, dtype=out_np_dtype),
        "bot_d": np.full(tile_w, -1, dtype=np.int32),
        "left_l": np.full(tile_h, eff_out_nd, dtype=out_np_dtype),
        "left_d": np.full(tile_h, -1, dtype=np.int32),
        "right_l": np.full(tile_h, eff_out_nd, dtype=out_np_dtype),
        "right_d": np.full(tile_h, -1, dtype=np.int32),
        "tl": (eff_out_nd, -1),
        "tr": (eff_out_nd, -1),
        "bl": (eff_out_nd, -1),
        "br": (eff_out_nd, -1),
    }

    edge_pairs = (
        ("top", "bot"),
        ("bot", "top"),
        ("left", "right"),
        ("right", "left"),
    )
    for my_side, nbr_side in edge_pairs:
        for nbr_idx, sl_a, sl_b in tile_link[my_side]:
            nbr_e = tile_edges.get(nbr_idx)
            if nbr_e is not None:
                halo[f"{my_side}_l"][sl_a] = nbr_e[f"{nbr_side}_l"][sl_b]
                halo[f"{my_side}_d"][sl_a] = nbr_e[f"{nbr_side}_d"][sl_b]

    for c_key in ("tl", "tr", "bl", "br"):
        c_spec = tile_link[c_key]
        if c_spec is not None:
            nbr_idx, edge_name, pos = c_spec
            nbr_e = tile_edges.get(nbr_idx)
            if nbr_e is not None:
                halo[c_key] = (
                    nbr_e[f"{edge_name}_l"][pos],
                    int(nbr_e[f"{edge_name}_d"][pos]),
                )

    return halo


def _edge_has_improvement(
    halo_d_1d: np.ndarray,
    corner_a_d: int,
    corner_b_d: int,
    core_d_1d: np.ndarray,
    w_card: int,
    w_diag: Optional[int],
    max_dist: Optional[int],
    inf_dist: int,
) -> bool:
    """Return True if a 1D halo border strip + corners can improve ``core_d_1d``."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    padded = np.empty(halo_d_1d.size + 2, dtype=np.int32)
    padded[0] = corner_a_d
    padded[1:-1] = halo_d_1d
    padded[-1] = corner_b_d

    limit = inf_dist if max_dist is None else min(inf_dist, max_dist)

    def _check(src_d: np.ndarray, weight: int) -> bool:
        valid = (src_d >= 0) & (src_d < limit)
        if not np.any(valid):
            return False
        return bool(
            np.any(src_d[valid].astype(np.int64) + weight < core_d_1d[valid])
        )

    if _check(padded[1:-1], w_card):
        return True
    if w_diag is not None:
        if _check(padded[:-2], w_diag) or _check(padded[2:], w_diag):
            return True
    return False


def _tile_has_incoming_improvement(
    my_edges: Optional[Dict[str, Any]],
    halo: Mapping[str, Any],
    connectivity: int,
    max_dist: Optional[int],
    inf_dist: int,
) -> bool:
    """Check in memory whether incoming ``halo`` can improve ``my_edges``."""
    if my_edges is None:
        return False

    w_card = 5 if connectivity == 8 else 1
    w_diag: Optional[int] = 7 if connectivity == 8 else None
    limit = inf_dist if max_dist is None else min(inf_dist, max_dist)

    checks = (
        (halo["top_d"], halo["tl"][1], halo["tr"][1], my_edges["top_d"]),
        (halo["bot_d"], halo["bl"][1], halo["br"][1], my_edges["bot_d"]),
        (halo["left_d"], halo["tl"][1], halo["bl"][1], my_edges["left_d"]),
        (halo["right_d"], halo["tr"][1], halo["br"][1], my_edges["right_d"]),
    )
    for halo_1d, ca_d, cb_d, core_1d in checks:
        if _edge_has_improvement(
            halo_1d, ca_d, cb_d, core_1d, w_card, w_diag, max_dist, inf_dist
        ):
            return True

    if w_diag is not None:
        for h_c_key, my_c_key in (
            ("tl", "tl"),
            ("tr", "tr"),
            ("bl", "bl"),
            ("br", "br"),
        ):
            c_d = int(halo[h_c_key][1])
            if 0 <= c_d < limit and c_d + w_diag < my_edges[my_c_key][1]:
                return True

    return False


def _inject_halo_into_tile(
    pad_labels: np.ndarray,
    pad_dist: np.ndarray,
    halo: Mapping[str, Any],
    offsets: Tuple[Tuple[int, int, int], ...],
    max_dist: Optional[int],
    inf_dist: int,
    buckets: Dict[int, List[Tuple[np.ndarray, np.ndarray]]],
    heap: List[int],
) -> int:
    """
    Write incoming neighbour halo strips into the 1-px border of ``pad_labels`` /
    ``pad_dist`` and push any improved core border pixels into ``buckets`` / ``heap``.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    tile_h = pad_labels.shape[0] - 2
    tile_w = pad_labels.shape[1] - 2

    pad_labels[0, 1:-1] = halo["top_l"]
    pad_dist[0, 1:-1] = halo["top_d"]
    pad_labels[-1, 1:-1] = halo["bot_l"]
    pad_dist[-1, 1:-1] = halo["bot_d"]
    pad_labels[1:-1, 0] = halo["left_l"]
    pad_dist[1:-1, 0] = halo["left_d"]
    pad_labels[1:-1, -1] = halo["right_l"]
    pad_dist[1:-1, -1] = halo["right_d"]

    pad_labels[0, 0], pad_dist[0, 0] = halo["tl"]
    pad_labels[0, -1], pad_dist[0, -1] = halo["tr"]
    pad_labels[-1, 0], pad_dist[-1, 0] = halo["bl"]
    pad_labels[-1, -1], pad_dist[-1, -1] = halo["br"]

    top_cols = np.arange(tile_w + 2, dtype=np.intp)
    side_rows = np.arange(1, tile_h + 1, dtype=np.intp)
    hr = np.concatenate(
        [
            np.zeros(tile_w + 2, dtype=np.intp),
            np.full(tile_w + 2, tile_h + 1, dtype=np.intp),
            side_rows,
            side_rows,
        ]
    )
    hc = np.concatenate(
        [
            top_cols,
            top_cols,
            np.zeros(tile_h, dtype=np.intp),
            np.full(tile_h, tile_w + 1, dtype=np.intp),
        ]
    )

    h_dist = pad_dist[hr, hc]
    limit = inf_dist if max_dist is None else min(inf_dist, max_dist)
    valid = (h_dist >= 0) & (h_dist < limit)
    if not np.any(valid):
        return 0
    hr = hr[valid]
    hc = hc[valid]

    n_improved = 0
    for dr, dc, weight in offsets:
        nr = hr + dr
        nc = hc + dc
        in_core = (nr >= 1) & (nr <= tile_h) & (nc >= 1) & (nc <= tile_w)
        if not np.any(in_core):
            continue

        vr = hr[in_core]
        vc = hc[in_core]
        cnr = nr[in_core]
        cnc = nc[in_core]
        nd_vals = pad_dist[vr, vc].astype(np.int64) + weight
        cur_nbr_dist = pad_dist[cnr, cnc]
        improve = nd_vals < cur_nbr_dist
        if not np.any(improve):
            continue

        unr = cnr[improve]
        unc = cnc[improve]
        u_nd = nd_vals[improve].astype(np.int32)
        pad_dist[unr, unc] = u_nd
        pad_labels[unr, unc] = pad_labels[vr[improve], vc[improve]]
        n_improved += int(unr.size)

        order = np.argsort(u_nd, kind="mergesort")
        s_d = u_nd[order]
        s_r = unr[order]
        s_c = unc[order]
        split_idxs = np.nonzero(np.diff(s_d))[0] + 1
        for d_grp, r_grp, c_grp in zip(
            np.split(s_d, split_idxs),
            np.split(s_r, split_idxs),
            np.split(s_c, split_idxs),
        ):
            d_val = int(d_grp[0])
            if d_val not in buckets:
                buckets[d_val] = [(r_grp, c_grp)]
                heapq.heappush(heap, d_val)
            else:
                buckets[d_val].append((r_grp, c_grp))

    return n_improved


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

    For very large extents (such as country-wide mosaics with offshore EEZ
    zones) that exceed available RAM, use :func:`region_growing_fill_tiled`
    to obtain the exact same global result without tiling artefacts.

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

        eff_ref_nd, out_rio_dtype, out_np_dtype, eff_out_nd = (
            _resolve_out_dtype_and_nodata(
                reg_src, ref_no_data, datatype, no_data_val
            )
        )

        seed_mask = _valid_seed_mask(reg_arr, eff_ref_nd)
        if not np.any(seed_mask):
            raise rsgispy.RSGISPyException(
                "No valid seed region pixels found in in_regions_img."
            )

        fill_mask = _fill_mask_from_arr(mask_arr, fill_region_val)

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
        thematic=bool(
            np.issubdtype(out_np_dtype, np.integer) and eff_out_nd == 0
        ),
        use_pbar=use_pbar,
    )


def _inspect_tile_inputs(
    reg_paths: Sequence[str],
    mask_paths: Sequence[str],
    out_paths: Sequence[str],
    ref_no_data: Optional[Union[int, float]],
    datatype: Optional[int],
    no_data_val: Optional[float],
) -> Tuple[
    List[Dict[str, Any]],
    Union[int, float],
    str,
    np.dtype,
    Union[int, float],
]:
    """
    Validate tile inputs, compute per-tile intersection windows, enforce a
    consistent non-rotated CRS/resolution/pixel-grid across all tiles, and
    resolve output dtype and nodata settings.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    tile_infos: List[Dict[str, Any]] = []
    base_crs: Optional[CRS] = None
    base_transform: Optional[Affine] = None
    eff_ref_nd: Union[int, float] = 0
    out_rio_dtype = "uint8"
    out_np_dtype = np.dtype(np.uint8)
    eff_out_nd: Union[int, float] = 0

    for idx, (reg_p, mask_p) in enumerate(zip(reg_paths, mask_paths)):
        ensure_file_exists(reg_p, label=f"Regions image [{idx}]")
        ensure_file_exists(mask_p, label=f"Fill regions image [{idx}]")

        if not do_gdal_layers_have_same_proj(reg_p, mask_p):
            raise rsgispy.RSGISPyException(
                f"Tile {idx}: regions image and fill regions image "
                "do not have the same projection."
            )

        with ExitStack() as stack:
            try:
                reg_src = stack.enter_context(rasterio.open(reg_p, "r"))
            except Exception as err:
                raise rsgispy.RSGISPyException(
                    f"Could not open regions image '{reg_p}': {err}"
                ) from err
            try:
                mask_src = stack.enter_context(rasterio.open(mask_p, "r"))
            except Exception as err:
                raise rsgispy.RSGISPyException(
                    f"Could not open fill regions image '{mask_p}': {err}"
                ) from err

            grid = compute_spatial_intersection({reg_p: reg_src, mask_p: mask_src})
            reg_off_c, reg_off_r = grid.offsets[reg_p]
            mask_off_c, mask_off_r = grid.offsets[mask_p]
            t = grid.transform

            if abs(t.b) > 1e-9 or abs(t.d) > 1e-9:
                raise rsgispy.RSGISPyException(
                    f"Tile {idx} has a rotated transform, which is not supported."
                )

            if idx == 0:
                base_crs = grid.crs
                base_transform = t
                eff_ref_nd, out_rio_dtype, out_np_dtype, eff_out_nd = (
                    _resolve_out_dtype_and_nodata(
                        reg_src, ref_no_data, datatype, no_data_val
                    )
                )
                r0, c0 = 0, 0
            else:
                if base_transform is None:
                    raise rsgispy.RSGISPyException(
                        "Failed to determine base transform from tile 0."
                    )
                if (grid.crs is None) != (base_crs is None) or (
                    grid.crs is not None and grid.crs != base_crs
                ):
                    raise rsgispy.RSGISPyException(
                        f"Tile {idx} has a different CRS ({grid.crs}) "
                        f"from tile 0 ({base_crs})."
                    )
                if not (
                    math.isclose(t.a, base_transform.a, rel_tol=1e-5, abs_tol=1e-9)
                    and math.isclose(t.e, base_transform.e, rel_tol=1e-5, abs_tol=1e-9)
                ):
                    raise rsgispy.RSGISPyException(
                        f"Tile {idx} has a different pixel resolution "
                        f"({t.a}, {t.e}) from tile 0 "
                        f"({base_transform.a}, {base_transform.e})."
                    )
                col_f = (t.c - base_transform.c) / base_transform.a
                row_f = (t.f - base_transform.f) / base_transform.e
                c0 = int(round(col_f))
                r0 = int(round(row_f))
                if abs(col_f - c0) > 1e-3 or abs(row_f - r0) > 1e-3:
                    raise rsgispy.RSGISPyException(
                        f"Tile {idx} origin is not aligned to the pixel grid of tile 0."
                    )

            tile_infos.append(
                {
                    "idx": idx,
                    "reg_path": reg_p,
                    "mask_path": mask_p,
                    "out_path": out_paths[idx],
                    "reg_win": (reg_off_c, reg_off_r, grid.width, grid.height),
                    "mask_win": (mask_off_c, mask_off_r, grid.width, grid.height),
                    "bounds": (r0, c0, grid.height, grid.width),
                    "transform": (t.a, t.b, t.c, t.d, t.e, t.f),
                    "crs_wkt": grid.crs.to_wkt() if grid.crs is not None else None,
                }
            )

    return tile_infos, eff_ref_nd, out_rio_dtype, out_np_dtype, eff_out_nd


def _tiled_init_worker(
    task: Tuple[Any, ...],
) -> Tuple[int, bool, bool, bool, Optional[Dict[str, Any]]]:
    """Top-level multiprocessing worker for Pass 0 local Dial initialisation."""
    # pylint: disable=too-many-locals
    (
        idx,
        reg_path,
        mask_path,
        reg_win_t,
        mask_win_t,
        eff_ref_nd,
        fill_region_val,
        out_np_dtype_str,
        eff_out_nd,
        offsets,
        max_dist,
        inf_dist,
        lbl_path,
        dst_path,
    ) = task

    out_np_dtype = np.dtype(out_np_dtype_str)
    with rasterio.open(reg_path, "r") as reg_src:
        reg_arr = reg_src.read(1, window=Window(*reg_win_t))
    with rasterio.open(mask_path, "r") as mask_src:
        mask_arr = mask_src.read(1, window=Window(*mask_win_t))

    seed_mask = _valid_seed_mask(reg_arr, eff_ref_nd)
    fill_mask = _fill_mask_from_arr(mask_arr, fill_region_val)
    has_seed = bool(np.any(seed_mask))
    has_fill = bool(np.any(fill_mask))

    if not (has_seed or has_fill):
        return idx, False, False, False, None

    th, tw = reg_arr.shape
    fill_target = fill_mask & (~seed_mask)
    has_outside_seed = bool(has_seed and np.any(seed_mask & (~fill_mask)))

    pad_labels = np.full((th + 2, tw + 2), eff_out_nd, dtype=out_np_dtype)
    pad_dist = np.full((th + 2, tw + 2), -1, dtype=np.int32)
    pad_dist[1:-1, 1:-1][fill_target] = inf_dist

    if has_seed:
        pad_labels[1:-1, 1:-1][seed_mask] = reg_arr[seed_mask].astype(
            out_np_dtype, copy=False
        )
        pad_dist[1:-1, 1:-1][seed_mask] = 0

    if has_seed and np.any(fill_target):
        fr, fc = _seed_frontier_coords(seed_mask, pad_dist, offsets, inf_dist)
        if fr.size > 0:
            buckets: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = {0: [(fr, fc)]}
            heap: List[int] = [0]
            _run_dial_wavefront(
                pad_labels=pad_labels,
                pad_dist=pad_dist,
                buckets=buckets,
                heap=heap,
                offsets=offsets,
                max_dist=max_dist,
                inf_dist=inf_dist,
                core_bounds=(th, tw),
            )

    core_labels = pad_labels[1:-1, 1:-1]
    core_dist = pad_dist[1:-1, 1:-1]
    np.save(lbl_path, core_labels)
    np.save(dst_path, core_dist)
    edges = _extract_tile_edges(core_labels, core_dist)
    return idx, True, has_seed, has_outside_seed, edges


def _tiled_sweep_worker(
    task: Tuple[Any, ...],
) -> Tuple[int, bool, Optional[Dict[str, Any]]]:
    """Top-level multiprocessing worker for boundary halo injection & relaxation."""
    # pylint: disable=too-many-locals
    (
        idx,
        th,
        tw,
        lbl_path,
        dst_path,
        halo,
        eff_out_nd,
        out_np_dtype_str,
        offsets,
        max_dist,
        inf_dist,
    ) = task

    out_np_dtype = np.dtype(out_np_dtype_str)
    pad_labels = np.full((th + 2, tw + 2), eff_out_nd, dtype=out_np_dtype)
    pad_dist = np.full((th + 2, tw + 2), -1, dtype=np.int32)
    pad_labels[1:-1, 1:-1] = np.load(lbl_path)
    pad_dist[1:-1, 1:-1] = np.load(dst_path)

    buckets: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = {}
    heap: List[int] = []
    n_imp = _inject_halo_into_tile(
        pad_labels=pad_labels,
        pad_dist=pad_dist,
        halo=halo,
        offsets=offsets,
        max_dist=max_dist,
        inf_dist=inf_dist,
        buckets=buckets,
        heap=heap,
    )
    if n_imp == 0:
        return idx, False, None

    _run_dial_wavefront(
        pad_labels=pad_labels,
        pad_dist=pad_dist,
        buckets=buckets,
        heap=heap,
        offsets=offsets,
        max_dist=max_dist,
        inf_dist=inf_dist,
        core_bounds=(th, tw),
    )
    core_labels = pad_labels[1:-1, 1:-1]
    core_dist = pad_dist[1:-1, 1:-1]
    np.save(lbl_path, core_labels)
    np.save(dst_path, core_dist)
    return idx, True, _extract_tile_edges(core_labels, core_dist)


def _tiled_write_worker(task: Tuple[Any, ...]) -> int:
    """Top-level multiprocessing worker for writing an output tile dataset."""
    # pylint: disable=too-many-locals
    (
        idx,
        out_path,
        mask_path,
        mask_win_t,
        th,
        tw,
        transform_t,
        crs_wkt,
        gdalformat,
        out_rio_dtype,
        out_np_dtype_str,
        eff_out_nd,
        norm_opts,
        fill_region_val,
        active,
        has_outside_seed,
        lbl_path,
        calc_stats,
    ) = task

    out_np_dtype = np.dtype(out_np_dtype_str)
    if active:
        tile_out = np.load(lbl_path)
        if has_outside_seed:
            with rasterio.open(mask_path, "r") as mask_src:
                mask_arr = mask_src.read(1, window=Window(*mask_win_t))
            fill_mask = _fill_mask_from_arr(mask_arr, fill_region_val)
            if not np.all(fill_mask):
                tile_out[~fill_mask] = eff_out_nd
    else:
        tile_out = np.full((th, tw), eff_out_nd, dtype=out_np_dtype)

    out_parent = os.path.dirname(os.path.abspath(out_path))
    if out_parent:
        os.makedirs(out_parent, exist_ok=True)

    tile_grid = IntersectionGrid(
        width=tw,
        height=th,
        transform=Affine(*transform_t),
        crs=CRS.from_wkt(crs_wkt) if crs_wkt is not None else None,
        offsets={},
    )
    with ExitStack() as stack:
        dst = open_output_dataset(
            stack=stack,
            output_img=out_path,
            gdalformat=gdalformat,
            out_rio_dtype=out_rio_dtype,
            grid=tile_grid,
            count=1,
            creation_opts=norm_opts,
        )
        dst.nodata = eff_out_nd
        dst.write(cast_result_block(tile_out, th, tw, out_np_dtype), 1)

    maybe_pop_stats(
        output_img=out_path,
        calc_stats=calc_stats,
        no_data_val=float(eff_out_nd),
        thematic=bool(
            np.issubdtype(out_np_dtype, np.integer) and eff_out_nd == 0
        ),
        use_pbar=False,
    )
    return idx


def region_growing_fill_tiled(
    in_regions_imgs: Sequence[Union[str, os.PathLike]],
    in_fill_regions_imgs: Sequence[Union[str, os.PathLike]],
    output_imgs: Sequence[Union[str, os.PathLike]],
    ref_no_data: Optional[Union[int, float]] = None,
    fill_region_val: Union[int, float] = 1,
    connectivity: int = 8,
    max_iters: Optional[int] = None,
    max_sweeps: Optional[int] = None,
    n_cores: int = 1,
    tmp_dir: Optional[Union[str, os.PathLike]] = None,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
    creation_opts: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
) -> None:
    """
    Tiled multi-image version of :func:`region_growing_fill` using iterative
    multi-sweep tile-boundary distance propagation and optional multiprocessing
    to eliminate tiling artefacts across a set of abutting image tiles.

    Takes matching sequences of input seed region tiles (``in_regions_imgs``),
    input fillable target mask tiles (``in_fill_regions_imgs``), and output
    tile paths (``output_imgs``). All tiles must share the same CRS, pixel
    resolution, and aligned pixel grid, and must be spatially non-overlapping.

    Each tile persists its exact cumulative geodesic distances (Chamfer
    ``(5, 7)`` for ``connectivity=8`` or unit steps for ``connectivity=4``) in
    a temporary directory and exchanges 1-pixel shared border and corner
    ``(label, distance)`` strips with abutting neighbour tiles across
    relaxation sweeps until global convergence. Because full cumulative
    distances are carried across tile boundaries, regions propagate seamlessly
    across tiles (including tiles with zero initial seeds such as offshore EEZ
    tiles) without tiling seams.

    :param in_regions_imgs: Sequence of paths to input seed region tile images
                            (band 1).
    :param in_fill_regions_imgs: Sequence of paths to input target fill mask
                                 tile images (band 1), ordered 1-to-1 with
                                 ``in_regions_imgs``.
    :param output_imgs: Sequence of output tile file paths, ordered 1-to-1 with
                        ``in_regions_imgs``.
    :param ref_no_data: No-data value in ``in_regions_imgs`` excluded from seed
                        regions. If ``None`` (default), uses tile 0's input
                        header no-data value, or ``0`` if unset.
    :param fill_region_val: Pixel value in ``in_fill_regions_imgs`` defining
                            the fillable domain (Default = ``1``).
    :param connectivity: Neighbourhood connectivity: ``4`` (cardinal) or ``8``
                         (Chamfer ``(5, 7)`` balanced diagonal+cardinal,
                         Default = ``8``).
    :param max_iters: Optional maximum number of expansion steps per cardinal
                      pixel unit. If ``None`` (default), expands until all
                      reachable pixels are filled.
    :param max_sweeps: Optional maximum number of boundary-relaxation sweeps
                       after the initial pass (Default = ``None``, iterate
                       until no tile boundary improves).
    :param n_cores: Number of parallel worker processes (Default = ``1``).
                    Values ``<= 0`` automatically use all available CPU cores.
    :param tmp_dir: Optional parent directory for temporary per-tile state
                    files (automatically cleaned up on exit; defaults to system
                    temp directory).
    :param gdalformat: GDAL/rasterio driver format for output files
                       (Default = ``'GTIFF'``).
    :param datatype: Output data type constant from ``rsgispy.TYPE_*``. When
                     ``None`` (default), inferred from tile 0 of
                     ``in_regions_imgs``.
    :param no_data_val: Output no-data value. When ``None`` (default), uses
                        ``ref_no_data`` (or ``0`` if ``ref_no_data`` is NaN
                        and the output data type is integer).
    :param calc_stats: If ``True``, populate image statistics and overview
                       pyramids on each output tile (Default = ``True``).
    :param use_pbar: If ``True``, display tqdm progress bars during tiled
                     passes and output writing (Default = ``True``).
    :param creation_opts: Optional GDAL creation options as a mapping or sequence
                          of ``'KEY=VALUE'`` strings.

    .. note::

        When calling this function from a Python script with multiple cores
        (``n_cores > 1`` or ``n_cores <= 0``), wrap the call inside an
        ``if __name__ == "__main__":`` guard. On macOS, Windows, and Python
        3.14+ on Linux, Python's ``multiprocessing`` uses ``spawn`` (or
        ``forkserver``), which re-imports the calling script in every spawned
        worker process. Without the guard, each worker re-executes the
        top-level call and fails during process bootstrapping with a
        ``RuntimeError``.

    .. code:: python

        import rsgispy.imagecalc.fill

        if __name__ == "__main__":
            rsgispy.imagecalc.fill.region_growing_fill_tiled(
                in_regions_imgs=["tileA_rgn.tif", "tileB_rgn.tif"],
                in_fill_regions_imgs=["tileA_ext.tif", "tileB_ext.tif"],
                output_imgs=["tileA_out.tif", "tileB_out.tif"],
                n_cores=4,
                creation_opts={"TILED": "YES", "COMPRESS": "DEFLATE"},
            )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    reg_paths = [os.fspath(p) for p in in_regions_imgs]
    mask_paths = [os.fspath(p) for p in in_fill_regions_imgs]
    out_paths = [os.fspath(p) for p in output_imgs]

    n_tiles = len(reg_paths)
    if n_tiles == 0:
        raise rsgispy.RSGISPyException(
            "Input tile sequences must not be empty."
        )
    if n_tiles != len(mask_paths) or n_tiles != len(out_paths):
        raise rsgispy.RSGISPyException(
            "in_regions_imgs, in_fill_regions_imgs, and output_imgs "
            "must have the same length."
        )
    if connectivity not in (4, 8):
        raise rsgispy.RSGISPyException(
            f"connectivity must be 4 or 8, got {connectivity}."
        )
    if max_iters is not None and max_iters < 0:
        raise rsgispy.RSGISPyException("max_iters must be >= 0 or None.")
    if max_sweeps is not None and max_sweeps < 0:
        raise rsgispy.RSGISPyException("max_sweeps must be >= 0 or None.")

    tile_infos, eff_ref_nd, out_rio_dtype, out_np_dtype, eff_out_nd = (
        _inspect_tile_inputs(
            reg_paths=reg_paths,
            mask_paths=mask_paths,
            out_paths=out_paths,
            ref_no_data=ref_no_data,
            datatype=datatype,
            no_data_val=no_data_val,
        )
    )
    tile_bounds: List[Tuple[int, int, int, int]] = [
        info["bounds"] for info in tile_infos
    ]
    tile_links = _find_tile_links(tile_bounds)

    norm_opts = normalise_creation_options(creation_opts)
    offsets, step_unit = _connectivity_spec(connectivity)
    max_dist = None if max_iters is None else int(max_iters) * step_unit
    inf_dist = int(np.iinfo(np.int32).max)
    eff_cores = min(
        num_process_cores() if n_cores <= 0 else int(n_cores), n_tiles
    )
    out_np_dtype_str = out_np_dtype.str
    tqdm_cls = rsgispy.get_tqdm()

    tmp_parent: Optional[str] = None
    if tmp_dir is not None:
        tmp_parent = os.fspath(tmp_dir)
        os.makedirs(tmp_parent, exist_ok=True)

    with ExitStack() as stack:
        work_dir = stack.enter_context(
            tempfile.TemporaryDirectory(prefix="rsgispy_rg_", dir=tmp_parent)
        )
        state_paths = [
            (
                os.path.join(work_dir, f"lbl_{idx}.npy"),
                os.path.join(work_dir, f"dst_{idx}.npy"),
            )
            for idx in range(n_tiles)
        ]

        pool: Optional[Any] = None
        if eff_cores > 1 and n_tiles > 1:
            pool = stack.enter_context(Pool(eff_cores))

        tile_edges: Dict[int, Optional[Dict[str, Any]]] = {}
        tile_outside_seed: Dict[int, bool] = {}
        any_seed = False

        # Pass 0: local Dial propagation per tile
        init_tasks = [
            (
                idx,
                tile_infos[idx]["reg_path"],
                tile_infos[idx]["mask_path"],
                tile_infos[idx]["reg_win"],
                tile_infos[idx]["mask_win"],
                eff_ref_nd,
                fill_region_val,
                out_np_dtype_str,
                eff_out_nd,
                offsets,
                max_dist,
                inf_dist,
                state_paths[idx][0],
                state_paths[idx][1],
            )
            for idx in range(n_tiles)
        ]
        if pool is not None:
            init_iter = pool.imap_unordered(_tiled_init_worker, init_tasks)
        else:
            init_iter = (_tiled_init_worker(t) for t in init_tasks)

        for idx, active, has_seed, has_outside_seed, edges in tqdm_cls(
            init_iter,
            total=n_tiles,
            disable=not use_pbar,
            desc="Growing regions (init)",
        ):
            any_seed = any_seed or has_seed
            if active:
                tile_edges[idx] = edges
                tile_outside_seed[idx] = has_outside_seed
            else:
                tile_edges[idx] = None
                tile_outside_seed[idx] = False

        if not any_seed:
            raise rsgispy.RSGISPyException(
                "No valid seed region pixels found in in_regions_imgs."
            )

        # Iterative boundary-relaxation sweeps across abutting tiles
        sweep_orders = (
            sorted(
                range(n_tiles),
                key=lambda i: (-tile_bounds[i][0], -tile_bounds[i][1]),
            ),
            sorted(
                range(n_tiles),
                key=lambda i: (tile_bounds[i][0], -tile_bounds[i][1]),
            ),
            sorted(
                range(n_tiles),
                key=lambda i: (-tile_bounds[i][0], tile_bounds[i][1]),
            ),
            sorted(
                range(n_tiles),
                key=lambda i: (tile_bounds[i][0], tile_bounds[i][1]),
            ),
        )
        sweep_idx = 0
        while max_sweeps is None or sweep_idx < max_sweeps:
            any_changed = False
            if pool is None:
                order = sweep_orders[sweep_idx % 4]
                with tqdm_cls(
                    total=n_tiles,
                    disable=not use_pbar,
                    desc=f"Growing regions (sweep {sweep_idx + 1})",
                ) as pbar:
                    for idx in order:
                        my_edges = tile_edges[idx]
                        _, _, th, tw = tile_bounds[idx]
                        if my_edges is not None:
                            halo = _build_incoming_halo(
                                tile_edges=tile_edges,
                                tile_link=tile_links[idx],
                                tile_h=th,
                                tile_w=tw,
                                eff_out_nd=eff_out_nd,
                                out_np_dtype=out_np_dtype,
                            )
                            if _tile_has_incoming_improvement(
                                my_edges=my_edges,
                                halo=halo,
                                connectivity=connectivity,
                                max_dist=max_dist,
                                inf_dist=inf_dist,
                            ):
                                lbl_p, dst_p = state_paths[idx]
                                _, changed, new_edges = _tiled_sweep_worker(
                                    (
                                        idx,
                                        th,
                                        tw,
                                        lbl_p,
                                        dst_p,
                                        halo,
                                        eff_out_nd,
                                        out_np_dtype_str,
                                        offsets,
                                        max_dist,
                                        inf_dist,
                                    )
                                )
                                if changed:
                                    tile_edges[idx] = new_edges
                                    any_changed = True
                        pbar.update(1)
            else:
                dirty_tasks: List[Tuple[Any, ...]] = []
                for idx in range(n_tiles):
                    my_edges = tile_edges[idx]
                    if my_edges is None:
                        continue
                    _, _, th, tw = tile_bounds[idx]
                    halo = _build_incoming_halo(
                        tile_edges=tile_edges,
                        tile_link=tile_links[idx],
                        tile_h=th,
                        tile_w=tw,
                        eff_out_nd=eff_out_nd,
                        out_np_dtype=out_np_dtype,
                    )
                    if _tile_has_incoming_improvement(
                        my_edges=my_edges,
                        halo=halo,
                        connectivity=connectivity,
                        max_dist=max_dist,
                        inf_dist=inf_dist,
                    ):
                        lbl_p, dst_p = state_paths[idx]
                        dirty_tasks.append(
                            (
                                idx,
                                th,
                                tw,
                                lbl_p,
                                dst_p,
                                halo,
                                eff_out_nd,
                                out_np_dtype_str,
                                offsets,
                                max_dist,
                                inf_dist,
                            )
                        )
                if not dirty_tasks:
                    break

                if len(dirty_tasks) > 1:
                    sweep_iter = pool.imap_unordered(
                        _tiled_sweep_worker, dirty_tasks
                    )
                else:
                    sweep_iter = (_tiled_sweep_worker(t) for t in dirty_tasks)

                for idx, changed, new_edges in tqdm_cls(
                    sweep_iter,
                    total=len(dirty_tasks),
                    disable=not use_pbar,
                    desc=f"Growing regions (sweep {sweep_idx + 1})",
                ):
                    if changed:
                        tile_edges[idx] = new_edges
                        any_changed = True

            sweep_idx += 1
            if not any_changed:
                break

        # Write seamless output raster tile by tile
        write_tasks = [
            (
                idx,
                tile_infos[idx]["out_path"],
                tile_infos[idx]["mask_path"],
                tile_infos[idx]["mask_win"],
                tile_bounds[idx][2],
                tile_bounds[idx][3],
                tile_infos[idx]["transform"],
                tile_infos[idx]["crs_wkt"],
                gdalformat,
                out_rio_dtype,
                out_np_dtype_str,
                eff_out_nd,
                norm_opts,
                fill_region_val,
                tile_edges[idx] is not None,
                tile_outside_seed[idx],
                state_paths[idx][0],
                calc_stats,
            )
            for idx in range(n_tiles)
        ]
        if pool is not None:
            write_iter = pool.imap_unordered(_tiled_write_worker, write_tasks)
        else:
            write_iter = (_tiled_write_worker(t) for t in write_tasks)

        for _ in tqdm_cls(
            write_iter,
            total=n_tiles,
            disable=not use_pbar,
            desc="Writing output tiles",
        ):
            pass
