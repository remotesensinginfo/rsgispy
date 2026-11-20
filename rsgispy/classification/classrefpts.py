# pylint: disable=too-many-lines
"""
Reference point generation and accuracy point population utilities for
classification accuracy assessment.
"""

import math
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import Affine, xy as transform_xy
from rasterio.windows import Window

import rsgispy
from rsgispy.classification._helpers import resolve_rng
from rsgispy.imageprocess import (
    build_valid_mask,
    ensure_file_exists,
    get_default_block_size,
    iter_dataset_windows,
    read_raster_band,
    validate_band_index,
)
from rsgispy.vector.attrs import write_vec_column_to_layer
from rsgispy.vectorprocess import VectorInput, finish_vec_output, read_vec_gdf
from rsgispy.zonalstats._helpers import (
    open_validated_zonal_inputs,
    sample_point_pixel,
)

__all__ = [
    "create_random_ref_smpls_darts",
    "create_stratified_random_ref_smpls_darts",
    "create_stratified_random_ref_smpls_lut",
    "pop_class_info_accuracy_pts",
]

_IN_MEM_BYTE_LIMIT = 128 * 1024 * 1024
DEFAULT_RARE_PXL_THRES = 100000


def _build_ref_points_gdf(
    sel_rows: np.ndarray,
    sel_cols: np.ndarray,
    sel_vals: np.ndarray,
    img_crs: Any,
    img_transform: Affine,
    img_cls_col: str,
    ref_cls_col: str,
    processed_col: str,
    class_names_lut: Optional[Dict[int, str]],
) -> gpd.GeoDataFrame:
    """Construct a point GeoDataFrame at pixel centres for sampled pixels."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    xs, ys = transform_xy(img_transform, sel_rows, sel_cols, offset="center")
    x_coords = np.asarray(xs, dtype=float)
    y_coords = np.asarray(ys, dtype=float)
    n_pts = len(x_coords)

    data_dict: Dict[str, Any] = {}
    if class_names_lut is not None:
        img_cls_names: List[str] = []
        for raw_val in sel_vals:
            int_val = int(raw_val)
            if int_val not in class_names_lut:
                raise rsgispy.RSGISPyException(
                    f"Sampled pixel value {int_val} is not present in "
                    "class_names_lut."
                )
            img_cls_names.append(str(class_names_lut[int_val]))
        data_dict[img_cls_col] = img_cls_names
        data_dict[ref_cls_col] = ["" for _ in range(n_pts)]
    else:
        int_vals = sel_vals.astype(int)
        data_dict[img_cls_col] = int_vals
        data_dict[ref_cls_col] = int_vals.copy()

    data_dict[processed_col] = np.zeros(n_pts, dtype=int)
    data_df = pd.DataFrame(data=data_dict)
    return gpd.GeoDataFrame(
        data_df,
        geometry=gpd.points_from_xy(x_coords, y_coords),
        crs=img_crs,
    )


def _lookup_class_name(
    raw_val: Optional[float],
    class_names_lut: Dict[int, str],
) -> str:
    """Resolve ``raw_val`` against ``class_names_lut``, returning ``''`` if absent."""
    if raw_val is None or math.isnan(raw_val):
        return ""
    if raw_val in class_names_lut:
        return str(class_names_lut[raw_val])  # type: ignore[index]
    if float(raw_val).is_integer():
        int_val = int(raw_val)
        if int_val in class_names_lut:
            return str(class_names_lut[int_val])
    return ""


def create_random_ref_smpls_darts(
    input_img: Union[str, os.PathLike],
    n_smpls: int,
    out_vec_file: Optional[Union[str, os.PathLike]],
    out_vec_lyr: Optional[str] = None,
    out_format: Optional[str] = None,
    img_cls_col: str = "img_cls",
    ref_cls_col: str = "cls_ref",
    processed_col: str = "Processed",
    rnd_seed: Optional[int] = None,
    img_band: int = 1,
    cls_no_data: Optional[float] = None,
    class_names_lut: Optional[Dict[int, str]] = None,
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Create a simple random sample of reference points (without replacement) at
    pixel centres across valid pixels of ``input_img``.

    :param input_img: Path to the input classified raster file.
    :param n_smpls: Number of random reference points to generate.
    :param out_vec_file: Path to the output vector file, or ``None`` to return
                         an in-memory :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Layer name for multi-layer formats such as ``GPKG``
                        (Default = ``None``).
    :param out_format: Optional output vector driver (inferred from extension
                       if ``None``).
    :param img_cls_col: Output attribute column name for image class
                        (Default = ``'img_cls'``).
    :param ref_cls_col: Output attribute column name for reference class
                        (Default = ``'cls_ref'``).
    :param processed_col: Output attribute column name tracking review status
                          (Default = ``'Processed'``).
    :param rnd_seed: Optional random seed for reproducibility (Default = ``None``).
    :param img_band: 1-based band index to sample from ``input_img``
                     (Default = ``1``).
    :param cls_no_data: Pixel value treated as no-data (reads image header
                        no-data value when ``None``).
    :param class_names_lut: Optional dictionary mapping integer pixel values to
                            string class names. When provided, ``img_cls_col``
                            stores class names and ``ref_cls_col`` is
                            initialised to empty strings.
    :param out_compress: Optional GeoParquet compression setting.
    :return: :class:`geopandas.GeoDataFrame` when ``out_vec_file`` is ``None``,
             otherwise ``None``.

    .. code:: python

        import rsgispy.classification.classrefpts

        rsgispy.classification.classrefpts.create_random_ref_smpls_darts(
            input_img="classified.tif",
            n_smpls=500,
            out_vec_file="ref_pts.gpkg",
            out_vec_lyr="ref_pts",
            cls_no_data=0,
            class_names_lut={1: "Forest", 2: "Water"},
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    if n_smpls < 1:
        raise rsgispy.RSGISPyException("n_smpls must be at least 1.")

    img_data, img_crs, img_transform, valid_mask = read_raster_band(
        input_img, img_band, cls_no_data
    )
    valid_rows, valid_cols = np.nonzero(valid_mask)
    n_valid = valid_rows.size
    if n_valid == 0:
        raise rsgispy.RSGISPyException(
            "No valid pixels found in input image."
        )
    if n_smpls > n_valid:
        raise rsgispy.RSGISPyException(
            f"Requested {n_smpls} samples, but only {n_valid} valid pixels "
            "exist in the input image."
        )

    rng = resolve_rng(rnd_seed)
    chosen = rng.choice(n_valid, size=int(n_smpls), replace=False)
    sel_rows = valid_rows[chosen]
    sel_cols = valid_cols[chosen]
    sel_vals = img_data[sel_rows, sel_cols]

    pts_gdf = _build_ref_points_gdf(
        sel_rows=sel_rows,
        sel_cols=sel_cols,
        sel_vals=sel_vals,
        img_crs=img_crs,
        img_transform=img_transform,
        img_cls_col=img_cls_col,
        ref_cls_col=ref_cls_col,
        processed_col=processed_col,
        class_names_lut=class_names_lut,
    )
    return finish_vec_output(
        pts_gdf,
        out_vec_file,
        out_vec_lyr=out_vec_lyr,
        out_format=out_format,
        overwrite=True,
        out_compress=out_compress,
    )


def _compute_stratified_allocations(
    cls_unq_vals: np.ndarray,
    cls_counts: np.ndarray,
    n_smpls: int,
    eq_n_smpls: bool,
    min_cls_smpls: Optional[int],
) -> Tuple[Dict[Any, int], Dict[Any, int], int]:
    """Compute per-class sample allocations for stratified sampling."""
    n_smpls_prop = float(n_smpls) / float(int(np.sum(cls_counts)))
    eq_target = int(math.ceil(float(n_smpls) / float(len(cls_unq_vals))))

    cls_id_count: Dict[Any, int] = {}
    cls_count_out: Dict[Any, int] = {}
    n_out_smpls = 0
    for cls_val, cls_ct_np in zip(cls_unq_vals, cls_counts):
        cls_ct = int(cls_ct_np)
        cls_id_count[cls_val] = cls_ct
        target = eq_target if eq_n_smpls else int(math.ceil(cls_ct * n_smpls_prop))
        if min_cls_smpls is not None and target < min_cls_smpls:
            target = int(min_cls_smpls)
        target = min(target, cls_ct)
        cls_count_out[cls_val] = target
        n_out_smpls += target

    return cls_id_count, cls_count_out, n_out_smpls


def _print_stratified_summary(
    cls_unq_vals: np.ndarray,
    cls_id_count: Dict[Any, int],
    cls_count_out: Dict[Any, int],
    n_out_smpls: int,
    cls_methods: Optional[Dict[Any, str]] = None,
) -> None:
    """Print summary table of per-class stratified sample counts."""
    print("The number of output samples to be generated per-class:")
    cls_info: Dict[str, List[Any]] = {
        "Class ID": cls_unq_vals.tolist(),
        "N Image Pixels": [cls_id_count[v] for v in cls_unq_vals],
        "N Ref Samples": [cls_count_out[v] for v in cls_unq_vals],
    }
    if cls_methods is not None:
        cls_info["Method"] = [cls_methods[v] for v in cls_unq_vals]
    try:
        import tabulate  # pylint: disable=import-outside-toplevel

        print(tabulate.tabulate(cls_info, headers="keys", tablefmt="grid"))
    except ImportError:
        for cls_val in cls_unq_vals:
            method_sfx = f":{cls_methods[cls_val]}" if cls_methods is not None else ""
            print(
                f"{cls_val}:{cls_id_count[cls_val]}:"
                f"{cls_count_out[cls_val]}{method_sfx}"
            )
    print(f"In total {n_out_smpls} will be generated.")


def create_stratified_random_ref_smpls_darts(
    input_img: Union[str, os.PathLike],
    n_smpls: int,
    out_vec_file: Optional[Union[str, os.PathLike]],
    out_vec_lyr: Optional[str] = None,
    eq_n_smpls: bool = False,
    min_cls_smpls: Optional[int] = None,
    out_format: Optional[str] = None,
    img_cls_col: str = "img_cls",
    ref_cls_col: str = "cls_ref",
    processed_col: str = "Processed",
    rnd_seed: Optional[int] = None,
    img_band: int = 1,
    cls_no_data: Optional[float] = None,
    class_names_lut: Optional[Dict[int, str]] = None,
    out_compress: Optional[Union[str, bool]] = None,
    verbose: bool = True,
) -> Optional[gpd.GeoDataFrame]:
    """
    Create a stratified random sample of reference points (without replacement)
    distributed across the classes in ``input_img``.

    :param input_img: Path to the input classified raster file.
    :param n_smpls: Target total number of random reference samples.
    :param out_vec_file: Path to the output vector file, or ``None`` to return
                         an in-memory :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Layer name for multi-layer formats such as ``GPKG``
                        (Default = ``None``).
    :param eq_n_smpls: If ``True``, distribute samples equally among classes;
                       if ``False`` (Default), allocate proportionally to mapped
                       class pixel counts.
    :param min_cls_smpls: Optional minimum number of samples per class (capped
                          at available pixels for that class).
    :param out_format: Optional output vector driver (inferred from extension
                       if ``None``).
    :param img_cls_col: Output attribute column name for image class
                        (Default = ``'img_cls'``).
    :param ref_cls_col: Output attribute column name for reference class
                        (Default = ``'cls_ref'``).
    :param processed_col: Output attribute column name tracking review status
                          (Default = ``'Processed'``).
    :param rnd_seed: Optional random seed for reproducibility (Default = ``None``).
    :param img_band: 1-based band index to sample from ``input_img``
                     (Default = ``1``).
    :param cls_no_data: Pixel value treated as no-data (reads image header
                        no-data value when ``None``).
    :param class_names_lut: Optional dictionary mapping integer pixel values to
                            string class names.
    :param out_compress: Optional GeoParquet compression setting.
    :param verbose: If ``True``, print per-class sampling summary table
                    (Default = ``True``).
    :return: :class:`geopandas.GeoDataFrame` when ``out_vec_file`` is ``None``,
             otherwise ``None``.

    .. code:: python

        import rsgispy.classification.classrefpts

        rsgispy.classification.classrefpts.create_stratified_random_ref_smpls_darts(
            input_img="classified.tif",
            n_smpls=500,
            out_vec_file="strat_pts.gpkg",
            out_vec_lyr="strat_pts",
            min_cls_smpls=25,
            cls_no_data=0,
            class_names_lut={1: "Forest", 2: "Water"},
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    if n_smpls < 1:
        raise rsgispy.RSGISPyException("n_smpls must be at least 1.")
    if min_cls_smpls is not None and min_cls_smpls < 0:
        raise rsgispy.RSGISPyException("min_cls_smpls must be >= 0.")

    img_data, img_crs, img_transform, valid_mask = read_raster_band(
        input_img, img_band, cls_no_data
    )
    valid_rows, valid_cols = np.nonzero(valid_mask)
    if valid_rows.size == 0:
        raise rsgispy.RSGISPyException(
            "No valid pixels found in input image."
        )

    valid_vals = img_data[valid_rows, valid_cols]
    cls_unq_vals, cls_counts = np.unique(valid_vals, return_counts=True)
    cls_id_count, cls_count_out, n_out_smpls = _compute_stratified_allocations(
        cls_unq_vals, cls_counts, n_smpls, eq_n_smpls, min_cls_smpls
    )

    if verbose:
        _print_stratified_summary(
            cls_unq_vals, cls_id_count, cls_count_out, n_out_smpls
        )

    rng = resolve_rng(rnd_seed)
    sel_row_chunks: List[np.ndarray] = []
    sel_col_chunks: List[np.ndarray] = []
    sel_val_chunks: List[np.ndarray] = []

    for cls_val in cls_unq_vals:
        n_pick = cls_count_out[cls_val]
        if n_pick <= 0:
            continue
        cls_idx = np.flatnonzero(valid_vals == cls_val)
        chosen = rng.choice(cls_idx, size=n_pick, replace=False)
        sel_row_chunks.append(valid_rows[chosen])
        sel_col_chunks.append(valid_cols[chosen])
        sel_val_chunks.append(valid_vals[chosen])

    sel_rows = np.concatenate(sel_row_chunks)
    sel_cols = np.concatenate(sel_col_chunks)
    sel_vals = np.concatenate(sel_val_chunks)

    perm = rng.permutation(sel_rows.size)
    pts_gdf = _build_ref_points_gdf(
        sel_rows=sel_rows[perm],
        sel_cols=sel_cols[perm],
        sel_vals=sel_vals[perm],
        img_crs=img_crs,
        img_transform=img_transform,
        img_cls_col=img_cls_col,
        ref_cls_col=ref_cls_col,
        processed_col=processed_col,
        class_names_lut=class_names_lut,
    )
    return finish_vec_output(
        pts_gdf,
        out_vec_file,
        out_vec_lyr=out_vec_lyr,
        out_format=out_format,
        overwrite=True,
        out_compress=out_compress,
    )


def _sample_rare_from_lut(
    rare_lut: Tuple[np.ndarray, np.ndarray, np.ndarray],
    rare_classes: List[Any],
    cls_count_out: Dict[Any, int],
    rng: np.random.Generator,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Sample rare classes without replacement from their coordinate LUT."""
    rare_rows, rare_cols, rare_vals = rare_lut
    row_chunks: List[np.ndarray] = []
    col_chunks: List[np.ndarray] = []
    val_chunks: List[np.ndarray] = []
    for cls_val in rare_classes:
        n_pick = cls_count_out[cls_val]
        cls_idx = np.flatnonzero(rare_vals == cls_val)
        chosen = rng.choice(cls_idx, size=n_pick, replace=False)
        row_chunks.append(rare_rows[chosen])
        col_chunks.append(rare_cols[chosen])
        val_chunks.append(rare_vals[chosen])
    return row_chunks, col_chunks, val_chunks


def _draw_unique_dart_coords(
    height: int,
    width: int,
    batch_n: int,
    seen_flat: set,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a batch of unique (row, col) darts excluding ``seen_flat``."""
    cand_rows = rng.integers(0, height, size=batch_n, dtype=np.int64)
    cand_cols = rng.integers(0, width, size=batch_n, dtype=np.int64)
    flat_ids = cand_rows * width + cand_cols
    _, uniq_idx = np.unique(flat_ids, return_index=True)
    if seen_flat:
        keep_mask = np.array(
            [int(flat_ids[i]) not in seen_flat for i in uniq_idx],
            dtype=bool,
        )
        uniq_idx = uniq_idx[keep_mask]
    return cand_rows[uniq_idx], cand_cols[uniq_idx]


def _accept_dart_hits(
    cand_rows: np.ndarray,
    cand_cols: np.ndarray,
    cand_vals: np.ndarray,
    cand_valid: np.ndarray,
    width: int,
    remaining: Dict[Any, int],
    seen_flat: set,
    rng: np.random.Generator,
    row_chunks: List[np.ndarray],
    col_chunks: List[np.ndarray],
    val_chunks: List[np.ndarray],
) -> None:
    """Record accepted dart hits into output chunks and update ``remaining``."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    for cls_val in list(remaining.keys()):
        need = remaining[cls_val]
        hits = np.flatnonzero(cand_valid & (cand_vals == cls_val))
        if hits.size == 0:
            continue
        if hits.size > need:
            hits = rng.choice(hits, size=need, replace=False)
        sel_r = cand_rows[hits]
        sel_c = cand_cols[hits]
        row_chunks.append(sel_r)
        col_chunks.append(sel_c)
        val_chunks.append(cand_vals[hits])
        seen_flat.update((sel_r * width + sel_c).tolist())
        remaining[cls_val] -= int(hits.size)
        if remaining[cls_val] <= 0:
            del remaining[cls_val]


def _estimate_dart_batch_size(
    height: int,
    width: int,
    remaining: Dict[Any, int],
    cls_id_count: Dict[Any, int],
    max_batch: int = 131072,
) -> int:
    """Estimate a responsive dart batch size from remaining class prevalence."""
    total_rem = sum(remaining.values())
    rem_pxls = sum(cls_id_count[c] for c in remaining)
    common_frac = max(float(rem_pxls) / float(height * width), 1e-6)
    est = int(math.ceil(4.0 * float(total_rem) / common_frac))
    return max(1024, min(est, max_batch))


def _sample_lut_and_darts_in_memory(
    src: rasterio.DatasetReader,
    img_band: int,
    cls_no_data: Optional[float],
    n_smpls: int,
    eq_n_smpls: bool,
    min_cls_smpls: Optional[int],
    rare_pxl_thres: int,
    rng: np.random.Generator,
    verbose: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run hybrid rare-LUT + common-darts stratified sampling in memory."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    img_data = src.read(img_band)
    height, width = img_data.shape
    valid_mask = build_valid_mask(img_data, cls_no_data)
    valid_vals = img_data[valid_mask]
    if valid_vals.size == 0:
        raise rsgispy.RSGISPyException(
            "No valid pixels found in input image."
        )

    cls_unq_vals, cls_counts = np.unique(valid_vals, return_counts=True)
    cls_id_count, cls_count_out, n_out_smpls = _compute_stratified_allocations(
        cls_unq_vals, cls_counts, n_smpls, eq_n_smpls, min_cls_smpls
    )
    rare_classes, common_classes, cls_methods = _partition_classes(
        cls_unq_vals, cls_id_count, cls_count_out, rare_pxl_thres
    )
    if verbose:
        _print_stratified_summary(
            cls_unq_vals, cls_id_count, cls_count_out, n_out_smpls, cls_methods
        )

    row_chunks: List[np.ndarray] = []
    col_chunks: List[np.ndarray] = []
    val_chunks: List[np.ndarray] = []

    if rare_classes:
        rare_mask = valid_mask & np.isin(img_data, np.asarray(rare_classes))
        rare_rows, rare_cols = np.nonzero(rare_mask)
        rare_vals = img_data[rare_rows, rare_cols]
        r_c, c_c, v_c = _sample_rare_from_lut(
            (rare_rows, rare_cols, rare_vals), rare_classes, cls_count_out, rng
        )
        row_chunks.extend(r_c)
        col_chunks.extend(c_c)
        val_chunks.extend(v_c)

    if common_classes:
        remaining = {v: cls_count_out[v] for v in common_classes}
        seen_flat: set = set()
        while remaining:
            batch_n = _estimate_dart_batch_size(
                height, width, remaining, cls_id_count, max_batch=262144
            )
            cand_r, cand_c = _draw_unique_dart_coords(
                height, width, batch_n, seen_flat, rng
            )
            if cand_r.size == 0:
                continue
            cand_vals = img_data[cand_r, cand_c]
            cand_valid = valid_mask[cand_r, cand_c]
            _accept_dart_hits(
                cand_r,
                cand_c,
                cand_vals,
                cand_valid,
                width,
                remaining,
                seen_flat,
                rng,
                row_chunks,
                col_chunks,
                val_chunks,
            )

    return (
        np.concatenate(row_chunks),
        np.concatenate(col_chunks),
        np.concatenate(val_chunks),
    )


def _sample_lut_and_darts_tiled(
    src: rasterio.DatasetReader,
    img_band: int,
    cls_no_data: Optional[float],
    n_smpls: int,
    eq_n_smpls: bool,
    min_cls_smpls: Optional[int],
    rare_pxl_thres: int,
    blk_sz: int,
    rng: np.random.Generator,
    use_pbar: bool,
    verbose: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run out-of-core tiled rare-LUT + windowed common-darts sampling."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    height, width = src.height, src.width
    counts_dict: Dict[Any, int] = {}
    for win in iter_dataset_windows(
        height, width, block_size=blk_sz, use_pbar=use_pbar
    ):
        blk_arr = src.read(img_band, window=win)
        blk_valid = blk_arr[build_valid_mask(blk_arr, cls_no_data)]
        if blk_valid.size == 0:
            continue
        uvals, ucts = np.unique(blk_valid, return_counts=True)
        for uval, uct in zip(uvals, ucts):
            key = uval.item()
            counts_dict[key] = counts_dict.get(key, 0) + int(uct)

    if not counts_dict:
        raise rsgispy.RSGISPyException(
            "No valid pixels found in input image."
        )

    sorted_keys = sorted(counts_dict.keys())
    cls_unq_vals = np.asarray(sorted_keys, dtype=src.dtypes[img_band - 1])
    cls_counts = np.asarray([counts_dict[k] for k in sorted_keys], dtype=np.int64)
    cls_id_count, cls_count_out, n_out_smpls = _compute_stratified_allocations(
        cls_unq_vals, cls_counts, n_smpls, eq_n_smpls, min_cls_smpls
    )
    rare_classes, common_classes, cls_methods = _partition_classes(
        cls_unq_vals, cls_id_count, cls_count_out, rare_pxl_thres
    )
    if verbose:
        _print_stratified_summary(
            cls_unq_vals, cls_id_count, cls_count_out, n_out_smpls, cls_methods
        )

    row_chunks: List[np.ndarray] = []
    col_chunks: List[np.ndarray] = []
    val_chunks: List[np.ndarray] = []

    if rare_classes:
        rare_arr = np.asarray(rare_classes)
        rare_r_parts: List[np.ndarray] = []
        rare_c_parts: List[np.ndarray] = []
        rare_v_parts: List[np.ndarray] = []
        for win in iter_dataset_windows(
            height, width, block_size=blk_sz, use_pbar=use_pbar
        ):
            blk_arr = src.read(img_band, window=win)
            rare_mask = build_valid_mask(blk_arr, cls_no_data) & np.isin(
                blk_arr, rare_arr
            )
            if np.any(rare_mask):
                l_rows, l_cols = np.nonzero(rare_mask)
                rare_r_parts.append(l_rows.astype(np.int64) + int(win.row_off))
                rare_c_parts.append(l_cols.astype(np.int64) + int(win.col_off))
                rare_v_parts.append(blk_arr[l_rows, l_cols])
        r_c, c_c, v_c = _sample_rare_from_lut(
            (
                np.concatenate(rare_r_parts),
                np.concatenate(rare_c_parts),
                np.concatenate(rare_v_parts),
            ),
            rare_classes,
            cls_count_out,
            rng,
        )
        row_chunks.extend(r_c)
        col_chunks.extend(c_c)
        val_chunks.extend(v_c)

    if common_classes:
        _sample_common_darts_tiled(
            src,
            img_band,
            cls_no_data,
            blk_sz,
            common_classes,
            cls_id_count,
            cls_count_out,
            rng,
            row_chunks,
            col_chunks,
            val_chunks,
        )

    return (
        np.concatenate(row_chunks),
        np.concatenate(col_chunks),
        np.concatenate(val_chunks),
    )


def _sample_common_darts_tiled(
    src: rasterio.DatasetReader,
    img_band: int,
    cls_no_data: Optional[float],
    blk_sz: int,
    common_classes: List[Any],
    cls_id_count: Dict[Any, int],
    cls_count_out: Dict[Any, int],
    rng: np.random.Generator,
    row_chunks: List[np.ndarray],
    col_chunks: List[np.ndarray],
    val_chunks: List[np.ndarray],
) -> None:
    """Sample common classes via batched darts grouped by raster tile."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    height, width = src.height, src.width
    n_tile_cols = int(math.ceil(float(width) / float(blk_sz)))
    remaining = {v: cls_count_out[v] for v in common_classes}
    seen_flat: set = set()

    while remaining:
        batch_n = _estimate_dart_batch_size(
            height, width, remaining, cls_id_count, max_batch=131072
        )
        cand_r, cand_c = _draw_unique_dart_coords(
            height, width, batch_n, seen_flat, rng
        )
        if cand_r.size == 0:
            continue

        tile_keys = (cand_r // blk_sz) * n_tile_cols + (cand_c // blk_sz)
        order = np.argsort(tile_keys, kind="mergesort")
        s_rows = cand_r[order]
        s_cols = cand_c[order]
        s_keys = tile_keys[order]
        split_pts = np.flatnonzero(np.diff(s_keys)) + 1
        starts = np.concatenate(([0], split_pts))
        ends = np.concatenate((split_pts, [s_keys.size]))

        s_vals = np.empty(s_rows.size, dtype=src.dtypes[img_band - 1])
        s_valid = np.zeros(s_rows.size, dtype=bool)
        for st_idx, en_idx in zip(starts, ends):
            t_key = int(s_keys[st_idx])
            tr, tc = divmod(t_key, n_tile_cols)
            row_off = tr * blk_sz
            col_off = tc * blk_sz
            win = Window(
                col_off,
                row_off,
                min(blk_sz, width - col_off),
                min(blk_sz, height - row_off),
            )
            tile_arr = src.read(img_band, window=win)
            tile_valid = build_valid_mask(tile_arr, cls_no_data)
            lr = s_rows[st_idx:en_idx] - row_off
            lc = s_cols[st_idx:en_idx] - col_off
            s_vals[st_idx:en_idx] = tile_arr[lr, lc]
            s_valid[st_idx:en_idx] = tile_valid[lr, lc]

        _accept_dart_hits(
            s_rows,
            s_cols,
            s_vals,
            s_valid,
            width,
            remaining,
            seen_flat,
            rng,
            row_chunks,
            col_chunks,
            val_chunks,
        )


def _partition_classes(
    cls_unq_vals: np.ndarray,
    cls_id_count: Dict[Any, int],
    cls_count_out: Dict[Any, int],
    rare_pxl_thres: int,
) -> Tuple[List[Any], List[Any], Dict[Any, str]]:
    """Split classes with positive targets into rare (LUT) and common (darts)."""
    rare_classes: List[Any] = []
    common_classes: List[Any] = []
    cls_methods: Dict[Any, str] = {}
    for cls_val in cls_unq_vals:
        if cls_count_out[cls_val] <= 0:
            cls_methods[cls_val] = "Skipped"
        elif cls_id_count[cls_val] <= rare_pxl_thres:
            rare_classes.append(cls_val)
            cls_methods[cls_val] = "LUT"
        else:
            common_classes.append(cls_val)
            cls_methods[cls_val] = "Darts"
    return rare_classes, common_classes, cls_methods


def create_stratified_random_ref_smpls_lut(
    input_img: Union[str, os.PathLike],
    n_smpls: int,
    out_vec_file: Optional[Union[str, os.PathLike]],
    out_vec_lyr: Optional[str] = None,
    eq_n_smpls: bool = False,
    min_cls_smpls: Optional[int] = None,
    rare_pxl_thres: int = DEFAULT_RARE_PXL_THRES,
    out_format: Optional[str] = None,
    img_cls_col: str = "img_cls",
    ref_cls_col: str = "cls_ref",
    processed_col: str = "Processed",
    rnd_seed: Optional[int] = None,
    img_band: int = 1,
    cls_no_data: Optional[float] = None,
    class_names_lut: Optional[Dict[int, str]] = None,
    out_compress: Optional[Union[str, bool]] = None,
    in_memory: Optional[bool] = None,
    block_size: Optional[int] = None,
    use_pbar: bool = True,
    verbose: bool = True,
) -> Optional[gpd.GeoDataFrame]:
    """
    Create a stratified random sample of reference points (without replacement)
    using a hybrid rare-class coordinate look-up table (LUT) and common-class
    random dart strategy.

    Classes whose valid pixel count in ``input_img`` is at or below
    ``rare_pxl_thres`` are indexed into a pixel coordinate LUT so all rare
    classes are sampled deterministically in a single scan without wasting
    random darts. Abundant classes above ``rare_pxl_thres`` are sampled using
    batched random darts without replacement, avoiding large coordinate LUT
    allocations for dominant classes.

    :param input_img: Path to the input classified raster file.
    :param n_smpls: Target total number of random reference samples.
    :param out_vec_file: Path to the output vector file, or ``None`` to return
                         an in-memory :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Layer name for multi-layer formats such as ``GPKG``
                        (Default = ``None``).
    :param eq_n_smpls: If ``True``, distribute samples equally among classes;
                       if ``False`` (Default), allocate proportionally to mapped
                       class pixel counts.
    :param min_cls_smpls: Optional minimum number of samples per class (capped
                          at available pixels for that class).
    :param rare_pxl_thres: Pixel count threshold at or below which a class is
                           treated as rare and sampled via a pixel coordinate
                           LUT rather than random darts
                           (Default = ``DEFAULT_RARE_PXL_THRES``, ``100000``).
    :param out_format: Optional output vector driver (inferred from extension
                       if ``None``).
    :param img_cls_col: Output attribute column name for image class
                        (Default = ``'img_cls'``).
    :param ref_cls_col: Output attribute column name for reference class
                        (Default = ``'cls_ref'``).
    :param processed_col: Output attribute column name tracking review status
                          (Default = ``'Processed'``).
    :param rnd_seed: Optional random seed for reproducibility (Default = ``None``).
    :param img_band: 1-based band index to sample from ``input_img``
                     (Default = ``1``).
    :param cls_no_data: Pixel value treated as no-data (reads image header
                        no-data value when ``None``).
    :param class_names_lut: Optional dictionary mapping integer pixel values to
                            string class names.
    :param out_compress: Optional GeoParquet compression setting.
    :param in_memory: Processing mode: ``True`` loads the band into RAM,
                      ``False`` streams blocks out-of-core, and ``None``
                      (Default) auto-selects ``True`` when the uncompressed band
                      size is at most 128 MiB.
    :param block_size: Tile edge length in pixels for out-of-core processing
                       when ``in_memory=False`` (Default = ``None`` uses library
                       default).
    :param use_pbar: If ``True``, display a tqdm progress bar during tiled scans
                     (Default = ``True``).
    :param verbose: If ``True``, print per-class sampling summary table
                    (Default = ``True``).
    :return: :class:`geopandas.GeoDataFrame` when ``out_vec_file`` is ``None``,
             otherwise ``None``.

    .. code:: python

        import rsgispy.classification.classrefpts

        rsgispy.classification.classrefpts.create_stratified_random_ref_smpls_lut(
            input_img="classified.tif",
            n_smpls=500,
            out_vec_file="strat_pts_lut.gpkg",
            out_vec_lyr="strat_pts",
            min_cls_smpls=25,
            rare_pxl_thres=100000,
            cls_no_data=0,
            class_names_lut={1: "Forest", 2: "Water", 3: "RareWetland"},
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    if n_smpls < 1:
        raise rsgispy.RSGISPyException("n_smpls must be at least 1.")
    if min_cls_smpls is not None and min_cls_smpls < 0:
        raise rsgispy.RSGISPyException("min_cls_smpls must be >= 0.")
    if rare_pxl_thres < 0:
        raise rsgispy.RSGISPyException("rare_pxl_thres must be >= 0.")
    if block_size is not None and block_size < 1:
        raise rsgispy.RSGISPyException("block_size must be at least 1.")

    in_path = os.fspath(input_img)
    ensure_file_exists(in_path, label="Input image")

    rng = resolve_rng(rnd_seed)
    try:
        with rasterio.open(in_path, "r") as src:
            validate_band_index(img_band, src.count, in_path)
            eff_nodata = (
                src.nodatavals[img_band - 1]
                if cls_no_data is None
                else cls_no_data
            )
            img_crs = src.crs
            img_transform = src.transform
            itemsize = np.dtype(src.dtypes[img_band - 1]).itemsize
            use_in_mem = (
                src.height * src.width * itemsize <= _IN_MEM_BYTE_LIMIT
                if in_memory is None
                else bool(in_memory)
            )
            if use_in_mem:
                sel_rows, sel_cols, sel_vals = _sample_lut_and_darts_in_memory(
                    src,
                    img_band,
                    eff_nodata,
                    n_smpls,
                    eq_n_smpls,
                    min_cls_smpls,
                    rare_pxl_thres,
                    rng,
                    verbose,
                )
            else:
                blk_sz = (
                    get_default_block_size()
                    if block_size is None
                    else int(block_size)
                )
                sel_rows, sel_cols, sel_vals = _sample_lut_and_darts_tiled(
                    src,
                    img_band,
                    eff_nodata,
                    n_smpls,
                    eq_n_smpls,
                    min_cls_smpls,
                    rare_pxl_thres,
                    blk_sz,
                    rng,
                    use_pbar,
                    verbose,
                )
    except rsgispy.RSGISPyException:
        raise
    except Exception as err:
        raise rsgispy.RSGISPyException(
            f"Could not sample reference points from '{in_path}': {err}"
        ) from err

    perm = rng.permutation(sel_rows.size)
    pts_gdf = _build_ref_points_gdf(
        sel_rows=sel_rows[perm],
        sel_cols=sel_cols[perm],
        sel_vals=sel_vals[perm],
        img_crs=img_crs,
        img_transform=img_transform,
        img_cls_col=img_cls_col,
        ref_cls_col=ref_cls_col,
        processed_col=processed_col,
        class_names_lut=class_names_lut,
    )
    return finish_vec_output(
        pts_gdf,
        out_vec_file,
        out_vec_lyr=out_vec_lyr,
        out_format=out_format,
        overwrite=True,
        out_compress=out_compress,
    )


def pop_class_info_accuracy_pts(
    input_cls_img: Union[str, os.PathLike],
    acc_pts_vec_file: VectorInput,
    acc_pts_vec_lyr: Optional[str],
    out_vec_file: Optional[Union[str, os.PathLike]],
    out_vec_lyr: Optional[str],
    class_names_lut: Dict[int, str],
    img_cls_col: str = "img_cls",
    ref_cls_col: Optional[str] = None,
    processed_col: Optional[str] = None,
    out_format: Optional[str] = None,
    cls_img_band: int = 1,
    reproj_vec: bool = False,
    vec_def_epsg: Optional[int] = None,
    out_compress: Optional[Union[str, bool]] = None,
    use_pbar: bool = True,
) -> Optional[gpd.GeoDataFrame]:
    """
    Populate an existing point vector layer with mapped class names sampled
    from ``input_cls_img`` at each point location.

    :param input_cls_img: Path to the input classification image file.
    :param acc_pts_vec_file: Input reference points vector file path or
                             :class:`geopandas.GeoDataFrame`.
    :param acc_pts_vec_lyr: Input vector layer name (or ``None``).
    :param out_vec_file: Output vector file path (may equal ``acc_pts_vec_file``
                         to overwrite), or ``None`` to return a
                         :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name (or ``None``).
    :param class_names_lut: Dictionary mapping integer pixel class IDs to class
                            name strings.
    :param img_cls_col: Column name populated with mapped image class names
                        (Default = ``'img_cls'``).
    :param ref_cls_col: Optional column name created/reset to empty strings for
                        reference class labels (Default = ``None``).
    :param processed_col: Optional column name created/reset to ``0`` to track
                          review progress (Default = ``None``).
    :param out_format: Optional output vector driver (inferred from extension
                       if ``None``).
    :param cls_img_band: 1-based band index within ``input_cls_img``
                         (Default = ``1``).
    :param reproj_vec: Reproject point coordinates on the fly when ``True`` if
                       vector and raster CRSs differ (Default = ``False``).
    :param vec_def_epsg: Optional EPSG code overriding vector CRS if undefined.
    :param out_compress: Optional GeoParquet compression setting.
    :param use_pbar: If ``True``, display a tqdm progress bar while sampling
                     points (Default = ``True``).
    :return: Updated :class:`geopandas.GeoDataFrame` when ``out_vec_file`` is
             ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.classification.classrefpts

        rsgispy.classification.classrefpts.pop_class_info_accuracy_pts(
            input_cls_img="classified.tif",
            acc_pts_vec_file="pts.gpkg",
            acc_pts_vec_lyr="pts",
            out_vec_file="pts_pop.gpkg",
            out_vec_lyr="pts",
            class_names_lut={1: "Forest", 2: "Water"},
            ref_cls_col="cls_ref",
            processed_col="Processed",
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    cls_img_path = os.fspath(input_cls_img)
    ensure_file_exists(cls_img_path, label="Input classification image")

    pts_gdf = read_vec_gdf(acc_pts_vec_file, acc_pts_vec_lyr)
    geom_types = set(pts_gdf.geom_type.dropna().unique())
    if not geom_types or not geom_types.issubset({"Point"}):
        raise rsgispy.RSGISPyException(
            f"The layer geometry type must be Point: Got '{sorted(geom_types)}'."
        )

    img_cls_names: List[str] = []
    tqdm_cls = rsgispy.get_tqdm()
    with open_validated_zonal_inputs(
        pts_gdf,
        cls_img_path,
        img_band=cls_img_band,
        vec_def_epsg=vec_def_epsg,
        reproj_vec=reproj_vec,
    ) as (src, _img_nodata, geom_series):
        for geom in tqdm_cls(
            geom_series, total=len(pts_gdf), disable=not use_pbar
        ):
            if geom is None or geom.is_empty:
                img_cls_names.append("")
                continue
            sampled = sample_point_pixel(
                src, cls_img_band, geom.x, geom.y, clamp=False
            )
            img_cls_names.append(_lookup_class_name(sampled, class_names_lut))

    write_vec_column_to_layer(pts_gdf, img_cls_col, str, img_cls_names)
    if ref_cls_col is not None:
        write_vec_column_to_layer(
            pts_gdf, ref_cls_col, str, ["" for _ in range(len(pts_gdf))]
        )
    if processed_col is not None:
        write_vec_column_to_layer(
            pts_gdf, processed_col, int, [0 for _ in range(len(pts_gdf))]
        )

    return finish_vec_output(
        pts_gdf,
        out_vec_file,
        out_vec_lyr=out_vec_lyr,
        out_format=out_format,
        overwrite=True,
        out_compress=out_compress,
    )
