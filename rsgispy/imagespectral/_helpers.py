"""
Internal shared helpers for :mod:`rsgispy.imagespectral` submodules.
"""

from contextlib import ExitStack
import math
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
from rasterio.io import DatasetReader

import rsgispy
from rsgispy.imageprocess import (
    compute_spatial_intersection,
    ensure_file_exists,
    open_multi_image_datasets,
    validate_band_index,
)
from rsgispy.imageutils.metadata import get_img_no_data_value


def resolve_selected_bands(
    src: DatasetReader,
    bands: Optional[Sequence[int]],
    img_path: str,
) -> List[int]:
    """Return validated 1-based band indices for ``src``."""
    if bands is None:
        return list(range(1, int(src.count) + 1))
    band_list = [int(b) for b in bands]
    if not band_list:
        raise rsgispy.RSGISPyException(
            "bands must contain at least one 1-based band index."
        )
    for b_idx in band_list:
        validate_band_index(b_idx, src.count, img_path)
    return band_list


def resolve_eff_nodata(
    input_img: str,
    no_data_val: Optional[float],
    out_no_data_val: Optional[float],
    out_np_dtype: np.dtype,
) -> Tuple[Optional[float], float]:
    """Resolve input and output nodata values consistently."""
    eff_in_nd = (
        no_data_val if no_data_val is not None else get_img_no_data_value(input_img)
    )
    if out_no_data_val is not None:
        eff_out_nd = float(out_no_data_val)
    elif eff_in_nd is not None and math.isfinite(float(eff_in_nd)):
        eff_out_nd = float(eff_in_nd)
    elif np.issubdtype(out_np_dtype, np.floating):
        eff_out_nd = -9999.0
    else:
        eff_out_nd = 0.0
    return eff_in_nd, eff_out_nd


def valid_cube_mask(
    blk: np.ndarray,
    eff_in_nd: Optional[float],
    msk_blk: Optional[np.ndarray] = None,
    valid_msk_val: int = 1,
) -> np.ndarray:
    """Compute a 2D boolean mask of finite, non-nodata, unmasked pixels."""
    valid = np.all(np.isfinite(blk), axis=0)
    if eff_in_nd is not None and math.isfinite(float(eff_in_nd)):
        valid &= np.all(blk != float(eff_in_nd), axis=0)
    if msk_blk is not None:
        valid &= msk_blk == valid_msk_val
    return valid


def open_spectral_inputs(
    stack: ExitStack,
    input_img: str,
    valid_msk_img: Optional[str],
) -> Tuple[
    DatasetReader,
    Optional[DatasetReader],
    Any,
    Tuple[int, int],
    Optional[Tuple[int, int]],
]:
    """Open input image and optional valid mask aligned on a shared grid."""
    ensure_file_exists(input_img)
    in_list = [input_img]
    if valid_msk_img is not None:
        ensure_file_exists(valid_msk_img, label="Valid mask image")
        in_list.append(valid_msk_img)

    datasets, paths = open_multi_image_datasets(stack, in_list)
    img_ds = datasets[paths[0]]
    msk_ds = datasets[paths[1]] if valid_msk_img is not None else None
    if msk_ds is not None and msk_ds.count < 1:
        raise rsgispy.RSGISPyException("Valid mask image must have at least 1 band.")
    grid = compute_spatial_intersection(datasets)
    img_off = grid.offsets[paths[0]]
    msk_off = grid.offsets[paths[1]] if valid_msk_img is not None else None
    return img_ds, msk_ds, grid, img_off, msk_off
