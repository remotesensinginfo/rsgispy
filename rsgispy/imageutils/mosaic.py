"""
The imageutils.mosaic module provides image warping, mosaicking, image inclusion,
band combining, and binary mask bit-packing functions using ``rasterio``.
"""

from contextlib import ExitStack
import math
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.windows import Window

import rsgispy
from rsgispy.imagecalc.stats import get_unique_values
from rsgispy.imageprocess import (
    IntersectionGrid,
    cast_result_block,
    compute_spatial_intersection,
    get_default_block_size,
    open_multi_image_datasets,
    open_output_dataset,
    validate_crs_and_res,
)
from rsgispy.imageutils.colourtable import set_img_thematic
from rsgispy.tools.utils import create_var_list, write_dict_to_json

__all__ = [
    "create_img_mosaic",
    "include_imgs",
    "include_imgs_with_overlap",
    "combine_imgs_to_band",
    "combine_binary_masks",
]


def create_img_mosaic(
    input_imgs: Sequence[str],
    output_img: str,
    background_val: float,
    skip_val: float,
    skip_band: int,
    overlap_behaviour: int,
    gdalformat: str,
    datatype: int,
    options: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Mosaic a list of images sharing a CRS, pixel resolution, and band count.

    :param input_imgs: Sequence of input image paths.
    :param output_img: Path to the output mosaic image file.
    :param background_val: Initial background fill value for areas without data.
    :param skip_val: Pixel value in ``skip_band`` treated as no-data / skipped.
    :param skip_band: 1-based band index inspected for ``skip_val`` and overlap rules.
    :param overlap_behaviour: Overlap rule: ``0`` = overwrite, ``1`` = minimum of
                              ``skip_band``, ``2`` = maximum of ``skip_band``.
    :param gdalformat: Output GDAL format name (e.g., ``'KEA'``, ``'GTIFF'``).
    :param datatype: ``rsgispy.TYPE_*`` constant for the output image.
    :param options: Optional GDAL creation options mapping.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals,too-many-branches,too-many-statements
    if overlap_behaviour not in (0, 1, 2):
        raise rsgispy.RSGISPyException(
            f"overlap_behaviour must be 0, 1, or 2, got {overlap_behaviour}."
        )

    out_rio_dtype = rsgispy.get_rasterio_datatype(datatype)
    out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(datatype))
    creation_opts = dict(options) if isinstance(options, Mapping) else None
    skip_idx = int(skip_band) - 1
    skip_is_nan = math.isnan(float(skip_val))
    bg_is_nan = math.isnan(float(background_val))
    block_size = get_default_block_size()

    with ExitStack() as stack:
        datasets, paths = open_multi_image_datasets(stack, input_imgs)
        first_ds = datasets[paths[0]]
        validate_crs_and_res(datasets, first_ds)

        n_bands = int(first_ds.count)
        if skip_idx < 0 or skip_idx >= n_bands:
            raise rsgispy.RSGISPyException(
                f"skip_band {skip_band} is out of range for images with {n_bands} band(s)."
            )
        for path in paths[1:]:
            if datasets[path].count != n_bands:
                raise rsgispy.RSGISPyException(
                    f"Image '{path}' has {datasets[path].count} bands, expected {n_bands}."
                )

        min_x = min(float(datasets[p].bounds.left) for p in paths)
        max_x = max(float(datasets[p].bounds.right) for p in paths)
        min_y = min(float(datasets[p].bounds.bottom) for p in paths)
        max_y = max(float(datasets[p].bounds.top) for p in paths)

        x_res = abs(float(first_ds.transform.a))
        y_res = abs(float(first_ds.transform.e))
        out_w = max(1, int(round((max_x - min_x) / x_res)))
        out_h = max(1, int(round((max_y - min_y) / y_res)))
        out_transform = Affine(x_res, 0.0, min_x, 0.0, -y_res, max_y)

        grid = IntersectionGrid(
            width=out_w,
            height=out_h,
            transform=out_transform,
            crs=first_ds.crs,
            offsets={},
        )

        with ExitStack() as init_stack:
            dst_init = open_output_dataset(
                stack=init_stack,
                output_img=output_img,
                gdalformat=gdalformat,
                out_rio_dtype=out_rio_dtype,
                grid=grid,
                count=n_bands,
                output_exists=False,
                creation_opts=creation_opts,
            )
            dst_init.nodata = float(background_val)
            for bidx in range(1, n_bands + 1):
                desc = first_ds.descriptions[bidx - 1]
                if desc:
                    dst_init.set_band_description(bidx, desc)

            for r0 in range(0, out_h, block_size):
                win_h = min(block_size, out_h - r0)
                for c0 in range(0, out_w, block_size):
                    win_w = min(block_size, out_w - c0)
                    bg_block = np.full(
                        (n_bands, win_h, win_w),
                        background_val,
                        dtype=out_np_dtype,
                    )
                    dst_init.write(bg_block, window=Window(c0, r0, win_w, win_h))

        dst = stack.enter_context(rasterio.open(output_img, "r+"))
        for path in paths:
            ds = datasets[path]
            ds_col = int(round((float(ds.bounds.left) - min_x) / x_res))
            ds_row = int(round((max_y - float(ds.bounds.top)) / y_res))

            for r0 in range(0, ds.height, block_size):
                win_h = min(block_size, ds.height - r0)
                for c0 in range(0, ds.width, block_size):
                    win_w = min(block_size, ds.width - c0)
                    in_win = Window(c0, r0, win_w, win_h)
                    dst_win = Window(ds_col + c0, ds_row + r0, win_w, win_h)

                    in_block = ds.read(window=in_win, out_dtype=np.float64)
                    in_skip = in_block[skip_idx]
                    in_valid = (
                        ~np.isnan(in_skip)
                        if skip_is_nan
                        else (in_skip != float(skip_val))
                    )

                    out_block = dst.read(window=dst_win)
                    if overlap_behaviour == 0:
                        take_in = in_valid
                    else:
                        out_skip = out_block[skip_idx].astype(np.float64)
                        out_is_bg = (
                            np.isnan(out_skip)
                            if bg_is_nan
                            else (out_skip == float(background_val))
                        )
                        if overlap_behaviour == 1:
                            take_in = in_valid & (
                                out_is_bg | (in_skip < out_skip)
                            )
                        else:
                            take_in = in_valid & (
                                out_is_bg | (in_skip > out_skip)
                            )

                    if np.any(take_in):
                        cast_in = cast_result_block(
                            in_block, win_h, win_w, out_np_dtype
                        )
                        out_block[:, take_in] = cast_in[:, take_in]
                        dst.write(out_block, window=dst_win)


def include_imgs(
    input_img: str,
    include_imgs: Sequence[str],
    input_bands: Optional[Sequence[int]] = None,
    skip_val: Optional[float] = None,
) -> None:
    """
    Insert pixel values from ``include_imgs`` into an existing ``input_img`` in place.

    :param input_img: Path to the existing base image updated in place.
    :param include_imgs: Sequence of image paths to insert into ``input_img``.
    :param input_bands: Optional 1-based band subset from each included image
                        (mapped sequentially onto bands ``1..N`` of ``input_img``).
    :param skip_val: Optional pixel value to skip (preserving existing base values).
    """
    # pylint: disable=redefined-outer-name,too-many-locals,too-many-branches
    if not os.path.exists(input_img):
        raise rsgispy.RSGISPyException(
            f"Base image file does not exist: '{input_img}'."
        )
    if not include_imgs:
        raise rsgispy.RSGISPyException("include_imgs must not be empty.")

    block_size = get_default_block_size()
    with ExitStack() as stack:
        base_ds = stack.enter_context(rasterio.open(input_img, "r+"))
        base_dtype = np.dtype(base_ds.dtypes[0])
        x_res = abs(float(base_ds.transform.a))
        y_res = abs(float(base_ds.transform.e))

        inc_datasets, inc_paths = open_multi_image_datasets(stack, include_imgs)
        validate_crs_and_res(inc_datasets, base_ds)

        if input_bands is not None:
            src_bands = [int(b) for b in input_bands]
            if not src_bands or len(src_bands) > base_ds.count:
                raise rsgispy.RSGISPyException(
                    f"input_bands length ({len(src_bands)}) must be between 1 "
                    f"and {base_ds.count}."
                )
            dst_bands = list(range(1, len(src_bands) + 1))
        else:
            src_bands = None
            dst_bands = list(range(1, base_ds.count + 1))

        skip_is_nan = (
            math.isnan(float(skip_val)) if skip_val is not None else False
        )

        for inc_path in inc_paths:
            inc_ds = inc_datasets[inc_path]
            if src_bands is None:
                if inc_ds.count != base_ds.count:
                    raise rsgispy.RSGISPyException(
                        f"Included image '{inc_path}' has {inc_ds.count} bands, "
                        f"expected {base_ds.count}."
                    )
                cur_src_bands = dst_bands
            else:
                for b_idx in src_bands:
                    if b_idx < 1 or b_idx > inc_ds.count:
                        raise rsgispy.RSGISPyException(
                            f"Band {b_idx} is out of range for '{inc_path}' ({inc_ds.count} bands)."
                        )
                cur_src_bands = src_bands

            col_off = int(
                round((float(inc_ds.bounds.left) - float(base_ds.bounds.left)) / x_res)
            )
            row_off = int(
                round((float(base_ds.bounds.top) - float(inc_ds.bounds.top)) / y_res)
            )
            if (
                col_off < 0
                or row_off < 0
                or col_off + inc_ds.width > base_ds.width
                or row_off + inc_ds.height > base_ds.height
            ):
                raise rsgispy.RSGISPyException(
                    f"Included image '{inc_path}' falls outside the extent of '{input_img}'."
                )

            for r0 in range(0, inc_ds.height, block_size):
                win_h = min(block_size, inc_ds.height - r0)
                for c0 in range(0, inc_ds.width, block_size):
                    win_w = min(block_size, inc_ds.width - c0)
                    in_win = Window(c0, r0, win_w, win_h)
                    dst_win = Window(col_off + c0, row_off + r0, win_w, win_h)

                    inc_arr = inc_ds.read(
                        cur_src_bands, window=in_win, out_dtype=np.float64
                    )
                    cast_inc = cast_result_block(
                        inc_arr, win_h, win_w, base_dtype
                    )

                    if skip_val is not None:
                        base_arr = base_ds.read(dst_bands, window=dst_win)
                        valid = (
                            ~np.isnan(inc_arr)
                            if skip_is_nan
                            else (inc_arr != float(skip_val))
                        )
                        base_arr[valid] = cast_inc[valid]
                        base_ds.write(base_arr, dst_bands, window=dst_win)
                    else:
                        base_ds.write(cast_inc, dst_bands, window=dst_win)


def include_imgs_with_overlap(
    input_img: str,
    include_imgs: Sequence[str],
    overlap: int,
) -> None:
    """
    Insert cropped cores (removing ``overlap`` border pixels from all four sides)
    from ``include_imgs`` into ``input_img`` in place.

    :param input_img: Path to the existing base image updated in place.
    :param include_imgs: Sequence of overlapping tile/image paths.
    :param overlap: Number of border pixels to crop from each side.
    """
    # pylint: disable=redefined-outer-name,too-many-locals
    if not os.path.exists(input_img):
        raise rsgispy.RSGISPyException(
            f"Base image file does not exist: '{input_img}'."
        )
    if not include_imgs:
        raise rsgispy.RSGISPyException("include_imgs must not be empty.")
    ovl = int(overlap)
    if ovl < 0:
        raise rsgispy.RSGISPyException(
            f"overlap must be non-negative, got {overlap}."
        )

    block_size = get_default_block_size()
    with ExitStack() as stack:
        base_ds = stack.enter_context(rasterio.open(input_img, "r+"))
        base_dtype = np.dtype(base_ds.dtypes[0])
        x_res = abs(float(base_ds.transform.a))
        y_res = abs(float(base_ds.transform.e))

        inc_datasets, inc_paths = open_multi_image_datasets(stack, include_imgs)
        validate_crs_and_res(inc_datasets, base_ds)

        for inc_path in inc_paths:
            inc_ds = inc_datasets[inc_path]
            if inc_ds.count != base_ds.count:
                raise rsgispy.RSGISPyException(
                    f"Included image '{inc_path}' has {inc_ds.count} bands, "
                    f"expected {base_ds.count}."
                )

            crop_w = int(inc_ds.width) - 2 * ovl
            crop_h = int(inc_ds.height) - 2 * ovl
            if crop_w <= 0 or crop_h <= 0:
                raise rsgispy.RSGISPyException(
                    f"Overlap ({ovl}) is too large for image '{inc_path}' "
                    f"with dimensions ({inc_ds.width}x{inc_ds.height})."
                )

            crop_left = float(inc_ds.bounds.left) + ovl * x_res
            crop_top = float(inc_ds.bounds.top) - ovl * y_res
            dst_col = int(
                round((crop_left - float(base_ds.bounds.left)) / x_res)
            )
            dst_row = int(
                round((float(base_ds.bounds.top) - crop_top) / y_res)
            )

            if (
                dst_col < 0
                or dst_row < 0
                or dst_col + crop_w > base_ds.width
                or dst_row + crop_h > base_ds.height
            ):
                raise rsgispy.RSGISPyException(
                    f"Cropped image '{inc_path}' falls outside the extent of '{input_img}'."
                )

            for r0 in range(0, crop_h, block_size):
                win_h = min(block_size, crop_h - r0)
                for c0 in range(0, crop_w, block_size):
                    win_w = min(block_size, crop_w - c0)
                    src_win = Window(ovl + c0, ovl + r0, win_w, win_h)
                    dst_win = Window(dst_col + c0, dst_row + r0, win_w, win_h)
                    data = inc_ds.read(window=src_win)
                    out_arr = cast_result_block(data, win_h, win_w, base_dtype)
                    base_ds.write(out_arr, window=dst_win)


def combine_imgs_to_band(
    input_imgs: Sequence[str],
    output_img: str,
    gdalformat: str,
    datatype: int,
    no_data_val: float = 0.0,
    options: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Combine all bands across ``input_imgs`` into a single-band output image by
    overlaying non-no-data pixel values in order (later non-no-data values
    overwrite earlier ones).

    :param input_imgs: Sequence of input image paths.
    :param output_img: Path to the single-band output image file.
    :param gdalformat: Output GDAL format name (e.g., ``'KEA'``, ``'GTIFF'``).
    :param datatype: ``rsgispy.TYPE_*`` constant for the output image.
    :param no_data_val: No-data value ignored during combination (Default = ``0.0``).
    :param options: Optional GDAL creation options mapping.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    out_rio_dtype = rsgispy.get_rasterio_datatype(datatype)
    out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(datatype))
    creation_opts = dict(options) if isinstance(options, Mapping) else None
    nd_is_nan = math.isnan(float(no_data_val))
    block_size = get_default_block_size()

    with ExitStack() as stack:
        datasets, paths = open_multi_image_datasets(stack, input_imgs)
        grid = compute_spatial_intersection(datasets)

        dst = open_output_dataset(
            stack=stack,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype=out_rio_dtype,
            grid=grid,
            count=1,
            output_exists=False,
            creation_opts=creation_opts,
        )
        dst.nodata = float(no_data_val)

        for r0 in range(0, grid.height, block_size):
            win_h = min(block_size, grid.height - r0)
            for c0 in range(0, grid.width, block_size):
                win_w = min(block_size, grid.width - c0)
                out_win = Window(c0, r0, win_w, win_h)
                out_arr = np.full(
                    (win_h, win_w), float(no_data_val), dtype=np.float64
                )

                for path in paths:
                    ds = datasets[path]
                    base_col, base_row = grid.offsets[path]
                    in_win = Window(
                        base_col + c0, base_row + r0, win_w, win_h
                    )
                    for b_idx in range(1, ds.count + 1):
                        arr = ds.read(
                            b_idx, window=in_win, out_dtype=np.float64
                        )
                        valid = (
                            ~np.isnan(arr)
                            if nd_is_nan
                            else (arr != float(no_data_val))
                        )
                        out_arr[valid] = arr[valid]

                dst.write(
                    cast_result_block(out_arr, win_h, win_w, out_np_dtype),
                    1,
                    window=out_win,
                )


def combine_binary_masks(
    msk_imgs_dict: Mapping[str, str],
    output_img: str,
    out_lut_file: str,
    gdalformat: str = "GTIFF",
) -> None:
    """
    Combine up to 8 binary masks (where value ``1`` indicates presence) into a
    single 8-bit bit-packed raster image and write a JSON lookup table mapping
    each present packed value to the binary state of each mask name.

    :param msk_imgs_dict: Mapping of mask name -> binary mask image file path
                          (1 to 8 entries).
    :param output_img: Path to the 8-bit unsigned bit-packed output image.
    :param out_lut_file: Path to the output JSON lookup table file.
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    """
    # pylint: disable=too-many-locals
    if not msk_imgs_dict:
        raise rsgispy.RSGISPyException(
            "msk_imgs_dict must contain between 1 and 8 mask entries."
        )
    if len(msk_imgs_dict) > 8:
        raise rsgispy.RSGISPyException(
            f"combine_binary_masks supports at most 8 binary masks, got {len(msk_imgs_dict)}."
        )

    keys: List[str] = list(msk_imgs_dict.keys())
    img_paths: List[str] = [os.fspath(msk_imgs_dict[k]) for k in keys]
    block_size = get_default_block_size()

    with ExitStack() as stack:
        datasets, resolved_paths = open_multi_image_datasets(stack, img_paths)
        grid = compute_spatial_intersection(datasets)

        dst = open_output_dataset(
            stack=stack,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype="uint8",
            grid=grid,
            count=1,
            output_exists=False,
        )

        for r0 in range(0, grid.height, block_size):
            win_h = min(block_size, grid.height - r0)
            for c0 in range(0, grid.width, block_size):
                win_w = min(block_size, grid.width - c0)
                out_win = Window(c0, r0, win_w, win_h)
                bits = np.zeros((win_h, win_w, len(keys)), dtype=np.uint8)
                for idx, path in enumerate(resolved_paths):
                    ds = datasets[path]
                    base_col, base_row = grid.offsets[path]
                    in_win = Window(
                        base_col + c0, base_row + r0, win_w, win_h
                    )
                    arr = ds.read(1, window=in_win)
                    bits[:, :, idx] = (arr == 1).astype(np.uint8)

                packed = np.packbits(bits, axis=-1)[:, :, 0]
                dst.write(packed, 1, window=out_win)

    set_img_thematic(output_img)

    uniq_vals = set(int(v) for v in get_unique_values(output_img, img_band=1))
    lut_vars = {key: [0, 1] for key in keys}
    lut_combs = create_var_list(lut_vars)

    out_lut: Dict[str, Dict[str, int]] = {}
    for comb in lut_combs:
        comb_bits = np.array([int(comb[key]) for key in keys], dtype=np.uint8)
        packed_val = int(np.packbits(comb_bits)[0])
        if packed_val in uniq_vals:
            out_lut[str(packed_val)] = {key: int(comb[key]) for key in keys}

    write_dict_to_json(out_lut, out_lut_file)
