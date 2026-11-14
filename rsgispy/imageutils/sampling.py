"""
The imageutils.sampling module provides random and mask-based pixel sampling
and extraction functions using ``rasterio`` and ``numpy``.
"""

from contextlib import ExitStack
import math
from typing import List, Optional, Sequence, Union

import numpy as np
import rasterio
from rasterio.windows import Window

import rsgispy
from rsgispy.imageprocess import (
    compute_spatial_intersection,
    create_output_from_dataset,
    ensure_file_exists,
    iter_dataset_windows,
    iter_single_image_band_blocks,
    open_multi_image_datasets,
    validate_band_index,
)
from rsgispy.imageutils.arrays import set_img_pxl_values
from rsgispy.imageutils.blank import create_copy_img
from rsgispy.imageutils.metadata import get_img_no_data_value, get_img_size

__all__ = [
    "perform_random_pxl_sample_in_mask",
    "extract_img_pxl_sample",
    "extract_img_pxl_vals_in_msk",
    "assign_random_pxls",
    "generate_random_pxl_vals_img",
]


def perform_random_pxl_sample_in_mask(
    input_img: str,
    output_img: str,
    gdalformat: str,
    mask_vals: Union[int, Sequence[int]],
    n_samples: int,
    rnd_seed: Optional[int] = None,
) -> None:
    """
    Randomly sample ``n_samples`` pixels for each value in ``mask_vals`` from
    band 1 of ``input_img`` and write them to ``output_img`` (non-sampled pixels
    are set to ``0``).

    :param input_img: Path to the input integer mask image file.
    :param output_img: Path to the output sampled mask image file.
    :param gdalformat: Output GDAL format name (e.g., ``'KEA'``, ``'GTIFF'``).
    :param mask_vals: Single mask value or sequence of mask values to sample.
    :param n_samples: Number of pixels to randomly sample per mask value.
    :param rnd_seed: Optional random seed for reproducibility (Default = ``None``).

    .. code:: python

        import rsgispy.imageutils.sampling
        rsgispy.imageutils.sampling.perform_random_pxl_sample_in_mask(
            "mask_img.tif", "sampled_mask.tif", "GTIFF", [1, 2], 100, rnd_seed=42
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    ensure_file_exists(input_img)
    if n_samples < 1:
        raise rsgispy.RSGISPyException("n_samples must be at least 1.")

    if isinstance(mask_vals, (int, np.integer)):
        vals_list = [int(mask_vals)]
    else:
        vals_list = [int(v) for v in mask_vals]
    if not vals_list:
        raise rsgispy.RSGISPyException(
            "mask_vals must contain at least one integer value."
        )

    rng = np.random.default_rng(rnd_seed)
    with ExitStack() as stack:
        src = stack.enter_context(rasterio.open(input_img, "r"))
        msk = src.read(1)
        out = np.zeros(msk.shape, dtype=msk.dtype)

        for mval in vals_list:
            rows, cols = np.where(msk == mval)
            if rows.size == 0:
                continue
            count = min(int(n_samples), int(rows.size))
            sel = rng.choice(rows.size, size=count, replace=False)
            out[rows[sel], cols[sel]] = mval

        dst = create_output_from_dataset(
            stack=stack,
            src=src,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype=str(msk.dtype),
            count=1,
        )
        dst.write(out, 1)


def extract_img_pxl_sample(
    input_img: str,
    pxl_n_sample: int,
    no_data_val: Optional[float] = None,
    use_pbar: bool = True,
) -> np.ndarray:
    """
    Extract every ``pxl_n_sample``-th valid pixel from ``input_img`` into a
    2D NumPy array of shape ``(n_sampled, n_bands)``.

    :param input_img: Path to the input image file.
    :param pxl_n_sample: Sampling stride (e.g., ``100`` keeps every 100th valid pixel).
    :param no_data_val: Optional no-data value to exclude across all bands.
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).
    :return: 2D NumPy array ``(n_sampled, n_bands)``.

    .. code:: python

        import rsgispy.imageutils.sampling
        pxls = rsgispy.imageutils.sampling.extract_img_pxl_sample(
            "input_img.tif", 100, no_data_val=0.0
        )
    """
    # pylint: disable=too-many-locals
    if pxl_n_sample < 1:
        raise rsgispy.RSGISPyException("pxl_n_sample must be at least 1.")
    ensure_file_exists(input_img)

    with rasterio.open(input_img, "r") as src:
        n_bands = src.count
        np_dtype = np.dtype(src.dtypes[0])

    nd_is_nan = no_data_val is not None and math.isnan(float(no_data_val))
    chunks: List[np.ndarray] = []
    offset = 0

    for _, band_arrays in iter_single_image_band_blocks(
        input_img=input_img,
        read_dtype=np_dtype,
        use_pbar=use_pbar,
    ):
        pxls = np.stack([b.ravel() for b in band_arrays], axis=1)
        if no_data_val is not None:
            if nd_is_nan:
                valid = ~np.isnan(pxls).any(axis=1)
            else:
                valid = (pxls != no_data_val).all(axis=1)
            pxls = pxls[valid]

        n_valid = pxls.shape[0]
        if n_valid == 0:
            continue

        start_idx = (-offset) % pxl_n_sample
        if start_idx < n_valid:
            chunks.append(pxls[start_idx::pxl_n_sample])
        offset = (offset + n_valid) % pxl_n_sample

    if not chunks:
        return np.empty((0, n_bands), dtype=np_dtype)
    return np.concatenate(chunks, axis=0)


def extract_img_pxl_vals_in_msk(
    input_img: str,
    img_bands: Sequence[int],
    in_msk_img: str,
    img_mask_val: int,
    no_data_val: Optional[float] = None,
    use_pbar: bool = True,
) -> np.ndarray:
    """
    Extract pixel values from ``img_bands`` of ``input_img`` where band 1 of
    ``in_msk_img`` equals ``img_mask_val``.

    :param input_img: Path to the input image file.
    :param img_bands: Sequence of 1-based band indices to extract.
    :param in_msk_img: Path to the mask image file (band 1 is used).
    :param img_mask_val: Mask value identifying pixels to extract.
    :param no_data_val: Optional no-data value to exclude across selected bands.
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).
    :return: 2D NumPy array ``(n_values, len(img_bands))``.

    .. code:: python

        import rsgispy.imageutils.sampling
        pxls = rsgispy.imageutils.sampling.extract_img_pxl_vals_in_msk(
            "input_img.tif", [1, 2, 3], "mask_img.tif", 1
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    if not img_bands:
        raise rsgispy.RSGISPyException(
            "img_bands must contain at least one band index."
        )
    band_list = [int(b) for b in img_bands]
    nd_is_nan = no_data_val is not None and math.isnan(float(no_data_val))

    with ExitStack() as stack:
        datasets, paths = open_multi_image_datasets(stack, [input_img, in_msk_img])
        img_ds = datasets[paths[0]]
        msk_ds = datasets[paths[1]]
        for bidx in band_list:
            validate_band_index(bidx, img_ds.count, input_img)

        np_dtype = np.dtype(img_ds.dtypes[0])
        grid = compute_spatial_intersection(datasets)
        img_off_x, img_off_y = grid.offsets[paths[0]]
        msk_off_x, msk_off_y = grid.offsets[paths[1]]

        chunks: List[np.ndarray] = []
        for dst_win in iter_dataset_windows(
            grid.height, grid.width, use_pbar=use_pbar
        ):
            col_off, row_off = int(dst_win.col_off), int(dst_win.row_off)
            win_w, win_h = int(dst_win.width), int(dst_win.height)
            img_win = Window(
                img_off_x + col_off, img_off_y + row_off, win_w, win_h
            )
            msk_win = Window(
                msk_off_x + col_off, msk_off_y + row_off, win_w, win_h
            )
            msk_arr = msk_ds.read(1, window=msk_win).ravel()
            mask_hits = msk_arr == img_mask_val
            if np.any(mask_hits):
                sel_bands = np.stack(
                    [
                        img_ds.read(bidx, window=img_win).ravel()
                        for bidx in band_list
                    ],
                    axis=1,
                )
                pxls = sel_bands[mask_hits]
                if no_data_val is not None:
                    if nd_is_nan:
                        valid = ~np.isnan(pxls).any(axis=1)
                    else:
                        valid = (pxls != no_data_val).all(axis=1)
                    pxls = pxls[valid]
                if pxls.shape[0] > 0:
                    chunks.append(pxls)

    if not chunks:
        return np.empty((0, len(band_list)), dtype=np_dtype)
    return np.concatenate(chunks, axis=0)


def assign_random_pxls(
    input_img: str,
    output_img: str,
    n_pts: int,
    img_band: int = 1,
    gdalformat: str = "GTIFF",
    edge_pxl: int = 0,
    use_no_data: bool = True,
    rnd_seed: Optional[int] = None,
    use_pbar: bool = True,
) -> None:
    """
    Create a single-band ``uint8`` binary mask with ``n_pts`` randomly located
    pixels set to ``1`` (remaining pixels ``0``), optionally excluding an edge
    border and no-data pixels from ``input_img``.

    :param input_img: Reference input image path.
    :param output_img: Output binary mask image path.
    :param n_pts: Number of random pixels to select.
    :param img_band: 1-based band index inspected for no-data values (Default = ``1``).
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param edge_pxl: Edge buffer in pixels excluded from sampling (Default = ``0``).
    :param use_no_data: If ``True``, avoid pixels equal to the band no-data value
                        (Default = ``True``).
    :param rnd_seed: Optional random seed for reproducibility (Default = ``None``).
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).

    .. code:: python

        import rsgispy.imageutils.sampling
        rsgispy.imageutils.sampling.assign_random_pxls(
            "input_img.tif", "rand_pts.tif", 500, edge_pxl=5, rnd_seed=42
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    if edge_pxl < 0:
        raise rsgispy.RSGISPyException(
            "edge_pxl value must be greater than or equal to 0."
        )
    if n_pts < 1:
        raise rsgispy.RSGISPyException("n_pts must be at least 1.")

    x_size, y_size = get_img_size(input_img)
    x_min = edge_pxl
    x_max = x_size - edge_pxl
    y_min = edge_pxl
    y_max = y_size - edge_pxl
    if x_min >= x_max or y_min >= y_max:
        raise rsgispy.RSGISPyException(
            "edge_pxl leaves no valid interior pixels for sampling."
        )

    rng = np.random.default_rng(rnd_seed)
    no_data_val = (
        get_img_no_data_value(input_img, img_band) if use_no_data else None
    )

    if no_data_val is not None:
        with rasterio.open(input_img, "r") as src:
            validate_band_index(img_band, src.count, input_img)
            interior = src.read(
                img_band,
                window=Window(x_min, y_min, x_max - x_min, y_max - y_min),
            )
        if math.isnan(float(no_data_val)):
            valid_r, valid_c = np.where(~np.isnan(interior))
        else:
            valid_r, valid_c = np.where(interior != no_data_val)
        if valid_r.size < n_pts:
            raise rsgispy.RSGISPyException(
                f"Requested {n_pts} random pixels, but only {valid_r.size} "
                "valid interior pixels are available."
            )
        sel = rng.choice(valid_r.size, size=n_pts, replace=False)
        out_x_coords = valid_c[sel] + x_min
        out_y_coords = valid_r[sel] + y_min
    else:
        int_w = x_max - x_min
        int_h = y_max - y_min
        total_int = int_w * int_h
        if total_int < n_pts:
            raise rsgispy.RSGISPyException(
                f"Requested {n_pts} random pixels, but only {total_int} "
                "interior pixels exist."
            )
        flat_sel = rng.choice(total_int, size=n_pts, replace=False)
        out_y_coords = (flat_sel // int_w) + y_min
        out_x_coords = (flat_sel % int_w) + x_min

    create_copy_img(
        input_img,
        output_img,
        1,
        0,
        gdalformat,
        rsgispy.TYPE_8UINT,
        use_pbar=use_pbar,
    )
    set_img_pxl_values(output_img, 1, out_x_coords, out_y_coords, 1)


def generate_random_pxl_vals_img(
    input_img: str,
    output_img: str,
    gdalformat: str,
    low_val: int,
    up_val: int,
    rnd_seed: Optional[int] = None,
    use_pbar: bool = True,
) -> None:
    """
    Create a single-band ``int32`` raster matching ``input_img`` populated with
    random integer values in ``[low_val, up_val)``.

    :param input_img: Reference input image path.
    :param output_img: Output image path.
    :param gdalformat: Output GDAL format name (e.g., ``'KEA'``, ``'GTIFF'``).
    :param low_val: Lower inclusive integer bound.
    :param up_val: Upper exclusive integer bound.
    :param rnd_seed: Optional random seed for reproducibility (Default = ``None``).
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).

    .. code:: python

        import rsgispy.imageutils.sampling
        rsgispy.imageutils.sampling.generate_random_pxl_vals_img(
            "input_img.tif", "rand_vals.tif", "GTIFF", 1, 10, rnd_seed=42
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    ensure_file_exists(input_img)
    if int(low_val) >= int(up_val):
        raise rsgispy.RSGISPyException(
            f"low_val ({low_val}) must be strictly less than up_val ({up_val})."
        )

    rng = np.random.default_rng(rnd_seed)

    with ExitStack() as stack:
        src = stack.enter_context(rasterio.open(input_img, "r"))
        dst = create_output_from_dataset(
            stack=stack,
            src=src,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype="int32",
            count=1,
        )
        for win in iter_dataset_windows(
            src.height, src.width, use_pbar=use_pbar
        ):
            blk = rng.integers(
                int(low_val),
                int(up_val),
                size=(int(win.height), int(win.width)),
                dtype=np.int32,
            )
            dst.write(blk, 1, window=win)
