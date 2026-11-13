"""
The imagecalc.encoding module provides integer recoding and one-hot category
encoding utilities for raster images using ``rasterio``.
"""

from contextlib import ExitStack
from typing import Any, Dict, Mapping, Optional

import numpy as np
import rasterio

import rsgispy
from rsgispy.imagecalc.stats import get_unique_values
from rsgispy.imageprocess import (
    cast_result_block,
    compute_spatial_intersection,
    ensure_file_exists,
    iter_dataset_windows,
    open_output_dataset,
)

__all__ = [
    "recode_int_raster",
    "one_hot_category_encoding",
]


def recode_int_raster(
    input_img: str,
    output_img: str,
    recode_dict: Mapping[int, int],
    keep_vals_not_in_dict: bool = True,
    gdalformat: str = "GTIFF",
    datatype: int = rsgispy.TYPE_32INT,
    use_pbar: bool = True,
    creation_opts: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Recode pixel values in band 1 of an integer raster using ``recode_dict``.

    :param input_img: Path to the input image file (band 1 is recoded).
    :param output_img: Path to the output image file.
    :param recode_dict: Mapping of ``{input_val: output_val}`` integer pairs.
    :param keep_vals_not_in_dict: If ``True``, copy pixels not in ``recode_dict``
                                  to the output; if ``False``, set them to ``0``
                                  (Default = ``True``).
    :param gdalformat: GDAL/rasterio driver format for the output file
                       (Default = ``'GTIFF'``).
    :param datatype: Output data type constant from ``rsgispy.TYPE_*``
                     (Default = ``rsgispy.TYPE_32INT``).
    :param use_pbar: If True, display a tqdm progress bar (Default = True).
    :param creation_opts: Optional mapping of GDAL creation options.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    if recode_dict is None:
        raise rsgispy.RSGISPyException("recode_dict must not be None.")
    ensure_file_exists(input_img)

    out_rio_dtype = rsgispy.get_rasterio_datatype(datatype)
    out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(datatype))

    with ExitStack() as stack:
        try:
            src = stack.enter_context(rasterio.open(input_img, "r"))
        except Exception as err:
            raise rsgispy.RSGISPyException(
                f"Could not open input image '{input_img}': {err}"
            ) from err

        grid = compute_spatial_intersection({input_img: src})
        dst = open_output_dataset(
            stack=stack,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype=out_rio_dtype,
            grid=grid,
            count=1,
            creation_opts=creation_opts,
        )
        if src.nodata is not None:
            dst.nodata = src.nodata

        for out_win in iter_dataset_windows(
            grid.height, grid.width, use_pbar=use_pbar
        ):
            in_block = src.read(1, window=out_win)
            win_h = int(out_win.height)
            win_w = int(out_win.width)
            if keep_vals_not_in_dict:
                out_arr = cast_result_block(in_block, win_h, win_w, out_np_dtype)
            else:
                out_arr = np.zeros((win_h, win_w), dtype=out_np_dtype)

            for src_val, dst_val in recode_dict.items():
                out_arr[in_block == src_val] = dst_val

            dst.write(out_arr, 1, window=out_win)


def one_hot_category_encoding(
    input_img: str,
    output_img: str,
    recode_dict: Optional[Mapping[int, int]] = None,
    gdalformat: str = "GTIFF",
    no_data_val: Optional[int] = None,
    use_pbar: bool = True,
    creation_opts: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Encode a categorical single-band raster image into a multi-band binary
    (``uint8``) one-hot encoded image.

    If ``recode_dict`` is provided, keys map input pixel values to 1-based output
    band indices. Otherwise, unique values in band 1 of ``input_img`` (excluding
    ``no_data_val``) are sorted ascending and mapped to consecutive bands.

    :param input_img: Path to the input categorical image file.
    :param output_img: Path to the output one-hot encoded image file.
    :param recode_dict: Optional mapping ``{pixel_value: output_band_1_based}``.
    :param gdalformat: GDAL/rasterio driver format for the output file
                       (Default = ``'GTIFF'``).
    :param no_data_val: Optional no-data value to exclude when inferring unique
                        categories from ``input_img``.
    :param use_pbar: If True, display a tqdm progress bar (Default = True).
    :param creation_opts: Optional mapping of GDAL creation options.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    ensure_file_exists(input_img)

    lcl_recode_dict: Dict[Any, int] = {}
    if recode_dict is None:
        unq_vals = sorted(get_unique_values(input_img, img_band=1).tolist())
        if no_data_val is not None:
            unq_vals = [val for val in unq_vals if val != no_data_val]
        if not unq_vals:
            raise rsgispy.RSGISPyException(
                "No valid category values found to encode."
            )
        for idx, unq_val in enumerate(unq_vals):
            lcl_recode_dict[unq_val] = idx
        n_out_bands = len(lcl_recode_dict)
    else:
        if len(recode_dict) == 0:
            raise rsgispy.RSGISPyException("recode_dict must not be empty.")
        n_out_bands = 0
        for img_val, band in recode_dict.items():
            band_int = int(band)
            if band_int < 1:
                raise rsgispy.RSGISPyException(
                    "The band number must be greater than or equal to 1."
                )
            n_out_bands = max(n_out_bands, band_int)
            lcl_recode_dict[img_val] = band_int - 1

    with ExitStack() as stack:
        try:
            src = stack.enter_context(rasterio.open(input_img, "r"))
        except Exception as err:
            raise rsgispy.RSGISPyException(
                f"Could not open input image '{input_img}': {err}"
            ) from err

        grid = compute_spatial_intersection({input_img: src})
        dst = open_output_dataset(
            stack=stack,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype="uint8",
            grid=grid,
            count=n_out_bands,
            creation_opts=creation_opts,
        )

        for out_win in iter_dataset_windows(
            grid.height, grid.width, use_pbar=use_pbar
        ):
            in_block = src.read(1, window=out_win)
            win_h = int(out_win.height)
            win_w = int(out_win.width)
            out_arr = np.zeros((n_out_bands, win_h, win_w), dtype=np.uint8)
            for img_val, band_idx in lcl_recode_dict.items():
                out_arr[band_idx, in_block == img_val] = 1
            dst.write(out_arr, window=out_win)
