"""
The imagecalc.rescale module provides gain/offset and per-band min-max raster
pixel rescaling utilities using ``rasterio``.
"""

from contextlib import ExitStack
from typing import Any, Dict, Mapping, Optional, Sequence, Union

import numpy as np
from rasterio.windows import Window

import rsgispy
from rsgispy.imageprocess import (
    cast_result_block,
    compute_spatial_intersection,
    iter_dataset_windows,
    open_multi_image_datasets,
    open_output_dataset,
)

__all__ = [
    "ImageBandRescale",
    "calc_img_rescale",
    "rescale_img_pxl_vals",
]


class ImageBandRescale:
    """
    Data container storing per-band min-max rescaling parameters for
    :func:`rescale_img_pxl_vals`.

    :param band: 1-based image band index.
    :param in_min: Input image band minimum value for rescaling.
    :param in_max: Input image band maximum value for rescaling.
    :param no_data_val: No-data value in the input image band.
    :param out_min: Output image band minimum value after rescaling.
    :param out_max: Output image band maximum value after rescaling.
    :param out_no_data: No-data value in the output image band.
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-few-public-methods
    def __init__(
        self,
        band: int = 0,
        in_min: float = 0.0,
        in_max: float = 0.0,
        no_data_val: float = 0.0,
        out_min: float = 0.0,
        out_max: float = 0.0,
        out_no_data: float = 0.0,
    ) -> None:
        self.band = band
        self.in_min = in_min
        self.in_max = in_max
        self.no_data_val = no_data_val
        self.out_min = out_min
        self.out_max = out_max
        self.out_no_data = out_no_data

    def __str__(self) -> str:
        return (
            f"Band {self.band} [{self.in_min}, {self.in_max}, {self.no_data_val}] "
            f"[{self.out_min}, {self.out_max}, {self.out_no_data}]"
        )

    def __repr__(self) -> str:
        return self.__str__()


def calc_img_rescale(
    input_imgs: Union[str, Sequence[str]],
    output_img: str,
    gdalformat: str = "GTIFF",
    datatype: int = rsgispy.TYPE_32FLOAT,
    c_no_data_val: float = 0.0,
    c_offset: float = 0.0,
    c_gain: float = 1.0,
    n_no_data_val: float = 0.0,
    n_offset: float = 0.0,
    n_gain: float = 1.0,
    use_pbar: bool = True,
    creation_opts: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Rescale and stack one or more input images by converting from current
    ``(c_offset, c_gain)`` to new ``(n_offset, n_gain)`` scaling.

    Valid pixels are transformed via
    ``(((val - c_offset) / c_gain) * n_gain) + n_offset``, while pixels equal
    to ``c_no_data_val`` (or non-finite) are mapped to ``n_no_data_val``.

    :param input_imgs: Single input image path or sequence of image paths to stack.
    :param output_img: Path to the output stacked image file.
    :param gdalformat: GDAL/rasterio driver format for the output file
                       (Default = ``'GTIFF'``).
    :param datatype: Output data type constant from ``rsgispy.TYPE_*``
                     (Default = ``rsgispy.TYPE_32FLOAT``).
    :param c_no_data_val: Current no-data value in the input imagery.
    :param c_offset: Current offset value.
    :param c_gain: Current gain value (must be non-zero).
    :param n_no_data_val: New no-data value written to the output image.
    :param n_offset: New offset value.
    :param n_gain: New gain value.
    :param use_pbar: If True, display a tqdm progress bar (Default = True).
    :param creation_opts: Optional mapping of GDAL creation options.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    if c_gain == 0.0:
        raise rsgispy.RSGISPyException("c_gain must not be zero.")

    img_list = [input_imgs] if isinstance(input_imgs, str) else list(input_imgs)
    out_rio_dtype = rsgispy.get_rasterio_datatype(datatype)
    out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(datatype))

    with ExitStack() as stack:
        datasets, img_paths = open_multi_image_datasets(stack, img_list)
        grid = compute_spatial_intersection(datasets)
        total_bands = sum(datasets[path].count for path in img_paths)

        dst = open_output_dataset(
            stack=stack,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype=out_rio_dtype,
            grid=grid,
            count=total_bands,
            creation_opts=creation_opts,
        )
        dst.nodata = n_no_data_val

        scale = float(n_gain) / float(c_gain)
        c_off = float(c_offset)
        n_off = float(n_offset)
        c_nd_is_nan = np.isnan(c_no_data_val)

        for out_win in iter_dataset_windows(
            grid.height, grid.width, use_pbar=use_pbar
        ):
            win_h = int(out_win.height)
            win_w = int(out_win.width)
            out_band_idx = 1
            for path in img_paths:
                ds = datasets[path]
                base_col, base_row = grid.offsets[path]
                src_win = Window(
                    base_col + int(out_win.col_off),
                    base_row + int(out_win.row_off),
                    win_w,
                    win_h,
                )
                in_block = ds.read(window=src_win, out_dtype=np.float64)
                for b_idx in range(ds.count):
                    band_arr = in_block[b_idx]
                    if c_nd_is_nan:
                        nodata_mask = ~np.isfinite(band_arr)
                    else:
                        nodata_mask = (band_arr == c_no_data_val) | ~np.isfinite(
                            band_arr
                        )
                    rescaled = ((band_arr - c_off) * scale) + n_off
                    res = np.where(nodata_mask, n_no_data_val, rescaled)
                    out_block = cast_result_block(res, win_h, win_w, out_np_dtype)
                    dst.write(out_block, out_band_idx, window=out_win)
                    out_band_idx += 1


def rescale_img_pxl_vals(
    input_img: str,
    output_img: str,
    gdalformat: str = "GTIFF",
    datatype: int = rsgispy.TYPE_32FLOAT,
    band_rescale_objs: Sequence[ImageBandRescale] = (),
    trim_to_limits: bool = True,
    use_pbar: bool = True,
    creation_opts: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Rescale image pixel values per band using a sequence of
    :class:`ImageBandRescale` specifications.

    :param input_img: Path to the input image file.
    :param output_img: Path to the output image file.
    :param gdalformat: GDAL/rasterio driver format for the output file
                       (Default = ``'GTIFF'``).
    :param datatype: Output data type constant from ``rsgispy.TYPE_*``
                     (Default = ``rsgispy.TYPE_32FLOAT``).
    :param band_rescale_objs: Sequence of :class:`ImageBandRescale` objects
                              defining per-band input/output ranges.
    :param trim_to_limits: If ``True``, clamp non-nodata outputs to the
                           ``[out_min, out_max]`` interval (Default = ``True``).
    :param use_pbar: If True, display a tqdm progress bar (Default = True).
    :param creation_opts: Optional mapping of GDAL creation options.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    if not band_rescale_objs:
        raise rsgispy.RSGISPyException(
            "band_rescale_objs must contain at least one ImageBandRescale."
        )

    out_rio_dtype = rsgispy.get_rasterio_datatype(datatype)
    out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(datatype))

    rescale_dict: Dict[int, ImageBandRescale] = {}
    for rescale_obj in band_rescale_objs:
        if rescale_obj.band < 1:
            raise rsgispy.RSGISPyException(
                f"Band index must be >= 1, got {rescale_obj.band}."
            )
        if rescale_obj.in_max == rescale_obj.in_min:
            raise rsgispy.RSGISPyException(
                f"Band {rescale_obj.band} has in_max == in_min ({rescale_obj.in_min})."
            )
        rescale_dict[rescale_obj.band - 1] = rescale_obj

    with ExitStack() as stack:
        datasets, _ = open_multi_image_datasets(stack, [input_img])
        src = datasets[input_img]
        n_bands = src.count
        for b_idx in range(n_bands):
            if b_idx not in rescale_dict:
                raise rsgispy.RSGISPyException(
                    f"Missing ImageBandRescale specification for band {b_idx + 1}."
                )

        grid = compute_spatial_intersection(datasets)
        dst = open_output_dataset(
            stack=stack,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype=out_rio_dtype,
            grid=grid,
            count=n_bands,
            creation_opts=creation_opts,
        )

        for out_win in iter_dataset_windows(
            grid.height, grid.width, use_pbar=use_pbar
        ):
            win_h = int(out_win.height)
            win_w = int(out_win.width)
            in_block = src.read(window=out_win, out_dtype=np.float64)
            out_bands = []
            for b_idx in range(n_bands):
                cfg = rescale_dict[b_idx]
                band_arr = in_block[b_idx]
                if np.isnan(cfg.no_data_val):
                    nodata_mask = ~np.isfinite(band_arr)
                else:
                    nodata_mask = (band_arr == cfg.no_data_val) | ~np.isfinite(
                        band_arr
                    )

                scaled = (
                    ((band_arr - cfg.in_min) / (cfg.in_max - cfg.in_min))
                    * (cfg.out_max - cfg.out_min)
                ) + cfg.out_min
                if trim_to_limits:
                    low = min(cfg.out_min, cfg.out_max)
                    high = max(cfg.out_min, cfg.out_max)
                    scaled = np.clip(scaled, low, high)

                res = np.where(nodata_mask, cfg.out_no_data, scaled)
                out_bands.append(
                    cast_result_block(res, win_h, win_w, out_np_dtype)
                )
            dst.write(np.stack(out_bands, axis=0), window=out_win)
