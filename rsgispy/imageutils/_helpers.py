"""
Internal shared raster helpers for :mod:`rsgispy.imageutils`.
"""

from contextlib import ExitStack
from typing import Any, Dict, Mapping, Optional, Sequence, Union

import numpy as np
import rasterio
import rasterio.crs
from rasterio.windows import Window
from rasterio.windows import transform as window_transform

import rsgispy
from rsgispy.imageprocess import (
    IntersectionGrid,
    cast_result_block,
    ensure_file_exists,
    iter_dataset_windows,
    maybe_pop_stats,
    normalise_creation_options,
    open_output_dataset,
    parse_epsg_crs,
    resolve_out_dtype,
    rsgis_dtype_from_np_dtype,
)

__all__ = [
    "ensure_img_exists",
    "normalise_creation_options",
    "parse_epsg_crs",
    "rsgis_dtype_from_np_dtype",
    "maybe_pop_stats",
    "copy_raster_window",
]


def ensure_img_exists(img_path: str) -> None:
    """Raise :class:`rsgispy.RSGISPyException` if ``img_path`` does not exist."""
    ensure_file_exists(img_path, label="Input image")


def copy_raster_window(
    input_img: str,
    output_img: str,
    col_off: int,
    row_off: int,
    width: int,
    height: int,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
    creation_opts: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
) -> None:
    """Copy a pixel window ``(col_off, row_off, width, height)`` to ``output_img``."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    norm_opts = normalise_creation_options(creation_opts)

    with ExitStack() as stack:
        src = stack.enter_context(rasterio.open(input_img, "r"))
        _, out_rio_dtype, out_np_dtype = resolve_out_dtype(
            datatype, ref_np_dtype=src.dtypes[0]
        )

        roi_win = Window(col_off, row_off, width, height)
        out_transform = window_transform(roi_win, src.transform)
        grid = IntersectionGrid(
            width=width,
            height=height,
            transform=out_transform,
            crs=src.crs,
            offsets={},
        )
        dst = open_output_dataset(
            stack=stack,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype=out_rio_dtype,
            grid=grid,
            count=src.count,
            output_exists=False,
            creation_opts=norm_opts,
        )
        eff_nd = no_data_val if no_data_val is not None else src.nodata
        if eff_nd is not None:
            dst.nodata = eff_nd
        for bidx in range(1, src.count + 1):
            desc = src.descriptions[bidx - 1]
            if desc:
                dst.set_band_description(bidx, desc)

        for dst_win in iter_dataset_windows(height, width, use_pbar=use_pbar):
            src_win = Window(
                col_off + int(dst_win.col_off),
                row_off + int(dst_win.row_off),
                int(dst_win.width),
                int(dst_win.height),
            )
            data = src.read(window=src_win)
            out_arr = cast_result_block(
                data, int(dst_win.height), int(dst_win.width), out_np_dtype
            )
            dst.write(out_arr, window=dst_win)

    maybe_pop_stats(
        output_img=output_img,
        calc_stats=calc_stats,
        no_data_val=eff_nd,
        use_pbar=use_pbar,
        gdalformat=gdalformat,
    )
