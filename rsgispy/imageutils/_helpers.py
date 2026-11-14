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
    open_output_dataset,
)


def ensure_img_exists(img_path: str) -> None:
    """Raise :class:`rsgispy.RSGISPyException` if ``img_path`` does not exist."""
    ensure_file_exists(img_path, label="Input image")


def normalise_creation_options(
    options: Optional[Union[Sequence[str], Mapping[str, Any]]],
) -> Optional[Dict[str, Any]]:
    """
    Convert ``options`` given as either a sequence of ``'KEY=VALUE'`` strings
    or a dictionary into a ``Dict[str, Any]`` mapping, or ``None`` if empty.
    """
    if not options:
        return None
    if isinstance(options, Mapping):
        return dict(options) if options else None
    parsed: Dict[str, Any] = {}
    for item in options:
        item_str = str(item).strip()
        if "=" in item_str:
            key, val = item_str.split("=", 1)
            parsed[key.strip()] = val.strip()
    return parsed if parsed else None


def parse_epsg_crs(epsg_code: Optional[int]) -> Optional[Any]:
    """
    Resolve a ``rasterio.crs.CRS`` from an integer ``epsg_code``.
    """
    # pylint: disable=c-extension-no-member
    if epsg_code is None:
        return None
    try:
        return rasterio.crs.CRS.from_epsg(int(epsg_code))
    except Exception as err:
        raise rsgispy.RSGISPyException(
            f"Could not parse EPSG code '{epsg_code}': {err}"
        ) from err


def get_epsg_from_dataset(src: Any, input_img: str) -> int:
    """
    Extract the integer EPSG code from an opened rasterio dataset ``src``.
    """
    epsg_code = src.crs.to_epsg() if src.crs is not None else None
    if epsg_code is None:
        raise rsgispy.RSGISPyException(
            f"Could not determine EPSG code from input image '{input_img}'."
        )
    return int(epsg_code)


def copy_raster_window(
    input_img: str,
    output_img: str,
    col_off: int,
    row_off: int,
    width: int,
    height: int,
    gdalformat: str,
    datatype: int,
    options: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
    use_pbar: bool = True,
) -> None:
    """Copy a pixel window ``(col_off, row_off, width, height)`` to ``output_img``."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    out_rio_dtype = rsgispy.get_rasterio_datatype(datatype)
    out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(datatype))
    creation_opts = normalise_creation_options(options)

    with ExitStack() as stack:
        src = stack.enter_context(rasterio.open(input_img, "r"))
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
            creation_opts=creation_opts,
        )
        if src.nodata is not None:
            dst.nodata = src.nodata
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
