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


def rsgis_dtype_from_np_dtype(
    dtype_obj: Any, fallback: int = rsgispy.TYPE_32FLOAT
) -> int:
    """Resolve an ``rsgispy.TYPE_*`` constant from a NumPy/rasterio dtype."""
    dtype_str = str(np.dtype(dtype_obj))
    try:
        return rsgispy.get_rsgis_datatype(dtype_str)
    except rsgispy.RSGISPyException:
        if dtype_str == "bool":
            return rsgispy.TYPE_8UINT
        return fallback


def maybe_pop_stats(
    output_img: str,
    calc_stats: bool,
    no_data_val: Optional[float] = None,
    thematic: bool = False,
    use_pbar: bool = True,
) -> None:
    """Populate continuous or thematic image statistics if ``calc_stats`` is True."""
    if not calc_stats:
        return
    # Lazy import avoids circular dependencies with stats -> imagecalc.
    from rsgispy.imageutils.stats import (  # pylint: disable=import-outside-toplevel
        pop_img_stats,
        pop_thmt_img_stats,
    )

    if thematic:
        pop_thmt_img_stats(
            input_img=output_img,
            add_clr_tab=True,
            calc_pyramids=True,
            ignore_zero=True,
            use_pbar=use_pbar,
        )
    else:
        pop_img_stats(
            input_img=output_img,
            no_data_val=no_data_val,
            calc_pyramids=True,
            use_pbar=use_pbar,
        )


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
        eff_dtype = (
            rsgis_dtype_from_np_dtype(src.dtypes[0])
            if datatype is None
            else datatype
        )
        out_rio_dtype = rsgispy.get_rasterio_datatype(eff_dtype)
        out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(eff_dtype))

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
    )
