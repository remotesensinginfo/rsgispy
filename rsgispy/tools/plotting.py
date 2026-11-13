"""
The tools.plotting module provides functions for reading raster data into
NumPy arrays and applying image stretches for visualisation with Matplotlib
using rasterio.
"""

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from matplotlib.patches import Patch
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.windows import bounds as window_bounds
from rasterio.windows import from_bounds as window_from_bounds

import rsgispy
from rsgispy.imageprocess import ensure_file_exists

__all__ = [
    "get_raster_mpl_imshow",
    "get_thematic_raster_mpl_imshow",
    "limit_range_np_arr",
    "manual_stretch_np_arr",
    "linear_stretch_np_arr",
    "cumulative_stretch_np_arr",
    "stdev_stretch_np_arr",
]


def _get_subset_window_and_coords(
    src: rasterio.io.DatasetReader,
    bbox: Optional[Tuple[float, float, float, float]],
) -> Tuple[Window, Tuple[float, float, float, float]]:
    """
    Compute the clipped rasterio pixel Window and exact spatial extent
    ``(MinX, MaxX, MinY, MaxY)`` for an optional bounding box.
    """
    # pylint: disable=too-many-locals
    if bbox is None:
        win = Window(0, 0, src.width, src.height)
        left, bottom, right, top = src.bounds
        return win, (float(left), float(right), float(bottom), float(top))

    img_left, img_bottom, img_right, img_top = src.bounds
    sub_left, sub_right, sub_bottom, sub_top = (float(v) for v in bbox[:4])

    intersects = (
        (img_right > sub_left)
        and (sub_right > img_left)
        and (img_top > sub_bottom)
        and (sub_top > img_bottom)
    )
    if not intersects:
        raise rsgispy.RSGISPyException(
            "The subset BBOX does not intersect with the image bbox - "
            "is the projection of the bbox correct?"
        )

    inter_left = max(img_left, sub_left)
    inter_right = min(img_right, sub_right)
    inter_bottom = max(img_bottom, sub_bottom)
    inter_top = min(img_top, sub_top)

    raw_win = window_from_bounds(
        inter_left,
        inter_bottom,
        inter_right,
        inter_top,
        transform=src.transform,
    )

    col_start = max(0, min(src.width - 1, math.floor(raw_win.col_off + 0.5)))
    col_stop = max(
        col_start + 1,
        min(src.width, math.floor(raw_win.col_off + raw_win.width + 0.5)),
    )
    row_start = max(0, min(src.height - 1, math.floor(raw_win.row_off + 0.5)))
    row_stop = max(
        row_start + 1,
        min(src.height, math.floor(raw_win.row_off + raw_win.height + 0.5)),
    )

    win = Window(
        col_off=col_start,
        row_off=row_start,
        width=col_stop - col_start,
        height=row_stop - row_start,
    )
    w_left, w_bottom, w_right, w_top = window_bounds(win, src.transform)
    return win, (float(w_left), float(w_right), float(w_bottom), float(w_top))


def get_raster_mpl_imshow(
    input_img: str,
    bands: Optional[Sequence[int]] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """
    Retrieve raster image data as a NumPy array structured for
    ``matplotlib.axes.Axes.imshow`` together with its spatial extent.

    :param input_img: Path to the input raster image file.
    :param bands: Sequence of 1-based band indices (length 1 or 3). If ``None``,
                  all bands in the image are used (must be 1 or 3 bands).
    :param bbox: Optional bounding box ``(MinX, MaxX, MinY, MaxY)`` in the image
                 CRS used to spatially subset the raster.
    :return: Tuple ``(img_data_arr, coords_bbox)`` where ``img_data_arr`` has
             shape ``(height, width)`` for 1 band or ``(height, width, 3)``
             for 3 bands, and ``coords_bbox`` is ``(MinX, MaxX, MinY, MaxY)``.

    .. code:: python

        import rsgispy.tools.plotting

        img_arr, extent = rsgispy.tools.plotting.get_raster_mpl_imshow(
            "sentinel2.tif", bands=[4, 3, 2]
        )
    """
    ensure_file_exists(input_img)

    with rasterio.open(input_img, "r") as src:
        n_bands = src.count
        ref_bands = list(range(1, n_bands + 1))
        sel_bands = list(ref_bands) if bands is None else [int(b) for b in bands]

        for band in sel_bands:
            if band not in ref_bands:
                raise rsgispy.RSGISPyException(
                    f"Band {band} is not valid (i.e., within the image)"
                )

        if len(sel_bands) not in (1, 3):
            raise rsgispy.RSGISPyException(
                "The number of bands specified must be 3 or 1."
            )

        win, coords_bbox = _get_subset_window_and_coords(src, bbox)
        if len(sel_bands) == 1:
            img_data_arr = src.read(sel_bands[0], window=win)
        else:
            img_data_arr = np.moveaxis(src.read(sel_bands, window=win), 0, -1)

    return img_data_arr, coords_bbox


def get_thematic_raster_mpl_imshow(
    input_img: str,
    band: int = 1,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    out_patches: bool = False,
    cls_names_lut: Optional[Mapping[int, Any]] = None,
    alpha_lyr: bool = False,
    ctab_alpha: bool = True,
) -> Tuple[np.ndarray, Tuple[float, float, float, float], Optional[List[Patch]]]:
    """
    Retrieve thematic image data with a colour table as an RGB or RGBA NumPy
    array for ``matplotlib.axes.Axes.imshow``, its spatial extent, and optional
    legend patches.

    :param input_img: Path to the input thematic raster image file.
    :param band: 1-based band index (Default = 1).
    :param bbox: Optional bounding box ``(MinX, MaxX, MinY, MaxY)`` in the image
                 CRS used to spatially subset the raster.
    :param out_patches: If True, build a list of :class:`matplotlib.patches.Patch`
                        objects for creating a legend (Default = False).
    :param cls_names_lut: Optional mapping from pixel class value to class label.
    :param alpha_lyr: If True, include an alpha channel where background pixel
                      value 0 is transparent (Default = False).
    :param ctab_alpha: If True, include alpha values from the colour table
                       (Default = True).
    :return: Tuple ``(img_clr_data_arr, coords_bbox, lgd_out_patches)``.

    .. code:: python

        import rsgispy.tools.plotting

        img_rgba, extent, patches = (
            rsgispy.tools.plotting.get_thematic_raster_mpl_imshow(
                "landcover.kea", out_patches=True, cls_names_lut={1: "Forest"}
            )
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-branches
    ensure_file_exists(input_img)

    with rasterio.open(input_img, "r") as src:
        if band <= 0 or band > src.count:
            raise rsgispy.RSGISPyException(
                f"Band {band} is not valid (i.e., within the image)"
            )

        try:
            clr_tab = src.colormap(band)
        except ValueError as err:
            raise rsgispy.RSGISPyException("No colour table was present") from err

        if not clr_tab:
            raise rsgispy.RSGISPyException("No colour table was present")

        win, coords_bbox = _get_subset_window_and_coords(src, bbox)
        img_data_arr = src.read(band, window=win)

    red_arr = np.zeros_like(img_data_arr, dtype=np.uint8)
    grn_arr = np.zeros_like(img_data_arr, dtype=np.uint8)
    blu_arr = np.zeros_like(img_data_arr, dtype=np.uint8)
    use_alpha = alpha_lyr or ctab_alpha
    alp_arr = np.zeros_like(img_data_arr, dtype=np.uint8) if use_alpha else None

    lgd_out_patches: Optional[List[Patch]] = [] if out_patches else None

    for val, rgba in sorted(clr_tab.items()):
        r_val = int(rgba[0])
        g_val = int(rgba[1])
        b_val = int(rgba[2])
        a_val = int(rgba[3]) if len(rgba) >= 4 else 255

        mask = img_data_arr == val
        if np.any(mask):
            red_arr[mask] = r_val
            grn_arr[mask] = g_val
            blu_arr[mask] = b_val
            if alp_arr is not None:
                if ctab_alpha:
                    alp_arr[mask] = 0 if (alpha_lyr and val == 0) else a_val
                elif alpha_lyr and val > 0:
                    alp_arr[mask] = 255

        if lgd_out_patches is not None and val > 0:
            cls_name = (
                f"{cls_names_lut[val]}"
                if (cls_names_lut is not None and val in cls_names_lut)
                else f"{val}"
            )
            if ctab_alpha:
                patch_clr: Tuple[float, ...] = (
                    r_val / 255.0,
                    g_val / 255.0,
                    b_val / 255.0,
                    a_val / 255.0,
                )
            else:
                patch_clr = (r_val / 255.0, g_val / 255.0, b_val / 255.0)
            lgd_out_patches.append(
                Patch(facecolor=patch_clr, edgecolor=patch_clr, label=cls_name)
            )

    if alp_arr is not None:
        img_clr_data_arr = np.stack([red_arr, grn_arr, blu_arr, alp_arr], axis=-1)
    else:
        img_clr_data_arr = np.stack([red_arr, grn_arr, blu_arr], axis=-1)

    return img_clr_data_arr, coords_bbox, lgd_out_patches


def limit_range_np_arr(
    arr_data: np.ndarray,
    min_thres: float = 0.0,
    min_out_val: float = 0.0,
    max_thres: float = 1.0,
    max_out_val: float = 1.0,
) -> np.ndarray:
    """
    Clamp values of a NumPy array outside ``[min_thres, max_thres]``.

    :param arr_data: Input NumPy array.
    :param min_thres: Lower threshold (Default = 0.0).
    :param min_out_val: Replacement value below ``min_thres`` (Default = 0.0).
    :param max_thres: Upper threshold (Default = 1.0).
    :param max_out_val: Replacement value above ``max_thres`` (Default = 1.0).
    :return: Clamped NumPy array copy.

    .. code:: python

        import numpy as np
        import rsgispy.tools.plotting

        arr = np.array([-0.2, 0.5, 1.4])
        clamped = rsgispy.tools.plotting.limit_range_np_arr(arr, 0.0, 0.0, 1.0, 1.0)
    """
    arr_data_out = arr_data.copy()
    arr_data_out[arr_data < min_thres] = min_out_val
    arr_data_out[arr_data > max_thres] = max_out_val
    return arr_data_out


def _prep_stretch_array(
    arr_data: np.ndarray, no_data_val: Optional[float]
) -> np.ndarray:
    """Copy ``arr_data`` to float64 and mask ``no_data_val`` as NaN."""
    arr_data_out = arr_data.astype(float, copy=True)
    if no_data_val is not None:
        arr_data_out[arr_data == no_data_val] = np.nan
    return arr_data_out


def _rescale_band(
    band_arr: np.ndarray,
    low_val: float,
    up_val: float,
    out_off: float,
    out_gain: float,
) -> np.ndarray:
    """Rescale a single band array cleanly, guarding against zero range."""
    range_val = up_val - low_val
    if range_val == 0 or not np.isfinite(range_val):
        out_arr = np.full_like(band_arr, out_off, dtype=float)
        out_arr[np.isnan(band_arr)] = np.nan
        return out_arr
    return (((band_arr - low_val) / range_val) * out_gain) + out_off


def _finalize_stretch_array(
    arr_data: np.ndarray,
    arr_data_out: np.ndarray,
    no_data_val: Optional[float],
    no_data_clr: Optional[Union[float, Sequence[float]]],
    min_out_val: float,
    max_out_val: float,
    out_int_type: bool,
) -> np.ndarray:
    """Clamp range, apply optional no-data display colour, and cast output dtype."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    arr_data_out = limit_range_np_arr(
        arr_data_out,
        min_thres=min_out_val,
        min_out_val=min_out_val,
        max_thres=max_out_val,
        max_out_val=max_out_val,
    )

    if (no_data_val is not None) and (no_data_clr is not None):
        if arr_data.ndim == 2:
            clr_val = (
                no_data_clr[0]
                if isinstance(no_data_clr, (tuple, list))
                else no_data_clr
            )
            arr_data_out[arr_data == no_data_val] = clr_val
        else:
            for band_idx in range(arr_data.shape[2]):
                clr_val = (
                    no_data_clr[band_idx]
                    if isinstance(no_data_clr, (tuple, list))
                    else no_data_clr
                )
                no_data_msk = arr_data[..., band_idx] == no_data_val
                arr_data_out[no_data_msk, band_idx] = clr_val

    if out_int_type:
        arr_data_out = arr_data_out.astype(int)

    return arr_data_out


def manual_stretch_np_arr(
    arr_data: np.ndarray,
    min_max_vals: Union[Dict[str, float], List[Dict[str, float]]],
    no_data_val: Optional[float] = None,
    out_off: float = 0.0,
    out_gain: float = 1.0,
    out_int_type: bool = False,
    min_out_val: float = 0.0,
    max_out_val: float = 1.0,
    no_data_clr: Optional[Union[float, Tuple[float, float, float]]] = None,
) -> np.ndarray:
    """
    Apply a manual per-band linear stretch using user-specified ``min``/``max`` values.

    :param arr_data: Input NumPy array with shape ``(n, m)`` or ``(n, m, b)``.
    :param min_max_vals: Dict ``{'min': ..., 'max': ...}`` for a 2D array or list
                         of dicts (one per band) for a 3D array.
    :param no_data_val: Optional no-data pixel value.
    :param out_off: Output offset (Default = 0.0).
    :param out_gain: Output gain (Default = 1.0).
    :param out_int_type: If True, cast stretched output to integer (Default = False).
    :param min_out_val: Minimum output clamp value (Default = 0.0).
    :param max_out_val: Maximum output clamp value (Default = 1.0).
    :param no_data_clr: Optional colour value(s) assigned to no-data pixels.
    :return: Stretched NumPy array.

    .. code:: python

        import numpy as np
        import rsgispy.tools.plotting

        arr = np.array([[0.0, 500.0], [1000.0, 2000.0]])
        stretched = rsgispy.tools.plotting.manual_stretch_np_arr(
            arr, min_max_vals={"min": 0.0, "max": 2000.0}
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    arr_data_out = _prep_stretch_array(arr_data, no_data_val)

    if arr_data.ndim == 2:
        if not isinstance(min_max_vals, dict):
            raise rsgispy.RSGISPyException(
                "Just 1 dimension within arr_data and therefore "
                "min_max_vals variable must be a dict."
            )
        if ("min" not in min_max_vals) or ("max" not in min_max_vals):
            raise rsgispy.RSGISPyException(
                "min and max keys must be provided within the dict"
            )
        arr_data_out = _rescale_band(
            arr_data_out,
            float(min_max_vals["min"]),
            float(min_max_vals["max"]),
            out_off,
            out_gain,
        )
    else:
        if not isinstance(min_max_vals, list):
            raise rsgispy.RSGISPyException(
                "arr_data has more than 1 dimension and therefore "
                "min_max_vals variable must be a list."
            )
        n_bands = arr_data.shape[2]
        if n_bands != len(min_max_vals):
            raise rsgispy.RSGISPyException(
                "length of min_max_vals must be the same as the number "
                "of bands in arr_data."
            )
        for band_idx in range(n_bands):
            band_mm = min_max_vals[band_idx]
            if ("min" not in band_mm) or ("max" not in band_mm):
                raise rsgispy.RSGISPyException(
                    "min and max keys must be provided within the dict"
                )
            arr_data_out[..., band_idx] = _rescale_band(
                arr_data_out[..., band_idx],
                float(band_mm["min"]),
                float(band_mm["max"]),
                out_off,
                out_gain,
            )

    return _finalize_stretch_array(
        arr_data=arr_data,
        arr_data_out=arr_data_out,
        no_data_val=no_data_val,
        no_data_clr=no_data_clr,
        min_out_val=min_out_val,
        max_out_val=max_out_val,
        out_int_type=out_int_type,
    )


def linear_stretch_np_arr(
    arr_data: np.ndarray,
    no_data_val: Optional[float] = None,
    out_off: float = 0.0,
    out_gain: float = 1.0,
    out_int_type: bool = False,
    min_out_val: float = 0.0,
    max_out_val: float = 1.0,
    no_data_clr: Optional[Union[float, Tuple[float, float, float]]] = None,
) -> np.ndarray:
    """
    Apply a per-band min-max linear stretch to a NumPy array.

    :param arr_data: Input NumPy array with shape ``(n, m)`` or ``(n, m, b)``.
    :param no_data_val: Optional no-data pixel value.
    :param out_off: Output offset (Default = 0.0).
    :param out_gain: Output gain (Default = 1.0).
    :param out_int_type: If True, cast stretched output to integer (Default = False).
    :param min_out_val: Minimum output clamp value (Default = 0.0).
    :param max_out_val: Maximum output clamp value (Default = 1.0).
    :param no_data_clr: Optional colour value(s) assigned to no-data pixels.
    :return: Stretched NumPy array.

    .. code:: python

        import numpy as np
        import rsgispy.tools.plotting

        arr = np.array([[10.0, 50.0], [100.0, 200.0]])
        stretched = rsgispy.tools.plotting.linear_stretch_np_arr(arr)
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    arr_data_out = _prep_stretch_array(arr_data, no_data_val)

    if arr_data.ndim == 2:
        min_val = float(np.nanmin(arr_data_out))
        max_val = float(np.nanmax(arr_data_out))
        arr_data_out = _rescale_band(
            arr_data_out, min_val, max_val, out_off, out_gain
        )
    else:
        for band_idx in range(arr_data.shape[2]):
            band_slice = arr_data_out[..., band_idx]
            min_val = float(np.nanmin(band_slice))
            max_val = float(np.nanmax(band_slice))
            arr_data_out[..., band_idx] = _rescale_band(
                band_slice, min_val, max_val, out_off, out_gain
            )

    return _finalize_stretch_array(
        arr_data=arr_data,
        arr_data_out=arr_data_out,
        no_data_val=no_data_val,
        no_data_clr=no_data_clr,
        min_out_val=min_out_val,
        max_out_val=max_out_val,
        out_int_type=out_int_type,
    )


def cumulative_stretch_np_arr(
    arr_data: np.ndarray,
    no_data_val: Optional[float] = None,
    lower: float = 2.0,
    upper: float = 98.0,
    out_off: float = 0.0,
    out_gain: float = 1.0,
    out_int_type: bool = False,
    min_out_val: float = 0.0,
    max_out_val: float = 1.0,
    no_data_clr: Optional[Union[float, Tuple[float, float, float]]] = None,
) -> np.ndarray:
    """
    Apply a per-band cumulative percentile stretch to a NumPy array.

    :param arr_data: Input NumPy array with shape ``(n, m)`` or ``(n, m, b)``.
    :param no_data_val: Optional no-data pixel value.
    :param lower: Lower percentile (Default = 2.0).
    :param upper: Upper percentile (Default = 98.0).
    :param out_off: Output offset (Default = 0.0).
    :param out_gain: Output gain (Default = 1.0).
    :param out_int_type: If True, cast stretched output to integer (Default = False).
    :param min_out_val: Minimum output clamp value (Default = 0.0).
    :param max_out_val: Maximum output clamp value (Default = 1.0).
    :param no_data_clr: Optional colour value(s) assigned to no-data pixels.
    :return: Stretched NumPy array.

    .. code:: python

        import numpy as np
        import rsgispy.tools.plotting

        arr = np.arange(100, dtype=float).reshape(10, 10)
        stretched = rsgispy.tools.plotting.cumulative_stretch_np_arr(
            arr, lower=2.0, upper=98.0
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    arr_data_out = _prep_stretch_array(arr_data, no_data_val)

    if arr_data.ndim == 2:
        min_val, max_val = np.nanpercentile(arr_data_out, [lower, upper])
        arr_data_out = _rescale_band(
            arr_data_out, float(min_val), float(max_val), out_off, out_gain
        )
    else:
        for band_idx in range(arr_data.shape[2]):
            band_slice = arr_data_out[..., band_idx]
            min_val, max_val = np.nanpercentile(band_slice, [lower, upper])
            arr_data_out[..., band_idx] = _rescale_band(
                band_slice, float(min_val), float(max_val), out_off, out_gain
            )

    return _finalize_stretch_array(
        arr_data=arr_data,
        arr_data_out=arr_data_out,
        no_data_val=no_data_val,
        no_data_clr=no_data_clr,
        min_out_val=min_out_val,
        max_out_val=max_out_val,
        out_int_type=out_int_type,
    )


def stdev_stretch_np_arr(
    arr_data: np.ndarray,
    no_data_val: Optional[float] = None,
    n_stdevs: float = 2.0,
    out_off: float = 0.0,
    out_gain: float = 1.0,
    out_int_type: bool = False,
    min_out_val: float = 0.0,
    max_out_val: float = 1.0,
    no_data_clr: Optional[Union[float, Tuple[float, float, float]]] = None,
) -> np.ndarray:
    """
    Apply a per-band standard deviation stretch to a NumPy array.

    :param arr_data: Input NumPy array with shape ``(n, m)`` or ``(n, m, b)``.
    :param no_data_val: Optional no-data pixel value.
    :param n_stdevs: Number of standard deviations around the mean (Default = 2.0).
    :param out_off: Output offset (Default = 0.0).
    :param out_gain: Output gain (Default = 1.0).
    :param out_int_type: If True, cast stretched output to integer (Default = False).
    :param min_out_val: Minimum output clamp value (Default = 0.0).
    :param max_out_val: Maximum output clamp value (Default = 1.0).
    :param no_data_clr: Optional colour value(s) assigned to no-data pixels.
    :return: Stretched NumPy array.

    .. code:: python

        import numpy as np
        import rsgispy.tools.plotting

        arr = np.arange(100, dtype=float).reshape(10, 10)
        stretched = rsgispy.tools.plotting.stdev_stretch_np_arr(arr, n_stdevs=2.0)
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    arr_data_out = _prep_stretch_array(arr_data, no_data_val)

    def _stretch_1d_bounds(band_slice: np.ndarray) -> Tuple[float, float]:
        min_val = float(np.nanmin(band_slice))
        max_val = float(np.nanmax(band_slice))
        mean_val = float(np.nanmean(band_slice))
        std_val = float(np.nanstd(band_slice))
        low_val = max(min_val, mean_val - (std_val * n_stdevs))
        up_val = min(max_val, mean_val + (std_val * n_stdevs))
        return low_val, up_val

    if arr_data.ndim == 2:
        low_val, up_val = _stretch_1d_bounds(arr_data_out)
        arr_data_out = _rescale_band(
            arr_data_out, low_val, up_val, out_off, out_gain
        )
    else:
        for band_idx in range(arr_data.shape[2]):
            band_slice = arr_data_out[..., band_idx]
            low_val, up_val = _stretch_1d_bounds(band_slice)
            arr_data_out[..., band_idx] = _rescale_band(
                band_slice, low_val, up_val, out_off, out_gain
            )

    arr_data_out[arr_data_out < out_off] = out_off
    arr_data_out[arr_data_out > (out_gain + out_off)] = out_gain + out_off

    return _finalize_stretch_array(
        arr_data=arr_data,
        arr_data_out=arr_data_out,
        no_data_val=no_data_val,
        no_data_clr=no_data_clr,
        min_out_val=min_out_val,
        max_out_val=max_out_val,
        out_int_type=out_int_type,
    )
