"""
Projection of species distribution models to other scenarios (e.g. future
climates) for :mod:`rsgispy.modelling.species_distribution`: copying the
environmental variable descriptors to scenario rasters, and range-change maps
comparing two presence/absence maps.
"""

import copy
import os
from contextlib import ExitStack
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

import rsgispy
from rsgispy.imageprocess import (
    IntersectionGrid,
    get_default_block_size,
    open_output_dataset,
)
from rsgispy.imageutils.colourtable import set_thematic_classes
from rsgispy.modelling._helpers import EnvVarInfo, lookup_env_var
import rsgispy.tools.files

__all__ = [
    "copy_env_vars_for_scenario",
    "calc_range_change",
]

RANGE_CHANGE_CLASSES: Dict[int, Tuple[str, str]] = {
    1: ("Absent", "#d9d9d9"),
    2: ("Lost", "#d7301f"),
    3: ("Gained", "#2b8cbe"),
    4: ("Stable", "#31a354"),
}
_RANGE_NO_DATA = 0


def copy_env_vars_for_scenario(
    env_vars: Mapping[str, EnvVarInfo],
    file_map: Mapping[str, Union[str, os.PathLike]],
    *,
    band_map: Optional[Mapping[str, int]] = None,
) -> Dict[str, EnvVarInfo]:
    """
    Copy environmental variable descriptors for projecting a model to another
    scenario (e.g. a future climate or another region): each variable in
    ``file_map`` gets the new raster (and optionally band), and everything else
    is kept, including the normalisation coefficients (``norm_mean``,
    ``norm_std``) calculated from the training data. Variables not in
    ``file_map`` (e.g. terrain that does not change) keep their original
    rasters. ``env_vars`` is not modified.

    Pass the result to the ``pred_*`` functions (with the same
    ``analysis_vars`` and ``normalise_data`` as for the current map) and to
    :func:`calc_mess` / :func:`calc_exdet` to see where the projection
    extrapolates.

    :param env_vars: Environmental variable descriptors used to fit the model.
    :param file_map: Mapping of variable name to the scenario raster path.
    :param band_map: Optional mapping of variable name to the band of its
                     scenario raster (Default: keep the original band).
    :return: New mapping of variable name to :class:`EnvVarInfo`.

    .. code:: python

        import rsgispy.modelling.species_distribution as sdm

        future_vars = sdm.copy_env_vars_for_scenario(
            env_vars, {"winter_temp": "winter_temp_2080.tif"}
        )
        sdm.pred_ensemble_sklearn_mdls_prob(
            mdls, "valid_mask.tif", 1, future_vars, analysis_vars,
            "sdm_prob_2080.tif",
        )
    """
    unknown = [name for name in file_map if name not in env_vars]
    if band_map is not None:
        unknown += [
            name for name in band_map if name not in env_vars and name not in unknown
        ]
    if unknown:
        raise rsgispy.RSGISPyException(
            f"Variable(s) {unknown} are not in env_vars."
        )
    out: Dict[str, EnvVarInfo] = {}
    for key, info in env_vars.items():
        new_info = copy.copy(info)
        if key in file_map:
            new_info.file = os.fspath(file_map[key])
        if band_map is not None and key in band_map:
            new_info.band = int(band_map[key])
        out[key] = new_info
    # Check every new raster exists before any processing starts.
    for key in file_map:
        info = lookup_env_var(out, key)
        if info.file is None or not os.path.exists(info.file):
            raise rsgispy.RSGISPyException(
                f"Scenario raster for '{key}' does not exist: {info.file!r}."
            )
    return out


def _iter_windows(n_rows: int, n_cols: int, block_size: Optional[int]):
    blk = int(block_size) if block_size is not None else get_default_block_size()
    for row_off in range(0, n_rows, blk):
        for col_off in range(0, n_cols, blk):
            yield Window(
                col_off,
                row_off,
                min(blk, n_cols - col_off),
                min(blk, n_rows - row_off),
            )


def _same_grid(src_a: Any, src_b: Any) -> bool:
    return (
        src_a.width == src_b.width
        and src_a.height == src_b.height
        and src_a.transform.almost_equals(src_b.transform)
        and (src_a.crs == src_b.crs or src_a.crs is None or src_b.crs is None)
    )


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
def calc_range_change(
    current_bin_img: str,
    future_bin_img: str,
    output_img: str,
    *,
    in_msk_img: Optional[str] = None,
    img_msk_val: int = 1,
    presence_val: int = 1,
    absence_val: int = 0,
    out_csv: Optional[str] = None,
    gdalformat: str = "GTIFF",
    calc_stats: bool = True,
    block_size: Optional[int] = None,
    use_pbar: bool = True,
) -> Dict[str, float]:
    """
    Compare two presence/absence maps (e.g. thresholded current and future
    predictions) and write a range-change map, a ``uint8`` thematic raster
    with class names and colours:

    * ``1`` Absent — absent in both;
    * ``2`` Lost — present now, absent in the future;
    * ``3`` Gained — absent now, present in the future;
    * ``4`` Stable — present in both;
    * ``0`` no data — outside ``in_msk_img == img_msk_val`` or where either map
      is neither ``presence_val`` nor ``absence_val`` (e.g. its no data).

    The rasters must be on the same grid. Areas are in squared map units of
    the grid (pixel count × pixel area).

    :param current_bin_img: Current presence/absence raster.
    :param future_bin_img: Future (scenario) presence/absence raster.
    :param output_img: Output range-change raster.
    :param in_msk_img: Optional mask raster on the same grid.
    :param img_msk_val: Value of the pixels to use in ``in_msk_img``
                        (Default = ``1``).
    :param presence_val: Presence value in the input maps (Default = ``1``).
    :param absence_val: Absence value in the input maps (Default = ``0``).
    :param out_csv: Optional CSV path for the summary (columns ``metric`` and
                    ``value``, one row per item of the returned dictionary).
    :param gdalformat: Output GDAL format (Default = ``'GTIFF'``).
    :param calc_stats: Calculate statistics and pyramids (Default = ``True``).
    :param block_size: Processing block size in pixels (Default = ``None``).
    :param use_pbar: Show a progress bar (Default = ``True``).
    :return: Dictionary with ``n_absent``, ``n_lost``, ``n_gained``,
             ``n_stable`` (pixel counts), the matching ``area_*`` values, the
             ``current_area`` (lost + stable) and ``future_area`` (gained +
             stable), and, relative to the current range, ``pct_loss``,
             ``pct_gain`` and ``pct_net_change`` (``NaN`` when the current
             range is empty).

    .. code:: python

        import rsgispy.modelling.species_distribution as sdm

        summary = sdm.calc_range_change(
            "sdm_bin_current.tif", "sdm_bin_2080.tif", "range_change_2080.tif",
            in_msk_img="valid_mask.tif", out_csv="range_change_2080.csv",
        )
    """
    for img in (current_bin_img, future_bin_img) + (
        (in_msk_img,) if in_msk_img is not None else ()
    ):
        if not os.path.exists(img):
            raise rsgispy.RSGISPyException(f"Image '{img}' does not exist.")
    if presence_val == absence_val:
        raise rsgispy.RSGISPyException("presence_val and absence_val must differ.")

    counts = np.zeros(5, dtype=np.int64)
    out_path = rsgispy.tools.files.ensure_parent_dir(output_img)
    with rasterio.open(current_bin_img) as cur_src, rasterio.open(
        future_bin_img
    ) as fut_src:
        msk_src = rasterio.open(in_msk_img) if in_msk_img is not None else None
        try:
            for other in (fut_src, msk_src):
                if other is not None and not _same_grid(cur_src, other):
                    raise rsgispy.RSGISPyException(
                        f"'{other.name}' is not on the same grid as "
                        f"'{current_bin_img}'; resample it first (e.g. with "
                        "rsgispy.imageutils.warp.resample_img_to_match)."
                    )
            trans = cur_src.transform
            pxl_area = abs(trans.a * trans.e - trans.b * trans.d)
            with ExitStack() as stack:
                dst = open_output_dataset(
                    stack=stack,
                    output_img=os.fspath(out_path),
                    gdalformat=gdalformat,
                    out_rio_dtype="uint8",
                    grid=IntersectionGrid.from_dataset(cur_src),
                    count=1,
                )
                dst.nodata = _RANGE_NO_DATA
                windows = list(_iter_windows(cur_src.height, cur_src.width, block_size))
                pbar = rsgispy.get_tqdm()
                for win in pbar(windows, disable=not use_pbar, desc="Range change"):
                    cur = cur_src.read(1, window=win)
                    fut = fut_src.read(1, window=win)
                    valid = np.isin(cur, (presence_val, absence_val)) & np.isin(
                        fut, (presence_val, absence_val)
                    )
                    if msk_src is not None:
                        valid &= msk_src.read(1, window=win) == img_msk_val
                    cur_p = cur == presence_val
                    fut_p = fut == presence_val
                    out = np.full(cur.shape, _RANGE_NO_DATA, dtype=np.uint8)
                    out[valid & ~cur_p & ~fut_p] = 1
                    out[valid & cur_p & ~fut_p] = 2
                    out[valid & ~cur_p & fut_p] = 3
                    out[valid & cur_p & fut_p] = 4
                    counts += np.bincount(out.ravel(), minlength=5)[:5]
                    dst.write(out, 1, window=win)
        finally:
            if msk_src is not None:
                msk_src.close()

    set_thematic_classes(
        os.fspath(out_path),
        RANGE_CHANGE_CLASSES,
        ignore_zero=True,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )

    n_absent, n_lost, n_gained, n_stable = (int(c) for c in counts[1:])
    cur_area = (n_lost + n_stable) * pxl_area
    fut_area = (n_gained + n_stable) * pxl_area
    summary: Dict[str, float] = {
        "n_absent": float(n_absent),
        "n_lost": float(n_lost),
        "n_gained": float(n_gained),
        "n_stable": float(n_stable),
        "area_absent": n_absent * pxl_area,
        "area_lost": n_lost * pxl_area,
        "area_gained": n_gained * pxl_area,
        "area_stable": n_stable * pxl_area,
        "current_area": cur_area,
        "future_area": fut_area,
        "pct_loss": 100.0 * n_lost * pxl_area / cur_area if cur_area > 0 else np.nan,
        "pct_gain": 100.0 * n_gained * pxl_area / cur_area if cur_area > 0 else np.nan,
        "pct_net_change": (
            100.0 * (fut_area - cur_area) / cur_area if cur_area > 0 else np.nan
        ),
    }
    if out_csv is not None:
        csv_path = rsgispy.tools.files.ensure_parent_dir(out_csv)
        pd.DataFrame(
            {"metric": list(summary.keys()), "value": list(summary.values())}
        ).to_csv(csv_path, index=False)
    return summary
