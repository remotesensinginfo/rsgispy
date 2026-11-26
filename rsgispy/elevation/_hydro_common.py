"""
Shared constants, optional-dependency accessors, validation, and raster I/O
helpers for :mod:`rsgispy.elevation.hydrology`.
"""

from contextlib import ExitStack
import math
import sys
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pyproj
import rasterio

try:
    from scipy.ndimage import distance_transform_edt  # pylint: disable=import-error
    from scipy.ndimage import label as nd_label  # pylint: disable=import-error

    HAVE_SCIPY = True
except ImportError:  # pragma: no cover
    distance_transform_edt = None
    nd_label = None
    HAVE_SCIPY = False

import rsgispy
from rsgispy import RSGISPyException
from rsgispy._numba import HAS_NUMBA, numba_enabled, warn_numba_fallback
from rsgispy.elevation._hydro_kernels import (
    D8_CODES,
    apply_flat_routing_d8_kernel,
    flow_dir_d8_njit,
    flow_dir_d8_numpy,
    resolve_flats_kernel,
)
from rsgispy.imageprocess import (
    IntersectionGrid,
    build_valid_mask,
    cast_result_block,
    open_output_dataset,
    resolve_out_dtype,
    resolve_out_nodata,
)
from rsgispy.imageutils.stats import pop_img_stats


_D8_DIR_TO_OFFSET: Dict[int, Tuple[int, int]] = {
    1: (0, 1),
    2: (1, 1),
    4: (1, 0),
    8: (1, -1),
    16: (0, -1),
    32: (-1, -1),
    64: (-1, 0),
    128: (-1, 1),
}

_D8_CODE_TO_ANGLE: Dict[int, float] = {
    1: 0.0,
    128: 0.25 * math.pi,
    64: 0.5 * math.pi,
    32: 0.75 * math.pi,
    16: math.pi,
    8: 1.25 * math.pi,
    4: 1.5 * math.pi,
    2: 1.75 * math.pi,
}

_D8_CODE_TO_IDX: Dict[int, int] = {int(code): idx for idx, code in enumerate(D8_CODES)}

# Dataset metadata tag in which ``flow_accumulation`` records the units of its
# output (``'cells'`` or ``'area'``, i.e. m2), read by ``extract_streams`` and
# the terrain index functions.
FLOW_ACC_UNITS_TAG = "RSGISPY_FLOW_ACC_UNITS"
FLOW_ACC_UNITS = ("cells", "area")
FLOW_ACC_BAND_DESCS: Dict[str, str] = {
    "cells": "Flow accumulation (cells)",
    "area": "Flow accumulation (m2)",
}


def _hydro_facade() -> Any:
    """Return the loaded ``rsgispy.elevation.hydrology`` facade module if present."""
    return sys.modules.get("rsgispy.elevation.hydrology")


def _has_numba() -> bool:
    """Return active Numba availability, respecting facade monkeypatches."""
    facade = _hydro_facade()
    if facade is not None and hasattr(facade, "HAS_NUMBA"):
        return bool(facade.HAS_NUMBA)
    return bool(HAS_NUMBA)


def _have_scipy() -> bool:
    """Return active SciPy availability, respecting facade monkeypatches."""
    facade = _hydro_facade()
    if facade is not None and hasattr(facade, "HAVE_SCIPY"):
        return bool(facade.HAVE_SCIPY)
    return bool(HAVE_SCIPY)


def _get_nd_label() -> Any:
    """Return ``scipy.ndimage.label``, respecting facade monkeypatches."""
    facade = _hydro_facade()
    if facade is not None and hasattr(facade, "nd_label"):
        return facade.nd_label
    return nd_label


def _get_distance_transform_edt() -> Any:
    """Return ``scipy.ndimage.distance_transform_edt``, respecting facade monkeypatches."""
    facade = _hydro_facade()
    if facade is not None and hasattr(facade, "distance_transform_edt"):
        return facade.distance_transform_edt
    return distance_transform_edt


def _call_topo_kernel(
    kernel: Any,
    allow_pure_python: bool,
    func_name: str,
    *args: Any,
    warn: bool = True,
) -> Any:
    """Execute a graph/PQ/topological kernel respecting ``HAS_NUMBA`` policy."""
    if _has_numba():
        return kernel(*args)
    if not allow_pure_python:
        raise RSGISPyException(
            f"numba is required to run '{func_name}' efficiently. "
            "Install numba (`pip install numba`) or pass "
            "`allow_pure_python=True` to run the slow pure-Python fallback."
        )
    if warn:
        warn_numba_fallback(func_name, stacklevel=3)
    py_fn = getattr(kernel, "py_func", kernel)
    return py_fn(*args)


def _run_optional_kernel(kernel: Any, *args: Any) -> Any:
    """
    Run an :func:`rsgispy._numba.optional_njit` kernel compiled when ``numba``
    is active (respecting the ``hydrology.HAS_NUMBA`` facade flag), otherwise
    run its pure-Python ``py_func`` silently (these kernels were pure Python
    before compilation was added, so no fallback warning is emitted).
    """
    if _has_numba() and numba_enabled() and kernel.jit_func is not None:
        return kernel(*args)
    return kernel.py_func(*args)


def _warn_numpy_fallback(func_name: str) -> None:
    """Emit a ``RuntimeWarning`` when falling back to vectorised NumPy."""
    warn_numba_fallback(
        func_name,
        fallback_desc="the vectorised NumPy fallback",
        stacklevel=3,
    )


def _validate_projected_crs(
    src: rasterio.DatasetReader, img_path: str
) -> Tuple[float, float]:
    """
    Validate that ``src`` uses a projected CRS with metric units and valid
    positive pixel dimensions, returning ``(dx, dy)`` in metres.
    """
    if src.crs is None:
        raise RSGISPyException(
            f"Input raster '{img_path}' has no CRS defined. A projected CRS "
            "with linear units in metres is required."
        )
    crs = pyproj.CRS.from_user_input(src.crs)
    if crs.is_geographic or not crs.is_projected:
        raise RSGISPyException(
            f"Input raster '{img_path}' uses a non-projected/geographic CRS "
            f"({crs.name}); a projected CRS with units in metres is required."
        )
    if crs.axis_info:
        factor = crs.axis_info[0].unit_conversion_factor
        if factor is not None and not math.isclose(factor, 1.0, rel_tol=1e-4, abs_tol=1e-4):
            raise RSGISPyException(
                f"Input raster '{img_path}' CRS '{crs.name}' linear unit is "
                f"'{crs.axis_info[0].unit_name}'; units in metres are required."
            )

    dx = abs(float(src.transform.a))
    dy = abs(float(src.transform.e))
    if not (math.isfinite(dx) and dx > 0.0 and math.isfinite(dy) and dy > 0.0):
        raise RSGISPyException(
            f"Input raster '{img_path}' has invalid pixel resolution: ({dx}, {dy})."
        )
    return dx, dy


def _validate_matching_grid(
    ref_src: rasterio.DatasetReader,
    other_src: rasterio.DatasetReader,
    ref_path: str,
    other_path: str,
) -> None:
    """Verify two open datasets share identical shape, CRS, and transform."""
    if ref_src.width != other_src.width or ref_src.height != other_src.height:
        raise RSGISPyException(
            f"Raster dimensions of '{other_path}' "
            f"({other_src.height}x{other_src.width}) do not match '{ref_path}' "
            f"({ref_src.height}x{ref_src.width})."
        )
    if (ref_src.crs is None) != (other_src.crs is None):
        raise RSGISPyException(
            f"CRS mismatch between '{ref_path}' and '{other_path}'."
        )
    if ref_src.crs is not None and other_src.crs is not None:
        if pyproj.CRS.from_user_input(ref_src.crs) != pyproj.CRS.from_user_input(
            other_src.crs
        ):
            raise RSGISPyException(
                f"CRS mismatch between '{ref_path}' ({ref_src.crs}) and "
                f"'{other_path}' ({other_src.crs})."
            )
    t_ref = np.array(
        [
            ref_src.transform.a,
            ref_src.transform.b,
            ref_src.transform.c,
            ref_src.transform.d,
            ref_src.transform.e,
            ref_src.transform.f,
        ],
        dtype=np.float64,
    )
    t_oth = np.array(
        [
            other_src.transform.a,
            other_src.transform.b,
            other_src.transform.c,
            other_src.transform.d,
            other_src.transform.e,
            other_src.transform.f,
        ],
        dtype=np.float64,
    )
    if not np.allclose(t_ref, t_oth, rtol=1e-5, atol=1e-6):
        raise RSGISPyException(
            f"Geotransform mismatch between '{ref_path}' and '{other_path}'."
        )


def _resolve_out_dtype(
    datatype: Optional[int], default_rsgis_dtype: int
) -> Tuple[int, str, np.dtype]:
    """Resolve ``(rsgis_dtype, rasterio_dtype_str, numpy_dtype)``."""
    return resolve_out_dtype(datatype, default_dtype=default_rsgis_dtype)


def _resolve_out_nodata(
    out_np_dtype: np.dtype,
    no_data_val: Optional[float],
    default_nodata: Optional[float],
) -> Optional[float]:
    """Return output nodata scalar compatible with ``out_np_dtype``."""
    return resolve_out_nodata(
        out_np_dtype,
        out_no_data_val=no_data_val,
        fallback_nodata=default_nodata,
        default_float=None,
        default_int=None,
        clamp_int=True,
        unsigned_clamp="max",
        allow_round_int=True,
        allow_nonfinite_float=True,
    )


def _write_raster_array(
    output_img: str,
    arr: np.ndarray,
    ref_src: rasterio.DatasetReader,
    gdalformat: str,
    out_rio_dtype: str,
    out_np_dtype: np.dtype,
    out_no_data_val: Optional[float],
    calc_stats: bool,
    use_pbar: bool,
    *,
    tags: Optional[Mapping[str, str]] = None,
    band_descs: Optional[Sequence[str]] = None,
) -> None:
    """
    Write a 2D or 3D array to disk using ``rsgispy`` conventions and stats,
    optionally with dataset metadata ``tags`` and band descriptions.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    grid = IntersectionGrid.from_dataset(ref_src)
    if arr.ndim == 2:
        count = 1
        out_block = cast_result_block(arr, ref_src.height, ref_src.width, out_np_dtype)
        write_data = out_block[np.newaxis, :, :]
    else:
        count = int(arr.shape[0])
        write_data = np.empty(
            (count, ref_src.height, ref_src.width), dtype=out_np_dtype
        )
        for b_idx in range(count):
            write_data[b_idx] = cast_result_block(
                arr[b_idx], ref_src.height, ref_src.width, out_np_dtype
            )

    with ExitStack() as stack:
        dst = open_output_dataset(
            stack=stack,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype=out_rio_dtype,
            grid=grid,
            count=count,
        )
        if out_no_data_val is not None:
            dst.nodata = out_no_data_val
        dst.write(write_data)
        if tags:
            dst.update_tags(**{str(key): str(val) for key, val in tags.items()})
        if band_descs:
            for b_idx, desc in enumerate(band_descs, start=1):
                dst.set_band_description(b_idx, desc)

    if calc_stats:
        pop_img_stats(
            input_img=output_img,
            no_data_val=out_no_data_val,
            calc_pyramids=True,
            ignore_cog=gdalformat.upper().strip() == "COG",
            use_pbar=use_pbar,
        )


def _read_flow_acc_units_tag(
    src: rasterio.DatasetReader, img_path: str
) -> Optional[str]:
    """
    Return the units (``'cells'`` or ``'area'``) recorded in the
    ``RSGISPY_FLOW_ACC_UNITS`` tag of a flow accumulation raster, or ``None``
    if the raster has no tag (e.g. written before the tag was introduced).
    """
    tag_val = src.tags().get(FLOW_ACC_UNITS_TAG)
    if tag_val is None:
        return None
    units = str(tag_val).strip().lower()
    if units not in FLOW_ACC_UNITS:
        raise RSGISPyException(
            f"Unrecognised {FLOW_ACC_UNITS_TAG} tag value '{tag_val}' in "
            f"'{img_path}'. Expected 'cells' or 'area'."
        )
    return units


def _resolve_flow_acc_units(
    src: rasterio.DatasetReader, img_path: str, acc_units: str
) -> str:
    """
    Resolve the units of a flow accumulation raster from ``acc_units``
    (``'auto'``, ``'cells'`` or ``'area'``) and its ``RSGISPY_FLOW_ACC_UNITS``
    tag. ``'auto'`` uses the tag, or ``'cells'`` when there is no tag; an
    explicit value that contradicts the tag raises an exception.
    """
    acc_norm = str(acc_units).strip().lower()
    if acc_norm not in ("auto",) + FLOW_ACC_UNITS:
        raise RSGISPyException(
            f"Invalid acc_units '{acc_units}'. Must be 'auto', 'cells' or 'area'."
        )
    tag_units = _read_flow_acc_units_tag(src, img_path)
    if acc_norm == "auto":
        return tag_units if tag_units is not None else "cells"
    if tag_units is not None and tag_units != acc_norm:
        raise RSGISPyException(
            f"acc_units='{acc_norm}' contradicts the {FLOW_ACC_UNITS_TAG}="
            f"{tag_units} tag of '{img_path}' "
            f"({FLOW_ACC_BAND_DESCS[tag_units].lower()}). "
            f"Use acc_units='auto' or '{tag_units}'."
        )
    return acc_norm


def _reject_flow_acc_input(
    src: rasterio.DatasetReader, img_path: str, func_name: str
) -> None:
    """
    Raise if ``img_path`` is tagged as a flow accumulation: the terrain index
    functions need a specific catchment area (m2/m) instead.
    """
    tag_units = _read_flow_acc_units_tag(src, img_path)
    if tag_units is not None:
        raise RSGISPyException(
            f"'{img_path}' is a flow accumulation ({FLOW_ACC_UNITS_TAG}="
            f"{tag_units}), but {func_name} needs a specific catchment area "
            "(m2/m). Divide a flow accumulation with out_units='area' by the "
            "pixel size (the contour width) first, e.g. with "
            "rsgispy.imagecalc.maths.band_maths."
        )


def _validate_slope_units(slope_units: str) -> int:
    """Validate and map ``slope_units`` to integer code ``(0=deg, 1=rad, 2=pct)``."""
    norm = slope_units.strip().lower()
    if norm == "degrees":
        return 0
    if norm == "radians":
        return 1
    if norm == "percent":
        return 2
    raise RSGISPyException(
        f"Invalid slope_units '{slope_units}'. Must be 'degrees', 'radians', or 'percent'."
    )


def _read_d8_grid(
    src: rasterio.DatasetReader, no_data_val: Optional[float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Read a D8 flow direction raster and validate it is single-band integer."""
    if src.count != 1:
        raise RSGISPyException(
            f"D8 flow direction raster must have 1 band, found {src.count}."
        )
    dtype = np.dtype(src.dtypes[0])
    if not np.issubdtype(dtype, np.integer):
        raise RSGISPyException(
            f"Expected integer D8 flow direction raster, got dtype '{dtype}'."
        )
    eff_nd = no_data_val if no_data_val is not None else src.nodata
    if eff_nd is None:
        eff_nd = 255.0
    arr = src.read(1)
    valid = build_valid_mask(arr, eff_nd)
    return arr.astype(np.uint8, copy=False), valid


def _detect_flow_dir_mode(src: rasterio.DatasetReader) -> str:
    """Auto-detect ``'d8'``, ``'dinf'``, or ``'mfd'`` from raster band count/dtype."""
    if src.count == 8:
        return "mfd"
    if src.count != 1:
        raise RSGISPyException(
            f"Unsupported flow direction band count ({src.count}). "
            "Expected 1 band (D8 uint8 or D-Inf float32) or 8 bands (MFD float32)."
        )
    dtype = np.dtype(src.dtypes[0])
    if np.issubdtype(dtype, np.integer):
        return "d8"
    if np.issubdtype(dtype, np.floating):
        return "dinf"
    raise RSGISPyException(f"Unsupported flow direction dtype '{dtype}'.")


def _d8_flat_resolve_in_memory(
    dem: np.ndarray,
    valid: np.ndarray,
    dx: float,
    dy: float,
    allow_pure_python: bool,
    func_name: str,
    *,
    warn: bool = True,
) -> np.ndarray:
    """Compute D8 directions and route any flats via Barnes double-gradient."""
    if _has_numba():
        fdir = flow_dir_d8_njit(dem, valid, dx, dy)
    else:
        fdir = flow_dir_d8_numpy(dem, valid, dx, dy)

    if np.any(valid & (fdir == 0)):
        g_grid = _call_topo_kernel(
            resolve_flats_kernel,
            allow_pure_python,
            func_name,
            dem,
            valid,
            warn=warn,
        )
        _call_topo_kernel(
            apply_flat_routing_d8_kernel,
            allow_pure_python,
            func_name,
            dem,
            valid,
            fdir,
            g_grid,
            dx,
            dy,
            warn=False,
        )
    return fdir


def _run_progress_steps(
    desc: str, n_steps: int, use_pbar: bool
) -> Any:
    """Create a ``tqdm`` progress bar instance for stage tracking."""
    tqdm_cls = rsgispy.get_tqdm()
    return tqdm_cls(total=n_steps, desc=desc, unit="step", disable=not use_pbar)
