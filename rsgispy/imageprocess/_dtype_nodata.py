"""
Block-size configuration, GDAL creation options, CRS parsing, data-type
resolution, no-data handling, and result-block casting for
:mod:`rsgispy.imageprocess`.
"""

import math
import os
import sys
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import rasterio

import rsgispy

DEFAULT_BLOCK_SIZE = 512  # Match COG
ENV_VAR_BLOCK_SIZE = "RSGISPY_BLOCK_SIZE"
ENV_VAR_RSGISLIB_BLOCK_SIZE = "RSGISLIB_BLOCK_SIZE"
_INT_DTYPES = {"uint8", "int8", "uint16", "int16", "uint32", "int32"}

#: Built-in GDAL creation options used when neither
#: ``RSGISPY_IMG_CRT_OPTS_<FORMAT>`` nor ``RSGISLIB_IMG_CRT_OPTS_<FORMAT>`` is
#: set. Keys are upper-case GDAL driver names; formats without an entry get no
#: creation options. GeoTIFF outputs are tiled (GDAL's default 256 x 256 tiles)
#: and DEFLATE-compressed, the same options that
#: :func:`rsgispy.imageutils.envopts.set_env_vars_deflate_gtiff_outs` sets.
DEFAULT_IMG_CREATION_OPTS: Dict[str, Dict[str, str]] = {
    "GTIFF": {
        "TILED": "YES",
        "COMPRESS": "DEFLATE",
        "INTERLEAVE": "BAND",
        "BIGTIFF": "IF_SAFER",
    },
}
ENV_VAR_IMG_CRT_OPTS_PREFIX = "RSGISPY_IMG_CRT_OPTS_"
ENV_VAR_RSGISLIB_IMG_CRT_OPTS_PREFIX = "RSGISLIB_IMG_CRT_OPTS_"


def get_default_block_size() -> int:
    """
    Return the default processing block size in pixels.

    Checks the ``RSGISPY_BLOCK_SIZE`` (and fallback ``RSGISLIB_BLOCK_SIZE``)
    environment variables at runtime, falling back to :data:`DEFAULT_BLOCK_SIZE`.

    :return: Positive integer block size in pixels.

    .. code:: python

        import rsgispy.imageprocess

        blk_size = rsgispy.imageprocess.get_default_block_size()
    """
    raw_val = os.environ.get(ENV_VAR_BLOCK_SIZE) or os.environ.get(
        ENV_VAR_RSGISLIB_BLOCK_SIZE
    )
    if raw_val is not None and raw_val.strip() != "":
        try:
            block_size = int(raw_val.strip())
        except ValueError as err:
            raise rsgispy.RSGISPyException(
                f"Invalid block size in environment variable: '{raw_val}'. "
                "Must be a positive integer."
            ) from err
        if block_size <= 0:
            raise rsgispy.RSGISPyException(
                f"Invalid block size in environment variable: '{raw_val}'. "
                "Must be a positive integer."
            )
        return block_size

    facade = sys.modules.get("rsgispy.imageprocess")
    default_blk = (
        getattr(facade, "DEFAULT_BLOCK_SIZE", DEFAULT_BLOCK_SIZE)
        if facade is not None
        else DEFAULT_BLOCK_SIZE
    )
    if (
        not isinstance(default_blk, int)
        or isinstance(default_blk, bool)
        or default_blk <= 0
    ):
        raise rsgispy.RSGISPyException(
            f"DEFAULT_BLOCK_SIZE must be a positive integer, got {default_blk!r}."
        )
    return default_blk


def set_default_block_size(block_size: int) -> None:
    """
    Override the default processing block size at runtime by setting the
    ``RSGISPY_BLOCK_SIZE`` environment variable.

    :param block_size: Positive integer block size in pixels.

    .. code:: python

        import rsgispy.imageprocess

        rsgispy.imageprocess.set_default_block_size(256)
    """
    if (
        isinstance(block_size, bool)
        or not isinstance(block_size, (int, np.integer))
        or int(block_size) <= 0
    ):
        raise rsgispy.RSGISPyException(
            f"block_size must be a positive integer, got {block_size!r}."
        )
    os.environ[ENV_VAR_BLOCK_SIZE] = str(int(block_size))


def reset_default_block_size() -> None:
    """
    Clear any runtime block size override from the ``RSGISPY_BLOCK_SIZE`` and
    ``RSGISLIB_BLOCK_SIZE`` environment variables so :data:`DEFAULT_BLOCK_SIZE`
    is used again.

    .. code:: python

        import rsgispy.imageprocess

        rsgispy.imageprocess.reset_default_block_size()
    """
    os.environ.pop(ENV_VAR_BLOCK_SIZE, None)
    os.environ.pop(ENV_VAR_RSGISLIB_BLOCK_SIZE, None)


def _parse_img_creation_opts_str(raw_opts: str) -> Dict[str, str]:
    """Parse colon-separated ``KEY=VALUE`` pairs into a dictionary."""
    opts_dict: Dict[str, str] = {}
    for item in raw_opts.split(":"):
        item = item.strip()
        if "=" in item:
            key, val = item.split("=", 1)
            opts_dict[key.strip()] = val.strip()
    return opts_dict


def get_gdal_img_creation_opts(gdalformat: str) -> Dict[str, str]:
    """
    Resolve the GDAL creation options used for new output images of the GDAL
    driver ``gdalformat``.

    The options are taken from the first of these that applies:

    1. the ``RSGISPY_IMG_CRT_OPTS_<GDAL_FORMAT>`` environment variable, if it
       is set;
    2. otherwise the ``RSGISLIB_IMG_CRT_OPTS_<GDAL_FORMAT>`` environment
       variable, if it is set;
    3. otherwise the built-in default in
       :data:`rsgispy.imageprocess.DEFAULT_IMG_CREATION_OPTS`. For GeoTIFF
       (``'GTIFF'``) this is ``TILED=YES``, ``COMPRESS=DEFLATE``,
       ``INTERLEAVE=BAND`` and ``BIGTIFF=IF_SAFER``; other formats (including
       ``'COG'`` and ``'KEA'``) have no default and get ``{}``.

    Environment variables hold colon-separated ``KEY=VALUE`` pairs (e.g.
    ``TILED=YES:COMPRESS=LZW``). A variable that is set replaces the default
    rather than being merged with it, and a variable set to an empty string
    means "no creation options", which gives a plain striped, uncompressed
    GeoTIFF. Explicit ``creation_opts`` passed to an rsgispy function are
    merged on top of the resolved options. The environment is read on each
    call, and rsgispy never sets these variables itself; use
    :mod:`rsgispy.imageutils.envopts` to change them at runtime.

    :param gdalformat: GDAL format string (e.g., 'GTIFF', 'COG', 'KEA').
    :return: New dictionary of creation option key-value strings.

    .. code:: python

        import rsgispy.imageprocess

        # {'TILED': 'YES', 'COMPRESS': 'DEFLATE', 'INTERLEAVE': 'BAND',
        #  'BIGTIFF': 'IF_SAFER'} unless an environment variable is set.
        opts = rsgispy.imageprocess.get_gdal_img_creation_opts("GTIFF")
    """
    fmt_upper = gdalformat.upper().strip()
    for env_prefix in (
        ENV_VAR_IMG_CRT_OPTS_PREFIX,
        ENV_VAR_RSGISLIB_IMG_CRT_OPTS_PREFIX,
    ):
        raw_opts = os.environ.get(f"{env_prefix}{fmt_upper}")
        if raw_opts is not None:
            return _parse_img_creation_opts_str(raw_opts)

    facade = sys.modules.get("rsgispy.imageprocess")
    defaults = (
        getattr(facade, "DEFAULT_IMG_CREATION_OPTS", DEFAULT_IMG_CREATION_OPTS)
        if facade is not None
        else DEFAULT_IMG_CREATION_OPTS
    )
    for def_fmt, def_opts in defaults.items():
        if str(def_fmt).upper().strip() == fmt_upper:
            return {str(key): str(val) for key, val in def_opts.items()}
    return {}


def rsgis_dtype_from_np_dtype(
    dtype_obj: Any, fallback: int = rsgispy.TYPE_32FLOAT
) -> int:
    """Resolve an ``rsgispy.TYPE_*`` constant from a NumPy/rasterio dtype."""
    return rsgispy.rsgis_dtype_from_np_dtype(dtype_obj, fallback=fallback)


def resolve_out_dtype(
    datatype: Optional[int],
    default_dtype: Optional[int] = rsgispy.TYPE_32FLOAT,
    ref_np_dtype: Optional[Any] = None,
    require_int: bool = False,
) -> Tuple[int, str, np.dtype]:
    """
    Resolve an output ``(rsgis_dtype, rasterio_dtype_str, numpy_dtype)`` triplet.

    :param datatype: Explicit ``rsgispy.TYPE_*`` constant, or ``None``.
    :param default_dtype: Fallback ``rsgispy.TYPE_*`` constant when ``datatype``
                          and ``ref_np_dtype`` are ``None``
                          (Default = ``rsgispy.TYPE_32FLOAT``).
    :param ref_np_dtype: Optional reference dataset/array dtype inspected when
                         ``datatype is None``.
    :param require_int: If ``True``, raise :class:`rsgispy.RSGISPyException`
                        when the resolved dtype is not an integer type.
    :return: Tuple ``(rsgis_dtype, out_rio_dtype, out_np_dtype)``.
    """
    if datatype is not None:
        eff_dt = int(datatype)
    elif ref_np_dtype is not None:
        fb = rsgispy.TYPE_32FLOAT if default_dtype is None else int(default_dtype)
        eff_dt = rsgis_dtype_from_np_dtype(ref_np_dtype, fallback=fb)
    elif default_dtype is not None:
        eff_dt = int(default_dtype)
    else:
        raise rsgispy.RSGISPyException(
            "An explicit datatype, reference dtype, or default_dtype is required."
        )
    rio_dtype = rsgispy.get_rasterio_datatype(eff_dt)
    np_dtype = np.dtype(rsgispy.get_numpy_datatype(eff_dt))
    if require_int and not np.issubdtype(np_dtype, np.integer):
        raise rsgispy.RSGISPyException(
            f"Output datatype must be an integer type, got '{rio_dtype}'."
        )
    return eff_dt, rio_dtype, np_dtype


def fits_dtype_range(
    value: float,
    out_np_dtype: np.dtype,
    allow_round_int: bool = False,
) -> bool:
    """
    Return ``True`` if finite scalar ``value`` is representable in ``out_np_dtype``.
    """
    out_np_dtype = np.dtype(out_np_dtype)
    f_val = float(value)
    if not math.isfinite(f_val):
        return False
    if np.issubdtype(out_np_dtype, np.floating):
        info = np.finfo(out_np_dtype)
        return float(info.min) <= f_val <= float(info.max)  # pylint: disable=no-member
    if np.issubdtype(out_np_dtype, np.integer):
        if not allow_round_int and not f_val.is_integer():
            return False
        info_i = np.iinfo(out_np_dtype)
        i_val = int(round(f_val)) if allow_round_int else int(f_val)
        return int(info_i.min) <= i_val <= int(info_i.max)
    return False


def resolve_out_nodata(
    out_np_dtype: np.dtype,
    out_no_data_val: Optional[float] = None,
    fallback_nodata: Optional[float] = None,
    *,
    default_float: Optional[float] = -9999.0,
    default_int: Optional[float] = 0.0,
    clamp_int: bool = False,
    unsigned_clamp: str = "max",
    strict_explicit: bool = False,
    allow_round_int: bool = False,
    allow_nonfinite_float: bool = False,
) -> Optional[float]:
    """
    Resolve an output no-data value compatible with ``out_np_dtype``.

    :param out_np_dtype: Target NumPy dtype of the output raster.
    :param out_no_data_val: Explicit user-requested output no-data value.
    :param fallback_nodata: Secondary candidate (e.g. input image header nodata).
    :param default_float: Fallback for floating-point outputs when candidate is
                          ``None`` or out of range (Default = ``-9999.0``).
    :param default_int: Fallback for integer outputs when candidate is ``None``
                        or out of range (Default = ``0.0``).
    :param clamp_int: If ``True``, clamp out-of-range integer candidates to the
                      target integer dtype's bounds.
    :param unsigned_clamp: When ``clamp_int=True`` and ``out_np_dtype`` is an
                           unsigned integer type, select ``'max'`` (``info.max``)
                           or ``'zero'`` (``0.0``).
    :param strict_explicit: If ``True`` and ``out_no_data_val`` is explicitly
                            provided, raise :class:`rsgispy.RSGISPyException` if
                            it is not representable in ``out_np_dtype``.
    :param allow_round_int: If ``True``, round floating candidates to the nearest
                            integer when checking integer range fit.
    :param allow_nonfinite_float: If ``True``, preserve ``NaN``/``Inf`` candidates
                                  for floating-point outputs.
    :return: Resolved output no-data scalar, or ``None``.
    """
    # pylint: disable=too-many-arguments,too-many-locals
    # pylint: disable=too-many-branches,too-many-return-statements
    out_np_dtype = np.dtype(out_np_dtype)
    is_int = np.issubdtype(out_np_dtype, np.integer)
    is_flt = np.issubdtype(out_np_dtype, np.floating)

    if out_no_data_val is not None:
        f_exp = float(out_no_data_val)
        if strict_explicit and is_int:
            if not math.isfinite(f_exp):
                raise rsgispy.RSGISPyException(
                    "out_no_data_val must be finite for integer output datatypes."
                )
            info_i = np.iinfo(out_np_dtype)
            if f_exp < info_i.min or f_exp > info_i.max:
                raise rsgispy.RSGISPyException(
                    f"out_no_data_val ({out_no_data_val}) is outside the valid range "
                    f"[{info_i.min}, {info_i.max}] for dtype '{out_np_dtype.name}'."
                )
        if not clamp_int and not strict_explicit:
            return f_exp
        cand: Optional[float] = f_exp
    else:
        cand = float(fallback_nodata) if fallback_nodata is not None else None

    if cand is None:
        cand = default_int if is_int else default_float
        if cand is None:
            return None

    f_cand = float(cand)
    if not math.isfinite(f_cand):
        if is_flt and allow_nonfinite_float:
            return f_cand
        return float(default_int) if (is_int and default_int is not None) else (
            float(default_float) if (is_flt and default_float is not None) else None
        )

    if fits_dtype_range(f_cand, out_np_dtype, allow_round_int=allow_round_int):
        if is_int and allow_round_int:
            return float(int(round(f_cand)))
        return f_cand

    if is_int and clamp_int:
        info_i = np.iinfo(out_np_dtype)
        if np.issubdtype(out_np_dtype, np.unsignedinteger):
            return 0.0 if unsigned_clamp == "zero" else float(info_i.max)
        return float(info_i.min)

    if is_flt:
        return float(default_float) if default_float is not None else None
    if is_int:
        return float(default_int) if default_int is not None else None
    return f_cand


def resolve_nodata_and_fill(
    eff_nd: Optional[float],
    out_np_dtype: np.dtype,
) -> Tuple[Optional[float], float]:
    """
    Return ``(dst_nodata_header, fill_value)`` compatible with ``out_np_dtype``.
    """
    out_np_dtype = np.dtype(out_np_dtype)
    if np.issubdtype(out_np_dtype, np.floating):
        if eff_nd is not None:
            nd_val = float(eff_nd)
            return nd_val, nd_val
        return None, float("nan")

    if np.issubdtype(out_np_dtype, np.integer):
        if eff_nd is not None and fits_dtype_range(
            float(eff_nd), out_np_dtype, allow_round_int=True
        ):
            nd_val = float(int(round(float(eff_nd))))
            return nd_val, nd_val
        if eff_nd is not None:
            return 0.0, 0.0
        return None, 0.0

    return eff_nd, (float(eff_nd) if eff_nd is not None else 0.0)


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


def cast_result_block(
    res: Union[np.ndarray, float, int],
    win_h: int,
    win_w: int,
    out_np_dtype: np.dtype,
    clamp_int: bool = False,
) -> np.ndarray:
    """
    Convert a block calculation result to an array of the target NumPy dtype.

    :param res: Result array or scalar from block calculation.
    :param win_h: Height of the current window in pixels.
    :param win_w: Width of the current window in pixels.
    :param out_np_dtype: Target NumPy dtype for the output raster.
    :param clamp_int: If ``True`` and ``out_np_dtype`` is an integer dtype,
                      clamp values to the dtype's valid range before casting
                      (Default = ``False``).
    :return: NumPy array of shape ``(win_h, win_w)`` or ``(bands, win_h, win_w)``.

    .. code:: python

        import numpy as np
        import rsgispy.imageprocess

        arr = np.ones((256, 256), dtype=np.float64)
        out = rsgispy.imageprocess.cast_result_block(
            arr, 256, 256, np.dtype(np.float32)
        )
    """
    out_np_dtype = np.dtype(out_np_dtype)
    if np.ndim(res) == 0:
        arr = np.full((win_h, win_w), res, dtype=np.float64)
    else:
        arr = np.asarray(res)

    with np.errstate(invalid="ignore"):
        if np.issubdtype(out_np_dtype, np.integer):
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            if clamp_int:
                info = np.iinfo(out_np_dtype)
                arr = np.clip(arr, info.min, info.max)
        return arr.astype(out_np_dtype, copy=False)
