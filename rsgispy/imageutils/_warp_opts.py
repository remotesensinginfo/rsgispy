"""
Internal configuration parser and spatial grid/cutline helpers for
:func:`rsgispy.imageutils.warp.gdal_warp`.
"""

from dataclasses import dataclass, field
import json
import math
import os
import shlex
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import geopandas
import numpy as np
import rasterio
import rasterio.crs
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.transform import Affine
from rasterio.warp import calculate_default_transform, transform_bounds
from rasterio.windows import transform as window_transform
from scipy import ndimage
import shapely.geometry
import shapely.ops
import shapely.wkt

import rsgispy
from rsgispy.imageprocess import iter_dataset_windows, validate_band_index
from rsgispy.imageutils._helpers import normalise_creation_options, parse_epsg_crs
from rsgispy.imageutils._translate import (
    _DEFAULT_GTIFF_CO,
    _consume_args,
    _is_cli_flag,
    _parse_crs_input,
    _parse_datatype_arg,
    _resolve_resampling,
    _split_kv,
)


@dataclass
class WarpGrid:
    """Resolved source/destination georeferencing and target grid."""

    src_crs_list: List[Optional[rasterio.crs.CRS]]
    src_georef_list: List[Dict[str, Any]]
    dst_crs: Optional[rasterio.crs.CRS]
    width: int
    height: int
    transform: Affine
    cutline_geoms: Optional[List[Any]]


@dataclass
class WarpConfig:
    """Normalised configuration container for ``gdal_warp``."""

    # pylint: disable=too-many-instance-attributes
    gdalformat: str = "GTIFF"
    in_format: Optional[str] = None
    datatype: Optional[int] = None
    working_datatype: Optional[str] = None
    creation_opts: Dict[str, Any] = field(default_factory=dict)
    open_opts: Dict[str, Any] = field(default_factory=dict)
    dst_open_opts: Dict[str, Any] = field(default_factory=dict)
    overwrite: bool = True
    src_srs: Optional[Any] = None
    dst_srs: Optional[Any] = None
    src_coord_epoch: Optional[float] = None
    dst_coord_epoch: Optional[float] = None
    coord_transform: Optional[str] = None
    resampling: Resampling = Resampling.nearest
    target_res: Optional[Tuple[float, float]] = None
    target_res_square: bool = False
    target_size: Optional[Tuple[int, int]] = None
    target_extent: Optional[Tuple[float, float, float, float]] = None
    target_extent_srs: Optional[Any] = None
    tap: bool = False
    src_nodata: Optional[Union[str, Tuple[float, ...]]] = None
    has_src_nodata: bool = False
    dst_nodata: Optional[Union[str, Tuple[float, ...]]] = None
    has_dst_nodata: bool = False
    src_bands: Optional[List[int]] = None
    dst_bands: Optional[List[int]] = None
    src_alpha: Optional[bool] = None
    dst_alpha: bool = False
    ovr: Optional[str] = None
    err_threshold: float = 0.125
    warp_mem_limit: int = 0
    num_threads: int = 1
    tps: bool = False
    rpc: bool = False
    geoloc: bool = False
    order: Optional[int] = None
    refine_gcps: Optional[Tuple[float, Optional[int]]] = None
    warp_options: Dict[str, str] = field(default_factory=dict)
    transformer_options: Dict[str, str] = field(default_factory=dict)
    vshift: Optional[bool] = None
    cutline: Optional[Any] = None
    cutline_srs: Optional[Any] = None
    cutline_layer: Optional[str] = None
    cutline_where: Optional[str] = None
    cutline_sql: Optional[str] = None
    cutline_blend: float = 0.0
    crop_to_cutline: bool = False
    no_metadata: bool = False
    meta_conflict_value: Optional[str] = None
    set_ci: bool = False
    use_pbar: bool = False


def _parse_nodata_values(
    val: Any,
) -> Union[str, Tuple[float, ...]]:
    """Parse ``-srcnodata`` / ``-dstnodata`` (``'none'``, float, or sequence)."""
    if isinstance(val, str):
        stripped = val.strip()
        if stripped.lower() == "none":
            return "none"
        parts = stripped.split()
        if not parts:
            raise rsgispy.RSGISPyException("Nodata value string cannot be empty.")
        return tuple(float(p) for p in parts)
    if isinstance(val, (int, float, np.number)) and not isinstance(val, bool):
        return (float(val),)
    if isinstance(val, Sequence):
        return tuple(float(v) for v in val)
    raise rsgispy.RSGISPyException(f"Unsupported nodata specification: {val!r}.")


def _is_float_token(token: str) -> bool:
    """Return True if ``token`` parses cleanly as a finite float."""
    if _is_cli_flag(token):
        return False
    try:
        return math.isfinite(float(token))
    except ValueError:
        return False


def _parse_warp_cli_tokens(tokens: Sequence[str], cfg: WarpConfig) -> None:
    """Populate ``cfg`` from GDAL ``gdalwarp`` CLI tokens."""
    # pylint: disable=too-many-branches,too-many-statements,too-many-locals
    idx = 0
    while idx < len(tokens):
        tok = tokens[idx]
        idx += 1

        if "=" in tok and not _is_cli_flag(tok):
            key, val = _split_kv(tok, "KEY=VALUE")
            cfg.creation_opts[key] = val
            continue

        if tok == "-of":
            cfg.gdalformat = _consume_args(tokens, idx, 1, tok)[0]
            idx += 1
        elif tok == "-if":
            cfg.in_format = _consume_args(tokens, idx, 1, tok)[0]
            idx += 1
        elif tok == "-ot":
            cfg.datatype = _parse_datatype_arg(
                _consume_args(tokens, idx, 1, tok)[0]
            )
            idx += 1
        elif tok == "-wt":
            dt_str = _consume_args(tokens, idx, 1, tok)[0]
            _parse_datatype_arg(dt_str)
            cfg.working_datatype = dt_str
            idx += 1
        elif tok == "-overwrite":
            cfg.overwrite = True
        elif tok == "-s_srs":
            cfg.src_srs = _consume_args(tokens, idx, 1, tok)[0]
            idx += 1
        elif tok == "-t_srs":
            cfg.dst_srs = _consume_args(tokens, idx, 1, tok)[0]
            idx += 1
        elif tok == "-s_coord_epoch":
            cfg.src_coord_epoch = float(_consume_args(tokens, idx, 1, tok)[0])
            idx += 1
        elif tok == "-t_coord_epoch":
            cfg.dst_coord_epoch = float(_consume_args(tokens, idx, 1, tok)[0])
            idx += 1
        elif tok == "-ct":
            cfg.coord_transform = _consume_args(tokens, idx, 1, tok)[0]
            idx += 1
        elif tok == "-r":
            cfg.resampling = _resolve_resampling(
                _consume_args(tokens, idx, 1, tok)[0]
            )
            idx += 1
        elif tok == "-tr":
            first = _consume_args(tokens, idx, 1, tok)[0]
            idx += 1
            if first.strip().lower() == "square":
                cfg.target_res_square = True
                cfg.target_res = None
            else:
                second = _consume_args(tokens, idx, 1, tok)[0]
                idx += 1
                cfg.target_res = (float(first), float(second))
                cfg.target_res_square = False
        elif tok == "-ts":
            vals = _consume_args(tokens, idx, 2, tok)
            cfg.target_size = (int(float(vals[0])), int(float(vals[1])))
            idx += 2
        elif tok == "-te":
            vals = _consume_args(tokens, idx, 4, tok)
            cfg.target_extent = (
                float(vals[0]),
                float(vals[1]),
                float(vals[2]),
                float(vals[3]),
            )
            idx += 4
        elif tok == "-te_srs":
            cfg.target_extent_srs = _consume_args(tokens, idx, 1, tok)[0]
            idx += 1
        elif tok == "-tap":
            cfg.tap = True
        elif tok == "-srcnodata":
            cfg.has_src_nodata = True
            cfg.src_nodata = _parse_nodata_values(
                _consume_args(tokens, idx, 1, tok)[0]
            )
            idx += 1
        elif tok == "-dstnodata":
            cfg.has_dst_nodata = True
            cfg.dst_nodata = _parse_nodata_values(
                _consume_args(tokens, idx, 1, tok)[0]
            )
            idx += 1
        elif tok == "-srcband":
            if cfg.src_bands is None:
                cfg.src_bands = []
            cfg.src_bands.append(int(_consume_args(tokens, idx, 1, tok)[0]))
            idx += 1
        elif tok == "-dstband":
            if cfg.dst_bands is None:
                cfg.dst_bands = []
            cfg.dst_bands.append(int(_consume_args(tokens, idx, 1, tok)[0]))
            idx += 1
        elif tok == "-srcalpha":
            cfg.src_alpha = True
        elif tok == "-nosrcalpha":
            cfg.src_alpha = False
        elif tok == "-dstalpha":
            cfg.dst_alpha = True
        elif tok == "-ovr":
            cfg.ovr = _consume_args(tokens, idx, 1, tok)[0]
            idx += 1
        elif tok == "-et":
            cfg.err_threshold = float(_consume_args(tokens, idx, 1, tok)[0])
            idx += 1
        elif tok == "-wm":
            cfg.warp_mem_limit = int(float(_consume_args(tokens, idx, 1, tok)[0]))
            idx += 1
        elif tok == "-multi":
            cfg.num_threads = max(cfg.num_threads, os.cpu_count() or 1)
        elif tok == "-tps":
            cfg.tps = True
        elif tok == "-rpc":
            cfg.rpc = True
        elif tok == "-geoloc":
            cfg.geoloc = True
        elif tok == "-order":
            cfg.order = int(_consume_args(tokens, idx, 1, tok)[0])
            idx += 1
        elif tok == "-refine_gcps":
            tol = float(_consume_args(tokens, idx, 1, tok)[0])
            idx += 1
            min_g: Optional[int] = None
            if idx < len(tokens) and _is_float_token(tokens[idx]):
                min_g = int(float(tokens[idx]))
                idx += 1
            cfg.refine_gcps = (tol, min_g)
        elif tok == "-wo":
            key, val = _split_kv(_consume_args(tokens, idx, 1, tok)[0], tok)
            cfg.warp_options[key] = val
            idx += 1
        elif tok == "-to":
            key, val = _split_kv(_consume_args(tokens, idx, 1, tok)[0], tok)
            cfg.transformer_options[key] = val
            idx += 1
        elif tok == "-vshift":
            cfg.vshift = True
        elif tok == "-novshiftgrid":
            cfg.vshift = False
        elif tok == "-cutline":
            cfg.cutline = _consume_args(tokens, idx, 1, tok)[0]
            idx += 1
        elif tok == "-cutline_srs":
            cfg.cutline_srs = _consume_args(tokens, idx, 1, tok)[0]
            idx += 1
        elif tok == "-cl":
            cfg.cutline_layer = _consume_args(tokens, idx, 1, tok)[0]
            idx += 1
        elif tok == "-cwhere":
            cfg.cutline_where = _consume_args(tokens, idx, 1, tok)[0]
            idx += 1
        elif tok == "-csql":
            cfg.cutline_sql = _consume_args(tokens, idx, 1, tok)[0]
            idx += 1
        elif tok == "-cblend":
            cfg.cutline_blend = float(_consume_args(tokens, idx, 1, tok)[0])
            idx += 1
        elif tok == "-crop_to_cutline":
            cfg.crop_to_cutline = True
        elif tok == "-nomd":
            cfg.no_metadata = True
        elif tok == "-cvmd":
            cfg.meta_conflict_value = _consume_args(tokens, idx, 1, tok)[0]
            idx += 1
        elif tok == "-setci":
            cfg.set_ci = True
        elif tok == "-co":
            key, val = _split_kv(_consume_args(tokens, idx, 1, tok)[0], tok)
            cfg.creation_opts[key] = val
            idx += 1
        elif tok == "-oo":
            key, val = _split_kv(_consume_args(tokens, idx, 1, tok)[0], tok)
            cfg.open_opts[key] = val
            idx += 1
        elif tok == "-doo":
            key, val = _split_kv(_consume_args(tokens, idx, 1, tok)[0], tok)
            cfg.dst_open_opts[key] = val
            idx += 1
        elif tok in ("--quiet", "-q"):
            cfg.use_pbar = False
        elif _is_cli_flag(tok):
            raise rsgispy.RSGISPyException(
                f"Unrecognised gdal_warp option '{tok}'."
            )


def _apply_warp_kwargs_part1(cfg: WarpConfig, kwargs: Mapping[str, Any]) -> None:
    """Apply format, CRS, resolution, extent, nodata, and band keyword arguments."""
    # pylint: disable=too-many-branches,too-many-statements
    if kwargs.get("in_format") is not None:
        cfg.in_format = str(kwargs["in_format"])
    if kwargs.get("datatype") is not None:
        cfg.datatype = _parse_datatype_arg(kwargs["datatype"])
    if kwargs.get("working_datatype") is not None:
        wt_val = kwargs["working_datatype"]
        if isinstance(wt_val, str):
            _parse_datatype_arg(wt_val)
            cfg.working_datatype = wt_val
        else:
            dt_code = _parse_datatype_arg(wt_val)
            cfg.working_datatype = rsgispy.get_rasterio_datatype(dt_code)
    if kwargs.get("open_options"):
        cfg.open_opts.update(
            normalise_creation_options(kwargs["open_options"]) or {}
        )
    if kwargs.get("dst_open_options"):
        cfg.dst_open_opts.update(
            normalise_creation_options(kwargs["dst_open_options"]) or {}
        )
    if kwargs.get("overwrite") is not None:
        cfg.overwrite = bool(kwargs["overwrite"])
    if kwargs.get("src_srs") is not None:
        cfg.src_srs = kwargs["src_srs"]
    if kwargs.get("t_srs") is not None:
        cfg.dst_srs = kwargs["t_srs"]
    if kwargs.get("out_epsg") is not None:
        cfg.dst_srs = parse_epsg_crs(int(kwargs["out_epsg"]))
    if kwargs.get("src_coord_epoch") is not None:
        cfg.src_coord_epoch = float(kwargs["src_coord_epoch"])
    if kwargs.get("dst_coord_epoch") is not None:
        cfg.dst_coord_epoch = float(kwargs["dst_coord_epoch"])
    if kwargs.get("coord_transform") is not None:
        cfg.coord_transform = str(kwargs["coord_transform"])

    tr = kwargs.get("target_res")
    if tr is not None:
        if isinstance(tr, str) and tr.strip().lower() == "square":
            cfg.target_res_square = True
            cfg.target_res = None
        elif len(tr) == 2:
            cfg.target_res = (float(tr[0]), float(tr[1]))
            cfg.target_res_square = False
        else:
            raise rsgispy.RSGISPyException(
                "target_res must be 'square' or a 2-tuple (xres, yres)."
            )

    ts = kwargs.get("target_size")
    if ts is not None:
        if len(ts) != 2:
            raise rsgispy.RSGISPyException("target_size must have 2 elements.")
        cfg.target_size = (int(ts[0]), int(ts[1]))

    te = kwargs.get("target_extent")
    if te is not None:
        if len(te) != 4:
            raise rsgispy.RSGISPyException("target_extent must have 4 elements.")
        cfg.target_extent = (
            float(te[0]),
            float(te[1]),
            float(te[2]),
            float(te[3]),
        )
    if kwargs.get("target_extent_srs") is not None:
        cfg.target_extent_srs = kwargs["target_extent_srs"]
    if kwargs.get("tap"):
        cfg.tap = True

    if kwargs.get("src_nodata") is not None:
        cfg.has_src_nodata = True
        cfg.src_nodata = _parse_nodata_values(kwargs["src_nodata"])
    if kwargs.get("dst_nodata") is not None:
        cfg.has_dst_nodata = True
        cfg.dst_nodata = _parse_nodata_values(kwargs["dst_nodata"])
    if kwargs.get("src_bands") is not None:
        cfg.src_bands = [int(b) for b in kwargs["src_bands"]]
    if kwargs.get("dst_bands") is not None:
        cfg.dst_bands = [int(b) for b in kwargs["dst_bands"]]
    if kwargs.get("src_alpha") is not None:
        cfg.src_alpha = bool(kwargs["src_alpha"])
    if kwargs.get("dst_alpha"):
        cfg.dst_alpha = True


def _apply_warp_kwargs_part2(cfg: WarpConfig, kwargs: Mapping[str, Any]) -> None:
    """Apply algorithm, transformer, cutline, and metadata keyword arguments."""
    # pylint: disable=too-many-branches,too-many-statements
    if kwargs.get("ovr") is not None:
        cfg.ovr = str(kwargs["ovr"])
    if kwargs.get("err_threshold") is not None:
        cfg.err_threshold = float(kwargs["err_threshold"])
    if kwargs.get("warp_mem_limit") is not None:
        cfg.warp_mem_limit = int(kwargs["warp_mem_limit"])
    if kwargs.get("tps"):
        cfg.tps = True
    if kwargs.get("rpc"):
        cfg.rpc = True
    if kwargs.get("geoloc"):
        cfg.geoloc = True
    if kwargs.get("order") is not None:
        cfg.order = int(kwargs["order"])
    if kwargs.get("refine_gcps") is not None:
        rg = kwargs["refine_gcps"]
        if isinstance(rg, Sequence) and not isinstance(rg, (str, bytes)):
            tol = float(rg[0])
            mg = int(rg[1]) if len(rg) > 1 and rg[1] is not None else None
            cfg.refine_gcps = (tol, mg)
        else:
            cfg.refine_gcps = (float(rg), None)
    if kwargs.get("warp_options"):
        extra_wo = normalise_creation_options(kwargs["warp_options"]) or {}
        cfg.warp_options.update({str(k): str(v) for k, v in extra_wo.items()})
    if kwargs.get("transformer_options"):
        extra_to = normalise_creation_options(kwargs["transformer_options"]) or {}
        cfg.transformer_options.update(
            {str(k): str(v) for k, v in extra_to.items()}
        )
    if kwargs.get("vshift") is not None:
        cfg.vshift = bool(kwargs["vshift"])
    if kwargs.get("cutline") is not None:
        cfg.cutline = kwargs["cutline"]
    if kwargs.get("cutline_srs") is not None:
        cfg.cutline_srs = kwargs["cutline_srs"]
    if kwargs.get("cutline_layer") is not None:
        cfg.cutline_layer = str(kwargs["cutline_layer"])
    if kwargs.get("cutline_where") is not None:
        cfg.cutline_where = str(kwargs["cutline_where"])
    if kwargs.get("cutline_sql") is not None:
        cfg.cutline_sql = str(kwargs["cutline_sql"])
    if kwargs.get("cutline_blend") is not None:
        cfg.cutline_blend = float(kwargs["cutline_blend"])
    if kwargs.get("crop_to_cutline"):
        cfg.crop_to_cutline = True
    if kwargs.get("no_metadata"):
        cfg.no_metadata = True
    if kwargs.get("meta_conflict_value") is not None:
        cfg.meta_conflict_value = str(kwargs["meta_conflict_value"])
    if kwargs.get("set_ci"):
        cfg.set_ci = True
    if kwargs.get("use_pbar"):
        cfg.use_pbar = True


def build_warp_config(
    out_epsg: Optional[Union[int, str, Any]],
    interp_method: Union[int, Resampling, str],
    gdalformat: str,
    use_multi_threaded: bool,
    options: Union[str, Sequence[str], Mapping[str, Any], None],
    **kwargs: Any,
) -> WarpConfig:
    """
    Merge ``options`` (CLI string, creation option list, or dict) with
    explicit keyword parameters into a :class:`WarpConfig`.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    cfg = WarpConfig()
    if interp_method is not None:
        cfg.resampling = _resolve_resampling(interp_method)
    if use_multi_threaded:
        cfg.num_threads = os.cpu_count() or 1

    if isinstance(options, Mapping):
        cfg.creation_opts = normalise_creation_options(options) or {}
    elif isinstance(options, str) and options.strip():
        _parse_warp_cli_tokens(shlex.split(options), cfg)
    elif isinstance(options, Sequence):
        str_items = [str(item) for item in options]
        if any(_is_cli_flag(item) for item in str_items):
            _parse_warp_cli_tokens(str_items, cfg)
        else:
            cfg.creation_opts = normalise_creation_options(str_items) or {}

    if gdalformat != "GTIFF" or cfg.gdalformat == "GTIFF":
        if gdalformat != "GTIFF":
            cfg.gdalformat = gdalformat
    if not cfg.creation_opts and cfg.gdalformat.upper().strip() == "GTIFF":
        cfg.creation_opts = dict(_DEFAULT_GTIFF_CO)

    if out_epsg is not None:
        if isinstance(out_epsg, (int, np.integer)) and not isinstance(
            out_epsg, bool
        ):
            cfg.dst_srs = parse_epsg_crs(int(out_epsg))
        else:
            cfg.dst_srs = _parse_crs_input(out_epsg)

    _apply_warp_kwargs_part1(cfg, kwargs)
    _apply_warp_kwargs_part2(cfg, kwargs)

    if "NUM_THREADS" in cfg.warp_options:
        nt = cfg.warp_options["NUM_THREADS"].strip().upper()
        cfg.num_threads = (os.cpu_count() or 1) if nt == "ALL_CPUS" else int(nt)

    return cfg


def build_transformer_and_warp_kwargs(cfg: WarpConfig) -> Dict[str, Any]:
    """Assemble ``**kwargs`` passed to ``rasterio.warp.reproject`` & grid calculation."""
    extra: Dict[str, Any] = {}
    extra.update(cfg.transformer_options)
    extra.update(cfg.warp_options)
    if cfg.tps:
        extra["METHOD"] = "GCP_TPS"
    elif cfg.rpc:
        extra["METHOD"] = "RPC"
    elif cfg.geoloc:
        extra["METHOD"] = "GEOLOC_ARRAY"
    elif cfg.order is not None:
        extra["METHOD"] = "GCP_POLYNOMIAL"
        extra["MAX_GCP_ORDER"] = str(cfg.order)
    if cfg.refine_gcps is not None:
        tol, min_g = cfg.refine_gcps
        extra["REFINE_TOLERANCE"] = str(tol)
        if min_g is not None:
            extra["REFINE_MINIMUM_GCPS"] = str(min_g)
    if cfg.coord_transform is not None:
        extra["COORDINATE_OPERATION"] = cfg.coord_transform
    if cfg.working_datatype is not None:
        extra["WORKING_DATA_TYPE"] = cfg.working_datatype
    if cfg.src_coord_epoch is not None:
        extra["SRC_COORDINATE_EPOCH"] = str(cfg.src_coord_epoch)
    if cfg.dst_coord_epoch is not None:
        extra["DST_COORDINATE_EPOCH"] = str(cfg.dst_coord_epoch)
    if cfg.vshift is not None:
        extra["APPLY_VERTCS"] = "YES" if cfg.vshift else "NO"
    return extra


def load_cutline_geometries(
    cfg: WarpConfig,
    dst_crs: Optional[rasterio.crs.CRS],
    default_srs: Optional[rasterio.crs.CRS],
) -> Optional[Tuple[List[Any], Tuple[float, float, float, float]]]:
    """Load cutline geometries, reproject to ``dst_crs``, return ``(geoms, bounds)``."""
    # pylint: disable=too-many-branches
    if cfg.cutline is None:
        return None

    cl_override = (
        _parse_crs_input(cfg.cutline_srs)
        if cfg.cutline_srs is not None
        else None
    )
    gdf: Optional[geopandas.GeoDataFrame] = None
    geoms: List[Any] = []
    geom_crs = cl_override

    if isinstance(cfg.cutline, geopandas.GeoDataFrame):
        gdf = cfg.cutline
    elif isinstance(cfg.cutline, geopandas.GeoSeries):
        gdf = geopandas.GeoDataFrame(geometry=cfg.cutline)
    elif hasattr(cfg.cutline, "geom_type"):
        geoms = [cfg.cutline]
    elif isinstance(cfg.cutline, (str, os.PathLike)):
        cl_str = os.fspath(cfg.cutline)
        if os.path.exists(cl_str):
            read_kw: Dict[str, Any] = {}
            if cfg.cutline_layer is not None:
                read_kw["layer"] = cfg.cutline_layer
            if cfg.cutline_where is not None:
                read_kw["where"] = cfg.cutline_where
            if cfg.cutline_sql is not None:
                read_kw["sql"] = cfg.cutline_sql
            gdf = geopandas.read_file(cl_str, **read_kw)
        else:
            stripped = cl_str.strip()
            if stripped.startswith("{"):
                geoms = [shapely.geometry.shape(json.loads(stripped))]
            else:
                geoms = [shapely.wkt.loads(stripped)]
    else:
        raise rsgispy.RSGISPyException(
            f"Unsupported cutline input: {cfg.cutline!r}."
        )

    if gdf is not None:
        if cl_override is not None:
            gdf = gdf.set_crs(cl_override, allow_override=True)
        elif gdf.crs is not None:
            geom_crs = gdf.crs
        if dst_crs is not None and gdf.crs is not None and gdf.crs != dst_crs:
            gdf = gdf.to_crs(dst_crs)
        geoms = [g for g in gdf.geometry if g is not None and not g.is_empty]
    else:
        src_g_crs = geom_crs if geom_crs is not None else default_srs
        gs = geopandas.GeoSeries(geoms, crs=src_g_crs)
        if dst_crs is not None and gs.crs is not None and gs.crs != dst_crs:
            gs = gs.to_crs(dst_crs)
        geoms = [g for g in gs.geometry if g is not None and not g.is_empty]

    if not geoms:
        raise rsgispy.RSGISPyException("Cutline contains no valid geometries.")

    unioned = shapely.ops.unary_union(geoms)
    bounds = tuple(float(v) for v in unioned.bounds)
    return geoms, (bounds[0], bounds[1], bounds[2], bounds[3])


def _get_src_georef(
    src: rasterio.io.DatasetReader,
    cfg: WarpConfig,
) -> Tuple[Optional[rasterio.crs.CRS], Dict[str, Any]]:
    """Determine effective source CRS and georeferencing kwargs for warp."""
    src_gcps, gcp_crs = src.gcps
    if cfg.src_srs is not None:
        s_crs = _parse_crs_input(cfg.src_srs)
    elif src.crs is not None:
        s_crs = src.crs
    elif src_gcps and gcp_crs is not None:
        s_crs = gcp_crs
    else:
        s_crs = None

    use_gcps = bool(src_gcps) and (
        cfg.tps
        or cfg.order is not None
        or cfg.refine_gcps is not None
        or src.transform == Affine.identity()
    )
    use_rpcs = src.rpcs is not None and (
        cfg.rpc or src.transform == Affine.identity()
    )

    georef: Dict[str, Any] = {}
    if use_gcps:
        georef["gcps"] = src_gcps
    elif use_rpcs:
        georef["rpcs"] = src.rpcs
    else:
        b = src.bounds
        georef["left"] = float(b.left)
        georef["bottom"] = float(b.bottom)
        georef["right"] = float(b.right)
        georef["top"] = float(b.top)
    return s_crs, georef


def _calc_natural_grid(
    src_datasets: Sequence[rasterio.io.DatasetReader],
    src_crs_list: Sequence[Optional[rasterio.crs.CRS]],
    src_georef_list: Sequence[Dict[str, Any]],
    dst_crs: Optional[rasterio.crs.CRS],
    extra_kw: Dict[str, Any],
) -> Tuple[Affine, int, int]:
    """Compute default ``(transform, width, height)`` across source datasets."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    if len(src_datasets) == 1:
        src = src_datasets[0]
        s_crs = src_crs_list[0]
        gref = src_georef_list[0]
        if s_crs is None or dst_crs is None:
            return src.transform, int(src.width), int(src.height)
        return calculate_default_transform(
            s_crs, dst_crs, src.width, src.height, **gref, **extra_kw
        )

    xmin_u, ymin_u = math.inf, math.inf
    xmax_u, ymax_u = -math.inf, -math.inf
    min_rx, min_ry = math.inf, math.inf
    for src, s_crs, gref in zip(src_datasets, src_crs_list, src_georef_list):
        if s_crs is not None and dst_crs is not None:
            t, w, h = calculate_default_transform(
                s_crs, dst_crs, src.width, src.height, **gref, **extra_kw
            )
        else:
            t, w, h = src.transform, int(src.width), int(src.height)
        rx, ry = abs(float(t.a)), abs(float(t.e))
        min_rx = min(min_rx, rx)
        min_ry = min(min_ry, ry)
        xmin_u = min(xmin_u, float(t.c))
        xmax_u = max(xmax_u, float(t.c) + w * rx)
        ymax_u = max(ymax_u, float(t.f))
        ymin_u = min(ymin_u, float(t.f) - h * ry)

    out_w = max(1, int(round((xmax_u - xmin_u) / min_rx)))
    out_h = max(1, int(round((ymax_u - ymin_u) / min_ry)))
    return Affine(min_rx, 0.0, xmin_u, 0.0, -min_ry, ymax_u), out_w, out_h


def resolve_warp_grid(
    src_datasets: Sequence[rasterio.io.DatasetReader],
    cfg: WarpConfig,
    extra_kw: Dict[str, Any],
) -> WarpGrid:
    """
    Resolve source/target CRSs, target ``(width, height, transform)``, and
    reprojected cutline geometries.
    """
    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    if (
        cfg.target_res is not None or cfg.target_res_square
    ) and cfg.target_size is not None:
        raise rsgispy.RSGISPyException(
            "Cannot specify both 'target_res' (-tr) and 'target_size' (-ts)."
        )

    src_crs_list: List[Optional[rasterio.crs.CRS]] = []
    src_georef_list: List[Dict[str, Any]] = []
    for src in src_datasets:
        s_crs, gref = _get_src_georef(src, cfg)
        src_crs_list.append(s_crs)
        src_georef_list.append(gref)

    dst_crs = (
        _parse_crs_input(cfg.dst_srs)
        if cfg.dst_srs is not None
        else src_crs_list[0]
    )

    cutline_res = load_cutline_geometries(cfg, dst_crs, src_crs_list[0])
    cutline_geoms = cutline_res[0] if cutline_res is not None else None

    bounds_override: Optional[Tuple[float, float, float, float]] = None
    if cfg.target_extent is not None:
        xmin, ymin, xmax, ymax = cfg.target_extent
        if cfg.target_extent_srs is not None and dst_crs is not None:
            te_crs = _parse_crs_input(cfg.target_extent_srs)
            if te_crs is not None and te_crs != dst_crs:
                xmin, ymin, xmax, ymax = transform_bounds(
                    te_crs, dst_crs, xmin, ymin, xmax, ymax
                )
        bounds_override = (xmin, ymin, xmax, ymax)
    elif cfg.crop_to_cutline and cutline_res is not None:
        bounds_override = cutline_res[1]

    nat_t, nat_w, nat_h = _calc_natural_grid(
        src_datasets, src_crs_list, src_georef_list, dst_crs, extra_kw
    )

    if bounds_override is not None:
        xmin, ymin, xmax, ymax = bounds_override
        if xmax <= xmin or ymax <= ymin:
            raise rsgispy.RSGISPyException(
                f"Target extent has non-positive dimensions: {bounds_override}."
            )
    else:
        xmin = float(nat_t.c)
        ymax = float(nat_t.f)
        xmax = xmin + nat_w * abs(float(nat_t.a))
        ymin = ymax - nat_h * abs(float(nat_t.e))

    if cfg.target_size is not None:
        req_w, req_h = cfg.target_size
        if req_w < 0 or req_h < 0 or (req_w == 0 and req_h == 0):
            raise rsgispy.RSGISPyException(
                f"Invalid target_size (-ts): {cfg.target_size}."
            )
        if req_w == 0:
            xres_tmp = (ymax - ymin) / float(req_h)
            req_w = max(1, int(round((xmax - xmin) / xres_tmp)))
        elif req_h == 0:
            yres_tmp = (xmax - xmin) / float(req_w)
            req_h = max(1, int(round((ymax - ymin) / yres_tmp)))
        out_w, out_h = req_w, req_h
        xres = (xmax - xmin) / float(out_w)
        yres = (ymax - ymin) / float(out_h)
        out_t = Affine(xres, 0.0, xmin, 0.0, -yres, ymax)
        return WarpGrid(
            src_crs_list=src_crs_list,
            src_georef_list=src_georef_list,
            dst_crs=dst_crs,
            width=out_w,
            height=out_h,
            transform=out_t,
            cutline_geoms=cutline_geoms,
        )

    if cfg.target_res is not None:
        xres, yres = abs(float(cfg.target_res[0])), abs(float(cfg.target_res[1]))
        if xres <= 0.0 or yres <= 0.0:
            raise rsgispy.RSGISPyException("target_res values must be positive.")
    elif cfg.target_res_square:
        sq = min(abs(float(nat_t.a)), abs(float(nat_t.e)))
        xres, yres = sq, sq
    else:
        xres, yres = abs(float(nat_t.a)), abs(float(nat_t.e))

    if cfg.tap:
        xmin = math.floor(xmin / xres) * xres
        xmax = math.ceil(xmax / xres) * xres
        ymin = math.floor(ymin / yres) * yres
        ymax = math.ceil(ymax / yres) * yres

    out_w = max(1, int(round((xmax - xmin) / xres)))
    out_h = max(1, int(round((ymax - ymin) / yres)))
    out_t = Affine(xres, 0.0, xmin, 0.0, -yres, ymax)
    return WarpGrid(
        src_crs_list=src_crs_list,
        src_georef_list=src_georef_list,
        dst_crs=dst_crs,
        width=out_w,
        height=out_h,
        transform=out_t,
        cutline_geoms=cutline_geoms,
    )


def apply_cutline_to_dataset(
    dst: rasterio.io.DatasetWriter,
    cutline_geoms: Sequence[Any],
    fill_values: Sequence[float],
    dst_alpha_band: int,
    blend_dist: float,
    use_pbar: bool = False,
) -> None:
    """
    Mask ``dst`` pixels outside ``cutline_geoms`` and optionally feather inside
    the cutline boundary over ``blend_dist`` pixels.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    if blend_dist > 0.0:
        inside = geometry_mask(
            cutline_geoms,
            out_shape=(dst.height, dst.width),
            transform=dst.transform,
            invert=True,
        )
        dist_px = ndimage.distance_transform_edt(inside)
        weights = np.clip(dist_px / float(blend_dist), 0.0, 1.0)
        for b_idx in range(1, dst.count + 1):
            arr = dst.read(b_idx)
            out_dtype = arr.dtype
            if b_idx == dst_alpha_band:
                res = arr.astype(np.float64) * weights
            else:
                bg = float(fill_values[min(b_idx - 1, len(fill_values) - 1)])
                res = (arr.astype(np.float64) * weights) + (
                    bg * (1.0 - weights)
                )
            if np.issubdtype(out_dtype, np.integer):
                res = np.rint(res)
            dst.write(res.astype(out_dtype), b_idx)
        return

    for out_win in iter_dataset_windows(dst.height, dst.width, use_pbar=use_pbar):
        win_t = window_transform(out_win, dst.transform)
        inside = geometry_mask(
            cutline_geoms,
            out_shape=(int(out_win.height), int(out_win.width)),
            transform=win_t,
            invert=True,
        )
        if np.all(inside):
            continue
        outside = ~inside
        for b_idx in range(1, dst.count + 1):
            validate_band_index(b_idx, dst.count, "output")
            arr = dst.read(b_idx, window=out_win)
            fill_v = (
                0.0
                if b_idx == dst_alpha_band
                else float(fill_values[min(b_idx - 1, len(fill_values) - 1)])
            )
            arr[outside] = arr.dtype.type(fill_v)
            dst.write(arr, b_idx, window=out_win)
