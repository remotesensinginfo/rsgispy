"""
Internal helpers and CLI option parser for :func:`rsgispy.imageutils.warp.gdal_translate`.
"""

from dataclasses import dataclass, field
import math
import shlex
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import rasterio
import rasterio.crs
from rasterio.control import GroundControlPoint
from rasterio.enums import ColorInterp, MaskFlags, Resampling
from rasterio.transform import Affine
from rasterio.warp import transform_bounds
from rasterio.windows import Window
from rasterio.windows import transform as window_transform

import rsgispy
from rsgispy.imageprocess import iter_dataset_windows, validate_band_index
from rsgispy.imageutils._helpers import normalise_creation_options

_GDAL_OT_TO_RSGIS: Dict[str, int] = {
    "byte": rsgispy.TYPE_8UINT,
    "uint8": rsgispy.TYPE_8UINT,
    "int8": rsgispy.TYPE_8INT,
    "uint16": rsgispy.TYPE_16UINT,
    "int16": rsgispy.TYPE_16INT,
    "uint32": rsgispy.TYPE_32UINT,
    "int32": rsgispy.TYPE_32INT,
    "uint64": rsgispy.TYPE_64UINT,
    "int64": rsgispy.TYPE_64INT,
    "float32": rsgispy.TYPE_32FLOAT,
    "float64": rsgispy.TYPE_64FLOAT,
}

_CLI_RESAMPLE_MAP: Dict[str, Resampling] = {
    "nearest": Resampling.nearest,
    "near": Resampling.nearest,
    "bilinear": Resampling.bilinear,
    "cubic": Resampling.cubic,
    "cubicspline": Resampling.cubic_spline,
    "cubic_spline": Resampling.cubic_spline,
    "lanczos": Resampling.lanczos,
    "average": Resampling.average,
    "mode": Resampling.mode,
    "gauss": Resampling.gauss,
    "max": Resampling.max,
    "min": Resampling.min,
    "med": Resampling.med,
    "median": Resampling.med,
    "q1": Resampling.q1,
    "q3": Resampling.q3,
    "sum": Resampling.sum,
    "rms": Resampling.rms,
}

_COLORINTERP_MAP: Dict[str, ColorInterp] = {
    c.name.lower(): c for c in ColorInterp
}

_DEFAULT_GTIFF_CO: Dict[str, Any] = {
    "TILED": "YES",
    "INTERLEAVE": "PIXEL",
    "BLOCKXSIZE": "256",
    "BLOCKYSIZE": "256",
    "COMPRESS": "LZW",
    "BIGTIFF": "YES",
    "COPY_SRC_OVERVIEWS": "YES",
}


@dataclass
class TranslateConfig:
    """Normalised configuration container for ``gdal_translate``."""

    # pylint: disable=too-many-instance-attributes
    gdalformat: str = "GTIFF"
    in_format: Optional[str] = None
    datatype: Optional[int] = None
    creation_opts: Dict[str, Any] = field(default_factory=dict)
    open_opts: Dict[str, Any] = field(default_factory=dict)
    bands: Optional[List[int]] = None
    src_win: Optional[Tuple[float, float, float, float]] = None
    proj_win: Optional[Tuple[float, float, float, float]] = None
    proj_win_srs: Optional[Any] = None
    error_partially_outside: bool = False
    error_completely_outside: bool = False
    out_size: Optional[Tuple[Union[int, float, str], Union[int, float, str]]] = (
        None
    )
    target_res: Optional[Tuple[float, float]] = None
    ovr: Optional[str] = None
    resampling: Resampling = Resampling.nearest
    out_srs: Optional[Any] = None
    has_out_srs: bool = False
    ullr: Optional[Tuple[float, float, float, float]] = None
    geotransform: Optional[Tuple[float, ...]] = None
    no_data_val: Optional[Union[float, str]] = None
    has_no_data: bool = False
    unscale: bool = False
    scale_all: Optional[List[float]] = None
    scale_per_band: Dict[int, List[float]] = field(default_factory=dict)
    exponent_all: Optional[float] = None
    exponent_per_band: Dict[int, float] = field(default_factory=dict)
    band_scale: Optional[Union[float, Sequence[float]]] = None
    band_offset: Optional[Union[float, Sequence[float]]] = None
    expand: Optional[str] = None
    color_interp_list: Optional[List[ColorInterp]] = None
    color_interp_per_band: Dict[int, ColorInterp] = field(default_factory=dict)
    mask_band: Optional[Union[int, str]] = None
    no_gcp: bool = False
    gcps: List[GroundControlPoint] = field(default_factory=list)
    stats: bool = False
    approx_stats: bool = False
    metadata: Dict[str, str] = field(default_factory=dict)
    domain_metadata: Dict[str, Dict[str, str]] = field(default_factory=dict)
    no_metadata: bool = False
    no_xmp: bool = False
    strict: bool = False
    use_pbar: bool = True


def _is_cli_flag(token: str) -> bool:
    """Return True if ``token`` looks like a CLI flag (not a negative number)."""
    return (
        token.startswith("-")
        and len(token) > 1
        and (token[1].isalpha() or token[1] == "-")
    )


def _is_numeric(token: str) -> bool:
    """Return True if ``token`` parses cleanly as a finite float."""
    if _is_cli_flag(token):
        return False
    try:
        val = float(token)
        return math.isfinite(val)
    except ValueError:
        return False


def _consume_args(tokens: Sequence[str], idx: int, count: int, flag: str) -> List[str]:
    """Consume ``count`` arguments following ``flag`` or raise an exception."""
    end = idx + count
    if end > len(tokens):
        raise rsgispy.RSGISPyException(
            f"Option '{flag}' requires {count} argument(s)."
        )
    return list(tokens[idx:end])


def _parse_datatype_arg(dtype: Union[int, str]) -> int:
    """Resolve an ``rsgispy.TYPE_*`` int or GDAL type string to ``TYPE_*``."""
    if isinstance(dtype, (int, np.integer)) and not isinstance(dtype, bool):
        return int(dtype)
    if isinstance(dtype, str):
        key = dtype.strip().lower()
        if key in _GDAL_OT_TO_RSGIS:
            return _GDAL_OT_TO_RSGIS[key]
        return rsgispy.get_rsgis_datatype(key)
    raise rsgispy.RSGISPyException(f"Unsupported datatype value: {dtype!r}.")


def _resolve_resampling(alg: Union[int, Resampling, str]) -> Resampling:
    """Convert ``Resampling`` enum, ``INTERP_*`` int, or CLI string."""
    if isinstance(alg, Resampling):
        return alg
    if isinstance(alg, (int, np.integer)) and not isinstance(alg, bool):
        return rsgispy.get_rasterio_interp_type(int(alg))
    if isinstance(alg, str):
        key = alg.strip().lower()
        if key in _CLI_RESAMPLE_MAP:
            return _CLI_RESAMPLE_MAP[key]
    raise rsgispy.RSGISPyException(f"Unsupported resampling method: {alg!r}.")


def _parse_colorinterp_value(val: Union[str, int, ColorInterp]) -> ColorInterp:
    """Resolve a color interpretation string, integer, or ColorInterp enum."""
    if isinstance(val, ColorInterp):
        return val
    if isinstance(val, (int, np.integer)) and not isinstance(val, bool):
        try:
            return ColorInterp(int(val))
        except ValueError as err:
            raise rsgispy.RSGISPyException(
                f"Invalid color interpretation value: {val!r}."
            ) from err
    if isinstance(val, str):
        key = val.strip().lower()
        if key in _COLORINTERP_MAP:
            return _COLORINTERP_MAP[key]
    raise rsgispy.RSGISPyException(f"Unsupported color interpretation: {val!r}.")


def _parse_crs_input(srs_def: Any) -> Optional[rasterio.crs.CRS]:
    """Convert an SRS definition (EPSG int, string, CRS) or ``'none'`` to CRS."""
    # pylint: disable=c-extension-no-member
    if srs_def is None:
        return None
    if isinstance(srs_def, str):
        cleaned = srs_def.strip()
        if cleaned == "" or cleaned.lower() == "none":
            return None
        if cleaned.isdigit():
            return rasterio.crs.CRS.from_epsg(int(cleaned))
        return rasterio.crs.CRS.from_user_input(cleaned)
    if isinstance(srs_def, (int, np.integer)) and not isinstance(srs_def, bool):
        return rasterio.crs.CRS.from_epsg(int(srs_def))
    return rasterio.crs.CRS.from_user_input(srs_def)


def _split_kv(item: str, flag: str) -> Tuple[str, str]:
    """Split a ``KEY=VALUE`` option string."""
    if "=" not in item:
        raise rsgispy.RSGISPyException(
            f"Option '{flag}' expects KEY=VALUE, got {item!r}."
        )
    key, val = item.split("=", 1)
    return key.strip(), val.strip()


def _parse_cli_tokens(tokens: Sequence[str], cfg: TranslateConfig) -> None:
    """Populate ``cfg`` from GDAL-style CLI flag tokens."""
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
            vals = _consume_args(tokens, idx, 1, tok)
            cfg.gdalformat = vals[0]
            idx += 1
        elif tok == "-if":
            cfg.in_format = _consume_args(tokens, idx, 1, tok)[0]
            idx += 1
        elif tok == "-ot":
            cfg.datatype = _parse_datatype_arg(
                _consume_args(tokens, idx, 1, tok)[0]
            )
            idx += 1
        elif tok == "-b":
            bval = _consume_args(tokens, idx, 1, tok)[0]
            idx += 1
            if cfg.bands is None:
                cfg.bands = []
            try:
                cfg.bands.append(int(bval))
            except ValueError as err:
                raise rsgispy.RSGISPyException(
                    f"Invalid band index for -b: {bval!r}."
                ) from err
        elif tok == "-mask":
            mval = _consume_args(tokens, idx, 1, tok)[0]
            idx += 1
            m_low = mval.strip().lower()
            if m_low in ("none", "auto"):
                cfg.mask_band = m_low
            else:
                try:
                    cfg.mask_band = int(mval)
                except ValueError as err:
                    raise rsgispy.RSGISPyException(
                        f"Invalid -mask value: {mval!r}."
                    ) from err
        elif tok == "-expand":
            cfg.expand = _consume_args(tokens, idx, 1, tok)[0].strip().lower()
            idx += 1
        elif tok == "-outsize":
            vals = _consume_args(tokens, idx, 2, tok)
            cfg.out_size = (vals[0], vals[1])
            idx += 2
        elif tok == "-tr":
            vals = _consume_args(tokens, idx, 2, tok)
            cfg.target_res = (float(vals[0]), float(vals[1]))
            idx += 2
        elif tok == "-ovr":
            cfg.ovr = _consume_args(tokens, idx, 1, tok)[0]
            idx += 1
        elif tok == "-r":
            cfg.resampling = _resolve_resampling(
                _consume_args(tokens, idx, 1, tok)[0]
            )
            idx += 1
        elif tok == "-unscale":
            cfg.unscale = True
        elif tok == "-scale" or tok.startswith("-scale_"):
            num_vals: List[float] = []
            while idx < len(tokens) and len(num_vals) < 4 and _is_numeric(tokens[idx]):
                num_vals.append(float(tokens[idx]))
                idx += 1
            if len(num_vals) not in (0, 2, 4):
                raise rsgispy.RSGISPyException(
                    f"Option '{tok}' expects 0, 2, or 4 numeric arguments."
                )
            if tok == "-scale":
                cfg.scale_all = num_vals
            else:
                band_idx = int(tok.split("_", 1)[1])
                cfg.scale_per_band[band_idx] = num_vals
        elif tok == "-exponent" or tok.startswith("-exponent_"):
            eval_str = _consume_args(tokens, idx, 1, tok)[0]
            idx += 1
            if tok == "-exponent":
                cfg.exponent_all = float(eval_str)
            else:
                band_idx = int(tok.split("_", 1)[1])
                cfg.exponent_per_band[band_idx] = float(eval_str)
        elif tok == "-srcwin":
            vals = _consume_args(tokens, idx, 4, tok)
            cfg.src_win = (
                float(vals[0]),
                float(vals[1]),
                float(vals[2]),
                float(vals[3]),
            )
            idx += 4
        elif tok == "-projwin":
            vals = _consume_args(tokens, idx, 4, tok)
            cfg.proj_win = (
                float(vals[0]),
                float(vals[1]),
                float(vals[2]),
                float(vals[3]),
            )
            idx += 4
        elif tok == "-projwin_srs":
            cfg.proj_win_srs = _consume_args(tokens, idx, 1, tok)[0]
            idx += 1
        elif tok == "-epo":
            cfg.error_partially_outside = True
        elif tok == "-eco":
            cfg.error_completely_outside = True
        elif tok == "-a_srs":
            cfg.out_srs = _consume_args(tokens, idx, 1, tok)[0]
            cfg.has_out_srs = True
            idx += 1
        elif tok == "-a_ullr":
            vals = _consume_args(tokens, idx, 4, tok)
            cfg.ullr = (
                float(vals[0]),
                float(vals[1]),
                float(vals[2]),
                float(vals[3]),
            )
            idx += 4
        elif tok == "-a_gt":
            vals = _consume_args(tokens, idx, 6, tok)
            cfg.geotransform = tuple(float(v) for v in vals)
            idx += 6
        elif tok == "-a_nodata":
            nd_str = _consume_args(tokens, idx, 1, tok)[0]
            idx += 1
            cfg.has_no_data = True
            cfg.no_data_val = (
                "none" if nd_str.strip().lower() == "none" else float(nd_str)
            )
        elif tok == "-a_scale":
            cfg.band_scale = float(_consume_args(tokens, idx, 1, tok)[0])
            idx += 1
        elif tok == "-a_offset":
            cfg.band_offset = float(_consume_args(tokens, idx, 1, tok)[0])
            idx += 1
        elif tok == "-nogcp":
            cfg.no_gcp = True
        elif tok == "-gcp":
            vals = _consume_args(tokens, idx, 4, tok)
            idx += 4
            elev = 0.0
            if idx < len(tokens) and _is_numeric(tokens[idx]):
                elev = float(tokens[idx])
                idx += 1
            cfg.gcps.append(
                GroundControlPoint(
                    row=float(vals[1]),
                    col=float(vals[0]),
                    x=float(vals[2]),
                    y=float(vals[3]),
                    z=elev,
                )
            )
        elif tok == "-colorinterp":
            c_str = _consume_args(tokens, idx, 1, tok)[0]
            idx += 1
            cfg.color_interp_list = [
                _parse_colorinterp_value(part) for part in c_str.split(",")
            ]
        elif tok.startswith("-colorinterp_"):
            band_idx = int(tok.split("_", 1)[1])
            c_str = _consume_args(tokens, idx, 1, tok)[0]
            idx += 1
            cfg.color_interp_per_band[band_idx] = _parse_colorinterp_value(
                c_str
            )
        elif tok == "-stats":
            cfg.stats = True
        elif tok == "-approx_stats":
            cfg.approx_stats = True
        elif tok == "-nomd":
            cfg.no_metadata = True
        elif tok == "-noxmp":
            cfg.no_xmp = True
        elif tok == "-norat":
            pass
        elif tok == "-strict":
            cfg.strict = True
        elif tok == "-not_strict":
            cfg.strict = False
        elif tok == "-co":
            key, val = _split_kv(_consume_args(tokens, idx, 1, tok)[0], tok)
            cfg.creation_opts[key] = val
            idx += 1
        elif tok == "-oo":
            key, val = _split_kv(_consume_args(tokens, idx, 1, tok)[0], tok)
            cfg.open_opts[key] = val
            idx += 1
        elif tok == "-mo":
            key, val = _split_kv(_consume_args(tokens, idx, 1, tok)[0], tok)
            cfg.metadata[key] = val
            idx += 1
        elif tok == "-dmo":
            dmo_raw = _consume_args(tokens, idx, 1, tok)[0]
            idx += 1
            if ":" not in dmo_raw or "=" not in dmo_raw:
                raise rsgispy.RSGISPyException(
                    f"Option '-dmo' expects DOMAIN:KEY=VALUE, got {dmo_raw!r}."
                )
            dom, kv = dmo_raw.split(":", 1)
            key, val = _split_kv(kv, tok)
            cfg.domain_metadata.setdefault(dom.strip(), {})[key] = val
        elif tok in ("--quiet", "-q"):
            cfg.use_pbar = False
        elif _is_cli_flag(tok):
            raise rsgispy.RSGISPyException(
                f"Unrecognised gdal_translate option '{tok}'."
            )


def build_translate_config(
    gdalformat: str,
    datatype: Optional[Union[int, str]],
    options: Union[str, Sequence[str], Mapping[str, Any], None],
    **kwargs: Any,
) -> TranslateConfig:
    """
    Merge ``options`` (CLI string, creation option list, or dict) with
    explicit keyword parameters into a :class:`TranslateConfig`.
    """
    # pylint: disable=too-many-branches,too-many-statements,too-many-locals
    cfg = TranslateConfig()

    if isinstance(options, Mapping):
        cfg.creation_opts = normalise_creation_options(options) or {}
    elif isinstance(options, str) and options.strip():
        _parse_cli_tokens(shlex.split(options), cfg)
    elif isinstance(options, Sequence):
        str_items = [str(item) for item in options]
        if any(_is_cli_flag(item) for item in str_items):
            _parse_cli_tokens(str_items, cfg)
        else:
            cfg.creation_opts = normalise_creation_options(str_items) or {}

    if gdalformat != "GTIFF" or cfg.gdalformat == "GTIFF":
        # Preserve explicit gdalformat unless CLI -of changed it while gdalformat stayed default.
        if gdalformat != "GTIFF":
            cfg.gdalformat = gdalformat
    if not cfg.creation_opts and cfg.gdalformat.upper().strip() == "GTIFF":
        cfg.creation_opts = dict(_DEFAULT_GTIFF_CO)

    if kwargs.get("in_format") is not None:
        cfg.in_format = str(kwargs["in_format"])
    if datatype is not None:
        cfg.datatype = _parse_datatype_arg(datatype)

    if kwargs.get("open_options"):
        extra_oo = normalise_creation_options(kwargs["open_options"]) or {}
        cfg.open_opts.update(extra_oo)
    if kwargs.get("bands") is not None:
        cfg.bands = [int(b) for b in kwargs["bands"]]
    if kwargs.get("src_win") is not None:
        sw = kwargs["src_win"]
        if len(sw) != 4:
            raise rsgispy.RSGISPyException("src_win must have 4 elements.")
        cfg.src_win = (float(sw[0]), float(sw[1]), float(sw[2]), float(sw[3]))
    if kwargs.get("proj_win") is not None:
        pw = kwargs["proj_win"]
        if len(pw) != 4:
            raise rsgispy.RSGISPyException("proj_win must have 4 elements.")
        cfg.proj_win = (float(pw[0]), float(pw[1]), float(pw[2]), float(pw[3]))
    if kwargs.get("proj_win_srs") is not None:
        cfg.proj_win_srs = kwargs["proj_win_srs"]
    if kwargs.get("error_partially_outside"):
        cfg.error_partially_outside = True
    if kwargs.get("error_completely_outside"):
        cfg.error_completely_outside = True
    if kwargs.get("out_size") is not None:
        cfg.out_size = tuple(kwargs["out_size"])  # type: ignore[assignment]
    if kwargs.get("target_res") is not None:
        tr = kwargs["target_res"]
        cfg.target_res = (float(tr[0]), float(tr[1]))
    if kwargs.get("ovr") is not None:
        cfg.ovr = str(kwargs["ovr"])
    if kwargs.get("resample_alg") is not None:
        cfg.resampling = _resolve_resampling(kwargs["resample_alg"])
    if kwargs.get("out_srs") is not None:
        cfg.out_srs = kwargs["out_srs"]
        cfg.has_out_srs = True
    if kwargs.get("ullr") is not None:
        ul = kwargs["ullr"]
        if len(ul) != 4:
            raise rsgispy.RSGISPyException("ullr must have 4 elements.")
        cfg.ullr = (float(ul[0]), float(ul[1]), float(ul[2]), float(ul[3]))
    if kwargs.get("geotransform") is not None:
        gt = tuple(float(v) for v in kwargs["geotransform"])
        if len(gt) != 6:
            raise rsgispy.RSGISPyException("geotransform must have 6 elements.")
        cfg.geotransform = gt
    if kwargs.get("no_data_val") is not None:
        nd = kwargs["no_data_val"]
        cfg.has_no_data = True
        cfg.no_data_val = (
            "none"
            if isinstance(nd, str) and nd.strip().lower() == "none"
            else float(nd)
        )
    if kwargs.get("unscale"):
        cfg.unscale = True

    sp = kwargs.get("scale_params")
    if sp is not None:
        if sp is True:
            cfg.scale_all = []
        elif isinstance(sp, Sequence) and not isinstance(sp, (str, bytes)):
            if len(sp) == 0:
                cfg.scale_all = []
            elif isinstance(sp[0], Sequence) and not isinstance(
                sp[0], (str, bytes)
            ):
                for b_idx, b_vals in enumerate(sp, start=1):
                    cfg.scale_per_band[b_idx] = [float(v) for v in b_vals]
            else:
                cfg.scale_all = [float(v) for v in sp]  # type: ignore[arg-type]

    exp_kw = kwargs.get("exponent")
    if exp_kw is not None:
        if isinstance(exp_kw, Sequence) and not isinstance(exp_kw, (str, bytes)):
            for b_idx, val in enumerate(exp_kw, start=1):
                cfg.exponent_per_band[b_idx] = float(val)
        else:
            cfg.exponent_all = float(exp_kw)

    if kwargs.get("band_scale") is not None:
        cfg.band_scale = kwargs["band_scale"]
    if kwargs.get("band_offset") is not None:
        cfg.band_offset = kwargs["band_offset"]
    if kwargs.get("expand") is not None:
        cfg.expand = str(kwargs["expand"]).strip().lower()

    ci_kw = kwargs.get("color_interp")
    if ci_kw is not None:
        if isinstance(ci_kw, (str, ColorInterp)):
            parts = ci_kw.split(",") if isinstance(ci_kw, str) else [ci_kw]
            cfg.color_interp_list = [
                _parse_colorinterp_value(p) for p in parts
            ]
        elif isinstance(ci_kw, Mapping):
            for k, v in ci_kw.items():
                cfg.color_interp_per_band[int(k)] = _parse_colorinterp_value(v)
        elif isinstance(ci_kw, Sequence):
            cfg.color_interp_list = [
                _parse_colorinterp_value(p) for p in ci_kw
            ]

    if kwargs.get("mask_band") is not None:
        mb = kwargs["mask_band"]
        cfg.mask_band = (
            mb.strip().lower() if isinstance(mb, str) else int(mb)
        )
    if kwargs.get("no_gcp"):
        cfg.no_gcp = True
    if kwargs.get("gcps") is not None:
        for gcp_item in kwargs["gcps"]:
            if isinstance(gcp_item, GroundControlPoint):
                cfg.gcps.append(gcp_item)
            else:
                if len(gcp_item) not in (4, 5):
                    raise rsgispy.RSGISPyException(
                        "Each GCP must have 4 or 5 elements: "
                        "(pixel, line, easting, northing[, elevation])."
                    )
                elev = float(gcp_item[4]) if len(gcp_item) == 5 else 0.0
                cfg.gcps.append(
                    GroundControlPoint(
                        row=float(gcp_item[1]),
                        col=float(gcp_item[0]),
                        x=float(gcp_item[2]),
                        y=float(gcp_item[3]),
                        z=elev,
                    )
                )
    if kwargs.get("stats"):
        cfg.stats = True
    if kwargs.get("approx_stats"):
        cfg.approx_stats = True
    if kwargs.get("metadata"):
        extra_md = normalise_creation_options(kwargs["metadata"]) or {}
        cfg.metadata.update({k: str(v) for k, v in extra_md.items()})
    if kwargs.get("domain_metadata"):
        for dom, dom_tags in kwargs["domain_metadata"].items():
            cfg.domain_metadata.setdefault(dom, {}).update(
                {str(k): str(v) for k, v in dom_tags.items()}
            )
    if kwargs.get("no_metadata"):
        cfg.no_metadata = True
    if kwargs.get("no_xmp"):
        cfg.no_xmp = True
    if kwargs.get("strict"):
        cfg.strict = True
    if kwargs.get("use_pbar") is False:
        cfg.use_pbar = False

    return cfg


def resolve_src_window(
    src: rasterio.io.DatasetReader, cfg: TranslateConfig
) -> Window:
    """Compute and validate the clipped source pixel :class:`Window`."""
    # pylint: disable=too-many-locals,too-many-branches
    if cfg.src_win is not None and cfg.proj_win is not None:
        raise rsgispy.RSGISPyException(
            "Cannot specify both 'src_win' (-srcwin) and 'proj_win' (-projwin)."
        )

    if cfg.src_win is not None:
        xoff, yoff, xsize, ysize = cfg.src_win
    elif cfg.proj_win is not None:
        ulx, uly, lrx, lry = cfg.proj_win
        if cfg.proj_win_srs is not None and src.crs is not None:
            pw_crs = _parse_crs_input(cfg.proj_win_srs)
            if pw_crs is not None and pw_crs != src.crs:
                minx, miny, maxx, maxy = transform_bounds(
                    pw_crs,
                    src.crs,
                    min(ulx, lrx),
                    min(uly, lry),
                    max(ulx, lrx),
                    max(uly, lry),
                )
                ulx, uly, lrx, lry = minx, maxy, maxx, miny
        inv_t = ~src.transform
        c1, r1 = inv_t @ (ulx, uly)
        c2, r2 = inv_t @ (lrx, lry)
        xoff = float(math.floor(min(c1, c2) + 1e-6))
        yoff = float(math.floor(min(r1, r2) + 1e-6))
        x_end = float(math.ceil(max(c1, c2) - 1e-6))
        y_end = float(math.ceil(max(r1, r2) - 1e-6))
        xsize = x_end - xoff
        ysize = y_end - yoff
    else:
        return Window(0, 0, src.width, src.height)

    if xsize <= 0 or ysize <= 0:
        raise rsgispy.RSGISPyException(
            f"Source window dimensions must be positive (got {xsize}x{ysize})."
        )

    completely_outside = (
        (xoff + xsize) <= 0
        or (yoff + ysize) <= 0
        or xoff >= src.width
        or yoff >= src.height
    )
    if completely_outside:
        raise rsgispy.RSGISPyException(
            "Requested window lies completely outside the input raster extent."
        )

    partially_outside = (
        xoff < 0
        or yoff < 0
        or (xoff + xsize) > src.width
        or (yoff + ysize) > src.height
    )
    if cfg.error_partially_outside and partially_outside:
        raise rsgispy.RSGISPyException(
            "Requested window extends partially outside the input raster (-epo)."
        )

    c_start = max(0, int(round(xoff)))
    r_start = max(0, int(round(yoff)))
    c_stop = min(src.width, int(round(xoff + xsize)))
    r_stop = min(src.height, int(round(yoff + ysize)))
    if c_stop <= c_start or r_stop <= r_start:
        raise rsgispy.RSGISPyException(
            "Clipped source window has zero width or height."
        )
    return Window(c_start, r_start, c_stop - c_start, r_stop - r_start)


def _parse_size_dim(
    val: Union[int, float, str], src_dim: int
) -> Optional[int]:
    """Parse an ``-outsize`` component (integer, percentage string, or ``0``)."""
    if isinstance(val, str):
        stripped = val.strip()
        if stripped.endswith("%"):
            pct = float(stripped[:-1])
            if pct <= 0:
                raise rsgispy.RSGISPyException(
                    f"Outsize percentage must be positive (got {val!r})."
                )
            return max(1, int(round(src_dim * (pct / 100.0))))
        num = float(stripped)
    else:
        num = float(val)
    if num == 0:
        return None
    if num < 0:
        raise rsgispy.RSGISPyException(
            f"Outsize dimension cannot be negative (got {val!r})."
        )
    return max(1, int(round(num)))


def resolve_output_grid(
    src_win: Window,
    src_transform: Affine,
    cfg: TranslateConfig,
) -> Tuple[int, int, Affine]:
    """Resolve output ``(width, height, transform)``."""
    # pylint: disable=too-many-locals,too-many-branches
    if cfg.out_size is not None and cfg.target_res is not None:
        raise rsgispy.RSGISPyException(
            "Cannot specify both 'out_size' (-outsize) and 'target_res' (-tr)."
        )
    if cfg.ullr is not None and cfg.geotransform is not None:
        raise rsgispy.RSGISPyException(
            "Cannot specify both 'ullr' (-a_ullr) and 'geotransform' (-a_gt)."
        )

    win_w = int(round(src_win.width))
    win_h = int(round(src_win.height))
    base_t = window_transform(src_win, src_transform)

    if cfg.out_size is not None:
        if len(cfg.out_size) != 2:
            raise rsgispy.RSGISPyException("out_size must have 2 elements.")
        req_w = _parse_size_dim(cfg.out_size[0], win_w)
        req_h = _parse_size_dim(cfg.out_size[1], win_h)
        if req_w is None and req_h is None:
            raise rsgispy.RSGISPyException(
                "Both out_size dimensions cannot be 0."
            )
        if req_w is None:
            assert req_h is not None
            req_w = max(1, int(round(win_w * (float(req_h) / float(win_h)))))
        if req_h is None:
            req_h = max(1, int(round(win_h * (float(req_w) / float(win_w)))))
        out_w, out_h = req_w, req_h
        out_t = base_t @ Affine.scale(
            float(win_w) / float(out_w), float(win_h) / float(out_h)
        )
    elif cfg.target_res is not None:
        xres, yres = abs(float(cfg.target_res[0])), abs(float(cfg.target_res[1]))
        if xres == 0.0 or yres == 0.0:
            raise rsgispy.RSGISPyException("target_res values must be non-zero.")
        cur_xres = math.hypot(base_t.a, base_t.b)
        cur_yres = math.hypot(base_t.d, base_t.e)
        out_w = max(1, int(round((cur_xres * win_w) / xres)))
        out_h = max(1, int(round((cur_yres * win_h) / yres)))
        if base_t.b == 0.0 and base_t.d == 0.0:
            out_t = Affine(
                math.copysign(xres, base_t.a),
                0.0,
                base_t.c,
                0.0,
                math.copysign(yres, base_t.e),
                base_t.f,
            )
        else:
            out_t = base_t @ Affine.scale(
                float(win_w) / float(out_w), float(win_h) / float(out_h)
            )
    else:
        out_w, out_h = win_w, win_h
        out_t = base_t

    if cfg.ullr is not None:
        ulx, uly, lrx, lry = cfg.ullr
        out_t = Affine.translation(ulx, uly) @ Affine.scale(
            (lrx - ulx) / float(out_w), (lry - uly) / float(out_h)
        )
    elif cfg.geotransform is not None:
        out_t = Affine.from_gdal(*cfg.geotransform)

    return out_w, out_h, out_t


def compute_src_minmax(
    src: rasterio.io.DatasetReader,
    src_bands: Sequence[int],
    src_win: Window,
    unscale: bool = False,
) -> Dict[int, Tuple[float, float]]:
    """Compute ``(min, max)`` for each requested source band inside ``src_win``."""
    # pylint: disable=too-many-locals
    mins = {b: math.inf for b in src_bands}
    maxs = {b: -math.inf for b in src_bands}
    src_w = int(round(src_win.width))
    src_h = int(round(src_win.height))
    c0 = int(round(src_win.col_off))
    r0 = int(round(src_win.row_off))
    nd_vals = src.nodatavals

    for sub_win in iter_dataset_windows(src_h, src_w, use_pbar=False):
        r_win = Window(
            c0 + int(sub_win.col_off),
            r0 + int(sub_win.row_off),
            int(sub_win.width),
            int(sub_win.height),
        )
        block = src.read(list(src_bands), window=r_win).astype(np.float64)
        for pos, bidx in enumerate(src_bands):
            band_arr = block[pos]
            mask = np.isfinite(band_arr)
            b_nd = nd_vals[bidx - 1]
            if b_nd is not None:
                mask &= band_arr != float(b_nd)
            valid = band_arr[mask]
            if unscale and valid.size > 0:
                s_val = float(src.scales[bidx - 1])
                o_val = float(src.offsets[bidx - 1])
                if s_val != 1.0 or o_val != 0.0:
                    valid = (valid * s_val) + o_val
            if valid.size > 0:
                mins[bidx] = min(mins[bidx], float(np.min(valid)))
                maxs[bidx] = max(maxs[bidx], float(np.max(valid)))

    result: Dict[int, Tuple[float, float]] = {}
    for bidx in src_bands:
        if math.isinf(mins[bidx]):
            result[bidx] = (0.0, 255.0)
        else:
            result[bidx] = (mins[bidx], maxs[bidx])
    return result


def resolve_scale_specs(
    src: rasterio.io.DatasetReader,
    src_bands: Sequence[int],
    src_win: Window,
    cfg: TranslateConfig,
) -> Optional[List[Optional[Tuple[float, float, float, float, float]]]]:
    """
    Build per-output-band tuples ``(src_min, src_max, dst_min, dst_max, exp)``
    or ``None`` if no ``-scale`` / ``-exponent`` transform is active.
    """
    # pylint: disable=too-many-branches,too-many-locals
    has_scale = cfg.scale_all is not None or bool(cfg.scale_per_band)
    has_exp = cfg.exponent_all is not None or bool(cfg.exponent_per_band)
    if not has_scale and not has_exp:
        return None

    need_minmax = False
    if cfg.scale_all is not None and len(cfg.scale_all) == 0:
        need_minmax = True
    for vals in cfg.scale_per_band.values():
        if len(vals) == 0:
            need_minmax = True
    if not has_scale and has_exp:
        need_minmax = True

    auto_mm = (
        compute_src_minmax(src, src_bands, src_win, unscale=cfg.unscale)
        if need_minmax
        else {}
    )

    specs: List[Optional[Tuple[float, float, float, float, float]]] = []
    for out_pos, bidx in enumerate(src_bands, start=1):
        s_vals = cfg.scale_per_band.get(out_pos, cfg.scale_all)
        exp_val = cfg.exponent_per_band.get(out_pos, cfg.exponent_all)
        if s_vals is None and exp_val is None:
            specs.append(None)
            continue
        if s_vals is None or len(s_vals) == 0:
            smin, smax = auto_mm[bidx]
            dmin, dmax = 0.0, 255.0
        elif len(s_vals) == 2:
            smin, smax = s_vals[0], s_vals[1]
            dmin, dmax = 0.0, 255.0
        else:
            smin, smax, dmin, dmax = s_vals[0], s_vals[1], s_vals[2], s_vals[3]
        specs.append(
            (smin, smax, dmin, dmax, 1.0 if exp_val is None else float(exp_val))
        )
    return specs


def expand_colormap_block(
    block: np.ndarray,
    cmap: Mapping[int, Tuple[int, int, int, int]],
    mode: str,
) -> np.ndarray:
    """Expand a single-band indexed block ``(1, H, W)`` into gray/RGB/RGBA."""
    if mode not in ("gray", "rgb", "rgba"):
        raise rsgispy.RSGISPyException(
            f"Unsupported -expand mode '{mode}' (expected 'gray', 'rgb', or 'rgba')."
        )
    indices = block[0]
    height, width = indices.shape
    max_key = max((int(k) for k in cmap), default=255)
    lut = np.zeros((max(max_key + 1, 256), 4), dtype=np.uint8)
    for key, rgba in cmap.items():
        if int(key) >= 0:
            lut[int(key)] = (
                int(rgba[0]),
                int(rgba[1]),
                int(rgba[2]),
                int(rgba[3]) if len(rgba) > 3 else 255,
            )

    safe_idx = np.clip(indices.astype(np.int64), 0, lut.shape[0] - 1)
    rgba_arr = lut[safe_idx]  # (H, W, 4)

    if mode == "rgba":
        return np.moveaxis(rgba_arr, -1, 0)
    if mode == "rgb":
        return np.moveaxis(rgba_arr[:, :, :3], -1, 0)

    # gray
    gray = np.rint(
        0.299 * rgba_arr[:, :, 0]
        + 0.587 * rgba_arr[:, :, 1]
        + 0.114 * rgba_arr[:, :, 2]
    ).astype(np.uint8)
    return gray.reshape(1, height, width)


def validate_requested_bands(
    src: rasterio.io.DatasetReader,
    cfg: TranslateConfig,
    input_img: str,
) -> List[int]:
    """Validate and return 1-based source band indices to read."""
    if cfg.bands is None:
        return list(range(1, src.count + 1))
    if not cfg.bands:
        raise rsgispy.RSGISPyException("At least one band must be selected.")
    for bidx in cfg.bands:
        validate_band_index(bidx, src.count, input_img)
    return list(cfg.bands)


def has_per_dataset_mask(src: rasterio.io.DatasetReader) -> bool:
    """Return True if ``src`` has a per-dataset raster mask."""
    return any(
        MaskFlags.per_dataset in flags for flags in src.mask_flag_enums
    )
