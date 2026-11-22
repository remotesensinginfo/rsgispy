"""
Command-line tool ``rsgisimginfo``: display complete, compact raster metadata
for one or more input images in a single dataset open.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyproj
import rasterio

from rsgispy import RSGISPyException
from rsgispy.cli._common import (
    add_input_files_arg,
    add_json_arg,
    build_parser,
    format_bytes,
    format_table,
    resolve_input_files,
    run_cli,
)
from rsgispy.tools.projection import degrees_to_metres

__all__ = [
    "collect_img_info",
    "format_img_info",
    "build_imginfo_parser",
    "main",
    "cli_main",
]

_HISTO_TAG_KEYS = frozenset({"STATISTICS_HISTOBINVALUES"})
_TILED_DRIVERS = frozenset({"COG", "KEA", "JP2OpenJPEG", "JP2KAK", "JP2ECW"})


def _fmt_num(val: Optional[float], sig: int = 5) -> str:
    """Format a numeric value compactly for terminal display."""
    if val is None:
        return "-"
    if isinstance(val, (int, np.integer)):
        return str(int(val))
    fval = float(val)
    if math.isnan(fval):
        return "nan"
    if math.isinf(fval):
        return "inf" if fval > 0 else "-inf"
    if fval == 0.0:
        return "0"
    if abs(fval - round(fval)) < 1e-9 and abs(fval) < 1e12:
        return str(int(round(fval)))
    return f"{fval:.{sig}g}"


def _parse_float_tag(tags: Dict[str, str], key: str) -> Optional[float]:
    """Parse an optional float value from a raster band tag dictionary."""
    raw = tags.get(key)
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _get_crs_summary(crs: Any) -> Dict[str, Any]:
    """Extract structured CRS information using ``pyproj.CRS``."""
    if crs is None:
        return {
            "defined": False,
            "authority": None,
            "name": None,
            "type": None,
            "unit": None,
            "is_geographic": False,
            "wkt": None,
            "wkt_pretty": None,
            "proj4": None,
        }

    py_crs = pyproj.CRS.from_user_input(crs)
    auth = py_crs.to_authority()
    auth_str = f"{auth[0]}:{auth[1]}" if auth else None
    crs_type = (
        "geographic"
        if py_crs.is_geographic
        else "projected"
        if py_crs.is_projected
        else "compound"
        if py_crs.is_compound
        else "other"
    )
    unit_name = None
    if py_crs.axis_info:
        unit_name = py_crs.axis_info[0].unit_name

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            proj4_str = py_crs.to_proj4()
    except Exception:  # pylint: disable=broad-exception-caught
        proj4_str = None

    return {
        "defined": True,
        "authority": auth_str,
        "name": py_crs.name,
        "type": crs_type,
        "unit": unit_name,
        "is_geographic": bool(py_crs.is_geographic),
        "wkt": py_crs.to_wkt(),
        "wkt_pretty": py_crs.to_wkt(pretty=True),
        "proj4": proj4_str,
    }


def _compute_wgs84_bbox(
    bounds: Tuple[float, float, float, float],
    crs: Any,
) -> Optional[List[float]]:
    """Transform a projected bounding box ``(left, bottom, right, top)`` to WGS84."""
    try:
        py_crs = pyproj.CRS.from_user_input(crs)
        if py_crs.is_geographic:
            return None
        transformer = pyproj.Transformer.from_crs(
            py_crs, pyproj.CRS.from_epsg(4326), always_xy=True
        )
        left, bottom, right, top = bounds
        west, south, east, north = transformer.transform_bounds(
            left, bottom, right, top, densify_pts=21
        )
        if not all(math.isfinite(v) for v in (west, east, south, north)):
            return None
        return [float(west), float(east), float(south), float(north)]
    except Exception:  # pylint: disable=broad-exception-caught
        return None


def _inspect_kea_rat(input_img: str) -> Optional[Dict[str, Any]]:
    """Inspect KEA/HDF5 Raster Attribute Table columns if available."""
    if os.path.splitext(input_img)[1].lower() != ".kea":
        return None
    try:
        import h5py  # pylint: disable=import-outside-toplevel
    except ImportError:
        return None

    try:
        with h5py.File(input_img, "r") as h5f:
            if "BAND1" not in h5f or "ATT" not in h5f["BAND1"]:
                return None
            att_grp = h5f["BAND1"]["ATT"]
            if "DATA" not in att_grp:
                return None
            cols = sorted(att_grp["DATA"].keys())
            n_rows = 0
            if cols:
                first_ds = att_grp["DATA"][cols[0]]
                n_rows = int(first_ds.shape[0]) if first_ds.shape else 0
            return {"n_rows": n_rows, "columns": cols}
    except Exception:  # pylint: disable=broad-exception-caught
        return None


def _compute_band_stats(
    ds: rasterio.DatasetReader,
    bidx: int,
    nodata: Optional[float],
) -> Dict[str, Optional[float]]:
    """Stream blocks of ``bidx`` with running min/max and Chan/Welford mean/std."""
    nodata_is_finite = nodata is not None and not (
        isinstance(nodata, float) and math.isnan(nodata)
    )
    count = 0
    min_val = math.inf
    max_val = -math.inf
    mean_val = 0.0
    m2_val = 0.0

    for _, window in ds.block_windows(bidx):
        block = ds.read(bidx, window=window, masked=False).astype(
            np.float64, copy=False
        )
        valid = np.isfinite(block)
        if nodata_is_finite:
            valid &= block != nodata
        if not np.any(valid):
            continue

        vals = block[valid]
        n_block = int(vals.size)
        blk_min = float(np.min(vals))
        blk_max = float(np.max(vals))
        blk_mean = float(np.mean(vals))
        blk_m2 = float(np.sum((vals - blk_mean) ** 2))

        if count == 0:
            min_val = blk_min
            max_val = blk_max
            mean_val = blk_mean
            m2_val = blk_m2
            count = n_block
        else:
            min_val = min(min_val, blk_min)
            max_val = max(max_val, blk_max)
            new_count = count + n_block
            delta = blk_mean - mean_val
            mean_val += delta * (n_block / new_count)
            m2_val += blk_m2 + (delta ** 2) * (count * n_block / new_count)
            count = new_count

    if count == 0:
        return {
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "median": None,
            "mode": None,
            "valid_percent": 0.0,
        }

    total_px = ds.width * ds.height
    valid_pct = 100.0 * float(count) / float(total_px) if total_px > 0 else 0.0
    return {
        "min": min_val,
        "max": max_val,
        "mean": mean_val,
        "std": math.sqrt(max(0.0, m2_val / count)),
        "median": None,
        "mode": None,
        "valid_percent": valid_pct,
    }


def _detect_is_tiled(
    ds: rasterio.DatasetReader,
    block_shape: Optional[List[int]],
    is_cog: bool,
) -> bool:
    """Detect tiled raster storage without calling deprecated ``ds.is_tiled``."""
    if is_cog or ds.driver in _TILED_DRIVERS:
        return True
    if block_shape is None or len(block_shape) != 2:
        return False
    blk_h, blk_w = block_shape
    if blk_w != ds.width:
        return True
    if blk_h == blk_w and blk_h >= 128 and (blk_h & (blk_h - 1)) == 0:
        return True
    return False


def _collect_band_info(
    ds: rasterio.DatasetReader,
    bidx: int,
    *,
    calc_stats: bool = False,
) -> Dict[str, Any]:
    """Collect complete metadata for a single 1-based band ``bidx``."""
    idx0 = bidx - 1
    b_tags = dict(ds.tags(bidx))
    nodata = ds.nodatavals[idx0]
    layer_type = b_tags.get("LAYER_TYPE")
    is_thematic = (layer_type or "").lower() == "thematic"

    cmap_len = 0
    try:
        cmap = ds.colormap(bidx)
        cmap_len = len(cmap)
    except ValueError:
        cmap_len = 0

    stats = {
        "min": _parse_float_tag(b_tags, "STATISTICS_MINIMUM"),
        "max": _parse_float_tag(b_tags, "STATISTICS_MAXIMUM"),
        "mean": _parse_float_tag(b_tags, "STATISTICS_MEAN"),
        "std": _parse_float_tag(b_tags, "STATISTICS_STDDEV"),
        "median": _parse_float_tag(b_tags, "STATISTICS_MEDIAN"),
        "mode": _parse_float_tag(b_tags, "STATISTICS_MODE"),
        "valid_percent": _parse_float_tag(b_tags, "STATISTICS_VALID_PERCENT"),
    }
    if calc_stats and stats["min"] is None and stats["max"] is None:
        stats = _compute_band_stats(ds, bidx, nodata)

    mask_flags = [flag.name for flag in ds.mask_flag_enums[idx0]]

    return {
        "band": bidx,
        "name": ds.descriptions[idx0] or None,
        "dtype": str(ds.dtypes[idx0]),
        "nodata": None if nodata is None else float(nodata),
        "colorinterp": ds.colorinterp[idx0].name,
        "layer_type": layer_type,
        "thematic": is_thematic,
        "colormap_entries": cmap_len,
        "scale": float(ds.scales[idx0]),
        "offset": float(ds.offsets[idx0]),
        "unit": ds.units[idx0] or None,
        "block_shape": list(ds.block_shapes[idx0]),
        "overviews": list(ds.overviews(bidx)),
        "mask_flags": mask_flags,
        "stats": stats,
        "tags": b_tags,
    }


def collect_img_info(
    input_img: str,
    *,
    calc_stats: bool = False,
) -> Dict[str, Any]:
    """
    Collect comprehensive raster metadata in a single ``rasterio.open`` pass.

    :param input_img: Path to the input raster file.
    :param calc_stats: If ``True``, compute min/max/mean/std from pixel data
                       for bands that lack stored statistics tags.
    :return: Dictionary containing structured raster metadata.
    """
    if not os.path.exists(input_img):
        raise RSGISPyException(f"Input image does not exist: '{input_img}'")

    try:
        file_size = os.path.getsize(input_img)
    except OSError:
        file_size = 0

    try:
        with rasterio.open(input_img, "r") as ds:
            img_struct = dict(ds.tags(ns="IMAGE_STRUCTURE"))
            ds_tags = dict(ds.tags())
            compression = (
                ds.compression.name.upper()
                if ds.compression is not None
                else img_struct.get("COMPRESSION")
            )
            interleave = (
                ds.interleaving.name.upper()
                if ds.interleaving is not None
                else img_struct.get("INTERLEAVE")
            )
            layout = img_struct.get("LAYOUT")
            area_or_point = ds_tags.get("AREA_OR_POINT")

            transform = ds.transform
            x_res, y_res = float(ds.res[0]), float(ds.res[1])
            bounds = ds.bounds
            bbox = [
                float(bounds.left),
                float(bounds.right),
                float(bounds.bottom),
                float(bounds.top),
            ]
            crs_info = _get_crs_summary(ds.crs)

            res_metres = None
            wgs84_bbox = None
            if crs_info["defined"]:
                if crs_info["is_geographic"]:
                    centre_lat = (bounds.bottom + bounds.top) / 2.0
                    xm, ym = degrees_to_metres(centre_lat, x_res, y_res)
                    res_metres = [float(xm), float(ym)]
                else:
                    wgs84_bbox = _compute_wgs84_bbox(
                        (bounds.left, bounds.bottom, bounds.right, bounds.top),
                        ds.crs,
                    )

            gcps_list, gcp_crs = ds.gcps
            gcp_info = None
            if gcps_list:
                gcp_auth = None
                if gcp_crs is not None:
                    try:
                        auth = pyproj.CRS.from_user_input(gcp_crs).to_authority()
                        gcp_auth = f"{auth[0]}:{auth[1]}" if auth else str(gcp_crs)
                    except Exception:  # pylint: disable=broad-exception-caught
                        gcp_auth = str(gcp_crs)
                gcp_info = {"count": len(gcps_list), "crs": gcp_auth}

            sidecars = [
                f for f in ds.files if os.path.abspath(f) != os.path.abspath(input_img)
            ]

            bands = [
                _collect_band_info(ds, bidx, calc_stats=calc_stats)
                for bidx in ds.indexes
            ]

            block_shape = list(ds.block_shapes[0]) if ds.block_shapes else None
            is_cog = (layout or "").upper() == "COG"
            is_tiled = _detect_is_tiled(ds, block_shape, is_cog)

            info: Dict[str, Any] = {
                "path": os.path.abspath(input_img),
                "filename": os.path.basename(input_img),
                "file_size_bytes": file_size,
                "file_size_human": format_bytes(file_size),
                "sidecar_files": sidecars,
                "driver": ds.driver,
                "layout": layout,
                "is_cog": is_cog,
                "compression": compression,
                "interleave": interleave,
                "is_tiled": is_tiled,
                "block_shape": block_shape,
                "area_or_point": area_or_point,
                "width": int(ds.width),
                "height": int(ds.height),
                "n_bands": int(ds.count),
                "dtypes": [str(dt) for dt in ds.dtypes],
                "res": [x_res, y_res],
                "res_metres_approx": res_metres,
                "origin": [float(transform.c), float(transform.f)],
                "rotation": [float(transform.b), float(transform.d)],
                "transform": list(transform)[:6],
                "bbox": bbox,
                "extent": [
                    float(abs(bounds.right - bounds.left)),
                    float(abs(bounds.top - bounds.bottom)),
                ],
                "wgs84_bbox": wgs84_bbox,
                "crs": crs_info,
                "gcps": gcp_info,
                "has_rpcs": ds.rpcs is not None,
                "dataset_tags": ds_tags,
                "image_structure": img_struct,
                "rat": _inspect_kea_rat(input_img),
                "bands": bands,
            }
    except RSGISPyException:
        raise
    except Exception as err:
        raise RSGISPyException(
            f"Could not read raster metadata for '{input_img}': {err}"
        ) from err

    return info


def _format_overviews_summary(bands: Sequence[Dict[str, Any]]) -> str:
    """Build a compact summary string of overview levels across bands."""
    if not bands:
        return "None"
    ovrs = [tuple(b["overviews"]) for b in bands]
    if all(len(o) == 0 for o in ovrs):
        return "None"
    unique = set(ovrs)
    if len(unique) == 1:
        levels = ovrs[0]
        return f"{len(levels)} levels ({', '.join(str(x) for x in levels)})"
    parts = [f"B{b['band']}:{len(b['overviews'])}" for b in bands]
    return "varies (" + ", ".join(parts) + ")"


def _format_mask_summary(bands: Sequence[Dict[str, Any]]) -> str:
    """Build a compact summary string of dataset mask flags."""
    if not bands:
        return "None"
    masks = [tuple(b["mask_flags"]) for b in bands]
    if len(set(masks)) == 1:
        return ", ".join(masks[0]) if masks[0] else "None"
    return "per-band"


def _format_theme_col(band: Dict[str, Any]) -> str:
    """Format the thematic/palette column cell for a band."""
    cmap_n = band["colormap_entries"]
    if band["thematic"] and cmap_n > 0:
        return f"thematic ({cmap_n})"
    if band["thematic"]:
        return "thematic"
    if cmap_n > 0:
        return f"palette ({cmap_n})"
    return "-"


def _format_bands_table(bands: Sequence[Dict[str, Any]]) -> str:
    """Render the compact per-band summary table."""
    if not bands:
        return "  (no bands)"

    show_scale = any(b["scale"] != 1.0 or b["offset"] != 0.0 for b in bands)
    show_unit = any(b["unit"] is not None for b in bands)
    show_theme = any(b["thematic"] or b["colormap_entries"] > 0 for b in bands)
    show_stats = any(
        b["stats"]["min"] is not None or b["stats"]["max"] is not None for b in bands
    )

    headers = ["#", "Name", "Type", "NoData", "Interp"]
    aligns = ["r", "l", "l", "r", "l"]

    if show_theme:
        headers.append("Theme/Clr")
        aligns.append("l")
    if show_scale:
        headers.append("Scale/Off")
        aligns.append("l")
    if show_unit:
        headers.append("Unit")
        aligns.append("l")
    if show_stats:
        headers.extend(["Min", "Max", "Mean", "Std"])
        aligns.extend(["r", "r", "r", "r"])

    rows: List[List[str]] = []
    for band in bands:
        row = [
            str(band["band"]),
            band["name"] or "-",
            band["dtype"],
            _fmt_num(band["nodata"]),
            band["colorinterp"],
        ]
        if show_theme:
            row.append(_format_theme_col(band))
        if show_scale:
            row.append(f"{_fmt_num(band['scale'])}/{_fmt_num(band['offset'])}")
        if show_unit:
            row.append(band["unit"] or "-")
        if show_stats:
            st = band["stats"]
            row.extend(
                [
                    _fmt_num(st["min"]),
                    _fmt_num(st["max"]),
                    _fmt_num(st["mean"]),
                    _fmt_num(st["std"]),
                ]
            )
        rows.append(row)

    return format_table(headers, rows, alignments=aligns, indent="  ")


def _format_metadata_section(
    info: Dict[str, Any],
    *,
    full_metadata: bool = False,
) -> List[str]:
    """Render full dataset, image-structure, and per-band metadata tags."""
    lines: List[str] = ["", "Metadata:"]
    if info["dataset_tags"]:
        lines.append("  Dataset:")
        for key, val in sorted(info["dataset_tags"].items()):
            lines.append(f"    {key} = {val}")
    if info["image_structure"]:
        lines.append("  Image Structure:")
        for key, val in sorted(info["image_structure"].items()):
            lines.append(f"    {key} = {val}")
    for band in info["bands"]:
        if not band["tags"]:
            continue
        lines.append(f"  Band {band['band']}:")
        for key, val in sorted(band["tags"].items()):
            disp = val
            if not full_metadata and key in _HISTO_TAG_KEYS and len(val) > 60:
                disp = f"{val[:57]}... ({len(val)} chars)"
            lines.append(f"    {key} = {disp}")
    return lines


def format_img_info(
    info: Dict[str, Any],
    *,
    show_wkt: bool = False,
    show_proj4: bool = False,
    show_metadata: bool = False,
    full_metadata: bool = False,
) -> str:
    """
    Format structured raster metadata into a compact, complete terminal report.

    :param info: Metadata dictionary returned by :func:`collect_img_info`.
    :param show_wkt: Append full pretty-printed CRS WKT.
    :param show_proj4: Append PROJ4 CRS string.
    :param show_metadata: Append dataset and per-band metadata tags.
    :param full_metadata: Do not truncate long histogram metadata values.
    :return: Formatted multi-line report string.
    """
    crs = info["crs"]
    lines: List[str] = [f"=== {info['filename']} ==="]

    # File line
    sidecar_note = ""
    if info["sidecar_files"]:
        names = ", ".join(os.path.basename(f) for f in info["sidecar_files"])
        sidecar_note = f" | Sidecars: {names}"
    lines.append(
        f"File      : {info['path']} ({info['file_size_human']}){sidecar_note}"
    )

    # Format / storage line
    drv = info["driver"] + (" [COG]" if info["is_cog"] else "")
    comp = info["compression"] or "None"
    interleave = info["interleave"] or "-"
    blk = info["block_shape"]
    if blk:
        tile_str = (
            f"Tiled ({blk[0]}x{blk[1]})"
            if info["is_tiled"]
            else f"Stripped ({blk[0]}x{blk[1]})"
        )
    else:
        tile_str = "Tiled" if info["is_tiled"] else "Stripped"
    fmt_parts = [drv, f"Compression: {comp}", f"Interleave: {interleave}", tile_str]
    if info["area_or_point"]:
        fmt_parts.append(info["area_or_point"])
    lines.append("Format    : " + " | ".join(fmt_parts))

    # Grid / dtype line
    unique_dtypes = list(dict.fromkeys(info["dtypes"]))
    dtype_str = unique_dtypes[0] if len(unique_dtypes) == 1 else ", ".join(unique_dtypes)
    band_lbl = "band" if info["n_bands"] == 1 else "bands"
    lines.append(
        f"Grid      : {info['width']} x {info['height']} px | "
        f"{info['n_bands']} {band_lbl} | {dtype_str}"
    )

    # Pixel size & origin line
    unit_short = "deg" if crs["is_geographic"] else (crs["unit"] or "px")
    rx, ry = info["res"]
    res_str = f"{_fmt_num(rx, 6)} x {_fmt_num(ry, 6)} {unit_short}"
    if info["res_metres_approx"] is not None:
        mx, my = info["res_metres_approx"]
        res_str += f" (~{_fmt_num(mx, 4)} x {_fmt_num(my, 4)} m)"
    ox, oy = info["origin"]
    rot_b, rot_d = info["rotation"]
    rot_str = ""
    if rot_b != 0.0 or rot_d != 0.0:
        rot_str = f" | Rotation: ({_fmt_num(rot_b)}, {_fmt_num(rot_d)})"
    lines.append(
        f"Pixel Size: {res_str} | Origin: ({_fmt_num(ox, 8)}, {_fmt_num(oy, 8)}){rot_str}"
    )

    # Spatial extent line (+ WGS84 bbox for projected images)
    min_x, max_x, min_y, max_y = info["bbox"]
    ext_w, ext_h = info["extent"]
    lines.append(
        f"Extent    : X [{_fmt_num(min_x, 8)}, {_fmt_num(max_x, 8)}]  "
        f"Y [{_fmt_num(min_y, 8)}, {_fmt_num(max_y, 8)}]  "
        f"({_fmt_num(ext_w, 6)} x {_fmt_num(ext_h, 6)} {unit_short})"
    )
    if info["wgs84_bbox"] is not None:
        w_min_x, w_max_x, w_min_y, w_max_y = info["wgs84_bbox"]
        lines.append(
            f"WGS84 BBox: Lon [{_fmt_num(w_min_x, 6)}, {_fmt_num(w_max_x, 6)}]  "
            f"Lat [{_fmt_num(w_min_y, 6)}, {_fmt_num(w_max_y, 6)}]"
        )

    # CRS line
    if crs["defined"]:
        auth_prefix = f"{crs['authority']} — " if crs["authority"] else ""
        detail_parts = [p for p in (crs["type"], crs["unit"]) if p]
        detail_str = f" ({', '.join(detail_parts)})" if detail_parts else ""
        lines.append(f"CRS       : {auth_prefix}{crs['name'] or 'Custom'}{detail_str}")
    else:
        lines.append("CRS       : None (unprojected)")

    # Overviews, masks, GCPs/RPCs, RAT
    struct_parts = [
        f"Overviews : {_format_overviews_summary(info['bands'])}",
        f"Mask: {_format_mask_summary(info['bands'])}",
    ]
    if info["gcps"] is not None:
        gcp_crs = f" ({info['gcps']['crs']})" if info["gcps"]["crs"] else ""
        struct_parts.append(f"GCPs: {info['gcps']['count']}{gcp_crs}")
    if info["has_rpcs"]:
        struct_parts.append("RPCs: yes")
    lines.append(" | ".join(struct_parts))

    if info["rat"] is not None:
        rat_cols = info["rat"]["columns"]
        preview = ", ".join(rat_cols[:8])
        if len(rat_cols) > 8:
            preview += f", ... (+{len(rat_cols) - 8})"
        lines.append(
            f"RAT       : {info['rat']['n_rows']} rows, "
            f"{len(rat_cols)} cols ({preview})"
        )

    # Compact extra dataset tags (excluding AREA_OR_POINT already shown)
    extra_tags = {
        k: v for k, v in info["dataset_tags"].items() if k != "AREA_OR_POINT"
    }
    if extra_tags and not show_metadata:
        tag_items = [
            f"{k}={v if len(v) <= 24 else v[:21] + '...'}"
            for k, v in list(sorted(extra_tags.items()))[:4]
        ]
        suffix = f" (+{len(extra_tags) - 4} more)" if len(extra_tags) > 4 else ""
        lines.append(f"Tags      : {', '.join(tag_items)}{suffix}")

    # Compact per-band table
    lines.append("")
    lines.append("Bands:")
    lines.append(_format_bands_table(info["bands"]))

    if show_proj4 and crs["proj4"]:
        lines.extend(["", "PROJ4:", f"  {crs['proj4']}"])

    if show_wkt and crs["wkt_pretty"]:
        lines.extend(["", "WKT:", crs["wkt_pretty"]])

    if show_metadata:
        lines.extend(_format_metadata_section(info, full_metadata=full_metadata))

    return "\n".join(lines)


def build_imginfo_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for ``rsgisimginfo``."""
    parser = build_parser(
        prog="rsgisimginfo",
        description=(
            "Display complete, compact metadata for one or more raster images "
            "(format, grid, pixel size, extent, CRS, overviews, and per-band table)."
        ),
    )
    add_input_files_arg(parser, "Input raster image path(s) or glob pattern(s).")
    parser.add_argument(
        "--wkt",
        action="store_true",
        default=False,
        help="Include full pretty-printed CRS WKT representation.",
    )
    parser.add_argument(
        "--proj4",
        action="store_true",
        default=False,
        help="Include PROJ4 coordinate reference system string.",
    )
    parser.add_argument(
        "--metadata",
        action="store_true",
        default=False,
        help="Include dataset, image-structure, and per-band metadata tags.",
    )
    parser.add_argument(
        "--full-metadata",
        action="store_true",
        default=False,
        help="Include untruncated metadata values (implies --metadata).",
    )
    parser.add_argument(
        "--calc-stats",
        action="store_true",
        default=False,
        help="Compute min/max/mean/std from pixels for bands lacking stored statistics.",
    )
    add_json_arg(parser, "Emit complete raster metadata as JSON.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Execute ``rsgisimginfo`` and return an exit status code."""
    parser = build_imginfo_parser()
    args = parser.parse_args(argv)

    imgs = resolve_input_files(args.input, natural_sort=True)
    infos = [collect_img_info(img, calc_stats=args.calc_stats) for img in imgs]

    if args.json:
        payload: Any = infos[0] if len(infos) == 1 else infos
        print(json.dumps(payload, indent=2))
        return 0

    show_meta = args.metadata or args.full_metadata
    blocks = [
        format_img_info(
            info,
            show_wkt=args.wkt,
            show_proj4=args.proj4,
            show_metadata=show_meta,
            full_metadata=args.full_metadata,
        )
        for info in infos
    ]
    print("\n\n".join(blocks))
    return 0


def cli_main() -> int:
    """Console script entry point for ``rsgisimginfo``."""
    return run_cli(main)


if __name__ == "__main__":
    sys.exit(cli_main())
