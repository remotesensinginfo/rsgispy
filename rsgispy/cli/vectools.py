"""
Command-line tool ``rsgisvectools``: inspect vector files (GeoPackage,
Shapefile, GeoJSON, FlatGeobuf, GeoParquet, etc.) and print metadata or
attribute summaries.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

import pyproj

from rsgispy import RSGISPyException
from rsgispy.cli._common import (
    add_input_file_arg,
    add_json_arg,
    add_layer_arg,
    build_parser,
    format_bytes,
    run_cli,
)
from rsgispy.vector.attrs import get_unq_col_values
from rsgispy.vector.io import (
    get_geom_type_name,
    get_vec_feat_count,
    get_vec_lyr_bbox,
    get_vec_lyr_cols,
    get_vec_lyr_geom_type,
    get_vec_lyrs_lst,
    get_wkt_proj_from_vec,
)


def _crs_label_from_wkt(wkt_str: str) -> str:
    """Return a concise CRS description string from a WKT representation."""
    if not wkt_str:
        return "None"
    try:
        crs = pyproj.CRS.from_wkt(wkt_str)
        auth = crs.to_authority()
        auth_prefix = f"{auth[0]}:{auth[1]} — " if auth else ""
        return f"{auth_prefix}{crs.name}"
    except Exception:  # pylint: disable=broad-exception-caught
        return wkt_str.splitlines()[0][:80]


def _json_safe(val: Any) -> Any:
    """Convert numpy scalar values into JSON-serialisable Python types."""
    if hasattr(val, "item"):
        return val.item()
    return val


def build_vectools_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for ``rsgisvectools``."""
    parser = build_parser(
        prog="rsgisvectools",
        description=(
            "Inspect vector datasets (OGR formats and GeoParquet). If no "
            "specific query flag is given, prints a compact summary of the "
            "vector file and layer."
        ),
    )
    add_input_file_arg(parser, "Input vector file path.")
    add_layer_arg(parser)
    parser.add_argument(
        "-c",
        "--column",
        type=str,
        default=None,
        metavar="COL",
        help="Attribute column name (used with --unique-vals).",
    )
    parser.add_argument(
        "--layers",
        action="store_true",
        default=False,
        help="List all layers in the vector file.",
    )
    parser.add_argument(
        "--columns",
        action="store_true",
        default=False,
        help="List attribute column names in the selected layer.",
    )
    parser.add_argument(
        "--n-feats",
        action="store_true",
        default=False,
        help="Print the number of features in the selected layer.",
    )
    parser.add_argument(
        "--proj",
        action="store_true",
        default=False,
        help="Print the CRS WKT string of the selected layer.",
    )
    parser.add_argument(
        "--bbox",
        action="store_true",
        default=False,
        help="Print the bounding box [minX, maxX, minY, maxY] of the layer.",
    )
    parser.add_argument(
        "--geom",
        action="store_true",
        default=False,
        help="Print the geometry type of the selected layer.",
    )
    parser.add_argument(
        "--unique-vals",
        action="store_true",
        default=False,
        help="Print unique values in --column.",
    )
    add_json_arg(parser, "Emit query results as JSON.")
    return parser


def _print_summary(
    vec_file: str,
    vec_lyr: Optional[str],
    *,
    as_json: bool = False,
) -> None:
    """Collect and display a compact all-in-one vector summary."""
    lyrs = get_vec_lyrs_lst(vec_file)
    active_lyr = vec_lyr or (lyrs[0] if lyrs else None)
    cols = get_vec_lyr_cols(vec_file, active_lyr)
    n_feats = get_vec_feat_count(vec_file, active_lyr)
    geom_type = get_geom_type_name(get_vec_lyr_geom_type(vec_file, active_lyr))
    bbox = list(get_vec_lyr_bbox(vec_file, active_lyr))
    wkt = get_wkt_proj_from_vec(vec_file, active_lyr)
    size_bytes = os.path.getsize(vec_file)

    summary: Dict[str, Any] = {
        "path": os.path.abspath(vec_file),
        "file_size_bytes": size_bytes,
        "layers": lyrs,
        "layer": active_lyr,
        "geometry_type": geom_type,
        "n_features": int(n_feats),
        "bbox": [float(v) for v in bbox],
        "crs": _crs_label_from_wkt(wkt),
        "columns": cols,
    }
    if as_json:
        print(json.dumps(summary, indent=2))
        return

    print(f"=== {os.path.basename(vec_file)} ===")
    print(f"File      : {summary['path']} ({format_bytes(size_bytes)})")
    print(f"Layers    : {', '.join(lyrs) if lyrs else '-'} (active: {active_lyr or '-'})")
    print(f"Geometry  : {geom_type} | Features: {n_feats}")
    print(
        f"Extent    : X [{bbox[0]:.6g}, {bbox[1]:.6g}]  "
        f"Y [{bbox[2]:.6g}, {bbox[3]:.6g}]"
    )
    print(f"CRS       : {summary['crs']}")
    print(f"Columns ({len(cols)}): {', '.join(cols) if cols else '-'}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Execute ``rsgisvectools`` and return an exit status code."""
    parser = build_vectools_parser()
    args = parser.parse_args(argv)

    vec_file = os.path.expanduser(args.input)
    if not os.path.exists(vec_file):
        raise RSGISPyException(f"Input vector file does not exist: '{args.input}'")

    any_query = any(
        (
            args.layers,
            args.columns,
            args.n_feats,
            args.proj,
            args.bbox,
            args.geom,
            args.unique_vals,
        )
    )

    if not any_query:
        _print_summary(vec_file, args.layer, as_json=args.json)
        return 0

    if args.unique_vals and not args.column:
        raise RSGISPyException("--column is required when using --unique-vals.")

    results: Dict[str, Any] = {}
    output_lines: List[str] = []

    if args.layers:
        lyrs = get_vec_lyrs_lst(vec_file)
        results["layers"] = lyrs
        output_lines.extend(lyrs)

    if args.columns:
        cols = get_vec_lyr_cols(vec_file, args.layer)
        results["columns"] = cols
        output_lines.extend(cols)

    if args.n_feats:
        n_feats = int(get_vec_feat_count(vec_file, args.layer))
        results["n_features"] = n_feats
        output_lines.append(str(n_feats))

    if args.proj:
        wkt = get_wkt_proj_from_vec(vec_file, args.layer)
        results["wkt"] = wkt
        output_lines.append(wkt)

    if args.bbox:
        bbox = [float(v) for v in get_vec_lyr_bbox(vec_file, args.layer)]
        results["bbox"] = bbox
        output_lines.append(str(bbox))

    if args.geom:
        geom = get_geom_type_name(get_vec_lyr_geom_type(vec_file, args.layer))
        results["geometry_type"] = geom
        output_lines.append(geom)

    if args.unique_vals:
        unq = [_json_safe(v) for v in get_unq_col_values(vec_file, args.layer, args.column)]
        results["unique_values"] = unq
        output_lines.extend(str(v) for v in unq)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for line in output_lines:
            print(line)
    return 0


def cli_main() -> int:
    """Console script entry point for ``rsgisvectools``."""
    return run_cli(main)


if __name__ == "__main__":
    sys.exit(cli_main())
