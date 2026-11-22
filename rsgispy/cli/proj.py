"""
Command-line tool ``rsgisproj``: inspect and convert coordinate reference
systems between raster/vector datasets, EPSG codes, WKT, PROJ4, ESRI WKT,
and UTM zones using pure ``pyproj`` and ``rasterio``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import warnings
from typing import Dict, List, Optional, Sequence

import pyproj
import rasterio

from rsgispy import RSGISPyException
from rsgispy.cli._common import (
    add_layer_arg,
    add_output_arg,
    build_parser,
    run_cli,
)
from rsgispy.vector.io import get_wkt_proj_from_vec

__all__ = [
    "build_proj_parser",
    "main",
    "cli_main",
]

_UTM_RE = re.compile(r"^(\d{1,2})([NSns])$")
_OUTPUT_FORMATS = ("epsg", "wkt", "wkt2", "wkt-pretty", "proj4", "json")


def _read_text_file(path: str) -> str:
    """Read and strip a UTF-8 text file, raising ``RSGISPyException`` if missing."""
    expanded = os.path.expanduser(path)
    if not os.path.exists(expanded):
        raise RSGISPyException(f"Input CRS file does not exist: '{path}'")
    with open(expanded, "r", encoding="utf-8") as handle:
        content = handle.read().strip()
    if not content:
        raise RSGISPyException(f"Input CRS file is empty: '{path}'")
    return content


def _crs_from_utm(utm_str: str) -> pyproj.CRS:
    """Parse a WGS84 UTM zone specification (e.g., ``36N`` or ``01S``)."""
    match = _UTM_RE.match(utm_str.strip())
    if not match:
        raise RSGISPyException(
            f"Invalid --utm specification '{utm_str}': expected <zone><N|S> "
            "(e.g., 36N or 01S)."
        )
    zone = int(match.group(1))
    hemi = match.group(2).upper()
    if not 1 <= zone <= 60:
        raise RSGISPyException(f"UTM zone must be between 1 and 60 (got {zone}).")
    epsg = (32600 if hemi == "N" else 32700) + zone
    return pyproj.CRS.from_epsg(epsg)


def _load_crs(args: argparse.Namespace) -> pyproj.CRS:
    """Resolve the input ``pyproj.CRS`` from parsed CLI arguments."""
    try:
        if args.image:
            img_path = os.path.expanduser(args.image)
            if not os.path.exists(img_path):
                raise RSGISPyException(f"Input image does not exist: '{args.image}'")
            with rasterio.open(img_path, "r") as ds:
                if ds.crs is None:
                    raise RSGISPyException(
                        f"Input image '{args.image}' does not have a CRS defined."
                    )
                return pyproj.CRS.from_user_input(ds.crs)

        if args.vector:
            vec_path = os.path.expanduser(args.vector)
            if not os.path.exists(vec_path):
                raise RSGISPyException(f"Input vector does not exist: '{args.vector}'")
            wkt = get_wkt_proj_from_vec(vec_path, args.layer)
            if not wkt:
                raise RSGISPyException(
                    f"Input vector '{args.vector}' does not have a CRS defined."
                )
            return pyproj.CRS.from_wkt(wkt)

        if args.epsg is not None:
            return pyproj.CRS.from_epsg(args.epsg)
        if args.wkt is not None:
            return pyproj.CRS.from_wkt(args.wkt)
        if args.wkt_file is not None:
            return pyproj.CRS.from_wkt(_read_text_file(args.wkt_file))
        if args.proj4 is not None:
            return pyproj.CRS.from_proj4(args.proj4)
        if args.proj4_file is not None:
            return pyproj.CRS.from_proj4(_read_text_file(args.proj4_file))
        if args.esri_file is not None:
            return pyproj.CRS.from_user_input(_read_text_file(args.esri_file))
        if args.utm is not None:
            return _crs_from_utm(args.utm)
    except RSGISPyException:
        raise
    except Exception as err:
        raise RSGISPyException(f"Could not parse input CRS: {err}") from err

    raise RSGISPyException("No CRS input source was specified.")


def _format_crs(crs: pyproj.CRS, fmt: str) -> str:
    """Format a ``pyproj.CRS`` into the requested output representation."""
    try:
        if fmt == "epsg":
            epsg = crs.to_epsg()
            if epsg is None:
                auth = crs.to_authority()
                return f"{auth[0]}:{auth[1]}" if auth else "None"
            return f"EPSG:{epsg}"
        if fmt == "wkt":
            return crs.to_wkt(version="WKT1_GDAL")
        if fmt == "wkt2":
            return crs.to_wkt(version="WKT2_2019")
        if fmt == "wkt-pretty":
            return crs.to_wkt(pretty=True)
        if fmt == "proj4":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                proj4_str = crs.to_proj4()
            if not proj4_str:
                raise RSGISPyException(
                    f"CRS '{crs.name}' cannot be represented as a PROJ4 string."
                )
            return proj4_str
        if fmt == "json":
            return json.dumps(crs.to_json_dict(), indent=2)
    except pyproj.exceptions.CRSError as err:
        raise RSGISPyException(
            f"Could not format CRS as '{fmt}': {err}"
        ) from err
    raise RSGISPyException(f"Unsupported output format: '{fmt}'")


def build_proj_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for ``rsgisproj``."""
    parser = build_parser(
        prog="rsgisproj",
        description=(
            "Convert or inspect a Coordinate Reference System (CRS) from a "
            "raster, vector, EPSG code, WKT, PROJ4, ESRI WKT, or UTM zone."
        ),
    )
    src_group = parser.add_mutually_exclusive_group(required=True)
    src_group.add_argument(
        "--image",
        type=str,
        default=None,
        metavar="PATH",
        help="Read CRS from a raster image file.",
    )
    src_group.add_argument(
        "--vector",
        type=str,
        default=None,
        metavar="PATH",
        help="Read CRS from a vector dataset file.",
    )
    src_group.add_argument(
        "--epsg",
        type=int,
        default=None,
        metavar="CODE",
        help="Input EPSG code (e.g., 4326).",
    )
    src_group.add_argument(
        "--wkt",
        type=str,
        default=None,
        metavar="WKT",
        help="Input WKT CRS string.",
    )
    src_group.add_argument(
        "--wkt-file",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to text file containing a WKT CRS string.",
    )
    src_group.add_argument(
        "--proj4",
        type=str,
        default=None,
        metavar="PROJ4",
        help="Input PROJ4 CRS string.",
    )
    src_group.add_argument(
        "--proj4-file",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to text file containing a PROJ4 CRS string.",
    )
    src_group.add_argument(
        "--esri-file",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to ESRI .prj WKT file.",
    )
    src_group.add_argument(
        "--utm",
        type=str,
        default=None,
        metavar="ZONE",
        help="WGS84 UTM zone specification (e.g., 36N or 01S).",
    )

    add_layer_arg(parser, "Vector layer name (used only with --vector).")
    parser.add_argument(
        "-f",
        "--format",
        type=str.lower,
        nargs="+",
        choices=list(_OUTPUT_FORMATS),
        default=["wkt-pretty"],
        help="Output CRS format(s).",
    )
    add_output_arg(
        parser,
        help_text="Optional output file path (writes to stdout if omitted).",
        required=False,
        default=None,
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Execute ``rsgisproj`` and return an exit status code."""
    parser = build_proj_parser()
    args = parser.parse_args(argv)

    crs = _load_crs(args)
    formats: List[str] = list(dict.fromkeys(args.format))

    rendered_map: Dict[str, str] = {fmt: _format_crs(crs, fmt) for fmt in formats}
    if len(formats) == 1:
        text = rendered_map[formats[0]]
    else:
        sections = [f"[{fmt.upper()}]\n{rendered_map[fmt]}" for fmt in formats]
        text = "\n\n".join(sections)

    if args.output:
        out_path = os.path.expanduser(args.output)
        parent = os.path.dirname(os.path.abspath(out_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    else:
        print(text)
    return 0


def cli_main() -> int:
    """Console script entry point for ``rsgisproj``."""
    return run_cli(main)


if __name__ == "__main__":
    sys.exit(cli_main())
