"""
Command-line tool ``rsgisbuildimglut``: build a vector bounding-box look-up
table (LUT) from one or more input raster images.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Sequence

from rsgispy.cli._common import (
    add_format_arg,
    add_input_files_arg,
    add_layer_arg,
    add_no_pbar_arg,
    add_output_arg,
    build_parser,
    resolve_input_files,
    run_cli,
)
from rsgispy.imageutils.imagelut import create_img_extent_lut


def build_buildimglut_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for ``rsgisbuildimglut``."""
    parser = build_parser(
        prog="rsgisbuildimglut",
        description=(
            "Build a vector spatial look-up table (LUT) containing the "
            "bounding-box footprints of input raster images."
        ),
    )
    add_input_files_arg(parser, "Input raster image path(s) or glob pattern(s).")
    add_output_arg(parser, "Output vector LUT file path.")
    add_layer_arg(
        parser,
        help_text="Output vector layer name (defaults to output file basename).",
    )
    add_format_arg(
        parser,
        default="GPKG",
        help_text="Output vector format (e.g., GPKG, GeoJSON, ESRI Shapefile, Parquet).",
    )
    parser.add_argument(
        "--wgs84",
        action="store_true",
        default=False,
        help="Reproject image extents to WGS84 (EPSG:4326).",
    )
    parser.add_argument(
        "--ignore-noproj",
        action="store_true",
        default=False,
        help="Skip input images that do not have a valid projection.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite the output LUT file if it already exists.",
    )
    add_no_pbar_arg(parser)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Execute ``rsgisbuildimglut`` and return an exit status code."""
    parser = build_buildimglut_parser()
    args = parser.parse_args(argv)

    in_imgs = resolve_input_files(args.input, natural_sort=True)
    out_vec = os.path.expanduser(args.output)

    create_img_extent_lut(
        input_imgs=in_imgs,
        vec_file=out_vec,
        vec_lyr=args.layer,
        out_format=args.format,
        ignore_none_imgs=args.ignore_noproj,
        out_proj_wgs84=args.wgs84,
        overwrite_lut_file=args.overwrite,
        use_pbar=not args.no_pbar,
    )
    return 0


def cli_main() -> int:
    """Console script entry point for ``rsgisbuildimglut``."""
    return run_cli(main)


if __name__ == "__main__":
    sys.exit(cli_main())
