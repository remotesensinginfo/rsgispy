"""
Command-line tool ``rsgiscalcimgstats``: populate continuous or thematic image
statistics, histograms, and pyramids for one or more raster images.
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

from rsgispy import RSGISPyException
from rsgispy.cli._common import (
    add_input_files_arg,
    add_n_cores_arg,
    add_no_pbar_arg,
    build_parser,
    resolve_input_files,
    run_cli,
)
from rsgispy.imageutils.stats import pop_img_stats, pop_thmt_img_stats


def build_calcimgstats_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for ``rsgiscalcimgstats``."""
    parser = build_parser(
        prog="rsgiscalcimgstats",
        description=(
            "Calculate image statistics, histograms, and image pyramids for "
            "one or more continuous or thematic raster images."
        ),
    )
    add_input_files_arg(parser, "Input raster image path(s) or glob pattern(s).")
    parser.add_argument(
        "-t",
        "--thematic",
        action="store_true",
        default=False,
        help="Treat input image(s) as thematic integer classification rasters.",
    )
    parser.add_argument(
        "-n",
        "--nodata",
        type=float,
        default=None,
        metavar="VALUE",
        help=(
            "No-data value for continuous images (defaults to the image "
            "header no-data value if omitted)."
        ),
    )
    parser.add_argument(
        "--include-zero",
        action="store_true",
        default=False,
        help="For thematic images, include pixel value 0 in statistics.",
    )
    parser.add_argument(
        "--no-pyramids",
        action="store_true",
        default=False,
        help="Do not build image pyramids / overviews.",
    )
    parser.add_argument(
        "--no-clrtab",
        action="store_true",
        default=False,
        help="For thematic images, do not populate a colour table.",
    )
    parser.add_argument(
        "--approx",
        action="store_true",
        default=False,
        help="Compute approximate statistics (continuous images only).",
    )
    parser.add_argument(
        "--ignore-cog",
        action="store_true",
        default=False,
        help="Ignore COG layout check when populating continuous statistics.",
    )
    add_n_cores_arg(
        parser,
        default=1,
        help_text="Number of worker cores for multi-core pyramid generation.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=1024,
        metavar="PX",
        help="Processing tile size in pixels.",
    )
    add_no_pbar_arg(parser)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Execute ``rsgiscalcimgstats`` and return an exit status code."""
    parser = build_calcimgstats_parser()
    args = parser.parse_args(argv)

    if args.n_cores < 1:
        raise RSGISPyException("--n-cores must be >= 1.")
    if args.tile_size < 1:
        raise RSGISPyException("--tile-size must be >= 1.")

    imgs = resolve_input_files(args.input, natural_sort=True)
    calc_pyramids = not args.no_pyramids
    use_pbar = not args.no_pbar
    multi_core = args.n_cores > 1

    for img in imgs:
        if args.thematic:
            pop_thmt_img_stats(
                input_img=img,
                add_clr_tab=not args.no_clrtab,
                calc_pyramids=calc_pyramids,
                ignore_zero=not args.include_zero,
                use_pbar=use_pbar,
                use_multi_core_pyramids=multi_core,
                n_cores=args.n_cores,
                tile_size=args.tile_size,
            )
        else:
            pop_img_stats(
                input_img=img,
                no_data_val=args.nodata,
                calc_pyramids=calc_pyramids,
                ignore_cog=args.ignore_cog,
                use_approx=args.approx,
                use_pbar=use_pbar,
                use_multi_core_pyramids=multi_core,
                n_cores=args.n_cores,
                tile_size=args.tile_size,
            )
    return 0


def cli_main() -> int:
    """Console script entry point for ``rsgiscalcimgstats``."""
    return run_cli(main)


if __name__ == "__main__":
    sys.exit(cli_main())
