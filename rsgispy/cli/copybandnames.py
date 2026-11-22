"""
Command-line tool ``rsgiscopybandnames``: copy band names (descriptions) from
a reference raster image to one or more target raster images.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Sequence

import rasterio

from rsgispy import RSGISPyException
from rsgispy.cli._common import (
    add_input_files_arg,
    build_parser,
    resolve_input_files,
    run_cli,
)
from rsgispy.imageutils.metadata import get_band_names, set_band_names

__all__ = [
    "build_copybandnames_parser",
    "main",
    "cli_main",
]


def build_copybandnames_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for ``rsgiscopybandnames``."""
    parser = build_parser(
        prog="rsgiscopybandnames",
        description=(
            "Copy band names (descriptions) from a reference raster image to "
            "one or more target raster images."
        ),
    )
    parser.add_argument(
        "-r",
        "--ref",
        type=str,
        required=True,
        metavar="PATH",
        help="Reference raster image to read band names from.",
    )
    add_input_files_arg(
        parser,
        "Target raster image path(s) or glob pattern(s) to update.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Print band names copied to each target image.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Execute ``rsgiscopybandnames`` and return an exit status code."""
    parser = build_copybandnames_parser()
    args = parser.parse_args(argv)

    ref_img = os.path.expanduser(args.ref)
    if not os.path.exists(ref_img):
        raise RSGISPyException(f"Reference image does not exist: '{args.ref}'")

    band_names = get_band_names(ref_img)
    target_imgs = resolve_input_files(args.input, natural_sort=True)

    for target_img in target_imgs:
        with rasterio.open(target_img, "r") as ds:
            n_bands = ds.count
        if n_bands != len(band_names):
            raise RSGISPyException(
                f"Band count mismatch: reference '{args.ref}' has "
                f"{len(band_names)} band(s), but target '{target_img}' has "
                f"{n_bands} band(s)."
            )
        set_band_names(target_img, band_names, feedback=args.verbose)

    return 0


def cli_main() -> int:
    """Console script entry point for ``rsgiscopybandnames``."""
    return run_cli(main)


if __name__ == "__main__":
    sys.exit(cli_main())
