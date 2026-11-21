"""
Command-line tool ``rsgischkgdalfile``: validate one or more GDAL raster or
vector datasets and optionally delete corrupt files.
"""

from __future__ import annotations

import argparse
import datetime
import sys
from typing import Optional, Sequence

from rsgispy import RSGISPyException
from rsgispy.cli._common import (
    add_input_files_arg,
    build_parser,
    resolve_input_files,
    run_cli,
)
from rsgispy.tools.checkdatasets import (
    run_check_gdal_image_files,
    run_check_gdal_vector_files,
)
from rsgispy.tools.files import get_files_mod_time


def _parse_mod_date(date_str: str) -> datetime.datetime:
    """Parse a ``YYYY-MM-DD`` date string into a ``datetime`` at midnight."""
    try:
        return datetime.datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError as err:
        raise RSGISPyException(
            f"Invalid --mod-date '{date_str}': expected format YYYY-MM-DD."
        ) from err


def build_chkgdalfile_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for ``rsgischkgdalfile``."""
    parser = build_parser(
        prog="rsgischkgdalfile",
        description=(
            "Validate GDAL raster or vector datasets (header, projection, "
            "bands, pixel readability, checksums, and file size)."
        ),
    )
    add_input_files_arg(parser, "Input file path(s) or glob pattern(s).")
    parser.add_argument(
        "--type",
        type=str.lower,
        choices=["raster", "vector"],
        default="raster",
        help="Dataset type to validate.",
    )
    parser.add_argument(
        "--no-sort",
        action="store_true",
        default=False,
        help="Do not naturally sort matched input files.",
    )
    parser.add_argument(
        "--mod-date",
        type=str,
        default=None,
        metavar="YYYY-MM-DD",
        help="Only check files modified on or after this date.",
    )
    parser.add_argument(
        "--rm-err",
        action="store_true",
        default=False,
        help="Delete files that fail validation.",
    )
    parser.add_argument(
        "--print-names",
        action="store_true",
        default=False,
        help="Print each filename as it is checked.",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        default=False,
        help="Suppress error detail messages for failing files.",
    )
    parser.add_argument(
        "--chk-proj",
        action="store_true",
        default=False,
        help="Verify that datasets have a valid CRS/projection.",
    )
    parser.add_argument(
        "--epsg",
        type=int,
        default=0,
        metavar="CODE",
        help="Expected EPSG code (0 disables specific EPSG check).",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=None,
        metavar="BYTES",
        help="Maximum expected file size in bytes.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=4,
        metavar="SEC",
        help="Per-file validation timeout in seconds.",
    )

    raster_grp = parser.add_argument_group("raster options")
    raster_grp.add_argument(
        "--no-band-check",
        action="store_true",
        default=False,
        help="Skip band-count validation.",
    )
    raster_grp.add_argument(
        "--n-bands",
        type=int,
        default=0,
        metavar="N",
        help="Expected number of bands (0 allows any >= 1).",
    )
    raster_grp.add_argument(
        "--read-img",
        action="store_true",
        default=False,
        help="Read sample pixels to verify raster data integrity.",
    )
    raster_grp.add_argument(
        "--sample-pixels",
        type=int,
        default=10,
        metavar="N",
        help="Number of sample pixels to read with --read-img.",
    )
    raster_grp.add_argument(
        "--chk-sum",
        action="store_true",
        default=False,
        help="Compute pixel checksums across bands.",
    )

    vec_grp = parser.add_argument_group("vector options")
    vec_grp.add_argument(
        "--multi-file",
        action="store_true",
        default=False,
        help="Remove companion sidecar files when deleting failed multi-file vectors.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Execute ``rsgischkgdalfile`` and return ``0`` if all pass else ``1``."""
    parser = build_chkgdalfile_parser()
    args = parser.parse_args(argv)

    if args.timeout < 1:
        raise RSGISPyException("--timeout must be >= 1.")

    files = resolve_input_files(args.input, natural_sort=not args.no_sort)

    if args.mod_date is not None:
        dt_after = _parse_mod_date(args.mod_date)
        files = get_files_mod_time(files, dt_after=dt_after)
        if not files:
            print(f"No files modified on or after {args.mod_date}.")
            return 0

    chk_proj = args.chk_proj or (args.epsg > 0)
    print_err = not args.quiet

    if args.type == "vector":
        all_ok = run_check_gdal_vector_files(
            vec_files=files,
            chk_proj=chk_proj,
            epsg_code=args.epsg,
            max_file_size=args.max_size,
            rm_err=args.rm_err,
            print_err=print_err,
            multi_file=args.multi_file,
            print_file_names=args.print_names,
            timeout=args.timeout,
        )
    else:
        all_ok = run_check_gdal_image_files(
            input_imgs=files,
            check_bands=not args.no_band_check,
            n_bands=args.n_bands,
            chk_proj=chk_proj,
            epsg_code=args.epsg,
            read_img=args.read_img,
            smpl_n_pxls=args.sample_pixels,
            calc_chk_sum=args.chk_sum,
            max_file_size=args.max_size,
            rm_err=args.rm_err,
            print_err=print_err,
            print_file_names=args.print_names,
            timeout=args.timeout,
        )

    return 0 if all_ok else 1


def cli_main() -> int:
    """Console script entry point for ``rsgischkgdalfile``."""
    return run_cli(main)


if __name__ == "__main__":
    sys.exit(cli_main())
