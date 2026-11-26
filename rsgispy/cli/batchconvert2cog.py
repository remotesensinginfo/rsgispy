"""
Command-line tool ``rsgisbatchconvert2cog``: convert one or more raster images
into Cloud-Optimised GeoTIFFs (COGs) with optional parallel execution.
"""

from __future__ import annotations

import argparse
from multiprocessing import Pool
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import rasterio

import rsgispy
from rsgispy import RSGISPyException
from rsgispy.cli._common import (
    add_input_files_arg,
    add_n_cores_arg,
    add_no_pbar_arg,
    add_output_arg,
    build_parser,
    resolve_input_files,
    run_cli,
)
from rsgispy.imageutils.colourtable import is_img_thematic
from rsgispy.imageutils.warp import gdal_translate
from rsgispy.tools.files import get_file_basename

__all__ = [
    "build_batchconvert2cog_parser",
    "main",
    "cli_main",
]


def _is_valid_raster(img_path: str) -> bool:
    """Return True if ``img_path`` exists and can be opened cleanly by rasterio."""
    if not os.path.exists(img_path) or os.path.getsize(img_path) == 0:
        return False
    try:
        with rasterio.open(img_path, "r") as ds:
            return ds.count > 0 and ds.width > 0 and ds.height > 0
    except Exception:  # pylint: disable=broad-exception-caught
        return False


def _convert_single_to_cog(
    task: Tuple[str, str, str, str, Optional[int], str, bool, bool],
) -> Tuple[str, str]:
    """Worker function converting a single image to a Cloud-Optimised GeoTIFF."""
    (
        in_img,
        out_img,
        compress,
        resampling,
        datatype,
        bigtiff,
        skip_existing,
        check_existing,
    ) = task

    if skip_existing and os.path.exists(out_img):
        if not check_existing or _is_valid_raster(out_img):
            return out_img, "skipped"

    if resampling == "auto":
        ovr_resamp = "NEAREST" if is_img_thematic(in_img) else "CUBIC"
    else:
        ovr_resamp = resampling.upper()

    options: Dict[str, Any] = {
        "OVERVIEW_RESAMPLING": ovr_resamp,
        "BIGTIFF": bigtiff,
    }
    if compress != "NONE":
        options["COMPRESS"] = compress

    # Plain conversion without statistics (gdal_translate can also store
    # them in the COG with calc_stats=True, at the cost of reading it twice).
    gdal_translate(
        input_img=in_img,
        output_img=out_img,
        options=options,
        gdalformat="COG",
        datatype=datatype,
        calc_stats=False,
        use_pbar=False,
    )
    return out_img, "converted"


def build_batchconvert2cog_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for ``rsgisbatchconvert2cog``."""
    parser = build_parser(
        prog="rsgisbatchconvert2cog",
        description=(
            "Convert one or more raster images into Cloud-Optimised GeoTIFFs "
            "(COGs) using pure rsgispy/rasterio without external shell calls."
        ),
    )
    add_input_files_arg(parser, "Input raster image path(s) or glob pattern(s).")
    add_output_arg(
        parser,
        help_text="Output directory where converted COG .tif files will be written.",
        metavar="DIR",
    )
    parser.add_argument(
        "--compress",
        type=str.upper,
        choices=["LZW", "DEFLATE", "ZSTD", "NONE"],
        default="LZW",
        help="COG compression algorithm.",
    )
    parser.add_argument(
        "--resampling",
        type=str.lower,
        choices=["auto", "nearest", "bilinear", "cubic", "average", "mode"],
        default="auto",
        help=(
            "Overview resampling method ('auto' uses NEAREST for thematic "
            "images and CUBIC for continuous images)."
        ),
    )
    parser.add_argument(
        "--datatype",
        type=str,
        default=None,
        metavar="DTYPE",
        help="Optional output pixel data type (e.g., byte, uint8, uint16, float32).",
    )
    parser.add_argument(
        "--byte",
        action="store_true",
        default=False,
        help="Shortcut for --datatype uint8.",
    )
    parser.add_argument(
        "--bigtiff",
        type=str.upper,
        choices=["IF_SAFER", "YES", "NO", "IF_NEEDED"],
        default="IF_SAFER",
        help="BigTIFF creation option.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=False,
        help="Skip images whose output COG file already exists.",
    )
    parser.add_argument(
        "--check-existing",
        action="store_true",
        default=False,
        help="When skipping existing outputs, validate them and reconvert if corrupt.",
    )
    add_n_cores_arg(
        parser,
        default=1,
        help_text="Number of worker processes for parallel COG conversion.",
    )
    add_no_pbar_arg(parser)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Execute ``rsgisbatchconvert2cog`` and return an exit status code."""
    parser = build_batchconvert2cog_parser()
    args = parser.parse_args(argv)

    if args.n_cores < 1:
        raise RSGISPyException("--n-cores must be >= 1.")

    datatype: Optional[int] = None
    if args.datatype is not None:
        datatype = rsgispy.get_rsgis_datatype(args.datatype)
    elif args.byte:
        datatype = rsgispy.TYPE_8UINT

    in_imgs = resolve_input_files(args.input, natural_sort=True)
    out_dir = os.path.expanduser(args.output)

    skip_existing = args.skip_existing or args.check_existing
    tasks: List[Tuple[str, str, str, str, Optional[int], str, bool, bool]] = []
    seen_outputs: Dict[str, str] = {}
    for in_img in in_imgs:
        basename = get_file_basename(in_img)
        out_img = os.path.join(out_dir, f"{basename}.tif")
        abs_in = os.path.abspath(in_img)
        abs_out = os.path.abspath(out_img)
        if abs_out == abs_in:
            raise RSGISPyException(
                f"Output COG path '{out_img}' would overwrite input image "
                f"'{in_img}' in-place. Specify a different output directory."
            )
        if abs_out in seen_outputs:
            raise RSGISPyException(
                f"Duplicate output filename '{out_img}' produced by inputs "
                f"'{seen_outputs[abs_out]}' and '{in_img}'."
            )
        seen_outputs[abs_out] = in_img
        tasks.append(
            (
                in_img,
                out_img,
                args.compress,
                args.resampling,
                datatype,
                args.bigtiff,
                skip_existing,
                args.check_existing,
            )
        )

    os.makedirs(out_dir, exist_ok=True)

    use_pbar = not args.no_pbar
    tqdm_cls = rsgispy.get_tqdm()
    n_converted = 0
    n_skipped = 0

    if args.n_cores == 1 or len(tasks) == 1:
        iterator = (
            tqdm_cls(tasks, desc="Converting to COG", unit="img")
            if use_pbar
            else tasks
        )
        for task in iterator:
            _, status = _convert_single_to_cog(task)
            if status == "skipped":
                n_skipped += 1
            else:
                n_converted += 1
    else:
        eff_cores = min(args.n_cores, len(tasks))
        with Pool(processes=eff_cores) as pool:
            results_iter = pool.imap_unordered(_convert_single_to_cog, tasks)
            if use_pbar:
                results_iter = tqdm_cls(
                    results_iter,
                    total=len(tasks),
                    desc="Converting to COG",
                    unit="img",
                )
            for _, status in results_iter:
                if status == "skipped":
                    n_skipped += 1
                else:
                    n_converted += 1

    print(
        f"Done: {n_converted} converted, {n_skipped} skipped "
        f"(output directory: {out_dir})."
    )
    return 0


def cli_main() -> int:
    """Console script entry point for ``rsgisbatchconvert2cog``."""
    return run_cli(main)


if __name__ == "__main__":
    sys.exit(cli_main())
