"""
Command-line tool ``rsgiswgs84imgres``: compute approximate pixel resolution
in metres at the centre latitude of one or more WGS84 geographic raster images.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

import pyproj
import rasterio

from rsgispy import RSGISPyException
from rsgispy.cli._common import (
    add_input_files_arg,
    add_json_arg,
    build_parser,
    format_table,
    resolve_input_files,
    run_cli,
)
from rsgispy.tools.projection import degrees_to_metres

__all__ = [
    "build_wgs84imgres_parser",
    "main",
    "cli_main",
]


def _calc_img_res_metres(img_path: str) -> Dict[str, Any]:
    """Read image bounds/resolution once and convert degree resolution to metres."""
    try:
        with rasterio.open(img_path, "r") as ds:
            if ds.crs is None:
                raise RSGISPyException(
                    f"Image '{img_path}' has no CRS defined; expected geographic (WGS84)."
                )
            crs = pyproj.CRS.from_user_input(ds.crs)
            if not crs.is_geographic:
                raise RSGISPyException(
                    f"Image '{img_path}' uses a non-geographic CRS ({crs.name}); "
                    "rsgiswgs84imgres requires a geographic (lat/lon) image."
                )
            bounds = ds.bounds
            x_deg, y_deg = float(ds.res[0]), float(ds.res[1])
            centre_lat = (float(bounds.bottom) + float(bounds.top)) / 2.0
            x_m, y_m = degrees_to_metres(centre_lat, x_deg, y_deg)
    except RSGISPyException:
        raise
    except Exception as err:
        raise RSGISPyException(
            f"Could not read image '{img_path}': {err}"
        ) from err

    return {
        "file": os.path.abspath(img_path),
        "filename": os.path.basename(img_path),
        "centre_lat": centre_lat,
        "res_deg_x": x_deg,
        "res_deg_y": y_deg,
        "res_m_x": float(x_m),
        "res_m_y": float(y_m),
    }


def build_wgs84imgres_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for ``rsgiswgs84imgres``."""
    parser = build_parser(
        prog="rsgiswgs84imgres",
        description=(
            "Calculate the approximate pixel resolution in metres (X and Y) "
            "at the centre latitude of one or more geographic (WGS84) images."
        ),
    )
    add_input_files_arg(
        parser,
        "Input geographic (WGS84 / lat-lon) raster image path(s) or glob pattern(s).",
    )
    add_json_arg(parser, "Emit resolution results as JSON.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Execute ``rsgiswgs84imgres`` and return an exit status code."""
    parser = build_wgs84imgres_parser()
    args = parser.parse_args(argv)

    imgs = resolve_input_files(args.input, natural_sort=True)
    results: List[Dict[str, Any]] = [_calc_img_res_metres(img) for img in imgs]

    if args.json:
        payload: Any = results[0] if len(results) == 1 else results
        print(json.dumps(payload, indent=2))
        return 0

    if len(results) == 1:
        item = results[0]
        print(
            f"{item['filename']} (centre lat {item['centre_lat']:.4f} deg): "
            f"{item['res_deg_x']:.6g} x {item['res_deg_y']:.6g} deg  ->  "
            f"X: {item['res_m_x']:.4f} m, Y: {item['res_m_y']:.4f} m"
        )
    else:
        rows = [
            [
                item["filename"],
                f"{item['centre_lat']:.4f}",
                f"{item['res_deg_x']:.6g} x {item['res_deg_y']:.6g}",
                f"{item['res_m_x']:.4f}",
                f"{item['res_m_y']:.4f}",
            ]
            for item in results
        ]
        print(
            format_table(
                headers=["Image", "Centre Lat", "Res (deg)", "Res X (m)", "Res Y (m)"],
                rows=rows,
                alignments=["l", "r", "r", "r", "r"],
                indent="",
            )
        )
    return 0


def cli_main() -> int:
    """Console script entry point for ``rsgiswgs84imgres``."""
    return run_cli(main)


if __name__ == "__main__":
    sys.exit(cli_main())
