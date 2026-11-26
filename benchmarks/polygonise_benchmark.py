"""
Benchmark :func:`rsgispy.vector.create.polygonise_raster_tiled` against
:func:`rsgispy.vector.create.polygonise_raster_to_vec_lyr` on large,
irregular, segmentation-like label rasters.

The label rasters are Voronoi segments with jittered boundaries (about one
segment per 60 pixels, with many small fragments along the boundaries, so
there are roughly 10 polygons per segment), similar to the output of an
image segmentation.

Each configuration runs in a fresh Python process so the wall-clock time
includes worker start-up and the peak memory (resident set size) of the main
process and of the largest worker process can be reported. Results are
printed as a Markdown table.

Usage::

    python benchmarks/polygonise_benchmark.py --sizes 3000 10000 \\
        --out-dir /tmp/poly_bench --n-jobs 1 4 8 --tile-sizes 1024 2048

The current (non-tiled) function needs roughly 2.5 kB of memory per polygon;
pass ``--skip-old-above`` to skip it for large rasters.
"""

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from typing import Any, Dict, List

import numpy as np

# Prefer the rsgispy source tree containing this script over an installed copy.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.exists(os.path.join(_REPO, "rsgispy", "__init__.py")):
    sys.path.insert(0, _REPO)

SEG_DENSITY = 150_000 / 9_000_000  # segments per pixel
JITTER = 1.5  # pixel jitter of the Voronoi boundaries


def make_label_raster(path: str, size: int, seed: int = 0) -> None:
    """Write a ``size`` x ``size`` jittered-Voronoi label raster (int32)."""
    # pylint: disable=import-outside-toplevel,too-many-locals
    import rasterio
    from rasterio.transform import from_origin
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)
    n_seg = int(size * size * SEG_DENSITY)
    tree = cKDTree(rng.uniform(0, size, (n_seg, 2)))
    block = 1024
    profile = {
        "driver": "GTiff",
        "height": size,
        "width": size,
        "count": 1,
        "dtype": "int32",
        "crs": "EPSG:32630",
        "transform": from_origin(500000, 5200000, 10, 10),
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "nodata": 0,
    }
    with rasterio.open(path, "w", **profile) as dst:
        for row0 in range(0, size, block):
            n_rows = min(block, size - row0)
            rows, cols = np.mgrid[row0 : row0 + n_rows, 0:size]  # noqa: E203
            pts = np.column_stack(
                [
                    rows.ravel() + rng.normal(0, JITTER, rows.size),
                    cols.ravel() + rng.normal(0, JITTER, cols.size),
                ]
            )
            _, lab = tree.query(pts, workers=-1)
            arr = (lab.reshape(n_rows, size) + 1).astype(np.int32)
            dst.write(arr, 1, window=rasterio.windows.Window(0, row0, size, n_rows))


def _peak_rss_mb(who: int) -> float:
    """Peak RSS in MB (``ru_maxrss`` is bytes on macOS, kB on Linux)."""
    val = resource.getrusage(who).ru_maxrss
    return val / 1e6 if sys.platform == "darwin" else val / 1e3


def run_one(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Run one configuration (inside a fresh process)."""
    # pylint: disable=import-outside-toplevel,import-error,no-name-in-module
    import rsgispy._polygonise as poly_impl
    import rsgispy.ml.vector  # noqa: F401  (one-off import, not timed)
    import rsgispy.vector.create as vec_create

    out: Dict[str, Any] = dict(cfg)
    t_0 = time.perf_counter()
    if cfg["method"] == "old":
        gdf = vec_create.polygonise_raster_to_vec_lyr(
            None, None, "GPKG", cfg["img"], use_8_conn=False
        )
        t_1 = time.perf_counter()
        gdf.to_parquet(cfg["out"], compression="snappy")
        t_2 = time.perf_counter()
        out["n_polys"] = len(gdf)
        out["stages"] = {"polygonise": t_1 - t_0, "write": t_2 - t_1}
    else:
        vec_create.polygonise_raster_tiled(
            cfg["img"],
            cfg["out"],
            tile_size=cfg["tile_size"],
            n_jobs=cfg["n_jobs"],
            backend=cfg["backend"],
            dissolve_by_value=cfg.get("dissolve", False),
            sort_by_value=cfg.get("dissolve", False),
            overwrite=True,
            use_pbar=False,
        )
        import pyarrow.parquet as pq  # pylint: disable=import-error

        out["n_polys"] = pq.ParquetFile(cfg["out"]).metadata.num_rows
        out["stages"] = dict(poly_impl.LAST_RUN_TIMINGS)
    out["time"] = time.perf_counter() - t_0
    out["rss_main_mb"] = _peak_rss_mb(resource.RUSAGE_SELF)
    out["rss_worker_mb"] = _peak_rss_mb(resource.RUSAGE_CHILDREN)
    return out


def _spawn(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Run ``cfg`` in a new Python process and return its result."""
    res = subprocess.run(
        [sys.executable, __file__, "--run-one", json.dumps(cfg)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(res.stdout.strip().splitlines()[-1])


def _fmt_stages(stages: Dict[str, float]) -> str:
    return ", ".join(f"{k} {v:.1f}" for k, v in stages.items() if k != "total")


def main(argv: List[str]) -> None:
    """Command line entry point."""
    # pylint: disable=too-many-locals
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    parser.add_argument("--sizes", type=int, nargs="+", default=[3000])
    parser.add_argument("--out-dir", default="poly_bench")
    parser.add_argument("--n-jobs", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--tile-sizes", type=int, nargs="+", default=[2048])
    parser.add_argument("--backends", nargs="+", default=["numba", "rasterio"])
    parser.add_argument("--skip-old-above", type=int, default=20_000)
    parser.add_argument("--dissolve", action="store_true",
                        help="Also run dissolve_by_value + sort_by_value.")
    parser.add_argument("--run-one", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.run_one is not None:
        print(json.dumps(run_one(json.loads(args.run_one))))
        return

    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    for size in args.sizes:
        img = os.path.abspath(os.path.join(args.out_dir, f"labels_{size}.tif"))
        if not os.path.exists(img):
            t_0 = time.perf_counter()
            make_label_raster(img, size)
            print(f"Created {img} in {time.perf_counter() - t_0:.0f} s", flush=True)
        out = os.path.join(args.out_dir, f"out_{size}.parquet")
        cfgs: List[Dict[str, Any]] = []
        if size <= args.skip_old_above:
            cfgs.append({"method": "old", "img": img, "out": out, "size": size})
        for backend in args.backends:
            for tile_size in args.tile_sizes:
                for n_jobs in args.n_jobs:
                    cfgs.append(
                        {"method": "tiled", "img": img, "out": out, "size": size,
                         "backend": backend, "tile_size": tile_size, "n_jobs": n_jobs}
                    )
        if args.dissolve:
            cfgs.append(
                {"method": "tiled", "img": img, "out": out, "size": size,
                 "backend": args.backends[0], "tile_size": args.tile_sizes[0],
                 "n_jobs": max(args.n_jobs), "dissolve": True}
            )
        for cfg in cfgs:
            res = _spawn(cfg)
            rows.append(res)
            print(json.dumps(res), flush=True)

    print()
    print("| Image | Method | Backend | Tile | n_jobs | Polygons | Time (s) "
          "| Stages (s) | Peak RSS main / worker (MB) |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for res in rows:
        method = "current" if res["method"] == "old" else (
            "tiled + dissolve/sort" if res.get("dissolve") else "tiled")
        print(
            f"| {res['size']} x {res['size']} | {method} | {res.get('backend', '-')} "
            f"| {res.get('tile_size', '-')} | {res.get('n_jobs', 1)} | {res['n_polys']:,} "
            f"| {res['time']:.1f} | {_fmt_stages(res['stages'])} "
            f"| {res['rss_main_mb']:.0f} / {res['rss_worker_mb']:.0f} |"
        )


if __name__ == "__main__":
    main(sys.argv[1:])
