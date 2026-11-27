"""
Run the hydrology worked example (``hydrology_workflow.py``) on the 10 m DEM
of the Aberystwyth area from the rsgispy tutorials
(https://github.com/remotesensinginfo/rsgispy-tutorials, ``data/dem_10m_osgb.tif``).
The figures and printed outputs in ``docs/guides/hydrology_workflow.rst``
were made with this script::

    python hydrology_workflow_tutorial.py --data-dir rsgispy-tutorials/data \\
        --out-dir hydro_tutorial
"""

import argparse
import os
from typing import Any, Dict

import hydrology_workflow


def run(data_dir: str, out_dir: str, use_pbar: bool = True) -> Dict[str, Any]:
    """Prepare the tutorial inputs and run the hydrology worked example."""
    # [tutorial-data]
    import geopandas as gpd
    import rsgispy
    from rsgispy.imagecalc.maths import BandDefn, band_maths

    os.makedirs(out_dir, exist_ok=True)

    # The sea has an elevation of 0 m in this DEM. Set it to no data, so the
    # rivers leave the DEM at the coast instead of crossing a flat "sea".
    dem_img = os.path.join(out_dir, "dem_land.tif")
    band_maths(dem_img, "dem <= 0 ? -32768 : dem",
               [BandDefn("dem", os.path.join(data_dir, "dem_10m_osgb.tif"), 1)],
               datatype=rsgispy.TYPE_32FLOAT, no_data_val=-32768,
               use_pbar=use_pbar)

    # Approximate positions (to about 100 m) of the Natural Resources Wales
    # gauging stations on the Rheidol and the Ystwyth.
    gauges_vec = os.path.join(out_dir, "gauges.gpkg")
    gpd.GeoDataFrame(
        {"gauge_id": [1, 2],
         "name": ["Rheidol at Llanbadarn Fawr", "Ystwyth at Pont Llolwyn"]},
        geometry=gpd.points_from_xy([260100, 259100], [280400, 277400]),
        crs="EPSG:27700",
    ).to_file(gauges_vec)

    out = hydrology_workflow.run_workflow(
        dem_img,
        gauges_vec,
        out_dir,
        max_breach_length=30,  # pixels (300 m at 10 m)
        stream_threshold=1e6,  # m2 (1 km2)
        snap_dist=100.0,  # the positional error of the gauges
        flood_stages=(1.0, 2.0, 5.0, 10.0),
        use_pbar=use_pbar,
    )
    # [end-tutorial-data]
    return out


def main() -> None:
    """Command line interface."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    parser.add_argument("--data-dir", required=True,
                        help="The data folder of the rsgispy-tutorials repository")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    args = parser.parse_args()
    run(args.data_dir, args.out_dir)


if __name__ == "__main__":
    main()
