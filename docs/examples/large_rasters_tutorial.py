"""
Run the large raster worked example (``large_rasters.py``) on the 20 m
Sentinel-2 scene of mid-Wales from the rsgispy tutorials
(https://github.com/petebunting/rsgispy-tutorials,
``data/sen2_20180629_t30uvd_orb037_osgb_stdsref_20m.tif``: 10 bands, surface
reflectance x 10000, no data 0, British National Grid). The printed outputs
in ``docs/guides/large_rasters.rst`` were made with this script::

    python large_rasters_tutorial.py --data-dir rsgispy-tutorials/data \\
        --out-dir large_tutorial
"""

import argparse
import os
from typing import Any, Dict

import large_rasters


def run(data_dir: str, out_dir: str, use_pbar: bool = True) -> Dict[str, Any]:
    """Run the large raster worked example on the tutorial Sentinel-2 scene."""
    # [tutorial-data]
    # Bands: 1 Blue, 2 Green, 3 Red, 4-6 Red Edge (B5-B7), 7 NIR (B8),
    # 8 NIR (B8A), 9 SWIR1, 10 SWIR2.
    out = large_rasters.run_workflow(
        os.path.join(data_dir, "sen2_20180629_t30uvd_orb037_osgb_stdsref_20m.tif"),
        out_dir,
        red_band=3,
        nir_band=8,  # B8A, the narrow NIR band
        seg_bands=(1, 2, 3, 8, 9, 10),  # Blue, Green, Red, NIR, SWIR1, SWIR2
        block_size=256,
        tile_size=1024,  # 4 x 3 tiles for a 3295 x 2880 pixel image
        n_jobs=4,
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
