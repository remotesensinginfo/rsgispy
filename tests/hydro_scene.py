"""
Synthetic catchment used by the documented hydrology worked example test
(``tests/test_hydrology_workflow_example.py``).

The DEM is a 64 x 64 pixel analytic surface delivered, like most global DEMs
(SRTM, Copernicus GLO-30), in geographic coordinates (EPSG:4326, about 10 m
pixels at 47 N, 3 W) so the example has to reproject it to UTM zone 30N
(EPSG:32630). The surface slopes to the south and has:

* two tributary valleys (starting near columns 14 and 50) that join a main
  valley along column 32 half way down the scene, with a gently sloping
  floodplain along the main valley;
* a road embankment across the main valley (rows 44-45) that dams it and
  creates a depression (about 2 m deep) upstream;
* a small closed pit (3 x 3 pixels) on the interfluve between the
  tributaries.

The scene also includes a land cover raster on the DEM grid (1 = forest,
2 = grassland, 3 = urban), a mapped river line along the main valley and two
gauge points placed slightly off the channel so they have to be snapped.
"""

import os
from typing import Dict

import geopandas as gpd
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin
from shapely.geometry import LineString, Point

ROWS = 64
COLS = 64
LON0 = -3.0
LAT0 = 47.0
D_LON = 1.32e-4  # ~10 m at 47 N
D_LAT = 9.0e-5  # ~10 m
MAIN_COL = 32.0
JOIN_ROW = 32
OUT_EPSG = 32630
LC_NAMES = {1: "Forest", 2: "Grassland", 3: "Urban"}


def _valley_col(row: np.ndarray) -> tuple:
    """Columns of the left and right tributaries for each row."""
    frac = np.minimum(row, JOIN_ROW) / JOIN_ROW
    left = 14.0 + (MAIN_COL - 14.0) * frac
    right = 50.0 - (50.0 - MAIN_COL) * frac
    return left, right


def make_dem_array() -> np.ndarray:
    """Analytic DEM (metres) on the 64 x 64 grid."""
    r_idx, c_idx = np.indices((ROWS, COLS), dtype=float)
    left, right = _valley_col(r_idx)
    dist = np.where(
        r_idx <= JOIN_ROW,
        np.minimum(np.abs(c_idx - left), np.abs(c_idx - right)),
        np.abs(c_idx - MAIN_COL),
    )
    # Floodplain (gentle, within 4 pixels of the channel) then steeper sides.
    side = np.where(dist <= 4.0, 0.5 * dist, 2.0 + 3.0 * (dist - 4.0))
    dem = 120.0 + (ROWS - 1 - r_idx) * 0.3 + side
    # Road embankment damming the main valley.
    emb = (r_idx >= 44) & (r_idx <= 45) & (np.abs(c_idx - MAIN_COL) <= 8)
    dem[emb] += 3.0
    # Closed pit on the interfluve.
    dem[10:13, 31:34] -= 8.0
    return dem.astype(np.float32)


def _pix_lonlat(row: float, col: float) -> tuple:
    return LON0 + (col + 0.5) * D_LON, LAT0 - (row + 0.5) * D_LAT


def _write_geo(path: str, arr: np.ndarray, nodata) -> str:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=ROWS,
        width=COLS,
        count=1,
        dtype=arr.dtype,
        crs="EPSG:4326",
        transform=from_origin(LON0, LAT0, D_LON, D_LAT),
        nodata=nodata,
    ) as dst:
        dst.write(arr[np.newaxis])
    return path


def make_hydro_scene(out_dir: str) -> Dict[str, str]:
    """
    Write the synthetic catchment to ``out_dir`` and return the file paths:
    ``dem`` (EPSG:4326), ``landcover`` (EPSG:4326, uint8), ``rivers`` (mapped
    river line, EPSG:32630) and ``gauges`` (two points with a ``gauge_id``
    column, EPSG:32630).
    """
    # pylint: disable=too-many-locals
    os.makedirs(out_dir, exist_ok=True)
    dem = make_dem_array()
    dem[0, 0] = -9999.0  # a nodata pixel, as in real DEM tiles
    dem_file = _write_geo(os.path.join(out_dir, "dem_wgs84.tif"), dem, -9999.0)

    r_idx, c_idx = np.mgrid[0:ROWS, 0:COLS]
    left, right = _valley_col(r_idx.astype(float))
    near_main = np.where(
        r_idx <= JOIN_ROW,
        np.minimum(np.abs(c_idx - left), np.abs(c_idx - right)),
        np.abs(c_idx - MAIN_COL),
    ) <= 4.0
    landcover = np.where(near_main, 2, 1).astype(np.uint8)
    landcover[48:58, 40:50] = 3
    lc_file = _write_geo(os.path.join(out_dir, "landcover_wgs84.tif"), landcover, 0)

    to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{OUT_EPSG}", always_xy=True)

    def utm_pt(row: float, col: float) -> Point:
        return Point(*to_utm.transform(*_pix_lonlat(row, col)))

    river = LineString(
        [utm_pt(row, MAIN_COL).coords[0] for row in range(JOIN_ROW, ROWS)]
    )
    rivers_file = os.path.join(out_dir, "rivers.gpkg")
    gpd.GeoDataFrame({"name": ["main"]}, geometry=[river],
                     crs=f"EPSG:{OUT_EPSG}").to_file(rivers_file)

    left_col = float(_valley_col(np.array(24.0))[0])
    gauges = gpd.GeoDataFrame(
        {"gauge_id": [1, 2]},
        geometry=[
            utm_pt(58, MAIN_COL + 2),  # main river, two pixels off the channel
            utm_pt(24, left_col - 1),  # left tributary, one pixel off
        ],
        crs=f"EPSG:{OUT_EPSG}",
    )
    gauges_file = os.path.join(out_dir, "gauges.gpkg")
    gauges.to_file(gauges_file)
    return {"dem": dem_file, "landcover": lc_file, "rivers": rivers_file,
            "gauges": gauges_file}
