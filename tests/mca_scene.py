"""
Synthetic coastal scene used by the multi-criteria analysis (mangrove
restoration potential) worked example test.

The scene is a 60 x 80 pixel (10 m) area in UTM zone 50N (EPSG:32650):

* the sea is to the west (left) and the land rises eastwards: the DEM
  (elevation in metres relative to mean sea level, 20 m pixels, nodata
  -9999) runs from -2 m at the left edge to about +2.8 m at the right edge,
  with a tidal channel (-1.5 m) cut into it along rows 28-31;
* a 6-band Sentinel-2-like surface reflectance image (Blue, Green, Red, NIR,
  SWIR1, SWIR2; uint16 reflectance x 10000, nodata 0) shows water, mangrove,
  bare mud, ponds, a settlement and other land;
* current and historical (baseline) mangrove extent masks (uint8, 0/1, 20 m
  pixels, like Global Mangrove Watch): the historical extent includes an area
  in the south that has since been converted to aquaculture ponds;
* vector layers (GeoPackage): aquaculture ponds with the number of years
  since they were abandoned (``yrs_aband``, 0 = active), a settlement,
  protected areas with a WDPA-style ``IUCN_CAT`` column, and two
  administrative units (``name``).
"""

import os
from typing import Dict

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

X0 = 600000.0
Y0 = 150000.0
RES = 10.0
HEIGHT = 60
WIDTH = 80
EPSG = 32650

# Blue, Green, Red, NIR, SWIR1, SWIR2 reflectance x 10000
SPECTRA = {
    "water": (500, 600, 400, 200, 100, 80),
    "mangrove": (300, 600, 300, 3500, 1200, 500),
    "mud": (900, 1000, 1100, 1400, 1800, 1500),
    "urban": (1200, 1300, 1400, 1800, 2200, 2000),
    "land": (500, 900, 600, 2500, 2000, 1200),
}

# Regions as (row_min, row_max, col_min, col_max), in 10 m pixels.
MANGROVE = (0, 26, 34, 47)
PONDS_ABANDONED = (34, 60, 34, 42)
PONDS_ACTIVE = (34, 60, 42, 50)
SETTLEMENT = (48, 60, 54, 72)
CHANNEL_ROWS = (28, 32)
CHANNEL_END_COL = 56


def _elevation(rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Elevation (m) at fractional pixel coordinates of the 10 m grid."""
    elev = -2.0 + 0.06 * cols
    in_chan = (
        (rows >= CHANNEL_ROWS[0])
        & (rows < CHANNEL_ROWS[1])
        & (cols < CHANNEL_END_COL)
    )
    return np.where(in_chan, -1.5, elev)


def _region_mask(region, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    r_min, r_max, c_min, c_max = region
    return (rows >= r_min) & (rows < r_max) & (cols >= c_min) & (cols < c_max)


def _write(path: str, arr: np.ndarray, res: float, nodata=None) -> str:
    if arr.ndim == 2:
        arr = arr[np.newaxis]
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=arr.shape[1],
        width=arr.shape[2],
        count=arr.shape[0],
        dtype=arr.dtype,
        crs=f"EPSG:{EPSG}",
        transform=from_origin(X0, Y0, res, res),
        nodata=nodata,
    ) as dst:
        dst.write(arr)
    return path


def _region_box(region):
    r_min, r_max, c_min, c_max = region
    return box(X0 + c_min * RES, Y0 - r_max * RES, X0 + c_max * RES, Y0 - r_min * RES)


def make_mca_scene(out_dir: str, seed: int = 0) -> Dict[str, str]:
    """
    Write the synthetic coastal scene to ``out_dir`` and return the file
    paths: ``s2``, ``dem``, ``mng_now``, ``mng_hist``, ``ponds``,
    ``settlements``, ``protected`` and ``admin``.
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    out: Dict[str, str] = {}

    # 10 m grid (pixel centres) for the S2 image.
    rows, cols = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float64) + 0.5
    elev = _elevation(rows, cols)
    cover = np.full((HEIGHT, WIDTH), "land", dtype=object)
    cover[elev < 0.0] = "water"
    cover[elev >= 0.0] = "mud"
    cover[elev > 1.6] = "land"
    cover[_region_mask(MANGROVE, rows, cols)] = "mangrove"
    cover[_region_mask(PONDS_ABANDONED, rows, cols)] = "mud"
    cover[_region_mask(PONDS_ACTIVE, rows, cols)] = "water"
    cover[_region_mask(SETTLEMENT, rows, cols)] = "urban"
    s2 = np.zeros((6, HEIGHT, WIDTH), dtype=np.float64)
    for name, spec in SPECTRA.items():
        msk = cover == name
        for b_idx, val in enumerate(spec):
            s2[b_idx][msk] = val
    s2 += rng.normal(0, 20, s2.shape)
    s2 = np.clip(s2, 1, 10000).astype(np.uint16)
    s2[:, :, :2] = 0  # a strip of no data along the western edge
    out["s2"] = _write(os.path.join(out_dir, "s2_sref.tif"), s2, RES, nodata=0)

    # 20 m grid for the DEM and the mangrove extent masks.
    rows20, cols20 = np.mgrid[:HEIGHT // 2, :WIDTH // 2].astype(np.float64)
    rows20 = rows20 * 2 + 1.0
    cols20 = cols20 * 2 + 1.0
    dem = _elevation(rows20, cols20) + rng.normal(0, 0.02, rows20.shape)
    dem = dem.astype(np.float32)
    dem[-1, -1] = -9999.0
    out["dem"] = _write(os.path.join(out_dir, "dem.tif"), dem, 2 * RES, -9999.0)

    mng_now = _region_mask(MANGROVE, rows20, cols20).astype(np.uint8)
    lost = _region_mask(PONDS_ABANDONED, rows20, cols20) | _region_mask(
        PONDS_ACTIVE, rows20, cols20
    )
    mng_hist = (mng_now.astype(bool) | lost).astype(np.uint8)
    out["mng_now"] = _write(os.path.join(out_dir, "gmw_2020.tif"), mng_now, 2 * RES)
    out["mng_hist"] = _write(os.path.join(out_dir, "gmw_1996.tif"), mng_hist, 2 * RES)

    crs = f"EPSG:{EPSG}"
    ponds = gpd.GeoDataFrame(
        {"pond_id": [1, 2], "yrs_aband": [8.0, 0.0]},
        geometry=[_region_box(PONDS_ABANDONED), _region_box(PONDS_ACTIVE)],
        crs=crs,
    )
    out["ponds"] = os.path.join(out_dir, "ponds.gpkg")
    ponds.to_file(out["ponds"], layer="ponds", driver="GPKG")

    settle = gpd.GeoDataFrame(
        {"landuse": ["residential"]}, geometry=[_region_box(SETTLEMENT)], crs=crs
    )
    out["settlements"] = os.path.join(out_dir, "settlements.gpkg")
    settle.to_file(out["settlements"], layer="settlements", driver="GPKG")

    prot = gpd.GeoDataFrame(
        {"NAME": ["Mangrove reserve", "Coastal park"], "IUCN_CAT": ["II", "VI"]},
        geometry=[_region_box((0, 30, 20, 60)), _region_box((30, 60, 20, 50))],
        crs=crs,
    )
    out["protected"] = os.path.join(out_dir, "protected_areas.gpkg")
    prot.to_file(out["protected"], layer="protected_areas", driver="GPKG")

    admin = gpd.GeoDataFrame(
        {"name": ["North district", "South district"]},
        geometry=[_region_box((0, 30, 0, WIDTH)), _region_box((30, HEIGHT, 0, WIDTH))],
        crs=crs,
    )
    out["admin"] = os.path.join(out_dir, "admin_units.gpkg")
    admin.to_file(out["admin"], layer="admin_units", driver="GPKG")
    return out
