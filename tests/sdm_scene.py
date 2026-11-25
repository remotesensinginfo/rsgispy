"""
Synthetic species distribution modelling scene used by the documented worked
example test (``tests/test_species_distribution_modelling_example.py``).

The scene is a 50 x 50 pixel, 100 m grid in UTM zone 30N (EPSG:32630) with
four environmental covariates:

* ``elevation.tif`` (m; a slope rising to the south plus a central hill, with
  a small no data hole);
* ``climate.tif`` (band 1 mean annual temperature in degrees C, band 2 annual
  precipitation in mm);
* ``ndvi.tif`` (unitless).

The species prefers cool (around 11 degrees C), wet locations with dense
vegetation. Presence points are drawn from that suitability surface and
written to ``presences.gpkg``. A study area mask (``roi_mask.tif``, 1 inside,
0 outside) excludes a 2 pixel border.
"""

import os
from typing import Dict

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point

X0 = 500000.0
Y0 = 5200000.0
RES = 100.0
SIZE = 50
EPSG = 32630
NODATA = -9999.0


def _write(path: str, arr: np.ndarray, nodata=None, band_names=None) -> str:
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
        transform=from_origin(X0, Y0, RES, RES),
        nodata=nodata,
    ) as dst:
        dst.write(arr)
        if band_names:
            for idx, name in enumerate(band_names, start=1):
                dst.set_band_description(idx, name)
    return path


def suitability(temp: np.ndarray, precip: np.ndarray, ndvi: np.ndarray) -> np.ndarray:
    """True habitat suitability (0-1) used to simulate the presences."""
    s_temp = np.exp(-0.5 * ((temp - 11.0) / 1.2) ** 2)
    s_precip = 1.0 / (1.0 + np.exp(-(precip - 1000.0) / 60.0))
    s_ndvi = np.clip((ndvi - 0.2) / 0.5, 0.0, 1.0)
    return s_temp * s_precip * s_ndvi


def make_sdm_scene(out_dir: str, n_pres: int = 80, seed: int = 0) -> Dict[str, str]:
    """
    Write the synthetic covariates, study area mask and presence points to
    ``out_dir`` and return the file paths (``elevation``, ``climate``,
    ``ndvi``, ``roi_mask``, ``presences``).
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows, cols = np.mgrid[0:SIZE, 0:SIZE].astype(np.float64)

    hill = 450.0 * np.exp(-((rows - 22.0) ** 2 + (cols - 30.0) ** 2) / 120.0)
    elev = 150.0 + 12.0 * rows + hill + rng.normal(0.0, 5.0, rows.shape)
    temp = 16.0 - 0.0065 * elev - 0.02 * cols + rng.normal(0.0, 0.1, rows.shape)
    precip = 700.0 + 9.0 * cols + 0.4 * hill + rng.normal(0.0, 10.0, rows.shape)
    ndvi = 0.25 + 0.4 * (cols / SIZE) + 0.15 * np.sin(rows / 6.0)
    ndvi = np.clip(ndvi + rng.normal(0.0, 0.02, rows.shape), -1.0, 1.0)

    suit = suitability(temp, precip, ndvi)

    roi = np.zeros((SIZE, SIZE), dtype=np.uint8)
    roi[2:-2, 2:-2] = 1

    elev_out = elev.astype(np.float32)
    elev_out[40:43, 5:8] = NODATA  # a small hole of missing data

    paths = {
        "elevation": _write(os.path.join(out_dir, "elevation.tif"), elev_out,
                            nodata=NODATA, band_names=["elevation"]),
        "climate": _write(
            os.path.join(out_dir, "climate.tif"),
            np.stack([temp, precip]).astype(np.float32),
            nodata=NODATA,
            band_names=["temp", "precip"],
        ),
        "ndvi": _write(os.path.join(out_dir, "ndvi.tif"), ndvi.astype(np.float32),
                       nodata=NODATA, band_names=["ndvi"]),
        "roi_mask": _write(os.path.join(out_dir, "roi_mask.tif"), roi, nodata=0),
    }

    # Presences: sample pixels in proportion to suitability (inside the ROI).
    weights = (suit * roi).ravel() ** 2
    weights = weights / weights.sum()
    idx = rng.choice(weights.size, size=n_pres, replace=False, p=weights)
    p_rows, p_cols = np.unravel_index(idx, (SIZE, SIZE))
    pts = [
        Point(X0 + (c + rng.uniform(0.1, 0.9)) * RES,
              Y0 - (r + rng.uniform(0.1, 0.9)) * RES)
        for r, c in zip(p_rows, p_cols)
    ]
    pres_gdf = gpd.GeoDataFrame(
        {"species": ["Species A"] * n_pres}, geometry=pts, crs=f"EPSG:{EPSG}"
    )
    paths["presences"] = os.path.join(out_dir, "presences.gpkg")
    pres_gdf.to_file(paths["presences"], layer="presences", driver="GPKG")
    return paths
