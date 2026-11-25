"""
Synthetic Sentinel-1 style dual-polarisation (VV, VH) scene used by the SAR
processing guide test (``tests/test_sar_processing_example.py``).

The scene is a 60 x 80 pixel, 2-band image of linear backscatter (sigma0,
float32, no data value 0) on a projected 10 m grid (UTM zone 30N). Four land
cover classes (Water, Forest, Grass, Urban) occupy the four quadrants, with
multiplicative gamma-distributed speckle (as in a multi-looked GRD product),
a few bright point targets in the urban area and a no data strip along the
left edge (as at the edge of a Sentinel-1 swath). A class reference raster is
written on the same grid, plus two polygon layers: training areas (one per
class, ``cls_id`` / ``cls_name`` columns) and homogeneous areas (``area_id``)
for checking speckle reduction.
"""

import os
from typing import Dict

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

X0 = 500000.0
Y0 = 5200000.0
RES = 10.0
HEIGHT = 60
WIDTH = 80
EPSG = 32630
ENL = 4.4  # equivalent number of looks of Sentinel-1 IW GRD (high res.)

# Mean (VV, VH) backscatter in dB for each class (typical C-band values).
CLASS_DB = {
    "Water": (-20.0, -27.0),
    "Forest": (-7.5, -13.0),
    "Grass": (-11.0, -18.0),
    "Urban": (-4.0, -11.0),
}
CLASS_IDS = {"Water": 1, "Forest": 2, "Grass": 3, "Urban": 4}
NO_DATA_COLS = 2


def class_map() -> np.ndarray:
    """Pixel class map (four quadrants)."""
    cmap = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    cmap[:30, :40] = CLASS_IDS["Water"]
    cmap[:30, 40:] = CLASS_IDS["Forest"]
    cmap[30:, :40] = CLASS_IDS["Grass"]
    cmap[30:, 40:] = CLASS_IDS["Urban"]
    cmap[:, :NO_DATA_COLS] = 0
    return cmap


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


def _pxl_box(row0: int, col0: int, row1: int, col1: int):
    """Polygon covering pixel rows ``row0:row1`` and columns ``col0:col1``."""
    return box(X0 + col0 * RES, Y0 - row1 * RES, X0 + col1 * RES, Y0 - row0 * RES)


def make_sar_scene(out_dir: str, seed: int = 0, in_db: bool = False) -> Dict[str, str]:
    """
    Write the synthetic scene to ``out_dir`` and return the file paths:
    ``s1`` (VV, VH backscatter; linear power, or dB with no data value
    -9999 if ``in_db=True``), ``cmap`` (class reference raster, 0 = no
    data), ``train`` (training polygons) and ``homog`` (homogeneous areas).
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    cmap = class_map()
    sigma0 = np.zeros((2, HEIGHT, WIDTH), dtype=np.float64)
    for cls_name, db_vals in CLASS_DB.items():
        msk = cmap == CLASS_IDS[cls_name]
        for b_idx in range(2):
            sigma0[b_idx][msk] = 10.0 ** (db_vals[b_idx] / 10.0)
    # Multiplicative speckle: gamma distributed with mean 1 and ENL looks.
    sigma0 *= rng.gamma(ENL, 1.0 / ENL, sigma0.shape)
    # Bright point targets (buildings) in the urban quadrant.
    for row, col in ((40, 50), (45, 65), (52, 58), (55, 75)):
        sigma0[:, row, col] = (3.0, 0.5)
    sigma0[:, cmap == 0] = 0.0
    if in_db:
        img = np.where(sigma0 > 0, 10.0 * np.log10(np.maximum(sigma0, 1e-12)), -9999.0)
        nodata = -9999.0
    else:
        img = sigma0
        nodata = 0.0
    crs = f"EPSG:{EPSG}"
    # Training polygons inside each quadrant (away from the class edges).
    train_boxes = {
        "Water": (5, 8, 25, 35),
        "Forest": (5, 45, 25, 75),
        "Grass": (35, 8, 55, 35),
        "Urban": (35, 45, 55, 75),
    }
    train_vec = os.path.join(out_dir, "train_areas.gpkg")
    gpd.GeoDataFrame(
        {
            "cls_id": [CLASS_IDS[name] for name in train_boxes],
            "cls_name": list(train_boxes),
        },
        geometry=[_pxl_box(*rc) for rc in train_boxes.values()],
        crs=crs,
    ).to_file(train_vec)
    # Homogeneous areas (open water and a uniform field) for ENL checks.
    homog_vec = os.path.join(out_dir, "homog_areas.gpkg")
    gpd.GeoDataFrame(
        {"area_id": [1, 2], "name": ["water", "grass"]},
        geometry=[_pxl_box(8, 10, 22, 32), _pxl_box(38, 10, 52, 32)],
        crs=crs,
    ).to_file(homog_vec)

    return {
        "train": train_vec,
        "homog": homog_vec,
        "s1": _write(
            os.path.join(out_dir, "s1_vv_vh.tif"),
            img.astype(np.float32),
            nodata=nodata,
            band_names=["VV", "VH"],
        ),
        "cmap": _write(os.path.join(out_dir, "cmap.tif"), cmap, nodata=0),
    }
