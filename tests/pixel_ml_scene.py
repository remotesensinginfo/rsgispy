"""
Synthetic scene used by the documented pixel-based classification and
regression worked example tests (``docs/examples/pixel_*_workflow.py``).

The scene mimics a 10 m Sentinel-2 surface reflectance stack in the band
order written by
:func:`rsgispy.imagecalibration.sensorlvl2data.create_stacked_sen2_esa_l2a_img`
(B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12; uint16 reflectance x 10000 with
no data value 0) on a 60 x 80 pixel grid in UTM (EPSG:32630), together with a
20 m DEM on a coarser grid (so it must be resampled) and vector training data:

* four land cover classes (Water, Forest, Grass, Urban) laid out as a 4 x 4
  chequer of 15 x 20 pixel blocks, so every class occurs in four separate
  places (needed for spatial cross-validation);
* training polygons (one per block, two or three blocks per class);
* accuracy assessment points with the reference class name (``ref_name``);
* canopy cover plots (``canopy_cover``, %) for training a regression and a
  separate set of validation plots.

Reflectance is a linear mixture of a class background spectrum and a
vegetation spectrum weighted by canopy cover, so canopy cover can be
predicted from the spectra.
"""

import os
from typing import Dict

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point, box

X0 = 500000.0
Y0 = 5200000.0
RES = 10.0
HEIGHT = 60
WIDTH = 80
BLK_H = 15
BLK_W = 20
EPSG = 32630

CLASS_IDS = {"Water": 1, "Forest": 2, "Grass": 3, "Urban": 4}
CLASS_COLOURS = {
    "Water": "#1f78b4",
    "Forest": "#33a02c",
    "Grass": "#b2df8a",
    "Urban": "#808080",
}
S2_BAND_NAMES = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]

# Background (non-vegetation) reflectance x 10000 for each class and a pure
# vegetation spectrum (10 Sentinel-2 bands).
_BACKGROUND = {
    "Water": (700, 600, 400, 300, 250, 220, 200, 180, 100, 80),
    "Forest": (600, 800, 900, 1200, 1500, 1700, 1800, 1900, 2400, 1800),
    "Grass": (700, 1000, 1100, 1400, 1700, 1900, 2000, 2100, 2600, 2000),
    "Urban": (1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2700, 2400),
}
_VEG = (300, 700, 350, 1200, 3000, 3800, 4200, 4300, 1900, 900)
# Canopy cover (fraction) range for each class
_COVER = {
    "Water": (0.0, 0.0),
    "Forest": (0.55, 0.95),
    "Grass": (0.05, 0.35),
    "Urban": (0.0, 0.1),
}
# Training blocks (block row, block col) for each class
_TRAIN_BLOCKS = {
    "Water": [(0, 0), (2, 2), (1, 3)],
    "Forest": [(0, 1), (2, 3), (1, 0)],
    "Grass": [(0, 2), (2, 0), (3, 3)],
    "Urban": [(0, 3), (2, 1), (1, 2)],
}


def class_map() -> np.ndarray:
    """Pixel class map (class IDs 1-4) of the synthetic scene."""
    rows = np.arange(HEIGHT)[:, None] // BLK_H
    cols = np.arange(WIDTH)[None, :] // BLK_W
    return ((rows + cols) % 4 + 1).astype(np.uint8)


def _write(path: str, arr: np.ndarray, res: float = RES, nodata=None,
           band_names=None) -> str:
    """Write a GeoTIFF on the scene grid (or a coarser grid with ``res``)."""
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
        if band_names:
            for idx, name in enumerate(band_names, start=1):
                dst.set_band_description(idx, name)
    return path


def pix_centre(row: float, col: float) -> Point:
    """Map coordinate of a pixel centre."""
    return Point(X0 + (col + 0.5) * RES, Y0 - (row + 0.5) * RES)


def _cover_map(cmap: np.ndarray) -> np.ndarray:
    """Canopy cover fraction varying smoothly within each class."""
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH]
    wave = 0.5 + 0.5 * np.sin(xx / 7.0) * np.cos(yy / 9.0)
    cover = np.zeros((HEIGHT, WIDTH), dtype=np.float64)
    for name, cls_id in CLASS_IDS.items():
        low, high = _COVER[name]
        msk = cmap == cls_id
        cover[msk] = low + (high - low) * wave[msk]
    return cover


def make_pixel_scene(out_dir: str, seed: int = 0) -> Dict[str, str]:
    """
    Write the synthetic scene to ``out_dir`` and return the file paths:
    ``s2`` (10-band stack), ``dem`` (20 m DEM), ``cmap`` (true class map),
    ``train_<class>`` (training polygons), ``acc_pts`` (reference points),
    ``plots`` and ``plots_valid`` (canopy cover plots).
    """
    # pylint: disable=too-many-locals
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    cmap = class_map()
    cover = _cover_map(cmap)
    veg = np.asarray(_VEG, dtype=np.float64)[:, None, None]
    img = np.zeros((10, HEIGHT, WIDTH), dtype=np.float64)
    for name, cls_id in CLASS_IDS.items():
        msk = cmap == cls_id
        bkgd = np.asarray(_BACKGROUND[name], dtype=np.float64)[:, None]
        img[:, msk] = bkgd * (1.0 - cover[msk]) + veg[:, 0] * cover[msk]
    img += rng.normal(0.0, 60.0, img.shape)
    img = np.clip(img, 1, 10000).astype(np.uint16)
    img[:, -3:, :4] = 0  # a small no data corner

    # 20 m DEM: a tilted plane with hills (steeper in the forest blocks).
    yy, xx = np.mgrid[0:HEIGHT // 2, 0:WIDTH // 2].astype(np.float64)  # pylint: disable=no-member
    dem = 100.0 + 0.5 * xx + 6.0 * np.sin(xx / 3.0) * np.cos(yy / 4.0)
    dem += rng.normal(0.0, 0.2, dem.shape)

    paths = {
        "s2": _write(os.path.join(out_dir, "s2_sref.tif"), img, nodata=0,
                     band_names=S2_BAND_NAMES),
        "dem": _write(os.path.join(out_dir, "dem_20m.tif"),
                      dem.astype(np.float32), res=2 * RES, nodata=-9999.0),
        "cmap": _write(os.path.join(out_dir, "true_classes.tif"), cmap),
    }

    crs = f"EPSG:{EPSG}"
    for name, blocks in _TRAIN_BLOCKS.items():
        polys = []
        for b_row, b_col in blocks:
            row0, col0 = b_row * BLK_H + 4, b_col * BLK_W + 5
            polys.append(box(X0 + col0 * RES, Y0 - (row0 + 7) * RES,
                             X0 + (col0 + 10) * RES, Y0 - row0 * RES))
        path = os.path.join(out_dir, f"train_{name.lower()}.gpkg")
        gpd.GeoDataFrame({"cls": [name] * len(polys)}, geometry=polys,
                         crs=crs).to_file(path)
        paths[f"train_{name.lower()}"] = path

    id_names = {v: k for k, v in CLASS_IDS.items()}
    acc_rc = [(r, c) for r in range(2, HEIGHT - 3, 4) for c in range(6, WIDTH, 5)]
    gpd.GeoDataFrame(
        {"ref_name": [id_names[int(cmap[r, c])] for r, c in acc_rc]},
        geometry=[pix_centre(r, c) for r, c in acc_rc],
        crs=crs,
    ).to_file(os.path.join(out_dir, "acc_pts.gpkg"))
    paths["acc_pts"] = os.path.join(out_dir, "acc_pts.gpkg")

    for key, offset in (("plots", 0), ("plots_valid", 2)):
        plot_rc = [(r, c) for r in range(1 + offset, HEIGHT - 4, 4)
                   for c in range(1 + offset, WIDTH, 4)]
        gpd.GeoDataFrame(
            {"canopy_cover": [round(100.0 * cover[r, c], 1) for r, c in plot_rc]},
            geometry=[pix_centre(r, c) for r, c in plot_rc],
            crs=crs,
        ).to_file(os.path.join(out_dir, f"{key}.gpkg"))
        paths[key] = os.path.join(out_dir, f"{key}.gpkg")
    return paths
