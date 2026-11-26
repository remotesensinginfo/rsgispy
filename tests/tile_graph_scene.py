"""
Synthetic scene for the tile-graph guide example
(``docs/examples/tile_graph.py``) and its test.

The scene is a Sentinel-2-like 4-band image (Blue, Green, Red, NIR; uint16
reflectance x 10000 with no data value 0), a DEM on the same grid (float32,
no data value -9999) and a training label raster (uint8, 0 = unlabelled).
Four land cover classes (1 Water, 2 Forest, 3 Grass, 4 Bare) form irregular
regions from a smoothed random field; the spectra are the class means plus
noise. A strip of no data runs down the left edge of the image and a small
hole is cut in the DEM. The size is a parameter, so the same builder makes
the small test scene and large benchmark scenes.
"""

import os
from typing import Dict

import numpy as np
import rasterio
from rasterio.transform import from_origin

X0 = 500000.0
Y0 = 5200000.0
RES = 10.0
EPSG = 32630

# (Blue, Green, Red, NIR) mean reflectance x 10000 per class
CLASS_SPECTRA = {
    1: (600, 500, 300, 150),  # Water
    2: (300, 600, 300, 3500),  # Forest
    3: (500, 900, 600, 2500),  # Grass
    4: (1200, 1300, 1400, 1800),  # Bare
}


def _smooth_field(rng: np.random.Generator, height: int, width: int,
                  cell: int) -> np.ndarray:
    """Random field interpolated from a coarse grid (cells of ``cell`` px)."""
    n_y = height // cell + 2
    n_x = width // cell + 2
    coarse = rng.random((n_y, n_x))
    y_idx = np.arange(height) / cell
    x_idx = np.arange(width) / cell
    y_0 = np.floor(y_idx).astype(int)
    x_0 = np.floor(x_idx).astype(int)
    f_y = (y_idx - y_0)[:, None]
    f_x = (x_idx - x_0)[None, :]
    c_00 = coarse[y_0][:, x_0]
    c_01 = coarse[y_0][:, x_0 + 1]
    c_10 = coarse[y_0 + 1][:, x_0]
    c_11 = coarse[y_0 + 1][:, x_0 + 1]
    return (c_00 * (1 - f_y) * (1 - f_x) + c_01 * (1 - f_y) * f_x
            + c_10 * f_y * (1 - f_x) + c_11 * f_y * f_x)


def _write(path: str, arr: np.ndarray, nodata=None) -> str:
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
        tiled=True,
        blockxsize=256,
        blockysize=256,
        compress="deflate",
    ) as dst:
        dst.write(arr)
    return path


def make_tile_graph_scene(out_dir: str, height: int = 120, width: int = 150,
                          seed: int = 0) -> Dict[str, str]:
    """
    Write the scene to ``out_dir`` and return the paths of ``'s2'``,
    ``'dem'``, ``'train'`` (training labels) and ``'classes'`` (the true
    class map).
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    cell = max(8, min(height, width) // 6)
    field = _smooth_field(rng, height, width, cell)
    classes = np.digitize(field, np.quantile(field, [0.2, 0.5, 0.75])) + 1
    classes = classes.astype(np.uint8)

    s2 = np.zeros((4, height, width), dtype=np.float64)
    for cls_id, spectrum in CLASS_SPECTRA.items():
        sel = classes == cls_id
        for band, mean in enumerate(spectrum):
            s2[band][sel] = mean
    s2 += rng.normal(0.0, 60.0, size=s2.shape)
    s2 = np.clip(np.rint(s2), 1, 10000).astype(np.uint16)
    edge = max(2, width // 50)
    s2[:, :, :edge] = 0  # no data strip down the left edge

    rows = np.arange(height)[:, None]
    cols = np.arange(width)[None, :]
    dem = (200.0 + 300.0 * _smooth_field(rng, height, width, cell * 2)
           + 0.5 * rows - 0.2 * cols).astype(np.float32)
    hole = (slice(height // 3, height // 3 + 4), slice(width // 2, width // 2 + 4))
    dem[hole] = -9999.0

    train = np.zeros((height, width), dtype=np.uint8)
    keep = rng.random((height, width)) < 0.05  # label 5 % of the pixels
    keep[:, :edge] = False
    train[keep] = classes[keep]

    return {
        "s2": _write(os.path.join(out_dir, "s2_sref.tif"), s2, nodata=0),
        "dem": _write(os.path.join(out_dir, "dem.tif"), dem, nodata=-9999.0),
        "train": _write(os.path.join(out_dir, "train_labels.tif"), train, nodata=0),
        "classes": _write(os.path.join(out_dir, "classes_true.tif"), classes),
    }
