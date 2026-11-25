"""
Synthetic two-date scene used by the documented change detection worked
example test (``tests/test_change_detection_example.py``).

Two 60 x 80 pixel, 6-band images (Blue, Green, Red, NIR, SWIR1, SWIR2;
reflectance x 10000 as uint16 with no data 0; 10 m pixels in EPSG:32630) of
the same four land cover classes as ``seg_ml_scene.py`` (Water, Forest,
Grass, Urban). Between the dates two patches of forest are cleared to bare
soil (deforestation), the second image has a small radiometric difference
(gain and offset, e.g. a different atmosphere) and a cloud-masked hole (no
data). A land cover map of the first date, a reference change map and
labelled accuracy points are also written.
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
RES = 10.0
HEIGHT = 60
WIDTH = 80
EPSG = 32630

BAND_NAMES = ("Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2")
SPECTRA = {
    "Water": (600, 500, 300, 150, 100, 50),
    "Forest": (300, 600, 300, 3500, 1500, 700),
    "Grass": (500, 900, 600, 2500, 2200, 1300),
    "Urban": (1200, 1300, 1400, 1800, 2200, 2000),
    "Bare": (1200, 1400, 1700, 2300, 3000, 2600),
}
CLASS_IDS = {"Water": 1, "Forest": 2, "Grass": 3, "Urban": 4}
CHANGE_PATCHES = ((slice(5, 15), slice(45, 60)), (slice(18, 24), slice(65, 75)))
CLOUD = (slice(45, 50), slice(5, 12))


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


def _class_map() -> np.ndarray:
    cmap = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    cmap[:30, :40] = CLASS_IDS["Water"]
    cmap[:30, 40:] = CLASS_IDS["Forest"]
    cmap[30:, :40] = CLASS_IDS["Grass"]
    cmap[30:, 40:] = CLASS_IDS["Urban"]
    return cmap


def _render(names: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    img = np.zeros((len(BAND_NAMES), HEIGHT, WIDTH), dtype=np.float64)
    for name, spec in SPECTRA.items():
        msk = names == name
        for band, val in enumerate(spec):
            img[band][msk] = val
    img *= rng.normal(1.0, 0.04, img.shape)  # within-class variability
    return img


def make_change_scene(out_dir: str, seed: int = 0) -> Dict[str, str]:
    """
    Write the scene to ``out_dir`` and return the file paths: ``s2_t1``,
    ``s2_t2``, ``lc_t1`` (land cover of date 1), ``ref_chng`` (reference
    change: 0 no data, 1 no change, 2 change) and ``acc_pts`` (points with
    the reference class name in ``ref_cls``).
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    lc_map = _class_map()
    id_to_name = {v: k for k, v in CLASS_IDS.items()}
    names_t1 = np.vectorize(id_to_name.get)(lc_map).astype(object)
    names_t2 = names_t1.copy()
    for rows, cols in CHANGE_PATCHES:
        names_t2[rows, cols] = "Bare"

    img_t1 = _render(names_t1, rng)
    img_t2 = _render(names_t2, rng) * 1.05 + 40.0  # different atmosphere
    img_t1 = np.clip(img_t1, 1, 10000).astype(np.uint16)
    img_t2 = np.clip(img_t2, 1, 10000).astype(np.uint16)
    img_t2[:, CLOUD[0], CLOUD[1]] = 0  # cloud masked in date 2

    ref = np.ones((HEIGHT, WIDTH), dtype=np.uint8)
    for rows, cols in CHANGE_PATCHES:
        ref[rows, cols] = 2
    ref[CLOUD] = 0

    paths = {
        "s2_t1": _write(os.path.join(out_dir, "s2_t1.tif"), img_t1, nodata=0,
                        band_names=BAND_NAMES),
        "s2_t2": _write(os.path.join(out_dir, "s2_t2.tif"), img_t2, nodata=0,
                        band_names=BAND_NAMES),
        "lc_t1": _write(os.path.join(out_dir, "lc_t1.tif"), lc_map, nodata=0),
        "ref_chng": _write(os.path.join(out_dir, "ref_chng.tif"), ref, nodata=0),
    }

    # Labelled accuracy points: 40 in change, 60 in no change (pixel centres).
    pts, labels = [], []
    for value, label, n_pts in ((2, "Change", 40), (1, "No change", 60)):
        rows, cols = np.nonzero(ref == value)
        sel = rng.choice(rows.size, size=n_pts, replace=False)
        for r, c in zip(rows[sel], cols[sel]):
            pts.append(Point(X0 + (c + 0.5) * RES, Y0 - (r + 0.5) * RES))
            labels.append(label)
    acc_gdf = gpd.GeoDataFrame({"ref_cls": labels}, geometry=pts,
                               crs=f"EPSG:{EPSG}")
    paths["acc_pts"] = os.path.join(out_dir, "acc_pts.gpkg")
    acc_gdf.to_file(paths["acc_pts"], layer="acc_pts", driver="GPKG")
    return paths
