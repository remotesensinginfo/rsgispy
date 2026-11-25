"""
Synthetic segmented scene used by the object-based (segment) ML tests and the
documented worked example test.

The scene is a 60 x 80 pixel, 4-band image (Blue, Green, Red, NIR; uint16 with
nodata 0) plus a slope image, covered by 5 x 5 pixel segments. Four land cover
classes occupy segment-aligned regions so every segment is pure. Training
points/polygons and canopy cover plot points are written as GeoPackages.
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
SEG = 5
EPSG = 32630

# (Blue, Green, Red, NIR) mean reflectance x 10000, and canopy cover (%)
CLASS_SPECTRA = {
    "Water": ((600, 500, 300, 150), 0.0),
    "Forest": ((300, 600, 300, 3500), 80.0),
    "Grass": ((500, 900, 600, 2500), 20.0),
    "Urban": ((1200, 1300, 1400, 1800), 5.0),
}
CLASS_IDS = {"Water": 1, "Forest": 2, "Grass": 3, "Urban": 4}


def _class_map() -> np.ndarray:
    """Pixel class map: four segment-aligned quadrants-like regions."""
    cmap = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    cmap[:30, :40] = CLASS_IDS["Water"]
    cmap[:30, 40:] = CLASS_IDS["Forest"]
    cmap[30:, :40] = CLASS_IDS["Grass"]
    cmap[30:, 40:] = CLASS_IDS["Urban"]
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


def pix_centre(row: float, col: float) -> Point:
    """Map coordinate of a pixel centre."""
    return Point(X0 + (col + 0.5) * RES, Y0 - (row + 0.5) * RES)


def make_seg_scene(out_dir: str, seed: int = 0, tag: str = "a") -> Dict[str, str]:
    """
    Write the synthetic scene to ``out_dir`` and return the file paths:
    ``s2``, ``slope``, ``segs`` (label raster), ``train_<class>`` (training
    vectors), ``plots`` (canopy cover points) and ``acc_pts``.
    """
    # pylint: disable=too-many-locals
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    cmap = _class_map()
    img = np.zeros((4, HEIGHT, WIDTH), dtype=np.float64)
    cover = np.zeros((HEIGHT, WIDTH), dtype=np.float64)
    for cls_name, (spec, cc_val) in CLASS_SPECTRA.items():
        msk = cmap == CLASS_IDS[cls_name]
        for b_idx in range(4):
            img[b_idx][msk] = spec[b_idx]
        cover[msk] = cc_val
    # Within-class gradient in NIR / canopy cover so regression has signal
    grad = np.tile(np.linspace(-1.0, 1.0, WIDTH), (HEIGHT, 1))
    forest = cmap == CLASS_IDS["Forest"]
    img[3][forest] += 400.0 * grad[forest]
    cover[forest] += 10.0 * grad[forest]
    # Per-object texture: a random offset for each 5 x 5 block (as real objects
    # differ), so a segmentation algorithm finds several segments per class.
    blk_off = rng.normal(0.0, 120.0, (HEIGHT // SEG, WIDTH // SEG))
    img += np.kron(blk_off, np.ones((SEG, SEG)))[np.newaxis] * np.array(
        [0.5, 0.7, 0.6, 1.0]
    )[:, np.newaxis, np.newaxis]
    img += rng.normal(0.0, 40.0, img.shape)
    img = np.clip(img, 1, 10000).astype(np.uint16)
    img[:, 55:60, 0:5] = 0  # one all-no-data segment
    slope = (cmap.astype(np.float32) * 2.0 + rng.normal(0.0, 0.5, cmap.shape)).astype(
        np.float32
    )
    labels = (
        (np.arange(HEIGHT)[:, None] // SEG) * (WIDTH // SEG)
        + (np.arange(WIDTH)[None, :] // SEG)
        + 1
    ).astype(np.uint32)

    paths = {
        "s2": _write(
            os.path.join(out_dir, f"s2_{tag}.tif"),
            img,
            nodata=0,
            band_names=["Blue", "Green", "Red", "NIR"],
        ),
        "slope": _write(os.path.join(out_dir, f"slope_{tag}.tif"), slope),
        "segs": _write(os.path.join(out_dir, f"segs_{tag}.tif"), labels),
        "cmap": _write(os.path.join(out_dir, f"cmap_{tag}.tif"), cmap),
    }

    crs = f"EPSG:{EPSG}"
    # Training points: several per class, two in the same segment (dedup).
    pts = {
        "Water": [(2, 2), (2, 3), (12, 12), (22, 27), (7, 32), (17, 17)],
        "Forest": [(2, 47), (12, 57), (22, 72), (7, 62), (17, 77), (27, 52)],
        "Grass": [(32, 7), (42, 17), (52, 27), (37, 32), (47, 12)],
        "Urban": [(32, 47), (42, 57), (52, 67), (37, 77), (57, 52)],
    }
    for cls_name, rc_list in pts.items():
        gdf = gpd.GeoDataFrame(
            {"cls": [cls_name] * len(rc_list)},
            geometry=[pix_centre(r, c) for r, c in rc_list],
            crs=crs,
        )
        path = os.path.join(out_dir, f"train_{cls_name.lower()}_{tag}.gpkg")
        gdf.to_file(path)
        paths[f"train_{cls_name.lower()}"] = path
    # Grass also has a training polygon covering 2 full segments + 40% of one.
    poly = box(X0 + 15 * RES, Y0 - 50 * RES, X0 + 27 * RES, Y0 - 45 * RES)
    gpd.GeoDataFrame({"cls": ["Grass"]}, geometry=[poly], crs=crs).to_file(
        os.path.join(out_dir, f"train_grass_poly_{tag}.gpkg")
    )
    paths["train_grass_poly"] = os.path.join(out_dir, f"train_grass_poly_{tag}.gpkg")

    # Canopy cover plots (two plots in one segment are averaged)
    plot_rc = [(r, c) for r in range(2, 60, 6) for c in range(2, 80, 9)]
    plot_rc.append((2, 3))
    cc_vals = [float(cover[r, c]) for r, c in plot_rc]
    gpd.GeoDataFrame(
        {"canopy_cover": cc_vals, "canopy_x2": [2.0 * v for v in cc_vals]},
        geometry=[pix_centre(r, c) for r, c in plot_rc],
        crs=crs,
    ).to_file(os.path.join(out_dir, f"plots_{tag}.gpkg"))
    paths["plots"] = os.path.join(out_dir, f"plots_{tag}.gpkg")

    # Accuracy points with reference class IDs
    acc_rc = [(r, c) for r in range(1, 60, 4) for c in range(1, 80, 6)]
    id_names = {v: k for k, v in CLASS_IDS.items()}
    gpd.GeoDataFrame(
        {
            "ref_cls": [int(cmap[r, c]) for r, c in acc_rc],
            "ref_name": [id_names[int(cmap[r, c])] for r, c in acc_rc],
        },
        geometry=[pix_centre(r, c) for r, c in acc_rc],
        crs=crs,
    ).to_file(os.path.join(out_dir, f"acc_pts_{tag}.gpkg"))
    paths["acc_pts"] = os.path.join(out_dir, f"acc_pts_{tag}.gpkg")
    return paths


def seg_class_truth() -> Dict[int, int]:
    """Map of segment ID to true class ID for the synthetic scene."""
    cmap = _class_map()
    out = {}
    for row in range(0, HEIGHT, SEG):
        for col in range(0, WIDTH, SEG):
            seg_id = (row // SEG) * (WIDTH // SEG) + (col // SEG) + 1
            out[seg_id] = int(cmap[row, col])
    return out
