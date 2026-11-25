"""
Synthetic image pair used by the image co-registration worked example test
(``tests/test_image_coregistration_example.py``).

The reference and floating images are 2-band (Red, NIR) 10 m images
(``int16``, nodata 0, EPSG:32630) on the same pixel grid. The land surface is
an analytic function of map coordinates; in the floating image it is shifted
by ``SHIFT`` metres and rotated by ``ROTATION_DEG`` degrees about the image
centre, so a translation alone cannot register the images exactly. The
floating image also has a "cloud" (bright, no matchable texture) and a strip of
no data. Ground control points (GCPs) digitised on the floating image, with the
true (reference) coordinates in ``x_match``/``y_match``, are written as a
GeoPackage.
"""

import os
from typing import Dict, Tuple

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point

EPSG = 32630
X0 = 500000.0
Y0 = 5700000.0
RES = 10.0
SIZE = 160
SHIFT = (23.0, -14.0)  # metres: features move 23 m east and 14 m south
ROTATION_DEG = 0.3
CENTRE = (X0 + SIZE * RES / 2.0, Y0 - SIZE * RES / 2.0)


def _surface(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Red and NIR reflectance (x 10000) of the synthetic landscape."""
    veg = np.clip(
        0.5
        + 0.22 * np.sin(x / 61.0) * np.cos(y / 73.0)
        + 0.15 * np.sin((x + y) / 37.0)
        + 0.10 * np.cos((x - 1.7 * y) / 29.0),
        0.0,
        1.0,
    )
    red = 1800.0 - 1400.0 * veg
    nir = 2600.0 + 1200.0 * veg
    return red, nir


def flt_to_ref(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Map floating image coordinates to the true (reference) coordinates."""
    ang = np.deg2rad(ROTATION_DEG)
    dx, dy = x - CENTRE[0], y - CENTRE[1]
    rx = CENTRE[0] + np.cos(ang) * dx - np.sin(ang) * dy - SHIFT[0]
    ry = CENTRE[1] + np.sin(ang) * dx + np.cos(ang) * dy - SHIFT[1]
    return rx, ry


def ref_to_flt(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Inverse of :func:`flt_to_ref`."""
    ang = -np.deg2rad(ROTATION_DEG)
    dx, dy = x + SHIFT[0] - CENTRE[0], y + SHIFT[1] - CENTRE[1]
    return (
        CENTRE[0] + np.cos(ang) * dx - np.sin(ang) * dy,
        CENTRE[1] + np.sin(ang) * dx + np.cos(ang) * dy,
    )


def _write(path: str, arr: np.ndarray) -> str:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=SIZE,
        width=SIZE,
        count=2,
        dtype="int16",
        crs=f"EPSG:{EPSG}",
        transform=from_origin(X0, Y0, RES, RES),
        nodata=0,
    ) as dst:
        dst.write(arr)
        dst.descriptions = ("Red", "NIR")
    return path


def make_coreg_scene(out_dir: str, seed: int = 0) -> Dict[str, str]:
    """Write the reference and floating images and the GCPs; return paths."""
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    cols = X0 + (np.arange(SIZE) + 0.5) * RES
    rows = Y0 - (np.arange(SIZE) + 0.5) * RES
    xx, yy = np.meshgrid(cols, rows)

    ref = np.stack(_surface(xx, yy))
    ref += rng.normal(0.0, 15.0, ref.shape)

    flt = np.stack(_surface(*flt_to_ref(xx, yy)))
    flt = flt * 1.1 + 150.0  # different illumination / sensor gain
    flt += rng.normal(0.0, 15.0, flt.shape)
    cloud = ((xx - (X0 + 1150.0)) ** 2 + (yy - (Y0 - 400.0)) ** 2) < 100.0**2
    flt[:, cloud] = 3500.0
    flt[:, :, :6] = 0.0  # no data at the left edge

    # GCPs: features identified on the floating image (geometry) with their
    # true coordinates measured on the reference (e.g. GNSS or orthophoto).
    true_x = X0 + np.array([200.0, 800.0, 1400.0, 300.0, 1300.0, 800.0, 250.0])
    true_y = Y0 - np.array([200.0, 250.0, 300.0, 1350.0, 1400.0, 800.0, 750.0])
    flt_x, flt_y = ref_to_flt(true_x, true_y)
    gcps = gpd.GeoDataFrame(
        {"x_match": true_x, "y_match": true_y},
        geometry=[Point(x, y) for x, y in zip(flt_x, flt_y)],
        crs=f"EPSG:{EPSG}",
    )
    gcp_file = os.path.join(out_dir, "flt_gcps.gpkg")
    gcps.to_file(gcp_file, driver="GPKG")

    return {
        "ref": _write(os.path.join(out_dir, "ref_img.tif"),
                      np.round(ref).astype(np.int16)),
        "flt": _write(os.path.join(out_dir, "flt_img.tif"),
                      np.round(flt).astype(np.int16)),
        "gcps": gcp_file,
    }
