"""
Synthetic scene for the zonal statistics and vector analysis worked example
test (``tests/test_zonal_stats_and_vectors_example.py``).

The scene is a 60 x 80 pixel, 4-band Sentinel-2-like image (Blue, Green, Red,
NIR; uint16 surface reflectance x 10000, no data value 0) in UTM zone 30N with
10 m pixels, plus:

* ``fields.gpkg`` — 12 non-overlapping rectangular field polygons (``field_id``
  1..12) aligned to the pixel grid, each with a uniform crop type, and one
  field partly over a strip of no data pixels;
* ``plots.gpkg`` — field survey points with a ``crop`` column;
* ``roi.gpkg`` — a region of interest covering the left half of the scene.
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
EPSG = 32630
FIELD_PX = 20  # fields are 20 x 20 pixels (200 m x 200 m)

# (Blue, Green, Red, NIR) reflectance x 10000 for each crop type
CROP_SPECTRA = {
    "wheat": (400, 700, 500, 3000),
    "grass": (500, 900, 600, 2500),
    "bare": (1100, 1200, 1300, 1600),
}
CROPS = ("wheat", "grass", "bare")


def _field_grid():
    """Yield (field_id, row0, col0, crop) for the 3 x 4 grid of fields."""
    fid = 0
    for r_idx in range(HEIGHT // FIELD_PX):
        for c_idx in range(WIDTH // FIELD_PX):
            fid += 1
            yield fid, r_idx * FIELD_PX, c_idx * FIELD_PX, CROPS[fid % len(CROPS)]


def make_zonal_scene(out_dir: str, seed: int = 0) -> Dict[str, str]:
    """Write the synthetic scene to ``out_dir`` and return the file paths."""
    # pylint: disable=too-many-locals
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    img = np.zeros((4, HEIGHT, WIDTH), dtype=np.uint16)
    fields = []
    for fid, row0, col0, crop in _field_grid():
        spec = np.array(CROP_SPECTRA[crop], dtype=float)[:, None, None]
        noise = rng.normal(0, 40, (4, FIELD_PX, FIELD_PX))
        img[:, row0:row0 + FIELD_PX, col0:col0 + FIELD_PX] = np.clip(
            spec + noise, 1, 10000
        ).astype(np.uint16)
        x_min = X0 + col0 * RES
        y_max = Y0 - row0 * RES
        fields.append(
            {
                "field_id": fid,
                "crop": crop,
                "geometry": box(x_min, y_max - FIELD_PX * RES,
                                x_min + FIELD_PX * RES, y_max),
            }
        )
    img[:, :, -3:] = 0  # a strip of no data on the right edge

    s2_img = os.path.join(out_dir, "s2_sref.tif")
    with rasterio.open(
        s2_img, "w", driver="GTiff", height=HEIGHT, width=WIDTH, count=4,
        dtype="uint16", crs=f"EPSG:{EPSG}", transform=from_origin(X0, Y0, RES, RES),
        nodata=0,
    ) as dst:
        dst.write(img)
        for idx, name in enumerate(("Blue", "Green", "Red", "NIR")):
            dst.set_band_description(idx + 1, name)

    fields_vec = os.path.join(out_dir, "fields.gpkg")
    fields_gdf = gpd.GeoDataFrame(fields, crs=f"EPSG:{EPSG}")
    fields_gdf.to_file(fields_vec, driver="GPKG")

    plots = []
    for fld in fields:
        x_min, y_min, x_max, y_max = fld["geometry"].bounds
        for _ in range(4):
            plots.append(
                {
                    "crop": fld["crop"],
                    "geometry": Point(rng.uniform(x_min + 15, x_max - 45),
                                      rng.uniform(y_min + 15, y_max - 15)),
                }
            )
    plots_vec = os.path.join(out_dir, "plots.gpkg")
    gpd.GeoDataFrame(plots, crs=f"EPSG:{EPSG}").to_file(plots_vec, driver="GPKG")

    roi_vec = os.path.join(out_dir, "roi.gpkg")
    gpd.GeoDataFrame(
        {"name": ["west"]},
        geometry=[box(X0, Y0 - HEIGHT * RES, X0 + WIDTH * RES / 2, Y0)],
        crs=f"EPSG:{EPSG}",
    ).to_file(roi_vec, driver="GPKG")
    return {"s2": s2_img, "fields": fields_vec, "plots": plots_vec, "roi": roi_vec}
