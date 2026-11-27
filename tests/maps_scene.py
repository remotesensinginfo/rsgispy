"""
Synthetic scene used by the "Making maps" worked example test
(``tests/test_making_maps_example.py``) and to regenerate the figures in
``docs/guides/images/maps_*.png``.

The scene is a small landscape in UTM zone 30N (EPSG:32630, 10 m pixels):

* a 4-band Sentinel-2-like surface reflectance image (Blue, Green, Red, NIR;
  ``uint16`` reflectance x 10000, no data 0, with a no data corner);
* a land cover raster (Water, Forest, Grass, Urban) with a colour table, as
  produced by the pixel classification guide;
* a DEM (elevation in metres, no data -9999) with a valley along the river;
* a continuous 0-1 score raster (e.g. a multi-criteria suitability score or
  a vegetation fraction; no data -9999);
* administrative regions (polygons), a river (line) and survey sites (points);
* an overview layer (a coarse "country" outline in EPSG:4326).
"""

import os
from typing import Dict

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString, Point, Polygon, box

X0 = 500000.0
Y0 = 5200000.0
RES = 10.0
EPSG = 32630

CLASS_NAMES = {1: "Water", 2: "Forest", 3: "Grass", 4: "Urban"}
CLASS_COLOURS = {
    1: (31, 120, 180, 255),
    2: (51, 160, 44, 255),
    3: (178, 223, 138, 255),
    4: (128, 128, 128, 255),
}
# (Blue, Green, Red, NIR) reflectance x 10000
CLASS_SPECTRA = {
    1: (600, 500, 300, 150),
    2: (300, 600, 300, 3500),
    3: (500, 900, 600, 2500),
    4: (1200, 1300, 1400, 1800),
}


def _smooth_field(rng: np.random.Generator, height: int, width: int,
                  scale: float) -> np.ndarray:
    """Gaussian-smoothed random field (FFT filter), rescaled to 0-1."""
    noise = rng.normal(size=(height, width))
    fy = np.fft.fftfreq(height)[:, np.newaxis]
    fx = np.fft.fftfreq(width)[np.newaxis, :]
    kernel = np.exp(-2.0 * (np.pi * scale) ** 2 * (fx ** 2 + fy ** 2))
    field = np.real(np.fft.ifft2(np.fft.fft2(noise) * kernel))
    field -= field.min()
    return field / max(field.max(), 1e-9)


def _write(path: str, arr: np.ndarray, nodata=None, band_names=None,
           ctab=None) -> str:
    if arr.ndim == 2:
        arr = arr[np.newaxis]
    with rasterio.open(
        path, "w", driver="GTiff", height=arr.shape[1], width=arr.shape[2],
        count=arr.shape[0], dtype=arr.dtype, crs=f"EPSG:{EPSG}",
        transform=from_origin(X0, Y0, RES, RES), nodata=nodata,
    ) as dst:
        dst.write(arr)
        if band_names:
            for idx, name in enumerate(band_names, start=1):
                dst.set_band_description(idx, name)
        if ctab is not None:
            dst.write_colormap(1, ctab)
    return path


def make_maps_scene(out_dir: str, height: int = 120, width: int = 160,
                    seed: int = 0) -> Dict[str, str]:
    """
    Write the synthetic scene to ``out_dir`` and return the file paths:
    ``s2``, ``cls``, ``score``, ``dem``, ``regions``, ``rivers``, ``sites`` and
    ``overview``.
    """
    # pylint: disable=too-many-locals
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows, cols = np.mgrid[0:height, 0:width]

    # River: a sine wave running left to right, 3 pixels wide.
    river_row = height * 0.55 + height * 0.12 * np.sin(cols / width * 2 * np.pi)
    river = np.abs(rows - river_row) < 1.6

    veg = _smooth_field(rng, height, width, scale=height / 10.0)
    urban = _smooth_field(rng, height, width, scale=height / 12.0)
    cls = np.full((height, width), 3, dtype=np.uint8)  # Grass
    cls[veg > 0.55] = 2  # Forest
    cls[(urban > 0.72) & (veg < 0.55)] = 4  # Urban
    cls[river] = 1
    cls[(rows - height * 0.2) ** 2 + (cols - width * 0.8) ** 2
        < (height * 0.08) ** 2] = 1  # lake

    s2 = np.zeros((4, height, width), dtype=np.float64)
    for cls_id, spec in CLASS_SPECTRA.items():
        msk = cls == cls_id
        for b_idx, val in enumerate(spec):
            s2[b_idx][msk] = val
    s2 *= 0.85 + 0.3 * veg[np.newaxis]  # within-class variation
    s2 += rng.normal(0, 15, size=s2.shape)
    s2 = np.clip(s2, 1, 10000).astype(np.uint16)
    nodata_corner = (rows + cols) < min(height, width) * 0.12
    s2[:, nodata_corner] = 0
    cls[nodata_corner] = 0

    score = np.clip(0.2 + 0.7 * veg - 0.4 * (cls == 4), 0, 1).astype(np.float32)
    score[(cls == 1) | nodata_corner] = -9999.0

    # Elevation (m): hills with a valley along the river, which is the
    # lowest ground (generated from its own random stream so the other
    # layers do not change).
    dem_rng = np.random.default_rng(seed + 100)
    hills = (0.6 * _smooth_field(dem_rng, height, width, scale=height / 12.0)
             + 0.3 * _smooth_field(dem_rng, height, width, scale=height / 30.0)
             + 0.1 * _smooth_field(dem_rng, height, width, scale=height / 60.0))
    valley = np.abs(rows - river_row) / height
    dem = (40.0 + 260.0 * hills * np.clip(valley * 4.0, 0.1, 1.0)
           + 300.0 * valley).astype(np.float32)

    paths = {
        "s2": _write(os.path.join(out_dir, "s2_sref.tif"), s2, nodata=0,
                     band_names=["Blue", "Green", "Red", "NIR"]),
        "cls": _write(os.path.join(out_dir, "land_cover.tif"), cls, nodata=0,
                      ctab={0: (0, 0, 0, 0), **CLASS_COLOURS}),
        "score": _write(os.path.join(out_dir, "score.tif"), score,
                        nodata=-9999.0, band_names=["score"]),
        "dem": _write(os.path.join(out_dir, "dem.tif"), dem, nodata=-9999.0,
                      band_names=["elevation"]),
    }

    x1 = X0 + width * RES
    y1 = Y0 - height * RES
    xm = X0 + width * RES * 0.45
    ym = Y0 - height * RES * 0.5
    regions = gpd.GeoDataFrame(
        {"name": ["North West", "North East", "South West", "South East"]},
        geometry=[box(X0, ym, xm, Y0), box(xm, ym, x1, Y0),
                  box(X0, y1, xm, ym), box(xm, y1, x1, ym)],
        crs=f"EPSG:{EPSG}",
    )
    paths["regions"] = os.path.join(out_dir, "regions.gpkg")
    regions.to_file(paths["regions"])

    line_cols = np.linspace(0, width, 60)
    line_rows = height * 0.55 + height * 0.12 * np.sin(line_cols / width * 2 * np.pi)
    rivers = gpd.GeoDataFrame(
        {"name": ["River"]},
        geometry=[LineString([(X0 + c * RES, Y0 - r * RES)
                              for c, r in zip(line_cols, line_rows)])],
        crs=f"EPSG:{EPSG}",
    )
    paths["rivers"] = os.path.join(out_dir, "rivers.gpkg")
    rivers.to_file(paths["rivers"])

    site_rc = rng.uniform([0.15, 0.15], [0.9, 0.9], size=(8, 2))
    sites = gpd.GeoDataFrame(
        {"site_id": np.arange(1, 9)},
        geometry=[Point(X0 + c * width * RES, Y0 - r * height * RES)
                  for r, c in site_rc],
        crs=f"EPSG:{EPSG}",
    )
    paths["sites"] = os.path.join(out_dir, "sites.gpkg")
    sites.to_file(paths["sites"])

    # A coarse "country" outline around the scene (lon -3, lat 46.9).
    overview = gpd.GeoDataFrame(
        {"name": ["Country"]},
        geometry=[Polygon([(-4.8, 43.4), (-1.8, 43.3), (3.0, 42.6),
                           (7.5, 43.8), (6.2, 46.4), (8.1, 49.0),
                           (2.5, 51.0), (-1.6, 49.6), (-4.7, 48.4),
                           (-1.2, 46.2)])],
        crs="EPSG:4326",
    )
    paths["overview"] = os.path.join(out_dir, "overview.gpkg")
    overview.to_file(paths["overview"])
    return paths
