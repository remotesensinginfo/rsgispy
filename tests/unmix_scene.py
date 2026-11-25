"""
Synthetic wetland scene used by the spectral unmixing (TropWet) worked
example test (``tests/test_spectral_unmixing_example.py``) and to regenerate
the figures in ``docs/guides/images/unmix_*.png``.

The scene is a time series of Landsat-like surface reflectance images (Blue,
Green, Red, NIR, SWIR1, SWIR2; ``uint16`` reflectance x 10000, no data 0) in
UTM zone 34S (EPSG:32734, 30 m pixels), plus a coarser (90 m) DEM. Every
pixel is a linear mixture of three endmembers (Water, Vegetation, Sand) with
known fractions, except for a burned patch, whose spectrum is dark and is not
a mixture of the endmembers. The 60 x 80 pixel scene is divided into blocks
(rows x columns, see ``REGIONS``):

* ``lake`` — open water (pure Water);
* ``efv`` — emergent flooded vegetation (50 % Water, 50 % Vegetation);
* ``forest`` — dense vegetation (pure Vegetation);
* ``hill`` — a wet-looking mixture (45 % Water, 55 % Vegetation) on a slope
  steeper than 2.5 degrees;
* ``wet_sand`` — 50 % Water, 50 % Sand;
* ``flood`` — five strips of 4 columns, flooded (85 % Water, 15 %
  Vegetation) on 1 to 5 of the 5 dates and otherwise sparse vegetation (40 %
  Vegetation, 60 % Sand), so the water frequency is 20, 40, 60, 80 and 100 %;
* ``sparse_veg`` — 50 % Vegetation, 50 % Sand;
* ``dunes`` — dry bare sand (pure Sand);
* ``burn`` — a freshly burned area;
* ``mixed`` — a third of each endmember;
* ``scrub`` — 60 % Vegetation, 40 % Sand;
* ``sand_water`` — 20 % Water, 80 % Sand.

The last 3 columns are no data on every date and the north-west corner of
the lake is cloud-masked (no data) on the second date. The sand spectrum has
an NDVI of 0.165 and an MNDWI of -0.2, so pure sand pixels meet the narrow
TropWet sand thresholds.

The true fractions of the median composite are written as a 3-band image
(``truth``, -1 where undefined: the burned patch and the no data strip).
"""

import os
from typing import Dict, List, Tuple

import numpy as np
import rasterio
from rasterio.transform import from_origin

X0 = 600000.0
Y0 = 8300000.0
RES = 30.0
EPSG = 32734
HEIGHT = 60
WIDTH = 80
N_DATES = 5
BAND_NAMES = ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"]
WAVELENGTHS = [482, 561, 655, 865, 1609, 2201]  # Landsat 8 OLI band centres
# Reflectance x 10000 for each band.
ENDMEMBERS = {
    "Water": (700, 800, 650, 300, 120, 70),  # turbid water
    "Vegetation": (300, 800, 300, 4000, 1800, 800),
    "Sand": (1200, 1600, 2000, 2790, 2400, 2000),
}
EM_NAMES = list(ENDMEMBERS)
BURN_SPECTRUM = (700, 800, 1000, 700, 1100, 1000)
# Region name: ((row_start, row_end), (col_start, col_end)), end exclusive.
REGIONS: Dict[str, Tuple[Tuple[int, int], Tuple[int, int]]] = {
    "lake": ((0, 20), (0, 20)),
    "efv": ((0, 20), (20, 40)),
    "forest": ((0, 20), (40, 60)),
    "hill": ((0, 20), (60, 77)),
    "wet_sand": ((20, 40), (0, 20)),
    "flood": ((20, 40), (20, 40)),
    "sparse_veg": ((20, 40), (40, 60)),
    "dunes": ((20, 40), (60, 77)),
    "burn": ((40, 60), (0, 20)),
    "mixed": ((40, 60), (20, 40)),
    "scrub": ((40, 60), (40, 60)),
    "sand_water": ((40, 60), (60, 77)),
}
# (Water, Vegetation, Sand) fractions of each region.
REGION_FRACS = {
    "lake": (1.0, 0.0, 0.0),
    "efv": (0.5, 0.5, 0.0),
    "forest": (0.0, 1.0, 0.0),
    "hill": (0.45, 0.55, 0.0),
    "wet_sand": (0.5, 0.0, 0.5),
    "sparse_veg": (0.0, 0.5, 0.5),
    "dunes": (0.0, 0.0, 1.0),
    "mixed": (1 / 3, 1 / 3, 1 / 3),
    "scrub": (0.0, 0.6, 0.4),
    "sand_water": (0.2, 0.0, 0.8),
}
FLOODED = (0.85, 0.15, 0.0)
DRY = (0.0, 0.4, 0.6)
NODATA_COLS = 3
CLOUD = ((0, 10), (0, 10), 1)  # rows, cols, date index
HILL_DEM_ROWS = 8  # 90 m DEM rows covered by the hill (30 m rows 0-23)
HILL_DEM_COL0 = 19  # first 90 m DEM column of the hill (30 m column 57)
HILL_GRADIENT = 0.12  # rise / run of the hill (about 6.8 degrees)


def region_slices(name: str) -> Tuple[slice, slice]:
    """Return the (row, column) slices of a region."""
    (r0, r1), (c0, c1) = REGIONS[name]
    return slice(r0, r1), slice(c0, c1)


def flood_strip_cols(n_flooded: int) -> slice:
    """Columns of the flood strip that is flooded on ``n_flooded`` dates."""
    c0 = REGIONS["flood"][1][0] + 4 * (n_flooded - 1)
    return slice(c0, c0 + 4)


def _fractions(date_idx: int) -> np.ndarray:
    """(3, HEIGHT, WIDTH) Water, Vegetation, Sand fractions for a date."""
    fracs = np.zeros((3, HEIGHT, WIDTH))
    for name, frac in REGION_FRACS.items():
        rows, cols = region_slices(name)
        fracs[:, rows, cols] = np.array(frac)[:, np.newaxis, np.newaxis]
    rows = region_slices("flood")[0]
    for n_flooded in range(1, N_DATES + 1):
        state = FLOODED if date_idx < n_flooded else DRY
        fracs[:, rows, flood_strip_cols(n_flooded)] = np.array(state)[
            :, np.newaxis, np.newaxis]
    return fracs


def _write(path: str, arr: np.ndarray, res: float = RES, nodata=None,
           band_names=None) -> str:
    with rasterio.open(
        path, "w", driver="GTiff", height=arr.shape[1], width=arr.shape[2],
        count=arr.shape[0], dtype=arr.dtype, crs=f"EPSG:{EPSG}",
        transform=from_origin(X0, Y0, res, res), nodata=nodata,
    ) as dst:
        dst.write(arr)
        for idx, name in enumerate(band_names or [], start=1):
            dst.set_band_description(idx, name)
    return path


def _make_dem(rng: np.random.Generator) -> np.ndarray:
    """90 m DEM: flat floodplain (1000 m) with a hill sloping up to the east."""
    dem = 1000.0 + rng.normal(0.0, 0.1, size=(HEIGHT // 3, WIDTH // 3 + 1))
    cols = np.arange(dem.shape[1]) - HILL_DEM_COL0 + 1
    rise = np.clip(cols, 0, None) * 90.0 * HILL_GRADIENT
    dem[:HILL_DEM_ROWS] += rise[np.newaxis, :]
    return dem[np.newaxis].astype(np.float32)


def make_unmix_scene(out_dir: str, seed: int = 0,
                     noise_sd: float = 20.0) -> Dict[str, object]:
    """
    Write the synthetic scene to ``out_dir`` and return the paths ``dates``
    (list of surface reflectance images, one per date), ``dem`` and
    ``truth``.
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    em_arr = np.array([ENDMEMBERS[name] for name in EM_NAMES], dtype=np.float64)
    burn_rows, burn_cols = region_slices("burn")
    date_imgs: List[str] = []
    for date_idx in range(N_DATES):
        refl = np.einsum("khw,kb->bhw", _fractions(date_idx), em_arr)
        refl[:, burn_rows, burn_cols] = np.array(BURN_SPECTRUM)[
            :, np.newaxis, np.newaxis]
        refl += rng.normal(0.0, noise_sd, size=refl.shape)
        refl = np.clip(np.round(refl), 1, 10000).astype(np.uint16)
        refl[:, :, WIDTH - NODATA_COLS:] = 0
        if date_idx == CLOUD[2]:
            refl[:, slice(*CLOUD[0]), slice(*CLOUD[1])] = 0
        date_imgs.append(_write(
            os.path.join(out_dir, f"sref_date{date_idx + 1}.tif"), refl,
            nodata=0, band_names=BAND_NAMES))

    # The median composite takes the state seen on most dates.
    truth = _fractions(N_DATES // 2)
    truth[:, burn_rows, burn_cols] = -1.0
    truth[:, :, WIDTH - NODATA_COLS:] = -1.0
    return {
        "dates": date_imgs,
        "dem": _write(os.path.join(out_dir, "dem_90m.tif"),
                      _make_dem(rng), res=RES * 3, band_names=["elevation"]),
        "truth": _write(os.path.join(out_dir, "true_fractions.tif"),
                        truth.astype(np.float32), band_names=EM_NAMES),
    }
