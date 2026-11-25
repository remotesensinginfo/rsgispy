"""
Synthetic Sentinel-2 Level-2A scenes used by the analysis-ready data worked
example test (``tests/test_analysis_ready_data_example.py``).

Each scene is a minimal ESA ``.SAFE`` directory (``MTD_MSIL2A.xml`` plus
GeoTIFF band files in ``GRANULE/<id>/IMG_DATA/R10m|R20m|R60m``) laid out like a
real product, so it can be read by
:func:`rsgispy.imagecalibration.sensorlvl2data.create_stacked_sen2_esa_l2a_img`.
The land surface is an analytic function of map coordinates, so every band
and resolution (10, 20 and 60 m) is consistent and a scene can be shifted by
any (sub-pixel) amount to simulate misregistration. Each date has a cloud
(SCL 9) and its shadow (SCL 3) in a different place, and processing baseline
04.00 offsets (``BOA_ADD_OFFSET = -1000``) are applied, as in real data.
"""

import os
import shutil
import zipfile
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import rasterio
from rasterio.transform import from_origin

EPSG = 32630
Y0 = 5700000.0
SIZE_10M = 96  # 10 m pixels per tile side (so 48 at 20 m and 16 at 60 m)
TILE_X0 = {"T30UVD": 500000.0, "T30UWD": 500640.0}  # tiles overlap by 32 px

# Band code -> (resolution folder, veg reflectance, soil reflectance)
_BANDS = {
    "B01": (60, 0.03, 0.08),
    "B02": (10, 0.04, 0.10),
    "B03": (10, 0.07, 0.14),
    "B04": (10, 0.04, 0.18),
    "B05": (20, 0.10, 0.20),
    "B06": (20, 0.25, 0.22),
    "B07": (20, 0.32, 0.24),
    "B08": (10, 0.38, 0.26),
    "B8A": (20, 0.40, 0.27),
    "B09": (60, 0.30, 0.20),
    "B11": (20, 0.20, 0.32),
    "B12": (20, 0.10, 0.28),
}
BOA_OFFSET = -1000


def _veg_fraction(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Smooth, textured vegetation fraction (0-1) from map coordinates."""
    val = (
        0.5
        + 0.22 * np.sin(x / 61.0) * np.cos(y / 73.0)
        + 0.15 * np.sin((x + y) / 37.0)
        + 0.10 * np.cos((x - 1.7 * y) / 29.0)
    )
    return np.clip(val, 0.0, 1.0)


def _blob(x: np.ndarray, y: np.ndarray, cx: float, cy: float, rad: float):
    return ((x - cx) ** 2 + (y - cy) ** 2) <= rad**2


def _grid(tile: str, res: float) -> Tuple[np.ndarray, np.ndarray]:
    n_pxl = int(SIZE_10M * 10 / res)
    x0 = TILE_X0[tile]
    cols = x0 + (np.arange(n_pxl) + 0.5) * res
    rows = Y0 - (np.arange(n_pxl) + 0.5) * res
    return np.meshgrid(cols, rows)


def _write(path: str, arr: np.ndarray, tile: str, res: float) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype=arr.dtype,
        crs=f"EPSG:{EPSG}",
        transform=from_origin(TILE_X0[tile], Y0, res, res),
        nodata=0,
    ) as dst:
        dst.write(arr, 1)


def _mtd_xml(prod_uri: str, image_files: Sequence[str]) -> str:
    """Minimal Sentinel-2 L2A ``MTD_MSIL2A.xml`` (processing baseline 04.00)."""
    img_tags = "\n".join(f"<IMAGE_FILE>{entry}</IMAGE_FILE>" for entry in image_files)
    phys = [(0, "B1"), (1, "B2"), (2, "B3"), (3, "B4"), (4, "B5"), (5, "B6"),
            (6, "B7"), (7, "B8"), (8, "B8A"), (9, "B9"), (11, "B11"), (12, "B12")]
    waves = [(60, 443), (10, 490), (10, 560), (10, 665), (20, 705), (20, 740),
             (20, 783), (10, 842), (20, 865), (60, 945), (20, 1610), (20, 2190)]
    offsets = "\n".join(
        f'<BOA_ADD_OFFSET band_id="{bid}">{BOA_OFFSET}</BOA_ADD_OFFSET>'
        for bid, _ in phys
    )
    spec = "\n".join(
        f'<Spectral_Information bandId="{bid}" physicalBand="{name}">'
        f"<RESOLUTION>{res}</RESOLUTION><Wavelength><MIN>{wl - 10}</MIN>"
        f"<MAX>{wl + 10}</MAX><CENTRAL>{wl}</CENTRAL></Wavelength>"
        "</Spectral_Information>"
        for (bid, name), (res, wl) in zip(phys, waves)
    )
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<n1:Level-2A_User_Product xmlns:n1="https://psd-15.sentinel2.eo.esa.int/PSD/User_Product_Level-2A.xsd">
<n1:General_Info>
<Product_Info>
<PRODUCT_START_TIME>2024-06-02T11:21:19.024Z</PRODUCT_START_TIME>
<PRODUCT_STOP_TIME>2024-06-02T11:21:29.024Z</PRODUCT_STOP_TIME>
<GENERATION_TIME>2024-06-02T14:00:00.000Z</GENERATION_TIME>
<PRODUCT_URI>{prod_uri}</PRODUCT_URI>
<PROCESSING_LEVEL>Level-2A</PROCESSING_LEVEL>
<PRODUCT_TYPE>S2MSI2A</PRODUCT_TYPE>
<PROCESSING_BASELINE>05.10</PROCESSING_BASELINE>
<Datatake datatakeIdentifier="GS2A_20240602T112119">
<SPACECRAFT_NAME>Sentinel-2A</SPACECRAFT_NAME>
<DATATAKE_TYPE>INS-NOBS</DATATAKE_TYPE>
<DATATAKE_SENSING_START>2024-06-02T11:21:19.024Z</DATATAKE_SENSING_START>
<SENSING_ORBIT_NUMBER>37</SENSING_ORBIT_NUMBER>
<SENSING_ORBIT_DIRECTION>DESCENDING</SENSING_ORBIT_DIRECTION>
</Datatake>
<Query_Options><PRODUCT_FORMAT>SAFE</PRODUCT_FORMAT></Query_Options>
<Product_Organisation><Granule_List>
<Granule datastripIdentifier="DS_1" granuleIdentifier="GR_1" imageFormat="GeoTIFF">
{img_tags}
</Granule>
</Granule_List></Product_Organisation>
</Product_Info>
<Product_Image_Characteristics>
<Special_Values><SPECIAL_VALUE_TEXT>NODATA</SPECIAL_VALUE_TEXT>
<SPECIAL_VALUE_INDEX>0</SPECIAL_VALUE_INDEX></Special_Values>
<QUANTIFICATION_VALUES_LIST>
<BOA_QUANTIFICATION_VALUE unit="none">10000</BOA_QUANTIFICATION_VALUE>
<AOT_QUANTIFICATION_VALUE unit="none">1000.0</AOT_QUANTIFICATION_VALUE>
<WVP_QUANTIFICATION_VALUE unit="none">1000.0</WVP_QUANTIFICATION_VALUE>
</QUANTIFICATION_VALUES_LIST>
<BOA_ADD_OFFSET_VALUES_LIST>
{offsets}
</BOA_ADD_OFFSET_VALUES_LIST>
<Spectral_Information_List>
{spec}
</Spectral_Information_List>
</Product_Image_Characteristics>
</n1:General_Info>
</n1:Level-2A_User_Product>
"""


def product_name(tile: str, date: str) -> str:
    """Return an ESA-style product name (without ``.SAFE``)."""
    return f"S2A_MSIL2A_{date}T112119_N0510_R037_{tile}_{date}T125034"


def make_safe(
    root: str,
    tile: str,
    date: str,
    cloud_xy: Tuple[float, float],
    shift: Tuple[float, float] = (0.0, 0.0),
    nodata_cols: int = 0,
) -> str:
    """
    Write one synthetic Sentinel-2 L2A ``.SAFE`` directory and return its path.

    :param cloud_xy: cloud centre as fractions (0-1) of the tile width/height.
    :param shift: (dx, dy) shift of the image content in metres (a positive
                  ``dx`` moves features east), i.e. the misregistration.
    :param nodata_cols: number of 10 m columns at the left of the tile with no
                        data (as at the edge of an orbit swath).
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    name = product_name(tile, date)
    safe_dir = os.path.join(root, f"{name}.SAFE")
    img_root = os.path.join(safe_dir, "GRANULE", f"L2A_{tile}", "IMG_DATA")
    entries: List[str] = []
    extent = SIZE_10M * 10.0
    cx = TILE_X0[tile] + cloud_xy[0] * extent
    cy = Y0 - cloud_xy[1] * extent
    for res in (10, 20, 60):
        res_dir = os.path.join(img_root, f"R{res}m")
        os.makedirs(res_dir, exist_ok=True)
        xx, yy = _grid(tile, res)
        veg = _veg_fraction(xx - shift[0], yy - shift[1])
        cloud = _blob(xx, yy, cx, cy, 100.0)
        shadow = _blob(xx, yy, cx + 200.0, cy - 150.0, 80.0) & ~cloud
        # The (20 m) SCL cloud and shadow classes are a little larger than
        # the features, as a cloud mask should be.
        scl_cloud = _blob(xx, yy, cx, cy, 130.0)
        scl_shadow = _blob(xx, yy, cx + 200.0, cy - 150.0, 110.0) & ~scl_cloud
        nodata = xx < TILE_X0[tile] + nodata_cols * 10.0
        bands = [b for b, info in _BANDS.items() if info[0] == res]
        if res == 20:  # the 10 m bands are also provided at 20 m in L2A
            bands += ["B02", "B03", "B04"]
        for band in bands:
            _, veg_r, soil_r = _BANDS[band]
            refl = veg * veg_r + (1.0 - veg) * soil_r
            refl = np.where(cloud, 0.55, refl)
            refl = np.where(shadow, refl * 0.3, refl)
            dn = np.round(refl * 10000.0 - BOA_OFFSET).astype(np.uint16)
            dn[nodata] = 0
            stem = f"{tile}_{date}T112119_{band}_{res}m"
            _write(os.path.join(res_dir, f"{stem}.tif"), dn, tile, res)
            entries.append(f"GRANULE/L2A_{tile}/IMG_DATA/R{res}m/{stem}")
        if res == 20:
            scl = np.full(xx.shape, 4, dtype=np.uint8)  # vegetation
            scl[veg < 0.35] = 5  # bare soil
            scl[scl_shadow] = 3
            scl[scl_cloud] = 9
            scl[nodata] = 0
            stem = f"{tile}_{date}T112119_SCL_20m"
            _write(os.path.join(res_dir, f"{stem}.tif"), scl, tile, res)
            entries.append(f"GRANULE/L2A_{tile}/IMG_DATA/R20m/{stem}")
    with open(os.path.join(safe_dir, "MTD_MSIL2A.xml"), "w", encoding="utf-8") as f_obj:
        f_obj.write(_mtd_xml(f"{name}.SAFE", entries))
    return safe_dir


# Three dates per tile: the cloud moves, the second date is misregistered by
# (+15 m, -10 m) and the third has a strip of no data at the swath edge.
DATES = (
    ("20240602", (0.25, 0.3), (0.0, 0.0), 0),
    ("20240612", (0.7, 0.7), (15.0, -10.0), 0),
    ("20240622", (0.6, 0.25), (0.0, 0.0), 12),
)


def make_ard_scenes(
    root: str, tiles: Sequence[str] = ("T30UVD", "T30UWD")
) -> Dict[str, List[str]]:
    """Write three dates for each tile; return ``{tile: [safe_dir, ...]}``."""
    os.makedirs(root, exist_ok=True)
    return {
        tile: [
            make_safe(root, tile, date, cloud, shift=shift, nodata_cols=nd_cols)
            for date, cloud, shift, nd_cols in DATES
        ]
        for tile in tiles
    }


def zip_safe(safe_dir: str, out_dir: str, name: Optional[str] = None) -> str:
    """Zip a ``.SAFE`` directory as the Copernicus Data Space download does."""
    name = name or os.path.basename(safe_dir)
    zip_path = os.path.join(out_dir, f"{name}.zip")
    parent = os.path.dirname(safe_dir)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zip_obj:
        for dir_path, _, files in os.walk(safe_dir):
            for fname in files:
                full = os.path.join(dir_path, fname)
                zip_obj.write(full, os.path.relpath(full, parent))
    return zip_path


def remove_safe(safe_dir: str) -> None:
    """Delete a ``.SAFE`` directory (after zipping it)."""
    shutil.rmtree(safe_dir)
