"""
Tests for rsgispy.imagecalibration (radiometry, solarangles, sensorlvl1data,
sensorlvl2data) and rsgispy.tools.sensors.
"""

import datetime
import math
import os
import tarfile
import warnings
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio

import rsgispy
import rsgispy.imagecalibration as imgcal
import rsgispy.imagecalibration.sensorlvl1data as l1
import rsgispy.imagecalibration.sensorlvl2data as l2
import rsgispy.imagecalibration.solarangles as solarangles
import rsgispy.tools.sensors as sensors

from conftest import write_raster


def _write_gtiff(
    path: str,
    data: np.ndarray,
    *,
    res: float = 10.0,
    x0: float = 500000.0,
    y0: float = 5700000.0,
    epsg: int = 32630,
    nodata=None,
    band_names=None,
) -> str:
    """Write a single- or multi-band GeoTIFF helper for tests."""
    return write_raster(
        path,
        data,
        nodata=nodata,
        crs=f"EPSG:{epsg}",
        origin=(x0, y0),
        pixel_size=res,
        band_names=band_names,
    )


def _build_synthetic_landsat_mtl(
    prod_id: str,
    rescaling_group: str,
    bands,
    *,
    qa_key: str = "FILE_NAME_QUALITY_L1_PIXEL",
    qa_filename: str = "QA_PIXEL.TIF",
    sun_elev: float = 45.0,
) -> str:
    """Create minimal Landsat Collection 2 MTL text content."""
    lines = [
        "GROUP = LANDSAT_METADATA_FILE",
        "  GROUP = PRODUCT_CONTENTS",
        f'    LANDSAT_PRODUCT_ID = "{prod_id}"',
    ]
    for bnum, fname, _, _ in bands:
        lines.append(f'    FILE_NAME_BAND_{bnum} = "{fname}"')
    lines.extend(
        [
            f'    {qa_key} = "{qa_filename}"',
            "  END_GROUP = PRODUCT_CONTENTS",
            "  GROUP = IMAGE_ATTRIBUTES",
            f"    SUN_ELEVATION = {sun_elev:.4f}",
            "  END_GROUP = IMAGE_ATTRIBUTES",
            f"  GROUP = {rescaling_group}",
        ]
    )
    for bnum, _, mult, add in bands:
        lines.append(f"    REFLECTANCE_MULT_BAND_{bnum} = {mult:.8e}")
    for bnum, _, mult, add in bands:
        lines.append(f"    REFLECTANCE_ADD_BAND_{bnum} = {add:.8f}")
    lines.extend(
        [
            f"  END_GROUP = {rescaling_group}",
            "END_GROUP = LANDSAT_METADATA_FILE",
        ]
    )
    return "\n".join(lines) + "\n"


def _build_sen2_mtd_xml(
    prod_uri: str,
    image_files,
    *,
    namespace: str = "https://psd-15.sentinel2.eo.esa.int/PSD/User_Product_Level-2A.xsd",
    quant_boa: int = 10000,
    boa_offsets=None,
) -> str:
    """Create minimal Sentinel-2 L2A MTD_MSIL2A.xml content."""
    img_tags = "\n".join(
        f"            <IMAGE_FILE>{entry}</IMAGE_FILE>" for entry in image_files
    )
    offset_block = ""
    if boa_offsets:
        items = "\n".join(
            f'          <BOA_ADD_OFFSET band_id="{bid}">{val}</BOA_ADD_OFFSET>'
            for bid, val in boa_offsets.items()
        )
        offset_block = (
            "        <BOA_ADD_OFFSET_VALUES_LIST>\n"
            f"{items}\n"
            "        </BOA_ADD_OFFSET_VALUES_LIST>"
        )

    phys_bands = [
        (0, "B1", 60, 443),
        (1, "B02", 10, 490),
        (2, "B03", 10, 560),
        (3, "B04", 10, 665),
        (4, "B05", 20, 705),
        (5, "B06", 20, 740),
        (6, "B07", 20, 783),
        (7, "B8", 10, 842),
        (8, "B8A", 20, 865),
        (9, "B9", 60, 945),
        (11, "B11", 20, 1610),
        (12, "B12", 20, 2190),
    ]
    spec_items = "\n".join(
        f"""          <Spectral_Information bandId="{bid}" physicalBand="{pname}">
            <RESOLUTION>{res}</RESOLUTION>
            <Wavelength>
              <MIN>{wl - 10}</MIN>
              <MAX>{wl + 10}</MAX>
              <CENTRAL>{wl}</CENTRAL>
            </Wavelength>
          </Spectral_Information>"""
        for bid, pname, res, wl in phys_bands
    )

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<n1:Level-2A_User_Product xmlns:n1="{namespace}">
  <n1:General_Info>
    <Product_Info>
      <PRODUCT_START_TIME>2022-06-21T11:21:19.024Z</PRODUCT_START_TIME>
      <PRODUCT_STOP_TIME>2022-06-21T11:21:29.024Z</PRODUCT_STOP_TIME>
      <GENERATION_TIME>2022-06-21T14:00:00.000Z</GENERATION_TIME>
      <PRODUCT_URI>{prod_uri}</PRODUCT_URI>
      <PROCESSING_LEVEL>Level-2A</PROCESSING_LEVEL>
      <PRODUCT_TYPE>S2MSI2A</PRODUCT_TYPE>
      <PROCESSING_BASELINE>04.00</PROCESSING_BASELINE>
      <Datatake datatakeIdentifier="GS2A_20220621T112119">
        <SPACECRAFT_NAME>Sentinel-2A</SPACECRAFT_NAME>
        <DATATAKE_TYPE>INS-NOBS</DATATAKE_TYPE>
        <DATATAKE_SENSING_START>2022-06-21T11:21:19.024Z</DATATAKE_SENSING_START>
        <SENSING_ORBIT_NUMBER>37</SENSING_ORBIT_NUMBER>
        <SENSING_ORBIT_DIRECTION>DESCENDING</SENSING_ORBIT_DIRECTION>
      </Datatake>
      <Query_Options>
        <PRODUCT_FORMAT>SAFE</PRODUCT_FORMAT>
      </Query_Options>
      <Product_Organisation>
        <Granule_List>
          <Granule datastripIdentifier="DS_1" granuleIdentifier="GR_1" imageFormat="JPEG2000">
{img_tags}
          </Granule>
        </Granule_List>
      </Product_Organisation>
    </Product_Info>
    <Product_Image_Characteristics>
      <Special_Values>
        <SPECIAL_VALUE_TEXT>NODATA</SPECIAL_VALUE_TEXT>
        <SPECIAL_VALUE_INDEX>0</SPECIAL_VALUE_INDEX>
      </Special_Values>
      <QUANTIFICATION_VALUES_LIST>
        <BOA_QUANTIFICATION_VALUE unit="none">{quant_boa}</BOA_QUANTIFICATION_VALUE>
        <AOT_QUANTIFICATION_VALUE unit="none">1000.0</AOT_QUANTIFICATION_VALUE>
        <WVP_QUANTIFICATION_VALUE unit="none">1000.0</WVP_QUANTIFICATION_VALUE>
      </QUANTIFICATION_VALUES_LIST>
{offset_block}
      <Spectral_Information_List>
{spec_items}
      </Spectral_Information_List>
    </Product_Image_Characteristics>
  </n1:General_Info>
</n1:Level-2A_User_Product>
"""


# ---------------------------------------------------------------------------
# Scalar astronomy / atmospheric utilities
# ---------------------------------------------------------------------------


def test_julian_day_and_solar_distance_and_esun_and_aot():
    assert imgcal.get_julian_day(2000, 1, 1) == 2451545
    assert imgcal.get_julian_day(2020, 6, 21) == 2459022
    with pytest.raises(rsgispy.RSGISPyException):
        imgcal.get_julian_day(2020, 2, 31)

    d_peri = imgcal.calc_solar_distance(imgcal.get_julian_day(2020, 1, 4))
    d_aphe = imgcal.calc_solar_distance(imgcal.get_julian_day(2020, 7, 4))
    assert 0.98 < d_peri < 1.0 < d_aphe < 1.02

    with pytest.raises(rsgispy.RSGISPyException):
        imgcal.calc_solar_distance(float("nan"))

    esun = imgcal.get_esun_value(120.0, 0.25, 21, 6, 2020, 30.0)
    d = imgcal.calc_solar_distance(imgcal.get_julian_day(2020, 6, 21))
    expected_esun = (math.pi * 120.0 * (d**2)) / (
        0.25 * math.cos(math.radians(30.0))
    )
    assert math.isclose(esun, expected_esun, rel_tol=1e-9)

    with pytest.raises(rsgispy.RSGISPyException):
        imgcal.get_esun_value(100.0, 0.0, 21, 6, 2020, 30.0)
    with pytest.raises(rsgispy.RSGISPyException):
        imgcal.get_esun_value(100.0, 0.2, 21, 6, 2020, 90.0)

    assert math.isclose(
        imgcal.visibility_to_aot(25.0), (3.9449 / 25.0) + 0.08498, rel_tol=1e-9
    )
    with pytest.raises(rsgispy.RSGISPyException):
        imgcal.visibility_to_aot(0.0)


# ---------------------------------------------------------------------------
# Radiance -> TOA reflectance, Thermal brightness, DOS
# ---------------------------------------------------------------------------


def test_radiance_to_toa_refl(tmp_path):
    in_img = os.path.join(tmp_path, "rad.tif")
    out_img = os.path.join(tmp_path, "toa.tif")

    b1 = np.array([[100.0, 200.0], [-9999.0, 50.0]], dtype=np.float32)
    b2 = np.array([[80.0, -9999.0], [40.0, 120.0]], dtype=np.float32)
    _write_gtiff(
        in_img,
        np.stack([b1, b2], axis=0),
        nodata=-9999.0,
        band_names=["Blue", "Green"],
    )

    esun_entries = [SimpleNamespace(irradiance=1850.0), {"irradiance": 1550.0}]
    imgcal.radiance_to_toa_refl(
        in_img,
        out_img,
        scale_factor=1000.0,
        year=2020,
        month=6,
        day=21,
        solar_zenith=30.0,
        solar_irradiance=esun_entries,
        calc_stats=False,
        use_pbar=False,
    )

    jd = imgcal.get_julian_day(2020, 6, 21)
    d = imgcal.calc_solar_distance(jd)
    cos_z = math.cos(math.radians(30.0))

    with rasterio.open(out_img) as src:
        out_arr = src.read()
        assert src.descriptions == ("Blue", "Green")
        assert src.nodata == -9999.0

    # Valid pixel matches exact TOA reflectance formula; nodata is preserved
    exp_b1_00 = (math.pi * 100.0 * (d**2) * 1000.0) / (1850.0 * cos_z)
    assert math.isclose(float(out_arr[0, 0, 0]), exp_b1_00, rel_tol=1e-5)
    assert out_arr[0, 1, 0] == -9999.0
    assert out_arr[1, 0, 1] == -9999.0

    # Wrong band count raises
    with pytest.raises(rsgispy.RSGISPyException):
        imgcal.radiance_to_toa_refl(
            in_img,
            out_img,
            scale_factor=1.0,
            year=2020,
            month=6,
            day=21,
            solar_zenith=30.0,
            solar_irradiance=[1850.0],
            calc_stats=False,
            use_pbar=False,
        )


def test_landsat_thermal_rad_to_brightness(tmp_path):
    in_img = os.path.join(tmp_path, "therm.tif")
    out_img = os.path.join(tmp_path, "bt.tif")

    rad = np.array([[9.0, 10.5], [-1.0, -9999.0]], dtype=np.float32)
    _write_gtiff(in_img, rad, nodata=-9999.0)

    k1 = 774.8853
    k2 = 1321.0789
    coeffs = [imgcal.LandsatThermalCoeffs("ThermalB10", 1, k1, k2)]
    imgcal.landsat_thermal_rad_to_brightness(
        in_img,
        out_img,
        band_defs=coeffs,
        scale_factor=100.0,
        calc_stats=False,
        use_pbar=False,
    )

    with rasterio.open(out_img) as src:
        out = src.read(1)
        assert src.descriptions == ("ThermalB10",)
        assert src.nodata == -9999.0

    expected_00 = ((k2 / math.log((k1 / 9.0) + 1.0)) - 273.15) * 100.0
    assert math.isclose(float(out[0, 0]), expected_00, rel_tol=1e-4)
    # Non-positive radiance and nodata never produce NaN
    assert np.all(np.isfinite(out))
    assert out[1, 0] == -9999.0
    assert out[1, 1] == -9999.0


def test_perform_dos_calc(tmp_path):
    in_img = os.path.join(tmp_path, "toa.tif")
    out_img = os.path.join(tmp_path, "dos.tif")
    out_neg = os.path.join(tmp_path, "dos_neg.tif")

    # Dark pixel = 0.05 in b1, 100 values so percentile 0.01 picks ~0.05..0.06
    b1 = np.linspace(0.05, 0.50, 100, dtype=np.float32).reshape(10, 10)
    b1[0, 0] = 0.0  # nodata
    b1[0, 1] = 0.01  # darkest pixel (below percentile offset, tests clamp!)
    _write_gtiff(in_img, b1, nodata=0.0, band_names=["Red"])

    # non_negative=True clamps negative residuals to dark_obj_refl_val (0.02)
    imgcal.perform_dos_calc(
        in_img,
        out_img,
        non_negative=True,
        no_data_val=0.0,
        dark_obj_refl_val=0.02,
        dark_obj_percentile=5.0,
        copy_band_names=True,
        calc_stats=False,
        use_pbar=False,
    )

    with rasterio.open(out_img) as src:
        dos_arr = src.read(1)
        assert src.descriptions == ("Red",)
    assert dos_arr[0, 0] == 0.0
    assert math.isclose(float(dos_arr[0, 1]), 0.02, abs_tol=1e-6)
    assert np.all(dos_arr[dos_arr != 0.0] >= 0.02 - 1e-6)

    # non_negative=False allows negative values
    imgcal.perform_dos_calc(
        in_img,
        out_neg,
        non_negative=False,
        no_data_val=0.0,
        dark_obj_refl_val=0.0,
        dark_obj_percentile=10.0,
        calc_stats=False,
        use_pbar=False,
    )
    with rasterio.open(out_neg) as src:
        neg_arr = src.read(1)
    assert neg_arr[0, 1] < 0.0

    # dark_obj_percentile is on the 0-100 scale: a value in (0, 1) runs
    # without a warning, and values outside [0, 100] raise.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        imgcal.perform_dos_calc(
            in_img, str(tmp_path / "dos_frac.tif"), no_data_val=0.0,
            dark_obj_percentile=0.5, calc_stats=False, use_pbar=False,
        )
    for bad in (-1.0, 150.0):
        with pytest.raises(rsgispy.RSGISPyException, match="dark_obj_percentile"):
            imgcal.perform_dos_calc(
                in_img, str(tmp_path / "dos_bad.tif"), no_data_val=0.0,
                dark_obj_percentile=bad, calc_stats=False, use_pbar=False,
            )

    # All-nodata band raises
    nd_only = os.path.join(tmp_path, "all_nd.tif")
    _write_gtiff(nd_only, np.zeros((4, 4), dtype=np.float32), nodata=0.0)
    with pytest.raises(rsgispy.RSGISPyException):
        imgcal.perform_dos_calc(
            nd_only, out_img, no_data_val=0.0, calc_stats=False, use_pbar=False
        )


# ---------------------------------------------------------------------------
# Solar angles
# ---------------------------------------------------------------------------


def test_solar_azimuth_conventions_and_raster(tmp_path):
    assert solarangles.get_solar_irr_convention_solar_azimuth_from_usgs(-90.0) == 270.0
    assert solarangles.get_solar_irr_convention_solar_azimuth_from_usgs(90.0) == 90.0
    assert solarangles.get_solar_irr_convention_solar_azimuth_from_usgs(-180.0) == 180.0

    assert solarangles.get_solar_irr_convention_solar_azimuth_from_trad(0.0) == 180.0
    assert solarangles.get_solar_irr_convention_solar_azimuth_from_trad(90.0) == 90.0
    assert solarangles.get_solar_irr_convention_solar_azimuth_from_trad(-90.0) == 270.0
    assert solarangles.get_solar_irr_convention_solar_azimuth_from_trad(-180.0) == 0.0

    # London UTM 30N near solar noon on summer solstice -> Az ~ 180 deg, Zen ~ 28 deg
    in_img = os.path.join(tmp_path, "london.tif")
    out_aware = os.path.join(tmp_path, "angles_aware.tif")
    out_naive = os.path.join(tmp_path, "angles_naive.tif")
    _write_gtiff(
        in_img,
        np.ones((6, 6), dtype=np.float32),
        res=1000.0,
        x0=699000.0,
        y0=5712000.0,
        epsg=32630,
    )

    dt_aware = datetime.datetime(2020, 6, 21, 12, 0, 0, tzinfo=datetime.timezone.utc)
    dt_naive = datetime.datetime(2020, 6, 21, 12, 0, 0)

    solarangles.calc_solar_azimuth_zenith(
        in_img, dt_aware, out_aware, calc_stats=False, use_pbar=False
    )
    solarangles.calc_solar_azimuth_zenith(
        in_img, dt_naive, out_naive, calc_stats=False, use_pbar=False
    )

    with rasterio.open(out_aware) as s1, rasterio.open(out_naive) as s2:
        assert s1.descriptions == ("SolarAzimuth", "SolarZenith")
        a1 = s1.read()
        a2 = s2.read()

    np.testing.assert_allclose(a1, a2, rtol=1e-6)
    assert 165.0 < float(a1[0, 0, 0]) < 195.0
    assert 24.0 < float(a1[1, 0, 0]) < 33.0


# ---------------------------------------------------------------------------
# Sensor header parsers
# ---------------------------------------------------------------------------


def test_sensors_mtl_and_sen2_mtd(tmp_path):
    mtl_path = os.path.join(tmp_path, "LC08_MTL.txt")
    with open(mtl_path, "w", encoding="utf-8") as fobj:
        fobj.write(
            _build_synthetic_landsat_mtl(
                "LC08_L2SP_204024_20220621_02_T1",
                "LEVEL2_SURFACE_REFLECTANCE_PARAMETERS",
                [(1, "B1.TIF", 2.75e-5, -0.2)],
            )
        )
    parsed_mtl = sensors.read_landsat_mtl_to_dict(mtl_path)
    assert (
        parsed_mtl["PRODUCT_CONTENTS"]["LANDSAT_PRODUCT_ID"]
        == "LC08_L2SP_204024_20220621_02_T1"
    )
    assert parsed_mtl["IMAGE_ATTRIBUTES"]["SUN_ELEVATION"] == "45.0000"

    for ns in (
        "https://psd-14.sentinel2.eo.esa.int/PSD/User_Product_Level-2A.xsd",
        "https://psd-15.sentinel2.eo.esa.int/PSD/User_Product_Level-2A.xsd",
    ):
        xml_path = os.path.join(tmp_path, "MTD_MSIL2A.xml")
        with open(xml_path, "w", encoding="utf-8") as fobj:
            fobj.write(
                _build_sen2_mtd_xml(
                    "S2A_MSIL2A_TEST.SAFE",
                    ["GRANULE/GR1/IMG_DATA/R10m/T_B02_10m"],
                    namespace=ns,
                    boa_offsets={1: -1000, 2: -1000},
                )
            )
        mtd = sensors.read_sen2_l2a_mtd_to_dict(xml_path)
        assert mtd["Product_Info"]["PROCESSING_BASELINE"] == "04.00"
        assert (
            mtd["Product_Image_Characteristics"]["Quantification_values"][
                "BOA_QUANTIFICATION_VALUE"
            ]
            == 10000
        )
        assert mtd["Product_Image_Characteristics"]["BOA_Add_Offsets"][1] == -1000.0


# ---------------------------------------------------------------------------
# Landsat Collection 2 QA & stacked L1 / L2 workflows
# ---------------------------------------------------------------------------


def test_parse_landsat_c2_qa_pixel_img_bit_order(tmp_path):
    qa_in = os.path.join(tmp_path, "qa.tif")
    qa_out = os.path.join(tmp_path, "qa_parsed.tif")

    # Pixel (0, 0): Fill=1 (bit 0) -> value 1
    # Pixel (0, 1): Clear=1 (bit 6), CloudConf=Low (bit 8=1, bit 9=0 -> conf 2),
    #               CloudShadowConf=Medium (bit 10=0, bit 11=1 -> conf 3),
    #               SnowConf=High (bits 12,13=1 -> conf 4),
    #               CirrusConf=None (bits 14,15=0 -> conf 1)
    px_01 = (1 << 6) | (1 << 8) | (1 << 11) | (3 << 12)
    # Pixel (1, 0): DilatedCloud(bit 1) + Cloud(bit 3) + Water(bit 7)
    px_10 = (1 << 1) | (1 << 3) | (1 << 7)
    # Pixel (1, 1): CloudShadow(bit 4) + Snow(bit 5) + Cirrus(bit 2)
    px_11 = (1 << 2) | (1 << 4) | (1 << 5)

    qa_arr = np.array([[1, px_01], [px_10, px_11]], dtype=np.uint16)
    _write_gtiff(qa_in, qa_arr)

    l2.parse_landsat_c2_qa_pixel_img(
        qa_in, qa_out, calc_stats=False, use_pbar=False
    )

    with rasterio.open(qa_out) as src:
        assert src.count == 12
        assert src.dtypes[0] == "uint8"
        out = src.read()

    # Single-bit bands (0..7)
    assert out[0, 0, 0] == 1  # Fill
    assert out[6, 0, 1] == 1  # Clear
    assert out[1, 1, 0] == 1  # DilatedCloud
    assert out[3, 1, 0] == 1  # Cloud
    assert out[7, 1, 0] == 1  # Water
    assert out[2, 1, 1] == 1  # Cirrus
    assert out[4, 1, 1] == 1  # CloudShadow
    assert out[5, 1, 1] == 1  # Snow

    # Confidence bands (8..11): Low=2 must NOT swap with Medium=3!
    assert out[8, 0, 1] == 2  # CloudConfidence = Low
    assert out[9, 0, 1] == 3  # CloudShadowConfidence = Medium
    assert out[10, 0, 1] == 4  # SnowIceConfidence = High
    assert out[11, 0, 1] == 1  # CirrusConfidence = None


@pytest.mark.parametrize(
    "is_l1, is_tm",
    [
        (True, False),
        (False, False),
        (False, True),
    ],
)
def test_landsat_c2_stacked_workflows(tmp_path, is_l1, is_tm):
    src_dir = os.path.join(tmp_path, "src")
    out_dir = os.path.join(tmp_path, "out")
    tmp_work = os.path.join(tmp_path, "work")
    os.makedirs(src_dir)

    band_nums = (
        ("1", "2", "3", "4", "5", "7")
        if is_tm
        else ("1", "2", "3", "4", "5", "6", "7")
    )
    if is_l1:
        prod_id = "LC08_L1TP_204024_20220621_02_T1"
    elif is_tm:
        prod_id = "LT05_L2SP_204024_20060715_02_T1"
    else:
        prod_id = "LC08_L2SP_204024_20220621_02_T1"
    rescaling_grp = (
        "LEVEL1_RADIOMETRIC_RESCALING"
        if is_l1
        else "LEVEL2_SURFACE_REFLECTANCE_PARAMETERS"
    )
    sun_elev = 30.0  # sin(30) == 0.5

    band_specs = []
    dn_grid = np.array([[0, 10000], [12000, 15000]], dtype=np.uint16)
    for bnum in band_nums:
        fname = f"{prod_id}_SR_B{bnum}.TIF"
        _write_gtiff(os.path.join(src_dir, fname), dn_grid, res=30.0)
        band_specs.append((bnum, fname, 2.0e-5, -0.1))

    qa_fname = f"{prod_id}_QA_PIXEL.TIF"
    # (0,0)=Fill(1), (0,1)=Cloud(8), (1,0)=CloudShadow(16), (1,1)=Clear(64)
    qa_grid = np.array([[1, 8], [16, 64]], dtype=np.uint16)
    _write_gtiff(os.path.join(src_dir, qa_fname), qa_grid, res=30.0)

    mtl_path = os.path.join(src_dir, f"{prod_id}_MTL.txt")
    qa_key = (
        "FILE_NAME_QUALITY_L2_PIXEL"
        if (not is_l1 and not is_tm)
        else "FILE_NAME_QUALITY_L1_PIXEL"
    )
    with open(mtl_path, "w", encoding="utf-8") as fobj:
        fobj.write(
            _build_synthetic_landsat_mtl(
                prod_id,
                rescaling_grp,
                band_specs,
                qa_key=qa_key,
                qa_filename=qa_fname,
                sun_elev=sun_elev,
            )
        )

    tar_path = os.path.join(tmp_path, f"{prod_id}.tar")
    with tarfile.open(tar_path, "w") as tar_obj:
        for fname in os.listdir(src_dir):
            tar_obj.add(os.path.join(src_dir, fname), arcname=fname)

    if is_l1:
        l1.create_stacked_toa_ls_oli_cl2_lv1_img(
            tar_path,
            out_dir,
            tmp_work,
            scale_factor=10000,
            gdalformat="GTIFF",
            inc_full_qa_pxl_img=True,
            calc_stats=True,
            use_pbar=False,
        )
        out_suffix = "toa"
        exp_names = ("coastal", "blue", "green", "red", "nir", "swir1", "swir2")
    elif is_tm:
        l2.create_stacked_sref_ls_4_5_7_lv2_img(
            tar_path,
            out_dir,
            tmp_work,
            scale_factor=10000,
            gdalformat="GTIFF",
            inc_full_qa_pxl_img=False,
            calc_stats=False,
            use_pbar=False,
        )
        out_suffix = "sref"
        exp_names = ("blue", "green", "red", "nir", "swir1", "swir2")
    else:
        l2.create_stacked_sref_ls_oli_cl2_lv2_img(
            tar_path,
            out_dir,
            tmp_work,
            scale_factor=10000,
            gdalformat="GTIFF",
            inc_full_qa_pxl_img=False,
            calc_stats=False,
            use_pbar=False,
        )
        out_suffix = "sref"
        exp_names = ("coastal", "blue", "green", "red", "nir", "swir1", "swir2")

    prod_lower = prod_id.lower()
    stacked_path = os.path.join(out_dir, f"{prod_lower}_{out_suffix}.tif")
    clouds_path = os.path.join(out_dir, f"{prod_lower}_clouds.tif")
    vld_path = os.path.join(out_dir, f"{prod_lower}_vld_msk.tif")
    qa_out_path = os.path.join(out_dir, f"{prod_lower}_pxl_qa.tif")

    assert os.path.isfile(stacked_path)
    assert os.path.isfile(clouds_path)
    assert os.path.isfile(vld_path)
    assert os.path.isfile(qa_out_path) is is_l1
    assert os.listdir(tmp_work) == []  # Temp extraction directory cleaned up

    with rasterio.open(stacked_path) as src:
        assert src.count == len(exp_names)
        assert src.descriptions == exp_names
        assert src.nodata == -9999
        arr = src.read()

    # DN=0 -> -9999; DN=10000 -> (10000*2e-5 - 0.1)*10000 = 1000 (/0.5 = 2000 for L1)
    assert arr[0, 0, 0] == -9999
    expected_01 = 2000 if is_l1 else 1000
    assert arr[0, 0, 1] == expected_01

    with rasterio.open(clouds_path) as csrc, rasterio.open(vld_path) as vsrc:
        carr = csrc.read(1)
        varr = vsrc.read(1)
    assert carr[0, 1] == 1  # Cloud
    assert carr[1, 0] == 2  # Cloud shadow
    assert carr[1, 1] == 0  # Clear
    assert varr[0, 0] == 0  # Fill -> invalid
    assert varr[1, 1] == 1  # Non-fill -> valid


# ---------------------------------------------------------------------------
# Sentinel-2 L2A SAFE workflow
# ---------------------------------------------------------------------------


@pytest.fixture(
    params=[(True, True), (True, False), (False, False)],
    ids=["True-True", "True-False", "False-False"],
)
def sen2_l2a_outputs(request, tmp_path):
    """Run create_stacked_sen2_esa_l2a_img on a synthetic L2A SAFE for each
    (out_res_10m, sharpen_10m) combination and return the output paths."""
    out_res_10m, sharpen_10m = request.param
    safe_dir = os.path.join(tmp_path, "S2A_MSIL2A_TEST.SAFE")
    r10_dir = os.path.join(safe_dir, "GRANULE", "GR1", "IMG_DATA", "R10m")
    r20_dir = os.path.join(safe_dir, "GRANULE", "GR1", "IMG_DATA", "R20m")
    r60_dir = os.path.join(safe_dir, "GRANULE", "GR1", "IMG_DATA", "R60m")
    out_dir = os.path.join(tmp_path, "out")
    tmp_work = os.path.join(tmp_path, "work")
    os.makedirs(r10_dir)
    os.makedirs(r20_dir)
    os.makedirs(r60_dir)

    # SCL appears BEFORE B02_20m..B12_20m to verify the early-break regression is fixed
    image_entries = [
        "GRANULE/GR1/IMG_DATA/R10m/T_B02_10m",
        "GRANULE/GR1/IMG_DATA/R10m/T_B03_10m",
        "GRANULE/GR1/IMG_DATA/R10m/T_B04_10m",
        "GRANULE/GR1/IMG_DATA/R10m/T_B08_10m",
        "GRANULE/GR1/IMG_DATA/R20m/T_SCL_20m",
        "GRANULE/GR1/IMG_DATA/R20m/T_B02_20m",
        "GRANULE/GR1/IMG_DATA/R20m/T_B03_20m",
        "GRANULE/GR1/IMG_DATA/R20m/T_B04_20m",
        "GRANULE/GR1/IMG_DATA/R20m/T_B05_20m",
        "GRANULE/GR1/IMG_DATA/R20m/T_B06_20m",
        "GRANULE/GR1/IMG_DATA/R20m/T_B07_20m",
        "GRANULE/GR1/IMG_DATA/R20m/T_B8A_20m",
        "GRANULE/GR1/IMG_DATA/R20m/T_B11_20m",
        "GRANULE/GR1/IMG_DATA/R20m/T_B12_20m",
        # B01 at both 60 m and 20 m (20 m should be preferred); B09 at 60 m only
        "GRANULE/GR1/IMG_DATA/R60m/T_B01_60m",
        "GRANULE/GR1/IMG_DATA/R20m/T_B01_20m",
        "GRANULE/GR1/IMG_DATA/R60m/T_B09_60m",
    ]

    # 10m grid (16x16), 20m grid (8x8)
    grid_10 = np.full((16, 16), 3000, dtype=np.uint16)
    # Add spatial texture so 10m sharpening regression has variance
    yy, xx = np.mgrid[0:16, 0:16]
    grid_10 = (2500 + (yy * 100) + (xx * 50)).astype(np.uint16)
    grid_10[0:2, 0:2] = 0  # Nodata region aligned to 20m top-left pixel

    grid_20 = np.full((8, 8), 3500, dtype=np.uint16)
    yy20, xx20 = np.mgrid[0:8, 0:8]
    grid_20 = (2800 + (yy20 * 150) + (xx20 * 75)).astype(np.uint16)
    grid_20[0, 0] = 0  # Nodata

    for bcode in ("B02", "B03", "B04", "B08"):
        _write_gtiff(
            os.path.join(r10_dir, f"T_{bcode}_10m.tif"),
            grid_10,
            res=10.0,
            nodata=0,
        )
    for bcode in ("B02", "B03", "B04", "B05", "B06", "B07", "B8A", "B11", "B12"):
        _write_gtiff(
            os.path.join(r20_dir, f"T_{bcode}_20m.tif"),
            grid_20,
            res=20.0,
            nodata=0,
        )

    # B01 20 m uses a constant so the selection (20 m over 60 m) is testable.
    b01_grid_20 = np.full((8, 8), 1500, dtype=np.uint16)
    b01_grid_20[0, 0] = 0
    _write_gtiff(
        os.path.join(r20_dir, "T_B01_20m.tif"), b01_grid_20, res=20.0, nodata=0
    )
    _write_gtiff(
        os.path.join(r60_dir, "T_B01_60m.tif"),
        np.full((3, 3), 9000, dtype=np.uint16),
        res=60.0,
        nodata=0,
    )
    _write_gtiff(
        os.path.join(r60_dir, "T_B09_60m.tif"),
        np.full((3, 3), 2000, dtype=np.uint16),
        res=60.0,
        nodata=0,
    )

    scl_grid = np.full((8, 8), 4, dtype=np.uint8)  # 4 = vegetation (valid)
    scl_grid[0, 0] = 0  # NO_DATA
    scl_grid[0, 1] = 8  # Cloud medium prob -> cloud mask 1
    scl_grid[1, 0] = 3  # Cloud shadow -> cloud mask 2
    _write_gtiff(
        os.path.join(r20_dir, "T_SCL_20m.tif"),
        scl_grid,
        res=20.0,
        nodata=0,
    )

    # Include Baseline 04+ BOA_ADD_OFFSET = -1000 on all bands
    boa_offsets = {bid: -1000 for bid in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12)}
    with open(os.path.join(safe_dir, "MTD_MSIL2A.xml"), "w", encoding="utf-8") as fobj:
        fobj.write(
            _build_sen2_mtd_xml(
                "S2A_MSIL2A_TEST.SAFE",
                image_entries,
                quant_boa=10000,
                boa_offsets=boa_offsets,
            )
        )

    l2.create_stacked_sen2_esa_l2a_img(
        safe_dir,
        out_dir,
        tmp_work,
        out_res_10m=out_res_10m,
        sharpen_10m=sharpen_10m,
        scale_factor=10000,
        gdalformat="GTIFF",
        resample_10m_method=rsgispy.INTERP_BILINEAR,
        inc_cloudsen12_img=True,
        calc_stats=False,
        use_pbar=False,
    )

    res_tag = "10m" if out_res_10m else "20m"
    sref_path = os.path.join(out_dir, f"S2A_MSIL2A_TEST_{res_tag}_sref.tif")
    scl_path = os.path.join(out_dir, f"S2A_MSIL2A_TEST_{res_tag}_scl.tif")
    clouds_path = os.path.join(out_dir, f"S2A_MSIL2A_TEST_{res_tag}_clouds.tif")
    vld_path = os.path.join(out_dir, f"S2A_MSIL2A_TEST_{res_tag}_vld.tif")

    return {
        "out_dir": out_dir,
        "tmp_work": tmp_work,
        "res_tag": res_tag,
        "out_res_10m": out_res_10m,
        "grid_10": grid_10,
        "grid_20": grid_20,
        "sref_path": sref_path,
        "scl_path": scl_path,
        "clouds_path": clouds_path,
        "vld_path": vld_path,
    }


def _read_sen2_sref(sen2_l2a_outputs):
    with rasterio.open(sen2_l2a_outputs["sref_path"]) as ssrc:
        return ssrc.read()


def test_create_stacked_sen2_esa_l2a_img_sref_stack(sen2_l2a_outputs):
    """The 10-band SREF stack has the band names, grid and nodata, with the
    BOA_ADD_OFFSET applied, and the temporary directory is cleaned up."""
    tmp_work = sen2_l2a_outputs["tmp_work"]
    sref_path = sen2_l2a_outputs["sref_path"]
    out_res_10m = sen2_l2a_outputs["out_res_10m"]
    grid_10 = sen2_l2a_outputs["grid_10"]
    grid_20 = sen2_l2a_outputs["grid_20"]
    assert os.listdir(tmp_work) == []
    with rasterio.open(sref_path) as ssrc:
        assert ssrc.count == 10
        assert ssrc.descriptions == (
            "Blue",
            "Green",
            "Red",
            "RE_B5",
            "RE_B6",
            "RE_B7",
            "NIR_B8",
            "NIR_B8A",
            "SWIR1",
            "SWIR2",
        )
        expected_shape = (10, 16, 16) if out_res_10m else (10, 8, 8)
        sarr = ssrc.read()
        assert sarr.shape == expected_shape
        assert ssrc.nodata == 0.0

    # BOA_ADD_OFFSET (-1000) applied on valid pixels while nodata stays 0
    assert sarr[0, 0, 0] == 0
    if out_res_10m:
        assert sarr[0, 4, 4] == int(grid_10[4, 4]) - 1000
    else:
        assert sarr[0, 2, 2] == int(grid_20[2, 2]) - 1000


def test_create_stacked_sen2_esa_l2a_img_scl_cloud_and_valid_masks(sen2_l2a_outputs):
    """The SCL keeps its categorical values and the cloud / valid masks are set."""
    scl_path = sen2_l2a_outputs["scl_path"]
    clouds_path = sen2_l2a_outputs["clouds_path"]
    vld_path = sen2_l2a_outputs["vld_path"]
    with (
        rasterio.open(scl_path) as scl_src,
        rasterio.open(clouds_path) as cld_src,
        rasterio.open(vld_path) as vld_src,
    ):
        scl_arr = scl_src.read(1)
        cld_arr = cld_src.read(1)
        vld_arr = vld_src.read(1)

    # Categorical SCL must NEVER be interpolated into non-source values
    assert set(np.unique(scl_arr).tolist()).issubset({0, 3, 4, 8})
    assert cld_arr[0, 0] == 0
    assert vld_arr[0, 0] == 0
    assert vld_arr[-1, -1] == 1
    assert 1 in cld_arr and 2 in cld_arr


def test_create_stacked_sen2_esa_l2a_img_cloudsen12_stack(sen2_l2a_outputs):
    """The 12-band CloudSEN12 stack matches the SREF stack, picks B01 at 20 m and
    is accepted by the CloudSEN12 SREF cloud masking."""
    out_dir = sen2_l2a_outputs["out_dir"]
    res_tag = sen2_l2a_outputs["res_tag"]
    sarr = _read_sen2_sref(sen2_l2a_outputs)
    # CloudSEN12 12-band stack for the cloud masking functions
    cld12_path = os.path.join(
        out_dir, f"S2A_MSIL2A_TEST_{res_tag}_cloudsen12_sref.tif"
    )
    with rasterio.open(cld12_path) as csrc:
        assert csrc.count == 12
        assert csrc.descriptions == (
            "Coastal",
            "Blue",
            "Green",
            "Red",
            "RE_B5",
            "RE_B6",
            "RE_B7",
            "NIR_B8",
            "NIR_B8A",
            "WV_B9",
            "SWIR1",
            "SWIR2",
        )
        assert csrc.nodata == 0.0
        carr = csrc.read()
    assert carr.shape[1:] == sarr.shape[1:]
    # Bands shared with the 10-band stack are identical (B02..B08, B8A, B11, B12)
    np.testing.assert_array_equal(carr[1:9], sarr[0:8])
    np.testing.assert_array_equal(carr[10:12], sarr[8:10])
    # B01 from the 20 m (not 60 m) file and B09 from 60 m, offsets applied
    centre = (carr.shape[1] // 2, carr.shape[2] // 2)
    assert carr[0][centre] == 1500 - 1000
    assert carr[9][centre] == 2000 - 1000

    # The stack is accepted directly by the CloudSEN12 SREF cloud masking.
    import rsgispy.imagecalibration.cloudmasking as cldmsk

    cld_mask_path = os.path.join(out_dir, "cloudsen12_mask.tif")
    cldmsk._apply_cloudsen12_cloud_mask(
        input_img=cld12_path,
        output_img=cld_mask_path,
        product="sref",
        model_name="cloudsen12l2a",
        img_bands=None,
        weights_dir=None,
        download_weights=False,
        device="cpu",
        scale_factor=10000.0,
        add_offset=0.0,
        no_data_val=None,
        tile_size=32,
        tile_overlap=8,
        gdalformat="GTIFF",
        calc_stats=False,
        use_pbar=False,
        creation_opts=None,
        predict_fn=lambda arr: np.zeros(arr.shape[1:], dtype=np.uint8),
    )
    with rasterio.open(cld_mask_path) as msrc:
        marr = msrc.read(1)
    assert marr[0, 0] == cldmsk.CLOUDSEN12_NO_DATA
    assert marr[centre] == cldmsk.CLOUDSEN12_CLEAR
