"""Tests for rsgispy.imagecalc.calcindices."""

# pylint: disable=too-many-locals

import math

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

import rsgispy
import rsgispy.imagecalc.calcindices as ic_idx
from rsgispy.imageprocess import reset_default_block_size, set_default_block_size
from rsgispy.imageutils.metadata import get_band_names


def _write_multiband_raster(path, data, nodata=None):
    """Helper to write a multi-band rasterio GeoTIFF for tests."""
    bands, height, width = data.shape
    profile = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": bands,
        "dtype": data.dtype.name,
        "crs": "EPSG:32630",
        "transform": from_origin(500000.0, 5600000.0, 10.0, 10.0),
    }
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)


def _sample_reflectance_cube():
    """
    Create a 12-band float32 test cube (band 2 unused to test sparse mappings).

    Band map:
      1: blue, 2: unused, 3: green, 4: red, 5: red_edge, 6: nir, 7: swir1,
      8: swir2, 9: re1, 10: re2, 11: re3, 12: nir_narrow
    """
    blue = np.array([[0.10, 0.20, 0.00], [0.15, 0.25, 0.30]], dtype=np.float32)
    unused = np.full_like(blue, 999.0)
    green = np.array([[0.20, 0.10, 0.00], [0.25, 0.35, 0.10]], dtype=np.float32)
    red = np.array([[0.30, 0.20, 0.00], [0.10, 0.20, 0.40]], dtype=np.float32)
    red_edge = np.array([[0.40, 0.30, 0.00], [0.35, 0.45, 0.25]], dtype=np.float32)
    nir = np.array([[0.70, 0.50, 0.00], [0.60, 0.80, 0.20]], dtype=np.float32)
    swir1 = np.array([[0.50, 0.40, 0.00], [0.20, 0.30, 0.50]], dtype=np.float32)
    swir2 = np.array([[0.25, 0.20, 0.00], [0.10, 0.15, 0.25]], dtype=np.float32)
    re1 = np.array([[0.35, 0.25, 0.00], [0.28, 0.38, 0.22]], dtype=np.float32)
    re2 = np.array([[0.48, 0.36, 0.00], [0.42, 0.55, 0.26]], dtype=np.float32)
    re3 = np.array([[0.62, 0.44, 0.00], [0.54, 0.72, 0.24]], dtype=np.float32)
    nir_narrow = np.array([[0.72, 0.52, 0.00], [0.62, 0.82, 0.21]], dtype=np.float32)
    data = np.stack(
        [
            blue,
            unused,
            green,
            red,
            red_edge,
            nir,
            swir1,
            swir2,
            re1,
            re2,
            re3,
            nir_narrow,
        ],
        axis=0,
    )
    band_names = {
        1: "Blue",
        3: "green",
        4: "RED",
        5: "re",
        6: "nir",
        7: "SWIR1",
        8: "swir2",
        9: "RE1",
        10: "red_edge_2",
        11: "re-3",
        12: "re4",
    }
    return data, band_names


def test_registry_helpers():
    """Test get_available_indices, get_index_req_bands, and get_index_equation."""
    avail = ic_idx.get_available_indices()
    assert len(avail) == 97
    for expected in (
        "NDVI",
        "NDWI_McFeeters",
        "NDWI_Gao",
        "NDMI",
        "NDII",
        "LSWI",
        "MNDWI",
        "SAVI",
        "MSAVI2",
        "OSAVI",
        "AWEI_nsh",
        "AWEI_sh",
        "NDRE",
        "NDRE_RE",
        "NDBI",
        "NBR2",
        "Brightness_Scaled",
        "NDBSI",
        "CMRI",
        "EMI",
        "NDMaI",
        "NDWVI",
        "MVI",
        "MVI_Abs",
        "MI",
        "AMMI",
        "IMFI",
        "MDI1",
        "MDI2",
        "WFI",
        "FDI",
        "DNVI",
        "MMRI",
        "MFI",
        "NIMI",
        "EMVI",
        "GCMI",
        "REMI",
        "IMI",
        "REMCH",
        "PVI",
        "WDRVI",
        "GCVI",
        "CI_red_edge",
        "MTCI",
        "MCARI",
        "IRECI",
        "S2REP",
        "PSRI",
        "PSRI_Green",
        "NGRDI",
        "WRI",
        "RNDWI",
    ):
        assert expected in avail
    assert ic_idx.get_index_req_bands("ndvi") == ("red", "nir")
    assert ic_idx.get_index_req_bands("REMI") == (
        "green",
        "red",
        "red_edge",
        "swir1",
    )
    assert ic_idx.get_index_req_bands("MFI") == (
        "red",
        "re1",
        "re2",
        "re3",
        "nir_narrow",
        "swir2",
    )
    assert ic_idx.get_index_req_bands("NDWI_McFeeters") == ("green", "nir")
    assert ic_idx.get_index_req_bands("NDWI_Gao") == ("nir", "swir1")
    assert "(green - nir) / (green + nir)" in ic_idx.get_index_equation(
        "NDWI_McFeeters"
    )
    assert "(nir - swir1) / (nir + swir1)" in ic_idx.get_index_equation("NDWI_Gao")
    assert "(red_edge - red) / (swir1 - green)" == ic_idx.get_index_equation("REMI")

    with pytest.raises(rsgispy.RSGISPyException):
        ic_idx.get_index_req_bands("")
    with pytest.raises(rsgispy.RSGISPyException):
        ic_idx.get_index_req_bands("UNKNOWN_IDX")
    with pytest.raises(rsgispy.RSGISPyException):
        ic_idx.get_index_equation("UNKNOWN_IDX")


def test_all_indices_stack_and_band_names(tmp_path):
    """Calculate all supported indices in one stack and verify formulas and band names."""
    data, band_names = _sample_reflectance_cube()
    # Pixel (0, 2) has all zeros -> divide-by-zero invalid for ratio indices
    in_path = tmp_path / "refl.tif"
    out_path = tmp_path / "all_indices.tif"
    _write_multiband_raster(in_path, data)

    requested = ic_idx.get_available_indices()

    ic_idx.calc_indices(
        input_img=str(in_path),
        band_names=band_names,
        indices=requested,
        output_img=str(out_path),
        gdalformat="GTIFF",
        datatype=rsgispy.TYPE_32FLOAT,
        out_no_data_val=-999.0,
        calc_stats=True,
        use_pbar=False,
    )

    assert get_band_names(str(out_path)) == requested

    b = float(data[0, 0, 0])
    g = float(data[2, 0, 0])
    r = float(data[3, 0, 0])
    re = float(data[4, 0, 0])
    n = float(data[5, 0, 0])
    s1 = float(data[6, 0, 0])
    s2 = float(data[7, 0, 0])
    re1 = float(data[8, 0, 0])
    re2 = float(data[9, 0, 0])
    re3 = float(data[10, 0, 0])
    nn = float(data[11, 0, 0])

    mean_vis_arr = (data[0] + data[2] + data[3]).astype(np.float64) / 3.0
    b_mean = float(np.mean(mean_vis_arr))
    b_std = float(np.std(mean_vis_arr))
    b_scaled_00 = float(
        np.clip((mean_vis_arr[0, 0] - (b_mean - 2.0 * b_std)) / (4.0 * b_std), 0.0, 1.0)
    )

    with rasterio.open(out_path) as src:
        assert src.count == len(requested)
        assert src.nodata == -999.0
        stack = src.read()
        assert "STATISTICS_MEAN" in src.tags(1)

    ndvi_00 = (n - r) / (n + r)
    ndwi_mcf_00 = (g - n) / (g + n)
    mndwi_00 = (g - s1) / (g + s1)
    evi_00 = 2.5 * (n - r) / (n + 6.0 * r - 7.5 * b + 1.0)
    gndvi_00 = (n - g) / (n + g)
    lswi_00 = (n - s1) / (n + s1)
    rndwi_00 = (r - s1) / (r + s1)
    mean_vis_00 = (b + g + r) / 3.0
    eta_00 = (2.0 * (n ** 2 - r ** 2) + 1.5 * n + 0.5 * r) / (n + r + 0.5)
    msavi2_00 = 0.5 * (
        (2.0 * n + 1.0) - math.sqrt((2.0 * n + 1.0) ** 2 - 8.0 * (n - r))
    )
    rvi_inv_00 = r / n
    mfi_00 = 0.25 * sum(
        val - (s2 + (r - s2) * (2190.0 - wl) / (2190.0 - 665.0))
        for val, wl in ((re1, 705.0), (re2, 740.0), (re3, 783.0), (nn, 865.0))
    )

    expected_p00 = {
        "NDVI": ndvi_00,
        "EVI": evi_00,
        "EVI2": 2.5 * (n - r) / (n + 2.4 * r + 1.0),
        "GNDVI": gndvi_00,
        "SAVI": ((n - r) / (n + r + 0.5)) * 1.5,
        "OSAVI": 1.16 * (n - r) / (n + r + 0.16),
        "MSAVI2": msavi2_00,
        "MSAVI": msavi2_00,
        "SATVI": ((s1 - r) / (s1 + r + 0.5)) * 1.5 - (s2 / 2.0),
        "SR": n / r,
        "RVI": n / r,
        "RVI_NIR_Red": n / r,
        "RVI_Red_NIR": r / n,
        "DVI": n - r,
        "WDVI": n - r,
        "PVI": (n - r) / math.sqrt(2.0),
        "NRVI": (rvi_inv_00 - 1.0) / (rvi_inv_00 + 1.0),
        "CTVI": (ndvi_00 + 0.5) / math.sqrt(abs(ndvi_00 + 0.5)),
        "TVI": math.sqrt(ndvi_00 + 0.5),
        "TTVI": math.sqrt(abs(ndvi_00 + 0.5)),
        "GEMI": eta_00 * (1.0 - 0.25 * eta_00) - ((r - 0.125) / (1.0 - r)),
        "ARVI": (n - (2.0 * r - b)) / (n + (2.0 * r - b)),
        "SLAVI": n / (r + s2),
        "kNDVI": math.tanh(ndvi_00 ** 2),
        "NIRv": ndvi_00 * n,
        "WDRVI": (0.2 * n - r) / (0.2 * n + r),
        "NDRE": (n - re) / (n + re),
        "NDRE_RE": (re2 - re1) / (re2 + re1),
        "CIre": (n / re) - 1.0,
        "CI_red_edge": (n / re) - 1.0,
        "CIgreen": (n / g) - 1.0,
        "GCVI": (n / g) - 1.0,
        "NPCRI": (r - b) / (r + b),
        "MTCI": (re2 - re1) / (re1 - r),
        "MCARI": ((re1 - r) - 0.2 * (re1 - g)) * (re1 / r),
        "IRECI": (re3 - r) / (re1 / re2),
        "S2REP": 705.0 + 35.0 * (((re3 + r) / 2.0) - re1) / (re2 - re1),
        "PSRI": (r - b) / re2,
        "PSRI_Green": (r - g) / re,
        "CMRI": ndvi_00 - ndwi_mcf_00,
        "EMI": (n - s1) / (g + n),
        "NDMaI": (s2 - g) / (s2 + g),
        "NDWVI": (s2 - g) / (s2 + g),
        "MVI": (n - g) / (s1 - g),
        "MVI_Abs": abs(n - g) / abs(s1 - g),
        "MI": ((n - s1) / (n * s1)) * 10000.0,
        "AMMI": ((n - r) / (r + s1)) * ((n - s1) / (s1 - 0.65 * r)),
        "IMFI": (b + g - 2.0 * n) / (b + g + 2.0 * n),
        "MDI1": (n - s1) / s1,
        "MDI2": (n - s2) / s2,
        "WFI": (n - r) / s2,
        "FDI": n - (r + g),
        "DNVI": ((s1 - s2) ** 2) / (s1 + s2),
        "MMRI": (abs(mndwi_00) - abs(ndvi_00)) / (abs(mndwi_00) + abs(ndvi_00)),
        "MFI": mfi_00,
        "NIMI": (3.0 * r - (re2 + re3 + n)) / (3.0 * r + re2 + re3 + n),
        "EMVI": (g - s2) / (s1 - g),
        "GCMI": (3.4 * evi_00 + 2.9 * gndvi_00) / (4.3 * lswi_00 + 1.1 * rndwi_00),
        "REMI": (re - r) / (s1 - g),
        "IMI": (n - r) / (s1 + g),
        "REMCH": (n / s1) - ((n / s2) / (n / re1)) + (n / re1),
        "VARI": (g - r) / (g + r - b),
        "GLI": (2.0 * g - r - b) / (2.0 * g + r + b),
        "ExG": 2.0 * g - r - b,
        "NGRDI": (g - r) / (g + r),
        "NDWI_McFeeters": ndwi_mcf_00,
        "NDWI": ndwi_mcf_00,
        "GNDWI": ndwi_mcf_00,
        "NDWI_Gao": lswi_00,
        "NDMI": lswi_00,
        "NDII": lswi_00,
        "LSWI": lswi_00,
        "MNDWI": mndwi_00,
        "GMNDWI": mndwi_00,
        "NDSI": mndwi_00,
        "NDGI": (g - r) / (g + r),
        "AWEI_nsh": 4.0 * (g - s1) - (0.25 * n + 2.75 * s2),
        "AWEI_sh": b + 2.5 * g - 1.5 * (n + s1) - 0.25 * s2,
        "WI2015": 1.7204 + 171.0 * g + 3.0 * r - 70.0 * n - 45.0 * s1 - 71.0 * s2,
        "WBI_BlueNIR": b / n,
        "WBI": b / n,
        "WRI": (g + r) / (n + s1),
        "RNDWI": rndwi_00,
        "NBR": (n - s2) / (n + s2),
        "NBRI": (n - s2) / (n + s2),
        "NBR2": (s1 - s2) / (s1 + s2),
        "BAI": 1.0 / ((0.1 - r) ** 2 + (0.06 - n) ** 2),
        "BSI": ((s1 + r) - (n + b)) / ((s1 + r) + (n + b)),
        "NDBSI": (s1 - b) / (s1 + b),
        "NDBI": (s1 - n) / (s1 + n),
        "Brightness": mean_vis_00,
        "Brightness_Scaled": b_scaled_00,
        "Whiteness": (
            abs(b - mean_vis_00) + abs(g - mean_vis_00) + abs(r - mean_vis_00)
        )
        / mean_vis_00,
        "SI": ((1.0 - b) * (1.0 - g) * (1.0 - r)) ** (1.0 / 3.0),
        "CMR": s1 / s2,
        "FMR": s1 / n,
        "IOR": r / b,
    }
    assert set(expected_p00.keys()) == set(requested)

    for band_pos, idx_name in enumerate(requested):
        assert stack[band_pos, 0, 0] == pytest.approx(
            expected_p00[idx_name], rel=1e-5
        ), f"Mismatch for {idx_name}"

    # Verify MVI vs MVI_Abs sign difference at (1, 0) where swir1 < green
    g10 = float(data[2, 1, 0])
    n10 = float(data[5, 1, 0])
    s1_10 = float(data[6, 1, 0])
    assert stack[requested.index("MVI"), 1, 0] == pytest.approx(
        (n10 - g10) / (s1_10 - g10), rel=1e-5
    )
    assert stack[requested.index("MVI_Abs"), 1, 0] == pytest.approx(
        abs(n10 - g10) / abs(s1_10 - g10), rel=1e-5
    )
    assert stack[requested.index("MVI"), 1, 0] < 0.0
    assert stack[requested.index("MVI_Abs"), 1, 0] > 0.0

    # Verify NDBSI negative-k branch at (1, 0)
    b10 = float(data[0, 1, 0])
    ndbsi_idx = requested.index("NDBSI")
    assert stack[ndbsi_idx, 1, 0] == pytest.approx(
        -abs(s1_10 - b10) / (s1_10 + b10), rel=1e-5
    )

    # Verify divide-by-zero pixel (0, 2) gets -999 for ratio indices
    for zero_idx in (
        "NDVI",
        "NDWI_McFeeters",
        "NDWI_Gao",
        "CMR",
        "CMRI",
        "MMRI",
        "AMMI",
        "GCMI",
        "REMCH",
        "S2REP",
        "IRECI",
        "MTCI",
        "MCARI",
        "PSRI",
        "WRI",
    ):
        assert stack[requested.index(zero_idx), 0, 2] == -999.0


def test_input_output_gain_offset_and_integerisation(tmp_path):
    """Verify input unscaling and output gain/offset integerisation."""
    data, band_names = _sample_reflectance_cube()
    in_gain = 10000.0
    in_offset = 100.0
    dn_data = np.rint(data * in_gain + in_offset).astype(np.int32)
    dn_data[:, 0, 2] = -9999

    in_path = tmp_path / "dn_input.tif"
    float_ref_path = tmp_path / "ref_float.tif"
    int16_out_path = tmp_path / "indices_int16.tif"
    uint16_out_path = tmp_path / "indices_uint16.tif"

    _write_multiband_raster(in_path, dn_data, nodata=-9999)
    float_data = data.copy()
    float_data[:, 0, 2] = -9999.0
    _write_multiband_raster(float_ref_path, float_data, nodata=-9999.0)

    indices = ["NDVI", "EVI", "BAI", "Brightness"]

    ic_idx.calc_indices(
        input_img=str(float_ref_path),
        band_names=band_names,
        indices=indices,
        output_img=str(tmp_path / "expected_float.tif"),
        datatype=rsgispy.TYPE_32FLOAT,
        out_no_data_val=-999.0,
        calc_stats=False,
        use_pbar=False,
    )

    ic_idx.calc_indices(
        input_img=str(in_path),
        band_names=band_names,
        indices=indices,
        output_img=str(int16_out_path),
        datatype=rsgispy.TYPE_16INT,
        in_gain=in_gain,
        in_offset=in_offset,
        out_gain=1000.0,
        out_offset=0.0,
        out_no_data_val=-32768,
        calc_stats=False,
        use_pbar=False,
    )

    with (
        rasterio.open(tmp_path / "expected_float.tif") as ref_src,
        rasterio.open(int16_out_path) as int_src,
    ):
        assert int_src.dtypes[0] == "int16"
        assert int_src.nodata == -32768
        assert get_band_names(str(int16_out_path)) == indices
        ref_arr = ref_src.read()
        int_arr = int_src.read()
        assert np.all(int_arr[:, 0, 2] == -32768)
        valid = ref_arr != -999.0
        np.testing.assert_array_equal(
            int_arr[valid], np.rint(ref_arr[valid] * 1000.0).astype(np.int16)
        )

    ic_idx.calc_indices(
        input_img=str(in_path),
        band_names=band_names,
        indices=["NDVI", "NDWI_Gao"],
        output_img=str(uint16_out_path),
        datatype=rsgispy.TYPE_16UINT,
        in_gain=in_gain,
        in_offset=in_offset,
        out_gain={"NDVI": 1000.0, "NDWI_Gao": 2000.0},
        out_offset=[1000.0, 2000.0],
        calc_stats=False,
        use_pbar=False,
    )
    with rasterio.open(uint16_out_path) as u_src:
        assert u_src.dtypes[0] == "uint16"
        assert u_src.nodata == 65535  # default: the uint16 maximum
        u_arr = u_src.read()
        assert u_arr[0, 0, 2] == 65535
        assert u_arr[1, 0, 2] == 65535
        assert u_arr[0, 0, 0] == 1400
        assert u_arr[1, 0, 0] == 2333


def test_per_band_input_scaling_and_multiblock(tmp_path):
    """Verify per-band input gain/offset mapping, sparse nodata, and block iteration."""
    data, band_names = _sample_reflectance_cube()
    scaled = data.copy()
    scaled[0] = data[0] * 1000.0 + 10.0
    scaled[3] = data[3] * 2000.0 - 20.0
    scaled[5] = data[5] * 5000.0
    scaled[0, 1, 1] = -1.0

    in_path = tmp_path / "per_band.tif"
    out_path = tmp_path / "out_blocks.tif"
    _write_multiband_raster(in_path, scaled)

    set_default_block_size(1)
    try:
        ic_idx.calc_indices(
            input_img=str(in_path),
            band_names=band_names,
            indices=["NDVI", "WBI_BlueNIR"],
            output_img=str(out_path),
            no_data_val=-1.0,
            in_gain={1: 1000.0, 2: 1.0, "red": 2000.0, "nir": 5000.0},
            in_offset={1: 10.0, "red": -20.0, "nir": 0.0},
            calc_stats=False,
            use_pbar=False,
        )
    finally:
        reset_default_block_size()

    with rasterio.open(out_path) as src:
        arr = src.read()
        expected_ndvi_11 = (0.8 - 0.2) / (0.8 + 0.2)
        assert arr[0, 1, 1] == pytest.approx(expected_ndvi_11, rel=1e-5)
        assert arr[1, 1, 1] == -9999.0


def test_custom_index_params_and_edge_cases(tmp_path):
    """Test custom SAVI/WDVI/ARVI/EVI params, singularities, and flat Brightness_Scaled."""
    blue = np.array([[0.2, 1.5], [0.2, 0.2]], dtype=np.float32)
    green = np.array([[0.2, 0.2], [0.2, 0.2]], dtype=np.float32)
    red = np.array([[0.75, 0.2], [0.2, 1.0]], dtype=np.float32)
    nir = np.array([[0.25, 0.5], [0.5, 0.2]], dtype=np.float32)
    swir1 = np.array([[0.3, 0.3], [0.3, 0.3]], dtype=np.float32)
    swir2 = np.array([[0.2, 0.2], [0.2, 0.2]], dtype=np.float32)
    data = np.stack([blue, green, red, nir, swir1, swir2], axis=0)
    bmap = {1: "blue", 2: "green", 3: "red", 4: "nir", 5: "swir1", 6: "swir2"}

    in_path = tmp_path / "edge.tif"
    out_path = tmp_path / "edge_out.tif"
    _write_multiband_raster(in_path, data)

    # At (0, 0): nir=0.25, red=0.75 -> ndvi=-0.5 -> CTVI sqrt(0) -> nodata; TVI = 0
    # At (0, 1): blue=1.5 > 1, red=0.2, green=0.2 -> SI neg product -> nodata
    # At (1, 1): red=1.0, nir=0.2 -> GEMI (1-red)=0 -> nodata; TVI (NDVI+0.5 < 0) -> nodata
    # At (1, 0): nir=0.5, red=0.2, blue=0.2 -> check custom SAVI, WDVI, ARVI, EVI, OSAVI
    ic_idx.calc_indices(
        input_img=str(in_path),
        band_names=bmap,
        indices=[
            "CTVI",
            "SI",
            "GEMI",
            "TVI",
            "EVI",
            "SAVI",
            "OSAVI",
            "WDVI",
            "ARVI",
            "PVI",
            "WDRVI",
            "GCMI",
        ],
        output_img=str(out_path),
        index_params={
            "EVI": {"g": 2.0, "c1": 5.0, "c2": 7.0, "l": 1.0},
            "SAVI": {"l": 0.25},
            "OSAVI": {"y": 0.2},
            "WDVI": {"s": 1.5},
            "ARVI": {"gamma": 0.5},
            "PVI": {"a": 1.2, "b": 0.05},
            "WDRVI": {"a": 0.1},
            "GCMI": {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0},
        },
        calc_stats=False,
        use_pbar=False,
    )
    with rasterio.open(out_path) as src:
        arr = src.read()
        assert arr[0, 0, 0] == -9999.0
        assert arr[1, 0, 1] == -9999.0
        assert arr[2, 1, 1] == -9999.0
        assert arr[3, 0, 0] == pytest.approx(0.0, abs=1e-6)
        assert arr[3, 1, 1] == -9999.0
        assert arr[4, 1, 0] == pytest.approx(
            2.0 * (0.5 - 0.2) / (0.5 + 5.0 * 0.2 - 7.0 * 0.2 + 1.0), rel=1e-5
        )
        assert arr[5, 1, 0] == pytest.approx(
            ((0.5 - 0.2) / (0.5 + 0.2 + 0.25)) * 1.25, rel=1e-5
        )
        assert arr[6, 1, 0] == pytest.approx(
            1.2 * (0.5 - 0.2) / (0.5 + 0.2 + 0.2), rel=1e-5
        )
        assert arr[7, 1, 0] == pytest.approx(0.5 - 1.5 * 0.2, rel=1e-5)
        rb = 0.2 - 0.5 * (0.2 - 0.2)
        assert arr[8, 1, 0] == pytest.approx((0.5 - rb) / (0.5 + rb), rel=1e-5)
        assert arr[9, 1, 0] == pytest.approx(
            (0.5 - 1.2 * 0.2 - 0.05) / math.sqrt(1.0 + 1.2 ** 2), rel=1e-5
        )
        assert arr[10, 1, 0] == pytest.approx(
            (0.1 * 0.5 - 0.2) / (0.1 * 0.5 + 0.2), rel=1e-5
        )
        evi_10 = 2.5 * (0.5 - 0.2) / (0.5 + 6.0 * 0.2 - 7.5 * 0.2 + 1.0)
        gndvi_10 = (0.5 - 0.2) / (0.5 + 0.2)
        lswi_10 = (0.5 - 0.3) / (0.5 + 0.3)
        rndwi_10 = (0.2 - 0.3) / (0.2 + 0.3)
        assert arr[11, 1, 0] == pytest.approx(
            (1.0 * evi_10 + 2.0 * gndvi_10) / (3.0 * lswi_10 + 4.0 * rndwi_10),
            rel=1e-5,
        )

    # MFI with invalid equal baseline wavelengths raises RSGISPyException
    full_data, full_bmap = _sample_reflectance_cube()
    full_in = tmp_path / "full_mfi.tif"
    _write_multiband_raster(full_in, full_data)
    with pytest.raises(rsgispy.RSGISPyException, match="lambda_swir2"):
        ic_idx.calc_indices(
            input_img=str(full_in),
            band_names=full_bmap,
            indices=["MFI"],
            output_img=str(tmp_path / "mfi_bad.tif"),
            index_params={"MFI": {"lambda_red": 665.0, "lambda_swir2": 665.0}},
            calc_stats=False,
            use_pbar=False,
        )

    # Constant visible bands -> Brightness_Scaled has zero stddev -> outputs 0.0
    const_data = np.full((3, 2, 2), 0.25, dtype=np.float32)
    const_in = tmp_path / "const.tif"
    const_out = tmp_path / "const_out.tif"
    _write_multiband_raster(const_in, const_data)
    ic_idx.calc_indices(
        input_img=str(const_in),
        band_names={1: "blue", 2: "green", 3: "red"},
        indices=["Brightness_Scaled"],
        output_img=str(const_out),
        calc_stats=False,
        use_pbar=False,
    )
    with rasterio.open(const_out) as src:
        np.testing.assert_allclose(src.read(1), np.zeros((2, 2), dtype=np.float32))


def test_calc_indices_validation_errors(tmp_path):
    """Ensure clear RSGISPyException errors on invalid parameters."""
    data = np.ones((4, 2, 2), dtype=np.float32) * 0.2
    in_path = tmp_path / "valid.tif"
    _write_multiband_raster(in_path, data)
    bmap = {1: "blue", 2: "green", 3: "red", 4: "nir"}

    with pytest.raises(rsgispy.RSGISPyException):
        ic_idx.calc_indices(
            str(tmp_path / "missing.tif"),
            str(tmp_path / "o.tif"),
            bmap,
            ["NDVI"],
            use_pbar=False,
        )

    # Missing required band (swir1) for MNDWI
    with pytest.raises(rsgispy.RSGISPyException, match="swir1"):
        ic_idx.calc_indices(
            str(in_path), str(tmp_path / "o.tif"), bmap, ["MNDWI"], use_pbar=False
        )

    # Out-of-range band index
    with pytest.raises(rsgispy.RSGISPyException):
        ic_idx.calc_indices(
            str(in_path),
            str(tmp_path / "o.tif"),
            {1: "red", 9: "nir"},
            ["NDVI"],
            use_pbar=False,
        )

    # Duplicate canonical band alias
    with pytest.raises(rsgispy.RSGISPyException, match="Duplicate"):
        ic_idx.calc_indices(
            str(in_path),
            str(tmp_path / "o.tif"),
            {1: "swir", 2: "swir1", 3: "red", 4: "nir"},
            ["NDVI"],
            use_pbar=False,
        )

    # Unknown index name
    with pytest.raises(rsgispy.RSGISPyException, match="Unknown"):
        ic_idx.calc_indices(
            str(in_path), str(tmp_path / "o.tif"), bmap, ["BADIDX"], use_pbar=False
        )

    # Zero gain
    with pytest.raises(rsgispy.RSGISPyException, match="zero"):
        ic_idx.calc_indices(
            str(in_path),
            str(tmp_path / "o.tif"),
            bmap,
            ["NDVI"],
            in_gain=0.0,
            use_pbar=False,
        )
    with pytest.raises(rsgispy.RSGISPyException, match="zero"):
        ic_idx.calc_indices(
            str(in_path),
            str(tmp_path / "o.tif"),
            bmap,
            ["NDVI"],
            out_gain=0.0,
            use_pbar=False,
        )

    # Out-of-range out_no_data_val for uint16
    with pytest.raises(rsgispy.RSGISPyException, match="outside"):
        ic_idx.calc_indices(
            str(in_path),
            str(tmp_path / "o.tif"),
            bmap,
            ["NDVI"],
            datatype=rsgispy.TYPE_16UINT,
            out_no_data_val=-999,
            use_pbar=False,
        )

    # Mismatched sequence lengths
    with pytest.raises(rsgispy.RSGISPyException):
        ic_idx.calc_indices(
            str(in_path),
            str(tmp_path / "o.tif"),
            bmap,
            ["NDVI"],
            in_gain=[1.0, 1.0],
            use_pbar=False,
        )
    with pytest.raises(rsgispy.RSGISPyException):
        ic_idx.calc_indices(
            str(in_path),
            str(tmp_path / "o.tif"),
            bmap,
            ["NDVI", "WBI_BlueNIR"],
            out_gain=[1000.0],
            use_pbar=False,
        )


def test_sar_index_discovery_helpers():
    """Verify SAR index discovery, required bands, equations, and working units."""
    avail = ic_idx.get_available_sar_indices()
    assert len(avail) >= 40
    for expected in (
        "RVI",
        "RVI_VV_VH",
        "RVI_HH_HV",
        "DpRVIc_VV_VH",
        "DPSVI",
        "DPSVIm",
        "VDDPI",
        "RFDI",
        "RFDI_VV_VH",
        "NDPDI",
        "NDPDI_VV_VH",
        "NDPDI_HH_HV",
        "NDI_VV_VH",
        "NDI_HH_HV",
        "NDI_VH_VV",
        "BMI",
        "CSI",
        "VSI",
        "SDWI",
        "CR_VH_VV",
        "CR_VH_VV_dB",
        "Mean_VV_VH_dB",
        "Span_VV_VH",
        "Span_VV_VH_dB",
        "Span_Quad",
        "Span_Quad_dB",
    ):
        assert expected in avail

    assert ic_idx.get_sar_index_req_bands("rvi_vv_vh") == ("vv", "vh")
    assert ic_idx.get_sar_index_req_bands("RVI") == ("hh", "vv", "hv")
    assert "vh" in ic_idx.get_sar_index_equation("RVI_VV_VH")
    assert ic_idx.get_sar_index_units("RVI_VV_VH") == "power"
    assert ic_idx.get_sar_index_units("SDWI") == "power"
    assert ic_idx.get_sar_index_units("CR_VH_VV_dB") == "dB"
    assert ic_idx.get_sar_index_units("Mean_VV_VH_dB") == "dB"

    with pytest.raises(rsgispy.RSGISPyException, match="Unknown SAR index"):
        ic_idx.get_sar_index_req_bands("NON_EXISTENT_SAR")
    with pytest.raises(rsgispy.RSGISPyException, match="non-empty"):
        ic_idx.get_sar_index_units("   ")


def test_calc_sar_indices_power_db_equivalence_and_values(tmp_path):
    """Verify every SAR index yields identical values from power and dB inputs."""
    # Realistic linear-power backscatter values across 2x3 pixels
    hh_p = np.array([[0.25, 0.16, 0.04], [0.40, 0.10, 0.02]], dtype=np.float32)
    hv_p = np.array([[0.05, 0.04, 0.01], [0.10, 0.02, 0.005]], dtype=np.float32)
    vv_p = np.array([[0.20, 0.12, 0.05], [0.32, 0.08, 0.01]], dtype=np.float32)
    vh_p = np.array([[0.04, 0.03, 0.01], [0.08, 0.02, 0.002]], dtype=np.float32)

    pwr_cube = np.stack([hh_p, hv_p, vv_p, vh_p], axis=0)
    db_cube = (10.0 * np.log10(pwr_cube.astype(np.float64))).astype(np.float32)

    pwr_path = tmp_path / "sar_power.tif"
    db_path = tmp_path / "sar_db.tif"
    _write_multiband_raster(pwr_path, pwr_cube)
    _write_multiband_raster(db_path, db_cube)

    bmap = {1: "HH", 2: "HV", 3: "VV", 4: "VH"}
    all_indices = ic_idx.get_available_sar_indices()

    out_from_pwr = tmp_path / "out_from_power.tif"
    out_from_db = tmp_path / "out_from_db.tif"
    ic_idx.calc_sar_indices(
        str(pwr_path),
        str(out_from_pwr),
        band_names=bmap,
        indices=all_indices,
        in_units="power",
        calc_stats=False,
        use_pbar=False,
    )
    ic_idx.calc_sar_indices(
        str(db_path),
        str(out_from_db),
        band_names=bmap,
        indices=all_indices,
        in_units="dB",
        calc_stats=False,
        use_pbar=False,
    )

    assert get_band_names(str(out_from_pwr)) == all_indices
    with rasterio.open(out_from_pwr) as s_pwr, rasterio.open(out_from_db) as s_db:
        arr_pwr = s_pwr.read()
        arr_db = s_db.read()
        assert arr_pwr.shape == (len(all_indices), 2, 3)
        np.testing.assert_allclose(arr_pwr, arr_db, rtol=1e-4, atol=1e-5)

    # Analytical spot-checks at pixel (0, 0): hh=0.25, hv=0.05, vv=0.20, vh=0.04
    by_name = {name: arr_pwr[i] for i, name in enumerate(all_indices)}
    hh0, hv0, vv0, vh0 = 0.25, 0.05, 0.20, 0.04
    hh_db0 = 10.0 * math.log10(hh0)
    vv_db0 = 10.0 * math.log10(vv0)
    vh_db0 = 10.0 * math.log10(vh0)
    q_vv = vh0 / vv0
    vv_max = float(np.max(vv_p))

    assert by_name["RVI"][0, 0] == pytest.approx(
        (8.0 * hv0) / (hh0 + vv0 + 2.0 * hv0), rel=1e-5
    )
    assert by_name["RVI_VV_VH"][0, 0] == pytest.approx(
        (4.0 * vh0) / (vv0 + vh0), rel=1e-5
    )
    assert by_name["RVI_HH_HV"][0, 0] == pytest.approx(
        (4.0 * hv0) / (hh0 + hv0), rel=1e-5
    )
    assert by_name["DpRVIc_VV_VH"][0, 0] == pytest.approx(
        (q_vv * (q_vv + 3.0)) / ((q_vv + 1.0) ** 2), rel=1e-5
    )
    assert by_name["DPSVI"][0, 0] == pytest.approx(
        (((vv_max - vv0) + vh0) / math.sqrt(2.0)) * ((vv0 + vh0) / vv0) * vh0,
        rel=1e-5,
    )
    assert by_name["DPSVIm"][0, 0] == pytest.approx(
        (vv0 * vv0 + vv0 * vh0) / math.sqrt(2.0), rel=1e-5
    )
    assert by_name["VDDPI"][0, 0] == pytest.approx((vv0 + vh0) / vv0, rel=1e-5)
    assert by_name["HDDPI"][0, 0] == pytest.approx((hh0 + hv0) / hh0, rel=1e-5)
    assert by_name["RFDI"][0, 0] == pytest.approx((hh0 - hv0) / (hh0 + hv0), rel=1e-5)
    assert by_name["NDPDI_HH_HV"][0, 0] == pytest.approx(
        (hh0 - hv0) / (hh0 + hv0), rel=1e-5
    )
    assert by_name["NDI_HH_HV"][0, 0] == pytest.approx(
        (hh0 - hv0) / (hh0 + hv0), rel=1e-5
    )
    assert by_name["RFDI_VV_VH"][0, 0] == pytest.approx(
        (vv0 - vh0) / (vv0 + vh0), rel=1e-5
    )
    assert by_name["NDPDI"][0, 0] == pytest.approx(
        (vv0 - vh0) / (vv0 + vh0), rel=1e-5
    )
    assert by_name["NDPDI_VV_VH"][0, 0] == pytest.approx(
        (vv0 - vh0) / (vv0 + vh0), rel=1e-5
    )
    assert by_name["NDI_VV_VH"][0, 0] == pytest.approx(
        (vv0 - vh0) / (vv0 + vh0), rel=1e-5
    )
    assert by_name["NDI_VH_VV"][0, 0] == pytest.approx(
        (vh0 - vv0) / (vh0 + vv0), rel=1e-5
    )
    assert by_name["BMI"][0, 0] == pytest.approx(0.5 * (hh0 + vv0), rel=1e-5)
    assert by_name["CSI"][0, 0] == pytest.approx(vv0 / (vv0 + hh0), rel=1e-5)
    assert by_name["VSI"][0, 0] == pytest.approx(
        hv0 / (hv0 + 0.5 * (hh0 + vv0)), rel=1e-5
    )
    assert by_name["SDWI"][0, 0] == pytest.approx(
        math.log(10.0 * vv0 * vh0) - 8.0, rel=1e-5
    )
    assert by_name["CR_VH_VV"][0, 0] == pytest.approx(vh0 / vv0, rel=1e-5)
    assert by_name["CR_VH_VV_dB"][0, 0] == pytest.approx(vh_db0 - vv_db0, rel=1e-5)
    assert by_name["PR_HH_VV_dB"][0, 0] == pytest.approx(hh_db0 - vv_db0, rel=1e-5)
    assert by_name["Mean_VV_VH_dB"][0, 0] == pytest.approx(
        0.5 * (vv_db0 + vh_db0), rel=1e-5
    )
    assert by_name["Span_VV_VH"][0, 0] == pytest.approx(vv0 + vh0, rel=1e-5)
    assert by_name["Span_VV_VH_dB"][0, 0] == pytest.approx(
        10.0 * math.log10(vv0 + vh0), rel=1e-5
    )
    assert by_name["Span_Quad"][0, 0] == pytest.approx(
        hh0 + vv0 + 2.0 * hv0, rel=1e-5
    )
    assert by_name["Span_Quad_dB"][0, 0] == pytest.approx(
        10.0 * math.log10(hh0 + vv0 + 2.0 * hv0), rel=1e-5
    )


def test_calc_sar_indices_scaling_nodata_and_overrides(tmp_path):
    """Test scaled int16 dB input, non-positive power masking, and DPSVI vv_max override."""
    # 1. Scaled int16 dB input (stored as dB * 100) with nodata = -32768
    vv_db = np.array([[-10.0, -7.0], [-13.0, -327.68]], dtype=np.float64)
    vh_db = np.array([[-17.0, -14.0], [-20.0, -327.68]], dtype=np.float64)
    dn_cube = np.rint(np.stack([vv_db, vh_db], axis=0) * 100.0).astype(np.int16)
    dn_cube[:, 1, 1] = -32768

    in_db_scaled = tmp_path / "s1_db_int16.tif"
    _write_multiband_raster(in_db_scaled, dn_cube, nodata=-32768)

    out_int16 = tmp_path / "s1_out_int16.tif"
    ic_idx.calc_sar_indices(
        str(in_db_scaled),
        str(out_int16),
        band_names={1: "sigma0_vv", 2: "gamma0_vh"},
        indices=["RVI_VV_VH", "CR_VH_VV_dB", "DPSVI"],
        in_units="decibels",
        in_gain=100.0,
        out_gain=[1000.0, 100.0, 1000.0],
        index_params={"DPSVI": {"vv_max": 1.0}},
        datatype=rsgispy.TYPE_16INT,
        out_no_data_val=-9999,
        calc_stats=False,
        use_pbar=False,
    )
    with rasterio.open(out_int16) as src:
        assert src.dtypes == ("int16", "int16", "int16")
        assert src.nodata == -9999
        out = src.read()
        # Pixel (0, 0): vv_db=-10 -> 0.1, vh_db=-17 -> 10**-1.7
        vv0 = 10.0 ** (-1.0)
        vh0 = 10.0 ** (-1.7)
        exp_rvi = round(1000.0 * (4.0 * vh0) / (vv0 + vh0))
        exp_cr_db = round(100.0 * (-17.0 - (-10.0)))
        exp_dpsvi = round(
            1000.0
            * ((((1.0 - vv0) + vh0) / math.sqrt(2.0)) * ((vv0 + vh0) / vv0) * vh0)
        )
        assert out[0, 0, 0] == exp_rvi
        assert out[1, 0, 0] == exp_cr_db
        assert out[2, 0, 0] == exp_dpsvi
        # No-data pixel (1, 1) masked in all bands
        assert np.all(out[:, 1, 1] == -9999)

    # 2. Linear power with zero / negative invalid pixels masked for dB & SDWI
    vv_p = np.array([[0.2, 0.0], [-0.1, 0.1]], dtype=np.float32)
    vh_p = np.array([[0.05, 0.05], [0.05, 0.0]], dtype=np.float32)
    in_pwr = tmp_path / "s1_pwr_edge.tif"
    _write_multiband_raster(in_pwr, np.stack([vv_p, vh_p], axis=0))

    out_edge = tmp_path / "s1_edge_out.tif"
    ic_idx.calc_sar_indices(
        str(in_pwr),
        str(out_edge),
        band_names={1: "VV", 2: "VH"},
        indices=["RVI_VV_VH", "CR_VH_VV_dB", "SDWI", "Span_VV_VH_dB"],
        in_units="linear",
        out_no_data_val=-999.0,
        calc_stats=False,
        use_pbar=False,
    )
    with rasterio.open(out_edge) as src:
        out_e = src.read()
        # Pixel (0, 0) valid for all 4 indices
        assert np.all(out_e[:, 0, 0] != -999.0)
        # Pixel (0, 1) has vv=0 -> invalid for CR_VH_VV_dB & SDWI, valid for RVI & Span_dB
        assert out_e[0, 0, 1] == pytest.approx(4.0, rel=1e-5)
        assert out_e[1, 0, 1] == pytest.approx(-999.0)
        assert out_e[2, 0, 1] == pytest.approx(-999.0)
        assert out_e[3, 0, 1] == pytest.approx(10.0 * math.log10(0.05), rel=1e-5)
        # Pixel (1, 0) has negative power -> invalid for all 4 indices
        assert np.all(out_e[:, 1, 0] == -999.0)
        # Pixel (1, 1) has vh=0 -> invalid for CR_VH_VV_dB & SDWI
        assert out_e[1, 1, 1] == pytest.approx(-999.0)
        assert out_e[2, 1, 1] == pytest.approx(-999.0)


def test_calc_sar_indices_validation_errors(tmp_path):
    """Test error handling for invalid in_units, polarisations, and SAR index names."""
    in_path = tmp_path / "sar_small.tif"
    _write_multiband_raster(
        in_path, np.ones((2, 2, 2), dtype=np.float32) * 0.1
    )
    bmap = {1: "VV", 2: "VH"}

    with pytest.raises(rsgispy.RSGISPyException, match="Unsupported in_units"):
        ic_idx.calc_sar_indices(
            str(in_path),
            str(tmp_path / "o.tif"),
            bmap,
            ["RVI_VV_VH"],
            in_units="amplitude",
            use_pbar=False,
        )
    with pytest.raises(rsgispy.RSGISPyException, match="in_units"):
        ic_idx.calc_sar_indices(
            str(in_path),
            str(tmp_path / "o.tif"),
            bmap,
            ["RVI_VV_VH"],
            in_units=" ",
            use_pbar=False,
        )
    with pytest.raises(rsgispy.RSGISPyException, match="requires polarisation"):
        ic_idx.calc_sar_indices(
            str(in_path),
            str(tmp_path / "o.tif"),
            bmap,
            ["RFDI"],
            in_units="power",
            use_pbar=False,
        )
    with pytest.raises(rsgispy.RSGISPyException, match="Unknown SAR index"):
        ic_idx.calc_sar_indices(
            str(in_path),
            str(tmp_path / "o.tif"),
            bmap,
            ["BAD_SAR_IDX"],
            in_units="power",
            use_pbar=False,
        )
    with pytest.raises(rsgispy.RSGISPyException, match="Duplicate"):
        ic_idx.calc_sar_indices(
            str(in_path),
            str(tmp_path / "o.tif"),
            {1: "VV", 2: "sigma0_vv"},
            ["RVI_VV_VH"],
            in_units="power",
            use_pbar=False,
        )
    with pytest.raises(rsgispy.RSGISPyException, match="non-empty sequence"):
        ic_idx.calc_sar_indices(
            str(in_path),
            str(tmp_path / "o.tif"),
            bmap,
            [],
            in_units="dB",
            use_pbar=False,
        )


def test_dprvi_is_mandal_2020_index():
    """DpRVI_VV, DpRVI_HH and RVI4S1 are 1 - m * beta (Mandal et al. 2020)."""
    from rsgispy.imagecalc import _sar_indices as sar  # pylint: disable=import-outside-toplevel

    co = np.array([0.20, 0.04, 0.10, 0.0])
    cr = np.array([0.04, 0.20, 0.10, 0.05])
    tot = co + cr
    with np.errstate(invalid="ignore", divide="ignore"):
        m_dop = np.abs(co - cr) / tot
        beta = np.maximum(co, cr) / tot
    expected = 1.0 - m_dop * beta
    for key, bands in (("dprvi_vv", ("vv", "vh")), ("rvi4s1", ("vv", "vh")),
                       ("dprvi_hh", ("hh", "hv"))):
        spec = sar._SAR_INDEX_REGISTRY[key]  # pylint: disable=protected-access
        out, valid = spec.calc_fn({bands[0]: co, bands[1]: cr}, {})
        np.testing.assert_allclose(out[:3], expected[:3], rtol=1e-12)
        assert valid.tolist() == [True, True, True, False]
    # For VH <= VV it equals the q (q + 3) / (q + 1)^2 form.
    q_vv = 0.04 / 0.20
    assert expected[0] == pytest.approx(q_vv * (q_vv + 3.0) / (q_vv + 1.0) ** 2)
