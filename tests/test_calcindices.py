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
    Create an 8-band float32 test cube (band 2 unused to test sparse mappings).

    Band map:
      1: blue, 2: unused, 3: green, 4: red, 5: red_edge, 6: nir, 7: swir1, 8: swir2
    """
    blue = np.array([[0.10, 0.20, 0.00], [0.15, 0.25, 0.30]], dtype=np.float32)
    unused = np.full_like(blue, 999.0)
    green = np.array([[0.20, 0.10, 0.00], [0.25, 0.35, 0.10]], dtype=np.float32)
    red = np.array([[0.30, 0.20, 0.00], [0.10, 0.20, 0.40]], dtype=np.float32)
    red_edge = np.array([[0.40, 0.30, 0.00], [0.35, 0.45, 0.25]], dtype=np.float32)
    nir = np.array([[0.70, 0.50, 0.00], [0.60, 0.80, 0.20]], dtype=np.float32)
    swir1 = np.array([[0.50, 0.40, 0.00], [0.20, 0.30, 0.50]], dtype=np.float32)
    swir2 = np.array([[0.25, 0.20, 0.00], [0.10, 0.15, 0.25]], dtype=np.float32)
    data = np.stack([blue, unused, green, red, red_edge, nir, swir1, swir2], axis=0)
    band_names = {
        1: "Blue",
        3: "green",
        4: "RED",
        5: "re",
        6: "nir",
        7: "SWIR1",
        8: "swir2",
    }
    return data, band_names


def test_registry_helpers():
    """Test get_available_indices, get_index_req_bands, and get_index_equation."""
    avail = ic_idx.get_available_indices()
    assert len(avail) == 61
    for expected in (
        "NDVI",
        "NDWI_McFeeters",
        "NDWI_Gao",
        "MNDWI",
        "SAVI",
        "MSAVI2",
        "OSAVI",
        "AWEI_nsh",
        "AWEI_sh",
        "NDRE",
        "NDBI",
        "NBR2",
        "Brightness_Scaled",
        "NDBSI",
    ):
        assert expected in avail
    assert ic_idx.get_index_req_bands("ndvi") == ("red", "nir")
    assert ic_idx.get_index_req_bands("REMI") == (
        "green",
        "red",
        "red_edge",
        "swir1",
    )
    assert ic_idx.get_index_req_bands("NDWI_McFeeters") == ("green", "nir")
    assert ic_idx.get_index_req_bands("NDWI_Gao") == ("nir", "swir1")
    assert "(green - nir) / (green + nir)" in ic_idx.get_index_equation(
        "NDWI_McFeeters"
    )
    assert "(nir - swir1) / (nir + swir1)" in ic_idx.get_index_equation("NDWI_Gao")

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
    mean_vis_00 = (b + g + r) / 3.0
    eta_00 = (2.0 * (n ** 2 - r ** 2) + 1.5 * n + 0.5 * r) / (n + r + 0.5)
    msavi2_00 = 0.5 * (
        (2.0 * n + 1.0) - math.sqrt((2.0 * n + 1.0) ** 2 - 8.0 * (n - r))
    )
    rvi_inv_00 = r / n

    expected_p00 = {
        "NDVI": ndvi_00,
        "EVI": 2.5 * (n - r) / (n + 6.0 * r - 7.5 * b + 1.0),
        "EVI2": 2.5 * (n - r) / (n + 2.4 * r + 1.0),
        "GNDVI": (n - g) / (n + g),
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
        "NRVI": (rvi_inv_00 - 1.0) / (rvi_inv_00 + 1.0),
        "CTVI": (ndvi_00 + 0.5) / math.sqrt(abs(ndvi_00 + 0.5)),
        "TVI": math.sqrt(ndvi_00 + 0.5),
        "TTVI": math.sqrt(abs(ndvi_00 + 0.5)),
        "GEMI": eta_00 * (1.0 - 0.25 * eta_00) - ((r - 0.125) / (1.0 - r)),
        "ARVI": (n - (2.0 * r - b)) / (n + (2.0 * r - b)),
        "SLAVI": n / (r + s2),
        "kNDVI": math.tanh(ndvi_00 ** 2),
        "NIRv": ndvi_00 * n,
        "NDRE": (n - re) / (n + re),
        "CIre": (n / re) - 1.0,
        "CIgreen": (n / g) - 1.0,
        "NPCRI": (r - b) / (r + b),
        "REMI": (re - r) / (s1 + g),
        "MVI": (n - g) / (s1 - g),
        "VARI": (g - r) / (g + r - b),
        "GLI": (2.0 * g - r - b) / (2.0 * g + r + b),
        "ExG": 2.0 * g - r - b,
        "NDWI_McFeeters": (g - n) / (g + n),
        "NDWI": (g - n) / (g + n),
        "GNDWI": (g - n) / (g + n),
        "NDWI_Gao": (n - s1) / (n + s1),
        "NDMI": (n - s1) / (n + s1),
        "MNDWI": (g - s1) / (g + s1),
        "GMNDWI": (g - s1) / (g + s1),
        "NDSI": (g - s1) / (g + s1),
        "NDGI": (g - r) / (g + r),
        "AWEI_nsh": 4.0 * (g - s1) - (0.25 * n + 2.75 * s2),
        "AWEI_sh": b + 2.5 * g - 1.5 * (n + s1) - 0.25 * s2,
        "WI2015": 1.7204 + 171.0 * g + 3.0 * r - 70.0 * n - 45.0 * s1 - 71.0 * s2,
        "WBI_BlueNIR": b / n,
        "WBI": b / n,
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

    # Verify NDBSI negative-k branch at (1, 0)
    b10 = float(data[0, 1, 0])
    s10 = float(data[6, 1, 0])
    ndbsi_idx = requested.index("NDBSI")
    assert stack[ndbsi_idx, 1, 0] == pytest.approx(
        -abs(s10 - b10) / (s10 + b10), rel=1e-5
    )

    # Verify divide-by-zero pixel (0, 2) gets -999 for ratio indices
    assert stack[requested.index("NDVI"), 0, 2] == -999.0
    assert stack[requested.index("NDWI_McFeeters"), 0, 2] == -999.0
    assert stack[requested.index("NDWI_Gao"), 0, 2] == -999.0
    assert stack[requested.index("CMR"), 0, 2] == -999.0


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
        assert u_src.nodata == 0
        u_arr = u_src.read()
        assert u_arr[0, 0, 2] == 0
        assert u_arr[1, 0, 2] == 0
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
            in_no_data_val=-1.0,
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
        assert arr[1, 1, 1] == -999.0


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
        indices=["CTVI", "SI", "GEMI", "TVI", "EVI", "SAVI", "OSAVI", "WDVI", "ARVI"],
        output_img=str(out_path),
        index_params={
            "EVI": {"g": 2.0, "c1": 5.0, "c2": 7.0, "l": 1.0},
            "SAVI": {"l": 0.25},
            "OSAVI": {"y": 0.2},
            "WDVI": {"s": 1.5},
            "ARVI": {"gamma": 0.5},
        },
        calc_stats=False,
        use_pbar=False,
    )
    with rasterio.open(out_path) as src:
        arr = src.read()
        assert arr[0, 0, 0] == -999.0
        assert arr[1, 0, 1] == -999.0
        assert arr[2, 1, 1] == -999.0
        assert arr[3, 0, 0] == pytest.approx(0.0, abs=1e-6)
        assert arr[3, 1, 1] == -999.0
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
