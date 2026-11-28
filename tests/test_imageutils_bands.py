"""
Tests for :mod:`rsgispy.imageutils.bands`: stacking and selecting bands and
sharpening low resolution bands.
"""

from pathlib import Path

import numpy as np
import rasterio
import rasterio.crs
from rasterio.transform import Affine, from_origin

import rsgispy
import rsgispy.imageutils.bands as iu_bands
import rsgispy.imageutils.warp as iu_warp


def _write_test_tif(
    path,
    data,
    tl_x=500000.0,
    tl_y=5200000.0,
    res=10.0,
    crs="EPSG:32630",
    nodata=None,
):
    """Write a 2D or 3D GeoTIFF test image."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    arr = np.asarray(data)
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    count, height, width = arr.shape
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": count,
        "dtype": str(arr.dtype),
        "crs": crs,
        "transform": from_origin(tl_x, tl_y, res, res),
    }
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(str(path), "w", **profile) as dst:
        dst.write(arr)


def _write_raster(
    path: Path,
    arr: np.ndarray,
    *,
    transform: Affine = from_origin(500000.0, 5800000.0, 10.0, 10.0),
    crs: object = "EPSG:32630",
    nodata=None,
) -> str:
    data = arr[np.newaxis, :, :] if arr.ndim == 2 else arr
    profile = {
        "driver": "GTiff",
        "height": int(data.shape[1]),
        "width": int(data.shape[2]),
        "count": int(data.shape[0]),
        "dtype": str(data.dtype),
        "crs": crs,
        "transform": transform,
    }
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
    return str(path)


def test_bands_stack_select_and_translate(tmp_path):
    """Test stack_img_bands, select_img_bands, and gdal_translate."""
    b1_p = tmp_path / "b1.tif"
    b2_p = tmp_path / "b2.tif"
    a1 = np.full((8, 8), 5.0, dtype=np.float32)
    a1[0, 0] = -99.0
    _write_test_tif(b1_p, a1)
    _write_test_tif(b2_p, np.full((8, 8), 10.0, dtype=np.float32))

    stk_p = tmp_path / "stacked.tif"
    iu_bands.stack_img_bands(
        [str(b1_p), str(b2_p)],
        str(stk_p),
        ["B1", "B2"],
        skip_value=-99.0,
        no_data_val=-999.0,
        gdalformat="GTIFF",
        datatype=rsgispy.TYPE_32FLOAT,
    )
    with rasterio.open(stk_p) as src:
        assert src.count == 2
        assert src.descriptions == ("B1", "B2")
        sdata = src.read()
        assert np.allclose(sdata[:, 0, 0], [-999.0, -999.0])
        assert np.allclose(sdata[:, 1, 1], [5.0, 10.0])

    sel_p = tmp_path / "selected.tif"
    iu_bands.select_img_bands(
        str(stk_p),
        str(sel_p),
        [2],
        gdalformat="GTIFF",
        datatype=rsgispy.TYPE_32INT,
    )
    with rasterio.open(sel_p) as src:
        assert src.count == 1
        assert src.dtypes == ("int32",)
        assert src.read(1)[1, 1] == 10

    trans_p = tmp_path / "translated.tif"
    iu_warp.gdal_translate(
        str(stk_p), str(trans_p), gdalformat="GTIFF", datatype=rsgispy.TYPE_64FLOAT
    )
    with rasterio.open(trans_p) as src:
        assert src.dtypes == ("float64", "float64")
        assert src.descriptions == ("B1", "B2")


def test_sharpen_low_res_bands(tmp_path):
    """Test sharpen_low_res_bands with SharpBandInfo."""
    rng = np.random.default_rng(11)
    pan = rng.uniform(50.0, 150.0, (16, 16)).astype(np.float32)
    low = (pan * 0.4 + 20.0).astype(np.float32)
    ign = np.full((16, 16), 3.0, dtype=np.float32)
    sh_in = tmp_path / "sh_in.tif"
    sh_out = tmp_path / "sh_out.tif"
    _write_test_tif(sh_in, np.stack([low, pan, ign], axis=0))
    binfo = [
        iu_bands.SharpBandInfo(band=1, status=rsgispy.SHARP_RES_LOW, name="Low"),
        iu_bands.SharpBandInfo(
            band=2, status=rsgispy.SHARP_RES_HIGH, name="High"
        ),
        iu_bands.SharpBandInfo(
            band=3, status=rsgispy.SHARP_RES_IGNORE, name="Ign"
        ),
    ]
    iu_bands.sharpen_low_res_bands(
        str(sh_in),
        str(sh_out),
        binfo,
        win_size=7,
        no_data_val=0,
        gdalformat="GTIFF",
    )
    with rasterio.open(sh_out) as src:
        out_s = src.read()
        assert src.descriptions == ("Low", "High", "Ign")
        assert np.allclose(out_s[0], low, atol=1e-3)
        assert np.allclose(out_s[1], pan)
        assert np.allclose(out_s[2], ign)


def test_stack_img_bands_common_dtype_and_inherited_metadata(tmp_path):
    """The output dtype holds every input; nodata and band names are inherited."""
    a_path = str(tmp_path / "a.tif")
    b_path = str(tmp_path / "b.tif")
    _write_test_tif(a_path, np.full((2, 3), 7, dtype=np.uint8), nodata=0)
    _write_test_tif(b_path, np.array([[1.5, -2.25, 1000.0], [0.0, 3.0, 4.0]],
                                     dtype=np.float32), nodata=0)
    with rasterio.open(a_path, "r+") as ds:
        ds.set_band_description(1, "alpha")
    with rasterio.open(b_path, "r+") as ds:
        ds.set_band_description(1, "beta")

    out_path = str(tmp_path / "stack.tif")
    iu_bands.stack_img_bands([a_path, b_path], out_path, calc_stats=False,
                             use_pbar=False)
    with rasterio.open(out_path) as ds:
        assert ds.dtypes == ("float32", "float32")
        assert ds.nodata == 0.0
        assert ds.descriptions == ("alpha", "beta")
        np.testing.assert_array_equal(
            ds.read(2), [[1.5, -2.25, 1000.0], [0.0, 3.0, 4.0]]
        )

    # An explicit narrower datatype rounds and clamps rather than wrapping.
    out_u8 = str(tmp_path / "stack_u8.tif")
    iu_bands.stack_img_bands([a_path, b_path], out_u8, datatype=rsgispy.TYPE_8UINT,
                             calc_stats=False, use_pbar=False)
    with rasterio.open(out_u8) as ds:
        np.testing.assert_array_equal(ds.read(2), [[2, 0, 255], [0, 3, 4]])

    # Inputs that disagree on nodata give no output nodata.
    c_path = str(tmp_path / "c.tif")
    _write_test_tif(c_path, np.full((2, 3), 5, dtype=np.uint8), nodata=255)
    out_mixed = str(tmp_path / "stack_mixed.tif")
    iu_bands.stack_img_bands([a_path, c_path], out_mixed, calc_stats=False,
                             use_pbar=False)
    with rasterio.open(out_mixed) as ds:
        assert ds.nodata is None
        assert ds.dtypes == ("uint8", "uint8")


def test_sharpen_low_res_bands_sets_nodata_and_multi_high(tmp_path):
    rng = np.random.default_rng(123)
    h1 = rng.uniform(10.0, 100.0, size=(20, 20)).astype(np.float32)
    h2 = rng.uniform(5.0, 50.0, size=(20, 20)).astype(np.float32)
    low = (1.5 * h1 + 0.5 * h2 + 3.0).astype(np.float32)
    low[0, 0] = -999.0
    h1[0, 0] = -999.0
    h2[0, 0] = -999.0

    in_img = _write_raster(
        tmp_path / "sharp_in.tif",
        np.stack([h1, h2, low], axis=0),
        nodata=-999.0,
    )
    out_img = str(tmp_path / "sharp_out.tif")
    band_info = [
        iu_bands.SharpBandInfo(
            band=1, status=rsgispy.SHARP_RES_HIGH, name="H1"
        ),
        iu_bands.SharpBandInfo(
            band=2, status=rsgispy.SHARP_RES_HIGH, name="H2"
        ),
        iu_bands.SharpBandInfo(
            band=3, status=rsgispy.SHARP_RES_LOW, name="L1"
        ),
    ]
    iu_bands.sharpen_low_res_bands(
        input_img=in_img,
        output_img=out_img,
        band_info=band_info,
        win_size=5,
        no_data_val=-999.0,
        calc_stats=False,
        use_pbar=False,
    )
    with rasterio.open(out_img) as dst:
        assert dst.nodata == -999.0
        out_low = dst.read(3)
        assert out_low[0, 0] == -999.0
        # Exact linear relation should be recovered closely on interior pixels
        assert np.allclose(out_low[5:15, 5:15], low[5:15, 5:15], atol=1e-2)
