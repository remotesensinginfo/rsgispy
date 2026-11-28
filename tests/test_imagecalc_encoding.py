"""
Tests for :mod:`rsgispy.imagecalc.encoding`: integer raster recoding and
one-hot category encoding.
"""

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine, from_origin

import rsgispy
import rsgispy.imagecalc.encoding as ic_enc


def _write_raster(
    path,
    data,
    origin=(0.0, 100.0),
    pixel_size=(10.0, 10.0),
    crs="EPSG:32630",
    dtype="float32",
    nodata=None,
):
    """Write a 3D ``(bands, height, width)`` NumPy array to a GeoTIFF."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    count, height, width = data.shape
    transform = from_origin(origin[0], origin[1], pixel_size[0], pixel_size[1])
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data.astype(dtype))


def _write_geotiff(
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


def test_recode_and_one_hot(tmp_path):
    """Test recode_int_raster and one_hot_category_encoding."""
    in_p = tmp_path / "cats.tif"
    rec_keep = tmp_path / "rec_keep.tif"
    rec_drop = tmp_path / "rec_drop.tif"
    oh_auto = tmp_path / "oh_auto.tif"
    oh_dict = tmp_path / "oh_dict.tif"

    cats = np.array([[[1, 2, 3], [2, 99, 1]]], dtype=np.int32)
    _write_raster(in_p, cats, dtype="int32", nodata=99)

    ic_enc.recode_int_raster(
        str(in_p),
        str(rec_keep),
        {1: 10, 2: 20},
        keep_vals_not_in_dict=True,
        datatype=rsgispy.TYPE_16UINT,
        use_pbar=False,
    )
    with rasterio.open(rec_keep) as src:
        np.testing.assert_array_equal(
            src.read(1), np.array([[10, 20, 3], [20, 99, 10]], dtype=np.uint16)
        )

    ic_enc.recode_int_raster(
        str(in_p),
        str(rec_drop),
        {1: 10},
        keep_vals_not_in_dict=False,
        datatype=rsgispy.TYPE_8UINT,
        use_pbar=False,
    )
    with rasterio.open(rec_drop) as src:
        np.testing.assert_array_equal(
            src.read(1), np.array([[10, 0, 0], [0, 0, 10]], dtype=np.uint8)
        )

    # Auto one-hot excluding nodata=99 -> values [1, 2, 3] => 3 bands
    ic_enc.one_hot_category_encoding(
        str(in_p), str(oh_auto), no_data_val=99, use_pbar=False
    )
    with rasterio.open(oh_auto) as src:
        assert src.count == 3
        assert src.dtypes[0] == "uint8"
        out = src.read()
        np.testing.assert_array_equal(out[0], np.array([[1, 0, 0], [0, 0, 1]]))
        np.testing.assert_array_equal(out[1], np.array([[0, 1, 0], [1, 0, 0]]))
        np.testing.assert_array_equal(out[2], np.array([[0, 0, 1], [0, 0, 0]]))

    # Explicit recode_dict mapping and invalid band 0 guard
    ic_enc.one_hot_category_encoding(
        str(in_p), str(oh_dict), recode_dict={3: 1, 1: 2}, use_pbar=False
    )
    with rasterio.open(oh_dict) as src:
        assert src.count == 2
        out = src.read()
        np.testing.assert_array_equal(out[0], np.array([[0, 0, 1], [0, 0, 0]]))
        np.testing.assert_array_equal(out[1], np.array([[1, 0, 0], [0, 0, 1]]))

    with pytest.raises(rsgispy.RSGISPyException):
        ic_enc.one_hot_category_encoding(
            str(in_p), str(tmp_path / "bad_oh.tif"), recode_dict={1: 0}, use_pbar=False
        )


def test_recode_int_raster_remaps_and_validates_nodata(tmp_path):
    arr = np.array([[-9999, 1, 2], [3, -9999, 1]], dtype=np.int32)
    in_img = _write_geotiff(tmp_path / "enc_in.tif", arr, nodata=-9999)

    # Case 1: nodata remapped via recode_dict into valid uint8 range
    out1 = str(tmp_path / "enc_out1.tif")
    ic_enc.recode_int_raster(
        input_img=in_img,
        output_img=out1,
        recode_dict={-9999: 255, 1: 10, 2: 20},
        keep_vals_not_in_dict=False,
        datatype=rsgispy.TYPE_8UINT,
        use_pbar=False,
    )
    with rasterio.open(out1) as dst:
        assert dst.nodata == 255.0
        assert dst.read(1)[0, 0] == 255

    # Case 2: nodata not in dict, keep_vals_not_in_dict=True, out of uint8 range
    out2 = str(tmp_path / "enc_out2.tif")
    ic_enc.recode_int_raster(
        input_img=in_img,
        output_img=out2,
        recode_dict={1: 10, 2: 20},
        keep_vals_not_in_dict=True,
        datatype=rsgispy.TYPE_8UINT,
        use_pbar=False,
    )
    with rasterio.open(out2) as dst:
        assert dst.nodata is None
