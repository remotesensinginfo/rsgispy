"""
Tests for :mod:`rsgispy.imagecalc.distance`: distance to pixel values and
pixel value buffers.
"""

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

import rsgispy.imagecalc.distance as ic_dist


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


def test_distance_and_buffer_functions(tmp_path):
    """Test calc_dist_to_img_vals, calc_dist_to_img_vals_tiled, and buffer_img_pxl_vals."""
    in_p = tmp_path / "dist_in.tif"
    d_full = tmp_path / "d_full.tif"
    d_tiled = tmp_path / "d_tiled.tif"
    buf_p = tmp_path / "buf.tif"

    arr = np.zeros((1, 8, 8), dtype=np.int16)
    arr[0, 0, 0] = 1
    arr[0, 5, 5] = 2
    arr[0, 7, 7] = -1
    _write_raster(in_p, arr, origin=(0.0, 80.0), pixel_size=(10.0, 10.0), dtype="int16", nodata=-1)

    ic_dist.calc_dist_to_img_vals(
        str(in_p),
        str(d_full),
        pxl_vals=[1, 2],
        max_dist=25.0,
        no_data_val=-1,
        out_no_data_val=-999.0,
        unit_geo=True,
    )
    ic_dist.calc_dist_to_img_vals_tiled(
        str(in_p),
        str(d_tiled),
        pxl_vals=[1, 2],
        max_dist=25.0,
        no_data_val=-1,
        out_no_data_val=-999.0,
        unit_geo=True,
        tile_size=4,
        n_cores=1,
    )
    with rasterio.open(d_full) as s1, rasterio.open(d_tiled) as s2:
        a1 = s1.read(1)
        a2 = s2.read(1)
        np.testing.assert_allclose(a1, a2)
        assert a1[0, 0] == pytest.approx(0.0)
        assert a1[0, 1] == pytest.approx(10.0)
        assert a1[1, 1] == pytest.approx(10.0 * np.sqrt(2.0), rel=1e-5)
        assert a1[7, 7] == pytest.approx(-999.0)

    # Also verify multi-core tiled execution produces identical result
    d_tiled_mc = tmp_path / "d_tiled_mc.tif"
    ic_dist.calc_dist_to_img_vals_tiled(
        str(in_p),
        str(d_tiled_mc),
        pxl_vals=[1, 2],
        max_dist=25.0,
        no_data_val=-1,
        out_no_data_val=-999.0,
        unit_geo=True,
        tile_size=4,
        n_cores=2,
    )
    with rasterio.open(d_full) as s1, rasterio.open(d_tiled_mc) as s3:
        np.testing.assert_allclose(s1.read(1), s3.read(1))

    # Empty target mask must return out_no_data_val everywhere (no corner artefact)
    d_empty = tmp_path / "d_empty.tif"
    ic_dist.calc_dist_to_img_vals(
        str(in_p),
        str(d_empty),
        pxl_vals=[999],
        max_dist=25.0,
        out_no_data_val=-999.0,
    )
    with rasterio.open(d_empty) as s_emp:
        np.testing.assert_allclose(s_emp.read(1), np.full((8, 8), -999.0, dtype=np.float32))

    # Buffer mask (< 15m around value 1 -> centre + 4-neighbours within 10m)
    ic_dist.buffer_img_pxl_vals(
        str(in_p),
        str(buf_p),
        pxl_vals=[1],
        buf_thres=15.0,
        tmp_dir=str(tmp_path),
        unit_geo=True,
    )
    with rasterio.open(buf_p) as s_buf:
        b = s_buf.read(1)
        assert b.dtype == np.uint8
        assert b[0, 0] == 1 and b[0, 1] == 1 and b[1, 1] == 1
        assert b[0, 2] == 0
