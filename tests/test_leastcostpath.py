"""
Tests for :mod:`rsgispy.imagecalc.leastcostpath`.
"""

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine, from_origin

import rsgispy
import rsgispy.imagecalc.leastcostpath as lcp


def _write_cost_raster(path, arr, origin=(0.0, 100.0), res=10.0, crs="EPSG:32630"):
    """Write a single- or multi-band GeoTIFF cost surface."""
    data = arr[np.newaxis, :, :] if arr.ndim == 2 else arr
    count, height, width = data.shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=str(data.dtype),
        crs=crs,
        transform=from_origin(origin[0], origin[1], res, res),
    ) as dst:
        dst.write(data)


def test_least_cost_path_avoids_barrier(tmp_path):
    """Verify least-cost path routes around a high-cost wall."""
    cost_img = tmp_path / "cost.tif"
    out_img = tmp_path / "path.tif"

    cost = np.ones((7, 7), dtype=np.float32)
    cost[1:6, 3] = 1000.0  # high barrier forcing path to route around row 0 or 6
    _write_cost_raster(cost_img, cost)

    # Start (row=3, col=1) -> x=15, y=65 ; Stop (row=3, col=5) -> x=55, y=65
    lcp.perform_least_cost_path_calc(
        str(cost_img),
        str(out_img),
        start_coord=(15.0, 65.0),
        stop_coord=(55.0, 65.0),
    )

    with rasterio.open(out_img) as src:
        path = src.read(1)
        assert src.dtypes[0] == "uint8"

    assert path[3, 1] == 1
    assert path[3, 5] == 1
    assert np.all(path[1:6, 3] == 0)
    assert np.sum(path) >= 7


def test_least_cost_path_dijkstra_fallback_matches_skimage(tmp_path, monkeypatch):
    """Verify pure-Python Dijkstra fallback runs and avoids high-cost barrier."""
    cost_img = tmp_path / "cost_fb.tif"
    out_sk = tmp_path / "path_sk.tif"
    out_fb = tmp_path / "path_fb.tif"

    cost = np.ones((6, 6), dtype=np.float32)
    cost[0:5, 2] = 500.0  # only row 5 column 2 has cost 1
    _write_cost_raster(cost_img, cost)

    start = (5.0, 85.0)  # row=1, col=0
    stop = (45.0, 85.0)  # row=1, col=4

    monkeypatch.setattr(lcp, "HAVE_SKIMAGE", False)
    lcp.perform_least_cost_path_calc(str(cost_img), str(out_fb), start, stop)

    with rasterio.open(out_fb) as src:
        fb_path = src.read(1)
    assert fb_path[1, 0] == 1
    assert fb_path[1, 4] == 1
    assert fb_path[5, 2] == 1
    assert np.all(fb_path[0:5, 2] == 0)

    monkeypatch.undo()
    if lcp.HAVE_SKIMAGE:
        lcp.perform_least_cost_path_calc(str(cost_img), str(out_sk), start, stop)
        with rasterio.open(out_sk) as src:
            sk_path = src.read(1)
        np.testing.assert_array_equal(fb_path, sk_path)


def test_least_cost_path_band_selection_and_validation(tmp_path):
    """Verify band selection, out-of-bounds validation, and negative cost error."""
    cost_img = tmp_path / "multi_cost.tif"
    out_img = tmp_path / "path_b2.tif"

    b1 = np.ones((5, 5), dtype=np.float32)
    b2 = np.ones((5, 5), dtype=np.float32)
    b2[0:4, 2] = 500.0
    _write_cost_raster(cost_img, np.stack([b1, b2], axis=0))

    lcp.perform_least_cost_path_calc(
        str(cost_img),
        str(out_img),
        start_coord=(5.0, 95.0),
        stop_coord=(45.0, 95.0),
        cost_img_band=2,
    )
    with rasterio.open(out_img) as src:
        arr = src.read(1)
    assert arr[4, 2] == 1
    assert np.all(arr[0:4, 2] == 0)

    with pytest.raises(rsgispy.RSGISPyException):
        lcp.perform_least_cost_path_calc(
            str(cost_img),
            str(out_img),
            start_coord=(-10.0, 95.0),
            stop_coord=(45.0, 95.0),
        )

    neg_img = tmp_path / "neg.tif"
    neg_arr = np.ones((4, 4), dtype=np.float32)
    neg_arr[1, 1] = -2.0
    _write_cost_raster(neg_img, neg_arr)
    with pytest.raises(rsgispy.RSGISPyException):
        lcp.perform_least_cost_path_calc(
            str(neg_img),
            str(out_img),
            start_coord=(5.0, 95.0),
            stop_coord=(35.0, 65.0),
        )


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


def test_least_cost_path_avoids_nan_and_nodata_barriers(tmp_path):
    cost = np.ones((7, 7), dtype=np.float32)
    # Build vertical wall at col=3 except row=6, mixing NaN and nodata (-999)
    cost[0:3, 3] = np.nan
    cost[3:6, 3] = -999.0
    tf = from_origin(0.0, 70.0, 10.0, 10.0)
    cost_img = _write_raster(
        tmp_path / "cost.tif", cost, transform=tf, nodata=-999.0
    )
    out_img = str(tmp_path / "lcp.tif")

    # Start (row=1, col=1) -> (15, 55), Stop (row=1, col=5) -> (55, 55)
    lcp.perform_least_cost_path_calc(
        cost_surface_img=cost_img,
        output_img=out_img,
        start_coord=(15.0, 55.0),
        stop_coord=(55.0, 55.0),
    )
    with rasterio.open(out_img) as dst:
        path_arr = dst.read(1)
        assert path_arr[1, 1] == 1
        assert path_arr[1, 5] == 1
        # Must not cross any barrier cell, and must pass through gap at (6, 3)
        assert not np.any(path_arr[0:6, 3] == 1)
        assert path_arr[6, 3] == 1

    # Start inside nodata cell must raise RSGISPyException
    with pytest.raises(rsgispy.RSGISPyException):
        lcp.perform_least_cost_path_calc(
            cost_surface_img=cost_img,
            output_img=str(tmp_path / "lcp_bad.tif"),
            start_coord=(35.0, 35.0),  # (row=3, col=3) is -999
            stop_coord=(55.0, 55.0),
        )
