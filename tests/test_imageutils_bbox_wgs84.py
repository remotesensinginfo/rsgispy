"""
Direct tests of :func:`rsgispy.imageutils.metadata.get_img_bbox_wgs84`.
"""

import numpy as np
import pyproj
import pytest
import rasterio
from rasterio.transform import from_origin

import rsgispy
from rsgispy.imageutils.metadata import get_img_bbox_wgs84


def _write(path, crs, x0, y0, res, width=40, height=30):
    with rasterio.open(
        path, "w", driver="GTiff", width=width, height=height, count=1,
        dtype="uint8", crs=crs, transform=from_origin(x0, y0, res, res),
    ) as dst:
        dst.write(np.ones((1, height, width), dtype=np.uint8))
    return str(path)


def test_projected_image(tmp_path):
    img = _write(tmp_path / "utm.tif", "EPSG:32630", 500000.0, 5600000.0, 100.0)
    bbox = get_img_bbox_wgs84(img)
    assert isinstance(bbox, rsgispy.BBox) and bbox.epsg == 4326
    # Reference: the image corners and edge points transformed with pyproj.
    trans = pyproj.Transformer.from_crs(32630, 4326, always_xy=True)
    xs = np.r_[np.linspace(500000, 504000, 50), np.full(50, 504000.0),
               np.linspace(500000, 504000, 50), np.full(50, 500000.0)]
    ys = np.r_[np.full(50, 5600000.0), np.linspace(5597000, 5600000, 50),
               np.full(50, 5597000.0), np.linspace(5597000, 5600000, 50)]
    lon, lat = trans.transform(xs, ys)
    np.testing.assert_allclose(
        tuple(bbox), (lon.min(), lon.max(), lat.min(), lat.max()), atol=1e-6
    )
    # 500000 E is the central meridian of zone 30 (3 W), near 50.5 N.
    assert bbox.west == pytest.approx(-3.0, abs=1e-6)
    assert bbox.west < bbox.east and bbox.south < bbox.north
    assert 50.4 < bbox.south < bbox.north < 50.6


def test_geographic_image_unchanged(tmp_path):
    img = _write(tmp_path / "geo.tif", "EPSG:4326", -4.0, 53.0, 0.01)
    bbox = get_img_bbox_wgs84(img)
    np.testing.assert_allclose(tuple(bbox), (-4.0, -3.6, 52.7, 53.0), atol=1e-9)
    assert bbox.epsg == 4326


def test_no_crs_and_missing_file(tmp_path):
    img = _write(tmp_path / "nocrs.tif", None, 0.0, 30.0, 1.0)
    assert get_img_bbox_wgs84(img) is None
    with pytest.raises(rsgispy.RSGISPyException):
        get_img_bbox_wgs84(str(tmp_path / "missing.tif"))
