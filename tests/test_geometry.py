"""
Tests for rsgispy.tools.geometry using pyproj.
"""

import pyproj
import pytest
import shapely.wkt

import rsgispy
import rsgispy.tools.geometry as gt


def test_reproj_point_and_bbox():
    """Test reproj_point, reproj_point_to_wgs84, reproj_bbox, and reproj_bbox_epsg."""
    crs_utm = pyproj.CRS.from_epsg(32630)
    crs_wgs = pyproj.CRS.from_epsg(4326)

    lon, lat = gt.reproj_point(crs_utm, crs_wgs, 500000.0, 5200000.0)
    assert lon == pytest.approx(-3.0, abs=1e-6)
    assert lat == pytest.approx(46.953529, abs=1e-5)

    lon2, lat2 = gt.reproj_point_to_wgs84(crs_utm, 500000.0, 5200000.0)
    assert (lon2, lat2) == pytest.approx((lon, lat))

    # Round-trip back to UTM 30N
    x_rt, y_rt = gt.reproj_point("EPSG:4326", 32630, lon, lat)
    assert x_rt == pytest.approx(500000.0, abs=1e-3)
    assert y_rt == pytest.approx(5200000.0, abs=1e-3)

    utm_bbox = (500000.0, 510000.0, 5200000.0, 5210000.0)
    wgs_bbox = gt.reproj_bbox(utm_bbox, crs_utm, crs_wgs)
    wgs_bbox_epsg = gt.reproj_bbox_epsg(utm_bbox, 32630, 4326)
    assert wgs_bbox == pytest.approx(wgs_bbox_epsg)
    assert wgs_bbox[0] == pytest.approx(-3.0, abs=1e-5)
    assert wgs_bbox[1] > wgs_bbox[0]
    assert wgs_bbox[3] > wgs_bbox[2]

    with pytest.raises(rsgispy.RSGISPyException):
        gt.reproj_point("NOT_A_VALID_CRS", 4326, 0.0, 0.0)


def test_bbox_relations_and_operations():
    """Test intersection, containment, equality, area, buffer, union, and unwrap."""
    b1 = (0.0, 10.0, 0.0, 10.0)
    b2 = (5.0, 15.0, 5.0, 15.0)
    b3 = (20.0, 30.0, 20.0, 30.0)
    b_inner = (2.0, 8.0, 2.0, 8.0)

    assert gt.do_bboxes_intersect(b1, b2)
    assert not gt.do_bboxes_intersect(b1, b3)
    assert gt.does_bbox_contain(b1, b_inner)
    assert not gt.does_bbox_contain(b1, b2)
    assert gt.calc_bbox_area(b1) == pytest.approx(100.0)
    assert gt.bbox_equal(b1, (0.0, 10.0, 0.0, 10.0))
    assert not gt.bbox_equal(b1, b2)

    assert gt.bbox_intersection(b1, b2) == (5.0, 10.0, 5.0, 10.0)
    with pytest.raises(rsgispy.RSGISPyException):
        gt.bbox_intersection(b1, b3)

    assert gt.bboxes_intersection([b1, b2, b_inner]) == (5.0, 8.0, 5.0, 8.0)
    with pytest.raises(rsgispy.RSGISPyException):
        gt.bboxes_intersection([])

    assert gt.buffer_bbox(b1, 2.5) == (-2.5, 12.5, -2.5, 12.5)
    assert gt.find_bbox_union([b1, b2, b3]) == (0.0, 30.0, 0.0, 30.0)
    assert gt.find_bbox_union([]) is None

    assert gt.unwrap_wgs84_bbox((170.0, -170.0, -10.0, 10.0)) == [
        (-180.0, -170.0, -10.0, 10.0),
        (170.0, 180.0, -10.0, 10.0),
    ]
    assert gt.unwrap_wgs84_bbox((-10.0, 10.0, -5.0, 5.0)) == [(-10.0, 10.0, -5.0, 5.0)]


def test_grid_and_extent_helpers():
    """Test extent grid alignment, bbox tiling, polygons, and point helpers."""
    common_contain = gt.find_common_extent_on_grid(
        (0.0, 100.0, 0.0, 100.0), 10.0, (12.0, 88.0, 15.0, 84.0), full_contain=True
    )
    assert common_contain == (10.0, 90.0, 10.0, 90.0)

    common_inner = gt.find_common_extent_on_grid(
        (0.0, 100.0, 0.0, 100.0), 10.0, (12.0, 88.0, 15.0, 84.0), full_contain=False
    )
    assert common_inner == (20.0, 80.0, 20.0, 80.0)

    ext_grid = gt.find_extent_on_grid((12.3, 47.2, 5.1, 38.6), 5.0, full_contain=True)
    assert ext_grid[0] <= 12.3 and ext_grid[1] >= 47.2
    assert ext_grid[2] <= 5.1 and ext_grid[3] >= 38.6

    whole_grid = gt.find_extent_on_whole_num_grid(
        (12.3, 47.2, 5.1, 38.6), 10.0, full_contain=True, round_vals=2
    )
    assert whole_grid == (10.0, 50.0, 0.0, 40.0)

    tiles = gt.get_bbox_grid((0.0, 25.0, 0.0, 15.0), 10.0, 10.0)
    assert len(tiles) == 6
    assert tiles[0] == (0.0, 10.0, 0.0, 10.0)
    assert tiles[1] == (10.0, 20.0, 0.0, 10.0)
    assert tiles[2] == (20.0, 25.0, 0.0, 10.0)
    assert tiles[3] == (0.0, 10.0, 10.0, 15.0)
    tiles_full = gt.get_bbox_grid((0.0, 25.0, 0.0, 25.0), 10.0, 10.0)
    assert len(tiles_full) == 9

    geojson = gt.get_bbox_geojson_poly((0.0, 10.0, 2.0, 8.0))
    assert geojson["type"] == "Polygon"
    assert len(geojson["coordinates"][0]) == 5

    wkt_poly = gt.get_bbox_wkt_poly((0.0, 10.0, 2.0, 8.0))
    geom = shapely.wkt.loads(wkt_poly)
    assert geom.bounds == (0.0, 2.0, 10.0, 8.0)

    assert gt.get_bbox_centre_pt((0.0, 10.0, 2.0, 8.0)) == (5.0, 5.0)
    assert gt.calc_pt_distance(0.0, 0.0, 3.0, 4.0) == pytest.approx(5.0)
    assert gt.pt_in_bbox((5.0, 5.0), (0.0, 10.0, 2.0, 8.0))
    assert not gt.pt_in_bbox((15.0, 5.0), (0.0, 10.0, 2.0, 8.0))

    snapped = gt.find_point_on_whole_num_grid((2.3, 6.2), (0.0, 10.0, 0.0, 10.0), 1.0, 1.0)
    assert snapped == (3.0, 6.0)
    with pytest.raises(rsgispy.RSGISPyException):
        gt.find_point_on_whole_num_grid((20.0, 5.0), (0.0, 10.0, 0.0, 10.0), 1.0, 1.0)
