"""Tests for the typed rsgispy.BBox class and cross-module BBox integration."""

import json
import pickle

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Polygon

import rsgispy
import rsgispy.imageutils.metadata as img_meta
import rsgispy.imageutils.vrt as img_vrt
import rsgispy.imageutils.warp as img_warp
import rsgispy.mapping.layout as map_layout
import rsgispy.mapping.legend as map_legend
import rsgispy.tools.geometry as geom_tools
import rsgispy.vectorprocess as vec_proc


def test_bbox_core_api_and_tuple_compat():
    """Verify BBox tuple compatibility, conversions, geometry methods, and pickle/JSON."""
    bbox = rsgispy.BBox(500000.0, 501000.0, 5200000.0, 5202000.0, epsg=32630)

    # Tuple unpacking & indexing (RSGIS order)
    w, e, s, n = bbox
    assert (w, e, s, n) == (500000.0, 501000.0, 5200000.0, 5202000.0)
    assert len(bbox) == 4
    assert bbox[:4] == (500000.0, 501000.0, 5200000.0, 5202000.0)
    assert bbox == (500000.0, 501000.0, 5200000.0, 5202000.0)
    assert bbox == pytest.approx((500000.0, 501000.0, 5200000.0, 5202000.0))

    # Named properties
    assert bbox.west == bbox.xmin == 500000.0
    assert bbox.east == bbox.xmax == 501000.0
    assert bbox.south == bbox.ymin == 5200000.0
    assert bbox.north == bbox.ymax == 5202000.0
    assert bbox.epsg == 32630
    assert bbox.width == 1000.0
    assert bbox.height == 2000.0
    assert bbox.area == 2_000_000.0
    assert bbox.centre == (500500.0, 5201000.0)

    # Explicit RSGIS vs GEO conversions
    assert bbox.as_rsgis() == (500000.0, 501000.0, 5200000.0, 5202000.0)
    assert bbox.as_geo() == (500000.0, 5200000.0, 501000.0, 5202000.0)
    assert rsgispy.BBox.from_geo(bbox.as_geo(), epsg=32630).equals(bbox, check_epsg=True)
    assert rsgispy.BBox.from_rsgis(bbox.as_rsgis(), epsg=32630).equals(bbox, check_epsg=True)
    assert rsgispy.BBox.from_geo(bbox).equals(bbox, check_epsg=True)

    # Polygon, buffer, intersection, union, contains_point
    poly = bbox.to_polygon()
    assert poly.bounds == bbox.as_geo()
    assert bbox.contains_point(500500.0, 5201000.0)
    assert not bbox.contains_point(499000.0, 5201000.0)

    buf = bbox.buffer(50.0)
    assert buf == (499950.0, 501050.0, 5199950.0, 5202050.0)
    assert buf.epsg == 32630

    other = rsgispy.BBox(500500.0, 501500.0, 5201000.0, 5203000.0, epsg=32630)
    assert bbox.intersects(other)
    inter = bbox.intersection(other)
    assert inter == (500500.0, 501000.0, 5201000.0, 5202000.0)
    assert inter.epsg == 32630
    union = bbox.union(other)
    assert union == (500000.0, 501500.0, 5200000.0, 5203000.0)
    assert bbox.intersection((0.0, 10.0, 0.0, 10.0)) is None

    # Reprojection
    b_wgs = bbox.reproject(4326)
    assert isinstance(b_wgs, rsgispy.BBox)
    assert b_wgs.epsg == 4326
    assert -5.0 < b_wgs.west < -2.0

    # Pickle, JSON, dict roundtrip
    unpickled = pickle.loads(pickle.dumps(bbox))
    assert unpickled.equals(bbox, check_epsg=True)
    assert json.loads(json.dumps(bbox)) == [500000.0, 501000.0, 5200000.0, 5202000.0]
    assert rsgispy.BBox.from_dict(bbox.to_dict()).equals(bbox, check_epsg=True)


def test_bbox_integration_raster_vector_warp_vrt_legend(tmp_path):
    """Verify BBox round-trips cleanly across raster, vector, VRT, warp, and legend."""
    img_path = tmp_path / "thm.tif"
    arr = np.array(
        [
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [3, 3, 3, 3],
            [3, 3, 3, 3],
        ],
        dtype=np.uint8,
    )
    with rasterio.open(
        img_path,
        "w",
        driver="GTiff",
        height=4,
        width=4,
        count=1,
        dtype="uint8",
        crs="EPSG:32630",
        transform=from_origin(500000.0, 5200040.0, 10.0, 10.0),
        nodata=0,
    ) as dst:
        dst.write(arr[np.newaxis, ...])
        dst.write_colormap(
            1,
            {
                0: (0, 0, 0, 0),
                1: (34, 139, 34, 255),
                2: (218, 165, 32, 255),
                3: (30, 144, 255, 255),
            },
        )

    img_bbox = img_meta.get_img_bbox(str(img_path))
    assert isinstance(img_bbox, rsgispy.BBox)
    assert img_bbox.epsg == 32630
    assert img_bbox == (500000.0, 500040.0, 5200000.0, 5200040.0)

    # Legend scan with sub-BBox (verifies fixed RSGIS->GEO window conversion)
    top_left_bbox = rsgispy.BBox(500000.0, 500020.0, 5200020.0, 5200040.0, epsg=32630)
    handles = map_legend.create_categorical_legend_patches(
        classes_info={1: "Forest", 2: "Crop", 3: "Water"},
        input_img=str(img_path),
        bbox=top_left_bbox,
        only_present_in_img=True,
    )
    assert [h.get_label() for h in handles] == ["Forest"]

    # VRT extent accepts BBox directly (converts to GEO order internally)
    vrt_out = tmp_path / "sub.vrt"
    img_vrt.create_mosaic_images_vrt([str(img_path)], str(vrt_out), vrt_extent=top_left_bbox)
    assert img_meta.get_img_bbox(str(vrt_out)) == pytest.approx(top_left_bbox)

    # gdal_warp target_extent accepts BBox directly
    warped_out = tmp_path / "warped.tif"
    img_warp.gdal_warp(str(img_path), str(warped_out), target_extent=top_left_bbox)
    assert img_meta.get_img_bbox(str(warped_out)) == pytest.approx(top_left_bbox)

    # Vector read_vec_gdf bbox accepts BBox directly & get_gdf_bbox returns BBox
    gdf = gpd.GeoDataFrame(
        {"id": [1, 2]},
        geometry=[
            Polygon([(500002, 5200022), (500018, 5200022), (500018, 5200038), (500002, 5200038)]),
            Polygon([(500022, 5200002), (500038, 5200002), (500038, 5200018), (500022, 5200018)]),
        ],
        crs="EPSG:32630",
    )
    vec_bbox = vec_proc.get_gdf_bbox(gdf)
    assert isinstance(vec_bbox, rsgispy.BBox)
    assert vec_bbox.epsg == 32630

    sub_gdf = vec_proc.read_vec_gdf(gdf, bbox=top_left_bbox)
    assert list(sub_gdf["id"]) == [1]

    # tools.geometry & mapping.layout return BBox
    reproj_b = geom_tools.reproj_bbox_epsg(img_bbox, 32630, 4326)
    assert isinstance(reproj_b, rsgispy.BBox) and reproj_b.epsg == 4326
    over_b, _ = map_layout.get_overview_info(
        img_bbox, roi_epsg=32630, overview_buf=2.0, out_epsg=4326
    )
    assert isinstance(over_b, rsgispy.BBox) and over_b.epsg == 4326
