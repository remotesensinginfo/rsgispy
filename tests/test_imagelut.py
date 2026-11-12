"""
Tests for rsgispy.imageutils.imagelut using rasterio and geopandas.
"""

import os
import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Polygon

import rsgispy
import rsgispy.imageutils.imagelut


def _write_test_tile(path, x_min, y_max, width=10, height=10, res=10.0, val=1, crs="EPSG:32630"):
    """Helper to write a single-band uint8 GeoTIFF tile."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=1,
        dtype="uint8",
        crs=crs,
        transform=from_origin(x_min, y_max, res, res),
        nodata=0,
    ) as dst:
        dst.write(np.full((1, height, width), val, dtype="uint8"))


def test_create_and_query_img_extent_lut_and_get_raster_lyr(tmp_path):
    """Test create_img_extent_lut, get_all_lut_imgs, query_img_lut, and get_raster_lyr."""
    tile1 = tmp_path / "tile1.tif"
    tile2 = tmp_path / "tile2.tif"
    tile3 = tmp_path / "tile3.tif"
    tile_nocrs = tmp_path / "tile_nocrs.tif"
    tile_diffcrs = tmp_path / "tile_diffcrs.tif"

    # tile1: x=[500000, 500100], y=[5199900, 5200000]
    _write_test_tile(tile1, 500000.0, 5200000.0, val=10)
    # tile2: x=[500100, 500200], y=[5199900, 5200000]
    _write_test_tile(tile2, 500100.0, 5200000.0, val=20)
    # tile3: x=[500500, 500600], y=[5199900, 5200000] (disjoint)
    _write_test_tile(tile3, 500500.0, 5200000.0, val=30)
    # tile without CRS and tile with different CRS
    _write_test_tile(tile_nocrs, 500000.0, 5200000.0, val=1, crs=None)
    _write_test_tile(tile_diffcrs, 500000.0, 5200000.0, val=1, crs="EPSG:3857")

    lut_gpkg = tmp_path / "img_lut.gpkg"

    # Error if tile has no CRS and ignore_none_imgs=False
    with pytest.raises(rsgispy.RSGISPyException):
        rsgispy.imageutils.imagelut.create_img_extent_lut(
            [str(tile1), str(tile_nocrs)],
            str(lut_gpkg),
            "tiles",
            "GPKG",
            ignore_none_imgs=False,
            use_pbar=False,
        )

    # Error if tiles have differing projections and out_proj_wgs84=False
    with pytest.raises(rsgispy.RSGISPyException):
        rsgispy.imageutils.imagelut.create_img_extent_lut(
            [str(tile1), str(tile_diffcrs)],
            str(lut_gpkg),
            "tiles",
            "GPKG",
            use_pbar=False,
        )

    # Create LUT skipping tile_nocrs
    rsgispy.imageutils.imagelut.create_img_extent_lut(
        [str(tile1), str(tile2), str(tile3), str(tile_nocrs)],
        str(lut_gpkg),
        "tiles",
        "GPKG",
        ignore_none_imgs=True,
        use_pbar=False,
    )

    all_imgs = rsgispy.imageutils.imagelut.get_all_lut_imgs(str(lut_gpkg), "tiles")
    assert set(all_imgs) == {str(tile1), str(tile2), str(tile3)}

    # Query bbox (MinX, MaxX, MinY, MaxY) covering only tile1
    res_single = rsgispy.imageutils.imagelut.query_img_lut(
        (500010.0, 500050.0, 5199910.0, 5199950.0),
        str(lut_gpkg),
        "tiles",
    )
    assert res_single == [str(tile1)]

    # Query bbox covering tile1 and tile2
    res_two = rsgispy.imageutils.imagelut.query_img_lut(
        (500050.0, 500150.0, 5199910.0, 5199950.0),
        str(lut_gpkg),
        "tiles",
    )
    assert set(res_two) == {str(tile1), str(tile2)}

    # Query bbox outside all tiles
    res_none = rsgispy.imageutils.imagelut.query_img_lut(
        (501000.0, 501100.0, 5199910.0, 5199950.0),
        str(lut_gpkg),
        "tiles",
    )
    assert res_none == []

    # get_raster_lyr: 0 matches -> None
    vrt_tmp = tmp_path / "vrt_tmp"
    assert (
        rsgispy.imageutils.imagelut.get_raster_lyr(
            (501000.0, 501100.0, 5199910.0, 5199950.0),
            str(lut_gpkg),
            "tiles",
            str(vrt_tmp),
        )
        is None
    )

    # get_raster_lyr: 1 match -> direct path
    assert (
        rsgispy.imageutils.imagelut.get_raster_lyr(
            (500010.0, 500050.0, 5199910.0, 5199950.0),
            str(lut_gpkg),
            "tiles",
            str(vrt_tmp),
        )
        == str(tile1)
    )

    # get_raster_lyr: >1 matches -> VRT in tmp_dir readable by rasterio
    vrt_out = rsgispy.imageutils.imagelut.get_raster_lyr(
        (500050.0, 500150.0, 5199910.0, 5199950.0),
        str(lut_gpkg),
        "tiles",
        str(vrt_tmp),
    )
    assert vrt_out is not None
    assert vrt_out.endswith(".vrt")
    assert os.path.exists(vrt_out)
    with rasterio.open(vrt_out) as vrt_src:
        assert vrt_src.width == 20
        assert vrt_src.height == 10
        arr = vrt_src.read(1)
        assert np.all(arr[:, :10] == 10)
        assert np.all(arr[:, 10:] == 20)

    # Add second layer in WGS84 without overwriting file, then verify overwrite_lut_file=True
    rsgispy.imageutils.imagelut.create_img_extent_lut(
        [str(tile1), str(tile_diffcrs)],
        str(lut_gpkg),
        "tiles_wgs84",
        "GPKG",
        out_proj_wgs84=True,
        overwrite_lut_file=False,
        use_pbar=False,
    )
    wgs84_gdf = gpd.read_file(lut_gpkg, layer="tiles_wgs84")
    assert wgs84_gdf.crs.to_epsg() == 4326
    assert len(wgs84_gdf) == 2
    # Original 'tiles' layer still exists
    assert len(rsgispy.imageutils.imagelut.get_all_lut_imgs(str(lut_gpkg), "tiles")) == 3

    # Overwrite GPKG with a single layer
    rsgispy.imageutils.imagelut.create_img_extent_lut(
        [str(tile1)],
        str(lut_gpkg),
        "only_tile1",
        "GPKG",
        overwrite_lut_file=True,
        use_pbar=False,
    )
    assert rsgispy.imageutils.imagelut.get_all_lut_imgs(
        str(lut_gpkg), "only_tile1"
    ) == [str(tile1)]


def test_query_file_lut(tmp_path):
    """Test query_file_lut with polygon ROI, cross-CRS reprojection, and command modes."""
    tile1 = tmp_path / "tile1.tif"
    tile2 = tmp_path / "tile2.tif"
    tile3 = tmp_path / "tile3.tif"

    _write_test_tile(tile1, 500000.0, 5200000.0, val=1)
    _write_test_tile(tile2, 500100.0, 5200000.0, val=2)
    _write_test_tile(tile3, 500500.0, 5200000.0, val=3)

    lut_gpkg = tmp_path / "lut.gpkg"
    rsgispy.imageutils.imagelut.create_img_extent_lut(
        [str(tile1), str(tile2), str(tile3)],
        str(lut_gpkg),
        "tiles",
        "GPKG",
        use_pbar=False,
    )

    # Diagonal ROI polygon whose bounding box would cover tile3 if not checking exact intersection,
    # Wait: let's build an L-shaped ROI touching tile1 and tile2 but NOT tile3.
    roi_poly = Polygon(
        [
            (500020.0, 5199920.0),
            (500150.0, 5199920.0),
            (500150.0, 5199980.0),
            (500020.0, 5199980.0),
            (500020.0, 5199920.0),
        ]
    )
    roi_gpkg = tmp_path / "roi.gpkg"
    roi_gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[roi_poly], crs="EPSG:32630")
    roi_gdf.to_file(roi_gpkg, layer="roi", driver="GPKG")

    # Reproject ROI to WGS84 to verify automatic CRS alignment in query_file_lut
    roi_wgs84_gpkg = tmp_path / "roi_wgs84.gpkg"
    roi_gdf.to_crs("EPSG:4326").to_file(roi_wgs84_gpkg, layer="roi", driver="GPKG")

    files_same_crs = rsgispy.imageutils.imagelut.query_file_lut(
        str(lut_gpkg), "tiles", str(roi_gpkg), "roi"
    )
    assert set(files_same_crs) == {str(tile1), str(tile2)}

    files_diff_crs = rsgispy.imageutils.imagelut.query_file_lut(
        str(lut_gpkg), "tiles", str(roi_wgs84_gpkg), "roi"
    )
    assert set(files_diff_crs) == {str(tile1), str(tile2)}

    cp_cmds = rsgispy.imageutils.imagelut.query_file_lut(
        str(lut_gpkg),
        "tiles",
        str(roi_gpkg),
        "roi",
        out_dest="/tmp/dest",
        cp_cmds=True,
    )
    assert cp_cmds == [f"cp {p} /tmp/dest" for p in files_same_crs]

    tar_cmds = rsgispy.imageutils.imagelut.query_file_lut(
        str(lut_gpkg),
        "tiles",
        str(roi_gpkg),
        "roi",
        out_dest="/tmp/out.tar.gz",
        targz_out=True,
    )
    assert len(tar_cmds) == 1
    assert tar_cmds[0].startswith("tar -czf /tmp/out.tar.gz ")
    assert str(tile1) in tar_cmds[0]
    assert str(tile2) in tar_cmds[0]

    with pytest.raises(rsgispy.RSGISPyException):
        rsgispy.imageutils.imagelut.query_file_lut(
            str(lut_gpkg),
            "tiles",
            str(roi_gpkg),
            "roi",
            out_dest="/tmp/out",
            targz_out=True,
            cp_cmds=True,
        )
