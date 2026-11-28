"""
Tests for :mod:`rsgispy.imageutils.rasterise`: rasterisation, LUT score
images and distance zones, from files and in-memory GeoDataFrames.
"""

import math

import geopandas as gpd
import numpy as np
import pytest
import rasterio
import rasterio.crs
from rasterio.transform import from_origin
from shapely.geometry import Point, box

import rsgispy
import rsgispy.imageutils.rasterise as iu_ras
import rsgispy.imageutils.spatial as iu_spat
import rsgispy.imageutils.subsetting as iu_sub


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


def test_imageutils_rasterise(tmp_path):
    """Test rsgispy.imageutils.rasterise rasterisation, LUT score images, and distance zones."""
    ref_img = str(tmp_path / "ref.tif")
    _write_test_tif(
        ref_img,
        np.zeros((20, 20), dtype=np.uint8),
        tl_x=500000.0,
        tl_y=5600200.0,
        res=10.0,
        crs="EPSG:32630",
    )

    vec_file = str(tmp_path / "polys.gpkg")
    gpd.GeoDataFrame(
        {"val": [15.0, 75.0], "cls": ["Low", "High"]},
        geometry=[
            box(500010.0, 5600150.0, 500050.0, 5600190.0),
            box(500100.0, 5600050.0, 500150.0, 5600100.0),
        ],
        crs="EPSG:32630",
    ).to_file(vec_file, layer="polys", driver="GPKG")

    ras_fid = str(tmp_path / "ras_fid.tif")
    iu_ras.rasterise_vec_lyr(
        vec_file,
        "polys",
        ref_img,
        ras_fid,
        datatype=rsgispy.TYPE_8UINT,
        att_column="FID",
        calc_stats=True,
        use_pbar=False,
    )
    with rasterio.open(ras_fid) as src:
        data = src.read(1)
        assert set(np.unique(data)) == {0, 1, 2}

    ras_ext = str(tmp_path / "ras_ext.tif")
    iu_ras.rasterise_vec_lyr(
        vec_file,
        "polys",
        ref_img,
        ras_ext,
        burn_val=5,
        use_vec_extent=True,
        calc_stats=False,
        use_pbar=False,
    )
    with rasterio.open(ras_ext) as src:
        assert src.width == 14
        assert src.height == 14
        assert 5 in np.unique(src.read(1))

    range_img = str(tmp_path / "range_score.tif")
    iu_ras.create_vec_range_lut_score_img(
        vec_file,
        "polys",
        "val",
        ref_img,
        range_img,
        scrs_lut={1: (0.0, 50.0), 2: (50.0, 100.0)},
        tmp_vec_file=str(tmp_path / "tmp_range.gpkg"),
        tmp_vec_lyr="polys",
        tmp_vec_col="score",
        use_pbar=False,
    )
    with rasterio.open(range_img) as src:
        assert set(np.unique(src.read(1))) == {0, 1, 2}

    lst_img = str(tmp_path / "lst_score.tif")
    iu_ras.create_vec_lst_lut_score_img(
        vec_file,
        "polys",
        "cls",
        ref_img,
        lst_img,
        scrs_lut=[("Low", 3), ("High", 7)],
        tmp_vec_file=str(tmp_path / "tmp_lst.gpkg"),
        tmp_vec_lyr="polys",
        tmp_vec_col="score",
        use_pbar=False,
    )
    with rasterio.open(lst_img) as src:
        assert set(np.unique(src.read(1))) == {0, 3, 7}

    dist_zones = str(tmp_path / "dist_zones.tif")
    iu_ras.create_dist_zones_to_vec_lyr(
        vec_file,
        "polys",
        ref_img,
        dist_zones,
        recode_lut=[(1, (math.nan, 15.0)), (2, (15.0, math.nan))],
        max_dist_thres=40.0,
        backgrd_val=0,
        tmp_vec_img=str(tmp_path / "tmp_v.tif"),
        tmp_dist_img=str(tmp_path / "tmp_d.tif"),
        use_pbar=False,
    )
    with rasterio.open(dist_zones) as src:
        assert set(np.unique(src.read(1))) == {0, 1, 2}


def test_imageutils_gdf_in_memory(tmp_path):
    """Test grid_scattered_pts, subset, and rasterise with GeoDataFrame inputs."""
    ref_img = str(tmp_path / "ref.tif")
    _write_test_tif(
        ref_img,
        np.ones((20, 20), dtype=np.float32),
        tl_x=500000.0,
        tl_y=5600200.0,
        res=10.0,
    )

    pts_gdf = gpd.GeoDataFrame(
        {"val": [10.0, 20.0, 30.0, 40.0]},
        geometry=[
            Point(500005.0, 5600195.0),
            Point(500195.0, 5600195.0),
            Point(500005.0, 5600005.0),
            Point(500195.0, 5600005.0),
        ],
        crs="EPSG:32630",
    )
    grid_out = str(tmp_path / "grid_gdf.tif")
    iu_spat.grid_scattered_pts(
        pts_gdf,
        None,
        "val",
        ref_img,
        grid_out,
        gdal_grid_alg="invdistnn:power=2.0:radius=300.0:max_points=4",
        gdalformat="GTIFF",
    )
    with rasterio.open(grid_out) as g_src:
        assert g_src.read(1)[0, 0] == pytest.approx(10.0)

    polys_gdf = gpd.GeoDataFrame(
        {"val": [15.0, 75.0], "cls": ["Low", "High"]},
        geometry=[
            box(500010.0, 5600150.0, 500050.0, 5600190.0),
            box(500100.0, 5600050.0, 500150.0, 5600100.0),
        ],
        crs="EPSG:32630",
    )
    sub_out = str(tmp_path / "sub_gdf.tif")
    iu_sub.subset(
        ref_img,
        polys_gdf,
        None,
        sub_out,
        gdalformat="GTIFF",
        datatype=rsgispy.TYPE_32FLOAT,
    )
    with rasterio.open(sub_out) as s_src:
        assert s_src.width > 0 and s_src.height > 0

    ras_out = str(tmp_path / "ras_gdf.tif")
    iu_ras.rasterise_vec_lyr(
        polys_gdf,
        None,
        ref_img,
        ras_out,
        datatype=rsgispy.TYPE_8UINT,
        att_column="FID",
        calc_stats=False,
        use_pbar=False,
    )
    with rasterio.open(ras_out) as r_src:
        assert set(np.unique(r_src.read(1))) == {0, 1, 2}

    range_img = str(tmp_path / "range_gdf.tif")
    iu_ras.create_vec_range_lut_score_img(
        polys_gdf,
        None,
        "val",
        ref_img,
        range_img,
        scrs_lut={1: (0.0, 50.0), 2: (50.0, 100.0)},
        tmp_vec_file=None,
        tmp_vec_lyr=None,
        tmp_vec_col="score",
        use_pbar=False,
    )
    with rasterio.open(range_img) as src:
        assert set(np.unique(src.read(1))) == {0, 1, 2}

    lst_img = str(tmp_path / "lst_gdf.tif")
    iu_ras.create_vec_lst_lut_score_img(
        polys_gdf,
        None,
        "cls",
        ref_img,
        lst_img,
        scrs_lut=[("Low", 3), ("High", 7)],
        tmp_vec_file=None,
        tmp_vec_lyr=None,
        tmp_vec_col="score",
        use_pbar=False,
    )
    with rasterio.open(lst_img) as src:
        assert set(np.unique(src.read(1))) == {0, 3, 7}

    dist_zones = str(tmp_path / "dist_zones_gdf.tif")
    iu_ras.create_dist_zones_to_vec_lyr(
        polys_gdf,
        None,
        ref_img,
        dist_zones,
        recode_lut=[(1, (math.nan, 15.0)), (2, (15.0, math.nan))],
        max_dist_thres=40.0,
        backgrd_val=0,
        tmp_vec_img=str(tmp_path / "tmp_v2.tif"),
        tmp_dist_img=str(tmp_path / "tmp_d2.tif"),
        use_pbar=False,
    )
    with rasterio.open(dist_zones) as src:
        assert set(np.unique(src.read(1))) == {0, 1, 2}
