"""
Tests for rsgispy.vectorgeoms (lines, overlay, polygons, spatial) using
geopandas without osgeo dependencies.
"""

import datetime
import os

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import (
    LineString,
    MultiPolygon,
    Point,
    Polygon,
    box,
)

import rsgispy
import rsgispy.vectorgeoms
import rsgispy.vectorgeoms.create as vg_create
import rsgispy.vectorgeoms.lines as vg_lines
import rsgispy.vectorgeoms.overlay as vg_overlay
import rsgispy.vectorgeoms.polygons as vg_polys
import rsgispy.vectorgeoms.spatial as vg_spatial
import rsgispy.vectorstats.metrics as vs_metrics
import rsgispy.vectorutils.metadata as vu_meta


def test_vectorgeoms_lines(tmp_path):
    """Test line geometry helpers, transects, splitting, intersections, and coastline."""
    x, y = vg_lines.get_pt_on_line(Point(0.0, 0.0), Point(10.0, 0.0), 4.0)
    assert (x, y) == pytest.approx((4.0, 0.0))

    xr, yr = vg_lines.find_pt_to_side(
        Point(0.0, 0.0), Point(5.0, 0.0), Point(10.0, 0.0), 3.0, left_hand=False
    )
    xl, yl = vg_lines.find_pt_to_side(
        Point(0.0, 0.0), Point(5.0, 0.0), Point(10.0, 0.0), 3.0, left_hand=True
    )
    assert (xr, yr) == pytest.approx((5.0, -3.0))
    assert (xl, yl) == pytest.approx((5.0, 3.0))

    lines_gpkg = str(tmp_path / "lines.gpkg")
    gpd.GeoDataFrame(
        {"id": [1]}, geometry=[LineString([(0.0, 0.0), (100.0, 0.0)])], crs="EPSG:32630"
    ).to_file(lines_gpkg, layer="lines", driver="GPKG")

    orthg_gpkg = str(tmp_path / "orthg.gpkg")
    vg_lines.create_orthg_lines(
        lines_gpkg,
        "lines",
        orthg_gpkg,
        "orthg",
        pt_step=25.0,
        line_len=10.0,
        use_pbar=False,
    )
    orthg_gdf = gpd.read_file(orthg_gpkg, layer="orthg")
    assert len(orthg_gdf) == 3
    assert orthg_gdf["start_x"].tolist() == pytest.approx([25.0, 50.0, 75.0])
    assert orthg_gdf["end_y"].tolist() == pytest.approx([-10.0, -10.0, -10.0])

    reg_pts_gpkg = str(tmp_path / "reg_pts.gpkg")
    vg_lines.split_lines_to_reg_pts(
        lines_gpkg, "lines", reg_pts_gpkg, "pts", pt_step=25.0, use_pbar=False
    )
    reg_pts_gdf = gpd.read_file(reg_pts_gpkg, layer="pts")
    assert len(reg_pts_gdf) == 3
    assert reg_pts_gdf["pt_x"].tolist() == pytest.approx([25.0, 50.0, 75.0])

    # Line crossing a polygon wall at x=10 and x=30 from start_x=0
    ray_gpkg = str(tmp_path / "ray.gpkg")
    gpd.GeoDataFrame(
        {"uid": [7], "start_x": [0.0], "start_y": [5.0]},
        geometry=[LineString([(0.0, 5.0), (40.0, 5.0)])],
        crs="EPSG:32630",
    ).to_file(ray_gpkg, layer="ray", driver="GPKG")

    wall_gpkg = str(tmp_path / "wall.gpkg")
    gpd.GeoDataFrame(
        {"id": [1]}, geometry=[box(10.0, 0.0, 30.0, 10.0)], crs="EPSG:32630"
    ).to_file(wall_gpkg, layer="wall", driver="GPKG")

    closest_gpkg = str(tmp_path / "closest.gpkg")
    vg_lines.closest_line_intersection(
        ray_gpkg, "ray", wall_gpkg, "wall", closest_gpkg, "closest", use_pbar=False
    )
    closest_gdf = gpd.read_file(closest_gpkg, layer="closest")
    assert closest_gdf["len"].iloc[0] == pytest.approx(10.0)

    range_gpkg = str(tmp_path / "range.gpkg")
    vg_lines.line_intersection_range(
        ray_gpkg, "ray", wall_gpkg, "wall", range_gpkg, "range", use_pbar=False
    )
    range_gdf = gpd.read_file(range_gpkg, layer="range")
    assert range_gdf["len"].iloc[0] == pytest.approx(20.0)

    scnd_gpkg = str(tmp_path / "scnd.gpkg")
    vg_lines.scnd_line_intersection_range(
        ray_gpkg, "ray", wall_gpkg, "wall", scnd_gpkg, "scnd", use_pbar=False
    )
    scnd_gdf = gpd.read_file(scnd_gpkg, layer="scnd")
    assert scnd_gdf["len"].iloc[0] == pytest.approx(30.0)
    assert scnd_gdf["end_x"].iloc[0] == pytest.approx(30.0)

    # Points to line + angle lines
    pts_gpkg = str(tmp_path / "track_pts.gpkg")
    gpd.GeoDataFrame(
        {"seq": [2, 1]},
        geometry=[Point(10.0, 0.0), Point(0.0, 0.0)],
        crs="EPSG:32630",
    ).to_file(pts_gpkg, layer="pts", driver="GPKG")

    track_line_gpkg = str(tmp_path / "track_line.gpkg")
    vg_lines.pts_to_line_geoms(
        pts_gpkg, "pts", track_line_gpkg, "line", sort_col="seq"
    )
    track_gdf = gpd.read_file(track_line_gpkg, layer="line")
    assert list(track_gdf.geometry.iloc[0].coords) == [(0.0, 0.0), (10.0, 0.0)]

    rays_gpkg = str(tmp_path / "angle_rays.gpkg")
    vg_lines.create_angle_lines_from_points(
        pts_gpkg,
        "pts",
        angle=90.0,
        line_len=5.0,
        out_vec_file=rays_gpkg,
        out_vec_lyr="rays",
        use_pbar=False,
    )
    assert len(gpd.read_file(rays_gpkg, layer="rays")) == 8

    # Estimate future coastline
    coast_geojson = str(tmp_path / "coast.geojson")
    gpd.GeoDataFrame(
        {"id": [1]},
        geometry=[LineString([(0.0, 10.0), (20.0, 10.0)])],
        crs="EPSG:32630",
    ).to_file(coast_geojson, driver="GeoJSON")

    transects_geojson = str(tmp_path / "transects.geojson")
    gpd.GeoDataFrame(
        {"LRR": [1.0, 2.0]},
        geometry=[
            LineString([(5.0, 0.0), (5.0, 20.0)]),
            LineString([(15.0, 0.0), (15.0, 20.0)]),
        ],
        crs="EPSG:32630",
    ).to_file(transects_geojson, driver="GeoJSON")

    pred_geojson = str(tmp_path / "pred_coast.geojson")
    vg_lines.estimate_future_coastline(
        curr_vec_file=coast_geojson,
        curr_vec_lyr=None,
        curr_vec_date=datetime.date(2020, 1, 1),
        trans_vec_file=transects_geojson,
        trans_vec_lyr=None,
        out_vec_file=pred_geojson,
        out_vec_lyr=None,
        out_pred_date=datetime.date(2021, 1, 1),
        out_format="GeoJSON",
        use_pbar=False,
    )
    pred_gdf = gpd.read_file(pred_geojson)
    coords = list(pred_gdf.geometry.iloc[0].coords)
    assert len(coords) == 2
    assert coords[0][1] > 10.9 and coords[1][1] > 11.9


def test_vectorgeoms_overlay_and_polygons(tmp_path):
    """Test overlay operations, clipping, and all polygon/geometry transformations."""
    a_gpkg = str(tmp_path / "a.gpkg")
    b_gpkg = str(tmp_path / "b.gpkg")
    gpd.GeoDataFrame(
        {"cls": ["A", "B"]},
        geometry=[box(0.0, 0.0, 10.0, 10.0), box(10.0, 0.0, 20.0, 10.0)],
        crs="EPSG:32630",
    ).to_file(a_gpkg, layer="a", driver="GPKG")
    gpd.GeoDataFrame(
        {"roi": ["R1"]},
        geometry=[box(5.0, 0.0, 15.0, 10.0)],
        crs="EPSG:32630",
    ).to_file(b_gpkg, layer="b", driver="GPKG")

    inter_gpkg = str(tmp_path / "inter.gpkg")
    vg_overlay.vec_lyr_intersection_gp(a_gpkg, "a", b_gpkg, "b", inter_gpkg, "inter")
    assert gpd.read_file(inter_gpkg, layer="inter").area.sum() == pytest.approx(100.0)

    diff_gpkg = str(tmp_path / "diff.gpkg")
    vg_overlay.vec_lyr_difference_gp(a_gpkg, "a", b_gpkg, "b", diff_gpkg, "diff")
    assert gpd.read_file(diff_gpkg, layer="diff").area.sum() == pytest.approx(100.0)

    sym_gpkg = str(tmp_path / "sym.gpkg")
    vg_overlay.vec_lyr_sym_difference_gp(a_gpkg, "a", b_gpkg, "b", sym_gpkg, "sym")
    assert gpd.read_file(sym_gpkg, layer="sym").area.sum() == pytest.approx(100.0)

    ident_gpkg = str(tmp_path / "ident.gpkg")
    vg_overlay.vec_lyr_identity_gp(a_gpkg, "a", b_gpkg, "b", ident_gpkg, "ident")
    assert gpd.read_file(ident_gpkg, layer="ident").area.sum() == pytest.approx(200.0)

    union_gpkg = str(tmp_path / "union.gpkg")
    vg_overlay.vec_lyr_union_gp(a_gpkg, "a", b_gpkg, "b", union_gpkg, "union")
    assert gpd.read_file(union_gpkg, layer="union").area.sum() == pytest.approx(200.0)

    clip_gpkg = str(tmp_path / "clip.gpkg")
    vg_overlay.clip_vec_lyr(a_gpkg, "a", b_gpkg, "b", clip_gpkg, "clip")
    assert gpd.read_file(clip_gpkg, layer="clip").area.sum() == pytest.approx(100.0)

    merge_gpkg = str(tmp_path / "merge.gpkg")
    vg_overlay.clip_and_merge_with_roi(
        a_gpkg,
        "a",
        b_gpkg,
        "b",
        merge_gpkg,
        "merge",
        dissolve_lyr=True,
        explode_rslt_lyr=True,
    )
    merge_gdf = gpd.read_file(merge_gpkg, layer="merge")
    assert merge_gdf.area.sum() == pytest.approx(100.0)

    sets_dir = str(tmp_path / "sets")
    vg_overlay.clip_vec_into_sets(
        a_gpkg,
        "a",
        b_gpkg,
        "b",
        "roi",
        sets_dir,
        out_vec_base_pre="sub",
        out_name_lower=True,
        use_pbar=False,
    )
    assert os.path.exists(os.path.join(sets_dir, "sub_r1.gpkg"))

    sets_pq_dir = str(tmp_path / "sets_pq")
    vg_overlay.clip_vec_into_sets(
        a_gpkg,
        "a",
        b_gpkg,
        "b",
        "roi",
        sets_pq_dir,
        out_vec_ext="parquet.sz",
        out_format="PARQUET",
        use_pbar=False,
    )
    assert os.path.exists(os.path.join(sets_pq_dir, "R1.parquet.sz"))

    # Polygon with hole + multi-polygon tests
    hole_poly = Polygon(
        [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
        holes=[[(2.0, 2.0), (4.0, 2.0), (4.0, 4.0), (2.0, 4.0)]],
    )
    hole_gpkg = str(tmp_path / "holes.gpkg")
    gpd.GeoDataFrame(
        {"cls": ["H"]}, geometry=[hole_poly], crs="EPSG:32630"
    ).to_file(hole_gpkg, layer="holes", driver="GPKG")

    ext_lines_gpkg = str(tmp_path / "ext_lines.gpkg")
    vg_polys.convert_polys_to_lines_gp(
        hole_gpkg, "holes", ext_lines_gpkg, "lines", exterior_lines=True
    )
    assert gpd.read_file(ext_lines_gpkg, layer="lines").length.iloc[0] == pytest.approx(40.0)

    bnd_lines_gpkg = str(tmp_path / "bnd_lines.gpkg")
    vg_polys.convert_polys_to_lines_gp(
        hole_gpkg, "holes", bnd_lines_gpkg, "lines", exterior_lines=False
    )
    assert gpd.read_file(bnd_lines_gpkg, layer="lines").length.iloc[0] == pytest.approx(48.0)

    assert vs_metrics.get_poly_hole_area(hole_gpkg, "holes", use_pbar=False) == pytest.approx([4.0])
    no_holes_gpkg = str(tmp_path / "no_holes.gpkg")
    vg_polys.delete_polygon_holes(
        hole_gpkg,
        "holes",
        "GPKG",
        no_holes_gpkg,
        "clean",
        area_thres=5.0,
        use_pbar=False,
    )
    assert vs_metrics.get_poly_hole_area(no_holes_gpkg, "clean", use_pbar=False) == []

    cent_gpkg = str(tmp_path / "centroids.gpkg")
    vg_polys.calc_poly_centroids(a_gpkg, "a", "GPKG", cent_gpkg, "centroids")
    cent_gdf = gpd.read_file(cent_gpkg, layer="centroids")
    assert cent_gdf.geometry.x.tolist() == pytest.approx([5.0, 15.0])

    multi_gpkg = str(tmp_path / "multi.gpkg")
    gpd.GeoDataFrame(
        {"id": [1]},
        geometry=[MultiPolygon([box(0, 0, 2, 2), box(4, 4, 6, 6)])],
        crs="EPSG:32630",
    ).to_file(multi_gpkg, layer="multi", driver="GPKG")

    exploded_gpkg = str(tmp_path / "exploded.gpkg")
    vg_polys.explode_vec_lyr(multi_gpkg, "multi", exploded_gpkg, "single")
    assert len(gpd.read_file(exploded_gpkg, layer="single")) == 2

    exp_dir = str(tmp_path / "exp_dir")
    vg_polys.explode_vec_files([multi_gpkg], exp_dir, use_pbar=False)
    assert len(gpd.read_file(os.path.join(exp_dir, "multi.gpkg"), layer="multi")) == 2

    simp_gpkg = str(tmp_path / "simple.gpkg")
    vg_polys.simplify_geometries(a_gpkg, "a", 0.5, "GPKG", simp_gpkg, "a")
    assert len(gpd.read_file(simp_gpkg, layer="a")) == 2

    buf_gpkg = str(tmp_path / "buf.gpkg")
    vg_polys.buffer_vec_layer_gp(a_gpkg, "a", 1.0, buf_gpkg, "buf")
    assert gpd.read_file(buf_gpkg, layer="buf").area.iloc[0] > 100.0

    grid_split_gpkg = str(tmp_path / "grid_split.gpkg")
    vg_polys.split_vec_by_grid(a_gpkg, "a", 5.0, 5.0, grid_split_gpkg, "split")
    assert len(gpd.read_file(grid_split_gpkg, layer="split")) == 8

    small_big_gpkg = str(tmp_path / "sb.gpkg")
    gpd.GeoDataFrame(
        {"id": [1, 2]},
        geometry=[box(0, 0, 2, 2), box(10, 10, 20, 20)],
        crs="EPSG:32630",
    ).to_file(small_big_gpkg, layer="sb", driver="GPKG")
    rm_gpkg = str(tmp_path / "rm.gpkg")
    vg_polys.rm_polys_area(small_big_gpkg, "sb", 10.0, rm_gpkg, "rm", less_than=False)
    assert len(gpd.read_file(rm_gpkg, layer="rm")) == 1

    bbox_gpkg = str(tmp_path / "bbox.gpkg")
    vg_polys.create_bbox_vec_lyr(
        a_gpkg, "a", bbox_gpkg, "bbox", whole_num_grid=True, whole_num_base_grid=10.0
    )
    assert gpd.read_file(bbox_gpkg, layer="bbox").area.iloc[0] == pytest.approx(200.0)

    dis_in_gpkg = str(tmp_path / "dis_in.gpkg")
    gpd.GeoDataFrame(
        {"cls": ["X", "X", "Y"]},
        geometry=[box(0, 0, 5, 5), box(5, 0, 10, 5), box(20, 0, 25, 5)],
        crs="EPSG:32630",
    ).to_file(dis_in_gpkg, layer="dis", driver="GPKG")
    dis_out_gpkg = str(tmp_path / "dis_out.gpkg")
    vg_polys.vec_lyr_dissolve(dis_in_gpkg, "dis", "cls", dis_out_gpkg, "dis")
    assert len(gpd.read_file(dis_out_gpkg, layer="dis")) == 2

    pts_thin_in = str(tmp_path / "pts_thin_in.gpkg")
    gpd.GeoDataFrame(
        {"id": [1, 2, 3]},
        geometry=[Point(1.0, 1.0), Point(2.0, 2.0), Point(12.0, 2.0)],
        crs="EPSG:32630",
    ).to_file(pts_thin_in, layer="pts", driver="GPKG")
    pts_thin_out = str(tmp_path / "pts_thin_out.gpkg")
    vg_polys.thin_vector_geoms(
        pts_thin_in, "pts", a_gpkg, "a", pts_thin_out, "pts"
    )
    assert len(gpd.read_file(pts_thin_out, layer="pts")) == 2

    shift_gpkg = str(tmp_path / "shift.gpkg")
    vg_polys.shiftxy_vec_lyr(a_gpkg, "a", 10.0, -5.0, shift_gpkg, "shift")
    shift_bounds = gpd.read_file(shift_gpkg, layer="shift").total_bounds
    assert tuple(shift_bounds) == pytest.approx((10.0, -5.0, 30.0, 5.0))


def test_vectorgeoms_spatial_and_deduplication(tmp_path):
    """Test point extraction, alpha shape, bboxes, rtree, predicates, and deduplication."""
    # Confirm OGR duplicate functions were removed and package __init__ has no __all__
    assert not hasattr(rsgispy.vectorgeoms, "__all__")
    assert not hasattr(rsgispy.vectorgeoms, "clip_vec_lyr")
    for mod, removed_name in (
        (vg_polys, "convert_polygon_to_polyline"),
        (vg_polys, "convert_multi_geoms_to_single"),
        (vg_overlay, "vec_lyr_intersection"),
        (vg_overlay, "vec_lyr_difference"),
        (vg_overlay, "clip_vec_lyr_ogr"),
    ):
        assert not hasattr(mod, removed_name)

    pts = vg_spatial.get_geom_pts(box(0.0, 0.0, 10.0, 10.0))
    assert len(pts) == 5

    pts_gpkg = str(tmp_path / "hull_pts.gpkg")
    gpd.GeoDataFrame(
        {"id": range(5)},
        geometry=[
            Point(0.0, 0.0),
            Point(10.0, 0.0),
            Point(10.0, 10.0),
            Point(0.0, 10.0),
            Point(5.0, 5.0),
        ],
        crs="EPSG:32630",
    ).to_file(pts_gpkg, layer="pts", driver="GPKG")

    lyr_pts = vg_spatial.get_vec_lyr_as_pts(pts_gpkg, "pts", use_pbar=False)
    assert len(lyr_pts) == 5

    hull_gpkg = str(tmp_path / "hull.gpkg")
    created, used_alpha = vg_spatial.create_alpha_shape(
        pts_gpkg, "pts", hull_gpkg, "hull", alpha_vals=[0.5, 1.0]
    )
    assert created is True
    assert used_alpha == 0.5
    assert gpd.read_file(hull_gpkg, layer="hull").area.iloc[0] > 0.0

    with pytest.raises(rsgispy.RSGISPyException):
        vg_spatial.create_alpha_shape(
            pts_gpkg, "pts", hull_gpkg, "hull", alpha_val=0.5, del_exist_vec=False
        )

    base_gpkg = str(tmp_path / "base.gpkg")
    gpd.GeoDataFrame(
        {"id": [1]}, geometry=[box(0.0, 0.0, 10.0, 10.0)], crs="EPSG:32630"
    ).to_file(base_gpkg, layer="base", driver="GPKG")

    assert vg_spatial.get_geoms_as_bboxs(base_gpkg, "base") == [(0.0, 10.0, 0.0, 10.0)]
    assert vg_spatial.bbox_intersects_vec_lyr(base_gpkg, "base", (5.0, 15.0, 5.0, 15.0)) is True
    assert vg_spatial.bbox_intersects_vec_lyr(base_gpkg, "base", (20.0, 30.0, 20.0, 30.0)) is False

    rt_idx, geoms = vg_spatial.create_rtree_index(base_gpkg, "base", use_pbar=False)
    assert vg_spatial.bbox_intersects_index(rt_idx, geoms, (5.0, 15.0, 5.0, 15.0)) is True
    assert vg_spatial.bbox_intersects_index(rt_idx, geoms, (20.0, 30.0, 20.0, 30.0)) is False

    comp_in_gpkg = str(tmp_path / "comp_in.gpkg")
    gpd.GeoDataFrame(
        {"id": [1, 2]},
        geometry=[box(1.0, 1.0, 4.0, 4.0), box(5.0, 5.0, 8.0, 8.0)],
        crs="EPSG:32630",
    ).to_file(comp_in_gpkg, layer="comp", driver="GPKG")

    comp_partial_gpkg = str(tmp_path / "comp_partial.gpkg")
    gpd.GeoDataFrame(
        {"id": [1, 2]},
        geometry=[box(1.0, 1.0, 4.0, 4.0), box(20.0, 20.0, 25.0, 25.0)],
        crs="EPSG:32630",
    ).to_file(comp_partial_gpkg, layer="comp", driver="GPKG")

    comp_overlap_gpkg = str(tmp_path / "comp_overlap.gpkg")
    gpd.GeoDataFrame(
        {"id": [1]}, geometry=[box(5.0, 5.0, 15.0, 15.0)], crs="EPSG:32630"
    ).to_file(comp_overlap_gpkg, layer="comp", driver="GPKG")

    comp_touch_gpkg = str(tmp_path / "comp_touch.gpkg")
    gpd.GeoDataFrame(
        {"id": [1]}, geometry=[box(10.0, 0.0, 20.0, 10.0)], crs="EPSG:32630"
    ).to_file(comp_touch_gpkg, layer="comp", driver="GPKG")

    comp_line_gpkg = str(tmp_path / "comp_line.gpkg")
    gpd.GeoDataFrame(
        {"id": [1]}, geometry=[LineString([(-5.0, 5.0), (15.0, 5.0)])], crs="EPSG:32630"
    ).to_file(comp_line_gpkg, layer="comp", driver="GPKG")

    assert vg_spatial.vec_intersects_vec(
        base_gpkg, "base", comp_overlap_gpkg, "comp", use_pbar=False
    ) is True
    assert vg_spatial.vec_overlaps_vec(
        base_gpkg, "base", comp_overlap_gpkg, "comp", use_pbar=False
    ) is True
    assert vg_spatial.vec_within_vec(
        base_gpkg, "base", comp_in_gpkg, "comp", use_pbar=False
    ) is True
    assert vg_spatial.vec_within_vec(
        base_gpkg, "base", comp_partial_gpkg, "comp", use_pbar=False
    ) is False
    assert vg_spatial.vec_contains_vec(
        base_gpkg, "base", comp_in_gpkg, "comp", use_pbar=False
    ) is True
    assert vg_spatial.vec_contains_vec(
        base_gpkg, "base", comp_partial_gpkg, "comp", use_pbar=False
    ) is False
    assert vg_spatial.vec_touches_vec(
        base_gpkg, "base", comp_touch_gpkg, "comp", use_pbar=False
    ) is True
    assert vg_spatial.vec_crosses_vec(
        base_gpkg, "base", comp_line_gpkg, "comp", use_pbar=False
    ) is True


def _write_test_raster(
    path: str,
    arr: np.ndarray,
    epsg: int = 32630,
    tl_x: float = 500000.0,
    tl_y: float = 5600100.0,
    res: float = 10.0,
    nodata=None,
) -> str:
    """Helper to write a single-band GeoTIFF for vectorisation tests."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    height, width = arr.shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=arr.dtype,
        crs=f"EPSG:{epsg}",
        transform=from_origin(tl_x, tl_y, res, res),
        nodata=nodata,
    ) as dst:
        dst.write(arr, 1)
    return path


def test_vectorgeoms_create(tmp_path):
    """Test polygonisation, point vectorisation, footprints, grids, H3, lines, transects."""
    cls_tif = str(tmp_path / "cls.tif")
    arr = np.zeros((10, 10), dtype=np.uint8)
    arr[2:5, 2:5] = 1
    arr[6:9, 6:9] = 2
    _write_test_raster(
        cls_tif, arr, epsg=32630, tl_x=500000.0, tl_y=5600100.0, res=10.0, nodata=0
    )

    poly_gpkg = str(tmp_path / "poly_from_ras.gpkg")
    vg_create.polygonise_raster_to_vec_lyr(
        poly_gpkg, "cls", "GPKG", cls_tif, mask_img=cls_tif
    )
    p_gdf = gpd.read_file(poly_gpkg, layer="cls")
    assert set(p_gdf["PXLVAL"]) == {1, 2}

    pxl_pts = str(tmp_path / "pxl_pts.gpkg")
    vg_create.vectorise_pxls_to_pts(cls_tif, 1, 1, pxl_pts, "ones", use_pbar=False)
    assert vu_meta.get_vec_feat_count(pxl_pts, "ones") == 9

    fp_gpkg = str(tmp_path / "footprint.gpkg")
    vg_create.extract_img_footprint(
        cls_tif, fp_gpkg, "fp", reproj_to=4326, use_pbar=False
    )
    fp_gdf = gpd.read_file(fp_gpkg, layer="fp")
    assert vu_meta.get_epsg_proj_from_vec(fp_gpkg, "fp") == 4326
    assert fp_gdf["FileName"].tolist() == ["cls.tif", "cls.tif"]

    csv_path = str(tmp_path / "bboxes.csv")
    with open(csv_path, "w", encoding="utf-8") as handle:
        handle.write("minx,maxx,miny,maxy\n0,10,0,10\n10,20,10,20\n")
    csv_gpkg = str(tmp_path / "csv_boxes.gpkg")
    vg_create.create_poly_vec_for_lst_bboxs(
        csv_path, csv_gpkg, "boxes", "GPKG", 32630, ignore_rows=1
    )
    assert vu_meta.get_vec_feat_count(csv_gpkg, "boxes") == 2

    grid_gpkg = str(tmp_path / "grid.gpkg")
    vg_create.define_grid(
        (500000.0, 500200.0, 5600000.0, 5600200.0),
        100.0,
        100.0,
        32630,
        grid_gpkg,
        "grid",
    )
    assert vu_meta.get_vec_feat_count(grid_gpkg, "grid") == 4

    utm_multi_gpkg = str(tmp_path / "utm_multi.gpkg")
    vg_create.define_grid(
        (-0.5, 0.5, 52.0, 52.5),
        20000.0,
        20000.0,
        4326,
        utm_multi_gpkg,
        "tiles",
        utm_grid=True,
    )
    assert set(vu_meta.get_vec_lyrs_lst(utm_multi_gpkg)) == {
        "tiles_utm30n",
        "tiles_utm31n",
    }

    pts_gpkg = str(tmp_path / "write_pts.gpkg")
    vg_create.write_pts_to_vec(
        pts_gpkg,
        "pts",
        "GPKG",
        32630,
        [500050.0, 500150.0],
        [5600050.0, 5600150.0],
        atts={"lbl": ["a", "b"]},
        att_types={"names": ["lbl"], "types": [str]},
    )
    assert vu_meta.get_vec_feat_count(pts_gpkg, "pts") == 2

    pt_boxes = str(tmp_path / "pt_boxes.gpkg")
    vg_create.create_bboxs_for_pts(
        pts_gpkg, "pts", 20.0, 10.0, pt_boxes, "boxes", use_pbar=False
    )
    pb_gdf = gpd.read_file(pt_boxes, layer="boxes")
    assert pb_gdf.area.tolist() == pytest.approx([200.0, 200.0])

    wgs_grid = str(tmp_path / "wgs_grid.gpkg")
    vg_create.create_wgs84_vec_grid(
        wgs_grid, "wgrid", "GPKG", 1.0, 1.0, (-5.0, -3.0, 51.0, 53.0), overlap=0.1
    )
    wg_gdf = gpd.read_file(wgs_grid, layer="wgrid")
    assert len(wg_gdf) == 4
    assert "n0530w0050" in wg_gdf["tile_names"].tolist()

    img_vecs = vg_create.create_vec_for_imgs(
        [cls_tif],
        str(tmp_path / "img_vecs"),
        out_format="GeoJSON",
        geometry_type=rsgispy.GEOM_POLY,
        out_name_replace={"cls": "scene"},
        use_pbar=False,
    )
    assert len(img_vecs) == 1
    assert os.path.basename(img_vecs[0]) == "scene.geojson"

    hex_bbox_gpkg = str(tmp_path / "hex_bbox.gpkg")
    vg_create.create_hex_grid_bbox(
        (-4.2, -3.9, 52.3, 52.5), 4326, 6, hex_bbox_gpkg, "hex"
    )
    assert vu_meta.get_vec_feat_count(hex_bbox_gpkg, "hex") > 0

    roi_wgs_gpkg = str(tmp_path / "roi_wgs.gpkg")
    roi_poly_3d = Polygon([
        (-4.2, 52.3, 0.0),
        (-3.9, 52.3, 0.0),
        (-3.9, 52.5, 0.0),
        (-4.2, 52.5, 0.0),
    ])
    gpd.GeoDataFrame(
        {"id": [1]},
        geometry=[roi_poly_3d],
        crs="EPSG:4326",
    ).to_file(roi_wgs_gpkg, layer="roi", driver="GPKG")
    hex_polys_gpkg = str(tmp_path / "hex_polys.gpkg")
    vg_create.create_hex_grid_polys(
        roi_wgs_gpkg, "roi", 6, hex_polys_gpkg, "hex", use_pbar=False
    )
    assert vu_meta.get_vec_feat_count(hex_polys_gpkg, "hex") > 0

    rad_pts = str(tmp_path / "rad_pts.gpkg")
    vg_create.create_random_pts_in_radius(
        500000.0, 5600000.0, 100.0, 15, 32630, rad_pts, "rad", rnd_seed=42
    )
    rad_gdf = gpd.read_file(rad_pts, layer="rad")
    assert len(rad_gdf) == 15
    assert (rad_gdf["dist"] <= 100.0).all()
    assert ((rad_gdf["angle"] >= 0.0) & (rad_gdf["angle"] <= 360.0)).all()

    bbox_pts = str(tmp_path / "bbox_pts.gpkg")
    vg_create.create_random_pts_in_bbox(
        (500000.0, 500100.0, 5600000.0, 5600100.0),
        12,
        32630,
        bbox_pts,
        "pts",
        rnd_seed=42,
    )
    assert vu_meta.get_vec_feat_count(bbox_pts, "pts") == 12

    lines_out = str(tmp_path / "created_lines.gpkg")
    vg_lines.create_lines_vec(
        lines_out,
        "lines",
        "GPKG",
        32630,
        [[(0.0, 0.0), (10.0, 10.0)], [(10.0, 0.0), (20.0, 10.0)]],
    )
    assert vu_meta.get_vec_feat_count(lines_out, "lines") == 2

    transects_gpkg = str(tmp_path / "transects.gpkg")
    vg_lines.create_img_transects(
        cls_tif,
        transects_gpkg,
        "trans",
        x_intervals=[0.25, 0.75],
        y_intervals=[0.5],
    )
    assert vu_meta.get_vec_feat_count(transects_gpkg, "trans") == 3
