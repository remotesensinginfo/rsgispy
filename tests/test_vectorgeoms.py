"""
Tests for rsgispy.vectorgeoms (lines, overlay, polygons, spatial) using
geopandas without osgeo dependencies.
"""

import datetime
import importlib
import os

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPolygon,
    Point,
    Polygon,
    box,
)

import rsgispy
import rsgispy.vector
import rsgispy.vector.create as vg_create
import rsgispy.vector.geoms as vg_lines
import rsgispy.vector.geoms as vg_overlay
import rsgispy.vector.geoms as vg_polys
import rsgispy.vector.geoms as vg_spatial
import rsgispy.vector.io as vu_meta
import rsgispy.vector.stats as vs_metrics
from rsgispy import RSGISPyException
from rsgispy.vector import selection as vec_sel


def _can_import(mod_name: str) -> bool:
    try:
        importlib.import_module(mod_name)
        return True
    except ImportError:
        return False


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


@pytest.fixture
def overlay_layers(tmp_path):
    """Two adjacent 10 x 10 squares (layer 'a') and an ROI over their shared edge (layer 'b')."""
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
    return a_gpkg, b_gpkg


def test_vectorgeoms_overlay_operations(tmp_path, overlay_layers):
    """Test intersection, difference, symmetric difference, identity, union and clipping."""
    a_gpkg, b_gpkg = overlay_layers

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


def test_vectorgeoms_clip_vec_into_sets(tmp_path, overlay_layers):
    """Test clip_vec_into_sets to GPKG and (when pyarrow is available) Parquet."""
    a_gpkg, b_gpkg = overlay_layers

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

    if _can_import("pyarrow.parquet"):
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


def test_vectorgeoms_polygon_holes_and_lines(tmp_path):
    """Test polygon-to-line conversion, hole areas and hole deletion."""
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


def test_vectorgeoms_polygon_transformations(tmp_path, overlay_layers):
    """Test centroids, explode, simplify, buffer, grid split, area filter, bbox,
    dissolve, thinning and shifting."""
    a_gpkg, _ = overlay_layers

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
    vg_create.create_bbox_vec_lyr(
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
    assert rsgispy.vector.__all__ == ["io", "create", "geoms", "attrs", "stats", "selection"]
    assert not hasattr(rsgispy.vector, "clip_vec_lyr")
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

    if _can_import("alphashape"):
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

    if _can_import("rtree"):
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


@pytest.fixture
def cls_tif(tmp_path):
    """A 10 x 10 thematic raster with two 3 x 3 classes (1 and 2) and nodata 0."""
    cls_tif = str(tmp_path / "cls.tif")
    arr = np.zeros((10, 10), dtype=np.uint8)
    arr[2:5, 2:5] = 1
    arr[6:9, 6:9] = 2
    _write_test_raster(
        cls_tif, arr, epsg=32630, tl_x=500000.0, tl_y=5600100.0, res=10.0, nodata=0
    )
    return cls_tif


def test_vectorgeoms_create_from_raster(tmp_path, cls_tif):
    """Test polygonisation, point vectorisation, footprints, image vectors and transects."""
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

    transects_gpkg = str(tmp_path / "transects.gpkg")
    vg_create.create_img_transects(
        cls_tif,
        transects_gpkg,
        "trans",
        x_intervals=[0.25, 0.75],
        y_intervals=[0.5],
    )
    assert vu_meta.get_vec_feat_count(transects_gpkg, "trans") == 3


def test_vectorgeoms_create_grids_and_bboxes(tmp_path):
    """Test bbox lists from CSV, regular / UTM / WGS84 grids and H3 hexagon grids."""
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

    wgs_grid = str(tmp_path / "wgs_grid.gpkg")
    vg_create.create_wgs84_vec_grid(
        wgs_grid, "wgrid", "GPKG", 1.0, 1.0, (-5.0, -3.0, 51.0, 53.0), overlap=0.1
    )
    wg_gdf = gpd.read_file(wgs_grid, layer="wgrid")
    assert len(wg_gdf) == 4
    assert "n0530w0050" in wg_gdf["tile_names"].tolist()

    if _can_import("h3"):
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


def test_vectorgeoms_create_points_and_lines(tmp_path):
    """Test writing points, point bboxes, random points and lines."""
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
    vg_create.create_lines_vec(
        lines_out,
        "lines",
        "GPKG",
        32630,
        [[(0.0, 0.0), (10.0, 10.0)], [(10.0, 0.0), (20.0, 10.0)]],
    )
    assert vu_meta.get_vec_feat_count(lines_out, "lines") == 2


@pytest.fixture
def overlay_gdfs():
    """In-memory versions of the overlay test layers: two squares and an ROI."""
    a_gdf = gpd.GeoDataFrame(
        {"cls": ["A", "B"]},
        geometry=[box(0.0, 0.0, 10.0, 10.0), box(10.0, 0.0, 20.0, 10.0)],
        crs="EPSG:32630",
    )
    b_gdf = gpd.GeoDataFrame(
        {"roi": ["R1"]},
        geometry=[box(5.0, 0.0, 15.0, 10.0)],
        crs="EPSG:32630",
    )
    return a_gdf, b_gdf


@pytest.fixture
def line_100m_gdf():
    """A single 100 m east-west line."""
    return gpd.GeoDataFrame(
        {"id": [1]}, geometry=[LineString([(0.0, 0.0), (100.0, 0.0)])], crs="EPSG:32630"
    )


def test_vectorgeoms_spatial_overlay_polygons_gdf_in_memory(overlay_gdfs):
    """Test GeoDataFrame inputs and in-memory outputs of the spatial, overlay and
    polygon functions (and the alpha shape when alphashape is installed)."""
    a_gdf, b_gdf = overlay_gdfs

    # Spatial inspection
    assert vg_spatial.get_geoms_as_bboxs(b_gdf, None) == [(5.0, 15.0, 0.0, 10.0)]
    assert vg_spatial.bbox_intersects_vec_lyr(b_gdf, None, (0.0, 8.0, 0.0, 8.0)) is True
    assert vg_spatial.vec_intersects_vec(a_gdf, None, b_gdf, None, use_pbar=False) is True

    # Overlay
    inter_gdf = vg_overlay.vec_lyr_intersection_gp(a_gdf, None, b_gdf, None)
    assert isinstance(inter_gdf, gpd.GeoDataFrame)
    assert inter_gdf.area.sum() == pytest.approx(100.0)

    clip_gdf = vg_overlay.clip_vec_lyr(a_gdf, None, b_gdf, None)
    assert clip_gdf.area.sum() == pytest.approx(100.0)

    merge_gdf = vg_overlay.clip_and_merge_with_roi(
        a_gdf, None, b_gdf, None, dissolve_lyr=True, explode_rslt_lyr=True
    )
    assert merge_gdf.area.sum() == pytest.approx(100.0)

    # Polygons
    cent_gdf = vg_polys.calc_poly_centroids(a_gdf, None)
    assert cent_gdf.geometry.x.tolist() == pytest.approx([5.0, 15.0])

    buf_gdf = vg_polys.buffer_vec_layer_gp(a_gdf, None, 1.0)
    assert buf_gdf.area.iloc[0] > 100.0

    shift_gdf = vg_polys.shiftxy_vec_lyr(a_gdf, None, 10.0, -5.0)
    assert tuple(shift_gdf.total_bounds) == pytest.approx((10.0, -5.0, 30.0, 5.0))

    dis_gdf = vg_polys.vec_lyr_dissolve(
        gpd.GeoDataFrame(
            {"cls": ["X", "X", "Y"]},
            geometry=[box(0, 0, 5, 5), box(5, 0, 10, 5), box(20, 0, 25, 5)],
            crs="EPSG:32630",
        ),
        None,
        "cls",
    )
    assert len(dis_gdf) == 2

    if _can_import("alphashape"):
        hull_pts = gpd.GeoDataFrame(
            {"id": range(5)},
            geometry=[
                Point(0.0, 0.0),
                Point(10.0, 0.0),
                Point(10.0, 10.0),
                Point(0.0, 10.0),
                Point(5.0, 5.0),
            ],
            crs="EPSG:32630",
        )
        hull_gdf, used_alpha = vg_spatial.create_alpha_shape(
            hull_pts, None, alpha_vals=[0.5, 1.0]
        )
        assert isinstance(hull_gdf, gpd.GeoDataFrame)
        assert used_alpha == 0.5
        assert hull_gdf.area.iloc[0] > 0.0


def test_vectorgeoms_lines_gdf_in_memory(line_100m_gdf):
    """Test GeoDataFrame inputs and in-memory outputs of the line functions."""
    line_gdf = line_100m_gdf
    orthg_gdf = vg_lines.create_orthg_lines(
        line_gdf, None, pt_step=25.0, line_len=10.0, use_pbar=False
    )
    assert len(orthg_gdf) == 3

    pts_gdf = vg_lines.split_lines_to_reg_pts(
        line_gdf, None, pt_step=25.0, use_pbar=False
    )
    assert len(pts_gdf) == 3

    ray_gdf = gpd.GeoDataFrame(
        {"uid": [1], "start_x": [0.0], "start_y": [5.0]},
        geometry=[LineString([(0.0, 5.0), (40.0, 5.0)])],
        crs="EPSG:32630",
    )
    wall_gdf = gpd.GeoDataFrame(
        {"id": [1]}, geometry=[box(10.0, 0.0, 30.0, 10.0)], crs="EPSG:32630"
    )
    closest_gdf = vg_lines.closest_line_intersection(
        ray_gdf, None, wall_gdf, None, use_pbar=False
    )
    assert closest_gdf["len"].iloc[0] == pytest.approx(10.0)

    track_gdf = vg_lines.pts_to_line_geoms(
        gpd.GeoDataFrame(
            {"seq": [2, 1]},
            geometry=[Point(10.0, 0.0), Point(0.0, 0.0)],
            crs="EPSG:32630",
        ),
        None,
        sort_col="seq",
    )
    assert list(track_gdf.geometry.iloc[0].coords) == [(0.0, 0.0), (10.0, 0.0)]

    rays_gdf = vg_lines.create_angle_lines_from_points(
        pts_gdf, None, angle=90.0, line_len=5.0, use_pbar=False
    )
    assert len(rays_gdf) == 12

    coast_gdf = gpd.GeoDataFrame(
        {"id": [1]}, geometry=[LineString([(0.0, 10.0), (20.0, 10.0)])], crs="EPSG:32630"
    )
    trans_gdf = gpd.GeoDataFrame(
        {"LRR": [1.0, 2.0]},
        geometry=[
            LineString([(5.0, 0.0), (5.0, 20.0)]),
            LineString([(15.0, 0.0), (15.0, 20.0)]),
        ],
        crs="EPSG:32630",
    )
    pred_gdf = vg_lines.estimate_future_coastline(
        coast_gdf,
        None,
        datetime.date(2020, 1, 1),
        trans_gdf,
        None,
        None,
        None,
        datetime.date(2021, 1, 1),
        use_pbar=False,
    )
    assert isinstance(pred_gdf, gpd.GeoDataFrame)
    assert len(list(pred_gdf.geometry.iloc[0].coords)) == 2


def test_vectorgeoms_create_gdf_in_memory(line_100m_gdf):
    """Test in-memory GeoDataFrame outputs of the vector creation functions."""
    pts_gdf = vg_lines.split_lines_to_reg_pts(
        line_100m_gdf, None, pt_step=25.0, use_pbar=False
    )
    pt_boxes_gdf = vg_create.create_bboxs_for_pts(
        pts_gdf, None, 20.0, 10.0, use_pbar=False
    )
    assert pt_boxes_gdf.area.tolist() == pytest.approx([200.0, 200.0, 200.0])

    bbox_grid_gdf = vg_create.define_grid(
        (500000.0, 500200.0, 5600000.0, 5600200.0), 100.0, 100.0, 32630
    )
    assert isinstance(bbox_grid_gdf, gpd.GeoDataFrame)
    assert len(bbox_grid_gdf) == 4

    wgs_grid_gdf = vg_create.create_wgs84_vec_grid(
        None, None, "GPKG", 1.0, 1.0, (-5.0, -3.0, 51.0, 53.0), overlap=0.1
    )
    assert len(wgs_grid_gdf) == 4

    written_pts_gdf = vg_create.write_pts_to_vec(
        None, None, "GPKG", 32630, [500050.0, 500150.0], [5600050.0, 5600150.0],
        atts={"lbl": ["a", "b"]}, att_types={"names": ["lbl"], "types": [str]},
    )
    assert len(written_pts_gdf) == 2

    rad_pts_gdf = vg_create.create_random_pts_in_radius(
        500000.0, 5600000.0, 100.0, 10, 32630, rnd_seed=42
    )
    assert len(rad_pts_gdf) == 10

    bbox_pts_gdf = vg_create.create_random_pts_in_bbox(
        (500000.0, 500100.0, 5600000.0, 5600100.0), 8, 32630, rnd_seed=42
    )
    assert len(bbox_pts_gdf) == 8

    created_lines_gdf = vg_create.create_lines_vec(
        None, None, "GPKG", 32630,
        [[(0.0, 0.0), (10.0, 10.0)], [(10.0, 0.0), (20.0, 10.0)]],
    )
    assert len(created_lines_gdf) == 2

    if _can_import("h3"):
        roi_wgs = gpd.GeoDataFrame(
            {"id": [1]},
            geometry=[box(-4.2, 52.3, -3.9, 52.5)],
            crs="EPSG:4326",
        )
        hex_gdf = vg_create.create_hex_grid_polys(roi_wgs, None, 6, use_pbar=False)
        assert isinstance(hex_gdf, gpd.GeoDataFrame)
        assert len(hex_gdf) > 0
        hex_bbox_gdf = vg_create.create_hex_grid_bbox(
            (-4.2, -3.9, 52.3, 52.5), 4326, 6
        )
        assert isinstance(hex_bbox_gdf, gpd.GeoDataFrame)
        assert len(hex_bbox_gdf) > 0


def test_vectorgeoms_raster_to_gdf_in_memory(tmp_path):
    """Test raster-to-vector functions returning in-memory GeoDataFrames."""
    cls_tif = str(tmp_path / "cls.tif")
    arr = np.zeros((10, 10), dtype=np.uint8)
    arr[2:5, 2:5] = 1
    arr[6:9, 6:9] = 2
    _write_test_raster(
        cls_tif, arr, epsg=32630, tl_x=500000.0, tl_y=5600100.0, res=10.0, nodata=0
    )

    poly_gdf = vg_create.polygonise_raster_to_vec_lyr(
        None, None, "GPKG", cls_tif, mask_img=cls_tif
    )
    assert isinstance(poly_gdf, gpd.GeoDataFrame)
    assert set(poly_gdf["PXLVAL"]) == {1, 2}

    pxl_pts_gdf = vg_create.vectorise_pxls_to_pts(cls_tif, 1, 1, use_pbar=False)
    assert isinstance(pxl_pts_gdf, gpd.GeoDataFrame)
    assert len(pxl_pts_gdf) == 9

    fp_gdf = vg_create.extract_img_footprint(cls_tif, reproj_to=4326, use_pbar=False)
    assert isinstance(fp_gdf, gpd.GeoDataFrame)
    assert fp_gdf.crs.to_epsg() == 4326

    img_gdfs = vg_create.create_vec_for_imgs(
        [cls_tif], output_dir=None, geometry_type=rsgispy.GEOM_POLY, use_pbar=False
    )
    assert len(img_gdfs) == 1
    assert isinstance(img_gdfs[0], gpd.GeoDataFrame)

    trans_gdf = vg_create.create_img_transects(
        cls_tif, x_intervals=[0.25, 0.75], y_intervals=[0.5]
    )
    assert isinstance(trans_gdf, gpd.GeoDataFrame)
    assert len(trans_gdf) == 3


def test_spatial_predicates_align_comparison_crs():
    """H4: the comparison layer is reprojected onto the base layer's CRS."""
    base_gdf = gpd.GeoDataFrame(
        {"id": [1]},
        geometry=[box(500000.0, 5800000.0, 510000.0, 5810000.0)],
        crs="EPSG:32630",
    )
    # Same small square inside the base polygon, expressed in WGS84.
    comp_utm = gpd.GeoDataFrame(
        {"id": [1]},
        geometry=[box(504000.0, 5804000.0, 506000.0, 5806000.0)],
        crs="EPSG:32630",
    )
    comp_wgs = comp_utm.to_crs("EPSG:4326")
    assert vg_spatial.vec_intersects_vec(
        base_gdf, None, comp_wgs, None, use_pbar=False
    )
    assert vg_spatial.vec_within_vec(base_gdf, None, comp_wgs, None, use_pbar=False)
    assert vg_spatial.vec_contains_vec(
        base_gdf, None, comp_wgs, None, use_pbar=False
    )
    # The base layer's CRS wins: a WGS84 base with a UTM comparison also works.
    assert vg_spatial.vec_intersects_vec(
        comp_wgs, None, base_gdf, None, use_pbar=False
    )


def test_empty_results_write_empty_layer_with_warning(tmp_path):
    """Empty results are written as an empty layer (same schema) with a warning."""
    import rsgispy.vector.attrs as va_spatial

    polys = gpd.GeoDataFrame(
        {"cls": ["A"]}, geometry=[box(0.0, 0.0, 10.0, 10.0)], crs="EPSG:32630"
    )
    far = gpd.GeoDataFrame(
        {"roi": ["R"]}, geometry=[box(100.0, 100.0, 110.0, 110.0)], crs="EPSG:32630"
    )
    no_pts = gpd.GeoDataFrame(
        {"pid": []}, geometry=gpd.GeoSeries([], crs="EPSG:32630"), crs="EPSG:32630"
    )
    calls = {
        "overlay": lambda out: vg_overlay.vec_lyr_intersection_gp(
            polys, None, far, None, out, "lyr"
        ),
        "explode": lambda out: vg_polys.explode_vec_lyr(polys.iloc[0:0], None, out, "lyr"),
        "rm_area": lambda out: vg_polys.rm_polys_area(polys, None, 1.0, out, "lyr"),
        "thin": lambda out: vg_polys.thin_vector_geoms(no_pts, None, polys, None, out, "lyr"),
        "sjoin": lambda out: va_spatial.perform_spatial_join(
            polys, None, far, None, out, "lyr", join_op="intersects"
        ),
    }
    for name, call in calls.items():
        out_path = str(tmp_path / f"{name}.gpkg")
        with pytest.warns(UserWarning, match="empty"):
            assert call(out_path) is None
        assert os.path.exists(out_path), name
        out_gdf = gpd.read_file(out_path, layer="lyr")
        assert len(out_gdf) == 0, name
        assert out_gdf.crs == polys.crs, name
    assert "cls" in gpd.read_file(str(tmp_path / "rm_area.gpkg"), layer="lyr").columns


def _write_raster(
    path: str,
    arr: np.ndarray,
    *,
    origin=(500000.0, 5200000.0),
    res=(10.0, 10.0),
    crs: str = "EPSG:32630",
    nodata=None,
    tiled: bool = False,
    blocksize: int | None = None,
) -> str:
    """Write a 2D or 3D NumPy array to a GeoTIFF."""
    if arr.ndim == 2:
        data = arr[np.newaxis, :, :]
    else:
        data = arr
    count, height, width = data.shape
    profile = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": count,
        "dtype": str(data.dtype),
        "crs": crs,
        "transform": from_origin(origin[0], origin[1], res[0], res[1]),
        "nodata": nodata,
    }
    if tiled:
        profile["tiled"] = True
        if blocksize is not None:
            profile["blockxsize"] = blocksize
            profile["blockysize"] = blocksize
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
    return path


def test_custom_geometry_column_name_across_vector_ops():
    """Functions respect gdf.geometry.name when geometry column is not 'geometry'."""
    poly_gdf = gpd.GeoDataFrame(
        {"grp": ["a", "b"], "val": [1, 2]},
        geometry=[box(0, 0, 10, 10), box(20, 0, 30, 10)],
        crs="EPSG:32630",
    ).rename_geometry("geom")

    buf = vg_lines.buffer_vec_layer_gp(poly_gdf, None, 2.0, out_vec_file=None)
    assert buf.geometry.name == "geom"
    assert all(buf.geometry.area > 100.0)

    parts = vu_meta.split_by_attribute(poly_gdf, None, "grp")
    assert set(parts.keys()) == {"a", "b"}
    assert parts["a"].geometry.name == "geom"

    roi_gdf = gpd.GeoDataFrame(
        {"rid": [1]}, geometry=[box(0, 0, 15, 15)], crs="EPSG:32630"
    ).rename_geometry("geom")
    subs = vec_sel.subset_vec_lyr_to_feat_bboxs(
        roi_gdf, None, poly_gdf, None, use_pbar=False
    )
    assert len(subs) == 1
    assert len(subs[0]) == 1
    assert subs[0].geometry.name == "geom"


def test_thin_vector_geoms_existing_unq_id_and_border_dedup():
    """thin_vector_geoms handles existing unq_id, cleans grid cols, dedups borders."""
    # Point right on shared grid boundary (10.0, 5.0) plus clustered points
    pts = [
        Point(10.0, 5.0),
        Point(5.0, 5.0),
        Point(6.0, 5.0),
        Point(15.0, 5.0),
    ]
    gdf = gpd.GeoDataFrame(
        {"unq_id": [100, 101, 102, 103], "name": ["b", "c1", "c2", "d"]},
        geometry=pts,
        crs="EPSG:32630",
    ).rename_geometry("geom")
    grid_gdf = gpd.GeoDataFrame(
        {"grid_extra": [1, 2]},
        geometry=[box(0.0, 0.0, 10.0, 10.0), box(10.0, 0.0, 20.0, 10.0)],
        crs="EPSG:32630",
    )

    thinned = vg_lines.thin_vector_geoms(
        gdf, None, grid_gdf, None, out_vec_file=None
    )
    assert "index_right" not in thinned.columns
    assert "grid_extra" not in thinned.columns
    assert set(thinned.columns) == {"unq_id", "name", "geom"}
    # Border point (unq_id=100) must appear at most once
    assert list(thinned["unq_id"]).count(100) <= 1
    assert thinned["unq_id"].is_unique


def test_lines_pt_step_validation_dist_and_future_coastline():
    """Validate pt_step > 0, accurate dist values, and robust estimate_future_coastline."""
    line_gdf = gpd.GeoDataFrame(
        {"lid": [1]},
        geometry=[LineString([(0.0, 0.0), (10.0, 0.0)])],
        crs="EPSG:32630",
    ).rename_geometry("geom")

    with pytest.raises(RSGISPyException):
        vg_lines.create_orthg_lines(line_gdf, None, None, pt_step=0.0)
    with pytest.raises(RSGISPyException):
        vg_lines.split_lines_to_reg_pts(line_gdf, None, None, pt_step=-5.0)

    pts_gdf = vg_lines.split_lines_to_reg_pts(
        line_gdf, None, out_vec_file=None, pt_step=2.5
    )
    assert list(pts_gdf["dist"]) == pytest.approx([2.5, 5.0, 7.5])
    assert list(pts_gdf["pt_x"]) == pytest.approx(list(pts_gdf["dist"]))

    # estimate_future_coastline with MultiLineString, empty and null geometries
    coast_gdf = gpd.GeoDataFrame(
        {"cid": [1, 2, 3]},
        geometry=[
            MultiLineString(
                [[(0.0, 0.0), (10.0, 0.0)], [(10.0, 0.0), (20.0, 0.0)]]
            ),
            LineString(),
            None,
        ],
        crs="EPSG:32630",
    )
    trans_gdf = gpd.GeoDataFrame(
        {"LRR": [1.0, 2.0]},
        geometry=[
            LineString([(5.0, -5.0), (5.0, 5.0)]),
            LineString([(15.0, -5.0), (15.0, 5.0)]),
        ],
        crs="EPSG:32630",
    )
    fut = vg_lines.estimate_future_coastline(
        coast_gdf,
        None,
        curr_vec_date=datetime.date(2020, 1, 1),
        trans_vec_file=trans_gdf,
        trans_vec_lyr=None,
        out_vec_file=None,
        out_vec_lyr=None,
        out_pred_date=datetime.date(2025, 1, 1),
        trans_rate_col="LRR",
        use_pbar=False,
    )
    assert fut is not None
    assert len(fut) == 1
    assert not fut.geometry.iloc[0].is_empty


def test_create_polygonise_replace_grid_validation_and_bbox_layers(tmp_path):
    """Test polygonise replace_lyr, define_grid validation, create_bbox_vec_lyr,
    create_poly_vec_bboxs, and create_hex_grid_poly.
    """
    tif = _write_raster(
        str(tmp_path / "cls.tif"),
        np.array([[1, 1, 2, 2], [1, 1, 2, 2]], dtype=np.uint8),
    )
    gpkg = str(tmp_path / "poly.gpkg")
    vg_create.polygonise_raster_to_vec_lyr(
        gpkg, "cls", "GPKG", tif, replace_file=True, replace_lyr=True
    )
    with pytest.raises(RSGISPyException):
        vg_create.polygonise_raster_to_vec_lyr(
            gpkg, "cls", "GPKG", tif, replace_file=False, replace_lyr=False
        )

    # define_grid multi-UTM rejects non-multi-layer format
    with pytest.raises(RSGISPyException):
        vg_create.define_grid(
            bbox=(-8.0, 2.0, 50.0, 52.0),
            x_size=100000.0,
            y_size=100000.0,
            in_epsg_code=4326,
            out_vec_file=str(tmp_path / "grid.geojson"),
            out_vec_lyr="grid",
            out_format="GeoJSON",
            utm_grid=True,
        )

    # create_bbox_vec_lyr raises cleanly on empty layer and succeeds on valid layer
    empty_gdf = gpd.GeoDataFrame(
        {"id": []}, geometry=gpd.GeoSeries([], crs="EPSG:4326"), crs="EPSG:4326"
    )
    with pytest.raises(RSGISPyException):
        vg_create.create_bbox_vec_lyr(empty_gdf, None, out_vec_file=None)

    pts_gdf = gpd.GeoDataFrame(
        {"id": [1, 2]},
        geometry=[Point(0, 0), Point(4, 2)],
        crs="EPSG:4326",
    )
    bbox_gdf = vg_create.create_bbox_vec_lyr(pts_gdf, None, out_vec_file=None)
    assert len(bbox_gdf) == 1
    assert bbox_gdf.total_bounds == pytest.approx([0.0, 0.0, 4.0, 2.0])

    # Direct test: create_poly_vec_bboxs
    bboxes_gdf = vg_create.create_poly_vec_bboxs(
        vec_file=None,
        vec_lyr=None,
        out_format="GPKG",
        epsg_code=32630,
        bboxs=[(0.0, 10.0, 0.0, 5.0), (10.0, 20.0, 0.0, 5.0)],
        atts={"tile": ["t1", "t2"]},
        att_types={"names": ["tile"], "types": [str]},
    )
    assert len(bboxes_gdf) == 2
    assert list(bboxes_gdf["tile"]) == ["t1", "t2"]
    assert bboxes_gdf.geometry.area.sum() == pytest.approx(100.0)

    # Direct test: create_hex_grid_poly (if h3 available, else check error)
    roi_poly = box(-4.1, 52.35, -4.0, 52.45)
    try:
        import h3  # noqa: F401  # pylint: disable=import-outside-toplevel,unused-import

        has_h3 = True
    except ImportError:
        has_h3 = False
    if has_h3:
        hex_gdf = vg_create.create_hex_grid_poly(roi_poly, hex_scale=6)
        assert hex_gdf is not None and len(hex_gdf) > 0
        assert "hex_id" in hex_gdf.columns
    else:
        with pytest.raises(RSGISPyException):
            vg_create.create_hex_grid_poly(roi_poly, hex_scale=6)
