"""
Tests for rsgispy.vectorutils (metadata, io, selection) using geopandas,
pyogrio, shapely, rasterio, and pyproj without osgeo dependencies.
"""

import os

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box

import rsgispy
import rsgispy.vectorprocess
import rsgispy.vector
import rsgispy.vector.io as vu_io
import rsgispy.vector.io as vu_meta
import rsgispy.vector.selection as vu_sel
import rsgispy.vector.stats as vs_metrics
from rsgispy import vectorprocess as vp
from rsgispy import RSGISPyException
from rsgispy.vector import io as vec_io
from rsgispy.vector import selection as vec_sel
import pandas as pd


def _write_test_raster(
    path: str,
    arr: np.ndarray,
    epsg: int = 32630,
    tl_x: float = 500000.0,
    tl_y: float = 5600100.0,
    res: float = 10.0,
    nodata=None,
) -> str:
    """Helper to write a single-band uint8/float32 GeoTIFF for tests."""
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


def test_vectorutils_metadata(tmp_path):
    """Test metadata, bbox, projections, layer listing, counts, and deletion."""
    info_obj = vu_meta.VecLayersInfoObj("a.gpkg", "lyr_in", "lyr_out")
    assert info_obj.vec_file == "a.gpkg"
    assert info_obj.vec_lyr == "lyr_in"
    assert info_obj.vec_out_lyr == "lyr_out"

    assert vu_meta.check_format_name("GEOJSON") == "GeoJSON"
    assert vu_meta.check_format_name("GPKG") == "GPKG"
    assert vu_meta.get_file_vec_extension("GPKG") == "gpkg"
    assert vu_meta.get_file_vec_extension("ESRI Shapefile") == "shp"
    with pytest.raises(rsgispy.RSGISPyException):
        vu_meta.get_file_vec_extension("UNKNOWN_FMT")

    gpkg_path = str(tmp_path / "polys.gpkg")
    gdf1 = gpd.GeoDataFrame(
        {"cls": ["A", "B", "A"], "val": [10, 20, 30]},
        geometry=[
            box(500000.0, 5600000.0, 500100.0, 5600100.0),
            box(500100.0, 5600000.0, 500200.0, 5600100.0),
            box(500200.0, 5600000.0, 500300.0, 5600100.0),
        ],
        crs="EPSG:32630",
    )
    gdf1.to_file(gpkg_path, layer="polys", driver="GPKG")
    gpd.GeoDataFrame(
        {"pt_id": [1]}, geometry=[Point(500050.0, 5600050.0)], crs="EPSG:32630"
    ).to_file(gpkg_path, layer="pts", driver="GPKG")

    assert vu_meta.get_vec_lyrs_lst(gpkg_path) == ["polys", "pts"]
    assert vu_meta.get_epsg_proj_from_vec(gpkg_path, "polys") == 32630
    assert "UTM" in vu_meta.get_wkt_proj_from_vec(gpkg_path, "polys")
    assert vu_meta.get_vec_lyr_bbox(gpkg_path, "polys") == pytest.approx(
        (500000.0, 500300.0, 5600000.0, 5600100.0)
    )
    wgs_bbox = vu_meta.get_vec_lyr_bbox_in_proj(gpkg_path, 4326, "polys")
    assert -5.0 < wgs_bbox[0] < wgs_bbox[1] < 0.0
    assert 50.0 < wgs_bbox[2] < wgs_bbox[3] < 55.0

    assert vu_meta.get_vec_feat_count(gpkg_path, "polys") == 3
    assert vu_meta.get_vec_lyr_geom_type(gpkg_path, "polys") == rsgispy.GEOM_POLY
    assert vu_meta.get_vec_lyr_geom_type(gpkg_path, "pts") == rsgispy.GEOM_PT
    assert vu_meta.get_geom_type_name(rsgispy.GEOM_POLY) == "Polygon"
    assert vu_meta.get_vec_lyr_cols(gpkg_path, "polys") == ["cls", "val"]

    assert vs_metrics.count_feats_per_att_val(gpkg_path, "polys", "cls") == {
        "A": 2,
        "B": 1,
    }
    df_counts = vs_metrics.count_feats_per_att_val(
        gpkg_path, "polys", "cls", out_df_dict=True
    )
    assert set(df_counts["vals"]) == {"A", "B"}
    assert sum(df_counts["count"]) == 3

    shp_path = str(tmp_path / "test_del.shp")
    gdf1.to_file(shp_path)
    assert os.path.exists(shp_path)
    vu_meta.delete_vec_file(shp_path, feedback=False)
    assert not os.path.exists(shp_path)
    assert not os.path.exists(str(tmp_path / "test_del.dbf"))


def test_vectorutils_io_copy_reproj_merge_split(tmp_path):
    """Test copy, translate, reprojection, UTM/WGS84 unwrapping, merging, splitting."""
    in_gpkg = str(tmp_path / "in.gpkg")
    pts_wgs = gpd.GeoDataFrame(
        {"grp": ["x", "y", "x"], "id": [1, 2, 3]},
        geometry=[
            Point(-4.08, 52.41),
            Point(-4.05, 52.42),
            Point(-4.02, 52.40),
        ],
        crs="EPSG:4326",
    )
    pts_wgs.to_file(in_gpkg, layer="pts", driver="GPKG")

    copy_gpkg = str(tmp_path / "copy.gpkg")
    vu_io.create_copy_vec_lyr(in_gpkg, "pts", copy_gpkg, "pts_cp", replace=True)
    assert vu_meta.get_vec_feat_count(copy_gpkg, "pts_cp") == 3

    # Append via vector_translate
    vu_io.vector_translate(
        in_gpkg,
        "pts",
        copy_gpkg,
        out_vec_lyr="pts_cp",
        out_format="GPKG",
        access_mode="append",
    )
    assert vu_meta.get_vec_feat_count(copy_gpkg, "pts_cp") == 6

    geojson_out = str(tmp_path / "pts.geojson")
    vu_io.vector_translate(
        in_gpkg, "pts", geojson_out, out_format="GEOJSON", dst_srs=32630
    )
    assert vu_meta.get_epsg_proj_from_vec(geojson_out) == 32630

    reproj_gpkg = str(tmp_path / "reproj.gpkg")
    vu_io.reproj_vec_lyr(
        in_gpkg, reproj_gpkg, "pts_utm", epsg_code=32630, vec_lyr="pts"
    )
    assert vu_meta.get_epsg_proj_from_vec(reproj_gpkg, "pts_utm") == 32630

    utm_auto_gpkg = str(tmp_path / "utm_auto.gpkg")
    utm_epsg = vu_io.reproj_wgs84_vec_to_utm(
        in_gpkg, "pts", utm_auto_gpkg, "pts_auto"
    )
    assert utm_epsg == 32630
    assert vu_meta.get_epsg_proj_from_vec(utm_auto_gpkg, "pts_auto") == 32630

    redef_gpkg = str(tmp_path / "redef.gpkg")
    vu_io.redefine_vec_lyr_proj(in_gpkg, "pts", 4258, redef_gpkg, "pts_etrs")
    assert vu_meta.get_epsg_proj_from_vec(redef_gpkg, "pts_etrs") == 4258

    # Merging
    f1 = str(tmp_path / "m1.gpkg")
    f2 = str(tmp_path / "m2.gpkg")
    pts_wgs.iloc[:2].to_file(f1, layer="l1", driver="GPKG")
    pts_wgs.iloc[2:].to_file(f2, layer="l2", driver="GPKG")

    merged_files = str(tmp_path / "merged_files.gpkg")
    vu_io.merge_vec_files(
        [f1, f2], merged_files, out_vec_lyr="all_pts", use_pbar=False
    )
    assert vu_meta.get_vec_feat_count(merged_files, "all_pts") == 3

    merged_lyrs = str(tmp_path / "merged_lyrs.gpkg")
    vu_io.merge_vec_lyrs(
        [{"file": f1, "layer": "l1"}, {"file": f2, "layer": "l2"}],
        merged_lyrs,
        out_vec_lyr="all_pts",
        remove_cols=["grp"],
        use_pbar=False,
    )
    assert vu_meta.get_vec_lyr_cols(merged_lyrs, "all_pts") == ["id"]

    multi_out = str(tmp_path / "multi_out.gpkg")
    vu_io.merge_to_multi_lyr_vec(
        [
            vu_meta.VecLayersInfoObj(f1, "l1", "layer_a"),
            vu_meta.VecLayersInfoObj(f2, "l2", "layer_b"),
            vu_meta.VecLayersInfoObj(f2, "l2", "layer_a"),
        ],
        multi_out,
        rename_dup_lyrs=True,
        use_pbar=False,
    )
    assert len(vu_meta.get_vec_lyrs_lst(multi_out)) == 3

    merged_single = str(tmp_path / "merged_single.gpkg")
    vu_io.merge_vec_file_lyrs(
        multi_out, merged_single, out_vec_lyr="combined", use_pbar=False
    )
    assert vu_meta.get_vec_feat_count(merged_single, "combined") == 4

    # Antimeridian unwrapping + UTM-to-WGS84 merge
    wrap_gdf = gpd.GeoDataFrame(
        {"id": [1, 2]},
        geometry=[
            Polygon([(179.0, 0.0), (-179.0, 0.0), (-179.0, 2.0), (179.0, 2.0)]),
            MultiPolygon(
                [Polygon([(178.0, 1.0), (-178.0, 1.0), (-178.0, 2.0), (178.0, 2.0)])]
            ),
        ],
        crs="EPSG:4326",
    )
    unwrapped = vu_io.check_polys_wgs84_bounds_geometry(
        wrap_gdf, width_thres=350.0, use_pbar=False
    )
    assert (unwrapped.bounds["maxx"] - unwrapped.bounds["minx"]).max() < 10.0

    # UTM zones merge into WGS84
    n_zones = str(tmp_path / "n_utm.gpkg")
    s_zones = str(tmp_path / "s_utm.gpkg")
    gpd.GeoDataFrame(
        {"z": [30]}, geometry=[box(-6.0, 0.0, 0.0, 84.0)], crs="EPSG:4326"
    ).to_crs(epsg=32630).to_file(n_zones, layer="30", driver="GPKG")
    gpd.GeoDataFrame(
        {"z": [30]}, geometry=[box(-6.0, -80.0, 0.0, 0.0)], crs="EPSG:4326"
    ).to_crs(epsg=32730).to_file(s_zones, layer="30", driver="GPKG")
    utm_poly = str(tmp_path / "utm_poly.gpkg")
    gpd.GeoDataFrame(
        {"id": [1]},
        geometry=[box(-4.1, 52.3, -3.9, 52.5)],
        crs="EPSG:4326",
    ).to_crs(epsg=32630).to_file(utm_poly, layer="p30", driver="GPKG")
    wgs_merged = str(tmp_path / "wgs_merged.gpkg")
    vu_io.merge_utm_vecs_wgs84(
        [utm_poly],
        wgs_merged,
        out_vec_lyr="wgs",
        n_hemi_utm_file=n_zones,
        s_hemi_utm_file=s_zones,
        use_pbar=False,
    )
    assert vu_meta.get_epsg_proj_from_vec(wgs_merged, "wgs") == 4326
    assert vu_meta.get_vec_feat_count(wgs_merged, "wgs") == 1

    # Splitting
    split_dir = str(tmp_path / "splits")
    chunks = vu_io.split_vec_lyr(
        in_gpkg, "pts", 2, "GPKG", split_dir, "chunk_", "gpkg", use_pbar=False
    )
    assert len(chunks) == 2

    mlyrs = str(tmp_path / "mlyrs.gpkg")
    vu_io.split_feats_to_mlyrs(in_gpkg, "pts", mlyrs, use_pbar=False)
    assert vu_meta.get_vec_lyrs_lst(mlyrs) == ["veclyr_0", "veclyr_1", "veclyr_2"]

    attr_split = str(tmp_path / "attr_split.gpkg")
    vu_io.split_by_attribute(
        in_gpkg, "pts", "grp", multi_layers=True, out_vec_file=attr_split,
        use_pbar=False,
    )
    assert set(vu_meta.get_vec_lyrs_lst(attr_split)) == {"x", "y"}


def test_vectorutils_selection(tmp_path):
    """Test spatial, attribute, matching, and random split selection utilities."""
    # does_vmsk_img_intersect
    vmsk_tif = str(tmp_path / "vmsk.tif")
    arr = np.zeros((20, 20), dtype=np.uint8)
    arr[5:10, 5:10] = 1
    _write_test_raster(
        vmsk_tif, arr, epsg=32630, tl_x=500000.0, tl_y=5600200.0, res=10.0
    )

    roi_hit = str(tmp_path / "roi_hit.gpkg")
    gpd.GeoDataFrame(
        {"id": [1]},
        geometry=[box(500060.0, 5600110.0, 500090.0, 5600140.0)],
        crs="EPSG:32630",
    ).to_file(roi_hit, layer="roi", driver="GPKG")
    assert vu_sel.does_vmsk_img_intersect(vmsk_tif, roi_hit, "roi", use_pbar=False)

    roi_miss = str(tmp_path / "roi_miss.gpkg")
    gpd.GeoDataFrame(
        {"id": [1]},
        geometry=[box(500160.0, 5600010.0, 500190.0, 5600040.0)],
        crs="EPSG:32630",
    ).to_file(roi_miss, layer="roi", driver="GPKG")
    assert not vu_sel.does_vmsk_img_intersect(
        vmsk_tif, roi_miss, "roi", use_pbar=False
    )

    # Spatial selections & attribute extraction
    feats_gpkg = str(tmp_path / "feats.gpkg")
    n_feats = 20
    feats_gdf = gpd.GeoDataFrame(
        {
            "id": list(range(n_feats)),
            "cls": ["River Thames" if i % 2 == 0 else "Lake" for i in range(n_feats)],
            "flag": [0 if i < 15 else 1 for i in range(n_feats)],
        },
        geometry=[
            Point(500005.0 + (i * 10.0), 5600130.0) for i in range(n_feats)
        ],
        crs="EPSG:32630",
    )
    feats_gdf.to_file(feats_gpkg, layer="feats", driver="GPKG")

    sel_out = str(tmp_path / "sel_out.gpkg")
    vu_sel.spatial_select(feats_gpkg, "feats", roi_hit, "roi", sel_out, "sel")
    assert vu_meta.get_vec_feat_count(sel_out, "sel") == 3

    bbox_sel = str(tmp_path / "bbox_sel.gpkg")
    assert vu_sel.spatial_select_bbox(
        feats_gpkg,
        "feats",
        (500000.0, 500040.0, 5600100.0, 5600200.0),
        bbox_sel,
        "bsel",
    )
    assert vu_meta.get_vec_feat_count(bbox_sel, "bsel") == 4

    sub_files = vu_sel.subset_vec_lyr_to_feat_bboxs(
        roi_hit,
        "roi",
        feats_gpkg,
        "feats",
        "sub",
        str(tmp_path / "tile_"),
        use_pbar=False,
    )
    assert len(sub_files) == 1
    assert vu_meta.get_vec_feat_count(sub_files[0], "sub") == 3

    att_sel = vu_sel.get_att_lst_select_feats(
        feats_gpkg, "feats", ["id", "cls"], roi_hit, "roi"
    )
    assert [d["id"] for d in att_sel] == [6, 7, 8]

    att_bbox = vu_sel.get_att_lst_select_bbox_feats(
        feats_gpkg,
        "feats",
        ["id"],
        (500000.0, 500030.0, 5600100.0, 5600200.0),
    )
    assert [d["id"] for d in att_bbox] == [0, 1, 2]

    # Attribute subsetting & drops
    sub_attr = str(tmp_path / "sub_attr.gpkg")
    vu_sel.subset_by_attribute(
        feats_gpkg, "feats", "cls", ["River"], sub_attr, "rivers",
        match_type="start",
    )
    assert vu_meta.get_vec_feat_count(sub_attr, "rivers") == 10

    str_sel = str(tmp_path / "str_sel.gpkg")
    vu_sel.select_feats_str_search(
        feats_gpkg, "feats", "cls", "Thames", str_sel, "thames"
    )
    assert vu_meta.get_vec_feat_count(str_sel, "thames") == 10

    drop1 = str(tmp_path / "drop1.gpkg")
    vu_sel.drop_rows_by_attribute(feats_gpkg, "feats", "flag", [1], drop1, "clean")
    assert vu_meta.get_vec_feat_count(drop1, "clean") == 15

    drop2 = str(tmp_path / "drop2.gpkg")
    vu_sel.drop_rows_by_multi_attributes(
        feats_gpkg, "feats", {"flag": 1, "cls": "Lake"}, drop2, "clean2"
    )
    assert vu_meta.get_vec_feat_count(drop2, "clean2") == 8

    dedup = str(tmp_path / "dedup.gpkg")
    vu_sel.rm_attrib_duplicates(
        feats_gpkg, "feats", "cls", dedup, "dedup", keep_rows="First"
    )
    assert vu_meta.get_vec_feat_count(dedup, "dedup") == 2

    # Nearest point matching
    ref_pts = str(tmp_path / "ref_pts.gpkg")
    gpd.GeoDataFrame(
        {"name": ["P0", "P1"]},
        geometry=[Point(500005.0, 5600130.0), Point(500195.0, 5600130.0)],
        crs="EPSG:32630",
    ).to_file(ref_pts, layer="ref", driver="GPKG")
    matched_gpkg = str(tmp_path / "matched.gpkg")
    assert vu_sel.match_closest_vec_pts(
        feats_gpkg,
        "feats",
        ref_pts,
        "ref",
        matched_gpkg,
        "matched",
        out_format="GPKG",
        tolerance=5.0,
        cp_match_atts=True,
        use_pbar=False,
    )
    m_gdf = gpd.read_file(matched_gpkg, layer="matched")
    assert m_gdf["match_name"].tolist() == ["P0", "P1"]

    # Random splits
    rem_gpkg = str(tmp_path / "rem.gpkg")
    smp_gpkg = str(tmp_path / "smp.gpkg")
    vu_sel.split_vec_lyr_random_subset(
        feats_gpkg, "feats", rem_gpkg, "rem", smp_gpkg, "smp", 5, rnd_seed=42
    )
    assert vu_meta.get_vec_feat_count(rem_gpkg, "rem") == 15
    assert vu_meta.get_vec_feat_count(smp_gpkg, "smp") == 5

    rsub = str(tmp_path / "rsub.gpkg")
    vu_sel.vec_lyr_random_subset(feats_gpkg, "feats", rsub, "sub", 6, rnd_seed=42)
    assert vu_meta.get_vec_feat_count(rsub, "sub") == 6

    tr = str(tmp_path / "train.gpkg")
    te = str(tmp_path / "test.gpkg")
    vu_sel.create_train_test_smpls(
        feats_gpkg, "feats", tr, "tr", te, "te", prop_test=0.25, rnd_seed=42
    )
    tr_ids = set(gpd.read_file(tr, layer="tr")["id"])
    te_ids = set(gpd.read_file(te, layer="te")["id"])
    assert len(tr_ids) == 15 and len(te_ids) == 5 and tr_ids.isdisjoint(te_ids)

    tr2 = str(tmp_path / "tr2.gpkg")
    te2 = str(tmp_path / "te2.gpkg")
    va2 = str(tmp_path / "va2.gpkg")
    vu_sel.create_train_test_valid_smpls(
        feats_gpkg,
        "feats",
        tr2,
        "tr",
        te2,
        "te",
        va2,
        "va",
        prop_test=0.2,
        prop_valid=0.2,
        rnd_seed=42,
    )
    s_tr = set(gpd.read_file(tr2, layer="tr")["id"])
    s_te = set(gpd.read_file(te2, layer="te")["id"])
    s_va = set(gpd.read_file(va2, layer="va")["id"])
    assert len(s_tr) == 12 and len(s_te) == 4 and len(s_va) == 4
    assert s_tr.isdisjoint(s_te) and s_tr.isdisjoint(s_va) and s_te.isdisjoint(s_va)

    boot_files = vu_sel.create_n_random_subsets(
        feats_gpkg,
        "feats",
        str(tmp_path / "boots"),
        "boot",
        "gpkg",
        n_subs=3,
        n_smpl=5,
        rnd_seed=42,
        use_pbar=False,
    )
    assert len(boot_files) == 3
    assert all(vu_meta.get_vec_feat_count(bf) == 5 for bf in boot_files)

    for moved_name in (
        "count_feats_per_att_val",
        "rasterise_vec_lyr",
        "polygonise_raster_to_vec_lyr",
        "define_grid",
    ):
        assert not hasattr(vu_io, moved_name)
        assert not hasattr(vu_meta, moved_name)
        assert not hasattr(vu_sel, moved_name)

    assert rsgispy.vector.__all__ == ["io", "create", "geoms", "attrs", "stats", "selection"]
    for reexport_name in (
        "get_vec_lyr_bbox",
        "merge_vec_files",
        "spatial_select",
        "VecLayersInfoObj",
    ):
        assert not hasattr(rsgispy.vector, reexport_name)


def test_vectorutils_gdf_in_memory():
    """Test GeoDataFrame inputs and out_vec_file=None returns across vectorutils."""
    pts_gdf = gpd.GeoDataFrame(
        {
            "id": list(range(20)),
            "cls": ["A" if i % 2 == 0 else "B" for i in range(20)],
            "note": [None if i == 0 else f"n_{i}" for i in range(20)],
        },
        geometry=[
            Point(500005.0 + (i % 5) * 20.0, 5600005.0 + (i // 5) * 20.0)
            for i in range(20)
        ],
        crs="EPSG:32630",
    )
    orig_cols = list(pts_gdf.columns)

    # vectorutils.metadata
    assert vu_meta.get_vec_feat_count(pts_gdf) == 20
    assert vu_meta.get_epsg_proj_from_vec(pts_gdf) == 32630
    assert "32630" in vu_meta.get_wkt_proj_from_vec(pts_gdf)
    assert vu_meta.get_vec_lyrs_lst(pts_gdf) == ["layer"]
    assert "cls" in vu_meta.get_vec_lyr_cols(pts_gdf)
    assert vu_meta.get_vec_lyr_geom_type(pts_gdf) == rsgispy.GEOM_PT
    bbox = vu_meta.get_vec_lyr_bbox(pts_gdf)
    assert bbox[0] == pytest.approx(500005.0)
    wgs_bbox = vu_meta.get_vec_lyr_bbox_in_proj(pts_gdf, 4326)
    assert wgs_bbox[0] < 0.0 and wgs_bbox[2] > 50.0

    # vectorutils.io
    reproj_wgs = vu_io.reproj_vec_lyr(pts_gdf, None, epsg_code=4326)
    assert isinstance(reproj_wgs, gpd.GeoDataFrame)
    assert reproj_wgs.crs.to_epsg() == 4326

    utm_back = vu_io.reproj_wgs84_vec_to_utm(reproj_wgs, None, out_vec_file=None)
    assert isinstance(utm_back, gpd.GeoDataFrame)
    assert utm_back.crs.to_epsg() == 32630

    trans_gdf = vu_io.vector_translate(pts_gdf, None, dst_srs=4326)
    assert trans_gdf.crs.to_epsg() == 4326

    redef_gdf = vu_io.redefine_vec_lyr_proj(pts_gdf, None, 32630)
    assert redef_gdf.crs.to_epsg() == 32630

    merged_gdf = vu_io.merge_vec_files(
        [pts_gdf.iloc[:5], pts_gdf.iloc[5:10]], use_pbar=False
    )
    assert len(merged_gdf) == 10

    lyr_merged = vu_io.merge_vec_lyrs(
        [{"file": pts_gdf.iloc[:3], "layer": None}, {"file": pts_gdf.iloc[3:6], "layer": None}],
        use_pbar=False,
    )
    assert len(lyr_merged) == 6

    chunks = vu_io.split_vec_lyr(pts_gdf, None, n_feats=6, use_pbar=False)
    assert isinstance(chunks, list)
    assert [len(c) for c in chunks] == [6, 6, 6, 2]

    feat_dict = vu_io.split_feats_to_mlyrs(pts_gdf.iloc[:3], None, use_pbar=False)
    assert len(feat_dict) == 3

    cls_dict = vu_io.split_by_attribute(pts_gdf, None, "cls", use_pbar=False)
    assert set(cls_dict.keys()) == {"A", "B"}
    assert len(cls_dict["A"]) == 10

    # vectorutils.selection + multi-step in-memory pipeline
    roi_gdf = gpd.GeoDataFrame(
        {"roi": [1]},
        geometry=[box(500000.0, 5600000.0, 500050.0, 5600050.0)],
        crs="EPSG:32630",
    )
    sel_gdf = vu_sel.spatial_select(pts_gdf, None, roi_gdf, None)
    assert isinstance(sel_gdf, gpd.GeoDataFrame)
    assert 0 < len(sel_gdf) < 20

    sel_bbox_gdf = vu_sel.spatial_select_bbox(
        pts_gdf, None, (500000.0, 500050.0, 5600000.0, 5600050.0)
    )
    assert isinstance(sel_bbox_gdf, gpd.GeoDataFrame)
    assert len(sel_bbox_gdf) == len(sel_gdf)

    sub_att = vu_sel.subset_by_attribute(pts_gdf, None, "cls", ["A"])
    assert len(sub_att) == 10

    sub_str = vu_sel.select_feats_str_search(pts_gdf, None, "cls", "B")
    assert len(sub_str) == 10

    dropped = vu_sel.drop_rows_by_attribute(pts_gdf, None, "cls", ["B"])
    assert len(dropped) == 10

    dropped_multi = vu_sel.drop_rows_by_multi_attributes(pts_gdf, None, {"cls": "B"})
    assert len(dropped_multi) == 10

    dedup = vu_sel.rm_attrib_duplicates(pts_gdf, None, "cls")
    assert len(dedup) == 2

    rem_gdf, smp_gdf = vu_sel.split_vec_lyr_random_subset(
        pts_gdf, None, None, None, None, None, 5, rnd_seed=42
    )
    assert len(rem_gdf) == 15 and len(smp_gdf) == 5

    rsub_gdf = vu_sel.vec_lyr_random_subset(pts_gdf, None, n_smpl=6, rnd_seed=42)
    assert len(rsub_gdf) == 6

    tr_gdf, te_gdf = vu_sel.create_train_test_smpls(
        pts_gdf, None, prop_test=0.25, rnd_seed=42
    )
    assert len(tr_gdf) == 15 and len(te_gdf) == 5
    assert set(tr_gdf["id"]).isdisjoint(set(te_gdf["id"]))

    tr2_gdf, te2_gdf, va2_gdf = vu_sel.create_train_test_valid_smpls(
        pts_gdf, None, prop_test=0.2, prop_valid=0.2, rnd_seed=42
    )
    assert len(tr2_gdf) == 12 and len(te2_gdf) == 4 and len(va2_gdf) == 4

    boot_gdfs = vu_sel.create_n_random_subsets(
        pts_gdf, None, n_subs=3, n_smpl=4, rnd_seed=42, use_pbar=False
    )
    assert len(boot_gdfs) == 3 and all(len(b) == 4 for b in boot_gdfs)

    matched_gdf = vu_sel.match_closest_vec_pts(
        pts_gdf.iloc[:2], None, pts_gdf, None, cp_match_atts=True, use_pbar=False
    )
    assert isinstance(matched_gdf, gpd.GeoDataFrame)
    assert matched_gdf["dist_match"].tolist() == pytest.approx([0.0, 0.0])

    # Input GeoDataFrame must remain unmodified
    assert len(pts_gdf) == 20
    assert list(pts_gdf.columns) == orig_cols


def test_convert_gpx_to_vec(tmp_path):
    """Test convert_gpx_to_vec for in-memory GeoDataFrames, vector outputs, and edge cases."""
    gpx_content = """<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="rsgispy-test"
     xmlns="http://www.topografix.com/GPX/1/1"
     xmlns:gpxtpx="http://www.garmin.com/xmlschemas/TrackPointExtension/v1">
  <trk>
    <name>Morning Hike</name>
    <cmt>Test comment</cmt>
    <desc>Two-segment route</desc>
    <src>GPS</src>
    <type>hiking</type>
    <number>7</number>
    <trkseg>
      <trkpt lat="52.4150" lon="-4.0820">
        <ele>12.5</ele>
        <time>2025-06-01T08:00:00Z</time>
        <extensions>
          <gpxtpx:TrackPointExtension>
            <gpxtpx:hr>118</gpxtpx:hr>
            <gpxtpx:cad>82</gpxtpx:cad>
          </gpxtpx:TrackPointExtension>
        </extensions>
      </trkpt>
      <trkpt lat="52.4160" lon="-4.0810">
        <ele>18.0</ele>
        <time>2025-06-01T08:01:00Z</time>
        <extensions>
          <gpxtpx:TrackPointExtension>
            <gpxtpx:hr>125</gpxtpx:hr>
            <gpxtpx:cad>84</gpxtpx:cad>
          </gpxtpx:TrackPointExtension>
        </extensions>
      </trkpt>
    </trkseg>
    <trkseg>
      <trkpt lat="52.4170" lon="-4.0800">
        <ele>24.5</ele>
        <time>2025-06-01T08:05:00Z</time>
      </trkpt>
      <trkpt lat="52.4180" lon="-4.0790">
        <ele>30.0</ele>
        <time>2025-06-01T08:06:00Z</time>
      </trkpt>
    </trkseg>
  </trk>
</gpx>
"""
    gpx_file = str(tmp_path / "route.gpx")
    with open(gpx_file, "w", encoding="utf-8") as fobj:
        fobj.write(gpx_content)

    # 1. Both outputs returned in memory
    pts_gdf, lines_gdf = vu_io.convert_gpx_to_vec(gpx_file)
    assert isinstance(pts_gdf, gpd.GeoDataFrame)
    assert isinstance(lines_gdf, gpd.GeoDataFrame)
    assert pts_gdf.crs.to_epsg() == 4326
    assert lines_gdf.crs.to_epsg() == 4326
    assert len(pts_gdf) == 4
    assert list(pts_gdf.geometry.geom_type) == ["Point"] * 4
    assert not pts_gdf.geometry.has_z.any()
    assert pts_gdf["track_name"].tolist() == ["Morning Hike"] * 4
    assert pts_gdf["track_seg_id"].tolist() == [0, 0, 1, 1]
    assert pts_gdf["track_seg_point_id"].tolist() == [0, 1, 0, 1]
    assert pts_gdf["ele"].tolist() == pytest.approx([12.5, 18.0, 24.5, 30.0])
    assert pts_gdf["hr"].iloc[:2].tolist() == pytest.approx([118.0, 125.0])
    assert pts_gdf["cad"].iloc[:2].tolist() == pytest.approx([82.0, 84.0])

    assert len(lines_gdf) == 1
    assert lines_gdf.geometry.iloc[0].geom_type == "MultiLineString"
    assert not lines_gdf.geometry.has_z.any()
    assert lines_gdf["name"].iloc[0] == "Morning Hike"
    assert lines_gdf["cmt"].iloc[0] == "Test comment"
    assert lines_gdf["desc"].iloc[0] == "Two-segment route"
    assert lines_gdf["src"].iloc[0] == "GPS"
    assert lines_gdf["type"].iloc[0] == "hiking"
    assert int(lines_gdf["number"].iloc[0]) == 7
    assert int(lines_gdf["n_pts"].iloc[0]) == 4
    assert lines_gdf["start_time"].iloc[0] == "2025-06-01T08:00:00Z"
    assert lines_gdf["end_time"].iloc[0] == "2025-06-01T08:06:00Z"

    # 2. split_segments=True and include_z=True
    pts_z, seg_lines_z = vu_io.convert_gpx_to_vec(
        gpx_file, split_segments=True, include_z=True
    )
    assert pts_z.geometry.has_z.all()
    assert pts_z.geometry.iloc[0].z == pytest.approx(12.5)
    assert len(seg_lines_z) == 2
    assert list(seg_lines_z.geometry.geom_type) == ["LineString", "LineString"]
    assert seg_lines_z.geometry.has_z.all()
    assert seg_lines_z["track_seg_id"].tolist() == [0, 1]
    assert seg_lines_z["n_pts"].tolist() == [2, 2]

    # 3. Reprojection to UTM (EPSG:32630)
    pts_utm, lines_utm = vu_io.convert_gpx_to_vec(gpx_file, out_epsg=32630)
    assert pts_utm.crs.to_epsg() == 32630
    assert lines_utm.crs.to_epsg() == 32630

    # 4. Mixed output: points saved, lines returned
    pts_only_gpkg = str(tmp_path / "pts_only.gpkg")
    ret_lines = vu_io.convert_gpx_to_vec(
        gpx_file, pts_vec_file=pts_only_gpkg, pts_vec_lyr="pts"
    )
    assert isinstance(ret_lines, gpd.GeoDataFrame)
    assert len(ret_lines) == 1
    assert vu_meta.get_vec_feat_count(pts_only_gpkg, "pts") == 4

    # 5. Mixed output: lines saved, points returned
    lines_only_gpkg = str(tmp_path / "lines_only.gpkg")
    ret_pts = vu_io.convert_gpx_to_vec(
        gpx_file, line_vec_file=lines_only_gpkg, line_vec_lyr="lines"
    )
    assert isinstance(ret_pts, gpd.GeoDataFrame)
    assert len(ret_pts) == 4
    assert vu_meta.get_vec_feat_count(lines_only_gpkg, "lines") == 1

    # 6. Multi-layer single GPKG output with overwrite (del_exist_vec=True)
    multi_gpkg = str(tmp_path / "track_bundle.gpkg")
    ret_none = vu_io.convert_gpx_to_vec(
        gpx_file,
        pts_vec_file=multi_gpkg,
        line_vec_file=multi_gpkg,
    )
    assert ret_none is None
    assert set(vu_meta.get_vec_lyrs_lst(multi_gpkg)) == {"track_points", "tracks"}
    assert vu_meta.get_vec_feat_count(multi_gpkg, "track_points") == 4
    assert vu_meta.get_vec_feat_count(multi_gpkg, "tracks") == 1

    # Re-write with del_exist_vec=True
    assert (
        vu_io.convert_gpx_to_vec(
            gpx_file,
            pts_vec_file=multi_gpkg,
            line_vec_file=multi_gpkg,
            del_exist_vec=True,
        )
        is None
    )
    assert set(vu_meta.get_vec_lyrs_lst(multi_gpkg)) == {"track_points", "tracks"}

    # 7. Separate GeoParquet outputs (when pyarrow is installed)
    import importlib.util

    if importlib.util.find_spec("pyarrow") is not None:
        pts_pq = str(tmp_path / "pts.parquet")
        lines_pq = str(tmp_path / "lines.parquet")
        assert (
            vu_io.convert_gpx_to_vec(
                gpx_file,
                pts_vec_file=pts_pq,
                line_vec_file=lines_pq,
                out_format="Parquet",
            )
            is None
        )
        assert vu_meta.get_vec_feat_count(pts_pq) == 4
        assert vu_meta.get_vec_feat_count(lines_pq) == 1

    # 8. Empty GPX (valid GPX with zero track points)
    empty_gpx = str(tmp_path / "empty.gpx")
    with open(empty_gpx, "w", encoding="utf-8") as fobj:
        fobj.write('<?xml version="1.0" encoding="UTF-8"?><gpx version="1.1"></gpx>')
    empty_pts, empty_lines = vu_io.convert_gpx_to_vec(empty_gpx)
    assert len(empty_pts) == 0
    assert len(empty_lines) == 0
    assert empty_pts.crs.to_epsg() == 4326
    assert empty_lines.crs.to_epsg() == 4326
    assert {"track_fid", "track_seg_id", "track_seg_point_id", "ele", "time"}.issubset(
        empty_pts.columns
    )

    # 9. Error handling
    with pytest.raises(rsgispy.RSGISPyException):
        vu_io.convert_gpx_to_vec(str(tmp_path / "missing.gpx"))

    bad_xml = str(tmp_path / "bad.gpx")
    with open(bad_xml, "w", encoding="utf-8") as fobj:
        fobj.write("<kml><Document/></kml>")
    with pytest.raises(rsgispy.RSGISPyException):
        vu_io.convert_gpx_to_vec(bad_xml)

    with pytest.raises(rsgispy.RSGISPyException):
        vu_io.convert_gpx_to_vec(
            gpx_file,
            pts_vec_file=multi_gpkg,
            pts_vec_lyr="same",
            line_vec_file=multi_gpkg,
            line_vec_lyr="same",
        )

    same_pq = str(tmp_path / "same.parquet")
    with pytest.raises(rsgispy.RSGISPyException):
        vu_io.convert_gpx_to_vec(
            gpx_file,
            pts_vec_file=same_pq,
            line_vec_file=same_pq,
            out_format="Parquet",
        )


def test_vector_consolidation_bbox_and_uniform_vector_input(tmp_path):
    """Verify rsgispy.vector consolidation, BBox returns, and uniform VectorInput parity."""
    import importlib
    from shapely.geometry import MultiPolygon

    # Legacy top-level packages must be removed (no backwards-compat aliases)
    for removed_pkg in (
        "rsgispy.vectorutils",
        "rsgispy.vectorgeoms",
        "rsgispy.vectorattrs",
        "rsgispy.vectorstats",
    ):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(removed_pkg)

    assert rsgispy.vector.__all__ == [
        "io",
        "create",
        "geoms",
        "attrs",
        "stats",
        "selection",
    ]

    # BBox returns with EPSG metadata & automatic epsg inference in spatial_select_bbox
    pts_gdf = gpd.GeoDataFrame(
        {"id": [1, 2, 3]},
        geometry=[
            Point(500010.0, 5600010.0),
            Point(500050.0, 5600050.0),
            Point(500090.0, 5600090.0),
        ],
        crs="EPSG:32630",
    )
    pts_gpkg = str(tmp_path / "pts.gpkg")
    pts_gdf.to_file(pts_gpkg, layer="pts", driver="GPKG")

    bbox = rsgispy.vector.io.get_vec_lyr_bbox(pts_gpkg, "pts")
    assert isinstance(bbox, rsgispy.BBox)
    assert bbox.epsg == 32630

    bbox_wgs = rsgispy.vector.io.get_vec_lyr_bbox_in_proj(pts_gpkg, 4326, "pts")
    assert isinstance(bbox_wgs, rsgispy.BBox)
    assert bbox_wgs.epsg == 4326

    # Reproject sub-bbox to WGS84 and pass WITHOUT bbox_epsg -> uses BBox.epsg!
    sub_utm = rsgispy.BBox(500000.0, 500060.0, 5600000.0, 5600060.0, epsg=32630)
    sub_wgs = sub_utm.reproject(4326)
    sel = rsgispy.vector.selection.spatial_select_bbox(pts_gdf, None, sub_wgs)
    assert list(sel["id"]) == [1, 2]

    feat_bboxes = rsgispy.vector.geoms.get_geoms_as_bboxs(pts_gdf, None)
    assert all(isinstance(b, rsgispy.BBox) and b.epsg == 32630 for b in feat_bboxes)

    # write_vec_column_to_layer accepts BOTH file path and GeoDataFrame
    updated_file_gdf = rsgispy.vector.attrs.write_vec_column_to_layer(
        pts_gpkg, "score", float, [1.5, 2.5, 3.5], vec_lyr="pts"
    )
    assert list(updated_file_gdf["score"]) == [1.5, 2.5, 3.5]
    reloaded = gpd.read_file(pts_gpkg, layer="pts")
    assert list(reloaded["score"]) == [1.5, 2.5, 3.5]

    # check_polys_wgs84_bounds_geometry accepts BOTH file path and GeoDataFrame
    wrap_poly = Polygon(
        [(179.0, 10.0), (-179.0, 10.0), (-179.0, 12.0), (179.0, 12.0)]
    )
    wrap_gpkg = str(tmp_path / "wrap.gpkg")
    gpd.GeoDataFrame({"id": [1]}, geometry=[wrap_poly], crs="EPSG:4326").to_file(
        wrap_gpkg, layer="wrap", driver="GPKG"
    )
    unwrapped = rsgispy.vector.io.check_polys_wgs84_bounds_geometry(
        wrap_gpkg, vec_lyr="wrap", use_pbar=False
    )
    assert isinstance(unwrapped, gpd.GeoDataFrame)
    assert unwrapped.geometry.iloc[0].bounds[2] - unwrapped.geometry.iloc[0].bounds[0] < 10.0

    # explode_vec_files accepts GeoDataFrame inputs and output_dir=None
    mp_gdf = gpd.GeoDataFrame(
        {"id": [1]},
        geometry=[MultiPolygon([box(0, 0, 1, 1), box(2, 2, 3, 3)])],
        crs="EPSG:32630",
    )
    exploded_list = rsgispy.vector.geoms.explode_vec_files(
        [mp_gdf], output_dir=None, use_pbar=False
    )
    assert len(exploded_list) == 1
    assert len(exploded_list[0]) == 2


def _write_wgs84_utm_zone_files(tmp_path):
    """Write north/south UTM zone files (zones 30 and 31) in EPSG:4326."""
    n_zones = str(tmp_path / "n_utm_wgs.gpkg")
    s_zones = str(tmp_path / "s_utm_wgs.gpkg")
    for zone, min_lon in (("30", -6.0), ("31", 0.0)):
        gpd.GeoDataFrame(
            {"z": [int(zone)]},
            geometry=[box(min_lon, 0.0, min_lon + 6.0, 84.0)],
            crs="EPSG:4326",
        ).to_file(n_zones, layer=zone, driver="GPKG")
        gpd.GeoDataFrame(
            {"z": [int(zone)]},
            geometry=[box(min_lon, -80.0, min_lon + 6.0, 0.0)],
            crs="EPSG:4326",
        ).to_file(s_zones, layer=zone, driver="GPKG")
    return n_zones, s_zones


def test_merge_utm_vecs_wgs84_splits_at_zone_boundary_wgs84_zones(tmp_path):
    """H6: zone layers in WGS84 are reprojected before the overlay."""
    n_zones, s_zones = _write_wgs84_utm_zone_files(tmp_path)
    cross_gdf = gpd.GeoDataFrame(
        {"id": [1]},
        geometry=[box(-0.5, 52.0, 0.5, 52.5)],
        crs="EPSG:4326",
    ).to_crs(epsg=32630)
    merged = vu_io.merge_utm_vecs_wgs84(
        [cross_gdf],
        None,
        n_hemi_utm_file=n_zones,
        s_hemi_utm_file=s_zones,
        use_pbar=False,
    )
    assert merged.crs.to_epsg() == 4326
    # The polygon is split into the part inside zone 30 and the part outside.
    assert len(merged) == 2
    max_x = sorted(float(v) for v in merged.bounds["maxx"])
    assert max_x[0] == pytest.approx(0.0, abs=1e-6)
    assert merged.to_crs(epsg=32630).area.sum() == pytest.approx(
        cross_gdf.area.sum(), rel=1e-3
    )


def test_merge_utm_vecs_wgs84_keeps_zero_area_layers(tmp_path):
    """H6: single-point and axis-parallel line layers are not dropped."""
    n_zones, s_zones = _write_wgs84_utm_zone_files(tmp_path)
    pt_gdf = gpd.GeoDataFrame(
        {"id": [1]}, geometry=[Point(-3.0, 52.0)], crs="EPSG:4326"
    ).to_crs(epsg=32630)
    x0, y0 = pt_gdf.geometry.iloc[0].x, pt_gdf.geometry.iloc[0].y
    line_gdf = gpd.GeoDataFrame(
        {"id": [2]},
        geometry=[LineString([(x0, y0), (x0 + 1000.0, y0)])],
        crs="EPSG:32630",
    )
    merged = vu_io.merge_utm_vecs_wgs84(
        [pt_gdf, line_gdf],
        None,
        n_hemi_utm_file=n_zones,
        s_hemi_utm_file=s_zones,
        use_pbar=False,
    )
    assert sorted(merged["id"].tolist()) == [1, 2]
    assert merged.crs.to_epsg() == 4326


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


def test_write_vec_gdf_replace_layer_preserves_siblings(tmp_path):
    """replace_layer=True replaces only the target GPKG layer."""
    gpkg = str(tmp_path / "multi.gpkg")
    gdf_a = gpd.GeoDataFrame(
        {"id": [1]}, geometry=[Point(0, 0)], crs="EPSG:4326"
    )
    gdf_b = gpd.GeoDataFrame(
        {"id": [10, 20]},
        geometry=[Point(1, 1), Point(2, 2)],
        crs="EPSG:4326",
    )
    vp.write_vec_gdf(gdf_a, gpkg, "lyr_a", out_format="GPKG", overwrite=True)
    vp.write_vec_gdf(gdf_b, gpkg, "lyr_b", out_format="GPKG", overwrite=False)

    # Updating lyr_a with replace_layer=True keeps lyr_b intact
    gdf_a2 = gpd.GeoDataFrame(
        {"id": [99]}, geometry=[Point(5, 5)], crs="EPSG:4326"
    )
    vp.write_vec_gdf(
        gdf_a2,
        gpkg,
        "lyr_a",
        out_format="GPKG",
        overwrite=False,
        replace_layer=True,
    )
    assert set(vp.list_vec_layers(gpkg)) == {"lyr_a", "lyr_b"}
    assert list(vp.read_vec_gdf(gpkg, "lyr_a")["id"]) == [99]
    assert list(vp.read_vec_gdf(gpkg, "lyr_b")["id"]) == [10, 20]


def test_read_vec_info_resolve_layer_and_driver_helpers(tmp_path):
    """read_vec_info, resolve_vec_layer, is_multi_layer_driver, cast_col_data."""
    assert vp.is_multi_layer_driver(out_format="GPKG") is True
    assert vp.is_multi_layer_driver(out_vec_file="a.gpkg") is True
    assert vp.is_multi_layer_driver(out_format="GeoJSON") is False
    assert vp.is_multi_layer_driver(out_format="Parquet") is False

    gdf = gpd.GeoDataFrame(
        {"a": [1, 2], "b": ["x", "y"]},
        geometry=[Point(0, 0), Point(2, 3)],
        crs="EPSG:4326",
    )
    info = vp.read_vec_info(gdf)
    assert info["features"] == 2
    assert info["fields"] == ["a", "b"]
    assert info["total_bounds"] == (0.0, 0.0, 2.0, 3.0)

    gpkg = str(tmp_path / "sample.gpkg")
    vp.write_vec_gdf(gdf, gpkg, "pts", out_format="GPKG", overwrite=True)
    assert vp.resolve_vec_layer(gpkg, None) == "pts"
    with pytest.raises(RSGISPyException):
        vp.resolve_vec_layer(gpkg, "missing_layer")

    bbox = vp.bbox_to_rsgis((1.0, 2.0, 3.0, 4.0), epsg=4326)
    assert (bbox.west, bbox.east, bbox.south, bbox.north, bbox.epsg) == (
        1.0,
        3.0,
        2.0,
        4.0,
        4326,
    )
    assert vp.cast_col_data(["1", "2"], int) == [1, 2]
    assert vp.cast_col_data(["true", "0"], bool) == [True, False]
    with pytest.raises(RSGISPyException):
        vp.cast_col_data(["not_a_bool"], bool)
    atts_df = vp.build_atts_dataframe(
        2,
        atts={"k": ["10", "20"]},
        att_types={"names": ["k"], "types": [int]},
    )
    assert list(atts_df["k"]) == [10, 20]


def test_vector_io_invalid_inputs_raise_rsgispy_exception(tmp_path):
    """Invalid inputs in _io_ops raise RSGISPyException rather than AssertionError."""
    gdf = gpd.GeoDataFrame(
        {"id": [1]}, geometry=[Point(0, 0)], crs="EPSG:4326"
    )
    out_json = str(tmp_path / "out.geojson")
    lyr1 = vec_io.VecLayersInfoObj(gdf, None, "l1")
    lyr2 = vec_io.VecLayersInfoObj(gdf, None, "l2")
    with pytest.raises(RSGISPyException):
        vec_io.merge_to_multi_lyr_vec(
            [lyr1, lyr2], out_json, out_format="GeoJSON"
        )

    with pytest.raises(RSGISPyException):
        vec_io.merge_vec_lyrs([], out_vec_file=None)


def test_selection_crs_alignment_ckdtree_iloc_na_regex_and_empty_roi(
    tmp_path, monkeypatch
):
    """Test selection deduplication, CRS alignment, cKDTree, iloc, NA/regex, empty ROI."""
    # Deduplication when a left feature overlaps multiple right features
    left_gdf = gpd.GeoDataFrame(
        {"name": ["poly_a", "poly_b"]},
        geometry=[box(0, 0, 10, 10), box(100, 100, 110, 110)],
        crs="EPSG:32630",
    )
    right_gdf = gpd.GeoDataFrame(
        {"rid": [1, 2]},
        geometry=[box(0, 0, 5, 5), box(5, 5, 10, 10)],
        crs="EPSG:32630",
    )
    assert vec_sel.get_att_lst_select_feats(
        left_gdf, None, ["name"], right_gdf, None
    ) == [{"name": "poly_a"}]
    assert vec_sel.get_att_lst_select_bbox_feats(
        left_gdf, None, ["name"], (0.0, 10.0, 0.0, 10.0), bbox_epsg=32630
    ) == [{"name": "poly_a"}]

    # CRS alignment + null/empty filtering in match_closest_vec_pts
    pts_utm = gpd.GeoDataFrame(
        {"pid": [1, 2, 3]},
        geometry=[Point(500000.0, 5200000.0), None, Point()],
        crs="EPSG:32630",
    )
    ref_wgs84 = gpd.GeoDataFrame(
        {"ref_id": ["near", "far"]},
        geometry=[Point(500005.0, 5200000.0), Point(505000.0, 5200000.0)],
        crs="EPSG:32630",
    ).to_crs("EPSG:4326")
    matched = vec_sel.match_closest_vec_pts(
        pts_utm,
        None,
        ref_wgs84,
        None,
        out_vec_file=None,
        cp_match_atts=True,
        use_pbar=False,
    )
    assert len(matched) == 1
    assert matched["match_ref_id"].iloc[0] == "near"
    assert matched["dist_match"].iloc[0] == pytest.approx(5.0, abs=0.5)

    monkeypatch.setattr(vec_sel, "HAVE_SCIPY", False)
    matched_np = vec_sel.match_closest_vec_pts(
        pts_utm,
        None,
        ref_wgs84,
        None,
        out_vec_file=None,
        cp_match_atts=True,
        use_pbar=False,
    )
    assert len(matched_np) == 1
    assert matched_np["match_ref_id"].iloc[0] == "near"
    assert matched_np["dist_match"].iloc[0] == pytest.approx(
        matched["dist_match"].iloc[0]
    )
    monkeypatch.undo()

    # Positional .iloc splitting with non-contiguous index
    idx_gdf = gpd.GeoDataFrame(
        {"cls": ["a", "a", "a", "b", "b", "b"]},
        geometry=[Point(i, i) for i in range(6)],
        index=[10, 20, 30, 40, 50, 60],
        crs="EPSG:32630",
    )
    rem_gdf, sub_gdf = vec_sel.split_vec_lyr_random_subset(
        idx_gdf, None, None, None, None, None, n_smpl=2, rnd_seed=42
    )
    assert len(sub_gdf) == 2 and len(rem_gdf) == 4
    train_gdf, test_gdf = vec_sel.create_train_test_smpls(
        idx_gdf, None, prop_test=0.34, rnd_seed=42
    )
    assert len(test_gdf) == 2 and len(train_gdf) == 4

    # subset_by_attribute with NA values & regex metacharacters
    attr_gdf = gpd.GeoDataFrame(
        {"label": ["a[1]x", "a1x", None, "b"]},
        geometry=[Point(i, 0) for i in range(4)],
        crs="EPSG:4326",
    )
    sub_lit = vec_sel.subset_by_attribute(
        attr_gdf,
        None,
        sub_col="label",
        sub_vals=["a[1]"],
        match_type="contains",
        out_vec_file=None,
    )
    assert list(sub_lit["label"]) == ["a[1]x"]
    with pytest.raises(RSGISPyException):
        vec_sel.subset_by_attribute(
            attr_gdf,
            None,
            sub_col="label",
            sub_vals=["None"],
            match_type="contains",
            out_vec_file=None,
        )

    # drop_rows_by_multi_attributes keeps nullable NA rows
    na_gdf = gpd.GeoDataFrame(
        {
            "k": pd.Series(["drop", pd.NA, "keep"], dtype="string"),
            "v": pd.Series([1, pd.NA, 2], dtype="Int64"),
        },
        geometry=[Point(0, 0), Point(1, 1), Point(2, 2)],
        crs="EPSG:4326",
    )
    kept = vec_sel.drop_rows_by_multi_attributes(
        na_gdf,
        None,
        drop_atts={"k": "drop", "v": 99},
        out_vec_file=None,
    )
    assert len(kept) == 2

    # does_vmsk_img_intersect with empty ROI
    tif = _write_raster(
        str(tmp_path / "img.tif"),
        np.ones((4, 4), dtype=np.uint8),
    )
    empty_roi = gpd.GeoDataFrame(
        {"id": []}, geometry=gpd.GeoSeries([], crs="EPSG:32630"), crs="EPSG:32630"
    )
    assert vec_sel.does_vmsk_img_intersect(tif, empty_roi, None) is False
