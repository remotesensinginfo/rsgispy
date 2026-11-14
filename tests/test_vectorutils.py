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
from shapely.geometry import MultiPolygon, Point, Polygon, box

import rsgispy
import rsgispy.vectorprocess
import rsgispy.vectorstats.metrics as vs_metrics
import rsgispy.vectorutils
import rsgispy.vectorutils.io as vu_io
import rsgispy.vectorutils.metadata as vu_meta
import rsgispy.vectorutils.selection as vu_sel


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

    assert not hasattr(rsgispy.vectorutils, "__all__")
    assert not hasattr(rsgispy.vectorprocess, "__all__")
    for reexport_name in (
        "get_vec_lyr_bbox",
        "merge_vec_files",
        "spatial_select",
        "VecLayersInfoObj",
    ):
        assert not hasattr(rsgispy.vectorutils, reexport_name)
