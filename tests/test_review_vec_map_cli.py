"""Targeted unit tests for Workstream 3 review fixes across vectorprocess,
vector, zonalstats, mapping, and CLI modules.
"""

from __future__ import annotations

import builtins
import datetime
import json
import shutil
import sys
import warnings
from types import SimpleNamespace

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString, MultiLineString, Point, box

from rsgispy import RSGISPyException
from rsgispy.cli import (
    batchconvert2cog,
    buildimglut,
    calcimgstats,
    chkgdalfile,
    copybandnames,
    filehash,
    imginfo,
    proj,
    userpassfile,
    vectools,
    wgs84imgres,
)
from rsgispy.mapping import _legend as map_legend_internal
from rsgispy.mapping import basemaps as map_basemaps
from rsgispy.mapping import elements as map_elements
from rsgispy.mapping import legend as map_legend
from rsgispy.mapping import raster as map_raster
from rsgispy.mapping import vector as map_vector
from rsgispy.vector import attrs as vec_attrs
from rsgispy.vector import create as vec_create
from rsgispy.vector import geoms as vec_geoms
from rsgispy.vector import io as vec_io
from rsgispy.vector import selection as vec_sel
from rsgispy.vector import stats as vec_stats
from rsgispy import vectorprocess as vp
from rsgispy.zonalstats import point as zs_point
from rsgispy.zonalstats import polygon as zs_poly

matplotlib.use("Agg")


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


class TestVectorProcessHelpers:
    """Direct unit tests for rsgispy.vectorprocess helpers."""

    def test_write_vec_gdf_replace_layer_preserves_siblings(self, tmp_path):
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

    def test_read_vec_info_resolve_layer_and_driver_helpers(self, tmp_path):
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
        atts_df = vp.build_atts_dataframe(
            2,
            atts={"k": ["10", "20"]},
            att_types={"names": ["k"], "types": [int]},
        )
        assert list(atts_df["k"]) == [10, 20]


class TestVectorFixes:
    """Tests covering all rsgispy.vector review fixes and direct helpers."""

    def test_custom_geometry_column_name_across_vector_ops(self):
        """Functions respect gdf.geometry.name when geometry column is not 'geometry'."""
        poly_gdf = gpd.GeoDataFrame(
            {"grp": ["a", "b"], "val": [1, 2]},
            geometry=[box(0, 0, 10, 10), box(20, 0, 30, 10)],
            crs="EPSG:32630",
        ).rename_geometry("geom")

        buf = vec_geoms.buffer_vec_layer_gp(poly_gdf, None, 2.0, out_vec_file=None)
        assert buf.geometry.name == "geom"
        assert all(buf.geometry.area > 100.0)

        parts = vec_io.split_by_attribute(poly_gdf, None, "grp")
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

    def test_thin_vector_geoms_existing_unq_id_and_border_dedup(self):
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

        thinned = vec_geoms.thin_vector_geoms(
            gdf, None, grid_gdf, None, out_vec_file=None
        )
        assert "index_right" not in thinned.columns
        assert "grid_extra" not in thinned.columns
        assert set(thinned.columns) == {"unq_id", "name", "geom"}
        # Border point (unq_id=100) must appear at most once
        assert list(thinned["unq_id"]).count(100) <= 1
        assert thinned["unq_id"].is_unique

    def test_io_ops_exceptions_instead_of_asserts(self, tmp_path):
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

    def test_lines_pt_step_validation_dist_and_future_coastline(self):
        """Validate pt_step > 0, accurate dist values, and robust estimate_future_coastline."""
        line_gdf = gpd.GeoDataFrame(
            {"lid": [1]},
            geometry=[LineString([(0.0, 0.0), (10.0, 0.0)])],
            crs="EPSG:32630",
        ).rename_geometry("geom")

        with pytest.raises(RSGISPyException):
            vec_geoms.create_orthg_lines(line_gdf, None, None, pt_step=0.0)
        with pytest.raises(RSGISPyException):
            vec_geoms.split_lines_to_reg_pts(line_gdf, None, None, pt_step=-5.0)

        pts_gdf = vec_geoms.split_lines_to_reg_pts(
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
        fut = vec_geoms.estimate_future_coastline(
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

    def test_selection_fixes_crs_ckdtree_iloc_na_regex_and_empty_roi(self, tmp_path):
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

    def test_create_fixes_and_direct_helpers(self, tmp_path):
        """Test polygonise replace_lyr, define_grid validation, create_bbox_vec_lyr,
        create_poly_vec_bboxs, and create_hex_grid_poly.
        """
        tif = _write_raster(
            str(tmp_path / "cls.tif"),
            np.array([[1, 1, 2, 2], [1, 1, 2, 2]], dtype=np.uint8),
        )
        gpkg = str(tmp_path / "poly.gpkg")
        vec_create.polygonise_raster_to_vec_lyr(
            gpkg, "cls", "GPKG", tif, replace_file=True, replace_lyr=True
        )
        with pytest.raises(RSGISPyException):
            vec_create.polygonise_raster_to_vec_lyr(
                gpkg, "cls", "GPKG", tif, replace_file=False, replace_lyr=False
            )

        # define_grid multi-UTM rejects non-multi-layer format
        with pytest.raises(RSGISPyException):
            vec_create.define_grid(
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
            vec_create.create_bbox_vec_lyr(empty_gdf, None, out_vec_file=None)

        pts_gdf = gpd.GeoDataFrame(
            {"id": [1, 2]},
            geometry=[Point(0, 0), Point(4, 2)],
            crs="EPSG:4326",
        )
        bbox_gdf = vec_create.create_bbox_vec_lyr(pts_gdf, None, out_vec_file=None)
        assert len(bbox_gdf) == 1
        assert bbox_gdf.total_bounds == pytest.approx([0.0, 0.0, 4.0, 2.0])

        # Direct test: create_poly_vec_bboxs
        bboxes_gdf = vec_create.create_poly_vec_bboxs(
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
            hex_gdf = vec_create.create_hex_grid_poly(roi_poly, hex_scale=6)
            assert hex_gdf is not None and len(hex_gdf) > 0
            assert "hex_id" in hex_gdf.columns
        else:
            with pytest.raises(RSGISPyException):
                vec_create.create_hex_grid_poly(roi_poly, hex_scale=6)

    def test_attrs_io_columns_and_stats_metrics_fixes(self, tmp_path):
        """Test geometry-free attrs reads, NaN unq_numeric, 360 deg angle, stats metrics."""
        gdf = gpd.GeoDataFrame(
            {"cls": ["a", "b", "a"], "val": [1, 2, 3]},
            geometry=[Point(0, 0), Point(1, 1), Point(2, 2)],
            crs="EPSG:4326",
        )
        gpkg = str(tmp_path / "attrs.gpkg")
        pq = str(tmp_path / "attrs.parquet")
        vp.write_vec_gdf(gdf, gpkg, "pts", out_format="GPKG", overwrite=True)
        vp.write_vec_gdf(gdf, pq, "pts", out_format="Parquet", overwrite=True)

        assert vec_attrs.read_vec_column(gpkg, "pts", "val") == [1, 2, 3]
        assert vec_attrs.read_vec_column(pq, None, "val") == [1, 2, 3]
        assert set(vec_attrs.get_unq_col_values(pq, None, "cls")) == {"a", "b"}

        csv_out = str(tmp_path / "attrs.csv")
        vec_attrs.export_vec_attrs_to_csv(
            gdf.rename_geometry("geom"), None, csv_out
        )
        csv_df = pd.read_csv(csv_out, index_col=0)
        assert list(csv_df.columns) == ["cls", "val"]

        # add_unq_numeric_col with NaN/None
        nan_gdf = gpd.GeoDataFrame(
            {"cat": ["x", None, "y", np.nan]},
            geometry=[Point(i, 0) for i in range(4)],
            crs="EPSG:4326",
        )
        num_gdf = vec_attrs.add_unq_numeric_col(
            nan_gdf, None, unq_col="cat", out_col="cat_id", out_vec_file=None
        )
        assert num_gdf["cat_id"].iloc[1] == num_gdf["cat_id"].iloc[3]
        assert num_gdf["cat_id"].iloc[0] != num_gdf["cat_id"].iloc[1]
        assert num_gdf["cat_id"].iloc[0] != num_gdf["cat_id"].iloc[2]

        # create_angle_sets includes angle == 360.0 (wraps into 0 deg sector)
        ang_gdf = gpd.GeoDataFrame(
            {"ang": [0.0, 90.0, 360.0]},
            geometry=[Point(i, 0) for i in range(3)],
            crs="EPSG:4326",
        )
        sets_gdf = vec_attrs.create_angle_sets(
            ang_gdf,
            None,
            angle_col="ang",
            start_angle=0,
            angle_set_width=90,
            out_vec_file=None,
            out_angle_set_col="ang_set",
        )
        assert sets_gdf["ang_set"].iloc[2] == sets_gdf["ang_set"].iloc[0]
        assert sets_gdf["ang_set"].iloc[2] > 0

        # _stats_metrics: null/empty geoms, duplicate index, colliding col name
        pts_null = gpd.GeoDataFrame(
            {"id": [1, 2, 3]},
            geometry=[Point(0.0, 0.0), None, Point(10.0, 0.0)],
            crs="EPSG:32630",
        )
        npts = vec_stats.calc_npts_in_radius(
            pts_null, None, radius=15.0, out_col_name="cnt", out_vec_file=None
        )
        assert npts["cnt"].iloc[0] == 1
        assert npts["cnt"].iloc[1] == 0
        assert npts["cnt"].iloc[2] == 1

        dist_ang = vec_stats.calc_vec_pt_dist_angle(
            pts_null,
            None,
            out_vec_file=None,
            x_centre=0.0,
            y_centre=0.0,
            dist_col="d",
            angle_col="a",
        )
        assert np.isnan(dist_ang["d"].iloc[1])
        assert dist_ang["d"].iloc[2] == pytest.approx(10.0)

        dup_idx_gdf = gpd.GeoDataFrame(
            {"val": [10, 20]},
            geometry=[Point(0.0, 0.0), Point(10.0, 0.0)],
            index=[5, 5],
            crs="EPSG:32630",
        )
        ref_pts = gpd.GeoDataFrame(
            {"r": [1]}, geometry=[Point(3.0, 0.0)], crs="EPSG:32630"
        )
        near_gdf = vec_stats.calc_near_dist_to_feats(
            dup_idx_gdf,
            None,
            ref_pts,
            None,
            out_col="near_d",
            out_vec_file=None,
        )
        assert list(near_gdf["near_d"]) == pytest.approx([3.0, 7.0])

        polys_c = gpd.GeoDataFrame(
            {"shared": ["p1"]},
            geometry=[box(0, 0, 10, 10)],
            crs="EPSG:32630",
        )
        pts_c = gpd.GeoDataFrame(
            {"shared": ["x1", "x2"]},
            geometry=[Point(2, 2), Point(5, 5)],
            crs="EPSG:32630",
        )
        cnt_gdf = vec_stats.count_pt_intersects(
            polys_c, None, pts_c, None, out_count_col="n_pts", out_vec_file=None
        )
        assert cnt_gdf["n_pts"].iloc[0] == 2


class TestZonalStatsFixes:
    """Tests covering zonalstats.polygon & point fixes plus direct function tests."""

    def test_polygon_empty_count_float64_custom_geom_and_direct_helpers(
        self, tmp_path
    ):
        """Empty polygon count is 0.0, float64 accuracy, custom geom col, direct APIs."""
        # Large offset float32 array to verify float64 accumulation avoids precision loss
        arr = np.full((4, 4), 1e7, dtype=np.float32)
        arr[0, 0] = 1e7 + 1.0
        arr[0, 1] = 1e7 + 3.0
        tif = _write_raster(
            str(tmp_path / "f32.tif"), arr, nodata=-9999.0
        )

        poly_valid = box(500000.0, 5199980.0, 500020.0, 5200000.0)
        poly_outside = box(600000.0, 5199980.0, 600020.0, 5200000.0)
        gdf = gpd.GeoDataFrame(
            {"name": ["in", "out"]},
            geometry=[poly_valid, poly_outside],
            crs="EPSG:32630",
        )
        res = zs_poly.calc_zonal_band_stats(
            gdf.copy(),
            tif,
            img_band=1,
            min_thres=0.0,
            max_thres=1e8,
            out_no_data_val=-1.0,
            mean_field="mean_v",
            stddev_field="std_v",
            sum_field="sum_v",
            count_field="cnt_v",
            use_pbar=False,
        )
        assert res.loc[0, "cnt_v"] == pytest.approx(4.0)
        assert res.loc[0, "sum_v"] == pytest.approx(4e7 + 4.0)
        assert res.loc[0, "mean_v"] == pytest.approx(1e7 + 1.0)
        # Empty/outside polygon reports 0.0 count and out_no_data_val for stats
        assert res.loc[1, "cnt_v"] == pytest.approx(0.0)
        assert res.loc[1, "mean_v"] == pytest.approx(-1.0)

        # Direct test: calc_zonal_poly_pts_band_stats
        pt_res = zs_poly.calc_zonal_poly_pts_band_stats(
            gdf.copy(),
            tif,
            img_band=1,
            out_field="cent_val",
            use_pbar=False,
        )
        assert pt_res.loc[0, "cent_val"] == pytest.approx(1e7)

        # Direct test: calc_zonal_band_pxl_counts
        cat_arr = np.array(
            [[1, 1, 2, 2], [1, 2, 2, 3], [3, 3, 3, 3], [0, 0, 0, 0]],
            dtype=np.uint8,
        )
        cat_tif = _write_raster(str(tmp_path / "cat.tif"), cat_arr, nodata=0)
        cnt_res = zs_poly.calc_zonal_band_pxl_counts(
            gdf.copy(),
            cat_tif,
            img_band=1,
            vals_lut={1: "c1", 2: "c2"},
            total_field="tot",
            use_pbar=False,
        )
        assert cnt_res.loc[0, "c1"] == 3
        assert cnt_res.loc[0, "c2"] == 1
        assert cnt_res.loc[0, "tot"] == 4
        assert cnt_res.loc[1, "tot"] == 0

        # export_poly_pxls_to_df rejects custom geometry column name and extracts attrs
        custom_gdf = gdf.copy().rename_geometry("geom")
        with pytest.raises(RSGISPyException):
            zs_poly.export_poly_pxls_to_df(
                custom_gdf,
                tif,
                img_bands=[1],
                attr_cols=["geom"],
                use_pbar=False,
            )
        px_df = zs_poly.export_poly_pxls_to_df(
            custom_gdf,
            tif,
            img_bands=[1],
            attr_cols=["name"],
            use_pbar=False,
        )
        assert len(px_df) == 4
        assert set(px_df["name"]) == {"in"}

    def test_point_block_grouped_sampling_semantics(self, tmp_path):
        """Block-grouped sampling matches expected values, nodata, thresholds, OOB, null."""
        arr = np.arange(32 * 32, dtype=np.float32).reshape((32, 32))
        arr[0, 0] = -9999.0  # nodata
        arr[20, 20] = 5000.0  # above max_thres
        tif = _write_raster(
            str(tmp_path / "tiled.tif"),
            arr,
            nodata=-9999.0,
            tiled=True,
            blocksize=16,
        )
        pts = gpd.GeoDataFrame(
            {"pid": [0, 1, 2, 3, 4, 5]},
            geometry=[
                Point(500005.0, 5199995.0),  # (0, 0) nodata
                Point(500015.0, 5199995.0),  # (0, 1) -> 1.0
                Point(500205.0, 5199795.0),  # (20, 20) -> 5000.0 > max_thres
                Point(500215.0, 5199795.0),  # (20, 21) -> 20*32 + 21 = 661.0
                Point(600000.0, 5200000.0),  # out of bounds
                None,  # null geometry
            ],
            crs="EPSG:32630",
        )
        out = zs_point.ext_point_band_values(
            pts,
            tif,
            img_band=1,
            min_thres=0.0,
            max_thres=2000.0,
            out_no_data_val=-1.0,
            out_field="val",
            use_pbar=False,
        )
        assert list(out["val"]) == pytest.approx(
            [-1.0, 1.0, -1.0, 661.0, -1.0, -1.0]
        )


class TestMappingFixes:
    """Tests covering all rsgispy.mapping review fixes."""

    def test_elements_geodesic_scale_and_axis_label_trans_axes(self):
        """_compute_metres_per_map_unit uses geodesic distance; add_axis_label uses transAxes."""
        # EPSG:3857 at ~60 deg N: cos(60) = 0.5 -> ~0.5 m per Web Mercator metre
        t = pyproj.Transformer.from_crs(4326, 3857, always_xy=True)
        x0, y0 = t.transform(0.0, 60.0)
        m_3857 = map_elements._compute_metres_per_map_unit(
            3857, (x0 - 1000.0, x0 + 1000.0, y0 - 1000.0, y0 + 1000.0)
        )
        assert m_3857 is not None
        assert 0.49 < m_3857 < 0.51

        # EPSG:4326 at equator: ~111319 m/deg
        m_4326_eq = map_elements._compute_metres_per_map_unit(
            4326, (-0.5, 0.5, -0.5, 0.5)
        )
        assert m_4326_eq is not None
        assert 111000.0 < m_4326_eq < 111600.0

        # EPSG:32630 (UTM): ~1.0 m/unit
        m_utm = map_elements._compute_metres_per_map_unit(
            32630, (500000.0, 501000.0, 5700000.0, 5701000.0)
        )
        assert m_utm is not None
        assert 0.99 < m_utm < 1.01

        # Invalid EPSG falls back cleanly to 1.0
        assert map_elements._compute_metres_per_map_unit(
            99999999, (0, 1, 0, 1)
        ) == pytest.approx(1.0)

        # add_axis_label uses ax.transAxes at (0.0, 1.0)
        fig, ax = plt.subplots()
        try:
            ax.set_xlim(100, 200)
            ax.set_ylim(300, 400)
            map_elements.add_axis_label(ax, "(a)")
            txt = ax.texts[-1]
            assert txt.get_position() == (0.0, 1.0)
            assert txt.get_transform() == ax.transAxes
        finally:
            plt.close(fig)

    def test_legend_normalise_rgba_scan_present_and_resolve_loc(self, tmp_path):
        """scale_255 flag, tiled _scan_present_values, and resolve_mpl_location."""
        assert map_legend_internal._normalise_rgba(
            (1, 0, 0, 1), scale_255=False
        ) == pytest.approx((1.0, 0.0, 0.0, 1.0))
        assert map_legend_internal._normalise_rgba(
            (1, 0, 0, 1), scale_255=True
        ) == pytest.approx((1.0 / 255.0, 0.0, 0.0, 1.0 / 255.0))

        arr = np.zeros((32, 32), dtype=np.uint8)
        arr[0:16, 0:16] = 1
        arr[16:32, 16:32] = 2
        tif = _write_raster(
            str(tmp_path / "thm.tif"), arr, tiled=True, blocksize=16
        )
        # Top-left bbox sees only 1
        present_tl = map_legend_internal._scan_present_values(
            tif, band=1, bbox=(500000.0, 500160.0, 5199840.0, 5200000.0)
        )
        assert present_tl == {1}
        # Disjoint bbox returns empty set cleanly
        present_none = map_legend_internal._scan_present_values(
            tif, band=1, bbox=(600000.0, 600160.0, 5199840.0, 5200000.0)
        )
        assert present_none == set()

        # resolve_mpl_location in add_categorical_legend
        fig, ax = plt.subplots()
        try:
            leg = map_legend.add_categorical_legend(
                ax,
                classes_info={1: "Forest", 2: "Water"},
                colour_lut={1: "green", 2: "blue"},
                overlay=True,
                loc="top left",
            )
            assert leg is not None
            with pytest.raises(RSGISPyException):
                map_legend.add_categorical_legend(
                    ax,
                    classes_info={1: "Forest"},
                    colour_lut={1: "green"},
                    overlay=True,
                    loc="nowhere",
                )
        finally:
            plt.close(fig)

    def test_raster_cmap_nodata_mask_and_vals_under_white(self, tmp_path):
        """create_raster_cmap_img_map masks nodata before auto-norm and honours vals_under_white."""
        arr = np.array(
            [[-9999.0, 10.0], [20.0, 30.0]], dtype=np.float32
        )
        tif = _write_raster(str(tmp_path / "cmap.tif"), arr, nodata=-9999.0)

        fig, ax = plt.subplots()
        try:
            cmap, norm = map_raster.create_raster_cmap_img_map(
                ax=ax,
                input_img=tif,
                img_band=1,
                norm_img_vals=True,
                show_scale_bar=False,
            )
            assert norm is not None
            assert norm.vmin == pytest.approx(10.0)
            assert norm.vmax == pytest.approx(30.0)
            assert cmap is not None
        finally:
            plt.close(fig)

        fig2, ax2 = plt.subplots()
        try:
            cmap2, norm2 = map_raster.create_raster_cmap_img_map(
                ax=ax2,
                input_img=tif,
                img_band=1,
                norm_img_vals=False,
                vals_under_white=True,
                show_scale_bar=False,
            )
            assert norm2 is None
            assert tuple(cmap2.get_under()) == pytest.approx((1.0, 1.0, 1.0, 1.0))
        finally:
            plt.close(fig2)

    def test_vector_subset_exact_and_optional_dep_guards(self, monkeypatch, tmp_path):
        """_subset_vec_bbox has no +/-4 buffer; optional deps guarded; WMTS XML & 1-band."""
        pts = gpd.GeoDataFrame(
            {"id": [1, 2]},
            geometry=[Point(0.5, 0.5), Point(3.0, 0.5)],
            crs="EPSG:4326",
        )
        sub = map_vector._subset_vec_bbox(pts, (0.0, 1.0, 0.0, 1.0))
        assert list(sub["id"]) == [1]

        # Guard missing mpl_scatter_density and contextily
        real_import = builtins.__import__

        def _block_opt(name, *args, **kwargs):
            if name in ("mpl_scatter_density", "contextily"):
                raise ImportError(f"No module named '{name}'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _block_opt)
        fig, ax = plt.subplots()
        try:
            with pytest.raises(RSGISPyException, match="mpl-scatter-density"):
                map_vector.create_vec_pt_density_map(
                    ax, pts, bbox=(0.0, 1.0, 0.0, 1.0)
                )
            with pytest.raises(RSGISPyException, match="contextily"):
                map_basemaps.add_contextily_basemap(
                    ax, epsg_crs=3857, cx_src="dummy"
                )
        finally:
            plt.close(fig)
        monkeypatch.setattr(builtins, "__import__", real_import)

        # _update_wmts_xml replaces existing <ZoomLevel> instead of duplicating
        in_xml = tmp_path / "wmts_in.xml"
        out_xml = tmp_path / "wmts_out.xml"
        in_xml.write_text(
            "<GDAL_WMTS>"
            "<DataWindow>"
            "<UpperLeftX>0</UpperLeftX><UpperLeftY>10</UpperLeftY>"
            "<LowerRightX>10</LowerRightX><LowerRightY>0</LowerRightY>"
            "</DataWindow>"
            "<ZoomLevel>5</ZoomLevel>"
            "</GDAL_WMTS>",
            encoding="utf-8",
        )
        map_basemaps._update_wmts_xml(
            str(in_xml), str(out_xml), (1.0, 2.0, 3.0, 4.0), wmts_zoom_level=12
        )
        xml_out = out_xml.read_text(encoding="utf-8")
        assert xml_out.count("<ZoomLevel>") == 1
        assert "<ZoomLevel>12</ZoomLevel>" in xml_out

        # create_wmts_img_map handles a 1-band raster cleanly
        tif1 = _write_raster(
            str(tmp_path / "oneband.tif"),
            np.arange(16, dtype=np.uint8).reshape((4, 4)),
            crs="EPSG:3857",
        )

        def _fake_rio_copy(src, dst, **kwargs):
            if kwargs.get("driver") == "WMTS":
                with open(dst, "w", encoding="utf-8") as xml_f:
                    xml_f.write(
                        "<GDAL_WMTS>"
                        "<DataWindow>"
                        "<UpperLeftX>0</UpperLeftX><UpperLeftY>10</UpperLeftY>"
                        "<LowerRightX>10</LowerRightX><LowerRightY>0</LowerRightY>"
                        "</DataWindow>"
                        "</GDAL_WMTS>"
                    )
            else:
                shutil.copyfile(tif1, dst)

        monkeypatch.setattr(rasterio.shutil, "copy", _fake_rio_copy)
        fig3, ax3 = plt.subplots()
        try:
            map_basemaps.create_wmts_img_map(
                ax=ax3,
                wmts_url="https://example.org/wmts",
                wmts_lyr="lyr",
                bbox=(500000.0, 500040.0, 5199960.0, 5200000.0),
                bbox_epsg=3857,
                wmts_epsg=3857,
                show_scale_bar=False,
            )
            assert len(ax3.images) == 1
            assert ax3.images[0].get_cmap().name == "gray"
        finally:
            plt.close(fig3)


class TestCLIFixesAndEntrypoints:
    """Tests covering all CLI fixes, __all__, build_*_parser, and cli_main."""

    def test_all_cli_modules_exports_parsers_and_cli_main(self, monkeypatch, capsys):
        """All 11 CLI modules export __all__, build_*_parser, main, and cli_main."""
        cli_specs = [
            (batchconvert2cog, "build_batchconvert2cog_parser"),
            (buildimglut, "build_buildimglut_parser"),
            (calcimgstats, "build_calcimgstats_parser"),
            (chkgdalfile, "build_chkgdalfile_parser"),
            (copybandnames, "build_copybandnames_parser"),
            (filehash, "build_filehash_parser"),
            (imginfo, "build_imginfo_parser"),
            (proj, "build_proj_parser"),
            (userpassfile, "build_userpassfile_parser"),
            (vectools, "build_vectools_parser"),
            (wgs84imgres, "build_wgs84imgres_parser"),
        ]
        for mod, parser_fn_name in cli_specs:
            assert hasattr(mod, "__all__")
            assert parser_fn_name in mod.__all__
            assert "main" in mod.__all__
            assert "cli_main" in mod.__all__
            parser = getattr(mod, parser_fn_name)()
            assert parser.prog.startswith("rsgis")

        # Exercise cli_main end-to-end via sys.argv on proj
        monkeypatch.setattr(sys, "argv", ["rsgisproj", "--epsg", "4326", "-f", "epsg"])
        assert proj.cli_main() == 0
        assert capsys.readouterr().out.strip() == "EPSG:4326"

    def test_batchconvert2cog_rejects_inplace_and_duplicate_names(self, tmp_path):
        """batchconvert2cog rejects in-place overwrite and duplicate output basenames."""
        dir1 = tmp_path / "d1"
        dir2 = tmp_path / "d2"
        out_dir = tmp_path / "out"
        dir1.mkdir()
        dir2.mkdir()
        img1 = _write_raster(
            str(dir1 / "same.tif"), np.ones((8, 8), dtype=np.uint8)
        )
        img2 = _write_raster(
            str(dir2 / "same.tif"), np.ones((8, 8), dtype=np.uint8)
        )

        # In-place overwrite rejected
        with pytest.raises(RSGISPyException, match="in-place"):
            batchconvert2cog.main(["-i", img1, "-o", str(dir1), "--no-pbar"])

        # Duplicate output filename across dirs rejected
        with pytest.raises(RSGISPyException, match="Duplicate output filename"):
            batchconvert2cog.main(
                ["-i", img1, img2, "-o", str(out_dir), "--no-pbar"]
            )

    def test_vectools_empty_bbox_and_json_default_str(self, tmp_path, capsys):
        """vectools handles empty layers cleanly and serialises dates in --json."""
        empty_gpkg = str(tmp_path / "empty.gpkg")
        empty_gdf = gpd.GeoDataFrame(
            {"id": pd.Series(dtype="int64")},
            geometry=gpd.GeoSeries([], crs="EPSG:4326"),
            crs="EPSG:4326",
        )
        empty_gdf.to_file(empty_gpkg, layer="empty", driver="GPKG")

        assert vectools.main(["-i", empty_gpkg]) == 0
        txt = capsys.readouterr().out
        assert "Extent    : -" in txt

        assert vectools.main(["-i", empty_gpkg, "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["bbox"] is None
        assert payload["n_features"] == 0

        # Non-JSON-native date column serialises cleanly via default=str
        date_gpkg = str(tmp_path / "dates.gpkg")
        date_gdf = gpd.GeoDataFrame(
            {"obs_date": [datetime.date(2025, 1, 15)]},
            geometry=[Point(0.0, 0.0)],
            crs="EPSG:4326",
        )
        date_gdf.to_file(date_gpkg, layer="dates", driver="GPKG")
        assert (
            vectools.main(
                ["-i", date_gpkg, "-c", "obs_date", "--unique-vals", "--json"]
            )
            == 0
        )
        d_payload = json.loads(capsys.readouterr().out)
        assert "2025-01-15" in str(d_payload["unique_values"][0])

    def test_proj_format_crs_error_handling(self):
        """_format_crs wraps CRSError and raises RSGISPyException if to_proj4() returns None."""
        fake_none = SimpleNamespace(name="NoProj4", to_proj4=lambda: None)
        with pytest.raises(RSGISPyException, match="PROJ4"):
            proj._format_crs(fake_none, "proj4")

        def _raise_crs_err(**_kwargs):
            raise pyproj.exceptions.CRSError("unsupported wkt1")

        fake_err = SimpleNamespace(name="BadWKT", to_wkt=_raise_crs_err)
        with pytest.raises(RSGISPyException, match="Could not format CRS"):
            proj._format_crs(fake_err, "wkt")

    def test_imginfo_densified_bounds_streaming_stats_and_tiled_detection(
        self, tmp_path
    ):
        """Verify densified WGS84 bounds, streaming Welford stats, and tiled detection."""
        rng = np.random.default_rng(123)
        arr = rng.normal(loc=50.0, scale=12.5, size=(64, 64)).astype(np.float32)
        arr[0:5, 0:5] = -9999.0
        tiled_tif = _write_raster(
            str(tmp_path / "tiled_stats.tif"),
            arr,
            crs="EPSG:32630",
            nodata=-9999.0,
            tiled=True,
            blocksize=16,
        )
        valid = arr[arr != -9999.0].astype(np.float64)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            info = imginfo.collect_img_info(tiled_tif, calc_stats=True)
        assert not any("is_tiled" in str(w.message) for w in caught)
        assert info["is_tiled"] is True
        assert info["wgs84_bbox"] is not None
        st = info["bands"][0]["stats"]
        assert st["min"] == pytest.approx(float(np.min(valid)), rel=1e-6)
        assert st["max"] == pytest.approx(float(np.max(valid)), rel=1e-6)
        assert st["mean"] == pytest.approx(float(np.mean(valid)), rel=1e-6)
        assert st["std"] == pytest.approx(float(np.std(valid)), rel=1e-6)
        assert st["valid_percent"] == pytest.approx(
            100.0 * valid.size / arr.size, rel=1e-6
        )

        # 256x256 tiled GTiff (blk_w == width) detected as tiled; stripped 256x256 as False
        t256 = _write_raster(
            str(tmp_path / "t256.tif"),
            np.ones((256, 256), dtype=np.uint8),
            tiled=True,
            blocksize=256,
        )
        s256 = _write_raster(
            str(tmp_path / "s256.tif"),
            np.ones((256, 256), dtype=np.uint8),
            tiled=False,
        )
        assert imginfo.collect_img_info(t256)["is_tiled"] is True
        assert imginfo.collect_img_info(s256)["is_tiled"] is False
