"""
Tests for rsgispy.vectorattrs (io, columns, spatial) using geopandas.
"""

import importlib
import json
import os
import sys
import types
from unittest.mock import MagicMock, patch

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, Point, box

import rsgispy
import rsgispy.vector
import rsgispy.vector.attrs as va_cols
import rsgispy.vector.attrs as va_io
import rsgispy.vector.attrs as va_spatial
import rsgispy.vector.stats as vs_metrics
import rsgispy.vector.stats as vs_vario
from rsgispy import vectorprocess as vp


def _can_import(mod_name: str) -> bool:
    try:
        importlib.import_module(mod_name)
        return True
    except ImportError:
        return False


def _make_polys_gpkg(path: str, layer: str = "polys") -> None:
    gdf = gpd.GeoDataFrame(
        {
            "cls": ["Forest", "Water", "Forest"],
            "val": [10.0, 250.0, -5.0],
            "yr": [2024, 2025, 2026],
            "mo": [1, 6, 12],
            "dy": [5, 15, 25],
            "raw_str": ["Alpha-1", "Beta 2!", "Gamma  3"],
        },
        geometry=[
            box(0.0, 0.0, 10.0, 10.0),
            box(20.0, 0.0, 40.0, 10.0),
            box(50.0, 0.0, 60.0, 20.0),
        ],
        crs="EPSG:32630",
    )
    gdf.to_file(path, layer=layer, driver="GPKG")


def _make_points_gpkg(path: str, layer: str = "pts") -> None:
    gdf = gpd.GeoDataFrame(
        {
            "grp": [1, 1, 2, 2],
            "angle": [10.0, 95.0, 190.0, 275.0],
            "z": [1.2, 2.4, 3.6, 4.8],
        },
        geometry=[
            Point(5.0, 5.0),
            Point(5.0, 15.0),
            Point(25.0, 5.0),
            Point(25.0, 25.0),
        ],
        crs="EPSG:32630",
    )
    gdf.to_file(path, layer=layer, driver="GPKG")


def test_io_read_write_drop_rename_sort_export(tmp_path):
    """Test vectorattrs.io read, write, drop, rename, sort, and tabular exports."""
    gpkg = str(tmp_path / "polys.gpkg")
    _make_polys_gpkg(gpkg, "polys")

    # Also add a second layer to verify multi-layer GPKG updates preserve sibling layers
    gpd.GeoDataFrame({"k": [99]}, geometry=[Point(1, 1)], crs="EPSG:32630").to_file(
        gpkg, layer="other", driver="GPKG"
    )

    assert va_io.read_vec_column(gpkg, "polys", "cls") == ["Forest", "Water", "Forest"]
    rows = va_io.read_vec_columns(gpkg, "polys", ["cls", "val"])
    assert len(rows) == 3
    assert rows[1] == {"cls": "Water", "val": pytest.approx(250.0)}

    va_io.write_vec_column(gpkg, "polys", "score", float, [1.5, 2.5, 3.5])
    assert va_io.read_vec_column(gpkg, "polys", "score") == [1.5, 2.5, 3.5]

    # Case-insensitive overwrite on existing column & legacy OGR integer code support
    va_io.write_vec_column(gpkg, "polys", "SCORE", 0, [10, 20, 30])
    assert va_io.read_vec_column(gpkg, "polys", "score") == [10, 20, 30]
    assert len(gpd.read_file(gpkg, layer="other")) == 1

    # In-memory write_vec_column_to_layer
    mem_gdf = gpd.read_file(gpkg, layer="polys")
    va_io.write_vec_column_to_layer(mem_gdf, "flag", str, ["a", "b", "c"])
    assert mem_gdf["flag"].tolist() == ["a", "b", "c"]

    with pytest.raises(rsgispy.RSGISPyException):
        va_io.write_vec_column_to_layer(mem_gdf, "bad", int, [1, 2])
    with pytest.raises(rsgispy.RSGISPyException):
        va_io.read_vec_column(gpkg, "polys", "missing_col")

    assert set(va_io.get_unq_col_values(gpkg, "polys", "cls")) == {"Forest", "Water"}

    arr = va_io.get_vec_cols_as_array(
        gpkg, "polys", ["val", "score"], lower_limit=0.0, upper_limit=200.0
    )
    assert arr.shape == (1, 2)
    assert arr[0] == pytest.approx([10.0, 10.0])

    dropped = str(tmp_path / "dropped.gpkg")
    va_io.drop_vec_cols(gpkg, "polys", ["raw_str", "nonexistent"], dropped, "polys")
    assert "raw_str" not in gpd.read_file(dropped, layer="polys").columns

    renamed = str(tmp_path / "renamed.gpkg")
    va_io.rename_vec_cols(dropped, "polys", {"cls": "landcover"}, renamed, "polys")
    assert "landcover" in gpd.read_file(renamed, layer="polys").columns

    sorted_p = str(tmp_path / "sorted.gpkg")
    va_io.sort_vec_lyr(renamed, "polys", sorted_p, "polys", "val", False)
    assert va_io.read_vec_column(sorted_p, "polys", "val") == [250.0, 10.0, -5.0]

    # Exports
    csv_out = str(tmp_path / "attrs.csv")
    va_io.export_vec_attrs_to_csv(gpkg, "polys", csv_out)
    assert "geometry" not in pd.read_csv(csv_out).columns

    if _can_import("xlsxwriter") or _can_import("openpyxl"):
        xlsx_out = str(tmp_path / "attrs.xlsx")
        va_io.export_vec_attrs_to_excel(gpkg, "polys", xlsx_out, out_sheet_name="Data")
        assert os.path.exists(xlsx_out)
        if _can_import("openpyxl"):
            assert "geometry" not in pd.read_excel(xlsx_out, sheet_name="Data").columns

    if _can_import("pyarrow.parquet") or _can_import("fastparquet"):
        pq_out = str(tmp_path / "attrs.parquet")
        va_io.export_vec_attrs_to_parquet(gpkg, "polys", pq_out, gzip_output=False)
        assert "geometry" not in pd.read_parquet(pq_out).columns


def test_columns_creation_luts_text_dates_and_angles(tmp_path, monkeypatch):
    """Test vectorattrs.columns functions including GeoJSON format normalisation and use_pbar."""
    gpkg = str(tmp_path / "polys.gpkg")
    _make_polys_gpkg(gpkg, "polys")

    fid_out = str(tmp_path / "fid.gpkg")
    va_cols.add_fid_col(gpkg, "polys", fid_out, "polys", out_col="fid_val")
    assert va_io.read_vec_column(fid_out, "polys", "fid_val") == [1, 2, 3]

    unq_out = str(tmp_path / "unq.gpkg")
    lut_json = str(tmp_path / "lut.json")
    va_cols.add_unq_numeric_col(
        fid_out, "polys", "cls", "cls_id", unq_out, "polys", lut_json_file=lut_json
    )
    assert va_io.read_vec_column(unq_out, "polys", "cls_id") == [1, 2, 1]
    with open(lut_json, "r", encoding="utf-8") as fp:
        lut_data = json.load(fp)
    assert lut_data["val"]["Forest"] == 1
    assert lut_data["val"]["Water"] == 2

    num_out = str(tmp_path / "num.gpkg")
    va_cols.add_numeric_col(
        unq_out, "polys", "wt", num_out, "polys", out_val=5, out_col_int=True
    )
    assert va_io.read_vec_column(num_out, "polys", "wt") == [5, 5, 5]

    lut_num = str(tmp_path / "lut_num.gpkg")
    va_cols.add_numeric_col_lut(
        num_out,
        "polys",
        "cls",
        {"Forest": 1.5, "Water": 2.5},
        "cls_wt",
        lut_num,
        "polys",
        col_dtype="float",
    )
    assert va_io.read_vec_column(lut_num, "polys", "cls_wt") == [1.5, 2.5, 1.5]

    rng_out = str(tmp_path / "rng.gpkg")
    va_cols.add_numeric_col_range_lut(
        lut_num,
        "polys",
        "val",
        rng_out,
        "polys",
        "val_bin",
        {1: (0.0, 100.0), 2: (100.0, 500.0)},
    )
    assert va_io.read_vec_column(rng_out, "polys", "val_bin") == [1, 2, 0]

    lst_out = str(tmp_path / "lst.gpkg")
    va_cols.add_numeric_col_from_lst_lut(
        rng_out, "polys", "cls", [("Water", 9)], "w_id", lst_out, "polys"
    )
    assert va_io.read_vec_column(lst_out, "polys", "w_id") == [0, 9, 0]

    str_out = str(tmp_path / "str.gpkg")
    va_cols.add_string_col(lst_out, "polys", "tag", str_out, "polys", out_val="v1")
    assert va_io.read_vec_column(str_out, "polys", "tag") == ["v1", "v1", "v1"]

    str_lut_out = str(tmp_path / "str_lut.gpkg")
    va_cols.add_string_col_lut(
        str_out,
        "polys",
        "cls_id",
        {1: "Tree", 2: "H2O"},
        "cls_short",
        str_lut_out,
        "polys",
    )
    assert va_io.read_vec_column(str_lut_out, "polys", "cls_short") == ["Tree", "H2O", "Tree"]

    ran_out = str(tmp_path / "ran.gpkg")
    va_cols.add_ran_vals_cols(str_lut_out, "polys", ran_out, "polys", rnd_seed=123)
    assert len(va_io.read_vec_column(ran_out, "polys", "ran_vals")) == 3

    clean_out = str(tmp_path / "clean.geojson")
    pbar_states = []

    class _MockTqdm:
        def __init__(self, it, *args, **kwargs):
            pbar_states.append(bool(kwargs.get("disable", False)))
            self._it = iter(it)

        def __iter__(self):
            return self._it

    monkeypatch.setattr(rsgispy, "get_tqdm", lambda: _MockTqdm)

    va_cols.check_str_col(
        ran_out,
        "polys",
        "raw_str",
        clean_out,
        None,
        out_format="GEOJSON",
        rm_dashs=True,
        rm_spaces=True,
        rm_punc=True,
        use_pbar=False,
    )
    assert va_io.read_vec_column(clean_out, None, "raw_str") == ["Alpha_1", "Beta_2", "Gamma_3"]
    assert pbar_states == [True]

    repl_out = str(tmp_path / "repl.gpkg")
    va_cols.find_replace_str_vec_lyr(
        clean_out, None, repl_out, "polys", ["cls"], {"Forest": "Woodland"}
    )
    assert va_io.read_vec_column(repl_out, "polys", "cls") == [
        "Woodland",
        "Water",
        "Woodland",
    ]

    date_out = str(tmp_path / "dates.gpkg")
    va_cols.create_date_col(
        repl_out, "polys", "yr", "mo", "dy", "obs_dt", date_out, "polys"
    )
    assert len(va_io.read_vec_column(date_out, "polys", "obs_dt")) == 3

    # create_name_col + bbox columns
    bbox_out = str(tmp_path / "bbox.gpkg")
    va_spatial.add_geom_bbox_cols(date_out, "polys", bbox_out, "polys")
    name_out = str(tmp_path / "named.gpkg")
    va_cols.create_name_col(
        bbox_out,
        "polys",
        name_out,
        "polys",
        prefix="tile_",
        coords_lat_lon=False,
        zero_x_pad=3,
        zero_y_pad=3,
        use_pbar=True,
    )
    assert pbar_states[-1] is False
    assert va_io.read_vec_column(name_out, "polys", "names") == [
        "tile_E000N010",
        "tile_E020N010",
        "tile_E050N020",
    ]

    # Angle sets
    pts_gpkg = str(tmp_path / "pts.gpkg")
    _make_points_gpkg(pts_gpkg, "pts")
    ang_out = str(tmp_path / "ang_sets.gpkg")
    va_cols.create_angle_sets(pts_gpkg, "pts", "angle", 0, 90, ang_out, "pts")
    assert va_io.read_vec_column(ang_out, "pts", "angle_set") == [1, 2, 1, 2]

    orth_out = str(tmp_path / "orth_sets.gpkg")
    va_cols.create_orthogonal_angle_sets(pts_gpkg, "pts", "angle", 0, 20, orth_out, "pts")
    assert va_io.read_vec_column(orth_out, "pts", "angle_set") == [1, 2, 1, 2]


def test_spatial_attrs_and_vectorstats(tmp_path):
    """Test vectorattrs.spatial functions and vectorstats.variogram/metrics."""
    assert rsgispy.vector.__all__ == ["io", "create", "geoms", "attrs", "stats", "selection"]
    assert not hasattr(rsgispy.vector, "read_vec_column")
    assert not hasattr(rsgispy.vector, "calc_vec_area")

    polys_gpkg = str(tmp_path / "polys.gpkg")
    pts_gpkg = str(tmp_path / "pts.gpkg")
    lines_gpkg = str(tmp_path / "lines.gpkg")
    _make_polys_gpkg(polys_gpkg, "polys")
    _make_points_gpkg(pts_gpkg, "pts")
    gpd.GeoDataFrame(
        {"lid": [1]},
        geometry=[LineString([(0.0, 0.0), (0.0, 10.0), (10.0, 10.0)])],
        crs="EPSG:32630",
    ).to_file(lines_gpkg, layer="lines", driver="GPKG")

    # pop_bbox_cols (in-place)
    va_spatial.pop_bbox_cols(polys_gpkg, "polys")
    assert va_io.read_vec_column(polys_gpkg, "polys", "xmax") == [10.0, 40.0, 60.0]

    area_out = str(tmp_path / "areas.gpkg")
    vs_metrics.calc_vec_area(polys_gpkg, "polys", "area_m2", area_out, "polys")
    assert va_io.read_vec_column(area_out, "polys", "area_m2") == [100.0, 200.0, 200.0]

    len_out = str(tmp_path / "lengths.gpkg")
    vs_metrics.calc_vec_length(lines_gpkg, "lines", "len_m", len_out, "lines")
    assert va_io.read_vec_column(len_out, "lines", "len_m") == [20.0]

    dist_ang_out = str(tmp_path / "pt_dist_ang.gpkg")
    vs_metrics.calc_vec_pt_dist_angle(
        pts_gpkg, "pts", dist_ang_out, "pts", x_centre=5.0, y_centre=5.0
    )
    dists = va_io.read_vec_column(dist_ang_out, "pts", "dist")
    angles = va_io.read_vec_column(dist_ang_out, "pts", "angle")
    assert dists[0] == pytest.approx(0.0)
    assert dists[1] == pytest.approx(10.0)
    assert angles[1] == pytest.approx(0.0)

    near_out = str(tmp_path / "near.gpkg")
    vs_metrics.calc_near_dist_to_feats(pts_gpkg, "pts", pts_gpkg, "pts", near_out, "pts")
    near_dists = va_io.read_vec_column(near_out, "pts", "dist")
    assert all(d > 0.0 for d in near_dists)

    cnt_out = str(tmp_path / "cnt_pts.gpkg")
    vs_metrics.count_pt_intersects(polys_gpkg, "polys", pts_gpkg, "pts", cnt_out, "polys")
    assert va_io.read_vec_column(cnt_out, "polys", "n_points") == [1, 1, 0]

    rad_out = str(tmp_path / "rad_pts.gpkg")
    vs_metrics.calc_npts_in_radius(pts_gpkg, "pts", 12.0, rad_out, "pts")
    assert va_io.read_vec_column(rad_out, "pts", "n_pts_r") == [1, 1, 0, 0]

    sel_poly_gpkg = str(tmp_path / "sel_poly.gpkg")
    gpd.GeoDataFrame(
        {"roi": [1]}, geometry=[box(0.0, 0.0, 25.0, 15.0)], crs="EPSG:32630"
    ).to_file(sel_poly_gpkg, layer="roi", driver="GPKG")

    ann_out = str(tmp_path / "ann.gpkg")
    va_spatial.annotate_vec_selection(
        polys_gpkg, "polys", sel_poly_gpkg, "roi", ann_out, "polys"
    )
    assert va_io.read_vec_column(ann_out, "polys", "sel_feats") == [True, True, False]

    sjoin_out = str(tmp_path / "sjoin.gpkg")
    va_spatial.perform_spatial_join(
        pts_gpkg, "pts", polys_gpkg, "polys", sjoin_out, "pts", join_op="within"
    )
    joined = gpd.read_file(sjoin_out, layer="pts")
    assert len(joined) == 2

    # Test vectorstats.variogram.calc_empirical_semivariogram with mocked skgstat
    mock_skgstat = types.ModuleType("skgstat")
    mock_vario = MagicMock()
    mock_vario.get_empirical.return_value = (
        np.array([10.0, 20.0]),
        np.array([0.4, 1.1]),
    )
    mock_vario.lag_classes.return_value = [np.array([1.0]), np.array([2.0])]
    mock_skgstat.Variogram = MagicMock(return_value=mock_vario)

    with patch.dict(sys.modules, {"skgstat": mock_skgstat}):
        vario_df = vs_vario.calc_empirical_semivariogram(
            pts_gpkg,
            "pts",
            "z",
            pts_sel_col="grp",
            pts_sel_val=1,
            n_lags=2,
            normalise_vals=True,
        )
    assert list(vario_df.columns) == ["distance", "semivariance", "count"]
    assert len(vario_df) == 2

    with pytest.raises(rsgispy.RSGISPyException):
        vs_vario.calc_empirical_semivariogram(polys_gpkg, "polys", "val")


def test_vectorattrs_gdf_in_memory(tmp_path):
    """Test VectorInput GeoDataFrame input and in-memory GeoDataFrame returns in vectorattrs."""
    polys_gdf = gpd.GeoDataFrame(
        {
            "cls": ["Forest", "Water", "Forest"],
            "val": [10.0, 250.0, -5.0],
            "yr": [2024, 2025, 2026],
            "mo": [1, 6, 12],
            "dy": [5, 15, 25],
            "raw_str": ["Alpha-1", "Beta 2!", "Gamma  3"],
        },
        geometry=[
            box(0.0, 0.0, 10.0, 10.0),
            box(20.0, 0.0, 40.0, 10.0),
            box(50.0, 0.0, 60.0, 20.0),
        ],
        crs="EPSG:32630",
    )
    orig_cols = list(polys_gdf.columns)

    assert va_io.read_vec_column(polys_gdf, None, "cls") == [
        "Forest",
        "Water",
        "Forest",
    ]
    assert len(va_io.read_vec_columns(polys_gdf, None, ["cls", "val"])) == 3
    assert set(va_io.get_unq_col_values(polys_gdf, None, "cls")) == {
        "Forest",
        "Water",
    }

    written_gdf = va_io.write_vec_column(
        polys_gdf, None, "score", float, [1.5, 2.5, 3.5]
    )
    assert isinstance(written_gdf, gpd.GeoDataFrame)
    assert written_gdf["score"].tolist() == [1.5, 2.5, 3.5]
    assert list(polys_gdf.columns) == orig_cols

    dropped_gdf = va_io.drop_vec_cols(written_gdf, None, ["raw_str"])
    assert "raw_str" not in dropped_gdf.columns

    renamed_gdf = va_io.rename_vec_cols(dropped_gdf, None, {"cls": "lc"})
    assert "lc" in renamed_gdf.columns

    sorted_gdf = va_io.sort_vec_lyr(renamed_gdf, None, None, None, "val", False)
    assert sorted_gdf["val"].tolist() == [250.0, 10.0, -5.0]

    csv_out = str(tmp_path / "gdf_attrs.csv")
    va_io.export_vec_attrs_to_csv(polys_gdf, None, csv_out)
    assert os.path.exists(csv_out)

    # vectorattrs.columns in-memory chaining
    fid_gdf = va_cols.add_fid_col(polys_gdf, None, out_col="fid_val")
    assert fid_gdf["fid_val"].tolist() == [1, 2, 3]
    assert list(polys_gdf.columns) == orig_cols

    unq_gdf = va_cols.add_unq_numeric_col(fid_gdf, None, "cls", "cls_id")
    assert unq_gdf["cls_id"].tolist() == [1, 2, 1]

    num_gdf = va_cols.add_numeric_col(unq_gdf, None, "wt", out_val=5)
    assert num_gdf["wt"].tolist() == [5, 5, 5]

    lut_gdf = va_cols.add_numeric_col_lut(
        num_gdf, None, "cls", {"Forest": 1.5, "Water": 2.5}, "cls_wt", col_dtype="float"
    )
    assert lut_gdf["cls_wt"].tolist() == [1.5, 2.5, 1.5]

    rng_gdf = va_cols.add_numeric_col_range_lut(
        lut_gdf, None, "val", None, None, "val_bin", {1: (0.0, 100.0), 2: (100.0, 500.0)}
    )
    assert rng_gdf["val_bin"].tolist() == [1, 2, 0]

    str_gdf = va_cols.add_string_col(rng_gdf, None, "tag", out_val="v1")
    assert str_gdf["tag"].tolist() == ["v1", "v1", "v1"]

    repl_gdf = va_cols.find_replace_str_vec_lyr(
        str_gdf, None, None, None, ["cls"], {"Forest": "Woodland"}
    )
    assert repl_gdf["cls"].tolist() == ["Woodland", "Water", "Woodland"]

    date_gdf = va_cols.create_date_col(repl_gdf, None, "yr", "mo", "dy", "obs_dt")
    assert len(date_gdf["obs_dt"]) == 3

    # vectorattrs.spatial in-memory
    bbox_gdf = va_spatial.add_geom_bbox_cols(polys_gdf, None)
    assert bbox_gdf["MaxX"].tolist() == [10.0, 40.0, 60.0]
    assert list(polys_gdf.columns) == orig_cols

    pop_res = va_spatial.pop_bbox_cols(polys_gdf, None)
    assert isinstance(pop_res, gpd.GeoDataFrame)
    assert pop_res["xmax"].tolist() == [10.0, 40.0, 60.0]

    roi_gdf = gpd.GeoDataFrame(
        {"roi": [1]}, geometry=[box(0.0, 0.0, 25.0, 15.0)], crs="EPSG:32630"
    )
    ann_gdf = va_spatial.annotate_vec_selection(polys_gdf, None, roi_gdf, None)
    assert ann_gdf["sel_feats"].tolist() == [True, True, False]

    pts_gdf = gpd.GeoDataFrame(
        {"pid": [1, 2]},
        geometry=[Point(5.0, 5.0), Point(25.0, 5.0)],
        crs="EPSG:32630",
    )
    sjoin_gdf = va_spatial.perform_spatial_join(
        pts_gdf, None, polys_gdf, None, join_op="within"
    )
    assert len(sjoin_gdf) == 2

    # vectorstats.metrics & vectorstats.variogram in-memory
    area_gdf = vs_metrics.calc_vec_area(polys_gdf, None, "area_m2")
    assert isinstance(area_gdf, gpd.GeoDataFrame)
    assert area_gdf["area_m2"].tolist() == [100.0, 200.0, 200.0]
    assert list(polys_gdf.columns) == orig_cols

    lines_gdf = gpd.GeoDataFrame(
        {"lid": [1]},
        geometry=[LineString([(0.0, 0.0), (0.0, 10.0), (10.0, 10.0)])],
        crs="EPSG:32630",
    )
    len_gdf = vs_metrics.calc_vec_length(lines_gdf, None, "len_m")
    assert len_gdf["len_m"].tolist() == [20.0]

    dist_ang_gdf = vs_metrics.calc_vec_pt_dist_angle(
        pts_gdf, None, x_centre=5.0, y_centre=5.0
    )
    assert dist_ang_gdf["dist"].iloc[0] == pytest.approx(0.0)
    assert dist_ang_gdf["dist"].iloc[1] == pytest.approx(20.0)

    near_gdf = vs_metrics.calc_near_dist_to_feats(
        pts_gdf, None, pts_gdf, None
    )
    assert all(near_gdf["dist"] > 0.0)

    cnt_gdf = vs_metrics.count_pt_intersects(polys_gdf, None, pts_gdf, None)
    assert cnt_gdf["n_points"].tolist() == [1, 1, 0]

    if vs_metrics.HAVE_SCIPY:
        rad_gdf = vs_metrics.calc_npts_in_radius(pts_gdf, None, 25.0)
        assert rad_gdf["n_pts_r"].tolist() == [1, 1]

    vario_pts_gdf = gpd.GeoDataFrame(
        {"z": [1.0, 2.0, 4.0, 8.0]},
        geometry=[Point(0, 0), Point(10, 0), Point(20, 0), Point(30, 0)],
        crs="EPSG:32630",
    )
    mock_skgstat = types.ModuleType("skgstat")
    mock_vario = MagicMock()
    mock_vario.get_empirical.return_value = (
        np.array([10.0, 20.0]),
        np.array([0.4, 1.1]),
    )
    mock_vario.lag_classes.return_value = [np.array([1.0]), np.array([2.0])]
    mock_skgstat.Variogram = MagicMock(return_value=mock_vario)
    with patch.dict(sys.modules, {"skgstat": mock_skgstat}):
        vario_df = vs_vario.calc_empirical_semivariogram(
            vario_pts_gdf, None, "z", n_lags=2
        )
    assert len(vario_df) == 2


def test_vector_maths(tmp_path):
    """Test vector_maths with pymu-eval expressions, mappings, dtypes, and file output."""
    gdf = gpd.GeoDataFrame(
        {
            "nir": [0.8, 0.6, 0.0, np.nan],
            "red": [0.2, 0.2, 0.0, 0.4],
            "cls": ["Forest", "Grass", "Bare", "Water"],
            "flag": pd.Series([True, False, True, False], dtype="boolean"),
            "nullable_i": pd.Series([2, pd.NA, 4, 6], dtype="Int64"),
            "Band 1": [10.0, 20.0, 30.0, 40.0],
        },
        geometry=[Point(i, i) for i in range(4)],
        crs="EPSG:32630",
    )
    orig_cols = list(gdf.columns)

    # 1. Auto-discovered numeric columns + ternary conditional (in-memory, no mutation)
    ndvi_gdf = va_cols.vector_maths(
        gdf,
        None,
        out_col="ndvi",
        exp="(nir + red) != 0 ? (nir - red) / (nir + red) : -9999",
    )
    assert list(gdf.columns) == orig_cols
    assert ndvi_gdf is not None
    assert ndvi_gdf["ndvi"].iloc[0] == pytest.approx(0.6)
    assert ndvi_gdf["ndvi"].iloc[1] == pytest.approx(0.5)
    assert ndvi_gdf["ndvi"].iloc[2] == pytest.approx(-9999.0)
    assert np.isnan(ndvi_gdf["ndvi"].iloc[3])

    # 2. Aliased mapping for non-identifier column names + pi constant
    alias_gdf = va_cols.vector_maths(
        gdf,
        None,
        out_col="scaled",
        exp="b1 * 2 + flag + pi",
        vec_cols={"b1": "Band 1", "flag": "flag"},
    )
    assert alias_gdf["scaled"].iloc[0] == pytest.approx(20.0 + 1.0 + np.pi)
    assert alias_gdf["scaled"].iloc[1] == pytest.approx(40.0 + 0.0 + np.pi)

    # 3. Explicit column sequence + constant scalar expression broadcast + out_dtype
    const_gdf = va_cols.vector_maths(
        gdf,
        None,
        out_col="const_int",
        exp="rint(2 * pi)",
        vec_cols=["nir"],
        out_dtype=int,
    )
    assert list(const_gdf["const_int"]) == [6, 6, 6, 6]
    assert np.issubdtype(const_gdf["const_int"].dtype, np.integer)

    f32_gdf = va_cols.vector_maths(
        gdf,
        None,
        out_col="nir",
        exp="nullable_i * 1.5",
        out_dtype=np.float32,
    )
    assert f32_gdf["nir"].dtype == np.float32
    assert f32_gdf["nir"].iloc[0] == pytest.approx(3.0)
    assert np.isnan(f32_gdf["nir"].iloc[1])

    nan_int_gdf = va_cols.vector_maths(
        gdf,
        None,
        out_col="int_with_na",
        exp="nullable_i + 1",
        out_dtype=int,
    )
    assert nan_int_gdf["int_with_na"].iloc[0] == 3
    assert pd.isna(nan_int_gdf["int_with_na"].iloc[1])

    # 4. File input/output (GPKG) & del_exist_vec=True
    in_gpkg = str(tmp_path / "in.gpkg")
    out_gpkg = str(tmp_path / "out.gpkg")
    gdf.to_file(in_gpkg, layer="pts", driver="GPKG")

    ret = va_cols.vector_maths(
        in_gpkg,
        "pts",
        out_col="sum_nr",
        exp="nir + red",
        out_vec_file=out_gpkg,
        out_vec_lyr="pts",
    )
    assert ret is None
    assert va_io.read_vec_column(out_gpkg, "pts", "sum_nr")[:3] == pytest.approx(
        [1.0, 0.8, 0.0]
    )

    with pytest.raises(rsgispy.RSGISPyException):
        va_cols.vector_maths(
            in_gpkg,
            "pts",
            out_col="sum_nr",
            exp="nir + red",
            out_vec_file=out_gpkg,
            out_vec_lyr="pts",
            del_exist_vec=False,
        )
    va_cols.vector_maths(
        in_gpkg,
        "pts",
        out_col="sum_nr",
        exp="nir - red",
        out_vec_file=out_gpkg,
        out_vec_lyr="pts",
        del_exist_vec=True,
    )
    assert va_io.read_vec_column(out_gpkg, "pts", "sum_nr")[:3] == pytest.approx(
        [0.6, 0.4, 0.0]
    )

    # 5. Empty GeoDataFrame
    empty_gdf = gdf.iloc[0:0].copy()
    empty_out = va_cols.vector_maths(
        empty_gdf, None, out_col="calc", exp="nir + red"
    )
    assert len(empty_out) == 0
    assert "calc" in empty_out.columns

    # 6. Validation errors
    with pytest.raises(rsgispy.RSGISPyException, match="out_col"):
        va_cols.vector_maths(gdf, None, out_col="", exp="nir + red")
    with pytest.raises(rsgispy.RSGISPyException, match="exp"):
        va_cols.vector_maths(gdf, None, out_col="x", exp="   ")
    with pytest.raises(rsgispy.RSGISPyException, match="geometry"):
        va_cols.vector_maths(gdf, None, out_col="geometry", exp="nir + red")
    with pytest.raises(rsgispy.RSGISPyException, match="not numeric"):
        va_cols.vector_maths(
            gdf, None, out_col="x", exp="cls * 2", vec_cols=["cls"]
        )
    with pytest.raises(rsgispy.RSGISPyException, match="could not be bound"):
        va_cols.vector_maths(gdf, None, out_col="x", exp="cls * 2")
    with pytest.raises(rsgispy.RSGISPyException, match="out_dtype"):
        va_cols.vector_maths(
            gdf, None, out_col="x", exp="nir + red", out_dtype="complex128"
        )


def test_vector_maths_multi_output(tmp_path):
    """Test multi-output comma-separated pymu-eval expressions in vector_maths."""
    gdf = gpd.GeoDataFrame(
        {
            "blue": [0.1, 0.2, 0.0, 0.3],
            "green": [0.2, 0.3, 0.0, 0.2],
            "red": [0.2, 0.2, 0.0, 0.4],
            "nir": [0.8, 0.6, 0.0, 0.2],
            "Band 4": [8000.0, 6000.0, 4000.0, 2000.0],
            "Band 3": [2000.0, 2000.0, 2000.0, 2000.0],
        },
        geometry=[Point(i, i) for i in range(4)],
        crs="EPSG:32630",
    )

    # 1. Multi-line multi-output with conditionals, multi-arg functions, constant,
    #    and per-column out_dtype sequence
    multi_exp = """
        (nir + red) != 0 ? (nir - red) / (nir + red) : -9999,
        min(blue, green, red),
        avg(blue, green, red),
        (nir > red) ? 1 : 0,
        2 * pi
    """
    out_gdf = va_cols.vector_maths(
        gdf,
        None,
        out_col=["ndvi", "vis_min", "vis_avg", "is_veg", "two_pi"],
        exp=multi_exp,
        out_dtype=[np.float32, np.float32, float, int, None],
    )
    assert out_gdf is not None
    assert out_gdf["ndvi"].dtype == np.float32
    assert out_gdf["vis_min"].dtype == np.float32
    assert out_gdf["vis_avg"].dtype == np.float64
    assert np.issubdtype(out_gdf["is_veg"].dtype, np.integer)
    assert out_gdf["ndvi"].tolist() == pytest.approx([0.6, 0.5, -9999.0, -1.0 / 3.0], rel=1e-5)
    assert out_gdf["vis_min"].tolist() == pytest.approx([0.1, 0.2, 0.0, 0.2], rel=1e-5)
    assert out_gdf["vis_avg"].tolist() == pytest.approx(
        [0.5 / 3.0, 0.7 / 3.0, 0.0, 0.3], rel=1e-5
    )
    assert out_gdf["is_veg"].tolist() == [1, 1, 0, 0]
    assert out_gdf["two_pi"].tolist() == pytest.approx([2.0 * np.pi] * 4)

    # 2. Aliased columns + in-expression variable re-assignment + out_dtype mapping
    reassigned = va_cols.vector_maths(
        gdf,
        None,
        out_col=["nir_refl", "red_refl", "ndvi_scaled"],
        exp="nir = nir * 0.0001, red = red * 0.0001, (nir - red) / (nir + red)",
        vec_cols={"nir": "Band 4", "red": "Band 3"},
        out_dtype={"ndvi_scaled": np.float32},
    )
    assert reassigned["nir_refl"].tolist() == pytest.approx([0.8, 0.6, 0.4, 0.2])
    assert reassigned["red_refl"].tolist() == pytest.approx([0.2, 0.2, 0.2, 0.2])
    assert reassigned["ndvi_scaled"].dtype == np.float32
    assert reassigned["ndvi_scaled"].tolist() == pytest.approx(
        [0.6, 0.5, 1.0 / 3.0, 0.0], rel=1e-5
    )

    # 3. Single scalar out_dtype applied across multiple outputs + GPKG file write
    out_gpkg = str(tmp_path / "multi_cols.gpkg")
    ret = va_cols.vector_maths(
        gdf,
        None,
        out_col=["sum_nr", "diff_nr"],
        exp="nir + red, nir - red",
        out_vec_file=out_gpkg,
        out_vec_lyr="pts",
        out_dtype=np.float32,
    )
    assert ret is None
    assert va_io.read_vec_column(out_gpkg, "pts", "sum_nr") == pytest.approx(
        [1.0, 0.8, 0.0, 0.6], rel=1e-5
    )
    assert va_io.read_vec_column(out_gpkg, "pts", "diff_nr") == pytest.approx(
        [0.6, 0.4, 0.0, -0.2], rel=1e-5
    )

    # 4. Empty GeoDataFrame with multiple outputs
    empty_out = va_cols.vector_maths(
        gdf.iloc[0:0].copy(),
        None,
        out_col=["c1", "c2"],
        exp="nir + red, 42",
    )
    assert len(empty_out) == 0
    assert "c1" in empty_out.columns and "c2" in empty_out.columns

    # 5. Validation errors for multi-output configuration
    with pytest.raises(rsgispy.RSGISPyException, match="produced 2 output"):
        va_cols.vector_maths(gdf, None, out_col="only_one", exp="nir + red, nir - red")
    with pytest.raises(rsgispy.RSGISPyException, match="produced 1 output"):
        va_cols.vector_maths(gdf, None, out_col=["c1", "c2"], exp="nir + red")
    with pytest.raises(rsgispy.RSGISPyException, match="Duplicate"):
        va_cols.vector_maths(gdf, None, out_col=["c1", "c1"], exp="nir + red, nir - red")
    with pytest.raises(rsgispy.RSGISPyException, match="out_col"):
        va_cols.vector_maths(gdf, None, out_col=[], exp="nir + red")
    with pytest.raises(rsgispy.RSGISPyException, match="geometry"):
        va_cols.vector_maths(
            gdf, None, out_col=["ok", "geometry"], exp="nir + red, nir - red"
        )
    with pytest.raises(rsgispy.RSGISPyException, match="Length of out_dtype"):
        va_cols.vector_maths(
            gdf,
            None,
            out_col=["c1", "c2"],
            exp="nir + red, nir - red",
            out_dtype=[float],
        )
    with pytest.raises(rsgispy.RSGISPyException, match="Unknown column name"):
        va_cols.vector_maths(
            gdf,
            None,
            out_col=["c1", "c2"],
            exp="nir + red, nir - red",
            out_dtype={"missing": float},
        )


def test_vector_maths_intermediate_variables():
    """Test creating and reusing arbitrary intermediate variables in vector_maths."""
    gdf = gpd.GeoDataFrame(
        {
            "red": [0.1, 0.4, 0.2, 0.3],
            "nir": [0.8, 0.5, 0.6, 0.3],
            "swir1": [0.2, 0.3, 0.2, 0.1],
            "cls": ["Forest", "Grass", "Shrub", "Water"],
        },
        geometry=[Point(i, i) for i in range(4)],
        crs="EPSG:32630",
    )

    # 1. Multi-output expression creating `ndvi` and reusing it in a later column
    out_gdf = va_cols.vector_maths(
        gdf,
        None,
        out_col=["ndvi", "ndmi", "is_veg"],
        exp=(
            "ndvi = (nir - red) / (nir + red + 1e-6), "
            "(nir - swir1) / (nir + swir1 + 1e-6), "
            "ndvi >= 0.5 ? 1 : 0"
        ),
        out_dtype={"is_veg": int},
    )
    red_arr = gdf["red"].to_numpy()
    nir_arr = gdf["nir"].to_numpy()
    swir_arr = gdf["swir1"].to_numpy()
    exp_ndvi = (nir_arr - red_arr) / (nir_arr + red_arr + 1e-6)
    exp_ndmi = (nir_arr - swir_arr) / (nir_arr + swir_arr + 1e-6)
    exp_mask = np.where(exp_ndvi >= 0.5, 1, 0)

    assert out_gdf["ndvi"].to_numpy() == pytest.approx(exp_ndvi)
    assert out_gdf["ndmi"].to_numpy() == pytest.approx(exp_ndmi)
    assert out_gdf["is_veg"].tolist() == exp_mask.tolist()

    # 2. Inline parenthesized assignment producing a single output column & scalar intermediate
    single_gdf = va_cols.vector_maths(
        gdf,
        None,
        out_col="veg_mask",
        exp="(ndvi = (nir - red) / (nir + red + 1e-6)) >= 0.5 ? 1 : 0",
        out_dtype=int,
    )
    assert single_gdf["veg_mask"].tolist() == exp_mask.tolist()

    scalar_gdf = va_cols.vector_maths(
        gdf,
        None,
        out_col=["gain", "scaled_nir"],
        exp="k = 0.5, nir * k",
    )
    assert scalar_gdf["gain"].tolist() == pytest.approx([0.5] * 4)
    assert scalar_gdf["scaled_nir"].to_numpy() == pytest.approx(nir_arr * 0.5)

    # 3. Empty GeoDataFrame with intermediate variables
    empty_out = va_cols.vector_maths(
        gdf.iloc[0:0].copy(),
        None,
        out_col=["ndvi", "is_veg"],
        exp="ndvi = (nir - red) / (nir + red + 1e-6), ndvi >= 0.5 ? 1 : 0",
        out_dtype={"is_veg": int},
    )
    assert len(empty_out) == 0
    assert "ndvi" in empty_out.columns and "is_veg" in empty_out.columns

    # 4. Undefined RHS variable, skipped non-numeric column, read-before-assign,
    #    and initial self-reference must still raise RSGISPyException
    with pytest.raises(rsgispy.RSGISPyException):
        va_cols.vector_maths(
            gdf, None, out_col=["a", "b"], exp="ndvi = missing_col * 2, ndvi + 1"
        )
    with pytest.raises(rsgispy.RSGISPyException, match="could not be bound"):
        va_cols.vector_maths(
            gdf, None, out_col=["a", "b"], exp="tmp = cls * 2, tmp + 1"
        )
    with pytest.raises(rsgispy.RSGISPyException):
        va_cols.vector_maths(
            gdf, None, out_col=["a", "b"], exp="ndvi + 1, ndvi = nir - red"
        )
    with pytest.raises(rsgispy.RSGISPyException):
        va_cols.vector_maths(gdf, None, out_col="a", exp="ndvi = ndvi + 1")


def test_vector_joins_and_counts_align_crs():
    """H5: secondary layers in another CRS are reprojected before joining."""
    polys_utm = gpd.GeoDataFrame(
        {"pid": [1, 2]},
        geometry=[
            box(500000.0, 5800000.0, 501000.0, 5801000.0),
            box(502000.0, 5800000.0, 503000.0, 5801000.0),
        ],
        crs="EPSG:32630",
    )
    pts_utm = gpd.GeoDataFrame(
        {"val": [10, 20, 30]},
        geometry=[
            Point(500500.0, 5800500.0),
            Point(500600.0, 5800600.0),
            Point(502500.0, 5800500.0),
        ],
        crs="EPSG:32630",
    )
    pts_wgs = pts_utm.to_crs("EPSG:4326")

    cnt_gdf = vs_metrics.count_pt_intersects(polys_utm, None, pts_wgs, None)
    assert cnt_gdf.sort_values("pid")["n_points"].tolist() == [2, 1]

    ann_gdf = va_spatial.annotate_vec_selection(polys_utm, None, pts_wgs, None)
    assert ann_gdf["sel_feats"].tolist() == [True, True]

    sjoin_gdf = va_spatial.perform_spatial_join(
        pts_utm, None, polys_utm.to_crs("EPSG:4326"), None, join_op="within"
    )
    assert sorted(sjoin_gdf["pid"].tolist()) == [1, 1, 2]
    assert sjoin_gdf.crs == pts_utm.crs

    near_gdf = vs_metrics.calc_near_dist_to_feats(
        pts_utm.iloc[[0]], None, polys_utm.iloc[[1]].to_crs("EPSG:4326"), None
    )
    # Distance from x=500500 to the second polygon's edge at x=502000, in metres.
    assert near_gdf["dist"].iloc[0] == pytest.approx(1500.0, abs=1.0)


def test_in_place_column_writes_keep_layer_name(tmp_path):
    """H7: in-place writes with vec_lyr=None update the existing layer."""
    vec_path = str(tmp_path / "data.gpkg")
    _make_polys_gpkg(vec_path, layer="polys")
    n_feats = len(gpd.read_file(vec_path, layer="polys"))

    va_spatial.pop_bbox_cols(vec_path, None)
    va_io.write_vec_column(vec_path, None, "newcol", int, list(range(n_feats)))
    va_io.write_vec_column_to_layer(
        vec_path, "other", float, [1.5] * n_feats
    )

    assert rsgispy.vectorprocess.list_vec_layers(vec_path) == ["polys"]
    out_gdf = gpd.read_file(vec_path, layer="polys")
    for col in ("xmin", "newcol", "other"):
        assert col in out_gdf.columns


def test_attrs_geometry_free_reads_columns_and_stats_metrics(tmp_path):
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

    assert va_cols.read_vec_column(gpkg, "pts", "val") == [1, 2, 3]
    assert va_cols.read_vec_column(pq, None, "val") == [1, 2, 3]
    assert set(va_cols.get_unq_col_values(pq, None, "cls")) == {"a", "b"}

    csv_out = str(tmp_path / "attrs.csv")
    va_cols.export_vec_attrs_to_csv(
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
    num_gdf = va_cols.add_unq_numeric_col(
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
    sets_gdf = va_cols.create_angle_sets(
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
    npts = vs_metrics.calc_npts_in_radius(
        pts_null, None, radius=15.0, out_col_name="cnt", out_vec_file=None
    )
    assert npts["cnt"].iloc[0] == 1
    assert npts["cnt"].iloc[1] == 0
    assert npts["cnt"].iloc[2] == 1

    dist_ang = vs_metrics.calc_vec_pt_dist_angle(
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
    near_gdf = vs_metrics.calc_near_dist_to_feats(
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
    cnt_gdf = vs_metrics.count_pt_intersects(
        polys_c, None, pts_c, None, out_count_col="n_pts", out_vec_file=None
    )
    assert cnt_gdf["n_pts"].iloc[0] == 2
