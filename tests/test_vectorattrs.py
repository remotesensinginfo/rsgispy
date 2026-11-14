"""
Tests for rsgispy.vectorattrs (io, columns, spatial) using geopandas.
"""

import json
import sys
import types
from unittest.mock import MagicMock, patch

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, Point, box

import rsgispy
import rsgispy.vectorattrs
import rsgispy.vectorattrs.columns as va_cols
import rsgispy.vectorattrs.io as va_io
import rsgispy.vectorattrs.spatial as va_spatial
import rsgispy.vectorstats
import rsgispy.vectorstats.metrics as vs_metrics
import rsgispy.vectorstats.variogram as vs_vario


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
    xlsx_out = str(tmp_path / "attrs.xlsx")
    pq_out = str(tmp_path / "attrs.parquet")
    va_io.export_vec_attrs_to_csv(gpkg, "polys", csv_out)
    va_io.export_vec_attrs_to_excel(gpkg, "polys", xlsx_out, out_sheet_name="Data")
    va_io.export_vec_attrs_to_parquet(gpkg, "polys", pq_out, gzip_output=False)

    assert "geometry" not in pd.read_csv(csv_out).columns
    assert "geometry" not in pd.read_excel(xlsx_out, sheet_name="Data").columns
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
    assert not hasattr(rsgispy.vectorattrs, "__all__")
    assert not hasattr(rsgispy.vectorattrs, "read_vec_column")
    assert not hasattr(rsgispy.vectorstats, "__all__")
    assert not hasattr(rsgispy.vectorstats, "calc_vec_area")

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
