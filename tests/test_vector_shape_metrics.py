"""
Tests for the polygon shape metrics in rsgispy.vector.stats.
"""

import math
import os

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box

import rsgispy
import rsgispy.vector.stats as vstats


def _metrics(geoms, **kwargs):
    return vstats.calc_shape_metrics_geoseries(geoms, **kwargs)


def test_rectangle_metrics():
    df = _metrics([box(0, 0, 4, 2)])
    row = df.iloc[0]
    assert row["area"] == pytest.approx(8.0)
    assert row["perimeter"] == pytest.approx(12.0)
    assert row["n_parts"] == 1 and row["n_holes"] == 0 and row["hole_area"] == 0
    assert row["n_vertices"] == 5
    assert row["compactness"] == pytest.approx(4 * math.pi * 8 / 144)
    assert row["shape_index"] == pytest.approx(12 / (2 * math.sqrt(math.pi * 8)))
    assert row["fractal_dim"] == pytest.approx(2 * math.log(3) / math.log(8))
    assert row["convexity"] == pytest.approx(1.0)
    assert row["solidity"] == pytest.approx(1.0)
    assert row["rectangularity"] == pytest.approx(1.0)
    assert row["mrr_length"] == pytest.approx(4.0)
    assert row["mrr_width"] == pytest.approx(2.0)
    assert row["elongation"] == pytest.approx(2.0)
    assert row["orientation"] == pytest.approx(0.0)
    # Minimum bounding circle radius = half diagonal = sqrt(5)
    assert row["roundness"] == pytest.approx(8 / (math.pi * 5))
    assert row["equiv_diameter"] == pytest.approx(2 * math.sqrt(8 / math.pi))


def test_square_fractal_and_rotated_orientation():
    sq = _metrics([box(0, 0, 10, 10)]).iloc[0]
    assert sq["fractal_dim"] == pytest.approx(1.0)
    rot = Polygon([(0, 0), (4, 4), (3, 5), (-1, 1)])  # long axis at 45 degrees
    row = _metrics([rot]).iloc[0]
    assert row["orientation"] == pytest.approx(45.0)
    assert row["elongation"] == pytest.approx(math.hypot(4, 4) / math.hypot(1, 1))


def test_circle_metrics():
    circ = Point(0, 0).buffer(10.0, 512)
    row = _metrics([circ]).iloc[0]
    assert row["compactness"] == pytest.approx(1.0, rel=1e-4)
    assert row["shape_index"] == pytest.approx(1.0, rel=1e-4)
    assert row["roundness"] == pytest.approx(1.0, rel=1e-4)
    assert row["elongation"] == pytest.approx(1.0, rel=1e-4)
    assert row["equiv_diameter"] == pytest.approx(20.0, rel=1e-4)
    assert row["rectangularity"] == pytest.approx(math.pi / 4, rel=1e-3)


def test_holes_and_multipolygon():
    holed = box(0, 0, 10, 10).difference(box(4, 4, 6, 6))
    multi = MultiPolygon(
        [
            box(0, 0, 2, 2),
            Polygon(
                [(5, 5), (9, 5), (9, 9), (5, 9)],
                [[(6, 6), (7, 6), (7, 7), (6, 7)]],
            ),
        ]
    )
    df = _metrics([holed, multi])
    assert df.loc[0, "n_holes"] == 1
    assert df.loc[0, "hole_area"] == pytest.approx(4.0)
    assert df.loc[0, "perimeter"] == pytest.approx(48.0)
    assert df.loc[0, "solidity"] == pytest.approx(96.0 / 100.0)
    assert df.loc[1, "n_parts"] == 2
    assert df.loc[1, "n_holes"] == 1
    assert df.loc[1, "hole_area"] == pytest.approx(1.0)
    assert df.loc[1, "area"] == pytest.approx(4.0 + 16.0 - 1.0)


def test_empty_invalid_and_non_polygon_give_nan():
    bowtie = Polygon([(0, 0), (2, 2), (2, 0), (0, 2)])
    geoms = [Polygon(), None, bowtie, Point(1, 1), LineString([(0, 0), (1, 1)])]
    df = _metrics(geoms)
    assert df.isna().all().all()
    # make_valid repairs the bow-tie into two triangles
    fixed = _metrics([bowtie], make_valid=True).iloc[0]
    assert fixed["area"] == pytest.approx(2.0)
    assert fixed["n_parts"] == 2


def test_metric_selection_prefix_and_index():
    gser = gpd.GeoSeries([box(0, 0, 1, 1), box(0, 0, 2, 1)], index=[10, 20])
    df = _metrics(gser, metrics=["area", "elongation"], col_prefix="shp_")
    assert list(df.columns) == ["shp_area", "shp_elongation"]
    assert list(df.index) == [10, 20]
    with pytest.raises(rsgispy.RSGISPyException, match="Unknown shape metric"):
        _metrics(gser, metrics=["nope"])


def test_calc_poly_shape_metrics_file_and_crs(tmp_path):
    gdf = gpd.GeoDataFrame(
        {"seg_id": [1, 2]},
        geometry=[box(500000, 5200000, 500100, 5200050), box(500200, 5200000, 500250, 5200050)],
        crs="EPSG:32630",
    )
    out_vec = str(tmp_path / "shape.gpkg")
    vstats.calc_poly_shape_metrics(
        gdf, None, metrics=["area", "elongation"], out_vec_file=out_vec
    )
    assert os.path.exists(out_vec)
    res = gpd.read_file(out_vec)
    assert res["area"].tolist() == pytest.approx([5000.0, 2500.0])
    assert res["elongation"].tolist() == pytest.approx([2.0, 1.0])

    wgs = gdf.to_crs("EPSG:4326")
    with pytest.raises(rsgispy.RSGISPyException, match="geographic CRS"):
        vstats.calc_poly_shape_metrics(wgs, None, metrics=["area"])
    in_utm = vstats.calc_poly_shape_metrics(
        wgs, None, metrics=["area"], proj_epsg="utm"
    )
    assert in_utm.crs.is_geographic  # geometry unchanged
    assert in_utm["area"].tolist() == pytest.approx([5000.0, 2500.0], rel=1e-3)
    in_epsg = vstats.calc_poly_shape_metrics(
        wgs, None, metrics=["area"], proj_epsg=32630
    )
    assert in_epsg["area"].tolist() == pytest.approx([5000.0, 2500.0], rel=1e-6)
    deg = vstats.calc_poly_shape_metrics(
        wgs, None, metrics=["area"], allow_geographic=True
    )
    assert np.all(deg["area"] < 1e-3)
    with pytest.raises(rsgispy.RSGISPyException, match="already exist"):
        vstats.calc_poly_shape_metrics(in_epsg, None, metrics=["area"], proj_epsg=32630)
