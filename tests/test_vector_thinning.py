"""
Tests for spatial thinning (rarefaction) of point samples:
:func:`rsgispy.vector.geoms.thin_pts_min_dist`,
:func:`rsgispy.vector.geoms.thin_pts_by_raster_grid`, the ``keep`` option of
:func:`rsgispy.vector.geoms.thin_vector_geoms`, and
:func:`rsgispy.modelling.species_distribution.thin_pts_env_space`.
"""

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point, box

import rsgispy
import rsgispy.modelling.species_distribution as sdm
import rsgispy.vector.geoms as vgeoms

from conftest import write_raster

CRS = "EPSG:32630"


def _clustered_pts(n_clusters=6, n_per=15, spread=40.0, seed=1) -> gpd.GeoDataFrame:
    """Tight clusters of points with an ``id`` and ``year`` attribute."""
    rng = np.random.default_rng(seed)
    centres = rng.uniform(0.0, 5000.0, size=(n_clusters, 2))
    xy = np.vstack([c + rng.normal(0.0, spread, size=(n_per, 2)) for c in centres])
    n_pts = xy.shape[0]
    return gpd.GeoDataFrame(
        {"id": np.arange(n_pts), "year": rng.integers(1990, 2025, n_pts)},
        geometry=gpd.points_from_xy(xy[:, 0], xy[:, 1]),
        crs=CRS,
    )


def _min_pair_dist(gdf: gpd.GeoDataFrame) -> float:
    xy = np.column_stack([gdf.geometry.x, gdf.geometry.y])
    diff = xy[:, None, :] - xy[None, :, :]
    dist = np.hypot(diff[..., 0], diff[..., 1])
    np.fill_diagonal(dist, np.inf)
    return float(dist.min())


def _write_ref_img(path, n_rows=10, n_cols=10, res=100.0, x0=0.0, y0=1000.0):
    return write_raster(
        path,
        np.ones((n_rows, n_cols), dtype=np.uint8),
        crs=CRS,
        origin=(x0, y0),
        pixel_size=res,
    )


# --------------------------------------------------------------------------
# thin_pts_min_dist
# --------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["random", "greedy"])
def test_min_dist_guarantee_and_attributes(method):
    pts = _clustered_pts()
    out = vgeoms.thin_pts_min_dist(pts, None, 100.0, method=method, rnd_seed=3)
    assert 0 < len(out) < len(pts)
    assert _min_pair_dist(out) >= 100.0
    assert list(out.columns) == list(pts.columns)
    # Retained rows keep their input order and attributes.
    assert list(out["id"]) == sorted(out["id"])
    pd_pts = pts.set_index("id")
    assert (pd_pts.loc[out["id"], "year"].to_numpy() == out["year"].to_numpy()).all()


def test_min_dist_exact_distance_pairs_are_kept():
    pts = gpd.GeoDataFrame(
        geometry=[Point(0, 0), Point(100, 0), Point(200, 0)], crs=CRS
    )
    out = vgeoms.thin_pts_min_dist(pts, None, 100.0, method="greedy")
    assert len(out) == 3


def test_min_dist_reproducible_and_best_of_reps():
    pts = _clustered_pts(n_clusters=10, n_per=20, spread=80.0)
    out_a, reps_a = vgeoms.thin_pts_min_dist(
        pts, None, 60.0, n_reps=15, rnd_seed=7, return_reps=True
    )
    out_b = vgeoms.thin_pts_min_dist(pts, None, 60.0, n_reps=15, rnd_seed=7)
    assert list(out_a["id"]) == list(out_b["id"])
    assert len(reps_a) == 15
    sizes = [r.size for r in reps_a]
    assert len(out_a) == max(sizes)
    # Ties resolve to the first replicate with the maximum size.
    first_best = reps_a[sizes.index(max(sizes))]
    assert list(out_a["id"]) == list(pts["id"].iloc[first_best])
    for rep in reps_a:
        assert _min_pair_dist(pts.iloc[rep]) >= 60.0


def test_min_dist_greedy_keeps_at_least_single_random_rep():
    pts = _clustered_pts(n_clusters=8, n_per=25, spread=60.0, seed=4)
    greedy = vgeoms.thin_pts_min_dist(pts, None, 50.0, method="greedy")
    one_rep = vgeoms.thin_pts_min_dist(pts, None, 50.0, n_reps=1, rnd_seed=0)
    assert len(greedy) >= len(one_rep)
    # Greedy is deterministic.
    greedy_2 = vgeoms.thin_pts_min_dist(pts, None, 50.0, method="greedy")
    assert list(greedy["id"]) == list(greedy_2["id"])


def test_min_dist_polygons_use_centroids():
    polys = gpd.GeoDataFrame(
        geometry=[box(0, 0, 10, 10), box(2, 2, 12, 12), box(500, 500, 510, 510)],
        crs=CRS,
    )
    out = vgeoms.thin_pts_min_dist(polys, None, 50.0, method="greedy")
    assert len(out) == 2
    assert out.geom_type.eq("Polygon").all()


def test_min_dist_file_round_trip(tmp_path):
    pts = _clustered_pts()
    in_file = str(tmp_path / "pts.gpkg")
    pts.to_file(in_file, layer="pts", driver="GPKG")
    out_file = str(tmp_path / "thin.gpkg")
    res = vgeoms.thin_pts_min_dist(in_file, "pts", 100.0, out_file, "thin", rnd_seed=1)
    assert res is None
    mem = vgeoms.thin_pts_min_dist(in_file, "pts", 100.0, rnd_seed=1)
    disk = gpd.read_file(out_file, layer="thin")
    assert list(disk["id"]) == list(mem["id"])


def test_min_dist_empty_and_validation():
    empty = gpd.GeoDataFrame({"id": []}, geometry=[], crs=CRS)
    out = vgeoms.thin_pts_min_dist(empty, None, 10.0)
    assert len(out) == 0
    pts = _clustered_pts()
    with pytest.raises(rsgispy.RSGISPyException):
        vgeoms.thin_pts_min_dist(pts, None, 0.0)
    with pytest.raises(rsgispy.RSGISPyException):
        vgeoms.thin_pts_min_dist(pts, None, 10.0, method="bogus")
    with pytest.raises(rsgispy.RSGISPyException):
        vgeoms.thin_pts_min_dist(pts, None, 10.0, n_reps=0)


def test_min_dist_geographic_crs():
    pts = gpd.GeoDataFrame(
        geometry=[Point(-4.0, 53.0), Point(-4.0001, 53.0)], crs="EPSG:4326"
    )
    with pytest.raises(rsgispy.RSGISPyException, match="geographic"):
        vgeoms.thin_pts_min_dist(pts, None, 0.001)
    out = vgeoms.thin_pts_min_dist(
        pts, None, 0.001, allow_geographic=True, method="greedy"
    )
    assert len(out) == 1


# --------------------------------------------------------------------------
# thin_pts_by_raster_grid
# --------------------------------------------------------------------------


def test_raster_grid_one_per_pixel(tmp_path):
    ref_img = _write_ref_img(tmp_path / "ref.tif")
    rng = np.random.default_rng(2)
    xy = rng.uniform(0.0, 1000.0, size=(400, 2))
    pts = gpd.GeoDataFrame(
        {"id": np.arange(len(xy))},
        geometry=gpd.points_from_xy(xy[:, 0], xy[:, 1]),
        crs=CRS,
    )
    out = vgeoms.thin_pts_by_raster_grid(pts, None, ref_img, rnd_seed=5)
    cells = {
        (int((1000.0 - y) // 100), int(x // 100))
        for x, y in zip(out.geometry.x, out.geometry.y)
    }
    assert len(cells) == len(out)
    occupied = {(int((1000.0 - y) // 100), int(x // 100)) for x, y in xy}
    assert len(out) == len(occupied)
    assert list(out["id"]) == sorted(out["id"])
    again = vgeoms.thin_pts_by_raster_grid(pts, None, ref_img, rnd_seed=5)
    assert list(out["id"]) == list(again["id"])


def test_raster_grid_cell_mult_and_first(tmp_path):
    ref_img = _write_ref_img(tmp_path / "ref.tif")
    # Four points in four different pixels of one 2x2-pixel block.
    pts = gpd.GeoDataFrame(
        {"id": [0, 1, 2, 3]},
        geometry=[Point(50, 950), Point(150, 950), Point(50, 850), Point(150, 850)],
        crs=CRS,
    )
    assert len(vgeoms.thin_pts_by_raster_grid(pts, None, ref_img, keep="first")) == 4
    out = vgeoms.thin_pts_by_raster_grid(pts, None, ref_img, cell_mult=2, keep="first")
    assert list(out["id"]) == [0]


def test_raster_grid_prefer_col(tmp_path):
    ref_img = _write_ref_img(tmp_path / "ref.tif")
    pts = gpd.GeoDataFrame(
        {"id": [0, 1, 2], "year": [2001.0, np.nan, 2020.0]},
        geometry=[Point(10, 990), Point(20, 980), Point(30, 970)],
        crs=CRS,
    )
    newest = vgeoms.thin_pts_by_raster_grid(pts, None, ref_img, prefer_col="year")
    assert list(newest["id"]) == [2]
    oldest = vgeoms.thin_pts_by_raster_grid(
        pts, None, ref_img, prefer_col="year", prefer_max=False
    )
    assert list(oldest["id"]) == [0]


def test_raster_grid_outside_and_reprojection(tmp_path):
    ref_img = _write_ref_img(tmp_path / "ref.tif", x0=400000.0, y0=5900000.0)
    pts_utm = gpd.GeoDataFrame(
        {"id": [0, 1, 2]},
        geometry=[Point(400050, 5899950), Point(400060, 5899940), Point(0, 0)],
        crs=CRS,
    )
    with pytest.warns(UserWarning, match="outside"):
        out = vgeoms.thin_pts_by_raster_grid(pts_utm, None, ref_img, keep="first")
    assert list(out["id"]) == [0]
    # Same points in WGS84 map to the same raster cells; geometry is unchanged.
    pts_ll = pts_utm.iloc[:2].to_crs("EPSG:4326")
    out_ll = vgeoms.thin_pts_by_raster_grid(pts_ll, None, ref_img, keep="first")
    assert list(out_ll["id"]) == [0]
    assert out_ll.crs.to_epsg() == 4326


def test_raster_grid_validation(tmp_path):
    ref_img = _write_ref_img(tmp_path / "ref.tif")
    pts = _clustered_pts()
    with pytest.raises(rsgispy.RSGISPyException):
        vgeoms.thin_pts_by_raster_grid(pts, None, ref_img, keep="bogus")
    with pytest.raises(rsgispy.RSGISPyException):
        vgeoms.thin_pts_by_raster_grid(pts, None, ref_img, cell_mult=0)
    with pytest.raises(rsgispy.RSGISPyException):
        vgeoms.thin_pts_by_raster_grid(pts, None, str(tmp_path / "missing.tif"))


# --------------------------------------------------------------------------
# thin_vector_geoms keep option
# --------------------------------------------------------------------------


def test_thin_vector_geoms_keep_random():
    grid = gpd.GeoDataFrame(
        geometry=[box(0, 0, 100, 100), box(100, 0, 200, 100)], crs=CRS
    )
    pts = gpd.GeoDataFrame(
        {"id": range(6)},
        geometry=[
            Point(10, 10),
            Point(20, 20),
            Point(30, 30),
            Point(110, 10),
            Point(120, 20),
            Point(130, 30),
        ],
        crs=CRS,
    )
    first = vgeoms.thin_vector_geoms(pts, None, grid, None)
    assert list(first["id"]) == [0, 3]
    kept = set()
    for seed in range(20):
        out = vgeoms.thin_vector_geoms(
            pts, None, grid, None, keep="random", rnd_seed=seed
        )
        assert len(out) == 2
        assert list(out["id"]) == sorted(out["id"])
        kept.update(out["id"])
    assert len(kept) > 2
    with pytest.raises(rsgispy.RSGISPyException):
        vgeoms.thin_vector_geoms(pts, None, grid, None, keep="bogus")


# --------------------------------------------------------------------------
# thin_pts_env_space
# --------------------------------------------------------------------------


def test_thin_pts_env_space():
    rng = np.random.default_rng(0)
    n_smpls = 300
    temp = np.concatenate([rng.normal(5.0, 0.1, 250), rng.uniform(0.0, 20.0, 50)])
    elev = np.concatenate([rng.normal(500.0, 5.0, 250), rng.uniform(0.0, 1000.0, 50)])
    temp[0] = -9999.0
    smpls = gpd.GeoDataFrame(
        {"id": np.arange(n_smpls), "temp": temp, "elev": elev},
        geometry=gpd.points_from_xy(
            rng.uniform(0, 1, n_smpls), rng.uniform(0, 1, n_smpls)
        ),
        crs=CRS,
    )
    with pytest.warns(UserWarning, match="no-data"):
        out = sdm.thin_pts_env_space(
            smpls, None, ["temp", "elev"], n_bins=5, rnd_seed=1
        )
    assert 0 not in set(out["id"])
    valid = smpls.iloc[1:]
    t_bins = np.clip(
        np.floor((valid["temp"] - valid["temp"].min()) / np.ptp(valid["temp"]) * 5),
        0,
        4,
    )
    e_bins = np.clip(
        np.floor((valid["elev"] - valid["elev"].min()) / np.ptp(valid["elev"]) * 5),
        0,
        4,
    )
    occupied = set(zip(t_bins, e_bins))
    assert len(out) == len(occupied)
    assert list(out["id"]) == sorted(out["id"])
    with pytest.warns(UserWarning, match="no-data"):
        again = sdm.thin_pts_env_space(
            smpls, None, ["temp", "elev"], n_bins={"temp": 5, "elev": 5}, rnd_seed=1
        )
    assert list(out["id"]) == list(again["id"])
    with pytest.raises(rsgispy.RSGISPyException):
        sdm.thin_pts_env_space(smpls, None, ["temp", "elev"], n_bins={"temp": 5})
    with pytest.raises(rsgispy.RSGISPyException):
        sdm.thin_pts_env_space(smpls, None, ["temp"], keep="bogus")
