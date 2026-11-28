"""
Tests for :func:`rsgispy.vector.create.polygonise_raster_tiled` (tiled,
multi-core polygonisation) and its kernels in
:mod:`rsgispy._polygonise_tile` / :mod:`rsgispy._polygonise_kernels`.

The main check is equivalence with the existing non-tiled
:func:`rsgispy.vector.create.polygonise_raster_to_vec_lyr` (same polygons,
values, holes and connectivity; for 8-connectivity compared with
``shapely.make_valid`` of GDAL's invalid self-touching rings) for both
backends, several tile sizes and difficult geometries on tile seams.
"""

import os

import geopandas as gpd
import numpy as np
import pyarrow.parquet as pq
import pytest
import rasterio
import rasterio.features
from rasterio.transform import Affine
import shapely

import rsgispy
from rsgispy import _polygonise as poly_impl
from rsgispy import _polygonise_kernels as kern
from rsgispy import _polygonise_tile as ptile
from rsgispy._numba import HAS_NUMBA, disable_numba
import rsgispy.vector.create as vec_create

from conftest import write_raster

needs_numba = pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")

BACKENDS = ["numba", "rasterio"]


def _hard_case_array():
    """40 x 40 raster with difficult geometries on the 10-pixel tile seams."""
    arr = np.full((40, 40), 1, dtype=np.int32)  # spans all tiles, touches edges
    arr[6:15, 6:15] = 2  # hole of value 1 cut by the seams at row/col 10
    arr[9:12, 9:12] = 3  # island in the hole, cut by the seams
    arr[10, 10] = 1  # island in the island's hole
    arr[17:24, 17:24] = 4  # ring around the tile corner (20, 20)
    arr[18:23, 18:23] = 5
    arr[29, :] = 6  # 1-pixel line along the seam at row 30
    arr[:, 30] = 7  # 1-pixel line starting at the seam at col 30
    arr[9, 25] = 8  # diagonal pair across the seam at row 10
    arr[10, 26] = 8
    for k in range(7):  # diagonal chain across the seam at col 20
        arr[32 + k, 15 + k] = 9
    arr[9, 19] = 10  # touching only at the 4-tile corner (10, 20)
    arr[10, 20] = 10
    arr[35:40, 8:12] = 11 + (np.add.outer(np.arange(5), np.arange(4)) % 2)
    arr[0:3, 36:40] = 13  # touches the image edge
    arr[25:27, 2:6] = 0  # value 0 region
    return arr


def _blob_array(rng, height, width, n_vals):
    small = rng.integers(0, n_vals, size=(height // 3 + 1, width // 3 + 1))
    arr = np.kron(small, np.ones((3, 3), dtype=np.int64))[:height, :width]
    noise = rng.random((height, width)) < 0.15
    arr[noise] = rng.integers(0, n_vals, int(noise.sum()))
    return arr.astype(np.int32)


def _canon(gdf, col="PXLVAL"):
    geoms = np.asarray(gdf.geometry.values)
    geoms = np.array(
        [g if g.is_valid else shapely.make_valid(g) for g in geoms], dtype=object
    )
    cen = shapely.get_coordinates(shapely.centroid(geoms))
    key = np.lexsort(
        (
            np.round(cen[:, 1], 6),
            np.round(cen[:, 0], 6),
            np.round(shapely.area(geoms), 6),
            gdf[col].to_numpy(),
        )
    )
    return gdf[col].to_numpy()[key], geoms[key]


def _assert_equivalent(ref, new, col="PXLVAL", check_vertices=False):
    ref_v, ref_g = _canon(ref, col)
    new_v, new_g = _canon(new, col)
    assert len(new_v) == len(ref_v)
    np.testing.assert_array_equal(new_v, ref_v)
    assert shapely.equals(new_g, ref_g).all()
    assert shapely.is_valid(np.asarray(new.geometry.values)).all()
    if check_vertices:
        np.testing.assert_array_equal(
            shapely.get_num_coordinates(new_g), shapely.get_num_coordinates(ref_g)
        )


@pytest.fixture(name="hard_img")
def fixture_hard_img(tmp_path):
    return write_raster(tmp_path / "hard.tif", _hard_case_array())


# ---------------------------------------------------------------------------
# Equivalence with the existing function
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("use_8_conn", [False, True])
@pytest.mark.parametrize("tile_size", [10, 7, (1, 40), (40, 3), 13, 64])
def test_matches_existing_function(hard_img, backend, use_8_conn, tile_size):
    ref = vec_create.polygonise_raster_to_vec_lyr(
        hard_img, use_8_conn=use_8_conn
    )
    new = vec_create.polygonise_raster_tiled(
        hard_img,
        None,
        use_8_conn=use_8_conn,
        tile_size=tile_size,
        backend=backend,
        use_pbar=False,
    )
    _assert_equivalent(ref, new, check_vertices=not use_8_conn)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("use_8_conn", [False, True])
def test_matches_existing_function_random(tmp_path, backend, use_8_conn):
    rng = np.random.default_rng(42)
    for trial in range(4):
        arr = _blob_array(rng, 23 + 7 * trial, 31 - 3 * trial, 2 + trial)
        img = write_raster(tmp_path / f"blob_{trial}.tif", arr)
        ref = vec_create.polygonise_raster_to_vec_lyr(
            img, use_8_conn=use_8_conn
        )
        for tile_size in (1, 5, (3, 11)):
            if tile_size == 1 and trial > 0:
                continue
            new = vec_create.polygonise_raster_tiled(
                img,
                None,
                use_8_conn=use_8_conn,
                tile_size=tile_size,
                backend=backend,
                use_pbar=False,
            )
            _assert_equivalent(ref, new, check_vertices=not use_8_conn)


def test_eight_connectivity_writes_valid_multipolygons(tmp_path):
    arr = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8)
    img = write_raster(tmp_path / "diag.tif", arr)
    for backend in BACKENDS:
        gdf = vec_create.polygonise_raster_tiled(
            img, None, use_8_conn=True, tile_size=2, backend=backend, use_pbar=False
        )
        ones = gdf[gdf["PXLVAL"] == 1]
        assert len(ones) == 1
        assert ones.geometry.iloc[0].geom_type == "MultiPolygon"
        assert shapely.is_valid(gdf.geometry.values).all()
        assert ones.geometry.iloc[0].area == pytest.approx(300.0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_orientation_and_south_up_transform(tmp_path, backend):
    arr = _hard_case_array()
    north = write_raster(tmp_path / "north.tif", arr)
    south = write_raster(tmp_path / "south.tif", arr, transform=Affine(10, 0, 500000, 0, 10, 5200000))
    for img in (north, south):
        gdf = vec_create.polygonise_raster_tiled(
            img, None, tile_size=9, backend=backend, use_pbar=False
        )
        polys = shapely.get_parts(gdf.geometry.values)
        assert shapely.is_ccw(shapely.get_exterior_ring(polys)).all()
        holes = [h for p in polys for h in p.interiors]
        assert holes and not any(shapely.is_ccw(h) for h in holes)
        ref = vec_create.polygonise_raster_to_vec_lyr(img)
        _assert_equivalent(ref, gdf, check_vertices=True)


def test_awkward_geotransform_within_rounding(tmp_path):
    arr = _hard_case_array()
    img = write_raster(
        tmp_path / "geo.tif",
        arr,
        transform=Affine(0.000269494585236, 0, -3.123456789, 0, -0.000269494585236, 52.98765),
    )
    ref = vec_create.polygonise_raster_to_vec_lyr(img)
    new = vec_create.polygonise_raster_tiled(img, None, tile_size=8, use_pbar=False)
    ref_v, ref_g = _canon(ref)
    new_v, new_g = _canon(new)
    np.testing.assert_array_equal(new_v, ref_v)
    assert shapely.equals_exact(
        shapely.normalize(new_g), shapely.normalize(ref_g), tolerance=1e-12
    ).all()
    # pieces from different tiles share exactly the same vertices
    assert shapely.is_valid(new_g).all()
    assert shapely.area(shapely.union_all(new_g)) == pytest.approx(
        shapely.area(shapely.box(*shapely.total_bounds(new_g)))
    )


# ---------------------------------------------------------------------------
# Parallelism, kernels and fallbacks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_n_jobs_identical_output(tmp_path, backend):
    rng = np.random.default_rng(7)
    img = write_raster(tmp_path / "blob.tif", _blob_array(rng, 90, 70, 6))
    outs = []
    for n_jobs in (1, 2):
        out = str(tmp_path / f"out_{backend}_{n_jobs}.parquet")
        vec_create.polygonise_raster_tiled(
            img,
            out,
            tile_size=16,
            n_jobs=n_jobs,
            backend=backend,
            inc_pxl_count=True,
            use_pbar=False,
        )
        outs.append(pq.read_table(out))
    assert outs[0].equals(outs[1])


@needs_numba
def test_numba_tile_kernel_matches_fallback():
    rng = np.random.default_rng(1)
    for trial in range(6):
        arr = _blob_array(rng, 17 + trial, 23, 2 + trial % 4)
        valid = rng.random(arr.shape) > 0.1
        for use_8_conn in (False, True):
            comp = ptile.polygonise_tile_numba(arr, valid, use_8_conn, 3, 5)
            with disable_numba():
                fall = ptile.polygonise_tile_numba(arr, valid, use_8_conn, 3, 5)
            for name in (
                "coords",
                "ring_offsets",
                "part_offsets",
                "feat_offsets",
                "values",
                "counts",
                "bbox",
            ):
                np.testing.assert_array_equal(getattr(comp, name), getattr(fall, name))
            buf_c, off_c = ptile.build_wkb(
                comp.coords, comp.ring_offsets, comp.part_offsets, comp.feat_offsets
            )
            with disable_numba():
                buf_f, off_f = ptile.build_wkb(
                    comp.coords, comp.ring_offsets, comp.part_offsets, comp.feat_offsets
                )
            np.testing.assert_array_equal(buf_c, buf_f)
            np.testing.assert_array_equal(off_c, off_f)


def test_dissolve_wkb_kernel_matches_fallback(hard_img):
    gdf = vec_create.polygonise_raster_tiled(
        hard_img, None, use_8_conn=True, sort_by_value=True, tile_size=7, use_pbar=False
    )
    wkbs = shapely.to_wkb(gdf.geometry.values)
    lens = np.array([len(w) for w in wkbs], dtype=np.int64)
    offs = np.concatenate([[0], np.cumsum(lens)])
    buf = np.frombuffer(b"".join(wkbs), dtype=np.uint8)
    vals = gdf["PXLVAL"].to_numpy()
    run_start = np.concatenate([[0], np.nonzero(vals[1:] != vals[:-1])[0] + 1])
    run_len = np.diff(np.append(run_start, len(vals)))
    out_c = kern.merge_wkb_runs_kernel(buf, offs, run_start, run_len)
    # pylint: disable-next=protected-access
    out_f = kern._merge_wkb_runs_py(buf, offs, run_start, run_len)
    np.testing.assert_array_equal(out_c[0], out_f[0])
    np.testing.assert_array_equal(out_c[1], out_f[1])
    merged = shapely.from_wkb(
        [bytes(out_c[0][out_c[1][i] : out_c[1][i + 1]]) for i in range(len(run_start))]
    )
    ref = gdf.dissolve(by="PXLVAL").sort_index()
    assert shapely.equals(merged, ref.geometry.values).all()
    with disable_numba():
        diss = vec_create.polygonise_raster_tiled(
            hard_img,
            None,
            use_8_conn=True,
            dissolve_by_value=True,
            sort_by_value=True,
            tile_size=7,
            use_pbar=False,
        )
    assert shapely.equals(diss.geometry.values, ref.geometry.values).all()


def test_numba_backend_without_numba_matches(hard_img):
    ref = vec_create.polygonise_raster_tiled(
        hard_img, None, tile_size=10, backend="rasterio", use_pbar=False
    )
    with disable_numba():
        new = vec_create.polygonise_raster_tiled(
            hard_img, None, tile_size=10, backend="numba", use_pbar=False
        )
        auto = vec_create.polygonise_raster_tiled(
            hard_img, None, tile_size=10, use_pbar=False
        )
    _assert_equivalent(ref, new, check_vertices=True)
    _assert_equivalent(ref, auto, check_vertices=True)


@pytest.mark.parametrize("use_8_conn", [False, True])
def test_tile_kernels_match_gdal_on_random_tiles(use_8_conn):
    rng = np.random.default_rng(11)
    for _ in range(40):
        height, width = rng.integers(1, 14, 2)
        arr = rng.integers(0, rng.integers(1, 5), size=(height, width)).astype(np.int16)
        valid = rng.random((height, width)) > rng.choice([0.0, 0.25])
        ref = []
        for geom, val in rasterio.features.shapes(
            arr, mask=valid, connectivity=8 if use_8_conn else 4, transform=Affine.identity()
        ):
            ref_g = shapely.geometry.shape(geom)
            ref.append((val, ref_g if ref_g.is_valid else shapely.make_valid(ref_g)))
        for poly_fn in (ptile.polygonise_tile_numba, ptile.polygonise_tile_rasterio):
            tps = poly_fn(arr, valid, use_8_conn)
            geoms = ptile.features_to_geoms(
                tps.coords, tps.ring_offsets, tps.part_offsets, tps.feat_offsets
            )
            assert tps.n_feats == len(ref)
            assert shapely.is_valid(geoms).all()
            np.testing.assert_array_equal(tps.counts, np.rint(shapely.area(geoms)))
            ref_g = np.array([g for _, g in ref], dtype=object)
            ref_v = np.array([v for v, _ in ref])
            pairs = shapely.STRtree(ref_g).query(
                shapely.point_on_surface(geoms), predicate="within"
            )
            match = np.empty(len(geoms), dtype=np.int64)
            match[pairs[0]] = pairs[1]
            assert shapely.equals(geoms, ref_g[match]).all()
            np.testing.assert_array_equal(tps.values, ref_v[match])


# ---------------------------------------------------------------------------
# Masks, no data, dtypes and empty rasters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_mask_and_no_data(tmp_path, backend):
    arr = _hard_case_array()
    img = write_raster(tmp_path / "hard.tif", arr)
    msk_arr = ((np.arange(1600).reshape(40, 40) % 7) != 0).astype(np.uint8)
    msk = write_raster(tmp_path / "msk.tif", msk_arr)
    ref = vec_create.polygonise_raster_to_vec_lyr(img, mask_img=msk)
    new = vec_create.polygonise_raster_tiled(
        img, None, mask_img=msk, tile_size=9, backend=backend, use_pbar=False
    )
    _assert_equivalent(ref, new, check_vertices=True)

    no_one = vec_create.polygonise_raster_tiled(
        img, None, no_data_val=1, tile_size=9, backend=backend, use_pbar=False
    )
    assert 1 not in set(no_one["PXLVAL"])
    assert 0 in set(no_one["PXLVAL"])
    full = vec_create.polygonise_raster_tiled(
        img, None, tile_size=9, backend=backend, use_pbar=False
    )
    keep = full[full["PXLVAL"] != 1]
    _assert_equivalent(keep, no_one, check_vertices=True)


@pytest.mark.parametrize("backend", BACKENDS)
def test_float_raster(tmp_path, backend):
    arr = (_hard_case_array() % 5).astype(np.float32) * 0.25 - 0.5
    arr[3:6, 3:30] = np.nan
    arr[20, 20] = -0.0
    arr[21, 20] = 0.0
    img = write_raster(tmp_path / "f.tif", arr)
    ref = vec_create.polygonise_raster_to_vec_lyr(img)
    ref = ref[~np.isnan(ref["PXLVAL"])]
    new = vec_create.polygonise_raster_tiled(
        img, None, tile_size=8, backend=backend, use_pbar=False
    )
    assert new["PXLVAL"].dtype == np.float64
    _assert_equivalent(ref, new)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "dtype,offset",
    [
        (np.uint32, 3_000_000_000),
        (np.int64, -(2**40)),
        (np.uint64, 2**63 + 5),
    ],
)
def test_large_integer_labels(tmp_path, backend, dtype, offset):
    base = _hard_case_array()
    arr = (base.astype(np.int64).astype(dtype) + dtype(offset)).astype(dtype)
    img = write_raster(tmp_path / "big.tif", arr)
    ref_img = write_raster(tmp_path / "small.tif", base)
    ref = vec_create.polygonise_raster_tiled(
        ref_img, None, tile_size=9, backend=backend, use_pbar=False
    )
    ref["PXLVAL"] = (ref["PXLVAL"].to_numpy().astype(np.int64).astype(dtype) + dtype(offset))
    new = vec_create.polygonise_raster_tiled(
        img, None, tile_size=9, backend=backend, use_pbar=False
    )
    assert new["PXLVAL"].dtype == (np.uint64 if dtype == np.uint64 else np.int64)
    _assert_equivalent(ref, new, check_vertices=True)


def test_empty_and_all_masked(tmp_path):
    img = write_raster(tmp_path / "zeros.tif", np.zeros((25, 30), dtype=np.uint16))
    gdf = vec_create.polygonise_raster_tiled(img, None, no_data_val=0, use_pbar=False)
    assert len(gdf) == 0
    assert "PXLVAL" in gdf.columns
    out_pq = str(tmp_path / "empty.parquet")
    vec_create.polygonise_raster_tiled(
        img, out_pq, no_data_val=0, inc_pxl_count=True, sort_by_value=True, use_pbar=False
    )
    assert gpd.read_parquet(out_pq).shape == (0, 3)
    out_gpkg = str(tmp_path / "empty.gpkg")
    vec_create.polygonise_raster_tiled(img, out_gpkg, no_data_val=0, use_pbar=False)
    assert len(gpd.read_file(out_gpkg)) == 0


# ---------------------------------------------------------------------------
# Dissolve, sort and pixel counts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("use_8_conn", [False, True])
def test_dissolve_sort_and_pxl_count(hard_img, backend, use_8_conn):
    arr = _hard_case_array()
    ref = vec_create.polygonise_raster_to_vec_lyr(
        hard_img, use_8_conn=use_8_conn
    )
    # GDAL's 8-connected rings can be invalid; repair them before dissolving.
    ref["geometry"] = shapely.make_valid(ref.geometry.values)
    ref_diss = ref.dissolve(by="PXLVAL").sort_index()
    gdf = vec_create.polygonise_raster_tiled(
        hard_img,
        None,
        pxl_val_fieldname="seg_id",
        use_8_conn=use_8_conn,
        dissolve_by_value=True,
        sort_by_value=True,
        inc_pxl_count=True,
        tile_size=7,
        backend=backend,
        use_pbar=False,
    )
    counts = np.bincount(arr.ravel())
    np.testing.assert_array_equal(gdf["seg_id"].to_numpy(), np.nonzero(counts)[0])
    np.testing.assert_array_equal(gdf["pxl_count"].to_numpy(), counts[counts > 0])
    assert shapely.equals(gdf.geometry.values, ref_diss.geometry.values).all()
    assert shapely.is_valid(gdf.geometry.values).all()


def test_sorted_output_independent_of_tiling(tmp_path):
    rng = np.random.default_rng(9)
    arr = _blob_array(rng, 45, 38, 5)  # few values: many polygons per value
    img = write_raster(tmp_path / "blob.tif", arr)
    outs = [
        vec_create.polygonise_raster_tiled(
            img, None, sort_by_value=True, tile_size=ts, backend=be, use_pbar=False
        )
        for ts, be in ((64, "numba"), (6, "numba"), ((4, 13), "rasterio"))
    ]
    for gdf in outs[1:]:
        assert gdf["PXLVAL"].tolist() == outs[0]["PXLVAL"].tolist()
        assert shapely.equals(gdf.geometry.values, outs[0].geometry.values).all()


def test_pxl_count_per_polygon(hard_img):
    gdf = vec_create.polygonise_raster_tiled(
        hard_img, None, inc_pxl_count=True, tile_size=6, use_pbar=False
    )
    np.testing.assert_array_equal(
        gdf["pxl_count"].to_numpy(), np.rint(gdf.geometry.area.to_numpy() / 100.0)
    )


def test_sort_and_dissolve_out_of_core(tmp_path, monkeypatch):
    rng = np.random.default_rng(5)
    arr = _blob_array(rng, 60, 50, 40)
    img = write_raster(tmp_path / "blob.tif", arr)
    ref = vec_create.polygonise_raster_tiled(img, None, tile_size=11, use_pbar=False)
    in_mem_sorted = vec_create.polygonise_raster_tiled(
        img, None, tile_size=11, sort_by_value=True, use_pbar=False
    )
    in_mem_diss = vec_create.polygonise_raster_tiled(
        img, None, tile_size=11, dissolve_by_value=True, use_pbar=False
    )
    monkeypatch.setattr(poly_impl, "SORT_BUCKET_ROWS", 37)
    out_sort = str(tmp_path / "sorted.parquet")
    vec_create.polygonise_raster_tiled(
        img, out_sort, tile_size=11, sort_by_value=True, chunk_size=29, use_pbar=False
    )
    srt = gpd.read_parquet(out_sort)
    assert (np.diff(srt["PXLVAL"].to_numpy()) >= 0).all()
    _assert_equivalent(ref, srt, check_vertices=True)
    assert srt["PXLVAL"].tolist() == in_mem_sorted["PXLVAL"].tolist()
    assert shapely.equals(srt.geometry.values, in_mem_sorted.geometry.values).all()

    out_diss = str(tmp_path / "diss.gpkg")
    vec_create.polygonise_raster_tiled(
        img, out_diss, tile_size=11, dissolve_by_value=True, chunk_size=29, use_pbar=False
    )
    diss = gpd.read_file(out_diss)
    # without sorting: values in raster order of their first pixel
    vals, first_idx = np.unique(arr.ravel(), return_index=True)
    assert diss["PXLVAL"].tolist() == vals[np.argsort(first_idx)].tolist()
    assert diss["PXLVAL"].tolist() == in_mem_diss["PXLVAL"].tolist()
    assert shapely.equals(diss.geometry.values, in_mem_diss.geometry.values).all()
    leftovers = [f for f in os.listdir(tmp_path) if f.startswith((".", "rsgispy_"))]
    assert not leftovers


# ---------------------------------------------------------------------------
# Output formats and validation
# ---------------------------------------------------------------------------


def test_output_formats(tmp_path, hard_img):
    ref = vec_create.polygonise_raster_tiled(hard_img, None, tile_size=12, use_pbar=False)
    out_pq = str(tmp_path / "out.parquet")
    vec_create.polygonise_raster_tiled(hard_img, out_pq, tile_size=12, use_pbar=False)
    out_gpkg = str(tmp_path / "out.gpkg")
    vec_create.polygonise_raster_tiled(
        hard_img, out_gpkg, out_vec_lyr="polys", tile_size=12, use_pbar=False
    )
    for gdf in (gpd.read_parquet(out_pq), gpd.read_file(out_gpkg, layer="polys")):
        assert gdf.crs.to_epsg() == 32630
        _assert_equivalent(ref, gdf, check_vertices=True)
    assert pq.ParquetFile(out_pq).metadata.row_group(0).column(0).compression == "SNAPPY"
    with pytest.raises(rsgispy.RSGISPyException, match="already exists"):
        vec_create.polygonise_raster_tiled(hard_img, out_pq, use_pbar=False)
    vec_create.polygonise_raster_tiled(hard_img, out_pq, overwrite=True, use_pbar=False)


def test_validation_errors(tmp_path, hard_img):
    with pytest.raises(rsgispy.RSGISPyException, match="backend"):
        vec_create.polygonise_raster_tiled(hard_img, None, backend="gdal")
    with pytest.raises(rsgispy.RSGISPyException, match="n_jobs"):
        vec_create.polygonise_raster_tiled(hard_img, None, n_jobs=0)
    with pytest.raises(rsgispy.RSGISPyException, match="tile_size"):
        vec_create.polygonise_raster_tiled(hard_img, None, tile_size=0)
    with pytest.raises(rsgispy.RSGISPyException, match="chunk_size"):
        vec_create.polygonise_raster_tiled(hard_img, None, chunk_size=0)
    with pytest.raises(rsgispy.RSGISPyException, match="different"):
        vec_create.polygonise_raster_tiled(
            hard_img, None, inc_pxl_count=True, pxl_count_fieldname="PXLVAL"
        )
    small = write_raster(tmp_path / "small.tif", np.ones((5, 5), dtype=np.uint8))
    with pytest.raises(rsgispy.RSGISPyException, match="Mask image"):
        vec_create.polygonise_raster_tiled(hard_img, None, mask_img=small)
    with pytest.raises(rsgispy.RSGISPyException):
        vec_create.polygonise_raster_tiled(hard_img, None, img_band=2)
    with pytest.raises(rsgispy.RSGISPyException):
        vec_create.polygonise_raster_tiled(str(tmp_path / "missing.tif"), None)


def test_remove_collinear_keeps_corners():
    poly = shapely.Polygon([(0, 0), (2, 0), (4, 0), (4, 2), (4, 4), (0, 4), (0, 2), (0, 0)])
    out = ptile.remove_collinear(np.array([poly], dtype=object))[0]
    assert shapely.equals(out, poly)
    assert shapely.get_num_coordinates(out) == 5


def test_polygonise_raster_to_vec_lyr_new_order_and_no_data(tmp_path):
    """Same argument order as polygonise_raster_tiled, with no_data_val."""
    arr = np.zeros((6, 6), dtype=np.uint8)
    arr[1:3, 1:3] = 1
    arr[4:6, 3:6] = 2
    img = write_raster(tmp_path / "cls.tif", arr)
    every = vec_create.polygonise_raster_to_vec_lyr(img)
    assert sorted(every["PXLVAL"].unique().tolist()) == [0, 1, 2]
    no_zero = vec_create.polygonise_raster_to_vec_lyr(img, None, None, 1, no_data_val=0)
    assert sorted(no_zero["PXLVAL"].tolist()) == [1, 2]
    tiled = vec_create.polygonise_raster_tiled(img, None, no_data_val=0, use_pbar=False)
    assert sorted(tiled["PXLVAL"].tolist()) == [1, 2]

    out_gpkg = str(tmp_path / "out.gpkg")
    vec_create.polygonise_raster_to_vec_lyr(img, out_gpkg, "polys", no_data_val=0)
    assert len(gpd.read_file(out_gpkg, layer="polys")) == 2
    # The old RSGISLib order (out_vec_file, out_vec_lyr, out_format,
    # input_img) is no longer accepted.
    with pytest.raises((rsgispy.RSGISPyException, TypeError, ValueError)):
        vec_create.polygonise_raster_to_vec_lyr(None, None, "GPKG", img)
