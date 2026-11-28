"""
Tests for :mod:`rsgispy.imageutils.composite`: the geomedian, quality mosaic
(and maximum NDVI), land / water (and maximum NDVI / NDWI), medoid and
per-band percentile composites, their tile-graph nodes, and the numba and
NumPy paths of the geomedian kernel.
"""

# pylint: disable=missing-function-docstring,protected-access,too-many-locals

import warnings

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

import rsgispy
import rsgispy.imageutils.composite as comp
from rsgispy._numba import disable_numba
from rsgispy.imageprocess import ArraySource, MemorySink, TileGraph

from conftest import write_raster

ND = -9999.0
SHAPE = (23, 29)
TRANSFORM = from_origin(500000.0, 5700000.0, 10.0, 10.0)


def _write(path, arr, dtype="float64", nodata=ND, names=None, transform=TRANSFORM):
    return write_raster(
        path, arr, dtype=dtype, nodata=nodata, band_names=names, transform=transform
    )


def _read(path):
    with rasterio.open(path) as src:
        return src.read().astype(np.float64)


def _stack(seed=3, n_obs=6, n_bands=3, shape=SHAPE, gap=0.25, integer=False):
    """Random observations ``(n_obs, n_bands, rows, cols)`` with gaps (ND)."""
    rng = np.random.default_rng(seed)
    data = rng.normal(1000.0, 300.0, size=(n_obs, n_bands) + shape)
    if integer:
        data = np.round(data)
    for t_idx in range(n_obs):
        data[t_idx][:, rng.random(shape) < gap] = ND
    # a pixel where one band only is missing in observation 0
    data[0, 1, 2, 2] = ND
    # a pixel with no valid observation
    data[:, :, 0, 0] = ND
    return data


def _write_stack(tmp_path, data, prefix="obs", dtype="float64", names=None):
    return [
        _write(tmp_path / f"{prefix}{i}.tif", data[i], dtype=dtype, names=names)
        for i in range(data.shape[0])
    ]


def _valid(data):
    """``(n_obs, rows, cols)``: every band valid."""
    return np.all(np.isfinite(data) & (data != ND), axis=1)


def _np_select(qual, cand, mode):
    """Direct NumPy argmax / argmin (first on ties); -1 where no candidate."""
    if mode == "max":
        idx = np.argmax(np.where(cand, qual, -np.inf), axis=0)
    else:
        idx = np.argmin(np.where(cand, qual, np.inf), axis=0)
    idx[~cand.any(axis=0)] = -1
    return idx


def _np_gather(data, idx):
    """Composite of the observations ``idx`` (ND where -1)."""
    out = np.full(data.shape[1:], ND)
    for t_idx in range(data.shape[0]):
        msk = idx == t_idx
        out[:, msk] = data[t_idx][:, msk]
    return out


def _ndvi(data, red=1, nir=2):
    with np.errstate(divide="ignore", invalid="ignore"):
        return (data[:, nir] - data[:, red]) / (data[:, nir] + data[:, red])


# ---------------------------------------------------------------------------
# Geomedian
# ---------------------------------------------------------------------------


def _gm_obj(pnt, pts):
    return float(np.sum(np.linalg.norm(pts - pnt, axis=1)))


def _gm_points(pts, tol=1e-7, max_iter=1000):
    """Geomedian of one pixel's observations ``pts`` (n_obs, n_bands)."""
    data = np.asarray(pts, dtype=np.float64)[:, :, np.newaxis]
    valid = np.ones((pts.shape[0], 1), dtype=bool)
    return comp._geomedian_pixels(data, valid, max_iter, tol)[:, 0]


def test_geomedian_minimises_sum_of_distances():
    scipy_opt = pytest.importorskip("scipy.optimize")
    rng = np.random.default_rng(42)
    for _ in range(25):
        pts = rng.normal(0.0, 1.0, size=(int(rng.integers(3, 12)), 4)) * rng.uniform(1, 500)
        geo = _gm_points(pts)
        f_gm = _gm_obj(geo, pts)
        res = scipy_opt.minimize(
            _gm_obj, geo, args=(pts,), method="Nelder-Mead",
            options={"xatol": 1e-12, "fatol": 1e-14, "maxiter": 40000},
        )
        assert f_gm <= res.fun * (1.0 + 1e-6)
        assert abs(f_gm - res.fun) <= 1e-6 * res.fun
        assert f_gm < _gm_obj(pts.mean(axis=0), pts)
        for pnt in pts:
            assert f_gm <= _gm_obj(pnt, pts)


def test_geomedian_properties():
    rng = np.random.default_rng(7)
    # identical observations
    pnt = np.array([3.0, -2.0, 7.5])
    np.testing.assert_allclose(_gm_points(np.tile(pnt, (5, 1))), pnt)
    # one band: the median (odd count)
    vals = rng.normal(0.0, 10.0, size=(7, 1))
    np.testing.assert_allclose(_gm_points(vals, tol=1e-12)[0], np.median(vals), rtol=1e-6)
    # one / two observations
    np.testing.assert_array_equal(_gm_points(pnt[np.newaxis]), pnt)
    two = np.array([[0.0, 0.0, 0.0], [2.0, 4.0, -6.0]])
    np.testing.assert_array_equal(_gm_points(two), two.mean(axis=0))
    # translation and rotation equivariance
    pts = rng.normal(0.0, 5.0, size=(9, 3))
    geo = _gm_points(pts, tol=1e-12)
    shift = np.array([100.0, -50.0, 25.0])
    np.testing.assert_allclose(_gm_points(pts + shift, tol=1e-12), geo + shift, atol=1e-6)
    rot, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    np.testing.assert_allclose(_gm_points(pts @ rot.T, tol=1e-12), rot @ geo, atol=1e-6)


def test_geomedian_vardi_zhang_on_observation():
    # The geomedian is an observation (the centre of a symmetric star).
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    pts = np.vstack([pts, pts[:1]])  # the centre twice; the mean is the centre
    geo = _gm_points(pts + np.array([[0.1, 0.0]] + [[0.0, 0.0]] * 5))
    assert np.all(np.isfinite(geo))
    # an observation that dominates: the geomedian is (almost) on it
    pts = np.array([[5.0, 5.0]] * 6 + [[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    np.testing.assert_allclose(_gm_points(pts), [5.0, 5.0], atol=1e-6)


def test_geomedian_numba_and_numpy_agree():
    rng = np.random.default_rng(1)
    data = rng.normal(1000.0, 300.0, size=(8, 5, 3000))
    valid = rng.random((8, 3000)) > 0.3
    data[0, :, :50] = data[1, :, :50]  # coincident observations
    out_jit = comp._geomedian_pixels(data, valid, 1000, 1e-7)
    with disable_numba():
        out_np = comp._geomedian_pixels(data, valid, 1000, 1e-7)
    np.testing.assert_allclose(out_jit, out_np, rtol=1e-9, equal_nan=True)
    np.testing.assert_allclose(
        comp._geomedian_numpy(data, valid, 1000, 1e-7), out_jit, rtol=1e-9, equal_nan=True
    )


def test_geomedian_composite_outputs(tmp_path):
    data = _stack(seed=5, n_obs=7, n_bands=4)
    imgs = _write_stack(tmp_path, data, names=["B", "G", "R", "N"])
    out = str(tmp_path / "gm.tif")
    comp.create_geomedian_composite(
        imgs, out, out_mad_img=str(tmp_path / "mad.tif"),
        out_count_img=str(tmp_path / "cnt.tif"), out_ref_img=str(tmp_path / "ref.tif"),
        use_pbar=False,
    )
    geo = _read(out)
    mads = _read(tmp_path / "mad.tif")
    cnt = _read(tmp_path / "cnt.tif")[0]
    ref = _read(tmp_path / "ref.tif")[0]
    valid = _valid(data)
    np.testing.assert_array_equal(cnt, valid.sum(axis=0))
    with rasterio.open(out) as src:
        assert src.nodata == ND
        assert src.descriptions == ("B", "G", "R", "N")
        assert src.dtypes[0] == "float64"
    with rasterio.open(tmp_path / "mad.tif") as src:
        assert src.descriptions == ("EMAD", "SMAD", "BCMAD")
        assert np.isnan(src.nodata)
    with rasterio.open(tmp_path / "ref.tif") as src:
        assert src.dtypes[0] == "uint16"
        assert src.nodata == 0
        assert src.tags(1)["LAYER_TYPE"] == "thematic"
        assert src.colormap(1)[1][3] == 255
    with rasterio.open(tmp_path / "cnt.tif") as src:
        assert src.nodata == 0
    # empty pixel
    assert np.all(geo[:, 0, 0] == ND) and ref[0, 0] == 0 and cnt[0, 0] == 0
    assert np.all(np.isnan(mads[:, 0, 0]))
    # direct NumPy calculation, pixel by pixel
    for row in range(0, SHAPE[0], 3):
        for col in range(0, SHAPE[1], 2):
            vld = valid[:, row, col]
            if not vld.any():
                continue
            pts = data[vld, :, row, col]
            gm_px = geo[:, row, col]
            np.testing.assert_allclose(gm_px, _gm_points(pts), rtol=1e-12)
            dist = np.linalg.norm(pts - gm_px, axis=1)
            assert ref[row, col] == np.nonzero(vld)[0][np.argmin(dist)] + 1
            cos = 1.0 - pts @ gm_px / (np.linalg.norm(pts, axis=1) * np.linalg.norm(gm_px))
            bray = np.abs(pts - gm_px).sum(axis=1) / np.abs(pts + gm_px).sum(axis=1)
            np.testing.assert_allclose(
                mads[:, row, col], [np.median(dist), np.median(cos), np.median(bray)],
                rtol=1e-5,
            )


def test_geomedian_block_size_threads_and_numba(tmp_path):
    data = _stack(seed=9, n_obs=5, n_bands=3)
    imgs = _write_stack(tmp_path, data)
    outs = []
    for idx, (bsize, n_jobs) in enumerate([(256, 1), (8, 1), (5, 3)]):
        out = str(tmp_path / f"gm_{idx}.tif")
        comp.create_geomedian_composite(
            imgs, out, out_ref_img=str(tmp_path / f"ref_{idx}.tif"),
            block_size=bsize, n_jobs=n_jobs, use_pbar=False, calc_stats=False,
        )
        outs.append((_read(out), _read(tmp_path / f"ref_{idx}.tif")))
    for geo, ref in outs[1:]:
        np.testing.assert_array_equal(geo, outs[0][0])
        np.testing.assert_array_equal(ref, outs[0][1])
    with disable_numba():
        comp.create_geomedian_composite(
            imgs, str(tmp_path / "gm_np.tif"), use_pbar=False, calc_stats=False
        )
    np.testing.assert_allclose(_read(tmp_path / "gm_np.tif"), outs[0][0], rtol=1e-9)


def test_geomedian_integer_output_rounded(tmp_path):
    data = _stack(seed=2, n_obs=5, n_bands=2, integer=True)
    imgs = _write_stack(tmp_path, data, dtype="int16")
    out = str(tmp_path / "gm.tif")
    comp.create_geomedian_composite(imgs, out, use_pbar=False, calc_stats=False)
    comp.create_geomedian_composite(
        imgs, str(tmp_path / "gm_f.tif"), datatype=rsgispy.TYPE_64FLOAT,
        use_pbar=False, calc_stats=False,
    )
    with rasterio.open(out) as src:
        assert src.dtypes[0] == "int16"
    gm_f = _read(tmp_path / "gm_f.tif")
    exp = np.where(gm_f == ND, ND, np.rint(gm_f))
    np.testing.assert_array_equal(_read(out), exp)


def test_geomedian_bands_and_node(tmp_path):
    data = _stack(seed=4, n_obs=5, n_bands=4)
    imgs = _write_stack(tmp_path, data)
    out = str(tmp_path / "gm.tif")
    comp.create_geomedian_composite(
        imgs, out, bands=[2, 4], datatype=rsgispy.TYPE_64FLOAT, use_pbar=False,
        calc_stats=False,
    )
    srcs = [ArraySource(d, transform=TRANSFORM, nodata=ND) for d in data]
    node = comp.geomedian_node(srcs, bands=[2, 4])
    res = TileGraph(
        [MemorySink(node, name="gm"), MemorySink(node.count, name="n")], block_size=7
    ).run(use_pbar=False)
    np.testing.assert_allclose(res["gm"], _read(out), rtol=1e-12)
    valid = np.all(data[:, [1, 3]] != ND, axis=1)
    np.testing.assert_array_equal(res["n"][0], valid.sum(axis=0))


# ---------------------------------------------------------------------------
# Quality mosaic
# ---------------------------------------------------------------------------


def test_quality_mosaic_band_expr_callable(tmp_path):
    data = _stack(seed=11, n_obs=6, n_bands=3)
    imgs = _write_stack(tmp_path, data)
    outs = []
    for idx, qual in enumerate([3, "b3", lambda blk: blk[2]]):
        out = str(tmp_path / f"qm{idx}.tif")
        comp.create_quality_mosaic(
            imgs, out, qual, out_ref_img=str(tmp_path / f"ref{idx}.tif"),
            out_quality_img=str(tmp_path / f"q{idx}.tif"), use_pbar=False,
        )
        outs.append([_read(out), _read(tmp_path / f"ref{idx}.tif"), _read(tmp_path / f"q{idx}.tif")])
    for other in outs[1:]:
        for arr_a, arr_b in zip(outs[0], other):
            np.testing.assert_array_equal(arr_a, arr_b)
    valid = _valid(data)
    idx = _np_select(data[:, 2], valid, "max")
    np.testing.assert_array_equal(outs[0][0], _np_gather(data, idx))
    np.testing.assert_array_equal(outs[0][1][0], idx + 1)
    exp_q = np.where(idx >= 0, np.take_along_axis(data[:, 2], np.maximum(idx, 0)[None], 0)[0], np.nan)
    np.testing.assert_allclose(outs[0][2][0], exp_q, equal_nan=True)


@pytest.mark.parametrize("mode", ["max", "min"])
def test_quality_mosaic_matches_numpy(tmp_path, mode):
    data = _stack(seed=13, n_obs=7, n_bands=3)
    data[:, 1, 5, 5] = 0.0  # NDVI 0 / 0 in every observation
    data[:, 2, 5, 5] = 0.0
    data[2, 1:, 6, 6] = 0.0  # 0 / 0 in observation 2 only
    data[:, :, 7, 7] = 100.0  # identical observations: a tie
    imgs = _write_stack(tmp_path, data)
    out = str(tmp_path / "qm.tif")
    comp.create_quality_mosaic(
        imgs, out, "(b3 - b2) / (b3 + b2)", mode=mode,
        out_ref_img=str(tmp_path / "ref.tif"), use_pbar=False,
    )
    qual = _ndvi(data)
    cand = _valid(data) & np.isfinite(qual)
    idx = _np_select(qual, cand, mode)
    ref = _read(tmp_path / "ref.tif")[0]
    np.testing.assert_array_equal(ref, idx + 1)
    np.testing.assert_array_equal(_read(out), _np_gather(data, idx))
    assert ref[5, 5] == 0 and np.all(_read(out)[:, 5, 5] == ND)
    assert ref[6, 6] != 3
    assert ref[7, 7] == 1  # ties go to the earliest image


def test_max_ndvi_equals_quality_mosaic(tmp_path):
    data = _stack(seed=17, n_obs=5, n_bands=4)
    imgs = _write_stack(tmp_path, data)
    comp.create_max_ndvi_composite(
        imgs, str(tmp_path / "a.tif"), 3, 4, out_ref_img=str(tmp_path / "a_ref.tif"),
        use_pbar=False,
    )
    comp.create_quality_mosaic(
        imgs, str(tmp_path / "b.tif"), "(b4 - b3) / (b4 + b3)",
        out_ref_img=str(tmp_path / "b_ref.tif"), use_pbar=False,
    )
    np.testing.assert_array_equal(_read(tmp_path / "a.tif"), _read(tmp_path / "b.tif"))
    np.testing.assert_array_equal(_read(tmp_path / "a_ref.tif"), _read(tmp_path / "b_ref.tif"))


def _np_pct_select(qual, cand, pct):
    """Observation nearest the nanpercentile (ties: higher quality, then earliest)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        target = np.nanpercentile(np.where(cand, qual, np.nan), pct, axis=0)
    idx = np.full(qual.shape[1:], -1)
    for row in range(qual.shape[1]):
        for col in range(qual.shape[2]):
            best = None
            for t_idx in range(qual.shape[0]):
                if not cand[t_idx, row, col]:
                    continue
                key = (abs(qual[t_idx, row, col] - target[row, col]), -qual[t_idx, row, col], t_idx)
                if best is None or key < best:
                    best = key
            if best is not None:
                idx[row, col] = best[2]
    return idx


def test_quality_mosaic_percentile(tmp_path):
    data = _stack(seed=19, n_obs=8, n_bands=3)
    data[:, :, 4, 4] = np.arange(8)[:, None] * 10.0 + 50.0  # distinct values
    data[3:5, :, 9, 9] = 77.0  # equal qualities
    imgs = _write_stack(tmp_path, data)
    qual = data[:, 0]
    cand = _valid(data)
    for pct in (90.0, 33.3, 50.0):
        out = str(tmp_path / "p.tif")
        comp.create_quality_mosaic(
            imgs, out, 1, mode="percentile", percentile=pct,
            out_ref_img=str(tmp_path / "p_ref.tif"), use_pbar=False, calc_stats=False,
        )
        idx = _np_pct_select(qual, cand, pct)
        np.testing.assert_array_equal(_read(tmp_path / "p_ref.tif")[0], idx + 1)
        np.testing.assert_array_equal(_read(out), _np_gather(data, idx))
    for pct, mode in ((100.0, "max"), (0.0, "min")):
        comp.create_quality_mosaic(
            imgs, str(tmp_path / "p.tif"), 1, mode="percentile", percentile=pct,
            out_ref_img=str(tmp_path / "p_ref.tif"), use_pbar=False, calc_stats=False,
        )
        comp.create_quality_mosaic(
            imgs, str(tmp_path / "m.tif"), 1, mode=mode,
            out_ref_img=str(tmp_path / "m_ref.tif"), use_pbar=False, calc_stats=False,
        )
        np.testing.assert_array_equal(_read(tmp_path / "p.tif"), _read(tmp_path / "m.tif"))
        np.testing.assert_array_equal(
            _read(tmp_path / "p_ref.tif"), _read(tmp_path / "m_ref.tif")
        )
    with pytest.raises(rsgispy.RSGISPyException):
        comp.quality_mosaic_node([ArraySource(data[0])], 1, mode="percentile")
    with pytest.raises(rsgispy.RSGISPyException):
        comp.quality_mosaic_node([ArraySource(data[0])], 1, mode="max", percentile=50)


def test_quality_mosaic_smoothing(tmp_path):
    shape = (12, 14)
    data = np.zeros((3, 2) + shape)
    data[0, 0] = 10.0  # observation 1 is the best nearly everywhere
    data[1, 0] = 5.0
    data[2, 0] = 1.0
    data[:, 1] = np.arange(3)[:, None, None] + 1.0
    data[1, 0, 5, 5] = 20.0  # an isolated pixel from observation 2
    data[1, 0, 2, 9] = 20.0  # isolated, but observation 1 is invalid there
    data[0, :, 2, 9] = ND
    data[:, :, 10, 1] = ND  # an empty pixel
    imgs = _write_stack(tmp_path, data)
    raw = str(tmp_path / "raw_ref.tif")
    comp.create_quality_mosaic(imgs, str(tmp_path / "raw.tif"), 1, out_ref_img=raw, use_pbar=False)
    assert _read(raw)[0, 5, 5] == 2 and _read(raw)[0, 2, 9] == 2
    refs = []
    for bsize in (256, 4, 5):
        out = str(tmp_path / f"s{bsize}.tif")
        comp.create_quality_mosaic(
            imgs, out, 1, smooth_ref_size=3, out_ref_img=str(tmp_path / f"s{bsize}_ref.tif"),
            out_quality_img=str(tmp_path / f"s{bsize}_q.tif"), block_size=bsize,
            use_pbar=False, calc_stats=False,
        )
        refs.append((_read(out), _read(tmp_path / f"s{bsize}_ref.tif"), _read(tmp_path / f"s{bsize}_q.tif")))
    ref = refs[0][1][0]
    assert ref[5, 5] == 1  # takes the neighbours' observation
    assert ref[2, 9] == 2  # the majority observation is invalid: keeps its own
    assert ref[10, 1] == 0
    assert refs[0][2][0, 5, 5] == 10.0  # the quality of the smoothed choice
    np.testing.assert_array_equal(refs[0][0][:, 5, 5], data[0][:, 5, 5])
    for other in refs[1:]:
        for arr_a, arr_b in zip(refs[0], other):
            np.testing.assert_array_equal(arr_a, arr_b)


def test_quality_mosaic_node_matches_file(tmp_path):
    data = _stack(seed=23, n_obs=5, n_bands=3)
    imgs = _write_stack(tmp_path, data)
    out = str(tmp_path / "qm.tif")
    comp.create_quality_mosaic(
        imgs, out, "b1 - b2", mode="percentile", percentile=75, smooth_ref_size=3,
        out_ref_img=str(tmp_path / "ref.tif"), out_quality_img=str(tmp_path / "q.tif"),
        datatype=rsgispy.TYPE_64FLOAT, use_pbar=False,
    )
    srcs = [ArraySource(d, transform=TRANSFORM, nodata=ND) for d in data]
    node = comp.quality_mosaic_node(srcs, "b1 - b2", "percentile", 75, smooth_ref_size=3)
    res = TileGraph(
        [MemorySink(node, name="c"), MemorySink(node.ref, name="r"),
         MemorySink(node.quality, name="q")], block_size=6,
    ).run(use_pbar=False, n_jobs=2)
    np.testing.assert_array_equal(res["c"], _read(out))
    np.testing.assert_array_equal(res["r"], _read(tmp_path / "ref.tif"))
    np.testing.assert_allclose(res["q"], _read(tmp_path / "q.tif"), rtol=1e-6, equal_nan=True)
    assert res["r"].dtype == np.uint16


# ---------------------------------------------------------------------------
# Land / water
# ---------------------------------------------------------------------------


def _lw_scene(seed=29, n_obs=7):
    """Observations with bands (green, red, NIR), land on the left half."""
    data = _stack(seed=seed, n_obs=n_obs, n_bands=3)
    rng = np.random.default_rng(seed)
    water = np.zeros(SHAPE, dtype=bool)
    water[:, SHAPE[1] // 2:] = True
    for t_idx in range(n_obs):
        nir = np.where(water, rng.uniform(10, 400, SHAPE), rng.uniform(1500, 4000, SHAPE))
        data[t_idx, 2] = np.where(data[t_idx, 2] == ND, ND, nir)
    return data, water


def test_land_water_with_mask(tmp_path):
    data, water = _lw_scene()
    imgs = _write_stack(tmp_path, data)
    msk = np.where(water, 2, 1).astype(np.uint8)
    msk[3, 3] = 0  # neither: no data
    msk_img = _write(tmp_path / "msk.tif", msk, dtype="uint8", nodata=0)
    out = str(tmp_path / "lw.tif")
    comp.create_land_water_composite(
        imgs, out, "(b3 - b2) / (b3 + b2)", "(b1 - b3) / (b1 + b3)", lw_msk_img=msk_img,
        out_ref_img=str(tmp_path / "ref.tif"), out_lw_msk_img=str(tmp_path / "lwm.tif"),
        use_pbar=False,
    )
    for qual_exp, cls in (("(b3 - b2) / (b3 + b2)", 1), ("(b1 - b3) / (b1 + b3)", 2)):
        comp.create_quality_mosaic(
            imgs, str(tmp_path / "qm.tif"), qual_exp, out_ref_img=str(tmp_path / "qm_ref.tif"),
            use_pbar=False,
        )
        sel = msk == cls
        np.testing.assert_array_equal(_read(out)[:, sel], _read(tmp_path / "qm.tif")[:, sel])
        np.testing.assert_array_equal(
            _read(tmp_path / "ref.tif")[0][sel], _read(tmp_path / "qm_ref.tif")[0][sel]
        )
    lwm = _read(tmp_path / "lwm.tif")[0]
    exp = msk.astype(float)
    exp[~_valid(data).any(axis=0)] = 0
    np.testing.assert_array_equal(lwm, exp)
    assert np.all(_read(out)[:, 3, 3] == ND) and _read(tmp_path / "ref.tif")[0, 3, 3] == 0
    with rasterio.open(tmp_path / "lwm.tif") as src:
        assert src.dtypes[0] == "uint8" and src.nodata == 0
        assert src.colormap(1)[2][:3] == (0x1E, 0x88, 0xE5)


def test_land_water_vote_rule(tmp_path):
    data, _ = _lw_scene(seed=31, n_obs=6)
    # a pixel that is water in 3 of 6 observations, one in 2 of 6
    data[:, :, 4, 4] = 100.0
    data[:3, 2, 4, 4] = 3000.0
    data[:, :, 6, 6] = 100.0
    data[:4, 2, 6, 6] = 3000.0
    imgs = _write_stack(tmp_path, data)
    valid = _valid(data)
    votes = valid & (data[:, 2] < 500)
    frac = votes.sum(axis=0) / np.maximum(valid.sum(axis=0), 1)
    for min_frac in (0.5, 0.34, 0.6, 1.0):
        comp.create_land_water_composite(
            imgs, str(tmp_path / "lw.tif"), 3, 1, water_rule="b3 < 500 ? 1 : 0",
            min_water_frac=min_frac, out_lw_msk_img=str(tmp_path / "lwm.tif"),
            use_pbar=False, calc_stats=False,
        )
        lwm = _read(tmp_path / "lwm.tif")[0]
        exp = np.where(valid.any(axis=0), np.where(frac >= min_frac, 2, 1), 0)
        np.testing.assert_array_equal(lwm, exp)
    assert frac[4, 4] == 0.5 and frac[6, 6] == pytest.approx(1 / 3)
    # the same rule as an index with water_rule_mode='min' (water where < 0)
    comp.create_land_water_composite(
        imgs, str(tmp_path / "lw2.tif"), 3, 1, water_rule="b3 - 500", water_rule_mode="min",
        out_lw_msk_img=str(tmp_path / "lwm2.tif"), use_pbar=False, calc_stats=False,
    )
    exp = np.where(valid.any(axis=0), np.where(frac >= 0.5, 2, 1), 0)
    np.testing.assert_array_equal(_read(tmp_path / "lwm2.tif")[0], exp)
    with pytest.raises(rsgispy.RSGISPyException):
        comp.create_land_water_composite(imgs, str(tmp_path / "x.tif"), 3, 1, use_pbar=False)


@pytest.mark.parametrize("method", ["quality", "medoid", "geomedian"])
def test_land_water_methods(tmp_path, method):
    data, _ = _lw_scene(seed=37)
    imgs = _write_stack(tmp_path, data)
    out = str(tmp_path / "lw.tif")
    comp.create_land_water_composite(
        imgs, out, 3, "b1" if method == "quality" else None, water_rule="b3 < 500 ? 1 : 0",
        water_method=method, out_ref_img=str(tmp_path / "ref.tif"),
        out_lw_msk_img=str(tmp_path / "lwm.tif"), datatype=rsgispy.TYPE_64FLOAT,
        use_pbar=False,
    )
    res = _read(out)
    ref = _read(tmp_path / "ref.tif")[0]
    lwm = _read(tmp_path / "lwm.tif")[0]
    valid = _valid(data)
    land = lwm == 1
    wat = lwm == 2
    assert land.any() and wat.any()
    idx_land = _np_select(data[:, 2], valid, "max")
    np.testing.assert_array_equal(ref[land], idx_land[land] + 1)
    np.testing.assert_array_equal(res[:, land], _np_gather(data, idx_land)[:, land])
    if method == "quality":
        idx_w = _np_select(data[:, 0], valid, "max")
        np.testing.assert_array_equal(res[:, wat], _np_gather(data, idx_w)[:, wat])
    elif method == "medoid":
        out_m = str(tmp_path / "med.tif")
        comp.create_medoid_composite(imgs, out_m, out_ref_img=str(tmp_path / "mref.tif"), use_pbar=False)
        np.testing.assert_array_equal(res[:, wat], _read(out_m)[:, wat])
        np.testing.assert_array_equal(ref[wat], _read(tmp_path / "mref.tif")[0][wat])
    else:
        out_g = str(tmp_path / "gm.tif")
        comp.create_geomedian_composite(
            imgs, out_g, out_ref_img=str(tmp_path / "gref.tif"), datatype=rsgispy.TYPE_64FLOAT,
            use_pbar=False,
        )
        np.testing.assert_allclose(res[:, wat], _read(out_g)[:, wat], rtol=1e-12)
        np.testing.assert_array_equal(ref[wat], _read(tmp_path / "gref.tif")[0][wat])


def test_land_water_percentile_per_class(tmp_path):
    data, water = _lw_scene(seed=41)
    imgs = _write_stack(tmp_path, data)
    msk_img = _write(tmp_path / "msk.tif", np.where(water, 2, 1), dtype="uint8", nodata=0)
    comp.create_land_water_composite(
        imgs, str(tmp_path / "lw.tif"), 3, 1, land_mode="percentile", land_percentile=80,
        water_mode="percentile", water_percentile=20, lw_msk_img=msk_img,
        out_ref_img=str(tmp_path / "ref.tif"), use_pbar=False,
    )
    ref = _read(tmp_path / "ref.tif")[0]
    valid = _valid(data)
    idx_l = _np_pct_select(data[:, 2], valid, 80)
    idx_w = _np_pct_select(data[:, 0], valid, 20)
    np.testing.assert_array_equal(ref[~water], idx_l[~water] + 1)
    np.testing.assert_array_equal(ref[water], idx_w[water] + 1)


def test_land_water_smoothing(tmp_path):
    shape = (12, 14)
    data = np.zeros((3, 2) + shape)
    data[:, 0] = np.array([10.0, 5.0, 1.0])[:, None, None]  # land quality (b1): obs 1
    data[:, 1] = np.array([1.0, 5.0, 10.0])[:, None, None]  # water quality (b2): obs 3
    data[1, 0, 3, 3] = 20.0  # an isolated land pixel choosing obs 2
    data[1, 1, 3, 10] = 20.0  # an isolated water pixel choosing obs 2
    imgs = _write_stack(tmp_path, data)
    msk = np.ones(shape, dtype=np.uint8)
    msk[:, 7:] = 2
    msk[8, 3] = 2  # an isolated water pixel in the land
    msk_img = _write(tmp_path / "msk.tif", msk, dtype="uint8", nodata=0)
    for bsize in (256, 5):
        comp.create_land_water_composite(
            imgs, str(tmp_path / "lw.tif"), 1, 2, lw_msk_img=msk_img, smooth_lw_msk_size=3,
            smooth_ref_size=3, out_ref_img=str(tmp_path / "ref.tif"),
            out_lw_msk_img=str(tmp_path / "lwm.tif"), block_size=bsize, use_pbar=False,
        )
        lwm = _read(tmp_path / "lwm.tif")[0]
        ref = _read(tmp_path / "ref.tif")[0]
        assert lwm[8, 3] == 1  # the isolated water pixel is removed
        exp_lw = msk.astype(float)
        exp_lw[8, 3] = 1
        np.testing.assert_array_equal(lwm, exp_lw)
        assert ref[3, 3] == 1 and ref[3, 10] == 3
        # within each class: land pixels at the coast keep land observations
        np.testing.assert_array_equal(ref[lwm == 1], 1)
        np.testing.assert_array_equal(ref[lwm == 2], 3)


def _tutorial_numpy(data, red=3, nir=4, nir_thres=500):
    """The land / water mask and land selection of the 10/02 notebook (NumPy)."""
    vld = np.all(data != ND, axis=1)
    nir_v = data[:, nir]
    land_votes = np.sum(vld & (nir_v > nir_thres), axis=0)
    water_votes = np.sum(vld & (nir_v <= nir_thres), axis=0)
    lw_msk = np.zeros(SHAPE)
    lw_msk[(land_votes + water_votes) > 0] = 1
    lw_msk[water_votes > land_votes] = 2
    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = (data[:, nir] - data[:, red]) / (data[:, nir] + data[:, red])
    ndvi[~np.isfinite(ndvi)] = -1.0
    best = np.full(SHAPE, -np.inf)
    ref = np.zeros(SHAPE)
    for t_idx in range(data.shape[0]):
        better = vld[t_idx] & (ndvi[t_idx] > best)
        best[better] = ndvi[t_idx][better]
        ref[better] = t_idx + 1
    return lw_msk, ref


def test_max_ndvi_ndwi_preset(tmp_path):
    data = _stack(seed=43, n_obs=6, n_bands=6, integer=True)
    rng = np.random.default_rng(43)
    for t_idx in range(6):
        nir = np.where(rng.random(SHAPE) < 0.5, rng.uniform(10, 600, SHAPE), rng.uniform(450, 3000, SHAPE))
        data[t_idx, 4] = np.where(data[t_idx, 4] == ND, ND, np.round(nir))
    imgs = _write_stack(tmp_path, data, dtype="int32")
    # The tutorial's options: NIR vote (majority, ties land) and the medoid
    # over water.
    comp.create_max_ndvi_ndwi_composite(
        imgs, str(tmp_path / "t.tif"), 4, 5, green_band=3, water_rule="b5 <= 500 ? 1 : 0",
        min_water_frac=0.501, water_method="medoid", out_ref_img=str(tmp_path / "t_ref.tif"),
        out_lw_msk_img=str(tmp_path / "t_lw.tif"), use_pbar=False,
    )
    lw_exp, ref_exp = _tutorial_numpy(data)
    lwm = _read(tmp_path / "t_lw.tif")[0]
    np.testing.assert_array_equal(lwm, lw_exp)
    land = lwm == 1
    np.testing.assert_array_equal(_read(tmp_path / "t_ref.tif")[0][land], ref_exp[land])
    # The defaults: RSGISLib's rule, NDVI over land and McFeeters NDWI over water.
    comp.create_max_ndvi_ndwi_composite(
        imgs, str(tmp_path / "d.tif"), 4, 5, green_band=3, out_ref_img=str(tmp_path / "d_ref.tif"),
        use_pbar=False,
    )
    comp.create_land_water_composite(
        imgs, str(tmp_path / "e.tif"), "(b5 - b4) / (b5 + b4)", "(b3 - b5) / (b3 + b5)",
        water_rule=comp.RSGISLibLandWaterRule(4, 5, green_band=3),
        out_ref_img=str(tmp_path / "e_ref.tif"), use_pbar=False,
    )
    np.testing.assert_array_equal(_read(tmp_path / "d.tif"), _read(tmp_path / "e.tif"))
    np.testing.assert_array_equal(_read(tmp_path / "d_ref.tif"), _read(tmp_path / "e_ref.tif"))
    # RSGISLib's rule on one observation
    rule = comp.RSGISLibLandWaterRule(1, 2, swir_band=3)
    blk = np.array([[100.0, 100.0, 0.0, 100.0], [1000.0, 110.0, 0.0, 120.0], [50.0, 50.0, 0.0, 500.0]])
    blk = blk[:, np.newaxis, :]
    res = rule(blk)[0]
    assert res[0] == 0.0  # NDVI > 0.3: land
    assert res[1] == 1.0  # NDWI (NIR, SWIR) > 0.01: water
    assert np.isnan(res[2])  # NDVI undefined: no vote
    assert res[3] == 0.0  # neither: land
    with pytest.raises(rsgispy.RSGISPyException):
        comp.create_max_ndvi_ndwi_composite(imgs, str(tmp_path / "x.tif"), 4, 5)


def test_land_water_node_matches_file(tmp_path):
    data, _ = _lw_scene(seed=47)
    imgs = _write_stack(tmp_path, data)
    kwargs = {
        "water_rule": "b3 < 500 ? 1 : 0", "water_method": "medoid",
        "smooth_lw_msk_size": 3, "smooth_ref_size": 3,
    }
    comp.create_land_water_composite(
        imgs, str(tmp_path / "lw.tif"), 3, None, out_ref_img=str(tmp_path / "ref.tif"),
        out_lw_msk_img=str(tmp_path / "lwm.tif"), datatype=rsgispy.TYPE_64FLOAT,
        use_pbar=False, **kwargs,
    )
    srcs = [ArraySource(d, transform=TRANSFORM, nodata=ND) for d in data]
    node = comp.land_water_composite_node(srcs, 3, None, **kwargs)
    res = TileGraph(
        [MemorySink(node, name="c"), MemorySink(node.ref, name="r"),
         MemorySink(node.lw_msk, name="m")], block_size=6,
    ).run(use_pbar=False, n_jobs=3)
    np.testing.assert_array_equal(res["c"], _read(tmp_path / "lw.tif"))
    np.testing.assert_array_equal(res["r"], _read(tmp_path / "ref.tif"))
    np.testing.assert_array_equal(res["m"], _read(tmp_path / "lwm.tif"))


# ---------------------------------------------------------------------------
# Medoid
# ---------------------------------------------------------------------------


def _np_medoid(data):
    valid = _valid(data)
    idx = np.full(SHAPE, -1)
    for row in range(SHAPE[0]):
        for col in range(SHAPE[1]):
            vld = np.nonzero(valid[:, row, col])[0]
            if vld.size == 0:
                continue
            pts = data[vld, :, row, col]
            sums = [np.sum(np.linalg.norm(pts - pnt, axis=1)) for pnt in pts]
            idx[row, col] = vld[int(np.argmin(sums))]
    return idx


def test_medoid_matches_numpy(tmp_path):
    data = _stack(seed=53, n_obs=6, n_bands=3, integer=True)
    data[:, :, 5, 5] = data[2, :, 5, 5]  # all identical: tie to the earliest
    data[1, :, 6, 6] = data[4, :, 6, 6]  # two identical observations
    data[:, :, 7, 7] = ND
    data[3, :, 7, 7] = 42.0  # one valid observation
    imgs = _write_stack(tmp_path, data, dtype="int16")
    out = str(tmp_path / "med.tif")
    comp.create_medoid_composite(imgs, out, out_ref_img=str(tmp_path / "ref.tif"), use_pbar=False)
    idx = _np_medoid(data)
    ref = _read(tmp_path / "ref.tif")[0]
    np.testing.assert_array_equal(ref, idx + 1)
    np.testing.assert_array_equal(_read(out), _np_gather(data, idx))
    assert ref[5, 5] == 1 and ref[7, 7] == 4 and ref[0, 0] == 0
    with rasterio.open(out) as src:
        assert src.dtypes[0] == "int16" and src.nodata == ND


# ---------------------------------------------------------------------------
# Percentiles
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("valid_rule", ["all_bands", "per_band"])
def test_percentile_composite_matches_numpy(tmp_path, valid_rule):
    data = _stack(seed=59, n_obs=7, n_bands=3)
    data[2, 0, 3:6, 3:6] = ND  # band 1 only missing
    imgs = _write_stack(tmp_path, data, names=["Blue", "Red", "NIR"])
    pcts = [10, 50, 2.5, 90]
    out = str(tmp_path / "pct.tif")
    comp.create_percentile_composite(
        imgs, out, pcts, valid_rule=valid_rule, out_count_img=str(tmp_path / "cnt.tif"),
        use_pbar=False,
    )
    res = _read(out)
    with rasterio.open(out) as src:
        assert src.dtypes[0] == "float32" and src.nodata == ND
        assert src.descriptions == tuple(
            f"{nm}_p{p:g}" for nm in ("Blue", "Red", "NIR") for p in pcts
        )
    valid = _valid(data)
    cnt = _read(tmp_path / "cnt.tif")
    for b_idx in range(3):
        band = np.where(data[:, b_idx] == ND, np.nan, data[:, b_idx])
        if valid_rule == "all_bands":
            band = np.where(valid, band, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            exp = np.nanpercentile(band, pcts, axis=0)
        exp = np.where(np.isnan(exp), ND, exp).astype(np.float32)
        np.testing.assert_allclose(res[b_idx * 4:(b_idx + 1) * 4], exp, rtol=1e-6)
        if valid_rule == "per_band":
            np.testing.assert_array_equal(cnt[b_idx], np.isfinite(band).sum(axis=0))
    if valid_rule == "all_bands":
        assert cnt.shape[0] == 1
        np.testing.assert_array_equal(cnt[0], valid.sum(axis=0))
    else:
        with rasterio.open(tmp_path / "cnt.tif") as src:
            assert src.descriptions == ("Blue_count", "Red_count", "NIR_count")


def test_percentile_composite_options(tmp_path):
    data = _stack(seed=61, n_obs=6, n_bands=3)
    imgs = _write_stack(tmp_path, data)
    outs = []
    for bsize in (256, 7):
        out = str(tmp_path / f"p{bsize}.tif")
        comp.create_percentile_composite(
            imgs, out, [33.3, 50], bands=[3, 1], datatype=rsgispy.TYPE_16INT,
            block_size=bsize, use_pbar=False,
        )
        outs.append(_read(out))
    np.testing.assert_array_equal(outs[0], outs[1])
    with rasterio.open(tmp_path / "p256.tif") as src:
        assert src.descriptions == ("b3_p33.3", "b3_p50", "b1_p33.3", "b1_p50")
        assert src.dtypes[0] == "int16"
    valid = _valid(data)
    band = np.where(valid, data[:, 2], np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        exp = np.nanpercentile(band, 33.3, axis=0)
    exp = np.where(np.isnan(exp), ND, np.rint(exp))  # rounded, not truncated
    np.testing.assert_array_equal(outs[0][0], exp)
    # the node agrees with the file-based function
    srcs = [ArraySource(d, transform=TRANSFORM, nodata=ND) for d in data]
    node = comp.percentile_composite_node(srcs, [33.3, 50], bands=[3, 1], dtype=np.int16)
    res = TileGraph(MemorySink(node, name="p"), block_size=5).run(use_pbar=False)["p"]
    np.testing.assert_array_equal(res, outs[0])


# ---------------------------------------------------------------------------
# Common checks
# ---------------------------------------------------------------------------


def test_inputs_must_match(tmp_path):
    data = _stack(seed=67, n_obs=3, n_bands=2)
    imgs = _write_stack(tmp_path, data)
    shifted = _write(
        tmp_path / "shift.tif", data[0], transform=from_origin(500010.0, 5700000.0, 10.0, 10.0)
    )
    three = _write(tmp_path / "three.tif", np.concatenate([data[0], data[0][:1]]))
    for func in (comp.create_geomedian_composite, comp.create_medoid_composite,
                 comp.create_percentile_composite):
        with pytest.raises(rsgispy.RSGISPyException):
            func(imgs + [shifted], str(tmp_path / "x.tif"), use_pbar=False)
        with pytest.raises(rsgispy.RSGISPyException):
            func(imgs + [three], str(tmp_path / "x.tif"), use_pbar=False)
    with pytest.raises(rsgispy.RSGISPyException):
        comp.create_quality_mosaic(imgs, str(tmp_path / "x.tif"), 3, use_pbar=False)
    with pytest.raises(rsgispy.RSGISPyException):
        comp.create_quality_mosaic(imgs, str(tmp_path / "x.tif"), "b3 + 1", use_pbar=False)
    with pytest.raises(rsgispy.RSGISPyException):
        comp.create_geomedian_composite(imgs, str(tmp_path / "x.tif"), bands=[3], use_pbar=False)


def test_no_nodata_in_inputs(tmp_path):
    data = np.random.default_rng(71).normal(100.0, 10.0, size=(4, 2) + SHAPE)
    data[1, :, 2, 2] = np.nan
    imgs = [
        _write(tmp_path / f"nn{i}.tif", data[i], dtype="float32", nodata=None) for i in range(4)
    ]
    out = str(tmp_path / "med.tif")
    comp.create_medoid_composite(imgs, out, use_pbar=False)
    with rasterio.open(out) as src:
        assert src.nodata is not None and np.isnan(src.nodata)
    assert np.all(np.isfinite(_read(out)))


def test_thematic_outputs_have_class_names(tmp_path):
    """The reference image and land/water mask store class names."""
    from rsgispy.imageutils.colourtable import get_thematic_classes

    rng = np.random.default_rng(11)
    paths = []
    for idx in range(3):
        arr = rng.integers(100, 3000, size=(4, 16, 16)).astype(np.uint16)
        path = str(tmp_path / f"img_{idx}.tif")
        with rasterio.open(
            path, "w", driver="GTiff", width=16, height=16, count=4, dtype="uint16", nodata=0,
            crs="EPSG:32630", transform=rasterio.transform.from_origin(0, 160, 10, 10),
        ) as dst:
            dst.write(arr)
        paths.append(path)
    ref_img = str(tmp_path / "ref.tif")
    lw_img = str(tmp_path / "lw.tif")
    comp.create_land_water_composite(
        paths, str(tmp_path / "comp.tif"), "b4", "b2", water_rule="b2 - b4",
        out_ref_img=ref_img, out_lw_msk_img=lw_img, use_pbar=False,
    )
    ref_names = {k: v[0] for k, v in get_thematic_classes(ref_img).items()}
    assert ref_names == {1: "img_0.tif", 2: "img_1.tif", 3: "img_2.tif"}
    lw_names = {k: v[0] for k, v in get_thematic_classes(lw_img).items()}
    assert lw_names == {1: "Land", 2: "Water"}
