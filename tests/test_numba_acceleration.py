"""
Tests for optional ``numba`` acceleration (:mod:`rsgispy._numba`) and the
numerical equivalence of compiled kernels with their pure-Python / NumPy
fallbacks.

Compiled-vs-fallback comparisons are skipped when ``numba`` is not installed;
the infrastructure and no-numba import tests always run.
"""

import math
import os
import subprocess
import sys
import textwrap
import warnings

import numpy as np
import pytest
from numpy.lib.stride_tricks import sliding_window_view

from rsgispy._numba import (
    HAS_NUMBA,
    OptionalJitFunction,
    disable_numba,
    numba_enabled,
    optional_njit,
    reset_numba_warnings,
    use_compiled,
    warn_numba_fallback,
)

needs_numba = pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Infrastructure
# ---------------------------------------------------------------------------


def _double_py(arr):
    out = np.empty_like(arr)
    for i in range(arr.shape[0]):
        out[i] = arr[i] * 2.0
    return out


_double = optional_njit(_double_py)


def test_optional_njit_returns_wrapper_and_matches():
    arr = np.arange(10, dtype=np.float64)
    assert isinstance(_double, OptionalJitFunction)
    assert _double.py_func is _double_py
    assert _double.__name__ == "_double_py"
    np.testing.assert_array_equal(_double(arr), arr * 2.0)
    with disable_numba():
        assert not numba_enabled()
        np.testing.assert_array_equal(_double(arr), arr * 2.0)
    assert numba_enabled() == HAS_NUMBA


def test_optional_njit_decorator_with_options_and_fallback():
    calls = []

    def _fallback(x):
        calls.append(x)
        return x + 1

    @optional_njit(parallel=False, cache=False, fastmath=False, fallback=_fallback)
    def _inc(x):
        return x + 1

    assert _inc.jit_options == {"cache": False, "fastmath": False, "nogil": False}
    with disable_numba():
        assert _inc(3) == 4
    assert calls == [3]
    assert _inc(5) == 6
    if HAS_NUMBA:
        assert calls == [3]  # compiled path used
        assert _inc.jit_func is not None
        assert _inc.is_compiled_available


def test_disable_numba_is_silent():
    reset_numba_warnings()

    @optional_njit(cache=False)
    def _ident(x):
        return x

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with disable_numba():
            assert _ident(7) == 7
            assert use_compiled(_ident) is False


@needs_numba
def test_compile_failure_falls_back_with_single_warning():
    reset_numba_warnings()

    @optional_njit(cache=False)
    def _untypable(obj):
        return obj.some_attribute_numba_cannot_type  # pragma: no cover

    class _Obj:
        some_attribute_numba_cannot_type = 42

    with pytest.warns(RuntimeWarning, match="numba could not compile"):
        assert _untypable(_Obj()) == 42
    assert _untypable.jit_func is None
    # De-duplicated: no second warning
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _untypable(_Obj()) == 42


def test_warn_numba_fallback_once_deduplicates():
    reset_numba_warnings()
    with pytest.warns(RuntimeWarning, match="dedup_test_func"):
        warn_numba_fallback("dedup_test_func", once=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_numba_fallback("dedup_test_func", once=True)
    with pytest.warns(RuntimeWarning, match="the NumPy path"):
        warn_numba_fallback("other_func", fallback_desc="the NumPy path")


def test_env_var_disables_numba():
    code = textwrap.dedent(
        """
        import warnings
        warnings.simplefilter("error")
        from rsgispy import _numba
        assert _numba.HAS_NUMBA is False
        assert _numba.numba_enabled() is False
        from rsgispy._unionfind import uf_min_root_kernel
        import numpy as np
        lut = uf_min_root_kernel(np.array([3]), np.array([1]), 4, 0)
        assert lut.tolist() == [0, 1, 2, 1]
        print("OK")
        """
    )
    env = dict(os.environ, RSGISPY_DISABLE_NUMBA="1")
    res = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, res.stderr
    assert "OK" in res.stdout


def test_import_and_run_without_numba_installed():
    """Block ``import numba`` and check modules import and fall back."""
    code = textwrap.dedent(
        """
        import sys, warnings
        sys.modules["numba"] = None  # makes `import numba` raise ImportError
        import numpy as np
        from rsgispy import _numba
        assert _numba.HAS_NUMBA is False and _numba.NUMBA_INSTALLED is False
        import rsgispy.elevation.filtering
        import rsgispy.elevation.illumination
        import rsgispy.elevation.hydrology
        import rsgispy.imagecalc.leastcostpath as lcp
        import rsgispy.imagefilter._glcm_helpers
        import rsgispy.imagefilter._helpers as fh
        import rsgispy.imageregistration.tiepoints
        import rsgispy.segmentation.slic
        import rsgispy.segmentation.tiling as tiling

        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            lut = tiling.reconcile_seam_equivalences(4, [(4, 2), (2, 1)])
            path = lcp._dijkstra_geometric_path(
                np.ones((3, 3)), (0, 0), (2, 2)
            )
        assert lut.tolist() == [0, 1, 1, 3, 1], lut
        assert path == [(0, 0), (1, 1), (2, 2)], path
        msgs = [str(w.message) for w in rec if w.category is RuntimeWarning]
        assert any("numba is not installed" in m for m in msgs), msgs
        print("OK")
        """
    )
    res = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, res.stderr
    assert "OK" in res.stdout


def test_pyproject_numba_is_optional():
    tomllib = pytest.importorskip("tomllib")
    with open(os.path.join(REPO_ROOT, "pyproject.toml"), "rb") as fp:
        cfg = tomllib.load(fp)
    deps = cfg["project"]["dependencies"]
    extras = cfg["project"]["optional-dependencies"]
    assert not any(d.lower().startswith("numba") for d in deps)
    assert any(d.lower().startswith("numba") for d in extras["numba"])
    assert any("numba" in d for d in extras["all"])


# ---------------------------------------------------------------------------
# High-value kernels
# ---------------------------------------------------------------------------


@needs_numba
def test_dijkstra_kernel_matches_python():
    from rsgispy.imagecalc.leastcostpath import (
        _dijkstra_geometric_path,
        _dijkstra_geometric_path_py,
    )
    import rsgispy

    rng = np.random.default_rng(0)
    for trial in range(30):
        h, w = (int(v) for v in rng.integers(2, 30, 2))
        cost = rng.integers(1, 4, (h, w)).astype(np.float64)  # many ties
        if trial % 3 == 0:
            cost[rng.random((h, w)) < 0.25] = np.inf
        start = (int(rng.integers(h)), int(rng.integers(w)))
        stop = (int(rng.integers(h)), int(rng.integers(w)))
        cost[start] = 1.0
        cost[stop] = 1.0
        try:
            expected = _dijkstra_geometric_path_py(cost, start, stop)
        except rsgispy.RSGISPyException:
            with pytest.raises(rsgispy.RSGISPyException):
                _dijkstra_geometric_path(cost, start, stop)
            continue
        assert _dijkstra_geometric_path(cost, start, stop) == expected


@needs_numba
def test_priority_flood_kernel_matches_python():
    """The shared hydrology priority-flood equals the Soille & Gratin fallback."""
    from rsgispy.elevation._hydro_kernels import priority_flood_kernel
    from rsgispy.elevation.filtering import _priority_flood_fill_py

    rng = np.random.default_rng(1)
    for _ in range(20):
        h, w = (int(v) for v in rng.integers(3, 50, 2))
        dem = rng.integers(-20, 60, (h, w)).astype(np.int16)
        vld = rng.random((h, w)) > 0.1
        vld_pad = np.pad(vld, 1)
        seed = vld & np.any(~sliding_window_view(vld_pad, (3, 3)), axis=(2, 3))
        out_a = np.zeros((h, w), dtype=np.int64)
        out_b = out_a.copy()
        vis_b = ~vld
        filled = priority_flood_kernel(dem.astype(np.float64), vld)
        out_a[vld] = np.rint(filled[vld]).astype(np.int64)
        _priority_flood_fill_py(dem, seed, vis_b, out_b)
        np.testing.assert_array_equal(out_a[vld], out_b[vld])


def _write_raster(path, arr, nodata=None, res=10.0):
    import rasterio
    from rasterio.transform import from_origin

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype=arr.dtype.name,
        crs="EPSG:32630",
        transform=from_origin(400000.0, 5600000.0, res, res),
        nodata=nodata,
    ) as dst:
        dst.write(arr, 1)


@needs_numba
def test_fill_dem_and_shadow_mask_end_to_end_equivalent(tmp_path):
    import rasterio
    from rsgispy.elevation.filtering import fill_dem_soille_gratin_1994
    from rsgispy.elevation.illumination import shadow_mask

    rng = np.random.default_rng(2)
    h, w = 60, 70
    yy, xx = np.mgrid[0:h, 0:w]
    dem_f = 300 + 80 * np.sin(xx / 8.0) + 60 * np.cos(yy / 6.0) + rng.normal(0, 4, (h, w))
    dem_i = dem_f.astype(np.int32)
    dem_i[5:9, 5:9] = -9999
    vld = np.ones((h, w), dtype=np.uint8)
    vld[:, :2] = 0
    _write_raster(str(tmp_path / "dem_i.tif"), dem_i, nodata=-9999)
    _write_raster(str(tmp_path / "vld.tif"), vld)
    _write_raster(str(tmp_path / "dem_f.tif"), dem_f.astype(np.float32), nodata=-9999)

    outputs = {}
    for label, ctx in (("jit", None), ("py", disable_numba)):
        fill_out = str(tmp_path / f"fill_{label}.tif")
        shad_out = str(tmp_path / f"shadow_{label}.tif")
        if ctx is None:
            fill_dem_soille_gratin_1994(
                str(tmp_path / "dem_i.tif"), str(tmp_path / "vld.tif"), fill_out,
                calc_stats=False, use_pbar=False,
            )
            shadow_mask(
                str(tmp_path / "dem_f.tif"), shad_out, 135.0, 70.0, 2000.0,
                calc_stats=False, use_pbar=False,
            )
        else:
            with ctx():
                fill_dem_soille_gratin_1994(
                    str(tmp_path / "dem_i.tif"), str(tmp_path / "vld.tif"), fill_out,
                    calc_stats=False, use_pbar=False,
                )
                shadow_mask(
                    str(tmp_path / "dem_f.tif"), shad_out, 135.0, 70.0, 2000.0,
                    calc_stats=False, use_pbar=False,
                )
        with rasterio.open(fill_out) as src:
            fill_arr = src.read(1)
        with rasterio.open(shad_out) as src:
            shad_arr = src.read(1)
        outputs[label] = (fill_arr, shad_arr)

    np.testing.assert_array_equal(outputs["jit"][0], outputs["py"][0])
    np.testing.assert_array_equal(outputs["jit"][1], outputs["py"][1])
    assert outputs["jit"][1].sum() > 0  # some cast shadows were traced


@needs_numba
def test_shadow_ray_kernel_matches_numpy():
    from rsgispy.elevation.illumination import (
        _shadow_ray_kernel,
        _shadow_ray_trace_numpy,
    )

    rng = np.random.default_rng(3)
    for _ in range(10):
        h, w = (int(v) for v in rng.integers(5, 60, 2))
        yy, xx = np.mgrid[0:h, 0:w]
        dem = 50 * np.sin(xx / 7.0) + 30 * np.cos(yy / 5.0) + rng.normal(0, 3, (h, w))
        valid = rng.random((h, w)) > 0.05
        dem[~valid] = np.nan
        cand = valid & (rng.random((h, w)) > 0.2)
        sun_zen = math.radians(float(rng.uniform(20, 85)))
        sun_az = math.radians(float(rng.uniform(0, 360)))
        x_step = math.sin(sun_zen) * math.cos(sun_az) / 3.0
        y_step = math.sin(sun_zen) * math.sin(sun_az) / 3.0
        z_step = math.cos(sun_zen) / 3.0
        max_steps = int(math.ceil(math.hypot(w, h) * 3.0)) + 2
        args = (dem, valid, cand, x_step, -y_step, z_step, 80.0, max_steps)
        np.testing.assert_array_equal(
            _shadow_ray_kernel(*args), _shadow_ray_trace_numpy(*args, False)
        )


@needs_numba
@pytest.mark.parametrize("n_labels", [0, 1, 50, 2000, 1_000_000])
def test_reconcile_seam_equivalences_matches_python(n_labels):
    from rsgispy.segmentation.tiling import (
        _reconcile_seam_equivalences_py,
        reconcile_seam_equivalences,
    )

    rng = np.random.default_rng(n_labels)
    n_pairs = min(200_000, 3 * n_labels + 5)
    pairs = rng.integers(-2, n_labels + 3, (n_pairs, 2))  # includes invalid ids
    expected = _reconcile_seam_equivalences_py(n_labels, pairs.tolist())
    got = reconcile_seam_equivalences(n_labels, pairs)
    np.testing.assert_array_equal(got, expected)
    np.testing.assert_array_equal(
        reconcile_seam_equivalences(n_labels, [tuple(p) for p in pairs[:50].tolist()]),
        _reconcile_seam_equivalences_py(n_labels, pairs[:50].tolist()),
    )
    assert got.dtype == np.int64
    if n_labels > 0:
        assert np.all(got[1:] <= np.arange(1, n_labels + 1))
    np.testing.assert_array_equal(
        reconcile_seam_equivalences(n_labels, []), np.arange(n_labels + 1)
    )


@needs_numba
def test_rle_union_find_matches_python_and_scipy():
    from rsgispy.segmentation._helpers import (
        _rle_union_find_scanline,
        _rle_union_find_scanline_py,
    )

    scipy_cc = pytest.importorskip("scipy.sparse.csgraph")
    from scipy.sparse import coo_matrix

    rng = np.random.default_rng(5)
    for _ in range(20):
        n = int(rng.integers(1, 400))
        n_e = int(rng.integers(0, 2 * n))
        u = rng.integers(0, n, n_e).astype(np.int32)
        v = rng.integers(0, n, n_e).astype(np.int32)
        n_a, lab_a = _rle_union_find_scanline(u, v, n)
        n_b, lab_b = _rle_union_find_scanline_py(u, v, n)
        assert n_a == n_b
        np.testing.assert_array_equal(lab_a, lab_b)
        if n_e:
            adj = coo_matrix((np.ones(n_e), (u, v)), shape=(n, n))
            n_c, _ = scipy_cc.connected_components(adj, directed=False)
            assert n_c == n_a


@needs_numba
@pytest.mark.parametrize("stitch_seams", [True, False])
def test_tiled_slic_equivalent_with_and_without_numba(stitch_seams):
    """Full tiled SLIC pipeline (seeds, seam union-find, RLE labelling)."""
    from rsgispy.segmentation.slic import slic_segmentation_array

    rng = np.random.default_rng(6)
    data = rng.normal(size=(3, 70, 80)).cumsum(axis=1)
    data[:, 10:14, 20:30] = -9999.0
    kwargs = dict(
        n_segments=60,
        tile_size=32,
        overlap=6,
        stitch_seams=stitch_seams,
        min_pixels=5,
        no_data_val=-9999.0,
    )
    lab_jit = slic_segmentation_array(data, **kwargs)
    with disable_numba():
        lab_py = slic_segmentation_array(data, **kwargs)
    assert lab_jit.max() > 1
    np.testing.assert_array_equal(lab_jit, lab_py)


@needs_numba
def test_slic_seed_kernel_matches_python():
    from rsgispy.segmentation import slic
    from rsgispy.segmentation.tiling import plan_tiles

    rng = np.random.default_rng(7)
    for _ in range(6):
        h, w = (int(v) for v in rng.integers(20, 70, 2))
        arr = rng.normal(size=(2, h, w))
        valid = rng.random((h, w)) > 0.1
        step = float(rng.uniform(3, 10))
        for tile in plan_tiles(h, w, tile_size=24, overlap=4):
            rs = slice(tile.halo_row_off, tile.halo_row_off + tile.halo_height)
            cs = slice(tile.halo_col_off, tile.halo_col_off + tile.halo_width)
            if not valid[rs, cs].any():
                continue
            args = (arr[:, rs, cs], valid[rs, cs], tile, step, w)
            res_jit = slic._initialise_global_aligned_seeds(*args)
            with disable_numba():
                res_py = slic._initialise_global_aligned_seeds(*args)
            for a_arr, b_arr in zip(res_jit, res_py):
                assert a_arr.dtype == b_arr.dtype
                np.testing.assert_array_equal(a_arr, b_arr)


# ---------------------------------------------------------------------------
# Medium-value kernels
# ---------------------------------------------------------------------------


@needs_numba
def test_mutual_information_kernel_matches_numpy():
    from rsgispy.imageregistration._kernels import _mutual_information_1d
    from rsgispy.tools.stats import _mutual_information

    rng = np.random.default_rng(8)
    for _ in range(100):
        n = int(rng.integers(2, 300))
        a = np.round(rng.normal(size=n), int(rng.integers(0, 3)))
        b = 0.5 * a + np.round(rng.normal(size=n), 1)
        bins = int(rng.integers(2, 70))
        for norm in (False, True):
            assert _mutual_information_1d(a, b, n, bins, norm) == pytest.approx(
                _mutual_information(a, b, hist_bins=bins, normalised=norm),
                rel=1e-12,
                abs=1e-12,
            )


@needs_numba
@pytest.mark.parametrize("metric_id", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("masked", [False, True])
def test_template_match_surface_matches_numpy(metric_id, masked):
    from rsgispy.imageregistration.tiepoints import _match_template_patch

    rng = np.random.default_rng(metric_id + 10 * masked)
    for _ in range(8):
        win_h, win_w = (int(v) for v in rng.integers(6, 14, 2))
        sa = int(rng.integers(1, 5))
        base = rng.normal(size=(win_h + 2 * sa + 4, win_w + 2 * sa + 4)).cumsum(0)
        flt = base[: win_h + 2 * sa, : win_w + 2 * sa]
        ref = base[sa + 1: sa + 1 + win_h, sa: sa + win_w] + rng.normal(0, 0.1, (win_h, win_w))
        ref_m = rng.random(ref.shape) > (0.2 if masked else 0.0)
        flt_m = rng.random(flt.shape) > (0.1 if masked else 0.0)
        maximise = metric_id in (4, 5, 6)
        res_jit = _match_template_patch(ref, flt, ref_m, flt_m, sa, metric_id, maximise, 4, 32)
        with disable_numba():
            res_py = _match_template_patch(ref, flt, ref_m, flt_m, sa, metric_id, maximise, 4, 32)
        if res_py is None:
            assert res_jit is None
            continue
        assert res_jit[0] == res_py[0] and res_jit[1] == res_py[1]
        assert res_jit[2] == pytest.approx(res_py[2], rel=1e-10, abs=1e-12)


@needs_numba
@pytest.mark.parametrize("metric_id", [1, 2, 3, 4, 5, 6])
def test_search_shift_surface_matches_numpy(metric_id):
    from rsgispy.imageregistration._helpers import search_shift_surface

    rng = np.random.default_rng(20 + metric_id)
    n_bands, h, w = 2, 24, 28
    img = rng.normal(size=(n_bands, h + 4, w + 4)).cumsum(1)
    ref = img[:, 2: 2 + h, 2: 2 + w]
    flt = img[:, 1: 1 + h, 3: 3 + w]
    ref_v = rng.random((h, w)) > 0.1
    flt_v = rng.random((h, w)) > 0.1
    args = (ref, flt, ref_v, flt_v, metric_id, metric_id in (4, 5, 6), 3, 2, 32, 16)
    surf_jit, cnt_jit = search_shift_surface(*args)
    with disable_numba():
        surf_py, cnt_py = search_shift_surface(*args)
    np.testing.assert_array_equal(cnt_jit, cnt_py)
    np.testing.assert_allclose(surf_jit, surf_py, rtol=1e-10, atol=1e-12, equal_nan=True)


@needs_numba
def test_non_finite_inputs_use_numpy_path():
    from rsgispy.imageregistration.tiepoints import _match_template_patch

    rng = np.random.default_rng(30)
    ref = rng.normal(size=(8, 8))
    flt = rng.normal(size=(12, 12))
    flt[3, 3] = np.inf
    ref_m = np.ones(ref.shape, dtype=bool)
    ref_m[0, 0] = False
    flt_m = np.ones(flt.shape, dtype=bool)
    res_jit = _match_template_patch(ref, flt, ref_m, flt_m, 2, 2, False, 1, 32)
    with disable_numba():
        res_py = _match_template_patch(ref, flt, ref_m, flt_m, 2, 2, False, 1, 32)
    assert res_jit == res_py


@needs_numba
def test_stat_mode_kernel_matches_numpy():
    from rsgispy.imagefilter._helpers import _stat_mode_numpy, stat_mode

    rng = np.random.default_rng(9)
    dtypes = [np.float64, np.uint8, np.int32, np.float32]
    for trial in range(24):
        h, w = (int(v) for v in rng.integers(3, 30, 2))
        k = int(rng.choice([3, 5, 7, 9]))
        halo = rng.integers(0, int(rng.integers(2, 8)), (h + k - 1, w + k - 1))
        halo = halo.astype(dtypes[trial % 4])
        if trial % 4 == 0:
            halo[rng.random(halo.shape) < 0.05] = np.nan
            halo[rng.random(halo.shape) < 0.03] = np.inf
            halo[rng.random(halo.shape) < 0.03] = -np.inf
        valid_halo = rng.random(halo.shape) > 0.15
        wins = sliding_window_view(halo, (k, k))
        v_wins = sliding_window_view(valid_halo, (k, k))
        centre_valid = valid_halo[k // 2: k // 2 + h, k // 2: k // 2 + w]
        np.testing.assert_array_equal(
            stat_mode(wins, None, v_wins, centre_valid),
            _stat_mode_numpy(wins, v_wins, centre_valid),
        )


@needs_numba
def test_glcm_kernel_matches_numpy():
    from rsgispy.imagefilter._glcm_helpers import (
        GLCM_ALL_PROPERTIES,
        _compute_glcm_offset_tile_numpy,
        compute_glcm_offset_tile,
    )

    rng = np.random.default_rng(11)
    exact_props = {"asm", "energy", "max_prob", "contrast", "dissimilarity", "mean"}
    for trial in range(16):
        k = int(rng.choice([3, 5, 7]))
        h, w = (int(v) for v in rng.integers(2, 25, 2))
        levels = int(rng.integers(2, 33))
        q_halo = rng.integers(0, levels, (h + k - 1, w + k - 1)).astype(np.int32)
        v_halo = rng.random(q_halo.shape) > (0.2 if trial % 2 else 0.0)
        q_wins = sliding_window_view(q_halo, (k, k))
        v_wins = sliding_window_view(v_halo, (k, k))
        centre_valid = v_halo[k // 2: k // 2 + h, k // 2: k // 2 + w]
        props = list(GLCM_ALL_PROPERTIES) if trial % 3 else ["entropy", "asm", "contrast"]
        for d_r, d_c in [(0, 1), (1, 1), (1, 0), (1, -1), (-2, 1), (0, k)]:
            for sym in (True, False):
                res_jit = compute_glcm_offset_tile(
                    q_wins, v_wins, centre_valid, d_r, d_c, levels, sym, props
                )
                res_py = _compute_glcm_offset_tile_numpy(
                    q_wins, v_wins, centre_valid, d_r, d_c, levels, sym, props
                )
                assert set(res_jit) == set(res_py)
                for p_name in props:
                    if p_name in exact_props:
                        np.testing.assert_array_equal(res_jit[p_name], res_py[p_name])
                    else:
                        np.testing.assert_allclose(
                            res_jit[p_name], res_py[p_name],
                            rtol=1e-12, atol=1e-12, equal_nan=True,
                        )


def test_d8_lookup_tables_match_dicts():
    from rsgispy.elevation._hydro_common import (
        _D8_CODE_TO_ANGLE,
        _D8_CODE_TO_IDX,
        _D8_DIR_TO_OFFSET,
    )
    from rsgispy.elevation._hydro_kernels import (
        D8_LUT_ANGLE,
        D8_LUT_DC,
        D8_LUT_DR,
        D8_LUT_IDX,
        D8_LUT_OK,
    )

    assert set(np.flatnonzero(D8_LUT_OK).tolist()) == set(_D8_DIR_TO_OFFSET)
    for code, (d_r, d_c) in _D8_DIR_TO_OFFSET.items():
        assert (D8_LUT_DR[code], D8_LUT_DC[code]) == (d_r, d_c)
        assert D8_LUT_IDX[code] == _D8_CODE_TO_IDX[code]
        assert D8_LUT_ANGLE[code] == _D8_CODE_TO_ANGLE[code]


@needs_numba
def test_d8_trace_kernels_match_python():
    from rsgispy.elevation._hydro_kernels import (
        D8_CODES,
        D8_LUT_DC,
        D8_LUT_DR,
        D8_LUT_OK,
        trace_d8_path_kernel,
        trace_stream_reach_kernel,
    )

    rng = np.random.default_rng(12)
    for _ in range(10):
        h, w = (int(v) for v in rng.integers(5, 40, 2))
        # random codes produce cycles and out-of-bounds exits
        fdir = rng.choice(np.append(D8_CODES, [0, 255]), (h, w)).astype(np.uint8)
        valid = rng.random((h, w)) > 0.05
        acc = rng.random((h, w)) * 100
        acc_valid = rng.random((h, w)) > 0.05
        streams = (rng.random((h, w)) > 0.3).astype(np.uint8)
        links = rng.integers(1, 4, (h, w)).astype(np.int64)
        visited = np.zeros((h, w), dtype=np.uint8)
        for _ in range(15):
            r0, c0 = int(rng.integers(h)), int(rng.integers(w))
            for upstream in (False, True):
                args = (fdir, valid, acc, acc_valid, r0, c0, upstream, visited,
                        D8_LUT_OK, D8_LUT_DR, D8_LUT_DC)
                jit_r, jit_c = trace_d8_path_kernel(*args)
                assert not visited.any()
                py_r, py_c = trace_d8_path_kernel.py_func(*args)
                assert not visited.any()
                np.testing.assert_array_equal(jit_r, py_r)
                np.testing.assert_array_equal(jit_c, py_c)
            s_args = (fdir, valid, streams, links, r0, c0, D8_LUT_OK, D8_LUT_DR, D8_LUT_DC)
            res_jit = trace_stream_reach_kernel(*s_args)
            res_py = trace_stream_reach_kernel.py_func(*s_args)
            np.testing.assert_array_equal(res_jit[0], res_py[0])
            np.testing.assert_array_equal(res_jit[1], res_py[1])
            assert tuple(int(v) for v in res_jit[2:]) == tuple(int(v) for v in res_py[2:])
