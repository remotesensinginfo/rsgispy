"""
Regression tests for hydrology bugs found by the hydrology tutorial:

* D-Infinity flow accumulation losing flow through float32 facet-boundary
  angles (uphill edges forming cycles in the flow graph).

The tests on the tutorial DEM are slow and need the tutorial data, so they
only run when the ``RSGISPY_TUTORIAL_DEM`` environment variable gives the
path to ``dem_10m_osgb.tif`` (the sea stored as 0 m).
"""

# pylint: disable=redefined-outer-name

import math
import os
import warnings

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

import rsgispy
from rsgispy._numba import disable_numba
from rsgispy.elevation import hydrology
from rsgispy.elevation._hydro_kernels import _dinf_targets
from rsgispy.imagecalc.maths import BandDefn, band_maths

requires_numba = pytest.mark.skipif(
    not hydrology.HAS_NUMBA, reason="numba not installed or disabled"
)

TUTORIAL_DEM = os.environ.get("RSGISPY_TUTORIAL_DEM", "")
requires_tutorial_dem = pytest.mark.skipif(
    not (TUTORIAL_DEM and os.path.isfile(TUTORIAL_DEM)),
    reason="set RSGISPY_TUTORIAL_DEM to the tutorial dem_10m_osgb.tif",
)

ND = -9999.0


def _write(path, arr, res=10.0, nodata=ND, dtype="float32"):
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[np.newaxis]
    with rasterio.open(
        str(path),
        "w",
        driver="GTiff",
        height=arr.shape[1],
        width=arr.shape[2],
        count=arr.shape[0],
        dtype=dtype,
        crs="EPSG:32630",
        transform=from_origin(500000.0, 5200000.0, res, res),
        nodata=nodata,
    ) as dst:
        dst.write(arr.astype(dtype))


def _read(path):
    with rasterio.open(str(path)) as src:
        return src.read(1)


def _synthetic_dems():
    n = 61
    r, c = np.indices((n, n), dtype=np.float64)
    rng = np.random.default_rng(42)
    return {
        "plane_45": 1000.0 - r - c,
        "cone": 100.0 - np.hypot(r - 30, c - 30),
        "pyramid": 100.0 - np.maximum(np.abs(r - 30), np.abs(c - 30)),
        "diamond": 100.0 - np.abs(r - 30) - np.abs(c - 30),
        "ridges": 100.0 - np.minimum(np.abs(r - c), np.abs(r + c - 60)) - 0.1 * r,
        "random_int": np.asarray(rng.integers(0, 10, (n, n)), dtype=np.float64),
    }


def _dinf_terminal_total(tmp_path, name, dem, res=10.0, pure_python=False):
    """Fill, D-Inf route and accumulate ``dem``; return (outlet total, n valid, acc)."""
    dem_p = tmp_path / f"{name}_dem.tif"
    fill_p = tmp_path / f"{name}_fill.tif"
    fdir_p = tmp_path / f"{name}_dinf.tif"
    acc_p = tmp_path / f"{name}_acc.tif"
    _write(dem_p, dem, res=res)
    kw = {"allow_pure_python": True} if pure_python else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        hydrology.fill_depressions(str(dem_p), str(fill_p), use_pbar=False, **kw)
        hydrology.flow_direction(
            str(fill_p), str(fdir_p), method="dinf", use_pbar=False, **kw
        )
    with warnings.catch_warnings():
        # A cycle warning is a failure here, but the NumPy fallback warns too.
        warnings.filterwarnings("error", message=".*cycle.*")
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        hydrology.flow_accumulation(str(fdir_p), str(acc_p), use_pbar=False, **kw)
    ang = _read(fdir_p)
    acc = _read(acc_p)
    # Every cell drains to a terminal cell (no downslope neighbour, -1).
    terminal = ang == -1.0
    return float(acc[terminal].astype(np.float64).sum()), int(np.sum(ang != ND)), acc


@pytest.mark.parametrize(
    "ang",
    [
        0.0,
        0.25 * math.pi,
        0.5 * math.pi,
        0.75 * math.pi,
        math.pi,
        1.25 * math.pi,
        1.5 * math.pi,
        1.75 * math.pi,
        2.0 * math.pi,
    ],
)
def test_dinf_targets_snaps_float32_facet_boundaries(ang):
    """Boundary angles stored as float32 send all the flow to one neighbour."""
    f32 = float(np.float32(ang))
    for a in (f32, float(np.nextafter(np.float32(ang), np.float32(10.0))),
              float(np.nextafter(np.float32(ang), np.float32(-10.0)))):
        if a < 0.0:
            continue
        dr1, dc1, _, _, w1, w2 = _dinf_targets(a, 10.0, 10.0)
        assert w1 == 1.0 and w2 == 0.0
        k = int(round((ang % (2.0 * math.pi)) / (0.25 * math.pi))) % 8
        exp = [(0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1)][k]
        assert (dr1, dc1) == exp


def test_dinf_targets_splits_interior_angles_and_non_square_diagonal():
    """Interior angles still split; non-square diagonals snap to the diagonal."""
    dr1, dc1, dr2, dc2, w1, w2 = _dinf_targets(0.2, 10.0, 10.0)
    assert (dr1, dc1, dr2, dc2) == (0, 1, -1, 1)
    assert w2 == pytest.approx(0.2 / (0.25 * math.pi))
    assert w1 + w2 == pytest.approx(1.0)
    phi = float(np.float32(math.atan2(20.0, 10.0)))
    dr1, dc1, _, _, w1, w2 = _dinf_targets(phi, 10.0, 20.0)
    assert (dr1, dc1, w1, w2) == (-1, 1, 1.0, 0.0)


@requires_numba
@pytest.mark.parametrize("name", list(_synthetic_dems()))
def test_dinf_accumulation_mass_balance(tmp_path, name):
    """The D-Inf outlet total equals the number of valid cells."""
    total, n_valid, _ = _dinf_terminal_total(tmp_path, name, _synthetic_dems()[name])
    assert total == pytest.approx(n_valid, rel=1e-6)


@requires_numba
def test_dinf_accumulation_mass_balance_non_square_pixels(tmp_path):
    """Flats routed with D8 angles keep their flow with non-square pixels."""
    dem = _synthetic_dems()["random_int"]
    dem_p = tmp_path / "ns_dem.tif"
    with rasterio.open(
        str(dem_p), "w", driver="GTiff", height=dem.shape[0], width=dem.shape[1],
        count=1, dtype="float32", crs="EPSG:32630", nodata=ND,
        transform=rasterio.transform.Affine(10.0, 0.0, 500000.0, 0.0, -20.0, 5200000.0),
    ) as dst:
        dst.write(dem.astype(np.float32), 1)
    fill_p = tmp_path / "ns_fill.tif"
    fdir_p = tmp_path / "ns_dinf.tif"
    acc_p = tmp_path / "ns_acc.tif"
    hydrology.fill_depressions(str(dem_p), str(fill_p), use_pbar=False)
    hydrology.flow_direction(str(fill_p), str(fdir_p), method="dinf", use_pbar=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        hydrology.flow_accumulation(str(fdir_p), str(acc_p), use_pbar=False)
    ang = _read(fdir_p)
    acc = _read(acc_p)
    assert float(acc[ang == -1.0].sum()) == pytest.approx(dem.size, rel=1e-6)


@requires_numba
def test_dinf_accumulation_numba_numpy_agree(tmp_path, monkeypatch):
    """The numba and pure-Python / NumPy D-Inf paths give the same result."""
    dem = _synthetic_dems()["random_int"]
    total_nb, n_valid, acc_nb = _dinf_terminal_total(tmp_path, "nb", dem)
    monkeypatch.setattr(hydrology, "HAS_NUMBA", False)
    with disable_numba():
        total_py, _, acc_py = _dinf_terminal_total(tmp_path, "py", dem, pure_python=True)
    assert total_py == pytest.approx(n_valid, rel=1e-6)
    assert total_nb == pytest.approx(total_py, rel=1e-6)
    np.testing.assert_allclose(acc_nb, acc_py, rtol=1e-5, atol=1e-5)


@requires_numba
@pytest.mark.parametrize("mode", ["d8", "dinf", "mfd"])
def test_flow_accumulation_warns_on_cycle(tmp_path, mode):
    """A cycle in the flow directions is reported with the number of cells."""
    # Row of 4 cells: 0 -> 1 <-> 2 (cycle), 3 drains into 2.
    if mode == "d8":
        fdir = np.array([[1, 1, 16, 16]], dtype=np.uint8)
        _write(tmp_path / "f.tif", fdir, nodata=255, dtype="uint8")
    elif mode == "dinf":
        fdir = np.array([[0.0, 0.0, math.pi, math.pi]], dtype=np.float32)
        _write(tmp_path / "f.tif", fdir)
    else:
        fdir = np.zeros((8, 1, 4), dtype=np.float32)
        fdir[0, 0, :2] = 1.0  # East
        fdir[4, 0, 2:] = 1.0  # West
        _write(tmp_path / "f.tif", fdir)
    with pytest.warns(RuntimeWarning, match=r"cycle, so 2 cells"):
        hydrology.flow_accumulation(
            str(tmp_path / "f.tif"), str(tmp_path / "a.tif"), use_pbar=False
        )


@pytest.fixture(scope="module")
def tutorial_land_dem(tmp_path_factory):
    """The tutorial DEM with the sea (stored as 0 m) set to no data."""
    out_dir = tmp_path_factory.mktemp("tutorial_dem")
    dem_img = str(out_dir / "dem_land.tif")
    band_maths(
        dem_img,
        "dem <= 0 ? -32768 : dem",
        [BandDefn("dem", TUTORIAL_DEM, 1)],
        datatype=rsgispy.TYPE_32FLOAT,
        no_data_val=-32768,
        use_pbar=False,
    )
    return out_dir, dem_img


@requires_numba
@requires_tutorial_dem
def test_tutorial_dem_dinf_outlet_total_matches_d8_mfd(tutorial_land_dem):
    """On the tutorial DEM, D-Inf, D8 and MFD outlet totals agree to 0.1%."""
    out_dir, dem_img = tutorial_land_dem
    cond = str(out_dir / "cond.tif")
    hydrology.condition_dem(
        dem_img, cond, method="breach_and_fill", max_breach_depth=5.0,
        max_breach_length=30, use_pbar=False,
    )
    n_valid = int(np.sum(_read(dem_img) != -32768))
    totals = {}
    for mode in ("d8", "dinf", "mfd"):
        fdir = str(out_dir / f"fdir_{mode}.tif")
        acc = str(out_dir / f"acc_{mode}.tif")
        hydrology.flow_direction(cond, fdir, method=mode, use_pbar=False)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            hydrology.flow_accumulation(fdir, acc, use_pbar=False)
        f_arr = _read(fdir)
        a_arr = _read(acc).astype(np.float64)
        if mode == "d8":
            terminal = f_arr == 0
        elif mode == "dinf":
            terminal = f_arr == -1.0
        else:
            with rasterio.open(fdir) as src:
                terminal = (src.read().sum(axis=0) == 0.0) & (f_arr != ND)
        totals[mode] = a_arr[terminal].sum()
    for mode, total in totals.items():
        assert total == pytest.approx(n_valid, rel=1e-3), mode
