"""
Tests for the terrain and illumination functions of rsgispy.elevation
(slope, aspect, categorise_aspect, hillshade, local incidence and exitance
angles and the ``*_pxl_res_img`` variants): golden outputs generated with the
block-loop implementation (``tests/data/golden/terrain_golden.npz``, see
``tests/data/golden/make_terrain_golden.py``), error handling, and the tile
graph node builders.
"""

import json
import math
import os

import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine

import rsgispy
import rsgispy.elevation.illumination as elev_illum
import rsgispy.elevation.terrain as elev_terr

GOLDEN_NPZ = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "golden", "terrain_golden.npz"
)

FUNCS = {
    "slope": elev_terr.slope,
    "slope_pxl_res_img": elev_terr.slope_pxl_res_img,
    "aspect": elev_terr.aspect,
    "aspect_pxl_res_img": elev_terr.aspect_pxl_res_img,
    "categorise_aspect": elev_terr.categorise_aspect,
    "hillshade": elev_illum.hillshade,
    "hillshade_pxl_res_img": elev_illum.hillshade_pxl_res_img,
    "local_incidence_angle": elev_illum.local_incidence_angle,
    "local_exitance_angle": elev_illum.local_exitance_angle,
}


def _load_golden():
    with np.load(GOLDEN_NPZ) as data:
        arrays = {k: data[k] for k in data.files}
    manifest = json.loads(str(arrays.pop("manifest")))
    return arrays, manifest


_GOLDEN, _MANIFEST = _load_golden()
_CASES = {case["id"]: case for case in _MANIFEST["cases"]}


def _write(path, arr, crs, transform, nodata):
    arr3 = arr[np.newaxis] if arr.ndim == 2 else arr
    profile = {
        "driver": "GTiff",
        "height": arr3.shape[1],
        "width": arr3.shape[2],
        "count": arr3.shape[0],
        "dtype": str(arr3.dtype),
        "crs": crs,
        "transform": Affine(*transform) if not isinstance(transform, Affine) else transform,
        "nodata": nodata,
    }
    with rasterio.open(str(path), "w", **profile) as dst:
        dst.write(arr3)
    return str(path)


def _input_path(tmp_path, name):
    """Write golden input ``name`` (or ``@case`` for a golden output) to a file."""
    path = tmp_path / f"{name.lstrip('@')}.tif"
    if path.exists():
        return str(path)
    if name.startswith("@"):
        case = _CASES[name[1:]]
        meta = _MANIFEST["inputs"][case["inputs"][0]]
        nodata = None if case["nodata"] is None else float(case["nodata"])
        return _write(
            path, _GOLDEN[f"out_{case['id']}"], meta["crs"], meta["transform"], nodata
        )
    meta = _MANIFEST["inputs"][name]
    return _write(path, _GOLDEN[f"in_{name}"], meta["crs"], meta["transform"], meta["nodata"])


def _run_case(case, in_paths, out_path, calc_stats=True):
    kwargs = dict(case["kwargs"])
    args = list(in_paths) + [out_path]
    for key in ("unit", "azimuth", "zenith"):
        if key in kwargs:
            args.append(kwargs.pop(key))
    FUNCS[case["func"]](
        *args, gdalformat="GTIFF", calc_stats=calc_stats, use_pbar=False, **kwargs
    )


def _stats_match(stats_a, stats_b, rel=1e-9):
    if set(stats_a) != set(stats_b):
        return False
    for key, val_a in stats_a.items():
        val_b = stats_b[key]
        if val_a == val_b:
            continue
        try:
            f_a, f_b = float(val_a), float(val_b)
        except ValueError:
            return False
        if abs(f_a - f_b) > rel * max(abs(f_a), abs(f_b), 1e-300):
            return False
    return True


def assert_matches_golden(arr, expected):
    """Integers exactly; floats within 1 ulp with the same NaN positions."""
    assert arr.dtype == expected.dtype
    assert arr.shape == expected.shape
    if np.issubdtype(expected.dtype, np.floating):
        nan_a = np.isnan(arr)
        np.testing.assert_array_equal(nan_a, np.isnan(expected))
        np.testing.assert_array_max_ulp(arr[~nan_a], expected[~nan_a], maxulp=1)
    else:
        np.testing.assert_array_equal(arr, expected)


@pytest.mark.parametrize("block_size", [5, 16, 256])
@pytest.mark.parametrize("case_id", sorted(_CASES))
def test_file_functions_match_golden(tmp_path, monkeypatch, case_id, block_size):
    """Each function reproduces its golden output (data, dtype, no data, stats)."""
    monkeypatch.setenv("RSGISPY_BLOCK_SIZE", str(block_size))
    case = _CASES[case_id]
    in_paths = [_input_path(tmp_path, name) for name in case["inputs"]]
    out_path = str(tmp_path / f"{case_id}.tif")
    _run_case(case, in_paths, out_path)
    with rasterio.open(out_path) as ds:
        arr = ds.read(1)
        assert ds.count == 1
        assert ds.dtypes[0] == case["dtype"]
        exp_nd = None if case["nodata"] is None else float(case["nodata"])
        assert ds.nodata == exp_nd
        assert ds.descriptions[0] == case["description"]
        tags = {k: v for k, v in ds.tags(1).items() if k.startswith("STATISTICS_")}
        assert _stats_match(tags, case["stats"]), (tags, case["stats"])
    assert_matches_golden(arr, _GOLDEN[f"out_{case_id}"])


def test_file_functions_without_stats(tmp_path):
    """calc_stats=False writes no statistics."""
    case = _CASES["slope_deg_dem_a"]
    out_path = str(tmp_path / "slope.tif")
    _run_case(case, [_input_path(tmp_path, "dem_a")], out_path, calc_stats=False)
    with rasterio.open(out_path) as ds:
        assert not any(k.startswith("STATISTICS_") for k in ds.tags(1))
        assert_matches_golden(ds.read(1), _GOLDEN["out_slope_deg_dem_a"])


def test_file_function_errors(tmp_path):
    """Invalid parameters and inputs raise RSGISPyException."""
    dem = _input_path(tmp_path, "dem_a")
    res = _input_path(tmp_path, "res_a")
    out = str(tmp_path / "out.tif")
    with pytest.raises(rsgispy.RSGISPyException, match="'degrees' or 'radians'"):
        elev_terr.slope(dem, out, "percent")
    with pytest.raises(rsgispy.RSGISPyException, match="'degrees' or 'radians'"):
        elev_terr.slope_pxl_res_img(dem, res, out, "percent")
    for func in (
        elev_illum.hillshade,
        elev_illum.local_incidence_angle,
        elev_illum.local_exitance_angle,
    ):
        with pytest.raises(rsgispy.RSGISPyException, match="zenith angle"):
            func(dem, out, 180.0, 95.0)
        with pytest.raises(rsgispy.RSGISPyException, match="azimuth angle"):
            func(dem, out, -1.0, 45.0)
    with pytest.raises(rsgispy.RSGISPyException, match="azimuth angle"):
        elev_illum.hillshade_pxl_res_img(dem, res, out, 400.0, 45.0)
    with pytest.raises(rsgispy.RSGISPyException, match="at least 2 bands"):
        elev_terr.slope_pxl_res_img(dem, dem, out)
    small = _write(
        tmp_path / "small.tif", np.ones((2, 5, 5), np.float32), "EPSG:32630",
        (10.0, 0.0, 500000.0, 0.0, -10.0, 5200000.0), None,
    )
    with pytest.raises(rsgispy.RSGISPyException, match="matching dimensions"):
        elev_terr.aspect_pxl_res_img(dem, small, out)
    with pytest.raises(rsgispy.RSGISPyException):
        elev_terr.slope(str(tmp_path / "missing.tif"), out)
    with pytest.raises(rsgispy.RSGISPyException):
        elev_terr.slope_pxl_res_img(dem, str(tmp_path / "missing.tif"), out)
    neg = _write(
        tmp_path / "neg.tif", np.full((4, 4), -5.0, np.float32), "EPSG:32630",
        (10.0, 0.0, 500000.0, 0.0, -10.0, 5200000.0), None,
    )
    with pytest.raises(rsgispy.RSGISPyException, match="between 0 and 360"):
        elev_terr.categorise_aspect(neg, out)


def test_pxl_res_img_on_another_grid(tmp_path):
    """
    The pixel-resolution image only needs the same dimensions as the DEM
    (its transform and CRS are not compared).
    """
    dem = _input_path(tmp_path, "dem_d")
    res_other = _write(
        tmp_path / "res_other.tif", _GOLDEN["in_res_d"], "EPSG:32630",
        (10.0, 0.0, 0.0, 0.0, -10.0, 0.0), -1.0,
    )
    out = str(tmp_path / "slope.tif")
    elev_terr.slope_pxl_res_img(dem, res_other, out, "degrees", use_pbar=False)
    with rasterio.open(out) as ds:
        assert_matches_golden(ds.read(1), _GOLDEN["out_slope_pxl_deg_dem_d"])
        assert ds.crs == rasterio.crs.CRS.from_string("EPSG:4326")


def test_sanity_of_golden_values():
    """A few golden values are what the terrain formulas give."""
    slope = _GOLDEN["out_slope_deg_dem_b"]
    assert np.nanmax(slope) < 90.0
    hs = _GOLDEN["out_hillshade_dem_a"]
    assert hs.dtype == np.uint8 and hs.max() <= 255
    cat = _GOLDEN["out_cat_asp_e"]
    assert list(cat[0, :12]) == [1, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 1]
    assert math.isnan(float(_GOLDEN["out_aspect_dem_a"][7, 35]))
