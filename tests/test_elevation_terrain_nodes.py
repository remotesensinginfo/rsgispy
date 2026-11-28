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
from rsgispy.imagecalc.maths import BandDefn, band_maths
from rsgispy.imagefilter.stats import apply_mean_filter, mean_filter_node
from rsgispy.imageprocess import (
    ArraySource,
    BandMathOp,
    DiskSink,
    MemorySink,
    RasterSource,
    StackBands,
    TileGraph,
)

from conftest import write_raster

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
    return write_raster(
        path,
        arr,
        crs=crs,
        transform=transform if isinstance(transform, Affine) else Affine(*transform),
        nodata=nodata,
    )


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
    """
    Integers exactly; float32 within 1 ulp and float64 within a relative
    1e-12, with the same NaN positions. The trigonometric functions (e.g.
    ``np.arccos``) come from the platform's maths library and can differ by
    a few ulp between platforms; rounding to float32 hides that, but a
    float64 output keeps it.
    """
    assert arr.dtype == expected.dtype
    assert arr.shape == expected.shape
    if np.issubdtype(expected.dtype, np.floating):
        nan_a = np.isnan(arr)
        np.testing.assert_array_equal(nan_a, np.isnan(expected))
        if expected.dtype == np.float64:
            np.testing.assert_allclose(
                arr[~nan_a], expected[~nan_a], rtol=1e-12, atol=1e-12
            )
        else:
            np.testing.assert_array_max_ulp(
                arr[~nan_a], expected[~nan_a], maxulp=1
            )
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


def test_file_functions_cog_output(tmp_path):
    """COG outputs hold the golden values and statistics."""
    dem = _input_path(tmp_path, "dem_a")
    out = str(tmp_path / "hs_cog.tif")
    elev_illum.hillshade(dem, out, 315.0, 45.0, gdalformat="COG", use_pbar=False)
    with rasterio.open(out) as ds:
        assert ds.tags(ns="IMAGE_STRUCTURE").get("LAYOUT") == "COG"
        assert ds.nodata == 0.0
        assert "STATISTICS_MEAN" in ds.tags(1)
        np.testing.assert_array_equal(ds.read(1), _GOLDEN["out_hillshade_dem_a"])
    out = str(tmp_path / "cat_cog.tif")
    elev_terr.categorise_aspect(
        _input_path(tmp_path, "asp_e"), out, gdalformat="COG", use_pbar=False
    )
    with rasterio.open(out) as ds:
        assert ds.nodata is None
        np.testing.assert_array_equal(ds.read(1), _GOLDEN["out_cat_asp_e"])


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


# --------------------------------------------------------------------------
# Tile-graph node builders
# --------------------------------------------------------------------------

# (builder, golden case with the default data type)
_NODE_CASES = {
    "slope_deg": (lambda s: elev_terr.slope_node(s, "degrees"), "slope_deg_{dem}"),
    "slope_rad": (lambda s: elev_terr.slope_node(s, "radians"), "slope_rad_{dem}"),
    "aspect": (elev_terr.aspect_node, "aspect_{dem}"),
    "hillshade": (lambda s: elev_illum.hillshade_node(s, 315.0, 45.0), "hillshade_{dem}"),
    "lia": (lambda s: elev_illum.local_incidence_angle_node(s, 135.0, 35.0), "lia_{dem}"),
    "lea": (lambda s: elev_illum.local_exitance_angle_node(s, 200.0, 15.0), "lea_{dem}"),
}

# case id: (builder(dem, res), dem input, resolution input)
_PXL_NODE_CASES = {
    "slope_pxl_deg_dem_d": (
        lambda s, r: elev_terr.slope_pxl_res_img_node(s, r, "degrees"), "dem_d", "res_d"),
    "slope_pxl_rad_dem_a": (
        lambda s, r: elev_terr.slope_pxl_res_img_node(s, r, unit="radians"), "dem_a", "res_a"),
    "aspect_pxl_dem_d": (elev_terr.aspect_pxl_res_img_node, "dem_d", "res_d"),
    "aspect_pxl_dem_a": (elev_terr.aspect_pxl_res_img_node, "dem_a", "res_a"),
    "hillshade_pxl_dem_d": (
        lambda s, r: elev_illum.hillshade_pxl_res_img_node(s, r, 315.0, 45.0), "dem_d", "res_d"),
}

_DTYPES = {"float32": rsgispy.TYPE_32FLOAT, "uint8": rsgispy.TYPE_8UINT}


def _graph_to_file(node, out_path, case, block_size):
    TileGraph(
        DiskSink(node, out_path, datatype=_DTYPES[case["dtype"]]), block_size=block_size
    ).run(use_pbar=False)
    with rasterio.open(out_path) as ds:
        return ds.read(1), ds.nodata


@pytest.mark.parametrize("block_size", [7, 32, 256])
@pytest.mark.parametrize("dem", ["dem_a", "dem_b", "dem_c"])
@pytest.mark.parametrize("kind", sorted(_NODE_CASES))
def test_node_builders_equal_golden(tmp_path, kind, dem, block_size):
    """RasterSource -> builder -> DiskSink equals the file function's output."""
    builder, case_fmt = _NODE_CASES[kind]
    case = _CASES[case_fmt.format(dem=dem)]
    node = builder(RasterSource(_input_path(tmp_path, dem), bands=1))
    arr, nodata = _graph_to_file(node, str(tmp_path / "out.tif"), case, block_size)
    assert_matches_golden(arr, _GOLDEN[f"out_{case['id']}"])
    assert nodata == (None if case["nodata"] is None else float(case["nodata"]))


@pytest.mark.parametrize("block_size", [7, 256])
@pytest.mark.parametrize("case_id", sorted(_PXL_NODE_CASES))
def test_pxl_res_node_builders_equal_golden(tmp_path, case_id, block_size):
    """The *_pxl_res_img nodes (resolution as a second input) equal the golden outputs."""
    builder, dem, res = _PXL_NODE_CASES[case_id]
    case = _CASES[case_id]
    node = builder(
        RasterSource(_input_path(tmp_path, dem), bands=1),
        RasterSource(_input_path(tmp_path, res)),
    )
    arr, nodata = _graph_to_file(node, str(tmp_path / "out.tif"), case, block_size)
    assert_matches_golden(arr, _GOLDEN[f"out_{case_id}"])
    assert nodata == (None if case["nodata"] is None else float(case["nodata"]))


@pytest.mark.parametrize("block_size", [5, 256])
def test_categorise_aspect_node_equals_golden(tmp_path, block_size):
    """categorise_aspect_node equals categorise_aspect, also fed by aspect_node."""
    for case_id in ("cat_asp_e", "cat_aspect_dem_a", "cat_aspect_dem_b"):
        case = _CASES[case_id]
        src = RasterSource(_input_path(tmp_path, case["inputs"][0]))
        node = elev_terr.categorise_aspect_node(src)
        arr, nodata = _graph_to_file(
            node, str(tmp_path / f"{case_id}.tif"), case, block_size
        )
        assert_matches_golden(arr, _GOLDEN[f"out_{case_id}"])
        assert nodata is None
    asp = elev_terr.aspect_node(RasterSource(_input_path(tmp_path, "dem_a"), bands=1))
    out = TileGraph(
        MemorySink(elev_terr.categorise_aspect_node(asp), name="c"),
        block_size=block_size,
    ).run(use_pbar=False)["c"]
    np.testing.assert_array_equal(out[0], _GOLDEN["out_cat_aspect_dem_a"])


def test_node_metadata_and_describe(tmp_path):
    """Node metadata, halo radii and TileGraph.describe()."""
    dem = RasterSource(_input_path(tmp_path, "dem_a"), bands=1)
    nodes = {
        "slope": elev_terr.slope_node(dem),
        "hs": elev_illum.hillshade_node(dem, 315.0, 45.0),
        "cat": elev_terr.categorise_aspect_node(elev_terr.aspect_node(dem)),
        "lia": elev_illum.local_incidence_angle_node(dem, 10.0, 20.0, no_data_val=-1),
    }
    graph = TileGraph([MemorySink(n, name=k) for k, n in nodes.items()])
    meta = graph.node_meta
    assert meta[nodes["slope"]].n_bands == 1
    assert meta[nodes["slope"]].dtype == np.float64
    assert meta[nodes["slope"]].nodata == -9999.0
    assert meta[nodes["hs"]].dtype == np.uint8
    assert meta[nodes["hs"]].nodata == 0.0
    assert meta[nodes["cat"]].dtype == np.uint8
    assert meta[nodes["cat"]].nodata is None
    assert meta[nodes["lia"]].nodata == -1.0
    assert nodes["slope"].halo_radius == (1, 1)
    assert nodes["cat"].halo_radius == (0, 0)
    text = graph.describe()
    assert "slope unit=degrees" in text
    assert "hillshade azimuth=315.0 zenith=45.0" in text
    assert graph.required_halos[dem] == (1, 1)
    out = graph.run(use_pbar=False)
    assert out["hs"].dtype == np.uint8
    np.testing.assert_array_equal(out["hs"][0], _GOLDEN["out_hillshade_dem_a"])


def test_array_source_with_transform():
    """The pixel size comes from the graph grid (an ArraySource transform)."""
    meta = _MANIFEST["inputs"]["dem_b"]
    src = ArraySource(_GOLDEN["in_dem_b"], transform=Affine(*meta["transform"]))
    out = TileGraph(
        MemorySink(elev_terr.slope_node(src), dtype=np.float32, name="s"), block_size=16
    ).run(use_pbar=False)["s"]
    assert_matches_golden(out[0], _GOLDEN["out_slope_deg_dem_b"])


def test_node_builder_errors(tmp_path):
    """Builders validate like the file functions; graphs check the inputs."""
    dem_path = _input_path(tmp_path, "dem_a")
    dem = RasterSource(dem_path, bands=1)
    with pytest.raises(rsgispy.RSGISPyException, match="'degrees' or 'radians'"):
        elev_terr.slope_node(dem, "percent")
    with pytest.raises(rsgispy.RSGISPyException, match="'degrees' or 'radians'"):
        elev_terr.slope_pxl_res_img_node(dem, dem, "percent")
    with pytest.raises(rsgispy.RSGISPyException, match="zenith angle"):
        elev_illum.hillshade_node(dem, 315.0, 91.0)
    with pytest.raises(rsgispy.RSGISPyException, match="azimuth angle"):
        elev_illum.hillshade_pxl_res_img_node(dem, dem, 361.0, 45.0)
    with pytest.raises(rsgispy.RSGISPyException, match="zenith angle"):
        elev_illum.local_incidence_angle_node(dem, 0.0, -1.0)
    with pytest.raises(rsgispy.RSGISPyException, match="azimuth angle"):
        elev_illum.local_exitance_angle_node(dem, -5.0, 10.0)
    with pytest.raises(rsgispy.RSGISPyException, match="TileNode"):
        elev_terr.slope_node(dem_path)
    with pytest.raises(rsgispy.RSGISPyException, match="TileNode"):
        elev_terr.categorise_aspect_node(dem_path)
    # Multi-band DEM / single-band resolution input, at graph construction.
    res_path = _input_path(tmp_path, "res_a")
    with pytest.raises(rsgispy.RSGISPyException, match="must have 1 band"):
        TileGraph(MemorySink(elev_terr.aspect_node(RasterSource(res_path))))
    with pytest.raises(rsgispy.RSGISPyException, match="must have 1 band"):
        TileGraph(MemorySink(elev_terr.categorise_aspect_node(RasterSource(res_path))))
    with pytest.raises(rsgispy.RSGISPyException, match="at least 2 bands"):
        TileGraph(MemorySink(elev_terr.slope_pxl_res_img_node(dem, dem)))
    # The resolution node must be on the DEM's grid.
    res_other = _write(
        tmp_path / "res_other.tif", _GOLDEN["in_res_a"], "EPSG:4326",
        _MANIFEST["inputs"]["res_a"]["transform"], None,
    )
    with pytest.raises(rsgispy.RSGISPyException, match="same CRS"):
        TileGraph(MemorySink(elev_terr.slope_pxl_res_img_node(dem, RasterSource(res_other))))
    # No transform: the pixel size is unknown.
    no_tr = ArraySource(np.ones((5, 5)))
    with pytest.raises(rsgispy.RSGISPyException, match="no transform"):
        TileGraph(MemorySink(elev_terr.slope_node(no_tr))).run(use_pbar=False)
    zero_tr = ArraySource(np.ones((5, 5)), transform=Affine(0.0, 0.0, 0.0, 0.0, -1.0, 0.0))
    with pytest.raises(rsgispy.RSGISPyException, match="invalid pixel resolution"):
        TileGraph(MemorySink(elev_illum.hillshade_node(zero_tr, 0.0, 0.0))).run(
            use_pbar=False
        )
    neg = ArraySource(np.full((4, 4), -3.0))
    with pytest.raises(rsgispy.RSGISPyException, match="between 0 and 360"):
        TileGraph(MemorySink(elev_terr.categorise_aspect_node(neg))).run(use_pbar=False)


@pytest.mark.parametrize("block_size", [5, 13, 256])
def test_chain_smoothing_slope_classes_equals_file_sequence(tmp_path, block_size):
    """
    mean filter -> slope -> classes in one graph equals apply_mean_filter,
    slope and band_maths run one after another (float64 intermediates).
    """
    dem_path = _input_path(tmp_path, "dem_a")
    expr = "(s >= 5) + (s >= 15) + (s >= 30)"

    smooth_img = str(tmp_path / "smooth.tif")
    slope_img = str(tmp_path / "slope.tif")
    cls_img = str(tmp_path / "classes.tif")
    apply_mean_filter(dem_path, smooth_img, 5, datatype=rsgispy.TYPE_64FLOAT,
                      calc_stats=False, use_pbar=False)
    elev_terr.slope(smooth_img, slope_img, "degrees", datatype=rsgispy.TYPE_64FLOAT,
                    calc_stats=False, use_pbar=False)
    band_maths(cls_img, expr, [BandDefn("s", slope_img, 1)], datatype=rsgispy.TYPE_8UINT,
               use_pbar=False)

    dem = RasterSource(dem_path, bands=1)
    slp = elev_terr.slope_node(mean_filter_node(dem, 5, no_data_val=-9999))
    classes = BandMathOp({"s": slp}, expr)
    graph_cls = str(tmp_path / "graph_classes.tif")
    out = TileGraph(
        [
            DiskSink(classes, graph_cls, datatype=rsgispy.TYPE_8UINT),
            MemorySink(slp, name="slope"),
        ],
        block_size=block_size,
    ).run(use_pbar=False)

    with rasterio.open(slope_img) as ds:
        np.testing.assert_array_equal(out["slope"][0], ds.read(1))
    with rasterio.open(cls_img) as exp_ds, rasterio.open(graph_cls) as got_ds:
        got = got_ds.read(1)
        np.testing.assert_array_equal(got, exp_ds.read(1))
        assert got.max() == 3 and got.min() == 0


@pytest.mark.parametrize("block_size", [6, 256])
def test_terrain_node_followed_by_window_equals_file_sequence(tmp_path, block_size):
    """
    Terrain nodes feeding a moving window (halo through the terrain node,
    image edges re-padded) equal the file functions run one after another.
    """
    dem_path = _input_path(tmp_path, "dem_a")
    dem = RasterSource(dem_path, bands=1)

    def _slope_file(out_img):
        elev_terr.slope(dem_path, out_img, datatype=rsgispy.TYPE_64FLOAT,
                        calc_stats=False, use_pbar=False)

    def _lia_file(out_img):
        elev_illum.local_incidence_angle(
            dem_path, out_img, 100.0, 40.0, datatype=rsgispy.TYPE_64FLOAT,
            calc_stats=False, use_pbar=False,
        )

    steps = {
        "slope": (elev_terr.slope_node(dem), _slope_file),
        "lia": (elev_illum.local_incidence_angle_node(dem, 100.0, 40.0), _lia_file),
    }
    for key, (node, file_fn) in steps.items():
        step_img = str(tmp_path / f"{key}.tif")
        file_fn(step_img)
        filt_img = str(tmp_path / f"{key}_mean.tif")
        apply_mean_filter(step_img, filt_img, 7, datatype=rsgispy.TYPE_64FLOAT,
                          no_data_val=-9999, calc_stats=False, use_pbar=False)
        out = TileGraph(
            MemorySink(mean_filter_node(node, 7, no_data_val=-9999), name="m"),
            block_size=block_size,
        ).run(use_pbar=False)["m"]
        with rasterio.open(filt_img) as ds:
            np.testing.assert_array_equal(out[0], ds.read(1))

    # Several terrain nodes sharing one DEM source in one graph.
    stack = StackBands([
        elev_terr.slope_node(dem),
        elev_terr.aspect_node(dem),
        elev_illum.hillshade_node(dem, 315.0, 45.0),
    ])
    out = TileGraph(MemorySink(stack, name="s"), block_size=block_size).run(
        use_pbar=False
    )["s"]
    assert_matches_golden(out[0].astype(np.float32), _GOLDEN["out_slope_deg_dem_a"])
    assert_matches_golden(out[1].astype(np.float32), _GOLDEN["out_aspect_dem_a"])
    np.testing.assert_array_equal(out[2], _GOLDEN["out_hillshade_dem_a"])


_GEOGRAPHIC_FILE_CALLS = {
    "slope": lambda d, o: elev_terr.slope(d, o, use_pbar=False),
    "aspect": lambda d, o: elev_terr.aspect(d, o, use_pbar=False),
    "hillshade": lambda d, o: elev_illum.hillshade(d, o, 315.0, 45.0, use_pbar=False),
    "local_incidence_angle": lambda d, o: elev_illum.local_incidence_angle(
        d, o, 135.0, 35.0, use_pbar=False),
    "local_exitance_angle": lambda d, o: elev_illum.local_exitance_angle(
        d, o, 200.0, 15.0, use_pbar=False),
    "shadow_mask": lambda d, o: elev_illum.shadow_mask(
        d, o, 315.0, 45.0, 3000.0, use_pbar=False),
}


@pytest.mark.parametrize("func", sorted(_GEOGRAPHIC_FILE_CALLS))
def test_terrain_rejects_geographic_crs(tmp_path, func):
    """H12: a DEM in a geographic CRS raises, pointing to *_pxl_res_img."""
    dem = _input_path(tmp_path, "dem_d")  # EPSG:4326
    with pytest.raises(rsgispy.RSGISPyException, match="pxl_res_img"):
        _GEOGRAPHIC_FILE_CALLS[func](dem, str(tmp_path / "out.tif"))


@pytest.mark.parametrize("kind", sorted(_NODE_CASES))
def test_terrain_nodes_reject_geographic_crs(tmp_path, kind):
    """H12: the node builders without a pixel-resolution input raise too."""
    builder, _ = _NODE_CASES[kind]
    node = builder(RasterSource(_input_path(tmp_path, "dem_d"), bands=1))
    with pytest.raises(rsgispy.RSGISPyException, match="geographic"):
        TileGraph(MemorySink(node, name="out")).run(use_pbar=False)
