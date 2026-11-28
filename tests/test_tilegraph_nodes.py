"""
Tests for the tile-graph node protocol, node metadata and validation, grid
checks, shared reading, the structural nodes (SelectBands, StackBands, Cast),
TileGraph.describe(), the default block size, and the public filter node
builders (bit-exact against the file-based filters).
"""

import os

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

import rsgispy
import rsgispy.imagefilter.chained as if_chained
import rsgispy.imagefilter.edge as if_edge
import rsgispy.imagefilter.sar as if_sar
import rsgispy.imagefilter.smooth as if_smooth
import rsgispy.imagefilter.stats as if_stats
import rsgispy.imageprocess as ip
from rsgispy.imageprocess import (
    ArraySource,
    BandMathOp,
    Cast,
    DiskSink,
    MemorySink,
    NodeMeta,
    RasterSource,
    SelectBands,
    StackBands,
    TileGraph,
    TileNode,
    TileSink,
    WindowOp,
    same_bands_meta,
)


def _write(path, arr, nodata=None, crs="EPSG:32630", origin=(500000.0, 5000000.0),
           res=10.0, names=None):
    arr3 = arr[np.newaxis] if arr.ndim == 2 else arr
    profile = {
        "driver": "GTiff",
        "height": arr3.shape[1],
        "width": arr3.shape[2],
        "count": arr3.shape[0],
        "dtype": str(arr3.dtype),
        "crs": crs,
        "transform": from_origin(origin[0], origin[1], res, res),
        "nodata": nodata,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr3)
        if names:
            for idx, name in enumerate(names, start=1):
                dst.set_band_description(idx, name)
    return str(path)


def _read(path):
    with rasterio.open(path) as ds:
        return ds.read(), ds.nodata, ds.descriptions


def _box3(arr):
    """Whole-array 3x3 mean with edge padding."""
    pad = np.pad(arr, ((0, 0), (1, 1), (1, 1)), mode="edge")
    out = np.zeros_like(arr, dtype=np.float64)
    for dy in range(3):
        for dx in range(3):
            out += pad[:, dy:dy + arr.shape[1], dx:dx + arr.shape[2]]
    return out / 9.0


# --------------------------------------------------------------------------
# Node protocol
# --------------------------------------------------------------------------


class _Box3(TileNode):
    """Custom 3x3 mean node defined outside the package."""

    def __init__(self, source, name=None):
        super().__init__(inputs=(source,), name=name)

    @property
    def halo_radius(self):
        return (1, 1)

    def infer_meta(self, inputs):
        return inputs[0].replace(dtype=np.float64)

    def evaluate(self, inputs, halo_y, halo_x, ctx, state=None):
        tin = inputs[0].crop(halo_y + 1, halo_x + 1)
        arr = tin.data
        h = arr.shape[1] - 2
        w = arr.shape[2] - 2
        out = np.zeros((arr.shape[0], h, w))
        for dy in range(3):
            for dx in range(3):
                out += arr[:, dy:dy + h, dx:dx + w]
        # Output tensor: halo (halo_y, halo_x); out-of-image margins shrink
        # by the radius.
        return ip.TileTensor(
            data=out / 9.0,
            halo_y=halo_y,
            halo_x=halo_x,
            oob_top=max(0, tin.oob_top - 1),
            oob_bottom=max(0, tin.oob_bottom - 1),
            oob_left=max(0, tin.oob_left - 1),
            oob_right=max(0, tin.oob_right - 1),
        )


class _SumSink(TileSink):
    """Custom sink summing every pixel."""

    returns_result = True
    result_prefix = "sum"

    def open(self, grid, meta, *, stack, use_pbar=True):
        self._total = 0.0

    def write(self, block, window):
        self._total += float(np.sum(block))

    def close(self):
        self.result = self._total


def test_custom_node_and_sink_without_changing_tilegraph():
    rng = np.random.default_rng(1)
    arr = rng.uniform(0, 10, size=(2, 23, 31))
    src = ArraySource(arr, band_names=["a", "b"])
    box = _Box3(src, name="box")
    mem = MemorySink(box, name="box")
    total = _SumSink(src)
    graph = TileGraph([mem, total], block_size=7)
    assert graph.node_meta[box].band_names == ("a", "b")
    assert graph.required_halos[src] == (1, 1)
    res = graph.run(use_pbar=False)
    # The interior uses real neighbours; the edge rows use the edge-padded
    # input (the custom node does not re-pad, the source pads with 'edge').
    np.testing.assert_allclose(res["box"], _box3(arr), atol=1e-12)
    assert res["sum_0"] == pytest.approx(float(arr.sum()))


def test_tilegraph_rejects_non_sink_and_unknown_source():
    src = ArraySource(np.zeros((1, 4, 4)))
    with pytest.raises(rsgispy.RSGISPyException):
        TileGraph([src])

    class _NoInfo(TileNode):
        pass

    with pytest.raises(rsgispy.RSGISPyException, match="source_info"):
        TileGraph(MemorySink(_NoInfo()))


# --------------------------------------------------------------------------
# Metadata propagation and validation
# --------------------------------------------------------------------------


def test_metadata_propagates_to_disk_sink(tmp_path):
    arr = np.arange(2 * 12 * 14, dtype=np.float32).reshape(2, 12, 14)
    arr[0, 0, 0] = -9999.0
    in_img = _write(tmp_path / "in.tif", arr, nodata=-9999.0, names=["red", "nir"])
    src = RasterSource(in_img)
    med = if_stats.median_filter_node(src, 3, no_data_val=-9999.0)
    graph = TileGraph(DiskSink(med, tmp_path / "out.tif"), block_size=8)
    meta = graph.node_meta[med]
    assert meta == NodeMeta(2, ("red", "nir"), np.float64, -9999.0)
    assert graph.node_meta[src].dtype == np.float64
    graph.run(use_pbar=False)
    _, nodata, names = _read(tmp_path / "out.tif")
    assert nodata == -9999.0
    assert names == ("red", "nir")

    # Explicit values win; inherit_meta=False writes neither.
    TileGraph(
        DiskSink(med, tmp_path / "out2.tif", nodata=0.0, band_names=["x", "y"])
    ).run(use_pbar=False)
    assert _read(tmp_path / "out2.tif")[1:] == (0.0, ("x", "y"))
    TileGraph(DiskSink(med, tmp_path / "out3.tif", inherit_meta=False)).run(
        use_pbar=False
    )
    assert _read(tmp_path / "out3.tif")[1:] == (None, (None, None))

    # A no-data value the output type cannot store is not inherited.
    TileGraph(
        DiskSink(med, tmp_path / "out4.tif", datatype=rsgispy.TYPE_8UINT)
    ).run(use_pbar=False)
    assert _read(tmp_path / "out4.tif")[1] is None


def test_metadata_of_band_math_and_unknown_nodes():
    src3 = ArraySource(np.ones((3, 5, 5)), band_names=["b1", "b2", "b3"], nodata=0)
    src1 = ArraySource(np.ones((1, 5, 5)))
    bm = BandMathOp({"a": src3, "b": src1}, "a + b", nodata=-1.0)
    win = WindowOp(src3, 1, lambda a: a)
    win_meta = WindowOp(src3, 1, lambda a: a, meta_fn=same_bands_meta(np.float32, 5))
    graph = TileGraph(
        [MemorySink(bm), MemorySink(win), MemorySink(win_meta)], block_size=4
    )
    assert graph.node_meta[bm] == NodeMeta(3, ("b1", "b2", "b3"), np.float64, -1.0)
    assert graph.node_meta[win] == NodeMeta()
    assert graph.node_meta[win_meta] == NodeMeta(3, ("b1", "b2", "b3"), np.float32, 5)


def test_validation_errors_at_construction(tmp_path):
    two = ArraySource(np.ones((2, 6, 6)))
    three = ArraySource(np.ones((3, 6, 6)))
    with pytest.raises(rsgispy.RSGISPyException, match="incompatible band counts"):
        TileGraph(MemorySink(BandMathOp({"a": two, "b": three}, "a + b")))
    with pytest.raises(rsgispy.RSGISPyException, match="band 3"):
        TileGraph(MemorySink(SelectBands(two, [1, 3])))
    with pytest.raises(rsgispy.RSGISPyException):
        SelectBands(two, 0)
    with pytest.raises(rsgispy.RSGISPyException, match="band names"):
        TileGraph(DiskSink(two, tmp_path / "x.tif", band_names=["a", "b", "c"]))
    assert not (tmp_path / "x.tif").exists()
    in_img = _write(tmp_path / "in.tif", np.ones((2, 6, 6), dtype=np.float32))
    with pytest.raises(rsgispy.RSGISPyException, match="Band 5"):
        TileGraph(MemorySink(RasterSource(in_img, bands=5)))
    with pytest.raises(rsgispy.RSGISPyException, match="does not exist"):
        TileGraph(MemorySink(RasterSource(tmp_path / "missing.tif")))
    with pytest.raises(rsgispy.RSGISPyException):
        ArraySource(np.ones((2, 4, 4)), band_names=["a"])
    with pytest.raises(rsgispy.RSGISPyException):
        NodeMeta(n_bands=2, band_names=("a",))


# --------------------------------------------------------------------------
# Grid checks
# --------------------------------------------------------------------------


def test_grid_checks_transform_and_crs(tmp_path):
    arr = np.ones((1, 10, 10), dtype=np.float32)
    a_img = _write(tmp_path / "a.tif", arr)
    shifted = _write(tmp_path / "shifted.tif", arr, origin=(500010.0, 5000000.0))
    coarse = _write(tmp_path / "coarse.tif", arr, res=20.0)
    other_crs = _write(tmp_path / "crs.tif", arr, crs="EPSG:32631")
    nearly = _write(tmp_path / "nearly.tif", arr, origin=(500000.000001, 5000000.0))

    def _graph(path_b):
        return TileGraph(
            MemorySink(BandMathOp({"a": RasterSource(a_img), "b": RasterSource(path_b)},
                                  "a + b"))
        )

    for bad in (shifted, coarse):
        with pytest.raises(rsgispy.RSGISPyException, match="same pixel grid") as err:
            _graph(bad)
        assert "a.tif" in str(err.value) and os.path.basename(bad) in str(err.value)
    with pytest.raises(rsgispy.RSGISPyException, match="same CRS") as err:
        _graph(other_crs)
    assert "crs.tif" in str(err.value)
    _graph(nearly)  # within tolerance

    small = _write(tmp_path / "small.tif", np.ones((1, 9, 10), dtype=np.float32))
    with pytest.raises(rsgispy.RSGISPyException, match="matching dimensions"):
        _graph(small)

    # An ArraySource without transform/CRS is only checked on its size.
    res = TileGraph(
        MemorySink(
            BandMathOp({"a": RasterSource(a_img), "b": ArraySource(arr)}, "a + b"),
            name="x",
        )
    ).run(use_pbar=False)
    np.testing.assert_array_equal(res["x"], 2.0)
    with pytest.raises(rsgispy.RSGISPyException, match="same CRS"):
        TileGraph(
            MemorySink(
                BandMathOp(
                    {"a": RasterSource(a_img), "b": ArraySource(arr, crs="EPSG:4326")},
                    "a + b",
                )
            )
        )


# --------------------------------------------------------------------------
# Shared reading
# --------------------------------------------------------------------------


def test_raster_sources_on_same_file_share_one_dataset(tmp_path, monkeypatch):
    rng = np.random.default_rng(3)
    arr = rng.integers(0, 1000, size=(4, 29, 33)).astype(np.int16)
    in_img = _write(tmp_path / "in.tif", arr, nodata=-1)
    other_img = _write(tmp_path / "other.tif", arr[:1] * 2, nodata=-1)

    red = RasterSource(in_img, bands=3, name="red")
    nir = RasterSource(in_img, bands=[4], dtype=np.float32, name="nir")
    allb = RasterSource(in_img, name="all")
    other = RasterSource(other_img, name="other")
    nir_mean = if_stats.mean_filter_node(nir, 5)  # nir needs a halo of 2
    ndvi = BandMathOp({"red": red, "nir": nir_mean}, "(nir - red) / (nir + red)")
    stack = StackBands([SelectBands(allb, [2, 1]), other])
    sinks = [MemorySink(ndvi, name="ndvi"), MemorySink(stack, name="stack"),
             MemorySink(nir, name="nir")]
    graph = TileGraph(sinks, block_size=8)

    opened = []
    real_open = rasterio.open

    def _counting_open(path, *args, **kwargs):
        opened.append(str(path))
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(rasterio, "open", _counting_open)
    res = graph.run(use_pbar=False)
    monkeypatch.setattr(rasterio, "open", real_open)
    assert sorted(opened) == sorted([in_img, other_img])

    np.testing.assert_array_equal(res["nir"][0], arr[3].astype(np.float32))
    np.testing.assert_array_equal(res["stack"][:2], arr[[1, 0]].astype(np.float64))
    np.testing.assert_array_equal(res["stack"][2], arr[0] * 2.0)
    # Reference: each node reading its own copy of the file.
    ref_graph = TileGraph(
        MemorySink(
            BandMathOp(
                {
                    "red": RasterSource(in_img, bands=3),
                    "nir": if_stats.mean_filter_node(
                        RasterSource(_write(tmp_path / "nir.tif", arr[3:4], nodata=-1),
                                     dtype=np.float32),
                        5,
                    ),
                },
                "(nir - red) / (nir + red)",
            ),
            name="ndvi",
        ),
        block_size=8,
    )
    np.testing.assert_array_equal(res["ndvi"], ref_graph.run(use_pbar=False)["ndvi"])


# --------------------------------------------------------------------------
# Structural nodes and describe()
# --------------------------------------------------------------------------


def test_select_stack_cast_nodes():
    rng = np.random.default_rng(5)
    arr = rng.uniform(-5, 300, size=(3, 17, 19))
    src = ArraySource(arr, band_names=["r", "g", "b"], nodata=-9999.0)
    sel = SelectBands(src, [3, 1, 3])
    box = _Box3(SelectBands(src, 2))
    stack = StackBands([sel, box, ArraySource(arr[:1])], name="stack")
    cast = Cast(stack, np.uint8)
    graph = TileGraph(
        [MemorySink(stack, name="stack"), MemorySink(cast, name="cast")], block_size=6
    )
    meta = graph.node_meta[stack]
    assert meta.n_bands == 5
    assert meta.band_names == ("b", "r", "b", "g", "input3_b1")
    assert meta.nodata is None  # inputs disagree on no data
    assert graph.node_meta[cast].dtype == np.uint8
    assert graph.node_meta[sel].nodata == -9999.0
    res = graph.run(use_pbar=False)
    np.testing.assert_array_equal(res["stack"][:3], arr[[2, 0, 2]])
    np.testing.assert_allclose(res["stack"][3], _box3(arr[1:2])[0], atol=1e-12)
    np.testing.assert_array_equal(res["stack"][4], arr[0])
    assert res["cast"].dtype == np.uint8
    np.testing.assert_array_equal(
        res["cast"], np.clip(np.rint(res["stack"]), 0, 255).astype(np.uint8)
    )
    with pytest.raises(rsgispy.RSGISPyException):
        StackBands([])


def test_describe(tmp_path):
    in_img = _write(tmp_path / "in.tif", np.ones((2, 9, 9), dtype=np.float32),
                    nodata=0.0, names=["red", "nir"])
    src = RasterSource(in_img, name="src")
    med = if_stats.median_filter_node(src, 5)
    sobel = if_edge.sobel_filter_node(med)
    graph = TileGraph(DiskSink(sobel, tmp_path / "out.tif"), block_size=64)
    text = graph.describe()
    assert "block_size=64" in text
    assert "9 rows x 9 columns" in text
    assert "RasterSource 'src'" in text and "in.tif" in text
    assert "WindowOp 'median_filter' <- [0]" in text
    assert "WindowOp 'sobel_filter' <- [1]" in text
    assert "halo=(3, 3)" in text  # source halo: 2 (median 5x5) + 1 (sobel)
    assert "names=['red', 'nir']" in text
    assert "DiskSink" in text


# --------------------------------------------------------------------------
# Default block size
# --------------------------------------------------------------------------


def test_default_block_size(tmp_path, monkeypatch):
    monkeypatch.delenv("RSGISPY_BLOCK_SIZE", raising=False)
    monkeypatch.delenv("RSGISLIB_BLOCK_SIZE", raising=False)
    assert ip.DEFAULT_BLOCK_SIZE == 256
    assert ip.get_default_block_size() == 256
    sink = MemorySink(ArraySource(np.ones((1, 4, 4))))
    assert TileGraph(sink).block_size == 256
    ip.set_default_block_size(48)
    try:
        assert TileGraph(sink).block_size == 48
        seen = []
        real_graph = if_chained.TileGraph

        def _spy(*args, **kwargs):
            graph = real_graph(*args, **kwargs)
            seen.append(graph.block_size)
            return graph

        monkeypatch.setattr(if_chained, "TileGraph", _spy)
        in_img = _write(tmp_path / "in.tif", np.ones((1, 5, 5), dtype=np.float32))
        if_chained.apply_filter_chain(
            in_img, str(tmp_path / "o.tif"), [if_chained.FilterStep("mean")],
            use_pbar=False, calc_stats=False,
        )
        assert seen == [48]
    finally:
        ip.reset_default_block_size()


# --------------------------------------------------------------------------
# Public node builders vs the file-based filters (bit-exact)
# --------------------------------------------------------------------------

_KERNEL = np.array([[0.0, -1.0, 0.0], [-1.0, 5.0, -1.0], [0.0, -1.0, 0.0]])

_STAT_KEYS = [
    "median", "mean", "min", "max", "mode", "stddev", "range", "total",
    "mean_diff", "mean_diff_abs", "total_diff", "total_diff_abs",
    "coeff_of_var", "norm_var", "norm_var_sqrt", "norm_var_ln", "texture_var",
]

# (builder, builder kwargs, file function, file kwargs)
_BUILDER_CASES = [
    (getattr(if_stats, f"{k}_filter_node"), {"filter_size": 5},
     getattr(if_stats, f"apply_{k}_filter"), {"filter_size": 5})
    for k in _STAT_KEYS
] + [
    (if_smooth.kuwahara_filter_node, {"filter_size": 5},
     if_smooth.apply_kuwahara_filter, {"filter_size": 5}),
    (if_smooth.gaussian_smooth_filter_node,
     {"filter_size": 5, "stddev_x": 1.3, "stddev_y": 0.8, "filter_angle": 30.0},
     if_smooth.apply_gaussian_smooth_filter,
     {"filter_size": 5, "stddev_x": 1.3, "stddev_y": 0.8, "filter_angle": 30.0}),
    (if_smooth.kernel_filter_node, {"kernel": _KERNEL},
     if_smooth.apply_kernel_filter, {"kernel": _KERNEL}),
    (if_smooth.kernel_filter_node, {"kernel": np.ones((3, 5)), "normalise": True},
     if_smooth.apply_kernel_filter, {"kernel": np.ones((3, 5)), "normalise": True}),
    (if_edge.sobel_filter_node, {}, if_edge.apply_sobel_filter, {}),
    (if_edge.prewitt_filter_node, {}, if_edge.apply_prewitt_filter, {}),
    (if_edge.laplacian_filter_node, {"filter_size": 5, "stddev": 1.2},
     if_edge.apply_laplacian_filter, {"filter_size": 5, "stddev": 1.2}),
    (if_sar.lee_filter_node, {"filter_size": 5, "n_looks": 3.0},
     if_sar.apply_lee_filter, {"filter_size": 5, "n_looks": 3.0}),
]


def _case_id(case):
    return case[0].__name__ + ("_norm" if case[1].get("normalise") else "")


@pytest.fixture(scope="module")
def _builder_inputs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("builders")
    rng = np.random.default_rng(11)
    arr = rng.uniform(1.0, 50.0, size=(2, 37, 41)).astype(np.float32)
    arr[:, 10:13, 0:3] = np.round(arr[:, 10:13, 0:3])
    nd = arr.copy()
    nd[0, 0, 0] = -9999.0  # corner
    nd[1, 36, 20] = -9999.0  # bottom edge
    nd[0, 15:18, 15:18] = -9999.0  # interior hole
    nd[1, 5, 40] = -9999.0  # right edge
    nan = arr.copy()
    nan[0, 20, 0] = np.nan
    return {
        "nodata": (_write(tmp / "nd.tif", nd, nodata=-9999.0, names=["b1", "b2"]),
                   -9999.0),
        "none": (_write(tmp / "nan.tif", nan), None),
    }


@pytest.mark.parametrize("case", _BUILDER_CASES, ids=_case_id)
@pytest.mark.parametrize("variant", ["nodata", "none"])
@pytest.mark.parametrize("block_size", [16, 256])
def test_builder_equals_file_function(case, variant, block_size, _builder_inputs,
                                      tmp_path, monkeypatch):
    builder, b_kwargs, file_fn, f_kwargs = case
    in_img, nodata = _builder_inputs[variant]
    monkeypatch.setenv("RSGISPY_BLOCK_SIZE", str(block_size))
    ref_img = str(tmp_path / "ref.tif")
    file_fn(in_img, ref_img, **f_kwargs, calc_stats=False, use_pbar=False)

    node = builder(RasterSource(in_img), **b_kwargs, no_data_val=nodata)
    out_img = str(tmp_path / "node.tif")
    TileGraph(DiskSink(node, out_img), block_size=block_size).run(use_pbar=False)

    ref, ref_nd, ref_names = _read(ref_img)
    out, out_nd, out_names = _read(out_img)
    np.testing.assert_array_equal(out, ref)
    assert out_nd == ref_nd
    assert out_names == ref_names


def test_builder_integer_output_and_filter_step_equal_file_function(
    _builder_inputs, tmp_path
):
    in_img, nodata = _builder_inputs["nodata"]
    ref_img = str(tmp_path / "ref.tif")
    if_stats.apply_mode_filter(in_img, ref_img, filter_size=3,
                               datatype=rsgispy.TYPE_8UINT, calc_stats=False,
                               use_pbar=False)
    node = if_stats.mode_filter_node(RasterSource(in_img), 3, no_data_val=nodata)
    out_img = str(tmp_path / "node.tif")
    TileGraph(DiskSink(node, out_img, datatype=rsgispy.TYPE_8UINT)).run(use_pbar=False)
    np.testing.assert_array_equal(_read(out_img)[0], _read(ref_img)[0])

    # FilterStep kinds go through the same builders.
    chain_img = str(tmp_path / "chain.tif")
    if_chained.apply_filter_chain(
        in_img, chain_img,
        [if_chained.FilterStep("lee", 5, params={"n_looks": 3.0})],
        calc_stats=False, use_pbar=False,
    )
    lee_img = str(tmp_path / "lee.tif")
    if_sar.apply_lee_filter(in_img, lee_img, 5, 3.0, calc_stats=False, use_pbar=False)
    np.testing.assert_array_equal(_read(chain_img)[0], _read(lee_img)[0])


def test_builders_validate_like_file_functions():
    src = ArraySource(np.ones((1, 8, 8)))
    with pytest.raises(rsgispy.RSGISPyException):
        if_stats.median_filter_node(src, 4)
    with pytest.raises(rsgispy.RSGISPyException):
        if_stats.mean_filter_node(src, 1)
    with pytest.raises(rsgispy.RSGISPyException):
        if_sar.lee_filter_node(src, 3, n_looks=0)
    with pytest.raises(rsgispy.RSGISPyException):
        if_smooth.gaussian_smooth_filter_node(src, 3, stddev_x=-1)
    with pytest.raises(rsgispy.RSGISPyException):
        if_smooth.kernel_filter_node(src, np.array([[1, -1, 0]] * 3), normalise=True)
    with pytest.raises(rsgispy.RSGISPyException):
        if_smooth.kernel_filter_node(src, np.ones((2, 3)))
    with pytest.raises(rsgispy.RSGISPyException):
        if_edge.laplacian_filter_node(src, 3, stddev=0)
    with pytest.raises(rsgispy.RSGISPyException):
        if_chained.build_filter_chain_node(src, [if_chained.FilterStep("lee", 3, {"n_looks": -2})])
    assert if_stats.median_filter_node(src).name == "median_filter"
    assert if_stats.median_filter_node(src, name="m").name == "m"


@pytest.mark.parametrize(
    "file_dtype,node_dtype",
    [
        ("uint8", np.float64),
        ("int16", np.float64),
        ("uint16", np.float32),
        ("float32", np.float64),
        ("int32", np.float32),  # not a safe cast: read native, cast in NumPy
        ("float32", np.uint8),  # narrowing: read native, cast in NumPy
    ],
)
def test_raster_source_read_dtype_matches_numpy_cast(tmp_path, file_dtype, node_dtype):
    """GDAL-side conversion (when used) gives exactly ``raw.astype(dtype)``."""
    from rsgispy._tilegraph._exec import gdal_read_dtype

    rng = np.random.default_rng(5)
    if np.issubdtype(np.dtype(file_dtype), np.integer):
        info = np.iinfo(file_dtype)
        arr = rng.integers(info.min, info.max, size=(2, 45, 39), endpoint=True)
    else:
        arr = rng.normal(0.0, 300.0, size=(2, 45, 39))
        arr[0, :3, :3] = np.nan
        arr[1, 4, 4] = -np.inf
    arr = arr.astype(file_dtype)
    path = str(tmp_path / f"in_{file_dtype}.tif")
    with rasterio.open(
        path, "w", driver="GTiff", height=45, width=39, count=2, dtype=file_dtype,
        crs="EPSG:32630", transform=rasterio.transform.from_origin(0, 0, 10, 10),
    ) as dst:
        dst.write(arr)
    with np.errstate(invalid="ignore"):
        expected = arr.astype(node_dtype)
    single = MemorySink(RasterSource(path, dtype=node_dtype), name="single")
    shared = [
        MemorySink(RasterSource(path, bands=b, dtype=node_dtype), name=f"b{b}")
        for b in (1, 2)
    ]
    with np.errstate(invalid="ignore"):
        res = TileGraph([single] + shared, block_size=16).run(use_pbar=False)
        alone = TileGraph(
            MemorySink(RasterSource(path, dtype=node_dtype), name="x"), block_size=16
        ).run(use_pbar=False)["x"]
    for out in (res["single"], alone, np.concatenate([res["b1"], res["b2"]])):
        assert out.dtype == np.dtype(node_dtype)
        np.testing.assert_array_equal(out, expected)
    safe = np.can_cast(file_dtype, node_dtype, casting="safe")
    assert (gdal_read_dtype(file_dtype, node_dtype) is not None) == bool(safe)


def test_cast_remaps_unrepresentable_nodata():
    """H13: -9999 cast to uint8 becomes 255, in the pixels and the metadata."""
    arr = np.array([[[-9999.0, 10.4, 300.0], [-3.0, 254.6, 255.0]]])
    src = ArraySource(arr, nodata=-9999.0)
    cast = Cast(src, np.uint8)
    graph = TileGraph(MemorySink(cast, name="cast"), block_size=6)
    assert graph.node_meta[cast].nodata == 255.0
    out = graph.run(use_pbar=False)["cast"]
    # nodata -> 255; valid values round and clamp, and never become 255.
    np.testing.assert_array_equal(out[0], [[255, 10, 254], [0, 254, 254]])

    keep = Cast(ArraySource(arr, nodata=-9999.0), np.int16)
    g2 = TileGraph(MemorySink(keep, name="c"), block_size=6)
    assert g2.node_meta[keep].nodata == -9999.0
    np.testing.assert_array_equal(
        g2.run(use_pbar=False)["c"][0], [[-9999, 10, 300], [-3, 255, 255]]
    )
