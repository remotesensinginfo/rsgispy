"""
Tests for the tile-graph node builders of :mod:`rsgispy.imagemorphology`:
nodes chained in one graph must equal the file-based functions run one
after another, and the builders validate their parameters like the file
functions.
"""

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

import rsgispy
import rsgispy.imagemorphology as rm
from rsgispy.imageprocess import (
    DiskSink,
    MemorySink,
    RasterSource,
    TileGraph,
)


def _write(path, data, nodata=None, descriptions=None):
    arr = data if data.ndim == 3 else data[np.newaxis]
    profile = {
        "driver": "GTiff",
        "height": arr.shape[1],
        "width": arr.shape[2],
        "count": arr.shape[0],
        "dtype": str(arr.dtype),
        "crs": "EPSG:32630",
        "transform": from_origin(500000.0, 6000000.0, 10.0, 10.0),
    }
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr)
        for idx, desc in enumerate(descriptions or [], start=1):
            dst.set_band_description(idx, desc)
    return str(path)


def _read(path):
    with rasterio.open(path) as src:
        return src.read(), src.nodata, src.dtypes[0]


@pytest.fixture(name="inputs")
def fixture_inputs(tmp_path):
    rng = np.random.default_rng(5)
    u8 = rng.integers(1, 90, size=(2, 41, 39)).astype(np.uint8)
    u8[rng.random(u8.shape) < 0.05] = 0
    f32 = rng.normal(10.0, 3.0, size=(41, 39)).astype(np.float32)
    f32[rng.random(f32.shape) < 0.05] = -9999.0
    return {
        "u8": _write(tmp_path / "u8.tif", u8, nodata=0, descriptions=["b1", "b2"]),
        "f32": _write(tmp_path / "f32.tif", f32, nodata=-9999.0),
        "dir": tmp_path,
    }


@pytest.mark.parametrize("block_size", [16, 256])
def test_morph_chain_equals_file_functions(inputs, block_size, monkeypatch):
    """dilate -> opening -> white top-hat in one graph == three file calls."""
    monkeypatch.setenv("RSGISPY_BLOCK_SIZE", str(block_size))
    tmp = inputs["dir"]
    # File functions one after another (uint8 output, header no data 0).
    rm.image_dilate(inputs["u8"], str(tmp / "a.tif"), op_size=3, use_pbar=False)
    rm.image_opening(
        str(tmp / "a.tif"), str(tmp / "b.tif"), op_size=5, niters=2, use_pbar=False
    )
    rm.image_white_top_hat(str(tmp / "b.tif"), str(tmp / "c.tif"), use_pbar=False)

    src = RasterSource(inputs["u8"])
    node = rm.image_dilate_node(src, op_size=3, dtype=np.uint8, no_data_val=0)
    node = rm.image_opening_node(node, op_size=5, niters=2, dtype=np.uint8, no_data_val=0)
    node = rm.image_white_top_hat_node(node, dtype=np.uint8, no_data_val=0)
    out = str(tmp / "graph.tif")
    TileGraph(DiskSink(node, out, datatype=rsgispy.TYPE_8UINT)).run(use_pbar=False)

    ref, ref_nd, ref_dt = _read(str(tmp / "c.tif"))
    got, got_nd, got_dt = _read(out)
    np.testing.assert_array_equal(got, ref)
    assert (got_nd, got_dt) == (ref_nd, ref_dt)
    with rasterio.open(out) as src_ds:
        # Band names are carried through the nodes.
        assert src_ds.descriptions == ("b1", "b2")


def test_morph_float_chain_and_combined(inputs):
    """Float chain erode -> closing -> gradient (combined) == file calls."""
    tmp = inputs["dir"]
    rm.image_erode(inputs["f32"], str(tmp / "e.tif"), op_size=3, use_pbar=False)
    rm.image_closing(str(tmp / "e.tif"), str(tmp / "c.tif"), op_size=3, use_pbar=False)
    rm.image_black_top_hat(str(tmp / "c.tif"), str(tmp / "t.tif"), use_pbar=False)

    src = RasterSource(inputs["f32"])
    nd = -9999.0
    node = rm.image_erode_node(src, no_data_val=nd)
    node = rm.image_closing_node(node, no_data_val=nd)
    node = rm.image_black_top_hat_node(node, no_data_val=nd)
    res = TileGraph(MemorySink(node, dtype=np.float32, name="x"), block_size=13).run(
        use_pbar=False
    )["x"]
    ref, _, _ = _read(str(tmp / "t.tif"))
    np.testing.assert_array_equal(res, ref)

    rm.image_gradient_combined_out(
        inputs["u8"], str(tmp / "g.tif"), op_size=5, use_pbar=False
    )
    node = rm.image_gradient_node(
        RasterSource(inputs["u8"]), op_size=5, combined=True, no_data_val=0,
        dtype=np.uint8,
    )
    graph = TileGraph(MemorySink(node, dtype=np.uint8, name="g"), block_size=16)
    assert graph.node_meta[node].n_bands == 1
    ref, _, _ = _read(str(tmp / "g.tif"))
    np.testing.assert_array_equal(graph.run(use_pbar=False)["g"], ref)


def test_local_minima_node(inputs):
    """Binary local minima node == file function; sequential numbering refused."""
    tmp = inputs["dir"]
    rm.image_local_minima(
        inputs["f32"], str(tmp / "m.tif"), allow_equal=False, use_pbar=False
    )
    node = rm.image_local_minima_node(
        RasterSource(inputs["f32"]), allow_equal=False, no_data_val=-9999.0
    )
    res = TileGraph(MemorySink(node, dtype=np.uint8, name="m"), block_size=8).run(
        use_pbar=False
    )["m"]
    ref, _, _ = _read(str(tmp / "m.tif"))
    np.testing.assert_array_equal(res, ref)
    with pytest.raises(rsgispy.RSGISPyException, match="sequencial_out"):
        rm.image_local_minima_node(RasterSource(inputs["f32"]), sequencial_out=True)


def test_morph_builders_validate_like_file_functions(inputs):
    """Builders raise the same exceptions as the file functions."""
    src = RasterSource(inputs["f32"])
    with pytest.raises(rsgispy.RSGISPyException):
        rm.image_dilate_node(src, op_size=4)
    with pytest.raises(rsgispy.RSGISPyException):
        rm.image_erode_node(src, morph_op_file="missing.gmtxt", use_op_file=True)
    with pytest.raises(rsgispy.RSGISPyException):
        rm.image_opening_node(src, niters=0)
    with pytest.raises(rsgispy.RSGISPyException):
        rm.image_opening(inputs["f32"], str(inputs["dir"] / "x.tif"), niters=0)


def test_morph_node_operator_file(inputs):
    """An operator file gives the same result through the node and the file function."""
    tmp = inputs["dir"]
    rm.create_circular_op(str(tmp / "circ"), op_size=5)
    op_file = str(tmp / "circ.gmtxt")
    rm.image_dilate(
        inputs["f32"], str(tmp / "d.tif"), op_file, use_op_file=True, use_pbar=False
    )
    node = rm.image_dilate_node(
        RasterSource(inputs["f32"]), op_file, use_op_file=True, no_data_val=-9999.0,
        dtype=np.float32,
    )
    graph = TileGraph(MemorySink(node, dtype=np.float32, name="d"), block_size=10)
    assert graph.node_meta[node].nodata == -9999.0
    ref, _, _ = _read(str(tmp / "d.tif"))
    np.testing.assert_array_equal(graph.run(use_pbar=False)["d"], ref)


@pytest.mark.parametrize("mode", ["reflect", "constant"])
def test_morph_nodes_border_mode_equal_file_functions(inputs, mode):
    """The node builders take border_mode / border_fill like the file functions."""
    tmp = inputs["dir"]
    # Reflect padding cannot change a square-operator dilation (the reflected
    # pixels are already under the operator); a high constant does.
    fill = 250.0 if mode == "constant" else 0.0
    bkw = {"border_mode": mode, "border_fill": fill}
    rm.image_dilate(inputs["u8"], str(tmp / "d.tif"), op_size=5, use_pbar=False, **bkw)
    rm.image_closing(inputs["u8"], str(tmp / "c.tif"), op_size=3, use_pbar=False, **bkw)
    rm.image_white_top_hat(inputs["u8"], str(tmp / "w.tif"), use_pbar=False, **bkw)
    src = RasterSource(inputs["u8"])
    nodes = {
        "d": rm.image_dilate_node(src, op_size=5, dtype=np.uint8, **bkw),
        "c": rm.image_closing_node(src, op_size=3, dtype=np.uint8, **bkw),
        "w": rm.image_white_top_hat_node(src, dtype=np.uint8, **bkw),
    }
    res = TileGraph(
        [MemorySink(n, dtype=np.uint8, name=k) for k, n in nodes.items()], block_size=9
    ).run(use_pbar=False)
    for key in nodes:
        ref, _, _ = _read(str(tmp / f"{key}.tif"))
        np.testing.assert_array_equal(res[key], ref)
    default = TileGraph(
        MemorySink(rm.image_dilate_node(src, op_size=5, dtype=np.uint8), dtype=np.uint8,
                   name="e"), block_size=9,
    ).run(use_pbar=False)["e"]
    assert np.array_equal(default, res["d"]) == (mode != "constant")
    np.testing.assert_array_equal(default[:, 2:-2, 2:-2], res["d"][:, 2:-2, 2:-2])
    with pytest.raises(rsgispy.RSGISPyException):
        rm.image_erode_node(src, border_mode="wrap")
