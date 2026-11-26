"""
Tests for the pixel-wise tile-graph node builders (phase 3): each node in a
multi-step graph gives the same result as running the file-based functions
one after another, the node metadata (bands, names, dtype, no data) is
correct, and builders validate their parameters like the file-based
functions.
"""

# pylint: disable=too-many-locals,missing-function-docstring

import numpy as np
import pytest
import rasterio

import rsgispy
import rsgispy.imagecalc.calcindices as ic_idx
import rsgispy.imagefilter.sar as if_sar
import rsgispy.imagefilter.stats as if_stats
from rsgispy.imageprocess import (
    ArraySource,
    DiskSink,
    MemorySink,
    RasterSource,
    TileGraph,
)

import pixelwise_golden as pg


def _read(path):
    with rasterio.open(path) as ds:
        return ds.read()


def _meta(node):
    return TileGraph(MemorySink(node, name="m")).node_meta[node]


def _run_mem(node, block_size=16):
    return TileGraph(MemorySink(node, name="m"), block_size=block_size).run(use_pbar=False)["m"]


# --------------------------------------------------------------------------
# Indices
# --------------------------------------------------------------------------


def test_calc_indices_node_chain_matches_files(tmp_path):
    img = pg._s2_cube(str(tmp_path))  # pylint: disable=protected-access
    bands = {1: "blue", 2: "green", 3: "red", 4: "nir", 5: "swir1", 6: "swir2"}
    idx_file = str(tmp_path / "idx.tif")
    ic_idx.calc_indices(img, idx_file, bands, ["NDVI", "EVI", "NBR"], in_gain=10000.0,
                        calc_stats=False, use_pbar=False)
    med_file = str(tmp_path / "med.tif")
    if_stats.apply_median_filter(idx_file, med_file, 5, no_data_val=-999.0,
                                 datatype=rsgispy.TYPE_32FLOAT, calc_stats=False,
                                 use_pbar=False)

    node = ic_idx.calc_indices_node(RasterSource(img), bands, ["NDVI", "EVI", "NBR"],
                                    in_gain=10000.0)
    meta = _meta(node)
    assert meta.n_bands == 3
    assert meta.band_names == ("NDVI", "EVI", "NBR")
    assert meta.dtype == np.float32
    assert meta.nodata == -999.0
    chained = if_stats.median_filter_node(node, 5, no_data_val=-999.0)
    out = str(tmp_path / "chain.tif")
    TileGraph(DiskSink(chained, out), block_size=16).run(use_pbar=False)
    np.testing.assert_array_equal(_read(out), _read(med_file))
    # The index node alone equals calc_indices.
    np.testing.assert_array_equal(_run_mem(node, 23), _read(idx_file))


def test_calc_indices_node_int_and_errors(tmp_path):
    img = pg._s2_cube(str(tmp_path))  # pylint: disable=protected-access
    node = ic_idx.calc_indices_node(RasterSource(img), {3: "red", 4: "nir"}, ["NDVI"],
                                    in_gain=10000.0, out_gain=1000.0, out_dtype=np.int16)
    out_file = str(tmp_path / "ndvi.tif")
    ic_idx.calc_indices(img, out_file, {3: "red", 4: "nir"}, ["NDVI"], in_gain=10000.0,
                        out_gain=1000.0, datatype=rsgispy.TYPE_16INT, calc_stats=False,
                        use_pbar=False)
    res = _run_mem(node)
    assert res.dtype == np.int16
    np.testing.assert_array_equal(res, _read(out_file))
    with pytest.raises(rsgispy.RSGISPyException, match="out of range"):
        TileGraph(MemorySink(ic_idx.calc_indices_node(RasterSource(img), {9: "red", 4: "nir"},
                                                      ["NDVI"])))
    with pytest.raises(rsgispy.RSGISPyException, match="requires band"):
        TileGraph(MemorySink(ic_idx.calc_indices_node(RasterSource(img), {3: "red"}, ["NDVI"])))
    with pytest.raises(rsgispy.RSGISPyException, match="Brightness_Scaled"):
        TileGraph(MemorySink(ic_idx.calc_indices_node(
            RasterSource(img), {1: "blue", 2: "green", 3: "red"}, ["Brightness_Scaled"])))
    with pytest.raises(rsgispy.RSGISPyException):
        ic_idx.calc_indices_node(RasterSource(img), {3: "red", 4: "nir"}, ["NDVI"],
                                 out_dtype=np.uint8, out_no_data_val=-5)


def test_calc_sar_indices_node_after_speckle_filter(tmp_path):
    arr = pg.pattern(2, 3, 0.001, 0.5).astype(np.float32)
    img = pg.write_img(str(tmp_path / "s1.tif"), arr)
    lee_file = str(tmp_path / "lee.tif")
    if_sar.apply_lee_filter(img, lee_file, 5, datatype=rsgispy.TYPE_32FLOAT,
                            calc_stats=False, use_pbar=False)
    idx_file = str(tmp_path / "idx.tif")
    names = ["RVI_VV_VH", "CR_VH_VV_dB", "DPSVI"]
    ic_idx.calc_sar_indices(lee_file, idx_file, {1: "VV", 2: "VH"}, names, "power",
                            index_params={"DPSVI": {"vv_max": 0.6}}, calc_stats=False,
                            use_pbar=False)
    lee = if_sar.lee_filter_node(RasterSource(img), 5)
    # The Lee node gives float64 blocks; the file is float32.
    from rsgispy.imageprocess import Cast  # pylint: disable=import-outside-toplevel

    node = ic_idx.calc_sar_indices_node(Cast(lee, np.float32), {1: "VV", 2: "VH"}, names,
                                        "power", index_params={"DPSVI": {"vv_max": 0.6}})
    assert _meta(node).band_names == tuple(names)
    np.testing.assert_array_equal(_run_mem(node), _read(idx_file))
    with pytest.raises(rsgispy.RSGISPyException, match="vv_max"):
        TileGraph(MemorySink(ic_idx.calc_sar_indices_node(
            RasterSource(img), {1: "VV", 2: "VH"}, ["DPSVI"], "dB")))
    with pytest.raises(rsgispy.RSGISPyException):
        ic_idx.calc_sar_indices_node(RasterSource(img), {1: "VV"}, ["RVI"], "furlongs")


def test_calc_indices_node_on_array_source():
    data = np.stack([np.full((5, 6), 0.2), np.full((5, 6), 0.6)])
    node = ic_idx.calc_indices_node(ArraySource(data), {1: "red", 2: "nir"}, ["NDVI"],
                                    out_dtype=np.float64)
    np.testing.assert_allclose(_run_mem(node), 0.5)
