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
import rsgispy.imagecalc.encoding as ic_enc
import rsgispy.imagecalc.rescale as ic_resc
import rsgispy.imagefilter.sar as if_sar
import rsgispy.imagefilter.stats as if_stats
import rsgispy.imageutils.masking as iu_msk
from rsgispy.imageprocess import (
    ArraySource,
    Cast,
    DiskSink,
    MemorySink,
    RasterSource,
    StackBands,
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


# --------------------------------------------------------------------------
# Value transforms
# --------------------------------------------------------------------------


def test_rescale_then_indices_chain_matches_files(tmp_path):
    img = pg._s2_cube(str(tmp_path))  # pylint: disable=protected-access
    objs = [ic_resc.ImageBandRescale(band=b, in_min=0.0, in_max=10000.0, out_min=0.0,
                                     out_max=1.0) for b in range(1, 7)]
    resc_file = str(tmp_path / "resc.tif")
    ic_resc.rescale_img_pxl_vals(img, resc_file, objs, use_pbar=False)
    idx_file = str(tmp_path / "idx.tif")
    ic_idx.calc_indices(resc_file, idx_file, {3: "red", 4: "nir"}, ["NDVI"],
                        calc_stats=False, use_pbar=False)

    resc = ic_resc.rescale_img_pxl_vals_node(RasterSource(img), objs)
    meta = _meta(resc)
    assert (meta.n_bands, meta.dtype, meta.nodata) == (6, np.float32, -9999.0)
    assert meta.band_names[0] == "Blue"
    np.testing.assert_array_equal(_run_mem(resc), _read(resc_file))
    ndvi = ic_idx.calc_indices_node(resc, {3: "red", 4: "nir"}, ["NDVI"])
    np.testing.assert_array_equal(_run_mem(ndvi, 23), _read(idx_file))
    with pytest.raises(rsgispy.RSGISPyException, match="Missing ImageBandRescale"):
        TileGraph(MemorySink(ic_resc.rescale_img_pxl_vals_node(RasterSource(img), objs[:3])))
    with pytest.raises(rsgispy.RSGISPyException, match="in_max == in_min"):
        ic_resc.rescale_img_pxl_vals_node(RasterSource(img), [ic_resc.ImageBandRescale(band=1)])


def test_calc_img_rescale_and_norm_nodes(tmp_path):
    img = pg._s2_cube(str(tmp_path))  # pylint: disable=protected-access
    other = pg._float_cube(str(tmp_path), "fl.tif", 2, 3)  # pylint: disable=protected-access
    out_file = str(tmp_path / "cr.tif")
    ic_resc.calc_img_rescale([img, other], out_file, c_no_data_val=0.0, c_gain=10000.0,
                             n_no_data_val=-1.0, use_pbar=False)
    node = ic_resc.calc_img_rescale_node(StackBands([RasterSource(img), RasterSource(other)]),
                                         c_no_data_val=0.0, c_gain=10000.0, n_no_data_val=-1.0)
    assert _meta(node).nodata == -1.0
    np.testing.assert_array_equal(_run_mem(node), _read(out_file))
    with pytest.raises(rsgispy.RSGISPyException, match="c_gain"):
        ic_resc.calc_img_rescale_node(RasterSource(img), c_gain=0.0)

    params = ic_resc.calc_img_norm_params(img, method="zscore", bands=[4, 2], use_pbar=False)
    norm_file = str(tmp_path / "norm.tif")
    ic_resc.apply_img_norm_params(img, norm_file, norm_params=params, use_pbar=False)
    norm = ic_resc.img_norm_params_node(RasterSource(img), params)
    assert _meta(norm).band_names == ("NIR", "Green")
    np.testing.assert_array_equal(_run_mem(norm), _read(norm_file))
    with pytest.raises(rsgispy.RSGISPyException, match="Either norm_params"):
        ic_resc.img_norm_params_node(RasterSource(img))
    with pytest.raises(rsgispy.RSGISPyException, match="out of range"):
        TileGraph(MemorySink(ic_resc.img_norm_params_node(
            RasterSource(img, bands=[1, 2]), params)))


def test_recode_then_one_hot_chain_matches_files(tmp_path):
    img = pg._class_img(str(tmp_path))  # pylint: disable=protected-access
    rec_file = str(tmp_path / "rec.tif")
    ic_enc.recode_int_raster(img, rec_file, {1: 2, 3: 5}, datatype=rsgispy.TYPE_8UINT,
                             use_pbar=False)
    oh_file = str(tmp_path / "oh.tif")
    ic_enc.one_hot_category_encoding(rec_file, oh_file, {2: 1, 4: 2, 5: 3}, use_pbar=False)

    rec = ic_enc.recode_int_raster_node(RasterSource(img, dtype=np.uint8), {1: 2, 3: 5},
                                        out_dtype=np.uint8)
    assert _meta(rec).nodata == 0.0
    np.testing.assert_array_equal(_run_mem(rec), _read(rec_file))
    onehot = ic_enc.one_hot_category_encoding_node(rec, {2: 1, 4: 2, 5: 3})
    meta = _meta(onehot)
    assert (meta.n_bands, meta.dtype) == (3, np.uint8)
    np.testing.assert_array_equal(_run_mem(onehot, 23), _read(oh_file))
    with pytest.raises(rsgispy.RSGISPyException, match="greater than or equal to 1"):
        ic_enc.one_hot_category_encoding_node(rec, {2: 0})
    with pytest.raises(rsgispy.RSGISPyException, match="must not be None"):
        ic_enc.recode_int_raster_node(rec, None)


# --------------------------------------------------------------------------
# Masks
# --------------------------------------------------------------------------


def test_indices_then_mask_img_chain_matches_files(tmp_path):
    img = pg._s2_cube(str(tmp_path))  # pylint: disable=protected-access
    msk = pg._mask_img(str(tmp_path))  # pylint: disable=protected-access
    idx_file = str(tmp_path / "idx.tif")
    ic_idx.calc_indices(img, idx_file, {3: "red", 4: "nir"}, ["NDVI"], in_gain=10000.0,
                        calc_stats=False, use_pbar=False)
    masked_file = str(tmp_path / "masked.tif")
    iu_msk.mask_img(idx_file, msk, masked_file, -999.0, 0, calc_stats=False, use_pbar=False)

    idx = ic_idx.calc_indices_node(RasterSource(img), {3: "red", 4: "nir"}, ["NDVI"],
                                   in_gain=10000.0)
    masked = iu_msk.mask_img_node(idx, RasterSource(msk), -999.0, 0)
    meta = _meta(masked)
    assert (meta.band_names, meta.nodata) == (("NDVI",), -999.0)
    out = str(tmp_path / "chain.tif")
    TileGraph(DiskSink(masked, out, clamp_int=False), block_size=23).run(use_pbar=False)
    np.testing.assert_array_equal(_read(out), _read(masked_file))
    with pytest.raises(rsgispy.RSGISPyException, match="at least one"):
        iu_msk.mask_img_node(idx, RasterSource(msk), 0, [])


def test_mask_nodes_match_files(tmp_path):
    img = pg._s2_cube(str(tmp_path))  # pylint: disable=protected-access
    flt = pg._float_cube(str(tmp_path), "fl.tif", 3, 5)  # pylint: disable=protected-access
    files = {k: str(tmp_path / f"{k}.tif") for k in ("finite", "valid", "zero", "cvm")}
    iu_msk.gen_finite_mask(flt, files["finite"], calc_stats=False, use_pbar=False)
    iu_msk.gen_valid_mask([img, flt], files["valid"], calc_stats=False, use_pbar=False)
    iu_msk.mask_all_band_zero_vals(flt, files["zero"], out_val=3.0, calc_stats=False,
                                   use_pbar=False)
    iu_msk.create_valid_mask([iu_msk.ImageBandInfo(img, "a", [2, 3]),
                              iu_msk.ImageBandInfo(flt, "b")], files["cvm"],
                             calc_stats=False, use_pbar=False)
    nodes = {
        "finite": iu_msk.gen_finite_mask_node(RasterSource(flt)),
        "valid": iu_msk.gen_valid_mask_node([RasterSource(img), RasterSource(flt)]),
        "zero": Cast(iu_msk.mask_all_band_zero_vals_node(RasterSource(flt), 3.0),
                     np.float32),
        "cvm": iu_msk.create_valid_mask_node([RasterSource(img, bands=[2, 3]),
                                              RasterSource(flt)]),
    }
    for key, node in nodes.items():
        np.testing.assert_array_equal(_run_mem(node), _read(files[key]), err_msg=key)
    assert _meta(nodes["valid"]).dtype == np.uint8


def test_edge_mask_node_with_halo_matches_files(tmp_path):
    img = pg._class_img(str(tmp_path))  # pylint: disable=protected-access
    edge_file = str(tmp_path / "edge.tif")
    iu_msk.gen_img_edge_mask(img, edge_file, 3, calc_stats=False, use_pbar=False)
    mean_file = str(tmp_path / "mean.tif")
    if_stats.apply_mean_filter(edge_file, mean_file, 5, datatype=rsgispy.TYPE_32FLOAT,
                               calc_stats=False, use_pbar=False)
    edge = iu_msk.gen_img_edge_mask_node(RasterSource(img), 3)
    np.testing.assert_array_equal(_run_mem(edge), _read(edge_file))
    # A downstream window needs the edge mask on a halo.
    mean = if_stats.mean_filter_node(edge, 5)
    out = str(tmp_path / "chain.tif")
    TileGraph(DiskSink(mean, out), block_size=16).run(use_pbar=False)
    np.testing.assert_array_equal(_read(out), _read(mean_file))
    with pytest.raises(rsgispy.RSGISPyException, match="non-negative"):
        iu_msk.gen_img_edge_mask_node(RasterSource(img), -1)
