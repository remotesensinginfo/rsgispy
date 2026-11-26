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


# --------------------------------------------------------------------------
# Change detection
# --------------------------------------------------------------------------


def test_index_difference_chain_matches_files(tmp_path):
    import rsgispy.changedetect.imgchngmthds as icm  # pylint: disable=import-outside-toplevel

    base, chng = pg._change_pair(str(tmp_path))  # pylint: disable=protected-access
    bands = {1: "red", 2: "nir"}
    files = {k: str(tmp_path / f"{k}.tif") for k in ("i1", "i2", "diff", "ratio")}
    ic_idx.calc_indices(base, files["i1"], bands, ["NDVI"], calc_stats=False, use_pbar=False)
    ic_idx.calc_indices(chng, files["i2"], bands, ["NDVI"], calc_stats=False, use_pbar=False)
    icm.image_difference(files["i1"], files["i2"], files["diff"], base_img_no_data=-999,
                         chng_img_no_data=-999, no_data_val=-999, calc_stats=False,
                         use_pbar=False)
    icm.image_ratio(base, chng, files["ratio"], img_base_bands=[2, 3],
                    img_chng_bands=[1, 3], calc_stats=False, use_pbar=False)

    nd1 = ic_idx.calc_indices_node(RasterSource(base), bands, ["NDVI"])
    nd2 = ic_idx.calc_indices_node(RasterSource(chng), bands, ["NDVI"])
    diff = icm.image_difference_node(nd1, nd2, base_img_no_data=-999, chng_img_no_data=-999,
                                     no_data_val=-999)
    meta = _meta(diff)
    assert (meta.n_bands, meta.dtype, meta.nodata) == (1, np.float64, -999)
    out = str(tmp_path / "chain.tif")
    TileGraph(DiskSink(diff, out, clamp_int=False), block_size=23).run(use_pbar=False)
    np.testing.assert_array_equal(_read(out), _read(files["diff"]))

    ratio = icm.image_ratio_node(RasterSource(base, bands=[2, 3]),
                                 RasterSource(chng, bands=[1, 3]))
    np.testing.assert_array_equal(_run_mem(Cast(ratio, np.float32)), _read(files["ratio"]))
    with pytest.raises(rsgispy.RSGISPyException, match="do not match"):
        TileGraph(MemorySink(icm.image_ratio_node(RasterSource(base, bands=[1]),
                                                  RasterSource(chng))))


def test_image_difference_node_with_std_params_and_mask(tmp_path):
    import rsgispy.changedetect.imgchngmthds as icm  # pylint: disable=import-outside-toplevel

    base, chng = pg._change_pair(str(tmp_path))  # pylint: disable=protected-access
    msk = pg._mask_img(str(tmp_path))  # pylint: disable=protected-access
    out_file = str(tmp_path / "d.tif")
    icm.image_difference(base, chng, out_file, vld_msk_img=msk, apply_std=True,
                         calc_stats=False, use_pbar=False)
    params = icm._change_std_params(  # pylint: disable=protected-access
        base, chng, msk, 1, [1, 2, 3], [1, 2, 3], (0, 0), (pg.WIDTH, pg.HEIGHT), False
    )
    node = icm.image_difference_node(RasterSource(base), RasterSource(chng),
                                     RasterSource(msk, dtype=np.uint8), std_params=params)
    np.testing.assert_array_equal(_run_mem(Cast(node, np.float32)), _read(out_file))
    with pytest.raises(rsgispy.RSGISPyException, match="std_params"):
        icm.image_difference_node(RasterSource(base), RasterSource(chng), std_params=[1, 2])


# --------------------------------------------------------------------------
# Multi-date percentiles
# --------------------------------------------------------------------------


def test_index_median_composite_chain_matches_files(tmp_path):
    import rsgispy.imagecalc.percentiles as ic_pct  # pylint: disable=import-outside-toplevel

    imgs = []
    for idx in range(3):
        arr = np.rint(pg.pattern(2, 70 + idx, 0.0, 3000.0)).astype(np.uint16)
        arr[:, idx:idx + 4, 5] = 0
        imgs.append(pg.write_img(str(tmp_path / f"d{idx}.tif"), arr, nodata=0))
    bands = {1: "red", 2: "nir"}
    idx_files = []
    for idx, img in enumerate(imgs):
        idx_files.append(str(tmp_path / f"i{idx}.tif"))
        ic_idx.calc_indices(img, idx_files[-1], bands, ["NDVI"], calc_stats=False,
                            use_pbar=False)
    med_file = str(tmp_path / "med.tif")
    ic_pct.calc_imgs_pxl_percentiles(idx_files, med_file, [25, 50], no_data_val=-999,
                                     calc_stats=False, use_pbar=False)
    nodes = [ic_idx.calc_indices_node(RasterSource(img), bands, ["NDVI"]) for img in imgs]
    med = ic_pct.calc_imgs_pxl_percentiles_node(nodes, [25, 50], no_data_val=-999,
                                                out_dtype=np.float32)
    meta = _meta(med)
    assert (meta.band_names, meta.nodata) == (("p25", "p50"), -999)
    np.testing.assert_array_equal(_run_mem(med, 23), _read(med_file))

    band_file = str(tmp_path / "bpct.tif")
    ic_pct.calc_img_band_pxl_percentiles(imgs, band_file, [50, 90], calc_stats=False,
                                         use_pbar=False)
    bnode = ic_pct.calc_img_band_pxl_percentiles_node(
        [RasterSource(img, dtype=np.float32) for img in imgs], [50, 90], out_dtype=np.uint16
    )
    assert _meta(bnode).band_names == ("b1_p50", "b1_p90", "b2_p50", "b2_p90")
    np.testing.assert_array_equal(_run_mem(bnode), _read(band_file))
    with pytest.raises(rsgispy.RSGISPyException, match="different number of bands"):
        TileGraph(MemorySink(ic_pct.calc_img_band_pxl_percentiles_node(
            [RasterSource(imgs[0]), RasterSource(imgs[1], bands=1)], [50])))
    with pytest.raises(rsgispy.RSGISPyException):
        ic_pct.calc_imgs_pxl_percentiles_node(nodes, [150])


# --------------------------------------------------------------------------
# Spectral transforms, SAM / SCM and unmixing
# --------------------------------------------------------------------------


def test_tasseled_cap_and_pca_nodes_match_files(tmp_path):
    import rsgispy.imagespectral.transforms as st  # pylint: disable=import-outside-toplevel
    from rsgispy.imageprocess import SelectBands  # pylint: disable=import-outside-toplevel

    img = pg._s2_cube(str(tmp_path))  # pylint: disable=protected-access
    msk = pg._mask_img(str(tmp_path))  # pylint: disable=protected-access
    tc_file = str(tmp_path / "tc.tif")
    st.calc_tasseled_cap(img, tc_file, coeffs=[[0.5, 0.5], [0.3, -0.7]], bands=[4, 3],
                         valid_msk_img=msk, gain=10000.0, calc_stats=False, use_pbar=False)
    tc = st.calc_tasseled_cap_node(SelectBands(RasterSource(img), [4, 3]),
                                   coeffs=[[0.5, 0.5], [0.3, -0.7]], gain=10000.0,
                                   vld_msk=RasterSource(msk, dtype=np.uint8))
    meta = _meta(tc)
    assert (meta.band_names, meta.dtype, meta.nodata) == (("TC_1", "TC_2"), np.float32, 0.0)
    np.testing.assert_array_equal(_run_mem(tc), _read(tc_file))
    with pytest.raises(rsgispy.RSGISPyException, match="Expected 6 input bands"):
        TileGraph(MemorySink(st.calc_tasseled_cap_node(RasterSource(img, bands=[1, 2]))))
    with pytest.raises(rsgispy.RSGISPyException, match="Unknown Tasseled Cap sensor"):
        st.calc_tasseled_cap_node(RasterSource(img), sensor="hubble")

    pca_file = str(tmp_path / "pca.tif")
    info = st.calc_pca(img, pca_file, n_components=2, standardise=True, calc_stats=False,
                       use_pbar=False)
    pcs = st.pca_projection_node(RasterSource(img), info)
    assert _meta(pcs).band_names == ("PC_1", "PC_2")
    np.testing.assert_array_equal(_run_mem(pcs, 23), _read(pca_file))
    mnf_file = str(tmp_path / "mnf.tif")
    info = st.calc_mnf(img, mnf_file, n_components=2, valid_msk_img=msk, calc_stats=False,
                       use_pbar=False)
    mnf = st.mnf_projection_node(RasterSource(img), info, RasterSource(msk, dtype=np.uint8))
    np.testing.assert_array_equal(_run_mem(mnf), _read(mnf_file))
    with pytest.raises(rsgispy.RSGISPyException, match="calc_pca"):
        st.pca_projection_node(RasterSource(img), {})


def test_sam_scm_nodes_match_files(tmp_path):
    import rsgispy.imagespectral.classification as sc  # pylint: disable=import-outside-toplevel

    img = pg._s2_cube(str(tmp_path))  # pylint: disable=protected-access
    ems = pg._endmembers()  # pylint: disable=protected-access
    files = {k: str(tmp_path / f"{k}.tif") for k in ("ang", "cls", "min", "corr", "ccls")}
    sc.spectral_angle_mapper(img, files["ang"], ems, out_class_img=files["cls"],
                             out_min_angle_img=files["min"], img_gain=10000.0,
                             max_angle=0.3, calc_stats=False, use_pbar=False)
    sc.spectral_correlation_mapper(img, files["corr"], ems, out_class_img=files["ccls"],
                                   min_corr=0.5, calc_stats=False, use_pbar=False)
    src = RasterSource(img)
    for output, key in (("scores", "ang"), ("class", "cls"), ("best", "min")):
        node = sc.spectral_angle_mapper_node(src, ems, img_gain=10000.0, max_angle=0.3,
                                             output=output)
        np.testing.assert_array_equal(_run_mem(node), _read(files[key]), err_msg=output)
    meta = _meta(sc.spectral_angle_mapper_node(src, ems, output="class"))
    assert (meta.dtype, meta.nodata) == (np.uint8, 0.0)
    assert _meta(sc.spectral_angle_mapper_node(src, ems, output="all")).band_names == (
        "veg", "soil", "water", "min_angle", "class")
    for output, key in (("scores", "corr"), ("class", "ccls")):
        node = sc.spectral_correlation_mapper_node(src, ems, min_corr=0.5, output=output)
        np.testing.assert_array_equal(_run_mem(node, 23), _read(files[key]), err_msg=output)
    with pytest.raises(rsgispy.RSGISPyException, match="does not match"):
        TileGraph(MemorySink(sc.spectral_angle_mapper_node(RasterSource(img, bands=[1]), ems)))
    with pytest.raises(rsgispy.RSGISPyException, match="min_corr"):
        sc.spectral_correlation_mapper_node(src, ems, min_corr=2.0)
    with pytest.raises(rsgispy.RSGISPyException, match="output must be"):
        sc.spectral_angle_mapper_node(src, ems, output="angles")


def test_unmixing_then_prediction_chain_matches_files(tmp_path):
    import rsgispy.imagespectral.unmixing as su  # pylint: disable=import-outside-toplevel

    img = pg._mix_cube(str(tmp_path))  # pylint: disable=protected-access
    msk = pg._mask_img(str(tmp_path))  # pylint: disable=protected-access
    ems = pg._endmembers() * 10000.0  # pylint: disable=protected-access
    files = {k: str(tmp_path / f"{k}.tif") for k in ("fcls", "pred", "ucls")}
    su.spec_unmix_fcls(img, files["fcls"], ems, valid_msk_img=msk, gain=10000.0,
                       calc_stats=False, use_pbar=False)
    su.predict_refl_linear_unmixing(files["fcls"], files["pred"], ems, calc_stats=False,
                                    use_pbar=False)
    su.spec_unmix_ucls(img, files["ucls"], ems, gain=10000.0, weight=0.5, calc_stats=False,
                       use_pbar=False)

    fcls = su.spec_unmix_fcls_node(RasterSource(img), ems,
                                   RasterSource(msk, dtype=np.uint8), gain=10000.0)
    meta = _meta(fcls)
    assert (meta.band_names, meta.dtype, meta.nodata) == (
        ("veg", "soil", "water"), np.float32, -9999.0)
    pred = su.predict_refl_linear_unmixing_node(fcls, ems)
    # Same tiles as the file-based runs (package default block size).
    with_bs = TileGraph([MemorySink(fcls, name="f"), MemorySink(pred, name="p")]).run(
        use_pbar=False)
    np.testing.assert_array_equal(with_bs["f"], _read(files["fcls"]))
    np.testing.assert_array_equal(with_bs["p"], _read(files["pred"]))
    ucls = su.spec_unmix_ucls_node(RasterSource(img), ems, gain=10000.0, weight=0.5)
    np.testing.assert_array_equal(_run_mem(ucls, 256), _read(files["ucls"]))
    with pytest.raises(rsgispy.RSGISPyException, match="expected 6"):
        TileGraph(MemorySink(su.spec_unmix_nnls_node(RasterSource(img, bands=[1, 2]), ems)))
    with pytest.raises(rsgispy.RSGISPyException, match="endmembers define 3"):
        TileGraph(MemorySink(su.predict_refl_linear_unmixing_node(
            RasterSource(img, bands=[1, 2]), ems)))
