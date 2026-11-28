"""
Tests for the phase-2 filter node builders (directional edges, Gaussian
derivatives, SAR speckle filters, GLCM texture) and their ``FilterStep``
kinds: each builder is bit-exact against its file-based function, chains of
nodes equal the file functions run one after another, and the builders
validate their parameters like the file functions.
"""

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

import rsgispy
import rsgispy.imagefilter.chained as if_chained
import rsgispy.imagefilter.edge as if_edge
import rsgispy.imagefilter.sar as if_sar
import rsgispy.imagefilter.stats as if_stats
import rsgispy.imagefilter.texture as if_tex
from rsgispy.imageprocess import (
    ArraySource,
    DiskSink,
    MemorySink,
    RasterSource,
    TileGraph,
)


def _write(path, arr, nodata=None, names=None):
    profile = {
        "driver": "GTiff",
        "height": arr.shape[1],
        "width": arr.shape[2],
        "count": arr.shape[0],
        "dtype": str(arr.dtype),
        "crs": "EPSG:32630",
        "transform": from_origin(500000.0, 5000000.0, 10.0, 10.0),
        "nodata": nodata,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr)
        if names:
            for idx, name in enumerate(names, start=1):
                dst.set_band_description(idx, name)
    return str(path)


def _read(path):
    with rasterio.open(path) as ds:
        return ds.read(), ds.nodata, ds.descriptions


@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("phase2_nodes")
    rng = np.random.default_rng(5)
    arr = rng.gamma(3.0, 5.0, size=(2, 33, 39)).astype(np.float32) + 1.0
    arr[:, 8:12, 20:26] = 12.0
    nd = arr.copy()
    nd[0, 0, 0] = -9999.0
    nd[1, 32, 11] = -9999.0
    nd[0, 14:17, 14:17] = -9999.0
    nd[1, 6, 38] = -9999.0
    nan = arr[:1].copy()
    nan[0, 20, 0] = np.nan
    nan[0, 3:5, 30:33] = np.nan
    return {
        "nodata": (_write(tmp / "nd.tif", nd, nodata=-9999.0, names=["vv", "vh"]),
                   -9999.0),
        "none": (_write(tmp / "nan.tif", nan), None),
    }


# (builder, kwargs, file function, kwargs, FilterStep)
CASES = [
    (if_edge.sobel_x_filter_node, {}, if_edge.apply_sobel_x_filter, {},
     if_chained.FilterStep("sobel_x")),
    (if_edge.sobel_y_filter_node, {}, if_edge.apply_sobel_y_filter, {},
     if_chained.FilterStep("sobel_y")),
    (if_edge.prewitt_x_filter_node, {}, if_edge.apply_prewitt_x_filter, {},
     if_chained.FilterStep("prewitt_x")),
    (if_edge.prewitt_y_filter_node, {}, if_edge.apply_prewitt_y_filter, {},
     if_chained.FilterStep("prewitt_y")),
    (if_edge.gaussian_1st_deriv_filter_node,
     {"filter_size": 5, "stddev_x": 1.2, "stddev_y": 0.7, "filter_angle": 30.0},
     if_edge.apply_gaussian_1st_deriv_filter,
     {"filter_size": 5, "stddev_x": 1.2, "stddev_y": 0.7, "filter_angle": 30.0},
     if_chained.FilterStep("gaussian_1st_deriv", 5,
                           {"stddev_x": 1.2, "stddev_y": 0.7, "filter_angle": 30.0})),
    (if_edge.gaussian_2nd_deriv_filter_node,
     {"filter_size": 7, "stddev_x": 1.5, "stddev_y": 1.5},
     if_edge.apply_gaussian_2nd_deriv_filter,
     {"filter_size": 7, "stddev_x": 1.5, "stddev_y": 1.5},
     if_chained.FilterStep("gaussian_2nd_deriv", 7, {"stddev": 1.5})),
    (if_sar.enhanced_lee_filter_node, {"filter_size": 5, "n_looks": 2.0, "damping": 1.5},
     if_sar.apply_enhanced_lee_filter, {"filter_size": 5, "n_looks": 2.0, "damping": 1.5},
     if_chained.FilterStep("enhanced_lee", 5, {"n_looks": 2.0, "damping": 1.5})),
    (if_sar.frost_filter_node, {"filter_size": 5, "damping": 1.2},
     if_sar.apply_frost_filter, {"filter_size": 5, "damping": 1.2},
     if_chained.FilterStep("frost", 5, {"damping": 1.2})),
    (if_sar.enhanced_frost_filter_node, {"filter_size": 5, "n_looks": 3.0},
     if_sar.apply_enhanced_frost_filter, {"filter_size": 5, "n_looks": 3.0},
     if_chained.FilterStep("enhanced_frost", 5, {"n_looks": 3.0})),
    (if_sar.kuan_filter_node, {"filter_size": 5, "n_looks": 3.0},
     if_sar.apply_kuan_filter, {"filter_size": 5, "n_looks": 3.0},
     if_chained.FilterStep("kuan", 5, {"n_looks": 3.0})),
    (if_sar.gamma_map_filter_node, {"filter_size": 5, "n_looks": 4.0},
     if_sar.apply_gamma_map_filter, {"filter_size": 5, "n_looks": 4.0},
     if_chained.FilterStep("gamma_map", 5, {"n_looks": 4.0})),
    (if_sar.refined_lee_filter_node, {"filter_size": 7, "n_looks": 2.0},
     if_sar.apply_refined_lee_filter, {"filter_size": 7, "n_looks": 2.0},
     if_chained.FilterStep("refined_lee", 7, {"n_looks": 2.0})),
    (if_tex.glcm_filter_node,
     {"properties": ["contrast", "entropy"], "filter_size": 5, "min_val": 0.0,
      "max_val": 60.0, "n_levels": 16},
     if_tex.apply_glcm_filter,
     {"properties": ["contrast", "entropy"], "filter_size": 5, "min_val": 0.0,
      "max_val": 60.0, "n_levels": 16},
     if_chained.FilterStep("glcm", 5, {"properties": ["contrast", "entropy"],
                                       "min_val": 0.0, "max_val": 60.0,
                                       "n_levels": 16})),
]


def _case_id(case):
    return case[0].__name__


@pytest.mark.parametrize("case", CASES, ids=_case_id)
@pytest.mark.parametrize("variant", ["nodata", "none"])
@pytest.mark.parametrize("block_size", [16, 256])
def test_builder_equals_file_function(case, variant, block_size, inputs, tmp_path,
                                      monkeypatch):
    builder, b_kwargs, file_fn, f_kwargs, _ = case
    in_img, nodata = inputs[variant]
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


@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_filter_step_kind_equals_builder(case, inputs):
    builder, b_kwargs, _, _, step = case
    in_img, nodata = inputs["nodata"]
    via_step = if_chained.build_filter_chain_node(
        RasterSource(in_img), [step], no_data_val=nodata
    )
    direct = builder(RasterSource(in_img), **b_kwargs, no_data_val=nodata)
    res = TileGraph(
        [MemorySink(via_step, name="step"), MemorySink(direct, name="direct")],
        block_size=16,
    ).run(use_pbar=False)
    np.testing.assert_array_equal(res["step"], res["direct"])


def test_glcm_node_with_band_bounds_equals_file_function(inputs, tmp_path):
    in_img, nodata = inputs["nodata"]
    ref_img = str(tmp_path / "ref.tif")
    if_tex.apply_glcm_filter(in_img, ref_img, filter_size=5, calc_stats=False,
                             use_pbar=False)
    bounds = if_tex.calc_glcm_band_bounds(in_img, use_pbar=False)
    assert len(bounds) == 2 and all(lo < hi for lo, hi in bounds)
    node = if_tex.glcm_filter_node(RasterSource(in_img), filter_size=5,
                                   band_bounds=bounds, no_data_val=nodata)
    graph = TileGraph(DiskSink(node, str(tmp_path / "node.tif")))
    meta = graph.node_meta[node]
    assert meta.n_bands == 20
    assert meta.band_names[0] == "vv_glcm_contrast"
    assert meta.band_names[-1] == "vh_glcm_max_prob"
    graph.run(use_pbar=False)
    ref, ref_nd, ref_names = _read(ref_img)
    out, out_nd, out_names = _read(str(tmp_path / "node.tif"))
    np.testing.assert_array_equal(out, ref)
    assert out_nd == ref_nd
    assert out_names == ref_names


def test_glcm_chain_names_and_unnamed_bands(inputs, tmp_path):
    in_img, _ = inputs["none"]
    out_img = str(tmp_path / "chain.tif")
    if_chained.apply_filter_chain(
        in_img, out_img,
        [if_chained.FilterStep("median", 3),
         if_chained.FilterStep("glcm", 5, {"properties": "contrast",
                                           "min_val": 0.0, "max_val": 60.0})],
        calc_stats=False, use_pbar=False,
    )
    out, _, names = _read(out_img)
    assert out.shape[0] == 1
    assert names == ("b1_glcm_contrast",)


def test_chained_nodes_equal_sequential_file_functions(inputs, tmp_path, monkeypatch):
    """A multi-step graph equals the file functions run one after another."""
    monkeypatch.setenv("RSGISPY_BLOCK_SIZE", "16")
    in_img, nodata = inputs["nodata"]
    f64 = {"datatype": rsgispy.TYPE_64FLOAT, "calc_stats": False, "use_pbar": False}
    step1 = str(tmp_path / "s1.tif")
    step2 = str(tmp_path / "s2.tif")
    step3 = str(tmp_path / "s3.tif")
    if_sar.apply_refined_lee_filter(in_img, step1, 7, 2.0, **f64)
    if_edge.apply_gaussian_2nd_deriv_filter(step1, step2, 5, 1.0, 2.0, 60.0, **f64)
    if_tex.apply_glcm_filter(step2, step3, properties=["homogeneity", "mean"],
                             filter_size=5, min_val=-20.0, max_val=20.0, n_levels=8,
                             **f64)

    src = RasterSource(in_img)
    n1 = if_sar.refined_lee_filter_node(src, 7, 2.0, no_data_val=nodata)
    n2 = if_edge.gaussian_2nd_deriv_filter_node(n1, 5, 1.0, 2.0, 60.0,
                                                no_data_val=nodata)
    n3 = if_tex.glcm_filter_node(n2, properties=["homogeneity", "mean"],
                                 filter_size=5, min_val=-20.0, max_val=20.0,
                                 n_levels=8, no_data_val=nodata)
    res = TileGraph(
        [MemorySink(n2, name="n2"), MemorySink(n3, name="n3")], block_size=16
    ).run(use_pbar=False)
    np.testing.assert_array_equal(res["n2"], _read(step2)[0])
    np.testing.assert_array_equal(res["n3"], _read(step3)[0])


def test_new_builders_validate_like_file_functions():
    src = ArraySource(np.ones((1, 8, 8)))
    bad = [
        lambda: if_edge.gaussian_1st_deriv_filter_node(src, 4),
        lambda: if_edge.gaussian_2nd_deriv_filter_node(src, 3, stddev_y=0.0),
        lambda: if_sar.enhanced_lee_filter_node(src, 5, n_looks=0.0),
        lambda: if_sar.enhanced_lee_filter_node(src, 5, damping=-1.0),
        lambda: if_sar.frost_filter_node(src, 5, damping=0.0),
        lambda: if_sar.enhanced_frost_filter_node(src, 2),
        lambda: if_sar.kuan_filter_node(src, 5, n_looks=-1.0),
        lambda: if_sar.gamma_map_filter_node(src, 1),
        lambda: if_sar.refined_lee_filter_node(src, 7, n_looks=float("nan")),
        lambda: if_tex.glcm_filter_node(src, "bogus", min_val=0, max_val=1),
        lambda: if_tex.glcm_filter_node(src, distances=5, filter_size=5,
                                        min_val=0, max_val=1),
        lambda: if_tex.glcm_filter_node(src, n_levels=1, min_val=0, max_val=1),
        lambda: if_tex.glcm_filter_node(src, min_val=0.0),
        lambda: if_tex.glcm_filter_node(src, min_val=2.0, max_val=1.0),
        lambda: if_tex.glcm_filter_node(src),
        lambda: if_tex.glcm_filter_node(src, band_bounds=[(2.0, 1.0)]),
        lambda: if_chained.build_filter_chain_node(
            src, [if_chained.FilterStep("glcm", 5)]),
        lambda: if_chained.build_filter_chain_node(
            src, [if_chained.FilterStep("refined_lee", 7, {"n_looks": 0})]),
    ]
    for call in bad:
        with pytest.raises(rsgispy.RSGISPyException):
            call()
    # The number of band ranges must match the input bands.
    node = if_tex.glcm_filter_node(src, band_bounds=[(0.0, 1.0), (0.0, 2.0)])
    with pytest.raises(rsgispy.RSGISPyException, match="band_bounds"):
        TileGraph(MemorySink(node))
    assert if_sar.refined_lee_filter_node(src).name == "refined_lee_filter"
    assert if_edge.sobel_x_filter_node(src, name="gx").name == "gx"
    assert if_tex.glcm_filter_node(src, min_val=0, max_val=1).name == "glcm_filter"
    assert if_stats.median_filter_node(src).halo_radius == (1, 1)
    assert if_sar.refined_lee_filter_node(src, 9).halo_radius == (4, 4)


def test_kernel_filter_node_convolves_and_correlation_node_correlates():
    """kernel_filter_node is a true convolution and
    kernel_correlation_filter_node a correlation, as are their FilterStep
    kinds ('kernel' and 'kernel_correlation')."""
    import rsgispy.imagefilter.smooth as if_smooth  # pylint: disable=import-outside-toplevel

    arr = np.arange(1, 64, dtype=np.float64).reshape(1, 7, 9)
    shift = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    src = ArraySource(arr)
    conv = if_smooth.kernel_filter_node(src, shift)
    corr = if_smooth.kernel_correlation_filter_node(src, shift)
    step_conv = if_chained.build_filter_chain_node(
        src, [if_chained.FilterStep("kernel", params={"kernel": shift})]
    )
    step_corr = if_chained.build_filter_chain_node(
        src, [if_chained.FilterStep("kernel_correlation", params={"kernel": shift})]
    )
    res = TileGraph(
        [
            MemorySink(conv, name="conv"),
            MemorySink(corr, name="corr"),
            MemorySink(step_conv, name="step_conv"),
            MemorySink(step_corr, name="step_corr"),
        ],
        block_size=4,
    ).run(use_pbar=False)
    img = arr[0]
    # Convolution: out[r, c] = in[r - 1, c - 1]; correlation: in[r + 1, c + 1].
    np.testing.assert_array_equal(res["conv"][0, 1:, 1:], img[:-1, :-1])
    np.testing.assert_array_equal(res["corr"][0, :-1, :-1], img[1:, 1:])
    np.testing.assert_array_equal(res["step_conv"], res["conv"])
    np.testing.assert_array_equal(res["step_corr"], res["corr"])
