"""
Tests for the no-data convention of the tile-graph node builders: with
``no_data_val=None`` a builder uses the no-data value of its input node's
metadata; :class:`SetNoData` sets or clears it.
"""

import numpy as np
import pytest

from rsgispy.imageprocess import (
    ArraySource,
    MemorySink,
    PointwiseOp,
    SetNoData,
    TileGraph,
    infer_node_meta,
    resolve_input_nodata,
)
from rsgispy.classification.classpostprocess import majority_filter_node
from rsgispy.imagefilter.chained import FilterStep, build_filter_chain_node
from rsgispy.imagefilter.edge import sobel_filter_node
from rsgispy.imagefilter.sar import refined_lee_filter_node
from rsgispy.imagefilter.stats import median_filter_node
from rsgispy.imagefilter.texture import glcm_filter_node
from rsgispy.imagemorphology import (
    image_dilate_node,
    image_local_minima_node,
    image_opening_node,
    image_white_top_hat_node,
)

ND = -9999.0


def _arr():
    rng = np.random.default_rng(4)
    arr = rng.integers(1, 6, size=(2, 41, 37)).astype(np.float64)
    arr[:, 5:9, 10:20] = ND
    arr[1, 30:, :4] = ND
    return arr


BUILDERS = {
    "median": lambda src, nd: median_filter_node(src, 5, no_data_val=nd),
    "sobel": lambda src, nd: sobel_filter_node(src, no_data_val=nd),
    "refined_lee": lambda src, nd: refined_lee_filter_node(src, 5, no_data_val=nd),
    "glcm": lambda src, nd: glcm_filter_node(
        src, filter_size=5, n_levels=8, min_val=0, max_val=6, no_data_val=nd
    ),
    "dilate": lambda src, nd: image_dilate_node(src, op_size=3, no_data_val=nd),
    "minima": lambda src, nd: image_local_minima_node(src, op_size=3, no_data_val=nd),
    "opening": lambda src, nd: image_opening_node(src, op_size=3, no_data_val=nd),
    "top_hat": lambda src, nd: image_white_top_hat_node(src, op_size=3, no_data_val=nd),
    "majority": lambda src, nd: majority_filter_node(src, 3, no_data_val=nd),
    "chain": lambda src, nd: build_filter_chain_node(
        src,
        [FilterStep("median", 3), FilterStep("band_math", params={"expr": "x * 2"})],
        no_data_val=nd,
    ),
}


def _run(node):
    return TileGraph(MemorySink(node, name="out"), block_size=16).run(use_pbar=False)["out"]


@pytest.mark.parametrize("kind", sorted(BUILDERS))
def test_builder_default_nodata_is_input_metadata(kind):
    arr = _arr()
    build = BUILDERS[kind]
    with_meta = _run(build(ArraySource(arr, nodata=ND), None))
    explicit = _run(build(ArraySource(arr), ND))
    np.testing.assert_array_equal(with_meta, explicit)
    # SetNoData(..., None) disables the no-data handling.
    cleared = _run(build(SetNoData(ArraySource(arr, nodata=ND), None), None))
    no_nd = _run(build(ArraySource(arr), None))
    np.testing.assert_array_equal(cleared, no_nd)
    assert not np.array_equal(with_meta, no_nd, equal_nan=True)
    # SetNoData(..., value) declares one.
    declared = _run(build(SetNoData(ArraySource(arr), ND), None))
    np.testing.assert_array_equal(declared, explicit)


def test_explicit_value_overrides_metadata():
    arr = _arr()
    out = _run(median_filter_node(ArraySource(arr, nodata=1.0), 3, no_data_val=ND))
    ref = _run(median_filter_node(ArraySource(arr), 3, no_data_val=ND))
    np.testing.assert_array_equal(out, ref)


def test_set_nodata_metadata_and_pixels_unchanged():
    arr = _arr()
    src = ArraySource(arr, nodata=ND, band_names=["a", "b"])
    node = SetNoData(src, None)
    meta = infer_node_meta(node)
    assert meta.nodata is None and meta.band_names == ("a", "b")
    assert infer_node_meta(SetNoData(src, 0)).nodata == 0.0
    np.testing.assert_array_equal(_run(node), arr)
    assert "nodata=None" in TileGraph(MemorySink(node)).describe()


def test_resolve_input_nodata():
    src = ArraySource(np.zeros((4, 4)), nodata=3.0)
    assert resolve_input_nodata(src) == 3.0
    assert resolve_input_nodata(src, 5.0) == 5.0
    assert resolve_input_nodata(PointwiseOp(src, lambda a: a)) is None
    assert resolve_input_nodata(median_filter_node(src, 3)) == 3.0
