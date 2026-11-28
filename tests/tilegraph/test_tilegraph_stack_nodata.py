"""
Tests for the no-data value of :class:`StackBands`: inputs with different
no-data values get one no-data value written into every band of a pixel where
any input is invalid, and ``describe()`` labels the internal stack of
``SampleSink`` / ``ZonalStatsSink``.
"""

import math

import numpy as np
import pytest

import rsgispy
from rsgispy.imageprocess import (
    ArraySource,
    MemorySink,
    SampleSink,
    StackBands,
    TileGraph,
    ZonalStatsSink,
    infer_node_meta,
)

H, W = 23, 19


def _refl():
    """uint16 reflectance with no data 0."""
    rng = np.random.default_rng(1)
    arr = rng.integers(1, 1000, size=(2, H, W)).astype(np.uint16)
    arr[:, :3, :] = 0
    return arr


def _index():
    """float32 'index' with no data -999."""
    rng = np.random.default_rng(2)
    arr = rng.uniform(-1, 1, size=(1, H, W)).astype(np.float32)
    arr[0, :, -4:] = -999.0
    return arr


def _labels():
    arr = (np.arange(H * W).reshape(1, H, W) % 3 + 1).astype(np.uint8)
    arr[0, 10:12, :] = 0
    return arr


def _run(node, block_size=8):
    return TileGraph(MemorySink(node, name="x"), block_size=block_size).run(
        use_pbar=False
    )["x"]


def test_equal_nodata_unchanged():
    a = ArraySource(_refl(), nodata=0)
    b = ArraySource(_refl()[:1] + 1, nodata=0)
    stack = StackBands([a, b])
    assert infer_node_meta(stack).nodata == 0
    out = _run(stack)
    # Bands are concatenated as before: no masking across inputs.
    np.testing.assert_array_equal(out, np.concatenate([_refl(), _refl()[:1] + 1]))


def test_all_missing_nodata_unchanged():
    stack = StackBands([ArraySource(_refl()), ArraySource(_index())])
    assert infer_node_meta(stack).nodata is None
    out = _run(stack)
    np.testing.assert_array_equal(out[:2], _refl())
    np.testing.assert_array_equal(out[2:], _index())


def test_differing_nodata_default_warns_and_keeps_bands():
    stack = StackBands(
        [ArraySource(_refl(), nodata=0), ArraySource(_index(), nodata=-999)]
    )
    with pytest.warns(UserWarning, match="no_data_val"):
        meta = infer_node_meta(stack)
    assert meta.nodata is None
    with pytest.warns(UserWarning):
        out = _run(stack)
    np.testing.assert_array_equal(out[:2], _refl())
    np.testing.assert_array_equal(out[2:], _index())


def test_auto_uses_first_and_masks():
    stack = StackBands(
        [ArraySource(_refl(), nodata=0), ArraySource(_index(), nodata=-999)],
        no_data_val="auto",
    )
    meta = infer_node_meta(stack)
    assert meta.nodata == 0
    out = _run(stack)
    invalid = np.any(_refl() == 0, axis=0) | (_index()[0] == -999)
    assert np.all(out[:, invalid] == 0)
    np.testing.assert_array_equal(out[:2, ~invalid], _refl()[:, ~invalid])
    np.testing.assert_allclose(out[2, ~invalid], _index()[0, ~invalid])


def test_auto_nan_when_first_has_none():
    idx_nd = ArraySource(_index(), nodata=-999)
    stack = StackBands([ArraySource(_index() * 2), idx_nd], no_data_val="auto")
    meta = infer_node_meta(stack)
    assert math.isnan(meta.nodata)
    out = _run(stack)
    invalid = _index()[0] == -999
    assert np.all(np.isnan(out[:, invalid]))
    assert not np.any(np.isnan(out[:, ~invalid]))


def test_auto_integer_without_first_raises():
    stack = StackBands(
        [ArraySource(_refl()), ArraySource(_refl(), nodata=0)], no_data_val="auto"
    )
    with pytest.raises(rsgispy.RSGISPyException, match="no_data_val"):
        infer_node_meta(stack)
    with pytest.raises(rsgispy.RSGISPyException):
        StackBands([ArraySource(_refl())], no_data_val="first")


def test_given_no_data_val():
    stack = StackBands(
        [ArraySource(_refl(), nodata=0), ArraySource(_index(), nodata=-999)],
        no_data_val=-1,
    )
    assert infer_node_meta(stack).nodata == -1
    out = _run(stack)
    invalid = np.any(_refl() == 0, axis=0) | (_index()[0] == -999)
    assert np.all(out[:, invalid] == -1)
    assert not np.any(out[:, ~invalid] == -1)
    # A value that does not fit the stack dtype raises.
    with pytest.raises(rsgispy.RSGISPyException):
        infer_node_meta(StackBands([ArraySource(_refl(), nodata=0)], no_data_val=-1))


def test_block_size_invariant():
    stack = StackBands(
        [ArraySource(_refl(), nodata=0), ArraySource(_index(), nodata=-999)],
        no_data_val=-1,
    )
    np.testing.assert_array_equal(_run(stack, 8), _run(stack, 64))


def test_sample_sink_on_mixed_stack_excludes_every_input():
    feats = StackBands(
        [ArraySource(_refl(), nodata=0), ArraySource(_index(), nodata=-999)],
        band_names=["red", "nir", "ndvi"],
        no_data_val=-999,
    )
    df = TileGraph(
        SampleSink(feats, ArraySource(_labels(), nodata=0), name="s"), block_size=8
    ).run(use_pbar=False)["s"]
    valid = (
        (_labels()[0] != 0)
        & np.all(_refl() != 0, axis=0)
        & (_index()[0] != -999)
    )
    assert len(df) == int(valid.sum())
    assert not np.any(df["ndvi"] == -999)
    assert not np.any(df["red"] == 0)


def test_describe_shows_stack_nodata_and_internal_nodes():
    inputs = [ArraySource(_refl(), nodata=0), ArraySource(_index(), nodata=-999)]
    with pytest.warns(UserWarning):
        text = TileGraph(MemorySink(StackBands(inputs), name="x")).describe()
    assert "differ: stack nodata=None, not masked" in text

    feats = StackBands(inputs, no_data_val="auto")
    text = TileGraph(MemorySink(feats, name="x")).describe()
    assert "nodata=0" in text
    assert "invalid in any input" in text

    labels = ArraySource(_labels(), nodata=0)
    for sink in (
        SampleSink(feats, labels, name="s"),
        ZonalStatsSink(labels, feats, name="z"),
    ):
        text = TileGraph(sink).describe()
        assert "_ZonalStack" not in text
        assert "input1_b1" not in text
        assert "internal" in text
