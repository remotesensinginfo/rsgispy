"""
Golden-output tests for the pixel-wise file-based functions (indices, value
transforms, masks, change detection, multi-date percentiles, spectral
transforms and unmixing): the outputs (pixels, data type, no data, band
names, statistics, colour tables) must match the references generated with
the implementation before these functions became tile-graph wrappers,
integers exactly and floats within 1 ulp, for several block sizes.
"""

import os

import pytest

import pixelwise_golden as golden


@pytest.mark.parametrize("block_size", golden.BLOCK_SIZES)
@pytest.mark.parametrize("case", sorted(golden.CASES))
def test_pixelwise_golden(case, block_size):
    """Outputs of ``case`` match the stored golden outputs."""
    assert os.path.exists(golden.golden_path(case)), f"missing golden file for {case}"
    arrays, meta = golden.load_golden(case)
    outputs = golden.run_case(case, block_size)
    expected_labels = set(meta)
    if not golden.HAS_KEA:
        expected_labels -= golden.KEA_OUTPUTS.get(case, frozenset())
    assert sorted(outputs) == sorted(expected_labels)
    for label, out in outputs.items():
        key = f"{label}__all"
        expected = arrays[key] if key in arrays else arrays[f"{label}__bs{block_size}"]
        golden.assert_close_ulp(out["data"], expected, f"{case}/{label}")
        golden.assert_meta_equal(out["meta"], meta[label], f"{case}/{label}")
