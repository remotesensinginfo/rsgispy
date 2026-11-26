"""
Golden-output tests for the morphology, classification post-processing and
raster prediction file functions: the current implementation must reproduce
the reference outputs generated before they were rebuilt on the tile graph
(see ``tests/golden_morph_class_ml.py``), integers exactly and floats within
1 ulp, at several block sizes.
"""

import json

import numpy as np
import pytest

from golden_morph_class_ml import (
    BLOCK_SIZES,
    all_cases,
    load_golden,
    make_inputs,
    meta_equal,
    run_case,
)


@pytest.fixture(scope="module")
def golden_inputs(tmp_path_factory):
    """Synthetic inputs shared by all golden cases."""
    work = tmp_path_factory.mktemp("golden_phase5")
    return work, make_inputs(work)


@pytest.fixture(scope="module")
def golden_refs():
    """Golden arrays of the three groups."""
    return {g: load_golden(g) for g in ("morph", "postproc", "predict")}


def _assert_pixels_equal(name, got, ref):
    assert got.dtype == ref.dtype, f"{name}: dtype {got.dtype} != {ref.dtype}"
    assert got.shape == ref.shape, f"{name}: shape {got.shape} != {ref.shape}"
    if np.issubdtype(ref.dtype, np.floating):
        nan_ref = np.isnan(ref)
        np.testing.assert_array_equal(np.isnan(got), nan_ref, err_msg=name)
        np.testing.assert_array_max_ulp(got[~nan_ref], ref[~nan_ref], maxulp=1)
    else:
        np.testing.assert_array_equal(got, ref, err_msg=name)


@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("case", sorted(all_cases()))
def test_golden_output(case, block_size, golden_inputs, golden_refs, tmp_path):
    """The file function reproduces its golden output at ``block_size``."""
    _, inputs = golden_inputs
    refs = golden_refs[case.split("_", 1)[0]]
    tag = "all" if f"{case}__all__0__data" in refs else str(block_size)
    results = run_case(case, inputs, tmp_path, block_size)
    n_ref = len([k for k in refs if k.startswith(f"{case}__{tag}__") and k.endswith("__data")])
    assert len(results) == n_ref
    for idx, (data, meta, ctab) in enumerate(results):
        key = f"{case}__{tag}__{idx}"
        _assert_pixels_equal(key, data, refs[f"{key}__data"])
        ref_meta = json.loads(str(refs[f"{key}__meta"]))
        assert meta_equal(meta, ref_meta), f"{key}: {meta} != {ref_meta}"
        np.testing.assert_array_equal(ctab, refs[f"{key}__ctab"])
