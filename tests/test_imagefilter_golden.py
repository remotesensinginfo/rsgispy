"""
Golden-output tests for the file-based ``rsgispy.imagefilter`` functions.

The reference outputs in ``tests/data/golden/imagefilter_*.npz`` were made
with the block-loop implementation of every ``apply_*_filter`` function
(before they became wrappers around the tile graph), on small synthetic
inputs that cover:

* block sizes that do and do not divide the image (7, 16) and one larger
  than the image (256);
* image edges, interior no-data holes and NaN pixels;
* multi-band inputs (band names carried through);
* float32, float64 and uint16 inputs, and float32, uint16 and uint8 outputs,
  including an explicit ``no_data_val`` that the output type cannot store.

Integer outputs must match exactly and floating-point outputs to within 1 ulp
(no-data positions exactly). The references are platform specific in the
last bit (``libm``, SIMD summation order), so on a platform other than the
one recorded in the files the comparison falls back to a small tolerance.

To regenerate the references (only when an output change is intended)::

    PYTHONPATH=. python tests/test_imagefilter_golden.py --regen
"""

import json
import os
import platform
import sys
import tempfile

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

import rsgispy
import rsgispy.imagefilter.edge as if_edge
import rsgispy.imagefilter.sar as if_sar
import rsgispy.imagefilter.smooth as if_smooth
import rsgispy.imagefilter.stats as if_stats
import rsgispy.imagefilter.texture as if_tex

GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "golden")
BLOCK_SIZES = (7, 16, 256)
IMG_H, IMG_W = 21, 26

_SHARPEN = [[0.0, -1.0, 0.0], [-1.0, 5.0, -1.0], [0.0, -1.0, 0.0]]

# (module key, case id, function, keyword arguments)
CASES = [
    ("stats", "median3", if_stats.apply_median_filter, {"filter_size": 3}),
    ("stats", "mean5", if_stats.apply_mean_filter, {"filter_size": 5}),
    ("stats", "min3", if_stats.apply_min_filter, {"filter_size": 3}),
    ("stats", "max5", if_stats.apply_max_filter, {"filter_size": 5}),
    ("stats", "mode5", if_stats.apply_mode_filter, {"filter_size": 5}),
    ("stats", "stddev3", if_stats.apply_stddev_filter, {"filter_size": 3}),
    ("stats", "range5", if_stats.apply_range_filter, {"filter_size": 5}),
    ("stats", "total3", if_stats.apply_total_filter, {"filter_size": 3}),
    ("stats", "mean_diff5", if_stats.apply_mean_diff_filter, {"filter_size": 5}),
    ("stats", "mean_diff_abs3", if_stats.apply_mean_diff_abs_filter, {"filter_size": 3}),
    ("stats", "total_diff3", if_stats.apply_total_diff_filter, {"filter_size": 3}),
    ("stats", "total_diff_abs5", if_stats.apply_total_diff_abs_filter, {"filter_size": 5}),
    ("stats", "coeff_of_var5", if_stats.apply_coeff_of_var_filter, {"filter_size": 5}),
    ("stats", "norm_var3", if_stats.apply_norm_var_filter, {"filter_size": 3}),
    ("stats", "norm_var_sqrt5", if_stats.apply_norm_var_sqrt_filter, {"filter_size": 5}),
    ("stats", "norm_var_ln3", if_stats.apply_norm_var_ln_filter, {"filter_size": 3}),
    ("stats", "texture_var7", if_stats.apply_texture_var_filter, {"filter_size": 7}),
    ("smooth", "kuwahara5", if_smooth.apply_kuwahara_filter, {"filter_size": 5}),
    ("smooth", "gaussian5", if_smooth.apply_gaussian_smooth_filter,
     {"filter_size": 5, "stddev_x": 1.3, "stddev_y": 0.8, "filter_angle": 30.0}),
    ("smooth", "kernel_sharpen", if_smooth.apply_kernel_filter, {"kernel": _SHARPEN}),
    ("smooth", "kernel_norm", if_smooth.apply_kernel_filter,
     {"kernel": np.ones((3, 5)), "normalise": True}),
    ("edge", "sobel", if_edge.apply_sobel_filter, {}),
    ("edge", "sobel_x", if_edge.apply_sobel_x_filter, {}),
    ("edge", "sobel_y", if_edge.apply_sobel_y_filter, {}),
    ("edge", "prewitt", if_edge.apply_prewitt_filter, {}),
    ("edge", "prewitt_x", if_edge.apply_prewitt_x_filter, {}),
    ("edge", "prewitt_y", if_edge.apply_prewitt_y_filter, {}),
    ("edge", "gauss_d1", if_edge.apply_gaussian_1st_deriv_filter,
     {"filter_size": 5, "stddev_x": 1.2, "stddev_y": 0.9, "filter_angle": 20.0}),
    ("edge", "gauss_d2", if_edge.apply_gaussian_2nd_deriv_filter,
     {"filter_size": 5, "stddev_x": 1.0, "stddev_y": 1.5, "filter_angle": 45.0}),
    ("edge", "laplacian5", if_edge.apply_laplacian_filter,
     {"filter_size": 5, "stddev": 1.2}),
    ("sar", "lee5", if_sar.apply_lee_filter, {"filter_size": 5, "n_looks": 3.0}),
    ("sar", "enhanced_lee5", if_sar.apply_enhanced_lee_filter,
     {"filter_size": 5, "n_looks": 2.0, "damping": 1.5}),
    ("sar", "frost5", if_sar.apply_frost_filter, {"filter_size": 5, "damping": 1.2}),
    ("sar", "enhanced_frost5", if_sar.apply_enhanced_frost_filter,
     {"filter_size": 5, "n_looks": 2.0, "damping": 1.0}),
    ("sar", "kuan5", if_sar.apply_kuan_filter, {"filter_size": 5, "n_looks": 3.0}),
    ("sar", "gamma_map5", if_sar.apply_gamma_map_filter, {"filter_size": 5, "n_looks": 4.0}),
    ("sar", "refined_lee7", if_sar.apply_refined_lee_filter,
     {"filter_size": 7, "n_looks": 2.0}),
    ("texture", "glcm_default", if_tex.apply_glcm_filter, {"filter_size": 5}),
    ("texture", "glcm_offsets", if_tex.apply_glcm_filter,
     {"properties": ["contrast", "entropy", "correlation"], "filter_size": 5,
      "distances": (1, 2), "angles": (0.0, 90.0), "n_levels": 16,
      "min_val": 0.0, "max_val": 130.0, "symmetric": False,
      "average_angles": False}),
]

VARIANTS = ("f32nd", "u16nd", "f32nan", "f64x")


def _pattern(n_bands, seed, modulus):
    """Deterministic integer test pattern (no RNG, no transcendental maths)."""
    i = np.arange(IMG_H, dtype=np.int64)[:, None]
    j = np.arange(IMG_W, dtype=np.int64)[None, :]
    bands = []
    for b in range(n_bands):
        val = (i * 7919 + j * 104729 + (b + seed) * 1299709 + i * j * 31) % modulus
        # A flat patch and a gentle ramp, so windows with ties and windows
        # without no data both occur.
        val[2:8, 14:22] = 40 + b
        val[12:19, 1:6] = 20 + (i[12:19] + j[:, 1:6]) // 3
        bands.append(val)
    return np.stack(bands)


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
    return path


def make_inputs(tmp_dir):
    """
    Write the synthetic inputs; return ``{variant: (path, call_kwargs)}``.
    """
    # float32, 2 bands, header no data -9999 at a corner, edges and a hole.
    f32 = (_pattern(2, 1, 997).astype(np.float32) / np.float32(8.0)) + np.float32(1.0)
    f32[0, 0, 0] = -9999.0
    f32[1, IMG_H - 1, 9] = -9999.0
    f32[0, 9:12, 10:13] = -9999.0
    f32[1, 4, IMG_W - 1] = -9999.0
    f32[1, 0, :4] = -9999.0
    # uint16, 1 band, header no data 0 (occurs in the pattern), with ties.
    u16 = _pattern(1, 3, 61).astype(np.uint16)
    u16[0, IMG_H - 2:, IMG_W - 3:] = 0
    # float32, 1 band, no header no data, NaN pixels (also on edges).
    f32n = (_pattern(1, 5, 499).astype(np.float32) / np.float32(4.0)) + np.float32(0.5)
    f32n[0, 0, 7] = np.nan
    f32n[0, 10, 0] = np.nan
    f32n[0, 14:16, 18:20] = np.nan
    # float64, 1 band, explicit no_data_val=300 (not storable in uint8).
    f64 = _pattern(1, 7, 131).astype(np.float64) + 0.25
    f64[0, 3, 3] = 300.0
    f64[0, 17:20, 20:23] = 300.0
    f64[0, 0, IMG_W - 1] = 300.0
    return {
        "f32nd": (_write(os.path.join(tmp_dir, "f32nd.tif"), f32, nodata=-9999.0,
                         names=["red", "nir"]), {}),
        "u16nd": (_write(os.path.join(tmp_dir, "u16nd.tif"), u16, nodata=0),
                  {"datatype": rsgispy.TYPE_16UINT}),
        "f32nan": (_write(os.path.join(tmp_dir, "f32nan.tif"), f32n), {}),
        "f64x": (_write(os.path.join(tmp_dir, "f64x.tif"), f64, names=["dem"]),
                 {"no_data_val": 300.0, "datatype": rsgispy.TYPE_8UINT}),
    }


def _platform_tag():
    # pylint: disable=import-outside-toplevel
    from rsgispy._numba import use_compiled
    from rsgispy.imagefilter._helpers import _stat_mode_kernel

    compiled = use_compiled(_stat_mode_kernel, "golden", fallback_desc="numpy")
    return f"{sys.platform}-{platform.machine()}-numba{int(bool(compiled))}"


def _run_case(func, kwargs, in_img, call_kwargs, out_img, block_size, calc_stats=False):
    old = os.environ.get("RSGISPY_BLOCK_SIZE")
    os.environ["RSGISPY_BLOCK_SIZE"] = str(block_size)
    try:
        func(in_img, out_img, **kwargs, **call_kwargs, calc_stats=calc_stats,
             use_pbar=False)
    finally:
        if old is None:
            os.environ.pop("RSGISPY_BLOCK_SIZE", None)
        else:
            os.environ["RSGISPY_BLOCK_SIZE"] = old
    with rasterio.open(out_img) as ds:
        data = ds.read()
        meta = {
            "nodata": None if ds.nodata is None else repr(float(ds.nodata)),
            "names": list(ds.descriptions),
            "dtype": str(data.dtype),
        }
        if calc_stats:
            meta["tags"] = [ds.tags(b) for b in range(1, ds.count + 1)]
            meta["overviews"] = [ds.overviews(b) for b in range(1, ds.count + 1)]
    return data, meta


def _same_bits(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(
        a, b, equal_nan=np.issubdtype(a.dtype, np.floating)
    )


def regenerate():
    """Write the golden ``.npz`` files with the current implementation."""
    os.makedirs(GOLDEN_DIR, exist_ok=True)
    per_module = {}
    with tempfile.TemporaryDirectory() as tmp:
        inputs = make_inputs(tmp)
        for mod, case_id, func, kwargs in CASES:
            arrays, metas = per_module.setdefault(mod, ({}, {}))
            for variant in VARIANTS:
                in_img, call_kwargs = inputs[variant]
                ref = None
                for bs in sorted(BLOCK_SIZES, reverse=True):
                    out_img = os.path.join(tmp, f"{case_id}_{variant}_{bs}.tif")
                    stats = bs == 256 and variant == "f32nd"
                    data, meta = _run_case(func, kwargs, in_img, call_kwargs, out_img,
                                           bs, calc_stats=stats)
                    key = f"{case_id}|{variant}|{bs}"
                    if ref is None or not _same_bits(ref, data):
                        arrays[key] = data
                    if ref is None:
                        ref = data
                    metas[key] = meta
    tag = _platform_tag()
    for mod, (arrays, metas) in per_module.items():
        arrays["__meta__"] = np.array(json.dumps({"platform": tag, "outputs": metas}))
        np.savez_compressed(os.path.join(GOLDEN_DIR, f"imagefilter_{mod}.npz"), **arrays)


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

_GOLDEN_CACHE = {}


def _golden(mod):
    if mod not in _GOLDEN_CACHE:
        path = os.path.join(GOLDEN_DIR, f"imagefilter_{mod}.npz")
        with np.load(path) as npz:
            arrays = {k: npz[k] for k in npz.files if k != "__meta__"}
            meta = json.loads(str(npz["__meta__"]))
        _GOLDEN_CACHE[mod] = (arrays, meta)
    return _GOLDEN_CACHE[mod]


@pytest.fixture(scope="module")
def golden_inputs(tmp_path_factory):
    return make_inputs(str(tmp_path_factory.mktemp("golden_inputs")))


def _assert_matches(out, ref, same_platform, label):
    assert out.dtype == ref.dtype, label
    assert out.shape == ref.shape, label
    if np.issubdtype(ref.dtype, np.integer):
        if same_platform:
            np.testing.assert_array_equal(out, ref, err_msg=label)
        else:
            diff = np.abs(out.astype(np.int64) - ref.astype(np.int64))
            assert np.max(diff) <= 1, label
        return
    np.testing.assert_array_equal(np.isnan(out), np.isnan(ref), err_msg=label)
    fin = np.isfinite(ref)
    np.testing.assert_array_equal(np.isfinite(out), fin, err_msg=label)
    if same_platform:
        np.testing.assert_array_max_ulp(out[fin], ref[fin], maxulp=1)
    else:
        np.testing.assert_allclose(out[fin], ref[fin], rtol=1e-5, atol=1e-5,
                                   err_msg=label)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("case", CASES, ids=[c[1] for c in CASES])
def test_golden_outputs(case, variant, golden_inputs, tmp_path):
    mod, case_id, func, kwargs = case
    arrays, meta = _golden(mod)
    same_platform = meta["platform"] == _platform_tag()
    in_img, call_kwargs = golden_inputs[variant]
    for bs in BLOCK_SIZES:
        key = f"{case_id}|{variant}|{bs}"
        ref = arrays.get(key, arrays[f"{case_id}|{variant}|256"])
        ref_meta = meta["outputs"][key]
        stats = bs == 256 and variant == "f32nd"
        out, out_meta = _run_case(func, kwargs, in_img, call_kwargs,
                                  str(tmp_path / f"out_{bs}.tif"), bs,
                                  calc_stats=stats)
        _assert_matches(out, ref, same_platform, key)
        assert out_meta["nodata"] == ref_meta["nodata"], key
        assert out_meta["names"] == ref_meta["names"], key
        assert out_meta["dtype"] == ref_meta["dtype"], key
        if stats:
            assert out_meta["overviews"] == ref_meta["overviews"], key
            for out_tags, ref_tags in zip(out_meta["tags"], ref_meta["tags"]):
                assert set(out_tags) == set(ref_tags), key
                for tag_key, ref_val in ref_tags.items():
                    out_val = out_tags[tag_key]
                    if same_platform or not tag_key.startswith("STATISTICS_"):
                        assert out_val == ref_val, (key, tag_key)
                    else:
                        assert float(out_val) == pytest.approx(
                            float(ref_val), rel=1e-5, abs=1e-5
                        ), (key, tag_key)


def test_golden_files_are_small():
    total = sum(
        os.path.getsize(os.path.join(GOLDEN_DIR, name))
        for name in os.listdir(GOLDEN_DIR)
        if name.startswith("imagefilter_") and name.endswith(".npz")
    )
    assert total < 2 * 1024 * 1024


if __name__ == "__main__":
    if "--regen" in sys.argv:
        regenerate()
        print(f"Golden files written to {GOLDEN_DIR}")
    else:
        print(__doc__)
