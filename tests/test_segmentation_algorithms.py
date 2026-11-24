"""
Comprehensive unit and integration tests for :mod:`rsgispy.segmentation`
algorithms, tiling/seam stitching, multi-core execution, optional ``numba``
parity, post-processing, and downstream GEOBIA integration.
"""

import inspect
import os

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

import rsgispy
import rsgispy.segmentation as rseg


def _write_multiband_raster(
    path: str,
    arr: np.ndarray,
    nodata: float = None,
    origin_x: float = 0.0,
    origin_y: float = 1000.0,
    pxl_size: float = 10.0,
    crs: str = "EPSG:32630",
) -> None:
    """Helper to write a 2D or 3D GeoTIFF raster with optional nodata."""
    if arr.ndim == 2:
        count = 1
        height, width = arr.shape
        write_arr = arr[np.newaxis, :, :]
    else:
        count, height, width = arr.shape
        write_arr = arr

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=count,
        dtype=str(write_arr.dtype),
        crs=crs,
        transform=from_origin(origin_x, origin_y, pxl_size, pxl_size),
        nodata=nodata,
    ) as dst:
        dst.write(write_arr)


def _make_synthetic_scene(
    height: int = 40,
    width: int = 40,
    nodata_val: float = -9999.0,
    seed: int = 7,
) -> np.ndarray:
    """
    Build a 4-band synthetic remote sensing scene with 4 distinct land-cover
    quadrants, subtle noise, and a masked nodata corner block.
    """
    rng = np.random.default_rng(seed)
    arr = np.zeros((4, height, width), dtype=np.float32)

    mid_r = height // 2
    mid_c = width // 2

    # Quadrant 1 (Water): high blue, very low NIR
    arr[0, :mid_r, :mid_c] = 1200.0
    arr[1, :mid_r, :mid_c] = 900.0
    arr[2, :mid_r, :mid_c] = 600.0
    arr[3, :mid_r, :mid_c] = 200.0

    # Quadrant 2 (Forest): low red, high NIR
    arr[0, :mid_r, mid_c:] = 350.0
    arr[1, :mid_r, mid_c:] = 650.0
    arr[2, :mid_r, mid_c:] = 300.0
    arr[3, :mid_r, mid_c:] = 3800.0

    # Quadrant 3 (Soil): moderate red & NIR
    arr[0, mid_r:, :mid_c] = 900.0
    arr[1, mid_r:, :mid_c] = 1400.0
    arr[2, mid_r:, :mid_c] = 2100.0
    arr[3, mid_r:, :mid_c] = 2400.0

    # Quadrant 4 (Urban): bright across all bands
    arr[0, mid_r:, mid_c:] = 2800.0
    arr[1, mid_r:, mid_c:] = 3100.0
    arr[2, mid_r:, mid_c:] = 3300.0
    arr[3, mid_r:, mid_c:] = 3600.0

    arr += rng.normal(0.0, 15.0, size=arr.shape).astype(np.float32)

    # Corner nodata patch
    arr[:, :4, :4] = nodata_val
    return arr


def test_raster_function_signatures_and_exports():
    """Verify required positional/keyword-only rsgispy conventions on all new raster functions."""
    assert isinstance(rseg.is_numba_available(), bool)
    raster_funcs = [
        rseg.shepherd_segmentation,
        rseg.region_merge_segmentation,
        rseg.slic_segmentation,
        rseg.graph_segmentation,
        rseg.watershed_segmentation,
        rseg.eliminate_small_segments,
        rseg.relabel_segments,
    ]
    expected_kw_defaults = {
        "gdalformat": "GTIFF",
        "datatype": None,
        "no_data_val": None,
        "calc_stats": True,
        "use_pbar": True,
    }
    for fn in raster_funcs:
        sig = inspect.signature(fn)
        params = list(sig.parameters.values())
        required_pos = [
            p.name
            for p in params
            if p.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
            and p.default is inspect.Parameter.empty
        ]
        assert required_pos == ["input_img", "output_img"], fn.__name__
        for kw_name, kw_default in expected_kw_defaults.items():
            assert kw_name in sig.parameters, (fn.__name__, kw_name)
            p = sig.parameters[kw_name]
            assert p.kind == inspect.Parameter.KEYWORD_ONLY, (
                fn.__name__,
                kw_name,
            )
            assert p.default == kw_default, (fn.__name__, kw_name)


@pytest.mark.parametrize(
    "algo_fn,kwargs",
    [
        (
            rseg.shepherd_segmentation_array,
            {"n_clusters": 4, "min_pixels": 16, "sampling": 1},
        ),
        (
            rseg.region_merge_segmentation_array,
            {"scale": 15.0, "compactness": 0.5, "shape_weight": 0.1, "min_pixels": 16},
        ),
        (
            rseg.slic_segmentation_array,
            {"n_segments": 8, "compactness": 10.0, "min_pixels": 16},
        ),
        (
            rseg.graph_segmentation_array,
            {"scale": 60.0, "sigma": 0.5, "min_pixels": 16},
        ),
        (
            rseg.watershed_segmentation_array,
            {"min_distance": 8, "gradient_sigma": 0.8, "min_pixels": 16},
        ),
    ],
)
def test_all_algorithms_array_nodata_and_numba_parity(algo_fn, kwargs):
    """
    Verify every segmentation array entrypoint separates land-cover patches,
    preserves nodata=0, produces contiguous labels, and matches between
    numba=True and numba=False.
    """
    scene = _make_synthetic_scene(height=36, width=36, nodata_val=-9999.0)
    out_numba = algo_fn(
        scene, no_data_val=-9999.0, use_numba=True, **kwargs
    )
    out_py = algo_fn(
        scene, no_data_val=-9999.0, use_numba=False, **kwargs
    )

    assert out_numba.shape == (36, 36)
    assert out_numba.dtype == np.uint32
    np.testing.assert_array_equal(out_numba, out_py)

    # Nodata pixels strictly 0; valid pixels strictly > 0
    assert np.all(out_numba[:4, :4] == 0)
    valid_lbls = out_numba[4:, :]
    assert np.all(valid_lbls > 0)

    # Contiguous 1..N labels
    uniq = np.unique(out_numba[out_numba > 0])
    assert uniq[0] == 1
    assert uniq[-1] == uniq.size
    assert uniq.size >= 4

    # All-nodata edge case returns all zeros cleanly
    all_nd = np.full((2, 12, 12), -9999.0, dtype=np.float32)
    out_nd = algo_fn(all_nd, no_data_val=-9999.0, use_numba=True, **kwargs)
    np.testing.assert_array_equal(out_nd, np.zeros((12, 12), dtype=np.uint32))


def test_watershed_explicit_markers_array_and_raster(tmp_path):
    """Verify seeded watershed respects user markers in both array and raster modes."""
    arr = np.zeros((2, 24, 24), dtype=np.float32)
    arr[:, :, :12] = 10.0
    arr[:, :, 12:] = 200.0
    markers = np.zeros((24, 24), dtype=np.int32)
    markers[12, 4] = 1
    markers[12, 18] = 2

    seg_arr = rseg.watershed_segmentation_array(
        arr, markers=markers, gradient_sigma=0.0, min_pixels=1
    )
    assert int(seg_arr.max()) == 2
    assert np.all(seg_arr[:, :12] == seg_arr[12, 4])
    assert np.all(seg_arr[:, 12:] == seg_arr[12, 18])
    assert seg_arr[12, 4] != seg_arr[12, 18]

    in_img = str(tmp_path / "ws_in.tif")
    m_img = str(tmp_path / "ws_markers.tif")
    out_img = str(tmp_path / "ws_out.tif")
    _write_multiband_raster(in_img, arr)
    _write_multiband_raster(m_img, markers, nodata=0)

    rseg.watershed_segmentation(
        in_img,
        out_img,
        markers_img=m_img,
        gradient_sigma=0.0,
        min_pixels=1,
        tile_size=12,
        overlap=4,
        stitch_seams=True,
        calc_stats=False,
        use_pbar=False,
    )
    with rasterio.open(out_img) as src:
        np.testing.assert_array_equal(src.read(1), seg_arr)


def test_2d_single_band_input():
    """Verify single-band 2D (H, W) array inputs work across algorithms."""
    img = np.zeros((20, 20), dtype=np.float32)
    img[:10, :] = 10.0
    img[10:, :] = 100.0
    for fn, extra in [
        (rseg.shepherd_segmentation_array, {"n_clusters": 2, "min_pixels": 10, "sampling": 1}),
        (rseg.region_merge_segmentation_array, {"scale": 10.0, "min_pixels": 10}),
        (rseg.slic_segmentation_array, {"n_segments": 4, "min_pixels": 10}),
        (rseg.graph_segmentation_array, {"scale": 50.0, "sigma": 0.0, "min_pixels": 10}),
        (rseg.watershed_segmentation_array, {"min_distance": 6, "min_pixels": 10}),
    ]:
        lbl = fn(img, **extra)
        assert lbl.shape == (20, 20)
        assert int(lbl.max()) >= 2


@pytest.mark.parametrize(
    "raster_fn,kwargs",
    [
        (
            rseg.shepherd_segmentation,
            {"n_clusters": 4, "min_pixels": 20, "sampling": 1},
        ),
        (
            rseg.region_merge_segmentation,
            {"scale": 15.0, "compactness": 0.5, "shape_weight": 0.1, "min_pixels": 20},
        ),
        (
            rseg.slic_segmentation,
            {"segment_size": 10.0, "compactness": 10.0, "min_pixels": 10},
        ),
        (
            rseg.graph_segmentation,
            {"scale": 60.0, "sigma": 0.5, "min_pixels": 20},
        ),
        (
            rseg.watershed_segmentation,
            {"min_distance": 8, "gradient_sigma": 0.8, "min_pixels": 20},
        ),
    ],
)
def test_tiled_raster_parallel_worker_parity_and_seam_stitching(
    tmp_path, raster_fn, kwargs
):
    """
    Verify tiled execution with overlap produces identical results between
    n_jobs=1 and n_jobs=2, respects nodata, and stitches uniform regions
    spanning tile seams cleanly when stitch_seams=True.
    """
    # Create a uniform 2-class vertical split spanning tile boundaries at row=16, col=16
    scene = np.zeros((3, 32, 32), dtype=np.float32)
    scene[:, :, :16] = 100.0
    scene[:, :, 16:] = 5000.0
    # Add nodata patch in bottom-right corner
    scene[:, 28:, 28:] = -9999.0

    in_img = str(tmp_path / "scene.tif")
    _write_multiband_raster(in_img, scene, nodata=-9999.0)

    out_w1 = str(tmp_path / "w1.tif")
    out_w2 = str(tmp_path / "w2.tif")
    out_nostitch = str(tmp_path / "nostitch.tif")

    raster_fn(
        in_img,
        out_w1,
        tile_size=16,
        overlap=8,
        n_jobs=1,
        stitch_seams=True,
        calc_stats=False,
        use_pbar=False,
        **kwargs,
    )
    raster_fn(
        in_img,
        out_w2,
        tile_size=16,
        overlap=8,
        n_jobs=2,
        stitch_seams=True,
        calc_stats=False,
        use_pbar=False,
        **kwargs,
    )
    raster_fn(
        in_img,
        out_nostitch,
        tile_size=16,
        overlap=8,
        n_jobs=1,
        stitch_seams=False,
        calc_stats=False,
        use_pbar=False,
        **kwargs,
    )

    with (
        rasterio.open(out_w1) as s1,
        rasterio.open(out_w2) as s2,
        rasterio.open(out_nostitch) as sns,
    ):
        a1 = s1.read(1)
        a2 = s2.read(1)
        ans = sns.read(1)
        assert s1.dtypes == ("uint32",)
        assert s1.nodata == 0
        np.testing.assert_array_equal(a1, a2)
        assert np.all(a1[28:, 28:] == 0)
        assert np.all(a1[:, :16] > 0)
        # Stitching across tile boundaries should never increase segment count
        assert int(a1.max()) <= int(ans.max())


def test_shepherd_band_subset_and_stats_generation(tmp_path):
    """Verify band subsetting and statistics/pyramids population."""
    scene = _make_synthetic_scene(height=28, width=28, nodata_val=-9999.0)
    in_img = str(tmp_path / "mb.tif")
    out_img = str(tmp_path / "segs.tif")
    _write_multiband_raster(in_img, scene, nodata=-9999.0)

    rseg.shepherd_segmentation(
        in_img,
        out_img,
        n_clusters=4,
        min_pixels=12,
        sampling=1,
        bands=[3, 4],
        calc_stats=True,
        use_pbar=False,
    )
    assert os.path.exists(out_img)
    with rasterio.open(out_img) as src:
        arr = src.read(1)
        assert int(arr.max()) >= 4
        assert np.all(arr[:4, :4] == 0)


def test_relabel_and_eliminate_small_segments(tmp_path):
    """Verify relabel_segments and eliminate_small_segments (array & raster)."""
    raw = np.array(
        [
            [0, 10, 10, 10, 20],
            [0, 10, 10, 20, 20],
            [30, 30, 30, 20, 20],
            [30, 99, 30, 20, 20],
        ],
        dtype=np.int32,
    )
    relab = rseg.relabel_segments_array(raw, no_data_val=0)
    assert relab.dtype == np.uint32
    np.testing.assert_array_equal(
        np.unique(relab), np.array([0, 1, 2, 3, 4], dtype=np.uint32)
    )

    # Spectral data matches pixel (3,1) to region 30 (value ~1.0) rather than 10 (value ~100)
    spec = np.full((1, 4, 5), 1.0, dtype=np.float32)
    spec[0, :2, 1:4] = 100.0
    spec[0, 2:, 3:] = 50.0
    cleaned_spec = rseg.eliminate_small_segments_array(
        raw, data=spec, min_pixels=4, no_data_val=0, use_numba=True
    )
    cleaned_py = rseg.eliminate_small_segments_array(
        raw, data=spec, min_pixels=4, no_data_val=0, use_numba=False
    )
    np.testing.assert_array_equal(cleaned_spec, cleaned_py)
    # Pixel (3,1) which had label 99 merged into surrounding label 30
    assert cleaned_spec[3, 1] == cleaned_spec[3, 0]
    assert np.all(cleaned_spec[:, 0][:2] == 0)

    # Without spectral data, small region merges into largest neighbour
    cleaned_nospec = rseg.eliminate_small_segments_array(
        raw, data=None, min_pixels=4
    )
    assert cleaned_nospec[3, 1] == cleaned_nospec[3, 0]

    # File-backed relabel_segments and eliminate_small_segments
    raw_img = str(tmp_path / "raw_lbl.tif")
    spec_img = str(tmp_path / "spec.tif")
    relab_img = str(tmp_path / "relab.tif")
    clean_img = str(tmp_path / "clean.tif")
    _write_multiband_raster(raw_img, raw, nodata=0)
    _write_multiband_raster(spec_img, spec)

    rseg.relabel_segments(
        raw_img, relab_img, calc_stats=False, use_pbar=False
    )
    with rasterio.open(relab_img) as src:
        np.testing.assert_array_equal(src.read(1), relab)

    rseg.eliminate_small_segments(
        raw_img,
        clean_img,
        ref_img=spec_img,
        min_pixels=4,
        calc_stats=False,
        use_pbar=False,
    )
    with rasterio.open(clean_img) as src:
        np.testing.assert_array_equal(src.read(1), cleaned_spec)


def test_polygonise_and_extract_segment_stats(tmp_path):
    """Verify downstream polygonisation and zonal statistics on segmented objects."""
    scene = _make_synthetic_scene(height=24, width=24, nodata_val=-9999.0)
    feat_img = str(tmp_path / "features.tif")
    seg_img = str(tmp_path / "segs.tif")
    vec_out = str(tmp_path / "seg_stats.gpkg")
    _write_multiband_raster(feat_img, scene, nodata=-9999.0)

    rseg.shepherd_segmentation(
        feat_img,
        seg_img,
        n_clusters=4,
        min_pixels=20,
        sampling=1,
        rnd_seed=42,
        calc_stats=False,
        use_pbar=False,
    )

    poly_gdf = rseg.polygonise_segments(seg_img, out_vec_file=None, no_data_val=0)
    assert isinstance(poly_gdf, gpd.GeoDataFrame)
    assert len(poly_gdf) >= 4
    assert "seg_id" in poly_gdf.columns
    assert 0 not in set(poly_gdf["seg_id"].tolist())

    stats_gdf = rseg.extract_segment_stats(
        segments_img=seg_img,
        feature_img=feat_img,
        out_vec_file=None,
        bands=[1, 4],
        use_pbar=False,
    )
    assert isinstance(stats_gdf, gpd.GeoDataFrame)
    for col in ("seg_id", "b1_mean", "b1_std", "b4_mean", "b4_cnt"):
        assert col in stats_gdf.columns
    assert np.all(stats_gdf["b1_cnt"] > 0)

    rseg.extract_segment_stats(
        segments_img=seg_img,
        feature_img=feat_img,
        out_vec_file=vec_out,
        out_vec_lyr="segments",
        bands=[1, 4],
        use_pbar=False,
    )
    assert os.path.exists(vec_out)


def test_polygonise_uint32_legacy_rasterio_compat(tmp_path, monkeypatch):
    """Verify uint32 label rasters polygonise cleanly even under rasterio < 1.4 dtype rules."""
    orig_shapes = rasterio.features.shapes
    legacy_dtypes = {"int16", "int32", "uint8", "uint16", "float32", "float64", "int8"}

    def strict_shapes(source, mask=None, connectivity=4, transform=rasterio.Affine.identity()):
        if np.asarray(source).dtype.name not in legacy_dtypes:
            raise ValueError(
                "image dtype must be one of: int16, int32, uint8, uint16, float32, float64, int8"
            )
        return orig_shapes(source, mask=mask, connectivity=connectivity, transform=transform)

    monkeypatch.setattr(rasterio.features, "shapes", strict_shapes)

    lbl_small = np.array(
        [[0, 1, 1], [2, 2, 0], [2, 3, 3]], dtype=np.uint32
    )
    small_img = str(tmp_path / "u32_small.tif")
    _write_multiband_raster(small_img, lbl_small, nodata=0)
    gdf_small = rseg.polygonise_segments(small_img, out_vec_file=None)
    assert set(gdf_small["seg_id"].tolist()) == {1, 2, 3}

    # Also test uint32 values exceeding int32 max (exercises lossless ID remapping fallback)
    big_a = np.uint32(3_000_000_001)
    big_b = np.uint32(3_000_000_002)
    lbl_big = np.array(
        [[0, big_a, big_a], [big_b, big_b, 0], [big_b, big_b, 0]], dtype=np.uint32
    )
    big_img = str(tmp_path / "u32_big.tif")
    _write_multiband_raster(big_img, lbl_big, nodata=0)
    gdf_big = rseg.polygonise_segments(big_img, out_vec_file=None)
    assert set(gdf_big["seg_id"].tolist()) == {int(big_a), int(big_b)}


def test_validation_and_dtype_overflow(tmp_path):
    """Verify input validation exceptions and integer label overflow protection."""
    arr = np.arange(36 * 36, dtype=np.float32).reshape(1, 36, 36)
    in_img = str(tmp_path / "grad.tif")
    _write_multiband_raster(in_img, arr)

    with pytest.raises(rsgispy.RSGISPyException):
        rseg.shepherd_segmentation(
            str(tmp_path / "missing.tif"), str(tmp_path / "o.tif")
        )
    with pytest.raises(rsgispy.RSGISPyException):
        rseg.shepherd_segmentation_array(arr, n_clusters=0)
    with pytest.raises(rsgispy.RSGISPyException):
        rseg.region_merge_segmentation_array(arr, scale=-1.0)
    with pytest.raises(rsgispy.RSGISPyException):
        rseg.region_merge_segmentation_array(arr, compactness=1.5)
    with pytest.raises(rsgispy.RSGISPyException):
        rseg.slic_segmentation_array(arr, n_segments=0)
    with pytest.raises(rsgispy.RSGISPyException):
        rseg.graph_segmentation_array(arr, sigma=-0.5)
    with pytest.raises(rsgispy.RSGISPyException):
        rseg.watershed_segmentation_array(
            arr, markers=np.zeros((5, 5), dtype=np.int32)
        )
    with pytest.raises(rsgispy.RSGISPyException):
        rseg.slic_segmentation_array(arr, n_jobs=0)

    # Non-integer output datatype rejected
    with pytest.raises(rsgispy.RSGISPyException):
        rseg.slic_segmentation(
            in_img,
            str(tmp_path / "bad_dtype.tif"),
            datatype=rsgispy.TYPE_32FLOAT,
            use_pbar=False,
        )

    # Overflow uint8 when > 255 distinct segments produced
    with pytest.raises(rsgispy.RSGISPyException):
        rseg.slic_segmentation(
            in_img,
            str(tmp_path / "overflow_u8.tif"),
            segment_size=2.0,
            min_pixels=1,
            enforce_connectivity=False,
            datatype=rsgispy.TYPE_8UINT,
            use_pbar=False,
        )

    # Re-check with dense unique 1-pixel clumps (400 segments > 255)
    dense = np.arange(1, 401, dtype=np.int32).reshape(20, 20)
    dense_img = str(tmp_path / "dense_lbl.tif")
    _write_multiband_raster(dense_img, dense)
    with pytest.raises(rsgispy.RSGISPyException):
        rseg.relabel_segments(
            dense_img,
            str(tmp_path / "dense_u8.tif"),
            datatype=rsgispy.TYPE_8UINT,
            use_pbar=False,
        )


def test_true_no_numba_subprocess():
    """Verify segmentation algorithms and warning semantics when numba is absent."""
    import subprocess
    import sys

    script = """
import sys, warnings
sys.modules["numba"] = None
import numpy as np
import rsgispy.segmentation as rseg

assert rseg.is_numba_available() is False

arr = np.zeros((3, 16, 16), dtype=np.float32)
arr[:, :8, :8] = 10.0
arr[:, :8, 8:] = 100.0
arr[:, 8:, :8] = 200.0
arr[:, 8:, 8:] = 300.0

calls = [
    (
        "shepherd",
        lambda nb: rseg.shepherd_segmentation_array(
            arr, n_clusters=4, min_pixels=4, sampling=1, rnd_seed=42, use_numba=nb
        ),
    ),
    (
        "region_merge",
        lambda nb: rseg.region_merge_segmentation_array(
            arr, scale=20.0, min_pixels=4, use_numba=nb
        ),
    ),
    (
        "slic",
        lambda nb: rseg.slic_segmentation_array(
            arr, segment_size=8.0, min_pixels=4, use_numba=nb
        ),
    ),
    (
        "graph",
        lambda nb: rseg.graph_segmentation_array(
            arr, scale=50.0, min_pixels=4, use_numba=nb
        ),
    ),
    (
        "watershed",
        lambda nb: rseg.watershed_segmentation_array(
            arr, min_distance=4, min_pixels=4, use_numba=nb
        ),
    ),
]

for name, fn in calls:
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out_warn = fn(True)
    rt = [w for w in rec if issubclass(w.category, RuntimeWarning)]
    assert len(rt) >= 1, f"Expected RuntimeWarning for {name} with use_numba=True"
    assert int(out_warn.max()) >= 1

    with warnings.catch_warnings(record=True) as rec_silent:
        warnings.simplefilter("always")
        out_silent = fn(False)
    rt_silent = [w for w in rec_silent if issubclass(w.category, RuntimeWarning)]
    assert len(rt_silent) == 0, f"Unexpected RuntimeWarning for {name} with use_numba=False"
    np.testing.assert_array_equal(out_warn, out_silent)

relab = rseg.relabel_segments_array(out_silent)
assert int(relab.max()) >= 1
relab_small = relab.copy()
relab_small[0, 0] = int(relab.max()) + 1
with warnings.catch_warnings(record=True) as rec_elim:
    warnings.simplefilter("always")
    cleaned = rseg.eliminate_small_segments_array(
        relab_small, data=arr, min_pixels=8, use_numba=True
    )
assert any(issubclass(w.category, RuntimeWarning) for w in rec_elim)
assert cleaned.shape == relab.shape
"""
    res = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, f"Subprocess failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
