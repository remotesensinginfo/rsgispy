"""
Tests for :mod:`rsgispy.changedetect.imgchngmthds`,
:mod:`rsgispy.changedetect.pxloutlierchng`, and
:func:`rsgispy.tools.plotting.plot_histogram_threshold`.
"""

import inspect

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

import rsgispy
import rsgispy.changedetect
import rsgispy.changedetect.imgchngmthds as icm
import rsgispy.changedetect.pxloutlierchng as poc
import rsgispy.tools.plotting as rplot


def _write_raster(
    path,
    data,
    origin=(0.0, 100.0),
    res=10.0,
    crs="EPSG:32630",
    nodata=None,
):
    """Write a 3D ``(bands, height, width)`` GeoTIFF raster."""
    count, height, width = data.shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=str(data.dtype),
        crs=crs,
        transform=from_origin(origin[0], origin[1], res, res),
        nodata=nodata,
    ) as dst:
        dst.write(data)


class _DummyPyODDetector:
    """Minimal duck-typed PyOD estimator for dependency-free unit testing."""

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.fitted_count = 0

    def fit(self, data: np.ndarray) -> "_DummyPyODDetector":
        self.fitted_count = int(data.shape[0])
        return self

    def predict(self, data: np.ndarray) -> np.ndarray:
        return (data[:, 0] > self.threshold).astype(np.int64)

    def predict_proba(self, data: np.ndarray, method: str = "unify") -> np.ndarray:
        _ = method
        p1 = np.clip(data[:, 0], 0.0, 1.0)
        return np.stack([1.0 - p1, p1], axis=1)


def test_default_gdalformat_is_gtiff():
    """Verify all changedetect functions default gdalformat to 'GTIFF'."""
    funcs = [
        icm.image_difference,
        icm.image_ratio,
        icm.change_vector_analysis,
        icm.slow_feature_analysis,
        icm.multivariate_alteration_detection,
        poc.find_class_pyod_outliers,
        poc.find_class_otsu_outliers,
        poc.find_class_li_outliers,
        poc.find_class_kurt_skew_outliers,
    ]

    for fn in funcs:
        sig = inspect.signature(fn)
        assert sig.parameters["gdalformat"].default == "GTIFF"


def test_plot_histogram_threshold(tmp_path):
    """Verify plot_histogram_threshold writes valid PNG outputs (incl. constant data)."""
    out_png = tmp_path / "hist.png"
    rng = np.random.default_rng(0)
    vals = np.concatenate([rng.normal(2.0, 0.3, 100), rng.normal(6.0, 0.3, 100)])
    rplot.plot_histogram_threshold(
        vals, str(out_png), threshold=4.0, title_str="Test Histogram"
    )
    assert out_png.exists() and out_png.stat().st_size > 0

    const_png = tmp_path / "const.png"
    rplot.plot_histogram_threshold(np.ones(20), str(const_png), threshold=1.0)
    assert const_png.exists() and const_png.stat().st_size > 0


def test_image_difference_and_ratio_multiband(tmp_path):
    """Verify image_difference and image_ratio across multi-band imagery with zeros."""
    base_path = tmp_path / "base.tif"
    chng_path = tmp_path / "chng.tif"
    diff_path = tmp_path / "diff.tif"
    ratio_path = tmp_path / "ratio.tif"

    base_arr = np.array(
        [
            [[10.0, 20.0], [30.0, 0.0]],
            [[4.0, 8.0], [12.0, 0.0]],
        ],
        dtype=np.float32,
    )
    chng_arr = np.array(
        [
            [[5.0, 0.0], [10.0, 0.0]],
            [[2.0, 4.0], [3.0, 0.0]],
        ],
        dtype=np.float32,
    )
    _write_raster(base_path, base_arr)
    _write_raster(chng_path, chng_arr)

    icm.image_difference(str(base_path), str(chng_path), str(diff_path), use_pbar=False)
    with rasterio.open(diff_path) as src:
        np.testing.assert_allclose(src.read(), base_arr - chng_arr, atol=1e-6)

    icm.image_ratio(
        str(base_path),
        str(chng_path),
        str(ratio_path),
        img_base_bands=[2, 1],
        img_chng_bands=[2, 1],
        use_pbar=False,
    )
    with rasterio.open(ratio_path) as src:
        r = src.read()
    assert np.isclose(r[0, 0, 0], 2.0)  # band 2: 4 / 2
    assert np.isclose(r[1, 0, 0], 2.0)  # band 1: 10 / 5
    assert np.isclose(r[1, 0, 1], 0.0)  # zero divisor handled cleanly


def test_change_vector_analysis(tmp_path):
    """Verify CVA identifies a distinct changed block."""
    base_path = tmp_path / "cva_base.tif"
    chng_path = tmp_path / "cva_chng.tif"
    out_bin = tmp_path / "cva_bin.tif"
    out_dist = tmp_path / "cva_dist.tif"

    rng = np.random.default_rng(123)
    base_arr = rng.uniform(0.1, 0.15, size=(3, 12, 12)).astype(np.float32)
    chng_arr = base_arr.copy()
    chng_arr[:, 0:4, 0:4] += 0.6  # clear changed patch
    _write_raster(base_path, base_arr)
    _write_raster(chng_path, chng_arr)

    icm.change_vector_analysis(
        str(base_path),
        str(chng_path),
        str(out_bin),
        str(out_dist),
        use_kmeans=False,
        use_pbar=False,
    )
    with rasterio.open(out_bin) as src:
        b = src.read(1)
    assert np.all(b[0:4, 0:4] == 1)
    assert np.all(b[6:, 6:] == 0)

    if icm.HAVE_SKLEARN:
        out_bin_km = tmp_path / "cva_bin_km.tif"
        out_dist_km = tmp_path / "cva_dist_km.tif"
        icm.change_vector_analysis(
            str(base_path),
            str(chng_path),
            str(out_bin_km),
            str(out_dist_km),
            use_kmeans=True,
            use_pbar=False,
        )
        with rasterio.open(out_bin_km) as src:
            b_km = src.read(1)
        assert np.all(b_km[0:4, 0:4] == 1)
        assert np.all(b_km[6:, 6:] == 0)


@pytest.mark.skipif(not icm.HAVE_SCIPY, reason="scipy not installed")
def test_sfa_and_mad_change_detection(tmp_path):
    """Verify ISFA and IR-MAD run end-to-end and detect synthetic change."""
    base_path = tmp_path / "sfa_base.tif"
    chng_path = tmp_path / "sfa_chng.tif"
    sfa_bin = tmp_path / "sfa_bin.tif"
    sfa_dist = tmp_path / "sfa_dist.tif"
    mad_bin = tmp_path / "mad_bin.tif"
    mad_dist = tmp_path / "mad_dist.tif"
    mad_vars = tmp_path / "mad_vars.tif"

    rng = np.random.default_rng(7)
    base_arr = rng.uniform(0.1, 0.3, size=(3, 12, 12)).astype(np.float32)
    chng_arr = (base_arr + rng.normal(0.0, 0.005, size=base_arr.shape)).astype(
        np.float32
    )
    chng_arr[0, 0:4, 0:4] += 0.8
    chng_arr[1, 0:4, 0:4] -= 0.05
    chng_arr[2, 0:4, 0:4] += 0.7

    _write_raster(base_path, base_arr)
    _write_raster(chng_path, chng_arr)

    icm.slow_feature_analysis(
        str(base_path),
        str(chng_path),
        str(sfa_bin),
        str(sfa_dist),
        sfa_max_iter=15,
        use_pbar=False,
    )
    with rasterio.open(sfa_bin) as src:
        sfa_b = src.read(1)
    assert np.mean(sfa_b[0:4, 0:4]) > 0.75

    icm.multivariate_alteration_detection(
        str(base_path),
        str(chng_path),
        str(mad_bin),
        str(mad_dist),
        str(mad_vars),
        mad_max_iter=15,
        use_kmeans=False,
        use_pbar=False,
    )
    with rasterio.open(mad_bin) as src:
        mad_b = src.read(1)
    with rasterio.open(mad_vars) as src:
        assert src.count == 3
    assert np.mean(mad_b[0:4, 0:4]) > 0.75


def test_pxloutlierchng_methods(tmp_path):
    """Verify Otsu, Li, PyOD, and Kurtosis/Skewness class outlier methods."""
    img_path = tmp_path / "vals.tif"
    msk_path = tmp_path / "msk.tif"

    rng = np.random.default_rng(99)
    vals = rng.normal(0.75, 0.05, size=(1, 16, 16)).astype(np.float32)
    vals[0, 0:4, 0:4] = rng.normal(0.20, 0.03, size=(4, 4)).astype(np.float32)
    _write_raster(img_path, vals, nodata=-9999.0)

    msk = np.ones((1, 16, 16), dtype=np.uint8)
    msk[0, 12:, 12:] = 0  # outside mask region
    _write_raster(msk_path, msk)

    out_otsu = tmp_path / "otsu.tif"
    plot_otsu = tmp_path / "otsu.png"
    thres_otsu = poc.find_class_otsu_outliers(
        str(img_path),
        str(msk_path),
        str(out_otsu),
        low_thres=True,
        plot_thres_file=str(plot_otsu),
        use_pbar=False,
    )
    assert 0.25 < thres_otsu < 0.70
    assert plot_otsu.exists()
    with rasterio.open(out_otsu) as src:
        o_arr = src.read(1)
        cmap = src.colormap(1)
    assert np.all(o_arr[0:4, 0:4] == 2)
    assert np.all(o_arr[5:11, 5:11] == 1)
    assert np.all(o_arr[12:, 12:] == 0)
    assert cmap[1][:3] == (0, 255, 0)
    assert cmap[2][:3] == (255, 0, 0)

    out_li = tmp_path / "li.tif"
    thres_li = poc.find_class_li_outliers(
        str(img_path),
        str(msk_path),
        str(out_li),
        low_thres=True,
        use_pbar=False,
    )
    assert 0.25 < thres_li < 0.70

    out_pyod = tmp_path / "pyod.tif"
    out_scores = tmp_path / "pyod_scores.tif"
    det = _DummyPyODDetector(threshold=0.5)
    poc.find_class_pyod_outliers(
        det,
        str(img_path),
        str(msk_path),
        str(out_pyod),
        out_scores_img=str(out_scores),
        use_pbar=False,
    )
    assert det.fitted_count == 256 - 16
    with rasterio.open(out_pyod) as src:
        p_arr = src.read(1)
    assert np.all(p_arr[0:4, 0:4] == 1)  # <= 0.5 -> predict=0 -> class 1
    assert np.all(p_arr[5:11, 5:11] == 2)  # > 0.5 -> predict=1 -> class 2
    assert np.all(p_arr[12:, 12:] == 0)

    if poc.HAVE_SCIPY:
        out_ks = tmp_path / "ks.tif"
        thres_ks = poc.find_class_kurt_skew_outliers(
            str(img_path),
            str(msk_path),
            str(out_ks),
            vld_min=0.1,
            vld_max=0.65,
            init_thres=0.4,
            low_thres=True,
            use_pbar=False,
        )
        assert 0.1 <= thres_ks <= 0.65


def test_optional_scipy_missing_changedetect(monkeypatch):
    """Verify changedetect optional functions remain exposed and raise friendly errors."""
    import importlib
    import sys

    import rsgispy.tools.thresholding as thresh_mod

    for mod_name in (
        "scipy",
        "scipy.optimize",
        "scipy.stats",
        "sklearn",
        "sklearn.cluster",
    ):
        monkeypatch.setitem(sys.modules, mod_name, None)

    monkeypatch.setattr(rsgispy.tools, "thresholding", thresh_mod)
    monkeypatch.setattr(rsgispy.changedetect, "imgchngmthds", icm)
    monkeypatch.setattr(rsgispy.changedetect, "pxloutlierchng", poc)
    monkeypatch.delitem(sys.modules, "rsgispy.tools.thresholding", raising=False)
    monkeypatch.delitem(sys.modules, "rsgispy.changedetect.imgchngmthds", raising=False)
    monkeypatch.delitem(sys.modules, "rsgispy.changedetect.pxloutlierchng", raising=False)

    icm_fresh = importlib.import_module("rsgispy.changedetect.imgchngmthds")
    poc_fresh = importlib.import_module("rsgispy.changedetect.pxloutlierchng")

    assert not icm_fresh.HAVE_SCIPY
    assert not icm_fresh.HAVE_SKLEARN
    for fn_name in ("slow_feature_analysis", "multivariate_alteration_detection"):
        assert hasattr(icm_fresh, fn_name)
        assert fn_name in icm_fresh.__all__
    with pytest.raises(rsgispy.RSGISPyException, match="pip install"):
        icm_fresh.slow_feature_analysis("a.tif", "b.tif", "out.tif", "dist.tif")
    with pytest.raises(rsgispy.RSGISPyException, match="pip install"):
        icm_fresh.multivariate_alteration_detection(
            "a.tif", "b.tif", "out.tif", "dist.tif", "mads.tif"
        )
    with pytest.raises(rsgispy.RSGISPyException, match="pip install"):
        icm_fresh.change_vector_analysis("a.tif", "b.tif", "bin.tif", "dist.tif", use_kmeans=True)

    assert not poc_fresh.HAVE_SCIPY
    assert hasattr(poc_fresh, "find_class_kurt_skew_outliers")
    assert "find_class_kurt_skew_outliers" in poc_fresh.__all__
    with pytest.raises(rsgispy.RSGISPyException, match="pip install"):
        poc_fresh.find_class_kurt_skew_outliers(
            "img.tif", "msk.tif", "out.tif", 0.0, 1.0, 0.5, True
        )
