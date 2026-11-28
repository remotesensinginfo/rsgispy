"""
Tests for :mod:`rsgispy.changedetect.imgchngmthds`,
:mod:`rsgispy.changedetect.pxloutlierchng`, and
:func:`rsgispy.tools.plotting.plot_histogram_threshold`.
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine, from_origin

import rsgispy
import rsgispy.changedetect
import rsgispy.changedetect.imgchngmthds as icm
import rsgispy.changedetect.pxloutlierchng as poc
import rsgispy.imageprocess as imgproc
import rsgispy.tools.plotting as rplot
from rsgispy.imageutils.colourtable import is_img_thematic

matplotlib.use("Agg")


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


def test_plot_histogram_threshold(tmp_path):
    """Verify plot_histogram_threshold writes valid PNG outputs (incl. constant data)."""
    out_png = tmp_path / "hist.png"
    rng = np.random.default_rng(0)
    vals = np.concatenate([rng.normal(2.0, 0.3, 100), rng.normal(6.0, 0.3, 100)])
    fig, ax = rplot.plot_histogram_threshold(
        vals, 4.0, title_str="Test Histogram", out_plot_file=str(out_png)
    )
    assert out_png.exists() and out_png.stat().st_size > 0
    assert ax.get_title() == "Test Histogram"
    assert plt.fignum_exists(fig.number)
    plt.close(fig)

    const_png = tmp_path / "const.png"
    fig, _ = rplot.plot_histogram_threshold(
        np.ones(20), threshold=1.0, out_plot_file=str(const_png)
    )
    plt.close(fig)
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
    # Pixel (0, 1) has a no-data (0) band on the change date and pixel (1, 1)
    # on both dates, so neither is compared (every band must be valid).
    exp_diff = base_arr - chng_arr
    exp_diff[:, 0, 1] = 0.0
    exp_diff[:, 1, 1] = 0.0
    with rasterio.open(diff_path) as src:
        np.testing.assert_allclose(src.read(), exp_diff, atol=1e-6)

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
    assert np.isclose(r[1, 0, 1], 0.0)  # a no-data band: pixel not compared


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


@pytest.mark.filterwarnings("ignore:.*did not converge:UserWarning")
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


def _make_one_date_hole_pair(tmp_path):
    """
    Two 3-band dates; date 2 has a no-data (0) hole that date 1 does not, and
    single bands are no data (date 1) or NaN (date 2) in two more holes.
    """
    rng = np.random.default_rng(11)
    base_arr = rng.uniform(1000.0, 1500.0, size=(3, 12, 12)).astype(np.float32)
    chng_arr = (base_arr + rng.normal(0.0, 10.0, size=base_arr.shape)).astype(
        np.float32
    )
    chng_arr[:, 0:4, 0:4] += 800.0  # real change
    chng_arr[:, 8:11, 8:11] = 0.0  # no data on date 2 only (e.g. cloud)
    # H10: a pixel is valid only when every band is valid on both dates.
    base_arr[0, 1:3, 8:11] = 0.0  # one band of date 1 is no data
    chng_arr[1, 5:7, 0:2] = np.nan  # one band of date 2 is NaN
    base_path = tmp_path / "hole_base.tif"
    chng_path = tmp_path / "hole_chng.tif"
    _write_raster(base_path, base_arr)
    _write_raster(chng_path, chng_arr)
    hole = np.zeros((12, 12), dtype=bool)
    hole[8:11, 8:11] = True
    hole[1:3, 8:11] = True
    hole[5:7, 0:2] = True
    return str(base_path), str(chng_path), hole


@pytest.mark.parametrize("func_name", ["image_difference", "image_ratio"])
def test_band_change_no_mask_requires_both_dates(tmp_path, func_name):
    """Without a valid mask, a pixel with data on one date only is no data."""
    base_path, chng_path, hole = _make_one_date_hole_pair(tmp_path)
    out_img = tmp_path / f"{func_name}.tif"
    getattr(icm, func_name)(
        base_path, chng_path, str(out_img), no_data_val=-9999.0, use_pbar=False
    )
    with rasterio.open(out_img) as src:
        out_arr = src.read()
        assert src.nodata == -9999.0
    assert np.all(out_arr[:, hole] == -9999.0)
    assert np.all(out_arr[:, ~hole] != -9999.0)


def test_cva_no_mask_requires_both_dates(tmp_path):
    """CVA without a valid mask leaves pixels with data on one date as no data."""
    base_path, chng_path, hole = _make_one_date_hole_pair(tmp_path)
    out_bin = tmp_path / "cva_bin.tif"
    out_dist = tmp_path / "cva_dist.tif"
    icm.change_vector_analysis(
        base_path, chng_path, str(out_bin), str(out_dist),
        no_data_val=-9999.0, use_pbar=False,
    )
    with rasterio.open(out_bin) as src:
        bin_arr = src.read(1)
    with rasterio.open(out_dist) as src:
        dist_arr = src.read(1)
    assert np.all(bin_arr[hole] == 255)
    assert np.all(dist_arr[hole] == -9999.0)
    assert np.all(bin_arr[0:4, 0:4] == 1)
    assert np.all(bin_arr[~hole] != 255)


@pytest.mark.filterwarnings("ignore:.*did not converge:UserWarning")
@pytest.mark.skipif(not icm.HAVE_SCIPY, reason="scipy not installed")
def test_sfa_mad_no_mask_requires_both_dates(tmp_path):
    """SFA and IR-MAD without a valid mask leave one-date pixels as no data."""
    base_path, chng_path, hole = _make_one_date_hole_pair(tmp_path)
    sfa_bin = tmp_path / "sfa_bin.tif"
    sfa_dist = tmp_path / "sfa_dist.tif"
    icm.slow_feature_analysis(
        base_path, chng_path, str(sfa_bin), str(sfa_dist),
        sfa_max_iter=10, no_data_val=-9999.0, use_pbar=False,
    )
    with rasterio.open(sfa_bin) as src:
        assert np.all(src.read(1)[hole] == 255)
    with rasterio.open(sfa_dist) as src:
        assert np.all(src.read(1)[hole] == -9999.0)

    mad_bin = tmp_path / "mad_bin.tif"
    mad_dist = tmp_path / "mad_dist.tif"
    mad_vars = tmp_path / "mad_vars.tif"
    icm.multivariate_alteration_detection(
        base_path, chng_path, str(mad_bin), str(mad_dist), str(mad_vars),
        mad_max_iter=10, use_kmeans=False, no_data_val=-9999.0, use_pbar=False,
    )
    with rasterio.open(mad_bin) as src:
        assert np.all(src.read(1)[hole] == 255)
    with rasterio.open(mad_dist) as src:
        assert np.all(src.read(1)[hole] == -9999.0)
    with rasterio.open(mad_vars) as src:
        assert np.all(src.read()[:, hole] == -9999.0)


@pytest.mark.filterwarnings("ignore:.*did not converge:UserWarning")
def test_array_outputs_pass_use_pbar_and_skip_double_stats(tmp_path, monkeypatch):
    """CVA/SFA/MAD write their outputs without stats or progress bars there."""
    base_path, chng_path, _ = _make_one_date_hole_pair(tmp_path)
    calls = []
    orig = icm.create_img_from_array_ref_img

    def _spy(*args, **kwargs):
        calls.append(kwargs)
        return orig(*args, **kwargs)

    monkeypatch.setattr(icm, "create_img_from_array_ref_img", _spy)
    icm.change_vector_analysis(
        base_path, chng_path, str(tmp_path / "b.tif"), str(tmp_path / "d.tif"),
        use_pbar=False,
    )
    if icm.HAVE_SCIPY:
        icm.slow_feature_analysis(
            base_path, chng_path, str(tmp_path / "sb.tif"), str(tmp_path / "sd.tif"),
            sfa_max_iter=5, use_pbar=False,
        )
        icm.multivariate_alteration_detection(
            base_path, chng_path, str(tmp_path / "mb.tif"), str(tmp_path / "md.tif"),
            str(tmp_path / "mv.tif"), mad_max_iter=5, use_kmeans=False,
            use_pbar=False,
        )
    assert len(calls) == (7 if icm.HAVE_SCIPY else 2)
    for kwargs in calls:
        assert kwargs.get("use_pbar") is False
        assert kwargs.get("calc_stats") is False


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
        str(img_path),
        str(msk_path),
        str(out_pyod),
        det,
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


# ---------------------------------------------------------------------------
# Diagnostics, convergence and regularisation (CVA / ISFA / IR-MAD)
# ---------------------------------------------------------------------------


def _make_diag_pair(tmp_path, n_bands=4, size=30, seed=21, same_band=None):
    """Two correlated dates with a changed block (rows/cols 0:8)."""
    rng = np.random.default_rng(seed)
    base_arr = rng.uniform(0.1, 0.4, size=(n_bands, size, size))
    mix = np.eye(n_bands) + 0.1 * rng.normal(size=(n_bands, n_bands))
    chng_arr = np.einsum("ij,jhw->ihw", mix, base_arr) + 0.05
    chng_arr += rng.normal(0.0, 0.01, size=chng_arr.shape)
    chng_arr[:, 0:8, 0:8] += rng.uniform(0.3, 0.6, size=(n_bands, 1, 1))
    if same_band is not None:
        chng_arr[same_band] = base_arr[same_band]
    base_path = tmp_path / "diag_base.tif"
    chng_path = tmp_path / "diag_chng.tif"
    _write_raster(base_path, base_arr.astype(np.float32))
    _write_raster(chng_path, chng_arr.astype(np.float32))
    return (
        str(base_path),
        str(chng_path),
        base_arr.reshape(n_bands, -1).astype(np.float32).astype(np.float64),
        chng_arr.reshape(n_bands, -1).astype(np.float32).astype(np.float64),
    )


def _read_band(path):
    with rasterio.open(path) as src:
        return src.read(1)


def test_cva_returns_diagnostics(tmp_path):
    """CVA returns the threshold, its method and the pixel counts."""
    base_path, chng_path, x, y = _make_diag_pair(tmp_path)
    out_bin = str(tmp_path / "b.tif")
    out = icm.change_vector_analysis(
        base_path, chng_path, out_bin, str(tmp_path / "d.tif"), use_pbar=False
    )
    assert isinstance(out, dict)
    assert set(out) >= {"threshold", "threshold_method", "n_valid", "n_change"}
    mag = np.sqrt(np.sum(np.square(x - y), axis=0))
    exp_thres = icm.calc_otsu_threshold(mag)
    assert out["threshold_method"] == "otsu"
    assert out["threshold"] == pytest.approx(exp_thres, rel=1e-6)
    assert out["n_valid"] == x.shape[1]
    assert out["n_change"] == int(np.sum(mag > exp_thres))
    assert out["n_change"] == int(np.sum(_read_band(out_bin) == 1))

    if icm.HAVE_SKLEARN:
        out_km = icm.change_vector_analysis(
            base_path, chng_path, out_bin, str(tmp_path / "d.tif"),
            use_kmeans=True, use_pbar=False,
        )
        assert out_km["threshold_method"] == "kmeans"
        assert out_km["n_change"] == int(np.sum(_read_band(out_bin) == 1))
        assert out_km["n_change"] == int(np.sum(mag > out_km["threshold"]))


@pytest.mark.skipif(not icm.HAVE_SCIPY, reason="scipy not installed")
def test_sfa_returns_diagnostics(tmp_path):
    """ISFA (one iteration) eigenvalues match a direct NumPy calculation."""
    base_path, chng_path, x, y = _make_diag_pair(tmp_path)
    out_bin = str(tmp_path / "b.tif")
    out = icm.slow_feature_analysis(
        base_path, chng_path, out_bin, str(tmp_path / "d.tif"),
        sfa_max_iter=1, use_pbar=False,
    )
    assert set(out) >= {
        "threshold", "threshold_method", "n_valid", "n_change",
        "eigenvalues", "n_iter", "converged", "last_change",
    }
    norm_x = (x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, ddof=1, keepdims=True)
    norm_y = (y - y.mean(axis=1, keepdims=True)) / y.std(axis=1, ddof=1, keepdims=True)
    mat_a = np.cov(norm_x - norm_y)
    mat_b = 0.5 * (np.cov(norm_x) + np.cov(norm_y))
    exp_eig = np.sort(np.linalg.eigvals(np.linalg.solve(mat_b, mat_a)).real)
    np.testing.assert_allclose(out["eigenvalues"], exp_eig, rtol=1e-5, atol=1e-9)
    assert out["n_iter"] == 1
    assert out["n_valid"] == x.shape[1]
    assert out["threshold_method"] == "otsu"
    assert out["n_change"] == int(np.sum(_read_band(out_bin) == 1))

    out_conv = icm.slow_feature_analysis(
        base_path, chng_path, out_bin, str(tmp_path / "d.tif"),
        sfa_max_iter=200, sfa_epsilon=1e-4, use_pbar=False,
    )
    assert out_conv["converged"] is True
    assert out_conv["n_iter"] < 200
    assert out_conv["last_change"] < 1e-4


@pytest.mark.skipif(not icm.HAVE_SCIPY, reason="scipy not installed")
def test_mad_returns_diagnostics(tmp_path):
    """IR-MAD (one iteration) canonical correlations match a direct CCA."""
    base_path, chng_path, x, y = _make_diag_pair(tmp_path)
    out_bin = str(tmp_path / "b.tif")
    out = icm.multivariate_alteration_detection(
        base_path, chng_path, out_bin, str(tmp_path / "d.tif"),
        str(tmp_path / "v.tif"), mad_max_iter=1, use_kmeans=False, use_pbar=False,
    )
    assert set(out) >= {
        "threshold", "threshold_method", "n_valid", "n_change",
        "canonical_correlations", "n_iter", "converged", "last_change",
    }
    n_bands = x.shape[0]
    cov = np.cov(np.concatenate((x, y), axis=0))
    s11, s22 = cov[:n_bands, :n_bands], cov[n_bands:, n_bands:]
    s12 = cov[:n_bands, n_bands:]

    def _inv_sqrt(mat):
        vals, vecs = np.linalg.eigh(mat)
        return vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T

    exp_rho = np.sort(np.linalg.svd(_inv_sqrt(s11) @ s12 @ _inv_sqrt(s22))[1])
    np.testing.assert_allclose(out["canonical_correlations"], exp_rho, atol=1e-5)
    assert out["n_iter"] == 1
    assert out["n_valid"] == x.shape[1]
    assert out["threshold_method"] == "otsu"
    assert out["n_change"] == int(np.sum(_read_band(out_bin) == 1))

    out_conv = icm.multivariate_alteration_detection(
        base_path, chng_path, out_bin, str(tmp_path / "d.tif"),
        str(tmp_path / "v.tif"), mad_max_iter=200, mad_epsilon=1e-4,
        use_kmeans=False, use_pbar=False,
    )
    assert out_conv["converged"] is True
    assert out_conv["n_iter"] < 200
    assert out_conv["last_change"] < 1e-4


@pytest.mark.skipif(not icm.HAVE_SCIPY, reason="scipy not installed")
def test_sfa_mad_warn_without_convergence(tmp_path):
    """Reaching max_iter without meeting epsilon warns and names the change."""
    base_path, chng_path, _, _ = _make_diag_pair(tmp_path)
    with pytest.warns(UserWarning, match="did not converge(.|\n)*mask"):
        out = icm.slow_feature_analysis(
            base_path, chng_path, str(tmp_path / "b.tif"), str(tmp_path / "d.tif"),
            sfa_max_iter=2, sfa_epsilon=1e-12, use_pbar=False,
        )
    assert out["converged"] is False
    assert out["n_iter"] == 2
    with pytest.warns(UserWarning, match="did not converge(.|\n)*mask"):
        out = icm.multivariate_alteration_detection(
            base_path, chng_path, str(tmp_path / "b.tif"), str(tmp_path / "d.tif"),
            str(tmp_path / "v.tif"), mad_max_iter=2, mad_epsilon=1e-12,
            use_kmeans=False, use_pbar=False,
        )
    assert out["converged"] is False
    assert out["n_iter"] == 2
    assert np.isfinite(out["last_change"]) and out["last_change"] > 1e-12


@pytest.mark.skipif(not icm.HAVE_SCIPY, reason="scipy not installed")
def test_mad_regularisation_keeps_correlations_below_one(tmp_path):
    """With an identical band the MAD correlation is 1; regularisation shrinks it."""
    base_path, chng_path, _, _ = _make_diag_pair(tmp_path, same_band=0)
    kwargs = {"mad_max_iter": 1, "use_kmeans": False, "use_pbar": False}
    out = icm.multivariate_alteration_detection(
        base_path, chng_path, str(tmp_path / "b.tif"), str(tmp_path / "d.tif"),
        str(tmp_path / "v.tif"), **kwargs,
    )
    assert max(out["canonical_correlations"]) == pytest.approx(1.0, abs=1e-6)
    out_reg = icm.multivariate_alteration_detection(
        base_path, chng_path, str(tmp_path / "b.tif"), str(tmp_path / "d.tif"),
        str(tmp_path / "v.tif"), mad_regularisation=0.1, **kwargs,
    )
    rho = np.asarray(out_reg["canonical_correlations"])
    assert np.all(rho > 0.0) and np.all(rho < 1.0)
    assert max(rho) < 0.95
    with pytest.raises(rsgispy.RSGISPyException):
        icm.multivariate_alteration_detection(
            base_path, chng_path, str(tmp_path / "b.tif"), str(tmp_path / "d.tif"),
            str(tmp_path / "v.tif"), mad_regularisation=-1.0, **kwargs,
        )


# ---------------------------------------------------------------------------
# Change distance outputs: centring and no data
# ---------------------------------------------------------------------------


def _run_change_method(method, base_path, chng_path, tmp_path, **kwargs):
    """Run one of CVA / ISFA / IR-MAD and return (out, bin_path, dist_path)."""
    out_bin = str(tmp_path / f"{method}_bin.tif")
    out_dist = str(tmp_path / f"{method}_dist.tif")
    if method == "cva":
        out = icm.change_vector_analysis(
            base_path, chng_path, out_bin, out_dist, use_pbar=False, **kwargs
        )
    elif method == "sfa":
        out = icm.slow_feature_analysis(
            base_path, chng_path, out_bin, out_dist, sfa_max_iter=5,
            use_pbar=False, **kwargs,
        )
    else:
        out = icm.multivariate_alteration_detection(
            base_path, chng_path, out_bin, out_dist,
            str(tmp_path / f"{method}_vars.tif"), mad_max_iter=5,
            use_pbar=False, **kwargs,
        )
    return out, out_bin, out_dist


_METHODS = ["cva"] + (["sfa", "mad"] if icm.HAVE_SCIPY else [])
_KMEANS = [False] + ([True] if icm.HAVE_SKLEARN else [])


@pytest.mark.filterwarnings("ignore:.*did not converge:UserWarning")
@pytest.mark.parametrize("method", _METHODS)
@pytest.mark.parametrize("use_kmeans", _KMEANS)
def test_centred_distance_matches_binary(tmp_path, method, use_kmeans):
    """``distance > 0`` equals the binary map for every method and threshold."""
    base_path, chng_path, _, _ = _make_diag_pair(tmp_path)
    out, out_bin, out_dist = _run_change_method(
        method, base_path, chng_path, tmp_path,
        use_kmeans=use_kmeans, centre_on_threshold=True,
    )
    assert out["threshold_method"] == ("kmeans" if use_kmeans else "otsu")
    bin_arr = _read_band(out_bin)
    dist_arr = _read_band(out_dist)
    np.testing.assert_array_equal(dist_arr > 0, bin_arr == 1)
    assert 0 < out["n_change"] < out["n_valid"]

    # Uncentred: the raw distance, thresholded at the returned threshold.
    out, out_bin, out_dist = _run_change_method(
        method, base_path, chng_path, tmp_path,
        use_kmeans=use_kmeans, centre_on_threshold=False,
    )
    raw = _read_band(out_dist).astype(np.float64)
    np.testing.assert_allclose(raw, dist_arr + out["threshold"], atol=1e-4)
    assert np.all(raw >= 0.0)


@pytest.mark.filterwarnings("ignore:.*did not converge:UserWarning")
@pytest.mark.parametrize("method", _METHODS)
def test_default_no_data_is_set(tmp_path, method):
    """Invalid pixels are written as -9999 and the no-data value is set."""
    base_path, chng_path, hole = _make_one_date_hole_pair(tmp_path)
    kwargs = {} if method == "cva" else {"use_kmeans": False}
    _, out_bin, out_dist = _run_change_method(
        method, base_path, chng_path, tmp_path, **kwargs
    )
    with rasterio.open(out_dist) as src:
        assert src.nodata == -9999.0
        dist_arr = src.read(1)
    assert np.all(dist_arr[hole] == -9999.0)
    assert np.all(dist_arr[~hole] != -9999.0)
    assert np.all(_read_band(out_bin)[hole] == 255)
    if method == "mad":
        with rasterio.open(tmp_path / "mad_vars.tif") as src:
            assert src.nodata == -9999.0
            assert np.all(src.read()[:, hole] == -9999.0)


# ---------------------------------------------------------------------------
# Reproducible K-means thresholds
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not icm.HAVE_SKLEARN, reason="scikit-learn not installed")
@pytest.mark.filterwarnings("ignore:.*did not converge:UserWarning")
@pytest.mark.parametrize("method", _METHODS)
def test_kmeans_threshold_reproducible(tmp_path, monkeypatch, method):
    """K-means gets random_state=rnd_seed and two runs give identical outputs."""
    base_path, chng_path, _, _ = _make_diag_pair(tmp_path)
    seen = []
    orig_kmeans = icm.KMeans

    def _spy_kmeans(*args, **kwargs):
        seen.append(kwargs.get("random_state"))
        return orig_kmeans(*args, **kwargs)

    monkeypatch.setattr(icm, "KMeans", _spy_kmeans)
    runs = []
    for i in range(2):
        run_dir = tmp_path / f"run{i}"
        run_dir.mkdir()
        out, out_bin, out_dist = _run_change_method(
            method, base_path, chng_path, run_dir, use_kmeans=True, rnd_seed=7
        )
        runs.append((out, _read_band(out_bin), _read_band(out_dist)))
    assert seen == [7, 7]
    assert runs[0][0]["threshold"] == runs[1][0]["threshold"]
    np.testing.assert_array_equal(runs[0][1], runs[1][1])
    np.testing.assert_array_equal(runs[0][2], runs[1][2])


# ---------------------------------------------------------------------------
# Thematic binary change outputs
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore:.*did not converge:UserWarning")
@pytest.mark.parametrize("method", _METHODS)
@pytest.mark.parametrize("calc_stats", [True, False])
def test_binary_change_output_is_thematic(tmp_path, method, calc_stats):
    """The binary change map is thematic with a 0 / 1 colour table."""
    base_path, chng_path, hole = _make_one_date_hole_pair(tmp_path)
    kwargs = {"calc_stats": calc_stats}
    if method != "cva":
        kwargs["use_kmeans"] = False
    _, out_bin, out_dist = _run_change_method(
        method, base_path, chng_path, tmp_path, **kwargs
    )
    assert is_img_thematic(out_bin)
    assert not is_img_thematic(out_dist)
    with rasterio.open(out_bin) as src:
        assert src.nodata == 255
        cmap = src.colormap(1)
        bin_arr = src.read(1)
        tags = src.tags(1)
    assert cmap[0] == icm.CHANGE_CLR_LUT[0]
    assert cmap[1] == icm.CHANGE_CLR_LUT[1]
    assert cmap[1] != cmap[0]
    assert cmap[255][3] == 0  # no data is transparent
    assert np.all(bin_arr[hole] == 255)
    assert set(np.unique(bin_arr[~hole]).tolist()) == {0, 1}
    if calc_stats:
        assert tags["STATISTICS_HISTOBINFUNCTION"] == "direct"
        # No data (255) is left out of the statistics.
        assert tags["STATISTICS_MINIMUM"] == "0"
        assert tags["STATISTICS_MAXIMUM"] == "1"
        n_chng = int(np.sum(bin_arr == 1))
        n_none = int(np.sum(bin_arr == 0))
        assert tags["STATISTICS_HISTOBINVALUES"] == f"{n_none}|{n_chng}|"
        assert float(tags["STATISTICS_MEAN"]) == pytest.approx(
            n_chng / (n_chng + n_none)
        )


def test_binary_change_output_class_names(tmp_path):
    """The binary change maps carry 'No change' / 'Change' class names."""
    import rsgispy.changedetect.imgchngmthds as icm
    from rsgispy.imageutils.colourtable import get_thematic_classes

    rng = np.random.default_rng(3)
    base = rng.integers(100, 1000, size=(3, 40, 40)).astype(np.uint16)
    chng = base.copy()
    chng[:, 10:20, 10:20] += 800
    prof = dict(driver="GTiff", width=40, height=40, count=3, dtype="uint16",
                crs="EPSG:32630", transform=rasterio.transform.from_origin(0, 400, 10, 10), nodata=0)
    base_img, chng_img = str(tmp_path / "b.tif"), str(tmp_path / "c.tif")
    for path, arr in ((base_img, base), (chng_img, chng)):
        with rasterio.open(path, "w", **prof) as dst:
            dst.write(arr)
    out_bin = str(tmp_path / "bin.tif")
    icm.change_vector_analysis(base_img, chng_img, out_bin, str(tmp_path / "dist.tif"), use_pbar=False)
    names = {k: v[0] for k, v in get_thematic_classes(out_bin).items()}
    assert names[0] == "No change" and names[1] == "Change"


def test_image_ratio_zero_denominator_is_no_data(tmp_path):
    """A valid zero on the change date gives no data, not a ratio of 0."""
    base_path = tmp_path / "b.tif"
    chng_path = tmp_path / "c.tif"
    _write_raster(base_path, np.array([[[10.0, 6.0]]], dtype=np.float32))
    _write_raster(chng_path, np.array([[[0.0, 3.0]]], dtype=np.float32))
    out = tmp_path / "ratio.tif"
    icm.image_ratio(str(base_path), str(chng_path), str(out), no_data_val=-9999.0,
                    base_img_no_data=-1, chng_img_no_data=-1, use_pbar=False)
    with rasterio.open(out) as src:
        np.testing.assert_allclose(src.read(1), [[-9999.0, 2.0]])


def _write_geotiff(
    path: Path,
    arr: np.ndarray,
    *,
    transform: Affine = from_origin(500000.0, 5800000.0, 10.0, 10.0),
    crs: object = "EPSG:32630",
    nodata=None,
) -> str:
    data = arr[np.newaxis, :, :] if arr.ndim == 2 else arr
    profile = {
        "driver": "GTiff",
        "height": int(data.shape[1]),
        "width": int(data.shape[2]),
        "count": int(data.shape[0]),
        "dtype": str(data.dtype),
        "crs": crs,
        "transform": transform,
    }
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
    return str(path)


def test_changedetect_streaming_and_binary_nodata_255(tmp_path, monkeypatch):
    monkeypatch.setenv(imgproc.ENV_VAR_BLOCK_SIZE, "8")
    rng = np.random.default_rng(19)
    base = rng.uniform(10.0, 30.0, size=(3, 16, 16)).astype(np.float32)
    chng = base + rng.normal(0.0, 0.5, size=(3, 16, 16)).astype(np.float32)
    chng[:, 8:14, 8:14] += 15.0  # clear change patch
    msk = np.ones((16, 16), dtype=np.uint8)
    msk[0:2, 0:2] = 0  # masked-out corner

    base_img = _write_geotiff(tmp_path / "cd_base.tif", base)
    chng_img = _write_geotiff(tmp_path / "cd_chng.tif", chng)
    msk_img = _write_geotiff(tmp_path / "cd_msk.tif", msk)

    out_diff = str(tmp_path / "diff.tif")
    icm.image_difference(
        base_img,
        chng_img,
        out_diff,
        vld_msk_img=msk_img,
        vld_msk_val=1,
        apply_std=True,
        no_data_val=-999.0,
        calc_stats=False,
        use_pbar=False,
    )
    with rasterio.open(out_diff) as dst:
        assert dst.nodata == -999.0
        d_arr = dst.read()
        assert np.all(d_arr[:, 0:2, 0:2] == -999.0)
        assert np.all(np.isfinite(d_arr))

    out_ratio = str(tmp_path / "ratio.tif")
    icm.image_ratio(
        base_img,
        chng_img,
        out_ratio,
        vld_msk_img=msk_img,
        vld_msk_val=1,
        apply_std=False,
        no_data_val=-999.0,
        calc_stats=False,
        use_pbar=False,
    )
    with rasterio.open(out_ratio) as dst:
        r_arr = dst.read()
        assert np.all(r_arr[:, 0:2, 0:2] == -999.0)
        np.testing.assert_allclose(
            r_arr[:, 5, 5], base[:, 5, 5] / chng[:, 5, 5], rtol=1e-5
        )

    cva_bin = str(tmp_path / "cva_bin.tif")
    cva_dist = str(tmp_path / "cva_dist.tif")
    icm.change_vector_analysis(
        base_img,
        chng_img,
        cva_bin,
        cva_dist,
        vld_msk_img=msk_img,
        vld_msk_val=1,
        calc_stats=False,
        use_pbar=False,
    )
    sfa_bin = str(tmp_path / "sfa_bin.tif")
    sfa_dist = str(tmp_path / "sfa_dist.tif")
    icm.slow_feature_analysis(
        base_img,
        chng_img,
        sfa_bin,
        sfa_dist,
        vld_msk_img=msk_img,
        vld_msk_val=1,
        sfa_max_iter=5,
        calc_stats=False,
        use_pbar=False,
    )
    mad_bin = str(tmp_path / "mad_bin.tif")
    mad_dist = str(tmp_path / "mad_dist.tif")
    mad_vars = str(tmp_path / "mad_vars.tif")
    icm.multivariate_alteration_detection(
        base_img,
        chng_img,
        mad_bin,
        mad_dist,
        mad_vars,
        vld_msk_img=msk_img,
        vld_msk_val=1,
        mad_max_iter=5,
        use_kmeans=False,
        calc_stats=False,
        use_pbar=False,
    )
    for b_path in (cva_bin, sfa_bin, mad_bin):
        with rasterio.open(b_path) as dst:
            b_arr = dst.read(1)
            assert dst.nodata == 255.0
            assert np.all(b_arr[0:2, 0:2] == 255)
            assert set(np.unique(b_arr[msk == 1]).tolist()).issubset({0, 1})
