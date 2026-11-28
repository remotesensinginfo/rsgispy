"""
Tests for rsgispy.tools.stats, thresholding, featureselection, and accuracy.
"""

import importlib.util
import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

import rsgispy
import rsgispy.tools.accuracy as raccuracy
import rsgispy.tools.featureselection as rfeatsel
import rsgispy.tools.stats as rstats
import rsgispy.tools.thresholding as rthresh
import rsgispy.tools.stats as stats
import rsgispy.tools.thresholding as thresholding
import rsgispy.tools.featureselection as featureselection
import rsgispy.tools.accuracy as accuracy


def test_cqv_threshold_handles_negative_features():
    """CQV uses |Q3 + Q1| so negative-valued features (dB, NDVI) are not dropped."""
    rng = np.random.default_rng(0)
    pos = rng.uniform(10.0, 20.0, size=200)
    df = pd.DataFrame({"pos": pos, "neg": -pos})
    cols = rfeatsel.cqv_threshold(df, lowthreshold=0.1, highthreshold=0.9)
    assert cols == ["pos", "neg"]


@pytest.mark.skipif(not rfeatsel.HAVE_SKLEARN, reason="needs scikit-learn")
def test_lassolars_selection_is_scale_invariant():
    """Selection does not depend on the units of the features."""
    rng = np.random.default_rng(1)
    n = 200
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x_noise = rng.normal(size=n)
    y = 2.0 * x1 + 1.0 * x2 + rng.normal(0.0, 0.1, size=n)
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "x_noise": x_noise})
    cols = ["x1", "x2", "x_noise"]
    scaled = df.copy()
    scaled["x2"] = scaled["x2"] * 1e-3
    for alpha in (None, 0.05):
        base = rfeatsel.lassolars_feature_selection(df, ["y"], cols, alpha_val=alpha)
        sel = rfeatsel.lassolars_feature_selection(scaled, ["y"], cols, alpha_val=alpha)
        assert {"x1", "x2"} <= set(base)
        assert sel == base


@pytest.mark.skipif(not rfeatsel.HAVE_SKLEARN, reason="needs scikit-learn")
def test_lassolars_selection_uses_every_target():
    """Alpha is chosen per target, so a noisy last target cannot hide another's feature."""
    rng = np.random.default_rng(2)
    n = 200
    x1 = rng.normal(size=n)
    x_noise = rng.normal(size=n)
    df = pd.DataFrame(
        {
            # y1 depends weakly on x1; y2 (the last target) is pure noise, so
            # a BIC alpha chosen from y2 alone removes every predictor.
            "y1": 0.3 * x1 + rng.normal(0.0, 0.1, size=n),
            "y2": rng.normal(0.0, 5.0, size=n),
            "x1": x1,
            "x_noise": x_noise,
        }
    )
    sel = rfeatsel.lassolars_feature_selection(df, ["y1", "y2"], ["x1", "x_noise"])
    assert "x1" in sel


def test_feature_selection_and_vif():
    """Test calc_pandas_vif, cqv_threshold, corr_feature_selection, and lassolars."""
    rng = np.random.default_rng(42)
    n_samples = 60
    x1 = rng.uniform(10.0, 20.0, size=n_samples)
    x2 = rng.uniform(1.0, 50.0, size=n_samples)
    x3 = 2.0 * x1 + rng.normal(0.0, 0.05, size=n_samples)
    x_const = rng.uniform(99.0, 100.0, size=n_samples)
    y = 3.0 * x1 - 0.5 * x2 + rng.normal(0.0, 0.2, size=n_samples)

    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3, "x_const": x_const})

    cqv_cols = rfeatsel.cqv_threshold(
        df, cols=["x1", "x2", "x_const"], lowthreshold=0.1, highthreshold=0.9
    )
    assert "x1" in cqv_cols
    assert "x2" in cqv_cols
    assert "x_const" not in cqv_cols
    with pytest.raises(rsgispy.RSGISPyException):
        rfeatsel.cqv_threshold(df, cols=["x1"])

    if rfeatsel.HAVE_SKLEARN:
        vifs = rfeatsel.calc_pandas_vif(df, cols=["x1", "x2", "x3"])
        assert vifs["x1"] > 10.0
        assert vifs["x3"] > 10.0
        assert vifs["x2"] < 5.0
        with pytest.raises(rsgispy.RSGISPyException):
            rfeatsel.calc_pandas_vif(df, cols=["x1"])

        lasso_cols = rfeatsel.lassolars_feature_selection(
            df, dep_vars=["y"], ind_vars=["x1", "x2", "x3"]
        )
        assert "x1" in lasso_cols or "x3" in lasso_cols

        lasso_alpha = rfeatsel.lassolars_feature_selection(
            df, dep_vars=["y"], ind_vars=["x1", "x2", "x3"], alpha_val=0.01
        )
        assert isinstance(lasso_alpha, list)

    if rfeatsel.HAVE_SCIPY and rfeatsel.HAVE_SKLEARN:
        corr_cols = rfeatsel.corr_feature_selection(
            df,
            dep_vars=["y"],
            ind_vars=["x1", "x2", "x3", "x_const"],
            n_min_clusters=2,
            n_max_clusters=4,
        )
        assert len(corr_cols) >= 2
        assert set(corr_cols).issubset({"x1", "x2", "x3", "x_const"})


def test_regression_and_classification_metrics():
    """Test breusch_pagan, binary/prob accuracy scores, bias_score, and bias-variance."""
    rng = np.random.default_rng(7)
    x = rng.normal(size=(80, 2))
    y = 2.0 * x[:, 0] - x[:, 1] + rng.normal(0.0, 0.5, size=80)

    if rfeatsel.HAVE_SCIPY and rfeatsel.HAVE_SKLEARN:
        lm_stat, pval, msg = rfeatsel.breusch_pagan_test(x, y)
        assert lm_stat >= 0.0
        assert 0.0 <= pval <= 1.0
        assert isinstance(msg, str)
        with pytest.raises(rsgispy.RSGISPyException):
            rfeatsel.breusch_pagan_test(x[:10], y)

    y_true = np.array([0, 0, 0, 1, 1, 1, 0, 1])
    y_prob = np.array([0.1, 0.2, 0.7, 0.8, 0.9, 0.3, 0.1, 0.85])
    y_pred = np.array([0, 0, 1, 1, 1, 0, 0, 1])

    if raccuracy.HAVE_SCIPY and raccuracy.HAVE_SKLEARN:
        prob_names, prob_scores = raccuracy.bin_accuracy_scores_prob(y_true, y_prob)
        assert len(prob_names) == len(prob_scores) == 5
        assert prob_scores[0] > 0.7  # AUC
        with pytest.raises(rsgispy.RSGISPyException):
            raccuracy.bin_accuracy_scores_prob(y_true[:3], y_prob)

    if raccuracy.HAVE_SKLEARN:
        bin_names, bin_scores = raccuracy.accuracy_scores_binary(y_true, y_pred)
        assert len(bin_names) == len(bin_scores) == 16
        assert bin_scores[6] == pytest.approx(0.75)  # Accuracy
        with pytest.raises(rsgispy.RSGISPyException):
            raccuracy.accuracy_scores_binary(y_true, np.array([0, 2, 1, 1, 1, 0, 0, 1]))

    yt = np.array([[10.0, 20.0], [20.0, 40.0], [30.0, 60.0]])
    yp = np.array([[12.0, 18.0], [22.0, 38.0], [32.0, 58.0]])
    bias, norm_bias = raccuracy.bias_score(yt, yp)
    assert bias == pytest.approx([2.0, -2.0])
    assert norm_bias == pytest.approx([10.0, -5.0])

    mse, bias_sq, var, noise = raccuracy.decompose_bias_variance(yt, yp)
    assert mse == pytest.approx([4.0, 4.0])
    assert bias_sq == pytest.approx([4.0, 4.0])
    assert var == pytest.approx([0.0, 0.0])
    assert noise == pytest.approx([0.0, 0.0])


def test_thresholding_masking_normalisation_and_variogram(tmp_path):
    """Test thresholding, mask_data_to_valid, image normalise/standarise, and variogram."""
    rng = np.random.default_rng(123)
    mode1 = rng.normal(10.0, 1.5, size=500)
    mode2 = rng.normal(30.0, 1.5, size=500)
    bimodal = np.concatenate([mode1, mode2])

    n_bins, bin_w = rthresh.get_nbins_histogram(bimodal)
    assert n_bins > 5
    assert bin_w > 0.0

    centres = rthresh.get_bin_centres(np.array([1.0, 4.0, 9.0]))
    assert centres == pytest.approx([2.5, 6.5])
    geom_centres = rthresh.get_bin_centres(np.array([1.0, 4.0, 9.0]), geometric=True)
    assert geom_centres == pytest.approx([2.0, 6.0])

    for fn in (
        rthresh.calc_otsu_threshold,
        rthresh.calc_yen_threshold,
        rthresh.calc_isodata_threshold,
        rthresh.calc_li_threshold,
    ):
        thres = fn(bimodal)
        assert 12.0 < thres < 28.0

    ce = rthresh.calc_hist_cross_entropy(bimodal, 20.0)
    assert np.isfinite(ce)

    if rthresh.HAVE_SCIPY:
        kurt_thres = rthresh.calc_kurt_skew_threshold(
            bimodal, max_val=25.0, min_val=5.0, init_thres=15.0, low_thres=True, contamination=20.0
        )
        assert np.isfinite(kurt_thres)
        with pytest.raises(rsgispy.RSGISPyException):
            rthresh.calc_kurt_skew_threshold(
                bimodal, max_val=25.0, min_val=5.0, init_thres=15.0, contamination=0.5
            )

    raw = np.array([[1.0, 2.0], [np.nan, 3.0], [0.0, 5.0], [4.0, 12.0], [3.0, 6.0]])
    masked = rstats.mask_data_to_valid(raw, lower_limit=0.5, upper_limit=10.0)
    assert masked.shape == (2, 2)

    img3d = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[10.0, 20.0], [30.0, 40.0]],
        ]
    )
    std_img = rstats.standardise_img_data(img3d)
    assert std_img.shape == img3d.shape
    assert np.mean(std_img[0]) == pytest.approx(0.0, abs=1e-7)
    assert np.std(std_img[0]) == pytest.approx(1.0, abs=1e-7)

    norm_img = rstats.normalise_img_data(img3d)
    assert norm_img.shape == img3d.shape
    assert np.min(norm_img[1]) == pytest.approx(0.0)
    assert np.max(norm_img[1]) == pytest.approx(1.0)

    # Mock skgstat for calc_semivariogram
    mock_skgstat = types.ModuleType("skgstat")
    mock_vario = MagicMock()
    mock_vario.get_empirical.return_value = (
        np.array([10.0, 20.0, 30.0]),
        np.array([0.5, 1.2, 1.8]),
    )
    mock_vario.lag_classes.return_value = [
        np.array([1.0, 2.0]),
        np.array([3.0, 4.0, 5.0]),
        np.array([6.0]),
    ]
    mock_skgstat.Variogram = MagicMock(return_value=mock_vario)

    coords = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 5.0], [30.0, 10.0]])
    vals = np.array([1.0, 2.0, 3.0, 4.0])
    out_csv = tmp_path / "vario.csv"
    out_png = tmp_path / "vario.png"

    with patch.dict(sys.modules, {"skgstat": mock_skgstat}):
        vario_df = rstats.calc_semivariogram(
            coords,
            vals,
            out_data_file=str(out_csv),
            out_plot_file=str(out_png),
            normalise_vals=True,
        )
    assert list(vario_df["count"]) == [2, 3, 1]
    assert out_csv.exists()
    assert out_png.exists()


def test_optional_scipy_and_sklearn_missing():
    """Verify dependent functions are omitted when scipy and/or scikit-learn are unavailable."""
    import rsgispy.tools as rtools

    blocked_scipy = {"scipy": None, "scipy.optimize": None, "scipy.stats": None}
    blocked_sklearn = {
        "sklearn": None,
        "sklearn.cluster": None,
        "sklearn.exceptions": None,
        "sklearn.feature_selection": None,
        "sklearn.linear_model": None,
        "sklearn.metrics": None,
    }

    def _fresh_import(mod_name, blocked):
        with patch.dict(sys.modules, blocked):
            sys.modules.pop(mod_name, None)
            return importlib.import_module(mod_name)

    orig_thresh = rthresh
    orig_featsel = rfeatsel
    orig_acc = raccuracy
    try:
        # Missing both scipy and sklearn
        both_blocked = {**blocked_scipy, **blocked_sklearn}
        t_mod = _fresh_import("rsgispy.tools.thresholding", both_blocked)
        assert not t_mod.HAVE_SCIPY
        assert hasattr(t_mod, "calc_kurt_skew_threshold")
        assert "calc_kurt_skew_threshold" in t_mod.__all__
        assert hasattr(t_mod, "calc_otsu_threshold")
        assert "calc_otsu_threshold" in t_mod.__all__
        with pytest.raises(rsgispy.RSGISPyException, match="pip install"):
            t_mod.calc_kurt_skew_threshold(np.array([1.0, 2.0, 3.0]), 3.0, 1.0, 2.0)

        f_mod = _fresh_import("rsgispy.tools.featureselection", both_blocked)
        assert not f_mod.HAVE_SCIPY
        assert not f_mod.HAVE_SKLEARN
        for fn_name in (
            "calc_pandas_vif",
            "lassolars_feature_selection",
            "corr_feature_selection",
            "breusch_pagan_test",
        ):
            assert hasattr(f_mod, fn_name)
            assert fn_name in f_mod.__all__
        assert hasattr(f_mod, "cqv_threshold")
        assert "cqv_threshold" in f_mod.__all__
        dummy_df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]})
        with pytest.raises(rsgispy.RSGISPyException, match="pip install"):
            f_mod.calc_pandas_vif(dummy_df)
        with pytest.raises(rsgispy.RSGISPyException, match="pip install"):
            f_mod.lassolars_feature_selection(dummy_df, ["y"], ["a"])
        with pytest.raises(rsgispy.RSGISPyException, match="pip install"):
            f_mod.corr_feature_selection(dummy_df, ["y"], ["a"])
        with pytest.raises(rsgispy.RSGISPyException, match="pip install"):
            f_mod.breusch_pagan_test(dummy_df[["a"]].to_numpy(), dummy_df["y"].to_numpy())

        a_mod = _fresh_import("rsgispy.tools.accuracy", both_blocked)
        assert not a_mod.HAVE_SCIPY
        assert not a_mod.HAVE_SKLEARN
        for fn_name in ("accuracy_scores_binary", "bin_accuracy_scores_prob"):
            assert hasattr(a_mod, fn_name)
            assert fn_name in a_mod.__all__
        assert hasattr(a_mod, "bias_score")
        assert "bias_score" in a_mod.__all__
        with pytest.raises(rsgispy.RSGISPyException, match="pip install"):
            a_mod.accuracy_scores_binary(np.array([0, 1]), np.array([0, 1]))
        with pytest.raises(rsgispy.RSGISPyException, match="pip install"):
            a_mod.bin_accuracy_scores_prob(np.array([0, 1]), np.array([0.1, 0.9]))

        # Missing scipy only (when sklearn is installed)
        if orig_featsel.HAVE_SKLEARN or importlib.util.find_spec("sklearn") is not None:
            t_mod = _fresh_import("rsgispy.tools.thresholding", blocked_scipy)
            assert not t_mod.HAVE_SCIPY
            assert hasattr(t_mod, "calc_kurt_skew_threshold")
            with pytest.raises(rsgispy.RSGISPyException, match="pip install"):
                t_mod.calc_kurt_skew_threshold(np.array([1.0, 2.0, 3.0]), 3.0, 1.0, 2.0)

            f_mod = _fresh_import("rsgispy.tools.featureselection", blocked_scipy)
            assert not f_mod.HAVE_SCIPY
            assert f_mod.HAVE_SKLEARN
            for fn_name in ("corr_feature_selection", "breusch_pagan_test"):
                assert hasattr(f_mod, fn_name)
                assert fn_name in f_mod.__all__
            with pytest.raises(rsgispy.RSGISPyException, match="pip install"):
                f_mod.corr_feature_selection(dummy_df, ["y"], ["a"])
            with pytest.raises(rsgispy.RSGISPyException, match="pip install"):
                f_mod.breusch_pagan_test(dummy_df[["a"]].to_numpy(), dummy_df["y"].to_numpy())
            for fn_name in ("calc_pandas_vif", "lassolars_feature_selection", "cqv_threshold"):
                assert hasattr(f_mod, fn_name)
                assert fn_name in f_mod.__all__

            a_mod = _fresh_import("rsgispy.tools.accuracy", blocked_scipy)
            assert not a_mod.HAVE_SCIPY
            assert a_mod.HAVE_SKLEARN
            assert hasattr(a_mod, "bin_accuracy_scores_prob")
            assert hasattr(a_mod, "accuracy_scores_binary")
            with pytest.raises(rsgispy.RSGISPyException, match="pip install"):
                a_mod.bin_accuracy_scores_prob(np.array([0, 1]), np.array([0.1, 0.9]))

        # Missing sklearn only (when scipy is installed)
        if orig_featsel.HAVE_SCIPY or importlib.util.find_spec("scipy") is not None:
            f_mod = _fresh_import("rsgispy.tools.featureselection", blocked_sklearn)
            assert f_mod.HAVE_SCIPY
            assert not f_mod.HAVE_SKLEARN
            for fn_name in (
                "calc_pandas_vif",
                "lassolars_feature_selection",
                "corr_feature_selection",
                "breusch_pagan_test",
            ):
                assert hasattr(f_mod, fn_name)
                assert fn_name in f_mod.__all__
            with pytest.raises(rsgispy.RSGISPyException, match="pip install"):
                f_mod.calc_pandas_vif(dummy_df)
            with pytest.raises(rsgispy.RSGISPyException, match="pip install"):
                f_mod.lassolars_feature_selection(dummy_df, ["y"], ["a"])
            with pytest.raises(rsgispy.RSGISPyException, match="pip install"):
                f_mod.corr_feature_selection(dummy_df, ["y"], ["a"])
            with pytest.raises(rsgispy.RSGISPyException, match="pip install"):
                f_mod.breusch_pagan_test(dummy_df[["a"]].to_numpy(), dummy_df["y"].to_numpy())
            assert hasattr(f_mod, "cqv_threshold")

            a_mod = _fresh_import("rsgispy.tools.accuracy", blocked_sklearn)
            assert a_mod.HAVE_SCIPY
            assert not a_mod.HAVE_SKLEARN
            for fn_name in ("accuracy_scores_binary", "bin_accuracy_scores_prob"):
                assert hasattr(a_mod, fn_name)
                assert fn_name in a_mod.__all__
            with pytest.raises(rsgispy.RSGISPyException, match="pip install"):
                a_mod.accuracy_scores_binary(np.array([0, 1]), np.array([0, 1]))
            with pytest.raises(rsgispy.RSGISPyException, match="pip install"):
                a_mod.bin_accuracy_scores_prob(np.array([0, 1]), np.array([0.1, 0.9]))
            assert hasattr(a_mod, "bias_score")
    finally:
        sys.modules["rsgispy.tools.thresholding"] = orig_thresh
        sys.modules["rsgispy.tools.featureselection"] = orig_featsel
        sys.modules["rsgispy.tools.accuracy"] = orig_acc
        rtools.thresholding = orig_thresh
        rtools.featureselection = orig_featsel
        rtools.accuracy = orig_acc


def test_percentile_helpers():
    """Test validate_percentiles_0_100 and compute_nanpercentiles."""
    rstats.validate_percentiles_0_100(0)
    rstats.validate_percentiles_0_100(50.0)
    rstats.validate_percentiles_0_100([0.0, 25.0, 100.0])
    rstats.validate_percentiles_0_100(np.array([10.0, 90.0]))
    rstats.validate_percentiles_0_100(np.array(50.0))

    for bad in (None, [], (), np.array([]), -0.1, 100.1, [10.0, 120.0], float("nan"), True, "50"):
        with pytest.raises(rsgispy.RSGISPyException):
            rstats.validate_percentiles_0_100(bad)  # type: ignore[arg-type]

    stack = np.array(
        [
            [[10.0, -99.0], [-99.0, 40.0]],
            [[30.0, 20.0], [-99.0, 60.0]],
        ],
        dtype=np.float32,
    )
    pct_seq = rstats.compute_nanpercentiles(stack.copy(), [0.0, 50.0, 100.0], no_data_val=-99.0)
    assert pct_seq.shape == (3, 2, 2)
    assert pct_seq[:, 0, 0] == pytest.approx([10.0, 20.0, 30.0])
    assert pct_seq[:, 0, 1] == pytest.approx([20.0, 20.0, 20.0])
    assert pct_seq[:, 1, 0] == pytest.approx([-99.0, -99.0, -99.0])

    # Scalar percentile preserves 3D shape (1, H, W) and works when no_data_val=None
    pct_scalar = rstats.compute_nanpercentiles(
        np.array([[[1, 2]], [[3, 4]]], dtype=np.int32), 50.0, no_data_val=None
    )
    assert pct_scalar.shape == (1, 1, 2)
    assert pct_scalar[0, 0] == pytest.approx([2.0, 3.0])


def test_similarity_metrics_and_phase_corr():
    """Test resolve_metric_type, calc_similarity_metric, and phase_cross_correlation_2d."""
    assert rstats.resolve_metric_type(rsgispy.METRIC_CORRELATION) == (
        rsgispy.METRIC_CORRELATION,
        "correlation",
        True,
    )
    assert rstats.resolve_metric_type("manhattan") == (
        rsgispy.METRIC_MANHATTAN,
        "manhattan",
        False,
    )
    assert rstats.resolve_metric_type("phase_correlation") == (
        rsgispy.METRIC_PHASE_CORR,
        "phase_correlation",
        True,
    )
    for bad in (True, 999, "corelation", "manhatten", 1.5):
        with pytest.raises(rsgispy.RSGISPyException):
            rstats.resolve_metric_type(bad)  # type: ignore[arg-type]

    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = np.array([3.0, 5.0, 7.0, 9.0, 11.0])
    assert rstats.calc_similarity_metric(a, b, rsgispy.METRIC_CORRELATION) == pytest.approx(1.0)
    assert rstats.calc_similarity_metric(a, a, rsgispy.METRIC_EUCLIDEAN) == pytest.approx(0.0)
    assert rstats.calc_similarity_metric(a, a + 2.0, rsgispy.METRIC_SQDIFF) == pytest.approx(4.0)
    assert rstats.calc_similarity_metric(a, a - 3.0, rsgispy.METRIC_MANHATTAN) == pytest.approx(3.0)
    mi = rstats.calc_similarity_metric(a, b, rsgispy.METRIC_MUTUAL_INFO, hist_bins=5)
    nmi = rstats.calc_similarity_metric(a, b, rsgispy.METRIC_NORM_MUTUAL_INFO, hist_bins=5)
    assert mi > 0.0
    assert 0.0 < nmi <= 1.0001

    with pytest.raises(rsgispy.RSGISPyException):
        rstats.calc_similarity_metric(a, a[:3], rsgispy.METRIC_CORRELATION)
    with pytest.raises(rsgispy.RSGISPyException):
        rstats.calc_similarity_metric(a, b, rsgispy.METRIC_PHASE_CORR)

    yy, xx = np.mgrid[0:64, 0:64].astype(np.float64)

    def _sig(gx, gy):
        return (
            np.sin(gx / 2.2) * np.cos(gy / 2.5)
            + 0.7 * np.cos((gx + 0.8 * gy) / 1.7)
            + 0.5 * np.sin((gx - 1.3 * gy) / 1.4)
        )

    img1 = _sig(xx, yy)
    img2 = _sig(xx - 2.0, yy + 1.0)
    dy, dx, peak = rstats.phase_cross_correlation_2d(img1, img2, upsample_factor=4)
    assert dx == pytest.approx(-2.0, abs=0.25)
    assert dy == pytest.approx(1.0, abs=0.25)
    assert peak > 0.1
    assert rstats.calc_similarity_metric(img1, img1, rsgispy.METRIC_PHASE_CORR) > 0.5


def test_semivariogram_skgstat_guard_and_normalise_constant_band():
    """calc_semivariogram guards missing skgstat; normalise_img_data avoids NaN."""
    with patch.dict(sys.modules, {"skgstat": None}):
        with pytest.raises(rsgispy.RSGISPyException, match="scikit-gstat"):
            stats.calc_semivariogram(np.array([[0.0, 0.0]]), np.array([1.0]))

    img = np.array(
        [
            [[5.0, 5.0], [5.0, 5.0]],
            [[10.0, 20.0], [30.0, 50.0]],
        ],
        dtype=np.float64,
    )
    norm = stats.normalise_img_data(img)
    assert norm.shape == img.shape
    assert np.all(np.isfinite(norm))
    np.testing.assert_allclose(norm[0], np.zeros((2, 2)))
    np.testing.assert_allclose(norm[1], [[0.0, 0.25], [0.5, 1.0]])


def test_thresholding_cross_entropy_and_li_edge_cases():
    """Guard boundary cross-entropy splits, constant arrays, and invalid Li args."""
    sample = np.array([1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0])
    # Boundary thresholds where one side is empty must not raise IndexError/ZeroDivisionError
    for thres in (-10.0, 1.0, 5.0, 100.0):
        val = thresholding.calc_hist_cross_entropy(sample, thres)
        assert isinstance(val, float) and np.isfinite(val)

    # Constant / single-unique array returns the constant value cleanly
    assert thresholding.calc_li_threshold(np.full(50, 7.5)) == 7.5
    assert thresholding.calc_li_threshold(np.array([-3.0, -3.0, np.nan])) == -3.0

    with pytest.raises(rsgispy.RSGISPyException, match="tolerance"):
        thresholding.calc_li_threshold(np.array([1.0, 2.0, 3.0]), tolerance=0.0)
    with pytest.raises(rsgispy.RSGISPyException, match="max_iter"):
        thresholding.calc_li_threshold(np.array([1.0, 2.0, 3.0]), max_iter=0)
    with pytest.raises(rsgispy.RSGISPyException, match="finite"):
        thresholding.calc_li_threshold(np.array([np.nan, np.inf]))

    bimodal = np.concatenate([np.full(40, 10.0), np.full(40, 100.0)])
    t_val = thresholding.calc_li_threshold(bimodal, max_iter=5)
    assert 10.0 < t_val < 100.0


def test_featureselection_degenerate_inputs():
    """Handle zero denominators in CQV, constant predictors, and ss_tot == 0."""
    df_cqv = pd.DataFrame(
        {
            "zero_sum": [-2.0, -1.0, 1.0, 2.0],
            "all_zero": [0.0, 0.0, 0.0, 0.0],
            "valid": [1.0, 2.0, 4.0, 8.0],
        }
    )
    selected = featureselection.cqv_threshold(
        df_cqv, lowthreshold=0.1, highthreshold=0.9
    )
    assert selected == ["valid"]

    rng = np.random.default_rng(42)
    n_rows = 30
    x1 = np.linspace(0.0, 1.0, n_rows)
    df_corr = pd.DataFrame(
        {
            "y": 2.0 * x1 + rng.normal(0, 0.05, n_rows),
            "const_a": np.ones(n_rows),
            "const_b": np.full(n_rows, 5.0),
            "x1": x1,
            "x2": x1 * 1.1 + 0.01,
            "x3": rng.normal(0, 1, n_rows),
            "x4": rng.normal(0, 1, n_rows),
        }
    )
    sel_corr = featureselection.corr_feature_selection(
        df_corr,
        dep_vars=["y"],
        ind_vars=["const_a", "const_b", "x1", "x2", "x3", "x4"],
        n_min_clusters=2,
        n_max_clusters=4,
        use_pbar=False,
    )
    assert "const_a" not in sel_corr and "const_b" not in sel_corr
    assert len(sel_corr) >= 2

    df_all_const = pd.DataFrame(
        {"y": [1, 2, 3, 4], "c1": [1, 1, 1, 1], "c2": [2, 2, 2, 2]}
    )
    assert (
        featureselection.corr_feature_selection(
            df_all_const, ["y"], ["c1", "c2"], use_pbar=False
        )
        == []
    )

    # Exact linear fit => constant residuals => ss_tot == 0 in breusch_pagan_test
    x_mat = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0], [5.0, 6.0]])
    y_exact = 3.0 * x_mat[:, 0] - 2.0 * x_mat[:, 1] + 1.0
    bp_stat, p_val, msg = featureselection.breusch_pagan_test(x_mat, y_exact)
    assert bp_stat == 0.0
    assert p_val == 1.0
    assert "No significant" in msg


def test_accuracy_binary_validation_and_constant_arrays():
    """Validate binary values before uint8 cast and accept constant 0/1 arrays."""
    with pytest.raises(rsgispy.RSGISPyException, match="binary"):
        accuracy.accuracy_scores_binary(np.array([0, 256, 257]), np.array([0, 1, 1]))
    with pytest.raises(rsgispy.RSGISPyException, match="binary"):
        accuracy.accuracy_scores_binary(np.array([0.0, 0.5, 1.0]), np.array([0, 1, 1]))
    with pytest.raises(rsgispy.RSGISPyException, match="binary"):
        accuracy.bin_accuracy_scores_prob(
            np.array([0, 257, 256]), np.array([0.1, 0.8, 0.2])
        )

    metrics_z, vals_z = accuracy.accuracy_scores_binary(
        np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0])
    )
    scores_all_zero = dict(zip(metrics_z, vals_z))
    assert scores_all_zero["Accuracy"] == 1.0
    assert scores_all_zero["Specificity"] == 1.0
    assert np.all(np.isfinite(vals_z))

    metrics_o, vals_o = accuracy.accuracy_scores_binary(
        np.array([1, 1, 1, 1]), np.array([1, 1, 1, 1])
    )
    scores_all_one = dict(zip(metrics_o, vals_o))
    assert scores_all_one["Accuracy"] == 1.0
    assert scores_all_one["F1 Score"] == 1.0
    assert scores_all_one["Sensitivity"] == 1.0

    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.5, 1.5, 4.0, 3.0])
    mse, bias_sq, var, noise = accuracy.decompose_bias_variance(y_true, y_pred)
    assert np.isclose(mse, bias_sq + var)
    assert np.isclose(noise, 0.0)
    with pytest.raises(rsgispy.RSGISPyException, match="dimensions"):
        accuracy.decompose_bias_variance(np.zeros((2, 2)), np.zeros((4,)))
    with pytest.raises(rsgispy.RSGISPyException, match="size"):
        accuracy.decompose_bias_variance(np.zeros((2, 3)), np.zeros((3, 2)))
