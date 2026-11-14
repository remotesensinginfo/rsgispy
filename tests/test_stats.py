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
    std_img = rstats.standarise_img_data(img3d)
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
        assert not hasattr(t_mod, "calc_kurt_skew_threshold")
        assert "calc_kurt_skew_threshold" not in t_mod.__all__
        assert hasattr(t_mod, "calc_otsu_threshold")
        assert "calc_otsu_threshold" in t_mod.__all__

        f_mod = _fresh_import("rsgispy.tools.featureselection", both_blocked)
        assert not f_mod.HAVE_SCIPY
        assert not f_mod.HAVE_SKLEARN
        for fn_name in (
            "calc_pandas_vif",
            "lassolars_feature_selection",
            "corr_feature_selection",
            "breusch_pagan_test",
        ):
            assert not hasattr(f_mod, fn_name)
            assert fn_name not in f_mod.__all__
        assert hasattr(f_mod, "cqv_threshold")
        assert "cqv_threshold" in f_mod.__all__

        a_mod = _fresh_import("rsgispy.tools.accuracy", both_blocked)
        assert not a_mod.HAVE_SCIPY
        assert not a_mod.HAVE_SKLEARN
        for fn_name in ("accuracy_scores_binary", "bin_accuracy_scores_prob"):
            assert not hasattr(a_mod, fn_name)
            assert fn_name not in a_mod.__all__
        assert hasattr(a_mod, "bias_score")
        assert "bias_score" in a_mod.__all__

        # Missing scipy only (when sklearn is installed)
        if orig_featsel.HAVE_SKLEARN or importlib.util.find_spec("sklearn") is not None:
            t_mod = _fresh_import("rsgispy.tools.thresholding", blocked_scipy)
            assert not t_mod.HAVE_SCIPY
            assert not hasattr(t_mod, "calc_kurt_skew_threshold")

            f_mod = _fresh_import("rsgispy.tools.featureselection", blocked_scipy)
            assert not f_mod.HAVE_SCIPY
            assert f_mod.HAVE_SKLEARN
            for fn_name in ("corr_feature_selection", "breusch_pagan_test"):
                assert not hasattr(f_mod, fn_name)
                assert fn_name not in f_mod.__all__
            for fn_name in ("calc_pandas_vif", "lassolars_feature_selection", "cqv_threshold"):
                assert hasattr(f_mod, fn_name)
                assert fn_name in f_mod.__all__

            a_mod = _fresh_import("rsgispy.tools.accuracy", blocked_scipy)
            assert not a_mod.HAVE_SCIPY
            assert a_mod.HAVE_SKLEARN
            assert not hasattr(a_mod, "bin_accuracy_scores_prob")
            assert hasattr(a_mod, "accuracy_scores_binary")

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
                assert not hasattr(f_mod, fn_name)
                assert fn_name not in f_mod.__all__
            assert hasattr(f_mod, "cqv_threshold")

            a_mod = _fresh_import("rsgispy.tools.accuracy", blocked_sklearn)
            assert a_mod.HAVE_SCIPY
            assert not a_mod.HAVE_SKLEARN
            for fn_name in ("accuracy_scores_binary", "bin_accuracy_scores_prob"):
                assert not hasattr(a_mod, fn_name)
                assert fn_name not in a_mod.__all__
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
