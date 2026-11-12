"""
Tests for rsgispy.tools.stats.
"""

import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

import rsgispy
import rsgispy.tools.stats as rstats


def test_feature_selection_and_vif():
    """Test calc_pandas_vif, cqv_threshold, corr_feature_selection, and lassolars_feature_selection."""
    rng = np.random.default_rng(42)
    n_samples = 60
    x1 = rng.uniform(10.0, 20.0, size=n_samples)
    x2 = rng.uniform(1.0, 50.0, size=n_samples)
    x3 = 2.0 * x1 + rng.normal(0.0, 0.05, size=n_samples)
    x_const = rng.uniform(99.0, 100.0, size=n_samples)
    y = 3.0 * x1 - 0.5 * x2 + rng.normal(0.0, 0.2, size=n_samples)

    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3, "x_const": x_const})

    vifs = rstats.calc_pandas_vif(df, cols=["x1", "x2", "x3"])
    assert vifs["x1"] > 10.0
    assert vifs["x3"] > 10.0
    assert vifs["x2"] < 5.0
    with pytest.raises(rsgispy.RSGISPyException):
        rstats.calc_pandas_vif(df, cols=["x1"])

    cqv_cols = rstats.cqv_threshold(df, cols=["x1", "x2", "x_const"], lowthreshold=0.1, highthreshold=0.9)
    assert "x1" in cqv_cols
    assert "x2" in cqv_cols
    assert "x_const" not in cqv_cols
    with pytest.raises(rsgispy.RSGISPyException):
        rstats.cqv_threshold(df, cols=["x1"])

    corr_cols = rstats.corr_feature_selection(
        df, dep_vars=["y"], ind_vars=["x1", "x2", "x3", "x_const"], n_min_clusters=2, n_max_clusters=4
    )
    assert len(corr_cols) >= 2
    assert set(corr_cols).issubset({"x1", "x2", "x3", "x_const"})

    lasso_cols = rstats.lassolars_feature_selection(df, dep_vars=["y"], ind_vars=["x1", "x2", "x3"])
    assert "x1" in lasso_cols or "x3" in lasso_cols

    lasso_alpha = rstats.lassolars_feature_selection(
        df, dep_vars=["y"], ind_vars=["x1", "x2", "x3"], alpha_val=0.01
    )
    assert isinstance(lasso_alpha, list)


def test_regression_and_classification_metrics():
    """Test breusch_pagan_test, bin_accuracy_scores_prob, accuracy_scores_binary, bias_score, decompose_bias_variance."""
    rng = np.random.default_rng(7)
    x = rng.normal(size=(80, 2))
    y = 2.0 * x[:, 0] - x[:, 1] + rng.normal(0.0, 0.5, size=80)
    lm_stat, pval, msg = rstats.breusch_pagan_test(x, y)
    assert lm_stat >= 0.0
    assert 0.0 <= pval <= 1.0
    assert isinstance(msg, str)
    with pytest.raises(rsgispy.RSGISPyException):
        rstats.breusch_pagan_test(x[:10], y)

    y_true = np.array([0, 0, 0, 1, 1, 1, 0, 1])
    y_prob = np.array([0.1, 0.2, 0.7, 0.8, 0.9, 0.3, 0.1, 0.85])
    y_pred = np.array([0, 0, 1, 1, 1, 0, 0, 1])

    prob_names, prob_scores = rstats.bin_accuracy_scores_prob(y_true, y_prob)
    assert len(prob_names) == len(prob_scores) == 5
    assert prob_scores[0] > 0.7  # AUC

    bin_names, bin_scores = rstats.accuracy_scores_binary(y_true, y_pred)
    assert len(bin_names) == len(bin_scores) == 16
    assert bin_scores[6] == pytest.approx(0.75)  # Accuracy

    with pytest.raises(rsgispy.RSGISPyException):
        rstats.bin_accuracy_scores_prob(y_true[:3], y_prob)
    with pytest.raises(rsgispy.RSGISPyException):
        rstats.accuracy_scores_binary(y_true, np.array([0, 2, 1, 1, 1, 0, 0, 1]))

    yt = np.array([[10.0, 20.0], [20.0, 40.0], [30.0, 60.0]])
    yp = np.array([[12.0, 18.0], [22.0, 38.0], [32.0, 58.0]])
    bias, norm_bias = rstats.bias_score(yt, yp)
    assert bias == pytest.approx([2.0, -2.0])
    assert norm_bias == pytest.approx([10.0, -5.0])

    mse, bias_sq, var, noise = rstats.decompose_bias_variance(yt, yp)
    assert mse == pytest.approx([4.0, 4.0])
    assert bias_sq == pytest.approx([4.0, 4.0])
    assert var == pytest.approx([0.0, 0.0])
    assert noise == pytest.approx([0.0, 0.0])


def test_thresholding_masking_normalisation_and_variogram(tmp_path):
    """Test histogram thresholding, mask_data_to_valid, image normalise/standarise, and calc_semivariogram."""
    rng = np.random.default_rng(123)
    mode1 = rng.normal(10.0, 1.5, size=500)
    mode2 = rng.normal(30.0, 1.5, size=500)
    bimodal = np.concatenate([mode1, mode2])

    n_bins, bin_w = rstats.get_nbins_histogram(bimodal)
    assert n_bins > 5
    assert bin_w > 0.0

    centres = rstats.get_bin_centres(np.array([1.0, 4.0, 9.0]))
    assert centres == pytest.approx([2.5, 6.5])
    geom_centres = rstats.get_bin_centres(np.array([1.0, 4.0, 9.0]), geometric=True)
    assert geom_centres == pytest.approx([2.0, 6.0])

    for fn in (
        rstats.calc_otsu_threshold,
        rstats.calc_yen_threshold,
        rstats.calc_isodata_threshold,
        rstats.calc_li_threshold,
    ):
        thres = fn(bimodal)
        assert 12.0 < thres < 28.0

    ce = rstats.calc_hist_cross_entropy(bimodal, 20.0)
    assert np.isfinite(ce)

    kurt_thres = rstats.calc_kurt_skew_threshold(
        bimodal, max_val=25.0, min_val=5.0, init_thres=15.0, low_thres=True, contamination=20.0
    )
    assert np.isfinite(kurt_thres)
    with pytest.raises(rsgispy.RSGISPyException):
        rstats.calc_kurt_skew_threshold(bimodal, max_val=25.0, min_val=5.0, init_thres=15.0, contamination=0.5)

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
            normalize_vals=True,
        )
    assert list(vario_df["count"]) == [2, 3, 1]
    assert out_csv.exists()
    assert out_png.exists()
