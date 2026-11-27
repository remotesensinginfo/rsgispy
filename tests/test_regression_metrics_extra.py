"""
Tests for the MSE and normalised metrics (``mse``, ``nrmse``, ``nrmse_range``
and ``nbias``) added to rsgispy.regression.regmetrics.calc_reg_metrics, and
for their JSON / CSV outputs.
"""

import csv
import json
import math

import numpy as np
import pandas as pd
import pytest

import rsgispy.regression.regmetrics as rrm

pytest.importorskip("sklearn")

_BASE_KEYS = ("r2", "rmse", "mae", "bias", "explained_variance", "pearson_r")
_NEW_KEYS = ("mse", "nrmse", "nrmse_range", "nbias")


def _expected(obs, pred):
    """Calculate the new metrics directly from their formulas."""
    obs = np.asarray(obs, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    err = pred - obs
    mse = float(np.mean(err**2))
    rmse = math.sqrt(mse)
    bias = float(np.mean(err))
    mean_obs = float(np.mean(obs))
    range_obs = float(np.max(obs) - np.min(obs))
    return {
        "mse": mse,
        "nrmse": 100.0 * rmse / mean_obs if mean_obs != 0 else float("nan"),
        "nrmse_range": (
            100.0 * rmse / range_obs if range_obs != 0 else float("nan")
        ),
        "nbias": 100.0 * bias / mean_obs if mean_obs != 0 else float("nan"),
    }


def _sample_df():
    """A small table with two targets and one non-finite row."""
    rng = np.random.default_rng(3)
    obs_a = np.linspace(10.0, 50.0, 30)
    obs_b = np.linspace(-4.0, 9.0, 30)
    df = pd.DataFrame(
        {
            "obs_a": obs_a,
            "pred_a": obs_a + rng.normal(1.0, 2.0, 30),
            "obs_b": obs_b,
            "pred_b": obs_b + rng.normal(-0.5, 1.0, 30),
        }
    )
    df.loc[5, "pred_a"] = np.nan
    return df


def test_new_keys_univariate_match_formulas():
    """Each new key equals its formula for one target."""
    df = _sample_df()
    stats = rrm.calc_reg_metrics(df, ref_col="obs_a", pred_col="pred_a")
    valid = df.dropna(subset=["pred_a"])
    exp = _expected(valid["obs_a"], valid["pred_a"])
    for key in _NEW_KEYS:
        assert stats[key] == pytest.approx(exp[key], rel=1e-12)
    assert stats["mse"] == pytest.approx(stats["rmse"] ** 2, rel=1e-12)
    assert stats["n_samples"] == 29
    # Existing keys are still present and come first.
    assert list(stats)[: len(_BASE_KEYS)] == list(_BASE_KEYS)


def test_existing_keys_unchanged():
    """The existing keys keep their values (compared with sklearn directly)."""
    import sklearn.metrics  # pylint: disable=import-outside-toplevel

    df = _sample_df().dropna()
    stats = rrm.calc_reg_metrics(df, ref_col="obs_a", pred_col="pred_a")
    assert stats["rmse"] == pytest.approx(
        math.sqrt(sklearn.metrics.mean_squared_error(df["obs_a"], df["pred_a"]))
    )
    assert stats["r2"] == pytest.approx(
        sklearn.metrics.r2_score(df["obs_a"], df["pred_a"])
    )
    assert stats["bias"] == pytest.approx(float(np.mean(df["pred_a"] - df["obs_a"])))


def test_new_keys_multivariate_per_target_and_headline():
    """Per-target keys follow the formulas; headline keys are the target means."""
    df = _sample_df().dropna()
    stats = rrm.calc_reg_metrics(
        df,
        ref_cols=["obs_a", "obs_b"],
        pred_cols=["pred_a", "pred_b"],
        target_names=["a", "b"],
    )
    exp_a = _expected(df["obs_a"], df["pred_a"])
    exp_b = _expected(df["obs_b"], df["pred_b"])
    for key in _NEW_KEYS:
        assert stats[f"a_{key}"] == pytest.approx(exp_a[key], rel=1e-12)
        assert stats[f"b_{key}"] == pytest.approx(exp_b[key], rel=1e-12)
        assert stats[key] == pytest.approx((exp_a[key] + exp_b[key]) / 2.0)


def test_zero_mean_and_zero_range_give_nan():
    """nrmse / nbias are nan for a zero mean; nrmse_range for a zero range."""
    zero_mean = pd.DataFrame(
        {"obs": [-2.0, -1.0, 0.0, 1.0, 2.0], "pred": [-1.5, -1.0, 0.5, 1.0, 2.5]}
    )
    stats = rrm.calc_reg_metrics(zero_mean, ref_col="obs", pred_col="pred")
    assert math.isnan(stats["nrmse"])
    assert math.isnan(stats["nbias"])
    assert stats["nrmse_range"] == pytest.approx(
        _expected(zero_mean["obs"], zero_mean["pred"])["nrmse_range"]
    )
    assert stats["mse"] == pytest.approx(0.15)

    const_obs = pd.DataFrame({"obs": [3.0, 3.0, 3.0], "pred": [2.0, 3.0, 4.5]})
    stats = rrm.calc_reg_metrics(const_obs, ref_col="obs", pred_col="pred")
    assert math.isnan(stats["nrmse_range"])
    assert stats["nrmse"] == pytest.approx(_expected([3.0] * 3, [2.0, 3.0, 4.5])["nrmse"])


def test_negative_mean_keeps_sign():
    """The mean observed value is signed, so a negative mean gives a negative nrmse."""
    df = pd.DataFrame({"obs": [-10.0, -12.0, -8.0], "pred": [-9.0, -12.5, -8.0]})
    stats = rrm.calc_reg_metrics(df, ref_col="obs", pred_col="pred")
    assert stats["nrmse"] < 0.0
    assert stats["nrmse"] == pytest.approx(_expected(df["obs"], df["pred"])["nrmse"])


def test_json_and_csv_writers_include_new_keys(tmp_path):
    """The JSON and CSV outputs hold the new keys (CSV: one column each)."""
    df = _sample_df().dropna()
    out_json = tmp_path / "m.json"
    out_csv = tmp_path / "m.csv"
    stats = rrm.calc_reg_metrics(
        df,
        ref_cols=["obs_a", "obs_b"],
        pred_cols=["pred_a", "pred_b"],
        target_names=["a", "b"],
        out_json_file=out_json,
        out_csv_file=out_csv,
    )
    with open(out_json, encoding="utf-8") as in_fp:
        json_stats = json.load(in_fp)
    for key in _NEW_KEYS:
        assert json_stats[key] == pytest.approx(stats[key])
        assert json_stats[f"b_{key}"] == pytest.approx(stats[f"b_{key}"])

    with open(out_csv, newline="", encoding="utf-8") as in_fp:
        rows = list(csv.DictReader(in_fp))
    assert [r["target"] for r in rows] == ["a", "b", "overall_mean"]
    assert list(rows[0]) == ["target", "n_samples", *_BASE_KEYS, *_NEW_KEYS]
    for key in _NEW_KEYS:
        assert float(rows[0][key]) == pytest.approx(stats[f"a_{key}"])
        assert float(rows[2][key]) == pytest.approx(stats[key])


def test_json_writes_nan(tmp_path):
    """A nan metric is written to JSON as NaN and read back as nan."""
    df = pd.DataFrame({"obs": [-1.0, 0.0, 1.0], "pred": [-1.0, 0.5, 1.0]})
    out_json = tmp_path / "nan.json"
    out_csv = tmp_path / "nan.csv"
    rrm.calc_reg_metrics(
        df, ref_col="obs", pred_col="pred", out_json_file=out_json, out_csv_file=out_csv
    )
    with open(out_json, encoding="utf-8") as in_fp:
        json_stats = json.load(in_fp)
    assert math.isnan(json_stats["nrmse"])
    csv_df = pd.read_csv(out_csv)
    assert math.isnan(csv_df.loc[0, "nbias"])
