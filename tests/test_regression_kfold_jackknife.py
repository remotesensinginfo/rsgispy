"""
Tests for the repeated k-fold (perform_kfold_fit, summarise_kfold_metrics)
and jackknife (perform_jackknife_fit) assessment of regressors in
rsgispy.regression.regsklearn.
"""

# pylint: disable=too-many-locals,too-many-arguments,too-many-positional-arguments
# pylint: disable=attribute-defined-outside-init,ungrouped-imports

import os

import numpy as np
import pandas as pd
import pytest

import rsgispy
from rsgispy.regression.regmetrics import calc_reg_metrics

sklearn = pytest.importorskip("sklearn")
# pylint: disable=wrong-import-position,wrong-import-order
import sklearn.base  # noqa: E402
import sklearn.model_selection  # noqa: E402
from sklearn.ensemble import ExtraTreesRegressor  # noqa: E402
from sklearn.linear_model import LinearRegression, Ridge  # noqa: E402
from sklearn.neighbors import KNeighborsRegressor  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVR  # noqa: E402

import rsgispy.regression.regsklearn as rsk  # noqa: E402

_METRIC_KEYS = (
    "r2", "rmse", "mae", "bias", "explained_variance", "pearson_r",
    "mse", "nrmse", "nrmse_range", "nbias",
)


def _make_data(n_smpls=40, n_feats=3, seed=1):
    """Synthetic predictors and two (positive) targets."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_smpls, n_feats))
    y1 = 50.0 + x @ np.arange(1.0, n_feats + 1.0) * 4.0 + rng.normal(0, 2.0, n_smpls)
    y2 = 20.0 + 3.0 * x[:, 0] - 2.0 * x[:, -1] + rng.normal(0, 1.0, n_smpls)
    return x, np.column_stack([y1, y2])


def _tutorial_perform_kfold_fit(
    regrs_obj, x, y, target_names, n_splits=5, n_repeats=20, rnd_seed=0
):
    """A copy of the tutorials' perform_kfold_fit (05_regression 05 and 06)."""
    kfold = sklearn.model_selection.RepeatedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=rnd_seed
    )
    fold_metrics = {name: [] for name in target_names}
    y_true = []
    y_pred = []
    for train_idx, test_idx in kfold.split(x):
        fold_mdl = sklearn.base.clone(regrs_obj)
        fold_mdl.fit(x[train_idx], y[train_idx])
        y_hat = np.asarray(fold_mdl.predict(x[test_idx])).reshape(len(test_idx), -1)
        y_true.append(y[test_idx])
        y_pred.append(y_hat)
        for i, name in enumerate(target_names):
            fold_df = pd.DataFrame({"obs": y[test_idx, i], "pred": y_hat[:, i]})
            metrics = calc_reg_metrics(fold_df, ref_col="obs", pred_col="pred")
            mean_obs = np.mean(y[test_idx, i])
            metrics["nrmse"] = 100.0 * metrics["rmse"] / mean_obs
            metrics["nbias"] = 100.0 * metrics["bias"] / mean_obs
            fold_metrics[name].append(metrics)
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    metrics = {}
    residuals = {}
    for i, name in enumerate(target_names):
        metrics[name] = pd.DataFrame(fold_metrics[name]).drop(columns="n_samples")
        residuals[name] = pd.DataFrame({"y_true": y_true[:, i], "y_pred": y_pred[:, i]})
    return metrics, residuals


# ---------------------------------------------------------------------------
# perform_kfold_fit
# ---------------------------------------------------------------------------


def test_kfold_matches_hand_written_loop():
    """Metrics and residuals equal a hand-written RepeatedKFold loop."""
    x, y = _make_data()
    metrics, residuals = rsk.perform_kfold_fit(
        LinearRegression(), x, y, ["a", "b"], n_splits=4, n_repeats=3,
        rnd_seed=7, use_pbar=False,
    )
    rkf = sklearn.model_selection.RepeatedKFold(n_splits=4, n_repeats=3, random_state=7)
    exp_rmse = {"a": [], "b": []}
    exp_pred = []
    exp_idx = []
    for train_idx, test_idx in rkf.split(x):
        mdl = LinearRegression().fit(x[train_idx], y[train_idx])
        pred = mdl.predict(x[test_idx])
        exp_pred.append(pred)
        exp_idx.append(test_idx)
        for i, name in enumerate(["a", "b"]):
            exp_rmse[name].append(np.sqrt(np.mean((pred[:, i] - y[test_idx, i]) ** 2)))
    exp_pred = np.concatenate(exp_pred)
    exp_idx = np.concatenate(exp_idx)

    for i, name in enumerate(["a", "b"]):
        m_df = metrics[name]
        assert len(m_df) == 12
        assert list(m_df.columns) == ["repeat", "fold", "n_samples", *_METRIC_KEYS]
        assert m_df["repeat"].tolist() == [r for r in range(3) for _ in range(4)]
        assert m_df["fold"].tolist() == list(range(4)) * 3
        np.testing.assert_allclose(m_df["rmse"], exp_rmse[name], rtol=1e-10)
        r_df = residuals[name]
        assert list(r_df.columns) == ["repeat", "fold", "sample_idx", "y_true", "y_pred"]
        assert len(r_df) == 3 * x.shape[0]
        np.testing.assert_array_equal(r_df["sample_idx"], exp_idx)
        np.testing.assert_allclose(r_df["y_pred"], exp_pred[:, i], rtol=1e-10)
        np.testing.assert_allclose(r_df["y_true"], y[exp_idx, i])
        # Each sample is predicted once per repeat.
        assert (r_df.groupby("repeat")["sample_idx"].nunique() == x.shape[0]).all()


@pytest.mark.parametrize(
    "regrs_obj",
    [
        ExtraTreesRegressor(n_estimators=20, random_state=0),
        make_pipeline(StandardScaler(), KNeighborsRegressor()),
        LinearRegression(),
    ],
)
def test_kfold_reproduces_tutorial_function(regrs_obj):
    """With the same seed the results equal the tutorials' function."""
    x, y = _make_data()
    names = ["Vol / ha", "BA / ha"]
    tut_m, tut_r = _tutorial_perform_kfold_fit(
        regrs_obj, x, y, names, n_splits=5, n_repeats=3, rnd_seed=0
    )
    new_m, new_r = rsk.perform_kfold_fit(
        regrs_obj, x, y, names, n_splits=5, n_repeats=3, rnd_seed=0, use_pbar=False
    )
    for name in names:
        for col in tut_m[name].columns:
            np.testing.assert_allclose(new_m[name][col], tut_m[name][col], rtol=1e-12, err_msg=col)
        np.testing.assert_allclose(new_r[name]["y_true"], tut_r[name]["y_true"])
        np.testing.assert_allclose(new_r[name]["y_pred"], tut_r[name]["y_pred"], rtol=1e-12)


def test_kfold_deterministic_and_seed_changes_splits():
    """The same seed gives identical results; another seed other splits."""
    x, y = _make_data()
    est = ExtraTreesRegressor(n_estimators=10, random_state=3)
    run1 = rsk.perform_kfold_fit(est, x, y[:, 0], n_repeats=2, use_pbar=False)
    run2 = rsk.perform_kfold_fit(est, x, y[:, 0], n_repeats=2, use_pbar=False)
    pd.testing.assert_frame_equal(run1[0]["target"], run2[0]["target"])
    pd.testing.assert_frame_equal(run1[1]["target"], run2[1]["target"])
    run3 = rsk.perform_kfold_fit(est, x, y[:, 0], n_repeats=2, rnd_seed=1, use_pbar=False)
    assert not np.array_equal(run1[1]["target"]["sample_idx"], run3[1]["target"]["sample_idx"])
    # The estimator passed in is not fitted.
    assert not hasattr(est, "estimators_")


def test_kfold_dataframes_pipeline_and_multitarget():
    """DataFrame inputs keep the names; pipelines and single-output estimators work."""
    x, y = _make_data()
    x_df = pd.DataFrame(x, columns=["p95", "cover", "skew"])
    y_df = pd.DataFrame(y, columns=["vol", "ba"])
    metrics, residuals = rsk.perform_kfold_fit(
        make_pipeline(StandardScaler(), SVR()), x_df, y_df, n_splits=3, n_repeats=2,
        use_pbar=False,
    )
    assert set(metrics) == {"vol", "ba"} == set(residuals)
    assert len(metrics["vol"]) == 6
    assert metrics["ba"]["n_samples"].sum() == 2 * x.shape[0]
    # A Series target takes its name.
    m_ser, _ = rsk.perform_kfold_fit(
        Ridge(), x_df, y_df["ba"], n_splits=3, n_repeats=1, use_pbar=False
    )
    assert list(m_ser) == ["ba"]
    # Default names for a 2D NumPy y.
    m_np, _ = rsk.perform_kfold_fit(Ridge(), x, y, n_splits=3, n_repeats=1, use_pbar=False)
    assert list(m_np) == ["target_1", "target_2"]


def test_kfold_groups_never_split():
    """With groups, no group is in both the train and test samples."""
    x, y = _make_data(n_smpls=60)
    groups = np.repeat(np.arange(15), 4)
    seen_train_test = []

    class _Recorder(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin):
        """Records the groups of the training samples."""

        def fit(self, x_fit, y_fit):  # pylint: disable=unused-argument
            """Store the training rows."""
            self.train_rows_ = {tuple(r) for r in np.round(x_fit, 12)}
            return self

        def predict(self, x_pred):
            """Check the test rows are not in the training rows."""
            test_rows = {tuple(r) for r in np.round(x_pred, 12)}
            seen_train_test.append(len(test_rows & self.train_rows_))
            return np.zeros(x_pred.shape[0])

    _, residuals = rsk.perform_kfold_fit(
        _Recorder(), x, y[:, 0], groups=groups, n_splits=5, n_repeats=4, use_pbar=False
    )
    assert seen_train_test == [0] * 20
    r_df = residuals["target"]
    r_df["group"] = groups[r_df["sample_idx"]]
    # A group is in one fold per repeat; every sample is predicted once per repeat.
    assert (r_df.groupby(["repeat", "group"])["fold"].nunique() == 1).all()
    assert (r_df.groupby("repeat")["sample_idx"].nunique() == 60).all()
    # 15 groups in 5 folds: 3 groups (12 samples) per fold.
    assert (r_df.groupby(["repeat", "fold"]).size() == 12).all()
    # The group folds change between repeats.
    folds = r_df.pivot_table(index="group", columns="repeat", values="fold")
    assert not all((folds[0] == folds[rep]).all() for rep in range(1, 4))


def test_kfold_input_errors(tmp_path):
    """Bad inputs raise RSGISPyException."""
    x, y = _make_data()
    with pytest.raises(rsgispy.RSGISPyException):
        rsk.perform_kfold_fit(Ridge(), x[:-1], y, use_pbar=False)
    with pytest.raises(rsgispy.RSGISPyException):
        rsk.perform_kfold_fit(Ridge(), x, y, ["only_one"], use_pbar=False)
    with pytest.raises(rsgispy.RSGISPyException):
        rsk.perform_kfold_fit(Ridge(), x, y, groups=np.arange(40) % 3, n_splits=5, use_pbar=False)
    with pytest.raises(rsgispy.RSGISPyException):
        rsk.perform_kfold_fit(Ridge(), x, y, n_splits=1, use_pbar=False)
    with pytest.raises(rsgispy.RSGISPyException):
        rsk.perform_kfold_fit(Ridge(), x, y, ["a-b", "a b"], out_dir=tmp_path, use_pbar=False)
    assert not os.listdir(tmp_path)  # checked before any fit or file


def test_kfold_csv_round_trip(tmp_path):
    """The metrics and residuals CSVs read back to the returned tables."""
    x, y = _make_data()
    out_dir = tmp_path / "kfold_out"
    metrics, residuals = rsk.perform_kfold_fit(
        Ridge(), x, y, ["Vol / ha", "BA / ha"], n_splits=3, n_repeats=2,
        out_dir=out_dir, out_prefix="kf_ridge", use_pbar=False,
    )
    for name, stem in (("Vol / ha", "vol_ha"), ("BA / ha", "ba_ha")):
        m_file = os.path.join(out_dir, f"kf_ridge_metrics_{stem}.csv")
        r_file = os.path.join(out_dir, f"kf_ridge_residuals_{stem}.csv")
        pd.testing.assert_frame_equal(
            pd.read_csv(m_file), metrics[name], check_exact=False, rtol=1e-12
        )
        pd.testing.assert_frame_equal(
            pd.read_csv(r_file), residuals[name], check_exact=False, rtol=1e-12
        )


def test_summarise_kfold_metrics():
    """The summary equals pandas / NumPy statistics of the fold metrics."""
    x, y = _make_data()
    metrics, _ = rsk.perform_kfold_fit(Ridge(), x, y, ["a", "b"], n_repeats=4, use_pbar=False)
    summary = rsk.summarise_kfold_metrics(metrics)
    assert list(summary.columns) == ["mean", "median", "std", "p05", "p95"]
    assert summary.index.names == ["target", "metric"]
    assert list(summary.loc["a"].index) == list(_METRIC_KEYS)
    vals = metrics["b"]["nrmse"].to_numpy()
    row = summary.loc[("b", "nrmse")]
    assert row["mean"] == pytest.approx(vals.mean())
    assert row["median"] == pytest.approx(np.median(vals))
    assert row["std"] == pytest.approx(vals.std(ddof=1))
    assert row["p05"] == pytest.approx(np.percentile(vals, 5))
    assert row["p95"] == pytest.approx(np.percentile(vals, 95))


# ---------------------------------------------------------------------------
# perform_jackknife_fit
# ---------------------------------------------------------------------------


def test_jackknife_matches_leave_one_out_loop():
    """The predictions and metrics equal a hand-written LeaveOneOut loop."""
    x, y = _make_data(n_smpls=25)
    est = ExtraTreesRegressor(n_estimators=10, random_state=0)
    out = rsk.perform_jackknife_fit(est, x, y, ["a", "b"], use_pbar=False)
    exp = np.empty_like(y)
    for train_idx, test_idx in sklearn.model_selection.LeaveOneOut().split(x):
        mdl = sklearn.base.clone(est).fit(x[train_idx], y[train_idx])
        exp[test_idx] = mdl.predict(x[test_idx])
    for i, name in enumerate(["a", "b"]):
        r_df = out["residuals"][name]
        assert list(r_df.columns) == ["sample_idx", "y_true", "y_pred"]
        np.testing.assert_allclose(r_df["y_pred"], exp[:, i], rtol=1e-12)
        exp_m = calc_reg_metrics(
            pd.DataFrame({"o": y[:, i], "p": exp[:, i]}), ref_col="o", pred_col="p"
        )
        for key in _METRIC_KEYS:
            assert out["metrics"][name][key] == pytest.approx(exp_m[key], rel=1e-12)
        assert out["metrics"][name]["n_samples"] == 25
    assert out["coef_stats"] is None  # trees have no coefficients


def _ols_parts(x, y):
    """Design matrix, hat matrix, residuals and fitted coefficients of OLS."""
    x1 = np.column_stack([np.ones(x.shape[0]), x])
    beta = np.linalg.lstsq(x1, y, rcond=None)[0]
    hat = x1 @ np.linalg.inv(x1.T @ x1) @ x1.T
    resid = y - x1 @ beta
    return x1, beta, np.diag(hat), resid


def test_jackknife_ols_hat_matrix_and_cooks_distance():
    """OLS leave-one-out predictions equal the closed form; pred_shift gives Cook's D."""
    x, y = _make_data(n_smpls=30)
    y1 = y[:, 0]
    out = rsk.perform_jackknife_fit(LinearRegression(), x, y1, use_pbar=False)
    x1, _, h_ii, resid = _ols_parts(x, y1)
    loo_closed = y1 - resid / (1.0 - h_ii)
    np.testing.assert_allclose(out["residuals"]["target"]["y_pred"], loo_closed, rtol=1e-9)

    n_smpls, n_coefs = x1.shape
    s2 = np.sum(resid**2) / (n_smpls - n_coefs)
    cooks = resid**2 * h_ii / (n_coefs * s2 * (1.0 - h_ii) ** 2)
    infl = out["influence"]["target"]
    assert list(infl.columns) == [
        "sample_idx", "y_true", "y_pred", "delta_cv_rmse", "delta_fit_rmse", "pred_shift_rmse",
    ]
    np.testing.assert_allclose(
        n_smpls * infl["pred_shift_rmse"] ** 2 / (n_coefs * s2), cooks, rtol=1e-7
    )
    # Least squares: refitting without a sample never fits the others worse.
    assert (infl["delta_fit_rmse"] <= 1e-12).all()
    # delta_cv_rmse: the pooled RMSE without the sample, minus with it.
    err = loo_closed - y1
    full = np.sqrt(np.mean(err**2))
    keep = np.arange(n_smpls) != 4
    assert infl.loc[4, "delta_cv_rmse"] == pytest.approx(np.sqrt(np.mean(err[keep] ** 2)) - full)


def test_jackknife_coef_stats_match_direct_calculation():
    """The coefficient bias, SE and interval match a direct jackknife."""
    import scipy.stats  # pylint: disable=import-outside-toplevel

    x, y = _make_data(n_smpls=30)
    x_df = pd.DataFrame(x, columns=["p95", "cover", "skew"])
    out = rsk.perform_jackknife_fit(
        LinearRegression(), x_df, pd.DataFrame(y, columns=["vol", "ba"]), use_pbar=False
    )
    n_smpls = x.shape[0]
    for i, name in enumerate(["vol", "ba"]):
        _, beta, _, _ = _ols_parts(x, y[:, i])
        loo_beta = np.array(
            [
                _ols_parts(np.delete(x, j, axis=0), np.delete(y[:, i], j))[1]
                for j in range(n_smpls)
            ]
        )
        mean_b = loo_beta.mean(axis=0)
        bias = (n_smpls - 1) * (mean_b - beta)
        se = np.sqrt((n_smpls - 1) / n_smpls * np.sum((loo_beta - mean_b) ** 2, axis=0))
        t_crit = scipy.stats.t.ppf(0.975, n_smpls - 1)
        c_df = out["coef_stats"][name]
        assert list(c_df.index) == ["intercept", "p95", "cover", "skew"]
        np.testing.assert_allclose(c_df["estimate"], beta, rtol=1e-9)
        np.testing.assert_allclose(c_df["jack_mean"], mean_b, rtol=1e-9)
        np.testing.assert_allclose(c_df["bias"], bias, rtol=1e-6, atol=1e-10)
        np.testing.assert_allclose(c_df["std_err"], se, rtol=1e-9)
        np.testing.assert_allclose(c_df["jack_estimate"], beta - bias, rtol=1e-9)
        np.testing.assert_allclose(c_df["ci_lower"], beta - bias - t_crit * se, rtol=1e-9)
        np.testing.assert_allclose(c_df["ci_upper"], beta - bias + t_crit * se, rtol=1e-9)


def test_jackknife_pipeline_coefs_and_groups():
    """Pipeline coefficients come from the last step; groups leave out a group at a time."""
    x, y = _make_data(n_smpls=24)
    groups = np.repeat(np.array(["s1", "s2", "s3", "s4", "s5", "s6"]), 4)
    out = rsk.perform_jackknife_fit(
        make_pipeline(StandardScaler(), Ridge(alpha=0.5)), x, y[:, 1],
        groups=groups, use_pbar=False,
    )
    r_df = out["residuals"]["target"]
    assert list(r_df.columns) == ["sample_idx", "group", "y_true", "y_pred"]
    exp = np.empty(24)
    for train_idx, test_idx in sklearn.model_selection.LeaveOneGroupOut().split(x, groups=groups):
        mdl = make_pipeline(StandardScaler(), Ridge(alpha=0.5)).fit(x[train_idx], y[train_idx, 1])
        exp[test_idx] = mdl.predict(x[test_idx])
    np.testing.assert_allclose(r_df["y_pred"], exp, rtol=1e-10)
    infl = out["influence"]["target"]
    assert infl["group"].tolist() == ["s1", "s2", "s3", "s4", "s5", "s6"]
    assert infl["n_samples"].tolist() == [4] * 6
    c_df = out["coef_stats"]["target"]
    assert list(c_df.index) == ["intercept", "x0", "x1", "x2"]
    full = make_pipeline(StandardScaler(), Ridge(alpha=0.5)).fit(x, y[:, 1])
    np.testing.assert_allclose(c_df["estimate"].iloc[1:], full.steps[-1][1].coef_, rtol=1e-10)
    # calc_coef_stats=False skips them.
    out_nc = rsk.perform_jackknife_fit(Ridge(), x, y, calc_coef_stats=False, use_pbar=False)
    assert out_nc["coef_stats"] is None


def test_jackknife_csv_round_trip(tmp_path):
    """The jackknife CSVs read back to the returned tables."""
    x, y = _make_data(n_smpls=15)
    out = rsk.perform_jackknife_fit(
        LinearRegression(), x, y, ["Vol / ha", "BA / ha"], out_dir=tmp_path,
        use_pbar=False,
    )
    for name, stem in (("Vol / ha", "vol_ha"), ("BA / ha", "ba_ha")):
        m_df = pd.read_csv(tmp_path / f"jackknife_metrics_{stem}.csv")
        assert len(m_df) == 1
        for key, val in out["metrics"][name].items():
            assert m_df.loc[0, key] == pytest.approx(val, rel=1e-12)
        for kind in ("residuals", "influence"):
            pd.testing.assert_frame_equal(
                pd.read_csv(tmp_path / f"jackknife_{kind}_{stem}.csv"),
                out[kind][name], check_exact=False, rtol=1e-12,
            )
        c_df = pd.read_csv(tmp_path / f"jackknife_coef_stats_{stem}.csv", index_col="coef")
        pd.testing.assert_frame_equal(c_df, out["coef_stats"][name], check_exact=False, rtol=1e-12)


def test_jackknife_too_few_samples():
    """Fewer than three samples raise RSGISPyException."""
    with pytest.raises(rsgispy.RSGISPyException):
        rsk.perform_jackknife_fit(Ridge(), np.ones((2, 1)), np.ones(2), use_pbar=False)
