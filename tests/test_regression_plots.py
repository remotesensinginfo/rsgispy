"""
Tests for the regression diagnostic plots: :func:`plot_residuals` (with the
residual histogram beside the y-axis), :func:`plot_qq_residuals`,
:func:`plot_pred_vs_obs` and the generic
:func:`rsgispy.tools.plotting.plot_qq`.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import rsgispy
import rsgispy.regression.regmetrics as rm
import rsgispy.tools.plotting as rplot

matplotlib.use("Agg")
scipy_stats = pytest.importorskip("scipy.stats")
pytest.importorskip("sklearn")


@pytest.fixture(name="eval_df")
def _eval_df():
    rng = np.random.default_rng(7)
    obs = rng.uniform(0.0, 100.0, 120)
    return pd.DataFrame({
        "obs": obs,
        "pred": obs + rng.normal(1.0, 5.0, obs.size),
        "obs_h": obs / 3.0,
        "pred_h": obs / 3.0 + rng.normal(-0.5, 2.0, obs.size),
    })


def _scatter_offsets(ax):
    return np.asarray(ax.collections[0].get_offsets())


def test_plot_residuals_single_target(eval_df):
    """Scatter of predicted - observed against predicted, histogram sharing the y-axis."""
    fig, axs = rm.plot_residuals(eval_df, ref_col="obs", pred_col="pred", hist_bins=12)
    assert axs.shape == (1, 2)
    sc_ax, h_ax = axs[0]
    resid = eval_df["pred"].to_numpy() - eval_df["obs"].to_numpy()

    pts = _scatter_offsets(sc_ax)
    np.testing.assert_allclose(pts[:, 0], eval_df["pred"].to_numpy())
    np.testing.assert_allclose(pts[:, 1], resid)
    assert "predicted − observed" in sc_ax.get_ylabel()

    # The histogram counts equal numpy.histogram of the residuals.
    exp_counts, exp_edges = np.histogram(resid, bins=12)
    bar_widths = np.array([p.get_width() for p in h_ax.patches])
    bar_bottoms = np.array([p.get_y() for p in h_ax.patches])
    np.testing.assert_array_equal(bar_widths, exp_counts)
    np.testing.assert_allclose(bar_bottoms, exp_edges[:-1])

    # The histogram axis shares the scatter's y-axis.
    assert h_ax.get_shared_y_axes().joined(sc_ax, h_ax)
    assert h_ax.get_ylim() == sc_ax.get_ylim()

    # The normal curve is drawn, scaled to the counts.
    assert any(line.get_label() == "Normal" for line in h_ax.get_lines())
    plt.close(fig)


def test_plot_residuals_options(eval_df):
    """x_axis='obs', no histogram, no stats box."""
    fig, axs = rm.plot_residuals(
        eval_df, ref_col="obs", pred_col="pred", x_axis="obs", show_hist=False,
        show_stats=False,
    )
    assert axs.shape == (1, 1)
    np.testing.assert_allclose(_scatter_offsets(axs[0, 0])[:, 0], eval_df["obs"].to_numpy())
    assert axs[0, 0].get_xlabel().startswith("Observed")
    assert not axs[0, 0].artists
    plt.close(fig)
    with pytest.raises(rsgispy.RSGISPyException):
        rm.plot_residuals(eval_df, ref_col="obs", pred_col="pred", x_axis="other")


def test_plot_residuals_multi_target_and_given_axes(eval_df):
    """One row per target; given axes are used without a new figure."""
    kwargs = {"ref_cols": ["obs", "obs_h"], "pred_cols": ["pred", "pred_h"],
              "target_names": ["Cover", "Height"]}
    fig, axs = rm.plot_residuals(eval_df, **kwargs)
    assert axs.shape == (2, 2)
    assert [axs[i, 0].get_title() for i in range(2)] == ["Cover", "Height"]
    resid_h = eval_df["pred_h"].to_numpy() - eval_df["obs_h"].to_numpy()
    np.testing.assert_allclose(_scatter_offsets(axs[1, 0])[:, 1], resid_h)
    plt.close(fig)

    plt.close("all")
    given_fig, given_axs = plt.subplots(2, 2)
    fig2, axs2 = rm.plot_residuals(eval_df, axs=given_axs, **kwargs)
    assert fig2 is given_fig and plt.get_fignums() == [given_fig.number]
    assert axs2[1, 1].get_shared_y_axes().joined(axs2[1, 0], axs2[1, 1])
    plt.close(given_fig)


def test_plot_qq_matches_probplot_and_shapiro():
    """The Q-Q points and line match scipy.stats.probplot; the Shapiro text is correct."""
    rng = np.random.default_rng(3)
    vals = rng.normal(5.0, 2.0, 150)
    fig, ax = rplot.plot_qq(vals, title="Normal values")
    (osm, osr), (slope, intercept, _) = scipy_stats.probplot(vals, dist="norm", fit=True)
    pts = _scatter_offsets(ax)
    np.testing.assert_allclose(pts[:, 0], osm)
    np.testing.assert_allclose(pts[:, 1], osr)
    fit_line = [ln for ln in ax.get_lines() if ln.get_label() == "Fitted line"][0]
    x_line, y_line = fit_line.get_data()
    np.testing.assert_allclose(y_line, slope * np.asarray(x_line) + intercept)
    sw_stat, sw_p = scipy_stats.shapiro(vals)
    box_txt = ax.artists[0].txt.get_text()
    assert f"W = {sw_stat:.3f}" in box_txt
    assert ("not normal" in box_txt) == (sw_p < 0.05)
    assert ax.get_xlabel() == "Theoretical quantiles (normal)"
    plt.close(fig)

    # Clearly skewed values are flagged as not normal.
    fig, ax = rplot.plot_qq(rng.exponential(1.0, 300))
    assert "p < 0.05: not normal" in ax.artists[0].txt.get_text()
    plt.close(fig)

    # Other distributions: no Shapiro-Wilk box.
    fig, ax = rplot.plot_qq(rng.uniform(size=50), dist="uniform")
    assert not ax.artists
    assert "uniform" in ax.get_xlabel()
    plt.close(fig)

    with pytest.raises(rsgispy.RSGISPyException):
        rplot.plot_qq([1.0])


def test_describe_shapiro():
    """p < 0.05 means not normal; otherwise no evidence against normality."""
    assert "not normal" in rplot.describe_shapiro(0.9, 0.01)
    assert "no evidence against normality" in rplot.describe_shapiro(0.99, 0.4)


def test_plot_qq_residuals(eval_df):
    """The residual Q-Q plot uses predicted - observed, one panel per target."""
    fig, axs = rm.plot_qq_residuals(eval_df, ref_col="obs", pred_col="pred")
    assert axs.shape == (1, 1)
    resid = eval_df["pred"].to_numpy() - eval_df["obs"].to_numpy()
    (_, osr), _ = scipy_stats.probplot(resid, dist="norm", fit=True)
    np.testing.assert_allclose(_scatter_offsets(axs[0, 0])[:, 1], osr)
    plt.close(fig)

    fig, axs = rm.plot_qq_residuals(
        eval_df, ref_cols=["obs", "obs_h"], pred_cols=["pred", "pred_h"],
        target_names=["Cover", "Height"], show_shapiro=False,
    )
    assert axs.size == 2 and axs[0, 1].get_title() == "Height"
    assert not axs[0, 0].artists
    plt.close(fig)


def test_plot_pred_vs_obs_stats_line_and_ranges(eval_df):
    """1:1 line, fitted line with its equation, R2/RMSE/bias/n box and equal ranges."""
    fig, axs = rm.plot_pred_vs_obs(eval_df, ref_col="obs", pred_col="pred")
    ax = axs[0, 0]
    metrics = rm.calc_reg_metrics(eval_df, ref_col="obs", pred_col="pred")
    labels = [ln.get_label() for ln in ax.get_lines()]
    assert "1:1 line" in labels
    fit_lbl = [lbl for lbl in labels if lbl.startswith("Fit: y = ")]
    slope, _ = np.polyfit(eval_df["obs"], eval_df["pred"], 1)
    assert fit_lbl and f"{slope:.3g}x" in fit_lbl[0]
    box_txt = ax.artists[0].txt.get_text()
    assert f"R² = {metrics['r2']:.3f}" in box_txt
    assert f"n = {metrics['n_samples']}" in box_txt
    assert ax.get_xlim() == ax.get_ylim()
    plt.close(fig)

    fig, axs = rm.plot_pred_vs_obs(
        eval_df, ref_cols=["obs", "obs_h"], pred_cols=["pred", "pred_h"],
        target_names=["Cover", "Height"], add_trendline=False, show_stats=False,
    )
    assert axs.shape == (1, 2)
    assert not [ln for ln in axs[0, 1].get_lines() if ln.get_label().startswith("Fit")]
    assert not axs[0, 1].artists
    plt.close(fig)


def test_pred_vs_obs_and_qq_use_callers_axes_layout():
    """Four targets can be drawn on a caller's 2 x 2 grid."""
    import matplotlib.pyplot as plt
    import pandas as pd
    import rsgispy.regression.regmetrics as rm

    rng = np.random.default_rng(5)
    data = {}
    for idx in range(4):
        obs = rng.normal(10, 2, 50)
        data[f"obs{idx}"] = obs
        data[f"pred{idx}"] = obs + rng.normal(0, 0.5, 50)
    df = pd.DataFrame(data)
    cols = dict(ref_cols=[f"obs{i}" for i in range(4)], pred_cols=[f"pred{i}" for i in range(4)])
    for func in (rm.plot_pred_vs_obs, rm.plot_qq_residuals):
        fig, grid = plt.subplots(2, 2)
        out_fig, axes = func(df, axs=grid, **cols)
        assert out_fig is fig and axes.shape == (2, 2)
        assert all(ax.has_data() for ax in grid.ravel())
        plt.close(fig)
