"""The SALib Sobol sensitivity analysis is repeatable with ``rnd_seed``, and its
CSV outputs keep the variable names."""

import numpy as np
import pandas as pd
import pytest

from rsgispy.tools._ml_explain import run_salib_sobol_sensitivity

pytest.importorskip("SALib")
pytest.importorskip("sklearn")
from sklearn.linear_model import LogisticRegression  # noqa: E402


def _run(rnd_seed):
    rng = np.random.default_rng(0)
    train_df = pd.DataFrame(rng.random((50, 3)), columns=["a", "b", "c"])

    def model(x_df):
        return x_df["a"] * 2 + x_df["b"] ** 2 + 0.1 * x_df["c"]

    return run_salib_sobol_sensitivity(
        train_df, ["a", "b", "c"], model, "test", "modelling", n_samples=64,
        rnd_seed=rnd_seed,
    )


def test_same_seed_same_indices():
    first = _run(42)
    second = _run(42)
    for df_1, df_2 in zip(first, second):
        pd.testing.assert_frame_equal(df_1, df_2)


def test_different_seeds_differ():
    total_1 = _run(1)[0]
    total_2 = _run(2)[0]
    assert not np.allclose(total_1["ST"], total_2["ST"])


def _correlated_logreg():
    rng = np.random.default_rng(0)
    a = rng.normal(size=300)
    b = a + 0.1 * rng.normal(size=300)
    x_df = pd.DataFrame({"a": a, "b": b})
    y = (a + 0.3 * rng.normal(size=300) > 0).astype(int)
    mdl = LogisticRegression().fit(x_df, y)
    mdl.coef_[0, 1] = 0.0  # b has no effect on the model
    return mdl, x_df, y


def test_sobol_csvs_keep_variable_names(tmp_path):
    pytest.importorskip("SALib")
    import geopandas as gpd

    import rsgispy.modelling.species_distribution as sdm

    mdl, x_df, _ = _correlated_logreg()
    x_df["c"] = np.random.default_rng(1).normal(size=len(x_df))
    mdl = LogisticRegression().fit(x_df, (x_df["a"] > 0).astype(int))
    gdf = gpd.GeoDataFrame(
        x_df, geometry=gpd.points_from_xy(x_df["a"], x_df["b"]), crs="EPSG:27700"
    )
    files = {k: tmp_path / f"sobol_{k}.csv" for k in ("total", "first", "second")}
    total_si, first_si, second_si = sdm.salib_sklearn_mdl_sensitivity(
        mdl, gdf, None, ["a", "b", "c"], n_samples=64,
        sobol_overall_file=str(files["total"]),
        sobol_first_file=str(files["first"]),
        sobol_second_file=str(files["second"]),
    )
    total_csv = pd.read_csv(files["total"])
    first_csv = pd.read_csv(files["first"])
    second_csv = pd.read_csv(files["second"])
    assert list(total_csv["variable"]) == ["a", "b", "c"]
    assert list(first_csv["variable"]) == ["a", "b", "c"]
    assert {"ST", "ST_conf"} <= set(total_csv.columns)
    assert {"S1", "S1_conf"} <= set(first_csv.columns)
    pairs = list(zip(second_csv["variable_1"], second_csv["variable_2"]))
    assert pairs == [("a", "b"), ("a", "c"), ("b", "c")]
    np.testing.assert_allclose(total_csv["ST"], total_si["ST"].to_numpy())
    assert total_si.loc["a", "ST"] > total_si.loc["b", "ST"]
    assert len(second_si) == 3
