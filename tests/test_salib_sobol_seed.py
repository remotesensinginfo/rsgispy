"""The SALib Sobol sensitivity analysis is repeatable with ``rnd_seed``."""

import numpy as np
import pandas as pd
import pytest

from rsgispy.tools._ml_explain import run_salib_sobol_sensitivity

pytest.importorskip("SALib")


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
