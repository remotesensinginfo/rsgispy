"""Window mean/variance is computed in float64 whatever the input dtype."""

import numpy as np

from rsgispy.imagefilter._helpers import _window_mean_var


def test_float32_matches_float64():
    rng = np.random.default_rng(0)
    wins = (rng.random((20, 20, 7, 7)) * 1000 + 5000).astype(np.float32)
    valid = rng.random(wins.shape) > 0.1
    mean32, var32, n32 = _window_mean_var(wins, valid)
    mean64, var64, n64 = _window_mean_var(wins.astype(np.float64), valid)
    np.testing.assert_array_equal(n32, n64)
    np.testing.assert_array_equal(mean32, mean64)
    np.testing.assert_array_equal(var32, var64)
