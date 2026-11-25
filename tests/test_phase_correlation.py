"""
Accuracy tests for :func:`rsgispy.tools.stats.phase_cross_correlation_2d`
and the phase-correlation tie points of :mod:`rsgispy.imageregistration`.

The shifts follow the function's convention: a feature at ``(r, c)`` in the
reference window appears at ``(r - dy, c - dx)`` in the floating window, i.e.
``flt(y, x) = ref(y + dy, x + dx)``.
"""

import os

import numpy as np
import pytest

import rsgispy
import rsgispy.imageregistration as imreg
import rsgispy.imageregistration.tiepoints as imreg_tp
import rsgispy.tools.stats as rstats

from coreg_scene import SHIFT, make_coreg_scene

FIELD = 256
WINDOWS = (32, 64, 128)


def _smooth_field(seed: int, sigma: float) -> np.ndarray:
    """Spectrum of a periodic Gaussian-smoothed random field."""
    rng = np.random.default_rng(seed)
    freq = np.fft.fftfreq(FIELD)
    filt = np.exp(-2.0 * (np.pi * sigma) ** 2 * (freq[:, None] ** 2 + freq[None, :] ** 2))
    return np.fft.fft2(rng.normal(size=(FIELD, FIELD))) * filt


def _smooth_pair(spec: np.ndarray, win: int, dy: float, dx: float):
    """Reference and (Fourier, i.e. exactly) shifted floating windows."""
    freq = np.fft.fftfreq(FIELD)
    ramp = np.exp(2j * np.pi * (freq[:, None] * dy + freq[None, :] * dx))
    ref = np.real(np.fft.ifft2(spec))[60:60 + win, 70:70 + win]
    flt = np.real(np.fft.ifft2(spec * ramp))[60:60 + win, 70:70 + win]
    return ref, flt


def _analytic(win: int, dy: float, dx: float) -> np.ndarray:
    """Analytic texture (sum of sinusoids) sampled at shifted coordinates."""
    yy, xx = np.mgrid[0:win, 0:win].astype(np.float64)
    g_x = xx + 17.0 + dx
    g_y = yy + 11.0 + dy
    return (
        np.sin(g_x / 2.2) * np.cos(g_y / 2.5)
        + 0.7 * np.cos((g_x + 0.8 * g_y) / 1.7)
        + 0.5 * np.sin((g_x - 1.3 * g_y) / 1.4)
    )


def _shifts(win: int):
    """Integer and sub-pixel shifts up to a quarter of the window."""
    qtr = win / 4.0
    return [
        (0.0, 0.0), (2.0, 1.0), (2.0, 3.0), (5.0, 5.0), (0.4, -1.3), (-3.7, 2.2),
        (0.25, -0.25), (qtr, -qtr), (-qtr + 0.3, qtr - 0.6), (qtr - 0.5, qtr - 0.25),
    ]


@pytest.mark.parametrize("win", WINDOWS)
@pytest.mark.parametrize("sigma", [1.5, 3.0])
def test_phase_corr_smooth_random_texture(win, sigma):
    """Noise-free smoothed random textures: error <= 0.05 px (measured <= 0.01 px)."""
    spec = _smooth_field(seed=int(sigma * 10) + win, sigma=sigma)
    for dy, dx in _shifts(win):
        ref, flt = _smooth_pair(spec, win, dy, dx)
        est_y, est_x, peak = rstats.phase_cross_correlation_2d(ref, flt, upsample_factor=100)
        assert est_y == pytest.approx(dy, abs=0.05), (win, sigma, dy, dx, est_y, est_x)
        assert est_x == pytest.approx(dx, abs=0.05), (win, sigma, dy, dx, est_y, est_x)
        assert peak > 0.9


@pytest.mark.parametrize("win", WINDOWS)
def test_phase_corr_analytic_texture(win):
    """Noise-free analytic texture (three sinusoids, so nearly periodic: tested
    with shifts below a quarter of its shortest period)."""
    for dy, dx in [(0.0, 0.0), (2.0, 1.0), (2.0, -3.0), (0.4, -1.3), (-1.7, 2.6), (1.0, -2.0)]:
        ref = _analytic(win, 0.0, 0.0)
        flt = _analytic(win, dy, dx)
        est_y, est_x, peak = rstats.phase_cross_correlation_2d(ref, flt, upsample_factor=100)
        assert est_y == pytest.approx(dy, abs=0.05), (win, dy, dx, est_y, est_x)
        assert est_x == pytest.approx(dx, abs=0.05), (win, dy, dx, est_y, est_x)
        assert peak > 0.9


@pytest.mark.parametrize("win", WINDOWS)
def test_phase_corr_default_precision(win):
    """With the default upsample_factor=4 the shift is found to 0.25 px."""
    spec = _smooth_field(seed=3, sigma=2.0)
    for dy, dx in [(2.0, 1.0), (2.0, 3.0), (0.4, -1.3), (-3.7, 2.2)]:
        ref, flt = _smooth_pair(spec, win, dy, dx)
        est_y, est_x, _ = rstats.phase_cross_correlation_2d(ref, flt)
        assert est_y == pytest.approx(dy, abs=0.13)
        assert est_x == pytest.approx(dx, abs=0.13)
        # The estimate is a multiple of 1 / upsample_factor
        assert (est_y * 4) == pytest.approx(round(est_y * 4))
        assert (est_x * 4) == pytest.approx(round(est_x * 4))


def test_phase_corr_noise_and_edge_cases():
    spec = _smooth_field(seed=5, sigma=2.0)
    rng = np.random.default_rng(1)
    ref, flt = _smooth_pair(spec, 64, 2.4, -1.6)
    noise_sd = 0.1 * ref.std()
    est_y, est_x, peak = rstats.phase_cross_correlation_2d(
        ref + rng.normal(0, noise_sd, ref.shape), flt + rng.normal(0, noise_sd, flt.shape),
        upsample_factor=20,
    )
    assert est_y == pytest.approx(2.4, abs=0.2)
    assert est_x == pytest.approx(-1.6, abs=0.2)
    assert 0.5 < peak <= 1.0 + 1e-9

    # Unrelated windows give a lower peak than matching windows
    other, _ = _smooth_pair(_smooth_field(seed=6, sigma=2.0), 64, 0.0, 0.0)
    assert rstats.phase_cross_correlation_2d(ref, other)[2] < peak

    # Constant or tiny windows
    assert rstats.phase_cross_correlation_2d(np.ones((16, 16)), np.ones((16, 16))) == (0.0, 0.0, 0.0)
    assert rstats.phase_cross_correlation_2d(np.ones((3, 3)), np.ones((3, 3))) == (0.0, 0.0, 0.0)
    # Non-square windows and no taper
    ref_r, flt_r = _smooth_pair(spec, 64, 1.0, -2.0)
    est = rstats.phase_cross_correlation_2d(ref_r[:40], flt_r[:40], upsample_factor=10)
    assert est[0] == pytest.approx(1.0, abs=0.1) and est[1] == pytest.approx(-2.0, abs=0.1)
    est = rstats.phase_cross_correlation_2d(ref_r, flt_r, upsample_factor=1)
    assert est[:2] == (1.0, -2.0)
    # Without the taper the edges of the windows bias the estimate
    est = rstats.phase_cross_correlation_2d(ref_r, flt_r, upsample_factor=4, apply_window=False)
    assert est[0] == pytest.approx(1.0, abs=1.0) and est[1] == pytest.approx(-2.0, abs=1.0)


@pytest.mark.parametrize("win", [32, 64])
def test_phase_corr_tie_points_coreg_scene(tmp_path, win):
    """The co-registration guide's test pair: the mean tie point offset is the
    simulated shift (it was -13.7 m instead of -23 m with 32 px windows)."""
    scene = make_coreg_scene(str(tmp_path))
    tie_df = imreg_tp.generate_tie_points(
        scene["ref"],
        scene["flt"],
        method=imreg.TIE_METHOD_PHASE_CORR,
        ref_band=2,
        flt_band=2,
        pixel_gap=24,
        win_size=win,
        search_area=6,
        use_pbar=False,
    )
    assert len(tie_df) >= 10
    assert tie_df["dx_map"].median() == pytest.approx(-SHIFT[0], abs=2.5)
    assert tie_df["dy_map"].median() == pytest.approx(-SHIFT[1], abs=2.5)
    assert os.path.exists(scene["ref"])


def test_phase_corr_similarity_metric():
    spec = _smooth_field(seed=8, sigma=2.0)
    ref, flt = _smooth_pair(spec, 48, 1.5, 0.5)
    assert rstats.calc_similarity_metric(ref, ref, rsgispy.METRIC_PHASE_CORR) == pytest.approx(1.0)
    assert rstats.calc_similarity_metric(ref, flt, rsgispy.METRIC_PHASE_CORR) > 0.9
