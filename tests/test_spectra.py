"""
Tests for rsgispy.imagespectral.spectra: continuum removal of a spectrum and
an image, and the spectra and continuum removal plots.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import rasterio  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402

import rsgispy  # noqa: E402
from rsgispy._numba import disable_numba  # noqa: E402
import rsgispy.imagespectral.spectra as spectra  # noqa: E402

WVLS = np.arange(400.0, 801.0, 10.0)


def _veg_spectrum(depth=0.6, centre=670.0, width=40.0):
    """A sloping continuum with one Gaussian absorption feature."""
    continuum = 0.1 + 0.0005 * (WVLS - 400.0)
    absorb = 1.0 - depth * np.exp(-0.5 * ((WVLS - centre) / width) ** 2)
    return continuum * absorb


def _cr_reference(spec, wvls):
    """Continuum removal from scipy's convex hull (upper part)."""
    from scipy.spatial import ConvexHull  # pylint: disable=import-outside-toplevel

    pts = np.column_stack([wvls, spec])
    # Close the hull below the spectrum so only the upper hull is above it.
    low = spec.min() - 1.0
    pts_all = np.vstack([pts, [[wvls[0], low], [wvls[-1], low]]])
    hull = ConvexHull(pts_all)
    idx = sorted(i for i in hull.vertices if i < len(wvls))
    return spec / np.interp(wvls, wvls[idx], spec[idx])


def test_calc_continuum_removal_single_feature():
    spec = _veg_spectrum()
    cr_info = spectra.calc_continuum_removal(spec, WVLS)
    assert cr_info["crs"] == pytest.approx(_cr_reference(spec, WVLS), abs=1e-12)
    assert np.all(cr_info["crs"] <= 1.0 + 1e-12)
    assert cr_info["hull_x"][0] == 400.0 and cr_info["hull_x"][-1] == 800.0
    assert cr_info["continuum"] * cr_info["crs"] == pytest.approx(spec)
    assert len(cr_info["features"]) == 1
    feat = cr_info["features"][0]
    assert feat["id"] == 1
    assert feat["abs_wvl"] == 670.0
    assert feat["abs_depth"] == pytest.approx(cr_info["crs"].min())
    # The FWHM is around the minimum and its level is half way to 1.
    assert feat["FWHM_x"][0] < 670.0 < feat["FWHM_x"][1]
    assert feat["FWHM_y"][0] == pytest.approx(feat["abs_depth"] + (1 - feat["abs_depth"]) / 2)
    assert feat["FWHM_delta"] == pytest.approx(feat["FWHM_x"][1] - feat["FWHM_x"][0])
    area = np.sum(np.diff(feat["wvl"]) * (2 - feat["crs"][1:] - feat["crs"][:-1]) / 2)
    assert feat["area"] == pytest.approx(area)
    h_idx = list(cr_info["hull_x"]).index(feat["cstart_wvl"])
    assert cr_info["hull_x"][h_idx + 1] == feat["cstop_wvl"]
    assert feat["cslope"] == pytest.approx(
        np.diff(cr_info["hull_y"])[h_idx] / np.diff(cr_info["hull_x"])[h_idx]
    )


def test_calc_continuum_removal_range_and_baseline():
    # A concave continuum (so it touches the hull between the features) with
    # absorption features at 480 and 670 nm.
    continuum = 0.4 - 1.5e-6 * (WVLS - 600.0) ** 2
    spec = continuum * (
        1.0
        - 0.3 * np.exp(-0.5 * ((WVLS - 480.0) / 15.0) ** 2)
        - 0.6 * np.exp(-0.5 * ((WVLS - 670.0) / 20.0) ** 2)
    )
    cr_all = spectra.calc_continuum_removal(spec, WVLS)
    assert len(cr_all["features"]) == 2
    cr_sub = spectra.calc_continuum_removal(spec, WVLS, start_wvl=530, stop_wvl=750)
    assert cr_sub["wvl"][0] == 530.0 and cr_sub["wvl"][-1] == 750.0
    assert len(cr_sub["features"]) == 1
    # Only features deeper than the baseline (CRS below 0.5).
    cr_deep = spectra.calc_continuum_removal(spec, WVLS, baseline=0.5)
    assert [f["abs_wvl"] for f in cr_deep["features"]] == [670.0]
    # A convex spectrum has no features.
    flat = spectra.calc_continuum_removal(0.5 - 1e-6 * (WVLS - 600.0) ** 2, WVLS)
    assert flat["features"] == [] and np.allclose(flat["crs"], 1.0)


def test_calc_continuum_removal_checks():
    with pytest.raises(rsgispy.RSGISPyException):
        spectra.calc_continuum_removal([1.0, 2.0], [1.0, 2.0, 3.0])
    with pytest.raises(rsgispy.RSGISPyException):
        spectra.calc_continuum_removal([1.0, 2.0, 3.0], [3.0, 2.0, 1.0])
    with pytest.raises(rsgispy.RSGISPyException):
        spectra.calc_continuum_removal([1.0, np.nan, 3.0], [1.0, 2.0, 3.0])
    with pytest.raises(rsgispy.RSGISPyException):
        spectra.calc_continuum_removal(_veg_spectrum(), WVLS, start_wvl=900)


@pytest.fixture(name="spec_img")
def fixture_spec_img(tmp_path):
    """A 15 x 12 image of varied spectra (41 bands) with no data pixels."""
    rng = np.random.default_rng(3)
    n_rows, n_cols = 15, 12
    depths = rng.uniform(0.1, 0.8, (n_rows, n_cols))
    centres = rng.uniform(600.0, 720.0, (n_rows, n_cols))
    cube = np.empty((WVLS.size, n_rows, n_cols), dtype=np.float32)
    for r_idx in range(n_rows):
        for c_idx in range(n_cols):
            cube[:, r_idx, c_idx] = _veg_spectrum(depths[r_idx, c_idx], centres[r_idx, c_idx])
    cube += rng.normal(0.0, 0.002, cube.shape).astype(np.float32)
    cube[:, 0, 0] = 0.0
    cube[15, 3, 4] = 0.0
    img = os.path.join(tmp_path, "spec.tif")
    profile = {
        "driver": "GTiff", "width": n_cols, "height": n_rows, "count": WVLS.size,
        "dtype": "float32", "nodata": 0.0, "crs": "EPSG:32630",
        "transform": from_origin(500000.0, 5200000.0, 10.0, 10.0),
    }
    with rasterio.open(img, "w", **profile) as dst:
        dst.write(cube)
    return img, cube


def test_calc_continuum_removal_img(spec_img, tmp_path):
    img, cube = spec_img
    out_img = os.path.join(tmp_path, "cr.tif")
    spectra.calc_continuum_removal_img(
        img, out_img, WVLS, start_wvl=500, stop_wvl=780, use_pbar=False
    )
    keep = (WVLS >= 500) & (WVLS <= 780)
    with rasterio.open(out_img) as src:
        out = src.read()
        assert src.nodata == -9999.0
        assert src.descriptions[0] == "500" and src.descriptions[-1] == "780"
    assert out.shape == (int(keep.sum()), 15, 12)
    assert np.all(out[:, 0, 0] == -9999.0)
    assert np.all(out[:, 3, 4] == -9999.0)  # no data in one band of the range
    for r_idx, c_idx in [(1, 1), (7, 5), (14, 11)]:
        ref = spectra.calc_continuum_removal(cube[:, r_idx, c_idx], WVLS, 500, 780)["crs"]
        assert out[:, r_idx, c_idx] == pytest.approx(ref, abs=1e-6)


def test_calc_continuum_removal_img_numba_and_fallback_agree(spec_img, tmp_path):
    img, _ = spec_img
    out_a = os.path.join(tmp_path, "cr_a.tif")
    out_b = os.path.join(tmp_path, "cr_b.tif")
    spectra.calc_continuum_removal_img(img, out_a, WVLS, use_pbar=False)
    with disable_numba():
        spectra.calc_continuum_removal_img(img, out_b, WVLS, use_pbar=False)
    with rasterio.open(out_a) as src_a, rasterio.open(out_b) as src_b:
        np.testing.assert_allclose(src_a.read(), src_b.read(), rtol=0, atol=1e-7)


def test_calc_continuum_removal_img_checks(spec_img, tmp_path):
    img, _ = spec_img
    with pytest.raises(rsgispy.RSGISPyException):
        spectra.calc_continuum_removal_img(
            img, os.path.join(tmp_path, "x.tif"), WVLS[:-1], use_pbar=False
        )


def test_plot_continuum_removal(tmp_path):
    cr_info = spectra.calc_continuum_removal(_veg_spectrum(), WVLS)
    out_file = os.path.join(tmp_path, "cr.png")
    fig, axs = spectra.plot_continuum_removal(cr_info, title="Veg", out_plot_file=out_file)
    assert os.path.isfile(out_file)
    assert len(axs) == 2
    assert [t.get_text() for t in axs[1].get_legend().get_texts()] == [
        "Continuum removed", "Depth", "FWHM"
    ]
    plt.close(fig)
    fig, axs = spectra.plot_continuum_removal(cr_info, show_features=False)
    assert len(axs) == 1
    plt.close(fig)


def test_plot_spectra_inputs():
    df = pd.DataFrame({"WL": WVLS, "A": _veg_spectrum(), "B": _veg_spectrum(0.3)})
    fig, ax = spectra.plot_spectra(df, wavelengths="WL", gain=0.01)
    assert [line.get_label() for line in ax.get_lines()] == ["A", "B"]
    assert ax.get_lines()[0].get_xdata() == pytest.approx(WVLS)
    assert ax.get_lines()[0].get_ydata() == pytest.approx(_veg_spectrum() * 100)
    plt.close(fig)

    fig, ax = spectra.plot_spectra(df.set_index("WL"))
    assert ax.get_lines()[1].get_xdata() == pytest.approx(WVLS)
    plt.close(fig)

    fig, ax = spectra.plot_spectra({"Pixel": _veg_spectrum()}, wavelengths=WVLS,
                                   colours=["red"], title="Pixel")
    assert ax.get_lines()[0].get_color() == "red"
    assert ax.get_title() == "Pixel"
    plt.close(fig)

    arr = np.column_stack([_veg_spectrum(d) for d in np.linspace(0.1, 0.8, 12)])
    fig, ax = spectra.plot_spectra(arr, WVLS, labels=[str(i) for i in range(12)])
    assert len(ax.get_lines()) == 12
    plt.close(fig)

    with pytest.raises(rsgispy.RSGISPyException):
        spectra.plot_spectra(arr, WVLS[:-1])
    with pytest.raises(rsgispy.RSGISPyException):
        spectra.plot_spectra(arr, WVLS, labels=["a"])
    plt.close("all")
