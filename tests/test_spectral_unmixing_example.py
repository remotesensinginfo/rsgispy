"""
Run the documented spectral unmixing (TropWet wetland mapping) worked example
(``docs/examples/spectral_unmixing.py``) end-to-end on a synthetic wetland
scene so the code shown in ``docs/guides/spectral_unmixing.rst`` stays
correct, and check the outputs against the known truth.
"""

import importlib.util
import os

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import rasterio  # noqa: E402

from rsgispy.imageutils.colourtable import get_thematic_classes  # noqa: E402

from unmix_scene import (  # noqa: E402
    ENDMEMBERS,
    EM_NAMES,
    N_DATES,
    NODATA_COLS,
    flood_strip_cols,
    make_unmix_scene,
    region_slices,
)

pytest.importorskip("scipy")

pytestmark = pytest.mark.slow

EXAMPLE = os.path.join(
    os.path.dirname(__file__), os.pardir, "docs", "examples", "spectral_unmixing.py"
)


def _load_example():
    spec = importlib.util.spec_from_file_location("spectral_unmixing", EXAMPLE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _read(path):
    with rasterio.open(path) as src:
        return src.read(), list(src.descriptions)


def _inner(name, pad=2):
    """Region slices without the outer ``pad`` pixels (away from edges)."""
    rows, cols = region_slices(name)
    return (slice(rows.start + pad, rows.stop - pad),
            slice(cols.start + pad, cols.stop - pad))


@pytest.fixture(scope="module")
def tropwet(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("tropwet")
    scene = make_unmix_scene(str(tmp_path / "data"))
    out = _load_example().run_workflow(scene["dates"], scene["dem"],
                                       str(tmp_path / "out"), use_pbar=False)
    return scene, out


def test_endmembers(tropwet):
    _, out = tropwet
    for cnt in out["em_counts"].values():
        assert cnt >= 50
    msk, _ = _read(out["em_msk_img"])
    # Pure pixels come from the lake, the forest and the dunes.
    assert np.all(msk[0][_inner("lake")] == 1)
    assert np.all(msk[0][_inner("forest")] == 2)
    assert np.mean(msk[0][_inner("dunes")] == 3) > 0.5
    em_df = out["em_df"]
    assert list(em_df.columns) == EM_NAMES
    for name in EM_NAMES:
        assert np.allclose(em_df[name].to_numpy(), ENDMEMBERS[name], atol=40)
    assert os.path.getsize(out["em_plot"]) > 0


def test_fractions(tropwet):
    scene, out = tropwet
    truth, _ = _read(scene["truth"])
    pct, names = _read(out["pct_img"])
    assert names == EM_NAMES
    known = truth[0] >= 0
    assert np.abs(pct[:, known] / 100.0 - truth[:, known]).mean() < 0.02
    assert np.all(pct[:, :, -NODATA_COLS:] == -999)
    err = out["unmix_err"]
    assert err["fcls"]["neg_pct"] == 0
    assert err["fcls"]["sum_min"] == pytest.approx(1.0, abs=1e-3)
    assert err["ucls"]["neg_pct"] > 0
    # The burned patch is not a mixture of the endmembers: high RMSE.
    rmse = _read(out["fcls_err_img"])[0][0]
    assert rmse[_inner("burn")].min() > 5 * np.median(rmse[known])


def test_burn_correction(tropwet):
    _, out = tropwet
    pct, _ = _read(out["pct_img"])
    corr, _ = _read(out["corr_img"])
    burn, _ = _read(out["burn_img"])
    rows, cols = region_slices("burn")
    assert pct[0, rows, cols].mean() > 50  # looks like water before
    assert np.all(burn[0, rows, cols] == 1)
    assert np.all(corr[0, rows, cols] == 0)  # no water after
    assert np.allclose(corr[:, rows, cols].sum(axis=0), 100, atol=0.01)
    # Genuine wet sand and the lake are not flagged.
    assert burn[0].sum() == burn[0, rows, cols].sum()


def test_classes(tropwet):
    _, out = tropwet
    cls, _ = _read(out["cls_img"])
    expected = {"lake": 1, "efv": 2, "wet_sand": 3, "forest": 4,
                "sparse_veg": 5, "scrub": 5, "dunes": 6, "sand_water": 6,
                "burn": 6, "mixed": 7, "hill": 7}
    for name, code in expected.items():
        assert np.all(cls[0][_inner(name, pad=1)] == code), name
    slope, _ = _read(out["slope_img"])
    assert np.all(slope[0][_inner("hill")] > 2.5)
    assert np.all(slope[0][_inner("efv")] < 2.5)
    assert np.all(cls[0, :, -NODATA_COLS:] == 0)
    with rasterio.open(out["cls_img"]) as src:
        assert src.colormap(1)[1][:3] == (8, 81, 156)
    classes = get_thematic_classes(out["cls_img"])
    assert classes[1][0] == "Open Water (OW)"
    assert out["cls_area"].loc[2, "area_ha"] == pytest.approx(400 * 0.09)
    assert os.path.getsize(out["map_file"]) > 0


def test_water_frequency(tropwet):
    _, out = tropwet
    freq, names = _read(out["freq_img"])
    assert names == ["n_obs", "n_water", "water_freq_pct"]
    rows = _inner("flood")[0]
    for n_flooded in range(1, N_DATES + 1):
        cols = flood_strip_cols(n_flooded)
        assert np.all(freq[2, rows, cols] == 100.0 * n_flooded / N_DATES)
    assert np.all(freq[2][_inner("lake")] == 100)
    assert np.all(freq[0, 0:10, 0:10] == N_DATES - 1)  # cloud on one date
    assert np.all(freq[2][_inner("efv")] == 0)
    assert np.all(freq[2][_inner("burn")] == 0)
    assert np.all(freq[2, :, -NODATA_COLS:] == -1)
