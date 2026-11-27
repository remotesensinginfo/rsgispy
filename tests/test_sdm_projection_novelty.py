"""
Tests for ensemble uncertainty bands, multi-band raster prediction, MESS /
ExDet novelty maps, scenario variable copying and range-change maps in
:mod:`rsgispy.modelling.species_distribution`.
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

import rsgispy
import rsgispy.modelling.species_distribution as sdm
from rsgispy.imageutils.colourtable import get_thematic_classes

pytest.importorskip("sklearn")
from sklearn.ensemble import IsolationForest  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

CRS = "EPSG:32630"
X0, Y0, RES, SIZE = 400000.0, 5800000.0, 100.0, 40


def _write(path, arr, nodata=None, res=RES):
    arr = np.asarray(arr)
    with rasterio.open(
        path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
        dtype=arr.dtype, crs=CRS, transform=from_origin(X0, Y0, res, res),
        nodata=nodata,
    ) as dst:
        dst.write(arr[np.newaxis])
    return str(path)


def _read_all(path):
    with rasterio.open(path) as src:
        return src.read(), src.descriptions, src.nodata


def _scene(tmp_path):
    """v1 rises 0..10 down the rows, v2 rises 0..5 across the columns."""
    rows, cols = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    v1 = rows / (SIZE - 1) * 10.0
    v2 = cols / (SIZE - 1) * 5.0
    msk = np.zeros((SIZE, SIZE), np.uint8)
    msk[1:-1, 1:-1] = 1
    env_vars = {
        "v1": sdm.EnvVarInfo(name="v1", file=_write(tmp_path / "v1.tif", v1, -9999.0)),
        "v2": sdm.EnvVarInfo(name="v2", file=_write(tmp_path / "v2.tif", v2, -9999.0)),
    }
    return env_vars, _write(tmp_path / "mask.tif", msk), msk, v1, v2


def _models(n_models=4, seed=0):
    rng = np.random.default_rng(seed)
    mdls = []
    for _ in range(n_models):
        x_df = pd.DataFrame({"v1": rng.uniform(0, 10, 150), "v2": rng.uniform(0, 5, 150)})
        y = (x_df["v1"] + 2 * x_df["v2"] + rng.normal(0, 3, 150) > 10).astype(int)
        mdls.append(LogisticRegression().fit(x_df, y))
    return mdls


# --------------------------------------------------------------------------
# Ensemble uncertainty bands
# --------------------------------------------------------------------------


def test_ensemble_uncertainty_bands(tmp_path):
    env_vars, msk_img, msk, _, _ = _scene(tmp_path)
    mdls = _models()
    kwargs = dict(use_pbar=False, calc_stats=False)
    plain = str(tmp_path / "plain.tif")
    unc = str(tmp_path / "unc.tif")
    sdm.pred_ensemble_sklearn_mdls_prob(mdls, msk_img, 1, env_vars, ["v1", "v2"], plain, **kwargs)
    sdm.pred_ensemble_sklearn_mdls_prob(
        mdls, msk_img, 1, env_vars, ["v1", "v2"], unc,
        uncertainty=["std", "iqr", "min", "max", "cv"], **kwargs,
    )
    plain_arr, plain_desc, _ = _read_all(plain)
    unc_arr, unc_desc, nodata = _read_all(unc)
    assert plain_arr.shape[0] == 1
    assert unc_arr.shape[0] == 6
    assert unc_desc == ("median", "std", "iqr", "min", "max", "cv")
    np.testing.assert_array_equal(unc_arr[0], plain_arr[0])

    member_arrs = []
    for idx, mdl in enumerate(mdls):
        out = str(tmp_path / f"m{idx}.tif")
        sdm.pred_sklearn_mdl_prob(mdl, msk_img, 1, env_vars, ["v1", "v2"], out, **kwargs)
        member_arrs.append(_read_all(out)[0][0].astype(np.float64))
    stack = np.stack(member_arrs)
    inside = msk == 1
    np.testing.assert_allclose(unc_arr[1][inside], np.std(stack, axis=0)[inside], atol=1e-6)
    np.testing.assert_allclose(unc_arr[3][inside], np.min(stack, axis=0)[inside], atol=1e-6)
    np.testing.assert_allclose(unc_arr[4][inside], np.max(stack, axis=0)[inside], atol=1e-6)
    iqr = np.subtract(*np.percentile(stack, [75, 25], axis=0))
    np.testing.assert_allclose(unc_arr[2][inside], iqr[inside], atol=1e-6)
    assert nodata == -1.0 and np.all(unc_arr[:, ~inside] == -1.0)


def test_sgl_cls_uncertainty_and_validation(tmp_path):
    env_vars, msk_img, _, _, _ = _scene(tmp_path)
    rng = np.random.default_rng(1)
    pres = pd.DataFrame({"v1": rng.normal(5, 1, 80), "v2": rng.normal(2.5, 0.5, 80)})
    mdls = [IsolationForest(n_estimators=20, random_state=s).fit(pres) for s in range(3)]
    out = str(tmp_path / "sgl.tif")
    sdm.pred_ensemble_sklearn_sgl_cls_mdls_prob(
        mdls, msk_img, 1, env_vars, ["v1", "v2"], out, uncertainty="std",
        use_pbar=False, calc_stats=False,
    )
    arr, desc, _ = _read_all(out)
    assert desc == ("median_score", "std") and arr.shape[0] == 2
    for bad in (["bogus"], ["std", "std"]):
        with pytest.raises(rsgispy.RSGISPyException):
            sdm.pred_ensemble_sklearn_mdls_prob(
                _models(2), msk_img, 1, env_vars, ["v1", "v2"], out,
                uncertainty=bad, use_pbar=False,
            )


def test_predict_raster_multiband_shape_check(tmp_path):
    env_vars, msk_img, _, _, _ = _scene(tmp_path)
    out = str(tmp_path / "mb.tif")
    sdm.predict_raster_from_env_vars(
        msk_img, 1, env_vars, ["v1", "v2"], out,
        lambda df: np.vstack([df["v1"].to_numpy(), df["v2"].to_numpy()]),
        out_band_names=["a", "b"], use_pbar=False, calc_stats=False,
    )
    arr, desc, _ = _read_all(out)
    assert desc == ("a", "b") and arr.shape[0] == 2
    with pytest.raises(rsgispy.RSGISPyException, match="shape"):
        sdm.predict_raster_from_env_vars(
            msk_img, 1, env_vars, ["v1", "v2"], out,
            lambda df: df["v1"].to_numpy(), out_band_names=["a", "b"],
            use_pbar=False,
        )


# --------------------------------------------------------------------------
# MESS and ExDet
# --------------------------------------------------------------------------


def _ref(v1_vals, v2_vals, extra=None):
    data = {"v1": v1_vals, "v2": v2_vals}
    if extra:
        data.update(extra)
    n_pts = len(v1_vals)
    return gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(np.zeros(n_pts), np.zeros(n_pts)), crs=CRS)


def test_mess_and_mod(tmp_path):
    env_vars, msk_img, msk, v1, v2 = _scene(tmp_path)
    rng = np.random.default_rng(2)
    ref = _ref(rng.uniform(2.0, 6.0, 500), rng.uniform(1.0, 4.0, 500))
    mess_img = str(tmp_path / "mess.tif")
    mod_img = str(tmp_path / "mod.tif")
    sdm.calc_mess(
        ref, None, msk_img, 1, env_vars, ["v1", "v2"], mess_img,
        out_mod_var_img=mod_img, use_pbar=False,
    )
    mess = _read_all(mess_img)[0][0]
    mod = _read_all(mod_img)[0][0]
    v1_min, v1_max = ref["v1"].min(), ref["v1"].max()
    v2_min, v2_max = ref["v2"].min(), ref["v2"].max()
    inside = (msk == 1) & (v1 > v1_min) & (v1 < v1_max) & (v2 > v2_min) & (v2 < v2_max)
    assert np.all(mess[inside] >= 0) and np.all(mess[inside] <= 100)
    v1_out = (msk == 1) & (v1 > v1_max + 0.5) & (v2 > v2_min + 0.5) & (v2 < v2_max - 0.5)
    v2_out = (msk == 1) & (v2 > v2_max + 0.3) & (v1 > v1_min + 1) & (v1 < v1_max - 1)
    assert np.all(mess[v1_out] < 0) and np.all(mess[v2_out] < 0)
    assert np.all(mod[v1_out] == 1) and np.all(mod[v2_out] == 2)
    assert np.all(mess[msk == 0] == -9999.0) and np.all(mod[msk == 0] == 0)
    names = {val: name for val, (name, _) in get_thematic_classes(mod_img).items()}
    assert names == {1: "v1", 2: "v2"}
    # Hand check one pixel above the v1 range: 100 * (max - p) / (max - min).
    row, col = np.argwhere(v1_out)[0]
    expected_v1 = 100.0 * (v1_max - v1[row, col]) / (v1_max - v1_min)
    assert mess[row, col] == pytest.approx(expected_v1, rel=1e-5)


def test_mess_categorical(tmp_path):
    env_vars, msk_img, msk, _, _ = _scene(tmp_path)
    cat = np.ones((SIZE, SIZE), np.uint8)
    cat[:, : SIZE // 2] = 2
    cat[:5, :] = 3  # unseen category
    env_vars["lc"] = sdm.EnvVarInfo(
        name="lc", file=_write(tmp_path / "lc.tif", cat, 0),
        data_type=rsgispy.VAR_TYPE_CATEGORICAL,
    )
    rng = np.random.default_rng(3)
    ref = _ref(rng.uniform(0, 10, 300), rng.uniform(0, 5, 300), {"lc": rng.integers(1, 3, 300)})
    mess_img = str(tmp_path / "mess_cat.tif")
    mod_img = str(tmp_path / "mod_cat.tif")
    sdm.calc_mess(
        ref, None, msk_img, 1, env_vars, ["v1", "v2", "lc"], mess_img,
        out_mod_var_img=mod_img, use_pbar=False, calc_stats=False,
    )
    mess = _read_all(mess_img)[0][0]
    mod = _read_all(mod_img)[0][0]
    unseen = (msk == 1) & (cat == 3)
    assert np.all(mess[unseen] == -100.0) and np.all(mod[unseen] == 3)
    assert np.all(mess[(msk == 1) & (cat != 3)] > -100.0)


def test_exdet(tmp_path):
    env_vars, msk_img, msk, v1, v2 = _scene(tmp_path)
    rng = np.random.default_rng(4)
    # Reference: v2 strongly tied to v1 (diagonal band), both spanning most of the range.
    ref_v1 = rng.uniform(1.0, 9.0, 600)
    ref_v2 = ref_v1 / 2.0 + rng.normal(0.0, 0.15, 600)
    ref = _ref(ref_v1, ref_v2)
    out = str(tmp_path / "exdet.tif")
    sdm.calc_exdet(ref, None, msk_img, 1, env_vars, ["v1", "v2"], out, use_pbar=False)
    arr, desc, nodata = _read_all(out)
    nt1, nt2 = arr
    assert desc == ("NT1", "NT2") and nodata == -9999.0
    r1_min, r1_max = ref_v1.min(), ref_v1.max()
    r2_min, r2_max = ref_v2.min(), ref_v2.max()
    in_range = (msk == 1) & (v1 >= r1_min) & (v1 <= r1_max) & (v2 >= r2_min) & (v2 <= r2_max)
    assert np.all(nt1[in_range] == 0.0)
    out_range = (msk == 1) & ~in_range
    assert np.all(nt1[out_range] < 0.0)
    # Hand check NT1 for one pixel below both ranges.
    row, col = np.argwhere((msk == 1) & (v1 < r1_min) & (v2 < r2_min))[0]
    exp_nt1 = (v1[row, col] - r1_min) / (r1_max - r1_min) + (v2[row, col] - r2_min) / (r2_max - r2_min)
    assert nt1[row, col] == pytest.approx(exp_nt1, rel=1e-5)
    # High v1 with low v2: each within range, but off the reference diagonal.
    combo = in_range & (v1 > 7.0) & (v2 < 1.0)
    assert combo.any() and np.all(nt2[combo] > 1.0)
    on_diag = in_range & (np.abs(v2 - v1 / 2.0) < 0.1) & (v1 > 4) & (v1 < 6)
    assert on_diag.any() and np.all(nt2[on_diag] < 1.0)


def test_exdet_singular_warns(tmp_path):
    env_vars, msk_img, _, _, _ = _scene(tmp_path)
    env_vars["v1b"] = sdm.EnvVarInfo(name="v1b", file=env_vars["v1"].file)
    rng = np.random.default_rng(5)
    v1_ref = rng.uniform(0, 10, 100)
    ref = _ref(v1_ref, rng.uniform(0, 5, 100), {"v1b": v1_ref})
    with pytest.warns(UserWarning, match="singular"):
        sdm.calc_exdet(
            ref, None, msk_img, 1, env_vars, ["v1", "v2", "v1b"],
            str(tmp_path / "ex.tif"), use_pbar=False, calc_stats=False,
        )


# --------------------------------------------------------------------------
# Scenario projection and range change
# --------------------------------------------------------------------------


def test_copy_env_vars_for_scenario(tmp_path):
    env_vars, _, _, v1, _ = _scene(tmp_path)
    env_vars["v1"].set_norm_mean_std(5.0, 2.0)
    fut = _write(tmp_path / "v1_future.tif", (v1 + 2.0).astype(np.float32), -9999.0)
    new_vars = sdm.copy_env_vars_for_scenario(env_vars, {"v1": fut}, band_map={"v1": 1})
    assert new_vars["v1"].file == fut and env_vars["v1"].file != fut
    assert (new_vars["v1"].norm_mean, new_vars["v1"].norm_std) == (5.0, 2.0)
    assert new_vars["v2"].file == env_vars["v2"].file
    assert new_vars["v2"] is not env_vars["v2"]
    with pytest.raises(rsgispy.RSGISPyException, match="not in env_vars"):
        sdm.copy_env_vars_for_scenario(env_vars, {"v3": fut})
    with pytest.raises(rsgispy.RSGISPyException, match="does not exist"):
        sdm.copy_env_vars_for_scenario(env_vars, {"v1": str(tmp_path / "nope.tif")})


def test_calc_range_change(tmp_path):
    cur = np.array(
        [[1, 1, 0, 0], [1, 1, 0, 0], [1, 0, 0, 0], [255, 1, 1, 0]], dtype=np.uint8
    )
    fut = np.array(
        [[1, 0, 1, 0], [1, 0, 1, 0], [1, 1, 0, 0], [1, 1, 255, 0]], dtype=np.uint8
    )
    msk = np.ones((4, 4), np.uint8)
    msk[3, 3] = 0
    cur_img = _write(tmp_path / "cur.tif", cur, 255, res=10.0)
    fut_img = _write(tmp_path / "fut.tif", fut, 255, res=10.0)
    msk_img = _write(tmp_path / "msk.tif", msk, res=10.0)
    out_img = str(tmp_path / "change.tif")
    out_csv = str(tmp_path / "change.csv")
    res = sdm.calc_range_change(
        cur_img, fut_img, out_img, in_msk_img=msk_img, out_csv=out_csv, use_pbar=False
    )
    expected = np.array(
        [[4, 2, 3, 1], [4, 2, 3, 1], [4, 3, 1, 1], [0, 4, 0, 0]], dtype=np.uint8
    )
    np.testing.assert_array_equal(_read_all(out_img)[0][0], expected)
    assert (res["n_absent"], res["n_lost"], res["n_gained"], res["n_stable"]) == (4, 2, 3, 4)
    assert res["area_stable"] == pytest.approx(400.0)
    assert res["current_area"] == pytest.approx(600.0)
    assert res["future_area"] == pytest.approx(700.0)
    assert res["pct_loss"] == pytest.approx(100.0 * 2 / 6)
    assert res["pct_gain"] == pytest.approx(100.0 * 3 / 6)
    assert res["pct_net_change"] == pytest.approx(100.0 / 6)
    names = {v: n for v, (n, _) in get_thematic_classes(out_img).items()}
    assert names == {1: "Absent", 2: "Lost", 3: "Gained", 4: "Stable"}
    csv = pd.read_csv(out_csv).set_index("metric")["value"]
    assert csv["n_gained"] == 3

    bad = _write(tmp_path / "bad.tif", np.zeros((5, 5), np.uint8), res=10.0)
    with pytest.raises(rsgispy.RSGISPyException, match="same grid"):
        sdm.calc_range_change(cur_img, bad, str(tmp_path / "x.tif"), use_pbar=False)
