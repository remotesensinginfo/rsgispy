"""
Tests for sampling-bias support in :mod:`rsgispy.modelling.species_distribution`:
``create_bias_grid``, bias-weighted ``gen_pseudo_absences_smpls``,
``calc_bias_weights``, ``fixed_var_vals`` in raster prediction and response
curves, ``exclude_vars`` in jackknife importance, and ``sample_weight_col`` in
the fitting functions.
"""

import os

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from shapely.geometry import LineString, Point

import rsgispy
import rsgispy.modelling.species_distribution as sdm
from rsgispy.vector.create import create_random_pts_in_bbox
from rsgispy.zonalstats.point import ext_point_band_values

from conftest import write_raster

pytest.importorskip("sklearn")
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402

CRS = "EPSG:32630"
X0, Y0, RES, SIZE = 400000.0, 5800000.0, 100.0, 40


def _write(path, arr, nodata=None, dtype=None):
    return write_raster(
        path, arr, dtype=dtype, nodata=nodata, crs=CRS, origin=(X0, Y0), pixel_size=RES
    )


def _read(path):
    with rasterio.open(path) as src:
        return src.read(1), src.transform, src.nodata


def _mask(tmp_path, border=2):
    msk = np.zeros((SIZE, SIZE), dtype=np.uint8)
    msk[border:-border, border:-border] = 1
    return _write(tmp_path / "mask.tif", msk), msk


def _pt(row, col):
    return Point(X0 + (col + 0.5) * RES, Y0 - (row + 0.5) * RES)


# --------------------------------------------------------------------------
# create_bias_grid
# --------------------------------------------------------------------------


def test_bias_grid_kde(tmp_path):
    msk_img, msk = _mask(tmp_path)
    rng = np.random.default_rng(0)
    pts = [_pt(10 + rng.normal(0, 1), 12 + rng.normal(0, 1)) for _ in range(30)]
    pts += [_pt(30, 30)]
    recs = gpd.GeoDataFrame(geometry=pts, crs=CRS)
    out_img = str(tmp_path / "bias.tif")
    sdm.create_bias_grid(
        recs, None, msk_img, 1, out_img, bandwidth=300.0, min_weight=0.01,
        block_size=16, use_pbar=False,
    )
    vals, transform, nodata = _read(out_img)
    with rasterio.open(msk_img) as src:
        assert transform == src.transform and vals.shape == (SIZE, SIZE)
    assert nodata == 0.0
    assert np.all(vals[msk == 0] == 0.0)
    inside = vals[msk == 1]
    assert inside.min() >= 0.01 - 1e-6 and inside.max() == pytest.approx(1.0)
    peak = np.unravel_index(np.argmax(np.where(msk == 1, vals, -1)), vals.shape)
    assert abs(peak[0] - 10) <= 2 and abs(peak[1] - 12) <= 2
    assert vals[10, 12] > vals[30, 30] > vals[20, 35]
    # Block size does not change the result.
    out_img2 = str(tmp_path / "bias2.tif")
    sdm.create_bias_grid(
        recs, None, msk_img, 1, out_img2, bandwidth=300.0, min_weight=0.01,
        block_size=64, use_pbar=False,
    )
    np.testing.assert_allclose(_read(out_img2)[0], vals, rtol=1e-5)
    # Default bandwidth (Scott's rule) runs.
    sdm.create_bias_grid(
        recs, None, msk_img, 1, str(tmp_path / "bias3.tif"), use_pbar=False,
        calc_stats=False,
    )


def test_bias_grid_count_and_weights(tmp_path):
    msk_img, _ = _mask(tmp_path)
    recs = gpd.GeoDataFrame(
        {"w": [1.0, 1.0, 2.0, 1.0]},
        geometry=[_pt(5, 5), _pt(5, 5), _pt(5, 5), _pt(20, 20)],
        crs=CRS,
    )
    out_img = str(tmp_path / "count.tif")
    sdm.create_bias_grid(
        recs, None, msk_img, 1, out_img, method="count", min_weight=0.1,
        use_pbar=False,
    )
    vals = _read(out_img)[0]
    assert vals[5, 5] == pytest.approx(1.0)
    assert vals[20, 20] == pytest.approx(0.1 + 0.9 / 3.0)
    assert vals[10, 10] == pytest.approx(0.1)
    sdm.create_bias_grid(
        recs, None, msk_img, 1, out_img, method="count", min_weight=0.1,
        weight_col="w", use_pbar=False,
    )
    vals = _read(out_img)[0]
    assert vals[20, 20] == pytest.approx(0.1 + 0.9 / 4.0)


def test_bias_grid_distance(tmp_path):
    msk_img, msk = _mask(tmp_path)
    x_line = X0 + 10.5 * RES
    roads = gpd.GeoDataFrame(
        geometry=[LineString([(x_line, Y0 - 1.0), (x_line, Y0 - SIZE * RES + 1.0)])],
        crs=CRS,
    )
    out_img = str(tmp_path / "dist_bias.tif")
    sdm.create_bias_grid(
        roads, None, msk_img, 1, out_img, method="distance", dist_decay=500.0,
        tmp_dir=str(tmp_path), use_pbar=False,
    )
    vals = _read(out_img)[0]
    row = vals[20]
    assert row[10] == pytest.approx(1.0)
    assert np.all(np.diff(row[10:SIZE - 2]) < 0)
    assert np.all(np.diff(row[2:11]) > 0)
    assert row[20] == pytest.approx(1e-3 + (1 - 1e-3) * np.exp(-1000.0 / 500.0), rel=1e-4)
    assert np.all(vals[msk == 0] == 0.0)
    assert not [f for f in os.listdir(tmp_path) if f.startswith("tmp")]


def test_bias_grid_validation(tmp_path):
    msk_img, _ = _mask(tmp_path)
    recs = gpd.GeoDataFrame(geometry=[_pt(5, 5), _pt(6, 6)], crs=CRS)
    out_img = str(tmp_path / "b.tif")
    with pytest.raises(rsgispy.RSGISPyException):
        sdm.create_bias_grid(recs, None, msk_img, 1, out_img, method="bogus")
    with pytest.raises(rsgispy.RSGISPyException, match="dist_decay"):
        sdm.create_bias_grid(recs, None, msk_img, 1, out_img, method="distance")
    with pytest.raises(rsgispy.RSGISPyException, match="min_weight"):
        sdm.create_bias_grid(recs, None, msk_img, 1, out_img, min_weight=0.0)
    one = gpd.GeoDataFrame(geometry=[_pt(5, 5)], crs=CRS)
    with pytest.raises(rsgispy.RSGISPyException, match="bandwidth"):
        sdm.create_bias_grid(one, None, msk_img, 1, out_img, use_pbar=False)
    far = gpd.GeoDataFrame(geometry=[Point(0, 0), Point(10, 10)], crs=CRS)
    with pytest.raises(rsgispy.RSGISPyException, match="zero everywhere"):
        sdm.create_bias_grid(
            far, None, msk_img, 1, out_img, bandwidth=100.0, use_pbar=False
        )


# --------------------------------------------------------------------------
# Bias-weighted pseudo-absences
# --------------------------------------------------------------------------


def _half_bias(tmp_path, msk):
    bias = np.full((SIZE, SIZE), 1e-3, dtype=np.float32)
    bias[:, : SIZE // 2] = 1.0
    bias[msk == 0] = 0.0
    return _write(tmp_path / "half_bias.tif", bias, nodata=0.0)


def test_bias_weighted_pseudo_absences(tmp_path):
    msk_img, msk = _mask(tmp_path)
    bias_img = _half_bias(tmp_path, msk)
    pts = sdm.gen_pseudo_absences_smpls(
        msk_img, 1, n_smpls=2000, xtra_n_smpls=0, rnd_seed=4, bias_img=bias_img,
        use_pbar=False,
    )
    assert len(pts) == 2000
    x_mid = X0 + (SIZE // 2) * RES
    assert np.mean(pts.geometry.x < x_mid) > 0.98
    vals = ext_point_band_values(
        pts, msk_img, 1, -np.inf, np.inf, -1.0, out_field="m", reproj_vec=False,
        use_pbar=False,
    )["m"]
    assert (vals == 1).all()
    # Distinct points even where pixels repeat.
    assert len({(p.x, p.y) for p in pts.geometry}) == len(pts)
    again = sdm.gen_pseudo_absences_smpls(
        msk_img, 1, n_smpls=2000, xtra_n_smpls=0, rnd_seed=4, bias_img=bias_img,
        use_pbar=False,
    )
    assert pts.geometry.equals(again.geometry)


def test_bias_weighted_with_presence_exclusion(tmp_path):
    msk_img, msk = _mask(tmp_path)
    bias_img = _half_bias(tmp_path, msk)
    pres = gpd.GeoDataFrame(geometry=[_pt(20, 10)], crs=CRS)
    pts = sdm.gen_pseudo_absences_smpls(
        msk_img, 1, n_smpls=300, xtra_n_smpls=700, presence_smpls_vec_file=pres,
        presence_smpls_dist_thres=500.0, rnd_seed=1, bias_img=bias_img,
        use_pbar=False,
    )
    assert len(pts) == 300
    assert pts.distance(pres.geometry.iloc[0]).min() > 500.0


def test_uniform_pseudo_absences_unchanged(tmp_path):
    msk_img, _ = _mask(tmp_path)
    pts = sdm.gen_pseudo_absences_smpls(
        msk_img, 1, n_smpls=50, xtra_n_smpls=50, rnd_seed=9, use_pbar=False
    )
    # Reference: the original algorithm (uniform candidates over the bbox,
    # kept where the mask is 1, then a random subset).
    with rasterio.open(msk_img) as src:
        bounds = src.bounds
    cand = create_random_pts_in_bbox(
        bbox=(bounds.left, bounds.right, bounds.bottom, bounds.top),
        n_pts=100, epsg_code=32630, out_vec_file=None, rnd_seed=9,
    )
    cand = ext_point_band_values(
        cand, msk_img, 1, -np.inf, np.inf, -9999.0, out_field="m",
        reproj_vec=False, use_pbar=False,
    )
    inside = cand[cand["m"] == 1]
    ref = {(round(p.x, 6), round(p.y, 6)) for p in inside.geometry}
    got = {(round(p.x, 6), round(p.y, 6)) for p in pts.geometry}
    assert got <= ref
    assert len(pts) == min(50, len(inside))


def test_bias_weighted_errors(tmp_path):
    msk_img, msk = _mask(tmp_path)
    zero_img = _write(tmp_path / "zero.tif", np.zeros((SIZE, SIZE), np.float32))
    with pytest.raises(rsgispy.RSGISPyException, match="zero"):
        sdm.gen_pseudo_absences_smpls(
            msk_img, 1, n_smpls=10, bias_img=zero_img, use_pbar=False
        )
    small = np.ones((SIZE // 2, SIZE // 2), np.float32)
    small_img = _write(tmp_path / "small.tif", small)
    with pytest.raises(rsgispy.RSGISPyException, match="same grid"):
        sdm.gen_pseudo_absences_smpls(
            msk_img, 1, n_smpls=10, bias_img=small_img, use_pbar=False
        )


# --------------------------------------------------------------------------
# calc_bias_weights
# --------------------------------------------------------------------------


def test_calc_bias_weights(tmp_path):
    msk_img, msk = _mask(tmp_path)
    bias_img = _half_bias(tmp_path, msk)
    smpls = gpd.GeoDataFrame(
        {"clsid": [1, 1, 1, 0]},
        geometry=[_pt(10, 5), _pt(10, 6), _pt(10, 30), _pt(10, 31)],
        crs=CRS,
    )
    out = sdm.calc_bias_weights(smpls, None, bias_img)
    inv = np.array([1.0, 1.0, 1.0 / 1e-3])
    np.testing.assert_allclose(out["weight"].iloc[:3], inv / inv.mean(), rtol=1e-5)
    assert out["weight"].iloc[3] == 1.0
    assert out["weight"].iloc[:3].mean() == pytest.approx(1.0)
    out_all = sdm.calc_bias_weights(smpls, None, bias_img, presence_only=False)
    assert out_all["weight"].mean() == pytest.approx(1.0)
    outside = smpls.copy()
    outside.loc[3, "geometry"] = Point(0, 0)
    with pytest.raises(rsgispy.RSGISPyException, match="outside"):
        sdm.calc_bias_weights(outside, None, bias_img, presence_only=False)


# --------------------------------------------------------------------------
# fixed_var_vals in raster prediction
# --------------------------------------------------------------------------


def _env_scene(tmp_path):
    rng = np.random.default_rng(3)
    rows, cols = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    v1 = rows / SIZE * 10.0
    v2 = cols / SIZE * 5.0
    v1_img = _write(tmp_path / "v1.tif", v1, nodata=-9999.0)
    v2_img = _write(tmp_path / "v2.tif", v2, nodata=-9999.0)
    env_vars = {
        "v1": sdm.EnvVarInfo(name="v1", file=v1_img, band=1),
        "v2": sdm.EnvVarInfo(name="v2", file=v2_img, band=1),
    }
    x_df = pd.DataFrame({"v1": rng.uniform(0, 10, 200), "v2": rng.uniform(0, 5, 200)})
    y = (x_df["v1"] + 2 * x_df["v2"] + rng.normal(0, 2, 200) > 10).astype(int)
    clf = LogisticRegression().fit(x_df, y)
    return env_vars, clf


def test_fixed_var_vals_prediction(tmp_path):
    msk_img, msk = _mask(tmp_path)
    env_vars, clf = _env_scene(tmp_path)
    const_img = _write(tmp_path / "v2_const.tif", np.full((SIZE, SIZE), 4.5, np.float32))
    env_const = dict(env_vars)
    env_const["v2"] = sdm.EnvVarInfo(name="v2", file=const_img, band=1)
    env_missing = dict(env_vars)
    env_missing["v2"] = sdm.EnvVarInfo(
        name="v2", file=str(tmp_path / "does_not_exist.tif"), band=1
    )

    ref_img = str(tmp_path / "ref.tif")
    fix_img = str(tmp_path / "fix.tif")
    sdm.pred_sklearn_mdl_prob(
        clf, msk_img, 1, env_const, ["v1", "v2"], ref_img, use_pbar=False,
        calc_stats=False,
    )
    sdm.pred_sklearn_mdl_prob(
        clf, msk_img, 1, env_missing, ["v1", "v2"], fix_img, use_pbar=False,
        calc_stats=False, fixed_var_vals={"v2": 4.5},
    )
    np.testing.assert_allclose(_read(fix_img)[0], _read(ref_img)[0], rtol=1e-6)
    assert np.all(_read(fix_img)[0][msk == 0] == -1.0)

    # Ensemble and class outputs accept it too.
    sdm.pred_ensemble_sklearn_mdls_prob(
        [clf, clf], msk_img, 1, env_missing, ["v1", "v2"],
        str(tmp_path / "ens.tif"), use_pbar=False, calc_stats=False,
        fixed_var_vals={"v2": 4.5},
    )
    np.testing.assert_allclose(
        _read(str(tmp_path / "ens.tif"))[0], _read(ref_img)[0], rtol=1e-6
    )
    sdm.pred_sklearn_mdl_cls(
        clf, msk_img, 1, env_missing, ["v1", "v2"], str(tmp_path / "cls.tif"),
        use_pbar=False, calc_stats=False, fixed_var_vals={"v2": 4.5},
    )


def test_fixed_var_vals_normalised(tmp_path):
    msk_img, _ = _mask(tmp_path)
    env_vars, clf = _env_scene(tmp_path)
    for info, (mean, std) in zip(env_vars.values(), [(5.0, 2.0), (2.5, 1.0)]):
        info.set_norm_mean_std(mean, std)
    const_img = _write(tmp_path / "v2_const.tif", np.full((SIZE, SIZE), 4.5, np.float32))
    env_const = dict(env_vars)
    env_const["v2"] = sdm.EnvVarInfo(name="v2", file=const_img, band=1, norm_mean=2.5, norm_std=1.0)
    ref_img = str(tmp_path / "ref.tif")
    fix_img = str(tmp_path / "fix.tif")
    sdm.pred_sklearn_mdl_prob(
        clf, msk_img, 1, env_const, ["v1", "v2"], ref_img, normalise_data=True,
        use_pbar=False, calc_stats=False,
    )
    sdm.pred_sklearn_mdl_prob(
        clf, msk_img, 1, env_vars, ["v1", "v2"], fix_img, normalise_data=True,
        use_pbar=False, calc_stats=False, fixed_var_vals={"v2": 4.5},
    )
    np.testing.assert_allclose(_read(fix_img)[0], _read(ref_img)[0], rtol=1e-6)


def test_fixed_var_vals_validation(tmp_path):
    msk_img, _ = _mask(tmp_path)
    env_vars, clf = _env_scene(tmp_path)
    for bad in ({"v3": 1.0}, {"v2": np.nan}, {"v2": "x"}):
        with pytest.raises(rsgispy.RSGISPyException):
            sdm.pred_sklearn_mdl_prob(
                clf, msk_img, 1, env_vars, ["v1", "v2"], str(tmp_path / "o.tif"),
                use_pbar=False, fixed_var_vals=bad,
            )


# --------------------------------------------------------------------------
# Response curves, jackknife and sample weights
# --------------------------------------------------------------------------


def _train_test(seed=0, n=300):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "v1": rng.uniform(0, 10, n),
            "v2": rng.uniform(0, 5, n),
            "bias": rng.uniform(0, 1, n),
        }
    )
    df["clsid"] = (df["v1"] + 2 * df["v2"] + rng.normal(0, 2, n) > 10).astype(int)
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(rng.uniform(0, 1, n), rng.uniform(0, 1, n)),
        crs=CRS,
    )
    return gdf.iloc[: n // 2].copy(), gdf.iloc[n // 2 :].copy()


def test_response_curves_fixed_var(tmp_path):
    train, _ = _train_test()
    clf = LogisticRegression().fit(train[["v1", "v2", "bias"]], train["clsid"])
    resp = sdm.sklearn_mdl_variable_response_curves(
        clf, train, None, ["v1", "v2", "bias"], None, None, n_samples=20,
        fixed_var_vals={"bias": 1.0},
    )
    assert "bias_resp" not in resp.columns and "v1_resp" in resp.columns
    with pytest.raises(rsgispy.RSGISPyException):
        sdm.sklearn_mdl_variable_response_curves(
            clf, train, None, ["v1", "v2"], None, None, fixed_var_vals={"bias": 1.0}
        )


def test_jackknife_exclude_vars():
    train, test = _train_test()
    imp = sdm.sklearn_jackknife_var_importance(
        LogisticRegression(), train, None, test, None, ["v1", "v2", "bias"], None,
        use_pbar=False, exclude_vars=["bias"],
    )
    assert list(imp.index) == ["v1", "v2"]
    with pytest.raises(rsgispy.RSGISPyException):
        sdm.sklearn_jackknife_var_importance(
            LogisticRegression(), train, None, test, None, ["v1", "bias"], None,
            use_pbar=False, exclude_vars=["bias"],
        )


def test_sample_weight_col():
    train, test = _train_test()
    train["w1"] = 1.0
    mdl_plain, *_ = sdm.fit_sklearn_mdl(
        LogisticRegression(), train, None, test, None, ["v1", "v2"], return_model=True
    )
    mdl_w1, *_ = sdm.fit_sklearn_mdl(
        LogisticRegression(), train, None, test, None, ["v1", "v2"],
        return_model=True, sample_weight_col="w1",
    )
    np.testing.assert_allclose(mdl_w1.coef_, mdl_plain.coef_, rtol=1e-6)

    # Weight 2 equals duplicating those rows.
    train["w2"] = np.where(np.arange(len(train)) % 2 == 0, 2.0, 1.0)
    dup = pd.concat([train, train.iloc[::2]])
    mdl_w2, *_ = sdm.fit_sklearn_mdl(
        LogisticRegression(), train, None, test, None, ["v1", "v2"],
        return_model=True, sample_weight_col="w2",
    )
    mdl_dup, *_ = sdm.fit_sklearn_mdl(
        LogisticRegression(), dup, None, test, None, ["v1", "v2"], return_model=True
    )
    np.testing.assert_allclose(mdl_w2.coef_, mdl_dup.coef_, rtol=1e-3)

    mdls = sdm.fit_kfold_sklearn_mdls(
        LogisticRegression(), train, None, test, None, ["v1", "v2"], n_kfolds=3,
        rnd_seed=1, use_pbar=False, sample_weight_col="w2",
    )
    assert len(mdls) == 3

    with pytest.raises(rsgispy.RSGISPyException, match="sample_weight"):
        sdm.fit_sklearn_mdl(
            KNeighborsClassifier(), train, None, test, None, ["v1", "v2"],
            sample_weight_col="w2",
        )
    train["bad"] = -1.0
    with pytest.raises(rsgispy.RSGISPyException):
        sdm.fit_sklearn_mdl(
            LogisticRegression(), train, None, test, None, ["v1", "v2"],
            sample_weight_col="bad",
        )
