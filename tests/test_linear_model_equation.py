"""
Tests for the linear model helpers of rsgispy.regression.regsklearn:
get_linear_model_coefs, get_linear_model_equation and
get_linear_model_band_maths_exp.
"""

import os

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

pytest.importorskip("sklearn")

# pylint: disable=wrong-import-position
from sklearn.compose import TransformedTargetRegressor  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.linear_model import LinearRegression, Ridge  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import PolynomialFeatures, StandardScaler  # noqa: E402

import rsgispy  # noqa: E402
from rsgispy.imagecalc.maths import BandDefn, band_maths  # noqa: E402
import rsgispy.regression.regsklearn as regsk  # noqa: E402

COEFS = np.array([2.5, -0.75, 0.00004])
INTERCEPT = 12.0


def _data(n_rows=60, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 100.0, (n_rows, 3)) * np.array([1.0, 2.0, 1000.0])
    y = INTERCEPT + x @ COEFS
    return pd.DataFrame(x, columns=["h_mean", "cover", "dens"]), y


def test_coefs_plain_and_pipeline():
    x_df, y_arr = _data()
    lm = LinearRegression().fit(x_df, y_arr)
    coefs = regsk.get_linear_model_coefs(lm)
    assert list(coefs.columns) == ["intercept", "h_mean", "cover", "dens"]
    assert coefs.index.tolist() == ["y1"]
    assert coefs.iloc[0].to_numpy() == pytest.approx([INTERCEPT, *COEFS])

    # A scaled pipeline gives the coefficients in the input units.
    pipe = make_pipeline(StandardScaler(), Ridge(alpha=1e-9)).fit(x_df, y_arr)
    coefs_p = regsk.get_linear_model_coefs(pipe, target_names=["vol"])
    assert coefs_p.index.tolist() == ["vol"]
    assert coefs_p.iloc[0].to_numpy() == pytest.approx([INTERCEPT, *COEFS], rel=1e-6)

    # Fitted on an array: names given, or x1, x2, ...
    lm_arr = LinearRegression().fit(x_df.to_numpy(), y_arr)
    assert list(regsk.get_linear_model_coefs(lm_arr).columns[1:]) == ["x1", "x2", "x3"]
    ttr = TransformedTargetRegressor(
        regressor=LinearRegression(), func=lambda v: v * 2.0, inverse_func=lambda v: v / 2.0
    ).fit(x_df.to_numpy(), y_arr)
    coefs_t = regsk.get_linear_model_coefs(ttr, feat_names=["a", "b", "c"])
    assert coefs_t.iloc[0].to_numpy() == pytest.approx([INTERCEPT, *COEFS], rel=1e-6)


def test_coefs_multi_target_and_errors():
    x_df, y_arr = _data()
    y_2 = np.column_stack([y_arr, 3.0 - 2.0 * x_df["h_mean"].to_numpy()])
    lm = LinearRegression().fit(x_df, y_2)
    coefs = regsk.get_linear_model_coefs(lm, target_names=["vol", "ba"])
    assert coefs.loc["ba"].to_numpy() == pytest.approx([3.0, -2.0, 0.0, 0.0], abs=1e-8)
    eqns = regsk.get_linear_model_equation(lm, target_name=["vol", "ba"], precision=1)
    assert isinstance(eqns, list) and eqns[1].startswith("ba = 3.0 - 2.0 * h_mean ")

    with pytest.raises(rsgispy.RSGISPyException):
        regsk.get_linear_model_coefs(lm, feat_names=["a", "b"])
    with pytest.raises(rsgispy.RSGISPyException):
        regsk.get_linear_model_coefs(lm, target_names=["vol"])
    poly = make_pipeline(PolynomialFeatures(2), LinearRegression()).fit(x_df, x_df["h_mean"] ** 2)
    with pytest.raises(rsgispy.RSGISPyException):
        regsk.get_linear_model_coefs(poly)
    forest = RandomForestRegressor(n_estimators=5, random_state=0).fit(x_df, y_arr)
    with pytest.raises(rsgispy.RSGISPyException):
        regsk.get_linear_model_equation(forest)


def test_linear_model_equation_format():
    x_df, y_arr = _data()
    lm = LinearRegression().fit(x_df, y_arr)
    eqn = regsk.get_linear_model_equation(lm, target_name="vol")
    assert eqn == "vol = 12.000 + 2.500 * h_mean - 0.750 * cover + 4.000e-05 * dens"
    eqn = regsk.get_linear_model_equation(lm, ["h", "c", "d"], precision=5)
    assert eqn == "y = 12.00000 + 2.50000 * h - 0.75000 * c + 0.00004 * d"


def test_linear_model_band_maths_exp(tmp_path):
    x_df, y_arr = _data()
    pipe = make_pipeline(StandardScaler(), LinearRegression()).fit(x_df, y_arr)
    exp = regsk.get_linear_model_band_maths_exp(pipe, valid_msk_name="vld")
    assert exp.startswith("vld==1?") and exp.endswith(":-9999.0")

    # Apply the expression to an image of the features.
    rng = np.random.default_rng(5)
    feats = rng.uniform(0.0, 100.0, (3, 6, 5)) * np.array([1.0, 2.0, 1000.0])[:, None, None]
    vld = np.ones((1, 6, 5), dtype=np.float32)
    vld[0, 0, :] = 0
    profile = {"driver": "GTiff", "width": 5, "height": 6, "crs": "EPSG:32630",
               "transform": from_origin(0.0, 60.0, 10.0, 10.0), "dtype": "float32"}
    feat_img = os.path.join(tmp_path, "feats.tif")
    vld_img = os.path.join(tmp_path, "vld.tif")
    with rasterio.open(feat_img, "w", count=3, **profile) as dst:
        dst.write(feats.astype(np.float32))
    with rasterio.open(vld_img, "w", count=1, **profile) as dst:
        dst.write(vld)
    band_defns = [BandDefn("vld", vld_img, 1)] + [
        BandDefn(name, feat_img, i + 1) for i, name in enumerate(x_df.columns)
    ]
    out_img = os.path.join(tmp_path, "pred.tif")
    band_maths(out_img, exp, band_defns, datatype=rsgispy.TYPE_64FLOAT, no_data_val=-9999,
               use_pbar=False)
    with rasterio.open(out_img) as src:
        pred = src.read(1)
    f_32 = feats.astype(np.float32).astype(np.float64)
    expected = pipe.predict(pd.DataFrame(f_32.reshape(3, -1).T, columns=x_df.columns))
    expected = expected.reshape(6, 5)
    expected[0, :] = -9999.0
    assert pred == pytest.approx(expected, rel=1e-6, abs=1e-6)

    no_msk = regsk.get_linear_model_band_maths_exp(LinearRegression().fit(x_df, y_arr))
    assert "?" not in no_msk and "h_mean" in no_msk
