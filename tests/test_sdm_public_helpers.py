"""
Tests for the public species distribution modelling helpers exported from
:mod:`rsgispy.modelling.species_distribution`: ROC/AUC, threshold selection,
ROC plotting, positive-class probabilities, single-class predictions and
block-wise raster prediction with a user-supplied predict function.
"""

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
import rasterio
from shapely.geometry import Point
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve

import rsgispy
import rsgispy.modelling.species_distribution as sdm

from conftest import write_raster

matplotlib.use("Agg")

PUBLIC_HELPERS = [
    "get_positive_class_probs",
    "calc_single_class_predictions",
    "calc_binary_roc_auc",
    "calc_binary_threshold",
    "plot_roc_curve",
    "predict_raster_from_env_vars",
]


def _binary_data(n_smpls: int = 200, seed: int = 3):
    """Noisy binary labels and scores where higher scores mean presence."""
    rng = np.random.default_rng(seed)
    y_true = (rng.random(n_smpls) < 0.35).astype(np.uint8)
    scores = rng.normal(0.0, 1.0, n_smpls) + 1.2 * y_true
    return y_true, scores


def _sens_spec(y_true, scores, thres):
    pred = scores >= thres
    sens = np.count_nonzero(pred & (y_true == 1)) / np.count_nonzero(y_true == 1)
    spec = np.count_nonzero(~pred & (y_true == 0)) / np.count_nonzero(y_true == 0)
    return sens, spec


def test_calc_binary_roc_auc_matches_sklearn():
    """ROC curve and AUC equal scikit-learn's for list and array inputs."""
    y_true, scores = _binary_data()
    fpr, tpr, roc_auc = sdm.calc_binary_roc_auc(y_true, scores)
    sk_fpr, sk_tpr, _ = roc_curve(y_true, scores)
    np.testing.assert_allclose(fpr, sk_fpr)
    np.testing.assert_allclose(tpr, sk_tpr)
    assert isinstance(roc_auc, float)
    assert roc_auc == pytest.approx(roc_auc_score(y_true, scores))

    _, _, list_auc = sdm.calc_binary_roc_auc(y_true.tolist(), scores.tolist())
    assert list_auc == pytest.approx(roc_auc)

    _, _, perfect_auc = sdm.calc_binary_roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert perfect_auc == pytest.approx(1.0)


def test_calc_binary_roc_auc_validates_inputs():
    """Malformed inputs raise RSGISPyException."""
    with pytest.raises(rsgispy.RSGISPyException):
        sdm.calc_binary_roc_auc([0, 1, 1], [0.1, 0.9])
    with pytest.raises(rsgispy.RSGISPyException):
        sdm.calc_binary_roc_auc(np.zeros((2, 2)), np.zeros((2, 2)))
    with pytest.raises(rsgispy.RSGISPyException):
        sdm.calc_binary_roc_auc([], [])


def test_calc_binary_threshold_max_tss():
    """The max TSS threshold maximises sensitivity + specificity - 1."""
    y_true, scores = _binary_data()
    thres = sdm.calc_binary_threshold(y_true, scores)
    assert isinstance(thres, float)
    assert np.isfinite(thres)
    assert thres in set(scores.tolist())

    sens, spec = _sens_spec(y_true, scores, thres)
    best_tss = max(
        sum(_sens_spec(y_true, scores, cand)) - 1.0 for cand in np.unique(scores)
    )
    assert sens + spec - 1.0 == pytest.approx(best_tss)

    # Same as the guide's previous scikit-learn glue code.
    fpr, tpr, sk_thres = roc_curve(y_true, scores)
    assert thres == pytest.approx(float(sk_thres[np.argmax(tpr - fpr)]))

    # Perfectly separable data give TSS = 1.
    sep_thres = sdm.calc_binary_threshold([0, 0, 0, 1, 1], [0.1, 0.2, 0.3, 0.7, 0.9])
    assert 0.3 < sep_thres <= 0.7


def test_calc_binary_threshold_sens_eq_spec():
    """The sens_eq_spec threshold minimises |sensitivity - specificity|."""
    y_true, scores = _binary_data(seed=11)
    thres = sdm.calc_binary_threshold(y_true, scores, method="sens_eq_spec")
    sens, spec = _sens_spec(y_true, scores, thres)
    best_diff = min(
        abs(np.subtract(*_sens_spec(y_true, scores, cand)))
        for cand in np.unique(scores)
    )
    assert abs(sens - spec) == pytest.approx(best_diff)
    assert abs(sens - spec) < 0.05


def test_calc_binary_threshold_errors():
    """Bad methods, one-class labels and keyword misuse are rejected."""
    y_true, scores = _binary_data()
    with pytest.raises(rsgispy.RSGISPyException):
        sdm.calc_binary_threshold(y_true, scores, method="kappa")
    with pytest.raises(rsgispy.RSGISPyException):
        sdm.calc_binary_threshold([1, 1, 1], [0.2, 0.5, 0.9])
    with pytest.raises(rsgispy.RSGISPyException):
        sdm.calc_binary_threshold([0, 1, 2], [0.2, 0.5, 0.9])
    with pytest.raises(TypeError):
        sdm.calc_binary_threshold(  # pylint: disable=too-many-function-args
            y_true, scores, "max_tss"
        )


def test_plot_roc_curve(tmp_path):
    """The ROC plot is written (creating directories) and figures are closed."""
    y_true, scores = _binary_data()
    fpr, tpr, roc_auc = sdm.calc_binary_roc_auc(y_true, scores)
    out_png = tmp_path / "plots" / "roc.png"
    plt.close("all")
    fig, ax = sdm.plot_roc_curve(fpr, tpr, roc_auc, label="Ensemble",
                                 title="Test ROC", out_plot_file=str(out_png),
                                 out_dpi=80)
    assert out_png.is_file() and out_png.stat().st_size > 0
    # Saving leaves the returned figure open (so it shows in a notebook).
    assert plt.get_fignums() == [fig.number]
    assert ax.get_title() == "Test ROC"
    plt.close(fig)

    # Drawing on a given axis creates no new figure.
    fig2, ax2 = plt.subplots()
    out_fig, out_ax = sdm.plot_roc_curve(fpr, tpr, roc_auc, ax=ax2)
    assert out_fig is fig2 and out_ax is ax2
    assert plt.get_fignums() == [fig2.number]
    plt.close(fig2)

    with pytest.raises(TypeError):
        sdm.plot_roc_curve(  # pylint: disable=too-many-function-args
            fpr, tpr, roc_auc, "Ensemble"
        )


class _ReversedClassModel:  # pylint: disable=too-few-public-methods
    """Stub estimator whose probability columns are not in label order."""

    classes_ = np.array([1, 0])

    @staticmethod
    def predict_proba(x_data):
        """Return columns ``[P(1), P(0)]``."""
        p_pres = np.asarray(x_data, dtype=float)[:, 0]
        return np.column_stack([p_pres, 1.0 - p_pres])


class _OneClassModel:  # pylint: disable=too-few-public-methods
    """Stub estimator fitted on absences only."""

    classes_ = np.array([0])

    @staticmethod
    def predict_proba(x_data):
        """Return a single all-ones column."""
        return np.ones((len(x_data), 1))


def test_get_positive_class_probs():
    """The positive class column is chosen by label, not position."""
    x_data = np.array([[0.1], [0.7], [0.4]])
    np.testing.assert_allclose(
        sdm.get_positive_class_probs(_ReversedClassModel(), x_data), [0.1, 0.7, 0.4]
    )
    np.testing.assert_allclose(
        sdm.get_positive_class_probs(_ReversedClassModel(), x_data, pos_label=0),
        [0.9, 0.3, 0.6],
    )
    np.testing.assert_array_equal(
        sdm.get_positive_class_probs(_OneClassModel(), x_data), [0.0, 0.0, 0.0]
    )

    y_true, scores = _binary_data()
    x_df = np.column_stack([scores, np.arange(scores.size) % 7])
    rf = RandomForestClassifier(n_estimators=10, random_state=0).fit(x_df, y_true)
    np.testing.assert_allclose(
        sdm.get_positive_class_probs(rf, x_df), rf.predict_proba(x_df)[:, 1]
    )


def test_calc_single_class_predictions():
    """One-class +1/-1 predictions become uint8 1/0."""
    rng = np.random.default_rng(5)
    train = rng.normal(0.0, 1.0, (100, 2))
    test = np.vstack([rng.normal(0.0, 0.5, (20, 2)), rng.normal(8.0, 0.5, (20, 2))])
    mdl = IsolationForest(n_estimators=20, random_state=0).fit(train)
    pred = sdm.calc_single_class_predictions(mdl, test)
    assert pred.dtype == np.uint8
    np.testing.assert_array_equal(pred, (mdl.predict(test) == 1).astype(np.uint8))
    assert pred[20:].sum() == 0


@pytest.fixture(name="pred_scene")
def fixture_pred_scene(tmp_path):
    """Two covariates (one with a no data pixel) and a study area mask."""
    rows, cols = np.indices((24, 20), dtype=np.float32)
    var_a = rows.copy()
    var_b = cols * 10.0
    var_a[5, 5] = -9999.0
    mask = np.zeros(rows.shape, dtype=np.uint8)
    mask[1:-1, 1:-1] = 1
    cov_img = write_raster(tmp_path / "cov.tif", np.stack([var_a, var_b]), nodata=-9999.0)
    msk_img = write_raster(tmp_path / "mask.tif", mask, nodata=0)
    env_vars = {
        "a": sdm.EnvVarInfo(name="a", file=cov_img, band=1,
                            data_type=rsgispy.VAR_TYPE_CONTINUOUS),
        "b": sdm.EnvVarInfo(name="b", file=cov_img, band=2,
                            data_type=rsgispy.VAR_TYPE_CONTINUOUS),
    }
    valid = mask == 1
    valid[5, 5] = False
    return {
        "env_vars": env_vars,
        "msk_img": msk_img,
        "valid": valid,
        "expected": var_a + var_b,
        "tmp_path": tmp_path,
    }


def test_predict_raster_from_env_vars(pred_scene):
    """A custom predict function is applied block by block to valid pixels."""
    out_img = str(pred_scene["tmp_path"] / "out" / "sum.tif")
    seen_cols = []

    def _predict(env_df):
        seen_cols.append(list(env_df.columns))
        return env_df["a"].to_numpy() + env_df["b"].to_numpy()

    sdm.predict_raster_from_env_vars(
        pred_scene["msk_img"], 1, pred_scene["env_vars"], ["a", "b"], out_img,
        _predict, block_size=8, calc_stats=True, use_pbar=False,
    )
    assert seen_cols and all(cols == ["a", "b"] for cols in seen_cols)
    valid = pred_scene["valid"]
    with rasterio.open(out_img) as src:
        assert src.dtypes[0] == "float32"
        assert src.nodata == -1.0
        arr = src.read(1)
    np.testing.assert_allclose(arr[valid], pred_scene["expected"][valid])
    assert np.all(arr[~valid] == -1.0)


def test_predict_raster_from_env_vars_options(pred_scene):
    """Data type, no data, fill value and thematic options are honoured."""
    valid = pred_scene["valid"]
    env_vars = pred_scene["env_vars"]
    msk_img = pred_scene["msk_img"]

    # fill_val defaults to no_data_val.
    nd_img = str(pred_scene["tmp_path"] / "nd.tif")
    sdm.predict_raster_from_env_vars(
        msk_img, 1, env_vars, ["a", "b"], nd_img,
        lambda df: df["a"] * 2.0, no_data_val=-9999.0, calc_stats=False,
        use_pbar=False,
    )
    with rasterio.open(nd_img) as src:
        assert src.nodata == -9999.0
        arr = src.read(1)
    assert np.all(arr[~valid] == -9999.0)

    # Thematic output defaults to uint8 with a colour table and statistics.
    cls_img = str(pred_scene["tmp_path"] / "cls.tif")
    sdm.predict_raster_from_env_vars(
        msk_img, 1, env_vars, ["a", "b"], cls_img,
        lambda df: np.where(df["b"] >= 100.0, 2, 1),
        thematic=True, no_data_val=0, block_size=8, use_pbar=False,
    )
    with rasterio.open(cls_img) as src:
        assert src.dtypes[0] == "uint8"
        assert src.nodata == 0
        cls_arr = src.read(1)
    assert set(np.unique(cls_arr[valid])) == {1, 2}
    assert np.all(cls_arr[~valid] == 0)

    # An explicit rsgispy data type, and the legacy rasterio dtype string.
    i16_img = str(pred_scene["tmp_path"] / "i16.tif")
    sdm.predict_raster_from_env_vars(
        msk_img, 1, env_vars, ["a"], i16_img, lambda df: df["a"],
        datatype=rsgispy.TYPE_16INT, no_data_val=-1, calc_stats=False,
        use_pbar=False,
    )
    u16_img = str(pred_scene["tmp_path"] / "u16.tif")
    sdm.predict_raster_from_env_vars(
        msk_img, 1, env_vars, ["a"], u16_img, lambda df: df["a"],
        out_rio_dtype="uint16", no_data_val=999, calc_stats=False,
        use_pbar=False,
    )
    with rasterio.open(i16_img) as src:
        assert src.dtypes[0] == "int16"
    with rasterio.open(u16_img) as src:
        assert src.dtypes[0] == "uint16"
        assert np.all(src.read(1)[~valid] == 999)


def test_predict_raster_from_env_vars_errors(pred_scene):
    """Wrong prediction lengths, dtype conflicts and positional options fail."""
    out_img = str(pred_scene["tmp_path"] / "bad.tif")
    args = (pred_scene["msk_img"], 1, pred_scene["env_vars"], ["a", "b"], out_img)
    with pytest.raises(rsgispy.RSGISPyException, match="predict_fn"):
        sdm.predict_raster_from_env_vars(
            *args, lambda df: np.zeros(len(df) + 1), calc_stats=False,
            use_pbar=False,
        )
    with pytest.raises(rsgispy.RSGISPyException):
        sdm.predict_raster_from_env_vars(
            *args, lambda df: df["a"], datatype=rsgispy.TYPE_8UINT,
            out_rio_dtype="float32", use_pbar=False,
        )
    with pytest.raises(TypeError):
        sdm.predict_raster_from_env_vars(  # pylint: disable=too-many-function-args
            *args, lambda df: df["a"], "GTIFF"
        )


def test_fit_sklearn_mdl_roc_plot_only_when_requested(tmp_path, monkeypatch):
    """fit_sklearn_mdl does not create a ROC figure unless a file is given."""
    y_true, scores = _binary_data()
    gdf = gpd.GeoDataFrame(
        {"s": scores, "clsid": y_true},
        geometry=[Point(i, i) for i in range(scores.size)],
        crs="EPSG:32630",
    )
    calls = []
    monkeypatch.setattr(
        "rsgispy.modelling._sdm_models.plot_roc_curve",
        lambda *a, **k: (calls.append(k), plt.subplots())[1],
    )
    rf = RandomForestClassifier(n_estimators=5, random_state=0)
    sdm.fit_sklearn_mdl(rf, gdf, None, gdf, None, ["s"])
    assert not calls
    sdm.fit_sklearn_mdl(rf, gdf, None, gdf, None, ["s"],
                        roc_curve_plot=str(tmp_path / "roc.png"))
    assert len(calls) == 1
