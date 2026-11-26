"""
Tests for combining sample tables:
:func:`rsgispy.classification.classdata.combine_cls_samples` and
:func:`rsgispy.regression.regdata.combine_reg_samples`.
"""

import importlib.util

import numpy as np
import pandas as pd
import pytest

import rsgispy
import rsgispy.classification.classdata as rcd
import rsgispy.regression.regdata as rrd

_HAS_PARQUET = (
    importlib.util.find_spec("pyarrow") is not None
    or importlib.util.find_spec("fastparquet") is not None
)
_EXT = ".parquet" if _HAS_PARQUET else ".csv"

_CLASSES = {"Forest": (10, 0), "Water": (50, 1), "Urban": (300, 2)}
_FEATS = ["Blue_mean", "Red_mean", "NIR_mean"]
_CENTRES = {"Forest": (30.0, 25.0, 300.0), "Water": (20.0, 10.0, 5.0),
            "Urban": (90.0, 110.0, 120.0)}


def _seg_samples(n_per_cls=4, seed=0):
    """Segment samples in the layout of extract_class_segment_samples."""
    rng = np.random.default_rng(seed)
    rows = []
    seg_id = 100
    for cls_name, (out_id, int_id) in _CLASSES.items():
        for idx in range(n_per_cls):
            row = {
                "seg_id": np.uint32(seg_id),
                "out_cls_id": out_id,
                "internal_cls_id": int_id,
                "class_name": cls_name,
                "train_grp": f"{cls_name}:poly{idx}",
                "n_hits": int(rng.integers(1, 5)),
                "pxl_count": int(rng.integers(20, 80)),
            }
            for feat, centre in zip(_FEATS, _CENTRES[cls_name]):
                row[feat] = float(centre + rng.normal(0.0, 3.0))
            rows.append(row)
            seg_id += 1
    df = pd.DataFrame(rows)
    df["seg_id"] = df["seg_id"].astype(np.uint32)
    return df


def _pxl_samples(n_per_cls=30, seed=1, with_xy=False):
    """Pixel samples in the layout of extract_class_training_samples (uint16)."""
    rng = np.random.default_rng(seed)
    parts = []
    for cls_name, (out_id, int_id) in _CLASSES.items():
        data = {
            "class_name": [cls_name] * n_per_cls,
            "out_cls_id": [out_id] * n_per_cls,
            "internal_cls_id": [int_id] * n_per_cls,
            "train_grp": [f"{cls_name}:poly{idx % 3}" for idx in range(n_per_cls)],
        }
        if with_xy:
            data["x"] = rng.uniform(0, 1000, n_per_cls)
            data["y"] = rng.uniform(0, 1000, n_per_cls)
        for feat, centre in zip(_FEATS, _CENTRES[cls_name]):
            vals = np.clip(centre + rng.normal(0.0, 4.0, n_per_cls), 0, None)
            data[feat] = vals.round().astype(np.uint16)
        parts.append(pd.DataFrame(data))
    return pd.concat(parts, ignore_index=True)


def test_combine_cls_samples_matches_concat():
    """Same result as the 04/04 notebook's pandas.concat of selected columns."""
    seg_df = _seg_samples()
    pxl_df = _pxl_samples()
    smpl_cols = ["class_name", "out_cls_id", "internal_cls_id"] + _FEATS
    ref = pd.concat([seg_df[smpl_cols], pxl_df[smpl_cols]], ignore_index=True)
    out = rcd.combine_cls_samples([seg_df, pxl_df], feat_cols=_FEATS)
    pd.testing.assert_frame_equal(out, ref)
    assert out["Blue_mean"].dtype == np.float64
    # The inputs are not changed.
    assert "seg_id" in seg_df.columns and pxl_df["Blue_mean"].dtype == np.uint16


def test_combine_cls_samples_default_features_and_columns():
    """Without feat_cols the numeric features must match; group and x/y kept."""
    seg_df = _seg_samples()
    pxl_df = _pxl_samples()
    with pytest.raises(rsgispy.RSGISPyException, match="feature columns"):
        rcd.combine_cls_samples([seg_df, pxl_df])

    pxl_a = _pxl_samples(seed=2, with_xy=True)
    pxl_b = _pxl_samples(n_per_cls=5, seed=3, with_xy=True)
    out = rcd.combine_cls_samples([pxl_a, pxl_b], keep_xy=True, group_col="train_grp")
    assert list(out.columns) == [
        "class_name", "out_cls_id", "internal_cls_id", "train_grp", "x", "y"
    ] + _FEATS
    assert len(out) == len(pxl_a) + len(pxl_b)
    assert out["Red_mean"].dtype == np.uint16
    out_no_xy = rcd.combine_cls_samples([pxl_a, pxl_b])
    assert list(out_no_xy.columns) == ["class_name", "out_cls_id", "internal_cls_id"] + _FEATS

    with pytest.raises(rsgispy.RSGISPyException, match=r"\['x', 'y'\]"):
        rcd.combine_cls_samples([seg_df, pxl_a], feat_cols=_FEATS, keep_xy=True)
    with pytest.raises(rsgispy.RSGISPyException, match="feature column"):
        rcd.combine_cls_samples([seg_df, pxl_df], feat_cols=_FEATS + ["SWIR_mean"])
    with pytest.raises(rsgispy.RSGISPyException, match="train_grp2"):
        rcd.combine_cls_samples([seg_df, pxl_df], feat_cols=_FEATS, group_col="train_grp2")
    with pytest.raises(rsgispy.RSGISPyException):
        rcd.combine_cls_samples(pxl_a)
    with pytest.raises(rsgispy.RSGISPyException):
        rcd.combine_cls_samples([])


def test_combine_cls_samples_inconsistent_ids_raise():
    """One class name with two IDs, or one ID with two names, raises."""
    seg_df = _seg_samples()
    pxl_df = _pxl_samples()
    bad_out = pxl_df.copy()
    bad_out.loc[bad_out["class_name"] == "Water", "out_cls_id"] = 51
    with pytest.raises(rsgispy.RSGISPyException, match="Water.*out_cls_id"):
        rcd.combine_cls_samples([seg_df, bad_out], feat_cols=_FEATS)
    bad_int = pxl_df.copy()
    bad_int.loc[bad_int["class_name"] == "Urban", "internal_cls_id"] = 0
    with pytest.raises(rsgispy.RSGISPyException, match="internal_cls_id"):
        rcd.combine_cls_samples([seg_df, bad_int], feat_cols=_FEATS)
    renamed = pxl_df.copy()
    renamed.loc[renamed["class_name"] == "Urban", "class_name"] = "Built"
    with pytest.raises(rsgispy.RSGISPyException, match="different classes"):
        rcd.combine_cls_samples([seg_df, renamed], feat_cols=_FEATS)
    with pytest.raises(rsgispy.RSGISPyException, match="no column"):
        rcd.combine_cls_samples(
            [seg_df, pxl_df.drop(columns=["internal_cls_id"])], feat_cols=_FEATS
        )


def test_combine_cls_samples_source_col_and_grouped_split():
    """The source column names each row's input and keeps sources apart."""
    seg_df = _seg_samples(n_per_cls=6)
    pxl_df = _pxl_samples()
    out = rcd.combine_cls_samples(
        [seg_df, pxl_df], feat_cols=_FEATS, source_names=["segment", "pixel"],
        source_col="source",
    )
    assert list(out.columns) == [
        "class_name", "out_cls_id", "internal_cls_id", "source"
    ] + _FEATS
    assert (out["source"].iloc[: len(seg_df)] == "segment").all()
    assert (out["source"].iloc[len(seg_df):] == "pixel").all()
    # Default names for DataFrames.
    out_def = rcd.combine_cls_samples([seg_df, pxl_df], feat_cols=_FEATS, source_col="src")
    assert out_def["src"].unique().tolist() == ["input_1", "input_2"]
    with pytest.raises(rsgispy.RSGISPyException, match="source_names"):
        rcd.combine_cls_samples([seg_df, pxl_df], feat_cols=_FEATS, source_names=["a", "b"])
    with pytest.raises(rsgispy.RSGISPyException, match="1 names"):
        rcd.combine_cls_samples(
            [seg_df, pxl_df], feat_cols=_FEATS, source_names=["a"], source_col="source"
        )
    with pytest.raises(rsgispy.RSGISPyException, match="already has"):
        rcd.combine_cls_samples([seg_df, pxl_df], feat_cols=_FEATS, source_col="train_grp")

    # A grouped split on the source column never splits a source.
    many = [_pxl_samples(n_per_cls=4, seed=seed) for seed in range(12)]
    combined = rcd.combine_cls_samples(
        many, source_names=[f"site{idx}" for idx in range(12)], source_col="site"
    )
    parts = rcd.split_class_samples(combined, rnd_seed=1, group_col="site")
    site_sets = [set(part["site"]) for part in parts]
    assert all(site_sets)
    assert not site_sets[0] & site_sets[1]
    assert not site_sets[0] & site_sets[2]
    assert not site_sets[1] & site_sets[2]


def test_combine_cls_samples_file_round_trip(tmp_path):
    """Sample files as inputs (names from the file names) and out_file."""
    seg_file = tmp_path / f"seg_smpls{_EXT}"
    pxl_file = tmp_path / f"pxl_smpls{_EXT}"
    rcd.write_cls_samples(_seg_samples(), seg_file)
    rcd.write_cls_samples(_pxl_samples(), pxl_file)
    out_file = tmp_path / f"combined{_EXT}"
    out = rcd.combine_cls_samples(
        [seg_file, str(pxl_file)], feat_cols=_FEATS, source_col="source",
        out_file=out_file,
    )
    assert out["source"].unique().tolist() == ["seg_smpls", "pxl_smpls"]
    back = rcd.read_cls_samples(out_file)
    pd.testing.assert_frame_equal(back, out, check_dtype=False)
    ref = rcd.combine_cls_samples(
        [rcd.read_cls_samples(seg_file), rcd.read_cls_samples(pxl_file)],
        feat_cols=_FEATS,
    )
    pd.testing.assert_frame_equal(out.drop(columns=["source"]), ref)


def test_combine_cls_samples_train(tmp_path):
    """The combined table splits and trains a classifier."""
    ensemble = pytest.importorskip("sklearn.ensemble")
    import rsgispy.classification.classsklearn as rcsk  # pylint: disable=import-outside-toplevel

    combined = rcd.combine_cls_samples(
        [_seg_samples(n_per_cls=8), _pxl_samples()], feat_cols=_FEATS
    )
    train_df, test_df, valid_df = rcd.split_class_samples(combined, rnd_seed=0)
    assert len(train_df) + len(test_df) + len(valid_df) == len(combined)
    class_info = {
        name: rcd.ClassInfoObj(out_id=out_id, internal_id=int_id)
        for name, (out_id, int_id) in _CLASSES.items()
    }
    img_info = [
        rcd.ImageBandInfo(str(tmp_path / "img.tif"), name="img", bands=[1, 2, 3],
                          band_names=_FEATS)
    ]
    cfg = rcd.ClassificationConfig(img_band_info=img_info, class_info=class_info)
    _, metrics = rcsk.train_sklearn_classifier(
        cfg,
        ensemble.RandomForestClassifier(n_estimators=20, random_state=0),
        train_df,
        test_data=test_df,
        feature_cols=_FEATS,
        rnd_seed=0,
        return_model=True,
    )
    assert metrics
    acc = [val for key, val in metrics.items() if "acc" in key.lower()]
    assert acc and min(acc) > 0.9


def _reg_seg_samples(seed=0):
    """Segment regression samples in the layout of extract_reg_segment_samples."""
    rng = np.random.default_rng(seed)
    n_smpl = 15
    df = pd.DataFrame({
        "seg_id": np.arange(n_smpl, dtype=np.uint32),
        "canopy": rng.uniform(0, 100, n_smpl),
        "height": rng.uniform(0, 30, n_smpl),
        "train_grp": [f"poly{idx}" for idx in range(n_smpl)],
        "n_ref": rng.integers(1, 4, n_smpl),
        "pxl_count": rng.integers(10, 60, n_smpl),
    })
    for feat in _FEATS:
        df[feat] = df["canopy"] * 0.5 + rng.normal(0, 1, n_smpl)
    return df


def _reg_pxl_samples(seed=1):
    """Pixel regression samples in the layout of extract_reg_training_samples."""
    rng = np.random.default_rng(seed)
    n_smpl = 60
    df = pd.DataFrame({
        "canopy": rng.uniform(0, 100, n_smpl),
        "height": rng.uniform(0, 30, n_smpl),
        "train_grp": [f"poly{idx % 7}" for idx in range(n_smpl)],
    })
    for feat in _FEATS:
        df[feat] = df["canopy"] * 0.5 + rng.normal(0, 1, n_smpl)
    return df


def test_combine_reg_samples(tmp_path):
    """Regression: matches concat, checks, source column, files and training."""
    seg_df = _reg_seg_samples()
    pxl_df = _reg_pxl_samples()
    cols = ["canopy"] + _FEATS
    out = rrd.combine_reg_samples([seg_df, pxl_df], "canopy", feat_cols=_FEATS)
    pd.testing.assert_frame_equal(
        out, pd.concat([seg_df[cols], pxl_df[cols]], ignore_index=True)
    )
    multi = rrd.combine_reg_samples(
        [seg_df, pxl_df], ["canopy", "height"], feat_cols=_FEATS,
        source_names=["segment", "pixel"], source_col="source", group_col="train_grp",
    )
    assert list(multi.columns) == ["canopy", "height", "train_grp", "source"] + _FEATS
    assert multi["source"].value_counts().to_dict() == {"pixel": 60, "segment": 15}

    with pytest.raises(rsgispy.RSGISPyException, match="feature columns"):
        rrd.combine_reg_samples([seg_df, pxl_df], ["canopy", "height"])
    with pytest.raises(rsgispy.RSGISPyException, match="no column"):
        rrd.combine_reg_samples([seg_df, pxl_df.drop(columns="height")], ["canopy", "height"],
                                feat_cols=_FEATS)
    with pytest.raises(rsgispy.RSGISPyException):
        rrd.combine_reg_samples([seg_df, pxl_df], "canopy", feat_cols=["canopy"])

    seg_file = tmp_path / f"reg_seg{_EXT}"
    rrd.write_reg_samples(seg_df, seg_file)
    out_file = tmp_path / f"reg_combined{_EXT}"
    from_files = rrd.combine_reg_samples(
        [seg_file, pxl_df], "canopy", feat_cols=_FEATS, source_col="source",
        out_file=out_file,
    )
    assert from_files["source"].unique().tolist() == ["reg_seg", "input_2"]
    pd.testing.assert_frame_equal(
        rrd.read_reg_samples(out_file), from_files, check_dtype=False
    )

    linear = pytest.importorskip("sklearn.linear_model")
    import rsgispy.regression.regsklearn as rrsk  # pylint: disable=import-outside-toplevel

    train_df, test_df, _ = rrd.split_reg_samples(out, rnd_seed=0)
    _, metrics = rrsk.train_sklearn_regressor(
        None, linear.LinearRegression(), train_df, test_data=test_df,
        target_cols="canopy", feature_cols=_FEATS, return_model=True,
    )
    assert metrics
