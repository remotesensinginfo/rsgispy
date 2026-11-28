"""
Direct tests of :func:`rsgispy.ml.segments.assign_vectors_to_segments` and
:func:`rsgispy.classification.classsegments.extract_and_split_class_segment_samples`
on the synthetic segmented scene (``seg_ml_scene``: 60 x 80 pixels, 5 x 5
pixel segments, segment ID = ``(row // 5) * 16 + col // 5 + 1``).
"""

import warnings

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

import rsgispy
import rsgispy.classification.classsegments as cseg
from rsgispy.classification.classdata import (
    ClassificationConfig,
    ClassInfoObj,
    ImageBandInfo,
)
from rsgispy.ml.segments import assign_vectors_to_segments
from seg_ml_scene import CLASS_IDS, EPSG, RES, X0, Y0, make_seg_scene, pix_centre

pytest.importorskip("sklearn")

CRS = f"EPSG:{EPSG}"


def _seg_id(row, col):
    return (row // 5) * 16 + col // 5 + 1


@pytest.fixture(name="seg", scope="module")
def fixture_seg(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("assign")
    scene = make_seg_scene(str(tmp_path / "scene"))
    cls_info = {
        name: ClassInfoObj(
            out_id=10 * CLASS_IDS[name], vec_file=scene[f"train_{name.lower()}"]
        )
        for name in CLASS_IDS
    }
    cfg = ClassificationConfig(
        [ImageBandInfo(scene["s2"], name="s2", band_names=["Blue", "Green", "Red", "NIR"])],
        cls_info,
    )
    cfg_path = str(tmp_path / "cfg.json")
    cfg = cseg.create_cls_segment_features(
        cfg, scene["segs"], str(tmp_path / "segs.parquet"), out_cfg_file=cfg_path,
        band_opts={"s2": {"no_data_val": 0}}, use_pbar=False,
    )
    return {"scene": scene, "cfg": cfg, "cfg_path": cfg_path, "dir": tmp_path}


def test_assign_points(seg):
    seg_info = seg["cfg"].seg_info
    rc_list = [(2, 2), (2, 3), (12, 33), (59, 79)]
    pts = gpd.GeoDataFrame(
        {"val": [1.0, 2.0, 3.0, 4.0, 5.0], "name": list("abcde")},
        # the last point is outside the image (east of it): not assigned
        geometry=[pix_centre(r, c) for r, c in rc_list] + [pix_centre(5, 85)],
        crs=CRS,
    )
    hits = assign_vectors_to_segments(seg_info, pts, value_cols=["val", "name"],
                                      group_prefix="p_")
    assert list(hits.columns) == ["seg_id", "train_grp", "src_type", "val", "name"]
    assert hits["seg_id"].tolist() == [_seg_id(r, c) for r, c in rc_list]
    assert hits["val"].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert hits["name"].tolist() == list("abcd")
    assert (hits["src_type"] == "point").all()
    # One group per segment: both points in segment 1 share a group.
    assert hits["train_grp"].tolist() == [f"p_seg{_seg_id(r, c)}" for r, c in rc_list]


def test_assign_polygons_overlap_and_centroid(seg):
    seg_info = seg["cfg"].seg_info
    # Polygon 0: rows 45-50, cols 15-27 -> segments (9, 3) and (9, 4) fully,
    # (9, 5) 40 %. Polygon 1: covers 40 % of segment 1 (2 of 5 columns).
    polys = gpd.GeoDataFrame(
        {"cover": [55.0, 10.0]},
        geometry=[
            box(X0 + 15 * RES, Y0 - 50 * RES, X0 + 27 * RES, Y0 - 45 * RES),
            box(X0, Y0 - 5 * RES, X0 + 2 * RES, Y0),
        ],
        crs=CRS,
    )
    full = [_seg_id(45, 15), _seg_id(45, 20)]
    part = _seg_id(45, 25)

    hits = assign_vectors_to_segments(seg_info, polys, value_cols=["cover"],
                                      poly_assign="overlap", min_overlap=0.5)
    assert sorted(hits["seg_id"]) == full
    assert (hits["train_grp"] == "poly0").all()  # one group per polygon
    assert (hits["src_type"] == "polygon").all() and (hits["cover"] == 55.0).all()

    low = assign_vectors_to_segments(seg_info, polys, value_cols=["cover"],
                                     poly_assign="overlap", min_overlap=0.3)
    assert sorted(low.loc[low["train_grp"] == "poly0", "seg_id"]) == full + [part]
    assert low.loc[low["train_grp"] == "poly1", "seg_id"].tolist() == [1]
    assert low.loc[low["train_grp"] == "poly1", "cover"].tolist() == [10.0]

    cent = assign_vectors_to_segments(seg_info, polys, poly_assign="centroid")
    assert list(cent.columns) == ["seg_id", "train_grp", "src_type"]
    # The partly covered segment's centroid (col 27.5) is outside polygon 0,
    # and segment 1's centroid (col 2.5) is outside polygon 1 (cols 0-2).
    assert sorted(cent["seg_id"]) == full


def test_assign_mixed_reprojection_and_errors(seg, tmp_path):
    seg_info = seg["cfg"].seg_info
    mixed = gpd.GeoDataFrame(
        {"v": [1, 2]},
        geometry=[pix_centre(32, 47), box(X0, Y0 - 10 * RES, X0 + 10 * RES, Y0)],
        crs=CRS,
    )
    hits = assign_vectors_to_segments(seg_info, mixed, value_cols=["v"])
    assert hits.loc[hits["src_type"] == "point", "seg_id"].tolist() == [_seg_id(32, 47)]
    assert sorted(hits.loc[hits["src_type"] == "polygon", "seg_id"]) == [1, 2, 17, 18]

    wgs_path = str(tmp_path / "mixed_wgs84.gpkg")
    mixed.to_crs("EPSG:4326").to_file(wgs_path)
    with pytest.raises(rsgispy.RSGISPyException, match="reproj_vec"):
        assign_vectors_to_segments(seg_info, wgs_path)
    reproj = assign_vectors_to_segments(seg_info, wgs_path, value_cols=["v"], reproj_vec=True)
    pd.testing.assert_frame_equal(
        reproj.sort_values(["src_type", "seg_id"]).reset_index(drop=True),
        hits.sort_values(["src_type", "seg_id"]).reset_index(drop=True),
        check_dtype=False,
    )

    # No hits -> an empty table with the output columns.
    far = gpd.GeoDataFrame({"v": [1]}, geometry=[pix_centre(-50, -50)], crs=CRS)
    empty = assign_vectors_to_segments(seg_info, far, value_cols=["v"])
    assert empty.empty and list(empty.columns) == ["seg_id", "train_grp", "src_type", "v"]

    with pytest.raises(rsgispy.RSGISPyException, match="not found"):
        assign_vectors_to_segments(seg_info, mixed, value_cols=["missing"])
    with pytest.raises(rsgispy.RSGISPyException, match="poly_assign"):
        assign_vectors_to_segments(seg_info, mixed, poly_assign="bad")
    no_lab = seg_info.copy()
    no_lab.label_img = None
    with pytest.raises(rsgispy.RSGISPyException, match="label_img"):
        assign_vectors_to_segments(no_lab, mixed)


def test_extract_and_split_class_segment_samples(seg, tmp_path, monkeypatch):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        train, test, valid = cseg.extract_and_split_class_segment_samples(
            seg["cfg_path"],
            out_train_file=str(tmp_path / "train.parquet"),
            out_test_file=str(tmp_path / "test.parquet"),
            out_valid_file=str(tmp_path / "valid.parquet"),
            out_samples_file=str(tmp_path / "samples.parquet"),
            test_prop=0.2, valid_prop=0.2, rnd_seed=3,
        )
        samples = cseg.extract_class_segment_samples(seg["cfg_path"])
        ref = cseg.split_class_segment_samples(
            samples, test_prop=0.2, valid_prop=0.2, rnd_seed=3,
            stratify_col="internal_cls_id",
        )
    # Same as extracting then splitting (stratified by class, seeded).
    for got, exp in zip((train, test, valid), ref):
        assert sorted(got["seg_id"]) == sorted(exp["seg_id"])
    parts = pd.concat([train, test, valid])
    assert parts["seg_id"].is_unique
    assert sorted(parts["seg_id"]) == sorted(samples["seg_id"])
    for part in (train, test, valid):
        assert set(part["class_name"]) == set(CLASS_IDS)  # stratified
    assert len(train) > len(test) > 0 and len(valid) > 0
    # Outputs written.
    for name, part in (("train", train), ("test", test), ("valid", valid),
                       ("samples", samples)):
        back = pd.read_parquet(tmp_path / f"{name}.parquet")
        assert sorted(back["seg_id"]) == sorted(part["seg_id"])

    # Extra keyword arguments are passed on to the extraction.
    seen = {}
    orig = cseg.extract_class_segment_samples

    def _spy(cfg, out_file=None, **kwargs):
        seen.update(kwargs, out_file=out_file)
        return orig(cfg, out_file=out_file, **kwargs)

    monkeypatch.setattr(cseg, "extract_class_segment_samples", _spy)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cseg.extract_and_split_class_segment_samples(
            seg["cfg_path"], test_prop=0.2, valid_prop=0.0, rnd_seed=0,
            min_overlap=0.3, poly_assign="centroid",
        )
    assert seen == {"out_file": None, "min_overlap": 0.3, "poly_assign": "centroid"}
