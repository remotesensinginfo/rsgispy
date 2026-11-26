"""
Tests for rsgispy.zonalstats.labels (label raster zonal statistics, joining
and rasterising per-label values).
"""

import os

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

import rsgispy
import rsgispy.segmentation as rseg
import rsgispy.segmentation.utils as seg_utils
import rsgispy.zonalstats.labels as zs_lbl
import rsgispy.zonalstats.polygon as zs_poly
from rsgispy.ml.stack import BandSpec


def _write(path, arr, nodata=None, res=10.0, x0=500000.0):
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[np.newaxis]
    count, height, width = arr.shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=arr.dtype,
        crs="EPSG:32630",
        transform=from_origin(x0, 5200000.0, res, res),
        nodata=nodata,
    ) as dst:
        dst.write(arr)
    return str(path)


def _blocky_labels(height, width, blk, dtype=np.uint32):
    lab = (np.arange(height)[:, None] // blk) * (width // blk + 1) + (
        np.arange(width)[None, :] // blk
    ) + 1
    return lab.astype(dtype)


@pytest.fixture(name="scene")
def fixture_scene(tmp_path):
    rng = np.random.default_rng(7)
    height, width = 45, 60
    lab = _blocky_labels(height, width, 6)
    lab[:4, :4] = 0
    fvals = rng.normal(100.0, 20.0, (2, height, width)).astype(np.float32)
    fvals[0, 10:14, 10:14] = -9999.0
    # One label (the block at rows 36..41, cols 54..59) entirely no data in band 1
    fvals[0, 36:42, 54:60] = -9999.0
    ivals = rng.integers(0, 12, (height, width)).astype(np.int16)
    return {
        "lab": _write(tmp_path / "lab.tif", lab),
        "fval": _write(tmp_path / "fval.tif", fvals, nodata=-9999.0),
        "ival": _write(tmp_path / "ival.tif", ivals),
        "lab_arr": lab,
        "fvals": fvals,
        "ivals": ivals,
    }


def test_hand_calculation(tmp_path):
    lab = np.array(
        [
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [3, 3, 0, 0],
            [3, 3, 0, 0],
        ],
        dtype=np.uint8,
    )
    vals = np.array(
        [
            [1, 2, 10, 10],
            [3, 4, 20, 99],
            [-1, -1, 5, 5],
            [-1, -1, 5, 5],
        ],
        dtype=np.float32,
    )
    lab_img = _write(tmp_path / "l.tif", lab)
    val_img = _write(tmp_path / "v.tif", vals, nodata=-1.0)
    df = zs_lbl.calc_label_zonal_stats(
        lab_img,
        BandSpec(val_img, name="v", max_vld_val=50.0),
        stats=["count", "sum", "mean", "stddev", "min", "max", "median", "mode"],
        percentiles=[25],
        quantile_method="exact",
        inc_bbox=True,
        inc_centroid=True,
        use_pbar=False,
    )
    assert df["seg_id"].tolist() == [1, 2, 3]
    assert df["pxl_count"].tolist() == [4, 4, 4]
    row1 = df.iloc[0]
    assert row1["v_b1_count"] == 4
    assert row1["v_b1_sum"] == pytest.approx(10.0)
    assert row1["v_b1_mean"] == pytest.approx(2.5)
    assert row1["v_b1_stddev"] == pytest.approx(np.std([1, 2, 3, 4]))
    assert row1["v_b1_min"] == 1.0 and row1["v_b1_max"] == 4.0
    assert row1["v_b1_median"] == pytest.approx(2.5)
    assert row1["v_b1_p25"] == pytest.approx(np.percentile([1, 2, 3, 4], 25))
    assert row1["v_b1_mode"] == 1.0
    # Label 2: 99 is above max_vld_val so excluded
    row2 = df.iloc[1]
    assert row2["v_b1_count"] == 3
    assert row2["v_b1_mean"] == pytest.approx(40.0 / 3.0)
    assert row2["v_b1_mode"] == 10.0
    # Label 3: all no data
    row3 = df.iloc[2]
    assert row3["v_b1_count"] == 0
    for stat in ("sum", "mean", "stddev", "min", "max", "median", "mode", "p25"):
        assert row3[f"v_b1_{stat}"] == -9999.0
    assert (row3["min_row"], row3["max_row"], row3["min_col"], row3["max_col"]) == (
        2,
        3,
        0,
        1,
    )
    # Label 1 centroid: pixel centres cols 0..1, rows 0..1 -> col 1.0, row 1.0
    assert row1["centroid_x"] == pytest.approx(500000.0 + 10.0)
    assert row1["centroid_y"] == pytest.approx(5200000.0 - 10.0)


@pytest.mark.parametrize("block_size", [7, 16, 64])
def test_matches_polygon_zonal_stats(scene, block_size):
    df = zs_lbl.calc_label_zonal_stats(
        scene["lab"],
        [BandSpec(scene["fval"], name="f"), BandSpec(scene["ival"], name="i")],
        stats=["count", "sum", "mean", "stddev", "min", "max", "median", "mode"],
        block_size=block_size,
        use_pbar=False,
    )
    gdf = rseg.polygonise_segments(scene["lab"])
    for band in (1, 2):
        gdf = zs_poly.calc_zonal_band_stats_file(
            gdf,
            None,
            scene["fval"],
            band,
            -1e12,
            1e12,
            -9999.0,
            min_field=f"pf{band}_min",
            max_field=f"pf{band}_max",
            mean_field=f"pf{band}_mean",
            stddev_field=f"pf{band}_std",
            sum_field=f"pf{band}_sum",
            count_field=f"pf{band}_cnt",
            use_pbar=False,
        )
    gdf = zs_poly.calc_zonal_band_stats_file(
        gdf,
        None,
        scene["ival"],
        1,
        -1e12,
        1e12,
        -9999.0,
        mode_field="pi_mode",
        median_field="pi_med",
        mean_field="pi_mean",
        use_pbar=False,
    )
    merged = gdf.merge(df, on="seg_id")
    assert len(merged) == len(df) == len(gdf)
    for band in (1, 2):
        for lbl_stat, poly_stat in (
            ("min", "min"),
            ("max", "max"),
            ("mean", "mean"),
            ("stddev", "std"),
            ("sum", "sum"),
            ("count", "cnt"),
        ):
            np.testing.assert_allclose(
                merged[f"f_b{band}_{lbl_stat}"],
                merged[f"pf{band}_{poly_stat}"],
                rtol=1e-9,
                atol=1e-6,
            )
    np.testing.assert_allclose(merged["i_b1_mode"], merged["pi_mode"])
    np.testing.assert_allclose(merged["i_b1_median"], merged["pi_med"])
    np.testing.assert_allclose(merged["i_b1_mean"], merged["pi_mean"])
    # The all-no-data label in band 1
    assert (df["f_b1_count"] == 0).sum() == 1


def test_mixed_dtype_vrt_matches_separate_images(tmp_path, scene):
    """A VRT stacking int16 and float32 bands gives the same statistics."""
    from rsgispy.imageutils.vrt import create_stack_images_vrt

    vrt = str(tmp_path / "mixed.vrt")
    create_stack_images_vrt([scene["fval"], scene["ival"]], vrt)
    with rasterio.open(vrt) as src:
        assert src.dtypes == ("float32", "float32", "int16")
    kwargs = dict(
        stats=["count", "mean", "min", "max", "median", "mode"],
        block_size=16,
        use_pbar=False,
    )
    df_vrt = zs_lbl.calc_label_zonal_stats(
        scene["lab"], BandSpec(vrt, name="v"), **kwargs
    )
    df_sep = zs_lbl.calc_label_zonal_stats(
        scene["lab"],
        [BandSpec(scene["fval"], name="f"), BandSpec(scene["ival"], name="i")],
        **kwargs,
    )
    for stat in kwargs["stats"]:
        for vrt_col, sep_col in (("v_b1", "f_b1"), ("v_b2", "f_b2"), ("v_b3", "i_b1")):
            np.testing.assert_array_equal(
                df_vrt[f"{vrt_col}_{stat}"], df_sep[f"{sep_col}_{stat}"]
            )


def test_block_size_invariance_and_hist_quantiles(scene):
    kwargs = dict(
        stats=["mean", "stddev", "median"],
        percentiles=[10, 90],
        use_pbar=False,
    )
    spec = BandSpec(scene["fval"], name="f", bands=(2,))
    df_a = zs_lbl.calc_label_zonal_stats(scene["lab"], spec, block_size=5, **kwargs)
    df_b = zs_lbl.calc_label_zonal_stats(scene["lab"], spec, block_size=512, **kwargs)
    pd.testing.assert_frame_equal(df_a, df_b, check_exact=False, rtol=1e-10)

    df_exact = zs_lbl.calc_label_zonal_stats(
        scene["lab"], spec, quantile_method="exact", **kwargs
    )
    lab = scene["lab_arr"]
    vals = scene["fvals"][1]
    for _, row in df_exact.iterrows():
        pix = vals[lab == row["seg_id"]].astype(np.float64)
        assert row["f_b2_median"] == pytest.approx(np.median(pix))
        assert row["f_b2_p10"] == pytest.approx(np.percentile(pix, 10))
        assert row["f_b2_p90"] == pytest.approx(np.percentile(pix, 90))
    # Histogram approximation within half a bin width
    v_rng = float(vals.max() - vals.min())
    err = np.abs(df_a["f_b2_median"] - df_exact["f_b2_median"]).max()
    assert err <= 0.5 * v_rng / 1000 + 1e-9


def test_sparse_large_uint64_ids(tmp_path, scene):
    lab = scene["lab_arr"]
    big = np.where(lab == 0, 0, lab.astype(np.uint64) * np.uint64(2**40)).astype(
        np.uint64
    )
    big_img = _write(tmp_path / "big.tif", big)
    spec = BandSpec(scene["fval"], name="f")
    df_small = zs_lbl.calc_label_zonal_stats(scene["lab"], spec, use_pbar=False)
    df_big = zs_lbl.calc_label_zonal_stats(big_img, spec, use_pbar=False)
    assert df_big["seg_id"].dtype == np.uint64
    np.testing.assert_array_equal(
        df_big["seg_id"].to_numpy(), df_small["seg_id"].to_numpy().astype(np.uint64) * 2**40
    )
    pd.testing.assert_frame_equal(
        df_big.drop(columns="seg_id"), df_small.drop(columns="seg_id")
    )
    # Forcing the compact (non-dense) mapping gives identical results
    df_cmp = zs_lbl.calc_label_zonal_stats(
        scene["lab"], spec, dense_max_labels=1, use_pbar=False
    )
    pd.testing.assert_frame_equal(df_cmp, df_small)


def test_label_indexer_dense_and_sparse():
    ids = np.array([3, 7, 9], dtype=np.int64)
    dense = zs_lbl._LabelIndexer(ids, dense_max_labels=100)
    sparse = zs_lbl._LabelIndexer(ids, dense_max_labels=1)
    assert dense.is_dense and not sparse.is_dense
    probe = np.array([3, 4, 9, 100, 7, -1])
    for idx_obj in (dense, sparse):
        idx, found = idx_obj.lookup(probe)
        assert found.tolist() == [True, False, True, False, True, False]
        assert idx[found].tolist() == [0, 2, 1]


def test_categorical_band(tmp_path, scene):
    cat = np.zeros_like(scene["lab_arr"], dtype=np.uint8)
    cat[:, 30:] = 2
    cat[20:, :] = 5
    cat_img = _write(tmp_path / "cat.tif", cat)
    spec = BandSpec(
        cat_img, name="lc", band_names=("lc",), data_type=rsgispy.VAR_TYPE_CATEGORICAL
    )
    df = zs_lbl.calc_label_zonal_stats(
        scene["lab"], spec, stats=["mean", "count"], use_pbar=False
    )
    assert "lc_mean" not in df.columns
    assert df.attrs["categories"] == {"lc": [0.0, 2.0, 5.0]}
    lab = scene["lab_arr"]
    for _, row in df.iterrows():
        pix = cat[lab == row["seg_id"]]
        vals, cnts = np.unique(pix, return_counts=True)
        assert row["lc_mode"] == vals[np.argmax(cnts)]
        assert row["lc_count"] == pix.size
        for cval in (0, 2, 5):
            assert row[f"lc_prop_{cval}"] == pytest.approx(np.mean(pix == cval))
    props = df[["lc_prop_0", "lc_prop_2", "lc_prop_5"]].sum(axis=1)
    np.testing.assert_allclose(props, 1.0)

    # Fixed category list (as for a new scene): unseen 5 ignored, missing 9 -> 0
    df_fixed = zs_lbl.calc_label_zonal_stats(
        scene["lab"], spec, categories={"lc": [0, 2, 9]}, use_pbar=False
    )
    assert [c for c in df_fixed.columns if "_prop_" in c] == [
        "lc_prop_0",
        "lc_prop_2",
        "lc_prop_9",
    ]
    assert (df_fixed["lc_prop_9"] == 0).all()


def test_grid_mismatch_and_validation(tmp_path, scene):
    other = _write(tmp_path / "shift.tif", scene["fvals"], x0=500005.0)
    with pytest.raises(rsgispy.RSGISPyException, match="same pixel grid"):
        zs_lbl.calc_label_zonal_stats(scene["lab"], other, use_pbar=False)
    with pytest.raises(rsgispy.RSGISPyException, match="Unsupported"):
        zs_lbl.calc_label_zonal_stats(
            scene["lab"], scene["fval"], stats=["nope"], use_pbar=False
        )
    with pytest.raises(rsgispy.RSGISPyException, match="integer"):
        zs_lbl.calc_label_zonal_stats(scene["fval"], scene["fval"], use_pbar=False)


def test_table_output_and_join(tmp_path, scene):
    out_pq = str(tmp_path / "stats.parquet")
    df = zs_lbl.calc_label_zonal_stats(
        scene["lab"],
        BandSpec(scene["fval"], name="f", bands=(1,)),
        out_file=out_pq,
        use_pbar=False,
    )
    assert os.path.exists(out_pq)
    pd.testing.assert_frame_equal(pd.read_parquet(out_pq), df, check_dtype=False)

    polys = rseg.polygonise_segments(scene["lab"])
    joined = zs_lbl.join_label_stats_to_vec(polys, out_pq)
    assert isinstance(joined, gpd.GeoDataFrame)
    assert len(joined) == len(polys)
    assert joined["seg_id"].tolist() == polys["seg_id"].tolist()
    assert "f_b1_mean" in joined.columns

    out_vec = str(tmp_path / "joined.gpkg")
    zs_lbl.join_label_stats_to_vec(polys, df, out_vec_file=out_vec)
    assert os.path.exists(out_vec)
    with pytest.raises(rsgispy.RSGISPyException, match="already exist"):
        zs_lbl.join_label_stats_to_vec(joined, df)


def test_rasterise_label_values(tmp_path, scene):
    lab = scene["lab_arr"]
    ids = np.unique(lab[lab != 0])
    cls_vals = (ids % 3 + 1).astype(np.uint8)
    out_cls = str(tmp_path / "cls.tif")
    zs_lbl.rasterise_label_values(
        scene["lab"],
        ids[:-1],
        cls_vals[:-1],
        out_cls,
        thematic=True,
        clr_tab={1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255)},
        block_size=8,
        use_pbar=False,
    )
    lut = dict(zip(ids[:-1].tolist(), cls_vals[:-1].tolist()))
    expected = np.vectorize(lambda v: lut.get(int(v), 0))(lab).astype(np.uint8)
    with rasterio.open(out_cls) as src:
        assert src.dtypes[0] == "uint8"
        assert src.nodata == 0
        np.testing.assert_array_equal(src.read(1), expected)
        assert src.colormap(1)[2][:3] == (0, 255, 0)

    # Multi-band float, compact mapping, non-zero nodata
    reg_vals = np.column_stack([ids * 0.5, ids * -1.0])
    out_reg = str(tmp_path / "reg.tif")
    zs_lbl.rasterise_label_values(
        scene["lab"],
        ids,
        reg_vals,
        out_reg,
        out_no_data_val=-9999.0,
        band_names=["half", "neg"],
        dense_max_labels=1,
        use_pbar=False,
    )
    with rasterio.open(out_reg) as src:
        arr = src.read()
        assert src.dtypes[0] == "float32"
        assert src.descriptions == ("half", "neg")
        assert src.nodata == -9999.0
    np.testing.assert_allclose(arr[0][lab != 0], lab[lab != 0] * 0.5)
    assert np.all(arr[1][lab == 0] == -9999.0)

    with pytest.raises(rsgispy.RSGISPyException, match="duplicate"):
        zs_lbl.rasterise_label_values(
            scene["lab"], [1, 1], [1, 2], str(tmp_path / "x.tif"), use_pbar=False
        )


def test_extract_segment_stats_fast_path_matches_polygon(tmp_path, scene, monkeypatch):
    fast = rseg.extract_segment_stats(
        scene["lab"], scene["fval"], bands=[1, 2], use_pbar=False
    )
    monkeypatch.setattr(
        zs_lbl, "_label_is_grid_compatible", lambda *args, **kwargs: False
    )
    slow = seg_utils.extract_segment_stats(
        scene["lab"], scene["fval"], bands=[1, 2], use_pbar=False
    )
    assert list(fast.columns) == list(slow.columns)
    pd.testing.assert_frame_equal(
        pd.DataFrame(fast.drop(columns="geometry")),
        pd.DataFrame(slow.drop(columns="geometry")),
        check_exact=False,
        rtol=1e-9,
    )
