"""
Tests for rsgispy.classification (classrefpts and classaccuracymetrics).
"""

import csv
import inspect
import json

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point

import rsgispy
import rsgispy.classification as rcls
import rsgispy.classification.classaccuracymetrics as rcam
import rsgispy.classification.classrefpts as rcrp


def _write_raster(
    path: str,
    arr: np.ndarray,
    origin_x: float = 500000.0,
    origin_y: float = 5600000.0,
    pxl_size: float = 10.0,
    crs: str = "EPSG:32630",
    nodata: float = 0.0,
) -> None:
    """Write a 2D or 3D GeoTIFF classification raster for testing."""
    if arr.ndim == 2:
        count = 1
        height, width = arr.shape
        write_arr = arr[None, :, :]
    else:
        count, height, width = arr.shape
        write_arr = arr

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=count,
        dtype=str(arr.dtype),
        crs=crs,
        transform=from_origin(origin_x, origin_y, pxl_size, pxl_size),
        nodata=nodata,
    ) as dst:
        dst.write(write_arr)


def test_package_layout():
    """Verify submodules exist and expose expected functions without root leaks."""
    assert not hasattr(rcls, "calc_acc_stats_img")
    assert not hasattr(rcls, "create_random_ref_smpls_darts")
    assert inspect.ismodule(rcrp)
    assert inspect.ismodule(rcam)
    for fn in (
        "create_random_ref_smpls_darts",
        "create_stratified_random_ref_smpls_darts",
        "create_stratified_random_ref_smpls_lut",
        "pop_class_info_accuracy_pts",
    ):
        assert callable(getattr(rcrp, fn))
    for fn in (
        "calc_acc_stats_img",
        "calc_acc_stats_ptonly",
        "calc_acc_stats_img_bootstrap_conf",
        "calc_acc_stats_ptonly_bootstrap_conf",
    ):
        assert callable(getattr(rcam, fn))


def test_create_random_ref_smpls_darts(tmp_path):
    """Test simple random reference sampling with and without class LUT."""
    arr = np.zeros((20, 20), dtype=np.uint8)
    arr[2:12, 2:12] = 1  # 100 pixels of class 1
    arr[12:18, 2:18] = 2  # 96 pixels of class 2
    img_path = str(tmp_path / "cls.tif")
    _write_raster(img_path, arr, nodata=0.0)

    out_gpkg = str(tmp_path / "rand_pts.gpkg")
    lut = {1: "Forest", 2: "Water"}
    rcrp.create_random_ref_smpls_darts(
        input_img=img_path,
        n_smpls=50,
        out_vec_file=out_gpkg,
        out_vec_lyr="rand_pts",
        rnd_seed=42,
        class_names_lut=lut,
    )
    pts_gdf = gpd.read_file(out_gpkg, layer="rand_pts")
    assert len(pts_gdf) == 50
    assert set(pts_gdf["img_cls"].unique()).issubset({"Forest", "Water"})
    assert (pts_gdf["cls_ref"] == "").all()
    assert (pts_gdf["Processed"] == 0).all()
    # Without replacement: all point coordinates must be unique pixel centres.
    coords = {(round(g.x, 6), round(g.y, 6)) for g in pts_gdf.geometry}
    assert len(coords) == 50
    for x_coord, y_coord in coords:
        assert (x_coord - 500005.0) % 10.0 == pytest.approx(0.0)
        assert (5600000.0 - y_coord - 5.0) % 10.0 == pytest.approx(0.0)

    # Reproducibility + in-memory GeoDataFrame return without LUT.
    gdf_a = rcrp.create_random_ref_smpls_darts(
        input_img=img_path,
        n_smpls=30,
        out_vec_file=None,
        rnd_seed=99,
        class_names_lut=None,
    )
    gdf_b = rcrp.create_random_ref_smpls_darts(
        input_img=img_path,
        n_smpls=30,
        out_vec_file=None,
        rnd_seed=99,
        class_names_lut=None,
    )
    assert len(gdf_a) == 30
    assert (gdf_a["img_cls"] == gdf_a["cls_ref"]).all()
    assert gdf_a["img_cls"].tolist() == gdf_b["img_cls"].tolist()

    with pytest.raises(rsgispy.RSGISPyException):
        rcrp.create_random_ref_smpls_darts(img_path, 0, None)
    with pytest.raises(rsgispy.RSGISPyException):
        rcrp.create_random_ref_smpls_darts(img_path, 500, None)


def test_create_stratified_random_ref_smpls_darts(tmp_path):
    """Test proportional, equal, and min-per-class stratified sampling."""
    arr = np.zeros((20, 20), dtype=np.uint8)
    arr[0:15, 0:15] = 1  # 225 pixels
    arr[15:20, 0:15] = 2  # 75 pixels
    arr[0:2, 15:18] = 3  # 6 pixels (rare class)
    img_path = str(tmp_path / "strat_cls.tif")
    _write_raster(img_path, arr, nodata=0.0)
    lut = {1: "Forest", 2: "Grass", 3: "Urban"}

    # Equal allocation with min_cls_smpls exceeding rare class pixel count (6).
    out_eq = str(tmp_path / "strat_eq.gpkg")
    rcrp.create_stratified_random_ref_smpls_darts(
        input_img=img_path,
        n_smpls=60,
        out_vec_file=out_eq,
        out_vec_lyr="strat_eq",
        eq_n_smpls=True,
        min_cls_smpls=10,
        rnd_seed=7,
        class_names_lut=lut,
        verbose=False,
    )
    eq_gdf = gpd.read_file(out_eq, layer="strat_eq")
    counts_eq = eq_gdf["img_cls"].value_counts().to_dict()
    assert counts_eq == {"Forest": 20, "Grass": 20, "Urban": 6}

    # Proportional allocation.
    prop_gdf = rcrp.create_stratified_random_ref_smpls_darts(
        input_img=img_path,
        n_smpls=100,
        out_vec_file=None,
        eq_n_smpls=False,
        min_cls_smpls=4,
        rnd_seed=7,
        class_names_lut=lut,
        verbose=False,
    )
    counts_prop = prop_gdf["img_cls"].value_counts().to_dict()
    assert counts_prop["Forest"] > counts_prop["Grass"]
    assert counts_prop["Urban"] >= 4
    assert len({(round(g.x, 6), round(g.y, 6)) for g in prop_gdf.geometry}) == len(
        prop_gdf
    )


def test_create_stratified_random_ref_smpls_lut(tmp_path):
    """Test hybrid rare-LUT and common-darts stratified sampling (RAM & tiled)."""
    arr = np.zeros((100, 100), dtype=np.uint8)
    arr[0:50, 0:90] = 1  # 4500 pixels (common)
    arr[50:80, 0:100] = 2  # 3000 pixels (common)
    arr[95:97, 95:98] = 3  # 6 pixels (rare)
    img_path = str(tmp_path / "lut_cls.tif")
    _write_raster(img_path, arr, nodata=0.0)
    lut = {1: "Forest", 2: "Grass", 3: "Wetland"}

    # Hybrid in-memory path (class 3 uses LUT, classes 1 & 2 use darts)
    out_gpkg = str(tmp_path / "strat_lut_mem.gpkg")
    rcrp.create_stratified_random_ref_smpls_lut(
        input_img=img_path,
        n_smpls=90,
        out_vec_file=out_gpkg,
        out_vec_lyr="lut_pts",
        eq_n_smpls=True,
        min_cls_smpls=10,
        rare_pxl_thres=100,
        rnd_seed=17,
        class_names_lut=lut,
        in_memory=True,
        verbose=False,
    )
    mem_gdf = gpd.read_file(out_gpkg, layer="lut_pts")
    assert mem_gdf["img_cls"].value_counts().to_dict() == {
        "Forest": 30,
        "Grass": 30,
        "Wetland": 6,
    }
    mem_coords = {(round(g.x, 6), round(g.y, 6)) for g in mem_gdf.geometry}
    assert len(mem_coords) == len(mem_gdf)
    for x_coord, y_coord in mem_coords:
        assert (x_coord - 500005.0) % 10.0 == pytest.approx(0.0)
        assert (5600000.0 - y_coord - 5.0) % 10.0 == pytest.approx(0.0)

    # Tiled out-of-core path (block_size=16) with proportional allocation
    tiled_gdf_a = rcrp.create_stratified_random_ref_smpls_lut(
        input_img=img_path,
        n_smpls=100,
        out_vec_file=None,
        eq_n_smpls=False,
        min_cls_smpls=5,
        rare_pxl_thres=100,
        rnd_seed=23,
        class_names_lut=None,
        in_memory=False,
        block_size=16,
        use_pbar=False,
        verbose=False,
    )
    tiled_gdf_b = rcrp.create_stratified_random_ref_smpls_lut(
        input_img=img_path,
        n_smpls=100,
        out_vec_file=None,
        eq_n_smpls=False,
        min_cls_smpls=5,
        rare_pxl_thres=100,
        rnd_seed=23,
        class_names_lut=None,
        in_memory=False,
        block_size=16,
        use_pbar=False,
        verbose=False,
    )
    counts_tiled = tiled_gdf_a["img_cls"].value_counts().to_dict()
    assert counts_tiled[1] > counts_tiled[2]
    assert counts_tiled[3] == 5
    assert (tiled_gdf_a["img_cls"] == tiled_gdf_a["cls_ref"]).all()
    assert tiled_gdf_a["img_cls"].tolist() == tiled_gdf_b["img_cls"].tolist()
    assert len({(round(g.x, 6), round(g.y, 6)) for g in tiled_gdf_a.geometry}) == len(
        tiled_gdf_a
    )

    # Edge cases: rare_pxl_thres=0 (all darts) & large rare_pxl_thres (all LUT)
    all_darts = rcrp.create_stratified_random_ref_smpls_lut(
        input_img=img_path,
        n_smpls=45,
        out_vec_file=None,
        eq_n_smpls=True,
        rare_pxl_thres=0,
        rnd_seed=5,
        class_names_lut=lut,
        in_memory=False,
        block_size=32,
        use_pbar=False,
        verbose=False,
    )
    assert all_darts["img_cls"].value_counts().to_dict() == {
        "Forest": 15,
        "Grass": 15,
        "Wetland": 6,
    }
    all_lut = rcrp.create_stratified_random_ref_smpls_lut(
        input_img=img_path,
        n_smpls=45,
        out_vec_file=None,
        eq_n_smpls=True,
        rare_pxl_thres=rcrp.DEFAULT_RARE_PXL_THRES,
        rnd_seed=5,
        class_names_lut=lut,
        in_memory=True,
        verbose=False,
    )
    assert all_lut["img_cls"].value_counts().to_dict() == {
        "Forest": 15,
        "Grass": 15,
        "Wetland": 6,
    }

    # Validation checks
    with pytest.raises(rsgispy.RSGISPyException):
        rcrp.create_stratified_random_ref_smpls_lut(
            img_path, 10, None, rare_pxl_thres=-1, verbose=False
        )
    with pytest.raises(rsgispy.RSGISPyException):
        rcrp.create_stratified_random_ref_smpls_lut(
            img_path, 0, None, verbose=False
        )
    with pytest.raises(rsgispy.RSGISPyException):
        rcrp.create_stratified_random_ref_smpls_lut(
            img_path, 10, None, block_size=0, verbose=False
        )
    empty_path = str(tmp_path / "empty.tif")
    _write_raster(empty_path, np.zeros((10, 10), dtype=np.uint8), nodata=0.0)
    with pytest.raises(rsgispy.RSGISPyException):
        rcrp.create_stratified_random_ref_smpls_lut(
            empty_path, 10, None, in_memory=False, use_pbar=False, verbose=False
        )


def test_pop_class_info_accuracy_pts(tmp_path):
    """Test populating existing points, including out-of-bounds and in-place overwrite."""
    mb = np.zeros((2, 4, 4), dtype=np.uint8)
    mb[1] = np.array(
        [
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [3, 3, 0, 0],
            [3, 3, 0, 0],
        ],
        dtype=np.uint8,
    )
    img_path = str(tmp_path / "mb_cls.tif")
    _write_raster(
        img_path,
        mb,
        origin_x=500000.0,
        origin_y=5600040.0,
        pxl_size=10.0,
        crs="EPSG:32630",
        nodata=0.0,
    )

    pts = gpd.GeoDataFrame(
        {"id": [1, 2, 3, 4]},
        geometry=[
            Point(500005.0, 5600035.0),  # (0, 0) -> class 1 ("Forest")
            Point(500025.0, 5600035.0),  # (0, 2) -> class 2 ("Water")
            Point(500005.0, 5600015.0),  # (2, 0) -> class 3 ("Urban")
            Point(500100.0, 5600100.0),  # outside image -> ""
        ],
        crs="EPSG:32630",
    )
    in_pts = str(tmp_path / "pts.gpkg")
    pts.to_file(in_pts, layer="pts", driver="GPKG")

    lut = {1: "Forest", 2: "Water", 3: "Urban"}
    # In-place overwrite on same GPKG path using band 2.
    rcrp.pop_class_info_accuracy_pts(
        input_cls_img=img_path,
        acc_pts_vec_file=in_pts,
        acc_pts_vec_lyr="pts",
        out_vec_file=in_pts,
        out_vec_lyr="pts",
        class_names_lut=lut,
        img_cls_col="img_cls",
        ref_cls_col="cls_ref",
        processed_col="Processed",
        cls_img_band=2,
        use_pbar=False,
    )
    out_gdf = gpd.read_file(in_pts, layer="pts")
    assert out_gdf["img_cls"].tolist() == ["Forest", "Water", "Urban", ""]
    assert (out_gdf["cls_ref"] == "").all()
    assert (out_gdf["Processed"] == 0).all()

    # Test CRS reprojection on the fly in memory.
    pts_wgs84 = pts.iloc[:3].to_crs("EPSG:4326")
    out_reproj = rcrp.pop_class_info_accuracy_pts(
        input_cls_img=img_path,
        acc_pts_vec_file=pts_wgs84,
        acc_pts_vec_lyr=None,
        out_vec_file=None,
        out_vec_lyr=None,
        class_names_lut=lut,
        cls_img_band=2,
        reproj_vec=True,
        use_pbar=False,
    )
    assert out_reproj["img_cls"].tolist() == ["Forest", "Water", "Urban"]


def test_calc_acc_stats_ptonly_and_img(tmp_path):
    """Verify exact point-only and area-weighted accuracy metrics plus JSON/CSV."""
    # Create classification image with known class pixel counts:
    # 1 (Forest): 60 px (60%), 2 (Water): 40 px (40%), pixel area = 100 m^2.
    cls_arr = np.ones((10, 10), dtype=np.uint8)
    cls_arr[:, 6:] = 2
    img_path = str(tmp_path / "map.tif")
    _write_raster(img_path, cls_arr, pxl_size=10.0, nodata=0.0)
    lut = {1: "Forest", 2: "Water"}

    # Construct 10 reference samples:
    # Forest predicted (row 0): 4 Forest ref, 1 Water ref -> User = 80%
    # Water predicted (row 1): 1 Forest ref, 4 Water ref -> User = 80%
    ref_list = ["Forest"] * 4 + ["Water"] * 1 + ["Forest"] * 1 + ["Water"] * 4
    cls_list = ["Forest"] * 5 + ["Water"] * 5
    pts_gdf = gpd.GeoDataFrame(
        {"ref": ref_list, "cls": cls_list},
        geometry=[Point(500005.0, 5599995.0)] * 10,
        crs="EPSG:32630",
    )

    pt_json = str(tmp_path / "ptonly.json")
    pt_csv = str(tmp_path / "ptonly.csv")
    pt_stats = rcam.calc_acc_stats_ptonly(
        vec_file=pts_gdf,
        vec_lyr=None,
        ref_col="ref",
        cls_col="cls",
        out_json_file=pt_json,
        out_csv_file=pt_csv,
    )
    assert pt_stats["accuracy"] == pytest.approx(0.8)
    assert pt_stats["cohen_kappa"] == pytest.approx(0.6)
    assert pt_stats["confusion_matrix"] == [[4, 1], [1, 4]]
    assert pt_stats["user_accuracy"] == pytest.approx([80.0, 80.0])
    assert pt_stats["producer_accuracy"] == pytest.approx([80.0, 80.0])
    assert pt_stats["commission"] == pytest.approx([20.0, 20.0])
    assert pt_stats["omission"] == pytest.approx([20.0, 20.0])
    assert json.loads(open(pt_json, encoding="utf-8").read())["accuracy"] == pytest.approx(0.8)
    with open(pt_csv, encoding="utf-8") as fp:
        csv_rows = list(csv.reader(fp))
    assert csv_rows[0][0] == "overall accuracy"
    assert float(csv_rows[0][1]) == pytest.approx(0.8)

    # Image area-weighted metrics
    img_json = str(tmp_path / "img.json")
    img_csv = str(tmp_path / "img.csv")
    img_stats = rcam.calc_acc_stats_img(
        vec_file=pts_gdf,
        vec_lyr=None,
        ref_col="ref",
        cls_col="cls",
        cls_img=img_path,
        class_names_lut=lut,
        out_json_file=img_json,
        out_csv_file=img_csv,
        use_pbar=False,
    )
    assert img_stats["class_area"] == {"Forest": 6000.0, "Water": 4000.0}
    # norm_cm rows: [0.6*0.8, 0.6*0.2] = [0.48, 0.12]; [0.4*0.2, 0.4*0.8] = [0.08, 0.32]
    assert img_stats["norm_confusion_matrix"][0] == pytest.approx([0.48, 0.12])
    assert img_stats["norm_confusion_matrix"][1] == pytest.approx([0.08, 0.32])
    assert img_stats["est_prop_cls_area"] == pytest.approx([0.56, 0.44])
    qm = img_stats["quantity_metrics"]
    assert qm["Proportion Correct (C)"] == pytest.approx(0.80)
    assert qm["Quantity Disagreement (Q)"] == pytest.approx(0.04)
    assert qm["Allocation Disagreement (A)"] == pytest.approx(0.16)
    assert qm["Total Disagreement (D)"] == pytest.approx(0.20)
    assert json.loads(open(img_json, encoding="utf-8").read())["class_area"]["Forest"] == 6000.0
    with open(img_csv, encoding="utf-8") as fp:
        img_csv_rows = list(csv.reader(fp))
    assert any(row and row[0] == "Normalised Confusion Matrix" for row in img_csv_rows)

    # Error when LUT misses a class present in vector points
    with pytest.raises(rsgispy.RSGISPyException):
        rcam.calc_acc_stats_img(
            pts_gdf, None, "ref", "cls", img_path, {1: "Forest"}, use_pbar=False
        )


def test_calc_acc_stats_bootstrap_conf(tmp_path):
    """Verify point-only and image-area bootstrap confidence intervals."""
    cls_arr = np.ones((10, 10), dtype=np.uint8)
    cls_arr[:, 5:] = 2
    cls_arr[8:, 8:] = 3  # 4 pixels of rare class 3
    img_path = str(tmp_path / "boot_map.tif")
    _write_raster(img_path, cls_arr, pxl_size=10.0, nodata=0.0)
    lut = {1: "Forest", 2: "Water", 3: "Urban"}

    rng = np.random.default_rng(1234)
    n_pts = 80
    ref_vals = rng.choice(["Forest", "Water", "Urban"], size=n_pts, p=[0.48, 0.47, 0.05])
    cls_vals = ref_vals.copy()
    flip_mask = rng.random(n_pts) < 0.15
    cls_vals[flip_mask] = np.where(
        ref_vals[flip_mask] == "Forest", "Water", "Forest"
    )

    pts_gdf = gpd.GeoDataFrame(
        {"ref": ref_vals, "cls": cls_vals},
        geometry=[Point(500005.0, 5599995.0)] * n_pts,
        crs="EPSG:32630",
    )

    pt_boot_json = str(tmp_path / "pt_boot.json")
    pt_ci = rcam.calc_acc_stats_ptonly_bootstrap_conf(
        vec_file=pts_gdf,
        vec_lyr=None,
        ref_col="ref",
        cls_col="cls",
        out_json_file=pt_boot_json,
        sample_frac=0.25,
        bootstrap_n=50,
        rnd_seed=11,
        use_pbar=False,
    )
    for key in (
        "accuracy",
        "cohen_kappa",
        "bal_accuracy_score",
        "matthews_corrcoef",
        "hamming_loss",
    ):
        ci_block = pt_ci[key]
        assert ci_block["99th"][0] <= ci_block["95th"][0] <= ci_block["90th"][0]
        assert ci_block["90th"][0] <= ci_block["median"] <= ci_block["90th"][1]
        assert ci_block["90th"][1] <= ci_block["95th"][1] <= ci_block["99th"][1]
        assert np.isfinite(ci_block["mean"])
    assert "Urban" in pt_ci
    assert "f1-score" in pt_ci["Forest"]
    assert "user_accuracy" in pt_ci["Forest"]

    img_boot_json = str(tmp_path / "img_boot.json")
    img_ci = rcam.calc_acc_stats_img_bootstrap_conf(
        vec_file=pts_gdf,
        vec_lyr=None,
        ref_col="ref",
        cls_col="cls",
        cls_img=img_path,
        class_names_lut=lut,
        out_json_file=img_boot_json,
        sample_n_smps=30,
        bootstrap_n=50,
        rnd_seed=11,
        use_pbar=False,
    )
    assert "weighted area avg" in img_ci
    assert "quantity_metrics" in img_ci
    assert "Proportion Correct (C)" in img_ci["quantity_metrics"]
    assert "est_prop_cls_area" in img_ci["Forest"]
    assert img_ci["class_area"]["Urban"] == pytest.approx(400.0)
    with open(img_boot_json, encoding="utf-8") as fp:
        assert json.load(fp)["class_area"]["Forest"] == 5000.0

    # Reproducibility with same seed
    img_ci_2 = rcam.calc_acc_stats_img_bootstrap_conf(
        vec_file=pts_gdf,
        vec_lyr=None,
        ref_col="ref",
        cls_col="cls",
        cls_img=img_path,
        class_names_lut=lut,
        sample_n_smps=30,
        bootstrap_n=50,
        rnd_seed=11,
        use_pbar=False,
    )
    assert img_ci["accuracy"]["mean"] == pytest.approx(img_ci_2["accuracy"]["mean"])

    with pytest.raises(rsgispy.RSGISPyException):
        rcam.calc_acc_stats_ptonly_bootstrap_conf(
            pts_gdf, None, "ref", "cls", bootstrap_n=0, use_pbar=False
        )
    with pytest.raises(rsgispy.RSGISPyException):
        rcam.calc_acc_stats_ptonly_bootstrap_conf(
            pts_gdf, None, "ref", "cls", sample_frac=1.5, use_pbar=False
        )


def test_calc_acc_stats_img_non_ascii_class_names(tmp_path):
    """Class names with non-ASCII characters are matched between the LUT and points."""
    cls_arr = np.ones((10, 10), dtype=np.uint8)
    cls_arr[:, 6:] = 2
    img_path = str(tmp_path / "map.tif")
    _write_raster(img_path, cls_arr, pxl_size=10.0, nodata=0.0)
    lut = {1: "Wald-Fläche", 2: "Wasser"}
    ref_list = ["Wald-Fläche"] * 4 + ["Wasser"] + ["Wald-Fläche"] + ["Wasser"] * 4
    cls_list = ["Wald-Fläche"] * 5 + ["Wasser"] * 5
    pts_gdf = gpd.GeoDataFrame(
        {"ref": ref_list, "cls": cls_list},
        geometry=[Point(500005.0, 5599995.0)] * 10,
        crs="EPSG:32630",
    )
    stats = rcam.calc_acc_stats_img(
        pts_gdf, None, "ref", "cls", img_path, lut,
        out_csv_file=str(tmp_path / "acc.csv"), use_pbar=False,
    )
    assert stats["class_area"] == {"Wald-Fläche": 6000.0, "Wasser": 4000.0}


def test_area_accuracy_commission_omission_are_percentage_rates(tmp_path):
    """Area-weighted commission/omission are % error rates, like the point-only metrics."""
    cls_arr = np.ones((10, 10), dtype=np.uint8)
    cls_arr[:, 6:] = 2
    img_path = str(tmp_path / "map.tif")
    _write_raster(img_path, cls_arr, pxl_size=10.0, nodata=0.0)
    ref_list = ["Forest"] * 4 + ["Water"] + ["Forest"] + ["Water"] * 4
    cls_list = ["Forest"] * 5 + ["Water"] * 5
    pts_gdf = gpd.GeoDataFrame(
        {"ref": ref_list, "cls": cls_list},
        geometry=[Point(500005.0, 5599995.0)] * 10,
        crs="EPSG:32630",
    )
    stats = rcam.calc_acc_stats_img(
        pts_gdf, None, "ref", "cls", img_path, {1: "Forest", 2: "Water"},
        use_pbar=False,
    )
    # norm_cm = [[0.48, 0.12], [0.08, 0.32]] (rows = map classes).
    # Commission = 100 * row off-diagonal / row total; omission = by column.
    assert stats["commission"] == pytest.approx([20.0, 20.0])
    assert stats["omission"] == pytest.approx([100 * 0.08 / 0.56, 100 * 0.12 / 0.44])
    # The previous proportions of total area are kept under explicit keys.
    assert stats["commission_area_prop"] == pytest.approx([0.12, 0.08])
    assert stats["omission_area_prop"] == pytest.approx([0.08, 0.12])
