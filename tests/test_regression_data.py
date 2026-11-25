"""
Tests for rsgispy.regression.regdata: ImageBandInfo, RegressionConfig, JSON
persistence, point/polygon sample extraction, target bounds filtering, and
random/quantile-stratified sample splitting.
"""

import importlib.util
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point, box

import rsgispy
import rsgispy.regression as rreg
import rsgispy.regression.regdata as rrd

_HAS_PARQUET = (
    importlib.util.find_spec("pyarrow") is not None
    or importlib.util.find_spec("fastparquet") is not None
)


def _write_raster(
    path: Path,
    arr: np.ndarray,
    origin_x: float = 500000.0,
    origin_y: float = 5600000.0,
    pxl_size: float = 10.0,
    crs: str = "EPSG:32630",
    nodata: float = -999.0,
) -> str:
    """Write a 2D or 3D GeoTIFF raster for testing."""
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
    return str(path)


def _create_landsat_fixture(tmp_path: Path):
    """Build synthetic Landsat-like 3-band image and canopy/height vectors."""
    h, w = 16, 16
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    b_red = 0.05 + 0.01 * xx
    b_nir = 0.15 + 0.03 * yy
    b_swir = 0.25 - 0.01 * yy
    b_red[2, 2] = -999.0

    lsat_img = _write_raster(
        tmp_path / "landsat.tif", np.stack([b_red, b_nir, b_swir]), nodata=-999.0
    )

    pts = []
    covers = []
    heights = []
    for row in range(1, 15, 2):
        for col in range(1, 15, 2):
            px_x = 500005.0 + col * 10.0
            px_y = 5599995.0 - row * 10.0
            pts.append(Point(px_x, px_y))
            covers.append(min(100.0, 5.0 + 6.0 * row))
            heights.append(2.0 + 1.5 * row + 0.5 * col)

    # Add an out-of-range point and a non-finite target row to test filtering.
    pts.append(Point(500135.0, 5599875.0))
    covers.append(125.0)
    heights.append(25.0)

    pts_vec = str(tmp_path / "canopy_pts.gpkg")
    gpd.GeoDataFrame(
        {"canopy_cover": covers, "canopy_height": heights},
        geometry=pts,
        crs="EPSG:32630",
    ).to_file(pts_vec, layer="canopy", driver="GPKG")

    poly_vec = str(tmp_path / "canopy_polys.gpkg")
    poly_gdf = gpd.GeoDataFrame(
        {"canopy_cover": [30.0, 80.0], "canopy_height": [10.0, 28.0]},
        geometry=[
            box(500010.0, 5599950.0, 500060.0, 5599990.0),
            box(500080.0, 5599860.0, 500140.0, 5599910.0),
        ],
        crs="EPSG:32630",
    )
    poly_gdf.to_file(poly_vec, layer="polys", driver="GPKG")

    band_info = [
        rrd.ImageBandInfo(
            file_name=lsat_img,
            name="lsat",
            bands=[1, 2, 3],
            band_names=["red", "nir", "swir1"],
        )
    ]
    return band_info, pts_vec, poly_vec


def test_reg_package_exports_and_config_roundtrip(tmp_path: Path):
    """Verify package submodules, RegressionConfig properties, and JSON round-trip."""
    assert rreg.__doc__ is not None
    assert "regdata" in rreg.__doc__
    assert "regsklearn" in rreg.__doc__
    assert "regfeatures" in rreg.__doc__
    assert "regmetrics" in rreg.__doc__

    band_info, pts_vec, _ = _create_landsat_fixture(tmp_path)
    uni_cfg = rrd.RegressionConfig(
        img_band_info=band_info,
        target_cols="canopy_cover",
        vec_file=pts_vec,
        vec_lyr="canopy",
    )
    assert uni_cfg.n_targets == 1
    assert not uni_cfg.is_multivariate
    assert uni_cfg.get_band_names() == ["red", "nir", "swir1"]

    mv_cfg_path = tmp_path / "mv_reg_cfg.json"
    written = rrd.write_reg_config(
        mv_cfg_path,
        img_band_info=band_info,
        target_cols=["canopy_cover", "canopy_height"],
        vec_file=pts_vec,
        vec_lyr="canopy",
    )
    assert written.n_targets == 2
    assert written.is_multivariate

    loaded = rrd.read_reg_config(mv_cfg_path)
    assert loaded.target_cols == ["canopy_cover", "canopy_height"]
    assert loaded.vec_file == pts_vec
    assert loaded.vec_lyr == "canopy"
    assert loaded.get_band_names() == ["red", "nir", "swir1"]

    with pytest.raises(rsgispy.RSGISPyException):
        rrd.normalise_target_cols([])
    with pytest.raises(rsgispy.RSGISPyException):
        rrd.normalise_target_cols(["a", "a"])


def test_extract_reg_samples_points_polygons_and_bounds(tmp_path: Path):
    """Test point & polygon extraction, bounds filtering, and table I/O."""
    band_info, pts_vec, poly_vec = _create_landsat_fixture(tmp_path)

    # Point extraction with univariate target bounds [0, 100].
    df_pts = rrd.extract_reg_training_samples(
        img_band_info=band_info,
        target_cols="canopy_cover",
        vec_file=pts_vec,
        vec_lyr="canopy",
        target_min=0.0,
        target_max=100.0,
    )
    assert set(df_pts.columns) == {"red", "nir", "swir1", "canopy_cover"}
    assert df_pts["canopy_cover"].max() <= 100.0
    assert df_pts["canopy_cover"].min() >= 0.0
    assert len(df_pts) == 49  # 7x7 grid; 125.0 point excluded

    # Polygon extraction with multivariate target_cols.
    mv_cfg = rrd.RegressionConfig(
        img_band_info=band_info,
        target_cols=["canopy_cover", "canopy_height"],
        vec_file=poly_vec,
        vec_lyr="polys",
    )
    ext = ".parquet" if _HAS_PARQUET else ".csv"
    out_tbl = tmp_path / f"poly_samples{ext}"
    df_poly = rrd.extract_reg_training_samples(
        img_band_info=mv_cfg,
        out_file=out_tbl,
    )
    assert out_tbl.exists()
    assert set(df_poly.columns) == {
        "red",
        "nir",
        "swir1",
        "canopy_cover",
        "canopy_height",
    }
    assert np.all(np.isfinite(df_poly.to_numpy(dtype=np.float64)))
    reloaded = rrd.read_reg_samples(out_tbl)
    assert len(reloaded) == len(df_poly)


def test_split_and_extract_and_split_reg_samples(tmp_path: Path):
    """Test random & quantile-stratified splits plus combined extract-and-split."""
    band_info, pts_vec, _ = _create_landsat_fixture(tmp_path)
    cfg = rrd.RegressionConfig(
        img_band_info=band_info,
        target_cols=["canopy_cover", "canopy_height"],
        vec_file=pts_vec,
        vec_lyr="canopy",
    )

    ext = ".parquet" if _HAS_PARQUET else ".csv"
    train_f = tmp_path / f"train{ext}"
    test_f = tmp_path / f"test{ext}"
    valid_f = tmp_path / f"valid{ext}"

    train_df, test_df, valid_df = rrd.extract_and_split_reg_samples(
        img_band_info=cfg,
        test_prop=0.2,
        valid_prop=0.2,
        stratify_col="canopy_cover",
        n_bins=4,
        target_min=0.0,
        target_max=100.0,
        out_train_file=train_f,
        out_test_file=test_f,
        out_valid_file=valid_f,
        rnd_seed=42,
    )
    assert train_f.exists() and test_f.exists() and valid_f.exists()
    assert test_df is not None and valid_df is not None
    assert len(train_df) + len(test_df) + len(valid_df) == 49
    assert rrd.read_reg_samples(train_f).shape == train_df.shape

    with pytest.raises(rsgispy.RSGISPyException):
        rrd.split_reg_samples(train_df, test_prop=0.6, valid_prop=0.5)
    with pytest.raises(rsgispy.RSGISPyException):
        rrd.split_reg_samples(train_df, stratify_col="missing_col")


def _reg_polys_fixture(tmp_path: Path):
    """Ten 2 x 2 pixel plots with a canopy cover value each."""
    band_info, _, _ = _create_landsat_fixture(tmp_path)
    polys, covers = [], []
    for idx in range(10):
        row0, col0 = 4 * (idx // 4) + 1, 4 * (idx % 4) + 1
        x0 = 500000.0 + col0 * 10.0
        y1 = 5600000.0 - row0 * 10.0
        polys.append(box(x0, y1 - 20.0, x0 + 20.0, y1))
        covers.append(10.0 * idx)
    poly_vec = str(tmp_path / "plots.gpkg")
    gpd.GeoDataFrame(
        {"canopy_cover": covers}, geometry=polys, crs="EPSG:32630"
    ).to_file(poly_vec, layer="plots", driver="GPKG")
    return rrd.RegressionConfig(
        img_band_info=band_info, target_cols="canopy_cover",
        vec_file=poly_vec, vec_lyr="plots",
    )


def _groups_disjoint(parts, col):
    sets = [set(part[col]) for part in parts]
    return all(
        sets[i].isdisjoint(sets[j])
        for i in range(len(sets)) for j in range(i + 1, len(sets))
    )


def test_reg_poly_id_col_and_grouped_split(tmp_path: Path):
    """Plot IDs are recorded and persisted, and the grouped split keeps plots whole."""
    cfg = _reg_polys_fixture(tmp_path)
    df = rrd.extract_reg_training_samples(
        cfg, include_coords=True, poly_id_col="plot_id", use_pbar=False
    )
    # (2, 2) is no data in the red band: plot 0 keeps 3 of its 4 pixels.
    assert len(df) == 10 * 4 - 1
    assert list(df.columns) == ["canopy_cover", "plot_id", "x", "y",
                                "red", "nir", "swir1"]
    assert df.groupby("plot_id")["canopy_cover"].nunique().max() == 1
    assert sorted(df["plot_id"].unique()) == sorted(f"poly{i}" for i in range(10))

    formats = ["csv", "pkl"] + (["parquet"] if _HAS_PARQUET else [])
    for ext in formats:
        out_path = tmp_path / f"reg_samples.{ext}"
        rrd.write_reg_samples(df, out_path)
        assert rrd.read_reg_samples(out_path)["plot_id"].tolist() == df[
            "plot_id"
        ].tolist()

    for kwargs in ({}, {"stratify_col": "canopy_cover", "n_bins": 2}):
        for seed in range(5):
            parts = rrd.split_reg_samples(
                df, group_col="plot_id", rnd_seed=seed, **kwargs
            )
            assert _groups_disjoint(parts, "plot_id")
            assert sum(len(p) for p in parts) == len(df)
            assert all(len(p) > 0 for p in parts)
    # Exact counts are upper limits made of whole plots.
    parts = rrd.split_reg_samples(
        df, group_col="plot_id", n_train=20, n_test=8, n_valid=8, rnd_seed=2
    )
    assert _groups_disjoint(parts, "plot_id")
    assert len(parts[0]) <= 20 and len(parts[1]) <= 8

    with pytest.raises(rsgispy.RSGISPyException):
        rrd.split_reg_samples(df, group_col="missing")
    with pytest.raises(rsgispy.RSGISPyException):
        rrd.extract_reg_training_samples(cfg, poly_id_col="red", use_pbar=False)
    with pytest.raises(rsgispy.RSGISPyException):
        rrd.extract_reg_training_samples(
            cfg, poly_id_col="canopy_cover", use_pbar=False
        )

    tr_df, te_df, va_df = rrd.extract_and_split_reg_samples(
        cfg, group_col="plot_id", rnd_seed=1, use_pbar=False
    )
    assert _groups_disjoint((tr_df, te_df, va_df), "plot_id")
    with pytest.raises(rsgispy.RSGISPyException):
        rrd.extract_and_split_reg_samples(
            cfg, poly_id_col="a", group_col="b", use_pbar=False
        )


def test_reg_poly_id_col_points(tmp_path: Path):
    """Each reference point is its own group."""
    band_info, pts_vec, _ = _create_landsat_fixture(tmp_path)
    df = rrd.extract_reg_training_samples(
        band_info, target_cols="canopy_cover", vec_file=pts_vec,
        vec_lyr="canopy", poly_id_col="train_grp", use_pbar=False,
    )
    assert df["train_grp"].is_unique
    assert df["train_grp"].iloc[0] == "pt0"
