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
