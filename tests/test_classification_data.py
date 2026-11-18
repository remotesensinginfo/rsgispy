"""
Tests for rsgispy.classification.classdata: data structures, config JSON I/O,
Parquet (Snappy default) sample extraction/splitting, and class ID recoding.
"""

import importlib.util
import inspect
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point, box

import rsgispy
import rsgispy.classification as rcls
import rsgispy.classification.classdata as rcd
import rsgispy.imageutils.colourtable as rct
import rsgispy.imageutils.masking as rmask


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


def _write_points(
    path: Path,
    layer: str,
    coords: list,
    crs: str = "EPSG:32630",
) -> str:
    """Write a point GeoPackage layer for testing."""
    gdf = gpd.GeoDataFrame(
        {"pt_id": list(range(len(coords)))},
        geometry=[Point(x, y) for x, y in coords],
        crs=crs,
    )
    gdf.to_file(path, layer=layer, driver="GPKG")
    return str(path)


def _write_polygons(
    path: Path,
    layer: str,
    bounds_list: list,
    crs: str = "EPSG:32630",
) -> str:
    """Write a polygon GeoPackage layer for testing."""
    gdf = gpd.GeoDataFrame(
        {"poly_id": list(range(len(bounds_list)))},
        geometry=[box(*b) for b in bounds_list],
        crs=crs,
    )
    gdf.to_file(path, layer=layer, driver="GPKG")
    return str(path)


def test_classdata_module_layout():
    """Verify classdata is a submodule and does not leak onto rsgispy.classification."""
    assert inspect.ismodule(rcd)
    assert not hasattr(rcls, "ClassInfoObj")
    assert not hasattr(rcls, "extract_class_training_samples")
    assert rcd.ImageBandInfo is rmask.ImageBandInfo


def test_data_structures_and_normalisation(tmp_path):
    """Test ImageBandInfo, ClassInfoObj, internal/output IDs, and validation."""
    img_info = rcd.ImageBandInfo(
        tmp_path / "s2.tif",
        name="S2",
        bands=[1, 3],
        band_names=["Blue", "Red"],
    )
    assert img_info.file_name == str(tmp_path / "s2.tif")
    assert rcd.ImageBandInfo.from_dict(img_info.to_dict()) == img_info

    with pytest.raises(rsgispy.RSGISPyException):
        rcd.ImageBandInfo("a.tif", bands=[1, 2], band_names=["OnlyOne"])
    with pytest.raises(rsgispy.RSGISPyException):
        rcd.ImageBandInfo("a.tif", bands=[1], band_names=[" "])

    classes = {
        "Forest": rcd.ClassInfoObj(
            out_id=10, vec_file="forest.gpkg", vec_lyr="forest", green=200
        ),
        "Water": rcd.ClassInfoObj(
            out_id=250, vec_file="water.gpkg", vec_lyr="water", blue=255
        ),
        "Bare": rcd.ClassInfoObj(
            out_id=0, vec_file="bare.gpkg", vec_lyr="bare", red=220, green=180
        ),
    }
    norm = rcd.normalise_class_info(classes)
    assert [norm[k].internal_id for k in ("Forest", "Water", "Bare")] == [0, 1, 2]
    assert [norm[k].out_id for k in ("Forest", "Water", "Bare")] == [10, 250, 0]
    # Originals without explicit internal_id are not mutated in-place.
    assert classes["Forest"].internal_id is None

    # Explicit consecutive internal IDs in custom order are preserved.
    explicit = {
        "Forest": rcd.ClassInfoObj(out_id=10, internal_id=2),
        "Water": rcd.ClassInfoObj(out_id=250, internal_id=0),
        "Bare": rcd.ClassInfoObj(out_id=0, internal_id=1),
    }
    norm_exp = rcd.normalise_class_info(explicit)
    assert [norm_exp[k].internal_id for k in ("Forest", "Water", "Bare")] == [2, 0, 1]

    # Validation errors: negative IDs, duplicate out_id, partial/non-0-based internal_id.
    with pytest.raises(rsgispy.RSGISPyException):
        rcd.ClassInfoObj(out_id=-1)
    with pytest.raises(rsgispy.RSGISPyException):
        rcd.ClassInfoObj(out_id=-5)
    with pytest.raises(rsgispy.RSGISPyException):
        rcd.ClassInfoObj(out_id=1, internal_id=-1)
    with pytest.raises(rsgispy.RSGISPyException):
        rcd.ClassInfoObj.from_dict({"out_id": -2})
    with pytest.raises(rsgispy.RSGISPyException):
        rcd.normalise_class_info(
            {
                "A": rcd.ClassInfoObj(out_id=10),
                "B": rcd.ClassInfoObj(out_id=10),
            }
        )
    with pytest.raises(rsgispy.RSGISPyException):
        rcd.normalise_class_info(
            {
                "A": rcd.ClassInfoObj(out_id=10, internal_id=0),
                "B": rcd.ClassInfoObj(out_id=20),
            }
        )
    with pytest.raises(rsgispy.RSGISPyException):
        rcd.normalise_class_info(
            {
                "A": rcd.ClassInfoObj(out_id=10, internal_id=1),
                "B": rcd.ClassInfoObj(out_id=20, internal_id=2),
            }
        )
    with pytest.raises(rsgispy.RSGISPyException):
        rcd.ClassInfoObj(out_id=True)  # type: ignore[arg-type]
    with pytest.raises(rsgispy.RSGISPyException):
        rcd.ClassInfoObj(out_id=1, red=300)


def test_config_json_roundtrip_and_luts(tmp_path):
    """Verify ClassificationConfig, LUT helpers, and JSON save/load round-trip."""
    arr = np.arange(3 * 6 * 6, dtype=np.float32).reshape(3, 6, 6)
    img1 = _write_raster(tmp_path / "s2.tif", arr)
    img2 = _write_raster(tmp_path / "dem.tif", arr[:1])

    img_band_info = [
        rcd.ImageBandInfo(img1, name="S2", bands=[1, 3], band_names=["B2", "B4"]),
        rcd.ImageBandInfo(img2, name="DEM"),
    ]
    class_info = {
        "Forest": rcd.ClassInfoObj(
            out_id=10, vec_file=tmp_path / "forest.gpkg", vec_lyr="forest",
            red=0, green=180, blue=0,
        ),
        "Water": rcd.ClassInfoObj(
            out_id=42, vec_file=tmp_path / "water.gpkg", vec_lyr="water",
            red=0, green=0, blue=255,
        ),
    }

    cfg_path = tmp_path / "cfg" / "classification.json"
    written_cfg = rcd.write_cls_config(cfg_path, img_band_info, class_info)
    assert cfg_path.exists()

    loaded_cfg = rcd.read_cls_config(cfg_path)
    assert loaded_cfg == written_cfg
    assert loaded_cfg.get_band_names() == ["B2", "B4", "DEM_b1"]
    assert loaded_cfg.get_internal_to_out_lut() == {0: 10, 1: 42}
    assert loaded_cfg.get_out_to_internal_lut() == {10: 0, 42: 1}
    assert loaded_cfg.get_class_names_lut() == {10: "Forest", 42: "Water"}
    assert loaded_cfg.get_class_names_lut(use_internal_id=True) == {
        0: "Forest",
        1: "Water",
    }
    assert loaded_cfg.get_colour_table() == {
        10: [0, 180, 0, 255],
        42: [0, 0, 255, 255],
    }


def test_extract_point_and_polygon_samples_snappy_parquet(tmp_path):
    """
    Test sample extraction using ext_point_band_values_file (points) and
    export_poly_pxls_to_df (polygons), mixed vector types, multi-image rasters,
    and train/test/valid splitting into Snappy-compressed Parquet files.
    """
    pq = pytest.importorskip("pyarrow.parquet")

    # 10x10 rasters at 10m pixels: X=[500000..500100], Y=[5599900..5600000]
    b1 = np.arange(100, dtype=np.float32).reshape(10, 10)
    b2 = (np.arange(100, dtype=np.float32).reshape(10, 10)) * 2.0 + 10.0
    # Put nodata at row=0, col=0 to verify nodata filtering.
    b1[0, 0] = -999.0
    s2_img = _write_raster(tmp_path / "s2.tif", np.stack([b1, b2]), nodata=-999.0)

    dem = (np.arange(100, dtype=np.float32).reshape(10, 10)) + 1000.0
    dem_img = _write_raster(tmp_path / "dem.tif", dem, nodata=-999.0)

    # Coarse 20m raster (5x5) to test multi-resolution fallback sampling.
    coarse = (np.arange(25, dtype=np.float32).reshape(5, 5)) + 500.0
    coarse_img = _write_raster(
        tmp_path / "coarse.tif", coarse, pxl_size=20.0, nodata=-999.0
    )

    # Class 1 (out_id=10): 10 points including nodata pixel at (500005, 5599995).
    pt_coords = [(500005.0 + c * 10.0, 5599995.0) for c in range(10)]
    pt_vec = _write_points(tmp_path / "forest_pts.gpkg", "forest", pt_coords)
    orig_pt_cols = list(gpd.read_file(pt_vec, layer="forest").columns)

    # Class 2 (out_id=250): polygons covering rows 5..7, cols 0..4 (15 pixels).
    poly_vec = _write_polygons(
        tmp_path / "water_polys.gpkg",
        "water",
        [(500000.0, 5599920.0, 500050.0, 5599950.0)],
    )

    img_band_info = [
        rcd.ImageBandInfo(s2_img, name="S2", bands=[1, 2], band_names=["B1", "B2"]),
        rcd.ImageBandInfo(dem_img, name="DEM"),
        rcd.ImageBandInfo(coarse_img, name="Coarse"),
    ]
    class_info = {
        "Forest": rcd.ClassInfoObj(
            out_id=10, vec_file=pt_vec, vec_lyr="forest", green=200
        ),
        "Water": rcd.ClassInfoObj(
            out_id=250, vec_file=poly_vec, vec_lyr="water", blue=255
        ),
    }

    cfg_file = tmp_path / "cls_cfg.json"
    cfg = rcd.write_cls_config(cfg_file, img_band_info, class_info)

    all_pq = tmp_path / "samples_all.parquet"
    train_pq = tmp_path / "train.parquet"
    test_pq = tmp_path / "test.parquet"
    valid_pq = tmp_path / "valid.parquet"

    train_df, test_df, valid_df = rcd.extract_and_split_class_samples(
        cfg,
        train_prop=0.6,
        test_prop=0.2,
        valid_prop=0.2,
        out_train_file=train_pq,
        out_test_file=test_pq,
        out_valid_file=valid_pq,
        out_all_file=all_pq,
        include_coords=True,
        rnd_seed=123,
        use_pbar=False,
    )

    # Source point vector file must NOT be modified on disk.
    assert list(gpd.read_file(pt_vec, layer="forest").columns) == orig_pt_cols

    # Verify Parquet files exist and use SNAPPY compression by default.
    for pq_path in (all_pq, train_pq, test_pq, valid_pq):
        assert pq_path.exists()
        pf = pq.ParquetFile(pq_path)
        assert pf.metadata.row_group(0).column(0).compression == "SNAPPY"

    all_df = rcd.read_cls_samples(all_pq)
    # 9 valid forest points (1 nodata filtered) + 15 water polygon pixels = 24
    assert len(all_df) == 24
    assert list(all_df.columns) == [
        "class_name",
        "out_cls_id",
        "internal_cls_id",
        "x",
        "y",
        "B1",
        "B2",
        "DEM_b1",
        "Coarse_b1",
    ]
    assert set(all_df["out_cls_id"].unique()) == {10, 250}
    assert set(all_df["internal_cls_id"].unique()) == {0, 1}

    # Verify extracted raster values for a Forest point
    # (row=0, col=1 -> b1=1, b2=12, dem=1001, coarse=500).
    row_c1 = all_df[(all_df["out_cls_id"] == 10) & (all_df["B1"] == 1.0)].iloc[0]
    assert row_c1["B2"] == pytest.approx(12.0)
    assert row_c1["DEM_b1"] == pytest.approx(1001.0)
    assert row_c1["Coarse_b1"] == pytest.approx(500.0)

    # Verify split sizes, class stratification, and Parquet round-trip.
    assert len(train_df) + len(test_df) + len(valid_df) == 24
    assert set(train_df["internal_cls_id"].unique()) == {0, 1}
    assert set(test_df["internal_cls_id"].unique()) == {0, 1}
    assert set(valid_df["internal_cls_id"].unique()) == {0, 1}
    assert rcd.read_cls_samples(train_pq).equals(train_df)
    assert rcd.read_cls_samples(test_pq).equals(test_df)
    assert rcd.read_cls_samples(valid_pq).equals(valid_df)

    # Reproducibility check with identical seed.
    tr2, te2, va2 = rcd.split_class_samples(all_df, rnd_seed=123)
    assert tr2.equals(train_df)
    assert te2.equals(test_df)
    assert va2.equals(valid_df)


def test_sample_formats_other_than_parquet(tmp_path):
    """Test write_cls_samples/read_cls_samples and extraction with non-Parquet formats."""
    df = pd.DataFrame(
        {
            "class_name": ["Forest", "Water", "Forest", "Water"],
            "out_cls_id": [10, 250, 10, 250],
            "internal_cls_id": [0, 1, 0, 1],
            "b1": [1.5, 2.5, 3.5, 4.5],
        }
    )

    formats = ["csv", "tsv", "json", "pkl"]
    if importlib.util.find_spec("pyarrow") is not None:
        formats.append("feather")
    for ext in formats:
        out_path = tmp_path / f"samples.{ext}"
        rcd.write_cls_samples(df, out_path)
        assert out_path.exists()
        loaded = rcd.read_cls_samples(out_path)
        pd.testing.assert_frame_equal(loaded, df)

    # Explicit format override and gzip CSV compression.
    custom_path = tmp_path / "samples.dat"
    rcd.write_cls_samples(df, custom_path, out_format="csv")
    pd.testing.assert_frame_equal(
        rcd.read_cls_samples(custom_path, in_format="csv"), df
    )

    csv_gz = tmp_path / "samples.csv.gz"
    rcd.write_cls_samples(df, csv_gz, out_compress="gzip")
    pd.testing.assert_frame_equal(rcd.read_cls_samples(csv_gz), df)

    # Full extraction + splitting directly to CSV.
    arr = np.arange(16, dtype=np.float32).reshape(1, 4, 4)
    img_path = _write_raster(tmp_path / "img.tif", arr)
    pt_a = _write_points(
        tmp_path / "a.gpkg",
        "a",
        [(500005.0 + i * 10.0, 5599995.0) for i in range(4)],
    )
    pt_b = _write_points(
        tmp_path / "b.gpkg",
        "b",
        [(500005.0 + i * 10.0, 5599985.0) for i in range(4)],
    )
    cfg = rcd.ClassificationConfig(
        img_band_info=[rcd.ImageBandInfo(img_path, name="S2")],
        class_info={
            "A": rcd.ClassInfoObj(out_id=10, vec_file=pt_a, vec_lyr="a"),
            "B": rcd.ClassInfoObj(out_id=20, vec_file=pt_b, vec_lyr="b"),
        },
    )
    tr_csv = tmp_path / "train.csv"
    te_csv = tmp_path / "test.csv"
    va_csv = tmp_path / "valid.csv"
    tr_df, te_df, va_df = rcd.extract_and_split_class_samples(
        cfg,
        out_train_file=tr_csv,
        out_test_file=te_csv,
        out_valid_file=va_csv,
        rnd_seed=9,
        use_pbar=False,
    )
    assert rcd.read_cls_samples(tr_csv).equals(tr_df)
    assert rcd.read_cls_samples(te_csv).equals(te_df)
    assert rcd.read_cls_samples(va_csv).equals(va_df)


def test_split_counts_and_caps():
    """Test max_cls_smpls cap and explicit n_train/n_test/n_valid counts."""
    df = pd.DataFrame(
        {
            "class_name": ["A"] * 20 + ["B"] * 12,
            "out_cls_id": [10] * 20 + [20] * 12,
            "internal_cls_id": [0] * 20 + [1] * 12,
            "feat": np.arange(32, dtype=np.float64),
        }
    )
    tr, te, va = rcd.split_class_samples(
        df,
        max_cls_smpls=10,
        n_train=6,
        n_test=2,
        n_valid=2,
        rnd_seed=7,
    )
    assert (tr["internal_cls_id"] == 0).sum() == 6
    assert (tr["internal_cls_id"] == 1).sum() == 6
    assert (te["internal_cls_id"] == 0).sum() == 2
    assert (va["internal_cls_id"] == 1).sum() == 2
    # Zero overlap across splits.
    assert set(tr["feat"]).isdisjoint(set(te["feat"]))
    assert set(tr["feat"]).isdisjoint(set(va["feat"]))
    assert set(te["feat"]).isdisjoint(set(va["feat"]))


def test_recode_internal_to_output_ids_and_raster(tmp_path):
    """Test array/Series and classified raster recoding from 0..N-1 to out_ids."""
    class_info = {
        "Forest": rcd.ClassInfoObj(out_id=10, red=0, green=150, blue=0),
        "Water": rcd.ClassInfoObj(out_id=25, red=0, green=0, blue=255),
        "Urban": rcd.ClassInfoObj(out_id=200, red=200, green=50, blue=50),
    }

    preds = np.array([0, 1, 2, 255, 0], dtype=np.uint8)
    recoded = rcd.recode_internal_to_out_ids(preds, class_info, nodata_val=255)
    np.testing.assert_array_equal(
        recoded, np.array([10, 25, 200, 255, 10], dtype=np.uint8)
    )

    series_rec = rcd.recode_internal_to_out_ids(
        pd.Series([2, 1, 0], name="pred"), class_info
    )
    assert isinstance(series_rec, pd.Series)
    assert series_rec.tolist() == [200, 25, 10]

    # Raster recoding with uint8 output, thematic tag, and colour table.
    cls_arr = np.array(
        [[0, 1, 255], [2, 0, 1]],
        dtype=np.uint8,
    )
    in_cls_img = _write_raster(tmp_path / "cls_internal.tif", cls_arr, nodata=255)
    out_cls_img = str(tmp_path / "cls_out.tif")
    rcd.recode_cls_img_to_out_ids(
        input_img=in_cls_img,
        output_img=out_cls_img,
        class_info=class_info,
        no_data_val=0,
        use_pbar=False,
    )
    with rasterio.open(out_cls_img, "r") as src:
        assert src.dtypes[0] == "uint8"
        assert src.nodata == 0
        np.testing.assert_array_equal(
            src.read(1),
            np.array([[10, 25, 0], [200, 10, 25]], dtype=np.uint8),
        )
    assert rct.is_img_thematic(out_cls_img)
    clr_tab = rct.get_colour_tab_info(out_cls_img)
    assert clr_tab[10]["green"] == 150
    assert clr_tab[25]["blue"] == 255
    assert clr_tab[200]["red"] == 200

    # Larger unsigned IDs (>255) auto-select uint16.
    uint16_classes = {
        "ClassZero": rcd.ClassInfoObj(out_id=0, red=10, green=20, blue=30),
        "ClassHigh": rcd.ClassInfoObj(out_id=500, red=40, green=50, blue=60),
    }
    in_u16 = _write_raster(
        tmp_path / "cls_u16_in.tif",
        np.array([[0, 1, 255]], dtype=np.uint8),
        nodata=255,
    )
    out_u16 = str(tmp_path / "cls_u16_out.tif")
    rcd.recode_cls_img_to_out_ids(
        input_img=in_u16,
        output_img=out_u16,
        class_info=uint16_classes,
        no_data_val=65535,
        use_pbar=False,
    )
    with rasterio.open(out_u16, "r") as src:
        assert src.dtypes[0] == "uint16"
        assert src.nodata == 65535
        np.testing.assert_array_equal(
            src.read(1),
            np.array([[0, 500, 65535]], dtype=np.uint16),
        )

    # Larger unsigned IDs (>65535) auto-select uint32.
    uint32_classes = {
        "Low": rcd.ClassInfoObj(out_id=100),
        "VeryHigh": rcd.ClassInfoObj(out_id=100000),
    }
    in_u32 = _write_raster(
        tmp_path / "cls_u32_in.tif",
        np.array([[0, 1, 255]], dtype=np.uint8),
        nodata=255,
    )
    out_u32 = str(tmp_path / "cls_u32_out.tif")
    rcd.recode_cls_img_to_out_ids(
        input_img=in_u32,
        output_img=out_u32,
        class_info=uint32_classes,
        no_data_val=0,
        use_pbar=False,
    )
    with rasterio.open(out_u32, "r") as src:
        assert src.dtypes[0] == "uint32"
        assert src.nodata == 0
        np.testing.assert_array_equal(
            src.read(1),
            np.array([[100, 100000, 0]], dtype=np.uint32),
        )


def test_class_info_hex_colour(tmp_path):
    """Test specifying ClassInfoObj colours via hex colour string or RGB."""
    c_hex = rcd.ClassInfoObj(out_id=1, colour="#ff8800")
    assert (c_hex.red, c_hex.green, c_hex.blue) == (255, 136, 0)
    assert c_hex.colour == "#ff8800"

    c_no_hash = rcd.ClassInfoObj(out_id=2, colour="1a2B3c")
    assert (c_no_hash.red, c_no_hash.green, c_no_hash.blue) == (0x1A, 0x2B, 0x3C)
    assert c_no_hash.colour == "#1a2b3c"

    c_tuple = rcd.ClassInfoObj(out_id=3, colour=(10, 20, 30))
    assert (c_tuple.red, c_tuple.green, c_tuple.blue) == (10, 20, 30)
    assert c_tuple.colour == "#0a141e"

    c_rgb = rcd.ClassInfoObj(out_id=4, red=0, green=180, blue=0)
    assert c_rgb.colour == "#00b400"
    c_rgb.colour = "#112233"
    assert (c_rgb.red, c_rgb.green, c_rgb.blue) == (0x11, 0x22, 0x33)

    # Roundtrip via dict and ClassificationConfig JSON
    d_only_colour = {"out_id": 5, "colour": "#228b22"}
    c_from_dict = rcd.ClassInfoObj.from_dict(d_only_colour)
    assert (c_from_dict.red, c_from_dict.green, c_from_dict.blue) == (34, 139, 34)
    assert rcd.ClassInfoObj.from_dict(c_hex.to_dict()) == c_hex

    img_p = _write_raster(tmp_path / "feat.tif", np.ones((1, 4, 4), dtype=np.float32))
    cfg_p = tmp_path / "hex_cfg.json"
    cfg_w = rcd.write_cls_config(
        cfg_p,
        [rcd.ImageBandInfo(img_p, name="f")],
        {"Forest": rcd.ClassInfoObj(out_id=10, colour="#228b22")},
    )
    cfg_r = rcd.read_cls_config(cfg_p)
    assert cfg_r == cfg_w
    assert cfg_r.get_colour_table() == {10: [34, 139, 34, 255]}

    # Conflicting or invalid colour specifications raise RSGISPyException
    with pytest.raises(rsgispy.RSGISPyException):
        rcd.ClassInfoObj(out_id=1, red=10, colour="#ff0000")
    with pytest.raises(rsgispy.RSGISPyException):
        rcd.ClassInfoObj(out_id=1, colour="#zzzzzz")
    with pytest.raises(rsgispy.RSGISPyException):
        rcd.ClassInfoObj(out_id=1, colour="#123")
