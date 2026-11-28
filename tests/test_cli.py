"""Comprehensive tests for all 11 rsgispy.cli command-line tools."""

from __future__ import annotations

import datetime
import json
import os
import sys
import warnings
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point, Polygon

from rsgispy import RSGISPyException
from rsgispy.cli import (
    batchconvert2cog,
    buildimglut,
    calcimgstats,
    chkgdalfile,
    copybandnames,
    filehash,
    imginfo,
    proj,
    userpassfile,
    vectools,
    wgs84imgres,
)
from rsgispy.cli._common import (
    format_bytes,
    format_table,
    resolve_input_files,
    run_cli,
)
from rsgispy.imageutils.metadata import get_band_names, set_band_names
from rsgispy.tools.credentials import get_username_password


def _make_geotiff(
    path,
    *,
    width: int = 32,
    height: int = 32,
    count: int = 1,
    dtype: str = "uint8",
    crs: str = "EPSG:4326",
    origin=(10.0, 20.0),
    res=(0.01, 0.01),
    nodata=0,
    data=None,
    descriptions=None,
    thematic: bool = False,
):
    """Create a small synthetic GeoTIFF for testing."""
    transform = from_origin(origin[0], origin[1], res[0], res[1])
    if data is None:
        arr = np.arange(count * width * height, dtype=dtype).reshape(
            (count, height, width)
        )
        if np.issubdtype(np.dtype(dtype), np.integer):
            arr = (arr % 5).astype(dtype) + 1
    else:
        arr = data

    with rasterio.open(
        str(path),
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=count,
        dtype=dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(arr)
        if descriptions:
            for idx, name in enumerate(descriptions, start=1):
                dst.set_band_description(idx, name)
        if thematic:
            for idx in range(1, count + 1):
                dst.update_tags(idx, LAYER_TYPE="thematic")
    return str(path)


def _make_vector(path, *, layer: str = "polys", driver: str = "GPKG"):
    """Create a small synthetic polygon vector file (GPKG or GeoParquet)."""
    gdf = gpd.GeoDataFrame(
        {
            "id": [1, 2, 3],
            "cls": ["forest", "water", "forest"],
        },
        geometry=[
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
            Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]),
        ],
        crs="EPSG:4326",
    )
    if str(path).lower().endswith(".parquet"):
        gdf.to_parquet(str(path))
    else:
        gdf.to_file(str(path), layer=layer, driver=driver)
    return str(path)


class TestCommonHelpers:
    """Tests for shared CLI infrastructure helpers."""

    def test_format_bytes(self):
        """format_bytes produces compact human-readable strings."""
        assert format_bytes(512) == "512 B"
        assert format_bytes(2048) == "2.00 KB"
        assert format_bytes(5 * 1024 * 1024) == "5.00 MB"

    def test_format_table(self):
        """format_table aligns columns and renders header separator."""
        tbl = format_table(["#", "Name"], [["1", "Red"], ["10", "NIR"]], ["r", "l"])
        lines = tbl.splitlines()
        assert len(lines) == 4
        assert "10  NIR" in lines[3]

    def test_resolve_input_files_glob_and_list(self, tmp_path):
        """resolve_input_files expands both globs and direct paths with natural sort."""
        for name in ("tile_10.tif", "tile_2.tif", "tile_1.tif"):
            (tmp_path / name).write_bytes(b"x")
        pattern = str(tmp_path / "tile_*.tif")
        res = resolve_input_files([pattern])
        assert [os.path.basename(p) for p in res] == [
            "tile_1.tif",
            "tile_2.tif",
            "tile_10.tif",
        ]

        with pytest.raises(RSGISPyException):
            resolve_input_files([str(tmp_path / "missing_*.tif")])

    def test_run_cli_handles_errors(self, capsys):
        """run_cli converts RSGISPyException into exit code 1 with stderr output."""

        def _failing(_argv):
            raise RSGISPyException("boom")

        rc = run_cli(_failing, [])
        assert rc == 1
        assert "Error: boom" in capsys.readouterr().err


class TestImgInfoCLI:
    """Tests for rsgisimginfo CLI."""

    def test_imginfo_geographic_and_thematic(self, tmp_path, capsys):
        """Compact report displays format, grid, metre approx, CRS, and band table."""
        img = _make_geotiff(
            tmp_path / "thm.tif",
            width=64,
            height=64,
            count=1,
            dtype="uint8",
            crs="EPSG:4326",
            descriptions=["Classes"],
            thematic=True,
        )
        calcimgstats.main(["-i", img, "--thematic", "--no-pbar"])
        capsys.readouterr()

        rc = imginfo.main(["-i", img])
        assert rc == 0
        out = capsys.readouterr().out
        assert "=== thm.tif ===" in out
        assert "64 x 64 px | 1 band | uint8" in out
        assert "EPSG:4326" in out
        assert "~" in out and "m)" in out
        assert "thematic" in out
        assert "Classes" in out
        assert "Overviews" in out

    def test_imginfo_projected_wgs84_bbox_and_flags(self, tmp_path, capsys):
        """Projected image shows WGS84 BBox, WKT, PROJ4, metadata, and --calc-stats."""
        img = _make_geotiff(
            tmp_path / "utm.tif",
            width=32,
            height=32,
            count=2,
            dtype="float32",
            crs="EPSG:32630",
            origin=(500000.0, 5700000.0),
            res=(10.0, 10.0),
            nodata=-9999.0,
            descriptions=["Red", "NIR"],
        )
        rc = imginfo.main(
            ["-i", img, "--wkt", "--proj4", "--metadata", "--calc-stats"]
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "WGS84 BBox:" in out
        assert "EPSG:32630" in out
        assert "WKT:" in out
        assert "PROJ4:" in out
        assert "Metadata:" in out
        assert "Red" in out and "NIR" in out

    def test_imginfo_json_output(self, tmp_path, capsys):
        """--json returns valid structured JSON metadata."""
        img = _make_geotiff(tmp_path / "j.tif", count=2, descriptions=["B1", "B2"])
        rc = imginfo.main(["-i", img, "--json", "--calc-stats"])
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert data["width"] == 32
        assert data["n_bands"] == 2
        assert data["crs"]["authority"] == "EPSG:4326"
        assert len(data["bands"]) == 2
        assert data["bands"][0]["stats"]["min"] is not None


class TestCalcImgStatsCLI:
    """Tests for rsgiscalcimgstats CLI."""

    def test_continuous_and_thematic(self, tmp_path):
        """Populates continuous and thematic stats and pyramids cleanly."""
        cont = _make_geotiff(
            tmp_path / "cont.tif",
            width=64,
            height=64,
            dtype="float32",
            nodata=-999.0,
        )
        assert calcimgstats.main(["-i", cont, "-n", "-999", "--no-pbar"]) == 0
        with rasterio.open(cont) as ds:
            tags = ds.tags(1)
            assert "STATISTICS_MINIMUM" in tags
            assert "STATISTICS_MAXIMUM" in tags

        thm = _make_geotiff(
            tmp_path / "thm.tif",
            width=64,
            height=64,
            dtype="uint8",
            nodata=0,
        )
        assert (
            calcimgstats.main(
                ["-i", thm, "--thematic", "--no-pyramids", "--no-clrtab", "--no-pbar"]
            )
            == 0
        )
        with rasterio.open(thm) as ds:
            assert ds.tags(1).get("LAYER_TYPE") == "thematic"
            assert ds.overviews(1) == []


class TestVecToolsCLI:
    """Tests for rsgisvectools CLI."""

    def test_vectools_summary_and_queries(self, tmp_path, capsys):
        """Summary mode and individual query flags work on GeoPackage."""
        gpkg = _make_vector(tmp_path / "test.gpkg", layer="plots")

        assert vectools.main(["-i", gpkg]) == 0
        summary = capsys.readouterr().out
        assert "=== test.gpkg ===" in summary
        assert "Features: 3" in summary
        assert "EPSG:4326" in summary

        assert vectools.main(["-i", gpkg, "--layers"]) == 0
        assert "plots" in capsys.readouterr().out

        assert (
            vectools.main(
                [
                    "-i",
                    gpkg,
                    "-l",
                    "plots",
                    "-c",
                    "cls",
                    "--columns",
                    "--n-feats",
                    "--bbox",
                    "--geom",
                    "--unique-vals",
                    "--json",
                ]
            )
            == 0
        )
        payload = json.loads(capsys.readouterr().out)
        assert payload["n_features"] == 3
        assert set(payload["unique_values"]) == {"forest", "water"}
        assert "cls" in payload["columns"]

    def test_vectools_geoparquet(self, tmp_path, capsys):
        """vectools inspects GeoParquet files seamlessly."""
        pq = _make_vector(tmp_path / "test.parquet")
        assert vectools.main(["-i", pq, "--n-feats", "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["n_features"] == 3

    def test_vectools_unique_vals_requires_column(self, tmp_path):
        """--unique-vals without --column raises RSGISPyException."""
        gpkg = _make_vector(tmp_path / "test.gpkg")
        with pytest.raises(RSGISPyException):
            vectools.main(["-i", gpkg, "--unique-vals"])


class TestBatchConvert2CogCLI:
    """Tests for rsgisbatchconvert2cog CLI."""

    def test_batch_convert_deflate_byte_and_skip(self, tmp_path, capsys):
        """Converts to COG with DEFLATE + --byte, and respects --skip-existing."""
        in_dir = tmp_path / "in"
        out_dir = tmp_path / "out"
        in_dir.mkdir()
        _make_geotiff(
            in_dir / "a.tif", width=64, height=64, dtype="uint16", thematic=True
        )
        _make_geotiff(in_dir / "b.tif", width=64, height=64, dtype="uint16")

        rc = batchconvert2cog.main(
            [
                "-i",
                str(in_dir / "*.tif"),
                "-o",
                str(out_dir),
                "--compress",
                "DEFLATE",
                "--byte",
                "--no-pbar",
            ]
        )
        assert rc == 0
        out_txt = capsys.readouterr().out
        assert "2 converted, 0 skipped" in out_txt

        with rasterio.open(out_dir / "a.tif") as ds:
            assert ds.dtypes[0] == "uint8"
            assert ds.tags(ns="IMAGE_STRUCTURE").get("LAYOUT") == "COG"
            assert ds.compression.name.upper() == "DEFLATE"

        # Second run with --check-existing skips valid outputs
        rc2 = batchconvert2cog.main(
            [
                "-i",
                str(in_dir / "*.tif"),
                "-o",
                str(out_dir),
                "--check-existing",
                "--no-pbar",
            ]
        )
        assert rc2 == 0
        assert "0 converted, 2 skipped" in capsys.readouterr().out


class TestBuildImgLutCLI:
    """Tests for rsgisbuildimglut CLI."""

    def test_build_lut_gpkg(self, tmp_path):
        """Creates a GeoPackage LUT with one polygon per input raster."""
        img1 = _make_geotiff(tmp_path / "tile1.tif", origin=(10.0, 20.0))
        img2 = _make_geotiff(tmp_path / "tile2.tif", origin=(11.0, 20.0))
        lut_path = tmp_path / "lut.gpkg"

        rc = buildimglut.main(
            [
                "-i",
                img1,
                img2,
                "-o",
                str(lut_path),
                "-l",
                "tiles",
                "-f",
                "GPKG",
                "--wgs84",
                "--overwrite",
                "--no-pbar",
            ]
        )
        assert rc == 0
        gdf = gpd.read_file(str(lut_path), layer="tiles")
        assert len(gdf) == 2
        assert gdf.crs.to_epsg() == 4326


class TestChkGdalFileCLI:
    """Tests for rsgischkgdalfile CLI."""

    def test_check_valid_and_corrupt_raster(self, tmp_path):
        """Returns 0 for valid rasters and 1 for corrupt rasters."""
        good = _make_geotiff(tmp_path / "good.tif", crs="EPSG:4326")
        assert (
            chkgdalfile.main(
                ["-i", good, "--chk-proj", "--epsg", "4326", "--read-img"]
            )
            == 0
        )

        bad = tmp_path / "bad.tif"
        bad.write_bytes(b"not-a-valid-tiff")
        assert chkgdalfile.main(["-i", str(bad), "--quiet"]) == 1

    def test_check_vector(self, tmp_path):
        """Validates vector files with --type vector."""
        vec = _make_vector(tmp_path / "v.gpkg")
        assert (
            chkgdalfile.main(
                ["-i", vec, "--type", "vector", "--chk-proj", "--epsg", "4326"]
            )
            == 0
        )


class TestFileHashCLI:
    """Tests for rsgisfilehash CLI."""

    def test_single_and_multi_hash(self, tmp_path, capsys):
        """Computes SHA256 for single and multiple files plus JSON output."""
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("hello", encoding="utf-8")
        f2.write_text("world", encoding="utf-8")

        assert filehash.main(["-i", str(f1), "-a", "SHA256"]) == 0
        digest = capsys.readouterr().out.strip()
        assert len(digest) == 64

        assert filehash.main(["-i", str(f1), str(f2), "-a", "md5", "--json"]) == 0
        items = json.loads(capsys.readouterr().out)
        assert len(items) == 2
        assert items[0]["algorithm"] == "md5"


class TestProjCLI:
    """Tests for rsgisproj CLI."""

    def test_proj_epsg_utm_image_vector_and_file_output(self, tmp_path, capsys):
        """Converts CRS from EPSG, UTM, raster, vector, and WKT files."""
        assert proj.main(["--utm", "30N", "-f", "epsg"]) == 0
        assert capsys.readouterr().out.strip() == "EPSG:32630"

        assert proj.main(["--epsg", "4326", "-f", "epsg", "proj4"]) == 0
        out = capsys.readouterr().out
        assert "[EPSG]" in out and "EPSG:4326" in out
        assert "[PROJ4]" in out and "+proj=longlat" in out

        img = _make_geotiff(tmp_path / "p.tif", crs="EPSG:32648")
        out_wkt = tmp_path / "crs.wkt"
        assert proj.main(["--image", img, "-f", "wkt", "-o", str(out_wkt)]) == 0
        assert out_wkt.exists()

        assert proj.main(["--wkt-file", str(out_wkt), "-f", "epsg"]) == 0
        assert capsys.readouterr().out.strip() == "EPSG:32648"

        vec = _make_vector(tmp_path / "v.gpkg")
        assert proj.main(["--vector", vec, "-f", "epsg"]) == 0
        assert capsys.readouterr().out.strip() == "EPSG:4326"

    def test_proj_invalid_utm_raises(self):
        """Invalid UTM string raises RSGISPyException."""
        with pytest.raises(RSGISPyException):
            proj.main(["--utm", "99X"])


class TestUserPassFileCLI:
    """Tests for rsgisuserpassfile CLI."""

    def test_creates_readable_credentials_file(self, tmp_path, monkeypatch):
        """Creates base64 credentials file readable by get_username_password."""
        out_file = tmp_path / "creds.txt"
        monkeypatch.setattr("getpass.getpass", lambda _prompt="": "s3cr3t")

        rc = userpassfile.main(["-o", str(out_file), "-u", "alice"])
        assert rc == 0
        user, pwd = get_username_password(str(out_file))
        assert user == "alice"
        assert pwd == "s3cr3t"

        # Existing file without --overwrite raises
        with pytest.raises(RSGISPyException):
            userpassfile.main(["-o", str(out_file), "-u", "alice"])


class TestWGS84ImgResCLI:
    """Tests for rsgiswgs84imgres CLI."""

    def test_wgs84_resolution_and_non_geographic_error(self, tmp_path, capsys):
        """Computes metre resolution for WGS84 image and rejects projected CRS."""
        wgs = _make_geotiff(
            tmp_path / "wgs.tif",
            crs="EPSG:4326",
            origin=(106.0, 10.0),
            res=(0.0002, 0.0002),
        )
        assert wgs84imgres.main(["-i", wgs, "--json"]) == 0
        data = json.loads(capsys.readouterr().out)
        assert 20.0 < data["res_m_x"] < 25.0
        assert 20.0 < data["res_m_y"] < 25.0

        utm = _make_geotiff(tmp_path / "utm.tif", crs="EPSG:32630", res=(10.0, 10.0))
        with pytest.raises(RSGISPyException):
            wgs84imgres.main(["-i", utm])


class TestCopyBandNamesCLI:
    """Tests for rsgiscopybandnames CLI."""

    def test_copy_band_names_and_mismatch_error(self, tmp_path):
        """Copies band names across matching rasters and fails on count mismatch."""
        ref = _make_geotiff(
            tmp_path / "ref.tif",
            count=3,
            descriptions=["Blue", "Green", "Red"],
        )
        target = _make_geotiff(tmp_path / "target.tif", count=3)
        set_band_names(target, ["B1", "B2", "B3"])

        assert copybandnames.main(["-r", ref, "-i", target]) == 0
        assert get_band_names(target) == ["Blue", "Green", "Red"]

        mismatch = _make_geotiff(tmp_path / "oneband.tif", count=1)
        with pytest.raises(RSGISPyException):
            copybandnames.main(["-r", ref, "-i", mismatch])


def _write_raster(
    path: str,
    arr: np.ndarray,
    *,
    origin=(500000.0, 5200000.0),
    res=(10.0, 10.0),
    crs: str = "EPSG:32630",
    nodata=None,
    tiled: bool = False,
    blocksize: int | None = None,
) -> str:
    """Write a 2D or 3D NumPy array to a GeoTIFF."""
    if arr.ndim == 2:
        data = arr[np.newaxis, :, :]
    else:
        data = arr
    count, height, width = data.shape
    profile = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": count,
        "dtype": str(data.dtype),
        "crs": crs,
        "transform": from_origin(origin[0], origin[1], res[0], res[1]),
        "nodata": nodata,
    }
    if tiled:
        profile["tiled"] = True
        if blocksize is not None:
            profile["blockxsize"] = blocksize
            profile["blockysize"] = blocksize
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
    return path


def test_cli_main_reads_sys_argv(monkeypatch, capsys):
    """cli_main parses sys.argv and returns the exit code."""
    # Exercise cli_main end-to-end via sys.argv on proj
    monkeypatch.setattr(sys, "argv", ["rsgisproj", "--epsg", "4326", "-f", "epsg"])
    assert proj.cli_main() == 0
    assert capsys.readouterr().out.strip() == "EPSG:4326"


def test_batchconvert2cog_rejects_inplace_and_duplicate_names(tmp_path):
    """batchconvert2cog rejects in-place overwrite and duplicate output basenames."""
    dir1 = tmp_path / "d1"
    dir2 = tmp_path / "d2"
    out_dir = tmp_path / "out"
    dir1.mkdir()
    dir2.mkdir()
    img1 = _write_raster(
        str(dir1 / "same.tif"), np.ones((8, 8), dtype=np.uint8)
    )
    img2 = _write_raster(
        str(dir2 / "same.tif"), np.ones((8, 8), dtype=np.uint8)
    )

    # In-place overwrite rejected
    with pytest.raises(RSGISPyException, match="in-place"):
        batchconvert2cog.main(["-i", img1, "-o", str(dir1), "--no-pbar"])

    # Duplicate output filename across dirs rejected
    with pytest.raises(RSGISPyException, match="Duplicate output filename"):
        batchconvert2cog.main(
            ["-i", img1, img2, "-o", str(out_dir), "--no-pbar"]
        )


def test_vectools_empty_bbox_and_json_default_str(tmp_path, capsys):
    """vectools handles empty layers cleanly and serialises dates in --json."""
    empty_gpkg = str(tmp_path / "empty.gpkg")
    empty_gdf = gpd.GeoDataFrame(
        {"id": pd.Series(dtype="int64")},
        geometry=gpd.GeoSeries([], crs="EPSG:4326"),
        crs="EPSG:4326",
    )
    empty_gdf.to_file(empty_gpkg, layer="empty", driver="GPKG")

    assert vectools.main(["-i", empty_gpkg]) == 0
    txt = capsys.readouterr().out
    assert "Extent    : -" in txt

    assert vectools.main(["-i", empty_gpkg, "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["bbox"] is None
    assert payload["n_features"] == 0

    # Non-JSON-native date column serialises cleanly via default=str
    date_gpkg = str(tmp_path / "dates.gpkg")
    date_gdf = gpd.GeoDataFrame(
        {"obs_date": [datetime.date(2025, 1, 15)]},
        geometry=[Point(0.0, 0.0)],
        crs="EPSG:4326",
    )
    date_gdf.to_file(date_gpkg, layer="dates", driver="GPKG")
    assert (
        vectools.main(
            ["-i", date_gpkg, "-c", "obs_date", "--unique-vals", "--json"]
        )
        == 0
    )
    d_payload = json.loads(capsys.readouterr().out)
    assert "2025-01-15" in str(d_payload["unique_values"][0])


def test_proj_format_crs_error_handling():
    """_format_crs wraps CRSError and raises RSGISPyException if to_proj4() returns None."""
    fake_none = SimpleNamespace(name="NoProj4", to_proj4=lambda: None)
    with pytest.raises(RSGISPyException, match="PROJ4"):
        proj._format_crs(fake_none, "proj4")

    def _raise_crs_err(**_kwargs):
        raise pyproj.exceptions.CRSError("unsupported wkt1")

    fake_err = SimpleNamespace(name="BadWKT", to_wkt=_raise_crs_err)
    with pytest.raises(RSGISPyException, match="Could not format CRS"):
        proj._format_crs(fake_err, "wkt")


def test_imginfo_densified_bounds_streaming_stats_and_tiled_detection(
    tmp_path
):
    """Verify densified WGS84 bounds, streaming Welford stats, and tiled detection."""
    rng = np.random.default_rng(123)
    arr = rng.normal(loc=50.0, scale=12.5, size=(64, 64)).astype(np.float32)
    arr[0:5, 0:5] = -9999.0
    tiled_tif = _write_raster(
        str(tmp_path / "tiled_stats.tif"),
        arr,
        crs="EPSG:32630",
        nodata=-9999.0,
        tiled=True,
        blocksize=16,
    )
    valid = arr[arr != -9999.0].astype(np.float64)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        info = imginfo.collect_img_info(tiled_tif, calc_stats=True)
    assert not any("is_tiled" in str(w.message) for w in caught)
    assert info["is_tiled"] is True
    assert info["wgs84_bbox"] is not None
    st = info["bands"][0]["stats"]
    assert st["min"] == pytest.approx(float(np.min(valid)), rel=1e-6)
    assert st["max"] == pytest.approx(float(np.max(valid)), rel=1e-6)
    assert st["mean"] == pytest.approx(float(np.mean(valid)), rel=1e-6)
    assert st["std"] == pytest.approx(float(np.std(valid)), rel=1e-6)
    assert st["valid_percent"] == pytest.approx(
        100.0 * valid.size / arr.size, rel=1e-6
    )

    # 256x256 tiled GTiff (blk_w == width) detected as tiled; stripped 256x256 as False
    t256 = _write_raster(
        str(tmp_path / "t256.tif"),
        np.ones((256, 256), dtype=np.uint8),
        tiled=True,
        blocksize=256,
    )
    s256 = _write_raster(
        str(tmp_path / "s256.tif"),
        np.ones((256, 256), dtype=np.uint8),
        tiled=False,
    )
    assert imginfo.collect_img_info(t256)["is_tiled"] is True
    assert imginfo.collect_img_info(s256)["is_tiled"] is False
