"""Targeted review regression tests for tools, dataaccess, and droneutils."""

import asyncio
import datetime
import hashlib
import io
import os
import stat
import sys
import tarfile
import types
import zipfile
from unittest.mock import patch

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
import shapely.geometry
from PIL import Image

import rsgispy
import rsgispy.dataaccess.copernicus_odata as copernicus_odata
import rsgispy.dataaccess.nasa_cmr as nasa_cmr
import rsgispy.dataaccess.openstreetmap as openstreetmap
import rsgispy.dataaccess.planet_data_api as planet_data_api
import rsgispy.dataaccess.usgs_m2m as usgs_m2m
import rsgispy.droneutils as droneutils
import rsgispy.tools.accuracy as accuracy
import rsgispy.tools.archives as archives
import rsgispy.tools.credentials as credentials
import rsgispy.tools.featureselection as featureselection
import rsgispy.tools.ftp as ftp_tools
import rsgispy.tools.http as http_tools
import rsgispy.tools.imaging as imaging
import rsgispy.tools.stats as stats
import rsgispy.tools.tables as tables
import rsgispy.tools.thresholding as thresholding
import rsgispy.tools.tilecache as tilecache
import rsgispy.tools.wmts as wmts


# ---------------------------------------------------------------------------
# 1. Previously edited tools: http, archives, ftp, credentials, tables, tilecache
# ---------------------------------------------------------------------------


def test_http_json_5xx_and_wget_security(tmp_path, monkeypatch):
    """Verify HTTP >= 400 rejection and wget SSL/config-file handling."""

    class _Resp500:
        status_code = 500
        text = "Internal Server Error"

        def close(self):
            pass

    monkeypatch.setattr(http_tools, "HAVE_REQUESTS", True)
    monkeypatch.setattr(
        http_tools,
        "requests",
        types.SimpleNamespace(post=lambda *args, **kwargs: _Resp500()),
        raising=False,
    )
    with pytest.raises(rsgispy.RSGISPyException, match="500"):
        http_tools.send_http_json_request("https://example.com/api", {"a": 1})

    recorded_cmds = []
    recorded_cfg = {}

    def _fake_call(cmd):
        recorded_cmds.append(list(cmd))
        for arg in cmd:
            if arg.startswith("--config="):
                cfg_path = arg.split("=", 1)[1]
                recorded_cfg["path"] = cfg_path
                recorded_cfg["mode"] = stat.S_IMODE(os.stat(cfg_path).st_mode)
                with open(cfg_path, "r", encoding="utf-8") as fp:
                    recorded_cfg["content"] = fp.read()
        return 0

    monkeypatch.setattr(http_tools.subprocess, "call", _fake_call)

    out_path = str(tmp_path / "file.bin")
    ok, _ = http_tools.wget_download_file(
        "https://example.com/file.bin",
        out_path,
        username="alice",
        password="secret",
    )
    assert ok is True
    assert "--no-check-certificate" not in recorded_cmds[0]
    assert all("--password" not in arg for arg in recorded_cmds[0])
    assert any(arg.startswith("--config=") for arg in recorded_cmds[0])
    assert "user = alice\npassword = secret\n" == recorded_cfg["content"]
    if os.name != "nt":
        assert recorded_cfg["mode"] == 0o600
    assert not os.path.exists(recorded_cfg["path"])

    ok2, _ = http_tools.wget_download_file(
        "https://example.com/file.bin", out_path, verify_ssl=False
    )
    assert ok2 is True
    assert "--no-check-certificate" in recorded_cmds[1]

    with pytest.raises(rsgispy.RSGISPyException, match="newline"):
        http_tools.wget_download_file(
            "https://example.com/file.bin",
            out_path,
            username="alice\nbad",
            password="secret",
        )


def test_archives_zip_and_tar_traversal_guards(tmp_path):
    """Reject path traversal, absolute paths, and symlinks in zip/tar archives."""
    for name in ("out_z1", "out_z2", "out_z3", "out_t1", "out_t2", "out_good"):
        (tmp_path / name).mkdir()

    bad_zip = tmp_path / "traversal.zip"
    with zipfile.ZipFile(bad_zip, "w") as zf:
        zf.writestr("../escape.txt", "nope")
    with pytest.raises(rsgispy.RSGISPyException, match="Unsafe"):
        archives.unzip_file(str(bad_zip), str(tmp_path / "out_z1"))

    abs_zip = tmp_path / "abs.zip"
    with zipfile.ZipFile(abs_zip, "w") as zf:
        zf.writestr("/tmp/escape.txt", "nope")
    with pytest.raises(rsgispy.RSGISPyException, match="Unsafe"):
        archives.unzip_file(str(abs_zip), str(tmp_path / "out_z2"))

    sym_zip = tmp_path / "symlink.zip"
    with zipfile.ZipFile(sym_zip, "w") as zf:
        zinfo = zipfile.ZipInfo("link.txt")
        zinfo.external_attr = (stat.S_IFLNK | 0o777) << 16
        zf.writestr(zinfo, "/etc/passwd")
    with pytest.raises(rsgispy.RSGISPyException, match="symlink"):
        archives.unzip_file(str(sym_zip), str(tmp_path / "out_z3"))

    bad_tar = tmp_path / "traversal.tar.gz"
    with tarfile.open(bad_tar, "w:gz") as tf:
        tinfo = tarfile.TarInfo(name="../escape.txt")
        payload = b"evil"
        tinfo.size = len(payload)
        tf.addfile(tinfo, io.BytesIO(payload))
    with pytest.raises(rsgispy.RSGISPyException, match="Unsafe"):
        archives.untar_gz_file(str(bad_tar), str(tmp_path / "out_t1"))

    sym_tar = tmp_path / "symlink.tar.gz"
    with tarfile.open(sym_tar, "w:gz") as tf:
        tinfo = tarfile.TarInfo(name="bad_link")
        tinfo.type = tarfile.SYMTYPE
        tinfo.linkname = "../../outside"
        tf.addfile(tinfo)
    with pytest.raises(rsgispy.RSGISPyException, match="symlink"):
        archives.untar_gz_file(str(sym_tar), str(tmp_path / "out_t2"))

    good_zip = tmp_path / "good.zip"
    with zipfile.ZipFile(good_zip, "w") as zf:
        zf.writestr("sub/hello.txt", "world")
    out_dir = archives.unzip_file(str(good_zip), str(tmp_path / "out_good"))
    assert (tmp_path / "out_good" / "good" / "sub" / "hello.txt").read_text(
        encoding="utf-8"
    ) == "world"
    assert os.path.isdir(out_dir)


def test_ftp_list_parser_spaces_and_special_entries():
    """Preserve spaces in FTP LIST filenames and ignore total/. /.. entries."""
    # pylint: disable=protected-access
    assert ftp_tools._parse_ftp_list_line("") is None
    assert ftp_tools._parse_ftp_list_line("total 42") is None
    assert ftp_tools._parse_ftp_list_line(
        "-rw-r--r-- 1 user group 1024 Jun 01 12:00 my raster file.tif"
    ) == (False, "my raster file.tif")
    assert ftp_tools._parse_ftp_list_line(
        "drwxr-xr-x 2 user group 4096 Jun 01 12:00 sub folder"
    ) == (True, "sub folder")

    class _FakeFTP:
        def __init__(self):
            self.cwd_path = "/root"

        def cwd(self, path):
            self.cwd_path = path

        def retrlines(self, _cmd, callback):
            if self.cwd_path == "/root":
                lines = (
                    "total 16",
                    "drwxr-xr-x 2 u g 4096 Jun 01 12:00 .",
                    "drwxr-xr-x 2 u g 4096 Jun 01 12:00 ..",
                    "drwxr-xr-x 2 u g 4096 Jun 01 12:00 data dir",
                    "-rw-r--r-- 1 u g 100 Jun 01 12:00 scene 01.tif",
                )
            else:
                lines = ("-rw-r--r-- 1 u g 200 Jun 01 12:00 nested file.tif",)
            for ln in lines:
                callback(ln)

    ftp_files = {}
    non_dirs = ftp_tools._traverse_nlst_ftp_dirs(
        _FakeFTP(), "/root", ftp_files, try_n_times=1
    )
    assert "/root/data dir" in ftp_files
    assert ftp_files["/root"] == ["/root/scene 01.tif"]
    assert ftp_files["/root/data dir"] == ["/root/data dir/nested file.tif"]
    assert sorted(non_dirs) == [
        "/root/data dir/nested file.tif",
        "/root/scene 01.tif",
    ]


def test_credentials_file_permissions_and_roundtrip(tmp_path):
    """Credential files are created with 0o600 permissions and roundtrip cleanly."""
    cred_file = str(tmp_path / "creds.txt")
    credentials.create_username_password_file("alice", "s3cr3t!", cred_file)
    if os.name != "nt":
        assert stat.S_IMODE(os.stat(cred_file).st_mode) == 0o600
    assert credentials.get_username_password(cred_file) == ("alice", "s3cr3t!")


def test_tables_allow_pickle_guard(tmp_path):
    """read_dataframe_table rejects pickle inputs when allow_pickle=False."""
    df = pd.DataFrame({"a": [1, 2], "b": [3.5, 4.5]})
    pkl_path = tmp_path / "sample.pkl"
    df.to_pickle(pkl_path)

    with pytest.raises(rsgispy.RSGISPyException, match="allow_pickle=False"):
        tables.read_dataframe_table(pkl_path, allow_pickle=False)

    loaded = tables.read_dataframe_table(pkl_path, allow_pickle=True)
    pd.testing.assert_frame_equal(loaded, df)


def test_tilecache_tile_size_and_get_tiles():
    """Verify tile_size support and get_tiles boundary/TMS indexing."""
    res_256 = tilecache.resolution(10, tile_size=256)
    res_512 = tilecache.resolution(10, tile_size=512)
    assert np.isclose(res_512, res_256 / 2.0)

    t256 = tilecache.get_tile_for_point(-4.08, 52.41, zoom=10, tile_size=256)
    t512 = tilecache.get_tile_for_point(-4.08, 52.41, zoom=10, tile_size=512)
    assert t256 == t512

    # Exact quadrant at zoom=1: west=[-180,0], north=[0,85] -> XYZ (0,0,1), TMS (0,1,1)
    xyz_tiles = tilecache.get_tiles([-180.0, 0.0, 0.0, 85.0], zooms=1, tms=False)
    tms_tiles = tilecache.get_tiles([-180.0, 0.0, 0.0, 85.0], zooms=1, tms=True)
    assert xyz_tiles[1] == [(0, 0, 1)]
    assert tms_tiles[1] == [(0, 1, 1)]


# ---------------------------------------------------------------------------
# 2. stats.py & thresholding.py
# ---------------------------------------------------------------------------


def test_stats_skgstat_guard_and_normalise_constant_band():
    """calc_semivariogram guards missing skgstat; normalise_img_data avoids NaN."""
    with patch.dict(sys.modules, {"skgstat": None}):
        with pytest.raises(rsgispy.RSGISPyException, match="scikit-gstat"):
            stats.calc_semivariogram(np.array([[0.0, 0.0]]), np.array([1.0]))

    img = np.array(
        [
            [[5.0, 5.0], [5.0, 5.0]],
            [[10.0, 20.0], [30.0, 50.0]],
        ],
        dtype=np.float64,
    )
    norm = stats.normalise_img_data(img)
    assert norm.shape == img.shape
    assert np.all(np.isfinite(norm))
    np.testing.assert_allclose(norm[0], np.zeros((2, 2)))
    np.testing.assert_allclose(norm[1], [[0.0, 0.25], [0.5, 1.0]])


def test_thresholding_cross_entropy_and_li_edge_cases():
    """Guard boundary cross-entropy splits, constant arrays, and invalid Li args."""
    sample = np.array([1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0])
    # Boundary thresholds where one side is empty must not raise IndexError/ZeroDivisionError
    for thres in (-10.0, 1.0, 5.0, 100.0):
        val = thresholding.calc_hist_cross_entropy(sample, thres)
        assert isinstance(val, float) and np.isfinite(val)

    # Constant / single-unique array returns the constant value cleanly
    assert thresholding.calc_li_threshold(np.full(50, 7.5)) == 7.5
    assert thresholding.calc_li_threshold(np.array([-3.0, -3.0, np.nan])) == -3.0

    with pytest.raises(rsgispy.RSGISPyException, match="tolerance"):
        thresholding.calc_li_threshold(np.array([1.0, 2.0, 3.0]), tolerance=0.0)
    with pytest.raises(rsgispy.RSGISPyException, match="max_iter"):
        thresholding.calc_li_threshold(np.array([1.0, 2.0, 3.0]), max_iter=0)
    with pytest.raises(rsgispy.RSGISPyException, match="finite"):
        thresholding.calc_li_threshold(np.array([np.nan, np.inf]))

    bimodal = np.concatenate([np.full(40, 10.0), np.full(40, 100.0)])
    t_val = thresholding.calc_li_threshold(bimodal, max_iter=5)
    assert 10.0 < t_val < 100.0


# ---------------------------------------------------------------------------
# 3. wmts.py & imaging.py
# ---------------------------------------------------------------------------


def test_wmts_optional_deps_timeout_and_mixed_tile_modes(tmp_path, monkeypatch):
    """Verify optional dependency guards, timeout forwarding, and mode normalisation."""
    with patch.dict(sys.modules, {"owslib": None, "owslib.wmts": None}):
        with pytest.raises(rsgispy.RSGISPyException, match="owslib"):
            wmts.get_wmts_layer_list("https://example.com/wmts")

    monkeypatch.setattr(wmts, "HAVE_PIL", False)
    with pytest.raises(rsgispy.RSGISPyException, match="pillow"):
        wmts.get_wmts_as_img(
            "https://example.com/wmts",
            "layer",
            [0.0, 1000.0, 0.0, 1000.0],
            3857,
            str(tmp_path / "out.tif"),
        )
    monkeypatch.setattr(wmts, "HAVE_PIL", True)

    captured_timeout = {}
    tile_modes = ["RGB", "RGBA", "L", "P"]

    class _FakeTileMatrix:
        def __init__(self):
            self.identifier = "1"
            self.scaledenominator = 279541132.0143589
            self.topleftcorner = (-20037508.34, 20037508.34)
            self.tilewidth = 256
            self.tileheight = 256
            self.matrixwidth = 2
            self.matrixheight = 2

    class _FakeTMS:
        def __init__(self):
            self.crs = "EPSG:3857"
            self.tilematrix = {"1": _FakeTileMatrix()}

    class _FakeLayer:
        def __init__(self):
            self.tilematrixsets = ["GoogleMapsCompatible"]
            self.styles = {"default": {}}

    class _FakeWMTSWithTimeout:
        def __init__(self, url, timeout=None):
            captured_timeout["url"] = url
            captured_timeout["timeout"] = timeout
            self.contents = {"lyr": _FakeLayer()}
            self.tilematrixsets = {"GoogleMapsCompatible": _FakeTMS()}
            self._call_idx = 0

        def gettile(self, **_kwargs):
            mode = tile_modes[self._call_idx % len(tile_modes)]
            self._call_idx += 1
            if mode == "P":
                pil_img = Image.new("RGB", (256, 256), color=(10, 20, 30)).convert("P")
            elif mode == "L":
                pil_img = Image.new("L", (256, 256), color=128)
            elif mode == "RGBA":
                pil_img = Image.new("RGBA", (256, 256), color=(10, 20, 30, 255))
            else:
                pil_img = Image.new("RGB", (256, 256), color=(10, 20, 30))
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            buf.seek(0)
            return buf

    fake_owslib_wmts = types.ModuleType("owslib.wmts")
    fake_owslib_wmts.WebMapTileService = _FakeWMTSWithTimeout
    fake_owslib = types.ModuleType("owslib")
    fake_owslib.wmts = fake_owslib_wmts

    with patch.dict(
        sys.modules, {"owslib": fake_owslib, "owslib.wmts": fake_owslib_wmts}
    ):
        layers = wmts.get_wmts_layer_list("https://example.com/wmts", timeout=12.5)
        assert layers == ["lyr"]
        assert captured_timeout["timeout"] == 12.5

        out_tif = str(tmp_path / "mixed_modes.tif")
        wmts.get_wmts_as_img(
            "https://example.com/wmts",
            "lyr",
            [-5000000.0, 5000000.0, -5000000.0, 5000000.0],
            3857,
            out_tif,
            zoom_level="1",
            tmp_dir=str(tmp_path / "wmts_tmp"),
            timeout=15.0,
        )
        assert captured_timeout["timeout"] == 15.0

    with rasterio.open(out_tif) as ds:
        assert ds.count == 3
        assert ds.width > 0 and ds.height > 0


def test_imaging_optional_pil_guard(tmp_path, monkeypatch):
    """Verify PIL guards in imaging.py raise RSGISPyException with pillow/imaging."""
    monkeypatch.setattr(imaging, "HAVE_PIL", False)
    with pytest.raises(rsgispy.RSGISPyException, match="pillow"):
        imaging.list_exif_info_pil(str(tmp_path / "x.jpg"))
    monkeypatch.setattr(imaging, "HAVE_IMAGEIO", True)
    with pytest.raises(rsgispy.RSGISPyException, match="pillow"):
        imaging.animate_img_set(
            [str(tmp_path / "x.png")],
            str(tmp_path / "out.gif"),
            resize=True,
        )


# ---------------------------------------------------------------------------
# 4. featureselection.py & accuracy.py
# ---------------------------------------------------------------------------


def test_featureselection_degenerate_inputs():
    """Handle zero denominators in CQV, constant predictors, and ss_tot == 0."""
    df_cqv = pd.DataFrame(
        {
            "zero_sum": [-2.0, -1.0, 1.0, 2.0],
            "all_zero": [0.0, 0.0, 0.0, 0.0],
            "valid": [1.0, 2.0, 4.0, 8.0],
        }
    )
    selected = featureselection.cqv_threshold(
        df_cqv, lowthreshold=0.1, highthreshold=0.9
    )
    assert selected == ["valid"]

    rng = np.random.default_rng(42)
    n_rows = 30
    x1 = np.linspace(0.0, 1.0, n_rows)
    df_corr = pd.DataFrame(
        {
            "y": 2.0 * x1 + rng.normal(0, 0.05, n_rows),
            "const_a": np.ones(n_rows),
            "const_b": np.full(n_rows, 5.0),
            "x1": x1,
            "x2": x1 * 1.1 + 0.01,
            "x3": rng.normal(0, 1, n_rows),
            "x4": rng.normal(0, 1, n_rows),
        }
    )
    sel_corr = featureselection.corr_feature_selection(
        df_corr,
        dep_vars=["y"],
        ind_vars=["const_a", "const_b", "x1", "x2", "x3", "x4"],
        n_min_clusters=2,
        n_max_clusters=4,
        use_pbar=False,
    )
    assert "const_a" not in sel_corr and "const_b" not in sel_corr
    assert len(sel_corr) >= 2

    df_all_const = pd.DataFrame(
        {"y": [1, 2, 3, 4], "c1": [1, 1, 1, 1], "c2": [2, 2, 2, 2]}
    )
    assert (
        featureselection.corr_feature_selection(
            df_all_const, ["y"], ["c1", "c2"], use_pbar=False
        )
        == []
    )

    # Exact linear fit => constant residuals => ss_tot == 0 in breusch_pagan_test
    x_mat = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0], [5.0, 6.0]])
    y_exact = 3.0 * x_mat[:, 0] - 2.0 * x_mat[:, 1] + 1.0
    bp_stat, p_val, msg = featureselection.breusch_pagan_test(x_mat, y_exact)
    assert bp_stat == 0.0
    assert p_val == 1.0
    assert "No significant" in msg


def test_accuracy_binary_validation_and_constant_arrays():
    """Validate binary values before uint8 cast and accept constant 0/1 arrays."""
    with pytest.raises(rsgispy.RSGISPyException, match="binary"):
        accuracy.accuracy_scores_binary(np.array([0, 256, 257]), np.array([0, 1, 1]))
    with pytest.raises(rsgispy.RSGISPyException, match="binary"):
        accuracy.accuracy_scores_binary(np.array([0.0, 0.5, 1.0]), np.array([0, 1, 1]))
    with pytest.raises(rsgispy.RSGISPyException, match="binary"):
        accuracy.bin_accuracy_scores_prob(
            np.array([0, 257, 256]), np.array([0.1, 0.8, 0.2])
        )

    metrics_z, vals_z = accuracy.accuracy_scores_binary(
        np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0])
    )
    scores_all_zero = dict(zip(metrics_z, vals_z))
    assert scores_all_zero["Accuracy"] == 1.0
    assert scores_all_zero["Specificity"] == 1.0
    assert np.all(np.isfinite(vals_z))

    metrics_o, vals_o = accuracy.accuracy_scores_binary(
        np.array([1, 1, 1, 1]), np.array([1, 1, 1, 1])
    )
    scores_all_one = dict(zip(metrics_o, vals_o))
    assert scores_all_one["Accuracy"] == 1.0
    assert scores_all_one["F1 Score"] == 1.0
    assert scores_all_one["Sensitivity"] == 1.0

    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.5, 1.5, 4.0, 3.0])
    mse, bias_sq, var, noise = accuracy.decompose_bias_variance(y_true, y_pred)
    assert np.isclose(mse, bias_sq + var)
    assert np.isclose(noise, 0.0)
    with pytest.raises(rsgispy.RSGISPyException, match="dimensions"):
        accuracy.decompose_bias_variance(np.zeros((2, 2)), np.zeros((4,)))
    with pytest.raises(rsgispy.RSGISPyException, match="size"):
        accuracy.decompose_bias_variance(np.zeros((2, 3)), np.zeros((3, 2)))


# ---------------------------------------------------------------------------
# 5. dataaccess: nasa_cmr, openstreetmap, copernicus_odata, usgs_m2m, planet_data_api
# ---------------------------------------------------------------------------


def test_nasa_cmr_redirect_auth_find_granules_and_versions(monkeypatch):
    """Check strict HTTPS host auth forwarding, +start_date, time-of-day, and versions."""
    if nasa_cmr.HAVE_REQUESTS:
        sess = nasa_cmr.SessionWithHeaderNASARedirection("u", "p")
    else:
        with pytest.raises(rsgispy.RSGISPyException, match="requests"):
            nasa_cmr.SessionWithHeaderNASARedirection("u", "p")
        sess = types.SimpleNamespace(
            AUTH_HOST=nasa_cmr.SessionWithHeaderNASARedirection.AUTH_HOST
        )
        sess.rebuild_auth = (
            nasa_cmr.SessionWithHeaderNASARedirection.rebuild_auth.__get__(
                sess, type(sess)
            )
        )

    orig_req = types.SimpleNamespace(url="https://urs.earthdata.nasa.gov/orig")
    resp = types.SimpleNamespace(request=orig_req)

    req_ok = types.SimpleNamespace(
        url="https://urs.earthdata.nasa.gov/next",
        headers={"Authorization": "Basic abc"},
    )
    sess.rebuild_auth(req_ok, resp)
    assert "Authorization" in req_ok.headers

    for bad_url in (
        "http://urs.earthdata.nasa.gov/downgrade",
        "https://evil.example.com/steal",
        "https://urs.earthdata.nasa.gov.evil.com/steal",
    ):
        req_bad = types.SimpleNamespace(
            url=bad_url, headers={"Authorization": "Basic abc"}
        )
        sess.rebuild_auth(req_bad, resp)
        assert "Authorization" not in req_bad.headers

    captured_params = {}

    def _fake_send(url, data=None, **_kwargs):
        if "collections" in url:
            return {
                "feed": {
                    "entry": [
                        {"version_id": "6.1"},
                        {"version_id": "6.10"},
                        {"version_id": "6.2"},
                    ]
                }
            }
        captured_params.update(data or {})
        return {"feed": {"entry": []}}

    monkeypatch.setattr(nasa_cmr, "HAVE_REQUESTS", True)
    monkeypatch.setattr(rsgispy.tools.http, "send_http_json_request", _fake_send)

    nasa_cmr.find_granules(
        "MOD09GA",
        "6.10",
        start_date=datetime.datetime(2024, 6, 1, 8, 15, 30),
        end_date=datetime.datetime(2024, 6, 2, 16, 45, 5),
        sort_desc=False,
    )
    assert captured_params["sort_key"] == "+start_date"
    assert (
        captured_params["temporal"]
        == "2024-06-01T08:15:30Z,2024-06-02T16:45:05Z"
    )

    assert nasa_cmr.get_max_prod_version("MOD09GA") == "6.10"

    v_entries = [
        {"version_id": "V001"},
        {"version_id": "V010"},
        {"version_id": "V002"},
    ]
    monkeypatch.setattr(
        rsgispy.tools.http,
        "send_http_json_request",
        lambda *_a, **_k: {"feed": {"entry": v_entries}},
    )
    assert nasa_cmr.get_max_prod_version("PROD") == "V010"


def test_openstreetmap_preserves_all_tags_and_optional_regex(monkeypatch):
    """get_osm_gdf preserves all columns by default and supports col_filter_regex."""
    poly = shapely.geometry.box(-4.09, 52.41, -4.08, 52.42)
    sample_gdf = gpd.GeoDataFrame(
        {
            "addr:street": ["High St"],
            "building": ["yes"],
            "name": ["Town Hall"],
            "geometry": [poly],
        },
        crs="EPSG:4326",
    )
    monkeypatch.setattr(openstreetmap, "OSMNX_AVAIL", True)
    monkeypatch.setattr(
        openstreetmap,
        "osmnx",
        types.SimpleNamespace(
            features=types.SimpleNamespace(
                features_from_bbox=lambda bbox, tags: sample_gdf
            )
        ),
        raising=False,
    )

    gdf_all = openstreetmap.get_osm_gdf(
        bbox=(-4.1, -4.07, 52.4, 52.43),
        osm_tags={"building": True},
        rsgis_geom_type=rsgispy.GEOM_POLY,
    )
    assert list(gdf_all.columns) == ["addr:street", "building", "name", "geometry"]

    gdf_filtered = openstreetmap.get_osm_gdf(
        bbox=(-4.1, -4.07, 52.4, 52.43),
        osm_tags={"building": True},
        rsgis_geom_type=rsgispy.GEOM_POLY,
        col_filter_regex="addr:|geometry",
    )
    assert list(gdf_filtered.columns) == ["addr:street", "geometry"]


def test_copernicus_odata_escaping_and_download_hardening(tmp_path, monkeypatch):
    """Escape OData quotes, sanitize filenames, compare MD5 case-insensitively, clean up."""
    captured_urls = []

    class _FakeResp:
        def __init__(self, json_data=None, content=b""):
            self.status_code = 200
            self._json = json_data
            self._content = content
            self.exited = False

        def raise_for_status(self):
            pass

        def json(self):
            return self._json

        def iter_content(self, chunk_size=8192):
            yield self._content

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.exited = True
            return False

    def _fake_get(url, timeout=60):
        captured_urls.append(url)
        return _FakeResp(json_data={"value": [{"Name": "O'Scene"}]})

    if not http_tools.HAVE_REQUESTS:
        monkeypatch.setattr(http_tools, "HAVE_REQUESTS", True)
        monkeypatch.setattr(
            http_tools,
            "requests",
            types.SimpleNamespace(HTTPError=OSError),
            raising=False,
        )
    monkeypatch.setattr(copernicus_odata, "HAVE_REQUESTS", True)
    monkeypatch.setattr(
        copernicus_odata,
        "requests",
        types.SimpleNamespace(get=_fake_get),
        raising=False,
    )
    scn = copernicus_odata.query_scn("S2_O'Brien.SAFE")
    assert scn == {"Name": "O'Scene"}
    assert "Name eq 'S2_O''Brien.SAFE'" in captured_urls[0]

    payload = b"copernicus-scene-bytes"
    upper_md5 = hashlib.md5(payload).hexdigest().upper()

    lifecycle = {"ses_entered": False, "ses_exited": False, "resps": []}

    class _FakeSession:
        def __init__(self, body):
            self.headers = {}
            self._body = body

        def __enter__(self):
            lifecycle["ses_entered"] = True
            return self

        def __exit__(self, *_args):
            lifecycle["ses_exited"] = True
            return False

        def get(self, _url, **_kwargs):
            resp = _FakeResp(content=self._body)
            lifecycle["resps"].append(resp)
            return resp

    out_dir = tmp_path / "nested" / "downloads"
    scn_info = {
        "Name": "../../evil/SAFE_SCENE",
        "Locations": [
            {
                "FormatType": "Extracted",
                "DownloadLink": "https://catalogue.dataspace.copernicus.eu/d/1",
                "ContentLength": len(payload),
                "Checksum": [{"Algorithm": "MD5", "Value": upper_md5}],
            }
        ],
    }
    monkeypatch.setattr(
        copernicus_odata,
        "requests",
        types.SimpleNamespace(
            get=_fake_get, Session=lambda: _FakeSession(payload)
        ),
        raising=False,
    )
    copernicus_odata.download_scn("tok", scn_info, str(out_dir), use_pbar=False)
    assert lifecycle["ses_entered"] is True
    assert lifecycle["ses_exited"] is True
    assert len(lifecycle["resps"]) == 1
    assert lifecycle["resps"][0].exited is True
    assert (out_dir / "SAFE_SCENE.zip").read_bytes() == payload
    assert not (tmp_path / "SAFE_SCENE.zip").exists()

    bad_out_dir = tmp_path / "bad_dl"
    scn_bad = {
        "Name": "CORRUPT_SCENE",
        "Locations": [
            {
                "FormatType": "Extracted",
                "DownloadLink": "https://download.dataspace.copernicus.eu/d/2",
                "ContentLength": len(payload),
                "Checksum": [{"Algorithm": "MD5", "Value": "0" * 32}],
            }
        ],
    }
    with pytest.raises(rsgispy.RSGISPyException, match="checksum"):
        copernicus_odata.download_scn("tok", scn_bad, str(bad_out_dir), use_pbar=False)
    assert not (bad_out_dir / "CORRUPT_SCENE.incomplete").exists()
    assert not (bad_out_dir / "CORRUPT_SCENE.zip").exists()


def test_usgs_m2m_login_validation_pagination_and_secondary_downloads(monkeypatch):
    """Guard empty creds, validate dataset once across pages, check secondaryDownloads."""
    monkeypatch.delenv("RSGIS_USGS_USER", raising=False)
    monkeypatch.delenv("RSGIS_USGS_PASS", raising=False)
    with pytest.raises(rsgispy.RSGISPyException, match="username"):
        usgs_m2m.usgs_login(None, "pw")
    with pytest.raises(rsgispy.RSGISPyException, match="username"):
        usgs_m2m.usgs_login("   ", "pw")
    with pytest.raises(rsgispy.RSGISPyException, match="password"):
        usgs_m2m.usgs_login("user", "")

    calls = {"dataset-categories": 0, "scene-search": 0}

    def _fake_usgs(url, data=None, api_key=None):
        if url.endswith("dataset-categories"):
            calls["dataset-categories"] += 1
            return {
                "errorCode": None,
                "data": [
                    {
                        "categoryName": "Custom",
                        "datasets": [{"datasetAlias": "custom_ds"}],
                    }
                ],
            }
        if url.endswith("scene-search"):
            calls["scene-search"] += 1
            start_n = data.get("startingNumber", 1)
            if start_n == 1:
                return {
                    "errorCode": None,
                    "data": {
                        "totalHits": 150,
                        "startingNumber": 1,
                        "nextRecord": 101,
                        "recordsReturned": 100,
                        "results": [{"entityId": f"E{i}"} for i in range(100)],
                    },
                }
            return {
                "errorCode": None,
                "data": {
                    "totalHits": 150,
                    "startingNumber": 101,
                    "nextRecord": None,
                    "recordsReturned": 50,
                    "results": [{"entityId": f"E{i}"} for i in range(100, 150)],
                },
            }
        if url.endswith("permissions"):
            return {"errorCode": None, "data": ["download"]}
        if url.endswith("download-options"):
            return {
                "errorCode": None,
                "data": [
                    {
                        "entityId": "E1",
                        "id": "BUNDLE_1",
                        "bulkAvailable": False,
                        "secondaryDownloads": [
                            {
                                "entityId": "E1",
                                "id": "BAND_1",
                                "bulkAvailable": True,
                            },
                            {
                                "entityId": "E1",
                                "id": "BAND_2",
                                "bulkAvailable": False,
                            },
                        ],
                    }
                ],
            }
        if url.endswith("scene-list-remove"):
            return {"errorCode": None, "data": None}
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr(rsgispy.tools.http, "send_http_json_request", _fake_usgs)

    scns = usgs_m2m.get_all_usgs_search("custom_ds", "api-key", max_n_rslts=200)
    assert len(scns) == 150
    assert calls["scene-search"] == 2
    assert calls["dataset-categories"] == 1

    opts = usgs_m2m.check_dwnld_opts(
        "api-key", "lst", "custom_ds", dwnld_filetype="all", rm_lst=True
    )
    assert opts == [{"entityId": "E1", "productId": "BAND_1"}]


def test_planet_data_api_wait_bounds_timeout_and_running_loop(tmp_path, monkeypatch):
    """Test max_attempts forwarding, wait_timeout timeout, and running-loop error."""
    wait_calls = []
    validated = []

    class _FakeStateBar:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def update(self, **_kwargs):
            pass

        def update_state(self, *_args, **_kwargs):
            pass

    class _FakeDataClient:
        def __init__(self, delay=0.0):
            self._delay = delay

        async def get_asset(self, item_type, item_id, asset_type):
            return {"id": item_id, "type": asset_type, "item_type": item_type}

        async def activate_asset(self, _asset):
            return None

        async def wait_asset(self, asset, callback=None, max_attempts=None):
            wait_calls.append({"max_attempts": max_attempts, "callback": callback})
            if self._delay > 0:
                await asyncio.sleep(self._delay)
            return asset

        async def download_asset(self, asset, directory, overwrite=False, progress_bar=True):
            return os.path.join(directory, f"{asset['id']}.tif")

        def validate_checksum(self, asset, path):
            validated.append((asset["id"], path))

    def _make_fake_planet(delay=0.0):
        class _FakeSession:
            def __init__(self, auth=None):
                self.auth = auth

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return False

            def client(self, _name):
                return _FakeDataClient(delay=delay)

        return types.SimpleNamespace(
            Session=_FakeSession,
            reporting=types.SimpleNamespace(StateBar=_FakeStateBar),
        )

    monkeypatch.setattr(planet_data_api, "PLANET_AVAIL", True)
    monkeypatch.setattr(
        planet_data_api, "planet", _make_fake_planet(delay=0.0), raising=False
    )

    planet_data_api.run_download_and_validate_item(
        auth_obj="auth",
        item_type=planet_data_api.RSGIS_PLANET_ITEM_PSScene,
        item_id="ITEM_1",
        asset_type=planet_data_api.RSGIS_PLANET_ASSET_ortho_visual,
        out_file_path=str(tmp_path),
        max_attempts=25,
        wait_timeout=5.0,
    )
    assert wait_calls[-1]["max_attempts"] == 25
    assert validated[-1][0] == "ITEM_1"

    with pytest.raises(rsgispy.RSGISPyException, match="max_attempts"):
        planet_data_api.run_download_and_validate_item(
            "auth",
            planet_data_api.RSGIS_PLANET_ITEM_PSScene,
            "ITEM_1",
            planet_data_api.RSGIS_PLANET_ASSET_ortho_visual,
            str(tmp_path),
            max_attempts=0,
        )
    with pytest.raises(rsgispy.RSGISPyException, match="wait_timeout"):
        planet_data_api.run_download_and_validate_item(
            "auth",
            planet_data_api.RSGIS_PLANET_ITEM_PSScene,
            "ITEM_1",
            planet_data_api.RSGIS_PLANET_ASSET_ortho_visual,
            str(tmp_path),
            wait_timeout=0.0,
        )

    monkeypatch.setattr(
        planet_data_api, "planet", _make_fake_planet(delay=0.2), raising=False
    )
    with pytest.raises(rsgispy.RSGISPyException, match="Timed out"):
        planet_data_api.run_download_and_validate_item(
            "auth",
            planet_data_api.RSGIS_PLANET_ITEM_PSScene,
            "ITEM_SLOW",
            planet_data_api.RSGIS_PLANET_ASSET_ortho_visual,
            str(tmp_path),
            wait_timeout=0.02,
        )

    async def _call_inside_loop():
        with pytest.raises(rsgispy.RSGISPyException, match="already running"):
            planet_data_api.run_download_and_validate_item(
                "auth",
                planet_data_api.RSGIS_PLANET_ITEM_PSScene,
                "ITEM_LOOP",
                planet_data_api.RSGIS_PLANET_ASSET_ortho_visual,
                str(tmp_path),
            )

    asyncio.run(_call_inside_loop())


# ---------------------------------------------------------------------------
# 6. droneutils/__init__.py
# ---------------------------------------------------------------------------


def test_droneutils_pil_guard_exif_subifd_and_flightline_edge_cases(
    tmp_path, monkeypatch
):
    """Test PIL guard, IFD.Exif + null-byte stripping, GPS refs, and missing EXIF fields."""
    monkeypatch.setattr(droneutils, "HAVE_PIL", False)
    with pytest.raises(rsgispy.RSGISPyException, match="pillow"):
        droneutils.split_photos_by_time([], str(tmp_path / "out"))
    monkeypatch.setattr(droneutils, "HAVE_PIL", True)

    src_dir = tmp_path / "photos"
    out_dir = tmp_path / "grouped"
    src_dir.mkdir()

    def _save_with_exif_subifd(path, dt_str):
        img = Image.new("RGB", (8, 8), color=(0, 100, 200))
        exif = img.getexif()
        exif_subifd = exif.get_ifd(0x8769)
        exif_subifd[0x9003] = dt_str
        img.save(path, format="JPEG", exif=exif.tobytes())

    p1 = src_dir / "p1.jpg"
    p2 = src_dir / "p2.jpg"
    _save_with_exif_subifd(p1, "2024:07:10 12:00:00\x00")
    _save_with_exif_subifd(p2, "2024:07:10 12:00:20\x00")

    droneutils.split_photos_by_time(
        [str(p1), str(p2)],
        str(out_dir),
        time_split_secs=60,
        date_time_tag="DateTimeOriginal",
        use_pbar=False,
    )
    assert sorted(p.name for p in (out_dir / "20240710_120000").iterdir()) == [
        "p1.jpg",
        "p2.jpg",
    ]

    img_sw = str(src_dir / "sw.jpg")
    img_neg = str(src_dir / "neg.jpg")
    img_nogps = str(src_dir / "nogps.jpg")
    for fp in (img_sw, img_neg, img_nogps):
        Image.new("RGB", (8, 8)).save(fp, format="JPEG")

    meta = {
        img_sw: {
            "EXIF:DateTimeOriginal": "2024:07:10 12:00:00",
            "EXIF:GPSLatitude": 33.8688,
            "EXIF:GPSLatitudeRef": "S",
            "EXIF:GPSLongitude": 151.2093,
            "EXIF:GPSLongitudeRef": "W",
        },
        img_neg: {
            "Composite:GPSLatitude": -34.0,
            "Composite:GPSLatitudeRef": "South",
            "Composite:GPSLongitude": -151.5,
            "Composite:GPSLongitudeRef": "West",
        },
        img_nogps: {
            "EXIF:DateTimeOriginal": "2024:07:10 12:00:01",
        },
    }

    class _FakeExifToolHelper:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def get_metadata(self, path):
            return [meta[path]]

    monkeypatch.setattr(droneutils, "HAVE_EXIFTOOL", True)
    monkeypatch.setattr(
        droneutils,
        "exiftool",
        types.SimpleNamespace(ExifToolHelper=_FakeExifToolHelper),
        raising=False,
    )
    monkeypatch.setattr(
        rsgispy.tools.files, "sort_file_by_datetime", lambda imgs: list(imgs)
    )

    pts_gdf, lines_gdf = droneutils.create_flightline_vec(
        [img_neg, img_nogps, img_sw]
    )
    assert len(pts_gdf) == 2
    assert len(lines_gdf) == 1
    assert list(pts_gdf["image_filename"]) == ["sw.jpg", "neg.jpg"]
    np.testing.assert_allclose(pts_gdf["lat"].to_numpy(), [-33.8688, -34.0])
    np.testing.assert_allclose(pts_gdf["lon"].to_numpy(), [-151.2093, -151.5])
    assert pts_gdf["sensor_index"].iloc[0] is None
    assert pts_gdf["sensor_index"].iloc[1] is None

    ext_df = pd.DataFrame(
        [
            {
                "image": "sw.jpg",
                "lat": 33.8688,
                "lon": 151.2093,
                "GPSLatitudeRef": "S",
                "GPSLongitudeRef": "W",
                "alt": 80.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 90.0,
            },
            {
                "image": "neg.jpg",
                "lat": -34.0,
                "lon": -151.5,
                "GPSLatitudeRef": "S",
                "GPSLongitudeRef": "W",
                "alt": np.nan,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 95.0,
            },
        ]
    )
    ext_pts, ext_lines = droneutils.create_flightline_extern_gps_vec(
        ext_df, str(src_dir)
    )
    assert len(ext_pts) == 2
    assert len(ext_lines) == 1
    np.testing.assert_allclose(ext_pts["lat"].to_numpy(), [-33.8688, -34.0])
    np.testing.assert_allclose(ext_pts["lon"].to_numpy(), [-151.2093, -151.5])
    assert ext_pts["sensor_index"].iloc[0] is None
