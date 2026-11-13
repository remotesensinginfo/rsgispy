"""Tests for the newly ported rsgispy.tools modules."""

import io
import os
import sys
import types

import numpy
import pytest
import rasterio
from PIL import Image
import pyproj

import rsgispy
import rsgispy.tools.ftptools as ftptools
import rsgispy.tools.httptools as httptools
import rsgispy.tools.imagetools as imagetools
import rsgispy.tools.maths as maths
import rsgispy.tools.projection as projection
import rsgispy.tools.tilecacheutils as tilecacheutils
import rsgispy.tools.utm as utm
import rsgispy.tools.webtools as webtools
import rsgispy.tools.wmts_tools as wmts_tools


def test_maths_round_to_nearest_val():
    assert maths.round_to_nearest_val(12.3, 5, rsgispy.ROUND_NEAREST) == 10.0
    assert maths.round_to_nearest_val(12.6, 5, rsgispy.ROUND_NEAREST) == 15.0
    assert maths.round_to_nearest_val(10.1, 5, rsgispy.ROUND_UP) == 15.0
    assert maths.round_to_nearest_val(14.9, 5, rsgispy.ROUND_DOWN) == 10.0

    arr = numpy.array([1.2, 4.8, 7.5])
    out = maths.round_to_nearest_val(arr, 2.0, rsgispy.ROUND_UP)
    assert numpy.allclose(out, [2.0, 6.0, 8.0])

    with pytest.raises(rsgispy.RSGISPyException):
        maths.round_to_nearest_val(1.0, 1.0, 999)


def test_projection_utilities():
    crs = projection.get_crs_obj(4326)
    assert isinstance(crs, pyproj.CRS)
    wkt = projection.get_wkt_from_epsg_code(4326)
    assert isinstance(wkt, str) and len(wkt) > 0
    epsg = projection.get_epsg_code_from_wkt(wkt)
    assert epsg == 4326

    dx_m, dy_m = projection.degrees_to_metres(52.4, 0.001, 0.001)
    dlon, dlat = projection.metres_to_degrees(52.4, dx_m, dy_m)
    assert numpy.isclose(dlon, 0.001, rtol=1e-4)
    assert numpy.isclose(dlat, 0.001, rtol=1e-4)

    coord_str = projection.get_deg_coord_as_str(52.415, -4.082, n_chars=4)
    assert isinstance(coord_str, str)
    assert "n" in coord_str and "w" in coord_str

    dist = projection.great_circle_distance((-4.08, 52.41), (-4.08, 53.41))
    assert 110000 < dist < 112500


def test_utm_conversions_and_zones():
    assert utm.epsg_for_utm(30, "N") == 32630
    assert utm.epsg_for_utm(30, "S") == 32730
    assert utm.utm_from_epsg(32630) == (30, "N")
    assert utm.utm_from_epsg(32730) == (30, "S")

    easting, northing, zone, letter = utm.from_latlon(52.415, -4.082)
    assert zone == 30
    assert letter == "U"
    lat, lon = utm.to_latlon(easting, northing, zone, letter)
    assert numpy.isclose(lat, 52.415, atol=1e-5)
    assert numpy.isclose(lon, -4.082, atol=1e-5)

    assert utm.latlon_to_zone_number(52.415, -4.082) == 30
    assert utm.latitude_to_zone_letter(52.415) == "U"
    assert utm.zone_number_to_central_longitude(30) == -3.0
    assert utm.check_valid_zone(30, "U")

    min_e, max_e = utm.min_max_eastings_for_lat(52.415, zone=30)
    assert min_e < max_e

    lats = numpy.array([52.4, 52.5, 52.45])
    lons = numpy.array([-4.0, -4.1, -3.9])
    zones = utm.latlon_arr_to_utm_zone_number(lats, lons)
    assert numpy.all(zones == 30)
    assert utm.latlon_to_mode_utm_zone_number(lats, lons) == 30

    splits = utm.split_wgs84_bbox_utm_zones([-7.0, 1.0, 51.0, 53.0])
    assert len(splits) >= 2
    assert splits[0][0] == 29


def test_tilecacheutils_roundtrips(tmp_path):
    res0 = tilecacheutils.resolution(0)
    res1 = tilecacheutils.resolution(1)
    assert numpy.isclose(res0 / 2.0, res1)

    assert tilecacheutils.truncate_lng_lat(190.0, 95.0) == (180.0, 90.0)

    tx, ty, tz = tilecacheutils.get_tile_for_point(-4.08, 52.41, 10, tms=False)
    assert tz == 10
    bbox = tilecacheutils.get_tile_bbox(tx, ty, tz, tms=False)
    assert bbox[0] <= -4.08 <= bbox[1]
    assert bbox[2] <= 52.41 <= bbox[3]

    tl_lon, tl_lat = tilecacheutils.get_tile_tl_lonlat(tx, ty, tz, tms=False)
    assert numpy.isclose(tl_lon, bbox[0])
    assert numpy.isclose(tl_lat, bbox[3])

    fx, fy, fz = tilecacheutils.flip_xyz_tms_tiles(tx, ty, tz)
    bx, by, bz = tilecacheutils.flip_xyz_tms_tiles(fx, fy, fz)
    assert (bx, by, bz) == (tx, ty, tz)

    mx, my = tilecacheutils.get_web_mercator_xy(-4.08, 52.41)
    rlon, rlat = tilecacheutils.get_xy_mercator_lonlat(mx, my)
    assert numpy.isclose(rlon, -4.08, atol=1e-5)
    assert numpy.isclose(rlat, 52.41, atol=1e-5)

    bounds = tilecacheutils.xy_mercator_bounds(tx, ty, tz, tms=False)
    assert bounds[0] < bounds[1] and bounds[2] < bounds[3]

    qk = tilecacheutils.get_quadkey_for_tile_path(tx, ty, tz, tms=False)
    assert tilecacheutils.get_tile_path_from_quadkey(qk, tms=False) == (tx, ty, tz)

    tiles = tilecacheutils.get_tiles([-4.1, -4.05, 52.4, 52.43], [10], tms=False)
    assert 10 in tiles and len(tiles[10]) >= 1

    # Test convert_between_tms_xyz renaming
    z_dir = tmp_path / "2" / "1"
    z_dir.mkdir(parents=True)
    tile_file = z_dir / "0.png"
    tile_file.write_bytes(b"png")
    tilecacheutils.convert_between_tms_xyz(str(tmp_path))
    assert (z_dir / "3.png").exists()


def test_imagetools_gif_qr_exif_and_pdf(tmp_path, monkeypatch):
    # GIF animation
    im1_path = tmp_path / "im1.png"
    im2_path = tmp_path / "im2.png"
    im1 = Image.new("RGB", (32, 32), color=(255, 0, 0))
    im2 = Image.new("RGB", (32, 32), color=(0, 255, 0))
    exif = im1.getexif()
    exif[0x010F] = "TestCam"
    im1.save(im1_path, exif=exif.tobytes())
    im2.save(im2_path)

    gif_path = tmp_path / "anim.gif"
    imagetools.animate_img_set(
        [str(im1_path), str(im2_path)],
        str(gif_path),
        fps=2,
        resize=True,
        out_x_size=16,
        tmp_dir=str(tmp_path / "gif_tmp"),
    )
    assert gif_path.exists()

    # PIL EXIF listing
    imagetools.list_exif_info_pil(str(im1_path))

    # QR code
    qr_path = tmp_path / "qr.png"
    imagetools.create_qr_code(str(qr_path), "https://example.com")
    assert qr_path.exists()

    # PDF blank page + image extraction via PyMuPDF
    try:
        import pymupdf as fitz_mod
    except ImportError:
        import fitz as fitz_mod

    pdf_path = tmp_path / "sample.pdf"
    doc = fitz_mod.open()
    page = doc.new_page(width=200, height=200)
    page.insert_image(fitz_mod.Rect(10, 10, 100, 100), filename=str(im1_path))
    doc.save(str(pdf_path))
    doc.close()

    blank_pdf = tmp_path / "with_blanks.pdf"
    imagetools.add_pdf_blank_pages(str(pdf_path), str(blank_pdf))
    assert blank_pdf.exists()
    with fitz_mod.open(str(blank_pdf)) as out_doc:
        assert len(out_doc) == 2

    extract_dir = tmp_path / "extracted"
    extract_dir.mkdir()
    imagetools.extract_images_from_pdf(str(pdf_path), str(extract_dir))
    assert len(list(extract_dir.iterdir())) >= 1

    # ExifTool wrapper test (mocked)
    class _FakeExifToolHelper:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def get_metadata(self, path):
            return [{"EXIF:Make": "TestCam", "File:FileName": os.path.basename(path)}]

    fake_mod = types.ModuleType("exiftool")
    fake_mod.ExifToolHelper = _FakeExifToolHelper
    monkeypatch.setitem(sys.modules, "exiftool", fake_mod)
    exif_dict = imagetools.get_exif_info(str(im1_path))
    assert exif_dict["EXIF:Make"] == "TestCam"
    imagetools.list_exif_info(str(im1_path))


def test_webtools_httptools_and_ftptools(tmp_path, monkeypatch):
    html = (
        '<html><body>'
        '<img src="a.jpg"><img src="/sub/b.png">'
        '<a href="doc1.pdf">doc1</a><a href="http://other.org/doc2.pdf">doc2</a>'
        '</body></html>'
    )

    import json as _json

    class _SimpleResp:
        def __init__(self, text="", status_code=200, json_data=None, content=b""):
            self.text = _json.dumps(json_data) if (json_data is not None and not text) else text
            self.status_code = status_code
            self.reason = "OK"
            self.headers = {}
            self._json = json_data
            self._content = content

        def raise_for_status(self):
            if self.status_code >= 400:
                raise webtools.requests.HTTPError("err")

        def json(self):
            return self._json

        def iter_content(self, chunk_size=8192):
            yield self._content

        def close(self):
            return None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

    if webtools.HAVE_BS4 and webtools.HAVE_REQUESTS:
        monkeypatch.setattr(
            webtools.requests, "get", lambda url, timeout=30.0: _SimpleResp(text=html)
        )
        imgs = webtools.find_web_imgs("https://example.com/dir/page.html", file_ext="jpg")
        assert imgs == ["https://example.com/dir/a.jpg"]
        pdfs = webtools.find_web_files("https://example.com/dir/page.html", file_ext="pdf")
        assert "https://example.com/dir/doc1.pdf" in pdfs
        assert "http://other.org/doc2.pdf" in pdfs

    # httptools
    http_db = str(tmp_path / "http_db.json")
    httptools.create_file_listings_db(
        http_db, {"f1.txt": "https://example.com/f1.txt"}
    )

    if httptools.HAVE_REQUESTS:
        monkeypatch.setattr(
            httptools.requests,
            "head",
            lambda url, timeout=30.0: _SimpleResp(status_code=200),
        )
        assert httptools.check_url_exists("https://example.com")

        monkeypatch.setattr(
            httptools.requests,
            "post",
            lambda url, data=None, headers=None, timeout=60.0: _SimpleResp(
                json_data={"ok": True}
            ),
        )
        assert httptools.send_http_json_request("https://example.com/api") == {"ok": True}

        class _FakeHTTPSession:
            def __init__(self):
                self.headers = {}
                self.auth = None

            def get(self, url, **kwargs):
                return _SimpleResp(content=b"abc123")

        monkeypatch.setattr(httptools.requests, "Session", _FakeHTTPSession)
        out_http_dir = tmp_path / "http_dl"
        out_http_dir.mkdir()
        httptools.download_http_files_use_lst_db(http_db, str(out_http_dir))
        assert (out_http_dir / "f1.txt").read_bytes() == b"abc123"

    # ftptools
    class _FakeFTP:
        def __init__(self, url, timeout=300):
            self.cwd_path = "/"

        def login(self, user="", passwd=""):
            return None

        def mlsd(self, path, facts=None):
            if path == "/root":
                return [
                    ("sub", {"type": "dir"}),
                    ("file1.txt", {"type": "file"}),
                ]
            return [("file2.tif", {"type": "file"})]

        def cwd(self, path):
            self.cwd_path = path

        def retrlines(self, cmd, callback):
            callback("drwxr-xr-x 1 user group 0 Jan 1 00:00 dest")

        def mkd(self, dirname):
            return dirname

        def retrbinary(self, cmd, callback):
            callback(b"ftp-data")

        def storbinary(self, cmd, fp):
            fp.read()

        def quit(self):
            return None

    monkeypatch.setattr(ftptools.ftplib, "FTP", _FakeFTP)
    listings, file_paths = ftptools.get_ftp_file_listings("ftp.example.com", "/root")
    assert listings["/root"] == ["/root/file1.txt"]
    assert "/root/sub/file2.tif" in file_paths

    ftp_db = str(tmp_path / "ftp_db.json")
    ftptools.create_file_listings_db(ftp_db, "ftp.example.com", "/root")
    filtered_db = str(tmp_path / "ftp_filtered.json")
    ftptools.filter_file_listings_db(ftp_db, [".tif"], filtered_db)

    ftp_out = tmp_path / "ftp_dl"
    ftp_out.mkdir()
    ftptools.download_files_use_lst_db(filtered_db, str(ftp_out))
    assert (ftp_out / "file2.tif").read_bytes() == b"ftp-data"

    up_file = tmp_path / "upload.txt"
    up_file.write_bytes(b"up")
    assert ftptools.upload_ftp_file(str(up_file), "ftp.example.com", "/dest")


def test_wmts_tools_layer_list_and_render(tmp_path, monkeypatch):
    # Create a fake 256x256 RGB PNG tile in memory
    tile_buf = io.BytesIO()
    Image.new("RGB", (256, 256), color=(12, 34, 56)).save(tile_buf, format="PNG")
    tile_png = tile_buf.getvalue()

    class _FakeTileMatrix:
        def __init__(self, ident, scale_denom, top_left, tw=256, th=256, nw=4, nh=4):
            self.identifier = ident
            self.scaledenominator = scale_denom
            self.topleftcorner = top_left
            self.tilewidth = tw
            self.tileheight = th
            self.matrixwidth = nw
            self.matrixheight = nh

    class _FakeTMS:
        def __init__(self):
            self.crs = "EPSG:3857"
            self.tilematrix = {
                "0": _FakeTileMatrix("0", 559082264.0287178, (-20037508.34, 20037508.34))
            }

    class _FakeLayer:
        def __init__(self):
            self.tilematrixsets = ["GoogleMapsCompatible"]
            self.styles = {"default": {}}

    class _FakeWMTS:
        def __init__(self, url):
            self.contents = {"layer_a": _FakeLayer(), "layer_b": _FakeLayer()}
            self.tilematrixsets = {"GoogleMapsCompatible": _FakeTMS()}

        def gettile(self, **kwargs):
            return io.BytesIO(tile_png)

    fake_owslib = types.ModuleType("owslib")
    fake_owslib_wmts = types.ModuleType("owslib.wmts")
    fake_owslib_wmts.WebMapTileService = _FakeWMTS
    fake_owslib.wmts = fake_owslib_wmts
    monkeypatch.setitem(sys.modules, "owslib", fake_owslib)
    monkeypatch.setitem(sys.modules, "owslib.wmts", fake_owslib_wmts)

    lyrs = wmts_tools.get_wmts_layer_list("https://example.com/wmts", name_filter="_a")
    assert lyrs == ["layer_a"]

    out_tif = tmp_path / "wmts_out.tif"
    wmts_tools.get_wmts_as_img(
        wmts_url="https://example.com/wmts",
        wmts_lyr="layer_a",
        bbox=(-500000.0, 500000.0, 5000000.0, 6000000.0),
        bbox_epsg=3857,
        output_img=str(out_tif),
        gdalformat="GTIFF",
        zoom_level="0",
        tmp_dir=str(tmp_path / "wmts_tmp"),
    )
    assert out_tif.exists()
    with rasterio.open(out_tif) as src:
        assert src.count == 3
        data = src.read()
        assert data.shape[1] > 0 and data.shape[2] > 0


def test_optional_requests_missing_tools(tmp_path):
    """Verify webtools and httptools prevent executing optional-dep functions when missing."""
    import importlib
    from unittest.mock import patch
    import rsgispy.tools as rtools

    orig_webtools = webtools
    orig_httptools = httptools
    blocked_req = {"requests": None, "requests.exceptions": None}
    blocked_bs4 = {"bs4": None}
    try:
        with patch.dict(sys.modules, blocked_bs4):
            sys.modules.pop("rsgispy.tools.webtools", None)
            web_mod_no_bs4 = importlib.import_module("rsgispy.tools.webtools")

        assert not web_mod_no_bs4.HAVE_BS4
        for fn_name in ("find_web_imgs", "find_web_files"):
            assert not hasattr(web_mod_no_bs4, fn_name)
            assert fn_name not in web_mod_no_bs4.__all__

        with patch.dict(sys.modules, blocked_req):
            sys.modules.pop("rsgispy.tools.webtools", None)
            sys.modules.pop("rsgispy.tools.httptools", None)
            web_mod = importlib.import_module("rsgispy.tools.webtools")
            http_mod = importlib.import_module("rsgispy.tools.httptools")

        assert not web_mod.HAVE_REQUESTS
        for fn_name in ("find_web_imgs", "find_web_files"):
            assert not hasattr(web_mod, fn_name)
            assert fn_name not in web_mod.__all__

        assert not http_mod.HAVE_REQUESTS
        for fn_name in (
            "check_url_exists",
            "check_http_response",
            "send_http_json_request",
            "download_file_http",
        ):
            assert not hasattr(http_mod, fn_name)
            assert fn_name not in http_mod.__all__

        for sym_name in (
            "RSGISPyResponseException",
            "wget_download_file",
            "create_file_listings_db",
            "download_http_files_use_lst_db",
        ):
            assert hasattr(http_mod, sym_name)
            assert sym_name in http_mod.__all__

        db_path = str(tmp_path / "http_db.json")
        http_mod.create_file_listings_db(db_path, {"a.txt": "https://example.com/a.txt"})
        with pytest.raises(rsgispy.RSGISPyException):
            http_mod.download_http_files_use_lst_db(db_path, str(tmp_path), use_wget=False)
    finally:
        sys.modules["rsgispy.tools.webtools"] = orig_webtools
        sys.modules["rsgispy.tools.httptools"] = orig_httptools
        rtools.webtools = orig_webtools
        rtools.httptools = orig_httptools
