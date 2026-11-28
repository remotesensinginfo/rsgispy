"""Tests for rsgispy.dataaccess modules."""

import asyncio
import datetime
import hashlib
import os
import types

import geopandas as gpd
import pytest
import shapely.geometry

import rsgispy
import rsgispy.dataaccess.copernicus_odata as copernicus_odata
import rsgispy.dataaccess.nasa_cmr as nasa_cmr
import rsgispy.dataaccess.openstreetmap as openstreetmap
import rsgispy.dataaccess.planet_data_api as planet_data_api
import rsgispy.dataaccess.usgs_m2m as usgs_m2m
import rsgispy.tools.hashing
import rsgispy.tools.http
import rsgispy.tools.http as http_tools


class _FakeHTTPResponse:
    def __init__(self, json_data=None, content=b"", headers=None):
        self._json_data = json_data
        self._content = content
        self.headers = headers or {}
        self.status_code = 200
        self.reason = "OK"

    def raise_for_status(self):
        return None

    def json(self):
        return self._json_data

    def iter_content(self, chunk_size=8192):
        for i in range(0, len(self._content), chunk_size):
            yield self._content[i:i + chunk_size]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def test_copernicus_odata_mappings_and_queries(tmp_path, monkeypatch):
    assert copernicus_odata.get_sensor_collection_name(
        copernicus_odata.RSGIS_ODATA_SEN2
    ) == "SENTINEL-2"
    with pytest.raises(rsgispy.RSGISPyException):
        copernicus_odata.get_sensor_collection_name(9999)

    if not copernicus_odata.HAVE_REQUESTS:
        return

    monkeypatch.setenv("RSGIS_COP_USER", "user1")
    monkeypatch.setenv("RSGIS_COP_PASS", "pass1")
    monkeypatch.setattr(
        copernicus_odata.requests,
        "post",
        lambda url, data, timeout=60: _FakeHTTPResponse({"access_token": "tok123"}),
    )
    assert copernicus_odata.get_access_token() == "tok123"

    captured_urls = []

    def _fake_get(url, timeout=60):
        captured_urls.append(url)
        return _FakeHTTPResponse(
            {
                "value": [
                    {
                        "Name": "S2_TEST.SAFE",
                        "Locations": [
                            {
                                "FormatType": "Extracted",
                                "DownloadLink": "https://catalogue.example.com/scn.zip",
                                "ContentLength": 11,
                                "Checksum": [
                                    {
                                        "Algorithm": "MD5",
                                        "Value": (
                                            rsgispy.tools.hashing
                                            .create_md5_hash_str("hello world")
                                            if hasattr(
                                                rsgispy.tools.hashing,
                                                "create_md5_hash_str",
                                            )
                                            else "5eb63bbbe01eeed093cb22bb8f5acdc3"
                                        ),
                                    }
                                ],
                            }
                        ],
                    }
                ]
            }
        )

    monkeypatch.setattr(copernicus_odata.requests, "get", _fake_get)
    scns = copernicus_odata.query_scn_lst(
        sensor=copernicus_odata.RSGIS_ODATA_SEN2,
        bbox=(-4.2, -4.0, 52.3, 52.5),
        start_date=datetime.datetime(2024, 1, 1),
        end_date=datetime.datetime(2024, 2, 1),
        cloud_cover=20.0,
        orbit_dir=copernicus_odata.RSGIS_ODATA_ORBIT_DIR_ASC,
        product_type=copernicus_odata.RSGIS_ODATA_PROD_TYPE_S2_MSI_2A,
        order_by=copernicus_odata.RSGIS_ODATA_ORDERBY_DESC,
        max_n_rslts=10,
        start_n=5,
    )
    assert len(scns) == 1
    assert "SENTINEL-2" in captured_urls[0]
    assert "S2MSI2A" in captured_urls[0]
    assert "skip=5" in captured_urls[0]

    scn = copernicus_odata.query_scn("S2_TEST.SAFE")
    assert scn["Name"] == "S2_TEST.SAFE"

    with pytest.raises(rsgispy.RSGISPyException):
        copernicus_odata.query_scn_lst(
            sensor=copernicus_odata.RSGIS_ODATA_SEN2,
            bbox=(-4.2, -4.0, 52.3, 52.5),
            cloud_cover=150.0,
        )

    class _FakeSession:
        def __init__(self):
            self.headers = {}

        def get(self, url, **kwargs):
            assert "//download." in url
            return _FakeHTTPResponse(content=b"hello world")

    monkeypatch.setattr(copernicus_odata.requests, "Session", _FakeSession)
    copernicus_odata.download_scns("tok123", scns, str(tmp_path), no_except=False)
    out_zip = tmp_path / "S2_TEST.SAFE.zip"
    assert out_zip.exists()
    assert out_zip.read_bytes() == b"hello world"


def test_nasa_cmr_queries_and_download(tmp_path, monkeypatch):
    if not nasa_cmr.HAVE_REQUESTS:
        granules = [
            {
                "id": "G1",
                "producer_granule_id": "granule1.h5",
                "granule_size": "12.5",
                "links": [
                    {
                        "type": "application/x-hdfeos",
                        "href": "https://example.com/granule1.h5",
                    }
                ],
            }
        ]
        assert nasa_cmr.get_total_file_size(granules) == 12.5
        if nasa_cmr.HAVE_PYSONDB:
            db_json = str(tmp_path / "cmr.json")
            assert (
                nasa_cmr.create_cmr_dwnld_db(
                    db_json, granules, "application/x-hdfeos"
                )
                == []
            )
        return

    def _fake_send_json(url, data=None, convert_to_json=False, header_data=None):
        if "collections" in url:
            return {
                "feed": {
                    "entry": [
                        {"version_id": "001"},
                        {"version_id": "002"},
                    ]
                }
            }
        page_num = data["page_num"]
        if page_num == 1:
            return {
                "feed": {
                    "entry": [
                        {
                            "id": "G1",
                            "producer_granule_id": "granule1.h5",
                            "granule_size": "12.5",
                            "links": [
                                {
                                    "type": "application/x-hdfeos",
                                    "href": "https://example.com/granule1.h5",
                                }
                            ],
                        },
                        {
                            "id": "G2",
                            "producer_granule_id": "granule2.h5",
                            "granule_size": "7.5",
                            "links": [],
                        },
                    ]
                }
            }
        return {"feed": {"entry": []}}

    monkeypatch.setattr(
        rsgispy.tools.http, "send_http_json_request", _fake_send_json
    )

    assert nasa_cmr.check_prod_version_avail("GEDI02_A", "002")
    assert not nasa_cmr.check_prod_version_avail("GEDI02_A", "999")
    assert nasa_cmr.get_max_prod_version("GEDI02_A") == "002"

    granules = nasa_cmr.find_all_granules(
        "GEDI02_A",
        "002",
        bbox=[-4.2, -4.0, 52.3, 52.5],
        start_date=datetime.datetime(2024, 1, 1),
        end_date=datetime.datetime(2024, 2, 1),
        max_n_pages=5,
    )
    assert len(granules) == 2
    assert nasa_cmr.get_total_file_size(granules) == 20.0

    if nasa_cmr.HAVE_PYSONDB:
        db_json = str(tmp_path / "cmr.json")
        missing = nasa_cmr.create_cmr_dwnld_db(
            db_json, granules, "application/x-hdfeos"
        )
        assert missing == ["granule2.h5"]

    class _FakeCMRSession(nasa_cmr.SessionWithHeaderNASARedirection):
        def get(self, url, **kwargs):
            return _FakeHTTPResponse(
                content=b"cmr-bytes", headers={"content-length": "9"}
            )

    monkeypatch.setattr(
        nasa_cmr, "SessionWithHeaderNASARedirection", _FakeCMRSession
    )
    out_file = tmp_path / "granule1.h5"
    assert nasa_cmr.cmr_download_file_http(
        "https://example.com/granule1.h5", str(out_file), "u", "p", no_except=False
    )
    assert out_file.read_bytes() == b"cmr-bytes"


def test_openstreetmap(tmp_path, monkeypatch):
    assert openstreetmap.get_osm_geom_type_name(rsgispy.GEOM_POLY) == "Polygon"
    assert openstreetmap.get_osm_geom_type_name(rsgispy.GEOM_PT) == "Point"
    assert openstreetmap.get_osm_geom_type_name(rsgispy.GEOM_LINE) == "LineString"
    with pytest.raises(rsgispy.RSGISPyException):
        openstreetmap.get_osm_geom_type_name(999)

    poly = shapely.geometry.box(-4.09, 52.41, -4.08, 52.42)
    pt = shapely.geometry.Point(-4.085, 52.415)
    sample_gdf = gpd.GeoDataFrame(
        {
            "addr:street": ["High St", "Side Rd"],
            "other": [1, 2],
            "geometry": [poly, pt],
        },
        crs="EPSG:4326",
    )

    monkeypatch.setattr(openstreetmap, "OSMNX_AVAIL", True)
    fake_osmnx = types.SimpleNamespace(
        features=types.SimpleNamespace(
            features_from_bbox=lambda bbox, tags: sample_gdf
        )
    )
    monkeypatch.setattr(openstreetmap, "osmnx", fake_osmnx, raising=False)

    gdf = openstreetmap.get_osm_gdf(
        bbox=(-4.1, -4.07, 52.4, 52.43),
        osm_tags={"building": True},
        rsgis_geom_type=rsgispy.GEOM_POLY,
        clip_to_roi=True,
    )
    assert len(gdf) == 1
    assert list(gdf.columns) == ["addr:street", "other", "geometry"]

    gdf_filtered = openstreetmap.get_osm_gdf(
        bbox=(-4.1, -4.07, 52.4, 52.43),
        osm_tags={"building": True},
        rsgis_geom_type=rsgispy.GEOM_POLY,
        clip_to_roi=True,
        col_filter_regex="addr:|geometry",
    )
    assert list(gdf_filtered.columns) == ["addr:street", "geometry"]

    out_gpkg = str(tmp_path / "osm.gpkg")
    openstreetmap.get_osm_to_file(
        out_gpkg,
        "buildings",
        bbox=(-4.1, -4.07, 52.4, 52.43),
        osm_tags={"building": True},
    )
    loaded = gpd.read_file(out_gpkg, layer="buildings")
    assert len(loaded) == 1

    out_multi = str(tmp_path / "osm_multi.gpkg")
    openstreetmap.get_osm_multi_to_file(
        out_multi,
        "features",
        bbox=(-4.1, -4.07, 52.4, 52.43),
        osm_tags_lst=[{"building": True}, {"amenity": "cafe"}],
    )
    loaded_multi = gpd.read_file(out_multi, layer="features")
    assert len(loaded_multi) == 2

    mem_osm = openstreetmap.get_osm_to_file(
        None,
        None,
        bbox=(-4.1, -4.07, 52.4, 52.43),
        osm_tags={"building": True},
    )
    assert isinstance(mem_osm, gpd.GeoDataFrame)
    assert len(mem_osm) == 1

    mem_osm_multi = openstreetmap.get_osm_multi_to_file(
        None,
        None,
        bbox=(-4.1, -4.07, 52.4, 52.43),
        osm_tags_lst=[{"building": True}, {"amenity": "cafe"}],
    )
    assert isinstance(mem_osm_multi, gpd.GeoDataFrame)
    assert len(mem_osm_multi) == 2


def test_planet_data_api_mappings_and_optional_import(monkeypatch):
    assert (
        planet_data_api.get_item_type_str(planet_data_api.RSGIS_PLANET_ITEM_PSScene)
        == "PSScene"
    )
    assert (
        planet_data_api.get_asset_type_str(
            planet_data_api.RSGIS_PLANET_ASSET_ortho_visual
        )
        == "ortho_visual"
    )
    assert (
        planet_data_api.get_bundle_type_str(
            planet_data_api.RSGIS_PLANET_BUNDLE_analytic_udm2
        )
        == "analytic_udm2"
    )

    with pytest.raises(rsgispy.RSGISPyException):
        planet_data_api.get_item_type_str(999)
    with pytest.raises(rsgispy.RSGISPyException):
        planet_data_api.get_asset_type_str(999)
    with pytest.raises(rsgispy.RSGISPyException):
        planet_data_api.get_bundle_type_str(999)

    monkeypatch.setattr(planet_data_api, "PLANET_AVAIL", False)
    with pytest.raises(rsgispy.RSGISPyException):
        planet_data_api.planet_auth(api_key="test-key")

    # Test with a minimal mocked planet package
    fake_planet = types.SimpleNamespace(
        Auth=types.SimpleNamespace(
            from_key=lambda k: f"auth:{k}",
            from_login=lambda u, p: f"login:{u}",
        )
    )
    monkeypatch.setattr(planet_data_api, "PLANET_AVAIL", True)
    monkeypatch.setattr(planet_data_api, "planet", fake_planet, raising=False)
    assert planet_data_api.planet_auth(api_key="abc") == "auth:abc"


def test_usgs_m2m_workflow(monkeypatch):
    def _fake_usgs_request(url, data=None, api_key=None):
        if url.endswith("login"):
            return {"errorCode": None, "data": "usgs-api-key"}
        if url.endswith("permissions"):
            return {"errorCode": None, "data": ["download", "order"]}
        if url.endswith("grid2ll"):
            if data["responseShape"] == "point":
                return {
                    "errorCode": None,
                    "data": {
                        "shape": "point",
                        "coordinates": [{"longitude": -4.1, "latitude": 52.4}],
                    },
                }
            return {
                "errorCode": None,
                "data": {
                    "shape": "polygon",
                    "coordinates": [
                        {"longitude": -4.2, "latitude": 52.3},
                        {"longitude": -4.0, "latitude": 52.5},
                    ],
                },
            }
        if url.endswith("scene-search"):
            start_n = data.get("startingNumber", 1)
            if start_n == 1:
                results = [
                    {
                        "displayId": f"LC08_{i}",
                        "entityId": f"E{i}",
                        "options": {"download": True, "bulk": i % 2 == 0},
                    }
                    for i in range(100)
                ]
                return {
                    "errorCode": None,
                    "data": {
                        "totalHits": 120,
                        "startingNumber": 1,
                        "nextRecord": 101,
                        "recordsReturned": 100,
                        "results": results,
                    },
                }
            results = [
                {
                    "displayId": f"LC08_{i}",
                    "entityId": f"E{i}",
                    "options": {"download": True, "bulk": True},
                }
                for i in range(100, 120)
            ]
            return {
                "errorCode": None,
                "data": {
                    "totalHits": 120,
                    "startingNumber": 101,
                    "nextRecord": None,
                    "recordsReturned": 20,
                    "results": results,
                },
            }
        if url.endswith("scene-list-add"):
            return {"errorCode": None, "data": len(data["entityIds"])}
        if url.endswith("scene-list-remove"):
            return {"errorCode": None, "data": None}
        if url.endswith("download-options"):
            return {
                "errorCode": None,
                "data": [
                    {
                        "entityId": "E0",
                        "id": "P0",
                        "bulkAvailable": True,
                        "secondaryDownloads": [
                            {"entityId": "E0", "id": "P0_B1", "bulkAvailable": True}
                        ],
                    }
                ],
            }
        if url.endswith("download-request"):
            return {
                "errorCode": None,
                "data": {
                    "availableDownloads": [
                        {"downloadId": 1, "url": "https://example.com/scn.tar"}
                    ],
                    "preparingDownloads": [{"downloadId": 2}],
                },
            }
        return {"errorCode": None, "data": {}}

    monkeypatch.setattr(
        rsgispy.tools.http,
        "send_http_json_request",
        _fake_usgs_request,
        raising=False,
    )

    api_key = usgs_m2m.usgs_login("u", "p")
    assert api_key == "usgs-api-key"
    assert usgs_m2m.can_user_dwnld(api_key)
    assert usgs_m2m.can_user_order(api_key)
    assert usgs_m2m.get_wrs_pt(api_key, 204, 23) == (-4.1, 52.4)
    assert usgs_m2m.get_wrs_bbox(api_key, 204, 23) == (-4.2, -4.0, 52.3, 52.5)

    # Exercises the fixed multi-page pagination in get_all_usgs_search
    all_scns = usgs_m2m.get_all_usgs_search(
        "landsat_ot_c2_l2", api_key, max_n_rslts=110
    )
    assert len(all_scns) == 110

    dsp_ids, ent_ids = usgs_m2m.get_download_ids(all_scns[:4], bulk=True)
    assert dsp_ids == ["LC08_0", "LC08_2"]
    assert ent_ids == ["E0", "E2"]

    added = usgs_m2m.create_scene_list(api_key, "landsat_ot_c2_l2", ent_ids, "my_lst")
    assert added == 2

    opts = usgs_m2m.check_dwnld_opts(
        api_key, "my_lst", "landsat_ot_c2_l2", dwnld_filetype="all"
    )
    assert len(opts) == 2

    avail, prep = usgs_m2m.request_downloads(api_key, opts, "lbl")
    assert avail == {1: "https://example.com/scn.tar"}
    assert prep == [2]
    usgs_m2m.usgs_logout(api_key)


def test_optional_requests_missing_dataaccess(tmp_path):
    """Verify dataaccess modules omit/guard requests functions when requests is missing."""
    import importlib
    import sys
    from unittest.mock import patch

    import rsgispy.dataaccess as rdataaccess

    orig_odata = copernicus_odata
    orig_cmr = nasa_cmr
    try:
        with patch.dict(sys.modules, {"requests": None}):
            sys.modules.pop("rsgispy.dataaccess.copernicus_odata", None)
            sys.modules.pop("rsgispy.dataaccess.nasa_cmr", None)
            odata_mod = importlib.import_module("rsgispy.dataaccess.copernicus_odata")
            cmr_mod = importlib.import_module("rsgispy.dataaccess.nasa_cmr")

        assert not odata_mod.HAVE_REQUESTS
        for fn_name in (
            "get_access_token",
            "query_scn_lst",
            "query_scn",
            "download_scn",
            "download_scns",
        ):
            assert hasattr(odata_mod, fn_name)
            assert fn_name in odata_mod.__all__
        with pytest.raises(rsgispy.RSGISPyException, match="pip install"):
            odata_mod.get_access_token("user", "pass")
        with pytest.raises(rsgispy.RSGISPyException, match="pip install"):
            odata_mod.query_scn_lst("S2", (0.0, 1.0, 0.0, 1.0))
        assert (
            odata_mod.get_sensor_collection_name(odata_mod.RSGIS_ODATA_SEN2)
            == "SENTINEL-2"
        )

        assert not cmr_mod.HAVE_REQUESTS
        for sym_name in (
            "SessionWithHeaderNASARedirection",
            "get_prods_info",
            "check_prod_version_avail",
            "get_max_prod_version",
            "find_granules",
            "find_all_granules",
            "cmr_download_file_http",
        ):
            assert hasattr(cmr_mod, sym_name)
            assert sym_name in cmr_mod.__all__
        with pytest.raises(rsgispy.RSGISPyException, match="pip install"):
            cmr_mod.SessionWithHeaderNASARedirection("user", "pass")
        with pytest.raises(rsgispy.RSGISPyException, match="pip install"):
            cmr_mod.get_prods_info("MOD09GA")
        with pytest.raises(rsgispy.RSGISPyException, match="pip install"):
            cmr_mod.find_granules("MOD09GA", "061")

        assert hasattr(cmr_mod, "get_total_file_size")
        assert "get_total_file_size" in cmr_mod.__all__

        if cmr_mod.HAVE_PYSONDB:
            for sym_name in (
                "create_cmr_dwnld_db",
                "download_granules_use_dwnld_db",
            ):
                assert hasattr(cmr_mod, sym_name)
                assert sym_name in cmr_mod.__all__

            db_json = str(tmp_path / "cmr.json")
            cmr_mod.create_cmr_dwnld_db(db_json, [], "application/x-hdfeos")
            with pytest.raises(rsgispy.RSGISPyException):
                cmr_mod.download_granules_use_dwnld_db(
                    db_json, str(tmp_path), "unused_creds.txt", use_wget=False
                )
    finally:
        sys.modules["rsgispy.dataaccess.copernicus_odata"] = orig_odata
        sys.modules["rsgispy.dataaccess.nasa_cmr"] = orig_cmr
        rdataaccess.copernicus_odata = orig_odata
        rdataaccess.nasa_cmr = orig_cmr


def test_optional_pysondb_missing_dataaccess():
    """Verify nasa_cmr exposes pysondb functions and raises RSGISPyException."""
    import importlib
    import sys
    from unittest.mock import patch

    import rsgispy.dataaccess as rdataaccess

    orig_cmr = nasa_cmr
    try:
        with patch.dict(sys.modules, {"pysondb": None}):
            sys.modules.pop("rsgispy.dataaccess.nasa_cmr", None)
            cmr_mod = importlib.import_module("rsgispy.dataaccess.nasa_cmr")

        assert not cmr_mod.HAVE_PYSONDB
        for fn_name in ("create_cmr_dwnld_db", "download_granules_use_dwnld_db"):
            assert hasattr(cmr_mod, fn_name)
            assert fn_name in cmr_mod.__all__
        with pytest.raises(rsgispy.RSGISPyException, match="pip install"):
            cmr_mod.create_cmr_dwnld_db("db.json", [], "application/x-hdfeos")
        with pytest.raises(rsgispy.RSGISPyException, match="pip install"):
            cmr_mod.download_granules_use_dwnld_db("db.json", ".", "creds.txt")
        assert hasattr(cmr_mod, "get_total_file_size")
        assert "get_total_file_size" in cmr_mod.__all__
    finally:
        sys.modules["rsgispy.dataaccess.nasa_cmr"] = orig_cmr
        rdataaccess.nasa_cmr = orig_cmr


def test_copernicus_download_scns_network_errors_and_token_refresh(tmp_path, monkeypatch):
    """no_except also catches network errors; tokens are refreshed when old."""
    pytest.importorskip("requests")
    import requests  # pylint: disable=import-outside-toplevel

    tokens_used = []

    def _fake_download(token, scn_info, out_path, use_pbar=True):
        tokens_used.append(token)
        if scn_info["Name"] == "bad":
            raise requests.exceptions.ConnectionError("network down")

    clock = {"now": 1000.0}
    monkeypatch.setattr(copernicus_odata, "download_scn", _fake_download)
    monkeypatch.setattr(copernicus_odata.time, "monotonic", lambda: clock["now"])
    scns = [{"Name": "a"}, {"Name": "bad"}, {"Name": "b"}]

    # A network error is reported and the batch continues.
    copernicus_odata.download_scns("tok0", scns, str(tmp_path), use_pbar=False)
    assert tokens_used == ["tok0", "tok0", "tok0"]
    with pytest.raises(requests.exceptions.ConnectionError):
        copernicus_odata.download_scns("tok0", scns, str(tmp_path), no_except=False,
                                       use_pbar=False)

    # The token is refreshed once it is older than token_max_age.
    tokens_used.clear()
    issued = iter(["tok1", "tok2", "tok3"])

    def _advance_download(token, scn_info, out_path, use_pbar=True):
        tokens_used.append(token)
        clock["now"] += 300.0

    monkeypatch.setattr(copernicus_odata, "download_scn", _advance_download)
    copernicus_odata.download_scns(
        "tok0", [{"Name": n} for n in "abcd"], str(tmp_path), use_pbar=False,
        refresh_token_fn=lambda: next(issued), token_max_age=480.0,
    )
    assert tokens_used == ["tok0", "tok0", "tok1", "tok1"]


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
