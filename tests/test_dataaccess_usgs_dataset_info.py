"""
Direct tests of :func:`rsgispy.dataaccess.usgs_m2m.get_dataset_info` with the
HTTP layer (``rsgispy.tools.http.send_http_json_request``) mocked: no network.
"""

import pytest

import rsgispy
import rsgispy.dataaccess.usgs_m2m as m2m
import rsgispy.tools.http


def _mock(monkeypatch, response):
    calls = []

    def _fake_send(url, data=None, api_key=None, **kwargs):
        calls.append({"url": url, "data": data, "api_key": api_key, **kwargs})
        return response

    monkeypatch.setattr(rsgispy.tools.http, "send_http_json_request", _fake_send)
    return calls


def test_get_dataset_info_request_and_payload(monkeypatch):
    payload = {
        "datasetAlias": "landsat_ot_c2_l2",
        "collectionName": "Landsat 8-9 OLI/TIRS C2 L2",
        "datasetId": "5e83d14fec7cae84",
    }
    calls = _mock(monkeypatch, {"errorCode": None, "errorMessage": None, "data": payload})
    info = m2m.get_dataset_info("landsat_ot_c2_l2", "KEY123")
    assert info == payload
    assert calls == [{
        "url": m2m.USGS_M2M_URL + "dataset",
        "data": {"datasetName": "landsat_ot_c2_l2"},
        "api_key": "KEY123",
    }]
    assert calls[0]["url"] == "https://m2m.cr.usgs.gov/api/api/json/stable/dataset"


def test_get_dataset_info_api_error(monkeypatch):
    _mock(monkeypatch, {
        "errorCode": "DATASET_INVALID", "errorMessage": "Invalid dataset", "data": None,
    })
    with pytest.raises(rsgispy.RSGISPyException, match="DATASET_INVALID - Invalid dataset"):
        m2m.get_dataset_info("not_a_dataset", "KEY123")


def test_get_dataset_info_unexpected_response(monkeypatch, capsys):
    _mock(monkeypatch, {"errorCode": None, "something": 1})
    with pytest.raises(rsgispy.RSGISPyException, match="not as expected"):
        m2m.get_dataset_info("landsat_ot_c2_l2", "KEY123")
    capsys.readouterr()
