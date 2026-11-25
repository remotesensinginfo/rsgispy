"""
Run the documented analysis-ready data worked example
(``docs/examples/analysis_ready_data.py``) end-to-end on synthetic Sentinel-2
``.SAFE`` products so the code shown in ``docs/guides/analysis_ready_data.rst``
stays correct.

Nothing is downloaded: the Copernicus Data Space client is replaced with fakes
that "download" zipped synthetic products, and the CloudSEN12 model is
replaced with a simple threshold predictor (so neither ``torch`` nor the
model weights are needed).
"""

import importlib.util
import os
import shutil

import numpy as np
import pytest
import rasterio

import rsgispy.dataaccess.copernicus_odata as cdse
import rsgispy.imagecalibration.cloudmasking as cldmsk

from ard_scene import make_ard_scenes, remove_safe, zip_safe

EXAMPLE = os.path.join(
    os.path.dirname(__file__), os.pardir, "docs", "examples", "analysis_ready_data.py"
)
GUIDE = os.path.join(
    os.path.dirname(__file__), os.pardir, "docs", "guides", "analysis_ready_data.rst"
)


def _load_example():
    spec = importlib.util.spec_from_file_location("analysis_ready_data", EXAMPLE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(autouse=True)
def _restore_creation_opts(monkeypatch):
    """The example sets COG creation options in os.environ; restore them."""
    for fmt in ("COG", "GTIFF"):
        for prefix in ("RSGISPY", "RSGISLIB"):
            key = f"{prefix}_IMG_CRT_OPTS_{fmt}"
            monkeypatch.setenv(key, os.environ.get(key, "COMPRESS=DEFLATE"))


def _fake_cdse(monkeypatch, zips):
    """Replace the network calls of the Copernicus OData client."""
    monkeypatch.setenv("RSGIS_COP_USER", "test-user")
    monkeypatch.setenv("RSGIS_COP_PASS", "test-pass")
    calls = {}

    def _query(sensor, bbox, **kwargs):
        calls["query"] = (sensor, bbox, kwargs)
        return [{"Name": os.path.basename(z)[: -len(".zip")]} for z in zips]

    def _token(username, password):
        calls["token"] = (username, password)
        return "fake-token"

    def _download(token, scns, out_path, **kwargs):
        assert token == "fake-token"
        os.makedirs(out_path, exist_ok=True)
        for zip_file in zips:
            shutil.copy(zip_file, out_path)

    monkeypatch.setattr(cdse, "query_scn_lst", _query)
    monkeypatch.setattr(cdse, "get_access_token", _token)
    monkeypatch.setattr(cdse, "download_scns", _download)
    return calls


def test_worked_example_runs(tmp_path, monkeypatch):
    import datetime

    mod = _load_example()
    scenes = make_ard_scenes(str(tmp_path / "safe"))
    # The second tile is "downloaded": zip its products and delete the folders.
    zip_dir = tmp_path / "zips"
    zip_dir.mkdir()
    zips = []
    for safe_dir in scenes.pop("T30UWD"):
        zips.append(zip_safe(safe_dir, str(zip_dir)))
        remove_safe(safe_dir)
    calls = _fake_cdse(monkeypatch, zips)

    out = mod.run_workflow(
        str(tmp_path / "out"),
        scenes,
        bbox=(-3.0, -2.9, 51.4, 51.5),
        dates=(datetime.datetime(2024, 6, 1), datetime.datetime(2024, 6, 30)),
        coreg_win_size=32,
        coreg_pixel_gap=16,
        out_epsg=32630,
        use_pbar=False,
    )

    # Step 1: search with the credentials from the environment, unzip
    assert calls["query"][0] == cdse.RSGIS_ODATA_SEN2
    assert calls["token"] == ("test-user", "test-pass")
    assert sorted(out["scenes"]) == ["T30UVD", "T30UWD"]
    assert all(os.path.isdir(p) for p in out["scenes"]["T30UWD"])

    # Steps 2-3: 10-band stacks and clear-sky proportions
    for imgs in out["sref_imgs"].values():
        for img in imgs:
            with rasterio.open(img) as src:
                assert src.count == 10 and src.nodata == 0
    assert all(0.5 < prop < 1.0 for prop in out["clear_props"].values())

    # Step 4: the misregistered date (+15 m, -10 m) is corrected
    shifted = [v for k, v in out["coreg_info"].items() if "20240612" in k]
    assert len(shifted) == 2
    for info in shifted:
        assert info["n_inliers"] >= 3
        assert info["corr_after"] > info["corr_before"] + 0.05

    # Step 6: composites are complete and cloud-free
    for comp in out["composites"].values():
        with rasterio.open(comp["sref"]) as src:
            arr = src.read()
            assert src.descriptions[0] == "Blue"
        assert np.all(arr[0] > 0)  # every pixel has a clear observation
        assert arr[0].max() < 2000  # no cloud (blue reflectance 0.55)
        with rasterio.open(comp["n_obs"]) as src:
            assert src.read(1).min() >= 1
        with rasterio.open(comp["ndvi"]) as src:
            assert src.count == 3

    # Steps 7-8: COG mosaics covering both tiles, checked
    with rasterio.open(out["mosaics"]["sref"]) as src:
        assert src.width == 160 and src.count == 10
        assert src.tags(ns="IMAGE_STRUCTURE").get("COMPRESSION") == "DEFLATE"
        assert src.tags(ns="IMAGE_STRUCTURE").get("LAYOUT") == "COG"
        assert "STATISTICS_MEAN" in src.tags(1)  # calc_stats=True kept the COG
    assert out["sref_ok"] and out["ndvi_ok"]


def test_worked_example_cloudsen12(tmp_path, monkeypatch):
    """The CloudSEN12 path, with the model replaced by thresholds."""

    def _predict(arr):  # (12, H, W) reflectance -> 0 clear, 1 thick, 3 shadow
        out = np.zeros(arr.shape[1:], dtype=np.uint8)
        out[arr[1] > 0.3] = 1
        out[(arr[7] > 0) & (arr[7] < 0.15)] = 3
        return out

    monkeypatch.setattr(cldmsk, "get_cloudsen12_weights", lambda **kw: "w.pt")
    monkeypatch.setattr(cldmsk, "get_torch_device", lambda device: "cpu")
    monkeypatch.setattr(cldmsk, "_load_cloudsen12_model", lambda *args: None)
    monkeypatch.setattr(cldmsk, "_build_torch_predictor", lambda *args: _predict)

    mod = _load_example()
    scenes = make_ard_scenes(str(tmp_path / "safe"), tiles=("T30UVD",))
    out = mod.run_workflow(
        str(tmp_path / "out"),
        scenes,
        use_cloudsen12=True,
        coreg_win_size=32,
        coreg_pixel_gap=16,
        use_pbar=False,
    )
    for img in out["sref_imgs"]["T30UVD"]:
        with rasterio.open(img.replace("_sref.tif", "_cloudsen12.tif")) as src:
            assert {2, 4} <= set(np.unique(src.read(1)).tolist())
    with rasterio.open(out["composites"]["T30UVD"]["sref"]) as src:
        arr = src.read(1)
    assert np.all(arr > 0) and arr.max() < 2000


def test_guide_markers_exist():
    """Every literalinclude marker used by the guide exists in the script."""
    with open(EXAMPLE, encoding="utf-8") as f_obj:
        script = f_obj.read()
    with open(GUIDE, encoding="utf-8") as f_obj:
        guide = f_obj.read()
    markers = [
        line.split(":start-after:")[1].strip()
        for line in guide.splitlines()
        if ":start-after:" in line
    ]
    assert len(markers) >= 8
    for marker in markers:
        assert marker in script
        assert marker.replace("[step-", "[end-step-") in script
