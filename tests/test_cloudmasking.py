"""
Tests for rsgispy.imagecalibration.cloudmasking (CloudSEN12 cloud masking).

The model inference is replaced with a deterministic fake predictor so these
tests run without ``torch`` installed.
"""

import os

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

import rsgispy
import rsgispy.imagecalibration.cloudmasking as cldmsk


def _write_gtiff(path, arr, nodata=None):
    count, height, width = arr.shape
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": count,
        "dtype": str(arr.dtype),
        "crs": "EPSG:32630",
        "transform": from_origin(500000.0, 5700000.0, 10.0, 10.0),
    }
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr)
    return path


def _fake_predict_factory(calls):
    """Class from the reflectance of the first band: 0..3 by thresholds."""

    def _predict(arr):
        assert arr.dtype == np.float32
        assert arr.shape[1] % 32 == 0 and arr.shape[2] % 32 == 0
        calls.append(arr.shape)
        b1 = arr[0]
        out = np.zeros(b1.shape, dtype=np.uint8)
        out[b1 >= 0.1] = 1
        out[b1 >= 0.2] = 2
        out[b1 >= 0.3] = 3
        return out

    return _predict


def _make_input(tmp_path, n_bands, height=70, width=90):
    rng = np.random.default_rng(42)
    data = rng.integers(1, 4000, size=(n_bands, height, width)).astype(np.uint16)
    data[:, :5, :7] = 0  # no-data corner
    return _write_gtiff(str(tmp_path / "in.tif"), data), data


def _expected(data, scale=10000.0, offset=0.0):
    refl = (data[0].astype(np.float64) + offset) / scale
    exp = np.ones(refl.shape, dtype=np.uint8)
    exp[refl >= 0.1] = 2
    exp[refl >= 0.2] = 3
    exp[refl >= 0.3] = 4
    exp[np.all(data == 0, axis=0)] = 0
    return exp


@pytest.mark.parametrize("tile_size,tile_overlap", [(32, 0), (40, 8), (512, 32)])
def test_tiled_inference_matches_full(tmp_path, tile_size, tile_overlap):
    in_img, data = _make_input(tmp_path, 13)
    out_img = str(tmp_path / "clouds.tif")
    calls = []
    cldmsk._apply_cloudsen12_cloud_mask(
        input_img=in_img,
        output_img=out_img,
        product="toa",
        model_name="UNetMobV2_V2",
        img_bands=None,
        weights_dir=None,
        download_weights=False,
        device="cpu",
        scale_factor=10000.0,
        add_offset=0.0,
        no_data_val=0.0,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        gdalformat="GTIFF",
        calc_stats=True,
        use_pbar=False,
        creation_opts=None,
        predict_fn=_fake_predict_factory(calls),
    )
    assert calls
    with rasterio.open(out_img) as src:
        out = src.read(1)
        assert src.dtypes[0] == "uint8"
        assert src.nodata == 0
        assert src.descriptions[0] == "CloudSEN12"
        cmap = src.colormap(1)
        assert cmap[cldmsk.CLOUDSEN12_THICK_CLOUD][:3] == (220, 220, 220)
    np.testing.assert_array_equal(out, _expected(data))


def test_sref_img_bands_and_offset(tmp_path):
    in_img, data = _make_input(tmp_path, 14)
    out_img = str(tmp_path / "clouds.tif")
    bands = list(range(2, 14))  # 12 bands, starting from band 2
    cldmsk._apply_cloudsen12_cloud_mask(
        input_img=in_img,
        output_img=out_img,
        product="sref",
        model_name="cloudsen12l2a",
        img_bands=bands,
        weights_dir=None,
        download_weights=False,
        device="cpu",
        scale_factor=10000.0,
        add_offset=-1000.0,
        no_data_val=0.0,
        tile_size=64,
        tile_overlap=16,
        gdalformat="GTIFF",
        calc_stats=False,
        use_pbar=False,
        creation_opts=None,
        predict_fn=_fake_predict_factory([]),
    )
    sub = data[1:13]
    exp = _expected(sub, offset=-1000.0)
    with rasterio.open(out_img) as src:
        np.testing.assert_array_equal(src.read(1), exp)


def test_band_count_and_product_validation(tmp_path):
    in_img, _ = _make_input(tmp_path, 10)
    out_img = str(tmp_path / "c.tif")
    with pytest.raises(rsgispy.RSGISPyException, match="requires 13 bands"):
        cldmsk.apply_cloudsen12_s2_toa_cloud_mask(
            in_img, out_img, download_weights=False, use_pbar=False
        )
    with pytest.raises(rsgispy.RSGISPyException, match="requires 12 bands"):
        cldmsk.apply_cloudsen12_s2_sref_cloud_mask(
            in_img, out_img, img_bands=[1, 2, 3], use_pbar=False
        )
    with pytest.raises(rsgispy.RSGISPyException, match="for 'toa' input"):
        cldmsk.apply_cloudsen12_s2_sref_cloud_mask(
            in_img, out_img, model_name="UNetMobV2_V1", use_pbar=False
        )
    with pytest.raises(rsgispy.RSGISPyException, match="Unknown CloudSEN12"):
        cldmsk.apply_cloudsen12_s2_toa_cloud_mask(
            in_img, out_img, model_name="nope", use_pbar=False
        )
    with pytest.raises(rsgispy.RSGISPyException, match="tile_size"):
        cldmsk.apply_cloudsen12_s2_toa_cloud_mask(
            in_img, out_img, tile_size=16, use_pbar=False
        )


def test_weights_resolution_local(tmp_path, monkeypatch):
    wdir = tmp_path / "weights"
    wdir.mkdir()
    wfile = wdir / "UNetMobV2_V2.pt"
    wfile.write_bytes(b"x")
    assert cldmsk.get_cloudsen12_weights("UNetMobV2_V2", str(wdir)) == str(wfile)
    # Direct file path is also accepted.
    assert cldmsk.get_cloudsen12_weights("UNetMobV2_V2", str(wfile)) == str(wfile)
    with pytest.raises(rsgispy.RSGISPyException, match="download=False"):
        cldmsk.get_cloudsen12_weights("UNetMobV2_V1", str(wdir), download=False)

    monkeypatch.setenv(cldmsk.CLOUDSEN12_WEIGHTS_DIR_ENV, str(wdir))
    assert cldmsk.get_cloudsen12_weights_dir() == str(wdir)
    assert cldmsk.get_cloudsen12_weights("UNetMobV2_V2") == str(wfile)


def test_download_weights_mocked(tmp_path, monkeypatch):
    pytest.importorskip("requests")
    import rsgispy.tools._download_common as dlc

    urls = []

    def _fake_download(session, url, out_file_path, **kwargs):
        urls.append(url)
        with open(out_file_path, "wb") as fp:
            fp.write(b"weights")

    monkeypatch.setattr(dlc, "stream_http_download", _fake_download)
    out_dir = str(tmp_path / "cache")
    out = cldmsk.download_cloudsen12_weights(
        "cloudsen12l2a", out_dir=out_dir, use_pbar=False
    )
    assert out == os.path.join(out_dir, "cloudsen12-s2-l2a-1.0.0.pt")
    assert urls == [
        "https://huggingface.co/isp-uv-es/cloudsen12_models/resolve/main/"
        "cloudsen12-s2-l2a-1.0.0.pt"
    ]
    # Existing file is not re-downloaded.
    cldmsk.download_cloudsen12_weights(
        "cloudsen12l2a", out_dir=out_dir, use_pbar=False
    )
    assert len(urls) == 1


def test_get_torch_device():
    pytest.importorskip("torch")
    assert str(cldmsk.get_torch_device("cpu")) == "cpu"
    assert cldmsk.get_torch_device("auto") is not None
    with pytest.raises(rsgispy.RSGISPyException):
        cldmsk.get_torch_device("tpu")
