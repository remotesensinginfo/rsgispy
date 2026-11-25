"""
Tests for writing KEA outputs that have a no-data value and statistics.

GDAL's KEA driver (kealib) reports a failure from ``SetNoDataValue`` when a
band already has a no-data value, so rasterio's ``dataset.nodata = value``
raised ``ValueError: Invalid nodata value`` whenever statistics were
calculated (``calc_stats=True``) for a KEA output that already had one.
"""

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

import rsgispy
import rsgispy.imagefilter.sar
import rsgispy.imageprocess
import rsgispy.imageutils.masking
import rsgispy.imageutils.metadata
import rsgispy.imageutils.stats
import rsgispy.segmentation

with rasterio.Env() as _env:
    HAVE_KEA = "KEA" in _env.drivers()

pytestmark = pytest.mark.skipif(not HAVE_KEA, reason="GDAL KEA driver not available")


def _write_raster(path, data, driver="GTiff", dtype="float32", nodata=None):
    if data.ndim == 2:
        data = data[np.newaxis]
    count, height, width = data.shape
    with rasterio.open(
        path,
        "w",
        driver=driver,
        height=height,
        width=width,
        count=count,
        dtype=dtype,
        crs="EPSG:32630",
        transform=from_origin(500000.0, 5200000.0, 10.0, 10.0),
    ) as dst:
        dst.write(data.astype(dtype))
        if nodata is not None:
            dst.nodata = nodata


def _assert_stats(path, n_bands=1):
    with rasterio.open(path) as src:
        assert src.driver == "KEA"
        for bidx in range(1, n_bands + 1):
            assert "STATISTICS_MEAN" in src.tags(bidx)


def test_set_dataset_nodata_kea(tmp_path):
    """Setting the value a KEA file already has is a no-op; a new value raises clearly."""
    kea = tmp_path / "img.kea"
    _write_raster(kea, np.ones((2, 20, 30)), driver="KEA", nodata=-9999.0)

    with rasterio.open(kea, "r+") as dst:
        rsgispy.imageprocess.set_dataset_nodata(dst, -9999)
        rsgispy.imageprocess.set_dataset_nodata(dst, -9999.0)
    with rasterio.open(kea) as src:
        assert src.nodatavals == (-9999.0, -9999.0)

    with pytest.raises(rsgispy.RSGISPyException, match=r"KEA.*-9999.*-1"):
        with rasterio.open(kea, "r+") as dst:
            rsgispy.imageprocess.set_dataset_nodata(dst, -1)
    # Nothing was changed by the refused update.
    with rasterio.open(kea) as src:
        assert src.nodatavals == (-9999.0, -9999.0)

    # A KEA band without a no-data value can still be given one.
    kea_none = tmp_path / "img_none.kea"
    _write_raster(kea_none, np.ones((2, 20, 30)), driver="KEA")
    with rasterio.open(kea_none, "r+") as dst:
        rsgispy.imageprocess.set_dataset_nodata(dst, 0)
    with rasterio.open(kea_none) as src:
        assert src.nodatavals == (0.0, 0.0)

    # GeoTIFF (and other drivers) can change the value.
    tif = tmp_path / "img.tif"
    _write_raster(tif, np.ones((2, 20, 30)), nodata=-9999.0)
    with rasterio.open(tif, "r+") as dst:
        rsgispy.imageprocess.set_dataset_nodata(dst, 5)
    with rasterio.open(tif) as src:
        assert src.nodatavals == (5.0, 5.0)


def test_pop_img_stats_existing_kea_with_nodata(tmp_path):
    """pop_img_stats(no_data_val=...) on a KEA file that already has that value."""
    kea = tmp_path / "img.kea"
    data = np.arange(2 * 40 * 50, dtype=np.float32).reshape(2, 40, 50)
    data[:, :5, :5] = -9999.0
    _write_raster(kea, data, driver="KEA", nodata=-9999.0)

    rsgispy.imageutils.stats.pop_img_stats(
        str(kea), no_data_val=-9999, calc_pyramids=True, use_pbar=False
    )
    _assert_stats(kea, n_bands=2)
    with rasterio.open(kea) as src:
        assert src.nodatavals == (-9999.0, -9999.0)
        assert float(src.tags(1)["STATISTICS_MINIMUM"]) == pytest.approx(5.0)

    # A second run (the value is already set) also works.
    rsgispy.imageutils.stats.pop_img_stats(
        str(kea), no_data_val=-9999.0, calc_pyramids=False, use_pbar=False
    )
    _assert_stats(kea, n_bands=2)

    with pytest.raises(rsgispy.RSGISPyException, match="KEA"):
        rsgispy.imageutils.stats.pop_img_stats(
            str(kea), no_data_val=0, calc_pyramids=False, use_pbar=False
        )


def test_pop_thmt_img_stats_existing_kea(tmp_path):
    """pop_thmt_img_stats sets no data 0 on a KEA file that already has it."""
    kea = tmp_path / "cls.kea"
    data = np.repeat(np.arange(4, dtype=np.uint8), 250).reshape(20, 50)
    _write_raster(kea, data, driver="KEA", dtype="uint8", nodata=0)
    rsgispy.imageutils.stats.pop_thmt_img_stats(
        str(kea), add_clr_tab=True, calc_pyramids=False, use_pbar=False
    )
    rsgispy.imageutils.stats.pop_thmt_img_stats(
        str(kea), add_clr_tab=True, calc_pyramids=False, use_pbar=False
    )
    with rasterio.open(kea) as src:
        assert src.nodata == 0.0
        assert "STATISTICS_MAXIMUM" in src.tags(1)


def test_set_img_no_data_value_kea(tmp_path):
    """set_img_no_data_value with the existing value works for KEA."""
    kea = tmp_path / "img.kea"
    _write_raster(kea, np.ones((2, 20, 30)), driver="KEA", nodata=-9999.0)
    rsgispy.imageutils.metadata.set_img_no_data_value(str(kea), -9999.0)
    rsgispy.imageutils.metadata.set_img_no_data_value(str(kea), -9999.0, img_band=2)
    with rasterio.open(kea) as src:
        assert src.nodatavals == (-9999.0, -9999.0)
    with pytest.raises(rsgispy.RSGISPyException, match="KEA"):
        rsgispy.imageutils.metadata.set_img_no_data_value(str(kea), 3.0)


def test_mask_img_kea_with_nodata_and_stats(tmp_path):
    """mask_img(..., gdalformat='KEA', no_data_val=-9999, calc_stats=True)."""
    in_img = tmp_path / "in.tif"
    msk_img = tmp_path / "msk.tif"
    out_img = tmp_path / "out.kea"
    _write_raster(in_img, np.full((2, 40, 50), 7.0))
    msk = np.ones((40, 50), dtype=np.uint8)
    msk[:10] = 0
    _write_raster(msk_img, msk, dtype="uint8")
    rsgispy.imageutils.masking.mask_img(
        str(in_img),
        str(msk_img),
        str(out_img),
        out_value=-9999,
        mask_value=0,
        gdalformat="KEA",
        datatype=rsgispy.TYPE_32FLOAT,
        no_data_val=-9999,
        calc_stats=True,
        use_pbar=False,
    )
    _assert_stats(out_img, n_bands=2)
    with rasterio.open(out_img) as src:
        assert src.nodatavals == (-9999.0, -9999.0)
        assert float(src.tags(1)["STATISTICS_MEAN"]) == pytest.approx(7.0)


def test_apply_lee_filter_kea(tmp_path):
    """apply_lee_filter(..., gdalformat='KEA') with the default calc_stats=True."""
    in_img = tmp_path / "sar.tif"
    out_img = tmp_path / "sar_lee.kea"
    rng = np.random.default_rng(1)
    _write_raster(in_img, rng.gamma(4.0, 0.05, size=(40, 50)), nodata=0.0)
    rsgispy.imagefilter.sar.apply_lee_filter(
        str(in_img),
        str(out_img),
        filter_size=5,
        n_looks=4.0,
        gdalformat="KEA",
        no_data_val=0.0,
        use_pbar=False,
    )
    _assert_stats(out_img)


def test_shepherd_segmentation_kea(tmp_path):
    """shepherd_segmentation(..., gdalformat='KEA') with the default calc_stats=True."""
    in_img = tmp_path / "img.tif"
    out_img = tmp_path / "segs.kea"
    data = np.zeros((3, 40, 50), dtype=np.float32)
    data[:, :, 25:] = 100.0
    data[:, 20:, :] += 50.0
    data += np.random.default_rng(2).normal(0.0, 1.0, size=data.shape)
    _write_raster(in_img, data)
    rsgispy.segmentation.shepherd_segmentation(
        str(in_img),
        str(out_img),
        n_clusters=4,
        min_pixels=20,
        gdalformat="KEA",
        use_pbar=False,
    )
    with rasterio.open(out_img) as src:
        assert src.driver == "KEA"
        assert src.read(1).max() >= 4
        assert "STATISTICS_MAXIMUM" in src.tags(1)
