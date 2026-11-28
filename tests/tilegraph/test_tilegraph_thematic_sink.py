"""
Tests for thematic :class:`DiskSink` outputs (``thematic``, ``colour_table``
and ``classes``) and the band names of the class-map nodes
(:func:`predict_raster_node`, :func:`majority_filter_node`).
"""

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

import rsgispy
import rsgispy.imageprocess
import rsgispy.ml.predict as mp
from rsgispy.classification.classpostprocess import majority_filter_node
from rsgispy.imageprocess import (
    ArraySource,
    DiskSink,
    MemorySink,
    SetBandNames,
    TileGraph,
    infer_node_meta,
)
from rsgispy.imageutils import colourtable

H, W = 300, 280
TRANSFORM = from_origin(400000, 5000000, 10, 10)
CLASSES = {1: ("Water", "#1f78b4"), 2: ("Forest", (51, 160, 44)), 3: ("Urban", "#e31a1c")}


def _cls_src(dtype=np.uint8, nodata=0):
    arr = (np.add.outer(np.arange(H), np.arange(W)) // 40 % 4).astype(dtype)
    return ArraySource(
        arr[np.newaxis], transform=TRANSFORM, crs=CRS.from_epsg(32630), nodata=nodata
    ), arr


def _float_src():
    rng = np.random.default_rng(3)
    arr = rng.uniform(0, 1, size=(1, H, W)).astype(np.float32)
    arr[0, :5] = -9999.0
    return ArraySource(arr, transform=TRANSFORM, crs=CRS.from_epsg(32630), nodata=-9999.0)


def _run(sink):
    TileGraph(sink, block_size=64).run(use_pbar=False)


def test_disk_sink_classes(tmp_path):
    src, arr = _cls_src()
    out = str(tmp_path / "cls.tif")
    _run(DiskSink(src, out, thematic=True, classes=CLASSES))
    assert colourtable.is_img_thematic(out)
    assert colourtable.get_thematic_classes(out) == {
        1: ("Water", "#1f78b4"),
        2: ("Forest", "#33a02c"),
        3: ("Urban", "#e31a1c"),
    }
    clr = colourtable.get_colour_tab_info(out)
    assert (clr[2]["red"], clr[2]["green"], clr[2]["blue"]) == (51, 160, 44)
    with rasterio.open(out) as ds:
        np.testing.assert_array_equal(ds.read(1), arr)
        tags = ds.tags(1)
        assert ds.nodata == 0
        # Thematic statistics (no calc_stats needed) and pyramids.
        assert tags["STATISTICS_HISTOBINFUNCTION"] == "direct"
        assert ds.overviews(1)


def test_disk_sink_classes_imply_thematic(tmp_path):
    src, _ = _cls_src()
    out = str(tmp_path / "cls.tif")
    _run(DiskSink(src, out, classes=CLASSES, calc_stats=False))
    assert colourtable.is_img_thematic(out)
    assert colourtable.get_thematic_classes(out)[3] == ("Urban", "#e31a1c")
    with rasterio.open(out) as ds:
        assert "STATISTICS_MEAN" not in ds.tags(1)


def test_disk_sink_colour_table(tmp_path):
    src, _ = _cls_src()
    out = str(tmp_path / "cls.tif")
    lut = {1: [255, 0, 0], 2: "#00ff00", 3: (0, 0, 255, 255)}
    _run(DiskSink(src, out, thematic=True, colour_table=lut))
    assert colourtable.is_img_thematic(out)
    clr = colourtable.get_colour_tab_info_for_def(out)
    assert clr[1][:3] == [255, 0, 0]
    assert clr[2][:3] == [0, 255, 0]
    assert clr[3][:3] == [0, 0, 255]
    assert not colourtable.get_thematic_classes(out)


def test_disk_sink_thematic_without_table(tmp_path):
    src, _ = _cls_src()
    out = str(tmp_path / "cls.tif")
    _run(DiskSink(src, out, thematic=True))
    assert colourtable.is_img_thematic(out)
    with rasterio.open(out) as ds:
        assert ds.tags(1)["STATISTICS_HISTOBINFUNCTION"] == "direct"
        assert ds.colormap(1)


def test_disk_sink_thematic_cog(tmp_path):
    src, arr = _cls_src()
    out = str(tmp_path / "cls_cog.tif")
    _run(DiskSink(src, out, gdalformat="COG", classes=CLASSES))
    assert rsgispy.imageprocess.is_cog_layout(out)
    assert colourtable.is_img_thematic(out)
    assert colourtable.get_thematic_classes(out)[1] == ("Water", "#1f78b4")
    with rasterio.open(out) as ds:
        np.testing.assert_array_equal(ds.read(1), arr)
        assert "STATISTICS_MAXIMUM" in ds.tags(1)


def test_disk_sink_thematic_validation(tmp_path):
    src, _ = _cls_src()
    with pytest.raises(rsgispy.RSGISPyException):
        DiskSink(src, str(tmp_path / "a.tif"), colour_table={1: "#ff0000"}, classes=CLASSES)
    with pytest.raises(rsgispy.RSGISPyException):
        TileGraph(DiskSink(_float_src(), str(tmp_path / "b.tif"), thematic=True))
    i16, _ = _cls_src(np.int16)
    with pytest.raises(rsgispy.RSGISPyException):
        TileGraph(DiskSink(i16, str(tmp_path / "c.tif"), classes=CLASSES))
    with pytest.raises(rsgispy.RSGISPyException):
        DiskSink(src, str(tmp_path / "d.tif"), classes={1: "Water"})


def test_continuous_disk_sink_unchanged(tmp_path):
    out = str(tmp_path / "flt.tif")
    _run(DiskSink(_float_src(), out, calc_stats=True))
    assert not colourtable.is_img_thematic(out)
    with rasterio.open(out) as ds:
        assert ds.dtypes[0] == "float32"
        assert ds.nodata == -9999.0
        assert "STATISTICS_MEAN" in ds.tags(1)
        assert ds.tags(1).get("STATISTICS_HISTOBINFUNCTION") != "direct"
        with pytest.raises(ValueError):
            ds.colormap(1)
    out2 = str(tmp_path / "flt2.tif")
    _run(DiskSink(_float_src(), out2))
    with rasterio.open(out2) as ds:
        assert "STATISTICS_MEAN" not in ds.tags(1)


def test_majority_filter_node_band_name():
    src, _ = _cls_src()
    node = majority_filter_node(src, 3, dtype=np.uint8)
    assert infer_node_meta(node).band_names == ("class",)
    named = majority_filter_node(SetBandNames(src, ["landcover"]), 3)
    assert infer_node_meta(named).band_names == ("landcover",)


class _ConstClf:
    """Minimal classifier: class 1 where feature 0 < 0.5, else 2."""

    n_features_in_ = 1
    classes_ = np.array([1, 2])

    def predict(self, x):
        return np.where(np.asarray(x)[:, 0] < 0.5, 1, 2)


def test_predict_raster_node_band_name(tmp_path):
    node = mp.predict_raster_node(_float_src(), _ConstClf(), thematic=True)
    assert infer_node_meta(node).band_names == ("class",)
    named = mp.predict_raster_node(
        _float_src(), _ConstClf(), thematic=True, band_names=["lc"]
    )
    assert infer_node_meta(named).band_names == ("lc",)
    # The class map is written with its band name and classes in one pass.
    out = str(tmp_path / "pred.tif")
    smooth = majority_filter_node(node, 3, dtype=np.uint8, no_data_val=0)
    _run(DiskSink(smooth, out, classes={1: ("Low", "#000080"), 2: ("High", "#ffa500")}))
    with rasterio.open(out) as ds:
        assert ds.descriptions[0] == "class"
        assert ds.dtypes[0] == "uint8"
    assert colourtable.get_thematic_classes(out)[2] == ("High", "#ffa500")
    mem = TileGraph(MemorySink(node, name="m")).run(use_pbar=False)["m"]
    assert set(np.unique(mem)) <= {0, 1, 2}
