"""
Tests for :func:`rsgispy.classification.classpostprocess.majority_filter_node`:
chained majority nodes must equal the file-based majority filter run one
pass after another.
"""

import numpy as np
import pytest
import rasterio

import rsgispy
import rsgispy.classification.classpostprocess as cp
from rsgispy.classification import majority_filter_node
from rsgispy.imagemorphology import image_closing_node
import rsgispy.imagemorphology as rm
from rsgispy.imageprocess import DiskSink, MemorySink, RasterSource, SelectBands, TileGraph

from conftest import write_raster


def _read(path, band=1):
    with rasterio.open(path) as src:
        return src.read(band)


@pytest.fixture(name="cls_img")
def fixture_cls_img(tmp_path):
    rng = np.random.default_rng(11)
    coarse = rng.integers(1, 6, size=(8, 8))
    cls = np.kron(coarse, np.ones((6, 6), dtype=np.int64))[:43, :47].astype(np.uint8)
    noise = rng.random(cls.shape) < 0.2
    cls[noise] = rng.integers(0, 6, size=int(noise.sum()))
    other = rng.integers(1, 4, size=cls.shape).astype(np.uint8)
    return write_raster(
        tmp_path / "cls.tif", np.stack([other, cls]), nodata=0, origin=(500000.0, 6000000.0)
    )


@pytest.mark.parametrize("block_size", [16, 256])
def test_majority_chain_equals_file_passes(cls_img, tmp_path, block_size):
    """Two chained majority nodes == two apply_majority_filter runs."""
    cp.apply_majority_filter(
        cls_img, str(tmp_path / "m1.tif"), 3, img_band=2, use_pbar=False
    )
    cp.apply_majority_filter(
        str(tmp_path / "m1.tif"), str(tmp_path / "m2.tif"), 5,
        keep_centre_on_tie=False, use_pbar=False,
    )
    src = SelectBands(RasterSource(cls_img), 2)
    node = majority_filter_node(src, 3, dtype=np.uint8, no_data_val=0)
    node = majority_filter_node(
        node, 5, keep_centre_on_tie=False, dtype=np.uint8, no_data_val=0
    )
    out = str(tmp_path / "graph.tif")
    graph = TileGraph(DiskSink(node, out, datatype=rsgispy.TYPE_8UINT), block_size=block_size)
    assert graph.node_meta[node].nodata == 0
    graph.run(use_pbar=False)
    np.testing.assert_array_equal(_read(out), _read(str(tmp_path / "m2.tif")))
    with rasterio.open(out) as src_ds:
        assert src_ds.nodata == 0


def test_majority_node_all_bands_and_pipeline(cls_img, tmp_path):
    """The node filters every band; the pipeline equals the node chain."""
    node = majority_filter_node(RasterSource(cls_img), 3, no_data_val=0, dtype=np.uint8)
    res = TileGraph(MemorySink(node, dtype=np.uint8, name="m"), block_size=10).run(
        use_pbar=False
    )["m"]
    for band in (1, 2):
        cp.apply_majority_filter(
            cls_img, str(tmp_path / f"b{band}.tif"), 3, img_band=band, use_pbar=False
        )
        np.testing.assert_array_equal(res[band - 1], _read(str(tmp_path / f"b{band}.tif")))

    cp.apply_postprocess_pipeline(
        cls_img, str(tmp_path / "pipe.tif"), majority_sizes=(3, 5), img_band=2,
        use_pbar=False,
    )
    node = majority_filter_node(
        majority_filter_node(SelectBands(RasterSource(cls_img), 2), 3, no_data_val=0, dtype=np.uint8),
        5, no_data_val=0, dtype=np.uint8,
    )
    res = TileGraph(MemorySink(node, dtype=np.uint8, name="p")).run(use_pbar=False)["p"]
    np.testing.assert_array_equal(res[0], _read(str(tmp_path / "pipe.tif")))


def test_majority_after_morphology(cls_img, tmp_path):
    """A closing followed by a majority filter in one graph == file functions."""
    rm.image_closing(cls_img, str(tmp_path / "c.tif"), use_pbar=False)
    cp.apply_majority_filter(
        str(tmp_path / "c.tif"), str(tmp_path / "cm.tif"), 3, img_band=2, use_pbar=False
    )
    node = image_closing_node(RasterSource(cls_img), dtype=np.uint8, no_data_val=0)
    node = majority_filter_node(SelectBands(node, 2), 3, dtype=np.uint8, no_data_val=0)
    res = TileGraph(MemorySink(node, dtype=np.uint8, name="x"), block_size=12).run(
        use_pbar=False
    )["x"]
    np.testing.assert_array_equal(res[0], _read(str(tmp_path / "cm.tif")))


def test_majority_node_validation(cls_img):
    """The builder validates filter_size like apply_majority_filter."""
    src = RasterSource(cls_img)
    for bad in (2, 4, 1):
        with pytest.raises(rsgispy.RSGISPyException):
            majority_filter_node(src, bad)
