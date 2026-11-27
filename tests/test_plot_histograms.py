"""
Tests for the histogram plots in ``rsgispy.tools.plotting``:
``plot_img_histogram``, ``plot_2d_img_histogram``, ``draw_histogram`` and the
shared look of ``plot_histogram_threshold``. The plotted values must equal
the results of the histogram functions.
"""

import os

# pylint: disable=redefined-outer-name

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

import rsgispy
import rsgispy.imagecalc.stats as imgstats
import rsgispy.tools.plotting as rplot

matplotlib.use("Agg")

N_ROWS, N_COLS = 30, 40


@pytest.fixture(scope="module")
def hist_imgs(tmp_path_factory):
    """A 2-band image (no data 0, named bands), a second image and a mask."""
    data_dir = tmp_path_factory.mktemp("hist_plots")
    rng = np.random.default_rng(7)
    profile = {
        "driver": "GTiff",
        "width": N_COLS,
        "height": N_ROWS,
        "crs": "EPSG:32630",
        "transform": from_origin(500000.0, 5200000.0, 10.0, 10.0),
    }
    arr = rng.integers(1, 5000, size=(2, N_ROWS, N_COLS)).astype(np.float32)
    arr[:, :2, :] = 0.0  # no data rows
    img = os.path.join(data_dir, "img.tif")
    with rasterio.open(img, "w", count=2, dtype="float32", nodata=0.0, **profile) as dst:
        dst.write(arr)
        dst.set_band_description(1, "NIR")
        dst.set_band_description(2, "Red")
    img2 = os.path.join(data_dir, "img2.tif")
    with rasterio.open(img2, "w", count=1, dtype="float32", **profile) as dst:
        dst.write(rng.uniform(0.0, 1.0, (1, N_ROWS, N_COLS)).astype(np.float32))
    msk = os.path.join(data_dir, "msk.tif")
    msk_arr = np.zeros((1, N_ROWS, N_COLS), dtype=np.uint8)
    msk_arr[:, :, : N_COLS // 2] = 1
    with rasterio.open(msk, "w", count=1, dtype="uint8", **profile) as dst:
        dst.write(msk_arr)
    return {"img": img, "img2": img2, "msk": msk}


def _bar_heights_edges(ax):
    patches = ax.patches
    heights = np.array([p.get_height() for p in patches])
    lefts = np.array([p.get_x() for p in patches])
    rights = lefts + np.array([p.get_width() for p in patches])
    return heights, lefts, rights


@pytest.mark.parametrize(
    "kw",
    [
        {},
        {"val_range": (1000.0, 4000.0), "n_bins": 30},
        {"in_msk_img": "msk", "n_bins": 50},
    ],
)
def test_plot_img_histogram_matches_get_img_histogram(hist_imgs, kw):
    """The bars are the counts and bin edges of get_img_histogram."""
    kw = dict(kw)
    if "in_msk_img" in kw:
        kw["in_msk_img"] = hist_imgs[kw["in_msk_img"]]
    kw.setdefault("n_bins", 200)
    ref = imgstats.get_img_histogram(hist_imgs["img"], img_band=1, use_pbar=False, **kw)
    fig, ax = rplot.plot_img_histogram(hist_imgs["img"], 1, use_pbar=False, **kw)
    heights, lefts, rights = _bar_heights_edges(ax)
    np.testing.assert_array_equal(heights, ref["counts"])
    np.testing.assert_allclose(lefts, ref["bin_edges"][:-1])
    np.testing.assert_allclose(rights, ref["bin_edges"][1:])
    assert ax.get_xlabel() == "NIR"  # band name
    assert ax.get_ylabel() == "Pixel count"
    assert ax.get_legend() is None
    plt.close(fig)


def test_plot_img_histogram_thresholds_and_options(hist_imgs):
    """Thresholds are labelled vertical lines; log y; precomputed histogram."""
    ref = imgstats.get_img_histogram(hist_imgs["img"], img_band=2, n_bins=40, use_pbar=False)
    fig, ax = rplot.plot_img_histogram(
        "not_read.tif", 2, hist_data=ref, thresholds=[1000.0, 3000.0],
        threshold_labels=["Low", "High"], log_y=True, title="Red", x_label="Red (DN)",
    )
    heights, _, _ = _bar_heights_edges(ax)
    np.testing.assert_array_equal(heights, ref["counts"])
    v_lines = [ln for ln in ax.get_lines() if ln.get_label() in ("Low", "High")]
    assert [ln.get_xdata()[0] for ln in v_lines] == [1000.0, 3000.0]
    assert v_lines[0].get_color() != v_lines[1].get_color()
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["Low", "High"]
    assert ax.get_yscale() == "log"
    assert ax.get_title() == "Red" and ax.get_xlabel() == "Red (DN)"
    plt.close(fig)

    fig, ax = rplot.plot_img_histogram(hist_imgs["img"], 2, thresholds=2500.0, use_pbar=False)
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["Threshold = 2.5e+03"]
    assert ax.get_xlabel() == "Red"
    plt.close(fig)


def test_draw_histogram_checks_lengths():
    """Mismatched edges or labels raise an error."""
    fig, ax = plt.subplots()
    with pytest.raises(rsgispy.RSGISPyException):
        rplot.draw_histogram(ax, [1, 2], [0.0, 1.0])
    with pytest.raises(rsgispy.RSGISPyException):
        rplot.draw_histogram(ax, [1, 2], [0.0, 1.0, 2.0], thresholds=[0.5, 1.5],
                             threshold_labels=["a"])
    bars = rplot.draw_histogram(ax, [1, 2], [0.0, 1.0, 2.0], thresholds=1.0)
    assert len(bars) == 2
    plt.close(fig)


def test_plot_histogram_threshold_matches_numpy():
    """plot_histogram_threshold draws numpy.histogram of the data with the shared look."""
    vals = np.random.default_rng(3).normal(size=500)
    fig, ax = rplot.plot_histogram_threshold(vals, 0.5, n_bins=25)
    counts, edges = np.histogram(vals, bins=25)
    heights, lefts, _ = _bar_heights_edges(ax)
    np.testing.assert_array_equal(heights, counts)
    np.testing.assert_allclose(lefts, edges[:-1])
    assert ax.get_lines()[0].get_color() == rplot.THRESHOLD_COLOURS[0]
    plt.close(fig)


@pytest.mark.parametrize(
    "kw",
    [
        {"bins": 20},
        {"bins": 25, "x_range": (500.0, 4000.0), "y_range": (500.0, 4000.0)},
        {"bins": [10, 15], "in_msk_img": "msk"},
    ],
)
def test_plot_2d_img_histogram_matches_get_img_2d_histogram(hist_imgs, kw):
    """The image is the histogram (transposed, normalised), empty bins masked."""
    kw = dict(kw)
    if "in_msk_img" in kw:
        kw["in_msk_img"] = hist_imgs[kw["in_msk_img"]]
    hist, x_edges, y_edges = imgstats.get_img_2d_histogram(
        hist_imgs["img"], 1, 2, use_pbar=False, **kw
    )
    fig, ax = rplot.plot_2d_img_histogram(
        hist_imgs["img"], 1, 2, val_scale=100.0, use_pbar=False, **kw
    )
    img = ax.images[0]
    plotted = img.get_array()
    exp = hist.T / hist.sum()
    np.testing.assert_allclose(plotted.filled(0.0), exp)
    np.testing.assert_array_equal(np.ma.getmaskarray(plotted), exp == 0)
    np.testing.assert_allclose(
        img.get_extent(),
        [x_edges[0] / 100, x_edges[-1] / 100, y_edges[0] / 100, y_edges[-1] / 100],
    )
    assert isinstance(img.norm, mcolors.LogNorm)
    assert tuple(img.cmap.get_bad()[:3]) == (1.0, 1.0, 1.0)  # empty bins white
    assert ax.get_xlabel() == "NIR" and ax.get_ylabel() == "Red"
    assert img.colorbar is not None
    assert img.colorbar.ax.get_ylabel() == "Proportion of pixels"
    if "x_range" in kw:
        assert ax.get_aspect() == 1.0
    plt.close(fig)


def test_plot_2d_img_histogram_y_img_counts_and_points(hist_imgs):
    """A second image, counts on a linear scale, precomputed data and points."""
    hist_data = imgstats.get_img_2d_histogram(
        hist_imgs["img"], 1, 1, y_img=hist_imgs["img2"], bins=12, use_pbar=False
    )
    fig, ax = rplot.plot_2d_img_histogram(
        hist_imgs["img"], 1, 1, y_img=hist_imgs["img2"], bins=12, normalise=False,
        log_scale=False, show_colour_bar=False, use_pbar=False,
        points=[(1000.0, 0.2), (4000.0, 0.8)], point_labels=["A", "B"], title="FS",
    )
    img = ax.images[0]
    np.testing.assert_array_equal(img.get_array().filled(0.0), hist_data[0].T)
    assert not isinstance(img.norm, mcolors.LogNorm)
    assert img.colorbar is None
    assert ax.get_aspect() == "auto"  # different ranges
    assert ax.get_ylabel() == "Band 1"  # img2 has no band names
    offsets = np.vstack([coll.get_offsets() for coll in ax.collections])
    np.testing.assert_allclose(offsets, [[1000.0, 0.2], [4000.0, 0.8]])
    assert [t.get_text() for t in ax.texts] == ["A", "B"]
    assert ax.get_title() == "FS"
    plt.close(fig)

    # Precomputed histogram: the image is not read (labels given).
    fig, ax = rplot.plot_2d_img_histogram(
        "not_read.tif", 1, 1, hist_data=hist_data, x_label="x", y_label="y",
        points={"P": (2000.0, 0.5)},
    )
    np.testing.assert_allclose(
        ax.images[0].get_array().filled(0.0), hist_data[0].T / hist_data[0].sum()
    )
    assert [t.get_text() for t in ax.texts] == ["P"]
    plt.close(fig)


def test_plot_2d_img_histogram_bad_args(hist_imgs):
    """val_scale of 0 or a wrong number of point labels raise an error."""
    with pytest.raises(rsgispy.RSGISPyException):
        rplot.plot_2d_img_histogram(hist_imgs["img"], 1, 2, val_scale=0, use_pbar=False)
    with pytest.raises(rsgispy.RSGISPyException):
        rplot.plot_2d_img_histogram(
            hist_imgs["img"], 1, 2, points=[(1.0, 2.0)], point_labels=["a", "b"],
            use_pbar=False,
        )
    plt.close("all")
