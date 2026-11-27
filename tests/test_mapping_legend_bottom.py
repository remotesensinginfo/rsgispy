"""
A categorical legend below (or above) a map uses several columns, and a
smaller font if needed, so it is never taller than the space below the map.
"""

import matplotlib
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

import rsgispy.mapping.raster as map_raster
from rsgispy.imageutils.colourtable import set_thematic_classes

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  pylint: disable=wrong-import-position

NAMES = ["Artificial_Surfaces", "Bare_Rock_Sand", "Conifer_Forest", "Deciduous_Forest",
         "Grass_Long", "Grass_Short", "NonPhoto_Veg", "Scrub", "Water", "Bracken"]


@pytest.fixture(name="cls_img")
def fixture_cls_img(tmp_path):
    img = str(tmp_path / "cls.tif")
    arr = (np.arange(200 * 300).reshape(200, 300) // 6000 % 10 + 1).astype("uint8")
    with rasterio.open(img, "w", driver="GTiff", width=300, height=200, count=1,
                       dtype="uint8", crs="EPSG:27700",
                       transform=from_origin(0, 4000, 20, 20), nodata=0) as dst:
        dst.write(arr, 1)
    clrs = matplotlib.colormaps["tab10"].colors
    set_thematic_classes(img, {i + 1: (n, clrs[i]) for i, n in enumerate(NAMES)})
    return img


def _legend(fig):
    return [lgd for a in fig.axes if (lgd := a.get_legend()) is not None][0]


@pytest.mark.parametrize("position", ["bottom", "top"])
@pytest.mark.parametrize("figsize", [(4, 4), (6.7, 5)])
def test_legend_fits_below_map(cls_img, position, figsize):
    fig, ax = plt.subplots(figsize=figsize)
    map_raster.create_thematic_raster_map(ax, cls_img, show_legend=True,
                                          legend_overlay=False,
                                          legend_position=position)
    fig.canvas.draw()
    lgd = _legend(fig)
    renderer = fig.canvas.get_renderer()
    lgd_bb = lgd.get_window_extent(renderer)
    map_bb = ax.get_window_extent(renderer)
    assert lgd._ncols > 1  # pylint: disable=protected-access
    assert lgd_bb.height < 0.5 * map_bb.height
    fig_w = fig.get_size_inches()[0] * fig.dpi
    assert lgd_bb.x0 >= -1 and lgd_bb.x1 <= fig_w + 1  # within the figure
    plt.close(fig)


def test_explicit_ncols_is_kept(cls_img):
    fig, ax = plt.subplots(figsize=(4, 4))
    map_raster.create_thematic_raster_map(ax, cls_img, show_legend=True,
                                          legend_overlay=False,
                                          legend_position="bottom", legend_ncols=1)
    assert _legend(fig)._ncols == 1  # pylint: disable=protected-access
    plt.close(fig)
