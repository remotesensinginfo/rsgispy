"""
Tests for :func:`rsgispy.imageutils.colourtable.set_thematic_classes` and
:func:`rsgispy.imageutils.colourtable.get_thematic_classes`, and the
``create_thematic_raster_map`` legend built from the stored class names.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import rasterio  # noqa: E402
from rasterio.crs import CRS  # noqa: E402
from rasterio.enums import ColorInterp  # noqa: E402
import rasterio.shutil  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402

import rsgispy  # noqa: E402
import rsgispy.imageprocess  # noqa: E402
from rsgispy.imageutils import colourtable  # noqa: E402
import rsgispy.mapping.raster as map_raster  # noqa: E402

CLASSES = {
    1: ("Water", "#1f78b4"),
    2: ("Forest", (51, 160, 44)),
    3: ("Urban", (227, 26, 28, 128)),
}


def _cls_arr(height=40, width=40):
    arr = (np.add.outer(np.arange(height), np.arange(width)) // 10 % 4).astype(
        np.uint8
    )
    return arr


def _write_cls(path, arr=None, count=1, dtype="uint8", clr_tab=None):
    arr = _cls_arr() if arr is None else arr
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=arr.shape[1],
        height=arr.shape[0],
        count=count,
        dtype=dtype,
        crs=CRS.from_epsg(32630),
        transform=from_origin(400000, 5000000, 10, 10),
    ) as dst:
        for bidx in range(1, count + 1):
            dst.write(arr.astype(dtype), bidx)
        if clr_tab is not None:
            dst.write_colormap(1, clr_tab)
    return str(path)


def test_round_trip(tmp_path):
    img = _write_cls(tmp_path / "cls.tif")
    colourtable.set_thematic_classes(img, CLASSES)

    assert colourtable.is_img_thematic(img)
    classes = colourtable.get_thematic_classes(img)
    assert classes == {
        1: ("Water", "#1f78b4"),
        2: ("Forest", "#33a02c"),
        # GeoTIFF colour tables have no alpha channel.
        3: ("Urban", "#e31a1c"),
    }
    with rasterio.open(img) as src:
        tags = src.tags(1)
        assert json.loads(tags[colourtable.CLASS_NAMES_TAG]) == {
            "1": "Water",
            "2": "Forest",
            "3": "Urban",
        }
        # Thematic statistics and pyramids, as pop_thmt_img_stats.
        assert tags["STATISTICS_HISTOBINFUNCTION"] == "direct"
        assert tags["STATISTICS_EXCLUDEDVALUES"] == "0"
        assert src.nodata == 0
        assert src.colorinterp[0] == ColorInterp.palette
        cmap = src.colormap(1)
        assert cmap[0][:3] == (0, 0, 0)
        assert cmap[3][:3] == (227, 26, 28)
    # The stored classes can be written back unchanged.
    colourtable.set_thematic_classes(img, classes, calc_stats=False)
    assert colourtable.get_thematic_classes(img) == classes


def test_no_names_stored(tmp_path):
    img = _write_cls(tmp_path / "cls.tif", clr_tab={1: (255, 0, 0, 255)})
    assert not colourtable.get_thematic_classes(img)


def test_no_stats(tmp_path):
    img = _write_cls(tmp_path / "cls.tif")
    colourtable.set_thematic_classes(img, CLASSES, calc_stats=False)
    assert colourtable.is_img_thematic(img)
    with rasterio.open(img) as src:
        assert "STATISTICS_MEAN" not in src.tags(1)
        assert not src.overviews(1)
    assert colourtable.get_thematic_classes(img)[1] == ("Water", "#1f78b4")


def test_merge_with_existing_colour_table_and_names(tmp_path):
    existing = {1: (10, 10, 10, 255), 5: (200, 100, 50, 255), 7: (1, 2, 3, 255)}
    img = _write_cls(tmp_path / "cls.tif", clr_tab=existing)
    colourtable.set_thematic_classes(img, {1: ("Water", "#0000ff")})
    colourtable.set_thematic_classes(img, {5: ("Bare", (200, 100, 50))})

    with rasterio.open(img) as src:
        cmap = src.colormap(1)
    # Existing entries kept, the class entries replaced; not random colours.
    assert cmap[1] == (0, 0, 255, 255)
    assert cmap[5] == (200, 100, 50, 255)
    assert cmap[7] == (1, 2, 3, 255)
    # Names merged across calls.
    assert colourtable.get_thematic_classes(img) == {
        1: ("Water", "#0000ff"),
        5: ("Bare", "#c86432"),
    }
    # A later call renames a class.
    colourtable.set_thematic_classes(img, {1: ("Lake", "#0000ff")}, calc_stats=False)
    assert colourtable.get_thematic_classes(img)[1] == ("Lake", "#0000ff")


def test_validation(tmp_path):
    img = _write_cls(tmp_path / "cls.tif")
    with pytest.raises(rsgispy.RSGISPyException):
        colourtable.set_thematic_classes(img, {})
    with pytest.raises(rsgispy.RSGISPyException):
        colourtable.set_thematic_classes(img, {1: "Water"})
    with pytest.raises(rsgispy.RSGISPyException):
        colourtable.set_thematic_classes(img, {-1: ("Water", "#000000")})
    with pytest.raises(rsgispy.RSGISPyException):
        colourtable.set_thematic_classes(img, {1: ("Water", "#zzzzzz")})
    with pytest.raises(rsgispy.RSGISPyException):
        colourtable.set_thematic_classes(img, {300: ("Water", "#000000")})
    with pytest.raises(rsgispy.RSGISPyException):
        colourtable.set_thematic_classes(img, CLASSES, img_band=2)
    flt = _write_cls(tmp_path / "flt.tif", dtype="int16")
    with pytest.raises(rsgispy.RSGISPyException):
        colourtable.set_thematic_classes(flt, CLASSES)


def test_existing_cog(tmp_path):
    src_img = _write_cls(tmp_path / "cls.tif", arr=_cls_arr(600, 600))
    cog = str(tmp_path / "cls_cog.tif")
    rasterio.shutil.copy(src_img, cog, driver="COG")
    assert rsgispy.imageprocess.is_cog_layout(cog)

    colourtable.set_thematic_classes(cog, CLASSES)

    # Rebuilt as a COG, not edited in place, and no temporary files left.
    assert rsgispy.imageprocess.is_cog_layout(cog)
    assert sorted(os.listdir(tmp_path)) == ["cls.tif", "cls_cog.tif"]
    assert colourtable.is_img_thematic(cog)
    assert colourtable.get_thematic_classes(cog)[2] == ("Forest", "#33a02c")
    with rasterio.open(cog) as src:
        assert src.tags(ns="IMAGE_STRUCTURE").get("LAYOUT") == "COG"
        assert "STATISTICS_MAXIMUM" in src.tags(1)
        assert src.overviews(1)


def test_map_legend_from_stored_names(tmp_path):
    img = _write_cls(tmp_path / "cls.tif")
    colourtable.set_thematic_classes(img, CLASSES)

    fig, ax = plt.subplots()
    map_raster.create_thematic_raster_map(
        ax, img, show_legend=True, legend_overlay=True
    )
    lgd = ax.get_legend()
    assert lgd is not None
    assert [t.get_text() for t in lgd.get_texts()] == ["Water", "Forest", "Urban"]
    plt.close(fig)

    # An explicit classes_info still wins.
    fig, ax = plt.subplots()
    map_raster.create_thematic_raster_map(
        ax,
        img,
        show_legend=True,
        legend_overlay=True,
        classes_info={1: "A", 2: "B", 3: "C"},
    )
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["A", "B", "C"]
    plt.close(fig)


def test_colour_table_round_trip_from_get_colour_tab_info(tmp_path):
    """The output of get_colour_tab_info can be written with define_colour_table."""
    from rsgispy.imageutils.colourtable import define_colour_table, get_colour_tab_info

    imgs = []
    for name in ("a.tif", "b.tif"):
        path = str(tmp_path / name)
        with rasterio.open(
            path, "w", driver="GTiff", width=4, height=4, count=1, dtype="uint8",
            crs="EPSG:32630", transform=rasterio.transform.from_origin(0, 40, 10, 10),
        ) as dst:
            dst.write(np.arange(16, dtype=np.uint8).reshape(1, 4, 4) % 3)
        imgs.append(path)
    define_colour_table(imgs[0], {1: "#ff0000", 2: (0, 128, 255)})
    define_colour_table(imgs[1], get_colour_tab_info(imgs[0]))
    assert get_colour_tab_info(imgs[1]) == get_colour_tab_info(imgs[0])
