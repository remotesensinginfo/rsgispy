"""
Tests for rsgispy.mapping.layout.create_map_panels and MapPanel.
"""

import os

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.image import AxesImage  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import rasterio  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402
from shapely.geometry import box  # noqa: E402

import rsgispy  # noqa: E402
import rsgispy.imageutils.colourtable as colourtable  # noqa: E402
import rsgispy.mapping.elements as map_elements  # noqa: E402
import rsgispy.mapping.layout as map_layout  # noqa: E402

X_MIN, Y_MAX, RES = 500000.0, 5200000.0, 10.0
CLR_TAB = {
    0: (0, 0, 0, 0),
    1: (0, 114, 178, 255),
    2: (213, 94, 0, 255),
    3: (0, 158, 115, 255),
}


def _profile(count, dtype, nodata=None):
    return {
        "driver": "GTiff",
        "width": 20,
        "height": 20,
        "count": count,
        "dtype": dtype,
        "nodata": nodata,
        "crs": "EPSG:32630",
        "transform": from_origin(X_MIN, Y_MAX, RES, RES),
    }


@pytest.fixture(scope="module")
def panel_data(tmp_path_factory):
    """An RGB image, a continuous image, two class images and a vector layer."""
    data_dir = tmp_path_factory.mktemp("panels")
    rng = np.random.default_rng(0)

    rgb_img = os.path.join(data_dir, "rgb.tif")
    with rasterio.open(rgb_img, "w", **_profile(3, "float32")) as dst:
        dst.write(rng.uniform(1.0, 100.0, (3, 20, 20)).astype(np.float32))

    dist_img = os.path.join(data_dir, "dist.tif")
    with rasterio.open(dist_img, "w", **_profile(1, "float32", -9999)) as dst:
        dst.write(rng.normal(0.0, 2.0, (1, 20, 20)).astype(np.float32))

    # cls_a has classes 1 and 2; cls_b has classes 2 and 3.
    cls_a = os.path.join(data_dir, "cls_a.tif")
    cls_b = os.path.join(data_dir, "cls_b.tif")
    arr = np.ones((20, 20), dtype=np.uint8)
    arr[:, 10:] = 2
    for path, vals in ((cls_a, arr), (cls_b, arr + 1)):
        with rasterio.open(path, "w", **_profile(1, "uint8", 0)) as dst:
            dst.write(vals[np.newaxis])
            dst.write_colormap(1, CLR_TAB)

    vec = gpd.GeoDataFrame(
        {"id": [1]},
        geometry=[box(X_MIN + 20, Y_MAX - 180, X_MIN + 180, Y_MAX - 20)],
        crs="EPSG:32630",
    )
    return {
        "rgb": rgb_img,
        "dist": dist_img,
        "cls_a": cls_a,
        "cls_b": cls_b,
        "vec": vec,
    }


def test_map_panel_validation():
    with pytest.raises(rsgispy.RSGISPyException):
        map_layout.MapPanel("contour", "img.tif")
    with pytest.raises(rsgispy.RSGISPyException):
        map_layout.MapPanel("rgb")
    with pytest.raises(rsgispy.RSGISPyException):
        map_layout.MapPanel("vector")
    assert map_layout.MapPanel("RGB", "img.tif").panel_type == "rgb"


def test_create_map_panels_rejects_bad_input(panel_data):
    with pytest.raises(rsgispy.RSGISPyException):
        map_layout.create_map_panels([])
    with pytest.raises(rsgispy.RSGISPyException):
        map_layout.create_map_panels([{"panel_type": "rgb", "input": panel_data["rgb"]}])
    with pytest.raises(rsgispy.RSGISPyException):
        map_layout.create_map_panels(
            [{"panel_type": "rgb", "input_img": panel_data["rgb"]}],
            legend_position="top",
        )
    fig, axs = plt.subplots(1, 1)
    with pytest.raises(rsgispy.RSGISPyException):
        map_layout.create_map_panels(
            [
                {"panel_type": "rgb", "input_img": panel_data["rgb"]},
                {"panel_type": "thematic", "input_img": panel_data["cls_a"]},
            ],
            axs=axs,
        )
    plt.close("all")


def test_create_map_panels_layout_and_shared_legend(panel_data, tmp_path):
    out_file = os.path.join(tmp_path, "panels.png")
    names = {1: "Water", 2: "Forest", 3: "Grass"}
    fig, axs = map_layout.create_map_panels(
        [
            {"panel_type": "rgb", "input_img": panel_data["rgb"], "title": "Image"},
            {"panel_type": "thematic", "input_img": panel_data["cls_a"], "title": "A"},
            map_layout.MapPanel("thematic", panel_data["cls_b"], title="B"),
        ],
        ncols=2,
        classes_info=names,
        legend_title="Land cover",
        out_plot_file=out_file,
    )
    assert os.path.isfile(out_file)
    assert plt.fignum_exists(fig.number)
    assert axs.shape == (3,)
    # 2 x 2 grid: the fourth axis is hidden.
    assert len(fig.axes) == 4 and not fig.axes[3].axison
    assert [ax.get_title() for ax in axs] == ["Image", "A", "B"]
    # Every panel has the same extent.
    for ax in axs:
        assert ax.get_xlim() == pytest.approx((X_MIN, X_MIN + 200.0))
        assert ax.get_ylim() == pytest.approx((Y_MAX - 200.0, Y_MAX))
    # One legend with the classes of both thematic panels.
    assert len(fig.legends) == 1
    labels = [txt.get_text() for txt in fig.legends[0].get_texts()]
    assert labels == ["Water", "Forest", "Grass"]
    assert fig.legends[0].get_title().get_text() == "Land cover"
    # One scale bar and one north arrow, on the first panel.
    n_bars = [
        sum(isinstance(a, map_elements.MapScaleBar) for a in ax.artists) for ax in axs
    ]
    n_arrows = [
        sum(isinstance(a, map_elements.MapNorthArrow) for a in ax.artists) for ax in axs
    ]
    assert n_bars == [1, 0, 0]
    assert n_arrows == [1, 0, 0]
    plt.close("all")


def test_create_map_panels_stored_class_names(panel_data, tmp_path):
    cls_img = os.path.join(tmp_path, "cls_named.tif")
    with rasterio.open(panel_data["cls_a"]) as src:
        prof = src.profile
        arr = src.read()
    with rasterio.open(cls_img, "w", **prof) as dst:
        dst.write(arr)
    colourtable.set_thematic_classes(
        cls_img, {1: ("Water", "#0072B2"), 2: ("Forest", "#009E73")}
    )
    fig, _ = map_layout.create_map_panels(
        [{"panel_type": "thematic", "input_img": cls_img}],
        legend_position="bottom",
        show_scale_bar=False,
        show_north_arrow=False,
    )
    labels = [txt.get_text() for txt in fig.legends[0].get_texts()]
    assert labels == ["Water", "Forest"]
    plt.close("all")


def test_create_map_panels_cmap_base_and_vector(panel_data):
    fig, axs = map_layout.create_map_panels(
        [
            map_layout.MapPanel(
                "cmap",
                panel_data["dist"],
                cmap_name="RdBu_r",
                vmin=-3.0,
                vmax=3.0,
                show_colour_bar=True,
                cbar_label="Distance",
                title="Distance",
            ),
            map_layout.MapPanel(
                "thematic",
                panel_data["cls_a"],
                base_img=panel_data["rgb"],
                in_legend=False,
                title="Change",
            ),
            map_layout.MapPanel(
                "vector",
                vec=panel_data["vec"],
                base_img=panel_data["rgb"],
                map_kw={"vec_fill_clrs": "none", "vec_line_clrs": "red"},
            ),
        ],
        bbox=(X_MIN + 50.0, X_MIN + 150.0, Y_MAX - 150.0, Y_MAX - 50.0),
    )
    cmap_img = axs[0].images[-1]
    assert isinstance(cmap_img, AxesImage)
    assert cmap_img.colorbar is not None
    assert cmap_img.colorbar.ax.get_ylabel() == "Distance"
    assert cmap_img.norm.vmin == -3.0 and cmap_img.norm.vmax == 3.0
    # Base image below the layer.
    assert len(axs[1].images) == 2
    assert axs[1].images[0].get_zorder() < axs[1].images[1].get_zorder()
    assert len(axs[2].images) == 1 and axs[2].collections
    assert axs[2].get_xlim() == pytest.approx((X_MIN + 50.0, X_MIN + 150.0))
    # The only thematic panel is not in the legend, so there is none.
    assert not fig.legends
    plt.close("all")


def test_create_map_panels_vector_only_extent(panel_data):
    fig, axs = map_layout.create_map_panels(
        [{"panel_type": "vector", "vec": panel_data["vec"], "title": "Polygon"}]
    )
    assert axs[0].get_xlim() == pytest.approx((X_MIN + 20.0, X_MIN + 180.0))
    plt.close(fig)
